"""One orchestration loop; every algorithm owns its lifecycle policies.

Methods implement start/sample/advance/select_seeds/is_complete/finish.
The engine never identifies an algorithm, changes its probability law, or draws
random numbers.
"""
import copy
import numpy as np


def state_property(name, default=None):
    """Compatibility attribute backed solely by the method's G mapping."""
    def get(self): return self.G.get(name, default)
    def set_(self,value):
        state=dict(getattr(self,"G",{})); state[name]=value; self.G=state
    return property(get,set_)

def prescribed_state(current, specified, *, initial=False):
    """Apply method parameters while retaining run and kernel configuration.

    Explicit schedules historically replaced the method's parameter payload,
    while options lived on the run/sampler objects. Those options now live in
    G and must survive every prescribed transition. Live adaptation wins at
    later levels, as in the source loops; the first state uses its prescribed
    adaptation keys or the configured kernel defaults if they are absent.
    """
    requested=copy.deepcopy(specified)
    state=copy.deepcopy(current)
    state.update(requested)
    for key in ("run","p","Nsam","sampler","sampler_name"):
        if key in current: state[key]=copy.deepcopy(current[key])
    for key in ("lambda","HMC_time"):
        if initial:
            if key not in requested: state.pop(key,None)
        elif key in current:
            state[key]=copy.deepcopy(current[key])
    return state

class Method:
    """Policy protocol: algorithm-specific methods live with their repository."""
    def is_complete(self, context, record, phase): return False

class Engine:
    def __init__(self, method): self.method=method
    def run(self, owner, rng=None):
        method=self.method
        context=method.start(owner,rng)
        for level in range(context["limit"]):
            record=method.sample(context,level)
            method.advance(context,record,level)
            if method.is_complete(context,record,"before_seeds"): break
            method.select_seeds(context,record,level)
            if method.is_complete(context,record,"after_seeds"): break
        return method.finish(context)

class LegacyFacade:
    """Old constructors/properties without a private sequential run loop."""
    default_samples=1000
    default_verbose=False
    default_max_iterations=10000
    def __init__(self, Sampler, method):
        if Sampler is None or Sampler.BayStc is None:
            raise ValueError("Sampler with BayStc is required")
        self.Sampler=Sampler
        self.IntBay=Sampler.BayStc
        self._method=method
        self.IntBay.G=dict(self.IntBay.G)
        self.IntBay.G.setdefault("run",{})
        self.NumSam=getattr(self.IntBay,"Nsam",self.default_samples)
        self.ConPrb=self.IntBay.G.get("p",0.1)
        self.Verbose=self.default_verbose
    def _get(self,key,default): return self.IntBay.G.get("run",{}).get(key,default)
    def _set(self,key,value):
        g=dict(self.IntBay.G); run=dict(g.get("run",{})); run[key]=value
        g["run"]=run; self.IntBay.G=g
    @property
    def NumSam(self): return self._get("samples",self.default_samples)
    @NumSam.setter
    def NumSam(self,value): self._set("samples",value)
    @property
    def ConPrb(self): return self.IntBay.G.get("p",0.1)
    @ConPrb.setter
    def ConPrb(self,value): self.IntBay.G=dict(self.IntBay.G,p=value)
    @property
    def MaxIte(self): return self._get("max_iterations",self.default_max_iterations)
    @MaxIte.setter
    def MaxIte(self,value): self._set("max_iterations",value)
    @property
    def Verbose(self): return self._get("verbose",self.default_verbose)
    @Verbose.setter
    def Verbose(self,value): self._set("verbose",bool(value))
    @property
    def G(self): return self._get("schedule",[])
    @G.setter
    def G(self,value): self._set("schedule",value)
    def RunIte(self): return Engine(self._method).run(self)



"""Method policies for native Bayesian algorithms; no orchestration loop."""
from ._utils import mround
from .support_functions.DspRst import DspRst

class SubsetBayesMethod(Method):
    def start(self,owner,rng):
        sampler=copy.copy(owner.Sampler); model=owner.IntBay
        counts=np.atleast_1d(owner.NumSam)
        limit=owner.MaxIte
        # Preserve the historical max-iteration and schedule conventions.
        ns=np.zeros(limit+1,dtype=int)
        if counts.size>1: ns[:counts.size]=counts
        else: ns[:]=counts[0]
        schedule=[copy.deepcopy(v) if isinstance(v,dict) else
                  dict(parameters=np.asarray(v).copy(), **{"lambda":0.6})
                  for v in owner.G]
        if len(schedule):
            model=copy.copy(model)
            model.G=prescribed_state(model.G,schedule[0],initial=True)
            sampler.BayStc=model
        return dict(owner=owner,model=model,sampler=sampler,limit=limit,
                    targets=ns,schedule=schedule,nc=int(ns[0]),steps=1,
                    xseed=None,yseed=None,x=[],y=[],g=[copy.deepcopy(model.G)],
                    Nsam=[],Nsze=[],Ncal=[],complete=False)
    def sample(self,c,level):
        if level==0:
            x,y,calls,model=c["sampler"].SamGen(c["nc"],c["steps"])
        else:
            x,y,calls,model=c["sampler"].SamGen(c["xseed"],c["yseed"],c["steps"])
        c["model"]=model
        record=dict(x=x,y=y,calls=calls,count=c["nc"]*c["steps"],shape=(c["nc"],c["steps"]))
        c["x"].append(x);c["y"].append(y);c["Ncal"].append(calls)
        c["Nsam"].append(record["count"]);c["Nsze"].append(record["shape"])
        return record
    def advance(self,c,r,level):
        model=c["model"]
        if level+1<len(c["schedule"]):
            model=copy.copy(model)
            model.G=prescribed_state(model.G,c["schedule"][level+1])
        else:
            _,model=model.UpdObj(r["y"],c["owner"].ConPrb)
        c["model"]=model;c["sampler"].BayStc=model
        c["g"].append(copy.deepcopy(model.G))
    def select_seeds(self,c,r,level):
        target=int(c["targets"][level+1])
        xs,ys=c["model"].SltSed(r["x"],r["y"],target)
        c["xseed"],c["yseed"]=xs,ys
        c["nc"]=xs.shape[1]
        c["steps"]=mround(target/c["nc"]) if c["nc"] else np.inf
        if c["owner"].Verbose: DspRst(level+1,r["count"],np.asarray(r["shape"]),r["calls"])
    def is_complete(self,c,r,phase):
        if phase!="after_seeds": return False
        if c["nc"]<=0 or len(c["x"])+1==c["limit"]: return True
        c["complete"]=bool(c["model"].FlgCvg)
        return c["complete"]
    def finish(self,c):
        return dict(x=c["x"],y=c["y"],g=c["g"],Nsam=np.asarray(c["Nsam"]),
                    Nsze=np.asarray(c["Nsze"],dtype=int),Ncal=np.asarray(c["Ncal"]),
                    Nite=len(c["x"]),flg_cvg=c["complete"],intBay=c["model"])




class Main(LegacyFacade):
    default_verbose=True
    def __init__(self, Sampler):
        super().__init__(Sampler, SubsetBayesMethod())

main = Main
