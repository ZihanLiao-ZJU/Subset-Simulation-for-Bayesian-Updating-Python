"""One orchestration loop; every algorithm owns its lifecycle policies.

Methods implement start/sample/advance/select_seeds/is_complete/finish.
The engine never identifies an algorithm, changes its probability law, or draws
random numbers. Frozen level records describe the distribution actually sampled.
"""
import copy
from types import MappingProxyType
import numpy as np

def frozen(value):
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.setflags(write=False)
        return value
    if isinstance(value, dict):
        return MappingProxyType({k: frozen(v) for k,v in value.items()})
    if isinstance(value, (list,tuple)):
        return tuple(frozen(v) for v in value)
    return copy.deepcopy(value)

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
    def snapshot(self, context, record):
        return frozen(dict(G=context["model"].G, x=record["x"], y=record["y"],
                           Ncal=record["calls"], Nsam=record["count"],
                           chain_shape=record["shape"]))
    def is_complete(self, context, record, phase): return False

class Engine:
    def __init__(self, method): self.method=method
    def run(self, owner, rng=None):
        method=self.method
        context=method.start(owner,rng)
        levels=[]
        for level in range(context["limit"]):
            record=method.sample(context,level)
            # Capture sampler adaptation as part of the sampled method state.
            levels.append(method.snapshot(context,record))
            method.advance(context,record,level)
            if method.is_complete(context,record,"before_seeds"): break
            method.select_seeds(context,record,level)
            if method.is_complete(context,record,"after_seeds"): break
        output=method.finish(context)
        output["levels"]=tuple(levels)
        output["engine"]="semcs_core.Engine"
        return output

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
import copy
import numpy as np
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


class ClosedSubsetBayesMethod(Method):
    """Opt-in closed support and batch scale fallback; original stop order."""
    def start(self,owner,rng):
        sampler=copy.copy(owner.Sampler);model=owner.IntBay
        if type(sampler).__name__!="aCS":raise ValueError("The minimum closed variant supports aCS only")
        if model.G.get("sus_runtime",{}).get("stored_levels",0)>0:
            raise ValueError("Use a fresh model for a new closed-boundary run")
        if owner.G:raise ValueError("Use scalar fixed NumSam; prescribed G schedules are unsupported by this variant")
        counts=np.asarray(owner.NumSam)
        if counts.ndim or isinstance(owner.NumSam,(bool,np.bool_)) or not isinstance(owner.NumSam,(int,np.integer)) or owner.NumSam<1:
            raise ValueError("This variant requires a positive integer scalar NumSam")
        if not 0<model.G["p"]<1:raise ValueError("ConPrb must lie strictly between 0 and 1")
        base=sampler.Sigma0Mode
        if base not in ("unit","seed_std"):raise ValueError("Unsupported baseline Sigma0Mode")
        runtime=dict(chain_plan=dict(Nc=int(owner.NumSam),Ns=1,actual_count=int(owner.NumSam)),
            last_seed_selection=None,degenerate_selected_seed_set=False,
            effective_sigma0_mode=base,fallback_applied=False,stored_levels=0)
        model.G=dict(model.G,baseline_sigma0_mode=base,sus_runtime=runtime)
        sampler.BayStc=model
        return dict(owner=owner,model=model,sampler=sampler,limit=owner.MaxIte,
            xseed=None,yseed=None,x=[],y=[],g=[copy.deepcopy(model.G)],Nsam=[],Nsze=[],Ncal=[])

    def sample(self,c,level):
        plan=c["model"].G["sus_runtime"]["chain_plan"]
        if level==0:x,y,calls,model=c["sampler"].SamGen(plan["Nc"],plan["Ns"])
        else:x,y,calls,model=c["sampler"].SamGen(c["xseed"],c["yseed"],plan["Ns"])
        runtime=copy.deepcopy(model.G["sus_runtime"]);runtime["stored_levels"]=level+1
        model.G=dict(model.G,sus_runtime=runtime);c["model"]=model
        r=dict(x=x,y=y,calls=calls,count=plan["actual_count"],shape=(plan["Nc"],plan["Ns"]))
        c["x"].append(x);c["y"].append(y);c["Ncal"].append(calls);c["Nsam"].append(r["count"]);c["Nsze"].append(r["shape"])
        return r

    def advance(self,c,r,level):
        _,model=c["model"].UpdObj(r["y"],c["model"].G["p"])
        c["model"]=model;c["sampler"].BayStc=model;c["g"].append(copy.deepcopy(model.G))

    def select_seeds(self,c,r,level):
        from .SuS_Bay_Nataf import apply_selected_scale
        xs,ys=c["model"].SltSed(r["x"],r["y"],int(c["model"].G["run"]["samples"]))
        c["xseed"],c["yseed"]=xs,ys
        model=apply_selected_scale(c["model"],c["sampler"],xs)
        c["model"]=model;c["g"][-1]=copy.deepcopy(model.G)
        if c["owner"].Verbose:DspRst(level+1,r["count"],np.asarray(r["shape"]),r["calls"])

    def is_complete(self,c,r,phase):
        if phase!="after_seeds":return False
        runtime=copy.deepcopy(c["model"].G["sus_runtime"])
        empty=runtime["chain_plan"]["Nc"]<=0
        cap=len(c["x"])+1==c["limit"]
        # Match the source's seed-before-stop and early empty/cap returns.
        converged=False if empty or cap else bool(c["model"].FlgCvg)
        runtime.update(model_converged=converged,empty_seed=empty,iteration_cap_reached=cap,
            termination_reason="empty_seed" if empty else "iteration_cap" if cap else "model_converged" if converged else None)
        c["model"].G=dict(c["model"].G,sus_runtime=runtime)
        c["sampler"].BayStc=c["model"];c["g"][-1]=copy.deepcopy(c["model"].G)
        return empty or cap or converged

    def finish(self,c):
        runtime=c["model"].G["sus_runtime"]
        return dict(x=c["x"],y=c["y"],g=c["g"],Nsam=np.asarray(c["Nsam"]),
            Nsze=np.asarray(c["Nsze"],dtype=int),Ncal=np.asarray(c["Ncal"]),Nite=len(c["x"]),
            flg_cvg=bool(runtime.get("model_converged",False)),intBay=c["model"],
            variant_id="closed_ties_unit_fallback_v1",termination=copy.deepcopy(runtime))


class Main(LegacyFacade):
    default_verbose=True
    def __init__(self, Sampler):
        closed=Sampler is not None and getattr(Sampler,"BayStc",None) is not None and Sampler.BayStc.G.get("method_profile")=="closed_ties_unit_fallback_v1"
        method=ClosedSubsetBayesMethod() if closed else SubsetBayesMethod()
        super().__init__(Sampler, method)

main = Main
