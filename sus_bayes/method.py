"""Method policies for native Bayesian algorithms; no orchestration loop."""
import copy
import numpy as np
from semcs_core import Method
from semcs_core.engine import prescribed_state
from ._utils import mround
from .support_functions.DspRst import DspRst
from .support_functions.SedSlt import SedSlt

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
        from .closed_fallback import apply_selected_scale
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
