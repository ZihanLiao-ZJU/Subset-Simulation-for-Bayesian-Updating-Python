"""Generic validation/merge of a model-owned sampling profile specification.

New method classes supply sampler_profile_spec(profile); no core method-family
registration or algorithm-specific default is required.
"""
import copy
from collections.abc import Mapping

def configure_method_profile(model, family, method_profile=None):
    if method_profile is not None and (not isinstance(method_profile,str) or not method_profile):
        raise ValueError("Method profile name must be a nonempty string")
    provider=getattr(model,"sampler_profile_spec",None)
    if not callable(provider):raise TypeError("Model must supply sampler_profile_spec(profile)")
    spec=provider() if method_profile is None else provider(method_profile)
    if not isinstance(spec,Mapping) or spec.get("family")!=family:
        raise ValueError("Model profile family does not match the requested family")
    profile_name=spec.get("name")
    if not isinstance(profile_name,str) or not profile_name:raise ValueError("Model profile must supply its canonical name")
    kernels=spec.get("kernels",{});evidence=spec.get("evidence",{});method_state=spec.get("state",{})
    if not isinstance(kernels,Mapping) or not isinstance(evidence,Mapping):
        raise TypeError("Model profile kernels/evidence must be mappings")
    if not isinstance(method_state,Mapping):raise TypeError("Model profile state must be a mapping")
    from .mcmc_samplers.MCMCBase import MCMCBase
    g=copy.deepcopy(model.G);settings=copy.deepcopy(g.get("sampler",{}))
    names=list(settings)
    for name in ("aCS","ES","CW_aMH","HMC",*kernels):
        if name not in names:names.append(name)
    for name in names:
        override=kernels.get(name,{})
        if not isinstance(name,str) or not isinstance(override,Mapping):
            raise TypeError("Kernel profile overrides must be named mappings")
        settings[name]={**MCMCBase._defaults,**settings.get(name,{}),**copy.deepcopy(override)}
    g.update(method_profile=profile_name,method_family=family,sampler=settings)
    g.update(copy.deepcopy(dict(method_state)))
    g["profile_evidence"]=copy.deepcopy(dict(evidence))
    model.G=g
    return model


from .main import state_property
import copy

import numpy as np

from ._utils import flat, mround, randi, randperm
from .log_operations import logmean, logminus, logplus
from .support_functions.WgtEvd import _mmin
from .support_functions.logGauss import logGauss


class SuS_Bay_Nataf:
    @property
    def Nsam(self):
        if self.G.get("method_profile")=="closed_ties_unit_fallback_v1":
            return self.G.get("run",{}).get("samples",1000)
        return self.G.get("Nsam",1000)

    @Nsam.setter
    def Nsam(self,value):
        if self.G.get("method_profile")=="closed_ties_unit_fallback_v1":
            run=dict(self.G.get("run",{}));run["samples"]=value;self.G=dict(self.G,run=run)
        else:self.G=dict(self.G,Nsam=value)
    """Subset simulation with Nataf transformation for Bayesian updating.

    Properties
        LKF  : likelihood function object (must implement EvlLKF and Pdis.U2X)
        Nfun : number of outputs per sample (default: 4)
        Ndim : dimension of the parameter space
        X    : samples in standard normal space [Ndim x Nsam]
        Y    : function values
        G    : parameters of the intermediate likelihood function
               [L_i, p_i, L_i1, p_i1, z_i1, z_tmp1, z_sum1]
    """


    def sampler_profile_spec(self, method_profile="author_latest"):
        """Method-owned author/article/legacy choices; core knows no profile IDs."""
        variant="closed_ties_unit_fallback_v1"
        if getattr(self,"G",{}).get("method_profile")==variant and method_profile!=variant:
            raise ValueError("Use a fresh model to change an initialized closed-boundary method profile")
        closed_variant=method_profile==variant
        if closed_variant and getattr(self,"G",{}).get("method_profile") not in (None,variant):
            raise ValueError("Construct a fresh closed-boundary model instead of switching an initialized method")
        if closed_variant:method_profile="author_latest"
        if method_profile=="paper_settings":method_profile="paper"
        if method_profile not in {"author_latest","paper","public_20261102"}:
            raise ValueError("Unknown method profile: "+str(method_profile))
        paper=method_profile=="paper"
        spec={'family': 'sus_bayes', 'default_kernel': 'aCS' if paper else 'aCS', 'kernels': {'aCS': {'Sigma0Mode': 'seed_std' if paper else 'unit', 'CarryLambda': False if paper else True}, 'ES': {'ESMode': 'coordinate' if paper else 'coordinate'}}, 'evidence': {'historical_table_generator_commit_authenticated': False, 'unknown_paper_budget_or_tolerance_not_filled': True, 'reference_factor_separate_from_kernel_base_scale': True, 'aCS_profile_support': 'CS-MH/article family plus author seed_std/reset' if paper else 'public20261102 legacy', 'ES_profile_support': 'retained coordinate implementation; not a global-ESS paper claim' if paper else 'retained coordinate implementation; not a global-ESS paper claim'}}
        spec["name"]=method_profile
        if method_profile=="author_latest":
            spec["kernels"]["aCS"].update(Sigma0Mode='seed_std',CarryLambda=False,
                Lambda=.6,AcpOpt=.44,NumIte=5,SedSav=False)
            spec["kernels"]["aCS"].update(AcceptArithmetic="exp",GroupMode="fixed",AcceptanceRateMode="attempts")
            spec["state"]={"lifecycle":{"InitialSampling":"sampler"}}
            pass
            spec["evidence"]["aCS_profile_support"]='latest available author srcv2 seed_std/reset'
            spec["evidence"]["ES_profile_support"]='available author coordinate ES with Ntry20; not literal article global ESS'
            spec["evidence"]["configuration_only_source_loop_not_reproduced"]=False
        pass
        if closed_variant:
            spec["name"]=variant
            if self.G.get("method_profile")==variant:
                spec["kernels"]=copy.deepcopy(self.G.get("sampler",spec["kernels"]))
            spec.setdefault("state",{}).update(boundary_rule="closed",
                seed_scale_policy="unit_if_all_selected_reference_vectors_identical_else_baseline")
            spec["evidence"].update(numerical_variant=True,paper_prescribed_tie_policy=False,
                source_exact=False,completion_or_efficiency_guaranteed=False)
        return spec

    def __init__(self, LKF, *, method_profile="author_latest"):
        self.LKF = LKF
        self.Nfun = 4
        self.Ndim = LKF.Ndim
        self.Ncal = None
        self.X = None
        self.Y = None
        self.G = {"parameters": np.array([-np.inf, 0, -np.inf, 0, -np.inf, -np.inf, -np.inf]), "lambda": 0.6, "L": np.eye(self.Ndim), "p": 0.1, "Nsam": 1000,
                  "threshold_tol": 1e-1, "evidence_tol": 1e-1}
        configure_method_profile(self,"sus_bayes",method_profile)
        if self.G.get("method_profile")=="closed_ties_unit_fallback_v1":
            state=dict(self.G);initial=state.pop("Nsam",1000);run=dict(state.get("run",{}))
            run.setdefault("samples",initial);state["run"]=run;self.G=state

    def EvlY(self, u):
        """Evaluate [log-likelihood; log-prior] at samples u [Ndim x Nsam]."""
        obj = copy.copy(self)
        x = obj.U2X(u)
        L = np.asarray(obj.LKF.EvlLKF(x), dtype=float).ravel()
        P = logGauss(u)
        obj.X = u
        obj.Y = np.vstack([L, P])
        obj.Ncal = u.shape[1]
        return obj

    def EvlLKF(self):
        """Intermediate likelihood function (log indicator)."""
        L = self.Y[0, :]
        L_i = self.G["parameters"][2]
        if np.isneginf(L_i) or (self.G.get("boundary_rule")!="closed" and np.isposinf(L_i)):
            return np.zeros_like(L)
        with np.errstate(divide="ignore"):
            return np.log(self.contains(L,L_i).astype(float))

    def contains(self, log_likelihood, threshold):
        """One method-owned support rule, including finite cached ties."""
        values=np.asarray(log_likelihood)
        if self.G.get("boundary_rule")=="closed":
            if np.isneginf(threshold):return np.ones(values.shape,dtype=bool)
            return values>=threshold
        return values>threshold

    def EvlPDF(self):
        """Intermediate prior (log prior)."""
        return self.Y[1, :]

    def UpdObj(self, y, p):
        """Update the parameters of the intermediate distributions.

        y : function values [4 x Nc x Ns];  p : target conditional probability
        Returns (g, obj).
        """
        obj = copy.copy(self)
        g = obj.G["parameters"]
        L_i, p_i, z_i, z_tmp, z_sum = g[2], g[3], g[4], g[5], g[6]
        L = flat(y)[2, :]
        Nsam = L.size
        L_sort = np.sort(L)[::-1]
        Nsed = int(mround(Nsam * p))
        if Nsed > 0:
            L_i1 = (L_sort[Nsed - 1] + L_sort[Nsed]) / 2
        else:
            L_i1 = L_sort[0]
        hit_count=int(np.sum(self.contains(L,L_i1)))
        with np.errstate(divide="ignore"):
            p_i1 = p_i + np.log(hit_count / Nsam)

        z_i1 = p_i + logmean(_mmin(logminus(L, L_i), logminus(L_i1, L_i)), None)
        z_tmp1 = p_i + logmean(logminus(L, L_i), None)
        z_sum1 = logplus(logplus(logminus(z_sum, z_tmp), z_i), z_tmp1)

        g = np.array([L_i, p_i, L_i1, p_i1, z_i1, z_tmp1, z_sum1], dtype=float)
        obj.G = {**obj.G, "parameters": g}
        if obj.G.get("boundary_rule")=="closed":
            runtime=copy.deepcopy(obj.G.get("sus_runtime",{}))
            runtime.update(effective_boundary_hit_fraction=hit_count/Nsam,
                boundary_hit_count=hit_count,boundary_pool_count=int(Nsam))
            obj.G=dict(obj.G,sus_runtime=runtime)
        return obj.G, obj

    def SltSed(self, u, y, Nsam):
        """Select seed samples for the next iteration."""
        u = flat(u)[:self.Ndim, :]
        y = flat(y)[:self.Nfun, :]
        L = y[2, :]
        L_i1 = self.G["parameters"][2]
        Ns = mround(Nsam / np.arange(1, Nsam + 1))
        Nsed = mround(Nsam / Ns)
        ind_sed = self.contains(L,L_i1)
        x_sed_can = u[:, ind_sed]
        y_sed_can = y[:, ind_sed]
        Nsed_can = int(np.sum(ind_sed))
        ok = np.nonzero(Nsed_can - Nsed >= 0)[0]
        Nsed = int(Nsed[ok[-1]]) if ok.size else 0
        if Nsed > 0 and Nsed_can >= Nsed:
            randind = randperm(Nsed_can, Nsed)
        elif Nsed > 0 and Nsed_can > 0:
            randind = randi(Nsed_can, Nsed)
        else:
            randind = np.array([], dtype=int)
        u_sed = x_sed_can[:, randind]
        y_sed = y_sed_can[:, randind]
        if y_sed.size:
            y_sed = self.UpdY(y_sed)
        if self.G.get("boundary_rule")=="closed":
            eligible=np.flatnonzero(ind_sed)
            selected=eligible[randind]
            steps=int(mround(Nsam/Nsed)) if Nsed else None
            runtime=copy.deepcopy(self.G.get("sus_runtime",{}))
            runtime.update(chain_plan=dict(Nc=Nsed,Ns=steps,actual_count=Nsed*steps if Nsed else 0),
                last_seed_selection=dict(eligible_indices=eligible.tolist(),selected_indices=selected.tolist(),
                    selected_positions=randind.tolist(),eligible_count=Nsed_can,selected_count=Nsed,
                    pool_count=int(L.size),threshold=float(L_i1),boundary_rule="closed",index_base=0,
                    closed_hit_fraction=Nsed_can/L.size))
            self.G=dict(self.G,sus_runtime=runtime)
        return u_sed, y_sed

    def UpdY(self, y):
        """Recompute the intermediate rows [Li; Pi] of the function values."""
        obj = copy.copy(self)
        y = flat(y)[2:, :]
        obj.Y = y
        Pi = obj.EvlPDF()
        Li = obj.EvlLKF()
        return np.vstack([Li, Pi, y])

    @property
    def FlgCvg(self):
        """Convergence flag based on the change in thresholds and evidence."""
        g = self.G["parameters"]
        L_i, L_i1, z_tmp, z_sum = g[0], g[2], g[5], g[6]
        with np.errstate(invalid="ignore", over="ignore"):
            a = np.real(np.exp(logminus(L_i1, L_i) - logplus(L_i1, L_i)))
            b = np.real(np.exp(z_tmp - z_sum))
        return bool(a < self.G["threshold_tol"] and b < self.G["evidence_tol"])

    def U2X(self, u):
        """Standard normal -> physical space via the Nataf transformation."""
        return self.LKF.Pdis.U2X(u)


VARIANT="closed_ties_unit_fallback_v1"
SEED_SCALE_POLICY="unit_if_all_selected_reference_vectors_identical_else_baseline"

def selected_reference_vectors(model,sampler,seeds):
    """Use exactly aCS's model-reference map and Gaussian whitening factor."""
    reference=sampler._to_reference(model,np.asarray(seeds,dtype=float))
    factor=sampler._factor(model,int(model.Ndim))
    return reference if np.array_equal(factor,np.eye(model.Ndim)) else np.linalg.solve(factor,reference)

def apply_selected_scale(model,sampler,seeds):
    base=model.G["baseline_sigma0_mode"]
    if base not in ("unit","seed_std"):raise ValueError("Unsupported baseline Sigma0Mode")
    reference=selected_reference_vectors(model,sampler,seeds)
    identical=bool(reference.shape[1] and np.all(reference==reference[:,[0]]))
    effective="unit" if base=="seed_std" and identical else base
    runtime=copy.deepcopy(model.G["sus_runtime"])
    runtime.update(degenerate_selected_seed_set=identical,effective_sigma0_mode=effective,
        reference_coordinate_space="aCS reference coordinates whitened by G.L",
        fallback_applied=bool(base=="seed_std" and identical))
    settings=copy.deepcopy(model.G["sampler"])
    settings["aCS"]["Sigma0Mode"]=effective
    model.G=dict(model.G,sampler=settings,sus_runtime=runtime)
    sampler.BayStc=model
    return model
