"""Subset simulation method and native Bayesian lifecycle."""
import copy





import numpy as np

from ._utils import flat, mround, randi, randperm
from .operations_on_logarithmic_values import logmean, logminus, logplus
from .support_functions.WgtEvd import _mmin
from .support_functions.logGauss import logGauss


from .SeMCS import prescribed_state
from ._utils import mround
from .support_functions.DspRst import DspRst

class SuS_Bay_Nataf:
    @property
    def Nsam(self):
        return self.G.get("Nsam",1000)

    @Nsam.setter
    def Nsam(self,value):
        self.G=dict(self.G,Nsam=value)
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
        """Sampling options for the supported source and paper settings."""
        if method_profile not in {"author_latest","paper"}:
            raise ValueError("Unknown method profile: "+str(method_profile))
        spec={"family":"sus_bayes", "name":method_profile, "default_kernel":"aCS",
              "kernels":{"aCS":{"Sigma0Mode":"seed_std", "CarryLambda":False},
                         "ES":{"ESMode":"coordinate"}}}
        if method_profile=="author_latest":
            spec["kernels"]["aCS"].update(Sigma0Mode="seed_std",CarryLambda=False,
                Lambda=.6,AcpOpt=.44,NumIte=5,SedSav=False,
                AcceptArithmetic="exp",GroupMode="fixed",AcceptanceRateMode="attempts")
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
        self.MethodProfile=method_profile
        self.sampler_profile_spec(method_profile)

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
        if np.isneginf(L_i) or np.isposinf(L_i):
            return np.zeros_like(L)
        with np.errstate(divide="ignore"):
            return np.log((L > L_i).astype(float))


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
        hit_count=int(np.sum(L > L_i1))
        with np.errstate(divide="ignore"):
            p_i1 = p_i + np.log(hit_count / Nsam)

        z_i1 = p_i + logmean(_mmin(logminus(L, L_i), logminus(L_i1, L_i)), None)
        z_tmp1 = p_i + logmean(logminus(L, L_i), None)
        z_sum1 = logplus(logplus(logminus(z_sum, z_tmp), z_i), z_tmp1)

        g = np.array([L_i, p_i, L_i1, p_i1, z_i1, z_tmp1, z_sum1], dtype=float)
        obj.G = {**obj.G, "parameters": g}
        return obj.G, obj

    def SltSed(self, u, y, Nsam):
        """Select seed samples for the next iteration."""
        u = flat(u)[:self.Ndim, :]
        y = flat(y)[:self.Nfun, :]
        L = y[2, :]
        L_i1 = self.G["parameters"][2]
        Ns = mround(Nsam / np.arange(1, Nsam + 1))
        Nsed = mround(Nsam / Ns)
        ind_sed = L > L_i1
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

    def run_defaults(self):
        return dict(Verbose=True)

    def start_run(self,owner,sampler):
        owner.IntBay.G=dict(owner.IntBay.G,p=owner.ConPrb)
        sampler=copy.copy(sampler); model=owner.IntBay
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

    def sample_level(self,c,level):
        if level==0:
            x,y,calls,model=c["sampler"].SamGen(c["nc"],c["steps"])
        else:
            x,y,calls,model=c["sampler"].SamGen(c["xseed"],c["yseed"],c["steps"])
        c["model"]=model
        record=dict(x=x,y=y,calls=calls,count=c["nc"]*c["steps"],shape=(c["nc"],c["steps"]))
        c["x"].append(x);c["y"].append(y);c["Ncal"].append(calls)
        c["Nsam"].append(record["count"]);c["Nsze"].append(record["shape"])
        return record

    def advance_level(self,c,r,level):
        model=c["model"]
        if level+1<len(c["schedule"]):
            model=copy.copy(model)
            model.G=prescribed_state(model.G,c["schedule"][level+1])
        else:
            _,model=model.UpdObj(r["y"],c["owner"].ConPrb)
        c["model"]=model;c["sampler"].BayStc=model
        c["g"].append(copy.deepcopy(model.G))

    def select_run_seeds(self,c,r,level):
        target=int(c["targets"][level+1])
        xs,ys=c["model"].SltSed(r["x"],r["y"],target)
        c["xseed"],c["yseed"]=xs,ys
        c["nc"]=xs.shape[1]
        c["steps"]=mround(target/c["nc"]) if c["nc"] else np.inf
        if c["owner"].Verbose: DspRst(level+1,r["count"],np.asarray(r["shape"]),r["calls"])

    def run_complete(self,c,r,phase):
        if phase!="after_seeds": return False
        if c["nc"]<=0 or len(c["x"])+1==c["limit"]: return True
        c["complete"]=bool(c["model"].FlgCvg)
        return c["complete"]

    def finish_run(self,c):
        return dict(x=c["x"],y=c["y"],g=c["g"],Nsam=np.asarray(c["Nsam"]),
                    Nsze=np.asarray(c["Nsze"],dtype=int),Ncal=np.asarray(c["Ncal"]),
                    Nite=len(c["x"]),flg_cvg=c["complete"],intBay=c["model"])

    def sampler_options(self, kernel):
        return copy.deepcopy(self.sampler_profile_spec(self.MethodProfile)["kernels"].get(kernel,{}))


