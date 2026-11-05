# Adapted from Zihan Liao's public MATLAB MCMC samplers; repository LICENSE applies.
"""Adaptive Gaussian conditional sampling; final lambda is stored in G.lambda."""
import numpy as np
from .MCMCBase import MCMCBase

class aCS(MCMCBase):
    def SamGen(self,*args):
        b,u0,y0,n,save,L,nc,ns=self._prepare(args)
        physical=self.CoordinateMode=="model"
        x0=u0.copy() if physical else None
        u0=self._to_reference(b,u0)
        d=u0.shape[0]; u=np.zeros((d,nc,ns)); y=np.zeros((int(b.Nfun),nc,ns))
        white_v=self._randn(d,nc,ns);v=np.einsum('ij,jkl->ikl',L,white_v)
        p=self._rand(1,nc,ns)
        if not isinstance(self.CarryLambda,(bool,np.bool_)):raise ValueError('CarryLambda must be bool.')
        lam=self._state(b,'lambda',self.Lambda) if self.CarryLambda else float(self.Lambda)
        if not np.isfinite(lam) or lam<=0: raise ValueError('G.lambda must be finite and positive.')
        if self.Sigma0Mode not in ('unit','seed_std'):raise ValueError('Sigma0Mode must be unit or seed_std.')
        scaled=self.Sigma0Mode=='seed_std'
        identity=np.array_equal(L,np.eye(d))
        if scaled:
            z0=u0 if identity else np.linalg.solve(L,u0)
            sigma0=np.maximum(np.std(z0,axis=1,ddof=1) if nc>1 else np.zeros(d),np.finfo(float).eps)
        width=max(int(np.ceil(nc/self.NumIte)),1)
        if self.GroupMode=="fixed":
            groups=[np.arange(min(i*width,nc-1),min((i+1)*width,nc)) for i in range(int(self.NumIte))]
        elif self.GroupMode=="available":
            groups=[np.arange(start,min(start+width,nc)) for start in range(0,nc,width)]
        else:raise ValueError('GroupMode must be available or fixed.')
        for group,ids in enumerate(groups,1):
            if scaled:
                sigma=np.minimum(lam*sigma0,1.0)[:,None];rho=np.sqrt(1-sigma*sigma)
            else:
                sigma=min(lam,1.0);rho=np.sqrt(1-sigma*sigma)
            count=0; attempts=0
            for k in range(ns):
                if k==0 and save:
                    u[:,ids,k]=x0[:,ids] if physical else u0[:,ids]; y[:,ids,k]=y0[:,ids]; continue
                if scaled and not identity:
                    proposal_z=rho*z0[:,ids]+sigma*white_v[:,ids,k]
                    proposal=L@proposal_z
                else:
                    proposal=rho*u0[:,ids]+sigma*v[:,ids,k]
                xp=self._from_reference(b,proposal)
                b,yp,nn=self._evaluate(b,xp); n+=nn
                rej=self._reject(yp[0],y0[0,ids],p[0,ids,k]); yes=~rej
                u0[:,ids[yes]]=proposal[:,yes]; y0[:,ids[yes]]=yp[:,yes]
                if physical:x0[:,ids[yes]]=xp[:,yes]
                if scaled and not identity:z0[:,ids[yes]]=proposal_z[:,yes]
                u[:,ids,k]=x0[:,ids] if physical else u0[:,ids]; y[:,ids,k]=y0[:,ids]
                count+=int(yes.sum()); attempts+=len(ids)
            if self.AcceptanceRateMode=="attempts":denominator=attempts
            elif self.AcceptanceRateMode=="chain_steps":denominator=len(ids)*ns
            else:raise ValueError('AcceptanceRateMode must be attempts or chain_steps.')
            if denominator and self.Adapt:lam=np.exp(np.log(lam)+(count/denominator-self.AcpOpt)/np.sqrt(group))
        b=self._setstate(b,'lambda',lam)
        b.G=dict(b.G,aCS_sigma0=sigma0.copy() if scaled else np.ones(d))
        self.BayStc=b
        return u,y,n,b
