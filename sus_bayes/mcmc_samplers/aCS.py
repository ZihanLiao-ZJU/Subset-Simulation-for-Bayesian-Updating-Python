# Adapted from Zihan Liao's public MATLAB MCMC samplers; repository LICENSE applies.
"""Adaptive Gaussian conditional sampling; final lambda is stored in G.lambda."""
import numpy as np
try:
    from .MCMCBase import MCMCBase
except ImportError:
    from MCMCBase import MCMCBase

class aCS(MCMCBase):
    Adapt = True
    Lambda=0.6; AcpOpt=0.44; NumIte=5
    def SamGen(self,*args):
        b,u0,y0,n,save,L,nc,ns=self._prepare(args)
        d=u0.shape[0]; u=np.zeros((d,nc,ns)); y=np.zeros((int(b.Nfun),nc,ns))
        v=np.einsum('ij,jkl->ikl',L,self._randn(d,nc,ns))
        p=self._rand(1,nc,ns); lam=self._state(b,'lambda',self.Lambda)
        if not np.isfinite(lam) or lam<=0: raise ValueError('G.lambda must be finite and positive.')
        width=max(int(np.ceil(nc/self.NumIte)),1)
        for group,start in enumerate(range(0,nc,width),1):
            ids=np.arange(start,min(start+width,nc)); sigma=min(lam,1.0); rho=np.sqrt(1-sigma*sigma)
            count=0; attempts=0
            for k in range(ns):
                if k==0 and save:
                    u[:,ids,k]=u0[:,ids]; y[:,ids,k]=y0[:,ids]; continue
                proposal=rho*u0[:,ids]+sigma*v[:,ids,k]
                b,yp,nn=self._evaluate(b,proposal); n+=nn
                rej=self._reject(yp[0],y0[0,ids],p[0,ids,k]); yes=~rej
                u0[:,ids[yes]]=proposal[:,yes]; y0[:,ids[yes]]=yp[:,yes]
                u[:,ids,k]=u0[:,ids]; y[:,ids,k]=y0[:,ids]
                count+=int(yes.sum()); attempts+=len(ids)
            if attempts and self.Adapt: lam=np.exp(np.log(lam)+(count/attempts-self.AcpOpt)/np.sqrt(group))
        b=self._setstate(b,'lambda',lam); self.BayStc=b
        return u,y,n,b
