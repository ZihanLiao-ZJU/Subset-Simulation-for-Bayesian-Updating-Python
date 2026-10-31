# Adapted from Zihan Liao's public MATLAB MCMC samplers; repository LICENSE applies.
"""Componentwise adaptive-width Metropolis-Hastings in Gaussian coordinates."""
import numpy as np
try:
    from .MCMCBase import MCMCBase
except ImportError:
    from MCMCBase import MCMCBase

class CW_aMH(MCMCBase):
    def SamGen(self,*args):
        b,u0,y0,n,save,L,nc,ns=self._prepare(args)
        d=u0.shape[0]; z0=np.linalg.solve(L,u0)
        u=np.zeros((d,nc,ns)); y=np.zeros((int(b.Nfun),nc,ns))
        du=2*self._rand(d,nc,ns)-1; p=self._rand(d,nc,ns)
        sigma=np.maximum(np.std(z0,axis=1,ddof=1) if nc>1 else np.zeros(d),np.finfo(float).eps)
        for k in range(ns):
            if k==0 and save:
                u[:,:,k]=u0; y[:,:,k]=y0; continue
            z=z0+sigma[:,None]*du[:,:,k]
            rej=self._reject(-.5*z*z,-.5*z0*z0,p[:,:,k]); z[rej]=z0[rej]
            ids=np.flatnonzero(np.any(z!=z0,axis=0))
            if len(ids):
                proposal=L@z[:,ids]; b,yp,nn=self._evaluate(b,proposal); n+=nn
                rej=self._reject(yp[0],y0[0,ids],self._rand(1,len(ids)).reshape(-1))
                yes=~rej; z0[:,ids[yes]]=z[:,ids[yes]]; u0[:,ids[yes]]=proposal[:,yes]; y0[:,ids[yes]]=yp[:,yes]
            u[:,:,k]=u0; y[:,:,k]=y0
        self.BayStc=b
        return u,y,n,b
