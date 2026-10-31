# Adapted from Zihan Liao's public MATLAB MCMC samplers; repository LICENSE applies.
"""User's [0, pi] componentwise elliptical slice kernel in whitened coordinates."""
import numpy as np
try:
    from .MCMCBase import MCMCBase
except ImportError:
    from MCMCBase import MCMCBase

class ES(MCMCBase):
    Ntry = 20
    def SamGen(self,*args):
        b,u0,y0,n,save,L,nc,ns=self._prepare(args)
        d=u0.shape[0]; z0=np.linalg.solve(L,u0)
        u=np.zeros((d,nc,ns)); y=np.zeros((int(b.Nfun),nc,ns))
        v=self._randn(d,nc,ns); p=self._rand(1,nc,ns)
        for k in range(ns):
            if k == 0 and save:
                u[:,:,k]=u0; y[:,:,k]=y0; continue
            radius=np.sqrt(z0*z0+v[:,:,k]*v[:,:,k])
            ratio=np.divide(z0,radius,out=np.zeros_like(z0),where=radius>0)
            oldangle=np.arccos(np.clip(ratio,-1,1))
            ids=np.arange(nc); lo=np.zeros((d,nc)); hi=np.full((d,nc),np.pi)
            z=z0.copy(); u[:,:,k]=u0; y[:,:,k]=y0
            for _ in range(self.Ntry):
                angle=lo+(hi-lo)*self._rand(d,len(ids))
                proposal=np.cos(angle)*radius[:,ids]
                b,yp,nn=self._evaluate(b,L@proposal); n+=nn
                rej=self._reject(yp[0],y0[0,ids],p[0,ids,k]); accepted=~rej
                z[:,ids[accepted]]=proposal[:,accepted]
                u[:,ids[accepted],k]=(L@proposal)[:,accepted]; y[:,ids[accepted],k]=yp[:,accepted]
                if not np.any(rej): break
                ids=ids[rej]; angle=angle[:,rej]; lo=lo[:,rej]; hi=hi[:,rej]
                upper=angle>oldangle[:,ids]
                lo[~upper]=angle[~upper]; hi[upper]=angle[upper]
            z0=z; u0=u[:,:,k].copy(); y0=y[:,:,k].copy()
        self.BayStc=b
        return u,y,n,b
