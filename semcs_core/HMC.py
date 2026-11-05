# Adapted from Zihan Liao's public MATLAB MCMC samplers; repository LICENSE applies.
"""Exact Gaussian-reference HMC proposal with target likelihood MH correction.

This is rejection-based Gaussian Hamiltonian sampling, not a gradient HMC
integrator for an arbitrary physical-space potential.
"""
import numpy as np
from .MCMCBase import MCMCBase

class HMC(MCMCBase):
    def SamGen(self,*args):
        b,u0,y0,n,save,L,nc,ns=self._prepare(args)
        d=u0.shape[0]; u=np.zeros((d,nc,ns)); y=np.zeros((int(b.Nfun),nc,ns))
        raw=self._randn(d,nc,ns); v=np.einsum('ij,jkl->ikl',L,raw); p=self._rand(1,nc,ns)
        time=self._state(b,'HMC_time',self.InitialTime)
        if not np.isfinite(time) or not 0<time<=np.pi/2: raise ValueError('G.HMC_time must lie in (0,pi/2].')
        width=max(int(np.ceil(nc/self.NumIte)),1)
        for start in range(0,nc,width):
            ids=np.arange(start,min(start+width,nc)); count=0; attempts=0
            for k in range(ns):
                if k==0 and save:
                    u[:,ids,k]=u0[:,ids]; y[:,ids,k]=y0[:,ids]; continue
                proposal=np.cos(time)*u0[:,ids]+np.sin(time)*v[:,ids,k]
                b,yp,nn=self._evaluate(b,proposal); n+=nn
                rej=self._reject(yp[0],y0[0,ids],p[0,ids,k]); yes=~rej
                u0[:,ids[yes]]=proposal[:,yes]; y0[:,ids[yes]]=yp[:,yes]
                u[:,ids,k]=u0[:,ids]; y[:,ids,k]=y0[:,ids]
                count+=int(yes.sum()); attempts+=len(ids)
            if attempts and self.Adapt:
                rate=count/attempts
                if rate<.3: time=np.arcsin(min(1,np.sin(time)*np.exp((rate-.3)/2)))
                elif rate>.5: time=np.arcsin(min(1,np.sin(time)*np.exp((rate-.5)/2)))
        b=self._setstate(b,'HMC_time',time); self.BayStc=b
        return u,y,n,b
