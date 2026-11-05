# Adapted from Zihan Liao's public MATLAB MCMC samplers; repository LICENSE applies.
"""User's [0, pi] componentwise elliptical slice kernel in whitened coordinates."""
import numpy as np
from .MCMCBase import MCMCBase

class ES(MCMCBase):
    def SamGen(self,*args):
        if self.ESMode=='coordinate_2pi':return self._coordinate_2pi(*args)
        if self.ESMode=='global':return self._global(*args)
        if self.ESMode!='coordinate':raise ValueError('ESMode must be coordinate or global.')
        b,u0,y0,n,save,L,nc,ns=self._prepare(args)
        physical=self.CoordinateMode=="model"
        x0=u0.copy() if physical else None
        u0=self._to_reference(b,u0)
        d=u0.shape[0]; z0=np.linalg.solve(L,u0)
        u=np.zeros((d,nc,ns)); y=np.zeros((int(b.Nfun),nc,ns))
        v=self._randn(d,nc,ns); p=self._rand(1,nc,ns)
        for k in range(ns):
            if k == 0 and save:
                u[:,:,k]=x0 if physical else u0; y[:,:,k]=y0; continue
            radius=np.sqrt(z0*z0+v[:,:,k]*v[:,:,k])
            ratio=np.divide(z0,radius,out=np.zeros_like(z0),where=radius>0)
            oldangle=np.arccos(np.clip(ratio,-1,1))
            ids=np.arange(nc); lo=np.zeros((d,nc)); hi=np.full((d,nc),np.pi)
            z=z0.copy(); u[:,:,k]=x0 if physical else u0; y[:,:,k]=y0
            all_accepted=False
            for _ in range(self.Ntry):
                angle=lo+(hi-lo)*self._rand(d,len(ids))
                proposal=np.cos(angle)*radius[:,ids]
                xp=self._from_reference(b,L@proposal)
                b,yp,nn=self._evaluate(b,xp); n+=nn
                rej=self._reject(yp[0],y0[0,ids],p[0,ids,k]); accepted=~rej
                z[:,ids[accepted]]=proposal[:,accepted]
                u[:,ids[accepted],k]=xp[:,accepted]; y[:,ids[accepted],k]=yp[:,accepted]
                if not np.any(rej):
                    all_accepted=True
                    break
                ids=ids[rej]; angle=angle[:,rej]; lo=lo[:,rej]; hi=hi[:,rej]
                upper=angle>oldangle[:,ids]
                lo[~upper]=angle[~upper]; hi[upper]=angle[upper]
            if self.SeedUpdateMode not in ('each_step','all_accepted'):
                raise ValueError('SeedUpdateMode must be each_step or all_accepted.')
            if self.SeedUpdateMode=='each_step' or all_accepted:
                z0=z
                u0=L@z if physical else u[:,:,k].copy()
                if physical:x0=u[:,:,k].copy()
                y0=y[:,:,k].copy()
        self.BayStc=b
        return u,y,n,b

    def _coordinate_2pi(self,*args):
        """Independent coordinate angles, full2pi; external ES_2pi convention.

        The existing Gaussian/Nataf reference whitening is retained. For an
        identity factor this is the literal external2pi proposal; nonidentity
        factors use the declared whitened extension rather than direct mixing
        of correlated physical coordinates. Each step commits accepted chains,
        including when other chains exhaust the fixed20-attempt cap.
        """
        if self.SeedUpdateMode!='each_step':
            raise ValueError('coordinate_2pi requires SeedUpdateMode=each_step.')
        if isinstance(self.Ntry,(bool,np.bool_)) or self.Ntry!=20:
            raise ValueError('coordinate_2pi follows external ES_2pi with Ntry=20.')
        b,u0,y0,n,save,L,nc,ns=self._prepare(args)
        physical=self.CoordinateMode=="model"
        x0=u0.copy() if physical else None
        u0=self._to_reference(b,u0)
        d=u0.shape[0];z0=np.linalg.solve(L,u0)
        u=np.zeros((d,nc,ns));y=np.zeros((int(b.Nfun),nc,ns))
        v=self._randn(d,nc,ns);p=self._rand(1,nc,ns)
        for k in range(ns):
            if k==0 and save:
                u[:,:,k]=x0 if physical else u0;y[:,:,k]=y0;continue
            ids=np.arange(nc);lo=np.zeros((d,nc));hi=np.full((d,nc),2*np.pi)
            z=z0.copy();u[:,:,k]=x0 if physical else u0;y[:,:,k]=y0
            for attempt in range(20):
                angle=lo+(hi-lo)*self._rand(d,len(ids))
                proposal=np.cos(angle)*z0[:,ids]+np.sin(angle)*v[:,ids,k]
                xp=self._from_reference(b,L@proposal)
                b,yp,nn=self._evaluate(b,xp);n+=nn
                rejected=self._reject(yp[0],y0[0,ids],p[0,ids,k]);accepted=~rejected
                z[:,ids[accepted]]=proposal[:,accepted]
                u[:,ids[accepted],k]=xp[:,accepted];y[:,ids[accepted],k]=yp[:,accepted]
                if not np.any(rejected):break
                ids=ids[rejected];angle=angle[:,rejected]
                lo=lo[:,rejected];hi=hi[:,rejected]
                if attempt==0:
                    lo=angle-2*np.pi;hi=angle.copy()
                else:
                    positive=angle>0
                    lo[~positive]=angle[~positive];hi[positive]=angle[positive]
            z0=z;u0=L@z if physical else u[:,:,k].copy();y0=y[:,:,k].copy()
            if physical:x0=u[:,:,k].copy()
        self.BayStc=b
        return u,y,n,b

    def _global(self,*args):
        """Paper ESS: one angle per vector, full 2pi bracket, until accepted."""
        b,u0,y0,n,save,L,nc,ns=self._prepare(args)
        physical=self.CoordinateMode=="model"
        x0=u0.copy() if physical else None
        u0=self._to_reference(b,u0)
        d=u0.shape[0];z0=np.linalg.solve(L,u0)
        u=np.zeros((d,nc,ns));y=np.zeros((int(b.Nfun),nc,ns))
        v=self._randn(d,nc,ns);p=self._rand(1,nc,ns)
        for k in range(ns):
            if k==0 and save:
                u[:,:,k]=x0 if physical else u0;y[:,:,k]=y0;continue
            if not np.all(np.isfinite(y0[0])):
                raise ValueError('Global ESS requires finite current log likelihood.')
            ids=np.arange(nc)
            angle=2*np.pi*self._rand(1,nc).reshape(-1)
            lo=angle-2*np.pi;hi=angle.copy()
            z=z0.copy()
            while len(ids):
                proposal=z0[:,ids]*np.cos(angle)+v[:,ids,k]*np.sin(angle)
                xp=self._from_reference(b,L@proposal)
                b,yp,nn=self._evaluate(b,xp);n+=nn
                rej=self._reject(yp[0],y0[0,ids],p[0,ids,k]);yes=~rej
                z[:,ids[yes]]=proposal[:,yes]
                u[:,ids[yes],k]=xp[:,yes];y[:,ids[yes],k]=yp[:,yes]
                if not np.any(rej):break
                ids=ids[rej];angle=angle[rej];lo=lo[rej];hi=hi[rej]
                below=angle<0
                lo[below]=angle[below];hi[~below]=angle[~below]
                angle=lo+(hi-lo)*self._rand(1,len(ids)).reshape(-1)
            z0=z;u0=L@z if physical else u[:,:,k].copy();y0=y[:,:,k].copy()
            if physical:x0=u[:,:,k].copy()
        self.BayStc=b
        return u,y,n,b
