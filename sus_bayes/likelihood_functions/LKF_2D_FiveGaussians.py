"""Paper Table1 Example B: five unnormalized Gaussian peaks on [-1,1]^2."""
import numpy as np
from scipy.special import logsumexp
from ..Nataf import Nataf
from ..distributions import makedist

class LKF_2D_FiveGaussians:
    Ndim=2
    logz_ref=-5.512266674785554
    evidence_ref=.004036946559862885
    means=np.array([[-.4,-.4],[-.35,.2],[-.2,.15],[.1,-.15],[.45,.1]])
    sigmas=np.array([.01,.01,.03,.02,.05])
    amplitudes=np.array([.5,1.,.8,.5,.6])
    def __init__(self):
        self.Pdis=Nataf([makedist("Uniform",lower=-1,upper=1)]*2,np.eye(2))
    def EvlLKF(self,theta):
        theta=np.asarray(theta,dtype=float)
        if theta.ndim!=2 or theta.shape[0]!=2:raise ValueError("Five-Gaussian target expects (2,N)")
        delta=theta.T[None,:,:]-self.means[:,None,:]
        terms=np.log(self.amplitudes)[:,None]-np.sum(delta*delta,axis=2)/(2*self.sigmas[:,None]**2)
        return logsumexp(terms,axis=0)
