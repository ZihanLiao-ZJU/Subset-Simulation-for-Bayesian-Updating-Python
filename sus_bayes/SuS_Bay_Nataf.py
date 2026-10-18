import copy

import numpy as np

from ._utils import flat, mround, randi, randperm
from .log_operations import logmean, logminus, logplus
from .support_functions.WgtEvd import _mmin
from .support_functions.logGauss import logGauss


class SuS_Bay_Nataf:
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

    def __init__(self, LKF):
        self.LKF = LKF
        self.Nfun = 4
        self.Ndim = LKF.Ndim
        self.Ncal = None
        self.X = None
        self.Y = None
        self.G = np.array([-np.inf, 0, -np.inf, 0, -np.inf, -np.inf, -np.inf])

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
        L_i = self.G[2]
        if np.isinf(L_i):
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
        g = obj.G
        L_i, p_i, z_i, z_tmp, z_sum = g[2], g[3], g[4], g[5], g[6]
        L = flat(y)[2, :]
        Nsam = L.size
        L_sort = np.sort(L)[::-1]
        Nsed = int(mround(Nsam * p))
        if Nsed > 0:
            L_i1 = (L_sort[Nsed - 1] + L_sort[Nsed]) / 2
        else:
            L_i1 = L_sort[0]
        with np.errstate(divide="ignore"):
            p_i1 = p_i + np.log(np.sum(L > L_i1) / Nsam)

        z_i1 = p_i + logmean(_mmin(logminus(L, L_i), logminus(L_i1, L_i)), None)
        z_tmp1 = p_i + logmean(logminus(L, L_i), None)
        z_sum1 = logplus(logplus(logminus(z_sum, z_tmp), z_i), z_tmp1)

        g = np.array([L_i, p_i, L_i1, p_i1, z_i1, z_tmp1, z_sum1], dtype=float)
        obj.G = g
        return g, obj

    def SltSed(self, u, y, Nsam):
        """Select seed samples for the next iteration."""
        u = flat(u)[:self.Ndim, :]
        y = flat(y)[:self.Nfun, :]
        L = y[2, :]
        L_i1 = self.G[2]
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
        g = self.G
        L_i, L_i1, z_tmp, z_sum = g[0], g[2], g[5], g[6]
        with np.errstate(invalid="ignore", over="ignore"):
            a = np.real(np.exp(logminus(L_i1, L_i) - logplus(L_i1, L_i)))
            b = np.real(np.exp(z_tmp - z_sum))
        return bool(a < 1e-1 and b < 1e-1)

    def U2X(self, u):
        """Standard normal -> physical space via the Nataf transformation."""
        return self.LKF.Pdis.U2X(u)
