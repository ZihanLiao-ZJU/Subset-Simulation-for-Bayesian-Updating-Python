import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist


class LKF_nD_GaussianShells:
    """Likelihood function: two Gaussian shells in N-D space.

    Analytical log-evidence:
        Ndim = 2  -> logZ ~= -1.75
        Ndim = 5  -> logZ ~= -5.67
        Ndim = 10 -> logZ ~= -14.59
        Ndim = 20 -> logZ ~= -36.09
        Ndim = 30 -> logZ ~= -60.13

    Reference: Feroz & Hobson (2008), MNRAS, 384(2):449-463
    """

    _LOGZ = {2: -1.75, 5: -5.67, 10: -14.59, 20: -36.09, 30: -60.13}

    def __init__(self, Ndim):
        self.Ndim = Ndim
        self.Pdis = Nataf([makedist("Uniform", lower=-6, upper=6)] * Ndim, np.eye(Ndim))
        self.logz_ref = self._LOGZ.get(Ndim, np.nan)

    def EvlLKF(self, theta):
        theta = np.asarray(theta, dtype=float)
        ndim = self.Ndim
        c = np.zeros((ndim, 2))
        c[0, 0], c[0, 1] = 3.5, -3.5
        r = 2.0
        omega = 0.1
        X = -((np.sqrt(np.sum((theta - c[:, [0]]) ** 2, 0)) - r) ** 2) / (2 * omega ** 2)
        Y = -((np.sqrt(np.sum((theta - c[:, [1]]) ** 2, 0)) - r) ** 2) / (2 * omega ** 2)
        XY = np.maximum(X, Y)
        return np.log(1 / np.sqrt(2 * np.pi * omega ** 2)) + XY + np.log(np.exp(X - XY) + np.exp(Y - XY))
