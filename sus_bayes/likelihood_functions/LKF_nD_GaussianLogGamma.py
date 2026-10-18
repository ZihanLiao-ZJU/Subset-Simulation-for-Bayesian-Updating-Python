import numpy as np
from scipy.special import gammaln

from ..Nataf import Nataf
from ..distributions import makedist


class LKF_nD_GaussianLogGamma:
    """Likelihood function: Gaussian + LogGamma mixture in N-D.

    Features both skewed and symmetric components.

    Analytical log-evidence:
        Ndim = 2  -> logZ ~= -8.19
        Ndim = 5  -> logZ ~= -20.47
        Ndim = 10 -> logZ ~= -40.94
        Ndim = 20 -> logZ ~= -81.89
        Ndim = 30 -> logZ ~= -122.82

    Reference: Beaujean & Caldwell (2013), arXiv:1304.7808
    """

    def __init__(self, Ndim):
        self.Ndim = Ndim
        self.Pdis = Nataf([makedist("Uniform", lower=-30, upper=30)] * Ndim, np.eye(Ndim))
        self.logz_ref = -Ndim * np.log(60)

    def EvlLKF(self, theta):
        theta = np.asarray(theta, dtype=float)
        ndim = theta.shape[0]
        beta, alpha = 1.0, 1.0
        c_a, c_b = 10.0, -10.0
        mu_c, mu_d = 10.0, -10.0
        sd = 1.0
        t1, t2 = theta[0], theta[1]
        with np.errstate(over="ignore", invalid="ignore"):
            Log_g_a = alpha * (t1 - c_a) - np.exp(t1 - c_a) / beta - gammaln(alpha) - alpha * np.log(beta)
            Log_g_b = alpha * (t1 - c_b) - np.exp(t1 - c_b) / beta - gammaln(alpha) - alpha * np.log(beta)
            Log_g_ab = np.maximum(Log_g_a, Log_g_b)
            Log_n_c = -0.5 * ((t2 - mu_c) / sd) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sd)
            Log_n_d = -0.5 * ((t2 - mu_d) / sd) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sd)
            Log_n_cd = np.maximum(Log_n_c, Log_n_d)

            logL = (np.log(0.5) + Log_g_ab + np.log(np.exp(Log_g_a - Log_g_ab) + np.exp(Log_g_b - Log_g_ab))
                    + np.log(0.5) + Log_n_cd + np.log(np.exp(Log_n_c - Log_n_cd) + np.exp(Log_n_d - Log_n_cd)))

            # Gaussian or LogGamma for the other dimensions (i is 1-based as in MATLAB)
            for i in range(3, ndim + 1):
                ti = theta[i - 1]
                if i <= (ndim + 2) / 2:
                    logL = logL + (alpha * (ti - c_a) - np.exp(ti - c_a) / beta - gammaln(alpha) - alpha * np.log(beta))
                else:
                    logL = logL + (-0.5 * ((ti - mu_c) / sd) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sd))
        return logL
