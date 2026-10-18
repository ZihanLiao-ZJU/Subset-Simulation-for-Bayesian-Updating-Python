import numpy as np
from scipy.special import gammaln

from .._utils import rand


def _MargPDF(theta, idim):
    beta_a, alpha_a, c_a = 1, 1, 10
    beta_b, alpha_b, c_b = 1, 1, -10
    mu_c, sd_c = 10, 1
    mu_d, sd_d = -10, 1
    beta_i, alpha_i, c_i = 1, 1, 10
    mu_i, sd_i = 10, 1
    if idim == 1:  # mixture of log-gamma
        a = alpha_a * (theta - c_a) - np.exp(theta - c_a) / beta_a - gammaln(alpha_a) - alpha_a * np.log(beta_a)
        b = alpha_b * (theta - c_b) - np.exp(theta - c_b) / beta_b - gammaln(alpha_b) - alpha_b * np.log(beta_b)
        ab = np.maximum(a, b)
        return np.exp(np.log(0.5) + ab + np.log(np.exp(a - ab) + np.exp(b - ab)))
    if idim == 2:  # mixture of normal
        c = -0.5 * ((theta - mu_c) / sd_c) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sd_c)
        d = -0.5 * ((theta - mu_d) / sd_d) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sd_d)
        cd = np.maximum(c, d)
        return np.exp(np.log(0.5) + cd + np.log(np.exp(c - cd) + np.exp(d - cd)))
    if 3 <= idim <= 4:
        return np.exp(alpha_a * (theta - c_i) - np.exp(theta - c_i) / beta_a - gammaln(alpha_i) - alpha_a * np.log(beta_i))
    return np.exp(-0.5 * ((theta - mu_i) / sd_i) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sd_i))


def RS():
    """Rejection sampling of the six marginal reference densities (Gaussian + LogGamma)."""
    minmax = np.array([[-25, 15], [-15, 15], [0, 13], [0, 13], [6, 14], [6, 14]], dtype=float)
    sample = []
    for i in range(6):
        xtmp = []
        N = 0
        Nsam = 20000
        xmin, xmax = minmax[i]
        g = 1 / (xmax - xmin)
        p = rand(Nsam)
        while N <= 100000:
            x = rand(Nsam) * (xmax - xmin) + xmin
            yv = _MargPDF(x, i + 1) / (10 * g)
            acc = ~(yv < p)
            N += int(np.sum(acc))
            xtmp.append(x[acc])
        sample.append(np.concatenate(xtmp))
    return sample
