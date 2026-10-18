"""
Probability distribution objects in the style of MATLAB's ``makedist``.

The MATLAB code builds its priors and input models from Statistics and
Machine Learning Toolbox distribution objects. ``makedist`` below returns an
object with the members the algorithms use (``DistributionName``, the
parameters, ``icdf``, ``cdf``, ``pdf``, ``mean()``, ``std()``), backed by
``scipy.stats``. Parameter names follow MATLAB.

Any frozen ``scipy.stats`` distribution can also be wrapped directly with
``Distribution("Name", frozen)``.
"""
import numpy as np
from scipy import stats


class Distribution:
    """Thin wrapper around a frozen scipy.stats distribution."""

    def __init__(self, name, frozen, **params):
        self.DistributionName = name
        self._dist = frozen
        for key, val in params.items():
            setattr(self, key, val)

    def icdf(self, p):
        return self._dist.ppf(p)

    def cdf(self, x):
        return self._dist.cdf(x)

    def pdf(self, x):
        return self._dist.pdf(x)

    def logpdf(self, x):
        return self._dist.logpdf(x)

    def mean(self):
        return float(self._dist.mean())

    def std(self):
        return float(self._dist.std())

    def random(self, *size):
        return self._dist.rvs(size=size)

    def __repr__(self):
        params = {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k != "DistributionName"}
        return f"Distribution({self.DistributionName!r}, {params})"


def makedist(name, **p):
    """Create a distribution object, e.g. ``makedist("Normal", mu=0, sigma=1)``.

    Supported names and parameters (MATLAB conventions):

    ==========================  ==========================================
    ``"Normal"``                ``mu``, ``sigma``
    ``"Lognormal"``             ``mu``, ``sigma`` (of the underlying normal)
    ``"Uniform"``               ``lower``, ``upper``
    ``"GeneralizedExtremeValue"`` ``k``, ``sigma``, ``mu``
    ``"ExtremeValue"``          ``mu``, ``sigma`` (Gumbel for minima)
    ``"Gamma"``                 ``a`` (shape), ``b`` (scale)
    ``"Weibull"``               ``a`` (scale), ``b`` (shape)
    ``"Exponential"``           ``mu`` (mean)
    ``"Beta"``                  ``a``, ``b``
    ``"Rayleigh"``              ``b``
    ==========================  ==========================================
    """
    key = name.replace(" ", "").lower()
    if key == "normal":
        mu, sigma = p.get("mu", 0.0), p.get("sigma", 1.0)
        return Distribution("Normal", stats.norm(loc=mu, scale=sigma), mu=mu, sigma=sigma)
    if key == "lognormal":
        mu, sigma = p.get("mu", 0.0), p.get("sigma", 1.0)
        return Distribution("Lognormal", stats.lognorm(s=sigma, scale=np.exp(mu)), mu=mu, sigma=sigma)
    if key == "uniform":
        lower, upper = p.get("lower", 0.0), p.get("upper", 1.0)
        return Distribution("Uniform", stats.uniform(loc=lower, scale=upper - lower), lower=lower, upper=upper)
    if key == "generalizedextremevalue":
        k, sigma, mu = p.get("k", 0.0), p.get("sigma", 1.0), p.get("mu", 0.0)
        # MATLAB shape k corresponds to scipy's c = -k
        return Distribution("Generalized Extreme Value", stats.genextreme(c=-k, loc=mu, scale=sigma),
                            k=k, sigma=sigma, mu=mu)
    if key == "extremevalue":
        mu, sigma = p.get("mu", 0.0), p.get("sigma", 1.0)
        return Distribution("Extreme Value", stats.gumbel_l(loc=mu, scale=sigma), mu=mu, sigma=sigma)
    if key == "gamma":
        a, b = p.get("a", 1.0), p.get("b", 1.0)
        return Distribution("Gamma", stats.gamma(a, scale=b), a=a, b=b)
    if key == "weibull":
        a, b = p.get("a", 1.0), p.get("b", 1.0)
        return Distribution("Weibull", stats.weibull_min(c=b, scale=a), a=a, b=b)
    if key == "exponential":
        mu = p.get("mu", 1.0)
        return Distribution("Exponential", stats.expon(scale=mu), mu=mu)
    if key == "beta":
        a, b = p.get("a", 1.0), p.get("b", 1.0)
        return Distribution("Beta", stats.beta(a, b), a=a, b=b)
    if key == "rayleigh":
        b = p.get("b", 1.0)
        return Distribution("Rayleigh", stats.rayleigh(scale=b), b=b)
    raise ValueError(f"Unsupported distribution name: {name!r}")


def repmat(dist, n):
    """Column array of ``n`` copies of a distribution (MATLAB ``repmat(dist, n, 1)``)."""
    return [dist] * n
