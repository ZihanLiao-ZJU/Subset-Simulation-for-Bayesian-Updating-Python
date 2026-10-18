import numpy as np
from scipy.linalg import solve_triangular


def logGauss(u, mu=None, sigma=None):
    """Log pdf of a Gaussian distribution.

    u     : samples                          [Ndim, Nsam] (or [Ndim, ...])
    mu    : mean vector                      [Ndim, 1]
    sigma : covariance matrix                [Ndim, Ndim]
    Returns the log pdf with shape [Nsam] (or [...]).
    """
    u = np.asarray(u, dtype=float)
    ndim = u.shape[0]
    shp = u.shape[1:]
    u = u.reshape(ndim, -1, order="F")
    if mu is None:
        q = np.sum(u * u, axis=0)
        c = ndim * np.log(2 * np.pi)
    else:
        mu = np.asarray(mu, dtype=float).reshape(ndim, 1)
        sigma = np.asarray(sigma, dtype=float)
        assert sigma.shape == (ndim, ndim)
        try:
            R = np.linalg.cholesky(sigma)  # lower triangular, R @ R.T = sigma
        except np.linalg.LinAlgError:
            raise ValueError("ERROR: Sigma is not Positive Defined!")
        Q = solve_triangular(R, u - mu, lower=True)
        q = np.sum(Q * Q, axis=0)
        c = ndim * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(R)))
    p = -0.5 * (c + q)
    return p.reshape(shp, order="F")
