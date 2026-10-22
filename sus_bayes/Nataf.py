"""
Nataf transformation between the physical variable space (X) and the
standard normal space (U), given marginal distributions and a correlation
matrix.

Reference:
    Liu & Der Kiureghian (1986), Structural Safety 3(1): 1-12
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from ._utils import randn

from .support_functions.logGauss import logGauss


def GauIntPot(Nint, u_min, u_max):
    """Tensor-product Gauss-Legendre points and weights in 2D.

    Returns ``u`` of shape [Npts, 2] and ``w`` of shape [Npts].
    """
    x1, w1 = np.polynomial.legendre.leggauss(int(Nint[0]))
    x2, w2 = np.polynomial.legendre.leggauss(int(Nint[1]))
    x1 = 0.5 * (u_max[0] - u_min[0]) * x1 + 0.5 * (u_max[0] + u_min[0])
    x2 = 0.5 * (u_max[1] - u_min[1]) * x2 + 0.5 * (u_max[1] + u_min[1])
    w1 = 0.5 * (u_max[0] - u_min[0]) * w1
    w2 = 0.5 * (u_max[1] - u_min[1]) * w2
    U1, U2 = np.meshgrid(x1, x2, indexing="ij")
    W = np.outer(w1, w2)
    return np.column_stack([U1.ravel(), U2.ravel()]), W.ravel()


class Nataf:
    """Nataf transformation.

    Parameters
    ----------
    Marg_X : list of distribution objects (see ``makedist``), length Ndim
    Rho_X  : correlation matrix in X space, [Ndim, Ndim]
    """

    Nfun = 1
    LogP = True  # all outputs are in log-domain

    def __init__(self, Marg_X, Rho_X):
        self.Marg_X = list(Marg_X)
        self.Rho_X = np.asarray(Rho_X, dtype=float)
        self.Ndim = len(self.Marg_X)
        ndim = self.Ndim

        try:
            self.RhoL_X = np.linalg.cholesky(self.Rho_X)
        except np.linalg.LinAlgError:
            raise ValueError("Correlation matrix Rho_X is not positive definite.")

        if np.all(self.Rho_X == np.eye(ndim)):
            Rho_U = np.eye(ndim)
        else:
            Rho_U = np.eye(ndim)
            for i in range(ndim):
                for j in range(i + 1, ndim):
                    d1 = self.Marg_X[i].DistributionName.lower()
                    d2 = self.Marg_X[j].DistributionName.lower()
                    rho_x = self.Rho_X[i, j]
                    if rho_x == 0:
                        Rho_U[i, j] = 0.0
                    elif d1 == "normal" and d2 == "normal":
                        Rho_U[i, j] = rho_x
                    elif d1 == "normal" and d2 == "lognormal":
                        cv_j = self.Marg_X[j].std() / self.Marg_X[j].mean()
                        Rho_U[i, j] = rho_x * cv_j / np.sqrt(np.log(1 + cv_j ** 2))
                    elif d1 == "lognormal" and d2 == "normal":
                        cv_i = self.Marg_X[i].std() / self.Marg_X[i].mean()
                        Rho_U[i, j] = rho_x * cv_i / np.sqrt(np.log(1 + cv_i ** 2))
                    elif d1 == "lognormal" and d2 == "lognormal":
                        cv_i = self.Marg_X[i].std() / self.Marg_X[i].mean()
                        cv_j = self.Marg_X[j].std() / self.Marg_X[j].mean()
                        Rho_U[i, j] = np.log(1 + rho_x * cv_i * cv_j) / np.sqrt(
                            np.log(1 + cv_i ** 2) * np.log(1 + cv_j ** 2))
                    else:
                        # 2D Gauss-Legendre integration for nonstandard pairs
                        mi, mj = self.Marg_X[i], self.Marg_X[j]
                        u, w = GauIntPot([200, 200], [-8, -8], [8, 8])
                        fi = (mi.icdf(norm.cdf(u[:, 0])) - mi.mean()) / mi.std()
                        fj = (mj.icdf(norm.cdf(u[:, 1])) - mj.mean()) / mj.std()
                        coef = fi * fj * w

                        def rhoU2X_err(rho_u):
                            return abs(rho_x - np.sum(
                                coef / (2 * np.pi * np.sqrt(1 - rho_u ** 2)) *
                                np.exp(-1 / (2 * (1 - rho_u ** 2)) *
                                       (u[:, 0] ** 2 - 2 * rho_u * u[:, 0] * u[:, 1] + u[:, 1] ** 2))))

                        # The inverse-CDF maps are increasing, so the latent
                        # correlation has the same sign as the target. The
                        # MATLAB source's (0, 1) interval loses negative pairs.
                        bounds = (-1, 0) if rho_x < 0 else (0, 1)
                        res = minimize_scalar(rhoU2X_err, bounds=bounds, method="bounded",
                                              options={"xatol": 1e-4})
                        Rho_U[i, j] = res.x
                    Rho_U[j, i] = Rho_U[i, j]
            Rho_U = np.clip(Rho_U, -1, 1)
        self.Rho_U = Rho_U

        try:
            self.RhoL_U = np.linalg.cholesky(self.Rho_U)
        except np.linalg.LinAlgError:
            raise ValueError("Transformed correlation matrix Rho_U is not positive definite.")

    def U2X(self, U):
        """Standard normal U-space -> physical X-space. U has shape [Ndim, ...]."""
        U = np.asarray(U, dtype=float)
        Z = np.einsum("ij,j...->i...", self.RhoL_U, U)
        X = np.zeros_like(Z)
        for idim in range(self.Ndim):
            marg = self.Marg_X[idim]
            name = marg.DistributionName.lower()
            has_normal_params = all(hasattr(marg, p) for p in ("mu", "sigma"))
            if name == "normal" and has_normal_params:
                X[idim] = marg.mu + marg.sigma * Z[idim]
            elif name == "lognormal" and has_normal_params:
                X[idim] = np.exp(marg.mu + marg.sigma * Z[idim])
            elif name == "generalized extreme value" and all(
                    hasattr(marg, p) for p in ("mu", "sigma", "k")):
                p = np.clip(norm.cdf(Z[idim]), 1e-15, 1 - 1e-15)
                if abs(marg.k) < 1e-12:
                    X[idim] = marg.mu - marg.sigma * np.log(-np.log(p))
                else:
                    X[idim] = marg.mu + (marg.sigma / marg.k) * ((-np.log(p)) ** (-marg.k) - 1)
            else:
                X[idim] = marg.icdf(norm.cdf(Z[idim]))
        return X

    def X2U(self, X):
        """Physical X-space -> standard normal U-space."""
        X = np.asarray(X, dtype=float)
        Z = np.zeros_like(X)
        for idim in range(self.Ndim):
            Z[idim] = norm.ppf(self.Marg_X[idim].cdf(X[idim]))
        return np.einsum("ij,j...->i...", np.linalg.inv(self.RhoL_U), Z)

    def EvlPDF(self, X):
        """Log joint PDF of samples X [Ndim, Nsam]."""
        X = np.asarray(X, dtype=float)
        ndim, nsam = X.shape
        Z = np.zeros((ndim, nsam))
        logpdf_Z = np.zeros((ndim, nsam))
        logpdf_Y = np.zeros((ndim, nsam))
        for idim in range(ndim):
            Z[idim] = norm.ppf(self.Marg_X[idim].cdf(X[idim]))
            logpdf_Z[idim] = logGauss(Z[idim:idim + 1])
            with np.errstate(divide="ignore"):
                logpdf_Y[idim] = np.log(self.Marg_X[idim].pdf(X[idim]))
        logpdf_U = logGauss(Z, np.zeros((ndim, 1)), self.Rho_U)
        return np.sum(logpdf_Y, 0) - np.sum(logpdf_Z, 0) + logpdf_U

    def EvlCDF(self, X):
        """Log CDF-related quantity of samples X (as in the MATLAB code)."""
        X = np.asarray(X, dtype=float)
        ndim, nsam = X.shape
        Z = np.zeros((ndim, nsam))
        for idim in range(ndim):
            Z[idim] = norm.ppf(self.Marg_X[idim].cdf(X[idim]))
        return logGauss(Z, np.zeros((ndim, 1)), self.Rho_U)

    def SamGen(self, N):
        """Generate N random samples in X-space."""
        U = randn(self.Ndim, N)
        return self.U2X(U)
