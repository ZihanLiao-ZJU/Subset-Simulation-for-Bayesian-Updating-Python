import copy

import numpy as np

from ._utils import mround, randsample
from .log_operations import logmean, logsum
from .support_functions.CorMod import CorMod
from .support_functions.CorMod_cross import CorMod_cross
from .support_functions.ResRat import ResRat
from .support_functions.VabCmb import VabCmb
from .support_functions.WgtEvd import WgtEvd
from .support_functions.WgtPos import WgtPos


class NumSim:
    """Numerical post-processing of subset simulation samples.

    obj = NumSim(out)                      # out: output of Main.RunIte
    x_pos, y_pos, x_all, y_all = obj.PosSim()
    cv_z_hat, cv_z = obj.EvlVar()

    Properties
        Wevd : weights for evidence estimation
        Wpos : weights of the posterior distribution
        Wres : resampling weights between levels
        Z    : log-evidence estimate
        Zi   : log-evidence contribution of each level

    Reference: Handley & Lemos (2019), Physical Review D 100(2): 023512
    """

    def __init__(self, inp):
        self.X = inp["x"]
        self.Y = inp["y"]
        self.G = inp["g"]
        self.NumIte = inp["Nite"]
        self.NumSam = inp["Nsam"]
        self.SamSze = inp["Nsze"]
        self.IntBay = inp["intBay"]
        self.H = None

        Nite = self.NumIte
        y = [np.array(v, copy=True) for v in self.Y]
        g = self.G
        intBay = copy.copy(self.IntBay)

        # normalize log-likelihoods across iterations
        w_res = [None] * Nite
        c = np.zeros(Nite)
        for ite in range(Nite - 1):
            intBay.G = g[ite + 1]
            w_res[ite], _ = ResRat(intBay, y[ite])
            c[ite + 1] = c[ite] + logmean(w_res[ite], None)
            y[ite + 1][0] = y[ite + 1][0] - c[ite + 1]
        self.C = c

        # weights and evidence
        w_evd = WgtEvd(g, y)
        w_pos = WgtPos(g, y)
        z_i = np.zeros(Nite)
        for ite in range(Nite):
            z_i[ite] = np.real(c[ite] + logmean(w_evd[ite], None))
        z = logsum(z_i, None)

        # normalize posterior weights
        w_pos = [w - z for w in w_pos]

        self.Wevd = w_evd
        self.Wpos = w_pos
        self.Wres = w_res
        self.Z = float(np.real(z))
        self.Zi = z_i

    def PosSim(self):
        """Resample posterior samples according to their weights."""
        x_all, y_all = VabCmb(self.X, self.Y)
        w_pos = VabCmb(self.Wpos).ravel(order="F")
        Npos = int(mround(np.exp(2 * logsum(w_pos, None) - logsum(2 * w_pos, None))))
        Nall = y_all.shape[1]
        w_pos_norm = np.exp(w_pos - np.max(w_pos))
        ind_pos = randsample(Nall, Npos, w_pos_norm)
        return x_all[:, ind_pos], y_all[:, ind_pos], x_all, y_all

    def EvlVar(self):
        """Coefficient of variation of the evidence estimate.

        Returns (cv_z_hat, cv_z): with and without the Markov chain correlation correction.
        """
        w_res, w_evd = self.Wres, self.Wevd
        z_i, z = self.Zi, self.Z
        Nite = self.NumIte

        c_p_hat = np.zeros(Nite); c_p = np.zeros(Nite); gamma_p = np.zeros(Nite)
        for ite in range(Nite - 1):
            c_p_hat[ite], c_p[ite], gamma_p[ite] = CorMod(w_res[ite])

        c_h_hat = np.zeros(Nite); c_h = np.zeros(Nite); gamma_h = np.zeros(Nite)
        for ite in range(Nite):
            c_h_hat[ite], c_h[ite], gamma_h[ite] = CorMod(w_evd[ite])

        rho_hp = np.zeros(max(Nite - 1, 0)); rho_fi = np.zeros(max(Nite - 1, 0))
        gamma_hp = np.zeros(max(Nite - 1, 0))
        for ite in range(Nite - 1):
            rho_fi[ite], gamma_hp[ite] = CorMod_cross(w_evd[ite], w_res[ite])
            rho_hp[ite] = rho_fi[ite] * (1 + gamma_hp[ite]) / np.sqrt(1 + gamma_p[ite]) / np.sqrt(1 + gamma_h[ite])

        cov_z_hat = np.zeros((Nite, Nite))
        cov_z = np.zeros((Nite, Nite))
        with np.errstate(divide="ignore", invalid="ignore"):
            for ite in range(Nite):
                z1z1 = 2 * z_i[ite]
                sumcp_hat = np.sum(c_p_hat[:ite] ** 2)
                sumcp = np.sum(c_p[:ite] ** 2)
                cov_z_hat[ite, ite] = z1z1 + np.log(c_h_hat[ite] ** 2 + sumcp_hat)
                cov_z[ite, ite] = z1z1 + np.log(c_h[ite] ** 2 + sumcp)
                if ite < Nite - 1:
                    cpch_hat = c_p_hat[ite] * c_h_hat[ite] * rho_hp[ite]
                    cpch = c_p[ite] * c_h[ite] * rho_fi[ite]
                    z1z2 = z_i[ite] + z_i[ite + 1:]
                    cov_z_hat[ite, ite + 1:] = z1z2 + np.log(sumcp_hat + cpch_hat)
                    cov_z[ite, ite + 1:] = z1z2 + np.log(sumcp + cpch)
                    cov_z_hat[ite + 1:, ite] = cov_z_hat[ite, ite + 1:]
                    cov_z[ite + 1:, ite] = cov_z[ite, ite + 1:]
        cov_z_hat[np.isnan(cov_z_hat)] = -np.inf

        cv_z_hat = float(np.real(np.exp(logsum(cov_z_hat, None) / 2 - z)))
        cv_z = float(np.real(np.exp(logsum(cov_z, None) / 2 - z)))
        return cv_z_hat, cv_z
