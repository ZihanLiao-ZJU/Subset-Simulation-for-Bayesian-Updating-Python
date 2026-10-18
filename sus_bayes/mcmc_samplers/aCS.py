import copy

import numpy as np

from .._utils import rand, randn

_EPS = np.finfo(float).eps


class aCS:
    """Adaptive conditional sampling with Nataf transformation.

    Constants
        SedSav : save the seeds or not (default: False)
        AcpOpt : optimal acceptance rate (default: 0.44)
        NumIte : number of adaptation rounds (default: 5)
        Lambda : initial scaling factor (default: 0.6)
    """

    SedSav = False
    AcpOpt = 0.44
    NumIte = 5
    Lambda = 0.6

    def __init__(self, BayStc):
        self.BayStc = BayStc
        self.Nfun = BayStc.Nfun
        self.Ndim = BayStc.Ndim

    def SamGen(self, *args):
        """Generate samples distributed as prior * intermediate likelihood.

        SamGen(Nc, Ns)              : start from Nc prior samples (no seeds)
        SamGen(u_sed, y_sed, Ns)    : Markov chains started from seeds

        Returns u [Ndim x Nc x Ns], y [Nfun x Nc x Ns] and Ncal.
        """
        baystc = self.BayStc
        sedsav = self.SedSav or len(args) <= 2
        nDim, nFun = self.Ndim, self.Nfun
        Nite = self.NumIte
        lam = self.Lambda
        acpopt = self.AcpOpt
        if len(args) <= 2:
            Nc, Ns = int(args[0]), int(args[1])
            u_sed = randn(nDim, Nc)
            baystc = baystc.EvlY(u_sed)
            y_sed = np.vstack([baystc.EvlLKF(), baystc.EvlPDF(), baystc.Y])
            Ncal = baystc.Ncal
        else:
            u_sed = np.array(args[0], dtype=float, copy=True)
            y_sed = np.array(args[1], dtype=float, copy=True)
            Ns = int(args[2])
            Nc = u_sed.shape[1]
            Ncal = 0

        u = np.zeros((nDim, Nc, Ns))
        y = np.zeros((nFun, Nc, Ns))

        du = randn(nDim, Nc, Ns)
        p = rand(1, Nc, Ns)

        # group the chains; [start, end] (1-based, inclusive) of each group
        Ngrp = max(int(np.ceil(Nc / Nite)), 1)
        ind_grp = np.minimum(np.array([[1, Ngrp]] * Nite) + np.arange(0, Nite * Ngrp, Ngrp)[:, None], Nc)
        sigma0 = np.maximum(np.std(u_sed, axis=1, ddof=1 if Nc > 1 else 0, keepdims=True), _EPS)

        for it in range(1, Nite + 1):
            sigma = np.minimum(lam * sigma0, 1)
            rho = np.sqrt(1 - sigma ** 2)
            Nacp = 0
            dNcal = 0
            ind_sam = np.arange(ind_grp[it - 1, 0] - 1, ind_grp[it - 1, 1])
            for k in range(Ns):
                if k == 0 and sedsav:
                    u[:, ind_sam, k] = u_sed[:, ind_sam]
                    y[:, ind_sam, k] = y_sed[:, ind_sam]
                else:
                    u[:, ind_sam, k] = rho * u_sed[:, ind_sam] + sigma * du[:, ind_sam, k]
                    baystc = baystc.EvlY(u[:, ind_sam, k])
                    y[:, ind_sam, k] = np.vstack([baystc.EvlLKF(), baystc.EvlPDF(), baystc.Y])
                    with np.errstate(invalid="ignore"):
                        ind_rej = np.exp(y[0, ind_sam, k] - y_sed[0, ind_sam]) < p[0, ind_sam, k]
                    ind_sam_rej = ind_sam[ind_rej]
                    u[:, ind_sam_rej, k] = u_sed[:, ind_sam_rej]
                    y[:, ind_sam_rej, k] = y_sed[:, ind_sam_rej]
                    Nacp += len(ind_sam) - len(ind_sam_rej)
                    dNcal += baystc.Ncal
                    u_sed[:, ind_sam] = u[:, ind_sam, k]
                    y_sed[:, ind_sam] = y[:, ind_sam, k]
            with np.errstate(invalid="ignore", divide="ignore"):
                acp = Nacp / dNcal if dNcal else np.nan
            lam = np.exp(np.log(lam) + (acp - acpopt) / np.sqrt(it))
            Ncal += dNcal
        return u, y, Ncal
