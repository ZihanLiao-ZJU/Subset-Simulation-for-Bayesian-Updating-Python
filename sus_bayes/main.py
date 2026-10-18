"""
=======================================================================================

      Sequential Multiple Importance Sampling Framework
      -------------------------------------------------
      Developed by:
          Zihan Liao, Zhejiang University
          Binbin Li, Zhejiang University
          Hua-Ping Wan, Zhejiang University

      Initial Version: Jan 2023
      Last Modified  : Apr 2025
      Python port    : Oct 2026

=======================================================================================
"""
import copy

import numpy as np

from ._utils import mround
from .support_functions.DspRst import DspRst
from .support_functions.VabSrk import VabSrk


class Main:
    """Sequential sampling loop (MATLAB class ``main``).

    Properties
        IntBay : intermediate Bayesian model
        Sampler: sampler object (must contain .BayStc and .SamGen)
        ConPrb : target conditional probability (default 0.1)
        NumSam : number of samples per iteration (scalar or vector)
        G      : predefined intermediate distribution parameters
        Verbose: print the progress of each level (default True)
    """

    MaxIte = 10000

    def __init__(self, Sampler):
        if Sampler is None:
            raise ValueError("Inputs must include the Sampler object with a valid BayStc field.")
        self.Sampler = Sampler
        self.IntBay = Sampler.BayStc
        self.ConPrb = 0.1
        self.NumSam = 1000
        self.G = []
        self.X = []
        self.Y = []
        self.Verbose = True

    def RunIte(self):
        """Perform the sequential sampling iterations; returns a dict with all levels."""
        intBay = self.IntBay
        sampler = copy.copy(self.Sampler)
        p = self.ConPrb
        maxi = self.MaxIte
        NumSam = np.atleast_1d(self.NumSam)
        flg_Nsam = NumSam.size > 1
        flg_IntD = len(self.G) > 0

        Nsam = np.zeros(maxi + 1, dtype=int)
        Nsze = np.zeros((maxi + 1, 2))
        Ncal = np.zeros(maxi + 1, dtype=int)
        x = [None] * (maxi + 1)
        y = [None] * (maxi + 1)
        g = [None] * (maxi + 1)

        if flg_Nsam:
            Nsam[:NumSam.size] = NumSam
        else:
            Nsam[:] = NumSam[0]
        Nsze[0, :] = [Nsam[0], 1]

        if flg_IntD:
            Nite_int = len(self.G)
            g[:Nite_int] = list(self.G)
        else:
            Nite_int = 0
            g[0] = intBay.G

        x_sed = y_sed = None
        flg_cvg = False
        for ite in range(1, maxi + 1):
            i = ite - 1
            # === Sample generation ===
            if np.all(Nsze[i, :] > 0) and not np.isnan(Nsze[i, 1]):
                if ite == 1:
                    x[i], y[i], Ncal[i] = sampler.SamGen(int(Nsze[i, 0]), int(Nsze[i, 1]))[:3]
                else:
                    x[i], y[i], Ncal[i] = sampler.SamGen(x_sed, y_sed, int(Nsze[i, 1]))[:3]
            Nsam[i] = int(np.prod(Nsze[i, :]))

            # === Update intermediate distribution ===
            if ite + 1 <= Nite_int:
                intBay = copy.copy(intBay)
                intBay.G = g[i + 1]
            else:
                g[i + 1], intBay = intBay.UpdObj(y[i], p)
            sampler.BayStc = intBay

            # === Seed selection ===
            x_sed, y_sed = intBay.SltSed(x[i], y[i], int(Nsam[i + 1]))
            Nsze[i + 1, 0] = x_sed.shape[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                Nsze[i + 1, 1] = mround(Nsam[i + 1] / Nsze[i + 1, 0]) if Nsze[i + 1, 0] > 0 else np.inf

            if self.Verbose:
                DspRst(ite, Nsam[i], Nsze[i, :], Ncal[i])

            # === Convergence or termination ===
            if Nsze[i + 1, 0] <= 0 or ite + 1 == maxi:
                flg_cvg = False
                break
            elif intBay.FlgCvg:
                flg_cvg = True
                break

        Nite = ite
        x, y, Nsam, Nsze, Ncal = VabSrk(Nite, x, y, Nsam, Nsze, Ncal)
        g = VabSrk(Nite + 1, g)

        return dict(x=x, y=y, g=g, Nsam=Nsam, Nsze=Nsze.astype(int), Ncal=Ncal,
                    Nite=Nite, flg_cvg=flg_cvg, intBay=intBay)


# MATLAB name
main = Main
