import numpy as np

from .._utils import flat, mround, randsample
from .ResRat import ResRat


def SedSlt(IntBay, x, y, Nsam):
    """Select the seeds for the next iteration by weighted random resampling.

    x : generated samples [Ndim, ...];  y : function values [Nfun, ...]
    Returns (x_sed, y_sed).
    """
    lograt, _ = ResRat(IntBay, y)
    w_rat = np.exp(lograt)
    Nsed_can = np.sum(w_rat)
    Ns = mround(Nsam / np.arange(1, Nsam + 1))
    Nsed = mround(Nsam / Ns)
    ok = np.nonzero(Nsed_can - Nsed >= 0)[0]
    Nsed = int(Nsed[ok[-1]]) if ok.size else 0
    x = flat(x)
    y = flat(y)
    if Nsed > 0:
        ind = randsample(y.shape[1], Nsed, w_rat)
        x_sed = x[:, ind]
        y_sed = y[:, ind]
    else:
        x_sed = np.zeros((x.shape[0], 0))
        y_sed = np.zeros((y.shape[0], 0))
    if y_sed.size:
        y_sed = IntBay.UpdY(y_sed)
    return x_sed, y_sed
