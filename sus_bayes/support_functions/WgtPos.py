import numpy as np

from .._utils import flat, reshape_F, tail_shape


def WgtPos(g, y):
    """Log-weights of the posterior distribution at every level."""
    Nite = len(y)
    w_pos = [None] * Nite
    for ite in range(Nite):
        yi = flat(y[ite])
        with np.errstate(divide="ignore", invalid="ignore"):
            w = yi[2, :] + yi[3, :] - yi[0, :] - yi[1, :]
            if ite < Nite - 1:
                w = w + np.log((yi[2, :] < g[ite + 1][2]).astype(float))
        w = reshape_F(w, tail_shape(y[ite])).astype(float)
        w[np.isnan(w)] = -np.inf
        w_pos[ite] = w
    return w_pos
