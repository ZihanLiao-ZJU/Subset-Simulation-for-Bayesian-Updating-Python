import numpy as np

from .._utils import reshape_F, tail_shape
from ..log_operations import logminus
from .._utils import flat


def _mmin(a, b):
    """MATLAB min of two arrays (complex values are compared by magnitude)."""
    a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        return np.where(np.abs(b) < np.abs(a), b, a)
    return np.fmin(a, b)


def WgtEvd(g, y):
    """Log-weights for evidence estimation at every level."""
    Nite = len(y)
    w_evd = [None] * Nite
    for ite in range(Nite - 1):
        Li = g[ite][2]
        Li1 = g[ite + 1][2]
        w = _mmin(logminus(flat(y[ite])[2, :], Li), logminus(Li1, Li))
        w_evd[ite] = reshape_F(w, tail_shape(y[ite]))
    Li = g[Nite - 1][2]
    w = logminus(flat(y[Nite - 1])[2, :], Li)
    w_evd[Nite - 1] = reshape_F(w, tail_shape(y[Nite - 1]))
    return w_evd
