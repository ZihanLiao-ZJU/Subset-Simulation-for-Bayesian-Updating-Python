import copy

import numpy as np

from .._utils import flat, nanmax, reshape_F, tail_shape


def ResRat(intBay, y):
    """Log resampling ratio between the current and the next intermediate level.

    Returns ``lograt`` [Nc x Ns] (shifted so that its maximum is 0) and the shift ``c``.
    """
    y_tmp = flat(y)
    intBay = copy.copy(intBay)
    intBay.Y = y_tmp[2:, :]
    Li = intBay.EvlLKF()
    pi = intBay.EvlPDF()
    with np.errstate(invalid="ignore"):
        lograt_tmp = Li + pi - y_tmp[0, :] - y_tmp[3, :]
        c = nanmax(lograt_tmp)
        lograt_tmp = lograt_tmp - c
    lograt_tmp[np.isnan(lograt_tmp)] = -np.inf
    lograt = reshape_F(lograt_tmp, tail_shape(y))
    return lograt, c
