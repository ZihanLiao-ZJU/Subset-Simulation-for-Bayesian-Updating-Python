"""Operations on logarithmic values
--------------------------------
Python port of the MATLAB folder "Operations on Logarithmic values"
(Zihan Liao, Zhejiang University, 2022.11).

Basic arithmetic on values stored as logarithms, used to avoid overflow and
underflow. As in MATLAB, the logarithm of a negative number is complex,
log(-a) = log(a) + i*pi, so differences that become negative are carried as
complex values.

Axis convention: ``axis=None`` corresponds to MATLAB's ``"all"``, ``axis=0``
to MATLAB ``dim = 1`` (along rows) and ``axis=1`` to MATLAB ``dim = 2``.
Reductions along an axis keep that axis with length one, as MATLAB does.
"""

import numpy as np
from .logsum import logsum


def logmean(A, axis=None):
    """Logarithmic version of mean: log(mean(exp(A)))."""
    A = np.asarray(A)
    if axis is None:
        return logsum(A, None) - np.log(A.size)
    return logsum(A, axis) - np.log(A.shape[axis])

