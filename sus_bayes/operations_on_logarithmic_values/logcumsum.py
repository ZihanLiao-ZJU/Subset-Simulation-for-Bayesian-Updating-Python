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
from ._helpers import _real_if_possible
from .logsum import logsum


def logcumsum(A, axis):
    """Logarithmic version of cumsum along ``axis`` (0 or 1)."""
    A = np.atleast_2d(A)
    nrow, ncol = A.shape
    B = np.zeros((nrow, ncol), dtype=complex if np.iscomplexobj(A) else float)
    if axis == 0:
        for k in range(nrow):
            B[k, :] = logsum(A[:k + 1, :], 0)
    elif axis == 1:
        for k in range(ncol):
            B[:, k] = logsum(A[:, :k + 1], 1)[:, 0]
    return _real_if_possible(B)

