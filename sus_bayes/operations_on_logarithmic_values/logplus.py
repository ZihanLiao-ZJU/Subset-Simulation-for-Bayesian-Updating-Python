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


def logplus(A, B):
    """Logarithmic version of A + B."""
    A = np.asarray(A)
    B = np.asarray(B)
    cpx = np.iscomplexobj(A) or np.iscomplexobj(B)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        if cpx:
            C = np.log(np.exp(A.astype(complex)) + np.exp(B.astype(complex)))
            C = np.array(C)
        else:
            maxAB = np.fmax(A, B)
            C = np.array(maxAB + np.log(np.exp(A - maxAB) + np.exp(B - maxAB)), dtype=float)
    Ab, Bb = np.broadcast_arrays(np.real(A), np.real(B))
    ind1 = Ab > 0
    ind2 = Bb > 0
    ind3 = np.isinf(Ab)
    ind4 = np.isinf(Bb)
    C = np.broadcast_to(C, Ab.shape).copy()
    C[(ind1 & ind3) | (ind2 & ind4)] = np.inf
    C[(~ind1 & ind3) & (~ind2 & ind4)] = -np.inf
    C = _real_if_possible(C)
    return C[()] if C.ndim == 0 else C

