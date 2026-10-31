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


def logminus(A, B):
    """Logarithmic version of A - B (complex where A < B)."""
    A = np.asarray(A)
    B = np.asarray(B)
    cpx = np.iscomplexobj(A) or np.iscomplexobj(B)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        if cpx:
            # log(exp(A) - exp(B)) in complex arithmetic (same as MATLAB's
            # pi*1i + logplus(A - pi*1i, B) rules for complex inputs)
            C = np.log(np.exp(A.astype(complex)) - np.exp(B.astype(complex)))
        else:
            maxAB = np.fmax(A, B)
            D = np.exp(A - maxAB) - np.exp(B - maxAB)
            if np.any(D < 0):
                C = maxAB + np.log(D.astype(complex))
            else:
                C = maxAB + np.log(D)
    Ar, Br = np.broadcast_arrays(np.real(A), np.real(B))
    C = np.array(np.broadcast_to(C, Ar.shape))
    ind1 = Ar > 0
    ind2 = np.isinf(Ar)
    ind3 = Br > 0
    ind4 = np.isinf(Br)
    C[ind1 & ind2] = np.inf                       # A = +inf
    C[~ind3 & ind4 & ~ind1 & ind2] = -np.inf      # A = -inf & B = -inf
    C = _real_if_possible(C)
    return C[()] if C.ndim == 0 else C

