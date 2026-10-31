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
from ._helpers import _EPS, _real_if_possible


def logsum(A, axis=None):
    """Logarithmic version of sum: log(sum(exp(A)))."""
    A = np.asarray(A)
    if A.size == 0:
        return -np.inf
    if np.iscomplexobj(A):
        ind_rel = np.imag(A) == 0
        A_rel = np.where(ind_rel, np.real(A), -np.inf)
        A_cpx = np.where(ind_rel, -np.inf, np.real(A))
        maxlogx = np.max(A_rel, axis=axis, keepdims=axis is not None)
        minlogx = np.min(A_cpx, axis=axis, keepdims=axis is not None) + np.pi * 1j
    else:
        A_rel = np.where(np.isnan(A), -np.inf, A)
        maxlogx = np.max(A_rel, axis=axis, keepdims=axis is not None)
        minlogx = np.full_like(np.asarray(maxlogx, dtype=float), -np.inf) + np.pi * 1j
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        S = np.sum(np.exp(A - maxlogx), axis=axis, keepdims=axis is not None)
        if np.iscomplexobj(S) or np.any(np.real(S) < 0):
            B = maxlogx + np.log(S.astype(complex))
        else:
            B = maxlogx + np.log(S)
    B = np.array(B, dtype=complex if np.iscomplexobj(B) else float)
    maxlogx = np.broadcast_to(maxlogx, B.shape)
    minlogx = np.broadcast_to(minlogx, B.shape)
    ind1 = maxlogx < 0
    ind2 = np.real(minlogx) > 0
    ind3 = np.isinf(maxlogx)
    ind4 = np.isinf(minlogx)
    B[ind1 & ind3] = -np.inf          # max = -inf
    B[ind2 & ind4] = np.inf           # min = +inf
    B[~ind1 & ind3] = np.inf          # max = +inf
    if np.iscomplexobj(B):
        ind_img = np.imag(B) <= _EPS
        B[ind_img] = np.real(B[ind_img])
        B = _real_if_possible(B)
    if axis is None:
        return B[()] if B.ndim == 0 else B
    return B

