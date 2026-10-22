"""
Operations on logarithmic values
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

_EPS = np.finfo(float).eps


def _real_if_possible(C):
    """Drop the imaginary part when it is zero everywhere (MATLAB behaviour)."""
    if np.iscomplexobj(C):
        im = np.imag(C)
        if np.all((im == 0) | np.isnan(im)):
            return np.real(C)
    return C


def _isreal(A):
    """MATLAB isreal: True when the array is not stored as complex."""
    return not np.iscomplexobj(A)


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


def logmean(A, axis=None):
    """Logarithmic version of mean: log(mean(exp(A)))."""
    A = np.asarray(A)
    if axis is None:
        return logsum(A, None) - np.log(A.size)
    return logsum(A, axis) - np.log(A.shape[axis])


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


def logmultiply(A, B):
    """Logarithmic version of the matrix product A*B."""
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    m, n = A.shape[0], B.shape[1]
    C = np.zeros((m, n), dtype=complex if (np.iscomplexobj(A) or np.iscomplexobj(B)) else float)
    for irow in range(m):
        C[irow, :] = logsum(A[irow, :][:, None] + B, 0)
    return _real_if_possible(C)


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


def logstd(A, axis=None):
    """Logarithmic version of std: log(std(exp(A)))."""
    A = np.asarray(A)
    logmu = logmean(A, axis)
    n = A.size if axis is None else A.shape[axis]
    return (logsum(2 * np.real(logminus(A, logmu)), axis) - np.log(n - 1)) / 2
