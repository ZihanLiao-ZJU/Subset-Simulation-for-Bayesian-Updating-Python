"""Shared Python helpers for logarithmic arithmetic."""
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

