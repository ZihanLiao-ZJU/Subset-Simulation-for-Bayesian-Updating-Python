"""Backward-compatible imports for logarithmic arithmetic.

Implementations are in operations_on_logarithmic_values, with one module
per corresponding MATLAB function.
"""
from .operations_on_logarithmic_values import logsum, logmean, logplus, logminus, logmultiply, logcumsum, logstd
from .operations_on_logarithmic_values._helpers import _EPS, _isreal, _real_if_possible

__all__ = ['logsum', 'logmean', 'logplus', 'logminus', 'logmultiply', 'logcumsum', 'logstd']
