"""Compatibility imports for the original logarithmic-arithmetic API."""
from .operations_on_logarithmic_values import (
    logsum, logmean, logplus, logminus, logmultiply, logcumsum, logstd,
    _EPS, _isreal, _real_if_possible,
)
__all__ = ["logsum", "logmean", "logplus", "logminus", "logmultiply", "logcumsum", "logstd"]
