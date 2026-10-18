"""Small helpers reproducing MATLAB semantics that NumPy does differently."""
import numpy as np


def mround(x):
    """MATLAB round: halves are rounded away from zero."""
    x = np.asarray(x, dtype=float)
    r = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return r[()] if r.ndim == 0 else r


def flat(a):
    """MATLAB a(k,:) on an N-D array: [rows, everything else] in column-major order."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a.reshape(a.shape[0], -1, order="F")


def reshape_F(a, shape):
    """MATLAB reshape (column-major)."""
    return np.reshape(np.asarray(a), shape, order="F")


def tail_shape(a):
    """MATLAB size(a, 2:3) for an array of shape [rows, Nc] or [rows, Nc, Ns]."""
    s = np.shape(a)
    return (s[1], s[2] if len(s) > 2 else 1)


def nanmax(a):
    """MATLAB max(a, [], "all"): NaN values are ignored."""
    a = np.asarray(a, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return np.nan
    return np.nanmax(a)


# ---------------------------------------------------------------------------
# Random number generation
#
# All random numbers are drawn through the functions below, which mirror the
# MATLAB calls of the original code (randn, rand, randperm, randi, randsample).
# By default they use NumPy's global generator, so ``np.random.seed`` makes a
# run reproducible. ``set_rng`` installs another generator, for example one
# that replays a random stream recorded from MATLAB/Octave for a
# sample-by-sample comparison with the original implementation.
# ---------------------------------------------------------------------------
class NumpyRNG:
    """Default generator backed by numpy.random."""

    def randn(self, *shape):
        return np.random.randn(*shape)

    def rand(self, *shape):
        return np.random.rand(*shape)

    def randperm(self, n, k):
        return np.random.permutation(int(n))[:int(k)]

    def randi(self, n, k):
        return np.random.randint(int(n), size=int(k))

    def randsample(self, n, k, w):
        w = np.asarray(w, dtype=float).ravel(order="F")
        return np.random.choice(int(n), size=int(k), replace=True, p=w / np.sum(w))


_rng = NumpyRNG()


def set_rng(rng=None):
    """Install a random number generator (``None`` restores the NumPy default)."""
    global _rng
    _rng = NumpyRNG() if rng is None else rng


def randn(*shape):
    """MATLAB randn(shape)."""
    return _rng.randn(*shape)


def rand(*shape):
    """MATLAB rand(shape)."""
    return _rng.rand(*shape)


def randperm(n, k):
    """MATLAB randperm(n, k), 0-based."""
    return np.asarray(_rng.randperm(n, k), dtype=int)


def randi(n, k):
    """MATLAB randi(n, 1, k), 0-based."""
    return np.asarray(_rng.randi(n, k), dtype=int)


def randsample(n, k, weights):
    """MATLAB randsample(n, k, true, w): k draws with replacement, 0-based indices."""
    return np.asarray(_rng.randsample(n, k, weights), dtype=int)
