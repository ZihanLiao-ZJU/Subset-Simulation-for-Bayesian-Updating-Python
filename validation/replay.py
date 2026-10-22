"""Replay recorded Octave draws and indices (not weighted RNG equivalence).

Legacy recordings contain selected indices rather than the uniform draws
and weights that selected them. Validate their support and expose the actual
probabilities supplied by Python so validate.py can check them separately.
"""
import numpy as np
from scipy.io import loadmat

KIND = {1: "randn", 2: "rand", 3: "randperm", 4: "randi", 5: "randsample"}


class ReplayRNG:
    def __init__(self, fname):
        d = loadmat(fname)
        kinds = d["kinds"].ravel().astype(int)
        lens = d["lens"].ravel().astype(int)
        vals = d["vals"].ravel()
        offs = np.concatenate([[0], np.cumsum(lens)])
        self.entries = [(KIND[k], vals[offs[i]:offs[i + 1]]) for i, k in enumerate(kinds)]
        self.pos = 0
        self.last_probabilities = None

    def _pop(self, kind, n=None):
        if self.pos >= len(self.entries):
            raise RuntimeError(f"recorded stream exhausted while requesting {kind}")
        k, v = self.entries[self.pos]
        if k != kind or (n is not None and v.size != n):
            raise RuntimeError(f"replay mismatch at {self.pos}: expected {kind}({n}), recorded {k}({v.size})")
        self.pos += 1
        return v

    def randn(self, *shape):
        return self._pop("randn", int(np.prod(shape))).reshape(shape, order="F")

    def rand(self, *shape):
        return self._pop("rand", int(np.prod(shape))).reshape(shape, order="F")

    def randperm(self, n, k):
        return self._pop("randperm", int(k)).astype(int) - 1

    def randi(self, n, k):
        return self._pop("randi", int(k)).astype(int) - 1

    def randsample(self, n, k, w):
        w = np.asarray(w, dtype=float).ravel(order="F")
        total = np.sum(w)
        if (w.size != n or not np.all(np.isfinite(w)) or np.any(w < 0)
                or not np.isfinite(total) or total <= 0):
            raise ValueError("resampling weights must be finite, nonnegative, nonzero and match n")
        self.last_probabilities = w / total
        recorded = self._pop("randsample", int(k))
        if (not np.all(np.isfinite(recorded)) or np.any(recorded != np.floor(recorded))
                or np.any(recorded < 1) or np.any(recorded > n)):
            raise ValueError("recorded resampling indices are outside the requested population")
        indices = recorded.astype(int) - 1
        if np.any(w[indices] <= 0):
            raise ValueError("recorded resampling index has zero probability under supplied weights")
        return indices

    @property
    def done(self):
        return self.pos == len(self.entries)
