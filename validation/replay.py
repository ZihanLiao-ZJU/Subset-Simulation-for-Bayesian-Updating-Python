"""Replay a random stream recorded from Octave (see recshim/)."""
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

    def _pop(self, kind, n=None):
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
        return self._pop("randsample", int(k)).astype(int) - 1

    @property
    def done(self):
        return self.pos == len(self.entries)
