import numpy as np

from .._utils import flat


def _cmb(v):
    if not isinstance(v, (list, tuple)):
        return flat(v)
    cells = [np.asarray(c) for c in v]
    ndim = np.array([c.shape[0] if c.ndim > 0 and c.size else 0 for c in cells])
    keep = ndim != 0
    nd = ndim[keep]
    if nd.size and np.all(nd == nd[0]):
        return np.concatenate([flat(c)[:nd[0], :] for c, k in zip(cells, keep) if k], axis=1)
    return np.concatenate([c.ravel(order="F")[None, :] for c, k in zip(cells, keep) if k], axis=1)


def VabCmb(*args):
    """Flatten per-level cell arrays into one [rows, Nall] array each."""
    out = tuple(_cmb(v) for v in args)
    return out[0] if len(out) == 1 else out
