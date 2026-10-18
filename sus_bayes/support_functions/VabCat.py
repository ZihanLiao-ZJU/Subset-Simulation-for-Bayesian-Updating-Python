import numpy as np


def VabCat(axis, *args):
    """Concatenate pairs of variables along ``axis``: VabCat(axis, a1, b1, a2, b2, ...)."""
    return tuple(np.concatenate([args[2 * i], args[2 * i + 1]], axis=axis) for i in range(len(args) // 2))
