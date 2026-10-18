def VabSrk(Nl, *args):
    """Shrink each input to its first ``Nl`` entries along the first axis."""
    out = tuple(v[:Nl] for v in args)
    return out[0] if len(out) == 1 else out
