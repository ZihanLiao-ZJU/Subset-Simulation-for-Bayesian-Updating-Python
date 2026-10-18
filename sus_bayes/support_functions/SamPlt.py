import numpy as np


def SamPlt(x, show=True):
    """Scatter plot of the first two coordinates of samples (array or list of levels)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    levels = x if isinstance(x, (list, tuple)) else [x]
    for theta in levels:
        theta = np.asarray(theta).reshape(np.shape(theta)[0], -1, order="F")
        ax.scatter(theta[0], theta[1], s=4, alpha=0.5, color="0.1" if len(levels) == 1 else None)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    if show:
        plt.show()
    return fig
