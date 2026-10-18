import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist


class LKF_2D_Eggbox:
    """Likelihood function: 2D eggbox-shaped multimodal surface.

    Prior: Uniform(0, 10*pi) in each dimension
    Analytical log-evidence: logZ ~= 235.856

    Reference: Feroz et al. (2009), MNRAS, 398(4):1601-1614
    """

    Ndim = 2
    logz_ref = 235.856

    def __init__(self):
        self.Pdis = Nataf([makedist("Uniform", lower=0, upper=10 * np.pi)] * 2, np.eye(2))

    def EvlLKF(self, theta):
        return (2 + np.cos(theta[0] / 2) * np.cos(theta[1] / 2)) ** 5
