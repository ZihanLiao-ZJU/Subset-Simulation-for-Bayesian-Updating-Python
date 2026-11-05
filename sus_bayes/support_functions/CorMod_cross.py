import numpy as np

from ..log_operations import logmean, logminus, logsum


def CorMod_cross(F, I):
    """Correlation and correction factor between two correlated log-weights.

    INPUT
        F : log-weights of f(x), [Nc x Ns]
        I : log-weights of i(x), [Nc x Ns]
    OUTPUT
        rho_fi : marginal correlation coefficient between F and I (bounded by 1)
        gamma  : correction factor for intra-chain autocorrelation
    """
    F = np.asarray(F)
    I = np.asarray(I)
    Nc, Ns = F.shape
    N = Nc * Ns

    h = logmean(F, None)
    p = logmean(I, None)
    dF = logminus(F, h)
    dI = logminus(I, p)
    F_var = logsum(2 * np.real(dF), None) - np.log(N - 1)
    I_var = logsum(2 * np.real(dI), None) - np.log(N - 1)

    FI_mu = logmean(F + I, 0)
    dFI_mu = logmean(logminus(FI_mu, h + p), None)
    with np.errstate(invalid="ignore"):
        rho_fi = min(abs(np.exp(dFI_mu - (F_var + I_var) / 2)), 1)

    if Ns == 1:
        rhos_fi = 0
    else:
        F1, F2 = F[:, :Ns - 1], F[:, 1:Ns]
        I1, I2 = I[:, :Ns - 1], I[:, 1:Ns]
        F1I2_mu = logmean(F1 + I2, 0)
        I1F2_mu = logmean(I1 + F2, 0)
        dF1I2_mu = logmean(logminus(F1I2_mu, h + p), None)
        dI1F2_mu = logmean(logminus(I1F2_mu, h + p), None)
        with np.errstate(invalid="ignore"):
            rhos1 = min(abs(np.exp(dF1I2_mu - (F_var + I_var) / 2)), 1)
            rhos2 = min(abs(np.exp(dI1F2_mu - (F_var + I_var) / 2)), 1)
        rhos_fi = (rhos1 + rhos2) / 2

    if rhos_fi == 1:
        gamma = Ns
    else:
        gamma = 1 + 2 * rhos_fi / (1 - rhos_fi) ** 2 * (1 - rhos_fi - (1 / Ns) * (1 - rhos_fi ** Ns))
    return rho_fi, gamma
