import numpy as np

from ..log_operations import logmean, logminus, logsum

_EPS = np.finfo(float).eps


def CorMod(f):
    """Coefficient of variation (COV) of weighted samples with a correction
    for the autocorrelation of Markov chains.

    INPUT
        f       : log-weight matrix [Nc x Ns] (Nc chains, Ns samples per chain)
    OUTPUT
        c_f_hat : corrected coefficient of variation
        c_f     : standard coefficient of variation of exp(f)
        gamma   : correction factor
    """
    f = np.asarray(f)
    Nc, Ns = f.shape
    N = Nc * Ns

    # Step 1: raw coefficient of variation c_f
    f_bar = logmean(f, None)                                   # log(mean(exp(f)))
    df = logminus(f, f_bar)                                    # log(exp(f) - mean)
    f_var = logsum(2 * np.real(df), None) - np.log(N - 1)      # log(var)
    if np.isinf(f_bar) or np.isnan(f_bar) or np.isnan(f_var):
        c_f = 0.0
    else:
        c_f = float(np.real(np.exp((f_var - np.log(N)) / 2 - f_bar)))

    # Step 2: corrected coefficient of variation c_f_hat
    if c_f < _EPS or Ns == 1:
        gamma = np.sqrt(Ns)
    else:
        # lag-1 autocorrelation (within chain)
        f1 = f[:, :Ns - 1]
        f2 = f[:, 1:Ns]
        f1f2_mu = logmean(f1 + f2, 0)
        df1f2_mu = logmean(logminus(f1f2_mu, 2 * f_bar), None)
        rhos = min(abs(np.exp(df1f2_mu - f_var)), 1)
        if rhos == 1:
            gamma = Ns
        else:
            gamma = 1 + 2 * rhos / (1 - rhos) ** 2 * (1 - rhos - 1 / Ns * (1 - rhos ** Ns))

    c_f_hat = np.sqrt(gamma) * c_f
    return c_f_hat, c_f, gamma
