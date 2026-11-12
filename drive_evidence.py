"""Edit the settings below, then run this script from the repository root."""

import numpy as np

from sus_bayes.mcmc_samplers import CW_aMH, aCS, HMC, ES

from sus_bayes import (LKF_2D_Eggbox, LKF_nD_GaussianLogGamma, LKF_nD_GaussianShells,  # noqa: F401
                       SeMCS, NumSim, SuS_Bay_Nataf, aCS)

# Sampler selection: CW_aMH, aCS, HMC or ES.
SamplerClass = aCS

# %% Initialization and settings
Nsam = 1000         # number of samples per iteration
Nrun = 500    # number of repeated runs
Ndim = 5            # dimensionality of the parameter space

# %% Define likelihood function (choose one)
# ----------------------------------------------------------------------
Tfun = LKF_nD_GaussianShells(Ndim)       # multimodal shell-shaped posterior
# Tfun = LKF_2D_Eggbox()                 # eggbox test function (2D only)
# Tfun = LKF_nD_GaussianLogGamma(Ndim)   # skewed likelihood test case
# ----------------------------------------------------------------------

# %% Construct sampler and subset simulation framework
# ----------------------------------------------------------------------
IntBay = SuS_Bay_Nataf(Tfun)     # Bayesian model with Nataf transformation
sampler = SamplerClass(IntBay)            # adaptive MCMC sampler
ss = SeMCS(sampler)               # main subset simulation object
ss.NumSam = Nsam                 # sample size per level
# ----------------------------------------------------------------------

# %% Run simulations
logz = np.zeros(Nrun)            # log-evidence per run
Ncal = np.zeros(Nrun)            # likelihood evaluations per run

for irun in range(Nrun):
    out_ss = ss.RunIte()                   # one subset simulation run
    Ncal[irun] = np.sum(out_ss["Ncal"])    # total calls to the likelihood
    out_ps = NumSim(out_ss)                # post-processing
    logz[irun] = out_ps.Z                  # estimated log-evidence

# %% Summary statistics
logz_mu = np.mean(logz)                    # mean log-evidence
logz_sig = np.std(logz, ddof=1) if Nrun > 1 else 0.0   # std of log-evidence
cv_logz = logz_sig / logz_mu               # coefficient of variation
Ncal_mu = np.mean(Ncal)                    # mean likelihood calls per run

print("================================================")
print(f"SuS log-evidence over {Nrun} runs")
print(f"  logz_mu  : {logz_mu:.4f}")
print(f"  logz_sig : {logz_sig:.4f}")
print(f"  cv_logz  : {cv_logz:.4f}")
print(f"  Ncal_mu  : {Ncal_mu:.1f}")
if np.isfinite(getattr(Tfun, "logz_ref", np.nan)):
    print(f"  reference: {Tfun.logz_ref:.4f}")
print("================================================")
