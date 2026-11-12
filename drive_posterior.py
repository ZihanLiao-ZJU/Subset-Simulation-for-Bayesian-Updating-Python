"""Edit the settings below, then run this script from the repository root."""

import numpy as np

from sus_bayes.mcmc_samplers import CW_aMH, aCS, HMC, ES

from sus_bayes import (LKF_2D_Eggbox, LKF_nD_GaussianLogGamma, LKF_nD_GaussianShells,  # noqa: F401
                       SeMCS, NumSim, SuS_Bay_Nataf, aCS)
from sus_bayes.support_functions import SamPlt

# Sampler selection: CW_aMH, aCS, HMC or ES.
SamplerClass = aCS
PlotPosterior = False

# %% Initialization and settings
Nsam = 1000         # number of samples per iteration
Nrun = 5    # number of repeated runs
Ndim = 2            # dimensionality of the parameter space

# %% Define likelihood function (choose one)
# ----------------------------------------------------------------------
Tfun = LKF_nD_GaussianShells(Ndim)       # multimodal shell-shaped posterior
# Tfun = LKF_2D_Eggbox()                 # eggbox test function (2D only)
# Tfun = LKF_nD_GaussianLogGamma(Ndim)   # skewed likelihood test case
# ----------------------------------------------------------------------

# %% Construct sampler and subset simulation framework
IntBay = SuS_Bay_Nataf(Tfun)
sampler = SamplerClass(IntBay)
ss = SeMCS(sampler)
ss.NumSam = Nsam

# %% Run simulation
Ncal = np.zeros(Nrun)            # likelihood evaluations per run
Npos = np.zeros(Nrun)            # number of posterior samples
Ness = np.zeros(Nrun)            # effective sample size (posterior)

for irun in range(Nrun):
    out_ss = ss.RunIte()
    Ncal[irun] = np.sum(out_ss["Ncal"])

    out_ps = NumSim(out_ss)
    x_pos, y_pos = out_ps.PosSim()[:2]            # posterior samples (U-space)
    theta_pos = IntBay.LKF.Pdis.U2X(x_pos)        # parameter space

    Npos[irun] = x_pos.shape[1]

    c_z_hat, c_z = out_ps.EvlVar()                # estimated COVs
    Ness[irun] = (c_z / c_z_hat) ** 2 * Npos[irun]

# %% Summary statistics
Ness_rat = Ness / Ncal                            # ESS per likelihood evaluation
Ness_rat_mu = np.mean(Ness_rat)

print("================================================")
print(f"SuS posterior sampling over {Nrun} runs")
print(f"  Npos (mean)        : {np.mean(Npos):.1f}")
print(f"  Ness (mean)        : {np.mean(Ness):.1f}")
print(f"  Ness_rat_mu        : {Ness_rat_mu:.4f}")
print("================================================")

if PlotPosterior:
    SamPlt(theta_pos)
