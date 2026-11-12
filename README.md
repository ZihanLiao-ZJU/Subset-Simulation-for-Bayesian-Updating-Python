# Subset Simulation for Bayesian Updating (Python)

Python implementation of subset simulation (SuS) for Bayesian evidence estimation and posterior sampling, using a Nataf transformation to work in standard normal space. This is a line-by-line port of the MATLAB code in [Subset-Simulation-for-Bayesian-Updating](https://github.com/ZihanLiao-ZJU/Subset-Simulation-for-Bayesian-Updating), which accompanies the paper:

> Liao, Z., Li, B., & Wan, H.-P. (2025). From likelihood to limit state: A reliability-inspired framework for Bayesian evidence estimation and high-dimensional sampling. *Mechanical Systems and Signal Processing, 237*, Article 112969. [https://doi.org/10.1016/j.ymssp.2025.112969](https://doi.org/10.1016/j.ymssp.2025.112969)

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Built-in examples](#built-in-examples)
- [Using a custom likelihood](#using-a-custom-likelihood)
- [Outputs](#outputs)
- [Differences from the MATLAB version](#differences-from-the-matlab-version)
- [Citation](#citation)
- [License](#license)

## Overview

Given observed data $D$ and uncertain parameters $\theta$, Bayesian updating follows Bayes' theorem:

$$
p(\theta \mid D)
= \frac{p(D \mid \theta)\,p(\theta)}{p(D)},
\qquad
Z = p(D) = \int p(D \mid \theta)\,p(\theta)\,\mathrm{d}\theta.
$$

Here, $p(\theta)$ is the **prior distribution**, $p(D \mid \theta)$ is the **likelihood**, $p(\theta \mid D)$ is the **posterior distribution**, and $Z = p(D)$ is the **Bayesian evidence (marginal likelihood)** that normalizes the posterior.

The toolbox provides two main outputs:

1. an **evidence (marginal likelihood) estimate** $\widehat{Z}$, computed and reported in the natural-log domain as $\log \widehat{Z}$ for numerical stability;
2. **posterior distribution samples**, which give a Monte Carlo approximation to $p(\theta \mid D)$.

The evidence is an estimate, not an exact integral. Posterior samples are obtained by resampling MCMC-generated samples; they may be repeated or dependent and should not be treated as exact independent draws.

The Bayesian updating problem is reformulated in the spirit of a structural reliability problem: the likelihood plays the role of a limit-state function, and subset simulation generates samples from a sequence of nested likelihood levels, from the prior toward the high-likelihood region. Samples at each level are generated with adaptive conditional sampling (aCS) MCMC. The samples from all levels are then post-processed to

- estimate the log-evidence, together with its coefficient of variation (CoV), with a correction for the correlation of Markov chain samples, and
- obtain posterior samples by resampling.

The class structure, method names and variable names follow the MATLAB code, so the two versions can be read side by side.

## Requirements

- Python 3.9 or later
- NumPy and SciPy
- Matplotlib (optional, only for plotting)

```bash
pip install -r requirements.txt
```

## Repository structure

```text
.
├── LICENSE
├── README.md
├── drive_evidence.py
├── drive_posterior.py
├── requirements.txt
└── sus_bayes/
    ├── Nataf.py
    ├── NumSim.py
    ├── SuS_Bay_Nataf.py
    ├── __init__.py
    ├── _utils.py
    ├── distributions.py
    ├── likelihood_functions/
    │   ├── LKF_2D_Eggbox.py
    │   ├── LKF_nD_GaussianLogGamma.py
    │   ├── LKF_nD_GaussianShells.py
    │   └── __init__.py
    ├── SeMCS.py
    ├── mcmc_samplers/
    │   ├── CW_aMH.py
    │   ├── ES.py
    │   ├── HMC.py
    │   ├── LICENSE
    │   ├── MCMCBase.py
    │   ├── __init__.py
    │   └── aCS.py
    ├── operations_on_logarithmic_values/
    │   └── __init__.py
    └── support_functions/
        ├── CorMod.py
        ├── CorMod_cross.py
        ├── DspRst.py
        ├── RS.py
        ├── ResRat.py
        ├── SamPlt.py
        ├── VabCmb.py
        ├── WgtEvd.py
        ├── WgtPos.py
        ├── __init__.py
        └── logGauss.py
```

## Quick start

Edit `Tfun`, `Ndim`, `Nsam`, `Nrun` and `SamplerClass` inside `drive_evidence.py` or `drive_posterior.py`. Select `SamplerClass = aCS` (the default), or choose another supplied sampler: `CW_aMH`, `aCS`, `HMC` or `ES`.

Both scripts use `LKF_nD_GaussianShells` and 1000 samples per level. Evidence estimation defaults to dimension 5 and 500 runs; posterior sampling defaults to dimension 2 and 5 runs. For a single run, set `Nrun = 1` inside the script.

Run the chosen script from the repository root:

```bash
python drive_evidence.py
python drive_posterior.py
```

Use `drive_evidence` for evidence estimation or `drive_posterior` for posterior sampling.

Use multiple runs to estimate statistics across independent runs.

## Built-in examples

| File | Description | Dimension | Reference log-evidence |
|---|---|---|---|
| `LKF_nD_GaussianShells.py` | two Gaussian shells (multimodal), uniform prior on [−6, 6]ⁿ | n | −1.75 (n = 2), −5.67 (n = 5), −14.59 (n = 10), −36.09 (n = 20), −60.13 (n = 30) |
| `LKF_nD_GaussianLogGamma.py` | Gaussian + LogGamma mixture (skewed), uniform prior on [−30, 30]ⁿ | n | −8.19 (n = 2), −20.47 (n = 5), −40.94 (n = 10), −81.89 (n = 20), −122.82 (n = 30) |
| `LKF_2D_Eggbox.py` | eggbox (multimodal), uniform prior on [0, 10π]² | 2 | 235.856 |

Each class also stores its reference value as `logz_ref`. To solve your own problem, see [Custom likelihood](#using-a-custom-likelihood).

## Using a custom likelihood

Any user-defined Bayesian updating problem can be solved by writing a likelihood class with the same interface as the files in `sus_bayes/likelihood_functions/` and passing it to the driver in place of the benchmark. The algorithm only accesses the following members:

| Member | Type | Description |
|---|---|---|
| `Ndim` | int | number of uncertain parameters |
| `Pdis` | `Nataf` object | prior distribution, built as `Nataf(Marg_X, Rho_X)` from a list of distribution objects (`makedist`) and their correlation matrix |
| `EvlLKF(self, theta)` | method | log-likelihood of the samples `theta` (`Ndim × Nsam` array, in the physical parameter space); must return an array of length `Nsam` |

The sampler works in standard normal space and maps samples to the parameter space through `Pdis.U2X` before calling `EvlLKF`, so the likelihood is written directly in terms of the physical parameters. `EvlLKF` is called with many samples at once and must return the **logarithm** of the likelihood. Retain the likelihood normalization constants when estimating absolute evidence: omitting a parameter-independent factor leaves the normalized posterior unchanged but rescales the evidence.

`makedist` takes MATLAB's distribution names and parameter names (`"Normal"` with `mu`, `sigma`; `"Lognormal"`; `"Uniform"` with `lower`, `upper`; `"GeneralizedExtremeValue"`; `"Gamma"`; `"Weibull"`; ...). Any frozen `scipy.stats` distribution can also be used through `Distribution("Name", frozen)`.

Minimal example: two parameters with independent standard normal priors and a Gaussian likelihood for two noisy observations.

```python
import numpy as np
from sus_bayes import Nataf, makedist


class MyLikelihood:
    def __init__(self):
        self.Ndim = 2
        self.Pdis = Nataf([makedist("Normal", mu=0, sigma=1)] * 2, np.eye(2))

    def EvlLKF(self, theta):
        # theta : [Ndim x Nsam] parameter samples
        # logL  : [Nsam] log-likelihood
        y_obs = np.array([[1.0], [-0.5]])   # observed data
        sigma = 0.2                         # measurement noise standard deviation
        return -0.5 * np.sum((theta - y_obs) ** 2, axis=0) / sigma ** 2 - y_obs.size * np.log(np.sqrt(2 * np.pi) * sigma)
```

Then, in `drive_evidence.py` or `drive_posterior.py`, replace the benchmark with your class:

```python
Tfun = MyLikelihood()
```

The rest of the driver is unchanged.

## Outputs

The raw sampling history `g` contains `Nite` sampled states followed by one final candidate state. `Nsze` records the chain and step counts at each sampled level.

The post-processing object `out_ps = NumSim(out_ss)` provides both outputs from a single subset simulation run:

- **Evidence:** `out_ps.Z` is $\log \widehat{Z}$, the natural logarithm of the evidence estimate, despite the property name `Z`. `np.exp(out_ps.Z)` converts it to $\widehat{Z}$ when representable; keeping the log value avoids underflow or overflow.
- **Posterior samples:** `x_pos = out_ps.PosSim()[0]` returns samples in standard normal space. `theta_pos = Tfun.Pdis.U2X(x_pos)` maps them to the physical parameter space, with one posterior sample per column.

During each run, the progress of every level is printed (sample size, number of chains, samples per chain, likelihood evaluations); set `ss.Verbose = False` to silence it. `SeMCS.RunIte()` returns a dictionary with the samples `x`, function values `y` and level parameters `g` of every level, and the drivers print and keep the following:

**`drive_evidence.py`**

| Variable | Meaning |
|---|---|
| `logz` | estimated log-evidence of each run |
| `logz_mu`, `logz_sig` | mean and standard deviation of the log-evidence |
| `cv_logz` | coefficient of variation of the log-evidence |
| `Ncal_mu` | average number of likelihood evaluations per run |

**`drive_posterior.py`**

| Variable | Meaning |
|---|---|
| `theta_pos` | posterior samples in the parameter space (last run) |
| `Npos` | number of posterior samples of each run |
| `Ness` | effective sample size of each run |
| `Ness_rat_mu` | average effective sample size per likelihood evaluation |

Signed-log arithmetic retains an inherited limitation: an all-negative signed-log slice, including a negative first cumulative prefix, is not handled correctly by `logsum`. Preserving complex `logcumsum` outputs retains mixed-sign prefixes but does not change that reduction algorithm.

## Differences from the MATLAB version

- The iteration class is `SeMCS` in both languages, and `RunIte` returns a dictionary instead of a struct.
- Arrays keep the MATLAB layout (`Ndim × Nc × Ns`, flattened in column-major order), so indices printed by the two versions agree; Python indices start at 0.
- Frozen wrappers such as `Distribution("Normal", scipy.stats.norm(...))` need not expose `mu`/`sigma` attributes: they use their generic `icdf` mapping. Parameterized `makedist` Normal/Lognormal objects retain direct transforms to avoid CDF saturation in extreme tails.
- Parameterized GEV transforms retain the Python port's existing CDF clipping to `[1e-15, 1 - 1e-15]` (and near-zero-shape limit). This stabilizes tails but is intentionally not identical to the MATLAB source's unclipped `icdf(normcdf(...))`; generic frozen wrappers use their own `icdf` instead.
- General non-Normal/Lognormal correlation conversion now searches negative latent correlations for negative target pairs and preserves exactly zero pairs. This corrects the MATLAB source's inherited nonnegative search interval, rather than reproducing it. Python supplies `GauIntPot`, which is absent from the public MATLAB repository. Numerical quadrature and marginal correlation feasibility remain limitations for arbitrary distributions/extreme correlations.
- `support_functions/ThdGen.m` is not ported: it is not called by the algorithm and depends on functions that are not part of the MATLAB repository.

## Citation

If you use this toolbox in your research, please cite:

> Liao, Z., Li, B., & Wan, H.-P. (2025). From likelihood to limit state: A reliability-inspired framework for Bayesian evidence estimation and high-dimensional sampling. *Mechanical Systems and Signal Processing, 237*, Article 112969. [https://doi.org/10.1016/j.ymssp.2025.112969](https://doi.org/10.1016/j.ymssp.2025.112969)

```bibtex
@article{liao2025likelihood,
  title   = {From likelihood to limit state: A reliability-inspired framework for {B}ayesian evidence estimation and high-dimensional sampling},
  author  = {Liao, Zihan and Li, Binbin and Wan, Hua-Ping},
  journal = {Mechanical Systems and Signal Processing},
  volume  = {237},
  pages   = {112969},
  year    = {2025},
  doi     = {10.1016/j.ymssp.2025.112969}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

