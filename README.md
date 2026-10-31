# Subset Simulation for Bayesian Updating (Python)

Python implementation of subset simulation (SuS) for Bayesian evidence estimation and posterior sampling, using a Nataf transformation to work in standard normal space. This is a Python port of the MATLAB code in [Subset-Simulation-for-Bayesian-Updating](https://github.com/ZihanLiao-ZJU/Subset-Simulation-for-Bayesian-Updating), which accompanies the paper:

> **From likelihood to limit state: A reliability-inspired framework for Bayesian evidence estimation and high-dimensional sampling.**
> *Mechanical Systems and Signal Processing*, 237 (2025) 112969. doi: [10.1016/j.ymssp.2025.112969](https://doi.org/10.1016/j.ymssp.2025.112969)

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Benchmark likelihoods](#benchmark-likelihoods)
- [Custom likelihood](#custom-likelihood)
- [Outputs](#outputs)
- [Tests](#tests)
- [Implementation notes](#implementation-notes)
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

```
.
├── drive_evidence.py                    # driver: log-evidence estimation over repeated runs
├── drive_posterior.py                   # driver: posterior sampling and effective sample size
├── validate.py                          # recorded-reference and statistical checks
├── sus_bayes/
│   ├── SuS_Bay_Nataf.py                 # core: subset simulation for Bayesian updating (levels, seeds, convergence)
│   ├── main.py                          # sequential sampling loop (MATLAB class `main`, here `Main`)
│   ├── NumSim.py                        # post-processing: evidence, posterior resampling, CoV estimates
│   ├── Nataf.py                         # Nataf transformation between physical and standard normal space
│   ├── distributions.py                 # `makedist`: MATLAB-style distribution objects on top of scipy.stats
│   ├── mcmc_samplers/aCS.py             # adaptive conditional sampling
│   ├── likelihood_functions/            # benchmark likelihoods
│   ├── support_functions/               # weights, CoV correction, seed selection, plotting and other helpers
│   ├── operations_on_logarithmic_values/  # one .py file per MATLAB log function
│   └── log_operations.py                # backward-compatible imports
├── validation/                          # recorded fixtures used by validate.py
├── LICENSE
└── README.md
```

The logarithmic helpers use matching MATLAB/Python filenames: `logsum`,
`logmean`, `logplus`, `logminus`, `logmultiply`, `logcumsum`, and `logstd`.
Their Python implementations are in `operations_on_logarithmic_values/`;
existing imports from `log_operations` remain supported.


## Quick start

Run one of the drivers from the repository root:

```bash
python drive_evidence.py     # Bayesian evidence estimation
python drive_posterior.py    # posterior sampling
```

Both drivers use 1000 samples per level. `drive_evidence.py` runs 500 repetitions with `Ndim = 5`, and `drive_posterior.py` runs 5 repetitions with `Ndim = 2`; `--nrun` changes the number of runs and `--seed` fixes the random seed. To change the problem, edit `Ndim` and the `Tfun = ...` line in the *Define likelihood function* section of the driver.

The same steps from Python code:

```python
from sus_bayes import LKF_nD_GaussianShells, SuS_Bay_Nataf, aCS, Main, NumSim

Tfun = LKF_nD_GaussianShells(5)
IntBay = SuS_Bay_Nataf(Tfun)
ss = Main(aCS(IntBay))
ss.NumSam = 1000

out_ss = ss.RunIte()           # one subset simulation run
out_ps = NumSim(out_ss)        # post-processing
print(out_ps.Z)                # log-evidence
x_pos = out_ps.PosSim()[0]     # posterior samples in standard normal space
theta_pos = Tfun.Pdis.U2X(x_pos)
```

## Benchmark likelihoods

| File | Description | Dimension | Reference log-evidence |
|---|---|---|---|
| `LKF_nD_GaussianShells.py` | two Gaussian shells (multimodal), uniform prior on [−6, 6]ⁿ | n | −1.75 (n = 2), −5.67 (n = 5), −14.59 (n = 10), −36.09 (n = 20), −60.13 (n = 30) |
| `LKF_nD_GaussianLogGamma.py` | Gaussian + LogGamma mixture (skewed), uniform prior on [−30, 30]ⁿ | n | −8.19 (n = 2), −20.47 (n = 5), −40.94 (n = 10), −81.89 (n = 20), −122.82 (n = 30) |
| `LKF_2D_Eggbox.py` | eggbox (multimodal), uniform prior on [0, 10π]² | 2 | 235.856 |

Each class also stores its reference value as `logz_ref`. To solve your own problem, see [Custom likelihood](#custom-likelihood).

## Custom likelihood

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

The post-processing object `out_ps = NumSim(out_ss)` provides both outputs from a single subset simulation run:

- **Evidence:** `out_ps.Z` is $\log \widehat{Z}$, the natural logarithm of the evidence estimate, despite the property name `Z`. `np.exp(out_ps.Z)` converts it to $\widehat{Z}$ when representable; keeping the log value avoids underflow or overflow.
- **Posterior samples:** `x_pos = out_ps.PosSim()[0]` returns samples in standard normal space. `theta_pos = Tfun.Pdis.U2X(x_pos)` maps them to the physical parameter space, with one posterior sample per column.

During each run, the progress of every level is printed (sample size, number of chains, samples per chain, likelihood evaluations); set `ss.Verbose = False` to silence it. `Main.RunIte()` returns a dictionary with the samples `x`, function values `y` and level parameters `g` of every level, and the drivers print and keep the following:

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

## Tests

Run the unit tests and recorded-reference checks from the repository root:

```bash
python -m unittest discover -s tests -v
python validate.py --replay
```

The replay command requires all six recording/result pairs in `validation/reference/`.

## Implementation notes

- `Main` (`main` is an alias) returns run data as a dictionary.
- Arrays use the `Ndim × Nc × Ns` layout and column-major flattening. Python indices start at 0.
- Random draws use NumPy through `sus_bayes._utils`; `np.random.seed` sets the random seed.
- Frozen SciPy distribution wrappers use their inverse CDF. Parameterized Normal and Lognormal distributions use direct transforms.
- Parameterized GEV transforms clip CDF values to `[1e-15, 1 - 1e-15]`.
- Nataf correlation conversion supports positive, negative and zero target correlations. Numerical quadrature accuracy and feasible marginal correlations should be checked for custom distributions or extreme correlations.
- Signed-log arithmetic does not support slices containing only negative terms, including all-negative cumulative prefixes. The unit suite records this case as an expected failure.

## Citation

If you use this code in your research, please cite:

> Liao, Z., Li, B., & Wan, H.-P. (2025). From likelihood to limit state: A reliability-inspired framework for Bayesian evidence estimation and high-dimensional sampling. *Mechanical Systems and Signal Processing*, 237, 112969. https://doi.org/10.1016/j.ymssp.2025.112969

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
