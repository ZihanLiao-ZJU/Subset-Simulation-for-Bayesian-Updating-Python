# Subset Simulation for Bayesian Updating

A Python toolbox for Bayesian evidence estimation and posterior sampling
with subset simulation.

## Installation

Requires Python 3.10 or later, NumPy and SciPy.

```bash
python -m pip install .
```

The shared `semcs_core` package is bundled with this repository.
Install Matplotlib separately when using the plotting helpers.

## Example

```bash
python run_example.py --case sus_bayes_eggbox2 --describe
python run_example.py --case sus_bayes_eggbox2 --seed 1
```

Other examples are `sus_bayes_loggamma2`, `sus_bayes_loggamma5` and
`sus_bayes_shells5`. The script contains their settings and prints the
log evidence and number of likelihood evaluations for one run.

## Architecture

- `semcs_core/`: the common SeMCS iteration engine and MCMC samplers.
- `sus_bayes/`: intermediate distributions, seed selection, stopping rules
  and Bayesian postprocessing.
- `sus_bayes/likelihood_functions/`: example likelihoods and priors.
- `run_example.py`: a complete example; the original evidence and posterior
  drivers remain available for customized runs.

Configuration and evolving algorithm state are stored in `G`. Each method
owns its intermediate distribution and iteration policy; the common engine
handles the iteration and calls the selected sampler.

## Reference

[From likelihood to limit state](https://doi.org/10.1016/j.ymssp.2025.112969).
The repository and bundled core licenses are included in `LICENSE` and
`LICENSE_SEMCS_CORE`.
