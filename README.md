# SuS Bayesian — unified SeMCS (python)

This update retains the tested numerical algorithms and uses the common SeMCS engine and common MCMC implementation. Method-specific state and policies remain in the method classes; state evolves in G. The common core is bundled byte for byte so installation does not depend on an unreleased reliability repository.

## Qualified entrypoint

The qualified profile is `closed_ties_unit_fallback_v1`, with estimator `source_native_evidence`. The existing drivers and optional policies remain available for compatibility; use `run_qualified` for the settings in this table. No default is silently presented as paper-qualified.

```bash
python -m pip install .
python run_qualified.py --case sus_bayes_eggbox2 --describe
# python run_qualified.py --case sus_bayes_eggbox2 --seed 1  # one actual run
```

## Source-paper comparison

Each row uses its declared independent repeat count. Elog is mean Ncal × [SD(log Z)/|printed source log Z reference|]², with sample SD (ddof=1). R=Elog/source Elog; the fixed acceptance limit is R≤1.10.

| Case | Per-level N | Repeats | Mean log Z | SD log Z | Mean Ncal | Elog | Paper Elog | R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sus_bayes_eggbox2 | 3801 | 100 | 235.82589 | 0.317731 | 20178.67 | 0.036618718 | 0.03724 | 0.9833168 |
| sus_bayes_loggamma2 | 5975 | 100 | -8.1904692 | 0.072004191 | 23915 | 1.8484944 | 1.686384 | 1.096129 |
| sus_bayes_loggamma5 | 7262 | 1000 | -20.470712 | 0.17051219 | 68645.818 | 4.763095 | 4.970112 | 0.95834762 |
| sus_bayes_shells5 | 10399 | 1000 | -5.6759665 | 0.064453742 | 51999 | 6.7193173 | 6.292 | 1.0679144 |

Source papers: [DOI 10.1016/j.ymssp.2025.112969](https://doi.org/10.1016/j.ymssp.2025.112969).

## Scope and limitations

- Qualification applies to the listed profile, estimator, targets and budgets; it does not cover every optional setting.
- No global finite-sample unbiasedness or universal posterior accuracy claim is made.
- The migration bias check uses100 same-seed native/shared pairs per case, not1000 paired runs; the1000 cohorts are independent precision studies.
- Paper efficiency points are treated as fixed; their Monte Carlo uncertainty and rounding are not included. Confidence intervals do not waive the1.10 point gate.
- Existing100-run cases are listed as100, never relabelled1000. Prior100 and1000 cohorts are not pooled.
- The closed-support/identical-seed scale fallback is an explicitly declared numerical variant, not a boundary rule claimed to have been specified by the source paper. The old author_latest default remains available and is not the qualified entrypoint.

See `VALIDATION_SUMMARY.json` for exact profile settings and source-report SHA bindings; `SEMCS_CORE_MANIFEST.json` binds every bundled core file. Existing licenses are retained, with the core notice in `LICENSE_SEMCS_CORE`. No private papers, per-run research outputs or local machine paths are bundled.

## Public regression compatibility

Public regression compatibility: generic arbitrary-chain coverage explicitly selects GroupMode=available (36 calls for12 chains and3 steps). A separate author_latest fixed-group test preserves the source five-group behavior (39 calls for the same shape), including its repeated last-chain visit. No model, engine, kernel or profile default changed.

The preserved public suite ran21 small synthetic/contract tests:19 ordinary passes,1 documented expected failure for an inherited all-negative-prefix signed-log limitation, and1 explicit skip of a legacy MATLAB trajectory replay not qualifying the closed-support variant. Fixture-corruption checks remain active. This is a repository compatibility check, not another benchmark or evidence-unbiasedness certificate.
