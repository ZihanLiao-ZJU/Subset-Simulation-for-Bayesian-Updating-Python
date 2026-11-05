"""Run one Bayesian updating example with subset simulation."""
import argparse
import json


EXAMPLES = {
    "sus_bayes_eggbox2": {"N": 3801, "dimension": 2},
    "sus_bayes_loggamma2": {"N": 5975, "dimension": 2},
    "sus_bayes_loggamma5": {"N": 7262, "dimension": 5},
    "sus_bayes_shells5": {"N": 10399, "dimension": 5},
}


def describe(case):
    if case not in EXAMPLES:
        raise ValueError("Unknown example: " + str(case))
    return dict(EXAMPLES[case], case=case, profile="closed_ties_unit_fallback_v1",
                estimator="source_native_evidence", sampler="aCS",
                conditional_probability=0.1, max_iterations=10000)


def construct(case):
    import sus_bayes as package
    cfg = describe(case)
    if case == "sus_bayes_eggbox2":
        target = package.LKF_2D_Eggbox()
    elif case == "sus_bayes_shells5":
        target = package.LKF_nD_GaussianShells(5)
    else:
        target = package.LKF_nD_GaussianLogGamma(cfg["dimension"])
    model = package.SuS_Bay_Nataf(target, method_profile=cfg["profile"])
    sampler = package.aCS(model)
    simulation = package.Main(sampler)
    simulation.NumSam = cfg["N"]
    simulation.MaxIte = cfg["max_iterations"]
    simulation.ConPrb = cfg["conditional_probability"]
    return cfg, model, sampler, simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(EXAMPLES), default="sus_bayes_eggbox2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--describe", action="store_true", help="Print settings without running the example")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(describe(args.case), indent=2))
        return
    import numpy as np
    import sus_bayes as package
    np.random.seed(args.seed)
    cfg, model, sampler, simulation = construct(args.case)
    raw = simulation.RunIte()
    post = package.NumSim(raw)
    print(json.dumps(dict(case=args.case, seed=args.seed, profile=cfg["profile"],
        estimator=cfg["estimator"], logZ=np.asarray(post.Z).tolist(),
        Ncal=int(np.sum(raw["Ncal"])))))


if __name__ == "__main__":
    main()
