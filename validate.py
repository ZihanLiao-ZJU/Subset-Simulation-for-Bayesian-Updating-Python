"""Validate the Python port against the original MATLAB implementation.

Part 1 - replay (exact): validation/reference/ holds random number streams and
results recorded by running the ORIGINAL MATLAB code once per benchmark
(validation/matlab/record_reference.m). The Python code is driven by the same
random numbers and must reproduce the MATLAB levels, thresholds, evidence,
CoV estimates and posterior samples to round-off.

Part 2 - statistics: repeated Python runs on the bundled benchmarks; the mean
log-evidence is compared with the analytical reference value.

    python validate.py              # both parts
    python validate.py --replay     # part 1 only
    python validate.py --nrun 100   # part 2 with 100 runs per benchmark
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "validation"))

import sus_bayes as sb  # noqa: E402
from sus_bayes import _utils  # noqa: E402
from replay import ReplayRNG  # noqa: E402


def make_lkf(case, ndim):
    return {"shells": lambda: sb.LKF_nD_GaussianShells(ndim),
            "loggamma": lambda: sb.LKF_nD_GaussianLogGamma(ndim),
            "eggbox": lambda: sb.LKF_2D_Eggbox()}[case]()


def run_once(Tfun, nsam=1000):
    ss = sb.Main(sb.aCS(sb.SuS_Bay_Nataf(Tfun)))
    ss.NumSam = nsam
    ss.Verbose = False
    out = ss.RunIte()
    ps = sb.NumSim(out)
    return out, ps


def replay():
    print("Part 1: replay of MATLAB random streams (exact comparison)")
    print(f"{'case':14s} {'levels':>7s} {'logZ (Python)':>15s} {'logZ (MATLAB)':>15s} {'max|dG|':>9s} "
          f"{'max|dx_pos|':>11s} {'max|dCoV|':>9s}  result")
    ok_all = True
    for fn in sorted(glob.glob(os.path.join(HERE, "validation", "reference", "rec_*.mat"))):
        tag = os.path.basename(fn)[4:-4]
        case, ndim, _ = tag.split("_")
        ndim = int(ndim)
        rng = ReplayRNG(fn)
        _utils.set_rng(rng)
        try:
            out, ps = run_once(make_lkf(case, ndim))
            x_pos = ps.PosSim()[0]
            cv = np.array(ps.EvlVar())
        finally:
            _utils.set_rng(None)
        r = loadmat(os.path.join(HERE, "validation", "reference", f"res_{tag}.mat"))
        G = np.vstack(out["g"])
        with np.errstate(invalid="ignore"):
            dG = np.max(np.abs(np.where(np.isfinite(G), G - r["G"], 0)))
        same_inf = np.array_equal(np.isinf(G), np.isinf(r["G"]))
        dx = np.max(np.abs(x_pos - r["x_pos"])) if x_pos.shape == r["x_pos"].shape else np.inf
        dcv = np.max(np.abs(cv - [r["c_z_hat"].item(), r["c_z"].item()]))
        ok = (out["Nite"] == r["Nite"].item() and same_inf and dG < 1e-8 and dx < 1e-8 and dcv < 1e-8
              and abs(ps.Z - r["Z"].item()) < 1e-8 and np.array_equal(out["Ncal"], r["Ncal"].ravel()) and rng.done)
        ok_all &= ok
        print(f"{case + ' n=' + str(ndim):14s} {out['Nite']:7d} {ps.Z:15.8f} {r['Z'].item():15.8f} {dG:9.1e} "
              f"{dx:11.1e} {dcv:9.1e}  {'PASS' if ok else 'FAIL'}")
    return ok_all


def statistics(nrun, seed):
    print(f"\nPart 2: {nrun} independent Python runs per benchmark (1000 samples per level)")
    print(f"{'case':16s} {'mean logZ':>10s} {'std':>7s} {'reference':>10s} {'|bias|/s.e.':>11s} {'Ncal':>8s}  result")
    np.random.seed(seed)
    ok_all = True
    for case, ndim in [("shells", 2), ("shells", 5), ("loggamma", 2), ("loggamma", 5), ("eggbox", 2)]:
        T = make_lkf(case, ndim)
        z = np.zeros(nrun)
        ncal = np.zeros(nrun)
        for i in range(nrun):
            out, ps = run_once(T)
            z[i] = ps.Z
            ncal[i] = np.sum(out["Ncal"])
        se = np.std(z, ddof=1) / np.sqrt(nrun)
        t = abs(np.mean(z) - T.logz_ref) / se
        # pass: within 4 standard errors, or relative bias of logZ below 1 %
        ok = t < 4 or abs(np.mean(z) - T.logz_ref) < 0.01 * abs(T.logz_ref)
        ok_all &= ok
        print(f"{case + ' n=' + str(ndim):16s} {np.mean(z):10.3f} {np.std(z, ddof=1):7.3f} {T.logz_ref:10.3f} "
              f"{t:11.2f} {np.mean(ncal):8.0f}  {'PASS' if ok else 'CHECK'}")
    return ok_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="store_true", help="only run the replay comparison")
    ap.add_argument("--nrun", type=int, default=50, help="runs per benchmark in part 2 (default 50)")
    ap.add_argument("--seed", type=int, default=2025)
    a = ap.parse_args()
    ok = replay()
    if not a.replay:
        ok &= statistics(a.nrun, a.seed)
    print("\nAll checks passed." if ok else "\nSome checks did not pass.")
    sys.exit(0 if ok else 1)
