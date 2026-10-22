"""Validate the Python port against the original MATLAB implementation.

Part 1 - replay (exact): validation/reference/ holds random number streams and
results recorded by running the ORIGINAL MATLAB code once per benchmark
(validation/matlab/record_reference.m). The Python code is driven by the same
random numbers and must reproduce the MATLAB levels, thresholds, evidence,
CoV estimates and indexed posterior samples to round-off. The bundled
fixtures were recorded under Octave, not native MATLAB. Resampling indices
are replayed directly, so posterior weights are checked separately against
an independent expression of the SuS weighting equation (not recorded
MATLAB/Octave weights).

Part 2 - statistics: repeated Python runs on the bundled benchmarks; the mean
log-evidence is compared with the analytical reference value.

    python validate.py              # both parts
    python validate.py --replay     # part 1 only
    python validate.py --nrun 100   # part 2 with 100 runs per benchmark
"""
import argparse
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


EXPECTED_TAGS = ("eggbox_2_1", "loggamma_2_1", "loggamma_5_1",
                 "shells_2_1", "shells_5_1", "shells_10_1")


def array_error(actual, expected):
    """Maximum absolute error, rejecting shapes, NaNs and signed-inf mismatches."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or np.iscomplexobj(actual) or np.iscomplexobj(expected):
        return np.inf
    if np.any(np.isnan(actual)) or np.any(np.isnan(expected)):
        return np.inf
    if not (np.array_equal(np.isposinf(actual), np.isposinf(expected)) and
            np.array_equal(np.isneginf(actual), np.isneginf(expected))):
        return np.inf
    finite = np.isfinite(actual)
    return float(np.max(np.abs(actual[finite] - expected[finite]), initial=0.0))


def posterior_weight_error(out, ps):
    """Independent SuS weight invariant, NOT a recorded MATLAB comparison.

    In level i, log(w) = log(L) + log(P(level i)) - log(Z), restricted
    to L < the next threshold except at the last level. Conditional level
    probabilities are recomputed from the raw samples, independently of
    NumSim.C, ResRat and WgtPos. This catches weight/order mutations which
    replaying already-selected resampling indices cannot detect.
    """
    if len(ps.Wpos) != out["Nite"]:
        return np.inf
    error = 0.0
    log_probability = 0.0
    for i in range(out["Nite"]):
        likelihood = _utils.flat(out["y"][i])[2]
        expected = likelihood + log_probability - ps.Z
        if i + 1 < out["Nite"]:
            threshold = out["g"][i + 1][2]
            expected = np.where(likelihood < threshold, expected, -np.inf)
        expected = expected.reshape(tuple(out["Nsze"][i]), order="F")
        error = max(error, array_error(ps.Wpos[i], expected))
        if i + 1 < out["Nite"]:
            with np.errstate(divide="ignore"):
                log_probability += np.log(np.mean(likelihood > threshold))
    return error


def replay(reference_dir=None):
    print("Part 1: replay of recorded Octave streams and independent weight checks")
    print("  Indexed resampling replay is not a weighted-sampler equivalence test.")
    reference_dir = reference_dir or os.path.join(HERE, "validation", "reference")
    missing = [f"{prefix}_{tag}.mat" for tag in EXPECTED_TAGS for prefix in ("rec", "res")
               if not os.path.isfile(os.path.join(reference_dir, f"{prefix}_{tag}.mat"))]
    if missing:
        print("FAIL: missing required reference files: " + ", ".join(missing))
        return False
    print(f"{'case':14s} {'levels':>7s} {'logZ (Python)':>15s} {'logZ (recorded)':>15s} "
          f"{'max|dG|':>9s} {'max|dx_pos|':>11s} {'max|dCoV|':>9s} {'max|dW|':>9s}  result")
    ok_all = True
    for tag in EXPECTED_TAGS:
        case, ndim, _ = tag.split("_")
        ndim = int(ndim)
        try:
            rng = ReplayRNG(os.path.join(reference_dir, f"rec_{tag}.mat"))
            _utils.set_rng(rng)
            out, ps = run_once(make_lkf(case, ndim))
            dw = posterior_weight_error(out, ps)
            x_pos = ps.PosSim()[0]
            # Check the probabilities actually handed to the resampler, not
            # only the stored Wpos matrices. The matrix invariant above has
            # already checked these log-weights independently of WgtPos.
            log_weights = np.concatenate([w.ravel(order="F") for w in ps.Wpos])
            probabilities = np.exp(log_weights - np.max(log_weights))
            probabilities /= np.sum(probabilities)
            if rng.last_probabilities is None:
                dw = np.inf
            else:
                dw = max(dw, array_error(rng.last_probabilities, probabilities))
            cv = np.array(ps.EvlVar())
            r = loadmat(os.path.join(reference_dir, f"res_{tag}.mat"))
            dG = array_error(np.vstack(out["g"]), r["G"])
            dx = array_error(x_pos, r["x_pos"])
            dcv = array_error(cv, np.array([r["c_z_hat"].item(), r["c_z"].item()]))
            # Future independently recorded files may include the Wpos cell
            # array. Never synthesize or relabel Python weights as references.
            if "Wpos" in r:
                recorded_weights = r["Wpos"].ravel(order="F")
                if len(recorded_weights) != len(ps.Wpos):
                    dw = np.inf
                else:
                    dw = max(dw, *(array_error(w, ref) for w, ref in zip(ps.Wpos, recorded_weights)))
            ok = (out["Nite"] == r["Nite"].item() and dG < 1e-8 and dx < 1e-8
                  and dcv < 1e-8 and dw < 1e-8 and np.isfinite(ps.Z) and np.isfinite(r["Z"].item())
                  and abs(ps.Z - r["Z"].item()) < 1e-8 and np.all(np.isfinite(x_pos))
                  and np.all(np.isfinite(cv)) and np.array_equal(out["Ncal"], r["Ncal"].ravel())
                  and rng.done)
            ok_all &= ok
            print(f"{case + ' n=' + str(ndim):14s} {out['Nite']:7d} {ps.Z:15.8f} {r['Z'].item():15.8f} "
                  f"{dG:9.1e} {dx:11.1e} {dcv:9.1e} {dw:9.1e}  {'PASS' if ok else 'FAIL'}")
        except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            print(f"{tag}: FAIL ({type(exc).__name__}: {exc})")
            ok_all = False
        finally:
            _utils.set_rng(None)
    return bool(ok_all)


def statistics(nrun, seed):
    if nrun < 2:
        raise ValueError("Statistics require at least two runs per benchmark.")
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
