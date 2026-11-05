"""Secondary balance MIS from the retained SuS history and native constants."""
import numpy as np
from semcs_core import balance_log_integral


class CommonMIS:
    """Read-only same-history diagnostic; native NumSim remains primary."""
    model_calls = 0
    rng_draws = 0

    def __init__(self, run, native):
        count = int(run["Nite"])
        if count != len(run["y"]) or len(native.C) != count:
            raise ValueError("Expected one native normalizer per sampled level")
        states = run["g"][:count]  # terminal unsampled G is not a proposal
        targets, densities = [], []
        for raw, expected_count in zip(run["y"], run["Nsam"]):
            y = np.asarray(raw, dtype=float).reshape((np.asarray(raw).shape[0], -1), order="F")
            if y.shape[0] != 4 or y.shape[1] != expected_count:
                raise ValueError("Expected canonical4 history and actual sample counts")
            log_likelihood, log_prior = y[2], y[3]
            targets.append(log_likelihood + log_prior)
            proposals = []
            for state, log_normalizer in zip(states, native.C):
                threshold = np.asarray(state["parameters"])[2]
                support = (np.ones(log_likelihood.shape, dtype=bool) if np.isneginf(threshold) else
                    log_likelihood >= threshold if state.get("boundary_rule")=="closed" else log_likelihood > threshold)
                proposals.append(np.where(support, log_prior - log_normalizer, -np.inf))
            densities.append(np.stack(proposals))
        self.log_estimates = balance_log_integral(targets, densities)
        with np.errstate(over="ignore"):
            self.estimates = np.exp(self.log_estimates)
        self.sampled_log_normalizers = np.array(native.C, copy=True)
