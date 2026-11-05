"""Data-only balance MIS for cached, normalized proposal densities."""
import numpy as np
from scipy.special import logsumexp


def balance_log_integral(log_targets, log_proposal_densities):
    """Integrate cached targets using actual counts of sampled proposals.

    For sampled block i, targets have shape (responses, N_i), or (N_i,),
    and cross densities have shape (number_of_sampled_blocks, N_i).
    Each cross row is the normalized log density q_j, with its normalizer
    subtracted exactly once. No model evaluation, resampling, or RNG use.
    """
    if not log_targets or len(log_targets) != len(log_proposal_densities):
        raise ValueError("One target and cross-density array per sampled block required")
    counts = np.asarray([np.asarray(v).shape[-1] for v in log_targets], dtype=int)
    if np.any(counts <= 0):
        raise ValueError("Sampled proposal blocks must have positive actual counts")
    contributions, responses = [], None
    for target, cross, count in zip(log_targets, log_proposal_densities, counts):
        target = np.atleast_2d(np.asarray(target, dtype=float))
        cross = np.asarray(cross, dtype=float)
        if cross.shape != (len(counts), count) or target.shape[1] != count:
            raise ValueError("Cached target/cross-density shape mismatch")
        if responses is None:
            responses = target.shape[0]
        if target.shape[0] != responses:
            raise ValueError("Target response count changed between blocks")
        denominator = logsumexp(np.log(counts)[:, None] + cross, axis=0)
        if not np.isfinite(denominator).all():
            raise ValueError("A sample has no finite sampled-mixture density")
        contributions.append(target - denominator[None, :])
    return logsumexp(np.concatenate(contributions, axis=1), axis=1)
