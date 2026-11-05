"""Method-only seed scale policy. Shared aCS and Engine remain unchanged."""
import copy
import numpy as np

VARIANT="closed_ties_unit_fallback_v1"
SEED_SCALE_POLICY="unit_if_all_selected_reference_vectors_identical_else_baseline"

def selected_reference_vectors(model,sampler,seeds):
    """Use exactly aCS's model-reference map and Gaussian whitening factor."""
    reference=sampler._to_reference(model,np.asarray(seeds,dtype=float))
    factor=sampler._factor(model,int(model.Ndim))
    return reference if np.array_equal(factor,np.eye(model.Ndim)) else np.linalg.solve(factor,reference)

def apply_selected_scale(model,sampler,seeds):
    base=model.G["baseline_sigma0_mode"]
    if base not in ("unit","seed_std"):raise ValueError("Unsupported baseline Sigma0Mode")
    reference=selected_reference_vectors(model,sampler,seeds)
    identical=bool(reference.shape[1] and np.all(reference==reference[:,[0]]))
    effective="unit" if base=="seed_std" and identical else base
    runtime=copy.deepcopy(model.G["sus_runtime"])
    runtime.update(degenerate_selected_seed_set=identical,effective_sigma0_mode=effective,
        reference_coordinate_space="aCS reference coordinates whitened by G.L",
        fallback_applied=bool(base=="seed_std" and identical))
    settings=copy.deepcopy(model.G["sampler"])
    settings["aCS"]["Sigma0Mode"]=effective
    model.G=dict(model.G,sampler=settings,sus_runtime=runtime)
    sampler.BayStc=model
    return model
