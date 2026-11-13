"""Sequential sampling with one state, budget and raw-history convention."""

import copy
import numpy as np


def prescribed_state(current, specified, *, initial=False):
    """Patch G; later prescribed ISDs retain the proposal's live adaptation."""
    state = copy.deepcopy(current)
    patch = specified if isinstance(specified, dict) else dict(parameters=specified)
    state.update(copy.deepcopy(patch))
    if not initial:
        for key in ("lambda", "HMC_time"):
            if key in current:
                state[key] = copy.deepcopy(current[key])
    return state


class SeMCS:
    """Sample at most MaxIte levels; NumSam vectors are finite level budgets."""

    def __init__(self, Sampler):
        if Sampler is None or Sampler.BayStc is None:
            raise ValueError("Sampler with BayStc is required")
        self.Sampler = Sampler
        self.IntBay = Sampler.BayStc
        self.NumSam = getattr(self.IntBay, "Nsam", 1000)
        self.ConPrb = self.IntBay.G.get("p", 0.1)
        self.MaxIte = getattr(self.IntBay, "MaxIte", 10000)
        self.Verbose = getattr(self.IntBay, "Verbose", False)
        self.G = []

    def RunIte(self, rng=None):
        if not np.isscalar(self.ConPrb) or not np.isfinite(self.ConPrb) or not 0 < self.ConPrb < 1:
            raise ValueError("ConPrb must be a finite scalar strictly between zero and one")
        if not np.isscalar(self.MaxIte) or not np.isfinite(self.MaxIte) or self.MaxIte < 1 or self.MaxIte != int(self.MaxIte):
            raise ValueError("MaxIte must be a positive integer")
        targets = np.atleast_1d(np.asarray(self.NumSam, dtype=float))
        if targets.ndim != 1 or not targets.size or np.any(~np.isfinite(targets)) or np.any(targets < 1) or np.any(targets != np.floor(targets)):
            raise ValueError("NumSam must contain positive integer level budgets")
        targets = targets.astype(int)
        limit = int(self.MaxIte) if targets.size == 1 else min(int(self.MaxIte), targets.size)

        sampler = copy.copy(self.Sampler)
        if rng is not None:
            sampler.rng = rng
        model = copy.copy(self.IntBay)
        model.G = copy.deepcopy(model.G)
        model.G["p"] = self.ConPrb
        if len(self.G):
            model.G = prescribed_state(model.G, self.G[0], initial=True)
        sampler.BayStc = model

        x_history, y_history = [], []
        g_history = [copy.deepcopy(model.G)]
        sample_counts, chain_shapes, call_counts = [], [], []
        xseed = yseed = None
        nc, steps = int(targets[0]), 1
        complete = False
        reason = "max_iterations"

        for level in range(limit):
            # 1. Generate samples; retain the sampler's chain and step axes.
            if level == 0:
                x, y, calls, model = sampler.SamGen(nc, steps)
            else:
                x, y, calls, model = sampler.SamGen(xseed, yseed, steps)
            shape = (nc, steps)
            g_history[-1] = copy.deepcopy(model.G)
            x_history.append(x)
            y_history.append(y)
            sample_counts.append(int(np.prod(shape)))
            chain_shapes.append(shape)
            call_counts.append(calls)

            # 2. Update the intermediate distribution and cached sample values.
            if level + 1 < len(self.G):
                model.G = prescribed_state(model.G, self.G[level + 1])
            else:
                y, _, model, extra = model.UpdObj(x, y, model.G["p"])
                y_history[-1] = y
                call_counts[-1] += extra
            g_history.append(copy.deepcopy(model.G))
            sampler.BayStc = model
            if getattr(model, "StopBeforeSeeds", False) and model.FlgCvg:
                complete = True
                reason = "isd_complete"
                break
            if level + 1 == limit:
                complete = bool(model.FlgCvg)
                if complete:
                    reason = "method_complete"
                elif targets.size > 1 and level + 1 == targets.size:
                    reason = "sample_budget"
                break

            # 3. Each model owns its actual SedSlt rule.
            target = int(targets[level + 1] if targets.size > 1 else targets[0])
            xseed, yseed = model.SedSlt(x, y, target, sampler.rng)
            nc = xseed.shape[1]
            steps = int(np.floor(target / nc + 0.5)) if nc else 0
            if self.Verbose:
                print(f"level {level+1}: samples={sample_counts[-1]}, evaluations={call_counts[-1]}")
            # Evaluate the native criterion after its original seed operation.
            complete = bool(model.FlgCvg)
            if complete:
                reason = "method_complete"
                break
            if nc == 0:
                reason = "empty_support"
                break

        return dict(
            x=x_history, y=y_history, g=g_history,
            Nsam=np.asarray(sample_counts), Nsze=np.asarray(chain_shapes, dtype=int),
            Ncal=np.asarray(call_counts), Nite=len(x_history), flg_cvg=complete, stop_reason=reason,
            intBay=model, sampler=type(sampler).__name__,
        )
