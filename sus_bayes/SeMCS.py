"""Common sequential Monte Carlo loop; method policies belong to the model."""

import copy


def state_property(name, default=None):
    """Expose a model parameter stored in its evolving G state."""

    def get(self):
        return self.G.get(name, default)

    def set_(self, value):
        self.G = dict(getattr(self, "G", {}), **{name: value})

    return property(get, set_)


def prescribed_state(current, specified, *, initial=False):
    """Apply a prescribed intermediate state without resetting live adaptation."""
    requested = copy.deepcopy(specified)
    state = copy.deepcopy(current)
    state.update(requested)
    for key in ("p", "Nsam"):
        if key in current:
            state[key] = copy.deepcopy(current[key])
    for key in ("lambda", "HMC_time"):
        if initial:
            if key not in requested:
                state.pop(key, None)
        elif key in current:
            state[key] = copy.deepcopy(current[key])
    return state


class SeMCS:
    """Execute the same sampling lifecycle for every intermediate model."""

    def __init__(self, Sampler):
        if Sampler is None or Sampler.BayStc is None:
            raise ValueError("Sampler with BayStc is required")
        self.Sampler = Sampler
        self.IntBay = Sampler.BayStc
        defaults = self.IntBay.run_defaults()
        self.NumSam = defaults.get("NumSam", getattr(self.IntBay, "Nsam", 1000))
        self.ConPrb = defaults.get("ConPrb", self.IntBay.G.get("p", 0.1))
        self.MaxIte = defaults.get("MaxIte", 10000)
        self.Verbose = defaults.get("Verbose", False)
        self.G = []  # Optional prescribed sequence of intermediate states.

    def RunIte(self, rng=None):
        # Keep per-run sampler state separate while sharing the caller's RNG.
        sampler = copy.copy(self.Sampler)
        if rng is not None:
            sampler.rng = rng
        context = self.IntBay.start_run(self, sampler)

        for level in range(context["limit"]):
            # SamGen: sample the initial or current intermediate distribution.
            if hasattr(context["model"], "sample_level"):
                record = context["model"].sample_level(context, level)
            else:
                if level == 0:
                    x, y, calls, model = context["sampler"].SamGen(context["nc"], context["steps"])
                else:
                    x, y, calls, model = context["sampler"].SamGen(
                        context["xseed"], context["yseed"], context["steps"]
                    )
                context["model"] = model
                record = dict(
                    x=x,
                    y=y,
                    calls=calls,
                    count=context["nc"] * context["steps"],
                    shape=(context["nc"], context["steps"]),
                )
                context["x"].append(x)
                context["y"].append(y)
                context["Ncal"].append(calls)
                context["Nsam"].append(record["count"])
                context["Nsze"].append(record["shape"])

            # UpdObj/update: evolve the method's intermediate distribution.
            context["model"].advance_level(context, record, level)

            if context["model"].run_complete(context, record, "before_seeds"):
                break

            # SltSed/SedSlt: keep each method's own seed-selection strategy.
            context["model"].select_run_seeds(context, record, level)
            if context["model"].run_complete(context, record, "after_seeds"):
                break

        return context["model"].finish_run(context)
