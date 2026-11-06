"""Common sequential Monte Carlo sampling, iteration control, and run history."""

import copy
import numpy as np


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
    """Run the shared process using the model's native update and seed rules."""

    def __init__(self, Sampler):
        if Sampler is None or Sampler.BayStc is None:
            raise ValueError("Sampler with BayStc is required")
        self.Sampler = Sampler
        self.IntBay = Sampler.BayStc
        defaults = getattr(self.IntBay, "RunOptions", {}).get("Defaults", {})
        self.NumSam = defaults.get("NumSam", getattr(self.IntBay, "Nsam", 1000))
        self.ConPrb = defaults.get("ConPrb", self.IntBay.G.get("p", 0.1))
        self.MaxIte = defaults.get("MaxIte", 10000)
        self.Verbose = defaults.get("Verbose", False)
        self.G = []  # Optional prescribed sequence of intermediate states.

    def RunIte(self, rng=None):
        options = dict(
            CountPolicy="validated",
            LoopOffset=1,
            PostCapOffset=1,
            StopPhase="after_seeds",
            PresetMode="protected",
            SeedInputs="model",
            StateHistory="transition",
            Layout="raw",
            OutputFields=("selected", "sigma", "terminal"),
            DisplayFunction=None,
        )
        options.update(getattr(self.IntBay, "RunOptions", {}))
        if "SeedFunction" not in options:
            options["SeedFunction"] = type(self.IntBay).SltSed

        # Copies keep per-run adaptation separate and share the caller's RNG.
        sampler = copy.copy(self.Sampler)
        if rng is not None:
            sampler.rng = rng
        if hasattr(self.IntBay, "start_run"):
            model, sampler = self.IntBay.start_run(
                self.NumSam, self.ConPrb, self.MaxIte, self.G, sampler
            )
        else:
            if options["PresetMode"] != "replace":
                self.IntBay.G = dict(self.IntBay.G, p=self.ConPrb)
            model = copy.copy(self.IntBay)
            sampler = copy.copy(sampler)

        # Preserve legacy budget/endpoint conventions without changing each
        # method's numerical convergence criterion.
        policy = options["CountPolicy"]
        if policy == "padded":
            counts = np.atleast_1d(self.NumSam)
            targets = np.zeros(self.MaxIte + 1, dtype=int)
            if counts.size > 1:
                targets[: counts.size] = counts
            else:
                targets[:] = counts[0]
            limit = self.MaxIte - options["LoopOffset"]
        elif policy == "validated":
            counts = np.atleast_1d(self.NumSam).astype(int)
            if np.any(counts < 1) or self.MaxIte < 2:
                raise ValueError("NumSam must be positive and MaxIte at least 2")
            targets = counts
            limit = self.MaxIte - options["LoopOffset"]
        else:
            targets = np.atleast_1d(self.NumSam)
            limit = int(self.MaxIte) - options["LoopOffset"]

        # Apply prescribed ISDs while preserving the native G carry rules.
        schedule = self.G
        preset = options["PresetMode"]
        if preset == "protected_lambda":
            schedule = [
                copy.deepcopy(value) if isinstance(value, dict)
                else dict(parameters=np.asarray(value).copy(), **{"lambda": 0.6})
                for value in self.G
            ]
        if preset == "replace":
            initial = copy.deepcopy(schedule[0] if len(schedule) else model.G)
            model.G = initial if isinstance(initial, dict) else dict(model.G, parameters=initial)
            if "active" not in model.G:
                model.G["active"] = copy.deepcopy(self.IntBay.G["active"])
        elif len(schedule):
            specified = copy.deepcopy(schedule[0])
            initial_mapping = isinstance(specified, dict)
            if not initial_mapping:
                specified = dict(parameters=specified)
            model.G = prescribed_state(model.G, specified, initial=initial_mapping)
        if preset != "adaptive":
            sampler.BayStc = model

        # Product methods keep sampled-domain G only; transition histories
        # also retain the unsampled next G required by the native estimators.
        x_history, y_history = [], []
        g_history = [] if options["StateHistory"] == "sample" else [copy.deepcopy(model.G)]
        sample_counts, chain_shapes, call_counts = [], [], []
        selected, sigma, probabilities = [], [], []
        xseed = yseed = None
        nc, steps = int(targets[0]), 1
        complete = False

        for level in range(limit):
            # SamGen: sample the initial or current intermediate distribution.
            if hasattr(model, "sample_level"):
                x, y, calls, model, shape = model.sample_level(
                    sampler, self.NumSam, xseed, yseed, steps, level
                )
                count = y.shape[1]
            else:
                if level == 0:
                    x, y, calls, model = sampler.SamGen(nc, steps)
                else:
                    x, y, calls, model = sampler.SamGen(xseed, yseed, steps)
                shape, count = (nc, steps), nc * steps
            if options["Layout"] == "flat":
                x = x.reshape(x.shape[0], -1, order="F")
                y = y.reshape(y.shape[0], -1, order="F")
            x_history.append(x)
            y_history.append(y)
            sample_counts.append(count)
            chain_shapes.append(shape)
            call_counts.append(calls)
            if "sigma" in options["OutputFields"]:
                sigma.append(getattr(sampler, "sigma", None))
            if options["StateHistory"] == "sample":
                g_history.append(copy.deepcopy(model.G))

            # Candidate-domain methods retain update; ordinary ISDs share UpdObj.
            if hasattr(model, "update"):
                next_model = model.update(y[3], y[2], self.ConPrb)
            else:
                if level + 1 < len(schedule):
                    model = copy.copy(model)
                    previous = model.G
                    specified = copy.deepcopy(schedule[level + 1])
                    if preset == "replace":
                        model.G = (
                            dict(previous, **specified) if isinstance(specified, dict)
                            else dict(previous, parameters=specified)
                        )
                        for key in ("lambda", "HMC_time"):
                            if key in previous:
                                model.G[key] = previous[key]
                    else:
                        if not isinstance(specified, dict):
                            specified = dict(parameters=specified)
                        model.G = prescribed_state(previous, specified)
                else:
                    y, _, model, extra = model.UpdObj(x, y, self.ConPrb)
                    y_history[-1] = y
                    call_counts[-1] += extra
                sampler.BayStc = model
                g_history.append(copy.deepcopy(model.G))

            # Some native product methods stop before selecting a new seed set.
            if options["StopPhase"] == "before_seeds":
                if next_model.converged():
                    complete = True
                    break
                if len(x_history) == self.MaxIte - options["PostCapOffset"]:
                    break

            # SltSed/SedSlt: use the explicitly chosen original seed function.
            seed_function = options["SeedFunction"]
            if options["SeedInputs"] == "transition":
                xseed, yseed, steps, model, probability = seed_function(
                    next_model, x, y, self.NumSam, sampler.rng, model
                )
                probabilities.append(probability)
            else:
                if policy == "padded":
                    target = int(targets[level + 1])
                else:
                    if targets.size > 1 and level + 1 >= targets.size:
                        raise ValueError("NumSam schedule exhausted before seed selection")
                    target = int(targets[0] if targets.size == 1 else targets[level + 1])
                if options["SeedInputs"] == "state_rng":
                    xseed, yseed, indices = seed_function(x, y, model.G, target, sampler.rng)
                else:
                    xseed, yseed = seed_function(model, x, y, target)
                    indices = None
                if "selected" in options["OutputFields"]:
                    selected.append(indices)
                nc = xseed.shape[1]
                if policy == "padded":
                    ratio = np.asarray(target / nc, dtype=float) if nc else None
                    steps = (
                        (np.sign(ratio) * np.floor(np.abs(ratio) + 0.5))[()] if nc else np.inf
                    )
                else:
                    steps = int(np.floor(target / nc + 0.5)) if nc else 0
                if self.Verbose:
                    display = options["DisplayFunction"]
                    if display is not None:
                        display(level + 1, count, np.asarray(shape), calls)
                    else:
                        print(f"level {level+1}: samples={count}, evaluations={call_counts[-1]}")

            # Preserve the native empty/cap priority over the convergence flag.
            if options["StopPhase"] == "after_seeds":
                if nc <= 0 or len(x_history) == self.MaxIte - options["PostCapOffset"]:
                    break
                complete = bool(model.FlgCvg)
                if complete:
                    break

        result = dict(
            x=x_history,
            y=y_history,
            g=g_history,
            Nsam=np.asarray(sample_counts),
            Nsze=np.asarray(chain_shapes, dtype=int),
            Ncal=np.asarray(call_counts),
        )
        if "selected" in options["OutputFields"]:
            result["selected"] = selected
        if "sigma" in options["OutputFields"]:
            result["sigma"] = sigma
        if "terminal" in options["OutputFields"]:
            result.update(
                flg_cvg=complete,
                terminal=bool(model.FlgCvg),
                Nite=len(x_history),
                intBay=model,
            )
        else:
            result.update(Nite=len(x_history), flg_cvg=complete, intBay=model)
        if "conditional_probabilities" in options["OutputFields"]:
            result["conditional_probabilities"] = np.asarray(probabilities)
        if "profile" in options["OutputFields"]:
            result["profile"] = model.profile
        if "sampler" in options["OutputFields"]:
            result["sampler"] = type(sampler).__name__
        return result
