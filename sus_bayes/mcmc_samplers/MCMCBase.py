"""Shared MATLAB-compatible sampler contract in Gaussian/Nataf coordinates."""

import copy
import numpy as np


class MCMCBase:
    _defaults = {
        "SedSav": False,
        "Ntry": 20,
        "Adapt": True,
        "Lambda": 0.6,
        "AcpOpt": 0.44,
        "NumIte": 5,
        "InitialTime": np.pi / 4,
        "Sigma0Mode": "unit",
        "CarryLambda": True,
        "ESMode": "coordinate",
        "AcceptArithmetic": "log",
        "GroupMode": "available",
        "AcceptanceRateMode": "attempts",
        "SeedUpdateMode": "each_step",
        "CoordinateMode": "identity",
        "PriorMode": "standard_normal",
    }

    def __init__(self, BayStc=None, rng=None):
        self.BayStc = BayStc
        self.rng = np.random if rng is None else rng
        # Fixed options belong to the sampler; live adaptation stays in G.
        self.__dict__.update(copy.deepcopy(self._defaults))
        if BayStc is not None and hasattr(BayStc, "sampler_options"):
            self.__dict__.update(BayStc.sampler_options(type(self).__name__))

    @property
    def Ndim(self):
        return self.BayStc.Ndim

    @property
    def Nfun(self):
        return self.BayStc.Nfun

    def _rand(self, *shape):
        if hasattr(self.rng, 'rand'):
            return np.asarray(self.rng.rand(*shape))
        return self.rng.random(shape)

    def _randn(self, *shape):
        if hasattr(self.rng, 'randn'):
            return np.asarray(self.rng.randn(*shape))
        return self.rng.standard_normal(shape)

    def _factor(self, b, d):
        G = getattr(b, 'G', {})
        L = G.get('L', G.get('L_pi', np.eye(d)))
        L = np.asarray(L, dtype=float)
        if L.ndim == 0:
            L = np.eye(d) * float(L)
        if L.shape != (d, d) or not np.all(np.isfinite(L)) or np.linalg.matrix_rank(L) != d:
            raise ValueError(
                'Gaussian reference factor must be finite, nonsingular, and Ndim by Ndim.'
            )
        return L

    def _prepare(self, args):
        b = self.BayStc
        if b is None:
            raise ValueError('Set BayStc before SamGen.')
        d = int(b.Ndim)
        L = self._factor(b, d)
        if len(args) == 2:
            nc, ns = map(int, args)
            if nc != args[0] or ns != args[1] or nc < 1 or ns < 1:
                raise ValueError("Chain and step counts must be positive integers.")
            if self.CoordinateMode == "model":
                u = np.asarray(b.sample_reference(nc, self.rng), dtype=float)
            else:
                u = (
                    np.asarray(b.SamGen(nc), dtype=float)
                    if hasattr(b, "SamGen")
                    else L @ self._randn(d, nc)
                )
            b, y, n = self._evaluate(b, u)
            save = True
        elif len(args) == 3:
            u, y, ns = args
            u = np.array(u, dtype=float, copy=True)
            y = np.array(y, dtype=float, copy=True)
            if u.ndim != 2 or u.shape[0] != d:
                raise ValueError('Seeds must have shape (Ndim,Nchains).')
            if y.ndim == 1:
                y = y.reshape(1, -1)
            nc = u.shape[1]
            if int(ns) != ns:
                raise ValueError("Step count must be a positive integer.")
            ns = int(ns)
            n = 0
            save = self.SedSav
        else:
            raise TypeError('SamGen(Nchains,Nsteps) or SamGen(U_seed,Y_seed,Nsteps).')
        if nc < 1 or ns < 1 or y.shape != (int(b.Nfun), nc):
            raise ValueError('Invalid chain count, step count, or packed seed outputs.')
        return b, u, y, n, save, L, nc, ns

    @staticmethod
    def _evaluate(b, u):
        result = b.EvlY(u)
        if result is not None:
            b = result

        def val(name):
            x = getattr(b, name)
            x = x() if callable(x) else x
            x = np.asarray(x, dtype=float)
            return x.reshape(1, -1) if x.ndim <= 1 else x

        y = np.vstack((val('EvlLKF'), val('EvlPDF'), val('Y')))
        if y.shape != (int(b.Nfun), u.shape[1]):
            raise ValueError(
                'EvlY must provide packed likelihood, reference log density, and model outputs.'
            )
        return b, y, int(b.Ncal)

    def _to_reference(self, b, x):
        if self.CoordinateMode == "identity":
            return x
        if self.CoordinateMode != "model":
            raise ValueError('CoordinateMode must be identity or model.')
        return np.asarray(b.to_reference(x), dtype=float)

    def _from_reference(self, b, u):
        return np.asarray(b.from_reference(u), dtype=float) if self.CoordinateMode == "model" else u

    def _reject(self, new, old, p):
        if self.AcceptArithmetic == "exp":
            # Literal author comparison: NaN<uniform is false, as in MATLAB.
            # Supported source chains have finite current log likelihood.
            with np.errstate(invalid='ignore', over='ignore', under='ignore'):
                return np.exp(np.asarray(new) - np.asarray(old)) < p
        if self.AcceptArithmetic != "log":
            raise ValueError('AcceptArithmetic must be log or exp.')
        # Undefined likelihood ratios (including -Inf minus -Inf) reject the move.
        with np.errstate(invalid='ignore'):
            delta = np.asarray(new) - np.asarray(old)
        return np.isnan(delta) | (delta < np.log(np.maximum(p, np.finfo(float).tiny)))

    @staticmethod
    def _state(b, key, default):
        G = getattr(b, 'G', {})
        value = G.get(key, default)
        return default if value is None else float(value)

    @staticmethod
    def _setstate(b, key, value):
        G = dict(getattr(b, 'G', {}))
        G[key] = float(value)
        b.G = G
        return b
