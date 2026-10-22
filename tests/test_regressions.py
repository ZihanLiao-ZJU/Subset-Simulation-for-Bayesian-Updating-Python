"""Regression tests; no MATLAB runtime or fabricated reference data required."""
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import genextreme, lognorm, norm

from sus_bayes import Distribution, Nataf, makedist
from sus_bayes.log_operations import logcumsum
import validate
from validation.replay import ReplayRNG


class NatafTests(unittest.TestCase):
    def test_documented_frozen_wrappers(self):
        marginals = [Distribution("Normal", norm(loc=2, scale=3)),
                     Distribution("Lognormal", lognorm(s=.4, scale=np.exp(.2))),
                     Distribution("Generalized Extreme Value", genextreme(c=-.2))]
        u = np.array([[-2., 0., 1.]] * 3)
        prior = Nataf(marginals, np.eye(3))
        expected = np.array([m.icdf(norm.cdf(row)) for m, row in zip(marginals, u)])
        assert_allclose(prior.U2X(u), expected)
        assert_allclose(prior.U2X(u.reshape(3, 1, 3)), expected.reshape(3, 1, 3))

    def test_parameterized_fast_paths_keep_tail_stability(self):
        prior = Nataf([makedist("Normal", mu=2, sigma=3),
                       makedist("Lognormal", mu=.2, sigma=.4)], np.eye(2))
        u = np.array([[-12., 12.]] * 2)
        expected = np.array([2 + 3 * u[0], np.exp(.2 + .4 * u[1])])
        assert_allclose(prior.U2X(u), expected)
        self.assertTrue(np.all(np.isfinite(prior.U2X(u))))

    def test_gev_clipping_policy_preserved(self):
        marg = makedist("GeneralizedExtremeValue", k=.2)
        prior = Nataf([marg], np.eye(1))
        u = np.array([[-12., 0., 8., 12.]])
        expected = marg.icdf(np.clip(norm.cdf(u), 1e-15, 1 - 1e-15))
        assert_allclose(prior.U2X(u), expected)
        self.assertTrue(np.all(np.isfinite(prior.U2X(u))))

    def test_uniform_correlation_positive_negative_and_zero(self):
        for rho in (-.5, 0., .5):
            with self.subTest(rho=rho):
                prior = Nataf([makedist("Uniform")] * 2, [[1., rho], [rho, 1.]])
                expected = 2 * np.sin(np.pi * rho / 6)
                assert_allclose(prior.Rho_U[0, 1], expected, atol=5e-5)

    def test_zero_pair_inside_nonidentity_matrix(self):
        prior = Nataf([makedist("Uniform")] * 3,
                      [[1., .3, 0.], [.3, 1., -.3], [0., -.3, 1.]])
        self.assertEqual(prior.Rho_U[0, 2], 0.)
        self.assertLess(prior.Rho_U[1, 2], 0.)


class SignedLogTests(unittest.TestCase):
    def test_signed_cumsum_both_axes(self):
        values = np.array([[1., -3., 5.], [2., -5., 7.]])
        logged = np.log(values.astype(complex))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert_allclose(np.exp(logcumsum(logged, 1)), np.cumsum(values, axis=1), atol=1e-14)
            assert_allclose(np.exp(logcumsum(logged.T, 0)), np.cumsum(values.T, axis=0), atol=1e-14)

    def test_real_and_zero_prefixes(self):
        result = logcumsum(np.array([[-np.inf, 0., np.log(2.)]]), 1)
        self.assertFalse(np.iscomplexobj(result))
        assert_allclose(np.exp(result), [[0., 1., 3.]])

    @unittest.expectedFailure
    def test_inherited_logsum_all_negative_prefix_limitation(self):
        # MATLAB's existing logsum also cannot handle an all-negative slice.
        # A broader signed-log arithmetic redesign is deliberately not hidden
        # inside the output-dtype regression fix.
        logged = np.log(np.array([[-3., -5.]], dtype=complex))
        assert_allclose(np.exp(logcumsum(logged, 1)), [[-3., -8.]], atol=1e-14)


class ValidationTests(unittest.TestCase):
    def test_nonfinite_comparison_is_fail_closed(self):
        self.assertEqual(validate.array_error([1., -np.inf], [1., -np.inf]), 0.)
        self.assertEqual(validate.array_error([1.], [1. + 1e-10]), abs(1. - (1. + 1e-10)))
        for actual, expected in [([np.nan], [np.nan]), ([np.inf], [-np.inf]),
                                 ([np.nan], [1.]), ([np.inf], [1.]), ([[1.]], [1.])]:
            with self.subTest(actual=actual, expected=expected):
                self.assertTrue(np.isinf(validate.array_error(actual, expected)))

    def test_empty_reference_directory_fails(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(validate.replay(directory))

    def test_missing_single_required_fixture_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            for tag in validate.EXPECTED_TAGS:
                for prefix in ("rec", "res"):
                    Path(directory, f"{prefix}_{tag}.mat").touch()
            os.unlink(Path(directory, "res_shells_10_1.mat"))
            self.assertFalse(validate.replay(directory))

    def test_independent_weights_detect_permutation(self):
        y0, y1 = np.zeros((4, 4, 1)), np.zeros((4, 2, 1))
        y0[2, :, 0] = np.log([1., 2., 3., 4.])
        y1[2, :, 0] = np.log([3., 4.])
        out = dict(Nite=2, y=[y0, y1], Nsze=np.array([[4, 1], [2, 1]]),
                   g=[[-np.inf, 0., -np.inf], [0., 0., np.log(2.5)]])
        ps = SimpleNamespace(Z=np.log(2.5), Wpos=[np.array([[np.log(.4)], [np.log(.8)],
                                                          [-np.inf], [-np.inf]]),
                                                np.log([[.6], [.8]])])
        self.assertLess(validate.posterior_weight_error(out, ps), 1e-14)
        ps.Wpos[0] = ps.Wpos[0][::-1]
        self.assertTrue(np.isinf(validate.posterior_weight_error(out, ps)))

    def test_statistics_requires_repetitions(self):
        with self.assertRaises(ValueError):
            validate.statistics(1, 1)

    def test_bundled_replay(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(validate.replay())

    def test_replay_detects_wrong_probabilities_used_by_possim(self):
        original = validate.ReplayRNG.randsample

        def wrong_probabilities(rng, n, k, w):
            return original(rng, n, k, np.ones_like(w))

        with patch.object(validate.ReplayRNG, "randsample", wrong_probabilities):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(validate.replay())

    def test_replay_rejects_corrupt_nonfinite_reference(self):
        load_reference = validate.loadmat
        for invalid in (np.nan, np.inf):
            def corrupt(path):
                reference = load_reference(path)
                reference["G"][0, 0] = invalid  # the true initial threshold is -inf
                return reference
            with self.subTest(invalid=invalid), patch.object(validate, "loadmat", corrupt):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertFalse(validate.replay())


class ReplayRNGTests(unittest.TestCase):
    @staticmethod
    def rng(indices=(1., 2.)):
        rng = ReplayRNG.__new__(ReplayRNG)
        rng.entries = [("randsample", np.asarray(indices))]
        rng.pos = 0
        rng.last_probabilities = None
        return rng

    def test_probabilities_exposed(self):
        rng = self.rng()
        assert_allclose(rng.randsample(2, 2, [1., 3.]), [0, 1])
        assert_allclose(rng.last_probabilities, [.25, .75])

    def test_invalid_weights_rejected(self):
        for weights in ([0., 0.], [-1., 2.], [np.nan, 1.], [np.inf, 1.], [1.]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.rng().randsample(2, 2, weights)

    def test_indices_must_be_valid_and_supported(self):
        for indices, weights in [((0., 2.), [1., 1.]), ((1., 3.), [1., 1.]),
                                 ((1.5, 2.), [1., 1.]), ((np.nan, 2.), [1., 1.]),
                                 ((1., 2.), [0., 1.])]:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.rng(indices).randsample(2, 2, weights)


if __name__ == "__main__":
    unittest.main()
