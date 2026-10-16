"""Small numerical checks for the public runtime and saved model."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chebyshev import chebyshev_polynomial_derivatives, chebyshev_polynomials  # noqa: E402
from coefficients import extended_chebyshev_polynomials, fit_flux_coefficients_from_evidence  # noqa: E402
from flux_inference import infer_periodic_interface_flux  # noqa: E402

spec = importlib.util.spec_from_file_location("flux_example", ROOT / "fluxnet-repro/run_example.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NumericalTests(unittest.TestCase):
    def test_masked_fit_ignores_missing_values(self):
        x = torch.tensor([[-0.8, -0.2, 0.3, 0.9, 0.0]], dtype=torch.float64)
        y = x.square()
        mask = torch.tensor([[True, True, True, True, False]])
        options = dict(degree=3, ridge=1e-4, speed_bound=20.0, epsilon=1e-5)
        expected = fit_flux_coefficients_from_evidence(x, y, mask, **options)
        for bad in (float("nan"), float("inf"), -float("inf"), 1e300):
            with self.subTest(bad=bad):
                missing_x, missing_y = x.clone(), y.clone()
                missing_x[~mask] = bad
                missing_y[~mask] = bad
                actual = fit_flux_coefficients_from_evidence(missing_x, missing_y, mask, **options)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            fit_flux_coefficients_from_evidence(x, y, torch.zeros_like(mask), **options)
        y[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            fit_flux_coefficients_from_evidence(x, y, mask, **options)

    def test_flux_rejects_invalid_ratio(self):
        x = torch.zeros(1, 8, 1)
        for ratio in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                infer_periodic_interface_flux(x, x, ratio)

    def test_flux_rejects_invalid_states(self):
        x = torch.zeros(1, 8, 1)
        for other in (x.double(), torch.full_like(x, float("nan"))):
            with self.subTest(dtype=other.dtype), self.assertRaises(ValueError):
                infer_periodic_interface_flux(x, other, 0.5)
        with self.assertRaises(ValueError):
            infer_periodic_interface_flux(x[:, :0], x[:, :0], 0.5)

    def test_periodic_flux_recovers_transition(self):
        torch.manual_seed(7)
        x = torch.randn(2, 64, 1, dtype=torch.float64)
        flux = torch.randn_like(x)
        y = x + 0.5 * (torch.roll(flux, 1, -2) - flux)
        recovered = infer_periodic_interface_flux(x, y, 0.5)
        torch.testing.assert_close(recovered, flux - flux.mean(-2, keepdim=True), rtol=0, atol=2e-14)

    def test_extended_values_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for degree in (0, 1, 3, 8):
                x = torch.tensor([-3.0, -1.0, -0.2, 0.0, 0.4, 1.0, 2.0], dtype=dtype, requires_grad=True)
                clamped = x.clamp(-1, 1)
                expected = chebyshev_polynomials(clamped, degree)
                slopes = chebyshev_polynomial_derivatives(clamped, degree)
                for endpoint, mask in ((-1.0, x < -1), (1.0, x > 1)):
                    edge = torch.full_like(x, endpoint)
                    values = chebyshev_polynomials(edge, degree)
                    derivatives = chebyshev_polynomial_derivatives(edge, degree)
                    expected = torch.where(mask[:, None], values + derivatives * (x - endpoint)[:, None], expected)
                actual, actual_slopes = extended_chebyshev_polynomials(x, degree)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_slopes, slopes, rtol=0, atol=0)
                if degree:
                    grad = torch.autograd.grad(actual.sum(), x)[0]
                    torch.testing.assert_close(grad, slopes.sum(-1))


class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        cls.model = runner.load_model(runner.BUNDLE / "weights/step_37500.ckpt", torch.device("cpu"))
        cls.observed, *_ = runner.load_fixed_example(runner.BUNDLE / "truth/task-011-chebyshev_degree_4_7-grf/saved.h5")

    def test_saved_forecast(self):
        actual = runner.rollout(self.model, self.observed, torch.device("cpu"))
        with np.load(runner.BUNDLE / "example/forecast-example.npz") as expected:
            np.testing.assert_allclose(actual, expected["prediction"], rtol=0, atol=2e-5)

    def test_repeated_steps_conserve_sum_and_periodic_shift(self):
        data = runner.make_model_input(self.observed)
        with torch.no_grad():
            program = self.model.prepare_rollout(data)
            state = self.observed[-1:]
            initial_sum = state.double().sum()
            direct = self.model(data)[:, 0]
            torch.testing.assert_close(direct, self.model.step_prepared(state, program), rtol=0, atol=0)
            shifted = self.model.step_prepared(torch.roll(state, 7, 1), program)
            torch.testing.assert_close(shifted, torch.roll(direct, 7, 1), rtol=0, atol=2e-5)
            for _ in range(20):
                state = self.model.step_prepared(state, program)
                self.assertLess(abs((state.double().sum() - initial_sum).item()), 2e-5)
            with self.assertRaises(ValueError):
                self.model.step_prepared(torch.full_like(state, float("nan")), program)
            with self.assertRaises(ValueError):
                self.model.step_prepared(state[:, :10], program)


if __name__ == "__main__":
    unittest.main()
