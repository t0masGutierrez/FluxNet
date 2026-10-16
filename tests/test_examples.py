"""Check that output folders cannot disable the saved reference checks."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ExampleTests(unittest.TestCase):
    def test_icon_custom_output_checks_bundled_reference(self):
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run(
                [sys.executable, str(ROOT / "icon-repro/run_example.py"), "--seed", "0", "--output-dir", output],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("reference_max_abs_diff=", result.stdout)
            self.assertIn("prediction_max_abs_diff=", result.stdout)
            self.assertTrue((Path(output) / "reproduced-seed-0-forecast.npz").is_file())

    def test_flux_missing_reference_fails(self):
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "fluxnet-repro/run_example.py"),
                    "--expected",
                    str(Path(output) / "missing.npz"),
                    "--output",
                    str(Path(output) / "output.npz"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileNotFoundError", result.stderr)

    def test_flux_rejects_invalid_tolerance_and_input_overwrite(self):
        for options in (
            ["--atol", "nan"],
            ["--atol", "-1"],
            ["--output", str(ROOT / "fluxnet-repro/example/forecast-example.npz")],
        ):
            with self.subTest(options=options):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "fluxnet-repro/run_example.py"), *options],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ValueError", result.stderr)

    def test_flux_rejects_nonfinite_reference(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as output:
            expected_path = Path(output) / "expected.npz"
            with np.load(ROOT / "fluxnet-repro/example/forecast-example.npz") as saved:
                reference = saved["reference"].copy()
                reference[0, 0] = np.nan
                np.savez(expected_path, reference=reference, prediction=saved["prediction"])
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "fluxnet-repro/run_example.py"),
                    "--expected",
                    str(expected_path),
                    "--output",
                    str(Path(output) / "output.npz"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must contain only finite values", result.stderr)

    def test_flux_rejects_output_without_suffix_that_names_reference(self):
        import shutil

        with tempfile.TemporaryDirectory() as output:
            expected_path = Path(output) / "reference.npz"
            shutil.copyfile(ROOT / "fluxnet-repro/example/forecast-example.npz", expected_path)
            original = expected_path.read_bytes()
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "fluxnet-repro/run_example.py"),
                    "--expected",
                    str(expected_path),
                    "--output",
                    str(expected_path.with_suffix("")),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(expected_path.read_bytes(), original)
