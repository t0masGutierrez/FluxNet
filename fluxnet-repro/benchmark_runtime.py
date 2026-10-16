#!/usr/bin/env python3
"""Measure the fixed CPU forecast without writing reference files.

Run from the repository root with the same Python environment before and after
changes. Each timing is the median after three warm-up calls. Peak memory is
for the whole process and includes model loading.
"""

import json
import platform
import resource
import statistics
import time

import run_example as example
import torch

from coefficients import extended_chebyshev_polynomials


def measure(function, repeats):
    """Return the median elapsed time in seconds."""
    for _ in range(3):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main():
    """Measure fixed inputs with four CPU threads and full float32 precision."""
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(example.TRAINING_SEED)
    device = torch.device("cpu")
    x = torch.linspace(-2, 2, 1900).reshape(1, 19, 100)
    observed, *_ = example.load_fixed_example(example.BUNDLE / "truth/task-011-chebyshev_degree_4_7-grf/saved.h5")
    model = example.load_model(example.BUNDLE / "weights/step_37500.ckpt", device)
    with torch.no_grad():
        polynomial = measure(lambda: extended_chebyshev_polynomials(x, 8), 50)
        forecast = measure(lambda: example.rollout(model, observed, device), 10)
    # macOS reports bytes; Linux reports KiB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "torch": torch.__version__,
                "threads": 4,
                "device": str(device),
                "polynomial_seconds": polynomial,
                "rollout_seconds": forecast,
                "peak_rss_bytes": rss_bytes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
