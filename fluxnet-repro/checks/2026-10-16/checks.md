# Cleanup checks, 2026-10-16

These are small checks for the code cleanup from commit `844caf9`.
They are not new training runs or a new research evaluation. The historical
`post_unseal_exploratory` labels and all saved scientific records remain as they were.

Changes:

- Remove only the root README and its package metadata entry.
- Replace three `einops` calls with PyTorch flatten calls. Remove `einops` from
  package dependencies and the existing lock file. Keep all other locked versions.
- Use the shared finite-value and positive-number checks.
- Exclude masked values before coefficient-fit arithmetic. Reject invalid flux
  inputs with clear errors.
- Reuse polynomial values and slopes at the clamped input for the linear extension.
  The polynomial law, endpoint slopes, and periodic state update do not change.
- Require saved reference files for example checks. Changing the ICON output
  folder no longer skips the check. Reject invalid tolerances and nonfinite
  arrays. Protect FluxNet input files from an output path with or without `.npz`.

## Numerical checks

The main environment was Python 3.12.12, PyTorch 2.13.0, on an Apple M1.
The absolute comparison tolerance remained `2e-5`, with zero relative tolerance.

| Check | Result |
| --- | --- |
| `python -m unittest discover -v` | 12 tests passed |
| FluxNet CPU, 20 steps | Reference difference 0; prediction difference `1.1920929e-6` |
| FluxNet CPU before versus after | Predictions exactly equal |
| FluxNet Apple GPU, 20 steps | Reference difference 0; prediction difference `2.2649765e-6` |
| ICON CPU, all three saved models, before and after | Reference and prediction differences 0 |
| ICON Apple GPU, seed 0 | Failed: largest prediction difference `2.2411346e-5`; 4 of 2000 entries exceed tolerance |

The original ICON runner also fails on the Apple GPU at the same maximum
error. The fixed ICON source was not changed. The GPU run stops at seed 0;
seeds 1 and 2 were not checked on that device. CUDA was unavailable.

The tests cover masked NaN, infinity, and large finite values; invalid time
ratios and state arrays; inverse periodic flux; polynomial values and gradients;
strict saved-model loading; reference forecasts; periodic shifts; and 20-step
conservation. The conservation test checks the total state after each step
against the initial total with an absolute limit of `2e-5`.
Tests also cover missing references, a separate ICON output folder, invalid
comparison tolerances, nonfinite reference data, and input overwrite attempts.
The input and output-folder failures were reproduced before their fixes.

## Time and memory

Run `python fluxnet-repro/benchmark_runtime.py` with the same environment for
both versions. The baseline uses all six runtime files from `844caf9` and the
same saved weights and input files. Settings: CPU, four threads, float32,
`highest` matrix multiplication precision, seed 1307. The polynomial input has
shape `(1, 19, 100)`, spans `[-2, 2]`, and uses degree 8. The forecast uses the
fixed 20-step example. Each process runs three warm-up calls, then takes the
median of 50 polynomial calls and 10 forecasts. Three old/new pairs ran in turn,
after the large ICON checks had finished. See `timing.json` for every measurement,
including the earlier preliminary measurements.

| Median of the three process results | Before | After |
| --- | ---: | ---: |
| Polynomial evaluation | 0.677 ms | 0.235 ms |
| Full 20-step forecast | 69.2 ms | 58.8 ms |
| Peak process memory, including model loading | 317.7 MB | 317.8 MB |

The forecast median decreased by about 15%. The before range was 67.4–100.0 ms;
the after range was 58.8–78.1 ms. The ranges overlap. These local measurements do
not establish a speed increase on other hardware. Memory use was effectively
unchanged. The preliminary after measurement was slower than the preliminary
before measurement; those runs were separated by the large ICON checks.

## Package and record checks

- `uv build` created the source archive and then the wheel from that archive.
  Setuptools warns that the root README is absent; the build succeeds.
- A fresh environment installed the wheel with the `comparison` extra.
  `uv pip check` passed. The installed FluxNet module loaded the saved model and
  passed the CPU reference check under PyTorch 2.14.1 with difference `1.1920929e-6`.
- `uv lock --check` passed. The only removed locked package is `einops` 0.8.2.
  The lock file records this local setup, not the historical training setup.
- Ruff code and format checks passed for the six runtime modules, both runners,
  the timing script, and the tests. `git diff --check` passed.
- Direct Git content hashes matched for 298 tracked bundle records, fixed ICON
  source files, data files, saved model files, and licenses. Historical checksums,
  manifests, settings, environment records, and tables were not rewritten.
- New output files use ignored `reproduced-*` paths or temporary folders. They
  are not included in this commit. No full training or new study was run.
- A separate code review found the suffix-free output overwrite case. Its test
  failed before the fix and passed after it. The final review found no further
  substantive issues.
