# FluxNet

FluxNet uses examples of the state at different time steps on a periodic grid of one-dimensional cells to predict the next
values. It moves an amount from one cell to the next, so the total amount in
all cells stays the same apart from small computer rounding errors.

This folder lets you run the saved FluxNet model and repeat one fixed test. It
also includes the fixed ICON files used for the comparison test. 

## Install

Use Python 3.11 or newer:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Some computers need a different PyTorch installation. If the standard install
does not work, install the PyTorch version for your computer first.

## Run the fixed FluxNet check

From the repository root:

```sh
python fluxnet-repro/run_example.py
```

The script checks the saved model file, reads the fixed reference data, makes
20 predictions, and compares them with the saved answer. A successful run
currently reports:

```text
reference_max_abs_diff=0
prediction_max_abs_diff=1.1920929e-06
checkpoint_update=37500
dt=0.005 dx=0.01
```

`reference_max_abs_diff` checks that the reference data is the expected data.
`prediction_max_abs_diff` is the largest difference between this run and the
saved answer.

The new forecast is written to
`fluxnet-repro/example/reproduced-forecast-example.npz`. Git ignores this
file because the test can create it again.

## Run the ICON comparison

Install the extra packages needed for the ICON test:

```sh
python -m pip install -e '.[comparison]'
```

Then run all three saved ICON models:

```sh
python icon-repro/run_example.py
```

Use `--seed 0`, `--seed 1`, or `--seed 2` to run one saved model. The ICON
test checks each model file, loads the fixed ICON code and settings, makes the
same 20 predictions, and compares them with the saved answer for that run.

## Repository layout

- `src/` — the six code files that run FluxNet.
- `fluxnet-repro/` — the saved FluxNet model, reference data, saved answer,
  test script, and records for the fixed test.
- `icon-repro/src/` — the six code files that run the fixed ICON model.
- `icon-repro/weights/` — three saved ICON model files from training step
  1,000,000.
- `icon-repro/truth/` — the reference data used by the comparison tests.
- `icon-repro/results/primary/` — saved comparison tables.
- `icon-repro/provenance/` and `icon-repro/environment/` — records showing
  which files and settings were used, and what the Vista computer had
  installed.

The FluxNet example uses
`fluxnet-repro/truth/task-011-chebyshev_degree_4_7-grf/saved.h5`. This data
file contains 16 stored sequences on a line of 100 cells. The test uses the
fourth sequence, takes its first 20 snapshots as input, and uses the next 20
snapshots as the known answer.

The old ICON result tables carry the label `post_unseal_exploratory` in their
records. We keep that label because it describes how those results were made.
They are comparison records, not a new final test. The fixed tests do not
recreate all of those tables.

The project is released under the Apache-2.0 license. The pinned ICON source
retains its own MIT license in `icon-repro/LICENSE`.
