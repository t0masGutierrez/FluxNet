# ICON comparison check

This folder contains the fixed ICON code, three saved ICON models, the
reference data, saved answers, and comparison records. It lets you repeat the
saved ICON test.

From the project root, install the extra packages needed for this test:

```sh
python -m pip install -e '.[comparison]'
```

Run all three saved models:

```sh
python icon-repro/run_example.py
```

To run one saved model, choose its run number:

```sh
python icon-repro/run_example.py --seed 0
```

For each selected run, the script checks the model file and where it was saved
during training, loads the fixed ICON code and settings, makes 20 predictions,
and compares them with the saved answer. Small differences between different
types of computers are allowed. The new files named
`reproduced-seed-*.npz` are ignored by Git.

The bundle is organized as follows:

- `weights/` — three saved model files from training step 1,000,000.
- `src/` — the six code files used by the fixed ICON test. They come from
  code version `f8404c0edca7cca7f767c3562ade51d66b222d5d`.
- `config/resolved.yaml` — the settings used for the test.
- `truth/` — 16 sets of reference data, including the task-011 example used
  by the test.
- `example/` — saved answers for the three runs and a summary of the results.
- `results/primary/` — saved comparison tables.
- `provenance/` — records showing which code, model files, data, settings, and
  results were used.
- `environment/` — records of the Vista computer and installed packages.
- `LICENSE` — the license for the ICON code.

The old result files carry the label `post_unseal_exploratory` in their
records. We keep that label because it describes how those results were made.
They are comparison records, not a new final test.

For the saved dependency versions, use `uv sync --locked --extra comparison`.
The lock file does not replace the saved Vista environment records.

Use `--output-dir` to store new forecasts in another folder. The check always
reads saved answers from this bundle's `example/` folder, and fails if a saved
answer is missing.
