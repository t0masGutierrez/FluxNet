# Fixed FluxNet check

This folder holds the files for one repeatable FluxNet test. It does not train
FluxNet or run the full set of tests.

From the repository root, install the project and run:

```sh
python -m pip install -e .
python fluxnet-repro/run_example.py
```

The script does five things:

1. checks the saved model file and where it was saved during training;
2. reads the fixed reference data;
3. uses 20 snapshots as the model input;
4. makes 20 predictions; and
5. compares those predictions with the saved answer.

The bundled run reports:

```text
reference_max_abs_diff=0
prediction_max_abs_diff=1.1920929e-06
checkpoint_update=37500
dt=0.005 dx=0.01
```

The reference data file is
`truth/task-011-chebyshev_degree_4_7-grf/saved.h5`. It contains 16 stored
sequences, 101 saved times, and one value at each of 100 cells. The test uses
the fourth sequence, takes the first 20 snapshots as input, and compares the
next 20 snapshots with the FluxNet forecast.

The main files are:

- `weights/step_37500.ckpt` — the saved FluxNet model used by the test.
- `truth/.../saved.h5` — the saved reference data.
- `example/forecast-example.npz` — the saved answer and reference values.
- `example/forecast-example-provenance.json` — records showing which sequence
  and files were used.
- `manifest.json` — file fingerprints and test results.

The test writes `example/reproduced-forecast-example.npz`. Git ignores this
new file because the test can create it again.
