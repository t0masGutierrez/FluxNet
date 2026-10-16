#!/usr/bin/env python3
"""
Run the fixed FluxNet forecast stored with the poster evidence.

Purpose
-------
Check the saved checkpoint, load one selected truth window, build the input
dictionary expected by ``src/fluxnet.py``, and run 20 learned updates. The
script then compares the new arrays with the saved poster arrays.

Inputs
------
The command-line options select the checkpoint, truth file, stored expected
result, output file, device, and comparison tolerance.

Outputs
-------
The script writes one compressed result file and prints the comparison values.
It checks one exact example. It does not retrain the model or run the full
evaluation stored elsewhere in the reproduction folder.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

BUNDLE = Path(__file__).resolve().parent
SOURCE = BUNDLE.parent / "src"
# The source folder is deliberately flat.  Adding it directly to the import
# path keeps the six runtime files usable without a package setup step.
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from fluxnet import FluxNet  # noqa: E402

CHECKPOINT_SHA256 = "58cc13c189c27b37bc56551a08797335e4b71d07e5fed85bd6f5778ef26cf2f6"
CHECKPOINT_UPDATE = 37_500
TRAINING_SEED = 1307
CONTEXT_LENGTH = 20
HORIZON = 20
TRAJECTORY_KEY = 3
WINDOW_START = 0
EXPECTED_DT = 0.005
EXPECTED_DX = 0.01


def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line options for the fixed poster example.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the checkpoint, truth, expected-result, output, device,
        and comparison-tolerance options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BUNDLE / "weights/step_37500.ckpt",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=BUNDLE / "truth/task-011-chebyshev_degree_4_7-grf/saved.h5",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=BUNDLE / "example/forecast-example.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BUNDLE / "example/reproduced-forecast-example.npz",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=2.0e-5)
    return parser


def sha256(path: Path) -> str:
    """
    Return the file fingerprint used to identify saved evidence.

    Parameters
    ----------
    path : pathlib.Path
        File whose contents should be fingerprinted.

    Returns
    -------
    str
        Lowercase SHA-256 fingerprint of the file contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint_path: Path, device: torch.device) -> FluxNet:
    """
    Load the saved poster model and check its expected training step.

    The saved state uses the older internal name ``net``.  The loop below
    removes that wrapper name before strict loading into the public ``FluxNet``
    class.  Strict loading is important: it catches missing or unexpected
    learned values instead of silently running a partly loaded model.

    Parameters
    ----------
    checkpoint_path : pathlib.Path
        Saved checkpoint to load.
    device : torch.device
        Device on which to place the loaded model.

    Returns
    -------
    FluxNet
        Evaluation-mode model loaded with the fixed poster weights.
    """
    actual_sha256 = sha256(checkpoint_path)
    if actual_sha256 != CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint fingerprint does not match the poster checkpoint: {actual_sha256}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint.get("global_step", -1)) != CHECKPOINT_UPDATE:
        raise ValueError("checkpoint update does not match 37,500")

    model = FluxNet(
        dt_over_dx=0.5,
        fit_degree=3,
        speed_bound=20.0,
        normalization_epsilon=1.0e-5,
        evidence_ridge=1.0e-4,
        output_scaling="none",
        dilations=[1, 2, 4, 8],
        width=168,
    )
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, dict):
        raise ValueError("checkpoint state_dict is missing")

    model_state: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        # ``torch.compile`` may add ``_orig_mod`` to the saved name.  Accept
        # both saved forms, but always load the same inner model names.
        if name.startswith("net._orig_mod."):
            model_state[name.removeprefix("net._orig_mod.")] = value
        elif name.startswith("net."):
            model_state[name.removeprefix("net.")] = value

    model.load_state_dict(model_state, strict=True)
    model.to(device=device)
    model.eval()
    return model


def load_fixed_example(
    truth_path: Path,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Load the 20 observed and 20 reference states for the poster example.

    The observed values seed the model. The reference values remain separate
    for comparison with the reproduced forecast.

    Parameters
    ----------
    truth_path : pathlib.Path
        HDF5 file containing the saved truth trajectory and coordinates.

    Returns
    -------
    tuple[torch.Tensor, numpy.ndarray, numpy.ndarray, numpy.ndarray, float, float]
        Observed states, reference future states, cell coordinates, future
        times, time step, and cell width.
    """
    with h5py.File(truth_path, "r") as source:
        values = np.asarray(
            source["values"][
                TRAJECTORY_KEY,
                WINDOW_START : WINDOW_START + CONTEXT_LENGTH + HORIZON,
                0,
                :,
            ],
            dtype=np.float32,
        )
        x = np.asarray(source["x"][:], dtype=np.float64)
        time = np.asarray(
            source["t"][WINDOW_START + CONTEXT_LENGTH : WINDOW_START + CONTEXT_LENGTH + HORIZON],
            dtype=np.float64,
        )

    if values.shape != (CONTEXT_LENGTH + HORIZON, x.size):
        raise ValueError(f"unexpected truth shape: {values.shape}")
    dt = float(time[1] - time[0])
    dx = float(x[1] - x[0])
    if not np.isclose(dt, EXPECTED_DT) or not np.isclose(dx, EXPECTED_DX):
        raise ValueError(f"unexpected coordinates: dt={dt}, dx={dx}")

    # The model expects one final channel axis, even though this example has
    # only one physical state value per cell.
    observed = torch.from_numpy(values[:CONTEXT_LENGTH, :, None])
    reference = values[CONTEXT_LENGTH:]
    return observed, reference, x, time, dt, dx


def make_model_input(observed: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Build the prompt and query fields from the 20 observed states.

    The first 19 state changes become demonstrations.  The last observed state
    becomes the query state and must match the last demonstration output.
    Every entry is marked valid because this fixed example has no missing
    cells.

    Parameters
    ----------
    observed : torch.Tensor
        Observed states with shape ``(context, cells, 1)``.

    Returns
    -------
    dict[str, torch.Tensor]
        Prompt and query fields accepted by :class:`fluxnet.FluxNet`.
    """
    batch_observed = observed[None]
    demo_mask = torch.ones(
        (1, CONTEXT_LENGTH - 1, observed.shape[1]),
        dtype=torch.bool,
    )
    query_mask = torch.ones(
        (1, 1, observed.shape[1]),
        dtype=torch.bool,
    )
    return {
        "demo_cond_v": batch_observed[:, :-1],
        "demo_qoi_v": batch_observed[:, 1:],
        "demo_cond_mask": demo_mask,
        "demo_qoi_mask": demo_mask,
        "quest_cond_v": batch_observed[:, -1:],
        "quest_cond_mask": query_mask,
    }


def rollout(
    model: FluxNet,
    observed: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Run 20 updates while reusing one prompt program.

    Parameters
    ----------
    model : FluxNet
        Loaded FluxNet model in evaluation mode.
    observed : torch.Tensor
        Observed states used to build the prompt and the starting query.
    device : torch.device
        Device on which to run the model.

    Returns
    -------
    numpy.ndarray
        Twenty predicted states as a floating-point array with shape
        ``(20, cells)``.
    """
    data = {name: value.to(device=device) for name, value in make_model_input(observed).items()}
    with torch.no_grad():
        program = model.prepare_rollout(data)
        state = observed[-1:].to(device=device)
        predictions: list[np.ndarray] = []
        # The prompt is read once.  Each new state is then fed to the same
        # learned update, which is how the poster forecast was produced.
        for _ in range(HORIZON):
            state = model.step_prepared(state, program)
            predictions.append(state[0, :, 0].detach().cpu().numpy())
    return np.stack(predictions).astype(np.float32, copy=False)


def save_result(
    output_path: Path,
    prediction: np.ndarray,
    reference: np.ndarray,
    x: np.ndarray,
    time: np.ndarray,
) -> None:
    """
    Save prediction, truth, error, coordinates, and times in one file.

    Parameters
    ----------
    output_path : pathlib.Path
        Destination ``.npz`` file.
    prediction : numpy.ndarray
        Reproduced model states.
    reference : numpy.ndarray
        Saved truth states for the same future times.
    x : numpy.ndarray
        Cell coordinates.
    time : numpy.ndarray
        Future time values.

    Returns
    -------
    None
        This function writes the compressed result file and returns nothing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        prediction=prediction,
        reference=reference,
        absolute_error=np.abs(prediction - reference),
        x=x,
        time=time,
    )


def main() -> int:
    """
    Run all checks and return a shell exit code.

    Returns
    -------
    int
        Zero when the fixed example matches the saved poster result.
    """
    args = build_parser().parse_args()
    if not np.isfinite(args.atol) or args.atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    if not str(args.output).endswith(".npz"):
        args.output = Path(str(args.output) + ".npz")
    protected = (args.expected, args.truth, args.checkpoint, BUNDLE / "example/forecast-example.npz")
    if args.output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("output must be separate from saved input files")
    if not args.expected.is_file():
        raise FileNotFoundError(args.expected)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(TRAINING_SEED)
    device = torch.device(args.device)

    observed, reference, x, time, dt, dx = load_fixed_example(args.truth)
    model = load_model(args.checkpoint, device)
    prediction = rollout(model, observed, device)
    save_result(args.output, prediction, reference, x, time)

    with np.load(args.expected) as expected:
        # Check shape and finite values as well as the original absolute limit.
        for name, actual in (("reference", reference), ("prediction", prediction)):
            saved = expected[name]
            if not np.isfinite(actual).all() or not np.isfinite(saved).all():
                raise ValueError(f"{name} must contain only finite values")
            np.testing.assert_allclose(actual, saved, rtol=0, atol=args.atol)
            difference = float(np.max(np.abs(actual - saved)))
            print(f"{name}_max_abs_diff={difference:.8g}")

    print(f"checkpoint_sha256={sha256(args.checkpoint)}")
    print(f"checkpoint_update={CHECKPOINT_UPDATE}")
    print(f"dt={dt:g} dx={dx:g}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
