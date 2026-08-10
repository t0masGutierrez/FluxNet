#!/usr/bin/env python3
"""Reproduce a fixed 20-step ICON poster forecast for each training seed."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

BUNDLE = Path(__file__).resolve().parent
SOURCE = BUNDLE / "src"
CONFIG = BUNDLE / "config/resolved.yaml"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

os.environ.setdefault("MPLBACKEND", "Agg")

from src import data_utils, model_iconlm  # noqa: E402

CHECKPOINT_SHA256 = {
    0: "b592413d90d9d5ded6f4667f80cee365e6e850ea0ca93d3245474766160df44a",
    1: "98d557db718c56d2e8f5c3ca0b7ddabd3086f739bf7f694f7862c1b97718d442",
    2: "f27c43d896620622de5aaf4fb631493870e5d6972bbe1e37aab7413212c2cfb0",
}
CHECKPOINT_STEP = 1_000_000
SOURCE_COMMIT = "f8404c0edca7cca7f767c3562ade51d66b222d5d"
PARAMETER_COUNT = 2_681_473
CONTEXT_LENGTH = 20
HORIZON = 20
TRAJECTORY_KEY = 3
WINDOW_START = 0
EXPECTED_DT = 0.005
EXPECTED_DX = 0.01


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed poster-example command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=tuple(CHECKPOINT_SHA256))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BUNDLE / "example",
    )
    return parser


def sha256(path: Path) -> str:
    """Return the SHA-256 fingerprint of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixed_example(
    truth_path: Path,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Load the fixed trajectory window used by the poster example."""
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
            source["t"][
                WINDOW_START + CONTEXT_LENGTH : WINDOW_START + CONTEXT_LENGTH + HORIZON
            ],
            dtype=np.float64,
        )

    if values.shape != (CONTEXT_LENGTH + HORIZON, x.size):
        raise ValueError(f"unexpected truth shape: {values.shape}")
    dt = float(time[1] - time[0])
    dx = float(x[1] - x[0])
    if not np.isclose(dt, EXPECTED_DT) or not np.isclose(dx, EXPECTED_DX):
        raise ValueError(f"unexpected coordinates: dt={dt}, dx={dx}")

    observed = torch.from_numpy(values[:CONTEXT_LENGTH, :, None])
    reference = values[CONTEXT_LENGTH:]
    return observed, reference, x, time, dt, dx


def make_model_input(observed: torch.Tensor, x: np.ndarray) -> data_utils.Data:
    """Build the fixed-prompt query expected by the official ICON model."""
    batch_observed = observed[None]
    coordinates = torch.as_tensor(x, dtype=observed.dtype).reshape(1, 1, -1, 1)
    coordinates = coordinates.expand(1, CONTEXT_LENGTH - 1, -1, -1)
    query_coordinates = coordinates[:, :1]
    demo_mask = torch.ones(
        (1, CONTEXT_LENGTH - 1, observed.shape[1]),
        dtype=torch.bool,
    )
    query_mask = torch.ones(
        (1, 1, observed.shape[1]),
        dtype=torch.bool,
    )
    return data_utils.Data(
        equation=("withheld",),
        param=torch.empty((1, 0), dtype=observed.dtype),
        demo_cond_k=coordinates,
        demo_cond_v=batch_observed[:, :-1],
        demo_cond_mask=demo_mask,
        demo_qoi_k=coordinates,
        demo_qoi_v=batch_observed[:, 1:],
        demo_qoi_mask=demo_mask,
        quest_cond_k=query_coordinates,
        quest_cond_v=batch_observed[:, -1:],
        quest_cond_mask=query_mask,
        quest_qoi_k=query_coordinates,
        quest_qoi_mask=query_mask,
    )


def load_model(seed: int, device: torch.device) -> torch.nn.Module:
    """Load one exact poster checkpoint with strict parameter matching."""
    checkpoint_path = BUNDLE / f"weights/seed-{seed}-step_1000000.pt"
    actual_sha256 = sha256(checkpoint_path)
    if actual_sha256 != CHECKPOINT_SHA256[seed]:
        raise ValueError(
            f"seed {seed} checkpoint fingerprint does not match: {actual_sha256}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint.get("global_step", -1)) != CHECKPOINT_STEP:
        raise ValueError(f"seed {seed} checkpoint update does not match 1,000,000")

    config = OmegaConf.load(CONFIG)
    dummy, _ = data_utils.build_dummy_data(
        batch_size=1,
        demo_num=19,
        seq_len=100,
        k_dim=1,
        v_dim=1,
    )
    with tempfile.TemporaryDirectory(prefix="icon-repro-") as temporary:
        previous_directory = Path.cwd()
        os.chdir(temporary)
        try:
            model = model_iconlm.ICON_LM(
                config.model,
                dummy.get_shape(),
                config.loss.shot_num_min,
            )
        finally:
            os.chdir(previous_directory)

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"seed {seed} checkpoint state_dict is missing")
    model.load_state_dict(state_dict, strict=True)
    actual_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameter_count != PARAMETER_COUNT:
        raise ValueError(f"unexpected parameter count: {actual_parameter_count}")
    model.to(device=device)
    model.eval()
    return model


def rollout(
    model: torch.nn.Module,
    observed: torch.Tensor,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run the fixed-prompt recurrent forecast."""
    data = make_model_input(observed, x).to(device)
    with torch.no_grad():
        query_state = data.quest_cond_v[:, 0]
        predictions: list[np.ndarray] = []
        for _ in range(HORIZON):
            data.quest_cond_v = query_state[:, None]
            query_state = model(data, "test")
            predictions.append(query_state[0].detach().cpu().numpy())
    return np.stack(predictions).astype(np.float32, copy=False)[..., 0]


def save_result(
    output_path: Path,
    prediction: np.ndarray,
    reference: np.ndarray,
    x: np.ndarray,
    time: np.ndarray,
) -> None:
    """Save the forecast arrays in the poster artifact format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        prediction=prediction,
        reference=reference,
        absolute_error=np.abs(prediction - reference),
        x=x,
        time=time,
    )


def run_seed(
    seed: int,
    *,
    output_dir: Path,
    device: torch.device,
    atol: float,
) -> None:
    """Reproduce and check one seed's fixed forecast."""
    truth_path = BUNDLE / "truth/task-011-chebyshev_degree_4_7-grf/saved.h5"
    observed, reference, x, time, dt, dx = load_fixed_example(truth_path)
    model = load_model(seed, device)
    prediction = rollout(model, observed, x, device)

    expected_path = output_dir / f"seed-{seed}-forecast-example.npz"
    output_path = output_dir / f"reproduced-seed-{seed}-forecast.npz"
    save_result(output_path, prediction, reference, x, time)
    if expected_path.is_file():
        expected = np.load(expected_path)
        reference_diff = float(np.max(np.abs(reference - expected["reference"])))
        prediction_diff = float(np.max(np.abs(prediction - expected["prediction"])))
        if reference_diff > atol or prediction_diff > atol:
            raise ValueError(
                f"seed {seed} differs from the stored forecast: "
                f"reference={reference_diff}, prediction={prediction_diff}, atol={atol}"
            )
        print(f"seed={seed} reference_max_abs_diff={reference_diff:.8g}")
        print(f"seed={seed} prediction_max_abs_diff={prediction_diff:.8g}")

    print(f"seed={seed} checkpoint_sha256={CHECKPOINT_SHA256[seed]}")
    print(f"seed={seed} checkpoint_update={CHECKPOINT_STEP}")
    print(f"seed={seed} parameter_count={PARAMETER_COUNT}")
    print(f"seed={seed} source_commit={SOURCE_COMMIT}")
    print(f"seed={seed} dt={dt:g} dx={dx:g}")
    print(f"seed={seed} output={output_path}")


def main() -> int:
    """Load the pinned checkpoints and reproduce the fixed examples."""
    args = build_parser().parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    device = torch.device(args.device)
    seeds = (args.seed,) if args.seed is not None else tuple(CHECKPOINT_SHA256)
    for seed in seeds:
        run_seed(seed, output_dir=args.output_dir, device=device, atol=args.atol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
