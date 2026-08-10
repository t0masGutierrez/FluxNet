"""
Turn example state changes into the information used by FluxNet.

Purpose
-------
Check the clean examples, put them on a common scale, and build the feature
rows read by the prompt encoder. Each example has a state before a time step
(``demo_cond_v``), a state after that step (``demo_qoi_v``), and one true/false
mask for each state.

Inputs
------
The functions take the demonstration states, their masks, the time step
divided by cell width, and the prompt normalization settings.

Outputs
-------
The file returns prompt center and scale values, eight-value feature rows,
usable-position masks, and temporary flux observations. The temporary flux
observations fit the polynomial law once at the start of a rollout.
"""

from __future__ import annotations

import torch

from flux_inference import infer_periodic_interface_flux
from validation import (
    _require_finite,
    _require_masked_finite,
    _require_same_dtype_and_device,
    _validate_dt_over_dx,
    _validate_floating_tensor,
    _validate_mask,
    _validate_positive_finite_scalar,
    _validate_state_tensor,
)


def infer_masked_periodic_flux_evidence(
    cond: torch.Tensor,
    qoi: torch.Tensor,
    cond_mask: torch.Tensor,
    qoi_mask: torch.Tensor,
    dt_over_dx: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recover centered boundary fluxes and mark usable prompt positions.

    A position is usable only when both its before-state and after-state are
    present.  Missing values are replaced by zero before the inverse step so
    they cannot create invalid numbers.  The returned mask tells later code
    which entries came from complete examples.

    Parameters
    ----------
    cond : torch.Tensor
        State before each example step, with shape
        ``(batch, examples, cells, 1)``.
    qoi : torch.Tensor
        State after each example step, with the same shape as ``cond``.
    cond_mask : torch.Tensor
        Boolean mask for valid entries in ``cond``, without the channel axis.
    qoi_mask : torch.Tensor
        Boolean mask for valid entries in ``qoi``, without the channel axis.
    dt_over_dx : float
        Time step divided by cell width. Must be positive.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Centered interface-flux evidence and a boolean mask that marks usable
        cells. The evidence has the state shape and the mask has its last
        channel axis removed.
    """
    _validate_state_tensor(cond, name="cond", minimum_dimensions=3)
    _validate_state_tensor(qoi, name="qoi", minimum_dimensions=3)
    if cond.shape != qoi.shape:
        raise ValueError(f"cond and qoi must have the same shape, got {tuple(cond.shape)} and {tuple(qoi.shape)}")
    _require_same_dtype_and_device(cond, qoi, names=("cond", "qoi"))
    _validate_dt_over_dx(dt_over_dx)
    _validate_mask(cond_mask, cond, name="cond_mask")
    _validate_mask(qoi_mask, qoi, name="qoi_mask")
    _require_masked_finite(cond, cond_mask, name="cond")
    _require_masked_finite(qoi, qoi_mask, name="qoi")

    # Missing values are never used to infer a flux.  Replacing them with zero
    # lets the tensor operation run without allowing missing data to spread.
    safe_cond = torch.where(cond_mask[..., None], cond, torch.zeros_like(cond))
    safe_qoi = torch.where(qoi_mask[..., None], qoi, torch.zeros_like(qoi))
    centered_flux = infer_periodic_interface_flux(safe_cond, safe_qoi, dt_over_dx)

    # A whole state change is useful only when every cell is present in both
    # states.  One missing cell would make its recovered boundary values
    # ambiguous, so the mask is expanded back to all cells in that example.
    complete_transition = (cond_mask & qoi_mask).all(dim=-1, keepdim=True)
    evidence_mask = complete_transition.expand_as(cond_mask)
    _require_masked_finite(
        centered_flux,
        evidence_mask,
        name="centered interface-flux evidence",
    )
    evidence = torch.where(
        evidence_mask[..., None],
        centered_flux,
        torch.zeros_like(centered_flux),
    )
    _require_finite(evidence, name="interface-flux evidence")
    return evidence, evidence_mask


def compute_prompt_normalization(
    demo_cond: torch.Tensor,
    demo_qoi: torch.Tensor,
    cond_mask: torch.Tensor,
    qoi_mask: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the center and half-range used to scale prompt states.

    The center is the midpoint between the smallest and largest valid state
    value. The scale is half of that range, with a small lower limit for nearly
    constant data.

    Parameters
    ----------
    demo_cond : torch.Tensor
        State before each example step, with shape
        ``(batch, examples, cells, 1)``.
    demo_qoi : torch.Tensor
        State after each example step, with the same shape as ``demo_cond``.
    cond_mask : torch.Tensor
        Boolean mask for valid entries in ``demo_cond``.
    qoi_mask : torch.Tensor
        Boolean mask for valid entries in ``demo_qoi``.
    epsilon : float
        Small positive lower limit for the returned scale.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Center and positive half-range, each with shape ``(batch, 1, 1)``.
    """
    _validate_demo_tensors(
        demo_cond,
        demo_qoi,
        cond_mask,
        qoi_mask,
        require_nineteen=False,
    )
    _validate_positive_finite_scalar(epsilon, name="epsilon")
    _require_masked_finite(demo_cond, cond_mask, name="demo_cond")
    _require_masked_finite(demo_qoi, qoi_mask, name="demo_qoi")

    # Use both sides of every example.  This gives one scale for all prompt
    # states and avoids letting one direction of the transition set the range.
    states = torch.cat([demo_cond, demo_qoi], dim=1)
    valid = torch.cat([cond_mask, qoi_mask], dim=1)[..., None]
    has_valid = valid.any(dim=(1, 2, 3))
    if not torch.compiler.is_compiling() and not has_valid.all().item():
        invalid = torch.nonzero(~has_valid, as_tuple=False).flatten().tolist()
        raise ValueError(f"every sample requires at least one valid prompt state, invalid batches={invalid}")
    upper = torch.full(
        (),
        torch.finfo(states.dtype).max,
        dtype=states.dtype,
        device=states.device,
    )
    lower = torch.full(
        (),
        torch.finfo(states.dtype).min,
        dtype=states.dtype,
        device=states.device,
    )
    state_min = torch.where(valid, states, upper).amin(dim=(1, 2))
    state_max = torch.where(valid, states, lower).amax(dim=(1, 2))
    center = 0.5 * state_min + 0.5 * state_max
    scale = (0.5 * (state_max - state_min)).clamp_min(epsilon)
    center = center[:, None]
    scale = scale[:, None]
    _require_finite(center, name="prompt center")
    _require_finite(scale, name="prompt scale")
    return center, scale


def build_flux_evidence_tokens(
    data: dict[str, torch.Tensor],
    center: torch.Tensor,
    scale: torch.Tensor,
    dt_over_dx: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the feature rows read by the prompt encoder.

    The eight features are, in order: previous state, current state, next
    state, current/next average, state jump, observed rate of change, recovered
    boundary flux, and ``dt/dx``. The mask keeps rows with complete local data.

    Parameters
    ----------
    data : dict[str, torch.Tensor]
        Dictionary containing the 19 demonstration states and masks.
    center : torch.Tensor
        Prompt state center with shape ``(batch, 1, 1)``.
    scale : torch.Tensor
        Positive prompt state half-range with shape ``(batch, 1, 1)``.
    dt_over_dx : float
        Time step divided by cell width. Must be positive.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Eight-value feature rows with shape ``(batch, 19, cells, 8)`` and a
        boolean mask with shape ``(batch, 19, cells)``.
    """
    demo_cond, demo_qoi, cond_mask, qoi_mask = _demo_tensors_from_data(data)
    _validate_program_statistics(center, scale, demo_cond)
    _validate_dt_over_dx(dt_over_dx)
    # Add the example axis back before broadcasting center and scale over all
    # cells in each example.
    center_for_demo = center[:, None]
    scale_for_demo = scale[:, None]
    safe_cond = torch.where(cond_mask[..., None], demo_cond, center_for_demo)
    safe_qoi = torch.where(qoi_mask[..., None], demo_qoi, center_for_demo)
    normalized = (safe_cond - center_for_demo) / scale_for_demo
    # Periodic rolls make the first and last cells neighbors, as required by
    # the one-dimensional periodic problem used by this model.
    normalized_next = torch.roll(normalized, shifts=-1, dims=2)
    normalized_previous = torch.roll(normalized, shifts=1, dims=2)
    flux_evidence, evidence_mask = infer_masked_periodic_flux_evidence(
        demo_cond,
        demo_qoi,
        cond_mask,
        qoi_mask,
        dt_over_dx,
    )
    # Keep this order fixed: the prompt encoder was trained with these eight
    # meanings at these eight positions.
    features = torch.cat(
        [
            normalized_previous,
            normalized,
            normalized_next,
            0.5 * (normalized + normalized_next),
            normalized_next - normalized,
            (safe_qoi - safe_cond) / (scale_for_demo * dt_over_dx),
            flux_evidence / scale_for_demo,
            torch.full_like(normalized, dt_over_dx),
        ],
        dim=-1,
    )
    # The neighbor checks prevent a feature row from using a missing adjacent
    # cell when it describes local motion.
    token_mask = (
        evidence_mask
        & cond_mask
        & torch.roll(cond_mask, shifts=1, dims=2)
        & torch.roll(cond_mask, shifts=-1, dims=2)
        & qoi_mask
    )
    features = torch.where(token_mask[..., None], features, torch.zeros_like(features))
    _require_finite(features, name="flux evidence tokens")
    return features, token_mask


def _validate_demo_tensors(
    demo_cond: torch.Tensor,
    demo_qoi: torch.Tensor,
    cond_mask: torch.Tensor,
    qoi_mask: torch.Tensor,
    *,
    require_nineteen: bool,
) -> None:
    """
    Check the shape, type, and masks of the demonstration tensors.

    Parameters
    ----------
    demo_cond : torch.Tensor
        State before each example step.
    demo_qoi : torch.Tensor
        State after each example step.
    cond_mask : torch.Tensor
        Boolean mask for valid entries in ``demo_cond``.
    qoi_mask : torch.Tensor
        Boolean mask for valid entries in ``demo_qoi``.
    require_nineteen : bool
        Whether to require the fixed 19 examples used by FluxNet.

    Returns
    -------
    None
        This function returns nothing. It raises an error when the inputs do
        not meet the expected contract.
    """
    _validate_state_tensor(demo_cond, name="demo_cond", minimum_dimensions=4)
    _validate_state_tensor(demo_qoi, name="demo_qoi", minimum_dimensions=4)
    if demo_cond.ndim != 4 or demo_cond.shape[-1] != 1:
        raise ValueError(f"demo_cond must have shape (batch, demos, cells, channels=1), got {tuple(demo_cond.shape)}")
    if demo_qoi.shape != demo_cond.shape:
        raise ValueError(f"demo_qoi must have shape {tuple(demo_cond.shape)}, got {tuple(demo_qoi.shape)}")
    _require_same_dtype_and_device(demo_cond, demo_qoi, names=("demo_cond", "demo_qoi"))
    _validate_mask(cond_mask, demo_cond, name="cond_mask")
    _validate_mask(qoi_mask, demo_qoi, name="qoi_mask")
    if require_nineteen and demo_cond.shape[1] != 19:
        raise ValueError(f"FluxNet requires exactly 19 demo transitions, got {demo_cond.shape[1]}")


def _demo_tensors_from_data(
    data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Read and validate the four prompt tensors used by FluxNet.

    Parameters
    ----------
    data : dict[str, torch.Tensor]
        Input dictionary containing the demonstration fields.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Demonstration before-states, after-states, and their two masks.
    """
    required = ("demo_cond_v", "demo_qoi_v", "demo_cond_mask", "demo_qoi_mask")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"missing required prompt tensors: {missing}")
    demo_cond = data["demo_cond_v"]
    demo_qoi = data["demo_qoi_v"]
    cond_mask = data["demo_cond_mask"]
    qoi_mask = data["demo_qoi_mask"]
    _validate_demo_tensors(demo_cond, demo_qoi, cond_mask, qoi_mask, require_nineteen=True)
    return demo_cond, demo_qoi, cond_mask, qoi_mask


def _validate_program_statistics(
    center: torch.Tensor,
    scale: torch.Tensor,
    demo_cond: torch.Tensor,
) -> None:
    """
    Check that prompt center and scale match the demonstration batch.

    Parameters
    ----------
    center : torch.Tensor
        Prompt state center with the expected shape ``(batch, 1, 1)``.
    scale : torch.Tensor
        Prompt state half-range with the expected shape ``(batch, 1, 1)``.
    demo_cond : torch.Tensor
        Demonstration before-states used to determine the batch and number
        type that ``center`` and ``scale`` must match.

    Returns
    -------
    None
        This function returns nothing. It raises an error when the statistics
        do not match the demonstration batch.
    """
    expected = (demo_cond.shape[0], 1, 1)
    if center.shape != expected or scale.shape != expected:
        raise ValueError(
            f"center and scale must both have shape {expected}, got {tuple(center.shape)} and {tuple(scale.shape)}"
        )
    _validate_floating_tensor(center, name="center")
    _validate_floating_tensor(scale, name="scale")
    _require_same_dtype_and_device(center, demo_cond, names=("center", "demo_cond"))
    _require_same_dtype_and_device(scale, demo_cond, names=("scale", "demo_cond"))
    _require_finite(center, name="center")
    _require_finite(scale, name="scale")
    if not torch.compiler.is_compiling() and not torch.all(scale > 0.0).item():
        raise ValueError("scale must be strictly positive")
