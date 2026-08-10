"""
Fit and control the polynomial flux law used by FluxNet.

Purpose
-------
Turn flux values recovered from the prompt into polynomial coefficients. The
file also limits the largest slope of that polynomial, so the model does not
use a local wave speed above the configured safety limit.

Inputs
------
The fitting functions take normalized state values, recovered flux values,
valid-position masks, and the fit settings.

Outputs
-------
The fitting functions return nonconstant polynomial coefficients. The
polynomial evaluation helper returns values and slopes. The same code works
for state data with different sizes because the fit uses normalized values.
"""

from __future__ import annotations

import math

import torch

from chebyshev import (
    chebyshev_polynomial_derivatives,
    chebyshev_polynomials,
)
from validation import (
    _require_finite,
    _require_masked_finite,
    _require_same_dtype_and_device,
    _validate_floating_tensor,
    _validate_positive_finite_scalar,
)


def flux_derivative_envelope(coefficients: torch.Tensor) -> torch.Tensor:
    """
    Return a safe upper estimate of the polynomial slope.

    It is not the exact largest slope at one point. It is a cheap bound used
    to decide whether the coefficients need to be scaled down.

    Parameters
    ----------
    coefficients : torch.Tensor
        Nonconstant polynomial coefficients with shape ``(batch, degree)``.

    Returns
    -------
    torch.Tensor
        One nonnegative slope estimate for each batch item, with shape
        ``(batch,)``.
    """
    _validate_floating_tensor(coefficients, name="coefficients")
    if coefficients.ndim != 2 or coefficients.shape[-1] < 1:
        raise ValueError(f"coefficients must have shape (batch, degree>=1), got {tuple(coefficients.shape)}")
    _require_finite(coefficients, name="coefficients")
    orders = torch.arange(
        1,
        coefficients.shape[-1] + 1,
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    envelope = (coefficients.abs() * orders.square()).sum(dim=-1)
    _require_finite(envelope, name="flux derivative envelope")
    return envelope


def project_flux_coefficients(
    coefficients: torch.Tensor,
    speed_bound: float,
    epsilon: float,
) -> torch.Tensor:
    """
    Scale coefficients until the estimated slope is below ``speed_bound``.

    Scaling all nonconstant coefficients by one number keeps the shape of the
    fitted law while reducing its slope.  The extra correction passes protect
    against small floating-point errors near the limit.

    Parameters
    ----------
    coefficients : torch.Tensor
        Nonconstant polynomial coefficients with shape ``(batch, degree)``.
    speed_bound : float
        Largest allowed estimated slope.
    epsilon : float
        Small positive value used to keep divisions safe.

    Returns
    -------
    torch.Tensor
        Coefficients with the same shape as ``coefficients`` and an estimated
        slope at or below ``speed_bound``.
    """
    _validate_positive_finite_scalar(speed_bound, name="speed_bound")
    _validate_positive_finite_scalar(epsilon, name="epsilon")
    envelope = flux_derivative_envelope(coefficients)
    cap = torch.full_like(envelope, speed_bound)
    requires_projection = envelope > cap
    degree = coefficients.shape[-1]
    # Leave a very small margin so round-off does not place a result just over
    # the requested limit.
    relative_margin = min(0.25, (4 * degree + 4) * torch.finfo(coefficients.dtype).eps)
    safety_target = cap * (1.0 - relative_margin)
    safety_target = torch.minimum(
        safety_target,
        torch.nextafter(cap, torch.zeros_like(cap)),
    )
    factor = safety_target / envelope.clamp_min(epsilon)
    projected_candidate = coefficients * factor[:, None]
    projected = torch.where(
        requires_projection[:, None],
        projected_candidate,
        coefficients,
    )
    # Recheck after scaling.  The bound is an estimate, so two short correction
    # passes are safer than assuming the first scale factor is exact.
    for _ in range(2):
        recomputed = flux_derivative_envelope(projected)
        correction = torch.clamp(
            safety_target / recomputed.clamp_min(epsilon),
            max=1.0,
        )
        corrected = projected * correction[:, None]
        projected = torch.where(
            (requires_projection & (recomputed > cap))[:, None],
            corrected,
            projected,
        )
    _require_finite(projected, name="projected flux coefficients")
    return projected


def fit_flux_coefficients_from_evidence(
    normalized_midpoint: torch.Tensor,
    normalized_flux_evidence: torch.Tensor,
    evidence_mask: torch.Tensor,
    *,
    degree: int,
    ridge: float,
    speed_bound: float,
    epsilon: float,
) -> torch.Tensor:
    """
    Fit polynomial coefficients from normalized prompt observations.

    The constant part of the polynomial is removed because adding one
    constant to every boundary flux does not change a periodic state update.
    The fit is regularized so it remains stable when prompt observations are
    similar.

    Parameters
    ----------
    normalized_midpoint : torch.Tensor
        State values at the observed flux locations, after scaling, with shape
        ``(batch, positions...)``.
    normalized_flux_evidence : torch.Tensor
        Flux values at the same locations and with the same shape as
        ``normalized_midpoint``.
    evidence_mask : torch.Tensor
        Boolean mask that marks usable positions. It has the same shape as
        ``normalized_midpoint``.
    degree : int
        Number of nonconstant polynomial coefficients to fit.
    ridge : float
        Small nonnegative penalty that keeps the fit stable.
    speed_bound : float
        Largest allowed estimated slope of the fitted polynomial.
    epsilon : float
        Small positive value used to keep divisions safe.

    Returns
    -------
    torch.Tensor
        Fitted, projected nonconstant coefficients with shape
        ``(batch, degree)``.
    """
    _validate_floating_tensor(normalized_midpoint, name="normalized_midpoint")
    _validate_floating_tensor(
        normalized_flux_evidence,
        name="normalized_flux_evidence",
    )
    if normalized_midpoint.ndim < 2:
        raise ValueError(
            f"normalized_midpoint must have shape (batch, positions...), got {tuple(normalized_midpoint.shape)}"
        )
    if normalized_flux_evidence.shape != normalized_midpoint.shape:
        raise ValueError(
            "normalized_flux_evidence must match normalized_midpoint, got "
            f"{tuple(normalized_flux_evidence.shape)} and "
            f"{tuple(normalized_midpoint.shape)}"
        )
    if evidence_mask.shape != normalized_midpoint.shape or evidence_mask.dtype != torch.bool:
        raise ValueError(
            "evidence_mask must be boolean and match normalized_midpoint, got "
            f"{tuple(evidence_mask.shape)} and {evidence_mask.dtype}"
        )
    _require_same_dtype_and_device(
        normalized_midpoint,
        normalized_flux_evidence,
        names=("normalized_midpoint", "normalized_flux_evidence"),
    )
    if evidence_mask.device != normalized_midpoint.device:
        raise ValueError("evidence_mask and normalized_midpoint must share a device")
    if degree < 1:
        raise ValueError(f"degree must be positive, got {degree}")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError(f"ridge must be finite and nonnegative, got {ridge}")
    _validate_positive_finite_scalar(speed_bound, name="speed_bound")
    _validate_positive_finite_scalar(epsilon, name="epsilon")
    _require_masked_finite(
        normalized_midpoint[..., None],
        evidence_mask,
        name="normalized midpoint evidence",
    )
    _require_masked_finite(
        normalized_flux_evidence[..., None],
        evidence_mask,
        name="normalized flux evidence",
    )

    values, _ = extended_chebyshev_polynomials(normalized_midpoint, degree)
    zero_values, _ = extended_chebyshev_polynomials(
        torch.zeros_like(normalized_midpoint),
        degree,
    )
    design = values[..., 1:] - zero_values[..., 1:]
    batch_size = normalized_midpoint.shape[0]
    flat_design = design.reshape(batch_size, -1, degree)
    flat_evidence = normalized_flux_evidence.reshape(batch_size, -1)
    flat_mask = evidence_mask.reshape(batch_size, -1)
    counts = flat_mask.sum(dim=1)
    if not torch.compiler.is_compiling() and not torch.all(counts > 0).item():
        raise ValueError("every sample requires at least one valid flux-evidence position")
    weights = flat_mask.to(dtype=flat_design.dtype)
    safe_count = counts.clamp_min(1).to(dtype=flat_design.dtype)
    design_mean = (flat_design * weights[..., None]).sum(dim=1) / safe_count[:, None]
    evidence_mean = (flat_evidence * weights).sum(dim=1) / safe_count
    # Remove the mean from both sides.  This removes the unknown constant
    # boundary-flux level and makes the fitted law depend only on useful
    # changes in the flux.
    centered_design = (flat_design - design_mean[:, None]) * weights[..., None]
    centered_evidence = (flat_evidence - evidence_mean[:, None]) * weights
    gram = torch.einsum("bnd,bne->bde", centered_design, centered_design)
    right_hand_side = torch.einsum(
        "bnd,bn->bd",
        centered_design,
        centered_evidence,
    )
    identity = torch.eye(
        degree,
        dtype=gram.dtype,
        device=gram.device,
    )[None]
    # Solve one small ridge-regularized system for each batch item.
    coefficients = torch.linalg.solve(
        gram + ridge * identity,
        right_hand_side[..., None],
    )[..., 0]
    coefficients = project_flux_coefficients(
        coefficients,
        speed_bound,
        epsilon,
    )
    _require_finite(coefficients, name="fitted prompt flux coefficients")
    return coefficients


def extended_chebyshev_polynomials(
    x: torch.Tensor,
    degree: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return polynomial values and slopes, including values outside ``[-1, 1]``.

    Chebyshev polynomials can grow quickly outside their usual interval.  For
    rollout states outside that interval, FluxNet instead continues the value
    with the slope at the nearest endpoint.  This keeps the fitted law finite
    and makes the behavior outside the prompt range easy to understand.

    Parameters
    ----------
    x : torch.Tensor
        State values at which to evaluate the continued polynomial.
    degree : int
        Highest polynomial number to return. Must be nonnegative.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A pair containing polynomial values and slopes. Both tensors have
        shape ``x.shape + (degree + 1,)``.
    """
    _validate_floating_tensor(x, name="x")
    _require_finite(x, name="x")
    if degree < 0:
        raise ValueError(f"degree must be nonnegative, got {degree}")

    # Evaluate the normal polynomial only inside its stable range.
    clamped = x.clamp(min=-1.0, max=1.0)
    values = chebyshev_polynomials(clamped, degree)
    derivatives = chebyshev_polynomial_derivatives(clamped, degree)
    right_values = chebyshev_polynomials(torch.ones_like(x), degree)
    right_derivatives = chebyshev_polynomial_derivatives(torch.ones_like(x), degree)
    left_values = chebyshev_polynomials(-torch.ones_like(x), degree)
    left_derivatives = chebyshev_polynomial_derivatives(-torch.ones_like(x), degree)

    # Continue each endpoint with a straight line rather than allowing a
    # high-degree polynomial to grow without control outside the prompt range.
    values = torch.where(
        (x > 1.0)[..., None],
        right_values + right_derivatives * (x - 1.0)[..., None],
        values,
    )
    derivatives = torch.where((x > 1.0)[..., None], right_derivatives, derivatives)
    values = torch.where(
        (x < -1.0)[..., None],
        left_values + left_derivatives * (x + 1.0)[..., None],
        values,
    )
    derivatives = torch.where((x < -1.0)[..., None], left_derivatives, derivatives)
    _require_finite(values, name="extended Chebyshev values")
    _require_finite(derivatives, name="extended Chebyshev derivatives")
    return values, derivatives
