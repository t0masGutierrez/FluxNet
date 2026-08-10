"""
Run shared input checks for the FluxNet runtime.

Purpose
-------
Give clear errors when a value has the wrong shape, number type, device, or
contains an invalid number. The same rules apply to prompt data, fitted
coefficients, and later predictions.

Inputs
------
The checks take tensors, masks, numeric settings, and names used in error
messages.

Outputs
-------
The checks return nothing when the values are valid. They raise an error that
names the problem when a value is not valid.
"""

from __future__ import annotations

import math

import torch


def _validate_state_tensor(
    value: torch.Tensor,
    *,
    name: str,
    minimum_dimensions: int,
) -> None:
    """
    Validate a floating state tensor with cells and channels at the end.

    Parameters
    ----------
    value : torch.Tensor
        State tensor to check.
    name : str
        Name used in the error message.
    minimum_dimensions : int
        Smallest number of axes allowed for ``value``.

    Returns
    -------
    None
        This function returns nothing. It raises an error when ``value`` is
        not a valid state tensor.
    """
    _validate_floating_tensor(value, name=name)
    if value.ndim < minimum_dimensions:
        raise ValueError(f"{name} must have shape (..., cells, channels), got {tuple(value.shape)}")
    if value.shape[-2] < 1 or value.shape[-1] < 1:
        raise ValueError(f"{name} must have nonempty cells and channels, got {tuple(value.shape)}")


def _validate_floating_tensor(value: torch.Tensor, *, name: str) -> None:
    """
    Require a tensor with a floating-point dtype.

    Parameters
    ----------
    value : torch.Tensor
        Tensor to check.
    name : str
        Name used in the error message.

    Returns
    -------
    None
        This function returns nothing. It raises an error when ``value`` is
        not a floating-point tensor.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if not value.is_floating_point():
        raise ValueError(f"{name} must have a floating dtype, got {value.dtype}")


def _validate_mask(mask: torch.Tensor, value: torch.Tensor, *, name: str) -> None:
    """
    Require a boolean mask aligned with the value tensor.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean mask to check.
    value : torch.Tensor
        Tensor whose shape and device must match ``mask``.
    name : str
        Name used in the error message.

    Returns
    -------
    None
        This function returns nothing. It raises an error when ``mask`` does
        not align with ``value``.
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(mask).__name__}")
    if mask.dtype != torch.bool:
        raise ValueError(f"{name} must have boolean dtype, got {mask.dtype}")
    if mask.shape != value.shape[:-1]:
        raise ValueError(f"{name} mask must have shape {tuple(value.shape[:-1])}, got {tuple(mask.shape)}")
    if mask.device != value.device:
        raise ValueError(f"{name} and its values must be on the same device, got {mask.device} and {value.device}")


def _require_same_dtype_and_device(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    names: tuple[str, str],
) -> None:
    """
    Require two tensors to share dtype and device.

    Parameters
    ----------
    first : torch.Tensor
        First tensor to compare.
    second : torch.Tensor
        Second tensor to compare.
    names : tuple[str, str]
        Names of the tensors for error messages.

    Returns
    -------
    None
        This function returns nothing. It raises an error when the tensors do
        not share a number type or device.
    """
    if first.dtype != second.dtype:
        raise ValueError(f"{names[0]} and {names[1]} must have the same dtype, got {first.dtype} and {second.dtype}")
    if first.device != second.device:
        raise ValueError(
            f"{names[0]} and {names[1]} must be on the same device, got {first.device} and {second.device}"
        )


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    """
    Reject invalid numbers when Torch is running the code normally.

    Parameters
    ----------
    value : torch.Tensor
        Tensor to check.
    name : str
        Name used in the error message.

    Returns
    -------
    None
        This function returns nothing. It raises an error when ``value`` has
        an invalid number.
    """
    # ``.item()`` gives a useful error in normal use but cannot be used inside
    # a compiled graph, so this check is skipped while Torch builds that graph.
    if torch.compiler.is_compiling():
        return
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _require_masked_finite(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    name: str,
) -> None:
    """
    Check finite values only at positions selected by ``mask``.

    Parameters
    ----------
    value : torch.Tensor
        Tensor containing values to check.
    mask : torch.Tensor
        Boolean mask selecting valid positions.
    name : str
        Name used in the error message.

    Returns
    -------
    None
        This function returns nothing. It raises an error when a selected
        value is not finite.
    """
    if torch.compiler.is_compiling():
        return
    safe_value = torch.where(mask[..., None], value, torch.zeros_like(value))
    if not torch.isfinite(safe_value).all().item():
        raise ValueError(f"{name} must contain finite values at every valid position")


def _validate_dt_over_dx(dt_over_dx: float) -> None:
    """
    Require a positive finite time-to-space ratio.

    Parameters
    ----------
    dt_over_dx : float
        Time step divided by cell width.

    Returns
    -------
    None
        This function returns nothing. It raises an error when the ratio is
        not finite and positive.
    """
    if not math.isfinite(dt_over_dx) or dt_over_dx <= 0.0:
        raise ValueError(f"dt_over_dx must be finite and positive, got {dt_over_dx}")


def _validate_positive_finite_scalar(value: float, *, name: str) -> None:
    """
    Require a named scalar setting to be positive and finite.

    Parameters
    ----------
    value : float
        Number to check.
    name : str
        Name used in the error message.

    Returns
    -------
    None
        This function returns nothing. It raises an error when ``value`` is
        not finite and positive.
    """
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
