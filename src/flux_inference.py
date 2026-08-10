"""
Recover cell-boundary flux information from observed state changes.

Purpose
-------
Work backward from an observed state change to make a centered flux profile.
The profile helps fit the polynomial law before the model starts its rollout.

Inputs
------
The main function takes a state before a time step, a state after the time
step, and time step divided by cell width. The state tensors must have cells
and channels at the end.

Outputs
-------
The function returns centered flux values with the same shape as the input
states. This is an inference helper, not a numerical solver used during a
FluxNet rollout.
"""

from __future__ import annotations

import torch


def infer_periodic_interface_flux(
    cond: torch.Tensor,
    qoi: torch.Tensor,
    dt_over_dx: float,
) -> torch.Tensor:
    """
    Infer a centered periodic interface-flux profile from a transition.

    The state change tells us only the difference between nearby boundary
    fluxes. Adding the same value to every boundary would give the same state
    change, so the result uses a centered profile.

    Parameters
    ----------
    cond : torch.Tensor
        State before the observed time step, with shape
        ``(..., cells, channels)``.
    qoi : torch.Tensor
        State after the observed time step, with the same shape as ``cond``.
    dt_over_dx : float
        Time step divided by cell width. Must be positive.

    Returns
    -------
    torch.Tensor
        Centered interface-flux values with the same shape as ``cond``. The
        spatial mean is zero for every sample and channel.
    """
    if cond.shape != qoi.shape:
        raise ValueError(f"cond and qoi must have the same shape, got {tuple(cond.shape)} and {tuple(qoi.shape)}")
    if cond.ndim < 3:
        raise ValueError(f"cond and qoi must have shape (..., cells, channels), got {tuple(cond.shape)}")
    if dt_over_dx <= 0.0:
        raise ValueError(f"dt_over_dx must be positive, got {dt_over_dx}")

    # The change in each cell gives the difference between its two boundary
    # fluxes.  Remove the spatial mean first because a periodic update cannot
    # identify one constant added to every boundary.
    delta = qoi - cond
    conservative_delta = delta - delta.mean(dim=-2, keepdim=True)
    flux_jump = -conservative_delta / dt_over_dx

    # Add the boundary differences one cell at a time to recover a flux.  The
    # first boundary value is temporary; centering the result removes it.
    flux = torch.zeros_like(flux_jump)
    flux[..., 1:, :] = torch.cumsum(flux_jump[..., 1:, :], dim=-2)
    return flux - flux.mean(dim=-2, keepdim=True)
