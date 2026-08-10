"""
Build the Chebyshev polynomials used by the FluxNet flux law.

Purpose
-------
Keep polynomial values and polynomial slopes in one place. The coefficient
fitter and the model then use exactly the same calculations.

Inputs
------
The functions take a tensor of state values and a nonnegative highest
polynomial number.

Outputs
-------
The last tensor axis stores one value for each polynomial. An input with shape
``(...)`` produces an output with shape ``(..., degree + 1)``.
"""

from __future__ import annotations

import torch


def chebyshev_polynomials(x: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Evaluate ``T_0`` through ``T_degree`` and append the basis axis.

    The step-by-step formula is
    ``T_0(x)=1``, ``T_1(x)=x``, and
    ``T_n(x)=2*x*T_(n-1)(x)-T_(n-2)(x)``.  It is used instead of constructing
    a large polynomial table.

    Parameters
    ----------
    x : torch.Tensor
        Values at which to evaluate the polynomials.
    degree : int
        Highest polynomial number to return. Must be nonnegative.

    Returns
    -------
    torch.Tensor
        Polynomial values with shape ``x.shape + (degree + 1,)``.
    """
    if degree < 0:
        raise ValueError(f"degree must be nonnegative, got {degree}")
    values = [torch.ones_like(x)]
    if degree == 0:
        return torch.stack(values, dim=-1)
    values.append(x)
    # Each new polynomial needs only the two previous polynomials.  Building
    # them in order keeps this function small and avoids extra work.
    for _ in range(2, degree + 1):
        values.append(2.0 * x * values[-1] - values[-2])
    return torch.stack(values, dim=-1)


def chebyshev_polynomial_derivatives(x: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Evaluate derivatives of ``T_0`` through ``T_degree`` at ``x``.

    The slopes are built at the same time as the polynomial values.

    Parameters
    ----------
    x : torch.Tensor
        Values at which to evaluate the polynomial slopes.
    degree : int
        Highest polynomial number to return. Must be nonnegative.

    Returns
    -------
    torch.Tensor
        Polynomial slopes with shape ``x.shape + (degree + 1,)``.
    """
    if degree < 0:
        raise ValueError(f"degree must be nonnegative, got {degree}")
    values = [torch.ones_like(x)]
    derivatives = [torch.zeros_like(x)]
    if degree == 0:
        return torch.stack(derivatives, dim=-1)
    values.append(x)
    derivatives.append(torch.ones_like(x))
    # Build each slope from the same two earlier values and slopes.  The older
    # values stay in the list because the formula needs them.
    for _ in range(2, degree + 1):
        values.append(2.0 * x * values[-1] - values[-2])
        derivatives.append(2.0 * values[-2] + 2.0 * x * derivatives[-1] - derivatives[-2])
    return torch.stack(derivatives, dim=-1)
