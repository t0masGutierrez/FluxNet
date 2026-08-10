"""
Run the complete FluxNet model used for the fixed poster checkpoint.

Purpose
-------
Read clean example state changes, fit the small polynomial flux law, and use a
periodic graph to predict a transfer at every cell boundary. The next state is
built with ``state + roll(transfer, 1) - transfer``. Each transfer leaves one
cell and enters the next one, so the update keeps the total state sum unchanged.
The model does not call a numerical solver during this update.

Inputs
------
The public input is a dictionary. ``demo_cond_v`` is the state before each
example step, ``demo_qoi_v`` is the state after each step, and the two
``demo_*_mask`` fields mark present values. ``quest_cond_v`` is the current
state to advance and ``quest_cond_mask`` marks its present values. The main
tensors use shape ``(batch, examples, cells, 1)``.

Outputs
-------
The prompt encoder returns a reusable program. The graph returns one learned
transfer per cell boundary. The one-step public call returns a tensor with
shape ``(batch, 1, cells, 1)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from coefficients import (
    extended_chebyshev_polynomials,
    fit_flux_coefficients_from_evidence,
)
from prompt import (
    build_flux_evidence_tokens,
    compute_prompt_normalization,
)

# These sizes are part of the checkpoint's input and parameter layout.
_MAX_FLUX_DEGREE = 8
_CONTEXT_DIM = 80
_PROGRAM_DIM = _MAX_FLUX_DEGREE + _CONTEXT_DIM
_FIT_DEGREES = (3, 8)
_OUTPUT_SCALINGS = ("none", "local_psi")


@dataclass(frozen=True)
class FluxNetProgram:
    """
    Store the prompt information reused at every rollout step.

    The program is created once from the clean examples and reused at each
    later rollout step.

    Parameters
    ----------
    center : torch.Tensor
        Prompt state center with shape ``(batch, 1, 1)``.
    scale : torch.Tensor
        Positive prompt state half-range with shape ``(batch, 1, 1)``.
    flux_coefficients : torch.Tensor
        Nonconstant polynomial coefficients with shape ``(batch, 8)``.
        Unused entries are zero.
    context : torch.Tensor
        Prompt information used by the graph, with shape ``(batch, 80)``.
    dt_over_dx : torch.Tensor
        Time step divided by cell width with shape ``(batch, 1, 1)``.

    Attributes
    ----------
    center : torch.Tensor
        Prompt state center.
    scale : torch.Tensor
        Prompt state half-range.
    flux_coefficients : torch.Tensor
        Padded nonconstant polynomial coefficients.
    context : torch.Tensor
        Prompt information used by the graph.
    dt_over_dx : torch.Tensor
        Time step divided by cell width.
    """

    center: torch.Tensor
    scale: torch.Tensor
    flux_coefficients: torch.Tensor
    context: torch.Tensor
    dt_over_dx: torch.Tensor

    def __post_init__(self) -> None:
        """
        Check that every stored program value has the expected shape.

        Returns
        -------
        None
            This method returns nothing. It raises an error when a stored
            program value is not valid.
        """
        batch_size = self.center.shape[0]
        scalar_shape = (batch_size, 1, 1)
        if self.center.shape != scalar_shape:
            raise ValueError(f"center must have shape {scalar_shape}, got {tuple(self.center.shape)}")
        if self.scale.shape != scalar_shape:
            raise ValueError(f"scale must have shape {scalar_shape}, got {tuple(self.scale.shape)}")
        if self.dt_over_dx.shape != scalar_shape:
            raise ValueError(f"dt_over_dx must have shape {scalar_shape}, got {tuple(self.dt_over_dx.shape)}")
        if self.flux_coefficients.shape != (batch_size, _MAX_FLUX_DEGREE):
            raise ValueError(
                "flux_coefficients must have shape "
                f"({batch_size}, {_MAX_FLUX_DEGREE}), got "
                f"{tuple(self.flux_coefficients.shape)}"
            )
        if self.context.shape != (batch_size, _CONTEXT_DIM):
            raise ValueError(f"context must have shape ({batch_size}, {_CONTEXT_DIM}), got {tuple(self.context.shape)}")
        for value in (
            self.center,
            self.scale,
            self.flux_coefficients,
            self.context,
            self.dt_over_dx,
        ):
            if not value.is_floating_point():
                raise ValueError("all program values must have a floating dtype")
            if value.dtype != self.center.dtype or value.device != self.center.device:
                raise ValueError("all program values must share one dtype and device")
            _require_finite(value, name="program value")
        if not torch.compiler.is_compiling():
            if not torch.all(self.scale > 0.0).item():
                raise ValueError("scale must be strictly positive")
            if not torch.all(self.dt_over_dx > 0.0).item():
                raise ValueError("dt_over_dx must be strictly positive")

    @property
    def batch_size(self) -> int:
        """
        Return the number of prompt programs.

        Returns
        -------
        int
            Number of prompt programs stored in the batch.
        """
        return self.center.shape[0]


@dataclass(frozen=True)
class LocalPhysicsFeatures:
    """
    Store values that describe the current state at each cell.

    These values give the graph a scaled state, the fitted flux law, and a
    local estimate of how a change moves between neighboring cells.

    Parameters
    ----------
    z : torch.Tensor
        Current state after prompt scaling.
    g : torch.Tensor
        Fitted flux multiplied by time step divided by cell width.
    v : torch.Tensor
        Slope of the fitted flux multiplied by time step divided by cell
        width.
    shock_speed : torch.Tensor
        Local speed estimate between each cell and its right neighbor.

    Attributes
    ----------
    z : torch.Tensor
        Current scaled state.
    g : torch.Tensor
        Scaled fitted flux.
    v : torch.Tensor
        Scaled fitted-flux slope.
    shock_speed : torch.Tensor
        Local speed estimate used by the graph.
    """

    z: torch.Tensor
    g: torch.Tensor
    v: torch.Tensor
    shock_speed: torch.Tensor


@dataclass(frozen=True)
class FluxNetStepOutput:
    """
    Hold the two results of one FluxNet update.

    The two values let callers inspect both the learned transfer and the state
    produced by applying it.

    Parameters
    ----------
    interface_transfer : torch.Tensor
        Learned amount sent across each cell boundary.
    next_state : torch.Tensor
        State after the transfer enters and leaves each cell.

    Attributes
    ----------
    interface_transfer : torch.Tensor
        Learned boundary-transfer values.
    next_state : torch.Tensor
        Conservative next state.
    """

    interface_transfer: torch.Tensor
    next_state: torch.Tensor


@dataclass(frozen=True)
class _InterfaceFeatures:
    """
    Hold simple numbers that describe one neighboring-cell boundary.

    Parameters
    ----------
    head : torch.Tensor
        Three normalized boundary values read by the graph.
    psi : torch.Tensor
        Positive boundary scale used by the optional output scaling.

    Attributes
    ----------
    head : torch.Tensor
        Normalized average flux, state jump, and boundary size.
    psi : torch.Tensor
        Positive local boundary scale.
    """

    head: torch.Tensor
    psi: torch.Tensor


class _FluxNetPromptEncoder(nn.Module):
    """
    Read the examples once and make the program used by the graph.

    It combines learned prompt information with a fitted polynomial law and
    stores both results in a reusable program.

    Parameters
    ----------
    dt_over_dx : float
        Time step divided by cell width used to recover flux evidence.
    fit_degree : int
        Number of nonconstant polynomial terms to fit. Must be 3 or 8.
    speed_bound : float
        Largest allowed estimated slope of the fitted flux law.
    normalization_epsilon : float
        Small positive value used by scaling and safe divisions.
    evidence_ridge : float
        Small nonnegative penalty used to keep the polynomial fit stable.

    Attributes
    ----------
    token_mlp : torch.nn.Module
        Small network that reads one feature row at a time.
    pooling_attention : torch.nn.MultiheadAttention
        Layer that combines the valid prompt rows.
    pooling_queries : torch.Tensor
        Learned queries used to summarize the prompt rows.
    context_compression : torch.nn.Module
        Layer that turns the summaries into the stored context values.
    """

    def __init__(
        self,
        *,
        dt_over_dx: float,
        fit_degree: int,
        speed_bound: float,
        normalization_epsilon: float,
        evidence_ridge: float,
    ) -> None:
        """
        Store the prompt settings and create the prompt-reading layers.

        Parameters
        ----------
        dt_over_dx : float
            Time step divided by cell width used to recover flux evidence.
        fit_degree : int
            Number of nonconstant polynomial terms to fit. Must be 3 or 8.
        speed_bound : float
            Largest allowed estimated slope of the fitted flux law.
        normalization_epsilon : float
            Small positive value used by scaling and safe divisions.
        evidence_ridge : float
            Small nonnegative penalty used to keep the polynomial fit stable.

        Returns
        -------
        None
            This method initializes the prompt-reading layers.
        """
        super().__init__()
        _validate_positive_scalar(dt_over_dx, name="dt_over_dx")
        if fit_degree not in _FIT_DEGREES:
            raise ValueError(f"fit_degree must be one of {_FIT_DEGREES}, got {fit_degree}")
        _validate_positive_scalar(speed_bound, name="speed_bound")
        _validate_positive_scalar(
            normalization_epsilon,
            name="normalization_epsilon",
        )
        if not math.isfinite(evidence_ridge) or evidence_ridge < 0.0:
            raise ValueError(f"evidence_ridge must be finite and nonnegative, got {evidence_ridge}")
        self.dt_over_dx = dt_over_dx
        self.fit_degree = fit_degree
        self.speed_bound = speed_bound
        self.normalization_epsilon = normalization_epsilon
        self.evidence_ridge = evidence_ridge
        self.token_mlp = nn.Sequential(
            nn.Linear(8, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
        )
        self.pooling_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )
        self.pooling_queries = nn.Parameter(torch.empty(2, 512))
        nn.init.normal_(self.pooling_queries, mean=0.0, std=0.02)
        self.context_compression = nn.Sequential(
            nn.Linear(1024, _CONTEXT_DIM),
            nn.LayerNorm(_CONTEXT_DIM),
        )

    def forward(
        self,
        data: dict[str, torch.Tensor],
    ) -> FluxNetProgram:
        """
        Encode one prompt and return the values used during rollout.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Prompt states and masks with the field names expected by FluxNet.

        Returns
        -------
        FluxNetProgram
            Fitted flux coefficients, prompt context, and scaling values that
            can be reused for every later rollout step.
        """
        demo_cond, demo_qoi, cond_mask, qoi_mask = _prompt_tensors(data)
        center, scale = compute_prompt_normalization(
            demo_cond,
            demo_qoi,
            cond_mask,
            qoi_mask,
            self.normalization_epsilon,
        )
        tokens, token_mask = build_flux_evidence_tokens(
            data,
            center,
            scale,
            self.dt_over_dx,
        )
        # The fourth and seventh token entries are the scaled midpoint and the
        # recovered flux.  Fit the law once, before the recurrent rollout.
        fitted = fit_flux_coefficients_from_evidence(
            normalized_midpoint=tokens[..., 3],
            normalized_flux_evidence=tokens[..., 6],
            evidence_mask=token_mask,
            degree=self.fit_degree,
            ridge=self.evidence_ridge,
            speed_bound=self.speed_bound,
            epsilon=self.normalization_epsilon,
        )
        # The checkpoint always stores eight coefficient slots.  A degree-3
        # fit uses the first three nonconstant slots and zeroes the rest.
        coefficients = F.pad(
            fitted,
            (0, _MAX_FLUX_DEGREE - self.fit_degree),
        )
        # Attention reads one long list of prompt rows instead of treating
        # each example as a separate sequence.
        flat_tokens = einops.rearrange(
            tokens,
            "batch transitions cells features -> batch (transitions cells) features",
        )
        flat_mask = einops.rearrange(
            token_mask,
            "batch transitions cells -> batch (transitions cells)",
        )
        encoded = self.token_mlp(flat_tokens)
        queries = self.pooling_queries[None].expand(tokens.shape[0], -1, -1)
        # Padding masks are true where a row is unavailable, which is why the
        # stored valid-row mask is inverted here.
        pooled, _ = self.pooling_attention(
            queries,
            encoded,
            encoded,
            key_padding_mask=~flat_mask,
            need_weights=False,
        )
        context = self.context_compression(
            einops.rearrange(
                pooled,
                "batch queries hidden -> batch (queries hidden)",
            )
        )
        return FluxNetProgram(
            center=center,
            scale=scale,
            flux_coefficients=coefficients,
            context=context,
            dt_over_dx=torch.full_like(center, self.dt_over_dx),
        )


class _ShallowTanhMLP(nn.Module):
    """
    Apply the small two-layer nonlinear map used in the graph.

    Parameters
    ----------
    input_dim : int
        Number of values in each input row.
    hidden_dim : int
        Number of values in the hidden layer.
    output_dim : int
        Number of values returned for each input row.

    Attributes
    ----------
    layers : torch.nn.Sequential
        Two linear layers with one tanh step between them.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        """
        Create a small map with one hidden layer and tanh activations.

        Parameters
        ----------
        input_dim : int
            Number of values in each input row.
        hidden_dim : int
            Number of values in the hidden layer.
        output_dim : int
            Number of values returned for each input row.

        Returns
        -------
        None
            This method initializes the two linear layers.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """
        Map the last feature axis to the requested output width.

        Parameters
        ----------
        value : torch.Tensor
            Input values. The last axis must have length ``input_dim``.

        Returns
        -------
        torch.Tensor
            Mapped values with the same leading shape as ``value`` and last
            axis length ``output_dim``.
        """
        return self.layers(value)


class _DilatedMessageBlock(nn.Module):
    """
    Share information between cells at one periodic distance.

    Periodic wrapping keeps the first and last cells connected.

    Parameters
    ----------
    width : int
        Number of learned values stored at each cell.
    dilation : int
        Periodic distance used to choose the neighboring cells.

    Attributes
    ----------
    dilation : int
        Periodic distance used by this block.
    edge_network : _ShallowTanhMLP
        Network that turns two cell features into a message.
    node_network : _ShallowTanhMLP
        Network that combines the two messages at one cell.
    normalization : torch.nn.LayerNorm
        Layer that keeps the updated feature values at a useful scale.
    residual_scale : torch.Tensor
        Learned size of the block update.
    """

    def __init__(self, *, width: int, dilation: int) -> None:
        """
        Create one message block for the requested grid distance.

        Parameters
        ----------
        width : int
            Number of learned values stored at each cell.
        dilation : int
            Periodic distance used to choose the neighboring cells.

        Returns
        -------
        None
            This method initializes the edge and cell update layers.
        """
        super().__init__()
        self.dilation = dilation
        self.edge_network = _ShallowTanhMLP(
            input_dim=2 * width + 3,
            hidden_dim=width,
            output_dim=width,
        )
        self.node_network = _ShallowTanhMLP(
            input_dim=3 * width,
            hidden_dim=width,
            output_dim=width,
        )
        self.normalization = nn.LayerNorm(width)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        features: torch.Tensor,
        z: torch.Tensor,
        g: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """
        Return features after one left/right message exchange.

        Parameters
        ----------
        features : torch.Tensor
            Learned cell values with shape ``(batch, cells, width)``.
        z : torch.Tensor
            Scaled state values with shape ``(batch, cells, 1)``.
        g : torch.Tensor
            Scaled fitted-flux values with shape ``(batch, cells, 1)``.
        epsilon : float
            Small positive value used in boundary scaling.

        Returns
        -------
        torch.Tensor
            Updated cell features with the same shape as ``features``.
        """
        # A negative shift selects the cell to the right; a positive shift
        # selects the cell to the left on the periodic grid.
        right_features = torch.roll(
            features,
            shifts=-self.dilation,
            dims=1,
        )
        left_features = torch.roll(
            features,
            shifts=self.dilation,
            dims=1,
        )
        right_head = _interface_features(
            z,
            g,
            offset=self.dilation,
            epsilon=epsilon,
        ).head
        left_head = _interface_features(
            torch.roll(z, shifts=self.dilation, dims=1),
            torch.roll(g, shifts=self.dilation, dims=1),
            offset=self.dilation,
            epsilon=epsilon,
        ).head
        # Each edge network sees the two endpoint features and three simple
        # boundary values.  The node network combines both messages.
        right_message = self.edge_network(torch.cat((features, right_features, right_head), dim=-1))
        left_message = self.edge_network(torch.cat((left_features, features, left_head), dim=-1))
        update = self.node_network(torch.cat((features, left_message, right_message), dim=-1))
        return self.normalization(features + self.residual_scale * update)


class _FluxNetTransferGraph(nn.Module):
    """
    Predict the amount transferred through each cell boundary.

    It encodes cell values, shares information across periodic distances, and
    decodes a mean-zero transfer at each boundary.

    Parameters
    ----------
    width : int
        Number of learned values stored at each cell.
    input_dim : int
        Number of values supplied to the cell encoder.
    normalization_epsilon : float
        Small positive value used in boundary scaling.
    output_scaling : str
        Output scale rule. The supported values are ``"none"`` and
        ``"local_psi"``.
    dilations : tuple[int, ...]
        Positive periodic distances used by the message blocks.

    Attributes
    ----------
    normalization_epsilon : float
        Small value used in boundary scaling.
    output_scaling : str
        Selected boundary-transfer scale rule.
    dilations : tuple[int, ...]
        Periodic distances used by the message blocks.
    node_encoder : _ShallowTanhMLP
        Network that turns cell inputs into graph features.
    blocks : torch.nn.ModuleList
        Periodic message blocks applied in order.
    interface_decoder : _ShallowTanhMLP
        Network that predicts one transfer per cell boundary.
    """

    def __init__(
        self,
        *,
        width: int,
        input_dim: int,
        normalization_epsilon: float,
        output_scaling: str,
        dilations: tuple[int, ...],
    ) -> None:
        """
        Create the cell encoder, message blocks, and boundary decoder.

        Parameters
        ----------
        width : int
            Number of learned values stored at each cell.
        input_dim : int
            Number of values supplied to the cell encoder.
        normalization_epsilon : float
            Small positive value used in boundary scaling.
        output_scaling : str
            Output scale rule. The supported values are ``"none"`` and
            ``"local_psi"``.
        dilations : tuple[int, ...]
            Positive periodic distances used by the message blocks.

        Returns
        -------
        None
            This method initializes the graph layers.
        """
        super().__init__()
        if output_scaling not in _OUTPUT_SCALINGS:
            raise ValueError(f"output_scaling must be one of {_OUTPUT_SCALINGS}, got {output_scaling!r}")
        if not dilations or any(not isinstance(value, int) or value < 1 for value in dilations):
            raise ValueError("dilations must contain positive integers")
        self.normalization_epsilon = normalization_epsilon
        self.output_scaling = output_scaling
        self.dilations = dilations
        self.node_encoder = _ShallowTanhMLP(
            input_dim=input_dim,
            hidden_dim=width,
            output_dim=width,
        )
        self.blocks = nn.ModuleList([_DilatedMessageBlock(width=width, dilation=dilation) for dilation in dilations])
        self.interface_decoder = _ShallowTanhMLP(
            input_dim=2 * width + 3 + _PROGRAM_DIM,
            hidden_dim=width,
            output_dim=1,
        )

    def forward(
        self,
        *,
        node_input: torch.Tensor,
        z: torch.Tensor,
        g: torch.Tensor,
        program_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return one normalized, mean-zero transfer per boundary.

        Parameters
        ----------
        node_input : torch.Tensor
            Cell features supplied to the node encoder.
        z : torch.Tensor
            Scaled state values with shape ``(batch, cells, 1)``.
        g : torch.Tensor
            Scaled fitted-flux values with shape ``(batch, cells, 1)``.
        program_values : torch.Tensor
            Prompt values repeated across the cell graph, with shape
            ``(batch, program_values)``.

        Returns
        -------
        torch.Tensor
            Mean-zero transfer values with shape ``(batch, cells, 1)``.
        """
        features = self.node_encoder(node_input)
        # Each block adds information from another periodic distance while
        # preserving the cell axis.
        for block in self.blocks:
            features = block(
                features,
                z,
                g,
                self.normalization_epsilon,
            )
        right_features = torch.roll(features, shifts=-1, dims=1)
        interface = _interface_features(
            z,
            g,
            offset=1,
            epsilon=self.normalization_epsilon,
        )
        repeated_program = program_values[:, None].expand(
            -1,
            z.shape[1],
            -1,
        )
        raw_transfer = self.interface_decoder(
            torch.cat(
                (
                    features,
                    right_features,
                    interface.head,
                    repeated_program,
                ),
                dim=-1,
            )
        )
        # The optional scale makes large local jumps easier for the decoder to
        # represent.  The poster checkpoint uses ``"none"``.
        if self.output_scaling == "local_psi":
            raw_transfer = raw_transfer * interface.psi
        # Removing the spatial mean fixes the otherwise free constant and
        # ensures the transfers sum to zero across the periodic grid.
        return raw_transfer - raw_transfer.mean(dim=1, keepdim=True)


class FluxNet(nn.Module):
    """
    Predict the next state with a learned periodic transfer rule.

    Prepare the prompt once, then apply the learned transfer rule at each
    later time step. The constructor settings must match the checkpoint.

    Parameters
    ----------
    dt_over_dx : float
        Time step divided by cell width used to recover flux evidence.
    fit_degree : int
        Number of nonconstant polynomial terms to fit. Must be 3 or 8.
    speed_bound : float
        Largest allowed estimated slope of the fitted flux law.
    normalization_epsilon : float
        Small positive value used by scaling and safe divisions.
    evidence_ridge : float
        Small nonnegative penalty used to keep the polynomial fit stable.
    output_scaling : str
        Output scale rule. The supported values are ``"none"`` and
        ``"local_psi"``.
    dilations : tuple[int, ...] | list[int]
        Positive periodic distances used by the graph message blocks.
    width : int
        Number of learned values stored at each cell.

    Attributes
    ----------
    normalization_epsilon : float
        Small value used in local boundary calculations.
    prompt_encoder : _FluxNetPromptEncoder
        Learned part that reads the clean examples once.
    graph : _FluxNetTransferGraph
        Learned part that predicts boundary transfers from the current state.
    """

    def __init__(
        self,
        *,
        dt_over_dx: float,
        fit_degree: int,
        speed_bound: float,
        normalization_epsilon: float,
        evidence_ridge: float,
        output_scaling: str,
        dilations: tuple[int, ...] | list[int],
        width: int,
    ) -> None:
        """
        Create the prompt encoder and periodic transfer graph.

        Parameters
        ----------
        dt_over_dx : float
            Time step divided by cell width used to recover flux evidence.
        fit_degree : int
            Number of nonconstant polynomial terms to fit. Must be 3 or 8.
        speed_bound : float
            Largest allowed estimated slope of the fitted flux law.
        normalization_epsilon : float
            Small positive value used by scaling and safe divisions.
        evidence_ridge : float
            Small nonnegative penalty used to keep the polynomial fit stable.
        output_scaling : str
            Output scale rule. The supported values are ``"none"`` and
            ``"local_psi"``.
        dilations : tuple[int, ...] | list[int]
            Positive periodic distances used by the graph message blocks.
        width : int
            Number of learned values stored at each cell.

        Returns
        -------
        None
            This method initializes the prompt encoder and transfer graph.
        """
        super().__init__()
        _validate_positive_scalar(
            normalization_epsilon,
            name="normalization_epsilon",
        )
        if not isinstance(width, int) or width < 1:
            raise ValueError(f"width must be a positive integer, got {width}")
        resolved_dilations = tuple(dilations)
        self.normalization_epsilon = normalization_epsilon
        self.prompt_encoder = _FluxNetPromptEncoder(
            dt_over_dx=dt_over_dx,
            fit_degree=fit_degree,
            speed_bound=speed_bound,
            normalization_epsilon=normalization_epsilon,
            evidence_ridge=evidence_ridge,
        )
        self.graph = _FluxNetTransferGraph(
            width=width,
            input_dim=4 + _PROGRAM_DIM,
            normalization_epsilon=normalization_epsilon,
            output_scaling=output_scaling,
            dilations=resolved_dilations,
        )

    def prepare_rollout(
        self,
        data: dict[str, torch.Tensor],
    ) -> FluxNetProgram:
        """
        Read the prompt once and return a reusable rollout program.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Prompt states, masks, and the current query state.

        Returns
        -------
        FluxNetProgram
            Values fitted from the prompt and reused at each later step.
        """
        _query_state(data)
        return self.prompt_encoder(data)

    def step_prepared(
        self,
        state: torch.Tensor,
        program: FluxNetProgram,
    ) -> torch.Tensor:
        """
        Apply one learned update and return the next state.

        Parameters
        ----------
        state : torch.Tensor
            Current state with shape ``(batch, cells, 1)``.
        program : FluxNetProgram
            Prompt information prepared by :meth:`prepare_rollout`.

        Returns
        -------
        torch.Tensor
            Next state with the same shape as ``state``.
        """
        return self.forward_components(state, program).next_state

    def forward_components(
        self,
        state: torch.Tensor,
        program: FluxNetProgram,
    ) -> FluxNetStepOutput:
        """
        Return boundary transfers and the state made from those transfers.

        Parameters
        ----------
        state : torch.Tensor
            Current state with shape ``(batch, cells, 1)``.
        program : FluxNetProgram
            Prompt information prepared by :meth:`prepare_rollout`.

        Returns
        -------
        FluxNetStepOutput
            The learned boundary transfers and the conservative next state.
        """
        _validate_state_and_program(state, program)
        local = dimensionless_local_features(
            state,
            program,
            epsilon=self.normalization_epsilon,
        )
        program_values = torch.cat(
            (program.flux_coefficients, program.context),
            dim=-1,
        )
        repeated_program = program_values[:, None].expand(
            -1,
            state.shape[1],
            -1,
        )
        # The graph sees four local values plus the same prompt program at
        # every cell.  It returns a dimensionless transfer pattern.
        normalized_transfer = self.graph(
            node_input=torch.cat(
                (
                    local.z,
                    local.g,
                    local.v,
                    local.shock_speed,
                    repeated_program,
                ),
                dim=-1,
            ),
            z=local.z,
            g=local.g,
            program_values=program_values,
        )
        # Return to the original state scale before applying the update.
        interface_transfer = program.scale * normalized_transfer

        # A cell loses the transfer at its right boundary and gains the
        # transfer entering from its left boundary.  This form is the reason
        # the total state sum is unchanged on the periodic grid.
        next_state = state + torch.roll(interface_transfer, shifts=1, dims=1) - interface_transfer
        _require_finite(next_state, name="next state")
        return FluxNetStepOutput(
            interface_transfer=interface_transfer,
            next_state=next_state,
        )

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a prompt and return one next-state prediction.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Prompt states, masks, and the current query state.

        Returns
        -------
        torch.Tensor
            One predicted next state with shape ``(batch, 1, cells, 1)``.
        """
        state = _query_state(data)
        program = self.prepare_rollout(data)
        return self.step_prepared(state, program)[:, None]


def dimensionless_local_features(
    state: torch.Tensor,
    program: FluxNetProgram,
    *,
    epsilon: float,
) -> LocalPhysicsFeatures:
    """
    Evaluate the fitted law and estimate local motion between cells.

    A state jump uses the flux change divided by the state change. With almost
    no jump, the estimate uses the average of the two local slopes.

    Parameters
    ----------
    state : torch.Tensor
        Current state with shape ``(batch, cells, 1)``.
    program : FluxNetProgram
        Prompt-derived polynomial, scale, and time-step values.
    epsilon : float
        Small positive value used when a neighboring state jump is near zero.

    Returns
    -------
    LocalPhysicsFeatures
        Scaled state, fitted-flux values, fitted-flux slopes, and local speed
        estimates for each cell.
    """
    _validate_positive_scalar(epsilon, name="epsilon")
    _validate_state_and_program(state, program)
    z = (state - program.center) / program.scale
    # The polynomial is written in scaled state coordinates so its values are
    # comparable across prompt examples.
    values, derivatives = extended_chebyshev_polynomials(
        z[..., 0],
        _MAX_FLUX_DEGREE,
    )
    zeros = torch.zeros_like(z[..., 0])
    zero_values, _ = extended_chebyshev_polynomials(
        zeros,
        _MAX_FLUX_DEGREE,
    )
    # The constant polynomial term is omitted.  A constant boundary flux does
    # not change a periodic state, so only changes relative to the center are
    # useful here.
    centered_values = values[..., 1:] - zero_values[..., 1:]
    normalized_flux = torch.einsum(
        "bcd,bd->bc",
        centered_values,
        program.flux_coefficients,
    )[..., None]
    flux_derivative = torch.einsum(
        "bcd,bd->bc",
        derivatives[..., 1:],
        program.flux_coefficients,
    )[..., None]
    g = program.dt_over_dx * normalized_flux
    v = program.dt_over_dx * flux_derivative
    right_z = torch.roll(z, shifts=-1, dims=1)
    right_g = torch.roll(g, shifts=-1, dims=1)
    right_v = torch.roll(v, shifts=-1, dims=1)
    jump = right_z - z
    # Avoid division by a tiny state jump.  The fallback below supplies the
    # slope that is meaningful when the two neighboring states are equal.
    safe_jump = torch.where(
        jump.abs() > epsilon,
        jump,
        torch.ones_like(jump),
    )
    secant = (right_g - g) / safe_jump
    shock_speed = torch.where(
        jump.abs() > epsilon,
        secant,
        0.5 * (v + right_v),
    )
    return LocalPhysicsFeatures(
        z=z,
        g=g,
        v=v,
        shock_speed=shock_speed,
    )


def _interface_features(
    z: torch.Tensor,
    g: torch.Tensor,
    *,
    offset: int,
    epsilon: float,
) -> _InterfaceFeatures:
    """
    Describe the average flux and state jump at a periodic boundary.

    Parameters
    ----------
    z : torch.Tensor
        Scaled state values with shape ``(batch, cells, 1)``.
    g : torch.Tensor
        Scaled fitted-flux values with shape ``(batch, cells, 1)``.
    offset : int
        Number of cells to move when selecting the right neighbor.
    epsilon : float
        Small positive value used to keep the boundary scale nonzero.

    Returns
    -------
    _InterfaceFeatures
        Normalized boundary features and their positive local scale.
    """
    right_z = torch.roll(z, shifts=-offset, dims=1)
    right_g = torch.roll(g, shifts=-offset, dims=1)
    average_flux = 0.5 * (g + right_g)
    jump = right_z - z
    # ``psi`` is a local size used to make the first two values well scaled;
    # epsilon keeps the division safe when both the jump and flux are small.
    psi = torch.sqrt(average_flux.square() + jump.square() + epsilon**2)
    head = torch.cat(
        (
            average_flux / psi,
            jump / psi,
            torch.log1p(psi),
        ),
        dim=-1,
    )
    return _InterfaceFeatures(head=head, psi=psi)


def _prompt_tensors(
    data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Read the four demonstration tensors required by the prompt encoder.

    Parameters
    ----------
    data : dict[str, torch.Tensor]
        Input dictionary containing the four demonstration fields.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Demonstration before-states, after-states, and their two masks.
    """
    required = (
        "demo_cond_v",
        "demo_qoi_v",
        "demo_cond_mask",
        "demo_qoi_mask",
    )
    try:
        values = tuple(data[key] for key in required)
    except KeyError as error:
        raise ValueError(f"data must contain {', '.join(required)}") from error
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        values,
    )


def _query_state(data: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Read and validate the one current state used to start a rollout.

    Parameters
    ----------
    data : dict[str, torch.Tensor]
        Input dictionary containing the query state and prompt output fields.

    Returns
    -------
    torch.Tensor
        Current state with the query axis removed, with shape
        ``(batch, cells, 1)``.
    """
    try:
        query = data["quest_cond_v"]
        query_mask = data["quest_cond_mask"]
        demo_qoi = data["demo_qoi_v"]
        demo_qoi_mask = data["demo_qoi_mask"]
    except KeyError as error:
        raise ValueError("data must contain quest_cond_v, quest_cond_mask, demo_qoi_v, and demo_qoi_mask") from error
    if query.ndim != 4 or query.shape[1] != 1 or query.shape[2] < 64 or query.shape[-1] != 1:
        raise ValueError(
            f"quest_cond_v must have shape (batch, queries=1, cells>=64, channels=1), got {tuple(query.shape)}"
        )
    if query_mask.shape != query.shape[:-1] or query_mask.dtype != torch.bool:
        raise ValueError("quest_cond_mask must be boolean and match quest_cond_v")
    if demo_qoi.ndim != 4 or demo_qoi.shape[0] != query.shape[0]:
        raise ValueError("demo_qoi_v must share the query batch")
    if demo_qoi_mask.shape != demo_qoi.shape[:-1] or demo_qoi_mask.dtype != torch.bool:
        raise ValueError("demo_qoi_mask must be boolean and match demo_qoi_v")
    if not torch.compiler.is_compiling():
        # These checks are intentionally eager-only because ``.item()`` and
        # exact equality checks cannot be placed inside a compiled graph.
        if not query_mask.all().item():
            raise ValueError("quest_cond_mask must be complete")
        if not torch.equal(query[:, 0], demo_qoi[:, -1]):
            raise ValueError("query state must equal the final clean prompt state")
    return query[:, 0]


def _validate_state_and_program(
    state: torch.Tensor,
    program: FluxNetProgram,
) -> None:
    """
    Check that a state can be used with a prepared prompt program.

    Parameters
    ----------
    state : torch.Tensor
        Current state to validate.
    program : FluxNetProgram
        Prepared prompt information to match against ``state``.

    Returns
    -------
    None
        This function returns nothing. It raises an error when the values do
        not have compatible shapes, types, devices, or numbers.
    """
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"state must be a torch.Tensor, got {type(state).__name__}")
    if not state.is_floating_point():
        raise ValueError(f"state must have a floating dtype, got {state.dtype}")
    if state.ndim != 3 or state.shape[1] < 64 or state.shape[-1] != 1:
        raise ValueError(f"state must have shape (batch, cells>=64, channels=1), got {tuple(state.shape)}")
    if state.shape[0] != program.batch_size:
        raise ValueError(f"state batch {state.shape[0]} does not match program batch {program.batch_size}")
    if state.dtype != program.center.dtype or state.device != program.center.device:
        raise ValueError("state and program must share one dtype and device")
    _require_finite(state, name="state")


def _validate_positive_scalar(value: float, *, name: str) -> None:
    """
    Require a named model setting to be positive and finite.

    Parameters
    ----------
    value : float
        Model setting to check.
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


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    """
    Reject invalid tensor values during normal eager execution.

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
    if torch.compiler.is_compiling():
        return
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
