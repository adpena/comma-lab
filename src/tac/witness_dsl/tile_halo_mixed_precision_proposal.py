# SPDX-License-Identifier: MIT
"""Typed, default-OFF proposal for tile-halo waterfill and MLX precision.

This is intentionally a proposal surface, not a trainer integration.  It does
not emit command-line flags (the live trainer has no such flags), and therefore
cannot accidentally invent a launch configuration.  A future integration must
map these fields to real, parser-backed DSL primitives before enabling them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class ProposalState(StrEnum):
    OFF_UNWIRED = "off_unwired"
    PROPOSAL_ONLY = "proposal_only"


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


@dataclass(frozen=True)
class TileHaloSensitivityWaterfillProposal:
    """A typed declaration of a halo allocation experiment.

    ``enabled`` defaults to False and ``state`` is always explicit.  The
    proposal carries no flag name or argv rendering by design.
    """

    enabled: bool = False
    state: ProposalState = ProposalState.OFF_UNWIRED
    halo_radius_px: int = 0
    sensitivity_floor: float = 0.0
    max_tile_fraction: float = 1.0
    exact_on_selected_tiles_required: bool = True
    minimum_speedup: float = 2.0
    freshness_survival_bar: float = 0.90
    full_frame_refresh_cadence_steps: int | None = None
    measured_anchor: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        if not isinstance(self.state, ProposalState):
            raise TypeError("state must be ProposalState")
        if self.enabled and self.state is ProposalState.OFF_UNWIRED:
            raise ValueError("an enabled proposal cannot remain OFF_UNWIRED")
        if not self.enabled and self.state is not ProposalState.OFF_UNWIRED:
            raise ValueError("disabled proposal must be explicitly OFF_UNWIRED")
        if not isinstance(self.halo_radius_px, int) or isinstance(self.halo_radius_px, bool) or self.halo_radius_px < 0:
            raise ValueError("halo_radius_px must be a non-negative integer")
        _finite_nonnegative(self.sensitivity_floor, "sensitivity_floor")
        fraction = _finite_nonnegative(self.max_tile_fraction, "max_tile_fraction")
        if fraction > 1.0:
            raise ValueError("max_tile_fraction must be <= 1")
        if not isinstance(self.exact_on_selected_tiles_required, bool):
            raise TypeError("exact_on_selected_tiles_required must be bool")
        if _finite_nonnegative(self.minimum_speedup, "minimum_speedup") <= 1.0:
            raise ValueError("minimum_speedup must be > 1")
        freshness = _finite_nonnegative(
            self.freshness_survival_bar, "freshness_survival_bar"
        )
        if not (0.0 < freshness < 1.0):
            raise ValueError("freshness_survival_bar must be in (0,1)")
        if self.full_frame_refresh_cadence_steps is not None and (
            not isinstance(self.full_frame_refresh_cadence_steps, int)
            or isinstance(self.full_frame_refresh_cadence_steps, bool)
            or self.full_frame_refresh_cadence_steps < 1
        ):
            raise ValueError("full_frame_refresh_cadence_steps must be None or >=1")
        if self.enabled and not (self.measured_anchor and self.measured_anchor.strip()):
            raise ValueError("enabled proposal requires a measured_anchor artifact")
        if self.enabled and self.halo_radius_px < 1:
            raise ValueError("enabled proposal requires a positive derived halo_radius_px")

    def to_display_dict(self) -> dict[str, object]:
        return {
            "kind": "tile_halo_sensitivity_waterfill",
            "enabled": self.enabled,
            "state": self.state.value,
            "halo_radius_px": self.halo_radius_px,
            "sensitivity_floor": float(self.sensitivity_floor),
            "max_tile_fraction": float(self.max_tile_fraction),
            "exact_on_selected_tiles_required": self.exact_on_selected_tiles_required,
            "minimum_speedup": float(self.minimum_speedup),
            "freshness_survival_bar": float(self.freshness_survival_bar),
            "full_frame_refresh_cadence_steps": self.full_frame_refresh_cadence_steps,
            "measured_anchor": self.measured_anchor,
            "wired": False,
        }


@dataclass(frozen=True)
class MlxMixedPrecisionTrainingProposal:
    """Typed MLX training precision proposal; default OFF and unwired."""

    enabled: bool = False
    state: ProposalState = ProposalState.OFF_UNWIRED
    forward: Precision = Precision.FP32
    gradient: Precision = Precision.FP32
    loss_scale: float = 1.0
    minimum_speedup: float = 1.5
    minimum_global_gradient_cosine: float = 0.99
    minimum_pair_gradient_cosine: float = 0.99
    required_quality_pairs: int = 600
    measured_anchor: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.state, ProposalState):
            raise TypeError("enabled must be bool and state must be ProposalState")
        if self.enabled and self.state is ProposalState.OFF_UNWIRED:
            raise ValueError("an enabled proposal cannot remain OFF_UNWIRED")
        if not self.enabled and self.state is not ProposalState.OFF_UNWIRED:
            raise ValueError("disabled proposal must be explicitly OFF_UNWIRED")
        if not isinstance(self.forward, Precision) or not isinstance(self.gradient, Precision):
            raise TypeError("forward and gradient must be Precision values")
        scale = _finite_nonnegative(self.loss_scale, "loss_scale")
        if scale == 0:
            raise ValueError("loss_scale must be > 0")
        if _finite_nonnegative(self.minimum_speedup, "minimum_speedup") <= 1.0:
            raise ValueError("minimum_speedup must be > 1")
        for name, value in {
            "minimum_global_gradient_cosine": self.minimum_global_gradient_cosine,
            "minimum_pair_gradient_cosine": self.minimum_pair_gradient_cosine,
        }.items():
            cosine = _finite_nonnegative(value, name)
            if cosine > 1.0:
                raise ValueError(f"{name} must be <= 1")
        if (
            not isinstance(self.required_quality_pairs, int)
            or isinstance(self.required_quality_pairs, bool)
            or self.required_quality_pairs != 600
        ):
            raise ValueError("required_quality_pairs must be exactly 600")
        if self.enabled and not (self.measured_anchor and self.measured_anchor.strip()):
            raise ValueError("enabled proposal requires a measured_anchor artifact")

    def to_display_dict(self) -> dict[str, object]:
        return {
            "kind": "mlx_mixed_precision_training",
            "enabled": self.enabled,
            "state": self.state.value,
            "forward": self.forward.value,
            "gradient": self.gradient.value,
            "loss_scale": float(self.loss_scale),
            "minimum_speedup": float(self.minimum_speedup),
            "minimum_global_gradient_cosine": float(
                self.minimum_global_gradient_cosine
            ),
            "minimum_pair_gradient_cosine": float(
                self.minimum_pair_gradient_cosine
            ),
            "required_quality_pairs": self.required_quality_pairs,
            "measured_anchor": self.measured_anchor,
            "wired": False,
        }


__all__ = [
    "MlxMixedPrecisionTrainingProposal",
    "Precision",
    "ProposalState",
    "TileHaloSensitivityWaterfillProposal",
]
