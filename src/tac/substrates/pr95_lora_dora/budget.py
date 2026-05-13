"""Byte and break-even math for the PR95 LoRA/DoRA adapter lane.

This module is deliberately pure-Python: it does not load PR95, torch, the
scorers, or any archive. Its job is to make the adapter trailer cost explicit
before a training run can be interpreted as score-lowering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from tac.score_geometry import (
    CONTEST_REFERENCE_BYTES,
    POSE_COEFFICIENT_INSIDE_SQRT,
    RATE_COEFFICIENT,
    SEG_COEFFICIENT,
)

AdapterKind = Literal["lora", "dora"]


@dataclass(frozen=True)
class AdapterLayerBudget:
    """Raw adapter budget for one flattened weight tensor."""

    name: str
    out_dim: int
    in_dim: int
    rank: int
    kind: AdapterKind
    trainable_params: int
    raw_trailer_bytes: int


@dataclass(frozen=True)
class AdapterBreakEven:
    """Score penalty and exact single-axis break-even requirements."""

    raw_trailer_bytes: int
    rate_score_penalty: float
    required_seg_reduction: float
    pose_operating_point: float
    required_pose_reduction_exact: float
    pose_only_feasible: bool
    residual_score_after_zero_pose: float
    evidence_grade: str = "[prediction; closed-form adapter break-even]"
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


DEFAULT_TIER_C_LAYER_DIMS: tuple[tuple[str, int, int], ...] = (
    ("blocks.0", 144, 36 * 3 * 3),
    ("blocks.1", 144, 36 * 3 * 3),
    ("blocks.2", 108, 36 * 3 * 3),
    ("blocks.3", 80, 27 * 3 * 3),
    ("blocks.4", 72, 20 * 3 * 3),
    ("blocks.5", 72, 18 * 3 * 3),
)

_TRAILER_HEADER_BYTES = 8
_TRAILER_SUFFIX_BYTES = 4


def _check_rank(rank: int) -> None:
    if not 1 <= rank <= 255:
        raise ValueError(f"rank must be in [1, 255], got {rank}")


def adapter_trainable_params(*, out_dim: int, in_dim: int, rank: int, kind: AdapterKind) -> int:
    """Return trainable parameter count for one LoRA/DoRA-adapted tensor."""

    _check_rank(rank)
    if out_dim <= 0 or in_dim <= 0:
        raise ValueError("out_dim and in_dim must be positive")
    if kind not in ("lora", "dora"):
        raise ValueError(f"kind must be 'lora' or 'dora', got {kind!r}")
    lora_params = rank * (in_dim + out_dim)
    return lora_params + (out_dim if kind == "dora" else 0)


def adapter_raw_trailer_bytes(
    *,
    name: str,
    out_dim: int,
    in_dim: int,
    rank: int,
    kind: AdapterKind,
) -> int:
    """Return exact v1 raw trailer bytes for one adapter record.

    The archive encoder stores A and B as int8 plus per-tensor fp16 scales,
    fixed-shape metadata, alpha, rank, kind, and the UTF-8 adapter name. DoRA
    stores one additional int8 magnitude vector plus fp16 scale.
    """

    params = adapter_trainable_params(out_dim=out_dim, in_dim=in_dim, rank=rank, kind=kind)
    name_bytes = len(name.encode("utf-8"))
    common_metadata = (
        1  # name_len:u8
        + name_bytes
        + 1  # adapter_kind:u8
        + 1  # rank:u8
        + 2  # alpha:f16
        + 4  # B_shape:(u16,u16)
        + 2  # B_scale:f16
        + 4  # A_shape:(u16,u16)
        + 2  # A_scale:f16
    )
    dora_metadata = 6 if kind == "dora" else 0  # magnitude_len:u32 + m_scale:f16
    return common_metadata + dora_metadata + params


def tier_c_layer_budgets(*, rank: int = 8, kind: AdapterKind = "lora") -> tuple[AdapterLayerBudget, ...]:
    """Return per-layer budgets for the six PR95 upsample conv blocks."""

    return tuple(
        AdapterLayerBudget(
            name=name,
            out_dim=out_dim,
            in_dim=in_dim,
            rank=rank,
            kind=kind,
            trainable_params=adapter_trainable_params(
                out_dim=out_dim,
                in_dim=in_dim,
                rank=rank,
                kind=kind,
            ),
            raw_trailer_bytes=adapter_raw_trailer_bytes(
                name=name,
                out_dim=out_dim,
                in_dim=in_dim,
                rank=rank,
                kind=kind,
            ),
        )
        for name, out_dim, in_dim in DEFAULT_TIER_C_LAYER_DIMS
    )


def tier_c_trainable_params(*, rank: int = 8, kind: AdapterKind = "lora") -> int:
    """Return total trainable adapter params for Tier C."""

    return sum(row.trainable_params for row in tier_c_layer_budgets(rank=rank, kind=kind))


def tier_c_raw_trailer_bytes(*, rank: int = 8, kind: AdapterKind = "lora") -> int:
    """Return exact raw v1 trailer bytes for all six Tier C adapters."""

    return (
        _TRAILER_HEADER_BYTES
        + sum(row.raw_trailer_bytes for row in tier_c_layer_budgets(rank=rank, kind=kind))
        + _TRAILER_SUFFIX_BYTES
    )


def rate_score_penalty_for_bytes(byte_delta: int) -> float:
    """Return contest score penalty for adding charged archive bytes."""

    if byte_delta < 0:
        raise ValueError("byte_delta must be non-negative")
    return RATE_COEFFICIENT * byte_delta / CONTEST_REFERENCE_BYTES


def exact_pose_reduction_for_score_delta(pose_dist: float, score_delta: float) -> tuple[float, bool, float]:
    """Return exact pose reduction needed to offset ``score_delta``.

    The pose term is ``sqrt(10 * d_pose)``. Linearizing this term near the
    PR106/PR95 operating point is unsafe when the desired score delta is a
    large fraction of the current pose term, so this helper solves the inverse
    exactly.
    """

    if pose_dist < 0.0 or score_delta < 0.0:
        raise ValueError("pose_dist and score_delta must be non-negative")
    current_term = math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * pose_dist)
    if score_delta >= current_term:
        return pose_dist, False, score_delta - current_term
    new_term = current_term - score_delta
    new_pose_dist = (new_term * new_term) / POSE_COEFFICIENT_INSIDE_SQRT
    return pose_dist - new_pose_dist, True, 0.0


def adapter_break_even(
    *,
    raw_trailer_bytes: int,
    pose_operating_point: float = 3.4e-5,
) -> AdapterBreakEven:
    """Return exact score break-even requirements for a charged trailer."""

    penalty = rate_score_penalty_for_bytes(raw_trailer_bytes)
    pose_reduction, pose_feasible, residual = exact_pose_reduction_for_score_delta(
        pose_operating_point,
        penalty,
    )
    return AdapterBreakEven(
        raw_trailer_bytes=raw_trailer_bytes,
        rate_score_penalty=penalty,
        required_seg_reduction=penalty / SEG_COEFFICIENT,
        pose_operating_point=pose_operating_point,
        required_pose_reduction_exact=pose_reduction,
        pose_only_feasible=pose_feasible,
        residual_score_after_zero_pose=residual,
    )
