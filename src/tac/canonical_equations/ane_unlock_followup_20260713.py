# SPDX-License-Identifier: MIT
"""Canonical MEANS-only laws for the 2026-07-13 ANE follow-up.

These laws describe local wall-clock and storage measurements.  They do not
grant score, placement, promotion, or archive authority.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

EQUATION_ID = "ane_residency_concurrency_batch_sram_laws_v1"
CONCURRENCY_ACCEPT_FRACTION = 0.05
ANE_SRAM_CLIFF_BYTES = 32 * 2**20
VERDICT_SCOPE = (
    "matched local host, OS build, model bytes, CoreML/MLX builds, batch geometry, "
    "thermal state, and requested compute units; no score or contest-axis transfer"
)
REQ_R = (
    "repeat ABBA with the same model hashes and an independently proved direct-ANE "
    "placement receipt whenever host, OS, SDK, model, batch, or load geometry changes"
)


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def concurrency_degradation(*, concurrent_s: float, solo_s: float) -> float:
    """Return ``T_concurrent/T_solo - 1`` for one matched service."""

    return _positive(concurrent_s, "concurrent_s") / _positive(solo_s, "solo_s") - 1.0


@dataclass(frozen=True)
class ConcurrencyAdmission:
    teacher_degradation_fraction: float
    mlx_degradation_fraction: float
    timing_accept: bool
    placement_proved: bool
    architecture_accept: bool
    threshold_fraction: float = CONCURRENCY_ACCEPT_FRACTION
    evidence_label: str = "DERIVED_FROM_MATCHED_LOCAL_MEASUREMENTS"
    score_claim: bool = False
    promotion_eligible: bool = False
    pointer_moved: bool = False
    verdict_scope: str = VERDICT_SCOPE
    req_r: str = REQ_R

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def admit_concurrency(
    *,
    teacher_solo_s: float,
    teacher_concurrent_s: float,
    mlx_solo_s: float,
    mlx_concurrent_s: float,
    placement_proved: bool,
) -> ConcurrencyAdmission:
    """Apply the strict operator bar and keep placement as a separate gate."""

    teacher = concurrency_degradation(concurrent_s=teacher_concurrent_s, solo_s=teacher_solo_s)
    mlx = concurrency_degradation(concurrent_s=mlx_concurrent_s, solo_s=mlx_solo_s)
    timing_accept = teacher < CONCURRENCY_ACCEPT_FRACTION and mlx < CONCURRENCY_ACCEPT_FRACTION
    proved = bool(placement_proved)
    return ConcurrencyAdmission(
        teacher_degradation_fraction=teacher,
        mlx_degradation_fraction=mlx,
        timing_accept=timing_accept,
        placement_proved=proved,
        architecture_accept=timing_accept and proved,
    )


def batch_seconds_per_pair(*, batch_seconds: float, batch_size: int) -> float:
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    return _positive(batch_seconds, "batch_seconds") / int(batch_size)


def batch_pairs_per_second(*, batch_seconds: float, batch_size: int) -> float:
    return 1.0 / batch_seconds_per_pair(batch_seconds=batch_seconds, batch_size=batch_size)


@dataclass(frozen=True)
class WeightFitAccounting:
    measured_payload_bytes: int
    cliff_bytes: int
    derived_headroom_bytes: int
    clears_cliff: bool
    payload_evidence: str
    actual_ane_sram_residency: str = "UNKNOWN_NOT_MEASURED"
    score_claim: bool = False
    pointer_moved: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def account_weight_fit(
    measured_payload_bytes: int,
    *,
    payload_evidence: str,
    cliff_bytes: int = ANE_SRAM_CLIFF_BYTES,
) -> WeightFitAccounting:
    if isinstance(measured_payload_bytes, bool) or int(measured_payload_bytes) < 0:
        raise ValueError("measured_payload_bytes must be a non-negative integer")
    if isinstance(cliff_bytes, bool) or int(cliff_bytes) <= 0:
        raise ValueError("cliff_bytes must be a positive integer")
    payload = int(measured_payload_bytes)
    cliff = int(cliff_bytes)
    return WeightFitAccounting(
        measured_payload_bytes=payload,
        cliff_bytes=cliff,
        derived_headroom_bytes=cliff - payload,
        clears_cliff=payload < cliff,
        payload_evidence=str(payload_evidence),
    )


def forward_only_amdahl_speedup(*, forward_share: float, forward_speedup: float) -> float:
    """Parameterized bound; callers must supply a measured in-loop share."""

    share = float(forward_share)
    if not math.isfinite(share) or not 0.0 <= share <= 1.0:
        raise ValueError("forward_share must be finite in [0, 1]")
    speedup = _positive(forward_speedup, "forward_speedup")
    return 1.0 / ((1.0 - share) + share / speedup)


__all__ = [
    "ANE_SRAM_CLIFF_BYTES",
    "CONCURRENCY_ACCEPT_FRACTION",
    "EQUATION_ID",
    "REQ_R",
    "VERDICT_SCOPE",
    "ConcurrencyAdmission",
    "WeightFitAccounting",
    "account_weight_fit",
    "admit_concurrency",
    "batch_pairs_per_second",
    "batch_seconds_per_pair",
    "concurrency_degradation",
    "forward_only_amdahl_speedup",
]
