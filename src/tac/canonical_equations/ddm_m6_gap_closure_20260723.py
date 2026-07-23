# SPDX-License-Identifier: MIT
"""Pool-aware byte-gap composition law for DDM M6.

Singleton lever estimates are not additive.  Each non-additive pool contributes
only a joint, receiver-closed byte credit, and the final admitted reduction is
the measured delta of one composed artifact.  This prevents a mathematical
nullity, a scheduled score recovery, or two competing descriptions from being
silently converted into archive bytes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

BASELINE_ARCHIVE_BYTES = 177_169
STRICT_SUB015_CAP_BYTES = 154_524
DECISIVE_GAP_BYTES = BASELINE_ARCHIVE_BYTES - STRICT_SUB015_CAP_BYTES

KNOWN_POOLS = frozenset(
    {
        "P_REALIZE",
        "P_TEMPORAL_DESCRIPTION",
        "P_FRAME_OWNERSHIP",
        "P_NULL_GAUGE",
    }
)


@dataclass(frozen=True, slots=True)
class PoolCredit:
    """One pool's strict joint-replay credit, never a singleton sum."""

    pool_id: str
    lever_ids: tuple[str, ...]
    joint_reduction_bytes: int
    receiver_closed: bool
    evidence_scope: str

    def __post_init__(self) -> None:
        if self.pool_id not in KNOWN_POOLS:
            raise ValueError(f"unknown pool: {self.pool_id}")
        if not self.lever_ids or not all(self.lever_ids):
            raise ValueError("lever_ids must be nonempty")
        if self.joint_reduction_bytes < 0:
            raise ValueError("joint_reduction_bytes must be nonnegative")
        if not self.evidence_scope:
            raise ValueError("evidence_scope must be nonempty")

    @property
    def admitted_bytes(self) -> int:
        return self.joint_reduction_bytes if self.receiver_closed else 0


@dataclass(frozen=True, slots=True)
class GapClosure:
    """The one-artifact reduction and remaining fixed-C1 byte gap."""

    baseline_archive_bytes: int
    strict_cap_bytes: int
    final_archive_bytes: int
    admitted_reduction_bytes: int
    residual_gap_bytes: int
    sub015_reached: bool
    pool_credit_upper_bound_bytes: int


def compose_gap_closure(
    pool_credits: Iterable[PoolCredit],
    *,
    final_archive_bytes: int,
    final_same_artifact_receiver_closed: bool,
    baseline_archive_bytes: int = BASELINE_ARCHIVE_BYTES,
    strict_cap_bytes: int = STRICT_SUB015_CAP_BYTES,
) -> GapClosure:
    """Admit only the measured final-artifact delta.

    The sum of pool credits is merely an upper bound until a final composed
    artifact closes through the receiver.  The final artifact may realize less
    than that bound due to cross-pool interference, but never more.
    """

    if baseline_archive_bytes < 0 or strict_cap_bytes < 0 or final_archive_bytes < 0:
        raise ValueError("archive byte counts must be nonnegative")
    if strict_cap_bytes > baseline_archive_bytes:
        raise ValueError("strict cap cannot exceed baseline")
    if final_archive_bytes > baseline_archive_bytes:
        raise ValueError("final artifact cannot be larger in a reduction proof")

    credits = tuple(pool_credits)
    pool_ids = [credit.pool_id for credit in credits]
    if len(pool_ids) != len(set(pool_ids)):
        raise ValueError("each pool must have exactly one joint credit")

    pool_bound = sum(credit.admitted_bytes for credit in credits)
    measured_final_delta = baseline_archive_bytes - final_archive_bytes
    if final_same_artifact_receiver_closed:
        if measured_final_delta > pool_bound:
            raise ValueError("final reduction exceeds admitted pool credit bound")
        admitted = measured_final_delta
    else:
        admitted = 0

    admitted_final_bytes = baseline_archive_bytes - admitted
    residual = max(0, admitted_final_bytes - strict_cap_bytes)
    return GapClosure(
        baseline_archive_bytes=baseline_archive_bytes,
        strict_cap_bytes=strict_cap_bytes,
        final_archive_bytes=admitted_final_bytes,
        admitted_reduction_bytes=admitted,
        residual_gap_bytes=residual,
        sub015_reached=admitted_final_bytes <= strict_cap_bytes,
        pool_credit_upper_bound_bytes=pool_bound,
    )


__all__ = [
    "BASELINE_ARCHIVE_BYTES",
    "DECISIVE_GAP_BYTES",
    "STRICT_SUB015_CAP_BYTES",
    "GapClosure",
    "PoolCredit",
    "compose_gap_closure",
]
