# SPDX-License-Identifier: MIT
"""Dependency-free core for the V9 micro-batch launch-memory guard.

The canonical equation wrapper lives at
``tac.canonical_equations.micro_batch_memory_guard_20260715``.  This core deliberately stays outside
that package because the standalone memory preflight must not import the canonical-equations
registry's optional scientific dependencies merely to perform launch-safety arithmetic.

Equation::

    G(B) = max(0, B - 1) * (5918 MiB / 1024)
    M_guarded(B) = M_serial + G(B)

#261 measured complete n=8 process peaks of 5,907 MiB at B=1 and 5,918 MiB at B=4.  Charging the
entire measured B4 process peak for every extra pair is a conservative DERIVED guard, not an actual
current-V9 B2 n600 RSS measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

EQUATION_ID = "v9_micro_batch_conservative_launch_memory_guard_v1"
MEASURED_B1_N8_PROCESS_PEAK_MIB = 5907.0
MEASURED_B4_N8_PROCESS_PEAK_MIB = 5918.0
MICRO_BATCH_EXTRA_PAIR_GUARD_GIB = MEASURED_B4_N8_PROCESS_PEAK_MIB / 1024.0
MICRO_BATCH_GUARD_PROVENANCE = (
    "DERIVED_CONSERVATIVE_CHARGE_FROM_MEASURED_261_B4_N8_PEAK; "
    "ACTUAL_CURRENT_V9_B2_N600_RSS_UNMEASURED"
)
VERDICT_SCOPE = (
    "launch-memory projection for this V9 trainer shape only; no actual-RSS, throughput, score, "
    "contest-axis, archive, or promotion authority"
)
REQ_R = (
    "replace the conservative charge only after matched current-V9 B1/B2 n600 actual peak RSS "
    "is captured with config identity, host fingerprint, and uncontended process custody"
)


def micro_batch_guard_gib(micro_batch_pairs: int) -> float:
    """Return the conservative extra launch-memory charge for B>=1."""

    if isinstance(micro_batch_pairs, bool) or not isinstance(micro_batch_pairs, int):
        raise ValueError("micro_batch_pairs must be an integer >= 1")
    if micro_batch_pairs < 1:
        raise ValueError(f"micro_batch_pairs must be >= 1, got {micro_batch_pairs}")
    return max(0, micro_batch_pairs - 1) * MICRO_BATCH_EXTRA_PAIR_GUARD_GIB


@dataclass(frozen=True)
class MicroBatchMemoryGuardReceipt:
    micro_batch_pairs: int
    guard_gib: float
    equation_id: str = EQUATION_ID
    evidence_label: str = MICRO_BATCH_GUARD_PROVENANCE
    actual_current_v9_b2_n600_rss_gib: None = None
    score_claim: bool = False
    pointer_moved: bool = False
    verdict_scope: str = VERDICT_SCOPE
    req_r: str = REQ_R

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_guard_receipt(micro_batch_pairs: int) -> MicroBatchMemoryGuardReceipt:
    return MicroBatchMemoryGuardReceipt(
        micro_batch_pairs=micro_batch_pairs,
        guard_gib=round(micro_batch_guard_gib(micro_batch_pairs), 2),
    )


__all__ = [
    "EQUATION_ID",
    "MEASURED_B1_N8_PROCESS_PEAK_MIB",
    "MEASURED_B4_N8_PROCESS_PEAK_MIB",
    "MICRO_BATCH_EXTRA_PAIR_GUARD_GIB",
    "MICRO_BATCH_GUARD_PROVENANCE",
    "REQ_R",
    "VERDICT_SCOPE",
    "MicroBatchMemoryGuardReceipt",
    "build_guard_receipt",
    "micro_batch_guard_gib",
]
