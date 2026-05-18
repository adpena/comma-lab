# SPDX-License-Identifier: MIT
"""Canonical per-pair master gradient wire-in helper for §7.6 namespaces.

LOW gap closure widened wave 2026-05-17 — `lane_low_gap_closure_widened_8_modules_
plus_autopilot_wire_in_20260517` BUCKET A. Per CLAUDE.md "Subagent coherence-
by-default" + "Beauty, simplicity, and developer experience" + "UNIQUE-AND-
COMPLETE-PER-METHOD operating mode" Layer 1 canonical-helper-adoption-
serves-because-bug-class: this is the canonical wire-in helper shared across
the 5 §7.6 namespace packages (boosting / compress_time_optimization /
inflate_time_post_processing / side_information / search).

Each namespace re-exports ``compose_namespace_per_pair_wire_in(namespace=...)``
via its own per-namespace shim with the namespace-id baked in. The shim
preserves namespace-specific sister persistence (the §7.6 cargo-cult-audit
verdict: ``append_stage_outcome_locked`` / ``append_pass_outcome_locked`` etc.
are CORRECT engineering, NOT a gap; Hook 5 stays in the sister-persistence
domain, not ``tac.continual_learning`` proper).

Per CLAUDE.md "Apples-to-apples evidence discipline" the outcome is PLANNING-
ONLY — never emits a score claim, never promotes a candidate, never dispatches
a paid job.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Hook 1 sensitivity-reweight cascade | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.sensitivity_map.wyner_ziv_reweight.axis_level_reweight` is the canonical bias source |
| Hook 3 bit-allocation cascade | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.optimization.bit_allocator_end_to_end.allocate_per_pair_bits` is the canonical Hook 3 surface |
| Hook 5 continual-learning | FORK_BECAUSE_PRINCIPLED_MISMATCH | §7.6 namespaces use sister fcntl-locked stores (`append_stage_outcome_locked` etc.) per CLAUDE.md sister-store cargo-cult-audit verdict |
| Per-pair gradient loader | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.master_gradient_consumers.load_per_pair_gradient_from_anchor` is the single source of truth |
| Wyner-Ziv classification | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.master_gradient_consumers.wyner_ziv_side_info_covariance` is the canonical classifier |
| Per-namespace shim | FORK_BECAUSE_PRINCIPLED_MISMATCH | each namespace's __init__ exports its own thin wrapper with namespace-id baked in for trace clarity |
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "LEGAL_NAMESPACE_IDS",
    "NAMESPACE_WIRE_IN_SCHEMA",
    "NamespacePerPairWireInOutcome",
    "compose_namespace_per_pair_wire_in",
]

NAMESPACE_WIRE_IN_SCHEMA = "tac_namespace_per_pair_wire_in_v1"

# Per CLAUDE.md "Anti-fragmentation: unified-Lagrangian action": every §7.6
# namespace MUST be enumerated here so a future namespace addition is
# structurally surfaced (sister of `LEGAL_HOOK_*` constants in the contracts).
LEGAL_NAMESPACE_IDS: frozenset[str] = frozenset(
    {
        "boosting",
        "compress_time_optimization",
        "inflate_time_post_processing",
        "side_information",
        "search",
    }
)


@dataclass(frozen=True)
class NamespacePerPairWireInOutcome:
    """Typed wire-in outcome for §7.6 namespace per-pair master gradient cascade.

    Per CLAUDE.md "Beauty, simplicity, and developer experience": typed +
    JSON-safe + frozen + carries axis tagging at the dataclass surface.

    Fields ``hook_1_*`` and ``hook_3_*`` track per-Hook consumption status;
    downstream pipelines can inspect ``cascade_path_used`` to decide whether
    to inject ``per_byte_bit_allocation`` into a stage's archive-byte budget
    OR fall back to the legacy uniform allocation.
    """

    schema: str
    namespace_id: str  # one of LEGAL_NAMESPACE_IDS
    archive_sha256: str
    hook_1_sensitivity_reweight_consumed: bool
    hook_3_bit_allocation_consumed: bool
    per_byte_bit_allocation: dict[int, int]  # byte_index -> allocated bytes
    total_allocated_bytes: int
    total_bit_budget: int
    cascade_path_used: str  # "optimal_plan" / "wyner_ziv_composition" / "aggregate_fallback" / "no_signal"
    rationale: str
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    evidence_grade: str = "[predicted; namespace per-pair wire-in v1]"


def compose_namespace_per_pair_wire_in(
    *,
    namespace_id: str,
    archive_sha256: str,
    total_bit_budget: int,
    sensitivity_reweight: Mapping[int, float] | None = None,
    auto_load: bool = True,
) -> NamespacePerPairWireInOutcome:
    """Compose Hook 1 + Hook 3 wire-in for a §7.6 namespace.

    Per `.omx/research/comprehensive_wire_in_coverage_matrix_20260517.md`
    §7.6 GAP closure + sister #817 cascade pattern:

      1. **Hook 1**: if ``sensitivity_reweight`` is None and ``auto_load=True``,
         compute the canonical per-byte reweight via
         ``tac.sensitivity_map.wyner_ziv_reweight.axis_level_reweight`` over
         the archive's Wyner-Ziv side-info classification anchor.
      2. **Hook 3**: invoke ``tac.optimization.bit_allocator_end_to_end.
         allocate_per_pair_bits`` with the resolved reweight + auto-loaded
         optimal plan / per-pair gradient per the sister #817 cascade
         (OptimalPlan > Wyner-Ziv > aggregate fallback).
      3. **Hook 5**: NOT called here — each §7.6 namespace has its OWN
         canonical fcntl-locked sister store (``append_stage_outcome_locked``
         for boosting, ``append_pass_outcome_locked`` for
         compress_time_optimization, etc.) per CLAUDE.md sister-store
         cargo-cult-audit verdict. Callers persist via that path explicitly
         with the outcome of running the namespace's pipeline (which may
         consume this wire-in envelope).

    Per CLAUDE.md "Apples-to-apples evidence discipline" the outcome is
    `[predicted; namespace per-pair wire-in v1]` — NO score claim.

    Parameters
    ----------
    namespace_id
        One of LEGAL_NAMESPACE_IDS (boosting / compress_time_optimization /
        inflate_time_post_processing / side_information / search).
    archive_sha256
        64-char hex sha of the target archive bytes.
    total_bit_budget
        Hard global byte cap for the per-pair allocation cascade.
    sensitivity_reweight
        Optional pre-resolved per-byte reweight map. When None and
        ``auto_load=True``, attempts Hook 1 cascade auto-resolution.
    auto_load
        When True (default), the helpers auto-load missing inputs from canonical
        anchors / sidecars per the canonical loader contract.

    Returns
    -------
    NamespacePerPairWireInOutcome with per-byte allocation + cascade evidence.

    Raises
    ------
    ValueError
        On invalid namespace_id, negative budget, or malformed archive sha.
    """
    if namespace_id not in LEGAL_NAMESPACE_IDS:
        raise ValueError(
            f"namespace_id={namespace_id!r} not in LEGAL_NAMESPACE_IDS="
            f"{sorted(LEGAL_NAMESPACE_IDS)}"
        )
    if not isinstance(total_bit_budget, int) or total_bit_budget < 0:
        raise ValueError(
            f"total_bit_budget must be non-negative int; got {total_bit_budget!r}"
        )
    if (
        not isinstance(archive_sha256, str)
        or len(archive_sha256) < 12
        or any(c not in "0123456789abcdefABCDEF" for c in archive_sha256)
    ):
        raise ValueError(
            f"archive_sha256 must be a 12+ char hex string; got {archive_sha256!r}"
        )

    # ── Hook 1: sensitivity-reweight cascade ──────────────────────────────
    hook_1_consumed = False
    resolved_reweight: Mapping[int, float] | None = sensitivity_reweight
    if resolved_reweight is None and auto_load:
        try:
            from tac.sensitivity_map.wyner_ziv_reweight import axis_level_reweight
            from tac.master_gradient_consumers import (
                load_per_pair_gradient_from_anchor,
                wyner_ziv_side_info_covariance,
            )

            per_pair, anchor = load_per_pair_gradient_from_anchor(
                archive_sha256=archive_sha256
            )
            wz_classification = wyner_ziv_side_info_covariance(
                per_pair,
                archive_sha256=archive_sha256,
                measurement_axis=str(anchor.get("measurement_axis", "pose")),
                measurement_hardware=str(
                    anchor.get("measurement_hardware", "linux_x86_64_unknown_cuda")
                ),
                write_sidecar=False,
            )
            resolved_reweight = axis_level_reweight(wz_classification)
            hook_1_consumed = resolved_reweight is not None
        except (
            ImportError,
            ValueError,
            FileNotFoundError,
            OSError,
            AttributeError,
            TypeError,
        ):
            resolved_reweight = None
            hook_1_consumed = False
    elif resolved_reweight is not None:
        hook_1_consumed = True

    # ── Hook 3: bit-allocation cascade (delegates to sister #817 helper) ──
    try:
        from tac.optimization.bit_allocator_end_to_end import allocate_per_pair_bits

        outcome = allocate_per_pair_bits(
            archive_sha256=archive_sha256,
            total_bit_budget=total_bit_budget,
            sensitivity_reweight=resolved_reweight,
            auto_load=auto_load,
        )
        hook_3_consumed = True
        cascade_path = outcome.cascade_path_used
        per_byte_bit_allocation = dict(outcome.per_byte_bit_allocation)
        total_allocated = outcome.total_allocated_bytes
        rationale = (
            f"[predicted; namespace per-pair wire-in v1] namespace={namespace_id} "
            f"Hook 1 consumed={hook_1_consumed} via wyner_ziv_reweight cascade; "
            f"Hook 3 cascade={cascade_path}; {total_allocated} of "
            f"{total_bit_budget} bytes allocated; Hook 5 preserved via sister "
            f"fcntl-locked store in tac.{namespace_id}.persistence (NOT replaced)."
        )
    except (ImportError, ValueError, FileNotFoundError, OSError):
        hook_3_consumed = False
        cascade_path = "no_signal"
        per_byte_bit_allocation = {}
        total_allocated = 0
        rationale = (
            f"[predicted; namespace per-pair wire-in v1] namespace={namespace_id} "
            f"Hook 1 consumed={hook_1_consumed}; Hook 3 NOT consumed (signal "
            f"unavailable); no signal; Hook 5 preserved."
        )

    return NamespacePerPairWireInOutcome(
        schema=NAMESPACE_WIRE_IN_SCHEMA,
        namespace_id=namespace_id,
        archive_sha256=archive_sha256,
        hook_1_sensitivity_reweight_consumed=hook_1_consumed,
        hook_3_bit_allocation_consumed=hook_3_consumed,
        per_byte_bit_allocation=per_byte_bit_allocation,
        total_allocated_bytes=total_allocated,
        total_bit_budget=total_bit_budget,
        cascade_path_used=cascade_path,
        rationale=rationale,
    )
