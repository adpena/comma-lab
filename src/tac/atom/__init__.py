# SPDX-License-Identifier: MIT
"""Canonical atom-shaped element for the meta-Lagrangian search engine.

Subsumes 7 atom-shaped surfaces scattered across the codebase into ONE
canonical type that:
  - composes with ``tac.unified_action.Action`` via the bridge in
    ``tac.atom.unified_action_bridge``
  - emits canonical observability per Catalog #305 6-facet
  - carries provenance per Catalog #323
  - declares wire-in per Catalog #125 6-hook
  - serves as element in the meta-Lagrangian search per CLAUDE.md
    "Meta-Lagrangian/Pareto solver" non-negotiable

Operator-approved design 2026-05-18 verbatim "yes" on the
synthesis-memo-amendment design pitch.

Why frozen dataclass + Enum + Protocol (NOT pydantic):
  - Matches existing canonical patterns (``tac.unified_action.Action``,
    ``tac.provenance.Provenance``, ``tac.council_continual_learning.\
    CouncilDeliberationRecord``).
  - Zero runtime overhead (no validation framework).
  - Native Python stdlib (no new dependency).
  - ``__post_init__`` validates per-kind constraints; canonical builders
    centralize creation per Catalog #131.

Catalog #125 hook 4 (cathedral autopilot dispatch): ACTIVE — atoms feed
into ``tac.cathedral_autopilot_autonomous_loop.rank_candidates`` as
canonical input atoms via
``tac.atom.unified_action_bridge.atom_pool_to_cathedral_autopilot_candidates``.

The seven subsumed atom-shaped surfaces:
  =====================  =======================================  ======================
  AtomKind               Source                                   Subsume helper
  =====================  =======================================  ======================
  ARBITRARY_VALUE        ``.omx/state/arbitrariness_extinction_   atom_from_arbitrariness
                         audit_*.jsonl`` (sister audit)           _audit_row
  META_LAGRANGIAN        ``tac.meta_lagrangian_allocator.atoms_   atom_from_meta_
                         from_hnerv_decoder_recode_profile``       lagrangian_row
  CARGO_CULT_ASSUMPTION  per-substrate ``## Cargo-cult audit per  atom_from_cargo_cult_
                         assumption`` section (Catalog #303)      audit_row
  PREMISE_VERIFICATION   per-edit Catalog #229 (informal)         atom_from_*
                                                                  (built via builder)
  PROBE_OUTCOME          ``.omx/state/probe_outcomes.jsonl``      atom_from_probe_
                         (Catalog #313)                           outcomes_ledger_row
  COUNCIL_DELIBERATION   ``.omx/state/council_deliberation_       atom_from_council_
                         posterior.jsonl`` (Catalog #300)         deliberation_record
  DISPATCH_CLAIM         ``.omx/state/active_lane_dispatch_       atom_from_dispatch_
                         claims.md``                              claim_row
  =====================  =======================================  ======================
"""
from __future__ import annotations

from .atom import (
    ATOM_SCHEMA_VERSION,
    OBSERVABILITY_FACETS_CANONICAL,
    WIRED_HOOKS_CANONICAL,
    Atom,
)
from .builders import (
    build_arbitrary_value_atom,
    build_cargo_cult_atom,
    build_council_deliberation_atom,
    build_dispatch_claim_atom,
    build_meta_lagrangian_atom,
    build_premise_verification_atom,
    build_probe_outcome_atom,
)
from .ledger import (
    ATOM_LEDGER_LOCK,
    ATOM_LEDGER_PATH,
    SCHEMA_VERSION,
    AtomLedgerCorruptError,
    append_atom,
    append_atoms_batch,
    load_atoms_strict,
    query_by_kind,
    query_by_min_predicted_impact,
    query_by_resolution_path,
)
from .subsumption import (
    atom_from_arbitrariness_audit_row,
    atom_from_cargo_cult_audit_row,
    atom_from_council_deliberation_record,
    atom_from_dispatch_claim_row,
    atom_from_meta_lagrangian_row,
    atom_from_probe_outcomes_ledger_row,
)
from .types import (
    AtomKind,
    AtomProtocol,
    AtomValidationError,
    ResolutionPath,
)
from .unified_action_bridge import (
    atom_pool_to_cathedral_autopilot_candidates,
    atom_pool_to_meta_lagrangian_ledger,
    evaluate_action_with_atoms,
)
from .contest_granularity import (
    BudgetVector,
    ByteScope,
    CONTEST_RATE_DENOMINATOR_BYTES,
    ContestAtom,
    ContestAtomError,
    ContestScopeKind,
    ContestSignal,
    FrameScope,
    PairScope,
    PixelRegionScope,
    ScoreVector,
    build_lattice_report,
    byte_atoms_from_master_gradient,
    frame_and_pixel_atoms_from_xray_row,
    merge_atoms_by_id,
    pair_atom_from_component_row,
    pair_signal_overlap,
    select_latest_master_gradient_anchor,
)

# --- Linguistic extensions (TOP-3 per design memo APPENDIX B) ---
# ADDITIVE extension landed 2026-05-18 alongside ``tac.contest_oracle``.
# Per CLAUDE.md "Subagent coherence-by-default" + Catalog #314 absorption
# discipline: new symbols only, no mutation of existing canonical API.
from .linguistic_extensions import (
    ArbitrarinessClassification,
    AtomAlgebraError,
    TemporalLogicViolationError,
    always_invariant,
    classify_atom_arbitrariness,
    complement_atom,
    compose_atoms,
    eventually_extincted_within,
    intersect_atoms,
    union_atoms,
    valid_until,
)

__all__ = [
    # types
    "AtomKind",
    "ResolutionPath",
    "AtomProtocol",
    "AtomValidationError",
    # atom dataclass + constants
    "Atom",
    "ATOM_SCHEMA_VERSION",
    "WIRED_HOOKS_CANONICAL",
    "OBSERVABILITY_FACETS_CANONICAL",
    # builders
    "build_arbitrary_value_atom",
    "build_meta_lagrangian_atom",
    "build_cargo_cult_atom",
    "build_premise_verification_atom",
    "build_probe_outcome_atom",
    "build_council_deliberation_atom",
    "build_dispatch_claim_atom",
    # subsumption helpers
    "atom_from_meta_lagrangian_row",
    "atom_from_cargo_cult_audit_row",
    "atom_from_probe_outcomes_ledger_row",
    "atom_from_council_deliberation_record",
    "atom_from_dispatch_claim_row",
    "atom_from_arbitrariness_audit_row",
    # ledger
    "ATOM_LEDGER_PATH",
    "ATOM_LEDGER_LOCK",
    "AtomLedgerCorruptError",
    "SCHEMA_VERSION",
    "append_atom",
    "append_atoms_batch",
    "load_atoms_strict",
    "query_by_kind",
    "query_by_resolution_path",
    "query_by_min_predicted_impact",
    # unified-action bridge
    "evaluate_action_with_atoms",
    "atom_pool_to_meta_lagrangian_ledger",
    "atom_pool_to_cathedral_autopilot_candidates",
    # contest-granularity atom lattice
    "BudgetVector",
    "ByteScope",
    "CONTEST_RATE_DENOMINATOR_BYTES",
    "ContestAtom",
    "ContestAtomError",
    "ContestScopeKind",
    "ContestSignal",
    "FrameScope",
    "PairScope",
    "PixelRegionScope",
    "ScoreVector",
    "build_lattice_report",
    "byte_atoms_from_master_gradient",
    "frame_and_pixel_atoms_from_xray_row",
    "merge_atoms_by_id",
    "pair_atom_from_component_row",
    "pair_signal_overlap",
    "select_latest_master_gradient_anchor",
    # linguistic extensions (TOP-3 per design memo APPENDIX B)
    "ArbitrarinessClassification",
    "AtomAlgebraError",
    "TemporalLogicViolationError",
    "always_invariant",
    "classify_atom_arbitrariness",
    "complement_atom",
    "compose_atoms",
    "eventually_extincted_within",
    "intersect_atoms",
    "union_atoms",
    "valid_until",
]
