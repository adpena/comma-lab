# SPDX-License-Identifier: MIT
# LEGACY_SUBSTRATE_PRE_META_LAYER:L0_SCAFFOLD_skeleton_per_cargo_cult_first_methodology_phase_3_design_memo_2026_05_26_meta_layer_decorator_adoption_deferred_to_L1_per_HNeRV_parity_L7_substrate_engineering_split
"""Z7-Mamba-2-v2 fresh substrate (Path 3 candidate B') — L0 SCAFFOLD.

NEW substrate dir created per Phase 2 design decision memo
(`.omx/research/path_3_b_z7_mamba_2_substrate_design_decision_20260526.md`):
Path (c) FRESH SUBSTRATE DESIGN from first principles after rigorous
adversarial cargo-cult pass on existing `time_traveler_l5_z7_mamba2`
scaffold (`.omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`).

Why a NEW dir and not extension of the existing `time_traveler_l5_z7_mamba2`:
per Phase 1 audit's 8 NEW CARGO-CULTED assumptions (CC-A through CC-J),
the existing scaffold inherits 4 orthogonal axes of bolt-on engineering
(decoder force-fit Z6 / latent-dim Z7-LSTM-sister / training-pathway
sequential-autoregress / Z7MCM2 grammar inheritance). The 2026-05-26
binding operator directives mandate cargo-cult-pass-first methodology +
design-the-whole-stack-around-the-substrate approach. Per HNeRV parity
discipline L7 (bolt-on vs substrate-engineering split): substrate
engineering happens ONCE per architecture class; the v2 prefix encodes
the structural class-shift.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
``RESEARCH_ONLY = True`` until L1 EMPIRICAL anchor + Catalog #324
post-training Tier-C validation lands per the canonical
6-step per-substrate symposium contract (Catalog #325).

Per CLAUDE.md "Forbidden premature KILL": existing
`tac.substrates.time_traveler_l5_z7_mamba2` is PRESERVED as v1 historical
sister per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE; v2 is
the substrate-class-shift candidate, not a replacement for v1.

[verified-against: .omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md]
[verified-against: .omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md]
[verified-against: .omx/research/path_3_b_z7_mamba_2_substrate_design_decision_20260526.md]
"""

from __future__ import annotations

__all__ = (
    "RESEARCH_ONLY",
    "IMPLEMENTATION_STATUS",
    "DISPATCH_ENABLED",
    "SUBSTRATE_CLASS_SHIFT_HORIZON",
    "PLANNED_PUBLIC_API",
    "PHASE_1_AUDIT_PATH",
    "PHASE_2_DECISION_PATH",
    "PHASE_3_DESIGN_PATH",
)

RESEARCH_ONLY: bool = True
"""Substrate is research-only; no paid CUDA dispatch until L1+ post-training Tier-C anchor lands (Catalog #324)."""

DISPATCH_ENABLED: bool = False
"""No operator-authorize recipe shipped at L0; deferred to L1 per Catalog #240 recipe-vs-trainer-state consistency."""

IMPLEMENTATION_STATUS: str = (
    "L0_scaffold_skeleton_only_design_complete_implementation_pending_per_phase_3_L0_SCAFFOLD_landing_memo"
)
"""Coarse status string for `tools/audit_stale_l1_substrates.py` + cathedral autopilot ranker."""

SUBSTRATE_CLASS_SHIFT_HORIZON: str = "frontier_pursuit"
"""Per Catalog #309 horizon_class declaration; P50 ΔS=-0.018 → score ~0.175 in frontier_pursuit upper-region."""

PHASE_1_AUDIT_PATH: str = (
    ".omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md"
)
"""Phase 1 cargo-cult audit memo (8 NEW CARGO-CULTED assumptions surfaced)."""

PHASE_2_DECISION_PATH: str = (
    ".omx/research/path_3_b_z7_mamba_2_substrate_design_decision_20260526.md"
)
"""Phase 2 design-decision memo (Path c FRESH SUBSTRATE selected)."""

PHASE_3_DESIGN_PATH: str = (
    ".omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md"
)
"""Phase 3 L0 SCAFFOLD design memo (this scaffold's binding spec)."""

PLANNED_PUBLIC_API: tuple[str, ...] = (
    "Z7Mamba2V2Config",
    "Z7Mamba2V2Substrate",
    "Mamba2TemporalDecoder",
    "Mamba2V2Cell",
    "Z7MCM3Archive",
    "pack_archive",
    "parse_archive",
    "replay_latent_sequence",
    "inflate_one_video",
)
"""Public API surface to land at Phase 3 L0 SCAFFOLD implementation + L1 trainer build."""
