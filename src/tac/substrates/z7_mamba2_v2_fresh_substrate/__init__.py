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
    "SUBSTRATE_ID",
    "ARCHITECTURE_CLASS",
    "CANONICAL_EQUATION_IDS",
    "emit_landing_posterior_anchor",
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


# ─── Canonical landing-time posterior emission (WAVE-1 wire-in 2026-05-26) ──
# Per OPTIMIZATION-TOOLING-AUDIT roadmap commit `e757bb74c` META #1 + the
# canonical helper at `tac.substrates._shared.posterior_emission_helper`:
# lifts this substrate's L0 SCAFFOLD signal into the cathedral autopilot's
# 62 auto-discovered consumers via the canonical posterior surfaces.

SUBSTRATE_ID: str = "z7_mamba2_v2_fresh_substrate"
ARCHITECTURE_CLASS: str = "z7_mamba2_v2_predictive_coding_state_space_l0_scaffold_mlx"

# Per WAVE-3 op-routable #3 the NEW canonical equation for this paradigm
# is queued: predictive_coding_residual_capacity_v1 (B'/D/F shared per
# the audit). Until registered in tac.canonical_equations, the manifest
# row's canonical_equation_ids carries the proposed-equation token so
# audit tooling can trace the lineage per Catalog #344.
CANONICAL_EQUATION_IDS: tuple[str, ...] = (
    "predictive_coding_residual_capacity_v1_proposed_per_audit_e757bb74c_op_routable_3",
)


def emit_landing_posterior_anchor(
    *,
    archive_sha256: str | None = None,
    archive_bytes: int = 10_000,
    source_path: str | None = None,
    predicted_score: float = 0.175,
    predicted_d_seg: float | None = 0.00105,
    predicted_d_pose: float | None = 0.000025,
    notes: str = (
        "L0 SCAFFOLD SKELETON MLX landing per WAVE-1 canonical posterior emission "
        "wire-in 2026-05-26 (audit commit e757bb74c META #1 closure). Z7 Mamba-2 "
        "v2 fresh-substrate temporal predictive-coding state-space (Path c per "
        "phase 2 decision memo). Non-promotable per CLAUDE.md MLX research-signal "
        "discipline. Skeleton-only at this anchor; implementation pending Phase 3 "
        "L0 SCAFFOLD landing per design memo."
    ),
    posterior_path: object | None = None,
    posterior_lock_path: object | None = None,
    manifest_path: object | None = None,
):
    """Emit canonical landing-time posterior anchor for this substrate.

    Per WAVE-1-POSTERIOR-EMISSION-CANONICAL-WIRE-IN charter 2026-05-26 +
    OPTIMIZATION-TOOLING-AUDIT META #1 CRITICAL finding closure: invokes
    the canonical helper at
    ``tac.substrates._shared.posterior_emission_helper.emit_substrate_landing_posterior_anchor``
    with this substrate's canonical identifiers + canonical equation IDs
    threaded through ``extra_manifest_fields`` for cathedral consumer
    observability.

    Lifts this substrate's signal into:
    - ``.omx/state/continual_learning_posterior.json`` (refused as
      advisory-grade per custody validator; bumps ``refused_anchor_count``)
    - ``.omx/state/mps_research_signal_manifest.jsonl`` (canonical MLX
      research-signal posterior; cathedral-queryable surface)

    Per Catalog #287/#323/#341: anchor is non-promotable by construction.
    Per Catalog #128 + #131 + #138 sister discipline: writes through
    canonical fcntl-locked helpers only.
    """
    from tac.substrates._shared.posterior_emission_helper import (
        emit_substrate_landing_posterior_anchor,
        synthesize_substrate_archive_sha256,
    )

    sha = archive_sha256 or synthesize_substrate_archive_sha256(SUBSTRATE_ID)
    src = source_path or (
        "src/tac/substrates/z7_mamba2_v2_fresh_substrate/"
        "__init__.py:emit_landing_posterior_anchor_l0_skeleton"
    )

    return emit_substrate_landing_posterior_anchor(
        substrate_id=SUBSTRATE_ID,
        archive_sha256=sha,
        archive_bytes=int(archive_bytes),
        source_path=src,
        predicted_score=predicted_score,
        predicted_d_seg=predicted_d_seg,
        predicted_d_pose=predicted_d_pose,
        architecture_class=ARCHITECTURE_CLASS,
        notes=notes,
        posterior_path=posterior_path,  # type: ignore[arg-type]
        posterior_lock_path=posterior_lock_path,  # type: ignore[arg-type]
        manifest_path=manifest_path,  # type: ignore[arg-type]
        extra_manifest_fields={
            "paradigm": "temporal_predictive_coding_state_space",
            "lane_class": "substrate_engineering",
            "horizon_class": SUBSTRATE_CLASS_SHIFT_HORIZON,
            "canonical_equation_ids": list(CANONICAL_EQUATION_IDS),
            "research_only": RESEARCH_ONLY,
            "dispatch_enabled": DISPATCH_ENABLED,
            "implementation_status": IMPLEMENTATION_STATUS,
            "phase_1_audit_path": PHASE_1_AUDIT_PATH,
            "phase_2_decision_path": PHASE_2_DECISION_PATH,
            "phase_3_design_path": PHASE_3_DESIGN_PATH,
        },
    )
