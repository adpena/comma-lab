"""Centralized registry of all landed tracks (T7-T22 + Lane-12-v2 + A1).

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: "Every stackable
or substitutive idea should move toward a typed row consumed by the planner".
This registry is THE typed-row contract — every consumer (cathedral_autopilot,
meta_lagrangian, pareto, field_equation_planner, unified_action) reads from
this single source of truth so a track that lands here is automatically
visible to all solver components.

Wire-in points
--------------
- :mod:`tac.unified_action` — TrackKind enum mirrors registry keys
- :mod:`tac.continual_learning` — track_correction_posteriors keyed by registry id
- :mod:`tools.cathedral_autopilot` — plan-row schema includes registry tags
- :mod:`tac.optimizer.meta_lagrangian` — distortion_proxy parameterized by active set

CLAUDE.md compliance
--------------------
- Each track entry carries explicit ``[empirical:<artifact>]`` /
  ``[predicted:<method>]`` provenance per CLAUDE.md "FORBIDDEN_PATTERNS:
  empirical-claim-without-evidence-tag".
- ``promotion_eligible`` defaults to False — a track is dispatch-eligible only
  after its entry-conditions clear.
- ``planner_visibility`` field controls which solver components see the track,
  matching the audit table in
  ``.omx/research/unified_solver_integration_audit_20260509.json``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

TRACK_REGISTRY_SCHEMA_VERSION = "tac_track_registry_v1"


class ParetoAxis(str, Enum):
    """Pareto axes a track refines / contributes to."""

    SEG = "seg"
    POSE = "pose"
    RATE = "rate"
    TEMPORAL = "temporal"
    MULTI = "multi"  # contributes to >= 2 axes
    NONE = "none"  # numerical-solver step (no axis)


class TrackPhase(str, Enum):
    """Where in the contest pipeline the track participates."""

    LOSS_TERM = "loss_term"  # part of training loss
    REGULARIZER = "regularizer"  # auxiliary regularizer
    RATE_BOUND = "rate_bound"  # rate-axis floor / constraint
    NUMERICAL_SOLVER = "numerical_solver"  # ADMM step / dual update
    ARCHITECTURE = "architecture"  # architecture-class atom
    SUBSTRATE = "substrate"  # empirical substrate anchor


@dataclass(frozen=True)
class TrackEntry:
    """One canonical track entry."""

    track_id: str  # canonical identifier (matches TrackKind enum value)
    module_path: str  # repo-relative path to canonical module
    kind_summary: str  # one-line description
    phase: TrackPhase
    pareto_axis: ParetoAxis
    landed_commit_or_memo: str  # commit SHA or memory file reference
    evidence_grade: str  # one of [empirical:...] / [predicted:...] / [contest-CUDA] / etc.
    planner_visibility: tuple[str, ...]  # solver components that can see this track
    entry_conditions: tuple[str, ...] = ()  # what must be true before promotion
    promotion_eligible: bool = False
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# Canonical registry — matches the audit table in .omx/research/unified_solver_integration_audit_20260509.json.
TRACK_REGISTRY: dict[str, TrackEntry] = {
    "t7_fisher_rao": TrackEntry(
        track_id="t7_fisher_rao",
        module_path="src/tac/score_geometry_shannon_floor.py + src/tac/score_geometry.py",
        kind_summary="Riemannian distance closed-form on mask probability simplex; convex envelope of IoU gradient",
        phase=TrackPhase.LOSS_TERM,
        pareto_axis=ParetoAxis.SEG,
        landed_commit_or_memo="prior session (Shannon floor base)",
        evidence_grade="[predicted; closed-form]",
        planner_visibility=(
            "ib_lagrangian_aux_scorer",
            "unified_action",
            "non_arbitrariness_probe",
        ),
        entry_conditions=("probe_seg_loss_surrogate_disambiguator verdict KEEP/ENSEMBLE",),
        notes="probe_disambiguator current verdict: T8-alone for Phase 1; T7 deferred",
    ),
    "t8_sinkhorn_w2": TrackEntry(
        track_id="t8_sinkhorn_w2",
        module_path="src/tac/losses.py::sinkhorn_w2_mask_distortion_per_pixel",
        kind_summary="Wasserstein-2 mask surrogate via Sinkhorn (entropic OT)",
        phase=TrackPhase.LOSS_TERM,
        pareto_axis=ParetoAxis.SEG,
        landed_commit_or_memo="src/tac/losses.py L244 (active)",
        evidence_grade="[empirical: probe_seg_loss_surrogate_disambiguator]",
        planner_visibility=(
            "unified_action",
            "non_arbitrariness_probe",
        ),
        entry_conditions=(),
        promotion_eligible=True,
        notes="Phase 1 winner per probe-disambiguator T8-alone verdict",
    ),
    "t11_lovasz_hinge": TrackEntry(
        track_id="t11_lovasz_hinge",
        module_path="src/tac/lovasz_hinge.py",
        kind_summary="Convex envelope of IoU/argmax-disagreement (Berman 2018 CVPR §3.2)",
        phase=TrackPhase.LOSS_TERM,
        pareto_axis=ParetoAxis.SEG,
        landed_commit_or_memo="feedback_t11_t13_t19_free_lateral_leaps_landed_20260509.md",
        evidence_grade="[predicted; closed-form]",
        planner_visibility=(
            "unified_action",
            "non_arbitrariness_probe",
        ),
        entry_conditions=("probe verdict KEEP/ENSEMBLE",),
        notes="Phase 2 candidate; awaiting probe re-run with empirical signal",
    ),
    "t13_joint_source_rd": TrackEntry(
        track_id="t13_joint_source_rd",
        module_path="src/tac/joint_source_rd_bound.py",
        kind_summary="Fridrich √n undetectable embedding bound; Berger 1971 §4.5 Gauss-Markov rate floor",
        phase=TrackPhase.RATE_BOUND,
        pareto_axis=ParetoAxis.RATE,
        landed_commit_or_memo="b719386a (T13 + T19 wire-in into Phase 1 Ballé hyperprior trainer)",
        evidence_grade="[empirical: experiments/train_score_gradient_pr101_finetune.py]",
        planner_visibility=(
            "ib_lagrangian_aux_scorer",
            "joint_admm_coordinator",
            "unified_action",
            "meta_lagrangian_search",
            "pareto_3axis",
            "field_equation_planner",
        ),
        entry_conditions=(),
        promotion_eligible=True,
        notes="Already integrated into Phase 1 Ballé hyperprior trainer",
    ),
    "t19_adaptive_rho": TrackEntry(
        track_id="t19_adaptive_rho",
        module_path="src/tac/joint_admm_coordinator.py::adaptive_rho_step",
        kind_summary="Boyd 2011 §3.4.1 / He-Yang 2000 adaptive penalty step",
        phase=TrackPhase.NUMERICAL_SOLVER,
        pareto_axis=ParetoAxis.NONE,
        landed_commit_or_memo="feedback_t11_t13_t19_free_lateral_leaps_landed_20260509.md",
        evidence_grade="[empirical: 2-3x ADMM convergence speedup]",
        planner_visibility=("joint_admm_coordinator",),
        entry_conditions=(
            "run_admm migration to use adaptive_rho_step helper "
            "(currently uses inline Q4B + steady-state logic)"
        ),
        notes="ORPHAN: helper exists; coordinator hasn't migrated yet",
    ),
    "t20_kl_pose_distill": TrackEntry(
        track_id="t20_kl_pose_distill",
        module_path="src/tac/kl_pose_distill.py",
        kind_summary="KL-divergence pose-distill loss (Hinton T=2.0 form)",
        phase=TrackPhase.LOSS_TERM,
        pareto_axis=ParetoAxis.POSE,
        landed_commit_or_memo="a2f4677c",
        evidence_grade="[predicted; Hinton/Vinyals/Dean 2014]",
        planner_visibility=(
            "unified_action",
            "ib_lagrangian_aux_scorer",
            "cathedral_autopilot",
        ),
        entry_conditions=("trainer wire-in (currently a standalone loss; no trainer consumes it)",),
        notes="Promotion gated on trainer integration",
    ),
    "t22_temporal_consistency": TrackEntry(
        track_id="t22_temporal_consistency",
        module_path="src/tac/temporal_consistency_regularizer.py",
        kind_summary="Optical-flow-warped frame-to-frame smoothness regularizer",
        phase=TrackPhase.REGULARIZER,
        pareto_axis=ParetoAxis.TEMPORAL,
        landed_commit_or_memo="a2f4677c",
        evidence_grade="[predicted; closed-form]",
        planner_visibility=(
            "unified_action",
            "cathedral_autopilot",
        ),
        entry_conditions=("trainer wire-in",),
        notes="Phase 2 stack option",
    ),
    "lane_12_v2_nerv_as_renderer": TrackEntry(
        track_id="lane_12_v2_nerv_as_renderer",
        module_path="src/tac/lane_12_v2_nerv_as_renderer.py + src/tac/inflate/lane_12_v2_inflate.py",
        kind_summary="NeRV-class renderer with int8+fp16-scale weight quant + uint8-delta-split latent table",
        phase=TrackPhase.ARCHITECTURE,
        pareto_axis=ParetoAxis.MULTI,
        landed_commit_or_memo="cda997d7 (Phase A: design + module + inflate + 39 tests)",
        evidence_grade="[predicted; phase-A scaffold]",
        planner_visibility=(
            "unified_action",
            "cathedral_autopilot",
            "meta_lagrangian_search",
            "pareto_3axis",
            "field_equation_planner",
            "continual_learning_posterior",
        ),
        entry_conditions=(
            "Phase B preconditions per phase_b_preconditions_status()",
            "first [contest-CUDA] anchor",
        ),
        notes="New architecture class; continual-learning posterior must initialize uncalibrated-default",
    ),
    "a1_substrate": TrackEntry(
        track_id="a1_substrate",
        module_path="(latent-aligned 178,262 B archive at 0.19284 [contest-CPU GHA Linux x86_64]; sha 87ec7ca5...492b5)",
        kind_summary="Empirical anchor substrate signal",
        phase=TrackPhase.SUBSTRATE,
        pareto_axis=ParetoAxis.MULTI,
        landed_commit_or_memo="feedback_grand_council_a1_post_cpu_anchor_strategy_20260509.md",
        evidence_grade="[contest-CPU GHA Linux x86_64]",
        planner_visibility=(
            "meta_lagrangian_search",
            "pareto_3axis",
            "cathedral_autopilot",
            "continual_learning_posterior",
            "field_equation_planner",
        ),
        entry_conditions=(),
        promotion_eligible=True,
        notes="Empirical anchor; consumed by predictor + Pareto frontier",
    ),
}


def get_track(track_id: str) -> TrackEntry:
    """Look up a track by id; raise KeyError with the canonical list if missing."""
    if track_id not in TRACK_REGISTRY:
        raise KeyError(
            f"track_id {track_id!r} not in TRACK_REGISTRY; valid ids: "
            f"{sorted(TRACK_REGISTRY.keys())}"
        )
    return TRACK_REGISTRY[track_id]


def list_tracks_visible_to(component: str) -> list[TrackEntry]:
    """Return every track whose ``planner_visibility`` includes ``component``."""
    return [
        t for t in TRACK_REGISTRY.values()
        if component in t.planner_visibility
    ]


def list_promotable_tracks() -> list[TrackEntry]:
    """Return every track whose ``promotion_eligible`` is True."""
    return [t for t in TRACK_REGISTRY.values() if t.promotion_eligible]


def list_tracks_by_pareto_axis(axis: ParetoAxis) -> list[TrackEntry]:
    """Return every track contributing to a given Pareto axis."""
    return [t for t in TRACK_REGISTRY.values() if t.pareto_axis == axis]


def list_tracks_by_phase(phase: TrackPhase) -> list[TrackEntry]:
    """Return every track in a given pipeline phase."""
    return [t for t in TRACK_REGISTRY.values() if t.phase == phase]


__all__ = [
    "TRACK_REGISTRY_SCHEMA_VERSION",
    "ParetoAxis",
    "TrackPhase",
    "TrackEntry",
    "TRACK_REGISTRY",
    "get_track",
    "list_tracks_visible_to",
    "list_promotable_tracks",
    "list_tracks_by_pareto_axis",
    "list_tracks_by_phase",
]
