# SPDX-License-Identifier: MIT
"""Backend-neutral persistence and telemetry contract for witness training.

The active MLX trainer remains untouched while it is running.  This module fixes
the public row and stage-boundary vocabulary that a CUDA backend must emit so
observers do not need a backend-specific parser.  Values may differ by backend;
keys, stage names, and custody semantics may not.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LOSS_TERM_KEYS: tuple[str, ...] = (
    "seg", "pose", "eikonal", "length", "eik_steik", "boundary_distance",
    "lane_edge", "margin_saliency", "subpix", "chroma_boundary",
    "margin_satisfice", "horizon_margin", "temporal_screw", "phase_advect", "island_amplify",
    "area_constraint", "persistence", "rankfloor", "code_spectral",
    "thin_lane", "margin_field_head", "code_nuclear", "weight_entropy",
)

# NO-FAKE execution wall: each entry is an emitted, score-affecting typed-DSL
# surface whose stateful MLX semantics do not yet have a 1:1 Torch twin.
CUDA_V9_PORT_BLOCKERS: tuple[str, ...] = (
    "--ladder-island-homotopy eased targets and per-class lambda refresh",
    "--seed-islands / --birth-completion-event ramped classwise birth stack",
    "--muon-start-event AdamW-to-Muon transition with rewarmup and state persistence",
    "--tail-cycles-max governed tail-cycle stop controller",
)


def cuda_v9_port_receipt() -> dict[str, object]:
    """Return the machine-readable active-program semantic coverage receipt."""
    return {
        "schema": "cuda_v9_port_coverage.v1",
        "status": "BLOCKED_NOT_1_TO_1" if CUDA_V9_PORT_BLOCKERS else "COMPLETE_1_TO_1",
        "blockers": list(CUDA_V9_PORT_BLOCKERS),
        "score_bearing_primitives_ported": [
            "levelset_film_hosc_forward_and_contest_R",
            "seg_unified_tau_and_pose_score_domain_loss",
            "eikonal_length_chroma_boundary",
            "island_amplify_area_persistence_weight_entropy",
            "lane_receiver_band_phase_advection_temporal_screw",
            "numpy_fp32_forward_parity_and_backend_compile_probe",
            "atomic_resume_ema_and_per_stage_checkpoint_contract",
            "canonical_loss_terms_jsonl_contract",
            "generated_table_pose_carrier_frame0_dispatch_and_learnable_dxi",
            "structured_scorer_sdf_prefit_with_resume_suppression",
            "accum_pairs_8_chunk_atomic_updates_and_accepted_fraction",
            "dseg_aware_feature_taper_and_per_group_gradient_clipping",
            "scorer_derived_curriculum_latches_with_atomic_resume_state",
            "sigma_min_plateau_pose_finish_gate_with_degenerate_banked_r1",
            "polyak_finisher_resumable_additional_candidate_export",
        ],
    }

STAGE_ORDER: tuple[str, ...] = (
    "island_birth_boundary_form",
    "sharpen_repair",
    "muon_phase_finish",
    "polyak_finish",
)


def loss_terms_row(
    *,
    epoch: int,
    accum_batch: int,
    terms: Mapping[str, float],
    total: float,
    **extras: Any,
) -> dict[str, Any]:
    """Return the canonical ``loss_terms`` row with a stable complete key set."""
    stable_raw = {key: float(terms.get(key, 0.0)) for key in LOSS_TERM_KEYS}
    term_sum = float(sum(stable_raw.values()))
    stable = {key: round(value, 6) for key, value in stable_raw.items()}
    row: dict[str, Any] = {
        "stage": "loss_terms",
        "ep": int(epoch),
        "accum_batch": int(accum_batch),
        "terms": stable,
        "total": round(float(total), 6),
        "sum_terms": round(term_sum, 6),
        "sum_minus_total": round(term_sum - float(total), 8),
    }
    row.update(extras)
    return row


def curriculum_stage(epoch: int, flags: Mapping[str, Any]) -> str:
    """Derive the deterministic fail-safe stage from typed DSL epoch caps."""
    if epoch >= int(flags.get("--polyak-finisher-start-epoch", 2546)):
        return STAGE_ORDER[3]
    if epoch >= int(flags.get("--muon-start-epoch", 726)):
        return STAGE_ORDER[2]
    if epoch >= int(flags.get("--seg-chroma-boundary-start-epoch", 450)):
        return STAGE_ORDER[1]
    return STAGE_ORDER[0]


__all__ = [
    "CUDA_V9_PORT_BLOCKERS",
    "LOSS_TERM_KEYS",
    "STAGE_ORDER",
    "cuda_v9_port_receipt",
    "curriculum_stage",
    "loss_terms_row",
]
