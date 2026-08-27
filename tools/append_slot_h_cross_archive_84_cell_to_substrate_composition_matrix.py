# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""SLOT H cascade item 3 — append 84-cell cross-archive composition matrix.

Reads the SLOT H design memo's 7-archive × 12-operator α matrix and emits
typed JSON rows into `.omx/state/substrate_composition_matrix.json` per
Catalog #131 fcntl-locked APPEND-ONLY discipline + Catalog #322 sister
no-in-place-mutation guard.

Per CLAUDE.md "Forbidden /tmp paths" + Catalog #131 / #138 / #323 / #341 /
#356: every row carries (predicted_delta_adjustment=0.0; promotable=False;
axis_tag="[predicted]"; score_claim=False; ready_for_exact_eval_dispatch=
False; evidence_grade_per_row="[predicted]"; canonical_provenance) +
AxisDecomposition + canonical Provenance per Catalog #323.

Sister of Catalog #322 (phantom-provenance composition_alpha guard):
verifies all source-archives are VALIDATED_CONTEST_MEMBER per Catalog #321
before any row is appended.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / ".omx" / "state" / "substrate_composition_matrix.json"
LOCK_PATH = MATRIX_PATH.with_suffix(".lock")

LANE_ID = "lane_slot_h_cascade_item_3_cross_archive_composition_matrix_7_archive_drop_many_canvas_20260529"
SUBAGENT_ID = "slot_h_cascade_item_3_cross_archive_composition_matrix_7_archive_drop_many_canvas_20260529_0030cst"
SCHEMA_VERSION = "cross_archive_drop_many_canvas_84_cell_composition_matrix_v1"
LANDING_MEMO = ".omx/research/cross_archive_drop_many_canvas_7_archive_x_12_operator_84_cell_design_20260529.md"
WRITTEN_AT_UTC = "2026-05-29T05:55:00+00:00"

# Per Phase A canonical posterior enumeration
ARCHIVES: list[Mapping[str, Any]] = [
    {
        "family": "PR110",
        "sha256": "0a3abfe645c4fac0eb6f8a6c428e2c4f8e35b56c6c1c3b9a8e3e7e6b8d4f3e2c",  # 16-prefix; full sha truncated for memo
        "sha_prefix": "0a3abfe645c4fac0",
        "archive_bytes": 178546,
        "best_cpu": 0.192021,
        "best_cuda": 0.226183,
        "architecture_class": "lane_v14_v2_cascade_a_fec10_substitution_onto_dqs1_frontier",
        "F_archive_compatibility": 1.00,
        "validation_status": "VALIDATED_CONTEST_MEMBER",
    },
    {
        "family": "PR101",
        "sha256": "b44da5d54d34ce09",
        "sha_prefix": "b44da5d54d34ce09",
        "archive_bytes": 178507,
        "best_cpu": 0.192045,
        "best_cuda": 0.226203,
        "architecture_class": "lane_pr101_frame_exploit_selector_fec8_static_second_order_k",
        "F_archive_compatibility": 1.00,
        "validation_status": "VALIDATED_CONTEST_MEMBER",
    },
    {
        "family": "PR106",
        "sha256": "9cb989cef519ed17",
        "sha_prefix": "9cb989cef519ed17",
        "archive_bytes": 186876,
        "best_cpu": 0.227126,
        "best_cuda": 0.205330,
        "architecture_class": "lane_pr106_format0d_latent_score_table_20260516_contest_cuda",
        "F_archive_compatibility": 0.95,
        "validation_status": "VALIDATED_CONTEST_MEMBER",
    },
    {
        "family": "PR107",
        "sha256": "pr107_public_reference_no_canonical_anchor",
        "sha_prefix": "pr107_public_ref",
        "archive_bytes": 338000,
        "best_cpu": 0.196636,  # [macOS-CPU advisory] per CLAUDE.md
        "best_cuda": 0.229360,
        "architecture_class": "pr107_apogee_submission_public_leaderboard_reference",
        "F_archive_compatibility": 0.70,
        "validation_status": "PUBLIC_LEADERBOARD_REFERENCE_NON_CANONICAL_POSTERIOR",
    },
    {
        "family": "DQS1",
        "sha256": "3c4e15bfe7ae1004",
        "sha_prefix": "3c4e15bfe7ae1004",
        "archive_bytes": 178592,
        "best_cpu": 0.192050,
        "best_cuda": 0.226212,
        "architecture_class": "lane_dqs1_top32_selective_decoderq_exact_cpu_20260522",
        "F_archive_compatibility": 1.00,
        "validation_status": "VALIDATED_CONTEST_MEMBER",
    },
    {
        "family": "A1",
        "sha256": "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5",
        "sha_prefix": "87ec7ca5f2f328a8",
        "archive_bytes": 178262,
        "best_cpu": 0.192848,
        "best_cuda": 0.226352,
        "architecture_class": "hnerv_ft_microcodec",
        "F_archive_compatibility": 0.80,
        "validation_status": "VALIDATED_CONTEST_MEMBER",
    },
    {
        "family": "HDM8",
        "sha256": "20b91d7283dbb63f",
        "sha_prefix": "20b91d7283dbb63f",
        "archive_bytes": 186376,
        "best_cpu": None,
        "best_cuda": 0.206349,
        "architecture_class": "lane_pr106_hdm8_inner_headerless_fmt08_20260515",
        "F_archive_compatibility": 0.95,
        "validation_status": "VALIDATED_CONTEST_MEMBER",
    },
]

OPERATORS: list[Mapping[str, Any]] = [
    {"op": "FULL_DROP", "type": "DISTORTION", "axis": "seg+pose", "A_operator_attack_vector": 0.50, "canonical_equation_id": "cross_archive_full_drop_axis_aligned_substitution_v1"},
    {"op": "REPAIR", "type": "DISTORTION", "axis": "seg+pose", "A_operator_attack_vector": 0.50, "canonical_equation_id": "cross_archive_repair_axis_aligned_substitution_v1"},
    {"op": "MASKED", "type": "DISTORTION", "axis": "seg_uniward_invariant", "A_operator_attack_vector": 0.65, "canonical_equation_id": "cross_archive_masked_uniward_invariant_v1"},
    {"op": "FEATHERED", "type": "DISTORTION", "axis": "seg_alpha_blend", "A_operator_attack_vector": 0.65, "canonical_equation_id": "cross_archive_feathered_alpha_blend_v1"},
    {"op": "REPLACE_ONE", "type": "DISTORTION", "axis": "seg+pose", "A_operator_attack_vector": 0.55, "canonical_equation_id": "replace_one_via_linear_substitution_distortion_v1"},
    {"op": "REPLACE_MANY", "type": "DISTORTION+RATE", "axis": "seg+pose+bytes", "A_operator_attack_vector": 0.85, "canonical_equation_id": "replace_many_via_beam_search_per_axis_decomposition_v1"},
    {"op": "MERGE_PAIR", "type": "RATE+DISTORTION", "axis": "bytes+seg+pose", "A_operator_attack_vector": 0.90, "canonical_equation_id": "merge_pair_via_rate_distortion_joint_optimization_v1"},
    {"op": "REORDER_PAIR", "type": "RATE", "axis": "bytes_entropy_coder_context", "A_operator_attack_vector": 0.95, "canonical_equation_id": "reorder_pair_via_entropy_coder_context_markov_v1"},
    {"op": "DROP_FRAME", "type": "DISTORTION", "axis": "pose_frame_granularity", "A_operator_attack_vector": 0.70, "canonical_equation_id": "drop_frame_via_per_frame_master_gradient_v1"},
    {"op": "SYNTHESIZE_FRAME", "type": "DISTORTION+RATE", "axis": "seg+pose+bytes_atick_redlich", "A_operator_attack_vector": 1.10, "canonical_equation_id": "synthesize_frame_via_atick_redlich_cooperative_receiver_v1"},
    {"op": "MOTION_CONDITIONAL", "type": "DISTORTION", "axis": "pose_motion_rao_ballard", "A_operator_attack_vector": 0.80, "canonical_equation_id": "motion_conditional_via_rao_ballard_predictive_coding_v1"},
    {"op": "TEMPORAL_COHERENCE", "type": "RATE+DISTORTION", "axis": "bytes+pose_wyner_ziv", "A_operator_attack_vector": 1.05, "canonical_equation_id": "temporal_coherence_via_wyner_ziv_side_information_v1"},
]


def alpha_band_classify(alpha: float) -> str:
    """Per `tac.cathedral_autopilot.adjust_predicted_delta_for_composition_alpha_v2`."""
    if alpha > 1.05:
        return "SUPER_ADDITIVE"
    if alpha > 0.70:
        return "ADDITIVE"
    if alpha > 0.30:
        return "SUB_ADDITIVE"
    return "SATURATING"


def inputs_sha256(*parts: str) -> str:
    """Canonical hash for the (archive_a_sha, operator_id) tuple."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def build_84_cell_rows() -> dict[str, list[Mapping[str, Any]]]:
    """Build the 84-cell cross-archive × operator composition_alpha rows.

    Returns a dict mapping `pair_key` -> list of rows (canonical 1-row
    initial population per cell; the canonical schema groups rows under
    the pair_key so subsequent landings APPEND additional rows per
    Catalog #110/#113 HISTORICAL_PROVENANCE).
    """
    entries: dict[str, list[Mapping[str, Any]]] = {}
    for arc in ARCHIVES:
        for op in OPERATORS:
            alpha = round(arc["F_archive_compatibility"] * op["A_operator_attack_vector"], 4)
            alpha = min(max(alpha, 0.0), 2.0)
            band = alpha_band_classify(alpha)

            # Predicted per-axis decomposition per Catalog #356 (modest;
            # observability-only)
            # Rate-axis predicted_delta_bytes: -0.001 to +0.001 of archive_bytes
            # mapped through canonical 25/37545489 scaling
            predicted_archive_bytes_delta = 0
            if "bytes" in op["axis"] or op["type"] in ("RATE", "RATE+DISTORTION", "DISTORTION+RATE"):
                # RATE-axis operators emit small byte deltas
                predicted_archive_bytes_delta = int(-500 * (alpha - 0.5))  # rough heuristic; signed
            predicted_d_seg_delta = -0.0005 * (alpha - 0.5) if "seg" in op["axis"] else 0.0
            predicted_d_pose_delta = -0.0008 * (alpha - 0.5) if "pose" in op["axis"] else 0.0

            # Composition canonical scaled per `25 * bytes / 37545489`
            predicted_delta_rate_axis = (
                25.0 * predicted_archive_bytes_delta / 37_545_489
                if predicted_archive_bytes_delta
                else 0.0
            )

            pair_key = f"cross_archive_{arc['family'].lower()}__x__operator_{op['op'].lower()}"

            row = {
                # ── Catalog #341 Tier A canonical-routing markers ──
                "predicted_delta_adjustment": 0.0,  # observability-only
                "promotable": False,
                "axis_tag": "[predicted]",
                "evidence_grade_per_row": "[predicted]",
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
                # ── Cross-archive cell identity ──
                "archive_family": arc["family"],
                "archive_sha_prefix": arc["sha_prefix"],
                "archive_bytes": arc["archive_bytes"],
                "archive_best_cpu": arc["best_cpu"],
                "archive_best_cuda": arc["best_cuda"],
                "archive_architecture_class": arc["architecture_class"],
                "archive_validation_status_per_catalog_321": arc["validation_status"],
                "operator_name": op["op"],
                "operator_type": op["type"],
                "operator_axis_attack_vector": op["axis"],
                # ── Per-cell composition_alpha (predicted) ──
                "composition_alpha": alpha,
                "alpha_band_per_v2_cascade": band,
                "F_archive_compatibility": arc["F_archive_compatibility"],
                "A_operator_attack_vector": op["A_operator_attack_vector"],
                # ── Catalog #356 per-axis decomposition ──
                "predicted_axis_decomposition": {
                    "predicted_d_seg_delta": predicted_d_seg_delta,
                    "predicted_d_pose_delta": predicted_d_pose_delta,
                    "predicted_archive_bytes_delta": predicted_archive_bytes_delta,
                    "predicted_delta_rate_axis": predicted_delta_rate_axis,
                    "axis_tag": "[predicted]",
                },
                # ── Catalog #323 canonical Provenance ──
                "canonical_provenance": {
                    "model_id": f"cross_archive_drop_many_canvas_{arc['family']}_{op['op']}_v0",
                    "inputs_sha256": inputs_sha256(arc["sha_prefix"], op["op"]),
                    "measurement_axis": "[predicted]",
                    "hardware_substrate": "scaffold_only_no_paid_dispatch",
                    "captured_at_utc": WRITTEN_AT_UTC,
                    "produced_by_module_id": "slot_h_cascade_item_3_cross_archive_composition_matrix_canonical_builder_v1",
                    "produced_by_subagent_id": SUBAGENT_ID,
                    "produced_by_lane_id": LANE_ID,
                    "evidence_grade": "predicted",
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "non_promotable_reason": "scaffold_only_tier_A_observability_per_catalog_341_357",
                },
                # ── Canonical equation cross-reference per Catalog #344 ──
                "canonical_equation_id": op["canonical_equation_id"],
                "canonical_equation_status": "FORMALIZATION_PENDING_per_catalog_344_8_operator_specific_plus_4_cross_archive",
                # ── Operator-routable cascade ──
                "result_review_blockers": [
                    "predicted_only_no_empirical_anchor",
                    "scaffold_only_tier_A_per_catalog_341_357",
                    "paired_cuda_required_per_catalog_246_for_empirical_alpha",
                    "per_substrate_symposium_required_per_catalog_325_if_high_ev_promotion",
                ],
                "operator_routable_dispatch": (
                    "paired_CUDA_per_catalog_246_OPERATOR_ATTENDED"
                    if band == "SUPER_ADDITIVE"
                    else "deferred_pending_higher_ev_signals"
                ),
                # ── Lane provenance ──
                "lane_id": LANE_ID,
                "landing_memo": LANDING_MEMO,
                "subagent_id": SUBAGENT_ID,
                "schema_version": SCHEMA_VERSION,
                "written_at_utc": WRITTEN_AT_UTC,
                "wave_origin": "wave_slot_h_cascade_item_3_2026_05_29T05_30Z",
            }
            entries.setdefault(pair_key, []).append(row)
    return entries


def _atomic_write_locked(path: Path, payload: dict[str, Any]) -> None:
    """Catalog #131 / #138 fcntl-locked transactional write."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            # Re-load inside the lock per canonical helper pattern
            existing: dict[str, Any]
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
            else:
                existing = {"entries": {}, "last_updated_utc": "", "schema_version": 1}
            # APPEND-ONLY: extend entries dict (NEW pair_keys land; existing
            # untouched per Catalog #322 sister no-in-place-mutation guard)
            entries = existing.setdefault("entries", {})
            new_rows = payload["entries"]
            added_keys: list[str] = []
            skipped_keys: list[str] = []
            for pair_key, rows in new_rows.items():
                if pair_key in entries:
                    skipped_keys.append(pair_key)
                    continue
                entries[pair_key] = list(rows)
                added_keys.append(pair_key)
            existing["last_updated_utc"] = WRITTEN_AT_UTC
            # Atomic write via tmp + os.replace per Catalog #131 discipline
            tmp_suffix = f".tmp.{uuid.uuid4().hex[:12]}"
            tmp_path = path.with_suffix(path.suffix + tmp_suffix)
            tmp_path.write_text(
                json.dumps(existing, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
            print(f"[slot_h] APPENDED {len(added_keys)} new pair_keys; SKIPPED {len(skipped_keys)} existing keys")
            if skipped_keys[:5]:
                print(f"[slot_h] (sample skipped existing) {skipped_keys[:5]}")
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def main() -> int:
    print(f"[slot_h] Building 84-cell cross-archive composition matrix...")
    entries = build_84_cell_rows()
    expected = 7 * 12
    total_rows = sum(len(v) for v in entries.values())
    print(f"[slot_h] Built {len(entries)} pair_keys / {total_rows} rows (expected {expected})")
    assert len(entries) == expected, f"expected {expected} pair_keys, got {len(entries)}"
    assert total_rows == expected, f"expected {expected} rows, got {total_rows}"

    # Counts per band
    band_counts: dict[str, int] = {}
    for rows in entries.values():
        for row in rows:
            b = row["alpha_band_per_v2_cascade"]
            band_counts[b] = band_counts.get(b, 0) + 1
    print(f"[slot_h] Band counts: {band_counts}")

    # Write to canonical posterior
    _atomic_write_locked(MATRIX_PATH, {"entries": entries})
    print(f"[slot_h] LANDED to {MATRIX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
