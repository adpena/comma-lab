#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Z7 recurrent temporal-coherence vs static-capacity disambiguator.

Z7 is intentionally pre-build gated. This probe gives the future trainer-build
path a concrete arbitration surface without creating dispatch authority. It
compares a Z7 recurrent/GRU exact-eval JSON against a static-capacity control
JSON and only returns a method win when the paired result is same-axis,
same-sample-count, same-archive-byte, and at least 0.005 score better.

The emitted payload is never a score claim; it is a Wave N+1 council input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "z7_temporal_coherence_vs_static_capacity_disambiguator_v1"
PROBE_ID = "z7_temporal_coherence_vs_static_capacity_disambiguator"
SUBSTRATE_ID = "time_traveler_l5_z7_lstm_predictive_coding"
MIN_WIN_DELTA = 0.005
CONTEST_ARCHIVE_NORMALIZER_BYTES = 37_545_489.0
FALSE_AUTHORITY_FLAGS = {
    "research_only": True,
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_paid_dispatch": False,
    "paradigm_claim_allowed": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _score(seg: float, pose: float, archive_bytes: int) -> float:
    return (
        100.0 * seg
        + math.sqrt(10.0 * pose)
        + 25.0 * archive_bytes / CONTEST_ARCHIVE_NORMALIZER_BYTES
    )


def _eval_row(path: Path, *, mode: str) -> dict[str, Any]:
    payload = _load_json(path)
    seg = _first_present(payload, "avg_segnet_dist", "seg_dist", "segnet_dist")
    pose = _first_present(payload, "avg_posenet_dist", "pose_dist", "posenet_dist")
    archive_bytes = _first_present(payload, "archive_size_bytes", "archive_bytes")
    n_samples = _first_present(payload, "n_samples", "sample_count", "num_samples")
    axis = _first_present(payload, "score_axis", "axis", "evidence_axis")
    archive_sha = _first_present(payload, "archive_sha256")
    if archive_sha is None:
        archive_sha = _nested_get(payload, "provenance", "archive_sha256")
    score_claim_valid = payload.get("score_claim_valid")
    reported_score = _first_present(
        payload,
        "score",
        "score_recomputed_from_components",
        "final_score",
    )

    blockers: list[str] = []
    try:
        seg_f = float(seg)
        pose_f = float(pose)
        bytes_i = int(archive_bytes)
    except (TypeError, ValueError):
        seg_f = pose_f = float("nan")
        bytes_i = 0
        blockers.append("component_or_archive_bytes_missing")
    try:
        n_samples_i = int(n_samples)
    except (TypeError, ValueError):
        n_samples_i = None
        blockers.append("n_samples_missing")
    if not axis:
        blockers.append("score_axis_missing")
    if not archive_sha:
        blockers.append("archive_sha256_missing")
    if score_claim_valid is not True:
        blockers.append("score_claim_valid_missing_or_false")

    recomputed = (
        _score(seg_f, pose_f, bytes_i)
        if not math.isnan(seg_f) and not math.isnan(pose_f) and bytes_i > 0
        else None
    )
    try:
        reported_score_f = float(reported_score)
    except (TypeError, ValueError):
        reported_score_f = None
        blockers.append("reported_score_missing")
    if (
        reported_score_f is not None
        and recomputed is not None
        and abs(reported_score_f - recomputed) > 1e-6
    ):
        blockers.append("reported_score_mismatches_recomputed_formula")

    return {
        "mode": mode,
        "path": str(path),
        "json_sha256": _sha256(path),
        "score_axis": axis,
        "n_samples": n_samples_i,
        "archive_bytes": bytes_i if bytes_i > 0 else None,
        "archive_sha256": archive_sha,
        "avg_segnet_dist": seg_f if not math.isnan(seg_f) else None,
        "avg_posenet_dist": pose_f if not math.isnan(pose_f) else None,
        "reported_score": reported_score_f,
        "recomputed_score": recomputed,
        "score_claim_valid_in_source": score_claim_valid,
        "source_blockers": blockers,
    }


def build_plan_payload() -> dict[str, Any]:
    """Return the pre-build Z7 probe contract."""

    return {
        "schema": SCHEMA,
        "probe_id": PROBE_ID,
        "substrate_id": SUBSTRATE_ID,
        "verdict": "pending_paired_exact_eval_json",
        "decision_rule": {
            "preferred_mode": "z7_recurrent_temporal_coherence",
            "control_mode": "static_capacity_control",
            "minimum_score_delta": MIN_WIN_DELTA,
            "same_archive_bytes_required": True,
            "same_axis_required": True,
            "required_axis": "contest_cuda",
        },
        "required_inputs": [
            "z7_recurrent_exact_eval_json",
            "static_capacity_control_exact_eval_json",
        ],
        "required_future_artifacts": [
            "experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py",
            "src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/",
            ".omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_lstm_predictive_coding_modal_t4_dispatch.yaml",
        ],
        "blockers": [
            "no_paired_exact_eval_json",
            "z7_full_main_proxy_export_smoke_not_score_authority",
            "z7_proxy_trained_packet_not_score_aware_or_auth_eval_validated",
            "no_contest_cuda_pair",
            "not_score_authority",
        ],
        **FALSE_AUTHORITY_FLAGS,
    }


def evaluate_exact_eval_pair(
    recurrent_eval_json: Path,
    static_control_eval_json: Path,
) -> dict[str, Any]:
    recurrent = _eval_row(recurrent_eval_json, mode="z7_recurrent_temporal_coherence")
    static = _eval_row(static_control_eval_json, mode="static_capacity_control")

    blockers: list[str] = []
    for row in (recurrent, static):
        blockers.extend(f"{row['mode']}:{blocker}" for blocker in row["source_blockers"])

    if recurrent["score_axis"] != static["score_axis"]:
        blockers.append("paired_score_axis_mismatch")
    if recurrent["score_axis"] != "contest_cuda":
        blockers.append(f"score_axis_not_contest_cuda:{recurrent['score_axis']}")
    if recurrent["n_samples"] != static["n_samples"]:
        blockers.append("paired_n_samples_mismatch")
    if recurrent["archive_bytes"] != static["archive_bytes"]:
        blockers.append("same_archive_bytes_required")
    if recurrent["recomputed_score"] is None or static["recomputed_score"] is None:
        blockers.append("paired_recomputed_score_missing")

    if recurrent["recomputed_score"] is None or static["recomputed_score"] is None:
        recurrent_minus_static = None
        static_minus_recurrent = None
    else:
        recurrent_minus_static = round(
            recurrent["recomputed_score"] - static["recomputed_score"], 12
        )
        static_minus_recurrent = round(-recurrent_minus_static, 12)

    if blockers:
        verdict = "blocked_paired_exact_eval_not_comparable"
        preferred_mode = None
    elif static_minus_recurrent is not None and static_minus_recurrent >= MIN_WIN_DELTA:
        verdict = "z7_recurrent_temporal_coherence_win"
        preferred_mode = "z7_recurrent_temporal_coherence"
    elif recurrent_minus_static is not None and recurrent_minus_static >= MIN_WIN_DELTA:
        verdict = "static_capacity_control_win"
        preferred_mode = "static_capacity_control"
    else:
        verdict = "indeterminate_delta_below_threshold"
        preferred_mode = None

    return {
        "schema": SCHEMA,
        "probe_id": PROBE_ID,
        "substrate_id": SUBSTRATE_ID,
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "verdict": verdict,
        "preferred_mode": preferred_mode,
        "minimum_win_delta": MIN_WIN_DELTA,
        "evidence_grade": "paired_exact_eval_contest_cuda_comparison",
        "comparability": {
            "same_score_axis": recurrent["score_axis"] == static["score_axis"],
            "same_n_samples": recurrent["n_samples"] == static["n_samples"],
            "same_archive_bytes": recurrent["archive_bytes"] == static["archive_bytes"],
        },
        "deltas": {
            "recurrent_minus_static_score": recurrent_minus_static,
            "static_minus_recurrent_score": static_minus_recurrent,
            "recurrent_minus_static_archive_bytes": (
                recurrent["archive_bytes"] - static["archive_bytes"]
                if recurrent["archive_bytes"] is not None
                and static["archive_bytes"] is not None
                else None
            ),
        },
        "source_evals": [recurrent, static],
        "blockers": blockers,
        "result_review": {
            "classification": (
                "wave_n_plus_1_disambiguator_input"
                if not blockers
                else "blocked_exact_eval_pair_not_comparable"
            ),
            "score_authority": "not_authority_until_operator_review_and_lane_claim_custody",
        },
        **FALSE_AUTHORITY_FLAGS,
    }


def write_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recurrent-eval-json", type=Path)
    parser.add_argument("--static-control-eval-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    if args.recurrent_eval_json and args.static_control_eval_json:
        payload = evaluate_exact_eval_pair(
            args.recurrent_eval_json,
            args.static_control_eval_json,
        )
    else:
        payload = build_plan_payload()

    if args.output_json:
        write_payload(payload, args.output_json)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload.get("blockers") else 1


if __name__ == "__main__":
    raise SystemExit(main())
