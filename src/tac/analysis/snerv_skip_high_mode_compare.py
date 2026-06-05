# SPDX-License-Identifier: MIT
"""Compare SNeRV skip-high storage modes against local admission gates."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES
from tac.repo_io import read_json, write_json

SCHEMA = "snerv_skip_high_mode_comparison.v1"
DEFAULT_HARD_BYTE_CEILING = 178_000
RATE_SCORE_PER_BYTE = 25.0 / float(ORIGINAL_VIDEO_BYTES)

UPSTREAM_EVALUATE_GEOMETRY: dict[str, Any] = {
    "schema": "upstream_evaluate_geometry.v1",
    "source": "upstream/evaluate.py + upstream/modules.py + upstream/frame_utils.py",
    "camera_size_hw": [874, 1164],
    "scorer_model_input_hw": [384, 512],
    "seq_len": 2,
    "segnet_domain": "last_frame_only",
    "segnet_frame_index_within_pair": 1,
    "segnet_report_field": "segnet_frame1_argmax_distortion",
    "posenet_domain": "two_frame_pair_yuv6",
    "posenet_report_field": "posenet_two_frame_pose_distortion",
    "posenet_scored_pose_dims": 6,
}

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def build_skip_high_mode_comparison(
    *,
    binary_profiles: Mapping[str, str | Path],
    prefilter_profiles: Mapping[str, str | Path] | None = None,
    hard_byte_ceiling: int = DEFAULT_HARD_BYTE_CEILING,
    baseline_label: str = "scalar_mean",
    candidate_label: str | None = None,
    local_mlx_smoke_command: str | None = None,
) -> dict[str, Any]:
    """Build a false-authority comparison across SNeRV skip-high profiles."""

    rows = [
        _binary_profile_row(label, path, hard_byte_ceiling=hard_byte_ceiling)
        for label, path in sorted(binary_profiles.items())
    ]
    prefilter_rows = [
        _prefilter_profile_row(label, path, hard_byte_ceiling=hard_byte_ceiling)
        for label, path in sorted((prefilter_profiles or {}).items())
    ]
    pairwise = _pairwise_replacement_comparison(
        rows=rows,
        prefilter_rows=prefilter_rows,
        hard_byte_ceiling=hard_byte_ceiling,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
    )
    pairwise_candidates = [
        _pairwise_replacement_comparison(
            rows=rows,
            prefilter_rows=prefilter_rows,
            hard_byte_ceiling=hard_byte_ceiling,
            baseline_label=baseline_label,
            candidate_label=str(row["label"]),
        )
        for row in rows
        if _label_token(str(row.get("label") or ""))
        != _label_token(str(baseline_label))
    ]
    blockers = ["snerv_skip_high_mode_comparison_false_authority"]
    if not rows:
        blockers.append("snerv_skip_high_binary_profiles_missing")
    if not any(row["under_hard_byte_ceiling"] for row in rows):
        blockers.append("no_skip_high_binary_profile_under_hard_byte_ceiling")
    if not any(row["under_hard_byte_ceiling"] and not row["scalar_collapse_risk"] for row in rows):
        blockers.append("no_skip_high_mode_with_both_byte_cap_and_non_scalar_storage")
    if any(row["scorer_input_out_of_distribution"] for row in prefilter_rows):
        blockers.append("skip_high_prefilter_scorer_input_out_of_distribution")
    if any(row["partial_replay"] for row in prefilter_rows):
        blockers.append("skip_high_prefilter_partial_replay_only")
    if any(row["early_stop_uncompetitive"] for row in prefilter_rows):
        blockers.append("skip_high_prefilter_early_stopped_uncompetitive")
    if not any(
        row["under_hard_byte_ceiling"] and not row["skip_high_spatial_collapse_risk"]
        for row in rows
    ):
        blockers.append("no_skip_high_mode_with_byte_cap_and_spatial_storage")
    if not any(
        row["under_hard_byte_ceiling"] and row["local_replay_admissible"]
        for row in prefilter_rows
    ):
        blockers.append("no_skip_high_prefilter_profile_admissible_for_local_replay")
    blockers.extend(str(v) for v in pairwise.get("blockers") or [])

    best_rate = min(rows, key=lambda row: row["archive_bytes"]) if rows else None
    best_non_scalar = min(
        (row for row in rows if not row["scalar_collapse_risk"]),
        key=lambda row: row["archive_bytes"],
        default=None,
    )
    best_spatial = min(
        (row for row in rows if not row["skip_high_spatial_collapse_risk"]),
        key=lambda row: row["archive_bytes"],
        default=None,
    )
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "hard_byte_ceiling": int(hard_byte_ceiling),
        "original_video_bytes": int(ORIGINAL_VIDEO_BYTES),
        "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        "upstream_evaluate_geometry": dict(UPSTREAM_EVALUATE_GEOMETRY),
        "binary_profile_rows": rows,
        "prefilter_profile_rows": prefilter_rows,
        "scalar_to_non_scalar_replacement": pairwise,
        "scalar_to_candidate_replacements": pairwise_candidates,
        "best_rate_row": _row_ref(best_rate),
        "best_non_scalar_skip_high_row": _row_ref(best_non_scalar),
        "best_spatial_skip_high_row": _row_ref(best_spatial),
        "runnable_local_mlx_smoke_command": (
            str(local_mlx_smoke_command).strip()
            if local_mlx_smoke_command
            else None
        ),
        "verdict": (
            "NO_CURRENT_SKIP_HIGH_MODE_READY_FOR_EXACT_EVAL"
            if blockers
            else "LOCAL_SKIP_HIGH_PREFILTER_READY_FOR_CPU_REPLAY"
        ),
        "crux": _crux(rows=rows, prefilter_rows=prefilter_rows),
        "next_actions": _next_actions(rows=rows, prefilter_rows=prefilter_rows),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def write_skip_high_mode_comparison(
    *,
    output_json: str | Path,
    output_md: str | Path | None = None,
    binary_profiles: Mapping[str, str | Path],
    prefilter_profiles: Mapping[str, str | Path] | None = None,
    hard_byte_ceiling: int = DEFAULT_HARD_BYTE_CEILING,
    baseline_label: str = "scalar_mean",
    candidate_label: str | None = None,
    local_mlx_smoke_command: str | None = None,
) -> dict[str, Any]:
    payload = build_skip_high_mode_comparison(
        binary_profiles=binary_profiles,
        prefilter_profiles=prefilter_profiles,
        hard_byte_ceiling=hard_byte_ceiling,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        local_mlx_smoke_command=local_mlx_smoke_command,
    )
    payload["comparison_artifact_path"] = Path(output_json).expanduser().resolve(
        strict=False
    ).as_posix()
    write_json(output_json, payload)
    if output_md is not None:
        Path(output_md).write_text(render_markdown_report(payload), encoding="utf-8")
    return payload


def render_markdown_report(payload: Mapping[str, Any]) -> str:
    rows = payload.get("binary_profile_rows") or []
    prefilter_rows = payload.get("prefilter_profile_rows") or []
    lines = [
        "# SNeRV Skip-High Mode Comparison",
        "",
        f"Schema: `{payload.get('schema')}`",
        f"Verdict: `{payload.get('verdict')}`",
        "Axis: `[macOS-CPU/MLX planning:false-authority]`",
        f"Artifact: `{payload.get('comparison_artifact_path')}`",
        "",
        "## Binary Profiles",
        "",
        "| label | codec | archive bytes | stored shape | stored raw bytes | under cap | scalar collapse | spatial collapse |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {codec} | {archive_bytes} | `{shape}` | {stored_raw_bytes} | {under} | {collapse} | {spatial} |".format(
                label=row["label"],
                codec=row["skip_high_codec"],
                archive_bytes=row["archive_bytes"],
                shape=row["skip_high_stored_shape"],
                stored_raw_bytes=row["skip_high_stored_raw_bytes"],
                under=row["under_hard_byte_ceiling"],
                collapse=row["scalar_collapse_risk"],
                spatial=row["skip_high_spatial_collapse_risk"],
            )
        )
    lines.extend(["", "## Prefilter Profiles", ""])
    if prefilter_rows:
        lines.extend(
            [
                "| label | score | Seg term | Pose term | pairs | local replay | OOD |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in prefilter_rows:
            lines.append(
                "| {label} | {score} | {seg} | {pose} | {pairs} | {replay} | {ood} |".format(
                    label=row["label"],
                    score=_fmt(row["canonical_score"]),
                    seg=_fmt(row["seg_term"]),
                    pose=_fmt(row["pose_term"]),
                    pairs=(
                        f"{row['observed_pair_count']}/{row['required_pairs']}"
                        if row.get("required_pairs")
                        else "n/a"
                    ),
                    replay=row["local_replay_admissible"],
                    ood=row["scorer_input_out_of_distribution"],
                )
            )
    else:
        lines.append("- No scorer prefilter profiles attached.")
    pairwise = _mapping(payload.get("scalar_to_non_scalar_replacement"))
    lines.extend(["", "## Scalar To Non-Scalar Replacement", ""])
    if pairwise:
        byte_pressure = _mapping(pairwise.get("byte_pressure"))
        deltas = _mapping(pairwise.get("scorer_component_deltas"))
        lines.extend(
            [
                f"- baseline: `{pairwise.get('baseline_label')}`",
                f"- candidate: `{pairwise.get('candidate_label')}`",
                f"- byte delta candidate-minus-baseline: `{byte_pressure.get('archive_byte_delta_candidate_minus_baseline')}`",
                f"- rate score delta: `{_fmt(byte_pressure.get('rate_score_delta_candidate_minus_baseline'))}`",
                f"- required non-rate score drop: `{_fmt(byte_pressure.get('required_nonrate_score_drop_to_break_even'))}`",
                f"- SegNet frame-1 delta: `{_fmt(deltas.get('segnet_frame1_argmax_distortion_delta'))}`",
                f"- PoseNet two-frame delta: `{_fmt(deltas.get('posenet_two_frame_pose_distortion_delta'))}`",
                f"- component delta status: `{pairwise.get('component_delta_status')}`",
            ]
        )
    replacements = payload.get("scalar_to_candidate_replacements") or []
    if replacements:
        lines.extend(["", "## Scalar To Candidate Portfolio", ""])
        lines.extend(
            [
                "| candidate | byte delta | rate delta | SegNet frame-1 delta | PoseNet pair delta | status |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for replacement in replacements:
            replacement = _mapping(replacement)
            byte_pressure = _mapping(replacement.get("byte_pressure"))
            deltas = _mapping(replacement.get("scorer_component_deltas"))
            lines.append(
                "| {candidate} | {byte_delta} | {rate_delta} | {seg_delta} | {pose_delta} | {status} |".format(
                    candidate=replacement.get("candidate_label"),
                    byte_delta=byte_pressure.get(
                        "archive_byte_delta_candidate_minus_baseline"
                    ),
                    rate_delta=_fmt(
                        byte_pressure.get("rate_score_delta_candidate_minus_baseline")
                    ),
                    seg_delta=_fmt(
                        deltas.get("segnet_frame1_argmax_distortion_delta")
                    ),
                    pose_delta=_fmt(
                        deltas.get("posenet_two_frame_pose_distortion_delta")
                    ),
                    status=replacement.get("component_delta_status"),
                )
            )
    command = payload.get("runnable_local_mlx_smoke_command")
    if command:
        lines.extend(["", "## Runnable Local MLX Smoke", "", "```bash", str(command), "```"])
    lines.extend(["", "## Crux", ""])
    for item in payload.get("crux") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions", ""])
    for item in payload.get("next_actions") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Blockers", ""])
    for item in payload.get("blockers") or []:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def _binary_profile_row(
    label: str,
    path: str | Path,
    *,
    hard_byte_ceiling: int,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=False)
    payload = read_json(resolved)
    decoder = _mapping(payload.get("decoder_payload_header"))
    skip = _mapping(decoder.get("skip_high_storage"))
    archive_bytes = _int_or_none(payload.get("charged_archive_bytes"))
    if archive_bytes is None:
        package = _mapping(payload.get("package_profile"))
        archive_bytes = _int_or_none(package.get("archive_bytes")) or 0
    stored_shape = [int(v) for v in skip.get("stored_shape") or []]
    source_shape = [int(v) for v in skip.get("source_shape") or []]
    stored_raw = _int_or_none(skip.get("stored_raw_bytes")) or 0
    source_raw = _int_or_none(skip.get("source_raw_bytes")) or 0
    receiver_expands = bool(skip.get("receiver_expands_skip_high"))
    return {
        "label": str(label),
        "path": resolved.as_posix(),
        "sha256": _sha256_file(resolved),
        "schema": payload.get("schema"),
        "archive_bytes": int(archive_bytes),
        "packet_bytes": _int_or_none(
            payload.get("receiver_packet_bytes")
            or payload.get("packet_bytes")
            or payload.get("snar1_packet_bytes")
        ),
        "packet_wire_format": payload.get("receiver_packet_wire_format")
        or payload.get("packet_wire_format"),
        "under_hard_byte_ceiling": int(archive_bytes) <= int(hard_byte_ceiling),
        "bytes_over_hard_ceiling": int(archive_bytes) - int(hard_byte_ceiling),
        "skip_high_codec": skip.get("codec"),
        "skip_high_stored_shape": stored_shape,
        "skip_high_source_shape": source_shape,
        "skip_high_stored_raw_bytes": stored_raw,
        "skip_high_source_raw_bytes": source_raw,
        "skip_high_raw_byte_savings": _int_or_none(skip.get("raw_byte_savings")),
        "receiver_expands_skip_high": receiver_expands,
        "lossless_relative_to_source_skip_high": bool(
            skip.get("lossless_relative_to_source_skip_high")
        ),
        "scalar_collapse_risk": bool(
            receiver_expands and (stored_raw <= 8 or _shape_numel(stored_shape) <= 1)
        ),
        "skip_high_spatial_collapse_risk": _skip_high_spatial_collapse_risk(
            stored_shape=stored_shape,
            source_shape=source_shape,
            receiver_expands=receiver_expands,
        ),
        "decoder_payload_bytes": _int_or_none(
            _mapping(payload.get("section_summary")).get("largest_section_bytes")
        ),
        "largest_section": _mapping(payload.get("section_summary")).get("largest_section"),
        "verdict": payload.get("verdict"),
        "blockers": [str(v) for v in payload.get("blockers") or []],
        **FALSE_AUTHORITY,
    }


def _prefilter_profile_row(
    label: str,
    path: str | Path,
    *,
    hard_byte_ceiling: int,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=False)
    payload = read_json(resolved)
    if payload.get("schema") == "snerv_skip_high_channelmean_early_stop_summary.v1":
        return _early_stop_prefilter_summary_row(
            label,
            resolved,
            payload,
            hard_byte_ceiling=hard_byte_ceiling,
        )
    score = _mapping(payload.get("score_components"))
    diagnosis = _mapping(payload.get("scorer_input_diagnosis"))
    archive_bytes = _int_or_none(payload.get("archive_bytes")) or 0
    scope = _mapping(payload.get("scope_status"))
    batch_pairs = _int_or_none(payload.get("scorer_batch_pairs"))
    blockers = [str(v) for v in payload.get("blockers") or []]
    out_of_distribution = (
        diagnosis.get("candidate_output_out_of_distribution") is True
        or "mlx_renderer_prefilter_scorer_input_out_of_distribution" in blockers
        or any(blocker.startswith("scorer_input_") for blocker in blockers)
    )
    local_replay_admissible = bool(
        scope.get("full_video") == "executed"
        and batch_pairs == 1
        and int(archive_bytes) <= int(hard_byte_ceiling)
        and not out_of_distribution
    )
    return {
        "label": str(label),
        "path": resolved.as_posix(),
        "sha256": _sha256_file(resolved),
        "schema": payload.get("schema"),
        "archive_bytes": int(archive_bytes),
        "under_hard_byte_ceiling": int(archive_bytes) <= int(hard_byte_ceiling),
        "scope_full_video": scope.get("full_video"),
        "scorer_batch_pairs": batch_pairs,
        "observed_pair_count": None,
        "required_pairs": None,
        "partial_replay": False,
        "early_stop_uncompetitive": False,
        "early_stop_decision": None,
        "canonical_score": _float_or_none(score.get("canonical_score")),
        "seg_term": _float_or_none(score.get("seg_term")),
        "pose_term": _float_or_none(score.get("pose_term")),
        "rate_term": _float_or_none(score.get("rate_term")),
        "avg_segnet_dist": _component_distortion(
            payload,
            score,
            raw_key="avg_segnet_dist",
            term_key="seg_term",
            term_to_dist=lambda value: value / 100.0,
        ),
        "avg_posenet_dist": _component_distortion(
            payload,
            score,
            raw_key="avg_posenet_dist",
            term_key="pose_term",
            term_to_dist=lambda value: (value * value) / 10.0,
        ),
        "segnet_frame1_argmax_distortion": _component_distortion(
            payload,
            score,
            raw_key="avg_segnet_dist",
            term_key="seg_term",
            term_to_dist=lambda value: value / 100.0,
        ),
        "posenet_two_frame_pose_distortion": _component_distortion(
            payload,
            score,
            raw_key="avg_posenet_dist",
            term_key="pose_term",
            term_to_dist=lambda value: (value * value) / 10.0,
        ),
        "upstream_evaluate_geometry": dict(UPSTREAM_EVALUATE_GEOMETRY),
        "scorer_input_out_of_distribution": out_of_distribution,
        "scorer_input_verdict": diagnosis.get("verdict"),
        "local_replay_admissible": local_replay_admissible,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _early_stop_prefilter_summary_row(
    label: str,
    path: Path,
    payload: Mapping[str, Any],
    *,
    hard_byte_ceiling: int,
) -> dict[str, Any]:
    archive_bytes = _int_or_none(payload.get("archive_bytes")) or 0
    seg = _float_or_none(payload.get("cumulative_avg_segnet_dist"))
    pose = _float_or_none(payload.get("cumulative_avg_posenet_dist"))
    seg_term = None if seg is None else 100.0 * float(seg)
    pose_term = None if pose is None else math.sqrt(10.0 * float(pose))
    score = _float_or_none(payload.get("cumulative_canonical_score"))
    if score is None and seg_term is not None and pose_term is not None:
        score = float(seg_term) + float(pose_term) + float(archive_bytes) * RATE_SCORE_PER_BYTE
    observed = _int_or_none(payload.get("observed_pair_count"))
    required = _int_or_none(payload.get("required_pairs"))
    decision = str(payload.get("decision") or "")
    early_uncompetitive = (
        "uncompetitive" in decision
        or (seg is not None and float(seg) >= 0.25)
        or (score is not None and float(score) >= 10.0)
    )
    blockers = [
        "mlx_local_replay_not_contest_auth_axis",
        "snerv_skip_high_prefilter_early_stop_summary",
        "snerv_skip_high_prefilter_partial_replay",
    ]
    if early_uncompetitive:
        blockers.append("snerv_skip_high_prefilter_early_stopped_uncompetitive")
    return {
        "label": str(label),
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "schema": payload.get("schema"),
        "archive_bytes": int(archive_bytes),
        "under_hard_byte_ceiling": int(archive_bytes) <= int(hard_byte_ceiling),
        "scope_full_video": "partial_early_stop",
        "scorer_batch_pairs": None,
        "observed_pair_count": observed,
        "required_pairs": required,
        "partial_replay": True,
        "early_stop_uncompetitive": bool(early_uncompetitive),
        "early_stop_decision": decision or None,
        "canonical_score": score,
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": float(archive_bytes) * RATE_SCORE_PER_BYTE,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "segnet_frame1_argmax_distortion": seg,
        "posenet_two_frame_pose_distortion": pose,
        "upstream_evaluate_geometry": dict(UPSTREAM_EVALUATE_GEOMETRY),
        "scorer_input_out_of_distribution": False,
        "scorer_input_verdict": "EARLY_STOP_PROGRESS_ONLY_NO_DISTRIBUTION_GATE",
        "local_replay_admissible": False,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _pairwise_replacement_comparison(
    *,
    rows: list[dict[str, Any]],
    prefilter_rows: list[dict[str, Any]],
    hard_byte_ceiling: int,
    baseline_label: str,
    candidate_label: str | None,
) -> dict[str, Any]:
    baseline = _select_baseline_row(rows, baseline_label)
    candidate = _select_candidate_row(rows, candidate_label)
    blockers: list[str] = []
    if baseline is None:
        blockers.append("skip_high_replacement_baseline_profile_missing")
    if candidate is None:
        blockers.append("skip_high_replacement_non_scalar_candidate_profile_missing")
    if baseline is None or candidate is None:
        return {
            "schema": "snerv_skip_high_scalar_to_non_scalar_replacement.v1",
            "baseline_label": baseline_label,
            "candidate_label": candidate_label,
            "component_delta_status": "missing_binary_profile",
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }

    baseline_prefilter = _find_prefilter_for_mode(prefilter_rows, str(baseline["label"]))
    candidate_prefilter = _find_prefilter_for_mode(prefilter_rows, str(candidate["label"]))
    if baseline_prefilter is None:
        blockers.append("skip_high_replacement_baseline_prefilter_profile_missing")
    if candidate_prefilter is None:
        blockers.append("non_scalar_skip_high_prefilter_profile_missing")
    if bool(candidate.get("skip_high_spatial_collapse_risk")):
        blockers.append("skip_high_replacement_candidate_spatial_collapse")
    if bool(candidate["scalar_collapse_risk"]):
        blockers.append("skip_high_replacement_candidate_is_still_scalar_collapse")
    if not bool(candidate["under_hard_byte_ceiling"]):
        blockers.append("non_scalar_skip_high_candidate_over_hard_byte_ceiling")
    if any(
        bool(row.get("partial_replay"))
        for row in (baseline_prefilter, candidate_prefilter)
        if row is not None
    ):
        blockers.append("skip_high_replacement_component_profile_partial")
    if any(
        bool(row.get("early_stop_uncompetitive"))
        for row in (baseline_prefilter, candidate_prefilter)
        if row is not None
    ):
        blockers.append("skip_high_replacement_component_profile_uncompetitive")

    byte_delta = int(candidate["archive_bytes"]) - int(baseline["archive_bytes"])
    rate_delta = float(byte_delta) * RATE_SCORE_PER_BYTE
    deltas = _scorer_component_deltas(
        baseline_prefilter=baseline_prefilter,
        candidate_prefilter=candidate_prefilter,
    )
    component_delta_status = _component_delta_status(
        baseline_prefilter=baseline_prefilter,
        candidate_prefilter=candidate_prefilter,
    )
    return {
        "schema": "snerv_skip_high_scalar_to_non_scalar_replacement.v1",
        "baseline_label": baseline["label"],
        "candidate_label": candidate["label"],
        "baseline_binary_profile": _row_ref(baseline),
        "candidate_binary_profile": _row_ref(candidate),
        "baseline_prefilter_profile": _row_ref(baseline_prefilter),
        "candidate_prefilter_profile": _row_ref(candidate_prefilter),
        "component_delta_status": component_delta_status,
        "byte_pressure": {
            "schema": "snerv_skip_high_replacement_byte_pressure.v1",
            "hard_byte_ceiling": int(hard_byte_ceiling),
            "original_video_bytes": int(ORIGINAL_VIDEO_BYTES),
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "baseline_archive_bytes": int(baseline["archive_bytes"]),
            "candidate_archive_bytes": int(candidate["archive_bytes"]),
            "archive_byte_delta_candidate_minus_baseline": int(byte_delta),
            "rate_score_delta_candidate_minus_baseline": rate_delta,
            "required_nonrate_score_drop_to_break_even": max(0.0, rate_delta),
            "candidate_bytes_over_hard_ceiling": int(candidate["archive_bytes"])
            - int(hard_byte_ceiling),
            "candidate_under_hard_byte_ceiling": bool(
                candidate["under_hard_byte_ceiling"]
            ),
        },
        "scorer_component_deltas": deltas,
        "upstream_evaluate_geometry": dict(UPSTREAM_EVALUATE_GEOMETRY),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _select_baseline_row(
    rows: list[dict[str, Any]],
    label: str,
) -> dict[str, Any] | None:
    exact = _find_row_by_label(rows, label)
    if exact is not None:
        return exact
    scalar_rows = [row for row in rows if row.get("scalar_collapse_risk")]
    return min(scalar_rows, key=lambda row: row["archive_bytes"], default=None)


def _select_candidate_row(
    rows: list[dict[str, Any]],
    label: str | None,
) -> dict[str, Any] | None:
    if label:
        return _find_row_by_label(rows, label)
    non_scalar = [row for row in rows if not row.get("scalar_collapse_risk")]
    return min(non_scalar, key=lambda row: row["archive_bytes"], default=None)


def _find_row_by_label(
    rows: list[dict[str, Any]],
    label: str,
) -> dict[str, Any] | None:
    wanted = _label_token(label)
    for row in rows:
        if _label_token(str(row.get("label") or "")) == wanted:
            return row
    return None


def _find_prefilter_for_mode(
    rows: list[dict[str, Any]],
    mode_label: str,
) -> dict[str, Any] | None:
    wanted = _label_token(mode_label)
    for row in rows:
        row_label = _label_token(str(row.get("label") or ""))
        if _label_matches_mode(row_label, wanted):
            return row
    return None


def _label_matches_mode(row_label: str, wanted: str) -> bool:
    if row_label == wanted or row_label.startswith(wanted) or wanted in row_label:
        return True
    aliases = {
        "scalarmean": ("scalar",),
        "sharedmean": ("shared",),
        "channelmean": ("channel",),
    }
    return any(row_label.startswith(alias) for alias in aliases.get(wanted, ()))


def _label_token(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _scorer_component_deltas(
    *,
    baseline_prefilter: Mapping[str, Any] | None,
    candidate_prefilter: Mapping[str, Any] | None,
) -> dict[str, Any]:
    geometry = dict(UPSTREAM_EVALUATE_GEOMETRY)
    keys = {
        "segnet_frame1_argmax_distortion": "segnet_frame1_argmax_distortion_delta",
        "seg_term": "segnet_frame1_score_term_delta",
        "posenet_two_frame_pose_distortion": "posenet_two_frame_pose_distortion_delta",
        "pose_term": "posenet_two_frame_score_term_delta",
        "canonical_score": "canonical_score_delta",
    }
    out: dict[str, Any] = {
        "schema": "snerv_skip_high_scorer_component_deltas.v1",
        "upstream_evaluate_geometry": geometry,
        "measured": bool(baseline_prefilter is not None and candidate_prefilter is not None),
    }
    for source_key, out_key in keys.items():
        out[out_key] = _delta_or_none(
            baseline_prefilter,
            candidate_prefilter,
            source_key,
        )
    return out


def _component_delta_status(
    *,
    baseline_prefilter: Mapping[str, Any] | None,
    candidate_prefilter: Mapping[str, Any] | None,
) -> str:
    if baseline_prefilter is None or candidate_prefilter is None:
        return "missing_non_scalar_component_profile"
    if bool(baseline_prefilter.get("partial_replay")) or bool(
        candidate_prefilter.get("partial_replay")
    ):
        return "measured_partial_false_authority"
    return "measured_false_authority"


def _delta_or_none(
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
    key: str,
) -> float | None:
    if baseline is None or candidate is None:
        return None
    base = _float_or_none(baseline.get(key))
    cand = _float_or_none(candidate.get(key))
    if base is None or cand is None:
        return None
    return float(cand - base)


def _component_distortion(
    payload: Mapping[str, Any],
    score: Mapping[str, Any],
    *,
    raw_key: str,
    term_key: str,
    term_to_dist: Any,
) -> float | None:
    direct = _float_or_none(score.get(raw_key))
    if direct is None:
        direct = _float_or_none(payload.get(raw_key))
    if direct is not None:
        return direct
    term = _float_or_none(score.get(term_key))
    if term is None:
        term = _float_or_none(payload.get(term_key))
    if term is None:
        return None
    try:
        out = float(term_to_dist(term))
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _crux(
    *,
    rows: list[dict[str, Any]],
    prefilter_rows: list[dict[str, Any]],
) -> list[str]:
    out: list[str] = []
    scalar_rows = [row for row in rows if row["scalar_collapse_risk"]]
    non_scalar_rows = [row for row in rows if not row["scalar_collapse_risk"]]
    spatial_rows = [row for row in rows if not row["skip_high_spatial_collapse_risk"]]
    if scalar_rows:
        best = min(scalar_rows, key=lambda row: row["archive_bytes"])
        out.append(
            "rate-admissible scalar skip-high is cheap "
            f"({best['archive_bytes']} bytes) but collapses stored skip-high to "
            f"{best['skip_high_stored_raw_bytes']} raw bytes."
        )
    spatial_collapse_rows = [
        row
        for row in rows
        if row["under_hard_byte_ceiling"] and row["skip_high_spatial_collapse_risk"]
    ]
    if spatial_collapse_rows:
        best = min(spatial_collapse_rows, key=lambda row: row["archive_bytes"])
        out.append(
            "byte-cap-friendly channel/scalar skip-high still erases spatial structure; "
            f"best collapsed mode is {best['label']} at {best['archive_bytes']} bytes "
            f"with stored shape {best['skip_high_stored_shape']}."
        )
    if non_scalar_rows:
        best = min(non_scalar_rows, key=lambda row: row["archive_bytes"])
        out.append(
            "non-scalar skip-high preserves more value-domain structure but the "
            f"best attached profile is {best['archive_bytes']} bytes "
            f"({best['bytes_over_hard_ceiling']} vs hard ceiling)."
        )
    if spatial_rows:
        best = min(spatial_rows, key=lambda row: row["archive_bytes"])
        out.append(
            "spatial skip-high preservation currently starts at "
            f"{best['archive_bytes']} bytes; this is the representation-before-coding "
            "target for learned/generated storage."
        )
    if any(row["scorer_input_out_of_distribution"] for row in prefilter_rows):
        out.append(
            "attached scorer prefilter evidence is out of distribution; do not "
            "promote or exact-dispatch from these local scores."
        )
    if any(row["early_stop_uncompetitive"] for row in prefilter_rows):
        out.append(
            "at least one partial MLX scorer replay stopped early as uncompetitive; "
            "treat its deltas as falsification signal, not a full-video score."
        )
    if not out:
        out.append("no attached profiles were sufficient to localize the skip-high crux")
    return out


def _next_actions(
    *,
    rows: list[dict[str, Any]],
    prefilter_rows: list[dict[str, Any]],
) -> list[str]:
    actions = [
        "block Modal/exact auth eval until a byte-closed candidate also passes local scorer-input and cache-quality gates",
        "run the next SNeRV local skip-high smoke on a non-scalar storage mode only after current MLX claims clear",
        "record frame-1 SegNet, two-frame PoseNet, archive bytes, and skip-high storage shape for every mode",
    ]
    if any(row["scalar_collapse_risk"] and row["under_hard_byte_ceiling"] for row in rows):
        actions.append(
            "do not use scalar_mean as the promotion path unless a receiver value-domain xray disproves the collapse mechanism"
        )
    if any(
        row["skip_high_spatial_collapse_risk"] and row["under_hard_byte_ceiling"]
        for row in rows
    ):
        actions.append(
            "do not treat channel_mean as burning down non-scalar skip-high; it is a byte-saving falsification row unless new training repairs SegNet"
        )
    if any(not row["skip_high_spatial_collapse_risk"] for row in rows):
        actions.append(
            "attack the spatial skip-high byte gap with learned/generated shared structure instead of storing the full shared_mean plane verbatim"
        )
    if prefilter_rows and not any(row["local_replay_admissible"] for row in prefilter_rows):
        actions.append(
            "treat current local prefilter rows as acquisition/falsification evidence only"
        )
    return _ordered_unique(actions)


def _row_ref(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "label": row.get("label"),
        "archive_bytes": row.get("archive_bytes"),
        "path": row.get("path"),
        "skip_high_codec": row.get("skip_high_codec"),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _shape_numel(shape: list[int]) -> int:
    out = 1
    if not shape:
        return 0
    for dim in shape:
        out *= int(dim)
    return out


def _skip_high_spatial_collapse_risk(
    *,
    stored_shape: list[int],
    source_shape: list[int],
    receiver_expands: bool,
) -> bool:
    if not receiver_expands or len(stored_shape) < 4:
        return False
    if len(source_shape) < 4:
        return False
    source_h, source_w = int(source_shape[-2]), int(source_shape[-1])
    stored_h, stored_w = int(stored_shape[-2]), int(stored_shape[-1])
    return (source_h > 1 or source_w > 1) and (stored_h <= 1 and stored_w <= 1)


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _fmt(value: Any) -> str:
    parsed = _float_or_none(value)
    return "n/a" if parsed is None else f"{parsed:.6g}"


__all__ = [
    "DEFAULT_HARD_BYTE_CEILING",
    "SCHEMA",
    "build_skip_high_mode_comparison",
    "render_markdown_report",
    "write_skip_high_mode_comparison",
]
