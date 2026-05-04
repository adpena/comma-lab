#!/usr/bin/env python3
"""Plan full-pipeline self-compression and packing next waves for C089.

This tool is local planning only. It profiles archive bytes, normalizes nearby
local byte-screen artifacts, and emits deterministic recommendations for
contest-faithful build work. It does not dispatch remote jobs, run scorers, or
make promotion claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments import archive_bit_budget_profiler  # noqa: E402


SCHEMA = "full_pipeline_self_compression_nextwave_plan_v1"
TOOL = "experiments/plan_full_pipeline_self_compression_nextwave.py"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
DEFAULT_TARGET_SCORE = 0.314
DEFAULT_FRONTIER_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip"
)
DEFAULT_FRONTIER_EVAL_JSON = DEFAULT_FRONTIER_ARCHIVE.with_name(
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/results/full_pipeline_self_compression_nextwave_worker_20260503"
)

DEFAULT_ARTIFACT_PATHS = {
    "renderer_greenup": REPO_ROOT
    / "experiments/results/c067_renderer_self_compression_greenup_20260503_worker/"
    "trained_renderer_export_unlock_plan.json",
    "renderer_parity_summary": REPO_ROOT
    / "experiments/results/renderer_parity_shrink_search_20260503_worker/summary.json",
    "renderer_parity_dispatch": REPO_ROOT
    / "experiments/results/renderer_parity_shrink_search_20260503_worker/"
    "dispatch_recommendation.json",
    "pr75_action_recommendations": REPO_ROOT
    / "experiments/results/pr75_action_dict_v2_worker_20260503/"
    "candidate_recommendations.json",
    "c088_lossless_repack": REPO_ROOT
    / "experiments/results/c067_mask_topology_sub314_c088_lossless_repack_20260503/"
    "candidate_matrix.json",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _score_math(eval_payload: Mapping[str, Any], target_score: float) -> dict[str, Any]:
    archive_bytes = int(eval_payload["archive_size_bytes"])
    seg_dist = _finite_float(eval_payload["avg_segnet_dist"])
    pose_dist = _finite_float(eval_payload["avg_posenet_dist"])
    score = _finite_float(
        eval_payload.get("score_recomputed_from_components", eval_payload.get("final_score"))
    )
    seg_score = 100.0 * seg_dist
    pose_score = math.sqrt(10.0 * pose_dist)
    distortion_score = seg_score + pose_score
    rate_score = RATE_SCORE_PER_BYTE * archive_bytes
    gap = score - target_score
    bytes_to_target = max(0, int(math.floor(gap / RATE_SCORE_PER_BYTE)) + 1)
    target_archive_bytes_strict = archive_bytes - bytes_to_target
    return {
        "archive_bytes": archive_bytes,
        "bytes_to_save_for_strict_target_at_unchanged_distortion": bytes_to_target,
        "current_score": score,
        "distortion_score_from_components": distortion_score,
        "gap_to_target_score": gap,
        "pose_dist": pose_dist,
        "pose_score_contribution": pose_score,
        "rate_score_contribution": rate_score,
        "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        "score_after_byte_save": score - bytes_to_target * RATE_SCORE_PER_BYTE,
        "seg_dist": seg_dist,
        "seg_score_contribution": seg_score,
        "target_archive_bytes_strict": target_archive_bytes_strict,
        "target_score": target_score,
    }


def _frontier_profile(archive: Path, profile_json: Path | None) -> dict[str, Any]:
    if profile_json is not None:
        return _load_json(profile_json)
    return archive_bit_budget_profiler.build_report([archive])


def _frontier_archive_record(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    archives = profile.get("archives")
    if not isinstance(archives, list) or not archives:
        raise ValueError("profile must contain at least one archive record")
    archive = archives[0]
    if not isinstance(archive, Mapping):
        raise ValueError("profile archive record must be an object")
    return archive


def _payload_segments(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    archive = _frontier_archive_record(profile)
    members = archive.get("members")
    if not isinstance(members, list) or not members:
        return []
    anatomy = members[0].get("fixed_slice_or_payload_anatomy") or {}
    segments = anatomy.get("segments") or []
    return [dict(item) for item in segments if isinstance(item, Mapping)]


def _stream_screen(
    profile: Mapping[str, Any],
    score: Mapping[str, Any],
) -> list[dict[str, Any]]:
    needed = int(score["bytes_to_save_for_strict_target_at_unchanged_distortion"])
    archive = _frontier_archive_record(profile)
    rows: list[dict[str, Any]] = []
    for segment in _payload_segments(profile):
        encoded = int(segment["encoded_bytes"])
        probe = segment.get("compression_probe") or {}
        best_probe = probe.get("best_probe") or {}
        best_probe_bytes = int(best_probe.get("bytes", encoded))
        generic_savings = max(0, encoded - best_probe_bytes)
        max_stream_bytes = encoded - needed
        rows.append(
            {
                "codec": segment.get("codec"),
                "current_encoded_bytes": encoded,
                "current_rate_score": encoded * RATE_SCORE_PER_BYTE,
                "decoded_bytes_estimate": segment.get("decoded_bytes_estimate"),
                "generic_nested_recompression_savings_bytes": generic_savings,
                "generic_nested_recompression_deployable": False,
                "max_stream_bytes_if_this_stream_alone_crosses_target": max_stream_bytes,
                "name": segment["name"],
                "share_of_archive_bytes": encoded
                / float(score["archive_bytes"]),
                "stream_can_close_target_by_byte_count": encoded >= needed,
            }
        )
    nonsegment_bytes = int(score["archive_bytes"]) - sum(
        int(row["current_encoded_bytes"]) for row in rows
    )
    zip_overhead_bytes = int(
        archive.get("zip_member_header_bytes", 0)
    ) + int(archive.get("zip_global_overhead_bytes", 0))
    if zip_overhead_bytes <= 0 or zip_overhead_bytes > nonsegment_bytes:
        zip_overhead_bytes = nonsegment_bytes
    payload_internal_overhead = nonsegment_bytes - zip_overhead_bytes
    if payload_internal_overhead:
        rows.append(
            {
                "codec": "payload_internal_header",
                "current_encoded_bytes": payload_internal_overhead,
                "current_rate_score": payload_internal_overhead * RATE_SCORE_PER_BYTE,
                "decoded_bytes_estimate": None,
                "generic_nested_recompression_savings_bytes": 0,
                "generic_nested_recompression_deployable": False,
                "max_stream_bytes_if_this_stream_alone_crosses_target": (
                    payload_internal_overhead - needed
                ),
                "name": "payload_internal_header",
                "share_of_archive_bytes": payload_internal_overhead
                / float(score["archive_bytes"]),
                "stream_can_close_target_by_byte_count": payload_internal_overhead >= needed,
            }
        )
    rows.append(
        {
            "codec": "zip_payload_header_and_container",
            "current_encoded_bytes": zip_overhead_bytes,
            "current_rate_score": zip_overhead_bytes * RATE_SCORE_PER_BYTE,
            "decoded_bytes_estimate": None,
            "generic_nested_recompression_savings_bytes": 0,
            "generic_nested_recompression_deployable": False,
            "max_stream_bytes_if_this_stream_alone_crosses_target": zip_overhead_bytes
            - needed,
            "name": "archive_overhead",
            "share_of_archive_bytes": zip_overhead_bytes / float(score["archive_bytes"]),
            "stream_can_close_target_by_byte_count": zip_overhead_bytes >= needed,
        }
    )
    return rows


def _score_after_savings(current_score: float, bytes_saved: int) -> float:
    return current_score - bytes_saved * RATE_SCORE_PER_BYTE


def _opportunity(
    *,
    opportunity_id: str,
    family: str,
    rank_hint: int,
    current_score: float,
    bytes_saved: int | None,
    archive_bytes: int | None,
    status: str,
    action: str,
    evidence: str,
    dispatch_recommendation: str,
    source_artifact: Path | None = None,
    candidate_archive: str | None = None,
    not_noop: bool = True,
    crosses_byte_target: bool | None = None,
    component_gain_needed: float | None = None,
    notes: Iterable[str] = (),
) -> dict[str, Any]:
    projected = None if bytes_saved is None else _score_after_savings(current_score, bytes_saved)
    return {
        "action": action,
        "archive_bytes": archive_bytes,
        "bytes_saved_vs_c089": bytes_saved,
        "byte_only_crosses_0_314": (
            crosses_byte_target if crosses_byte_target is not None else (
                None if projected is None else projected < DEFAULT_TARGET_SCORE
            )
        ),
        "candidate_archive": candidate_archive,
        "component_score_gain_needed_after_rate": component_gain_needed,
        "dispatch_recommendation": dispatch_recommendation,
        "evidence": evidence,
        "family": family,
        "not_noop": bool(not_noop),
        "opportunity_id": opportunity_id,
        "projected_score_if_only_bytes_change": projected,
        "promotion_eligible": False,
        "rank_hint": rank_hint,
        "remote_gpu_dispatch_performed": False,
        "score_claim": False,
        "source_artifact": None if source_artifact is None else str(source_artifact),
        "status": status,
        "notes": list(notes),
    }


def _renderer_greenup_opportunity(
    path: Path,
    current_score: float,
    frontier_bytes: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    candidates = payload.get("candidates") or []
    rows = [c for c in candidates if isinstance(c, Mapping) and c.get("archive_bytes")]
    if not rows:
        return None
    best = min(rows, key=lambda row: (int(row["archive_bytes"]), str(row.get("archive_path"))))
    archive_bytes = int(best["archive_bytes"])
    bytes_saved = frontier_bytes - archive_bytes
    return _opportunity(
        opportunity_id="renderer_trained_self_compression_c089_transplant",
        family="renderer_self_compression",
        rank_hint=10,
        current_score=current_score,
        bytes_saved=bytes_saved,
        archive_bytes=archive_bytes,
        status="build_preflight_before_dispatch",
        action=(
            "Rebuild the best trained/self-compressed renderer candidate against "
            "the C089 P6/QP1 slices, then run renderer-transplant pose-safety "
            "and byte-closure preflights."
        ),
        evidence=(
            "Existing trained renderer export scan contains a byte-crossing "
            f"candidate at {archive_bytes} bytes."
        ),
        dispatch_recommendation=(
            "Do not dispatch from this planner. If C089-local transplant "
            "preflight passes, claim the lane before exact CUDA eval."
        ),
        source_artifact=path,
        candidate_archive=str(best.get("archive_path")) if best.get("archive_path") else None,
        not_noop=True,
        notes=(
            "Treat non-local /workspace paths as provenance hints, not current custody.",
            "All decoder/runtime bytes must remain charged inside archive.zip.",
        ),
    )


def _renderer_parity_opportunities(
    summary_path: Path,
    dispatch_path: Path,
    current_score: float,
    frontier_bytes: int,
) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    payload = _load_json(summary_path)
    candidates = [c for c in payload.get("candidates", []) if isinstance(c, Mapping)]
    out: list[dict[str, Any]] = []
    crossing = [
        c for c in candidates
        if c.get("byte_only_crosses_target") is True
        and c.get("pose_safety", {}).get("output_parity", {}).get("aggregate", {}).get("ok")
        is False
    ]
    if crossing:
        best = min(crossing, key=lambda row: (int(row["archive_bytes"]), str(row["candidate_id"])))
        archive_bytes = int(best["archive_bytes"])
        bytes_saved = frontier_bytes - archive_bytes
        parity = best.get("pose_safety", {}).get("output_parity", {}).get("aggregate", {})
        out.append(
            _opportunity(
                opportunity_id="renderer_zero_fp4_recovery_training",
                family="renderer_self_compression",
                rank_hint=20,
                current_score=current_score,
                bytes_saved=bytes_saved,
                archive_bytes=archive_bytes,
                status="do_not_dispatch_current_candidate",
                action=(
                    "Use the byte-crossing zero-FP4 transforms as masks for "
                    "a short recovery/QAT pass; rebuild only after local "
                    "pose-safety passes."
                ),
                evidence=(
                    f"{best['candidate_id']} crosses the byte target locally "
                    "but fails renderer output parity."
                ),
                dispatch_recommendation=(
                    "Do not exact-eval this raw shrink candidate; it is a "
                    "training/repair seed."
                ),
                source_artifact=summary_path,
                candidate_archive=str(best.get("archive")),
                not_noop=True,
                notes=(
                    f"mean_abs_delta={parity.get('mean_abs_delta')}",
                    f"rms_delta={parity.get('rms_delta')}",
                    f"max_abs_delta={parity.get('max_abs_delta')}",
                ),
            )
        )
    if dispatch_path.exists():
        dispatch = _load_json(dispatch_path)
        candidate = dispatch.get("candidate")
        if isinstance(candidate, Mapping):
            archive_bytes = int(candidate["archive_bytes"])
            bytes_saved = frontier_bytes - archive_bytes
            out.append(
                _opportunity(
                    opportunity_id="renderer_pose_safe_micro_shrink",
                    family="renderer_self_compression",
                    rank_hint=50,
                    current_score=current_score,
                    bytes_saved=bytes_saved,
                    archive_bytes=archive_bytes,
                    status="safe_but_too_small",
                    action=(
                        "Keep the largest pose-safe shrink as a regression "
                        "fixture and stack only behind a larger byte move."
                    ),
                    evidence=(
                        "Local pose-safety passed, but the byte-only score "
                        "does not cross 0.314."
                    ),
                    dispatch_recommendation=str(dispatch.get("recommendation")),
                    source_artifact=dispatch_path,
                    candidate_archive=str(candidate.get("archive")),
                    not_noop=True,
                )
            )
    return out


def _pr75_action_opportunity(
    path: Path,
    current_score: float,
    frontier_bytes: int,
    target_score: float,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    rows = payload.get("ranked_candidates") or []
    rows = [row for row in rows if isinstance(row, Mapping)]
    if not rows:
        return None
    best = min(
        rows,
        key=lambda row: (
            _finite_float(row.get("estimated_score_recomputed_from_c067_trace"), 1e9),
            int(row.get("archive_size_bytes", 10**18)),
            str(row.get("name")),
        ),
    )
    archive_bytes = int(best["archive_size_bytes"])
    bytes_saved = frontier_bytes - archive_bytes
    projected = _score_after_savings(current_score, bytes_saved)
    gain_needed = max(0.0, projected - target_score)
    return _opportunity(
        opportunity_id="pr75_p6_action_dictionary_v2_micro_stack",
        family="action_stream_transcoding",
        rank_hint=40,
        current_score=current_score,
        bytes_saved=bytes_saved,
        archive_bytes=archive_bytes,
        status="local_exact_optional_only",
        action=(
            "Use the P6 action candidate only as a cheap component probe or "
            "as a stack member with a larger renderer/mask byte move."
        ),
        evidence=(
            f"Top local action candidate {best.get('name')} is parser-closed "
            "and not a no-op, but the byte move is tiny."
        ),
        dispatch_recommendation=str(best.get("dispatch_recommendation")),
        source_artifact=path,
        candidate_archive=str(best.get("archive")),
        component_gain_needed=gain_needed,
        not_noop=True,
        notes=(
            f"estimated_score_recomputed_from_c067_trace={best.get('estimated_score_recomputed_from_c067_trace')}",
            f"decoded_stream_closure_ok={best.get('decoded_stream_closure_ok')}",
        ),
    )


def _lossless_repack_opportunity(
    path: Path,
    current_score: float,
    frontier_bytes: int,
    target_score: float,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    rows = payload.get("candidates") or []
    rows = [row for row in rows if isinstance(row, Mapping)]
    if not rows:
        return None
    best = min(rows, key=lambda row: (int(row["archive_bytes"]), str(row["candidate_id"])))
    archive_bytes = int(best["archive_bytes"])
    bytes_saved = frontier_bytes - archive_bytes
    projected = _score_after_savings(current_score, bytes_saved)
    gain_needed = max(0.0, projected - target_score)
    return _opportunity(
        opportunity_id="p6_lossless_stream_resweep_pack_base",
        family="lossless_packing",
        rank_hint=60,
        current_score=current_score,
        bytes_saved=bytes_saved,
        archive_bytes=archive_bytes,
        status="stack_packaging_base_only",
        action=(
            "Port the exhaustive stream-resweep to the exact C089 parent only "
            "as packaging base for a larger change; do not spend a standalone "
            "remote exact slot."
        ),
        evidence=(
            "Existing C088 lossless repack is decoded-stream preserving and "
            "single-member ZIP closed."
        ),
        dispatch_recommendation="no_remote_dispatch_standalone",
        source_artifact=path,
        candidate_archive=str(best.get("archive_path")),
        component_gain_needed=gain_needed,
        not_noop=not bool(best.get("noop")),
        notes=(
            "The known exact C082/C088 repack did not supersede C089 by score.",
            "Useful only if stacked with a larger not-noop transform.",
        ),
    )


def _mask_lossless_task(
    current_score: float,
    frontier_bytes: int,
    score: Mapping[str, Any],
) -> dict[str, Any]:
    needed = int(score["bytes_to_save_for_strict_target_at_unchanged_distortion"])
    target_mask_bytes = 219_472 - needed
    return _opportunity(
        opportunity_id="mask_exact_lossless_transcoder_target_217263",
        family="mask_stream_transcoding",
        rank_hint=30,
        current_score=current_score,
        bytes_saved=None,
        archive_bytes=None,
        status="implementation_task_no_candidate_yet",
        action=(
            "Build a decoded-mask-lossless transcoder that proves the PR75 "
            "mask decoded SHA before packaging; target <=217263 charged mask "
            "bytes or combine with renderer savings."
        ),
        evidence=(
            "The mask is 219472 bytes and dominates the archive, but generic "
            "nested recompression found 0 deployable bytes."
        ),
        dispatch_recommendation=(
            "No dispatch until decoded mask SHA parity, archive byte closure, "
            "and non-noop payload proof exist."
        ),
        not_noop=False,
        crosses_byte_target=None,
        notes=(
            f"mask_stream_target_if_alone_crosses={target_mask_bytes}",
            f"frontier_bytes={frontier_bytes}",
        ),
    )


def _rank_opportunities(opportunities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: Mapping[str, Any]) -> tuple[int, int, str]:
        crosses = 0 if item.get("byte_only_crosses_0_314") is True else 1
        return (
            crosses,
            int(item.get("rank_hint", 999)),
            str(item["opportunity_id"]),
        )

    ranked = sorted(opportunities, key=key)
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked


def _artifact_paths(overrides: Mapping[str, Path] | None = None) -> dict[str, Path]:
    paths = dict(DEFAULT_ARTIFACT_PATHS)
    if overrides:
        paths.update(overrides)
    return paths


def build_plan(
    *,
    archive: Path = DEFAULT_FRONTIER_ARCHIVE,
    eval_json: Path = DEFAULT_FRONTIER_EVAL_JSON,
    profile_json: Path | None = None,
    artifact_paths: Mapping[str, Path] | None = None,
    target_score: float = DEFAULT_TARGET_SCORE,
) -> dict[str, Any]:
    archive = archive.resolve()
    eval_json = eval_json.resolve()
    profile_json = None if profile_json is None else profile_json.resolve()
    eval_payload = _load_json(eval_json)
    profile = _frontier_profile(archive, profile_json)
    score = _score_math(eval_payload, target_score)
    current_score = float(score["current_score"])
    frontier_bytes = int(score["archive_bytes"])
    paths = _artifact_paths(artifact_paths)

    opportunities: list[dict[str, Any]] = []
    renderer_greenup = _renderer_greenup_opportunity(
        paths["renderer_greenup"],
        current_score,
        frontier_bytes,
    )
    if renderer_greenup is not None:
        opportunities.append(renderer_greenup)
    opportunities.extend(
        _renderer_parity_opportunities(
            paths["renderer_parity_summary"],
            paths["renderer_parity_dispatch"],
            current_score,
            frontier_bytes,
        )
    )
    opportunities.append(_mask_lossless_task(current_score, frontier_bytes, score))
    pr75 = _pr75_action_opportunity(
        paths["pr75_action_recommendations"],
        current_score,
        frontier_bytes,
        target_score,
    )
    if pr75 is not None:
        opportunities.append(pr75)
    lossless = _lossless_repack_opportunity(
        paths["c088_lossless_repack"],
        current_score,
        frontier_bytes,
        target_score,
    )
    if lossless is not None:
        opportunities.append(lossless)

    archive_record = _frontier_archive_record(profile)
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "evidence_grade": "empirical_planning_only",
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA"
        ),
        "frontier": {
            "archive": str(archive),
            "archive_bytes": frontier_bytes,
            "archive_sha256": _sha256_file(archive),
            "eval_json": str(eval_json),
            "eval_json_sha256": _sha256_file(eval_json),
            "profile_archive_sha256": archive_record.get("sha256"),
            "payload_format": (
                (archive_record.get("members") or [{}])[0]
                .get("fixed_slice_or_payload_anatomy", {})
                .get("payload_format")
            ),
        },
        "score_math": score,
        "stream_screen": _stream_screen(profile, score),
        "opportunities": _rank_opportunities(opportunities),
        "top5_recommendations": [
            {
                "rank": item["rank"],
                "opportunity_id": item["opportunity_id"],
                "action": item["action"],
                "dispatch_recommendation": item["dispatch_recommendation"],
                "bytes_saved_vs_c089": item["bytes_saved_vs_c089"],
                "projected_score_if_only_bytes_change": item[
                    "projected_score_if_only_bytes_change"
                ],
            }
            for item in _rank_opportunities(opportunities)[:5]
        ],
        "planning_constraints": {
            "candidate_archives_written": False,
            "hidden_sidecars_allowed": False,
            "scorers_loaded": False,
            "upstream_scorer_modifications": False,
            "deterministic_json": True,
        },
    }


def write_markdown(path: Path, plan: Mapping[str, Any]) -> None:
    score = plan["score_math"]
    lines = [
        "# Full-Pipeline Self-Compression Nextwave - 2026-05-03",
        "",
        "Evidence grade: empirical planning only. Score claim: false. "
        "Remote dispatch: none.",
        "",
        "## C089 Break-Even",
        "",
        f"- Archive bytes: `{score['archive_bytes']}`",
        f"- Current score: `{score['current_score']}`",
        f"- Target score: `{score['target_score']}`",
        "- Bytes needed at unchanged distortion: "
        f"`{score['bytes_to_save_for_strict_target_at_unchanged_distortion']}`",
        f"- Rate score per byte: `{score['rate_score_per_byte']}`",
        "",
        "## Stream Screen",
        "",
        "| stream | bytes | generic savings | max bytes if solo crossing |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in plan["stream_screen"]:
        lines.append(
            "| {name} | {bytes} | {savings} | {target} |".format(
                name=row["name"],
                bytes=row["current_encoded_bytes"],
                savings=row["generic_nested_recompression_savings_bytes"],
                target=row["max_stream_bytes_if_this_stream_alone_crosses_target"],
            )
        )
    lines.extend(["", "## Top Recommendations", ""])
    for item in plan["top5_recommendations"]:
        lines.append(
            f"{item['rank']}. `{item['opportunity_id']}`: {item['action']} "
            f"Recommendation: {item['dispatch_recommendation']}"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_FRONTIER_ARCHIVE)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_FRONTIER_EVAL_JSON)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_json = args.output_json or output_dir / "nextwave_plan.json"
    output_md = args.output_md or output_dir / "nextwave_plan.md"
    plan = build_plan(
        archive=args.archive,
        eval_json=args.eval_json,
        profile_json=args.profile_json,
        target_score=args.target_score,
    )
    _write_json(output_json, plan)
    write_markdown(output_md, plan)
    print(
        json.dumps(
            {
                "bytes_to_save": plan["score_math"][
                    "bytes_to_save_for_strict_target_at_unchanged_distortion"
                ],
                "output_json": str(output_json.resolve()),
                "output_md": str(output_md.resolve()),
                "promotion_eligible": False,
                "remote_gpu_dispatch_performed": False,
                "score_claim": False,
                "top_opportunity": plan["top5_recommendations"][0]["opportunity_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
