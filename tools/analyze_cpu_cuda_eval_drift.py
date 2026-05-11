#!/usr/bin/env python3
"""Analyze paired public PR CPU/CUDA auth-eval comment drift.

This is a diagnosis tool, not a score claimant. It consumes the external
GitHub PR-comment scorecard emitted by ``public_pr_eval_comment_scorecard.py``
and measures component-wise CPU/CUDA drift when the same PR has both device
rows.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.auth_eval_records import (
        inflated_output_manifest_summary,
        parse_auth_eval_payload,
        runtime_tree_sha256,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from auth_eval_records import (
        inflated_output_manifest_summary,
        parse_auth_eval_payload,
        runtime_tree_sha256,
    )

CONTEST_N_BYTES = 37_545_489


def score_terms(*, pose: float, seg: float, archive_bytes: int) -> dict[str, float]:
    rate = 25.0 * archive_bytes / CONTEST_N_BYTES
    return {
        "seg_term": 100.0 * seg,
        "pose_term": math.sqrt(10.0 * pose),
        "rate_term": rate,
        "score": 100.0 * seg + math.sqrt(10.0 * pose) + rate,
    }


def _latest_by_device(eval_comments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in eval_comments:
        device = str(row.get("device", "")).lower()
        if device not in {"cpu", "cuda"}:
            continue
        prior = rows.get(device)
        if prior is None or str(row.get("created_at") or "") >= str(prior.get("created_at") or ""):
            rows[device] = row
    return rows


def analyze_pair(pr_row: dict[str, Any]) -> dict[str, Any] | None:
    by_device = _latest_by_device(pr_row.get("eval_comments", []))
    if "cpu" not in by_device or "cuda" not in by_device:
        return None

    cpu = by_device["cpu"]
    cuda = by_device["cuda"]
    cpu_terms = score_terms(
        pose=float(cpu["pose"]),
        seg=float(cpu["seg"]),
        archive_bytes=int(cpu["archive_bytes"]),
    )
    cuda_terms = score_terms(
        pose=float(cuda["pose"]),
        seg=float(cuda["seg"]),
        archive_bytes=int(cuda["archive_bytes"]),
    )
    score_gap = cuda_terms["score"] - cpu_terms["score"]
    seg_gap = cuda_terms["seg_term"] - cpu_terms["seg_term"]
    pose_gap = cuda_terms["pose_term"] - cpu_terms["pose_term"]
    rate_gap = cuda_terms["rate_term"] - cpu_terms["rate_term"]
    return {
        "pr": pr_row.get("pr"),
        "title": pr_row.get("title"),
        "url": pr_row.get("url"),
        "same_archive_bytes": int(cpu["archive_bytes"]) == int(cuda["archive_bytes"]),
        "archive_bytes_cpu": int(cpu["archive_bytes"]),
        "archive_bytes_cuda": int(cuda["archive_bytes"]),
        "cpu": {
            "pose": float(cpu["pose"]),
            "seg": float(cpu["seg"]),
            **cpu_terms,
        },
        "cuda": {
            "pose": float(cuda["pose"]),
            "seg": float(cuda["seg"]),
            **cuda_terms,
        },
        "ratios": {
            "pose_distortion_cuda_over_cpu": float(cuda["pose"]) / float(cpu["pose"]),
            "pose_term_cuda_over_cpu": cuda_terms["pose_term"] / cpu_terms["pose_term"],
            "seg_distortion_cuda_over_cpu": float(cuda["seg"]) / float(cpu["seg"]),
            "seg_term_cuda_over_cpu": cuda_terms["seg_term"] / cpu_terms["seg_term"],
        },
        "gaps_cuda_minus_cpu": {
            "score": score_gap,
            "seg_term": seg_gap,
            "pose_term": pose_gap,
            "rate_term": rate_gap,
            "seg_gap_share": seg_gap / score_gap if score_gap else None,
            "pose_gap_share": pose_gap / score_gap if score_gap else None,
            "rate_gap_share": rate_gap / score_gap if score_gap else None,
        },
    }


def _component_pair_from_auth_eval(
    *,
    cpu_payload: dict[str, Any],
    cuda_payload: dict[str, Any],
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cpu_record = parse_auth_eval_payload(cpu_payload)
    cuda_record = parse_auth_eval_payload(cuda_payload)
    blockers: list[str] = []
    if cpu_record is None:
        blockers.append("cpu_auth_eval_unparseable")
    if cuda_record is None:
        blockers.append("cuda_auth_eval_unparseable")
    if cpu_record is None or cuda_record is None:
        return {
            "source": source or {},
            "valid_for_mechanism_analysis": False,
            "blockers": blockers,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
        }
    if cpu_record.score_axis != "contest_cpu":
        blockers.append(f"cpu_axis_not_contest_cpu:{cpu_record.score_axis}")
    if cuda_record.score_axis != "contest_cuda":
        blockers.append(f"cuda_axis_not_contest_cuda:{cuda_record.score_axis}")
    for axis_name, record in (("cpu", cpu_record), ("cuda", cuda_record)):
        if record.avg_posenet_dist is None:
            blockers.append(f"{axis_name}_pose_missing")
        if record.avg_segnet_dist is None:
            blockers.append(f"{axis_name}_seg_missing")
        if record.archive_bytes is None:
            blockers.append(f"{axis_name}_archive_bytes_missing")
        if record.archive_sha256 is None:
            blockers.append(f"{axis_name}_archive_sha256_missing")
        if record.samples != 600:
            blockers.append(f"{axis_name}_not_full_sample_600")
        if record.hardware_compliance_blocker:
            blockers.append(f"{axis_name}_hardware:{record.hardware_compliance_blocker}")
    cpu_runtime = runtime_tree_sha256(cpu_payload)
    cuda_runtime = runtime_tree_sha256(cuda_payload)
    if cpu_runtime is None:
        blockers.append("cpu_runtime_tree_sha256_missing")
    if cuda_runtime is None:
        blockers.append("cuda_runtime_tree_sha256_missing")
    same_archive_sha256 = (
        cpu_record.archive_sha256 == cuda_record.archive_sha256
        if cpu_record.archive_sha256 and cuda_record.archive_sha256
        else False
    )
    same_archive_bytes = (
        cpu_record.archive_bytes == cuda_record.archive_bytes
        if cpu_record.archive_bytes is not None and cuda_record.archive_bytes is not None
        else False
    )
    same_runtime_tree_sha256 = cpu_runtime == cuda_runtime if cpu_runtime and cuda_runtime else None
    if cpu_record.archive_sha256 and cuda_record.archive_sha256 and not same_archive_sha256:
        blockers.append("cpu_cuda_archive_sha256_mismatch")
    if cpu_record.archive_bytes is not None and cuda_record.archive_bytes is not None and not same_archive_bytes:
        blockers.append("cpu_cuda_archive_bytes_mismatch")
    if same_runtime_tree_sha256 is False:
        blockers.append("cpu_cuda_runtime_tree_sha256_mismatch")

    cpu_terms = score_terms(
        pose=float(cpu_record.avg_posenet_dist or 0.0),
        seg=float(cpu_record.avg_segnet_dist or 0.0),
        archive_bytes=int(cpu_record.archive_bytes or 0),
    )
    cuda_terms = score_terms(
        pose=float(cuda_record.avg_posenet_dist or 0.0),
        seg=float(cuda_record.avg_segnet_dist or 0.0),
        archive_bytes=int(cuda_record.archive_bytes or 0),
    )
    score_gap = cuda_terms["score"] - cpu_terms["score"]
    seg_gap = cuda_terms["seg_term"] - cpu_terms["seg_term"]
    pose_gap = cuda_terms["pose_term"] - cpu_terms["pose_term"]
    rate_gap = cuda_terms["rate_term"] - cpu_terms["rate_term"]

    cpu_raw = inflated_output_manifest_summary(cpu_payload)
    cuda_raw = inflated_output_manifest_summary(cuda_payload)
    same_raw = None
    raw_status = "raw_output_manifest_missing"
    if cpu_raw and cuda_raw:
        same_raw = cpu_raw.get("aggregate_sha256") == cuda_raw.get("aggregate_sha256")
        raw_status = "same_inflated_outputs" if same_raw else "different_inflated_outputs"
    elif cpu_raw or cuda_raw:
        raw_status = "partial_raw_output_manifest"

    if same_raw is True:
        mechanism_class = "same_raw_outputs_scorer_or_loader_drift"
    elif same_raw is False:
        mechanism_class = "different_raw_outputs_runtime_or_inflate_drift"
    elif same_runtime_tree_sha256 is True and same_archive_sha256:
        mechanism_class = "same_archive_runtime_raw_outputs_unmeasured"
    else:
        mechanism_class = "custody_incomplete"

    return {
        "source": source or {},
        "valid_for_mechanism_analysis": not blockers,
        "blockers": sorted(set(blockers)),
        "same_archive_sha256": same_archive_sha256,
        "same_archive_bytes": same_archive_bytes,
        "same_runtime_tree_sha256": same_runtime_tree_sha256,
        "raw_output_pairing_status": raw_status,
        "same_inflated_output_aggregate_sha256": same_raw,
        "mechanism_class": mechanism_class,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "cpu": {
            "score": cpu_record.score,
            "score_axis": cpu_record.score_axis,
            "evidence_grade": cpu_record.evidence_grade,
            "archive_sha256": cpu_record.archive_sha256,
            "archive_bytes": cpu_record.archive_bytes,
            "runtime_tree_sha256": cpu_runtime,
            "inflated_output_manifest": cpu_raw,
            "pose": cpu_record.avg_posenet_dist,
            "seg": cpu_record.avg_segnet_dist,
            **cpu_terms,
        },
        "cuda": {
            "score": cuda_record.score,
            "score_axis": cuda_record.score_axis,
            "evidence_grade": cuda_record.evidence_grade,
            "archive_sha256": cuda_record.archive_sha256,
            "archive_bytes": cuda_record.archive_bytes,
            "runtime_tree_sha256": cuda_runtime,
            "inflated_output_manifest": cuda_raw,
            "pose": cuda_record.avg_posenet_dist,
            "seg": cuda_record.avg_segnet_dist,
            **cuda_terms,
        },
        "gaps_cuda_minus_cpu": {
            "score": score_gap,
            "seg_term": seg_gap,
            "pose_term": pose_gap,
            "rate_term": rate_gap,
            "seg_gap_share": seg_gap / score_gap if score_gap else None,
            "pose_gap_share": pose_gap / score_gap if score_gap else None,
            "rate_gap_share": rate_gap / score_gap if score_gap else None,
        },
        "interpretation": [
            "Negative score gap means CUDA scored lower than CPU for this exact pair; positive means CPU scored lower.",
            "Do not infer global CPU-better or CUDA-better behavior from one row.",
            "If raw outputs differ, runtime/inflate device behavior is part of the mechanism.",
            "If raw outputs match but scores differ, localize through GT loader and scorer-kernel xray probes.",
        ],
    }


def analyze_exact_pair(cpu_json: Path, cuda_json: Path) -> dict[str, Any]:
    cpu_payload = json.loads(cpu_json.read_text(encoding="utf-8"))
    cuda_payload = json.loads(cuda_json.read_text(encoding="utf-8"))
    if not isinstance(cpu_payload, dict) or not isinstance(cuda_payload, dict):
        raise ValueError("auth eval artifacts must be JSON objects")
    return _component_pair_from_auth_eval(
        cpu_payload=cpu_payload,
        cuda_payload=cuda_payload,
        source={"cpu_json": str(cpu_json), "cuda_json": str(cuda_json)},
    )


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def build_analysis(scorecard: dict[str, Any]) -> dict[str, Any]:
    pairs = [pair for row in scorecard.get("rows", []) if (pair := analyze_pair(row))]
    pose_ratios = [float(row["ratios"]["pose_distortion_cuda_over_cpu"]) for row in pairs]
    seg_ratios = [float(row["ratios"]["seg_distortion_cuda_over_cpu"]) for row in pairs]
    cpu_poses = [float(row["cpu"]["pose"]) for row in pairs]
    pose_shares = [float(row["gaps_cuda_minus_cpu"]["pose_gap_share"]) for row in pairs]
    seg_shares = [float(row["gaps_cuda_minus_cpu"]["seg_gap_share"]) for row in pairs]
    median_cpu_pose = _median(cpu_poses)
    return {
        "schema": "cpu_cuda_eval_drift_analysis.v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "input_schema": scorecard.get("schema"),
        "evidence_grade": "external_github_pr_comment_analysis",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "mechanism_claim_proven": False,
        "paired_pr_count": len(pairs),
        "paired_rows": pairs,
        "summary": {
            "mean_pose_distortion_ratio_cuda_over_cpu": _mean(pose_ratios),
            "median_pose_distortion_ratio_cuda_over_cpu": _median(pose_ratios),
            "mean_seg_distortion_ratio_cuda_over_cpu": _mean(seg_ratios),
            "median_seg_distortion_ratio_cuda_over_cpu": _median(seg_ratios),
            "mean_pose_gap_share": _mean(pose_shares),
            "median_pose_gap_share": _median(pose_shares),
            "mean_seg_gap_share": _mean(seg_shares),
            "median_seg_gap_share": _median(seg_shares),
            "median_cpu_pose_distortion": median_cpu_pose,
            "pose_tau_from_public_cpu_comments": math.sqrt(median_cpu_pose) if median_cpu_pose else None,
        },
        "interpretation": [
            "Paired public comments support a device-axis drift signal when CPU and CUDA rows exist for the same PR.",
            "The mechanism is not proven by comments alone; T4-specific TF32 claims are suspect because T4 is not an Ampere TF32 GPU.",
            "Use this as a CPU-leaderboard reproduction hypothesis until paired Linux CPU and CUDA exact eval JSONs exist.",
            "Do not use public-comment drift to promote, rank, kill, or retire internal CUDA lanes.",
        ],
        "safe_next_actions": [
            "Run paired dual-device exact eval plans on Linux x86_64 CPU and T4-equivalent CUDA for PR101, PR102, PR103, and PR105.",
            "Fit CPU-score predictor only from exact paired JSON artifacts, not macOS CPU or rounded comments.",
            "If exact pairs confirm a stable CPU pose floor, test a pose-floor/Huber-style loss as CPU-axis research while preserving CUDA promotion gates.",
        ],
    }


def format_markdown(analysis: dict[str, Any]) -> str:
    if "pair" in analysis:
        pair = analysis["pair"]
        gaps = pair.get("gaps_cuda_minus_cpu") or {}
        lines = [
            "# CPU/CUDA exact-pair mechanism analysis",
            "",
            f"generated_at_utc: `{analysis['created_at_utc']}`",
            f"evidence_grade: `{analysis['evidence_grade']}`",
            "score_claim: `false`",
            "promotion_eligible: `false`",
            "",
            "## Pair",
            "",
            f"- valid_for_mechanism_analysis: `{pair.get('valid_for_mechanism_analysis')}`",
            f"- mechanism_class: `{pair.get('mechanism_class')}`",
            f"- raw_output_pairing_status: `{pair.get('raw_output_pairing_status')}`",
            f"- same_archive_sha256: `{pair.get('same_archive_sha256')}`",
            f"- same_archive_bytes: `{pair.get('same_archive_bytes')}`",
            f"- same_runtime_tree_sha256: `{pair.get('same_runtime_tree_sha256')}`",
            f"- same_inflated_output_aggregate_sha256: `{pair.get('same_inflated_output_aggregate_sha256')}`",
            "",
            "| Axis | Score | Pose | Seg | Archive bytes |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for axis in ("cpu", "cuda"):
            row = pair.get(axis) or {}
            lines.append(
                "| {axis} | {score:.12f} | {pose:.8g} | {seg:.8g} | {archive_bytes} |".format(
                    axis=axis.upper(),
                    score=float(row.get("score") or 0.0),
                    pose=float(row.get("pose") or 0.0),
                    seg=float(row.get("seg") or 0.0),
                    archive_bytes=row.get("archive_bytes"),
                )
            )
        lines.extend(
            [
                "",
                "## CUDA Minus CPU",
                "",
                f"- score: `{gaps.get('score')}`",
                f"- pose_term: `{gaps.get('pose_term')}`",
                f"- seg_term: `{gaps.get('seg_term')}`",
                f"- rate_term: `{gaps.get('rate_term')}`",
                "",
                "## Blockers",
                "",
            ]
        )
        blockers = pair.get("blockers") or []
        if blockers:
            lines.extend(f"- `{blocker}`" for blocker in blockers)
        else:
            lines.append("- none")
        lines.extend(["", "## Interpretation", ""])
        lines.extend(f"- {item}" for item in pair.get("interpretation", []))
        return "\n".join(lines) + "\n"

    lines = [
        "# CPU/CUDA auth-eval drift analysis",
        "",
        f"generated_at_utc: `{analysis['created_at_utc']}`",
        f"evidence_grade: `{analysis['evidence_grade']}`",
        "score_claim: `false`",
        "mechanism_claim_proven: `false`",
        "",
        "| PR | CUDA score | CPU score | pose ratio | seg ratio | pose gap share |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["paired_rows"]:
        lines.append(
            "| {pr} | {cuda:.12f} | {cpu:.12f} | {pose_ratio:.3f} | {seg_ratio:.3f} | {pose_share:.1%} |".format(
                pr=row["pr"],
                cuda=row["cuda"]["score"],
                cpu=row["cpu"]["score"],
                pose_ratio=row["ratios"]["pose_distortion_cuda_over_cpu"],
                seg_ratio=row["ratios"]["seg_distortion_cuda_over_cpu"],
                pose_share=row["gaps_cuda_minus_cpu"]["pose_gap_share"],
            )
        )
    summary = analysis["summary"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- paired PR count: `{analysis['paired_pr_count']}`",
            "- median pose distortion CUDA/CPU ratio: "
            f"`{summary['median_pose_distortion_ratio_cuda_over_cpu']:.3f}`",
            "- median seg distortion CUDA/CPU ratio: "
            f"`{summary['median_seg_distortion_ratio_cuda_over_cpu']:.3f}`",
            "- median CPU pose distortion: "
            f"`{summary['median_cpu_pose_distortion']:.8g}`",
            "- public-comment pose tau hypothesis: "
            f"`{summary['pose_tau_from_public_cpu_comments']:.8g}`",
            "",
            "## Guardrails",
            "",
        ]
    )
    for item in analysis["interpretation"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Safe next actions", ""])
    for item in analysis["safe_next_actions"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scorecard-json", type=Path)
    source.add_argument(
        "--exact-pair",
        type=Path,
        nargs=2,
        metavar=("CPU_JSON", "CUDA_JSON"),
        help="Analyze paired exact contest-CPU and contest-CUDA auth-eval artifacts.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    if args.exact_pair:
        analysis = {
            "schema": "cpu_cuda_exact_pair_mechanism_analysis.v1",
            "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "evidence_grade": "paired_exact_auth_eval_mechanism_diagnostic",
            "pair": analyze_exact_pair(args.exact_pair[0], args.exact_pair[1]),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
        }
    else:
        assert args.scorecard_json is not None
        scorecard = json.loads(args.scorecard_json.read_text(encoding="utf-8"))
        analysis = build_analysis(scorecard)
    text = json.dumps(analysis, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(format_markdown(analysis), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
