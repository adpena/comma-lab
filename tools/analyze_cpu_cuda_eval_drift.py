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
    parser.add_argument("--scorecard-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

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
