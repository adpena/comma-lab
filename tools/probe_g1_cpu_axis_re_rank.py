#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe G1 CPU-axis reranking against existing exact-eval anchors."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.frontier_scan import (  # noqa: E402
    Anchor,
    build_cpu_axis_optimal_payload,
    collect_all_anchors,
    cpu_axis_family_for_anchor,
    render_frontier_scan_json,
)


def _now_utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _anchor_key(anchor: Anchor) -> str:
    lane_id = anchor.extra.get("lane_id")
    if lane_id:
        return str(lane_id)
    if anchor.archive_sha256:
        return anchor.archive_sha256
    return anchor.source_path


def _anchor_row(anchor: Anchor) -> dict[str, Any]:
    return {
        "score": anchor.score,
        "axis": anchor.canonical_axis(),
        "archive_sha256": anchor.archive_sha256,
        "hardware_substrate": anchor.hardware_substrate,
        "source_path": anchor.source_path,
        "metadata_bucket": cpu_axis_family_for_anchor(anchor),
        "family": cpu_axis_family_for_anchor(anchor),
        "extra": anchor.extra,
    }


def _rank_by_axis(anchors: list[Anchor], axis: str) -> list[dict[str, Any]]:
    rows = [
        _anchor_row(anchor)
        for anchor in anchors
        if anchor.canonical_axis() == axis and anchor.is_qualifying()
    ]
    return sorted(rows, key=lambda row: float(row["score"]))


def _axis_gaps(anchors: list[Anchor]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Anchor]] = defaultdict(dict)
    for anchor in anchors:
        if not anchor.is_qualifying():
            continue
        axis = anchor.canonical_axis()
        if axis not in {"contest_cpu", "contest_cuda"}:
            continue
        key = _anchor_key(anchor)
        current = grouped[key].get(axis)
        if current is None or anchor.score < current.score:
            grouped[key][axis] = anchor

    gaps: dict[str, dict[str, Any]] = {}
    for key, by_axis in sorted(grouped.items()):
        cpu = by_axis.get("contest_cpu")
        cuda = by_axis.get("contest_cuda")
        if cpu is None or cuda is None:
            continue
        gaps[key] = {
            "contest_cpu_score": cpu.score,
            "contest_cuda_score": cuda.score,
            "cpu_minus_cuda": cpu.score - cuda.score,
            "archive_sha256_cpu": cpu.archive_sha256,
            "archive_sha256_cuda": cuda.archive_sha256,
            "hardware_cpu": cpu.hardware_substrate,
            "hardware_cuda": cuda.hardware_substrate,
            "source_cpu": cpu.source_path,
            "source_cuda": cuda.source_path,
        }
    return gaps


def _rank_change_opportunities(
    cpu_rank: list[dict[str, Any]],
    cuda_rank: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not cpu_rank or not cuda_rank:
        return []
    cpu_best = cpu_rank[0]
    cuda_best = cuda_rank[0]
    if cpu_best.get("archive_sha256") == cuda_best.get("archive_sha256"):
        return []
    return [
        {
            "kind": "cpu_best_differs_from_cuda_best",
            "cpu_best": cpu_best,
            "cuda_best": cuda_best,
        }
    ]


def _predict_pr101_lc_v2_cpu(anchors: list[Anchor]) -> float | None:
    matches = []
    for anchor in anchors:
        if anchor.canonical_axis() != "contest_cpu" or not anchor.is_qualifying():
            continue
        haystack = " ".join(
            str(value)
            for value in (
                anchor.extra.get("lane_id"),
                anchor.extra.get("archive_id"),
                anchor.extra.get("label"),
                anchor.source_path,
            )
            if value
        ).lower()
        if "pr101_lc_v2" in haystack:
            matches.append(anchor)
    if not matches:
        return None
    return min(matches, key=lambda anchor: anchor.score).score


def build_probe_payload(
    anchors: list[Anchor],
    *,
    current_frontier_cpu: float | None = None,
) -> dict[str, Any]:
    g1 = build_cpu_axis_optimal_payload(
        anchors,
        current_frontier_cpu=current_frontier_cpu,
    )
    cpu_rank = _rank_by_axis(anchors, "contest_cpu")
    cuda_rank = _rank_by_axis(anchors, "contest_cuda")
    if g1["improvement_found"]:
        verdict = "FRONTIER_MOVES_VIA_RE_RANK"
        predicted_delta_s_band = [
            float(g1["delta_vs_current_frontier"]),
            float(g1["delta_vs_current_frontier"]),
        ]
    elif g1["qualifying_cpu_anchor_count"]:
        verdict = "FRONTIER_STABLE_VIA_RE_RANK"
        predicted_delta_s_band = [0.0, 0.0]
    else:
        verdict = "INSUFFICIENT_DATA"
        predicted_delta_s_band = None
    return {
        "schema": "g1_cpu_axis_re_rank_probe_v1",
        "axis_rank_cpu": cpu_rank,
        "axis_rank_cuda": cuda_rank,
        "axis_gap_per_archive": _axis_gaps(anchors),
        "re_rank_opportunities": _rank_change_opportunities(cpu_rank, cuda_rank),
        "predicted_cpu_score_pr101_lc_v2": _predict_pr101_lc_v2_cpu(anchors),
        "verdict": verdict,
        "predicted_delta_s_band": predicted_delta_s_band,
        "g1_cpu_axis_optimization": g1,
        "score_claim": False,
        "score_claim_valid": False,
        "score_claim_kind": "existing_anchor_rerank_no_new_score_claim",
        "notes": [
            "Existing exact-eval anchors are reranked on contest-CPU only.",
            "No new archive score is claimed by this probe.",
        ],
    }


def _write_report(payload: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.json"
    path.write_text(render_frontier_scan_json(payload) + "\n", encoding="utf-8")
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--current-frontier-cpu",
        type=float,
        default=None,
        help="Optional CPU frontier override. Defaults to canonical best CPU anchor.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout.")
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write experiments/results/g1_cpu_axis_re_rank_<utc>/report.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory override. Implies --write-report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    payload = build_probe_payload(
        collect_all_anchors(repo_root),
        current_frontier_cpu=args.current_frontier_cpu,
    )
    if args.output_dir is not None:
        args.write_report = True
    if args.write_report:
        output_dir = args.output_dir or (
            repo_root / "experiments/results" / f"g1_cpu_axis_re_rank_{_now_utc()}"
        )
        report_path = _write_report(payload, output_dir.resolve())
        payload = {**payload, "report_path": report_path.relative_to(repo_root).as_posix()}
    if args.json or not args.write_report:
        print(render_frontier_scan_json(payload))
    else:
        print(payload["report_path"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
