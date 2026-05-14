#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
r"""Harvest GHA CPU eval results for the SegNet boundary smoothing sweep.

Usage:
  python tools/harvest_segnet_boundary_smoothing_sweep.py \\
      --rollup experiments/results/segnet_boundary_smoothing_rollup_<ts>.json \\
      --output experiments/results/segnet_boundary_smoothing_results_<ts>.json

Reads:
  - the per-variant rollup JSON
  - per-variant submission_name → finds matching GHA workflow runs on
    `adpena/comma_video_compression_challenge`
  - downloads each completed run's `eval-<submission_name>` artifact
  - parses report.txt → emits per-variant adjudicated row

Writes:
  - aggregate `segnet_boundary_smoothing_results.json` with per-variant
    recomputed score, components, report SHA-256, workflow URL, and lane_tag
    `[contest-CPU GHA Linux x86_64]` only after the downloaded `report.txt`
    has exact `submission_dir` custody for the harvested `submission_name`.

Per CLAUDE.md:
  - tags `[contest-CPU GHA Linux x86_64]` after successful run; `[in_progress]`
    or `[failed]` otherwise
  - per HNeRV-parity discipline lesson 13: variants that don't beat baseline
    are reported as DEFERRED-pending-research, not killed
  - per "Submission auth eval — BOTH CPU AND CUDA": for any sub-0.190 result,
    a paired CUDA dispatch is mandatory before the result can change frontier
    status
  - HIGH 1 fix in dispatcher (codex round-2) landed: submission_name matching
    is now exact-identity (no prefix collisions)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
FORK_REPO = "adpena/comma_video_compression_challenge"
EVAL_WORKFLOW_FILE = "eval.yml"

A1_BASELINE_SCORE = 0.19284757743677347
A1_BASELINE_TAG = "[contest-CPU GHA Linux x86_64]"
PR102_SILVER_SCORE = 0.19538
SUB_MEDAL_BAND_THRESHOLD = 0.190


REPORT_PATTERNS = {
    "avg_posenet_dist": re.compile(r"Average PoseNet Distortion:\s*([0-9.eE+-]+)"),
    "avg_segnet_dist": re.compile(r"Average SegNet Distortion:\s*([0-9.eE+-]+)"),
    "compression_rate": re.compile(r"Compression Rate:\s*([0-9.eE+-]+)"),
    "reported_score": re.compile(r"Final score:.*=\s*([0-9.eE+-]+)"),
    "n_samples": re.compile(r"Evaluation results over (\d+) samples"),
    "submission_file_size": re.compile(r"Submission file size:\s*([0-9,]+)\s*bytes"),
}
CONFIG_LINE_PATTERNS = {
    "device": re.compile(r"^\s*device:\s*(\S+)\s*$", re.MULTILINE),
    "submission_dir": re.compile(r"^\s*submission_dir:\s*(.+?)\s*$", re.MULTILINE),
}


def gh(args: list[str], capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["gh", *args], check=False, capture_output=capture, text=True)


def find_run_for_submission(submission_name: str) -> dict[str, Any] | None:
    """Find the most recent GHA run whose `submission_name` workflow input
    matches. Strategy: scan recent runs (-L 50), download artifact list per
    run, look for `eval-<submission_name>` (exact match)."""
    runs_q = gh([
        "run", "list",
        "-R", FORK_REPO,
        "-w", EVAL_WORKFLOW_FILE,
        "-L", "50",
        "--json", "databaseId,status,conclusion,createdAt,name",
    ])
    if runs_q.returncode != 0:
        return None
    runs = json.loads(runs_q.stdout)
    target = f"eval-{submission_name}"
    for run in runs:
        if run["status"] != "completed":
            continue
        rid = run["databaseId"]
        art_q = gh([
            "api", f"/repos/{FORK_REPO}/actions/runs/{rid}/artifacts",
        ])
        if art_q.returncode != 0:
            continue
        try:
            arts = json.loads(art_q.stdout).get("artifacts", [])
        except Exception:
            continue
        for a in arts:
            if a.get("name") == target:
                return {
                    "run_id": rid,
                    "conclusion": run["conclusion"],
                    "createdAt": run["createdAt"],
                    "artifact_name": target,
                }
    return None


def _submission_name_from_report_dir(value: str) -> str:
    return PurePosixPath(value.strip().replace("\\", "/")).name


def parse_report(
    report_text: str,
    *,
    expected_submission_name: str | None = None,
) -> dict[str, Any]:
    parsed: dict[str, Any] = {"report_text": report_text}
    for k, pat in CONFIG_LINE_PATTERNS.items():
        m = pat.search(report_text)
        if not m:
            return {"_error": f"could not parse config {k}", "report_text": report_text}
        parsed[f"report_{k}"] = m.group(1).strip()
    parsed["report_submission_name"] = _submission_name_from_report_dir(
        parsed["report_submission_dir"]
    )
    if expected_submission_name is not None:
        parsed["expected_submission_name"] = expected_submission_name
        if parsed["report_submission_name"] != expected_submission_name:
            return {
                "_error": (
                    "report submission_name mismatch: "
                    f"expected {expected_submission_name!r}, "
                    f"found {parsed['report_submission_name']!r}"
                ),
                "report_text": report_text,
                "report_submission_dir": parsed["report_submission_dir"],
                "report_submission_name": parsed["report_submission_name"],
                "expected_submission_name": expected_submission_name,
            }
    if parsed["report_device"] != "cpu":
        return {
            "_error": f"unexpected report device {parsed['report_device']!r}",
            "report_text": report_text,
            "report_device": parsed["report_device"],
        }
    for k, pat in REPORT_PATTERNS.items():
        m = pat.search(report_text)
        if not m:
            return {"_error": f"could not parse {k}", "report_text": report_text}
        if k in {"n_samples", "submission_file_size"}:
            parsed[k] = int(m.group(1).replace(",", ""))
        else:
            parsed[k] = float(m.group(1))
    if parsed["n_samples"] != 600:
        return {
            "_error": f"unexpected sample count {parsed['n_samples']!r}",
            "report_text": report_text,
            "n_samples": parsed["n_samples"],
        }
    parsed["score_recomputed"] = (
        100.0 * parsed["avg_segnet_dist"]
        + math.sqrt(10.0 * parsed["avg_posenet_dist"])
        + 25.0 * parsed["compression_rate"]
    )
    return parsed


def select_custodial_report_path(
    artifact_dir: Path, submission_name: str
) -> Path | None:
    matches: list[Path] = []
    for report_path in artifact_dir.rglob("report.txt"):
        parsed = parse_report(
            report_path.read_text(encoding="utf-8", errors="replace"),
            expected_submission_name=submission_name,
        )
        if "_error" not in parsed:
            matches.append(report_path)
    if len(matches) != 1:
        return None
    return matches[0]


def harvest_one(submission_name: str) -> dict[str, Any]:
    found = find_run_for_submission(submission_name)
    if not found:
        return {
            "submission_name": submission_name,
            "status": "not_found",
            "score": None,
            "tag": "[in_progress_or_not_dispatched]",
        }
    rid = found["run_id"]
    if found["conclusion"] != "success":
        return {
            "submission_name": submission_name,
            "status": "failed",
            "run_id": rid,
            "conclusion": found["conclusion"],
            "score": None,
            "tag": f"[GHA_failed:{found['conclusion']}]",
        }
    with tempfile.TemporaryDirectory() as td:
        dl = gh([
            "run", "download",
            str(rid),
            "-R", FORK_REPO,
            "-n", found["artifact_name"],
            "-D", td,
        ])
        if dl.returncode != 0:
            return {
                "submission_name": submission_name,
                "status": "download_failed",
                "run_id": rid,
                "score": None,
                "tag": "[GHA_download_failed]",
            }
        report_path = select_custodial_report_path(Path(td), submission_name)
        if report_path is None:
            return {
                "submission_name": submission_name,
                "status": "report_custody_failed",
                "run_id": rid,
                "score": None,
                "tag": "[report_custody_failed]",
                "artifact_name": found["artifact_name"],
            }
        report_bytes = report_path.read_bytes()
        report_text = report_bytes.decode("utf-8", errors="replace")
        report_sha256 = hashlib.sha256(report_bytes).hexdigest()
        parsed = parse_report(report_text, expected_submission_name=submission_name)
        if "_error" in parsed:
            return {
                "submission_name": submission_name,
                "status": "report_parse_failed",
                "run_id": rid,
                "score": None,
                "tag": "[report_parse_failed]",
                "artifact_name": found["artifact_name"],
                "report_error": parsed["_error"],
            }
    return {
        "submission_name": submission_name,
        "status": "completed",
        "run_id": rid,
        "run_url": f"https://github.com/{FORK_REPO}/actions/runs/{rid}",
        "score": parsed.get("score_recomputed"),
        "score_reported_rounded": parsed.get("reported_score"),
        "avg_posenet_dist": parsed.get("avg_posenet_dist"),
        "avg_segnet_dist": parsed.get("avg_segnet_dist"),
        "compression_rate": parsed.get("compression_rate"),
        "archive_bytes_from_report": parsed.get("submission_file_size"),
        "n_samples": parsed.get("n_samples"),
        "tag": "[contest-CPU GHA Linux x86_64]",
        "hardware": "github-actions-ubuntu-latest-x86_64",
        "evidence_grade": "contest-CPU-1to1",
        "score_claim": True,
        "promotion_eligible": False,
        "promotion_blockers": ["missing_paired_contest_cuda"],
        "exact_report_custody": True,
        "artifact_name": found["artifact_name"],
        "report_sha256": report_sha256,
        "report_submission_dir": parsed.get("report_submission_dir"),
        "report_submission_name": parsed.get("report_submission_name"),
        "expected_submission_name": submission_name,
    }


def make_verdict(score: float, baseline: float) -> str:
    """KEEP/REJECT/DEFER per directive matrix."""
    delta = score - baseline
    if delta <= -0.001:
        return "KEEP"
    if delta >= +0.001:
        return "REJECT"
    return "DEFER"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rollup", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rollup = json.loads(args.rollup.read_text())
    a1_baseline_score = A1_BASELINE_SCORE
    print(
        f"[start] harvesting {rollup['n_variants']} variants; "
        f"A1 baseline = {a1_baseline_score} {A1_BASELINE_TAG}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for v in rollup["variants"]:
        sub = v["submission_name"]
        result = harvest_one(sub)
        result["variant_id"] = v["variant_id"]
        result["smoothing_kind"] = v["smoothing_spec"]["smoothing_kind"]
        result["n_smoothing_lines"] = v["smoothing_spec"]["n_smoothing_lines"]
        result["predicted_delta_band"] = v.get("predicted_delta_band")
        if result.get("score") is not None:
            result["delta_vs_a1_baseline"] = result["score"] - a1_baseline_score
            result["verdict"] = make_verdict(result["score"], a1_baseline_score)
            result["sub_medal_band"] = result["score"] < SUB_MEDAL_BAND_THRESHOLD
            result["beats_pr102_silver"] = result["score"] < PR102_SILVER_SCORE
        else:
            result["delta_vs_a1_baseline"] = None
            result["verdict"] = "PENDING"
            result["sub_medal_band"] = None
            result["beats_pr102_silver"] = None
        rows.append(result)
        print(
            f"  {v['variant_id']:<28} status={result['status']:<12} "
            f"score={result.get('score')} verdict={result.get('verdict')} "
            f"{result.get('tag', '')}",
            flush=True,
        )

    out = {
        "schema_version": "segnet_boundary_smoothing_sweep_results_v1",
        "lane_id": "lane_a1_segnet_boundary_smoothing_inflate",
        "harvest_timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "n_variants": len(rows),
        "n_completed": sum(1 for r in rows if r["status"] == "completed"),
        "a1_canonical_baseline_score": a1_baseline_score,
        "a1_canonical_baseline_tag": A1_BASELINE_TAG,
        "pr102_silver_baseline": PR102_SILVER_SCORE,
        "sub_medal_band_threshold": SUB_MEDAL_BAND_THRESHOLD,
        "verdict_thresholds": {
            "KEEP": "delta <= -0.001",
            "REJECT": "delta >= +0.001",
            "DEFER": "delta in (-0.001, +0.001)",
        },
        "score_promotion_policy": (
            "GHA Linux x86_64 CPU rows are public-axis evidence only; "
            "internal score promotion requires paired exact contest-CUDA custody "
            "on the same archive/runtime packet."
        ),
        "rollup_input": str(args.rollup.relative_to(REPO_ROOT))
        if args.rollup.is_relative_to(REPO_ROOT) else str(args.rollup),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\n[done] results written to {args.output}", flush=True)

    completed = [r for r in rows if r["status"] == "completed"]
    if completed:
        best = min(completed, key=lambda r: r["score"])
        print(
            f"\n[BEST] variant={best['variant_id']} "
            f"score={best['score']:.10f} {best['tag']} "
            f"delta={best.get('delta_vs_a1_baseline'):+.10f} vs A1 baseline "
            f"verdict={best['verdict']}",
            flush=True,
        )
        if best.get("sub_medal_band"):
            print(
                "\n*** OPERATOR-DECISION-NEEDED ***\n"
                f"Variant {best['variant_id']} score {best['score']:.6f} "
                f"is sub-0.190 (medal-band proximity).\n"
                f"This is a PR submission candidate moment per directive §5.\n"
                f"Mandatory before promotion: paired exact contest-CUDA dispatch "
                f"on the same archive/runtime packet.",
                flush=True,
            )
        elif best.get("delta_vs_a1_baseline", 0) < -1e-6:
            print(
                f"\n[finding] variant {best['variant_id']} BEATS A1 baseline by "
                f"{-best['delta_vs_a1_baseline']:.6f}; below sub-medal threshold; "
                f"defer promotion until paired CUDA available.",
                flush=True,
            )
        else:
            print(
                "\n[finding] no variant beats A1 baseline. "
                "Per HNeRV-parity discipline lesson 13: DEFERRED-pending-research, "
                "NOT killed. PR101's bias correction MAY be optimal-or-tied for "
                "A1's substrate; finer smoothing parameter sweep recommended.",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
