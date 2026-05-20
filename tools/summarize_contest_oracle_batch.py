#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarize local contest-oracle batch results into compact custody tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _candidate_dirs(batch_root: Path) -> list[Path]:
    candidates_root = batch_root / "candidates"
    if not candidates_root.is_dir():
        return []
    return sorted(path for path in candidates_root.iterdir() if path.is_dir())


def _load_batch_manifest(batch_root: Path) -> dict[str, Any] | None:
    path = batch_root / "batch_manifest.json"
    return _read_json(path) if path.is_file() else None


def _load_eval(batch_root: Path, candidate_id: str) -> dict[str, Any] | None:
    path = batch_root / "advisory_raw_eval" / candidate_id / "raw_advisory_eval.json"
    return _read_json(path) if path.is_file() else None


def _row(
    *,
    batch_root: Path,
    candidate_dir: Path,
    baseline: dict[str, Any],
    cleanup_by_candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = candidate_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    candidate_id = str(manifest.get("candidate_id") or candidate_dir.name)
    eval_payload = _load_eval(batch_root, candidate_id)
    archive = manifest.get("archive", {}) if isinstance(manifest.get("archive"), dict) else {}
    selection = manifest.get("selection", {}) if isinstance(manifest.get("selection"), dict) else {}
    sidecar = manifest.get("sidecar", {}) if isinstance(manifest.get("sidecar"), dict) else {}
    control = (
        manifest.get("official_inflate_control", {})
        if isinstance(manifest.get("official_inflate_control"), dict)
        else {}
    )
    raw_comparison = (
        control.get("raw_comparison", {}) if isinstance(control.get("raw_comparison"), dict) else {}
    )
    score = _float(eval_payload.get("canonical_score")) if eval_payload else None
    base_score = _float(baseline.get("canonical_score"))
    pose = _float(eval_payload.get("avg_posenet_dist")) if eval_payload else None
    seg = _float(eval_payload.get("avg_segnet_dist")) if eval_payload else None
    rate = _float(eval_payload.get("rate_unscaled")) if eval_payload else None
    base_pose = _float(baseline.get("avg_posenet_dist"))
    base_seg = _float(baseline.get("avg_segnet_dist"))
    base_rate = _float(baseline.get("rate_unscaled"))
    cleanup = cleanup_by_candidate.get(candidate_id)
    return {
        "candidate_id": candidate_id,
        "returncode": eval_payload.get("returncode") if eval_payload else None,
        "axis": eval_payload.get("axis") if eval_payload else None,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "selected_pairs": selection.get("selected_pairs"),
        "selected_frames": selection.get("selected_frames"),
        "pair_count": len(selection.get("selected_pairs") or []),
        "sidecar_bytes": sidecar.get("bytes"),
        "archive_bytes": archive.get("bytes"),
        "archive_delta_bytes": archive.get("delta_bytes_vs_source_archive"),
        "archive_sha256": archive.get("sha256"),
        "raw_sha256": (
            control.get("output_raw", {}).get("sha256")
            if isinstance(control.get("output_raw"), dict)
            else None
        ),
        "raw_retained": (
            (candidate_dir / "inflated" / "0.raw").is_file()
            if (candidate_dir / "inflated").exists()
            else False
        ),
        "cleanup": cleanup,
        "official_inflate_returncode": control.get("returncode"),
        "locality_control_passed": raw_comparison.get("passed"),
        "changed_frames_match_selection": raw_comparison.get("changed_frames_match_selection"),
        "canonical_score": score,
        "delta_score_vs_baseline": score - base_score if score is not None and base_score is not None else None,
        "avg_posenet_dist": pose,
        "delta_posenet_vs_baseline": pose - base_pose if pose is not None and base_pose is not None else None,
        "avg_segnet_dist": seg,
        "delta_segnet_vs_baseline": seg - base_seg if seg is not None and base_seg is not None else None,
        "rate_unscaled": rate,
        "delta_rate_vs_baseline": rate - base_rate if rate is not None and base_rate is not None else None,
        "manifest_path": str(manifest_path),
        "advisory_eval_path": (
            str(batch_root / "advisory_raw_eval" / candidate_id / "raw_advisory_eval.json")
            if eval_payload
            else None
        ),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    batch_root = args.batch_root.resolve()
    baseline = _read_json(args.baseline_eval.resolve())
    batch_manifest = _load_batch_manifest(batch_root)
    cleanup_by_candidate: dict[str, dict[str, Any]] = {}
    if batch_manifest:
        for result in batch_manifest.get("results", []):
            if not isinstance(result, dict):
                continue
            cleanup = result.get("cleanup")
            if isinstance(cleanup, dict):
                cleanup_by_candidate[str(result.get("candidate_id"))] = cleanup
    rows = [
        _row(
            batch_root=batch_root,
            candidate_dir=candidate_dir,
            baseline=baseline,
            cleanup_by_candidate=cleanup_by_candidate,
        )
        for candidate_dir in _candidate_dirs(batch_root)
    ]
    completed = [row for row in rows if row["canonical_score"] is not None]
    completed.sort(key=lambda row: float(row["delta_score_vs_baseline"]))
    payload = {
        "schema": "contest_oracle_batch_summary_v1",
        "batch_root": str(batch_root),
        "baseline_eval": str(args.baseline_eval),
        "batch_manifest": str(batch_root / "batch_manifest.json") if batch_manifest else None,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "baseline": {
            "axis": baseline.get("axis"),
            "canonical_score": baseline.get("canonical_score"),
            "avg_posenet_dist": baseline.get("avg_posenet_dist"),
            "avg_segnet_dist": baseline.get("avg_segnet_dist"),
            "rate_unscaled": baseline.get("rate_unscaled"),
            "archive_bytes": baseline.get("archive", {}).get("bytes")
            if isinstance(baseline.get("archive"), dict)
            else None,
        },
        "candidate_count": len(rows),
        "completed_eval_count": len(completed),
        "raw_retained_count": sum(1 for row in rows if row["raw_retained"]),
        "raw_deleted_count": sum(
            1 for row in rows if isinstance(row.get("cleanup"), dict) and row["cleanup"].get("deleted")
        ),
        "best_completed": completed[0] if completed else None,
        "rows": rows,
        "completed_ranked_by_delta_score": completed,
    }
    return payload


def _fmt(value: Any, digits: int = 12) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Contest Oracle Batch Summary",
        "",
        f"- batch_root: `{payload['batch_root']}`",
        f"- baseline: `{payload['baseline_eval']}`",
        "- axis: `[macOS-CPU advisory]` unless row says otherwise",
        "- score_claim: `false`",
        f"- candidates: `{payload['candidate_count']}`",
        f"- completed_eval_count: `{payload['completed_eval_count']}`",
        f"- raw_retained_count: `{payload['raw_retained_count']}`",
        f"- raw_deleted_count: `{payload['raw_deleted_count']}`",
        "",
        "## Best Completed",
        "",
    ]
    best = payload.get("best_completed")
    if isinstance(best, dict):
        lines.extend(
            [
                f"- candidate: `{best['candidate_id']}`",
                f"- delta_score_vs_baseline: `{_fmt(best['delta_score_vs_baseline'])}`",
                f"- canonical_score: `{_fmt(best['canonical_score'])}`",
                f"- archive_delta_bytes: `{best['archive_delta_bytes']}`",
                "",
            ]
        )
    else:
        lines.append("- none yet")
        lines.append("")
    lines.extend(
        [
            "## Rows",
            "",
            "| candidate | k | delta score | score | delta pose | delta seg | delta rate | bytes delta | locality | raw retained | raw deleted |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in payload["rows"]:
        cleanup = row.get("cleanup") if isinstance(row.get("cleanup"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate_id']}`",
                    _fmt(row["pair_count"], 0),
                    _fmt(row["delta_score_vs_baseline"]),
                    _fmt(row["canonical_score"]),
                    _fmt(row["delta_posenet_vs_baseline"]),
                    _fmt(row["delta_segnet_vs_baseline"]),
                    _fmt(row["delta_rate_vs_baseline"]),
                    _fmt(row["archive_delta_bytes"], 0),
                    _fmt(row["locality_control_passed"]),
                    _fmt(row["raw_retained"]),
                    _fmt(cleanup.get("deleted") if cleanup else False),
                ]
            )
            + " |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = summarize(args)
    _write_json(args.output_json, payload)
    if args.output_md:
        write_markdown(args.output_md, payload)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md) if args.output_md else None,
                "candidate_count": payload["candidate_count"],
                "completed_eval_count": payload["completed_eval_count"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
