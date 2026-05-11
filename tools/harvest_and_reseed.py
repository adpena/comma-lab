#!/usr/bin/env python3
"""Harvest contest-CUDA results and reseed the meta-Lagrangian calibration anchors.

This is the third leg of the parallel-dispatch loop:
    1. meta_lagrangian_search_cli.py → ranks N candidates
    2. parallel_dispatch_top_k.py     → fires K paid dispatches, harvests JSONL
    3. THIS SCRIPT                    → reseeds .omx/calibration/anchors_*.json

Each harvested DispatchResult with a non-null contest_cuda_score becomes a new
empirical anchor. The next sweep uses the strengthened calibration and predicts
more accurately. Over a few iterations, the proxy converges on the true
rate-distortion frontier and the engine starts ranking correctly.

Per AGENTS.md "Auth eval EVERYWHERE": this script requires each harvested row
to include a readable `score_json_path`. The JSON must validate as an
authoritative contest-CUDA artifact via the canonical continual-learning
custody bridge; harvested JSONL tags are advisory only.

Usage:
    .venv/bin/python tools/harvest_and_reseed.py \\
        --harvested-jsonl reports/sweep_harvested.jsonl \\
        --anchors-path .omx/calibration/anchors_apogee_intN.json \\
        --rel-err-source-meta-glob 'experiments/results/parallel_sweep_*/repack_metadata.json'

After reseeding, re-run the meta_lagrangian_search_cli to get refreshed rankings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

try:
    from tools.tool_bootstrap import ensure_repo_imports
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports

ensure_repo_imports(REPO)

from tac.continual_learning import contest_result_from_auth_eval_payload  # noqa: E402


def _load_harvested(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARN: skipping malformed line {lineno}: {e}", file=sys.stderr)
    return rows


def _load_anchors(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def _read_score_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        result = contest_result_from_auth_eval_payload(
            payload,
            architecture_class="harvest_and_reseed",
            source_path=path,
        )
    except (TypeError, ValueError):
        return None
    verdict = result.validate_custody_verdict()
    if not verdict.accepted or result.axis != "cuda":
        return None
    payload["_harvest_score"] = float(result.score_value)
    payload["_harvest_archive_sha256"] = result.archive_sha256
    payload["_harvest_archive_bytes"] = result.archive_bytes
    payload["_harvest_pose_dist"] = result.cuda_pose
    payload["_harvest_seg_dist"] = result.cuda_seg
    payload["_harvest_hardware_substrate"] = result.hardware_substrate
    return payload


def _find_rel_err_for_label(label: str, meta_glob: str) -> tuple[float | None, int | None, int | None]:
    """Locate the repack_metadata.json that produced this label's archive.
    Returns (rel_err_pct, n_layers, archive_bytes). All None if not found."""
    candidates = list(REPO.glob(meta_glob))
    for meta_path in candidates:
        # Heuristic: match the candidate_id portion of label against the dir name.
        parent_name = meta_path.parent.name
        # Label format from parallel_dispatch_top_k.py: {prefix}_{candidate_id}
        parts = label.split("_")
        for tail_len in range(1, min(4, len(parts)) + 1):
            candidate_id = "_".join(parts[-tail_len:])
            if candidate_id in parent_name:
                meta = json.loads(meta_path.read_text())
                return (
                    meta.get("rel_err_pct_per_weight"),
                    meta.get("n_intn_layers", 13),
                    meta.get("archive_size_bytes"),
                )
    return None, None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harvested-jsonl", type=Path, required=True,
                        help="JSONL produced by parallel_dispatch_top_k.py --harvest-output")
    parser.add_argument("--anchors-path", type=Path, required=True,
                        help="anchors JSON to update (e.g. .omx/calibration/anchors_apogee_intN.json)")
    parser.add_argument("--rel-err-source-meta-glob", default="experiments/results/parallel_sweep_*/repack_metadata.json",
                        help="glob to find each candidate's repack_metadata.json (for rel_err_pct + n_layers)")
    parser.add_argument("--lossless-pose-dist", type=float, default=3.4e-5,
                        help="lossless baseline pose dist (default: PR106). Used as fallback when proxy split unknown.")
    parser.add_argument("--lossless-seg-dist", type=float, default=0.00067819,
                        help="lossless baseline seg dist (default: PR106).")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be added without writing")
    args = parser.parse_args(argv)

    rows = _load_harvested(args.harvested_jsonl)
    if not rows:
        print(f"FATAL: no rows in {args.harvested_jsonl}", file=sys.stderr)
        return 2

    anchors = _load_anchors(args.anchors_path)
    existing_labels = {a.get("lane_id") for a in anchors}

    new_anchors = []
    skipped = 0
    for r in rows:
        detail = _read_score_json(r.get("score_json_path"))
        if detail is None:
            skipped += 1
            print(
                f"  WARN: {r.get('label', '<unknown>')} missing authoritative contest-CUDA score_json_path — skipping",
                file=sys.stderr,
            )
            continue
        label = r["label"]
        if label in existing_labels:
            print(f"  skip (already in anchors): {label}")
            continue
        rel_err, n_layers, archive_bytes = _find_rel_err_for_label(label, args.rel_err_source_meta_glob)
        if rel_err is None:
            print(f"  WARN: could not find rel_err for {label} — skipping", file=sys.stderr)
            continue
        archive_bytes = archive_bytes or r.get("archive_size_bytes") or detail.get("_harvest_archive_bytes")
        if archive_bytes is None:
            print(f"  WARN: no archive_size_bytes for {label} — skipping", file=sys.stderr)
            continue
        row_score = r.get("contest_cuda_score")
        if row_score is not None:
            score_delta = abs(float(row_score) - float(detail["_harvest_score"]))
        else:
            score_delta = 0.0
        if score_delta > 1e-9:
            print(
                f"  WARN: {label} harvested score disagrees with score_json_path by {score_delta:.6g} — skipping",
                file=sys.stderr,
            )
            continue
        pose_source = detail.get("_harvest_pose_dist")
        if pose_source is None:
            pose_source = detail.get("avg_pose_dist")
        if pose_source is None:
            pose_source = detail.get("pose_dist")
        if pose_source is None:
            pose_source = args.lossless_pose_dist
        seg_source = detail.get("_harvest_seg_dist")
        if seg_source is None:
            seg_source = detail.get("avg_seg_dist")
        if seg_source is None:
            seg_source = detail.get("seg_dist")
        if seg_source is None:
            seg_source = args.lossless_seg_dist
        pose_dist = float(pose_source)
        seg_dist = float(seg_source)

        from datetime import UTC, datetime
        new_anchors.append({
            "lane_id": label,
            "rel_err_pct_per_weight": rel_err,
            "archive_bytes": archive_bytes,
            "contest_cuda_score": detail["_harvest_score"],
            "avg_pose_dist": pose_dist,
            "avg_seg_dist": seg_dist,
            "rate_unscaled": archive_bytes / 37545489,
            "measured_utc": r.get("started_utc") or r.get("completed_utc") or "",
            "job_id": r.get("label", "unknown"),
            "archive_sha256": detail["_harvest_archive_sha256"],
            "harvested_from": str(args.harvested_jsonl.resolve()),
            "score_json_path": str(Path(r["score_json_path"]).resolve()),
            "harvested_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "evidence_semantics": "contest_cuda_harvested_score_json_required",
            "hardware_substrate": detail["_harvest_hardware_substrate"],
            "score_claim": False,
        })

    if skipped:
        print(f"WARN: skipped {skipped} rows without authoritative contest-CUDA custody", file=sys.stderr)

    if not new_anchors:
        print("[harvest-reseed] no new anchors to add (all were duplicates or missing rel_err)")
        return 0

    if args.dry_run:
        print(f"[harvest-reseed] DRY-RUN — would add {len(new_anchors)} anchors:")
        for a in new_anchors:
            print(f"  {a['lane_id']}: score={a['contest_cuda_score']:.4f} rel_err={a['rel_err_pct_per_weight']}%")
        return 0

    merged = anchors + new_anchors
    args.anchors_path.write_text(json.dumps(merged, indent=2))
    print(f"[harvest-reseed] added {len(new_anchors)} new [contest-CUDA] anchors → {args.anchors_path}")
    print("[harvest-reseed] re-run tools/meta_lagrangian_search_cli.py to use the strengthened calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
