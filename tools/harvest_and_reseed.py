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

Per CLAUDE.md "every score must be tagged [contest-CUDA] or [advisory only]":
this script ONLY ingests rows tagged `[contest-CUDA]` from the harvested JSONL.
Any row with `tag != "[contest-CUDA]"` is dropped with a warning.

Per AGENTS.md "Auth eval EVERYWHERE": this script requires each harvested row
to include a readable `score_json_path`. The JSON must carry a score-like value
and is recorded as harvested evidence, not as a fresh score claim by this tool.

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

REPO = Path(__file__).resolve().parents[1]


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


def _read_score_json(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    score = payload.get("score_recomputed_from_components")
    score = payload.get("final_score", payload.get("contest_score", payload.get("score", score)))
    try:
        payload["_harvest_score"] = float(score)
    except (TypeError, ValueError):
        return None
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

    contest_rows = [r for r in rows if r.get("tag") == "[contest-CUDA]" and r.get("contest_cuda_score") is not None]
    skipped = len(rows) - len(contest_rows)
    if skipped:
        print(f"WARN: skipping {skipped} non-contest-CUDA rows (failed dispatches or pending scores)", file=sys.stderr)

    if not contest_rows:
        print(f"FATAL: 0 of {len(rows)} rows had [contest-CUDA] tag + parsed score", file=sys.stderr)
        return 3

    anchors = _load_anchors(args.anchors_path)
    existing_labels = {a.get("lane_id") for a in anchors}

    new_anchors = []
    for r in contest_rows:
        label = r["label"]
        if label in existing_labels:
            print(f"  skip (already in anchors): {label}")
            continue
        rel_err, n_layers, archive_bytes = _find_rel_err_for_label(label, args.rel_err_source_meta_glob)
        if rel_err is None:
            print(f"  WARN: could not find rel_err for {label} — skipping", file=sys.stderr)
            continue
        archive_bytes = archive_bytes or r.get("archive_size_bytes")
        if archive_bytes is None:
            print(f"  WARN: no archive_size_bytes for {label} — skipping", file=sys.stderr)
            continue
        detail = _read_score_json(r.get("score_json_path"))
        if detail is None:
            print(
                f"  WARN: {label} missing readable score_json_path with numeric score — skipping",
                file=sys.stderr,
            )
            continue
        score_delta = abs(float(r["contest_cuda_score"]) - float(detail["_harvest_score"]))
        if score_delta > 1e-9:
            print(
                f"  WARN: {label} harvested score disagrees with score_json_path by {score_delta:.6g} — skipping",
                file=sys.stderr,
            )
            continue
        pose_dist = float(detail.get("avg_pose_dist") or detail.get("pose_dist") or args.lossless_pose_dist)
        seg_dist = float(detail.get("avg_seg_dist") or detail.get("seg_dist") or args.lossless_seg_dist)

        from datetime import UTC, datetime
        new_anchors.append({
            "lane_id": label,
            "rel_err_pct_per_weight": rel_err,
            "archive_bytes": archive_bytes,
            "contest_cuda_score": r["contest_cuda_score"],
            "avg_pose_dist": pose_dist,
            "avg_seg_dist": seg_dist,
            "rate_unscaled": archive_bytes / 37545489,
            "measured_utc": r["started_utc"],
            "job_id": r.get("label", "unknown"),
            "archive_sha256": (r.get("archive_sha256") or "")[:16] or "unknown",
            "harvested_from": str(args.harvested_jsonl.resolve()),
            "score_json_path": str(Path(r["score_json_path"]).resolve()),
            "harvested_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "evidence_semantics": "contest_cuda_harvested_score_json_required",
            "score_claim": False,
        })

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
