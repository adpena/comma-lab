#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stream 3 — local Pareto curve from existing macOS-CPU + contest-CPU evals.

Operator directive 2026-05-13 AGGRESSIVE LOCAL HARDWARE SWEEP Stream 3.

Walks experiments/results/**/contest_auth_eval*.json and aggregates score
components (archive_bytes, segnet, posenet, rate, final_score) into a typed
manifest tagged by axis (`[contest-CPU]` / `[contest-CUDA]` / `[macOS-CPU
advisory]`). Emits a Pareto-frontier ranking by (bytes, score) so the operator
sees which substrates are already Pareto-dominated by existing archives.

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192.

NOT a score claim. NOT promotable. ranking_only=True.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _axis_from_path_or_body(path: Path, body: dict) -> str:
    name = path.name.lower()
    parts = str(path).lower()
    if "adjudicated" in name:
        # contest-CPU eval (GHA Linux x86_64) — adjudicated against CUDA
        eg = body.get("evidence_grade") or body.get("axis_label") or ""
        if "cpu" in eg.lower() or "cpu" in name:
            return "[contest-CPU]"
        if "cuda" in eg.lower():
            return "[contest-CUDA]"
        return "[contest-CPU]"  # adjudicated default
    if "macos_cpu" in name or "macos_cpu_advisory" in parts:
        return "[macOS-CPU advisory]"
    if "cpu_eval_gha" in parts:
        return "[contest-CPU]"
    if "modal_cuda" in parts or "cuda" in name:
        return "[contest-CUDA]"
    eg = body.get("evidence_grade", "")
    if eg == "contest-CPU-1to1":
        return "[contest-CPU]"
    if eg == "contest-CUDA":
        return "[contest-CUDA]"
    if eg == "macOS-CPU-advisory":
        return "[macOS-CPU advisory]"
    return "[unknown]"


def _extract_row(path: Path) -> dict | None:
    try:
        body = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    archive_bytes = body.get("archive_size_bytes") or body.get("submission_file_size_bytes")
    if not archive_bytes:
        return None
    score = (
        body.get("canonical_score")
        or body.get("canonical_score_recomputed")
        or body.get("score_recomputed_from_components")
        or body.get("final_score")
    )
    if score is None:
        return None
    return {
        "json_path": str(path.relative_to(REPO_ROOT)),
        "archive_bytes": int(archive_bytes),
        "segnet_dist": body.get("avg_segnet_dist"),
        "posenet_dist": body.get("avg_posenet_dist"),
        "rate_unscaled": body.get("rate_unscaled") or body.get("compression_rate"),
        "canonical_score": float(score),
        "axis": _axis_from_path_or_body(path, body),
        "archive_sha256": (body.get("provenance") or {}).get("archive_sha256")
        or body.get("archive_sha256"),
        "n_samples": body.get("n_samples"),
    }


def main(out_path: str) -> int:
    rows: list[dict] = []
    seen_keys: set[tuple] = set()
    for jpath in sorted((REPO_ROOT / "experiments" / "results").rglob("contest_auth_eval*.json")):
        row = _extract_row(jpath)
        if row is None:
            continue
        key = (row["archive_bytes"], row["axis"], row.get("archive_sha256"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(row)
    # Also include canonical submissions
    for jpath in sorted((REPO_ROOT / "submissions").rglob("contest_auth_eval*.json")):
        row = _extract_row(jpath)
        if row is None:
            continue
        key = (row["archive_bytes"], row["axis"], row.get("archive_sha256"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(row)

    # Pareto frontier per axis: sort by archive_bytes ascending, keep rows whose
    # canonical_score is strictly lower than all prior (smaller-byte) rows on
    # the same axis.
    by_axis: dict[str, list[dict]] = {}
    for r in rows:
        by_axis.setdefault(r["axis"], []).append(r)
    pareto_frontiers: dict[str, list[dict]] = {}
    for axis, axis_rows in by_axis.items():
        axis_rows.sort(key=lambda x: (x["archive_bytes"], x["canonical_score"]))
        frontier = []
        best_score_so_far = float("inf")
        for r in axis_rows:
            if r["canonical_score"] < best_score_so_far:
                frontier.append(r)
                best_score_so_far = r["canonical_score"]
        pareto_frontiers[axis] = frontier

    out = {
        "schema": "local_pareto_curve_existing_evals_v1",
        "lane_id": "lane_local_hardware_aggressive_sweep_20260513",
        "ranking_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "total_rows": len(rows),
        "rows_by_axis_count": {k: len(v) for k, v in by_axis.items()},
        "pareto_frontiers": pareto_frontiers,
        "all_rows": rows,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(f"total rows: {len(rows)}")
    for axis, frontier in pareto_frontiers.items():
        print(f"  {axis}: {len(by_axis[axis])} rows, {len(frontier)} Pareto-frontier points")
        for r in frontier[:5]:
            print(f"    bytes={r['archive_bytes']:>7}  score={r['canonical_score']:.6f}  "
                  f"sha={(r.get('archive_sha256') or 'NA')[:12]}  {r['json_path']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: build_local_pareto_curve_existing_evals.py <out_path.json>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
