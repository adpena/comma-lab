#!/usr/bin/env python3
"""Walk all evidence dirs, classify each candidate by score-geometry regime.

For every (d_seg, d_pose, archive_bytes) triple we can extract from
`pre_submission_compliance.json`, `auth_eval_renderer*.json`, or
`contest_eval_result.json` files in `experiments/results/`, run the
dispatch advisor and emit a per-candidate regime tag (pose-dominated /
seg-dominated), the importance-flip distance in log10 decades, and the
recommended next axis to attack.

Output is a single JSON manifest at the supplied path + a stdout summary
showing per-regime counts and the 10 candidates closest to the
importance-flip threshold (most ambiguous).

This surfaces cathedral lanes that are operating "off-leverage" — e.g.,
a seg-targeted lane that's actually in the pose-dominated regime where
seg gains barely move score.

CLAUDE.md compliance: pure CPU + JSON parsing + tac.score_geometry; no
scorer load; no contest score claims (only regime classification).

Usage::

    .venv/bin/python tools/classify_all_evidence_by_regime.py \\
        --evidence-root experiments/results \\
        --output reports/regime_classification.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.score_geometry import (  # noqa: E402
    contest_score,
    importance_flip_threshold,
    operating_regime,
)

EVIDENCE_FILE_PATTERNS = (
    "pre_submission_compliance.json",
    "pre_submission_compliance.contest_final.json",
    "pre_submission_compliance.static.json",
    "auth_eval_renderer.json",
    "auth_eval_renderer_fp4.json",
    "auth_eval_renderer_with_poses.json",
    "auth_eval_result.json",
    "auth_eval_2_29.json",
    "contest_eval_result.json",
    "metrics.json",
)


def _walk_evidence_files(root: Path) -> list[Path]:
    """Yield every JSON evidence file under root that matches a known pattern."""
    out: list[Path] = []
    for pattern in EVIDENCE_FILE_PATTERNS:
        out.extend(root.rglob(pattern))
    return sorted(set(out))


def _try_extract_triple(path: Path) -> dict[str, Any] | None:
    """Best-effort extraction of (d_seg, d_pose, archive_bytes) from any
    evidence-shaped JSON. Returns None if any axis can't be located.

    Tolerates several known variant key names because evidence schemas
    have drifted across cathedral history.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    # d_seg variants
    d_seg = (
        payload.get("seg_distortion")
        or payload.get("seg_avg")
        or payload.get("d_seg")
        or payload.get("avg_segnet_dist")
        or _nested(payload, ("auth_eval", "record", "avg_segnet_dist"))
        or _nested(payload, ("auth_eval", "record", "seg_distortion"))
        or _nested(payload, ("metrics", "seg_distortion"))
        or _nested(payload, ("metrics", "seg_avg"))
        or _nested(payload, ("metrics", "avg_segnet_dist"))
        or _nested(payload, ("auth_eval", "seg_distortion"))
        or _nested(payload, ("decomposition", "seg_distortion"))
        or _nested(payload, ("record", "avg_segnet_dist"))
    )
    # d_pose variants
    d_pose = (
        payload.get("pose_distortion")
        or payload.get("pose_avg")
        or payload.get("d_pose")
        or payload.get("avg_posenet_dist")
        or _nested(payload, ("auth_eval", "record", "avg_posenet_dist"))
        or _nested(payload, ("auth_eval", "record", "pose_distortion"))
        or _nested(payload, ("metrics", "pose_distortion"))
        or _nested(payload, ("metrics", "pose_avg"))
        or _nested(payload, ("metrics", "avg_posenet_dist"))
        or _nested(payload, ("auth_eval", "pose_distortion"))
        or _nested(payload, ("decomposition", "pose_distortion"))
        or _nested(payload, ("record", "avg_posenet_dist"))
    )
    # archive_bytes variants
    archive_bytes = (
        payload.get("archive_bytes")
        or payload.get("archive_size_bytes")
        or _nested(payload, ("archive", "bytes"))
        or _nested(payload, ("archive", "archive_size_bytes"))
        or _nested(payload, ("auth_eval", "record", "archive_bytes"))
        or _nested(payload, ("metrics", "archive_bytes"))
        or _nested(payload, ("record", "archive_bytes"))
    )

    try:
        d_seg_f = float(d_seg) if d_seg is not None else None
        d_pose_f = float(d_pose) if d_pose is not None else None
        archive_bytes_i = int(archive_bytes) if archive_bytes is not None else None
    except (TypeError, ValueError):
        return None

    if d_seg_f is None or d_pose_f is None or archive_bytes_i is None:
        return None
    if d_seg_f < 0 or d_pose_f < 0 or archive_bytes_i <= 0:
        return None

    return {
        "d_seg": d_seg_f,
        "d_pose": d_pose_f,
        "archive_bytes": archive_bytes_i,
    }


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Walk a nested dict path; return None if any key missing."""
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def classify_all(
    *,
    evidence_root: Path,
) -> dict[str, Any]:
    """Walk evidence_root and classify every candidate by regime.

    Returns a manifest with per-candidate rows and aggregate counts.
    """
    files = _walk_evidence_files(evidence_root)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for path in files:
        triple = _try_extract_triple(path)
        if triple is None:
            skipped += 1
            continue
        rel_path = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        regime = operating_regime(triple["d_pose"])
        score = contest_score(
            triple["d_seg"], triple["d_pose"], triple["archive_bytes"]
        )
        rows.append({
            "evidence_path": str(rel_path),
            "lane_dir": str(rel_path.parent.relative_to("experiments/results"))
                if "experiments/results" in str(rel_path) else str(rel_path.parent),
            "d_seg": triple["d_seg"],
            "d_pose": triple["d_pose"],
            "archive_bytes": triple["archive_bytes"],
            "score": score,
            "regime": "pose_dominated" if regime.pose_dominates else "seg_dominated",
            "flip_threshold": regime.flip_threshold,
            "log10_distance_to_flip": regime.crossover_distance_log10,
            "marginal_ratio_seg_over_pose": regime.marginal_ratio_seg_over_pose,
            "recommended_axis": _recommended_axis(regime, triple),
        })

    counts = Counter(r["regime"] for r in rows)
    closest_to_flip = sorted(
        rows,
        key=lambda r: abs(r["log10_distance_to_flip"]),
    )[:10]
    by_score = sorted(rows, key=lambda r: r["score"])[:10]

    return {
        "schema": "regime_classification.v1",
        "evidence_root": str(evidence_root),
        "n_evidence_files_seen": len(files),
        "n_rows_classified": len(rows),
        "n_skipped_missing_axes": skipped,
        "regime_counts": dict(counts),
        "flip_threshold": importance_flip_threshold(),
        "candidates_closest_to_flip_threshold": closest_to_flip,
        "best_10_by_score": by_score,
        "all_rows": rows,
    }


def _recommended_axis(regime: Any, triple: dict[str, Any]) -> str:
    """One-word recommendation: which axis to attack next."""
    if regime.pose_dominates:
        return "pose"
    return "seg"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if not args.evidence_root.is_dir():
        raise SystemExit(f"evidence root not found or not a directory: {args.evidence_root}")

    manifest = classify_all(evidence_root=args.evidence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"Saw {manifest['n_evidence_files_seen']} evidence files; "
        f"classified {manifest['n_rows_classified']} ({manifest['n_skipped_missing_axes']} skipped)."
    )
    print(f"Regime counts: {manifest['regime_counts']}")
    print(f"Flip threshold: d_pose={manifest['flip_threshold']:.2e}")
    if manifest["best_10_by_score"]:
        print("\nBest 10 candidates by score:")
        for row in manifest["best_10_by_score"]:
            print(
                f"  {row['score']:.5f}  {row['regime']:>15s}  "
                f"d_seg={row['d_seg']:.2e}  d_pose={row['d_pose']:.2e}  "
                f"B={row['archive_bytes']:,}  ({row['lane_dir']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
