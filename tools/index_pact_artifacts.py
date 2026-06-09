#!/usr/bin/env python3
"""Artifact indexer — mine the repo's scattered score/eval/candidate artifacts into ONE
typed currency (operator V6-implementation checklist #2). Walks the durable + results
trees, classifies each score-bearing JSON by authority_tier × metric_family, and emits
``artifact_index.jsonl`` so V3 inherits months of work instead of re-discovering it.

HONEST extraction discipline (NO FAKE): every field is read from the artifact or left
null. authority_tier / metric_family reuse the canonical classifiers from
``tools/ingest_exact_eval_to_candidate.py`` (single source of truth). Nothing is
fabricated; a missing d_seg is null, not invented.

Usage:
  .venv/bin/python tools/index_pact_artifacts.py            # default roots -> artifact_index.jsonl
  .venv/bin/python tools/index_pact_artifacts.py --out <path> --extra-root /Volumes/.../pact
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "src"), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ingest_exact_eval_to_candidate import _authority_tier, _metric_family  # noqa: E402

# Score-bearing artifact globs (NOT every JSON — only eval/score/candidate artifacts).
_PATTERNS = (
    ".omx/research/*.json",
    "experiments/results/**/contest_auth_eval.json",
    "experiments/results/**/*exact_eval*.json",
    "experiments/results/**/*candidate_action_evaluation*.json",
    "experiments/results/**/*campaign_decision*.json",
)

_VEHICLE_KEYWORDS = (
    ("pact_nerv_vq", "pact_nerv_vq"), ("pact_vq", "pact_nerv_vq"),
    ("boost_nerv_pr110", "pr110pp"), ("pr110", "pr110pp"),
    ("snerv", "snerv"), ("hi_nerv", "hinerv"), ("hinerv", "hinerv"), ("hnerv", "hinerv"),
    ("source_brotli_recode", "source_recode"), ("fp11", "source_recode"),
    ("target_region", "atom"), ("sidecar", "atom"), ("margin_field", "atlas"),
    ("gradient_atlas", "atlas"), ("recon_fit", "hinerv"),
)


def _classify_vehicle(path: str, schema: str) -> str:
    hay = (path + " " + schema).lower()
    for kw, veh in _VEHICLE_KEYWORDS:
        if kw in hay:
            return veh
    return "unknown"


def _first_num(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    # one level into common nested holders
    for holder in ("b2", "export", "b2_result"):
        sub = d.get(holder)
        if isinstance(sub, dict):
            for k in keys:
                v = sub.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
            sub2 = sub.get("b2_result")
            if isinstance(sub2, dict):
                for k in keys:
                    v = sub2.get(k)
                    if isinstance(v, (int, float)):
                        return float(v)
    return None


def _index_one(path: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    schema = str(d.get("schema", ""))
    axis = str(d.get("axis_tag") or d.get("lane_tag") or d.get("evidence_grade")
               or d.get("score_axis") or d.get("authority") or "")
    d_seg = _first_num(d, "avg_segnet_dist", "d_seg", "candidate_d_seg")
    d_pose = _first_num(d, "avg_posenet_dist", "d_pose", "candidate_d_pose")
    score = _first_num(d, "final_score", "canonical_score", "first_exact_score_advisory",
                       "candidate_score", "score")
    bytes_ = _first_num(d, "archive_bytes", "archive_size_bytes", "candidate_bytes")
    sha = (d.get("archive_sha256") or (d.get("export", {}) or {}).get("archive_sha256")
           if isinstance(d.get("export"), dict) else d.get("archive_sha256"))
    ran_eval = bool(d.get("pipeline_works") or "exact_eval" in schema or "contest_auth_eval" in schema)
    return {
        "path": str(path),
        "schema": schema,
        "vehicle": _classify_vehicle(str(path), schema),
        "authority_tier": _authority_tier(axis) if axis else "telemetry_proxy",
        "metric_family": _metric_family(
            schema, has_d_seg=d_seg is not None, has_d_pose=d_pose is not None,
            has_bytes=bytes_ is not None, ran_evaluate_py=ran_eval,
        ),
        "archive_sha256": str(sha) if sha else None,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "score": score,
        "bytes": int(bytes_) if bytes_ is not None else None,
        "axis_raw": axis or None,
    }


def build_index(roots: list[Path]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for root in roots:
        for pat in _PATTERNS:
            for p in glob.glob(str(root / pat), recursive=True):
                if p in seen:
                    continue
                seen.add(p)
                row = _index_one(Path(p))
                if row is not None:
                    rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_REPO / "artifact_index.jsonl")
    ap.add_argument("--extra-root", action="append", default=[], type=Path,
                    help="additional roots (e.g. /Volumes/VertigoDataTier/pact)")
    args = ap.parse_args(argv)
    roots = [_REPO, *args.extra_root]
    rows = build_index(roots)
    args.out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    # Summary by authority/metric (so the operator sees the mining yield).
    by_auth: dict[str, int] = {}
    by_metric: dict[str, int] = {}
    with_score = 0
    for r in rows:
        by_auth[r["authority_tier"]] = by_auth.get(r["authority_tier"], 0) + 1
        by_metric[r["metric_family"]] = by_metric.get(r["metric_family"], 0) + 1
        if r["score"] is not None:
            with_score += 1
    print(f"indexed {len(rows)} score-bearing artifacts -> {args.out}")
    print(f"  with a numeric score: {with_score}")
    print(f"  by authority_tier: {dict(sorted(by_auth.items()))}")
    print(f"  by metric_family:  {dict(sorted(by_metric.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
