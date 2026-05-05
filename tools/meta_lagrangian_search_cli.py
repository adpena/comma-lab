#!/usr/bin/env python3
"""Meta-Lagrangian search CLI — extreme automated Shannon-floor candidate raising.

Walks a candidate spec (JSON list, or auto-generated from the apogee bit-width
sweep) through the full pipeline:

    distortion_proxy → score_band predictor → Lagrangian → predispatch_sanity

Outputs a ranked queue of dispatch-eligible candidates plus the fully evaluated
non-eligible ones for forensic inspection. Per CLAUDE.md no-arbitrariness rule,
every numeric threshold the CLI uses carries a ``[heuristic:...]`` provenance
tag in :mod:`tac.optimizer.meta_lagrangian`.

CONTRACT: this CLI never spends GPU. It produces a ranked queue; the operator
or a downstream automation runs paid eval on the top-k via the existing
``tools/lightning_dispatch_pr106_stack.py`` or ``scripts/launch_lane_on_vastai.py``.

Usage:
    .venv/bin/python tools/meta_lagrangian_search_cli.py \\
        --lane-class apogee_intN \\
        --auto-sweep-bits 4,5,6,7,8 \\
        --top-k 3 \\
        --output reports/meta_lagrangian_apogee_search.json

Or with custom candidates:
    .venv/bin/python tools/meta_lagrangian_search_cli.py \\
        --lane-class apogee_intN \\
        --candidates-json /path/to/candidates.json \\
        --top-k 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tac.optimizer.meta_lagrangian import (  # noqa: E402
    LagrangianConstraints,
    MetaLagrangianSearch,
)
from tac.predictor.score_band import load_calibration_anchors  # noqa: E402

from experiments.distortion_proxy_local import make_distortion_proxy  # type: ignore  # noqa: E402


def _auto_sweep_apogee(bits: list[int]) -> list[dict]:
    """Auto-generate candidates from `experiments/results/apogee_int<N>_repack_*` archives."""
    candidates = []
    for n in bits:
        repack_dirs = sorted(REPO_ROOT.glob(f"experiments/results/apogee_int{n}_repack_*"))
        if not repack_dirs:
            continue
        # Pick the most recent
        repack_dir = repack_dirs[-1]
        meta_path = repack_dir / "repack_metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        archive_bytes = meta.get("archive_size_bytes")
        rel_err_pct = meta.get("rel_err_pct_per_weight")
        n_layers = meta.get("n_intn_layers", 13)
        archive_path = repack_dir / f"apogee_int{n}_archive.zip"
        if archive_bytes is None or rel_err_pct is None:
            continue
        candidates.append({
            "candidate_id": f"apogee_int{n}",
            "archive_bytes": archive_bytes,
            "rel_err_pct": rel_err_pct,
            "n_layers": n_layers,
            "lane_class": "apogee_intN",
            "archive_path": archive_path if archive_path.is_file() else None,
        })
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-class", required=True,
                        help="determines anchors file: anchors_<class>.json")
    parser.add_argument("--anchors-path", type=Path, default=None)
    parser.add_argument("--candidates-json", type=Path, default=None,
                        help="JSON file with a list of candidate dicts")
    parser.add_argument("--auto-sweep-bits", type=str, default=None,
                        help="comma-separated list of bit-widths to auto-walk apogee_int<N> archives")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rate-max", type=float, default=None,
                        help="constraint: rate_unscaled upper bound (default: anchor-derived)")
    parser.add_argument("--pose-max", type=float, default=None,
                        help="constraint: avg_pose_dist upper bound")
    parser.add_argument("--seg-max", type=float, default=None,
                        help="constraint: avg_seg_dist upper bound")
    parser.add_argument("--output", type=Path, default=None,
                        help="write full ranked report as JSON")
    args = parser.parse_args(argv)

    # Load anchors + proxy
    anchors_path = args.anchors_path or (REPO_ROOT / ".omx" / "calibration" / f"anchors_{args.lane_class}.json")
    if not anchors_path.is_file():
        print(f"FATAL: anchors not found at {anchors_path}", file=sys.stderr)
        return 2
    anchors = load_calibration_anchors(anchors_path)
    proxy = make_distortion_proxy(anchors_path)

    # Load candidates
    if args.candidates_json:
        candidates = json.loads(args.candidates_json.read_text())
    elif args.auto_sweep_bits:
        bits = [int(b.strip()) for b in args.auto_sweep_bits.split(",")]
        candidates = _auto_sweep_apogee(bits)
    else:
        print("FATAL: must specify --candidates-json or --auto-sweep-bits", file=sys.stderr)
        return 2
    if not candidates:
        print("WARN: candidate list is empty", file=sys.stderr)

    # Constraint construction (override defaults if provided)
    constraints_kwargs = {}
    if args.rate_max is not None:
        constraints_kwargs["rate_unscaled_max"] = args.rate_max
    if args.pose_max is not None:
        constraints_kwargs["pose_dist_max"] = args.pose_max
    if args.seg_max is not None:
        constraints_kwargs["seg_dist_max"] = args.seg_max
    constraints = LagrangianConstraints(**constraints_kwargs)

    search = MetaLagrangianSearch(
        calibration_anchors=anchors,
        distortion_proxy=proxy,
        constraints=constraints,
    )

    # Coerce archive_path to Path object if present
    for c in candidates:
        if c.get("archive_path") and not isinstance(c["archive_path"], Path):
            c["archive_path"] = Path(c["archive_path"])

    evaluations = search.evaluate_all(candidates)
    top_k = MetaLagrangianSearch.top_k(evaluations, k=args.top_k)

    print(f"[meta-lagrangian] {len(candidates)} candidates evaluated")
    print(f"[meta-lagrangian] {len([e for e in evaluations if e.eligible_for_dispatch])} eligible for dispatch")
    print(f"[meta-lagrangian] top-{args.top_k}:")
    for i, ev in enumerate(top_k):
        print(f"  #{i+1}  {ev.candidate_id}  L={ev.lagrangian:.4f}  band=[{ev.band_low:.4f}, {ev.band_high:.4f}]  method={ev.band_method}")

    if args.output:
        report = {
            "schema": "meta_lagrangian_search_v1",
            "tag": "[distortion-proxy:local | predictor:calibrated_weak | sanity:gated]",
            "ts_utc": dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lane_class": args.lane_class,
            "anchors_path": str(anchors_path.relative_to(REPO_ROOT)),
            "constraints": asdict(constraints),
            "candidates_evaluated": len(candidates),
            "eligible_for_dispatch": len([e for e in evaluations if e.eligible_for_dispatch]),
            "top_k": [
                {**asdict(ev), "archive_path": str(getattr(ev, "archive_path", "")) if getattr(ev, "archive_path", None) else None}
                for ev in top_k
            ],
            "all_evaluations": [
                {**asdict(ev), "archive_path": str(getattr(ev, "archive_path", "")) if getattr(ev, "archive_path", None) else None}
                for ev in evaluations
            ],
            "warning": (
                "Outputs are [distortion-proxy:local] estimates ranked by Lagrangian. "
                "NEVER promote to [contest-CUDA] without upstream/evaluate.py on the EXACT bytes."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        try:
            display = args.output.resolve().relative_to(REPO_ROOT)
        except ValueError:
            display = args.output
        print(f"[meta-lagrangian] full report -> {display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
