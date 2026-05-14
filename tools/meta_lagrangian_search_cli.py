#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Meta-Lagrangian search CLI — extreme automated Shannon-floor candidate raising.

Walks a candidate spec (JSON list, or auto-generated from the apogee bit-width
sweep) through the full pipeline:

    distortion_proxy → score_band predictor → Lagrangian → predispatch_sanity

Outputs a forensic ranked queue plus the fully evaluated non-eligible rows for
inspection. Per CLAUDE.md no-arbitrariness rule, every numeric threshold the
CLI uses carries a ``[heuristic:...]`` provenance tag in
:mod:`tac.optimizer.meta_lagrangian`.

CONTRACT: this CLI never spends GPU and never produces exact-eval readiness.
The search engine's local ``eligible_for_dispatch`` field is treated as an
internal ranking predicate only. CLI output remains
``ready_for_exact_eval_dispatch=false`` until a scorer-basin parity gate,
contest-faithful distortion model, or exact CUDA evidence validates the exact
candidate archive bytes.

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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.distortion_proxy_local import make_distortion_proxy  # type: ignore  # noqa: E402
from tac.optimization.meta_lagrangian_ledger_adapter import (  # noqa: E402
    adapt_artifact_to_atoms,
    search_candidates_from_atoms,
)
from tac.optimizer.meta_lagrangian import (  # noqa: E402
    CmaEsSearchBounds,
    LagrangianConstraints,
    MetaLagrangianSearch,
)
from tac.predictor.score_band import load_calibration_anchors  # noqa: E402

DISPATCH_BLOCKERS = [
    "meta_lagrangian_uses_distortion_proxy_local",
    "score_band_is_prediction_only",
    "missing_contest_faithful_distortion_model",
    "missing_scorer_basin_parity_gate",
    "missing_exact_cuda_candidate_evidence",
]


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


def _forensic_rank(evaluations) -> list:
    """Rank local evaluations for inspection without implying GPU readiness."""
    return sorted(evaluations, key=lambda ev: (ev.rank_key, ev.lagrangian, ev.candidate_id))


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_candidates_from_path(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Load strict candidates, adapting known ledger artifacts when needed."""

    if path.suffix != ".jsonl":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload, None
        if isinstance(payload, dict):
            embedded = payload.get("candidates_for_meta_lagrangian_search_cli")
            if isinstance(embedded, list):
                return embedded, {
                    "schema": "meta_lagrangian_candidate_input_adapter_v1",
                    "source_format": str(payload.get("schema") or "embedded_candidate_envelope"),
                    "source_path": _display_path(path),
                    "canonical_atom_count": len(payload.get("atoms") or payload.get("rows") or []),
                    "search_candidate_count": len(embedded),
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                    "dispatch_blockers": [
                        "embedded_candidates_still_forensic_until_exact_cuda",
                    ],
                }

    adapted = adapt_artifact_to_atoms(path, repo_root=REPO_ROOT)
    candidates = search_candidates_from_atoms(adapted.atoms)
    report = adapted.to_dict()
    report["search_projection_policy"] = (
        "only atoms with archive_bytes, rel_err_pct, and n_layers are projected "
        "to MetaLagrangianSearch.evaluate_candidate; all atoms remain available "
        "for field/Pareto planning through the canonical adapter rows"
    )
    report["search_candidate_count"] = len(candidates)
    report["candidates_for_meta_lagrangian_search_cli"] = candidates
    return candidates, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-class", required=True,
                        help="determines anchors file: anchors_<class>.json")
    parser.add_argument("--anchors-path", type=Path, default=None)
    parser.add_argument("--candidates-json", type=Path, default=None,
                        help="JSON file with a list of candidate dicts")
    parser.add_argument("--auto-sweep-bits", type=str, default=None,
                        help="comma-separated list of bit-widths to auto-walk apogee_int<N> archives")
    parser.add_argument(
        "--auto-cma-es",
        action="store_true",
        help="generate deterministic CPU-planning candidates with CMA-ES",
    )
    parser.add_argument("--cma-es-generations", type=int, default=4)
    parser.add_argument("--cma-es-population", type=int, default=8)
    parser.add_argument("--cma-es-seed", type=int, default=0)
    parser.add_argument("--cma-es-sigma", type=float, default=0.25)
    parser.add_argument("--cma-es-archive-min", type=int, default=100_000)
    parser.add_argument("--cma-es-archive-max", type=int, default=220_000)
    parser.add_argument("--cma-es-rel-err-min", type=float, default=0.0)
    parser.add_argument("--cma-es-rel-err-max", type=float, default=8.0)
    parser.add_argument("--cma-es-layers-min", type=int, default=1)
    parser.add_argument("--cma-es-layers-max", type=int, default=32)
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

    # Load or generate candidates. Exactly one source must be selected so
    # reports have an auditable provenance surface.
    source_count = sum(
        bool(value)
        for value in (args.candidates_json, args.auto_sweep_bits, args.auto_cma_es)
    )
    if source_count != 1:
        print(
            "FATAL: specify exactly one of --candidates-json, --auto-sweep-bits, "
            "or --auto-cma-es",
            file=sys.stderr,
        )
        return 2
    candidate_input_adapter = None
    candidate_generation_report = None
    if args.candidates_json:
        candidates, candidate_input_adapter = _load_candidates_from_path(args.candidates_json)
    elif args.auto_sweep_bits:
        bits = [int(b.strip()) for b in args.auto_sweep_bits.split(",")]
        candidates = _auto_sweep_apogee(bits)
    else:
        bounds = CmaEsSearchBounds(
            archive_bytes=(args.cma_es_archive_min, args.cma_es_archive_max),
            rel_err_pct=(args.cma_es_rel_err_min, args.cma_es_rel_err_max),
            n_layers=(args.cma_es_layers_min, args.cma_es_layers_max),
        )
        suggestions = search.suggest_cma_es_candidates(
            lane_class=args.lane_class,
            bounds=bounds,
            candidate_id_prefix=f"{args.lane_class}_cma_es",
            generations=args.cma_es_generations,
            population_size=args.cma_es_population,
            sigma=args.cma_es_sigma,
            seed=args.cma_es_seed,
        )
        candidates = [dict(item.candidate) for item in suggestions]
        candidate_generation_report = {
            "schema": "meta_lagrangian_cma_es_candidate_generation_v1",
            "planning_only": True,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_blockers": [
                "cma_es_candidate_generator_is_cpu_planning_only",
                "candidate_archive_missing",
                "requires_static_preflight",
                "requires_lane_dispatch_claim",
                "requires_exact_cuda_auth_eval",
            ],
            "seed": args.cma_es_seed,
            "generations": args.cma_es_generations,
            "population_size": args.cma_es_population,
            "sigma": args.cma_es_sigma,
            "bounds": asdict(bounds),
            "suggestion_count": len(suggestions),
            "top_suggestions": [
                {
                    "candidate": item.candidate,
                    "objective": item.objective,
                    "generation": item.generation,
                    "unit_vector": list(item.unit_vector),
                }
                for item in suggestions[: args.top_k]
            ],
        }
    if candidate_input_adapter is not None:
        print(
            "[meta-lagrangian] adapted "
            f"{candidate_input_adapter.get('atom_count', candidate_input_adapter.get('canonical_atom_count', 0))} "
            f"canonical atoms from {candidate_input_adapter.get('source_format')}; "
            f"{len(candidates)} strict search candidates"
        )
    if candidate_generation_report is not None:
        print(
            "[meta-lagrangian] generated "
            f"{candidate_generation_report['suggestion_count']} CMA-ES planning candidates "
            f"with seed={candidate_generation_report['seed']}"
        )
    if not candidates:
        print("WARN: candidate list is empty", file=sys.stderr)

    # Coerce archive_path to Path object if present
    for c in candidates:
        if c.get("archive_path") and not isinstance(c["archive_path"], Path):
            c["archive_path"] = Path(c["archive_path"])

    evaluations = search.evaluate_all(candidates)
    engine_top_k = MetaLagrangianSearch.top_k(evaluations, k=args.top_k)
    forensic_top_k = _forensic_rank(evaluations)[:args.top_k]
    dispatch_ready: list = []

    print(f"[meta-lagrangian] {len(candidates)} candidates evaluated")
    print(
        "[meta-lagrangian] "
        f"{len([e for e in evaluations if e.eligible_for_dispatch])} local search-engine eligible "
        "(forensic ranking only)"
    )
    print("[meta-lagrangian] 0 exact-eval dispatch-ready; local proxy/prediction evidence is non-promotable")
    print(f"[meta-lagrangian] forensic top-{args.top_k}:")
    for i, ev in enumerate(forensic_top_k):
        print(f"  #{i+1}  {ev.candidate_id}  L={ev.lagrangian:.4f}  band=[{ev.band_low:.4f}, {ev.band_high:.4f}]  method={ev.band_method}")

    if args.output:
        report = {
            "schema": "meta_lagrangian_search_v1",
            "tag": "[distortion-proxy:local | predictor:calibrated_weak | sanity:gated | dispatch:blocking]",
            "evidence_semantics": "local_proxy_prediction_forensic",
            "ready_for_exact_eval_dispatch": False,
            "dispatch_blockers": list(DISPATCH_BLOCKERS),
            "ts_utc": dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lane_class": args.lane_class,
            "anchors_path": _display_path(anchors_path),
            "constraints": asdict(constraints),
            "candidates_evaluated": len(candidates),
            "candidate_input_adapter": candidate_input_adapter,
            "candidate_generation_report": candidate_generation_report,
            "local_search_engine_eligible": len([e for e in evaluations if e.eligible_for_dispatch]),
            "ready_for_exact_eval_dispatch_count": len(dispatch_ready),
            "eligible_for_dispatch": 0,
            "top_k": [],
            "top_k_forensic": [
                {**asdict(ev), "archive_path": str(getattr(ev, "archive_path", "")) if getattr(ev, "archive_path", None) else None}
                for ev in forensic_top_k
            ],
            "engine_top_k_local_only": [
                {**asdict(ev), "archive_path": str(getattr(ev, "archive_path", "")) if getattr(ev, "archive_path", None) else None}
                for ev in engine_top_k
            ],
            "all_evaluations": [
                {**asdict(ev), "archive_path": str(getattr(ev, "archive_path", "")) if getattr(ev, "archive_path", None) else None}
                for ev in evaluations
            ],
            "warning": (
                "Outputs are [distortion-proxy:local] estimates ranked by Lagrangian. "
                "They are not GPU-dispatch readiness. NEVER promote to [contest-CUDA] "
                "without upstream/evaluate.py on the EXACT bytes or an explicit "
                "contest-faithful distortion/parity gate."
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
