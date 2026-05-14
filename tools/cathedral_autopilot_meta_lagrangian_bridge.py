#!/usr/bin/env python3
"""B1 wiring: cathedral_autopilot recommendations -> MetaLagrangianSearch.

Converts cathedral_autopilot's catalog-recommended techniques into the
apogee_intN-style candidate dicts that MetaLagrangianSearch expects, runs
the engine, and emits the unified ranked output.

The bridge translates technique attributes:
  - predicted_archive_bytes -> archive_bytes
  - empirical_distortion_increase_pct (if present, else 0.0) -> rel_err_pct
  - heuristic n_layers from technique name (lossless=0, lossy_intN=full count)
  - lane_class = "cathedral_recommendation" + technique category

The output is a Lagrangian-ranked + sanity-gated list; this is the
bridge from the recommender's prediction-driven catalog to the search
engine's optimization-driven Lagrangian over real PR106 constraints.

CLAUDE.md compliance: pure CPU + math; no scorer load; no contest score
claims; output is planning-only. The MetaLagrangianSearch engine itself
already enforces the [distortion-proxy:local] tag and refuses to mark
any candidate ready_for_exact_eval_dispatch.

Usage::

    .venv/bin/python tools/cathedral_autopilot_meta_lagrangian_bridge.py \\
        --plan-json reports/cathedral_autopilot_plan.json \\
        --output reports/cathedral_meta_lagrangian_ranking.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))  # so 'experiments.distortion_proxy_local' resolves

from tac.authority_contract import apply_false_authority_contract  # noqa: E402

TOOL_NAME = "tools/cathedral_autopilot_meta_lagrangian_bridge.py"
SCHEMA_VERSION = "cathedral_meta_lagrangian_bridge.v1"


# Heuristic n_layers per technique class. PR106 HNeRV has ~28 layers; the
# meta-Lagrangian engine uses n_layers as a quantization-stack-depth proxy.
HEURISTIC_N_LAYERS = {
    "brotli_optuna_default": 0,
    "per_tensor_adaptive_aac": 0,
    "tiny_nn_pmf_predictor": 0,
    "compressai_balle_hyperprior": 0,
    "kalle_fold_mixture_canonical_shapes": 0,
    "lossy_int4_quantization": 28,  # all 28 tensors quantized to int4
    "sparsity_alpha_0.7_imp_retrain": 0,  # sparse, not quantized
    "arch_shrink_x0.4_quantizr_class": 0,
    "self_compress_neural_codec": 0,
}


def _rank_axis_metadata(source_row: dict, plan_payload: dict) -> tuple[str, bool]:
    """Return recommender rank-axis metadata without silently preferring CUDA.

    Current cathedral_autopilot rows carry ``rank_axis`` explicitly. Older
    plan JSON may only have the value on ``operator_state`` or may be missing
    it entirely. Missing metadata is treated as the current conservative dual
    default, but the caller also receives a boolean so stale plans remain
    visible in downstream manifests.
    """
    operator_state = plan_payload.get("operator_state", {})
    value = (
        source_row.get("rank_axis")
        or operator_state.get("rank_axis")
        or plan_payload.get("rank_axis")
    )
    missing = value is None
    if value is None:
        value = "dual"
    if value not in {"dual", "cuda", "cpu"}:
        raise ValueError(
            "invalid rank_axis in cathedral_autopilot plan row "
            f"{source_row.get('name', '<unknown>')!r}: {value!r}"
        )
    return str(value), missing


def _current_score_axis_metadata(source_row: dict, plan_payload: dict) -> tuple[str, bool]:
    """Return current-score-axis metadata, preserving missing legacy state."""
    operator_state = plan_payload.get("operator_state", {})
    value = (
        source_row.get("current_score_axis")
        or operator_state.get("current_score_axis")
        or plan_payload.get("current_score_axis")
    )
    missing = value is None
    if value is None:
        return "unspecified", True
    if value not in {"cuda", "cpu"}:
        raise ValueError(
            "invalid current_score_axis in cathedral_autopilot plan row "
            f"{source_row.get('name', '<unknown>')!r}: {value!r}"
        )
    return str(value), False


def _technique_to_candidate(t: dict) -> dict:
    """Convert one recommender row into a MetaLagrangianSearch candidate dict."""
    name = t["name"]
    archive_bytes = int(t.get("predicted_archive_bytes", 0))
    rel_err_pct = float(t.get("empirical_distortion_increase_pct", 0.0))
    n_layers = HEURISTIC_N_LAYERS.get(name, 0)
    return {
        "candidate_id": f"autopilot_{name}",
        "archive_bytes": archive_bytes,
        "rel_err_pct": rel_err_pct,
        "n_layers": n_layers,
        "lane_class": "cathedral_recommendation",
        "archive_path": None,  # no archive yet built
        "_source_row": t,
    }


def run_bridge(plan_json_path: Path, anchors_path: Path) -> dict:
    from experiments.distortion_proxy_local import make_distortion_proxy
    from tac.optimizer.meta_lagrangian import (
        LagrangianConstraints,
        MetaLagrangianSearch,
    )
    from tac.predictor.score_band import load_calibration_anchors

    if not plan_json_path.is_file():
        raise SystemExit(f"plan json not found: {plan_json_path}")
    plan_payload = json.loads(plan_json_path.read_text(encoding="utf-8"))

    # Build candidate list from the plan's encoder + arch rankings (top combined)
    enc_rows = plan_payload.get("encoder_technique_ranking", [])
    arch_rows = plan_payload.get("arch_technique_ranking", [])
    all_rows = enc_rows + arch_rows

    candidates = [_technique_to_candidate(r) for r in all_rows]
    if not candidates:
        return {
            "schema": SCHEMA_VERSION,
            "tool": TOOL_NAME,
            "n_candidates": 0,
            "evaluations": [],
            "warning": "plan json has no encoder or arch technique rankings",
        }

    proxy = make_distortion_proxy(anchors_path)
    anchors = load_calibration_anchors(anchors_path)
    search = MetaLagrangianSearch(
        calibration_anchors=anchors,
        distortion_proxy=proxy,
        constraints=LagrangianConstraints(),
    )

    evaluations: list[dict] = []
    for cand in candidates:
        rank_axis, rank_axis_missing = _rank_axis_metadata(
            cand["_source_row"],
            plan_payload,
        )
        current_score_axis, current_score_axis_missing = _current_score_axis_metadata(
            cand["_source_row"],
            plan_payload,
        )
        ev = search.evaluate_candidate(
            candidate_id=cand["candidate_id"],
            archive_bytes=cand["archive_bytes"],
            rel_err_pct=cand["rel_err_pct"],
            n_layers=cand["n_layers"],
            lane_class=cand["lane_class"],
            archive_path=cand["archive_path"],
        )
        evaluations.append(apply_false_authority_contract({
            "candidate_id": ev.candidate_id,
            "technique_name": cand["_source_row"]["name"],
            "archive_bytes": ev.archive_bytes,
            "rel_err_pct": ev.rel_err_pct,
            "proxy_pose": ev.proxy_pose,
            "proxy_seg": ev.proxy_seg,
            "proxy_score": ev.proxy_score,
            "band_low": ev.band_low,
            "band_high": ev.band_high,
            "band_refused": ev.band_refused,
            "band_refusal_reason": ev.band_refusal_reason,
            "lagrangian": ev.lagrangian,
            "rate_violation": ev.rate_violation,
            "pose_violation": ev.pose_violation,
            "seg_violation": ev.seg_violation,
            "sanity_passed": ev.sanity_passed,
            "sanity_failures": list(ev.sanity_failures),
            "rank_key": ev.rank_key,
            "eligible_for_dispatch": ev.eligible_for_dispatch,
            "predicted_score_delta_from_recommender": (
                cand["_source_row"].get("predicted_score_delta", 0.0)
            ),
            "rank_axis_from_recommender": rank_axis,
            "rank_axis_missing_from_recommender": rank_axis_missing,
            "primary_score_delta_from_recommender": (
                cand["_source_row"].get("primary_score_delta", 0.0)
            ),
            "predicted_cuda_score_delta_from_recommender": (
                cand["_source_row"].get("predicted_cuda_score_delta", 0.0)
            ),
            "predicted_cpu_score_delta_from_recommender": (
                cand["_source_row"].get("predicted_cpu_score_delta", 0.0)
            ),
            "predicted_cpu_score_calibration": (
                cand["_source_row"].get("predicted_cpu_score_calibration", "")
            ),
            "current_score_axis_from_recommender": current_score_axis,
            "current_score_axis_missing_from_recommender": current_score_axis_missing,
        }, preserve_dispatch_ready=False))

    # Sort by Lagrangian (lower = better; refused/failed sort to bottom via inf)
    evaluations.sort(key=lambda e: (e["rank_key"], e["lagrangian"]))

    return apply_false_authority_contract({
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": "[distortion-proxy:local]",
        "input_plan_json": str(plan_json_path),
        "anchors_path": str(anchors_path),
        "n_candidates": len(candidates),
        "n_eligible_for_dispatch": sum(1 for e in evaluations if e["eligible_for_dispatch"]),
        "evaluations": evaluations,
    }, preserve_dispatch_ready=False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-json", type=Path, required=True,
                   help="Output of cathedral_autopilot plan or plan-from-pareto")
    p.add_argument("--anchors-path", type=Path,
                   default=REPO_ROOT / ".omx/calibration/anchors_apogee_intN.json")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.anchors_path.is_file():
        raise SystemExit(f"anchors path not found: {args.anchors_path}")

    manifest = run_bridge(args.plan_json, args.anchors_path)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output is None:
        out_dir = REPO_ROOT / f"reports/raw/cathedral_meta_lagrangian_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / "manifest.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"manifest: {args.output}\n")
    print(f"{'#':>3} {'technique':<40s} {'L':>10s} {'sanity':>8s} {'band':>20s}")
    for i, ev in enumerate(manifest.get("evaluations", []), 1):
        band_str = f"[{ev['band_low']:.4f}, {ev['band_high']:.4f}]"
        sanity_str = "PASS" if ev["sanity_passed"] else "FAIL"
        L_str = f"{ev['lagrangian']:.4f}" if ev["lagrangian"] != float("inf") else "inf"
        print(f"{i:>3} {ev['technique_name']:<40s} {L_str:>10s} {sanity_str:>8s} {band_str:>20s}")

    print(f"\nn_candidates={manifest['n_candidates']}, "
          f"n_eligible_for_dispatch={manifest.get('n_eligible_for_dispatch', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
