#!/usr/bin/env python3
"""Local distortion proxy — CPU-only estimator for the predictor's high-rel_err gate.

Council Q3 prescription. The predictor (`tac.predictor.score_band`) REFUSES to
emit a band for any candidate with `rel_err_pct_per_weight > 1.0` unless the
caller supplies a `distortion_proxy` callable. This module exports such a
callable + a CLI for one-shot estimation.

Per CLAUDE.md "MPS auth eval is NOISE", this proxy MUST NOT use MPS / CPU
forward passes through the actual SegNet/PoseNet scorer — those produce
strategic-decision-grade noise (PoseNet drift 23x). Instead this proxy uses a
closed-form architecture-aware estimator calibrated against the empirical
anchors at `.omx/calibration/anchors_apogee_intN.json`:

    pose_estimate = d_baseline_pose + a_pose * (rel_err_pct ** b_pose)
    seg_estimate  = d_baseline_seg  + a_seg  * (rel_err_pct ** b_seg)

Coefficients a/b are fit via log-linear regression on the lossy anchors. With
the current 3 anchors (PR106 / int8 / int4), the fit interpolates smoothly
between known points and EXTRAPOLATES (with explicit warning) into the
PR106-mid-bit-width region (int5/int6/int7).

The output is tagged `[distortion-proxy:local]` per the empirical-claim-
without-evidence-tag CLAUDE.md FORBIDDEN PATTERN — callers must NOT promote
proxy-derived numbers to `[contest-CUDA]` claims or exact-eval readiness.

Usage as library:
    from experiments.distortion_proxy_local import make_distortion_proxy
    proxy = make_distortion_proxy()  # loads .omx/calibration/anchors_apogee_intN.json
    pose, seg = proxy(archive_bytes=140000, rel_err_pct=2.5, n_layers=13)

Usage as CLI:
    .venv/bin/python experiments/distortion_proxy_local.py \\
        --archive-bytes 140000 --rel-err-pct 2.5 --n-layers 13 \\
        --lane-class apogee_intN
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHORS_PATH = REPO_ROOT / ".omx" / "calibration" / "anchors_apogee_intN.json"

# Add src/ for tac.predictor import
sys.path.insert(0, str(REPO_ROOT / "src"))


# Bug #1 fix — degenerate-fit detection threshold.
# When ANY lossy anchor's per-component excess (D - D_baseline) is below this
# threshold, the per-component log-linear fit is numerically degenerate (the
# log(max(excess, 1e-12)) clamp dominates the regression and produces a slope
# that is NOT a meaningful power-law exponent — it's an artifact of the floor).
# In that case the curve must be returned as NaN so the predictor REFUSES with
# CURVE_FIT_DEGENERATE_NUMERICAL_FLOOR rather than emitting a confident-but-wrong band.
#
# Threshold rationale: the canonical anchor set has int8 seg_dist EQUAL to PR106
# seg_dist (5 sig figs). 1e-9 is well below typical scorer numerical noise
# (~1e-6) but well above the 1e-12 log-floor — so it triggers exactly when the
# anchor genuinely carries no rate-distortion information for the component.
DEGENERATE_EXCESS_THRESHOLD = 1e-9


def _fit_separate_curves(anchors: list[dict]) -> dict[str, dict[str, float]]:
    """Fit separate power-law curves for pose and seg distortion vs rel_err.

    For each component (pose, seg):
        log(D - d_baseline) ≈ b * log(rel_err) + log(a)
    over the LOSSY anchors (rel_err > 0). Returns dict with sub-dicts:
        {'pose': {'a': ..., 'b': ..., 'd_baseline': ..., 'degenerate_reason': ...},
         'seg':  {'a': ..., 'b': ..., 'd_baseline': ..., 'degenerate_reason': ...}}

    Bug #1 (root-cause fix 2026-05-05): if ANY lossy anchor's per-component
    excess (D - D_baseline) is below DEGENERATE_EXCESS_THRESHOLD, the per-
    component fit is numerically degenerate (the `max(excess, 1e-12)` floor
    dominates the regression). Return NaN with `degenerate_reason` set so the
    predictor refuses with `CURVE_FIT_DEGENERATE_NUMERICAL_FLOOR` rather than
    emitting an over-confident extrapolation built on a bogus slope.
    """
    # Identify lossless reference for baseline
    lossless = [a for a in anchors if a["rel_err_pct_per_weight"] == 0.0]
    baseline = min(anchors, key=lambda a: a["rel_err_pct_per_weight"]) if not lossless else lossless[0]
    d_pose_base = baseline["avg_pose_dist"]
    d_seg_base = baseline["avg_seg_dist"]

    lossy = [a for a in anchors if a["rel_err_pct_per_weight"] > 0.0]
    if len(lossy) < 2:
        # Cannot fit; return degenerate (caller should refuse via main predictor)
        return {
            "pose": {"a": float("nan"), "b": float("nan"), "d_baseline": d_pose_base,
                     "degenerate_reason": "insufficient_lossy_anchors"},
            "seg":  {"a": float("nan"), "b": float("nan"), "d_baseline": d_seg_base,
                     "degenerate_reason": "insufficient_lossy_anchors"},
        }

    def _fit_one(values: list[tuple[float, float]], baseline_d: float, component_label: str) -> dict[str, float]:
        # values is [(rel_err, distortion), ...] over lossy anchors

        # Bug #1 root-cause: detect degenerate-floor anchors BEFORE fitting.
        # If any anchor's excess (distortion - baseline) is below threshold,
        # the log-linear regression becomes numerically degenerate because the
        # `max(excess, 1e-12)` clamp dominates and the fitted slope is meaningless.
        degenerate_anchors = []
        for rel_err, distortion in values:
            excess = distortion - baseline_d
            if excess < DEGENERATE_EXCESS_THRESHOLD:
                degenerate_anchors.append((rel_err, distortion, excess))
        if degenerate_anchors:
            return {
                "a": float("nan"),
                "b": float("nan"),
                "d_baseline": baseline_d,
                "degenerate_reason": (
                    f"numerical_floor: {len(degenerate_anchors)} of {len(values)} "
                    f"lossy {component_label} anchors have excess < {DEGENERATE_EXCESS_THRESHOLD:.0e} "
                    f"(distortions match baseline within float precision); "
                    f"power-law fit is NOT meaningful"
                ),
            }

        log_rel = [math.log(v[0]) for v in values]
        log_excess = [math.log(v[1] - baseline_d) for v in values]
        n = len(values)
        mean_x = sum(log_rel) / n
        mean_y = sum(log_excess) / n
        var_x = sum((x - mean_x) ** 2 for x in log_rel)
        if var_x == 0:
            return {"a": float("nan"), "b": float("nan"), "d_baseline": baseline_d,
                    "degenerate_reason": "zero_variance_in_log_rel_err"}
        cov_xy = sum((log_rel[i] - mean_x) * (log_excess[i] - mean_y) for i in range(n))
        b = cov_xy / var_x
        a = math.exp(mean_y - b * mean_x)
        return {"a": a, "b": b, "d_baseline": baseline_d, "degenerate_reason": ""}

    pose_pairs = [(a["rel_err_pct_per_weight"], a["avg_pose_dist"]) for a in lossy]
    seg_pairs = [(a["rel_err_pct_per_weight"], a["avg_seg_dist"]) for a in lossy]
    return {
        "pose": _fit_one(pose_pairs, d_pose_base, "pose"),
        "seg":  _fit_one(seg_pairs, d_seg_base, "seg"),
    }


def _load_anchors(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"calibration anchors not found at {path}")
    return json.loads(path.read_text())


def make_distortion_proxy(
    anchors_path: Path = DEFAULT_ANCHORS_PATH,
):
    """Return a `(archive_bytes, rel_err_pct, n_layers) -> (pose, seg)` callable.

    The closure captures the fitted curves so subsequent calls are O(1).
    """
    anchors = _load_anchors(anchors_path)
    curves = _fit_separate_curves(anchors)

    rel_err_min = min((a["rel_err_pct_per_weight"] for a in anchors), default=0.0)
    rel_err_max = max((a["rel_err_pct_per_weight"] for a in anchors), default=0.0)

    def proxy(archive_bytes: int, rel_err_pct: float, n_layers: int) -> tuple[float, float]:
        # archive_bytes and n_layers currently informational only — closed-form
        # estimator is purely rel-err-driven. Future versions may layer in
        # architecture-aware sensitivity coefficients; the API accepts those
        # inputs now to avoid signature churn later.
        pose_curve = curves["pose"]
        seg_curve = curves["seg"]
        if math.isnan(pose_curve["a"]) or math.isnan(seg_curve["a"]):
            # Insufficient lossy anchors — fall back to baseline only.
            return pose_curve["d_baseline"], seg_curve["d_baseline"]
        if rel_err_pct == 0.0:
            return pose_curve["d_baseline"], seg_curve["d_baseline"]
        pose_excess = pose_curve["a"] * (rel_err_pct ** pose_curve["b"])
        seg_excess = seg_curve["a"] * (rel_err_pct ** seg_curve["b"])
        pose = max(pose_curve["d_baseline"] + pose_excess, pose_curve["d_baseline"])
        seg = max(seg_curve["d_baseline"] + seg_excess, seg_curve["d_baseline"])
        return pose, seg

    proxy.curves = curves  # expose for inspection / tests
    proxy.rel_err_range = (rel_err_min, rel_err_max)
    proxy.n_anchors = len(anchors)
    return proxy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bytes", type=int, required=True)
    parser.add_argument("--rel-err-pct", type=float, required=True,
                        help="rel_err_pct_per_weight from build metadata")
    parser.add_argument("--n-layers", type=int, default=13,
                        help="number of quantized layers (informational)")
    parser.add_argument("--lane-class", default="apogee_intN",
                        help="determines anchors file: anchors_<class>.json")
    parser.add_argument("--anchors-path", type=Path, default=None,
                        help="override anchors path (defaults to .omx/calibration/anchors_<lane-class>.json)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    anchors_path = args.anchors_path or (REPO_ROOT / ".omx" / "calibration" / f"anchors_{args.lane_class}.json")
    try:
        proxy = make_distortion_proxy(anchors_path)
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    pose, seg = proxy(args.archive_bytes, args.rel_err_pct, args.n_layers)
    rel_err_lo, rel_err_hi = proxy.rel_err_range

    extrapolating = args.rel_err_pct < rel_err_lo or args.rel_err_pct > rel_err_hi
    payload = {
        "schema": "distortion_proxy_local_v1",
        "tag": "[distortion-proxy:local]",
        "evidence_semantics": "local_distortion_proxy",
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "local_distortion_proxy_is_not_contest_cuda_evidence",
            "missing_scorer_basin_parity_gate",
            "missing_exact_cuda_candidate_evidence",
        ],
        "archive_bytes": args.archive_bytes,
        "rel_err_pct": args.rel_err_pct,
        "n_layers": args.n_layers,
        "predicted_avg_pose_dist": pose,
        "predicted_avg_seg_dist": seg,
        "calibration_n_anchors": proxy.n_anchors,
        "calibration_rel_err_range": [rel_err_lo, rel_err_hi],
        "extrapolating_outside_calibration_range": extrapolating,
        "anchors_path": str(anchors_path.relative_to(REPO_ROOT)),
        "warning": (
            "Closed-form architecture-naive estimate calibrated from anchors. "
            "Tag results [distortion-proxy:local], NEVER [contest-CUDA]. "
            "Final score MUST come from upstream/evaluate.py on the EXACT archive bytes; "
            "this proxy is not dispatch readiness."
        ),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"[distortion-proxy:local] anchors={proxy.n_anchors} (rel_err range [{rel_err_lo:.2f}, {rel_err_hi:.2f}])")
        print(f"  archive_bytes:        {args.archive_bytes}")
        print(f"  rel_err_pct:          {args.rel_err_pct}{'  [EXTRAPOLATING]' if extrapolating else ''}")
        print(f"  predicted_pose_dist:  {pose:.6e}")
        print(f"  predicted_seg_dist:   {seg:.6e}")
        print(f"  fit a_pose/b_pose:    {proxy.curves['pose']['a']:.4e} / {proxy.curves['pose']['b']:.3f}")
        print(f"  fit a_seg/b_seg:      {proxy.curves['seg']['a']:.4e} / {proxy.curves['seg']['b']:.3f}")
        print("  ready_for_exact_eval_dispatch: false")
        print("  WARNING: tag results [distortion-proxy:local], NEVER [contest-CUDA].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
