#!/usr/bin/env python
"""Probe-disambiguator for (seg, pose) loss-weight ratio at the current operating point.

Council decision A-1 (2026-05-12, OOO commit 328bf2f9, 8/10 FOR Option C):
the legacy ``(seg=100, pose=10)`` loss-weight defaults were derived at the
**old 1.x operating point** where ``d_pose ~ 0.18``. At PR106 r2 (the current
contest frontier with ``d_pose ~ 3.4e-5``) the per-axis marginal sensitivity
**flips** — pose marginal exceeds SegNet's by roughly 2.71x. Per CLAUDE.md
"SegNet vs PoseNet importance — operating-point dependent" (2026-05-04
revision), the right loss weight ratio is OPERATING-POINT-dependent, not a
fixed constant.

This probe consumes the closed-form score gradient from
:func:`tac.score_geometry.score_gradient`:

    dS/d(d_seg)  = 100                       (SEG_COEFFICIENT)
    dS/d(d_pose) = 5 / sqrt(10 * d_pose)     (POSE_COEFFICIENT_INSIDE_SQRT=10)

and returns the operating-point-aware (seg_weight, pose_weight) ratio that
matches each axis's contribution to score motion.

CPU-only; $0 GPU. NOT a training loop. Outputs a single JSON line and exits.

Usage:
    python tools/probe_seg_pose_weight_at_operating_point.py \\
        --d-pose 3.4e-5 --d-seg 6.7e-4
    python tools/probe_seg_pose_weight_at_operating_point.py \\
        --operating-point pr106_r2
    python tools/probe_seg_pose_weight_at_operating_point.py \\
        --operating-point old_1x

Disambiguator output (JSON):
    {
      "operating_point": {"d_pose": float, "d_seg": float},
      "seg_weight_optimal": float,
      "pose_weight_optimal": float,
      "ratio_pose_over_seg": float,
      "basis": "closed-form gradient per tac.score_geometry.score_gradient",
      "legacy_ratio_pose_over_seg": 0.1,
      "ratio_flip_threshold_d_pose": 2.5e-4
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing tac.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.score_geometry import (  # noqa: E402
    POSE_COEFFICIENT_INSIDE_SQRT,
    SEG_COEFFICIENT,
    importance_flip_threshold,
    score_gradient,
)
from tac.sensitivity_map.axis_weights import (  # noqa: E402
    OPERATING_POINT_ANCHORS as _CANONICAL_AXIS_WEIGHT_ANCHORS,  # noqa: F401
)

# Operating-point d_pose / d_seg coordinates (DISTINCT from the canonical
# :data:`tac.sensitivity_map.axis_weights.OPERATING_POINT_ANCHORS`, which
# stores the COMPUTED :class:`AxisWeights` instances). This probe deals
# with the raw (d_pose, d_seg) coordinates the operator passes in and
# returns the score-gradient-balanced loss-weight ratio, so it needs the
# raw distortions, not the precomputed multipliers. The canonical
# AxisWeights table is re-exported above for any future caller that
# wants to consume the multiplier directly.
OPERATING_POINT_ANCHORS = {
    "old_1x": {"d_pose": 0.18, "d_seg": 0.001},  # 1.x scoreband
    "pr106_r2": {"d_pose": 3.4e-5, "d_seg": 6.7e-4},  # PR106 r2 frontier
    "pr102_cuda": {"d_pose": 5e-5, "d_seg": 5.8e-4},  # PR102 third-prize CUDA
}

# Legacy default ratio pose_weight / seg_weight = 10 / 100 = 0.1
LEGACY_POSE_OVER_SEG_RATIO = 10.0 / 100.0


def compute_optimal_weights(
    d_pose: float,
    d_seg: float,
) -> dict[str, float]:
    """Return operating-point-optimal (seg_weight, pose_weight) ratio.

    The score-gradient-balanced weights make the loss-gradient have the same
    per-axis magnitude as the score-gradient at the supplied operating point.

    Derivation:
        Score:      S = 100 * d_seg + sqrt(10 * d_pose) + RATE
        Gradient:   dS/d(d_seg) = 100
                    dS/d(d_pose) = 5 / sqrt(10 * d_pose)

        A loss L = w_seg * d_seg + w_pose * d_pose is "score-gradient
        balanced" when w_seg / w_pose == 100 / [5 / sqrt(10 * d_pose)]
                                     == 20 * sqrt(10 * d_pose).

        Equivalently:  w_pose / w_seg = 1 / [20 * sqrt(10 * d_pose)].

        We pin w_seg = 100 (the legacy/Shannon constant) and solve for w_pose.
    """
    if d_pose <= 0.0:
        raise ValueError("d_pose must be > 0 for a finite operating point")
    if d_seg < 0.0:
        raise ValueError("d_seg must be non-negative")

    grad = score_gradient(d_seg=d_seg, d_pose=d_pose)
    # By construction grad.d_seg == SEG_COEFFICIENT == 100.0
    seg_weight = float(SEG_COEFFICIENT)
    # pose_weight that makes per-axis gradient magnitudes equal:
    pose_weight = seg_weight * (grad.d_pose / grad.d_seg)
    ratio = pose_weight / seg_weight if seg_weight else float("inf")
    return {
        "seg_weight_optimal": seg_weight,
        "pose_weight_optimal": pose_weight,
        "ratio_pose_over_seg": ratio,
        "marginal_d_seg": grad.d_seg,
        "marginal_d_pose": grad.d_pose,
    }


def build_report(d_pose: float, d_seg: float) -> dict[str, object]:
    """Build the full probe JSON report."""
    weights = compute_optimal_weights(d_pose=d_pose, d_seg=d_seg)
    return {
        "operating_point": {"d_pose": d_pose, "d_seg": d_seg},
        "seg_weight_optimal": weights["seg_weight_optimal"],
        "pose_weight_optimal": weights["pose_weight_optimal"],
        "ratio_pose_over_seg": weights["ratio_pose_over_seg"],
        "marginal_d_seg": weights["marginal_d_seg"],
        "marginal_d_pose": weights["marginal_d_pose"],
        "basis": "closed-form gradient per tac.score_geometry.score_gradient",
        "legacy_ratio_pose_over_seg": LEGACY_POSE_OVER_SEG_RATIO,
        "ratio_flip_threshold_d_pose": importance_flip_threshold(),
        "score_coefficients": {
            "seg": float(SEG_COEFFICIENT),
            "pose_inside_sqrt": float(POSE_COEFFICIENT_INSIDE_SQRT),
        },
        "wire_in": {
            "sensitivity_map_hook": "marginal_d_seg + marginal_d_pose",
            "probe_disambiguator_hook": "this script IS the disambiguator",
        },
        "evidence_tag": "[derived: closed-form gradient per src/tac/score_geometry.py:253-257]",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--operating-point",
        choices=sorted(OPERATING_POINT_ANCHORS),
        help="A named operating-point anchor.",
    )
    group.add_argument(
        "--d-pose",
        type=float,
        help="Pose distortion at current operating point (provide --d-seg too).",
    )
    parser.add_argument(
        "--d-seg",
        type=float,
        help="Seg distortion at current operating point (required with --d-pose).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON file (default: stdout).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.operating_point is not None:
        anchor = OPERATING_POINT_ANCHORS[args.operating_point]
        d_pose, d_seg = anchor["d_pose"], anchor["d_seg"]
    else:
        if args.d_seg is None:
            print(
                "error: --d-seg is required when --d-pose is supplied",
                file=sys.stderr,
            )
            return 2
        d_pose, d_seg = args.d_pose, args.d_seg

    report = build_report(d_pose=d_pose, d_seg=d_seg)
    out_text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(out_text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(out_text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
