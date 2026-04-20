"""Experiment registry for Vast.ai deployments.

Defines all known experiment configurations as :class:`ExperimentConfig`
instances.  The registry is a plain dict so callers can extend it at runtime.
"""
from __future__ import annotations

from tac.deploy.base import ExperimentConfig

# ── Vast.ai experiment registry ──────────────────────────────────────────────
# Each entry uses ExperimentConfig from tac.deploy.base, which is
# platform-agnostic.  The estimated_cost_per_hour is set to typical
# Vast.ai RTX 4090 pricing (~$0.25/hr).

VASTAI_COST_PER_HOUR_4090 = 0.25
"""Typical on-demand rate for RTX 4090 on Vast.ai (USD/hr)."""


def _exp(
    name: str,
    script: str,
    args: str,
    timeout_hours: float = 4.0,
    needs_upstream: bool = True,
    needs_checkpoint: str | None = None,
) -> ExperimentConfig:
    """Helper to build an ExperimentConfig with Vast.ai defaults."""
    return ExperimentConfig(
        name=name,
        script=script,
        args=args.split(),
        needs_upstream=needs_upstream,
        needs_checkpoint=needs_checkpoint,
        timeout_hours=timeout_hours,
        gpu_type="RTX 4090",
        estimated_cost_per_hour=VASTAI_COST_PER_HOUR_4090,
    )


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "tto_v1": _exp(
        name="tto_v1",
        script="experiments/renderer_tto.py",
        args=(
            "--checkpoint renderer_best.pt --device cuda --n-frames 1200 "
            "--tto-steps 500 --tto-lr 0.005 --batch-pairs 10 "
            "--seg-weight 100 --pose-weight 10 --compress-weight 0.5 "
            "--simulate-resize"
        ),
        needs_checkpoint="renderer_best.pt",
        timeout_hours=2,
    ),
    "tto_v1_seg_odd": _exp(
        name="tto_v1_seg_odd",
        script="experiments/renderer_tto.py",
        args=(
            "--checkpoint renderer_best.pt --device cuda --n-frames 1200 "
            "--tto-steps 500 --tto-lr 0.005 --batch-pairs 10 "
            "--seg-weight 100 --pose-weight 10 --compress-weight 0.5 "
            "--seg-odd-only --simulate-resize"
        ),
        needs_checkpoint="renderer_best.pt",
        timeout_hours=2,
    ),
    "sensitivity_map": _exp(
        name="sensitivity_map",
        script="experiments/analysis/posenet_sensitivity.py",
        args="--device cuda --n-frames 1200",
        timeout_hours=1,
    ),
    "warp_baseline": _exp(
        name="warp_baseline",
        script="experiments/warp_gen_baseline.py",
        args="--device cuda --n-frames 1200",
        timeout_hours=1,
    ),
    "gt_sparse_tto": _exp(
        name="gt_sparse_tto",
        script="experiments/gt_sparse_tto.py",
        args=(
            "--device cuda --n-frames 1200 --n-patches 50 "
            "--n-restarts 20 --steps-per-restart 500"
        ),
        timeout_hours=4,
    ),
    "joint_pair_train": _exp(
        name="joint_pair_train",
        script="experiments/train_joint_pair.py",
        args="--device cuda --epochs 100",
        needs_checkpoint="renderer_best.pt",
        timeout_hours=20,
    ),
    # ── v5b config (embedding loss, NEW BEST auth=0.41) ──
    "tto_v5b_embedding": _exp(
        name="tto_v5b_embedding",
        script="experiments/renderer_tto.py",
        args=(
            "--checkpoint renderer_best.pt --device cuda --n-frames 1200 "
            "--tto-steps 500 --tto-lr 0.005 --batch-pairs 10 "
            "--seg-weight 100 --pose-weight 10 --compress-weight 0.5 "
            "--use-embedding-loss --seg-odd-only --early-stop-patience 150 "
            "--simulate-resize"
        ),
        needs_checkpoint="renderer_best.pt",
        timeout_hours=3,
    ),
    # ── TTO step curve (highest signal experiment) ──
    "tto_step_curve": _exp(
        name="tto_step_curve",
        script="experiments/tto_step_curve.py",
        args=(
            "--checkpoint renderer_best.pt --device cuda "
            "--step-counts 10,25,50,100,150,200,300,500"
        ),
        needs_checkpoint="renderer_best.pt",
        timeout_hours=3,
    ),
    # ── TTO step curve with hinge loss ──
    "tto_step_curve_hinge": _exp(
        name="tto_step_curve_hinge",
        script="experiments/tto_step_curve.py",
        args=(
            "--checkpoint renderer_best.pt --device cuda "
            "--step-counts 10,25,50,100,150,200,300,500 "
            "--segnet-loss-mode hinge"
        ),
        needs_checkpoint="renderer_best.pt",
        timeout_hours=3,
    ),
    # ── TTO step curve with cosine LR ──
    "tto_step_curve_cosine": _exp(
        name="tto_step_curve_cosine",
        script="experiments/tto_step_curve.py",
        args=(
            "--checkpoint renderer_best.pt --device cuda "
            "--step-counts 10,25,50,100,150,200,300,500 "
            "--lr-schedule cosine"
        ),
        needs_checkpoint="renderer_best.pt",
        timeout_hours=3,
    ),
    # ── v5c resume (PoseNet ceiling, aggressive 100:1 pose:seg) ──
    "tto_v5c_pose_aggressive": _exp(
        name="tto_v5c_pose_aggressive",
        script="experiments/renderer_tto.py",
        args=(
            "--checkpoint renderer_best.pt --device cuda --n-frames 1200 "
            "--tto-steps 1000 --tto-lr 0.005 --batch-pairs 10 "
            "--seg-weight 1 --pose-weight 100 --compress-weight 0.01 "
            "--seg-odd-only --early-stop-patience 300 "
            "--simulate-resize"
        ),
        needs_checkpoint="renderer_best.pt",
        timeout_hours=6,
    ),
    # ── v6 config: ALL improvements (hinge + phase2 + embedding + simulate_resize) ──
    "tto_v6_hinge_phase2": _exp(
        name="tto_v6_hinge_phase2",
        script="experiments/renderer_tto.py",
        args=(
            "--checkpoint renderer_best.pt --device cuda --n-frames 1200 "
            "--tto-steps 150 --tto-lr 0.005 --batch-pairs 10 "
            "--seg-weight 100 --pose-weight 10 --compress-weight 0.5 "
            "--use-embedding-loss --seg-odd-only --early-stop-patience 100 "
            "--simulate-resize "
            "--segnet-loss-mode hinge --hinge-margin 0.5 "
            "--tto-phase2-segnet-only --phase2-steps 200"
        ),
        needs_checkpoint="renderer_best.pt",
        timeout_hours=2,
    ),
    # ── Contest pipeline profiler (T4 timing data) ──
    "contest_dry_run": _exp(
        name="contest_dry_run",
        script="scripts/profile_contest_pipeline.py",
        args=(
            "--submission-dir submissions/robust_current --device cuda"
        ),
        timeout_hours=1,
    ),
    # ── v7 TTO: hinge + eval roundtrip + FiLM (when FiLM conditioning is available) ──
    # Adds --simulate-resize and eval-roundtrip to hinge baseline from v6.
    # FiLM pose conditioning gated: enable once FiLM arch lands in inflate_renderer.py.
    "tto_v7_hinge_roundtrip": _exp(
        name="tto_v7_hinge_roundtrip",
        script="experiments/renderer_tto.py",
        args=(
            "--checkpoint renderer_best.pt --device cuda --n-frames 1200 "
            "--tto-steps 500 --tto-lr 0.005 --batch-pairs 10 "
            "--seg-weight 100 --pose-weight 10 --compress-weight 0.5 "
            "--use-embedding-loss --seg-odd-only --early-stop-patience 500 "
            "--simulate-resize "
            "--segnet-loss-mode hinge --hinge-margin 0.5"
        ),
        needs_checkpoint="renderer_best.pt",
        timeout_hours=3,
    ),
}
"""Registry of all known Vast.ai experiments.

Usage::

    from tac.deploy.vastai.experiments import EXPERIMENTS
    config = EXPERIMENTS["tto_v1"]
"""
