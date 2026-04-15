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
}
"""Registry of all known Vast.ai experiments.

Usage::

    from tac.deploy.vastai.experiments import EXPERIMENTS
    config = EXPERIMENTS["tto_v1"]
"""
