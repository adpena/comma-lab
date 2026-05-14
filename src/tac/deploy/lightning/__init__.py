# SPDX-License-Identifier: MIT
"""Lightning AI Studio deployment helpers.

Lightning Studios are PERSISTENT (unlike Modal `.spawn()` 24h cache TTL or
Vast.ai per-lane spin-up). One Studio runs many lanes serially (one at a
time per GPU) or in parallel across multiple Studios.

Usage:
    from tac.deploy.lightning import LightningDispatcher

    d = LightningDispatcher(
        ssh_target="lightning-pact",
        remote_workspace="/teamspace/studios/this_studio/pact",
    )
    info = d.dispatch_lane(
        lane_script="scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh",
        label="lane_17_imp_2026-04-30_lightning",
        gpu_tier_required="A100",  # warns if Studio currently on different tier
        env_overrides={"AUTH_EVAL_DEVICE": "cuda", "IMP_QUICK_VARIANT": "0"},
    )
    # Later: d.harvest(info["session_id"], local_dir="experiments/results/...")

Cross-references:
- feedback_lightning_ai_ssh_credentials_20260430.md
- feedback_lightning_ai_activated_for_writeup_value_20260430.md
- experiments/modal_train_lane.py (Modal analog)
- scripts/launch_lane_with_retry.py (Vast.ai analog)
"""
from .batch_jobs import (
    LightningBatchJobsClient,
    LightningBatchJobSpec,
    exact_cuda_eval_command,
    make_exact_eval_spec,
)
from .lightning_dispatch import LightningDispatcher, DispatchResult

__all__ = [
    "DispatchResult",
    "LightningBatchJobsClient",
    "LightningBatchJobSpec",
    "LightningDispatcher",
    "exact_cuda_eval_command",
    "make_exact_eval_spec",
]
