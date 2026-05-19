# SPDX-License-Identifier: MIT
"""Modal A10G dispatch entry point for the MPS-train CUDA-score gap diagnostic.

Operator-routable surface: `tools/operator_authorize.py --recipe
mps_gap_experiment_tiny_renderer_modal_a10g_dispatch --target modal`.

The dispatch reads the canonical inputs (checkpoint_ema.pt + frame_cache.pt
that the local MPS training step produced), re-runs the checkpoint forward
on Modal A10G CUDA, and emits gap_results.json comparing the per-component
output values against the MPS reference forward.

NON-NEGOTIABLE per CLAUDE.md:

* No archive grammar. No contest score claim.
* Every artifact tagged ``evidence_grade="diagnostic-CUDA Modal A10G"``.
* All required input files exist before the Modal meter starts (Catalog #152).

This is a DIAGNOSTIC recipe — when run, the operator must already have
explicitly flipped ``dispatch_enabled: true`` in the recipe after reviewing
the landing memo.

NOTE: This file is intentionally a thin dispatch entry point. The heavy
lifting lives in ``tac.mps_gap_experiment.harvest_and_verdict``. The Modal
runtime side runs `compute_gap_components` with `target_device="cuda"`.

# NO_GRAD_WAIVED:thin-dispatch-entry-point; actual torch forward inference happens in tac.mps_gap_experiment.harvest_and_verdict::compute_gap_components which executes inside @torch.no_grad() context per the canonical helper contract. This dispatch entry point itself does NOT execute any torch graph; it just resolves CLI args + calls the canonical helper. Per Catalog #270 Tier-3 scope-fix for tool dispatches + the existing NO_GRAD_WAIVED token mechanism. Sister waiver pattern: see other dispatch_kind:tool entry points.

# TIER_1_OPERATOR_REQUIRED_FLAGS — declared per Catalog #151 so the
# operator wrapper threads them by default
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict] = {
    "--video-path": {
        "env": "MPS_GAP_VIDEO_PATH",
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
    },
    "--checkpoint-input": {
        "env": "MPS_GAP_CHECKPOINT_INPUT",
        "default": "experiments/results/mps_gap_experiment_local/checkpoint_ema.pt",
        "required_input_file": True,
    },
    "--frame-cache-input": {
        "env": "MPS_GAP_FRAME_CACHE_INPUT",
        "default": "experiments/results/mps_gap_experiment_local/frame_cache.pt",
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "MPS_GAP_OUTPUT_DIR",
        "default": "experiments/results/mps_gap_experiment_a10g_dispatch",
        "required_input_file": False,
    },
}

# Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM" + Catalog #152 WAVE-1 APPARATUS
# HARDENING extension 2026-05-16: required-input files under experiments/results/**
# are Modal-IGNORED by default (DEFAULT_RESULTS_IGNORE=("results/**",)); the trainer
# MUST declare them in TIER_1_EXTRA_MOUNT_PATHS so the canonical mount manifest
# stages them on the Modal worker. Without this, Catalog #152 fires the
# REQUIRED_INPUT_MODAL_STAGED_OK violation and refuses dispatch.
TIER_1_EXTRA_MOUNT_PATHS: tuple[str, ...] = (
    "experiments/results/mps_gap_experiment_local/checkpoint_ema.pt",
    "experiments/results/mps_gap_experiment_local/frame_cache.pt",
)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Catalog #151 manifest (module-level dict so the canonical extractor at
# `_check_151_extract_tier_manifests` picks it up; see Catalog #168 for the
# AST sister discipline that handles both Assign and AnnAssign).
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict] = {
    "--video-path": {
        "env": "MPS_GAP_VIDEO_PATH",
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
    },
    "--checkpoint-input": {
        "env": "MPS_GAP_CHECKPOINT_INPUT",
        "default": "experiments/results/mps_gap_experiment_local/checkpoint_ema.pt",
        "required_input_file": True,
    },
    "--frame-cache-input": {
        "env": "MPS_GAP_FRAME_CACHE_INPUT",
        "default": "experiments/results/mps_gap_experiment_local/frame_cache.pt",
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "MPS_GAP_OUTPUT_DIR",
        "default": "experiments/results/mps_gap_experiment_a10g_dispatch",
        "required_input_file": False,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", type=Path, default=Path("upstream/videos/0.mkv"))
    parser.add_argument("--checkpoint-input", type=Path, required=True)
    parser.add_argument("--frame-cache-input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/mps_gap_experiment_a10g_dispatch"),
    )
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument(
        "--include-scorer",
        action="store_true",
        help="include SegNet + PoseNet mean-output gap rows",
    )
    parser.add_argument(
        "--target-device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="target device for the comparison (cuda on Modal A10G; cpu for dry-run)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Per predecessor verdict mps_phase_b_gap_experiment_verdict_20260519T053530Z
    # Option A reactivation: the canonical split-device contract requires this
    # Modal-side dispatch to emit ONLY the target CUDA components (paired with
    # the local MPS reference captured during training). The local harvest
    # step diffs the two and emits the canonical gap_results.json.
    #
    # Import inside main so test-time imports don't pull torch + scorer.
    from tac.mps_gap_experiment.harvest_and_verdict import (
        compute_target_cuda_components,
    )

    modal_call_id = None
    # If the Modal runtime exposes the call id via env (per Catalog #245),
    # surface it into the manifest so the local harvest can cite the dispatch
    # provenance directly without a separate ledger lookup.
    import os

    modal_call_id = os.environ.get("MODAL_FUNCTION_CALL_ID") or os.environ.get(
        "DISPATCH_INSTANCE_JOB_ID"
    )

    components_path = compute_target_cuda_components(
        checkpoint_path=args.checkpoint_input,
        frame_cache_path=args.frame_cache_input,
        output_dir=args.output_dir,
        device=args.target_device,
        include_scorer_components=args.include_scorer,
        upstream_dir=args.upstream_dir,
        modal_call_id=modal_call_id,
    )
    print(
        f"[mps_gap_experiment] wrote target CUDA components to {components_path} "
        f"[diagnostic-CUDA Modal A10G]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
