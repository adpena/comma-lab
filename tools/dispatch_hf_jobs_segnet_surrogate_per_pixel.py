# SPDX-License-Identifier: MIT
"""Canonical SAM2 per-pixel SegNet surrogate distillation dispatcher.

Thin shim over ``tools/dispatch_hf_jobs_vision_training.py`` (slot 7's
canonical HF Jobs dispatcher) with SAM2-specific ``script_args`` baked in
per the ``huggingface-skills:hugging-face-vision-trainer`` plugin's
directive #4 SAM/SAM2 segmentation requirements.

Sister to slot 7's image-level dispatcher: honors the T1 symposium's
Contrarian VETO on image-level-only distillation metric per
``.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_per_pixel_symposium_20260519.md``.

**Per CLAUDE.md "Comment-only contracts are FORBIDDEN"**: the SAM2-specific
flag set assembled here is enforced as a list literal NOT a comment.

**Per Catalog #245 4-layer pattern**: every dispatch goes through
:func:`tac.deploy.hf_jobs.job_id_ledger.register_dispatched_hf_jobs_id`
via the canonical sister dispatcher BEFORE local entrypoint exit.

**Usage** (operator-facing):

.. code-block:: bash

   .venv/bin/python tools/dispatch_hf_jobs_segnet_surrogate_per_pixel.py \\
       --hub-dataset-repo adpena/comma-video-segnet-image-level-600pairs \\
       --hub-model-repo adpena/comma-segnet-surrogate-sam2-tiny-per-pixel \\
       --model facebook/sam2.1-hiera-tiny \\
       --flavor t4-small \\
       --num-epochs 30 \\
       --lane-id lane_hf_jobs_segnet_surrogate_distillation_per_pixel_20260519 \\
       --recipe .omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_per_pixel_t4_dispatch.yaml \\
       --label hf_jobs_segnet_surrogate_per_pixel_first_smoke \\
       --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists() and (parent / ".omx").exists():
            return parent
    raise RuntimeError(
        f"Could not locate repository root from {here!s}; expected "
        "pyproject.toml + .omx/ sibling."
    )


REPO_ROOT = _repo_root()
CANONICAL_TRAINING_SCRIPT = (
    REPO_ROOT / "experiments" / "hf_jobs_segnet_surrogate_distillation_per_pixel.py"
)
CANONICAL_DISPATCHER = REPO_ROOT / "tools" / "dispatch_hf_jobs_vision_training.py"


# SAM2-specific extra script args per the plugin's directive #4
# SAM/SAM2 segmentation requirements:
# - --remove_unused_columns False  (per-pixel mask + bbox columns must persist)
# - --prompt_type bbox             (bbox prompts extracted via scipy.ndimage)
# - --dataloader_pin_memory False  (SAM2 input_boxes tensors fail pin_memory)
# - --metric_for_best_model eval_mean_iou  (PRIMARY metric honoring Contrarian VETO)
# - --greater_is_better True       (mIoU: higher is better)
SAM2_PER_PIXEL_EXTRA_SCRIPT_ARGS: list[str] = [
    "--remove_unused_columns", "False",
    "--prompt_type", "bbox",
    "--dataloader_pin_memory", "False",
]


# SAM2-specific overrides for canonical sister flags (override
# image-classification defaults that don't apply to per-pixel SAM2):
# - metric_for_best_model: eval_mean_iou (NOT eval_accuracy)
# - default epochs: 30 (SAM2 fine-tune; canonical plugin default vs
#   image-classification's 200ep)
SAM2_DEFAULT_NUM_EPOCHS = 30
SAM2_DEFAULT_MODEL = "facebook/sam2.1-hiera-tiny"
SAM2_DEFAULT_METRIC = "eval_mean_iou"
SAM2_DEFAULT_HUB_MODEL_REPO = "adpena/comma-segnet-surrogate-sam2-tiny-per-pixel"


def _load_canonical_dispatcher():
    """Load the canonical sister dispatcher module via importlib."""

    spec = importlib.util.spec_from_file_location(
        "dispatch_hf_jobs_canonical", CANONICAL_DISPATCHER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load canonical dispatcher at {CANONICAL_DISPATCHER}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules["dispatch_hf_jobs_canonical"] = module
    spec.loader.exec_module(module)
    return module


def build_sam2_per_pixel_script_args(
    *,
    hub_dataset_repo: str,
    hub_model_repo: str,
    model: str = SAM2_DEFAULT_MODEL,
    num_epochs: int = SAM2_DEFAULT_NUM_EPOCHS,
    learning_rate: float = 1e-4,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
) -> list[str]:
    """Build the SAM2-specific script_args list for plan_dispatch.

    Mirrors the canonical sister dispatcher's plan_dispatch script_args
    construction but with the SAM2-specific overrides per
    SAM2_PER_PIXEL_EXTRA_SCRIPT_ARGS + SAM2_DEFAULT_METRIC + the
    DiceCELoss/per-pixel mIoU contract.
    """

    script_args: list[str] = [
        "--dataset_name", hub_dataset_repo,
        "--model_name_or_path", model,
        "--output_dir", "./output",
        "--hub_model_id", hub_model_repo,
        "--push_to_hub",
        "--metric_for_best_model", SAM2_DEFAULT_METRIC,
        "--greater_is_better", "True",
        "--num_train_epochs", str(num_epochs),
        "--learning_rate", str(learning_rate),
        "--per_device_train_batch_size", str(train_batch_size),
        "--per_device_eval_batch_size", str(eval_batch_size),
        "--do_train",
        "--do_eval",
        "--save_strategy", "epoch",
        "--eval_strategy", "epoch",
        "--load_best_model_at_end",
        "--logging_steps", "10",
        "--seed", "42",
    ]
    script_args.extend(SAM2_PER_PIXEL_EXTRA_SCRIPT_ARGS)
    return script_args


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dispatch_hf_jobs_segnet_surrogate_per_pixel",
        description=(
            "Dispatch a SAM2 per-pixel SegNet surrogate distillation HF Jobs run "
            "honoring the T1 symposium's Contrarian VETO on image-level-only "
            "distillation metric (Catalog #325 sister of slot 7's image-level lane)."
        ),
    )
    parser.add_argument(
        "--hub-dataset-repo",
        type=str,
        default="adpena/comma-video-segnet-image-level-600pairs",
        help="HF Hub source dataset repo id.",
    )
    parser.add_argument(
        "--hub-model-repo",
        type=str,
        default=SAM2_DEFAULT_HUB_MODEL_REPO,
        help="HF Hub destination model repo id.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=SAM2_DEFAULT_MODEL,
        help="SAM2 model identifier (default: sam2.1-hiera-tiny ~38.9M params).",
    )
    parser.add_argument(
        "--flavor",
        type=str,
        default="t4-small",
        help="HF Jobs hardware flavor (default: t4-small per slot 7 symposium).",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=SAM2_DEFAULT_NUM_EPOCHS,
        help="Training epochs (default: 30 per SAM2 plugin template).",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument(
        "--lane-id",
        type=str,
        default="lane_hf_jobs_segnet_surrogate_distillation_per_pixel_20260519",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="hf_jobs_segnet_surrogate_per_pixel_first_smoke",
    )
    parser.add_argument("--recipe", type=Path, default=None)
    parser.add_argument("--hub-dataset-sha", type=str, default=None)
    parser.add_argument(
        "--expected-axis", type=str, default="cuda", choices=("cuda", "cpu", "advisory")
    )
    parser.add_argument("--upstream-snapshot-sha256", type=str, default=None)
    parser.add_argument("--subagent-id", type=str, default=None)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    canonical = _load_canonical_dispatcher()

    # Build SAM2-specific script_args
    script_args = build_sam2_per_pixel_script_args(
        hub_dataset_repo=args.hub_dataset_repo,
        hub_model_repo=args.hub_model_repo,
        model=args.model,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    # Delegate to canonical dispatcher via plan_dispatch + extra_script_args
    # (we override its default script_args by passing them as
    # extra_script_args; the canonical dispatcher's defaults still emit but
    # the SAM2 ones override at the HF Trainer arg-parse level since
    # later --flag values win in HfArgumentParser).
    # Cleaner approach: bypass plan_dispatch and directly construct
    # HFJobsDispatchPlan with our pre-built script_args.

    if args.flavor not in canonical.HF_JOBS_FLAVOR_COSTS_USD_PER_HOUR:
        print(
            f"Error: flavor {args.flavor!r} not in canonical table; "
            f"choices: {sorted(canonical.HF_JOBS_FLAVOR_COSTS_USD_PER_HOUR)}",
            file=sys.stderr,
        )
        return 2
    if args.timeout_seconds <= 0:
        print("Error: --timeout-seconds must be positive", file=sys.stderr)
        return 2
    if args.num_epochs <= 0:
        print("Error: --num-epochs must be positive", file=sys.stderr)
        return 2

    estimated_cost = (
        canonical.HF_JOBS_FLAVOR_COSTS_USD_PER_HOUR[args.flavor]
        * (args.timeout_seconds / 3600.0)
    )

    plan = canonical.HFJobsDispatchPlan(
        script_path=CANONICAL_TRAINING_SCRIPT,
        script_args=script_args,
        flavor=args.flavor,
        timeout_seconds=args.timeout_seconds,
        estimated_cost_usd=estimated_cost,
        lane_id=args.lane_id,
        label=args.label,
        recipe_path=args.recipe,
        hub_model_repo=args.hub_model_repo,
        hub_dataset_repo=args.hub_dataset_repo,
    )

    if args.dry_run:
        import json
        print(json.dumps({"dry_run": True, "plan": plan.to_dict()}, indent=2, sort_keys=True))
        return 0

    try:
        result = canonical.dispatch(
            plan,
            subagent_id=args.subagent_id,
            session_id=args.session_id,
            hub_dataset_sha=args.hub_dataset_sha,
            expected_axis=args.expected_axis,
            upstream_snapshot_sha256=args.upstream_snapshot_sha256,
        )
    except RuntimeError as exc:
        import json
        print(
            json.dumps({"error": str(exc), "plan": plan.to_dict()}, indent=2, sort_keys=True),
            file=sys.stderr,
        )
        return 1

    import json
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
