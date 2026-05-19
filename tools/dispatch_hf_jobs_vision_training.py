# SPDX-License-Identifier: MIT
"""Canonical HF Jobs vision-training dispatcher (Catalog #342 + Item #878).

Wraps ``huggingface_hub.HfApi().run_uv_job(...)`` per the
``huggingface-skills:hugging-face-vision-trainer`` plugin canonical
patterns:

- **Directive #1**: pass all config via ``script_args`` (NOT by editing
  Python variables in the training script).
- **Directive #2**: inject ``HF_TOKEN`` into both the job's
  ``secrets={"HF_TOKEN": <actual token value>}`` AND environment so the
  training script can push to Hub AFTER the Trainer is constructed.
- **Directive #3**: return ``job_info.id`` so the operator + canonical
  ledger can track lifecycle.
- **Directive #4**: ensure the training script invocation includes
  ``--no_remove_unused_columns --push_to_hub --metric_for_best_model
  eval_accuracy --greater_is_better True`` required for the plugin's
  canonical eval contract.
- **Directive #5**: set ``timeout`` explicitly (4h default for surrogate
  training).
- **Directive #6**: ``flavor="t4-small"`` recommendation for OD/IC models
  under 100M params (e.g. ``timm/mobilenetv3_small_100.lamb_in1k`` at
  ~2.5M params).

Per Catalog #245 4-layer canonical pattern: every dispatch goes through
:func:`tac.deploy.hf_jobs.job_id_ledger.register_dispatched_hf_jobs_id`
BEFORE local entrypoint exit so the canonical ledger captures the
hf_jobs_id even if the per-dispatch sentinel files are not written.

**Usage** (operator-facing):

.. code-block:: bash

   .venv/bin/python tools/dispatch_hf_jobs_vision_training.py \\
       --script experiments/hf_jobs_segnet_surrogate_distillation.py \\
       --hub-dataset-repo adpena/comma-video-segnet-image-level-600pairs \\
       --hub-model-repo adpena/segnet-image-level-surrogate-mobilenet-v3-small-200ep \\
       --model timm/mobilenetv3_small_100.lamb_in1k \\
       --flavor t4-small \\
       --num-epochs 200 \\
       --lane-id lane_hf_jobs_segnet_surrogate_distillation_20260519 \\
       --recipe .omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml \\
       --label hf_jobs_segnet_surrogate_distillation_200ep_first_smoke \\
       --dry-run

   # Real dispatch (drops --dry-run; consult Catalog #270 dispatch protocol
   # + Catalog #243 local pre-deploy + Catalog #271 codex pre-dispatch review
   # via the canonical operator_authorize.py wrapper)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


# Canonical HF Jobs flavors per
# huggingface-skills:hugging-face-vision-trainer plugin directive #6.
# Costs fetched from the official Hugging Face Hub Jobs pricing page on
# 2026-05-19 (subject to change; verify via `hf jobs hardware` or the
# official docs before paid dispatch).
HF_JOBS_FLAVOR_COSTS_USD_PER_HOUR: dict[str, float] = {
    "cpu-basic": 0.01,  # for dataset prep / inference smoke
    "cpu-upgrade": 0.03,
    "cpu-xl": 1.00,
    "cpu-performance": 1.90,
    "t4-small": 0.40,  # baseline for OD/IC under 100M params
    "t4-medium": 0.60,
    "l4x1": 0.80,
    "l4x4": 3.80,
    "l40s-x1": 1.80,
    "l40s-x4": 8.30,
    "l40s-x8": 23.50,
    "a10g-small": 1.00,
    "a10g-large": 1.50,
    "a10g-large-x2": 3.00,
    "a10g-large-x4": 5.00,
    "a100-large": 2.50,
    "a100-large-x4": 10.00,
    "a100-large-x8": 20.00,
    "h200": 5.00,
    "h200-x2": 10.00,
    "h200-x4": 20.00,
    "h200-x8": 40.00,
}


@dataclass(frozen=True)
class HFJobsDispatchPlan:
    """Resolved dispatch plan, returned by :func:`plan_dispatch`.

    Per Catalog #229 premise-verification + Catalog #243 local pre-deploy
    + Catalog #271 codex pre-dispatch review pattern: the operator can
    audit the plan via ``--dry-run`` BEFORE the paid dispatch fires.
    """

    script_path: Path
    script_args: list[str]
    flavor: str
    timeout_seconds: int
    estimated_cost_usd: float
    lane_id: str
    label: str
    recipe_path: Path | None
    hub_model_repo: str
    hub_dataset_repo: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "script_path": str(self.script_path),
            "script_args": list(self.script_args),
            "flavor": self.flavor,
            "timeout_seconds": self.timeout_seconds,
            "estimated_cost_usd": self.estimated_cost_usd,
            "lane_id": self.lane_id,
            "label": self.label,
            "recipe_path": str(self.recipe_path) if self.recipe_path else None,
            "hub_model_repo": self.hub_model_repo,
            "hub_dataset_repo": self.hub_dataset_repo,
        }


def _resolve_hf_token() -> str | None:
    """Return the active HF token (env var or canonical ``hf auth whoami``).

    Per the plugin's directive #2: the dispatcher MUST resolve the token
    value (NOT the literal ``"$HF_TOKEN"`` placeholder string) so the HF
    Jobs ``secrets`` block injects the actual token into the remote
    environment.
    """

    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    # Try the canonical huggingface_hub helper if available.
    try:
        from huggingface_hub import get_token  # type: ignore[import-not-found]

        return get_token()
    except Exception:
        return None


def plan_dispatch(
    *,
    script: Path,
    hub_dataset_repo: str,
    hub_model_repo: str,
    model: str = "timm/mobilenetv3_small_100.lamb_in1k",
    flavor: str = "t4-small",
    num_epochs: int = 200,
    learning_rate: float = 1e-4,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    timeout_seconds: int = 14400,  # 4h per plugin directive #5
    lane_id: str = "lane_hf_jobs_segnet_surrogate_distillation_20260519",
    label: str = "hf_jobs_segnet_surrogate_distillation_first_smoke",
    recipe_path: Path | None = None,
    extra_script_args: list[str] | None = None,
) -> HFJobsDispatchPlan:
    """Resolve a complete HF Jobs dispatch plan.

    Pure-function: does NOT contact the HF Hub API (so it is safe to call
    from --dry-run / tests / Catalog #243 local pre-deploy harness).

    Per plugin directive #1 + #4 the script_args list assembles the full
    canonical training-script CLI INCLUDING the required flags (
    ``--no_remove_unused_columns --push_to_hub --metric_for_best_model
    eval_accuracy --greater_is_better True``).
    """

    if flavor not in HF_JOBS_FLAVOR_COSTS_USD_PER_HOUR:
        raise ValueError(
            f"flavor {flavor!r} not in {sorted(HF_JOBS_FLAVOR_COSTS_USD_PER_HOUR)}"
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")

    estimated_cost = (
        HF_JOBS_FLAVOR_COSTS_USD_PER_HOUR[flavor] * (timeout_seconds / 3600.0)
    )

    # Canonical training-script CLI per plugin directive #4 (REQUIRED flags
    # for the upstream image_classification_training.py template).
    script_args: list[str] = [
        "--dataset_name", hub_dataset_repo,
        "--model_name_or_path", model,
        "--output_dir", "./output",
        "--hub_model_id", hub_model_repo,
        "--push_to_hub",
        "--no_remove_unused_columns",
        "--metric_for_best_model", "eval_accuracy",
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
    if extra_script_args:
        script_args.extend(extra_script_args)

    return HFJobsDispatchPlan(
        script_path=script,
        script_args=script_args,
        flavor=flavor,
        timeout_seconds=timeout_seconds,
        estimated_cost_usd=estimated_cost,
        lane_id=lane_id,
        label=label,
        recipe_path=recipe_path,
        hub_model_repo=hub_model_repo,
        hub_dataset_repo=hub_dataset_repo,
    )


def _git_head_sha(repo_root: Path) -> str | None:
    """Return current git HEAD sha (None if not a git repo)."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def dispatch(
    plan: HFJobsDispatchPlan,
    *,
    subagent_id: str | None = None,
    session_id: str | None = None,
    register_in_ledger: bool = True,
    hub_dataset_sha: str | None = None,
    expected_axis: str = "cuda",
    upstream_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    """Execute the HF Jobs dispatch + register in the canonical ledger.

    Per Catalog #245 sister 4-layer pattern: the canonical ledger row
    lands BEFORE this function returns so a successor agent can query
    the hf_jobs_id via
    :func:`tac.deploy.hf_jobs.job_id_ledger.query_by_lane`.

    Returns a dict with keys ``hf_jobs_id``, ``ledger_row``, ``plan``.
    """

    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "`huggingface_hub` required for dispatch; install via "
            "`uv pip install huggingface_hub`"
        ) from exc

    token = _resolve_hf_token()
    if not token:
        raise RuntimeError(
            "No HF token resolved; set HF_TOKEN env var OR run `hf auth login`. "
            "Per plugin directive #2 the token MUST be the actual value, NOT "
            "the literal `$HF_TOKEN` placeholder."
        )

    api = HfApi(token=token)
    # Per directive #1: pass script_args (NOT edit script variables).
    # Per directive #2: secrets={"HF_TOKEN": <actual value>}.
    # Per directive #5: explicit timeout.
    # Per directive #6: flavor as configured.
    job_info = api.run_uv_job(
        script=str(plan.script_path),
        script_args=plan.script_args,
        flavor=plan.flavor,
        timeout=plan.timeout_seconds,
        secrets={"HF_TOKEN": token},
        env={"PYTHONUNBUFFERED": "1"},
    )
    hf_jobs_id = job_info.id  # per directive #3

    ledger_row: dict[str, Any] | None = None
    if register_in_ledger:
        from tac.deploy.hf_jobs.job_id_ledger import (
            register_dispatched_hf_jobs_id,
        )

        ledger_row = register_dispatched_hf_jobs_id(
            hf_jobs_id=hf_jobs_id,
            lane_id=plan.lane_id,
            label=plan.label,
            flavor=plan.flavor,
            expected_cost_usd=plan.estimated_cost_usd,
            expected_axis=expected_axis,
            recipe=str(plan.recipe_path) if plan.recipe_path else None,
            max_seconds=plan.timeout_seconds,
            mounted_code_git_head=_git_head_sha(REPO_ROOT),
            agent="claude",
            subagent_id=subagent_id,
            session_id=session_id,
            hub_model_repo=plan.hub_model_repo,
            hub_dataset_repo=plan.hub_dataset_repo,
            hub_dataset_sha=hub_dataset_sha,
            upstream_snapshot_sha256=upstream_snapshot_sha256,
        )

    return {
        "hf_jobs_id": hf_jobs_id,
        "ledger_row": ledger_row,
        "plan": plan.to_dict(),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dispatch_hf_jobs_vision_training",
        description=(
            "Dispatch a vision-training HF Jobs run + register in the canonical "
            "ledger (Catalog #342 sister of Catalog #245)."
        ),
    )
    parser.add_argument("--script", type=Path, required=True,
                        help="Path to the canonical HF Jobs training script.")
    parser.add_argument("--hub-dataset-repo", type=str, required=True,
                        help="HF Hub source dataset repo id.")
    parser.add_argument("--hub-model-repo", type=str, required=True,
                        help="HF Hub destination model repo id.")
    parser.add_argument("--model", type=str,
                        default="timm/mobilenetv3_small_100.lamb_in1k",
                        help="Base model identifier (default: mobilenetv3_small).")
    parser.add_argument("--flavor", type=str, default="t4-small",
                        choices=sorted(HF_JOBS_FLAVOR_COSTS_USD_PER_HOUR.keys()),
                        help="HF Jobs hardware flavor.")
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=14400,
                        help="Job timeout (default: 14400 = 4h).")
    parser.add_argument("--lane-id", type=str,
                        default="lane_hf_jobs_segnet_surrogate_distillation_20260519")
    parser.add_argument("--label", type=str,
                        default="hf_jobs_segnet_surrogate_distillation_first_smoke")
    parser.add_argument("--recipe", type=Path, default=None,
                        help="Operator-authorize recipe path.")
    parser.add_argument("--hub-dataset-sha", type=str, default=None,
                        help="HF Hub source dataset commit sha (provenance pin).")
    parser.add_argument("--expected-axis", type=str, default="cuda",
                        choices=("cuda", "cpu", "advisory"))
    parser.add_argument("--upstream-snapshot-sha256", type=str, default=None)
    parser.add_argument("--subagent-id", type=str, default=None)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Emit plan JSON only; do NOT dispatch.")
    parser.add_argument("--extra-script-arg", action="append", default=[],
                        help="Append extra arg to the training script CLI (repeatable).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    plan = plan_dispatch(
        script=args.script,
        hub_dataset_repo=args.hub_dataset_repo,
        hub_model_repo=args.hub_model_repo,
        model=args.model,
        flavor=args.flavor,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        timeout_seconds=args.timeout_seconds,
        lane_id=args.lane_id,
        label=args.label,
        recipe_path=args.recipe,
        extra_script_args=args.extra_script_arg,
    )

    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan.to_dict()}, indent=2, sort_keys=True))
        return 0

    try:
        result = dispatch(
            plan,
            subagent_id=args.subagent_id,
            session_id=args.session_id,
            hub_dataset_sha=args.hub_dataset_sha,
            expected_axis=args.expected_axis,
            upstream_snapshot_sha256=args.upstream_snapshot_sha256,
        )
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc), "plan": plan.to_dict()}, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
