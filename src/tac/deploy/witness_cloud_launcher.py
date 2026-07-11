# SPDX-License-Identifier: MIT
"""Provider-neutral, plan-first launcher contract for V9 CGauge CUDA training."""
from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

from tac.deploy.provider_contracts import PROVIDER_CONTRACTS
from tac.witness_training_contract import cuda_v9_port_receipt

LANE_ID = "lane_cloud_launcher_v9_cgauge_cuda_438_20260711"
REMOTE_DRIVER = "scripts/remote_v9_cgauge_cuda.sh"
TRAINER = "experiments/train_levelset_witness_realized_through_R_torch.py"
RESULTS_VOLUME = "pact-training-results"
CUDA_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "DALI_DISABLE_NVML": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONHASHSEED": "0",
    "WITNESS_TRAINER_MODE": "full",
}


@dataclass(frozen=True)
class WitnessCloudPlan:
    schema: str
    provider: str
    status: str
    lane_id: str
    label: str
    gpu: str
    local_gt_cache: str
    gt_cache_sha256: str | None
    remote_gt_cache: str
    remote_out_dir: str
    resume_from: str | None
    epochs: int
    num_pairs: int
    environment: dict[str, str]
    asset_stage_argv: tuple[str, ...]
    dispatch_argv: tuple[str, ...]
    harvest_argv: tuple[str, ...]
    setup_blockers: tuple[str, ...]
    execution_allowed: bool
    authority: str
    pointer: dict[str, object]
    plan_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _hash_payload(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_plan(
    *,
    provider: str,
    gt_cache: str,
    label: str,
    gpu: str,
    epochs: int,
    num_pairs: int,
    resume_from: str | None = None,
    gt_cache_sha256: str | None = None,
) -> WitnessCloudPlan:
    """Compile a deterministic plan without contacting any provider."""
    if provider not in {"modal", "aws", "gcp"}:
        raise ValueError("provider must be one of modal, aws, gcp")
    if epochs < 1 or num_pairs < 1:
        raise ValueError("epochs and num_pairs must be positive")
    if gt_cache_sha256 is not None and (
        len(gt_cache_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in gt_cache_sha256.lower())
    ):
        raise ValueError("gt_cache_sha256 must be a 64-character hexadecimal digest")
    contract = PROVIDER_CONTRACTS[provider]
    port_receipt = cuda_v9_port_receipt()
    remote_gt = "/modal_results/assets/v9_cgauge/gt_n600.npz"
    remote_out = f"/modal_results/{label}/output"
    env = dict(CUDA_ENV)
    env.update(
        {
            "WITNESS_GT_CACHE": remote_gt,
            "WITNESS_OUT_DIR": remote_out,
            "WITNESS_EPOCHS": str(epochs),
            "WITNESS_NUM_PAIRS": str(num_pairs),
            "WITNESS_RESUME_FROM": resume_from or "",
            "WITNESS_GT_CACHE_SHA256": gt_cache_sha256 or "",
        }
    )
    if provider == "modal":
        asset = (
            ".venv/bin/modal", "volume", "put", RESULTS_VOLUME,
            str(Path(gt_cache)), "assets/v9_cgauge/gt_n600.npz",
        )
        overrides = ",".join(f"{key}={value}" for key, value in sorted(env.items()))
        dispatch = (
            ".venv/bin/modal", "run", "--detach", "experiments/modal_train_lane.py",
            "--lane-script", REMOTE_DRIVER,
            "--label", label,
            "--gpu", gpu,
            "--timeout-hours", "10",
            "--lane-id", LANE_ID,
            "--trainer-module-path", TRAINER,
            "--env-overrides", overrides,
        )
        harvest = (
            ".venv/bin/python", "tools/harvest_modal_calls.py",
            "--from-ledger", "--execute",
        )
    else:
        # AWS/GCP remain honest lifecycle scaffolds.  They share the complete
        # portable recipe but cannot execute until their provider contract is
        # promoted from scaffold to implemented.
        asset = ()
        dispatch = ()
        harvest = ()
    base: dict[str, object] = {
        "schema": "witness_cloud_plan.v1",
        "provider": provider,
        "status": contract.status,
        "lane_id": LANE_ID,
        "label": label,
        "gpu": gpu,
        "local_gt_cache": str(Path(gt_cache)),
        "gt_cache_sha256": gt_cache_sha256,
        "remote_gt_cache": remote_gt,
        "remote_out_dir": remote_out,
        "resume_from": resume_from,
        "epochs": epochs,
        "num_pairs": num_pairs,
        "environment": env,
        "asset_stage_argv": asset,
        "dispatch_argv": dispatch,
        "harvest_argv": harvest,
        "setup_blockers": tuple(contract.setup_blockers) + tuple(
            f"CUDA port: {item}" for item in port_receipt["blockers"]
        ) + (
            () if gt_cache_sha256 else ("GT cache SHA-256 custody value is not supplied",)
        ),
        "execution_allowed": (
            contract.implemented
            and provider == "modal"
            and port_receipt["status"] == "COMPLETE_1_TO_1"
        ),
        "authority": "[contest-CUDA training-advisory] NON-PROMOTABLE",
        "pointer": {"score": 0.19108282, "axis": "contest-CPU", "moved": False},
    }
    return WitnessCloudPlan(**base, plan_sha256=_hash_payload(base))


def render_command(argv: tuple[str, ...]) -> str:
    return shlex.join(argv) if argv else "BLOCKED: provider lifecycle is scaffold-only"


__all__ = ["CUDA_ENV", "LANE_ID", "WitnessCloudPlan", "build_plan", "render_command"]
