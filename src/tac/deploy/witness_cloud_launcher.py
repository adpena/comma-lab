# SPDX-License-Identifier: MIT
"""Provider-neutral, plan-first launcher contract for V9 CGauge CUDA training.

This module is deliberately a *planning* surface.  It does not import Modal or
perform provider I/O; execution remains an explicit, separately verified CLI
transaction in :mod:`tools.launch_witness_cloud`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from tac.deploy.provider_contracts import PROVIDER_CONTRACTS
from tac.witness_training_contract import cuda_v9_port_receipt

LANE_ID = "lane_cloud_launcher_v9_cgauge_cuda_438_20260711"
REMOTE_DRIVER = "scripts/remote_v9_cgauge_cuda.sh"
TRAINER = "experiments/train_levelset_witness_realized_through_R_torch.py"
V9_DSL = "src/tac/witness_dsl/spec_v9_cgauge.py"
CUDA_CONTROLLER = "src/tac/cuda_v9_controller_runtime.py"
CUDA_MODEL_RUNTIME = "src/tac/cuda_v9_island_runtime.py"
CUDA_THROUGHPUT = "src/tac/cuda_v9_throughput.py"
SEGNET_WEIGHTS = "upstream/models/segnet.safetensors"
POSENET_WEIGHTS = "upstream/models/posenet.safetensors"
REVIEWED_SENTINELS = (
    REMOTE_DRIVER,
    TRAINER,
    V9_DSL,
    CUDA_CONTROLLER,
    CUDA_MODEL_RUNTIME,
    CUDA_THROUGHPUT,
    SEGNET_WEIGHTS,
    POSENET_WEIGHTS,
)
# MUST match modal_train_lane.py RESULTS_VOL (mounted at /modal_results).
RESULTS_VOLUME = "comma-train-lane-results"
TASK438_GT_CACHE_SHA256 = (
    "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
)
TASK438_GT_CACHE_BYTES = 5_078_017_610
TYPED_EPOCH_HORIZON = 3000
# Maximum explicit runtime stop (1..3). ``stop_after_epochs=None`` selects
# timeout-stop mode: the trainer runs toward the typed horizon and the
# 1500-second child timeout is the stop. Timeout-stop is the canonical #438
# smoke shape (operator 2026-07-15: a 3-warmup-epoch cutoff is a toy-regime
# measurement; the smoke must reach + measure the expensive post-event regime).
DEFAULT_STOP_AFTER_EPOCHS = 3
# Named typed-DSL configs the remote driver may compile. smoke_regime is the
# #438 cost-regime variant (post-event backstops forced to 1): NON-PROMOTABLE
# timing evidence, never a science arm.
V9_DSL_CONFIGS = ("v9_cgauge_432_launch", "v9_cgauge_432_smoke_regime")
DEFAULT_DSL_CONFIG = "v9_cgauge_432_launch"
CHILD_TIMEOUT_SECONDS = 1500
INVOCATION_TIMEOUT_SECONDS = 1800
# Canonical Modal GPU string (per https://modal.com/docs/guide/gpu). We use the
# plain "H100" — NOT the "H100!" strict-suffix — because (1) Modal's docs state
# the automatic H100->H200 upgrade "does not change the cost", so the strict `!`
# (which only suppresses that free upgrade) buys no money-safety, and (2) the
# `!` variant empirically failed to attach a GPU through .with_options(gpu=...)
# on modal 1.5.1 (2026-07-12: GPU dispatch stage ran GPU-less -> rc=13). Money
# safety is enforced by the cost ceiling + timeout guards, not the GPU string.
EXACT_H100_GPU = "H100"
H100_GPU_USD_PER_HOUR = 5.00
CPU_CORES = 4
CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_GIB = 32
MEMORY_USD_PER_GIB_SECOND = 0.00000222
BUDGETED_IMAGE_STAGING_ALLOWANCE_USD = 0.50
CPU_PREFLIGHT_SECONDS = 600
MAX_PLAN_COST_USD = 5.00
MAX_LABEL_LENGTH = 63
ASSET_REUSE_DEFER = (
    "DEFER: plan-only mode never reads Modal cache state; explicit execution "
    "uses an exact SHA-addressed GT name-and-size lookup and, for resume, an "
    "exact regular nonzero checkpoint-presence lookup; bounded same-image CPU "
    "preflight rehashes remote bytes before an H100! claim. Missing GT assets "
    "must be staged under separate explicit operator authorization; execute "
    "never runs asset_stage_argv or performs an implicit volume upload."
)
LABEL_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
GIT_HEAD_RE = re.compile(r"[0-9a-f]{40}\Z")

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
    source_git_head: str
    label: str
    gpu: str
    local_gt_cache: str
    gt_cache_sha256: str | None
    gt_cache_bytes: int
    segnet_sha256: str | None
    posenet_sha256: str | None
    remote_gt_cache: str | None
    remote_out_dir: str
    resume_from: str | None
    resume_sha256: str | None
    epochs: int
    num_pairs: int
    dsl_config: str
    stop_after_epochs: int | None
    child_timeout_seconds: int
    invocation_timeout_seconds: int
    gpu_usd_per_hour_ceiling: float
    cpu_cores: int
    cpu_usd_per_core_second: float
    memory_gib: int
    memory_usd_per_gib_second: float
    budgeted_image_staging_allowance_usd: float
    planned_total_cost_usd: float
    max_plan_cost_usd: float
    cost_assumptions_as_of: str
    environment: dict[str, str]
    asset_stage_argv: tuple[str, ...]
    dispatch_argv: tuple[str, ...]
    harvest_argv: tuple[str, ...]
    reviewed_sentinels: tuple[str, ...]
    setup_blockers: tuple[str, ...]
    deferred_egress_blockers: tuple[str, ...]
    execution_allowed: bool
    authority: str
    plan_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _hash_payload(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_modal_custody_path(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    raw = value
    if raw != raw.strip():
        raise ValueError(f"{name} must not contain leading or trailing whitespace")
    if not raw or not raw.startswith("/modal_results/"):
        raise ValueError(f"{name} must be a normalized /modal_results/** custody path")
    if any(char.isspace() or ord(char) < 32 for char in raw) or any(
        char in raw for char in ",="
    ):
        raise ValueError(f"{name} contains characters unsafe for env overrides")
    normalized = str(PurePosixPath(raw))
    if normalized != raw or any(part in {".", ".."} for part in PurePosixPath(raw).parts):
        raise ValueError(f"{name} must be a normalized POSIX path without traversal")
    return raw


def build_plan(
    *,
    provider: str,
    source_git_head: str,
    gt_cache: str,
    label: str,
    gpu: str,
    epochs: int,
    num_pairs: int,
    resume_from: str | None = None,
    resume_sha256: str | None = None,
    gt_cache_sha256: str | None = None,
    gt_cache_bytes: int = TASK438_GT_CACHE_BYTES,
    segnet_sha256: str | None = None,
    posenet_sha256: str | None = None,
    dsl_config: str = DEFAULT_DSL_CONFIG,
    stop_after_epochs: int | None = None,
    child_timeout_seconds: float = CHILD_TIMEOUT_SECONDS,
    invocation_timeout_seconds: float = INVOCATION_TIMEOUT_SECONDS,
    gpu_usd_per_hour_ceiling: float = H100_GPU_USD_PER_HOUR,
    budgeted_image_staging_allowance_usd: float = BUDGETED_IMAGE_STAGING_ALLOWANCE_USD,
    max_plan_cost_usd: float = MAX_PLAN_COST_USD,
) -> WitnessCloudPlan:
    """Compile a deterministic plan without contacting any provider.

    The V9 timing smoke intentionally preserves its 3000-epoch typed DSL
    horizon. ``stop_after_epochs`` is runtime control-plane state and is kept
    out of that scientific config/hash; ``None`` (the default) selects
    timeout-stop mode where the 1500-second child timeout is the stop.
    ``dsl_config`` selects the named typed factory (scientific identity lives
    in the factory's typed hash, not in this plan field).
    """
    if provider not in {"modal", "aws", "gcp"}:
        raise ValueError("provider must be one of modal, aws, gcp")
    if not GIT_HEAD_RE.fullmatch(source_git_head):
        raise ValueError("source_git_head must be an exact lowercase 40-character git SHA")
    if gpu != EXACT_H100_GPU:
        raise ValueError(
            "V9 CGauge timing smoke requires the canonical H100 GPU string"
        )
    if not LABEL_RE.fullmatch(label):
        raise ValueError("label must be a lowercase hyphenated slug")
    if len(label) > MAX_LABEL_LENGTH:
        raise ValueError(f"label must be at most {MAX_LABEL_LENGTH} characters")
    if epochs != TYPED_EPOCH_HORIZON:
        raise ValueError(f"epochs must preserve the typed horizon of {TYPED_EPOCH_HORIZON}")
    if num_pairs != 600:
        raise ValueError("num_pairs must equal the fixed 600-pair Task438 smoke custody")
    if not isinstance(gt_cache_bytes, int) or gt_cache_bytes <= 0:
        raise ValueError("gt_cache_bytes must be a positive integer")
    if dsl_config not in V9_DSL_CONFIGS:
        raise ValueError(
            f"dsl_config must be one of {V9_DSL_CONFIGS}; got {dsl_config!r}"
        )
    if stop_after_epochs is not None and not (
        1 <= stop_after_epochs <= DEFAULT_STOP_AFTER_EPOCHS
    ):
        raise ValueError("stop_after_epochs must be None (timeout-stop) or within 1..3")
    for name, value in (
        ("child_timeout_seconds", child_timeout_seconds),
        ("invocation_timeout_seconds", invocation_timeout_seconds),
        ("gpu_usd_per_hour_ceiling", gpu_usd_per_hour_ceiling),
        ("budgeted_image_staging_allowance_usd", budgeted_image_staging_allowance_usd),
        ("max_plan_cost_usd", max_plan_cost_usd),
    ):
        _validate_positive_finite(name, value)
    if child_timeout_seconds != CHILD_TIMEOUT_SECONDS:
        raise ValueError(f"child_timeout_seconds must equal {CHILD_TIMEOUT_SECONDS}")
    if invocation_timeout_seconds != INVOCATION_TIMEOUT_SECONDS:
        raise ValueError(f"invocation_timeout_seconds must equal {INVOCATION_TIMEOUT_SECONDS}")
    if gpu_usd_per_hour_ceiling != H100_GPU_USD_PER_HOUR:
        raise ValueError(
            f"gpu_usd_per_hour_ceiling must retain the conservative ${H100_GPU_USD_PER_HOUR:.2f}/hour"
        )
    if budgeted_image_staging_allowance_usd != BUDGETED_IMAGE_STAGING_ALLOWANCE_USD:
        raise ValueError(
            "budgeted_image_staging_allowance_usd must retain the "
            f"${BUDGETED_IMAGE_STAGING_ALLOWANCE_USD:.2f} budgeted allowance"
        )
    if max_plan_cost_usd > MAX_PLAN_COST_USD:
        raise ValueError(f"max_plan_cost_usd must not exceed ${MAX_PLAN_COST_USD:.2f}")
    resume_from = _validate_modal_custody_path("resume_from", resume_from)
    for name, value in (
        ("resume_sha256", resume_sha256),
        ("gt_cache_sha256", gt_cache_sha256),
        ("segnet_sha256", segnet_sha256),
        ("posenet_sha256", posenet_sha256),
    ):
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"{name} must be a 64-character hexadecimal digest")
    if (resume_from is None) != (resume_sha256 is None):
        raise ValueError("resume_sha256 is required iff resume_from is supplied")
    resume_digest = resume_sha256.lower() if resume_sha256 else None
    digest = gt_cache_sha256.lower() if gt_cache_sha256 else None
    segnet_digest = segnet_sha256.lower() if segnet_sha256 else None
    posenet_digest = posenet_sha256.lower() if posenet_sha256 else None
    contract = PROVIDER_CONTRACTS[provider]
    port_receipt = cuda_v9_port_receipt()
    remote_gt = f"/modal_results/assets/v9_cgauge/gt_{digest}.npz" if digest else None
    remote_out = f"/modal_results/{label}/output"
    planned_total_cost = (
        gpu_usd_per_hour_ceiling * INVOCATION_TIMEOUT_SECONDS / 3600
        + CPU_CORES * CPU_USD_PER_CORE_SECOND * INVOCATION_TIMEOUT_SECONDS
        + MEMORY_GIB * MEMORY_USD_PER_GIB_SECOND * INVOCATION_TIMEOUT_SECONDS
        + CPU_CORES * CPU_USD_PER_CORE_SECOND * CPU_PREFLIGHT_SECONDS
        + MEMORY_GIB * MEMORY_USD_PER_GIB_SECOND * CPU_PREFLIGHT_SECONDS
        + budgeted_image_staging_allowance_usd
    )
    if planned_total_cost > max_plan_cost_usd:
        raise ValueError(
            f"planned total ceiling ${planned_total_cost:.6f} exceeds plan maximum "
            f"${max_plan_cost_usd:.2f}"
        )
    env = dict(CUDA_ENV)
    env.update(
        {
            "WITNESS_GT_CACHE": remote_gt or "",
            "WITNESS_OUT_DIR": remote_out,
            "WITNESS_EPOCHS": str(epochs),
            "WITNESS_DSL_CONFIG": dsl_config,
            "WITNESS_STOP_AFTER_EPOCHS": (
                "" if stop_after_epochs is None else str(stop_after_epochs)
            ),
            "WITNESS_CHILD_TIMEOUT_SECONDS": str(CHILD_TIMEOUT_SECONDS),
            "WITNESS_NUM_PAIRS": str(num_pairs),
            "WITNESS_RESUME_FROM": resume_from or "",
            "WITNESS_RESUME_SHA256": resume_digest or "",
            "WITNESS_GT_CACHE_SHA256": digest or "",
            "WITNESS_SEGNET_SHA256": segnet_digest or "",
            "WITNESS_POSENET_SHA256": posenet_digest or "",
        }
    )
    if provider == "modal" and digest and segnet_digest and posenet_digest:
        asset = (
            ".venv/bin/modal", "volume", "put", RESULTS_VOLUME,
            str(Path(gt_cache)), f"assets/v9_cgauge/gt_{digest}.npz",
        )
        overrides = ",".join(f"{key}={value}" for key, value in sorted(env.items()))
        dispatch = (
            ".venv/bin/modal", "run", "--detach", "experiments/modal_train_lane.py",
            "--lane-script", REMOTE_DRIVER,
            "--label", label,
            "--gpu", gpu,
            "--timeout-hours", f"{CHILD_TIMEOUT_SECONDS / 3600:.9f}",
            "--lane-id", LANE_ID,
            "--expected-mounted-code-git-head", source_git_head,
            # This is the reviewed *total plan* ceiling: the detached GPU
            # invocation, bounded same-image CPU preflight, and image/staging
            # allowance.  modal_train_lane persists it against the call ID.
            "--expected-cost-usd", f"{planned_total_cost:.6f}",
            "--trainer-module-path", TRAINER,
            "--require-clean-head",
            "--preflight-first",
            "--sentinel-files", ",".join(REVIEWED_SENTINELS),
            "--env-overrides", overrides,
        )
        harvest = (
            ".venv/bin/python", "tools/harvest_modal_calls.py",
            "--from-ledger", "--call-id", "<call-id-from-spawn>", "--execute",
        )
    else:
        asset = ()
        dispatch = ()
        harvest = ()
    custody_blockers = tuple(
        message
        for supplied, message in (
            (digest, "GT cache SHA-256 custody value is not supplied"),
            (segnet_digest, "SegNet SHA-256 custody value is not supplied"),
            (posenet_digest, "PoseNet SHA-256 custody value is not supplied"),
        )
        if not supplied
    )
    port_blockers = port_receipt["blockers"]
    if not isinstance(port_blockers, list) or not all(
        isinstance(item, str) for item in port_blockers
    ):
        raise ValueError("CUDA port receipt blockers must be a list of strings")
    blockers = (
        tuple(contract.setup_blockers)
        + tuple(f"CUDA port: {item}" for item in port_blockers)
        + custody_blockers
    )
    base: dict[str, object] = {
        "schema": "witness_cloud_plan.v8",
        "provider": provider,
        "status": contract.status,
        "lane_id": LANE_ID,
        "source_git_head": source_git_head,
        "label": label,
        "gpu": gpu,
        "local_gt_cache": str(Path(gt_cache)),
        "gt_cache_sha256": digest,
        "gt_cache_bytes": gt_cache_bytes,
        "segnet_sha256": segnet_digest,
        "posenet_sha256": posenet_digest,
        "remote_gt_cache": remote_gt,
        "remote_out_dir": remote_out,
        "resume_from": resume_from,
        "resume_sha256": resume_digest,
        "epochs": epochs,
        "num_pairs": num_pairs,
        "dsl_config": dsl_config,
        "stop_after_epochs": stop_after_epochs,
        "child_timeout_seconds": CHILD_TIMEOUT_SECONDS,
        "invocation_timeout_seconds": INVOCATION_TIMEOUT_SECONDS,
        "gpu_usd_per_hour_ceiling": gpu_usd_per_hour_ceiling,
        "cpu_cores": CPU_CORES,
        "cpu_usd_per_core_second": CPU_USD_PER_CORE_SECOND,
        "memory_gib": MEMORY_GIB,
        "memory_usd_per_gib_second": MEMORY_USD_PER_GIB_SECOND,
        "budgeted_image_staging_allowance_usd": budgeted_image_staging_allowance_usd,
        "planned_total_cost_usd": planned_total_cost,
        "max_plan_cost_usd": max_plan_cost_usd,
        "cost_assumptions_as_of": (
            "2026-07-12 static Modal pricing assumptions plus a $0.50 budgeted "
            "image/staging allowance; the allowance is not provider-enforced or measured"
        ),
        "environment": env,
        "asset_stage_argv": asset,
        "dispatch_argv": dispatch,
        "harvest_argv": harvest,
        "reviewed_sentinels": REVIEWED_SENTINELS,
        "setup_blockers": blockers,
        "deferred_egress_blockers": (ASSET_REUSE_DEFER,),
        "execution_allowed": (
            bool(digest and segnet_digest and posenet_digest)
            and contract.implemented
            and provider == "modal"
            and port_receipt["status"] == "COMPLETE_1_TO_1"
        ),
        "authority": "[contest-CUDA training-advisory] NON-PROMOTABLE",
    }
    return WitnessCloudPlan(**base, plan_sha256=_hash_payload(base))


def render_command(argv: tuple[str, ...]) -> str:
    return shlex.join(argv) if argv else "BLOCKED: custody or provider lifecycle is incomplete"


__all__ = [
    "ASSET_REUSE_DEFER",
    "CHILD_TIMEOUT_SECONDS",
    "CPU_PREFLIGHT_SECONDS",
    "CUDA_ENV",
    "DEFAULT_DSL_CONFIG",
    "DEFAULT_STOP_AFTER_EPOCHS",
    "EXACT_H100_GPU",
    "H100_GPU_USD_PER_HOUR",
    "INVOCATION_TIMEOUT_SECONDS",
    "LANE_ID",
    "MAX_LABEL_LENGTH",
    "POSENET_WEIGHTS",
    "REVIEWED_SENTINELS",
    "SEGNET_WEIGHTS",
    "TASK438_GT_CACHE_BYTES",
    "TASK438_GT_CACHE_SHA256",
    "TYPED_EPOCH_HORIZON",
    "V9_DSL_CONFIGS",
    "WitnessCloudPlan",
    "build_plan", "render_command",
]
