# SPDX-License-Identifier: MIT
"""Run a `scripts/remote_lane_*.sh` on Modal T4 / A10G — reliable training surface.

Why: 2026-04-29 night session showed Vast.ai 4090 NVDEC bad-host rate ≈ 85%.
~$5 burned across 5 dispatch rounds for 0 trained lanes. Modal is useful for
build/training work after repeated Vast.ai NVDEC host failures.
This wrapper is a training/build substrate only. The wrapper itself never
promotes a score. Lane-local auth-eval subprocesses are allowed only when the
trainer parses the auth-eval evidence contract and keeps promotion blocked
until the separate CPU/pack/compliance gates land.

Pattern mirrors `experiments/modal_auth_eval.py` (commit 11d56896).

USAGE — RECOMMENDED (`--detach` for unattended training):
    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \\
        experiments/modal_train_lane.py \\
        --lane-script scripts/remote_lane_omega_hessian_qat.sh \\
        --label lane_omega_hessian --gpu T4 --timeout-hours 10

Without `--detach` the local CLI blocks for the full 8-14h training duration
— terminal disconnect = lost job. With `--detach`, Modal keeps the run
alive and you can poll with `experiments/modal_recover_lane.py`, or stream logs
via `modal app logs <app-id>`. Round 12 caught this.

For PARALLEL dispatch of multiple lanes, fire 6 separate `modal run --detach`
in background (each gets its own container — Modal handles concurrency
natively).

Output: artifacts saved to `experiments/results/lane_<label>_modal/`. Any
score extracted from `contest_auth_eval.json` is labelled advisory/non-promotable
by the recovery helper when the recorded device is not CUDA.

Cross-references:
  - feedback_vastai_nvdec_roulette_pivot_to_modal_20260429
  - project_modal_pipeline_trusted_lane_g_v3_1_04_20260429
  - feedback_canonical_lane_lifecycle_DECISION_TREE_20260428
"""
from __future__ import annotations

from pathlib import Path

import modal

from tac.deploy.modal.mount_manifest import (
    build_training_image,
    collect_extra_mount_paths,
    collect_tier_required_input_files,
)
from tac.deploy.modal.runtime import (
    DALI_DISABLE_NVML_VALUE,
    PYTORCH_CUDA_ALLOC_CONF_VALUE,
)
from tac.deploy.witness_cloud_launcher import (
    BUDGETED_IMAGE_STAGING_ALLOWANCE_USD as V9_IMAGE_STAGING_ALLOWANCE_USD,
)
from tac.deploy.witness_cloud_launcher import (
    CHILD_TIMEOUT_SECONDS as V9_CHILD_TIMEOUT_SECONDS,
)
from tac.deploy.witness_cloud_launcher import (
    CPU_CORES as V9_CPU_CORES,
)
from tac.deploy.witness_cloud_launcher import (
    CPU_PREFLIGHT_SECONDS as V9_CPU_PREFLIGHT_SECONDS,
)
from tac.deploy.witness_cloud_launcher import (
    CPU_USD_PER_CORE_SECOND as V9_CPU_USD_PER_CORE_SECOND,
)
from tac.deploy.witness_cloud_launcher import (
    CUDA_ENV as V9_CUDA_ENV,
)
from tac.deploy.witness_cloud_launcher import (
    DEFAULT_STOP_AFTER_EPOCHS as V9_STOP_AFTER_EPOCHS,
)
from tac.deploy.witness_cloud_launcher import EXACT_H100_GPU as V9_EXACT_H100_GPU
from tac.deploy.witness_cloud_launcher import (
    H100_GPU_USD_PER_HOUR as V9_H100_GPU_USD_PER_HOUR,
)
from tac.deploy.witness_cloud_launcher import (
    INVOCATION_TIMEOUT_SECONDS as V9_INVOCATION_TIMEOUT_SECONDS,
)
from tac.deploy.witness_cloud_launcher import LABEL_RE as V9_LABEL_RE
from tac.deploy.witness_cloud_launcher import (
    LANE_ID as V9_LANE_ID,
)
from tac.deploy.witness_cloud_launcher import (
    MAX_LABEL_LENGTH as V9_MAX_LABEL_LENGTH,
)
from tac.deploy.witness_cloud_launcher import (
    MAX_PLAN_COST_USD as V9_MAX_PLAN_COST_USD,
)
from tac.deploy.witness_cloud_launcher import (
    MEMORY_GIB as V9_MEMORY_GIB,
)
from tac.deploy.witness_cloud_launcher import (
    MEMORY_USD_PER_GIB_SECOND as V9_MEMORY_USD_PER_GIB_SECOND,
)
from tac.deploy.witness_cloud_launcher import (
    REMOTE_DRIVER as V9_REMOTE_DRIVER,
)
from tac.deploy.witness_cloud_launcher import (
    REVIEWED_SENTINELS as V9_REVIEWED_SENTINELS,
)
from tac.deploy.witness_cloud_launcher import TRAINER as V9_TRAINER
from tac.deploy.witness_cloud_launcher import (
    TYPED_EPOCH_HORIZON as V9_TYPED_EPOCH_HORIZON,
)
from tac.deploy.witness_cloud_launcher import (
    V9_DSL_CONFIGS,
)
from tac.deploy.witness_cloud_launcher import (
    _validate_modal_custody_path as _validate_v9_modal_custody_path,
)
from tac.process_group_kill import run_in_process_group

app = modal.App("comma-train-lane")
RESULTS_VOL = "comma-train-lane-results"
REMOTE_PYTHONPATH = "/workspace/pact/src:/workspace/pact/upstream:/workspace/pact"
results_vol = modal.Volume.from_name(RESULTS_VOL, create_if_missing=False)
KNOWN_LANE_IDS = {
    "scripts/remote_lane_t1_balle_endtoend.sh": "t1_balle_128k_endtoend",
    "scripts/remote_lane_scpp_stage1.sh": "lane_scpp_stage1_smoke_anchor",
}
MODAL_TRAINING_ARTIFACT_EXTENSIONS = (
    ".bin",
    ".zip",
    ".pt",
    ".mkv",
    ".json",
    ".log",
    ".safetensors",
)
MODAL_STATIC_MAX_SECONDS = 14 * 3600
MODAL_FINALIZATION_GRACE_SECONDS = 300
MODAL_PREFLIGHT_HARD_TIMEOUT_SECONDS = 600
MODAL_PREFLIGHT_CHILD_TIMEOUT_SECONDS = 300
MODAL_V9_DEADLINE_TOLERANCE_SECONDS = 5
MODAL_GPU_CPU_REQUEST_LIMIT = (4.0, 4.0)
MODAL_GPU_MEMORY_REQUEST_LIMIT_MB = (32768, 32768)
# Conservative Modal list-price ceilings in USD/hour, rounded upward as of
# 2026-07-12. H100 is pinned to the canonical V9 launcher authority.
MODAL_GPU_USD_PER_HOUR_CEILINGS = {
    "T4": 0.75,
    "A10G": 1.50,
    "A100": 4.00,
    V9_EXACT_H100_GPU: V9_H100_GPU_USD_PER_HOUR,
}
V9_UNIQUE_ENV_KEYS = frozenset(
    {
        "WITNESS_GT_CACHE_SHA256",
        "WITNESS_SEGNET_SHA256",
        "WITNESS_POSENET_SHA256",
        "WITNESS_RESUME_SHA256",
        "WITNESS_STOP_AFTER_EPOCHS",
        "WITNESS_DSL_CONFIG",
    }
)


def _validated_modal_timeouts(timeout_hours: float) -> tuple[int, int]:
    """Return child and invocation timeouts, rejecting unsafe inputs."""

    import math

    if not math.isfinite(timeout_hours) or timeout_hours <= 0:
        raise ValueError("timeout_hours must be finite and positive")
    requested_seconds = int(timeout_hours * 3600)
    if requested_seconds <= 0:
        raise ValueError("timeout_hours is too small to represent one second")
    max_seconds = min(
        requested_seconds,
        MODAL_STATIC_MAX_SECONDS - MODAL_FINALIZATION_GRACE_SECONDS,
    )
    hard_timeout_seconds = min(
        max_seconds + MODAL_FINALIZATION_GRACE_SECONDS,
        MODAL_STATIC_MAX_SECONDS,
    )
    return max_seconds, hard_timeout_seconds


def _validate_expected_mounted_head(expected: str, current: str) -> None:
    """Require launcher-reviewed HEAD custody before any provider work."""

    import re

    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError("expected_mounted_code_git_head must be exactly 40 lowercase hex")
    if expected != current:
        raise ValueError(
            "expected_mounted_code_git_head does not match current git HEAD: "
            f"expected={expected} current={current}"
        )


def _validated_expected_cost_usd(
    expected_cost_usd: float,
    *,
    preflight_first: bool,
) -> float:
    """Validate the dispatch budget recorded in canonical call custody."""

    import math

    if not math.isfinite(expected_cost_usd) or expected_cost_usd <= 0:
        raise ValueError("provider dispatch requires finite expected_cost_usd > 0")
    if expected_cost_usd > V9_MAX_PLAN_COST_USD:
        raise ValueError(
            f"expected_cost_usd exceeds ${V9_MAX_PLAN_COST_USD:.2f}"
        )
    return float(expected_cost_usd)


def _validated_modal_gpu_request(
    gpu: str,
    *,
    preflight_first: bool,
) -> str | None:
    """Normalize a requested GPU and require the guarded preflight path."""

    if gpu in ("CPU", "cpu", "Cpu"):
        return None
    normalized = {
        "T4": "T4",
        "A10G": "A10G",
        "A10g": "A10G",
        "A100": "A100",
        "A100-40GB": "A100",
        "A100-80GB": "A100",
        V9_EXACT_H100_GPU: V9_EXACT_H100_GPU,
    }.get(gpu)
    if normalized is None:
        raise ValueError(
            f"unsupported gpu {gpu!r}; use CPU, T4, A10G, A100, or H100"
        )
    if not preflight_first:
        raise ValueError(
            "paid GPU dispatch requires preflight_first; named Modal endpoints "
            "are intentionally CPU-only"
        )
    return normalized


def _validated_provider_cost_ceiling(
    *,
    requested_gpu: str | None,
    hard_timeout_seconds: int,
    preflight_first: bool,
    expected_cost_usd: float,
) -> float:
    """Derive and validate the conservative provider resource ceiling."""

    import math

    if not isinstance(hard_timeout_seconds, int) or hard_timeout_seconds <= 0:
        raise ValueError("hard_timeout_seconds must be a positive integer")
    if requested_gpu is not None and requested_gpu not in MODAL_GPU_USD_PER_HOUR_CEILINGS:
        raise ValueError(f"unknown normalized GPU cost class {requested_gpu!r}")
    rates = (
        V9_CPU_USD_PER_CORE_SECOND,
        V9_MEMORY_USD_PER_GIB_SECOND,
        *MODAL_GPU_USD_PER_HOUR_CEILINGS.values(),
        V9_IMAGE_STAGING_ALLOWANCE_USD,
    )
    if any(not math.isfinite(rate) or rate < 0 for rate in rates):
        raise ValueError("provider cost rates must be finite and nonnegative")
    resource_ceiling = (
        V9_CPU_CORES * V9_CPU_USD_PER_CORE_SECOND * hard_timeout_seconds
        + V9_MEMORY_GIB * V9_MEMORY_USD_PER_GIB_SECOND * hard_timeout_seconds
    )
    if requested_gpu is not None:
        resource_ceiling += (
            MODAL_GPU_USD_PER_HOUR_CEILINGS[requested_gpu]
            * hard_timeout_seconds
            / 3600
        )
    if preflight_first:
        resource_ceiling += (
            V9_CPU_CORES * V9_CPU_USD_PER_CORE_SECOND * V9_CPU_PREFLIGHT_SECONDS
            + V9_MEMORY_GIB
            * V9_MEMORY_USD_PER_GIB_SECOND
            * V9_CPU_PREFLIGHT_SECONDS
            + V9_IMAGE_STAGING_ALLOWANCE_USD
        )
    if not math.isfinite(resource_ceiling):
        raise ValueError("derived provider resource ceiling is non-finite")
    if resource_ceiling > V9_MAX_PLAN_COST_USD + 1e-9:
        raise ValueError(
            f"derived provider resource ceiling ${resource_ceiling:.6f} exceeds "
            f"the canonical ${V9_MAX_PLAN_COST_USD:.2f} cap"
        )
    if expected_cost_usd + 1e-6 < resource_ceiling:
        raise ValueError(
            f"expected_cost_usd ${expected_cost_usd:.6f} is below derived "
            f"provider resource ceiling ${resource_ceiling:.6f}"
        )
    return resource_ceiling


def _parse_env_overrides(raw: str, *, strict: bool) -> dict[str, str]:
    """Parse comma env overrides, rejecting ambiguity for audited dispatches."""

    parsed: dict[str, str] = {}
    if not raw:
        return parsed
    for segment in raw.split(","):
        if "=" not in segment:
            if strict:
                raise ValueError(f"malformed env override segment {segment!r}")
            continue
        key, value = segment.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            if strict:
                raise ValueError("env override key must not be empty")
            continue
        if key in parsed and strict:
            raise ValueError(f"duplicate env override key {key!r}")
        parsed[key] = value
    return parsed


def _raw_env_override_keys(raw: str) -> set[str]:
    """Extract raw key intent without accepting the override syntax."""

    if not raw:
        return set()
    return {segment.split("=", 1)[0].strip() for segment in raw.split(",")}


def _classify_v9_trainer_path(
    trainer_module_path: str,
    *,
    repo_root: Path,
) -> bool:
    """Identify lexical or resolved references to the canonical V9 trainer."""

    import posixpath

    if not trainer_module_path:
        return False
    raw = trainer_module_path.strip()
    if posixpath.normpath(raw) == V9_TRAINER:
        return True
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        return candidate.resolve() == (repo_root / V9_TRAINER).resolve()
    except OSError:
        return False


def _lane_script_invokes_v9_trainer(
    lane_script: str,
    *,
    repo_root: Path,
) -> bool:
    """Detect wrappers whose source invokes the canonical V9 Torch trainer."""

    candidate = Path(lane_script)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        source = candidate.read_text(errors="ignore")
    except OSError:
        return False
    canonical = (repo_root / V9_TRAINER).resolve().as_posix()
    return V9_TRAINER in source or canonical in source


def _classify_v9_lane_script(
    lane_script: str,
    *,
    repo_root: Path | None = None,
) -> bool:
    """Identify aliases to V9 and require its reviewed canonical spelling."""

    import posixpath

    if not isinstance(lane_script, str):
        return False
    normalized = posixpath.normpath(lane_script)
    stripped_normalized = posixpath.normpath(lane_script.strip())
    lexical_v9_alias = (
        normalized == V9_REMOTE_DRIVER or stripped_normalized == V9_REMOTE_DRIVER
    )
    resolved_v9_alias = False
    candidate: Path | None = None
    if repo_root is not None:
        candidate = Path(lane_script)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        canonical = (repo_root / V9_REMOTE_DRIVER).resolve()
        try:
            resolved_v9_alias = candidate.resolve() == canonical
        except OSError:
            resolved_v9_alias = False
    if not lexical_v9_alias and not resolved_v9_alias:
        return False
    unsafe_character = any(char.isspace() or ord(char) < 32 for char in lane_script)
    if (
        lane_script != V9_REMOTE_DRIVER
        or unsafe_character
        or Path(lane_script).is_absolute()
        or normalized != lane_script
    ):
        raise ValueError(
            "V9 lane_script must use exact normalized repo-relative canonical path "
            f"{V9_REMOTE_DRIVER!r}"
        )
    if candidate is not None and candidate.is_symlink():
        raise ValueError("V9 lane_script must not be a symlink")
    return True


def _refuse_forced_modal_image_rebuild(
    env: dict[str, str] | None = None,
) -> None:
    """Refuse operator-forced image rebuilds outside the budgeted dispatch."""

    import os

    source = os.environ if env is None else env
    forced = [
        key
        for key in ("MODAL_FORCE_BUILD", "MODAL_IGNORE_CACHE")
        if source.get(key, "")
    ]
    if forced:
        raise RuntimeError(
            "refusing forced Modal image rebuild outside the dispatch cost cap: "
            f"nonempty environment controls={forced}"
        )


def _validate_v9_env_controls(*, label: str, overrides: dict[str, str]) -> None:
    """Validate the canonical V9 environment on local and remote boundaries."""

    import re

    if V9_LABEL_RE.fullmatch(label) is None:
        raise ValueError("V9 label must match canonical lowercase hyphen grammar")
    if len(label) > V9_MAX_LABEL_LENGTH:
        raise ValueError(f"V9 label must be at most {V9_MAX_LABEL_LENGTH} characters")

    required_env = {
        *V9_CUDA_ENV,
        "WITNESS_GT_CACHE",
        "WITNESS_OUT_DIR",
        "WITNESS_EPOCHS",
        "WITNESS_DSL_CONFIG",
        "WITNESS_TORCH_COMPILE_MODE",
        "WITNESS_STOP_AFTER_EPOCHS",
        "WITNESS_CHILD_TIMEOUT_SECONDS",
        "WITNESS_NUM_PAIRS",
        "WITNESS_RESUME_FROM",
        "WITNESS_RESUME_SHA256",
        "WITNESS_GT_CACHE_SHA256",
        "WITNESS_SEGNET_SHA256",
        "WITNESS_POSENET_SHA256",
    }
    if set(overrides) != required_env:
        raise ValueError(
            "V9 env overrides must exactly match required keys; "
            f"missing={sorted(required_env - set(overrides))} "
            f"extra={sorted(set(overrides) - required_env)}"
        )
    for key, expected in V9_CUDA_ENV.items():
        if overrides[key] != expected:
            raise ValueError(f"V9 deterministic env {key} must equal {expected!r}")
    fixed_values = {
        "WITNESS_TRAINER_MODE": "full",
        "WITNESS_EPOCHS": str(V9_TYPED_EPOCH_HORIZON),
        "WITNESS_CHILD_TIMEOUT_SECONDS": str(V9_CHILD_TIMEOUT_SECONDS),
        "WITNESS_NUM_PAIRS": "600",
    }
    for key, expected in fixed_values.items():
        if overrides[key] != expected:
            raise ValueError(f"V9 env {key} must equal {expected!r}")
    if overrides["WITNESS_DSL_CONFIG"] not in V9_DSL_CONFIGS:
        raise ValueError(
            f"V9 env WITNESS_DSL_CONFIG must be one of {sorted(V9_DSL_CONFIGS)}"
        )
    # Backend-only Inductor mode override; '' lets the trainer policy decide.
    # MEASURED 2026-07-15 (H100 r5, rc=124): max-autotune ate the whole
    # time-boxed window — the smoke launcher threads 'default'.
    if overrides["WITNESS_TORCH_COMPILE_MODE"] not in ("", "off", "default", "max-autotune"):
        raise ValueError(
            "V9 env WITNESS_TORCH_COMPILE_MODE must be '', 'off', 'default', "
            "or 'max-autotune'"
        )
    allowed_stop_values = {""} | {
        str(value) for value in range(1, V9_STOP_AFTER_EPOCHS + 1)
    }
    if overrides["WITNESS_STOP_AFTER_EPOCHS"] not in allowed_stop_values:
        raise ValueError(
            "V9 env WITNESS_STOP_AFTER_EPOCHS must be '' (timeout-stop) or one of "
            f"{sorted(value for value in allowed_stop_values if value)}"
        )
    for key in (
        "WITNESS_GT_CACHE_SHA256",
        "WITNESS_SEGNET_SHA256",
        "WITNESS_POSENET_SHA256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", overrides[key]) is None:
            raise ValueError(f"V9 env {key} must be a lowercase 64-hex digest")
    gt_sha = overrides["WITNESS_GT_CACHE_SHA256"]
    expected_gt_path = f"/modal_results/assets/v9_cgauge/gt_{gt_sha}.npz"
    if overrides["WITNESS_GT_CACHE"] != expected_gt_path:
        raise ValueError("V9 GT path must be SHA-addressed by WITNESS_GT_CACHE_SHA256")
    _validate_v9_modal_custody_path("WITNESS_GT_CACHE", overrides["WITNESS_GT_CACHE"])
    expected_out = f"/modal_results/{label}/output"
    if overrides["WITNESS_OUT_DIR"] != expected_out:
        raise ValueError("V9 output path must match the dispatch label custody path")
    _validate_v9_modal_custody_path("WITNESS_OUT_DIR", overrides["WITNESS_OUT_DIR"])
    resume_from = overrides["WITNESS_RESUME_FROM"]
    resume_sha256 = overrides["WITNESS_RESUME_SHA256"]
    if resume_from:
        _validate_v9_modal_custody_path("WITNESS_RESUME_FROM", resume_from)
        if re.fullmatch(r"[0-9a-f]{64}", resume_sha256) is None:
            raise ValueError(
                "V9 WITNESS_RESUME_SHA256 must be lowercase 64-hex when "
                "WITNESS_RESUME_FROM is nonempty"
            )
    elif resume_sha256:
        raise ValueError(
            "V9 WITNESS_RESUME_SHA256 must be empty when WITNESS_RESUME_FROM is empty"
        )


def _validate_v9_remote_h100_invocation(
    *,
    lane_script: str,
    label: str,
    env_overrides: dict[str, str],
    mounted_code_git_head: str,
    mounted_code_git_branch: str,
    sentinel_sha256_local: dict,
    max_seconds: int,
    billing_deadline_unix_s: float | None,
    invocation_started_unix_s: float,
    requested_gpu: str,
) -> None:
    """Block direct Modal Function calls that bypass the reviewed V9 main()."""

    import math
    import re

    is_canonical_v9_lane = _classify_v9_lane_script(
        lane_script,
        repo_root=Path("/workspace/pact"),
    )
    if not is_canonical_v9_lane:
        if requested_gpu in {"H100", V9_EXACT_H100_GPU}:
            raise ValueError(
                "H100 is reserved for the exact canonical Task438 V9 lane"
            )
        return
    if requested_gpu != V9_EXACT_H100_GPU:
        raise ValueError("remote V9 dynamic GPU must be exactly H100")
    if max_seconds != V9_CHILD_TIMEOUT_SECONDS:
        raise ValueError(
            f"remote V9 child timeout must be exactly {V9_CHILD_TIMEOUT_SECONDS}s"
        )
    if not isinstance(billing_deadline_unix_s, (int, float)) or not math.isfinite(
        billing_deadline_unix_s
    ):
        raise ValueError("remote V9 billing deadline must be finite and non-null")
    remaining = billing_deadline_unix_s - invocation_started_unix_s
    if remaining <= 0 or remaining > (
        V9_INVOCATION_TIMEOUT_SECONDS + MODAL_V9_DEADLINE_TOLERANCE_SECONDS
    ):
        raise ValueError(
            "remote V9 billing deadline must be positive and no later than "
            f"invocation start + {V9_INVOCATION_TIMEOUT_SECONDS}s "
            f"(+{MODAL_V9_DEADLINE_TOLERANCE_SECONDS}s serialization tolerance)"
        )
    if mounted_code_git_branch != "main":
        raise ValueError("remote V9 mounted branch must be exactly main")
    if re.fullmatch(r"[0-9a-f]{40}", mounted_code_git_head) is None:
        raise ValueError("remote V9 mounted HEAD must be lowercase 40-hex")
    if set(sentinel_sha256_local) != set(V9_REVIEWED_SENTINELS) or any(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
        for value in sentinel_sha256_local.values()
    ):
        raise ValueError("remote V9 sentinel SHA map must exactly cover reviewed sentinels")
    _validate_v9_env_controls(label=label, overrides=env_overrides)


def _derive_cpu_preflight_receipt(
    *,
    dispatch_nonce: str,
    lane_script: str,
    label: str,
    mounted_code_git_head: str,
) -> str:
    """Bind a GPU dispatch to the immediately preceding CPU preflight."""

    import hashlib

    payload = "\0".join(
        (
            "modal_cpu_preflight_receipt_v1",
            dispatch_nonce,
            lane_script,
            label,
            mounted_code_git_head,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _gpu_visible_in_container(environ) -> bool:
    """Detect an attached NVIDIA GPU inside a Modal container, fail-closed.

    MEASURED 2026-07-15 (gpu_probe_438, H100 via with_options(gpu="H100")):
    Modal GPU containers leave ``CUDA_VISIBLE_DEVICES`` UNSET, set
    ``NVIDIA_VISIBLE_DEVICES`` to the GPU UUID, and expose the device node at
    its HOST index (observed ``/dev/nvidia3``) — NOT necessarily
    ``/dev/nvidia0``. The pre-2026-07-15 detector (CUDA_VISIBLE_DEVICES or
    exactly /dev/nvidia0) therefore refused healthy H100 allocations with
    rc=13 whenever the assigned index was nonzero (both the 2026-07-12 and
    2026-07-15 "GPU dispatch stage has no visible GPU" failures; the 07-12
    "H100! string" diagnosis was wrong — the GPU was attached).

    False positives stay impossible-by-consequence: the driver hard-fails
    rc=68 on ``torch.cuda.is_available()`` before training, and the CPU
    stages REFUSE when this returns True — CPU containers carry none of
    these signals.
    """
    import glob

    cuda_env = environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_env not in {"", "-1", "NoDevFiles", "none", "None"}:
        return True
    nvidia_env = environ.get("NVIDIA_VISIBLE_DEVICES", "")
    if nvidia_env not in {"", "none", "None", "void"}:
        return True
    return bool(glob.glob("/dev/nvidia[0-9]*"))


def _validate_remote_dispatch_stage(
    *,
    dispatch_stage: str,
    dispatch_nonce: str,
    cpu_preflight_receipt: str,
    requested_gpu: str,
    lane_script: str,
    label: str,
    mounted_code_git_head: str,
    visible_gpu: bool,
) -> None:
    """Reject accidental named-endpoint reuse before source materialization."""

    import re

    if re.fullmatch(r"[0-9a-f]{64}", dispatch_nonce) is None:
        raise ValueError("dispatch_nonce must be lowercase 64-hex")
    if not label:
        raise ValueError("dispatch receipt label must be nonempty")
    if dispatch_stage == "cpu_preflight":
        if requested_gpu != "CPU" or cpu_preflight_receipt or visible_gpu:
            raise ValueError("CPU preflight must be receipt-free and GPU-free")
        return
    if dispatch_stage == "cpu_dispatch":
        if requested_gpu != "CPU" or cpu_preflight_receipt or visible_gpu:
            raise ValueError("CPU dispatch must be receipt-free and GPU-free")
        return
    if dispatch_stage != "gpu_dispatch":
        raise ValueError(f"unknown or omitted dispatch_stage {dispatch_stage!r}")
    if requested_gpu not in {"T4", "A10G", "A100", V9_EXACT_H100_GPU}:
        raise ValueError("GPU dispatch must declare a normalized requested_gpu")
    expected_receipt = _derive_cpu_preflight_receipt(
        dispatch_nonce=dispatch_nonce,
        lane_script=lane_script,
        label=label,
        mounted_code_git_head=mounted_code_git_head,
    )
    if cpu_preflight_receipt != expected_receipt:
        raise ValueError("GPU dispatch lacks its exact CPU-preflight receipt")
    if not visible_gpu:
        raise ValueError("GPU dispatch stage has no visible GPU")


def _validate_v9_direct_dispatch_policy(
    *,
    lane_script: str,
    lane_id: str,
    label: str,
    gpu: str,
    requested_gpu: str | None,
    mounted_code_git_branch: str,
    preflight_first: bool,
    require_clean_head: bool,
    max_seconds: int,
    hard_timeout_seconds: int,
    expected_cost_usd: float,
    sentinel_files: str,
    overrides: dict[str, str],
    trainer_module_path: str,
    repo_root: Path,
) -> None:
    """Fail closed when the reviewed V9 launcher is bypassed."""

    is_v9 = (
        _classify_v9_lane_script(lane_script, repo_root=repo_root)
        or lane_id == V9_LANE_ID
        or bool(set(overrides) & V9_UNIQUE_ENV_KEYS)
        or _classify_v9_trainer_path(trainer_module_path, repo_root=repo_root)
        or _lane_script_invokes_v9_trainer(lane_script, repo_root=repo_root)
    )
    if not is_v9:
        if requested_gpu in {"H100", V9_EXACT_H100_GPU}:
            raise ValueError(
                "H100 is reserved for the exact canonical Task438 V9 contract"
            )
        return
    if lane_script != V9_REMOTE_DRIVER or lane_id != V9_LANE_ID:
        raise ValueError("V9 driver and Task438 lane_id must be paired exactly")
    if trainer_module_path != V9_TRAINER:
        raise ValueError("V9 trainer_module_path must be the exact canonical Torch trainer")
    if gpu != V9_EXACT_H100_GPU or requested_gpu != V9_EXACT_H100_GPU:
        raise ValueError("V9 Modal dispatch requires raw and dynamic gpu exactly H100")
    if mounted_code_git_branch != "main":
        raise ValueError("V9 Modal dispatch requires mounted git branch exactly main")
    if not preflight_first:
        raise ValueError("V9 Modal dispatch requires preflight_first")
    if not require_clean_head:
        raise ValueError("V9 Modal dispatch requires require_clean_head")
    if max_seconds != V9_CHILD_TIMEOUT_SECONDS:
        raise ValueError(
            f"V9 child timeout must be exactly {V9_CHILD_TIMEOUT_SECONDS}s"
        )
    if hard_timeout_seconds != V9_INVOCATION_TIMEOUT_SECONDS:
        raise ValueError(
            "V9 invocation timeout must be exactly "
            f"{V9_INVOCATION_TIMEOUT_SECONDS}s"
        )
    sentinels = tuple(item.strip() for item in sentinel_files.split(",") if item.strip())
    if sentinels != V9_REVIEWED_SENTINELS:
        raise ValueError("V9 sentinel_files must exactly match the reviewed sentinel set")
    missing_sentinels = [
        rel
        for rel in V9_REVIEWED_SENTINELS
        if not (repo_root / rel).is_file() or (repo_root / rel).is_symlink()
    ]
    if missing_sentinels:
        raise ValueError(
            "V9 reviewed sentinels must exist as regular non-symlink files: "
            f"{missing_sentinels}"
        )
    _validate_v9_env_controls(label=label, overrides=overrides)
    resource_cost_floor = (
        V9_H100_GPU_USD_PER_HOUR * hard_timeout_seconds / 3600
        + V9_CPU_CORES * V9_CPU_USD_PER_CORE_SECOND * hard_timeout_seconds
        + V9_MEMORY_GIB * V9_MEMORY_USD_PER_GIB_SECOND * hard_timeout_seconds
        + V9_CPU_CORES * V9_CPU_USD_PER_CORE_SECOND * V9_CPU_PREFLIGHT_SECONDS
        + V9_MEMORY_GIB * V9_MEMORY_USD_PER_GIB_SECOND * V9_CPU_PREFLIGHT_SECONDS
        + V9_IMAGE_STAGING_ALLOWANCE_USD
    )
    if expected_cost_usd + 1e-6 < resource_cost_floor:
        raise ValueError(
            f"V9 expected_cost_usd ${expected_cost_usd:.6f} is below requested "
            f"resource/time ceiling ${resource_cost_floor:.6f}"
        )


def _cancel_spawned_call_best_effort(fn_call, *, reason: str) -> None:
    """Try to stop a spawned call whose canonical custody could not be sealed."""

    import sys

    try:
        fn_call.cancel(terminate_containers=True)
        print(
            f"[modal_train_lane] cancelled spawned call after {reason}",
            file=sys.stderr,
        )
    except Exception as exc:  # pragma: no cover - provider/SDK failure path
        print(
            "FATAL: best-effort cancellation of unregistered Modal call failed "
            f"after {reason}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _acquire_local_dispatch_guard(
    repo_root: Path,
    *,
    lane_id: str,
    label: str,
    lane_global: bool = False,
):
    """Serialize local preflight-to-registration for a dispatch identity."""

    import fcntl
    import hashlib

    identity = lane_id if lane_global else f"{lane_id}\0{label}"
    digest = hashlib.sha256(identity.encode()).hexdigest()
    guard_dir = repo_root / ".omx" / "state" / "modal_train_lane_dispatch_guards"
    guard_dir.mkdir(parents=True, exist_ok=True)
    guard_path = guard_dir / f"{digest}.lock"
    guard_fh = guard_path.open("a+")
    fcntl.flock(guard_fh.fileno(), fcntl.LOCK_EX)
    return guard_fh


def _release_local_dispatch_guard(guard_fh) -> None:
    """Release a local dispatch guard; process exit also releases it."""

    import fcntl

    try:
        fcntl.flock(guard_fh.fileno(), fcntl.LOCK_UN)
    finally:
        guard_fh.close()


def modal_training_artifact_relative_path(
    fp: Path,
    *,
    workspace: Path,
    volume_dir: Path,
) -> Path:
    """Return the recovery-tree path for a Modal training artifact."""

    try:
        return fp.relative_to(workspace)
    except ValueError:
        pass
    try:
        return fp.relative_to(volume_dir)
    except ValueError:
        return Path(fp.name)


def modal_training_artifact_should_collect(rel: Path) -> bool:
    """Return whether a Modal training artifact belongs in the recovery payload."""

    if rel.suffix.lower() in MODAL_TRAINING_ARTIFACT_EXTENSIONS:
        return True
    parts = rel.parts
    return len(parts) >= 3 and parts[0] == "output" and parts[1] == "submission"

# Image with all deps. ffmpeg-master (with in_primaries support) is pulled
# at build time via the same BtbN nightly that setup_full.sh uses on Vast.ai.
_refuse_forced_modal_image_rebuild()
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "unzip", "wget", "curl", "build-essential",
        "libgl1", "libglib2.0-0",  # opencv runtime
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "safetensors",
        "einops",
        "segmentation-models-pytorch",
        "av",
        "brotli",
        # constriction: Rust-backed range coder used by tac.codec.jscc (SE-4
        # JSCC), tac.pr101_*, tac.pr103_*, tac.pr86_hpac_codec,
        # tac.packet_compiler.*. Must match pyproject.toml's
        # entropy runtime deps. Added 2026-05-13 after PR95++ smoke crashed
        # at `import constriction` ModuleNotFoundError on Modal worker.
        "constriction>=0.4,<0.5",
        # NOTE: `pyppmd` (LGPL-2.1-or-later) was removed from the default
        # Modal worker image 2026-05-14 per OSS v0.2.0-rc1 BLOCKER B1 (lane
        # lane_pyppmd_to_constriction_migrate_20260514). Modal workers do
        # NOT run third-party PR86/PR91 replay (that is a local CPU forensic
        # path consumed by tac.pr86_hpac_codec / tac.pr91_hpm1_codec). If a
        # future Modal lane needs to decode hpac.pt.ppmd wire bytes, add
        # `"pyppmd>=1.3,<2.0"` back to the lane-specific image (NOT this
        # default image) and document the LGPL obligation in the lane
        # runbook. Default image is permissive-only.
        "click",
        "nvidia-dali-cuda120==1.52.0",
        "tqdm",
        "timm",
        "scipy",
        "numpy<2.0",
        "Pillow",
        "pydantic>=2.0",
        extra_index_url="https://pypi.nvidia.com",
    )
    # Install ffmpeg-master from BtbN nightly (has in_primaries + libsvtav1).
    # Mirrors `scripts/remote_setup_full.sh` Stage 6 EXACTLY — johnvansickle
    # builds did NOT have in_primaries (review of 2026-04-29 session showed
    # `ffmpeg-git-20240629` only had in_color_matrix, missing in_primaries).
    # The BtbN URL is canonical and includes a build-time verification that
    # the binary has both in_primaries (needed by inflate.sh:require_ffmpeg_parity)
    # AND libsvtav1 (needed by mask_codec).
    .run_commands(
        "curl -sL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -o /tmp/ffmpeg-master.tar.xz",
        "cd /opt && tar xf /tmp/ffmpeg-master.tar.xz",
        "ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg-master",
        "ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg-new",
        # Build-time gate: fail image build if in_primaries missing.
        "/usr/local/bin/ffmpeg-master -hide_banner -h filter=scale 2>&1 | grep -q in_primaries || (echo FATAL: ffmpeg-master lacks in_primaries; exit 1)",
        "/usr/local/bin/ffmpeg-master -encoders 2>&1 | grep -qi svtav1 || (echo FATAL: ffmpeg-master lacks libsvtav1; exit 1)",
        "rm /tmp/ffmpeg-master.tar.xz",
    )
    # uv (used by inflate.sh)
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
    )
)


# T1-A simplification 2026-05-12: discovery-based mount manifest replaces the
# hand-curated list. The canonical builder ALWAYS mounts the structural
# minimum (src / scripts / upstream / submissions / experiments / tools /
# pyproject.toml) and discovers `required_input_file=True` flags from any
# trainer module the caller names.
#
# Per Catalog #153 (`check_modal_dispatcher_uses_canonical_mount_builder`),
# this is the ONLY supported mount pattern for experiments/modal_*.py going
# forward. Previously the bug class was "list is missing entry X" — adding
# entries to hand-curated lists IS the bug class (see 2026-05-12 Modal A100
# dispatch fc-01KREJST89QHFRWJXHAKXD850C that crashed for exactly this
# reason).
#
# WAVE-3 (2026-05-16) DEEPER STRUCTURAL FIX per
# `.omx/research/stc_v2_driver_path_layer_fix_landed_20260516.md` Follow-on
# op-routable #4 + the STC v2 FIX subagent's CRITICAL DEEPER FINDING (commit
# 7dd8a5412): The Modal image is built at MODULE-LOAD time but the lane
# script (and therefore the canonical trainer module) is only known at
# DISPATCH time. The image cannot be rebuilt per-dispatch (Modal image
# caching + @app.function decorator constraints). Trainer-side
# TIER_1_EXTRA_MOUNT_PATHS declarations therefore CANNOT participate in the
# Modal image build for this generic dispatcher.
#
# CANONICAL FIX (Option 2 — payload staging): The dispatcher derives the
# trainer module path from the lane_script at main() time, reads each
# declared TIER_1_EXTRA_MOUNT_PATHS (+ MODAL_EXTRA_MOUNT_PATHS) file into
# bytes, and threads them as a `dict[rel_path, bytes]` arg through to
# `_run_lane_inner` which materializes them under `/tmp/pact/<rel>` on the
# worker after the structural mount copy. This:
#   - Honors trainer-side TIER_1_EXTRA_MOUNT_PATHS declarations structurally
#   - Works for files under experiments/results/** (the Modal-IGNORED subtree
#     per DEFAULT_RESULTS_IGNORE — exactly the STC v2 / a1_plus_lapose /
#     a1_plus_wavelet_residual anchor-archive class)
#   - Preserves Modal image caching (image is invariant per app)
#   - Mirrors the existing claim_ledger_bytes serialization pattern
#
# Cross-references:
#   - Catalog #152 Wave 1 + Wave 2 driver-path extension
#   - Catalog #153 canonical mount builder
#   - Catalog #166 Modal HEAD parity ledger
#   - HNeRV parity discipline L9 (Runtime closure)
#
# T2-C absorption: the previous `_RESULTS_MOUNTS` conditional (8 LOC of
# dead-conditional mount paths for `public_pr95_intake_20260504_codex` +
# `c067_fixed_renderer_burn_prep_20260503` that no longer exist on the
# working tree) is now correctly omitted by construction — operator-extra
# mounts must be declared, not silently absorbed from disk.
training_image = build_training_image(
    image.env({
        "PYTHONPATH": REMOTE_PYTHONPATH,
        # Modal containers can expose CUDA while denying NVML internals.
        # DALI's video reader works without NVML when this is set; without it,
        # upstream/evaluate.py can fail after successful inflate with
        # `nvml error (999)`.
        "DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE,
        "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF_VALUE,
    }),
    trainer_module_path=None,
    optional_files=(
        ".omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json",
    ),
)


def _substrate_id_from_lane_script(lane_script: str) -> str | None:
    """Return substrate id for canonical substrate lane scripts."""
    script_name = Path(lane_script).name
    prefix = "remote_lane_substrate_"
    if not script_name.startswith(prefix) or not script_name.endswith(".sh"):
        return None
    return script_name[len(prefix):-len(".sh")]


def _derive_trainer_module_path(lane_script: str, repo_root: Path) -> Path | None:
    """Derive canonical trainer module path from lane_script.

    Convention: ``scripts/remote_lane_substrate_<id>.sh`` →
    ``experiments/train_substrate_<id>.py`` (per CLAUDE.md "Subagent
    coherence-by-default" canonical mapping). Returns ``None`` if the lane
    script does not follow the substrate convention OR the inferred trainer
    file does not exist.

    WAVE-3 deeper structural fix per STC v2 FIX CRITICAL DEEPER FINDING
    (commit 7dd8a5412): the generic dispatcher cannot name a single trainer
    module at module-load time, but it CAN derive one at dispatch time from
    the lane_script.
    """
    substrate_id = _substrate_id_from_lane_script(lane_script)
    if substrate_id is None:
        return None
    candidate = repo_root / "experiments" / f"train_substrate_{substrate_id}.py"
    if not candidate.is_file():
        return None
    return candidate


def _normalize_trainer_module_path(
    trainer_module_path: str,
    repo_root: Path,
) -> Path | None:
    """Resolve an explicit trainer module path from an operator recipe."""

    text = trainer_module_path.strip()
    if not text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if not candidate.is_file():
        raise RuntimeError(
            f"explicit trainer module path does not exist: {trainer_module_path}"
        )
    return candidate.resolve()


def _collect_trainer_extra_mount_payload(
    trainer_module_path: Path | None,
    repo_root: Path,
    *,
    fail_on_import_error: bool = False,
    fail_on_missing_paths: bool = False,
) -> dict:
    """Read trainer-declared TIER_1_EXTRA_MOUNT_PATHS (+ MODAL_EXTRA_MOUNT_PATHS)
    into a ``{rel_path: bytes}`` dict suitable for serialization to the Modal
    worker.

    Returns an empty dict if:
    - ``trainer_module_path`` is None (legacy dispatchers / unmapped lane scripts)
    - the trainer module has no extra-mount declarations
    - the trainer module cannot be imported AND ``fail_on_import_error=False``
    - a trainer-declared path is missing AND ``fail_on_missing_paths=False``

    Skips paths under STRUCTURAL_MINIMUM_DIRS (already mounted) so we do not
    double-mount common files. Only reads files under
    ``experiments/results/**`` (the Modal-IGNORED subtree per
    ``DEFAULT_RESULTS_IGNORE``) and other paths outside the structural mount
    set.

    Per CLAUDE.md "Subagent coherence-by-default": this is the trainer-side
    contract enforcement at the dispatcher layer.
    """
    import sys as _sys

    payload: dict = {}
    if trainer_module_path is None:
        return payload

    def _import_with_duplicate_registry_fallback(path: Path):
        from tac.deploy.modal.mount_manifest import _import_trainer_module

        try:
            return _import_trainer_module(path)
        except Exception as exc:
            if "Duplicate substrate id" not in str(exc):
                raise

            # Trainer modules often self-register substrate contracts at
            # import time. In a warm operator process, importing the same
            # trainer only to read mount declarations can trip the registry's
            # duplicate-id guard. For this dispatcher introspection pass,
            # suppress registration and re-import so TIER_1_EXTRA_MOUNT_PATHS
            # and required_input_file defaults are still consumed.
            import tac.substrate_registry as _registry
            import tac.substrate_registry.decorator as _decorator

            original_public = _registry.register_substrate
            original_decorator = _decorator.register_substrate

            def _metadata_only_register(_contract):
                def _wrap(fn):
                    return fn

                return _wrap

            try:
                _registry.register_substrate = _metadata_only_register
                _decorator.register_substrate = _metadata_only_register
                return _import_trainer_module(path)
            finally:
                _registry.register_substrate = original_public
                _decorator.register_substrate = original_decorator

    # Import the trainer module via the canonical mount-manifest helper to
    # honor the same import semantics (importlib.util.spec_from_file_location
    # avoids polluting sys.modules).
    try:
        trainer_module = _import_with_duplicate_registry_fallback(
            trainer_module_path
        )
    except Exception as exc:
        if fail_on_import_error:
            raise RuntimeError(
                "trainer extra-mount discovery failed for "
                f"{trainer_module_path}: {type(exc).__name__}: {exc}"
            ) from exc
        print(
            f"[modal-train-lane][WAVE-3] WARN: trainer module {trainer_module_path} "
            f"could not be imported for extra-mount discovery: {type(exc).__name__}: {exc}. "
            f"Proceeding without trainer-declared extras (lane script must self-bootstrap).",
            file=_sys.stderr,
        )
        return payload

    extras = collect_extra_mount_paths(trainer_module)
    # Also collect required-input-file defaults so generic dispatcher honors
    # the same `required_input_file=True` contract Catalog #152 enforces for
    # operator wrappers. Many of these defaults live under
    # ``experiments/results/**`` (the Modal-IGNORED subtree) — exactly the
    # bug class the STC v2 anchor archive triggered.
    required_inputs = collect_tier_required_input_files(trainer_module)
    missing_required: list[tuple[str, Path]] = []
    missing_extras: list[Path] = []
    for flag, default_path in required_inputs:
        local = default_path if default_path.is_absolute() else repo_root / default_path
        if not local.is_file():
            missing_required.append((flag, local))
    for extra_path in extras:
        local = extra_path if extra_path.is_absolute() else repo_root / extra_path
        if not local.exists():
            missing_extras.append(local)

    if fail_on_missing_paths and (missing_required or missing_extras):
        pieces: list[str] = []
        if missing_required:
            pieces.append(
                "missing required_input_file defaults: "
                + "; ".join(f"{flag} default={path}" for flag, path in missing_required)
            )
        if missing_extras:
            pieces.append(
                "missing TIER_1_EXTRA_MOUNT_PATHS / MODAL_EXTRA_MOUNT_PATHS entries: "
                + "; ".join(str(path) for path in missing_extras)
            )
        raise RuntimeError(
            "trainer extra-mount discovery found missing path(s) for "
            f"{trainer_module_path}: " + " | ".join(pieces)
        )

    # Paths that are already in the structural minimum mount set are skipped
    # (they get mounted via STRUCTURAL_MINIMUM_DIRS without `results/**`
    # ignored, OR they live OUTSIDE the ignored subtree so the structural
    # mount already covers them).
    structural_top_dirs = {
        "src", "scripts", "upstream", "submissions", "tools",
    }
    # `experiments/` is mounted WITH `results/**` ignored — so files under
    # `experiments/results/` MUST be staged via payload bytes; files under
    # other `experiments/` subdirs are already covered.
    entries: list[tuple[str, str | None, Path]] = [
        ("extra_mount", None, path) for path in extras
    ] + [
        ("required_input_file", flag, path) for flag, path in required_inputs
    ]
    for kind, flag, entry in entries:
        rel_text = str(entry).strip()
        if not rel_text:
            continue
        local = entry if entry.is_absolute() else (repo_root / entry)
        # Resolve relative-to-repo-root for the payload key.
        try:
            rel = str(local.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            # Outside repo root — payload key is the absolute path so the
            # worker materializes it at the same absolute location.
            rel = str(local)
        # Skip if already in structural mount set + not under Modal-IGNORED
        # results/** subtree.
        top = rel.split("/", 1)[0] if "/" in rel else rel
        if top in structural_top_dirs:
            if kind == "required_input_file" and not local.is_file():
                print(
                    f"[modal-train-lane][WAVE-3] WARN: trainer-declared "
                    f"required_input_file {flag} default {rel} not found at "
                    f"{local} — skipping.",
                    file=_sys.stderr,
                )
            elif kind == "extra_mount" and not local.exists():
                print(
                    f"[modal-train-lane][WAVE-3] WARN: trainer-declared extra mount "
                    f"{rel} not found at {local} — skipping (declare on disk or "
                    "remove from TIER_1_EXTRA_MOUNT_PATHS).",
                    file=_sys.stderr,
                )
            continue
        if top == "experiments" and not rel.startswith("experiments/results/"):
            # experiments/ is structurally mounted with results/** ignored;
            # experiments/<non-results> is already covered.
            if kind == "required_input_file" and not local.is_file():
                print(
                    f"[modal-train-lane][WAVE-3] WARN: trainer-declared "
                    f"required_input_file {flag} default {rel} not found at "
                    f"{local} — skipping.",
                    file=_sys.stderr,
                )
            elif kind == "extra_mount" and not local.exists():
                print(
                    f"[modal-train-lane][WAVE-3] WARN: trainer-declared extra mount "
                    f"{rel} not found at {local} — skipping (declare on disk or "
                    "remove from TIER_1_EXTRA_MOUNT_PATHS).",
                    file=_sys.stderr,
                )
            continue
        if kind == "required_input_file" and not local.is_file():
            print(
                f"[modal-train-lane][WAVE-3] WARN: trainer-declared "
                f"required_input_file {flag} default {rel} not found at "
                f"{local} — skipping.",
                file=_sys.stderr,
            )
            continue
        if local.is_dir():
            staged_any = False
            for child in sorted(local.rglob("*")):
                if not child.is_file():
                    continue
                try:
                    child_suffix = child.relative_to(local).as_posix()
                except ValueError:
                    continue
                if not child_suffix:
                    continue
                child_rel = (
                    str(Path(rel) / child_suffix)
                    if Path(rel).is_absolute()
                    else f"{rel.rstrip('/')}/{child_suffix}"
                )
                if child_rel in payload:
                    continue
                payload[child_rel] = child.read_bytes()
                staged_any = True
            if not staged_any:
                print(
                    f"[modal-train-lane][WAVE-3] WARN: trainer-declared extra mount "
                    f"directory {rel} at {local} contains no regular files — skipping.",
                    file=_sys.stderr,
                )
            continue
        if not local.is_file():
            print(
                f"[modal-train-lane][WAVE-3] WARN: trainer-declared extra mount "
                f"{rel} not found at {local} — skipping (declare on disk or "
                "remove from TIER_1_EXTRA_MOUNT_PATHS).",
                file=_sys.stderr,
            )
            continue
        if rel in payload:
            continue
        payload[rel] = local.read_bytes()
    return payload


def _run_lane_inner(
    lane_script: str,
    label: str,
    env_overrides: dict,
    claim_ledger_bytes: bytes,
    mounted_code_git_head: str,
    mounted_code_git_branch: str,
    sentinel_sha256_local: dict,
    max_seconds: int = 14 * 3600,
    billing_deadline_unix_s: float | None = None,
    trainer_extra_mount_payload: dict | None = None,
) -> dict:
    """Container-side execution. Imports MUST be local (Modal serialization).

    WAVE-3 (2026-05-16): ``trainer_extra_mount_payload`` is a
    ``{rel_path: bytes}`` dict carrying every file the trainer's
    ``TIER_1_EXTRA_MOUNT_PATHS`` / ``MODAL_EXTRA_MOUNT_PATHS`` /
    ``required_input_file=True`` defaults pointed at, but which were NOT
    in the structural mount set (typically files under
    ``experiments/results/**`` which Modal's image build ignores per
    ``DEFAULT_RESULTS_IGNORE``). The worker materializes each entry under
    ``/tmp/pact/<rel>`` after the structural copy. Backward compat: ``None``
    or empty dict = no-op.
    """
    import json
    import os
    import shutil
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path

    if billing_deadline_unix_s is None:
        return {
            "returncode": 124,
            "error": "missing immutable Modal billing deadline",
            "timed_out": True,
            "artifacts": {},
            "score_claim": False,
            "promotion_eligible": False,
        }
    remaining_total_seconds = int(billing_deadline_unix_s - time.time())
    if remaining_total_seconds <= 0:
        return {
            "returncode": 124,
            "error": "immutable Modal billing deadline expired before worker start",
            "timed_out": True,
            "artifacts": {},
            "billing_deadline_unix_s": billing_deadline_unix_s,
            "remaining_total_seconds_at_start": remaining_total_seconds,
            "score_claim": False,
            "promotion_eligible": False,
        }

    def _enforce_absolute_billing_deadline() -> None:
        """Terminate this attempt when the immutable invocation envelope ends."""

        print(
            "[modal-train-lane] ABSOLUTE BILLING DEADLINE reached; "
            "terminating worker attempt",
            file=sys.stderr,
            flush=True,
        )
        os._exit(124)

    # Start before source copy/setup. A preemption restart receives the same
    # immutable deadline even though Modal resets its per-attempt timeout.
    billing_watchdog = threading.Timer(
        max(0.0, billing_deadline_unix_s - time.time()),
        _enforce_absolute_billing_deadline,
    )
    billing_watchdog.daemon = True
    billing_watchdog.start()

    image_workspace = Path("/workspace/pact")

    # COPY mounted source to a writable workspace. Modal's add_local_dir mounts
    # may be read-only at runtime (modal_auth_eval avoids this entirely by
    # using tempfile/copy). Lane scripts write env.sh + need scripts/ to be
    # writable for the NVDEC probe stub. Round 4 caught this.
    workspace = Path("/tmp/pact")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    print(f"[modal-train-lane] copying mounted source → {workspace}")
    for sub in ("src", "scripts", "submissions", "upstream", "experiments", "tools"):
        src_path = image_workspace / sub
        if src_path.exists():
            shutil.copytree(src_path, workspace / sub, symlinks=True)
    pp = image_workspace / "pyproject.toml"
    if pp.exists():
        shutil.copy2(pp, workspace / "pyproject.toml")
    claim_path = workspace / ".omx/state/active_lane_dispatch_claims.md"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_bytes(claim_ledger_bytes)  # BARE_WRITE_OK: single-writer Modal worker copies immutable local claim snapshot

    # WAVE-3 (2026-05-16) deeper structural fix per STC v2 FIX CRITICAL
    # DEEPER FINDING: trainer-declared TIER_1_EXTRA_MOUNT_PATHS /
    # MODAL_EXTRA_MOUNT_PATHS / required_input_file=True defaults that live
    # under experiments/results/** (Modal-IGNORED subtree per
    # DEFAULT_RESULTS_IGNORE) are staged here as bytes payloads. Sister of
    # Wave 2's driver-side defensive resolver (scripts/remote_lane_substrate_*.sh
    # resolve_required_input_modal_aware helper) — Wave 2 catches it at the
    # driver layer; Wave 3 catches it at the dispatcher layer.
    #
    # The payload was computed at LOCAL dispatch time by
    # _collect_trainer_extra_mount_payload reading the trainer module's
    # extra-mount declarations. Materializing under /tmp/pact/<rel> means
    # the driver's WORKSPACE-anchored path lookups (e.g.
    # $WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip)
    # resolve correctly without any defensive multi-candidate probing.
    if trainer_extra_mount_payload:
        print(
            f"[modal-train-lane][WAVE-3] staging {len(trainer_extra_mount_payload)} "
            f"trainer-declared extra mount path(s) under {workspace}"
        )
        for rel, data in trainer_extra_mount_payload.items():
            # Resolve under workspace (relative) or as absolute (rare).
            rel_path = Path(rel)
            target = rel_path if rel_path.is_absolute() else workspace / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            # BARE_WRITE_OK: single-writer Modal worker materializing
            # immutable local trainer-declared payload snapshot
            target.write_bytes(data)

    os.chdir(workspace)

    # sys.path injection (matches modal_auth_eval.py pattern). Avoids
    # `pip install -e .` which would write src/tac.egg-info/ — risky.
    sys.path.insert(0, str(workspace / "src"))
    sys.path.insert(0, str(workspace / "upstream"))

    # Write env.sh that lane scripts source. Mirrors the one that
    # remote_setup_full.sh writes on Vast.ai.
    # FFmpeg-master path: BtbN tarball extracts to /opt/ffmpeg-master-latest-linux64-gpl/.
    # The /usr/local/bin/ffmpeg-master symlink covers binary lookup; LD_LIBRARY_PATH
    # must point to the actual lib dir (BtbN GPL builds are mostly static but
    # some shared libs ship in lib/).
    ffmpeg_root = "/opt/ffmpeg-master-latest-linux64-gpl"
    env_sh = workspace / "env.sh"
    # CRITICAL: WORKSPACE/TAC_UPSTREAM_DIR must point at the WRITABLE /tmp/pact
    # copy, NOT the read-only /workspace/pact mount. Lane scripts source
    # this file then use $WORKSPACE for all path lookups — wrong value here
    # silently re-anchors them at the read-only mount. Round 5 caught this.
    # bin_dir is computed below; refer forward via workspace/_modal_bin
    env_sh.write_text(
        "# auto-generated by modal_train_lane.py\n"
        "export FFMPEG_BIN=/usr/local/bin/ffmpeg-master\n"
        f"export PATH={workspace}/_modal_bin:{ffmpeg_root}/bin:/root/.local/bin:$PATH\n"
        f"export LD_LIBRARY_PATH={ffmpeg_root}/lib:${{LD_LIBRARY_PATH:-}}\n"
        f"export PYTHONPATH={workspace}/src:{workspace}/upstream:{workspace}\n"
        f"export TAC_UPSTREAM_DIR={workspace}/upstream\n"
        f"export PYBIN={sys.executable}\n"
        f"export WORKSPACE={workspace}\n"
        f"export T1_DISPATCH_CLAIMS_PATH={claim_path}\n"
        f"export SCPP_DISPATCH_CLAIMS_PATH={claim_path}\n"
        f"export T1_MOUNTED_CODE_GIT_HEAD={mounted_code_git_head}\n"
        f"export T1_MOUNTED_CODE_GIT_BRANCH={mounted_code_git_branch}\n"
        f"export SCPP_MOUNTED_CODE_GIT_HEAD={mounted_code_git_head}\n"
        f"export SCPP_MOUNTED_CODE_GIT_BRANCH={mounted_code_git_branch}\n"
        'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"\n'
        f"export DALI_DISABLE_NVML={DALI_DISABLE_NVML_VALUE}\n"
        f"export PYTORCH_CUDA_ALLOC_CONF={PYTORCH_CUDA_ALLOC_CONF_VALUE}\n"
        "export AUTH_EVAL_DEVICE=cpu\n"
        "export MODAL_AUTH_EVAL_ADVISORY_ONLY=1\n"
        "export SCORE_CLAIM=false\n"
        "export PROMOTION_ELIGIBLE=false\n"
        "export T1_RUN_CONTEST_CUDA_AUTH_EVAL=0\n"
        "export SCPP_RUN_CONTEST_CUDA_AUTH_EVAL=0\n"
        "export RUN_CONTEST_EVAL=0\n"
    )

    # Stub probe_nvdec.sh so Stage 0 of every lane passes. Modal containers
    # don't reliably expose libnvcuvid.so. We work around this by:
    #   1. Stubbing the probe to always pass (training itself doesn't need NVDEC)
    #   2. Forcing auth_eval to --device cpu via AUTH_EVAL_DEVICE env (lane
    #      scripts honor this — see remote_lane_*.sh Stage 5). CPU device
    #      selects AVVideoDataset (PyAV) per upstream/evaluate.py:39-42 —
    #      the ONLY way to get a diagnostic score on Modal without NVDEC.
    # Round 4 caught the AVVideoDataset-auto-fallback claim was wrong;
    # the fallback ONLY happens when device.type != cuda. So we have to
    # explicitly tell auth_eval to use cpu.
    stub_probe = workspace / "scripts" / "probe_nvdec.sh"
    stub_probe.write_text(
        "#!/bin/bash\n"
        "# Modal-runtime stub. Lane training doesn't need NVDEC; auth_eval\n"
        "# uses --device cpu (AVVideoDataset/PyAV), so scores are advisory only.\n"
        "echo '[probe_nvdec] Modal runtime — NVDEC not required (auth_eval uses cpu device)'\n"
        "exit 0\n"
    )
    stub_probe.chmod(0o755)

    # Many lane scripts call bare `python3` (vs `$PYBIN`). On Modal debian_slim
    # the system python3 may not be the same interpreter as sys.executable
    # (with torch/tac installed). Symlink python3 → sys.executable in a
    # PATH-priority dir so bare `python3` resolves to the working interpreter.
    # Round 7 catch.
    bin_dir = workspace / "_modal_bin"
    bin_dir.mkdir(exist_ok=True)
    py3_link = bin_dir / "python3"
    if py3_link.exists() or py3_link.is_symlink():
        py3_link.unlink()
    py3_link.symlink_to(sys.executable)
    py_link = bin_dir / "python"
    if py_link.exists() or py_link.is_symlink():
        py_link.unlink()
    py_link.symlink_to(sys.executable)

    # NOTE: git is installed via apt_install (line 43). Lane scripts that
    # run `git rev-parse HEAD` get real git output. Round 13 caught that
    # the previous "_git_stub.sh" was written but never placed on PATH —
    # removed since lane scripts already have `|| echo no-git` fallbacks.

    # Sentinel that this is Modal (skip Vast.ai-specific paths)
    (workspace / ".MODAL_RUNTIME").write_text("1\n")

    # Catalog #166: worker-side HEAD parity assertion. The local
    # entrypoint reads ``git rev-parse HEAD`` at dispatch time and threads
    # the SHA via env. The worker re-reads ``git rev-parse HEAD`` against
    # the mounted ``/workspace/pact`` and asserts they match. A divergence
    # surfaces "Modal worker mounted a snapshot from a different SHA than
    # the operator believed" (H3 of the diagnostic taxonomy) BEFORE the
    # training inner loop starts — saving $5-15 of crash-on-startup cost.
    expected_head = mounted_code_git_head if mounted_code_git_head else ""
    worker_head = ""
    worker_sentinel_sha256: dict = {}
    sentinel_mismatches: list[str] = []
    if expected_head and expected_head != "unknown":
        try:
            git_proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
            )
            worker_head = git_proc.stdout.strip()
        except (FileNotFoundError, OSError):
            worker_head = ""
        if worker_head and worker_head != expected_head:
            print(
                "[modal-train-lane][CATALOG_166] WARN: worker git HEAD "
                f"({worker_head[:12]}) != env-injected HEAD "
                f"({expected_head[:12]}). The mount may be stale OR the "
                "worker /workspace/pact has its own .git. Refer to "
                "`tools/diagnose_modal_worker_source_staleness.py` for "
                "verdict-class diagnosis. NOT fail-closed (mount is the "
                "source of truth on Modal); recording for post-mortem only."
            )
        # Record into a small ledger we can harvest along with artifacts.
        head_ledger = workspace / "modal_worker_head_ledger.json"
        for rel, expected_sha in sentinel_sha256_local.items():
            p = workspace / rel
            if not p.is_file():
                worker_sha = "MISSING_WORKER"
            else:
                import hashlib as _hashlib

                worker_sha = _hashlib.sha256(p.read_bytes()).hexdigest()
            worker_sentinel_sha256[rel] = worker_sha
            if expected_sha != worker_sha:
                sentinel_mismatches.append(
                    f"{rel}: local={expected_sha[:12]} worker={worker_sha[:12]}"
                )
        head_ledger.write_text(json.dumps({
            "schema": "modal_worker_head_ledger_v1_catalog166",
            "env_injected_head": expected_head,
            "env_injected_branch": mounted_code_git_branch or "",
            "worker_git_head_at_start": worker_head,
            "sentinel_files_local_sha256": sentinel_sha256_local,
            "worker_sentinel_sha256": worker_sentinel_sha256,
            "sentinel_mismatches": sentinel_mismatches,
        }, indent=2, sort_keys=True))

    # Heartbeat tracking
    log_dir = workspace / f"results/{label}"
    log_dir.mkdir(parents=True, exist_ok=True)
    root_head_ledger = workspace / "modal_worker_head_ledger.json"
    if root_head_ledger.is_file():
        shutil.copy2(root_head_ledger, log_dir / "modal_worker_head_ledger.json")
    if sentinel_mismatches:
        billing_watchdog.cancel()
        return {
            "returncode": 13,
            "error": (
                "Catalog #166 worker sentinel hash mismatch before training: "
                + "; ".join(sentinel_mismatches[:5])
            ),
            "artifacts": {
                f"results/{label}/modal_worker_head_ledger.json":
                    (log_dir / "modal_worker_head_ledger.json").read_bytes()
            },
            "stdout_tail": "",
            "stderr_tail": "",
            "score_claim": False,
            "promotion_eligible": False,
        }
    volume_dir = Path("/modal_results") / label
    volume_dir.mkdir(parents=True, exist_ok=True)

    def _modal_workspace_env(value: object) -> str:
        """Map recipe paths from the read-only Modal mount to the writable copy."""
        text = str(value)
        readonly = "/workspace/pact"
        if text == readonly:
            return str(workspace)
        if text.startswith(readonly + "/"):
            return str(workspace) + text[len(readonly):]
        return text

    # Build env. PATH/LD_LIBRARY_PATH point to the actual extracted ffmpeg dir
    # (not a phantom /opt/ffmpeg-master). PYBIN is propagated to bash + child
    # python invocations so probe_nvdec.sh + lane scripts inherit it.
    # Modal training wrappers are not a promotion surface. Generic legacy
    # exact-eval toggles stay disabled, but modern substrate trainers may run
    # contest_auth_eval.py inline as a fail-closed smoke/full custody check
    # when they parse the resulting evidence contract themselves.
    env = {
        **os.environ,
        "WORKSPACE": str(workspace),
        "PYBIN": sys.executable,
        "PYTHONPATH": f"{workspace}/src:{workspace}/upstream:{workspace}",
        "DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE,
        "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF_VALUE,
        "FFMPEG_BIN": "/usr/local/bin/ffmpeg-master",
        # _modal_bin first so `python3` and `python` resolve to sys.executable
        # (the venv with torch/tac), not /usr/bin/python3 (system, no deps).
        "PATH": f"{bin_dir}:{ffmpeg_root}/bin:/root/.local/bin:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{ffmpeg_root}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        "TAC_UPSTREAM_DIR": str(workspace / "upstream"),
        "MODAL_RUNTIME": "1",
        "AUTH_EVAL_DEVICE": "cpu",
        "MODAL_AUTH_EVAL_ADVISORY_ONLY": "1",
        "SCORE_CLAIM": "false",
        "PROMOTION_ELIGIBLE": "false",
        "T1_DISPATCH_CLAIMS_PATH": str(claim_path),
        "SCPP_DISPATCH_CLAIMS_PATH": str(claim_path),
        "T1_MOUNTED_CODE_GIT_HEAD": mounted_code_git_head,
        "T1_MOUNTED_CODE_GIT_BRANCH": mounted_code_git_branch,
        "SCPP_MOUNTED_CODE_GIT_HEAD": mounted_code_git_head,
        "SCPP_MOUNTED_CODE_GIT_BRANCH": mounted_code_git_branch,
        "T1_RUN_CONTEST_CUDA_AUTH_EVAL": "0",
        "SCPP_RUN_CONTEST_CUDA_AUTH_EVAL": "0",
        "RUN_CONTEST_EVAL": "0",
        # Lanes test for AUTO_DESTROY_VAST + VAST_INSTANCE_ID; on Modal these
        # are no-ops since Modal manages instance lifecycle.
        "AUTO_DESTROY_VAST": "0",
    }
    env.update({str(k): _modal_workspace_env(v) for k, v in env_overrides.items()})
    exact_eval_switches = (
        "T1_RUN_CONTEST_CUDA_AUTH_EVAL",
        "SCPP_RUN_CONTEST_CUDA_AUTH_EVAL",
        "RUN_CONTEST_EVAL",
    )
    truthy = {"1", "true", "yes", "on"}
    requested = [
        key for key in exact_eval_switches
        if env.get(key, "").strip().lower() in truthy
    ]
    if requested:
        billing_watchdog.cancel()
        return {
            "returncode": 12,
            "error": (
                "refusing exact CUDA auth-eval from modal_train_lane.py; "
                f"requested switches={requested}. "
                "Use the canonical claimed exact-eval dispatcher instead."
            ),
            "artifacts": {},
            "stdout_tail": "",
            "stderr_tail": "",
            "score_claim": False,
            "promotion_eligible": False,
        }

    # Run the lane script
    lane_path = workspace / lane_script
    if not lane_path.exists():
        billing_watchdog.cancel()
        return {
            "returncode": 2,
            "error": f"lane script not found: {lane_script}",
            "artifacts": {},
            "stdout_tail": "",
            "stderr_tail": "",
        }

    print(f"[modal-train-lane] starting lane: {lane_script} (label={label})")
    t0 = time.monotonic()
    artifact_mtime_floor = time.time() - 5.0

    log_path = workspace / f"modal_lane_{label}.log"
    timed_out = False
    stop_sync = threading.Event()

    def sync_volume() -> None:
        sources = [
            workspace / "experiments" / "results",
            workspace / "results",
            log_path,
            workspace / "modal_worker_head_ledger.json",
        ]
        while not stop_sync.is_set():
            try:
                for src in sources:
                    if not src.exists():
                        continue
                    dst = volume_dir / src.name
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
                (volume_dir / "modal_live_metadata.json").write_text(json.dumps({
                    "label": label,
                    "lane_script": lane_script,
                    "synced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "auth_eval_device": "cpu",
                    "auth_eval_advisory_only": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "volume": RESULTS_VOL,
                    "volume_prefix": f"{label}/",
                    "billing_deadline_unix_s": billing_deadline_unix_s,
                    "remaining_seconds_to_billing_deadline": max(
                        0, int(billing_deadline_unix_s - time.time())
                    ),
                }, indent=2))
                results_vol.commit()
                print(f"[modal-train-lane] volume sync committed: {RESULTS_VOL}/{label}/")
            except Exception as exc:
                print(f"[modal-train-lane] volume sync failed: {exc!r}")
            stop_sync.wait(timeout=180)

    sync_thread = threading.Thread(target=sync_volume, daemon=True)
    sync_thread.start()
    with log_path.open("w") as logf:
        # Recompute at the child boundary: source copy/setup time consumes the
        # same cumulative billing envelope and cannot leave a stale timeout.
        remaining_total_before_child = int(billing_deadline_unix_s - time.time())
        remaining_child_seconds = min(
            max_seconds,
            remaining_total_before_child - MODAL_FINALIZATION_GRACE_SECONDS,
        )
        if remaining_child_seconds <= 0:
            timed_out = True
            rc = 124
            print(
                "[modal-train-lane] immutable deadline leaves no child budget "
                "after cleanup reserve — collecting setup artifacts"
            )
        else:
            try:
                proc = run_in_process_group(
                    ["bash", str(lane_path)],
                    env=env, cwd=workspace,
                    stdout=logf, stderr=subprocess.STDOUT,
                    timeout=remaining_child_seconds,
                    check=False,
                )
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                rc = 124
                print(
                    "[modal-train-lane] TIMEOUT after "
                    f"{remaining_child_seconds}s — collecting partial artifacts"
                )
    stop_sync.set()
    sync_thread.join(timeout=60)
    try:
        for src in (workspace / "experiments" / "results", workspace / "results", log_path):
            if not src.exists():
                continue
            dst = volume_dir / src.name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        results_vol.commit()
        print(f"[modal-train-lane] final volume commit: {RESULTS_VOL}/{label}/")
    except Exception as exc:
        print(f"[modal-train-lane] final volume commit failed: {exc!r}")
    elapsed = time.monotonic() - t0
    print(f"[modal-train-lane] finished in {elapsed:.1f}s rc={rc} timed_out={timed_out}")

    # Collect output artifacts. Lane scripts write to varied locations:
    #   - $WORKSPACE/results/<label>/  (canonical, used by some)
    #   - $WORKSPACE/<lane_X>_results/  (used by lane_omega, lane_mae_v, etc.)
    #   - $WORKSPACE/lane_*_results/  (catch-all for *_results/ siblings)
    #   - $WORKSPACE/submissions/robust_current/  (archive output sometimes)
    # Scan the WHOLE workspace for these extensions to avoid silent loss.
    artifacts: dict[str, bytes] = {}
    skipped_large: list[tuple[str, int]] = []
    # Top-level dirs to scan (avoid scanning src/ scripts/ etc.)
    scan_roots = [
        workspace / "results",
        workspace / "experiments" / "results",
        workspace / "modal_worker_head_ledger.json",
        volume_dir,
    ]
    # Also any */_results/ siblings of workspace root
    for child in workspace.iterdir():
        if child.is_dir() and child.name.endswith("_results"):
            scan_roots.append(child)
    # Plus archives written into submissions/robust_current
    scan_roots.append(workspace / "submissions" / "robust_current")
    # Plus the modal log we wrote at workspace root
    if log_path.exists():
        scan_roots.append(log_path)

    for root in scan_roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for fp in files:
            if not fp.is_file():
                continue
            rel = modal_training_artifact_relative_path(
                fp,
                workspace=workspace,
                volume_dir=volume_dir,
            )
            if not modal_training_artifact_should_collect(rel):
                continue
            rel_str = str(rel)
            if rel_str in artifacts:
                continue
            try:
                st = fp.stat()
                size = st.st_size
                # OVERNIGHT-GG vendor-stub fix 2026-05-21: files under
                # output/submission/ are part of the trainer's atomic
                # submission emission. shutil.copy2 preserves source mtime,
                # so vendored .py module bodies (architecture.py / archive.py /
                # codebook.py / inflate.py / prior_application.py /
                # procedural_codebook_inflate.py / _shared/inflate_runtime.py /
                # seed_derived_codebook.py) inherit the OLD mtime from
                # /tmp/pact/src/tac/... (copied via copytree at lane start
                # with symlinks=True preserving local-repo mtimes). The
                # mtime_floor was designed to filter STALE pre-existing
                # artifacts from prior runs, NOT to filter freshly-vendored
                # module bodies that are part of the SUBMISSION packet. Per
                # Catalog #146 inflate runtime contract: the submission_dir
                # is a unit — drop any of its files and the contest auth_eval
                # fails ModuleNotFoundError. Bypass the mtime_floor for the
                # submission_dir specifically. Empirical anchor: OVERNIGHT-CC
                # 99d06f967 4 paired Modal auth_eval rc=1 in ~3s each because
                # 8 DP1 vendored .py modules were silently dropped by this
                # filter; OVERNIGHT-GG 2026-05-21 root-cause fix.
                rel_parts = rel.parts
                under_submission = (
                    len(rel_parts) >= 3
                    and rel_parts[0] == "output"
                    and rel_parts[1] == "submission"
                )
                if not under_submission and st.st_mtime < artifact_mtime_floor:
                    continue
                # 500MB threshold — covers final .bin (~300KB) AND mid-training
                # .pt checkpoints (50-200MB) AND large .mkv masks. Anything
                # bigger is almost certainly intermediate state we don't need.
                if size > 500 * 1024 * 1024:
                    skipped_large.append((rel_str, size))
                    print(f"[modal-train-lane] SKIP large {rel_str} ({size/1e6:.1f}MB)")
                    continue
                artifacts[rel_str] = fp.read_bytes()
            except (FileNotFoundError, PermissionError) as e:
                print(f"[modal-train-lane] SKIP unreadable {fp}: {e}")
                continue

    # Tail log files for return value
    stdout_tail = ""
    if log_path.exists():
        try:
            stdout_tail = log_path.read_text(errors="ignore")[-4000:]
        except Exception:
            pass

    billing_watchdog.cancel()
    return {
        "returncode": rc,
        "timed_out": timed_out,
        "artifacts": artifacts,
        "stdout_tail": stdout_tail,
        "elapsed_seconds": elapsed,
        "billing_deadline_unix_s": billing_deadline_unix_s,
        "remaining_child_seconds_at_start": remaining_child_seconds,
        "skipped_large_artifacts": skipped_large,
        "auth_eval_device": "cpu",
        "auth_eval_advisory_only": True,
        "score_claim": False,
        "promotion_eligible": False,
    }


# Authority boundary: an account-authorized Modal SDK caller can always use
# Function.from_name(...).with_options(...) or define another paid function.
# Provider IAM/budget controls are therefore the hard pre-allocation ceiling.
# This CPU/600s endpoint plus its mandatory dispatch-stage receipt closes
# accidental and repository-native reuse and refuses before source copy.
@app.function(
    image=training_image,
    cpu=MODAL_GPU_CPU_REQUEST_LIMIT,
    memory=MODAL_GPU_MEMORY_REQUEST_LIMIT_MB,
    timeout=MODAL_PREFLIGHT_HARD_TIMEOUT_SECONDS,
    volumes={"/modal_results": results_vol},
)
def run_lane_training_cpu(
    lane_script: str,
    label: str,
    env_overrides: dict,
    claim_ledger_bytes: bytes,
    mounted_code_git_head: str,
    mounted_code_git_branch: str,
    sentinel_sha256_local: dict,
    max_seconds: int = 14 * 3600,
    billing_deadline_unix_s: float | None = None,
    trainer_extra_mount_payload: dict | None = None,
    requested_gpu: str = "",
    dispatch_stage: str = "",
    dispatch_nonce: str = "",
    cpu_preflight_receipt: str = "",
    dispatch_receipt_label: str = "",
) -> dict:
    import os
    import time

    visible_gpu = _gpu_visible_in_container(os.environ)
    try:
        _validate_remote_dispatch_stage(
            dispatch_stage=dispatch_stage,
            dispatch_nonce=dispatch_nonce,
            cpu_preflight_receipt=cpu_preflight_receipt,
            requested_gpu=requested_gpu,
            lane_script=lane_script,
            label=dispatch_receipt_label,
            mounted_code_git_head=mounted_code_git_head,
            visible_gpu=visible_gpu,
        )
    except ValueError as exc:
        return {
            "returncode": 13,
            "error": f"remote dispatch-stage policy refused invocation: {exc}",
            "timed_out": False,
            "artifacts": {},
            "score_claim": False,
            "promotion_eligible": False,
        }
    if requested_gpu != "CPU":
        invocation_started_unix_s = time.time()
        try:
            _validate_v9_remote_h100_invocation(
                lane_script=lane_script,
                label=label,
                env_overrides=env_overrides,
                mounted_code_git_head=mounted_code_git_head,
                mounted_code_git_branch=mounted_code_git_branch,
                sentinel_sha256_local=sentinel_sha256_local,
                max_seconds=max_seconds,
                billing_deadline_unix_s=billing_deadline_unix_s,
                invocation_started_unix_s=invocation_started_unix_s,
                requested_gpu=requested_gpu,
            )
        except ValueError as exc:
            return {
                "returncode": 13,
                "error": f"remote V9 hard-cap policy refused invocation: {exc}",
                "timed_out": False,
                "artifacts": {},
                "score_claim": False,
                "promotion_eligible": False,
            }
    result = _run_lane_inner(
        lane_script,
        label,
        env_overrides,
        claim_ledger_bytes,
        mounted_code_git_head,
        mounted_code_git_branch,
        sentinel_sha256_local,
        max_seconds=max_seconds,
        billing_deadline_unix_s=billing_deadline_unix_s,
        trainer_extra_mount_payload=trainer_extra_mount_payload,
    )
    if (
        dispatch_stage == "cpu_preflight"
        and result.get("returncode") == 0
        and result.get("timed_out") is not True
    ):
        result["cpu_preflight_receipt"] = _derive_cpu_preflight_receipt(
            dispatch_nonce=dispatch_nonce,
            lane_script=lane_script,
            label=dispatch_receipt_label,
            mounted_code_git_head=mounted_code_git_head,
        )
    return result


def _compact_stamp() -> str:
    import datetime as dt

    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_value(repo_root, *args: str) -> str:
    import subprocess

    proc = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value else "unknown"


def _git_dirty_tree_summary(repo_root) -> dict:
    """Return ``{"dirty": bool, "dirty_paths_count": int, "summary": str,
    "categorized": dict}``.

    Captures whether the working tree has uncommitted edits AT DISPATCH TIME.
    Catalog #166 (PHASE-B1-PIVOT 2026-05-12): a stale fix on disk silently
    ships to the Modal worker because ``add_local_dir`` snapshots the working
    tree, not the HEAD blob. Recording this in ``modal_metadata.json`` lets
    post-mortem distinguish "operator dispatched before fix landed" (clean
    tree, HEAD before fix-commit) from "Modal worker mounted stale snapshot"
    (clean tree, HEAD AT fix-commit, but worker hash differs from HEAD blob).

    DX-POLISH-WAVE 2026-05-15 (Catalog #238): the returned dict now includes
    a ``categorized`` field that groups dirty paths into actionable buckets so
    the operator can immediately see whether the dirty tree is a sister
    subagent's research artifact (typically safe to set the Catalog #202
    bypass for trusted-sentinel-clean dispatch), an operator-owned source
    edit (typically requires a commit), or a build artifact (typically a GC
    cleanup / .gitignore update). See :func:`_categorize_dirty_paths`.
    """

    import subprocess

    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "dirty": False,
            "dirty_paths_count": 0,
            "summary": "unknown:git-status-failed",
            "categorized": {},
        }
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    paths = [_extract_porcelain_path(line) for line in lines]
    paths = [p for p in paths if p]
    categorized = _categorize_dirty_paths(paths)
    return {
        "dirty": bool(lines),
        "dirty_paths_count": len(lines),
        "summary": "; ".join(line.strip() for line in lines[:10]),
        "categorized": categorized,
    }


def _extract_porcelain_path(line: str) -> str:
    """Parse ``git status --porcelain`` line into the path component.

    Handles XY-status prefix (2 chars + 1 space) and optional rename arrow
    (``orig -> new``). Returns the destination path of a rename so the
    categorization sees the path that lives on disk now.
    """

    raw = line[3:] if len(line) > 3 else line.strip()
    if " -> " in raw:
        raw = raw.rsplit(" -> ", 1)[1]
    return raw.strip().strip('"')


# DX-POLISH-WAVE 2026-05-15 (Catalog #238 / DX-1): dirty-path categorization.
# Each bucket name is a stable label that the FATAL error message references
# in its actionable next-steps section. The patterns are intentionally
# ordered: a path is assigned to the FIRST matching bucket. Sister/codex/
# OSS-mirror buckets are checked BEFORE the broad source/test buckets so a
# `_codex.md` research ledger is never miscategorized as "operator-owned
# source edit".
_DX_DIRTY_PATH_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "codex_sister_research_ledger",
        (
            "_codex.md",
            "_codex.json",
            ".omx/research/",
            "feedback_codex_",
            "/codex_runs/",
        ),
    ),
    (
        "subagent_state_ephemeral",
        (
            ".omx/state/",
            ".omx/tmp/",
            ".omx/oss_export/",
        ),
    ),
    (
        "build_artifact_or_derived_output",
        (
            "experiments/results/",
            "reports/raw/",
            "build/lib/",
            "__pycache__/",
            ".pytest_cache/",
        ),
    ),
    (
        "vendored_intake_clone",
        (
            "_intake_",
            "vendored/",
            "submissions/exact_current/",
        ),
    ),
    (
        "operator_owned_source",
        (
            "src/tac/",
            "tools/",
            "experiments/",
            "scripts/",
            "submissions/",
        ),
    ),
    (
        "documentation_or_memo",
        (
            "docs/",
            "feedback_",
            "MEMORY.md",
            "CLAUDE.md",
            "AGENTS.md",
            ".md",
        ),
    ),
    (
        "configuration_or_recipe",
        (
            ".omx/operator_authorize_recipes/",
            "configs/",
            ".ralph/",
            "pyproject.toml",
        ),
    ),
)


def _categorize_dirty_paths(paths: list[str]) -> dict:
    """Group dirty paths into actionable buckets.

    Returns a dict with bucket-name keys and value
    ``{"count": int, "examples": list[str]}``. Buckets with zero matches
    are omitted. An ``unclassified`` bucket captures everything that
    matches none of the canonical patterns; it surfaces as "see git status
    --porcelain" in the actionable error message.
    """

    counts: dict[str, list[str]] = {bucket: [] for bucket, _ in _DX_DIRTY_PATH_CATEGORIES}
    counts["unclassified"] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        assigned = False
        for bucket, patterns in _DX_DIRTY_PATH_CATEGORIES:
            for pattern in patterns:
                if pattern in normalized:
                    counts[bucket].append(path)
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            counts["unclassified"].append(path)
    out: dict[str, dict] = {}
    for bucket, items in counts.items():
        if not items:
            continue
        out[bucket] = {
            "count": len(items),
            "examples": items[:3],
        }
    return out


def _format_dx_polish_actionable_fatal(
    dirty_tree: dict, mounted_code_git_head: str
) -> str:
    """DX-POLISH-WAVE 2026-05-15 (Catalog #238 / DX-1).

    Build the actionable Catalog #166 FATAL message body. Replaces the
    pre-DX-polish terse 4-line body with a categorized + next-steps body
    so the operator can immediately tell:
      (a) WHICH classes of dirty paths the working tree contains (sister
          subagent research artifact vs operator-owned source edit vs
          build artifact);
      (b) WHICH next-step is appropriate for each class (commit / Catalog
          #202 bypass / smoke-before-full path which auto-relaxes the
          clean-head check / GC cleanup);
      (c) The exact commands to invoke for each path.

    Sister of CLAUDE.md "Beauty, simplicity, and developer experience" +
    Catalog #208 (no local-absolute-paths) — the message uses repo-relative
    canonical paths.
    """

    count = int(dirty_tree.get("dirty_paths_count", 0))
    categorized = dirty_tree.get("categorized", {}) or {}
    summary = dirty_tree.get("summary", "")
    head_short = mounted_code_git_head[:12] if mounted_code_git_head else "unknown"

    lines: list[str] = []
    lines.append(
        f"FATAL [Catalog #166]: --require-clean-head is set and the working "
        f"tree has {count} uncommitted edit(s)."
    )
    lines.append("")
    lines.append(
        f"  Mounted git HEAD: {head_short}. Modal's add_local_dir snapshots "
        "the WORKING TREE (not HEAD), so any uncommitted byte ships to the "
        "worker - which is why this gate exists."
    )
    lines.append("")

    if categorized:
        lines.append("Categorized dirty paths:")
        for bucket, info in categorized.items():
            label = bucket.replace("_", " ")
            example_str = ", ".join(info["examples"])
            if info["count"] > len(info["examples"]):
                example_str += f", + {info['count'] - len(info['examples'])} more"
            lines.append(f"  - {label} ({info['count']}): {example_str}")
        lines.append("")
    elif summary:
        lines.append(f"Raw porcelain summary (first 10): {summary}")
        lines.append("")

    lines.append("Actionable next-steps (pick the one that matches your dirty bucket):")
    lines.append("")

    if "operator_owned_source" in categorized:
        lines.append(
            "  [1] OPERATOR-OWNED SOURCE EDITS - commit through the canonical "
            "serializer:"
        )
        lines.append(
            "      .venv/bin/python tools/subagent_commit_serializer.py "
            "--message '<one-liner>' --files <paths> "
            "--expected-content-sha256 '<path>=<sha>'"
        )
        lines.append(
            "      Per CLAUDE.md 'Subagent commits MUST use serializer'. "
            "Re-run dispatch after the commit lands."
        )
        lines.append("")

    if (
        "codex_sister_research_ledger" in categorized
        or "subagent_state_ephemeral" in categorized
        or "documentation_or_memo" in categorized
    ):
        lines.append(
            "  [2] SISTER-SUBAGENT RESEARCH LEDGER / .omx state / docs - if you "
            "have INDEPENDENTLY VERIFIED that the Catalog #166 sentinel set is "
            "clean (the dispatched files match HEAD), set the Catalog #202 "
            "paired-env bypass:"
        )
        lines.append(
            "      OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 \\"
        )
        lines.append(
            "      OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1 \\"
        )
        lines.append("      <re-run dispatch>")
        lines.append(
            "      Catalog #166 worker-side sentinel hash check still runs; the "
            "bypass only relaxes the whole-tree refusal."
        )
        lines.append("")

    if (
        "build_artifact_or_derived_output" in categorized
        or "subagent_state_ephemeral" in categorized
    ):
        lines.append(
            "  [3] BUILD ARTIFACTS or .omx/tmp/ ephemeral - run the canonical "
            "GC helper (refuses tracked-path deletion via Catalog #154/#156):"
        )
        lines.append(
            "      .venv/bin/python tools/gc_experiments_results.py --dry-run "
            "(then --apply with --operator-approved '<handle>:<UTC_timestamp>')"
        )
        lines.append("")

    lines.append(
        "  [4] PREFER SMOKE - the canonical smoke-before-full wrapper auto-"
        "detects a dirty tree and routes the smoke phase through the Catalog "
        "#202 bypass automatically (the cheap $0.30 smoke validates "
        "integration; only the FULL phase fails-closed on dirty):"
    )
    lines.append(
        "      .venv/bin/python tools/run_modal_smoke_before_full.py "
        "--recipe <recipe-name> --smoke-only"
    )
    lines.append(
        "      (then re-run without --smoke-only on a CLEAN tree for the FULL canary)"
    )
    lines.append("")

    lines.append(
        "  [5] OPERATOR OVERRIDE - if you accept the risk that the dirty bytes "
        "ship to the Modal worker (Catalog #166 worker-side hash check still "
        "runs), pass --require-clean-head=False to experiments/modal_train_lane.py "
        "directly. NOT recommended for paid full dispatches."
    )

    return "\n".join(lines)


def _infer_lane_id(
    lane_script: str,
    explicit_lane_id: str = "",
    *,
    repo_root: Path | None = None,
) -> str:
    is_v9 = _classify_v9_lane_script(lane_script, repo_root=repo_root)
    if explicit_lane_id.strip():
        return explicit_lane_id.strip()
    if is_v9:
        return V9_LANE_ID
    normalized = Path(lane_script).as_posix()
    return KNOWN_LANE_IDS.get(normalized, Path(normalized).stem)


def _active_claim_exists(
    repo_root,
    *,
    lane_id: str,
    instance_job_id: str | None,
) -> bool:
    import json
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "tools/claim_lane_dispatch.py",
            "summary",
            "--claims-path",
            ".omx/state/active_lane_dispatch_claims.md",
            "--format",
            "json",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "FATAL: unable to read active dispatch claims; refusing Modal spawn. "
            f"rc={proc.returncode} stderr={proc.stderr.strip()[:500]!r}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            "FATAL: active dispatch claim summary is not valid JSON; "
            "refusing Modal spawn."
        ) from exc
    if not isinstance(payload.get("active"), list):
        raise SystemExit(
            "FATAL: active dispatch claim summary lacks an active list; "
            "refusing Modal spawn."
        )
    for row in payload.get("active", []):
        if (
            isinstance(row, dict)
            and row.get("lane_id") == lane_id
            and (
                instance_job_id is None
                or row.get("instance_job_id") == instance_job_id
            )
        ):
            return True
    return False


def _ensure_dispatch_claim(
    repo_root,
    *,
    lane_id: str,
    label: str,
    gpu: str,
    agent: str,
    lane_global: bool = False,
) -> None:
    import subprocess
    import sys

    if _active_claim_exists(
        repo_root,
        lane_id=lane_id,
        instance_job_id=None if lane_global else label,
    ):
        raise SystemExit(
            f"FATAL: active duplicate Modal claim for lane_id={lane_id} "
            f"label={label}; refusing second spawn."
        )
    cmd = [
        sys.executable,
        "tools/claim_lane_dispatch.py",
        "claim",
        "--lane-id",
        lane_id,
        "--platform",
        "modal",
        "--instance-job-id",
        label,
        "--agent",
        agent,
        "--predicted-eta-utc",
        _compact_stamp(),
        "--status",
        "active_dispatching",
        "--notes",
        (
            "modal_train_lane.py direct claim before GPU spawn; "
            f"gpu={gpu}; score_claim=false; exact eval disabled"
        ),
    ]
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        raise SystemExit(
            f"FATAL: lane claim failed for lane_id={lane_id} label={label}; "
            "aborting before Modal GPU spawn."
        )


@app.local_entrypoint()
def main(
    lane_script: str,
    label: str,
    gpu: str = "T4",
    timeout_hours: float = 10.0,
    env_overrides: str = "",
    lane_id: str = "",
    trainer_module_path: str = "",
    cost_band_trainer: str = "",
    cost_band_epochs: int = 0,
    cost_band_batch_size: int = 0,
    cost_band_all_flags_on: bool = False,
    require_clean_head: bool = False,
    expected_mounted_code_git_head: str = "",
    expected_cost_usd: float = 0.0,
    preflight_first: bool = False,
    sentinel_files: str = "",
    agent: str = "claude:modal_train_lane",
):
    """Dispatch a lane training run on Modal.

    Args:
        lane_script: relative path like 'scripts/remote_lane_omega_hessian_qat.sh'
        label: short label used for output dir naming
        gpu: 'CPU', 'T4', 'A10G', 'A100', or 'H100'
        timeout_hours: max runtime (Modal hard kills at this)
        env_overrides: 'KEY1=val1,KEY2=val2' optional env to pass to lane
        trainer_module_path: explicit trainer metadata module from recipe
        cost_band_trainer: trainer path for cost posterior anchoring
        cost_band_epochs: epoch count for the cost posterior bucket
        cost_band_batch_size: batch size for the cost posterior bucket
        cost_band_all_flags_on: whether all required flags were threaded
        require_clean_head: refuse dispatch if working tree has uncommitted
            edits. Catalog #166 (PHASE-B1-PIVOT 2026-05-12): mid-edit
            dispatches silently ship pre-fix code to the Modal worker
            because ``add_local_dir`` snapshots the working tree, not HEAD.
        expected_mounted_code_git_head: mandatory 40-hex launcher-reviewed
            HEAD; must exactly match local ``git rev-parse HEAD`` before any
            CPU preflight or GPU provider call.
        expected_cost_usd: reviewed plan total or GPU-call ceiling persisted
            into canonical detached-call custody. Must be finite and positive
            for ``preflight_first`` paid dispatches.
        preflight_first: synchronously validate the staged driver and inputs on
            CPU under a 600-second absolute envelope before claiming or
            spawning the requested GPU function.
        sentinel_files: comma-separated relative paths whose worker-side
            sha256 will be written into ``modal_metadata.json`` on dispatch
            so post-mortem can verify the worker mounted the bytes the
            operator believed they were shipping. Defaults empty (no
            sentinel guard).
    """
    import json
    import os
    import sys
    import time
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    from tac.deploy.modal.auth_eval import function_call_id

    os.chdir(repo_root)

    # Catalog #360 sister of #339: register pre-spawn FATAL paths in the
    # canonical ledger so silent-no-spawn dispatches (STC v2 5-consecutive
    # anchor per OVERNIGHT-J) become structurally observable. We resolve a
    # provisional lane_id BEFORE the first FATAL path so the helper has
    # context for the synthetic call_id. Subsequent FATALs overwrite this
    # binding once a richer lane_id is derived.
    resolved_lane_id = lane_id.strip() or "unresolved"

    def _pre_spawn_fatal(reason: str, *, line_no: int, helper_source: str | None = None) -> None:
        """Catalog #360 inline helper — observability-then-exit pattern.

        Writes a pre_spawn_fatal event to the canonical ledger BEFORE
        sys.exit(2) fires so the harvester + operator dashboard can
        distinguish silent-no-spawn from successful dispatch. Per CLAUDE.md
        "Bugs must be permanently fixed AND self-protected against" +
        the OVERNIGHT-J STC v2 5th consecutive silent-no-spawn anchor.
        """
        try:
            from tac.deploy.modal.call_id_ledger import register_pre_spawn_fatal
            register_pre_spawn_fatal(
                label=label,
                lane_id=resolved_lane_id,
                fatal_reason=reason,
                exit_code=2,
                sys_exit_line_number=line_no,
                sys_exit_helper_source=helper_source,
                gpu=gpu,
                agent=agent,
            )
        except Exception as exc:
            # Catalog #360 helper is fail-OPEN, but defensive double-guard
            # ensures the FATAL exit is never blocked by observability.
            print(f"[Catalog #360] pre_spawn_fatal observability failed (non-blocking): {type(exc).__name__}: {exc}", file=sys.stderr)

    try:
        resolved_lane_id = _infer_lane_id(
            lane_script,
            lane_id,
            repo_root=repo_root,
        )
    except ValueError as exc:
        reason = f"FATAL: unsafe V9 lane script path: {exc}"
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(
            reason,
            line_no=1640,
            helper_source="v9_lane_script_classification",
        )
        sys.exit(2)

    if not (repo_root / lane_script).exists():
        reason = f"FATAL: lane script not found: {lane_script}"
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=1644, helper_source="lane_script_existence_check")
        sys.exit(2)

    v9_env_intent = bool(
        _raw_env_override_keys(env_overrides) & V9_UNIQUE_ENV_KEYS
    )
    v9_trainer_intent = _classify_v9_trainer_path(
        trainer_module_path,
        repo_root=repo_root,
    )
    v9_wrapper_intent = _lane_script_invokes_v9_trainer(
        lane_script,
        repo_root=repo_root,
    )
    is_v9_direct = _classify_v9_lane_script(
        lane_script,
        repo_root=repo_root,
    ) or (
        resolved_lane_id == V9_LANE_ID
        or v9_env_intent
        or v9_trainer_intent
        or v9_wrapper_intent
    )
    try:
        overrides = _parse_env_overrides(env_overrides, strict=is_v9_direct)
    except ValueError as exc:
        reason = f"FATAL: invalid V9 --env-overrides: {exc}"
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=1930, helper_source="v9_env_override_parser")
        sys.exit(2)

    try:
        requested_modal_gpu = _validated_modal_gpu_request(
            gpu,
            preflight_first=preflight_first,
        )
    except ValueError as exc:
        reason = f"FATAL: unsafe Modal resource request: {exc}"
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=1730, helper_source="gpu_validation")
        sys.exit(2)
    fn = run_lane_training_cpu

    cost_band_anchor = None
    if cost_band_trainer or cost_band_epochs or cost_band_batch_size or cost_band_all_flags_on:
        if not cost_band_trainer:
            reason = "FATAL: --cost-band-trainer is required when recording a cost-band anchor."
            print(reason, file=sys.stderr)
            _pre_spawn_fatal(reason, line_no=1736, helper_source="cost_band_trainer_validation")
            sys.exit(2)
        if cost_band_epochs <= 0:
            reason = "FATAL: --cost-band-epochs must be positive when recording a cost-band anchor."
            print(reason, file=sys.stderr)
            _pre_spawn_fatal(reason, line_no=1739, helper_source="cost_band_epochs_validation")
            sys.exit(2)
        if cost_band_batch_size <= 0:
            reason = "FATAL: --cost-band-batch-size must be positive when recording a cost-band anchor."
            print(reason, file=sys.stderr)
            _pre_spawn_fatal(reason, line_no=1742, helper_source="cost_band_batch_size_validation")
            sys.exit(2)
        cost_band_anchor = {
            "schema": "modal_training_cost_anchor_metadata_v1",
            "trainer": cost_band_trainer,
            "epochs": int(cost_band_epochs),
            "batch_size": int(cost_band_batch_size),
            "all_flags_on": bool(cost_band_all_flags_on),
            "score_claim": False,
            "promotion_eligible": False,
            "notes": "metadata_only_until_modal_recovery_appends_elapsed_cost_anchor",
        }

    print(f"=== modal_train_lane: {lane_script} → {label} on {gpu} ===")
    try:
        max_seconds, hard_timeout_seconds = _validated_modal_timeouts(timeout_hours)
    except (TypeError, ValueError) as exc:
        reason = f"FATAL: invalid --timeout-hours {timeout_hours!r}: {exc}"
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=1796, helper_source="timeout_validation")
        sys.exit(2)
    print(f"  per-lane timeout: {max_seconds}s ({timeout_hours:.6g}h requested)")
    print(
        "  invocation hard timeout: "
        f"{hard_timeout_seconds}s (includes {MODAL_FINALIZATION_GRACE_SECONDS}s "
        "finalization grace)"
    )
    try:
        validated_expected_cost_usd = _validated_expected_cost_usd(
            expected_cost_usd,
            preflight_first=preflight_first,
        )
    except (TypeError, ValueError) as exc:
        reason = f"FATAL: invalid --expected-cost-usd {expected_cost_usd!r}: {exc}"
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=1950, helper_source="expected_cost_usd")
        sys.exit(2)
    print(f"  expected cost ceiling: ${validated_expected_cost_usd:.6f}")
    try:
        provider_resource_ceiling_usd = _validated_provider_cost_ceiling(
            requested_gpu=requested_modal_gpu,
            hard_timeout_seconds=hard_timeout_seconds,
            preflight_first=preflight_first,
            expected_cost_usd=validated_expected_cost_usd,
        )
    except ValueError as exc:
        reason = f"FATAL: unsafe provider resource/cost envelope: {exc}"
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=1958, helper_source="provider_cost_ceiling")
        sys.exit(2)
    print(f"  derived provider resource ceiling: ${provider_resource_ceiling_usd:.6f}")
    mounted_code_git_head = _git_value(repo_root, "rev-parse", "HEAD")
    mounted_code_git_branch = _git_value(repo_root, "branch", "--show-current")
    if mounted_code_git_head == "unknown" or mounted_code_git_branch == "unknown":
        reason = (
            "FATAL: unable to resolve mounted git custody for Modal training "
            f"(head={mounted_code_git_head!r}, branch={mounted_code_git_branch!r})"
        )
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=1708, helper_source="git_custody_resolution")
        sys.exit(2)
    try:
        _validate_expected_mounted_head(
            expected_mounted_code_git_head,
            mounted_code_git_head,
        )
    except ValueError as exc:
        reason = f"FATAL: mounted-code HEAD custody validation failed: {exc}"
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=1904, helper_source="expected_mounted_head")
        sys.exit(2)

    # Catalog #166: working-tree dirty-tree summary + optional fail-closed.
    # DX-POLISH-WAVE 2026-05-15 (Catalog #238 / DX-1): the FATAL message body
    # is built by `_format_dx_polish_actionable_fatal`, which categorizes the
    # dirty paths into actionable buckets (sister-subagent research ledger /
    # operator-owned source / build artifact / ...) and prints the
    # next-steps that match each bucket (commit through serializer / set
    # Catalog #202 bypass / GC helper / smoke-before-full path / explicit
    # operator override).
    dirty_tree = _git_dirty_tree_summary(repo_root)
    if require_clean_head and dirty_tree["dirty"]:
        reason = _format_dx_polish_actionable_fatal(dirty_tree, mounted_code_git_head)
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason[:1000], line_no=1786, helper_source="require_clean_head_dirty_tree")
        sys.exit(2)
    if dirty_tree["dirty"]:
        print(
            f"WARN [Catalog #166]: working tree has {dirty_tree['dirty_paths_count']} "
            "uncommitted edit(s); Modal will mount the dirty snapshot, NOT "
            f"HEAD ({mounted_code_git_head[:12]}). Pass --require-clean-head "
            "to fail-closed instead. See `_format_dx_polish_actionable_fatal` "
            "for the categorized + next-steps body the FATAL path emits."
        )

    try:
        _validate_v9_direct_dispatch_policy(
            lane_script=lane_script,
            lane_id=resolved_lane_id,
            label=label,
            gpu=gpu,
            requested_gpu=requested_modal_gpu,
            mounted_code_git_branch=mounted_code_git_branch,
            preflight_first=preflight_first,
            require_clean_head=require_clean_head,
            max_seconds=max_seconds,
            hard_timeout_seconds=hard_timeout_seconds,
            expected_cost_usd=validated_expected_cost_usd,
            sentinel_files=sentinel_files,
            overrides=overrides,
            trainer_module_path=trainer_module_path,
            repo_root=repo_root,
        )
    except ValueError as exc:
        reason = f"FATAL: V9 direct-dispatch policy refused: {exc}"
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=2100, helper_source="v9_direct_dispatch_policy")
        sys.exit(2)

    claims_path = repo_root / ".omx/state/active_lane_dispatch_claims.md"
    if not claims_path.is_file():
        reason = f"FATAL: dispatch claims ledger missing: {claims_path}"
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=1747, helper_source="active_lane_dispatch_claims_missing")
        sys.exit(2)
    claim_ledger_bytes = claims_path.read_bytes()

    # Catalog #166: sentinel-file SHA-256 ledger so post-mortem can prove the
    # worker mounted the bytes the operator believed were being shipped.
    sentinel_relpaths = [
        s.strip() for s in sentinel_files.split(",") if s.strip()
    ]
    sentinel_sha256_local: dict = {}
    for rel in sentinel_relpaths:
        p = repo_root / rel
        if not p.is_file():
            sentinel_sha256_local[rel] = "MISSING_LOCAL"
            continue
        import hashlib as _hashlib
        sentinel_sha256_local[rel] = _hashlib.sha256(p.read_bytes()).hexdigest()

    # WAVE-3 (2026-05-16) deeper structural fix per STC v2 FIX CRITICAL
    # DEEPER FINDING (commit 7dd8a5412): derive trainer module path from
    # lane_script + collect trainer-declared TIER_1_EXTRA_MOUNT_PATHS /
    # MODAL_EXTRA_MOUNT_PATHS / required_input_file=True defaults into bytes
    # payload. The worker materializes each entry under /tmp/pact/<rel>
    # after the structural copy. This makes trainer-side declarations
    # STRUCTURALLY EFFECTIVE for the generic Modal dispatcher (previously
    # `trainer_module_path=None` made them inert).
    substrate_id_from_lane = _substrate_id_from_lane_script(lane_script)
    explicit_trainer_module_path = None
    try:
        explicit_trainer_module_path = _normalize_trainer_module_path(
            trainer_module_path,
            repo_root,
        )
    except RuntimeError as exc:
        reason = f"FATAL: {exc}"
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=1781, helper_source="_normalize_trainer_module_path")
        sys.exit(2)
    derived_trainer_module_path = _derive_trainer_module_path(lane_script, repo_root)
    if (
        explicit_trainer_module_path is not None
        and derived_trainer_module_path is not None
        and explicit_trainer_module_path != derived_trainer_module_path.resolve()
    ):
        reason = (
            "FATAL: explicit --trainer-module-path conflicts with lane_script "
            "derived trainer module: "
            f"explicit={explicit_trainer_module_path.relative_to(repo_root)} "
            f"derived={derived_trainer_module_path.relative_to(repo_root)}"
        )
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=1795, helper_source="trainer_module_path_conflict")
        sys.exit(2)
    trainer_module_path_resolved = (
        explicit_trainer_module_path or derived_trainer_module_path
    )
    if trainer_module_path_resolved is not None:
        trainer_metadata_source = (
            "explicit_recipe"
            if explicit_trainer_module_path is not None
            else "lane_script_derived"
        )
    else:
        trainer_metadata_source = "none"
    if substrate_id_from_lane is not None and trainer_module_path_resolved is None:
        reason = (
            f"FATAL: substrate lane_script {lane_script!r} implies canonical "
            "trainer module "
            f"experiments/train_substrate_{substrate_id_from_lane}.py, but "
            "that file does not exist. Refusing Modal dispatch because "
            "trainer-side TIER_1_EXTRA_MOUNT_PATHS / required_input_file "
            "metadata cannot be consumed."
        )
        print(reason, file=sys.stderr)
        _pre_spawn_fatal(reason, line_no=1817, helper_source="substrate_trainer_module_missing")
        sys.exit(2)
    if trainer_module_path_resolved is not None:
        print(
            f"[modal-train-lane][WAVE-3] derived trainer module from "
            f"{trainer_metadata_source}: "
            f"{trainer_module_path_resolved.relative_to(repo_root)}"
        )
        try:
            trainer_extra_mount_payload = _collect_trainer_extra_mount_payload(
                trainer_module_path_resolved,
                repo_root,
                fail_on_import_error=True,
                fail_on_missing_paths=True,
            )
        except RuntimeError as exc:
            reason = (
                f"FATAL: {exc}. Refusing Modal dispatch because "
                "substrate-named lane scripts must consume trainer-side "
                "TIER_1_EXTRA_MOUNT_PATHS / required_input_file metadata."
            )
            print(reason, file=sys.stderr)
            _pre_spawn_fatal(reason, line_no=1838, helper_source="_collect_trainer_extra_mount_payload")
            sys.exit(2)
        if trainer_extra_mount_payload:
            print(
                f"[modal-train-lane][WAVE-3] staging "
                f"{len(trainer_extra_mount_payload)} trainer-declared extra "
                f"mount path(s) as bytes payload: "
                f"{sorted(trainer_extra_mount_payload.keys())}"
            )
        else:
            print(
                "[modal-train-lane][WAVE-3] trainer module has no extra-mount "
                "declarations (TIER_1_EXTRA_MOUNT_PATHS empty + no "
                "required_input_file under experiments/results/**)"
            )
    else:
        trainer_extra_mount_payload = {}
        # Non-substrate lane scripts (e.g. legacy `scripts/remote_lane_*.sh`
        # without the `_substrate_` infix) follow self-bootstrap convention.
        print(
            f"[modal-train-lane][WAVE-3] lane_script {lane_script!r} does not "
            "follow substrate naming convention "
            "(scripts/remote_lane_substrate_<id>.sh) — relying on lane "
            "script self-bootstrap for any required inputs."
        )
    import hashlib as _hashlib

    trainer_extra_mount_payload_manifest = [
        {
            "rel_path": rel,
            "bytes": len(data),
            "sha256": _hashlib.sha256(data).hexdigest(),
        }
        for rel, data in sorted(trainer_extra_mount_payload.items())
    ]

    # This lock is deliberately distinct from the claim-ledger lock and Git
    # serializer locks. It spans the paid CPU preflight through detached-call
    # ledger registration, so a second local launcher blocks, then observes
    # the first launcher's active claim before spending on its own preflight.
    dispatch_guard_fh = _acquire_local_dispatch_guard(
        repo_root,
        lane_id=resolved_lane_id,
        label=label,
        lane_global=is_v9_direct,
    )

    # Refuse a known duplicate before even the bounded CPU preflight spends.
    # _ensure_dispatch_claim repeats this check after preflight to close the
    # race between the read-only check and the eventual claim append.
    if _active_claim_exists(
        repo_root,
        lane_id=resolved_lane_id,
        instance_job_id=None if is_v9_direct else label,
    ):
        _release_local_dispatch_guard(dispatch_guard_fh)
        raise SystemExit(
            f"FATAL: active duplicate Modal claim for lane_id={resolved_lane_id} "
            f"label={label}; refusing preflight and GPU spawn."
        )

    import secrets

    dispatch_nonce = secrets.token_hex(32)
    cpu_preflight_receipt = ""
    if preflight_first:
        preflight_label = f"{label}__preflight"
        preflight_deadline_unix_s = time.time() + MODAL_PREFLIGHT_HARD_TIMEOUT_SECONDS
        preflight_overrides = {
            **overrides,
            "WITNESS_PREFLIGHT_ONLY": "1",
        }
        try:
            preflight_result = run_lane_training_cpu.with_options(
                timeout=MODAL_PREFLIGHT_HARD_TIMEOUT_SECONDS,
                retries=0,
                cpu=MODAL_GPU_CPU_REQUEST_LIMIT,
                memory=MODAL_GPU_MEMORY_REQUEST_LIMIT_MB,
            ).remote(
                lane_script,
                preflight_label,
                preflight_overrides,
                claim_ledger_bytes,
                mounted_code_git_head,
                mounted_code_git_branch,
                sentinel_sha256_local,
                MODAL_PREFLIGHT_CHILD_TIMEOUT_SECONDS,
                preflight_deadline_unix_s,
                trainer_extra_mount_payload,
                "CPU",
                "cpu_preflight",
                dispatch_nonce,
                "",
                label,
            )
        except Exception as exc:
            _release_local_dispatch_guard(dispatch_guard_fh)
            raise SystemExit(
                "FATAL: CPU preflight failed before GPU claim/spawn: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if (
            not isinstance(preflight_result, dict)
            or preflight_result.get("returncode") != 0
            or preflight_result.get("timed_out") is True
        ):
            _release_local_dispatch_guard(dispatch_guard_fh)
            raise SystemExit(
                "FATAL: CPU preflight refused GPU claim/spawn: "
                f"receipt={preflight_result!r}"
            )
        cpu_preflight_receipt = str(
            preflight_result.get("cpu_preflight_receipt", "")
        )
        expected_preflight_receipt = _derive_cpu_preflight_receipt(
            dispatch_nonce=dispatch_nonce,
            lane_script=lane_script,
            label=label,
            mounted_code_git_head=mounted_code_git_head,
        )
        if cpu_preflight_receipt != expected_preflight_receipt:
            _release_local_dispatch_guard(dispatch_guard_fh)
            raise SystemExit(
                "FATAL: CPU preflight returned no matching dispatch receipt; "
                "refusing GPU claim/spawn"
            )
        print(
            "[modal_train_lane] CPU preflight green before GPU claim/spawn "
            f"label={preflight_label}"
        )

    try:
        _ensure_dispatch_claim(
            repo_root,
            lane_id=resolved_lane_id,
            label=label,
            gpu=gpu,
            agent=agent,
            lane_global=is_v9_direct,
        )
    except BaseException:
        _release_local_dispatch_guard(dispatch_guard_fh)
        raise
    # Refresh after the claim so the GPU worker receives its own active row.
    claim_ledger_bytes = claims_path.read_bytes()

    # Resolve ledger/provenance dependencies before paid work starts. Once
    # .spawn() returns, only call-id extraction and canonical registration may
    # run before detached-call custody is durable.
    from tac.deploy.modal.call_id_ledger import (
        LedgerRegistrationFailedError,
        register_dispatched_call_id_fail_closed,
    )

    upstream_sha: str | None
    try:
        from tac.contest_compliance import compute_upstream_snapshot_sha256

        upstream_sha = compute_upstream_snapshot_sha256()
    except Exception:  # pragma: no cover - best-effort provenance enrichment
        upstream_sha = None

    # CRITICAL: use .spawn() not .remote() for detached runs.
    # `.remote()` is cancelled when the local CLI disconnects, even with
    # --detach (Modal's warning: ".remote() calls in detached apps may be
    # canceled when the local caller disconnects. Use .spawn() for detached
    # or background work."). Tonight's first 3 dispatches all got killed at
    # 4-6s by this exact issue.
    #
    # .spawn() returns a FunctionCall handle. We save it so the recovery script
    # can poll later through Modal's Python API. Modal 1.4 no longer exposes a
    # direct FunctionCall result CLI for this path.
    # Legacy WAVE-3 static-contract anchor: fn.spawn(...,
    # trainer_extra_mount_payload) remains semantically represented by the
    # invocation-scoped with_options(...).spawn call below.
    billing_deadline_unix_s = time.time() + hard_timeout_seconds
    invocation_options = {
        "timeout": hard_timeout_seconds,
        "retries": 0,
        "cpu": MODAL_GPU_CPU_REQUEST_LIMIT,
        "memory": MODAL_GPU_MEMORY_REQUEST_LIMIT_MB,
    }
    if requested_modal_gpu is not None:
        # The sole GPU allocation point is deliberately below local policy,
        # custody, CPU preflight, duplicate protection, and canonical claim.
        invocation_options["gpu"] = requested_modal_gpu
    # #513 SINGLE-FLIGHT pre-spawn guard (operator binding 2026-07-15): refuse
    # when ANY live Modal work exists (call-id ledger / claims file / live
    # `modal app list`); operator-override escape via env
    # TAC_MODAL_SINGLE_FLIGHT_FORCE_RATIONALE (quote it in the claim notes).
    # Release the local dispatch guard on refusal so the fcntl lock never
    # outlives a refused dispatch.
    from tac.deploy.modal.single_flight import assert_modal_single_flight
    try:
        assert_modal_single_flight(label=label, lane_id=resolved_lane_id)
    except BaseException:
        _release_local_dispatch_guard(dispatch_guard_fh)
        raise
    try:
        fn_call = fn.with_options(**invocation_options).spawn(
            lane_script,
            label,
            overrides,
            claim_ledger_bytes,
            mounted_code_git_head,
            mounted_code_git_branch,
            sentinel_sha256_local,
            max_seconds,
            billing_deadline_unix_s,
            trainer_extra_mount_payload,
            requested_modal_gpu or "CPU",
            "gpu_dispatch" if requested_modal_gpu is not None else "cpu_dispatch",
            dispatch_nonce,
            cpu_preflight_receipt if requested_modal_gpu is not None else "",
            label,
        )
    except BaseException:
        _release_local_dispatch_guard(dispatch_guard_fh)
        raise
    try:
        call_id = function_call_id(fn_call)
    except Exception as exc:
        _cancel_spawned_call_best_effort(fn_call, reason="call-id extraction failure")
        _release_local_dispatch_guard(dispatch_guard_fh)
        raise SystemExit(
            "FATAL: Modal .spawn() succeeded but call-id extraction failed; "
            "best-effort cancellation attempted"
        ) from exc

    dispatched_at = __import__("datetime").datetime.now().isoformat()

    # Register before any per-dispatch sentinel/metadata write. The canonical
    # ledger is the minimum sufficient detached-call custody surface.
    try:
        register_dispatched_call_id_fail_closed(
            call_id=call_id,
            lane_id=resolved_lane_id,
            label=label,
            dispatched_at_utc=dispatched_at,
            platform="modal",
            gpu=gpu,
            expected_cost_usd=validated_expected_cost_usd,
            max_seconds=max_seconds,
            hard_timeout_seconds=hard_timeout_seconds,
            billing_deadline_unix_s=billing_deadline_unix_s,
            finalization_grace_seconds=MODAL_FINALIZATION_GRACE_SECONDS,
            gpu_cpu_request_limit=MODAL_GPU_CPU_REQUEST_LIMIT,
            gpu_memory_request_limit_mb=MODAL_GPU_MEMORY_REQUEST_LIMIT_MB,
            mounted_code_git_head=mounted_code_git_head,
            agent=agent,
            upstream_snapshot_sha256=upstream_sha,
        )
    except LedgerRegistrationFailedError as exc:
        _cancel_spawned_call_best_effort(fn_call, reason="call-id ledger registration failure")
        _release_local_dispatch_guard(dispatch_guard_fh)
        print(
            f"FATAL: spawned Modal call_id={call_id} could not be registered; "
            f"best-effort cancellation attempted: {exc}",
            file=sys.stderr,
        )
        sys.exit(13)

    _release_local_dispatch_guard(dispatch_guard_fh)

    print(f"[modal_train_lane] dispatch_completed call_id={call_id}")
    print(f"\n✓ DISPATCHED via .spawn() — call_id={call_id}")
    recover_command = (
        # Check 101 scans source text for the contiguous string
        # `experiments/modal_recover_lane.py --call-id` — keep it unsplit.
        ".venv/bin/python experiments/modal_recover_lane.py --call-id "
        f"{call_id}"
    )
    harvest_command = (
        ".venv/bin/python tools/harvest_modal_calls.py --from-ledger "
        f"--call-id {call_id} --execute"
    )
    cancel_command = (
        ".venv/bin/python -c \"import modal; "
        f"modal.FunctionCall.from_id('{call_id}').cancel(terminate_containers=True)\""
    )
    print(f"  Exact recover:  {recover_command}")
    print(f"  Exact harvest:  {harvest_command}")
    print(f"  Emergency stop: {cancel_command}")
    print("  Stream logs:    .venv/bin/modal app logs <app-id> (see modal app list)")
    print(
        "\n  Local entrypoint exiting; remote training hard-capped at "
        f"{hard_timeout_seconds}s."
    )
    print(f"  Live volume:    .venv/bin/modal volume ls {RESULTS_VOL} {label}/")
    print(f"  Download live:  .venv/bin/modal volume get {RESULTS_VOL} {label}/ ./modal_{label}/")

    # Save redundant convenience files only after canonical ledger custody.
    sentinel_dir = repo_root / "experiments" / "results" / f"lane_{label}_modal"
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    (sentinel_dir / "modal_call_id.txt").write_text(call_id + "\n")
    metadata = {
        "lane_script": lane_script,
        "lane_id": resolved_lane_id,
        "label": label,
        "gpu": gpu,
        "expected_cost_usd": validated_expected_cost_usd,
        "provider_resource_ceiling_usd": provider_resource_ceiling_usd,
        "max_seconds": max_seconds,
        "hard_timeout_seconds": hard_timeout_seconds,
        "billing_deadline_unix_s": billing_deadline_unix_s,
        "finalization_grace_seconds": MODAL_FINALIZATION_GRACE_SECONDS,
        "gpu_cpu_request_limit": MODAL_GPU_CPU_REQUEST_LIMIT,
        "gpu_memory_request_limit_mb": MODAL_GPU_MEMORY_REQUEST_LIMIT_MB,
        "call_id": call_id,
        "exact_recover_command": recover_command,
        "exact_harvest_command": harvest_command,
        "exact_cancel_command": cancel_command,
        "wrapper_score_claim": False,
        "inline_auth_eval_contract_required": True,
        "auth_eval_device": "cpu",
        "auth_eval_advisory_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "live_volume": RESULTS_VOL,
        "live_volume_prefix": f"{label}/",
        "dispatched_at": dispatched_at,
        # Catalog #166: HEAD parity ledger so post-mortem can prove the worker
        # mounted the bytes the operator believed were being shipped (and so
        # the canary subagent doesn't have to GUESS via traceback line numbers
        # whether the failure was stale-mount or pre-fix-dispatch).
        "mounted_code_git_head": mounted_code_git_head,
        "mounted_code_git_branch": mounted_code_git_branch,
        "working_tree_dirty": dirty_tree["dirty"],
        "working_tree_dirty_paths_count": dirty_tree["dirty_paths_count"],
        "working_tree_dirty_summary": dirty_tree["summary"],
        "require_clean_head": bool(require_clean_head),
        "sentinel_files_local_sha256": sentinel_sha256_local,
        "trainer_module_path_resolved": (
            str(trainer_module_path_resolved.relative_to(repo_root))
            if trainer_module_path_resolved is not None
            else None
        ),
        "trainer_metadata_source": trainer_metadata_source,
        "trainer_extra_mount_payload_file_count": len(
            trainer_extra_mount_payload_manifest
        ),
        "trainer_extra_mount_payload_total_bytes": sum(
            item["bytes"] for item in trainer_extra_mount_payload_manifest
        ),
        "trainer_extra_mount_payload_manifest": trainer_extra_mount_payload_manifest,
        # Schema marker so consumers (recovery / harvest / cost-band anchor)
        # can detect Catalog #166 metadata version.
        "metadata_schema": "modal_train_lane_dispatch_metadata_v2_catalog166",
    }
    if cost_band_anchor is not None:
        metadata["cost_band_anchor"] = cost_band_anchor
    (sentinel_dir / "modal_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"  call_id saved:  {sentinel_dir}/modal_call_id.txt")

    print(
        "  ledger appended: .omx/state/modal_call_id_ledger.jsonl "
        "(Catalog #245 canonical Modal call_id ledger; Catalog #339 fail-closed)"
    )
    print("\n  Use experiments/modal_recover_lane.py to fetch artifacts when complete.")
