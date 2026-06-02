# SPDX-License-Identifier: MIT
"""Readiness report for local Apple acceleration substrates."""

from __future__ import annotations

import importlib.util
import platform
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

APPLE_LOCAL_COMPUTE_READINESS_SCHEMA = "apple_local_compute_readiness.v1"
APPLE_LOCAL_COMPUTE_AXIS_TAG = "[macOS-local-acceleration:false-authority]"


def build_apple_local_compute_readiness(
    *,
    substrate_ids: Iterable[str] = ("hi_nerv", "snerv"),
    backend_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a false-authority readiness report for local dev acceleration."""

    overrides = dict(backend_overrides or {})
    backend_status = {
        "mlx": dict(overrides.get("mlx") or _probe_mlx()),
        "torch_mps": dict(overrides.get("torch_mps") or _probe_torch_mps()),
        "numpy_accelerate": dict(
            overrides.get("numpy_accelerate") or _probe_numpy_accelerate()
        ),
        "hf_accelerate": dict(overrides.get("hf_accelerate") or _probe_hf_accelerate()),
    }
    recommendation = _recommend_backend(backend_status)
    blockers = _readiness_blockers(backend_status, recommendation)
    return {
        "schema": APPLE_LOCAL_COMPUTE_READINESS_SCHEMA,
        "axis_tag": APPLE_LOCAL_COMPUTE_AXIS_TAG,
        "authority": "false_authority_local_dev_velocity_only",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "substrate_ids": [str(item) for item in substrate_ids],
        "backend_status": backend_status,
        "recommended_dev_velocity_backend": recommendation["backend"],
        "recommended_dev_velocity_role": recommendation["role"],
        "recommended_runner_notes": recommendation["notes"],
        "apple_gpu_ready": bool(
            backend_status["mlx"].get("metal_available")
            or backend_status["torch_mps"].get("mps_available")
        ),
        "mlx_metal_ready": bool(backend_status["mlx"].get("metal_available")),
        "torch_mps_ready": bool(backend_status["torch_mps"].get("mps_available")),
        "numpy_accelerate_ready": bool(
            backend_status["numpy_accelerate"].get("accelerate_framework_present")
        ),
        "allowed_uses": [
            "local_training_dev_velocity",
            "local_gradient_and_vjp_iteration",
            "local_candidate_prefilter_after_drift_calibration",
            "receiver_archive_byte_custody_when_export_path_is_byte_closed",
        ],
        "forbidden_uses": [
            "contest_score_claim",
            "leaderboard_rank_or_kill",
            "cpu_cuda_axis_conversion",
            "promotion_without_byte_closed_archive_runtime_and_exact_replay",
        ],
        "drift_policy": {
            "status": "measure_and_gate",
            "calibration_required_before_spend_triage": True,
            "promotion_requires_exact_contest_cpu_or_cuda": True,
            "drift_fields_must_be_attached": [
                "candidate_archive_sha256",
                "runtime_tree_sha256",
                "inflated_outputs_aggregate_sha256",
                "local_backend",
                "paired_auth_axis",
                "component_delta_pose",
                "component_delta_seg",
                "rate_delta_bytes",
            ],
        },
        "hinerv_binding": {
            "mlx_renderer_module": "tac.substrates.hi_nerv.mlx_renderer",
            "archive_ladder_tool": "tools/build_hinerv_archive_size_ladder.py",
            "current_priority": (
                "use MLX for score-aware trainer/export velocity; keep archive "
                "replay false-authority until scorer deltas are paired"
            ),
        },
        "snerv_binding": {
            "current_priority": (
                "port source-faithful MFU/HFR/TUB and wavelet primitives onto "
                "MLX-first NumPy-portable kernels before more LF-byte sweeps"
            ),
            "blockers": [
                "snerv_mlx_native_train_export_missing",
                "snerv_source_faithful_mfu_hfr_tub_missing",
            ],
        },
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def render_apple_local_compute_readiness_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact readiness report."""

    lines = [
        "# Apple Local Compute Readiness",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        f"Recommended backend: `{report.get('recommended_dev_velocity_backend')}`",
        "",
        "| backend | available | notes |",
        "|---|---:|---|",
    ]
    for backend, status in (report.get("backend_status") or {}).items():
        if not isinstance(status, Mapping):
            continue
        available = bool(
            status.get("available")
            or status.get("metal_available")
            or status.get("mps_available")
            or status.get("accelerate_framework_present")
        )
        lines.append(
            "| {backend} | {available} | {notes} |".format(
                backend=backend,
                available=available,
                notes=status.get("notes") or status.get("reason") or "",
            )
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.get("blockers") or ():
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def _probe_mlx() -> dict[str, Any]:
    if importlib.util.find_spec("mlx") is None:
        return {
            "available": False,
            "metal_available": False,
            "reason": "mlx_package_missing",
            "install_hint": "uv pip install --python .venv/bin/python mlx",
        }
    try:
        import mlx
        import mlx.core as mx

        metal_available = bool(mx.metal.is_available())
        try:
            info = dict(mx.device_info())
        except Exception:
            try:
                info = dict(mx.metal.device_info())
            except Exception as exc:
                info = {"device_info_error": f"{type(exc).__name__}: {exc}"}
        return {
            "available": True,
            "version": getattr(mlx, "__version__", None),
            "metal_available": metal_available,
            "default_device": str(mx.default_device()),
            "device_info": info,
            "notes": "MLX/Metal local research-signal backend; not contest authority",
        }
    except Exception as exc:
        return {
            "available": False,
            "metal_available": False,
            "reason": f"mlx_import_or_metal_probe_failed:{type(exc).__name__}",
            "detail": str(exc),
        }


def _probe_torch_mps() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {
            "available": False,
            "mps_available": False,
            "reason": "torch_package_missing",
        }
    try:
        import torch

        return {
            "available": True,
            "version": getattr(torch, "__version__", None),
            "mps_available": bool(torch.backends.mps.is_available()),
            "notes": "Torch MPS is local research-signal only; scorer drift must be measured",
        }
    except Exception as exc:
        return {
            "available": False,
            "mps_available": False,
            "reason": f"torch_mps_probe_failed:{type(exc).__name__}",
            "detail": str(exc),
        }


def _probe_numpy_accelerate() -> dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:
        return {
            "available": False,
            "accelerate_framework_present": False,
            "reason": f"numpy_import_failed:{type(exc).__name__}",
            "detail": str(exc),
        }
    accelerate_path = Path("/System/Library/Frameworks/Accelerate.framework")
    return {
        "available": True,
        "numpy_version": getattr(np, "__version__", None),
        "accelerate_framework_present": accelerate_path.exists(),
        "accelerate_framework_path": accelerate_path.as_posix(),
        "notes": "Apple Accelerate can help CPU NumPy kernels but is not a scorer authority",
    }


def _probe_hf_accelerate() -> dict[str, Any]:
    if importlib.util.find_spec("accelerate") is None:
        return {
            "available": False,
            "reason": "python_accelerate_package_missing",
            "notes": "Only needed for source-faithful HiNeRV OSS Accelerate launches",
        }
    try:
        import accelerate

        return {
            "available": True,
            "version": getattr(accelerate, "__version__", None),
            "notes": "HuggingFace Accelerate present for OSS-style launch parity",
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"accelerate_import_failed:{type(exc).__name__}",
            "detail": str(exc),
        }


def _recommend_backend(status: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    if status["mlx"].get("metal_available") is True:
        return {
            "backend": "mlx_metal",
            "role": "primary_local_dev_velocity_backend",
            "notes": "Use for SNeRV/HiNeRV local training, gradients, VJP, and export smoke",
        }
    if status["torch_mps"].get("mps_available") is True:
        return {
            "backend": "torch_mps",
            "role": "fallback_local_dev_velocity_backend",
            "notes": "Use only when MLX is unavailable; attach drift blockers",
        }
    if status["numpy_accelerate"].get("accelerate_framework_present") is True:
        return {
            "backend": "numpy_accelerate_cpu",
            "role": "portable_cpu_kernel_backend",
            "notes": "Use for NumPy reference, codec, and receiver primitives",
        }
    return {
        "backend": "numpy_cpu",
        "role": "last_resort_portable_backend",
        "notes": "No Apple GPU acceleration detected",
    }


def _readiness_blockers(
    status: Mapping[str, Mapping[str, Any]],
    recommendation: Mapping[str, str],
) -> list[str]:
    blockers = [
        "macos_local_acceleration_false_authority",
        "contest_cpu_cuda_exact_eval_required_for_promotion",
    ]
    if status["mlx"].get("metal_available") is not True:
        blockers.append("mlx_metal_not_ready")
    if status["hf_accelerate"].get("available") is not True:
        blockers.append("python_accelerate_package_missing_for_oss_hinerv_launch")
    if recommendation.get("backend") != "mlx_metal":
        blockers.append("primary_mlx_metal_backend_not_selected")
    return blockers


__all__ = [
    "APPLE_LOCAL_COMPUTE_READINESS_SCHEMA",
    "build_apple_local_compute_readiness",
    "render_apple_local_compute_readiness_markdown",
]
