#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe evaluator loader drift between CUDA DALI/NVDEC and CPU PyAV.

The public scorer changes two things between ``--device cuda`` and
``--device cpu``:

* CUDA ground-truth video path: ``DaliVideoDataset`` with GPU/NVDEC decode.
* CPU ground-truth video path: ``AVVideoDataset`` with PyAV/FFmpeg decode.

This tool always records the intended 2x2 diagnostic cells:

* CPU+AV
* CUDA+DALI
* CUDA+AV/shared-input
* CPU+DALI

The default comparison remains the decoded RGB uint8 tensor diff before
PoseNet or SegNet. With ``--run-forward-cells`` it can also run diagnostic
PoseNet/SegNet forward cells on shared input tensors to separate decoder/input
bytes from network forward/kernel drift. It is diagnostic only and never a
score claim. If CUDA/DALI is unavailable it still writes a non-promotable plan
artifact so the exact remote command is auditable.

With ``--save-shared-input-dir`` the decoded RGB tensors are also saved as
non-promotable ``eval_loader_shared_input_tensor.v1`` artifacts. Those files
are the explicit bridge into ``experiments/dump_scorer_activations.py
--shared-input-tensor`` and prevent future CUDA xray work from trying to
instantiate ``AVVideoDataset(device='cuda')``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"
COMPARISON_UNAVAILABLE_MISSING_PREREQUISITE = "missing_prerequisite"
COMPARISON_UNAVAILABLE_PROBE_RUNTIME_ERROR = "probe_runtime_error"
DIAGNOSTIC_REMOTE_LANE_ID = "eval_loader_drift_2x2_cuda_dali"
REMOTE_ARTIFACT_PLACEHOLDER = "${ARTIFACT_DIR}/eval_loader_drift_2x2_cuda_dali.json"
SHARED_INPUT_TENSOR_SCHEMA = "eval_loader_shared_input_tensor.v1"
SHARED_INPUT_TENSOR_ROLE = "raw_rgb_uint8_before_posenet_segnet"

RAW_DECODER_REQUIRED_ARTIFACTS = (
    "upstream_frame_utils_py",
    "pyav_available",
    "cuda_available",
    "dali_available",
    "video_names_file_exists",
    "data_dir_exists",
    "sample_videos_exist",
)

AV_LOADER_REQUIRED_ARTIFACTS = (
    "upstream_frame_utils_py",
    "pyav_available",
    "video_names_file_exists",
    "data_dir_exists",
    "sample_videos_exist",
)

DALI_LOADER_REQUIRED_ARTIFACTS = (
    "upstream_frame_utils_py",
    "cuda_available",
    "dali_available",
    "cuda_dali_runtime_available",
    "video_names_file_exists",
    "data_dir_exists",
    "sample_videos_exist",
)

FORWARD_MODEL_REQUIRED_ARTIFACTS = (
    "upstream_modules_py",
    "posenet_weights_exist",
    "segnet_weights_exist",
    "timm_available",
    "einops_available",
    "safetensors_torch_available",
    "segmentation_models_pytorch_available",
)

INTENDED_CELL_SPECS = (
    {
        "cell_id": "cpu_av",
        "label": "CPU+AV",
        "loader_key": "av",
        "loader_class": "AVVideoDataset",
        "decoder_backend": "PyAV/FFmpeg",
        "raw_decode_device": "cpu",
        "forward_device": "cpu",
        "shared_input": False,
        "official_evaluator_shape": True,
        "purpose": "official CPU evaluator-shaped decoder and CPU forward cell",
    },
    {
        "cell_id": "cuda_dali",
        "label": "CUDA+DALI",
        "loader_key": "dali",
        "loader_class": "DaliVideoDataset",
        "decoder_backend": "DALI/NVDEC",
        "raw_decode_device": "cuda",
        "forward_device": "cuda",
        "shared_input": False,
        "official_evaluator_shape": True,
        "purpose": "official CUDA evaluator-shaped decoder and CUDA forward cell",
    },
    {
        "cell_id": "cuda_av_shared_input",
        "label": "CUDA+AV/shared-input",
        "loader_key": "av",
        "loader_class": "AVVideoDataset",
        "decoder_backend": "PyAV/FFmpeg",
        "raw_decode_device": "cpu",
        "forward_device": "cuda",
        "shared_input": True,
        "shared_input_reference_cell": "cpu_av",
        "official_evaluator_shape": False,
        "purpose": "CUDA forward on PyAV bytes to isolate forward/kernel drift",
    },
    {
        "cell_id": "cpu_dali",
        "label": "CPU+DALI",
        "loader_key": "dali",
        "loader_class": "DaliVideoDataset",
        "decoder_backend": "DALI/NVDEC",
        "raw_decode_device": "cuda",
        "forward_device": "cpu",
        "shared_input": True,
        "shared_input_reference_cell": "cuda_dali",
        "official_evaluator_shape": False,
        "purpose": "CPU forward on DALI bytes to isolate decoder effects under CPU kernels",
    },
)

CELL_COMPARISON_SPECS = (
    {
        "comparison_id": "raw_decoder_input_byte_drift_pre_network",
        "isolates": "decoder_input_byte_drift",
        "cell_a": "cpu_av",
        "cell_b": "cuda_dali",
        "surface": "raw_rgb_uint8_before_posenet_segnet",
        "interpretation": "Compares PyAV and DALI/NVDEC decoded RGB bytes before scorer networks.",
    },
    {
        "comparison_id": "forward_kernel_drift_fixed_pyav_input",
        "isolates": "posenet_segnet_forward_kernel_drift",
        "cell_a": "cpu_av",
        "cell_b": "cuda_av_shared_input",
        "surface": "PoseNet/SegNet outputs on shared PyAV RGB input",
        "interpretation": "Holds PyAV input bytes fixed while changing CPU vs CUDA forward kernels.",
    },
    {
        "comparison_id": "forward_kernel_drift_fixed_dali_input",
        "isolates": "posenet_segnet_forward_kernel_drift",
        "cell_a": "cpu_dali",
        "cell_b": "cuda_dali",
        "surface": "PoseNet/SegNet outputs on shared DALI RGB input",
        "interpretation": "Holds DALI/NVDEC input bytes fixed while changing CPU vs CUDA forward kernels.",
    },
    {
        "comparison_id": "decoder_effect_fixed_cpu_forward",
        "isolates": "decoder_input_effect_after_cpu_forward",
        "cell_a": "cpu_av",
        "cell_b": "cpu_dali",
        "surface": "PoseNet/SegNet outputs on CPU forward",
        "interpretation": "Holds CPU forward fixed while changing PyAV vs DALI/NVDEC input bytes.",
    },
    {
        "comparison_id": "decoder_effect_fixed_cuda_forward",
        "isolates": "decoder_input_effect_after_cuda_forward",
        "cell_a": "cuda_av_shared_input",
        "cell_b": "cuda_dali",
        "surface": "PoseNet/SegNet outputs on CUDA forward",
        "interpretation": "Holds CUDA forward fixed while changing PyAV vs DALI/NVDEC input bytes.",
    },
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_custody(path: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    exists = path.exists()
    is_file = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "is_file": is_file,
        "size_bytes": path.stat().st_size if is_file else None,
        "sha256": _sha256_file(path) if include_sha256 and is_file else None,
    }


def _repo_display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().to(device="cpu", dtype=torch.float64)
    if t.numel() == 0:
        return {"shape": list(t.shape), "numel": 0}
    return {
        "shape": list(t.shape),
        "numel": int(t.numel()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "rms": float(torch.sqrt(torch.mean(t * t)).item()),
    }


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    aa = a.detach().to(device="cpu", dtype=torch.float64)
    bb = b.detach().to(device="cpu", dtype=torch.float64)
    if aa.shape != bb.shape:
        return {
            "shape_match": False,
            "shape_a": list(aa.shape),
            "shape_b": list(bb.shape),
        }
    diff = aa - bb
    abs_diff = diff.abs()
    if diff.numel() == 0:
        return {"shape_match": True, "numel": 0}
    return {
        "shape_match": True,
        "shape": list(diff.shape),
        "numel": int(diff.numel()),
        "max_abs_lsb": float(abs_diff.max().item()),
        "mean_abs_lsb": float(abs_diff.mean().item()),
        "rms_abs_lsb": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "nonzero_fraction": float((abs_diff > 0).to(torch.float64).mean().item()),
    }


def per_channel_compare(a: torch.Tensor, b: torch.Tensor) -> list[dict[str, Any]]:
    # Raw evaluator batches are (B, T, H, W, C).
    if a.shape != b.shape or a.ndim < 1:
        return []
    channel_dim = a.ndim - 1
    if a.shape[channel_dim] not in {1, 3, 6, 12}:
        return []
    rows = []
    for idx in range(a.shape[channel_dim]):
        rows.append({"channel": idx, **compare_tensors(a.select(channel_dim, idx), b.select(channel_dim, idx))})
    return rows


def tensor_custody(
    tensor: torch.Tensor, *, tensor_role: str = SHARED_INPUT_TENSOR_ROLE
) -> dict[str, Any]:
    """Return compact custody for a tensor used in the loader-drift matrix."""
    cpu = tensor.detach().to(device="cpu").contiguous()
    return {
        "schema": SHARED_INPUT_TENSOR_SCHEMA,
        "tensor_role": tensor_role,
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "device_recorded": "cpu",
        "contiguous": True,
        "numel": int(cpu.numel()),
        "element_size_bytes": int(cpu.element_size()),
        "byte_length": int(cpu.numel() * cpu.element_size()),
        "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
    }


def _non_promotable_tensor_fields() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
    }


def _cell_spec(cell_id: str) -> dict[str, Any]:
    for spec in INTENDED_CELL_SPECS:
        if spec["cell_id"] == cell_id:
            return spec
    raise KeyError(f"unknown loader drift cell_id: {cell_id}")


def write_shared_input_tensor_artifact(
    *,
    output_dir: Path,
    cell_id: str,
    batch_order: int,
    tensor: torch.Tensor,
    video_path: str,
    sequence_index: int,
) -> dict[str, Any]:
    """Persist a decoded RGB tensor for shared-input CPU/CUDA introspection.

    The artifact is diagnostic-only by construction. It carries the tensor plus
    enough metadata for ``dump_scorer_activations.py --shared-input-tensor`` to
    reject stale, malformed, or promotion-labeled inputs before a scorer
    forward pass.
    """
    spec = _cell_spec(cell_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    cpu = tensor.detach().to(device="cpu").contiguous()
    custody = tensor_custody(cpu)
    artifact_path = output_dir / (
        f"batch{batch_order:06d}_{cell_id}_{SHARED_INPUT_TENSOR_ROLE}.pt"
    )
    payload: dict[str, Any] = {
        "schema": SHARED_INPUT_TENSOR_SCHEMA,
        "created_by": "tools/probe_eval_loader_drift.py",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "cell_id": cell_id,
        "label": spec["label"],
        "tensor_role": SHARED_INPUT_TENSOR_ROLE,
        "loader_class": spec["loader_class"],
        "decoder_backend": spec["decoder_backend"],
        "raw_decode_device": spec["raw_decode_device"],
        "forward_device_intended": spec["forward_device"],
        "official_evaluator_shape": bool(spec["official_evaluator_shape"]),
        "shared_input_reference_cell": spec.get("shared_input_reference_cell"),
        "batch_order": int(batch_order),
        "source_video_path": str(video_path),
        "sequence_index": int(sequence_index),
        "tensor_custody": custody,
        "score_axis": "diagnostic_loader_drift",
        "diagnostic_non_promotable": True,
        **_non_promotable_tensor_fields(),
        "tensor": cpu,
    }
    torch.save(payload, artifact_path)
    record = {key: value for key, value in payload.items() if key != "tensor"}
    record["artifact_path"] = str(artifact_path)
    record["artifact_file_custody"] = _file_custody(artifact_path)
    return record


def compare_numeric_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    aa = a.detach().to(device="cpu", dtype=torch.float64)
    bb = b.detach().to(device="cpu", dtype=torch.float64)
    if aa.shape != bb.shape:
        return {
            "shape_match": False,
            "shape_a": list(aa.shape),
            "shape_b": list(bb.shape),
        }
    diff = aa - bb
    abs_diff = diff.abs()
    if diff.numel() == 0:
        return {"shape_match": True, "numel": 0}
    return {
        "shape_match": True,
        "shape": list(diff.shape),
        "numel": int(diff.numel()),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "rms_abs": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "nonzero_fraction": float((abs_diff > 0).to(torch.float64).mean().item()),
    }


def _segnet_argmax_stats(logits: torch.Tensor) -> dict[str, Any]:
    labels = logits.detach().to(device="cpu").argmax(dim=1)
    counts = torch.bincount(labels.reshape(-1), minlength=int(logits.shape[1]))
    return {
        "shape": list(labels.shape),
        "class_histogram": [int(value) for value in counts.tolist()],
    }


def _model_output_stats(outputs: tuple[dict[str, torch.Tensor], torch.Tensor]) -> dict[str, Any]:
    posenet_out, segnet_out = outputs
    return {
        "posenet": {
            str(name): tensor_stats(value.detach().to(device="cpu"))
            for name, value in sorted(posenet_out.items())
        },
        "segnet_logits": tensor_stats(segnet_out.detach().to(device="cpu")),
        "segnet_argmax": _segnet_argmax_stats(segnet_out),
    }


def _compare_model_outputs(
    a: tuple[dict[str, torch.Tensor], torch.Tensor],
    b: tuple[dict[str, torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    posenet_a, segnet_a = a
    posenet_b, segnet_b = b
    common_heads = sorted(set(posenet_a) & set(posenet_b))
    seg_a = segnet_a.detach().to(device="cpu")
    seg_b = segnet_b.detach().to(device="cpu")
    result: dict[str, Any] = {
        "posenet_heads": {
            str(name): compare_numeric_tensors(posenet_a[name], posenet_b[name])
            for name in common_heads
        },
        "segnet_logits": compare_numeric_tensors(seg_a, seg_b),
    }
    if seg_a.shape == seg_b.shape:
        labels_a = seg_a.argmax(dim=1)
        labels_b = seg_b.argmax(dim=1)
        result["segnet_argmax_disagreement_fraction"] = float(
            (labels_a != labels_b).to(torch.float64).mean().item()
        )
    else:
        result["segnet_argmax_disagreement_fraction"] = None
    return result


def _device_metadata() -> dict[str, Any]:
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "matmul_allow_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
    }
    if torch.cuda.is_available():
        cuda.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_name_0": torch.cuda.get_device_name(0),
                "capability_0": list(torch.cuda.get_device_capability(0)),
            }
        )
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": cuda,
        "mps_available": bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available(),
    }


def _load_video_names(path: Path, limit: int | None) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        return names[:limit]
    return names


def _sample_video_paths(video_names_file: Path, data_dir: Path, limit: int | None) -> tuple[list[str], list[str], str | None]:
    if not video_names_file.exists():
        return [], [], f"video names file missing: {video_names_file}"
    try:
        names = _load_video_names(video_names_file, limit)
    except OSError as exc:
        return [], [], f"video names file unreadable: {exc}"
    paths = [str(data_dir / name) for name in names]
    missing = [path for path in paths if not Path(path).exists()]
    if not names:
        return names, paths, "video names file contains no usable names"
    if missing:
        return names, paths, "sample video files missing: " + ", ".join(missing[:5])
    return names, paths, None


def detect_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    sampled_names, sampled_paths, sample_video_error = _sample_video_paths(
        args.video_names_file,
        args.data_dir,
        args.video_limit,
    )
    return {
        "upstream_frame_utils_py": (UPSTREAM / "frame_utils.py").exists(),
        "upstream_modules_py": (UPSTREAM / "modules.py").exists(),
        "posenet_weights_exist": (UPSTREAM / "models" / "posenet.safetensors").exists(),
        "segnet_weights_exist": (UPSTREAM / "models" / "segnet.safetensors").exists(),
        "video_names_file_exists": args.video_names_file.exists(),
        "data_dir_exists": args.data_dir.exists(),
        "sampled_video_names": sampled_names,
        "sampled_video_paths": sampled_paths,
        "sample_videos_exist": sample_video_error is None,
        "sample_video_error": sample_video_error,
        "pyav_available": _find_spec("av"),
        "cuda_available": torch.cuda.is_available(),
        "dali_available": _find_spec("nvidia.dali"),
        "cuda_dali_runtime_available": None,
        "cuda_dali_runtime_unavailable_reason": None,
        "timm_available": _find_spec("timm"),
        "einops_available": _find_spec("einops"),
        "safetensors_torch_available": _find_spec("safetensors.torch"),
        "segmentation_models_pytorch_available": _find_spec("segmentation_models_pytorch"),
    }


def _mark_comparison_unavailable(
    report: dict[str, Any],
    *,
    unavailable_class: str,
    reasons: list[str],
    codes: list[str],
) -> None:
    report["comparison_available"] = False
    report["comparison_unavailable_class"] = unavailable_class
    report["comparison_unavailable_reason"] = "; ".join(reasons)
    report["comparison_unavailable_reasons"] = reasons
    report["comparison_unavailable_codes"] = codes


def _missing_reasons(
    required_artifacts: tuple[str, ...],
    artifacts: dict[str, Any],
) -> tuple[list[str], list[str]]:
    reasons = []
    codes = []
    for key in required_artifacts:
        if artifacts.get(key) is not True:
            if key == "sample_videos_exist":
                detail = artifacts.get("sample_video_error")
            elif key == "cuda_dali_runtime_available":
                detail = artifacts.get("cuda_dali_runtime_unavailable_reason")
                if artifacts.get(key) is None and detail is None:
                    detail = "runtime check not reached"
            else:
                detail = None
            state = "not_checked" if artifacts.get(key) is None else "false"
            reasons.append(f"{key}={state}" + (f" ({detail})" if detail else ""))
            codes.append(key)
    return reasons, codes


def local_prerequisite_summary(artifacts: dict[str, Any]) -> dict[str, Any]:
    cuda_dali_keys = (
        "cuda_available",
        "dali_available",
        "cuda_dali_runtime_available",
    )
    missing_reasons, missing_codes = _missing_reasons(cuda_dali_keys, artifacts)
    return {
        "local_host_can_run_full_2x2": not missing_reasons,
        "cuda_available": artifacts.get("cuda_available") is True,
        "dali_available": artifacts.get("dali_available") is True,
        "cuda_dali_runtime_available": artifacts.get("cuda_dali_runtime_available"),
        "cuda_dali_runtime_unavailable_reason": artifacts.get(
            "cuda_dali_runtime_unavailable_reason"
        ),
        "missing_cuda_dali_prerequisite_codes": missing_codes,
        "missing_cuda_dali_prerequisite_reasons": missing_reasons,
    }


def future_remote_run_contract(args: argparse.Namespace) -> dict[str, Any]:
    diagnostic_command = [
        ".venv/bin/python",
        "tools/probe_eval_loader_drift.py",
        "--run-forward-cells",
        "--video-names-file",
        _repo_display_path(args.video_names_file),
        "--data-dir",
        _repo_display_path(args.data_dir),
        "--batch-size",
        str(args.batch_size),
        "--num-threads",
        str(args.num_threads),
        "--prefetch-queue-depth",
        str(args.prefetch_queue_depth),
        "--seed",
        str(args.seed),
        "--video-limit",
        str(args.video_limit),
        "--max-batches",
        str(args.max_batches),
        "--json-out",
        REMOTE_ARTIFACT_PLACEHOLDER,
    ]
    return {
        "lane_id": DIAGNOSTIC_REMOTE_LANE_ID,
        "dispatch_attempted": False,
        "requires_dispatch_claim_before_remote_gpu_run": True,
        "required_remote_prerequisite_codes": [
            "cuda_available",
            "dali_available",
            "cuda_dali_runtime_available",
            "pyav_available",
            "upstream_frame_utils_py",
            "upstream_modules_py",
            "posenet_weights_exist",
            "segnet_weights_exist",
        ],
        "claim_command_template": [
            "tools/claim_lane_dispatch.py",
            "claim",
            "--lane-id",
            DIAGNOSTIC_REMOTE_LANE_ID,
            "--platform",
            "lightning",
            "--instance-job-id",
            "${REMOTE_JOB_ID}",
            "--agent",
            "${AGENT_ID}",
            "--status",
            "diagnostic_eval",
            "--notes",
            "DALI/PyAV CPU/CUDA 2x2 loader-drift diagnostic; score_claim=false",
        ],
        "diagnostic_command": diagnostic_command,
        "output_json": REMOTE_ARTIFACT_PLACEHOLDER,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "interpretation": (
            "Future remote execution is a claimed diagnostic run only; it is "
            "not a contest score claim, promotion run, or rank/kill decision."
        ),
    }


def shared_input_artifact_contract(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = str(args.save_shared_input_dir) if args.save_shared_input_dir else None
    return {
        "schema": SHARED_INPUT_TENSOR_SCHEMA,
        "tensor_role": SHARED_INPUT_TENSOR_ROLE,
        "requested": args.save_shared_input_dir is not None,
        "artifact_dir": artifact_dir,
        "producer": "tools/probe_eval_loader_drift.py --save-shared-input-dir",
        "consumer": "experiments/dump_scorer_activations.py --shared-input-tensor",
        "consumer_command_template": [
            ".venv/bin/python",
            "experiments/dump_scorer_activations.py",
            "--upstream-dir",
            "upstream",
            "--shared-input-tensor",
            "${SHARED_INPUT_TENSOR_PT}",
            "--device",
            "${cpu_or_cuda}",
            "--output-dir",
            "${OUTPUT_DIR}",
            "--capture-mode",
            "fingerprint",
        ],
        "required_artifact_fields": [
            "schema",
            "cell_id",
            "tensor_role",
            "tensor_custody.sha256",
            "score_claim",
            "score_claim_valid",
            "promotion_eligible",
            "rank_or_kill_eligible",
            "ready_for_exact_eval_dispatch",
            "dispatch_attempted",
        ],
        "allowed_source_cell_ids": ["cpu_av", "cuda_dali"],
        "interpretation": (
            "Shared-input tensors are decoded RGB bytes for mechanism xray only; "
            "they are not score rows and cannot promote or rank an archive."
        ),
        **_non_promotable_tensor_fields(),
    }


def _unique_required_artifacts(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group))


def _cell_required_artifacts(
    spec: dict[str, Any], *, include_forward_requirements: bool
) -> tuple[str, ...]:
    loader_required = (
        AV_LOADER_REQUIRED_ARTIFACTS
        if spec["loader_key"] == "av"
        else DALI_LOADER_REQUIRED_ARTIFACTS
    )
    if not include_forward_requirements:
        return loader_required
    forward_required = FORWARD_MODEL_REQUIRED_ARTIFACTS
    if spec["forward_device"] == "cuda":
        forward_required = _unique_required_artifacts(forward_required, ("cuda_available",))
    return _unique_required_artifacts(loader_required, forward_required)


def build_intended_cells(
    artifacts: dict[str, Any], *, include_forward_requirements: bool = True
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for spec in INTENDED_CELL_SPECS:
        required = _cell_required_artifacts(
            spec, include_forward_requirements=include_forward_requirements
        )
        missing, missing_codes = _missing_reasons(required, artifacts)
        forward_cell_not_requested = (
            bool(spec.get("shared_input", False)) and not include_forward_requirements
        )
        if forward_cell_not_requested:
            missing = ["run_forward_cells=false"]
            missing_codes = ["run_forward_cells_false"]
        available = not missing and not forward_cell_not_requested
        cells.append(
            {
                "cell_id": spec["cell_id"],
                "label": spec["label"],
                "loader_class": spec["loader_class"],
                "decoder_backend": spec["decoder_backend"],
                "raw_decode_device": spec["raw_decode_device"],
                "forward_device": spec["forward_device"],
                "shared_input": bool(spec.get("shared_input", False)),
                "shared_input_reference_cell": spec.get("shared_input_reference_cell"),
                "official_evaluator_shape": bool(spec["official_evaluator_shape"]),
                "purpose": spec["purpose"],
                "available": available,
                "unsupported": not available,
                "unsupported_class": (
                    None
                    if available
                    else (
                        "forward_cells_not_requested"
                        if forward_cell_not_requested
                        else COMPARISON_UNAVAILABLE_MISSING_PREREQUISITE
                    )
                ),
                "unsupported_reason": None if available else "; ".join(missing),
                "unsupported_reasons": missing,
                "unsupported_codes": missing_codes,
                "required_artifacts": list(required),
                "forward_requirements_included": include_forward_requirements,
                "measurement_status": "available_not_run" if available else "unsupported_not_run",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "contest_cpu_axis_claim": False,
                "contest_cuda_axis_claim": False,
                "custody_label": "diagnostic_non_promotable_loader_forward_cell",
            }
        )
    return cells


def _cells_by_id(cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(cell["cell_id"]): cell for cell in cells}


def build_cell_discriminator_plan(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = _cells_by_id(cells)
    plan: list[dict[str, Any]] = []
    for spec in CELL_COMPARISON_SPECS:
        required_cells = [str(spec["cell_a"]), str(spec["cell_b"])]
        unavailable_reasons: list[str] = []
        unavailable_codes: list[str] = []
        for cell_id in required_cells:
            cell = by_id[cell_id]
            if cell.get("available") is not True:
                unavailable_reasons.extend(
                    f"{cell_id}: {reason}" for reason in cell.get("unsupported_reasons", [])
                )
                unavailable_codes.extend(
                    f"{cell_id}:{code}" for code in cell.get("unsupported_codes", [])
                )
        plan.append(
            {
                "comparison_id": spec["comparison_id"],
                "isolates": spec["isolates"],
                "cell_a": spec["cell_a"],
                "cell_b": spec["cell_b"],
                "surface": spec["surface"],
                "available": not unavailable_reasons,
                "unavailable_reasons": unavailable_reasons,
                "unavailable_codes": unavailable_codes,
                "interpretation": spec["interpretation"],
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
            }
        )
    return plan


def forward_matrix_summary(
    *,
    requested: bool,
    intended_cells: list[dict[str, Any]],
    forward_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize whether the diagnostic 2x2 forward matrix completed."""
    cell_ids = [str(cell.get("cell_id")) for cell in intended_cells]
    unavailable = [
        str(cell.get("cell_id"))
        for cell in intended_cells
        if cell.get("available") is not True
    ]
    if not requested:
        status = "not_requested"
        complete = False
    elif unavailable:
        status = "blocked_unavailable_cells"
        complete = False
    elif not forward_rows:
        status = "no_forward_rows"
        complete = False
    else:
        incomplete_rows = [
            int(row.get("batch_order", index))
            for index, row in enumerate(forward_rows)
            if row.get("forward_matrix_complete") is not True
        ]
        complete = not incomplete_rows
        status = "complete" if complete else "runtime_incomplete"
    return {
        "requested": bool(requested),
        "complete": complete,
        "status": status,
        "required_cell_ids": cell_ids,
        "unavailable_cell_ids": unavailable,
        "forward_row_count": len(forward_rows),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
    }


def _cuda_dali_available() -> tuple[bool, str | None]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is false"
    try:
        import nvidia.dali  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on CUDA environment
        return False, f"nvidia.dali import failed: {exc}"
    return True, None


def _load_distortion_net(device: torch.device) -> torch.nn.Module:
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # type: ignore

    model = DistortionNet().eval().to(device=device)
    model.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    return model


def _forward_model_outputs(
    *,
    batch_cpu: torch.Tensor,
    device: torch.device,
    model_cache: dict[str, torch.nn.Module],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    cache_key = str(device)
    if cache_key not in model_cache:
        model_cache[cache_key] = _load_distortion_net(device)
    model = model_cache[cache_key]
    with torch.inference_mode():
        batch = batch_cpu.to(device=device)
        posenet_out, segnet_out = model(batch)
    posenet_cpu = {
        str(name): value.detach().to(device="cpu", dtype=torch.float32)
        for name, value in posenet_out.items()
    }
    segnet_cpu = segnet_out.detach().to(device="cpu", dtype=torch.float32)
    return posenet_cpu, segnet_cpu


def _device_for_cell(cell: dict[str, Any]) -> torch.device:
    if cell["forward_device"] == "cuda":
        return torch.device("cuda", 0)
    return torch.device("cpu")


def _batch_for_cell(
    cell: dict[str, Any],
    *,
    av_cpu: torch.Tensor,
    dali_cpu: torch.Tensor,
) -> torch.Tensor:
    if cell["cell_id"] in {"cpu_av", "cuda_av_shared_input"}:
        return av_cpu
    return dali_cpu


def _run_forward_cells_for_batch(
    *,
    batch_order: int,
    cells: list[dict[str, Any]],
    av_cpu: torch.Tensor,
    dali_cpu: torch.Tensor,
    model_cache: dict[str, torch.nn.Module],
) -> dict[str, Any]:
    outputs: dict[str, tuple[dict[str, torch.Tensor], torch.Tensor]] = {}
    cell_reports: dict[str, Any] = {}
    for cell in cells:
        cell_id = str(cell["cell_id"])
        if cell.get("available") is not True:
            cell_reports[cell_id] = {
                "status": "unsupported",
                "unsupported_reasons": cell.get("unsupported_reasons", []),
                "unsupported_codes": cell.get("unsupported_codes", []),
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
            }
            continue
        try:
            outputs[cell_id] = _forward_model_outputs(
                batch_cpu=_batch_for_cell(cell, av_cpu=av_cpu, dali_cpu=dali_cpu),
                device=_device_for_cell(cell),
                model_cache=model_cache,
            )
        except Exception as exc:  # pragma: no cover - depends on local scorer stack
            cell_reports[cell_id] = {
                "status": "runtime_error",
                "runtime_error": f"{type(exc).__name__}: {exc}",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
            }
            continue
        cell_reports[cell_id] = {
            "status": "measured",
            "output_stats": _model_output_stats(outputs[cell_id]),
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
        }

    comparisons = []
    for spec in CELL_COMPARISON_SPECS:
        cell_a = str(spec["cell_a"])
        cell_b = str(spec["cell_b"])
        if cell_a in outputs and cell_b in outputs:
            comparisons.append(
                {
                    "comparison_id": spec["comparison_id"],
                    "cell_a": cell_a,
                    "cell_b": cell_b,
                    "isolates": spec["isolates"],
                    "surface": spec["surface"],
                    "available": True,
                    "comparison": _compare_model_outputs(outputs[cell_a], outputs[cell_b]),
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "dispatch_attempted": False,
                }
            )
        else:
            comparisons.append(
                {
                    "comparison_id": spec["comparison_id"],
                    "cell_a": cell_a,
                    "cell_b": cell_b,
                    "isolates": spec["isolates"],
                    "surface": spec["surface"],
                    "available": False,
                    "unavailable_reasons": [
                        f"{cid}: {cell_reports.get(cid, {}).get('status', 'not_measured')}"
                        for cid in (cell_a, cell_b)
                        if cid not in outputs
                    ],
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "dispatch_attempted": False,
                }
            )
    return {
        "batch_order": batch_order,
        "cells": cell_reports,
        "cell_comparisons": comparisons,
        "forward_matrix_complete": all(
            cell_reports.get(str(cell["cell_id"]), {}).get("status") == "measured"
            for cell in cells
        ),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
    }


def _batch_iterator(
    dataset: torch.utils.data.IterableDataset,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)


def _next_batch(iterator: Any) -> tuple[str, int, torch.Tensor]:
    path, idx, batch = next(iterator)
    return str(path), int(idx), batch


def _next_batch_or_none(iterator: Any) -> tuple[str, int, torch.Tensor] | None:
    try:
        return _next_batch(iterator)
    except StopIteration:
        return None


def _collect_av_shared_input_artifacts(
    args: argparse.Namespace, artifacts: dict[str, Any]
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Save locally available PyAV tensors when CUDA/DALI is not present."""
    if args.save_shared_input_dir is None:
        return [], "not_requested", None
    missing, _missing_codes = _missing_reasons(AV_LOADER_REQUIRED_ARTIFACTS, artifacts)
    if missing:
        return [], "unavailable_missing_prerequisite", "; ".join(missing)
    try:
        if str(UPSTREAM) not in sys.path:
            sys.path.insert(0, str(UPSTREAM))
        from frame_utils import AVVideoDataset  # type: ignore

        video_names = _load_video_names(args.video_names_file, args.video_limit)
        av = AVVideoDataset(
            video_names,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            device=torch.device("cpu"),
            num_threads=args.num_threads,
            seed=args.seed,
            prefetch_queue_depth=args.prefetch_queue_depth,
        )
        av.prepare_data()
        av_iter = iter(_batch_iterator(av))
        rows = []
        for batch_idx in range(args.max_batches):
            av_next = _next_batch_or_none(av_iter)
            if av_next is None:
                break
            av_path, av_seq_idx, av_batch = av_next
            rows.append(
                write_shared_input_tensor_artifact(
                    output_dir=args.save_shared_input_dir,
                    cell_id="cpu_av",
                    batch_order=batch_idx,
                    tensor=av_batch,
                    video_path=av_path,
                    sequence_index=av_seq_idx,
                )
            )
    except Exception as exc:  # pragma: no cover - depends on local video stack
        return [], "runtime_error", f"{type(exc).__name__}: {exc}"
    if not rows:
        return [], "runtime_empty", "no PyAV batches emitted"
    return rows, "written", None


def source_file_custody(args: argparse.Namespace, artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "upstream_frame_utils": _file_custody(UPSTREAM / "frame_utils.py"),
        "upstream_evaluate": _file_custody(UPSTREAM / "evaluate.py"),
        "upstream_modules": _file_custody(UPSTREAM / "modules.py"),
        "posenet_weights": _file_custody(UPSTREAM / "models" / "posenet.safetensors"),
        "segnet_weights": _file_custody(UPSTREAM / "models" / "segnet.safetensors"),
        "video_names_file": _file_custody(args.video_names_file),
        "sample_videos": [
            _file_custody(Path(path), include_sha256=False)
            for path in artifacts.get("sampled_video_paths", [])
        ],
    }


def loader_device_custody() -> dict[str, Any]:
    return {
        "comparison_scope": "raw_evaluator_loader_rgb_before_posenet_segnet",
        "score_path": "not_run",
        "network_forward_device": "not_run",
        "dispatch_attempted": False,
        "diagnostic_forward_cells": [
            {
                "cell_id": spec["cell_id"],
                "label": spec["label"],
                "loader_class": spec["loader_class"],
                "forward_device": spec["forward_device"],
                "official_evaluator_shape": spec["official_evaluator_shape"],
            }
            for spec in INTENDED_CELL_SPECS
        ],
        "loaders": [
            {
                "loader_class": "AVVideoDataset",
                "decoder_backend": "PyAV/FFmpeg",
                "raw_decode_device": "cpu",
                "evaluator_axis": "cpu_loader_path",
            },
            {
                "loader_class": "DaliVideoDataset",
                "decoder_backend": "DALI/NVDEC",
                "raw_decode_device": "cuda",
                "evaluator_axis": "cuda_loader_path",
            },
        ],
    }


def device_axis_custody() -> dict[str, Any]:
    return {
        "score_axis": "diagnostic_loader_drift",
        "claimed_score_axes": [],
        "score_claim_axis": "none",
        "contest_cpu_axis_claim": False,
        "contest_cuda_axis_claim": False,
        "contest_cuda_claim": False,
        "contest_cpu_claim": False,
        "macos_cpu_advisory_claim": False,
        "mps_claim": False,
        "rank_or_kill_eligible": False,
        "promotion_eligible": False,
        "score_claim_valid": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "compared_ground_truth_loader_axes": ["cpu_loader_path", "cuda_loader_path"],
        "cpu_loader_path": {
            "loader_class": "AVVideoDataset",
            "decoder_backend": "PyAV/FFmpeg",
            "raw_decode_device": "cpu",
        },
        "cuda_loader_path": {
            "loader_class": "DaliVideoDataset",
            "decoder_backend": "DALI/NVDEC",
            "raw_decode_device": "cuda",
        },
        "compressed_loader_path": {
            "loader_class": "TensorVideoDataset",
            "shared_between_score_axes": True,
        },
    }


def build_probe_report(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = detect_artifacts(args)
    raw_missing, raw_missing_codes = _missing_reasons(RAW_DECODER_REQUIRED_ARTIFACTS, artifacts)
    cuda_ok: bool | None = None
    cuda_reason: str | None = None
    if not raw_missing:
        cuda_ok, cuda_reason = _cuda_dali_available()
        artifacts["cuda_dali_runtime_available"] = cuda_ok
        artifacts["cuda_dali_runtime_unavailable_reason"] = cuda_reason
    intended_cells = build_intended_cells(
        artifacts,
        include_forward_requirements=bool(args.run_forward_cells),
    )
    cell_discriminator_plan = build_cell_discriminator_plan(intended_cells)
    report: dict[str, Any] = {
        "schema": "eval_loader_device_drift_probe.v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "score_claim": False,
        "score_axis": "diagnostic_loader_drift",
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "evidence_grade": "diagnostic",
        "diagnostic_kind": "loader_drift_probe",
        "repo": str(REPO),
        "upstream_evaluate": str(UPSTREAM / "evaluate.py"),
        "upstream_frame_utils": str(UPSTREAM / "frame_utils.py"),
        "upstream_modules": str(UPSTREAM / "modules.py"),
        "video_names_file": str(args.video_names_file),
        "data_dir": str(args.data_dir),
        "batch_size": args.batch_size,
        "num_threads": args.num_threads,
        "prefetch_queue_depth": args.prefetch_queue_depth,
        "video_limit": args.video_limit,
        "max_batches": args.max_batches,
        "run_forward_cells": bool(args.run_forward_cells),
        "environment": _device_metadata(),
        "artifact_status": artifacts,
        "local_prerequisite_summary": local_prerequisite_summary(artifacts),
        "source_file_custody": source_file_custody(args, artifacts),
        "loader_device_custody": loader_device_custody(),
        "device_axis_custody": device_axis_custody(),
        "future_remote_run_contract": future_remote_run_contract(args),
        "shared_input_artifact_contract": shared_input_artifact_contract(args),
        "shared_input_artifacts": [],
        "shared_input_artifact_status": (
            "requested_pending" if args.save_shared_input_dir else "not_requested"
        ),
        "shared_input_artifact_unavailable_reason": None,
        "custody_labels": {
            "artifact_kind": "diagnostic_loader_forward_drift_probe",
            "score_path": "not_run",
            "score_claim_axis": "none",
            "contest_cpu_axis_claim": False,
            "contest_cuda_axis_claim": False,
            "dispatch_attempted": False,
            "dispatch_claim_required_before_remote_run": True,
            "diagnostic_non_promotable": True,
        },
        "intended_cells": intended_cells,
        "cell_discriminator_plan": cell_discriminator_plan,
        "forward_matrix_complete": False,
        "forward_matrix_summary": forward_matrix_summary(
            requested=bool(args.run_forward_cells),
            intended_cells=intended_cells,
            forward_rows=[],
        ),
        "forward_cell_rows": [],
        "interpretation_guardrails": [
            "This probe compares decoded evaluator input tensors before PoseNet/SegNet by default.",
            "Optional forward-cell rows are diagnostic PoseNet/SegNet output comparisons only; they are not score rows.",
            "It does not run inflate.sh, evaluate.py scoring, or any contest promotion path.",
            "dispatch_attempted=false; any future CUDA+DALI remote run must claim the diagnostic lane first.",
            "A nonzero DALI-vs-PyAV raw RGB diff proves loader drift exists, but score impact still requires paired exact eval.",
            "Cell labels such as CPU+AV and CUDA+DALI describe diagnostic mechanisms, not contest_cpu or contest_cuda score axes.",
        ],
    }
    if raw_missing:
        shared_rows, shared_status, shared_reason = _collect_av_shared_input_artifacts(
            args, artifacts
        )
        report["shared_input_artifacts"] = shared_rows
        report["shared_input_artifact_status"] = shared_status
        report["shared_input_artifact_unavailable_reason"] = shared_reason
        _mark_comparison_unavailable(
            report,
            unavailable_class=COMPARISON_UNAVAILABLE_MISSING_PREREQUISITE,
            reasons=raw_missing,
            codes=raw_missing_codes,
        )
        return report

    if not cuda_ok:
        shared_rows, shared_status, shared_reason = _collect_av_shared_input_artifacts(
            args, artifacts
        )
        report["shared_input_artifacts"] = shared_rows
        report["shared_input_artifact_status"] = shared_status
        report["shared_input_artifact_unavailable_reason"] = shared_reason
        _mark_comparison_unavailable(
            report,
            unavailable_class=COMPARISON_UNAVAILABLE_MISSING_PREREQUISITE,
            reasons=[cuda_reason or "cuda_dali_runtime_unavailable"],
            codes=["cuda_dali_runtime_available"],
        )
        return report
    report["comparison_available"] = True
    report["comparison_unavailable_class"] = None
    report["comparison_unavailable_reason"] = None
    report["comparison_unavailable_reasons"] = []
    report["comparison_unavailable_codes"] = []

    try:
        if str(UPSTREAM) not in sys.path:
            sys.path.insert(0, str(UPSTREAM))
        from frame_utils import AVVideoDataset, DaliVideoDataset, camera_size, seq_len  # type: ignore

        video_names = _load_video_names(args.video_names_file, args.video_limit)
        cuda_device = torch.device("cuda", 0)
        dali = DaliVideoDataset(
            video_names,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            device=cuda_device,
            num_threads=args.num_threads,
            seed=args.seed,
            prefetch_queue_depth=args.prefetch_queue_depth,
        )
        av = AVVideoDataset(
            video_names,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            device=torch.device("cpu"),
            num_threads=args.num_threads,
            seed=args.seed,
            prefetch_queue_depth=args.prefetch_queue_depth,
        )
        dali.prepare_data()
        av.prepare_data()

        rows = []
        forward_rows = []
        shared_input_artifacts = []
        model_cache: dict[str, torch.nn.Module] = {}
        dali_iter = iter(_batch_iterator(dali))
        av_iter = iter(_batch_iterator(av))
        incomplete_reason: str | None = None
        for batch_idx in range(args.max_batches):
            dali_next = _next_batch_or_none(dali_iter)
            av_next = _next_batch_or_none(av_iter)
            if dali_next is None or av_next is None:
                incomplete_reason = (
                    "dataset_iterator_exhausted_before_requested_max_batches"
                )
                break
            dali_path, dali_seq_idx, dali_batch = dali_next
            av_path, av_seq_idx, av_batch = av_next
            dali_cpu = dali_batch.detach().to(device="cpu")
            av_cpu = av_batch.detach().to(device="cpu")
            if list(dali_cpu.shape)[1:] != [seq_len, camera_size[1], camera_size[0], 3]:
                raise RuntimeError(f"unexpected DALI batch shape: {list(dali_cpu.shape)}")
            if list(av_cpu.shape)[1:] != [seq_len, camera_size[1], camera_size[0], 3]:
                raise RuntimeError(f"unexpected AV batch shape: {list(av_cpu.shape)}")
            batch_shared_artifacts: dict[str, Any] = {}
            if args.save_shared_input_dir is not None:
                batch_shared_artifacts["av_rgb_uint8"] = write_shared_input_tensor_artifact(
                    output_dir=args.save_shared_input_dir,
                    cell_id="cpu_av",
                    batch_order=batch_idx,
                    tensor=av_cpu,
                    video_path=av_path,
                    sequence_index=av_seq_idx,
                )
                batch_shared_artifacts["dali_rgb_uint8"] = write_shared_input_tensor_artifact(
                    output_dir=args.save_shared_input_dir,
                    cell_id="cuda_dali",
                    batch_order=batch_idx,
                    tensor=dali_cpu,
                    video_path=dali_path,
                    sequence_index=dali_seq_idx,
                )
                shared_input_artifacts.extend(batch_shared_artifacts.values())
            rows.append(
                {
                    "batch_order": batch_idx,
                    "dali_path": dali_path,
                    "av_path": av_path,
                    "dali_sequence_index": dali_seq_idx,
                    "av_sequence_index": av_seq_idx,
                    "path_match": dali_path == av_path,
                    "sequence_index_match": dali_seq_idx == av_seq_idx,
                    "dali_stats": tensor_stats(dali_cpu),
                    "av_stats": tensor_stats(av_cpu),
                    "tensor_custody": {
                        "dali_rgb_uint8": tensor_custody(dali_cpu),
                        "av_rgb_uint8": tensor_custody(av_cpu),
                    },
                    "comparison": compare_tensors(dali_cpu, av_cpu),
                    "per_rgb_channel": per_channel_compare(dali_cpu, av_cpu),
                    "shared_input_artifacts": batch_shared_artifacts,
                }
            )
            if args.run_forward_cells:
                forward_rows.append(
                    _run_forward_cells_for_batch(
                        batch_order=batch_idx,
                        cells=intended_cells,
                        av_cpu=av_cpu,
                        dali_cpu=dali_cpu,
                        model_cache=model_cache,
                    )
                )
    except Exception as exc:
        reason = f"probe_runtime_error: {type(exc).__name__}: {exc}"
        _mark_comparison_unavailable(
            report,
            unavailable_class=COMPARISON_UNAVAILABLE_PROBE_RUNTIME_ERROR,
            reasons=[reason],
            codes=[COMPARISON_UNAVAILABLE_PROBE_RUNTIME_ERROR],
        )
        return report

    report["comparison_rows"] = rows
    report["shared_input_artifacts"] = shared_input_artifacts
    report["shared_input_artifact_status"] = (
        "written" if shared_input_artifacts else report["shared_input_artifact_status"]
    )
    report["shared_input_artifact_unavailable_reason"] = None
    report["forward_cell_rows"] = forward_rows
    report["forward_matrix_summary"] = forward_matrix_summary(
        requested=bool(args.run_forward_cells),
        intended_cells=intended_cells,
        forward_rows=forward_rows,
    )
    report["forward_matrix_complete"] = bool(
        report["forward_matrix_summary"]["complete"]
    )
    report["comparison_available"] = bool(rows)
    report["comparison_incomplete"] = incomplete_reason is not None
    report["comparison_incomplete_reason"] = incomplete_reason
    if not rows:
        reason = incomplete_reason or "no_comparison_rows_emitted"
        _mark_comparison_unavailable(
            report,
            unavailable_class=COMPARISON_UNAVAILABLE_PROBE_RUNTIME_ERROR,
            reasons=[f"probe_runtime_error: {reason}"],
            codes=[COMPARISON_UNAVAILABLE_PROBE_RUNTIME_ERROR],
        )
    else:
        report["comparison_unavailable_class"] = None
        report["comparison_unavailable_reason"] = None
        report["comparison_unavailable_reasons"] = []
        report["comparison_unavailable_codes"] = []
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-names-file", type=Path, default=UPSTREAM / "public_test_video_names.txt")
    parser.add_argument("--data-dir", type=Path, default=UPSTREAM / "videos")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--prefetch-queue-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--video-limit", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument(
        "--run-forward-cells",
        action="store_true",
        help=(
            "also run diagnostic PoseNet/SegNet forward cells for the 2x2 "
            "loader/device matrix; still emits no score claims"
        ),
    )
    parser.add_argument(
        "--save-shared-input-dir",
        type=Path,
        default=None,
        help=(
            "optional directory for non-promotable decoded RGB tensor artifacts "
            "consumable by experiments/dump_scorer_activations.py "
            "--shared-input-tensor"
        ),
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_probe_report(args)
    text = json.dumps(_jsonable(report), indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report.get("comparison_available") is False:
        if report.get("comparison_unavailable_class") == COMPARISON_UNAVAILABLE_PROBE_RUNTIME_ERROR:
            return 3
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
