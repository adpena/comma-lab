#!/usr/bin/env python3
"""Build a fail-closed 2x2 evaluator drift diagnostic matrix.

This is a diagnostic harness, not a scorer. It keeps the public evaluator's
two drift axes explicit:

* decoder/preprocess source: CPU PyAV/AVVideoDataset vs CUDA DALI/NVDEC
* network forward device: CPU vs CUDA

The resulting 2x2 plan separates row-wise decoder/preprocess questions from
column-wise network-forward questions and records why any cell cannot be run
locally. Missing CUDA, DALI, PyAV, videos, or model artifacts fail closed by
emitting a non-promotable JSON artifact and returning exit code 2.
"""

from __future__ import annotations

import argparse
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
TOOLS = REPO / "tools"

DECODER_ROWS = (
    {
        "decoder_source": "pyav_av",
        "label": "CPU PyAV via AVVideoDataset",
        "raw_decode_device": "cpu",
        "loader_class": "AVVideoDataset",
        "required_artifacts": (
            "upstream_frame_utils_py",
            "pyav_available",
            "video_names_file_exists",
            "data_dir_exists",
            "sample_videos_exist",
        ),
    },
    {
        "decoder_source": "dali_nvdec",
        "label": "CUDA DALI/NVDEC via DaliVideoDataset",
        "raw_decode_device": "cuda",
        "loader_class": "DaliVideoDataset",
        "required_artifacts": (
            "upstream_frame_utils_py",
            "cuda_available",
            "dali_available",
            "video_names_file_exists",
            "data_dir_exists",
            "sample_videos_exist",
        ),
    },
)

FORWARD_COLUMNS = (
    {
        "forward_device": "cpu",
        "label": "CPU DistortionNet forward",
        "required_artifacts": (
            "upstream_modules_py",
            "posenet_weights_exist",
            "segnet_weights_exist",
        ),
    },
    {
        "forward_device": "cuda",
        "label": "CUDA DistortionNet forward",
        "required_artifacts": (
            "upstream_modules_py",
            "posenet_weights_exist",
            "segnet_weights_exist",
            "cuda_available",
        ),
    },
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
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
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": cuda,
        "mps_available": bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available(),
    }


def _sample_video_paths(video_names_file: Path, data_dir: Path, limit: int) -> tuple[list[str], list[str], str | None]:
    if not video_names_file.exists():
        return [], [], f"video names file missing: {video_names_file}"
    try:
        names = [
            line.strip()
            for line in video_names_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][:limit]
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
        "upstream_evaluate_py": (UPSTREAM / "evaluate.py").exists(),
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
    }


def _missing_reasons(required_artifacts: tuple[str, ...], artifacts: dict[str, Any]) -> list[str]:
    reasons = []
    for key in required_artifacts:
        if artifacts.get(key) is not True:
            detail = artifacts.get("sample_video_error") if key == "sample_videos_exist" else None
            reasons.append(f"{key}=false" + (f" ({detail})" if detail else ""))
    return reasons


def _cell_id(decoder_source: str, forward_device: str) -> str:
    return f"{decoder_source}__forward_{forward_device}"


def build_matrix_cells(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for row in DECODER_ROWS:
        for column in FORWARD_COLUMNS:
            required = tuple(dict.fromkeys(row["required_artifacts"] + column["required_artifacts"]))
            missing = _missing_reasons(required, artifacts)
            cells.append(
                {
                    "cell_id": _cell_id(row["decoder_source"], column["forward_device"]),
                    "decoder_source": row["decoder_source"],
                    "decoder_label": row["label"],
                    "raw_decode_device": row["raw_decode_device"],
                    "loader_class": row["loader_class"],
                    "forward_device": column["forward_device"],
                    "forward_label": column["label"],
                    "preprocess_device": column["forward_device"],
                    "available": not missing,
                    "required_artifacts": list(required),
                    "unavailable_reasons": missing,
                    "measurement_contract": {
                        "raw_rgb_surface": "(B,T,H,W,3) uint8 from upstream/frame_utils.py",
                        "preprocess_surface": "DistortionNet.preprocess_input on the forward device",
                        "network_surface": "PoseNet and SegNet forward outputs only; no archive scoring",
                    },
                }
            )
    return cells


def _cells_by_id(cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(cell["cell_id"]): cell for cell in cells}


def _comparison(
    *,
    comparison_id: str,
    isolates: str,
    cell_a: str,
    cell_b: str,
    cells: dict[str, dict[str, Any]],
    interpretation: str,
) -> dict[str, Any]:
    required_cells = [cell_a, cell_b]
    reasons: list[str] = []
    for cell_id in required_cells:
        cell = cells[cell_id]
        if cell.get("available") is not True:
            reasons.extend(f"{cell_id}: {reason}" for reason in cell.get("unavailable_reasons", []))
    return {
        "comparison_id": comparison_id,
        "isolates": isolates,
        "cell_a": cell_a,
        "cell_b": cell_b,
        "available": not reasons,
        "unavailable_reasons": reasons,
        "interpretation": interpretation,
    }


def build_comparison_plan(cells: list[dict[str, Any]], artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = _cells_by_id(cells)
    comparisons = [
        _comparison(
            comparison_id="decoder_preprocess_drift_fixed_cpu_forward",
            isolates="decoder_plus_preprocess_source_drift",
            cell_a="pyav_av__forward_cpu",
            cell_b="dali_nvdec__forward_cpu",
            cells=by_id,
            interpretation=(
                "Same CPU forward path; row difference localizes drift to raw decode "
                "and preprocessing input source before network kernels."
            ),
        ),
        _comparison(
            comparison_id="decoder_preprocess_drift_fixed_cuda_forward",
            isolates="decoder_plus_preprocess_source_drift",
            cell_a="pyav_av__forward_cuda",
            cell_b="dali_nvdec__forward_cuda",
            cells=by_id,
            interpretation=(
                "Same CUDA forward path; row difference localizes drift to PyAV vs "
                "DALI/NVDEC decode plus preprocessing input source."
            ),
        ),
        _comparison(
            comparison_id="network_forward_drift_fixed_pyav_decode",
            isolates="preprocess_plus_network_device_drift",
            cell_a="pyav_av__forward_cpu",
            cell_b="pyav_av__forward_cuda",
            cells=by_id,
            interpretation=(
                "Same PyAV raw RGB source; column difference localizes drift to "
                "preprocess and network execution device."
            ),
        ),
        _comparison(
            comparison_id="network_forward_drift_fixed_dali_decode",
            isolates="preprocess_plus_network_device_drift",
            cell_a="dali_nvdec__forward_cpu",
            cell_b="dali_nvdec__forward_cuda",
            cells=by_id,
            interpretation=(
                "Same DALI/NVDEC raw RGB source; column difference localizes drift "
                "to preprocess and network execution device."
            ),
        ),
    ]

    loader_required = (
        "upstream_frame_utils_py",
        "pyav_available",
        "cuda_available",
        "dali_available",
        "video_names_file_exists",
        "data_dir_exists",
        "sample_videos_exist",
    )
    loader_missing = _missing_reasons(loader_required, artifacts)
    comparisons.append(
        {
            "comparison_id": "raw_decoder_drift_pre_network",
            "isolates": "decoder_only_raw_rgb_drift",
            "available": not loader_missing,
            "unavailable_reasons": loader_missing,
            "tool": "tools/probe_eval_loader_drift.py",
            "interpretation": "Compares DALI/NVDEC and PyAV RGB uint8 tensors before PoseNet or SegNet.",
        }
    )

    posenet_required = (
        "upstream_modules_py",
        "posenet_weights_exist",
        "cuda_available",
    )
    posenet_missing = _missing_reasons(posenet_required, artifacts)
    comparisons.append(
        {
            "comparison_id": "posenet_shared_input_forward_drift",
            "isolates": "posenet_network_forward_drift_on_shared_input",
            "available": not posenet_missing,
            "unavailable_reasons": posenet_missing,
            "tool": "tools/probe_posenet_layer_drift.py",
            "interpretation": (
                "Runs PoseNet layer probes on one shared input tensor so decoder "
                "differences cannot explain activation drift."
            ),
        }
    )
    return comparisons


def build_probe_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    output_dir = args.output_dir
    return {
        "raw_decoder_drift_pre_network": [
            sys.executable,
            str(TOOLS / "probe_eval_loader_drift.py"),
            "--video-names-file",
            str(args.video_names_file),
            "--data-dir",
            str(args.data_dir),
            "--batch-size",
            str(args.batch_size),
            "--video-limit",
            str(args.video_limit),
            "--max-batches",
            str(args.max_batches),
            "--json-out",
            str(output_dir / "raw_decoder_drift_pre_network.json"),
        ],
        "posenet_shared_input_forward_drift": [
            sys.executable,
            str(TOOLS / "probe_posenet_layer_drift.py"),
            "--device-a",
            "cpu",
            "--device-b",
            "cuda",
            "--input-kind",
            "evaluator_rgb",
            "--batch-size",
            str(args.batch_size),
            "--json-out",
            str(output_dir / "posenet_shared_input_forward_drift.json"),
        ],
    }


def build_probe_report(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = detect_artifacts(args)
    cells = build_matrix_cells(artifacts)
    comparisons = build_comparison_plan(cells, artifacts)
    fail_closed_reasons = [
        f"{comparison['comparison_id']}: " + "; ".join(comparison["unavailable_reasons"])
        for comparison in comparisons
        if comparison.get("available") is not True
    ]
    return {
        "schema": "eval_drift_matrix_probe.v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "evidence_grade": "diagnostic",
        "repo": str(REPO),
        "upstream_evaluate": str(UPSTREAM / "evaluate.py"),
        "upstream_frame_utils": str(UPSTREAM / "frame_utils.py"),
        "upstream_modules": str(UPSTREAM / "modules.py"),
        "video_names_file": str(args.video_names_file),
        "data_dir": str(args.data_dir),
        "batch_size": args.batch_size,
        "video_limit": args.video_limit,
        "max_batches": args.max_batches,
        "environment": _device_metadata(),
        "artifact_status": artifacts,
        "matrix_rows": [row["decoder_source"] for row in DECODER_ROWS],
        "matrix_columns": [column["forward_device"] for column in FORWARD_COLUMNS],
        "matrix_cells": cells,
        "comparison_plan": comparisons,
        "probe_commands": build_probe_commands(args),
        "fail_closed": bool(fail_closed_reasons),
        "fail_closed_reasons": fail_closed_reasons,
        "interpretation_guardrails": [
            "Diagnostic only: this tool does not run inflate.sh or upstream/evaluate.py scoring.",
            "Rows compare decoder/preprocess source while holding forward device fixed.",
            "Columns compare preprocess/network device while holding decoded source fixed.",
            "A filled 2x2 matrix can localize mechanism, but exact score, rank, promotion, or kill claims still require separate contest-grade eval evidence.",
            "Missing CUDA, DALI, PyAV, model weights, or video artifacts must remain fail-closed and non-promotable.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-names-file", type=Path, default=UPSTREAM / "public_test_video_names.txt")
    parser.add_argument("--data-dir", type=Path, default=UPSTREAM / "videos")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--video-limit", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=REPO / "experiments" / "results" / "eval_drift_matrix_probe")
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
    if report.get("fail_closed") is True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
