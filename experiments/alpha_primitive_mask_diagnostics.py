#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deterministic Alpha primitive mask diagnostics.

This is a bounded, contest-safe, non-promotable diagnostic helper for Alpha
mask-payload redesign. It reads a canonical archive member without extracting
the ZIP, decodes masks through the existing mask codec, and records primitive
geometry statistics that can guide scorer-preserving representation work.

It does not load scorer networks and it does not produce score evidence. Any
score claim still requires exact CUDA auth eval through:

    archive.zip -> inflate.sh -> upstream/evaluate.py
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCHEMA = "alpha_primitive_mask_diagnostics_v1"
EVIDENCE_GRADE = "empirical"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
SCORE_CLAIM_WARNING = (
    "No score claim is made by this diagnostic. Exact CUDA auth eval is "
    "required before any score claim, promotion, ranking, or method retirement."
)
DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/results/alpha_primitive_mask_diagnostics/alpha_primitive_mask_diagnostics.json"
)
DEFAULT_MAX_FRAMES = 64
CLASS_IDS = (0, 1, 2, 3, 4)
_HIDDEN_SYSTEM_NAMES = {"__MACOSX", ".DS_Store", "Thumbs.db"}


@dataclass(frozen=True)
class DiagnosticConfig:
    max_frames: int | None = None
    max_components_per_class: int = 8
    connectivity: int = 4


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_parts(name: str) -> tuple[str, ...]:
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"unsafe archive member path: {name!r}")
    member_path = PurePosixPath(name)
    parts = member_path.parts
    if member_path.is_absolute() or ".." in parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    if not parts or any(part in ("", ".") for part in parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    first = parts[0]
    if len(first) == 2 and first[1] == ":":
        raise ValueError(f"unsafe archive member path: {name!r}")
    return parts


def _reject_hidden_or_system_member(name: str, parts: tuple[str, ...]) -> None:
    if any(part in _HIDDEN_SYSTEM_NAMES for part in parts):
        raise ValueError(f"hidden/system archive member: {name!r}")
    if any(part.startswith(".") or part.startswith("._") for part in parts):
        raise ValueError(f"hidden/system archive member: {name!r}")


def _validate_requested_member(member: str) -> None:
    parts = _safe_member_parts(member)
    _reject_hidden_or_system_member(member, parts)


def _validated_zip_infos(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    for info in zf.infolist():
        if info.filename in infos:
            raise ValueError(f"duplicate archive member: {info.filename!r}")
        if info.is_dir():
            raise ValueError(f"unsafe archive directory member: {info.filename!r}")
        parts = _safe_member_parts(info.filename)
        _reject_hidden_or_system_member(info.filename, parts)
        infos[info.filename] = info
    return infos


def _member_inventory(infos: dict[str, zipfile.ZipInfo]) -> list[dict[str, Any]]:
    inventory = []
    for name in sorted(infos):
        info = infos[name]
        inventory.append(
            {
                "name": name,
                "size_bytes": int(info.file_size),
                "compressed_size_bytes": int(info.compress_size),
                "crc32": f"{info.CRC:08x}",
            }
        )
    return inventory


def _read_archive_member(archive: Path, member: str) -> tuple[bytes, dict[str, Any]]:
    """Read one archive member after rejecting zip-slip and hidden sidecars."""
    archive = Path(archive)
    _validate_requested_member(member)
    with zipfile.ZipFile(archive, "r") as zf:
        infos = _validated_zip_infos(zf)
        if member not in infos:
            raise FileNotFoundError(f"{archive} missing archive member {member!r}")
        info = infos[member]
        data = zf.read(info)

    member_meta = {
        "name": member,
        "size_bytes": int(info.file_size),
        "compressed_size_bytes": int(info.compress_size),
        "crc32": f"{info.CRC:08x}",
        "sha256": _sha256_bytes(data),
    }
    return data, {
        "archive_path": str(archive),
        "archive_size_bytes": int(archive.stat().st_size),
        "archive_sha256": _sha256_file(archive),
        "member_inventory": _member_inventory(infos),
        "mask_member": member_meta,
    }


def _decode_av1_masks_from_member(data: bytes, member: str) -> torch.Tensor:
    from tac.mask_codec import decode_masks

    suffix = PurePosixPath(member).suffix or ".mkv"
    with tempfile.TemporaryDirectory() as tmp_dir:
        mask_path = Path(tmp_dir) / f"mask_member{suffix}"
        mask_path.write_bytes(data)
        masks = decode_masks(mask_path)
    return _validate_masks(masks)


def _validate_masks(masks: torch.Tensor) -> torch.Tensor:
    if not isinstance(masks, torch.Tensor):
        raise TypeError(f"masks must be a torch.Tensor, got {type(masks).__name__}")
    if masks.dim() != 3:
        raise ValueError(f"masks must have shape (T,H,W), got {tuple(masks.shape)}")
    if masks.numel() == 0:
        raise ValueError("masks tensor is empty")
    masks = masks.detach().cpu().to(torch.int64).contiguous()
    min_value = int(masks.min().item())
    max_value = int(masks.max().item())
    if min_value < min(CLASS_IDS) or max_value > max(CLASS_IDS):
        raise ValueError(f"masks must contain class ids in [0,4], got [{min_value},{max_value}]")
    return masks


def _slice_masks(masks: torch.Tensor, max_frames: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
    if max_frames is None:
        return masks, {
            "max_frames": None,
            "original_frames": int(masks.shape[0]),
            "analyzed_frames": int(masks.shape[0]),
            "truncated": False,
        }
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive when provided, got {max_frames}")
    sliced = masks[:max_frames].contiguous()
    return sliced, {
        "max_frames": int(max_frames),
        "original_frames": int(masks.shape[0]),
        "analyzed_frames": int(sliced.shape[0]),
        "truncated": bool(sliced.shape[0] != masks.shape[0]),
    }


def _round_float(value: float, digits: int = 12) -> float:
    return round(float(value), digits)


def _mask_stats(masks: torch.Tensor) -> dict[str, Any]:
    masks = _validate_masks(masks)
    counts = torch.bincount(masks.reshape(-1), minlength=max(CLASS_IDS) + 1).to(torch.int64)
    total = int(masks.numel())
    vertical_edges = int((masks[:, 1:, :] != masks[:, :-1, :]).sum().item()) if masks.shape[1] > 1 else 0
    horizontal_edges = int((masks[:, :, 1:] != masks[:, :, :-1]).sum().item()) if masks.shape[2] > 1 else 0
    return {
        "shape": [int(v) for v in masks.shape],
        "dtype": str(masks.dtype),
        "num_pixels": total,
        "class_histogram": {str(class_id): int(counts[class_id].item()) for class_id in CLASS_IDS},
        "class_fractions": {
            str(class_id): _round_float(int(counts[class_id].item()) / total) for class_id in CLASS_IDS
        },
        "spatial_boundary_edges_4conn": vertical_edges + horizontal_edges,
    }


def _component_sort_key(component: dict[str, Any]) -> tuple[int, int, int, int]:
    bbox = component["bbox_xyxy_exclusive"]
    return (-int(component["area"]), int(bbox[1]), int(bbox[0]), int(component["scan_index"]))


def _frame_components(frame: np.ndarray) -> dict[int, list[dict[str, Any]]]:
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2D, got shape {frame.shape}")
    height, width = [int(v) for v in frame.shape]
    visited = np.zeros((height, width), dtype=np.bool_)
    components: dict[int, list[dict[str, Any]]] = {class_id: [] for class_id in CLASS_IDS}
    allowed = set(CLASS_IDS)
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    scan_index = 0

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x]:
                continue
            class_id = int(frame[start_y, start_x])
            if class_id not in allowed:
                raise ValueError(f"unexpected class id {class_id} at ({start_y}, {start_x})")

            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            area = 0
            sum_x = 0
            sum_y = 0
            min_x = start_x
            max_x = start_x
            min_y = start_y
            max_y = start_y
            boundary_pixels = 0
            boundary_edges = 0

            while stack:
                y, x = stack.pop()
                area += 1
                sum_x += x
                sum_y += y
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                is_boundary_pixel = False

                for dy, dx in neighbors:
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        is_boundary_pixel = True
                        boundary_edges += 1
                        continue
                    if int(frame[ny, nx]) != class_id:
                        is_boundary_pixel = True
                        boundary_edges += 1
                        continue
                    if not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

                if is_boundary_pixel:
                    boundary_pixels += 1

            components[class_id].append(
                {
                    "scan_index": int(scan_index),
                    "area": int(area),
                    "bbox_xyxy_exclusive": [int(min_x), int(min_y), int(max_x + 1), int(max_y + 1)],
                    "centroid_xy": [_round_float(sum_x / area, 6), _round_float(sum_y / area, 6)],
                    "boundary_pixels_4conn": int(boundary_pixels),
                    "boundary_edges_4conn": int(boundary_edges),
                }
            )
            scan_index += 1

    return components


def _class_component_record(
    components: list[dict[str, Any]],
    *,
    max_components_per_class: int,
) -> dict[str, Any]:
    if max_components_per_class < 0:
        raise ValueError(f"max_components_per_class must be non-negative, got {max_components_per_class}")
    sorted_components = sorted(components, key=_component_sort_key)
    emitted = sorted_components[:max_components_per_class]
    total_area = sum(int(component["area"]) for component in components)
    total_boundary_pixels = sum(int(component["boundary_pixels_4conn"]) for component in components)
    total_boundary_edges = sum(int(component["boundary_edges_4conn"]) for component in components)
    return {
        "component_count": len(components),
        "total_area": int(total_area),
        "total_boundary_pixels_4conn": int(total_boundary_pixels),
        "total_boundary_edges_4conn": int(total_boundary_edges),
        "emitted_components": emitted,
        "emitted_component_count": len(emitted),
        "omitted_component_count": int(max(0, len(components) - len(emitted))),
        "component_emit_policy": "largest_area_then_top_left_then_scan_index",
    }


def _temporal_change_from_previous(current: np.ndarray, previous: np.ndarray | None) -> dict[str, Any]:
    if previous is None:
        return {
            "previous_frame": None,
            "changed_pixels": None,
            "changed_fraction": None,
        }
    changed = int(np.count_nonzero(current != previous))
    return {
        "previous_frame": "index_minus_1",
        "changed_pixels": changed,
        "changed_fraction": _round_float(changed / current.size),
    }


def _transition_counts(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    class_count = max(CLASS_IDS) + 1
    encoded = previous.reshape(-1).astype(np.int64) * class_count + current.reshape(-1).astype(np.int64)
    counts = np.bincount(encoded, minlength=class_count * class_count)
    return counts.reshape(class_count, class_count)


def _temporal_stats(masks: np.ndarray) -> dict[str, Any]:
    if masks.shape[0] <= 1:
        return {
            "pair_count": 0,
            "changed_pixels_by_pair": [],
            "total_changed_pixels": 0,
            "mean_changed_pixels_per_pair": None,
            "mean_changed_fraction_per_pair": None,
            "max_changed_pair": None,
            "transition_counts": {str(src): {str(dst): 0 for dst in CLASS_IDS} for src in CLASS_IDS},
        }

    changed_by_pair: list[dict[str, Any]] = []
    transitions = np.zeros((max(CLASS_IDS) + 1, max(CLASS_IDS) + 1), dtype=np.int64)
    for frame_index in range(1, int(masks.shape[0])):
        previous = masks[frame_index - 1]
        current = masks[frame_index]
        changed = int(np.count_nonzero(current != previous))
        changed_by_pair.append(
            {
                "from_frame": int(frame_index - 1),
                "to_frame": int(frame_index),
                "changed_pixels": changed,
                "changed_fraction": _round_float(changed / current.size),
            }
        )
        transitions += _transition_counts(previous, current)

    total_changed = sum(item["changed_pixels"] for item in changed_by_pair)
    max_pair = max(changed_by_pair, key=lambda item: (item["changed_pixels"], -item["from_frame"]))
    pair_count = len(changed_by_pair)
    return {
        "pair_count": int(pair_count),
        "changed_pixels_by_pair": changed_by_pair,
        "total_changed_pixels": int(total_changed),
        "mean_changed_pixels_per_pair": _round_float(total_changed / pair_count),
        "mean_changed_fraction_per_pair": _round_float(
            sum(item["changed_fraction"] for item in changed_by_pair) / pair_count
        ),
        "max_changed_pair": max_pair,
        "transition_counts": {
            str(src): {str(dst): int(transitions[src, dst]) for dst in CLASS_IDS} for src in CLASS_IDS
        },
    }


def _primitive_diagnostics(masks: torch.Tensor, config: DiagnosticConfig) -> dict[str, Any]:
    if config.connectivity != 4:
        raise ValueError("only 4-connectivity is supported by this deterministic prototype")
    masks_np = masks.numpy()
    frame_records: list[dict[str, Any]] = []
    total_components_by_class = {str(class_id): 0 for class_id in CLASS_IDS}
    max_components_in_frame_by_class = {str(class_id): 0 for class_id in CLASS_IDS}
    total_area_by_class = {str(class_id): 0 for class_id in CLASS_IDS}
    total_boundary_pixels_by_class = {str(class_id): 0 for class_id in CLASS_IDS}
    total_boundary_edges_by_class = {str(class_id): 0 for class_id in CLASS_IDS}

    previous: np.ndarray | None = None
    for frame_index, frame in enumerate(masks_np):
        frame_components = _frame_components(frame)
        class_records = {}
        frame_component_total = 0
        for class_id in CLASS_IDS:
            record = _class_component_record(
                frame_components[class_id],
                max_components_per_class=config.max_components_per_class,
            )
            class_key = str(class_id)
            class_records[class_key] = record
            frame_component_total += record["component_count"]
            total_components_by_class[class_key] += record["component_count"]
            max_components_in_frame_by_class[class_key] = max(
                max_components_in_frame_by_class[class_key], record["component_count"]
            )
            total_area_by_class[class_key] += record["total_area"]
            total_boundary_pixels_by_class[class_key] += record["total_boundary_pixels_4conn"]
            total_boundary_edges_by_class[class_key] += record["total_boundary_edges_4conn"]

        frame_records.append(
            {
                "frame_index": int(frame_index),
                "component_count_total": int(frame_component_total),
                "temporal_change_from_previous": _temporal_change_from_previous(frame, previous),
                "classes": class_records,
            }
        )
        previous = frame

    frame_count = int(masks.shape[0])
    return {
        "class_ids": [int(class_id) for class_id in CLASS_IDS],
        "connectivity": int(config.connectivity),
        "frame_count": frame_count,
        "height": int(masks.shape[1]),
        "width": int(masks.shape[2]),
        "summary": {
            "total_components_by_class": total_components_by_class,
            "max_components_in_frame_by_class": max_components_in_frame_by_class,
            "mean_components_per_frame_by_class": {
                str(class_id): _round_float(total_components_by_class[str(class_id)] / frame_count)
                for class_id in CLASS_IDS
            },
            "total_area_by_class": total_area_by_class,
            "total_boundary_pixels_by_class": total_boundary_pixels_by_class,
            "total_boundary_edges_by_class": total_boundary_edges_by_class,
        },
        "temporal": _temporal_stats(masks_np),
        "frames": frame_records,
    }


def _module_availability() -> dict[str, bool]:
    modules = {
        "tac.mask_codec": "mask_codec",
        "numpy": "numpy",
        "torch": "torch",
    }
    return {label: importlib.util.find_spec(module) is not None for module, label in modules.items()}


def _selected_environment() -> dict[str, str]:
    keys = [
        "TAC_FFMPEG",
        "TAC_UPSTREAM_DIR",
        "PYTHONHASHSEED",
        "UV_PROJECT_ENVIRONMENT",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _provenance(command: list[str] | None) -> dict[str, Any]:
    upstream_ffmpeg = Path(os.environ.get("TAC_UPSTREAM_DIR", str(REPO_ROOT / "upstream"))) / "ffmpeg-new"
    return {
        "tool": "experiments/alpha_primitive_mask_diagnostics.py",
        "command": list(command) if command is not None else list(sys.argv),
        "cwd": str(Path.cwd()),
        "repo_root": str(REPO_ROOT),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "selected_environment": _selected_environment(),
        "module_available": _module_availability(),
        "ffmpeg_resolution": {
            "TAC_FFMPEG": os.environ.get("TAC_FFMPEG"),
            "upstream_ffmpeg_new": str(upstream_ffmpeg),
            "upstream_ffmpeg_new_exists": upstream_ffmpeg.exists(),
            "path_ffmpeg": shutil.which("ffmpeg"),
        },
    }


def _assert_empirical_no_promotion(report: dict[str, Any]) -> None:
    if report.get("score_claim") is not False:
        raise AssertionError("top-level score_claim must be false")
    if report.get("promotion_eligible") is not False:
        raise AssertionError("top-level promotion_eligible must be false")
    if report.get("evidence_grade") != EVIDENCE_GRADE:
        raise AssertionError("top-level evidence_grade must be empirical")
    if "contest_auth_eval.py --device cuda" not in report.get("canonical_score_source_required", ""):
        raise AssertionError("report must state exact CUDA auth eval score source")
    if report.get("scorer_network_loaded") is not False:
        raise AssertionError("diagnostic must not load scorer networks")


def _build_diagnostic_report_from_masks(
    *,
    masks: torch.Tensor,
    source_meta: dict[str, Any],
    config: DiagnosticConfig,
    command: list[str] | None = None,
) -> dict[str, Any]:
    original_masks = _validate_masks(masks)
    analyzed_masks, frame_subset = _slice_masks(original_masks, config.max_frames)
    source = dict(source_meta)
    source["decoded_masks"] = _mask_stats(original_masks)
    source["analyzed_masks"] = _mask_stats(analyzed_masks)
    source["frame_subset"] = frame_subset

    report = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "local_diagnostic_only": True,
        "scorer_network_loaded": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "purpose": "Alpha mask primitive diagnostics for representation redesign; not score evidence",
        "diagnostic_config": dataclasses.asdict(config),
        "source": source,
        "diagnostics": _primitive_diagnostics(analyzed_masks, config),
        "provenance": _provenance(command),
    }
    _assert_empirical_no_promotion(report)
    return report


def diagnose_archive(
    *,
    archive: Path,
    mask_member: str,
    output: Path,
    config: DiagnosticConfig,
    command: list[str] | None = None,
) -> dict[str, Any]:
    member_data, source_meta = _read_archive_member(archive, mask_member)
    masks = _decode_av1_masks_from_member(member_data, mask_member)
    report = _build_diagnostic_report_from_masks(
        masks=masks,
        source_meta=source_meta,
        config=config,
        command=command,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--mask-member", default="masks.mkv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"Maximum decoded frames to analyze by default ({DEFAULT_MAX_FRAMES}); use --all-frames for full corpus.",
    )
    parser.add_argument("--all-frames", action="store_true", help="Analyze every decoded mask frame.")
    parser.add_argument("--max-components-per-class", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = DiagnosticConfig(
        max_frames=None if args.all_frames else args.max_frames,
        max_components_per_class=args.max_components_per_class,
    )
    report = diagnose_archive(
        archive=args.archive,
        mask_member=args.mask_member,
        output=args.output,
        config=config,
        command=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
    )
    frame_count = report["diagnostics"]["frame_count"]
    print(
        f"[empirical:{args.output}] Alpha primitive diagnostics wrote {frame_count} frame records. "
        "No score claim; CUDA auth eval required.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
