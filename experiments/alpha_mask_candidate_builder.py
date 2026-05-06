#!/usr/bin/env python3
"""Build deterministic Alpha mask candidate payloads from an archive masks.mkv.

This is a bounded, custody-oriented Alpha builder for the post-screen step:
it turns a decoded archive mask stream into an Alpha4 grayscale-LUT payload and
a deterministic residual-run repair payload that later archive builders can
consume when assembling exact-eval finalist archives.

It does not run scorer networks and it does not produce score evidence. Any
candidate artifact emitted here still requires a deterministic archive builder
and exact CUDA auth eval through:

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
import struct
import subprocess
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

SCHEMA = "alpha_mask_candidate_builder_v1"
SWEEP_SCHEMA = "alpha_mask_candidate_crf_sweep_v1"
REPAIR_SCHEMA = "alpha4_residual_repair_amr1_v1"
EVIDENCE_GRADE = "empirical"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
SCORE_CLAIM_WARNING = (
    "No score claim is made by this builder. A deterministic finalist archive "
    "and exact CUDA auth eval are required before any score claim, promotion, "
    "ranking, or method retirement."
)

DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/alpha_mask_candidate_builder"
DEFAULT_MAX_FRAMES = 64
DEFAULT_MAX_REPAIR_PIXELS = 1_000_000
DEFAULT_MAX_REPAIR_RUNS = 250_000
CLASS_IDS = (0, 1, 2, 3, 4)
DEFAULT_CLASS_PRIORITY = (2, 1, 3, 4, 0)

GRAYSCALE_MEMBER = "grayscale.mkv"
REPAIR_MEMBER = "alpha4_residual_repair.amr1"
MANIFEST_NAME = "alpha_mask_candidate_manifest.json"
SWEEP_MANIFEST_NAME = "alpha_mask_candidate_sweep_manifest.json"

_HIDDEN_SYSTEM_NAMES = {"__MACOSX", ".DS_Store", "Thumbs.db"}
_REPAIR_MAGIC = b"AMR1"
_REPAIR_HEADER_STRUCT = ">I"
_REPAIR_RECORD_STRUCT = ">IHHHB"
_REPAIR_RECORD_SIZE = struct.calcsize(_REPAIR_RECORD_STRUCT)


@dataclass(frozen=True)
class BuilderConfig:
    max_frames: int | None = DEFAULT_MAX_FRAMES
    alpha4_crf: int = 50
    alpha4_fps: int = 20
    class_priority: tuple[int, ...] = DEFAULT_CLASS_PRIORITY
    max_repair_pixels: int | None = DEFAULT_MAX_REPAIR_PIXELS
    max_repair_runs: int | None = DEFAULT_MAX_REPAIR_RUNS
    fail_on_partial_repair: bool = True


@dataclass(frozen=True)
class RepairRun:
    frame_index: int
    y: int
    x0: int
    length: int
    class_id: int


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


def _resolve_executable(value: str) -> str | None:
    path = Path(value)
    if path.exists() and os.access(path, os.X_OK):
        return str(path.resolve())
    resolved = shutil.which(value)
    return str(Path(resolved).resolve()) if resolved else None


def _tool_usable(executable: str) -> bool:
    try:
        proc = subprocess.run(
            [executable, "-hide_banner", "-version"],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _resolve_ffmpeg_binary() -> str:
    override = os.environ.get("TAC_FFMPEG")
    if override:
        resolved = _resolve_executable(override)
        if resolved is None or not _tool_usable(resolved):
            raise RuntimeError(f"TAC_FFMPEG={override!r} is not a usable ffmpeg")
        return resolved

    upstream_dir = Path(os.environ.get("TAC_UPSTREAM_DIR", str(REPO_ROOT / "upstream")))
    upstream_ffmpeg = upstream_dir / "ffmpeg-new"
    if (
        upstream_ffmpeg.exists()
        and os.access(upstream_ffmpeg, os.X_OK)
        and _tool_usable(str(upstream_ffmpeg.resolve()))
    ):
        return str(upstream_ffmpeg.resolve())

    resolved = shutil.which("ffmpeg")
    if resolved is None or not _tool_usable(resolved):
        raise RuntimeError("ffmpeg not found; set TAC_FFMPEG or install ffmpeg")
    return str(Path(resolved).resolve())


def _resolve_ffprobe_binary(ffmpeg_binary: str) -> str:
    override = os.environ.get("TAC_FFPROBE")
    if override:
        resolved = _resolve_executable(override)
        if resolved is None or not _tool_usable(resolved):
            raise RuntimeError(f"TAC_FFPROBE={override!r} is not a usable ffprobe")
        return resolved

    ffmpeg_path = Path(ffmpeg_binary)
    sibling = ffmpeg_path.with_name("ffprobe")
    if sibling.exists() and os.access(sibling, os.X_OK) and _tool_usable(str(sibling.resolve())):
        return str(sibling.resolve())

    resolved = shutil.which("ffprobe")
    if resolved is None or not _tool_usable(resolved):
        raise RuntimeError("ffprobe not found; set TAC_FFPROBE or install ffprobe")
    return str(Path(resolved).resolve())


def _probe_video(path: Path, *, ffprobe: str) -> dict[str, Any]:
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")
    try:
        payload = json.loads(proc.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ffprobe did not return usable video dimensions: {proc.stdout!r}") from exc
    frame_count_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    frame_count = None
    if frame_count_raw not in (None, "N/A"):
        try:
            frame_count = int(frame_count_raw)
        except (TypeError, ValueError):
            frame_count = None
    return {"width": width, "height": height, "reported_frames": frame_count}


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
    if masks.shape[1] > 65535 or masks.shape[2] > 65535:
        raise ValueError(f"repair payload supports H,W <= 65535, got {tuple(masks.shape[1:])}")
    if masks.shape[0] > 0xFFFFFFFF:
        raise ValueError(f"repair payload supports frame count <= 2^32-1, got {masks.shape[0]}")
    return masks


def _validate_config(config: BuilderConfig) -> None:
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError(f"max_frames must be positive when provided, got {config.max_frames}")
    if not (0 <= config.alpha4_crf <= 63):
        raise ValueError(f"alpha4_crf must be in [0,63], got {config.alpha4_crf}")
    if config.alpha4_fps <= 0:
        raise ValueError(f"alpha4_fps must be positive, got {config.alpha4_fps}")
    if sorted(config.class_priority) != list(CLASS_IDS):
        raise ValueError(
            "class_priority must be a permutation of "
            f"{list(CLASS_IDS)}, got {list(config.class_priority)}"
        )
    if config.max_repair_pixels is not None and config.max_repair_pixels <= 0:
        raise ValueError(f"max_repair_pixels must be positive when provided, got {config.max_repair_pixels}")
    if config.max_repair_runs is not None and config.max_repair_runs <= 0:
        raise ValueError(f"max_repair_runs must be positive when provided, got {config.max_repair_runs}")


def _decode_legacy_av1_masks_from_member(
    data: bytes,
    member: str,
    *,
    max_frames: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    suffix = PurePosixPath(member).suffix or ".mkv"
    ffmpeg = _resolve_ffmpeg_binary()
    ffprobe = _resolve_ffprobe_binary(ffmpeg)
    with tempfile.TemporaryDirectory() as tmp_dir:
        mask_path = Path(tmp_dir) / f"mask_member{suffix}"
        mask_path.write_bytes(data)
        probe = _probe_video(mask_path, ffprobe=ffprobe)
        width = int(probe["width"])
        height = int(probe["height"])
        cmd = [
            ffmpeg,
            "-i",
            str(mask_path),
        ]
        if max_frames is not None:
            cmd.extend(["-frames:v", str(max_frames)])
        cmd.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-v",
                "error",
                "pipe:1",
            ]
        )
        proc = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg mask decode failed: {stderr}")

    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = height * width
    if frame_size <= 0:
        raise ValueError(f"invalid decoded frame size {height}x{width}")
    if raw.size % frame_size != 0:
        raise ValueError(f"decoded data size {raw.size} not divisible by frame size {height}x{width}")
    frame_count = raw.size // frame_size
    if frame_count <= 0:
        raise ValueError("ffmpeg decoded zero mask frames")
    if max_frames is not None and frame_count > max_frames:
        raise ValueError(f"decoded {frame_count} frames despite max_frames={max_frames}")

    pixels = raw.reshape(frame_count, height, width)
    scale_factor = 255 // (len(CLASS_IDS) - 1)
    classes = np.round(pixels.astype(np.float32) / scale_factor).astype(np.int64)
    classes = np.clip(classes, min(CLASS_IDS), max(CLASS_IDS))
    masks = _validate_masks(torch.from_numpy(classes))
    return masks, {
        "decoder": "legacy_av1_class_scaled_gray",
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "width": width,
        "height": height,
        "reported_frames": probe["reported_frames"],
        "decoded_frames": int(frame_count),
        "max_frames": max_frames,
        "truncated_by_builder": bool(max_frames is not None and probe["reported_frames"] != frame_count),
    }


def _round_float(value: float, digits: int = 12) -> float:
    return round(float(value), digits)


def _mask_stats(masks: torch.Tensor) -> dict[str, Any]:
    masks = _validate_masks(masks)
    counts = torch.bincount(masks.reshape(-1), minlength=max(CLASS_IDS) + 1).to(torch.int64)
    total = int(masks.numel())
    vertical_edges = int((masks[:, 1:, :] != masks[:, :-1, :]).sum().item()) if masks.shape[1] > 1 else 0
    horizontal_edges = int((masks[:, :, 1:] != masks[:, :, :-1]).sum().item()) if masks.shape[2] > 1 else 0
    temporal_changes = int((masks[1:] != masks[:-1]).sum().item()) if masks.shape[0] > 1 else 0
    return {
        "shape": [int(v) for v in masks.shape],
        "dtype": str(masks.dtype),
        "num_pixels": total,
        "class_histogram": {str(class_id): int(counts[class_id].item()) for class_id in CLASS_IDS},
        "class_fractions": {
            str(class_id): _round_float(int(counts[class_id].item()) / total) for class_id in CLASS_IDS
        },
        "spatial_boundary_edges_4conn": vertical_edges + horizontal_edges,
        "temporal_changed_pixels": temporal_changes,
        "class_id_u8_sha256": _tensor_u8_sha256(masks),
    }


def _tensor_u8_sha256(masks: torch.Tensor) -> str:
    masks = _validate_masks(masks)
    return _sha256_bytes(masks.to(torch.uint8).contiguous().numpy().tobytes())


def _agreement_metrics(source: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    source = _validate_masks(source)
    candidate = _validate_masks(candidate)
    if tuple(source.shape) != tuple(candidate.shape):
        return {
            "shape_match": False,
            "source_shape": [int(v) for v in source.shape],
            "candidate_shape": [int(v) for v in candidate.shape],
            "argmax_agreement": None,
            "argmax_disagreement": None,
            "different_pixels": None,
            "num_pixels": int(source.numel()),
        }
    diff = source != candidate
    different = int(diff.sum().item())
    total = int(source.numel())
    confusion = torch.zeros((max(CLASS_IDS) + 1, max(CLASS_IDS) + 1), dtype=torch.int64)
    if different:
        encoded = source[diff].reshape(-1) * (max(CLASS_IDS) + 1) + candidate[diff].reshape(-1)
        counts = torch.bincount(encoded, minlength=confusion.numel()).reshape_as(confusion)
        confusion += counts
    different_by_source = {
        str(class_id): int((diff & (source == class_id)).sum().item()) for class_id in CLASS_IDS
    }
    different_by_candidate = {
        str(class_id): int((diff & (candidate == class_id)).sum().item()) for class_id in CLASS_IDS
    }
    return {
        "shape_match": True,
        "source_shape": [int(v) for v in source.shape],
        "candidate_shape": [int(v) for v in candidate.shape],
        "num_pixels": total,
        "equal_pixels": total - different,
        "different_pixels": different,
        "argmax_agreement": _round_float(1.0 - (different / total)),
        "argmax_disagreement": _round_float(different / total),
        "different_pixels_by_source_class": different_by_source,
        "different_pixels_by_candidate_class": different_by_candidate,
        "residual_confusion_source_to_candidate": {
            str(src): {str(dst): int(confusion[src, dst].item()) for dst in CLASS_IDS}
            for src in CLASS_IDS
        },
    }


def _encode_gray_av1(gray: torch.Tensor, output_path: Path, *, crf: int, fps: int) -> dict[str, Any]:
    if gray.dtype != torch.uint8 or gray.dim() != 3:
        raise ValueError(f"gray must be uint8 (T,H,W), got {gray.dtype} {tuple(gray.shape)}")
    t, h, w = [int(v) for v in gray.shape]
    raw = gray.cpu().contiguous().numpy().tobytes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _resolve_ffmpeg_binary(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+bitexact",
        "-flags",
        "+bitexact",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "gray",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-map_metadata",
        "-1",
        "-c:v",
        "libsvtav1",
        "-crf",
        str(crf),
        "-preset",
        "6",
        "-svtav1-params",
        "enable-restoration=0:enable-cdef=0:lp=1",
        "-pix_fmt",
        "gray",
        "-an",
        str(output_path),
    ]
    proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=600, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg alpha4 encode failed: {stderr}")
    payload = output_path.read_bytes()
    return {
        "path": output_path,
        "bytes": payload,
        "command": cmd,
        "frames": t,
        "height": h,
        "width": w,
    }


def _decode_gray_av1(path: Path, *, expected_shape: tuple[int, int, int]) -> torch.Tensor:
    from tac.mask_grayscale_lut import decode_grayscale_to_classes

    t, h, w = expected_shape
    cmd = [
        _resolve_ffmpeg_binary(),
        "-i",
        str(path),
        "-frames:v",
        str(t),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-v",
        "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg alpha4 decode failed: {stderr}")
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    expected = t * h * w
    if raw.size != expected:
        raise ValueError(f"alpha4 decoded {raw.size} pixels, expected {expected}")
    gray = torch.from_numpy(raw.copy()).reshape(t, h, w)
    return _validate_masks(decode_grayscale_to_classes(gray))


def _parse_class_priority(value: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            class_id = int(token)
        except ValueError as exc:
            raise ValueError(f"class priority entries must be integers, got {raw!r}") from exc
        parsed.append(class_id)
    priority = tuple(parsed)
    if sorted(priority) != list(CLASS_IDS):
        raise ValueError(f"class priority must be a permutation of {list(CLASS_IDS)}, got {list(priority)}")
    return priority


def _parse_alpha4_crf_sweep(value: str) -> tuple[int, ...]:
    crfs: list[int] = []
    seen: set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            crf = int(token)
        except ValueError as exc:
            raise ValueError(f"alpha4 CRF sweep entries must be integers, got {raw!r}") from exc
        if not (0 <= crf <= 63):
            raise ValueError(f"alpha4 CRF sweep entries must be in [0,63], got {crf}")
        if crf in seen:
            raise ValueError(f"alpha4 CRF sweep contains duplicate value {crf}")
        seen.add(crf)
        crfs.append(crf)
    if not crfs:
        raise ValueError("alpha4 CRF sweep must contain at least one value")
    return tuple(crfs)


def _build_repair_runs(
    source: torch.Tensor,
    candidate: torch.Tensor,
    *,
    config: BuilderConfig,
) -> tuple[list[RepairRun], dict[str, Any]]:
    source = _validate_masks(source)
    candidate = _validate_masks(candidate)
    if tuple(source.shape) != tuple(candidate.shape):
        raise ValueError(f"source/candidate shape mismatch: {tuple(source.shape)} vs {tuple(candidate.shape)}")

    source_np = source.numpy()
    candidate_np = candidate.numpy()
    diff_np = source_np != candidate_np
    total_residual_pixels = int(np.count_nonzero(diff_np))
    runs: list[RepairRun] = []
    selected_pixels = 0
    selected_by_class = {str(class_id): 0 for class_id in CLASS_IDS}
    total_by_class = {
        str(class_id): int(np.count_nonzero(diff_np & (source_np == class_id))) for class_id in CLASS_IDS
    }
    partial_reason: str | None = None

    for class_id in config.class_priority:
        for frame_index in range(source_np.shape[0]):
            frame_source = source_np[frame_index]
            frame_diff = diff_np[frame_index]
            for y in range(source_np.shape[1]):
                xs = np.flatnonzero(frame_diff[y] & (frame_source[y] == class_id))
                if xs.size == 0:
                    continue
                split_points = np.flatnonzero(np.diff(xs) != 1) + 1
                for segment in np.split(xs, split_points):
                    if segment.size == 0:
                        continue
                    x0 = int(segment[0])
                    length = int(segment[-1] - segment[0] + 1)
                    projected_pixels = selected_pixels + length
                    projected_runs = len(runs) + 1
                    if config.max_repair_pixels is not None and projected_pixels > config.max_repair_pixels:
                        partial_reason = (
                            "max_repair_pixels would be exceeded: "
                            f"{projected_pixels} > {config.max_repair_pixels}"
                        )
                    if config.max_repair_runs is not None and projected_runs > config.max_repair_runs:
                        partial_reason = (
                            "max_repair_runs would be exceeded: "
                            f"{projected_runs} > {config.max_repair_runs}"
                        )
                    if partial_reason is not None:
                        if config.fail_on_partial_repair:
                            raise ValueError(
                                "repair payload would be partial; rerun with larger limits or "
                                f"--allow-partial-repair ({partial_reason})"
                            )
                        return runs, _repair_selection_meta(
                            total_residual_pixels=total_residual_pixels,
                            selected_pixels=selected_pixels,
                            selected_by_class=selected_by_class,
                            total_by_class=total_by_class,
                            partial_reason=partial_reason,
                            config=config,
                        )
                    runs.append(
                        RepairRun(
                            frame_index=int(frame_index),
                            y=int(y),
                            x0=x0,
                            length=length,
                            class_id=int(class_id),
                        )
                    )
                    selected_pixels += length
                    selected_by_class[str(class_id)] += length

    return runs, _repair_selection_meta(
        total_residual_pixels=total_residual_pixels,
        selected_pixels=selected_pixels,
        selected_by_class=selected_by_class,
        total_by_class=total_by_class,
        partial_reason=None,
        config=config,
    )


def _repair_selection_meta(
    *,
    total_residual_pixels: int,
    selected_pixels: int,
    selected_by_class: dict[str, int],
    total_by_class: dict[str, int],
    partial_reason: str | None,
    config: BuilderConfig,
) -> dict[str, Any]:
    coverage = 1.0 if total_residual_pixels == 0 else selected_pixels / total_residual_pixels
    return {
        "strategy": "alpha4_residual_runs_by_source_class_priority",
        "class_priority": [int(v) for v in config.class_priority],
        "total_residual_pixels": int(total_residual_pixels),
        "selected_repair_pixels": int(selected_pixels),
        "selected_repair_pixels_by_source_class": selected_by_class,
        "total_residual_pixels_by_source_class": total_by_class,
        "residual_pixel_coverage": _round_float(coverage),
        "partial_repair": partial_reason is not None,
        "partial_reason": partial_reason,
        "fail_on_partial_repair": bool(config.fail_on_partial_repair),
        "max_repair_pixels": config.max_repair_pixels,
        "max_repair_runs": config.max_repair_runs,
    }


def _encode_repair_payload(
    runs: list[RepairRun],
    *,
    shape: tuple[int, int, int],
    source_mask_sha256: str,
    candidate_mask_sha256: str,
    selection_meta: dict[str, Any],
) -> bytes:
    t, h, w = shape
    header = {
        "schema": REPAIR_SCHEMA,
        "magic": _REPAIR_MAGIC.decode("ascii"),
        "shape": [int(t), int(h), int(w)],
        "source_mask_u8_sha256": source_mask_sha256,
        "candidate_mask_u8_sha256": candidate_mask_sha256,
        "record_struct": _REPAIR_RECORD_STRUCT,
        "record_count": len(runs),
        "operation": "set decoded_candidate[frame,y,x0:x0+length] = class_id for each record",
        "selection": selection_meta,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = bytearray()
    payload.extend(_REPAIR_MAGIC)
    payload.extend(struct.pack(_REPAIR_HEADER_STRUCT, len(header_bytes)))
    payload.extend(header_bytes)
    for run in runs:
        if run.frame_index < 0 or run.frame_index >= t:
            raise ValueError(f"repair run frame out of range: {run}")
        if run.y < 0 or run.y >= h:
            raise ValueError(f"repair run y out of range: {run}")
        if run.x0 < 0 or run.x0 >= w:
            raise ValueError(f"repair run x0 out of range: {run}")
        if run.length <= 0 or run.x0 + run.length > w or run.length > 65535:
            raise ValueError(f"repair run length out of range: {run}")
        if run.class_id not in CLASS_IDS:
            raise ValueError(f"repair run class out of range: {run}")
        payload.extend(
            struct.pack(
                _REPAIR_RECORD_STRUCT,
                int(run.frame_index),
                int(run.y),
                int(run.x0),
                int(run.length),
                int(run.class_id),
            )
        )
    return bytes(payload)


def _decode_repair_payload(payload: bytes) -> tuple[dict[str, Any], list[RepairRun]]:
    if not payload.startswith(_REPAIR_MAGIC):
        raise ValueError("repair payload missing AMR1 magic")
    offset = len(_REPAIR_MAGIC)
    if len(payload) < offset + struct.calcsize(_REPAIR_HEADER_STRUCT):
        raise ValueError("repair payload missing header length")
    (header_length,) = struct.unpack(_REPAIR_HEADER_STRUCT, payload[offset : offset + 4])
    offset += 4
    header_end = offset + header_length
    if header_end > len(payload):
        raise ValueError("repair payload header extends past payload")
    header = json.loads(payload[offset:header_end].decode("utf-8"))
    offset = header_end
    record_count = int(header["record_count"])
    expected = offset + record_count * _REPAIR_RECORD_SIZE
    if expected != len(payload):
        raise ValueError(f"repair payload size mismatch: expected {expected}, got {len(payload)}")
    runs: list[RepairRun] = []
    for _ in range(record_count):
        frame_index, y, x0, length, class_id = struct.unpack(
            _REPAIR_RECORD_STRUCT, payload[offset : offset + _REPAIR_RECORD_SIZE]
        )
        offset += _REPAIR_RECORD_SIZE
        runs.append(
            RepairRun(
                frame_index=int(frame_index),
                y=int(y),
                x0=int(x0),
                length=int(length),
                class_id=int(class_id),
            )
        )
    return header, runs


def _apply_repair_payload(candidate: torch.Tensor, payload: bytes) -> torch.Tensor:
    candidate = _validate_masks(candidate).clone()
    header, runs = _decode_repair_payload(payload)
    if tuple(header["shape"]) != tuple(int(v) for v in candidate.shape):
        raise ValueError(f"repair payload shape {header['shape']} does not match candidate {tuple(candidate.shape)}")
    for run in runs:
        candidate[run.frame_index, run.y, run.x0 : run.x0 + run.length] = run.class_id
    return _validate_masks(candidate)


def _artifact_record(path: Path, *, role: str, archive_member: str | None = None) -> dict[str, Any]:
    record = {
        "role": role,
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }
    if archive_member is not None:
        record["candidate_archive_member"] = archive_member
    return record


def _selected_environment() -> dict[str, str]:
    keys = [
        "TAC_FFMPEG",
        "TAC_FFPROBE",
        "TAC_UPSTREAM_DIR",
        "PYTHONHASHSEED",
        "UV_PROJECT_ENVIRONMENT",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _module_availability() -> dict[str, bool]:
    modules = {
        "tac.mask_grayscale_lut": "alpha4_grayscale_lut",
        "numpy": "numpy",
        "torch": "torch",
    }
    return {label: importlib.util.find_spec(module) is not None for module, label in modules.items()}


def _binary_meta(path: str | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None}
    binary = Path(path)
    return {
        "path": str(binary),
        "exists": binary.exists(),
        "size_bytes": int(binary.stat().st_size) if binary.exists() else None,
        "sha256": _sha256_file(binary) if binary.exists() and binary.is_file() else None,
    }


def _provenance(command: list[str] | None) -> dict[str, Any]:
    ffmpeg = None
    ffprobe = None
    try:
        ffmpeg = _resolve_ffmpeg_binary()
        ffprobe = _resolve_ffprobe_binary(ffmpeg)
    except RuntimeError:
        pass
    upstream_ffmpeg = Path(os.environ.get("TAC_UPSTREAM_DIR", str(REPO_ROOT / "upstream"))) / "ffmpeg-new"
    return {
        "tool": "experiments/alpha_mask_candidate_builder.py",
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
            "TAC_FFPROBE": os.environ.get("TAC_FFPROBE"),
            "upstream_ffmpeg_new": str(upstream_ffmpeg),
            "upstream_ffmpeg_new_exists": upstream_ffmpeg.exists(),
            "path_ffmpeg": shutil.which("ffmpeg"),
            "path_ffprobe": shutil.which("ffprobe"),
            "resolved_ffmpeg": _binary_meta(ffmpeg),
            "resolved_ffprobe": _binary_meta(ffprobe),
        },
    }


def _assert_empirical_no_promotion(report: dict[str, Any]) -> None:
    if report.get("score_claim") is not False:
        raise AssertionError("top-level score_claim must be false")
    if report.get("promotion_eligible") is not False:
        raise AssertionError("top-level promotion_eligible must be false")
    if report.get("evidence_grade") != EVIDENCE_GRADE:
        raise AssertionError("top-level evidence_grade must be empirical")
    if report.get("scorer_network_loaded") is not False:
        raise AssertionError("builder must not load scorer networks")
    if "contest_auth_eval.py --device cuda" not in report.get("canonical_score_source_required", ""):
        raise AssertionError("report must state exact CUDA auth eval score source")
    for artifact in report.get("candidate", {}).get("artifacts", []):
        if "sha256" not in artifact or "size_bytes" not in artifact:
            raise AssertionError(f"artifact missing custody fields: {artifact}")


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [GRAYSCALE_MEMBER, REPAIR_MEMBER, MANIFEST_NAME]
    existing = [name for name in targets if (output_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(
            f"{output_dir} already contains builder artifacts {existing}; use --force to overwrite"
        )


def _build_manifest(
    *,
    source_masks: torch.Tensor,
    candidate_masks: torch.Tensor,
    repaired_masks: torch.Tensor,
    source_meta: dict[str, Any],
    decode_meta: dict[str, Any] | None,
    output_dir: Path,
    grayscale_record: dict[str, Any],
    repair_record: dict[str, Any],
    repair_selection: dict[str, Any],
    repair_header: dict[str, Any],
    config: BuilderConfig,
    encode_command: list[str],
    command: list[str] | None,
) -> dict[str, Any]:
    source_masks = _validate_masks(source_masks)
    candidate_masks = _validate_masks(candidate_masks)
    repaired_masks = _validate_masks(repaired_masks)
    source = dict(source_meta)
    source["decoded_masks"] = _mask_stats(source_masks)
    source["decode"] = decode_meta or {
        "decoder": "provided_tensor",
        "decoded_frames": int(source_masks.shape[0]),
        "max_frames": config.max_frames,
        "truncated_by_builder": False,
    }
    frame_subset = {
        "max_frames": config.max_frames,
        "decoded_frames": int(source_masks.shape[0]),
        "reported_source_frames": source["decode"].get("reported_frames"),
        "truncated": bool(source["decode"].get("truncated_by_builder", False)),
    }
    full_sequence = not frame_subset["truncated"]
    repaired_agreement = _agreement_metrics(source_masks, repaired_masks)
    repair_full = repair_selection["residual_pixel_coverage"] == 1.0 and not repair_selection["partial_repair"]
    exact_eval_candidate_ready = bool(full_sequence and repair_full)
    report = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "local_builder_only": True,
        "scorer_network_loaded": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "purpose": (
            "build Alpha4 grayscale-LUT plus deterministic residual repair artifacts "
            "for later exact-eval finalist archive assembly"
        ),
        "builder_config": _config_record(config),
        "source": source,
        "frame_subset": frame_subset,
        "candidate": {
            "name": "alpha4_grayscale_lut_with_residual_runs",
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": EVIDENCE_GRADE,
            "candidate_archive_readiness": {
                "artifacts_complete_for_selected_frames": True,
                "full_sequence_candidate": full_sequence,
                "residual_repair_full_coverage": repair_full,
                "exact_eval_archive_builder_required": True,
                "exact_cuda_auth_eval_required": CUDA_AUTH_EVAL_PATH,
                "ready_for_exact_eval_finalist_archive_assembly": exact_eval_candidate_ready,
            },
            "archive_member_plan": [
                {
                    "candidate_archive_member": GRAYSCALE_MEMBER,
                    "source_artifact_role": "alpha4_grayscale_lut_video",
                    "required": True,
                },
                {
                    "candidate_archive_member": REPAIR_MEMBER,
                    "source_artifact_role": "alpha4_residual_repair_payload",
                    "required": True,
                    "runtime_integration_required": True,
                },
            ],
            "artifacts": [grayscale_record, repair_record],
            "alpha4": {
                "payload_format": "grayscale_lut_av1_mask_video",
                "class_to_gray": _class_to_gray_manifest(),
                "crf": int(config.alpha4_crf),
                "fps": int(config.alpha4_fps),
                "encode_command": encode_command,
                "decoded_candidate_masks": _mask_stats(candidate_masks),
                "agreement_before_repair": _agreement_metrics(source_masks, candidate_masks),
            },
            "repair": {
                "payload_format": REPAIR_SCHEMA,
                "selection": repair_selection,
                "header": repair_header,
                "agreement_after_repair": repaired_agreement,
            },
        },
        "provenance": _provenance(command),
    }
    _assert_empirical_no_promotion(report)
    return report


def _class_to_gray_manifest() -> dict[str, int]:
    from tac.mask_grayscale_lut import CLASS_TO_GRAY

    return {str(k): int(v) for k, v in sorted(CLASS_TO_GRAY.items())}


def _config_record(config: BuilderConfig) -> dict[str, Any]:
    record = dataclasses.asdict(config)
    record["class_priority"] = [int(v) for v in config.class_priority]
    return record


def _build_candidate_artifacts_from_masks(
    *,
    masks: torch.Tensor,
    source_meta: dict[str, Any],
    output_dir: Path,
    config: BuilderConfig,
    command: list[str] | None = None,
    decode_meta: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    _validate_config(config)
    input_masks = _validate_masks(masks)
    source_masks = input_masks
    effective_decode_meta = decode_meta
    if config.max_frames is not None and int(input_masks.shape[0]) > config.max_frames:
        source_masks = input_masks[: config.max_frames].contiguous()
        effective_decode_meta = dict(decode_meta or {})
        effective_decode_meta.update(
            {
                "decoder": effective_decode_meta.get("decoder", "provided_tensor"),
                "reported_frames": effective_decode_meta.get("reported_frames", int(input_masks.shape[0])),
                "decoded_frames": int(source_masks.shape[0]),
                "max_frames": int(config.max_frames),
                "truncated_by_builder": True,
            }
        )
    _prepare_output_dir(output_dir, force=force)

    from tac.mask_grayscale_lut import encode_masks_grayscale

    grayscale_path = output_dir / GRAYSCALE_MEMBER
    gray = encode_masks_grayscale(source_masks)
    encoded = _encode_gray_av1(
        gray,
        grayscale_path,
        crf=config.alpha4_crf,
        fps=config.alpha4_fps,
    )
    candidate_masks = _decode_gray_av1(
        grayscale_path,
        expected_shape=tuple(int(v) for v in source_masks.shape),
    )
    candidate_sha = _tensor_u8_sha256(candidate_masks)
    source_sha = _tensor_u8_sha256(source_masks)
    runs, repair_selection = _build_repair_runs(
        source_masks,
        candidate_masks,
        config=config,
    )
    repair_payload = _encode_repair_payload(
        runs,
        shape=tuple(int(v) for v in source_masks.shape),
        source_mask_sha256=source_sha,
        candidate_mask_sha256=candidate_sha,
        selection_meta=repair_selection,
    )
    repair_path = output_dir / REPAIR_MEMBER
    repair_path.write_bytes(repair_payload)
    repair_header, _decoded_runs = _decode_repair_payload(repair_payload)
    repaired_masks = _apply_repair_payload(candidate_masks, repair_payload)

    grayscale_record = _artifact_record(
        grayscale_path,
        role="alpha4_grayscale_lut_video",
        archive_member=GRAYSCALE_MEMBER,
    )
    grayscale_record["raw_payload_sha256"] = _sha256_bytes(encoded["bytes"])
    repair_record = _artifact_record(
        repair_path,
        role="alpha4_residual_repair_payload",
        archive_member=REPAIR_MEMBER,
    )

    report = _build_manifest(
        source_masks=source_masks,
        candidate_masks=candidate_masks,
        repaired_masks=repaired_masks,
        source_meta=source_meta,
        decode_meta=effective_decode_meta,
        output_dir=output_dir,
        grayscale_record=grayscale_record,
        repair_record=repair_record,
        repair_selection=repair_selection,
        repair_header=repair_header,
        config=config,
        encode_command=list(encoded["command"]),
        command=command,
    )
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_candidate_from_archive(
    *,
    archive: Path,
    mask_member: str,
    output_dir: Path,
    config: BuilderConfig,
    command: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    _validate_config(config)
    member_data, source_meta = _read_archive_member(archive, mask_member)
    masks, decode_meta = _decode_legacy_av1_masks_from_member(
        member_data,
        mask_member,
        max_frames=config.max_frames,
    )
    return _build_candidate_artifacts_from_masks(
        masks=masks,
        source_meta=source_meta,
        output_dir=output_dir,
        config=config,
        command=command,
        decode_meta=decode_meta,
        force=force,
    )


def _prepare_sweep_output_dir(output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / SWEEP_MANIFEST_NAME
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} already exists; use --force to overwrite")


def _sweep_candidate_record(*, report: dict[str, Any], candidate_dir: Path, manifest_path: Path) -> dict[str, Any]:
    artifacts = {str(item["role"]): item for item in report["candidate"]["artifacts"]}
    grayscale = artifacts["alpha4_grayscale_lut_video"]
    repair = artifacts["alpha4_residual_repair_payload"]
    alpha4 = report["candidate"]["alpha4"]
    before = alpha4["agreement_before_repair"]
    source_mask_bytes = int(report["source"]["mask_member"]["size_bytes"])
    grayscale_bytes = int(grayscale["size_bytes"])
    repair_bytes = int(repair["size_bytes"])
    total_bytes = grayscale_bytes + repair_bytes
    return {
        "alpha4_crf": int(alpha4["crf"]),
        "alpha4_fps": int(alpha4["fps"]),
        "candidate_dir": str(candidate_dir),
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_size_bytes": int(manifest_path.stat().st_size),
        "candidate_manifest_sha256": _sha256_file(manifest_path),
        "grayscale_size_bytes": grayscale_bytes,
        "grayscale_sha256": grayscale["sha256"],
        "repair_size_bytes": repair_bytes,
        "repair_sha256": repair["sha256"],
        "candidate_total_payload_bytes": total_bytes,
        "source_mask_member_size_bytes": source_mask_bytes,
        "grayscale_delta_vs_source_mask_member_bytes": grayscale_bytes - source_mask_bytes,
        "total_payload_delta_vs_source_mask_member_bytes": total_bytes - source_mask_bytes,
        "grayscale_under_source_mask_member": bool(grayscale_bytes <= source_mask_bytes),
        "total_payload_under_source_mask_member": bool(total_bytes <= source_mask_bytes),
        "agreement_before_repair": before,
        "repair_selection": report["candidate"]["repair"]["selection"],
        "candidate_archive_readiness": report["candidate"]["candidate_archive_readiness"],
        "score_claim": False,
        "promotion_eligible": False,
    }


def _sweep_top(records: list[dict[str, Any]], *, key: str, limit: int = 12) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            int(record[key]),
            -float(record["agreement_before_repair"]["argmax_agreement"]),
            int(record["alpha4_crf"]),
        ),
    )[:limit]


def build_candidate_crf_sweep_from_archive(
    *,
    archive: Path,
    mask_member: str,
    output_dir: Path,
    config: BuilderConfig,
    crfs: tuple[int, ...],
    command: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    _validate_config(config)
    for crf in crfs:
        _validate_config(dataclasses.replace(config, alpha4_crf=int(crf)))
    _prepare_sweep_output_dir(output_dir, force=force)

    member_data, source_meta = _read_archive_member(archive, mask_member)
    masks, decode_meta = _decode_legacy_av1_masks_from_member(
        member_data,
        mask_member,
        max_frames=config.max_frames,
    )

    candidate_records: list[dict[str, Any]] = []
    for crf in crfs:
        candidate_config = dataclasses.replace(config, alpha4_crf=int(crf))
        candidate_dir = output_dir / f"crf_{int(crf):02d}"
        candidate_report = _build_candidate_artifacts_from_masks(
            masks=masks,
            source_meta=source_meta,
            output_dir=candidate_dir,
            config=candidate_config,
            command=command,
            decode_meta=decode_meta,
            force=force,
        )
        candidate_records.append(
            _sweep_candidate_record(
                report=candidate_report,
                candidate_dir=candidate_dir,
                manifest_path=candidate_dir / MANIFEST_NAME,
            )
        )

    source_mask_bytes = int(source_meta["mask_member"]["size_bytes"])
    byte_plausible_base = [
        record for record in candidate_records if record["grayscale_under_source_mask_member"] is True
    ]
    report = {
        "schema": SWEEP_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "local_builder_only": True,
        "scorer_network_loaded": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "purpose": (
            "run a deterministic Alpha4 compact-base CRF sweep over one decoded mask tensor; "
            "candidate subdirectories remain empirical/no-score artifacts"
        ),
        "builder_config_template": _config_record(config),
        "sweep_axis": "alpha4_crf",
        "sweep_values": [int(value) for value in crfs],
        "source": {
            **dict(source_meta),
            "decode": decode_meta,
            "decoded_masks": _mask_stats(masks),
        },
        "sweep_summary": {
            "candidate_count": int(len(candidate_records)),
            "source_mask_member_size_bytes": source_mask_bytes,
            "byte_plausible_base_candidate_count": int(len(byte_plausible_base)),
            "best_grayscale_by_bytes": _sweep_top(candidate_records, key="grayscale_size_bytes"),
            "best_full_repair_total_by_bytes": _sweep_top(
                candidate_records,
                key="candidate_total_payload_bytes",
            ),
            "next_step": (
                "run experiments/alpha_mask_residual_planner.py for candidate manifests on the "
                "byte-plausible base frontier; planner reports are still empirical/no-score"
            ),
        },
        "candidate_records": candidate_records,
        "provenance": _provenance(command),
    }
    _assert_empirical_no_promotion(report)
    manifest_path = output_dir / SWEEP_MANIFEST_NAME
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--mask-member", default="masks.mkv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"Maximum decoded frames to build by default ({DEFAULT_MAX_FRAMES}); use --all-frames for full corpus.",
    )
    parser.add_argument("--all-frames", action="store_true", help="Build artifacts from every decoded mask frame.")
    parser.add_argument("--alpha4-crf", type=int, default=50)
    parser.add_argument(
        "--alpha4-crf-sweep",
        default=None,
        help=(
            "Comma-separated CRF values. When set, output-dir is a sweep directory "
            "and each CRF writes a deterministic candidate subdirectory."
        ),
    )
    parser.add_argument("--alpha4-fps", type=int, default=20)
    parser.add_argument(
        "--class-priority",
        default=",".join(str(v) for v in DEFAULT_CLASS_PRIORITY),
        help="Comma-separated source-class repair priority permutation; default protects lane/road first.",
    )
    parser.add_argument("--max-repair-pixels", type=int, default=DEFAULT_MAX_REPAIR_PIXELS)
    parser.add_argument("--max-repair-runs", type=int, default=DEFAULT_MAX_REPAIR_RUNS)
    parser.add_argument(
        "--allow-partial-repair",
        action="store_true",
        help="Emit a partial repair payload if repair limits are hit. Default fails closed.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing builder artifacts in output-dir.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = BuilderConfig(
        max_frames=None if args.all_frames else args.max_frames,
        alpha4_crf=args.alpha4_crf,
        alpha4_fps=args.alpha4_fps,
        class_priority=_parse_class_priority(args.class_priority),
        max_repair_pixels=args.max_repair_pixels,
        max_repair_runs=args.max_repair_runs,
        fail_on_partial_repair=not args.allow_partial_repair,
    )
    if args.alpha4_crf_sweep is not None:
        crfs = _parse_alpha4_crf_sweep(args.alpha4_crf_sweep)
        report = build_candidate_crf_sweep_from_archive(
            archive=args.archive,
            mask_member=args.mask_member,
            output_dir=args.output_dir,
            config=config,
            crfs=crfs,
            command=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
            force=args.force,
        )
        summary = report["sweep_summary"]
        print(
            f"[empirical:{args.output_dir / SWEEP_MANIFEST_NAME}] Alpha mask CRF sweep wrote "
            f"{summary['candidate_count']} candidate manifests; "
            f"byte_plausible_base={summary['byte_plausible_base_candidate_count']}. "
            "No score claim; run planner and CUDA auth eval before any finalist claim.",
            flush=True,
        )
        return 0

    report = build_candidate_from_archive(
        archive=args.archive,
        mask_member=args.mask_member,
        output_dir=args.output_dir,
        config=config,
        command=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
        force=args.force,
    )
    readiness = report["candidate"]["candidate_archive_readiness"]
    print(
        f"[empirical:{args.output_dir / MANIFEST_NAME}] Alpha mask candidate artifacts wrote "
        f"{len(report['candidate']['artifacts'])} payload records; "
        f"full_sequence={readiness['full_sequence_candidate']} "
        f"repair_full={readiness['residual_repair_full_coverage']}. "
        "No score claim; CUDA auth eval required.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
