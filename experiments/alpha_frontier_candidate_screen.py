#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Local Alpha frontier candidate screen.

This is a deterministic, contest-safe, non-promotable screening helper for
Alpha mask-representation candidates. It reads a canonical archive, validates
ZIP custody, decodes the requested mask member, and compares byte/proxy
properties for already-supported local mask payloads.

It does not load scorer networks and it does not produce score evidence. Any
candidate selected by this screen still requires exact CUDA auth eval through:

    archive.zip -> inflate.sh -> upstream/evaluate.py
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
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

ORIGINAL_VIDEO_BYTES = 37_545_489
SCHEMA = "alpha_frontier_candidate_screen_v1"
EVIDENCE_GRADE = "empirical"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)

DEFAULT_PFP16_A_PLUS_PLUS_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip"
)
DEFAULT_PFP16_A_PLUS_PLUS_BYTES = 686_635
DEFAULT_PFP16_A_PLUS_PLUS_SHA256 = (
    "0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/results/alpha_frontier_candidate_screen/alpha_frontier_candidate_screen.json"
)

_HIDDEN_SYSTEM_NAMES = {"__MACOSX", ".DS_Store", "Thumbs.db"}
_CANDIDATE_ALIASES = {
    "av1": "av1",
    "av1_baseline": "av1",
    "av1_archive_member": "av1",
    "alpha2": "alpha2",
    "alpha2_wavelet": "alpha2",
    "wavelet": "alpha2",
    "alpha3": "alpha3",
    "alpha3_vqvae": "alpha3",
    "vqvae": "alpha3",
    "alpha4": "alpha4",
    "alpha4_grayscale_lut": "alpha4",
    "grayscale": "alpha4",
    "grayscale_lut": "alpha4",
}


@dataclass(frozen=True)
class ScreenConfig:
    candidates: tuple[str, ...]
    include_raw_stats: bool = True
    max_frames: int | None = None
    alpha2_levels: int = 2
    alpha2_step_ll: float = 0.5
    alpha2_step_detail: float = 1.0
    alpha3_patch_size: int = 4
    alpha3_codebook_size: int = 256
    alpha4_crf: int = 50
    alpha4_fps: int = 20
    write_payloads: bool = False


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


def _resolve_ffmpeg_binary() -> str:
    override = os.environ.get("TAC_FFMPEG")
    if override:
        resolved = _resolve_executable(override)
        if resolved is None or not _ffmpeg_usable(resolved):
            raise RuntimeError(f"TAC_FFMPEG={override!r} is not a usable ffmpeg")
        return resolved

    upstream_dir = Path(os.environ.get("TAC_UPSTREAM_DIR", str(REPO_ROOT / "upstream")))
    upstream_ffmpeg = upstream_dir / "ffmpeg-new"
    if (
        upstream_ffmpeg.exists()
        and os.access(upstream_ffmpeg, os.X_OK)
        and _ffmpeg_usable(str(upstream_ffmpeg.resolve()))
    ):
        return str(upstream_ffmpeg.resolve())

    resolved = shutil.which("ffmpeg")
    if resolved is None or not _ffmpeg_usable(resolved):
        raise RuntimeError("ffmpeg not found; set TAC_FFMPEG or install ffmpeg")
    return resolved


def _resolve_executable(value: str) -> str | None:
    path = Path(value)
    if path.exists() and os.access(path, os.X_OK):
        return str(path)
    return shutil.which(value)


def _ffmpeg_usable(executable: str) -> bool:
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
    if min_value < 0 or max_value > 4:
        raise ValueError(f"masks must contain class ids in [0,4], got [{min_value},{max_value}]")
    return masks


def _slice_masks_for_screen(masks: torch.Tensor, max_frames: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
    if max_frames is None:
        return masks, {
            "max_frames": None,
            "original_frames": int(masks.shape[0]),
            "screened_frames": int(masks.shape[0]),
            "truncated": False,
        }
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive when provided, got {max_frames}")
    screened = masks[:max_frames].contiguous()
    return screened, {
        "max_frames": int(max_frames),
        "original_frames": int(masks.shape[0]),
        "screened_frames": int(screened.shape[0]),
        "truncated": bool(screened.shape[0] != masks.shape[0]),
    }


def _mask_stats(masks: torch.Tensor) -> dict[str, Any]:
    masks = _validate_masks(masks)
    counts = torch.bincount(masks.reshape(-1), minlength=5).to(torch.int64)
    total = int(masks.numel())
    probs = [float(c.item()) / total for c in counts]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0.0)
    temporal_changes = 0
    if masks.shape[0] > 1:
        temporal_changes = int((masks[1:] != masks[:-1]).sum().item())
    vertical_boundaries = int((masks[:, 1:, :] != masks[:, :-1, :]).sum().item()) if masks.shape[1] > 1 else 0
    horizontal_boundaries = int((masks[:, :, 1:] != masks[:, :, :-1]).sum().item()) if masks.shape[2] > 1 else 0
    return {
        "shape": [int(v) for v in masks.shape],
        "dtype": str(masks.dtype),
        "num_pixels": total,
        "class_histogram": {str(i): int(counts[i].item()) for i in range(5)},
        "class_fractions": {str(i): round(probs[i], 12) for i in range(5)},
        "class_entropy_bits_per_pixel": round(entropy, 12),
        "zero_order_entropy_floor_bytes": int(math.ceil(entropy * total / 8.0)),
        "temporal_changed_pixels": temporal_changes,
        "spatial_boundary_pixels": vertical_boundaries + horizontal_boundaries,
    }


def _raw_grayscale_stats(masks: torch.Tensor) -> dict[str, Any]:
    from tac.mask_grayscale_lut import encode_masks_grayscale

    raw_class_bytes = masks.to(torch.uint8).contiguous().numpy().tobytes()
    gray = encode_masks_grayscale(masks)
    gray_bytes = gray.cpu().contiguous().numpy().tobytes()
    total = int(masks.numel())
    return {
        "class_id_u8_bytes": len(raw_class_bytes),
        "class_id_u8_sha256": _sha256_bytes(raw_class_bytes),
        "class_id_bitpacked_min_3bpp_bytes": int(math.ceil(total * 3 / 8.0)),
        "one_hot_u8_bytes": total * 5,
        "grayscale_lut_raw_bytes": len(gray_bytes),
        "grayscale_lut_raw_sha256": _sha256_bytes(gray_bytes),
        "raw_stats_are_payload_lower_bounds": True,
    }


def _agreement_metrics(source: torch.Tensor, recovered: torch.Tensor) -> dict[str, Any]:
    source = _validate_masks(source)
    recovered = _validate_masks(recovered)
    shape_match = tuple(source.shape) == tuple(recovered.shape)
    if not shape_match:
        return {
            "shape_match": False,
            "source_shape": [int(v) for v in source.shape],
            "candidate_shape": [int(v) for v in recovered.shape],
            "argmax_agreement": None,
            "argmax_disagreement": None,
            "different_pixels": None,
            "num_pixels": int(source.numel()),
        }
    different = int((source != recovered).sum().item())
    total = int(source.numel())
    agreement = 1.0 - (different / total)
    return {
        "shape_match": True,
        "source_shape": [int(v) for v in source.shape],
        "candidate_shape": [int(v) for v in recovered.shape],
        "num_pixels": total,
        "equal_pixels": total - different,
        "different_pixels": different,
        "argmax_agreement": round(agreement, 12),
        "argmax_disagreement": round(different / total, 12),
    }


def _byte_metrics(payload_bytes: int, baseline_bytes: int) -> dict[str, Any]:
    if baseline_bytes <= 0:
        raise ValueError(f"baseline_bytes must be positive, got {baseline_bytes}")
    delta = int(payload_bytes) - int(baseline_bytes)
    saved = int(baseline_bytes) - int(payload_bytes)
    return {
        "baseline_av1_member_bytes": int(baseline_bytes),
        "payload_bytes": int(payload_bytes),
        "bytes_delta_vs_av1_member": delta,
        "bytes_saved_vs_av1_member": saved,
        "pct_delta_vs_av1_member": round(100.0 * delta / baseline_bytes, 12),
        "pct_saved_vs_av1_member": round(100.0 * saved / baseline_bytes, 12),
        "contest_rate_term_delta_proxy_vs_av1_member": round(25.0 * delta / ORIGINAL_VIDEO_BYTES, 12),
        "rate_term_proxy_only": True,
    }


def _empirical_candidate_base(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "exact_cuda_auth_eval_required": CUDA_AUTH_EVAL_PATH,
    }


def _candidate_ok(
    *,
    name: str,
    payload_format: str,
    payload_bytes: int,
    payload_sha256: str,
    baseline_bytes: int,
    agreement_metrics: dict[str, Any],
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = _empirical_candidate_base(name)
    result.update(
        {
            "status": "ok",
            "payload_format": payload_format,
            "payload_bytes": int(payload_bytes),
            "payload_sha256": payload_sha256,
            "byte_metrics": _byte_metrics(payload_bytes, baseline_bytes),
            "agreement_metrics": agreement_metrics,
            "config": config,
        }
    )
    if extra:
        result.update(extra)
    return result


def _candidate_error(name: str, reason: str) -> dict[str, Any]:
    result = _empirical_candidate_base(name)
    result.update({"status": "error", "error": reason})
    return result


def _screen_av1_baseline(
    masks: torch.Tensor,
    *,
    member_data: bytes,
    source_meta: dict[str, Any],
    baseline_bytes: int,
) -> dict[str, Any]:
    return _candidate_ok(
        name="av1_archive_member",
        payload_format="archive_member_av1_mask_video",
        payload_bytes=baseline_bytes,
        payload_sha256=source_meta["mask_member"]["sha256"],
        baseline_bytes=baseline_bytes,
        agreement_metrics=_agreement_metrics(masks, masks),
        config={"source": "existing archive member", "member": source_meta["mask_member"]["name"]},
        extra={
            "payload_source": "archive_member",
            "archive_member_compressed_size_bytes": int(source_meta["mask_member"]["compressed_size_bytes"]),
            "archive_member_crc32": source_meta["mask_member"]["crc32"],
            "payload_read_sha256": _sha256_bytes(member_data),
        },
    )


def _screen_alpha2_wavelet(
    masks: torch.Tensor,
    *,
    config: ScreenConfig,
    baseline_bytes: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    try:
        from tac.wavelet_mask_codec import WaveletConfig, decode_wavelet_codec, encode_wavelet_codec

        wavelet_config = WaveletConfig(
            levels=config.alpha2_levels,
            step_ll=config.alpha2_step_ll,
            step_detail=config.alpha2_step_detail,
            num_classes=5,
        )
        blob = encode_wavelet_codec(masks, config=wavelet_config)
        recovered = decode_wavelet_codec(blob)
        extra: dict[str, Any] = {}
        if config.write_payloads:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            payload_path = artifact_dir / "alpha2_wavelet.wmc1"
            payload_path.write_bytes(blob)
            extra["payload_path"] = str(payload_path)
        return _candidate_ok(
            name="alpha2_wavelet",
            payload_format="WMC1_wavelet_mask_codec",
            payload_bytes=len(blob),
            payload_sha256=_sha256_bytes(blob),
            baseline_bytes=baseline_bytes,
            agreement_metrics=_agreement_metrics(masks, recovered),
            config={
                "levels": wavelet_config.levels,
                "step_ll": wavelet_config.step_ll,
                "step_detail": wavelet_config.step_detail,
                "num_classes": wavelet_config.num_classes,
            },
            extra=extra,
        )
    except Exception as exc:  # noqa: BLE001 - candidate failures are recorded, not promoted.
        return _candidate_error("alpha2_wavelet", f"{type(exc).__name__}: {exc}")


def _screen_alpha3_vqvae(
    masks: torch.Tensor,
    *,
    config: ScreenConfig,
    baseline_bytes: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    try:
        from tac.vqvae_mask_codec import (
            VQVAEConfig,
            build_codebook_top_k,
            decode_vqvae_codec,
            encode_vqvae_codec,
        )

        vq_config = VQVAEConfig(
            patch_size=config.alpha3_patch_size,
            codebook_size=config.alpha3_codebook_size,
            num_classes=5,
        )
        codebook = build_codebook_top_k(
            masks,
            patch_size=vq_config.patch_size,
            k=vq_config.codebook_size,
        )
        blob = encode_vqvae_codec(masks, codebook=codebook, config=vq_config)
        recovered = decode_vqvae_codec(blob)
        extra: dict[str, Any] = {}
        if config.write_payloads:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            payload_path = artifact_dir / "alpha3_vqvae.vqm1"
            payload_path.write_bytes(blob)
            extra["payload_path"] = str(payload_path)
        return _candidate_ok(
            name="alpha3_vqvae",
            payload_format="VQM1_vqvae_mask_codec",
            payload_bytes=len(blob),
            payload_sha256=_sha256_bytes(blob),
            baseline_bytes=baseline_bytes,
            agreement_metrics=_agreement_metrics(masks, recovered),
            config={
                "patch_size": vq_config.patch_size,
                "codebook_size": vq_config.codebook_size,
                "num_classes": vq_config.num_classes,
            },
            extra=extra,
        )
    except Exception as exc:  # noqa: BLE001 - candidate failures are recorded, not promoted.
        return _candidate_error("alpha3_vqvae", f"{type(exc).__name__}: {exc}")


def _encode_gray_av1(gray: torch.Tensor, output_path: Path, *, crf: int, fps: int) -> tuple[int, bytes]:
    if gray.dtype != torch.uint8 or gray.dim() != 3:
        raise ValueError(f"gray must be uint8 (T,H,W), got {gray.dtype} {tuple(gray.shape)}")
    t, h, w = [int(v) for v in gray.shape]
    raw = gray.cpu().contiguous().numpy().tobytes()
    cmd = [
        _resolve_ffmpeg_binary(),
        "-y",
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
        "-c:v",
        "libsvtav1",
        "-crf",
        str(crf),
        "-preset",
        "6",
        "-svtav1-params",
        "enable-restoration=0:enable-cdef=0",
        "-pix_fmt",
        "gray",
        "-an",
        str(output_path),
    ]
    proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=600, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg alpha4 encode failed: {stderr}")
    encoded = output_path.read_bytes()
    return int(len(encoded)), encoded


def _decode_gray_av1(path: Path, *, expected_shape: tuple[int, int, int]) -> torch.Tensor:
    from tac.mask_grayscale_lut import decode_grayscale_to_classes

    t, h, w = expected_shape
    cmd = [
        _resolve_ffmpeg_binary(),
        "-i",
        str(path),
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
    return decode_grayscale_to_classes(gray)


def _screen_alpha4_grayscale_lut(
    masks: torch.Tensor,
    *,
    config: ScreenConfig,
    baseline_bytes: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    try:
        from tac.mask_grayscale_lut import CLASS_TO_GRAY, encode_masks_grayscale

        gray = encode_masks_grayscale(masks)
        raw_gray_bytes = gray.cpu().contiguous().numpy().tobytes()
        if config.write_payloads:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            payload_path = artifact_dir / "alpha4_grayscale_lut.mkv"
            payload_bytes, payload_data = _encode_gray_av1(
                gray,
                payload_path,
                crf=config.alpha4_crf,
                fps=config.alpha4_fps,
            )
            recovered = _decode_gray_av1(payload_path, expected_shape=tuple(int(v) for v in masks.shape))
            extra: dict[str, Any] = {"payload_path": str(payload_path)}
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                payload_path = Path(tmp_dir) / "alpha4_grayscale_lut.mkv"
                payload_bytes, payload_data = _encode_gray_av1(
                    gray,
                    payload_path,
                    crf=config.alpha4_crf,
                    fps=config.alpha4_fps,
                )
                recovered = _decode_gray_av1(payload_path, expected_shape=tuple(int(v) for v in masks.shape))
            extra = {}
        return _candidate_ok(
            name="alpha4_grayscale_lut",
            payload_format="grayscale_lut_av1_mask_video",
            payload_bytes=payload_bytes,
            payload_sha256=_sha256_bytes(payload_data),
            baseline_bytes=baseline_bytes,
            agreement_metrics=_agreement_metrics(masks, recovered),
            config={
                "crf": int(config.alpha4_crf),
                "fps": int(config.alpha4_fps),
                "class_to_gray": {str(k): int(v) for k, v in sorted(CLASS_TO_GRAY.items())},
            },
            extra={
                **extra,
                "raw_grayscale_bytes": len(raw_gray_bytes),
                "raw_grayscale_sha256": _sha256_bytes(raw_gray_bytes),
            },
        )
    except Exception as exc:  # noqa: BLE001 - candidate failures are recorded, not promoted.
        return _candidate_error("alpha4_grayscale_lut", f"{type(exc).__name__}: {exc}")


def _module_availability() -> dict[str, bool]:
    modules = {
        "tac.mask_codec": "av1",
        "tac.wavelet_mask_codec": "alpha2_wavelet",
        "tac.vqvae_mask_codec": "alpha3_vqvae",
        "tac.mask_grayscale_lut": "alpha4_grayscale_lut",
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
        "tool": "experiments/alpha_frontier_candidate_screen.py",
        "command": list(command) if command is not None else list(sys.argv),
        "cwd": str(Path.cwd()),
        "repo_root": str(REPO_ROOT),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "selected_environment": _selected_environment(),
        "codec_module_available": _module_availability(),
        "ffmpeg_resolution": {
            "TAC_FFMPEG": os.environ.get("TAC_FFMPEG"),
            "upstream_ffmpeg_new": str(upstream_ffmpeg),
            "upstream_ffmpeg_new_exists": upstream_ffmpeg.exists(),
            "path_ffmpeg": shutil.which("ffmpeg"),
        },
    }


def _parse_candidates(value: str) -> tuple[str, ...]:
    parsed: list[str] = []
    for raw in value.split(","):
        token = raw.strip().lower().replace("-", "_")
        if not token:
            continue
        if token not in _CANDIDATE_ALIASES:
            allowed = ", ".join(sorted(_CANDIDATE_ALIASES))
            raise ValueError(f"unknown candidate {raw!r}; allowed aliases: {allowed}")
        canonical = _CANDIDATE_ALIASES[token]
        if canonical not in parsed:
            parsed.append(canonical)
    if not parsed:
        raise ValueError("at least one candidate is required")
    return tuple(parsed)


def _screen_candidates(
    *,
    masks: torch.Tensor,
    member_data: bytes,
    source_meta: dict[str, Any],
    config: ScreenConfig,
    artifact_dir: Path,
) -> list[dict[str, Any]]:
    baseline_bytes = int(source_meta["mask_member"]["size_bytes"])
    results: list[dict[str, Any]] = []
    for candidate in config.candidates:
        if candidate == "av1":
            results.append(
                _screen_av1_baseline(
                    masks,
                    member_data=member_data,
                    source_meta=source_meta,
                    baseline_bytes=baseline_bytes,
                )
            )
        elif candidate == "alpha2":
            results.append(
                _screen_alpha2_wavelet(
                    masks,
                    config=config,
                    baseline_bytes=baseline_bytes,
                    artifact_dir=artifact_dir,
                )
            )
        elif candidate == "alpha3":
            results.append(
                _screen_alpha3_vqvae(
                    masks,
                    config=config,
                    baseline_bytes=baseline_bytes,
                    artifact_dir=artifact_dir,
                )
            )
        elif candidate == "alpha4":
            results.append(
                _screen_alpha4_grayscale_lut(
                    masks,
                    config=config,
                    baseline_bytes=baseline_bytes,
                    artifact_dir=artifact_dir,
                )
            )
        else:
            results.append(_candidate_error(candidate, "unreachable unknown candidate"))
    return results


def _assert_empirical_no_promotion(report: dict[str, Any]) -> None:
    if report.get("score_claim") is not False:
        raise AssertionError("top-level score_claim must be false")
    if report.get("promotion_eligible") is not False:
        raise AssertionError("top-level promotion_eligible must be false")
    if report.get("evidence_grade") != EVIDENCE_GRADE:
        raise AssertionError("top-level evidence_grade must be empirical")
    for candidate in report.get("candidates", []):
        if candidate.get("score_claim") is not False:
            raise AssertionError(f"{candidate.get('name')} score_claim must be false")
        if candidate.get("promotion_eligible") is not False:
            raise AssertionError(f"{candidate.get('name')} promotion_eligible must be false")
        if candidate.get("evidence_grade") != EVIDENCE_GRADE:
            raise AssertionError(f"{candidate.get('name')} evidence_grade must be empirical")


def _build_screen_report_from_masks(
    *,
    masks: torch.Tensor,
    member_data: bytes,
    source_meta: dict[str, Any],
    config: ScreenConfig,
    artifact_dir: Path,
    command: list[str] | None = None,
) -> dict[str, Any]:
    original_masks = _validate_masks(masks)
    screened_masks, frame_subset = _slice_masks_for_screen(original_masks, config.max_frames)
    source = dict(source_meta)
    source["decoded_masks"] = _mask_stats(original_masks)
    source["screened_masks"] = _mask_stats(screened_masks)
    source["frame_subset"] = frame_subset
    source["matches_default_pfp16_anchor"] = (
        source.get("archive_sha256") == DEFAULT_PFP16_A_PLUS_PLUS_SHA256
        and source.get("archive_size_bytes") == DEFAULT_PFP16_A_PLUS_PLUS_BYTES
    )

    raw_stats: dict[str, Any] | None = None
    if config.include_raw_stats:
        try:
            raw_stats = _raw_grayscale_stats(screened_masks)
        except Exception as exc:  # noqa: BLE001 - raw stats are diagnostic only.
            raw_stats = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    report = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "local_screen_only": True,
        "scorer_network_loaded": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "purpose": "choose Alpha mask representations for later exact CUDA auth eval; not score evidence",
        "default_anchor": {
            "name": "PFP16_A++_current_anchor",
            "archive_path": str(DEFAULT_PFP16_A_PLUS_PLUS_ARCHIVE),
            "archive_size_bytes": DEFAULT_PFP16_A_PLUS_PLUS_BYTES,
            "archive_sha256": DEFAULT_PFP16_A_PLUS_PLUS_SHA256,
        },
        "screen_config": dataclasses.asdict(config),
        "source": source,
        "raw_stats": raw_stats,
        "candidates": _screen_candidates(
            masks=screened_masks,
            member_data=member_data,
            source_meta=source_meta,
            config=config,
            artifact_dir=artifact_dir,
        ),
        "provenance": _provenance(command),
    }
    _assert_empirical_no_promotion(report)
    return report


def screen_archive(
    *,
    archive: Path,
    mask_member: str,
    output: Path,
    artifact_dir: Path,
    config: ScreenConfig,
    command: list[str] | None = None,
    require_pfp16_anchor: bool = False,
) -> dict[str, Any]:
    member_data, source_meta = _read_archive_member(archive, mask_member)
    if require_pfp16_anchor:
        if source_meta["archive_sha256"] != DEFAULT_PFP16_A_PLUS_PLUS_SHA256:
            raise ValueError(
                "archive SHA does not match the configured PFP16 A++ anchor: "
                f"{source_meta['archive_sha256']}"
            )
        if source_meta["archive_size_bytes"] != DEFAULT_PFP16_A_PLUS_PLUS_BYTES:
            raise ValueError(
                "archive byte count does not match the configured PFP16 A++ anchor: "
                f"{source_meta['archive_size_bytes']}"
            )

    masks = _decode_av1_masks_from_member(member_data, mask_member)
    report = _build_screen_report_from_masks(
        masks=masks,
        member_data=member_data,
        source_meta=source_meta,
        config=config,
        artifact_dir=artifact_dir,
        command=command,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PFP16_A_PLUS_PLUS_ARCHIVE)
    parser.add_argument("--mask-member", default="masks.mkv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional payload artifact directory. Defaults to <output parent>/artifacts.",
    )
    parser.add_argument(
        "--candidates",
        default="av1,alpha2,alpha3,alpha4",
        help="Comma-separated candidates: av1, alpha2, alpha3, alpha4.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-raw-stats", action="store_true")
    parser.add_argument("--write-payloads", action="store_true")
    parser.add_argument("--require-pfp16-anchor", action="store_true")
    parser.add_argument("--alpha2-levels", type=int, default=2)
    parser.add_argument("--alpha2-step-ll", type=float, default=0.5)
    parser.add_argument("--alpha2-step-detail", type=float, default=1.0)
    parser.add_argument("--alpha3-patch-size", type=int, default=4)
    parser.add_argument("--alpha3-codebook-size", type=int, default=256)
    parser.add_argument("--alpha4-crf", type=int, default=50)
    parser.add_argument("--alpha4-fps", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = ScreenConfig(
        candidates=_parse_candidates(args.candidates),
        include_raw_stats=not args.no_raw_stats,
        max_frames=args.max_frames,
        alpha2_levels=args.alpha2_levels,
        alpha2_step_ll=args.alpha2_step_ll,
        alpha2_step_detail=args.alpha2_step_detail,
        alpha3_patch_size=args.alpha3_patch_size,
        alpha3_codebook_size=args.alpha3_codebook_size,
        alpha4_crf=args.alpha4_crf,
        alpha4_fps=args.alpha4_fps,
        write_payloads=args.write_payloads,
    )
    artifact_dir = args.artifact_dir if args.artifact_dir is not None else args.output.parent / "artifacts"
    report = screen_archive(
        archive=args.archive,
        mask_member=args.mask_member,
        output=args.output,
        artifact_dir=artifact_dir,
        config=config,
        command=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
        require_pfp16_anchor=args.require_pfp16_anchor,
    )
    ok_count = sum(1 for candidate in report["candidates"] if candidate["status"] == "ok")
    print(
        f"[empirical:{args.output}] Alpha frontier screen wrote {len(report['candidates'])} "
        f"candidate records ({ok_count} ok). No score claim; CUDA auth eval required.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
