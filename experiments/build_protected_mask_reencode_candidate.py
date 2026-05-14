#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build deterministic protected/foveated mask reencode candidates.

This is a local, non-scoring candidate builder for the post-C-063 mask branch.
It reads an existing contest archive, decodes the existing mask stream, makes a
lossy AV1 reencode proposal, restores protected pixels in the pre-encode class
tensor according to an explicit policy, and writes a deterministic archive.zip
with the mask member replaced.  It never runs scorers and makes no score claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHA_BUILDER_PATH = REPO_ROOT / "experiments" / "alpha_mask_candidate_builder.py"
SCHEMA = "protected_mask_reencode_candidate_v1"
TOOL = "experiments/build_protected_mask_reencode_candidate.py"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FIXED_ZIP_PERMISSIONS = 0o644
DEFAULT_BASE_ARCHIVE = REPO_ROOT / "experiments/results/c063/archive.zip"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/c063_protected_mask_reencode_20260502"
DEFAULT_MASK_MEMBER = "masks.mkv"
CLASS_IDS = (0, 1, 2, 3, 4)
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)


@dataclass(frozen=True)
class RegionSpec:
    name: str
    x0: int
    y0: int
    x1: int
    y1: int
    frames: tuple[int, ...] | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "frames": None if self.frames is None else list(self.frames),
        }


@dataclass(frozen=True)
class ProtectionPolicy:
    hard_pair_indices: tuple[int, ...] = ()
    hard_frame_indices: tuple[int, ...] = ()
    hard_pair_frame_mode: str = "full_frames"
    class_ids: tuple[int, ...] = ()
    boundary_dilation: int = 0
    horizon_bands: tuple[RegionSpec, ...] = ()
    foveal_boxes: tuple[RegionSpec, ...] = ()
    ego_boxes: tuple[RegionSpec, ...] = ()
    label: str = "protected_foveated_mask_reencode"

    def to_manifest(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "hard_pair_indices": list(self.hard_pair_indices),
            "hard_pair_frame_mode": self.hard_pair_frame_mode,
            "expanded_hard_pair_frames": _expand_pair_indices(
                self.hard_pair_indices,
                mode=self.hard_pair_frame_mode,
            ),
            "hard_frame_indices": list(self.hard_frame_indices),
            "class_ids": list(self.class_ids),
            "boundary_dilation": self.boundary_dilation,
            "horizon_bands": [region.to_manifest() for region in self.horizon_bands],
            "foveal_boxes": [region.to_manifest() for region in self.foveal_boxes],
            "ego_boxes": [region.to_manifest() for region in self.ego_boxes],
        }


def _load_alpha_builder() -> Any:
    spec = importlib.util.spec_from_file_location("alpha_mask_candidate_builder_for_protected_reencode", ALPHA_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Alpha mask builder from {ALPHA_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _safe_member_name(name: str) -> str:
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe archive member path: {name!r}")
    parts = Path(name).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    if any(part.startswith("._") for part in parts) or any(part.startswith(".") for part in parts):
        raise ValueError(f"hidden/system archive member: {name!r}")
    if "__MACOSX" in parts or ".DS_Store" in parts or "Thumbs.db" in parts:
        raise ValueError(f"hidden/system archive member: {name!r}")
    return name


def _read_archive_members(archive: Path) -> tuple[list[tuple[str, bytes]], list[dict[str, Any]]]:
    archive = archive.resolve()
    members: list[tuple[str, bytes]] = []
    inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "r") as zf:
        seen: set[str] = set()
        for info in zf.infolist():
            name = _safe_member_name(info.filename)
            if info.is_dir():
                raise ValueError(f"directory member not allowed in archive: {name!r}")
            if name in seen:
                raise ValueError(f"duplicate archive member: {name!r}")
            seen.add(name)
            data = zf.read(info)
            members.append((name, data))
            inventory.append(
                {
                    "name": name,
                    "size_bytes": int(info.file_size),
                    "compressed_size_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": _sha256_bytes(data),
                }
            )
    return members, inventory


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_member_name(name), date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (FIXED_ZIP_PERMISSIONS & 0xFFFF) << 16
    return info


def _archive_bytes(members: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in members:
            zf.writestr(_zip_info(name), data, compresslevel=9)
    return buffer.getvalue()


def _archive_inventory_from_bytes(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        for info in zf.infolist():
            member = zf.read(info)
            rows.append(
                {
                    "name": info.filename,
                    "size_bytes": int(info.file_size),
                    "compressed_size_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "date_time": list(info.date_time),
                    "compress_type": int(info.compress_type),
                    "permissions_octal": oct((info.external_attr >> 16) & 0o777),
                    "sha256": _sha256_bytes(member),
                }
            )
    return rows


def _validate_masks(masks: torch.Tensor) -> torch.Tensor:
    alpha = _load_alpha_builder()
    return alpha._validate_masks(masks)


def _decode_source_masks(data: bytes, member: str, *, max_frames: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
    alpha = _load_alpha_builder()
    return alpha._decode_legacy_av1_masks_from_member(data, member, max_frames=max_frames)


def _decode_candidate_masks(data: bytes, member: str, *, expected_shape: tuple[int, int, int]) -> torch.Tensor:
    masks, _meta = _decode_source_masks(data, member, max_frames=expected_shape[0])
    if tuple(int(v) for v in masks.shape) != expected_shape:
        raise ValueError(f"candidate decode shape {tuple(masks.shape)} != expected {expected_shape}")
    return masks


def _mask_stats(masks: torch.Tensor) -> dict[str, Any]:
    alpha = _load_alpha_builder()
    return alpha._mask_stats(masks)


def _tensor_u8_sha256(masks: torch.Tensor) -> str:
    alpha = _load_alpha_builder()
    return alpha._tensor_u8_sha256(masks)


def _agreement_metrics(source: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    alpha = _load_alpha_builder()
    return alpha._agreement_metrics(source, candidate)


def _resolve_ffmpeg_binary() -> str:
    alpha = _load_alpha_builder()
    return alpha._resolve_ffmpeg_binary()


def _encode_legacy_av1_masks(
    masks: torch.Tensor,
    output_path: Path,
    *,
    crf: int,
    fps: int,
    svtav1_params: str,
) -> dict[str, Any]:
    masks = _validate_masks(masks)
    if not (0 <= crf <= 63):
        raise ValueError(f"crf must be in [0,63], got {crf}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    t, h, w = [int(v) for v in masks.shape]
    scale_factor = 255 // (len(CLASS_IDS) - 1)
    pixels = (masks.to(torch.int32) * scale_factor).clamp(0, 255).to(torch.uint8).cpu().contiguous()
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
        svtav1_params,
        "-pix_fmt",
        "gray",
        "-an",
        str(output_path),
    ]
    proc = subprocess.run(
        cmd,
        input=pixels.numpy().tobytes(),
        capture_output=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg protected mask encode failed: {stderr}")
    payload = output_path.read_bytes()
    return {
        "path": str(output_path),
        "bytes": payload,
        "command": cmd,
        "frames": t,
        "height": h,
        "width": w,
        "crf": crf,
        "fps": fps,
        "svtav1_params": svtav1_params,
    }


def _expand_pair_indices(
    pair_indices: tuple[int, ...],
    *,
    mode: str = "full_frames",
) -> list[int]:
    if mode == "half_frame_masks":
        return [int(pair_index) for pair_index in pair_indices]
    if mode == "auto":
        # ``auto`` is resolved in ``_expand_pair_indices_for_frame_count`` once
        # the decoded mask frame count is known.
        mode = "full_frames"
    if mode != "full_frames":
        raise ValueError(
            "hard_pair_frame_mode must be one of "
            "full_frames, half_frame_masks, auto"
        )
    frames: list[int] = []
    for pair_index in pair_indices:
        frames.extend([2 * int(pair_index), 2 * int(pair_index) + 1])
    return frames


def _validate_pair_frame_mode(mode: str) -> str:
    if mode not in {"full_frames", "half_frame_masks", "auto"}:
        raise ValueError(
            "hard_pair_frame_mode must be one of "
            "full_frames, half_frame_masks, auto"
        )
    return mode


def _expand_pair_indices_for_frame_count(
    pair_indices: tuple[int, ...],
    *,
    frame_count: int,
    mode: str,
) -> tuple[list[int], str]:
    mode = _validate_pair_frame_mode(mode)
    if mode == "auto":
        full_frames = _expand_pair_indices(pair_indices, mode="full_frames")
        if all(0 <= frame < frame_count for frame in full_frames):
            return full_frames, "full_frames"
        half_frame_masks = _expand_pair_indices(pair_indices, mode="half_frame_masks")
        if all(0 <= frame < frame_count for frame in half_frame_masks):
            return half_frame_masks, "half_frame_masks"
        return full_frames, "full_frames_out_of_range"
    return _expand_pair_indices(pair_indices, mode=mode), mode


def _parse_int_set(value: str | None, *, field: str) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return ()
    parsed: set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", token)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            if end < start:
                raise ValueError(f"{field} range has end before start: {token!r}")
            parsed.update(range(start, end + 1))
            continue
        try:
            parsed.add(int(token))
        except ValueError as exc:
            raise ValueError(f"{field} entries must be integers or ranges, got {token!r}") from exc
    if any(value < 0 for value in parsed):
        raise ValueError(f"{field} entries must be nonnegative")
    return tuple(sorted(parsed))


def _parse_region_spec(value: str, *, name: str) -> RegionSpec:
    region_part, sep, frames_part = value.partition("@")
    coords = [part.strip() for part in region_part.split(",")]
    if len(coords) != 4:
        raise ValueError(f"{name} region must be x0,y0,x1,y1[@frames], got {value!r}")
    x0, y0, x1, y1 = [int(part) for part in coords]
    frames = None
    if sep:
        frames_text = frames_part.removeprefix("frames=")
        frames = _parse_int_set(frames_text, field=f"{name}.frames")
    return RegionSpec(name=name, x0=x0, y0=y0, x1=x1, y1=y1, frames=frames)


def _parse_horizon_band(value: str) -> RegionSpec:
    band_part, sep, frames_part = value.partition("@")
    pieces = [part.strip() for part in band_part.replace(",", ":").split(":")]
    if len(pieces) != 2:
        raise ValueError(f"horizon band must be y0:y1[@frames], got {value!r}")
    frames = None
    if sep:
        frames = _parse_int_set(frames_part.removeprefix("frames="), field="horizon.frames")
    return RegionSpec(name="horizon_band", x0=0, y0=int(pieces[0]), x1=-1, y1=int(pieces[1]), frames=frames)


def _regions_from_json(items: Any, *, default_name: str) -> tuple[RegionSpec, ...]:
    if not items:
        return ()
    regions: list[RegionSpec] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{default_name}[{index}] must be an object")
        frames_raw = item.get("frames")
        frames = None if frames_raw is None else tuple(sorted({int(value) for value in frames_raw}))
        regions.append(
            RegionSpec(
                name=str(item.get("name") or default_name),
                x0=int(item["x0"]),
                y0=int(item["y0"]),
                x1=int(item["x1"]),
                y1=int(item["y1"]),
                frames=frames,
            )
        )
    return tuple(regions)


def _policy_from_json(path: Path) -> ProtectionPolicy:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("policy JSON must contain an object")
    return ProtectionPolicy(
        hard_pair_indices=tuple(sorted({int(value) for value in payload.get("hard_pair_indices", [])})),
        hard_frame_indices=tuple(sorted({int(value) for value in payload.get("hard_frame_indices", [])})),
        hard_pair_frame_mode=_validate_pair_frame_mode(str(payload.get("hard_pair_frame_mode", "full_frames"))),
        class_ids=tuple(sorted({int(value) for value in payload.get("class_ids", [])})),
        boundary_dilation=int(payload.get("boundary_dilation", 0)),
        horizon_bands=_regions_from_json(payload.get("horizon_bands"), default_name="horizon_band"),
        foveal_boxes=_regions_from_json(payload.get("foveal_boxes"), default_name="foveal_box"),
        ego_boxes=_regions_from_json(payload.get("ego_boxes"), default_name="ego_box"),
        label=str(payload.get("label") or "protected_foveated_mask_reencode"),
    )


def _validate_region(region: RegionSpec, *, t: int, h: int, w: int) -> tuple[slice, slice, list[int]]:
    x1 = w if region.x1 < 0 else region.x1
    y1 = h if region.y1 < 0 else region.y1
    if not (0 <= region.x0 < x1 <= w and 0 <= region.y0 < y1 <= h):
        raise ValueError(f"region {region.name!r} is outside mask shape {(t, h, w)}: {region}")
    if region.frames is None:
        frames = list(range(t))
    else:
        frames = [int(frame) for frame in region.frames]
        bad = [frame for frame in frames if frame < 0 or frame >= t]
        if bad:
            raise ValueError(f"region {region.name!r} has frame indices outside [0,{t}): {bad}")
    return slice(region.y0, y1), slice(region.x0, x1), frames


def _boundary_pixels(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    if labels.shape[1] > 1:
        diff_y = labels[:, 1:, :] != labels[:, :-1, :]
        boundary[:, 1:, :] |= diff_y
        boundary[:, :-1, :] |= diff_y
    if labels.shape[2] > 1:
        diff_x = labels[:, :, 1:] != labels[:, :, :-1]
        boundary[:, :, 1:] |= diff_x
        boundary[:, :, :-1] |= diff_x
    return boundary


def _dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    padded = np.pad(mask, ((0, 0), (radius, radius), (radius, radius)), mode="constant", constant_values=False)
    h, w = mask.shape[1:]
    for dy in range(0, 2 * radius + 1):
        for dx in range(0, 2 * radius + 1):
            out |= padded[:, dy : dy + h, dx : dx + w]
    return out


def build_protection_mask(source_masks: torch.Tensor, policy: ProtectionPolicy) -> tuple[torch.Tensor, dict[str, Any]]:
    source_masks = _validate_masks(source_masks)
    labels = source_masks.cpu().numpy()
    t, h, w = [int(value) for value in labels.shape]
    protected = np.zeros((t, h, w), dtype=bool)
    rule_counts: dict[str, int] = {}

    expanded_pair_frames, resolved_pair_frame_mode = _expand_pair_indices_for_frame_count(
        policy.hard_pair_indices,
        frame_count=t,
        mode=policy.hard_pair_frame_mode,
    )
    hard_frames = sorted(set(policy.hard_frame_indices) | set(expanded_pair_frames))
    bad_hard_frames = [frame for frame in hard_frames if frame < 0 or frame >= t]
    if bad_hard_frames:
        raise ValueError(
            f"hard protected frames outside [0,{t}): {bad_hard_frames}; "
            "if these are contest pair indices for a half-frame mask stream, "
            "pass --hard-pair-frame-mode half_frame_masks or use "
            "--hard-frame-indices explicitly"
        )
    if hard_frames:
        rule = np.zeros_like(protected)
        rule[hard_frames, :, :] = True
        protected |= rule
        rule_counts["hard_frames"] = int(rule.sum())

    if policy.class_ids:
        invalid = [class_id for class_id in policy.class_ids if class_id not in CLASS_IDS]
        if invalid:
            raise ValueError(f"class_ids outside {CLASS_IDS}: {invalid}")
        rule = np.isin(labels, np.asarray(policy.class_ids, dtype=labels.dtype))
        protected |= rule
        rule_counts["class_ids"] = int(rule.sum())

    if policy.boundary_dilation > 0:
        boundary = _dilate_binary(_boundary_pixels(labels), int(policy.boundary_dilation))
        protected |= boundary
        rule_counts["boundary_dilation"] = int(boundary.sum())

    for collection_name, regions in (
        ("horizon_bands", policy.horizon_bands),
        ("foveal_boxes", policy.foveal_boxes),
        ("ego_boxes", policy.ego_boxes),
    ):
        total = 0
        for region in regions:
            y_slice, x_slice, frames = _validate_region(region, t=t, h=h, w=w)
            rule = np.zeros_like(protected)
            rule[np.asarray(frames, dtype=np.int64), y_slice, x_slice] = True
            protected |= rule
            total += int(rule.sum())
        if total:
            rule_counts[collection_name] = total

    protected_t = torch.from_numpy(protected)
    per_frame = protected.reshape(t, -1).sum(axis=1).astype(np.int64)
    protected_pixels = int(protected.sum())
    total_pixels = int(protected.size)
    summary = {
        "protected_pixels": protected_pixels,
        "total_pixels": total_pixels,
        "protected_fraction": round(protected_pixels / total_pixels, 12) if total_pixels else 0.0,
        "frames_with_any_protection": int(np.count_nonzero(per_frame)),
        "hard_pair_frame_mode_requested": policy.hard_pair_frame_mode,
        "hard_pair_frame_mode_resolved": resolved_pair_frame_mode,
        "expanded_hard_pair_frames": [int(frame) for frame in expanded_pair_frames],
        "per_rule_pixel_counts_before_overlap": rule_counts,
        "per_frame_protected_pixels_nonzero": [
            {"frame": int(frame), "protected_pixels": int(count)}
            for frame, count in enumerate(per_frame.tolist())
            if count
        ],
    }
    return protected_t, summary


def _masked_agreement(source: torch.Tensor, candidate: torch.Tensor, selector: torch.Tensor) -> dict[str, Any]:
    source = _validate_masks(source)
    candidate = _validate_masks(candidate)
    selector = selector.to(dtype=torch.bool)
    if tuple(source.shape) != tuple(candidate.shape) or tuple(source.shape) != tuple(selector.shape):
        raise ValueError("source, candidate, and selector shapes must match")
    selected = int(selector.sum().item())
    if selected == 0:
        return {
            "num_pixels": 0,
            "different_pixels": 0,
            "argmax_agreement": None,
            "argmax_disagreement": None,
        }
    different = int((source[selector] != candidate[selector]).sum().item())
    return {
        "num_pixels": selected,
        "different_pixels": different,
        "argmax_agreement": round(1.0 - different / selected, 12),
        "argmax_disagreement": round(different / selected, 12),
    }


def _sanitize_ffmpeg_command(command: list[str], *, work_dir: Path) -> list[str]:
    work_prefix = str(work_dir.resolve())
    sanitized: list[str] = []
    for token in command:
        token_text = str(token)
        resolved_text = None
        if token_text.startswith("/"):
            resolved_text = str(Path(token_text).resolve())
        if resolved_text is not None and resolved_text.startswith(work_prefix):
            sanitized.append("<work_dir>/" + os.path.relpath(resolved_text, work_prefix))
        elif token_text.startswith(work_prefix):
            sanitized.append("<work_dir>/" + os.path.relpath(token_text, work_prefix))
        else:
            sanitized.append(token_text)
    return sanitized


def build_candidate(
    *,
    base_archive: Path,
    output_archive: Path,
    manifest_json: Path | None,
    policy: ProtectionPolicy,
    mask_member: str = DEFAULT_MASK_MEMBER,
    crf: int = 56,
    fps: int = 20,
    protection_iterations: int = 1,
    max_frames: int | None = None,
    svtav1_params: str = "enable-restoration=0:enable-cdef=0:lp=1",
) -> dict[str, Any]:
    if protection_iterations < 0:
        raise ValueError(f"protection_iterations must be nonnegative, got {protection_iterations}")
    mask_member = _safe_member_name(mask_member)
    base_archive = base_archive.resolve()
    output_archive = output_archive.resolve()
    members, base_inventory = _read_archive_members(base_archive)
    member_names = [name for name, _data in members]
    if mask_member not in member_names:
        raise FileNotFoundError(f"{base_archive} missing mask member {mask_member!r}")
    source_mask_bytes = dict(members)[mask_member]
    source_masks, decode_meta = _decode_source_masks(source_mask_bytes, mask_member, max_frames=max_frames)
    source_shape = tuple(int(value) for value in source_masks.shape)
    protected_mask, protection_summary = build_protection_mask(source_masks, policy)

    encode_steps: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="protected_mask_reencode_") as tmp_dir:
        work_dir = Path(tmp_dir)
        initial = _encode_legacy_av1_masks(
            source_masks,
            work_dir / "candidate_iter_00.mkv",
            crf=crf,
            fps=fps,
            svtav1_params=svtav1_params,
        )
        current_bytes = bytes(initial["bytes"])
        current_decoded = _decode_candidate_masks(current_bytes, mask_member, expected_shape=source_shape)
        encode_steps.append(
            {
                "stage": "naive_reencode",
                "bytes": len(current_bytes),
                "sha256": _sha256_bytes(current_bytes),
                "ffmpeg_args": _sanitize_ffmpeg_command(list(initial["command"]), work_dir=work_dir),
            }
        )

        for iteration in range(1, protection_iterations + 1):
            if int(protected_mask.sum().item()) == 0:
                break
            composite = current_decoded.clone()
            composite[protected_mask] = source_masks[protected_mask]
            encoded = _encode_legacy_av1_masks(
                composite,
                work_dir / f"candidate_iter_{iteration:02d}.mkv",
                crf=crf,
                fps=fps,
                svtav1_params=svtav1_params,
            )
            current_bytes = bytes(encoded["bytes"])
            current_decoded = _decode_candidate_masks(current_bytes, mask_member, expected_shape=source_shape)
            encode_steps.append(
                {
                    "stage": "protected_preencode_composite",
                    "iteration": iteration,
                    "bytes": len(current_bytes),
                    "sha256": _sha256_bytes(current_bytes),
                    "ffmpeg_args": _sanitize_ffmpeg_command(list(encoded["command"]), work_dir=work_dir),
                }
            )

    output_members = [(name, current_bytes if name == mask_member else data) for name, data in members]
    archive_blob = _archive_bytes(output_members)
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    output_archive.write_bytes(archive_blob)

    output_inventory = _archive_inventory_from_bytes(archive_blob)
    archive_sha = _sha256_bytes(archive_blob)
    base_bytes = base_archive.stat().st_size
    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_build_only_non_score",
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "non_promotable_reason": "candidate archive has not been run through exact CUDA auth eval",
        "archive": {
            "path": str(output_archive),
            "size_bytes": len(archive_blob),
            "sha256": archive_sha,
            "delta_vs_base_archive_bytes": len(archive_blob) - base_bytes,
            "members": output_inventory,
        },
        "base_archive": {
            "path": str(base_archive),
            "size_bytes": base_bytes,
            "sha256": _sha256_file(base_archive),
            "member_inventory": base_inventory,
        },
        "source_mask_stream": {
            "member": mask_member,
            "size_bytes": len(source_mask_bytes),
            "sha256": _sha256_bytes(source_mask_bytes),
            "decoded_class_u8_sha256": _tensor_u8_sha256(source_masks),
            "decoded_stats": _mask_stats(source_masks),
            "decode": decode_meta,
        },
        "candidate_mask_stream": {
            "member": mask_member,
            "size_bytes": len(current_bytes),
            "sha256": _sha256_bytes(current_bytes),
            "decoded_class_u8_sha256": _tensor_u8_sha256(current_decoded),
            "decoded_stats": _mask_stats(current_decoded),
            "overall_agreement_vs_source": _agreement_metrics(source_masks, current_decoded),
            "protected_agreement_vs_source": _masked_agreement(source_masks, current_decoded, protected_mask),
            "unprotected_agreement_vs_source": _masked_agreement(source_masks, current_decoded, ~protected_mask),
        },
        "policy": policy.to_manifest(),
        "protection_summary": protection_summary,
        "ffmpeg": {
            "crf": crf,
            "fps": fps,
            "svtav1_params": svtav1_params,
            "encode_steps": encode_steps,
        },
        "runtime_contract": {
            "archive_members_preserved_except_mask_member": True,
            "replaced_member": mask_member,
            "external_sidecars_allowed": False,
            "scorer_loaded_by_builder": False,
        },
    }
    if manifest_json is None:
        manifest_json = output_archive.with_name("protected_mask_reencode_manifest.json")
    manifest_json = manifest_json.resolve()
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_bytes(_canonical_json_bytes(manifest))
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE_ARCHIVE)
    parser.add_argument("--mask-member", default=DEFAULT_MASK_MEMBER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-archive", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--policy-json", type=Path, default=None)
    parser.add_argument("--policy-label", default="protected_foveated_mask_reencode")
    parser.add_argument("--hard-pair-indices", default=None, help="Comma/range list; pair i expands to frames 2*i,2*i+1.")
    parser.add_argument(
        "--hard-pair-frame-mode",
        choices=("full_frames", "half_frame_masks", "auto"),
        default="full_frames",
        help=(
            "How --hard-pair-indices map to decoded mask frames. "
            "Use half_frame_masks for C067/PR67 600-frame mask streams."
        ),
    )
    parser.add_argument("--hard-frame-indices", default=None, help="Comma/range list of absolute decoded mask frames.")
    parser.add_argument("--protect-classes", default=None, help="Comma/range list of class ids to preserve from source.")
    parser.add_argument("--boundary-dilation", type=int, default=0)
    parser.add_argument("--horizon-band", action="append", default=[], help="y0:y1[@frames=csv/ranges], x spans full width.")
    parser.add_argument("--foveal-box", action="append", default=[], help="x0,y0,x1,y1[@frames=csv/ranges].")
    parser.add_argument("--ego-box", action="append", default=[], help="x0,y0,x1,y1[@frames=csv/ranges].")
    parser.add_argument("--crf", type=int, default=56)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--protection-iterations", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--svtav1-params", default="enable-restoration=0:enable-cdef=0:lp=1")
    return parser


def _policy_from_args(args: argparse.Namespace) -> ProtectionPolicy:
    if args.policy_json is not None:
        return _policy_from_json(args.policy_json)
    return ProtectionPolicy(
        hard_pair_indices=_parse_int_set(args.hard_pair_indices, field="hard_pair_indices"),
        hard_frame_indices=_parse_int_set(args.hard_frame_indices, field="hard_frame_indices"),
        hard_pair_frame_mode=args.hard_pair_frame_mode,
        class_ids=_parse_int_set(args.protect_classes, field="protect_classes"),
        boundary_dilation=int(args.boundary_dilation),
        horizon_bands=tuple(_parse_horizon_band(value) for value in args.horizon_band),
        foveal_boxes=tuple(_parse_region_spec(value, name="foveal_box") for value in args.foveal_box),
        ego_boxes=tuple(_parse_region_spec(value, name="ego_box") for value in args.ego_box),
        label=args.policy_label,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    output_archive = args.output_archive or (args.output_dir / "archive.zip")
    manifest_json = args.manifest_json or (args.output_dir / "protected_mask_reencode_manifest.json")
    manifest = build_candidate(
        base_archive=args.base_archive,
        output_archive=output_archive,
        manifest_json=manifest_json,
        policy=_policy_from_args(args),
        mask_member=args.mask_member,
        crf=args.crf,
        fps=args.fps,
        protection_iterations=args.protection_iterations,
        max_frames=args.max_frames,
        svtav1_params=args.svtav1_params,
    )
    archive = manifest["archive"]
    candidate = manifest["candidate_mask_stream"]
    protected = manifest["protection_summary"]
    print(
        "[protected-mask-reencode] "
        f"wrote {archive['path']} ({archive['size_bytes']:,}B sha256={archive['sha256']}) "
        f"mask={candidate['size_bytes']:,}B mask_sha256={candidate['sha256']} "
        f"protected_pixels={protected['protected_pixels']:,} "
        f"score_claim=false promotion_eligible=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
