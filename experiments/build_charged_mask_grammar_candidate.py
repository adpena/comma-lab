#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a strict, charged CMG1 mask-grammar prototype artifact.

By default this emits the original standalone build-only scaffold.  With
``--base-archive`` it can also build a full archive candidate by replacing a
mask stream with a charged CMG1 member.  The output remains non-promotable
until the hardened auth-eval and local-smoke allowlists admit ``.cmg1`` and
CUDA auth eval runs on the exact archive bytes.  The standalone output ZIP
charges the CMG1 wire constants and manifest as archive members:

    mask.cmg1
    cmg1_manifest.json

If an input mask stream is supplied, the CMG1 payload stores that stream
byte-for-byte after a deterministic charged header so a future decoder can
prove bit-identical mask-stream recovery.  Without an input stream it emits a
strict placeholder payload and manifest only.  In both modes
``score_claim=false`` and ``promotion_eligible=false``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import sys
import struct
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zlib

import numpy as np


SCHEMA = "cmg1_strict_mask_grammar_candidate_v1"
TOOL = "experiments/build_charged_mask_grammar_candidate.py"
CMG1_MAGIC = b"CMG1"
CMG1_SCHEMA_VERSION = 1
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
DEFAULT_PAYLOAD_MEMBER = "mask.cmg1"
DEFAULT_RUNTIME_PAYLOAD_MEMBER = "masks.cmg1"
DEFAULT_MANIFEST_MEMBER = "cmg1_manifest.json"
DEFAULT_REPLACED_MASK_MEMBER = "masks.mkv"
DEFAULT_SOURCE_MASK_MEMBER = "masks.mkv"
DEFAULT_LOSSY_MASK_MEMBER = "grayscale.mkv"
DEFAULT_REPAIR_MEMBER_RAW = "alpha4_residual_repair.amr1"
DEFAULT_FRAMES = 600
DEFAULT_HEIGHT = 384
DEFAULT_WIDTH = 512
DEFAULT_CLASS_COUNT = 5
DEFAULT_CLASS_PRIORITY = (2, 1, 3, 4, 0)
REPAIR_COMPRESSORS = ("raw", "zlib", "lzma_xz", "brotli")
RUNTIME_REQUIRED_MEMBERS = ("renderer.bin", "optimized_poses.bin")
AMR1_REPAIR_SCHEMA = "charged_mask_grammar_amr1_runtime_candidate_v1"
MODE_PLACEHOLDER = "placeholder_strict_manifest"
MODE_RAW_BIT_IDENTICAL = "raw_bit_identical_mask_stream"
MODE_AMR1_RUNTIME_REPAIR = "runtime_grayscale_amr1_residual_repair"
MODE_AMR1_RUNTIME_LEGACY_REPAIR = "runtime_legacy_amr1_residual_repair"
MODE_CODES = {
    MODE_PLACEHOLDER: 0,
    MODE_RAW_BIT_IDENTICAL: 1,
}
ORIGINAL_VIDEO_BYTES = 37_545_489
HEADER_STRUCT = struct.Struct("<4sHHHHBBI")
FORBIDDEN_ARCHIVE_NAME_PARTS = (".DS_Store", "__MACOSX", "._", "Thumbs.db")
REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHA_BUILDER_PATH = REPO_ROOT / "experiments" / "alpha_mask_candidate_builder.py"
GRAYSCALE_RUNTIME_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer_grayscale.py"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)


@dataclass(frozen=True)
class CMG1Shape:
    frames: int = DEFAULT_FRAMES
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    class_count: int = DEFAULT_CLASS_COUNT

    def as_manifest(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "height": self.height,
            "width": self.width,
            "class_count": self.class_count,
        }


@dataclass(frozen=True)
class RegionSpec:
    name: str
    x0: int
    y0: int
    x1: int
    y1: int
    frames: tuple[int, ...] | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x0": int(self.x0),
            "y0": int(self.y0),
            "x1": int(self.x1),
            "y1": int(self.y1),
            "frames": None if self.frames is None else [int(v) for v in self.frames],
        }


@dataclass(frozen=True)
class ChargedRepairPolicy:
    label: str = "pose_sensitive_boundary_horizon_foveal_ego_v1"
    hard_pair_indices: tuple[int, ...] = ()
    hard_frame_indices: tuple[int, ...] = ()
    class_ids: tuple[int, ...] = ()
    boundary_dilation: int = 1
    horizon_bands: tuple[RegionSpec, ...] = ()
    foveal_boxes: tuple[RegionSpec, ...] = ()
    ego_boxes: tuple[RegionSpec, ...] = ()
    select_all_differences: bool = False

    def as_manifest(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "hard_pair_indices": [int(v) for v in self.hard_pair_indices],
            "expanded_hard_pair_frames": _expand_pair_indices(self.hard_pair_indices),
            "hard_frame_indices": [int(v) for v in self.hard_frame_indices],
            "class_ids": [int(v) for v in self.class_ids],
            "boundary_dilation": int(self.boundary_dilation),
            "horizon_bands": [region.as_manifest() for region in self.horizon_bands],
            "foveal_boxes": [region.as_manifest() for region in self.foveal_boxes],
            "ego_boxes": [region.as_manifest() for region in self.ego_boxes],
            "select_all_differences": bool(self.select_all_differences),
            "repair_only_changed_pixels": True,
        }


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _pretty_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _safe_member_name(name: str) -> str:
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe archive member path: {name!r}")
    parts = Path(name).parts
    if len(parts) != 1 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    if name.startswith(".") or name == "__MACOSX":
        raise ValueError(f"hidden/system archive member path: {name!r}")
    return name


def _safe_existing_archive_member(name: str) -> str:
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe base archive member path: {name!r}")
    parts = Path(name).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe base archive member path: {name!r}")
    if any(part.startswith(".") for part in parts):
        raise ValueError(f"hidden base archive member path: {name!r}")
    if any(forbidden in name for forbidden in FORBIDDEN_ARCHIVE_NAME_PARTS):
        raise ValueError(f"forbidden housekeeping member in base archive: {name!r}")
    return name


def _validate_base_archive_members(infos: list[zipfile.ZipInfo]) -> None:
    seen: set[str] = set()
    for info in infos:
        name = _safe_existing_archive_member(info.filename)
        if name.endswith("/"):
            raise ValueError(f"directory member not allowed in base archive: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate member in base archive: {name!r}")
        seen.add(name)


def _validate_runtime_archive_members(infos: list[zipfile.ZipInfo], *, source_mask_member: str) -> None:
    allowed = set(RUNTIME_REQUIRED_MEMBERS) | {source_mask_member}
    seen: set[str] = set()
    for info in infos:
        name = _safe_existing_archive_member(info.filename)
        if name.endswith("/") or info.is_dir():
            raise ValueError(f"directory member not allowed in runtime archive: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate member in runtime archive: {name!r}")
        seen.add(name)
        if name not in allowed:
            raise ValueError(
                f"unexpected runtime archive member {name!r}; allowed members are {sorted(allowed)!r}"
            )
    missing = sorted(allowed.difference(seen))
    if missing:
        raise ValueError(f"runtime archive missing required member(s): {missing}")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _base_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_existing_archive_member(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _deflated_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _load_module_from_path(*, module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_alpha_builder() -> Any:
    return _load_module_from_path(
        module_name="alpha_mask_candidate_builder_for_charged_mask_grammar",
        path=ALPHA_BUILDER_PATH,
    )


def _load_grayscale_runtime() -> Any:
    return _load_module_from_path(
        module_name="inflate_renderer_grayscale_for_charged_mask_grammar",
        path=GRAYSCALE_RUNTIME_PATH,
    )


def _read_runtime_archive_members(
    source_archive: Path,
    *,
    source_mask_member: str,
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    source_archive = source_archive.resolve()
    if not source_archive.is_file():
        raise FileNotFoundError(f"source runtime archive not found: {source_archive}")
    members: dict[str, bytes] = {}
    inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(source_archive, "r") as zf:
        infos = zf.infolist()
        _validate_runtime_archive_members(infos, source_mask_member=source_mask_member)
        for info in infos:
            name = _safe_existing_archive_member(info.filename)
            data = zf.read(info)
            members[name] = data
            inventory.append(
                {
                    "name": name,
                    "size_bytes": int(info.file_size),
                    "compressed_size_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": _sha256_bytes(data),
                }
            )
    inventory.sort(key=lambda item: item["name"])
    return members, inventory


def _archive_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            name = _safe_existing_archive_member(info.filename)
            member = zf.read(info)
            rows.append(
                {
                    "name": name,
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


def _write_deterministic_deflated_archive(output: Path, members: list[tuple[str, bytes]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        seen: set[str] = set()
        for name, data in members:
            name = _safe_member_name(name)
            if name in seen:
                raise ValueError(f"duplicate output archive member: {name!r}")
            seen.add(name)
            zf.writestr(_deflated_zip_info(name), data, compresslevel=9)


def _decode_source_masks_from_runtime_member(
    data: bytes,
    member: str,
    *,
    max_frames: int | None,
) -> tuple[Any, dict[str, Any]]:
    alpha = _load_alpha_builder()
    return alpha._decode_legacy_av1_masks_from_member(data, member, max_frames=max_frames)


def _decode_lossy_grayscale_masks(
    lossy_mask_stream: Path,
    *,
    expected_shape: tuple[int, int, int],
) -> tuple[Any, dict[str, Any]]:
    runtime = _load_grayscale_runtime()
    lossy_mask_stream = lossy_mask_stream.resolve()
    masks = runtime._decode_grayscale_mkv_to_classes(
        lossy_mask_stream,
        target_h=int(expected_shape[1]),
        target_w=int(expected_shape[2]),
    )
    shape = tuple(int(v) for v in masks.shape)
    if shape != expected_shape:
        raise ValueError(f"lossy decoded mask shape {shape} != expected source shape {expected_shape}")
    return masks, {
        "decoder": "submissions/robust_current/inflate_renderer_grayscale.py::_decode_grayscale_mkv_to_classes",
        "path": str(lossy_mask_stream),
        "shape": [int(v) for v in shape],
    }


def _wire_contract(shape: CMG1Shape, *, payload_member: str, manifest_member: str) -> dict[str, Any]:
    return {
        "payload_member": payload_member,
        "manifest_member": manifest_member,
        "payload_magic_ascii": CMG1_MAGIC.decode("ascii"),
        "payload_magic_hex": CMG1_MAGIC.hex(),
        "schema_version": CMG1_SCHEMA_VERSION,
        "byte_order": "little",
        "shape": shape.as_manifest(),
        "header_struct": "<4sHHHHBBI",
        "header_fields": [
            "magic",
            "schema_version",
            "frames",
            "height",
            "width",
            "class_count",
            "mode_code",
            "header_manifest_json_bytes",
        ],
        "body": "raw input mask stream bytes when mode=raw_bit_identical_mask_stream; empty otherwise",
    }


def _build_inner_header_manifest(
    *,
    mode: str,
    shape: CMG1Shape,
    source_bytes: bytes | None,
    payload_member: str,
    manifest_member: str,
) -> dict[str, Any]:
    source_record: dict[str, Any] | None = None
    if source_bytes is not None:
        source_record = {
            "bytes": len(source_bytes),
            "sha256": _sha256_bytes(source_bytes),
            "recovery_contract": "decoder must reproduce these exact bytes before any mask tensor claim",
        }
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "mode": mode,
        "score_claim": False,
        "promotion_eligible": False,
        "wire_contract": _wire_contract(
            shape,
            payload_member=payload_member,
            manifest_member=manifest_member,
        ),
        "source_mask_stream": source_record,
        "charged_constants_statement": (
            "CMG1 magic, version, shape, class count, mode, source stream "
            "checksum, and recovery contract are serialized inside this payload header."
        ),
    }


def encode_cmg1_payload(
    *,
    source_bytes: bytes | None,
    shape: CMG1Shape = CMG1Shape(),
    payload_member: str = DEFAULT_PAYLOAD_MEMBER,
    manifest_member: str = DEFAULT_MANIFEST_MEMBER,
) -> bytes:
    """Encode a deterministic CMG1 prototype payload."""
    payload_member = _safe_member_name(payload_member)
    manifest_member = _safe_member_name(manifest_member)
    mode = MODE_RAW_BIT_IDENTICAL if source_bytes is not None else MODE_PLACEHOLDER
    if not (0 <= shape.frames <= 65535 and 0 <= shape.height <= 65535 and 0 <= shape.width <= 65535):
        raise ValueError(f"CMG1 shape dimensions must fit uint16 header fields: {shape}")
    if not (0 <= shape.class_count <= 255):
        raise ValueError(f"CMG1 class_count must fit uint8 header field: {shape.class_count}")
    inner_manifest = _build_inner_header_manifest(
        mode=mode,
        shape=shape,
        source_bytes=source_bytes,
        payload_member=payload_member,
        manifest_member=manifest_member,
    )
    header_json = _canonical_json_bytes(inner_manifest)
    body = source_bytes or b""
    header = HEADER_STRUCT.pack(
        CMG1_MAGIC,
        CMG1_SCHEMA_VERSION,
        shape.frames,
        shape.height,
        shape.width,
        shape.class_count,
        MODE_CODES[mode],
        len(header_json),
    )
    return header + header_json + body


def decode_cmg1_payload(payload: bytes) -> dict[str, Any]:
    """Decode enough of the CMG1 scaffold for tests and future validators."""
    if len(payload) < HEADER_STRUCT.size:
        raise ValueError("CMG1 payload is shorter than the fixed header")
    magic, version, frames, height, width, class_count, mode_code, header_len = HEADER_STRUCT.unpack(
        payload[: HEADER_STRUCT.size]
    )
    if magic != CMG1_MAGIC:
        raise ValueError(f"unexpected CMG1 magic: {magic!r}")
    if version != CMG1_SCHEMA_VERSION:
        raise ValueError(f"unexpected CMG1 schema version: {version}")
    if mode_code not in set(MODE_CODES.values()):
        raise ValueError(f"unknown CMG1 mode code: {mode_code}")
    header_start = HEADER_STRUCT.size
    header_end = header_start + header_len
    if header_end > len(payload):
        raise ValueError("CMG1 header manifest length exceeds payload length")
    header_manifest = json.loads(payload[header_start:header_end].decode("utf-8"))
    raw_stream = payload[header_end:]
    return {
        "fixed_header": {
            "magic": magic.decode("ascii"),
            "schema_version": version,
            "frames": frames,
            "height": height,
            "width": width,
            "class_count": class_count,
            "mode_code": mode_code,
            "header_manifest_json_bytes": header_len,
        },
        "header_manifest": header_manifest,
        "raw_stream": raw_stream,
    }


def _expand_pair_indices(pair_indices: tuple[int, ...]) -> list[int]:
    frames: list[int] = []
    for pair_index in pair_indices:
        frames.extend([2 * int(pair_index), 2 * int(pair_index) + 1])
    return frames


def _expand_pair_indices_for_frame_count(pair_indices: tuple[int, ...], *, frame_count: int) -> tuple[list[int], str]:
    if not pair_indices:
        return [], "none"
    pair_frames = _expand_pair_indices(pair_indices)
    if max(pair_frames) < frame_count:
        return pair_frames, "pair_index_to_frames_2i_2i_plus_1"
    direct_frames = [int(pair_index) for pair_index in pair_indices]
    if max(direct_frames) < frame_count:
        return direct_frames, "pair_index_to_single_mask_frame_i"
    raise ValueError(
        f"hard_pair_indices cannot map into decoded mask frame_count={frame_count}: "
        f"max_pair={max(pair_indices)}"
    )


def _parse_int_set(value: str | None, *, field: str) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return ()
    parsed: set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"{field} range has end before start: {token!r}")
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(token))
    if any(item < 0 for item in parsed):
        raise ValueError(f"{field} entries must be nonnegative")
    return tuple(sorted(parsed))


def _parse_int_sequence(value: str | None, *, field: str) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return ()
    parsed: list[int] = []
    seen: set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        item = int(token)
        if item < 0:
            raise ValueError(f"{field} entries must be nonnegative")
        if item in seen:
            raise ValueError(f"{field} entries must not repeat values: {item}")
        seen.add(item)
        parsed.append(item)
    return tuple(parsed)


def _parse_region_spec(value: str, *, name: str) -> RegionSpec:
    region_part, sep, frames_part = value.partition("@")
    coords = [part.strip() for part in region_part.split(",")]
    if len(coords) != 4:
        raise ValueError(f"{name} must be x0,y0,x1,y1[@frames=csv/ranges], got {value!r}")
    frames = None
    if sep:
        frames = _parse_int_set(frames_part.removeprefix("frames="), field=f"{name}.frames")
    return RegionSpec(
        name=name,
        x0=int(coords[0]),
        y0=int(coords[1]),
        x1=int(coords[2]),
        y1=int(coords[3]),
        frames=frames,
    )


def _parse_horizon_band(value: str) -> RegionSpec:
    band_part, sep, frames_part = value.partition("@")
    pieces = [part.strip() for part in band_part.replace(",", ":").split(":")]
    if len(pieces) != 2:
        raise ValueError(f"horizon band must be y0:y1[@frames=csv/ranges], got {value!r}")
    frames = None
    if sep:
        frames = _parse_int_set(frames_part.removeprefix("frames="), field="horizon.frames")
    return RegionSpec(name="horizon_band", x0=0, y0=int(pieces[0]), x1=-1, y1=int(pieces[1]), frames=frames)


def _hard_pair_indices_from_component_trace(
    path: Path | None,
    *,
    top_k: int,
    sort_key: str,
) -> tuple[int, ...]:
    if path is None or top_k <= 0:
        return ()
    payload = json.loads(path.read_text())
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        samples = payload.get("top_combined_samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"component trace has no samples/top_combined_samples: {path}")
    scored: list[tuple[float, int]] = []
    for item in samples:
        if not isinstance(item, dict):
            continue
        raw_pair = item.get("pair_index", item.get("video_pair_index"))
        if raw_pair is None:
            continue
        pair_index = int(raw_pair)
        raw_score = item.get(sort_key)
        if raw_score is None:
            raw_score = item.get("score_combined_contribution_first_order", item.get("posenet_dist", 0.0))
        scored.append((float(raw_score), pair_index))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    selected: list[int] = []
    seen: set[int] = set()
    for _score, pair_index in ranked:
        if pair_index in seen:
            continue
        seen.add(pair_index)
        selected.append(pair_index)
        if len(selected) >= top_k:
            break
    return tuple(selected)


def _validate_policy_indices(policy: ChargedRepairPolicy, *, frame_count: int) -> None:
    if any(pair < 0 for pair in policy.hard_pair_indices):
        raise ValueError(f"hard_pair_indices must be nonnegative: {policy.hard_pair_indices}")
    _expand_pair_indices_for_frame_count(policy.hard_pair_indices, frame_count=frame_count)
    bad_frames = [frame for frame in policy.hard_frame_indices if frame < 0 or frame >= frame_count]
    if bad_frames:
        raise ValueError(f"hard_frame_indices outside [0,{frame_count}): {bad_frames}")
    bad_classes = [class_id for class_id in policy.class_ids if class_id < 0 or class_id >= DEFAULT_CLASS_COUNT]
    if bad_classes:
        raise ValueError(f"class_ids outside [0,{DEFAULT_CLASS_COUNT}): {bad_classes}")
    if policy.boundary_dilation < 0:
        raise ValueError(f"boundary_dilation must be nonnegative, got {policy.boundary_dilation}")


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


def _region_to_slices(region: RegionSpec, *, t: int, h: int, w: int) -> tuple[slice, slice, list[int]]:
    x1 = w if region.x1 < 0 else int(region.x1)
    y1 = h if region.y1 < 0 else int(region.y1)
    if not (0 <= region.x0 < x1 <= w and 0 <= region.y0 < y1 <= h):
        raise ValueError(f"region {region.name!r} is outside mask shape {(t, h, w)}: {region}")
    if region.frames is None:
        frames = list(range(t))
    else:
        frames = [int(frame) for frame in region.frames]
        bad = [frame for frame in frames if frame < 0 or frame >= t]
        if bad:
            raise ValueError(f"region {region.name!r} has frame indices outside [0,{t}): {bad}")
    return slice(int(region.y0), y1), slice(int(region.x0), x1), frames


def build_repair_selector(
    source_masks: Any,
    candidate_masks: Any,
    policy: ChargedRepairPolicy,
) -> tuple[np.ndarray, dict[str, Any]]:
    alpha = _load_alpha_builder()
    source_masks = alpha._validate_masks(source_masks)
    candidate_masks = alpha._validate_masks(candidate_masks)
    if tuple(int(v) for v in source_masks.shape) != tuple(int(v) for v in candidate_masks.shape):
        raise ValueError(
            "source/candidate mask shape mismatch: "
            f"{tuple(source_masks.shape)} vs {tuple(candidate_masks.shape)}"
        )
    source_np = source_masks.cpu().numpy()
    candidate_np = candidate_masks.cpu().numpy()
    t, h, w = [int(v) for v in source_np.shape]
    _validate_policy_indices(policy, frame_count=t)
    expanded_pair_frames, pair_frame_mapping = _expand_pair_indices_for_frame_count(
        policy.hard_pair_indices,
        frame_count=t,
    )

    diff = source_np != candidate_np
    selector = np.zeros((t, h, w), dtype=bool)
    rule_counts: dict[str, int] = {}

    if policy.select_all_differences:
        selector[:, :, :] = True
        rule_counts["all_differences"] = int(selector.sum())

    hard_frames = sorted(set(policy.hard_frame_indices) | set(expanded_pair_frames))
    if hard_frames:
        rule = np.zeros_like(selector)
        rule[np.asarray(hard_frames, dtype=np.int64), :, :] = True
        selector |= rule
        rule_counts["hard_pair_or_frame_full_frames"] = int(rule.sum())

    if policy.class_ids:
        rule = np.isin(source_np, np.asarray(policy.class_ids, dtype=source_np.dtype))
        selector |= rule
        rule_counts["source_classes"] = int(rule.sum())

    if policy.boundary_dilation > 0:
        boundary = _dilate_binary(_boundary_pixels(source_np), int(policy.boundary_dilation))
        selector |= boundary
        rule_counts["source_boundary_dilation"] = int(boundary.sum())

    for collection_name, regions in (
        ("horizon_bands", policy.horizon_bands),
        ("foveal_boxes", policy.foveal_boxes),
        ("ego_boxes", policy.ego_boxes),
    ):
        total = 0
        for region in regions:
            y_slice, x_slice, frames = _region_to_slices(region, t=t, h=h, w=w)
            rule = np.zeros_like(selector)
            rule[np.asarray(frames, dtype=np.int64), y_slice, x_slice] = True
            selector |= rule
            total += int(rule.sum())
        if total:
            rule_counts[collection_name] = total

    selected = selector & diff
    per_frame = selected.reshape(t, -1).sum(axis=1).astype(np.int64)
    total_residual = int(diff.sum())
    selected_residual = int(selected.sum())
    summary = {
        "policy": policy.as_manifest(),
        "hard_pair_frame_mapping": pair_frame_mapping,
        "actual_expanded_hard_pair_frames": [int(frame) for frame in expanded_pair_frames],
        "total_pixels": int(diff.size),
        "total_residual_pixels": total_residual,
        "policy_selector_pixels_before_diff_intersection": int(selector.sum()),
        "selected_repair_pixels_before_budget": selected_residual,
        "selected_residual_fraction": round(selected_residual / total_residual, 12) if total_residual else 1.0,
        "per_rule_pixel_counts_before_overlap": rule_counts,
        "frames_with_selected_repairs": int(np.count_nonzero(per_frame)),
        "per_frame_selected_repair_pixels_nonzero": [
            {"frame": int(frame), "selected_repair_pixels": int(count)}
            for frame, count in enumerate(per_frame.tolist())
            if count
        ],
    }
    return selected, summary


def _selected_repair_runs(
    source_masks: Any,
    selected: np.ndarray,
    *,
    class_priority: tuple[int, ...],
    max_repair_pixels: int | None,
    max_repair_runs: int | None,
    allow_partial_repair: bool,
) -> tuple[list[Any], dict[str, Any]]:
    alpha = _load_alpha_builder()
    source_masks = alpha._validate_masks(source_masks)
    source_np = source_masks.cpu().numpy()
    if tuple(int(v) for v in source_np.shape) != tuple(int(v) for v in selected.shape):
        raise ValueError(f"selected repair mask shape {selected.shape} != source shape {tuple(source_np.shape)}")
    if any(class_id < 0 or class_id >= DEFAULT_CLASS_COUNT for class_id in class_priority):
        raise ValueError(f"class_priority entries must be class ids in [0,{DEFAULT_CLASS_COUNT})")
    if max_repair_pixels is not None and max_repair_pixels < 0:
        raise ValueError(f"max_repair_pixels must be nonnegative, got {max_repair_pixels}")
    if max_repair_runs is not None and max_repair_runs < 0:
        raise ValueError(f"max_repair_runs must be nonnegative, got {max_repair_runs}")

    runs: list[Any] = []
    selected_pixels = 0
    total_selected_pixels = int(np.count_nonzero(selected))
    selected_by_class = {str(class_id): 0 for class_id in range(DEFAULT_CLASS_COUNT)}
    total_by_class = {
        str(class_id): int(np.count_nonzero(selected & (source_np == class_id)))
        for class_id in range(DEFAULT_CLASS_COUNT)
    }
    partial_reason: str | None = None

    for class_id in class_priority:
        for frame_index in range(source_np.shape[0]):
            frame_source = source_np[frame_index]
            frame_selected = selected[frame_index]
            for y in range(source_np.shape[1]):
                xs = np.flatnonzero(frame_selected[y] & (frame_source[y] == class_id))
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
                    if max_repair_pixels is not None and projected_pixels > max_repair_pixels:
                        partial_reason = (
                            "max_repair_pixels would be exceeded: "
                            f"{projected_pixels} > {max_repair_pixels}"
                        )
                    if max_repair_runs is not None and projected_runs > max_repair_runs:
                        partial_reason = (
                            "max_repair_runs would be exceeded: "
                            f"{projected_runs} > {max_repair_runs}"
                        )
                    if partial_reason is not None:
                        if not allow_partial_repair:
                            raise ValueError(f"repair payload would be partial ({partial_reason})")
                        return runs, {
                            "selected_repair_pixels": int(selected_pixels),
                            "selected_repair_runs": int(len(runs)),
                            "selected_repair_pixels_by_source_class": selected_by_class,
                            "policy_selected_repair_pixels_by_source_class": total_by_class,
                            "policy_selected_repair_pixels_before_budget": total_selected_pixels,
                            "partial_reason": partial_reason,
                            "budget_limited": True,
                        }
                    runs.append(
                        alpha.RepairRun(
                            frame_index=int(frame_index),
                            y=int(y),
                            x0=x0,
                            length=length,
                            class_id=int(class_id),
                        )
                    )
                    selected_pixels += length
                    selected_by_class[str(class_id)] += length

    return runs, {
        "selected_repair_pixels": int(selected_pixels),
        "selected_repair_runs": int(len(runs)),
        "selected_repair_pixels_by_source_class": selected_by_class,
        "policy_selected_repair_pixels_by_source_class": total_by_class,
        "policy_selected_repair_pixels_before_budget": total_selected_pixels,
        "partial_reason": None,
        "budget_limited": False,
    }


def _masked_agreement(source_masks: Any, candidate_masks: Any, selector: np.ndarray) -> dict[str, Any]:
    source_np = source_masks.cpu().numpy()
    candidate_np = candidate_masks.cpu().numpy()
    if tuple(source_np.shape) != tuple(candidate_np.shape) or tuple(source_np.shape) != tuple(selector.shape):
        raise ValueError("source, candidate, and selector shapes must match")
    selected = int(np.count_nonzero(selector))
    if selected == 0:
        return {
            "num_pixels": 0,
            "different_pixels": 0,
            "argmax_agreement": None,
            "argmax_disagreement": None,
        }
    different = int(np.count_nonzero(source_np[selector] != candidate_np[selector]))
    return {
        "num_pixels": selected,
        "different_pixels": different,
        "argmax_agreement": round(1.0 - different / selected, 12),
        "argmax_disagreement": round(different / selected, 12),
    }


def _compress_amr1_payload(raw_payload: bytes, compressor: str) -> tuple[str, bytes]:
    if compressor not in REPAIR_COMPRESSORS:
        raise ValueError(f"unknown repair compressor {compressor!r}")
    if compressor == "raw":
        return DEFAULT_REPAIR_MEMBER_RAW, raw_payload
    if compressor == "zlib":
        return f"{DEFAULT_REPAIR_MEMBER_RAW}.zlib", zlib.compress(raw_payload, level=9)
    if compressor == "lzma_xz":
        return f"{DEFAULT_REPAIR_MEMBER_RAW}.xz", lzma.compress(
            raw_payload,
            format=lzma.FORMAT_XZ,
            preset=9 | lzma.PRESET_EXTREME,
        )
    try:
        import brotli  # type: ignore
    except Exception as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("brotli repair compression requested but brotli is unavailable") from exc
    return f"{DEFAULT_REPAIR_MEMBER_RAW}.br", brotli.compress(raw_payload, quality=11, lgwin=24)


def build_amr1_repair_candidate(
    *,
    source_archive: Path,
    lossy_mask_stream: Path,
    output_archive: Path,
    manifest_json: Path | None = None,
    source_mask_member: str = DEFAULT_SOURCE_MASK_MEMBER,
    lossy_mask_member: str = DEFAULT_LOSSY_MASK_MEMBER,
    lossy_decode_mode: str = "grayscale",
    repair_compressor: str = "zlib",
    policy: ChargedRepairPolicy = ChargedRepairPolicy(),
    class_priority: tuple[int, ...] = DEFAULT_CLASS_PRIORITY,
    max_frames: int | None = None,
    max_repair_pixels: int | None = None,
    max_repair_runs: int | None = None,
    allow_partial_repair: bool = True,
) -> dict[str, Any]:
    source_mask_member = _safe_member_name(source_mask_member)
    lossy_mask_member = _safe_member_name(lossy_mask_member)
    output_archive = output_archive.resolve()
    source_archive = source_archive.resolve()
    lossy_mask_stream = lossy_mask_stream.resolve()
    if not lossy_mask_stream.is_file():
        raise FileNotFoundError(f"lossy mask stream not found: {lossy_mask_stream}")
    if lossy_decode_mode not in {"grayscale", "legacy"}:
        raise ValueError(f"lossy_decode_mode must be 'grayscale' or 'legacy', got {lossy_decode_mode!r}")

    runtime_members, runtime_inventory = _read_runtime_archive_members(
        source_archive,
        source_mask_member=source_mask_member,
    )
    source_masks, source_decode_meta = _decode_source_masks_from_runtime_member(
        runtime_members[source_mask_member],
        source_mask_member,
        max_frames=max_frames,
    )
    alpha = _load_alpha_builder()
    source_masks = alpha._validate_masks(source_masks)
    source_shape = tuple(int(v) for v in source_masks.shape)
    lossy_mask_bytes = lossy_mask_stream.read_bytes()
    if lossy_decode_mode == "legacy":
        candidate_masks, candidate_decode_meta = _decode_source_masks_from_runtime_member(
            lossy_mask_bytes,
            lossy_mask_member,
            max_frames=max_frames,
        )
        candidate_shape = tuple(int(v) for v in candidate_masks.shape)
        if candidate_shape != source_shape:
            raise ValueError(f"lossy decoded mask shape {candidate_shape} != expected source shape {source_shape}")
        candidate_decode_meta = {
            **candidate_decode_meta,
            "decoder_contract": "legacy masks.mkv decode through inflate_renderer.py, then AMR1 repair hook",
        }
    else:
        candidate_masks, candidate_decode_meta = _decode_lossy_grayscale_masks(
            lossy_mask_stream,
            expected_shape=source_shape,
        )
    candidate_masks = alpha._validate_masks(candidate_masks)
    source_sha = alpha._tensor_u8_sha256(source_masks)
    candidate_sha = alpha._tensor_u8_sha256(candidate_masks)
    selector, selector_summary = build_repair_selector(source_masks, candidate_masks, policy)
    runs, run_summary = _selected_repair_runs(
        source_masks,
        selector,
        class_priority=class_priority,
        max_repair_pixels=max_repair_pixels,
        max_repair_runs=max_repair_runs,
        allow_partial_repair=allow_partial_repair,
    )
    total_residual = int(selector_summary["total_residual_pixels"])
    partial_repair = bool(
        run_summary["selected_repair_pixels"] != total_residual
        or run_summary["partial_reason"] is not None
    )
    selection_meta = {
        "strategy": "charged_mask_grammar_pose_sensitive_amr1_repair_v1",
        "policy": policy.as_manifest(),
        "class_priority": [int(v) for v in class_priority],
        "total_residual_pixels": total_residual,
        "policy_selected_repair_pixels_before_budget": int(
            run_summary["policy_selected_repair_pixels_before_budget"]
        ),
        "selected_repair_pixels": int(run_summary["selected_repair_pixels"]),
        "selected_repair_runs": int(run_summary["selected_repair_runs"]),
        "selected_repair_pixels_by_source_class": run_summary["selected_repair_pixels_by_source_class"],
        "policy_selected_repair_pixels_by_source_class": run_summary[
            "policy_selected_repair_pixels_by_source_class"
        ],
        "residual_pixel_coverage": (
            1.0
            if total_residual == 0
            else round(int(run_summary["selected_repair_pixels"]) / total_residual, 12)
        ),
        "policy_selected_residual_fraction": selector_summary["selected_residual_fraction"],
        "partial_repair": partial_repair,
        "partial_reason": run_summary["partial_reason"],
        "budget_limited": bool(run_summary["budget_limited"]),
        "fail_on_partial_repair": not allow_partial_repair,
        "max_repair_pixels": max_repair_pixels,
        "max_repair_runs": max_repair_runs,
    }
    raw_repair_payload = alpha._encode_repair_payload(
        runs,
        shape=source_shape,
        source_mask_sha256=source_sha,
        candidate_mask_sha256=candidate_sha,
        selection_meta=selection_meta,
    )
    repair_member, repair_bytes = _compress_amr1_payload(raw_repair_payload, repair_compressor)
    repaired_masks = alpha._apply_repair_payload(candidate_masks, raw_repair_payload)
    output_members = [
        ("renderer.bin", runtime_members["renderer.bin"]),
        (lossy_mask_member, lossy_mask_bytes),
        (repair_member, repair_bytes),
        ("optimized_poses.bin", runtime_members["optimized_poses.bin"]),
    ]
    _write_deterministic_deflated_archive(output_archive, output_members)
    archive_bytes = output_archive.stat().st_size
    archive_sha = _sha256_path(output_archive)
    member_inventory = _archive_inventory(output_archive)
    member_accounting = {
        name: {
            "role": (
                "runtime_renderer"
                if name == "renderer.bin"
                else "lossy_legacy_mask_base"
                if name == lossy_mask_member and lossy_decode_mode == "legacy"
                else "lossy_grayscale_mask_base"
                if name == lossy_mask_member
                else "charged_amr1_residual_repair"
                if name == repair_member
                else "runtime_optimized_poses"
            ),
            "bytes": len(data),
            "sha256": _sha256_bytes(data),
        }
        for name, data in output_members
    }
    manifest = {
        "schema": AMR1_REPAIR_SCHEMA,
        "tool": TOOL,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_family": (
            "AMR1_charged_residual_over_lossy_legacy_masks"
            if lossy_decode_mode == "legacy"
            else "AMR1_charged_residual_over_lossy_grayscale"
        ),
        "mode": MODE_AMR1_RUNTIME_LEGACY_REPAIR if lossy_decode_mode == "legacy" else MODE_AMR1_RUNTIME_REPAIR,
        "score_claim": False,
        "promotion_eligible": False,
        "exact_evaluable_archive": True,
        "evidence_grade": "empirical_byte_screen_non_score",
        "cuda_jobs_launched": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "non_promotable_reason": "byte-screen archive only; exact CUDA auth eval has not been run",
        "archive": {
            "path": str(output_archive),
            "size_bytes": int(archive_bytes),
            "sha256": archive_sha,
            "members": member_inventory,
            "formula_rate_score_if_components_unchanged": 25.0 * archive_bytes / ORIGINAL_VIDEO_BYTES,
        },
        "source_runtime_archive": {
            "path": str(source_archive),
            "size_bytes": int(source_archive.stat().st_size),
            "sha256": _sha256_path(source_archive),
            "member_inventory": runtime_inventory,
            "source_mask_member": source_mask_member,
        },
        "source_mask_stream": {
            "member": source_mask_member,
            "size_bytes": len(runtime_members[source_mask_member]),
            "sha256": _sha256_bytes(runtime_members[source_mask_member]),
            "decoded_class_u8_sha256": source_sha,
            "decoded_stats": alpha._mask_stats(source_masks),
            "decode": source_decode_meta,
        },
        "lossy_mask_base": {
            "input_path": str(lossy_mask_stream),
            "archive_member": lossy_mask_member,
            "decode_mode": lossy_decode_mode,
            "size_bytes": len(lossy_mask_bytes),
            "sha256": _sha256_bytes(lossy_mask_bytes),
            "decoded_class_u8_sha256": candidate_sha,
            "decoded_stats": alpha._mask_stats(candidate_masks),
            "decode": candidate_decode_meta,
            "agreement_vs_source_before_repair": alpha._agreement_metrics(source_masks, candidate_masks),
        },
        "repair_payload": {
            "archive_member": repair_member,
            "compressor": repair_compressor,
            "raw_amr1_size_bytes": int(len(raw_repair_payload)),
            "raw_amr1_sha256": _sha256_bytes(raw_repair_payload),
            "compressed_size_bytes": int(len(repair_bytes)),
            "compressed_sha256": _sha256_bytes(repair_bytes),
            "record_count": int(len(runs)),
            "selection": selection_meta,
        },
        "repair_selector": selector_summary,
        "candidate_after_repair": {
            "decoded_class_u8_sha256": alpha._tensor_u8_sha256(repaired_masks),
            "decoded_stats": alpha._mask_stats(repaired_masks),
            "agreement_vs_source_after_repair": alpha._agreement_metrics(source_masks, repaired_masks),
            "selected_region_agreement_after_repair": _masked_agreement(source_masks, repaired_masks, selector),
        },
        "charged_member_accounting": member_accounting,
        "runtime_contract": {
            "archive_members": [name for name, _data in output_members],
            "masks_mkv_omitted": lossy_mask_member != "masks.mkv",
            "legacy_masks_member": lossy_mask_member if lossy_mask_member == "masks.mkv" else None,
            "grayscale_runtime_member": lossy_mask_member if lossy_mask_member == "grayscale.mkv" else None,
            "repair_member_supported_by_runtime": repair_member
            in {
                DEFAULT_REPAIR_MEMBER_RAW,
                f"{DEFAULT_REPAIR_MEMBER_RAW}.zlib",
                f"{DEFAULT_REPAIR_MEMBER_RAW}.xz",
                f"{DEFAULT_REPAIR_MEMBER_RAW}.br",
            },
            "inflate_auto_dispatch": (
                "submissions/robust_current/inflate.sh uses legacy renderer when masks.mkv exists; "
                "inflate_renderer.py applies charged AMR1 repair after legacy mask decode"
                if lossy_mask_member == "masks.mkv"
                else "submissions/robust_current/inflate.sh selects renderer_grayscale when grayscale.mkv exists and masks.mkv is absent"
            ),
            "decoder_loads_scorer": False,
            "external_sidecars_allowed": False,
        },
    }
    if manifest_json is None:
        manifest_json = output_archive.with_name("charged_mask_grammar_manifest.json")
    manifest_json = manifest_json.resolve()
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_bytes(_pretty_json_bytes(manifest))
    return manifest


def _archive_manifest(
    *,
    mode: str,
    shape: CMG1Shape,
    payload_member: str,
    manifest_member: str,
    payload_bytes: bytes,
    source_bytes: bytes | None,
    base_archive: Path | None = None,
    replaced_mask_member: str | None = None,
) -> dict[str, Any]:
    source_record: dict[str, Any] | None = None
    if source_bytes is not None:
        source_record = {
            "bytes": len(source_bytes),
            "sha256": _sha256_bytes(source_bytes),
            "bit_identical_payload_body": True,
        }
    runtime_integrated = source_bytes is not None
    full_archive_candidate = base_archive is not None and source_bytes is not None
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "candidate_family": "CMG1",
        "mode": mode,
        "score_claim": False,
        "promotion_eligible": False,
        "exact_evaluable_archive": False,
        "evidence_grade": "empirical_build_only_non_score",
        "non_promotable_reason": (
            "CMG1 runtime decode path is present, but auth-eval/local-smoke "
            "archive allowlists must admit .cmg1 before a full archive is "
            "score-evaluable through the hardened exact-eval harness; no CUDA "
            "auth eval has been run"
            if runtime_integrated
            else "CMG1 payload has no raw mask stream and no CUDA auth eval has been run"
        ),
        "runtime_integration": {
            "inflate_runtime_touched": runtime_integrated,
            "requires_future_decoder": False if runtime_integrated else True,
            "decoder_loads_scorer": False,
            "external_sidecars_allowed": False,
            "cmg1_raw_stream_decoder": runtime_integrated,
            "auth_eval_allowlist_update_required": True,
        },
        "archive_assembly": {
            "full_archive_candidate": full_archive_candidate,
            "base_archive": str(base_archive) if base_archive is not None else None,
            "replaced_mask_member": replaced_mask_member,
            "default_behavior_unchanged": True,
            "opt_in_required": True,
        },
        "wire_contract": _wire_contract(
            shape,
            payload_member=payload_member,
            manifest_member=manifest_member,
        ),
        "charged_member_accounting": {
            payload_member: {
                "role": "cmg1_payload",
                "bytes": len(payload_bytes),
                "sha256": _sha256_bytes(payload_bytes),
            },
            manifest_member: {
                "role": "cmg1_manifest_json",
                "self_hash_recorded_in_adjacent_provenance": True,
            },
        },
        "source_mask_stream": source_record,
        "formula_context": {
            "original_video_bytes_denominator": ORIGINAL_VIDEO_BYTES,
            "rate_term_requires_final_archive_bytes": True,
        },
        "required_next_steps_for_exact_evaluable_archive": [
            "add .cmg1 to experiments/contest_auth_eval.py and canonical_local_auth_eval_smoke.py allowlists together",
            "run a hardened auth-eval smoke extraction/whitelist pass on the full archive",
            "validate decoded mask shape and SHA from the inflated archive",
            "run experiments/contest_auth_eval.py --device cuda on the exact archive bytes",
        ],
    }


def build_candidate(
    *,
    input_mask_stream: Path | None,
    output_archive: Path,
    provenance_json: Path | None = None,
    payload_member: str = DEFAULT_PAYLOAD_MEMBER,
    manifest_member: str = DEFAULT_MANIFEST_MEMBER,
    base_archive: Path | None = None,
    replaced_mask_member: str = DEFAULT_REPLACED_MASK_MEMBER,
    shape: CMG1Shape = CMG1Shape(),
) -> dict[str, Any]:
    if base_archive is not None and payload_member == DEFAULT_PAYLOAD_MEMBER:
        payload_member = DEFAULT_RUNTIME_PAYLOAD_MEMBER
    payload_member = _safe_member_name(payload_member)
    manifest_member = _safe_member_name(manifest_member)
    replaced_mask_member = _safe_existing_archive_member(replaced_mask_member)
    if payload_member == manifest_member:
        raise ValueError("payload_member and manifest_member must be distinct")
    if base_archive is not None and input_mask_stream is None:
        raise ValueError("base_archive CMG1 replacement requires input_mask_stream")

    source_bytes: bytes | None = None
    source_path: Path | None = None
    if input_mask_stream is not None:
        source_path = input_mask_stream.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"input mask stream not found: {input_mask_stream}")
        source_bytes = source_path.read_bytes()

    mode = MODE_RAW_BIT_IDENTICAL if source_bytes is not None else MODE_PLACEHOLDER
    payload_bytes = encode_cmg1_payload(
        source_bytes=source_bytes,
        shape=shape,
        payload_member=payload_member,
        manifest_member=manifest_member,
    )
    manifest = _archive_manifest(
        mode=mode,
        shape=shape,
        payload_member=payload_member,
        manifest_member=manifest_member,
        payload_bytes=payload_bytes,
        source_bytes=source_bytes,
        base_archive=base_archive.resolve() if base_archive is not None else None,
        replaced_mask_member=replaced_mask_member if base_archive is not None else None,
    )
    manifest_bytes = _pretty_json_bytes(manifest)

    output_archive = output_archive.resolve()
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    copied_members: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(output_archive, "w") as zf:
        if base_archive is not None:
            base_archive = base_archive.resolve()
            with zipfile.ZipFile(base_archive, "r") as base_zf:
                infos = base_zf.infolist()
                _validate_base_archive_members(infos)
                names = [info.filename for info in infos]
                if replaced_mask_member not in names:
                    raise ValueError(
                        f"base archive does not contain mask member to replace: {replaced_mask_member!r}"
                    )
                reserved = {payload_member, manifest_member}
                conflicts = sorted(name for name in names if name in reserved and name != replaced_mask_member)
                if conflicts:
                    raise ValueError(f"base archive already contains CMG1 output member(s): {conflicts}")
                for info in infos:
                    name = _safe_existing_archive_member(info.filename)
                    if name == replaced_mask_member:
                        continue
                    data = base_zf.read(info)
                    zf.writestr(_base_zip_info(name), data)
                    copied_members[name] = {
                        "bytes": len(data),
                        "sha256": _sha256_bytes(data),
                        "source": "base_archive",
                    }
        zf.writestr(_zip_info(payload_member), payload_bytes)
        zf.writestr(_zip_info(manifest_member), manifest_bytes)

    member_records = {
        payload_member: {
            "bytes": len(payload_bytes),
            "sha256": _sha256_bytes(payload_bytes),
        },
        manifest_member: {
            "bytes": len(manifest_bytes),
            "sha256": _sha256_bytes(manifest_bytes),
        },
    }
    archive_bytes = output_archive.stat().st_size
    provenance = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "exact_evaluable_archive": False,
        "output_archive": str(output_archive),
        "output_archive_bytes": archive_bytes,
        "output_archive_sha256": _sha256_path(output_archive),
        "output_members": member_records,
        "copied_members": copied_members,
        "archive_manifest_sha256": member_records[manifest_member]["sha256"],
        "archive_manifest_bytes": member_records[manifest_member]["bytes"],
        "payload_sha256": member_records[payload_member]["sha256"],
        "payload_bytes": member_records[payload_member]["bytes"],
        "mode": mode,
        "base_archive": str(base_archive.resolve()) if base_archive is not None else None,
        "replaced_mask_member": replaced_mask_member if base_archive is not None else None,
        "source_mask_stream": manifest["source_mask_stream"],
        "source_mask_stream_path": str(source_path) if source_path is not None else None,
        "formula_rate_score_if_standalone_archive": 25.0 * archive_bytes / ORIGINAL_VIDEO_BYTES,
        "note": (
            "full CMG1 archive candidate pending .cmg1 auth-eval/smoke allowlist and CUDA eval; not a score claim"
            if base_archive is not None
            else "standalone CMG1 scaffold only; not a full contest archive and not a score claim"
        ),
    }
    if provenance_json is not None:
        provenance_json = provenance_json.resolve()
        provenance_json.parent.mkdir(parents=True, exist_ok=True)
        provenance_json.write_bytes(_pretty_json_bytes(provenance))
    return provenance


def _default_pose_sensitive_policy(
    *,
    hard_pair_indices: tuple[int, ...],
    hard_frame_indices: tuple[int, ...],
    class_ids: tuple[int, ...],
    boundary_dilation: int,
    horizon_bands: tuple[RegionSpec, ...],
    foveal_boxes: tuple[RegionSpec, ...],
    ego_boxes: tuple[RegionSpec, ...],
    select_all_differences: bool,
    label: str,
) -> ChargedRepairPolicy:
    if not horizon_bands:
        horizon_bands = (RegionSpec(name="horizon_band", x0=0, y0=150, x1=-1, y1=210),)
    if not foveal_boxes:
        foveal_boxes = (RegionSpec(name="foveal_vanish_region", x0=160, y0=96, x1=352, y1=288),)
    if not ego_boxes:
        ego_boxes = (RegionSpec(name="ego_hood_motion_region", x0=144, y0=252, x1=368, y1=-1),)
    return ChargedRepairPolicy(
        label=label,
        hard_pair_indices=hard_pair_indices,
        hard_frame_indices=hard_frame_indices,
        class_ids=class_ids,
        boundary_dilation=boundary_dilation,
        horizon_bands=horizon_bands,
        foveal_boxes=foveal_boxes,
        ego_boxes=ego_boxes,
        select_all_differences=select_all_differences,
    )


def _charged_policy_from_args(args: argparse.Namespace) -> ChargedRepairPolicy:
    explicit_pairs = _parse_int_set(args.hard_pair_indices, field="hard_pair_indices")
    trace_pairs = _hard_pair_indices_from_component_trace(
        args.component_trace_json,
        top_k=int(args.top_hard_pairs),
        sort_key=args.hard_pair_sort_key,
    )
    hard_pair_indices = tuple(sorted(set(explicit_pairs) | set(trace_pairs)))
    hard_frame_indices = _parse_int_set(args.hard_frame_indices, field="hard_frame_indices")
    class_ids = _parse_int_set(args.protect_classes, field="protect_classes")
    horizon_bands = tuple(_parse_horizon_band(value) for value in args.horizon_band)
    foveal_boxes = tuple(_parse_region_spec(value, name="foveal_box") for value in args.foveal_box)
    ego_boxes = tuple(_parse_region_spec(value, name="ego_box") for value in args.ego_box)
    boundary_dilation = 1 if args.boundary_dilation is None else int(args.boundary_dilation)
    if args.policy_label:
        label = args.policy_label
    elif args.policy_preset == "all_differences":
        label = "all_residual_pixels_amr1"
    else:
        label = "pose_sensitive_boundary_horizon_foveal_ego_v1"
    return _default_pose_sensitive_policy(
        hard_pair_indices=hard_pair_indices,
        hard_frame_indices=hard_frame_indices,
        class_ids=class_ids,
        boundary_dilation=boundary_dilation,
        horizon_bands=() if args.policy_preset == "all_differences" else horizon_bands,
        foveal_boxes=() if args.policy_preset == "all_differences" else foveal_boxes,
        ego_boxes=() if args.policy_preset == "all_differences" else ego_boxes,
        select_all_differences=bool(args.repair_all_differences or args.policy_preset == "all_differences"),
        label=label,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-mode", choices=("cmg1", "amr1-repair"), default="cmg1")
    parser.add_argument("--input-mask-stream", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, default=None)
    parser.add_argument("--provenance-json", type=Path, default=None)
    parser.add_argument("--payload-member", default=DEFAULT_PAYLOAD_MEMBER)
    parser.add_argument("--manifest-member", default=DEFAULT_MANIFEST_MEMBER)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--class-count", type=int, default=DEFAULT_CLASS_COUNT)
    parser.add_argument(
        "--base-archive",
        type=Path,
        default=None,
        help=(
            "Optional full candidate archive to copy while replacing "
            "--replaced-mask-member with a charged CMG1 payload. This is "
            "opt-in and still non-promotable until .cmg1 is admitted by the "
            "auth-eval and smoke allowlists and CUDA eval is run."
        ),
    )
    parser.add_argument("--replaced-mask-member", default=DEFAULT_REPLACED_MASK_MEMBER)
    parser.add_argument("--source-runtime-archive", "--source-archive", type=Path, default=None)
    parser.add_argument("--lossy-mask-stream", type=Path, default=None)
    parser.add_argument("--source-mask-member", default=DEFAULT_SOURCE_MASK_MEMBER)
    parser.add_argument("--lossy-mask-member", default=DEFAULT_LOSSY_MASK_MEMBER)
    parser.add_argument("--lossy-decode-mode", choices=("grayscale", "legacy"), default="grayscale")
    parser.add_argument("--repair-compressor", choices=REPAIR_COMPRESSORS, default="zlib")
    parser.add_argument(
        "--policy-preset",
        choices=("pose_sensitive_boundary_horizon_foveal_ego", "all_differences"),
        default="pose_sensitive_boundary_horizon_foveal_ego",
    )
    parser.add_argument("--policy-label", default=None)
    parser.add_argument("--component-trace-json", type=Path, default=None)
    parser.add_argument("--top-hard-pairs", type=int, default=0)
    parser.add_argument("--hard-pair-sort-key", default="posenet_dist")
    parser.add_argument("--hard-pair-indices", default=None, help="Comma/range list; pair i expands to frames 2*i,2*i+1.")
    parser.add_argument("--hard-frame-indices", default=None, help="Comma/range list of decoded mask frame indices.")
    parser.add_argument("--protect-classes", default=None, help="Comma/range list of source class ids to repair.")
    parser.add_argument("--boundary-dilation", type=int, default=None)
    parser.add_argument("--horizon-band", action="append", default=[], help="y0:y1[@frames=csv/ranges], x spans full width.")
    parser.add_argument("--foveal-box", action="append", default=[], help="x0,y0,x1,y1[@frames=csv/ranges].")
    parser.add_argument("--ego-box", action="append", default=[], help="x0,y0,x1,y1[@frames=csv/ranges].")
    parser.add_argument("--repair-all-differences", action="store_true")
    parser.add_argument("--class-priority", default="2,1,3,4,0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-repair-pixels", type=int, default=None)
    parser.add_argument("--max-repair-runs", type=int, default=None)
    parser.add_argument("--fail-on-partial-repair", action="store_true")
    args = parser.parse_args(argv)

    output_archive = args.output_archive or (args.output_dir / "archive.zip")
    provenance_json = args.provenance_json or (args.output_dir / "build_provenance.json")
    if args.candidate_mode == "amr1-repair":
        if args.source_runtime_archive is None:
            parser.error("--candidate-mode amr1-repair requires --source-runtime-archive")
        if args.lossy_mask_stream is None:
            parser.error("--candidate-mode amr1-repair requires --lossy-mask-stream")
        manifest = build_amr1_repair_candidate(
            source_archive=args.source_runtime_archive,
            lossy_mask_stream=args.lossy_mask_stream,
            output_archive=output_archive,
            manifest_json=provenance_json,
            source_mask_member=args.source_mask_member,
            lossy_mask_member=args.lossy_mask_member,
            lossy_decode_mode=args.lossy_decode_mode,
            repair_compressor=args.repair_compressor,
            policy=_charged_policy_from_args(args),
            class_priority=_parse_int_sequence(args.class_priority, field="class_priority"),
            max_frames=args.max_frames,
            max_repair_pixels=args.max_repair_pixels,
            max_repair_runs=args.max_repair_runs,
            allow_partial_repair=not args.fail_on_partial_repair,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    provenance = build_candidate(
        input_mask_stream=args.input_mask_stream,
        output_archive=output_archive,
        provenance_json=provenance_json,
        payload_member=args.payload_member,
        manifest_member=args.manifest_member,
        base_archive=args.base_archive,
        replaced_mask_member=args.replaced_mask_member,
        shape=CMG1Shape(
            frames=args.frames,
            height=args.height,
            width=args.width,
            class_count=args.class_count,
        ),
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
