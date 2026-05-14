#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build deterministic Alpha mask primitive component-response plans.

This generator prepares fast exact-eval sweeps over Alpha mask geometry
perturbations.  It emits an ``official_component_response_plan_v1`` compatible
JSON plan plus deterministic archive variants, but it does not run scorers and
does not make score evidence.  Every emitted point is diagnostic and
non-promotable until later CUDA auth eval artifacts are attached through:

    archive.zip -> inflate.sh -> upstream/evaluate.py
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PRODUCER = "experiments/build_alpha_mask_primitive_response_plan.py"
PLAN_FORMAT = "official_component_response_plan_v1"
ALPHA_PLAN_FORMAT = "alpha_mask_primitive_component_response_plan_v1"
VARIANT_MANIFEST_FORMAT = "alpha_mask_primitive_archive_variants_v1"
EVIDENCE_GRADE = "empirical"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
SCORE_CLAIM_WARNING = (
    "No score claim is made by this plan. Exact CUDA auth eval must be attached "
    "before any score claim, promotion, ranking, or method retirement."
)
DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/alpha_mask_primitive_response_plan"
CLASS_IDS = (0, 1, 2, 3, 4)
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FIXED_ZIP_PERMISSIONS = 0o644
SCORE_EPS = 1e-12  # [heuristic: numerical guard for log/division stability]
NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HIDDEN_SYSTEM_NAMES = {"__MACOSX", ".DS_Store", "Thumbs.db", "desktop.ini"}

BUILDER_PATH = REPO_ROOT / "experiments" / "alpha_mask_candidate_builder.py"


class AlphaPrimitivePlanError(ValueError):
    """Raised when Alpha primitive plan inputs or outputs are unsafe."""


@dataclass(frozen=True)
class PlanConfig:
    scan_frame_count: int = 48
    max_points: int = 32
    max_component_points: int = 8
    max_boundary_points: int = 8
    max_class_flip_points: int = 6
    max_morph_points: int = 8
    max_transition_points: int = 8
    max_components_per_class: int = 2
    max_pixels_per_point: int = 4096
    boundary_width: int = 1
    mask_crf: int = 63
    mask_fps: int = 20
    max_frames: int | None = None


@dataclass(frozen=True)
class ArchiveMember:
    name: str
    data: bytes
    source_info: Mapping[str, Any]


@dataclass(frozen=True)
class ComponentRef:
    frame_index: int
    class_id: int
    scan_index: int
    seed_y: int
    seed_x: int
    area: int
    bbox_xyxy_exclusive: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    boundary_pixels_4conn: int
    boundary_edges_4conn: int
    neighbor_histogram: Mapping[str, int]


@dataclass(frozen=True)
class PrimitiveMutation:
    primitive_id: str
    kind: str
    operation: str
    frame_index: int | None
    source_class: int | None
    target_class: int | None
    rank_pixels: int
    selection_weight: float
    params: Mapping[str, Any]
    description: str


def _load_experiment_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_experiment_module("alpha_mask_candidate_builder_for_primitive_plan", BUILDER_PATH)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _canonical_hash_any(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round_float(value: float, digits: int = 12) -> float:
    return round(float(value), digits)


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlphaPrimitivePlanError(f"{field} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise AlphaPrimitivePlanError(f"{field} must be finite")
    return out


def _finite_weight(value: Any, *, field: str) -> float:
    out = _finite_float(value, field=field)
    if out < 0.0 or out > 1.0:
        raise AlphaPrimitivePlanError(f"{field} must be in [0, 1], got {out!r}")
    return out


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AlphaPrimitivePlanError(f"{field} must be an integer")
    if value <= 0:
        raise AlphaPrimitivePlanError(f"{field} must be positive")
    return int(value)


def _validate_config(config: PlanConfig) -> None:
    _require_positive_int(config.scan_frame_count, field="scan_frame_count")
    _require_positive_int(config.max_points, field="max_points")
    _require_positive_int(config.max_component_points, field="max_component_points")
    _require_positive_int(config.max_boundary_points, field="max_boundary_points")
    _require_positive_int(config.max_class_flip_points, field="max_class_flip_points")
    _require_positive_int(config.max_morph_points, field="max_morph_points")
    _require_positive_int(config.max_transition_points, field="max_transition_points")
    if config.max_components_per_class < 0:
        raise AlphaPrimitivePlanError("max_components_per_class must be nonnegative")
    _require_positive_int(config.max_pixels_per_point, field="max_pixels_per_point")
    _require_positive_int(config.boundary_width, field="boundary_width")
    if not (0 <= config.mask_crf <= 63):
        raise AlphaPrimitivePlanError(f"mask_crf must be in [0, 63], got {config.mask_crf}")
    _require_positive_int(config.mask_fps, field="mask_fps")
    if config.max_frames is not None:
        _require_positive_int(config.max_frames, field="max_frames")


def _validate_member_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise AlphaPrimitivePlanError("archive member name must be non-empty")
    if "\x00" in name or "\\" in name:
        raise AlphaPrimitivePlanError(f"unsafe archive member path: {name!r}")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise AlphaPrimitivePlanError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute():
        raise AlphaPrimitivePlanError(f"unsafe archive member path: {name!r}")
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise AlphaPrimitivePlanError(f"unsafe archive member path: {name!r}")
    lowered = [part.lower() for part in parts]
    if any(part.startswith(".") or part.startswith("._") for part in parts):
        raise AlphaPrimitivePlanError(f"hidden archive sidecar rejected: {name!r}")
    if any(part in {item.lower() for item in HIDDEN_SYSTEM_NAMES} for part in lowered):
        raise AlphaPrimitivePlanError(f"hidden archive sidecar rejected: {name!r}")
    if any(part == "__macosx" or part.startswith("._") for part in lowered):
        raise AlphaPrimitivePlanError(f"resource fork sidecar rejected: {name!r}")
    return path.as_posix()


def _validate_plan_relative_path(path_text: str, *, field: str) -> str:
    if not isinstance(path_text, str) or not path_text:
        raise AlphaPrimitivePlanError(f"{field} must be a non-empty relative path")
    if "\x00" in path_text or "\\" in path_text:
        raise AlphaPrimitivePlanError(f"{field} is unsafe: {path_text!r}")
    if path_text.startswith("/") or re.match(r"^[A-Za-z]:", path_text):
        raise AlphaPrimitivePlanError(f"{field} must be relative: {path_text!r}")
    path = PurePosixPath(path_text)
    if path.is_absolute():
        raise AlphaPrimitivePlanError(f"{field} must be relative: {path_text!r}")
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise AlphaPrimitivePlanError(f"{field} must not contain traversal: {path_text!r}")
    lowered = [part.lower() for part in parts]
    if any(part.startswith(".") or part.startswith("._") for part in parts):
        raise AlphaPrimitivePlanError(f"{field} hidden path rejected: {path_text!r}")
    if any(part == "__macosx" or part in {"thumbs.db", "desktop.ini"} for part in lowered):
        raise AlphaPrimitivePlanError(f"{field} hidden/system path rejected: {path_text!r}")
    return path.as_posix()


def _zipinfo_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def _read_custody_checked_archive(path: Path) -> dict[str, ArchiveMember]:
    if not path.is_file():
        raise AlphaPrimitivePlanError(f"baseline archive not found: {path}")
    members: dict[str, ArchiveMember] = {}
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    raise AlphaPrimitivePlanError(
                        f"directory entries are not allowed in archive: {info.filename!r}"
                    )
                name = _validate_member_name(info.filename)
                if name in members:
                    raise AlphaPrimitivePlanError(f"duplicate archive member rejected: {name!r}")
                if _zipinfo_is_symlink(info):
                    raise AlphaPrimitivePlanError(f"symlink archive member rejected: {name!r}")
                if info.flag_bits & 0x1:
                    raise AlphaPrimitivePlanError(f"encrypted archive member rejected: {name!r}")
                data = zf.read(info)
                members[name] = ArchiveMember(
                    name=name,
                    data=data,
                    source_info={
                        "name": name,
                        "raw_bytes": int(info.file_size),
                        "compressed_bytes": int(info.compress_size),
                        "crc32": f"{info.CRC:08x}",
                        "date_time": list(info.date_time),
                        "compress_type": int(info.compress_type),
                        "permissions_octal": oct((info.external_attr >> 16) & 0o777),
                        "sha256": _sha256_bytes(data),
                    },
                )
    except zipfile.BadZipFile as exc:
        raise AlphaPrimitivePlanError(f"baseline archive is not a valid zip: {path}") from exc
    if not members:
        raise AlphaPrimitivePlanError("baseline archive has no members")
    return members


def _file_meta(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    record = {
        "path": _path_for_plan(path, root=root) if root is not None else str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }
    return record


def _repo_relative_hint(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def _path_for_plan(path: Path, *, root: Path) -> str:
    rel = os.path.relpath(path.resolve(), root.resolve())
    rel_posix = Path(rel).as_posix()
    return _validate_plan_relative_path(rel_posix, field="plan path")


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    label = label.strip("._-")
    return label[:96] or "point"


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_validate_member_name(name), date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (FIXED_ZIP_PERMISSIONS & 0xFFFF) << 16
    return info


def _archive_bytes(members: Mapping[str, bytes], *, member_order: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in member_order:
            zf.writestr(_zip_info(name), members[name], compresslevel=9)
    return buffer.getvalue()


def _archive_manifest_from_bytes(data: bytes) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        for info in zf.infolist():
            member_data = zf.read(info)
            manifest.append(
                {
                    "name": info.filename,
                    "raw_bytes": int(info.file_size),
                    "compressed_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "date_time": list(info.date_time),
                    "compress_type": int(info.compress_type),
                    "permissions_octal": oct((info.external_attr >> 16) & 0o777),
                    "sha256": _sha256_bytes(member_data),
                }
            )
    return manifest


def _validate_masks(masks: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(masks, torch.Tensor):
        arr = masks.detach().cpu().to(torch.int64).contiguous().numpy()
    elif isinstance(masks, np.ndarray):
        arr = np.asarray(masks)
    else:
        raise TypeError(f"masks must be a torch.Tensor or np.ndarray, got {type(masks).__name__}")
    if arr.ndim != 3:
        raise ValueError(f"masks must have shape (T,H,W), got {tuple(arr.shape)}")
    if arr.size == 0:
        raise ValueError("masks tensor is empty")
    if not np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.floating) and np.all(np.isfinite(arr)):
            rounded = np.rint(arr)
            if not np.array_equal(rounded, arr):
                raise ValueError("floating masks contain non-integer class ids")
            arr = rounded.astype(np.int64)
        else:
            raise ValueError(f"masks dtype must be integer class ids, got {arr.dtype}")
    min_value = int(arr.min())
    max_value = int(arr.max())
    if min_value < min(CLASS_IDS) or max_value > max(CLASS_IDS):
        raise ValueError(f"masks must contain class ids in [0,4], got [{min_value},{max_value}]")
    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def _tensor_u8_sha256(masks: np.ndarray) -> str:
    masks = _validate_masks(masks)
    return _sha256_bytes(masks.tobytes())


def _mask_stats(masks: np.ndarray) -> dict[str, Any]:
    masks = _validate_masks(masks)
    counts = np.bincount(masks.reshape(-1), minlength=max(CLASS_IDS) + 1)
    total = int(masks.size)
    vertical_edges = int(np.count_nonzero(masks[:, 1:, :] != masks[:, :-1, :])) if masks.shape[1] > 1 else 0
    horizontal_edges = int(np.count_nonzero(masks[:, :, 1:] != masks[:, :, :-1])) if masks.shape[2] > 1 else 0
    temporal_changes = int(np.count_nonzero(masks[1:] != masks[:-1])) if masks.shape[0] > 1 else 0
    return {
        "shape": [int(value) for value in masks.shape],
        "dtype": "uint8_class_ids",
        "num_pixels": total,
        "class_histogram": {str(class_id): int(counts[class_id]) for class_id in CLASS_IDS},
        "class_fractions": {
            str(class_id): _round_float(int(counts[class_id]) / total) for class_id in CLASS_IDS
        },
        "spatial_boundary_edges_4conn": int(vertical_edges + horizontal_edges),
        "temporal_changed_pixels": int(temporal_changes),
        "class_id_u8_sha256": _tensor_u8_sha256(masks),
    }


def _decode_masks_from_archive_member(
    *,
    members: Mapping[str, ArchiveMember],
    mask_member: str,
    max_frames: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    member = _validate_member_name(mask_member)
    if member not in members:
        raise AlphaPrimitivePlanError(f"baseline archive missing mask member {member!r}")
    masks, decode_meta = builder._decode_legacy_av1_masks_from_member(
        members[member].data,
        member,
        max_frames=max_frames,
    )
    return _validate_masks(masks), dict(decode_meta)


def _load_decoded_masks_source(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise AlphaPrimitivePlanError(f"decoded mask source not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping):
            for key in ("masks", "decoded_masks", "candidate_masks"):
                if key in payload:
                    payload = payload[key]
                    break
        masks = _validate_masks(payload)
        loader = "torch.load"
    elif suffix == ".npy":
        masks = _validate_masks(np.load(path, allow_pickle=False))
        loader = "numpy.load_npy"
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            key = "masks" if "masks" in payload.files else sorted(payload.files)[0]
            masks = _validate_masks(payload[key])
        loader = "numpy.load_npz"
    else:
        raise AlphaPrimitivePlanError(
            f"unsupported decoded mask source suffix {path.suffix!r}; use .pt, .pth, .npy, or .npz"
        )
    return masks, {
        "source_kind": "decoded_masks_source",
        "loader": loader,
        "path_hint": _repo_relative_hint(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _resolve_existing_manifest_path(raw: str, *, manifest_dir: Path) -> Path:
    if not raw or "\x00" in raw:
        raise AlphaPrimitivePlanError(f"unsafe empty or NUL path in manifest: {raw!r}")
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [manifest_dir / path, REPO_ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"manifest path does not exist: {raw!r}")


def _source_from_candidate_manifest(path: Path) -> tuple[Path, str, dict[str, Any]]:
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AlphaPrimitivePlanError(f"{path}: invalid candidate manifest JSON") from exc
    if not isinstance(manifest, Mapping):
        raise AlphaPrimitivePlanError(f"{path}: candidate manifest must be a JSON object")
    if manifest.get("score_claim") is not False or manifest.get("promotion_eligible") is not False:
        raise AlphaPrimitivePlanError("candidate manifest must be non-promotable planning evidence")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise AlphaPrimitivePlanError("candidate manifest missing source object")
    mask_member = source.get("mask_member")
    if not isinstance(mask_member, Mapping) or not isinstance(mask_member.get("name"), str):
        raise AlphaPrimitivePlanError("candidate manifest missing source.mask_member.name")
    archive_path = _resolve_existing_manifest_path(str(source.get("archive_path", "")), manifest_dir=path.parent)
    member = _validate_member_name(str(mask_member["name"]))
    return archive_path, member, {
        "source_kind": "alpha_mask_candidate_manifest",
        "candidate_manifest": {
            "path_hint": _repo_relative_hint(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
            "schema": manifest.get("schema"),
        },
    }


def _run_git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _source_file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _provenance(command: list[str] | None) -> dict[str, Any]:
    status = _run_git(["status", "--short", "--untracked-files=no"])
    return {
        "tool": PRODUCER,
        "command": list(command) if command is not None else list(sys.argv),
        "cwd_hint": _repo_relative_hint(Path.cwd()) or ".",
        "repo_root_hint": REPO_ROOT.name,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "selected_environment": {
            key: os.environ[key]
            for key in (
                "PYTHONHASHSEED",
                "UV_PROJECT_ENVIRONMENT",
                "TAC_FFMPEG",
                "TAC_FFPROBE",
                "TAC_UPSTREAM_DIR",
                "CUDA_VISIBLE_DEVICES",
            )
            if key in os.environ
        },
        "git": {
            "head": _run_git(["rev-parse", "HEAD"]),
            "dirty_tracked_files": bool(status),
            "status_short_tracked": status.splitlines() if status else [],
        },
        "source_files": [
            _source_file_meta(REPO_ROOT / "experiments" / "build_alpha_mask_primitive_response_plan.py"),
            _source_file_meta(BUILDER_PATH),
            _source_file_meta(REPO_ROOT / "experiments" / "alpha_primitive_mask_diagnostics.py"),
            _source_file_meta(REPO_ROOT / "experiments" / "alpha_mask_residual_planner.py"),
            _source_file_meta(REPO_ROOT / "src" / "tac" / "submission_archive.py"),
        ],
    }


def _sample_frame_indices(frame_count: int, max_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= max_count:
        return list(range(frame_count))
    values = np.linspace(0, frame_count - 1, num=max_count)
    indices = sorted({int(round(value)) for value in values})
    cursor = 0
    while len(indices) < max_count and cursor < frame_count:
        indices.append(cursor)
        indices = sorted(set(indices))
        cursor += 1
    return indices[:max_count]


def _component_sort_key(component: ComponentRef) -> tuple[int, int, int, int, int]:
    x0, y0, _x1, _y1 = component.bbox_xyxy_exclusive
    return (-component.area, -component.boundary_edges_4conn, y0, x0, component.scan_index)


def _fallback_target_class(class_id: int) -> int:
    return int((class_id + 1) % (max(CLASS_IDS) + 1))


def _dominant_neighbor_from_hist(histogram: Mapping[str, int], source_class: int) -> int:
    candidates = [
        (int(count), int(class_id))
        for class_id, count in histogram.items()
        if int(class_id) in CLASS_IDS and int(class_id) != source_class and int(count) > 0
    ]
    if not candidates:
        return _fallback_target_class(source_class)
    count, class_id = max(candidates, key=lambda item: (item[0], -item[1]))
    _ = count
    return int(class_id)


def _frame_components(frame: np.ndarray, *, frame_index: int) -> list[ComponentRef]:
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2D, got {frame.shape}")
    height, width = [int(v) for v in frame.shape]
    visited = np.zeros((height, width), dtype=np.bool_)
    components: list[ComponentRef] = []
    allowed = set(CLASS_IDS)
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
            neighbor_counts = {str(class_key): 0 for class_key in CLASS_IDS}

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

                for dy, dx in NEIGHBORS_4:
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        is_boundary_pixel = True
                        boundary_edges += 1
                        continue
                    neighbor_class = int(frame[ny, nx])
                    if neighbor_class != class_id:
                        is_boundary_pixel = True
                        boundary_edges += 1
                        if neighbor_class in allowed:
                            neighbor_counts[str(neighbor_class)] += 1
                        continue
                    if not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

                if is_boundary_pixel:
                    boundary_pixels += 1

            components.append(
                ComponentRef(
                    frame_index=int(frame_index),
                    class_id=int(class_id),
                    scan_index=int(scan_index),
                    seed_y=int(start_y),
                    seed_x=int(start_x),
                    area=int(area),
                    bbox_xyxy_exclusive=(int(min_x), int(min_y), int(max_x + 1), int(max_y + 1)),
                    centroid_xy=(_round_float(sum_x / area, 6), _round_float(sum_y / area, 6)),
                    boundary_pixels_4conn=int(boundary_pixels),
                    boundary_edges_4conn=int(boundary_edges),
                    neighbor_histogram=neighbor_counts,
                )
            )
            scan_index += 1

    return components


def _component_mask_from_seed(frame: np.ndarray, *, seed_y: int, seed_x: int, class_id: int) -> np.ndarray:
    height, width = [int(value) for value in frame.shape]
    if seed_y < 0 or seed_y >= height or seed_x < 0 or seed_x >= width:
        raise AlphaPrimitivePlanError("component seed is outside frame")
    if int(frame[seed_y, seed_x]) != int(class_id):
        raise AlphaPrimitivePlanError("component seed class no longer matches baseline masks")
    selected = np.zeros((height, width), dtype=np.bool_)
    stack = [(int(seed_y), int(seed_x))]
    selected[seed_y, seed_x] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in NEIGHBORS_4:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if selected[ny, nx] or int(frame[ny, nx]) != int(class_id):
                continue
            selected[ny, nx] = True
            stack.append((ny, nx))
    return selected


def _adjacent_to(frame: np.ndarray, class_id: int, *, width: int) -> np.ndarray:
    frontier = frame == int(class_id)
    reached = frontier.copy()
    for _ in range(width):
        expanded = frontier.copy()
        expanded[1:, :] |= frontier[:-1, :]
        expanded[:-1, :] |= frontier[1:, :]
        expanded[:, 1:] |= frontier[:, :-1]
        expanded[:, :-1] |= frontier[:, 1:]
        frontier = expanded & ~reached
        reached |= expanded
    return reached


def _boundary_pair_count(frame: np.ndarray, *, source_class: int, target_class: int, width: int) -> int:
    selected = (frame == int(source_class)) & _adjacent_to(frame, int(target_class), width=width)
    return int(np.count_nonzero(selected))


def _boundary_pair_counts(frame: np.ndarray, *, width: int) -> list[tuple[int, int, int]]:
    counts: list[tuple[int, int, int]] = []
    for source_class in CLASS_IDS:
        for target_class in CLASS_IDS:
            if source_class == target_class:
                continue
            count = _boundary_pair_count(
                frame,
                source_class=int(source_class),
                target_class=int(target_class),
                width=width,
            )
            if count:
                counts.append((int(count), int(source_class), int(target_class)))
    return counts


def _dominant_neighbor_for_class(frame: np.ndarray, class_id: int, *, width: int) -> int:
    pairs = _boundary_pair_counts(frame, width=width)
    filtered = [(count, target) for count, source, target in pairs if source == int(class_id)]
    if not filtered:
        return _fallback_target_class(class_id)
    _count, target = max(filtered, key=lambda item: (item[0], -item[1]))
    return int(target)


def _cap_selection(selection: np.ndarray, max_pixels: int) -> tuple[np.ndarray, int, bool]:
    if selection.dtype != np.bool_:
        selection = selection.astype(np.bool_)
    total = int(np.count_nonzero(selection))
    if total <= max_pixels:
        return selection, total, False
    ys, xs = np.nonzero(selection)
    capped = np.zeros_like(selection, dtype=np.bool_)
    capped[ys[:max_pixels], xs[:max_pixels]] = True
    return capped, int(max_pixels), True


def _selection_weight(selected_pixels: int, frame_pixels: int) -> float:
    if frame_pixels <= 0:
        return 0.0
    return _finite_weight(min(1.0, selected_pixels / frame_pixels), field="selection_weight")


def _mutation_record(
    *,
    kind: str,
    operation: str,
    frame_index: int | None,
    source_class: int | None,
    target_class: int | None,
    rank_pixels: int,
    frame_pixels: int,
    params: Mapping[str, Any],
    description: str,
) -> PrimitiveMutation:
    base = {
        "kind": kind,
        "operation": operation,
        "frame_index": frame_index,
        "source_class": source_class,
        "target_class": target_class,
        "params": dict(params),
    }
    primitive_hash = _canonical_hash_any(base)[:16]
    readable = "_".join(
        str(value)
        for value in (kind, operation, frame_index, source_class, target_class)
        if value is not None
    )
    primitive_id = f"{_safe_label(readable)}_{primitive_hash}"
    return PrimitiveMutation(
        primitive_id=primitive_id,
        kind=kind,
        operation=operation,
        frame_index=frame_index,
        source_class=source_class,
        target_class=target_class,
        rank_pixels=int(rank_pixels),
        selection_weight=_selection_weight(int(rank_pixels), frame_pixels),
        params=dict(params),
        description=description,
    )


def _component_mutations(masks: np.ndarray, frame_indices: list[int], config: PlanConfig) -> list[PrimitiveMutation]:
    _, height, width = masks.shape
    frame_pixels = int(height * width)
    candidates: list[tuple[tuple[int, int, int, int, int], PrimitiveMutation]] = []
    for frame_index in frame_indices:
        frame = masks[frame_index]
        components = _frame_components(frame, frame_index=frame_index)
        for class_id in CLASS_IDS:
            class_components = [component for component in components if component.class_id == int(class_id)]
            for component in sorted(class_components, key=_component_sort_key)[: config.max_components_per_class]:
                target = _dominant_neighbor_from_hist(component.neighbor_histogram, component.class_id)
                params = {
                    "seed_y": int(component.seed_y),
                    "seed_x": int(component.seed_x),
                    "scan_index": int(component.scan_index),
                    "bbox_xyxy_exclusive": [int(v) for v in component.bbox_xyxy_exclusive],
                    "component_area": int(component.area),
                    "component_centroid_xy": [float(v) for v in component.centroid_xy],
                    "boundary_pixels_4conn": int(component.boundary_pixels_4conn),
                    "boundary_edges_4conn": int(component.boundary_edges_4conn),
                    "neighbor_histogram": dict(component.neighbor_histogram),
                    "pixel_cap_policy": "row_major_within_component_after_exact_component_selection",
                }
                mutation = _mutation_record(
                    kind="connected_component",
                    operation="set_component_to_neighbor_class",
                    frame_index=frame_index,
                    source_class=component.class_id,
                    target_class=target,
                    rank_pixels=component.area,
                    frame_pixels=frame_pixels,
                    params=params,
                    description="flip one 4-connected component to its dominant neighboring class",
                )
                candidates.append(
                    (
                        (
                            -int(component.area),
                            -int(component.boundary_edges_4conn),
                            int(frame_index),
                            int(component.class_id),
                            int(component.scan_index),
                        ),
                        mutation,
                    )
                )
    return [item[1] for item in sorted(candidates, key=lambda item: item[0])[: config.max_component_points]]


def _boundary_mutations(masks: np.ndarray, frame_indices: list[int], config: PlanConfig) -> list[PrimitiveMutation]:
    _, height, width = masks.shape
    frame_pixels = int(height * width)
    candidates: list[tuple[tuple[int, int, int, int], PrimitiveMutation]] = []
    for frame_index in frame_indices:
        frame = masks[frame_index]
        for count, source_class, target_class in _boundary_pair_counts(frame, width=config.boundary_width):
            mutation = _mutation_record(
                kind="boundary_band",
                operation="set_source_boundary_band_to_target_class",
                frame_index=frame_index,
                source_class=source_class,
                target_class=target_class,
                rank_pixels=count,
                frame_pixels=frame_pixels,
                params={
                    "boundary_width": int(config.boundary_width),
                    "boundary_pair_pixels": int(count),
                    "pixel_cap_policy": "row_major_within_boundary_band",
                },
                description="flip a one-sided class boundary band to the adjacent target class",
            )
            candidates.append(((-int(count), int(frame_index), int(source_class), int(target_class)), mutation))
    return [item[1] for item in sorted(candidates, key=lambda item: item[0])[: config.max_boundary_points]]


def _class_flip_mutations(masks: np.ndarray, frame_indices: list[int], config: PlanConfig) -> list[PrimitiveMutation]:
    _, height, width = masks.shape
    frame_pixels = int(height * width)
    candidates: list[tuple[tuple[int, int, int], PrimitiveMutation]] = []
    for frame_index in frame_indices:
        frame = masks[frame_index]
        counts = np.bincount(frame.reshape(-1), minlength=max(CLASS_IDS) + 1)
        for source_class in CLASS_IDS:
            count = int(counts[source_class])
            if count == 0:
                continue
            target_class = _dominant_neighbor_for_class(frame, int(source_class), width=config.boundary_width)
            mutation = _mutation_record(
                kind="class_flip",
                operation="set_source_class_prefix_to_target_class",
                frame_index=frame_index,
                source_class=int(source_class),
                target_class=target_class,
                rank_pixels=count,
                frame_pixels=frame_pixels,
                params={
                    "source_class_pixels_in_frame": count,
                    "pixel_cap_policy": "row_major_prefix_of_source_class",
                },
                description="flip a bounded row-major prefix of one frame/class to its dominant neighboring class",
            )
            candidates.append(((-count, int(frame_index), int(source_class)), mutation))
    return [item[1] for item in sorted(candidates, key=lambda item: item[0])[: config.max_class_flip_points]]


def _morph_mutations(masks: np.ndarray, frame_indices: list[int], config: PlanConfig) -> list[PrimitiveMutation]:
    _, height, width = masks.shape
    frame_pixels = int(height * width)
    erode: list[tuple[tuple[int, int, int, int], PrimitiveMutation]] = []
    dilate: list[tuple[tuple[int, int, int, int], PrimitiveMutation]] = []
    for frame_index in frame_indices:
        frame = masks[frame_index]
        for count, source_class, target_class in _boundary_pair_counts(frame, width=config.boundary_width):
            erode_mutation = _mutation_record(
                kind="morphology_erode",
                operation="erode_source_class_toward_target_class",
                frame_index=frame_index,
                source_class=source_class,
                target_class=target_class,
                rank_pixels=count,
                frame_pixels=frame_pixels,
                params={
                    "boundary_width": int(config.boundary_width),
                    "boundary_pair_pixels": int(count),
                    "pixel_cap_policy": "row_major_within_source_boundary",
                },
                description="erode source-class boundary pixels into an adjacent target class",
            )
            erode.append(((-int(count), int(frame_index), int(source_class), int(target_class)), erode_mutation))

            dilate_count = _boundary_pair_count(
                frame,
                source_class=target_class,
                target_class=source_class,
                width=config.boundary_width,
            )
            if dilate_count:
                dilate_mutation = _mutation_record(
                    kind="morphology_dilate",
                    operation="dilate_source_class_into_target_class",
                    frame_index=frame_index,
                    source_class=source_class,
                    target_class=target_class,
                    rank_pixels=dilate_count,
                    frame_pixels=frame_pixels,
                    params={
                        "boundary_width": int(config.boundary_width),
                        "target_pixels_adjacent_to_source": int(dilate_count),
                        "pixel_cap_policy": "row_major_within_adjacent_target_pixels",
                    },
                    description="dilate source class into adjacent target-class pixels",
                )
                dilate.append(
                    ((-int(dilate_count), int(frame_index), int(source_class), int(target_class)), dilate_mutation)
                )
    half = max(1, config.max_morph_points // 2)
    selected = [item[1] for item in sorted(erode, key=lambda item: item[0])[:half]]
    selected.extend(
        item[1] for item in sorted(dilate, key=lambda item: item[0])[: max(1, config.max_morph_points - len(selected))]
    )
    return selected[: config.max_morph_points]


def _transition_mutations(masks: np.ndarray, config: PlanConfig) -> list[PrimitiveMutation]:
    frame_count, height, width = masks.shape
    if frame_count <= 1:
        return []
    frame_pixels = int(height * width)
    candidates: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for frame_index in range(1, frame_count):
        previous = masks[frame_index - 1]
        current = masks[frame_index]
        changed = previous != current
        if not np.any(changed):
            continue
        encoded = previous[changed].astype(np.int64) * (max(CLASS_IDS) + 1) + current[changed].astype(np.int64)
        counts = np.bincount(encoded, minlength=(max(CLASS_IDS) + 1) ** 2)
        for src in CLASS_IDS:
            for dst in CLASS_IDS:
                if src == dst:
                    continue
                count = int(counts[int(src) * (max(CLASS_IDS) + 1) + int(dst)])
                if count:
                    candidates.append(((-count, int(frame_index), int(src), int(dst)), (frame_index, src, dst, count)))

    selected_pairs = [item[1] for item in sorted(candidates, key=lambda item: item[0])[: config.max_transition_points]]
    mutations: list[PrimitiveMutation] = []
    for frame_index, source_class, target_class, count in selected_pairs:
        mutations.append(
            _mutation_record(
                kind="transition_endpoint",
                operation="hold_current_endpoint_at_previous_class",
                frame_index=int(frame_index),
                source_class=int(source_class),
                target_class=int(target_class),
                rank_pixels=int(count),
                frame_pixels=frame_pixels,
                params={
                    "from_frame": int(frame_index - 1),
                    "to_frame": int(frame_index),
                    "previous_class": int(source_class),
                    "current_class": int(target_class),
                    "changed_pixels_for_pair": int(count),
                    "pixel_cap_policy": "row_major_within_transition_pair",
                },
                description="set changed pixels in the later endpoint back to their previous-frame class",
            )
        )
        mutations.append(
            _mutation_record(
                kind="transition_endpoint",
                operation="advance_previous_endpoint_to_current_class",
                frame_index=int(frame_index - 1),
                source_class=int(source_class),
                target_class=int(target_class),
                rank_pixels=int(count),
                frame_pixels=frame_pixels,
                params={
                    "from_frame": int(frame_index - 1),
                    "to_frame": int(frame_index),
                    "previous_class": int(source_class),
                    "current_class": int(target_class),
                    "changed_pixels_for_pair": int(count),
                    "pixel_cap_policy": "row_major_within_transition_pair",
                },
                description="set changed pixels in the earlier endpoint forward to their next-frame class",
            )
        )
    return mutations[: max(1, config.max_transition_points * 2)]


def _dedupe_and_limit_mutations(mutations: list[PrimitiveMutation], config: PlanConfig) -> list[PrimitiveMutation]:
    priority = {
        "connected_component": 10,
        "boundary_band": 20,
        "class_flip": 30,
        "morphology_erode": 40,
        "morphology_dilate": 50,
        "transition_endpoint": 60,
    }
    seen: set[str] = set()
    ordered: list[PrimitiveMutation] = []
    for mutation in sorted(
        mutations,
        key=lambda item: (
            priority.get(item.kind, 99),
            -int(item.rank_pixels),
            int(item.frame_index if item.frame_index is not None else -1),
            int(item.source_class if item.source_class is not None else -1),
            int(item.target_class if item.target_class is not None else -1),
            item.operation,
            item.primitive_id,
        ),
    ):
        if mutation.primitive_id in seen:
            continue
        seen.add(mutation.primitive_id)
        ordered.append(mutation)
        if len(ordered) >= config.max_points:
            break
    return ordered


def build_primitives(masks: np.ndarray, config: PlanConfig) -> list[PrimitiveMutation]:
    masks = _validate_masks(masks)
    frame_indices = _sample_frame_indices(int(masks.shape[0]), config.scan_frame_count)
    mutations: list[PrimitiveMutation] = []
    mutations.extend(_component_mutations(masks, frame_indices, config))
    mutations.extend(_boundary_mutations(masks, frame_indices, config))
    mutations.extend(_class_flip_mutations(masks, frame_indices, config))
    mutations.extend(_morph_mutations(masks, frame_indices, config))
    mutations.extend(_transition_mutations(masks, config))
    return _dedupe_and_limit_mutations(mutations, config)


def _selection_for_mutation(baseline: np.ndarray, mutation: PrimitiveMutation, config: PlanConfig) -> tuple[int, np.ndarray]:
    if mutation.frame_index is None:
        raise AlphaPrimitivePlanError(f"{mutation.primitive_id}: mutation is missing frame_index")
    frame_index = int(mutation.frame_index)
    frame = baseline[frame_index]
    source_class = int(mutation.source_class) if mutation.source_class is not None else None
    target_class = int(mutation.target_class) if mutation.target_class is not None else None
    params = dict(mutation.params)

    if mutation.kind == "connected_component":
        if source_class is None:
            raise AlphaPrimitivePlanError("connected component mutation missing source_class")
        selection = _component_mask_from_seed(
            frame,
            seed_y=int(params["seed_y"]),
            seed_x=int(params["seed_x"]),
            class_id=source_class,
        )
    elif mutation.kind in {"boundary_band", "morphology_erode"}:
        if source_class is None or target_class is None:
            raise AlphaPrimitivePlanError(f"{mutation.kind} mutation missing class ids")
        selection = (frame == source_class) & _adjacent_to(frame, target_class, width=int(params["boundary_width"]))
    elif mutation.kind == "class_flip":
        if source_class is None:
            raise AlphaPrimitivePlanError("class flip mutation missing source_class")
        selection = frame == source_class
    elif mutation.kind == "morphology_dilate":
        if source_class is None or target_class is None:
            raise AlphaPrimitivePlanError("dilate mutation missing class ids")
        selection = (frame == target_class) & _adjacent_to(frame, source_class, width=int(params["boundary_width"]))
    elif mutation.kind == "transition_endpoint":
        from_frame = int(params["from_frame"])
        to_frame = int(params["to_frame"])
        previous_class = int(params["previous_class"])
        current_class = int(params["current_class"])
        previous = baseline[from_frame]
        current = baseline[to_frame]
        selection = (previous == previous_class) & (current == current_class)
    else:
        raise AlphaPrimitivePlanError(f"unsupported primitive kind: {mutation.kind!r}")

    selected, selected_count, capped = _cap_selection(selection, config.max_pixels_per_point)
    if selected_count <= 0:
        raise AlphaPrimitivePlanError(f"{mutation.primitive_id}: primitive selected zero pixels")
    _ = capped
    return frame_index, selected


def apply_mutation(baseline: np.ndarray, mutation: PrimitiveMutation, config: PlanConfig) -> tuple[np.ndarray, dict[str, Any]]:
    baseline = _validate_masks(baseline)
    mutated = baseline.copy()
    frame_index, selection = _selection_for_mutation(baseline, mutation, config)
    params = dict(mutation.params)
    target_class = int(mutation.target_class) if mutation.target_class is not None else None
    selected_before_cap = int(mutation.rank_pixels)
    selected_after_cap = int(np.count_nonzero(selection))
    capped = bool(selected_after_cap < selected_before_cap)

    if mutation.kind == "transition_endpoint":
        if mutation.operation == "hold_current_endpoint_at_previous_class":
            target_value = int(params["previous_class"])
        elif mutation.operation == "advance_previous_endpoint_to_current_class":
            target_value = int(params["current_class"])
        else:
            raise AlphaPrimitivePlanError(f"unsupported transition operation: {mutation.operation!r}")
    elif mutation.kind == "morphology_dilate":
        if mutation.source_class is None:
            raise AlphaPrimitivePlanError(f"{mutation.primitive_id}: dilation has no source class")
        target_value = int(mutation.source_class)
    elif target_class is not None:
        target_value = target_class
    else:
        raise AlphaPrimitivePlanError(f"{mutation.primitive_id}: mutation has no target class")

    before_values = mutated[frame_index][selection].copy()
    mutated[frame_index][selection] = np.uint8(target_value)
    changed = mutated != baseline
    changed_pixels = int(np.count_nonzero(changed))
    if changed_pixels <= 0:
        raise AlphaPrimitivePlanError(f"{mutation.primitive_id}: mutation produced no mask delta")

    source_counts = np.bincount(before_values.astype(np.int64), minlength=max(CLASS_IDS) + 1)
    return mutated, {
        "changed_pixels": changed_pixels,
        "changed_fraction": _round_float(changed_pixels / int(baseline.size)),
        "selected_pixels_before_cap": int(selected_before_cap),
        "selected_pixels_after_cap": int(selected_after_cap),
        "selection_was_capped": capped,
        "target_class_written": int(target_value),
        "source_class_histogram_in_selection": {
            str(class_id): int(source_counts[class_id]) for class_id in CLASS_IDS
        },
        "pixel_selection_policy": params.get("pixel_cap_policy"),
    }


def _encode_masks_mkv(masks: np.ndarray, output_path: Path, *, crf: int, fps: int) -> dict[str, Any]:
    from tac.mask_codec import encode_masks

    tensor = torch.from_numpy(_validate_masks(masks).astype(np.int64, copy=False))
    size_bytes = int(encode_masks(tensor, output_path, crf=crf, fps=fps))
    return {
        "path": output_path,
        "size_bytes": size_bytes,
        "sha256": _sha256_file(output_path),
        "encoder": "tac.mask_codec.encode_masks",
        "crf": int(crf),
        "fps": int(fps),
    }


def _primitive_payload(mutation: PrimitiveMutation) -> dict[str, Any]:
    return {
        "primitive_id": mutation.primitive_id,
        "kind": mutation.kind,
        "operation": mutation.operation,
        "frame_index": mutation.frame_index,
        "source_class": mutation.source_class,
        "target_class": mutation.target_class,
        "rank_pixels": int(mutation.rank_pixels),
        "selection_weight": _finite_weight(mutation.selection_weight, field="primitive.selection_weight"),
        "description": mutation.description,
        "params": dict(mutation.params),
    }


def _validate_plan_non_promotable(plan: Mapping[str, Any]) -> None:
    if plan.get("score_claim") is not False:
        raise AssertionError("Alpha primitive plan must not claim score")
    if plan.get("promotion_eligible") is not False:
        raise AssertionError("Alpha primitive plan must not be promotion eligible")
    if plan.get("official_component_response") is not False:
        raise AssertionError("Alpha primitive plan is not an official component response result")
    if plan.get("scorer_network_loaded") is not False:
        raise AssertionError("Alpha primitive plan must not load scorer networks")
    if "contest_auth_eval.py --device cuda" not in str(plan.get("canonical_score_source_required", "")):
        raise AssertionError("plan must preserve exact CUDA auth eval score source")
    for index, point in enumerate(plan.get("points", [])):
        if not isinstance(point, Mapping):
            raise AssertionError(f"points[{index}] must be an object")
        if point.get("score_claim") is not False:
            raise AssertionError(f"points[{index}] must not claim score")
        if point.get("promotion_eligible") is not False:
            raise AssertionError(f"points[{index}] must not be promotion eligible")
        archive = point.get("archive")
        if archive is not None:
            _validate_plan_relative_path(str(archive), field=f"points[{index}].archive")
        if "selection_weight" in point:
            _finite_weight(point["selection_weight"], field=f"points[{index}].selection_weight")
        primitive = point.get("primitive")
        if isinstance(primitive, Mapping) and "selection_weight" in primitive:
            _finite_weight(primitive["selection_weight"], field=f"points[{index}].primitive.selection_weight")


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        output_dir / "alpha_mask_primitive_response_plan.json",
        output_dir / "alpha_mask_primitive_archive_variants_manifest.json",
    ]
    existing = [path.name for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(f"{output_dir} already contains {existing}; use --force to overwrite")


def build_alpha_mask_primitive_response_plan(
    *,
    baseline_archive: Path,
    output_dir: Path,
    mask_member: str = "masks.mkv",
    decoded_masks_source: Path | None = None,
    candidate_manifest: Path | None = None,
    baseline_contest_auth_eval_json: Path | None = None,
    config: PlanConfig = PlanConfig(),
    command: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    _validate_config(config)
    output_dir = output_dir.resolve()
    _prepare_output_dir(output_dir, force=force)

    source_extra: dict[str, Any] = {"source_kind": "baseline_archive_mask_member"}
    if candidate_manifest is not None:
        baseline_archive, mask_member, source_extra = _source_from_candidate_manifest(candidate_manifest.resolve())
    baseline_archive = baseline_archive.resolve()
    mask_member = _validate_member_name(mask_member)
    members = _read_custody_checked_archive(baseline_archive)
    if mask_member not in members:
        raise AlphaPrimitivePlanError(f"baseline archive missing mask member {mask_member!r}")

    if decoded_masks_source is not None:
        masks, decoded_source_meta = _load_decoded_masks_source(decoded_masks_source.resolve())
        decode_meta = {
            "decoder": "provided_decoded_masks_source",
            "decoded_frames": int(masks.shape[0]),
            "max_frames": None,
            "truncated_by_builder": False,
            "decoded_source": decoded_source_meta,
        }
        source_extra = {
            **source_extra,
            "source_kind": "decoded_masks_source_with_baseline_archive",
            "decoded_source": decoded_source_meta,
        }
    else:
        masks, decode_meta = _decode_masks_from_archive_member(
            members=members,
            mask_member=mask_member,
            max_frames=config.max_frames,
        )

    masks = _validate_masks(masks)
    primitives = build_primitives(masks, config)
    if not primitives:
        raise AlphaPrimitivePlanError("no Alpha primitive perturbations were generated")

    plan_output = output_dir / "alpha_mask_primitive_response_plan.json"
    variants_manifest_output = output_dir / "alpha_mask_primitive_archive_variants_manifest.json"
    archives_dir = output_dir / "archives"
    mask_members_dir = output_dir / "mask_members"
    archives_dir.mkdir(parents=True, exist_ok=True)
    mask_members_dir.mkdir(parents=True, exist_ok=True)

    baseline_meta = _file_meta(baseline_archive)
    member_order = sorted(members)
    source_members = [dict(members[name].source_info) for name in member_order]
    plan_points: list[dict[str, Any]] = [
        {
            "index": 0,
            "epsilon": 0.0,
            "epsilon_semantics": "ordinal_alpha_primitive_point_index_not_metric_magnitude",
            "role": "baseline",
            "archive_bytes": int(baseline_meta["bytes"]),
            "archive_sha256": str(baseline_meta["sha256"]),
            "predicted_delta": {"combined": 0.0, "posenet": 0.0, "segnet": 0.0},
            "score_claim": False,
            "promotion_eligible": False,
            "official_component_response": False,
        }
    ]
    variant_records: list[dict[str, Any]] = []

    for index, mutation in enumerate(primitives, start=1):
        mutated_masks, mask_delta = apply_mutation(masks, mutation, config)
        label = _safe_label(f"point_{index:03d}_{mutation.kind}_{mutation.operation}_{mutation.primitive_id}")
        mask_path = mask_members_dir / f"{label}.mkv"
        encoded = _encode_masks_mkv(mutated_masks, mask_path, crf=config.mask_crf, fps=config.mask_fps)
        mask_bytes = mask_path.read_bytes()
        point_members = {name: member.data for name, member in members.items()}
        point_members[mask_member] = mask_bytes
        archive_data = _archive_bytes(point_members, member_order=member_order)
        archive_rebuild = _archive_bytes(point_members, member_order=member_order)
        if archive_data != archive_rebuild:
            raise AlphaPrimitivePlanError(f"deterministic archive rebuild mismatch for {mutation.primitive_id}")
        archive_path = archives_dir / f"{label}.zip"
        archive_path.write_bytes(archive_data)
        archive_meta = _file_meta(archive_path)
        archive_rel = _path_for_plan(archive_path, root=plan_output.parent)
        mask_rel = _path_for_plan(mask_path, root=variants_manifest_output.parent)
        primitive_payload = _primitive_payload(mutation)

        point = {
            "index": int(index),
            "epsilon": float(index),
            "epsilon_semantics": "ordinal_alpha_primitive_point_index_not_metric_magnitude",
            "archive": archive_rel,
            "archive_bytes": int(archive_meta["bytes"]),
            "archive_sha256": str(archive_meta["sha256"]),
            "archive_byte_delta_vs_baseline": int(archive_meta["bytes"]) - int(baseline_meta["bytes"]),
            "primitive_id": mutation.primitive_id,
            "primitive": primitive_payload,
            "selection_weight": primitive_payload["selection_weight"],
            "mask_delta": mask_delta,
            "mask_member": {
                "name": mask_member,
                "size_bytes": int(encoded["size_bytes"]),
                "sha256": str(encoded["sha256"]),
                "encoder": encoded["encoder"],
                "crf": int(encoded["crf"]),
                "fps": int(encoded["fps"]),
            },
            "predicted_delta": None,
            "score_claim": False,
            "promotion_eligible": False,
            "official_component_response": False,
        }
        plan_points.append(point)
        variant_records.append(
            {
                "index": int(index),
                "epsilon": float(index),
                "archive": _file_meta(archive_path, root=variants_manifest_output.parent),
                "archive_byte_delta_vs_baseline": point["archive_byte_delta_vs_baseline"],
                "deterministic_rebuild": True,
                "primitive": primitive_payload,
                "mask_delta": mask_delta,
                "mask_member_artifact": {
                    "path": mask_rel,
                    "name": mask_member,
                    "size_bytes": int(encoded["size_bytes"]),
                    "sha256": str(encoded["sha256"]),
                    "encoder": encoded["encoder"],
                    "crf": int(encoded["crf"]),
                    "fps": int(encoded["fps"]),
                },
                "members": _archive_manifest_from_bytes(archive_data),
                "score_claim": False,
                "promotion_eligible": False,
            }
        )

    source = {
        **source_extra,
        "baseline_archive": {
            "path_hint": _repo_relative_hint(baseline_archive),
            "bytes": int(baseline_meta["bytes"]),
            "sha256": str(baseline_meta["sha256"]),
        },
        "mask_member": dict(members[mask_member].source_info),
        "source_members": source_members,
        "decoded_masks": _mask_stats(masks),
        "decode": decode_meta,
        "validated_zip_safety": True,
    }
    partial_decode = bool(config.max_frames is not None)
    variants_manifest = {
        "schema_version": 1,
        "format": VARIANT_MANIFEST_FORMAT,
        "producer": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "official_component_response": False,
        "evidence_grade": EVIDENCE_GRADE,
        "baseline_archive": {
            "path_hint": _repo_relative_hint(baseline_archive),
            "bytes": int(baseline_meta["bytes"]),
            "sha256": str(baseline_meta["sha256"]),
        },
        "mask_member": dict(members[mask_member].source_info),
        "member_order": member_order,
        "points": variant_records,
        "canonical_response_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
        "auth_eval_required": "cuda",
    }
    _write_json(variants_manifest_output, variants_manifest)

    plan: dict[str, Any] = {
        "schema_version": 1,
        "format": PLAN_FORMAT,
        "alpha_plan_format": ALPHA_PLAN_FORMAT,
        "producer": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "official_component_response": False,
        "evidence_grade": EVIDENCE_GRADE,
        "local_diagnostic_only": True,
        "scorer_network_loaded": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "purpose": (
            "build deterministic Alpha mask primitive archive variants for later "
            "SegNet/PoseNet CUDA component-response exact eval"
        ),
        "path_policy": {
            "plan_point_paths": "relative_to_plan_json_parent",
            "absolute_point_paths_allowed": False,
            "traversal_point_paths_allowed": False,
            "hidden_or_system_paths_allowed": False,
        },
        "baseline_archive": {
            "path_hint": _repo_relative_hint(baseline_archive),
            "bytes": int(baseline_meta["bytes"]),
            "sha256": str(baseline_meta["sha256"]),
        },
        "baseline_archive_required_at_eval_time": True,
        "source": source,
        "generation_config": dataclasses.asdict(config),
        "perturbation": {
            "format": ALPHA_PLAN_FORMAT,
            "basis_kind": "alpha_mask_geometry_primitive",
            "epsilon_units": "ordinal_primitive_point_index",
            "archive_variants_manifest": _path_for_plan(variants_manifest_output, root=plan_output.parent),
            "archive_variants_manifest_sha256": _sha256_file(variants_manifest_output),
            "primitive_count": len(primitives),
            "primitive_ids": [mutation.primitive_id for mutation in primitives],
            "primitive_generation": {
                "frame_sampling": "deterministic_linspace_over_decoded_frames",
                "class_ids": [int(value) for value in CLASS_IDS],
                "connectivity": 4,
                "selection_cap_pixels_per_point": int(config.max_pixels_per_point),
                "partial_decode": partial_decode,
                "partial_decode_non_promotable": partial_decode,
            },
            "canonical_response_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
            "auth_eval_required": "cuda",
        },
        "points": plan_points,
        "provenance": _provenance(command),
    }
    if baseline_contest_auth_eval_json is not None:
        baseline_contest_auth_eval_json = baseline_contest_auth_eval_json.resolve()
        if not baseline_contest_auth_eval_json.is_file():
            raise AlphaPrimitivePlanError(
                f"baseline contest auth eval JSON not found: {baseline_contest_auth_eval_json}"
            )
        plan["baseline_contest_auth_eval_json"] = _path_for_plan(
            baseline_contest_auth_eval_json,
            root=plan_output.parent,
        )
        plan["baseline_contest_auth_eval"] = _file_meta(
            baseline_contest_auth_eval_json,
            root=plan_output.parent,
        )

    _validate_plan_non_promotable(plan)
    _write_json(plan_output, plan)
    summary = {
        "schema_version": 1,
        "format": "alpha_mask_primitive_component_response_plan_summary_v1",
        "producer": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "official_component_response": False,
        "plan": _file_meta(plan_output),
        "archive_variants_manifest": _file_meta(variants_manifest_output),
        "baseline_archive": baseline_meta,
        "point_count": len(plan_points),
        "nonzero_point_count": len(plan_points) - 1,
        "primitive_count": len(primitives),
        "epsilon_ladder": [float(point["epsilon"]) for point in plan_points],
        "canonical_response_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
        "auth_eval_required": "cuda",
    }
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--candidate-manifest", type=Path, default=None)
    source.add_argument("--decoded-masks-source", type=Path, default=None)
    parser.add_argument("--baseline-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--mask-member", default="masks.mkv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-contest-auth-eval-json", type=Path, default=None)
    parser.add_argument("--scan-frame-count", type=int, default=PlanConfig.scan_frame_count)
    parser.add_argument("--max-points", type=int, default=PlanConfig.max_points)
    parser.add_argument("--max-component-points", type=int, default=PlanConfig.max_component_points)
    parser.add_argument("--max-boundary-points", type=int, default=PlanConfig.max_boundary_points)
    parser.add_argument("--max-class-flip-points", type=int, default=PlanConfig.max_class_flip_points)
    parser.add_argument("--max-morph-points", type=int, default=PlanConfig.max_morph_points)
    parser.add_argument("--max-transition-points", type=int, default=PlanConfig.max_transition_points)
    parser.add_argument("--max-components-per-class", type=int, default=PlanConfig.max_components_per_class)
    parser.add_argument("--max-pixels-per-point", type=int, default=PlanConfig.max_pixels_per_point)
    parser.add_argument("--boundary-width", type=int, default=PlanConfig.boundary_width)
    parser.add_argument("--mask-crf", type=int, default=PlanConfig.mask_crf)
    parser.add_argument("--mask-fps", type=int, default=PlanConfig.mask_fps)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Diagnostic decode limit. Plans with this set remain non-promotable partial-mask probes.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing plan outputs in output-dir.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = PlanConfig(
        scan_frame_count=args.scan_frame_count,
        max_points=args.max_points,
        max_component_points=args.max_component_points,
        max_boundary_points=args.max_boundary_points,
        max_class_flip_points=args.max_class_flip_points,
        max_morph_points=args.max_morph_points,
        max_transition_points=args.max_transition_points,
        max_components_per_class=args.max_components_per_class,
        max_pixels_per_point=args.max_pixels_per_point,
        boundary_width=args.boundary_width,
        mask_crf=args.mask_crf,
        mask_fps=args.mask_fps,
        max_frames=args.max_frames,
    )
    try:
        summary = build_alpha_mask_primitive_response_plan(
            baseline_archive=args.baseline_archive,
            output_dir=args.output_dir,
            mask_member=args.mask_member,
            decoded_masks_source=args.decoded_masks_source,
            candidate_manifest=args.candidate_manifest,
            baseline_contest_auth_eval_json=args.baseline_contest_auth_eval_json,
            config=config,
            command=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
            force=args.force,
        )
    except (AlphaPrimitivePlanError, FileExistsError, FileNotFoundError, ValueError, TypeError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
