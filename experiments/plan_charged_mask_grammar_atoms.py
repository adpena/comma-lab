#!/usr/bin/env python3
"""Plan charged mask-grammar atoms from a public-floor one-blob archive.

This is a deterministic, CPU-only planning tool.  It extracts the charged
mask stream from an existing single-member public-floor-style archive using
the repository's runtime payload unpacker, then emits bounded atom tables for
future strict mask grammar work.  It does not build a scoreable archive, does
not run CUDA, and always writes ``score_claim=false``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
UNPACKER_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"
INFLATE_RENDERER_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"
SCHEMA = "charged_mask_grammar_atom_plan_v1"
TOOL = "experiments/plan_charged_mask_grammar_atoms.py"
EVIDENCE_GRADE = "empirical_planning_only_non_score"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "charged_mask_grammar_atoms_20260502_codex"
DEFAULT_MASK_MEMBER_CANDIDATES = (
    "masks.mkv",
    "grayscale.mkv",
    "masks.alpha4.mkv",
    "masks.amrc",
    "masks.nrv",
)
DEFAULT_FRAME_COUNT = 600
DEFAULT_DECODE_EXPECTED_FRAMES = 1200
DEFAULT_HEIGHT = 384
DEFAULT_WIDTH = 512
DEFAULT_CLASS_COUNT = 5
DEFAULT_FRONTIER_ARCHIVE_BYTES = 276_223
VANISHING_POINT = (256, 174)
HORIZON_BAND = (155, 195)
STREAM_CHUNK_BYTES = 4096
ALLOCATION_SCHEMA = "charged_mask_grammar_trace_weighted_allocation_v1"
DEFAULT_ALLOCATION_BYTE_BUDGETS = (1024, 2048, 4096)
DEFAULT_TRACE_TOP_PAIRS = 32
TRACE_PAIR_WEIGHT_SCALE = 5.0
TRACE_POSE_WEIGHT_SCALE = 2.0
ATOM_SIDE_INFO_BYTES = 6
GRAMMAR_HEADER_SIDE_INFO_BYTES = 32
GRAMMAR_CODEBOOK_SIDE_INFO_BYTES = 48
RESIDUAL_MODEL_SIDE_INFO_BYTES = 64


class PlannerError(ValueError):
    """Raised for unsafe or unsupported planning inputs."""


@dataclass(frozen=True)
class Shape:
    frames: int
    height: int
    width: int
    class_count: int

    def as_json(self) -> dict[str, int]:
        return {
            "frames": int(self.frames),
            "height": int(self.height),
            "width": int(self.width),
            "class_count": int(self.class_count),
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _artifact_ref(path: Path, *, base: Path) -> str:
    path = path.resolve()
    base = base.resolve()
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _safe_member_name(name: str) -> str:
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        raise PlannerError(f"unsafe archive member path: {name!r}")
    member_path = PurePosixPath(name)
    parts = member_path.parts
    if member_path.is_absolute() or ".." in parts or any(part in {"", "."} for part in parts):
        raise PlannerError(f"unsafe archive member path: {name!r}")
    if "__MACOSX" in parts or any(part.startswith(".") for part in parts):
        raise PlannerError(f"hidden/system archive member path: {name!r}")
    return name


def _safe_single_level_name(name: str) -> str:
    name = _safe_member_name(name)
    if len(PurePosixPath(name).parts) != 1:
        raise PlannerError(f"expected single-level archive member path: {name!r}")
    return name


def _read_single_blob_archive(archive: Path, *, member_name: str | None) -> dict[str, Any]:
    archive = archive.resolve()
    with zipfile.ZipFile(archive, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        for info in infos:
            _safe_member_name(info.filename)
        names = [info.filename for info in infos]
        if member_name is None:
            if len(names) != 1:
                raise PlannerError(f"expected one non-directory archive member; got {names!r}")
            selected = names[0]
        else:
            selected = _safe_single_level_name(member_name)
            if selected not in names:
                raise PlannerError(f"archive missing requested member {selected!r}; got {names!r}")
        info = zf.getinfo(selected)
        payload = zf.read(selected)

    return {
        "archive": {
            "path": str(archive),
            "bytes": int(archive.stat().st_size),
            "sha256": _sha256_file(archive),
        },
        "single_blob_member": {
            "name": selected,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            "zip_compress_size": int(info.compress_size),
            "zip_file_size": int(info.file_size),
            "zip_compress_type": int(info.compress_type),
        },
        "payload": payload,
    }


def _load_unpacker_module():
    spec = importlib.util.spec_from_file_location("pact_runtime_unpack_renderer_payload", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise PlannerError(f"could not load renderer payload unpacker at {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_inflate_renderer_module():
    spec = importlib.util.spec_from_file_location("pact_runtime_inflate_renderer_for_cmg_plan", INFLATE_RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise PlannerError(f"could not load inflate renderer runtime at {INFLATE_RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _try_brotli_decompress(payload: bytes) -> bytes | None:
    try:
        import brotli
    except ImportError:
        return None
    try:
        return brotli.decompress(payload)
    except brotli.error:
        return None


def _parse_renderer_payload(single_blob_payload: bytes) -> dict[str, Any]:
    unpacker = _load_unpacker_module()
    candidates: list[tuple[str, bytes]] = []
    decompressed = _try_brotli_decompress(single_blob_payload)
    if decompressed is not None:
        candidates.append(("single_brotli_member_decompressed", decompressed))
    candidates.append(("raw_or_concatenated_member_payload", single_blob_payload))

    failures: list[str] = []
    for source_kind, payload in candidates:
        try:
            header, members = unpacker._parse_payload(payload)
        except Exception as exc:  # pragma: no cover - surfaced in error message
            failures.append(f"{source_kind}: {type(exc).__name__}: {exc}")
            continue
        return {
            "source_kind": source_kind,
            "payload_bytes": len(payload),
            "payload_sha256": _sha256_bytes(payload),
            "header": header,
            "members": members,
            "helper": str(UNPACKER_PATH.relative_to(REPO_ROOT)),
        }
    raise PlannerError("unable to parse renderer payload via existing unpacker: " + "; ".join(failures))


def _member_meta_by_name(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for meta in header.get("members", []):
        if isinstance(meta, dict) and "name" in meta:
            out[str(meta["name"])] = dict(meta)
    return out


def _select_mask_member(
    members: dict[str, bytes],
    header: dict[str, Any],
    *,
    requested_member: str | None,
) -> dict[str, Any]:
    metas = _member_meta_by_name(header)
    candidates = (requested_member,) if requested_member else DEFAULT_MASK_MEMBER_CANDIDATES
    for name in candidates:
        if name and name in members:
            data = members[name]
            meta = metas.get(name, {})
            return {
                "name": name,
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
                "payload_member_meta": meta,
            }
    raise PlannerError(
        "parsed payload did not contain a supported mask member; "
        f"available={sorted(members)} requested={requested_member!r}"
    )


def _varint_len(value: int) -> int:
    if value < 0:
        raise PlannerError(f"varint value must be nonnegative: {value}")
    count = 1
    while value >= 128:
        value >>= 7
        count += 1
    return count


def _rate_cost(byte_count: int) -> float:
    return float(byte_count) * RATE_SCORE_PER_BYTE


def _atom_id(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:16]}"


def _clamp_bbox(x0: int, y0: int, x1: int, y1: int, *, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise PlannerError(f"empty bbox after clamp: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def _bbox_pixels(bbox: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _region_defs(shape: Shape) -> list[dict[str, Any]]:
    width = shape.width
    height = shape.height
    vp_x = int(round(VANISHING_POINT[0] * width / DEFAULT_WIDTH))
    vp_y = int(round(VANISHING_POINT[1] * height / DEFAULT_HEIGHT))
    horizon_top = int(round(HORIZON_BAND[0] * height / DEFAULT_HEIGHT))
    horizon_bottom = int(round(HORIZON_BAND[1] * height / DEFAULT_HEIGHT))
    specs = [
        ("full_frame", "global_template", (0, 0, width, height), 1.0),
        (
            "vanishing_point_core_r32",
            "ego_foveal_core",
            (vp_x - 32, vp_y - 32, vp_x + 32, vp_y + 32),
            3.0,
        ),
        (
            "vanishing_point_context_r80",
            "ego_foveal_context",
            (vp_x - 80, vp_y - 60, vp_x + 80, vp_y + 70),
            2.2,
        ),
        ("horizon_band", "seg_pose_boundary_band", (0, horizon_top, width, horizon_bottom), 2.7),
        (
            "ego_lane_corridor",
            "near_field_ego_motion_corridor",
            (int(width * 0.32), int(height * 0.47), int(width * 0.68), height),
            2.0,
        ),
        (
            "near_road_fovea",
            "near_field_road",
            (int(width * 0.20), int(height * 0.62), int(width * 0.80), height),
            1.7,
        ),
        ("left_periphery", "peripheral_context", (0, 0, width // 4, height), 0.8),
        ("right_periphery", "peripheral_context", (width - width // 4, 0, width, height), 0.8),
    ]
    regions: list[dict[str, Any]] = []
    for name, kind, raw_bbox, priority in specs:
        bbox = _clamp_bbox(*raw_bbox, width=width, height=height)
        identity = {
            "region_name": name,
            "region_kind": kind,
            "bbox_xyxy": [int(v) for v in bbox],
        }
        cost = 8 + sum(_varint_len(v) for v in bbox)
        regions.append(
            {
                "atom_id": _atom_id("region", identity),
                "atom_family": "ego_foveal_region",
                "identity": identity,
                "pixel_count": _bbox_pixels(bbox),
                "byte_cost_estimate": cost,
                "rate_score_cost_estimate": _rate_cost(cost),
                "priority_weight_prior": priority,
                "score_claim": False,
            }
        )
    return regions


def _frame_groups(frame_count: int, *, max_groups: int) -> list[dict[str, Any]]:
    groups: list[tuple[str, int, int]] = [("full_sequence", 0, frame_count)]
    for size in (2, 4, 8, 16, 32, 64, 128):
        if size > frame_count:
            continue
        for start in range(0, frame_count, size):
            end = min(frame_count, start + size)
            if end - start == size:
                groups.append((f"frames_{start:04d}_{end - 1:04d}", start, end))
    atoms: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for name, start, end in groups:
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        identity = {
            "group_name": name,
            "frame_start": int(start),
            "frame_end_exclusive": int(end),
            "frame_count": int(end - start),
        }
        cost = 4 + _varint_len(start) + _varint_len(end - start)
        atoms.append(
            {
                "atom_id": _atom_id("framegrp", identity),
                "atom_family": "frame_group",
                "identity": identity,
                "byte_cost_estimate": cost,
                "rate_score_cost_estimate": _rate_cost(cost),
                "score_claim": False,
            }
        )
    atoms.sort(key=lambda atom: (atom["identity"]["frame_count"], atom["identity"]["frame_start"], atom["atom_id"]))
    full = [atom for atom in atoms if atom["identity"]["group_name"] == "full_sequence"]
    rest = [atom for atom in atoms if atom["identity"]["group_name"] != "full_sequence"]
    return (full + rest)[:max_groups]


def _stream_chunk_atoms(mask_stream: bytes, *, max_chunks: int) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for offset in range(0, len(mask_stream), STREAM_CHUNK_BYTES):
        chunk = mask_stream[offset : offset + STREAM_CHUNK_BYTES]
        identity = {
            "offset": int(offset),
            "length": int(len(chunk)),
            "sha256": _sha256_bytes(chunk),
        }
        cost = len(chunk) + 2 + _varint_len(offset) + _varint_len(len(chunk))
        atoms.append(
            {
                "atom_id": _atom_id("stream", identity),
                "atom_family": "mask_stream_chunk",
                "identity": identity,
                "byte_cost_estimate": cost,
                "rate_score_cost_estimate": _rate_cost(cost),
                "exact_copy_only": True,
                "score_claim": False,
            }
        )
        if len(atoms) >= max_chunks:
            break
    return atoms


def _runtime_mask_suffix(member_name: str, mask_stream: bytes) -> str:
    suffix = PurePosixPath(member_name).suffix.lower()
    if suffix in {".mkv", ".amrc", ".stcb", ".nrv", ".cmg1"}:
        return suffix
    if mask_stream.startswith(b"AMRC"):
        return ".amrc"
    if mask_stream.startswith(b"STCB"):
        return ".stcb"
    if mask_stream.startswith(b"NRV1"):
        return ".nrv"
    if mask_stream.startswith(b"CMG1"):
        return ".cmg1"
    return ".mkv"


def _as_numpy_mask_array(decoded: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is present in repo env
        raise PlannerError("decoded mask contract requires numpy") from exc

    if hasattr(decoded, "detach") and hasattr(decoded, "cpu"):
        arr = decoded.detach().cpu().numpy()
    else:
        arr = np.asarray(decoded)
    if arr.ndim != 3:
        raise PlannerError(f"decoded mask tensor must have shape (T,H,W), got {tuple(arr.shape)}")
    if arr.size == 0:
        raise PlannerError("decoded mask tensor is empty")
    if not np.issubdtype(arr.dtype, np.integer):
        raise PlannerError(f"decoded mask tensor must contain integer class ids, got {arr.dtype}")
    min_class = int(arr.min())
    max_class = int(arr.max())
    if min_class < 0:
        raise PlannerError(f"decoded mask tensor contains negative class ids: min={min_class}")
    if max_class >= DEFAULT_CLASS_COUNT:
        raise PlannerError(
            f"decoded mask class ids outside declared CMG class range: "
            f"max={max_class}, class_count={DEFAULT_CLASS_COUNT}"
        )
    return arr.astype(np.uint8, copy=False)


def _decode_mask_stream_with_runtime(
    *,
    mask_stream: bytes,
    archive_member_name: str,
    output_dir: Path,
    decoded_array_path: Path,
    expected_frames: int,
) -> dict[str, Any]:
    """Decode charged mask bytes through the existing inflate mask loader."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise PlannerError("--decode-mask-array requires numpy") from exc

    runtime = _load_inflate_renderer_module()
    loader = getattr(runtime, "_load_masks_from_archive", None)
    if loader is None:
        raise PlannerError(f"{INFLATE_RENDERER_PATH} does not expose _load_masks_from_archive")

    suffix = _runtime_mask_suffix(archive_member_name, mask_stream)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="cmg_plan_extracted_mask_stream_",
            suffix=suffix,
            dir=str(output_dir),
            delete=False,
        ) as tmp:
            tmp.write(mask_stream)
            tmp_name = tmp.name
        decoded = loader(Path(tmp_name), expected_frames=expected_frames)
        arr = _as_numpy_mask_array(decoded)
    except Exception as exc:
        raise PlannerError(
            "runtime mask decode failed; blocker_class=runtime_decode_contract; "
            f"member={archive_member_name!r}; suffix={suffix!r}; "
            f"expected_frames={expected_frames}; error={type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if tmp_name is not None:
            try:
                Path(tmp_name).unlink()
            except FileNotFoundError:
                pass

    decoded_array_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(decoded_array_path, arr, allow_pickle=False)
    return {
        "requested": True,
        "status": "decoded",
        "helper": str(INFLATE_RENDERER_PATH.relative_to(REPO_ROOT)),
        "runtime_function": "_load_masks_from_archive",
        "archive_member_name": archive_member_name,
        "temporary_suffix": suffix,
        "expected_frames": int(expected_frames),
        "output_array": {
            "artifact_path": _artifact_ref(decoded_array_path, base=output_dir),
            "bytes": int(decoded_array_path.stat().st_size),
            "sha256": _sha256_file(decoded_array_path),
            "shape": [int(v) for v in arr.shape],
            "dtype_saved": str(arr.dtype),
            "class_min": int(arr.min()),
            "class_max": int(arr.max()),
        },
        "contract": (
            "The tensor was decoded only from charged extracted mask bytes via "
            "submissions/robust_current/inflate_renderer.py::_load_masks_from_archive; "
            "no scorer path, external sidecar, or score claim is involved."
        ),
    }


def _load_mask_array(path: Path) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is present in repo env
        raise PlannerError("--mask-array requires numpy") from exc

    path = path.resolve()
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "masks" in payload:
                arr = payload["masks"]
            elif len(payload.files) == 1:
                arr = payload[payload.files[0]]
            else:
                raise PlannerError(f"{path} must contain a 'masks' array or exactly one array")
    else:
        raise PlannerError(f"unsupported mask-array suffix for {path}; expected .npy or .npz")
    if arr.ndim != 3:
        raise PlannerError(f"mask array must have shape (T,H,W), got {tuple(arr.shape)}")
    if arr.size == 0:
        raise PlannerError("mask array is empty")
    if not np.issubdtype(arr.dtype, np.integer):
        raise PlannerError(f"mask array must be integer class ids, got {arr.dtype}")
    arr = arr.astype(np.int16, copy=False)
    if int(arr.min()) < 0:
        raise PlannerError(f"mask array contains negative class ids: min={int(arr.min())}")
    return arr


def _infer_mask_array_shape(path: Path) -> Shape:
    masks = _load_mask_array(path)
    frames, height, width = (int(v) for v in masks.shape)
    class_count = max(DEFAULT_CLASS_COUNT, int(masks.max()) + 1)
    return Shape(frames=frames, height=height, width=width, class_count=class_count)


def _analysis_frames(frame_count: int, *, max_dense_frames: int) -> list[int]:
    if frame_count <= max_dense_frames:
        return list(range(frame_count))
    if max_dense_frames <= 1:
        return [0]
    step = (frame_count - 1) / float(max_dense_frames - 1)
    frames = sorted({int(round(i * step)) for i in range(max_dense_frames)})
    while len(frames) > max_dense_frames:
        frames.pop(-2)
    return frames


def _region_hits_for_bbox(
    bbox: tuple[int, int, int, int],
    regions: list[dict[str, Any]],
) -> list[str]:
    hits = []
    for region in regions:
        rb = tuple(int(v) for v in region["identity"]["bbox_xyxy"])
        if _intersects(bbox, rb):
            hits.append(str(region["identity"]["region_name"]))
    return sorted(hits)


def _component_atoms_for_frame_class(
    frame: int,
    class_id: int,
    binary: Any,
    *,
    regions: list[dict[str, Any]],
    min_area: int,
) -> list[dict[str, Any]]:
    height, width = binary.shape
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise PlannerError("connected component planning requires numpy") from exc

    visited = np.zeros((height, width), dtype=bool)
    atoms: list[dict[str, Any]] = []
    ys, xs = np.nonzero(binary)
    for seed_y, seed_x in zip(ys.tolist(), xs.tolist()):
        if visited[seed_y, seed_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(seed_y, seed_x)])
        visited[seed_y, seed_x] = True
        area = 0
        min_x = seed_x
        max_x = seed_x
        min_y = seed_y
        max_y = seed_y
        while queue:
            y, x = queue.popleft()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not binary[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))
        if area < min_area:
            continue
        bbox = (min_x, min_y, max_x + 1, max_y + 1)
        identity = {
            "frame_index": int(frame),
            "class_id": int(class_id),
            "bbox_xyxy": [int(v) for v in bbox],
            "area_pixels": int(area),
        }
        cost = (
            7
            + _varint_len(frame)
            + _varint_len(class_id)
            + sum(_varint_len(v) for v in bbox)
            + _varint_len(area)
        )
        region_hits = _region_hits_for_bbox(bbox, regions)
        priority = float(area) / max(cost, 1)
        if "vanishing_point_core_r32" in region_hits:
            priority *= 3.0
        elif "horizon_band" in region_hits:
            priority *= 2.0
        elif "ego_lane_corridor" in region_hits:
            priority *= 1.5
        atoms.append(
            {
                "atom_id": _atom_id("cc", identity),
                "atom_family": "connected_component",
                "identity": identity,
                "region_hits": region_hits,
                "byte_cost_estimate": cost,
                "rate_score_cost_estimate": _rate_cost(cost),
                "priority_density": priority,
                "score_claim": False,
            }
        )
    return atoms


def _rle_span_atoms_for_frame(
    frame: int,
    mask_hw: Any,
    *,
    class_count: int,
    regions: list[dict[str, Any]],
    min_span_len: int,
) -> list[dict[str, Any]]:
    height, width = mask_hw.shape
    atoms: list[dict[str, Any]] = []
    for y in range(height):
        x0 = 0
        current = int(mask_hw[y, 0])
        for x in range(1, width + 1):
            value = int(mask_hw[y, x]) if x < width else None
            if value == current:
                continue
            length = x - x0
            if 0 <= current < class_count and length >= min_span_len:
                bbox = (x0, y, x, y + 1)
                identity = {
                    "frame_index": int(frame),
                    "class_id": int(current),
                    "y": int(y),
                    "x0": int(x0),
                    "length": int(length),
                }
                cost = (
                    4
                    + _varint_len(frame)
                    + _varint_len(current)
                    + _varint_len(y)
                    + _varint_len(x0)
                    + _varint_len(length)
                )
                region_hits = _region_hits_for_bbox(bbox, regions)
                priority = float(length) / max(cost, 1)
                if "vanishing_point_core_r32" in region_hits:
                    priority *= 2.5
                elif "horizon_band" in region_hits:
                    priority *= 1.8
                atoms.append(
                    {
                        "atom_id": _atom_id("rle", identity),
                        "atom_family": "scanline_span",
                        "identity": identity,
                        "region_hits": region_hits,
                        "byte_cost_estimate": cost,
                        "rate_score_cost_estimate": _rate_cost(cost),
                        "priority_density": priority,
                        "score_claim": False,
                    }
                )
            x0 = x
            current = int(value) if value is not None else current
    return atoms


def _class_region_atoms_for_frame(
    frame: int,
    mask_hw: Any,
    *,
    class_count: int,
    regions: list[dict[str, Any]],
    min_area: int,
) -> list[dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise PlannerError("class-region planning requires numpy") from exc

    atoms: list[dict[str, Any]] = []
    for class_id in range(class_count):
        ys, xs = np.nonzero(mask_hw == class_id)
        area = int(ys.size)
        if area < min_area:
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        identity = {
            "frame_index": int(frame),
            "class_id": int(class_id),
            "bbox_xyxy": [int(v) for v in bbox],
            "area_pixels": area,
        }
        cost = 5 + _varint_len(frame) + _varint_len(class_id) + sum(_varint_len(v) for v in bbox)
        region_hits = _region_hits_for_bbox(bbox, regions)
        priority = float(area) / max(cost, 1)
        if "horizon_band" in region_hits:
            priority *= 1.8
        if "ego_lane_corridor" in region_hits:
            priority *= 1.4
        atoms.append(
            {
                "atom_id": _atom_id("classreg", identity),
                "atom_family": "class_region",
                "identity": identity,
                "region_hits": region_hits,
                "byte_cost_estimate": cost,
                "rate_score_cost_estimate": _rate_cost(cost),
                "priority_density": priority,
                "score_claim": False,
            }
        )
    return atoms


def _boundary_atoms_for_frame(
    frame: int,
    mask_hw: Any,
    *,
    class_count: int,
    regions: list[dict[str, Any]],
    min_boundary_edges: int,
) -> list[dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise PlannerError("boundary planning requires numpy") from exc

    atoms: list[dict[str, Any]] = []
    hdiff = mask_hw[:, 1:] != mask_hw[:, :-1]
    vdiff = mask_hw[1:, :] != mask_hw[:-1, :]
    for class_a in range(class_count):
        for class_b in range(class_a + 1, class_count):
            hmask = hdiff & (
                ((mask_hw[:, :-1] == class_a) & (mask_hw[:, 1:] == class_b))
                | ((mask_hw[:, :-1] == class_b) & (mask_hw[:, 1:] == class_a))
            )
            vmask = vdiff & (
                ((mask_hw[:-1, :] == class_a) & (mask_hw[1:, :] == class_b))
                | ((mask_hw[:-1, :] == class_b) & (mask_hw[1:, :] == class_a))
            )
            hy, hx = np.nonzero(hmask)
            vy, vx = np.nonzero(vmask)
            edge_count = int(hy.size + vy.size)
            if edge_count < min_boundary_edges:
                continue
            xs: list[int] = []
            ys: list[int] = []
            if hy.size:
                xs.extend(hx.tolist())
                xs.extend((hx + 1).tolist())
                ys.extend(hy.tolist())
            if vy.size:
                xs.extend(vx.tolist())
                ys.extend(vy.tolist())
                ys.extend((vy + 1).tolist())
            bbox = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
            identity = {
                "frame_index": int(frame),
                "class_pair": [int(class_a), int(class_b)],
                "bbox_xyxy": [int(v) for v in bbox],
                "boundary_edges": edge_count,
            }
            cost = (
                9
                + _varint_len(frame)
                + _varint_len(class_a)
                + _varint_len(class_b)
                + sum(_varint_len(v) for v in bbox)
                + _varint_len(edge_count)
            )
            region_hits = _region_hits_for_bbox(bbox, regions)
            priority = float(edge_count) / max(cost, 1)
            if "vanishing_point_core_r32" in region_hits:
                priority *= 3.0
            elif "horizon_band" in region_hits:
                priority *= 2.2
            elif "ego_lane_corridor" in region_hits:
                priority *= 1.5
            atoms.append(
                {
                    "atom_id": _atom_id("boundary", identity),
                    "atom_family": "class_boundary",
                    "identity": identity,
                    "region_hits": region_hits,
                    "boundary_direction_counts": {
                        "horizontal": int(hy.size),
                        "vertical": int(vy.size),
                    },
                    "byte_cost_estimate": cost,
                    "rate_score_cost_estimate": _rate_cost(cost),
                    "priority_density": priority,
                    "score_claim": False,
                }
            )
    return atoms


def _dense_mask_plan(
    mask_array_path: Path,
    *,
    regions: list[dict[str, Any]],
    max_dense_frames: int,
    max_component_atoms: int,
    max_rle_atoms: int,
    max_class_atoms: int,
    max_boundary_atoms: int,
    min_component_area: int,
    min_span_len: int,
    min_boundary_edges: int,
) -> tuple[Shape, dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise PlannerError("--mask-array requires numpy") from exc

    masks = _load_mask_array(mask_array_path)
    frames, height, width = (int(v) for v in masks.shape)
    class_count = max(DEFAULT_CLASS_COUNT, int(masks.max()) + 1)
    shape = Shape(frames=frames, height=height, width=width, class_count=class_count)
    selected_frames = _analysis_frames(frames, max_dense_frames=max_dense_frames)

    hist = np.bincount(masks.reshape(-1), minlength=class_count).astype("int64")
    component_atoms: list[dict[str, Any]] = []
    rle_atoms: list[dict[str, Any]] = []
    class_atoms: list[dict[str, Any]] = []
    boundary_atoms: list[dict[str, Any]] = []
    for frame in selected_frames:
        mask_hw = masks[frame]
        class_atoms.extend(
            _class_region_atoms_for_frame(
                frame,
                mask_hw,
                class_count=class_count,
                regions=regions,
                min_area=min_component_area,
            )
        )
        boundary_atoms.extend(
            _boundary_atoms_for_frame(
                frame,
                mask_hw,
                class_count=class_count,
                regions=regions,
                min_boundary_edges=min_boundary_edges,
            )
        )
        rle_atoms.extend(
            _rle_span_atoms_for_frame(
                frame,
                mask_hw,
                class_count=class_count,
                regions=regions,
                min_span_len=min_span_len,
            )
        )
        for class_id in range(class_count):
            binary = mask_hw == class_id
            component_atoms.extend(
                _component_atoms_for_frame_class(
                    frame,
                    class_id,
                    binary,
                    regions=regions,
                    min_area=min_component_area,
                )
            )

    component_atoms.sort(
        key=lambda atom: (
            -float(atom["priority_density"]),
            -int(atom["identity"]["area_pixels"]),
            int(atom["identity"]["frame_index"]),
            int(atom["identity"]["class_id"]),
            atom["identity"]["bbox_xyxy"],
            atom["atom_id"],
        )
    )
    rle_atoms.sort(
        key=lambda atom: (
            -float(atom["priority_density"]),
            -int(atom["identity"]["length"]),
            int(atom["identity"]["frame_index"]),
            int(atom["identity"]["class_id"]),
            int(atom["identity"]["y"]),
            int(atom["identity"]["x0"]),
            atom["atom_id"],
        )
    )
    class_atoms.sort(
        key=lambda atom: (
            -float(atom["priority_density"]),
            -int(atom["identity"]["area_pixels"]),
            int(atom["identity"]["frame_index"]),
            int(atom["identity"]["class_id"]),
            atom["identity"]["bbox_xyxy"],
            atom["atom_id"],
        )
    )
    boundary_atoms.sort(
        key=lambda atom: (
            -float(atom["priority_density"]),
            -int(atom["identity"]["boundary_edges"]),
            int(atom["identity"]["frame_index"]),
            atom["identity"]["class_pair"],
            atom["identity"]["bbox_xyxy"],
            atom["atom_id"],
        )
    )
    dense = {
        "mask_array": {
            "path": str(mask_array_path.resolve()),
            "bytes": int(mask_array_path.stat().st_size),
            "sha256": _sha256_file(mask_array_path),
            "shape": shape.as_json(),
            "dtype_loaded": str(masks.dtype),
        },
        "analysis_frames": selected_frames,
        "analysis_frame_count": len(selected_frames),
        "class_histogram_pixels": {str(i): int(hist[i]) for i in range(class_count)},
        "class_fraction": {
            str(i): (float(hist[i]) / float(masks.size)) for i in range(class_count)
        },
        "connected_components": component_atoms[:max_component_atoms],
        "rle_spans": rle_atoms[:max_rle_atoms],
        "class_regions": class_atoms[:max_class_atoms],
        "class_boundaries": boundary_atoms[:max_boundary_atoms],
        "component_atoms_considered": len(component_atoms),
        "rle_atoms_considered": len(rle_atoms),
        "class_region_atoms_considered": len(class_atoms),
        "boundary_atoms_considered": len(boundary_atoms),
    }
    return shape, dense


def _policy_table(
    *,
    regions: list[dict[str, Any]],
    frame_groups: list[dict[str, Any]],
    stream_chunks: list[dict[str, Any]],
    dense: dict[str, Any] | None,
    max_policies: int,
) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    region_by_name = {r["identity"]["region_name"]: r["atom_id"] for r in regions}
    frame_pair_ids = [
        atom["atom_id"]
        for atom in frame_groups
        if atom["identity"]["frame_count"] == 2
    ][:16]
    stream_ids = [atom["atom_id"] for atom in stream_chunks[:8]]
    dense_components = dense.get("connected_components", []) if dense else []
    dense_rle = dense.get("rle_spans", []) if dense else []
    dense_class_regions = dense.get("class_regions", []) if dense else []
    dense_boundaries = dense.get("class_boundaries", []) if dense else []

    policies.append(
        {
            "policy_name": "cmg_foveal_horizon_boundary_first",
            "policy_kind": "region_priority",
            "atom_refs": [
                region_by_name[name]
                for name in ("vanishing_point_core_r32", "vanishing_point_context_r80", "horizon_band")
                if name in region_by_name
            ],
            "future_builder_contract": (
                "Spend charged grammar bytes first around VP and horizon boundaries; "
                "all learned tables and selected atoms must be archive members."
            ),
            "score_claim": False,
        }
    )
    policies.append(
        {
            "policy_name": "cmg_pair_block_temporal_tracks",
            "policy_kind": "frame_group_priority",
            "atom_refs": frame_pair_ids,
            "future_builder_contract": (
                "Track births/deaths over adjacent contest pair frames, then lower "
                "to charged component tracks or residual tiles."
            ),
            "score_claim": False,
        }
    )
    policies.append(
        {
            "policy_name": "cmg_stream_chunk_custody_baseline",
            "policy_kind": "byte_stream_reference",
            "atom_refs": stream_ids,
            "future_builder_contract": (
                "Exact-copy stream chunks are byte-custody anchors only; a promoted "
                "grammar must replace them with charged decodeable atoms."
            ),
            "score_claim": False,
        }
    )
    if dense_components:
        policies.append(
            {
                "policy_name": "cmg_large_component_templates",
                "policy_kind": "connected_component_priority",
                "atom_refs": [atom["atom_id"] for atom in dense_components[:32]],
                "future_builder_contract": (
                    "Promote stable high-density components into charged templates, "
                    "with residual tiles for boundaries and class transitions."
                ),
                "score_claim": False,
            }
        )
    if dense_rle:
        policies.append(
            {
                "policy_name": "cmg_scanline_span_residual_tiles",
                "policy_kind": "rle_span_priority",
                "atom_refs": [atom["atom_id"] for atom in dense_rle[:32]],
                "future_builder_contract": (
                    "Use spans for irregular mask areas that are cheaper than spline "
                    "or component-track descriptions."
                ),
                "score_claim": False,
            }
        )
    if dense_class_regions:
        policies.append(
            {
                "policy_name": "cmg_class_region_support_maps",
                "policy_kind": "class_region_priority",
                "atom_refs": [atom["atom_id"] for atom in dense_class_regions[:32]],
                "future_builder_contract": (
                    "Use class support atoms as charged coarse masks before "
                    "boundary and residual refinement."
                ),
                "score_claim": False,
            }
        )
    if dense_boundaries:
        policies.append(
            {
                "policy_name": "cmg_boundary_pair_refinement",
                "policy_kind": "class_boundary_priority",
                "atom_refs": [atom["atom_id"] for atom in dense_boundaries[:32]],
                "future_builder_contract": (
                    "Boundary atoms must lower to charged edge runs or templates "
                    "inside the archive, never scorer-derived side information."
                ),
                "score_claim": False,
            }
        )
    return policies[:max_policies]


def _load_component_trace(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.resolve()
    payload = json.loads(path.read_text())
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        samples = payload.get("top_combined_samples")
    if not isinstance(samples, list) or not samples:
        raise PlannerError(f"component trace has no samples/top_combined_samples: {path}")

    pair_scores: dict[int, dict[str, Any]] = {}
    for item in samples:
        if not isinstance(item, dict):
            continue
        raw_pair = item.get("pair_index", item.get("video_pair_index"))
        if raw_pair is None:
            continue
        pair_index = int(raw_pair)
        combined = float(
            item.get(
                "score_combined_contribution_first_order",
                item.get("combined_score_contribution", item.get("posenet_dist", 0.0)),
            )
        )
        pose = float(item.get("posenet_dist", 0.0))
        seg = float(item.get("segnet_dist", 0.0))
        pair_scores[pair_index] = {
            "pair_index": pair_index,
            "frame_indices": [int(v) for v in item.get("frame_indices", [2 * pair_index, 2 * pair_index + 1])],
            "posenet_dist": pose,
            "segnet_dist": seg,
            "score_combined_contribution_first_order": combined,
            "score_pose_contribution_first_order": float(
                item.get("score_pose_contribution_first_order", 0.0)
            ),
            "score_seg_contribution_exact": float(item.get("score_seg_contribution_exact", 0.0)),
        }
    if not pair_scores:
        raise PlannerError(f"component trace has no usable pair-indexed samples: {path}")

    max_combined = max(abs(float(v["score_combined_contribution_first_order"])) for v in pair_scores.values())
    max_pose = max(abs(float(v["posenet_dist"])) for v in pair_scores.values())
    max_combined = max(max_combined, 1e-12)
    max_pose = max(max_pose, 1e-12)
    for record in pair_scores.values():
        combined_norm = abs(float(record["score_combined_contribution_first_order"])) / max_combined
        pose_norm = abs(float(record["posenet_dist"])) / max_pose
        record["opportunity_weight"] = round(
            1.0 + TRACE_PAIR_WEIGHT_SCALE * combined_norm + TRACE_POSE_WEIGHT_SCALE * pose_norm,
            12,
        )

    ranked_pairs = sorted(
        pair_scores.values(),
        key=lambda record: (
            -float(record["opportunity_weight"]),
            -float(record["score_combined_contribution_first_order"]),
            int(record["pair_index"]),
        ),
    )
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "score_claim": bool(payload.get("score_claim", False)),
        "evidence_grade": payload.get("evidence_grade"),
        "n_samples": payload.get("n_samples", len(pair_scores)),
        "archive_size_bytes": payload.get("archive_size_bytes"),
        "score_recomputed_from_components": payload.get("score_recomputed_from_components"),
        "pair_scores": pair_scores,
        "ranked_pairs": ranked_pairs,
    }


def _atom_frame_indices(atom: dict[str, Any], *, shape: Shape) -> list[int]:
    identity = atom.get("identity", {})
    if "frame_index" in identity:
        frame = int(identity["frame_index"])
        return [frame] if 0 <= frame < shape.frames else []
    if "frame_start" in identity and "frame_end_exclusive" in identity:
        start = max(0, int(identity["frame_start"]))
        end = min(shape.frames, int(identity["frame_end_exclusive"]))
        return list(range(start, end))
    frames = identity.get("frames")
    if isinstance(frames, list):
        return [int(frame) for frame in frames if 0 <= int(frame) < shape.frames]
    return []


def _pairs_for_frames(frames: list[int]) -> list[int]:
    return sorted({int(frame) // 2 for frame in frames if frame >= 0})


def _parse_positive_int_csv(value: str, *, field: str) -> tuple[int, ...]:
    parsed: list[int] = []
    seen: set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        item = int(token)
        if item <= 0:
            raise PlannerError(f"{field} entries must be positive")
        if item not in seen:
            seen.add(item)
            parsed.append(item)
    if not parsed:
        raise PlannerError(f"{field} must contain at least one positive integer")
    return tuple(parsed)


def _family_priority_multiplier(atom: dict[str, Any]) -> float:
    family = str(atom.get("atom_family", ""))
    if family == "class_boundary":
        return 3.0
    if family == "connected_component":
        return 2.25
    if family == "class_region":
        return 1.45
    if family == "scanline_span":
        return 1.25
    if family == "ego_foveal_region":
        return 1.7
    if family == "frame_group":
        return 1.0
    if family == "mask_stream_chunk":
        return 0.15
    return 1.0


def _region_priority_multiplier(atom: dict[str, Any]) -> float:
    hits = set(str(v) for v in atom.get("region_hits", []))
    if "vanishing_point_core_r32" in hits:
        return 2.4
    if "horizon_band" in hits:
        return 2.0
    if "ego_lane_corridor" in hits:
        return 1.55
    if "vanishing_point_context_r80" in hits:
        return 1.35
    identity = atom.get("identity", {})
    name = str(identity.get("region_name", ""))
    if name in {"vanishing_point_core_r32", "horizon_band"}:
        return 2.0
    if name in {"vanishing_point_context_r80", "ego_lane_corridor"}:
        return 1.5
    return 1.0


def _trace_weight_for_pairs(
    pair_indices: list[int],
    trace: dict[str, Any] | None,
    *,
    trace_top_pairs: int,
) -> tuple[float, list[int], list[dict[str, Any]]]:
    if trace is None:
        return 1.0, [], []
    pair_scores = trace["pair_scores"]
    ranked = trace["ranked_pairs"][:trace_top_pairs]
    if pair_indices:
        records = [pair_scores[pair] for pair in pair_indices if pair in pair_scores]
    else:
        records = ranked[: min(8, len(ranked))]
    if not records:
        return 1.0, [], []
    weight = max(float(record["opportunity_weight"]) for record in records)
    records = sorted(records, key=lambda item: (-float(item["opportunity_weight"]), int(item["pair_index"])))
    return weight, [int(record["pair_index"]) for record in records[:8]], records[:8]


def _side_info_breakdown(*, selected_atoms: list[dict[str, Any]], include_residual_model: bool) -> dict[str, Any]:
    atom_count = len(selected_atoms)
    family_count = len({str(atom.get("atom_family", "")) for atom in selected_atoms})
    atom_index_bytes = atom_count * ATOM_SIDE_INFO_BYTES
    codebook_bytes = GRAMMAR_CODEBOOK_SIDE_INFO_BYTES + 8 * family_count
    residual_model_bytes = RESIDUAL_MODEL_SIDE_INFO_BYTES if include_residual_model else 0
    total = GRAMMAR_HEADER_SIDE_INFO_BYTES + codebook_bytes + atom_index_bytes + residual_model_bytes
    return {
        "grammar_header_bytes": GRAMMAR_HEADER_SIDE_INFO_BYTES,
        "family_codebook_bytes": codebook_bytes,
        "atom_index_table_bytes": atom_index_bytes,
        "residual_entropy_model_bytes": residual_model_bytes,
        "total_side_info_bytes": total,
        "statement": (
            "All grammar headers, codebooks, model tables, selected atom ids, "
            "and residual bytes are charged archive payload estimates. Selector "
            "training weights are compress-time-only and are not required at inflate time."
        ),
    }


def _build_allocation_table(
    *,
    atom_tables: dict[str, list[dict[str, Any]]],
    shape: Shape,
    component_trace: dict[str, Any] | None,
    trace_top_pairs: int,
    max_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table_name, atoms in atom_tables.items():
        for atom in atoms:
            byte_cost = int(atom.get("byte_cost_estimate", 0)) + ATOM_SIDE_INFO_BYTES
            frames = _atom_frame_indices(atom, shape=shape)
            pair_indices = _pairs_for_frames(frames)
            trace_weight, trace_pairs, trace_records = _trace_weight_for_pairs(
                pair_indices,
                component_trace,
                trace_top_pairs=trace_top_pairs,
            )
            prior = float(atom.get("priority_density", atom.get("priority_weight_prior", 1.0)))
            opportunity = (
                prior
                * _family_priority_multiplier(atom)
                * _region_priority_multiplier(atom)
                * trace_weight
            )
            density = opportunity / max(byte_cost, 1)
            rows.append(
                {
                    "allocation_rank": 0,
                    "atom_id": atom["atom_id"],
                    "atom_family": atom.get("atom_family"),
                    "source_table": table_name,
                    "identity": atom.get("identity", {}),
                    "estimated_payload_bytes": int(atom.get("byte_cost_estimate", 0)),
                    "estimated_side_info_bytes": ATOM_SIDE_INFO_BYTES,
                    "estimated_charged_bytes": byte_cost,
                    "estimated_rate_score_cost": _rate_cost(byte_cost),
                    "opportunity_score": round(opportunity, 12),
                    "opportunity_per_charged_byte": round(density, 12),
                    "covered_frame_indices_sample": frames[:12],
                    "covered_pair_indices_sample": pair_indices[:12],
                    "trace_top_pair_hits": trace_pairs,
                    "trace_records_sample": [
                        {
                            "pair_index": int(record["pair_index"]),
                            "opportunity_weight": float(record["opportunity_weight"]),
                            "posenet_dist": float(record["posenet_dist"]),
                            "segnet_dist": float(record["segnet_dist"]),
                            "score_combined_contribution_first_order": float(
                                record["score_combined_contribution_first_order"]
                            ),
                        }
                        for record in trace_records[:3]
                    ],
                    "score_claim": False,
                    "evidence_grade": "empirical_allocation_only",
                    "payload_charged": True,
                }
            )
    rows.sort(
        key=lambda row: (
            -float(row["opportunity_per_charged_byte"]),
            -float(row["opportunity_score"]),
            int(row["estimated_charged_bytes"]),
            str(row["atom_id"]),
        )
    )
    for index, row in enumerate(rows[:max_rows], start=1):
        row["allocation_rank"] = index
    return rows[:max_rows]


def _candidate_specs_from_allocation(
    *,
    allocation_rows: list[dict[str, Any]],
    byte_budgets: tuple[int, ...],
    anchor_archive_bytes: int,
    mask_stream_bytes: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for budget in sorted(set(int(v) for v in byte_budgets)):
        selected: list[dict[str, Any]] = []
        payload_bytes = 0
        side = _side_info_breakdown(selected_atoms=[], include_residual_model=True)
        for row in allocation_rows:
            projected_payload = payload_bytes + int(row["estimated_payload_bytes"])
            projected_selected = selected + [row]
            projected_side = _side_info_breakdown(
                selected_atoms=projected_selected,
                include_residual_model=True,
            )
            projected_total = projected_payload + int(projected_side["total_side_info_bytes"])
            if projected_total <= budget:
                selected = projected_selected
                payload_bytes = projected_payload
                side = projected_side
        charged_bytes = payload_bytes + int(side["total_side_info_bytes"])
        family_counts: dict[str, int] = {}
        pair_hits: set[int] = set()
        for row in selected:
            family = str(row["atom_family"])
            family_counts[family] = family_counts.get(family, 0) + 1
            pair_hits.update(int(v) for v in row.get("trace_top_pair_hits", []))
        archive_bytes_if_replaces_mask = max(0, anchor_archive_bytes - mask_stream_bytes + charged_bytes)
        specs.append(
            {
                "candidate_id": f"cmg_trace_alloc_budget_{budget}",
                "candidate_family": "predictive_learned_mask_grammar_allocation",
                "score_claim": False,
                "promotion_eligible": False,
                "evidence_grade": "empirical_allocation_only",
                "budget_bytes": int(budget),
                "selected_atom_count": len(selected),
                "selected_atom_refs": [row["atom_id"] for row in selected],
                "selected_family_counts": dict(sorted(family_counts.items())),
                "charged_payload_bytes_estimate": int(payload_bytes),
                "charged_side_info_accounting": side,
                "charged_total_bytes_estimate": int(charged_bytes),
                "rate_score_cost_estimate": _rate_cost(charged_bytes),
                "anchor_archive_bytes": int(anchor_archive_bytes),
                "mask_stream_bytes_to_replace": int(mask_stream_bytes),
                "archive_bytes_if_replaces_mask_estimate": int(archive_bytes_if_replaces_mask),
                "rate_score_if_components_unchanged_estimate": _rate_cost(archive_bytes_if_replaces_mask),
                "estimated_rate_score_delta_vs_anchor_if_components_unchanged": _rate_cost(
                    archive_bytes_if_replaces_mask - anchor_archive_bytes
                ),
                "trace_pair_hits": sorted(pair_hits)[:32],
                "build_contract": {
                    "all_score_affecting_payloads_charged": True,
                    "external_sidecars_allowed": False,
                    "decoder_loads_scorer": False,
                    "score_truth_required": (
                        "archive.zip -> inflate.sh -> upstream/evaluate.py via "
                        "experiments/contest_auth_eval.py --device cuda"
                    ),
                },
                "why_higher_ev_than_naive_crf_or_rpk1": (
                    "allocates scarce mask bytes to traced PoseNet/SegNet hard pairs, "
                    "horizon/vanishing-point/boundary atoms, and charged side info; "
                    "naive CRF/RPK1 changed broad mask bytes without this component-aware trust region."
                ),
            }
        )
    specs.sort(
        key=lambda spec: (
            -int(spec["selected_atom_count"]),
            int(spec["charged_total_bytes_estimate"]),
            int(spec["budget_bytes"]),
        )
    )
    return specs


def build_plan(
    *,
    source_archive: Path,
    output_dir: Path,
    output_json: Path | None = None,
    member_name: str | None = None,
    mask_member: str | None = None,
    mask_array: Path | None = None,
    decode_mask_array: bool = False,
    decoded_mask_array_output: Path | None = None,
    decode_expected_frames: int = DEFAULT_DECODE_EXPECTED_FRAMES,
    max_frame_groups: int = 128,
    max_stream_chunks: int = 64,
    max_dense_frames: int = 64,
    max_component_atoms: int = 256,
    max_rle_atoms: int = 256,
    max_class_atoms: int = 256,
    max_boundary_atoms: int = 256,
    max_policies: int = 8,
    min_component_area: int = 16,
    min_span_len: int = 8,
    min_boundary_edges: int = 8,
    component_trace_json: Path | None = None,
    trace_top_pairs: int = DEFAULT_TRACE_TOP_PAIRS,
    allocation_byte_budgets: tuple[int, ...] = DEFAULT_ALLOCATION_BYTE_BUDGETS,
    max_allocation_rows: int = 256,
    anchor_archive_bytes: int = DEFAULT_FRONTIER_ARCHIVE_BYTES,
) -> dict[str, Any]:
    """Extract the mask stream and write a deterministic planning manifest."""
    output_dir = output_dir.resolve()
    output_json = (output_json or (output_dir / "atom_plan_manifest.json")).resolve()
    if decode_mask_array and mask_array is not None:
        raise PlannerError("--decode-mask-array and --mask-array are mutually exclusive")
    single_blob = _read_single_blob_archive(source_archive, member_name=member_name)
    parsed = _parse_renderer_payload(single_blob["payload"])
    mask_meta = _select_mask_member(parsed["members"], parsed["header"], requested_member=mask_member)
    mask_stream = parsed["members"][mask_meta["name"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_mask_path = output_dir / "extracted_mask_stream.bin"
    extracted_mask_path.write_bytes(mask_stream)

    mask_decode_contract: dict[str, Any]
    dense_mask_array = mask_array
    if decode_mask_array:
        decoded_path = (decoded_mask_array_output or (output_dir / "decoded_mask_array.npy")).resolve()
        mask_decode_contract = _decode_mask_stream_with_runtime(
            mask_stream=mask_stream,
            archive_member_name=mask_meta["name"],
            output_dir=output_dir,
            decoded_array_path=decoded_path,
            expected_frames=decode_expected_frames,
        )
        dense_mask_array = decoded_path
    elif mask_array is not None:
        mask_decode_contract = {
            "requested": False,
            "status": "caller_supplied_mask_array",
            "contract": (
                "Planner consumed the supplied mask-array path; caller is "
                "responsible for custody/provenance of that tensor."
            ),
        }
    else:
        mask_decode_contract = {
            "requested": False,
            "status": "not_requested",
            "contract": (
                "No dense tensor was decoded; only byte-stream custody, frame, "
                "and ego-region planning atoms are emitted."
            ),
        }

    if dense_mask_array is not None:
        shape = _infer_mask_array_shape(dense_mask_array)
    else:
        shape = Shape(
            frames=DEFAULT_FRAME_COUNT,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            class_count=DEFAULT_CLASS_COUNT,
        )

    regions = _region_defs(shape)
    dense: dict[str, Any] | None = None
    if dense_mask_array is not None:
        shape, dense = _dense_mask_plan(
            dense_mask_array,
            regions=regions,
            max_dense_frames=max_dense_frames,
            max_component_atoms=max_component_atoms,
            max_rle_atoms=max_rle_atoms,
            max_class_atoms=max_class_atoms,
            max_boundary_atoms=max_boundary_atoms,
            min_component_area=min_component_area,
            min_span_len=min_span_len,
            min_boundary_edges=min_boundary_edges,
        )
    frame_groups = _frame_groups(shape.frames, max_groups=max_frame_groups)
    stream_chunks = _stream_chunk_atoms(mask_stream, max_chunks=max_stream_chunks)
    policies = _policy_table(
        regions=regions,
        frame_groups=frame_groups,
        stream_chunks=stream_chunks,
        dense=dense,
        max_policies=max_policies,
    )
    component_trace = _load_component_trace(component_trace_json)
    atom_tables = {
        "frame_groups": frame_groups,
        "ego_foveal_regions": regions,
        "mask_stream_chunks": stream_chunks,
        "connected_components": dense["connected_components"] if dense else [],
        "rle_spans": dense["rle_spans"] if dense else [],
        "class_regions": dense["class_regions"] if dense else [],
        "class_boundaries": dense["class_boundaries"] if dense else [],
    }
    allocation_table = _build_allocation_table(
        atom_tables=atom_tables,
        shape=shape,
        component_trace=component_trace,
        trace_top_pairs=trace_top_pairs,
        max_rows=max_allocation_rows,
    )
    candidate_specs = _candidate_specs_from_allocation(
        allocation_rows=allocation_table,
        byte_budgets=allocation_byte_budgets,
        anchor_archive_bytes=anchor_archive_bytes,
        mask_stream_bytes=len(mask_stream),
    )

    member_records = []
    for name in sorted(parsed["members"]):
        data = parsed["members"][name]
        member_records.append(
            {
                "name": name,
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
                "payload_member_meta": _member_meta_by_name(parsed["header"]).get(name, {}),
            }
        )

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "gpu_required": False,
        "cuda_jobs_launched": False,
        "required_score_truth": (
            "No score truth is produced here. A future full archive must run "
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda."
        ),
        "input_archive": single_blob["archive"],
        "single_blob_member": single_blob["single_blob_member"],
        "renderer_payload_extraction": {
            "helper": parsed["helper"],
            "payload_source_kind": parsed["source_kind"],
            "payload_bytes": parsed["payload_bytes"],
            "payload_sha256": parsed["payload_sha256"],
            "payload_schema": parsed["header"].get("schema"),
            "payload_format": parsed["header"].get("payload_format"),
            "members": member_records,
        },
        "extracted_mask_stream": {
            "archive_member_name": mask_meta["name"],
            "artifact_path": "extracted_mask_stream.bin",
            "bytes": len(mask_stream),
            "sha256": _sha256_bytes(mask_stream),
            "payload_member_meta": mask_meta["payload_member_meta"],
        },
        "planning_schema": {
            "shape": shape.as_json(),
            "atom_tables": [
                "frame_groups",
                "ego_foveal_regions",
                "mask_stream_chunks",
                "connected_components_when_mask_array_available",
                "rle_spans_when_mask_array_available",
                "class_regions_when_mask_array_available",
                "class_boundaries_when_mask_array_available",
            ],
            "byte_cost_units": "estimated charged bytes before entropy coding",
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "dense_component_connectivity": "4-neighbor",
            "determinism": {
                "wall_clock_timestamps_recorded": False,
                "randomness_used": False,
                "sort_keys": "priority_density desc, then stable identity fields",
            },
        },
        "planning_config": {
            "max_frame_groups": max_frame_groups,
            "max_stream_chunks": max_stream_chunks,
            "max_dense_frames": max_dense_frames,
            "max_component_atoms": max_component_atoms,
            "max_rle_atoms": max_rle_atoms,
            "max_class_atoms": max_class_atoms,
            "max_boundary_atoms": max_boundary_atoms,
            "max_policies": max_policies,
            "min_component_area": min_component_area,
            "min_span_len": min_span_len,
            "min_boundary_edges": min_boundary_edges,
            "decode_mask_array": decode_mask_array,
            "decode_expected_frames": decode_expected_frames,
            "component_trace_json": str(component_trace_json.resolve()) if component_trace_json else None,
            "trace_top_pairs": trace_top_pairs,
            "allocation_byte_budgets": [int(v) for v in allocation_byte_budgets],
            "max_allocation_rows": max_allocation_rows,
            "anchor_archive_bytes": anchor_archive_bytes,
        },
        "mask_decode_contract": mask_decode_contract,
        "atom_tables": atom_tables,
        "dense_mask_analysis": dense,
        "candidate_policies": policies,
        "component_trace_prior": None
        if component_trace is None
        else {
            key: value
            for key, value in component_trace.items()
            if key not in {"pair_scores", "ranked_pairs"}
        }
        | (
            {}
            if component_trace is None
            else {
                "top_pairs": [
                    {
                        "pair_index": int(record["pair_index"]),
                        "frame_indices": record["frame_indices"],
                        "opportunity_weight": float(record["opportunity_weight"]),
                        "posenet_dist": float(record["posenet_dist"]),
                        "segnet_dist": float(record["segnet_dist"]),
                        "score_combined_contribution_first_order": float(
                            record["score_combined_contribution_first_order"]
                        ),
                    }
                    for record in component_trace["ranked_pairs"][:trace_top_pairs]
                ]
            }
        ),
        "trace_weighted_allocation": {
            "schema": ALLOCATION_SCHEMA,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": "empirical_allocation_only",
            "all_score_affecting_payloads_charged": True,
            "external_sidecars_allowed": False,
            "allocation_table": allocation_table,
            "candidate_specs": candidate_specs,
        },
        "non_promotable_reason": (
            "Planning-only atom table: no grammar archive was built, no decoder "
            "was integrated, and no CUDA auth eval was run on exact archive bytes."
        ),
        "future_submission_sidecar_policy": {
            "external_sidecars_allowed": False,
            "charged_payload_required": True,
            "note": (
                "Any future grammar table, selector, codebook, residual, or learned "
                "parameter must be serialized inside archive.zip."
            ),
        },
    }
    if not math.isfinite(float(manifest["planning_schema"]["rate_score_per_byte"])):
        raise PlannerError("non-finite rate score per byte")
    _json_dump(output_json, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--member-name", default=None, help="Single blob member name; default requires exactly one member.")
    parser.add_argument("--mask-member", default=None, help="Mask member to select after payload unpacking; default auto-detects.")
    parser.add_argument("--mask-array", type=Path, help="Optional decoded (T,H,W) integer .npy/.npz mask tensor for CC/RLE atoms.")
    parser.add_argument(
        "--decode-mask-array",
        action="store_true",
        help="Decode the extracted mask stream through the existing inflate runtime and save decoded_mask_array.npy.",
    )
    parser.add_argument("--decoded-mask-array-output", type=Path, help="Optional output .npy path for --decode-mask-array.")
    parser.add_argument("--decode-expected-frames", type=int, default=DEFAULT_DECODE_EXPECTED_FRAMES)
    parser.add_argument("--max-frame-groups", type=int, default=128)
    parser.add_argument("--max-stream-chunks", type=int, default=64)
    parser.add_argument("--max-dense-frames", type=int, default=64)
    parser.add_argument("--max-component-atoms", type=int, default=256)
    parser.add_argument("--max-rle-atoms", type=int, default=256)
    parser.add_argument("--max-class-atoms", type=int, default=256)
    parser.add_argument("--max-boundary-atoms", type=int, default=256)
    parser.add_argument("--max-policies", type=int, default=8)
    parser.add_argument("--min-component-area", type=int, default=16)
    parser.add_argument("--min-span-len", type=int, default=8)
    parser.add_argument("--min-boundary-edges", type=int, default=8)
    parser.add_argument(
        "--component-trace-json",
        type=Path,
        help="Optional exact/component trace JSON used only as a compress-time allocation prior.",
    )
    parser.add_argument("--trace-top-pairs", type=int, default=DEFAULT_TRACE_TOP_PAIRS)
    parser.add_argument(
        "--allocation-byte-budgets",
        default=",".join(str(v) for v in DEFAULT_ALLOCATION_BYTE_BUDGETS),
        help="Comma-separated charged-byte budgets for candidate spec emission.",
    )
    parser.add_argument("--max-allocation-rows", type=int, default=256)
    parser.add_argument(
        "--anchor-archive-bytes",
        type=int,
        default=DEFAULT_FRONTIER_ARCHIVE_BYTES,
        help="Anchor archive bytes used only for rate-screen candidate estimates.",
    )
    args = parser.parse_args(argv)

    for name in (
        "decode_expected_frames",
        "max_frame_groups",
        "max_stream_chunks",
        "max_dense_frames",
        "max_component_atoms",
        "max_rle_atoms",
        "max_class_atoms",
        "max_boundary_atoms",
        "max_policies",
        "min_component_area",
        "min_span_len",
        "min_boundary_edges",
        "trace_top_pairs",
        "max_allocation_rows",
        "anchor_archive_bytes",
    ):
        if getattr(args, name) <= 0:
            raise PlannerError(f"--{name.replace('_', '-')} must be positive")
    allocation_byte_budgets = _parse_positive_int_csv(
        args.allocation_byte_budgets,
        field="allocation_byte_budgets",
    )

    manifest = build_plan(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        output_json=args.output_json,
        member_name=args.member_name,
        mask_member=args.mask_member,
        mask_array=args.mask_array,
        decode_mask_array=args.decode_mask_array,
        decoded_mask_array_output=args.decoded_mask_array_output,
        decode_expected_frames=args.decode_expected_frames,
        max_frame_groups=args.max_frame_groups,
        max_stream_chunks=args.max_stream_chunks,
        max_dense_frames=args.max_dense_frames,
        max_component_atoms=args.max_component_atoms,
        max_rle_atoms=args.max_rle_atoms,
        max_class_atoms=args.max_class_atoms,
        max_boundary_atoms=args.max_boundary_atoms,
        max_policies=args.max_policies,
        min_component_area=args.min_component_area,
        min_span_len=args.min_span_len,
        min_boundary_edges=args.min_boundary_edges,
        component_trace_json=args.component_trace_json,
        trace_top_pairs=args.trace_top_pairs,
        allocation_byte_budgets=allocation_byte_budgets,
        max_allocation_rows=args.max_allocation_rows,
        anchor_archive_bytes=args.anchor_archive_bytes,
    )
    print(
        json.dumps(
            {
                "score_claim": manifest["score_claim"],
                "input_archive": manifest["input_archive"],
                "extracted_mask_stream": manifest["extracted_mask_stream"],
                "mask_decode_contract": manifest["mask_decode_contract"],
                "trace_weighted_candidate_specs": manifest["trace_weighted_allocation"][
                    "candidate_specs"
                ],
                "output_dir": str(args.output_dir.resolve()),
                "output_json": str((args.output_json or args.output_dir / "atom_plan_manifest.json").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
