# SPDX-License-Identifier: MIT
"""Archive-bound runtime materializer for SegNet boundary repair overlays.

This module converts a SegNet semantic bridge surface into a byte-closed
candidate archive. The encoder side selects repair pixels from the hinge/wrong
boundary surface, stores a deterministic overlay inside ``archive.zip``, and
adds a decode-only runtime hook that applies the overlay after the base
inflater writes the contest ``.raw`` file.
"""

from __future__ import annotations

import json
import math
import shutil
import stat
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tac.optimization.archive_bound_candidate_runtime_bridge import (
    ARCHIVE_BOUND_RUNTIME_ADAPTER_PACKAGE_SCHEMA,
    emit_archive_bound_candidate_runtime_package,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.repo_io import sha256_file, tree_sha256, write_json

BOUNDARY_REPAIR_RUNTIME_MATERIALIZER_SCHEMA = (
    "segnet_boundary_repair_runtime_materializer.v1"
)
BOUNDARY_REPAIR_OVERLAY_SCHEMA = "segnet_boundary_repair_runtime_overlay.v1"
BOUNDARY_REPAIR_CANDIDATE_ROW_SCHEMA = "segnet_boundary_repair_candidate_row.v1"

DEFAULT_RAW_SHAPE = (1200, 874, 1164, 3)
DEFAULT_GRID_SHAPE = (384, 512)
DEFAULT_MEMBER_NAME = "x"
DEFAULT_OVERLAY_NAME = "boundary_repair_overlay.json"

BoundaryRepairStrategy = Literal["source_pixel_patch", "masked_local_median"]


class BoundaryRepairMaterializerError(ValueError):
    """Raised when a boundary repair candidate cannot be materialized."""


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    resolved = Path(path)
    repo = Path(repo_root)
    try:
        return resolved.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BoundaryRepairMaterializerError(f"{path} must contain a JSON object")
    return payload


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _read_single_member(archive_zip: Path, member_name: str) -> bytes:
    with zipfile.ZipFile(archive_zip, "r") as zf:
        names = zf.namelist()
        if member_name not in names:
            raise BoundaryRepairMaterializerError(
                f"archive {archive_zip} lacks member {member_name!r}; names={names!r}"
            )
        return zf.read(member_name)


def _zip_write_deterministic(
    archive_zip: Path,
    *,
    members: Mapping[str, bytes],
    compression: int = zipfile.ZIP_STORED,
) -> None:
    archive_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip, "w") as zf:
        for name in sorted(members):
            info = zipfile.ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = compression
            info.external_attr = 0o644 << 16
            zf.writestr(info, members[name])


def _copy_runtime_tree(base_submission_dir: Path, submission_dir: Path) -> None:
    if submission_dir.exists() and any(submission_dir.iterdir()):
        raise BoundaryRepairMaterializerError(
            f"refusing to overwrite non-empty submission dir: {submission_dir}"
        )
    submission_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        base_submission_dir,
        submission_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        dirs_exist_ok=True,
    )


def _write_inflate_sh(submission_dir: Path, *, overlay_name: str) -> None:
    text = f"""#!/usr/bin/env bash
export PYTHONDONTWRITEBYTECODE=1
set -euo pipefail

HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

if [ -n "${{PACT_PYTHON_BIN:-}}" ]; then
  PYTHON_BIN="$PACT_PYTHON_BIN"
elif [ -n "${{PYTHON:-}}" ]; then
  PYTHON_BIN="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "ERROR: neither python nor python3 is available" >&2
  exit 127
fi

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${{line%.*}}"
  SRC="${{DATA_DIR}}/{DEFAULT_MEMBER_NAME}"
  if [ ! -f "$SRC" ]; then
    SRC="${{DATA_DIR}}/${{BASE}}.bin"
  fi
  DST="${{OUTPUT_DIR}}/${{BASE}}.raw"
  OVERLAY="${{DATA_DIR}}/{overlay_name}"

  [ ! -f "$SRC" ] && echo "ERROR: ${{SRC}} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  "$PYTHON_BIN" "$HERE/inflate.py" "$SRC" "$DST"
  if [ -f "$OVERLAY" ]; then
    "$PYTHON_BIN" "$HERE/boundary_repair_runtime.py" "$DST" "$OVERLAY" "$BASE"
  fi
done < "$FILE_LIST"
"""
    path = submission_dir / "inflate.sh"
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_runtime_hook(submission_dir: Path) -> None:
    (submission_dir / "boundary_repair_runtime.py").write_text(
        """#!/usr/bin/env python
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _array(payload: dict, key: str, dtype: object) -> np.ndarray:
    return np.asarray(payload.get(key, []), dtype=dtype)


def apply_overlay(raw_path: Path, overlay_path: Path, video_stem: str) -> dict:
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    if overlay.get("video_stem") not in (None, "", video_stem):
        return {"applied": False, "reason": "video_stem_mismatch"}
    raw_shape = tuple(int(v) for v in overlay["raw_shape"])
    raw = np.memmap(raw_path, dtype=np.uint8, mode="r+", shape=raw_shape)
    frame = _array(overlay, "frame_indices", np.int64)
    y = _array(overlay, "y", np.int64)
    x = _array(overlay, "x", np.int64)
    strategy = str(overlay.get("strategy") or "")
    if strategy == "source_pixel_patch":
        rgb = _array(overlay, "rgb", np.uint8).reshape((-1, 3))
        raw[frame, y, x, :] = rgb
    elif strategy == "masked_local_median":
        radius = max(1, int(overlay.get("radius", 1)))
        for f, yy, xx in zip(frame.tolist(), y.tolist(), x.tolist(), strict=True):
            y0 = max(0, yy - radius)
            y1 = min(raw_shape[1], yy + radius + 1)
            x0 = max(0, xx - radius)
            x1 = min(raw_shape[2], xx + radius + 1)
            raw[f, yy, xx, :] = np.median(raw[f, y0:y1, x0:x1, :], axis=(0, 1)).astype(np.uint8)
    else:
        raise ValueError(f"unsupported boundary repair strategy: {strategy!r}")
    raw.flush()
    return {"applied": True, "strategy": strategy, "points": int(len(frame))}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print("usage: boundary_repair_runtime.py RAW_PATH OVERLAY_JSON VIDEO_STEM", file=sys.stderr)
        return 2
    result = apply_overlay(Path(args[0]), Path(args[1]), args[2])
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _surface_record(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    surface = payload.get("semantic_surface_artifacts")
    if not isinstance(surface, Mapping):
        return {}
    record = surface.get("argmax_margin_boundary_npz")
    return record if isinstance(record, Mapping) else {}


def _select_grid_pixels(
    *,
    surface_path: Path,
    max_grid_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if max_grid_pixels <= 0:
        raise BoundaryRepairMaterializerError("max_grid_pixels must be positive")
    surface = np.load(surface_path)
    wrong = surface["wrong_mask"].astype(bool)
    boundary = surface["boundary_mask"].astype(bool)
    hinge = surface["hinge_map"].astype(np.float64)
    sample_ids = surface["sample_ids"].astype(np.int64)
    eligible = wrong & boundary
    coords = np.argwhere(eligible)
    if coords.size == 0:
        raise BoundaryRepairMaterializerError("semantic surface has no wrong boundary pixels")
    scores = hinge[eligible]
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], -scores))
    coords = coords[order[:max_grid_pixels]]
    scores = scores[order[:max_grid_pixels]]
    selected = np.column_stack(
        [
            sample_ids[coords[:, 0]],
            coords[:, 1],
            coords[:, 2],
            scores,
        ]
    )
    summary = {
        "surface_path": str(surface_path),
        "eligible_wrong_boundary_pixels": int(eligible.sum()),
        "selected_grid_pixels": int(selected.shape[0]),
        "selected_hinge_sum": float(scores.sum()),
        "max_selected_hinge": float(scores.max()) if scores.size else 0.0,
        "min_selected_hinge": float(scores.min()) if scores.size else 0.0,
    }
    return selected, summary


def _grid_to_raw_points(
    selected: np.ndarray,
    *,
    raw_shape: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    max_raw_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if max_raw_points <= 0:
        raise BoundaryRepairMaterializerError("max_raw_points must be positive")
    _, raw_h, raw_w, _ = raw_shape
    grid_h, grid_w = grid_shape
    frames: list[int] = []
    ys: list[int] = []
    xs: list[int] = []
    seen: set[tuple[int, int, int]] = set()
    for pair_id_f, gy_f, gx_f, _score in selected:
        pair_id = int(pair_id_f)
        gy = int(gy_f)
        gx = int(gx_f)
        frame_idx = 2 * pair_id + 1
        y0 = math.floor(gy * raw_h / grid_h)
        y1 = max(y0 + 1, math.ceil((gy + 1) * raw_h / grid_h))
        x0 = math.floor(gx * raw_w / grid_w)
        x1 = max(x0 + 1, math.ceil((gx + 1) * raw_w / grid_w))
        for yy in range(max(0, y0), min(raw_h, y1)):
            for xx in range(max(0, x0), min(raw_w, x1)):
                key = (frame_idx, yy, xx)
                if key in seen:
                    continue
                seen.add(key)
                frames.append(frame_idx)
                ys.append(yy)
                xs.append(xx)
                if len(frames) >= max_raw_points:
                    break
            if len(frames) >= max_raw_points:
                break
        if len(frames) >= max_raw_points:
            break
    if not frames:
        raise BoundaryRepairMaterializerError("selected grid pixels mapped to no raw points")
    arrays = (
        np.asarray(frames, dtype=np.int32),
        np.asarray(ys, dtype=np.int16),
        np.asarray(xs, dtype=np.int16),
    )
    summary = {
        "selected_raw_points": len(frames),
        "selected_frame_count": len(set(frames)),
        "first_frame_index": int(min(frames)),
        "last_frame_index": int(max(frames)),
    }
    return *arrays, summary


def _rgb_from_source_raw(
    source_raw_path: Path,
    *,
    raw_shape: tuple[int, int, int, int],
    frame: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    raw = np.memmap(source_raw_path, dtype=np.uint8, mode="r", shape=raw_shape)
    return np.asarray(raw[frame, y, x, :], dtype=np.uint8)


def _load_upstream_yuv420_to_rgb(repo_root: Path):
    import importlib.util

    module_path = repo_root / "upstream" / "frame_utils.py"
    spec = importlib.util.spec_from_file_location("_pact_upstream_frame_utils", module_path)
    if spec is None or spec.loader is None:
        raise BoundaryRepairMaterializerError("unable to import upstream/frame_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.yuv420_to_rgb


def _rgb_from_source_video(
    source_video_path: Path,
    *,
    repo_root: Path,
    raw_shape: tuple[int, int, int, int],
    frame: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BoundaryRepairMaterializerError("PyAV is required for source video repair overlays") from exc
    yuv420_to_rgb = _load_upstream_yuv420_to_rgb(repo_root)
    wanted = {int(v) for v in np.unique(frame)}
    by_frame: dict[int, np.ndarray] = {}
    container = av.open(str(source_video_path))
    try:
        stream = container.streams.video[0]
        for index, decoded in enumerate(container.decode(stream)):
            if index in wanted:
                by_frame[index] = np.asarray(yuv420_to_rgb(decoded).numpy(), dtype=np.uint8)
                if len(by_frame) == len(wanted):
                    break
    finally:
        container.close()
    missing = sorted(wanted.difference(by_frame))
    if missing:
        raise BoundaryRepairMaterializerError(f"source video missing frames: {missing[:8]}")
    _, raw_h, raw_w, channels = raw_shape
    if channels != 3:
        raise BoundaryRepairMaterializerError("source video repair expects RGB raw_shape channel count 3")
    for image in by_frame.values():
        if image.shape != (raw_h, raw_w, 3):
            raise BoundaryRepairMaterializerError(
                f"source frame shape {image.shape} does not match {(raw_h, raw_w, 3)}"
            )
    return np.asarray(
        [by_frame[int(f)][int(yy), int(xx), :] for f, yy, xx in zip(frame, y, x, strict=True)],
        dtype=np.uint8,
    )


def _overlay_payload(
    *,
    strategy: BoundaryRepairStrategy,
    candidate_id: str,
    video_stem: str,
    raw_shape: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    frame: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    rgb: np.ndarray | None,
    radius: int,
    bridge_path: Path,
    surface_path: Path,
    selection_summary: Mapping[str, Any],
    point_summary: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": BOUNDARY_REPAIR_OVERLAY_SCHEMA,
        "candidate_id": candidate_id,
        "strategy": strategy,
        "video_stem": video_stem,
        "raw_shape": list(raw_shape),
        "grid_shape": list(grid_shape),
        "frame_indices": [int(v) for v in frame.tolist()],
        "y": [int(v) for v in y.tolist()],
        "x": [int(v) for v in x.tolist()],
        "radius": int(radius),
        "selection_summary": dict(selection_summary),
        "point_summary": dict(point_summary),
        "source_bridge": {
            "path": _repo_rel(bridge_path, repo_root),
            "sha256": sha256_file(bridge_path),
        },
        "source_surface": {
            "path": _repo_rel(surface_path, repo_root),
            "sha256": sha256_file(surface_path),
        },
        **FALSE_AUTHORITY,
    }
    if rgb is not None:
        payload["rgb"] = [int(v) for v in rgb.reshape(-1).tolist()]
    require_no_truthy_authority_fields(payload, context="boundary_repair_overlay")
    return payload


def _write_archive_dir(
    archive_dir: Path,
    *,
    member_name: str,
    member_data: bytes,
    overlay_name: str,
    overlay_data: bytes,
) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / member_name).write_bytes(member_data)
    (archive_dir / overlay_name).write_bytes(overlay_data)


def materialize_boundary_repair_runtime_candidate(
    *,
    bridge_path: str | Path,
    surface_path: str | Path | None,
    base_submission_dir: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
    strategy: BoundaryRepairStrategy,
    candidate_id: str,
    source_raw_path: str | Path | None = None,
    source_video_path: str | Path | None = None,
    base_archive_path: str | Path | None = None,
    video_name: str = "0.mkv",
    member_name: str = DEFAULT_MEMBER_NAME,
    overlay_name: str = DEFAULT_OVERLAY_NAME,
    raw_shape: Sequence[int] = DEFAULT_RAW_SHAPE,
    grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE,
    max_grid_pixels: int = 2048,
    max_raw_points: int = 16384,
    postfilter_radius: int = 1,
    expected_receiver_output_bytes: int | None = None,
    retain_receiver_output: bool = False,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Materialize one boundary repair runtime candidate and receiver proof."""

    repo = Path(repo_root).resolve(strict=False)
    bridge = _resolve(bridge_path, repo)
    base_submission = _resolve(base_submission_dir, repo)
    out = _resolve(output_dir, repo)
    if strategy not in ("source_pixel_patch", "masked_local_median"):
        raise BoundaryRepairMaterializerError(f"unsupported strategy: {strategy!r}")
    if out.exists() and any(out.iterdir()):
        raise BoundaryRepairMaterializerError(f"refusing to overwrite non-empty output dir: {out}")
    bridge_payload = _read_json(bridge)
    require_no_truthy_authority_fields(bridge_payload, context="boundary_repair_source_bridge")
    resolved_surface = (
        _resolve(surface_path, repo)
        if surface_path is not None
        else _resolve(str(_surface_record(bridge_payload).get("path") or ""), repo)
    )
    if not resolved_surface.is_file():
        raise BoundaryRepairMaterializerError(f"surface NPZ missing: {resolved_surface}")
    raw_shape_tuple = tuple(int(v) for v in raw_shape)
    grid_shape_tuple = tuple(int(v) for v in grid_shape)
    if len(raw_shape_tuple) != 4 or raw_shape_tuple[3] != 3:
        raise BoundaryRepairMaterializerError("raw_shape must be (frames, height, width, 3)")
    if len(grid_shape_tuple) != 2:
        raise BoundaryRepairMaterializerError("grid_shape must be (height, width)")
    selected, selection_summary = _select_grid_pixels(
        surface_path=resolved_surface,
        max_grid_pixels=max_grid_pixels,
    )
    frame, y, x, point_summary = _grid_to_raw_points(
        selected,
        raw_shape=raw_shape_tuple,
        grid_shape=grid_shape_tuple,
        max_raw_points=max_raw_points,
    )
    rgb: np.ndarray | None = None
    if strategy == "source_pixel_patch":
        if source_raw_path is not None:
            rgb = _rgb_from_source_raw(
                _resolve(source_raw_path, repo),
                raw_shape=raw_shape_tuple,
                frame=frame,
                y=y,
                x=x,
            )
        elif source_video_path is not None:
            rgb = _rgb_from_source_video(
                _resolve(source_video_path, repo),
                repo_root=repo,
                raw_shape=raw_shape_tuple,
                frame=frame,
                y=y,
                x=x,
            )
        else:
            raise BoundaryRepairMaterializerError(
                "source_pixel_patch requires source_raw_path or source_video_path"
            )
    overlay = _overlay_payload(
        strategy=strategy,
        candidate_id=candidate_id,
        video_stem=Path(video_name).stem,
        raw_shape=raw_shape_tuple,
        grid_shape=grid_shape_tuple,
        frame=frame,
        y=y,
        x=x,
        rgb=rgb,
        radius=postfilter_radius,
        bridge_path=bridge,
        surface_path=resolved_surface,
        selection_summary=selection_summary,
        point_summary=point_summary,
        repo_root=repo,
    )
    overlay_data = _json_bytes(overlay)
    source_archive = _resolve(
        base_archive_path if base_archive_path is not None else base_submission / "archive.zip",
        repo,
    )
    member_data = _read_single_member(source_archive, member_name)
    submission_dir = out / "submission"
    archive_dir = out / "archive_dir"
    _copy_runtime_tree(base_submission, submission_dir)
    _write_runtime_hook(submission_dir)
    _write_inflate_sh(submission_dir, overlay_name=overlay_name)
    _write_archive_dir(
        archive_dir,
        member_name=member_name,
        member_data=member_data,
        overlay_name=overlay_name,
        overlay_data=overlay_data,
    )
    candidate_archive = submission_dir / "archive.zip"
    _zip_write_deterministic(
        candidate_archive,
        members={member_name: member_data, overlay_name: overlay_data},
    )
    overlay_path = archive_dir / overlay_name
    overlay_manifest = {
        "schema": "segnet_boundary_repair_overlay_manifest.v1",
        "overlay_path": _repo_rel(overlay_path, repo),
        "overlay_sha256": sha256_file(overlay_path),
        "overlay_bytes": overlay_path.stat().st_size,
        "overlay_payload_sha256": _sha256_bytes(overlay_data),
        "strategy": strategy,
        "selection_summary": selection_summary,
        "point_summary": point_summary,
        **FALSE_AUTHORITY,
    }
    write_json(out / "boundary_repair_overlay_manifest.json", overlay_manifest)
    archive_sha = sha256_file(candidate_archive)
    archive_bytes = candidate_archive.stat().st_size
    package = emit_archive_bound_candidate_runtime_package(
        adapter_id="segnet_boundary_repair_runtime_overlay_adapter",
        candidate_family=f"segnet_boundary_{strategy}",
        candidate_id_prefix=candidate_id,
        transform_kind=f"segnet_boundary_{strategy}_runtime_overlay",
        archive_zip_path=candidate_archive,
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
        submission_dir=submission_dir,
        archive_dir_for_inflate=archive_dir,
        output_dir=out,
        repo_root=repo,
        receiver_contract_kind="segnet_boundary_repair_decode_only_runtime_overlay",
        proof_filename="boundary_repair_receiver_proof.json",
        candidate_label=f"segnet_boundary_{strategy}",
        video_name=video_name,
        expected_receiver_output_name=f"{Path(video_name).stem}.raw",
        expected_receiver_output_bytes=expected_receiver_output_bytes,
        retain_receiver_output=retain_receiver_output,
        timeout_seconds=timeout_seconds,
        runtime_adapter_manifest_extra={
            "boundary_repair_overlay_manifest_path": _repo_rel(
                out / "boundary_repair_overlay_manifest.json",
                repo,
            ),
            "boundary_repair_strategy": strategy,
            "base_submission_dir": _repo_rel(base_submission, repo),
            "source_bridge_path": _repo_rel(bridge, repo),
            "source_surface_path": _repo_rel(resolved_surface, repo),
        },
        candidate_row_schema=BOUNDARY_REPAIR_CANDIDATE_ROW_SCHEMA,
        package_filename="archive_bound_candidate_adapter_package.json",
        input_artifacts=ordered_unique(
            [
                _repo_rel(source_archive, repo),
                _repo_rel(bridge, repo),
                _repo_rel(resolved_surface, repo),
                _repo_rel(out / "boundary_repair_overlay_manifest.json", repo),
            ]
        ),
        mlx_triage_argv=[
            "tools/build_segnet_semantic_bridge.py",
            "--candidate-id",
            candidate_id,
        ],
    )
    receiver_proof = package.get("receiver_proof")
    if not isinstance(receiver_proof, Mapping):
        receiver_proof = {}
    adapter_package = package.get("archive_bound_candidate_adapter_package")
    if not isinstance(adapter_package, Mapping):
        adapter_package = {}
    manifest = {
        "schema": BOUNDARY_REPAIR_RUNTIME_MATERIALIZER_SCHEMA,
        "candidate_id": candidate_id,
        "strategy": strategy,
        "source_bridge_path": _repo_rel(bridge, repo),
        "source_surface_path": _repo_rel(resolved_surface, repo),
        "base_submission_dir": _repo_rel(base_submission, repo),
        "source_archive": {
            "path": _repo_rel(source_archive, repo),
            "sha256": sha256_file(source_archive),
            "bytes": source_archive.stat().st_size,
        },
        "candidate_submission_dir": _repo_rel(submission_dir, repo),
        "candidate_archive": {
            "path": _repo_rel(candidate_archive, repo),
            "sha256": archive_sha,
            "bytes": archive_bytes,
        },
        "archive_dir_for_inflate": _repo_rel(archive_dir, repo),
        "runtime_tree_sha256": tree_sha256(submission_dir),
        "overlay_manifest": overlay_manifest,
        "receiver_proof_path": receiver_proof.get("proof_path"),
        "receiver_contract_satisfied": receiver_proof.get("receiver_contract_satisfied")
        is True,
        "runtime_consumption_proof_ready": receiver_proof.get(
            "runtime_consumption_proof_ready"
        )
        is True,
        "archive_bound_candidate_adapter_package_schema": (
            ARCHIVE_BOUND_RUNTIME_ADAPTER_PACKAGE_SCHEMA
        ),
        "archive_bound_candidate_adapter_package_path": _repo_rel(
            out / "archive_bound_candidate_adapter_package.json",
            repo,
        ),
        "archive_bound_candidate_adapter_package": dict(adapter_package),
        "byte_closed_candidate_emitted": True,
        "byte_closed_candidate_materialized": True,
        "candidate_archive_materialized": True,
        "readiness_blockers": list(receiver_proof.get("blockers") or []),
        "allowed_use": "archive_bound_boundary_repair_exact_handoff_planning_only",
        "forbidden_use": "score_claim_or_promotion_without_exact_cpu_cuda_axis",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(manifest, context="boundary_repair_materializer_manifest")
    write_json(out / "boundary_repair_materializer_manifest.json", manifest)
    return manifest


__all__ = [
    "BOUNDARY_REPAIR_CANDIDATE_ROW_SCHEMA",
    "BOUNDARY_REPAIR_OVERLAY_SCHEMA",
    "BOUNDARY_REPAIR_RUNTIME_MATERIALIZER_SCHEMA",
    "BoundaryRepairMaterializerError",
    "materialize_boundary_repair_runtime_candidate",
]
