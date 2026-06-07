#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize official SNeRV source-frame triplets from the contest video."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    repo_relative,
    sha256_file,
    write_json_artifact,
)
from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    git_head_sha,
    load_upstream_yuv420_to_rgb,
)

SCHEMA = "snerv_official_source_frame_triplets.v1"
DEFAULT_SOURCE_VIDEO = REPO_ROOT / "upstream" / "videos" / "0.mkv"
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "source_forward_authority": False,
}


@dataclass(frozen=True)
class NpyWriteResult:
    path: Path
    bytes_written: int
    sha256: str
    free_bytes_before: int
    allow_overwrite: bool


def parse_pair_ids(raw: str) -> list[int]:
    values = [item.strip() for item in raw.replace(",", " ").split()]
    pair_ids = [int(item) for item in values if item]
    if not pair_ids:
        raise ValueError("--pair-ids must name at least one pair id")
    if any(pair_id < 0 for pair_id in pair_ids):
        raise ValueError("--pair-ids must be non-negative")
    return pair_ids


def materialize_source_frame_triplets(
    *,
    video_path: str | Path,
    pair_ids: Sequence[int],
    output_npy: str | Path,
    manifest_path: str | Path | None = None,
    allow_overwrite: bool = False,
    expected_output_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    min_free_bytes: int = 1 << 30,
    command_argv: Sequence[str] | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Decode selected pair triplets and write a custody manifest.

    For each pair id ``p``, the triplet order matches
    ``SNeRV_T.forward(current, previous, next)``:
    ``current=frame[2*p+1]``, ``previous=frame[2*p]``,
    ``next=frame[2*p+2]``.  The output array is uint8
    ``(pairs, 3, 3, H, W)`` in NCHW/0..255 coordinates.
    """

    normalized_pair_ids = [int(value) for value in pair_ids]
    if not normalized_pair_ids:
        raise ValueError("pair_ids must contain at least one item")
    if any(pair_id < 0 for pair_id in normalized_pair_ids):
        raise ValueError("pair_ids must be non-negative")

    source = _resolve_repo_path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"source video not found: {source}")
    output = _resolve_repo_path(output_npy)
    manifest = (
        output.with_suffix(output.suffix + ".manifest.json")
        if manifest_path is None
        else _resolve_repo_path(manifest_path)
    )
    generated = generated_utc or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    triplets, frame_plan, decode_status = _decode_triplets(
        source,
        normalized_pair_ids,
    )
    estimated_npy_bytes = int(triplets.nbytes) + 4096
    write_result = _write_npy_artifact(
        output,
        triplets,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_output_sha256,
        min_free_bytes=max(int(min_free_bytes), estimated_npy_bytes),
    )
    payload = {
        "schema": SCHEMA,
        "generated_utc": generated,
        "producer": "tools/materialize_snerv_source_triplets.py",
        "source": {
            "path": source.as_posix(),
            "repo_relative_path": repo_relative(source, REPO_ROOT),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "pair_ids": normalized_pair_ids,
        "triplet_order": ["current", "previous", "next"],
        "frame_index_formula": {
            "current": "2 * pair_id + 1",
            "previous": "2 * pair_id",
            "next": "2 * pair_id + 2",
        },
        "frame_plan": frame_plan,
        "output": {
            "path": write_result.path.as_posix(),
            "repo_relative_path": repo_relative(write_result.path, REPO_ROOT),
            "shape": [int(value) for value in triplets.shape],
            "dtype": str(triplets.dtype),
            "value_range": "0..255",
            "layout": "NCHW",
            "geometry": {
                "height": int(triplets.shape[-2]),
                "width": int(triplets.shape[-1]),
                "coordinate_system": "source_rgb_frame_geometry",
                "scorer_resized": False,
            },
            "bytes": write_result.bytes_written,
            "sha256": write_result.sha256,
            "free_bytes_before": write_result.free_bytes_before,
        },
        "authority_boundary": {
            "source_frame_triplets_for_official_snerv_t_forward": True,
            "scorer_cache": False,
            "receiver_output": False,
            "source_forward_authority": False,
            "reason": (
                "Triplets are necessary inputs to the strict official Torch "
                "SNeRV_T.forward witness, but do not by themselves prove "
                "MFU/HFR/TUB/output_2 checkpoint source authority."
            ),
        },
        "decode": decode_status,
        "provenance": {
            "cwd": Path.cwd().resolve().as_posix(),
            "argv": list(command_argv or sys.argv),
            "git_head": git_head_sha(repo_root=REPO_ROOT),
            "decode_semantics": (
                "PyAV plus upstream/frame_utils.py::yuv420_to_rgb, matching "
                "upstream/evaluate.py CPU AVVideoDataset RGB semantics"
            ),
            "output_write": "staged np.save followed by guarded link/replace",
        },
        "false_authority_flags": dict(FALSE_AUTHORITY),
        **FALSE_AUTHORITY,
    }
    manifest_result = write_json_artifact(
        manifest,
        payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_manifest_sha256,
    )
    payload["manifest"] = {
        "path": manifest_result.path,
        "bytes": manifest_result.bytes_written,
        "sha256": manifest_result.sha256,
    }
    return payload


def _decode_triplets(
    video_path: Path,
    pair_ids: Sequence[int],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    try:
        import av  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - exercised when env lacks av.
        raise RuntimeError("pyav (`av`) is required to decode the source video") from exc

    yuv420_to_rgb = load_upstream_yuv420_to_rgb(
        substrate_tag="snerv_source_triplet_materializer",
        repo_root=REPO_ROOT,
    )
    frame_plan: list[dict[str, Any]] = []
    wanted: dict[int, list[tuple[int, int]]] = {}
    for out_index, pair_id in enumerate(pair_ids):
        mapping = {
            "current": 2 * int(pair_id) + 1,
            "previous": 2 * int(pair_id),
            "next": 2 * int(pair_id) + 2,
        }
        frame_plan.append(
            {
                "pair_id": int(pair_id),
                "output_pair_index": out_index,
                "source_frame_indices": dict(mapping),
            }
        )
        for triplet_index, name in enumerate(("current", "previous", "next")):
            wanted.setdefault(mapping[name], []).append((out_index, triplet_index))

    max_needed_frame = max(wanted)
    triplets: np.ndarray | None = None
    decoded_frames = 0
    filled: set[tuple[int, int]] = set()
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            decoded_frames = frame_index + 1
            slots = wanted.get(frame_index)
            if slots:
                rgb = _coerce_rgb_frame(yuv420_to_rgb(frame), frame_index=frame_index)
                chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))
                if triplets is None:
                    channels, height, width = chw.shape
                    triplets = np.empty(
                        (len(pair_ids), 3, channels, height, width),
                        dtype=np.uint8,
                    )
                elif tuple(chw.shape) != tuple(triplets.shape[2:]):
                    raise ValueError(
                        f"decoded frame {frame_index} shape {tuple(chw.shape)} "
                        f"does not match first decoded frame {tuple(triplets.shape[2:])}"
                    )
                for out_index, triplet_index in slots:
                    triplets[out_index, triplet_index] = chw
                    filled.add((out_index, triplet_index))
            if frame_index >= max_needed_frame:
                break
    finally:
        container.close()

    expected_slots = {
        (out_index, triplet_index)
        for out_index in range(len(pair_ids))
        for triplet_index in range(3)
    }
    missing = sorted(expected_slots - filled)
    if triplets is None or missing:
        missing_descriptions = [
            {
                "pair_id": int(pair_ids[out_index]),
                "triplet_slot": ("current", "previous", "next")[triplet_index],
                "source_frame_index": frame_plan[out_index]["source_frame_indices"][
                    ("current", "previous", "next")[triplet_index]
                ],
            }
            for out_index, triplet_index in missing
        ]
        raise RuntimeError(
            "source video ended before all requested SNeRV triplet frames were decoded: "
            + json.dumps(missing_descriptions, sort_keys=True)
        )

    return (
        np.ascontiguousarray(triplets),
        frame_plan,
        {
            "decoded_frame_count_until_last_needed": decoded_frames,
            "last_needed_frame_index": max_needed_frame,
            "decoder": "pyav",
            "frame_utils": "upstream/frame_utils.py::yuv420_to_rgb",
        },
    )


def _coerce_rgb_frame(value: Any, *, frame_index: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"decoded frame {frame_index} has shape {arr.shape}; expected HWC RGB"
        )
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _write_npy_artifact(
    path: Path,
    array: np.ndarray,
    *,
    allow_overwrite: bool,
    expected_existing_sha256: str | None,
    min_free_bytes: int,
) -> NpyWriteResult:
    target = path
    if min_free_bytes < 0:
        raise ArtifactWriteError("min_free_bytes must be non-negative")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not allow_overwrite:
        raise ArtifactWriteError(f"refusing to overwrite existing artifact: {target}")
    if target.exists() and allow_overwrite and expected_existing_sha256 is None:
        raise ArtifactWriteError(
            f"{target}: expected_output_sha256 is required before overwrite"
        )
    if expected_existing_sha256 is not None:
        if not target.is_file():
            raise ArtifactWriteError(f"{target}: expected existing artifact is missing")
        actual = sha256_file(target)
        if actual != expected_existing_sha256:
            raise ArtifactWriteError(
                f"{target}: existing artifact sha256 mismatch "
                f"expected={expected_existing_sha256} actual={actual}"
            )
    free_bytes_before = _free_bytes_for_write(target)
    if free_bytes_before < min_free_bytes:
        raise ArtifactWriteError(
            f"{target}: insufficient free space before artifact write "
            f"free={free_bytes_before} required={min_free_bytes}"
        )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(target.parent)) as handle:
            temp_path = Path(handle.name)
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        if allow_overwrite:
            temp_path.replace(target)
        else:
            try:
                os.link(temp_path, target)
            except FileExistsError as exc:
                raise ArtifactWriteError(
                    f"refusing to overwrite existing artifact: {target}"
                ) from exc
        _fsync_directory(target.parent)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
    return NpyWriteResult(
        path=target,
        bytes_written=target.stat().st_size,
        sha256=sha256_file(target),
        free_bytes_before=free_bytes_before,
        allow_overwrite=allow_overwrite,
    )


def _default_output_path(pair_ids: Sequence[int], *, allow_local_output: bool) -> Path:
    slug = "_".join(f"{int(pair_id):04d}" for pair_id in pair_ids[:8])
    if len(pair_ids) > 8:
        slug += f"_plus{len(pair_ids) - 8}"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    filename = f"source_frame_triplets_pairs_{slug}_{stamp}.npy"
    for root in (
        Path("/Volumes/VertigoDataTier/pact"),
        Path("/Volumes/APDataStore/pact"),
    ):
        if root.parent.exists():
            return root / "artifacts" / "snerv_source_triplets" / filename
    if allow_local_output:
        return REPO_ROOT / ".omx" / "research" / "snerv_source_triplets" / filename
    raise ArtifactWriteError(
        "no SSD artifact tier is mounted; pass --out on an SSD path or "
        "--allow-local-output for an explicit local-disk opt-in"
    )


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve(strict=False)


def _free_bytes_for_write(path: Path) -> int:
    root = path.parent
    while not root.exists() and root.parent != root:
        root = root.parent
    return int(shutil.disk_usage(root).free)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_SOURCE_VIDEO)
    parser.add_argument("--pair-ids", default="0")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--allow-local-output", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256", default=None)
    parser.add_argument("--expected-manifest-sha256", default=None)
    parser.add_argument("--min-free-bytes", type=int, default=1 << 30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = _parser().parse_args(raw_argv)
    pair_ids = parse_pair_ids(args.pair_ids)
    out = args.out or _default_output_path(
        pair_ids,
        allow_local_output=bool(args.allow_local_output),
    )
    payload = materialize_source_frame_triplets(
        video_path=args.video,
        pair_ids=pair_ids,
        output_npy=out,
        manifest_path=args.manifest,
        allow_overwrite=bool(args.allow_overwrite),
        expected_output_sha256=args.expected_output_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        min_free_bytes=int(args.min_free_bytes),
        command_argv=[Path(__file__).as_posix(), *raw_argv],
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "pair_ids": payload["pair_ids"],
                "output_npy": payload["output"]["path"],
                "output_shape": payload["output"]["shape"],
                "output_sha256": payload["output"]["sha256"],
                "manifest": payload["manifest"]["path"],
                "manifest_sha256": payload["manifest"]["sha256"],
                "score_claim": payload["score_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
