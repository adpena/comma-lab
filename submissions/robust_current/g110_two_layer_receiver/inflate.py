#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Standalone scorer-free G110 semantic-Y1 plus conditional-Y0 receiver."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np

PACKET_MEMBER: Final = "taskspace_two_layer_v1.bin"
PAIR_COUNT: Final = 600
FRAME_COUNT: Final = 1200
SCORER_H: Final = 384
SCORER_W: Final = 512
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
CHANNELS: Final = 3
EXPECTED_RAW_BYTES: Final = FRAME_COUNT * CAMERA_H * CAMERA_W * CHANNELS
MAX_PACKET_BYTES: Final = 2_100_000
MIN_OUTPUT_HEADROOM_BYTES: Final = EXPECTED_RAW_BYTES + 64 * 1024 * 1024
SEMANTIC_PLUGIN_CALLS: Final = (
    "accepts_packet",
    "parse_packet",
    "render_scorer_y1",
)
FRAME0_PLUGIN_CALLS: Final = (
    "accepts_packet",
    "parse_packet",
    "semantic_packet",
    "render_scorer_y0",
    "verify_final_y1_population",
)
EXPECTED_SEMANTIC_PLUGINS: Final = {
    "original_coordinr_film_mlp_v1.py": "tac.semantic_root_y1.original_coordinr_film_mlp.v1",
    "v9_hosc_dual_head_odd_y1_v1.py": "tac.semantic_root_y1.v9_hosc_dual_head_odd_y1.v1",
}
EXPECTED_FRAME0_PLUGINS: Final = {
    "conditional_lowrank_rice_v1.py": "tac.semantic_root_y0.conditional_lowrank_rice.v1",
}


class PublicInflateError(RuntimeError):
    """The extracted public product or generic runtime failed closed."""


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise PublicInflateError(f"{label} must be a regular non-symlink file")


def _load_packet(archive_root: Path) -> bytes:
    if archive_root.is_symlink() or not archive_root.is_dir():
        raise PublicInflateError("archive root must be a regular directory")
    observed: list[str] = []
    for path in sorted(archive_root.rglob("*")):
        if path.is_symlink():
            raise PublicInflateError("archive root contains a symlink")
        if path.is_file():
            observed.append(path.relative_to(archive_root).as_posix())
    if observed != [PACKET_MEMBER]:
        raise PublicInflateError("archive root member set differs from the closed product")
    packet_path = archive_root / PACKET_MEMBER
    _regular_file(packet_path, label=PACKET_MEMBER)
    packet = packet_path.read_bytes()
    if not 0 < len(packet) <= MAX_PACKET_BYTES:
        raise PublicInflateError("counted two-layer packet is empty or exceeds sparse cap")
    return packet


def _load_plugins(
    directory: Path,
    *,
    calls: tuple[str, ...],
    expected: dict[str, str],
) -> dict[str, ModuleType]:
    if directory.is_symlink() or not directory.is_dir():
        raise PublicInflateError(f"runtime plugin directory is absent: {directory.name}")
    observed_entries = {path.name for path in directory.iterdir()}
    if observed_entries != set(expected):
        raise PublicInflateError(
            f"runtime plugin filenames differ: {directory.name}"
        )
    result: dict[str, ModuleType] = {}
    for filename, expected_variant in sorted(expected.items()):
        path = directory / filename
        _regular_file(path, label=f"runtime plugin {filename}")
        spec = importlib.util.spec_from_file_location(
            f"_public_{directory.name}_{path.stem}",
            path,
        )
        if spec is None or spec.loader is None:
            raise PublicInflateError(f"cannot load runtime plugin {path.name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
        variant = getattr(module, "VARIANT_ID", None)
        if (
            type(variant) is not str
            or not 3 <= len(variant) <= 128
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for character in variant)
        ):
            raise PublicInflateError(f"runtime plugin has invalid VARIANT_ID: {path.name}")
        if variant != expected_variant:
            raise PublicInflateError(
                f"runtime plugin filename/variant binding differs: {path.name}"
            )
        if variant in result:
            raise PublicInflateError(f"duplicate runtime variant: {variant}")
        missing = [name for name in calls if not callable(getattr(module, name, None))]
        if missing:
            raise PublicInflateError(f"runtime plugin {variant} lacks calls: {missing}")
        result[variant] = module
    return result


def _axis_support_indices(input_size: int, output_size: int) -> np.ndarray:
    denominator = 2 * output_size
    rows: list[tuple[int, int]] = []
    claimed: set[int] = set()
    for output_index in range(output_size):
        coordinate_numerator = (2 * output_index + 1) * input_size - output_size
        left = coordinate_numerator // denominator
        fraction_numerator = coordinate_numerator - left * denominator
        taps: dict[int, int] = {}
        for raw_index, numerator in (
            (left, denominator - fraction_numerator),
            (left + 1, fraction_numerator),
        ):
            if numerator:
                index = min(max(raw_index, 0), input_size - 1)
                taps[index] = taps.get(index, 0) + numerator
        indices = tuple(sorted(taps))
        if len(indices) != 2 or sum(taps.values()) != denominator or claimed.intersection(indices):
            raise PublicInflateError("camera/scorer geometry is not certified disjoint factor-2")
        claimed.update(indices)
        rows.append((indices[0], indices[1]))
    return np.asarray(rows, dtype=np.intp)


_ROW_INDICES = _axis_support_indices(CAMERA_H, SCORER_H)
_COL_INDICES = _axis_support_indices(CAMERA_W, SCORER_W)


def _realize_factor2(scorer_rgb: np.ndarray) -> np.ndarray:
    raw = np.asarray(scorer_rgb)
    if raw.dtype != np.uint8 or raw.shape != (SCORER_H, SCORER_W, CHANNELS):
        raise PublicInflateError("semantic variant did not return uint8[384,512,3]")
    scorer = np.ascontiguousarray(raw)
    camera = np.zeros((CAMERA_H, CAMERA_W, CHANNELS), dtype=np.uint8)
    for row_offset in range(2):
        for column_offset in range(2):
            camera[
                _ROW_INDICES[:, row_offset, None],
                _COL_INDICES[None, :, column_offset],
                :,
            ] = scorer
    return camera


def _output_name(video_names_path: Path) -> str:
    _regular_file(video_names_path, label="video names")
    try:
        names = [line.strip() for line in video_names_path.read_text("utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError) as exc:
        raise PublicInflateError("video names file cannot be read as UTF-8") from exc
    if len(names) != 1:
        raise PublicInflateError("SemanticRoot n600 product requires exactly one public test video")
    path = Path(names[0])
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."} or "\\" in names[0] or not path.stem:
        raise PublicInflateError("public test video name is unsafe")
    return f"{path.stem}.raw"


def _prepare_output(output_root: Path, output_name: str) -> tuple[Path, Path]:
    if output_root.is_symlink() or not output_root.is_dir():
        raise PublicInflateError("output root must be an existing regular directory")
    if any(output_root.iterdir()):
        raise PublicInflateError("output root must be empty")
    free_bytes = os.statvfs(output_root).f_bavail * os.statvfs(output_root).f_frsize
    if free_bytes < MIN_OUTPUT_HEADROOM_BYTES:
        raise PublicInflateError("insufficient output storage for exact n600 raw video")
    final_path = output_root / output_name
    temporary_path = output_root / f".{output_name}.g110.tmp"
    if final_path.exists() or temporary_path.exists():
        raise PublicInflateError("output target already exists")
    return final_path, temporary_path


def inflate(archive_root: Path, output_root: Path, video_names_path: Path) -> Path:
    packet = _load_packet(archive_root)
    runtime_root = Path(__file__).resolve().parent
    semantic_plugins = _load_plugins(
        runtime_root / "semantic_variants",
        calls=SEMANTIC_PLUGIN_CALLS,
        expected=EXPECTED_SEMANTIC_PLUGINS,
    )
    frame0_plugins = _load_plugins(
        runtime_root / "frame0_variants",
        calls=FRAME0_PLUGIN_CALLS,
        expected=EXPECTED_FRAME0_PLUGINS,
    )
    frame0_matches = [
        module for module in frame0_plugins.values() if module.accepts_packet(packet) is True
    ]
    if len(frame0_matches) != 1:
        raise PublicInflateError(
            f"two-layer packet matched {len(frame0_matches)} conditional variants; exactly one required"
        )
    frame0 = frame0_matches[0]
    frame0_state = frame0.parse_packet(packet)
    semantic_packet = frame0.semantic_packet(frame0_state)
    matches = [
        module
        for module in semantic_plugins.values()
        if module.accepts_packet(semantic_packet) is True
    ]
    if len(matches) != 1:
        raise PublicInflateError(
            f"semantic packet matched {len(matches)} public runtime variants; exactly one required"
        )
    semantic = matches[0]
    parsed = semantic.parse_packet(semantic_packet)
    final_path, temporary_path = _prepare_output(output_root, _output_name(video_names_path))
    previous_scorer: np.ndarray | None = None
    previous_camera: np.ndarray | None = None
    final_y1_population = hashlib.sha256()
    try:
        with temporary_path.open("xb", buffering=4 * 1024 * 1024) as output:
            for pair_id in range(PAIR_COUNT):
                scorer_y1 = semantic.render_scorer_y1(parsed, pair_id)
                if (
                    type(scorer_y1) is not np.ndarray
                    or scorer_y1.dtype != np.uint8
                    or scorer_y1.shape != (SCORER_H, SCORER_W, CHANNELS)
                ):
                    raise PublicInflateError("semantic variant violated uint8 scorer-Y1 ABI")
                final_y1_population.update(struct.pack(">H", pair_id))
                final_y1_population.update(
                    memoryview(np.ascontiguousarray(scorer_y1)).cast("B")
                )
                if scorer_y1 is previous_scorer:
                    assert previous_camera is not None
                    camera_y1 = previous_camera
                else:
                    camera_y1 = _realize_factor2(scorer_y1)
                    previous_scorer = scorer_y1
                    previous_camera = camera_y1
                scorer_y0 = frame0.render_scorer_y0(
                    frame0_state,
                    pair_id,
                    scorer_y1,
                )
                camera_y0 = _realize_factor2(scorer_y0)
                output.write(memoryview(np.ascontiguousarray(camera_y0)).cast("B"))
                output.write(memoryview(camera_y1).cast("B"))
            output.flush()
            os.fsync(output.fileno())
        frame0.verify_final_y1_population(
            frame0_state,
            final_y1_population.digest(),
        )
        if temporary_path.stat().st_size != EXPECTED_RAW_BYTES:
            raise PublicInflateError("inflated raw byte length differs from exact n600 geometry")
        os.replace(temporary_path, final_path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return final_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("video_names_file", type=Path)
    args = parser.parse_args(argv)
    try:
        path = inflate(args.archive_root, args.output_root, args.video_names_file)
    except (OSError, PublicInflateError, ValueError) as exc:
        print(f"G110_PUBLIC_INFLATE_REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"G110_PUBLIC_INFLATE_OK path={path.name} bytes={path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
