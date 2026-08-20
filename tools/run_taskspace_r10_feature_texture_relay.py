#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a bounded real-video R10 receiver-contract packet and receipt.

This runner deliberately has no scorer, evaluator, dispatcher, or pointer
mutation surface.  Its n1..n24 output proves packet/receiver mechanics on real
input bytes only; it is not scientific evidence and not a contest candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.dash_phase_carrier import (  # noqa: E402
    DashPhaseConfig,
    encode_dash_phase_carrier,
)
from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom  # noqa: E402
from tac.boundary_math.xi_pose_coder import quantize_xi, serialize_xi_payload  # noqa: E402
from tac.witness_dsl.taskspace_r10_feature_texture_relay import (  # noqa: E402
    R10BaseFeatureRecordV1,
    R10IdentityV1,
    R10PacketV1,
    R10PullbackPolygonV1,
    R10RelayMode,
    R10ShootingKnotV1,
    R10StratifiedFlowV1,
    R10TextureRecordV1,
    build_r10_decode_receipt,
    build_r10_selected_solution_adapter,
    pair_population_sha256,
    realization_sha256,
    serialize_r10_packet,
)

RUNNER_SCHEMA: Final = "taskspace_r10_feature_texture_relay_real_video_contract_run.v1"
DEFAULT_VIDEO: Final = REPO / "upstream/videos/0.mkv"
DEFAULT_OUTPUT_DIR: Final = (
    REPO
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "g27_r10_feature_texture_relay"
)


class R10RunnerError(RuntimeError):
    """A bounded runner custody, extraction, or durable-write invariant failed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _read_regular(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise R10RunnerError(f"cannot open {label} as a non-symlink regular file: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < 1:
            raise R10RunnerError(f"{label} must be a nonempty regular file")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise R10RunnerError(f"short read while hashing {label}")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        stable_before = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        stable_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        )
        if stable_after != stable_before:
            raise R10RunnerError(f"{label} changed during custody read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_once_or_equal(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular(path, label="existing output") != payload:
            raise R10RunnerError(f"immutable output already exists with different bytes: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read_regular(path, label="raced output") != payload:
                raise R10RunnerError(f"concurrent immutable output differs: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _ffmpeg_identity(ffmpeg: str) -> dict[str, str]:
    resolved = Path(ffmpeg).resolve()
    binary = _read_regular(resolved, label="ffmpeg binary")
    version = subprocess.run(
        [str(resolved), "-version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.splitlines()[0]
    return {
        "path": str(resolved),
        "sha256": _sha256(binary),
        "version_line": version,
    }


def _extract_real_frames(
    video: Path,
    *,
    pair_count: int,
    height: int,
    width: int,
    ffmpeg: str,
) -> tuple[np.ndarray, list[str]]:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        f"scale={width}:{height}:flags=bicubic",
        "-frames:v",
        str(pair_count * 2),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise R10RunnerError(
            f"ffmpeg frame extraction failed: {exc.stderr.decode('utf-8', errors='replace')}"
        ) from exc
    expected = pair_count * 2 * height * width * 3
    if len(result.stdout) != expected:
        raise R10RunnerError(
            f"ffmpeg returned {len(result.stdout)} raw bytes; expected exactly {expected}"
        )
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        pair_count,
        2,
        height,
        width,
        3,
    )
    return frames.copy(), command


def _luma(frame: np.ndarray) -> np.ndarray:
    work = frame.astype(np.int64)
    return (77 * work[..., 0] + 150 * work[..., 1] + 29 * work[..., 2] + 128) >> 8


def _gradient(luma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = np.zeros_like(luma, dtype=np.int64)
    dy = np.zeros_like(luma, dtype=np.int64)
    dx[:, 1:-1] = luma[:, 2:] - luma[:, :-2]
    dy[1:-1, :] = luma[2:, :] - luma[:-2, :]
    return np.abs(dx) + np.abs(dy), dx, dy


def _intensity_centroid(luma: np.ndarray) -> tuple[float, float]:
    weights = luma.astype(np.float64) + 1.0
    total = float(weights.sum())
    rows = float((weights * np.arange(luma.shape[0])[:, None]).sum() / total)
    cols = float((weights * np.arange(luma.shape[1])[None, :]).sum() / total)
    return rows, cols


def _derive_xi(base: np.ndarray) -> np.ndarray:
    pair_count, _two, height, width, _channels = base.shape
    xi = np.zeros((pair_count, 6), dtype=np.float64)
    for pair in range(pair_count):
        luma0 = _luma(base[pair, 0])
        luma1 = _luma(base[pair, 1])
        row0, col0 = _intensity_centroid(luma0)
        row1, col1 = _intensity_centroid(luma1)
        mean_delta = float(luma1.mean() - luma0.mean())
        xi[pair, 0] = (col1 - col0) / max(1, width) * 0.002
        xi[pair, 1] = (row1 - row0) / max(1, height) * 0.002
        xi[pair, 2] = mean_delta * 0.00001
        grad0, dx0, dy0 = _gradient(luma0)
        grad1, dx1, dy1 = _gradient(luma1)
        xi[pair, 3] = float(dy1.mean() - dy0.mean()) * 0.000001
        xi[pair, 4] = float(dx1.mean() - dx0.mean()) * 0.000001
        xi[pair, 5] = float(grad1.mean() - grad0.mean()) * 0.000001
    return xi


def _derive_support_labels(base: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pair_count, _two, height, width, _channels = base.shape
    labels = np.zeros((pair_count, height, width), dtype=np.uint8)
    centers: list[tuple[int, int]] = []
    border = max(3, min(height, width) // 12)
    for pair in range(pair_count):
        luma = _luma(base[pair, 1])
        magnitude, dx, dy = _gradient(luma)
        interior = magnitude[border : height - border, border : width - border]
        if interior.size == 0:
            raise R10RunnerError("requested receiver geometry is too small for feature support")
        local_row, local_col = np.unravel_index(int(np.argmax(interior)), interior.shape)
        row, col = int(local_row + border), int(local_col + border)
        centers.append((row, col))
        horizontal = abs(int(dx[row, col])) >= abs(int(dy[row, col]))
        if horizontal:
            labels[pair, max(0, row - 1) : min(height, row + 2), max(0, col - 4) : min(width, col + 5)] = 1
        else:
            labels[pair, max(0, row - 4) : min(height, row + 5), max(0, col - 1) : min(width, col + 2)] = 1
    return labels, centers


def _clip_i16(value: int) -> int:
    return max(-32768, min(32767, int(value)))


def _derive_relay_records(
    base: np.ndarray,
) -> tuple[tuple[R10BaseFeatureRecordV1, ...], tuple[R10TextureRecordV1, ...]]:
    base_records: list[R10BaseFeatureRecordV1] = []
    texture_records: list[R10TextureRecordV1] = []
    for pair in range(base.shape[0]):
        frame = base[pair, 1]
        luma = _luma(frame)
        magnitude, _dx, _dy = _gradient(luma)
        channel_std = frame.reshape(-1, 3).std(axis=0)
        weights = tuple(max(64, min(320, round(96 + value))) for value in channel_std)
        base_records.append(
            R10BaseFeatureRecordV1(
                luma_bias_q8=_clip_i16(round((float(luma.mean()) - 128.0) * 8.0)),
                contrast_q8=_clip_i16(24 + round(float(luma.std()) / 4.0)),
                edge_gain_q8=_clip_i16(8 + round(float(magnitude.mean()) / 12.0)),
                feature_gain_q8=_clip_i16(16 + round(float(magnitude.std()) / 12.0)),
                channel_weights_q8=weights,
            )
        )
        texture_records.append(
            R10TextureRecordV1(
                amplitude_q8=_clip_i16(384 + round(float(magnitude.mean()) * 4.0)),
                frequency_q8=max(1, min(65535, 64 + round(float(luma.std())))),
                phase_q10=round(float(luma.mean()) * 4.0) & 1023,
                texture_gain_q8=_clip_i16(224 + round(float(magnitude.std()))),
            )
        )
    return tuple(base_records), tuple(texture_records)


def _derive_knots(base: np.ndarray) -> tuple[R10ShootingKnotV1, ...]:
    indices = (0,) if base.shape[0] == 1 else (0, base.shape[0] - 1)
    rows = []
    for pair in indices:
        luma0 = _luma(base[pair, 0])
        luma1 = _luma(base[pair, 1])
        delta = float(luma1.mean() - luma0.mean())
        rows.append(
            R10ShootingKnotV1(
                pair,
                _clip_i16(round(delta * 16.0) or 16),
                _clip_i16(round(float(luma1.std() - luma0.std()) * 4.0) or 8),
                _clip_i16(round(abs(delta) * 4.0) or 4),
                _clip_i16(round(abs(delta) * 8.0) or 16),
                _clip_i16(round(delta * 8.0) or 8),
                _clip_i16(round(abs(delta) * 8.0) or 16),
            )
        )
    return tuple(rows)


def _derive_pullback(
    center: tuple[int, int],
    *,
    height: int,
    width: int,
    xi0: np.ndarray,
) -> tuple[R10PullbackPolygonV1, R10StratifiedFlowV1]:
    row, col = center
    half_h = max(3, height // 8)
    half_w = max(3, width // 8)
    corners = (
        (max(0, col - half_w), max(0, row - half_h)),
        (min(width - 1, col + half_w), max(0, row - half_h)),
        (min(width - 1, col + half_w), min(height - 1, row + half_h)),
        (max(0, col - half_w), min(height - 1, row + half_h)),
    )
    vertices_q15 = tuple(
        (
            round(x * 32767 / max(1, width - 1)),
            round(y * 32767 / max(1, height - 1)),
        )
        for x, y in corners
    )
    translation_x = _clip_i16(round(float(xi0[0]) * 1_000_000.0) or 8)
    translation_y = _clip_i16(round(float(xi0[1]) * 1_000_000.0) or -8)
    return (
        R10PullbackPolygonV1(0, vertices_q15),
        R10StratifiedFlowV1(0, (8, 0, 0, 8, translation_x, translation_y)),
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    video = Path(args.video).resolve()
    source_bytes = _read_regular(video, label="source video")
    source_sha256 = _sha256(source_bytes)
    ffmpeg_identity = _ffmpeg_identity(args.ffmpeg)
    base, extraction_command = _extract_real_frames(
        video,
        pair_count=args.pair_count,
        height=args.height,
        width=args.width,
        ffmpeg=ffmpeg_identity["path"],
    )
    pair_indices = tuple(range(args.pair_count))
    population_sha256 = pair_population_sha256(source_sha256, pair_indices)
    xi = _derive_xi(base)
    q, scales = quantize_xi(xi, q_levels=args.xi_q_levels)
    xip2 = serialize_xi_payload(q, scales, coder="none")
    labels, support_centers = _derive_support_labels(base)
    pitch_q20 = 0
    pitch = pitch_q20 / float(1 << 20)
    geom = GroundHomographyGeom.eon(native_hw=(args.height, args.width), pitch=pitch)
    dash_cfg = DashPhaseConfig(
        min_area=3,
        border_px=1,
        match_radius_px=float(max(args.height, args.width)),
        q_px=1.0,
        pitch=pitch,
        include_xi=False,
    )
    dash1, dash_report, _decoded_dashes = encode_dash_phase_carrier(
        labels,
        xi,
        dash_cfg,
        geom=geom,
    )
    base_records, texture_records = _derive_relay_records(base)
    polygon, flow = _derive_pullback(
        support_centers[0],
        height=args.height,
        width=args.width,
        xi0=xi[0],
    )
    packet = R10PacketV1(
        mode=R10RelayMode.JOINT,
        identity=R10IdentityV1(
            source_sha256,
            population_sha256,
            realization_sha256(base),
        ),
        pair_indices=pair_indices,
        height=args.height,
        width=args.width,
        pitch_q20=pitch_q20,
        base_features=base_records,
        xip2_payload=xip2,
        textures=texture_records,
        shooting_knots=_derive_knots(base),
        dash1_payload=dash1,
        pullback_polygons=(polygon,),
        stratified_flows=(flow,),
    )
    packet_bytes = serialize_r10_packet(packet)
    output, decode_receipt = build_r10_decode_receipt(
        packet_bytes,
        base,
        expected_source_sha256=source_sha256,
        expected_pair_population_sha256=population_sha256,
    )
    adapter = build_r10_selected_solution_adapter(packet_bytes)
    runner_source = _read_regular(Path(__file__).resolve(), label="runner source")
    packet_sha256 = _sha256(packet_bytes)
    output_dir = Path(args.output_dir).resolve()
    packet_path = output_dir / f"r10_feature_texture_relay_{packet_sha256}.packet"
    receipt_core = {
        "schema": RUNNER_SCHEMA,
        "authority": "bounded_real_video_contract_only",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "not_a_candidate": True,
        "pair_count": args.pair_count,
        "bounded_pair_limit": 24,
        "n600_evidence": False,
        "source_video": {
            "path": str(video),
            "bytes": len(source_bytes),
            "sha256": source_sha256,
        },
        "base_realization_sha256": realization_sha256(base),
        "pair_population_sha256": population_sha256,
        "extraction": {
            "argv": extraction_command,
            "ffmpeg": ffmpeg_identity,
            "height": args.height,
            "width": args.width,
            "frames": args.pair_count * 2,
        },
        "encoder_support_provenance": {
            "kind": "source_rgb_integer_gradient_extremum",
            "support_centers_rc": [list(point) for point in support_centers],
            "scorer_used": False,
            "gt_argmax_used": False,
            "teacher_used": False,
            "payload_is_counted_dash1": True,
        },
        "packet": {
            "path": str(packet_path),
            "bytes": len(packet_bytes),
            "sha256": packet_sha256,
        },
        "output_sha256": realization_sha256(output),
        "dash1_accounting": asdict(dash_report),
        "decode_receipt": decode_receipt,
        "selected_solution_adapter": asdict(adapter),
        "runner_source_sha256": _sha256(runner_source),
        "resumability": {
            "required": False,
            "reason": "bounded n1..n24 in-memory contract run; not a long launch",
        },
        "cleanup": {
            "raw_frames_persisted": False,
            "inflated_video_persisted": False,
            "scratch_paths": [],
        },
    }
    receipt_content_sha256 = _sha256(_canonical_json(receipt_core))
    receipt = {
        **receipt_core,
        "receipt_content_sha256": receipt_content_sha256,
    }
    receipt_bytes = _canonical_json(receipt)
    receipt_path = output_dir / f"r10_feature_texture_relay_{receipt_content_sha256}.receipt.json"
    _write_once_or_equal(packet_path, packet_bytes)
    _write_once_or_equal(receipt_path, receipt_bytes)
    return {
        "packet_path": str(packet_path),
        "packet_sha256": packet_sha256,
        "packet_bytes": len(packet_bytes),
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_bytes),
        "receipt_content_sha256": receipt_content_sha256,
        "output_sha256": realization_sha256(output),
        "pointer_moved": False,
        "score_claim": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--pair-count", type=int, choices=range(1, 25), default=2)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--xi-q-levels", type=int, choices=range(1, 32768), default=2048)
    parser.add_argument("--ffmpeg", default="/opt/homebrew/bin/ffmpeg")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.height < 24 or args.width < 24:
        raise R10RunnerError("receiver geometry must be at least 24x24")
    result = materialize(args)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
