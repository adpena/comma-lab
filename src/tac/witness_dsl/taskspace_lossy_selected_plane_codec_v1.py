# SPDX-License-Identifier: MIT
"""Strict full-n600 lossy selected-plane codec diagnostic.

This module is intentionally a research-only encoder/decoder harness.  It may
consume the allowlisted historical C1 packet as an encoder-side oracle, but its
outputs are never candidate lineage and never score authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from tac.witness_dsl.c0b_semantic_quotient import storage_preflight
from tac.witness_dsl.v10_production_receiver import (
    decode_y_plane_pair,
    parse_packet,
    realize_pair_frame1,
)

SCHEMA = "taskspace_lossy_selected_plane_codec.v1"
CONFIG_SCHEMA = "taskspace_lossy_selected_plane_codec_config.v1"
SEGMENT_RECEIPT_SCHEMA = "taskspace_lossy_selected_plane_segment_receipt.v1"
AGGREGATE_RECEIPT_SCHEMA = "taskspace_lossy_selected_plane_aggregate_receipt.v1"
BUNDLE_MANIFEST_SCHEMA = "taskspace_lossy_selected_plane_bundle_manifest.v1"
FAILURE_RECEIPT_SCHEMA = "taskspace_lossy_selected_plane_failure_receipt.v1"
RAW_BRIDGE_RECEIPT_SCHEMA = "taskspace_lossy_selected_plane_raw_bridge_receipt.v1"
RAW_BRIDGE_STAGE_SCHEMA = "taskspace_lossy_selected_plane_raw_bridge_stage.v1"

PAIR_COUNT = 600
PAIRS_PER_SEGMENT = 120
SEGMENT_COUNT = 5
HEIGHT = 384
WIDTH = 512
CHANNELS = 3
RATE_CEILING_BYTES = 258_312
CONDITIONAL_PLANNING_MAX_SUB0172_BYTES = 187_563
CONDITIONAL_PLANNING_MAX_SUB015_BYTES = 154_523
CANONICAL_PLANNING_ANCHOR_PATH = (
    ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "c1_live_target_debt_n600_batch16.json"
)
CANONICAL_PLANNING_ANCHOR_SHA256 = "0db8e47a994cad5367e5eb3028055e667bc4caf3f174026d13171be662e7fbd3"
CANONICAL_PLANNING_DSEG = 0.00015196058485243054
CANONICAL_PLANNING_DPOSE = 0.00010184347386600314
FPS = 20

DIRECT_ARM = "DIRECT_INTERLEAVED_RGB"
LAYERED_ARM = "TASK_LAYERED"
ARMS = (DIRECT_ARM, LAYERED_ARM)
DIRECT_STREAM = "interleaved_y0_y1"
BASE_STREAM = "temporal_y1_base"
ENHANCEMENT_STREAM = "conditional_y0_given_y1_q2"

HISTORICAL_C1_ARCHIVE_SHA256 = "e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42"
HISTORICAL_C1_MEMBER_SHA256 = "aa1dbb5e2efff28cd0d31f5ee2a4b0575a248a27a431151bfcae64eb320d385b"
HISTORICAL_C1_MEMBER = "0.bin"

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "research_only",
        "historical_c1_encoder_oracle_only",
        "candidate_lineage_allowed",
        "pair_count",
        "pairs_per_segment",
        "segment_count",
        "geometry",
        "rate_ceiling_bytes",
        "source",
        "codec",
        "endpoints",
        "required_free_bytes",
        "test_only_small_fixture",
    }
)
_GEOMETRY_FIELDS = frozenset({"height", "width", "channels"})
_SOURCE_FIELDS = frozenset({"archive_path", "archive_sha256", "member_name", "member_sha256"})
_CODEC_FIELDS = frozenset(
    {
        "ffmpeg_executable",
        "encoder",
        "container",
        "input_pixel_format",
        "encoded_pixel_format",
        "decoded_pixel_format",
        "frame_rate",
        "preset",
        "threads",
        "color_range",
        "colorspace",
        "color_primaries",
        "color_transfer",
    }
)
_ENDPOINT_FIELDS = frozenset({"name", "direct_bitrate_bps", "base_bitrate_bps", "enhancement_bitrate_bps"})


class LossySelectedPlaneCodecError(RuntimeError):
    """Fail-closed config, custody, codec, resume, or aggregate error."""


@dataclass(frozen=True)
class SourcePlanes:
    """Strictly reopened historical encoder-oracle planes."""

    frame0: np.ndarray
    frame1: np.ndarray
    archive_sha256: str
    member_sha256: str
    frame0_sha256: str
    frame1_sha256: str


@dataclass(frozen=True)
class FfmpegCodecSpec:
    executable: str
    encoder: str
    container: str
    input_pixel_format: str
    encoded_pixel_format: str
    decoded_pixel_format: str
    frame_rate: int
    preset: int | str
    threads: int
    color_range: str
    colorspace: str
    color_primaries: str
    color_transfer: str


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(value)).hexdigest()


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise LossySelectedPlaneCodecError(f"{label} must be an integer >= {minimum}")
    return value


def _exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise LossySelectedPlaneCodecError(f"{label} must be a boolean")
    return value


def _exact_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LossySelectedPlaneCodecError(f"{label} must be a non-empty string")
    return value


def _require_fields(value: Any, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != expected:
        raise LossySelectedPlaneCodecError(f"{label} has missing or unknown fields")
    return value


def _require_sha(value: Any, label: str) -> str:
    text = _exact_str(value, label).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise LossySelectedPlaneCodecError(f"{label} must be a lowercase SHA-256")
    return text


def _atomic_write(path: Path, payload: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and path.exists():
            raise LossySelectedPlaneCodecError(f"write-once path already exists: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_once_or_equal(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise LossySelectedPlaneCodecError(f"preserved write-once bytes drifted: {path}")
        return
    _atomic_write(path, payload, replace=False)


def load_config(path: Path | str) -> Mapping[str, Any]:
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LossySelectedPlaneCodecError(f"cannot load config: {config_path}") from exc
    config = _require_fields(value, _TOP_LEVEL_FIELDS, "config")
    if config["schema"] != CONFIG_SCHEMA:
        raise LossySelectedPlaneCodecError("config schema drift")
    if _exact_bool(config["research_only"], "research_only") is not True:
        raise LossySelectedPlaneCodecError("experiment must remain research_only")
    if _exact_bool(config["historical_c1_encoder_oracle_only"], "historical_c1_encoder_oracle_only") is not True:
        raise LossySelectedPlaneCodecError("historical C1 may only be an encoder oracle")
    if _exact_bool(config["candidate_lineage_allowed"], "candidate_lineage_allowed") is not False:
        raise LossySelectedPlaneCodecError("historical C1 candidate lineage must be forbidden")
    test_only = _exact_bool(config["test_only_small_fixture"], "test_only_small_fixture")
    expected_counts = (PAIR_COUNT, PAIRS_PER_SEGMENT, SEGMENT_COUNT)
    actual_counts = (
        _exact_int(config["pair_count"], "pair_count", minimum=1),
        _exact_int(config["pairs_per_segment"], "pairs_per_segment", minimum=1),
        _exact_int(config["segment_count"], "segment_count", minimum=1),
    )
    if not test_only and actual_counts != expected_counts:
        raise LossySelectedPlaneCodecError("full diagnostic must use 600 pairs in five 120-pair segments")
    if actual_counts[0] != actual_counts[1] * actual_counts[2]:
        raise LossySelectedPlaneCodecError("segment lattice does not exactly cover pair_count")
    geometry = _require_fields(config["geometry"], _GEOMETRY_FIELDS, "geometry")
    actual_geometry = (
        _exact_int(geometry["height"], "geometry.height", minimum=1),
        _exact_int(geometry["width"], "geometry.width", minimum=1),
        _exact_int(geometry["channels"], "geometry.channels", minimum=1),
    )
    if not test_only and actual_geometry != (HEIGHT, WIDTH, CHANNELS):
        raise LossySelectedPlaneCodecError("full diagnostic geometry must be 384x512x3")
    if _exact_int(config["rate_ceiling_bytes"], "rate_ceiling_bytes", minimum=1) != RATE_CEILING_BYTES:
        raise LossySelectedPlaneCodecError("rate ceiling must remain the current 258312-byte scale")
    _exact_int(config["required_free_bytes"], "required_free_bytes", minimum=1)
    source = _require_fields(config["source"], _SOURCE_FIELDS, "source")
    _exact_str(source["archive_path"], "source.archive_path")
    if _require_sha(source["archive_sha256"], "source.archive_sha256") != HISTORICAL_C1_ARCHIVE_SHA256:
        raise LossySelectedPlaneCodecError("source archive is not the allowlisted historical C1 oracle")
    if source["member_name"] != HISTORICAL_C1_MEMBER:
        raise LossySelectedPlaneCodecError("source member name drift")
    if _require_sha(source["member_sha256"], "source.member_sha256") != HISTORICAL_C1_MEMBER_SHA256:
        raise LossySelectedPlaneCodecError("source member is not the allowlisted historical C1 oracle")
    codec = _require_fields(config["codec"], _CODEC_FIELDS, "codec")
    for field in (
        "ffmpeg_executable",
        "encoder",
        "container",
        "input_pixel_format",
        "encoded_pixel_format",
        "decoded_pixel_format",
        "color_range",
        "colorspace",
        "color_primaries",
        "color_transfer",
    ):
        _exact_str(codec[field], f"codec.{field}")
    codec_identity = (codec["encoder"], codec["container"], codec["encoded_pixel_format"])
    supported = {
        ("libsvtav1", "ivf", "yuv420p"),
        ("libx265", "hevc", "yuv444p"),
        ("libx264rgb", "h264", "rgb24"),
    }
    if codec_identity not in supported:
        raise LossySelectedPlaneCodecError("unsupported explicit encoder/container/pixel-format contract")
    if (codec["input_pixel_format"], codec["decoded_pixel_format"]) != ("rgb24", "rgb24"):
        raise LossySelectedPlaneCodecError("input and decoded pixel formats must remain explicit rgb24")
    for field in ("frame_rate", "threads"):
        _exact_int(codec[field], f"codec.{field}", minimum=1)
    if codec["encoder"] == "libsvtav1":
        _exact_int(codec["preset"], "codec.preset", minimum=1)
    elif not isinstance(codec["preset"], str) or codec["preset"] not in {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }:
        raise LossySelectedPlaneCodecError("x264/x265 preset must be one supported explicit preset name")
    if codec["threads"] != 1:
        raise LossySelectedPlaneCodecError("deterministic diagnostic requires one encoder thread")
    endpoints = config["endpoints"]
    if not isinstance(endpoints, list) or not endpoints:
        raise LossySelectedPlaneCodecError("endpoints must be a non-empty list")
    endpoint_names: set[str] = set()
    for index, raw_endpoint in enumerate(endpoints):
        endpoint = _require_fields(raw_endpoint, _ENDPOINT_FIELDS, f"endpoint[{index}]")
        name = _exact_str(endpoint["name"], f"endpoint[{index}].name")
        if name in endpoint_names:
            raise LossySelectedPlaneCodecError("endpoint names must be unique")
        endpoint_names.add(name)
        for field in ("direct_bitrate_bps", "base_bitrate_bps", "enhancement_bitrate_bps"):
            _exact_int(endpoint[field], f"endpoint[{index}].{field}", minimum=1)
    return config


def config_identity(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(config))


def codec_spec(config: Mapping[str, Any]) -> FfmpegCodecSpec:
    raw = config["codec"]
    return FfmpegCodecSpec(
        executable=raw["ffmpeg_executable"],
        encoder=raw["encoder"],
        container=raw["container"],
        input_pixel_format=raw["input_pixel_format"],
        encoded_pixel_format=raw["encoded_pixel_format"],
        decoded_pixel_format=raw["decoded_pixel_format"],
        frame_rate=raw["frame_rate"],
        preset=raw["preset"],
        threads=raw["threads"],
        color_range=raw["color_range"],
        colorspace=raw["colorspace"],
        color_primaries=raw["color_primaries"],
        color_transfer=raw["color_transfer"],
    )


def ffmpeg_version(spec: FfmpegCodecSpec) -> Mapping[str, Any]:
    resolved = shutil.which(spec.executable)
    if resolved is None:
        raise LossySelectedPlaneCodecError(f"ffmpeg executable not found: {spec.executable}")
    proc = subprocess.run(
        [resolved, "-version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode:
        raise LossySelectedPlaneCodecError(f"ffmpeg version query failed: {proc.stderr[:400]}")
    first_line = proc.stdout.splitlines()[0] if proc.stdout.splitlines() else ""
    if not first_line.startswith("ffmpeg version "):
        raise LossySelectedPlaneCodecError("ffmpeg version output drift")
    return {"resolved_executable": resolved, "version_line": first_line}


def load_historical_c1_source(config: Mapping[str, Any]) -> SourcePlanes:
    if config["test_only_small_fixture"]:
        raise LossySelectedPlaneCodecError("test fixtures must inject arrays, never impersonate historical C1")
    source = config["source"]
    archive_path = Path(source["archive_path"])
    if archive_path.is_symlink() or not archive_path.is_file():
        raise LossySelectedPlaneCodecError("historical C1 archive must be a regular non-symlink file")
    archive_hash = sha256_file(archive_path)
    if archive_hash != source["archive_sha256"]:
        raise LossySelectedPlaneCodecError("historical C1 archive SHA-256 mismatch")
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != HISTORICAL_C1_MEMBER:
                raise LossySelectedPlaneCodecError("historical C1 archive member set drift")
            info = infos[0]
            if info.is_dir() or PurePosixPath(info.filename).is_absolute() or ".." in PurePosixPath(info.filename).parts:
                raise LossySelectedPlaneCodecError("historical C1 member path is unsafe")
            packet_bytes = archive.read(info)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise LossySelectedPlaneCodecError("cannot reopen historical C1 archive") from exc
    member_hash = sha256_bytes(packet_bytes)
    if member_hash != source["member_sha256"]:
        raise LossySelectedPlaneCodecError("historical C1 member SHA-256 mismatch")
    try:
        decoded = decode_y_plane_pair(parse_packet(packet_bytes))
    except Exception as exc:  # receiver owns detailed structural error types
        raise LossySelectedPlaneCodecError("historical C1 strict V10 receiver refused the packet") from exc
    expected_shape = (
        config["pair_count"],
        config["geometry"]["height"],
        config["geometry"]["width"],
        config["geometry"]["channels"],
    )
    if decoded.frame0.shape != expected_shape or decoded.frame1.shape != expected_shape:
        raise LossySelectedPlaneCodecError("historical C1 decoded plane geometry/order drift")
    if decoded.frame0.dtype != np.uint8 or decoded.frame1.dtype != np.uint8:
        raise LossySelectedPlaneCodecError("historical C1 decoded planes must be uint8")
    return SourcePlanes(
        frame0=np.ascontiguousarray(decoded.frame0),
        frame1=np.ascontiguousarray(decoded.frame1),
        archive_sha256=archive_hash,
        member_sha256=member_hash,
        frame0_sha256=sha256_array(decoded.frame0),
        frame1_sha256=sha256_array(decoded.frame1),
    )


def _load_historical_c1_packet(config: Mapping[str, Any]):
    source = config["source"]
    archive_path = Path(source["archive_path"])
    if archive_path.is_symlink() or not archive_path.is_file():
        raise LossySelectedPlaneCodecError("historical C1 archive must be a regular non-symlink file")
    if sha256_file(archive_path) != source["archive_sha256"]:
        raise LossySelectedPlaneCodecError("historical C1 archive SHA-256 mismatch")
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != HISTORICAL_C1_MEMBER:
                raise LossySelectedPlaneCodecError("historical C1 archive member set drift")
            packet_bytes = archive.read(infos[0])
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise LossySelectedPlaneCodecError("cannot reopen historical C1 archive") from exc
    if sha256_bytes(packet_bytes) != source["member_sha256"]:
        raise LossySelectedPlaneCodecError("historical C1 member SHA-256 mismatch")
    try:
        return parse_packet(packet_bytes)
    except Exception as exc:
        raise LossySelectedPlaneCodecError("historical C1 strict V10 receiver refused the packet") from exc


def conditional_enhancement(frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
    _validate_same_plane_arrays(frame0, frame1, "conditional enhancement")
    delta = np.rint((frame0.astype(np.int16) - frame1.astype(np.int16)) / 2.0)
    return np.clip(delta + 128, 0, 255).astype(np.uint8)


def reconstruct_frame0(frame1: np.ndarray, enhancement: np.ndarray) -> np.ndarray:
    _validate_same_plane_arrays(frame1, enhancement, "conditional reconstruction")
    value = frame1.astype(np.int16) + 2 * (enhancement.astype(np.int16) - 128)
    return np.clip(value, 0, 255).astype(np.uint8)


def direct_interleave(frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
    _validate_same_plane_arrays(frame0, frame1, "direct interleave")
    output = np.empty((frame0.shape[0] * 2, *frame0.shape[1:]), dtype=np.uint8)
    output[0::2] = frame0
    output[1::2] = frame1
    return output


def direct_deinterleave(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _validate_frames(frames, "direct decoded frames")
    if len(frames) % 2:
        raise LossySelectedPlaneCodecError("direct decoded frame count must be even")
    return np.ascontiguousarray(frames[0::2]), np.ascontiguousarray(frames[1::2])


def _validate_frames(frames: np.ndarray, label: str) -> None:
    if (
        not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[-1] != CHANNELS
        or not frames.flags.c_contiguous
    ):
        raise LossySelectedPlaneCodecError(f"{label} must be contiguous NHWC uint8 RGB")


def _validate_same_plane_arrays(left: np.ndarray, right: np.ndarray, label: str) -> None:
    _validate_frames(left, f"{label} left")
    _validate_frames(right, f"{label} right")
    if left.shape != right.shape:
        raise LossySelectedPlaneCodecError(f"{label} arrays must have identical shape")


def encode_argv(
    spec: FfmpegCodecSpec,
    *,
    width: int,
    height: int,
    frame_count: int,
    bitrate_bps: int,
    output_path: Path,
) -> list[str]:
    gop = frame_count
    common = [
        spec.executable,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        spec.input_pixel_format,
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(spec.frame_rate),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        spec.encoder,
    ]
    if spec.encoder == "libsvtav1":
        encoder_args = [
            "-preset",
            str(spec.preset),
            "-b:v",
            str(bitrate_bps),
            "-svtav1-params",
            f"rc=1:keyint={gop}:lp=1",
        ]
    elif spec.encoder == "libx265":
        encoder_args = [
            "-preset",
            str(spec.preset),
            "-b:v",
            str(bitrate_bps),
            "-x265-params",
            (
                f"pools=1:frame-threads=1:wpp=0:keyint={gop}:min-keyint={gop}:"
                "scenecut=0:bframes=0:repeat-headers=1:log-level=error"
            ),
        ]
    elif spec.encoder == "libx264rgb":
        encoder_args = [
            "-preset",
            str(spec.preset),
            "-b:v",
            str(bitrate_bps),
            "-x264-params",
            (
                f"threads=1:sliced-threads=0:keyint={gop}:min-keyint={gop}:"
                "scenecut=0:bframes=0:repeat-headers=1"
            ),
        ]
    else:
        raise LossySelectedPlaneCodecError(f"unsupported encoder: {spec.encoder}")
    return [
        *common,
        *encoder_args,
        "-pix_fmt",
        spec.encoded_pixel_format,
        "-threads",
        str(spec.threads),
        "-color_range",
        spec.color_range,
        "-colorspace",
        spec.colorspace,
        "-color_primaries",
        spec.color_primaries,
        "-color_trc",
        spec.color_transfer,
        "-f",
        spec.container,
        str(output_path),
    ]


def decode_argv(spec: FfmpegCodecSpec, *, input_path: Path) -> list[str]:
    return [
        spec.executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        spec.decoded_pixel_format,
        "-color_range",
        spec.color_range,
        "-colorspace",
        spec.colorspace,
        "pipe:1",
    ]


def encode_stream(
    frames: np.ndarray,
    *,
    spec: FfmpegCodecSpec,
    bitrate_bps: int,
    output_path: Path,
) -> Mapping[str, Any]:
    _validate_frames(frames, "encode frames")
    frame_count, height, width, _ = frames.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    argv = encode_argv(
        spec,
        width=width,
        height=height,
        frame_count=frame_count,
        bitrate_bps=bitrate_bps,
        output_path=output_path,
    )
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    assert proc.stderr is not None
    try:
        for frame in frames:
            proc.stdin.write(memoryview(frame))
        proc.stdin.close()
        stderr = proc.stderr.read()
        returncode = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if returncode:
        raise LossySelectedPlaneCodecError(
            f"ffmpeg encode failed rc={returncode}: {stderr.decode('utf-8', 'replace')[:1000]}"
        )
    if not output_path.is_file() or output_path.is_symlink() or output_path.stat().st_size <= 0:
        raise LossySelectedPlaneCodecError("ffmpeg did not produce a regular non-empty bitstream")
    return {
        "argv": argv,
        "bitrate_bps": bitrate_bps,
        "rate_control": f"{spec.encoder}-bitrate.v1",
        "gop_frames": frame_count,
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def decode_stream(
    input_path: Path,
    *,
    spec: FfmpegCodecSpec,
    expected_frame_count: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    argv = decode_argv(spec, input_path=input_path)
    proc = subprocess.run(argv, check=False, capture_output=True)
    if proc.returncode:
        raise LossySelectedPlaneCodecError(
            f"ffmpeg decode failed rc={proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[:1000]}"
        )
    expected_bytes = expected_frame_count * height * width * CHANNELS
    if len(proc.stdout) != expected_bytes:
        raise LossySelectedPlaneCodecError(
            f"decoded frame bytes drifted: expected {expected_bytes}, received {len(proc.stdout)}"
        )
    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(
        (expected_frame_count, height, width, CHANNELS)
    ).copy()
    return frames, {
        "argv": argv,
        "frame_count": expected_frame_count,
        "height": height,
        "width": width,
        "decoded_bytes": expected_bytes,
        "decoded_sha256": sha256_array(frames),
    }


def _double_decode(
    path: Path,
    *,
    spec: FfmpegCodecSpec,
    expected_frame_count: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    first, first_receipt = decode_stream(
        path,
        spec=spec,
        expected_frame_count=expected_frame_count,
        height=height,
        width=width,
    )
    second, second_receipt = decode_stream(
        path,
        spec=spec,
        expected_frame_count=expected_frame_count,
        height=height,
        width=width,
    )
    if first_receipt["decoded_sha256"] != second_receipt["decoded_sha256"] or not np.array_equal(first, second):
        raise LossySelectedPlaneCodecError("double decode was not byte-identical")
    return first, {
        "decode_argv": first_receipt["argv"],
        "decoded_bytes": first_receipt["decoded_bytes"],
        "decoded_sha256": first_receipt["decoded_sha256"],
        "repeat_decoded_sha256": second_receipt["decoded_sha256"],
        "double_decode_identical": True,
    }


def _endpoint_by_name(config: Mapping[str, Any], endpoint_name: str) -> Mapping[str, Any]:
    matches = [row for row in config["endpoints"] if row["name"] == endpoint_name]
    if len(matches) != 1:
        raise LossySelectedPlaneCodecError(f"unknown or duplicate endpoint: {endpoint_name}")
    return matches[0]


def _segment_bounds(config: Mapping[str, Any], segment_index: int) -> tuple[int, int]:
    count = config["segment_count"]
    if type(segment_index) is not int or not 0 <= segment_index < count:
        raise LossySelectedPlaneCodecError("segment index is out of bounds")
    start = segment_index * config["pairs_per_segment"]
    return start, start + config["pairs_per_segment"]


def _stage_paths(output_root: Path, endpoint_name: str, arm: str, segment_index: int) -> tuple[Path, Path]:
    stage = output_root / "stages" / endpoint_name / arm.lower() / f"segment_{segment_index:02d}"
    return stage, stage / "receipt.json"


def _validate_preserved_receipt(
    receipt_path: Path,
    *,
    config_sha256: str,
    endpoint_name: str,
    arm: str,
    segment_index: int,
) -> Mapping[str, Any]:
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise LossySelectedPlaneCodecError(f"cannot reopen preserved receipt: {receipt_path}") from exc
    if (
        receipt.get("schema") != SEGMENT_RECEIPT_SCHEMA
        or receipt.get("config_sha256") != config_sha256
        or receipt.get("endpoint") != endpoint_name
        or receipt.get("arm") != arm
        or receipt.get("segment_index") != segment_index
        or receipt.get("research_only") is not True
        or receipt.get("candidate_lineage_allowed") is not False
    ):
        raise LossySelectedPlaneCodecError(f"preserved receipt identity drifted: {receipt_path}")
    stage = receipt_path.parent
    streams = receipt.get("streams")
    if not isinstance(streams, list) or not streams:
        raise LossySelectedPlaneCodecError("preserved receipt has no streams")
    for stream in streams:
        if not isinstance(stream, Mapping):
            raise LossySelectedPlaneCodecError("preserved stream row malformed")
        relative = PurePosixPath(stream.get("relative_path", ""))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise LossySelectedPlaneCodecError("preserved stream relative path is unsafe")
        path = stage / relative.as_posix()
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != stream.get("bytes")
            or sha256_file(path) != stream.get("sha256")
        ):
            raise LossySelectedPlaneCodecError(f"preserved stream custody drifted: {path}")
    return receipt


def run_segment(
    config: Mapping[str, Any],
    source: SourcePlanes,
    *,
    output_root: Path | str,
    endpoint_name: str,
    arm: str,
    segment_index: int,
    ffmpeg_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    if arm not in ARMS:
        raise LossySelectedPlaneCodecError(f"unknown arm: {arm}")
    output = Path(output_root)
    config_sha = config_identity(config)
    stage, receipt_path = _stage_paths(output, endpoint_name, arm, segment_index)
    if receipt_path.exists():
        return _validate_preserved_receipt(
            receipt_path,
            config_sha256=config_sha,
            endpoint_name=endpoint_name,
            arm=arm,
            segment_index=segment_index,
        )
    if stage.exists() and any(stage.iterdir()):
        raise LossySelectedPlaneCodecError(f"incomplete non-empty stage requires operator inspection: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    start, stop = _segment_bounds(config, segment_index)
    frame0 = np.ascontiguousarray(source.frame0[start:stop])
    frame1 = np.ascontiguousarray(source.frame1[start:stop])
    expected_shape = (
        stop - start,
        config["geometry"]["height"],
        config["geometry"]["width"],
        config["geometry"]["channels"],
    )
    if frame0.shape != expected_shape or frame1.shape != expected_shape:
        raise LossySelectedPlaneCodecError("source segment shape/order drift")
    endpoint = _endpoint_by_name(config, endpoint_name)
    spec = codec_spec(config)
    scratch = Path(tempfile.mkdtemp(prefix=".scratch.", dir=stage))
    success = False
    stream_rows: list[Mapping[str, Any]] = []
    try:
        if arm == DIRECT_ARM:
            inputs = [
                (
                    DIRECT_STREAM,
                    direct_interleave(frame0, frame1),
                    endpoint["direct_bitrate_bps"],
                )
            ]
        else:
            inputs = [
                (BASE_STREAM, frame1, endpoint["base_bitrate_bps"]),
                (
                    ENHANCEMENT_STREAM,
                    conditional_enhancement(frame0, frame1),
                    endpoint["enhancement_bitrate_bps"],
                ),
            ]
        decoded_by_name: dict[str, np.ndarray] = {}
        for stream_name, frames, bitrate in inputs:
            temporary_path = scratch / f"{stream_name}.{spec.container}"
            encode_receipt = encode_stream(frames, spec=spec, bitrate_bps=bitrate, output_path=temporary_path)
            decoded, decode_receipt = _double_decode(
                temporary_path,
                spec=spec,
                expected_frame_count=len(frames),
                height=frames.shape[1],
                width=frames.shape[2],
            )
            final_path = stage / temporary_path.name
            os.replace(temporary_path, final_path)
            decoded_by_name[stream_name] = decoded
            stream_rows.append(
                {
                    "stream_name": stream_name,
                    "relative_path": final_path.name,
                    "source_frame_count": len(frames),
                    "source_sha256": sha256_array(frames),
                    "bytes": final_path.stat().st_size,
                    "sha256": sha256_file(final_path),
                    "encode": encode_receipt,
                    "decode": decode_receipt,
                }
            )
        if arm == DIRECT_ARM:
            reconstructed0, reconstructed1 = direct_deinterleave(decoded_by_name[DIRECT_STREAM])
        else:
            reconstructed1 = decoded_by_name[BASE_STREAM]
            reconstructed0 = reconstruct_frame0(reconstructed1, decoded_by_name[ENHANCEMENT_STREAM])
        if reconstructed0.shape != frame0.shape or reconstructed1.shape != frame1.shape:
            raise LossySelectedPlaneCodecError("reconstructed segment shape/order drift")
        receipt: Mapping[str, Any] = {
            "schema": SEGMENT_RECEIPT_SCHEMA,
            "experiment_schema": SCHEMA,
            "research_only": True,
            "historical_c1_encoder_oracle_only": True,
            "candidate_lineage_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "config_sha256": config_sha,
            "endpoint": endpoint_name,
            "arm": arm,
            "segment_index": segment_index,
            "pair_start": start,
            "pair_stop": stop,
            "pair_count": stop - start,
            "geometry": config["geometry"],
            "source": {
                "archive_sha256": source.archive_sha256,
                "member_sha256": source.member_sha256,
                "full_frame0_sha256": source.frame0_sha256,
                "full_frame1_sha256": source.frame1_sha256,
                "segment_frame0_sha256": sha256_array(frame0),
                "segment_frame1_sha256": sha256_array(frame1),
            },
            "codec": dict(config["codec"]),
            "ffmpeg": dict(ffmpeg_receipt),
            "transform": (
                "direct-interleaved-y0-y1.v1"
                if arm == DIRECT_ARM
                else "y1-base-plus-centered-signed-diff-q2-y0-given-y1.v1"
            ),
            "streams": stream_rows,
            "stream_bytes_total": sum(row["bytes"] for row in stream_rows),
            "reconstructed_frame0_sha256": sha256_array(reconstructed0),
            "reconstructed_frame1_sha256": sha256_array(reconstructed1),
            "reconstructed_frame0_sse": int(
                np.square(reconstructed0.astype(np.int32) - frame0.astype(np.int32), dtype=np.int64).sum()
            ),
            "reconstructed_frame1_sse": int(
                np.square(reconstructed1.astype(np.int32) - frame1.astype(np.int32), dtype=np.int64).sum()
            ),
            "double_decode_identical": all(row["decode"]["double_decode_identical"] for row in stream_rows),
            "status": "closed",
        }
        write_once_or_equal(receipt_path, canonical_json(receipt))
        success = True
        return receipt
    except BaseException as exc:
        artifacts: list[Mapping[str, Any]] = []
        for path in sorted(scratch.rglob("*")):
            if path.is_file() and not path.is_symlink():
                artifacts.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        failure = {
            "schema": FAILURE_RECEIPT_SCHEMA,
            "config_sha256": config_sha,
            "endpoint": endpoint_name,
            "arm": arm,
            "segment_index": segment_index,
            "scratch_path": str(scratch),
            "preserved_artifacts": artifacts,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "rebuildable": True,
            "rebuild_command_source": "typed experiment config and runner argv",
            "reason_preserved": "failed stage retained for forensic custody",
        }
        write_once_or_equal(stage / "failure_receipt.json", canonical_json(failure))
        raise
    finally:
        if success:
            shutil.rmtree(scratch)


def decode_closed_segment_planes(
    config: Mapping[str, Any],
    *,
    codec_output_root: Path | str,
    endpoint_name: str,
    arm: str,
    segment_index: int,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Reopen one persisted codec stage and recover its exact decoded planes."""

    if arm not in ARMS:
        raise LossySelectedPlaneCodecError(f"unknown arm: {arm}")
    root = Path(codec_output_root)
    stage, receipt_path = _stage_paths(root, endpoint_name, arm, segment_index)
    receipt = _validate_preserved_receipt(
        receipt_path,
        config_sha256=config_identity(config),
        endpoint_name=endpoint_name,
        arm=arm,
        segment_index=segment_index,
    )
    spec = codec_spec(config)
    decoded: dict[str, np.ndarray] = {}
    decode_rows: list[Mapping[str, Any]] = []
    for stream in receipt["streams"]:
        path = stage / stream["relative_path"]
        frames, reopened = decode_stream(
            path,
            spec=spec,
            expected_frame_count=stream["source_frame_count"],
            height=config["geometry"]["height"],
            width=config["geometry"]["width"],
        )
        if reopened["decoded_sha256"] != stream["decode"]["decoded_sha256"]:
            raise LossySelectedPlaneCodecError("scoring bridge decoded bytes differ from closed codec stage")
        decoded[stream["stream_name"]] = frames
        decode_rows.append(
            {
                "stream_name": stream["stream_name"],
                "path": str(path),
                "bytes": stream["bytes"],
                "sha256": stream["sha256"],
                "decoded_sha256": reopened["decoded_sha256"],
                "decode_argv": reopened["argv"],
            }
        )
    if arm == DIRECT_ARM:
        frame0, frame1 = direct_deinterleave(decoded[DIRECT_STREAM])
    else:
        frame1 = decoded[BASE_STREAM]
        frame0 = reconstruct_frame0(frame1, decoded[ENHANCEMENT_STREAM])
    if (
        sha256_array(frame0) != receipt["reconstructed_frame0_sha256"]
        or sha256_array(frame1) != receipt["reconstructed_frame1_sha256"]
    ):
        raise LossySelectedPlaneCodecError("scoring bridge reconstructed plane hashes drifted")
    return frame0, frame1, {
        "codec_stage_receipt_path": str(receipt_path),
        "codec_stage_receipt_sha256": sha256_file(receipt_path),
        "streams": decode_rows,
        "reconstructed_frame0_sha256": sha256_array(frame0),
        "reconstructed_frame1_sha256": sha256_array(frame1),
    }


def _validate_raw_bridge_stage(
    receipt_path: Path,
    *,
    config_sha256: str,
    endpoint_name: str,
    arm: str,
    segment_index: int,
) -> Mapping[str, Any]:
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise LossySelectedPlaneCodecError("cannot reopen raw-bridge stage receipt") from exc
    if (
        receipt.get("schema") != RAW_BRIDGE_STAGE_SCHEMA
        or receipt.get("config_sha256") != config_sha256
        or receipt.get("endpoint") != endpoint_name
        or receipt.get("arm") != arm
        or receipt.get("segment_index") != segment_index
    ):
        raise LossySelectedPlaneCodecError("raw-bridge stage receipt identity drifted")
    raw_path = receipt_path.parent / receipt.get("raw_file", "")
    if (
        raw_path.is_symlink()
        or not raw_path.is_file()
        or raw_path.stat().st_size != receipt.get("raw_bytes")
        or sha256_file(raw_path) != receipt.get("raw_sha256")
    ):
        raise LossySelectedPlaneCodecError("raw-bridge preserved stage bytes drifted")
    return receipt


def materialize_v10_scoring_raw(
    config: Mapping[str, Any],
    *,
    codec_output_root: Path | str,
    endpoint_name: str,
    arm: str,
    output_root: Path | str,
) -> Mapping[str, Any]:
    """Decode one arm through the strict V10 factor-2 receiver into n600 raw."""

    if config["test_only_small_fixture"]:
        raise LossySelectedPlaneCodecError("scoring raw bridge refuses test-only fixtures")
    if arm not in ARMS:
        raise LossySelectedPlaneCodecError(f"unknown arm: {arm}")
    _endpoint_by_name(config, endpoint_name)
    output = Path(output_root)
    preflight = storage_preflight(
        output,
        required_bytes=9 << 30,
        test_only_small_fixture=False,
        allow_local_storage=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    config_sha = config_identity(config)
    write_once_or_equal(output / "storage_preflight.json", canonical_json(preflight))
    packet = _load_historical_c1_packet(config)
    geometry = packet.header["geometry"]
    camera_h = geometry["camera_height"]
    camera_w = geometry["camera_width"]
    channels = geometry["channels"]
    pair_frame_bytes = camera_h * camera_w * channels * 2
    expected_segment_bytes = config["pairs_per_segment"] * pair_frame_bytes
    stage_root = output / "raw_stages"
    stage_receipts: list[Mapping[str, Any]] = []
    for segment_index in range(config["segment_count"]):
        stage_dir = stage_root / f"segment_{segment_index:02d}"
        stage_receipt_path = stage_dir / "receipt.json"
        if stage_receipt_path.exists():
            stage_receipts.append(
                _validate_raw_bridge_stage(
                    stage_receipt_path,
                    config_sha256=config_sha,
                    endpoint_name=endpoint_name,
                    arm=arm,
                    segment_index=segment_index,
                )
            )
            continue
        if stage_dir.exists() and any(stage_dir.iterdir()):
            raise LossySelectedPlaneCodecError("incomplete raw-bridge stage requires operator inspection")
        stage_dir.mkdir(parents=True, exist_ok=True)
        frame0, frame1, decode_custody = decode_closed_segment_planes(
            config,
            codec_output_root=codec_output_root,
            endpoint_name=endpoint_name,
            arm=arm,
            segment_index=segment_index,
        )
        raw_path = stage_dir / "pairs.raw"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".pairs.", suffix=".partial", dir=stage_dir)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        numerator_values = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for local_index in range(len(frame0)):
                    camera0, count0 = realize_pair_frame1(packet, frame0[local_index])
                    camera1, count1 = realize_pair_frame1(packet, frame1[local_index])
                    payload0 = camera0.tobytes(order="C")
                    payload1 = camera1.tobytes(order="C")
                    handle.write(payload0)
                    handle.write(payload1)
                    digest.update(payload0)
                    digest.update(payload1)
                    numerator_values += count0 + count1
                handle.flush()
                os.fsync(handle.fileno())
            if temporary.stat().st_size != expected_segment_bytes:
                raise LossySelectedPlaneCodecError("raw-bridge segment byte count drifted")
            os.replace(temporary, raw_path)
        finally:
            temporary.unlink(missing_ok=True)
        start, stop = _segment_bounds(config, segment_index)
        receipt: Mapping[str, Any] = {
            "schema": RAW_BRIDGE_STAGE_SCHEMA,
            "research_only": True,
            "candidate_lineage_allowed": False,
            "score_claim": False,
            "config_sha256": config_sha,
            "endpoint": endpoint_name,
            "arm": arm,
            "segment_index": segment_index,
            "pair_start": start,
            "pair_stop": stop,
            "raw_file": raw_path.name,
            "raw_bytes": raw_path.stat().st_size,
            "raw_sha256": digest.hexdigest(),
            "camera_geometry": [camera_h, camera_w, channels],
            "numerator_values_verified": numerator_values,
            "decode_custody": decode_custody,
            "receiver": "tac.witness_dsl.v10_production_receiver.realize_pair_frame1",
        }
        write_once_or_equal(stage_receipt_path, canonical_json(receipt))
        stage_receipts.append(receipt)

    raw_path = output / "0.raw"
    expected_raw_bytes = config["pair_count"] * pair_frame_bytes
    if raw_path.exists():
        if raw_path.is_symlink() or not raw_path.is_file() or raw_path.stat().st_size != expected_raw_bytes:
            raise LossySelectedPlaneCodecError("preserved assembled raw has wrong custody")
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".0.raw.", suffix=".partial", dir=output)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination:
                for segment_index, _receipt in enumerate(stage_receipts):
                    stage_receipt_path = stage_root / f"segment_{segment_index:02d}" / "receipt.json"
                    validated = _validate_raw_bridge_stage(
                        stage_receipt_path,
                        config_sha256=config_sha,
                        endpoint_name=endpoint_name,
                        arm=arm,
                        segment_index=segment_index,
                    )
                    stage_raw = stage_receipt_path.parent / validated["raw_file"]
                    with stage_raw.open("rb") as source:
                        shutil.copyfileobj(source, destination, length=8 << 20)
                destination.flush()
                os.fsync(destination.fileno())
            if temporary.stat().st_size != expected_raw_bytes:
                raise LossySelectedPlaneCodecError("assembled scoring raw byte count drifted")
            os.replace(temporary, raw_path)
        finally:
            temporary.unlink(missing_ok=True)
    codec_aggregate_path = Path(codec_output_root) / "aggregate_receipt.json"
    try:
        codec_aggregate = json.loads(codec_aggregate_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise LossySelectedPlaneCodecError("cannot reopen codec aggregate for raw bridge") from exc
    bundles = [row for row in codec_aggregate["bundles"] if row["arm"] == arm]
    if len(bundles) != 1:
        raise LossySelectedPlaneCodecError("codec aggregate arm binding drifted")
    bundle_path = Path(bundles[0]["bundle_path"])
    if sha256_file(bundle_path) != bundles[0]["bundle_sha256"]:
        raise LossySelectedPlaneCodecError("codec bundle custody drifted before scoring bridge")
    bridge: Mapping[str, Any] = {
        "schema": RAW_BRIDGE_RECEIPT_SCHEMA,
        "research_only": True,
        "historical_c1_encoder_oracle_only": True,
        "candidate_lineage_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "config_sha256": config_sha,
        "endpoint": endpoint_name,
        "arm": arm,
        "codec_aggregate_path": str(codec_aggregate_path),
        "codec_aggregate_sha256": sha256_file(codec_aggregate_path),
        "diagnostic_bundle_path": str(bundle_path),
        "diagnostic_bundle_bytes": bundles[0]["bundle_bytes"],
        "diagnostic_bundle_sha256": bundles[0]["bundle_sha256"],
        "raw_path": str(raw_path),
        "raw_bytes": raw_path.stat().st_size,
        "raw_sha256": sha256_file(raw_path),
        "pair_count": config["pair_count"],
        "camera_geometry": [camera_h, camera_w, channels],
        "stage_receipts": [
            {
                "path": str(stage_root / f"segment_{index:02d}" / "receipt.json"),
                "sha256": sha256_file(stage_root / f"segment_{index:02d}" / "receipt.json"),
                "raw_sha256": receipt["raw_sha256"],
            }
            for index, receipt in enumerate(stage_receipts)
        ],
        "receiver": "tac.witness_dsl.v10_production_receiver.realize_pair_frame1",
        "status": "closed",
    }
    write_once_or_equal(output / "bridge_receipt.json", canonical_json(bridge))
    return bridge


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _publish_exact_bundle(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    files: Sequence[tuple[str, Path, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
            archive.writestr(_zip_info("manifest.json"), canonical_json(manifest))
            for name, source_path, _ in files:
                archive.writestr(_zip_info(name), source_path.read_bytes())
        if path.exists():
            if sha256_file(path) != sha256_file(temporary):
                raise LossySelectedPlaneCodecError(f"preserved diagnostic bundle drifted: {path}")
            temporary.unlink()
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_arm_bundle(
    config: Mapping[str, Any],
    *,
    output_root: Path | str,
    endpoint_name: str,
    arm: str,
) -> Mapping[str, Any]:
    if arm not in ARMS:
        raise LossySelectedPlaneCodecError(f"unknown arm: {arm}")
    output = Path(output_root)
    config_sha = config_identity(config)
    receipts = [
        _validate_preserved_receipt(
            _stage_paths(output, endpoint_name, arm, index)[1],
            config_sha256=config_sha,
            endpoint_name=endpoint_name,
            arm=arm,
            segment_index=index,
        )
        for index in range(config["segment_count"])
    ]
    expected = 0
    files: list[tuple[str, Path, Mapping[str, Any]]] = []
    for receipt in receipts:
        if receipt["pair_start"] != expected or receipt["pair_stop"] != expected + config["pairs_per_segment"]:
            raise LossySelectedPlaneCodecError("segment receipts do not form one contiguous ordered cover")
        expected = receipt["pair_stop"]
        stage = _stage_paths(output, endpoint_name, arm, receipt["segment_index"])[0]
        for stream in receipt["streams"]:
            archive_name = f"segments/{receipt['segment_index']:02d}/{stream['relative_path']}"
            stream_with_segment = dict(stream)
            stream_with_segment["segment_index"] = receipt["segment_index"]
            files.append((archive_name, stage / stream["relative_path"], stream_with_segment))
    if expected != config["pair_count"]:
        raise LossySelectedPlaneCodecError("segment receipts do not cover all configured pairs")
    manifest: Mapping[str, Any] = {
        "schema": BUNDLE_MANIFEST_SCHEMA,
        "experiment_schema": SCHEMA,
        "research_only": True,
        "historical_c1_encoder_oracle_only": True,
        "candidate_lineage_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "config_sha256": config_sha,
        "endpoint": endpoint_name,
        "arm": arm,
        "pair_count": config["pair_count"],
        "pairs_per_segment": config["pairs_per_segment"],
        "segment_count": config["segment_count"],
        "rate_ceiling_bytes": config["rate_ceiling_bytes"],
        "codec": dict(config["codec"]),
        "streams": [
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "decoded_sha256": row["decode"]["decoded_sha256"],
            }
            for name, path, row in files
        ],
        "payload_stream_bytes": sum(path.stat().st_size for _, path, _ in files),
    }
    bundle_dir = output / "bundles" / endpoint_name
    bundle_path = bundle_dir / f"{arm.lower()}.diagnostic.zip"
    published = _publish_exact_bundle(bundle_path, manifest=manifest, files=files)
    bundle_bytes = published["bytes"]

    action_specs: list[tuple[str, list[tuple[str, Path, Mapping[str, Any]]], str, bool]] = []
    for segment_index in range(config["segment_count"]):
        action_specs.append(
            (
                f"evict_entire_segment_{segment_index:02d}",
                [row for row in files if row[2]["segment_index"] != segment_index],
                f"all streams in pair segment {segment_index} absent",
                False,
            )
        )
    if arm == LAYERED_ARM:
        action_specs.append(
            (
                "enhancement_off_all",
                [row for row in files if row[2]["stream_name"] != ENHANCEMENT_STREAM],
                "Y0 reconstructed by typed fallback Y0_hat=Y1_hat for all pairs",
                True,
            )
        )
        for segment_index in range(config["segment_count"]):
            action_specs.extend(
                [
                    (
                        f"evict_enhancement_segment_{segment_index:02d}",
                        [
                            row
                            for row in files
                            if not (
                                row[2]["segment_index"] == segment_index
                                and row[2]["stream_name"] == ENHANCEMENT_STREAM
                            )
                        ],
                        f"Y0 uses typed fallback Y0_hat=Y1_hat in pair segment {segment_index}",
                        True,
                    ),
                    (
                        f"evict_base_segment_{segment_index:02d}",
                        [
                            row
                            for row in files
                            if not (
                                row[2]["segment_index"] == segment_index
                                and row[2]["stream_name"] == BASE_STREAM
                            )
                        ],
                        f"Y1 base absent in pair segment {segment_index}; decoder fallback not selected",
                        False,
                    ),
                ]
            )
    operating_points: list[Mapping[str, Any]] = [
        {
            "action_id": "full",
            "exact_bundle_path": published["path"],
            "exact_bundle_bytes": published["bytes"],
            "exact_bundle_sha256": published["sha256"],
            "rate_only_proposal": False,
            "decoder_closed": True,
            "scorer_measured": False,
            "semantic_effect": "all streams retained",
            "included_stream_count": len(files),
            "conditional_planning_only": {
                "sub0172_max_bytes": CONDITIONAL_PLANNING_MAX_SUB0172_BYTES,
                "sub015_max_bytes": CONDITIONAL_PLANNING_MAX_SUB015_BYTES,
                "below_sub0172_coordinate": published["bytes"] <= CONDITIONAL_PLANNING_MAX_SUB0172_BYTES,
                "below_sub015_coordinate": published["bytes"] <= CONDITIONAL_PLANNING_MAX_SUB015_BYTES,
                "rejection_gate": False,
                "score_authority": False,
            },
        }
    ]
    for action_id, selected_files, semantic_effect, decoder_closed in action_specs:
        action_manifest: Mapping[str, Any] = {
            **manifest,
            "rate_only_proposal": True,
            "decoder_closed": decoder_closed,
            "scorer_measured": False,
            "action_id": action_id,
            "semantic_effect": semantic_effect,
            "streams": [
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "decoded_sha256": row["decode"]["decoded_sha256"],
                }
                for name, path, row in selected_files
            ],
            "payload_stream_bytes": sum(path.stat().st_size for _, path, _ in selected_files),
        }
        action_path = bundle_dir / "operating_points" / arm.lower() / f"{action_id}.diagnostic.zip"
        action_published = _publish_exact_bundle(action_path, manifest=action_manifest, files=selected_files)
        operating_points.append(
            {
                "action_id": action_id,
                "exact_bundle_path": action_published["path"],
                "exact_bundle_bytes": action_published["bytes"],
                "exact_bundle_sha256": action_published["sha256"],
                "rate_only_proposal": True,
                "decoder_closed": decoder_closed,
                "scorer_measured": False,
                "semantic_effect": semantic_effect,
                "included_stream_count": len(selected_files),
                "conditional_planning_only": {
                    "sub0172_max_bytes": CONDITIONAL_PLANNING_MAX_SUB0172_BYTES,
                    "sub015_max_bytes": CONDITIONAL_PLANNING_MAX_SUB015_BYTES,
                    "below_sub0172_coordinate": (
                        action_published["bytes"] <= CONDITIONAL_PLANNING_MAX_SUB0172_BYTES
                    ),
                    "below_sub015_coordinate": action_published["bytes"] <= CONDITIONAL_PLANNING_MAX_SUB015_BYTES,
                    "rejection_gate": False,
                    "score_authority": False,
                },
            }
        )
    return {
        "arm": arm,
        "bundle_path": str(bundle_path),
        "bundle_bytes": bundle_bytes,
        "bundle_sha256": published["sha256"],
        "payload_stream_bytes": manifest["payload_stream_bytes"],
        "container_and_manifest_bytes": bundle_bytes - manifest["payload_stream_bytes"],
        "rate_ceiling_bytes": config["rate_ceiling_bytes"],
        "rate_ceiling_pass": bundle_bytes <= config["rate_ceiling_bytes"],
        "operating_points": operating_points,
    }


def run_full_experiment(config: Mapping[str, Any], *, output_root: Path | str) -> Mapping[str, Any]:
    output = Path(output_root)
    repository_root = Path(__file__).resolve().parents[3]
    planning_anchor_path = repository_root / CANONICAL_PLANNING_ANCHOR_PATH
    if (
        planning_anchor_path.is_symlink()
        or not planning_anchor_path.is_file()
        or sha256_file(planning_anchor_path) != CANONICAL_PLANNING_ANCHOR_SHA256
    ):
        raise LossySelectedPlaneCodecError("canonical C1 batch16 planning anchor custody drifted")
    preflight = storage_preflight(
        output,
        required_bytes=config["required_free_bytes"],
        test_only_small_fixture=config["test_only_small_fixture"],
        allow_local_storage=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_once_or_equal(output / "config.json", canonical_json(config))
    write_once_or_equal(output / "storage_preflight.json", canonical_json(preflight))
    ffmpeg_receipt = ffmpeg_version(codec_spec(config))
    write_once_or_equal(output / "ffmpeg_version.json", canonical_json(ffmpeg_receipt))
    source = load_historical_c1_source(config)
    source_receipt = {
        "research_only": True,
        "historical_c1_encoder_oracle_only": True,
        "candidate_lineage_allowed": False,
        "archive_sha256": source.archive_sha256,
        "member_sha256": source.member_sha256,
        "frame0_sha256": source.frame0_sha256,
        "frame1_sha256": source.frame1_sha256,
        "shape": list(source.frame0.shape),
        "dtype": str(source.frame0.dtype),
    }
    write_once_or_equal(output / "source_receipt.json", canonical_json(source_receipt))
    bundle_rows: list[Mapping[str, Any]] = []
    for endpoint in config["endpoints"]:
        for arm in ARMS:
            for segment_index in range(config["segment_count"]):
                run_segment(
                    config,
                    source,
                    output_root=output,
                    endpoint_name=endpoint["name"],
                    arm=arm,
                    segment_index=segment_index,
                    ffmpeg_receipt=ffmpeg_receipt,
                )
            bundle_rows.append(
                build_arm_bundle(config, output_root=output, endpoint_name=endpoint["name"], arm=arm)
            )
    aggregate: Mapping[str, Any] = {
        "schema": AGGREGATE_RECEIPT_SCHEMA,
        "experiment_schema": SCHEMA,
        "research_only": True,
        "historical_c1_encoder_oracle_only": True,
        "candidate_lineage_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "config_sha256": config_identity(config),
        "pair_count": config["pair_count"],
        "segment_count": config["segment_count"],
        "pairs_per_segment": config["pairs_per_segment"],
        "source": source_receipt,
        "ffmpeg": ffmpeg_receipt,
        "bundles": bundle_rows,
        "all_rate_endpoints_pass": all(row["rate_ceiling_pass"] for row in bundle_rows),
        "canonical_conditional_planning_anchor": {
            "path": CANONICAL_PLANNING_ANCHOR_PATH,
            "sha256": CANONICAL_PLANNING_ANCHOR_SHA256,
            "d_seg": CANONICAL_PLANNING_DSEG,
            "d_pose": CANONICAL_PLANNING_DPOSE,
            "sub0172_max_bytes": CONDITIONAL_PLANNING_MAX_SUB0172_BYTES,
            "sub015_max_bytes": CONDITIONAL_PLANNING_MAX_SUB015_BYTES,
            "role": "planning coordinate only",
            "rejection_gate": False,
            "score_authority_for_this_lossy_output": False,
        },
        "pointer_delta": "UNMOVED",
        "next_authority_gate": "fresh current own-lineage planes plus exact upstream/evaluate.py n600",
        "status": "closed",
    }
    write_once_or_equal(output / "aggregate_receipt.json", canonical_json(aggregate))
    return aggregate


__all__ = [
    "AGGREGATE_RECEIPT_SCHEMA",
    "ARMS",
    "BASE_STREAM",
    "BUNDLE_MANIFEST_SCHEMA",
    "CANONICAL_PLANNING_ANCHOR_PATH",
    "CANONICAL_PLANNING_ANCHOR_SHA256",
    "CANONICAL_PLANNING_DPOSE",
    "CANONICAL_PLANNING_DSEG",
    "CONDITIONAL_PLANNING_MAX_SUB015_BYTES",
    "CONDITIONAL_PLANNING_MAX_SUB0172_BYTES",
    "CONFIG_SCHEMA",
    "DIRECT_ARM",
    "DIRECT_STREAM",
    "ENHANCEMENT_STREAM",
    "FAILURE_RECEIPT_SCHEMA",
    "HISTORICAL_C1_ARCHIVE_SHA256",
    "HISTORICAL_C1_MEMBER_SHA256",
    "LAYERED_ARM",
    "RATE_CEILING_BYTES",
    "RAW_BRIDGE_RECEIPT_SCHEMA",
    "RAW_BRIDGE_STAGE_SCHEMA",
    "SCHEMA",
    "SEGMENT_RECEIPT_SCHEMA",
    "FfmpegCodecSpec",
    "LossySelectedPlaneCodecError",
    "SourcePlanes",
    "build_arm_bundle",
    "canonical_json",
    "codec_spec",
    "conditional_enhancement",
    "config_identity",
    "decode_closed_segment_planes",
    "decode_stream",
    "direct_deinterleave",
    "direct_interleave",
    "encode_stream",
    "ffmpeg_version",
    "load_config",
    "load_historical_c1_source",
    "materialize_v10_scoring_raw",
    "reconstruct_frame0",
    "run_full_experiment",
    "run_segment",
    "sha256_array",
    "sha256_bytes",
    "sha256_file",
    "write_once_or_equal",
]
