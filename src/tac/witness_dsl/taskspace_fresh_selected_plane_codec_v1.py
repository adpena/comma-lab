# SPDX-License-Identifier: MIT
"""Fresh full-n600 selected-plane codec with resumable long-stream recode.

The codec consumes an injected, strictly validated own-lineage operand loader.
It does not know how the operands were produced and cannot fall back to a
historical archive.  Five 120-pair encoder checkpoints are retained for crash
recovery, while the counted result is recoded from the fresh operands into two
chronological whole-population streams.

This is the DIRECT_TASK_LAYERED representation:

    base = Y1
    enhancement = clip(round((Y0 - Y1) / 2) + 128)

It is not PROGRAM_RESIDUAL_LAYERED.  That representation requires a fresh
semantic predictor/base byte stream supplied by its producer; equality with an
older plane hash is neither required nor forbidden.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.c0b_semantic_quotient import storage_preflight
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetError,
    load_dynamic_frontier_target,
    verify_dynamic_frontier_target_snapshot,
)
from tac.witness_dsl.taskspace_lossy_selected_plane_codec_v1 import (
    FfmpegCodecSpec,
    LossySelectedPlaneCodecError,
    canonical_json,
    conditional_enhancement,
    encode_argv,
    encode_stream,
    ffmpeg_version,
    reconstruct_frame0,
    sha256_array,
    sha256_bytes,
    sha256_file,
    write_once_or_equal,
)

SCHEMA = "taskspace_fresh_selected_plane_codec.v1"
CONFIG_SCHEMA = "taskspace_fresh_selected_plane_codec_config.v1"
STAGE_SCHEMA = "taskspace_fresh_selected_plane_codec_stage.v1"
FINAL_SCHEMA = "taskspace_fresh_selected_plane_codec_final_recode.v1"
BUNDLE_SCHEMA = "taskspace_fresh_selected_plane_stream_bundle.v1"
AGGREGATE_SCHEMA = "taskspace_fresh_selected_plane_codec_aggregate.v1"
FAILURE_SCHEMA = "taskspace_fresh_selected_plane_codec_failure.v1"

PAIR_COUNT = 600
PAIRS_PER_STAGE = 120
STAGE_COUNT = 5
HEIGHT = 384
WIDTH = 512
CHANNELS = 3
FPS = 20
CAMERA_HEIGHT = 874
CAMERA_WIDTH = 1164
PUBLIC_PYAV_VERSION = "17.0.0"
UPSTREAM_LOCK_RELATIVE_PATH = Path("upstream/uv.lock")

REPRESENTATION_MODE = "DIRECT_TASK_LAYERED"
PROGRAM_RESIDUAL_MODE = "PROGRAM_RESIDUAL_LAYERED"
BASE_STREAM = "y1_base"
ENHANCEMENT_STREAM = "y0_given_y1_q2"
TRANSFORM_ID = "y1-base-plus-centered-signed-diff-q2-y0-given-y1.v1"

HISTORICAL_C1_ARCHIVE_SHA256 = "e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42"
HISTORICAL_C1_MEMBER_SHA256 = "aa1dbb5e2efff28cd0d31f5ee2a4b0575a248a27a431151bfcae64eb320d385b"
_FORBIDDEN_INPUT_PATH_TOKENS = ("historical", "/c1_", "/v15_", "prepared_plane", "precomputed_plane")

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "research_only",
        "candidate_lineage_allowed",
        "historical_payload_reused",
        "pair_count",
        "pairs_per_stage",
        "stage_count",
        "geometry",
        "representation",
        "operand_provider",
        "codec",
        "endpoint",
        "required_free_bytes",
        "test_only_small_fixture",
    }
)
_GEOMETRY_FIELDS = frozenset({"height", "width", "channels"})
_REPRESENTATION_FIELDS = frozenset(
    {
        "mode",
        "program_residual_layered_available",
        "program_residual_layered_blocker",
    }
)
_PROVIDER_FIELDS = frozenset({"aggregate_receipt_path", "aggregate_receipt_sha256"})
_CODEC_FIELDS = frozenset(
    {
        "ffmpeg_executable",
        "ffmpeg_executable_bytes",
        "ffmpeg_executable_sha256",
        "ffmpeg_version_line",
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
_ENDPOINT_FIELDS = frozenset({"name", "base_bitrate_bps", "enhancement_bitrate_bps"})


class FreshSelectedPlaneCodecError(LossySelectedPlaneCodecError):
    """Fail-closed config, lineage, operand, codec, or resume error."""


@runtime_checkable
class FreshOperandProviderV1(Protocol):
    """Structural API implemented by the fresh scorer-plane operand loader."""

    def iter_stages(self, *, max_pairs: int = PAIRS_PER_STAGE) -> Iterator[object]:
        """Yield the exact chronological operand stages."""


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise FreshSelectedPlaneCodecError(f"{label} must be a boolean")
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FreshSelectedPlaneCodecError(f"{label} must be an integer >= {minimum}")
    return value


def _exact_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FreshSelectedPlaneCodecError(f"{label} must be a non-empty string")
    return value


def _require_sha(value: object, label: str) -> str:
    text = _exact_str(value, label).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise FreshSelectedPlaneCodecError(f"{label} must be a lowercase SHA-256")
    return text


def _require_fields(value: object, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        raise FreshSelectedPlaneCodecError(f"{label} has missing or unknown fields")
    return value


def _reject_historical_input_reference(path: str, sha256: str) -> None:
    normalized = path.lower()
    if any(token in normalized for token in _FORBIDDEN_INPUT_PATH_TOKENS):
        raise FreshSelectedPlaneCodecError("operand provider path names a forbidden historical input")
    if sha256 in {HISTORICAL_C1_ARCHIVE_SHA256, HISTORICAL_C1_MEMBER_SHA256}:
        raise FreshSelectedPlaneCodecError("operand provider identity equals a forbidden historical payload")


def _preserve_first_storage_preflight(path: Path, current: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep the first admission receipt stable while rechecking space on resume."""

    if not path.exists():
        write_once_or_equal(path, canonical_json(current))
        return current
    try:
        preserved = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshSelectedPlaneCodecError("preserved storage preflight is unreadable") from exc
    if (
        not isinstance(preserved, Mapping)
        or preserved.get("passed") is not True
        or preserved.get("required_bytes") != current.get("required_bytes")
        or preserved.get("storage_root") != current.get("storage_root")
    ):
        raise FreshSelectedPlaneCodecError("preserved storage preflight identity drift")
    return preserved


def validate_config(value: object) -> Mapping[str, Any]:
    config = _require_fields(value, _TOP_LEVEL_FIELDS, "config")
    if config["schema"] != CONFIG_SCHEMA:
        raise FreshSelectedPlaneCodecError("config schema drift")
    if _exact_bool(config["research_only"], "research_only") is not True:
        raise FreshSelectedPlaneCodecError("pre-evaluation codec output must remain research_only")
    if _exact_bool(config["candidate_lineage_allowed"], "candidate_lineage_allowed") is not True:
        raise FreshSelectedPlaneCodecError("fresh codec config must preserve candidate-capable lineage")
    if _exact_bool(config["historical_payload_reused"], "historical_payload_reused") is not False:
        raise FreshSelectedPlaneCodecError("historical payload reuse is forbidden")

    test_only = _exact_bool(config["test_only_small_fixture"], "test_only_small_fixture")
    pair_count = _exact_int(config["pair_count"], "pair_count", minimum=1)
    pairs_per_stage = _exact_int(config["pairs_per_stage"], "pairs_per_stage", minimum=1)
    stage_count = _exact_int(config["stage_count"], "stage_count", minimum=1)
    if pair_count != pairs_per_stage * stage_count:
        raise FreshSelectedPlaneCodecError("stage lattice does not exactly cover pair_count")
    if not test_only and (pair_count, pairs_per_stage, stage_count) != (
        PAIR_COUNT,
        PAIRS_PER_STAGE,
        STAGE_COUNT,
    ):
        raise FreshSelectedPlaneCodecError("production codec requires 600 pairs in five 120-pair stages")

    geometry = _require_fields(config["geometry"], _GEOMETRY_FIELDS, "geometry")
    shape = tuple(_exact_int(geometry[field], f"geometry.{field}", minimum=1) for field in _GEOMETRY_FIELDS)
    if not test_only and (geometry["height"], geometry["width"], geometry["channels"]) != (
        HEIGHT,
        WIDTH,
        CHANNELS,
    ):
        raise FreshSelectedPlaneCodecError("production scorer geometry must be 384x512x3")
    if geometry["channels"] != CHANNELS:
        raise FreshSelectedPlaneCodecError("codec requires three semantic color channels")
    del shape

    representation = _require_fields(config["representation"], _REPRESENTATION_FIELDS, "representation")
    if representation["mode"] != REPRESENTATION_MODE:
        raise FreshSelectedPlaneCodecError("this codec implements DIRECT_TASK_LAYERED only")
    if (
        _exact_bool(
            representation["program_residual_layered_available"],
            "representation.program_residual_layered_available",
        )
        is not False
    ):
        raise FreshSelectedPlaneCodecError("PROGRAM_RESIDUAL_LAYERED cannot be claimed without semantic base bytes")
    blocker = _exact_str(
        representation["program_residual_layered_blocker"],
        "representation.program_residual_layered_blocker",
    )
    if "fresh semantic" not in blocker.lower():
        raise FreshSelectedPlaneCodecError("program-residual blocker must name the missing fresh semantic base")

    provider = _require_fields(config["operand_provider"], _PROVIDER_FIELDS, "operand_provider")
    provider_path = _exact_str(provider["aggregate_receipt_path"], "operand_provider.aggregate_receipt_path")
    provider_sha = _require_sha(
        provider["aggregate_receipt_sha256"],
        "operand_provider.aggregate_receipt_sha256",
    )
    _reject_historical_input_reference(provider_path, provider_sha)

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
    ffmpeg_path = Path(codec["ffmpeg_executable"])
    if not ffmpeg_path.is_absolute() or ffmpeg_path != ffmpeg_path.resolve():
        raise FreshSelectedPlaneCodecError(
            "codec.ffmpeg_executable must be a resolved absolute encoder path"
        )
    _exact_int(codec["ffmpeg_executable_bytes"], "codec.ffmpeg_executable_bytes", minimum=1)
    _require_sha(codec["ffmpeg_executable_sha256"], "codec.ffmpeg_executable_sha256")
    if not _exact_str(codec["ffmpeg_version_line"], "codec.ffmpeg_version_line").startswith(
        "ffmpeg version "
    ):
        raise FreshSelectedPlaneCodecError("codec.ffmpeg_version_line has invalid syntax")
    identity = (codec["encoder"], codec["container"], codec["encoded_pixel_format"])
    if identity not in {
        ("libx265", "hevc", "yuv444p"),
        ("libx264rgb", "h264", "rgb24"),
    }:
        raise FreshSelectedPlaneCodecError("fresh candidate race supports x265-444 or x264rgb only")
    if (codec["input_pixel_format"], codec["decoded_pixel_format"]) != ("rgb24", "rgb24"):
        raise FreshSelectedPlaneCodecError("input and decoded pixel formats must be explicit rgb24")
    if _exact_int(codec["frame_rate"], "codec.frame_rate", minimum=1) != FPS:
        raise FreshSelectedPlaneCodecError("frame rate must remain 20 fps")
    if _exact_int(codec["threads"], "codec.threads", minimum=1) != 1:
        raise FreshSelectedPlaneCodecError("deterministic encoder contract requires one thread")
    if not isinstance(codec["preset"], str) or codec["preset"] not in {
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
        raise FreshSelectedPlaneCodecError("x264/x265 preset is unsupported")

    endpoint = _require_fields(config["endpoint"], _ENDPOINT_FIELDS, "endpoint")
    _exact_str(endpoint["name"], "endpoint.name")
    _exact_int(endpoint["base_bitrate_bps"], "endpoint.base_bitrate_bps", minimum=1)
    _exact_int(endpoint["enhancement_bitrate_bps"], "endpoint.enhancement_bitrate_bps", minimum=1)
    _exact_int(config["required_free_bytes"], "required_free_bytes", minimum=1)
    return config


def load_config(path: Path | str) -> Mapping[str, Any]:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshSelectedPlaneCodecError(f"cannot load config: {config_path}") from exc
    return validate_config(value)


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


def resolve_ffmpeg_binary_identity(executable: str = "ffmpeg") -> Mapping[str, Any]:
    """Resolve and hash-close the encoder binary before a config is sealed."""

    resolved = shutil.which(executable)
    if resolved is None:
        raise FreshSelectedPlaneCodecError(f"ffmpeg executable not found: {executable}")
    path = Path(resolved).resolve()
    if not path.is_file():
        raise FreshSelectedPlaneCodecError("resolved ffmpeg executable is not a regular file")
    spec = FfmpegCodecSpec(
        executable=str(path),
        encoder="identity-probe-only",
        container="identity-probe-only",
        input_pixel_format="identity-probe-only",
        encoded_pixel_format="identity-probe-only",
        decoded_pixel_format="identity-probe-only",
        frame_rate=FPS,
        preset="identity-probe-only",
        threads=1,
        color_range="identity-probe-only",
        colorspace="identity-probe-only",
        color_primaries="identity-probe-only",
        color_transfer="identity-probe-only",
    )
    version = ffmpeg_version(spec)
    return {
        "ffmpeg_executable": str(path),
        "ffmpeg_executable_bytes": path.stat().st_size,
        "ffmpeg_executable_sha256": sha256_file(path),
        "ffmpeg_version_line": version["version_line"],
    }


def verify_ffmpeg_binary_identity(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed if the sealed encoder binary changed before launch or resume."""

    codec = config["codec"]
    path = Path(codec["ffmpeg_executable"])
    if path.is_symlink() or not path.is_file():
        raise FreshSelectedPlaneCodecError("sealed ffmpeg executable is unavailable or symlinked")
    current = resolve_ffmpeg_binary_identity(str(path))
    expected = {
        field: codec[field]
        for field in (
            "ffmpeg_executable",
            "ffmpeg_executable_bytes",
            "ffmpeg_executable_sha256",
            "ffmpeg_version_line",
        )
    }
    if current != expected:
        raise FreshSelectedPlaneCodecError("sealed ffmpeg executable identity drift")
    return {
        "resolved_executable": current["ffmpeg_executable"],
        "executable_bytes": current["ffmpeg_executable_bytes"],
        "executable_sha256": current["ffmpeg_executable_sha256"],
        "version_line": current["ffmpeg_version_line"],
    }


def pyav_version_receipt(*, required_version: str | None = None) -> Mapping[str, Any]:
    """Return the exact public-runtime decoder library identity."""

    try:
        import av
    except ImportError as exc:
        raise FreshSelectedPlaneCodecError("PyAV is required by the public decode contract") from exc
    if required_version is not None and av.__version__ != required_version:
        raise FreshSelectedPlaneCodecError(
            f"PyAV version mismatch: required {required_version}, observed {av.__version__}"
        )
    return {
        "module": "av",
        "version": av.__version__,
        "library_versions": {
            name: list(version)
            for name, version in sorted(av.library_versions.items())
        },
        "decode_path": (
            "av.open->container.decode(video=0)->"
            "native-gbrp-extract-or-VideoFrame.to_ndarray(rgb24)"
        ),
        "thread_count": 1,
        "authority": "public-runtime parse-back",
        "required_public_version": required_version,
    }


def upstream_pyav_lock_receipt(repo_root: Path | str) -> Mapping[str, Any]:
    """Derive the public PyAV pin from the authoritative upstream lock."""

    path = Path(repo_root) / UPSTREAM_LOCK_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise FreshSelectedPlaneCodecError("authoritative upstream uv.lock is unavailable")
    try:
        lock = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise FreshSelectedPlaneCodecError("authoritative upstream uv.lock is unreadable") from exc
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise FreshSelectedPlaneCodecError("authoritative upstream lock package table is missing")
    rows = [row for row in packages if isinstance(row, Mapping) and row.get("name") == "av"]
    if len(rows) != 1 or not isinstance(rows[0].get("version"), str):
        raise FreshSelectedPlaneCodecError("authoritative upstream lock has no unique PyAV pin")
    version = rows[0]["version"]
    if version != PUBLIC_PYAV_VERSION:
        raise FreshSelectedPlaneCodecError(
            f"compiled public PyAV contract drifted from upstream lock: {version}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "package": "av",
        "version": version,
    }


def _native_gbrp_to_rgb(frame: object, *, height: int, width: int) -> np.ndarray:
    """Extract FFmpeg GBR planes without a libswscale color conversion."""

    planes = getattr(frame, "planes", ())
    if len(planes) != 3:
        raise FreshSelectedPlaneCodecError("native gbrp frame must expose exactly three planes")
    channels = []
    for plane in planes:
        line_size = int(plane.line_size)
        if line_size < width:
            raise FreshSelectedPlaneCodecError("native gbrp plane stride is smaller than frame width")
        values = np.frombuffer(plane, dtype=np.uint8)
        if values.size < height * line_size:
            raise FreshSelectedPlaneCodecError("native gbrp plane buffer is truncated")
        channels.append(values[: height * line_size].reshape(height, line_size)[:, :width])
    # FFmpeg's gbrp plane order is G, B, R.
    return np.ascontiguousarray(np.stack((channels[2], channels[0], channels[1]), axis=-1))


def decode_stream_pyav(
    input_path: Path,
    *,
    expected_frame_count: int,
    height: int,
    width: int,
    required_version: str | None = None,
    require_native_gbrp: bool = False,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Decode through the exact dependency available to public ``inflate.py``."""

    try:
        import av
    except ImportError as exc:
        raise FreshSelectedPlaneCodecError("PyAV is required by the public decode contract") from exc
    version = pyav_version_receipt(required_version=required_version)
    if input_path.is_symlink() or not input_path.is_file():
        raise FreshSelectedPlaneCodecError("PyAV input must be a regular non-symlink stream")
    frames: list[np.ndarray] = []
    native_pixel_formats: set[str] = set()
    conversion_paths: set[str] = set()
    codec_name = ""
    try:
        with av.open(str(input_path), mode="r") as container:
            video_streams = list(container.streams.video)
            if len(video_streams) != 1:
                raise FreshSelectedPlaneCodecError("counted stream must contain exactly one video stream")
            stream = video_streams[0]
            stream.codec_context.thread_count = 1
            codec_name = str(stream.codec_context.name)
            for frame in container.decode(video=0):
                native_format = str(frame.format.name)
                native_pixel_formats.add(native_format)
                if require_native_gbrp:
                    if native_format != "gbrp":
                        raise FreshSelectedPlaneCodecError(
                            f"x264rgb public decode expected native gbrp, observed {native_format}"
                        )
                    rgb = _native_gbrp_to_rgb(frame, height=height, width=width)
                    conversion_paths.add("native-gbrp-plane-extraction-and-rgb-reorder.v1")
                else:
                    rgb = frame.to_ndarray(format="rgb24")
                    conversion_paths.add("PyAV-VideoFrame.to_ndarray-rgb24")
                if rgb.dtype != np.uint8 or rgb.shape != (height, width, CHANNELS):
                    raise FreshSelectedPlaneCodecError("PyAV decoded frame geometry or dtype drift")
                frames.append(np.ascontiguousarray(rgb))
    except FreshSelectedPlaneCodecError:
        raise
    except Exception as exc:
        raise FreshSelectedPlaneCodecError(f"PyAV refused counted stream: {input_path}") from exc
    if len(frames) != expected_frame_count:
        raise FreshSelectedPlaneCodecError(
            f"PyAV frame count drift: expected {expected_frame_count}, received {len(frames)}"
        )
    decoded = np.stack(frames, axis=0)
    return decoded, {
        "decoder": version,
        "stream_probe": {
            "actual_codec_name": codec_name,
            "actual_native_pixel_formats": sorted(native_pixel_formats),
            "rgb_conversion_paths": sorted(conversion_paths),
        },
        "frame_count": len(frames),
        "height": height,
        "width": width,
        "decoded_bytes": decoded.nbytes,
        "decoded_sha256": sha256_array(decoded),
    }


def _field(stage: object, name: str) -> object:
    if isinstance(stage, Mapping):
        if name not in stage:
            raise FreshSelectedPlaneCodecError(f"operand stage lacks {name}")
        return stage[name]
    try:
        return getattr(stage, name)
    except AttributeError as exc:
        raise FreshSelectedPlaneCodecError(f"operand stage lacks {name}") from exc


def _validated_stage(
    stage: object,
    *,
    expected_start: int,
    config: Mapping[str, Any],
) -> tuple[tuple[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_range = _field(stage, "pair_range")
    if (
        not isinstance(pair_range, (tuple, list))
        or len(pair_range) != 2
        or type(pair_range[0]) is not int
        or type(pair_range[1]) is not int
    ):
        raise FreshSelectedPlaneCodecError("operand pair_range must be two exact integers")
    start, stop = pair_range
    if start != expected_start or stop <= start or stop - start > config["pairs_per_stage"]:
        raise FreshSelectedPlaneCodecError("operand stages are not an ordered bounded contiguous cover")

    pair_ids = np.asarray(_field(stage, "pair_ids"))
    if pair_ids.shape != (stop - start,) or not np.issubdtype(pair_ids.dtype, np.integer):
        raise FreshSelectedPlaneCodecError("operand pair_ids shape/dtype drift")
    if not np.array_equal(pair_ids.astype(np.int64, copy=False), np.arange(start, stop, dtype=np.int64)):
        raise FreshSelectedPlaneCodecError("operand pair_ids are not canonical chronological ids")

    expected_plane_shape = (
        stop - start,
        config["geometry"]["height"],
        config["geometry"]["width"],
        config["geometry"]["channels"],
    )
    y0 = np.asarray(_field(stage, "y0_u8"))
    y1 = np.asarray(_field(stage, "y1_u8"))
    labels = np.asarray(_field(stage, "target_labels_u8"))
    poses = np.asarray(_field(stage, "gt_poses_f32"))
    if y0.dtype != np.uint8 or y1.dtype != np.uint8 or y0.shape != expected_plane_shape or y1.shape != expected_plane_shape:
        raise FreshSelectedPlaneCodecError("operand Y0/Y1 shape or dtype drift")
    if labels.dtype != np.uint8 or labels.shape != expected_plane_shape[:3]:
        raise FreshSelectedPlaneCodecError("fresh target-label shape or dtype drift")
    if poses.dtype != np.float32 or poses.shape != (stop - start, 6):
        raise FreshSelectedPlaneCodecError("source-cache advisory pose shape or dtype drift")
    return (
        (start, stop),
        np.ascontiguousarray(y0),
        np.ascontiguousarray(y1),
        np.ascontiguousarray(labels),
        np.ascontiguousarray(poses),
    )


def iter_validated_stages(
    provider: FreshOperandProviderV1,
    config: Mapping[str, Any],
) -> Iterator[tuple[tuple[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not isinstance(provider, FreshOperandProviderV1):
        raise FreshSelectedPlaneCodecError("operand provider lacks iter_stages(max_pairs=...)")
    expected_start = 0
    count = 0
    try:
        stages = provider.iter_stages(max_pairs=config["pairs_per_stage"])
    except TypeError as exc:
        raise FreshSelectedPlaneCodecError("operand provider API drift") from exc
    for stage in stages:
        validated = _validated_stage(stage, expected_start=expected_start, config=config)
        expected_start = validated[0][1]
        count += 1
        yield validated
    if expected_start != config["pair_count"] or count != config["stage_count"]:
        raise FreshSelectedPlaneCodecError("operand provider does not yield the exact full population")


def _double_decode_pyav(
    path: Path,
    *,
    expected_frame_count: int,
    height: int,
    width: int,
    required_version: str | None,
    require_native_gbrp: bool,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    first, first_receipt = decode_stream_pyav(
        path,
        expected_frame_count=expected_frame_count,
        height=height,
        width=width,
        required_version=required_version,
        require_native_gbrp=require_native_gbrp,
    )
    second, second_receipt = decode_stream_pyav(
        path,
        expected_frame_count=expected_frame_count,
        height=height,
        width=width,
        required_version=required_version,
        require_native_gbrp=require_native_gbrp,
    )
    if (
        first_receipt["decoded_sha256"] != second_receipt["decoded_sha256"]
        or first_receipt["stream_probe"] != second_receipt["stream_probe"]
        or not np.array_equal(first, second)
    ):
        raise FreshSelectedPlaneCodecError("double decode was not byte-identical")
    del second
    return first, {
        "decoder": first_receipt["decoder"],
        "stream_probe": first_receipt["stream_probe"],
        "decoded_bytes": first_receipt["decoded_bytes"],
        "decoded_sha256": first_receipt["decoded_sha256"],
        "repeat_decoded_sha256": second_receipt["decoded_sha256"],
        "double_decode_identical": True,
    }


def _stream_inputs(
    y0: np.ndarray,
    y1: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[tuple[str, np.ndarray, int], tuple[str, np.ndarray, int]]:
    return (
        (BASE_STREAM, y1, config["endpoint"]["base_bitrate_bps"]),
        (
            ENHANCEMENT_STREAM,
            conditional_enhancement(y0, y1),
            config["endpoint"]["enhancement_bitrate_bps"],
        ),
    )


def _validate_preserved_stage(
    path: Path,
    *,
    config_sha256: str,
    operand_receipt_sha256: str,
    stage_index: int,
    pair_range: tuple[int, int],
    operand_hashes: Mapping[str, str],
) -> Mapping[str, Any]:
    try:
        receipt = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshSelectedPlaneCodecError(f"cannot reopen stage receipt: {path}") from exc
    if (
        receipt.get("schema") != STAGE_SCHEMA
        or receipt.get("config_sha256") != config_sha256
        or receipt.get("operand_receipt_sha256") != operand_receipt_sha256
        or receipt.get("stage_index") != stage_index
        or receipt.get("pair_start") != pair_range[0]
        or receipt.get("pair_stop") != pair_range[1]
        or receipt.get("fresh_operand_hashes") != operand_hashes
        or receipt.get("status") != "closed"
    ):
        raise FreshSelectedPlaneCodecError("preserved stage identity drift")
    for row in receipt.get("streams", []):
        stream_path = path.parent / row.get("relative_path", "")
        if (
            stream_path.is_symlink()
            or not stream_path.is_file()
            or stream_path.stat().st_size != row.get("bytes")
            or sha256_file(stream_path) != row.get("sha256")
        ):
            raise FreshSelectedPlaneCodecError("preserved stage stream custody drift")
    return receipt


def _run_stage(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    stage_index: int,
    pair_range: tuple[int, int],
    y0: np.ndarray,
    y1: np.ndarray,
    labels: np.ndarray,
    poses: np.ndarray,
    ffmpeg_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    stage_dir = output_root / "stages" / f"stage_{stage_index:02d}"
    receipt_path = stage_dir / "receipt.json"
    config_sha = config_identity(config)
    operand_sha = config["operand_provider"]["aggregate_receipt_sha256"]
    operand_hashes = {
        "y0_u8": sha256_array(y0),
        "y1_u8": sha256_array(y1),
        "target_labels_u8": sha256_array(labels),
        "gt_poses_f32": sha256_array(poses),
    }
    if receipt_path.exists():
        return _validate_preserved_stage(
            receipt_path,
            config_sha256=config_sha,
            operand_receipt_sha256=operand_sha,
            stage_index=stage_index,
            pair_range=pair_range,
            operand_hashes=operand_hashes,
        )
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise FreshSelectedPlaneCodecError(f"incomplete non-empty stage requires inspection: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".scratch.", dir=stage_dir))
    success = False
    try:
        streams = []
        spec = codec_spec(config)
        for stream_name, frames, bitrate in _stream_inputs(y0, y1, config):
            temporary = scratch / f"{stream_name}.{spec.container}"
            encode_receipt = encode_stream(
                frames,
                spec=spec,
                bitrate_bps=bitrate,
                output_path=temporary,
            )
            decoded, decode_receipt = _double_decode_pyav(
                temporary,
                expected_frame_count=len(frames),
                height=frames.shape[1],
                width=frames.shape[2],
                required_version=None if config["test_only_small_fixture"] else PUBLIC_PYAV_VERSION,
                require_native_gbrp=config["codec"]["encoder"] == "libx264rgb",
            )
            final = stage_dir / temporary.name
            os.replace(temporary, final)
            streams.append(
                {
                    "stream_name": stream_name,
                    "relative_path": final.name,
                    "source_sha256": sha256_array(frames),
                    "decoded_sha256": sha256_array(decoded),
                    "bytes": final.stat().st_size,
                    "sha256": sha256_file(final),
                    "encode": encode_receipt,
                    "decode": decode_receipt,
                }
            )
        receipt: Mapping[str, Any] = {
            "schema": STAGE_SCHEMA,
            "experiment_schema": SCHEMA,
            "status": "closed",
            "research_only": True,
            "candidate_lineage_allowed": True,
            "historical_payload_reused": False,
            "score_claim": False,
            "promotion_eligible": False,
            "config_sha256": config_sha,
            "operand_receipt_sha256": operand_sha,
            "stage_index": stage_index,
            "pair_start": pair_range[0],
            "pair_stop": pair_range[1],
            "pair_count": pair_range[1] - pair_range[0],
            "representation_mode": REPRESENTATION_MODE,
            "program_residual_layered": {
                "available": False,
                "status": "blocked_missing_fresh_semantic_base_bytes",
            },
            "pose_custody": "SEALED_SOURCE_CACHE_ADVISORY_ONLY",
            "pose_authority": False,
            "fresh_operand_hashes": operand_hashes,
            "codec": dict(config["codec"]),
            "ffmpeg": dict(ffmpeg_receipt),
            "streams": streams,
            "stream_bytes_total": sum(row["bytes"] for row in streams),
        }
        write_once_or_equal(receipt_path, canonical_json(receipt))
        success = True
        return receipt
    except BaseException as exc:
        artifacts = []
        for path in sorted(scratch.rglob("*")):
            if path.is_file() and not path.is_symlink():
                artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        failure = {
            "schema": FAILURE_SCHEMA,
            "config_sha256": config_sha,
            "operand_receipt_sha256": operand_sha,
            "stage_index": stage_index,
            "scratch_path": str(scratch),
            "preserved_artifacts": artifacts,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "rebuildable": True,
            "reason_preserved": "failed stage retained for forensic custody",
        }
        write_once_or_equal(stage_dir / "failure_receipt.json", canonical_json(failure))
        raise
    finally:
        if success:
            shutil.rmtree(scratch)


def _encode_whole_population_stream(
    provider: FreshOperandProviderV1,
    config: Mapping[str, Any],
    *,
    stream_name: str,
    output_path: Path,
) -> Mapping[str, Any]:
    spec = codec_spec(config)
    bitrate = (
        config["endpoint"]["base_bitrate_bps"]
        if stream_name == BASE_STREAM
        else config["endpoint"]["enhancement_bitrate_bps"]
    )
    argv = encode_argv(
        spec,
        width=config["geometry"]["width"],
        height=config["geometry"]["height"],
        frame_count=config["pair_count"],
        bitrate_bps=bitrate,
        output_path=output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    assert proc.stderr is not None
    source_digest = hashlib.sha256()
    frame_count = 0
    try:
        for _, y0, y1, _, _ in iter_validated_stages(provider, config):
            frames = y1 if stream_name == BASE_STREAM else conditional_enhancement(y0, y1)
            source_digest.update(memoryview(frames))
            for frame in frames:
                proc.stdin.write(memoryview(frame))
                frame_count += 1
        proc.stdin.close()
        stderr = proc.stderr.read()
        returncode = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if frame_count != config["pair_count"]:
        raise FreshSelectedPlaneCodecError("whole-population recode frame count drift")
    if returncode:
        raise FreshSelectedPlaneCodecError(
            f"whole-population ffmpeg encode failed rc={returncode}: "
            f"{stderr.decode('utf-8', 'replace')[:1000]}"
        )
    if output_path.is_symlink() or not output_path.is_file() or output_path.stat().st_size <= 0:
        raise FreshSelectedPlaneCodecError("whole-population encoder did not produce a regular stream")
    return {
        "stream_name": stream_name,
        "argv": argv,
        "bitrate_bps": bitrate,
        "gop_frames": config["pair_count"],
        "frame_count": frame_count,
        "source_sha256": source_digest.hexdigest(),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (0o100644 << 16)
    return info


def _public_raw_contract(
    config: Mapping[str, Any],
    reconstructed_y0: np.ndarray,
    reconstructed_y1: np.ndarray,
) -> Mapping[str, Any]:
    """Hash the exact chronological camera bytes expanded from PyAV planes."""

    if config["test_only_small_fixture"]:
        camera_height = config["geometry"]["height"] * 2
        camera_width = config["geometry"]["width"] * 2
    else:
        camera_height = CAMERA_HEIGHT
        camera_width = CAMERA_WIDTH
    operator = DisjointResizeOperator.build(
        camera_h=camera_height,
        camera_w=camera_width,
        scorer_h=config["geometry"]["height"],
        scorer_w=config["geometry"]["width"],
    )
    digest = hashlib.sha256()
    frame_count = 0
    for y0, y1 in zip(reconstructed_y0, reconstructed_y1, strict=True):
        for scorer_plane in (y0, y1):
            frame = realize_factor2_uint8_scorer_plane(operator, scorer_plane)
            digest.update(memoryview(frame))
            frame_count += 1
    expected_bytes = (
        config["pair_count"] * 2 * camera_height * camera_width * config["geometry"]["channels"]
    )
    if frame_count != config["pair_count"] * 2:
        raise FreshSelectedPlaneCodecError("public raw frame count drift")
    helper_path = Path(inspect.getsourcefile(realize_factor2_uint8_scorer_plane) or "")
    if not helper_path.is_file():
        raise FreshSelectedPlaneCodecError("factor2 public helper source is unavailable")
    return {
        "receiver_contract_id": "factor2-disjoint-half-pixel-uint8.v1",
        "realization_helper": (
            "tac.optimization.uint8_lattice_feasibility."
            "realize_factor2_uint8_scorer_plane"
        ),
        "realization_helper_source_sha256": sha256_file(helper_path),
        "chronological_order": "pair_id ascending, frame0 from Y0 then frame1 from Y1",
        "camera_geometry": [camera_height, camera_width, config["geometry"]["channels"]],
        "frame_count": frame_count,
        "expected_raw_bytes": expected_bytes,
        "expected_raw_sha256": digest.hexdigest(),
        "source_planes": "PyAV-authoritative decoded and conditionally reconstructed bytes",
    }


def _publish_bundle(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    stream_rows: list[Mapping[str, Any]],
    reconstructed_y0_sha256: str,
    reconstructed_y1_sha256: str,
    public_raw_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    final_dir = output_root / "final"
    members = {
        f"streams/{row['stream_name']}.{config['codec']['container']}": (
            final_dir / f"{row['stream_name']}.{config['codec']['container']}"
        )
        for row in stream_rows
    }
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "pair_count": config["pair_count"],
        "geometry": dict(config["geometry"]),
        "representation_mode": REPRESENTATION_MODE,
        "transform_id": TRANSFORM_ID,
        "program_residual_layered": {
            "available": False,
            "status": "blocked_missing_fresh_semantic_base_bytes",
            "v15_composition_claim": False,
        },
        "historical_payload_reused": False,
        "freshness_rule": "fresh derivation and input custody; output hash novelty is not required",
        "config_sha256": config_identity(config),
        "operand_receipt_sha256": config["operand_provider"]["aggregate_receipt_sha256"],
        "external_stage_checkpoint_receipt_sha256": [
            sha256_file(output_root / "stages" / f"stage_{index:02d}" / "receipt.json")
            for index in range(config["stage_count"])
        ],
        "public_decode": pyav_version_receipt(
            required_version=None if config["test_only_small_fixture"] else PUBLIC_PYAV_VERSION
        ),
        "pose_custody": "SEALED_SOURCE_CACHE_ADVISORY_ONLY",
        "pose_authority": False,
        "expected_reconstructed_scorer_planes": {
            "y0_u8_sha256": reconstructed_y0_sha256,
            "y1_u8_sha256": reconstructed_y1_sha256,
        },
        "public_raw_contract": dict(public_raw_contract),
        "streams": [
            {
                "path": name,
                "stream_name": row["stream_name"],
                "frame_count": row["frame_count"],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "decoded_sha256": row["decode"]["decoded_sha256"],
                "requested_encoder": config["codec"]["encoder"],
                "requested_encoded_pixel_format": config["codec"]["encoded_pixel_format"],
                "actual_codec_name": row["decode"]["stream_probe"]["actual_codec_name"],
                "actual_native_pixel_formats": row["decode"]["stream_probe"][
                    "actual_native_pixel_formats"
                ],
                "rgb_conversion_paths": row["decode"]["stream_probe"]["rgb_conversion_paths"],
                "public_output_pixel_format": config["codec"]["decoded_pixel_format"],
            }
            for row in stream_rows
            for name, path in members.items()
            if name.startswith(f"streams/{row['stream_name']}.")
        ],
    }
    manifest_payload = canonical_json(manifest)
    bundle_path = final_dir / "counted_stream_bundle.zip"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".bundle.", suffix=".zip.tmp", dir=final_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            archive.writestr(_zip_info("manifest.json"), manifest_payload)
            for name in sorted(members):
                archive.writestr(_zip_info(name), members[name].read_bytes())
        os.replace(temporary, bundle_path)
    finally:
        temporary.unlink(missing_ok=True)
    payload_bytes = sum(path.stat().st_size for path in members.values())
    return {
        "path": str(bundle_path),
        "bytes": bundle_path.stat().st_size,
        "sha256": sha256_file(bundle_path),
        "payload_stream_bytes": payload_bytes,
        "container_and_manifest_bytes": bundle_path.stat().st_size - payload_bytes,
        "manifest_sha256": sha256_bytes(manifest_payload),
        "manifest": manifest,
    }


def _run_final_recode(
    provider: FreshOperandProviderV1,
    config: Mapping[str, Any],
    *,
    output_root: Path,
) -> Mapping[str, Any]:
    final_dir = output_root / "final"
    receipt_path = final_dir / "receipt.json"
    config_sha = config_identity(config)
    operand_sha = config["operand_provider"]["aggregate_receipt_sha256"]
    if receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise FreshSelectedPlaneCodecError("cannot reopen final recode receipt") from exc
        if (
            receipt.get("schema") != FINAL_SCHEMA
            or receipt.get("config_sha256") != config_sha
            or receipt.get("operand_receipt_sha256") != operand_sha
            or receipt.get("status") != "closed"
        ):
            raise FreshSelectedPlaneCodecError("preserved final recode identity drift")
        for row in receipt.get("streams", []):
            path = final_dir / row.get("relative_path", "")
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != row.get("bytes")
                or sha256_file(path) != row.get("sha256")
            ):
                raise FreshSelectedPlaneCodecError("preserved final stream custody drift")
        bundle = receipt.get("bundle", {})
        bundle_path = Path(bundle.get("path", ""))
        if (
            bundle_path.is_symlink()
            or not bundle_path.is_file()
            or bundle_path.stat().st_size != bundle.get("bytes")
            or sha256_file(bundle_path) != bundle.get("sha256")
        ):
            raise FreshSelectedPlaneCodecError("preserved counted bundle custody drift")
        return receipt
    if final_dir.exists() and any(final_dir.iterdir()):
        raise FreshSelectedPlaneCodecError("incomplete non-empty final recode requires operator inspection")
    final_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".scratch.", dir=final_dir))
    success = False
    try:
        spec = codec_spec(config)
        stream_rows = []
        decoded: dict[str, np.ndarray] = {}
        for stream_name in (BASE_STREAM, ENHANCEMENT_STREAM):
            temporary = scratch / f"{stream_name}.{spec.container}"
            encode_receipt = _encode_whole_population_stream(
                provider,
                config,
                stream_name=stream_name,
                output_path=temporary,
            )
            frames, decode_receipt = _double_decode_pyav(
                temporary,
                expected_frame_count=config["pair_count"],
                height=config["geometry"]["height"],
                width=config["geometry"]["width"],
                required_version=None if config["test_only_small_fixture"] else PUBLIC_PYAV_VERSION,
                require_native_gbrp=config["codec"]["encoder"] == "libx264rgb",
            )
            final = final_dir / temporary.name
            os.replace(temporary, final)
            decoded[stream_name] = frames
            stream_rows.append(
                {
                    **encode_receipt,
                    "relative_path": final.name,
                    "decode": decode_receipt,
                }
            )

        reconstructed_y1 = decoded[BASE_STREAM]
        reconstructed_y0 = reconstruct_frame0(reconstructed_y1, decoded[ENHANCEMENT_STREAM])
        y0_digest = hashlib.sha256()
        y1_digest = hashlib.sha256()
        labels_digest = hashlib.sha256()
        poses_digest = hashlib.sha256()
        y0_sse = 0
        y1_sse = 0
        for (start, stop), y0, y1, labels, poses in iter_validated_stages(provider, config):
            y0_digest.update(memoryview(y0))
            y1_digest.update(memoryview(y1))
            labels_digest.update(memoryview(labels))
            poses_digest.update(memoryview(poses))
            y0_delta = reconstructed_y0[start:stop].astype(np.int32) - y0.astype(np.int32)
            y1_delta = reconstructed_y1[start:stop].astype(np.int32) - y1.astype(np.int32)
            y0_sse += int(np.square(y0_delta, dtype=np.int64).sum())
            y1_sse += int(np.square(y1_delta, dtype=np.int64).sum())

        reconstructed_y0_sha = sha256_array(reconstructed_y0)
        reconstructed_y1_sha = sha256_array(reconstructed_y1)
        public_raw_contract = _public_raw_contract(config, reconstructed_y0, reconstructed_y1)
        bundle = _publish_bundle(
            config,
            output_root=output_root,
            stream_rows=stream_rows,
            reconstructed_y0_sha256=reconstructed_y0_sha,
            reconstructed_y1_sha256=reconstructed_y1_sha,
            public_raw_contract=public_raw_contract,
        )
        receipt: Mapping[str, Any] = {
            "schema": FINAL_SCHEMA,
            "experiment_schema": SCHEMA,
            "status": "closed",
            "research_only": True,
            "candidate_lineage_allowed": True,
            "historical_payload_reused": False,
            "score_claim": False,
            "promotion_eligible": False,
            "config_sha256": config_sha,
            "operand_receipt_sha256": operand_sha,
            "pair_count": config["pair_count"],
            "representation_mode": REPRESENTATION_MODE,
            "program_residual_layered": {
                "available": False,
                "status": "blocked_missing_fresh_semantic_base_bytes",
                "v15_composition_claim": False,
            },
            "pose_custody": "SEALED_SOURCE_CACHE_ADVISORY_ONLY",
            "pose_authority": False,
            "fresh_operand_hashes": {
                "y0_u8": y0_digest.hexdigest(),
                "y1_u8": y1_digest.hexdigest(),
                "target_labels_u8": labels_digest.hexdigest(),
                "gt_poses_f32": poses_digest.hexdigest(),
            },
            "streams": stream_rows,
            "stream_bytes_total": sum(row["bytes"] for row in stream_rows),
            "reconstructed_y0_sha256": reconstructed_y0_sha,
            "reconstructed_y1_sha256": reconstructed_y1_sha,
            "reconstructed_y0_sse": y0_sse,
            "reconstructed_y1_sse": y1_sse,
            "public_raw_contract": public_raw_contract,
            "bundle": bundle,
        }
        write_once_or_equal(receipt_path, canonical_json(receipt))
        success = True
        return receipt
    except BaseException as exc:
        artifacts = []
        for path in sorted(scratch.rglob("*")):
            if path.is_file() and not path.is_symlink():
                artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        write_once_or_equal(
            final_dir / "failure_receipt.json",
            canonical_json(
                {
                    "schema": FAILURE_SCHEMA,
                    "config_sha256": config_sha,
                    "operand_receipt_sha256": operand_sha,
                    "stage_index": "final_recode",
                    "scratch_path": str(scratch),
                    "preserved_artifacts": artifacts,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "rebuildable": True,
                    "reason_preserved": "failed final recode retained for forensic custody",
                }
            ),
        )
        raise
    finally:
        if success:
            shutil.rmtree(scratch)


def run_full_experiment(
    config: Mapping[str, Any],
    provider: FreshOperandProviderV1,
    *,
    output_root: Path | str,
    repo_root: Path | str | None = None,
) -> Mapping[str, Any]:
    config = validate_config(config)
    output = Path(output_root)
    repository = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    try:
        frontier = load_dynamic_frontier_target(repo_root=repository)
        verify_dynamic_frontier_target_snapshot(frontier)
    except DynamicFrontierTargetError as exc:
        raise FreshSelectedPlaneCodecError(f"dynamic canonical frontier unavailable: {exc}") from exc
    upstream_pyav_lock = upstream_pyav_lock_receipt(repository)

    if not config["test_only_small_fixture"]:
        receipt_path = Path(config["operand_provider"]["aggregate_receipt_path"])
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise FreshSelectedPlaneCodecError("fresh operand aggregate receipt must be a regular file")
        if sha256_file(receipt_path) != config["operand_provider"]["aggregate_receipt_sha256"]:
            raise FreshSelectedPlaneCodecError("fresh operand aggregate receipt SHA-256 mismatch")
    preflight = storage_preflight(
        output,
        required_bytes=config["required_free_bytes"],
        test_only_small_fixture=config["test_only_small_fixture"],
        allow_local_storage=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_once_or_equal(output / "config.json", canonical_json(config))
    _preserve_first_storage_preflight(output / "storage_preflight.json", preflight)
    ffmpeg_receipt = verify_ffmpeg_binary_identity(config)
    write_once_or_equal(output / "ffmpeg_version.json", canonical_json(ffmpeg_receipt))
    pyav_receipt = pyav_version_receipt(
        required_version=None if config["test_only_small_fixture"] else PUBLIC_PYAV_VERSION
    )
    write_once_or_equal(output / "pyav_version.json", canonical_json(pyav_receipt))

    stage_rows = []
    for stage_index, (pair_range, y0, y1, labels, poses) in enumerate(iter_validated_stages(provider, config)):
        stage_rows.append(
            _run_stage(
                config,
                output_root=output,
                stage_index=stage_index,
                pair_range=pair_range,
                y0=y0,
                y1=y1,
                labels=labels,
                poses=poses,
                ffmpeg_receipt=ffmpeg_receipt,
            )
        )
    final = _run_final_recode(provider, config, output_root=output)
    verify_dynamic_frontier_target_snapshot(frontier)
    frontier_record = asdict(frontier)
    aggregate: Mapping[str, Any] = {
        "schema": AGGREGATE_SCHEMA,
        "experiment_schema": SCHEMA,
        "status": "closed_pending_public_receiver_and_exact_eval",
        "research_only": True,
        "candidate_lineage_allowed": True,
        "historical_payload_reused": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_delta": "UNMOVED",
        "config_sha256": config_identity(config),
        "pair_count": config["pair_count"],
        "stage_count": len(stage_rows),
        "stage_receipt_sha256": [
            sha256_file(output / "stages" / f"stage_{index:02d}" / "receipt.json")
            for index in range(len(stage_rows))
        ],
        "operand_provider": dict(config["operand_provider"]),
        "representation_mode": REPRESENTATION_MODE,
        "program_residual_layered": {
            "available": False,
            "status": "blocked_missing_fresh_semantic_base_bytes",
            "v15_composition_claim": False,
        },
        "pose_custody": "SEALED_SOURCE_CACHE_ADVISORY_ONLY",
        "pose_authority": False,
        "codec": dict(config["codec"]),
        "public_decode": pyav_receipt,
        "upstream_pyav_lock": upstream_pyav_lock,
        "endpoint": dict(config["endpoint"]),
        "final_recode_receipt_sha256": sha256_file(output / "final" / "receipt.json"),
        "counted_stream_bundle": final["bundle"],
        "rate_term_if_used_as_exact_archive": (
            25.0 * final["bundle"]["bytes"] / 37_545_489.0
        ),
        "dynamic_frontier": frontier_record,
        "admission_rule": "exact public score must be strictly below dynamic_frontier.target_score",
        "next_authority_gate": (
            "generic V10 public inflate closure, double-inflate identity, then upstream/evaluate.py n600"
        ),
    }
    write_once_or_equal(output / "aggregate_receipt.json", canonical_json(aggregate))
    return aggregate


__all__ = [
    "AGGREGATE_SCHEMA",
    "BASE_STREAM",
    "BUNDLE_SCHEMA",
    "CONFIG_SCHEMA",
    "ENHANCEMENT_STREAM",
    "FINAL_SCHEMA",
    "PROGRAM_RESIDUAL_MODE",
    "REPRESENTATION_MODE",
    "SCHEMA",
    "STAGE_SCHEMA",
    "TRANSFORM_ID",
    "FreshOperandProviderV1",
    "FreshSelectedPlaneCodecError",
    "codec_spec",
    "config_identity",
    "decode_stream_pyav",
    "iter_validated_stages",
    "load_config",
    "pyav_version_receipt",
    "resolve_ffmpeg_binary_identity",
    "run_full_experiment",
    "upstream_pyav_lock_receipt",
    "validate_config",
    "verify_ffmpeg_binary_identity",
]
