#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Counted public archive builder for typed selected-plane stream layers.

This module adapts a fresh scorer-plane operand compiler bundle into the
contest-facing archive shape. It never accepts historical plane payloads and
deliberately does not decide that a lossy payload has an exact score. A typed
research-only staging action may copy the preview bytes to the evaluator-required
``archive.zip`` name; only a separately sealed public exact-evaluation receipt
can make the resulting candidate promotion-eligible.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import tomllib
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import av
import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetError,
    load_dynamic_frontier_target,
    verify_dynamic_frontier_target_snapshot,
)
from tac.witness_dsl.taskspace_fresh_selected_plane_codec_v1 import (
    AGGREGATE_SCHEMA,
    BASE_STREAM,
    BUNDLE_SCHEMA,
    ENHANCEMENT_STREAM,
    TRANSFORM_ID,
)

PAIR_COUNT = 600
SCORER_GEOMETRY = [384, 512, 3]
CAMERA_GEOMETRY = [874, 1164, 3]
EXPECTED_RAW_BYTES = 3_662_409_600
ARCHIVE_SCHEMA = "taskspace_layered_public_archive.v1"
BUILD_SCHEMA = "taskspace_layered_public_build_receipt.v1"
AUTH_SCHEMA = "taskspace_layered_public_auth_receipt.v1"
STAGING_SCHEMA = "taskspace_layered_public_exact_eval_staging_receipt.v1"
G52_AGGREGATE_FIELDS = frozenset(
    {
        "schema",
        "experiment_schema",
        "status",
        "research_only",
        "candidate_lineage_allowed",
        "historical_payload_reused",
        "score_claim",
        "promotion_eligible",
        "pointer_delta",
        "config_sha256",
        "pair_count",
        "stage_count",
        "stage_receipt_sha256",
        "operand_provider",
        "representation_mode",
        "program_residual_layered",
        "pose_custody",
        "pose_authority",
        "codec",
        "public_decode",
        "upstream_pyav_lock",
        "endpoint",
        "final_recode_receipt_sha256",
        "counted_stream_bundle",
        "rate_term_if_used_as_exact_archive",
        "dynamic_frontier",
        "admission_rule",
        "next_authority_gate",
    }
)
G52_BUNDLE_FIELDS = frozenset(
    {
        "schema",
        "pair_count",
        "geometry",
        "representation_mode",
        "transform_id",
        "program_residual_layered",
        "historical_payload_reused",
        "freshness_rule",
        "config_sha256",
        "operand_receipt_sha256",
        "external_stage_checkpoint_receipt_sha256",
        "public_decode",
        "pose_custody",
        "pose_authority",
        "expected_reconstructed_scorer_planes",
        "public_raw_contract",
        "streams",
    }
)
G52_STREAM_FIELDS = frozenset(
    {
        "path",
        "stream_name",
        "frame_count",
        "bytes",
        "sha256",
        "decoded_sha256",
        "requested_encoder",
        "requested_encoded_pixel_format",
        "actual_codec_name",
        "actual_native_pixel_formats",
        "rgb_conversion_paths",
        "public_output_pixel_format",
    }
)
G52_PUBLIC_DECODE_FIELDS = frozenset(
    {
        "module",
        "version",
        "library_versions",
        "decode_path",
        "thread_count",
        "authority",
        "required_public_version",
    }
)
G52_RAW_CONTRACT_FIELDS = frozenset(
    {
        "receiver_contract_id",
        "realization_helper",
        "realization_helper_source_sha256",
        "chronological_order",
        "camera_geometry",
        "frame_count",
        "expected_raw_bytes",
        "expected_raw_sha256",
        "source_planes",
    }
)


class ClosureError(RuntimeError):
    """Fail-closed custody, lineage, archive, or promotion error."""


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, sha256: str, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ClosureError(f"{label} must be a regular non-symlink file: {path}")
    if sha256_file(path) != sha256:
        raise ClosureError(f"{label} SHA-256 custody mismatch: {path}")


def read_json(path: Path, sha256: str, label: str) -> dict[str, Any]:
    require_file(path, sha256, label)
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClosureError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ClosureError(f"{label} must be a JSON object")
    return value


def atomic_write(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_dependency(row: object, label: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ClosureError(f"{label} dependency record is missing")
    path = row.get("path")
    sha256 = row.get("sha256")
    if not isinstance(path, str) or not isinstance(sha256, str):
        raise ClosureError(f"{label} dependency path/SHA is missing")
    require_file(Path(path), sha256, label)
    return row


def _verify_fresh_operand_receipt(
    receipt_path: Path,
    receipt_sha256: str,
    bundle_path: Path,
    bundle_sha256: str,
) -> dict[str, Any]:
    receipt = read_json(receipt_path, receipt_sha256, "fresh scorer-plane operand receipt")
    if receipt.get("schema") == AGGREGATE_SCHEMA:
        bundle = receipt.get("counted_stream_bundle")
        provider = receipt.get("operand_provider")
        program = receipt.get("program_residual_layered")
        if (
            frozenset(receipt) != G52_AGGREGATE_FIELDS
            or receipt.get("status") != "closed_pending_public_receiver_and_exact_eval"
            or receipt.get("research_only") is not True
            or receipt.get("pair_count") != PAIR_COUNT
            or receipt.get("stage_count") != 5
            or receipt.get("candidate_lineage_allowed") is not True
            or receipt.get("historical_payload_reused") is not False
            or receipt.get("score_claim") is not False
            or receipt.get("promotion_eligible") is not False
            or receipt.get("pointer_delta") != "UNMOVED"
            or receipt.get("representation_mode") != "DIRECT_TASK_LAYERED"
            or not isinstance(program, dict)
            or program.get("available") is not False
            or program.get("v15_composition_claim") is not False
            or receipt.get("pose_custody") != "SEALED_SOURCE_CACHE_ADVISORY_ONLY"
            or receipt.get("pose_authority") is not False
            or not isinstance(bundle, dict)
            or bundle.get("path") != str(bundle_path)
            or bundle.get("sha256") != bundle_sha256
            or bundle.get("bytes") != bundle_path.stat().st_size
            or not isinstance(provider, dict)
        ):
            raise ClosureError("G52 aggregate is not the closed fresh DIRECT_TASK_LAYERED row")
        provider_path = Path(provider.get("aggregate_receipt_path", ""))
        provider_sha = provider.get("aggregate_receipt_sha256")
        if not isinstance(provider_sha, str):
            raise ClosureError("G52 aggregate lacks fresh G51 provider custody")
        require_file(provider_path, provider_sha, "fresh G51 scorer-plane provider")
        try:
            from tac.witness_control.taskspace_fresh_scorer_plane_materializer_v1 import (
                FreshScorerPlaneOperandLoaderV1,
            )

            loader = FreshScorerPlaneOperandLoaderV1.open(
                provider_path,
                expected_sha256=provider_sha,
            )
        except Exception as exc:
            raise ClosureError("fresh G51 scorer-plane provider failed recursive reopen") from exc
        if (
            loader.receipt.get("schema") != "tac.taskspace_fresh_scorer_plane_aggregate.v1"
            or loader.receipt.get("pair_count") != PAIR_COUNT
            or loader.receipt.get("aggregate_receipt_sha256") is None
        ):
            raise ClosureError("fresh G51 scorer-plane provider is not n600")
        stage_shas = receipt.get("stage_receipt_sha256")
        if not isinstance(stage_shas, list) or len(stage_shas) != 5:
            raise ClosureError("G52 aggregate lacks five external encoder checkpoints")
        for index, expected_sha in enumerate(stage_shas):
            stage_path = receipt_path.parent / "stages" / f"stage_{index:02d}" / "receipt.json"
            require_file(stage_path, expected_sha, f"G52 encoder checkpoint {index}")
        final_sha = receipt.get("final_recode_receipt_sha256")
        if not isinstance(final_sha, str):
            raise ClosureError("G52 aggregate lacks final recode receipt SHA")
        require_file(
            receipt_path.parent / "final" / "receipt.json",
            final_sha,
            "G52 final recode receipt",
        )
        return {
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_sha256,
            "fresh_current_scorer_plane_compile": True,
            "historical_payload_reused": False,
            "source_plane_hash_equality_is_not_lineage_reuse": True,
            "v15_composition_claim": False,
            "embedded_v15_bytes": 0,
            "counted_operand_bundle_sha256": bundle_sha256,
            "g51_provider_receipt_path": str(provider_path),
            "g51_provider_receipt_sha256": provider_sha,
            "g51_run_id": loader.receipt["run_id"],
            "g51_stage_chain_sha256": loader.receipt["stage_digest_chain_sha256"],
            "g51_self_seal_sha256": loader.receipt["aggregate_receipt_sha256"],
            "g52_encoder_stage_receipt_sha256": stage_shas,
            "g52_final_recode_receipt_sha256": final_sha,
            "g52_config_sha256": receipt["config_sha256"],
            "g52_public_decode": receipt["public_decode"],
            "g52_upstream_pyav_lock": receipt["upstream_pyav_lock"],
            "g52_bundle_manifest_sha256": bundle.get("manifest_sha256"),
            "g52_bundle_manifest": bundle.get("manifest"),
        }
    raise ClosureError("fresh operand receipt must be the exact G52 aggregate schema")


def _live_frontier(repo_root: Path) -> tuple[object, dict[str, Any]]:
    try:
        snapshot = load_dynamic_frontier_target(repo_root=repo_root)
        verify_dynamic_frontier_target_snapshot(snapshot)
    except DynamicFrontierTargetError as exc:
        raise ClosureError(f"live canonical frontier is unavailable: {exc}") from exc
    record = asdict(snapshot)
    return snapshot, {
        "pointer_path": record["pointer_path"],
        "pointer_bytes": record["pointer_bytes"],
        "pointer_sha256": record["pointer_sha256"],
        "last_refreshed_utc": record["last_refreshed_utc"],
        "target_score": record["target_score"],
        "selected_axis": record["selected_axis"],
        "selected_source": record["selected_source"],
        "selected_archive_sha256": record["selected_archive_sha256"],
        "selection_rule": record["selection_rule"],
        "admission_rule": ("candidate_exact_score_strictly_less_than_live_effective_frontier"),
    }


def _pyav_decode_member(
    payload: bytes,
    suffix: str,
    decoded_pixel_format: str,
    frame_count: int,
    rgb_conversion_path: str,
) -> dict[str, Any]:
    descriptor, name = tempfile.mkstemp(suffix=suffix)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        with av.open(str(path), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise ClosureError("PyAV found other than one video stream")
            stream = streams[0]
            context = stream.codec_context
            context.thread_count = 1
            codec_name = str(context.name)
            coded_pixel_format = str(context.pix_fmt)
            width = int(context.width)
            height = int(context.height)
            native_pixel_formats: set[str] = set()
            frames = []
            for frame in container.decode(video=0):
                native_format = str(frame.format.name)
                native_pixel_formats.add(native_format)
                if rgb_conversion_path == "native-gbrp-plane-extraction-and-rgb-reorder.v1":
                    if native_format != "gbrp":
                        raise ClosureError("native RGB decode requires an actual gbrp frame")
                    channels = []
                    for plane in frame.planes:
                        line_size = int(plane.line_size)
                        if line_size < SCORER_GEOMETRY[1]:
                            raise ClosureError("native gbrp plane stride is too small")
                        values = np.frombuffer(plane, dtype=np.uint8)
                        channels.append(
                            values[: SCORER_GEOMETRY[0] * line_size].reshape(SCORER_GEOMETRY[0], line_size)[
                                :, : SCORER_GEOMETRY[1]
                            ]
                        )
                    if len(channels) != 3:
                        raise ClosureError("native gbrp decode requires three planes")
                    frames.append(
                        np.ascontiguousarray(
                            np.stack(
                                (channels[2], channels[0], channels[1]),
                                axis=-1,
                            )
                        )
                    )
                elif rgb_conversion_path == "PyAV-VideoFrame.to_ndarray-rgb24":
                    frames.append(frame.to_ndarray(format=decoded_pixel_format))
                else:
                    raise ClosureError("G52 RGB conversion path is unsupported")
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise ClosureError(f"PyAV refused counted stream: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
    if len(frames) != frame_count:
        raise ClosureError("PyAV decoded frame count disagrees with chunk range")
    decoded = np.stack(frames, axis=0)
    expected_shape = (frame_count, 384, 512, 3) if decoded_pixel_format == "rgb24" else (frame_count, 384, 512)
    if decoded.dtype != np.uint8 or decoded.shape != expected_shape:
        raise ClosureError("PyAV decoded dtype/shape disagrees with typed layer")
    return {
        "codec_name": codec_name,
        "coded_pixel_format": coded_pixel_format,
        "width": width,
        "height": height,
        "native_pixel_formats": sorted(native_pixel_formats),
        "decoded_sha256": sha256_bytes(decoded.tobytes(order="C")),
        "decoded": decoded,
    }


def _zip_info(name: str, *, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    mode = stat.S_IFREG | (0o755 if executable else 0o644)
    info.external_attr = mode << 16
    return info


def _stream_row(
    public_path: str,
    payload: bytes,
    source_row: dict[str, Any],
    frame_count: int,
    decoded_pixel_format: str,
    semantic_channels: str,
) -> tuple[dict[str, Any], np.ndarray]:
    conversion_paths = source_row.get("rgb_conversion_paths")
    if not isinstance(conversion_paths, list) or len(conversion_paths) != 1:
        raise ClosureError("G52 stream must have one deterministic RGB conversion path")
    rgb_conversion_path = conversion_paths[0]
    pyav_decode = _pyav_decode_member(
        payload,
        PurePosixPath(public_path).suffix,
        decoded_pixel_format,
        frame_count,
        rgb_conversion_path,
    )
    encoder_decoded_sha = source_row.get("decoded_sha256")
    if not isinstance(encoder_decoded_sha, str) or len(encoder_decoded_sha) != 64:
        raise ClosureError("encoder-side stream parse-back SHA is missing")
    if (
        pyav_decode["codec_name"] != source_row.get("actual_codec_name")
        or pyav_decode["native_pixel_formats"] != source_row.get("actual_native_pixel_formats")
        or pyav_decode["width"] != SCORER_GEOMETRY[1]
        or pyav_decode["height"] != SCORER_GEOMETRY[0]
    ):
        raise ClosureError("G52 and public PyAV stream typing disagree")
    if pyav_decode["decoded_sha256"] != encoder_decoded_sha:
        raise ClosureError("encoder-side and PyAV parse-back bytes are not identical")
    return (
        {
            "path": public_path,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "frame_count": frame_count,
            "codec_name": pyav_decode["codec_name"],
            "coded_pixel_format": pyav_decode["coded_pixel_format"],
            "decoded_pixel_format": decoded_pixel_format,
            "rgb_conversion_path": rgb_conversion_path,
            "semantic_channels": semantic_channels,
            "decoded_sha256": pyav_decode["decoded_sha256"],
        },
        pyav_decode["decoded"],
    )


def _require_exact_sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ClosureError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_g52_bundle_manifest(
    value: object,
    *,
    lineage: dict[str, Any],
    expected_pyav_version: str,
    repo_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != G52_BUNDLE_FIELDS:
        raise ClosureError("G52 counted bundle manifest fields drifted")
    program = value["program_residual_layered"]
    geometry = value["geometry"]
    public_decode = value["public_decode"]
    raw_contract = value["public_raw_contract"]
    reconstructed = value["expected_reconstructed_scorer_planes"]
    if (
        value["schema"] != BUNDLE_SCHEMA
        or value["pair_count"] != PAIR_COUNT
        or geometry != {"height": 384, "width": 512, "channels": 3}
        or value["representation_mode"] != "DIRECT_TASK_LAYERED"
        or value["transform_id"] != TRANSFORM_ID
        or not isinstance(program, dict)
        or frozenset(program) != frozenset({"available", "status", "v15_composition_claim"})
        or program["available"] is not False
        or program["status"] != "blocked_missing_fresh_semantic_base_bytes"
        or program["v15_composition_claim"] is not False
        or value["historical_payload_reused"] is not False
        or value["config_sha256"] != lineage["g52_config_sha256"]
        or value["operand_receipt_sha256"] != lineage["g51_provider_receipt_sha256"]
        or value["external_stage_checkpoint_receipt_sha256"] != lineage["g52_encoder_stage_receipt_sha256"]
        or value["pose_custody"] != "SEALED_SOURCE_CACHE_ADVISORY_ONLY"
        or value["pose_authority"] is not False
    ):
        raise ClosureError("G52 counted bundle is not the fresh DIRECT_TASK_LAYERED n600 ABI")
    if (
        not isinstance(public_decode, dict)
        or frozenset(public_decode) != G52_PUBLIC_DECODE_FIELDS
        or public_decode["module"] != "av"
        or public_decode["version"] != expected_pyav_version
        or public_decode["required_public_version"] != expected_pyav_version
        or public_decode["thread_count"] != 1
        or public_decode["authority"] != "public-runtime parse-back"
    ):
        raise ClosureError("G52 public PyAV dependency does not match the official runtime pin")
    if public_decode != lineage["g52_public_decode"]:
        raise ClosureError("G52 aggregate and counted bundle public decode receipts disagree")
    if not isinstance(reconstructed, dict) or frozenset(reconstructed) != frozenset({"y0_u8_sha256", "y1_u8_sha256"}):
        raise ClosureError("G52 reconstructed scorer-plane contract drifted")
    _require_exact_sha(reconstructed["y0_u8_sha256"], "G52 reconstructed Y0")
    _require_exact_sha(reconstructed["y1_u8_sha256"], "G52 reconstructed Y1")
    if (
        not isinstance(raw_contract, dict)
        or frozenset(raw_contract) != G52_RAW_CONTRACT_FIELDS
        or raw_contract["receiver_contract_id"] != "factor2-disjoint-half-pixel-uint8.v1"
        or raw_contract["realization_helper"]
        != ("tac.optimization.uint8_lattice_feasibility.realize_factor2_uint8_scorer_plane")
        or raw_contract["chronological_order"] != "pair_id ascending, frame0 from Y0 then frame1 from Y1"
        or raw_contract["camera_geometry"] != CAMERA_GEOMETRY
        or raw_contract["frame_count"] != PAIR_COUNT * 2
        or raw_contract["expected_raw_bytes"] != EXPECTED_RAW_BYTES
        or raw_contract["source_planes"] != "PyAV-authoritative decoded and conditionally reconstructed bytes"
    ):
        raise ClosureError("G52 public raw receiver contract drifted")
    _require_exact_sha(raw_contract["expected_raw_sha256"], "G52 expected public raw")
    helper_path = repo_root / "src/tac/optimization/uint8_lattice_feasibility.py"
    require_file(
        helper_path,
        _require_exact_sha(
            raw_contract["realization_helper_source_sha256"],
            "G52 realization helper source",
        ),
        "G52 realization helper source",
    )
    streams = value["streams"]
    if (
        not isinstance(streams, list)
        or len(streams) != 2
        or not all(isinstance(row, dict) and frozenset(row) == G52_STREAM_FIELDS for row in streams)
        or {row["stream_name"] for row in streams} != {BASE_STREAM, ENHANCEMENT_STREAM}
    ):
        raise ClosureError("G52 counted bundle must contain exactly two typed population streams")
    for row in streams:
        if (
            type(row["frame_count"]) is not int
            or row["frame_count"] != PAIR_COUNT
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
            or row["public_output_pixel_format"] != "rgb24"
            or not isinstance(row["actual_codec_name"], str)
            or not row["actual_codec_name"]
            or not isinstance(row["actual_native_pixel_formats"], list)
            or not row["actual_native_pixel_formats"]
            or not all(
                isinstance(pixel_format, str) and pixel_format for pixel_format in row["actual_native_pixel_formats"]
            )
            or not isinstance(row["rgb_conversion_paths"], list)
            or not row["rgb_conversion_paths"]
        ):
            raise ClosureError("G52 stream typing is incomplete")
        _require_exact_sha(row["sha256"], "G52 counted stream")
        _require_exact_sha(row["decoded_sha256"], "G52 decoded stream")
    return value


def _derive_public_raw_contract(
    reconstructed_y0: np.ndarray,
    reconstructed_y1: np.ndarray,
) -> dict[str, Any]:
    if (
        reconstructed_y0.dtype != np.uint8
        or reconstructed_y1.dtype != np.uint8
        or reconstructed_y0.shape != (PAIR_COUNT, *SCORER_GEOMETRY)
        or reconstructed_y1.shape != (PAIR_COUNT, *SCORER_GEOMETRY)
    ):
        raise ClosureError("reconstructed scorer planes do not have the exact n600 ABI")
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_GEOMETRY[0],
        camera_w=CAMERA_GEOMETRY[1],
        scorer_h=SCORER_GEOMETRY[0],
        scorer_w=SCORER_GEOMETRY[1],
    )
    digest = hashlib.sha256()
    frame_count = 0
    for y0, y1 in zip(reconstructed_y0, reconstructed_y1, strict=True):
        for plane in (y0, y1):
            realized = realize_factor2_uint8_scorer_plane(operator, plane)
            digest.update(memoryview(realized))
            frame_count += 1
    return {
        "frame_count": frame_count,
        "expected_raw_bytes": (frame_count * math.prod(CAMERA_GEOMETRY)),
        "expected_raw_sha256": digest.hexdigest(),
    }


def _runtime_tree(runtime_dir: Path) -> dict[str, Any]:
    rows = []
    digest = hashlib.sha256()
    for name in ("inflate.py", "inflate.sh"):
        path = runtime_dir / name
        if path.is_symlink() or not path.is_file():
            raise ClosureError(f"public runtime file is missing: {path}")
        payload = path.read_bytes()
        row = {"path": name, "bytes": len(payload), "sha256": sha256_bytes(payload)}
        rows.append(row)
        digest.update(canonical_json(row))
    return {
        "role": "generic free decoder code; not in counted archive.zip",
        "files": rows,
        "tree_sha256": digest.hexdigest(),
        "bytes": sum(row["bytes"] for row in rows),
    }


def build_preview(config: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(config, dict)
        or frozenset(config)
        != frozenset(
            {
                "schema",
                "counted_operand_bundle",
                "operand_compiler_receipt",
                "runtime_dir",
                "output_dir",
                "repo_root",
                "pyav_runtime",
            }
        )
        or config.get("schema") != "taskspace_g55_public_layered_codec_closure_config.v1"
    ):
        raise ClosureError("G55 closure config schema or fields drifted")
    bundle_path = Path(config["counted_operand_bundle"]["path"])
    bundle_sha = config["counted_operand_bundle"]["sha256"]
    operand_receipt_path = Path(config["operand_compiler_receipt"]["path"])
    operand_receipt_sha = config["operand_compiler_receipt"]["sha256"]
    runtime_dir = Path(config["runtime_dir"])
    output_dir = Path(config["output_dir"])
    repo_root = Path(config.get("repo_root", "."))
    expected_pyav_version = config["pyav_runtime"]["version"]
    pyav_lock_path = Path(config["pyav_runtime"]["lock_path"])
    pyav_lock_sha = config["pyav_runtime"]["lock_sha256"]

    require_file(bundle_path, bundle_sha, "fresh counted operand bundle")
    require_file(pyav_lock_path, pyav_lock_sha, "upstream PyAV runtime lock")
    try:
        lock = tomllib.loads(pyav_lock_path.read_text(encoding="utf-8"))
        av_rows = [row for row in lock["package"] if isinstance(row, dict) and row.get("name") == "av"]
    except (KeyError, OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ClosureError("upstream PyAV runtime lock is unreadable") from exc
    if len(av_rows) != 1 or av_rows[0].get("version") != expected_pyav_version:
        raise ClosureError("configured PyAV version is not the authoritative upstream lock pin")
    if av.__version__ != expected_pyav_version:
        raise ClosureError(f"builder PyAV {av.__version__} disagrees with upstream lock {expected_pyav_version}")
    lineage = _verify_fresh_operand_receipt(
        operand_receipt_path,
        operand_receipt_sha,
        bundle_path,
        bundle_sha,
    )
    g52_lock = lineage["g52_upstream_pyav_lock"]
    if (
        not isinstance(g52_lock, dict)
        or frozenset(g52_lock) != frozenset({"path", "bytes", "sha256", "package", "version"})
        or Path(g52_lock["path"]).resolve() != pyav_lock_path.resolve()
        or g52_lock["bytes"] != pyav_lock_path.stat().st_size
        or g52_lock["sha256"] != pyav_lock_sha
        or g52_lock["package"] != "av"
        or g52_lock["version"] != expected_pyav_version
    ):
        raise ClosureError("G52 and G55 authoritative upstream PyAV lock custody disagrees")
    try:
        source_archive = zipfile.ZipFile(bundle_path)
    except zipfile.BadZipFile as exc:
        raise ClosureError("fresh counted operand bundle is not a ZIP archive") from exc
    with source_archive as source:
        if len(source.namelist()) != len(set(source.namelist())):
            raise ClosureError("fresh operand bundle contains duplicate member names")
        if source.namelist().count("manifest.json") != 1:
            raise ClosureError("fresh operand bundle manifest is missing or duplicated")
        try:
            source_manifest_payload = source.read("manifest.json")
            operand_manifest = json.loads(source_manifest_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ClosureError("fresh operand bundle manifest is invalid") from exc
        if canonical_json(operand_manifest) != source_manifest_payload:
            raise ClosureError("fresh operand bundle manifest is not canonical JSON")
        operand_manifest = _validate_g52_bundle_manifest(
            operand_manifest,
            lineage=lineage,
            expected_pyav_version=expected_pyav_version,
            repo_root=repo_root,
        )
        if (
            sha256_bytes(source_manifest_payload) != lineage["g52_bundle_manifest_sha256"]
            or operand_manifest != lineage["g52_bundle_manifest"]
        ):
            raise ClosureError("G52 aggregate and counted bundle manifests disagree")
        source_streams = {row["stream_name"]: row for row in operand_manifest["streams"]}
        source_paths = {"manifest.json", *(row["path"] for row in source_streams.values())}
        if set(source.namelist()) != source_paths:
            raise ClosureError("fresh operand bundle contains undeclared members")
        members: dict[str, bytes] = {}
        public_rows: dict[str, dict[str, Any]] = {}
        decoded: dict[str, np.ndarray] = {}
        for stream_name, layer_name in (
            (BASE_STREAM, "base"),
            (ENHANCEMENT_STREAM, "enhancement"),
        ):
            source_row = source_streams[stream_name]
            source_name = source_row["path"]
            pure_name = PurePosixPath(source_name)
            if (
                pure_name.is_absolute()
                or ".." in pure_name.parts
                or len(pure_name.parts) != 2
                or pure_name.parts[0] != "streams"
            ):
                raise ClosureError("G52 stream path is unsafe")
            info = source.getinfo(source_name)
            mode = info.external_attr >> 16
            if info.is_dir() or (mode and not stat.S_ISREG(mode)):
                raise ClosureError("G52 counted stream is not a regular ZIP member")
            try:
                payload = source.read(source_name)
            except (KeyError, zipfile.BadZipFile) as exc:
                raise ClosureError(f"G52 {stream_name} stream is absent") from exc
            if source_row["bytes"] != len(payload) or source_row["sha256"] != sha256_bytes(payload):
                raise ClosureError(f"G52 {stream_name} stream custody mismatch")
            suffix = pure_name.suffix or ".bin"
            public_name = f"streams/chunk-0000/{layer_name}-rgb{suffix}"
            public_row, decoded_frames = _stream_row(
                public_name,
                payload,
                source_row,
                PAIR_COUNT,
                "rgb24",
                "rgb",
            )
            members[public_name] = payload
            public_rows[layer_name] = public_row
            decoded[stream_name] = decoded_frames

        reconstructed_y1 = decoded[BASE_STREAM]
        reconstructed_y0 = np.clip(
            reconstructed_y1.astype(np.int16) + 2 * (decoded[ENHANCEMENT_STREAM].astype(np.int16) - 128),
            0,
            255,
        ).astype(np.uint8)
        reconstructed_contract = operand_manifest["expected_reconstructed_scorer_planes"]
        if (
            sha256_bytes(reconstructed_y0.tobytes(order="C")) != reconstructed_contract["y0_u8_sha256"]
            or sha256_bytes(reconstructed_y1.tobytes(order="C")) != reconstructed_contract["y1_u8_sha256"]
        ):
            raise ClosureError("G55 independent layer inversion disagrees with G52")
        derived_raw = _derive_public_raw_contract(
            reconstructed_y0,
            reconstructed_y1,
        )
        source_raw_contract = operand_manifest["public_raw_contract"]
        if any(
            derived_raw[field] != source_raw_contract[field]
            for field in (
                "frame_count",
                "expected_raw_bytes",
                "expected_raw_sha256",
            )
        ):
            raise ClosureError("G55 independent generic V10 realization disagrees with G52")
        expected_raw_sha = derived_raw["expected_raw_sha256"]
        chunks = [
            {
                "pair_start": 0,
                "pair_stop": PAIR_COUNT,
                "base": {
                    "packing": "packed_rgb24",
                    "streams": [public_rows["base"]],
                },
                "enhancement": {
                    "packing": "packed_rgb24",
                    "streams": [public_rows["enhancement"]],
                },
            }
        ]

    manifest = {
        "schema": ARCHIVE_SCHEMA,
        "pair_count": PAIR_COUNT,
        "scorer_geometry": SCORER_GEOMETRY,
        "camera_geometry": CAMERA_GEOMETRY,
        "frame_rate": 20,
        "receiver_contract_id": "factor2-disjoint-half-pixel-uint8.v1",
        "layer_transform_id": TRANSFORM_ID,
        "stream_contract_id": "typed-pyav-packed-or-separate-rgb.v1",
        "pyav_decode_contract_id": ("pyav-single-video-stream-typed-rgb-conversion.v1"),
        "pyav_version": av.__version__,
        "expected_raw_bytes": EXPECTED_RAW_BYTES,
        "expected_raw_sha256": expected_raw_sha,
        "chunks": chunks,
    }
    manifest_payload = canonical_json(manifest)
    frontier_snapshot, competitive_target = _live_frontier(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "archive.preview.zip"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".archive.preview.", suffix=".zip.tmp", dir=output_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as target:
            target.writestr(_zip_info("manifest.json"), manifest_payload)
            for name in sorted(members):
                target.writestr(_zip_info(name), members[name])
        os.replace(temporary, preview_path)
    finally:
        temporary.unlink(missing_ok=True)

    for name in ("inflate.py", "inflate.sh"):
        source_path = runtime_dir / name
        destination = output_dir / name
        atomic_write(
            destination,
            source_path.read_bytes(),
            0o755 if name.endswith(".sh") else 0o644,
        )
    runtime = _runtime_tree(output_dir)
    payload_stream_bytes = sum(len(payload) for payload in members.values())
    archive_bytes = preview_path.stat().st_size
    operand_bundle_bytes = bundle_path.stat().st_size
    receipt = {
        "schema": BUILD_SCHEMA,
        "status": "preview_built_promotion_refused_without_public_auth_receipt",
        "pointer_delta": "UNMOVED",
        "score_claim": False,
        "promotion_eligible": False,
        "research_only": True,
        "candidate_lineage_allowed": True,
        "pair_count": PAIR_COUNT,
        "frame_count": PAIR_COUNT * 2,
        "archive_preview": {
            "path": str(preview_path),
            "bytes": archive_bytes,
            "sha256": sha256_file(preview_path),
        },
        "counted_byte_accounting": {
            "payload_stream_bytes": payload_stream_bytes,
            "manifest_bytes": len(manifest_payload),
            "zip_container_bytes": archive_bytes - payload_stream_bytes - len(manifest_payload),
            "production_archive_bytes": archive_bytes,
            "operand_bundle_bytes": operand_bundle_bytes,
            "operand_container_and_manifest_bytes": operand_bundle_bytes - payload_stream_bytes,
            "historical_payload_bytes_in_production_archive": 0,
        },
        "runtime": runtime,
        "runtime_dependency": {
            "pyav_version": av.__version__,
            "upstream_lock_path": str(pyav_lock_path),
            "upstream_lock_sha256": pyav_lock_sha,
            "decode_contract_id": ("pyav-single-video-stream-typed-rgb-conversion.v1"),
            "encoder_public_cross_path_equality_required": True,
        },
        "source_custody": {
            "fresh_operand_bundle_path": str(bundle_path),
            "fresh_operand_bundle_sha256": bundle_sha,
            "fresh_operand_compiler_receipt_path": str(operand_receipt_path),
            "fresh_operand_compiler_receipt_sha256": operand_receipt_sha,
            "expected_public_raw_sha256": expected_raw_sha,
        },
        "lineage": lineage,
        "competitive_target": competitive_target,
        "stream_abi": {
            "current_chunk_count": len(chunks),
            "chunk_count_is_semantic": False,
            "future_two_long_gop_streams_allowed": True,
            "packing_modes": ["packed_rgb24", "separate_gray8_rgb"],
            "coded_pixel_format_is_typed_not_fixed": True,
            "current_coded_pixel_formats": sorted(
                {
                    stream["coded_pixel_format"]
                    for chunk in chunks
                    for layer in ("base", "enhancement")
                    for stream in chunk[layer]["streams"]
                }
            ),
        },
        "promotion_gate": {
            "required_auth_schema": AUTH_SCHEMA,
            "required_target": ("candidate_exact_score_strictly_less_than_live_effective_frontier"),
            "required_entrypoint": "upstream/evaluate.py",
            "required_pair_count": PAIR_COUNT,
            "required_exact_eval_staging_receipt": STAGING_SCHEMA,
            "required_two_distinct_clean_root_fresh_inflates": True,
            "authority_axes": ["contest-CPU", "contest-CUDA"],
        },
        "cleanup": {
            "temporary_files": "success-only scratch deleted atomically",
            "durable_artifacts": ["archive.preview.zip", "inflate.py", "inflate.sh"],
        },
    }
    try:
        verify_dynamic_frontier_target_snapshot(frontier_snapshot)
    except DynamicFrontierTargetError as exc:
        raise ClosureError(f"canonical frontier changed during build: {exc}") from exc
    receipt_payload = canonical_json(receipt)
    receipt_path = output_dir / "build_receipt.json"
    atomic_write(receipt_path, receipt_payload)
    return {**receipt, "receipt_path": str(receipt_path), "receipt_sha256": sha256_bytes(receipt_payload)}


def stage_exact_eval(
    preview_path: Path,
    build_receipt: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Materialize the evaluator-required name without making a promotion claim."""

    if (
        build_receipt.get("schema") != BUILD_SCHEMA
        or build_receipt.get("research_only") is not True
        or build_receipt.get("candidate_lineage_allowed") is not True
        or build_receipt.get("promotion_eligible") is not False
        or build_receipt.get("score_claim") is not False
        or build_receipt.get("pointer_delta") != "UNMOVED"
    ):
        raise ClosureError("exact-eval staging requires a non-promoted candidate-lineage build")
    if preview_path.is_symlink() or not preview_path.is_file():
        raise ClosureError("exact-eval staging preview must be a regular file")
    preview = build_receipt.get("archive_preview")
    if (
        not isinstance(preview, dict)
        or preview.get("path") != str(preview_path)
        or preview.get("bytes") != preview_path.stat().st_size
        or preview.get("sha256") != sha256_file(preview_path)
    ):
        raise ClosureError("exact-eval staging preview custody mismatch")
    archive_path = preview_path.with_name("archive.zip")
    if archive_path.exists():
        if (
            archive_path.is_symlink()
            or not archive_path.is_file()
            or archive_path.stat().st_size != preview_path.stat().st_size
            or sha256_file(archive_path) != preview["sha256"]
        ):
            raise ClosureError("existing exact-eval archive.zip custody mismatch")
    else:
        atomic_write(archive_path, preview_path.read_bytes())
    competitive_target = build_receipt.get("competitive_target")
    if not isinstance(competitive_target, dict):
        raise ClosureError("exact-eval staging lacks dynamic target custody")
    receipt = {
        "schema": STAGING_SCHEMA,
        "status": "staged_for_exact_evaluation_not_promoted",
        "research_only": True,
        "candidate_lineage_allowed": True,
        "promotion_eligible": False,
        "score_claim": False,
        "pointer_delta": "UNMOVED",
        "archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
        "archive_preview": dict(preview),
        "canonical_frontier_pointer_sha256": competitive_target.get("pointer_sha256"),
        "selection_rule": competitive_target.get("selection_rule"),
        "purpose": (
            "upstream/evaluate.sh requires archive.zip before exact authority "
            "exists; this filename is staging, not promotion"
        ),
    }
    receipt_payload = canonical_json(receipt)
    receipt_path = preview_path.with_name("exact_eval_staging_receipt.json")
    if receipt_path.exists() and receipt_path.read_bytes() != receipt_payload:
        raise ClosureError("preserved exact-eval staging receipt drifted")
    if not receipt_path.exists():
        atomic_write(receipt_path, receipt_payload)
    return archive_path, {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_bytes(receipt_payload),
    }


def promote(
    preview_path: Path,
    build_receipt: dict[str, Any],
    auth_path: Path,
    auth_sha256: str,
    *,
    repo_root: Path = Path("."),
) -> Path:
    if (
        build_receipt.get("schema") != BUILD_SCHEMA
        or build_receipt.get("candidate_lineage_allowed") is not True
        or build_receipt.get("promotion_eligible") is not False
        or build_receipt.get("score_claim") is not False
    ):
        raise ClosureError("promotion refused: build lineage/truth boundary is invalid")
    if preview_path.is_symlink() or not preview_path.is_file():
        raise ClosureError("promotion refused: preview archive is not a regular file")
    auth = read_json(auth_path, auth_sha256, "public exact auth receipt")
    preview_sha = sha256_file(preview_path)
    raw_sha = build_receipt["source_custody"]["expected_public_raw_sha256"]
    frontier_snapshot, competitive_target = _live_frontier(repo_root)
    evidence = auth.get("evidence")
    if not isinstance(evidence, dict):
        raise ClosureError("promotion refused: public exact evidence map is missing")
    for field in (
        "upstream_evaluate",
        "upstream_snapshot_receipt",
        "evaluate_stdout_log",
        "authority_hardware_receipt",
        "exact_eval_staging_receipt",
    ):
        _verify_dependency(evidence.get(field), f"public auth {field}")
    staging_record = evidence["exact_eval_staging_receipt"]
    staging = read_json(
        Path(staging_record["path"]),
        staging_record["sha256"],
        "public auth exact-eval staging receipt",
    )
    archive_path = preview_path.with_name("archive.zip")
    if (
        staging.get("schema") != STAGING_SCHEMA
        or staging.get("status") != "staged_for_exact_evaluation_not_promoted"
        or staging.get("candidate_lineage_allowed") is not True
        or staging.get("promotion_eligible") is not False
        or staging.get("score_claim") is not False
        or staging.get("archive_preview")
        != {
            "path": str(preview_path),
            "bytes": preview_path.stat().st_size,
            "sha256": preview_sha,
        }
        or staging.get("archive", {}).get("path") != str(archive_path)
        or staging.get("archive", {}).get("sha256") != preview_sha
        or staging.get("archive", {}).get("bytes") != preview_path.stat().st_size
        or staging.get("canonical_frontier_pointer_sha256") != competitive_target["pointer_sha256"]
        or staging.get("selection_rule") != competitive_target["selection_rule"]
    ):
        raise ClosureError("promotion refused: exact-eval staging custody is invalid")
    require_file(archive_path, preview_sha, "staged exact-eval archive.zip")
    upstream_evaluate = evidence["upstream_evaluate"]
    if not str(upstream_evaluate["path"]).endswith("upstream/evaluate.py"):
        raise ClosureError("promotion refused: evaluator dependency is not upstream/evaluate.py")
    inflate_rows = evidence.get("double_inflate_receipts")
    if not isinstance(inflate_rows, list) or len(inflate_rows) != 2:
        raise ClosureError("promotion refused: exactly two inflate receipts are required")
    inflate_receipts = []
    build_pyav_version = build_receipt.get("runtime_dependency", {}).get("pyav_version")
    for index, row in enumerate(inflate_rows):
        checked = _verify_dependency(row, f"public auth inflate receipt {index}")
        receipt = read_json(
            Path(checked["path"]),
            checked["sha256"],
            f"public auth inflate receipt {index}",
        )
        if (
            receipt.get("schema") != "taskspace_layered_public_inflate_receipt.v1"
            or receipt.get("raw_sha256") != raw_sha
            or receipt.get("raw_bytes") != EXPECTED_RAW_BYTES
            or receipt.get("pair_count") != PAIR_COUNT
            or receipt.get("frame_count") != PAIR_COUNT * 2
            or receipt.get("initial_output_root_was_clean") is not True
            or type(receipt.get("stage_count")) is not int
            or receipt.get("stage_count") <= 0
            or receipt.get("stage_fresh_decode_count") != receipt.get("stage_count")
            or receipt.get("stage_resume_count") != 0
            or receipt.get("final_assembly_action") != "fresh_assembly"
            or receipt.get("invocation_mode") != "fresh"
            or not isinstance(receipt.get("output_root_identity_sha256"), str)
            or len(receipt["output_root_identity_sha256"]) != 64
            or receipt.get("pyav", {}).get("thread_count") != 1
            or receipt.get("pyav", {}).get("version") != build_pyav_version
            or receipt.get("pyav", {}).get("decode_path")
            != (
                "av.open->container.decode(video=0)->typed native-gbrp-plane-extraction-or-VideoFrame.to_ndarray(rgb24)"
            )
        ):
            raise ClosureError("promotion refused: inflate receipt is not a fresh clean-root n600 decode")
        inflate_receipts.append(receipt)
        _require_exact_sha(
            receipt["output_root_identity_sha256"],
            f"public auth inflate receipt {index} output root identity",
        )
    stable_inflate_fields = (
        "manifest_sha256",
        "raw_bytes",
        "raw_sha256",
        "pair_count",
        "frame_count",
        "pyav",
        "stage_count",
        "stage_receipts",
    )
    if inflate_receipts[0]["output_root_identity_sha256"] == inflate_receipts[1]["output_root_identity_sha256"] or any(
        inflate_receipts[0].get(field) != inflate_receipts[1].get(field) for field in stable_inflate_fields
    ):
        raise ClosureError("promotion refused: two distinct clean-root public inflates were not byte-identical")
    d_seg = auth.get("d_seg")
    d_pose = auth.get("d_pose")
    score = auth.get("score")
    if (
        not isinstance(d_seg, (int, float))
        or not isinstance(d_pose, (int, float))
        or not isinstance(score, (int, float))
        or not math.isfinite(float(d_seg))
        or not math.isfinite(float(d_pose))
        or not math.isfinite(float(score))
        or d_seg < 0
        or d_pose < 0
    ):
        raise ClosureError("promotion refused: exact score components are invalid")
    recomputed_score = (
        100.0 * float(d_seg) + math.sqrt(10.0 * float(d_pose)) + 25.0 * preview_path.stat().st_size / 37_545_489
    )
    if (
        auth.get("schema") != AUTH_SCHEMA
        or auth.get("archive_sha256") != preview_sha
        or auth.get("archive_bytes") != preview_path.stat().st_size
        or auth.get("raw_sha256") != raw_sha
        or auth.get("raw_bytes") != EXPECTED_RAW_BYTES
        or auth.get("frame_count") != PAIR_COUNT * 2
        or auth.get("pair_count") != PAIR_COUNT
        or auth.get("evaluation_entrypoint") != "upstream/evaluate.py"
        or auth.get("authority_axis") not in {"contest-CPU", "contest-CUDA"}
        or auth.get("double_inflate_identical") is not True
        or auth.get("candidate_lineage_allowed") is not True
        or auth.get("score_claim") is not True
        or abs(float(score) - recomputed_score) > 1e-12
        or float(score) >= competitive_target["target_score"]
        or auth.get("competitive_target_score") != competitive_target["target_score"]
        or auth.get("canonical_frontier_pointer_sha256") != competitive_target["pointer_sha256"]
        or auth.get("selection_rule") != competitive_target["selection_rule"]
    ):
        raise ClosureError("promotion refused: public exact authority gate is not closed")
    try:
        verify_dynamic_frontier_target_snapshot(frontier_snapshot)
    except DynamicFrontierTargetError as exc:
        raise ClosureError(f"canonical frontier changed during promotion: {exc}") from exc
    return archive_path
