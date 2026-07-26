#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Public, scorer-free decoder for the counted TASK_LAYERED selected-plane ABI."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

import av
import numpy as np

SCHEMA = "taskspace_layered_public_archive.v1"
PAIR_COUNT = 600
SCORER_GEOMETRY = (384, 512, 3)
CAMERA_GEOMETRY = (874, 1164, 3)
EXPECTED_RAW_BYTES = PAIR_COUNT * 2 * np.prod(CAMERA_GEOMETRY, dtype=np.int64).item()
MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "pair_count",
        "scorer_geometry",
        "camera_geometry",
        "frame_rate",
        "receiver_contract_id",
        "layer_transform_id",
        "stream_contract_id",
        "pyav_decode_contract_id",
        "pyav_version",
        "expected_raw_bytes",
        "expected_raw_sha256",
        "chunks",
    }
)
CHUNK_FIELDS = frozenset({"pair_start", "pair_stop", "base", "enhancement"})
LAYER_FIELDS = frozenset({"packing", "streams"})
STREAM_FIELDS = frozenset(
    {
        "path",
        "bytes",
        "sha256",
        "frame_count",
        "codec_name",
        "coded_pixel_format",
        "decoded_pixel_format",
        "rgb_conversion_path",
        "semantic_channels",
        "decoded_sha256",
    }
)


class PublicDecodeError(RuntimeError):
    """Fail-closed archive, codec, resume, or output error."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PublicDecodeError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PublicDecodeError(f"{label} must be an integer >= {minimum}")
    return value


def _safe_member(root: Path, row: object, label: str) -> tuple[Path, dict]:
    if not isinstance(row, dict) or frozenset(row) != STREAM_FIELDS:
        raise PublicDecodeError(f"{label} has missing or unknown fields")
    if not isinstance(row["path"], str):
        raise PublicDecodeError(f"{label}.path must be a string")
    relative = PurePosixPath(row["path"])
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
        raise PublicDecodeError(f"{label}.path is unsafe")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise PublicDecodeError(f"{label}.path must be a regular non-symlink file")
    expected_bytes = _exact_int(row["bytes"], f"{label}.bytes", minimum=1)
    if path.stat().st_size != expected_bytes or _sha256_file(path) != _require_sha(row["sha256"], f"{label}.sha256"):
        raise PublicDecodeError(f"{label} counted-byte custody mismatch")
    _exact_int(row["frame_count"], f"{label}.frame_count", minimum=1)
    for field in (
        "codec_name",
        "coded_pixel_format",
        "decoded_pixel_format",
        "semantic_channels",
    ):
        if (
            not isinstance(row[field], str)
            or not row[field]
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_," for character in row[field])
        ):
            raise PublicDecodeError(f"{label}.{field} is not a safe typed token")
    if row["rgb_conversion_path"] not in {
        "native-gbrp-plane-extraction-and-rgb-reorder.v1",
        "PyAV-VideoFrame.to_ndarray-rgb24",
    }:
        raise PublicDecodeError(f"{label}.rgb_conversion_path is unsupported")
    _require_sha(row["decoded_sha256"], f"{label}.decoded_sha256")
    return path, row


def _validate_layer(
    archive_root: Path,
    layer: object,
    label: str,
    frame_count: int,
    declared_paths: set[str],
) -> None:
    if not isinstance(layer, dict) or frozenset(layer) != LAYER_FIELDS:
        raise PublicDecodeError(f"{label} has missing or unknown fields")
    packing = layer["packing"]
    streams = layer["streams"]
    if packing == "packed_rgb24":
        expected_channels = ("rgb",)
        expected_pixel_formats = ("rgb24",)
    elif packing == "separate_gray8_rgb":
        expected_channels = ("r", "g", "b")
        expected_pixel_formats = ("gray", "gray", "gray")
    else:
        raise PublicDecodeError(f"{label}.packing is unsupported")
    if not isinstance(streams, list) or len(streams) != len(expected_channels):
        raise PublicDecodeError(f"{label}.streams disagrees with packing")
    observed_channels: list[str] = []
    observed_pixel_formats: list[str] = []
    for stream_index, stream in enumerate(streams):
        _, row = _safe_member(archive_root, stream, f"{label}.streams[{stream_index}]")
        if row["frame_count"] != frame_count:
            raise PublicDecodeError("stream frame count disagrees with chunk range")
        if row["path"] in declared_paths:
            raise PublicDecodeError("counted stream paths must be unique")
        declared_paths.add(row["path"])
        observed_channels.append(row["semantic_channels"])
        observed_pixel_formats.append(row["decoded_pixel_format"])
    if tuple(observed_channels) != expected_channels:
        raise PublicDecodeError(f"{label} semantic channel order disagrees with packing")
    if tuple(observed_pixel_formats) != expected_pixel_formats:
        raise PublicDecodeError(f"{label} decoded pixel formats disagree with packing")


def load_manifest(archive_root: Path) -> tuple[dict, str]:
    manifest_path = archive_root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise PublicDecodeError("manifest.json must be a regular non-symlink file")
    payload = manifest_path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicDecodeError("manifest.json is not valid JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != payload:
        raise PublicDecodeError("manifest.json must be canonical JSON")
    if frozenset(value) != MANIFEST_FIELDS or value["schema"] != SCHEMA:
        raise PublicDecodeError("manifest schema or fields drifted")
    if value["pair_count"] != PAIR_COUNT:
        raise PublicDecodeError("public archive must contain exactly 600 ordered pairs")
    if not isinstance(value["scorer_geometry"], list) or tuple(value["scorer_geometry"]) != SCORER_GEOMETRY:
        raise PublicDecodeError("scorer geometry drifted")
    if not isinstance(value["camera_geometry"], list) or tuple(value["camera_geometry"]) != CAMERA_GEOMETRY:
        raise PublicDecodeError("camera geometry drifted")
    if value["frame_rate"] != 20:
        raise PublicDecodeError("frame rate drifted")
    if value["receiver_contract_id"] != "factor2-disjoint-half-pixel-uint8.v1":
        raise PublicDecodeError("receiver contract drifted")
    if value["layer_transform_id"] != "y1-base-plus-centered-signed-diff-q2-y0-given-y1.v1":
        raise PublicDecodeError("layer transform drifted")
    if value["stream_contract_id"] != "typed-pyav-packed-or-separate-rgb.v1":
        raise PublicDecodeError("typed stream contract drifted")
    if value["pyav_decode_contract_id"] != "pyav-single-video-stream-typed-rgb-conversion.v1":
        raise PublicDecodeError("PyAV decode contract drifted")
    if value["pyav_version"] != av.__version__:
        raise PublicDecodeError(f"PyAV version drifted: archive={value['pyav_version']} runtime={av.__version__}")
    if value["expected_raw_bytes"] != EXPECTED_RAW_BYTES:
        raise PublicDecodeError("expected raw byte count drifted")
    _require_sha(value["expected_raw_sha256"], "expected_raw_sha256")
    chunks = value["chunks"]
    if not isinstance(chunks, list) or not chunks:
        raise PublicDecodeError("chunks must be a non-empty ordered list")
    expected_start = 0
    declared_paths: set[str] = set()
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or frozenset(chunk) != CHUNK_FIELDS:
            raise PublicDecodeError(f"chunk[{index}] has missing or unknown fields")
        start = _exact_int(chunk["pair_start"], f"chunk[{index}].pair_start")
        stop = _exact_int(chunk["pair_stop"], f"chunk[{index}].pair_stop", minimum=1)
        if start != expected_start or stop <= start:
            raise PublicDecodeError("chunk table must be one exact contiguous ordered cover")
        expected_start = stop
        for layer in ("base", "enhancement"):
            _validate_layer(
                archive_root,
                chunk[layer],
                f"chunk[{index}].{layer}",
                stop - start,
                declared_paths,
            )
    if expected_start != PAIR_COUNT:
        raise PublicDecodeError("chunk table does not cover exactly 600 pairs")
    observed_files = set()
    for path in archive_root.rglob("*"):
        if path.is_symlink():
            raise PublicDecodeError("archive directory contains a symlink")
        if path.is_file():
            observed_files.add(path.relative_to(archive_root).as_posix())
        elif not path.is_dir():
            raise PublicDecodeError("archive directory contains an unsupported node")
    if observed_files != {"manifest.json", *declared_paths}:
        raise PublicDecodeError("archive directory contains missing, extra, or symlinked files")
    return value, _sha256_bytes(payload)


def _decode_stream(path: Path, row: dict) -> np.ndarray:
    try:
        with av.open(str(path), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise PublicDecodeError("counted member must have exactly one video stream")
            stream = streams[0]
            context = stream.codec_context
            context.thread_count = 1
            if (
                context.name != row["codec_name"]
                or context.pix_fmt != row["coded_pixel_format"]
                or context.height != SCORER_GEOMETRY[0]
                or context.width != SCORER_GEOMETRY[1]
            ):
                raise PublicDecodeError("coded codec/pixel-format/geometry disagrees with manifest")
            frames = []
            for frame in container.decode(video=0):
                if row["rgb_conversion_path"] == "native-gbrp-plane-extraction-and-rgb-reorder.v1":
                    if str(frame.format.name) != "gbrp" or len(frame.planes) != 3:
                        raise PublicDecodeError("native RGB decode requires exactly three gbrp planes")
                    channels = []
                    for plane in frame.planes:
                        line_size = int(plane.line_size)
                        if line_size < SCORER_GEOMETRY[1]:
                            raise PublicDecodeError("native gbrp plane stride is too small")
                        values = np.frombuffer(plane, dtype=np.uint8)
                        if values.size < SCORER_GEOMETRY[0] * line_size:
                            raise PublicDecodeError("native gbrp plane is truncated")
                        channels.append(
                            values[: SCORER_GEOMETRY[0] * line_size].reshape(SCORER_GEOMETRY[0], line_size)[
                                :, : SCORER_GEOMETRY[1]
                            ]
                        )
                    frames.append(
                        np.ascontiguousarray(
                            np.stack(
                                (channels[2], channels[0], channels[1]),
                                axis=-1,
                            )
                        )
                    )
                elif row["rgb_conversion_path"] == "PyAV-VideoFrame.to_ndarray-rgb24":
                    frames.append(frame.to_ndarray(format=row["decoded_pixel_format"]))
                else:
                    raise PublicDecodeError("typed RGB conversion path is unsupported")
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise PublicDecodeError(f"PyAV decode failed: {exc}") from exc
    if len(frames) != row["frame_count"]:
        raise PublicDecodeError("PyAV decoded frame count drifted")
    decoded = np.stack(frames, axis=0)
    expected_shape = (
        (row["frame_count"], *SCORER_GEOMETRY)
        if row["decoded_pixel_format"] == "rgb24"
        else (row["frame_count"], SCORER_GEOMETRY[0], SCORER_GEOMETRY[1])
    )
    if decoded.dtype != np.uint8 or decoded.shape != expected_shape:
        raise PublicDecodeError("PyAV decoded dtype/shape drifted")
    if _sha256_bytes(decoded.tobytes(order="C")) != row["decoded_sha256"]:
        raise PublicDecodeError("PyAV parse-back hash drifted; host codec semantics are not proven")
    return decoded


def _decode_layer(archive_root: Path, layer: dict, label: str) -> tuple[np.ndarray, list[dict]]:
    rows = []
    decoded = []
    for stream_index, row in enumerate(layer["streams"]):
        path, checked = _safe_member(archive_root, row, f"{label}.streams[{stream_index}]")
        rows.append(checked)
        decoded.append(_decode_stream(path, checked))
    if layer["packing"] == "packed_rgb24":
        return decoded[0], rows
    if layer["packing"] == "separate_gray8_rgb":
        return np.stack(decoded, axis=-1), rows
    raise PublicDecodeError(f"{label}.packing is unsupported")


def _axis_indices(input_size: int, output_size: int) -> np.ndarray:
    denominator = 2 * output_size
    rows: list[tuple[int, ...]] = []
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
        if len(taps) != 2 or sum(taps.values()) != denominator:
            raise PublicDecodeError("factor-2 support derivation drifted")
        rows.append(tuple(sorted(taps)))
    indices = np.asarray(rows, dtype=np.intp)
    if np.unique(indices).size != indices.size:
        raise PublicDecodeError("factor-2 supports overlap")
    return indices


ROW_INDICES = _axis_indices(CAMERA_GEOMETRY[0], SCORER_GEOMETRY[0])
COL_INDICES = _axis_indices(CAMERA_GEOMETRY[1], SCORER_GEOMETRY[1])


def realize_factor2(plane: np.ndarray) -> np.ndarray:
    if plane.dtype != np.uint8 or plane.shape != SCORER_GEOMETRY:
        raise PublicDecodeError("selected plane must be one 384x512x3 uint8 frame")
    output = np.zeros(CAMERA_GEOMETRY, dtype=np.uint8)
    for row_offset in range(2):
        for col_offset in range(2):
            output[
                ROW_INDICES[:, row_offset, None],
                COL_INDICES[None, :, col_offset],
                :,
            ] = plane
    return output


def _stage_paths(output_root: Path, video_stem: str, index: int) -> tuple[Path, Path]:
    root = output_root / ".taskspace-layered-public" / video_stem
    return root / f"stage-{index:04d}.raw", root / f"stage-{index:04d}.json"


def _write_atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_stage(
    archive_root: Path,
    output_root: Path,
    video_stem: str,
    manifest_sha256: str,
    index: int,
    chunk: dict,
) -> tuple[Path, dict, str]:
    stage_path, receipt_path = _stage_paths(output_root, video_stem, index)
    pair_count = chunk["pair_stop"] - chunk["pair_start"]
    expected_bytes = pair_count * 2 * np.prod(CAMERA_GEOMETRY, dtype=np.int64).item()
    if stage_path.exists() or receipt_path.exists():
        if (
            stage_path.is_symlink()
            or receipt_path.is_symlink()
            or not stage_path.is_file()
            or not receipt_path.is_file()
        ):
            raise PublicDecodeError("preserved stage checkpoint is incomplete")
        receipt_payload = receipt_path.read_bytes()
        try:
            receipt = json.loads(receipt_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicDecodeError("preserved stage receipt is invalid") from exc
        if (
            _canonical_json(receipt) != receipt_payload
            or receipt.get("manifest_sha256") != manifest_sha256
            or receipt.get("chunk_index") != index
            or receipt.get("raw_bytes") != expected_bytes
            or stage_path.stat().st_size != expected_bytes
            or _sha256_file(stage_path) != receipt.get("raw_sha256")
        ):
            raise PublicDecodeError("preserved stage checkpoint custody drifted")
        return stage_path, receipt, "resumed_verified_checkpoint"

    frame1, base_rows = _decode_layer(archive_root, chunk["base"], f"chunk[{index}].base")
    enhancement, enhancement_rows = _decode_layer(archive_root, chunk["enhancement"], f"chunk[{index}].enhancement")
    frame0 = np.clip(
        frame1.astype(np.int16) + 2 * (enhancement.astype(np.int16) - 128),
        0,
        255,
    ).astype(np.uint8)
    digest = hashlib.sha256()

    def write_stage(handle) -> None:
        for local_index in range(pair_count):
            for plane in (frame0[local_index], frame1[local_index]):
                payload = realize_factor2(plane).tobytes(order="C")
                handle.write(payload)
                digest.update(payload)

    _write_atomic(stage_path, write_stage)
    if stage_path.stat().st_size != expected_bytes:
        raise PublicDecodeError("stage raw byte count drifted")
    receipt = {
        "schema": "taskspace_layered_public_stage.v1",
        "manifest_sha256": manifest_sha256,
        "chunk_index": index,
        "pair_start": chunk["pair_start"],
        "pair_stop": chunk["pair_stop"],
        "raw_bytes": expected_bytes,
        "raw_sha256": digest.hexdigest(),
        "base_decoded_sha256": [row["decoded_sha256"] for row in base_rows],
        "enhancement_decoded_sha256": [row["decoded_sha256"] for row in enhancement_rows],
    }
    _write_atomic(receipt_path, lambda handle: handle.write(_canonical_json(receipt)))
    return stage_path, receipt, "fresh_decode"


def _single_video_stem(video_names_file: Path) -> str:
    names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    if len(names) != 1:
        raise PublicDecodeError("public decoder requires exactly one video name")
    path = PurePosixPath(names[0])
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise PublicDecodeError("video name is unsafe")
    return path.stem


def inflate(archive_root: Path, output_root: Path, video_names_file: Path) -> dict:
    manifest, manifest_sha = load_manifest(archive_root)
    video_stem = _single_video_stem(video_names_file)
    if output_root.exists() and (output_root.is_symlink() or not output_root.is_dir()):
        raise PublicDecodeError("output root must be absent or a regular non-symlink directory")
    output_root_was_absent = not output_root.exists()
    output_root_was_empty = (
        output_root.is_dir() and not output_root.is_symlink() and next(output_root.iterdir(), None) is None
        if output_root.exists()
        else False
    )
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    required = EXPECTED_RAW_BYTES * 2 + (1 << 30)
    if usage.free < required:
        raise PublicDecodeError(f"storage preflight refused: need {required} free bytes, observed {usage.free}")
    stages = [
        _build_stage(
            archive_root,
            output_root,
            video_stem,
            manifest_sha,
            index,
            chunk,
        )
        for index, chunk in enumerate(manifest["chunks"])
    ]
    raw_path = output_root / f"{video_stem}.raw"
    digest = hashlib.sha256()

    def assemble(handle) -> None:
        for stage_path, _, _ in stages:
            with stage_path.open("rb") as source:
                for payload in iter(lambda: source.read(8 << 20), b""):
                    handle.write(payload)
                    digest.update(payload)

    if raw_path.exists():
        if (
            raw_path.is_symlink()
            or not raw_path.is_file()
            or raw_path.stat().st_size != EXPECTED_RAW_BYTES
            or _sha256_file(raw_path) != manifest["expected_raw_sha256"]
        ):
            raise PublicDecodeError("existing final raw output custody drifted")
        final_assembly_action = "resumed_verified_final"
    else:
        _write_atomic(raw_path, assemble)
        final_assembly_action = "fresh_assembly"
    raw_sha = _sha256_file(raw_path)
    if raw_path.stat().st_size != EXPECTED_RAW_BYTES or raw_sha != manifest["expected_raw_sha256"]:
        raise PublicDecodeError("final 1200-frame raw size/hash disagrees with counted manifest")
    receipt = {
        "schema": "taskspace_layered_public_inflate_receipt.v1",
        "manifest_sha256": manifest_sha,
        "raw_path": raw_path.name,
        "raw_bytes": EXPECTED_RAW_BYTES,
        "raw_sha256": raw_sha,
        "pair_count": PAIR_COUNT,
        "frame_count": PAIR_COUNT * 2,
        "pyav": {
            "version": av.__version__,
            "library_versions": {name: list(version) for name, version in sorted(av.library_versions.items())},
            "thread_count": 1,
            "decode_path": (
                "av.open->container.decode(video=0)->typed native-gbrp-plane-extraction-or-VideoFrame.to_ndarray(rgb24)"
            ),
        },
        "output_root_identity_sha256": _sha256_bytes(str(output_root.resolve()).encode("utf-8")),
        "initial_output_root_was_clean": (output_root_was_absent or output_root_was_empty),
        "stage_count": len(stages),
        "stage_fresh_decode_count": sum(action == "fresh_decode" for _, _, action in stages),
        "stage_resume_count": sum(action == "resumed_verified_checkpoint" for _, _, action in stages),
        "final_assembly_action": final_assembly_action,
        "stage_receipts": [row for _, row, _ in stages],
        "scratch_cleanup": "atomic temporary files removed; stage checkpoints intentionally preserved",
    }
    invocation_mode = (
        "fresh"
        if receipt["stage_fresh_decode_count"] == receipt["stage_count"] and final_assembly_action == "fresh_assembly"
        else "resume"
    )
    receipt["invocation_mode"] = invocation_mode
    receipt_path = output_root / ".taskspace-layered-public" / video_stem / f"inflate-receipt-{invocation_mode}.json"
    payload = _canonical_json(receipt)
    if receipt_path.exists() and receipt_path.read_bytes() != payload:
        raise PublicDecodeError("preserved inflate receipt drifted")
    if not receipt_path.exists():
        _write_atomic(receipt_path, lambda handle: handle.write(payload))
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print("usage: inflate.py ARCHIVE_DIR OUTPUT_DIR VIDEO_NAMES_FILE", file=sys.stderr)
        return 2
    try:
        receipt = inflate(Path(args[0]), Path(args[1]), Path(args[2]))
    except (OSError, PublicDecodeError) as exc:
        print(f"taskspace layered public decode refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
