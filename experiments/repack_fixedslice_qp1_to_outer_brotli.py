#!/usr/bin/env python3
"""Repack PR67/QZS3/QP1 fixed slices into a PR64-style outer Brotli payload.

This is a deterministic, byte-only transform for line-search archives whose
single ZIP member ``p`` is ``mask_br + model_br + pose_br`` with slice lengths
recorded in metadata JSON.  The output is a single stored ZIP member ``p``:

    brotli(<III> + masks.mkv + renderer.bin + optimized_poses.qp1)

The resulting archive is not a score claim until exact CUDA auth eval runs on
the exact output bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Any

import brotli


FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MEMBER_NAME = "p"
PR64_LEN_TABLE_STRUCT = "<III"
PRODUCER = "experiments/repack_fixedslice_qp1_to_outer_brotli.py"
SCHEMA_VERSION = 1
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
MODE_NAMES = {
    0: "generic",
    1: "text",
    2: "font",
}
REQUIRED_METADATA_FIELDS = ("mask_br_bytes", "model_br_bytes", "pose_br_bytes")


class RepackFixedSliceError(ValueError):
    """Raised when the source archive cannot be safely repacked."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_single_member_name(name: str) -> str:
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        raise RepackFixedSliceError(f"unsafe archive member path: {name!r}")
    path = Path(name)
    if len(path.parts) != 1 or any(part in {"", ".", ".."} for part in path.parts):
        raise RepackFixedSliceError(f"unsafe archive member path: {name!r}")
    if name.startswith(".") or name == "__MACOSX":
        raise RepackFixedSliceError(f"hidden/system archive member path: {name!r}")
    return name


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_single_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _assert_local_header_name_matches(archive: Path, info: zipfile.ZipInfo) -> None:
    """Reject ZIP parser-divergence cases where central/local names disagree."""
    with archive.open("rb") as handle:
        handle.seek(info.header_offset)
        fixed = handle.read(30)
        if len(fixed) != 30 or fixed[:4] != b"PK\x03\x04":
            raise RepackFixedSliceError("invalid ZIP local file header")
        name_len, extra_len = struct.unpack_from("<HH", fixed, 26)
        local_name = handle.read(name_len).decode("utf-8")
        if local_name != info.filename:
            raise RepackFixedSliceError(
                "ZIP central/local member name mismatch: "
                f"central={info.filename!r} local={local_name!r}"
            )
        if extra_len:
            handle.read(extra_len)


def read_fixedslice_payload(archive: Path, *, member_name: str = MEMBER_NAME) -> bytes:
    """Read a safe single-member fixed-slice ZIP payload."""
    member_name = _safe_single_member_name(member_name)
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
            if names != [member_name]:
                raise RepackFixedSliceError(
                    f"expected single member {member_name!r}; got {names!r}"
                )
            info = infos[0]
            _safe_single_member_name(info.filename)
            if info.compress_type != zipfile.ZIP_STORED:
                raise RepackFixedSliceError(
                    f"expected ZIP_STORED source member {member_name!r}; "
                    f"got compress_type={info.compress_type}"
                )
            _assert_local_header_name_matches(archive, info)
            return zf.read(info)
    except zipfile.BadZipFile as exc:
        raise RepackFixedSliceError(f"not a valid ZIP archive: {archive}") from exc


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise RepackFixedSliceError(f"metadata is not valid JSON: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise RepackFixedSliceError("metadata root must be a JSON object")
    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise RepackFixedSliceError(f"metadata missing required fields: {missing}")
    return metadata


def _positive_int_field(metadata: dict[str, Any], field: str) -> int:
    value = metadata[field]
    if isinstance(value, bool):
        raise RepackFixedSliceError(f"metadata field {field!r} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RepackFixedSliceError(
            f"metadata field {field!r} must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise RepackFixedSliceError(
            f"metadata field {field!r} must be a positive integer; got {parsed}"
        )
    return parsed


def slice_fixed_payload(payload: bytes, metadata: dict[str, Any]) -> tuple[bytes, bytes, bytes]:
    mask_len = _positive_int_field(metadata, "mask_br_bytes")
    model_len = _positive_int_field(metadata, "model_br_bytes")
    pose_len = _positive_int_field(metadata, "pose_br_bytes")
    expected = mask_len + model_len + pose_len
    if expected != len(payload):
        raise RepackFixedSliceError(
            f"payload length {len(payload)} != metadata slices {expected}"
        )
    mask_br = payload[:mask_len]
    model_br = payload[mask_len:mask_len + model_len]
    pose_br = payload[mask_len + model_len:]
    _assert_optional_sha(metadata, "mask_br_sha256", mask_br)
    _assert_optional_sha(metadata, "model_br_sha256", model_br)
    _assert_optional_sha(metadata, "pose_br_sha256", pose_br)
    return mask_br, model_br, pose_br


def _assert_optional_sha(metadata: dict[str, Any], field: str, data: bytes) -> None:
    expected = metadata.get(field)
    if expected is None:
        return
    actual = _sha256_bytes(data)
    if str(expected) != actual:
        raise RepackFixedSliceError(
            f"metadata {field} mismatch: expected={expected} actual={actual}"
        )


def _assert_optional_file_sha(metadata: dict[str, Any], field: str, path: Path) -> None:
    expected = metadata.get(field)
    if expected is None:
        return
    actual = _sha256_file(path)
    if str(expected) != actual:
        raise RepackFixedSliceError(
            f"metadata {field} mismatch: expected={expected} actual={actual}"
        )


def _assert_optional_int(metadata: dict[str, Any], field: str, actual: int) -> None:
    expected = metadata.get(field)
    if expected is None:
        return
    try:
        expected_int = int(expected)
    except (TypeError, ValueError) as exc:
        raise RepackFixedSliceError(f"metadata {field} must be an integer") from exc
    if expected_int != actual:
        raise RepackFixedSliceError(
            f"metadata {field} mismatch: expected={expected_int} actual={actual}"
        )


def _brotli_decompress_slice(data: bytes, label: str) -> bytes:
    try:
        return brotli.decompress(data)
    except brotli.error as exc:
        raise RepackFixedSliceError(f"{label} is not a valid Brotli stream") from exc


def _looks_like_mask_obu(data: bytes) -> bool:
    return data.startswith(b"\x12\x00\x0a\x0a") or data.startswith(b"\x12\x00")


def decode_fixed_slices(
    mask_br: bytes,
    model_br: bytes,
    pose_br: bytes,
) -> tuple[bytes, bytes, bytes]:
    masks = _brotli_decompress_slice(mask_br, "mask_br")
    renderer = _brotli_decompress_slice(model_br, "model_br")
    pose_qp1 = _brotli_decompress_slice(pose_br, "pose_br")
    if not _looks_like_mask_obu(masks):
        raise RepackFixedSliceError(f"invalid masks.mkv magic: {masks[:4]!r}")
    if not renderer.startswith(b"QZS3"):
        raise RepackFixedSliceError(f"invalid renderer.bin magic: {renderer[:4]!r}")
    if not pose_qp1.startswith(b"QP1"):
        raise RepackFixedSliceError(f"invalid QP1 pose payload magic: {pose_qp1[:4]!r}")
    if len(pose_qp1) < 5:
        raise RepackFixedSliceError("QP1 pose payload is too short")
    return masks, renderer, pose_qp1


def build_mask_first_len_table_payload(
    masks: bytes,
    renderer: bytes,
    pose_qp1: bytes,
) -> bytes:
    return (
        struct.pack(PR64_LEN_TABLE_STRUCT, len(masks), len(renderer), len(pose_qp1))
        + masks
        + renderer
        + pose_qp1
    )


def _compress_outer_brotli(
    raw_payload: bytes,
    *,
    quality: int,
    mode: int,
    lgwin: int,
    lgblock: int,
) -> bytes:
    return brotli.compress(
        raw_payload,
        quality=quality,
        mode=mode,
        lgwin=lgwin,
        lgblock=lgblock,
    )


def write_single_member_archive(path: Path, *, member_name: str, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(member_name), payload)


def _validate_brotli_params(quality: int, mode: int, lgwin: int, lgblock: int) -> None:
    if quality < 0 or quality > 11:
        raise RepackFixedSliceError(f"quality out of Brotli range: {quality}")
    if mode not in MODE_NAMES:
        raise RepackFixedSliceError(f"unsupported Brotli mode {mode}")
    if lgwin < 10 or lgwin > 24:
        raise RepackFixedSliceError(f"lgwin out of Brotli range: {lgwin}")
    if lgblock < 0 or lgblock > 24:
        raise RepackFixedSliceError(f"lgblock out of Brotli range: {lgblock}")


def repack_fixedslice_archive(
    *,
    source_archive: Path,
    metadata_path: Path,
    output_archive: Path,
    manifest_json: Path,
    member_name: str = MEMBER_NAME,
    quality: int = 11,
    mode: int = 2,
    lgwin: int = 18,
    lgblock: int = 0,
) -> dict[str, Any]:
    """Repack a fixed-slice QZS3/QP1 archive and emit custody metadata."""
    _validate_brotli_params(quality, mode, lgwin, lgblock)
    source_archive = source_archive.resolve()
    metadata_path = metadata_path.resolve()
    output_archive = output_archive.resolve()
    manifest_json = manifest_json.resolve()

    metadata = load_metadata(metadata_path)
    fixed_payload = read_fixedslice_payload(source_archive, member_name=member_name)
    _assert_optional_file_sha(metadata, "archive_sha256", source_archive)
    _assert_optional_int(metadata, "archive_bytes", source_archive.stat().st_size)
    _assert_optional_sha(metadata, "blob_sha256", fixed_payload)
    _assert_optional_int(metadata, "blob_bytes", len(fixed_payload))
    mask_br, model_br, pose_br = slice_fixed_payload(fixed_payload, metadata)
    masks, renderer, pose_qp1 = decode_fixed_slices(mask_br, model_br, pose_br)
    raw_payload = build_mask_first_len_table_payload(masks, renderer, pose_qp1)
    outer_br = _compress_outer_brotli(
        raw_payload,
        quality=quality,
        mode=mode,
        lgwin=lgwin,
        lgblock=lgblock,
    )
    if brotli.decompress(outer_br) != raw_payload:
        raise RepackFixedSliceError("outer Brotli payload failed round-trip")

    write_single_member_archive(output_archive, member_name=member_name, payload=outer_br)

    source_archive_bytes = source_archive.stat().st_size
    output_archive_bytes = output_archive.stat().st_size
    archive_delta = output_archive_bytes - source_archive_bytes
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_lossless_container_transform",
        "required_score_truth": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
        "source": {
            "archive_path": str(source_archive),
            "archive_bytes": source_archive_bytes,
            "archive_sha256": _sha256_file(source_archive),
            "metadata_path": str(metadata_path),
            "metadata_sha256": _sha256_file(metadata_path),
            "fixed_payload_bytes": len(fixed_payload),
            "fixed_payload_sha256": _sha256_bytes(fixed_payload),
            "payload_format": metadata.get("payload_format"),
        },
        "output": {
            "archive_path": str(output_archive),
            "archive_bytes": output_archive_bytes,
            "archive_sha256": _sha256_file(output_archive),
            "member_name": member_name,
            "payload_bytes": len(outer_br),
            "payload_sha256": _sha256_bytes(outer_br),
            "payload_format": "public_pr64_mask_first_len_table_outer_brotli",
            "payload_schema": "renderer_payload_pr64_len_table_v1",
        },
        "input_slices": {
            "mask_br_bytes": len(mask_br),
            "mask_br_sha256": _sha256_bytes(mask_br),
            "model_br_bytes": len(model_br),
            "model_br_sha256": _sha256_bytes(model_br),
            "pose_br_bytes": len(pose_br),
            "pose_br_sha256": _sha256_bytes(pose_br),
        },
        "raw_len_table_payload": {
            "struct": PR64_LEN_TABLE_STRUCT,
            "order": ["masks.mkv", "renderer.bin", "optimized_poses.bin"],
            "bytes": len(raw_payload),
            "sha256": _sha256_bytes(raw_payload),
            "lengths": {
                "masks.mkv": len(masks),
                "renderer.bin": len(renderer),
                "optimized_poses.bin": len(pose_qp1),
            },
        },
        "decoded_members": {
            "masks.mkv": {
                "bytes": len(masks),
                "sha256": _sha256_bytes(masks),
                "source_codec": "brotli_av1_obu",
            },
            "renderer.bin": {
                "bytes": len(renderer),
                "sha256": _sha256_bytes(renderer),
                "source_codec": "brotli_qzs3",
                "wire_format": "QZS3",
            },
            "optimized_poses.bin": {
                "bytes": len(pose_qp1),
                "sha256": _sha256_bytes(pose_qp1),
                "source_codec": "brotli_qp1",
                "wire_format": "QP1",
            },
        },
        "roundtrip": {
            "input_slices_brotli_decode": True,
            "outer_brotli_decode_matches_raw_len_table_payload": True,
            "raw_len_table_header": list(
                struct.unpack_from(PR64_LEN_TABLE_STRUCT, raw_payload, 0)
            ),
            "raw_len_table_payload_sha256_after_outer_decode": _sha256_bytes(
                brotli.decompress(outer_br)
            ),
        },
        "brotli_params": {
            "quality": quality,
            "mode": mode,
            "mode_name": MODE_NAMES[mode],
            "lgwin": lgwin,
            "lgblock": lgblock,
        },
        "determinism": {
            "zip_compress_type": "ZIP_STORED",
            "zip_timestamp": list(FIXED_ZIP_TIMESTAMP),
            "zip_permissions": "0644",
            "single_member_order": [member_name],
        },
        "archive_delta_bytes": archive_delta,
        "formula_only_rate_score_delta": archive_delta * RATE_SCORE_PER_BYTE,
        "non_promotable_warning": (
            "This is a deterministic container repack only. It remains "
            "score_claim=false until exact CUDA auth eval validates the exact archive."
        ),
    }
    if not math.isfinite(float(manifest["formula_only_rate_score_delta"])):
        raise RepackFixedSliceError("non-finite rate score delta")
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--member-name", default=MEMBER_NAME)
    parser.add_argument("--quality", type=int, default=11)
    parser.add_argument("--mode", type=int, default=2)
    parser.add_argument("--lgwin", type=int, default=18)
    parser.add_argument("--lgblock", type=int, default=0)
    args = parser.parse_args(argv)
    manifest = repack_fixedslice_archive(
        source_archive=args.source_archive,
        metadata_path=args.metadata_path,
        output_archive=args.output_archive,
        manifest_json=args.manifest_json,
        member_name=args.member_name,
        quality=args.quality,
        mode=args.mode,
        lgwin=args.lgwin,
        lgblock=args.lgblock,
    )
    print(json.dumps({
        "output_archive": manifest["output"],
        "archive_delta_bytes": manifest["archive_delta_bytes"],
        "formula_only_rate_score_delta": manifest["formula_only_rate_score_delta"],
        "score_claim": manifest["score_claim"],
        "promotion_eligible": manifest["promotion_eligible"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
