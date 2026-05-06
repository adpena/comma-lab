#!/usr/bin/env python3
"""Profile contest archive byte accounting and self-compression opportunities.

This tool is empirical analysis only. It does not inflate frames, load scorers,
dispatch jobs, or make score claims. It answers a narrower but critical
question: where do the archive bytes go, what is already self-compressed, and
which byte streams remain plausible optimization targets under contest
custody.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import importlib.util
import json
import lzma
import math
import shutil
import subprocess
import struct
import sys
import zlib
import zipfile
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
UNPACKER_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"
SCHEMA = "archive_byte_accounting_profile_v1"
TOOL = "experiments/profile_archive_byte_accounting.py"
EVIDENCE_GRADE = "empirical_byte_profile"
ORIGINAL_VIDEO_BYTES = 37_545_489
LAMBDA_RATE = 25.0 / ORIGINAL_VIDEO_BYTES
CONTEST_SCORE_SOURCE = "archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA"
DEFAULT_TARGET_SCORE = 0.3


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _json_file_sha256(path: Path) -> str:
    return _sha256_file(path)


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("_archive_byte_accounting_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entropy_bits_per_byte(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for value in data:
        counts[value] += 1
    total = float(len(data))
    entropy = 0.0
    for count in counts:
        if count:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def _byte_entropy_record(data: bytes) -> dict[str, Any]:
    entropy = _entropy_bits_per_byte(data)
    entropy_bytes = int(math.ceil(entropy * len(data) / 8.0))
    return {
        "bytes": int(len(data)),
        "entropy_bits_per_byte": round(entropy, 12),
        "zero_order_entropy_bytes": int(entropy_bytes),
        "zero_order_entropy_ratio": None if not data else round(entropy_bytes / len(data), 12),
        "unique_byte_count": int(len(set(data))),
    }


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _bitplane_record(data: bytes) -> dict[str, Any]:
    n = len(data)
    planes: list[dict[str, Any]] = []
    total_entropy_bits = 0.0
    for bit in range(8):
        ones = sum((value >> bit) & 1 for value in data)
        one_fraction = 0.0 if n == 0 else ones / n
        entropy = _binary_entropy(one_fraction)
        total_entropy_bits += entropy * n
        planes.append(
            {
                "bit": bit,
                "one_count": int(ones),
                "zero_count": int(n - ones),
                "one_fraction": round(one_fraction, 12),
                "entropy_bits_per_symbol": round(entropy, 12),
                "zero_order_entropy_bits": round(entropy * n, 6),
            }
        )
    return {
        "bytes": int(n),
        "bits": int(n * 8),
        "planes_lsb0": planes,
        "zero_order_bitplane_entropy_bits": round(total_entropy_bits, 6),
        "zero_order_bitplane_entropy_bytes": int(math.ceil(total_entropy_bits / 8.0)),
        "zero_order_bitplane_entropy_ratio": None
        if n == 0
        else round(math.ceil(total_entropy_bits / 8.0) / n, 12),
    }


def _brotli_compress(data: bytes, *, quality: int) -> bytes | None:
    try:
        import brotli
    except ImportError:
        return None
    return brotli.compress(data, quality=quality)


def _zstd_cli_compress(data: bytes, *, level: int) -> bytes | None:
    zstd = shutil.which("zstd")
    if zstd is None:
        return None
    try:
        completed = subprocess.run(
            [zstd, "-q", f"-{level}", "--no-progress", "-c"],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _compression_probe(data: bytes) -> dict[str, Any]:
    probes: dict[str, Any] = {
        "input_bytes": int(len(data)),
        "zlib_9_bytes": int(len(zlib.compress(data, level=9))),
        "lzma_preset9_bytes": int(len(lzma.compress(data, preset=9))),
    }
    brotli_results: dict[str, int] = {}
    for quality in (5, 9, 11):
        compressed = _brotli_compress(data, quality=quality)
        if compressed is not None:
            brotli_results[f"brotli_q{quality}_bytes"] = int(len(compressed))
    probes.update(brotli_results)
    if len(data) >= 1024:
        zstd_results: dict[str, int] = {}
        for level in (3, 9, 19, 22):
            compressed = _zstd_cli_compress(data, level=level)
            if compressed is not None:
                zstd_results[f"zstd_{level}_bytes"] = int(len(compressed))
        probes.update(zstd_results)
    else:
        probes["zstd_probe_skipped_reason"] = "input_below_1024_bytes"
    best = min(
        ((key, value) for key, value in probes.items() if key.endswith("_bytes")),
        key=lambda item: (int(item[1]), item[0]),
    )
    probes["best_probe"] = {"codec": best[0], "bytes": int(best[1])}
    probes["best_probe_delta_vs_input"] = int(best[1]) - int(len(data))
    return probes


def _zip_bytes_for_payload(
    payload: bytes,
    *,
    compress_type: int,
    compresslevel: int | None = None,
    member_name: str = "p",
) -> bytes:
    buffer = io.BytesIO()
    kwargs: dict[str, Any] = {"compression": compress_type}
    if compresslevel is not None:
        kwargs["compresslevel"] = compresslevel
    with zipfile.ZipFile(buffer, "w", **kwargs) as zf:
        info = zipfile.ZipInfo(member_name, (1980, 1, 1, 0, 0, 0))
        info.compress_type = compress_type
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload)
    return buffer.getvalue()


def _container_option(
    *,
    option_id: str,
    archive_bytes: bytes,
    source_archive_bytes: int,
    zip_method: str,
    payload_container_codec: str,
    runtime_supported: bool,
    dispatch_status: str,
    note: str,
) -> dict[str, Any]:
    archive_size = len(archive_bytes)
    return {
        "option_id": option_id,
        "archive_bytes": int(archive_size),
        "archive_delta_bytes_vs_source": int(archive_size - source_archive_bytes),
        "archive_sha256": _sha256_bytes(archive_bytes),
        "zip_method": zip_method,
        "payload_container_codec": payload_container_codec,
        "runtime_supported_by_current_unpacker": runtime_supported,
        "dispatch_status": dispatch_status,
        "note": note,
        "score_claim": False,
    }


def _finite_container_probe(
    *,
    parser_payload: bytes,
    source_archive_bytes: int,
) -> dict[str, Any]:
    """Screen deterministic outer-container choices without changing streams."""

    options: list[dict[str, Any]] = []
    options.append(
        _container_option(
            option_id="zip_stored_raw_p",
            archive_bytes=_zip_bytes_for_payload(parser_payload, compress_type=zipfile.ZIP_STORED),
            source_archive_bytes=source_archive_bytes,
            zip_method="stored",
            payload_container_codec="raw",
            runtime_supported=True,
            dispatch_status="dispatchable_if_decoded_stream_parity_matches_source",
            note="Single stored member p; this is the strict baseline container.",
        )
    )
    for level in (0, 1, 6, 9):
        options.append(
            _container_option(
                option_id=f"zip_deflate{level}_raw_p",
                archive_bytes=_zip_bytes_for_payload(
                    parser_payload,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=level,
                ),
                source_archive_bytes=source_archive_bytes,
                zip_method=f"deflate:{level}",
                payload_container_codec="raw",
                runtime_supported=True,
                dispatch_status="dispatchable_if_archive_size_wins_and_unzip_preflight_passes",
                note="ZIP deflate preserves the p payload after extraction; exact eval still required.",
            )
        )
    for level in (1, 9):
        options.append(
            _container_option(
                option_id=f"zip_bzip2_{level}_raw_p",
                archive_bytes=_zip_bytes_for_payload(
                    parser_payload,
                    compress_type=zipfile.ZIP_BZIP2,
                    compresslevel=level,
                ),
                source_archive_bytes=source_archive_bytes,
                zip_method=f"bzip2:{level}",
                payload_container_codec="raw",
                runtime_supported=False,
                dispatch_status="blocked_until_contest_unzip_support_is_preflighted",
                note="Analysis-only container screen; do not dispatch without archive extraction compatibility proof.",
            )
        )
    options.append(
        _container_option(
            option_id="zip_lzma_raw_p",
            archive_bytes=_zip_bytes_for_payload(parser_payload, compress_type=zipfile.ZIP_LZMA),
            source_archive_bytes=source_archive_bytes,
            zip_method="zip_lzma",
            payload_container_codec="raw",
            runtime_supported=False,
            dispatch_status="blocked_until_contest_unzip_support_is_preflighted",
            note="Analysis-only container screen; do not dispatch without archive extraction compatibility proof.",
        )
    )

    for quality in (0, 5, 9, 11):
        wrapped = _brotli_compress(parser_payload, quality=quality)
        if wrapped is None:
            continue
        options.append(
            _container_option(
                option_id=f"zip_stored_outer_brotli_q{quality}_p",
                archive_bytes=_zip_bytes_for_payload(wrapped, compress_type=zipfile.ZIP_STORED),
                source_archive_bytes=source_archive_bytes,
                zip_method="stored",
                payload_container_codec=f"outer_brotli:q{quality}",
                runtime_supported=True,
                dispatch_status="dispatchable_if_archive_size_wins_and_p_brotli_unwrap_preflight_passes",
                note="Current unpacker treats single-member p as Brotli when decompression succeeds.",
            )
        )

    for level in (3, 9, 19, 22):
        wrapped = _zstd_cli_compress(parser_payload, level=level)
        if wrapped is None:
            continue
        options.append(
            _container_option(
                option_id=f"zip_stored_outer_zstd_{level}_p",
                archive_bytes=_zip_bytes_for_payload(wrapped, compress_type=zipfile.ZIP_STORED),
                source_archive_bytes=source_archive_bytes,
                zip_method="stored",
                payload_container_codec=f"outer_zstd:{level}",
                runtime_supported=False,
                dispatch_status="blocked_runtime_support_required",
                note="zstd is analysis-only here; robust_current does not decode zstd p payloads.",
            )
        )

    options.sort(key=lambda row: (int(row["archive_bytes"]), str(row["option_id"])))
    return {
        "schema": "finite_lossless_container_probe_v1",
        "score_claim": False,
        "source_archive_bytes": int(source_archive_bytes),
        "best_option": options[0] if options else None,
        "options": options,
    }


def _zip_member_overhead(archive: Path) -> tuple[list[dict[str, Any]], int]:
    archive_bytes = archive.stat().st_size
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            local_header_bytes = 30 + len(info.filename.encode("utf-8")) + len(info.extra)
            central_header_bytes = (
                46
                + len(info.filename.encode("utf-8"))
                + len(info.extra)
                + len(info.comment or b"")
            )
            records.append(
                {
                    "filename": info.filename,
                    "compress_type": int(info.compress_type),
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "header_offset": int(info.header_offset),
                    "local_header_bytes": int(local_header_bytes),
                    "central_directory_header_bytes": int(central_header_bytes),
                    "crc": f"{info.CRC:08x}",
                }
            )
    zip_payload_bytes = sum(int(item["compress_size"]) for item in records)
    overhead = archive_bytes - zip_payload_bytes
    return records, int(overhead)


def _extract_single_payload(archive: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive, "r") as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        if len(names) != 1:
            raise ValueError(f"{archive} must contain exactly one file member for payload profiling; got {names}")
        return names[0], zf.read(names[0])


def _normalize_payload_member(payload_name: str, payload: bytes) -> tuple[bytes, dict[str, Any]]:
    """Return parser-ready payload plus a record of the ZIP member container."""
    record: dict[str, Any] = {
        "zip_member_name": payload_name,
        "zip_member_payload_bytes": int(len(payload)),
        "zip_member_payload_sha256": _sha256_bytes(payload),
        "payload_container_codec": "raw",
    }
    if payload.startswith((b"RPK1", b"RP2\x01")):
        return payload, record
    try:
        import brotli
    except ImportError:
        return payload, record
    try:
        decoded = brotli.decompress(payload)
    except brotli.error:
        return payload, record
    if decoded.startswith((b"RPK1", b"RP2\x01")):
        record.update(
            {
                "payload_container_codec": "brotli",
                "payload_member_decoded_bytes": int(len(decoded)),
                "payload_member_decoded_sha256": _sha256_bytes(decoded),
                "payload_container_roundtrip_required": True,
            }
        )
        return decoded, record
    return payload, record


def _encoded_slices_from_header(payload: bytes, header: dict[str, Any]) -> list[dict[str, Any]]:
    payload_format = str(header.get("payload_format", ""))
    members_meta = header.get("members")
    if not isinstance(members_meta, list):
        raise ValueError("payload header does not contain members list")

    if payload_format == "public_pr67_qzs3_qp1_fixed_slices":
        by_name = {str(item["name"]): item for item in members_meta if isinstance(item, dict)}
        mask_len = int(by_name["masks.mkv"]["bytes"])
        renderer_len = int(by_name["renderer.bin"]["bytes"])
        pose_len = int(by_name["optimized_poses.bin"]["bytes"])
        offsets = {
            "masks.mkv": (0, mask_len),
            "renderer.bin": (mask_len, mask_len + renderer_len),
            "optimized_poses.bin": (mask_len + renderer_len, mask_len + renderer_len + pose_len),
        }
        return [
            {
                "name": name,
                "encoded_offset": int(start),
                "encoded_bytes": int(end - start),
                "encoded_sha256": _sha256_bytes(payload[start:end]),
                "encoded_payload_order": rank,
                "codec": by_name[name].get("codec"),
                "decoded_bytes": int(by_name[name].get("decoded_bytes", 0)),
                "decoded_sha256": by_name[name].get("decoded_sha256"),
            }
            for rank, (name, (start, end)) in enumerate(offsets.items())
        ]

    if payload.startswith(b"RPK1"):
        header_len = struct.unpack_from("<I", payload, 4)[0]
        offset = 8 + int(header_len)
        out: list[dict[str, Any]] = []
        for rank, meta in enumerate(members_meta):
            if not isinstance(meta, dict):
                raise ValueError("payload member metadata must be object")
            n_bytes = int(meta["bytes"])
            data = payload[offset: offset + n_bytes]
            out.append(
                {
                    "name": str(meta["name"]),
                    "encoded_offset": int(offset),
                    "encoded_bytes": int(n_bytes),
                    "encoded_sha256": _sha256_bytes(data),
                    "encoded_payload_order": rank,
                    "codec": meta.get("codec", "raw"),
                    "decoded_bytes": int(meta.get("decoded_bytes", n_bytes)),
                    "decoded_sha256": meta.get("decoded_sha256"),
                }
            )
            offset += n_bytes
        return out

    # Generic fallback: trust member order as listed.
    offset = 0
    out = []
    for rank, meta in enumerate(members_meta):
        if not isinstance(meta, dict):
            raise ValueError("payload member metadata must be object")
        n_bytes = int(meta["bytes"])
        data = payload[offset: offset + n_bytes]
        out.append(
            {
                "name": str(meta["name"]),
                "encoded_offset": int(offset),
                "encoded_bytes": int(n_bytes),
                "encoded_sha256": _sha256_bytes(data),
                "encoded_payload_order": rank,
                "codec": meta.get("codec", "raw"),
                "decoded_bytes": int(meta.get("decoded_bytes", n_bytes)),
                "decoded_sha256": meta.get("decoded_sha256"),
            }
        )
        offset += n_bytes
    return out


def _encoded_slices_from_public_pr65_bundle(
    payload_name: str,
    payload: bytes,
) -> tuple[dict[str, Any], dict[str, bytes], list[dict[str, Any]]] | None:
    """Parse PR65/henosis compact member ``x`` for byte forensics.

    The public PR65 bundle is not a robust_current runtime contract; it is an
    external archive grammar that stores 24-bit stream lengths followed by
    Brotli-compressed core and postprocess atoms.  This parser is deliberately
    analysis-only so a public forensics run can account for every byte without
    teaching the contest inflate path to accept a new archive format.
    """

    if payload_name != "x" or len(payload) < 30:
        return None
    stream_names = (
        "masks.mkv",
        "renderer.bin",
        "optimized_poses.bin",
        "qpost.post",
        "qpost.shift",
        "qpost.frac",
        "qpost.frac2",
        "qpost.frac3",
        "qpost.bias",
        "qpost.region",
    )
    lengths = [int.from_bytes(payload[offset : offset + 3], "little") for offset in range(0, 30, 3)]
    mask_len, model_len, pose_len = lengths[:3]
    if not (mask_len > 1000 and model_len > 1000 and pose_len > 100):
        return None
    if sum(lengths) > len(payload) - 30:
        return None

    try:
        import brotli
    except ImportError:
        return None

    slices: list[tuple[str, int, int, bytes]] = []
    offset = 30
    for name, n_bytes in zip(stream_names, lengths):
        chunk = payload[offset : offset + n_bytes]
        if len(chunk) != n_bytes:
            return None
        slices.append((name, offset, n_bytes, chunk))
        offset += n_bytes
    randmulti = payload[offset:]
    if not randmulti:
        return None
    slices.append(("qpost.randmulti", offset, len(randmulti), randmulti))

    decoded_members: dict[str, bytes] = {}
    encoded_slices: list[dict[str, Any]] = []
    members_meta: list[dict[str, Any]] = []
    codec_by_name = {
        "masks.mkv": "brotli_av1_obu",
        "renderer.bin": "brotli_pr65_custom_model",
        "optimized_poses.bin": "brotli_pr65_pose",
        "qpost.post": "brotli_qpost_post",
        "qpost.shift": "brotli_qpost_shift",
        "qpost.frac": "brotli_qpost_frac",
        "qpost.frac2": "brotli_qpost_frac2",
        "qpost.frac3": "brotli_qpost_frac3",
        "qpost.bias": "brotli_qpost_bias",
        "qpost.region": "brotli_qpost_region",
        "qpost.randmulti": "brotli_qpost_randmulti",
    }
    for rank, (name, start, n_bytes, chunk) in enumerate(slices):
        try:
            decoded = brotli.decompress(chunk)
            decoded_codec = codec_by_name[name]
        except brotli.error:
            decoded = chunk
            decoded_codec = f"{codec_by_name[name]}_undecoded"
        decoded_members[name] = decoded
        meta = {
            "name": name,
            "bytes": int(n_bytes),
            "sha256": _sha256_bytes(chunk),
            "codec": decoded_codec,
            "decoded_bytes": int(len(decoded)),
            "decoded_sha256": _sha256_bytes(decoded),
        }
        members_meta.append(meta)
        encoded_slices.append(
            {
                "name": name,
                "encoded_offset": int(start),
                "encoded_bytes": int(n_bytes),
                "encoded_sha256": _sha256_bytes(chunk),
                "encoded_payload_order": rank,
                "codec": decoded_codec,
                "decoded_bytes": int(len(decoded)),
                "decoded_sha256": _sha256_bytes(decoded),
            }
        )

    header = {
        "schema": "public_pr65_compact_bundle_v1",
        "payload_format": "public_pr65_qpost_compact_v4",
        "header_bytes": 30,
        "length_encoding": "ten_24bit_little_endian_lengths_then_randmulti_tail",
        "members": members_meta,
        "score_claim": False,
        "evidence_grade": "external_byte_forensics_only",
    }
    return header, decoded_members, encoded_slices


def _stream_attackability(name: str, codec: str | None, encoded_bytes: int, decoded_bytes: int) -> dict[str, Any]:
    rate_score = LAMBDA_RATE * float(encoded_bytes)
    if name == "masks.mkv":
        risk = "very_high_pose_seg_cliff"
        opportunity = "largest stream; attack only with scorer-trust-region mask grammar, multimask subset repair, or learned decoder"
        priority = 1
    elif name == "renderer.bin":
        risk = "medium_pose_seg_runtime_contract"
        opportunity = "self-compress JFG/QZS/QBF only after local byte win and loader parity"
        priority = 2
    elif name == "optimized_poses.bin":
        risk = "low_to_medium_pose_sensitive"
        opportunity = "small stream; optimize with qpose/VLQ/manifold residuals but cannot close sub-0.3 alone"
        priority = 3
    elif name.startswith("qpost."):
        risk = "medium_postprocess_can_shift_pose_and_seg"
        opportunity = "PR65-style output atom; use only as learned/pair-local charged repair after exact component evidence"
        priority = 4
    else:
        risk = "unknown"
        opportunity = "inspect before dispatch"
        priority = 9
    return {
        "priority": int(priority),
        "risk": risk,
        "opportunity": opportunity,
        "rate_score_contribution": round(rate_score, 12),
        "decoded_to_encoded_ratio": None if encoded_bytes <= 0 else round(decoded_bytes / encoded_bytes, 12),
        "codec": codec,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _target_gap_record(
    *,
    eval_payload: dict[str, Any] | None,
    target_score: float,
    profiled_archive_size_bytes: int | None = None,
    profiled_archive_sha256: str | None = None,
) -> dict[str, Any] | None:
    if eval_payload is None:
        return None
    score = _float_or_none(eval_payload.get("canonical_score"))
    score_source = "canonical_score" if score is not None else None
    recomputed_score = _float_or_none(eval_payload.get("score_recomputed_from_components"))
    if recomputed_score is not None:
        score = recomputed_score
        score_source = "score_recomputed_from_components"
    if score is None:
        score = _float_or_none(eval_payload.get("final_score"))
        score_source = "final_score" if score is not None else None
    archive_bytes_value = eval_payload.get("archive_size_bytes")
    if archive_bytes_value is None:
        archive_bytes_value = (eval_payload.get("provenance") or {}).get("archive_size_bytes")
    try:
        archive_bytes = int(archive_bytes_value)
    except (TypeError, ValueError):
        archive_bytes = None
    if score is None or archive_bytes is None:
        return {
            "target_score": target_score,
            "available": False,
            "reason": "eval JSON does not expose canonical_score/score_recomputed_from_components/final_score and archive_size_bytes",
        }
    provenance = eval_payload.get("provenance") or {}
    reference_archive_sha256 = provenance.get("archive_sha256")
    size_matches = profiled_archive_size_bytes is None or archive_bytes == int(profiled_archive_size_bytes)
    sha_matches = (
        profiled_archive_sha256 is None
        or reference_archive_sha256 is None
        or str(reference_archive_sha256) == str(profiled_archive_sha256)
    )
    reference_matches_profiled_archive = bool(size_matches and sha_matches)
    score_gap = max(0.0, score - target_score)
    bytes_to_remove = int(math.ceil(score_gap / LAMBDA_RATE)) if score_gap > 0 else 0
    buffered_bytes_to_remove = bytes_to_remove + 1 if bytes_to_remove > 0 else 0
    current_rate_score = LAMBDA_RATE * archive_bytes
    current_distortion_score = score - current_rate_score
    target_archive_bytes = archive_bytes - bytes_to_remove
    buffered_target_archive_bytes = archive_bytes - buffered_bytes_to_remove
    return {
        "target_score": target_score,
        "available": True,
        "score": score,
        "score_source": score_source,
        "score_gap": score_gap,
        "current_archive_size_bytes": archive_bytes,
        "reference_archive_size_bytes": archive_bytes,
        "reference_archive_sha256": reference_archive_sha256,
        "profiled_archive_size_bytes": profiled_archive_size_bytes,
        "profiled_archive_sha256": profiled_archive_sha256,
        "reference_matches_profiled_archive": reference_matches_profiled_archive,
        "reference_warning": None
        if reference_matches_profiled_archive
        else (
            "target_gap was computed from eval_json's scored archive, which does not "
            "match the profiled archive bytes/SHA; treat it as a reference gap only"
        ),
        "current_rate_score": current_rate_score,
        "current_distortion_score": current_distortion_score,
        "bytes_to_remove_if_distortion_unchanged": bytes_to_remove,
        "buffered_bytes_to_remove_if_distortion_unchanged": buffered_bytes_to_remove,
        "target_archive_size_bytes_if_distortion_unchanged": target_archive_bytes,
        "target_rate_score_if_distortion_unchanged": LAMBDA_RATE * target_archive_bytes,
        "buffered_target_archive_size_bytes_if_distortion_unchanged": buffered_target_archive_bytes,
        "buffered_target_score_if_distortion_unchanged": current_distortion_score
        + LAMBDA_RATE * buffered_target_archive_bytes,
    }


def _target_pressure_for_stream(
    *,
    encoded_bytes: int,
    target_gap: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not target_gap or not target_gap.get("available"):
        return None
    bytes_to_remove = int(target_gap["bytes_to_remove_if_distortion_unchanged"])
    if bytes_to_remove <= 0:
        return {
            "bytes_to_remove_if_distortion_unchanged": 0,
            "percent_of_this_stream": 0.0,
            "stream_can_close_gap_alone_by_bytes": True,
            "stream_bytes_after_gap_if_alone": encoded_bytes,
        }
    return {
        "bytes_to_remove_if_distortion_unchanged": bytes_to_remove,
        "buffered_bytes_to_remove_if_distortion_unchanged": int(
            target_gap.get("buffered_bytes_to_remove_if_distortion_unchanged", bytes_to_remove)
        ),
        "percent_of_this_stream": None if encoded_bytes <= 0 else round(bytes_to_remove / encoded_bytes, 12),
        "stream_can_close_gap_alone_by_bytes": bool(encoded_bytes >= bytes_to_remove),
        "stream_bytes_after_gap_if_alone": int(encoded_bytes - bytes_to_remove),
    }


def _stream_self_compression_signal(
    *,
    encoded_bytes: int,
    encoded_probe: dict[str, Any],
    decoded_probe: dict[str, Any],
) -> dict[str, Any]:
    best_encoded = int(encoded_probe["best_probe"]["bytes"])
    best_decoded = int(decoded_probe["best_probe"]["bytes"])
    return {
        "best_nested_recompression_savings_bytes": max(0, encoded_bytes - best_encoded),
        "best_decoded_reencode_savings_vs_current_encoded_bytes": encoded_bytes - best_decoded,
        "encoded_probe_is_directly_deployable": False,
        "decoded_probe_is_directly_deployable": False,
        "deployability_note": (
            "Compression probes are attack-surface signals only. A deployed saving requires "
            "a decoder/unpacker contract that consumes the new bytes and an exact CUDA auth eval."
        ),
    }


def _first_present(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate_cost_or_none(byte_count: int | None) -> float | None:
    return None if byte_count is None else round(LAMBDA_RATE * byte_count, 12)


def _component_score_from_sample(sample: dict[str, Any]) -> float | None:
    direct = _float_or_none(
        _first_present(
            sample,
            (
                "score_combined_contribution_first_order",
                "score_combined_contribution",
                "component_score_contribution",
                "score_contribution",
            ),
        )
    )
    if direct is not None:
        return direct
    pose = _float_or_none(
        _first_present(
            sample,
            ("score_pose_contribution_first_order", "score_pose_contribution", "pose_score_contribution"),
        )
    )
    seg = _float_or_none(
        _first_present(sample, ("score_seg_contribution_exact", "score_seg_contribution", "seg_score_contribution"))
    )
    if pose is None and seg is None:
        return None
    return float(pose or 0.0) + float(seg or 0.0)


def _trace_archive_sha(payload: dict[str, Any]) -> str | None:
    for container_name in ("trace_inputs", "provenance", "eval_provenance"):
        container = payload.get(container_name)
        if isinstance(container, dict):
            value = container.get("archive_sha256") or container.get("archive_sha")
            if value is not None:
                return str(value)
    return None


def _sample_key(sample: dict[str, Any]) -> tuple[int, str]:
    pair_index = _int_or_none(sample.get("pair_index"))
    return (pair_index if pair_index is not None else 10**9, json.dumps(sample, sort_keys=True, default=str))


def _trace_sample_rows(payload: dict[str, Any], *, limit: int = 20) -> list[dict[str, Any]]:
    by_pair: dict[int, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for key in ("top_combined_samples", "top_excess_combined_samples", "top_pose_samples", "top_seg_samples"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            pair_index = _int_or_none(row.get("pair_index"))
            if pair_index is None or pair_index in by_pair:
                continue
            by_pair[pair_index] = row
            ordered.append(row)
    if len(ordered) < limit and isinstance(payload.get("samples"), list):
        candidates = [row for row in payload["samples"] if isinstance(row, dict)]
        candidates.sort(
            key=lambda row: (
                -float(_component_score_from_sample(row) or 0.0),
                _sample_key(row),
            )
        )
        for row in candidates:
            pair_index = _int_or_none(row.get("pair_index"))
            if pair_index is None or pair_index in by_pair:
                continue
            by_pair[pair_index] = row
            ordered.append(row)
            if len(ordered) >= limit:
                break
    out: list[dict[str, Any]] = []
    for row in ordered[:limit]:
        score = _component_score_from_sample(row)
        break_even = None if score is None or score <= 0.0 else int(math.floor(score / LAMBDA_RATE))
        out.append(
            {
                "pair_index": _int_or_none(row.get("pair_index")),
                "frame_start": _int_or_none(row.get("frame_start")),
                "frame_indices": row.get("frame_indices"),
                "posenet_dist": _float_or_none(row.get("posenet_dist")),
                "segnet_dist": _float_or_none(row.get("segnet_dist")),
                "score_pose_contribution_first_order": _float_or_none(
                    _first_present(
                        row,
                        ("score_pose_contribution_first_order", "score_pose_contribution", "pose_score_contribution"),
                    )
                ),
                "score_seg_contribution_exact": _float_or_none(
                    _first_present(row, ("score_seg_contribution_exact", "score_seg_contribution", "seg_score_contribution"))
                ),
                "score_combined_contribution_first_order": score,
                "break_even_bytes_at_rate_only": break_even,
            }
        )
    return out


def _component_trace_record(
    *,
    component_trace: Path | None,
    profiled_archive_size_bytes: int,
    profiled_archive_sha256: str,
) -> dict[str, Any] | None:
    if component_trace is None:
        return None
    payload = json.loads(component_trace.read_text(encoding="utf-8"))
    trace_sha = _json_file_sha256(component_trace)
    archive_bytes = _int_or_none(payload.get("archive_size_bytes"))
    archive_sha = _trace_archive_sha(payload)
    size_matches = archive_bytes is None or archive_bytes == int(profiled_archive_size_bytes)
    sha_matches = archive_sha is None or archive_sha == profiled_archive_sha256
    hard_pairs = _trace_sample_rows(payload)
    return {
        "path": str(component_trace.resolve()),
        "sha256": trace_sha,
        "schema_version": payload.get("schema_version"),
        "score_claim": payload.get("score_claim"),
        "evidence_grade": payload.get("evidence_grade"),
        "promotion_policy": payload.get("promotion_policy"),
        "n_samples": payload.get("n_samples"),
        "archive_size_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "profiled_archive_size_bytes": int(profiled_archive_size_bytes),
        "profiled_archive_sha256": profiled_archive_sha256,
        "matches_profiled_archive": bool(size_matches and sha_matches),
        "reference_warning": None
        if size_matches and sha_matches
        else "component_trace archive bytes/SHA do not match the profiled archive; use as reference-only pressure",
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "score_pose_contribution": payload.get("score_pose_contribution"),
        "score_seg_contribution": payload.get("score_seg_contribution"),
        "score_rate_contribution": payload.get("score_rate_contribution"),
        "score_recomputed_from_components": payload.get("score_recomputed_from_components"),
        "hard_pair_atoms": hard_pairs,
    }


ACTION_RECORD_KEYS = {
    "action_records",
    "actions",
    "atoms",
    "atom_records",
    "atom_table",
    "selected_atoms",
    "selected_actions",
    "selected_records",
    "candidate_actions",
    "component_atoms",
}


def _walk_action_record_lists(payload: Any, *, path: str = "$") -> Iterable[tuple[str, list[dict[str, Any]]]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}"
            if key in ACTION_RECORD_KEYS and isinstance(value, list) and all(isinstance(row, dict) for row in value):
                yield child_path, value
            yield from _walk_action_record_lists(value, path=child_path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from _walk_action_record_lists(value, path=f"{path}[{index}]")


def _score_delta_from_action(row: dict[str, Any]) -> float | None:
    direct = _float_or_none(
        _first_present(
            row,
            (
                "score_delta",
                "known_score_delta",
                "predicted_score_delta",
                "estimated_score_delta",
                "component_score_delta",
            ),
        )
    )
    if direct is not None:
        return direct
    pose = _float_or_none(
        _first_present(row, ("score_pose_delta", "pose_score_delta", "known_pose_score_delta"))
    )
    seg = _float_or_none(_first_present(row, ("score_seg_delta", "seg_score_delta", "known_seg_score_delta")))
    rate = _float_or_none(_first_present(row, ("score_rate_delta", "rate_score_delta")))
    if pose is None and seg is None and rate is None:
        return None
    return float(pose or 0.0) + float(seg or 0.0) + float(rate or 0.0)


def _normalize_action_record(row: dict[str, Any], *, source: dict[str, Any], index: int) -> dict[str, Any]:
    atom_id = _first_present(row, ("atom_id", "action_id", "id", "name", "label"))
    byte_cost = _int_or_none(
        _first_present(
            row,
            (
                "charged_bytes",
                "encoded_bytes",
                "compressed_bytes",
                "bytes",
                "byte_cost",
                "payload_bytes",
            ),
        )
    )
    byte_delta = _int_or_none(
        _first_present(row, ("byte_delta", "delta_bytes", "archive_delta_bytes", "bytes_delta"))
    )
    if byte_delta is None and byte_cost is not None:
        byte_delta = byte_cost
    score_delta = _score_delta_from_action(row)
    benefit = _float_or_none(
        _first_present(
            row,
            (
                "score_benefit",
                "benefit_score",
                "estimated_benefit_score",
                "delta_combined",
                "combined_delta",
            ),
        )
    )
    if benefit is None and score_delta is not None:
        benefit = -score_delta
    if benefit is None and byte_delta is not None and byte_delta < 0:
        benefit = -float(byte_delta) * LAMBDA_RATE
    benefit_per_byte = None
    denom = byte_cost if byte_cost and byte_cost > 0 else (abs(byte_delta) if byte_delta else None)
    if benefit is not None and denom:
        benefit_per_byte = benefit / denom
    return {
        "row_type": "action_record",
        "atom_id": str(atom_id) if atom_id is not None else f"action:{index:06d}",
        "name": str(_first_present(row, ("name", "label", "stream", "component", "kind")) or atom_id or f"action:{index:06d}"),
        "stream": _first_present(row, ("stream", "member", "payload_member", "component")),
        "bytes": byte_cost,
        "byte_delta": byte_delta,
        "sha256": _first_present(row, ("sha256", "payload_sha256", "encoded_sha256")),
        "rate_score_cost": _rate_cost_or_none(byte_cost),
        "rate_score_delta": _rate_cost_or_none(byte_delta),
        "known_posenet_delta": _float_or_none(
            _first_present(row, ("posenet_delta", "pose_delta", "delta_pose", "avg_posenet_delta"))
        ),
        "known_segnet_delta": _float_or_none(
            _first_present(row, ("segnet_delta", "seg_delta", "delta_seg", "avg_segnet_delta"))
        ),
        "known_score_delta": score_delta,
        "benefit_score_estimate": benefit,
        "benefit_per_byte_estimate": None if benefit_per_byte is None else round(benefit_per_byte, 12),
        "break_even_bytes_at_rate_only": None
        if benefit is None or benefit <= 0
        else int(math.floor(benefit / LAMBDA_RATE)),
        "source_path": source["path"],
        "source_sha256": source["sha256"],
        "source_json_pointer": source["json_pointer"],
        "raw_record": row,
    }


def _load_action_record_inputs(paths: Iterable[Path] | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths or ():
        payload = json.loads(path.read_text(encoding="utf-8"))
        file_record = {"path": str(path.resolve()), "sha256": _json_file_sha256(path)}
        for pointer, rows in _walk_action_record_lists(payload):
            for row in rows:
                records.append(
                    _normalize_action_record(
                        row,
                        source={**file_record, "json_pointer": pointer},
                        index=len(records),
                    )
                )
    records.sort(key=lambda row: (str(row["source_path"]), str(row["source_json_pointer"]), str(row["atom_id"])))
    return records


def _build_rate_distortion_atom_table(
    *,
    archive: dict[str, Any],
    zip_members: list[dict[str, Any]],
    zip_overhead: int,
    payload_record: dict[str, Any],
    stream_records: list[dict[str, Any]],
    component_trace_record: dict[str, Any] | None,
    action_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "row_type": "archive",
            "atom_id": "archive",
            "name": Path(str(archive["path"])).name,
            "bytes": archive["bytes"],
            "byte_delta": None,
            "sha256": archive["sha256"],
            "rate_score_cost": archive["rate_score_contribution"],
            "rate_score_delta": None,
            "known_posenet_delta": None,
            "known_segnet_delta": None,
            "known_score_delta": None,
            "benefit_score_estimate": None,
            "benefit_per_byte_estimate": None,
            "break_even_bytes_at_rate_only": None,
            "source_path": archive["path"],
            "source_sha256": archive["sha256"],
        },
        {
            "row_type": "zip_overhead",
            "atom_id": "zip_overhead",
            "name": "zip_overhead",
            "bytes": zip_overhead,
            "byte_delta": None,
            "sha256": None,
            "rate_score_cost": _rate_cost_or_none(zip_overhead),
            "rate_score_delta": None,
            "known_posenet_delta": None,
            "known_segnet_delta": None,
            "known_score_delta": None,
            "benefit_score_estimate": None,
            "benefit_per_byte_estimate": None,
            "break_even_bytes_at_rate_only": None,
            "source_path": archive["path"],
            "source_sha256": archive["sha256"],
        },
        {
            "row_type": "payload_overhead",
            "atom_id": "payload_internal_overhead",
            "name": "payload_internal_overhead",
            "bytes": payload_record["payload_internal_overhead_bytes"],
            "byte_delta": None,
            "sha256": None,
            "rate_score_cost": _rate_cost_or_none(payload_record["payload_internal_overhead_bytes"]),
            "rate_score_delta": None,
            "known_posenet_delta": None,
            "known_segnet_delta": None,
            "known_score_delta": None,
            "benefit_score_estimate": None,
            "benefit_per_byte_estimate": None,
            "break_even_bytes_at_rate_only": None,
            "source_path": archive["path"],
            "source_sha256": archive["sha256"],
        },
    ]
    for member in zip_members:
        rows.append(
            {
                "row_type": "zip_member",
                "atom_id": f"zip_member:{member['filename']}",
                "name": member["filename"],
                "bytes": member["compress_size"],
                "byte_delta": None,
                "sha256": None,
                "rate_score_cost": _rate_cost_or_none(int(member["compress_size"])),
                "rate_score_delta": None,
                "known_posenet_delta": None,
                "known_segnet_delta": None,
                "known_score_delta": None,
                "benefit_score_estimate": None,
                "benefit_per_byte_estimate": None,
                "break_even_bytes_at_rate_only": None,
                "source_path": archive["path"],
                "source_sha256": archive["sha256"],
            }
        )
    for stream in stream_records:
        savings = int(stream["self_compression_signal"]["best_nested_recompression_savings_bytes"])
        benefit = savings * LAMBDA_RATE if savings > 0 else None
        rows.append(
            {
                "row_type": "packed_stream",
                "atom_id": f"packed_stream:{stream['name']}",
                "name": stream["name"],
                "bytes": stream["encoded_bytes"],
                "byte_delta": None,
                "sha256": stream.get("encoded_sha256"),
                "rate_score_cost": stream["attackability"]["rate_score_contribution"],
                "rate_score_delta": None,
                "known_posenet_delta": None,
                "known_segnet_delta": None,
                "known_score_delta": None,
                "benefit_score_estimate": None if benefit is None else round(benefit, 12),
                "benefit_per_byte_estimate": None if benefit is None else round(LAMBDA_RATE, 12),
                "break_even_bytes_at_rate_only": None,
                "byte_savings_estimate": savings,
                "codec": stream.get("codec"),
                "source_path": archive["path"],
                "source_sha256": archive["sha256"],
            }
        )
    rows.extend(action_records)
    if component_trace_record is not None:
        for row in component_trace_record["hard_pair_atoms"]:
            benefit = row.get("score_combined_contribution_first_order")
            rows.append(
                {
                    "row_type": "component_trace_pair",
                    "atom_id": f"component_trace_pair:{row['pair_index']}",
                    "name": f"pair_{row['pair_index']}",
                    "bytes": None,
                    "byte_delta": None,
                    "sha256": None,
                    "rate_score_cost": None,
                    "rate_score_delta": None,
                    "known_posenet_delta": row.get("posenet_dist"),
                    "known_segnet_delta": row.get("segnet_dist"),
                    "known_score_delta": None,
                    "benefit_score_estimate": benefit,
                    "benefit_per_byte_estimate": None,
                    "break_even_bytes_at_rate_only": row.get("break_even_bytes_at_rate_only"),
                    "frame_start": row.get("frame_start"),
                    "frame_indices": row.get("frame_indices"),
                    "source_path": component_trace_record["path"],
                    "source_sha256": component_trace_record["sha256"],
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["row_type"]),
            -(int(row["bytes"]) if row.get("bytes") is not None else -1),
            str(row["atom_id"]),
        )
    )
    return rows


def _write_csv_table(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "row_type",
        "atom_id",
        "name",
        "stream",
        "bytes",
        "byte_delta",
        "sha256",
        "rate_score_cost",
        "rate_score_delta",
        "known_posenet_delta",
        "known_segnet_delta",
        "known_score_delta",
        "benefit_score_estimate",
        "benefit_per_byte_estimate",
        "break_even_bytes_at_rate_only",
        "byte_savings_estimate",
        "codec",
        "candidate_index",
        "candidate_archive_path",
        "candidate_archive_sha256",
        "source_path",
        "source_sha256",
        "source_json_pointer",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_profile(
    *,
    archive: Path,
    output_json: Path,
    eval_json: Path | None = None,
    component_trace: Path | None = None,
    action_jsons: Iterable[Path] | None = None,
    target_score: float = DEFAULT_TARGET_SCORE,
) -> dict[str, Any]:
    archive = archive.resolve()
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")
    payload_name, zip_member_payload = _extract_single_payload(archive)
    archive_sha256 = _sha256_file(archive)
    payload, payload_container_record = _normalize_payload_member(payload_name, zip_member_payload)
    public_pr65 = _encoded_slices_from_public_pr65_bundle(payload_name, payload)
    if public_pr65 is not None:
        header, decoded_members, encoded_slices = public_pr65
    else:
        unpacker = _load_unpacker()
        header, decoded_members = unpacker._parse_payload(payload)  # noqa: SLF001 - contest runtime parser
        encoded_slices = _encoded_slices_from_header(payload, header)
    decoded_by_name = {name: bytes(data) for name, data in decoded_members.items()}
    eval_payload: dict[str, Any] | None = None
    if eval_json is not None:
        eval_payload = json.loads(eval_json.read_text())
    archive_bytes = archive.stat().st_size
    target_gap = _target_gap_record(
        eval_payload=eval_payload,
        target_score=target_score,
        profiled_archive_size_bytes=int(archive_bytes),
        profiled_archive_sha256=archive_sha256,
    )
    trace_record = _component_trace_record(
        component_trace=component_trace,
        profiled_archive_size_bytes=int(archive_bytes),
        profiled_archive_sha256=archive_sha256,
    )
    action_records = _load_action_record_inputs(action_jsons)

    stream_records: list[dict[str, Any]] = []
    for item in encoded_slices:
        name = str(item["name"])
        start = int(item["encoded_offset"])
        end = start + int(item["encoded_bytes"])
        encoded_data = payload[start:end]
        decoded_data = decoded_by_name.get(name, b"")
        encoded_probe = _compression_probe(encoded_data)
        decoded_probe = _compression_probe(decoded_data)
        stream_records.append(
            {
                **item,
                "encoded_entropy": _byte_entropy_record(encoded_data),
                "decoded_entropy": _byte_entropy_record(decoded_data),
                "encoded_bitplanes": _bitplane_record(encoded_data),
                "decoded_bitplanes": _bitplane_record(decoded_data),
                "encoded_self_compression_probe": encoded_probe,
                "decoded_self_compression_probe": decoded_probe,
                "self_compression_signal": _stream_self_compression_signal(
                    encoded_bytes=int(item["encoded_bytes"]),
                    encoded_probe=encoded_probe,
                    decoded_probe=decoded_probe,
                ),
                "target_gap_pressure": _target_pressure_for_stream(
                    encoded_bytes=int(item["encoded_bytes"]),
                    target_gap=target_gap,
                ),
                "attackability": _stream_attackability(
                    name,
                    item.get("codec"),
                    int(item["encoded_bytes"]),
                    int(item.get("decoded_bytes") or len(decoded_data)),
                ),
            }
        )
    stream_records.sort(key=lambda item: (-int(item["encoded_bytes"]), str(item["name"])))

    zip_members, zip_overhead = _zip_member_overhead(archive)
    payload_stream_bytes = sum(int(item["encoded_bytes"]) for item in stream_records)
    eval_record: dict[str, Any] | None = None
    if eval_json is not None and eval_payload is not None:
        eval_record = {
            "path": str(eval_json.resolve()),
            "archive_size_bytes": eval_payload.get("archive_size_bytes"),
            "archive_sha256": (eval_payload.get("provenance") or {}).get("archive_sha256"),
            "profiled_archive_size_bytes": int(archive_bytes),
            "profiled_archive_sha256": archive_sha256,
            "matches_profiled_archive": None if target_gap is None else target_gap.get("reference_matches_profiled_archive"),
            "reference_warning": None if target_gap is None else target_gap.get("reference_warning"),
            "score_recomputed_from_components": eval_payload.get("score_recomputed_from_components"),
            "avg_segnet_dist": eval_payload.get("avg_segnet_dist"),
            "avg_posenet_dist": eval_payload.get("avg_posenet_dist"),
            "n_samples": eval_payload.get("n_samples"),
            "gpu_model": (eval_payload.get("provenance") or {}).get("gpu_model"),
            "gpu_t4_match": (eval_payload.get("provenance") or {}).get("gpu_t4_match"),
            "target_gap": target_gap,
        }

    payload_record = {
        "zip_member_name": payload_name,
        **payload_container_record,
        "payload_bytes": int(len(payload)),
        "payload_sha256": _sha256_bytes(payload),
        "payload_format": header.get("payload_format"),
        "payload_schema": header.get("schema"),
        "payload_stream_encoded_bytes_sum": int(payload_stream_bytes),
        "payload_internal_overhead_bytes": int(len(payload) - payload_stream_bytes),
        "payload_entropy": _byte_entropy_record(payload),
        "payload_self_compression_probe": _compression_probe(payload),
        "zip_member_payload_entropy": _byte_entropy_record(zip_member_payload),
        "zip_member_payload_self_compression_probe": _compression_probe(zip_member_payload),
        "finite_container_probe": _finite_container_probe(
            parser_payload=payload,
            source_archive_bytes=int(archive_bytes),
        ),
    }
    archive_record = {
        "path": str(archive),
        "bytes": int(archive_bytes),
        "sha256": archive_sha256,
        "rate_score_contribution": round(LAMBDA_RATE * archive_bytes, 12),
        "lambda_rate": LAMBDA_RATE,
        "target_score": target_score,
    }
    atom_table = _build_rate_distortion_atom_table(
        archive=archive_record,
        zip_members=zip_members,
        zip_overhead=zip_overhead,
        payload_record=payload_record,
        stream_records=stream_records,
        component_trace_record=trace_record,
        action_records=action_records,
    )
    action_inputs = []
    seen_action_inputs: set[tuple[str, str, str]] = set()
    for row in action_records:
        key = (str(row["source_path"]), str(row["source_sha256"]), str(row["source_json_pointer"]))
        if key in seen_action_inputs:
            continue
        seen_action_inputs.add(key)
        action_inputs.append(
            {
                "path": row["source_path"],
                "sha256": row["source_sha256"],
                "json_pointer": row["source_json_pointer"],
            }
        )
    ranked_opportunities = sorted(
        [
            {
                "stream": item["name"],
                "encoded_bytes": item["encoded_bytes"],
                "rate_score_contribution": item["attackability"]["rate_score_contribution"],
                "priority": item["attackability"]["priority"],
                "risk": item["attackability"]["risk"],
                "opportunity": item["attackability"]["opportunity"],
                "best_encoded_recompression_delta": item["encoded_self_compression_probe"]["best_probe_delta_vs_input"],
                "best_nested_recompression_savings_bytes": item["self_compression_signal"]["best_nested_recompression_savings_bytes"],
                "best_decoded_reencode_savings_vs_current_encoded_bytes": item["self_compression_signal"]["best_decoded_reencode_savings_vs_current_encoded_bytes"],
                "decoded_zero_order_entropy_bytes": item["decoded_entropy"]["zero_order_entropy_bytes"],
                "target_gap_percent_of_stream": (
                    item["target_gap_pressure"] or {}
                ).get("percent_of_this_stream"),
            }
            for item in stream_records
        ],
        key=lambda item: (int(item["priority"]), -int(item["encoded_bytes"])),
    )
    payload = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "canonical_score_source_required": CONTEST_SCORE_SOURCE,
        "archive": archive_record,
        "eval_json": eval_record,
        "component_trace": trace_record,
        "target_gap": target_gap,
        "zip_container": {
            "member_count": len(zip_members),
            "members": zip_members,
            "overhead_bytes": int(zip_overhead),
            "overhead_rate_score": round(LAMBDA_RATE * zip_overhead, 12),
        },
        "single_payload": payload_record,
        "streams": stream_records,
        "action_record_inputs": action_inputs,
        "byte_ledger": atom_table,
        "rate_distortion_atom_table": atom_table,
        "ranked_self_compression_opportunities": ranked_opportunities,
        "thirty_k_foot_summary": {
            "archive_is_single_blob": len(zip_members) == 1,
            "largest_stream": stream_records[0]["name"] if stream_records else None,
            "zip_overhead_is_material": bool(zip_overhead > 1024),
            "primary_attack_surface": "mask stream with scorer-constrained learned/grammar repair, not human visual smoothing",
            "secondary_attack_surface": "renderer self-compression only after local byte win",
            "tertiary_attack_surface": "pose bytes and archive overhead are too small to close the full sub-0.300 gap alone",
            "bytes_to_remove_if_distortion_unchanged": (
                None if target_gap is None else target_gap.get("bytes_to_remove_if_distortion_unchanged")
            ),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_bytes(_json_bytes(payload))
    return payload


def _markdown_report(payload: dict[str, Any]) -> str:
    archive = payload["archive"]
    target_gap = payload.get("target_gap") or {}
    lines = [
        "# Archive Byte Accounting",
        "",
        "This report is empirical byte profiling only. It is not score evidence and does not promote a candidate.",
        "",
        "## Archive",
        "",
        f"- path: `{archive['path']}`",
        f"- bytes: `{archive['bytes']}`",
        f"- sha256: `{archive['sha256']}`",
        f"- rate contribution: `{archive['rate_score_contribution']}`",
        f"- score claim: `{payload['score_claim']}`",
        f"- promotion eligible: `{payload['promotion_eligible']}`",
    ]
    if target_gap.get("available"):
        reference_warning = target_gap.get("reference_warning")
        lines.extend(
            [
                "",
                "## Sub-0.300 Byte Gap",
                "",
                f"- exact crossing bytes at unchanged distortion: `{target_gap['bytes_to_remove_if_distortion_unchanged']}`",
                f"- buffered planning bytes at unchanged distortion: `{target_gap['buffered_bytes_to_remove_if_distortion_unchanged']}`",
                f"- buffered target archive bytes: `{target_gap['buffered_target_archive_size_bytes_if_distortion_unchanged']}`",
                f"- buffered target score: `{target_gap['buffered_target_score_if_distortion_unchanged']}`",
                f"- scored reference matches profiled archive: `{target_gap['reference_matches_profiled_archive']}`",
            ]
        )
        if reference_warning:
            lines.append(f"- reference warning: `{reference_warning}`")
    lines.extend(
        [
            "",
            "## Container",
            "",
            f"- zip overhead bytes: `{payload['zip_container']['overhead_bytes']}`",
            f"- payload bytes: `{payload['single_payload']['payload_bytes']}`",
            f"- payload format: `{payload['single_payload']['payload_format']}`",
            f"- payload internal overhead bytes: `{payload['single_payload']['payload_internal_overhead_bytes']}`",
            "",
            "## Streams",
            "",
            "| stream | encoded bytes | decoded bytes | codec | rate score | entropy b/B | bitplane entropy bytes | buffered gap % of stream | best nested savings |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stream in payload["streams"]:
        pressure = stream.get("target_gap_pressure") or {}
        lines.append(
            "| {name} | {encoded} | {decoded} | {codec} | {rate} | {entropy} | {bitentropy} | {gap_pct} | {savings} |".format(
                name=stream["name"],
                encoded=stream["encoded_bytes"],
                decoded=stream["decoded_bytes"],
                codec=stream.get("codec"),
                rate=stream["attackability"]["rate_score_contribution"],
                entropy=stream["encoded_entropy"]["entropy_bits_per_byte"],
                bitentropy=stream["encoded_bitplanes"]["zero_order_bitplane_entropy_bytes"],
                gap_pct=pressure.get("percent_of_this_stream"),
                savings=stream["self_compression_signal"]["best_nested_recompression_savings_bytes"],
            )
        )
    lines.extend(
        [
            "",
            "## Dispatch Interpretation",
            "",
            "- Direct nested compression is exhausted for the current byte streams.",
            "- The mask stream is the only single stream large enough to close the gap with a modest relative byte change.",
            "- Renderer self-compression must beat the existing QZS/QZS4-style local byte baseline before exact eval.",
            "- Pose bytes are already too small to close the sub-0.300 gap alone.",
            "- Any useful self-compression must change the representation/decoder grammar while charging all decoder bits inside the archive.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_png_report(payload: dict[str, Any], output_png: Path) -> None:
    """Write a compact stream-byte and sub-target pressure figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    streams = list(payload.get("streams") or [])
    streams.sort(key=lambda item: int(item["encoded_bytes"]))
    names = [str(item["name"]) for item in streams]
    encoded = [int(item["encoded_bytes"]) for item in streams]
    rates = [float(item["attackability"]["rate_score_contribution"]) for item in streams]
    target_gap = payload.get("target_gap") or {}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    y = list(range(len(names)))
    ax.barh(y, encoded, color="#2563eb")
    ax.set_yticks(y, names)
    ax.set_xlabel("encoded bytes")
    ax.set_title("Archive stream byte allocation")
    for idx, value in enumerate(encoded):
        ax.text(value, idx, f" {value:,}", va="center", fontsize=9)

    ax2 = axes[1]
    ax2.barh(y, rates, color="#16a34a")
    ax2.set_yticks(y, names)
    ax2.set_xlabel("contest rate-score contribution")
    ax2.set_title("Rate term by stream")
    for idx, value in enumerate(rates):
        ax2.text(value, idx, f" {value:.6f}", va="center", fontsize=9)

    if target_gap.get("available"):
        bytes_to_remove = int(target_gap["bytes_to_remove_if_distortion_unchanged"])
        for axis in axes:
            axis.axvline(
                bytes_to_remove if axis is ax else LAMBDA_RATE * bytes_to_remove,
                color="#dc2626",
                linestyle="--",
                linewidth=1.2,
            )
        fig.suptitle(
            "Archive byte accounting: "
            f"{payload['archive']['bytes']:,} bytes; "
            f"{bytes_to_remove:,} bytes to remove for target "
            f"{target_gap['target_score']:.3f} at unchanged distortion",
            fontsize=12,
        )
    else:
        fig.suptitle(
            f"Archive byte accounting: {payload['archive']['bytes']:,} bytes",
            fontsize=12,
        )

    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _sidecar_for_index(paths: list[Path], index: int, archive_count: int, label: str) -> Path | None:
    if not paths:
        return None
    if len(paths) == 1 and archive_count == 1:
        return paths[0]
    if len(paths) == archive_count:
        return paths[index]
    raise ValueError(
        f"{label} count must be 0, 1 for a single archive, or match --archive count; "
        f"got {len(paths)} for {archive_count} archive(s)"
    )


def _profile_output_path(output_dir: Path, archive: Path, index: int) -> Path:
    stem = archive.stem or "archive"
    return output_dir / f"{index:03d}_{stem}_byte_accounting.json"


def build_profile_collection(
    *,
    archives: list[Path],
    output_json: Path,
    eval_jsons: Iterable[Path] | None = None,
    component_traces: Iterable[Path] | None = None,
    action_jsons: Iterable[Path] | None = None,
    output_dir: Path | None = None,
    target_score: float = DEFAULT_TARGET_SCORE,
) -> dict[str, Any]:
    eval_list = list(eval_jsons or ())
    trace_list = list(component_traces or ())
    action_list = list(action_jsons or ())
    if not archives:
        raise ValueError("at least one --archive is required")
    output_dir = output_dir or output_json.with_suffix("")
    profiles: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for index, archive in enumerate(archives):
        profile_json = _profile_output_path(output_dir, archive, index)
        profile = build_profile(
            archive=archive,
            output_json=profile_json,
            eval_json=_sidecar_for_index(eval_list, index, len(archives), "--eval-json"),
            component_trace=_sidecar_for_index(trace_list, index, len(archives), "--component-trace"),
            action_jsons=action_list,
            target_score=target_score,
        )
        profile["collection_profile_json"] = str(profile_json.resolve())
        profiles.append(profile)
        for row in profile["rate_distortion_atom_table"]:
            combined_rows.append(
                {
                    **row,
                    "candidate_index": index,
                    "candidate_archive_path": profile["archive"]["path"],
                    "candidate_archive_sha256": profile["archive"]["sha256"],
                }
            )
    combined_rows.sort(
        key=lambda row: (
            int(row["candidate_index"]),
            str(row["row_type"]),
            -(int(row["bytes"]) if row.get("bytes") is not None else -1),
            str(row["atom_id"]),
        )
    )
    payload = {
        "schema": "archive_byte_accounting_profile_collection_v1",
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "canonical_score_source_required": CONTEST_SCORE_SOURCE,
        "archive_count": len(profiles),
        "archives": [
            {
                **profile["archive"],
                "collection_profile_json": profile["collection_profile_json"],
                "eval_json": None if profile.get("eval_json") is None else profile["eval_json"]["path"],
                "component_trace": None
                if profile.get("component_trace") is None
                else profile["component_trace"]["path"],
            }
            for profile in profiles
        ],
        "profiles": profiles,
        "byte_ledger": combined_rows,
        "rate_distortion_atom_table": combined_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_bytes(_json_bytes(payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument(
        "--eval-json",
        type=Path,
        action="append",
        default=[],
        help="contest_auth_eval.json sidecar; repeat once per archive for collection mode",
    )
    parser.add_argument(
        "--component-trace",
        type=Path,
        action="append",
        default=[],
        help="component_trace.json sidecar; repeat once per archive for collection mode",
    )
    parser.add_argument(
        "--action-json",
        type=Path,
        action="append",
        default=[],
        help="optional local action/atom JSON to normalize into the rate-distortion atom table",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--output-csv", type=Path, help="write byte ledger/rate-distortion atom table as CSV")
    parser.add_argument("--output-dir", type=Path, help="collection mode per-archive profile directory")
    parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE)
    args = parser.parse_args(argv)
    if len(args.archive) > 1 and args.output_png is not None:
        raise ValueError("--output-png is only supported for single-archive profiling")
    if len(args.archive) == 1:
        payload = build_profile(
            archive=args.archive[0],
            eval_json=_sidecar_for_index(args.eval_json, 0, 1, "--eval-json"),
            component_trace=_sidecar_for_index(args.component_trace, 0, 1, "--component-trace"),
            action_jsons=args.action_json,
            output_json=args.output_json,
            target_score=args.target_score,
        )
        if args.output_md is not None:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(_markdown_report(payload), encoding="utf-8")
        if args.output_png is not None:
            _write_png_report(payload, args.output_png)
    else:
        payload = build_profile_collection(
            archives=args.archive,
            eval_jsons=args.eval_json,
            component_traces=args.component_trace,
            action_jsons=args.action_json,
            output_json=args.output_json,
            output_dir=args.output_dir,
            target_score=args.target_score,
        )
        if args.output_md is not None:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(
                "\n\n".join(_markdown_report(profile) for profile in payload["profiles"]),
                encoding="utf-8",
            )
    if args.output_csv is not None:
        _write_csv_table(payload["rate_distortion_atom_table"], args.output_csv)
    print(
        json.dumps(
            {
                "archive_count": payload.get("archive_count", 1),
                "archive_bytes": payload["archive"]["bytes"] if "archive" in payload else None,
                "output_json": str(args.output_json),
                "schema": payload["schema"],
                "score_claim": payload["score_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
