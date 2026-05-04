#!/usr/bin/env python3
"""Build local PR85 lossless pure-rate candidates.

This lane is deliberately narrow and local-only. It starts from the verified
public PR85 single-member ``x`` archive, profiles public PR86/PR90/PR91 anatomy
for context, and screens only transformations that preserve decoded PR85
payload semantics:

* strict ZIP container repacks of the unchanged ``x`` member;
* Brotli recodes of non-mask segments whose decoded payload is unchanged;
* canonical/reordered P1D1 pose streams whose decoded pose semantics are
  unchanged under the public PR85 replay parser.

The tool writes deterministic candidate archives only when a screened transform
is strictly byte-negative versus the PR85 frontier archive. It never loads
scorers, runs CUDA, claims score evidence, or dispatches remote work.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import io
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import (  # noqa: E402
    Pr85Bundle,
    Pr85BundleError,
    SEGMENT_ORDER,
    pack_pr85_bundle,
    parse_pr85_bundle,
    validate_pr85_member_name,
)
from tac.submission_archive import write_deterministic_zip_member  # noqa: E402


TOOL = "experiments/build_pr85_lossless_pure_rate_candidates.py"
SCHEMA = "pr85_lossless_pure_rate_candidates_v1"
MANIFEST_SCHEMA = "pr85_lossless_pure_rate_candidate_v1"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_PR85_EXACT_EVAL = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/"
    "contest_auth_eval.json"
)
DEFAULT_PUBLIC_ARCHIVES = {
    "PR86": REPO_ROOT / "experiments/results/public_pr86_intake_20260504_codex/archive.zip",
    "PR90": REPO_ROOT / "experiments/results/public_pr90_intake_20260504_worker/archive.zip",
    "PR91": REPO_ROOT / "experiments/results/public_pr91_intake_20260504_worker/archive.zip",
}
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_lossless_pure_rate_candidates_20260504_codex"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
NON_MASK_SEGMENTS = tuple(name for name in SEGMENT_ORDER if name != "mask")


class LosslessPureRateError(RuntimeError):
    """Raised when a PR85 pure-rate candidate cannot be screened safely."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LosslessPureRateError(f"expected JSON object: {_repo_rel(path)}")
    return payload


def _rate_delta(delta_bytes: int) -> float:
    return float(delta_bytes) * RATE_SCORE_PER_BYTE


def _parse_int_csv(text: str, *, label: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            values.append(int(stripped))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    return tuple(values)


def _read_single_x_archive(path: Path) -> tuple[dict[str, Any], bytes, zipfile.ZipInfo]:
    if not path.is_file():
        raise LosslessPureRateError(f"source archive not found: {_repo_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise LosslessPureRateError(
                f"PR85 source must contain exactly one non-directory member; got {len(infos)}"
            )
        info = infos[0]
        validate_pr85_member_name(info.filename)
        raw = zf.read(info)
    meta = {
        "path": _repo_rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "member_name": info.filename,
        "member_file_size": int(info.file_size),
        "member_compress_size": int(info.compress_size),
        "member_crc32_hex": f"{info.CRC:08x}",
        "member_sha256": _sha256_bytes(raw),
        "zip_compress_type": int(info.compress_type),
        "zip_flag_bits": int(info.flag_bits),
        "zip_timestamp": list(info.date_time),
    }
    return meta, raw, info


def _write_single_x_archive(
    path: Path,
    x_payload: bytes,
    *,
    compress_type: int = zipfile.ZIP_STORED,
    compresslevel: int | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=compress_type) as zf:
        write_deterministic_zip_member(
            zf,
            "x",
            x_payload,
            compress_type=compress_type,
            compresslevel=compresslevel,
        )
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if [info.filename for info in infos] != ["x"]:
            raise LosslessPureRateError("candidate archive wrote the wrong member set")
        info = infos[0]
        raw = zf.read("x")
    if raw != x_payload:
        raise LosslessPureRateError("candidate archive does not round-trip its x member")
    return {
        "path": _repo_rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "member_name": "x",
        "member_file_size": int(info.file_size),
        "member_compress_size": int(info.compress_size),
        "member_crc32_hex": f"{info.CRC:08x}",
        "member_sha256": _sha256_bytes(raw),
        "zip_compress_type": int(info.compress_type),
        "zip_flag_bits": int(info.flag_bits),
        "zip_timestamp": list(info.date_time),
    }


def _zip_bytes(
    x_payload: bytes,
    *,
    compress_type: int,
    compresslevel: int | None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compress_type) as zf:
        write_deterministic_zip_member(
            zf,
            "x",
            x_payload,
            compress_type=compress_type,
            compresslevel=compresslevel,
        )
    return buffer.getvalue()


def _zip_repack_rows(source_archive: Mapping[str, Any], x_raw: bytes) -> list[dict[str, Any]]:
    variants = [
        ("zip_stored_deterministic", zipfile.ZIP_STORED, None, True),
        ("zip_deflate_level1", zipfile.ZIP_DEFLATED, 1, True),
        ("zip_deflate_level6", zipfile.ZIP_DEFLATED, 6, True),
        ("zip_deflate_level9", zipfile.ZIP_DEFLATED, 9, True),
    ]
    rows: list[dict[str, Any]] = []
    for candidate_id, compress_type, compresslevel, strict_dispatchable in variants:
        encoded = _zip_bytes(x_raw, compress_type=compress_type, compresslevel=compresslevel)
        with zipfile.ZipFile(io.BytesIO(encoded), "r") as zf:
            info = zf.getinfo("x")
            member = zf.read("x")
        if member != x_raw:
            raise LosslessPureRateError(f"{candidate_id} changed x member bytes")
        delta = len(encoded) - int(source_archive["archive_bytes"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_kind": "zip_repack",
                "archive_bytes_formula": int(len(encoded)),
                "archive_sha256_formula": _sha256_bytes(encoded),
                "archive_delta_bytes_vs_source": int(delta),
                "rate_score_delta_if_components_identical_formula_only": _rate_delta(delta),
                "zip_compress_type": int(compress_type),
                "zip_compresslevel": compresslevel,
                "member_compress_size": int(info.compress_size),
                "member_file_size": int(info.file_size),
                "member_sha256": _sha256_bytes(member),
                "strict_dispatchable_zip_method": bool(strict_dispatchable),
                "decoded_semantics": "x_member_byte_identical",
            }
        )
    return rows


def _brotli_module():
    try:
        import brotli  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment guard
        raise LosslessPureRateError("brotli is required for PR85 segment recode screening") from exc
    return brotli


def _brotli_grid(data: bytes, *, qualities: Sequence[int], lgwins: Sequence[int]) -> list[dict[str, Any]]:
    brotli = _brotli_module()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for quality in qualities:
        for lgwin in lgwins:
            try:
                encoded = brotli.compress(data, quality=int(quality), lgwin=int(lgwin))
            except brotli.error:
                continue
            decoded = brotli.decompress(encoded)
            if decoded != data:
                raise LosslessPureRateError("Brotli recode did not preserve decoded bytes")
            digest = _sha256_bytes(encoded)
            key = (len(encoded), digest)
            rows.append(
                {
                    "codec": "brotli",
                    "quality": int(quality),
                    "lgwin": int(lgwin),
                    "bytes": int(len(encoded)),
                    "sha256": digest,
                    "payload": encoded,
                    "duplicate_stream": key in seen,
                }
            )
            seen.add(key)
    return rows


def _header_mode(bundle: Pr85Bundle) -> str:
    return "explicit_30" if bundle.format == "pr85_explicit_30byte_lengths" else "v5"


def _segment_recode_rows(
    source_archive: Mapping[str, Any],
    bundle: Pr85Bundle,
    *,
    qualities: Sequence[int],
    lgwins: Sequence[int],
) -> list[dict[str, Any]]:
    brotli = _brotli_module()
    rows: list[dict[str, Any]] = []
    for name in NON_MASK_SEGMENTS:
        source_segment = bytes(bundle.segments[name])
        try:
            decoded = brotli.decompress(source_segment)
        except brotli.error:
            rows.append(
                {
                    "candidate_id": f"{name}_brotli_recode_unavailable",
                    "candidate_kind": "segment_brotli_recode",
                    "segment_name": name,
                    "screen_status": "skipped_not_brotli_decodable",
                    "source_segment_bytes": int(len(source_segment)),
                    "source_segment_sha256": _sha256_bytes(source_segment),
                }
            )
            continue
        grid = _brotli_grid(decoded, qualities=qualities, lgwins=lgwins)
        best = min(grid, key=lambda row: (int(row["bytes"]), int(row["quality"]), int(row["lgwin"])))
        delta = int(best["bytes"]) - len(source_segment)
        rows.append(
            {
                "candidate_id": f"{name}_brq{best['quality']}_lg{best['lgwin']}",
                "candidate_kind": "segment_brotli_recode",
                "segment_name": name,
                "screen_status": "screened",
                "source_segment_bytes": int(len(source_segment)),
                "source_segment_sha256": _sha256_bytes(source_segment),
                "decoded_bytes": int(len(decoded)),
                "decoded_sha256": _sha256_bytes(decoded),
                "candidate_segment_bytes": int(best["bytes"]),
                "candidate_segment_sha256": best["sha256"],
                "candidate_brotli_quality": best["quality"],
                "candidate_brotli_lgwin": best["lgwin"],
                "candidate_segment_delta_bytes_vs_source": int(delta),
                "archive_delta_bytes_vs_source_formula": int(delta),
                "rate_score_delta_if_components_identical_formula_only": _rate_delta(delta),
                "decoded_semantics": "brotli_decoded_payload_byte_identical",
                "candidate_payload": best["payload"],
                "source_archive_bytes": int(source_archive["archive_bytes"]),
            }
        )
    return rows


def _read_uvarint(data: bytes, cursor: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while cursor < len(data):
        byte = int(data[cursor])
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 128:
            return value, cursor
        shift += 7
        if shift > 35:
            break
    raise LosslessPureRateError("truncated or overlong P1D1 varint")


def _write_uvarint(value: int) -> bytes:
    if value < 0:
        raise LosslessPureRateError(f"cannot encode negative uvarint: {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def _zigzag_encode(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def _parse_p1d1(raw: bytes, *, frame_count: int = 600) -> dict[str, Any]:
    if not raw.startswith(b"P1D1"):
        raise LosslessPureRateError(f"pose payload is not P1D1: {raw[:4]!r}")
    cursor = 4
    if cursor >= len(raw):
        raise LosslessPureRateError("P1D1 payload is missing stream count")
    stream_count = int(raw[cursor])
    cursor += 1
    dims: list[int] = []
    lengths: list[int] = []
    for _ in range(stream_count):
        if cursor + 3 > len(raw):
            raise LosslessPureRateError("P1D1 header is truncated")
        dim = int(raw[cursor])
        cursor += 1
        length = int.from_bytes(raw[cursor : cursor + 2], "little")
        cursor += 2
        if dim < 0 or dim >= 6:
            raise LosslessPureRateError(f"P1D1 dim out of bounds: {dim}")
        if dim in dims:
            raise LosslessPureRateError(f"P1D1 duplicate dim: {dim}")
        dims.append(dim)
        lengths.append(length)
    streams: dict[int, bytes] = {}
    q_by_dim: dict[int, list[int]] = {dim: [0] * frame_count for dim in range(6)}
    for dim, length in zip(dims, lengths):
        stream = raw[cursor : cursor + length]
        if len(stream) != length:
            raise LosslessPureRateError(f"P1D1 stream {dim} is truncated")
        cursor += length
        values: list[int] = []
        stream_cursor = 0
        while stream_cursor < len(stream) and len(values) < frame_count:
            value, stream_cursor = _read_uvarint(stream, stream_cursor)
            values.append(value)
        if len(values) != frame_count:
            raise LosslessPureRateError(
                f"P1D1 dim {dim} decoded {len(values)} values, expected {frame_count}"
            )
        acc = 0
        decoded_q: list[int] = []
        for value in values:
            acc += _zigzag_decode(value)
            if dim != 0:
                acc = max(-32768, min(32767, acc))
            decoded_q.append(int(acc))
        q_by_dim[dim] = decoded_q
        streams[dim] = bytes(stream)
    if cursor != len(raw):
        raise LosslessPureRateError(f"P1D1 payload has trailing bytes: {len(raw) - cursor}")
    return {
        "frame_count": frame_count,
        "dims": tuple(dims),
        "stream_lengths": {int(dim): int(length) for dim, length in zip(dims, lengths)},
        "streams": streams,
        "q_by_dim": q_by_dim,
    }


def _encode_p1d1(parsed: Mapping[str, Any], order: Sequence[int]) -> bytes:
    q_by_dim = parsed["q_by_dim"]
    header = bytearray(b"P1D1")
    header.append(len(order))
    body = bytearray()
    for dim in order:
        q_values = q_by_dim[dim]
        previous = 0
        stream = bytearray()
        for value in q_values:
            delta = int(value) - previous
            previous = int(value)
            stream += _write_uvarint(_zigzag_encode(delta))
        header.append(int(dim))
        header += len(stream).to_bytes(2, "little")
        body += stream
    return bytes(header + body)


def _p1d1_semantic_sha256(raw: bytes) -> str:
    parsed = _parse_p1d1(raw)
    out = bytearray()
    out += int(parsed["frame_count"]).to_bytes(2, "little")
    present = set(parsed["dims"])
    for dim in range(6):
        out.append(1 if dim in present else 0)
        for value in parsed["q_by_dim"][dim]:
            out += int(value).to_bytes(4, "little", signed=True)
    return _sha256_bytes(bytes(out))


def _pose_reorder_rows(
    source_archive: Mapping[str, Any],
    bundle: Pr85Bundle,
    *,
    qualities: Sequence[int],
    lgwins: Sequence[int],
) -> list[dict[str, Any]]:
    brotli = _brotli_module()
    source_segment = bytes(bundle.segments["pose"])
    try:
        source_raw = brotli.decompress(source_segment)
    except brotli.error:
        return [
            {
                "candidate_id": "pose_p1d1_reorder_unavailable",
                "candidate_kind": "pose_p1d1_reorder",
                "screen_status": "skipped_pose_segment_not_brotli_decodable",
            }
        ]
    if not source_raw.startswith(b"P1D1"):
        return [
            {
                "candidate_id": "pose_p1d1_reorder_unavailable",
                "candidate_kind": "pose_p1d1_reorder",
                "screen_status": "skipped_pose_payload_not_p1d1",
                "decoded_magic": source_raw[:4].hex(),
            }
        ]
    parsed = _parse_p1d1(source_raw)
    dims = tuple(parsed["dims"])
    if len(dims) > 8:
        raise LosslessPureRateError(f"refusing factorial pose order search for {len(dims)} dims")
    source_semantic = _p1d1_semantic_sha256(source_raw)
    rows: list[dict[str, Any]] = []
    for order in itertools.permutations(dims):
        candidate_raw = _encode_p1d1(parsed, order)
        if _p1d1_semantic_sha256(candidate_raw) != source_semantic:
            raise LosslessPureRateError(f"P1D1 reorder changed decoded semantics: {order}")
        grid = _brotli_grid(candidate_raw, qualities=qualities, lgwins=lgwins)
        best = min(grid, key=lambda row: (int(row["bytes"]), int(row["quality"]), int(row["lgwin"])))
        delta = int(best["bytes"]) - len(source_segment)
        rows.append(
            {
                "candidate_id": "pose_p1d1_order_" + "_".join(str(dim) for dim in order)
                + f"_brq{best['quality']}_lg{best['lgwin']}",
                "candidate_kind": "pose_p1d1_reorder",
                "screen_status": "screened",
                "source_pose_segment_bytes": int(len(source_segment)),
                "source_pose_segment_sha256": _sha256_bytes(source_segment),
                "source_p1d1_bytes": int(len(source_raw)),
                "source_p1d1_sha256": _sha256_bytes(source_raw),
                "source_dim_order": list(dims),
                "candidate_dim_order": list(order),
                "candidate_p1d1_bytes": int(len(candidate_raw)),
                "candidate_p1d1_sha256": _sha256_bytes(candidate_raw),
                "candidate_segment_bytes": int(best["bytes"]),
                "candidate_segment_sha256": best["sha256"],
                "candidate_brotli_quality": best["quality"],
                "candidate_brotli_lgwin": best["lgwin"],
                "candidate_segment_delta_bytes_vs_source": int(delta),
                "archive_delta_bytes_vs_source_formula": int(delta),
                "rate_score_delta_if_components_identical_formula_only": _rate_delta(delta),
                "decoded_semantics": "p1d1_decoded_pose_semantic_sha256_identical",
                "pose_semantic_sha256": source_semantic,
                "candidate_payload": best["payload"],
                "source_archive_bytes": int(source_archive["archive_bytes"]),
            }
        )
    return rows


def _strip_private(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "candidate_payload"}


def _public_archive_anatomy(paths: Mapping[str, Path]) -> dict[str, Any]:
    anatomy: dict[str, Any] = {}
    for label, path in paths.items():
        if not path.is_file():
            anatomy[label] = {"path": _repo_rel(path), "present": False}
            continue
        with zipfile.ZipFile(path, "r") as zf:
            members = []
            for info in zf.infolist():
                if info.is_dir():
                    continue
                data = zf.read(info)
                members.append(
                    {
                        "name": info.filename,
                        "file_size": int(info.file_size),
                        "compress_size": int(info.compress_size),
                        "zip_compress_type": int(info.compress_type),
                        "sha256": _sha256_bytes(data),
                        "magic_hex": data[:12].hex(),
                    }
                )
        anatomy[label] = {
            "path": _repo_rel(path),
            "present": True,
            "archive_bytes": int(path.stat().st_size),
            "archive_sha256": _sha256_file(path),
            "member_count": len(members),
            "members": members,
        }
    return anatomy


def _frontier_context(exact_eval_path: Path, source_archive: Mapping[str, Any]) -> dict[str, Any]:
    exact_eval = _read_json(exact_eval_path)
    score = exact_eval.get("score_recomputed_from_components", exact_eval.get("canonical_score"))
    return {
        "source": "current verified public PR85 T4 exact eval plus local archive custody",
        "exact_eval_json": _repo_rel(exact_eval_path),
        "archive_bytes": int(source_archive["archive_bytes"]),
        "archive_sha256": source_archive["archive_sha256"],
        "member_sha256": source_archive["member_sha256"],
        "score_recomputed_from_components": score,
        "score_rate_contribution": exact_eval.get("score_rate_contribution"),
        "score_seg_contribution": exact_eval.get("score_seg_contribution"),
        "score_pose_contribution": exact_eval.get("score_pose_contribution"),
        "final_score_display_rounded": exact_eval.get("final_score"),
        "evidence_grade": "contest_cuda_t4_exact_eval_existing_artifact"
        if exact_eval
        else "archive_custody_only_exact_eval_json_missing",
    }


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return (
        int(row.get("archive_delta_bytes_vs_source_formula", row.get("archive_delta_bytes_vs_source", 0))),
        str(row.get("candidate_id", "")),
    )


def _build_candidate_archive(
    row: Mapping[str, Any],
    *,
    source_archive: Mapping[str, Any],
    source_x: bytes,
    bundle: Pr85Bundle,
    out_dir: Path,
) -> dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    candidate_dir = out_dir / candidate_id
    candidate_kind = str(row["candidate_kind"])
    if candidate_kind == "zip_repack":
        x_candidate = source_x
        compress_type = int(row["zip_compress_type"])
        compresslevel = row["zip_compresslevel"]
    elif candidate_kind in ("segment_brotli_recode", "pose_p1d1_reorder"):
        payload = row.get("candidate_payload")
        if not isinstance(payload, (bytes, bytearray)):
            raise LosslessPureRateError(f"{candidate_id} is missing candidate payload bytes")
        segment_name = str(row.get("segment_name", "pose"))
        segments = dict(bundle.segments)
        segments[segment_name] = bytes(payload)
        x_candidate = pack_pr85_bundle(segments, header_mode=_header_mode(bundle))
        parse_pr85_bundle(x_candidate)
        compress_type = zipfile.ZIP_STORED
        compresslevel = None
    else:
        raise LosslessPureRateError(f"unsupported candidate kind: {candidate_kind}")

    archive_meta = _write_single_x_archive(
        candidate_dir / "archive.zip",
        x_candidate,
        compress_type=compress_type,
        compresslevel=compresslevel,
    )
    archive_delta = int(archive_meta["archive_bytes"]) - int(source_archive["archive_bytes"])
    if archive_delta >= 0:
        raise LosslessPureRateError(
            f"refusing to build non-winning candidate {candidate_id}: delta={archive_delta}"
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_gpu_dispatch_performed": False,
        "source_archive": source_archive,
        "screen": _strip_private(row),
        "candidate_archive": archive_meta,
        "candidate_archive_delta_bytes_vs_source": archive_delta,
        "rate_score_delta_if_components_identical_formula_only": _rate_delta(archive_delta),
        "validation": {
            "single_member_x": True,
            "bundle_parse_ok": True,
            "decoded_semantics": row.get("decoded_semantics"),
        },
        "exact_dispatch_gate": {
            "eligible_for_remote_dispatch_now": True,
            "required_before_dispatch": [
                "tools/claim_lane_dispatch.py claim --lane-id pr85_lossless_pure_rate_<candidate_id> ...",
                "rerun this builder and verify manifest source/candidate SHA fields match",
                "run experiments/preflight_pr85_fixed_runtime_readiness.py or the public PR85 replay preflight against the exact archive",
                "run experiments/contest_auth_eval.py --device cuda on the exact archive.zip",
                "adjudicate contest_auth_eval.json and recompute score from components",
            ],
        },
    }
    (candidate_dir / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def _write_markdown(summary: Mapping[str, Any], markdown_path: Path) -> None:
    source = summary["source_archive"]
    frontier = summary["frontier_context"]
    best = summary.get("best_screened_candidate")
    built = summary.get("best_built_candidate")
    lines = [
        "# PR85 lossless pure-rate local candidate lane - 2026-05-04",
        "",
        "## Scope",
        "",
        (
            "Disjoint from STBM, PR91/HPM1, and Lightning CLI hardening. "
            "Local-only screen over strict ZIP repack, non-mask Brotli recode, "
            "and P1D1 pose-order recode."
        ),
        "",
        "## Source frontier",
        "",
        f"- Archive: `{source['path']}`",
        f"- Bytes/SHA: `{source['archive_bytes']}` / `{source['archive_sha256']}`",
        f"- Exact PR85 score artifact: `{frontier['exact_eval_json']}`",
        f"- Recomputed score: `{frontier.get('score_recomputed_from_components')}`",
        "",
        "## Result",
        "",
    ]
    if built:
        lines += [
            "- Built byte-negative candidate: yes",
            f"- Best archive: `{built['archive_path']}`",
            f"- Bytes/SHA: `{built['archive_bytes']}` / `{built['archive_sha256']}`",
            f"- Pure-rate score delta: `{built['rate_score_delta_if_components_identical_formula_only']}`",
        ]
    else:
        lines += [
            "- Built byte-negative candidate: no",
            f"- Reason: `{summary.get('reason_no_candidate_built')}`",
        ]
        if best:
            lines += [
                f"- Best screened candidate: `{best.get('candidate_id')}`",
                f"- Best screened byte delta: `{best.get('archive_delta_bytes_vs_source_formula', best.get('archive_delta_bytes_vs_source'))}`",
                (
                    "- Best screened pure-rate score delta: "
                    f"`{best.get('rate_score_delta_if_components_identical_formula_only')}`"
                ),
            ]
    lines += [
        "",
        "## Dispatch gate",
        "",
        (
            "No remote/GPU dispatch from this lane unless a byte-negative manifest exists, "
            "`tools/claim_lane_dispatch.py` has an active non-conflicting claim, local "
            "manifest SHA fields still match, PR85 replay/fixed-runtime preflight passes, "
            "and exact CUDA auth eval is run through `archive.zip -> inflate.sh -> "
            "upstream/evaluate.py`."
        ),
        "",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_candidates(
    archive: Path,
    out_dir: Path,
    *,
    exact_eval_json: Path = DEFAULT_PR85_EXACT_EVAL,
    public_archives: Mapping[str, Path] = DEFAULT_PUBLIC_ARCHIVES,
    qualities: Sequence[int] = tuple(range(12)),
    lgwins: Sequence[int] = tuple(range(10, 25)),
    build_limit: int = 8,
) -> dict[str, Any]:
    source_archive, source_x, _source_info = _read_single_x_archive(archive)
    bundle = parse_pr85_bundle(source_x)
    zip_rows = _zip_repack_rows(source_archive, source_x)
    segment_rows = _segment_recode_rows(
        source_archive,
        bundle,
        qualities=qualities,
        lgwins=lgwins,
    )
    pose_rows = _pose_reorder_rows(
        source_archive,
        bundle,
        qualities=qualities,
        lgwins=lgwins,
    )
    screened_rows = zip_rows + segment_rows + pose_rows
    buildable = [
        row
        for row in screened_rows
        if int(row.get("archive_delta_bytes_vs_source_formula", row.get("archive_delta_bytes_vs_source", 0)))
        < 0
        and row.get("screen_status", "screened") == "screened"
    ]
    built: list[dict[str, Any]] = []
    for row in sorted(buildable, key=_candidate_sort_key)[:build_limit]:
        built.append(
            _build_candidate_archive(
                row,
                source_archive=source_archive,
                source_x=source_x,
                bundle=bundle,
                out_dir=out_dir,
            )
        )

    best_screened = min(
        [_strip_private(row) for row in screened_rows if "candidate_id" in row],
        key=_candidate_sort_key,
    )
    best_built = None
    if built:
        best_manifest = min(built, key=lambda row: int(row["candidate_archive_delta_bytes_vs_source"]))
        best_built = {
            "candidate_id": best_manifest["candidate_id"],
            "archive_path": best_manifest["candidate_archive"]["path"],
            "archive_bytes": best_manifest["candidate_archive"]["archive_bytes"],
            "archive_sha256": best_manifest["candidate_archive"]["archive_sha256"],
            "archive_delta_bytes_vs_source": best_manifest["candidate_archive_delta_bytes_vs_source"],
            "rate_score_delta_if_components_identical_formula_only": best_manifest[
                "rate_score_delta_if_components_identical_formula_only"
            ],
            "manifest_path": _repo_rel(out_dir / best_manifest["candidate_id"] / "manifest.json"),
        }
    reason_no_candidate = None
    if not built:
        best_delta = int(
            best_screened.get(
                "archive_delta_bytes_vs_source_formula",
                best_screened.get("archive_delta_bytes_vs_source", 0),
            )
        )
        if best_delta == 0:
            reason_no_candidate = (
                "local lossless screen found byte-neutral best case only; strict ZIP overhead is "
                "already minimal and all decoded-identical non-mask/P1D1 recodes are no smaller"
            )
        else:
            reason_no_candidate = (
                "local lossless screen found no byte-negative candidate; best screened delta "
                f"was {best_delta} bytes"
            )

    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "planning_only": not bool(built),
        "score_claim": False,
        "dispatch_performed": False,
        "remote_gpu_dispatch_performed": False,
        "source_archive": source_archive,
        "frontier_context": _frontier_context(exact_eval_json, source_archive),
        "public_anatomy": _public_archive_anatomy(public_archives),
        "bundle": {
            "format": bundle.format,
            "header_bytes": bundle.header_bytes,
            "segment_lengths": bundle.segment_lengths,
            "segment_offsets": {name: int(offset) for name, offset in bundle.segment_offsets.items()},
            "fixed_length_segments": dict(bundle.fixed_length_segments),
        },
        "screen_config": {
            "brotli_qualities": list(qualities),
            "brotli_lgwins": list(lgwins),
            "build_limit": int(build_limit),
            "candidate_build_policy": "build_only_byte_negative_decoded_identical_candidates",
        },
        "score_rate_formula": {
            "formula_only": True,
            "original_video_bytes": ORIGINAL_VIDEO_BYTES,
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "score_claim_from_this_profile": False,
        },
        "screened_candidate_count": len([row for row in screened_rows if "candidate_id" in row]),
        "built_candidate_count": len(built),
        "reason_no_candidate_built": reason_no_candidate,
        "best_screened_candidate": best_screened,
        "best_built_candidate": best_built,
        "candidate_manifests": [
            {
                "candidate_id": manifest["candidate_id"],
                "manifest_path": _repo_rel(out_dir / manifest["candidate_id"] / "manifest.json"),
                "archive_path": manifest["candidate_archive"]["path"],
                "archive_bytes": manifest["candidate_archive"]["archive_bytes"],
                "archive_sha256": manifest["candidate_archive"]["archive_sha256"],
                "archive_delta_bytes_vs_source": manifest["candidate_archive_delta_bytes_vs_source"],
                "rate_score_delta_if_components_identical_formula_only": manifest[
                    "rate_score_delta_if_components_identical_formula_only"
                ],
            }
            for manifest in built
        ],
        "top_screened_candidates": sorted(
            [_strip_private(row) for row in screened_rows if "candidate_id" in row],
            key=_candidate_sort_key,
        )[:64],
        "exact_dispatch_gate": {
            "remote_gpu_dispatch_performed": False,
            "dispatch_allowed_from_this_summary": bool(built),
            "claim_required": "tools/claim_lane_dispatch.py claim ...",
            "required_gate": [
                "active non-conflicting lane claim before any remote/GPU job",
                "byte-negative candidate manifest present",
                "source and candidate archive SHA fields match current files",
                "decoded-semantic parity proof still passes",
                "PR85 replay/fixed-runtime local preflight passes against the exact candidate archive",
                "experiments/contest_auth_eval.py --device cuda exact eval",
                "structured contest_auth_eval.json adjudication and score recomputation from components",
            ],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_summary.json").write_bytes(_json_bytes(summary))
    _write_markdown(summary, out_dir / "pr85_lossless_pure_rate_candidates.md")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--exact-eval-json", type=Path, default=DEFAULT_PR85_EXACT_EVAL)
    parser.add_argument("--qualities", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--lgwins", default="10,11,12,13,14,15,16,17,18,19,20,21,22,23,24")
    parser.add_argument("--build-limit", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_candidates(
        args.archive,
        args.out_dir,
        exact_eval_json=args.exact_eval_json,
        qualities=_parse_int_csv(args.qualities, label="qualities"),
        lgwins=_parse_int_csv(args.lgwins, label="lgwins"),
        build_limit=args.build_limit,
    )
    print(
        json.dumps(
            {
                "summary_path": _repo_rel(args.out_dir / "candidate_summary.json"),
                "built_candidate_count": summary["built_candidate_count"],
                "best_built_candidate": summary["best_built_candidate"],
                "best_screened_candidate": summary["best_screened_candidate"],
                "reason_no_candidate_built": summary["reason_no_candidate_built"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
