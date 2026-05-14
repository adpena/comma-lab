#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Local-only byte-squeeze screen around the C091/PR75 public anchor.

This worker emits deterministic candidate archives and manifests for lossless
or raw-parity-preserving packing probes.  It never dispatches remote work and
never records a score claim; exact CUDA auth eval remains required for any
candidate that changes rendered bytes or scorer-visible state.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR_ARCHIVE = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "experiments/results/lossless_byte_squeeze_worker_20260503"
)
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
TOOL = "experiments/build_lossless_byte_squeeze_worker_20260503.py"

ANCHOR_SCORE = 0.31516575028285976  # [external: PR-65 contest-CUDA T4 frontier anchor]
ANCHOR_BYTES = 276_481
ANCHOR_SHA256 = "03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746"
SUB314_TARGET = 0.314
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MEMBER_NAME = "p"
PUBLIC_PR75_MASK_LEN = 219_472
PUBLIC_PR75_RENDERER_LEN = 55_756
PUBLIC_PR75_ACTIONS_LEN = 255

CUDA_AUTH_EVAL_REQUIRED = (
    "No score claim: exact CUDA auth eval on the exact archive bytes is "
    "required via archive.zip -> inflate.sh -> upstream/evaluate.py, preferably "
    "experiments/contest_auth_eval.py --device cuda, after a dispatch claim."
)


@dataclass(frozen=True)
class BrotliChoice:
    data: bytes
    params: dict[str, int] | str


@dataclass(frozen=True)
class SourceStreams:
    mask_br: bytes
    renderer_br: bytes
    actions_br: bytes
    pose_br: bytes
    mask_raw: bytes
    renderer_raw: bytes
    actions_wire_raw: bytes
    pose_raw: bytes


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("lossless_byte_squeeze_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_member_name(name: str) -> str:
    path = Path(name)
    hidden = name.startswith(".") or name.startswith("__MACOSX/") or "/." in name
    resource_fork = name.startswith("._") or "/._" in name
    if hidden or resource_fork:
        raise ValueError(f"hidden/system ZIP member is forbidden: {name!r}")
    if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return name


def _assert_local_header_name_matches(archive: Path, info: zipfile.ZipInfo) -> None:
    with archive.open("rb") as handle:
        handle.seek(info.header_offset)
        fixed = handle.read(30)
        if len(fixed) != 30 or fixed[:4] != b"PK\x03\x04":
            raise ValueError(f"invalid ZIP local header for {info.filename!r}")
        name_len = int.from_bytes(fixed[26:28], "little")
        extra_len = int.from_bytes(fixed[28:30], "little")
        local_raw = handle.read(name_len)
        if len(local_raw) != name_len:
            raise ValueError(f"truncated ZIP local filename for {info.filename!r}")
        if len(handle.read(extra_len)) != extra_len:
            raise ValueError(f"truncated ZIP local extra for {info.filename!r}")
        local_name = local_raw.decode("utf-8" if info.flag_bits & 0x800 else "cp437")
        if local_name != info.filename:
            raise ValueError(
                f"ZIP central/local name mismatch: central={info.filename!r} local={local_name!r}"
            )


def _read_single_member_payload(path: Path) -> tuple[bytes, dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as zf:
        seen: set[str] = set()
        infos: list[zipfile.ZipInfo] = []
        for info in zf.infolist():
            name = _safe_member_name(info.filename)
            if name in seen:
                raise ValueError(f"duplicate ZIP member: {name!r}")
            seen.add(name)
            if not info.is_dir():
                infos.append(info)
        names = [info.filename for info in infos]
        if names != [MEMBER_NAME]:
            raise ValueError(f"{path} must contain exactly member {MEMBER_NAME!r}; got {names!r}")
        info = infos[0]
        _assert_local_header_name_matches(path, info)
        payload = zf.read(info)
        zip_summary = {
            "member": info.filename,
            "compress_size": int(info.compress_size),
            "compress_type": int(info.compress_type),
            "date_time": list(info.date_time),
            "external_attr": int(info.external_attr),
            "extra_len": len(info.extra),
            "file_size": int(info.file_size),
            "zip_overhead_bytes": path.stat().st_size - int(info.compress_size),
        }
        return payload, zip_summary


def _zip_info(name: str, *, compress_type: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = compress_type
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(
    path: Path,
    payload: bytes,
    *,
    compress_type: int = zipfile.ZIP_STORED,
    compresslevel: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        info = _zip_info(MEMBER_NAME, compress_type=compress_type)
        if compresslevel is None:
            zf.writestr(info, payload, compress_type=compress_type)
        else:
            zf.writestr(
                info,
                payload,
                compress_type=compress_type,
                compresslevel=compresslevel,
            )


def _focused_brotli_grid() -> list[tuple[int, int, int, int]]:
    params: list[tuple[int, int, int, int]] = []
    qualities = (11, 10, 9, 8, 6, 4, 2, 0)
    for quality in qualities:
        for mode in (0, 1, 2):
            for lgwin in (10, 12, 14, 16, 17, 18, 19, 20, 22, 24):
                for lgblock in (0, 16, 17, 18, 19, 20):
                    if lgblock and lgblock > lgwin:
                        continue
                    params.append((quality, mode, lgwin, lgblock))
    # Known winners observed around C091; keep them even if the grid changes.
    params.extend(
        [
            (11, 0, 20, 17),
            (11, 0, 19, 17),
            (11, 0, 16, 0),
            (9, 0, 16, 0),
            (4, 0, 16, 0),
            (0, 0, 20, 0),
        ]
    )
    return list(dict.fromkeys(params))


def _best_brotli(
    raw: bytes,
    *,
    source: bytes | None,
    params: Iterable[tuple[int, int, int, int]],
) -> BrotliChoice:
    best = source
    best_params: dict[str, int] | str = "source" if source is not None else {}
    for quality, mode, lgwin, lgblock in params:
        candidate = brotli.compress(
            raw,
            quality=quality,
            mode=mode,
            lgwin=lgwin,
            lgblock=lgblock,
        )
        if best is None or len(candidate) < len(best):
            best = candidate
            best_params = {
                "quality": quality,
                "mode": mode,
                "lgwin": lgwin,
                "lgblock": lgblock,
            }
    if best is None:
        raise ValueError("no Brotli candidate generated")
    if brotli.decompress(best) != raw:
        raise ValueError("selected Brotli stream failed round-trip")
    return BrotliChoice(best, best_params)


def _slice_source_payload(payload: bytes) -> SourceStreams:
    if len(payload) != (
        PUBLIC_PR75_MASK_LEN + PUBLIC_PR75_RENDERER_LEN + PUBLIC_PR75_ACTIONS_LEN + 898
    ):
        raise ValueError(f"unexpected C091 payload length: {len(payload)}")
    mask_end = PUBLIC_PR75_MASK_LEN
    renderer_end = mask_end + PUBLIC_PR75_RENDERER_LEN
    actions_end = renderer_end + PUBLIC_PR75_ACTIONS_LEN
    mask_br = payload[:mask_end]
    renderer_br = payload[mask_end:renderer_end]
    actions_br = payload[renderer_end:actions_end]
    pose_br = payload[actions_end:]
    return SourceStreams(
        mask_br=mask_br,
        renderer_br=renderer_br,
        actions_br=actions_br,
        pose_br=pose_br,
        mask_raw=brotli.decompress(mask_br),
        renderer_raw=brotli.decompress(renderer_br),
        actions_wire_raw=brotli.decompress(actions_br),
        pose_raw=brotli.decompress(pose_br),
    )


def _build_p3_payload(
    *,
    mask_br: bytes,
    renderer_br: bytes,
    actions_br: bytes,
    pose_br: bytes,
) -> bytes:
    return (
        b"P3"
        + struct.pack("<IHH", len(mask_br), len(renderer_br), len(actions_br))
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )


def _member_summary(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in header.get("members", []):
        name = str(item["name"])
        out[name] = {
            "bytes": int(item["bytes"]),
            "codec": str(item["codec"]),
            "decoded_bytes": int(item["decoded_bytes"]),
            "decoded_sha256": str(item["decoded_sha256"]),
            "sha256": str(item["sha256"]),
        }
    return out


def _decoded_summary(decoded: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
        for name, data in sorted(decoded.items())
    }


def _effective_payload_for_unpacker(payload: bytes) -> tuple[bytes, str]:
    try:
        return brotli.decompress(payload), "outer_brotli_p_member"
    except brotli.error:
        return payload, "raw_p_member"


def _validate_payload(
    *,
    payload: bytes,
    source_decoded: dict[str, bytes],
    unpacker: Any,
) -> tuple[dict[str, Any], dict[str, bytes] | None]:
    effective_payload, p_member_encoding = _effective_payload_for_unpacker(payload)
    try:
        header, decoded = unpacker._parse_payload(effective_payload)  # noqa: SLF001
    except Exception as exc:  # pragma: no cover - exact exception comes from runtime parser
        return {
            "error": str(exc),
            "p_member_encoding": p_member_encoding,
            "payload_format": None,
            "status": "failed_parser_rejected",
        }, None

    names = sorted(set(source_decoded) | set(decoded))
    changed = []
    member_changes: dict[str, Any] = {}
    for name in names:
        source_data = source_decoded.get(name)
        candidate_data = decoded.get(name)
        changed_vs_source = source_data != candidate_data
        if changed_vs_source:
            changed.append(name)
        member_changes[name] = {
            "candidate_bytes": len(candidate_data) if candidate_data is not None else None,
            "candidate_sha256": _sha256_bytes(candidate_data) if candidate_data is not None else None,
            "changed_vs_source": changed_vs_source,
            "source_bytes": len(source_data) if source_data is not None else None,
            "source_sha256": _sha256_bytes(source_data) if source_data is not None else None,
        }
    return {
        "changed_decoded_members_vs_anchor": changed,
        "decoded_members": member_changes,
        "decoded_stream_parity_vs_anchor": not changed,
        "members": _member_summary(header),
        "p_member_encoding": p_member_encoding,
        "payload_format": str(header.get("payload_format")),
        "status": "passed",
    }, decoded


def _zip_extracts_same_payload(archive: Path, expected_payload: bytes) -> bool:
    extracted, _summary = _read_single_member_payload(archive)
    return extracted == expected_payload


def _score_if_components_unchanged(archive_bytes: int) -> float:
    return ANCHOR_SCORE + (archive_bytes - ANCHOR_BYTES) * RATE_SCORE_PER_BYTE


def _needed_bytes_for_sub314(score: float) -> int:
    if score < SUB314_TARGET:
        return 0
    return math.floor((score - SUB314_TARGET) / RATE_SCORE_PER_BYTE) + 1


def _emit_candidate(
    *,
    candidate_id: str,
    payload: bytes,
    output_dir: Path,
    source_payload: bytes,
    source_decoded: dict[str, bytes],
    unpacker: Any,
    changed_payload_members: list[str],
    semantic_contract: str,
    dispatch_status: str,
    next_gate: str,
    notes: list[str],
    stream_packing: dict[str, Any],
    compress_type: int = zipfile.ZIP_STORED,
    compresslevel: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    candidate_dir = output_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    manifest_path = candidate_dir / "manifest.json"
    if archive_path.exists() and not force:
        raise FileExistsError(f"{archive_path} exists; pass --force")
    _write_archive(
        archive_path,
        payload,
        compress_type=compress_type,
        compresslevel=compresslevel,
    )
    validation, _decoded = _validate_payload(
        payload=payload,
        source_decoded=source_decoded,
        unpacker=unpacker,
    )
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256_file(archive_path)
    score_if = _score_if_components_unchanged(archive_bytes)
    extracted_same = _zip_extracts_same_payload(archive_path, payload)
    payload_identical = payload == source_payload
    manifest = {
        "archive_delta_bytes_vs_anchor": archive_bytes - ANCHOR_BYTES,
        "candidate_id": candidate_id,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "changed_payload_members": changed_payload_members,
        "dispatch_status": dispatch_status,
        "evidence_grade": "empirical_local_byte_screen",
        "formula_only": {
            "bytes_needed_for_sub314_at_candidate_components": _needed_bytes_for_sub314(score_if),
            "rate_score_delta_vs_anchor": (archive_bytes - ANCHOR_BYTES) * RATE_SCORE_PER_BYTE,
            "score_if_components_unchanged": score_if,
            "target_score": SUB314_TARGET,
        },
        "noop": bool(payload_identical and archive_sha == ANCHOR_SHA256),
        "next_gate": next_gate,
        "notes": notes,
        "output_archive": {
            "bytes": archive_bytes,
            "path": str(archive_path),
            "sha256": archive_sha,
            "zip_compress_type": compress_type,
            "zip_compresslevel": compresslevel,
        },
        "payload": {
            "bytes": len(payload),
            "extracted_payload_sha256": _sha256_bytes(payload),
            "member": MEMBER_NAME,
            "payload_identical_to_anchor": payload_identical,
        },
        "promotion_eligible": False,
        "runtime_parse_validation": validation,
        "score_claim": False,
        "semantic_contract": semantic_contract,
        "source_anchor": {
            "bytes": ANCHOR_BYTES,
            "path": str(ANCHOR_ARCHIVE),
            "score": ANCHOR_SCORE,
            "sha256": ANCHOR_SHA256,
        },
        "stream_packing": stream_packing,
        "tool": TOOL,
        "zip_extract_payload_match": extracted_same,
    }
    _write_json(manifest_path, manifest)
    return {
        "archive_bytes": archive_bytes,
        "archive_delta_bytes_vs_anchor": archive_bytes - ANCHOR_BYTES,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha,
        "candidate_id": candidate_id,
        "changed_decoded_members_vs_anchor": validation.get("changed_decoded_members_vs_anchor"),
        "changed_payload_members": changed_payload_members,
        "dispatch_status": dispatch_status,
        "formula_only_score_if_components_unchanged": score_if,
        "manifest_path": str(manifest_path),
        "next_gate": next_gate,
        "payload_bytes": len(payload),
        "payload_identical_to_anchor": payload_identical,
        "runtime_parse_status": validation["status"],
        "score_claim": False,
        "semantic_contract": semantic_contract,
    }


def _screen_zip_wrappers(payload: bytes, source_bytes: int) -> dict[str, Any]:
    best_raw_deflate = None
    for strategy_name, strategy in (
        ("DEFAULT", zlib.Z_DEFAULT_STRATEGY),
        ("FILTERED", zlib.Z_FILTERED),
        ("HUFFMAN", zlib.Z_HUFFMAN_ONLY),
        ("RLE", zlib.Z_RLE),
        ("FIXED", zlib.Z_FIXED),
    ):
        for level in range(10):
            compressor = zlib.compressobj(level, zlib.DEFLATED, -15, 9, strategy)
            compressed = compressor.compress(payload) + compressor.flush()
            row = {
                "archive_bytes_estimate": len(compressed) + 100,
                "compressed_bytes": len(compressed),
                "level": level,
                "sha256": _sha256_bytes(compressed),
                "strategy": strategy_name,
            }
            if best_raw_deflate is None or row["archive_bytes_estimate"] < best_raw_deflate["archive_bytes_estimate"]:
                best_raw_deflate = row
    return {
        "best_raw_deflate_estimate": best_raw_deflate,
        "source_archive_bytes": source_bytes,
    }


def _qzs3_reblock_payloads(
    *,
    streams: SourceStreams,
    mask_choice: BrotliChoice,
    actions_choice: BrotliChoice,
    pose_br: bytes,
    params: Iterable[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, bytes, BrotliChoice, dict[str, Any]]], dict[str, Any]]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tac.quantizr_qzs3_codec import (  # pylint: disable=import-outside-toplevel
        decode_qzs3_state_dict,
        encode_qzs3_state_dict,
    )

    state = decode_qzs3_state_dict(streams.renderer_raw, device="cpu")
    candidates: list[tuple[int, bytes, BrotliChoice, dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []
    for block_size in (48, 64, 96, 128):
        renderer_raw = encode_qzs3_state_dict(state, block_size=block_size)
        renderer_choice = _best_brotli(renderer_raw, source=None, params=params)
        payload = _build_p3_payload(
            mask_br=mask_choice.data,
            renderer_br=renderer_choice.data,
            actions_br=actions_choice.data,
            pose_br=pose_br,
        )
        summary = {
            "block_size": block_size,
            "renderer_brotli_bytes": len(renderer_choice.data),
            "renderer_brotli_delta_vs_anchor": len(renderer_choice.data) - len(streams.renderer_br),
            "renderer_brotli_sha256": _sha256_bytes(renderer_choice.data),
            "renderer_raw_bytes": len(renderer_raw),
            "renderer_raw_delta_vs_anchor": len(renderer_raw) - len(streams.renderer_raw),
            "renderer_raw_sha256": _sha256_bytes(renderer_raw),
            "renderer_raw_identical_to_anchor": renderer_raw == streams.renderer_raw,
            "renderer_brotli_params": renderer_choice.params,
        }
        candidates.append((block_size, payload, renderer_choice, summary))
        summaries.append(summary)
    return candidates, {"qzs3_reblock_summaries": summaries}


def build_candidates(
    *,
    anchor_archive: Path,
    output_dir: Path,
    include_qzs3_reblock: bool,
    force: bool,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_payload, source_zip_summary = _read_single_member_payload(anchor_archive)
    source_sha = _sha256_file(anchor_archive)
    if anchor_archive.stat().st_size != ANCHOR_BYTES or source_sha != ANCHOR_SHA256:
        raise ValueError(
            "anchor archive custody mismatch: "
            f"bytes={anchor_archive.stat().st_size} sha={source_sha}"
        )
    unpacker = _load_unpacker()
    source_header, source_decoded = unpacker._parse_payload(source_payload)  # noqa: SLF001
    streams = _slice_source_payload(source_payload)
    params = _focused_brotli_grid()

    mask_choice = _best_brotli(streams.mask_raw, source=streams.mask_br, params=params)
    renderer_choice = _best_brotli(streams.renderer_raw, source=streams.renderer_br, params=params)
    actions_wire_choice = _best_brotli(
        streams.actions_wire_raw,
        source=streams.actions_br,
        params=params,
    )
    pose_choice = _best_brotli(streams.pose_raw, source=streams.pose_br, params=params)
    outer_choice = _best_brotli(source_payload, source=None, params=params)
    zip_wrapper_screen = _screen_zip_wrappers(source_payload, ANCHOR_BYTES)

    stream_screen = {
        "actions_wire": {
            "best_bytes": len(actions_wire_choice.data),
            "best_delta": len(actions_wire_choice.data) - len(streams.actions_br),
            "best_params": actions_wire_choice.params,
            "best_sha256": _sha256_bytes(actions_wire_choice.data),
            "source_bytes": len(streams.actions_br),
            "source_sha256": _sha256_bytes(streams.actions_br),
            "wire_raw_bytes": len(streams.actions_wire_raw),
            "wire_raw_sha256": _sha256_bytes(streams.actions_wire_raw),
        },
        "masks": {
            "best_bytes": len(mask_choice.data),
            "best_delta": len(mask_choice.data) - len(streams.mask_br),
            "best_params": mask_choice.params,
            "best_sha256": _sha256_bytes(mask_choice.data),
            "source_bytes": len(streams.mask_br),
            "source_sha256": _sha256_bytes(streams.mask_br),
        },
        "optimized_poses_qp1": {
            "best_bytes": len(pose_choice.data),
            "best_delta": len(pose_choice.data) - len(streams.pose_br),
            "best_params": pose_choice.params,
            "best_sha256": _sha256_bytes(pose_choice.data),
            "source_bytes": len(streams.pose_br),
            "source_sha256": _sha256_bytes(streams.pose_br),
        },
        "renderer": {
            "best_bytes": len(renderer_choice.data),
            "best_delta": len(renderer_choice.data) - len(streams.renderer_br),
            "best_params": renderer_choice.params,
            "best_sha256": _sha256_bytes(renderer_choice.data),
            "source_bytes": len(streams.renderer_br),
            "source_sha256": _sha256_bytes(streams.renderer_br),
        },
    }

    rows: list[dict[str, Any]] = []
    rows.append(
        _emit_candidate(
            candidate_id="anchor_stored_zip_rewrite_payload_identical",
            payload=source_payload,
            output_dir=output_dir,
            source_payload=source_payload,
            source_decoded=source_decoded,
            unpacker=unpacker,
            changed_payload_members=[],
            semantic_contract="zip_metadata_rewrite_only_payload_identical",
            dispatch_status="do_not_dispatch_no_byte_win_payload_identical",
            next_gate="none; no-op byte screen control",
            notes=[
                "Rewrites the exact p payload with deterministic stored ZIP metadata.",
                "Extracted p bytes are byte-identical to the anchor, so local raw-output parity is inherited.",
            ],
            stream_packing={"source_zip_summary": source_zip_summary},
            force=force,
        )
    )
    rows.append(
        _emit_candidate(
            candidate_id="anchor_zip_deflate_payload_identical_negative",
            payload=source_payload,
            output_dir=output_dir,
            source_payload=source_payload,
            source_decoded=source_decoded,
            unpacker=unpacker,
            changed_payload_members=[],
            semantic_contract="zip_deflate_payload_identical_negative_probe",
            dispatch_status="do_not_dispatch_byte_regression",
            next_gate="none; ZIP deflate is larger on this payload",
            notes=[
                "ZIP_DEFLATED extraction returns the exact anchor p payload.",
                "Outer ZIP deflate is a byte regression for the already-Brotli fixed-slice payload.",
            ],
            stream_packing={"zip_wrapper_screen": zip_wrapper_screen},
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            force=force,
        )
    )
    rows.append(
        _emit_candidate(
            candidate_id="anchor_outer_brotli_fixed_payload_negative",
            payload=outer_choice.data,
            output_dir=output_dir,
            source_payload=source_payload,
            source_decoded=source_decoded,
            unpacker=unpacker,
            changed_payload_members=["p_outer_brotli_wrapper"],
            semantic_contract="outer_brotli_wrapper_decodes_to_anchor_fixed_slices",
            dispatch_status="do_not_dispatch_byte_regression",
            next_gate="none; outer Brotli wrapper is larger than raw fixed slices",
            notes=[
                "Current unpacker recognizes a Brotli-compressed p member and then parses the decompressed fixed-slice payload.",
                "Decoded logical members are byte-identical to the anchor, but the wrapper costs bytes.",
            ],
            stream_packing={
                "outer_brotli": {
                    "bytes": len(outer_choice.data),
                    "delta_vs_anchor_payload": len(outer_choice.data) - len(source_payload),
                    "params": outer_choice.params,
                    "sha256": _sha256_bytes(outer_choice.data),
                }
            },
            force=force,
        )
    )
    p3_payload = _build_p3_payload(
        mask_br=mask_choice.data,
        renderer_br=renderer_choice.data,
        actions_br=actions_wire_choice.data,
        pose_br=pose_choice.data,
    )
    rows.append(
        _emit_candidate(
            candidate_id="anchor_p3_best_lossless_streams_negative",
            payload=p3_payload,
            output_dir=output_dir,
            source_payload=source_payload,
            source_decoded=source_decoded,
            unpacker=unpacker,
            changed_payload_members=["P3_header", "masks.mkv.br", "seg_tile_actions.br"],
            semantic_contract="self_describing_p3_decoded_stream_parity_vs_anchor",
            dispatch_status="do_not_dispatch_byte_regression",
            next_gate="none; P3 header dominates the 8 bytes of stream savings",
            notes=[
                "Mask and SG2 action wire streams re-Brotli smaller, renderer and pose stay source.",
                "The self-describing P3 header makes the archive larger than the fixed-slice anchor.",
            ],
            stream_packing={"stream_screen": stream_screen},
            force=force,
        )
    )
    invalid_raw_payload = (
        mask_choice.data
        + renderer_choice.data
        + actions_wire_choice.data
        + pose_choice.data
    )
    rows.append(
        _emit_candidate(
            candidate_id="anchor_raw_fixedslice_short_streams_parser_rejected",
            payload=invalid_raw_payload,
            output_dir=output_dir,
            source_payload=source_payload,
            source_decoded=source_decoded,
            unpacker=unpacker,
            changed_payload_members=["masks.mkv.br", "seg_tile_actions.br"],
            semantic_contract="invalid_headerless_short_fixedslice_probe",
            dispatch_status="not_contest_faithful_parser_rejected_do_not_dispatch",
            next_gate="requires reviewed runtime support and parser tests before any exact eval; not recommended for this 8-byte ceiling",
            notes=[
                "This removes the P3 header and keeps the shorter streams, but current robust_current fixed-slice parser assumes a 219472-byte mask segment.",
                "The parser rejection is the expected fail-closed result.",
            ],
            stream_packing={"stream_screen": stream_screen},
            force=force,
        )
    )

    qzs3_summary: dict[str, Any] = {"included": False}
    if include_qzs3_reblock:
        qzs3_candidates, qzs3_summary = _qzs3_reblock_payloads(
            streams=streams,
            mask_choice=mask_choice,
            actions_choice=actions_wire_choice,
            pose_br=pose_choice.data,
            params=params,
        )
        qzs3_summary["included"] = True
        for block_size, payload, renderer_reblock_choice, reblock_summary in qzs3_candidates:
            rows.append(
                _emit_candidate(
                    candidate_id=f"anchor_p3_qzs3_block{block_size}_nonparity_probe",
                    payload=payload,
                    output_dir=output_dir,
                    source_payload=source_payload,
                    source_decoded=source_decoded,
                    unpacker=unpacker,
                    changed_payload_members=[
                        "P3_header",
                        "masks.mkv.br",
                        "renderer.bin.br",
                        "seg_tile_actions.br",
                    ],
                    semantic_contract="contest_parseable_qzs3_reblock_probe_not_raw_parity",
                    dispatch_status="not_raw_parity_exact_cuda_required_before_any_score_claim",
                    next_gate=(
                        "claim-ready only after operator approval: first run any desired local CPU "
                        "render sanity/parity screens, then claim lane and exact CUDA auth eval"
                    ),
                    notes=[
                        "This is not lossless: QZS3 block-size re-encoding changes decoded renderer.bin bytes.",
                        "It is included because it is the only local parser-clean byte probe here with enough rate headroom to cross <0.314 if components did not move.",
                    ],
                    stream_packing={
                        "qzs3_reblock": reblock_summary,
                        "renderer_reblock_brotli": {
                            "bytes": len(renderer_reblock_choice.data),
                            "params": renderer_reblock_choice.params,
                            "sha256": _sha256_bytes(renderer_reblock_choice.data),
                        },
                        "stream_screen": stream_screen,
                    },
                    force=force,
                )
            )

    rows = sorted(rows, key=lambda row: (row["archive_bytes"], row["candidate_id"]))
    summary = {
        "anchor": {
            "archive_bytes": ANCHOR_BYTES,
            "archive_sha256": ANCHOR_SHA256,
            "payload_bytes": len(source_payload),
            "payload_format": str(source_header.get("payload_format")),
            "payload_sha256": _sha256_bytes(source_payload),
            "score": ANCHOR_SCORE,
            "sub314_gap_bytes_at_unchanged_components": _needed_bytes_for_sub314(ANCHOR_SCORE),
        },
        "candidate_count": len(rows),
        "candidates": rows,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "evidence_grade": "empirical_local_byte_screen",
        "include_qzs3_reblock": include_qzs3_reblock,
        "promotion_eligible": False,
        "qzs3_reblock_screen": qzs3_summary,
        "schema": "lossless_byte_squeeze_worker_20260503_v1",
        "score_claim": False,
        "source_decoded_members": _decoded_summary(source_decoded),
        "stream_screen": stream_screen,
        "tool": TOOL,
        "zip_wrapper_screen": zip_wrapper_screen,
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-archive", type=Path, default=ANCHOR_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-qzs3-reblock",
        action="store_true",
        help="Skip non-raw-parity QZS3 block-size probes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_candidates(
        anchor_archive=args.anchor_archive.resolve(),
        output_dir=args.output_dir,
        include_qzs3_reblock=not args.skip_qzs3_reblock,
        force=bool(args.force),
    )
    top_raw_parity = next(
        (
            row
            for row in summary["candidates"]
            if row["runtime_parse_status"] == "passed"
            and not row.get("changed_decoded_members_vs_anchor")
        ),
        None,
    )
    top_rate_probe = summary["candidates"][0] if summary["candidates"] else None
    print(
        json.dumps(
            {
                "candidate_count": summary["candidate_count"],
                "output_dir": str(Path(args.output_dir).resolve()),
                "score_claim": False,
                "top_by_bytes": top_rate_probe,
                "top_raw_parity_by_bytes": top_raw_parity,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
