#!/usr/bin/env python3
"""Build deterministic PR75/minp and C089 lossless micro-packer candidates.

This is a local byte-screening tool.  It emits deterministic single-member
stored-ZIP archives, validates every dispatchable payload with the contest
runtime unpacker, and marks speculative wire probes as non-dispatchable until
their stated safety gate passes.  No output is score evidence until exact CUDA
auth eval runs on the exact archive bytes.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
DEFAULT_C089_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip"
)
DEFAULT_PUBLIC_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr75_lossless_micro_packer_worker_20260503"
TOOL = "experiments/build_pr75_minp_lossless_micro_candidates.py"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MEMBER_NAME = "p"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
BASELINE_SCORE = 0.3154707273953505  # [external: PR-75 contest-CUDA T4 anchor (== PR-65 frontier)]
BASELINE_BYTES = 276_342
BASELINE_SHA256 = "0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8"
PUBLIC_MINP_SHA256 = "03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746"
PUBLIC_PR79_MINP_V2_SHA256 = "01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446"
CUDA_AUTH_EVAL_REQUIRED = (
    "claim lane via tools/claim_lane_dispatch.py, then exact CUDA auth eval: "
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
RENDERER_TRANSPLANT_GATE = (
    "run experiments/preflight_renderer_transplant_pose_safety.py against the "
    "exact C089 source archive SHA and candidate archive SHA; only dispatch if "
    "safe_for_exact_eval_dispatch=true, then claim lane and run exact CUDA auth eval"
)
LOCAL_PARITY_GATE = (
    "run local robust_current unpack/runtime raw-output parity against the named "
    "source archive; only then claim lane and run exact CUDA auth eval"
)
NON_DISPATCHABLE_GATE = "non-dispatchable byte probe; requires runtime support and tests before any exact eval"


@dataclass(frozen=True)
class BrotliChoice:
    data: bytes
    params: dict[str, int] | str


@dataclass(frozen=True)
class EncodedStreams:
    mask_br: bytes
    renderer_br: bytes
    actions_br: bytes
    pose_br: bytes
    action_record_count: int | None = None
    action_dict_br: bytes | None = None


@dataclass(frozen=True)
class SourceArchive:
    label: str
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    payload_format: str
    encoded: EncodedStreams
    decoded: dict[str, bytes]
    runtime_members: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    payload: bytes
    expected_decoded: dict[str, bytes] | None
    semantic_contract: str
    selected_sources: dict[str, str]
    next_dispatch_safety_gate: str
    dispatchable_after_gate: bool
    source_preserving_vs_c089: bool
    notes: list[str]
    stream_packing: dict[str, Any]


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
    spec = importlib.util.spec_from_file_location("pr75_lossless_micro_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_archive_member_name(name: str) -> str:
    path = Path(name)
    hidden = name.startswith(".") or name.startswith("__MACOSX/") or "/." in name
    resource_fork = name.startswith("._") or "/._" in name
    if hidden or resource_fork:
        raise ValueError(f"hidden/system archive member is forbidden: {name!r}")
    if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return name


def _read_single_payload_member(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        seen: set[str] = set()
        names: list[str] = []
        for info in zf.infolist():
            name = _safe_archive_member_name(info.filename)
            if name in seen:
                raise ValueError(f"duplicate ZIP member: {name}")
            seen.add(name)
            if not info.is_dir():
                names.append(name)
        if names != [MEMBER_NAME]:
            raise ValueError(f"{path} must contain exactly member {MEMBER_NAME!r}; got {names!r}")
        return zf.read(MEMBER_NAME)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(MEMBER_NAME), payload)


def _parse_encoded_streams(payload: bytes) -> EncodedStreams:
    action_record_count: int | None = None
    action_dict_len = 0
    if payload.startswith(b"P3"):
        cursor = 2 + struct.calcsize("<IHH")
        mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
    elif payload.startswith(b"P5"):
        cursor = 2 + struct.calcsize("<IHHHH")
        mask_len, renderer_len, action_dict_len, actions_len, action_record_count = struct.unpack_from(
            "<IHHHH",
            payload,
            2,
        )
    elif payload.startswith(b"P6"):
        cursor = 2 + struct.calcsize("<IHHH")
        mask_len, renderer_len, actions_len, action_record_count = struct.unpack_from(
            "<IHHH",
            payload,
            2,
        )
    elif len(payload) == 276_381:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 55_756, 255
    elif len(payload) == 276_379:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 55_756, 253
    elif len(payload) == 276_520:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 55_914, 236
    elif len(payload) == 276_641:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 56_034, 236
    elif len(payload) == 277_288:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 55_756, 1_162
    else:
        raise ValueError(
            f"unsupported PR75/minp payload form: prefix={payload[:4]!r} len={len(payload)}"
        )
    required = [mask_len, renderer_len, actions_len]
    if action_dict_len:
        required.append(action_dict_len)
    if action_record_count is not None:
        required.append(action_record_count)
    if min(required) <= 0:
        raise ValueError("empty encoded stream in source payload")
    mask_end = cursor + mask_len
    renderer_end = mask_end + renderer_len
    dict_end = renderer_end + action_dict_len
    actions_end = dict_end + actions_len
    if actions_end >= len(payload):
        raise ValueError("encoded stream lengths leave no pose payload")
    return EncodedStreams(
        mask_br=payload[cursor:mask_end],
        renderer_br=payload[mask_end:renderer_end],
        action_dict_br=payload[renderer_end:dict_end] if action_dict_len else None,
        actions_br=payload[dict_end:actions_end],
        pose_br=payload[actions_end:],
        action_record_count=action_record_count,
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


def _load_source(label: str, path: Path, unpacker: Any) -> SourceArchive:
    path = path.resolve()
    payload = _read_single_payload_member(path)
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    encoded = _parse_encoded_streams(payload)
    return SourceArchive(
        label=label,
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=str(header.get("payload_format")),
        encoded=encoded,
        decoded=decoded,
        runtime_members=_member_summary(header),
    )


def default_brotli_param_grid() -> list[tuple[int, int, int, int]]:
    """Return a focused deterministic grid containing the observed PR75 winners."""
    params: list[tuple[int, int, int, int]] = []
    for quality in (11, 10, 9, 8, 6, 4, 2, 0):
        for mode in (0, 1, 2):
            for lgwin in (10, 12, 14, 16, 17, 18, 19, 20, 22, 24):
                for lgblock in (0, 16, 17, 18, 19, 20):
                    if lgblock and lgblock > lgwin:
                        continue
                    params.append((quality, mode, lgwin, lgblock))
    return params


def exhaustive_brotli_param_grid() -> list[tuple[int, int, int, int]]:
    params: list[tuple[int, int, int, int]] = []
    for quality in range(11, -1, -1):
        for mode in (0, 1, 2):
            for lgwin in range(10, 25):
                for lgblock in (0, 16, 17, 18, 19, 20, 21, 22, 23, 24):
                    if lgblock and lgblock > lgwin:
                        continue
                    params.append((quality, mode, lgwin, lgblock))
    return params


def _compress(raw: bytes, params: tuple[int, int, int, int]) -> bytes:
    quality, mode, lgwin, lgblock = params
    return brotli.compress(raw, quality=quality, mode=mode, lgwin=lgwin, lgblock=lgblock)


def _best_brotli(
    raw: bytes,
    *,
    source: bytes | None,
    params: Iterable[tuple[int, int, int, int]],
) -> BrotliChoice:
    best = source
    best_params: dict[str, int] | str = "source" if source is not None else {}
    for param in params:
        candidate = _compress(raw, param)
        if best is None or len(candidate) < len(best):
            quality, mode, lgwin, lgblock = param
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


def _read_action_records(raw_actions: bytes) -> list[tuple[int, int, int]]:
    if len(raw_actions) % 4:
        raise ValueError(f"PR75 action records must be 4 bytes each, got {len(raw_actions)}")
    records: list[tuple[int, int, int]] = []
    for offset in range(0, len(raw_actions), 4):
        pair_index = int.from_bytes(raw_actions[offset : offset + 2], "little")
        records.append((pair_index, raw_actions[offset + 2], raw_actions[offset + 3]))
    return records


def _records_to_raw(records: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for pair_index, tile_id, action_id in records:
        out += int(pair_index).to_bytes(2, "little") + bytes([int(tile_id), int(action_id)])
    return bytes(out)


def _uleb128(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"cannot varint-encode negative value {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def encode_delta_varint_actions(raw_actions: bytes) -> bytes:
    """Encode runtime action records as P6 pair-delta varints."""
    records = _read_action_records(raw_actions)
    out = bytearray()
    previous_pair = 0
    for index, (pair_index, tile_id, action_id) in enumerate(records):
        delta = pair_index if index == 0 else pair_index - previous_pair
        if delta < 0:
            raise ValueError("P6 delta-varint action encoding requires nondecreasing pairs")
        out.extend(_uleb128(delta))
        out.append(tile_id)
        out.append(action_id)
        previous_pair = pair_index
    return bytes(out)


def sorted_action_records_for_p6(raw_actions: bytes) -> tuple[bytes, dict[str, Any]]:
    records = _read_action_records(raw_actions)
    duplicate_pair_tiles = [
        {"pair": pair, "tile": tile}
        for pair, tile in sorted(
            {
                (pair, tile)
                for pair, tile, _action in records
                if sum(1 for p2, t2, _a2 in records if p2 == pair and t2 == tile) > 1
            }
        )
    ]
    indexed = [(pair, tile, action, index) for index, (pair, tile, action) in enumerate(records)]
    sorted_records = [
        (pair, tile, action)
        for pair, tile, action, _index in sorted(indexed, key=lambda item: (item[0], item[3]))
    ]
    return _records_to_raw(sorted_records), {
        "duplicate_pair_tile_count": len(duplicate_pair_tiles),
        "duplicate_pair_tiles": duplicate_pair_tiles[:20],
        "original_record_sha256": _sha256_bytes(raw_actions),
        "record_count": len(records),
        "sorted_record_sha256": _sha256_bytes(_records_to_raw(sorted_records)),
    }


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _default_action_specs() -> list[tuple[float, float, float]]:
    directions = [
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (-0.35, 0.15, 0.45),
        (0.25, 0.15, -0.20),
    ]
    specs: list[tuple[float, float, float]] = []
    for vec in directions:
        scale = _f32(max(abs(component) for component in vec) or 1.0)
        unit = tuple(_f32(_f32(component) / scale) for component in vec)
        for amp in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0):
            specs.append(tuple(_f32(_f32(component) * _f32(amp)) for component in unit))
            specs.append(tuple(_f32(-_f32(component) * _f32(amp)) for component in unit))
    return specs


def encode_p5_action_dict_probe(raw_actions: bytes) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    records = _read_action_records(raw_actions)
    unique_actions = sorted({action_id for _pair, _tile, action_id in records})
    if len(unique_actions) > 64:
        raise ValueError(f"P5 compact action dictionary only supports 64 actions, got {len(unique_actions)}")
    remap = {action_id: index for index, action_id in enumerate(unique_actions)}
    specs = _default_action_specs()
    dict_body = bytearray()
    for action_id in unique_actions:
        dict_body.extend(struct.pack("<fff", *specs[action_id]))
    action_dict_raw = b"TAD1" + struct.pack("<HH", 1, len(unique_actions)) + bytes(dict_body)
    packed = bytearray()
    remapped_records: list[tuple[int, int, int]] = []
    for pair_index, tile_id, action_id in records:
        remapped_action = remap[action_id]
        if pair_index >= 1024:
            raise ValueError(f"P5 pair index exceeds 10-bit field: {pair_index}")
        word = int(pair_index) | (int(tile_id) << 10) | (int(remapped_action) << 18)
        packed += bytes([word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF])
        remapped_records.append((pair_index, tile_id, remapped_action))
    remapped_actions_raw = _records_to_raw(remapped_records)
    return bytes(packed), action_dict_raw, remapped_actions_raw, {
        "original_action_ids": unique_actions,
        "original_record_sha256": _sha256_bytes(raw_actions),
        "record_count": len(records),
        "remapped_record_sha256": _sha256_bytes(remapped_actions_raw),
        "semantic_equivalence_basis": "custom TAD1 dictionary stores default runtime action vectors for used original action ids",
        "unique_action_count": len(unique_actions),
    }


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


def _build_p5_payload(
    *,
    mask_br: bytes,
    renderer_br: bytes,
    action_dict_br: bytes,
    actions_br: bytes,
    record_count: int,
    pose_br: bytes,
) -> bytes:
    return (
        b"P5"
        + struct.pack(
            "<IHHHH",
            len(mask_br),
            len(renderer_br),
            len(action_dict_br),
            len(actions_br),
            record_count,
        )
        + mask_br
        + renderer_br
        + action_dict_br
        + actions_br
        + pose_br
    )


def _build_p6_payload(
    *,
    mask_br: bytes,
    renderer_br: bytes,
    actions_br: bytes,
    record_count: int,
    pose_br: bytes,
) -> bytes:
    return (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(actions_br), record_count)
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )


def _build_raw_concat_payload(streams: EncodedStreams) -> bytes:
    pieces = [streams.mask_br, streams.renderer_br]
    if streams.action_dict_br is not None:
        pieces.append(streams.action_dict_br)
    pieces.extend([streams.actions_br, streams.pose_br])
    return b"".join(pieces)


def _validate_candidate_payload(
    *,
    payload: bytes,
    expected_decoded: dict[str, bytes] | None,
    unpacker: Any,
) -> tuple[dict[str, Any], dict[str, bytes] | None]:
    try:
        header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    except Exception as exc:  # pragma: no cover - exact type comes from runtime parser
        return {
            "error": str(exc),
            "payload_format": None,
            "status": "failed_parser_rejected",
        }, None
    runtime_members = _member_summary(header)
    validation: dict[str, Any] = {
        "members": runtime_members,
        "payload_format": str(header.get("payload_format")),
        "status": "parsed",
    }
    if expected_decoded is None:
        validation["decoded_parity_status"] = "not_checked_probe"
        return validation, decoded
    missing = sorted(set(expected_decoded) - set(decoded))
    extra = sorted(set(decoded) - set(expected_decoded))
    mismatches = []
    for name, expected in expected_decoded.items():
        actual = decoded.get(name)
        if actual is not None and actual != expected:
            mismatches.append(
                {
                    "actual_sha256": _sha256_bytes(actual),
                    "expected_sha256": _sha256_bytes(expected),
                    "name": name,
                }
            )
    if missing or extra or mismatches:
        validation.update(
            {
                "decoded_parity_status": "failed",
                "extra": extra,
                "missing": missing,
                "mismatches": mismatches,
                "status": "failed_decoded_parity",
            }
        )
    else:
        validation.update(
            {
                "decoded_parity_status": "passed",
                "members_compared": sorted(expected_decoded),
                "status": "passed",
            }
        )
    return validation, decoded


def _stream_choices_summary(choices: dict[str, BrotliChoice]) -> dict[str, Any]:
    return {
        name: {
            "bytes": len(choice.data),
            "params": choice.params,
            "sha256": _sha256_bytes(choice.data),
        }
        for name, choice in sorted(choices.items())
    }


def _source_summary(source: SourceArchive) -> dict[str, Any]:
    encoded: dict[str, Any] = {
        "actions": {
            "bytes": len(source.encoded.actions_br),
            "record_count": source.encoded.action_record_count,
            "sha256": _sha256_bytes(source.encoded.actions_br),
        },
        "mask": {
            "bytes": len(source.encoded.mask_br),
            "sha256": _sha256_bytes(source.encoded.mask_br),
        },
        "pose": {
            "bytes": len(source.encoded.pose_br),
            "sha256": _sha256_bytes(source.encoded.pose_br),
        },
        "renderer": {
            "bytes": len(source.encoded.renderer_br),
            "sha256": _sha256_bytes(source.encoded.renderer_br),
        },
    }
    if source.encoded.action_dict_br is not None:
        encoded["action_dict"] = {
            "bytes": len(source.encoded.action_dict_br),
            "sha256": _sha256_bytes(source.encoded.action_dict_br),
        }
    return {
        "archive_bytes": source.archive_bytes,
        "archive_sha256": source.archive_sha256,
        "decoded_members": _decoded_summary(source.decoded),
        "encoded_streams": encoded,
        "path": str(source.path),
        "payload_bytes": len(source.payload),
        "payload_format": source.payload_format,
        "payload_sha256": source.payload_sha256,
    }


def _decoded_change_summary(
    *,
    c089: SourceArchive,
    decoded: dict[str, bytes] | None,
) -> dict[str, Any]:
    if decoded is None:
        return {"status": "not_available_parser_rejected"}
    names = sorted(set(c089.decoded) | set(decoded))
    changes: dict[str, Any] = {}
    changed: list[str] = []
    for name in names:
        c089_data = c089.decoded.get(name)
        candidate_data = decoded.get(name)
        changed_vs_c089 = c089_data != candidate_data
        if changed_vs_c089:
            changed.append(name)
        changes[name] = {
            "candidate_bytes": len(candidate_data) if candidate_data is not None else None,
            "candidate_sha256": _sha256_bytes(candidate_data) if candidate_data is not None else None,
            "c089_bytes": len(c089_data) if c089_data is not None else None,
            "c089_sha256": _sha256_bytes(c089_data) if c089_data is not None else None,
            "changed_vs_c089": changed_vs_c089,
        }
    return {
        "changed_decoded_streams_vs_c089": changed,
        "decoded_streams": changes,
        "status": "decoded_stream_changed_vs_c089" if changed else "decoded_stream_byte_identical_vs_c089",
    }


def _candidate_manifest(
    *,
    spec: CandidateSpec,
    archive_path: Path,
    c089: SourceArchive,
    public: SourceArchive,
    validation: dict[str, Any],
    decoded: dict[str, bytes] | None,
) -> dict[str, Any]:
    archive_bytes = archive_path.stat().st_size
    archive_sha256 = _sha256_file(archive_path)
    delta_bytes = archive_bytes - c089.archive_bytes
    payload_sha = _sha256_bytes(spec.payload)
    decoded_changes = _decoded_change_summary(c089=c089, decoded=decoded)
    payload_identical = spec.payload == c089.payload
    archive_identical = archive_sha256 == c089.archive_sha256 and archive_bytes == c089.archive_bytes
    noop = payload_identical and archive_identical
    parse_passed = validation.get("status") == "passed"
    dispatchable = bool(spec.dispatchable_after_gate and parse_passed and not noop)
    manifest: dict[str, Any] = {
        "archive_delta_bytes_vs_c089": delta_bytes,
        "baseline_score_if_components_unchanged": BASELINE_SCORE
        + (archive_bytes - BASELINE_BYTES) * RATE_SCORE_PER_BYTE,
        "candidate_id": spec.candidate_id,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "decoded_change_summary_vs_c089": decoded_changes,
        "dispatch_safety": {
            "dispatchable_after_gate": dispatchable,
            "next_dispatch_safety_gate": spec.next_dispatch_safety_gate,
            "reason_not_dispatchable_now": None
            if dispatchable
            else (
                "byte-identical no-op"
                if noop
                else "parser/parity gate failed or candidate is an explicit non-dispatchable probe"
            ),
        },
        "evidence_grade": "empirical_byte_screen_only",
        "formula_only_rate_score_delta_vs_c089": delta_bytes * RATE_SCORE_PER_BYTE,
        "noop": noop,
        "noop_status": (
            "byte_identical_to_c089_archive"
            if noop
            else (
                "payload_identical_zip_metadata_changed"
                if payload_identical
                else "non_noop_payload"
            )
        ),
        "notes": spec.notes,
        "output_archive": {
            "bytes": archive_bytes,
            "path": str(archive_path),
            "sha256": archive_sha256,
        },
        "payload": {
            "bytes": len(spec.payload),
            "member": MEMBER_NAME,
            "sha256": payload_sha,
        },
        "promotion_eligible": False,
        "runtime_parse_validation": validation,
        "schema": "pr75_lossless_micro_candidate_manifest_v1",
        "score_claim": False,
        "selected_sources": spec.selected_sources,
        "semantic_contract": spec.semantic_contract,
        "source_archives": {
            "c089": _source_summary(c089),
            "public_pr75_minp": _source_summary(public),
        },
        "source_preservation": {
            "archive_byte_identical_to_c089": archive_identical,
            "payload_byte_identical_to_c089": payload_identical,
            "source_preserving_vs_c089": spec.source_preserving_vs_c089,
        },
        "stream_packing": spec.stream_packing,
        "tool": TOOL,
    }
    if not math.isfinite(float(manifest["formula_only_rate_score_delta_vs_c089"])):
        raise ValueError("non-finite formula-only rate score delta")
    return manifest


def _emit_candidate(
    *,
    spec: CandidateSpec,
    output_dir: Path,
    c089: SourceArchive,
    public: SourceArchive,
    unpacker: Any,
    force: bool,
) -> dict[str, Any]:
    candidate_dir = output_dir / spec.candidate_id
    archive_path = candidate_dir / "archive.zip"
    manifest_path = candidate_dir / "manifest.json"
    if archive_path.exists() and not force:
        raise FileExistsError(f"{archive_path} exists; pass --force")
    _write_archive(archive_path, spec.payload)
    validation, decoded = _validate_candidate_payload(
        payload=spec.payload,
        expected_decoded=spec.expected_decoded,
        unpacker=unpacker,
    )
    manifest = _candidate_manifest(
        spec=spec,
        archive_path=archive_path,
        c089=c089,
        public=public,
        validation=validation,
        decoded=decoded,
    )
    _write_json(manifest_path, manifest)
    return {
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_path": manifest["output_archive"]["path"],
        "archive_sha256": manifest["output_archive"]["sha256"],
        "candidate_id": spec.candidate_id,
        "delta_bytes_vs_c089": manifest["archive_delta_bytes_vs_c089"],
        "dispatchable_after_gate": manifest["dispatch_safety"]["dispatchable_after_gate"],
        "formula_only_rate_score_delta_vs_c089": manifest[
            "formula_only_rate_score_delta_vs_c089"
        ],
        "manifest_path": str(manifest_path),
        "next_dispatch_safety_gate": spec.next_dispatch_safety_gate,
        "noop": manifest["noop"],
        "noop_status": manifest["noop_status"],
        "payload_bytes": manifest["payload"]["bytes"],
        "parse_status": manifest["runtime_parse_validation"]["status"],
        "score_claim": False,
        "semantic_contract": spec.semantic_contract,
        "source_preserving_vs_c089": spec.source_preserving_vs_c089,
    }


def _choice(raw: bytes, source: bytes | None, params: Iterable[tuple[int, int, int, int]]) -> BrotliChoice:
    return _best_brotli(raw, source=source, params=params)


def _build_candidate_specs(
    *,
    c089: SourceArchive,
    public: SourceArchive,
    params: Iterable[tuple[int, int, int, int]],
) -> list[CandidateSpec]:
    params = list(params)
    c089_mask = _choice(c089.decoded["masks.mkv"], c089.encoded.mask_br, params)
    c089_renderer = _choice(c089.decoded["renderer.bin"], c089.encoded.renderer_br, params)
    c089_pose = _choice(c089.decoded["optimized_poses.qp1"], c089.encoded.pose_br, params)
    c089_delta_raw = encode_delta_varint_actions(c089.decoded["seg_tile_actions.bin"])
    c089_actions_p6 = _choice(c089_delta_raw, c089.encoded.actions_br, params)
    c089_actions_p3 = _choice(c089.decoded["seg_tile_actions.bin"], None, params)

    public_mask = _choice(public.decoded["masks.mkv"], public.encoded.mask_br, params)
    public_renderer = _choice(public.decoded["renderer.bin"], public.encoded.renderer_br, params)
    public_pose = _choice(public.decoded["optimized_poses.qp1"], public.encoded.pose_br, params)
    public_actions_p3 = _choice(public.decoded["seg_tile_actions.bin"], None, params)
    public_sorted_actions_raw, public_sort_summary = sorted_action_records_for_p6(
        public.decoded["seg_tile_actions.bin"]
    )
    public_sorted_delta_raw = encode_delta_varint_actions(public_sorted_actions_raw)
    public_actions_p6_sorted = _choice(public_sorted_delta_raw, None, params)

    p5_packed_raw, p5_dict_raw, p5_remapped_actions_raw, p5_summary = encode_p5_action_dict_probe(
        c089.decoded["seg_tile_actions.bin"]
    )
    p5_actions = _choice(p5_packed_raw, None, params)
    p5_dict = _choice(p5_dict_raw, None, params)

    c089_best_choices = {
        "masks.mkv": c089_mask,
        "optimized_poses.qp1": c089_pose,
        "renderer.bin": c089_renderer,
        "seg_tile_actions.delta_varint": c089_actions_p6,
    }
    c089_best_streams = EncodedStreams(
        mask_br=c089_mask.data,
        renderer_br=c089_renderer.data,
        actions_br=c089_actions_p6.data,
        pose_br=c089_pose.data,
        action_record_count=len(c089.decoded["seg_tile_actions.bin"]) // 4,
    )

    specs: list[CandidateSpec] = []
    specs.append(
        CandidateSpec(
            candidate_id="c089_zip_rewrite_noop",
            payload=c089.payload,
            expected_decoded=c089.decoded,
            semantic_contract="strict_source_payload_noop_control",
            selected_sources={"actions": "c089", "mask": "c089", "pose": "c089", "renderer": "c089"},
            next_dispatch_safety_gate="do not dispatch: no-op control",
            dispatchable_after_gate=False,
            source_preserving_vs_c089=True,
            notes=["Rewrites the exact C089 payload into deterministic single-member ZIP; proves ZIP overhead floor/no-op handling."],
            stream_packing={"source_payload_sha256": c089.payload_sha256},
        )
    )
    specs.append(
        CandidateSpec(
            candidate_id="c089_p6_action_resweep",
            payload=_build_p6_payload(
                mask_br=c089.encoded.mask_br,
                renderer_br=c089.encoded.renderer_br,
                actions_br=c089_actions_p6.data,
                record_count=len(c089.decoded["seg_tile_actions.bin"]) // 4,
                pose_br=c089.encoded.pose_br,
            ),
            expected_decoded=c089.decoded,
            semantic_contract="strict_decoded_byte_parity_vs_c089_actions_rebrotli_only",
            selected_sources={"actions": "c089_resweep", "mask": "c089", "pose": "c089", "renderer": "c089"},
            next_dispatch_safety_gate=CUDA_AUTH_EVAL_REQUIRED,
            dispatchable_after_gate=True,
            source_preserving_vs_c089=True,
            notes=["P6 action delta-varint stream re-Brotlied; decoded masks/renderer/poses/actions must remain byte-identical to C089."],
            stream_packing={
                "actions_delta_varint": {
                    "raw_bytes": len(c089_delta_raw),
                    "raw_sha256": _sha256_bytes(c089_delta_raw),
                    **_stream_choices_summary({"brotli": c089_actions_p6})["brotli"],
                }
            },
        )
    )
    specs.append(
        CandidateSpec(
            candidate_id="c089_p6_lossless_stream_resweep",
            payload=_build_p6_payload(
                mask_br=c089_best_streams.mask_br,
                renderer_br=c089_best_streams.renderer_br,
                actions_br=c089_best_streams.actions_br,
                record_count=c089_best_streams.action_record_count or 0,
                pose_br=c089_best_streams.pose_br,
            ),
            expected_decoded=c089.decoded,
            semantic_contract="strict_decoded_byte_parity_vs_c089_all_streams_rebrotli",
            selected_sources={"actions": "c089_resweep", "mask": "c089_resweep", "pose": "c089_resweep", "renderer": "c089_resweep"},
            next_dispatch_safety_gate=CUDA_AUTH_EVAL_REQUIRED,
            dispatchable_after_gate=True,
            source_preserving_vs_c089=True,
            notes=["Lossless stream-level Brotli resweep; this is the strict decoded-byte-preserving C089 byte-screen candidate."],
            stream_packing={
                "choices": _stream_choices_summary(c089_best_choices),
                "actions_delta_varint_raw": {
                    "bytes": len(c089_delta_raw),
                    "sha256": _sha256_bytes(c089_delta_raw),
                },
            },
        )
    )
    specs.append(
        CandidateSpec(
            candidate_id="c089_p3_raw_actions_probe",
            payload=_build_p3_payload(
                mask_br=c089_mask.data,
                renderer_br=c089_renderer.data,
                actions_br=c089_actions_p3.data,
                pose_br=c089_pose.data,
            ),
            expected_decoded=c089.decoded,
            semantic_contract="strict_decoded_byte_parity_vs_c089_p3_raw_actions_probe",
            selected_sources={"actions": "c089_raw_actions_p3", "mask": "c089_resweep", "pose": "c089_resweep", "renderer": "c089_resweep"},
            next_dispatch_safety_gate="do not dispatch unless a byte win appears; exact CUDA auth eval would still be required",
            dispatchable_after_gate=False,
            source_preserving_vs_c089=True,
            notes=["P3 raw action wire is decoded-byte preserving but loses to P6 delta-varint on C089."],
            stream_packing={
                "actions_raw_brotli": {
                    "raw_bytes": len(c089.decoded["seg_tile_actions.bin"]),
                    "raw_sha256": _sha256_bytes(c089.decoded["seg_tile_actions.bin"]),
                    **_stream_choices_summary({"brotli": c089_actions_p3})["brotli"],
                },
                "choices": _stream_choices_summary(
                    {
                        "masks.mkv": c089_mask,
                        "optimized_poses.qp1": c089_pose,
                        "renderer.bin": c089_renderer,
                    }
                ),
            },
        )
    )
    specs.append(
        CandidateSpec(
            candidate_id="public_renderer_c089_p6_lossless_stream_resweep",
            payload=_build_p6_payload(
                mask_br=c089_mask.data,
                renderer_br=public_renderer.data,
                actions_br=c089_actions_p6.data,
                record_count=len(c089.decoded["seg_tile_actions.bin"]) // 4,
                pose_br=c089_pose.data,
            ),
            expected_decoded={
                "masks.mkv": c089.decoded["masks.mkv"],
                "optimized_poses.qp1": c089.decoded["optimized_poses.qp1"],
                "renderer.bin": public.decoded["renderer.bin"],
                "seg_tile_actions.bin": c089.decoded["seg_tile_actions.bin"],
            },
            semantic_contract="renderer_transplant_probe_public_renderer_only_c089_other_streams_byte_identical",
            selected_sources={"actions": "c089_resweep", "mask": "c089_resweep", "pose": "c089_resweep", "renderer": "public_pr75_minp"},
            next_dispatch_safety_gate=RENDERER_TRANSPLANT_GATE,
            dispatchable_after_gate=True,
            source_preserving_vs_c089=False,
            notes=["Same decoded renderer swap as the queued P6 public renderer-only candidate, with lossless C089 mask/pose/action repack folded in."],
            stream_packing={
                "choices": _stream_choices_summary(
                    {
                        "masks.mkv": c089_mask,
                        "optimized_poses.qp1": c089_pose,
                        "renderer.bin": public_renderer,
                        "seg_tile_actions.delta_varint": c089_actions_p6,
                    }
                )
            },
        )
    )
    specs.append(
        CandidateSpec(
            candidate_id="public_renderer_pose_c089_p6_lossless_stream_resweep",
            payload=_build_p6_payload(
                mask_br=c089_mask.data,
                renderer_br=public_renderer.data,
                actions_br=c089_actions_p6.data,
                record_count=len(c089.decoded["seg_tile_actions.bin"]) // 4,
                pose_br=public_pose.data,
            ),
            expected_decoded={
                "masks.mkv": c089.decoded["masks.mkv"],
                "optimized_poses.qp1": public.decoded["optimized_poses.qp1"],
                "renderer.bin": public.decoded["renderer.bin"],
                "seg_tile_actions.bin": c089.decoded["seg_tile_actions.bin"],
            },
            semantic_contract="renderer_pose_transplant_probe_public_renderer_pose_c089_actions_masks",
            selected_sources={"actions": "c089_resweep", "mask": "c089_resweep", "pose": "public_pr75_minp", "renderer": "public_pr75_minp"},
            next_dispatch_safety_gate=LOCAL_PARITY_GATE,
            dispatchable_after_gate=False,
            source_preserving_vs_c089=False,
            notes=["Changes renderer and pose decoded streams; kept as matrix context because it is not a byte win versus C089."],
            stream_packing={
                "choices": _stream_choices_summary(
                    {
                        "masks.mkv": c089_mask,
                        "optimized_poses.qp1": public_pose,
                        "renderer.bin": public_renderer,
                        "seg_tile_actions.delta_varint": c089_actions_p6,
                    }
                )
            },
        )
    )
    specs.append(
        CandidateSpec(
            candidate_id="public_minp_p3_raw_actions_probe",
            payload=_build_p3_payload(
                mask_br=public_mask.data,
                renderer_br=public_renderer.data,
                actions_br=public_actions_p3.data,
                pose_br=public_pose.data,
            ),
            expected_decoded=public.decoded,
            semantic_contract="strict_decoded_byte_parity_vs_public_minp_p3_raw_actions_probe",
            selected_sources={"actions": "public_pr75_minp_raw_actions_p3", "mask": "public_pr75_minp_resweep", "pose": "public_pr75_minp", "renderer": "public_pr75_minp"},
            next_dispatch_safety_gate="do not dispatch: byte regression versus C089 and public fixed-slice source",
            dispatchable_after_gate=False,
            source_preserving_vs_c089=False,
            notes=["Public/minp fixed SG2 action stream beats self-describing P3 raw action stream after header costs."],
            stream_packing={
                "actions_raw_brotli": {
                    "raw_bytes": len(public.decoded["seg_tile_actions.bin"]),
                    "raw_sha256": _sha256_bytes(public.decoded["seg_tile_actions.bin"]),
                    **_stream_choices_summary({"brotli": public_actions_p3})["brotli"],
                },
                "choices": _stream_choices_summary(
                    {
                        "masks.mkv": public_mask,
                        "optimized_poses.qp1": public_pose,
                        "renderer.bin": public_renderer,
                    }
                ),
            },
        )
    )
    specs.append(
        CandidateSpec(
            candidate_id="public_minp_p6_sorted_actions_probe",
            payload=_build_p6_payload(
                mask_br=public_mask.data,
                renderer_br=public_renderer.data,
                actions_br=public_actions_p6_sorted.data,
                record_count=len(public_sorted_actions_raw) // 4,
                pose_br=public_pose.data,
            ),
            expected_decoded={
                "masks.mkv": public.decoded["masks.mkv"],
                "optimized_poses.qp1": public.decoded["optimized_poses.qp1"],
                "renderer.bin": public.decoded["renderer.bin"],
                "seg_tile_actions.bin": public_sorted_actions_raw,
            },
            semantic_contract="non_dispatchable_public_actions_ordering_probe_p6_sorted_delta_varint",
            selected_sources={"actions": "public_pr75_minp_sorted_probe", "mask": "public_pr75_minp_resweep", "pose": "public_pr75_minp", "renderer": "public_pr75_minp"},
            next_dispatch_safety_gate=LOCAL_PARITY_GATE,
            dispatchable_after_gate=False,
            source_preserving_vs_c089=False,
            notes=[
                "Public/minp source actions are not pair-sorted; sorting enables P6 but changes decoded action record order.",
                "Duplicate pair/tile entries make commutation unsafe without runtime output parity.",
            ],
            stream_packing={
                "actions_sorted_delta_varint": {
                    "brotli": _stream_choices_summary({"brotli": public_actions_p6_sorted})["brotli"],
                    "raw_bytes": len(public_sorted_delta_raw),
                    "raw_sha256": _sha256_bytes(public_sorted_delta_raw),
                    "sort_summary": public_sort_summary,
                },
                "choices": _stream_choices_summary(
                    {
                        "masks.mkv": public_mask,
                        "optimized_poses.qp1": public_pose,
                        "renderer.bin": public_renderer,
                    }
                ),
            },
        )
    )
    specs.append(
        CandidateSpec(
            candidate_id="c089_p5_action_dict_probe",
            payload=_build_p5_payload(
                mask_br=c089_mask.data,
                renderer_br=c089_renderer.data,
                action_dict_br=p5_dict.data,
                actions_br=p5_actions.data,
                record_count=len(c089.decoded["seg_tile_actions.bin"]) // 4,
                pose_br=c089_pose.data,
            ),
            expected_decoded={
                "masks.mkv": c089.decoded["masks.mkv"],
                "optimized_poses.qp1": c089.decoded["optimized_poses.qp1"],
                "renderer.bin": c089.decoded["renderer.bin"],
                "seg_tile_action_dict.bin": p5_dict_raw,
                "seg_tile_actions.bin": p5_remapped_actions_raw,
            },
            semantic_contract="non_dispatchable_p5_action_dictionary_remap_probe",
            selected_sources={"actions": "c089_p5_remapped", "mask": "c089_resweep", "pose": "c089_resweep", "renderer": "c089_resweep"},
            next_dispatch_safety_gate=LOCAL_PARITY_GATE,
            dispatchable_after_gate=False,
            source_preserving_vs_c089=False,
            notes=["P5 3-byte action packing needs a charged custom dictionary and is a byte regression for C089."],
            stream_packing={
                "actions_p5_packed": {
                    "brotli": _stream_choices_summary({"brotli": p5_actions})["brotli"],
                    "raw_bytes": len(p5_packed_raw),
                    "raw_sha256": _sha256_bytes(p5_packed_raw),
                },
                "action_dict": {
                    "brotli": _stream_choices_summary({"brotli": p5_dict})["brotli"],
                    "raw_bytes": len(p5_dict_raw),
                    "raw_sha256": _sha256_bytes(p5_dict_raw),
                },
                "semantic_summary": p5_summary,
            },
        )
    )
    specs.append(
        CandidateSpec(
            candidate_id="c089_raw_no_header_fixedslice_probe",
            payload=_build_raw_concat_payload(c089_best_streams),
            expected_decoded=None,
            semantic_contract="invalid_current_runtime_raw_fixedslice_header_removal_probe",
            selected_sources={"actions": "c089_resweep", "mask": "c089_resweep", "pose": "c089_resweep", "renderer": "c089_resweep"},
            next_dispatch_safety_gate=NON_DISPATCHABLE_GATE,
            dispatchable_after_gate=False,
            source_preserving_vs_c089=False,
            notes=[
                "Drops the P6 self-describing header and shows the fixed-slice overhead opportunity.",
                "Current robust_current parser rejects this C089 stream-length tuple, so it is not contest-faithful today.",
            ],
            stream_packing={
                "choices": _stream_choices_summary(c089_best_choices),
                "unsupported_raw_payload_contract": {
                    "reason": "robust_current fixed PR75 slice table has no C089 tuple for mask/model/actions/pose lengths",
                    "would_remove_p6_header_bytes": 10,
                },
            },
        )
    )
    return specs


def build_candidates(
    *,
    c089_archive: Path,
    public_archive: Path,
    output_dir: Path,
    force: bool = False,
    params: Iterable[tuple[int, int, int, int]] | None = None,
    unpacker: Any | None = None,
) -> dict[str, Any]:
    if unpacker is None:
        unpacker = _load_unpacker()
    if params is None:
        params = default_brotli_param_grid()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    c089 = _load_source("c089_p6_frontier", c089_archive, unpacker)
    public = _load_source("public_pr75_minp", public_archive, unpacker)
    if c089.archive_sha256 != BASELINE_SHA256:
        raise ValueError(f"C089 archive SHA mismatch: expected {BASELINE_SHA256}, got {c089.archive_sha256}")
    accepted_public_shas = {PUBLIC_MINP_SHA256, PUBLIC_PR79_MINP_V2_SHA256}
    if public.archive_sha256 not in accepted_public_shas:
        raise ValueError(
            "public PR75/PR79 minp archive SHA mismatch: "
            f"expected one of {sorted(accepted_public_shas)}, got {public.archive_sha256}"
        )

    rows = [
        _emit_candidate(
            spec=spec,
            output_dir=output_dir,
            c089=c089,
            public=public,
            unpacker=unpacker,
            force=force,
        )
        for spec in _build_candidate_specs(c089=c089, public=public, params=params)
    ]
    rows = sorted(rows, key=lambda row: (row["archive_bytes"], row["candidate_id"]))
    summary = {
        "baseline": {
            "archive_bytes": c089.archive_bytes,
            "archive_sha256": c089.archive_sha256,
            "canonical_score": BASELINE_SCORE,
            "candidate_id": "C089/c067_pr75_qp1_top40_p6",
        },
        "candidate_count": len(rows),
        "candidates": rows,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "evidence_grade": "empirical_byte_screen_only",
        "promotion_eligible": False,
        "schema": "pr75_lossless_micro_candidate_matrix_v1",
        "score_claim": False,
        "source_archives": {
            "c089": _source_summary(c089),
            "public_pr75_minp": _source_summary(public),
        },
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c089-archive", type=Path, default=DEFAULT_C089_ARCHIVE)
    parser.add_argument("--public-archive", type=Path, default=DEFAULT_PUBLIC_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--exhaustive-brotli-grid",
        action="store_true",
        help="Search the full Brotli parameter grid instead of the focused default grid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_candidates(
        c089_archive=args.c089_archive,
        public_archive=args.public_archive,
        output_dir=args.output_dir,
        force=bool(args.force),
        params=exhaustive_brotli_param_grid()
        if args.exhaustive_brotli_grid
        else default_brotli_param_grid(),
    )
    print(
        json.dumps(
            {
                "best_dispatchable_by_bytes": next(
                    (row for row in summary["candidates"] if row["dispatchable_after_gate"]),
                    None,
                ),
                "best_overall_by_bytes": summary["candidates"][0] if summary["candidates"] else None,
                "candidate_count": len(summary["candidates"]),
                "output_dir": str(Path(args.output_dir).resolve()),
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
