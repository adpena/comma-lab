#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build deterministic PR77-action / C089-pose mixed container candidates.

This is a local byte-screening builder only. It targets the gap that the
one-hop PR77 fixed-slice transplant cannot cover: PR77's action stream is not
P6-delta encodable in source order, but the current runtime can carry it
through the fixed-slice or P3 PR75 action path while reusing C089's smaller QP1
pose stream. Outputs are not score evidence until exact CUDA auth eval runs on
the identical archive bytes after the dispatch-claim protocol.
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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
DEFAULT_C091_ARCHIVE = (
    REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip"
)
DEFAULT_C089_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip"
)
DEFAULT_PR77_ARCHIVE = (
    REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/pr77_action_pose_mixed_container_20260503_codex"
)
TOOL = "experiments/build_pr77_action_pose_mixed_container_candidates.py"
SCHEMA = "pr77_action_pose_mixed_container_candidate_matrix_v1"
MANIFEST_SCHEMA = "pr77_action_pose_mixed_container_candidate_manifest_v1"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MEMBER_NAME = "p"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
SUB314_TARGET = 0.314
CUDA_AUTH_EVAL_REQUIRED = (
    "claim lane via tools/claim_lane_dispatch.py, then exact CUDA auth eval: "
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
LOCAL_ACTION_ORDER_PARITY_GATE = (
    "non-dispatchable until a local PR75 action-order raw-output parity proof "
    "or equivalent reviewed runtime proof shows sorted action records are "
    "output-equivalent; then claim lane and run exact CUDA auth eval"
)
STREAM_ORDER = ("masks.mkv", "renderer.bin", "seg_tile_actions.bin", "optimized_poses.qp1")

ANCHORS = {
    "c091_pr75_replay": {
        "archive_bytes": 276_481,
        "archive_sha256": "03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746",
        "score": 0.31516575028285976,
    },
    "c089_pr75_qp1_top40_p6": {
        "archive_bytes": 276_342,
        "archive_sha256": "0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8",
        "score": 0.3154707273953505,
    },
    "pr77_action_delta_public": {
        "archive_bytes": 276_551,
        "archive_sha256": "f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af",
        "score": None,
    },
}


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
    action_record_count: int | None


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
    expected_decoded: dict[str, bytes]
    selected_sources: dict[str, str]
    semantic_contract: str
    next_dispatch_safety_gate: str
    dispatchable_after_gate: bool
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
    spec = importlib.util.spec_from_file_location("pr77_action_pose_unpacker", UNPACKER_PATH)
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


def _member_summary(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    members = header.get("members")
    if not isinstance(members, list):
        raise ValueError("payload parser returned no member table")
    out: dict[str, dict[str, Any]] = {}
    for item in members:
        if not isinstance(item, Mapping):
            raise ValueError("payload parser returned malformed member metadata")
        name = str(item["name"])
        out[name] = {
            "bytes": int(item["bytes"]),
            "codec": str(item["codec"]),
            "decoded_bytes": int(item.get("decoded_bytes", -1)),
            "decoded_sha256": str(item.get("decoded_sha256", "")),
            "sha256": str(item["sha256"]),
        }
    return out


def _decoded_summary(decoded: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
        for name, data in sorted(decoded.items())
    }


def _parse_self_describing_payload(payload: bytes) -> EncodedStreams:
    action_record_count: int | None = None
    if payload.startswith(b"P3"):
        cursor = 2 + struct.calcsize("<IHH")
        mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
    elif payload.startswith(b"P6"):
        cursor = 2 + struct.calcsize("<IHHH")
        mask_len, renderer_len, actions_len, action_record_count = struct.unpack_from(
            "<IHHH",
            payload,
            2,
        )
    else:
        raise ValueError(f"unsupported self-describing payload prefix {payload[:2]!r}")
    if min(mask_len, renderer_len, actions_len) <= 0:
        raise ValueError("empty encoded stream in source payload")
    mask_end = cursor + mask_len
    renderer_end = mask_end + renderer_len
    actions_end = renderer_end + actions_len
    if actions_end >= len(payload):
        raise ValueError("encoded stream lengths leave no pose payload")
    return EncodedStreams(
        mask_br=payload[cursor:mask_end],
        renderer_br=payload[mask_end:renderer_end],
        actions_br=payload[renderer_end:actions_end],
        pose_br=payload[actions_end:],
        action_record_count=action_record_count,
    )


def _parse_fixed_slice_payload(
    *,
    label: str,
    payload: bytes,
    runtime_members: Mapping[str, Mapping[str, Any]],
) -> EncodedStreams:
    missing = sorted(set(STREAM_ORDER) - set(runtime_members))
    if missing:
        raise ValueError(f"{label}: missing fixed-slice members {missing}")
    offset = 0
    raw: dict[str, bytes] = {}
    for name in STREAM_ORDER:
        size = int(runtime_members[name]["bytes"])
        if size <= 0:
            raise ValueError(f"{label}: non-positive raw segment size for {name}: {size}")
        segment = payload[offset : offset + size]
        offset += size
        expected_sha = str(runtime_members[name].get("sha256", ""))
        actual_sha = _sha256_bytes(segment)
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(
                f"{label}: raw SHA mismatch for {name}: expected {expected_sha}, got {actual_sha}"
            )
        raw[name] = segment
    if offset != len(payload):
        raise ValueError(f"{label}: fixed slices consume {offset}, payload has {len(payload)}")
    return EncodedStreams(
        mask_br=raw["masks.mkv"],
        renderer_br=raw["renderer.bin"],
        actions_br=raw["seg_tile_actions.bin"],
        pose_br=raw["optimized_poses.qp1"],
        action_record_count=None,
    )


def _parse_encoded_streams(
    *,
    label: str,
    payload: bytes,
    payload_format: str,
    runtime_members: Mapping[str, Mapping[str, Any]],
) -> EncodedStreams:
    if payload.startswith((b"P3", b"P6")):
        return _parse_self_describing_payload(payload)
    if payload_format == "public_pr75_qzs3_qp1_segactions_fixed_slices":
        return _parse_fixed_slice_payload(
            label=label,
            payload=payload,
            runtime_members=runtime_members,
        )
    raise ValueError(f"{label}: unsupported payload format {payload_format!r}")


def _load_source(label: str, path: Path, unpacker: Any) -> SourceArchive:
    path = path.resolve()
    payload = _read_single_payload_member(path)
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    payload_format = str(header.get("payload_format"))
    runtime_members = _member_summary(header)
    encoded = _parse_encoded_streams(
        label=label,
        payload=payload,
        payload_format=payload_format,
        runtime_members=runtime_members,
    )
    return SourceArchive(
        label=label,
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=payload_format,
        encoded=encoded,
        decoded=decoded,
        runtime_members=runtime_members,
    )


def _compress(raw: bytes, params: tuple[int, int, int, int]) -> bytes:
    quality, mode, lgwin, lgblock = params
    return brotli.compress(
        raw,
        quality=quality,
        mode=mode,
        lgwin=lgwin,
        lgblock=lgblock,
    )


def _best_brotli(
    raw: bytes,
    *,
    source: bytes | None,
    params: Iterable[tuple[int, int, int, int]],
    default_quality_11: bool = False,
) -> BrotliChoice:
    best = source
    best_params: dict[str, int] | str = "source" if source is not None else {}
    if default_quality_11:
        default = brotli.compress(raw, quality=11)
        best = default if best is None or len(default) < len(best) else best
        if best is default:
            best_params = "brotli_quality_11_default"
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


def _small_action_param_grid() -> list[tuple[int, int, int, int]]:
    params: list[tuple[int, int, int, int]] = []
    for quality in (11, 10, 9, 6, 0):
        for mode in (0, 1, 2):
            for lgwin in (10, 12, 16, 18, 20, 24):
                params.append((quality, mode, lgwin, 0))
    return params


def _known_c089_mask_choice(c089: SourceArchive) -> BrotliChoice:
    return _best_brotli(
        c089.decoded["masks.mkv"],
        source=c089.encoded.mask_br,
        params=[(11, 0, 19, 17)],
    )


def _known_c089_pose_choice(c089: SourceArchive) -> BrotliChoice:
    return _best_brotli(
        c089.decoded["optimized_poses.qp1"],
        source=c089.encoded.pose_br,
        params=[(11, 0, 16, 0)],
    )


def _read_action_records(raw_actions: bytes) -> list[tuple[int, int, int]]:
    if len(raw_actions) % 4:
        raise ValueError(f"PR75 action records must be 4 bytes each, got {len(raw_actions)}")
    records: list[tuple[int, int, int]] = []
    for offset in range(0, len(raw_actions), 4):
        pair_index = int.from_bytes(raw_actions[offset : offset + 2], "little")
        records.append((pair_index, raw_actions[offset + 2], raw_actions[offset + 3]))
    return records


def _records_to_raw(records: Iterable[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for pair_index, tile_id, action_id in records:
        if not (0 <= pair_index < 10_000):
            raise ValueError(f"action pair index out of bounds: {pair_index}")
        if not (0 <= tile_id < 192):
            raise ValueError(f"action tile id out of bounds: {tile_id}")
        if not (0 <= action_id < 256):
            raise ValueError(f"action id out of bounds: {action_id}")
        out += pair_index.to_bytes(2, "little") + bytes([tile_id, action_id])
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


def _encode_delta_varint_actions(raw_actions: bytes) -> bytes:
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


def _action_stats(raw_actions: bytes) -> dict[str, Any]:
    records = _read_action_records(raw_actions)
    if not records:
        raise ValueError("empty action stream")
    pair_tile_counts = Counter((pair, tile) for pair, tile, _action in records)
    duplicate_pair_tiles = [
        {"pair": pair, "tile": tile, "count": count}
        for (pair, tile), count in sorted(pair_tile_counts.items())
        if count > 1
    ]
    return {
        "action_max": max(action for _pair, _tile, action in records),
        "action_min": min(action for _pair, _tile, action in records),
        "duplicate_pair_tile_count": len(duplicate_pair_tiles),
        "duplicate_pair_tiles_first10": duplicate_pair_tiles[:10],
        "nondecreasing_pair_order": all(
            records[index][0] <= records[index + 1][0]
            for index in range(len(records) - 1)
        ),
        "pair_max": max(pair for pair, _tile, _action in records),
        "pair_min": min(pair for pair, _tile, _action in records),
        "record_count": len(records),
        "records_sha256": _sha256_bytes(raw_actions),
        "tile_max": max(tile for _pair, tile, _action in records),
        "tile_min": min(tile for _pair, tile, _action in records),
        "unique_actions": len({action for _pair, _tile, action in records}),
        "unique_pairs": len({pair for pair, _tile, _action in records}),
        "unique_tiles": len({tile for _pair, tile, _action in records}),
    }


def _sorted_p6_action_choice(pr77_actions: bytes) -> tuple[bytes, BrotliChoice, dict[str, Any]]:
    records = _read_action_records(pr77_actions)
    sorted_raw = _records_to_raw(sorted(records, key=lambda item: (item[0], item[1], item[2])))
    stats = _action_stats(pr77_actions)
    sorted_stats = _action_stats(sorted_raw)
    delta_raw = _encode_delta_varint_actions(sorted_raw)
    encoded = _best_brotli(
        delta_raw,
        source=None,
        params=_small_action_param_grid(),
        default_quality_11=True,
    )
    proof_status = (
        "structural_disjoint_pair_tile_targets_only"
        if stats["duplicate_pair_tile_count"] == 0
        else "failed_duplicate_pair_tile_targets"
    )
    return sorted_raw, encoded, {
        "encoded_delta_varint": {
            "brotli_bytes": len(encoded.data),
            "brotli_params": encoded.params,
            "brotli_sha256": _sha256_bytes(encoded.data),
            "raw_bytes": len(delta_raw),
            "raw_sha256": _sha256_bytes(delta_raw),
        },
        "original_action_stats": stats,
        "proof_status": proof_status,
        "sorted_action_stats": sorted_stats,
        "sorted_records_sha256": _sha256_bytes(sorted_raw),
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


def _validate_candidate_payload(
    *,
    candidate_id: str,
    payload: bytes,
    expected_decoded: Mapping[str, bytes],
    unpacker: Any,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    runtime_members = _member_summary(header)
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
    status = "passed" if not missing and not extra and not mismatches else "failed"
    validation = {
        "candidate_id": candidate_id,
        "decoded_parity_status": status,
        "extra": extra,
        "members": runtime_members,
        "mismatches": mismatches,
        "missing": missing,
        "payload_format": str(header.get("payload_format")),
        "runtime_parser": str(UNPACKER_PATH),
        "status": "passed" if status == "passed" else "failed_decoded_parity",
    }
    if status != "passed":
        raise ValueError(f"{candidate_id}: decoded parity failed: {validation}")
    return validation, decoded


def _fixed_slice_boundary_validation(
    *,
    payload: bytes,
    validation: Mapping[str, Any],
    selected_raw: Mapping[str, bytes],
) -> dict[str, Any]:
    if validation.get("payload_format") != "public_pr75_qzs3_qp1_segactions_fixed_slices":
        return {"status": "not_fixed_slice_payload"}
    members = validation.get("members")
    if not isinstance(members, Mapping):
        raise ValueError("fixed-slice validation requires runtime members")
    offset = 0
    segments: dict[str, dict[str, Any]] = {}
    for name in STREAM_ORDER:
        meta = members[name]
        size = int(meta["bytes"])
        raw = payload[offset : offset + size]
        offset += size
        expected = selected_raw[name]
        raw_sha = _sha256_bytes(raw)
        expected_sha = _sha256_bytes(expected)
        if raw != expected:
            raise ValueError(f"fixed-slice boundary mismatch for {name}")
        segments[name] = {
            "bytes": size,
            "raw_sha256": raw_sha,
            "selected_raw_sha256": expected_sha,
        }
    if offset != len(payload):
        raise ValueError(f"fixed-slice boundary consumed {offset}, payload has {len(payload)}")
    return {
        "raw_wire_order": list(STREAM_ORDER),
        "segments": segments,
        "status": "passed",
    }


def _source_summary(source: SourceArchive) -> dict[str, Any]:
    return {
        "archive_bytes": source.archive_bytes,
        "archive_sha256": source.archive_sha256,
        "decoded_members": _decoded_summary(source.decoded),
        "encoded_streams": {
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
        },
        "path": str(source.path),
        "payload_bytes": len(source.payload),
        "payload_format": source.payload_format,
        "payload_sha256": source.payload_sha256,
    }


def _decoded_change_summary(
    *,
    anchors: Mapping[str, SourceArchive],
    decoded: Mapping[str, bytes],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, source in anchors.items():
        names = sorted(set(source.decoded) | set(decoded))
        changed = []
        members = {}
        for name in names:
            candidate_data = decoded.get(name)
            source_data = source.decoded.get(name)
            changed_vs_source = candidate_data != source_data
            if changed_vs_source:
                changed.append(name)
            members[name] = {
                "candidate_bytes": len(candidate_data) if candidate_data is not None else None,
                "candidate_sha256": _sha256_bytes(candidate_data) if candidate_data is not None else None,
                "source_bytes": len(source_data) if source_data is not None else None,
                "source_sha256": _sha256_bytes(source_data) if source_data is not None else None,
                "changed_vs_source": changed_vs_source,
            }
        out[label] = {
            "changed_decoded_streams": changed,
            "members": members,
            "status": "decoded_stream_changed" if changed else "decoded_stream_byte_identical",
        }
    return out


def _anchor_break_even(archive_bytes: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, anchor in ANCHORS.items():
        anchor_bytes = int(anchor["archive_bytes"])
        delta_bytes = archive_bytes - anchor_bytes
        score = anchor["score"]
        row: dict[str, Any] = {
            "archive_delta_bytes": delta_bytes,
            "rate_score_delta": delta_bytes * RATE_SCORE_PER_BYTE,
        }
        if score is not None:
            score_if_components_unchanged = float(score) + row["rate_score_delta"]
            required_score_improvement = max(0.0, score_if_components_unchanged - SUB314_TARGET)
            row.update(
                {
                    "anchor_score": float(score),
                    "score_if_components_unchanged": score_if_components_unchanged,
                    "sub314_component_score_improvement_needed": required_score_improvement,
                    "sub314_equivalent_bytes_needed_after_candidate": math.ceil(
                        required_score_improvement / RATE_SCORE_PER_BYTE
                    )
                    if required_score_improvement > 0
                    else 0,
                }
            )
        out[label] = row
    return out


def _archive_profile(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        members = [
            {
                "compress_size": info.compress_size,
                "compress_type": info.compress_type,
                "date_time": list(info.date_time),
                "external_attr": info.external_attr,
                "filename": info.filename,
                "file_size": info.file_size,
            }
            for info in infos
        ]
    return {
        "archive_bytes": path.stat().st_size,
        "member_count": len(members),
        "members": members,
        "schema": "minimal_zip_profile_v1",
        "single_stored_member_floor_bytes": members[0]["file_size"] + 100
        if len(members) == 1 and members[0]["filename"] == MEMBER_NAME
        else None,
    }


def _candidate_manifest(
    *,
    spec: CandidateSpec,
    archive_path: Path,
    sources: Mapping[str, SourceArchive],
    validation: Mapping[str, Any],
    decoded: Mapping[str, bytes],
    fixed_slice_validation: Mapping[str, Any],
) -> dict[str, Any]:
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256_file(archive_path)
    payload_sha = _sha256_bytes(spec.payload)
    noops = {
        label: {
            "archive_byte_identical": archive_bytes == source.archive_bytes
            and archive_sha == source.archive_sha256,
            "payload_byte_identical": spec.payload == source.payload,
        }
        for label, source in sources.items()
    }
    semantic_noop = any(item["payload_byte_identical"] for item in noops.values())
    archive_noop = any(
        item["archive_byte_identical"] and item["payload_byte_identical"]
        for item in noops.values()
    )
    if archive_noop:
        noop_status = "byte_identical_to_source_archive"
    elif semantic_noop:
        noop_status = "payload_identical_zip_metadata_changed"
    else:
        noop_status = "non_noop_payload"
    return {
        "anchor_break_even": _anchor_break_even(archive_bytes),
        "archive_profile": _archive_profile(archive_path),
        "candidate_id": spec.candidate_id,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "decoded_change_summary": _decoded_change_summary(anchors=sources, decoded=decoded),
        "dispatch_safety": {
            "dispatchable_after_gate": bool(
                spec.dispatchable_after_gate
                and validation.get("status") == "passed"
                and not semantic_noop
            ),
            "next_dispatch_safety_gate": spec.next_dispatch_safety_gate,
        },
        "evidence_grade": "empirical_byte_screen_only",
        "fixed_slice_boundary_validation": fixed_slice_validation,
        "archive_noop": archive_noop,
        "noop": semantic_noop,
        "noop_checks": noops,
        "noop_status": noop_status,
        "notes": spec.notes,
        "output_archive": {
            "bytes": archive_bytes,
            "path": str(archive_path),
            "sha256": archive_sha,
        },
        "payload": {
            "bytes": len(spec.payload),
            "member": MEMBER_NAME,
            "sha256": payload_sha,
        },
        "promotion_eligible": False,
        "runtime_parse_validation": validation,
        "schema": MANIFEST_SCHEMA,
        "score_claim": False,
        "selected_sources": spec.selected_sources,
        "semantic_contract": spec.semantic_contract,
        "source_archives": {label: _source_summary(source) for label, source in sources.items()},
        "stream_packing": spec.stream_packing,
        "tool": TOOL,
    }


def _emit_candidate(
    *,
    spec: CandidateSpec,
    output_dir: Path,
    sources: Mapping[str, SourceArchive],
    selected_raw: Mapping[str, bytes] | None,
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
        candidate_id=spec.candidate_id,
        payload=spec.payload,
        expected_decoded=spec.expected_decoded,
        unpacker=unpacker,
    )
    fixed_slice_validation = (
        _fixed_slice_boundary_validation(
            payload=spec.payload,
            validation=validation,
            selected_raw=selected_raw,
        )
        if selected_raw is not None
        else {"status": "not_fixed_slice_payload"}
    )
    manifest = _candidate_manifest(
        spec=spec,
        archive_path=archive_path,
        sources=sources,
        validation=validation,
        decoded=decoded,
        fixed_slice_validation=fixed_slice_validation,
    )
    _write_json(manifest_path, manifest)
    anchor_break_even = manifest["anchor_break_even"]
    return {
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_path": manifest["output_archive"]["path"],
        "archive_sha256": manifest["output_archive"]["sha256"],
        "candidate_id": spec.candidate_id,
        "delta_bytes_vs_c089": anchor_break_even["c089_pr75_qp1_top40_p6"]["archive_delta_bytes"],
        "delta_bytes_vs_c091": anchor_break_even["c091_pr75_replay"]["archive_delta_bytes"],
        "delta_bytes_vs_pr77": anchor_break_even["pr77_action_delta_public"]["archive_delta_bytes"],
        "dispatchable_after_gate": manifest["dispatch_safety"]["dispatchable_after_gate"],
        "manifest_path": str(manifest_path),
        "noop": manifest["noop"],
        "noop_status": manifest["noop_status"],
        "payload_bytes": manifest["payload"]["bytes"],
        "payload_format": validation["payload_format"],
        "score_claim": False,
        "semantic_contract": spec.semantic_contract,
        "sub314_equivalent_bytes_needed_vs_c091": anchor_break_even["c091_pr75_replay"][
            "sub314_equivalent_bytes_needed_after_candidate"
        ],
        "sub314_score_improvement_needed_vs_c091": anchor_break_even["c091_pr75_replay"][
            "sub314_component_score_improvement_needed"
        ],
    }


def _build_candidate_specs(
    *,
    c091: SourceArchive,
    c089: SourceArchive,
    pr77: SourceArchive,
) -> tuple[list[tuple[CandidateSpec, dict[str, bytes] | None]], dict[str, Any]]:
    c089_mask = _known_c089_mask_choice(c089)
    c089_pose = _known_c089_pose_choice(c089)
    pr77_actions = pr77.encoded.actions_br
    pr77_actions_decoded = pr77.decoded["seg_tile_actions.bin"]
    sorted_pr77_actions, sorted_action_br, sorted_action_summary = _sorted_p6_action_choice(
        pr77_actions_decoded
    )
    record_count = len(pr77_actions_decoded) // 4
    common_decoded = {
        "masks.mkv": c091.decoded["masks.mkv"],
        "optimized_poses.qp1": c089.decoded["optimized_poses.qp1"],
        "renderer.bin": c091.decoded["renderer.bin"],
        "seg_tile_actions.bin": pr77_actions_decoded,
    }

    fixed_raw = {
        "masks.mkv": c091.encoded.mask_br,
        "optimized_poses.qp1": c089_pose.data,
        "renderer.bin": c091.encoded.renderer_br,
        "seg_tile_actions.bin": pr77_actions,
    }
    fixed_payload = b"".join(fixed_raw[name] for name in STREAM_ORDER)

    candidates: list[tuple[CandidateSpec, dict[str, bytes] | None]] = [
        (
            CandidateSpec(
                candidate_id="pr77_actions_pr75mask_renderer_c089pose_fixedslice",
                payload=fixed_payload,
                expected_decoded=common_decoded,
                selected_sources={
                    "actions": "pr77_action_delta_public",
                    "mask": "c091_pr75_replay",
                    "pose": "c089_pr75_qp1_top40_p6_lossless_resweep",
                    "renderer": "c091_pr75_replay",
                },
                semantic_contract="current_runtime_fixedslice_pr77_actions_with_c089_pose",
                next_dispatch_safety_gate=CUDA_AUTH_EVAL_REQUIRED,
                dispatchable_after_gate=True,
                notes=[
                    "Carries PR77's exact source-order action stream without P6 reordering.",
                    "Uses current robust fixed-slice parser; manifest records raw wire boundaries.",
                    "Uses C089 QP1 pose bytes to avoid PR77/public 898-byte pose cost.",
                ],
                stream_packing={
                    "c089_pose_choice": {
                        "bytes": len(c089_pose.data),
                        "params": c089_pose.params,
                        "sha256": _sha256_bytes(c089_pose.data),
                    },
                    "pr77_actions_encoded": {
                        "bytes": len(pr77_actions),
                        "sha256": _sha256_bytes(pr77_actions),
                    },
                },
            ),
            fixed_raw,
        ),
        (
            CandidateSpec(
                candidate_id="pr77_actions_c089mask_pr75renderer_c089pose_p3",
                payload=_build_p3_payload(
                    mask_br=c089_mask.data,
                    renderer_br=c091.encoded.renderer_br,
                    actions_br=pr77_actions,
                    pose_br=c089_pose.data,
                ),
                expected_decoded=common_decoded,
                selected_sources={
                    "actions": "pr77_action_delta_public",
                    "mask": "c089_pr75_qp1_top40_p6_lossless_resweep",
                    "pose": "c089_pr75_qp1_top40_p6_lossless_resweep",
                    "renderer": "c091_pr75_replay",
                },
                semantic_contract="self_describing_p3_pr77_actions_with_c089_mask_pose",
                next_dispatch_safety_gate=CUDA_AUTH_EVAL_REQUIRED,
                dispatchable_after_gate=True,
                notes=[
                    "Self-describing fallback for the fixed-slice candidate.",
                    "Costs a P3 header but uses the 7-byte smaller C089 mask resweep.",
                ],
                stream_packing={
                    "c089_mask_choice": {
                        "bytes": len(c089_mask.data),
                        "params": c089_mask.params,
                        "sha256": _sha256_bytes(c089_mask.data),
                    },
                    "c089_pose_choice": {
                        "bytes": len(c089_pose.data),
                        "params": c089_pose.params,
                        "sha256": _sha256_bytes(c089_pose.data),
                    },
                    "pr77_actions_encoded": {
                        "bytes": len(pr77_actions),
                        "sha256": _sha256_bytes(pr77_actions),
                    },
                },
            ),
            None,
        ),
        (
            CandidateSpec(
                candidate_id="pr77_actions_sorted_c089mask_pr75renderer_c089pose_p6_probe",
                payload=_build_p6_payload(
                    mask_br=c089_mask.data,
                    renderer_br=c091.encoded.renderer_br,
                    actions_br=sorted_action_br.data,
                    record_count=record_count,
                    pose_br=c089_pose.data,
                ),
                expected_decoded={
                    **common_decoded,
                    "seg_tile_actions.bin": sorted_pr77_actions,
                },
                selected_sources={
                    "actions": "pr77_action_delta_public_sorted_for_p6_probe",
                    "mask": "c089_pr75_qp1_top40_p6_lossless_resweep",
                    "pose": "c089_pr75_qp1_top40_p6_lossless_resweep",
                    "renderer": "c091_pr75_replay",
                },
                semantic_contract="non_dispatchable_sorted_pr77_actions_p6_probe",
                next_dispatch_safety_gate=LOCAL_ACTION_ORDER_PARITY_GATE,
                dispatchable_after_gate=False,
                notes=[
                    "PR77 has no duplicate pair/tile records, so sorting targets disjoint tiles, but decoded record order changes.",
                    "Fail closed until a local raw-output parity proof confirms the runtime output is unchanged.",
                ],
                stream_packing={
                    "action_order_probe": sorted_action_summary,
                    "c089_mask_choice": {
                        "bytes": len(c089_mask.data),
                        "params": c089_mask.params,
                        "sha256": _sha256_bytes(c089_mask.data),
                    },
                    "c089_pose_choice": {
                        "bytes": len(c089_pose.data),
                        "params": c089_pose.params,
                        "sha256": _sha256_bytes(c089_pose.data),
                    },
                },
            ),
            None,
        ),
        (
            CandidateSpec(
                candidate_id="pr77_actions_c089mask_renderer_pose_p3_isolation",
                payload=_build_p3_payload(
                    mask_br=c089_mask.data,
                    renderer_br=c089.encoded.renderer_br,
                    actions_br=pr77_actions,
                    pose_br=c089_pose.data,
                ),
                expected_decoded={
                    "masks.mkv": c089.decoded["masks.mkv"],
                    "optimized_poses.qp1": c089.decoded["optimized_poses.qp1"],
                    "renderer.bin": c089.decoded["renderer.bin"],
                    "seg_tile_actions.bin": pr77_actions_decoded,
                },
                selected_sources={
                    "actions": "pr77_action_delta_public",
                    "mask": "c089_pr75_qp1_top40_p6_lossless_resweep",
                    "pose": "c089_pr75_qp1_top40_p6_lossless_resweep",
                    "renderer": "c089_pr75_qp1_top40_p6",
                },
                semantic_contract="action_only_vs_c089_isolation_p3",
                next_dispatch_safety_gate=CUDA_AUTH_EVAL_REQUIRED,
                dispatchable_after_gate=True,
                notes=[
                    "Byte-regression isolation row: changes PR77 actions only versus C089 decoded renderer/mask/pose.",
                    "Useful only if component attribution is worth an exact eval; not the byte-priority candidate.",
                ],
                stream_packing={
                    "c089_mask_choice": {
                        "bytes": len(c089_mask.data),
                        "params": c089_mask.params,
                        "sha256": _sha256_bytes(c089_mask.data),
                    },
                    "c089_pose_choice": {
                        "bytes": len(c089_pose.data),
                        "params": c089_pose.params,
                        "sha256": _sha256_bytes(c089_pose.data),
                    },
                    "pr77_actions_encoded": {
                        "bytes": len(pr77_actions),
                        "sha256": _sha256_bytes(pr77_actions),
                    },
                },
            ),
            None,
        ),
        (
            CandidateSpec(
                candidate_id="c091_pr75_replay_noop_control",
                payload=c091.payload,
                expected_decoded=c091.decoded,
                selected_sources={
                    "actions": "c091_pr75_replay",
                    "mask": "c091_pr75_replay",
                    "pose": "c091_pr75_replay",
                    "renderer": "c091_pr75_replay",
                },
                semantic_contract="byte_identical_c091_noop_control",
                next_dispatch_safety_gate="do not dispatch: no-op control",
                dispatchable_after_gate=False,
                notes=["No-op control for deterministic ZIP/member handling and matrix sanity."],
                stream_packing={"source_payload_sha256": c091.payload_sha256},
            ),
            None,
        ),
        (
            CandidateSpec(
                candidate_id="pr77_replay_noop_control",
                payload=pr77.payload,
                expected_decoded=pr77.decoded,
                selected_sources={
                    "actions": "pr77_action_delta_public",
                    "mask": "pr77_action_delta_public",
                    "pose": "pr77_action_delta_public",
                    "renderer": "pr77_action_delta_public",
                },
                semantic_contract="byte_identical_pr77_noop_control",
                next_dispatch_safety_gate="do not dispatch: no-op control",
                dispatchable_after_gate=False,
                notes=["No-op control for deterministic ZIP/member handling and PR77 SHA custody."],
                stream_packing={"source_payload_sha256": pr77.payload_sha256},
            ),
            None,
        ),
    ]
    stream_report = {
        "c089_lossless_choices": {
            "mask": {
                "bytes": len(c089_mask.data),
                "delta_vs_source": len(c089_mask.data) - len(c089.encoded.mask_br),
                "params": c089_mask.params,
                "sha256": _sha256_bytes(c089_mask.data),
            },
            "pose": {
                "bytes": len(c089_pose.data),
                "delta_vs_source": len(c089_pose.data) - len(c089.encoded.pose_br),
                "params": c089_pose.params,
                "sha256": _sha256_bytes(c089_pose.data),
            },
        },
        "pr77_action_stream": {
            "decoded": _action_stats(pr77_actions_decoded),
            "encoded_bytes": len(pr77_actions),
            "encoded_sha256": _sha256_bytes(pr77_actions),
        },
        "sorted_pr77_action_probe": sorted_action_summary,
    }
    return candidates, stream_report


def _verify_anchor(label: str, source: SourceArchive, expected: Mapping[str, Any]) -> None:
    expected_sha = str(expected["archive_sha256"])
    expected_bytes = int(expected["archive_bytes"])
    if source.archive_sha256 != expected_sha:
        raise ValueError(f"{label}: expected SHA {expected_sha}, got {source.archive_sha256}")
    if source.archive_bytes != expected_bytes:
        raise ValueError(f"{label}: expected {expected_bytes} bytes, got {source.archive_bytes}")


def build_candidates(
    *,
    c091_archive: Path,
    c089_archive: Path,
    pr77_archive: Path,
    output_dir: Path,
    force: bool = False,
    unpacker: Any | None = None,
    verify_anchor_hashes: bool = True,
) -> dict[str, Any]:
    if unpacker is None:
        unpacker = _load_unpacker()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    c091 = _load_source("c091_pr75_replay", c091_archive, unpacker)
    c089 = _load_source("c089_pr75_qp1_top40_p6", c089_archive, unpacker)
    pr77 = _load_source("pr77_action_delta_public", pr77_archive, unpacker)
    sources = {
        "c091_pr75_replay": c091,
        "c089_pr75_qp1_top40_p6": c089,
        "pr77_action_delta_public": pr77,
    }
    if verify_anchor_hashes:
        _verify_anchor("c091_pr75_replay", c091, ANCHORS["c091_pr75_replay"])
        _verify_anchor("c089_pr75_qp1_top40_p6", c089, ANCHORS["c089_pr75_qp1_top40_p6"])
        _verify_anchor("pr77_action_delta_public", pr77, ANCHORS["pr77_action_delta_public"])

    specs, stream_report = _build_candidate_specs(c091=c091, c089=c089, pr77=pr77)
    rows = [
        _emit_candidate(
            spec=spec,
            output_dir=output_dir,
            sources=sources,
            selected_raw=selected_raw,
            unpacker=unpacker,
            force=force,
        )
        for spec, selected_raw in specs
    ]
    rows = sorted(rows, key=lambda row: (row["archive_bytes"], row["candidate_id"]))
    summary = {
        "anchors": ANCHORS,
        "candidate_count": len(rows),
        "candidates": rows,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "evidence_grade": "empirical_byte_screen_only",
        "promotion_eligible": False,
        "schema": SCHEMA,
        "score_claim": False,
        "source_archives": {label: _source_summary(source) for label, source in sources.items()},
        "stream_report": stream_report,
        "sub314_target": SUB314_TARGET,
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c091-archive", type=Path, default=DEFAULT_C091_ARCHIVE)
    parser.add_argument("--c089-archive", type=Path, default=DEFAULT_C089_ARCHIVE)
    parser.add_argument("--pr77-archive", type=Path, default=DEFAULT_PR77_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-anchor-hash-mismatch",
        action="store_true",
        help="Allow non-default source archives; manifests still record observed hashes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_candidates(
        c091_archive=args.c091_archive,
        c089_archive=args.c089_archive,
        pr77_archive=args.pr77_archive,
        output_dir=args.output_dir,
        force=bool(args.force),
        verify_anchor_hashes=not bool(args.allow_anchor_hash_mismatch),
    )
    dispatchable = [row for row in summary["candidates"] if row["dispatchable_after_gate"]]
    print(
        json.dumps(
            {
                "best_by_bytes": summary["candidates"][0] if summary["candidates"] else None,
                "best_dispatchable_by_bytes": dispatchable[0] if dispatchable else None,
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
