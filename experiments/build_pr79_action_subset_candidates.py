#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build local C102-native PR79 action-subset candidates.

This worker is byte-screening only. It keeps C102 mask, renderer, and pose
streams intact, mines PR79's decoded ``seg_tile_actions.bin`` records, writes
deterministic P6 action-stream archives, and records no-op, closure, and
break-even fields. It does not dispatch GPU jobs and does not make score
claims.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.archive_byte_profile import profile_archive
from tac.submission_archive import validate_seg_tile_actions_payload


UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
DEFAULT_C102_ARCHIVE = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/"
    "archive.zip"
)
DEFAULT_C102_EVAL = DEFAULT_C102_ARCHIVE.parent / "contest_auth_eval.json"
DEFAULT_C102_TRACE = DEFAULT_C102_ARCHIVE.parent / "component_trace.json"
DEFAULT_PR79_ARCHIVE = REPO_ROOT / (
    "experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip"
)
DEFAULT_PR79_PROFILE = DEFAULT_PR79_ARCHIVE.parent / "pr79_minp_grammar_profile.json"
DEFAULT_PRIOR_POLICY = (
    REPO_ROOT / "experiments/results/c102_native_action_atoms_20260503_worker/ranked_atom_policy.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/pr79_action_subset_worker_20260503"
)

TOOL = "experiments/build_pr79_action_subset_candidates.py"
SCHEMA = "pr79_action_subset_candidate_matrix_v1"
MANIFEST_SCHEMA = "pr79_action_subset_candidate_manifest_v1"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PAYLOAD_MEMBER = "p"
SEGMENT_NAMES = ("masks.mkv", "renderer.bin", "seg_tile_actions.bin", "optimized_poses.qp1")
NON_ACTION_SEGMENTS = ("masks.mkv", "renderer.bin", "optimized_poses.qp1")
RATE_DENOM = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / RATE_DENOM
TARGET_SCORE = 0.31  # [heuristic: aspirational floor below PR-79 S2 frontier 0.3145]
C102_SCORE = 0.31514430182167497  # [external: C102 ladder contest-CUDA T4]
C102_BYTES = 276_485
C102_SHA256 = "79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8"
PR79_SHA256 = "01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446"
CUDA_AUTH_EVAL_REQUIRED = (
    "No dispatch from this worker. To dispatch later, first claim the lane with "
    "tools/claim_lane_dispatch.py claim, then run exact CUDA auth eval on the "
    "identical archive bytes through archive.zip -> inflate.sh -> "
    "upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda."
)


class CandidateBuildError(ValueError):
    """Raised when a local candidate is not byte-closed or useful."""


@dataclass(frozen=True)
class ActionRecord:
    index: int
    pair_index: int
    tile_id: int
    action_id: int
    source_label: str = "pr79"
    source_index: int | None = None
    source_action_id: int | None = None
    prior: dict[str, Any] | None = None

    @property
    def key(self) -> tuple[int, int, int]:
        return (int(self.pair_index), int(self.tile_id), int(self.action_id))

    def encode4(self) -> bytes:
        return (
            int(self.pair_index).to_bytes(2, "little")
            + bytes([int(self.tile_id), int(self.action_id)])
        )


@dataclass(frozen=True)
class LoadedArchive:
    label: str
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    payload_format: str
    raw_segments: dict[str, bytes]
    decoded: dict[str, bytes]
    runtime_members: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class PriorIndex:
    path: Path | None
    loaded: bool
    atoms_by_key: dict[tuple[int, int, int], dict[str, Any]]
    ranked_atoms_count: int
    reason: str | None = None


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


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _safe_zip_member_name(name: str) -> str:
    path = Path(name)
    if (
        not name
        or name.startswith("/")
        or ".." in path.parts
        or len(path.parts) != 1
        or name.startswith(".")
        or name.startswith("__MACOSX/")
        or name.startswith("._")
    ):
        raise CandidateBuildError(f"unsafe ZIP member path: {name!r}")
    return name


def _read_single_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        seen: set[str] = set()
        names: list[str] = []
        for info in zf.infolist():
            name = _safe_zip_member_name(info.filename)
            if name in seen:
                raise CandidateBuildError(f"duplicate ZIP member: {name}")
            seen.add(name)
            if not info.is_dir():
                names.append(name)
        if names != [PAYLOAD_MEMBER]:
            raise CandidateBuildError(
                f"{path} must contain exactly member {PAYLOAD_MEMBER!r}; got {names!r}"
            )
        return zf.read(PAYLOAD_MEMBER)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_zip_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(PAYLOAD_MEMBER), payload)


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("pr79_action_subset_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _member_summary(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    members = header.get("members")
    if not isinstance(members, list):
        raise CandidateBuildError("runtime parser returned no member table")
    out: dict[str, dict[str, Any]] = {}
    for item in members:
        if not isinstance(item, Mapping):
            raise CandidateBuildError("runtime parser returned malformed member metadata")
        name = str(item.get("name"))
        out[name] = {
            "bytes": int(item["bytes"]),
            "codec": str(item["codec"]),
            "decoded_bytes": int(item["decoded_bytes"]),
            "decoded_sha256": str(item["decoded_sha256"]),
            "sha256": str(item["sha256"]),
        }
    return out


def _parse_self_describing_slices(payload: bytes) -> dict[str, bytes] | None:
    if payload.startswith(b"P3"):
        header_size = 2 + struct.calcsize("<IHH")
        if len(payload) <= header_size:
            raise CandidateBuildError("P3 payload is too short")
        mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        cursor = header_size
    elif payload.startswith(b"P6"):
        header_size = 2 + struct.calcsize("<IHHH")
        if len(payload) <= header_size:
            raise CandidateBuildError("P6 payload is too short")
        mask_len, renderer_len, actions_len, _record_count = struct.unpack_from(
            "<IHHH", payload, 2
        )
        cursor = header_size
    else:
        return None
    if min(mask_len, renderer_len, actions_len) <= 0:
        raise CandidateBuildError("self-describing action payload has an empty stream")
    mask_end = cursor + mask_len
    renderer_end = mask_end + renderer_len
    actions_end = renderer_end + actions_len
    if actions_end >= len(payload):
        raise CandidateBuildError("self-describing stream lengths leave no pose stream")
    return {
        "masks.mkv": payload[cursor:mask_end],
        "renderer.bin": payload[mask_end:renderer_end],
        "seg_tile_actions.bin": payload[renderer_end:actions_end],
        "optimized_poses.qp1": payload[actions_end:],
    }


def _slice_fixed_payload(
    *,
    label: str,
    payload: bytes,
    runtime_members: Mapping[str, Mapping[str, Any]],
) -> dict[str, bytes]:
    missing = sorted(set(SEGMENT_NAMES) - set(runtime_members))
    if missing:
        raise CandidateBuildError(f"{label}: missing runtime members {missing}")
    offset = 0
    raw_segments: dict[str, bytes] = {}
    for name in SEGMENT_NAMES:
        size = int(runtime_members[name]["bytes"])
        if size <= 0:
            raise CandidateBuildError(f"{label}: non-positive raw size for {name}: {size}")
        raw = payload[offset : offset + size]
        offset += size
        if len(raw) != size:
            raise CandidateBuildError(f"{label}: truncated fixed segment {name}")
        expected_sha = str(runtime_members[name].get("sha256", ""))
        if expected_sha and _sha256_bytes(raw) != expected_sha:
            raise CandidateBuildError(f"{label}: raw SHA mismatch for {name}")
        raw_segments[name] = raw
    if offset != len(payload):
        raise CandidateBuildError(
            f"{label}: fixed slices consume {offset}, payload has {len(payload)}"
        )
    return raw_segments


def load_archive(label: str, path: Path, *, unpacker: Any | None = None) -> LoadedArchive:
    if unpacker is None:
        unpacker = _load_unpacker()
    path = path.resolve()
    payload = _read_single_payload(path)
    header, decoded_raw = unpacker._parse_payload(payload)  # noqa: SLF001
    decoded = {str(name): bytes(data) for name, data in decoded_raw.items()}
    runtime_members = _member_summary(header)
    raw_segments = _parse_self_describing_slices(payload)
    if raw_segments is None:
        raw_segments = _slice_fixed_payload(
            label=label,
            payload=payload,
            runtime_members=runtime_members,
        )
    required = set(SEGMENT_NAMES)
    missing = sorted(required - set(decoded))
    if missing:
        raise CandidateBuildError(f"{label}: runtime parser missed members {missing}")
    if not decoded["renderer.bin"].startswith(b"QZS3"):
        raise CandidateBuildError(f"{label}: decoded renderer.bin is not QZS3")
    if not decoded["optimized_poses.qp1"].startswith(b"QP1"):
        raise CandidateBuildError(f"{label}: decoded optimized_poses.qp1 is not QP1")
    validate_seg_tile_actions_payload(
        decoded["seg_tile_actions.bin"],
        source_name=f"{label} seg_tile_actions.bin",
    )
    return LoadedArchive(
        label=label,
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=str(header.get("payload_format")),
        raw_segments=raw_segments,
        decoded=decoded,
        runtime_members=runtime_members,
    )


def _minimal_uvarint_length(value: int) -> int:
    if value < 0:
        raise CandidateBuildError(f"negative uvarint value: {value}")
    if value == 0:
        return 1
    return (value.bit_length() + 6) // 7


def _read_uvarint(data: bytes, cursor: int, *, max_value: int | None = None) -> tuple[int, int]:
    value = 0
    shift = 0
    start = cursor
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            consumed = cursor - start
            if consumed != _minimal_uvarint_length(value):
                raise CandidateBuildError(
                    f"noncanonical SG2 varint at byte {start}: value {value} used {consumed} bytes"
                )
            if max_value is not None and value > max_value:
                raise CandidateBuildError(
                    f"SG2 varint value {value} at byte {start} exceeds max {max_value}"
                )
            return value, cursor
        shift += 7
        if shift > 63:
            break
    raise CandidateBuildError(f"truncated or overlong SG2 varint at byte {start}")


def decode_sg2_action_wire(raw: bytes) -> list[ActionRecord]:
    """Decode PR79/PR75 SG2 grouped action wire bytes into runtime records."""

    if raw.startswith(b"SG2"):
        cursor = 3
    elif len(raw) % 4 != 0 and len(raw) % 5 != 0:
        cursor = 0
    else:
        raise CandidateBuildError(
            f"SG2 action wire must be tagged or length-ambiguous, got prefix {raw[:3]!r}"
        )
    records: list[ActionRecord] = []
    while cursor < len(raw):
        tile_id, cursor = _read_uvarint(raw, cursor, max_value=191)
        count, cursor = _read_uvarint(raw, cursor, max_value=10_000)
        if count <= 0:
            raise CandidateBuildError("SG2 group has zero action records")
        frame = 0
        for idx in range(count):
            delta, cursor = _read_uvarint(raw, cursor, max_value=9_999)
            frame = delta if idx == 0 else frame + delta
            if frame >= 10_000:
                raise CandidateBuildError(f"SG2 frame out of range: {frame}")
            if cursor >= len(raw):
                raise CandidateBuildError("SG2 action wire ended inside record")
            action_id = raw[cursor]
            cursor += 1
            if action_id >= 108:
                raise CandidateBuildError(f"SG2 action id outside PR75 dictionary: {action_id}")
            records.append(
                ActionRecord(
                    index=len(records),
                    pair_index=frame,
                    tile_id=tile_id,
                    action_id=action_id,
                    source_label="sg2",
                    source_index=len(records),
                    source_action_id=action_id,
                )
            )
    if not records:
        raise CandidateBuildError("SG2 action wire decoded no records")
    return records


def _parse_runtime_action_records(raw: bytes, *, source_label: str) -> list[ActionRecord]:
    validate_seg_tile_actions_payload(raw, source_name=f"{source_label} seg_tile_actions.bin")
    if len(raw) % 4:
        raise CandidateBuildError(f"{source_label}: expected runtime raw4 action records")
    records: list[ActionRecord] = []
    for offset in range(0, len(raw), 4):
        action_id = raw[offset + 3]
        if action_id >= 108:
            raise CandidateBuildError(f"{source_label}: action id outside PR75 dictionary: {action_id}")
        records.append(
            ActionRecord(
                index=offset // 4,
                pair_index=int.from_bytes(raw[offset : offset + 2], "little"),
                tile_id=raw[offset + 2],
                action_id=action_id,
                source_label=source_label,
                source_index=offset // 4,
                source_action_id=action_id,
            )
        )
    if not records:
        raise CandidateBuildError(f"{source_label}: decoded no action records")
    return records


def _encode_runtime_records(records: Sequence[ActionRecord]) -> bytes:
    return b"".join(record.encode4() for record in records)


def _uleb128(value: int) -> bytes:
    if value < 0:
        raise CandidateBuildError(f"cannot varint encode negative value {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _p6_order(records: Sequence[ActionRecord]) -> list[ActionRecord]:
    return sorted(
        records,
        key=lambda item: (
            int(item.pair_index),
            int(item.source_index if item.source_index is not None else item.index),
            int(item.tile_id),
            int(item.action_id),
        ),
    )


def _encode_p6_actions(records: Sequence[ActionRecord]) -> tuple[bytes, bytes, list[ActionRecord]]:
    ordered = _p6_order(records)
    packed = bytearray()
    previous_pair = 0
    for index, record in enumerate(ordered):
        pair_index = int(record.pair_index)
        tile_id = int(record.tile_id)
        action_id = int(record.action_id)
        if pair_index < 0 or pair_index >= 10_000:
            raise CandidateBuildError(f"P6 pair index out of range: {pair_index}")
        if tile_id < 0 or tile_id >= 192:
            raise CandidateBuildError(f"P6 tile id out of 384x512/32 grid bounds: {tile_id}")
        if action_id < 0 or action_id >= 108:
            raise CandidateBuildError(f"P6 action id outside PR75 dictionary: {action_id}")
        delta = pair_index if index == 0 else pair_index - previous_pair
        if delta < 0:
            raise CandidateBuildError("P6 encoding requires nondecreasing pair indices")
        packed.extend(_uleb128(delta))
        packed.append(tile_id)
        packed.append(action_id)
        previous_pair = pair_index
    return _encode_runtime_records(ordered), bytes(packed), ordered


def _build_p6_payload(
    base: LoadedArchive,
    records: Sequence[ActionRecord],
) -> tuple[bytes, bytes, bytes, list[ActionRecord]]:
    if not records:
        raise CandidateBuildError("policy selected no action records")
    runtime_raw, packed, ordered = _encode_p6_actions(records)
    actions_br = brotli.compress(packed, quality=11)
    payload = (
        b"P6"
        + struct.pack(
            "<IHHH",
            len(base.raw_segments["masks.mkv"]),
            len(base.raw_segments["renderer.bin"]),
            len(actions_br),
            len(records),
        )
        + base.raw_segments["masks.mkv"]
        + base.raw_segments["renderer.bin"]
        + actions_br
        + base.raw_segments["optimized_poses.qp1"]
    )
    return payload, runtime_raw, actions_br, ordered


def _records_stats(records: Sequence[ActionRecord]) -> dict[str, Any]:
    exact: dict[tuple[int, int, int], int] = {}
    pair_tile: dict[tuple[int, int], int] = {}
    pairs: dict[int, int] = {}
    for record in records:
        exact[record.key] = exact.get(record.key, 0) + 1
        pair_tile_key = (record.pair_index, record.tile_id)
        pair_tile[pair_tile_key] = pair_tile.get(pair_tile_key, 0) + 1
        pairs[record.pair_index] = pairs.get(record.pair_index, 0) + 1
    return {
        "duplicate_exact_record_count": sum(count - 1 for count in exact.values() if count > 1),
        "pair_tile_duplicate_group_count": sum(1 for count in pair_tile.values() if count > 1),
        "record_count": len(records),
        "unique_action_count": len({record.action_id for record in records}),
        "unique_pair_count": len(pairs),
        "unique_pair_tile_count": len(pair_tile),
        "unique_tile_count": len({record.tile_id for record in records}),
    }


def _action_guard(
    *,
    base_records: Sequence[ActionRecord],
    selected: Sequence[ActionRecord],
    base_raw: bytes,
    decoded_raw: bytes,
) -> dict[str, Any]:
    base_keys = {record.key for record in base_records}
    selected_keys = [record.key for record in selected]
    stats = _records_stats(selected)
    if stats["duplicate_exact_record_count"]:
        raise CandidateBuildError(
            f"candidate has {stats['duplicate_exact_record_count']} duplicate exact action records"
        )
    if decoded_raw == base_raw:
        raise CandidateBuildError("unchanged decoded action semantics")
    exact_c102_duplicates = sum(1 for key in selected_keys if key in base_keys)
    new_vs_c102 = len(selected_keys) - exact_c102_duplicates
    return {
        **stats,
        "decoded_action_sha256": _sha256_bytes(decoded_raw),
        "source_decoded_action_sha256": _sha256_bytes(base_raw),
        "exact_c102_duplicate_record_count": exact_c102_duplicates,
        "new_record_count_vs_c102": new_vs_c102,
        "no_op_status": "changes_c102_action_stream",
    }


def _validate_candidate_payload(
    *,
    base: LoadedArchive,
    base_records: Sequence[ActionRecord],
    payload: bytes,
    selected: Sequence[ActionRecord],
    expected_runtime_raw: bytes,
    unpacker: Any,
) -> dict[str, Any]:
    header, decoded_raw = unpacker._parse_payload(payload)  # noqa: SLF001
    payload_format = str(header.get("payload_format"))
    if payload_format != "public_pr75_qzs3_qp1_segactions_p6_delta_varint":
        raise CandidateBuildError(f"runtime parser returned invalid payload_format={payload_format!r}")
    decoded = {str(name): bytes(data) for name, data in decoded_raw.items()}
    for name in NON_ACTION_SEGMENTS:
        if decoded.get(name) != base.decoded[name]:
            raise CandidateBuildError(f"non-action stream changed unexpectedly: {name}")
    actions = decoded.get("seg_tile_actions.bin")
    if actions is None:
        raise CandidateBuildError("runtime parser did not decode seg_tile_actions.bin")
    if actions != expected_runtime_raw:
        raise CandidateBuildError("runtime parser decoded unexpected action bytes")
    action_validation = validate_seg_tile_actions_payload(
        actions,
        source_name="candidate seg_tile_actions.bin",
    )
    guard = _action_guard(
        base_records=base_records,
        selected=selected,
        base_raw=base.decoded["seg_tile_actions.bin"],
        decoded_raw=actions,
    )
    return {
        "action_semantic_guard": guard,
        "non_action_streams_preserved": True,
        "payload_format": payload_format,
        "runtime_members": _member_summary(header),
        "runtime_parser": _repo_rel(UNPACKER_PATH),
        "seg_tile_actions_validation": action_validation,
    }


def _load_prior_index(path: Path | None) -> PriorIndex:
    if path is None:
        return PriorIndex(path=None, loaded=False, atoms_by_key={}, ranked_atoms_count=0, reason="disabled")
    if not path.exists():
        return PriorIndex(path=path, loaded=False, atoms_by_key={}, ranked_atoms_count=0, reason="missing")
    payload = json.loads(path.read_text())
    atoms = payload.get("ranked_atoms")
    if not isinstance(atoms, list):
        return PriorIndex(path=path, loaded=False, atoms_by_key={}, ranked_atoms_count=0, reason="no ranked_atoms")
    atoms_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    for rank, atom in enumerate(atoms, start=1):
        try:
            key = (int(atom["pair_index"]), int(atom["tile_id"]), int(atom["action_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        row = dict(atom)
        row["prior_rank"] = rank
        atoms_by_key.setdefault(key, row)
    return PriorIndex(
        path=path,
        loaded=True,
        atoms_by_key=atoms_by_key,
        ranked_atoms_count=len(atoms),
    )


def _load_anchor_samples(path: Path | None) -> dict[int, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    samples: dict[int, dict[str, float]] = {}
    for item in payload.get("samples", []):
        pair = int(item["pair_index"])
        samples[pair] = {
            "combined": float(item["score_combined_contribution_first_order"]),
            "pose": float(item["score_pose_contribution_first_order"]),
            "seg": float(item["score_seg_contribution_exact"]),
        }
    return samples


def _load_anchor_eval(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "archive_bytes": C102_BYTES,
            "archive_sha256": C102_SHA256,
            "path": _repo_rel(path) if path else None,
            "score": C102_SCORE,
            "status": "fallback_constants",
        }
    payload = json.loads(path.read_text())
    archive_sha = None
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        archive_sha = provenance.get("archive_sha256")
    return {
        "archive_bytes": int(payload.get("archive_size_bytes", C102_BYTES)),
        "archive_sha256": str(archive_sha or C102_SHA256),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "path": _repo_rel(path),
        "score": float(payload.get("score_recomputed_from_components", C102_SCORE)),
        "status": "loaded",
    }


def _annotate_pr79_records(
    records: Sequence[ActionRecord],
    *,
    priors: PriorIndex,
    anchor_samples: Mapping[int, Mapping[str, float]],
) -> list[ActionRecord]:
    annotated: list[ActionRecord] = []
    for record in records:
        prior = dict(priors.atoms_by_key.get(record.key, {}))
        sample = anchor_samples.get(record.pair_index)
        if sample:
            prior["anchor_pair_combined_contribution"] = float(sample["combined"])
            prior["anchor_pair_pose_contribution"] = float(sample["pose"])
            prior["anchor_pair_seg_contribution"] = float(sample["seg"])
        prior.setdefault("mean_weighted_combined_equal_share", 0.0)
        prior.setdefault("mean_weighted_pose_equal_share", 0.0)
        prior.setdefault("classification", "no_exact_prior")
        annotated.append(
            ActionRecord(
                index=record.index,
                pair_index=record.pair_index,
                tile_id=record.tile_id,
                action_id=record.action_id,
                source_label="pr79",
                source_index=record.source_index,
                source_action_id=record.source_action_id,
                prior=prior,
            )
        )
    return annotated


def _prior_value(record: ActionRecord, field: str, default: float = 0.0) -> float:
    if not record.prior:
        return default
    try:
        return float(record.prior.get(field, default))
    except (TypeError, ValueError):
        return default


def _record_rank_key(record: ActionRecord) -> tuple[float, float, float, int, int, int]:
    return (
        _prior_value(record, "mean_weighted_combined_equal_share"),
        _prior_value(record, "mean_weighted_pose_equal_share"),
        _prior_value(record, "anchor_pair_combined_contribution"),
        -int(record.source_index if record.source_index is not None else record.index),
        -int(record.tile_id),
        -int(record.action_id),
    )


def _dedupe_records(records: Iterable[ActionRecord]) -> list[ActionRecord]:
    out: list[ActionRecord] = []
    seen: set[tuple[int, int, int]] = set()
    for record in records:
        if record.key in seen:
            continue
        seen.add(record.key)
        out.append(record)
    return out


def _records_with_positive_exact_prior(records: Sequence[ActionRecord]) -> list[ActionRecord]:
    allowed = {
        "component_positive_pose_safe",
        "component_positive_pose_neutral",
        "component_positive_pose_risky",
    }
    return [
        record
        for record in sorted(records, key=_record_rank_key, reverse=True)
        if (record.prior or {}).get("classification") in allowed
        and _prior_value(record, "mean_weighted_combined_equal_share") > 0.0
    ]


def _records_with_pose_safe_prior(records: Sequence[ActionRecord]) -> list[ActionRecord]:
    return [
        record
        for record in sorted(records, key=_record_rank_key, reverse=True)
        if (record.prior or {}).get("classification") == "component_positive_pose_safe"
        and _prior_value(record, "mean_weighted_combined_equal_share") > 0.0
    ]


def _records_by_pair_opportunity(records: Sequence[ActionRecord]) -> list[ActionRecord]:
    return sorted(
        records,
        key=lambda record: (
            _prior_value(record, "anchor_pair_combined_contribution"),
            _prior_value(record, "anchor_pair_pose_contribution"),
            -int(record.source_index if record.source_index is not None else record.index),
        ),
        reverse=True,
    )


def _select_policy_records(
    policy: str,
    *,
    base_records: Sequence[ActionRecord],
    pr79_records: Sequence[ActionRecord],
) -> tuple[list[ActionRecord], dict[str, Any]]:
    mode = "replace"
    base_prefix: list[ActionRecord] = []
    raw_policy = policy
    if policy.startswith("augment_c102_"):
        mode = "augment_c102"
        policy = policy.removeprefix("augment_c102_")
        base_prefix = list(base_records)
    elif policy.startswith("replace_"):
        policy = policy.removeprefix("replace_")

    base_keys = {record.key for record in base_records}
    pool = list(pr79_records)
    if mode == "augment_c102":
        pool = [record for record in pool if record.key not in base_keys]

    selected_suffix: list[ActionRecord]
    rationale: str
    match = re.fullmatch(r"pr79_first(\d+)_p6", policy)
    if match:
        selected_suffix = list(pool[: int(match.group(1))])
        rationale = "source-order PR79 smoke subset; useful for deterministic unit and parser closure checks."
    else:
        match = re.fullmatch(r"pr79_all_p6", policy)
        if match:
            selected_suffix = list(pool)
            rationale = "All PR79 decoded action records repacked as P6 on C102 non-action streams."
        else:
            match = re.fullmatch(r"pr79_exact_positive_top(\d+)_p6", policy)
            if match:
                selected_suffix = _records_with_positive_exact_prior(pool)[: int(match.group(1))]
                rationale = "PR79 records with exact positive C102 atom-prior matches, sorted by expected component proxy."
            else:
                match = re.fullmatch(r"pr79_pose_safe_top(\d+)_p6", policy)
                if match:
                    selected_suffix = _records_with_pose_safe_prior(pool)[: int(match.group(1))]
                    rationale = "PR79 records whose exact C102 atom-prior match is pose-safe positive."
                else:
                    match = re.fullmatch(r"pr79_pair_opportunity_top(\d+)_p6", policy)
                    if match:
                        selected_suffix = _records_by_pair_opportunity(pool)[: int(match.group(1))]
                        rationale = (
                            "Fallback PR79 records ranked by C102 component-trace pair burden; "
                            "no exact response prior is implied."
                        )
                    else:
                        raise CandidateBuildError(f"unsupported policy: {raw_policy}")

    selected = _dedupe_records([*base_prefix, *selected_suffix])
    return selected, {
        "mode": mode,
        "policy": raw_policy,
        "pr79_selected_before_dedupe": len(selected_suffix),
        "rationale": rationale,
    }


def _source_summary(source: LoadedArchive) -> dict[str, Any]:
    return {
        "archive_bytes": source.archive_bytes,
        "archive_path": _repo_rel(source.path),
        "archive_sha256": source.archive_sha256,
        "decoded_segments": {
            name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
            for name, data in sorted(source.decoded.items())
            if name in SEGMENT_NAMES
        },
        "payload_bytes": len(source.payload),
        "payload_format": source.payload_format,
        "payload_sha256": source.payload_sha256,
        "raw_segments": {
            name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
            for name, data in sorted(source.raw_segments.items())
            if name in SEGMENT_NAMES
        },
    }


def _break_even_screen(
    *,
    archive_bytes: int,
    base_archive_bytes: int,
    base_score: float,
    target_score: float = TARGET_SCORE,
) -> dict[str, Any]:
    delta_bytes = int(archive_bytes) - int(base_archive_bytes)
    rate_delta = delta_bytes * RATE_SCORE_PER_BYTE
    score_if_components_unchanged = float(base_score) + rate_delta
    required = max(0.0, score_if_components_unchanged - float(target_score))
    return {
        "archive_delta_bytes_vs_c102": delta_bytes,
        "rate_score_delta_vs_c102": rate_delta,
        "score_if_components_unchanged": score_if_components_unchanged,
        "target_component_score_improvement_needed": required,
        "target_equivalent_bytes_needed_after_candidate": (
            math.ceil(required / RATE_SCORE_PER_BYTE) if required > 0.0 else 0
        ),
        "target_score": float(target_score),
    }


def _expected_component_proxy(records: Sequence[ActionRecord]) -> dict[str, Any]:
    pr79_records = [record for record in records if record.source_label == "pr79"]
    positive_exact = [
        record
        for record in pr79_records
        if _prior_value(record, "mean_weighted_combined_equal_share") > 0.0
        and (record.prior or {}).get("classification") != "no_exact_prior"
    ]
    return {
        "exact_positive_prior_record_count": len(positive_exact),
        "sum_exact_positive_component_proxy": sum(
            max(0.0, _prior_value(record, "mean_weighted_combined_equal_share"))
            for record in positive_exact
        ),
        "sum_exact_pose_proxy": sum(
            _prior_value(record, "mean_weighted_pose_equal_share")
            for record in positive_exact
        ),
        "records_without_exact_prior_count": sum(
            1
            for record in pr79_records
            if (record.prior or {}).get("classification") == "no_exact_prior"
        ),
    }


def _selected_record_manifest(record: ActionRecord) -> dict[str, Any]:
    prior = record.prior or {}
    return {
        "action_id": int(record.action_id),
        "classification": prior.get("classification"),
        "expected_component_benefit_proxy": prior.get("mean_weighted_combined_equal_share", 0.0),
        "expected_pose_benefit_proxy": prior.get("mean_weighted_pose_equal_share", 0.0),
        "pair_index": int(record.pair_index),
        "prior_rank": prior.get("prior_rank"),
        "source_action_id": (
            int(record.source_action_id)
            if record.source_action_id is not None
            else int(record.action_id)
        ),
        "source_index": int(record.source_index if record.source_index is not None else record.index),
        "source_label": record.source_label,
        "tile_id": int(record.tile_id),
    }


def _dispatch_criteria(candidate_id: str, *, plausible: bool) -> dict[str, Any]:
    blockers = [
        "no GPU dispatch was requested or performed by this worker",
        "lane dispatch claim is required before exact eval",
        "exact CUDA auth eval on identical archive bytes is required for any score claim",
    ]
    if not plausible:
        blockers.append("local proxy/byte screen does not clear the <=0.31 break-even")
    return {
        "candidate_id": candidate_id,
        "dispatch_ready_now": False,
        "dispatch_blockers": blockers,
        "minimum_before_dispatch": [
            "candidate manifest runtime_parse_validation.non_action_streams_preserved must be true",
            "candidate manifest action_semantic_guard.no_op_status must be changes_c102_action_stream",
            "candidate archive bytes and SHA-256 must match the manifest",
            "operator must claim a non-conflicting lane with tools/claim_lane_dispatch.py claim",
            "exact eval command must pass --expected-archive-sha256 and --expected-archive-size-bytes",
        ],
        "promotion_criteria_after_dispatch": [
            "contest_auth_eval.json reports device cuda and n_samples 600",
            "archive SHA-256 and size match this manifest",
            "score_recomputed_from_components <= 0.31",
            "PoseNet and SegNet component gates do not collapse relative to C102",
            "runtime tree hash and payload closure are recorded",
        ],
    }


def _build_one_candidate(
    *,
    base: LoadedArchive,
    pr79: LoadedArchive,
    base_records: Sequence[ActionRecord],
    pr79_records: Sequence[ActionRecord],
    policy: str,
    output_dir: Path,
    anchor_eval: Mapping[str, Any],
    unpacker: Any,
    force: bool,
) -> dict[str, Any]:
    selected, policy_info = _select_policy_records(
        policy,
        base_records=base_records,
        pr79_records=pr79_records,
    )
    payload, runtime_raw, actions_br, ordered = _build_p6_payload(base, selected)
    validation = _validate_candidate_payload(
        base=base,
        base_records=base_records,
        payload=payload,
        selected=ordered,
        expected_runtime_raw=runtime_raw,
        unpacker=unpacker,
    )
    candidate_id = policy.replace("_p6", "") + "_on_c102_p6"
    candidate_dir = output_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    if archive_path.exists() and not force:
        raise FileExistsError(f"{archive_path} exists; pass --force")
    _write_archive(archive_path, payload)
    if _read_single_payload(archive_path) != payload:
        raise CandidateBuildError(f"{candidate_id}: archive readback payload mismatch")
    archive_sha = _sha256_file(archive_path)
    archive_bytes = archive_path.stat().st_size
    profile = profile_archive(archive_path)
    expected = _expected_component_proxy(ordered)
    break_even = _break_even_screen(
        archive_bytes=archive_bytes,
        base_archive_bytes=int(anchor_eval.get("archive_bytes", base.archive_bytes)),
        base_score=float(anchor_eval.get("score", C102_SCORE)),
        target_score=TARGET_SCORE,
    )
    expected_sum = float(expected["sum_exact_positive_component_proxy"])
    required = float(break_even["target_component_score_improvement_needed"])
    plausible = expected_sum >= required and expected_sum > 0.0
    readiness = {
        "evidence_grade": "empirical_byte_screen_only",
        "expected_proxy_minus_required": expected_sum - required,
        "plausible_for_le_0_31": plausible,
        "promotion_eligible": False,
        "score_claim": False,
    }
    manifest = {
        "archive_byte_profile": profile,
        "candidate_id": candidate_id,
        "cuda_auth_eval_required": CUDA_AUTH_EVAL_REQUIRED,
        "dispatch_criteria": _dispatch_criteria(candidate_id, plausible=plausible),
        "evidence_grade": "empirical_byte_screen_only",
        "expected_component_proxy": expected,
        "output_archive": {
            "bytes": archive_bytes,
            "path": str(archive_path),
            "repo_relative_path": _repo_rel(archive_path),
            "sha256": archive_sha,
        },
        "payload": {
            "bytes": len(payload),
            "format": "public_pr75_qzs3_qp1_segactions_p6_delta_varint",
            "member": PAYLOAD_MEMBER,
            "sha256": _sha256_bytes(payload),
        },
        "policy": policy_info,
        "readiness": readiness,
        "remote_dispatch_performed": False,
        "runtime_parse_validation": validation,
        "schema": MANIFEST_SCHEMA,
        "score_claim": False,
        "selected_records": [_selected_record_manifest(record) for record in ordered],
        "source_archives": {
            "c102": _source_summary(base),
            "pr79": _source_summary(pr79),
        },
        "stream_delta": {
            "actions_brotli_bytes": len(actions_br),
            "actions_brotli_sha256": _sha256_bytes(actions_br),
            "archive_delta_bytes_vs_c102": archive_bytes - base.archive_bytes,
            "payload_delta_bytes_vs_c102": len(payload) - len(base.payload),
            "runtime_action_raw_bytes": len(runtime_raw),
            "runtime_action_raw_sha256": _sha256_bytes(runtime_raw),
        },
        "target_screen": break_even,
        "tool": TOOL,
        "wire_format": "p6",
    }
    manifest_path = candidate_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "archive_bytes": archive_bytes,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha,
        "candidate_id": candidate_id,
        "delta_bytes_vs_c102": archive_bytes - base.archive_bytes,
        "dispatch_ready_now": False,
        "expected_component_benefit_proxy": expected_sum,
        "expected_proxy_minus_required": expected_sum - required,
        "manifest_path": str(manifest_path),
        "plausible_for_le_0_31": plausible,
        "policy": policy,
        "records_selected_count": len(ordered),
        "score_claim": False,
        "target_component_score_improvement_needed": required,
    }


def _default_policies() -> list[str]:
    return [
        "augment_c102_pr79_pose_safe_top16_p6",
        "augment_c102_pr79_exact_positive_top16_p6",
        "replace_pr79_pose_safe_top16_p6",
        "replace_pr79_exact_positive_top16_p6",
        "replace_pr79_exact_positive_top32_p6",
        "replace_pr79_pair_opportunity_top32_p6",
        "replace_pr79_pair_opportunity_top64_p6",
        "replace_pr79_all_p6",
    ]


def _pr79_sg2_decode_summary(pr79: LoadedArchive) -> dict[str, Any]:
    action_br = pr79.raw_segments["seg_tile_actions.bin"]
    try:
        raw_wire = brotli.decompress(action_br)
    except brotli.error as exc:
        return {
            "charged_brotli_bytes": len(action_br),
            "decoded_by_worker": False,
            "error": str(exc),
        }
    out: dict[str, Any] = {
        "charged_brotli_bytes": len(action_br),
        "charged_brotli_sha256": _sha256_bytes(action_br),
        "wire_raw_bytes": len(raw_wire),
        "wire_raw_sha256": _sha256_bytes(raw_wire),
        "wire_starts_sg2": raw_wire.startswith(b"SG2"),
    }
    if raw_wire.startswith(b"SG2") or (len(raw_wire) % 4 != 0 and len(raw_wire) % 5 != 0):
        records = decode_sg2_action_wire(raw_wire)
        runtime = _encode_runtime_records(records)
        out.update(
            {
                "decoded_by_worker": True,
                "record_count": len(records),
                "runtime_record_bytes": len(runtime),
                "runtime_record_sha256": _sha256_bytes(runtime),
                "runtime_matches_robust_unpacker": runtime == pr79.decoded["seg_tile_actions.bin"],
            }
        )
    else:
        out["decoded_by_worker"] = False
    return out


def build_candidates(
    *,
    c102_archive: Path = DEFAULT_C102_ARCHIVE,
    c102_eval: Path = DEFAULT_C102_EVAL,
    c102_trace: Path = DEFAULT_C102_TRACE,
    pr79_archive: Path = DEFAULT_PR79_ARCHIVE,
    pr79_profile: Path = DEFAULT_PR79_PROFILE,
    prior_policy: Path | None = DEFAULT_PRIOR_POLICY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    policies: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    unpacker = _load_unpacker()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_archive("c102_top192", c102_archive, unpacker=unpacker)
    pr79 = load_archive("pr79_public", pr79_archive, unpacker=unpacker)
    if c102_archive == DEFAULT_C102_ARCHIVE and base.archive_sha256 != C102_SHA256:
        raise CandidateBuildError(
            f"default C102 archive SHA mismatch: expected {C102_SHA256}, got {base.archive_sha256}"
        )
    if pr79_archive == DEFAULT_PR79_ARCHIVE and pr79.archive_sha256 != PR79_SHA256:
        raise CandidateBuildError(
            f"default PR79 archive SHA mismatch: expected {PR79_SHA256}, got {pr79.archive_sha256}"
        )
    base_records = _parse_runtime_action_records(
        base.decoded["seg_tile_actions.bin"],
        source_label="c102",
    )
    raw_pr79_records = _parse_runtime_action_records(
        pr79.decoded["seg_tile_actions.bin"],
        source_label="pr79",
    )
    priors = _load_prior_index(prior_policy)
    anchor_samples = _load_anchor_samples(c102_trace)
    pr79_records = _annotate_pr79_records(
        raw_pr79_records,
        priors=priors,
        anchor_samples=anchor_samples,
    )
    anchor_eval = _load_anchor_eval(c102_eval)
    selected_policies = list(policies) if policies else _default_policies()
    candidates = []
    skipped = []
    for policy in selected_policies:
        try:
            candidates.append(
                _build_one_candidate(
                    base=base,
                    pr79=pr79,
                    base_records=base_records,
                    pr79_records=pr79_records,
                    policy=policy,
                    output_dir=output_dir,
                    anchor_eval=anchor_eval,
                    unpacker=unpacker,
                    force=force,
                )
            )
        except Exception as exc:
            skipped.append(
                {
                    "policy": policy,
                    "reason": str(exc),
                    "status": "skipped_or_failed_local_build",
                }
            )
    exact_prior_matches = [
        record
        for record in pr79_records
        if (record.prior or {}).get("classification") != "no_exact_prior"
    ]
    summary = {
        "anchor_eval": anchor_eval,
        "candidates": sorted(
            candidates,
            key=lambda row: (
                not bool(row["plausible_for_le_0_31"]),
                -float(row["expected_proxy_minus_required"]),
                int(row["archive_bytes"]),
                str(row["candidate_id"]),
            ),
        ),
        "cuda_auth_eval_required": CUDA_AUTH_EVAL_REQUIRED,
        "dispatch_decision": {
            "exact_eval_justified": any(row["plausible_for_le_0_31"] for row in candidates),
            "no_remote_dispatch_performed": True,
            "reason": (
                "local byte/proxy matrix only; dispatch requires a lane claim and "
                "an exact CUDA eval candidate whose byte-closed proxy clears the <=0.31 break-even"
            ),
        },
        "evidence_grade": "empirical_byte_screen_only",
        "policy_count": len(selected_policies),
        "pr79_action_stream": {
            **_records_stats(pr79_records),
            "decoded_runtime_sha256": _sha256_bytes(pr79.decoded["seg_tile_actions.bin"]),
            "exact_prior_match_count": len(exact_prior_matches),
            "pose_safe_exact_prior_match_count": sum(
                1
                for record in exact_prior_matches
                if (record.prior or {}).get("classification") == "component_positive_pose_safe"
            ),
            "sg2_decode": _pr79_sg2_decode_summary(pr79),
        },
        "prior_policy": {
            "loaded": priors.loaded,
            "path": _repo_rel(priors.path),
            "ranked_atoms_count": priors.ranked_atoms_count,
            "reason": priors.reason,
        },
        "profile_inputs": {
            "c102_trace": _repo_rel(c102_trace),
            "pr79_profile": _repo_rel(pr79_profile) if pr79_profile.exists() else None,
        },
        "remote_dispatch_performed": False,
        "schema": SCHEMA,
        "score_claim": False,
        "skipped_policies": skipped,
        "source_archives": {
            "c102": _source_summary(base),
            "pr79": _source_summary(pr79),
        },
        "target_score": TARGET_SCORE,
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c102-archive", type=Path, default=DEFAULT_C102_ARCHIVE)
    parser.add_argument("--c102-eval", type=Path, default=DEFAULT_C102_EVAL)
    parser.add_argument("--c102-trace", type=Path, default=DEFAULT_C102_TRACE)
    parser.add_argument("--pr79-archive", type=Path, default=DEFAULT_PR79_ARCHIVE)
    parser.add_argument("--pr79-profile", type=Path, default=DEFAULT_PR79_PROFILE)
    parser.add_argument("--prior-policy", type=Path, default=DEFAULT_PRIOR_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_candidates(
        c102_archive=args.c102_archive,
        c102_eval=args.c102_eval,
        c102_trace=args.c102_trace,
        pr79_archive=args.pr79_archive,
        pr79_profile=args.pr79_profile,
        prior_policy=args.prior_policy,
        output_dir=args.output_dir,
        policies=args.policy or None,
        force=bool(args.force),
    )
    print(
        json.dumps(
            {
                "best_candidates": summary["candidates"][:5],
                "candidate_count": len(summary["candidates"]),
                "exact_eval_justified": summary["dispatch_decision"]["exact_eval_justified"],
                "output_dir": str(Path(args.output_dir).resolve()),
                "score_claim": False,
                "skipped_count": len(summary["skipped_policies"]),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
