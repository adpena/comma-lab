"""PR82/Henosis atom-transfer helpers.

The helpers in this module are intentionally scorer-free.  They parse the
public PR82 compact bundle, expose deterministic per-pair activity summaries,
and build runtime-compatible ``QPS1`` postprocess sidecars for local archive
screening.
"""
from __future__ import annotations

import ast
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import brotli
import numpy as np


QPOST_MAGIC = b"QPS1"
QPOST_STREAM_NAMES = (
    "post",
    "shift",
    "frac",
    "frac2",
    "frac3",
    "bias",
    "region",
    "randmulti",
)
PR82_HEADER_STREAM_NAMES = (
    "mask",
    "model",
    "pose",
    "post",
    "shift",
    "frac",
    "frac2",
    "frac3",
)
QPOST_DEFAULTS = {
    "post": 0,
    "shift": 40,
    "frac": 4,
    "frac2": 4,
    "frac3": 4,
    "bias": 13,
    "region": 0,
    "randmulti": 0,
}


class HenosisPr82TransferError(ValueError):
    """Raised when PR82 atom parsing or transfer fails a closed guard."""


@dataclass(frozen=True)
class Pr82ReplayContract:
    """Constants recovered from the public replay inflate source."""

    fixed_bias_bytes: int
    fixed_region_bytes: int
    randmulti_specs: tuple[tuple[int, int, int, int], ...]
    source_path: str | None = None
    source_sha256: str | None = None


@dataclass(frozen=True)
class Pr82Bundle:
    """Encoded PR82 compact bundle segments."""

    payload_bytes: int
    header_lengths: dict[str, int]
    encoded_segments: dict[str, bytes]


@dataclass(frozen=True)
class Pr82RandmultiGroup:
    """Decoded headerless PR82 randmulti group."""

    group_index: int
    height: int
    width: int
    amplitude: int
    scount: int
    rows: np.ndarray
    payload_bytes: int


PR82_RANDMULTI_SPECIAL_SEMANTICS: dict[tuple[int, int, int], str] = {
    (224, 222, 4): "replay_special_f2_tile_bias_2x2_channel_radius4",
    (223, 223, 4): "replay_special_f2_boundary_all_channel_radius4",
    (223, 222, 2): "replay_special_f2_class_conditioned_channel_radius2",
    (223, 224, 4): "replay_special_f2_boundary_class_channel_radius4",
    (223, 219, 4): "replay_special_f2_width2_boundary_channel_class_radius4",
    (223, 218, 4): "replay_special_f2_width3_boundary_channel_class_radius4",
    (223, 221, 4): "replay_special_f2_class_conditioned_channel_radius4",
    (222, 223, 4): "replay_special_f2_class_conditioned_all_channel_radius4",
    (222, 222, 4): "replay_special_f2_global_rgb_bias_radius4",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_replay_contract(path: Path | None) -> Pr82ReplayContract:
    """Parse fixed tail lengths and randmulti specs from replay ``inflate.py``."""

    if path is None:
        return Pr82ReplayContract(223, 273, ())
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    fixed_bias: int | None = None
    fixed_region: int | None = None
    specs: tuple[tuple[int, int, int, int], ...] = ()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in {"l_bias", "l_region"} and isinstance(node.value, ast.Constant):
            value = int(node.value.value)
            if target.id == "l_bias":
                fixed_bias = value
            else:
                fixed_region = value
        if target.id == "specs_n":
            try:
                raw_specs = ast.literal_eval(node.value)
            except (ValueError, SyntaxError) as exc:
                raise HenosisPr82TransferError(f"cannot parse specs_n from {path}") from exc
            specs = tuple(tuple(int(value) for value in row) for row in raw_specs)
    return Pr82ReplayContract(
        fixed_bias_bytes=223 if fixed_bias is None else fixed_bias,
        fixed_region_bytes=273 if fixed_region is None else fixed_region,
        randmulti_specs=specs,
        source_path=str(path),
        source_sha256=sha256_path(path),
    )


def parse_pr82_bundle(raw: bytes, contract: Pr82ReplayContract) -> Pr82Bundle:
    """Split a PR82 compact ``x`` payload into encoded stream bytes."""

    if len(raw) < 24:
        raise HenosisPr82TransferError("PR82 payload is too short for 8x u24 header")
    lengths = {
        name: int.from_bytes(raw[index * 3 : index * 3 + 3], "little")
        for index, name in enumerate(PR82_HEADER_STREAM_NAMES)
    }
    pos = 24
    segments: dict[str, bytes] = {}
    for name in PR82_HEADER_STREAM_NAMES:
        n_bytes = lengths[name]
        end = pos + n_bytes
        if end > len(raw):
            raise HenosisPr82TransferError(f"PR82 segment {name!r} overruns payload")
        segments[name] = raw[pos:end]
        pos = end
    for name, n_bytes in (
        ("bias", contract.fixed_bias_bytes),
        ("region", contract.fixed_region_bytes),
    ):
        end = pos + int(n_bytes)
        if end > len(raw):
            raise HenosisPr82TransferError(f"PR82 fixed segment {name!r} overruns payload")
        segments[name] = raw[pos:end]
        pos = end
    if pos >= len(raw):
        raise HenosisPr82TransferError("PR82 payload is missing randmulti tail")
    segments["randmulti"] = raw[pos:]
    return Pr82Bundle(payload_bytes=len(raw), header_lengths=lengths, encoded_segments=segments)


def brotli_decompress_segment(encoded: bytes, name: str) -> bytes:
    try:
        return brotli.decompress(encoded)
    except brotli.error as exc:
        raise HenosisPr82TransferError(f"PR82 segment {name!r} is not Brotli-decodable") from exc


def _read_vlq(data: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            break
    raise HenosisPr82TransferError("truncated or overlong VLQ stream")


def _vlq_indices_values(raw: bytes, cursor: int, count: int) -> tuple[list[int], np.ndarray, int]:
    idx = -1
    indices: list[int] = []
    for _ in range(count):
        delta, cursor = _read_vlq(raw, cursor)
        idx += delta + 1
        if idx < 0 or idx >= 600:
            raise HenosisPr82TransferError(f"sparse qpost index out of range: {idx}")
        indices.append(idx)
    end = cursor + count
    if end > len(raw):
        raise HenosisPr82TransferError("sparse qpost value stream is truncated")
    values = np.frombuffer(raw, dtype=np.uint8, count=count, offset=cursor).astype(np.uint8)
    return indices, values, end


def randmulti_semantic_label(height: int, width: int, amplitude: int) -> str:
    return PR82_RANDMULTI_SPECIAL_SEMANTICS.get(
        (int(height), int(width), int(amplitude)),
        "generic_frame0_nearest_random_pattern",
    )


def randmulti_group_qps1_nm2_compatible(group: Pr82RandmultiGroup) -> bool:
    """Whether current QPS1 ``NM2`` can represent and replay this group exactly."""

    if randmulti_semantic_label(group.height, group.width, group.amplitude) != "generic_frame0_nearest_random_pattern":
        return False
    return all(
        0 <= int(value) <= 255
        for value in (group.height, group.width, group.amplitude, group.scount)
    )


def _decode_randmulti_rows(raw: bytes, cursor: int, scount: int) -> tuple[np.ndarray, int]:
    rows = np.zeros((int(scount), 600), dtype=np.uint8)
    for row_index in range(int(scount)):
        if cursor >= len(raw):
            raise HenosisPr82TransferError("randmulti stream ended before count byte")
        count = int(raw[cursor])
        cursor += 1
        if count == 255:
            if cursor + 2 > len(raw):
                raise HenosisPr82TransferError("randmulti extended count is truncated")
            count = int.from_bytes(raw[cursor : cursor + 2], "little")
            cursor += 2
        indices, values, cursor = _vlq_indices_values(raw, cursor, count)
        if count:
            rows[row_index, np.asarray(indices, dtype=np.int64)] = values
    return rows, cursor


def _write_vlq(value: int) -> bytes:
    if value < 0:
        raise HenosisPr82TransferError(f"cannot VLQ-encode negative value: {value}")
    out = bytearray()
    while True:
        byte = int(value) & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _encode_randmulti_rows(rows: np.ndarray) -> bytes:
    if rows.ndim != 2 or rows.shape[1] != 600:
        raise HenosisPr82TransferError(f"randmulti rows must have shape (scount, 600), got {rows.shape}")
    out = bytearray()
    for row in rows.astype(np.uint8, copy=False):
        indices = np.flatnonzero(row)
        count = int(indices.size)
        if count > 0xFFFF:
            raise HenosisPr82TransferError(f"randmulti sparse row has too many choices: {count}")
        if count >= 255:
            out.append(255)
            out.extend(count.to_bytes(2, "little"))
        else:
            out.append(count)
        previous = -1
        for index in indices:
            out.extend(_write_vlq(int(index) - previous - 1))
            previous = int(index)
        out.extend(row[indices].astype(np.uint8, copy=False).tobytes())
    return bytes(out)


def decode_randmulti_groups(
    encoded: bytes,
    specs: Sequence[Sequence[int]],
) -> tuple[Pr82RandmultiGroup, ...]:
    """Decode PR82 headerless sparse randmulti rows.

    PR82's replay runtime has a hard-coded 72-group spec table.  This decoder
    keeps that table out of the archive bytes and validates that the sparse
    stream closes exactly under the supplied replay contract.
    """

    raw = brotli_decompress_segment(encoded, "randmulti")
    if raw[:3] in {b"NM1", b"NM2"}:
        raise HenosisPr82TransferError("PR82 randmulti deconstruction expects headerless sparse payload")
    cursor = 0
    groups: list[Pr82RandmultiGroup] = []
    for group_index, spec in enumerate(specs):
        if len(spec) != 4:
            raise HenosisPr82TransferError(f"randmulti spec {group_index} is malformed")
        height, width, amplitude, scount = [int(value) for value in spec]
        group_start = cursor
        rows, cursor = _decode_randmulti_rows(raw, cursor, scount)
        groups.append(
            Pr82RandmultiGroup(
                group_index=group_index,
                height=height,
                width=width,
                amplitude=amplitude,
                scount=scount,
                rows=rows,
                payload_bytes=cursor - group_start,
            )
        )
    if cursor != len(raw):
        raise HenosisPr82TransferError("randmulti stream has trailing bytes for replay specs")
    return tuple(groups)


def encode_randmulti_qrm1(
    groups: Sequence[Pr82RandmultiGroup],
    *,
    pair_indices: Sequence[int] | None = None,
) -> bytes:
    """Encode PR82-native sparse randmulti groups as a charged ``QRM1`` stream.

    ``QRM1`` is the minimal self-describing extension needed for PR82-native
    groups: the archive carries explicit replay group ids and sparse rows,
    while the runtime supplies the reviewed PR82 group semantics for those ids.
    It avoids ``NM2``'s u8 dimension limit and keeps the original sparse
    economics instead of expanding large groups to dense 600-column rows.
    """

    if len(groups) > 0xFFFF:
        raise HenosisPr82TransferError("QRM1 supports at most 65535 randmulti groups")
    keep: np.ndarray | None = None
    if pair_indices is not None:
        keep = _keep_mask(pair_indices)
    group_ids = [int(group.group_index) for group in groups]
    if len(set(group_ids)) != len(group_ids):
        raise HenosisPr82TransferError("QRM1 group ids must be unique")
    raw = bytearray(b"QRM1")
    raw.extend(len(groups).to_bytes(2, "little"))
    for group in sorted(groups, key=lambda item: int(item.group_index)):
        if not 0 <= int(group.group_index) <= 0xFFFF:
            raise HenosisPr82TransferError(f"QRM1 group id out of range: {group.group_index}")
        rows = group.rows.astype(np.uint8, copy=True)
        if rows.shape != (int(group.scount), 600):
            raise HenosisPr82TransferError(
                f"QRM1 group {group.group_index} rows shape {rows.shape} does not match scount {group.scount}"
            )
        if keep is not None:
            rows[:, ~keep] = 0
        payload = _encode_randmulti_rows(rows)
        raw.extend(int(group.group_index).to_bytes(2, "little"))
        raw.extend(payload)
    return brotli.compress(bytes(raw), quality=11)


def decode_randmulti_qrm1(
    encoded: bytes,
    specs: Sequence[Sequence[int]],
) -> tuple[Pr82RandmultiGroup, ...]:
    """Decode charged ``QRM1`` sparse randmulti groups using PR82 replay specs."""

    raw = brotli_decompress_segment(encoded, "randmulti")
    if raw[:4] != b"QRM1":
        raise HenosisPr82TransferError(f"randmulti stream is not QRM1: {raw[:4]!r}")
    if len(raw) < 6:
        raise HenosisPr82TransferError("QRM1 stream is truncated")
    cursor = 4
    group_count = int.from_bytes(raw[cursor : cursor + 2], "little")
    cursor += 2
    groups: list[Pr82RandmultiGroup] = []
    seen: set[int] = set()
    for _ in range(group_count):
        if cursor + 2 > len(raw):
            raise HenosisPr82TransferError("QRM1 group id is truncated")
        group_index = int.from_bytes(raw[cursor : cursor + 2], "little")
        cursor += 2
        if group_index in seen:
            raise HenosisPr82TransferError(f"QRM1 duplicate group id: {group_index}")
        seen.add(group_index)
        if group_index < 0 or group_index >= len(specs):
            raise HenosisPr82TransferError(f"QRM1 group id {group_index} outside replay specs")
        spec = specs[group_index]
        if len(spec) != 4:
            raise HenosisPr82TransferError(f"QRM1 replay spec {group_index} is malformed")
        height, width, amplitude, scount = [int(value) for value in spec]
        group_start = cursor
        rows, cursor = _decode_randmulti_rows(raw, cursor, scount)
        groups.append(
            Pr82RandmultiGroup(
                group_index=group_index,
                height=height,
                width=width,
                amplitude=amplitude,
                scount=scount,
                rows=rows,
                payload_bytes=cursor - group_start,
            )
        )
    if cursor != len(raw):
        raise HenosisPr82TransferError("QRM1 randmulti stream has trailing bytes")
    return tuple(sorted(groups, key=lambda item: int(item.group_index)))


def randmulti_qrm1_parity_profile(
    original_groups: Sequence[Pr82RandmultiGroup],
    decoded_groups: Sequence[Pr82RandmultiGroup],
    *,
    encoded: bytes,
    source_encoded: bytes | None = None,
) -> dict[str, Any]:
    """Summarize local group-row parity for a ``QRM1`` encoded stream."""

    original_by_id = {int(group.group_index): group for group in original_groups}
    decoded_by_id = {int(group.group_index): group for group in decoded_groups}
    missing = sorted(set(original_by_id) - set(decoded_by_id))
    extra = sorted(set(decoded_by_id) - set(original_by_id))
    row_mismatches: list[int] = []
    spec_mismatches: list[int] = []
    for group_id in sorted(set(original_by_id) & set(decoded_by_id)):
        original = original_by_id[group_id]
        decoded = decoded_by_id[group_id]
        if (
            int(original.height),
            int(original.width),
            int(original.amplitude),
            int(original.scount),
        ) != (
            int(decoded.height),
            int(decoded.width),
            int(decoded.amplitude),
            int(decoded.scount),
        ):
            spec_mismatches.append(group_id)
        if not np.array_equal(original.rows, decoded.rows):
            row_mismatches.append(group_id)
    nonzero_total = int(sum(np.count_nonzero(group.rows) for group in decoded_groups))
    special_count = int(
        sum(
            randmulti_semantic_label(group.height, group.width, group.amplitude).startswith("replay_special")
            for group in decoded_groups
        )
    )
    encoded_raw = brotli_decompress_segment(encoded, "randmulti")
    profile: dict[str, Any] = {
        "contract": "QRM1_sparse_group_id_stream",
        "decoded_group_count": len(decoded_groups),
        "encoded_brotli_bytes": len(encoded),
        "encoded_decoded_bytes": len(encoded_raw),
        "encoded_sha256": sha256_bytes(encoded),
        "exact_group_row_parity": not (missing or extra or spec_mismatches or row_mismatches),
        "extra_group_ids": extra,
        "generic_group_count": len(decoded_groups) - special_count,
        "missing_group_ids": missing,
        "nonzero_choice_total": nonzero_total,
        "replay_special_group_count": special_count,
        "row_mismatch_group_ids": row_mismatches,
        "spec_mismatch_group_ids": spec_mismatches,
    }
    if source_encoded is not None:
        source_raw = brotli_decompress_segment(source_encoded, "randmulti")
        profile.update(
            {
                "source_brotli_bytes": len(source_encoded),
                "source_decoded_bytes": len(source_raw),
                "source_sha256": sha256_bytes(source_encoded),
                "qrm1_brotli_delta_bytes_vs_source_tail": len(encoded) - len(source_encoded),
            }
        )
    return profile


def randmulti_group_summary(group: Pr82RandmultiGroup) -> dict[str, Any]:
    sparse_choice_total = int(np.count_nonzero(group.rows))
    semantic = randmulti_semantic_label(group.height, group.width, group.amplitude)
    return {
        "amplitude": int(group.amplitude),
        "group_index": int(group.group_index),
        "height": int(group.height),
        "nonzero_choice_total": sparse_choice_total,
        "payload_bytes": int(group.payload_bytes),
        "qps1_nm2_runtime_compatible": randmulti_group_qps1_nm2_compatible(group),
        "scount": int(group.scount),
        "semantic": semantic,
        "width": int(group.width),
    }


def encode_randmulti_nm2(groups: Sequence[Pr82RandmultiGroup], *, pair_indices: Sequence[int] | None = None) -> bytes:
    """Encode current-runtime-compatible generic randmulti groups as ``NM2``."""

    if len(groups) > 255:
        raise HenosisPr82TransferError("NM2 supports at most 255 randmulti groups")
    keep: np.ndarray | None = None
    if pair_indices is not None:
        keep = _keep_mask(pair_indices)
    raw = bytearray(b"NM2" + bytes([len(groups)]))
    for group in groups:
        if not randmulti_group_qps1_nm2_compatible(group):
            raise HenosisPr82TransferError(
                f"randmulti group {group.group_index} is not exactly representable by current QPS1 NM2"
            )
        rows = group.rows.astype(np.uint8, copy=True)
        if keep is not None:
            rows[:, ~keep] = 0
        raw.extend(bytes([group.height, group.width, group.amplitude, group.scount]))
        raw.extend(rows.tobytes())
    return brotli.compress(bytes(raw), quality=11)


def _decode_dense_or_delta(raw: bytes, *, full: bytes, delta: bytes, default: int, sparse: bytes | None = None) -> np.ndarray:
    magic = raw[:3]
    if magic == full:
        arr = np.frombuffer(raw, dtype=np.uint8, offset=3).copy()
    elif magic == delta:
        encoded = np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)
        arr = np.where(encoded == 0, default, encoded - 1).astype(np.uint8)
    elif sparse is not None and magic == sparse:
        count = int.from_bytes(raw[3:5], "little")
        indices, values, cursor = _vlq_indices_values(raw, 5, count)
        if cursor != len(raw):
            raise HenosisPr82TransferError(f"{sparse!r} stream has trailing bytes")
        arr = np.full(600, default, dtype=np.uint8)
        for idx, value in zip(indices, values):
            arr[idx] = np.uint8(int(value) - 1)
    else:
        raise HenosisPr82TransferError(f"unsupported qpost stream magic {magic!r}")
    if arr.shape != (600,):
        raise HenosisPr82TransferError(f"qpost stream decoded to shape {arr.shape}, expected (600,)")
    return arr


def decode_postprocess(encoded: bytes) -> np.ndarray:
    raw = brotli_decompress_segment(encoded, "post")
    if raw[:4] == b"PCD1":
        cursor = 5
        stages: list[np.ndarray] = []
        for _ in range(raw[4]):
            if cursor + 3 > len(raw):
                raise HenosisPr82TransferError("PCD1 postprocess header is truncated")
            _stage_id = raw[cursor]
            n = struct.unpack_from("<H", raw, cursor + 1)[0]
            cursor += 3
            choices = np.frombuffer(raw, dtype=np.uint8, count=n, offset=cursor).copy()
            cursor += n
            if choices.shape != (600,):
                raise HenosisPr82TransferError("PCD1 postprocess stage does not contain 600 choices")
            stages.append(choices)
        if cursor != len(raw):
            raise HenosisPr82TransferError("PCD1 postprocess has trailing bytes")
        return np.stack(stages)
    if len(raw) % 600 != 0 or len(raw) // 600 not in (3, 4):
        raise HenosisPr82TransferError("headerless postprocess must be 3 or 4 stages of 600 choices")
    return np.frombuffer(raw, dtype=np.uint8).copy().reshape(len(raw) // 600, 600)


def decode_control_arrays(encoded_segments: Mapping[str, bytes]) -> dict[str, np.ndarray]:
    """Decode runtime-compatible PR82 qpost controls to per-pair arrays."""

    post = decode_postprocess(encoded_segments["post"])
    shift = _decode_dense_or_delta(
        brotli_decompress_segment(encoded_segments["shift"], "shift"),
        full=b"SH4",
        delta=b"SD4",
        default=40,
    )
    frac = _decode_dense_or_delta(
        brotli_decompress_segment(encoded_segments["frac"], "frac"),
        full=b"FH1",
        delta=b"FD1",
        sparse=b"FV1",
        default=4,
    )
    frac2 = _decode_dense_or_delta(
        brotli_decompress_segment(encoded_segments["frac2"], "frac2"),
        full=b"FH2",
        delta=b"FD2",
        default=4,
    )
    frac3 = _decode_dense_or_delta(
        brotli_decompress_segment(encoded_segments["frac3"], "frac3"),
        full=b"FH3",
        delta=b"FD3",
        default=4,
    )
    bias = _decode_dense_or_delta(
        brotli_decompress_segment(encoded_segments["bias"], "bias"),
        full=b"BH1",
        delta=b"BD1",
        sparse=b"BV1",
        default=13,
    )
    region = _decode_dense_or_delta(
        brotli_decompress_segment(encoded_segments["region"], "region"),
        full=b"RH1",
        delta=b"RD1",
        sparse=b"RV1",
        default=0,
    )
    return {
        "post": post,
        "shift": shift,
        "frac": frac,
        "frac2": frac2,
        "frac3": frac3,
        "bias": bias,
        "region": region,
    }


def decode_randmulti_activity(encoded: bytes, specs: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Return per-pair nonzero counts for PR82 headerless randmulti."""

    raw = brotli_decompress_segment(encoded, "randmulti")
    counts_by_pair = np.zeros(600, dtype=np.int32)
    decoded_groups = decode_randmulti_groups(encoded, specs)
    groups: list[dict[str, Any]] = []
    for group in decoded_groups:
        rows = group.rows
        counts_by_pair += np.count_nonzero(rows, axis=0).astype(np.int32)
        groups.append(randmulti_group_summary(group))
    qps1_compatible_groups = [row for row in groups if row["qps1_nm2_runtime_compatible"]]
    return {
        "decoded_bytes": len(raw),
        "groups": groups,
        "nonzero_pair_count": int(np.count_nonzero(counts_by_pair)),
        "per_pair_nonzero_counts": counts_by_pair,
        "qps1_nm2_compatible_group_count": len(qps1_compatible_groups),
        "qps1_nm2_compatible_nonzero_choice_total": int(
            sum(int(row["nonzero_choice_total"]) for row in qps1_compatible_groups)
        ),
        "replay_special_group_count": int(
            sum(str(row["semantic"]).startswith("replay_special") for row in groups)
        ),
        "runtime_compatible_with_qps1_helper": len(qps1_compatible_groups) == len(groups),
    }


def summarize_pair_activity(
    arrays: Mapping[str, np.ndarray],
    *,
    randmulti_counts: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in range(600):
        by_stream: dict[str, int] = {}
        total = 0
        for name, arr in arrays.items():
            if name == "post":
                count = int(np.count_nonzero(arr[:, pair] != QPOST_DEFAULTS[name]))
            else:
                count = int(arr[pair] != QPOST_DEFAULTS[name])
            by_stream[name] = count
            total += count
        if randmulti_counts is not None:
            by_stream["randmulti"] = int(randmulti_counts[pair])
            total += int(randmulti_counts[pair])
        rows.append({"pair_index": pair, "active_atom_count": total, "by_stream": by_stream})
    return rows


def rank_pairs_by_activity(pair_rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        int(row["pair_index"])
        for row in sorted(
            pair_rows,
            key=lambda row: (-int(row["active_atom_count"]), int(row["pair_index"])),
        )
    ]


def _keep_mask(pair_indices: Sequence[int]) -> np.ndarray:
    keep = np.zeros(600, dtype=bool)
    for pair in pair_indices:
        if pair < 0 or pair >= 600:
            raise HenosisPr82TransferError(f"pair index out of range: {pair}")
        keep[int(pair)] = True
    return keep


def filter_qpost_streams_to_pairs(
    encoded_segments: Mapping[str, bytes],
    pair_indices: Sequence[int],
    *,
    include_streams: Sequence[str],
) -> dict[str, bytes]:
    """Build QPS1 stream blobs where non-selected pairs decode to identity."""

    unknown = sorted(set(include_streams) - set(QPOST_STREAM_NAMES))
    if unknown:
        raise HenosisPr82TransferError(f"unknown qpost stream(s): {unknown}")
    if "randmulti" in include_streams:
        raise HenosisPr82TransferError(
            "PR82 randmulti has 72 replay groups and is not QPS1-runtime-compatible"
        )
    arrays = decode_control_arrays(encoded_segments)
    keep = _keep_mask(pair_indices)
    include = set(include_streams)
    out: dict[str, bytes] = {}
    for name in QPOST_STREAM_NAMES:
        if name not in include:
            out[name] = b""
            continue
        if name == "post":
            arr = arrays[name].copy()
            arr[:, ~keep] = 0
            out[name] = brotli.compress(arr.astype(np.uint8, copy=False).tobytes(), quality=11)
        elif name == "shift":
            arr = arrays[name].copy()
            arr[~keep] = 40
            out[name] = brotli.compress(b"SH4" + arr.astype(np.uint8, copy=False).tobytes(), quality=11)
        elif name in {"frac", "frac2", "frac3"}:
            arr = arrays[name].copy()
            arr[~keep] = 4
            magic = {"frac": b"FH1", "frac2": b"FH2", "frac3": b"FH3"}[name]
            out[name] = brotli.compress(magic + arr.astype(np.uint8, copy=False).tobytes(), quality=11)
        elif name == "bias":
            arr = arrays[name].copy()
            arr[~keep] = 13
            out[name] = brotli.compress(b"BH1" + arr.astype(np.uint8, copy=False).tobytes(), quality=11)
        elif name == "region":
            arr = arrays[name].copy()
            arr[~keep] = 0
            out[name] = brotli.compress(b"RH1" + arr.astype(np.uint8, copy=False).tobytes(), quality=11)
    return out


def encode_qpost(streams: Mapping[str, bytes]) -> bytes:
    missing = [name for name in QPOST_STREAM_NAMES if name not in streams]
    if missing:
        raise HenosisPr82TransferError(f"missing qpost stream(s): {missing}")
    lengths = [len(streams[name]) for name in QPOST_STREAM_NAMES]
    return QPOST_MAGIC + struct.pack("<" + "I" * len(QPOST_STREAM_NAMES), *lengths) + b"".join(
        streams[name] for name in QPOST_STREAM_NAMES
    )


def qpost_stream_summary(streams: Mapping[str, bytes], original: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "active": bool(streams[name]),
            "bytes": len(streams[name]),
            "original_bytes": len(original[name]),
            "sha256": sha256_bytes(streams[name]),
            "original_sha256": sha256_bytes(original[name]),
        }
        for name in QPOST_STREAM_NAMES
    }


def decode_pr82_p1d1_pose(encoded_pose: bytes) -> np.ndarray:
    raw = brotli_decompress_segment(encoded_pose, "pose")
    if not raw.startswith(b"P1D1"):
        raise HenosisPr82TransferError(f"PR82 pose stream is not P1D1: {raw[:4]!r}")
    cursor = 4
    dim_count = raw[cursor]
    cursor += 1
    dims: list[int] = []
    lengths: list[int] = []
    for _ in range(dim_count):
        if cursor + 3 > len(raw):
            raise HenosisPr82TransferError("P1D1 dimension header is truncated")
        dims.append(int(raw[cursor]))
        lengths.append(int.from_bytes(raw[cursor + 1 : cursor + 3], "little"))
        cursor += 3
    pose = np.zeros((600, 6), dtype=np.float32)
    for dim, n_bytes in zip(dims, lengths):
        stream = raw[cursor : cursor + n_bytes]
        if len(stream) != n_bytes:
            raise HenosisPr82TransferError("P1D1 dimension stream is truncated")
        cursor += n_bytes
        values: list[int] = []
        pos = 0
        while pos < len(stream):
            zz, pos = _read_vlq(stream, pos)
            delta = (zz >> 1) ^ -(zz & 1)
            previous = values[-1] if values else 0
            values.append(previous + delta)
        if len(values) != 600:
            raise HenosisPr82TransferError(f"P1D1 dim {dim} decoded {len(values)} values, expected 600")
        q = np.asarray(values, dtype=np.int32)
        if dim == 0:
            pose[:, 0] = q.astype(np.float32) / 512.0 + 20.0
        else:
            pose[:, dim] = q.clip(-32768, 32767).astype(np.int16).astype(np.float32) / 2048.0
    if cursor != len(raw):
        raise HenosisPr82TransferError("P1D1 pose stream has trailing bytes")
    return pose


def pose_velocity_atom_ranking(source_pose: np.ndarray, pr82_pose: np.ndarray) -> list[dict[str, Any]]:
    if source_pose.shape[0] != 600 or pr82_pose.shape[0] != 600:
        raise HenosisPr82TransferError("pose atom ranking expects 600 pose rows")
    delta_q = np.rint((pr82_pose[:, 0] - source_pose[:, 0]) * 512.0).astype(np.int64)
    atoms = [
        {
            "abs_delta_q": int(abs(value)),
            "delta_q": int(value),
            "dimension": 0,
            "pair_index": int(pair),
        }
        for pair, value in enumerate(delta_q)
        if int(value) != 0
    ]
    return sorted(atoms, key=lambda row: (-int(row["abs_delta_q"]), int(row["pair_index"])))
