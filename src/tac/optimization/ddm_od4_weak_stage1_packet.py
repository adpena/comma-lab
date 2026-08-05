# SPDX-License-Identifier: MIT
"""Weak Stage-1 packet and mask-domain receiver for DDM OD4.

The packet carries counted, video-derived sparse constraints.  The receiver
code is generic: parse the packet, apply target labels at named scorer-lattice
sites, and report mask-domain fidelity against caller-supplied references.  It
does not load scorers, scorer weights, GT tables, or dense solved fields.
"""

from __future__ import annotations

import hashlib
import lzma
import math
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import brotli
import numpy as np

SEG_H: Final = 384
SEG_W: Final = 512
N_PAIRS: Final = 600
PIXELS_PER_PAIR: Final = SEG_H * SEG_W
RATE_DENOMINATOR_BYTES: Final = 37_545_489
RATE_PER_BYTE: Final = 25.0 / RATE_DENOMINATOR_BYTES
CURRENT_OWN_S: Final = 0.7539807296911207
CURRENT_OWN_BYTES: Final = 357_836
CURRENT_OWN_AXIS: Final = "[macOS-CPU advisory]"
OD2_STAGE2_K4_BYTES_N600: Final = 57_600
OD2_STAGE2_POSE_DELTA_S: Final = -0.0024094072901496427

PACKET_SCHEMA: Final = "ddm_od4_weak_stage1_sparse_packet.v1"
RECEIPT_SCHEMA: Final = "ddm_od4_weak_stage1_packet_receipt.v1"
MAGIC: Final = b"OD4WPK1\0"
VERSION: Final = 1
HEADER = struct.Struct("<8sBHHHI32s")
OD5_PACKET_SCHEMA: Final = "ddm_od5_generator_coordinate_packet.v1"
OD5_RECEIPT_SCHEMA: Final = "ddm_od5_generator_coordinate_receipt.v1"
OD5_MAGIC: Final = b"OD5GPK1\0"
OD5_VERSION: Final = 1
OD5_HEADER = struct.Struct("<8sBHHHII32s")
OD5_MAX_SECTION_NAME_BYTES: Final = 80
LZMA1_FILTERS: Final = (
    {"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0},
)


class OD4PacketError(ValueError):
    """The OD4 weak packet or replay proof failed closed."""


@dataclass(frozen=True, slots=True)
class SparsePairCorrections:
    pair: int
    flat_indices: tuple[int, ...]
    target_labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.pair < N_PAIRS:
            raise OD4PacketError(f"pair id out of range: {self.pair}")
        if len(self.flat_indices) != len(self.target_labels):
            raise OD4PacketError("flat index and target label lengths differ")
        previous = -1
        for flat in self.flat_indices:
            if not 0 <= int(flat) < PIXELS_PER_PAIR:
                raise OD4PacketError(f"flat index out of range: {flat}")
            if int(flat) <= previous:
                raise OD4PacketError("flat indices must be strictly increasing per pair")
            previous = int(flat)
        for label in self.target_labels:
            if not 0 <= int(label) <= 4:
                raise OD4PacketError(f"target label out of range: {label}")

    @property
    def count(self) -> int:
        return len(self.flat_indices)


@dataclass(frozen=True, slots=True)
class ParsedSparsePacket:
    h: int
    w: int
    pair_records: tuple[SparsePairCorrections, ...]
    payload_sha256: str

    @property
    def correction_count(self) -> int:
        return sum(record.count for record in self.pair_records)


@dataclass(frozen=True, slots=True)
class CoderRow:
    codec: str
    bytes: int
    sha256: str
    parseback_exact: bool
    blocker: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "codec": self.codec,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "parseback_exact": self.parseback_exact,
            "blocker": self.blocker,
        }


@dataclass(frozen=True, slots=True)
class OD5Section:
    name: str
    payload: bytes

    def __post_init__(self) -> None:
        try:
            encoded = self.name.encode("ascii")
        except UnicodeEncodeError as exc:
            raise OD4PacketError("OD5 section names must be ASCII") from exc
        if not encoded:
            raise OD4PacketError("OD5 section name is empty")
        if len(encoded) > OD5_MAX_SECTION_NAME_BYTES:
            raise OD4PacketError("OD5 section name is too long")
        if not isinstance(self.payload, bytes):
            raise OD4PacketError("OD5 section payload must be bytes")


@dataclass(frozen=True, slots=True)
class ParsedOD5Packet:
    h: int
    w: int
    n_pairs: int
    sections: tuple[OD5Section, ...]
    payload_sha256: str

    @property
    def section_count(self) -> int:
        return len(self.sections)

    @property
    def body_bytes(self) -> int:
        return sum(len(_od5_section_body(section)) for section in self.sections)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _varint(value: int) -> bytes:
    if value < 0:
        raise OD4PacketError("varint cannot encode a negative value")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload):
            raise OD4PacketError("truncated varint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise OD4PacketError("varint is too long")


def _od5_section_body(section: OD5Section) -> bytes:
    name = section.name.encode("ascii")
    return _varint(len(name)) + name + _varint(len(section.payload)) + section.payload


def serialize_od5_packet(sections: Iterable[OD5Section]) -> bytes:
    ordered = tuple(sections)
    names = [section.name for section in ordered]
    if len(set(names)) != len(names):
        raise OD4PacketError("duplicate OD5 section names")
    body = b"".join(_od5_section_body(section) for section in ordered)
    header = OD5_HEADER.pack(
        OD5_MAGIC,
        OD5_VERSION,
        SEG_H,
        SEG_W,
        N_PAIRS,
        len(ordered),
        len(body),
        hashlib.sha256(body).digest(),
    )
    return header + body


def parse_od5_packet(payload: bytes) -> ParsedOD5Packet:
    if len(payload) < OD5_HEADER.size:
        raise OD4PacketError("OD5 packet header is truncated")
    magic, version, h, w, n_pairs, section_count, body_bytes, body_sha = OD5_HEADER.unpack_from(payload)
    if magic != OD5_MAGIC:
        raise OD4PacketError("OD5 packet magic mismatch")
    if version != OD5_VERSION:
        raise OD4PacketError(f"OD5 packet version mismatch: {version}")
    if (h, w, n_pairs) != (SEG_H, SEG_W, N_PAIRS):
        raise OD4PacketError(f"OD5 packet geometry mismatch: {(h, w, n_pairs)}")
    body = payload[OD5_HEADER.size :]
    if len(body) != body_bytes:
        raise OD4PacketError("OD5 packet body length mismatch")
    if hashlib.sha256(body).digest() != body_sha:
        raise OD4PacketError("OD5 packet body SHA-256 mismatch")
    sections: list[OD5Section] = []
    offset = 0
    for _ in range(section_count):
        name_len, offset = _read_varint(body, offset)
        if not 0 < name_len <= OD5_MAX_SECTION_NAME_BYTES:
            raise OD4PacketError("OD5 section name length invalid")
        name_bytes = body[offset : offset + name_len]
        if len(name_bytes) != name_len:
            raise OD4PacketError("OD5 section name is truncated")
        offset += name_len
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise OD4PacketError("OD5 section name is not ASCII") from exc
        section_len, offset = _read_varint(body, offset)
        section_payload = body[offset : offset + section_len]
        if len(section_payload) != section_len:
            raise OD4PacketError("OD5 section payload is truncated")
        offset += section_len
        sections.append(OD5Section(name, section_payload))
    if offset != len(body):
        raise OD4PacketError("OD5 packet has trailing body bytes")
    if len({section.name for section in sections}) != len(sections):
        raise OD4PacketError("OD5 packet contains duplicate section names")
    return ParsedOD5Packet(h, w, n_pairs, tuple(sections), sha256_bytes(payload))


def _pack_nibbles(values: Sequence[int]) -> bytes:
    out = bytearray()
    items = [int(value) for value in values]
    for label in items:
        if not 0 <= label <= 15:
            raise OD4PacketError(f"nibble value out of range: {label}")
    for idx in range(0, len(items), 2):
        lo = items[idx]
        hi = items[idx + 1] if idx + 1 < len(items) else 0
        out.append(lo | (hi << 4))
    return bytes(out)


def _unpack_nibbles(payload: bytes, count: int) -> tuple[int, ...]:
    values: list[int] = []
    for byte in payload:
        values.append(byte & 15)
        if len(values) == count:
            break
        values.append(byte >> 4)
        if len(values) == count:
            break
    if len(values) != count:
        raise OD4PacketError("nibble payload is truncated")
    return tuple(values)


def _pair_body(record: SparsePairCorrections) -> bytes:
    body = bytearray()
    body += _varint(record.pair)
    body += _varint(record.count)
    previous = -1
    for flat in record.flat_indices:
        body += _varint(int(flat) - previous - 1)
        previous = int(flat)
    body += _pack_nibbles(record.target_labels)
    return bytes(body)


def _read_pair_body(payload: bytes, offset: int) -> tuple[SparsePairCorrections, int]:
    pair, offset = _read_varint(payload, offset)
    count, offset = _read_varint(payload, offset)
    flat_indices: list[int] = []
    previous = -1
    for _ in range(count):
        delta, offset = _read_varint(payload, offset)
        flat = previous + 1 + delta
        flat_indices.append(flat)
        previous = flat
    label_bytes = (count + 1) // 2
    labels_payload = payload[offset : offset + label_bytes]
    if len(labels_payload) != label_bytes:
        raise OD4PacketError("target-label nibble payload is truncated")
    offset += label_bytes
    return SparsePairCorrections(pair, tuple(flat_indices), _unpack_nibbles(labels_payload, count)), offset


def serialize_sparse_packet(records: Iterable[SparsePairCorrections]) -> bytes:
    ordered = tuple(sorted(records, key=lambda record: record.pair))
    if len({record.pair for record in ordered}) != len(ordered):
        raise OD4PacketError("duplicate pair records in OD4 packet")
    body = b"".join(_pair_body(record) for record in ordered)
    header = HEADER.pack(MAGIC, VERSION, SEG_H, SEG_W, len(ordered), len(body), hashlib.sha256(body).digest())
    return header + body


def parse_sparse_packet(payload: bytes) -> ParsedSparsePacket:
    if len(payload) < HEADER.size:
        raise OD4PacketError("OD4 packet header is truncated")
    magic, version, h, w, row_count, body_bytes, body_sha = HEADER.unpack_from(payload)
    if magic != MAGIC:
        raise OD4PacketError("OD4 packet magic mismatch")
    if version != VERSION:
        raise OD4PacketError(f"OD4 packet version mismatch: {version}")
    if (h, w) != (SEG_H, SEG_W):
        raise OD4PacketError(f"OD4 packet grid mismatch: {(h, w)}")
    body = payload[HEADER.size :]
    if len(body) != body_bytes:
        raise OD4PacketError("OD4 packet body length mismatch")
    if hashlib.sha256(body).digest() != body_sha:
        raise OD4PacketError("OD4 packet body SHA-256 mismatch")
    records: list[SparsePairCorrections] = []
    offset = 0
    for _ in range(row_count):
        record, offset = _read_pair_body(body, offset)
        records.append(record)
    if offset != len(body):
        raise OD4PacketError("OD4 packet has trailing body bytes")
    if len({record.pair for record in records}) != len(records):
        raise OD4PacketError("OD4 packet contains duplicate pair records")
    return ParsedSparsePacket(h, w, tuple(records), sha256_bytes(payload))


def apply_sparse_packet(base_argmax: np.ndarray, packet: ParsedSparsePacket) -> dict[int, np.ndarray]:
    base = np.asarray(base_argmax)
    if base.ndim != 3 or base.shape[1:] != (SEG_H, SEG_W):
        raise OD4PacketError(f"base argmax shape mismatch: {base.shape}")
    out: dict[int, np.ndarray] = {}
    for record in packet.pair_records:
        if record.pair >= base.shape[0]:
            raise OD4PacketError(f"base argmax is missing pair {record.pair}")
        labels = np.array(base[record.pair], copy=True)
        flat = labels.reshape(-1)
        if record.flat_indices:
            flat[np.asarray(record.flat_indices, dtype=np.int64)] = np.asarray(record.target_labels, dtype=flat.dtype)
        out[record.pair] = labels
    return out


def select_sparse_corrections(
    *,
    pair: int,
    current_argmax: np.ndarray,
    gt_argmax: np.ndarray,
    target_argmax: np.ndarray,
    desired_fix_count: int,
    fraction: float,
) -> SparsePairCorrections:
    if not 0.0 <= fraction <= 1.0:
        raise OD4PacketError("fraction must lie in [0, 1]")
    cur = np.asarray(current_argmax, dtype=np.uint8)
    gt = np.asarray(gt_argmax, dtype=np.uint8)
    target = np.asarray(target_argmax, dtype=np.uint8)
    if cur.shape != (SEG_H, SEG_W) or gt.shape != cur.shape or target.shape != cur.shape:
        raise OD4PacketError("argmax grid shape mismatch")
    useful = (cur != gt) & (target == gt) & (target != cur)
    useful_flat = np.flatnonzero(useful.reshape(-1))
    keep = min(int(math.floor(max(0, desired_fix_count) * fraction + 1e-9)), int(useful_flat.size))
    selected = np.sort(useful_flat[:keep].astype(np.int64))
    labels = tuple(int(value) for value in target.reshape(-1)[selected])
    return SparsePairCorrections(pair, tuple(int(value) for value in selected), labels)


def select_masked_sparse_corrections(
    *,
    pair: int,
    current_argmax: np.ndarray,
    gt_argmax: np.ndarray,
    target_argmax: np.ndarray,
    constraint_mask: np.ndarray,
    max_count: int | None = None,
) -> SparsePairCorrections:
    cur = np.asarray(current_argmax, dtype=np.uint8)
    gt = np.asarray(gt_argmax, dtype=np.uint8)
    target = np.asarray(target_argmax, dtype=np.uint8)
    mask = np.asarray(constraint_mask, dtype=bool)
    if cur.shape != (SEG_H, SEG_W) or gt.shape != cur.shape or target.shape != cur.shape:
        raise OD4PacketError("argmax grid shape mismatch")
    if mask.shape != cur.shape:
        raise OD4PacketError("constraint mask shape mismatch")
    useful = (cur != gt) & (target == gt) & (target != cur) & mask
    useful_flat = np.flatnonzero(useful.reshape(-1)).astype(np.int64)
    if max_count is not None:
        if max_count < 0:
            raise OD4PacketError("max_count cannot be negative")
        useful_flat = useful_flat[:max_count]
    selected = np.sort(useful_flat)
    labels = tuple(int(value) for value in target.reshape(-1)[selected])
    return SparsePairCorrections(pair, tuple(int(value) for value in selected), labels)


def fidelity_for_packet(
    *,
    current_argmax: np.ndarray,
    gt_argmax: np.ndarray,
    packet: ParsedSparsePacket,
    od2_rows_by_pair: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    decoded = apply_sparse_packet(current_argmax, packet)
    rows: list[dict[str, Any]] = []
    total_before = 0
    total_after = 0
    total_od2_after = 0
    total_n_described = 0
    changed_pixels = 0
    for record in packet.pair_records:
        row = od2_rows_by_pair[record.pair]
        gt = np.asarray(gt_argmax[record.pair], dtype=np.uint8)
        before_map = np.asarray(current_argmax[record.pair], dtype=np.uint8)
        after_map = decoded[record.pair]
        before = int((before_map != gt).sum())
        after = int((after_map != gt).sum())
        od2_after = int(row["stage1"]["flips_after"])
        n_described = int(row["n_described"])
        changed = int((after_map != before_map).sum())
        fixed = before - after
        od2_fixed = before - od2_after
        rows.append(
            {
                "pair": record.pair,
                "flips_before": before,
                "flips_after_receiver": after,
                "flips_after_od2_stage1": od2_after,
                "retained_fix_count": fixed,
                "od2_fix_count": od2_fixed,
                "retained_fraction_vs_od2": fixed / od2_fixed if od2_fixed else None,
                "n_described": n_described,
                "eta_receiver": fixed / n_described if n_described else None,
                "changed_pixels": changed,
                "packet_corrections": record.count,
            }
        )
        total_before += before
        total_after += after
        total_od2_after += od2_after
        total_n_described += n_described
        changed_pixels += changed
    total_fixed = total_before - total_after
    od2_fixed_total = total_before - total_od2_after
    return {
        "rows": rows,
        "totals": {
            "pairs": len(rows),
            "flips_before": total_before,
            "flips_after_receiver": total_after,
            "flips_after_od2_stage1": total_od2_after,
            "retained_fix_count": total_fixed,
            "od2_fix_count": od2_fixed_total,
            "retained_fraction_vs_od2": total_fixed / od2_fixed_total if od2_fixed_total else None,
            "n_described": total_n_described,
            "eta_receiver": total_fixed / total_n_described if total_n_described else None,
            "changed_pixels": changed_pixels,
            "parseback_exact": True,
        },
    }


def lzma1_raw(payload: bytes) -> bytes:
    return lzma.compress(payload, format=lzma.FORMAT_RAW, filters=list(LZMA1_FILTERS))


def unlzma1_raw(payload: bytes, expected_len: int) -> bytes:
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=list(LZMA1_FILTERS))
    got = dec.decompress(payload, max_length=expected_len + 1)
    if len(got) != expected_len or not dec.eof or dec.unused_data:
        raise OD4PacketError("LZMA1 raw packet roundtrip failed")
    return got


def race_packet_coders(
    raw_packet: bytes,
    *,
    smevr_encode: Callable[[list[bytes]], bytes] | None = None,
    smevr_decode: Callable[[bytes], list[bytes]] | None = None,
) -> tuple[CoderRow, ...]:
    rows: list[CoderRow] = []
    encoded_brotli = brotli.compress(raw_packet, quality=11)
    rows.append(
        CoderRow(
            codec="brotli-q11",
            bytes=len(encoded_brotli),
            sha256=sha256_bytes(encoded_brotli),
            parseback_exact=brotli.decompress(encoded_brotli) == raw_packet,
        )
    )
    encoded_lzma = lzma1_raw(raw_packet)
    rows.append(
        CoderRow(
            codec="lzma1-raw",
            bytes=len(encoded_lzma),
            sha256=sha256_bytes(encoded_lzma),
            parseback_exact=unlzma1_raw(encoded_lzma, len(raw_packet)) == raw_packet,
        )
    )
    if (smevr_encode is None) != (smevr_decode is None):
        raise OD4PacketError("SMEVR encode/decode callbacks must be supplied together")
    if smevr_encode is not None and smevr_decode is not None:
        try:
            encoded_smevr = smevr_encode([raw_packet])
            decoded_records = smevr_decode(encoded_smevr)
            rows.append(
                CoderRow(
                    codec="smevr-r7-nibble",
                    bytes=len(encoded_smevr),
                    sha256=sha256_bytes(encoded_smevr),
                    parseback_exact=decoded_records == [raw_packet],
                )
            )
        except Exception as exc:  # pragma: no cover - exercised by integration CLI when R7 is unavailable.
            rows.append(
                CoderRow(
                    codec="smevr-r7-nibble",
                    bytes=0,
                    sha256="",
                    parseback_exact=False,
                    blocker=str(exc),
                )
            )
    return tuple(sorted(rows, key=lambda row: (not row.parseback_exact, row.bytes if row.bytes else 10**18, row.codec)))


def projected_stage1_delta_s(retained_fix_count: int, n_pairs: int) -> float:
    if n_pairs <= 0:
        raise OD4PacketError("n_pairs must be positive")
    return -100.0 * retained_fix_count / (n_pairs * PIXELS_PER_PAIR)


def projected_n600_packet_bytes(measured_bytes: int, n_pairs: int) -> int:
    if n_pairs <= 0:
        raise OD4PacketError("n_pairs must be positive")
    return int(math.ceil(measured_bytes * (N_PAIRS / n_pairs)))


def projection_rows(
    *,
    n32_packet_bytes: int,
    n_pairs: int,
    retained_fix_count: int,
    include_od2_pose_credit: bool,
) -> dict[str, Any]:
    stage1_delta_s = projected_stage1_delta_s(retained_fix_count, n_pairs)
    packet_bytes_n600_projected = projected_n600_packet_bytes(n32_packet_bytes, n_pairs)
    packet_rate_s = packet_bytes_n600_projected * RATE_PER_BYTE
    stage2_rate_s = OD2_STAGE2_K4_BYTES_N600 * RATE_PER_BYTE
    pose_delta_s = OD2_STAGE2_POSE_DELTA_S if include_od2_pose_credit else 0.0
    projected_s = CURRENT_OWN_S + stage1_delta_s + pose_delta_s + stage2_rate_s + packet_rate_s
    return {
        "stage1_delta_s_from_mask_replay": stage1_delta_s,
        "stage2_pose_delta_s": pose_delta_s,
        "stage2_k4_rate_s": stage2_rate_s,
        "packet_rate_s_projected_n600": packet_rate_s,
        "packet_bytes_n32_exact": n32_packet_bytes,
        "packet_bytes_n600_linear_projection": packet_bytes_n600_projected,
        "projected_s": projected_s,
        "beats_current_own_line": projected_s < CURRENT_OWN_S,
        "projection_scope": (
            "n32 exact packet bytes with linear n600 byte projection; mask-domain Stage-1 replay; "
            "OD2 pose credit included only when requested, not remeasured by OD4"
        ),
    }


def projection_rows_with_projected_packet_bytes(
    *,
    n32_packet_bytes: int,
    n600_packet_bytes_projected: int,
    n_pairs: int,
    retained_fix_count: int,
    include_od2_pose_credit: bool,
    projection_scope: str,
) -> dict[str, Any]:
    if n32_packet_bytes < 0 or n600_packet_bytes_projected < 0:
        raise OD4PacketError("packet byte counts cannot be negative")
    stage1_delta_s = projected_stage1_delta_s(retained_fix_count, n_pairs)
    packet_rate_s = n600_packet_bytes_projected * RATE_PER_BYTE
    stage2_rate_s = OD2_STAGE2_K4_BYTES_N600 * RATE_PER_BYTE
    pose_delta_s = OD2_STAGE2_POSE_DELTA_S if include_od2_pose_credit else 0.0
    projected_s = CURRENT_OWN_S + stage1_delta_s + pose_delta_s + stage2_rate_s + packet_rate_s
    rate_cost_over_seg_win = packet_rate_s / abs(stage1_delta_s) if stage1_delta_s else math.inf
    return {
        "stage1_delta_s_from_mask_replay": stage1_delta_s,
        "stage2_pose_delta_s": pose_delta_s,
        "stage2_k4_rate_s": stage2_rate_s,
        "packet_rate_s_projected_n600": packet_rate_s,
        "packet_bytes_n32_exact": n32_packet_bytes,
        "packet_bytes_n600_projected": n600_packet_bytes_projected,
        "projected_s": projected_s,
        "beats_current_own_line": projected_s < CURRENT_OWN_S,
        "rate_cost_over_seg_win": rate_cost_over_seg_win,
        "projection_scope": projection_scope,
    }
