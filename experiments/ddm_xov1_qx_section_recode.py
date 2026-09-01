#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""XOV1 scorer-free QX-form recode of AFR1 through the GF1 generator baseline.

This is the one OPEN-$0 component cell in the XOV1 crossover matrix.  It
decodes GF1's counted generator packet, then codes every disagreement with the
exact AFR1 semantic field in ten independently resumable 60-pair sections
using QX2's address-free boundary-enumerative form.  Every raw section, every
real-coder payload and repeat, every section parse-back, and the complete
aggregate receiver parse-back is retained.

No scorer is imported or invoked.  This instrument measures bytes and exact
semantic-field identity only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for _import_root in (REPO, REPO / "src", REPO / "experiments"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))
STORE: Final = Path("/Volumes/APDataStore/pact/ddm_xov1_crossover_pass")
GF1_PACKET: Final = Path(
    "/Volumes/APDataStore/pact/ddm_gf1_generator_form_on_lb1_field/retained/"
    "gf1_lb1_fit.packet"
)
GF1_PACKET_SHA256: Final = "87d79345982dde33e30ca328de2dcde9c66c20e12e7729a3690ae8e23b4e1497"
GF1_FIELD: Final = Path(
    "/Volumes/APDataStore/pact/ddm_gf1_generator_form_on_lb1_field/retained/"
    "generated_from_lb1_fit.u8"
)
GF1_FIELD_SHA256: Final = "4026c4e2c805beb5b79be2879bb4a84311655d0d7d80dbc766654847522a5d19"
AFR1_FIELD: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/identity_v1/out/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
AFR1_FIELD_SHA256: Final = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
AFR1_BITS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/retained/"
    "afc1_control/bits_per_frame_afc1_tile48_groupbin8.npy"
)
AFR1_BITS_SHA256: Final = "9954e90eb88fe5227f2899a569fe47105cf34473e12d0c018fcdc8b176722d1d"
GT_FIELD: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
)
GT_FIELD_SHA256: Final = "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"
GT_SEMANTIC_SHA256: Final = "a98b90678ca5d4e12b385d2c8596839b368af8d52277eea3c1d3666f7a4c9b3d"
QX2_SOURCE: Final = REPO / "experiments/ddm_qx2_events_section_redesign.py"
QX2_SOURCE_SHA256: Final = "88457037f5cbc272b494306a1613f8c6e2abe3499fdf83164274e3db76b1311c"

N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
SECTION_PAIRS: Final = 60
SECTION_COUNT: Final = N_PAIRS // SECTION_PAIRS
FIELD_BYTES: Final = N_PAIRS * HEIGHT * WIDTH
MINIMUM_FREE_BYTES: Final = 700_000_000
AFR1_MODEL_BYTES: Final = 13_515
AFR1_TOKEN_BYTES: Final = 113_411
AFR1_JOINT_POOL_BYTES: Final = AFR1_MODEL_BYTES + AFR1_TOKEN_BYTES
AFR1_FIXED_REMAINDER_BYTES: Final = 53_076
STRICT_ARCHIVE_CAP_BYTES: Final = 137_986
STRICT_REPLACEMENT_POOL_MAX_EXCLUSIVE: Final = STRICT_ARCHIVE_CAP_BYTES - AFR1_FIXED_REMAINDER_BYTES
AXIS: Final = "[macOS-CPU scorer-free exact byte and receiver parse-back measurement]"

AGGREGATE_HEADER: Final = struct.Struct(">4sBBHII32s32s")
AGGREGATE_ROW: Final = struct.Struct(">HHB3xII32s32sI")
BHW_HEADER: Final = struct.Struct(">4sBBHQQQ32s32s32s")
BHW_RECORD: Final = struct.Struct("<IBB")
CODEC_IDS: Final = {"brotli_q11": 1, "lzma9e": 2, "zlib9": 3}
ID_CODECS: Final = {value: key for key, value in CODEC_IDS.items()}


class XOV1Error(RuntimeError):
    """A custody, storage, determinism, or receiver-identity gate failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")


def require(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise XOV1Error(f"required input is absent: {path}")
    observed = fact(path)
    if observed["sha256"] != expected_sha256:
        raise XOV1Error(f"required input SHA drifted: {path}")
    if expected_bytes is not None and observed["bytes"] != expected_bytes:
        raise XOV1Error(f"required input length drifted: {path}")
    return observed


def checkpoint_valid(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    for retained in payload.get("retained", []):
        retained_path = Path(retained["path"])
        if not retained_path.is_file() or fact(retained_path) != retained:
            raise XOV1Error(f"checkpoint payload drifted: {retained_path}")
    return payload


def source_preflight(store: Path) -> dict[str, Any]:
    store.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(store).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise XOV1Error(f"APDataStore free space {free_bytes} < {MINIMUM_FREE_BYTES}")
    inputs = {
        "gf1_packet": require(GF1_PACKET, GF1_PACKET_SHA256, 47_603),
        "gf1_retained_field": require(GF1_FIELD, GF1_FIELD_SHA256, FIELD_BYTES),
        "afr1_exact_field": require(AFR1_FIELD, AFR1_FIELD_SHA256, FIELD_BYTES),
        "afr1_bits_per_frame": require(AFR1_BITS, AFR1_BITS_SHA256),
        "dali_gt_field": require(GT_FIELD, GT_FIELD_SHA256, FIELD_BYTES + 128),
        "qx2_source": require(QX2_SOURCE, QX2_SOURCE_SHA256),
    }
    bits = np.load(AFR1_BITS, allow_pickle=False)
    if bits.shape != (N_PAIRS,) or int(np.ceil(float(bits.sum()) / 8.0)) != AFR1_TOKEN_BYTES:
        raise XOV1Error("AFR1 per-frame bit ledger does not close to the shipped token stream")
    receipt = {
        "schema": "ddm_xov1_qx_section_recode_preflight.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "storage": {"path": str(store), "observed_free_bytes": free_bytes},
        "inputs": inputs,
        "denominators": {
            "pairs": N_PAIRS,
            "sections": SECTION_COUNT,
            "pairs_per_section": SECTION_PAIRS,
            "semantic_sites": FIELD_BYTES,
        },
    }
    atomic_json(store / "checkpoints/STAGE_00_PREFLIGHT.json", receipt)
    return receipt


def build_cross_parent_bhw(
    store: Path, baseline: np.memmap, target: np.memmap
) -> dict[str, Any]:
    """Classify both directed parent edits against the exact DALI GT field."""

    checkpoint = store / "checkpoints/STAGE_31_CROSS_BHW.json"
    resumed = checkpoint_valid(checkpoint)
    if resumed is not None:
        return resumed
    gt = np.load(GT_FIELD, mmap_mode="r", allow_pickle=False)
    if gt.shape != (N_PAIRS, HEIGHT, WIDTH) or gt.dtype != np.uint8:
        raise XOV1Error("DALI GT semantic geometry or dtype drifted")
    if sha256_bytes(np.asarray(gt).tobytes(order="C")) != GT_SEMANTIC_SHA256:
        raise XOV1Error("DALI GT semantic content drifted")

    # group 0: AFR1 -> GF1 is label-beneficial (GF1 equals GT)
    # group 1: GF1 -> AFR1 is label-beneficial (AFR1 equals GT)
    # group 2: both parents disagree with GT (wash at token-label level)
    group_indices: list[list[np.ndarray]] = [[], [], []]
    group_old: list[list[np.ndarray]] = [[], [], []]
    group_new: list[list[np.ndarray]] = [[], [], []]
    for pair in range(N_PAIRS):
        generated = np.asarray(baseline[pair]).reshape(-1)
        afr1 = np.asarray(target[pair]).reshape(-1)
        wanted = np.asarray(gt[pair]).reshape(-1)
        changed = generated != afr1
        masks = (
            changed & (afr1 != wanted) & (generated == wanted),
            changed & (generated != wanted) & (afr1 == wanted),
            changed & (generated != wanted) & (afr1 != wanted),
        )
        for group_id, mask in enumerate(masks):
            local = np.flatnonzero(mask).astype(np.uint32, copy=False)
            group_indices[group_id].append(local + np.uint32(pair * HEIGHT * WIDTH))
            if group_id == 0:
                group_old[group_id].append(afr1[local].astype(np.uint8, copy=False))
                group_new[group_id].append(generated[local].astype(np.uint8, copy=False))
            else:
                group_old[group_id].append(generated[local].astype(np.uint8, copy=False))
                group_new[group_id].append(afr1[local].astype(np.uint8, copy=False))
    indices = [np.concatenate(parts) for parts in group_indices]
    old_values = [np.concatenate(parts) for parts in group_old]
    new_values = [np.concatenate(parts) for parts in group_new]
    counts = tuple(int(values.size) for values in indices)
    if sum(counts) != int(np.count_nonzero(np.asarray(baseline) != np.asarray(target))):
        raise XOV1Error("B/H/W groups do not partition the cross-parent disagreement set")

    header = BHW_HEADER.pack(
        b"XBH1",
        1,
        0,
        0,
        *counts,
        bytes.fromhex(GF1_FIELD_SHA256),
        bytes.fromhex(AFR1_FIELD_SHA256),
        bytes.fromhex(GT_SEMANTIC_SHA256),
    )
    body = bytearray()
    for group_id in range(3):
        for index, old, new in zip(
            indices[group_id].tolist(),
            old_values[group_id].tolist(),
            new_values[group_id].tolist(),
            strict=True,
        ):
            body.extend(BHW_RECORD.pack(index, old, new))
    payload = header + bytes(body)
    payload_path = store / "retained/cross_bhw/cross_parent_bhw.records"
    repeat_path = store / "retained/cross_bhw/cross_parent_bhw.repeat.records"
    atomic_bytes(payload_path, payload)
    atomic_bytes(repeat_path, payload)

    # Exact parse-back: recover every record and re-evaluate its directed label.
    decoded = BHW_HEADER.unpack_from(payload)
    if decoded[:7] != (b"XBH1", 1, 0, 0, *counts):
        raise XOV1Error("B/H/W payload header drifted")
    if tuple(value.hex() for value in decoded[7:]) != (
        GF1_FIELD_SHA256,
        AFR1_FIELD_SHA256,
        GT_SEMANTIC_SHA256,
    ):
        raise XOV1Error("B/H/W payload source identities drifted")
    offset = BHW_HEADER.size
    flat_generated = baseline.reshape(-1)
    flat_afr1 = target.reshape(-1)
    flat_gt = gt.reshape(-1)
    for group_id, count in enumerate(counts):
        for _ in range(count):
            index, old, new = BHW_RECORD.unpack_from(payload, offset)
            offset += BHW_RECORD.size
            generated = int(flat_generated[index])
            afr1 = int(flat_afr1[index])
            wanted = int(flat_gt[index])
            expected = (
                (old, new) == (afr1, generated) and generated == wanted and afr1 != wanted
                if group_id == 0
                else (old, new) == (generated, afr1)
                and (
                    (afr1 == wanted and generated != wanted)
                    if group_id == 1
                    else (afr1 != wanted and generated != wanted)
                )
            )
            if not expected:
                raise XOV1Error("B/H/W payload record failed directed parse-back")
    if offset != len(payload):
        raise XOV1Error("B/H/W payload has trailing bytes")
    retained = [fact(payload_path), fact(repeat_path)]
    if retained[0]["sha256"] != retained[1]["sha256"]:
        raise XOV1Error("B/H/W deterministic repeat changed")
    receipt = {
        "schema": "ddm_xov1_cross_parent_bhw.v1",
        "complete": True,
        "disagreement_denominator": sum(counts),
        "semantic_site_denominator": FIELD_BYTES,
        "directed_rows": {
            "afr1_to_gf1": {
                "B_benefit": counts[0],
                "H_harm": counts[1],
                "W_wash": counts[2],
                "benefit_definition": "AFR1 != GT and GF1 == GT",
            },
            "gf1_to_afr1": {
                "B_benefit": counts[1],
                "H_harm": counts[0],
                "W_wash": counts[2],
                "benefit_definition": "GF1 != GT and AFR1 == GT",
            },
        },
        "candidate_for_scmdl": {
            "direction": "afr1_to_gf1",
            "record_group": 0,
            "coordinates": counts[0],
            "reason": (
                "only this directed subset is token-label-beneficial when importing the RATE "
                "parent into the DISTORTION parent; its joint RC64 price belongs to #1374"
            ),
        },
        "payload": retained[0],
        "repeat": retained[1],
        "deterministic_repeat_equal": True,
        "exact_record_parseback": True,
        "scorer_loaded": False,
        "retained": retained,
    }
    atomic_json(checkpoint, receipt)
    return receipt


def decode_gf1_generator_packet(hg1: Any, packet: bytes, output: Path) -> dict[str, Any]:
    """Decode GF1's four-stream generator packet without inventing a residual."""

    original_streams = hg1.STREAMS
    hg1.STREAMS = hg1.GENERATOR_STREAMS
    try:
        streams = hg1.parse_packet(packet)
    finally:
        hg1.STREAMS = original_streams
    return hg1.render_generators(streams, output)


def decode_gf1_packet(store: Path, hg1: Any) -> tuple[Path, dict[str, Any]]:
    output = store / "retained/gf1_packet_parseback.u8"
    checkpoint = store / "checkpoints/STAGE_01_GF1_PACKET_DECODE.json"
    resumed = checkpoint_valid(checkpoint)
    if resumed is not None:
        return output, resumed
    output.parent.mkdir(parents=True, exist_ok=True)
    decode_fact = decode_gf1_generator_packet(hg1, GF1_PACKET.read_bytes(), output)
    observed = fact(output)
    if observed["sha256"] != GF1_FIELD_SHA256 or decode_fact != observed:
        raise XOV1Error("GF1 packet did not decode to its pinned zero-residual baseline")
    receipt = {
        "schema": "ddm_xov1_qx_section_recode_gf1_decode.v1",
        "complete": True,
        "packet": fact(GF1_PACKET),
        "decoded_field": observed,
        "expected_retained_field": fact(GF1_FIELD),
        "bit_identity": True,
        "retained": [observed],
    }
    atomic_json(checkpoint, receipt)
    return output, receipt


def section_events(
    qx2: Any, baseline: np.memmap, target: np.memmap, start: int
) -> tuple[list[list[Any]], tuple[Any, ...], int]:
    by_pair: list[list[Any]] = []
    all_events: list[Any] = []
    for local_pair in range(SECTION_PAIRS):
        source = np.asarray(baseline[start + local_pair])
        wanted = np.asarray(target[start + local_pair])
        indices = np.flatnonzero((source != wanted).reshape(-1))
        pair_events = []
        source_flat = source.reshape(-1)
        target_flat = wanted.reshape(-1)
        for index in indices.tolist():
            row, col = divmod(index, WIDTH)
            pair_events.append(
                qx2.PartitionEvent(
                    local_pair,
                    row,
                    col,
                    int(target_flat[index]),
                    int(source_flat[index]),
                )
            )
        by_pair.append(pair_events)
        all_events.extend(pair_events)
    return by_pair, tuple(sorted(all_events)), len(all_events)


def run_section(
    store: Path,
    qx2: Any,
    baseline_path: Path,
    baseline: np.memmap,
    target: np.memmap,
    bits: np.ndarray,
    section_id: int,
) -> dict[str, Any]:
    checkpoint = store / f"checkpoints/STAGE_{section_id + 10:02d}_SECTION.json"
    resumed = checkpoint_valid(checkpoint)
    if resumed is not None:
        return resumed
    start = section_id * SECTION_PAIRS
    stop = start + SECTION_PAIRS
    by_pair, expected_events, event_count = section_events(qx2, baseline, target, start)
    qx2.N_PAIRS = SECTION_PAIRS
    qx2.EVENT_COUNT = event_count

    original_baseline_frame = qx2.baseline_frame

    def local_baseline_frame(path: Path, pair: int) -> np.ndarray:
        if path != baseline_path or not 0 <= pair < SECTION_PAIRS:
            raise XOV1Error("section baseline request escaped its pinned window")
        return baseline[start + pair]

    qx2.baseline_frame = local_baseline_frame
    try:
        raw = qx2.encode_boundary_enumerative(baseline_path, by_pair, sha256_file(baseline_path))
        decoded_events = tuple(qx2.decode_boundary_enumerative(raw, baseline_path))
    finally:
        qx2.baseline_frame = original_baseline_frame
    if decoded_events != expected_events:
        raise XOV1Error(f"section {section_id}: QX event receiver changed the event object")

    root = store / f"retained/sections/{section_id:02d}_{start:03d}_{stop:03d}"
    raw_path = root / "qx_boundary_enumerative.raw"
    atomic_bytes(raw_path, raw)
    coder_rows = []
    for codec in qx2.CODECS:
        coded = qx2.compress(codec, raw)
        repeat = qx2.compress(codec, raw)
        if coded != repeat or qx2.decompress(codec, coded) != raw:
            raise XOV1Error(f"section {section_id}/{codec}: coder repeat or parse-back failed")
        coded_path = root / f"qx_boundary_enumerative.{codec}.bin"
        repeat_path = root / f"qx_boundary_enumerative.{codec}.repeat.bin"
        atomic_bytes(coded_path, coded)
        atomic_bytes(repeat_path, repeat)
        coder_rows.append(
            {
                "codec": codec,
                "payload": fact(coded_path),
                "repeat": fact(repeat_path),
                "deterministic_repeat_equal": True,
                "raw_parseback_equal": True,
            }
        )
    winner = min(coder_rows, key=lambda row: (row["payload"]["bytes"], row["codec"]))

    decoded = np.asarray(baseline[start:stop]).copy()
    for event in decoded_events:
        decoded[event.pair, event.row, event.col] = event.target_class
    expected = np.asarray(target[start:stop])
    if not np.array_equal(decoded, expected):
        raise XOV1Error(f"section {section_id}: exact semantic parse-back failed")
    decoded_path = root / "semantic_parseback.u8"
    atomic_bytes(decoded_path, decoded.tobytes(order="C"))
    target_section_sha = sha256_bytes(expected.tobytes(order="C"))
    if sha256_file(decoded_path) != target_section_sha:
        raise XOV1Error(f"section {section_id}: persisted semantic parse-back changed")

    enum_values = qx2.ENUM_HEADER.unpack_from(raw)
    counts_len, ranks_len, residual_len = struct.unpack_from(">III", raw, qx2.ENUM_HEADER.size)
    shipped_bits = float(bits[start:stop].sum())
    retained = [fact(raw_path), fact(decoded_path)]
    for row in coder_rows:
        retained.extend([row["payload"], row["repeat"]])
    receipt = {
        "schema": "ddm_xov1_qx_section_recode_section.v1",
        "complete": True,
        "section_id": section_id,
        "pair_range_half_open": [start, stop],
        "semantic_sites": SECTION_PAIRS * HEIGHT * WIDTH,
        "mismatches": event_count,
        "mismatch_fraction": event_count / (SECTION_PAIRS * HEIGHT * WIDTH),
        "raw": fact(raw_path),
        "representation_anatomy": {
            "near_events": enum_values[5],
            "far_residual_events": enum_values[6],
            "enumerative_rank_bits": enum_values[7],
            "transition_count_stream_bytes": counts_len,
            "enumerative_rank_bytes": ranks_len,
            "far_residual_stream_bytes": residual_len,
            "outer_header_and_lengths_bytes": qx2.ENUM_HEADER.size + 12,
        },
        "coders": coder_rows,
        "winner_codec": winner["codec"],
        "winner_payload": winner["payload"],
        "semantic_parseback": fact(decoded_path),
        "target_section_sha256": target_section_sha,
        "bit_identity": True,
        "shipped_rc64_same_section": {
            "ledger_bits": shipped_bits,
            "fractional_bytes": shipped_bits / 8.0,
            "ceil_bytes_if_independently_framed": int(np.ceil(shipped_bits / 8.0)),
        },
        "retained": retained,
    }
    atomic_json(checkpoint, receipt)
    return receipt


def build_aggregate(
    store: Path,
    qx2: Any,
    hg1: Any,
    baseline_path: Path,
    baseline: np.memmap,
    sections: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = store / "checkpoints/STAGE_30_AGGREGATE.json"
    resumed = checkpoint_valid(checkpoint)
    if resumed is not None:
        return resumed
    packet = GF1_PACKET.read_bytes()
    header = AGGREGATE_HEADER.pack(
        b"XOV1",
        1,
        0,
        SECTION_COUNT,
        len(packet),
        FIELD_BYTES,
        bytes.fromhex(GF1_FIELD_SHA256),
        bytes.fromhex(AFR1_FIELD_SHA256),
    )
    rows = bytearray()
    bodies = bytearray(packet)
    for section in sections:
        codec = section["winner_codec"]
        raw_path = Path(section["raw"]["path"])
        coded_path = Path(section["winner_payload"]["path"])
        raw = raw_path.read_bytes()
        coded = coded_path.read_bytes()
        start, stop = section["pair_range_half_open"]
        rows.extend(
            AGGREGATE_ROW.pack(
                start,
                stop - start,
                CODEC_IDS[codec],
                len(raw),
                len(coded),
                bytes.fromhex(sha256_bytes(raw)),
                bytes.fromhex(sha256_bytes(coded)),
                zlib.crc32(coded) & 0xFFFFFFFF,
            )
        )
        bodies.extend(coded)
    payload = header + bytes(rows) + bytes(bodies)
    payload_repeat = header + bytes(rows) + bytes(bodies)
    if payload != payload_repeat:
        raise XOV1Error("aggregate deterministic repeat changed")
    aggregate_path = store / "retained/aggregate/xov1_gf1_qx_afr1.bin"
    repeat_path = store / "retained/aggregate/xov1_gf1_qx_afr1.repeat.bin"
    atomic_bytes(aggregate_path, payload)
    atomic_bytes(repeat_path, payload_repeat)

    # Receiver closure starts from the counted GF1 packet, not the retained
    # convenience field.  It then consumes every coded QX section.
    parseback_path = store / "retained/aggregate/semantic_parseback.u8"
    decode_gf1_generator_packet(hg1, packet, parseback_path)
    parsed = np.memmap(parseback_path, dtype=np.uint8, mode="r+", shape=(N_PAIRS, HEIGHT, WIDTH))
    offset = AGGREGATE_HEADER.size
    magic, version, flags, count, packet_len, field_len, baseline_sha, target_sha = (
        AGGREGATE_HEADER.unpack_from(payload)
    )
    if (
        magic,
        version,
        flags,
        count,
        packet_len,
        field_len,
        baseline_sha.hex(),
        target_sha.hex(),
    ) != (
        b"XOV1",
        1,
        0,
        SECTION_COUNT,
        len(packet),
        FIELD_BYTES,
        GF1_FIELD_SHA256,
        AFR1_FIELD_SHA256,
    ):
        raise XOV1Error("aggregate header identity changed")
    row_values = []
    for _ in range(count):
        row_values.append(AGGREGATE_ROW.unpack_from(payload, offset))
        offset += AGGREGATE_ROW.size
    decoded_packet = payload[offset : offset + packet_len]
    offset += packet_len
    if decoded_packet != packet:
        raise XOV1Error("aggregate GF1 packet changed")
    for start, pair_count, codec_id, raw_len, coded_len, raw_sha, coded_sha, crc in row_values:
        coded = payload[offset : offset + coded_len]
        offset += coded_len
        if (
            codec_id not in ID_CODECS
            or len(coded) != coded_len
            or sha256_bytes(coded) != coded_sha.hex()
            or zlib.crc32(coded) & 0xFFFFFFFF != crc
        ):
            raise XOV1Error("aggregate coded section integrity failed")
        raw = qx2.decompress(ID_CODECS[codec_id], coded)
        if len(raw) != raw_len or sha256_bytes(raw) != raw_sha.hex():
            raise XOV1Error("aggregate raw section integrity failed")
        qx2.N_PAIRS = pair_count
        event_count = struct.unpack_from(">I", raw, 8)[0]
        qx2.EVENT_COUNT = event_count
        original_baseline_frame = qx2.baseline_frame

        def local_baseline_frame(
            path: Path,
            pair: int,
            section_start: int = start,
            section_pair_count: int = pair_count,
        ) -> np.ndarray:
            if path != baseline_path or not 0 <= pair < section_pair_count:
                raise XOV1Error("aggregate section escaped its pinned baseline window")
            return baseline[section_start + pair]

        qx2.baseline_frame = local_baseline_frame
        try:
            events = qx2.decode_boundary_enumerative(raw, baseline_path)
        finally:
            qx2.baseline_frame = original_baseline_frame
        for event in events:
            parsed[start + event.pair, event.row, event.col] = event.target_class
    if offset != len(payload):
        raise XOV1Error("aggregate has trailing bytes")
    parsed.flush()
    del parsed
    parseback = fact(parseback_path)
    if parseback["sha256"] != AFR1_FIELD_SHA256:
        raise XOV1Error("aggregate receiver did not reproduce the exact AFR1 field")
    aggregate = fact(aggregate_path)
    repeat = fact(repeat_path)
    if aggregate["sha256"] != repeat["sha256"]:
        raise XOV1Error("persisted aggregate repeat changed")
    receipt = {
        "schema": "ddm_xov1_qx_section_recode_aggregate.v1",
        "complete": True,
        "aggregate": aggregate,
        "repeat": repeat,
        "deterministic_repeat_equal": True,
        "semantic_parseback": parseback,
        "bit_identity_to_afr1_field": True,
        "generator_packet_bytes": len(packet),
        "qx_section_headers_and_payloads_bytes": aggregate["bytes"] - len(packet),
        "candidate_joint_pool_bytes": aggregate["bytes"],
        "afr1_joint_pool_bytes": AFR1_JOINT_POOL_BYTES,
        "delta_vs_afr1_joint_pool_bytes": aggregate["bytes"] - AFR1_JOINT_POOL_BYTES,
        "strict_replacement_pool_max_exclusive": STRICT_REPLACEMENT_POOL_MAX_EXCLUSIVE,
        "delta_vs_largest_passing_integer_bytes": aggregate["bytes"]
        - (STRICT_REPLACEMENT_POOL_MAX_EXCLUSIVE - 1),
        "would_clear_fixed_distortion_sub012_byte_gate": aggregate["bytes"]
        < STRICT_REPLACEMENT_POOL_MAX_EXCLUSIVE,
        "retained": [aggregate, repeat, parseback],
    }
    atomic_json(checkpoint, receipt)
    return receipt


def run(store: Path) -> dict[str, Any]:
    if store.resolve() != STORE.resolve():
        raise XOV1Error(f"custody root is pinned to {STORE}")
    started = time.perf_counter()
    preflight = source_preflight(store)
    qx2 = importlib.import_module("experiments.ddm_qx2_events_section_redesign")
    hg1 = importlib.import_module("experiments.ddm_hg1_heterogeneous_analytic_generator_gate")
    baseline_path, gf1_decode = decode_gf1_packet(store, hg1)
    baseline = np.memmap(baseline_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    target = np.memmap(AFR1_FIELD, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    bits = np.load(AFR1_BITS, allow_pickle=False)
    sections = []
    for section_id in range(SECTION_COUNT):
        sections.append(run_section(store, qx2, baseline_path, baseline, target, bits, section_id))
        print(
            json.dumps(
                {
                    "section": section_id,
                    "mismatches": sections[-1]["mismatches"],
                    "winner": sections[-1]["winner_codec"],
                    "winner_bytes": sections[-1]["winner_payload"]["bytes"],
                    "rc64_fractional_bytes": sections[-1]["shipped_rc64_same_section"][
                        "fractional_bytes"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    aggregate = build_aggregate(store, qx2, hg1, baseline_path, baseline, sections)
    cross_bhw = build_cross_parent_bhw(store, baseline, target)
    total_mismatches = sum(row["mismatches"] for row in sections)
    result = {
        "schema": "ddm_xov1_qx_section_recode.v1",
        "complete": True,
        "verdict": "PASS" if aggregate["would_clear_fixed_distortion_sub012_byte_gate"] else "OVER",
        "verdict_scope": (
            "FORMULATION: GF1 decoded generator field plus QX2 boundary-enumerative exact residual "
            "to the pinned AFR1 semantic field; scorer-free, not a family theorem"
        ),
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "preflight": preflight,
        "gf1_packet_decode": gf1_decode,
        "sections": sections,
        "aggregate": aggregate,
        "cross_parent_bhw": cross_bhw,
        "denominators": {
            "pairs": N_PAIRS,
            "sections": SECTION_COUNT,
            "semantic_sites": FIELD_BYTES,
            "mismatches": total_mismatches,
            "mismatch_fraction": total_mismatches / FIELD_BYTES,
        },
        "authority_boundaries": {
            "scorers_loaded": 0,
            "contest_eval_invocations": 0,
            "modal_invocations": 0,
            "upstream_writes": 0,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(store / "RESULT.json", result)
    manifest = {
        "schema": "ddm_xov1_qx_section_recode_manifest.v1",
        "complete": True,
        "result": fact(store / "RESULT.json"),
        "source": fact(Path(__file__).resolve()),
        "command": f"{Path(os.sys.executable).resolve()} {Path(__file__).resolve()} --resume-from {store}",
        "retention": (
            "GF1 packet parse-back, all ten raw QX forms, every real-coder candidate and repeat, "
            "all section semantic parse-backs, aggregate payload/repeat, and aggregate semantic "
            "parse-back are retained"
        ),
        "cleanup": "none fired",
    }
    atomic_json(store / "RUN_MANIFEST.json", manifest)
    atomic_json(
        store / "checkpoints/STAGE_40_COMPLETE.json",
        {
            "schema": "ddm_xov1_qx_section_recode_complete.v1",
            "complete": True,
            "verdict": result["verdict"],
            "result": fact(store / "RESULT.json"),
            "retained": [fact(store / "RESULT.json"), fact(store / "RUN_MANIFEST.json")],
        },
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, default=STORE)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args().resume_from)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "aggregate_bytes": result["aggregate"]["candidate_joint_pool_bytes"],
                "delta_vs_afr1_joint_pool_bytes": result["aggregate"][
                    "delta_vs_afr1_joint_pool_bytes"
                ],
                "bit_identity": result["aggregate"]["bit_identity_to_afr1_field"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
