#!/usr/bin/env python3
"""Scorer-free exact rate-shape probes for the WX1 divergent-object hunt.

The runner never treats a hypothetical changed core as a complete archive.  It
retains the hypothetical conditioning field, every raw/coded residual payload,
deterministic coder repeats, and exact parse-back records.  Only same-core
alphabet controls are assembled into receiver-valid QXE/ZIP candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import struct
import subprocess
import time
import zipfile
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np
from scipy import ndimage

STORE: Final = Path("/Volumes/APDataStore/pact/ddm_wx1_divergent_object_hunt/v5")
BASELINE: Final = Path(
    "/Volumes/APDataStore/pact/ddm_qx1_qxo1_target_overwrite_grammar/retained/derived/"
    "qx1_decoder_baseline.u8"
)
TARGET: Final = Path(
    "/Volumes/APDataStore/pact/ddm_qx1_qxo1_target_overwrite_grammar/retained/"
    "semantic_proof/full_target_implicit.u8"
)
EVENTS: Final = Path(
    "/Volumes/APDataStore/pact/ddm_qx1_qxo1_target_overwrite_grammar/retained/"
    "grammar_v1/verification_expected_overwrites.u32be_target_u8"
)
GT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
)
CORE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qx1/retained/envelopes/"
    "core_without_events_exceptions/envelope.qxe"
)

PINS: Final = {
    BASELINE: "afeb8c94d5181b03992aefad1daef49ee7aaf1f768d11aa5964dacbfa1e22dbd",
    TARGET: "9079929d004cc9638a80159d61371c2982c198f0eb2b19eac4084da981ababc7",
    EVENTS: "5bf9c95058ae1df25de6305d1077853dcc04dede7a32f0096a19b5e3048370ef",
    GT: "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248",
    CORE: "4e6a2f6669c590258fc6c5d194ae6cb30951f5881e2055761de0bff753bdfb95",
}

N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
FRAME_SITES: Final = HEIGHT * WIDTH
SITES: Final = N_PAIRS * FRAME_SITES
EVENT_COUNT: Final = 8_749
QX_CORE_ARCHIVE_BYTES: Final = 113_844
QXE_SECTION_BYTES: Final = 48
ARCHIVE_GATE_EXCLUSIVE: Final = 137_986
MIN_FREE_BYTES: Final = 3_000_000_000
AXIS: Final = "[scorer-free exact receiver/rate and DALI-token-agreement measurement]"

EVENT_RECORD: Final = struct.Struct(">IB")
GRAMMAR_HEADER: Final = struct.Struct(">4sBBBBIIIII32s")
QXE_HEADER: Final = struct.Struct(">4sBBH")
QXE_SECTION: Final = struct.Struct(">BBHII32sI")
MAGIC: Final = b"WXG1"
VERSION: Final = 1

ALPHABETS: Final = {
    "target_boundary": 1,
    "target_raster": 2,
    "target_source_boundary": 3,
    "target_source_target_distance": 4,
}
ALPHABET_NAMES: Final = {value: key for key, value in ALPHABETS.items()}
VARIANT_IDS: Final = {
    "same_core": 0,
    "absorb_gap_top25": 1,
    "absorb_gap_top50": 2,
    "absorb_target_lane": 3,
}
CODECS: Final = {"brotli_q11": 1, "lzma9e": 2, "zlib9": 3}
CODEC_NAMES: Final = {value: key for key, value in CODECS.items()}


class WX1Error(RuntimeError):
    """A custody, exactness, retention, or typing gate failed closed."""


@dataclass(frozen=True, order=True)
class Event:
    site: int
    target: int

    @property
    def pair(self) -> int:
        return self.site // FRAME_SITES

    @property
    def local(self) -> int:
        return self.site % FRAME_SITES


def sha256_bytes(payload: bytes | memoryview) -> str:
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
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")


def verify_fact(row: dict[str, Any], label: str) -> None:
    path = Path(row["path"])
    if not path.is_file() or fact(path) != row:
        raise WX1Error(f"{label} retained fact drifted: {path}")


def require_pins() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path, expected in PINS.items():
        if not path.is_file():
            raise WX1Error(f"required input is absent: {path}")
        row = fact(path)
        if row["sha256"] != expected:
            raise WX1Error(f"input SHA drifted: {path}: {row['sha256']} != {expected}")
        rows[str(path)] = row
    return rows


def read_events() -> tuple[Event, ...]:
    payload = EVENTS.read_bytes()
    if len(payload) % EVENT_RECORD.size:
        raise WX1Error("overwrite record payload is truncated")
    events = tuple(
        Event(*EVENT_RECORD.unpack_from(payload, offset))
        for offset in range(0, len(payload), EVENT_RECORD.size)
    )
    if len(events) != EVENT_COUNT or tuple(sorted(events)) != events:
        raise WX1Error("overwrite event identity/order drifted")
    if len({event.site for event in events}) != EVENT_COUNT:
        raise WX1Error("overwrite event sites are not unique")
    return events


def serialize_events(events: Iterable[Event]) -> bytes:
    return b"".join(EVENT_RECORD.pack(event.site, event.target) for event in events)


def write_uleb(value: int, output: bytearray) -> int:
    if value < 0:
        raise WX1Error("negative ULEB value")
    start = len(output)
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return len(output) - start


def read_uleb(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload) or shift > 35:
            raise WX1Error("invalid ULEB stream")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def boundary_distance(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:-1] |= labels[:-1] != labels[1:]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return ndimage.distance_transform_cdt(~boundary, metric="chessboard").astype(np.int16)


def frame_groups(labels: np.ndarray, alphabet: str) -> list[tuple[int, np.ndarray]]:
    flat = labels.reshape(-1)
    raster = np.arange(FRAME_SITES, dtype=np.int32)
    groups: list[tuple[int, np.ndarray]] = []
    if alphabet in {"target_boundary", "target_source_boundary"}:
        distance = boundary_distance(labels).reshape(-1)
        base_order = np.lexsort((raster, distance)).astype(np.int32, copy=False)
    elif alphabet == "target_raster":
        base_order = raster
    else:
        base_order = np.empty(0, dtype=np.int32)

    if alphabet in {"target_boundary", "target_raster"}:
        for target in range(5):
            groups.append((target, base_order[flat[base_order] != target]))
        return groups

    for target in range(5):
        if alphabet == "target_source_target_distance":
            target_distance = ndimage.distance_transform_cdt(
                labels != target, metric="chessboard"
            ).reshape(-1)
            order = np.lexsort((raster, target_distance)).astype(np.int32, copy=False)
        elif alphabet == "target_source_boundary":
            order = base_order
        else:
            raise WX1Error(f"unknown alphabet: {alphabet}")
        for source in range(5):
            if source != target:
                groups.append((target, order[flat[order] == source]))
    return groups


def split_by_pair(events: Sequence[Event]) -> list[list[Event]]:
    result: list[list[Event]] = [[] for _ in range(N_PAIRS)]
    for event in events:
        result[event.pair].append(event)
    return result


def encode_grammar(
    conditioning_path: Path,
    events: Sequence[Event],
    variant: str,
    alphabet: str,
) -> tuple[bytes, dict[str, Any], dict[int, int]]:
    by_pair = split_by_pair(events)
    conditioning = np.memmap(
        conditioning_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH)
    )
    body = bytearray()
    count_bytes = 0
    gap_bytes = 0
    per_event_gap_bytes: dict[int, int] = {}
    group_count = 0
    for pair in range(N_PAIRS):
        labels = np.asarray(conditioning[pair])
        pair_events = by_pair[pair]
        for target, eligible in frame_groups(labels, alphabet):
            group_count += 1
            inverse = np.empty(FRAME_SITES, dtype=np.int32)
            inverse.fill(-1)
            inverse[eligible] = np.arange(eligible.size, dtype=np.int32)
            selected: list[tuple[int, int]] = []
            for event in pair_events:
                if event.target != target:
                    continue
                rank = int(inverse[event.local])
                if rank >= 0:
                    selected.append((rank, event.site))
            selected.sort()
            count_bytes += write_uleb(len(selected), body)
            previous = -1
            for rank, site in selected:
                width = write_uleb(rank - previous - 1, body)
                gap_bytes += width
                per_event_gap_bytes[site] = width
                previous = rank
    del conditioning
    if len(per_event_gap_bytes) != len(events):
        raise WX1Error(
            f"alphabet {alphabet} encoded {len(per_event_gap_bytes)} / {len(events)} events"
        )
    header = GRAMMAR_HEADER.pack(
        MAGIC,
        VERSION,
        VARIANT_IDS[variant],
        ALPHABETS[alphabet],
        0,
        N_PAIRS,
        HEIGHT,
        WIDTH,
        len(events),
        len(body),
        bytes.fromhex(sha256_file(conditioning_path)),
    )
    return header + bytes(body), {
        "header_bytes": len(header),
        "body_bytes": len(body),
        "count_bytes": count_bytes,
        "rank_gap_bytes": gap_bytes,
        "fixed_group_count": group_count,
        "event_count": len(events),
    }, per_event_gap_bytes


def decode_grammar(raw: bytes, conditioning_path: Path) -> tuple[Event, ...]:
    if len(raw) < GRAMMAR_HEADER.size:
        raise WX1Error("grammar is truncated")
    (
        magic,
        version,
        variant_id,
        alphabet_id,
        reserved,
        pairs,
        height,
        width,
        event_count,
        body_len,
        conditioning_sha,
    ) = GRAMMAR_HEADER.unpack_from(raw)
    if (
        magic != MAGIC
        or version != VERSION
        or variant_id not in VARIANT_IDS.values()
        or alphabet_id not in ALPHABET_NAMES
        or reserved
        or (pairs, height, width) != (N_PAIRS, HEIGHT, WIDTH)
        or GRAMMAR_HEADER.size + body_len != len(raw)
        or conditioning_sha.hex() != sha256_file(conditioning_path)
    ):
        raise WX1Error("grammar identity/geometry/input pin failed")
    alphabet = ALPHABET_NAMES[alphabet_id]
    conditioning = np.memmap(
        conditioning_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH)
    )
    body = raw[GRAMMAR_HEADER.size :]
    offset = 0
    events: list[Event] = []
    for pair in range(N_PAIRS):
        labels = np.asarray(conditioning[pair])
        seen: set[int] = set()
        for target, eligible in frame_groups(labels, alphabet):
            count, offset = read_uleb(body, offset)
            previous = -1
            for _ in range(count):
                delta, offset = read_uleb(body, offset)
                rank = previous + delta + 1
                if rank >= eligible.size:
                    raise WX1Error("rank exceeds decoder-derived alphabet")
                local = int(eligible[rank])
                if local in seen:
                    raise WX1Error("grammar writes a frame site twice")
                seen.add(local)
                events.append(Event(pair * FRAME_SITES + local, target))
                previous = rank
    del conditioning
    if offset != len(body) or len(events) != event_count:
        raise WX1Error("decoder did not consume exact body/event denominator")
    return tuple(sorted(events))


def compress(codec: str, raw: bytes) -> bytes:
    if codec == "brotli_q11":
        return brotli.compress(raw, quality=11)
    if codec == "lzma9e":
        return lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    if codec == "zlib9":
        return zlib.compress(raw, level=9)
    raise WX1Error(f"unknown codec: {codec}")


def decompress(codec: str, coded: bytes) -> bytes:
    if codec == "brotli_q11":
        return brotli.decompress(coded)
    if codec == "lzma9e":
        return lzma.decompress(coded)
    if codec == "zlib9":
        return zlib.decompress(coded)
    raise WX1Error(f"unknown codec: {codec}")


def split_core_records(core: bytes) -> list[bytes]:
    if len(core) < QXE_HEADER.size or QXE_HEADER.unpack_from(core) != (b"QXE1", 1, 0, 7):
        raise WX1Error("QXE core header drifted")
    offset = QXE_HEADER.size
    records: list[bytes] = []
    for expected_id in range(1, 8):
        start = offset
        section_id, codec_id, reserved, raw_len, coded_len, raw_sha, crc = QXE_SECTION.unpack_from(
            core, offset
        )
        offset += QXE_SECTION.size
        coded = core[offset : offset + coded_len]
        offset += coded_len
        if section_id != expected_id or codec_id not in CODEC_NAMES or reserved:
            raise WX1Error("QXE core section roster drifted")
        raw = decompress(CODEC_NAMES[codec_id], coded)
        if (
            len(raw) != raw_len
            or sha256_bytes(raw) != raw_sha.hex()
            or zlib.crc32(coded) & 0xFFFFFFFF != crc
        ):
            raise WX1Error("QXE core section integrity failed")
        records.append(core[start:offset])
    if offset != len(core):
        raise WX1Error("QXE core has trailing bytes")
    return records


def build_archive(core: bytes, raw: bytes, coded: bytes, codec: str) -> bytes:
    section = QXE_SECTION.pack(
        8,
        CODECS[codec],
        0,
        len(raw),
        len(coded),
        bytes.fromhex(sha256_bytes(raw)),
        zlib.crc32(coded) & 0xFFFFFFFF,
    ) + coded
    packet = QXE_HEADER.pack(b"QXE1", 1, 0, 8) + b"".join(split_core_records(core)) + section
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        info = zipfile.ZipInfo("state/qx1.qxe", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, packet)
    return output.getvalue()


def retain_coders(root: Path, raw: bytes) -> list[dict[str, Any]]:
    raw_path = root / "raw.wxg"
    atomic_bytes(raw_path, raw)
    rows: list[dict[str, Any]] = []
    for codec in CODECS:
        coded = compress(codec, raw)
        repeat = compress(codec, raw)
        coded_path = root / f"candidate.{codec}.bin"
        repeat_path = root / f"candidate.{codec}.repeat.bin"
        atomic_bytes(coded_path, coded)
        atomic_bytes(repeat_path, repeat)
        if coded != repeat or decompress(codec, coded) != raw:
            raise WX1Error(f"{codec} repeat or parse-back failed")
        rows.append(
            {
                "codec": codec,
                "payload": fact(coded_path),
                "repeat": fact(repeat_path),
                "deterministic_repeat": True,
                "parseback_exact": True,
            }
        )
    return rows


def write_conditioning(absorbed: Sequence[Event], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    source = np.memmap(BASELINE, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    with partial.open("wb") as handle:
        handle.truncate(SITES)
    target = np.memmap(partial, dtype=np.uint8, mode="r+", shape=(N_PAIRS, HEIGHT, WIDTH))
    for pair in range(N_PAIRS):
        target[pair] = source[pair]
    for event in absorbed:
        target.reshape(-1)[event.site] = event.target
    target.flush()
    del source, target
    os.replace(partial, output_path)
    return fact(output_path)


def verify_application(
    conditioning_path: Path, decoded: Sequence[Event], output_path: Path
) -> dict[str, Any]:
    source = np.memmap(conditioning_path, dtype=np.uint8, mode="r", shape=(SITES,))
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    with partial.open("wb") as handle:
        handle.truncate(SITES)
    output = np.memmap(partial, dtype=np.uint8, mode="r+", shape=(SITES,))
    block = 8 << 20
    for start in range(0, SITES, block):
        output[start : start + block] = source[start : start + block]
    for event in decoded:
        output[event.site] = event.target
    output.flush()
    del source, output
    os.replace(partial, output_path)
    row = fact(output_path)
    if row["sha256"] != PINS[TARGET]:
        raise WX1Error(f"decoded field is not the pinned QXO target: {row['sha256']}")
    return row


def preflight(resume_from: Path) -> dict[str, Any]:
    if resume_from.resolve() != STORE.resolve():
        raise WX1Error(f"--resume-from must be the pinned store {STORE}")
    STORE.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(STORE).free < MIN_FREE_BYTES:
        raise WX1Error("storage preflight failed below 3 GB free")
    runner = fact(Path(__file__).resolve())
    checkpoint = STORE / "checkpoints/STAGE0_PREFLIGHT.json"
    if checkpoint.is_file():
        row = json.loads(checkpoint.read_text())
        if row["runner"] != runner:
            raise WX1Error("preflight checkpoint belongs to a different runner source")
        for input_row in row["inputs"].values():
            verify_fact(input_row, "preflight input")
        return row
    row = {
        "schema": "ddm_wx1_preflight.v1",
        "axis": AXIS,
        "runner": runner,
        "inputs": require_pins(),
        "store": str(STORE),
        "free_bytes_at_launch": shutil.disk_usage(STORE).free,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "selection_mode": "full population; no sampled verdict",
    }
    atomic_json(checkpoint, row)
    return row


def stage1_token_agreement(events: Sequence[Event]) -> dict[str, Any]:
    checkpoint = STORE / "checkpoints/STAGE1_TOKEN_AGREEMENT.json"
    if checkpoint.is_file():
        return json.loads(checkpoint.read_text())
    baseline = np.memmap(BASELINE, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    target = np.memmap(TARGET, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    gt = np.load(GT, mmap_mode="r")
    if gt.shape != (N_PAIRS, HEIGHT, WIDTH) or gt.dtype != np.uint8:
        raise WX1Error(f"DALI GT geometry drifted: {gt.shape} {gt.dtype}")
    counts = {"benefit": 0, "harm": 0, "wrong_to_wrong": 0}
    transitions: dict[str, int] = {}
    records = bytearray()
    baseline_errors = 0
    target_errors = 0
    per_pair: list[dict[str, int]] = []
    for pair in range(N_PAIRS):
        base_frame = np.asarray(baseline[pair])
        target_frame = np.asarray(target[pair])
        gt_frame = np.asarray(gt[pair])
        base_count = int(np.count_nonzero(base_frame != gt_frame))
        target_count = int(np.count_nonzero(target_frame != gt_frame))
        baseline_errors += base_count
        target_errors += target_count
        per_pair.append({"pair": pair, "baseline_errors": base_count, "target_errors": target_count})
    flat_base = baseline.reshape(-1)
    flat_target = target.reshape(-1)
    flat_gt = gt.reshape(-1)
    class_code = {"benefit": 1, "harm": 2, "wrong_to_wrong": 3}
    for event in events:
        base = int(flat_base[event.site])
        changed = int(flat_target[event.site])
        truth = int(flat_gt[event.site])
        if changed != event.target or base == changed:
            raise WX1Error("event disagrees with baseline/target fields")
        if base != truth and changed == truth:
            label = "benefit"
        elif base == truth and changed != truth:
            label = "harm"
        else:
            label = "wrong_to_wrong"
        counts[label] += 1
        transitions[f"{base}->{changed}"] = transitions.get(f"{base}->{changed}", 0) + 1
        records.extend(struct.pack(">IBBBB", event.site, base, changed, truth, class_code[label]))
    del baseline, target, gt
    records_path = STORE / "retained/token_agreement/events.u32be_base_target_gt_class.u8x4"
    atomic_bytes(records_path, bytes(records))
    per_pair_path = STORE / "retained/token_agreement/per_pair.json"
    atomic_json(per_pair_path, per_pair)
    row = {
        "schema": "ddm_wx1_token_agreement.v1",
        "axis": AXIS,
        "denominator_sites": SITES,
        "denominator_changed_sites": len(events),
        "baseline_native_dali_errors": baseline_errors,
        "target_native_dali_errors": target_errors,
        "native_error_delta": target_errors - baseline_errors,
        "changed_site_classes": counts,
        "transition_counts": dict(sorted(transitions.items())),
        "retained_records": fact(records_path),
        "retained_per_pair": fact(per_pair_path),
        "realized_seg_or_pose_measured": False,
    }
    atomic_json(checkpoint, row)
    return row


def retain_one_form(
    variant: str,
    alphabet: str,
    conditioning_path: Path,
    events: Sequence[Event],
    complete_same_core: bool,
) -> tuple[dict[str, Any], dict[int, int]]:
    root = STORE / "retained/rate_shape" / variant / alphabet
    raw, anatomy, gap_costs = encode_grammar(conditioning_path, events, variant, alphabet)
    decoded = decode_grammar(raw, conditioning_path)
    if decoded != tuple(sorted(events)):
        raise WX1Error(f"{variant}/{alphabet}: decoded events differ")
    expected_path = root / "expected.u32be_target_u8"
    decoded_path = root / "decoded.u32be_target_u8"
    atomic_bytes(expected_path, serialize_events(sorted(events)))
    atomic_bytes(decoded_path, serialize_events(decoded))
    if expected_path.read_bytes() != decoded_path.read_bytes():
        raise WX1Error(f"{variant}/{alphabet}: retained parse-back differs")
    coder_rows = retain_coders(root, raw)
    winner = min(coder_rows, key=lambda item: (item["payload"]["bytes"], item["codec"]))
    row: dict[str, Any] = {
        "variant": variant,
        "alphabet": alphabet,
        "conditioning": fact(conditioning_path),
        "anatomy": anatomy,
        "raw": fact(root / "raw.wxg"),
        "expected_events": fact(expected_path),
        "decoded_events": fact(decoded_path),
        "parseback_exact": True,
        "coders": coder_rows,
        "winner": winner,
        "complete_archive": None,
    }
    if complete_same_core:
        core = CORE.read_bytes()
        coded = Path(winner["payload"]["path"]).read_bytes()
        archive = build_archive(core, raw, coded, winner["codec"])
        repeat = build_archive(core, raw, coded, winner["codec"])
        archive_path = root / "archive.zip"
        repeat_path = root / "archive.repeat.zip"
        atomic_bytes(archive_path, archive)
        atomic_bytes(repeat_path, repeat)
        if archive != repeat:
            raise WX1Error(f"{variant}/{alphabet}: archive repeat differs")
        expected_bytes = QX_CORE_ARCHIVE_BYTES + QXE_SECTION_BYTES + len(coded)
        if len(archive) != expected_bytes:
            raise WX1Error(f"archive arithmetic drifted: {len(archive)} != {expected_bytes}")
        row["complete_archive"] = {
            "archive": fact(archive_path),
            "repeat": fact(repeat_path),
            "receiver_valid_section_envelope": True,
            "deterministic_repeat": True,
            "under_gate": len(archive) < ARCHIVE_GATE_EXCLUSIVE,
        }
    return row, gap_costs


def stage2_same_core(events: Sequence[Event]) -> tuple[dict[str, Any], dict[int, int]]:
    checkpoint = STORE / "checkpoints/STAGE2_SAME_CORE_ALPHABETS.json"
    if checkpoint.is_file():
        row = json.loads(checkpoint.read_text())
        gap_rows = row["target_boundary_gap_width_by_site"]
        return row, {int(site): int(width) for site, width in gap_rows.items()}
    forms: list[dict[str, Any]] = []
    boundary_costs: dict[int, int] = {}
    for alphabet in ALPHABETS:
        form, gap_costs = retain_one_form(
            "same_core", alphabet, BASELINE, events, complete_same_core=True
        )
        forms.append(form)
        if alphabet == "target_boundary":
            boundary_costs = gap_costs
    row = {
        "schema": "ddm_wx1_same_core_alphabet_race.v1",
        "axis": AXIS,
        "forms": forms,
        "winner": min(
            forms,
            key=lambda item: (
                item["complete_archive"]["archive"]["bytes"], item["alphabet"]
            ),
        ),
        "target_boundary_gap_width_by_site": {
            str(site): width for site, width in sorted(boundary_costs.items())
        },
    }
    atomic_json(checkpoint, row)
    return row, boundary_costs


def select_absorbed(
    variant: str, events: Sequence[Event], gap_costs: dict[int, int]
) -> tuple[Event, ...]:
    if variant == "absorb_target_lane":
        return tuple(event for event in events if event.target == 1)
    fraction = {"absorb_gap_top25": 0.25, "absorb_gap_top50": 0.50}[variant]
    count = round(len(events) * fraction)
    ranked = sorted(events, key=lambda event: (-gap_costs[event.site], event.site))
    return tuple(sorted(ranked[:count]))


def stage3_changed_core(
    events: Sequence[Event], gap_costs: dict[int, int]
) -> dict[str, Any]:
    checkpoint = STORE / "checkpoints/STAGE3_CHANGED_CORE_CONDITIONALS.json"
    if checkpoint.is_file():
        return json.loads(checkpoint.read_text())
    rows: list[dict[str, Any]] = []
    event_by_site = {event.site: event for event in events}
    for variant in ("absorb_gap_top25", "absorb_gap_top50", "absorb_target_lane"):
        variant_checkpoint = STORE / f"checkpoints/STAGE3_{variant.upper()}.json"
        if variant_checkpoint.is_file():
            rows.append(json.loads(variant_checkpoint.read_text()))
            continue
        absorbed = select_absorbed(variant, events, gap_costs)
        absorbed_sites = {event.site for event in absorbed}
        remaining = tuple(event for event in events if event.site not in absorbed_sites)
        if len(absorbed) + len(remaining) != len(events):
            raise WX1Error("changed-core event partition failed")
        if any(event_by_site[event.site] != event for event in absorbed):
            raise WX1Error("changed-core absorbed event identity failed")
        variant_root = STORE / "retained/changed_core" / variant
        conditioning_path = variant_root / "hypothetical_core_field.u8"
        conditioning_fact = write_conditioning(absorbed, conditioning_path)
        absorbed_path = variant_root / "absorbed.u32be_target_u8"
        remaining_path = variant_root / "remaining.u32be_target_u8"
        atomic_bytes(absorbed_path, serialize_events(absorbed))
        atomic_bytes(remaining_path, serialize_events(remaining))
        forms: list[dict[str, Any]] = []
        for alphabet in ("target_boundary", "target_source_target_distance"):
            form, _ = retain_one_form(
                variant, alphabet, conditioning_path, remaining, complete_same_core=False
            )
            forms.append(form)
        winner = min(forms, key=lambda item: (item["winner"]["payload"]["bytes"], item["alphabet"]))
        decoded = decode_grammar(
            Path(winner["raw"]["path"]).read_bytes(), conditioning_path
        )
        output_fact = verify_application(
            conditioning_path, decoded, variant_root / "decoded_target_field.u8"
        )
        residual_bytes = winner["winner"]["payload"]["bytes"]
        implied_budget = ARCHIVE_GATE_EXCLUSIVE - 1 - QXE_SECTION_BYTES - residual_bytes
        variant_row = {
                "variant": variant,
                "absorbed_events": len(absorbed),
                "remaining_events": len(remaining),
                "conditioning": conditioning_fact,
                "absorbed_records": fact(absorbed_path),
                "remaining_records": fact(remaining_path),
                "forms": forms,
                "winner": winner,
                "decoded_target": output_fact,
                "conditional_exactness": "exact if a new core emits the retained conditioning field",
                "new_core_packet_bytes": "UNKNOWN_NOT_MATERIALIZED",
                "realized_distortion": "UNKNOWN_NOT_MEASURED",
                "maximum_total_core_archive_bytes_to_hold_strict_gate": implied_budget,
                "complete_archive_claim": False,
            }
        atomic_json(variant_checkpoint, variant_row)
        rows.append(variant_row)
    row = {
        "schema": "ddm_wx1_changed_core_conditionals.v1",
        "axis": AXIS,
        "rows": rows,
        "honesty_boundary": (
            "These are exact residual coders and exact conditional parse-backs, not archives. "
            "Each row changes the core field; its new model bytes and realized distortion are unknown."
        ),
    }
    atomic_json(checkpoint, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    stage0 = preflight(args.resume_from)
    events = read_events()
    stage1 = stage1_token_agreement(events)
    stage2, gap_costs = stage2_same_core(events)
    stage3 = stage3_changed_core(events, gap_costs)
    result = {
        "schema": "ddm_wx1_divergent_object_hunt.v1",
        "axis": AXIS,
        "score_claim": False,
        "scorer_used": False,
        "pointer_moved": False,
        "stage0": stage0,
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "elapsed_seconds_this_invocation": time.time() - started,
    }
    atomic_json(STORE / "RESULT.json", result)
    atomic_json(
        STORE / "RUN_MANIFEST.json",
        {
            "schema": "ddm_wx1_run_manifest.v1",
            "command": [str(Path(__file__).resolve()), "--resume-from", str(STORE)],
            "result": fact(STORE / "RESULT.json"),
            "runner": fact(Path(__file__).resolve()),
            "stages": [
                fact(STORE / "checkpoints/STAGE0_PREFLIGHT.json"),
                fact(STORE / "checkpoints/STAGE1_TOKEN_AGREEMENT.json"),
                fact(STORE / "checkpoints/STAGE2_SAME_CORE_ALPHABETS.json"),
                fact(STORE / "checkpoints/STAGE3_ABSORB_GAP_TOP25.json"),
                fact(STORE / "checkpoints/STAGE3_ABSORB_GAP_TOP50.json"),
                fact(STORE / "checkpoints/STAGE3_ABSORB_TARGET_LANE.json"),
                fact(STORE / "checkpoints/STAGE3_CHANGED_CORE_CONDITIONALS.json"),
            ],
        },
    )
    print(json.dumps({"result": fact(STORE / "RESULT.json")}, indent=2))


if __name__ == "__main__":
    main()
