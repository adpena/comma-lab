#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""bd1 scorer-free Road/Lane class-field receiver closure for qo1.

This builds a counted, tagged ``BD1CLF1`` section from the cached n600
``lstars`` target labels, appends it to the live qo1 IX2 archive, and proves the
receiver can parse and consume it without running SegNet or PoseNet.

The section is video-derived payload and is therefore counted in ``archive.zip``.
The decode code carries only a generic parser/coder and a fixed Road/Lane paint
operation.  No scorer weights, GT tables, or cached argmax arrays are hidden in
the receiver source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np
from scipy import ndimage

_REPO: Final = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO / "experiments"))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_ix2_archive_container import (  # noqa: E402
    build_payload,
    build_single_member_zip,
    parse_payload,
)


BD1_MAGIC: Final = b"BD1CLF1!"
BD1_VERSION: Final = 1
BD1_HEADER: Final = struct.Struct("<8sBHHHBBBBBII32s")
BD1_RAW: Final = 0
BD1_LZMA1_RAW: Final = 1
BD1_BROTLI_Q11: Final = 2
BD1_SMEVR_R7_NIBBLE: Final = 3
CODEC_IDS: Final = {
    "lzma1-raw": BD1_LZMA1_RAW,
    "brotli-q11": BD1_BROTLI_Q11,
    "smevr-r7-nibble": BD1_SMEVR_R7_NIBBLE,
}
ID_CODECS: Final = {value: key for key, value in CODEC_IDS.items()}
LZMA_FILTERS: Final = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}
]

SEG_H: Final = 384
SEG_W: Final = 512
N_PAIRS: Final = 600
ROAD: Final = 0
LANE: Final = 1
RATE_DENOM: Final = 37_545_489
BASELINE_S: Final = 0.7539807296911207
BASELINE_BYTES: Final = 357_836
BASELINE_AXIS: Final = "[macOS-CPU advisory]"
BASELINE_ARCHIVE_SHA256: Final = (
    "d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a"
)
BASE_RAW_SHA256: Final = (
    "3ce7d269a7080a4024a576694cd0ddc697099c64cd02fdd2bb879339e4b03f31"
)
BASE_RAW_BYTES: Final = 3_662_409_600

DEFAULT_BASE_SUB: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit")
DEFAULT_GT_CACHE: Final = _REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_RESEARCH_DIR: Final = _REPO / ".omx/research/ddm_bd1_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_bd1_20260805")
DEFAULT_CANDIDATE_DIR: Final = DEFAULT_SSD_DIR / "sub_auto_pairbit_class_field"
DEFAULT_IDENTITY_DIR: Final = DEFAULT_SSD_DIR / "qo1_identity_extended_receiver"

SE3_SIDE_BYTES: Final = 81_365
SE3_EXPLICIT_BYTES: Final = 100_904
SE3_CAPTURED_FLIPS: Final = 161_660
CQ2_THRESHOLDS: Final = {
    "25KB_student_side_implied": 0.516810,
    "25KB_student_explicit_direction": 0.611747,
    "75KB_student_side_implied": 0.759752,
    "75KB_student_explicit_direction": 0.854688,
}


class BD1Error(ValueError):
    """The bd1 class-field build or proof failed closed."""


@dataclass(frozen=True)
class CoderResult:
    codec: str
    bytes: int
    sha256: str
    artifact_path: str | None


@dataclass(frozen=True)
class ClassFieldBuild:
    section: bytes
    raw: bytes
    records: tuple[bytes, ...]
    coder_race: tuple[CoderResult, ...]
    selected_codec: str
    selected_payload: bytes
    band_pixels: int
    lane_side_bits: int
    road_side_bits: int
    per_pair_band_pixels: tuple[int, ...]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, CoderResult):
        return {
            "codec": value.codec,
            "bytes": value.bytes,
            "sha256": value.sha256,
            "artifact_path": value.artifact_path,
        }
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def pack_bits(bits: np.ndarray) -> bytes:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((values != 0) & (values != 1)):
        raise BD1Error("bit stream contains values outside {0,1}")
    return np.packbits(values, bitorder="big").tobytes()


def unpack_bits(payload: bytes, count: int) -> np.ndarray:
    need = (count + 7) // 8
    if len(payload) != need:
        raise BD1Error(f"bit payload expected {need} bytes, got {len(payload)}")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    if np.any(bits[count:] != 0):
        raise BD1Error("bit payload has nonzero padding")
    return np.ascontiguousarray(bits[:count].astype(bool))


def lzma1_raw(payload: bytes) -> bytes:
    return lzma.compress(payload, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)


def unlzma1_raw(payload: bytes, expected_len: int) -> bytes:
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    got = dec.decompress(payload, max_length=expected_len + 1)
    if len(got) != expected_len or not dec.eof or dec.unused_data:
        raise BD1Error("LZMA1 raw stream length or termination mismatch")
    return got


def import_r7() -> Any:
    path = _REPO / "experiments/ddm_r7_token_coder.py"
    spec = importlib.util.spec_from_file_location("ddm_r7_token_coder_bd1", path)
    if spec is None or spec.loader is None:
        raise BD1Error(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bytes_to_nibbles(byte_matrix: np.ndarray) -> np.ndarray:
    hi = byte_matrix >> 4
    lo = byte_matrix & 15
    return np.stack([hi, lo], axis=2).astype(np.uint8)


def nibbles_to_bytes(nibbles: np.ndarray) -> np.ndarray:
    if nibbles.shape[2] != 2:
        raise BD1Error("nibble matrix shape mismatch")
    return ((nibbles[:, :, 0].astype(np.uint16) << 4) | nibbles[:, :, 1].astype(np.uint16)).astype(
        np.uint8
    )


def smevr_records(records: list[bytes]) -> bytes:
    """R7 SMEVR over fixed-width record rows; same framing as cg3/se3."""

    r7 = import_r7()
    n = len(records)
    max_len = max((len(record) for record in records), default=0)
    cols_total = max_len + 4
    matrix = np.zeros((n, cols_total), dtype=np.uint8)
    for i, record in enumerate(records):
        matrix[i, :4] = np.frombuffer(struct.pack("<I", len(record)), dtype=np.uint8)
        matrix[i, 4:4 + len(record)] = np.frombuffer(record, dtype=np.uint8)
    max_values = 16_000_000
    max_cols = max(1, max_values // max(1, n * 2))
    chunks: list[bytes] = []
    for start in range(0, cols_total, max_cols):
        part = matrix[:, start:start + max_cols]
        codes = bytes_to_nibbles(part).reshape(n, part.shape[1], 2, 1)
        frame = r7.encode_token_codes(codes, levels=16, codec="smevr")
        decoded = r7.decode_token_codes(frame, verify=r7.VERIFY_DIGEST)
        if not np.array_equal(decoded, codes):
            raise BD1Error("SMEVR class-field frame failed digest decode")
        chunks.append(frame)
    out = bytearray(b"CGSV1" + struct.pack("<IIH", n, cols_total, len(chunks)))
    for chunk in chunks:
        out += struct.pack("<I", len(chunk))
        out += chunk
    return bytes(out)


def unsmevr_records(payload: bytes) -> list[bytes]:
    if payload[:5] != b"CGSV1":
        raise BD1Error("bad CGSV1 magic")
    if len(payload) < 15:
        raise BD1Error("CGSV1 header truncated")
    n, cols_total, chunk_count = struct.unpack("<IIH", payload[5:15])
    off = 15
    parts = []
    r7 = import_r7()
    for _ in range(chunk_count):
        if len(payload) < off + 4:
            raise BD1Error("CGSV1 chunk length truncated")
        (length,) = struct.unpack_from("<I", payload, off)
        off += 4
        frame = payload[off:off + length]
        off += length
        if len(frame) != length:
            raise BD1Error("CGSV1 chunk truncated")
        decoded = r7.decode_token_codes(frame, verify=r7.VERIFY_DIGEST)
        parts.append(nibbles_to_bytes(decoded.reshape(decoded.shape[0], decoded.shape[1], 2)))
    if off != len(payload):
        raise BD1Error("CGSV1 stream has trailing bytes")
    matrix = np.concatenate(parts, axis=1)[:, :cols_total] if parts else np.zeros((n, 0), dtype=np.uint8)
    records: list[bytes] = []
    for row in matrix:
        (length,) = struct.unpack("<I", row[:4].tobytes())
        if length > max(0, cols_total - 4):
            raise BD1Error("CGSV1 record length exceeds row width")
        records.append(row[4:4 + length].tobytes())
    return records


def decode_body(codec: str, payload: bytes, expected_raw_len: int) -> bytes:
    if codec == "lzma1-raw":
        return unlzma1_raw(payload, expected_raw_len)
    if codec == "brotli-q11":
        raw = brotli.decompress(payload)
        if len(raw) != expected_raw_len:
            raise BD1Error("Brotli raw length mismatch")
        return raw
    if codec == "smevr-r7-nibble":
        raw = b"".join(unsmevr_records(payload))
        if len(raw) != expected_raw_len:
            raise BD1Error("SMEVR raw length mismatch")
        return raw
    raise BD1Error(f"unknown codec {codec!r}")


def road_lane_band(frame_argmax: np.ndarray, radius: int = 1) -> np.ndarray:
    st3 = ndimage.generate_binary_structure(2, 2)
    road = frame_argmax == ROAD
    lane = frame_argmax == LANE
    return ndimage.binary_dilation(road, st3, radius) & ndimage.binary_dilation(lane, st3, radius)


def encode_class_field_records(lstars: np.ndarray, *, radius: int = 1) -> tuple[bytes, tuple[bytes, ...], dict[str, Any]]:
    if tuple(lstars.shape) != (N_PAIRS, SEG_H, SEG_W):
        raise BD1Error(f"unexpected lstars shape {lstars.shape}")
    records: list[bytes] = []
    per_pair: list[int] = []
    band_pixels = 0
    lane_side_bits = 0
    band_bitmap_bytes = (SEG_H * SEG_W + 7) // 8
    for pair in range(N_PAIRS):
        labels = np.asarray(lstars[pair], dtype=np.uint8)
        band = road_lane_band(labels, radius=radius)
        lane_side = np.asarray(labels[band] == LANE, dtype=bool)
        band_payload = pack_bits(band)
        if len(band_payload) != band_bitmap_bytes:
            raise BD1Error("band bitmap width drifted")
        side_payload = pack_bits(lane_side)
        records.append(band_payload + side_payload)
        count = int(band.sum())
        per_pair.append(count)
        band_pixels += count
        lane_side_bits += int(lane_side.sum())
    raw = b"".join(records)
    meta = {
        "radius": radius,
        "n_pairs": N_PAIRS,
        "seg_h": SEG_H,
        "seg_w": SEG_W,
        "slots": N_PAIRS * SEG_H * SEG_W,
        "band_bitmap_bytes_per_pair": band_bitmap_bytes,
        "band_pixels": band_pixels,
        "lane_side_bits": lane_side_bits,
        "road_side_bits": band_pixels - lane_side_bits,
        "per_pair_band_min": min(per_pair),
        "per_pair_band_max": max(per_pair),
        "per_pair_band_mean": float(np.mean(per_pair)),
    }
    return raw, tuple(records), meta


def race_coders(
    *,
    raw: bytes,
    records: tuple[bytes, ...],
    artifact_dir: Path,
    store_best: bool,
) -> tuple[tuple[CoderResult, ...], str, bytes]:
    encoded = {
        "brotli-q11": brotli.compress(raw, quality=11),
        "lzma1-raw": lzma1_raw(raw),
        "smevr-r7-nibble": smevr_records(list(records)),
    }
    if brotli.decompress(encoded["brotli-q11"]) != raw:
        raise BD1Error("Brotli class-field roundtrip failed")
    if unlzma1_raw(encoded["lzma1-raw"], len(raw)) != raw:
        raise BD1Error("LZMA1 class-field roundtrip failed")
    if tuple(unsmevr_records(encoded["smevr-r7-nibble"])) != records:
        raise BD1Error("SMEVR class-field record roundtrip failed")
    best_codec = min(encoded, key=lambda key: len(encoded[key]))
    results = []
    for codec, payload in sorted(encoded.items(), key=lambda item: len(item[1])):
        artifact_path = None
        if store_best and codec == best_codec:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            path = artifact_dir / f"bd1_r1_road_lane_class_field.{codec}.bin"
            path.write_bytes(payload)
            artifact_path = str(path)
        results.append(CoderResult(codec, len(payload), sha256_bytes(payload), artifact_path))
    return tuple(results), best_codec, encoded[best_codec]


def build_section(
    lstars: np.ndarray,
    *,
    radius: int,
    artifact_dir: Path,
    store_best: bool,
) -> ClassFieldBuild:
    raw, records, meta = encode_class_field_records(lstars, radius=radius)
    coder_race, selected_codec, selected_payload = race_coders(
        raw=raw,
        records=records,
        artifact_dir=artifact_dir,
        store_best=store_best,
    )
    header = BD1_HEADER.pack(
        BD1_MAGIC,
        BD1_VERSION,
        SEG_H,
        SEG_W,
        N_PAIRS,
        radius,
        ROAD,
        LANE,
        1,  # paint Road/Lane RGB in scorer-grid support
        CODEC_IDS[selected_codec],
        len(raw),
        meta["band_bitmap_bytes_per_pair"],
        hashlib.sha256(raw).digest(),
    )
    section = header + selected_payload
    parsed = parse_class_field_section(section)
    if parsed["band_pixels"] != meta["band_pixels"]:
        raise BD1Error("class-field parse-back band-pixel count drifted")
    return ClassFieldBuild(
        section=section,
        raw=raw,
        records=records,
        coder_race=coder_race,
        selected_codec=selected_codec,
        selected_payload=selected_payload,
        band_pixels=meta["band_pixels"],
        lane_side_bits=meta["lane_side_bits"],
        road_side_bits=meta["road_side_bits"],
        per_pair_band_pixels=tuple(parsed["per_pair_band_pixels"]),
    )


def parse_class_field_section(section: bytes) -> dict[str, Any]:
    if len(section) < BD1_HEADER.size:
        raise BD1Error("BD1 section header truncated")
    (magic, version, seg_h, seg_w, n_pairs, radius, road_cls, lane_cls, paint_mode,
     codec_id, raw_len, band_bytes, raw_sha) = BD1_HEADER.unpack_from(section, 0)
    if magic != BD1_MAGIC:
        raise BD1Error("BD1 section magic differs")
    if version != BD1_VERSION:
        raise BD1Error("BD1 section version differs")
    if (seg_h, seg_w, n_pairs, radius, road_cls, lane_cls, paint_mode) != (
        SEG_H,
        SEG_W,
        N_PAIRS,
        1,
        ROAD,
        LANE,
        1,
    ):
        raise BD1Error("BD1 section geometry or class contract differs")
    codec = ID_CODECS.get(int(codec_id))
    if codec is None:
        raise BD1Error(f"unknown BD1 codec id {codec_id}")
    raw = decode_body(codec, section[BD1_HEADER.size:], int(raw_len))
    if hashlib.sha256(raw).digest() != raw_sha:
        raise BD1Error("BD1 raw SHA-256 mismatch")
    if band_bytes != (SEG_H * SEG_W + 7) // 8:
        raise BD1Error("BD1 band bitmap width differs")
    off = 0
    per_pair: list[int] = []
    lane_side_bits = 0
    slots = SEG_H * SEG_W
    for _ in range(N_PAIRS):
        band = unpack_bits(raw[off:off + int(band_bytes)], slots)
        off += int(band_bytes)
        count = int(band.sum())
        side_bytes = (count + 7) // 8
        side = unpack_bits(raw[off:off + side_bytes], count)
        off += side_bytes
        per_pair.append(count)
        lane_side_bits += int(side.sum())
    if off != len(raw):
        raise BD1Error("BD1 raw body has trailing bytes")
    return {
        "codec": codec,
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "section_bytes": len(section),
        "section_sha256": sha256_bytes(section),
        "band_pixels": int(sum(per_pair)),
        "lane_side_bits": lane_side_bits,
        "road_side_bits": int(sum(per_pair)) - lane_side_bits,
        "per_pair_band_pixels": per_pair,
    }


def copy_runtime_tree(base_sub: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "archive").mkdir()
    for name in (
        "ddm_r7_token_coder.py",
        "ddm_tr1_runtime.py",
        "inflate.sh",
        "pfs1_warp_receiver.py",
        "repair_entropy_coder_runtime_adapters.py",
    ):
        shutil.copy2(base_sub / name, out_dir / name)
    shutil.copy2(_REPO / "experiments/inflate_runner_v4d.py", out_dir / "inflate_runner.py")
    shutil.copy2(
        _REPO / "src/tac/optimization/ddm_ix2_archive_container.py",
        out_dir / "ddm_ix2_archive_container.py",
    )


def import_generated_runner(out_dir: Path) -> Any:
    sys.path.insert(0, str(out_dir))
    try:
        spec = importlib.util.spec_from_file_location("bd1_generated_inflate_runner", out_dir / "inflate_runner.py")
        if spec is None or spec.loader is None:
            raise BD1Error("could not load generated inflate_runner")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(out_dir))
        except ValueError:
            pass


def receiver_smoke(out_dir: Path, pair_index: int) -> dict[str, Any]:
    module = import_generated_runner(out_dir)
    decoder = module.Decoder(out_dir / "archive")
    with_field = decoder.f1(pair_index)
    field = decoder._bd1_class_field
    decoder._bd1_class_field = None
    without_field = decoder.f1(pair_index)
    changed = np.any(with_field != without_field, axis=2)
    return {
        "pair": pair_index,
        "receiver_class_field_present": field is not None,
        "band_pixels_this_pair": int(field["pair_counts"][pair_index]),
        "camera_pixels_changed": int(changed.sum()),
        "frame1_before_sha256": sha256_bytes(without_field.tobytes()),
        "frame1_after_sha256": sha256_bytes(with_field.tobytes()),
        "mutated": bool(np.any(changed)),
    }


def read_archive_payload(archive_zip: Path) -> bytes:
    with zipfile.ZipFile(archive_zip, "r") as archive:
        names = archive.namelist()
        if names != ["0.bin"]:
            raise BD1Error(f"expected single 0.bin member, found {names}")
        return archive.read("0.bin")


def build_local_ledger(archive_zip: Path, joint_names: tuple[str, ...]) -> dict[str, Any]:
    with zipfile.ZipFile(archive_zip, "r") as archive:
        info = archive.infolist()
        members = [
            {
                "name": row.filename,
                "compress_type": int(row.compress_type),
                "compressed_bytes": int(row.compress_size),
                "uncompressed_bytes": int(row.file_size),
            }
            for row in info
        ]
        payload = archive.read("0.bin")
    bulk, sections = parse_payload(payload)
    return {
        "archive_bytes": archive_zip.stat().st_size,
        "archive_sha256": sha256_file(archive_zip),
        "payload_bytes": len(payload),
        "bulk_bytes": len(bulk),
        "joint_section_count": len(sections),
        "joint_sections": [
            {
                "index": index,
                "name": joint_names[index] if index < len(joint_names) else f"section_{index}",
                "raw_bytes": len(section),
                "sha256": sha256_bytes(section),
                "magic": section[:8].decode("ascii", "ignore") if len(section) >= 8 else None,
            }
            for index, section in enumerate(sections)
        ],
        "zip_members": members,
        "payload_reencodes_identically": build_payload(bulk, list(sections)) == payload,
    }


def run_identity_decode(
    *,
    base_sub: Path,
    identity_dir: Path,
    expected_raw_sha256: str,
    expected_raw_bytes: int,
) -> dict[str, Any]:
    if identity_dir.exists():
        raise BD1Error(f"identity proof dir already exists: {identity_dir}")
    copy_runtime_tree(base_sub, identity_dir)
    payload = read_archive_payload(base_sub / "archive.zip")
    (identity_dir / "archive" / "0.bin").write_bytes(payload)
    names = identity_dir / "names.txt"
    names.write_text("0.mkv\n")
    output = identity_dir / "inflated"
    output.mkdir()
    t0 = time.time()
    subprocess.run(
        [sys.executable, str(identity_dir / "inflate_runner.py"), str(identity_dir / "archive"), str(output), str(names)],
        check=True,
        cwd=str(identity_dir),
    )
    raw_path = output / "0.raw"
    raw_bytes = raw_path.stat().st_size
    raw_sha = sha256_file(raw_path)
    return {
        "command": f"{sys.executable} {identity_dir / 'inflate_runner.py'} {identity_dir / 'archive'} {output} {names}",
        "output_raw": str(raw_path),
        "raw_bytes": raw_bytes,
        "raw_sha256": raw_sha,
        "expected_raw_bytes": expected_raw_bytes,
        "expected_raw_sha256": expected_raw_sha256,
        "byte_identical_to_qo1_shipped_decode": raw_bytes == expected_raw_bytes and raw_sha == expected_raw_sha256,
        "wall_seconds": round(time.time() - t0, 1),
    }


def write_markdown_receipt(path: Path, receipt: dict[str, Any]) -> None:
    pricing = receipt["class_field_pricing"]
    candidate = receipt["candidate"]
    identity = receipt["old_archive_identity_proof"]
    lines = [
        "# BD1 receiver class-field grammar receipt - 2026-08-05",
        "",
        "Status: **RECEIVER-CLOSED / SURVIVAL-UNMEASURED**.",
        "",
        "Axis: `[macOS-CPU advisory / scorer-free receiver-byte custody]`.",
        "`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.",
        "",
        "## Measurements",
        "",
        "| item | value | label |",
        "|---|---:|---|",
        f"| base qo1 archive bytes | `{receipt['base']['archive_bytes']}` | MEASURED stat |",
        f"| base qo1 archive sha256 | `{receipt['base']['archive_sha256']}` | MEASURED sha256 |",
        f"| class-field raw bytes | `{pricing['raw_bytes']}` | MEASURED from cached `lstars` |",
        f"| class-field band pixels | `{pricing['band_pixels']}` | MEASURED n600 denominator |",
        f"| best class-field codec | `{pricing['best_codec']}` | MEASURED coder race |",
        f"| best coded body bytes | `{pricing['best_body_bytes']}` | MEASURED coder race |",
        f"| class-field section bytes | `{pricing['section_bytes']}` | MEASURED section bytes |",
        f"| candidate archive bytes | `{candidate['archive_bytes']}` | MEASURED stat |",
        f"| candidate archive sha256 | `{candidate['archive_sha256']}` | MEASURED sha256 |",
        f"| candidate delta vs qo1 | `{candidate['delta_bytes_vs_qo1']}` | DERIVED |",
        "",
        "Coder race over the real r1 Road/Lane band-field payload:",
        "",
        "| codec | bytes | sha256 |",
        "|---|---:|---|",
    ]
    for row in pricing["coder_race"]:
        lines.append(f"| `{row['codec']}` | `{row['bytes']}` | `{row['sha256']}` |")
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            "| reference | bytes / threshold | bd1 relation |",
            "|---|---:|---|",
            f"| SE3 side-implied stream | `{SE3_SIDE_BYTES}` B | bd1 section is `{pricing['section_bytes'] - SE3_SIDE_BYTES:+d}` B |",
            f"| SE3 explicit-direction stream | `{SE3_EXPLICIT_BYTES}` B | bd1 section is `{pricing['section_bytes'] - SE3_EXPLICIT_BYTES:+d}` B |",
            f"| CQ2 25KB side-implied survival threshold | `{CQ2_THRESHOLDS['25KB_student_side_implied']}` | reference only |",
            f"| CQ2 25KB explicit-direction survival threshold | `{CQ2_THRESHOLDS['25KB_student_explicit_direction']}` | reference only |",
            f"| CQ2 75KB side-implied survival threshold | `{CQ2_THRESHOLDS['75KB_student_side_implied']}` | reference only |",
            f"| CQ2 75KB explicit-direction survival threshold | `{CQ2_THRESHOLDS['75KB_student_explicit_direction']}` | reference only |",
            "",
            "## Receiver Proofs",
            "",
            f"- Old qo1 archive through the extended receiver byte-identical to shipped decode: `{identity['byte_identical_to_qo1_shipped_decode']}`.",
            f"- Old decode raw: `{identity['output_raw']}`, bytes `{identity['raw_bytes']}`, sha256 `{identity['raw_sha256']}`.",
            f"- Candidate parse-back section count: `{receipt['candidate_parse_back']['joint_section_count']}`.",
            f"- Receiver smoke pair `{receipt['receiver_smoke']['pair']}` changed `{receipt['receiver_smoke']['camera_pixels_changed']}` camera pixels with the class field active.",
            "",
            "## Boundaries",
            "",
            "- No SegNet/PoseNet forward was run.",
            "- No `upstream/` file was edited.",
            "- No `/tmp` evidence is cited.",
            "- The class field is video-derived counted payload; no GT/class table was moved into receiver code.",
            "- Survival is unmeasured; this is mechanism and custody only.",
            "",
            "## Follow-On Disposition",
            "",
            "QUEUED-WITH-FIRE-ORDER: after `sq2` lands its scorer result, compose or replace this first-cut dense class-field section with the sq2 field payload if it is the selected value source, then run one scorer-slot-owned n600 evaluation from the byte-closed archive. Do not run that scorer step while `sq2` owns the slot.",
            "",
            "## NEXT-IF-RESUMED",
            "",
            "Start from the JSON receipt and candidate archive in this directory. The immediate optimization target is reducing the counted class-field bytes; the receiver grammar itself is closed for tagged optional sections.",
            "",
            f"Own-vehicle frontier line: `S = {BASELINE_S} @ {BASELINE_BYTES:,} B {BASELINE_AXIS}`; bd1 did not run a scorer and did not move the contest pointer.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def build(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sub", type=Path, default=DEFAULT_BASE_SUB)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--identity-dir", type=Path, default=DEFAULT_IDENTITY_DIR)
    parser.add_argument("--smoke-pair", type=int, default=0)
    parser.add_argument("--skip-identity-decode", action="store_true")
    parser.add_argument("--store-best", action="store_true")
    parser.add_argument("--hash-inputs", action="store_true")
    args = parser.parse_args(argv)

    if args.candidate_dir.exists():
        raise BD1Error(f"candidate dir already exists: {args.candidate_dir}")
    args.research_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.ssd_dir / "class_field_payloads"
    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    field = build_section(lstars, radius=1, artifact_dir=artifact_dir, store_best=args.store_best)

    base_archive = args.base_sub / "archive.zip"
    base_sha = sha256_file(base_archive)
    if base_sha != BASELINE_ARCHIVE_SHA256:
        raise BD1Error(f"base archive SHA drift: {base_sha}")
    base_payload = read_archive_payload(base_archive)
    bulk, sections = parse_payload(base_payload)
    if len(sections) != 5:
        raise BD1Error(f"qo1 base expected 5 joint sections, got {len(sections)}")
    new_payload = build_payload(bulk, [*sections, field.section])
    archive_zip = build_single_member_zip(new_payload, name="0.bin")

    copy_runtime_tree(args.base_sub, args.candidate_dir)
    (args.candidate_dir / "archive" / "0.bin").write_bytes(new_payload)
    (args.candidate_dir / "archive.zip").write_bytes(archive_zip)

    parse_back = build_local_ledger(
        args.candidate_dir / "archive.zip",
        ("config", "renderer", "selector", "pose_warp", "frame0_pose_repair", "road_lane_class_field"),
    )
    receiver = receiver_smoke(args.candidate_dir, args.smoke_pair)
    if not receiver["receiver_class_field_present"] or not receiver["mutated"]:
        raise BD1Error(f"receiver smoke did not consume/mutate from class field: {receiver}")

    if args.skip_identity_decode:
        identity = {
            "skipped": True,
            "reason": "--skip-identity-decode",
            "byte_identical_to_qo1_shipped_decode": False,
        }
    else:
        identity = run_identity_decode(
            base_sub=args.base_sub,
            identity_dir=args.identity_dir,
            expected_raw_sha256=BASE_RAW_SHA256,
            expected_raw_bytes=BASE_RAW_BYTES,
        )
        if not identity["byte_identical_to_qo1_shipped_decode"]:
            raise BD1Error(f"old archive identity proof failed: {identity}")

    best = min(field.coder_race, key=lambda row: row.bytes)
    candidate_archive = args.candidate_dir / "archive.zip"
    candidate_bytes = candidate_archive.stat().st_size
    rate_delta_s = 25.0 * (candidate_bytes - BASELINE_BYTES) / RATE_DENOM
    receipt = {
        "schema": "ddm_bd1_class_field_receiver.v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "axis": "[macOS-CPU advisory / scorer-free receiver-byte custody]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "base": {
            "submission_dir": str(args.base_sub),
            "archive_bytes": base_archive.stat().st_size,
            "archive_sha256": base_sha,
            "payload_sha256": sha256_bytes(base_payload),
            "joint_section_count": len(sections),
            "own_vehicle_S": BASELINE_S,
            "axis": BASELINE_AXIS,
        },
        "inputs": {
            "gt_cache": str(args.gt_cache),
            "gt_cache_sha256": sha256_file(args.gt_cache) if args.hash_inputs else None,
            "lstars_member": "lstars.npy",
            "selection_mode": "n600 all pairs; no prefix",
            "shape": [N_PAIRS, SEG_H, SEG_W],
            "class_order": {"Road": ROAD, "Lane": LANE},
        },
        "class_field_pricing": {
            "section_magic": BD1_MAGIC.decode("ascii"),
            "version": BD1_VERSION,
            "field_schema": "r1 Road/Lane band bitmap per pair + Road/Lane side bits over raster-order band coordinates",
            "raw_bytes": len(field.raw),
            "raw_sha256": sha256_bytes(field.raw),
            "records": len(field.records),
            "band_pixels": field.band_pixels,
            "lane_side_bits": field.lane_side_bits,
            "road_side_bits": field.road_side_bits,
            "best_codec": best.codec,
            "best_body_bytes": best.bytes,
            "best_body_sha256": best.sha256,
            "section_bytes": len(field.section),
            "section_sha256": sha256_bytes(field.section),
            "coder_race": list(field.coder_race),
        },
        "comparison": {
            "se3_side_implied_stream_bytes": SE3_SIDE_BYTES,
            "se3_explicit_direction_stream_bytes": SE3_EXPLICIT_BYTES,
            "se3_captured_flips": SE3_CAPTURED_FLIPS,
            "cq2_thresholds": CQ2_THRESHOLDS,
        },
        "candidate": {
            "submission_dir": str(args.candidate_dir),
            "archive_bytes": candidate_bytes,
            "archive_sha256": sha256_file(candidate_archive),
            "payload_sha256": sha256_bytes(new_payload),
            "delta_bytes_vs_qo1": candidate_bytes - BASELINE_BYTES,
            "rate_delta_S_vs_qo1": rate_delta_s,
            "label": "RECEIVER-CLOSED, SURVIVAL-UNMEASURED, score_claim=false",
        },
        "candidate_parse_back": parse_back,
        "receiver_smoke": receiver,
        "old_archive_identity_proof": identity,
        "boundaries": [
            "No SegNet/PoseNet forward was run.",
            "No upstream/ files were edited.",
            "No /tmp evidence is cited.",
            "The field is video-derived counted payload.",
            "Survival through scorer cells is not measured or claimed.",
        ],
        "own_vehicle_frontier_line": f"S = {BASELINE_S} @ {BASELINE_BYTES} B {BASELINE_AXIS}",
    }
    json_path = args.research_dir / "bd1_class_field_candidate_receipt.json"
    md_path = args.research_dir / "BD1_RECEIPT_20260805.md"
    json_path.write_text(json.dumps(jsonable(receipt), indent=2, sort_keys=True) + "\n")
    write_markdown_receipt(md_path, jsonable(receipt))
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "candidate_archive": str(candidate_archive)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
