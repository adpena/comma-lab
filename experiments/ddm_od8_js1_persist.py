#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""OD8 native-DOF persistence and scorer-free pricing for JS1.

This is a sibling of ``ddm_js1_staging_discriminator.py``.  It does not edit or
wrap the running OD3 harness.  The default ``analyze-recorded`` command is
scorer-free: it reads recorded OD2 outcomes, recomputes only cached-mask
deterministic objects, prices native-DOF packet shapes, and fixes the OD6
self-containment gap by adding the missing shipped coverage sections.

The future ``solve-persist`` command is intentionally gated by ``--allow-scorer``.
It is the post-OD3 path that re-runs the JS1 solve and persists the actual
Stage-1 paint support/values plus Stage-2 quantized DCT coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "src", REPO / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ddm_od5_generator_packet as od5
import ddm_od6_decoder_legal_context as od6
import ddm_pe1_per_edge_partition_race as pe1
from ddm_et1_ph1_block16_on_our_vehicle import solve_blocks, translate_blocks
from ddm_sq1_eta_seg_realization import (
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    Scorer,
    decode_gt_frames,
    seq_len,
)
from ddm_sq1_stage_decomposition_and_solved_paint import (
    realize_scorer_paint_to_camera,
    resize_to_scorer,
    solve_margin_optimal_paint,
)

from tac.optimization import ddm_od4_weak_stage1_packet as od4

DEFAULT_RESEARCH_DIR: Final = REPO / ".omx/research/ddm_od8_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_od8_20260805")
DEFAULT_OD2_JSON: Final = REPO / ".omx/research/ddm_od2_20260805/od2_js1_n32_cprime_k4.json"
DEFAULT_PAIR_SELECTION: Final = REPO / ".omx/research/ddm_od2_20260805/PAIR_SELECTION.json"
DEFAULT_PAIRS_NPY: Final = REPO / ".omx/research/ddm_od2_20260805/od2_pairs_n32_seed20260805.npy"
DEFAULT_ARGMAX_CACHE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
DEFAULT_SUB_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2")
DEFAULT_GT_MKV: Final = REPO / "upstream/videos/0.mkv"
DEFAULT_OD6_RECEIPT: Final = REPO / ".omx/research/ddm_od6_20260805/ddm_od6_decoder_legal_receipt.json"
DEFAULT_OD6_PACKET: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_od6_20260805/"
    "run_20260805T030410Z/packets/base_rgb_generator_geometry_b1024.od5.raw_packet"
)
DEFAULT_PE1_RECEIPT: Final = REPO / ".omx/research/ddm_pe1_20260805/ddm_pe1_repr_race_receipt.json"
DEFAULT_PE3_RECEIPT: Final = REPO / ".omx/research/ddm_pe3_20260805/ddm_pe3_hybrid_receipt.json"
DEFAULT_GT_CACHE: Final = pe1.DEFAULT_GT_CACHE
DEFAULT_G4_RECURRENCE: Final = od5.DEFAULT_G4_RECURRENCE

RATE_PER_BYTE: Final = 25.0 / 37_545_489.0
OWN_FRONTIER_LINE: Final = "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]"
FINAL_FRONTIER_SENTENCE: Final = (
    "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; "
    "contest pointer borrowed/unmoved."
)
OD8_STAGE1_MAGIC: Final = b"OD8S1P1\0"
OD8_STAGE2_MAGIC: Final = b"OD8S2C1\0"


class OD8Error(ValueError):
    """OD8 persistence or pricing failed a typed invariant."""


@dataclass(frozen=True, slots=True)
class Stage1NativeRecord:
    pair: int
    flat_indices: np.ndarray
    rgb_values: np.ndarray
    band_sha256: str
    rgb_sha256: str

    @property
    def count(self) -> int:
        return int(self.flat_indices.size)


@dataclass(frozen=True, slots=True)
class Stage2DCTRecord:
    pair: int
    k: int
    qcoeffs: np.ndarray
    coeff_sha256: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, od4.CoderRow):
        return value.as_json()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OD8Error(f"JSON root is not an object: {path}")
    return data


def _storage_preflight(path: Path, required_free_bytes: int) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    out = {
        "path": str(path),
        "required_free_bytes": int(required_free_bytes),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "ok": bool(usage.free >= required_free_bytes),
    }
    if not out["ok"]:
        raise OD8Error(f"SSD storage preflight failed: {out}")
    return out


def _varint(value: int) -> bytes:
    if value < 0:
        raise OD8Error("varint cannot encode a negative value")
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
            raise OD8Error("truncated varint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise OD8Error("varint too long")


def _best_coder(rows: tuple[od4.CoderRow, ...]) -> od4.CoderRow:
    candidates = [row for row in rows if row.parseback_exact and row.bytes > 0]
    if not candidates:
        raise OD8Error("no coder row survived parse-back")
    return min(candidates, key=lambda row: row.bytes)


def _store_packet(path: Path, payload: bytes, best: od4.CoderRow) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "selected_coder": best.codec,
        "selected_coder_bytes": int(best.bytes),
        "selected_coder_sha256": best.sha256,
    }


def _load_od2_pair_surface(
    *,
    od2_json: Path,
    pair_selection: Path,
    argmax_cache: Path,
    block: int,
    rmax: int,
) -> tuple[list[int], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    od2 = _load_json(od2_json)
    selection = _load_json(pair_selection)
    rows = od2.get("rows")
    if not isinstance(rows, list) or not rows:
        raise OD8Error("OD2 JSON has no rows")
    od2_rows = {int(row["pair"]): row for row in rows}
    pairs = [int(pair) for pair in selection["pairs"]]
    missing = [pair for pair in pairs if pair not in od2_rows]
    if missing:
        raise OD8Error(f"OD2 JSON missing selected pairs: {missing}")

    current = np.load(argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gt = np.load(argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")
    proofs: list[dict[str, Any]] = []
    for pair in pairs:
        row = od2_rows[pair]
        cur = np.asarray(current[pair], dtype=np.uint8)
        tgt_gt = np.asarray(gt[pair], dtype=np.uint8)
        before = int((cur != tgt_gt).sum())
        if before != int(row["flips_before"]):
            raise OD8Error(f"pair {pair}: flips_before {before} != OD2 {row['flips_before']}")
        offsets = solve_blocks(cur, tgt_gt, block, rmax)
        target = translate_blocks(cur, offsets.reshape(-1, 2), block)
        band = target != cur
        n_described = before - int((target != tgt_gt).sum())
        if n_described != int(row["n_described"]):
            raise OD8Error(f"pair {pair}: n_described {n_described} != OD2 {row['n_described']}")
        if int(band.sum()) != int(row["band_px"]):
            raise OD8Error(f"pair {pair}: band_px {int(band.sum())} != OD2 {row['band_px']}")
        stage1_after = int(row["stage1"]["flips_after"])
        proofs.append(
            {
                "pair": pair,
                "flips_before": before,
                "band_px": int(band.sum()),
                "n_described": int(n_described),
                "stage1_flips_after_recorded": stage1_after,
                "stage1_retained_fixes_recorded": before - stage1_after,
                "offsets_sha256": _sha256_bytes(np.ascontiguousarray(offsets.astype(np.int8)).tobytes()),
                "band_sha256": _sha256_bytes(np.packbits(band.reshape(-1), bitorder="little").tobytes()),
            }
        )
    return pairs, od2_rows, proofs


def _scorer_lattice_u8(cam_frame: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        tensor = resize_to_scorer(np.array(cam_frame, copy=True))
        arr = torch.round(torch.clamp(tensor, 0.0, 255.0))[0].permute(1, 2, 0).numpy()
    return np.ascontiguousarray(arr.astype(np.uint8))


def _stage1_records_from_base_rgb(
    *,
    pairs: list[int],
    argmax_cache: Path,
    sub_dir: Path,
    block: int,
    rmax: int,
    paint_mode: str,
) -> list[Stage1NativeRecord]:
    current = np.load(argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gt = np.load(argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")
    raw = np.memmap(
        sub_dir / "inflated/0.raw",
        dtype=np.uint8,
        mode="r",
        shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3),
    )
    records: list[Stage1NativeRecord] = []
    for pair in pairs:
        cur = np.asarray(current[pair], dtype=np.uint8)
        tgt_gt = np.asarray(gt[pair], dtype=np.uint8)
        offsets = solve_blocks(cur, tgt_gt, block, rmax)
        target = translate_blocks(cur, offsets.reshape(-1, 2), block)
        band = target != cur
        flat = np.flatnonzero(band.reshape(-1)).astype(np.int64)
        if paint_mode == "base_rgb_proxy":
            scorer_rgb = _scorer_lattice_u8(np.asarray(raw[seq_len * pair + 1]))
            values = np.ascontiguousarray(scorer_rgb.reshape(-1, 3)[flat])
        elif paint_mode == "deterministic_noise_proxy":
            rng = np.random.default_rng(0x0D800000 + int(pair))
            values = rng.integers(0, 256, size=(flat.size, 3), dtype=np.uint8)
        else:
            raise OD8Error(f"unknown paint_mode {paint_mode!r}")
        records.append(
            Stage1NativeRecord(
                pair=pair,
                flat_indices=np.ascontiguousarray(flat),
                rgb_values=np.ascontiguousarray(values),
                band_sha256=_sha256_bytes(np.packbits(band.reshape(-1), bitorder="little").tobytes()),
                rgb_sha256=_sha256_bytes(values.tobytes()),
            )
        )
    return records


def _encode_stage1_records(records: list[Stage1NativeRecord], *, paint_mode: str) -> bytes:
    body = bytearray()
    body += _varint(len(records))
    body += _varint(SEG_H)
    body += _varint(SEG_W)
    mode = paint_mode.encode("ascii")
    body += _varint(len(mode)) + mode
    for record in records:
        flat = np.asarray(record.flat_indices, dtype=np.int64)
        rgb = np.asarray(record.rgb_values, dtype=np.uint8)
        if rgb.shape != (flat.size, 3):
            raise OD8Error(f"pair {record.pair}: rgb shape {rgb.shape} does not match support")
        if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= SEG_H * SEG_W):
            raise OD8Error(f"pair {record.pair}: flat index out of range")
        if not np.all(flat[1:] > flat[:-1]):
            raise OD8Error(f"pair {record.pair}: flat indices must be strictly increasing")
        body += _varint(record.pair)
        body += _varint(int(flat.size))
        prev = -1
        for item in flat.tolist():
            body += _varint(int(item) - prev - 1)
            prev = int(item)
        body += rgb.tobytes()
    return OD8_STAGE1_MAGIC + bytes(body)


def _decode_stage1_records(payload: bytes) -> list[Stage1NativeRecord]:
    if not payload.startswith(OD8_STAGE1_MAGIC):
        raise OD8Error("stage1 payload magic mismatch")
    offset = len(OD8_STAGE1_MAGIC)
    n_records, offset = _read_varint(payload, offset)
    h, offset = _read_varint(payload, offset)
    w, offset = _read_varint(payload, offset)
    if (h, w) != (SEG_H, SEG_W):
        raise OD8Error(f"stage1 geometry mismatch: {(h, w)}")
    mode_len, offset = _read_varint(payload, offset)
    offset += mode_len
    records: list[Stage1NativeRecord] = []
    for _ in range(n_records):
        pair, offset = _read_varint(payload, offset)
        count, offset = _read_varint(payload, offset)
        flats: list[int] = []
        prev = -1
        for _idx in range(count):
            delta, offset = _read_varint(payload, offset)
            item = prev + 1 + delta
            flats.append(item)
            prev = item
        value_bytes = payload[offset : offset + count * 3]
        if len(value_bytes) != count * 3:
            raise OD8Error("stage1 rgb payload truncated")
        offset += count * 3
        flat_arr = np.asarray(flats, dtype=np.int64)
        rgb = np.frombuffer(value_bytes, dtype=np.uint8).reshape(count, 3).copy()
        band = np.zeros(SEG_H * SEG_W, dtype=bool)
        band[flat_arr] = True
        records.append(
            Stage1NativeRecord(
                pair=int(pair),
                flat_indices=flat_arr,
                rgb_values=rgb,
                band_sha256=_sha256_bytes(np.packbits(band, bitorder="little").tobytes()),
                rgb_sha256=_sha256_bytes(rgb.tobytes()),
            )
        )
    if offset != len(payload):
        raise OD8Error("stage1 payload has trailing bytes")
    return records


def _synthetic_stage2_records(
    *,
    pairs: list[int],
    od2_rows: dict[int, dict[str, Any]],
    k: int,
) -> list[Stage2DCTRecord]:
    records: list[Stage2DCTRecord] = []
    for pair in pairs:
        row = od2_rows[pair][f"arm_cprime_cheap_dct{k}"]
        max_abs = int(float(row["max_abs_int16_coefficient"]))
        rng = np.random.default_rng(0x0D820000 + int(pair) * 131 + k)
        if max_abs <= 0 or bool(row.get("solved_to_all_zero")):
            q = np.zeros((3, k * k), dtype=np.int16)
        else:
            scale = max(1.0, max_abs / 6.0)
            vals = np.rint(rng.laplace(loc=0.0, scale=scale, size=(3, k * k)))
            q = np.clip(vals, -max_abs, max_abs).astype(np.int16)
            q.reshape(-1)[0] = np.int16(max_abs)
        records.append(
            Stage2DCTRecord(
                pair=pair,
                k=k,
                qcoeffs=np.ascontiguousarray(q),
                coeff_sha256=_sha256_bytes(np.ascontiguousarray(q.astype("<i2")).tobytes()),
            )
        )
    return records


def _encode_stage2_records(records: list[Stage2DCTRecord], *, source: str) -> bytes:
    if not records:
        raise OD8Error("empty stage2 record set")
    k_values = {record.k for record in records}
    if len(k_values) != 1:
        raise OD8Error(f"mixed DCT k values: {sorted(k_values)}")
    k = next(iter(k_values))
    body = bytearray()
    body += _varint(len(records))
    body += _varint(k)
    source_b = source.encode("ascii")
    body += _varint(len(source_b)) + source_b
    for record in records:
        q = np.ascontiguousarray(record.qcoeffs.astype("<i2", copy=False))
        if q.shape != (3, k * k):
            raise OD8Error(f"pair {record.pair}: qcoeff shape {q.shape} != {(3, k * k)}")
        body += _varint(record.pair)
        body += _varint(q.size)
        body += q.tobytes()
    return OD8_STAGE2_MAGIC + bytes(body)


def _decode_stage2_records(payload: bytes) -> list[Stage2DCTRecord]:
    if not payload.startswith(OD8_STAGE2_MAGIC):
        raise OD8Error("stage2 payload magic mismatch")
    offset = len(OD8_STAGE2_MAGIC)
    n_records, offset = _read_varint(payload, offset)
    k, offset = _read_varint(payload, offset)
    source_len, offset = _read_varint(payload, offset)
    offset += source_len
    records: list[Stage2DCTRecord] = []
    for _ in range(n_records):
        pair, offset = _read_varint(payload, offset)
        count, offset = _read_varint(payload, offset)
        if count != 3 * k * k:
            raise OD8Error(f"stage2 qcoeff count {count} != {3 * k * k}")
        qbytes = payload[offset : offset + count * 2]
        if len(qbytes) != count * 2:
            raise OD8Error("stage2 coeff payload truncated")
        offset += count * 2
        q = np.frombuffer(qbytes, dtype="<i2").reshape(3, k * k).astype(np.int16, copy=True)
        records.append(
            Stage2DCTRecord(pair=int(pair), k=int(k), qcoeffs=q, coeff_sha256=_sha256_bytes(qbytes))
        )
    if offset != len(payload):
        raise OD8Error("stage2 payload has trailing bytes")
    return records


def _native_packet_price(
    *,
    stage1_records: list[Stage1NativeRecord],
    stage2_records: list[Stage2DCTRecord],
    paint_mode: str,
    ssd_dir: Path,
) -> dict[str, Any]:
    stage1 = _encode_stage1_records(stage1_records, paint_mode=paint_mode)
    decoded_stage1 = _decode_stage1_records(stage1)
    if [record.count for record in decoded_stage1] != [record.count for record in stage1_records]:
        raise OD8Error("stage1 parse-back changed support counts")
    stage2 = _encode_stage2_records(stage2_records, source="synthetic_from_recorded_od2_cmax")
    decoded_stage2 = _decode_stage2_records(stage2)
    if [record.coeff_sha256 for record in decoded_stage2] != [record.coeff_sha256 for record in stage2_records]:
        raise OD8Error("stage2 parse-back changed coefficients")

    manifest = {
        "schema": "ddm_od8_native_dof_packet_manifest.v1",
        "paint_mode": paint_mode,
        "stage1_schema": "sparse scorer-lattice support + RGB uint8 values",
        "stage2_schema": "k*k*3 int16 cheapdct coefficients per pair",
        "stage1_pair_count": len(stage1_records),
        "stage1_support_points": int(sum(record.count for record in stage1_records)),
        "stage2_pair_count": len(stage2_records),
        "stage2_k": int(stage2_records[0].k),
        "stage1_payload_sha256": _sha256_bytes(stage1),
        "stage2_payload_sha256": _sha256_bytes(stage2),
    }
    packet = od4.serialize_od5_packet(
        [
            od4.OD5Section(f"od8_stage1_{paint_mode}", stage1),
            od4.OD5Section("od8_stage2_cheapdct4_synthetic", stage2),
            od4.OD5Section(
                "od8_native_dof_manifest_json",
                json.dumps(_jsonable(manifest), sort_keys=True, separators=(",", ":")).encode("utf-8"),
            ),
        ]
    )
    parsed = od4.parse_od5_packet(packet)
    if od4.serialize_od5_packet(parsed.sections) != packet:
        raise OD8Error("native OD5 packet did not reserialize exactly")
    coder_rows = od4.race_packet_coders(packet)
    best = _best_coder(coder_rows)
    artifact = _store_packet(ssd_dir / "packets" / f"od8_native_dof_{paint_mode}.od5.raw_packet", packet, best)
    n_pairs = len(stage1_records)
    projected = od4.projected_n600_packet_bytes(best.bytes, n_pairs)
    raw_stage1_support_bytes = len(stage1)
    raw_stage2_bytes = len(stage2)
    return {
        "paint_mode": paint_mode,
        "schema": od4.OD5_PACKET_SCHEMA,
        "n_pairs": n_pairs,
        "raw_packet_bytes": len(packet),
        "raw_packet_sha256": od4.sha256_bytes(packet),
        "stage1_payload_bytes": raw_stage1_support_bytes,
        "stage2_payload_bytes": raw_stage2_bytes,
        "stage1_support_points": int(sum(record.count for record in stage1_records)),
        "stage2_raw_int16_bytes": int(sum(record.qcoeffs.size * 2 for record in stage2_records)),
        "coder_race": [row.as_json() for row in coder_rows],
        "best_coder": best.as_json(),
        "artifact": artifact,
        "projected_n600_bytes_linear": projected,
        "projected_n600_rate_s": projected * RATE_PER_BYTE,
        "vs_gc18_floor_45k": int(projected - 45_000),
        "vs_gc18_floor_90k": int(projected - 90_000),
        "vs_od6_packet_76304": int(projected - 76_304),
        "manifest": manifest,
    }


def _build_od6_self_contained_packet(
    *,
    pairs: list[int],
    od6_packet: Path,
    od6_receipt: Path,
    pe1_receipt_path: Path,
    pe3_receipt_path: Path,
    gt_cache: Path,
    argmax_cache: Path,
    g4_recurrence: Path,
    ssd_dir: Path,
) -> dict[str, Any]:
    existing_raw = od6_packet.read_bytes()
    existing_parsed = od4.parse_od5_packet(existing_raw)
    sections = {section.name: section for section in existing_parsed.sections}
    hybrid75 = sections.get("pe3_hybrid75_coords_n32")
    table_section = next(
        (
            section
            for section in existing_parsed.sections
            if section.name.startswith("od6_decoder_legal_table_base_rgb_generator_geometry_1024")
        ),
        None,
    )
    if hybrid75 is None or table_section is None:
        raise OD8Error("OD6 incumbent packet is missing expected hybrid75/table sections")

    header, qlogits = od6._parse_qlogit_payload(table_section.payload, magic=od6.OD6_TABLE_MAGIC)
    required_columns = {
        "pe1_generator_coverage": "pe1_generator_coords_n32",
        "pe3_hybrid75_coverage": "pe3_hybrid75_coords_n32",
        "pe3_hybrid_knee_coverage": "pe3_hybrid_knee_coords_n32",
    }
    columns = {str(item) for item in header.get("feature_columns", [])}
    missing_columns = sorted(set(required_columns) - columns)
    if missing_columns:
        raise OD8Error(f"OD6 table header missing expected coverage columns: {missing_columns}")

    pe1_receipt = _load_json(pe1_receipt_path)
    pe3_receipt = _load_json(pe3_receipt_path)
    current_argmax = np.load(argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    lstars = pe1.open_stored_npy_memmap(gt_cache, "lstars")
    components, extraction = pe1.extract_components(lstars, current_argmax)
    surfaces = od5._build_generator_surfaces(
        components=components,
        lstars=lstars,
        current=current_argmax,
        pe1_receipt=pe1_receipt,
        g4_recurrence=g4_recurrence,
        depth_y1=190.0,
        depth_y2=230.0,
    )
    generator_rep: pe1.RepresentationBuild = surfaces["generator_rep"]
    hybrid_knee = surfaces["hybrid_knee"]
    generator_subset_raw = od5._subset_frame_records(generator_rep.frame_records, pairs)
    hybrid_knee_subset_raw = od5._subset_frame_records(hybrid_knee.frame_records, pairs)

    fixed_header = dict(header)
    fixed_header["schema"] = "ddm_od8_self_contained_od6_bucket_table.v1"
    fixed_header["self_contained_od8"] = True
    fixed_header["coverage_source_sections"] = required_columns
    fixed_header["source_table_schema"] = header.get("schema")
    fixed_header["format_gap_fixed"] = (
        "OD6 b1024 hash coverage features now have named shipped coordinate sections in the same packet"
    )
    fixed_table_payload = od6._serialize_qlogit_payload(
        magic=od6.OD6_TABLE_MAGIC,
        header=fixed_header,
        qlogits=qlogits,
    )
    parsed_header, parsed_qlogits = od6._parse_qlogit_payload(fixed_table_payload, magic=od6.OD6_TABLE_MAGIC)
    if not np.array_equal(parsed_qlogits, qlogits):
        raise OD8Error("OD6 self-contained table qlogits changed under parse-back")
    if parsed_header.get("coverage_source_sections") != required_columns:
        raise OD8Error("OD6 self-contained coverage-source map changed under parse-back")

    packet = od4.serialize_od5_packet(
        [
            od4.OD5Section("pe1_generator_coords_n32", generator_subset_raw),
            hybrid75,
            od4.OD5Section("pe3_hybrid_knee_coords_n32", hybrid_knee_subset_raw),
            od4.OD5Section("od8_self_contained_od6_table_b1024", fixed_table_payload),
        ]
    )
    parsed = od4.parse_od5_packet(packet)
    if od4.serialize_od5_packet(parsed.sections) != packet:
        raise OD8Error("OD6 self-contained OD5 packet did not reserialize exactly")
    present = {section.name for section in parsed.sections}
    missing_sections = sorted(set(required_columns.values()) - present)
    if missing_sections:
        raise OD8Error(f"self-contained OD6 packet missing required sections: {missing_sections}")
    coder_rows = od4.race_packet_coders(packet)
    best = _best_coder(coder_rows)
    artifact = _store_packet(ssd_dir / "packets/od8_od6_b1024_self_contained.od5.raw_packet", packet, best)

    od6_receipt_payload = _load_json(od6_receipt)
    incumbent = od6_receipt_payload["best_rung_by_projected_s_with_od2_pose_credit"]
    incumbent_best = int(incumbent["packet"]["best_coder"]["bytes"])
    incumbent_projected = int(incumbent["projection_with_od2_stage2_pose_credit"]["packet_bytes_n600_projected"])
    pe1_projected = int(pe1_receipt["surgical_winner"]["section_bytes"])
    knee_projected = int(pe3_receipt["hybrid_knee"]["section_bytes"])
    component_sum_projected = incumbent_projected + pe1_projected + knee_projected
    linear_projected = od4.projected_n600_packet_bytes(best.bytes, len(pairs))
    return {
        "schema": "ddm_od8_od6_self_contained_fix.v1",
        "source_packet": _source_file_entry(od6_packet),
        "source_receipt": _source_file_entry(od6_receipt),
        "source_table_payload_sha256": _sha256_bytes(table_section.payload),
        "qlogit_count": int(qlogits.size),
        "qlogits_unchanged": True,
        "coverage_source_sections": required_columns,
        "component_extraction": extraction,
        "added_sections": [
            {
                "name": "pe1_generator_coords_n32",
                "payload_bytes": len(generator_subset_raw),
                "payload_sha256": _sha256_bytes(generator_subset_raw),
            },
            {
                "name": "pe3_hybrid_knee_coords_n32",
                "payload_bytes": len(hybrid_knee_subset_raw),
                "payload_sha256": _sha256_bytes(hybrid_knee_subset_raw),
            },
        ],
        "raw_packet_bytes": len(packet),
        "raw_packet_sha256": od4.sha256_bytes(packet),
        "sections": [
            {
                "name": section.name,
                "payload_bytes": len(section.payload),
                "payload_sha256": od4.sha256_bytes(section.payload),
            }
            for section in parsed.sections
        ],
        "parseback_exact": True,
        "coder_race": [row.as_json() for row in coder_rows],
        "best_coder": best.as_json(),
        "artifact": artifact,
        "delta_exact_n32_best_bytes_vs_od6_incumbent": int(best.bytes - incumbent_best),
        "projected_n600_bytes_linear_from_exact_n32": linear_projected,
        "projected_n600_bytes_component_sum_conservative": component_sum_projected,
        "delta_projected_component_sum_vs_od6_76304": int(component_sum_projected - incumbent_projected),
        "projection_note": (
            "component-sum projection keeps OD6's 76,304 B incumbent projection and adds measured "
            "PE1 generator plus PE3 hybrid-knee n600 section bytes; it is a conservative "
            "self-contained fix, not an optimized replacement for overlapping sections"
        ),
    }


def _native_dof_summary(
    *,
    pairs: list[int],
    od2_rows: dict[int, dict[str, Any]],
    proofs: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    band_total = int(sum(row["band_px"] for row in proofs))
    retained_total = int(sum(row["stage1_retained_fixes_recorded"] for row in proofs))
    stage2_raw_per_pair = k * k * 3 * 2
    n_pairs = len(pairs)
    stage1_value_bytes_n32 = band_total * 3
    stage1_sparse_fixed_u24_support_bytes_n32 = band_total * 3
    stage1_sparse_fixed_u24_support_plus_values_n32 = stage1_value_bytes_n32 + stage1_sparse_fixed_u24_support_bytes_n32
    for pair in pairs:
        rec = od2_rows[pair][f"arm_cprime_cheap_dct{k}"]
        expected = k * k * 3 * 2
        if int(rec["counted_bytes_per_pair"]) != expected:
            raise OD8Error(
                f"pair {pair}: cheapdct{k} bytes {rec['counted_bytes_per_pair']} != {expected}"
            )
        if not bool(rec.get("value_measured_through_int16_quantiser")):
            raise OD8Error(f"pair {pair}: cheapdct{k} was not recorded through int16 quantizer")
    return {
        "schema": "ddm_od8_native_dof_identification.v1",
        "stage1_cprime": {
            "optimizer_variable": (
                "float scorer-lattice delta in solve_margin_optimal_paint; selected on rounded uint8 "
                "SegNet proxy flips"
            ),
            "persisted_native_object": (
                "sparse scorer-lattice frame_1 support (band flat indices) plus selected RGB uint8 "
                "paint values at that support"
            ),
            "receiver_application": (
                "realize_scorer_paint_to_camera writes each scorer pixel's RGB value to D's private "
                "camera support; no stamp re-solve at decode"
            ),
            "support_source": "block16 cprime target band, recomputed here from cached argmax arrays",
            "quantization": "RGB uint8 values; support as scorer-lattice flat indices",
            "pairs": n_pairs,
            "band_px_total_n32": band_total,
            "band_px_mean_per_pair": band_total / n_pairs,
            "optimized_value_params_n32": band_total * 3,
            "optimized_value_raw_bytes_n32": stage1_value_bytes_n32,
            "receiver_sparse_support_plus_values_raw_bytes_n32_fixed_u24": stage1_sparse_fixed_u24_support_plus_values_n32,
            "receiver_sparse_support_plus_values_raw_bytes_n600_fixed_u24_projection": math.ceil(
                stage1_sparse_fixed_u24_support_plus_values_n32 * (N_PAIRS_TOTAL / n_pairs)
            ),
            "full_dense_paint_u8_raw_bytes_per_pair_if_naive": SEG_H * SEG_W * 3,
        },
        "stage2_cheapdct": {
            "k": k,
            "params_per_pair": k * k * 3,
            "quantization": "int16 coefficients, selected through its own quantized synthesis",
            "raw_bytes_per_pair": stage2_raw_per_pair,
            "raw_bytes_n600": stage2_raw_per_pair * N_PAIRS_TOTAL,
            "recorded_n32_all_pairs_match_formula": True,
        },
        "recorded_outcome_denominators": {
            "pairs": n_pairs,
            "retained_stage1_fixes_recorded": retained_total,
            "band_px_total": band_total,
        },
    }


def _table_md(receipt: dict[str, Any]) -> str:
    rows = receipt["native_dof_price_estimates"]
    lines = [
        "| stream | exact n32 best B | projected n600 B | vs 45K | vs 90K | vs OD6 76,304 B | best coder |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['paint_mode']} | {row['best_coder']['bytes']} | "
            f"{row['projected_n600_bytes_linear']} | {row['vs_gc18_floor_45k']:+d} | "
            f"{row['vs_gc18_floor_90k']:+d} | {row['vs_od6_packet_76304']:+d} | "
            f"{row['best_coder']['codec']} |"
        )
    return "\n".join(lines)


def _write_receipt_md(path: Path, receipt: dict[str, Any]) -> None:
    native = receipt["native_dof_identification"]
    stage1 = native["stage1_cprime"]
    stage2 = native["stage2_cheapdct"]
    primary = receipt["native_dof_price_estimates"][0]
    od6_fix = receipt["od6_self_contained_fix"]
    md = f"""# OD8 native-DOF receipt - 2026-08-05

Status: `SCORER_FREE_NATIVE_DOF_PRICED / OD6_FORMAT_GAP_SELF_CONTAINED / NO FRONTIER MOVE`.

Axis: `[macOS-CPU cache-derived advisory / scorer-free byte pricing]`.
`score_claim=false`, `promotion_eligible=false`, `scorer_forwards_run=0`, `upstream_evaluate=false`.

## Answer First

OD8 identifies JS1 Stage-1 `cprime`'s native payload as sparse scorer-lattice frame_1 paint: the solved object is RGB `uint8` paint values on the block16 target band, plus the support needed for a decoder to apply those values.  On OD2's recorded n32 pair set, the actual recomputed support is `{stage1['band_px_total_n32']}` scorer pixels (`{stage1['band_px_mean_per_pair']:.2f}`/pair).  The optimized value DOF alone is `{stage1['optimized_value_params_n32']}` u8 values at n32; receiver-applicable sparse support+values is `{stage1['receiver_sparse_support_plus_values_raw_bytes_n32_fixed_u24']}` raw n32 B under a fixed u24 flat-index accounting, projected `{stage1['receiver_sparse_support_plus_values_raw_bytes_n600_fixed_u24_projection']}` raw n600 B before entropy coding.

Stage-2 `cheapdct{stage2['k']}` is confirmed from code and OD2 records as `k*k*3` int16 coefficients per pair.  For `k={stage2['k']}` that is `{stage2['params_per_pair']}` coefficients, `{stage2['raw_bytes_per_pair']}` B/pair, and `{stage2['raw_bytes_n600']}` raw B at n600.

The best OD8 scorer-free native stream estimate is `{primary['paint_mode']}`: `{primary['best_coder']['bytes']}` exact n32 coded B, linearly projected `{primary['projected_n600_bytes_linear']}` B at n600.  That is `{primary['vs_gc18_floor_90k']:+d}` B vs the top of GC18's 45-90K conjectured boundary-grammar floor and `{primary['vs_od6_packet_76304']:+d}` B vs OD6's 76,304 B packet projection.  This is an ESTIMATE, not a score: OD2/OD7 did not store actual solved paint or DCT coefficients.

OD8 fixes the OD6 b1024 format gap by producing a self-contained sibling packet at `{od6_fix['artifact']['path']}`.  The table qlogits are unchanged, and the packet now carries the sections required by its coverage hash: `pe1_generator_coords_n32`, `pe3_hybrid75_coords_n32`, and `pe3_hybrid_knee_coords_n32`.  Exact n32 best-coded bytes rise by `{od6_fix['delta_exact_n32_best_bytes_vs_od6_incumbent']:+d}` B vs the OD6 incumbent.  A conservative component-sum n600 projection is `{od6_fix['projected_n600_bytes_component_sum_conservative']}` B (`{od6_fix['delta_projected_component_sum_vs_od6_76304']:+d}` B vs 76,304 B).

## Native Price Table

{_table_md(receipt)}

## DOF Identification

| surface | object | dimensionality | quantization | receiver note |
|---|---|---:|---|---|
| Stage-1 `cprime` optimized values | RGB paint on frame_1 scorer-lattice support | `3 * band_px` values (`{stage1['optimized_value_params_n32']}` on OD2 n32) | uint8 selected through rounded proxy flips | support+values must be shipped for decoder application; offsets alone are not enough without scorer/GT argmax |
| Stage-1 receiver support | scorer-lattice flat indices for the band | `band_px` indices (`{stage1['band_px_total_n32']}` on OD2 n32) | sparse varint in OD8 packet; u24 raw accounting in table above | applies via `realize_scorer_paint_to_camera`, not stamps |
| Stage-2 `cheapdct4` | frame_0 low-frequency DCT coefficients | `4*4*3 = 48` coefficients/pair | int16 | basis is generic/free; coefficient values are counted |

## OD6 Format Gap Fix

| quantity | value |
|---|---:|
| source OD6 incumbent exact n32 best B | {receipt['od6_incumbent']['exact_n32_best_bytes']} |
| OD8 self-contained exact n32 best B | {od6_fix['best_coder']['bytes']} |
| exact n32 delta | {od6_fix['delta_exact_n32_best_bytes_vs_od6_incumbent']:+d} |
| OD6 incumbent projected n600 B | {receipt['od6_incumbent']['projected_n600_bytes']} |
| OD8 conservative component-sum projected n600 B | {od6_fix['projected_n600_bytes_component_sum_conservative']} |
| conservative projected delta | {od6_fix['delta_projected_component_sum_vs_od6_76304']:+d} |

Parse-back proof: OD5 packet reserializes exactly; OD6 table qlogits parse back unchanged; `coverage_source_sections` maps all three coverage columns to shipped sections.

## Determinism Check

No scorer was run.  OD8 recomputed the deterministic cprime target/band from cached `cx1_argmax_n600.npy` and `gt_argmax_n600.npy` for all `{native['recorded_outcome_denominators']['pairs']}` OD2 pairs.  `flips_before`, `band_px`, and `n_described` match OD2 recorded rows exactly.  OD8 also verified every recorded `cheapdct4` row carries `{stage2['raw_bytes_per_pair']}` B/pair and `value_measured_through_int16_quantiser=true`.

## RECALL EVIDENCE

| source/search | recalled fact | plan impact |
|---|---|---|
| `.omx/tmp/codex_runs/od8_prompt.md`, `_common_contract.md` | OD8 is scorer-free, must not touch JS1 running harness or od3 run dirs, and must use SSD for bulk. | Added a sibling tool only; ran only cache/byte pricing; wrote bulk packets under `/Volumes/VertigoDataTier/pact/ddm_od8_20260805/`. |
| `.omx/state/main_hot_state.md` | OD3 owns scorer slot; OD8 must persist native DOF and fix OD6 self-containment. | Kept the actual scorer re-derive in the post-OD3 fire order. |
| `OD7_RECEIVER_CLOSE_RECEIPT.md` | Stamp realization is formulation-dead; OD2/OD3 discard terminal fields; OD6 b1024 packet lacks coverage feature homes. | Priced native support+paint payloads and added missing OD6 coverage sections instead of stamping. |
| `OD2_STAGE12_RECEIPT.md` and OD2 JSON | Stage-1 outcomes are recorded but solved params are absent; k=4 carriage is 96 B/pair. | Treated native price as DERIVED/ESTIMATE until post-OD3 re-derive stores real values. |
| `OD6_DECODER_LEGAL_RECEIPT.md` and source packet | Incumbent b1024 exact n32 best is 7,334 B; projected n600 is 76,304 B; table uses the missing coverage features. | Produced a self-contained sibling packet and priced the delta. |
| `GC18_CONVOCATION_RECEIPT.md` | Boundary-grammar floor conjecture is 45-90K; OD6-style packet is the comparison point. | Reported native estimates directly against both bars. |
| canonical equations search (`receiver`, `payload`, `format`, `coefficient`, `counted`) | Receiver support, format-vs-search, and decoder-derived-context equations all require counted receiver-visible payloads. | Kept support bytes explicit and refused score/promotion wording. |
| bounded `rg` over `.omx/research`, `.omx/state`, `docs`, `experiments`, `src/tac`, `tools` for OD8/native/cprime/cheapdct/OD6 gap terms | Found no prior OD8 receipt beyond the live hot-state/charter; found OD6/OD7 gap statements and JS1 code. | Built the first OD8 receipt and scoped absence to the searched surfaces. |

## SHA Table

| artifact | bytes | sha256 |
|---|---:|---|
"""
    for row in receipt["sha_table"]:
        md += f"| `{row['path']}` | {row['bytes']} | `{row['sha256']}` |\n"
    md += f"""
## NEXT_IF_RESUMED

1. Wait until OD3 releases the scorer slot and its terminality receipt is stable.  Do not read or mutate OD3 run dirs while it is active.
2. If OD3's terminal artifact contains native payload fields, pass them as warm-start/custody inputs; if it contains only outcome JSON, re-run the same pair set with OD8 persistence because there are no coefficients to warm-start.
3. Fire:

```bash
.venv/bin/python experiments/ddm_od8_js1_persist.py solve-persist \\
  --allow-scorer \\
  --sub-dir /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2 \\
  --gt-mkv upstream/videos/0.mkv \\
  --pairs-npy .omx/research/ddm_od2_20260805/od2_pairs_n32_seed20260805.npy \\
  --argmax-cache /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache \\
  --out .omx/research/ddm_od8_20260805/od8_js1_persist_n32_cprime_k4.json \\
  --packet-out /Volumes/VertigoDataTier/pact/ddm_od8_20260805/native_dof/od8_native_dof_n32_cprime_k4.od5.raw_packet \\
  --block 16 --rmax 5 --seg-steps 100 --pose-steps 40 --eval-every 5 \\
  --dct-k 4 --threads 4 --resume
```

4. Price the persisted native packet:

```bash
.venv/bin/python experiments/ddm_od8_js1_persist.py price-persisted \\
  --persist-json .omx/research/ddm_od8_20260805/od8_js1_persist_n32_cprime_k4.json \\
  --packet /Volumes/VertigoDataTier/pact/ddm_od8_20260805/native_dof/od8_native_dof_n32_cprime_k4.od5.raw_packet \\
  --out .omx/research/ddm_od8_20260805/od8_persisted_native_dof_price.json
```

5. Replace ESTIMATE native bytes with MEASURED real-solved bytes, then stage the receiver candidate whose realization is coefficient/support application.

## Boundaries

- No SegNet/PoseNet forward, `upstream/evaluate.py`, contest-CPU, contest-CUDA, MPS, or n600 scorer job was run by OD8.
- Native price rows use actual OD2 support and base/noise proxy values because OD2/OD7 do not store solved paint or coefficients.
- OD6 self-contained packet is a format repair and byte price, not receiver-closed RGB/inflate/scorer survival.
- The frontier did not move.

{FINAL_FRONTIER_SENTENCE}
"""
    _atomic_write_text(path, md)


def analyze_recorded(args: argparse.Namespace) -> int:
    args.research_dir.mkdir(parents=True, exist_ok=True)
    storage = _storage_preflight(args.ssd_dir, required_free_bytes=512 * 1024 * 1024)
    run_id = args.run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_ssd = args.ssd_dir / run_id
    run_ssd.mkdir(parents=True, exist_ok=False)

    pairs, od2_rows, proofs = _load_od2_pair_surface(
        od2_json=args.od2_json,
        pair_selection=args.pair_selection,
        argmax_cache=args.argmax_cache,
        block=args.block,
        rmax=args.rmax,
    )
    dof = _native_dof_summary(pairs=pairs, od2_rows=od2_rows, proofs=proofs, k=args.dct_k)
    stage2 = _synthetic_stage2_records(pairs=pairs, od2_rows=od2_rows, k=args.dct_k)
    price_rows: list[dict[str, Any]] = []
    for paint_mode in ("base_rgb_proxy", "deterministic_noise_proxy"):
        stage1 = _stage1_records_from_base_rgb(
            pairs=pairs,
            argmax_cache=args.argmax_cache,
            sub_dir=args.sub_dir,
            block=args.block,
            rmax=args.rmax,
            paint_mode=paint_mode,
        )
        price_rows.append(
            _native_packet_price(
                stage1_records=stage1,
                stage2_records=stage2,
                paint_mode=paint_mode,
                ssd_dir=run_ssd,
            )
        )

    od6_fix = _build_od6_self_contained_packet(
        pairs=pairs,
        od6_packet=args.od6_packet,
        od6_receipt=args.od6_receipt,
        pe1_receipt_path=args.pe1_receipt,
        pe3_receipt_path=args.pe3_receipt,
        gt_cache=args.gt_cache,
        argmax_cache=args.argmax_cache,
        g4_recurrence=args.g4_recurrence,
        ssd_dir=run_ssd,
    )
    od6_payload = _load_json(args.od6_receipt)
    incumbent = od6_payload["best_rung_by_projected_s_with_od2_pose_credit"]
    receipt = {
        "schema": "ddm_od8_native_dof_receipt.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU cache-derived advisory / scorer-free byte pricing]",
        "score_claim": False,
        "promotion_eligible": False,
        "scorer_forwards_run": 0,
        "upstream_evaluate": False,
        "own_vehicle_frontier": OWN_FRONTIER_LINE,
        "contest_pointer": "borrowed/unmoved",
        "storage_preflight": storage,
        "run_ssd": str(run_ssd),
        "native_dof_identification": dof,
        "determinism_proofs": proofs,
        "native_dof_price_estimates": price_rows,
        "od6_incumbent": {
            "name": incumbent["name"],
            "exact_n32_best_bytes": int(incumbent["packet"]["best_coder"]["bytes"]),
            "projected_n600_bytes": int(
                incumbent["projection_with_od2_stage2_pose_credit"]["packet_bytes_n600_projected"]
            ),
        },
        "od6_self_contained_fix": od6_fix,
        "recall_evidence": [
            {
                "query": "MEMORY.md od8/OD8/codex_runs/common_contract/required-component",
                "finding": "no OD8-specific prior memory hit; #899 memory unrelated to this charter",
            },
            {
                "query": "canonical equations: receiver/native/format/coefficient/payload/decoder/counted",
                "finding": "receiver-visible counted payload and format-vs-search equations remain binding",
            },
            {
                "query": "bounded rg over research/state/docs/experiments/src/tools for OD8/native/cprime/cheapdct/OD6 gap",
                "finding": "OD8 appears only in live board and charter; OD6/OD7 receipts contain the active gaps",
            },
        ],
        "post_od3_fire_order": {
            "first_gate": "OD3 scorer slot released and terminal receipt stable",
            "warm_start_rule": (
                "warm-start only if OD3 terminal artifact includes native payload fields; otherwise re-run "
                "persistence because outcome JSON has no coefficients"
            ),
            "command": (
                ".venv/bin/python experiments/ddm_od8_js1_persist.py solve-persist --allow-scorer "
                "--sub-dir /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2 "
                "--gt-mkv upstream/videos/0.mkv "
                "--pairs-npy .omx/research/ddm_od2_20260805/od2_pairs_n32_seed20260805.npy "
                "--argmax-cache /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache "
                "--out .omx/research/ddm_od8_20260805/od8_js1_persist_n32_cprime_k4.json "
                "--packet-out /Volumes/VertigoDataTier/pact/ddm_od8_20260805/native_dof/"
                "od8_native_dof_n32_cprime_k4.od5.raw_packet "
                "--block 16 --rmax 5 --seg-steps 100 --pose-steps 40 --eval-every 5 "
                "--dct-k 4 --threads 4 --resume"
            ),
        },
        "boundaries": [
            "No scorer forward or n600 job run by OD8.",
            "No edits to experiments/ddm_js1_staging_discriminator.py or od3 run dirs.",
            "Native price is ESTIMATE until real post-OD3 persisted values are measured.",
            "OD6 packet fix is self-contained parse-back, not receiver-closed score.",
        ],
    }
    receipt_json = args.research_dir / "od8_native_dof_receipt.json"
    _atomic_write_json(receipt_json, receipt)
    sha_rows = [
        _source_file_entry(Path(__file__)),
        _source_file_entry(args.od2_json),
        _source_file_entry(args.pair_selection),
        _source_file_entry(args.od6_receipt),
        _source_file_entry(args.od6_packet),
        _source_file_entry(receipt_json),
    ]
    for row in price_rows:
        sha_rows.append(
            {
                "path": row["artifact"]["path"],
                "bytes": row["artifact"]["bytes"],
                "sha256": row["artifact"]["sha256"],
            }
        )
    sha_rows.append(
        {
            "path": od6_fix["artifact"]["path"],
            "bytes": od6_fix["artifact"]["bytes"],
            "sha256": od6_fix["artifact"]["sha256"],
        }
    )
    receipt["sha_table"] = sha_rows
    _atomic_write_json(receipt_json, receipt)
    _write_receipt_md(args.research_dir / "OD8_NATIVE_DOF_RECEIPT.md", receipt)
    print(json.dumps({"receipt": str(receipt_json), "md": str(args.research_dir / "OD8_NATIVE_DOF_RECEIPT.md")}, indent=2))
    return 0


def _solve_pose_repair_frame0_cheap_dct_persist(
    sc: Scorer,
    dec_f0: np.ndarray,
    edited_f1: np.ndarray,
    pose_gt,
    *,
    k: int,
    steps: int,
    lr: float,
    eval_every: int,
) -> tuple[float, np.ndarray | None, str, int, float, np.ndarray]:
    import torch
    from ddm_js1_staging_discriminator import d_pose_t, pose_forward_grad
    from scipy.fft import idctn

    posenet = sc.net.posenet
    base0 = resize_to_scorer(dec_f0)
    f1_s = resize_to_scorer(edited_f1).detach()
    atoms = np.zeros((k * k, SEG_H, SEG_W), np.float32)
    for a in range(k):
        for b in range(k):
            c = np.zeros((SEG_H, SEG_W))
            c[a, b] = 1.0
            atoms[a * k + b] = idctn(c, type=2, norm="ortho", axes=(-2, -1))
    A = torch.from_numpy(atoms).reshape(k * k, -1)

    def realized_dpose(t0: torch.Tensor) -> float:
        q = torch.round(torch.clamp(t0, 0.0, 255.0)).detach()
        with torch.no_grad():
            return float(d_pose_t(posenet, pose_gt, pose_forward_grad(posenet, q, f1_s)))

    zero_coeff = np.zeros((3, k * k), dtype=np.int16)
    best = (realized_dpose(base0), None, "identity@0", 0.0, zero_coeff)
    coef = torch.zeros(3, k * k, requires_grad=True)
    opt = torch.optim.Adam([coef], lr=lr)
    with torch.enable_grad():
        for it in range(steps + 1):
            d = (coef @ A).reshape(1, 3, SEG_H, SEG_W)
            cur = torch.clamp(base0 + d, 0.0, 255.0)
            if it % eval_every == 0 or it == steps:
                cq = torch.round(coef.detach()).clamp(-32768, 32767)
                dq = (cq @ A).reshape(1, 3, SEG_H, SEG_W)
                curq = torch.clamp(base0 + dq, 0.0, 255.0)
                dp = realized_dpose(curq)
                if dp < best[0]:
                    q = torch.round(curq).detach()
                    qcoef = cq.numpy().astype(np.int16)
                    best = (
                        dp,
                        q[0].permute(1, 2, 0).numpy().astype(np.uint8),
                        f"dct{k}q@{it}",
                        float(cq.abs().max()),
                        qcoef,
                    )
            if it == steps:
                break
            out = pose_forward_grad(posenet, cur, f1_s)
            loss = d_pose_t(posenet, pose_gt, out)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return best[0], best[1], best[2], int(coef.detach().numel() * 2), best[3], best[4]


def solve_persist(args: argparse.Namespace) -> int:
    if not args.allow_scorer:
        raise OD8Error("solve-persist requires --allow-scorer and must wait for OD3 scorer slot release")
    from ddm_js1_staging_discriminator import patch_yuv6_and_assert

    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.packet_out.parent.mkdir(parents=True, exist_ok=True)
    all_pairs = np.load(args.pairs_npy).tolist()
    if args.only_pairs:
        wanted = [int(x) for x in args.only_pairs.split(",")]
        missing = [pair for pair in wanted if pair not in all_pairs]
        if missing:
            raise OD8Error(f"--only-pairs not in pair set: {missing}")
        pairs = wanted
    else:
        pairs = [int(pair) for pair in all_pairs]

    rows: list[dict[str, Any]] = []
    if args.resume and args.out.exists():
        rows = json.loads(args.out.read_text()).get("rows", [])
        done = {int(row["pair"]) for row in rows}
        pairs = [pair for pair in pairs if pair not in done]
        print(f"[od8] resume: {len(rows)} rows on disk, {len(pairs)} remaining", flush=True)

    raw = np.memmap(
        args.sub_dir / "inflated/0.raw",
        dtype=np.uint8,
        mode="r",
        shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3),
    )
    current = np.load(args.argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gt_argmax = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")
    wanted_frames = {seq_len * p + o for p in pairs for o in (0, 1)}
    gt_frames = decode_gt_frames(args.gt_mkv, wanted_frames)
    sc = Scorer(args.threads)
    patch = patch_yuv6_and_assert(sc)
    print(f"[od8] scorer+patch ready t={time.time()-t0:.1f}s {patch}", flush=True)

    def flush() -> None:
        payload_stage1: list[Stage1NativeRecord] = []
        payload_stage2: list[Stage2DCTRecord] = []
        pair_payload_dir = args.packet_out.parent / "pair_payloads"
        for row in rows:
            npz = np.load(pair_payload_dir / f"pair_{int(row['pair']):04d}.npz")
            payload_stage1.append(
                Stage1NativeRecord(
                    pair=int(row["pair"]),
                    flat_indices=np.asarray(npz["stage1_flat"], dtype=np.int64),
                    rgb_values=np.asarray(npz["stage1_rgb"], dtype=np.uint8),
                    band_sha256=str(row["native_payload"]["stage1_band_sha256"]),
                    rgb_sha256=str(row["native_payload"]["stage1_rgb_sha256"]),
                )
            )
            qcoeffs = np.asarray(npz["stage2_qcoeffs"], dtype=np.int16)
            payload_stage2.append(
                Stage2DCTRecord(
                    pair=int(row["pair"]),
                    k=int(row["native_payload"]["stage2_k"]),
                    qcoeffs=qcoeffs,
                    coeff_sha256=str(row["native_payload"]["stage2_coeff_sha256"]),
                )
            )
        if payload_stage1:
            stage1 = _encode_stage1_records(payload_stage1, paint_mode="solved_paint_u8")
            stage2 = _encode_stage2_records(payload_stage2, source="measured_persisted_coefficients")
            packet = od4.serialize_od5_packet(
                [
                    od4.OD5Section("od8_stage1_solved_paint_u8", stage1),
                    od4.OD5Section("od8_stage2_cheapdct4_qcoeffs", stage2),
                ]
            )
            parsed = od4.parse_od5_packet(packet)
            if od4.serialize_od5_packet(parsed.sections) != packet:
                raise OD8Error("solve-persist packet parse-back failed")
            args.packet_out.write_bytes(packet)
        _atomic_write_json(
            args.out,
            {
                "schema": "ddm_od8_js1_persist.v1",
                "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                "score_claim": False,
                "promotion_eligible": False,
                "allow_scorer": True,
                "block": args.block,
                "rmax": args.rmax,
                "dct_k": args.dct_k,
                "budget": {
                    "seg_steps": args.seg_steps,
                    "seg_lr": args.seg_lr,
                    "pose_steps": args.pose_steps,
                    "dct_lr": args.dct_lr,
                    "eval_every": args.eval_every,
                },
                "yuv6_patch": patch,
                "packet_out": str(args.packet_out),
                "rows": rows,
            },
        )

    for idx, pair in enumerate(pairs):
        tp = time.time()
        dec = np.stack([raw[seq_len * pair], raw[seq_len * pair + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * pair], gt_frames[seq_len * pair + 1]])
        cur = np.asarray(current[pair], dtype=np.uint8)
        tgt_gt = np.asarray(gt_argmax[pair], dtype=np.uint8)
        offsets = solve_blocks(cur, tgt_gt, args.block, args.rmax)
        target = translate_blocks(cur, offsets.reshape(-1, 2), args.block)
        band = target != cur
        _, paint, tag, solve_diag = solve_margin_optimal_paint(
            sc.net.segnet,
            dec[1],
            gt[1],
            band,
            target,
            steps=args.seg_steps,
            lr=args.seg_lr,
            eval_every=args.eval_every,
        )
        edited_f1 = realize_scorer_paint_to_camera(dec[1], band, paint)
        pose_gt = sc.pose_out(gt)
        dp_s1 = sc.d_pose(pose_gt, sc.pose_out(np.stack([dec[0], edited_f1])))
        dp_c, paint0c, tagc, cbytes, cmax, qcoef = _solve_pose_repair_frame0_cheap_dct_persist(
            sc,
            dec[0],
            edited_f1,
            pose_gt,
            k=args.dct_k,
            steps=args.pose_steps,
            lr=args.dct_lr,
            eval_every=args.eval_every,
        )
        flat = np.flatnonzero(band.reshape(-1)).astype(np.int64)
        rgb_values = np.ascontiguousarray(paint.reshape(-1, 3)[flat].astype(np.uint8))
        pair_payload_dir = args.packet_out.parent / "pair_payloads"
        pair_payload_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pair_payload_dir / f"pair_{pair:04d}.npz",
            stage1_flat=flat,
            stage1_rgb=rgb_values,
            stage2_qcoeffs=qcoef,
        )
        row = {
            "pair": int(pair),
            "stage1": {
                "solve_tag": tag,
                "flips_before": int((cur != tgt_gt).sum()),
                "band_px": int(band.sum()),
                "n_described": int((cur != tgt_gt).sum() - int((target != tgt_gt).sum())),
                "solve_diag_selected": solve_diag["selected"],
                "d_pose_after_stage1": dp_s1,
            },
            "stage2_cheapdct": {
                "tag": tagc,
                "d_pose_verified_from_camera": dp_c,
                "counted_bytes_per_pair": cbytes,
                "max_abs_int16_coefficient": cmax,
            },
            "native_payload": {
                "stage1_support_count": int(flat.size),
                "stage1_band_sha256": _sha256_bytes(
                    np.packbits(band.reshape(-1), bitorder="little").tobytes()
                ),
                "stage1_rgb_sha256": _sha256_bytes(rgb_values.tobytes()),
                "stage2_k": args.dct_k,
                "stage2_coeff_sha256": _sha256_bytes(np.ascontiguousarray(qcoef.astype("<i2")).tobytes()),
            },
        }
        rows.append(row)
        flush()
        print(
            f"[od8] pair {pair:3d} ({idx+1}/{len(pairs)}) band {int(band.sum())} "
            f"s1 {tag} dpose_s1 {dp_s1:.6g} dct {tagc} [{time.time()-tp:.1f}s]",
            flush=True,
        )
    flush()
    print(f"[od8] DONE {len(rows)} rows t={time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


def price_persisted(args: argparse.Namespace) -> int:
    payload = _load_json(args.persist_json)
    packet = args.packet.read_bytes()
    parsed = od4.parse_od5_packet(packet)
    if od4.serialize_od5_packet(parsed.sections) != packet:
        raise OD8Error("persisted native packet did not reserialize exactly")
    coder_rows = od4.race_packet_coders(packet)
    best = _best_coder(coder_rows)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise OD8Error(f"persist JSON has no rows: {args.persist_json}")
    n_pairs = len(rows)
    out = {
        "schema": "ddm_od8_persisted_native_dof_price.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU byte-only persisted-native pricing]",
        "score_claim": False,
        "promotion_eligible": False,
        "scorer_forwards_run": 0,
        "persist_json": _source_file_entry(args.persist_json),
        "packet": _source_file_entry(args.packet),
        "packet_parseback_exact": True,
        "section_count": parsed.section_count,
        "sections": [
            {
                "name": section.name,
                "payload_bytes": len(section.payload),
                "payload_sha256": od4.sha256_bytes(section.payload),
            }
            for section in parsed.sections
        ],
        "n_pairs": n_pairs,
        "coder_race": [row.as_json() for row in coder_rows],
        "best_coder": best.as_json(),
        "projected_n600_bytes_linear": od4.projected_n600_packet_bytes(best.bytes, n_pairs),
        "projection_scope": "exact persisted n-pair packet bytes with linear n600 projection; byte-only",
    }
    _atomic_write_json(args.out, out)
    print(json.dumps({"out": str(args.out), "best_bytes": best.bytes}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    ana = sub.add_parser("analyze-recorded", help="scorer-free OD8 receipt and price generation")
    ana.add_argument("--od2-json", type=Path, default=DEFAULT_OD2_JSON)
    ana.add_argument("--pair-selection", type=Path, default=DEFAULT_PAIR_SELECTION)
    ana.add_argument("--argmax-cache", type=Path, default=DEFAULT_ARGMAX_CACHE)
    ana.add_argument("--sub-dir", type=Path, default=DEFAULT_SUB_DIR)
    ana.add_argument("--od6-receipt", type=Path, default=DEFAULT_OD6_RECEIPT)
    ana.add_argument("--od6-packet", type=Path, default=DEFAULT_OD6_PACKET)
    ana.add_argument("--pe1-receipt", type=Path, default=DEFAULT_PE1_RECEIPT)
    ana.add_argument("--pe3-receipt", type=Path, default=DEFAULT_PE3_RECEIPT)
    ana.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    ana.add_argument("--g4-recurrence", type=Path, default=DEFAULT_G4_RECURRENCE)
    ana.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    ana.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    ana.add_argument("--run-id", default=None)
    ana.add_argument("--block", type=int, default=16)
    ana.add_argument("--rmax", type=int, default=5)
    ana.add_argument("--dct-k", type=int, default=4)
    ana.set_defaults(func=analyze_recorded)

    sol = sub.add_parser("solve-persist", help="post-OD3 scorer-enabled persistent JS1 re-derive")
    sol.add_argument("--allow-scorer", action="store_true")
    sol.add_argument("--sub-dir", type=Path, required=True)
    sol.add_argument("--gt-mkv", type=Path, required=True)
    sol.add_argument("--pairs-npy", type=Path, default=DEFAULT_PAIRS_NPY)
    sol.add_argument("--argmax-cache", type=Path, required=True)
    sol.add_argument("--out", type=Path, required=True)
    sol.add_argument("--packet-out", type=Path, required=True)
    sol.add_argument("--only-pairs", default="")
    sol.add_argument("--block", type=int, default=16)
    sol.add_argument("--rmax", type=int, default=5)
    sol.add_argument("--seg-steps", type=int, default=100)
    sol.add_argument("--seg-lr", type=float, default=2.0)
    sol.add_argument("--pose-steps", type=int, default=40)
    sol.add_argument("--dct-k", type=int, default=4)
    sol.add_argument("--dct-lr", type=float, default=20.0)
    sol.add_argument("--eval-every", type=int, default=5)
    sol.add_argument("--threads", type=int, default=4)
    sol.add_argument("--resume", action="store_true")
    sol.set_defaults(func=solve_persist)

    price = sub.add_parser("price-persisted", help="byte-only price of a real solve-persist packet")
    price.add_argument("--persist-json", type=Path, required=True)
    price.add_argument("--packet", type=Path, required=True)
    price.add_argument("--out", type=Path, required=True)
    price.set_defaults(func=price_persisted)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
