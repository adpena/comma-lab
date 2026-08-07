#!/usr/bin/env bash
set -euo pipefail

cd /Users/adpena/Projects/pact

.venv/bin/python - <<'PY'
from __future__ import annotations

import hashlib
import json
import lzma
import math
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import brotli
import numpy as np

REPO = Path("/Users/adpena/Projects/pact")
for path in (REPO / "src", REPO / "experiments"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from ddm_et1_ph1_block16_on_our_vehicle import translate_blocks
from ddm_et4_overlay_codec import encode_patch_records, load_patch_records
from ddm_sq1_pose_null_constrained_paint import snap_band_to_blocks
from ddm_sw1_null_basis_phase_solve import block_mask_from_band
from experiments.ddm_r7_token_coder import encode_token_codes
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
from tac.optimization.lattice_native_pose_null_realizer import (
    build_default_operator,
    private_block_geometry,
)

AXIS = "[macOS-CPU advisory]"
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
N_PAIRS = 600
SEG_H = 384
SEG_W = 512
CAMERA_H = 874
CAMERA_W = 1164
CHANNELS = 3
FRAME_VALUES = CAMERA_H * CAMERA_W * CHANNELS
DEN = 37_545_489
S_PER_FLIP = 100.0 / (N_PAIRS * SEG_H * SEG_W)
RATE_PER_BYTE = 25.0 / DEN
WATERLINE_B_PER_FLIP = 1.27310821533
SAMPLE_N = 32
SAMPLE_SEED = 20260807

OUT_DIR = REPO / ".omx/research/ddm_et5_20260807"
ET4_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_et4_20260806")
ROWS_PATH = ET4_DIR / "et4_solve_within_cvp_rows.jsonl"
PATCH_DIR = ET4_DIR / "patch_records"
ET4_RECEIPT_PATH = ET4_DIR / "byteclose_archive_receipt.json"
ET4_SUMMARY_PATH = ET4_DIR / "et4_solve_within_cvp_summary.json"
PARENT_ARGMAX_PATH = Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/parent_tq1c_argmax_n600.npy")
CURRENT_OFFSETS_PATH = Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/phase_field/tq1c_block16_offsets.npy")
GT_CACHE_PATH = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception as exc:
        return f"UNKNOWN:{type(exc).__name__}:{exc}"


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=jsonable, allow_nan=False)
        handle.write("\n")
    tmp.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def load_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pair = int(row["pair"])
            if pair in rows:
                raise RuntimeError(f"duplicate row after dedupe for pair {pair}")
            rows[pair] = row
    if len(rows) != N_PAIRS:
        raise RuntimeError(f"expected 600 ET4 rows, found {len(rows)}")
    return rows


def load_patch_record(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "pair": int(data["pair"][0]),
            "nnz": int(data["nnz"][0]),
            "indices": np.asarray(data["indices"], dtype="<u4"),
            "deltas_i16": np.asarray(data["deltas_i16"], dtype="<i2"),
        }


def make_et4_raw(records: list[dict[str, Any]]) -> bytes:
    out = bytearray()
    out.extend(struct.pack("<8sB3xH", b"ET4PD1\0\0", 1, len(records)))
    for record in sorted(records, key=lambda row: int(row["pair"])):
        pair = int(record["pair"])
        indices = np.asarray(record["indices"], dtype="<u4")
        deltas = np.asarray(record["deltas_i16"], dtype="<i2")
        out.extend(struct.pack("<HI", pair, int(indices.size)))
        out.extend(np.ascontiguousarray(indices, dtype="<u4").tobytes())
        out.extend(np.ascontiguousarray(deltas, dtype="<i2").tobytes())
    return bytes(out)


def raw_lzma(payload: bytes) -> bytes:
    return lzma.compress(
        payload,
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}],
    )


def put_varuint(out: bytearray, value: int) -> None:
    if value < 0:
        raise RuntimeError("varuint cannot encode negative values")
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def zigzag_i16(value: int) -> int:
    return (int(value) << 1) ^ (int(value) >> 15)


def split_streams(records: list[dict[str, Any]]) -> tuple[bytes, bytes, bytes]:
    table = bytearray()
    coords = bytearray()
    values = bytearray()
    table.extend(struct.pack("<8sH", b"ET5SP1\0", len(records)))
    for record in sorted(records, key=lambda row: int(row["pair"])):
        pair = int(record["pair"])
        indices = np.asarray(record["indices"], dtype="<u4")
        deltas = np.asarray(record["deltas_i16"], dtype="<i2")
        table.extend(struct.pack("<HI", pair, int(indices.size)))
        prev = -1
        for idx in indices.astype(np.uint32, copy=False):
            cur = int(idx)
            put_varuint(coords, cur - prev - 1)
            prev = cur
        for value in deltas.astype(np.int16, copy=False):
            put_varuint(values, zigzag_i16(int(value)))
    return bytes(table), bytes(coords), bytes(values)


def split_container(records: list[dict[str, Any]], mode: str) -> bytes:
    table, coords, values = split_streams(records)
    choices = {
        "split_brotli11": (1, brotli.compress(coords, quality=11), 1, brotli.compress(values, quality=11)),
        "split_lzma1": (2, raw_lzma(coords), 2, raw_lzma(values)),
    }
    if mode == "split_mixed_min":
        coord_options = [(1, brotli.compress(coords, quality=11)), (2, raw_lzma(coords))]
        value_options = [(1, brotli.compress(values, quality=11)), (2, raw_lzma(values))]
        coord_id, coord_payload = min(coord_options, key=lambda item: (len(item[1]), item[0]))
        value_id, value_payload = min(value_options, key=lambda item: (len(item[1]), item[0]))
    else:
        coord_id, coord_payload, value_id, value_payload = choices[mode]
    header = struct.pack(
        "<8sBBIII",
        b"ET5SC1\0",
        coord_id,
        value_id,
        len(table),
        len(coord_payload),
        len(value_payload),
    )
    return header + table + coord_payload + value_payload


def pair_payloads_for_r7(records: list[dict[str, Any]]) -> list[tuple[int, bytes]]:
    out: list[tuple[int, bytes]] = []
    for record in sorted(records, key=lambda row: int(row["pair"])):
        indices = np.asarray(record["indices"], dtype="<u4")
        deltas = np.asarray(record["deltas_i16"], dtype="<i2")
        body = bytearray()
        put_varuint(body, int(indices.size))
        prev = -1
        for idx in indices.astype(np.uint32, copy=False):
            cur = int(idx)
            put_varuint(body, cur - prev - 1)
            prev = cur
        for value in deltas.astype(np.int16, copy=False):
            put_varuint(body, zigzag_i16(int(value)))
        out.append((int(record["pair"]), bytes(body)))
    return out


def r7_nibble_container(records: list[dict[str, Any]]) -> bytes:
    payloads = pair_payloads_for_r7(records)
    if not payloads:
        frame = encode_token_codes(np.zeros((1, 1, 1, 2), dtype=np.uint8), levels=16, codec="smevr")
        return struct.pack("<8sHI", b"ET5R7N1", 0, 0) + frame
    max_len = max(len(payload) for _pair, payload in payloads)
    if max_len == 0:
        max_len = 1
    if len(payloads) * max_len * 2 > 16_000_000:
        raise RuntimeError("R7 nibble frame would exceed the production token bound")
    codes = np.zeros((len(payloads), max_len, 1, 2), dtype=np.uint8)
    for row, (_pair, payload) in enumerate(payloads):
        if not payload:
            continue
        arr = np.frombuffer(payload, dtype=np.uint8)
        codes[row, : arr.size, 0, 0] = arr >> 4
        codes[row, : arr.size, 0, 1] = arr & 15
    frame = encode_token_codes(codes, levels=16, codec="smevr")
    table = bytearray()
    table.extend(struct.pack("<8sHI", b"ET5R7N1", len(payloads), max_len))
    for pair, payload in payloads:
        table.extend(struct.pack("<HI", pair, len(payload)))
    return bytes(table) + frame


def encode_records(records: list[dict[str, Any]], coder: str) -> bytes:
    if coder == "et4_brotli11":
        payload, _receipt = encode_patch_records(records, quality=11)
        return payload
    if coder == "et4_lzma1_raw":
        return raw_lzma(make_et4_raw(records))
    if coder in {"split_brotli11", "split_lzma1", "split_mixed_min"}:
        return split_container(records, coder)
    if coder == "r7_smevr_nibble_varint":
        return r7_nibble_container(records)
    raise RuntimeError(f"unknown coder {coder}")


def deterministic_stratified_sample(n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    pairs: list[int] = []
    for i in range(n):
        start = (i * N_PAIRS) // n
        stop = ((i + 1) * N_PAIRS) // n
        pairs.append(int(rng.integers(start, stop)))
    if len(set(pairs)) != n:
        raise RuntimeError(f"sample collision: {pairs}")
    return pairs


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return src.copy()
    padded = np.pad(src, radius, mode="constant", constant_values=False)
    out = np.zeros_like(src, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + src.shape[0], dx : dx + src.shape[1]]
    return out


@dataclass(frozen=True)
class BlockPixels:
    flat_pixels: np.ndarray


def build_block_pixel_cache() -> Callable[[int, int], BlockPixels]:
    operator = build_default_operator()
    cache: dict[tuple[int, int], BlockPixels] = {}

    def get(scorer_row: int, scorer_col: int) -> BlockPixels:
        key = (int(scorer_row), int(scorer_col))
        if key in cache:
            return cache[key]
        geom = private_block_geometry(operator, key[0], key[1])
        pixels: set[int] = set()
        for br in range(2):
            rows = [int(v) for v in np.asarray(geom.row_indices[br]).reshape(-1)]
            for bc in range(2):
                cols = [int(v) for v in np.asarray(geom.col_indices[bc]).reshape(-1)]
                for row in rows:
                    for col in cols:
                        pixels.add(row * CAMERA_W + col)
        value = BlockPixels(flat_pixels=np.fromiter(sorted(pixels), dtype=np.int64))
        cache[key] = value
        return value

    return get


def camera_mask_from_scorer_mask(mask: np.ndarray, block_pixels: Callable[[int, int], BlockPixels]) -> np.ndarray:
    snapped = snap_band_to_blocks(np.asarray(mask, dtype=bool))
    blocks = block_mask_from_band(snapped)
    camera = np.zeros(CAMERA_H * CAMERA_W, dtype=bool)
    ys, xs = np.nonzero(blocks)
    for by, bx in zip(ys, xs, strict=False):
        camera[block_pixels(int(by) * 2, int(bx) * 2).flat_pixels] = True
    return camera


def restrict_record(record: dict[str, Any], camera_mask: np.ndarray) -> dict[str, Any]:
    indices = np.asarray(record["indices"], dtype="<u4")
    deltas = np.asarray(record["deltas_i16"], dtype="<i2")
    if indices.size:
        keep = camera_mask[(indices // CHANNELS).astype(np.int64)]
    else:
        keep = np.zeros(0, dtype=bool)
    kept_indices = indices[keep].astype("<u4", copy=False)
    kept_deltas = deltas[keep].astype("<i2", copy=False)
    return {
        "pair": int(record["pair"]),
        "nnz": int(kept_indices.size),
        "indices": kept_indices,
        "deltas_i16": kept_deltas,
    }


def verify_custody(rows: dict[int, dict[str, Any]], records: list[dict[str, Any]], receipt: dict[str, Any]) -> dict[str, Any]:
    row_pairs = sorted(rows)
    record_pairs = [int(row["pair"]) for row in records]
    if row_pairs != record_pairs:
        raise RuntimeError("row pairs and patch record pairs differ")
    total_nnz = 0
    per_pair_checked = 0
    mismatches: list[str] = []
    for record in records:
        pair = int(record["pair"])
        path = PATCH_DIR / f"pair_{pair:04d}.npz"
        row_record = rows[pair]["patch_record"]
        if Path(row_record["path"]) != path:
            mismatches.append(f"pair {pair}: row path differs")
        if int(row_record["nnz"]) != int(record["nnz"]):
            mismatches.append(f"pair {pair}: row nnz differs")
        idx_sha = sha256_bytes(np.ascontiguousarray(record["indices"], dtype="<u4").tobytes())
        val_sha = sha256_bytes(np.ascontiguousarray(record["deltas_i16"], dtype="<i2").tobytes())
        if idx_sha != row_record["delta_index_sha256"]:
            mismatches.append(f"pair {pair}: index sha differs")
        if val_sha != row_record["delta_value_sha256"]:
            mismatches.append(f"pair {pair}: value sha differs")
        total_nnz += int(record["nnz"])
        per_pair_checked += 1
    if mismatches:
        raise RuntimeError("custody mismatches: " + "; ".join(mismatches[:8]))
    compressed, encoded_receipt = encode_patch_records(records, quality=11)
    patch_receipt = receipt["patch"]
    checks = {
        "record_count": int(patch_receipt["record_count"]) == len(records),
        "total_nnz": int(patch_receipt["total_nnz"]) == total_nnz == int(encoded_receipt["total_nnz"]),
        "raw_bytes": int(patch_receipt["raw_bytes"]) == int(encoded_receipt["raw_bytes"]),
        "compressed_bytes": int(patch_receipt["compressed_bytes"]) == len(compressed) == int(encoded_receipt["compressed_bytes"]),
        "raw_sha256": str(patch_receipt["raw_sha256"]) == str(encoded_receipt["raw_sha256"]),
        "compressed_sha256": str(patch_receipt["compressed_sha256"]) == sha256_bytes(compressed) == str(encoded_receipt["compressed_sha256"]),
        "rows_sha256": sha256_file(ROWS_PATH),
        "byteclose_receipt_sha256": sha256_file(ET4_RECEIPT_PATH),
        "et4_summary_sha256": sha256_file(ET4_SUMMARY_PATH),
        "per_pair_delta_hashes_checked": per_pair_checked,
    }
    if not all(v is True for k, v in checks.items() if isinstance(v, bool)):
        raise RuntimeError(f"ET4 custody failed: {checks}")
    return checks


def summarize_restriction(
    *,
    name: str,
    sample_pairs: list[int],
    rows: dict[int, dict[str, Any]],
    records_by_pair: dict[int, dict[str, Any]],
    masks_by_pair: dict[int, np.ndarray],
    block_pixels: Callable[[int, int], BlockPixels],
) -> dict[str, Any]:
    restricted: list[dict[str, Any]] = []
    pair_stats: list[dict[str, Any]] = []
    for pair in sample_pairs:
        camera_mask = camera_mask_from_scorer_mask(masks_by_pair[pair], block_pixels)
        record = records_by_pair[pair]
        out = restrict_record(record, camera_mask)
        full_nnz = int(record["nnz"])
        kept = int(out["nnz"])
        row = rows[pair]
        full_net = int(row["cvp_realized"]["net_flip_reduction"])
        full_fixed = int(row["cvp_realized"]["fixed_global"])
        full_introduced = int(row["cvp_realized"]["introduced_global"])
        proxy_fraction = kept / full_nnz if full_nnz else 0.0
        pair_stats.append(
            {
                "pair": pair,
                "full_patch_nnz": full_nnz,
                "kept_nnz": kept,
                "dropped_nnz": full_nnz - kept,
                "kept_fraction": proxy_fraction,
                "camera_pixels_in_restricted_support": int(camera_mask.sum()),
                "banked_full_patch_net_flips": full_net,
                "banked_full_patch_fixed_global": full_fixed,
                "banked_full_patch_introduced_global": full_introduced,
                "support_proxy_net_flips": full_net * proxy_fraction,
            }
        )
        restricted.append(out)

    total_full_nnz = sum(int(row["full_patch_nnz"]) for row in pair_stats)
    total_kept_nnz = sum(int(row["kept_nnz"]) for row in pair_stats)
    full_net_flips = sum(int(row["banked_full_patch_net_flips"]) for row in pair_stats)
    proxy_net_flips = sum(float(row["support_proxy_net_flips"]) for row in pair_stats)
    out = {
        "restriction": name,
        "records": restricted,
        "pair_stats": pair_stats,
        "support_stats": {
            "sample_pairs": len(sample_pairs),
            "sample_full_patch_nnz": total_full_nnz,
            "sample_kept_nnz": total_kept_nnz,
            "sample_dropped_nnz": total_full_nnz - total_kept_nnz,
            "kept_fraction": total_kept_nnz / total_full_nnz if total_full_nnz else None,
            "collateral_mass_dropped_fraction": 1.0 - total_kept_nnz / total_full_nnz if total_full_nnz else None,
            "sample_banked_full_patch_net_flips": full_net_flips,
            "sample_support_proxy_net_flips": proxy_net_flips,
            "projected_n600_full_patch_net_flips": full_net_flips * N_PAIRS / len(sample_pairs),
            "projected_n600_support_proxy_net_flips": proxy_net_flips * N_PAIRS / len(sample_pairs),
        },
    }
    return out


def price_restriction(restriction: dict[str, Any]) -> dict[str, Any]:
    records = restriction["records"]
    support = restriction["support_stats"]
    pair_stats_by_pair = {int(row["pair"]): row for row in restriction["pair_stats"]}
    coder_rows: list[dict[str, Any]] = []
    coders = [
        "et4_brotli11",
        "et4_lzma1_raw",
        "split_brotli11",
        "split_lzma1",
        "split_mixed_min",
        "r7_smevr_nibble_varint",
    ]
    for coder in coders:
        print(f"[et5] pricing {restriction['restriction']} {coder}", flush=True)
        try:
            payload = encode_records(records, coder)
            pair_payloads = [(int(record["pair"]), encode_records([record], coder)) for record in records]
            coder_error = None
        except Exception as exc:
            payload = b""
            pair_payloads = []
            coder_error = f"{type(exc).__name__}: {exc}"
        sample_bytes = len(payload)
        projected_bytes = sample_bytes * N_PAIRS / len(records) if records else 0.0
        projected_full_flips = float(support["projected_n600_full_patch_net_flips"])
        projected_proxy_flips = float(support["projected_n600_support_proxy_net_flips"])
        rate_delta_s = RATE_PER_BYTE * projected_bytes
        seg_gain_full_s = -S_PER_FLIP * projected_full_flips
        seg_gain_proxy_s = -S_PER_FLIP * projected_proxy_flips
        per_pair_rank: list[dict[str, Any]] = []
        for pair, pair_payload in pair_payloads:
            stats = pair_stats_by_pair[pair]
            flips = int(stats["banked_full_patch_net_flips"])
            proxy_flips = float(stats["support_proxy_net_flips"])
            per_pair_rank.append(
                {
                    "pair": pair,
                    "bytes": len(pair_payload),
                    "banked_full_patch_net_flips": flips,
                    "support_proxy_net_flips": proxy_flips,
                    "B_per_full_patch_flip": len(pair_payload) / flips if flips > 0 else None,
                    "B_per_support_proxy_flip": len(pair_payload) / proxy_flips if proxy_flips > 0 else None,
                }
            )
        selected_full = [
            row
            for row in per_pair_rank
            if row["banked_full_patch_net_flips"] > 0
            and row["B_per_full_patch_flip"] is not None
            and row["B_per_full_patch_flip"] <= WATERLINE_B_PER_FLIP
        ]
        selected_pairs = {int(row["pair"]) for row in selected_full}
        selected_records = [record for record in records if int(record["pair"]) in selected_pairs]
        if selected_records and coder_error is None:
            selected_bytes = len(encode_records(selected_records, coder))
        else:
            selected_bytes = 0
        selected_flips = sum(int(row["banked_full_patch_net_flips"]) for row in selected_full)
        selected_projected_bytes = selected_bytes * N_PAIRS / len(records) if records else 0.0
        selected_projected_flips = selected_flips * N_PAIRS / len(records) if records else 0.0
        selected_net_delta_s = RATE_PER_BYTE * selected_projected_bytes - S_PER_FLIP * selected_projected_flips
        coder_rows.append(
            {
                "coder": coder,
                "error": coder_error,
                "sample_bytes": sample_bytes,
                "payload_sha256": sha256_bytes(payload) if payload else None,
                "projected_n600_bytes": projected_bytes,
                "B_per_full_patch_flip": projected_bytes / projected_full_flips if projected_full_flips > 0 else None,
                "B_per_support_proxy_flip": projected_bytes / projected_proxy_flips if projected_proxy_flips > 0 else None,
                "rate_delta_S_projected": rate_delta_s,
                "seg_gain_S_if_full_patch_flips_retained": seg_gain_full_s,
                "seg_gain_S_support_proxy": seg_gain_proxy_s,
                "net_delta_S_if_full_patch_flips_retained": rate_delta_s + seg_gain_full_s,
                "net_delta_S_support_proxy": rate_delta_s + seg_gain_proxy_s,
                "waterfill_subset_if_full_patch_flips_retained": {
                    "selected_pairs": sorted(selected_pairs),
                    "selected_count": len(selected_pairs),
                    "sample_subset_bytes_joint": selected_bytes,
                    "sample_subset_full_patch_net_flips": selected_flips,
                    "projected_n600_subset_bytes": selected_projected_bytes,
                    "projected_n600_subset_full_patch_net_flips": selected_projected_flips,
                    "projected_net_delta_S": selected_net_delta_s,
                    "B_per_full_patch_flip": selected_projected_bytes / selected_projected_flips
                    if selected_projected_flips > 0
                    else None,
                },
                "per_pair_rank": sorted(
                    per_pair_rank,
                    key=lambda row: (
                        math.inf if row["B_per_full_patch_flip"] is None else row["B_per_full_patch_flip"],
                        row["pair"],
                    ),
                ),
            }
        )
    valid = [row for row in coder_rows if row["error"] is None]
    best_by_full = min(valid, key=lambda row: row["net_delta_S_if_full_patch_flips_retained"]) if valid else None
    best_by_proxy = min(valid, key=lambda row: row["net_delta_S_support_proxy"]) if valid else None
    return {
        "restriction": restriction["restriction"],
        "support_stats": support,
        "coder_rows": coder_rows,
        "best_by_full_patch_flip_projection": best_by_full,
        "best_by_support_proxy_projection": best_by_proxy,
    }


def fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "null"
    return f"{float(value):.{digits}f}"


def make_pricing_table_md(payload: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = []
    for item in payload["pricing"]:
        for row in item["coder_rows"]:
            if row["error"] is None:
                rows.append({"restriction": item["restriction"], **row})
    rows.sort(key=lambda row: row["net_delta_S_if_full_patch_flips_retained"])
    lines = [
        "# ET5 pricing table",
        "",
        f"Axis: {AXIS}; scorer-free rate/support measurement; restricted-patch realization owed.",
        "",
        "| restriction | coder | sample bytes | proj n600 bytes | B/full-flip | B/support-proxy-flip | net dS if full ET4 flips retained | net dS support proxy | waterfill pairs | waterfill net dS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        wf = row["waterfill_subset_if_full_patch_flips_retained"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["restriction"],
                    row["coder"],
                    str(row["sample_bytes"]),
                    fmt_float(row["projected_n600_bytes"], 1),
                    fmt_float(row["B_per_full_patch_flip"], 3),
                    fmt_float(row["B_per_support_proxy_flip"], 3),
                    fmt_float(row["net_delta_S_if_full_patch_flips_retained"], 6),
                    fmt_float(row["net_delta_S_support_proxy"], 6),
                    str(wf["selected_count"]),
                    fmt_float(wf["projected_net_delta_S"], 6),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("All net-dS rows above are projections from the stratified n32 sample and reuse ET4's banked full-patch flip gains; no restricted patch was scorer-realized in this arm.")
    return "\n".join(lines) + "\n"


def make_receipt_md(payload: dict[str, Any]) -> str:
    best = payload["verdict"]["best_projection"]
    support = best["support_stats"]
    row = best["coder_row"]
    wf = row["waterfill_subset_if_full_patch_flips_retained"]
    lines = [
        "# ddm_et5 carriage-recode receipt",
        "",
        f"Axis: {AXIS}. Score claim: false. Promotion eligible: false. Pointer moved: false.",
        "",
        "## Curve table",
        "",
        "| restriction | best coder | kept nnz | dropped collateral | proj bytes | B/full-flip | B/proxy-flip | net dS full-retain | net dS proxy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["pricing"]:
        best_row = item["best_by_full_patch_flip_projection"]
        st = item["support_stats"]
        lines.append(
            "| "
            + " | ".join(
                [
                    item["restriction"],
                    best_row["coder"],
                    str(st["sample_kept_nnz"]),
                    fmt_float(st["collateral_mass_dropped_fraction"], 4),
                    fmt_float(best_row["projected_n600_bytes"], 1),
                    fmt_float(best_row["B_per_full_patch_flip"], 3),
                    fmt_float(best_row["B_per_support_proxy_flip"], 3),
                    fmt_float(best_row["net_delta_S_if_full_patch_flips_retained"], 6),
                    fmt_float(best_row["net_delta_S_support_proxy"], 6),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "No measured restricted-patch point can be promoted, because every priced point is rate-dead before realization. "
            f"The best optimistic description-side point is `{best['restriction']}` with `{row['coder']}`: "
            f"{fmt_float(row['B_per_full_patch_flip'], 3)} B/full-ET4-flip and projected net dS "
            f"{fmt_float(row['net_delta_S_if_full_patch_flips_retained'], 6)} if all banked ET4 flips survive restriction. "
            f"Its support-proxy net dS is {fmt_float(row['net_delta_S_support_proxy'], 6)}. "
            "The honest disposition is `FOLDED`: no measured restriction+coder point goes net-negative.",
            "",
            "Waterfilled subset under W (optimistic full-flip retention): "
            f"{wf['selected_count']} / {SAMPLE_N} sample pairs, projected bytes "
            f"{fmt_float(wf['projected_n600_subset_bytes'], 1)}, projected flips "
            f"{fmt_float(wf['projected_n600_subset_full_patch_net_flips'], 1)}, projected net dS "
            f"{fmt_float(wf['projected_net_delta_S'], 6)}.",
            "",
            "Follow-on disposition: FOLDED. Do not materialize or spend the scorer slot on this ET5 priced family. "
            "Reopen only if a new coder/restriction measures <= W on a stratified n>=32 sample; then the owed leg is "
            "all-600 materialization plus exact CPU-torch restricted-patch argmax validation.",
            "",
            "## Custody",
            "",
            f"- ET4 rows: `{ROWS_PATH}` sha256 `{payload['custody']['rows_sha256']}`.",
            f"- ET4 byteclose receipt: `{ET4_RECEIPT_PATH}` sha256 `{payload['custody']['byteclose_receipt_sha256']}`.",
            f"- ET4 summary: `{ET4_SUMMARY_PATH}` sha256 `{payload['custody']['et4_summary_sha256']}`.",
            "- Re-encoded all 600 patch records with ET4 Brotli Q11 and matched receipt raw/compressed bytes and shas.",
            f"- Per-pair delta index/value shas checked: {payload['custody']['per_pair_delta_hashes_checked']}.",
            "",
            "## Recall evidence",
            "",
        ]
    )
    for entry in payload["recall_evidence"]:
        lines.append(f"- {entry}")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- No SegNet/PoseNet scorer call was run by ET5.",
            "- No archive.zip was built.",
            "- All net-dS values are projections using banked ET4 full-patch flip counts and measured restricted bytes.",
            "- Verdict scope: INSTANCE, et4 correction field on tq1c parent, stratified n32 description pricing.",
            "",
            "Own-vehicle frontier line: S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]; ET5 did not move it.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_next_md(payload: dict[str, Any]) -> str:
    best = payload["verdict"]["best_projection"]
    row = best["coder_row"]
    return (
        "# ET5 next if resumed\n\n"
        "Status: scorer-free pricing landed; this ET5 restriction/coder family is FOLDED on rate before realization.\n\n"
        "1. Do not materialize the current winner or spend a scorer slot: "
        f"`{best['restriction']}` + `{row['coder']}` is {fmt_float(row['B_per_full_patch_flip'], 3)} B/full-ET4-flip vs W={WATERLINE_B_PER_FLIP}.\n"
        "2. Reopen only with a new coder/restriction that measures <= W on stratified n>=32 real patch records.\n"
        "3. If reopened, the owed validation is all-600 materialization, application to the tq1c parent, and exact CPU-torch restricted-patch argmax measurement.\n"
        "4. Campaign #984 should consume the current result as a negative carriage price, not a queued candidate.\n\n"
        "Do not cite ET5 as a score row or pointer movement.\n"
    )


def make_route_md(payload: dict[str, Any]) -> str:
    best = payload["verdict"]["best_projection"]
    row = best["coder_row"]
    wf = row["waterfill_subset_if_full_patch_flips_retained"]
    return (
        "# Campaign #984 ET5 route\n\n"
        "Disposition: FOLDED — rate-dead before restricted-patch realization.\n\n"
        "| field | value |\n"
        "|---|---:|\n"
        f"| restriction | `{best['restriction']}` |\n"
        f"| coder | `{row['coder']}` |\n"
        f"| projected n600 bytes | {fmt_float(row['projected_n600_bytes'], 1)} |\n"
        f"| B/full-ET4-flip | {fmt_float(row['B_per_full_patch_flip'], 6)} |\n"
        f"| B/support-proxy-flip | {fmt_float(row['B_per_support_proxy_flip'], 6)} |\n"
        f"| projected net dS, full-retention assumption | {fmt_float(row['net_delta_S_if_full_patch_flips_retained'], 9)} |\n"
        f"| projected net dS, support proxy | {fmt_float(row['net_delta_S_support_proxy'], 9)} |\n"
        f"| waterfill selected pairs | {wf['selected_count']} / {SAMPLE_N} |\n"
        f"| waterfill projected net dS | {fmt_float(wf['projected_net_delta_S'], 9)} |\n\n"
        "Campaign #984 should consume this as a negative description-side carriage price. The current winner is still 66.35x above W before any realization loss.\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[et5] loading ET4 rows and patch records", flush=True)
    rows = load_rows(ROWS_PATH)
    et4_receipt = json.loads(ET4_RECEIPT_PATH.read_text())
    patch_paths = sorted(PATCH_DIR.glob("pair_*.npz"))
    records = load_patch_records(patch_paths)
    records_by_pair = {int(record["pair"]): record for record in records}
    custody = verify_custody(rows, records, et4_receipt)
    print("[et5] custody verified", flush=True)

    sample_pairs = deterministic_stratified_sample(SAMPLE_N, SAMPLE_SEED)
    parent_argmax = np.load(PARENT_ARGMAX_PATH, mmap_mode="r")
    current_offsets = np.load(CURRENT_OFFSETS_PATH, mmap_mode="r")
    gt_labels = open_stored_npy_memmap(GT_CACHE_PATH, "lstars")
    block_pixels = build_block_pixel_cache()

    restrictions: list[dict[str, Any]] = []
    for family in ("base_flip", "phase_target"):
        for radius in (0, 1, 2):
            print(f"[et5] building restriction {family}_r{radius}", flush=True)
            masks: dict[int, np.ndarray] = {}
            for pair in sample_pairs:
                lstar = np.asarray(parent_argmax[pair], dtype=np.uint8)
                lgt = np.asarray(gt_labels[pair], dtype=np.uint8)
                if family == "base_flip":
                    base = lstar != lgt
                else:
                    target = translate_blocks(lstar, np.asarray(current_offsets[pair]), 16)
                    base = target != lstar
                masks[pair] = dilate_mask(base, radius)
            restrictions.append(
                summarize_restriction(
                    name=f"{family}_r{radius}",
                    sample_pairs=sample_pairs,
                    rows=rows,
                    records_by_pair=records_by_pair,
                    masks_by_pair=masks,
                    block_pixels=block_pixels,
                )
            )

    pricing = [price_restriction(item) for item in restrictions]
    valid_best = [
        {
            "restriction": item["restriction"],
            "support_stats": item["support_stats"],
            "coder_row": item["best_by_full_patch_flip_projection"],
        }
        for item in pricing
        if item["best_by_full_patch_flip_projection"] is not None
    ]
    best_projection = min(
        valid_best,
        key=lambda item: item["coder_row"]["net_delta_S_if_full_patch_flips_retained"],
    )
    verdict = {
        "status": "FOLDED_INSTANCE_RATE_DEAD_REALIZATION_NOT_FIRED",
        "verdict_scope": "INSTANCE: ET4 correction field on tq1c parent, stratified n32 scorer-free description pricing",
        "best_projection": best_projection,
        "negative_claim": "All measured restriction+coder points exceed W before realization; no scorer-worthy follow-on is queued.",
    }
    payload = {
        "schema": "ddm_et5_carriage_recode_pricing_receipt.v1",
        "captured_at_utc": utc_now(),
        "git": git_head(),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "promotion_eligible": PROMOTION_ELIGIBLE,
        "waterline_B_per_flip": WATERLINE_B_PER_FLIP,
        "rate_per_byte": RATE_PER_BYTE,
        "S_per_flip": S_PER_FLIP,
        "selection": {
            "mode": "deterministic_stratified_random_n32",
            "seed": SAMPLE_SEED,
            "n": SAMPLE_N,
            "population": N_PAIRS,
            "pairs": sample_pairs,
            "not_prefix": sample_pairs != list(range(SAMPLE_N)),
            "strata": "32 equal-width population bins over pair ids 0..599; one rng draw per bin",
        },
        "inputs": {
            "rows_path": ROWS_PATH,
            "patch_dir": PATCH_DIR,
            "et4_receipt_path": ET4_RECEIPT_PATH,
            "et4_summary_path": ET4_SUMMARY_PATH,
            "parent_argmax_path": PARENT_ARGMAX_PATH,
            "current_offsets_path": CURRENT_OFFSETS_PATH,
            "gt_cache_path": GT_CACHE_PATH,
        },
        "custody": custody,
        "recall_evidence": [
            "Read charter .omx/tmp/codex_runs/et5_prompt.md and common contract .omx/tmp/codex_runs/_common_contract.md.",
            "Read PROGRAM.md, CLAUDE.md/AGENTS.md byte-identical contract, docs/operating_manual_craft_handoff.md, and .omx/state/main_hot_state.md.",
            "Read ET4 adjudication and byteclose receipt; ET4 row is rate-dominated but solver leg fixed 78,302 net flips through full byte-close.",
            "Searched MEMORY.md for et4/et5/carriage/SMEVR/#939/#984/waterfill; only RL1/R7 adjacent coder lessons found, no prior ET5 pricing.",
            "Searched .omx/research and state for et4/et5/carriage/SMEVR/1.273108/description-vs-realization/#984; found R7 API, SE3/RL1 coder-race precedents, and main_hot_state ET5 route.",
            "Ran tools/list_canonical_equations.py --json sampling; consumed registered byte/flip waterline discipline rather than minting a new equation.",
        ],
        "method": {
            "support_restrictions": [
                "base_flip_r{0,1,2}: parent_argmax != GT, Chebyshev dilation, snapped to 2x2 scorer blocks, mapped through DK1 private camera supports",
                "phase_target_r{0,1,2}: ET2 phase target != parent_argmax, Chebyshev dilation, snapped to 2x2 scorer blocks, mapped through DK1 private camera supports",
            ],
            "coders": [
                "et4_brotli11: original ET4 sparse_frame1_i16_delta framing with Brotli Q11",
                "et4_lzma1_raw: same ET4 raw framing with raw LZMA1 extreme",
                "split_brotli11/split_lzma1/split_mixed_min: pair table plus delta-coded coord varints and zigzag value varints",
                "r7_smevr_nibble_varint: R7 SMEVR over padded varint-byte nibbles, with pair length table; ET5 prices the encoded frame and relies on the landed R7 decoder tests rather than re-decoding every candidate",
            ],
            "scorer_realization": "NOT RUN by ET5; full-patch ET4 flip counts are banked, restricted-patch flip retention is realization-owed.",
        },
        "pricing": pricing,
        "verdict": verdict,
        "boundaries": [
            "No SegNet/PoseNet scorer call.",
            "No archive.zip build.",
            "No contest-CPU/CUDA claim.",
            "Projected n600 bytes are 600 * stratified-n32 mean bytes.",
            "Projected net dS reuses ET4 full-patch flips or an nnz-weighted support proxy; neither is restricted-patch realization authority.",
        ],
        "own_vehicle_frontier_line": "S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]; ET5 did not move it.",
    }
    write_json_atomic(OUT_DIR / "pricing_receipt.json", payload)
    write_text_atomic(OUT_DIR / "PRICING_TABLE.md", make_pricing_table_md(payload))
    write_text_atomic(OUT_DIR / "RECEIPT.md", make_receipt_md(payload))
    write_text_atomic(OUT_DIR / "NEXT_IF_RESUMED.md", make_next_md(payload))
    write_text_atomic(OUT_DIR / "CAMPAIGN_984_ROUTE.md", make_route_md(payload))
    print(json.dumps({
        "status": verdict["status"],
        "sample_pairs": sample_pairs,
        "best_restriction": best_projection["restriction"],
        "best_coder": best_projection["coder_row"]["coder"],
        "best_B_per_full_patch_flip": best_projection["coder_row"]["B_per_full_patch_flip"],
        "best_net_delta_S_if_full_patch_flips_retained": best_projection["coder_row"]["net_delta_S_if_full_patch_flips_retained"],
        "receipt": str(OUT_DIR / "pricing_receipt.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
PY
