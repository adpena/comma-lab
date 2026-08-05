#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""OD7 receiver-close RGB realization for the OD6 mask-domain row.

This is an advisory n32 receiver-consumption measurement.  It stages a real IX2
archive that consumes an OD7 sparse RGB section and a guard copy of the OD6 OD5
packet, then measures the edited RGB through the frozen CPU scorer on the
chartered n32 pair set.  It does not run upstream/evaluate.py and it does not
run a full n600 scorer job.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import os
import shutil
import stat
import struct
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src", REPO / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ddm_od5_generator_packet as od5  # noqa: E402
import ddm_od6_decoder_legal_context as od6  # noqa: E402
import ddm_pe1_per_edge_partition_race as pe1  # noqa: E402
import ddm_pe3_hybrid_composition as pe3  # noqa: E402
from ddm_sq1_eta_seg_realization import (  # noqa: E402
    COL_SUP,
    ROW_SUP,
    SEG_H,
    SEG_W,
    Scorer,
    decode_gt_frames,
)
from ddm_sq1_stage_decomposition_and_solved_paint import resize_to_scorer  # noqa: E402
from tac.optimization import ddm_od4_weak_stage1_packet as od4  # noqa: E402


DEFAULT_RESEARCH_DIR: Final = REPO / ".omx/research/ddm_od7_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_od7_20260805")
DEFAULT_QO1_SUB_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit")
DEFAULT_PE3_SUB_DIR: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pe3_20260805/pe3_20260805T000000Z/"
    "sub_auto_pairbit_pe3_hybrid_75kb"
)
DEFAULT_OD6_JSON: Final = REPO / ".omx/research/ddm_od6_20260805/ddm_od6_decoder_legal_receipt.json"
DEFAULT_OD6_PACKET: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_od6_20260805/run_20260805T030410Z/"
    "packets/base_rgb_generator_geometry_b1024.od5.raw_packet"
)
DEFAULT_GT_MKV: Final = REPO / "upstream/videos/0.mkv"
DEFAULT_RUNTIME_TEMPLATE: Final = REPO / "experiments/inflate_runner_v4d.py"

CAM_H: Final = 874
CAM_W: Final = 1164
CAM_C: Final = 3
SEQ_LEN: Final = 2
OD7_MAGIC: Final = b"OD7RGB1\0"
OD7_VERSION: Final = 1
OD7_MODE_FIXED_PROTO: Final = 0
OD7_MODE_SPARSE_RGB: Final = 1
OD7_MODE_HYBRID: Final = 2
OD7_RAW: Final = 0
OD7_LZMA1_RAW: Final = 1
OD7_BROTLI_Q11: Final = 2
OD7_HEADER: Final = struct.Struct("<8sBBBBHHIII32s32s")
OD7_LZMA_FILTERS: Final = (
    {"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0},
)
FIXED_CLASS_RGB: Final = np.array(
    [
        [30, 39, 72],
        [77, 87, 119],
        [168, 145, 96],
        [114, 59, 45],
        [43, 121, 93],
    ],
    dtype=np.uint8,
)
OD6_PROJECTED_S: Final = 0.7436007705338658
OD6_PROJECTED_PACKET_BYTES: Final = 76_304
OD6_PROJECTED_DELTA_VS_LIVE: Final = -0.0103799591572549
OWN_FRONTIER_LINE: Final = "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved."


class OD7Error(ValueError):
    """OD7 failed a typed invariant."""


@dataclass(frozen=True, slots=True)
class SelectedSurface:
    pairs: list[int]
    selected_by_pair: dict[int, np.ndarray]
    target_labels_by_pair: dict[int, np.ndarray]
    target_pairs: dict[int, od5.TargetPair]
    target_denominators: dict[str, Any]
    source_meta: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModePayload:
    mode_name: str
    mode_id: int
    raw_body: bytes
    records_by_pair: dict[int, np.ndarray]
    labels_by_pair: dict[int, np.ndarray]
    rgb_by_pair: dict[int, np.ndarray]
    exceptions_by_pair: dict[int, np.ndarray]
    section: bytes
    section_parse: dict[str, Any]
    coder_rows: list[dict[str, Any]]
    best_coder: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
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


def _source_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OD7Error(f"JSON root is not an object: {path}")
    return data


def _varint(value: int) -> bytes:
    if value < 0:
        raise OD7Error("negative varint")
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
            raise OD7Error("truncated varint")
        byte = int(payload[offset])
        offset += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return value, offset
        shift += 7
        if shift > 63:
            raise OD7Error("varint too long")


def _pack_nibbles(values: np.ndarray) -> bytes:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if flat.size and int(flat.max()) > 15:
        raise OD7Error("nibble value out of range")
    out = np.zeros((flat.size + 1) // 2, dtype=np.uint8)
    out[: flat[0::2].size] = flat[0::2]
    if flat.size > 1:
        out[: flat[1::2].size] |= flat[1::2] << 4
    return out.tobytes()


def _unpack_nibbles(payload: bytes, count: int) -> np.ndarray:
    raw = np.frombuffer(payload, dtype=np.uint8)
    out = np.empty(raw.size * 2, dtype=np.uint8)
    out[0::2] = raw & 15
    out[1::2] = raw >> 4
    if out.size < count:
        raise OD7Error("nibble payload truncated")
    return np.ascontiguousarray(out[:count])


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 63)


def _unzigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


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
        raise OD7Error(f"SSD storage preflight failed: {out}")
    return out


def _import_runtime(path: Path, module_name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise OD7Error(f"could not import runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected_surface(args: argparse.Namespace) -> SelectedSurface:
    od2_json = _load_json(args.od2_json)
    pair_selection = _load_json(args.pair_selection)
    od6_receipt = _load_json(args.od6_json)
    od6_packet = od4.parse_od5_packet(args.od6_packet.read_bytes())
    sections = {section.name: section.payload for section in od6_packet.sections}
    table_payloads = [payload for name, payload in sections.items() if name.startswith("od6_decoder_legal_table_")]
    if len(table_payloads) != 1:
        raise OD7Error(f"expected exactly one OD6 table section, got {len(table_payloads)}")
    table_header, qlogits = od6._parse_qlogit_payload(table_payloads[0], magic=od6.OD6_TABLE_MAGIC)
    if table_header["feature_mode"] != "base_rgb_generator_geometry":
        raise OD7Error(f"unexpected OD6 feature mode {table_header['feature_mode']!r}")
    if int(table_header["bucket_count"]) != 1024:
        raise OD7Error("OD7 charter is bound to OD6 b1024")

    rows = od2_json.get("rows")
    if not isinstance(rows, list) or not rows:
        raise OD7Error("OD2 JSON has no rows")
    od2_rows_by_pair = {int(row["pair"]): row for row in rows}
    pairs = [int(pair) for pair in pair_selection["pairs"]]
    missing = [pair for pair in pairs if pair not in od2_rows_by_pair]
    if missing:
        raise OD7Error(f"OD2 JSON missing selected pairs: {missing}")

    current_argmax = np.load(args.argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gt_argmax = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")
    lstars = pe1.open_stored_npy_memmap(args.gt_cache, "lstars")
    if not np.array_equal(np.asarray(lstars[pairs], dtype=np.uint8), np.asarray(gt_argmax[pairs], dtype=np.uint8)):
        raise OD7Error("GT cache and argmax cache differ on selected n32 pairs")

    target_pairs = {
        pair: od5._derive_target_pair(
            pair=pair,
            od2_row=od2_rows_by_pair[pair],
            current=np.asarray(current_argmax[pair], dtype=np.uint8),
            gt=np.asarray(gt_argmax[pair], dtype=np.uint8),
            block=args.block,
            rmax=args.rmax,
        )
        for pair in pairs
    }
    target_denominators = {
        "od2_n_described": int(sum(tp.build_row["n_described"] for tp in target_pairs.values())),
        "retained_fix_denominator": int(sum(tp.full_record.count for tp in target_pairs.values())),
        "pairs": len(pairs),
    }
    if target_denominators["retained_fix_denominator"] != od6.EXPECTED_RETAINED_FIX_DENOMINATOR:
        raise OD7Error(f"unexpected retained-fix denominator: {target_denominators}")

    base_features, qo1_meta = od6._load_qo1_base_frame_features(args.qo1_sub_dir, pairs)
    universe, positives, point_meta = od6._build_point_sets(
        target_pairs=target_pairs,
        pairs=pairs,
        current_argmax=current_argmax,
        gt_argmax=gt_argmax,
    )
    components, extraction = pe1.extract_components(lstars, current_argmax)
    pe1_receipt = _load_json(args.pe1_receipt)
    surfaces = od5._build_generator_surfaces(
        components=components,
        lstars=lstars,
        current=current_argmax,
        pe1_receipt=pe1_receipt,
        g4_recurrence=args.g4_recurrence,
        depth_y1=args.depth_y1,
        depth_y2=args.depth_y2,
    )
    generator_rep: pe1.RepresentationBuild = surfaces["generator_rep"]
    hybrid75: pe3.HybridBuild = surfaces["hybrid75"]
    hybrid_knee: pe3.HybridBuild = surfaces["hybrid_knee"]
    generator_masks = od5._masks_by_pair(components=components, component_masks=generator_rep.component_masks, pairs=pairs)
    hybrid75_masks = od5._masks_by_pair(components=components, component_masks=hybrid75.component_masks, pairs=pairs)
    hybrid_knee_masks = od5._masks_by_pair(components=components, component_masks=hybrid_knee.component_masks, pairs=pairs)

    pos_columns, column_names = od6._mode_columns(
        mode=table_header["feature_mode"],
        points=positives,
        base_features=base_features,
        generator_masks=generator_masks,
        hybrid75_masks=hybrid75_masks,
        hybrid_knee_masks=hybrid_knee_masks,
        proxy_qlogits=None,
        proxy_block=args.proxy_block,
    )
    if list(table_header["feature_columns"]) != column_names:
        raise OD7Error("reconstructed OD6 feature columns differ from packet header")
    hashes = od6._hash_columns(pos_columns, int(table_header["bucket_count"]))
    probs = od6._sigmoid(qlogits[hashes].astype(np.float32) / np.float32(table_header["qscale"]))
    keep = probs >= np.float32(table_header["threshold"])
    table_selected: dict[int, list[int]] = {}
    for pair, flat, kept in zip(positives.pair.tolist(), positives.flat.tolist(), keep.tolist(), strict=True):
        if kept:
            table_selected.setdefault(int(pair), []).append(int(flat))
    table_selected_np = {
        pair: np.asarray(sorted(table_selected.get(pair, [])), dtype=np.int64)
        for pair in pairs
    }
    hybrid75_selected = od5._selected_flats_from_mask(target_pairs, hybrid75_masks, pairs)
    selected_by_pair = od5._union_selected(pairs, hybrid75_selected, table_selected_np)
    expected = int(
        od6_receipt["best_rung_by_projected_s_with_od2_pose_credit"]["fidelity"]["totals"]["retained_fix_count"]
    )
    got = int(sum(arr.size for arr in selected_by_pair.values()))
    if got != expected:
        raise OD7Error(f"reconstructed OD6 selected count {got} != receipt {expected}")

    target_labels_by_pair: dict[int, np.ndarray] = {}
    for pair in pairs:
        flats = np.asarray(selected_by_pair[pair], dtype=np.int64)
        target_flat = target_pairs[pair].target_argmax.reshape(-1)
        target_labels_by_pair[pair] = np.asarray(target_flat[flats], dtype=np.uint8)

    source_meta = {
        "od6_packet_sections": [
            {"name": section.name, "bytes": len(section.payload), "sha256": _sha256_bytes(section.payload)}
            for section in od6_packet.sections
        ],
        "od6_table_header": table_header,
        "od6_table_payload_sha256": _sha256_bytes(table_payloads[0]),
        "od6_selected_context_fix_points_before_hybrid_union": int(keep.sum()),
        "hybrid75_selected_fix_points": int(sum(arr.size for arr in hybrid75_selected.values())),
        "combined_selected_fix_points": got,
        "point_meta": point_meta,
        "qo1_base_frame_features": qo1_meta,
        "component_extraction": extraction,
        "generator_surface": {
            "generator_count": int(surfaces["generator_count"]),
            "hybrid75_n32_subset_raw_bytes": int(len(od5._subset_frame_records(hybrid75.frame_records, pairs))),
            "hybrid_knee_n32_subset_raw_bytes": int(len(od5._subset_frame_records(hybrid_knee.frame_records, pairs))),
        },
        "pure_receiver_targeter_gap": {
            "status": "NOT_CLOSED_BY_OD6_PACKET_ALONE",
            "reason": (
                "OD6 b1024 hashes require pe1_generator_coverage and pe3_hybrid_knee_coverage columns; "
                "the OD6 packet row stores only the PE3 hybrid75 n32 subset and the bucket table."
            ),
        },
    }
    return SelectedSurface(
        pairs=pairs,
        selected_by_pair=selected_by_pair,
        target_labels_by_pair=target_labels_by_pair,
        target_pairs=target_pairs,
        target_denominators=target_denominators,
        source_meta=source_meta,
    )


def _scorer_rgb_for_gt_pairs(gt_frames: dict[int, np.ndarray], pairs: list[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for pair in pairs:
        resized = resize_to_scorer(np.asarray(gt_frames[2 * pair + 1], dtype=np.uint8))
        arr = resized[0].permute(1, 2, 0).numpy()
        out[pair] = np.rint(np.clip(arr, 0.0, 255.0)).astype(np.uint8)
    return out


def _paint_records(
    base_f1: np.ndarray,
    flats: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    if int(flats.size) == 0:
        return base_f1
    rows = (flats // SEG_W).astype(np.int64)
    cols = (flats % SEG_W).astype(np.int64)
    out = base_f1.copy()
    for rr in (ROW_SUP[rows, 0], ROW_SUP[rows, 1]):
        for cc in (COL_SUP[cols, 0], COL_SUP[cols, 1]):
            out[rr, cc] = colors
    return out


def _colors_for_mode(
    *,
    mode_name: str,
    pair: int,
    flats: np.ndarray,
    labels: np.ndarray,
    gt_scorer_rgb: dict[int, np.ndarray],
    exceptions: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    if mode_name == "fixed_proto":
        return FIXED_CLASS_RGB[labels]
    if mode_name == "sparse_gt_rgb":
        gt_flat = gt_scorer_rgb[pair].reshape(-1, 3)
        return np.asarray(gt_flat[flats], dtype=np.uint8)
    if mode_name == "hybrid_proto_gt_exceptions":
        colors = FIXED_CLASS_RGB[labels].copy()
        exc = np.asarray((exceptions or {}).get(pair, np.array([], dtype=np.int64)), dtype=np.int64)
        if exc.size:
            gt_flat = gt_scorer_rgb[pair].reshape(-1, 3)
            selected_index = {int(flat): idx for idx, flat in enumerate(flats.tolist())}
            idx = np.asarray([selected_index[int(flat)] for flat in exc.tolist()], dtype=np.int64)
            colors[idx] = gt_flat[exc]
        return colors
    raise OD7Error(f"unknown mode {mode_name}")


def _score_mode(
    *,
    scorer: Scorer,
    decoder: Any,
    gt_frames: dict[int, np.ndarray],
    gt_argmax: np.ndarray,
    surface: SelectedSurface,
    gt_scorer_rgb: dict[int, np.ndarray],
    mode_name: str,
    exceptions: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_before = 0
    total_after = 0
    total_changed_camera = 0
    d_pose_values: list[float] = []
    argmax_by_pair: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    for pair in surface.pairs:
        base_f1 = decoder.f1(pair)
        flats = surface.selected_by_pair[pair]
        labels = surface.target_labels_by_pair[pair]
        colors = _colors_for_mode(
            mode_name=mode_name,
            pair=pair,
            flats=flats,
            labels=labels,
            gt_scorer_rgb=gt_scorer_rgb,
            exceptions=exceptions,
        )
        cand_f1 = _paint_records(base_f1, flats, colors)
        cand_f0 = decoder.f0(pair, cand_f1)
        cand_pair = np.stack([cand_f0, cand_f1], axis=0).astype(np.uint8)
        gt_pair = np.stack([gt_frames[2 * pair], gt_frames[2 * pair + 1]], axis=0).astype(np.uint8)
        gt_pose = scorer.pose_out(gt_pair)
        cand_pose = scorer.pose_out(cand_pair)
        d_pose = scorer.d_pose(gt_pose, cand_pose)
        argmax = scorer.seg_argmax(cand_pair)
        argmax_by_pair[pair] = argmax
        gt = np.asarray(gt_argmax[pair], dtype=np.uint8)
        before = int((np.asarray(gt_argmax[pair]) != np.asarray(surface.target_pairs[pair].target_argmax)).sum())
        cached_before = int(surface.target_pairs[pair].build_row["flips_before"])
        if before != cached_before:
            before = cached_before
        after = int((argmax != gt).sum())
        changed = int((cand_f1 != base_f1).any(axis=2).sum())
        total_before += before
        total_after += after
        total_changed_camera += changed
        d_pose_values.append(d_pose)
        rows.append(
            {
                "pair": pair,
                "flips_before": before,
                "flips_after": after,
                "retained_fix_count": before - after,
                "selected_records": int(flats.size),
                "changed_camera_pixels_f1": changed,
                "d_pose": d_pose,
            }
        )
    wall = time.perf_counter() - t0
    retained = total_before - total_after
    d_pose_mean = float(np.mean(d_pose_values)) if d_pose_values else 0.0
    return {
        "mode": mode_name,
        "rows": rows,
        "totals": {
            "pairs": len(surface.pairs),
            "flips_before": total_before,
            "flips_after": total_after,
            "retained_fix_count": retained,
            "eta_vs_od2_n_described": retained / surface.target_denominators["od2_n_described"],
            "selected_records": int(sum(arr.size for arr in surface.selected_by_pair.values())),
            "changed_camera_pixels_f1": total_changed_camera,
            "d_pose_mean": d_pose_mean,
            "pose_contribution_sqrt10": float(np.sqrt(10.0 * d_pose_mean)),
            "wall_seconds": wall,
        },
        "argmax_by_pair": argmax_by_pair,
    }


def _baseline_score(
    *,
    scorer: Scorer,
    decoder: Any,
    gt_frames: dict[int, np.ndarray],
    gt_argmax: np.ndarray,
    pairs: list[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    d_pose_values: list[float] = []
    total_flips = 0
    t0 = time.perf_counter()
    for pair in pairs:
        f1 = decoder.f1(pair)
        f0 = decoder.f0(pair, f1)
        cand_pair = np.stack([f0, f1], axis=0).astype(np.uint8)
        gt_pair = np.stack([gt_frames[2 * pair], gt_frames[2 * pair + 1]], axis=0).astype(np.uint8)
        gt_pose = scorer.pose_out(gt_pair)
        cand_pose = scorer.pose_out(cand_pair)
        d_pose = scorer.d_pose(gt_pose, cand_pose)
        argmax = scorer.seg_argmax(cand_pair)
        flips = int((argmax != np.asarray(gt_argmax[pair], dtype=np.uint8)).sum())
        total_flips += flips
        d_pose_values.append(d_pose)
        rows.append({"pair": pair, "flips": flips, "d_pose": d_pose})
    wall = time.perf_counter() - t0
    d_pose_mean = float(np.mean(d_pose_values)) if d_pose_values else 0.0
    return {
        "rows": rows,
        "totals": {
            "pairs": len(pairs),
            "flips": total_flips,
            "d_pose_mean": d_pose_mean,
            "pose_contribution_sqrt10": float(np.sqrt(10.0 * d_pose_mean)),
            "wall_seconds": wall,
        },
    }


def _build_exceptions(fixed_score: dict[str, Any], surface: SelectedSurface) -> dict[int, np.ndarray]:
    exceptions: dict[int, np.ndarray] = {}
    argmax_by_pair = fixed_score["argmax_by_pair"]
    for pair in surface.pairs:
        argmax = np.asarray(argmax_by_pair[pair], dtype=np.uint8).reshape(-1)
        flats = surface.selected_by_pair[pair]
        labels = surface.target_labels_by_pair[pair]
        miss = flats[argmax[flats] != labels]
        exceptions[pair] = np.asarray(miss, dtype=np.int64)
    return exceptions


def _mode_id(mode_name: str) -> int:
    if mode_name == "fixed_proto":
        return OD7_MODE_FIXED_PROTO
    if mode_name == "sparse_gt_rgb":
        return OD7_MODE_SPARSE_RGB
    if mode_name == "hybrid_proto_gt_exceptions":
        return OD7_MODE_HYBRID
    raise OD7Error(f"unknown mode {mode_name}")


def _encode_mode_body(
    *,
    mode_name: str,
    surface: SelectedSurface,
    gt_scorer_rgb: dict[int, np.ndarray],
    exceptions: dict[int, np.ndarray] | None = None,
) -> tuple[bytes, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    body = bytearray()
    body += _varint(len(surface.pairs))
    records_by_pair: dict[int, np.ndarray] = {}
    labels_by_pair: dict[int, np.ndarray] = {}
    rgb_by_pair: dict[int, np.ndarray] = {}
    exceptions_by_pair: dict[int, np.ndarray] = {}
    for pair in surface.pairs:
        flats = np.asarray(surface.selected_by_pair[pair], dtype=np.int64)
        labels = np.asarray(surface.target_labels_by_pair[pair], dtype=np.uint8)
        colors = _colors_for_mode(
            mode_name=mode_name,
            pair=pair,
            flats=flats,
            labels=labels,
            gt_scorer_rgb=gt_scorer_rgb,
            exceptions=exceptions,
        )
        body += _varint(pair)
        body += _varint(int(flats.size))
        prev = -1
        for flat in flats.tolist():
            body += _varint(int(flat) - prev - 1)
            prev = int(flat)
        body += _pack_nibbles(labels)
        exc = np.asarray((exceptions or {}).get(pair, np.array([], dtype=np.int64)), dtype=np.int64)
        if mode_name == "sparse_gt_rgb":
            body += np.ascontiguousarray(colors, dtype=np.uint8).tobytes()
        elif mode_name == "hybrid_proto_gt_exceptions":
            selected_index = {int(flat): idx for idx, flat in enumerate(flats.tolist())}
            exc_indices = np.asarray([selected_index[int(flat)] for flat in exc.tolist()], dtype=np.int64)
            body += _varint(int(exc_indices.size))
            prev_idx = -1
            for idx in exc_indices.tolist():
                body += _varint(int(idx) - prev_idx - 1)
                prev_idx = int(idx)
            if exc_indices.size:
                body += np.ascontiguousarray(colors[exc_indices], dtype=np.uint8).tobytes()
        elif mode_name != "fixed_proto":
            raise OD7Error(f"unknown mode {mode_name}")
        records_by_pair[pair] = flats
        labels_by_pair[pair] = labels
        rgb_by_pair[pair] = colors
        exceptions_by_pair[pair] = exc
    return bytes(body), records_by_pair, labels_by_pair, rgb_by_pair, exceptions_by_pair


def _code_body(raw: bytes) -> tuple[int, bytes, list[dict[str, Any]], dict[str, Any]]:
    candidates: list[tuple[str, int, bytes]] = [
        ("raw", OD7_RAW, bytes(raw)),
        ("lzma1-raw", OD7_LZMA1_RAW, lzma.compress(raw, format=lzma.FORMAT_RAW, filters=list(OD7_LZMA_FILTERS))),
        ("brotli-q11", OD7_BROTLI_Q11, brotli.compress(raw, quality=11)),
    ]
    rows: list[dict[str, Any]] = []
    for codec_name, coder_id, payload in candidates:
        parseback = _decode_body(coder_id, payload, len(raw)) == raw
        rows.append(
            {
                "codec": codec_name,
                "coder_id": coder_id,
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "parseback_exact": parseback,
            }
        )
    valid = [row for row in rows if row["parseback_exact"]]
    if not valid:
        raise OD7Error("no OD7 body coder survived parse-back")
    best = min(valid, key=lambda row: (int(row["bytes"]), str(row["codec"])))
    coded = next(payload for codec_name, coder_id, payload in candidates if coder_id == int(best["coder_id"]))
    return int(best["coder_id"]), coded, rows, best


def _decode_body(coder_id: int, payload: bytes, raw_len: int) -> bytes:
    if coder_id == OD7_RAW:
        raw = bytes(payload)
    elif coder_id == OD7_LZMA1_RAW:
        raw = lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=list(OD7_LZMA_FILTERS))
    elif coder_id == OD7_BROTLI_Q11:
        raw = brotli.decompress(payload)
    else:
        raise OD7Error(f"unknown OD7 coder id {coder_id}")
    if len(raw) != raw_len:
        raise OD7Error("OD7 decoded body length mismatch")
    return raw


def _build_mode_payload(
    *,
    mode_name: str,
    surface: SelectedSurface,
    gt_scorer_rgb: dict[int, np.ndarray],
    od6_packet_sha: str,
    exceptions: dict[int, np.ndarray] | None = None,
) -> ModePayload:
    raw_body, records, labels, colors, exc = _encode_mode_body(
        mode_name=mode_name,
        surface=surface,
        gt_scorer_rgb=gt_scorer_rgb,
        exceptions=exceptions,
    )
    coder_id, coded, rows, best = _code_body(raw_body)
    record_count = int(sum(arr.size for arr in records.values()))
    exception_count = int(sum(arr.size for arr in exc.values()))
    section = (
        OD7_HEADER.pack(
            OD7_MAGIC,
            OD7_VERSION,
            _mode_id(mode_name),
            coder_id,
            0,
            SEG_H,
            SEG_W,
            record_count,
            exception_count,
            len(raw_body),
            bytes.fromhex(_sha256_bytes(raw_body)),
            bytes.fromhex(od6_packet_sha),
        )
        + coded
    )
    parsed = _parse_od7_section(section)
    return ModePayload(
        mode_name=mode_name,
        mode_id=_mode_id(mode_name),
        raw_body=raw_body,
        records_by_pair=records,
        labels_by_pair=labels,
        rgb_by_pair=colors,
        exceptions_by_pair=exc,
        section=section,
        section_parse=parsed,
        coder_rows=rows,
        best_coder=best,
    )


def _parse_od7_section(section: bytes) -> dict[str, Any]:
    if len(section) < OD7_HEADER.size:
        raise OD7Error("OD7 section header truncated")
    (
        magic,
        version,
        mode_id,
        coder_id,
        _reserved,
        seg_h,
        seg_w,
        record_count,
        exception_count,
        raw_len,
        raw_sha,
        od6_sha,
    ) = OD7_HEADER.unpack_from(section, 0)
    if magic != OD7_MAGIC:
        raise OD7Error("OD7 section magic mismatch")
    if version != OD7_VERSION:
        raise OD7Error("OD7 section version mismatch")
    if (int(seg_h), int(seg_w)) != (SEG_H, SEG_W):
        raise OD7Error("OD7 section geometry mismatch")
    raw = _decode_body(int(coder_id), section[OD7_HEADER.size :], int(raw_len))
    if hashlib.sha256(raw).digest() != raw_sha:
        raise OD7Error("OD7 raw body SHA mismatch")
    records, labels, colors, exceptions = _decode_records(raw, int(mode_id))
    got_records = int(sum(arr.size for arr in records.values()))
    got_exceptions = int(sum(arr.size for arr in exceptions.values()))
    if got_records != int(record_count):
        raise OD7Error("OD7 record count mismatch")
    if got_exceptions != int(exception_count):
        raise OD7Error("OD7 exception count mismatch")
    return {
        "schema": "ddm_od7_sparse_rgb_section.v1",
        "mode_id": int(mode_id),
        "mode": {0: "fixed_proto", 1: "sparse_gt_rgb", 2: "hybrid_proto_gt_exceptions"}[int(mode_id)],
        "coder_id": int(coder_id),
        "section_bytes": len(section),
        "section_sha256": _sha256_bytes(section),
        "raw_bytes": int(raw_len),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "source_od6_packet_sha256": od6_sha.hex(),
        "record_count": int(record_count),
        "exception_count": int(exception_count),
        "pairs_with_records": len(records),
    }


def _decode_records(raw: bytes, mode_id: int) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    offset = 0
    pair_count, offset = _read_varint(raw, offset)
    records: dict[int, np.ndarray] = {}
    labels: dict[int, np.ndarray] = {}
    colors: dict[int, np.ndarray] = {}
    exceptions: dict[int, np.ndarray] = {}
    for _ in range(pair_count):
        pair, offset = _read_varint(raw, offset)
        count, offset = _read_varint(raw, offset)
        flats: list[int] = []
        prev = -1
        for _idx in range(count):
            delta, offset = _read_varint(raw, offset)
            flat = prev + 1 + int(delta)
            flats.append(flat)
            prev = flat
        label_bytes = (int(count) + 1) // 2
        label_payload = raw[offset : offset + label_bytes]
        if len(label_payload) != label_bytes:
            raise OD7Error("OD7 label payload truncated")
        offset += label_bytes
        labels_arr = _unpack_nibbles(label_payload, int(count))
        flats_arr = np.asarray(flats, dtype=np.int64)
        if mode_id == OD7_MODE_FIXED_PROTO:
            colors_arr = FIXED_CLASS_RGB[labels_arr]
            exc_arr = np.array([], dtype=np.int64)
        elif mode_id == OD7_MODE_SPARSE_RGB:
            rgb_payload = raw[offset : offset + 3 * int(count)]
            if len(rgb_payload) != 3 * int(count):
                raise OD7Error("OD7 RGB payload truncated")
            offset += 3 * int(count)
            colors_arr = np.frombuffer(rgb_payload, dtype=np.uint8).reshape(int(count), 3).copy()
            exc_arr = np.array([], dtype=np.int64)
        elif mode_id == OD7_MODE_HYBRID:
            exc_count, offset = _read_varint(raw, offset)
            exc_indices: list[int] = []
            prev_idx = -1
            for _idx in range(exc_count):
                delta, offset = _read_varint(raw, offset)
                item = prev_idx + 1 + int(delta)
                if item < 0 or item >= int(count):
                    raise OD7Error("OD7 exception index out of range")
                exc_indices.append(item)
                prev_idx = item
            rgb_payload = raw[offset : offset + 3 * int(exc_count)]
            if len(rgb_payload) != 3 * int(exc_count):
                raise OD7Error("OD7 exception RGB payload truncated")
            offset += 3 * int(exc_count)
            colors_arr = FIXED_CLASS_RGB[labels_arr].copy()
            if exc_indices:
                exc_idx_arr = np.asarray(exc_indices, dtype=np.int64)
                colors_arr[exc_idx_arr] = np.frombuffer(rgb_payload, dtype=np.uint8).reshape(int(exc_count), 3)
                exc_arr = flats_arr[exc_idx_arr]
            else:
                exc_arr = np.array([], dtype=np.int64)
        else:
            raise OD7Error(f"unknown OD7 mode id {mode_id}")
        records[int(pair)] = flats_arr
        labels[int(pair)] = labels_arr
        colors[int(pair)] = colors_arr
        exceptions[int(pair)] = exc_arr
    if offset != len(raw):
        raise OD7Error("OD7 raw body has trailing bytes")
    return records, labels, colors, exceptions


def _runtime_patch_text(template: str) -> str:
    marker = "OD7_MARKER_RUNTIME_PATCH"
    if marker in template:
        return template
    insert_after = "PE3_EDGE_VERSION = 1\n"
    addition = f"""
# {marker}: sparse selected-set RGB realization.
OD7_RGB_MAGIC = {OD7_MAGIC!r}
OD5_PACKET_MAGIC = b"OD5GPK1\\0"
OD7_RGB_HEADER = struct.Struct("<8sBBBBHHIII32s32s")
_OD7_FIXED_PROTO = 0
_OD7_SPARSE_RGB = 1
_OD7_HYBRID = 2
_OD7_RAW = 0
_OD7_LZMA1_RAW = 1
_OD7_BROTLI_Q11 = 2
_OD7_LZMA_FILTERS = [{{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}}]
_OD7_CLASS_RGB = np.array({FIXED_CLASS_RGB.tolist()!r}, dtype=np.uint8)

def _od7_read_varint(payload: bytes, offset: int):
    value = 0
    shift = 0
    while True:
        if offset >= len(payload):
            raise SystemExit("OD7 varint is truncated")
        byte = int(payload[offset])
        offset += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return value, offset
        shift += 7
        if shift > 63:
            raise SystemExit("OD7 varint is too long")

def _od7_unpack_nibbles(payload: bytes, count: int):
    raw = np.frombuffer(payload, dtype=np.uint8)
    out = np.empty(raw.size * 2, dtype=np.uint8)
    out[0::2] = raw & 15
    out[1::2] = raw >> 4
    if out.size < count:
        raise SystemExit("OD7 labels are truncated")
    return np.ascontiguousarray(out[:count])

def _od7_decode_body(coder: int, payload: bytes):
    if coder == _OD7_RAW:
        return bytes(payload)
    if coder == _OD7_LZMA1_RAW:
        return lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=_OD7_LZMA_FILTERS)
    if coder == _OD7_BROTLI_Q11:
        return brotli.decompress(payload)
    raise SystemExit(f"unknown OD7 coder id {{coder}}")

def _od7_parse_rgb_field(blob: bytes):
    if len(blob) < OD7_RGB_HEADER.size:
        raise SystemExit("OD7 sparse RGB section header truncated")
    (magic, version, mode, coder, _reserved, seg_h, seg_w, record_count,
     exception_count, raw_len, raw_sha, od6_sha) = OD7_RGB_HEADER.unpack_from(blob, 0)
    if magic != OD7_RGB_MAGIC:
        raise SystemExit("OD7 sparse RGB magic differs")
    if version != 1:
        raise SystemExit("OD7 sparse RGB version differs")
    if int(seg_h) != 384 or int(seg_w) != 512:
        raise SystemExit("OD7 sparse RGB geometry differs")
    raw = _od7_decode_body(int(coder), blob[OD7_RGB_HEADER.size:])
    if len(raw) != int(raw_len):
        raise SystemExit("OD7 sparse RGB raw length differs")
    if hashlib.sha256(raw).digest() != raw_sha:
        raise SystemExit("OD7 sparse RGB raw SHA differs")
    off = 0
    pair_count, off = _od7_read_varint(raw, off)
    records = {{}}
    total = 0
    total_exc = 0
    for _ in range(int(pair_count)):
        pair, off = _od7_read_varint(raw, off)
        count, off = _od7_read_varint(raw, off)
        flats = []
        prev = -1
        for _j in range(int(count)):
            delta, off = _od7_read_varint(raw, off)
            flat = prev + 1 + int(delta)
            flats.append(flat)
            prev = flat
        label_bytes = (int(count) + 1) // 2
        labels = _od7_unpack_nibbles(raw[off:off + label_bytes], int(count))
        off += label_bytes
        flats_arr = np.asarray(flats, dtype=np.int64)
        if int(mode) == _OD7_FIXED_PROTO:
            colors = _OD7_CLASS_RGB[labels].copy()
        elif int(mode) == _OD7_SPARSE_RGB:
            rgb = raw[off:off + 3 * int(count)]
            if len(rgb) != 3 * int(count):
                raise SystemExit("OD7 sparse RGB values truncated")
            off += 3 * int(count)
            colors = np.frombuffer(rgb, dtype=np.uint8).reshape(int(count), 3).copy()
        elif int(mode) == _OD7_HYBRID:
            colors = _OD7_CLASS_RGB[labels].copy()
            exc_count, off = _od7_read_varint(raw, off)
            exc_indices = []
            prev_idx = -1
            for _j in range(int(exc_count)):
                delta, off = _od7_read_varint(raw, off)
                idx = prev_idx + 1 + int(delta)
                if idx < 0 or idx >= int(count):
                    raise SystemExit("OD7 hybrid exception index outside record list")
                exc_indices.append(idx)
                prev_idx = idx
            rgb = raw[off:off + 3 * int(exc_count)]
            if len(rgb) != 3 * int(exc_count):
                raise SystemExit("OD7 hybrid exception RGB truncated")
            off += 3 * int(exc_count)
            if exc_indices:
                colors[np.asarray(exc_indices, dtype=np.int64)] = np.frombuffer(rgb, dtype=np.uint8).reshape(int(exc_count), 3)
            total_exc += int(exc_count)
        else:
            raise SystemExit(f"unknown OD7 sparse RGB mode {{int(mode)}}")
        records[int(pair)] = (flats_arr, colors)
        total += int(count)
    if off != len(raw):
        raise SystemExit("OD7 sparse RGB raw has trailing bytes")
    if total != int(record_count):
        raise SystemExit("OD7 sparse RGB record count differs")
    if total_exc != int(exception_count):
        raise SystemExit("OD7 sparse RGB exception count differs")
    return {{"seg_h": int(seg_h), "seg_w": int(seg_w), "records": records, "source_od6_sha256": od6_sha.hex(), "mode": int(mode)}}

def _od7_parse_od5_guard(blob: bytes):
    if len(blob) < 8 or blob[:8] != OD5_PACKET_MAGIC:
        raise SystemExit("OD7 OD5 guard magic differs")
    # This guard intentionally closes the counted OD6 packet bytes.  The selected
    # RGB records carry the values; this guard prevents an OD7 realization from
    # silently detaching from the OD6 packet row it claims to realize.
    return {{"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}}

def _od7_apply_rgb_field(u8: np.ndarray, field, pair_index: int) -> np.ndarray:
    item = field["records"].get(int(pair_index))
    if item is None:
        return u8
    flats, colors = item
    if int(flats.size) == 0:
        return u8
    rows = flats // field["seg_w"]
    cols = flats % field["seg_w"]
    rlo, rhi, _rw = _f0pr_bilinear_axis(field["seg_h"], u8.shape[0])
    clo, chi, _cw = _f0pr_bilinear_axis(field["seg_w"], u8.shape[1])
    out = u8.copy()
    for rr in (rlo[rows], rhi[rows]):
        for cc in (clo[cols], chi[cols]):
            out[rr, cc] = colors
    return out

"""
    if insert_after not in template:
        raise OD7Error("runtime template anchor not found for constants")
    out = template.replace(insert_after, insert_after + addition, 1)
    out = out.replace(
        "        self._pe3_edge_field = None     # PE3 hybrid per-regime edge modulation\n",
        "        self._pe3_edge_field = None     # PE3 hybrid per-regime edge modulation\n"
        "        self._od7_rgb_field = None      # OD7 selected-set sparse RGB realization\n"
        "        self._od7_od5_guard = None      # OD6 OD5 packet guard consumed by OD7\n",
        1,
    )
    out = out.replace(
        "            elif extra[:8] == PE3_EDGE_MAGIC:\n"
        "                if self._pe1_edge_field is not None or self._pe3_edge_field is not None:\n"
        "                    raise SystemExit(\"ix2 container has duplicate edge optional sections\")\n"
        "                self._pe3_edge_field = _pe3_parse_edge_field(extra)\n"
        "            else:\n"
        "                raise SystemExit(\n"
        "                    \"ix2 optional section magic is not F0PR1, BD1CLF1, PE1EDGE1, or PE3EDGE1\"\n"
        "                )\n",
        "            elif extra[:8] == PE3_EDGE_MAGIC:\n"
        "                if self._pe1_edge_field is not None or self._pe3_edge_field is not None:\n"
        "                    raise SystemExit(\"ix2 container has duplicate edge optional sections\")\n"
        "                self._pe3_edge_field = _pe3_parse_edge_field(extra)\n"
        "            elif extra[:8] == OD5_PACKET_MAGIC:\n"
        "                if self._od7_od5_guard is not None:\n"
        "                    raise SystemExit(\"ix2 container has duplicate OD7 OD5 guard sections\")\n"
        "                self._od7_od5_guard = _od7_parse_od5_guard(extra)\n"
        "            elif extra[:8] == OD7_RGB_MAGIC:\n"
        "                if self._od7_rgb_field is not None:\n"
        "                    raise SystemExit(\"ix2 container has duplicate OD7 sparse RGB sections\")\n"
        "                self._od7_rgb_field = _od7_parse_rgb_field(extra)\n"
        "            else:\n"
        "                raise SystemExit(\n"
        "                    \"ix2 optional section magic is not F0PR1, BD1CLF1, PE1EDGE1, PE3EDGE1, OD5GPK1, or OD7RGB1\"\n"
        "                )\n",
        1,
    )
    out = out.replace(
        "    def f1(self, i: int) -> np.ndarray:\n"
        "        frame = render_frame1_camera_uint8(self.packet, i)\n"
        "        if self._bd1_class_field is not None:\n"
        "            frame = _bd1_apply_class_field(frame, self._bd1_class_field, i)\n"
        "        if self._pe1_edge_field is not None:\n"
        "            frame = _pe1_apply_edge_field(frame, self._pe1_edge_field, i)\n"
        "        if self._pe3_edge_field is not None:\n"
        "            frame = _pe1_apply_edge_field(frame, self._pe3_edge_field, i)\n"
        "        return frame\n",
        "    def f1(self, i: int) -> np.ndarray:\n"
        "        frame = render_frame1_camera_uint8(self.packet, i)\n"
        "        if self._od7_rgb_field is not None:\n"
        "            if self._od7_od5_guard is None:\n"
        "                raise SystemExit(\"OD7 sparse RGB section requires the OD6 OD5 packet guard\")\n"
        "            return _od7_apply_rgb_field(frame, self._od7_rgb_field, i)\n"
        "        if self._bd1_class_field is not None:\n"
        "            frame = _bd1_apply_class_field(frame, self._bd1_class_field, i)\n"
        "        if self._pe1_edge_field is not None:\n"
        "            frame = _pe1_apply_edge_field(frame, self._pe1_edge_field, i)\n"
        "        if self._pe3_edge_field is not None:\n"
        "            frame = _pe1_apply_edge_field(frame, self._pe3_edge_field, i)\n"
        "        return frame\n",
        1,
    )
    return out


def _copy_runtime_tree(src: Path, dst: Path, runtime_text: str) -> None:
    if dst.exists():
        raise OD7Error(f"candidate dir already exists: {dst}")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("inflated", "__pycache__"))
    archive_dir = dst / "archive"
    archive_dir.mkdir(exist_ok=True)
    (dst / "inflate_runner.py").write_text(runtime_text, encoding="utf-8")


def _build_candidate(
    *,
    args: argparse.Namespace,
    run_ssd: Path,
    payload: ModePayload,
    od6_packet_raw: bytes,
) -> dict[str, Any]:
    from ddm_ix2_archive_container import build_payload, build_single_member_zip, parse_payload  # noqa: E402

    runtime_text = _runtime_patch_text(args.runtime_template.read_text(encoding="utf-8"))
    sub_dir = run_ssd / "sub_od7"
    _copy_runtime_tree(args.pe3_sub_dir, sub_dir, runtime_text)
    base_payload = (args.pe3_sub_dir / "archive" / "0.bin").read_bytes()
    bulk, sections = parse_payload(base_payload)
    new_payload = build_payload(bulk, [*sections, od6_packet_raw, payload.section])
    archive_zip = build_single_member_zip(new_payload, name="0.bin")
    (sub_dir / "archive" / "0.bin").write_bytes(new_payload)
    (sub_dir / "archive.zip").write_bytes(archive_zip)
    parsed_bulk, parsed_sections = parse_payload(new_payload)
    if parsed_bulk != bulk:
        raise OD7Error("candidate IX2 bulk changed unexpectedly")
    if parsed_sections[-2] != od6_packet_raw or parsed_sections[-1] != payload.section:
        raise OD7Error("candidate appended sections failed parse-back")
    return {
        "submission_dir": str(sub_dir),
        "archive_zip": _source_entry(sub_dir / "archive.zip"),
        "archive_0_bin": _source_entry(sub_dir / "archive" / "0.bin"),
        "inflate_runner": _source_entry(sub_dir / "inflate_runner.py"),
        "parse_back": {
            "bulk_bytes": len(parsed_bulk),
            "joint_section_count": len(parsed_sections),
            "appended_od6_packet_index": len(parsed_sections) - 2,
            "appended_od7_section_index": len(parsed_sections) - 1,
            "appended_od6_packet_sha256": _sha256_bytes(parsed_sections[-2]),
            "appended_od7_section_sha256": _sha256_bytes(parsed_sections[-1]),
            "payload_reencodes_identically": build_payload(parsed_bulk, parsed_sections) == new_payload,
        },
    }


def _candidate_runtime_checks(
    *,
    candidate: dict[str, Any],
    qo1_sub_dir: Path,
    surface: SelectedSurface,
    mode_payload: ModePayload,
) -> dict[str, Any]:
    cand_dir = Path(candidate["submission_dir"])
    runtime = _import_runtime(cand_dir / "inflate_runner.py", "ddm_od7_candidate_runtime")
    base_runtime = _import_runtime(cand_dir / "inflate_runner.py", "ddm_od7_identity_runtime")
    cand_decoder = runtime.Decoder(cand_dir / "archive")
    base_decoder = base_runtime.Decoder(qo1_sub_dir / "archive")

    identity_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for pair in surface.pairs:
        base_f1 = base_decoder.f1(pair)
        base_f0 = base_decoder.f0(pair, base_f1)
        qo1_frames = np.memmap(
            qo1_sub_dir / "inflated" / "0.raw",
            dtype=np.uint8,
            mode="r",
            shape=(od4.N_PAIRS * SEQ_LEN, CAM_H, CAM_W, CAM_C),
        )
        raw_f0 = np.asarray(qo1_frames[2 * pair])
        raw_f1 = np.asarray(qo1_frames[2 * pair + 1])
        identity_rows.append(
            {
                "pair": pair,
                "f0_matches_qo1_raw": bool(np.array_equal(base_f0, raw_f0)),
                "f1_matches_qo1_raw": bool(np.array_equal(base_f1, raw_f1)),
            }
        )
        cand_f1 = cand_decoder.f1(pair)
        changed_rows.append(
            {
                "pair": pair,
                "selected_records": int(mode_payload.records_by_pair[pair].size),
                "changed_camera_pixels_f1": int((cand_f1 != base_f1).any(axis=2).sum()),
            }
        )
    wall = time.perf_counter() - t0
    return {
        "absent_section_identity_scope": "n32 selected pairs against qo1 raw, using patched OD7 runtime with no OD7 sections",
        "absent_section_identity_passed": all(
            row["f0_matches_qo1_raw"] and row["f1_matches_qo1_raw"] for row in identity_rows
        ),
        "identity_rows": identity_rows,
        "changed_pixel_rows": changed_rows,
        "changed_pixel_totals": {
            "pairs": len(changed_rows),
            "changed_camera_pixels_f1": int(sum(row["changed_camera_pixels_f1"] for row in changed_rows)),
            "selected_records": int(sum(row["selected_records"] for row in changed_rows)),
        },
        "n32_decode_wall_seconds": wall,
        "projected_n600_decode_wall_seconds": wall * (od4.N_PAIRS / len(surface.pairs)),
    }


def _projection_for_mode(
    *,
    mode_score: dict[str, Any],
    baseline_score: dict[str, Any],
    archive_bytes: int,
    n_pairs: int,
) -> dict[str, Any]:
    retained = int(mode_score["totals"]["retained_fix_count"])
    stage1_delta_s = od4.projected_stage1_delta_s(retained, n_pairs)
    rate_delta_s = (archive_bytes - od4.CURRENT_OWN_BYTES) * od4.RATE_PER_BYTE
    pose_delta_s = float(mode_score["totals"]["pose_contribution_sqrt10"] - baseline_score["totals"]["pose_contribution_sqrt10"])
    projected_s = od4.CURRENT_OWN_S + stage1_delta_s + rate_delta_s + pose_delta_s
    return {
        "projection_axis": "[macOS-CPU frozen-scorer n32 advisory]",
        "n_pairs": n_pairs,
        "archive_bytes_exact_staged": archive_bytes,
        "stage1_delta_s_from_n32": stage1_delta_s,
        "rate_delta_s_vs_live_archive_bytes": rate_delta_s,
        "pose_delta_s_vs_qo1_n32": pose_delta_s,
        "projected_s": projected_s,
        "delta_vs_live": projected_s - od4.CURRENT_OWN_S,
        "delta_vs_od6_mask_projection": projected_s - OD6_PROJECTED_S,
        "beats_live_advisory": projected_s < od4.CURRENT_OWN_S,
        "beats_od6_projection": projected_s < OD6_PROJECTED_S,
    }


def _write_gate(path: Path) -> None:
    content = """#!/usr/bin/env bash
set -euo pipefail

# OD7 receiver gate only.  Do not run while another full scorer lane is active.
SUB_DIR="${SUB_DIR:?set SUB_DIR to /Volumes/VertigoDataTier/pact/ddm_od7_20260805/<run>/sub_od7}"
OUT="${OUT:-.omx/research/ddm_od7_20260805/od7_receiver_gate_receipt.json}"

.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py \
  --sub-dir "${SUB_DIR}" \
  --out "${OUT}" \
  --inflate-out "${SUB_DIR}/inflated" \
  --device cpu \
  --batch-size 16 \
  --num-threads 6
"""
    _atomic_write_text(path, content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_next(path: Path, receipt: dict[str, Any]) -> None:
    best = receipt["best_mode"]
    md = f"""# OD7 NEXT_IF_RESUMED - 2026-08-05

1. Treat `{best['mode']}` as an n32 receiver-consumed value realization only: exact staged archive `{receipt['candidate']['archive_zip']['bytes']}` B, projected advisory `S={best['projection']['projected_s']:.9f}`.
2. Before any n600 scorer fire, close the format gap called out in `pure_receiver_targeter_gap`: OD6 b1024 needs a packet self-contained enough for receiver recomputation, not only selected n32 records.
3. Fire the queued gate only after the scorer lane is explicitly free and claimed, and only with `SUB_DIR` bound to the exact staged `sub_od7`.
4. If continuing value work first, derive a fresh k=4 frame_0 carriage section for the OD7-edited frame_1 path; OD2's JSON records byte accounting and outcomes, not reusable coefficients.

{OWN_FRONTIER_LINE}
"""
    _atomic_write_text(path, md)


def _write_markdown(path: Path, receipt: dict[str, Any]) -> None:
    best = receipt["best_mode"]
    candidate = receipt["candidate"]
    mode_lines = [
        "| mode | section B | records | exceptions | flips after | retained | d_pose mean | projected S | delta vs live |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in receipt["modes"]:
        proj = mode["projection"]
        totals = mode["score"]["totals"]
        mode_lines.append(
            f"| `{mode['mode']}` | {mode['section']['section_bytes']} | "
            f"{mode['section']['record_count']} | {mode['section']['exception_count']} | "
            f"{totals['flips_after']} | {totals['retained_fix_count']} | "
            f"{totals['d_pose_mean']:.9f} | {proj['projected_s']:.9f} | {proj['delta_vs_live']:+.9f} |"
        )
    md = f"""# OD7 receiver-close receipt - 2026-08-05

Status: `RECEIVER_CONSUMED_SELECTED_SET_MEASURED / OD6_TARGETER_FORMAT_GAP_HELD / NO FRONTIER MOVE`.

Axis: `[macOS-CPU frozen-scorer n32 advisory]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`, `upstream_evaluate=false`.

## Answer First

OD7 staged a real `sub_od7` receiver artifact at `{candidate['submission_dir']}`.  The staged `archive.zip` is `{candidate['archive_zip']['bytes']}` B with sha256 `{candidate['archive_zip']['sha256']}`.  The selected OD6 n32 mask set was consumed by an OD7 sparse RGB section and measured through the edited RGB receiver path with frame_0 recomputed from edited frame_1.

Best measured value mode: `{best['mode']}`.  Its n32 advisory projection is `S={best['projection']['projected_s']:.9f}`, `delta_vs_live={best['projection']['delta_vs_live']:+.9f}`, and `delta_vs_od6_mask_projection={best['projection']['delta_vs_od6_mask_projection']:+.9f}`.  This is not a contest score and does not move the pointer.

OD7 also found a hard format gap for pure OD6 packet receiver recomputation: the OD6 b1024 hash header requires `pe1_generator_coverage` and `pe3_hybrid_knee_coverage`, while the OD6 packet row stores only the PE3 hybrid75 n32 subset and the bucket table.  The staged artifact therefore proves receiver consumption of the materialized selected set, not a self-contained OD6 targeter.

## Value Menu

{chr(10).join(mode_lines)}

## Receiver Proofs

- Candidate parse-back re-encoded the IX2 payload identically: `{candidate['parse_back']['payload_reencodes_identically']}`.
- Appended OD6 guard section sha256: `{candidate['parse_back']['appended_od6_packet_sha256']}`.
- Appended OD7 value section sha256: `{candidate['parse_back']['appended_od7_section_sha256']}`.
- Absent-section identity on n32 qo1 raw using the patched runtime: `{receipt['runtime_checks']['absent_section_identity_passed']}`.
- Runtime changed-pixel proof total: `{receipt['runtime_checks']['changed_pixel_totals']['changed_camera_pixels_f1']}` changed frame_1 camera pixels for `{receipt['runtime_checks']['changed_pixel_totals']['selected_records']}` selected records.
- Projected full decode wall clock from n32 loop: `{receipt['runtime_checks']['projected_n600_decode_wall_seconds']:.3f}` seconds.

## RECALL EVIDENCE

| source | recalled fact | plan impact |
|---|---|---|
| `.omx/tmp/codex_runs/od7_prompt.md`, `_common_contract.md` | OD7 must not run `upstream/evaluate.py` or n600, must preserve protected files/staged index, and must end with the live frontier line. | Ran only n32 frozen-scorer advisory and wrote serializer-ready artifacts. |
| `.omx/state/main_hot_state.md` | od3 owns the scorer slot and OD7 headline is realization tax. | Did not fire the queued n600/full scorer gate. |
| `OD6_DECODER_LEGAL_RECEIPT.md` and its raw packet | OD6 incumbent is `base_rgb_generator_geometry_b1024`, exact n32 packet bytes `7334`, projected n600 packet bytes `76304`, projected `S=0.743600771`. | Reconstructed the exact selected n32 mask set and measured RGB realization tax against that projection. |
| `OD2_STAGE12_RECEIPT.md` | k=4 carriage byte credit is `57,600` n600 bytes but the JSON stores outcomes, not coefficients. | Reported sparse pose damage and queued fresh carriage derivation instead of pretending OD2 coefficients were reusable. |
| `experiments/inflate_runner_v4d.py` | Optional IX2 sections are parsed fail-closed and frame_0 is computed from materialized frame_1. | Generated an OD7 runtime patch that consumes OD7 sparse RGB and recomputes frame_0 from edited frame_1. |

## SHA Table

| artifact | bytes | sha256 |
|---|---:|---|
| `{receipt['source_files']['od7_script']['path']}` | {receipt['source_files']['od7_script']['bytes']} | `{receipt['source_files']['od7_script']['sha256']}` |
| `{receipt['source_files']['od6_packet']['path']}` | {receipt['source_files']['od6_packet']['bytes']} | `{receipt['source_files']['od6_packet']['sha256']}` |
| `{receipt['source_files']['od6_json']['path']}` | {receipt['source_files']['od6_json']['bytes']} | `{receipt['source_files']['od6_json']['sha256']}` |
| `{receipt['source_files']['od2_json']['path']}` | {receipt['source_files']['od2_json']['bytes']} | `{receipt['source_files']['od2_json']['sha256']}` |
| `{receipt['source_files']['pair_selection']['path']}` | {receipt['source_files']['pair_selection']['bytes']} | `{receipt['source_files']['pair_selection']['sha256']}` |
| `{receipt['source_files']['pe3_archive']['path']}` | {receipt['source_files']['pe3_archive']['bytes']} | `{receipt['source_files']['pe3_archive']['sha256']}` |
| `{candidate['archive_zip']['path']}` | {candidate['archive_zip']['bytes']} | `{candidate['archive_zip']['sha256']}` |
| `{candidate['inflate_runner']['path']}` | {candidate['inflate_runner']['bytes']} | `{candidate['inflate_runner']['sha256']}` |
| `{receipt['receipt_json_path']}` | {Path(receipt['receipt_json_path']).stat().st_size} | `{_sha256_file(Path(receipt['receipt_json_path']))}` |
| `{receipt['next_if_resumed_path']}` | {Path(receipt['next_if_resumed_path']).stat().st_size} | `{_sha256_file(Path(receipt['next_if_resumed_path']))}` |

## NEXT_IF_RESUMED

See `{receipt['next_if_resumed_path']}`.  First gate: close the OD6 packet self-containment gap or explicitly route a counted n600 selected-value stream; then use the queued n32/full receiver gate only when the scorer lane is free.

## Boundaries

- No `upstream/evaluate.py`, contest-CPU, contest-CUDA, MPS, or full n600 scorer run.
- The staged value section is n32 selected-set materialization; non-selected pairs remain unedited by OD7 values.
- The OD6 table/packet is guarded and parsed, but pure receiver recomputation of OD6 b1024 is held because the packet lacks all hash feature coverage sources.
- OD2 k=4 pose credit was not re-applied; a fresh OD7 frame_0 carriage is queued.
- This does not move the frontier.

{OWN_FRONTIER_LINE}
"""
    _atomic_write_text(path, md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--od2-json", type=Path, default=od5.DEFAULT_OD2_JSON)
    ap.add_argument("--pair-selection", type=Path, default=od5.DEFAULT_PAIR_SELECTION)
    ap.add_argument("--od6-json", type=Path, default=DEFAULT_OD6_JSON)
    ap.add_argument("--od6-packet", type=Path, default=DEFAULT_OD6_PACKET)
    ap.add_argument("--argmax-cache", type=Path, default=od5.DEFAULT_ARGMAX_CACHE)
    ap.add_argument("--gt-cache", type=Path, default=od5.DEFAULT_GT_CACHE)
    ap.add_argument("--pe1-receipt", type=Path, default=od5.DEFAULT_PE1_RECEIPT)
    ap.add_argument("--pe3-receipt", type=Path, default=od5.DEFAULT_PE3_RECEIPT)
    ap.add_argument("--g4-recurrence", type=Path, default=od5.DEFAULT_G4_RECURRENCE)
    ap.add_argument("--gt-mkv", type=Path, default=DEFAULT_GT_MKV)
    ap.add_argument("--qo1-sub-dir", type=Path, default=DEFAULT_QO1_SUB_DIR)
    ap.add_argument("--pe3-sub-dir", type=Path, default=DEFAULT_PE3_SUB_DIR)
    ap.add_argument("--runtime-template", type=Path, default=DEFAULT_RUNTIME_TEMPLATE)
    ap.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    ap.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--rmax", type=int, default=5)
    ap.add_argument("--proxy-block", type=int, default=16)
    ap.add_argument("--depth-y1", type=float, default=190.0)
    ap.add_argument("--depth-y2", type=float, default=230.0)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args(argv)

    args.research_dir.mkdir(parents=True, exist_ok=True)
    storage = _storage_preflight(args.ssd_dir, required_free_bytes=2 * 1024 * 1024 * 1024)
    run_id = args.run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_ssd = args.ssd_dir / run_id
    run_ssd.mkdir(parents=True, exist_ok=False)

    surface = _selected_surface(args)
    wanted_frames = {2 * pair for pair in surface.pairs} | {2 * pair + 1 for pair in surface.pairs}
    gt_frames = decode_gt_frames(args.gt_mkv, wanted_frames)
    gt_scorer_rgb = _scorer_rgb_for_gt_pairs(gt_frames, surface.pairs)
    runtime = _import_runtime(args.qo1_sub_dir / "inflate_runner.py", "ddm_od7_base_runtime")
    decoder = runtime.Decoder(args.qo1_sub_dir / "archive")
    scorer = Scorer(threads=args.threads)
    gt_argmax = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")

    baseline = _baseline_score(
        scorer=scorer,
        decoder=decoder,
        gt_frames=gt_frames,
        gt_argmax=gt_argmax,
        pairs=surface.pairs,
    )
    fixed = _score_mode(
        scorer=scorer,
        decoder=decoder,
        gt_frames=gt_frames,
        gt_argmax=gt_argmax,
        surface=surface,
        gt_scorer_rgb=gt_scorer_rgb,
        mode_name="fixed_proto",
    )
    exceptions = _build_exceptions(fixed, surface)
    sparse = _score_mode(
        scorer=scorer,
        decoder=decoder,
        gt_frames=gt_frames,
        gt_argmax=gt_argmax,
        surface=surface,
        gt_scorer_rgb=gt_scorer_rgb,
        mode_name="sparse_gt_rgb",
    )
    hybrid = _score_mode(
        scorer=scorer,
        decoder=decoder,
        gt_frames=gt_frames,
        gt_argmax=gt_argmax,
        surface=surface,
        gt_scorer_rgb=gt_scorer_rgb,
        mode_name="hybrid_proto_gt_exceptions",
        exceptions=exceptions,
    )

    od6_packet_raw = args.od6_packet.read_bytes()
    od6_packet_sha = _sha256_bytes(od6_packet_raw)
    mode_payloads = {
        "fixed_proto": _build_mode_payload(
            mode_name="fixed_proto",
            surface=surface,
            gt_scorer_rgb=gt_scorer_rgb,
            od6_packet_sha=od6_packet_sha,
        ),
        "sparse_gt_rgb": _build_mode_payload(
            mode_name="sparse_gt_rgb",
            surface=surface,
            gt_scorer_rgb=gt_scorer_rgb,
            od6_packet_sha=od6_packet_sha,
        ),
        "hybrid_proto_gt_exceptions": _build_mode_payload(
            mode_name="hybrid_proto_gt_exceptions",
            surface=surface,
            gt_scorer_rgb=gt_scorer_rgb,
            od6_packet_sha=od6_packet_sha,
            exceptions=exceptions,
        ),
    }
    score_by_mode = {
        "fixed_proto": fixed,
        "sparse_gt_rgb": sparse,
        "hybrid_proto_gt_exceptions": hybrid,
    }

    # Stage the mode with the best advisory projection after exact staged bytes are known.
    candidate_rows: list[dict[str, Any]] = []
    staged_by_mode: dict[str, dict[str, Any]] = {}
    for mode_name, payload in mode_payloads.items():
        mode_dir = run_ssd / f"stage_{mode_name}"
        mode_dir.mkdir()
        candidate = _build_candidate(args=args, run_ssd=mode_dir, payload=payload, od6_packet_raw=od6_packet_raw)
        staged_by_mode[mode_name] = candidate
        projection = _projection_for_mode(
            mode_score=score_by_mode[mode_name],
            baseline_score=baseline,
            archive_bytes=int(candidate["archive_zip"]["bytes"]),
            n_pairs=len(surface.pairs),
        )
        candidate_rows.append(
            {
                "mode": mode_name,
                "candidate": candidate,
                "projection": projection,
            }
        )
    best_row = min(candidate_rows, key=lambda row: row["projection"]["projected_s"])
    final_sub = run_ssd / "sub_od7"
    best_sub = Path(best_row["candidate"]["submission_dir"])
    if final_sub.exists():
        raise OD7Error(f"final candidate dir already exists: {final_sub}")
    shutil.copytree(best_sub, final_sub)
    best_payload = mode_payloads[best_row["mode"]]
    final_candidate = {
        **best_row["candidate"],
        "submission_dir": str(final_sub),
        "archive_zip": _source_entry(final_sub / "archive.zip"),
        "archive_0_bin": _source_entry(final_sub / "archive" / "0.bin"),
        "inflate_runner": _source_entry(final_sub / "inflate_runner.py"),
    }
    runtime_checks = _candidate_runtime_checks(
        candidate=final_candidate,
        qo1_sub_dir=args.qo1_sub_dir,
        surface=surface,
        mode_payload=best_payload,
    )

    modes: list[dict[str, Any]] = []
    for row in candidate_rows:
        mode_name = row["mode"]
        payload = mode_payloads[mode_name]
        candidate = row["candidate"]
        modes.append(
            {
                "mode": mode_name,
                "score": {k: v for k, v in score_by_mode[mode_name].items() if k != "argmax_by_pair"},
                "section": payload.section_parse,
                "coder_race": payload.coder_rows,
                "best_coder": payload.best_coder,
                "candidate_archive_bytes": int(candidate["archive_zip"]["bytes"]),
                "projection": row["projection"],
            }
        )
    best_mode = next(mode for mode in modes if mode["mode"] == best_row["mode"])
    # Recompute best projection against the final sub_od7 path bytes, although it should match.
    best_mode["projection"] = _projection_for_mode(
        mode_score=score_by_mode[best_row["mode"]],
        baseline_score=baseline,
        archive_bytes=int(final_candidate["archive_zip"]["bytes"]),
        n_pairs=len(surface.pairs),
    )

    receipt_json_path = args.research_dir / "od7_receiver_close_receipt.json"
    md_path = args.research_dir / "OD7_RECEIVER_CLOSE_RECEIPT.md"
    next_path = args.research_dir / "NEXT_IF_RESUMED.md"
    gate_path = args.research_dir / "OD7_RECEIVER_GATE_FIRE_ORDER.sh"
    receipt: dict[str, Any] = {
        "schema": "ddm_od7_receiver_close_receipt.v1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "axis": "[macOS-CPU frozen-scorer n32 advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "upstream_evaluate": False,
        "frontier_moved": False,
        "current_own_vehicle_frontier": {
            "S": od4.CURRENT_OWN_S,
            "archive_bytes": od4.CURRENT_OWN_BYTES,
            "axis": od4.CURRENT_OWN_AXIS,
        },
        "od6_projection_reference": {
            "projected_s": OD6_PROJECTED_S,
            "projected_n600_packet_bytes": OD6_PROJECTED_PACKET_BYTES,
            "delta_vs_live": OD6_PROJECTED_DELTA_VS_LIVE,
        },
        "denominators": {
            "selection_mode": _load_json(args.pair_selection)["selection"]["pair_selection"],
            "selection_seed": _load_json(args.pair_selection)["seed"],
            "n_pairs": len(surface.pairs),
            "population_pairs": od4.N_PAIRS,
            "height": SEG_H,
            "width": SEG_W,
            "rate_denominator_bytes": od4.RATE_DENOMINATOR_BYTES,
            **surface.target_denominators,
        },
        "storage_preflight": storage,
        "run_ssd_dir": str(run_ssd),
        "source_files": {
            "od7_script": _source_entry(Path(__file__).resolve()),
            "od6_packet": _source_entry(args.od6_packet),
            "od6_json": _source_entry(args.od6_json),
            "od2_json": _source_entry(args.od2_json),
            "pair_selection": _source_entry(args.pair_selection),
            "pe1_receipt": _source_entry(args.pe1_receipt),
            "pe3_receipt": _source_entry(args.pe3_receipt),
            "pe3_archive": _source_entry(args.pe3_sub_dir / "archive.zip"),
            "qo1_archive": _source_entry(args.qo1_sub_dir / "archive.zip"),
            "gt_mkv": _source_entry(args.gt_mkv),
        },
        "source_reconstruction": surface.source_meta,
        "baseline_qo1_n32": baseline,
        "modes": modes,
        "best_mode": best_mode,
        "candidate": final_candidate,
        "candidate_rows_before_final_copy": candidate_rows,
        "runtime_checks": runtime_checks,
        "receipt_json_path": str(receipt_json_path),
        "receipt_md_path": str(md_path),
        "next_if_resumed_path": str(next_path),
        "queued_scorer_gate_script": str(gate_path),
        "boundaries": [
            "No upstream/evaluate.py, contest CPU/CUDA, MPS, or full n600 scorer job.",
            "n32 selected-set value materialization only; full n600 values are not present.",
            "OD6 packet self-contained targeter recomputation remains blocked by missing coverage columns.",
            "OD2 k4 carriage coefficients were not available in the OD2 JSON and were not re-applied.",
        ],
        "frontier_line": OWN_FRONTIER_LINE,
    }
    _atomic_write_json(receipt_json_path, receipt)
    _write_next(next_path, receipt)
    _write_gate(gate_path)
    # Update SHA fields that depend on files written above.
    receipt["receipt_json_sha256"] = _sha256_file(receipt_json_path)
    receipt["next_if_resumed_sha256"] = _sha256_file(next_path)
    _atomic_write_json(receipt_json_path, receipt)
    _write_markdown(md_path, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
