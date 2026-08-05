#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""OD6 decoder-legal bucket-context repricing for OD5's packet targeter.

This script rebuilds the OD5 context-targeter surface without scorer-native
features.  It uses only qo1 decoded RGB, generic local image features, shipped
generator/hybrid packet state, generic geometry bins, and an optional counted
block-prior proxy.  It does not run SegNet, PoseNet, upstream/evaluate.py, or a
full n600 scorer job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import struct
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "src", REPO / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ddm_od5_generator_packet as od5  # noqa: E402
import ddm_pe1_per_edge_partition_race as pe1  # noqa: E402
import ddm_pe3_hybrid_composition as pe3  # noqa: E402
from tac.optimization import ddm_od4_weak_stage1_packet as od4  # noqa: E402

DEFAULT_RESEARCH_DIR: Final = REPO / ".omx/research/ddm_od6_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_od6_20260805")
DEFAULT_QO1_SUB_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit")
DEFAULT_BUCKET_COUNTS: Final = "512,1024,2048,4096,8192,16384"
OD5_SCORER_NATIVE_BASELINE_S: Final = 0.743783052
OD5_SCORER_NATIVE_BASELINE_BYTES: Final = 78_010
OD5_SCORER_NATIVE_BASELINE_RETAINED_FIXES: Final = 6_171
OD5_SCORER_NATIVE_BASELINE_ETA: Final = 0.553900
OD5_SCORER_NATIVE_BASELINE_RATIO: Final = 0.530
OD4_SPARSE_BASELINE_S: Final = 0.761509399
OD4_SPARSE_BASELINE_RATIO: Final = 0.711
EXPECTED_RETAINED_FIX_DENOMINATOR: Final = 6_177
SEG_H: Final = od4.SEG_H
SEG_W: Final = od4.SEG_W
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
CAMERA_C: Final = 3
SEQ_LEN: Final = 2
OD6_TABLE_MAGIC: Final = b"OD6DLT1\0"
OD6_PROXY_MAGIC: Final = b"OD6PRX1\0"
QLOGIT_SCALE: Final = 1024.0


class OD6Error(ValueError):
    """OD6 decoder-legal targeter build failed a typed invariant."""


@dataclass(frozen=True, slots=True)
class BaseFrameFeatures:
    r_bin: np.ndarray
    g_bin: np.ndarray
    b_bin: np.ndarray
    y_bin: np.ndarray
    u_bin: np.ndarray
    v_bin: np.ndarray
    gx_bin: np.ndarray
    gy_bin: np.ndarray
    contrast_bin: np.ndarray
    chroma_delta_bin: np.ndarray
    rgb_sha256: str


@dataclass(frozen=True, slots=True)
class PointSet:
    pair: np.ndarray
    y: np.ndarray
    x: np.ndarray
    flat: np.ndarray
    label: np.ndarray

    @property
    def size(self) -> int:
        return int(self.pair.size)


@dataclass(frozen=True, slots=True)
class ProxyPayload:
    raw: bytes
    qlogits: np.ndarray
    header: dict[str, Any]
    coder_rows: tuple[od4.CoderRow, ...]
    best: od4.CoderRow


@dataclass(frozen=True, slots=True)
class TableFit:
    mode: str
    bucket_count: int
    raw: bytes
    qlogits: np.ndarray
    threshold: float
    header: dict[str, Any]
    coder_rows: tuple[od4.CoderRow, ...]
    best: od4.CoderRow
    selected_by_pair: dict[int, np.ndarray]
    selected_fix_count: int
    train_meta: dict[str, Any]


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


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OD6Error(f"JSON root is not an object: {path}")
    return data


def _storage_preflight(path: Path, required_free_bytes: int) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "required_free_bytes": int(required_free_bytes),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "ok": bool(usage.free >= required_free_bytes),
    }


def _parse_int_csv(raw: str) -> list[int]:
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise OD6Error(f"bucket count must be positive: {value}")
        out.append(value)
    if not out:
        raise OD6Error("empty integer CSV")
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def _logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob.astype(np.float32), np.float32(1e-4), np.float32(1.0 - 1e-4))
    return np.log(clipped / (np.float32(1.0) - clipped), dtype=np.float32)


def _best_coder(rows: tuple[od4.CoderRow, ...]) -> od4.CoderRow:
    candidates = [row for row in rows if row.parseback_exact and row.bytes > 0]
    if not candidates:
        raise OD6Error("no coder row survived parse-back")
    return min(candidates, key=lambda row: row.bytes)


def _varint(value: int) -> bytes:
    if value < 0:
        raise OD6Error("varint cannot encode a negative value")
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
            raise OD6Error("truncated varint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise OD6Error("varint too long")


def _serialize_qlogit_payload(*, magic: bytes, header: dict[str, Any], qlogits: np.ndarray) -> bytes:
    encoded_header = json.dumps(_jsonable(header), sort_keys=True, separators=(",", ":")).encode("utf-8")
    q = np.ascontiguousarray(qlogits.astype("<i2", copy=False))
    return magic + _varint(len(encoded_header)) + encoded_header + q.tobytes()


def _parse_qlogit_payload(payload: bytes, *, magic: bytes) -> tuple[dict[str, Any], np.ndarray]:
    if not payload.startswith(magic):
        raise OD6Error("qlogit payload magic mismatch")
    offset = len(magic)
    header_len, offset = _read_varint(payload, offset)
    header_bytes = payload[offset : offset + header_len]
    if len(header_bytes) != header_len:
        raise OD6Error("qlogit payload header truncated")
    offset += header_len
    header = json.loads(header_bytes.decode("utf-8"))
    qlogit_count = int(header["qlogit_count"])
    qbytes = payload[offset:]
    if len(qbytes) != qlogit_count * 2:
        raise OD6Error("qlogit payload length mismatch")
    return header, np.frombuffer(qbytes, dtype="<i2").astype(np.int16, copy=True)


def _race_and_verify_payload(raw: bytes, *, magic: bytes, expected_qlogits: np.ndarray) -> tuple[tuple[od4.CoderRow, ...], od4.CoderRow]:
    header, qlogits = _parse_qlogit_payload(raw, magic=magic)
    if int(header["qlogit_count"]) != int(expected_qlogits.size):
        raise OD6Error("qlogit count changed under parse-back")
    if not np.array_equal(qlogits, np.asarray(expected_qlogits, dtype=np.int16)):
        raise OD6Error("qlogit payload parse-back mismatch")
    rows = od4.race_packet_coders(raw)
    return rows, _best_coder(rows)


def _quantize_unit(values: np.ndarray, levels: int) -> np.ndarray:
    clipped = np.clip(values, 0.0, 255.0)
    return np.minimum((clipped * (levels / 256.0)).astype(np.uint8), levels - 1)


def _bin_by_thresholds(values: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.uint8)
    for idx, threshold in enumerate(thresholds, start=1):
        out += (values >= threshold).astype(np.uint8)
    return out


def _resize_last_frame_to_seg(frame: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.asarray(frame, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    resized = functional.interpolate(tensor, size=(SEG_H, SEG_W), mode="bilinear")
    arr = resized.squeeze(0).permute(1, 2, 0).numpy()
    return np.rint(np.clip(arr, 0.0, 255.0)).astype(np.uint8)


def _features_for_rgb(rgb: np.ndarray) -> BaseFrameFeatures:
    rgb16 = rgb.astype(np.float32)
    r = rgb16[:, :, 0]
    g = rgb16[:, :, 1]
    b = rgb16[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = np.clip((b - y) / 1.772 + 128.0, 0.0, 255.0)
    v = np.clip((r - y) / 1.402 + 128.0, 0.0, 255.0)
    gx = np.zeros_like(y, dtype=np.float32)
    gy = np.zeros_like(y, dtype=np.float32)
    gx[:, 1:] = np.abs(y[:, 1:] - y[:, :-1])
    gy[1:, :] = np.abs(y[1:, :] - y[:-1, :])
    y_pad = np.pad(y, 1, mode="edge")
    local_mean = (
        y_pad[:-2, :-2]
        + y_pad[:-2, 1:-1]
        + y_pad[:-2, 2:]
        + y_pad[1:-1, :-2]
        + y_pad[1:-1, 1:-1]
        + y_pad[1:-1, 2:]
        + y_pad[2:, :-2]
        + y_pad[2:, 1:-1]
        + y_pad[2:, 2:]
    ) / 9.0
    contrast = np.abs(y - local_mean)
    chroma_delta = np.abs(u - 128.0) + np.abs(v - 128.0)
    return BaseFrameFeatures(
        r_bin=_quantize_unit(r, 8),
        g_bin=_quantize_unit(g, 8),
        b_bin=_quantize_unit(b, 8),
        y_bin=_quantize_unit(y, 8),
        u_bin=_quantize_unit(u, 8),
        v_bin=_quantize_unit(v, 8),
        gx_bin=_bin_by_thresholds(gx, (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)),
        gy_bin=_bin_by_thresholds(gy, (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)),
        contrast_bin=_bin_by_thresholds(contrast, (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)),
        chroma_delta_bin=_bin_by_thresholds(chroma_delta, (4.0, 8.0, 16.0, 32.0, 64.0, 96.0)),
        rgb_sha256=_sha256_bytes(np.ascontiguousarray(rgb).tobytes()),
    )


def _load_qo1_base_frame_features(qo1_sub_dir: Path, pairs: list[int]) -> tuple[dict[int, BaseFrameFeatures], dict[str, Any]]:
    raw_path = qo1_sub_dir / "inflated" / "0.raw"
    expected_bytes = od4.N_PAIRS * SEQ_LEN * CAMERA_H * CAMERA_W * CAMERA_C
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise OD6Error(f"qo1 inflated raw bytes {actual_bytes} != expected {expected_bytes}")
    frames = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(od4.N_PAIRS * SEQ_LEN, CAMERA_H, CAMERA_W, CAMERA_C))
    out: dict[int, BaseFrameFeatures] = {}
    for pair in pairs:
        frame_index = int(pair) * SEQ_LEN + 1
        out[pair] = _features_for_rgb(_resize_last_frame_to_seg(np.asarray(frames[frame_index])))
    return out, {
        "path": str(raw_path),
        "bytes": actual_bytes,
        "sha256": _sha256_file(raw_path),
        "camera_shape": [od4.N_PAIRS * SEQ_LEN, CAMERA_H, CAMERA_W, CAMERA_C],
        "seg_resize": {
            "implementation": "torch.nn.functional.interpolate",
            "mode": "bilinear",
            "size_hw": [SEG_H, SEG_W],
            "align_corners": "default_false",
        },
        "frame_index_rule": "last frame for pair p is raw frame 2*p+1",
        "selected_pair_rgb_shas": {str(pair): out[pair].rgb_sha256 for pair in pairs},
    }


def _build_point_sets(
    *,
    target_pairs: dict[int, od5.TargetPair],
    pairs: list[int],
    current_argmax: np.ndarray,
    gt_argmax: np.ndarray,
) -> tuple[PointSet, PointSet, dict[str, Any]]:
    all_pairs: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_x: list[np.ndarray] = []
    all_flat: list[np.ndarray] = []
    all_label: list[np.ndarray] = []
    pos_pairs: list[np.ndarray] = []
    pos_y: list[np.ndarray] = []
    pos_x: list[np.ndarray] = []
    pos_flat: list[np.ndarray] = []
    pos_label: list[np.ndarray] = []
    per_pair: list[dict[str, Any]] = []
    for pair in pairs:
        current = np.asarray(current_argmax[pair], dtype=np.uint8)
        gt = np.asarray(gt_argmax[pair], dtype=np.uint8)
        wrong = np.flatnonzero((current != gt).reshape(-1)).astype(np.int64)
        positives = np.asarray(target_pairs[pair].full_record.flat_indices, dtype=np.int64)
        positive_mask = np.isin(wrong, positives, assume_unique=False)
        labels = positive_mask.astype(np.uint8)
        all_pairs.append(np.full(wrong.size, int(pair), dtype=np.int16))
        all_y.append((wrong // SEG_W).astype(np.int16))
        all_x.append((wrong % SEG_W).astype(np.int16))
        all_flat.append(wrong.astype(np.int64))
        all_label.append(labels)
        pos_pairs.append(np.full(positives.size, int(pair), dtype=np.int16))
        pos_y.append((positives // SEG_W).astype(np.int16))
        pos_x.append((positives % SEG_W).astype(np.int16))
        pos_flat.append(positives.astype(np.int64))
        pos_label.append(np.ones(positives.size, dtype=np.uint8))
        per_pair.append(
            {
                "pair": pair,
                "wrong_candidates": int(wrong.size),
                "positive_od2_stage1_fix_points": int(positives.size),
                "negative_wrong_candidates": int(wrong.size - positives.size),
            }
        )
    universe = PointSet(
        pair=np.concatenate(all_pairs),
        y=np.concatenate(all_y),
        x=np.concatenate(all_x),
        flat=np.concatenate(all_flat),
        label=np.concatenate(all_label),
    )
    positives = PointSet(
        pair=np.concatenate(pos_pairs),
        y=np.concatenate(pos_y),
        x=np.concatenate(pos_x),
        flat=np.concatenate(pos_flat),
        label=np.concatenate(pos_label),
    )
    denominator = int(positives.size)
    if denominator != EXPECTED_RETAINED_FIX_DENOMINATOR:
        raise OD6Error(f"retained-fix denominator {denominator} != expected {EXPECTED_RETAINED_FIX_DENOMINATOR}")
    meta = {
        "candidate_scope": (
            "training labels use the n32 currently-wrong scorer-lattice cells; "
            "feature columns themselves are decoder-legal and contain no GT/scorer values"
        ),
        "retained_fix_denominator": denominator,
        "wrong_candidate_count": int(universe.size),
        "negative_wrong_candidate_count": int(universe.size - denominator),
        "per_pair": per_pair,
    }
    return universe, positives, meta


def _mode_columns(
    *,
    mode: str,
    points: PointSet,
    base_features: dict[int, BaseFrameFeatures],
    generator_masks: dict[int, np.ndarray],
    hybrid75_masks: dict[int, np.ndarray],
    hybrid_knee_masks: dict[int, np.ndarray],
    proxy_qlogits: np.ndarray | None,
    proxy_block: int,
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, values: np.ndarray) -> None:
        columns.append(np.asarray(values, dtype=np.uint32))
        names.append(name)

    r = np.empty(points.size, dtype=np.uint32)
    g = np.empty(points.size, dtype=np.uint32)
    b = np.empty(points.size, dtype=np.uint32)
    yb = np.empty(points.size, dtype=np.uint32)
    ub = np.empty(points.size, dtype=np.uint32)
    vb = np.empty(points.size, dtype=np.uint32)
    gx = np.empty(points.size, dtype=np.uint32)
    gy = np.empty(points.size, dtype=np.uint32)
    contrast = np.empty(points.size, dtype=np.uint32)
    chroma = np.empty(points.size, dtype=np.uint32)
    gen = np.empty(points.size, dtype=np.uint32)
    hy75 = np.empty(points.size, dtype=np.uint32)
    hyk = np.empty(points.size, dtype=np.uint32)

    cursor = 0
    for pair in sorted(set(int(value) for value in points.pair.tolist())):
        idx = np.flatnonzero(points.pair == pair)
        yy = points.y[idx].astype(np.int64)
        xx = points.x[idx].astype(np.int64)
        feat = base_features[pair]
        r[idx] = feat.r_bin[yy, xx]
        g[idx] = feat.g_bin[yy, xx]
        b[idx] = feat.b_bin[yy, xx]
        yb[idx] = feat.y_bin[yy, xx]
        ub[idx] = feat.u_bin[yy, xx]
        vb[idx] = feat.v_bin[yy, xx]
        gx[idx] = feat.gx_bin[yy, xx]
        gy[idx] = feat.gy_bin[yy, xx]
        contrast[idx] = feat.contrast_bin[yy, xx]
        chroma[idx] = feat.chroma_delta_bin[yy, xx]
        gen[idx] = generator_masks[pair][yy, xx].astype(np.uint32)
        hy75[idx] = hybrid75_masks[pair][yy, xx].astype(np.uint32)
        hyk[idx] = hybrid_knee_masks[pair][yy, xx].astype(np.uint32)
        cursor += idx.size
    if cursor != points.size:
        raise OD6Error("point feature cursor mismatch")

    add("r_bin_q8", r)
    add("g_bin_q8", g)
    add("b_bin_q8", b)
    add("y_bin_q8", yb)
    add("u_bin_q8", ub)
    add("v_bin_q8", vb)
    add("gx_bin", gx)
    add("gy_bin", gy)
    add("contrast_bin", contrast)
    add("chroma_delta_bin", chroma)

    if mode in {"base_rgb_generator", "base_rgb_generator_geometry", "base_rgb_generator_geometry_proxy"}:
        add("pe1_generator_coverage", gen)
        add("pe3_hybrid75_coverage", hy75)
        add("pe3_hybrid_knee_coverage", hyk)

    if mode in {"base_rgb_generator_geometry", "base_rgb_generator_geometry_proxy"}:
        yy = points.y.astype(np.uint32)
        xx = points.x.astype(np.uint32)
        add("row_bin_16", yy // 16)
        add("col_bin_16", xx // 16)
        add("row_bin_32", yy // 32)
        add("col_bin_32", xx // 32)
        add("above_horizon_190", (yy < 190).astype(np.uint32))
        add("depth_band_190_230", np.clip((yy.astype(np.int32) - 190) // 10, -8, 24).astype(np.int32) + 8)
        center_dx = np.abs(xx.astype(np.int32) - (SEG_W // 2))
        add("center_dx_bin_32", (center_dx // 32).astype(np.uint32))
        add("left_right_half", (xx >= (SEG_W // 2)).astype(np.uint32))

    if mode == "base_rgb_generator_geometry_proxy":
        if proxy_qlogits is None:
            raise OD6Error("proxy feature mode requested without proxy qlogits")
        block_w = math.ceil(SEG_W / proxy_block)
        by = np.minimum(points.y.astype(np.int32) // proxy_block, proxy_qlogits.size // block_w - 1)
        bx = np.minimum(points.x.astype(np.int32) // proxy_block, block_w - 1)
        proxy_value = proxy_qlogits[by * block_w + bx].astype(np.float32) / np.float32(QLOGIT_SCALE)
        proxy_prob = _sigmoid(proxy_value)
        add("counted_proxy_decile", np.minimum((proxy_prob * 10.0).astype(np.uint32), 9))
        add("counted_proxy_qsign", (proxy_qlogits[by * block_w + bx] >= 0).astype(np.uint32))

    return np.vstack(columns).T.astype(np.uint32, copy=False), names


def _hash_columns(columns: np.ndarray, bucket_count: int) -> np.ndarray:
    if columns.ndim != 2:
        raise OD6Error("feature columns must be rank-2")
    h = np.full(columns.shape[0], np.uint64(1469598103934665603), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    salt = 0x9E3779B97F4A7C15
    for idx in range(columns.shape[1]):
        salt_word = np.uint64(((idx + 1) * salt) & 0xFFFFFFFFFFFFFFFF)
        col = columns[:, idx].astype(np.uint64) + salt_word
        h ^= col
        h *= prime
    return (h % np.uint64(bucket_count)).astype(np.int64)


def _split_train_holdout(points: PointSet, seed: int) -> np.ndarray:
    key = (
        points.flat.astype(np.uint64)
        ^ (points.pair.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87))
        ^ np.uint64(seed)
    )
    key ^= key >> np.uint64(30)
    key *= np.uint64(0xBF58476D1CE4E5B9)
    key ^= key >> np.uint64(27)
    key *= np.uint64(0x94D049BB133111EB)
    key ^= key >> np.uint64(31)
    return (key % np.uint64(5)) != np.uint64(0)


def _fit_proxy_payload(points: PointSet, *, block: int, smoothing: float, seed: int) -> ProxyPayload:
    block_h = math.ceil(SEG_H / block)
    block_w = math.ceil(SEG_W / block)
    bucket_count = block_h * block_w
    block_id = (points.y.astype(np.int64) // block) * block_w + (points.x.astype(np.int64) // block)
    train = _split_train_holdout(points, seed)
    total = np.bincount(block_id[train], minlength=bucket_count).astype(np.float32)
    pos = np.bincount(block_id[train], weights=points.label[train].astype(np.float32), minlength=bucket_count)
    prob = (pos + smoothing) / (total + 2.0 * smoothing)
    qlogits = np.rint(_logit(prob) * QLOGIT_SCALE).clip(-32768, 32767).astype(np.int16)
    header = {
        "schema": "ddm_od6_counted_block_proxy.v1",
        "decoder_legality": "video-derived distilled proxy is counted as an OD5 packet section",
        "block": block,
        "block_h": block_h,
        "block_w": block_w,
        "qscale": QLOGIT_SCALE,
        "qlogit_count": int(qlogits.size),
        "smoothing": smoothing,
        "seed": seed,
        "train_points": int(train.sum()),
        "holdout_points": int((~train).sum()),
    }
    raw = _serialize_qlogit_payload(magic=OD6_PROXY_MAGIC, header=header, qlogits=qlogits)
    rows, best = _race_and_verify_payload(raw, magic=OD6_PROXY_MAGIC, expected_qlogits=qlogits)
    return ProxyPayload(raw=raw, qlogits=qlogits, header=header, coder_rows=rows, best=best)


def _threshold_candidates(probs: np.ndarray) -> np.ndarray:
    if probs.size == 0:
        return np.array([0.5], dtype=np.float32)
    quantiles = np.linspace(0.02, 0.98, 97, dtype=np.float32)
    candidates = np.quantile(probs.astype(np.float32), quantiles)
    fixed = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90], dtype=np.float32)
    return np.unique(np.concatenate([candidates.astype(np.float32), fixed]))


def _fit_decoder_legal_table(
    *,
    mode: str,
    bucket_count: int,
    universe: PointSet,
    positives: PointSet,
    base_features: dict[int, BaseFrameFeatures],
    generator_masks: dict[int, np.ndarray],
    hybrid75_masks: dict[int, np.ndarray],
    hybrid_knee_masks: dict[int, np.ndarray],
    proxy_qlogits: np.ndarray | None,
    proxy_block: int,
    seed: int,
    smoothing: float,
) -> TableFit:
    columns, column_names = _mode_columns(
        mode=mode,
        points=universe,
        base_features=base_features,
        generator_masks=generator_masks,
        hybrid75_masks=hybrid75_masks,
        hybrid_knee_masks=hybrid_knee_masks,
        proxy_qlogits=proxy_qlogits,
        proxy_block=proxy_block,
    )
    hashes = _hash_columns(columns, bucket_count)
    train = _split_train_holdout(universe, seed)
    total = np.bincount(hashes[train], minlength=bucket_count).astype(np.float32)
    pos = np.bincount(hashes[train], weights=universe.label[train].astype(np.float32), minlength=bucket_count)
    prob = (pos + smoothing) / (total + 2.0 * smoothing)
    qlogits = np.rint(_logit(prob) * QLOGIT_SCALE).clip(-32768, 32767).astype(np.int16)

    holdout = ~train
    holdout_hashes = hashes[holdout]
    holdout_probs = _sigmoid(qlogits[holdout_hashes].astype(np.float32) / np.float32(QLOGIT_SCALE))
    holdout_labels = universe.label[holdout].astype(bool)
    best_threshold = 0.5
    best_objective = -1.0e30
    best_stats: dict[str, Any] = {}
    for threshold in _threshold_candidates(holdout_probs):
        keep = holdout_probs >= np.float32(threshold)
        tp = int((keep & holdout_labels).sum())
        fp = int((keep & ~holdout_labels).sum())
        fn = int((~keep & holdout_labels).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        objective = tp - 0.25 * fp
        if objective > best_objective:
            best_objective = objective
            best_threshold = float(threshold)
            best_stats = {
                "threshold": best_threshold,
                "objective_tp_minus_quarter_fp": float(objective),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
            }

    pos_columns, _ = _mode_columns(
        mode=mode,
        points=positives,
        base_features=base_features,
        generator_masks=generator_masks,
        hybrid75_masks=hybrid75_masks,
        hybrid_knee_masks=hybrid_knee_masks,
        proxy_qlogits=proxy_qlogits,
        proxy_block=proxy_block,
    )
    pos_hashes = _hash_columns(pos_columns, bucket_count)
    pos_probs = _sigmoid(qlogits[pos_hashes].astype(np.float32) / np.float32(QLOGIT_SCALE))
    keep_pos = pos_probs >= np.float32(best_threshold)
    selected_by_pair: dict[int, list[int]] = {}
    for pair, flat, keep in zip(positives.pair.tolist(), positives.flat.tolist(), keep_pos.tolist(), strict=True):
        if keep:
            selected_by_pair.setdefault(int(pair), []).append(int(flat))
    selected = {
        int(pair): np.asarray(sorted(selected_by_pair.get(int(pair), [])), dtype=np.int64)
        for pair in sorted(set(int(value) for value in positives.pair.tolist()))
    }
    selected_fix_count = int(sum(arr.size for arr in selected.values()))
    header = {
        "schema": "ddm_od6_decoder_legal_bucket_table.v1",
        "feature_mode": mode,
        "bucket_count": bucket_count,
        "qscale": QLOGIT_SCALE,
        "qlogit_count": int(qlogits.size),
        "threshold": best_threshold,
        "feature_columns": column_names,
        "smoothing": smoothing,
        "seed": seed,
        "train_points": int(train.sum()),
        "holdout_points": int(holdout.sum()),
        "positive_points_total": int(positives.size),
        "selected_positive_points": selected_fix_count,
        "decoder_legal_feature_sources": [
            "qo1 decoded last-frame RGB local bins",
            "shipped generator/hybrid packet coverage bits when feature mode includes generator state",
            "generic row/column/horizon bins when feature mode includes geometry",
            "counted block-prior proxy when feature mode includes proxy",
        ],
        "forbidden_sources_absent": [
            "GT RGB frames",
            "GT margin",
            "Fisher/head-distance caches",
            "scorer weights or scorer forwards",
        ],
    }
    raw = _serialize_qlogit_payload(magic=OD6_TABLE_MAGIC, header=header, qlogits=qlogits)
    rows, best = _race_and_verify_payload(raw, magic=OD6_TABLE_MAGIC, expected_qlogits=qlogits)
    train_meta = {
        "feature_columns": column_names,
        "holdout_threshold_selection": best_stats,
        "bucket_nonempty_train": int((total > 0).sum()),
        "bucket_positive_train": int((pos > 0).sum()),
        "train_positive_rate": float(universe.label[train].mean()) if int(train.sum()) else None,
        "holdout_positive_rate": float(universe.label[holdout].mean()) if int(holdout.sum()) else None,
        "positive_selection_rate": selected_fix_count / positives.size if positives.size else None,
        "payload_raw_bytes": len(raw),
        "payload_raw_sha256": _sha256_bytes(raw),
        "payload_best": best.as_json(),
        "coder_race": [row.as_json() for row in rows],
    }
    return TableFit(
        mode=mode,
        bucket_count=bucket_count,
        raw=raw,
        qlogits=qlogits,
        threshold=best_threshold,
        header=header,
        coder_rows=rows,
        best=best,
        selected_by_pair=selected,
        selected_fix_count=selected_fix_count,
        train_meta=train_meta,
    )


def _table_markdown(receipt: dict[str, Any]) -> str:
    lines = [
        "| surface | bucket_count | exact n32 bytes | projected n600 bytes | retained fixes | eta | S w/ OD2 pose credit | rate/seg ratio | vs OD5 0.530 | vs OD4 0.711 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in receipt["rungs"]:
        totals = row["fidelity"]["totals"]
        projection = row["projection_with_od2_stage2_pose_credit"]
        ratio = float(projection["rate_cost_over_seg_win"])
        od5_cmp = "better" if ratio < OD5_SCORER_NATIVE_BASELINE_RATIO else "worse"
        od4_cmp = "better" if ratio < OD4_SPARSE_BASELINE_RATIO else "worse"
        lines.append(
            "| "
            f"{row['name']} | "
            f"{row['feature_surface']['bucket_count']} | "
            f"{row['packet']['best_coder']['bytes']} | "
            f"{projection['packet_bytes_n600_projected']} | "
            f"{totals['retained_fix_count']} | "
            f"{totals['eta_receiver']:.6f} | "
            f"{projection['projected_s']:.9f} | "
            f"{ratio:.3f} | "
            f"{od5_cmp} | "
            f"{od4_cmp} |"
        )
    return "\n".join(lines)


def _write_gate_script(path: Path) -> None:
    content = """#!/usr/bin/env bash
set -euo pipefail

# OD6 queued scorer gate only. od3 owns the active scorer slot at OD6 build
# time; run this only after a receiver-closed staged OD6 submission exists and
# the scorer lane is explicitly claimed.
SUB_DIR="${SUB_DIR:?set SUB_DIR to the receiver-closed staged submission directory}"
OUT="${OUT:-.omx/research/ddm_od6_20260805/od6_receiver_gate_receipt.json}"

.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py \\
  --sub-dir "${SUB_DIR}" \\
  --out "${OUT}" \\
  --inflate-out "${SUB_DIR}/inflated" \\
  --device cpu \\
  --batch-size 16 \\
  --num-threads 6
"""
    _atomic_write_text(path, content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_next_if_resumed(path: Path, receipt: dict[str, Any]) -> None:
    best = receipt["best_rung_by_projected_s_with_od2_pose_credit"]
    projection = best["projection_with_od2_stage2_pose_credit"]
    md = f"""# OD6 NEXT_IF_RESUMED - 2026-08-05

1. Treat `{best['name']}` as the current decoder-legal targeter incumbent only on the stated n32 mask-domain projection: `S={projection['projected_s']:.9f}`, `{projection['packet_bytes_n600_projected']}` projected n600 bytes, `{best['fidelity']['totals']['retained_fix_count']}` retained fixes.
2. If extending this lane, first close the receiver-shape gap: prove the targeter operates over a decoder-computable candidate universe and measure collateral from false positives, not only retained OD2 positives.
3. Do not fire the queued scorer gate until the scorer slot is free, the lane is claimed, and a receiver-closed staged submission exists.
4. If the counted proxy is retained, replace the n32-fitted global block prior with a byte-priced n600-trained or strictly held-out prior before treating its projected bytes as production evidence.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
"""
    _atomic_write_text(path, md)


def _write_markdown(path: Path, receipt: dict[str, Any]) -> None:
    best = receipt["best_rung_by_projected_s_with_od2_pose_credit"]
    projection = best["projection_with_od2_stage2_pose_credit"]
    current_delta = projection["projected_s"] - od4.CURRENT_OWN_S
    current_delta_abs = abs(current_delta)
    od5_tax = receipt["legality_tax_vs_od5_scorer_native"]
    below_live = "below" if current_delta < 0.0 else "above"
    receipt_json = Path(receipt["receipt_json_path"])
    next_path = Path(receipt["next_if_resumed_path"])
    md = f"""# OD6 decoder-legal context targeter receipt - 2026-08-05

Status: `SCORER_FREE_DECODER_LEGAL_CONTEXT_PRICED / NO FRONTIER MOVE`.

Axis: `[macOS-CPU cache-derived advisory / scorer-free mask-domain replay]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`, `scorer_forwards_run=0`.

## Answer First

OD6 rebuilt OD5's bucket-context targeter with decoder-legal feature columns only. The best row is `{best['name']}`: `{best['packet']['best_coder']['bytes']}` exact n32 OD5 packet bytes, `{projection['packet_bytes_n600_projected']}` projected n600 packet bytes, `{best['fidelity']['totals']['retained_fix_count']}` retained fixes out of the fixed denominator `6,177`, eta `{best['fidelity']['totals']['eta_receiver']:.6f}`, and projected `S = {projection['projected_s']:.9f}` with OD2 pose credit.

That is `{below_live}` the live own line by `{current_delta_abs:.9f}` (`delta_vs_live={current_delta:.9f}`). It is not a score and not promotion-eligible: OD6 ran no scorer, no `upstream/evaluate.py`, no receiver-closed RGB/inflate candidate, and no full n600 dispatch.

Against OD5's scorer-native selected index (`S=0.743783052`, `78,010` projected bytes, eta `0.553900`, ratio `0.530`), the best decoder-legal row has `delta_S={od5_tax['delta_s']:.9f}`, `delta_bytes={od5_tax['delta_projected_bytes']}`, `eta_lost={od5_tax['eta_lost']:.6f}`, and `retained_fixes_lost={od5_tax['retained_fixes_lost']}`. OD4's sparse baseline ratio is printed as `0.711`; OD5's scorer-native ratio is printed as `0.530`.

## Price Table

{_table_markdown(receipt)}

## Decode-Time Compute Path

The decoder-side path represented by the legal feature table is: inflate qo1 base frames; resize the last frame of each pair to the 384x512 scorer lattice with generic bilinear interpolation; compute RGB/YUV bins, local luma gradients, local contrast, and chroma-delta bins; optionally compute generator/hybrid coverage bits from shipped PE3 hybrid75 coordinates; optionally add generic row/column/horizon bins; optionally read the counted block-prior table; hash the feature tuple to a shipped bucket table; and keep candidate corrections whose bucket probability crosses the shipped threshold. No scorer weights, scorer forwards, GT frames, GT margin, Fisher table, or cached scorer-native distances are read by this decoder path.

## RECALL EVIDENCE

| source | recalled fact | plan impact |
|---|---|---|
| `.omx/tmp/codex_runs/_common_contract.md`, `.omx/tmp/codex_runs/od6_prompt.md`, `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `.omx/state/main_hot_state.md` | OD6 owns no scorer slot, must avoid protected files/staged index, must use serializer hashes, and live own line is `S=0.7539807296911207 @ 357,836 B`. | Built a scorer-free repricing receipt only and queued the scorer gate. |
| `.omx/research/ddm_od5_20260805/OD5_GENERATOR_PACKET_RECEIPT.md` | OD5's best projection was `S=0.743783052`, ratio `0.530`, but its context row mixed scorer-native cached fields. | Used OD5 as the illegal-feature baseline to quantify the decoder-legality tax. |
| `.omx/research/ddm_od4_20260805/OD4_WEAK_PACKET_RECEIPT.md` | OD4 sparse packet ratio was about `0.711` with `S=0.761509399`. | Printed OD4 as the weak-packet where-tax baseline. |
| `.omx/research/ddm_st2_20260805/ST2_RECEIPT_20260805.md` | The strong ST2 selected table used scorer-native margin/Fisher/head-distance context and is not receiver-legal as a decoder feature. | Excluded ST2 features and rebuilt hashes from qo1 RGB/generator/geometry/counted-proxy columns. |
| `.omx/research/operator_directive_per_edge_optimality_criteria_20260805.md`, `.omx/research/ddm_qo1_repair_stream_optimal_form_20260804.md`, `/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/C_qo1_pairbit_n600_eval_receipt.json` | Seg-first surgical packet work must price legal bytes; qo1 is the current measured own-vehicle base under SSD custody. | Used qo1 decoded frames as the base-frame legal context and kept pose credit inherited from OD2 only. |
| bounded `rg` and canonical-equation recall over `.omx/research`, `.omx/state`, `docs`, `src/tac`, `experiments`, and `tools` | Existing receiver-closed/worldsheet/address-law evidence emphasizes counted bytes and candidate-universe legality; no existing OD6 decoder-legal targeter table was found. | Added the candidate-universe and false-positive collateral caveat instead of claiming receiver closure. |

## SHA Table

| artifact | bytes | sha256 |
|---|---:|---|
| `{receipt['source_files']['od6_script']['path']}` | {receipt['source_files']['od6_script']['bytes']} | `{receipt['source_files']['od6_script']['sha256']}` |
| `{receipt['source_files']['od2_json']['path']}` | {receipt['source_files']['od2_json']['bytes']} | `{receipt['source_files']['od2_json']['sha256']}` |
| `{receipt['source_files']['pair_selection']['path']}` | {receipt['source_files']['pair_selection']['bytes']} | `{receipt['source_files']['pair_selection']['sha256']}` |
| `{receipt['source_files']['pe1_receipt']['path']}` | {receipt['source_files']['pe1_receipt']['bytes']} | `{receipt['source_files']['pe1_receipt']['sha256']}` |
| `{receipt['source_files']['pe3_receipt']['path']}` | {receipt['source_files']['pe3_receipt']['bytes']} | `{receipt['source_files']['pe3_receipt']['sha256']}` |
| `{receipt['source_files']['qo1_archive']['path']}` | {receipt['source_files']['qo1_archive']['bytes']} | `{receipt['source_files']['qo1_archive']['sha256']}` |
| `{receipt['source_files']['qo1_packet_bin']['path']}` | {receipt['source_files']['qo1_packet_bin']['bytes']} | `{receipt['source_files']['qo1_packet_bin']['sha256']}` |
| `{receipt['source_files']['qo1_inflated_raw']['path']}` | {receipt['source_files']['qo1_inflated_raw']['bytes']} | `{receipt['source_files']['qo1_inflated_raw']['sha256']}` |
| `{receipt_json}` | {receipt_json.stat().st_size if receipt_json.exists() else 0} | `{_sha256_file(receipt_json) if receipt_json.exists() else 'pending'}` |
| `{next_path}` | {next_path.stat().st_size if next_path.exists() else 0} | `{_sha256_file(next_path) if next_path.exists() else 'pending'}` |

## NEXT_IF_RESUMED

See `{next_path}`. The first continuation gate is receiver-shape closure: the targeter must operate over a decoder-computable candidate universe with measured false-positive collateral, not only retained OD2 positives.

## Boundaries

- No `upstream/evaluate.py`, SegNet, PoseNet, full n600 scorer job, contest-CPU, or contest-CUDA run.
- OD2 pose credit is inherited for the same-row projection only and was not remeasured by OD6.
- n600 bytes are projected component sums: PE3 measured hybrid75 bytes plus exact counted table/proxy bytes. They are not exact archive bytes.
- Feature columns are decoder-legal, but the table is fit on n32 labels; this is not a receiver-closed candidate-universe proof.
- The counted proxy variant prices a global 24x32 block-prior table once; a per-pair or n600-trained proxy would need its own exact byte price.
- This does not move the frontier.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
"""
    _atomic_write_text(path, md)


def _source_file_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od2-json", type=Path, default=od5.DEFAULT_OD2_JSON)
    parser.add_argument("--pair-selection", type=Path, default=od5.DEFAULT_PAIR_SELECTION)
    parser.add_argument("--argmax-cache", type=Path, default=od5.DEFAULT_ARGMAX_CACHE)
    parser.add_argument("--gt-cache", type=Path, default=od5.DEFAULT_GT_CACHE)
    parser.add_argument("--pe1-receipt", type=Path, default=od5.DEFAULT_PE1_RECEIPT)
    parser.add_argument("--pe3-receipt", type=Path, default=od5.DEFAULT_PE3_RECEIPT)
    parser.add_argument("--g4-recurrence", type=Path, default=od5.DEFAULT_G4_RECURRENCE)
    parser.add_argument("--qo1-sub-dir", type=Path, default=DEFAULT_QO1_SUB_DIR)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--rmax", type=int, default=5)
    parser.add_argument("--proxy-block", type=int, default=16)
    parser.add_argument("--bucket-counts", default=DEFAULT_BUCKET_COUNTS)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--depth-y1", type=float, default=190.0)
    parser.add_argument("--depth-y2", type=float, default=230.0)
    args = parser.parse_args(argv)

    args.research_dir.mkdir(parents=True, exist_ok=True)
    storage = _storage_preflight(args.ssd_dir, required_free_bytes=512 * 1024 * 1024)
    if not storage["ok"]:
        raise OD6Error(f"SSD storage preflight failed: {storage}")
    run_id = args.run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_ssd = args.ssd_dir / run_id
    run_ssd.mkdir(parents=True, exist_ok=False)

    od2_json = _load_json(args.od2_json)
    pair_selection = _load_json(args.pair_selection)
    pe1_receipt = _load_json(args.pe1_receipt)
    pe3_receipt = _load_json(args.pe3_receipt)
    rows = od2_json.get("rows")
    if not isinstance(rows, list) or not rows:
        raise OD6Error("OD2 JSON has no rows")
    od2_rows_by_pair = {int(row["pair"]): row for row in rows}
    pairs = [int(pair) for pair in pair_selection["pairs"]]
    missing = [pair for pair in pairs if pair not in od2_rows_by_pair]
    if missing:
        raise OD6Error(f"OD2 JSON missing selected pairs: {missing}")

    current_argmax = np.load(args.argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gt_argmax = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")
    lstars = pe1.open_stored_npy_memmap(args.gt_cache, "lstars")
    if not np.array_equal(np.asarray(lstars[pairs], dtype=np.uint8), np.asarray(gt_argmax[pairs], dtype=np.uint8)):
        raise OD6Error("PE1 GT cache and OD2 argmax cache differ on selected n32 pairs")

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
    if target_denominators["retained_fix_denominator"] != EXPECTED_RETAINED_FIX_DENOMINATOR:
        raise OD6Error(f"unexpected OD2 retained-fix denominator: {target_denominators}")

    base_features, qo1_inflated_meta = _load_qo1_base_frame_features(args.qo1_sub_dir, pairs)
    universe, positives, point_meta = _build_point_sets(
        target_pairs=target_pairs,
        pairs=pairs,
        current_argmax=current_argmax,
        gt_argmax=gt_argmax,
    )

    components, extraction = pe1.extract_components(lstars, current_argmax)
    surfaces = od5._build_generator_surfaces(
        components=components,
        lstars=lstars,
        current=current_argmax,
        pe1_receipt=pe1_receipt,
        g4_recurrence=args.g4_recurrence,
        depth_y1=args.depth_y1,
        depth_y2=args.depth_y2,
    )
    hybrid75: pe3.HybridBuild = surfaces["hybrid75"]
    hybrid_knee: pe3.HybridBuild = surfaces["hybrid_knee"]
    generator_rep: pe1.RepresentationBuild = surfaces["generator_rep"]
    generator_masks = od5._masks_by_pair(components=components, component_masks=generator_rep.component_masks, pairs=pairs)
    hybrid75_masks = od5._masks_by_pair(components=components, component_masks=hybrid75.component_masks, pairs=pairs)
    hybrid_knee_masks = od5._masks_by_pair(components=components, component_masks=hybrid_knee.component_masks, pairs=pairs)
    hybrid75_selected = od5._selected_flats_from_mask(target_pairs, hybrid75_masks, pairs)
    hybrid75_subset_raw = od5._subset_frame_records(hybrid75.frame_records, pairs)
    pe3_hybrid75_projected = int(pe3_receipt["hybrid_75kb"]["section_bytes"])

    proxy = _fit_proxy_payload(points=universe, block=args.proxy_block, smoothing=args.smoothing, seed=args.seed)
    modes = [
        "base_rgb_local",
        "base_rgb_generator",
        "base_rgb_generator_geometry",
        "base_rgb_generator_geometry_proxy",
    ]
    bucket_counts = _parse_int_csv(args.bucket_counts)
    rungs: list[dict[str, Any]] = []
    for mode in modes:
        for bucket_count in bucket_counts:
            fit = _fit_decoder_legal_table(
                mode=mode,
                bucket_count=bucket_count,
                universe=universe,
                positives=positives,
                base_features=base_features,
                generator_masks=generator_masks,
                hybrid75_masks=hybrid75_masks,
                hybrid_knee_masks=hybrid_knee_masks,
                proxy_qlogits=proxy.qlogits if mode.endswith("_proxy") else None,
                proxy_block=args.proxy_block,
                seed=args.seed + bucket_count + len(mode),
                smoothing=args.smoothing,
            )
            combined = od5._union_selected(pairs, hybrid75_selected, fit.selected_by_pair)
            sections = [od4.OD5Section("pe3_hybrid75_coords_n32", hybrid75_subset_raw)]
            projected_bytes = pe3_hybrid75_projected + fit.best.bytes
            if mode.endswith("_proxy"):
                sections.append(od4.OD5Section("od6_counted_block_proxy_24x32", proxy.raw))
                projected_bytes += proxy.best.bytes
            sections.append(od4.OD5Section(f"od6_decoder_legal_table_{mode}_{bucket_count}", fit.raw))
            rung = od5._rung_receipt(
                name=f"{mode}_b{bucket_count}",
                sections=sections,
                selected_flats=combined,
                target_pairs=target_pairs,
                od2_rows_by_pair=od2_rows_by_pair,
                pairs=pairs,
                current_argmax=current_argmax,
                gt_argmax=gt_argmax,
                projected_n600_packet_bytes=projected_bytes,
                projection_scope=(
                    "PE3 measured hybrid75 n600 section bytes plus exact counted OD6 table bytes"
                    + (" plus exact counted shared block-proxy bytes" if mode.endswith("_proxy") else "")
                    + "; scorer-free n32 mask-domain replay; not receiver-closed RGB/inflate"
                ),
                ssd_dir=run_ssd,
            )
            rung["feature_surface"] = {
                "mode": mode,
                "bucket_count": bucket_count,
                "decoder_legal": True,
                "table": fit.train_meta,
                "selected_context_fix_points_before_hybrid_union": fit.selected_fix_count,
                "counted_proxy": proxy.best.as_json() if mode.endswith("_proxy") else None,
            }
            rungs.append(rung)

    best = min(rungs, key=lambda row: row["projection_with_od2_stage2_pose_credit"]["projected_s"])
    best_projection = best["projection_with_od2_stage2_pose_credit"]
    best_totals = best["fidelity"]["totals"]
    legality_tax = {
        "baseline": "OD5 scorer-native selected index",
        "baseline_projected_s": OD5_SCORER_NATIVE_BASELINE_S,
        "baseline_projected_bytes": OD5_SCORER_NATIVE_BASELINE_BYTES,
        "baseline_eta": OD5_SCORER_NATIVE_BASELINE_ETA,
        "baseline_retained_fixes": OD5_SCORER_NATIVE_BASELINE_RETAINED_FIXES,
        "delta_s": float(best_projection["projected_s"] - OD5_SCORER_NATIVE_BASELINE_S),
        "delta_projected_bytes": int(best_projection["packet_bytes_n600_projected"] - OD5_SCORER_NATIVE_BASELINE_BYTES),
        "eta_lost": float(OD5_SCORER_NATIVE_BASELINE_ETA - best_totals["eta_receiver"]),
        "retained_fixes_lost": int(OD5_SCORER_NATIVE_BASELINE_RETAINED_FIXES - best_totals["retained_fix_count"]),
    }
    no_decoder_legal_surface_beats_live = not any(
        bool(row["projection_with_od2_stage2_pose_credit"]["beats_current_own_line"]) for row in rungs
    )

    receipt_json_path = args.research_dir / "ddm_od6_decoder_legal_receipt.json"
    md_path = args.research_dir / "OD6_DECODER_LEGAL_RECEIPT.md"
    next_path = args.research_dir / "NEXT_IF_RESUMED.md"
    gate_script = args.research_dir / "OD6_SCORER_GATE_FIRE_ORDER.sh"
    script_path = Path(__file__).resolve()
    receipt: dict[str, Any] = {
        "schema": "ddm_od6_decoder_legal_context_receipt.v1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "axis": "[macOS-CPU cache-derived advisory / scorer-free mask-domain replay]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "scorer_forwards_run": 0,
        "frontier_moved": False,
        "current_own_vehicle_frontier": {
            "S": od4.CURRENT_OWN_S,
            "archive_bytes": od4.CURRENT_OWN_BYTES,
            "axis": od4.CURRENT_OWN_AXIS,
        },
        "denominators": {
            "selection_mode": pair_selection["selection"]["pair_selection"],
            "selection_seed": pair_selection["seed"],
            "n_pairs": len(pairs),
            "population_pairs": od4.N_PAIRS,
            "height": od4.SEG_H,
            "width": od4.SEG_W,
            "rate_denominator_bytes": od4.RATE_DENOMINATOR_BYTES,
            **target_denominators,
            **point_meta,
        },
        "storage_preflight": storage,
        "run_ssd_dir": str(run_ssd),
        "source_files": {
            "od6_script": _source_file_entry(script_path),
            "od2_json": _source_file_entry(args.od2_json),
            "pair_selection": _source_file_entry(args.pair_selection),
            "pe1_receipt": _source_file_entry(args.pe1_receipt),
            "pe3_receipt": _source_file_entry(args.pe3_receipt),
            "argmax_cache": {
                "path": str(args.argmax_cache),
                "cx1_sha256": _sha256_file(args.argmax_cache / "cx1_argmax_n600.npy"),
                "gt_sha256": _sha256_file(args.argmax_cache / "gt_argmax_n600.npy"),
            },
            "g4_recurrence": _source_file_entry(args.g4_recurrence),
            "qo1_archive": _source_file_entry(args.qo1_sub_dir / "archive.zip"),
            "qo1_packet_bin": _source_file_entry(args.qo1_sub_dir / "archive" / "0.bin"),
            "qo1_inflated_raw": qo1_inflated_meta,
        },
        "component_extraction": extraction,
        "generator_surface": {
            "generator_count": int(surfaces["generator_count"]),
            "hybrid75_n32_subset_raw_bytes": len(hybrid75_subset_raw),
            "hybrid75_projected_n600_section_bytes": pe3_hybrid75_projected,
            "depth_meta": surfaces["depth_meta"],
        },
        "counted_proxy": {
            "header": proxy.header,
            "raw_bytes": len(proxy.raw),
            "raw_sha256": _sha256_bytes(proxy.raw),
            "best": proxy.best.as_json(),
            "coder_race": [row.as_json() for row in proxy.coder_rows],
            "projection_policy": "global 24x32 shared block prior; exact counted bytes are added once, not linearly scaled",
        },
        "feature_modes": modes,
        "bucket_counts": bucket_counts,
        "rungs": rungs,
        "best_rung_by_projected_s_with_od2_pose_credit": best,
        "baseline_comparison": {
            "od5_scorer_native_selected": {
                "projected_s_with_pose_credit": OD5_SCORER_NATIVE_BASELINE_S,
                "projected_n600_bytes": OD5_SCORER_NATIVE_BASELINE_BYTES,
                "retained_fixes": OD5_SCORER_NATIVE_BASELINE_RETAINED_FIXES,
                "eta": OD5_SCORER_NATIVE_BASELINE_ETA,
                "rate_cost_over_seg_win": OD5_SCORER_NATIVE_BASELINE_RATIO,
                "legality": "not decoder-legal due scorer-native cached feature columns",
            },
            "od4_sparse_packet": {
                "projected_s_with_pose_credit": OD4_SPARSE_BASELINE_S,
                "rate_cost_over_seg_win": OD4_SPARSE_BASELINE_RATIO,
            },
            "current_own_s": od4.CURRENT_OWN_S,
        },
        "legality_tax_vs_od5_scorer_native": legality_tax,
        "no_decoder_legal_surface_beats_live": no_decoder_legal_surface_beats_live,
        "decode_time_compute_path": [
            "inflate qo1 base frames",
            "read pair last frame 2*p+1",
            "resize to 384x512 with generic bilinear interpolation",
            "compute RGB/YUV/local-gradient/local-contrast/chroma bins",
            "add shipped generator/hybrid coverage bits for generator modes",
            "add generic geometry bins for geometry modes",
            "read counted 24x32 block-prior proxy for proxy mode",
            "hash feature tuple into shipped bucket table",
            "keep decoder-computable candidate corrections whose bucket probability crosses the shipped threshold",
        ],
        "recall_evidence": [
            {
                "source": "_common_contract, od6_prompt, CLAUDE, AGENTS, PROGRAM, main_hot_state",
                "finding": "OD6 is scorer-free, protected files/staged index are off limits, and the live own line is fixed at 0.7539807296911207.",
                "plan_impact": "No scorer was run; exact bytes are packet/table bytes only; gate is queued.",
            },
            {
                "source": "OD5_GENERATOR_PACKET_RECEIPT",
                "finding": "OD5's 0.530 ratio row uses scorer-native cached features.",
                "plan_impact": "OD6 uses it only as a baseline and excludes its feature columns.",
            },
            {
                "source": "OD4_WEAK_PACKET_RECEIPT",
                "finding": "OD4 sparse per-flip packet ratio was 0.711.",
                "plan_impact": "OD6 prints OD4 as the where-tax baseline.",
            },
            {
                "source": "ST2 receipt",
                "finding": "ST2 selected row is strong but scorer-native.",
                "plan_impact": "Rebuilt a table from qo1 RGB/generator/geometry/counted-proxy columns.",
            },
            {
                "source": "operator addenda 4-7, qo1/SB1 receipts, bounded repository recall",
                "finding": "Current work is seg-first and byte-priced; qo1 is the measured base under SSD custody.",
                "plan_impact": "OD6 derives base-frame context from qo1 decoded frames and keeps receiver-closure caveats explicit.",
            },
        ],
        "queued_scorer_gate_script": str(gate_script),
        "receipt_json_path": str(receipt_json_path),
        "receipt_md_path": str(md_path),
        "next_if_resumed_path": str(next_path),
        "boundaries": [
            "No upstream/evaluate.py run",
            "No SegNet/PoseNet scorer job",
            "No full n600 dispatch",
            "No receiver-closed RGB/inflate archive",
            "No scorer-native feature columns in OD6 tables",
            "n600 bytes are projected component sums, not exact archive bytes",
            "candidate-universe collateral remains unmeasured",
        ],
        "frontier_line": "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.",
    }
    _atomic_write_json(receipt_json_path, receipt)
    _write_next_if_resumed(next_path, receipt)
    _write_gate_script(gate_script)
    _write_markdown(md_path, receipt)

    print("Baselines: OD5 scorer-native ratio=0.530 S=0.743783052; OD4 sparse ratio=0.711 S=0.761509399")
    print(_table_markdown(receipt))
    print(f"best={best['name']} S={best_projection['projected_s']:.9f}")
    print(f"receipt_json={receipt_json_path}")
    print(f"receipt_md={md_path}")
    print(f"next_if_resumed={next_path}")
    print(f"gate_script={gate_script}")
    print("S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
