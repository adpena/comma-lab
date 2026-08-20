#!/usr/bin/env python3
"""PR130 CPR1 pose-carrier representation ladder.

This experiment is deliberately advisory.  It changes only the counted pose
carrier inside the reproduced PR130 archive, round-trips every candidate
through the pinned CPR1 codec and inflate parser, and optionally measures a
seeded stratified subset with the frozen CPU-torch scorers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import lzma
import math
import os
import platform
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from scipy.fft import dctn, idctn

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
UPSTREAM = REPO / "upstream"
PR130 = Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo")
PR130_CODE = PR130 / "code"
DEFAULT_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip")
DEFAULT_TARGET_CACHE = PR130 / "artifacts/caches/gt_cache_600_official_ada.pt.xz"  # GT_LINEAGE_OK: decompressed bytes are registry-classified DALI_NVDEC sha256 382d7dfe38b37c0c
DEFAULT_OUT = Path("/Volumes/VertigoDataTier/pact/ddm_pk2_20260809")
RESEARCH_OUT = REPO / ".omx/research/ddm_pk2_pose_carrier_representation_20260809"
EXPECTED_ARCHIVE_SHA = "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
EXPECTED_CODEC_SHA = "d2f14402374b4e622b7f981d736389fb04f0ca0165180e4c75f3a32ffe996bed"
EXPECTED_INFLATE_SHA = "335369c9b3b295707f1790feb0b5b7ae288338fae350056cc4bb03aaa18f0c9e"
EXPECTED_SEG_SHA = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"

N = 600
DIM = 12
BASIS_SHAPE = (DIM, 3, 24, 32)
CAMERA_H = 874
CAMERA_W = 1164
REFERENCE_BYTES = 37_545_489
AXIS = "[macOS-CPU advisory]"
SCORE_CLAIM = False
XZ_FILTERS = [
    {
        "id": lzma.FILTER_LZMA2,
        "dict_size": 1 << 16,
        "lc": 0,
        "lp": 1,
        "pb": 0,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 273,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]
# Experiment-owned receiver overlay.  CPR1 remains delegated byte-for-byte to
# the pinned intake runtime; PK2R only carries representation experiments that
# CPR1 cannot express (predictors, factor packets, transforms, per-plane scales).
OVERLAY_MAGIC = b"PK2R"
OVERLAY_HEADER = struct.Struct("<4sBBBBII")
OVERLAY_VERSION = 1
BASIS_CPR1 = 0
BASIS_HAAR = 1
BASIS_DCT = 2
BASIS_LOW_RANK = 3
BASIS_PER_PLANE = 4
COEFF_CPR1 = 0
COEFF_PREDICTOR = 1
COEFF_LOW_RANK_RESIDUAL = 2

PRED_FIRST = 1
PRED_SECOND = 2
PRED_AR = 3
PRED_LINEAR_KNOT = 4
PRED_CUBIC_KNOT = 5


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return sha256_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def array_sha256(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def utcnow() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checked_free_space(path: Path, required_bytes: int) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(path)
    free = int(stats.f_bavail * stats.f_frsize)
    result = {
        "path": str(path),
        "free_bytes": free,
        "required_free_bytes": int(required_bytes),
        "passed": free >= required_bytes,
        "measured_at_utc": utcnow(),
    }
    if not result["passed"]:
        raise RuntimeError(f"storage preflight failed: {free} < {required_bytes} at {path}")
    return result


def validate_out_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == Path("/tmp") or Path("/tmp") in resolved.parents:
        raise ValueError("persisted experiment output may not use /tmp")
    if resolved == Path("/private/tmp") or Path("/private/tmp") in resolved.parents:
        raise ValueError("persisted experiment output may not use /tmp")
    ssd_root = Path("/Volumes/VertigoDataTier/pact")
    if resolved == ssd_root or ssd_root not in resolved.parents:
        raise ValueError("persisted experiment output must use an arm-specific VertigoDataTier directory")
    return resolved


def setup_imports() -> tuple[Any, Any]:
    for path in (PR130_CODE, SRC, UPSTREAM):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    carrier_codec = importlib.import_module("carrier_codec")
    inflate = importlib.import_module("inflate")
    return carrier_codec, inflate


@dataclass(frozen=True)
class Bundle:
    archive_blob: bytes
    member: bytes
    models_compressed: bytes
    models_raw: bytes
    semantic: bytes
    carrier: bytes
    hpac: bytes
    tokens: bytes


def extract_bundle(archive_path: Path) -> Bundle:
    archive_blob = archive_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_blob), "r") as archive:
        names = archive.namelist()
        if names != ["p"]:
            raise ValueError(f"archive must contain only p, got {names}")
        info = archive.getinfo("p")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("p must be stored, not ZIP-compressed")
        member = archive.read("p")
    if len(member) < 4:
        raise ValueError("truncated p member")
    model_size = struct.unpack_from("<I", member)[0]
    models_compressed = member[4 : 4 + model_size]
    tokens = member[4 + model_size :]
    models_raw = lzma.decompress(models_compressed)
    semantic_size, carrier_size = struct.unpack_from("<II", models_raw)
    semantic_start = 8
    carrier_start = semantic_start + semantic_size
    hpac_start = carrier_start + carrier_size
    if hpac_start > len(models_raw):
        raise ValueError("model section lengths exceed raw model bundle")
    return Bundle(
        archive_blob=archive_blob,
        member=member,
        models_compressed=models_compressed,
        models_raw=models_raw,
        semantic=models_raw[semantic_start:carrier_start],
        carrier=models_raw[carrier_start:hpac_start],
        hpac=models_raw[hpac_start:],
        tokens=tokens,
    )


def deterministic_archive(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def replace_carrier(bundle: Bundle, carrier: bytes) -> bytes:
    raw = struct.pack("<II", len(bundle.semantic), len(carrier)) + bundle.semantic + carrier + bundle.hpac
    models = lzma.compress(raw, format=lzma.FORMAT_XZ, filters=XZ_FILTERS)
    member = struct.pack("<I", len(models)) + models + bundle.tokens
    return deterministic_archive(member)


def decode_absolute_codes(encoded: np.ndarray) -> np.ndarray:
    encoded = np.asarray(encoded, dtype=np.int64)
    delta = (encoded >> 1) ^ -(encoded & 1)
    unsigned = np.cumsum(delta, axis=0) & 0xFFF
    return np.where(unsigned >= 0x800, unsigned - 0x1000, unsigned).astype(np.int32)


def encode_absolute_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int64)
    if codes.shape != (N, DIM):
        raise ValueError(f"absolute coefficient shape must be {(N, DIM)}")
    if codes.min() < -2047 or codes.max() > 2047:
        raise ValueError("absolute codes exceed symmetric int12 deployment range")
    unsigned = codes & 0xFFF
    previous = np.zeros_like(unsigned)
    previous[1:] = unsigned[:-1]
    delta_u = (unsigned - previous) & 0xFFF
    delta = np.where(delta_u >= 0x800, delta_u - 0x1000, delta_u)
    return (((delta << 1) ^ (delta >> 63)) & 0xFFF).astype(np.int32)


def signed_mod12(value: np.ndarray | int) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    return (((array + 2048) & 0xFFF) - 2048).astype(np.int32)


def split_cpr1_components(blob: bytes, codec: Any) -> tuple[bytes, bytes]:
    """Split CPR1 without recoding either entropy stream.

    The four bytes of PK2R framing overhead are counted later.  This split is
    what lets an A-row retain the exact incumbent basis stream and a B-row
    retain the exact incumbent coefficient stream.
    """
    magic, basis_bits, coefficient_bits = codec.HEADER.unpack_from(blob)
    if magic != b"CPR1":
        raise ValueError("component split requires CPR1")
    cursor = codec.HEADER.size
    scale_bytes = DIM * 4
    basis_scales = blob[cursor : cursor + scale_bytes]
    cursor += scale_bytes
    coefficient_scales = blob[cursor : cursor + scale_bytes]
    cursor += scale_bytes
    lengths = blob[cursor : cursor + codec.ALPHABET_SIZE]
    cursor += codec.ALPHABET_SIZE
    ks = blob[cursor : cursor + DIM]
    cursor += DIM
    basis_payload_bytes = (basis_bits + 7) // 8
    coefficient_payload_bytes = (coefficient_bits + 7) // 8
    basis_payload = blob[cursor : cursor + basis_payload_bytes]
    cursor += basis_payload_bytes
    coefficient_payload = blob[cursor : cursor + coefficient_payload_bytes]
    cursor += coefficient_payload_bytes
    if cursor != len(blob):
        raise ValueError("CPR1 component split did not consume the carrier")
    return (
        struct.pack("<I", basis_bits) + basis_scales + lengths + basis_payload,
        struct.pack("<I", coefficient_bits) + coefficient_scales + ks + coefficient_payload,
    )


def decode_cpr1_basis_component(component: bytes, codec: Any) -> np.ndarray:
    bit_count = struct.unpack_from("<I", component)[0]
    cursor = 4
    scales = np.frombuffer(component[cursor : cursor + DIM * 4], dtype="<f4").copy()
    cursor += DIM * 4
    lengths = np.frombuffer(component[cursor : cursor + codec.ALPHABET_SIZE], dtype=np.uint8).copy()
    cursor += codec.ALPHABET_SIZE
    payload = component[cursor:]
    unsigned = codec._decode_huffman(lengths, payload, bit_count, math.prod(BASIS_SHAPE))
    codes = codec._unzigzag_unsigned(unsigned, codec.BASIS_BITS)
    return codes.reshape(BASIS_SHAPE).astype(np.float32) * scales[:, None, None, None]


def decode_cpr1_coefficient_component(component: bytes, codec: Any) -> np.ndarray:
    bit_count = struct.unpack_from("<I", component)[0]
    cursor = 4
    scales = np.frombuffer(component[cursor : cursor + DIM * 4], dtype="<f4").copy()
    cursor += DIM * 4
    ks = np.frombuffer(component[cursor : cursor + DIM], dtype=np.uint8).copy()
    cursor += DIM
    encoded = codec._decode_rice(ks, component[cursor:], bit_count, N, DIM)
    return decode_absolute_codes(encoded).astype(np.float32) * scales[None]


def encode_overlay(
    basis_mode: int,
    basis_component: bytes,
    coefficient_mode: int,
    coefficient_component: bytes,
) -> bytes:
    return b"".join(
        [
            OVERLAY_HEADER.pack(
                OVERLAY_MAGIC,
                OVERLAY_VERSION,
                int(basis_mode),
                int(coefficient_mode),
                0,
                len(basis_component),
                len(coefficient_component),
            ),
            basis_component,
            coefficient_component,
        ]
    )


def _haar_forward_axis(value: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(np.asarray(value, dtype=np.int32), axis, -1)
    if moved.shape[-1] % 2:
        raise ValueError("integer Haar requires an even axis")
    even = moved[..., 0::2]
    odd = moved[..., 1::2]
    detail = odd - even
    smooth = even + np.floor_divide(detail, 2)
    return np.moveaxis(np.concatenate([smooth, detail], axis=-1), -1, axis)


def _haar_inverse_axis(value: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(np.asarray(value, dtype=np.int32), axis, -1)
    half = moved.shape[-1] // 2
    smooth = moved[..., :half]
    detail = moved[..., half:]
    even = smooth - np.floor_divide(detail, 2)
    odd = detail + even
    restored = np.empty_like(moved)
    restored[..., 0::2] = even
    restored[..., 1::2] = odd
    return np.moveaxis(restored, -1, axis)


def encode_haar_basis_component(basis_codes: np.ndarray, basis_scales: np.ndarray) -> bytes:
    transformed = _haar_forward_axis(_haar_forward_axis(basis_codes, -1), -2).astype("<i2")
    return np.asarray(basis_scales, dtype="<f4").tobytes() + zlib.compress(transformed.tobytes(), level=9)


def decode_haar_basis_component(component: bytes) -> np.ndarray:
    scales = np.frombuffer(component[: DIM * 4], dtype="<f4").copy()
    raw = zlib.decompress(component[DIM * 4 :])
    expected = math.prod(BASIS_SHAPE) * 2
    if len(raw) != expected:
        raise ValueError("Haar basis payload length mismatch")
    transformed = np.frombuffer(raw, dtype="<i2").reshape(BASIS_SHAPE)
    codes = _haar_inverse_axis(_haar_inverse_axis(transformed, -2), -1)
    return codes.astype(np.float32) * scales[:, None, None, None]


def encode_dct_basis_component(basis: np.ndarray, keep: int, bits: int) -> tuple[bytes, np.ndarray]:
    if not 1 <= keep <= 768 or not 3 <= bits <= 15:
        raise ValueError("invalid DCT packet parameters")
    transformed = dctn(basis.astype(np.float64), axes=(-2, -1), norm="ortho")
    flat = transformed.reshape(DIM * 3, -1)
    limit = (1 << (bits - 1)) - 1
    indices = np.empty((DIM * 3, keep), dtype="<u2")
    codes = np.empty((DIM * 3, keep), dtype="<i2")
    scales = np.empty(DIM * 3, dtype="<f4")
    reconstructed = np.zeros_like(flat)
    for plane in range(DIM * 3):
        chosen = np.argpartition(np.abs(flat[plane]), -keep)[-keep:]
        chosen.sort()
        scale = max(float(np.max(np.abs(flat[plane, chosen]))) / limit, 1e-12)
        quantized = np.rint(flat[plane, chosen] / scale).clip(-limit, limit)
        indices[plane] = chosen.astype(np.uint16)
        codes[plane] = quantized.astype(np.int16)
        scales[plane] = scale
        reconstructed[plane, chosen] = quantized * scale
    raw = scales.tobytes() + indices.tobytes() + codes.tobytes()
    component = struct.pack("<HBB", keep, bits, 0) + zlib.compress(raw, level=9)
    restored = idctn(reconstructed.reshape(BASIS_SHAPE), axes=(-2, -1), norm="ortho").astype(np.float32)
    return component, restored


def decode_dct_basis_component(component: bytes) -> np.ndarray:
    keep, bits, reserved = struct.unpack_from("<HBB", component)
    if reserved or not 1 <= keep <= 768 or not 3 <= bits <= 15:
        raise ValueError("invalid DCT basis component header")
    raw = zlib.decompress(component[4:])
    plane_count = DIM * 3
    scale_bytes = plane_count * 4
    index_bytes = plane_count * keep * 2
    code_bytes = index_bytes
    if len(raw) != scale_bytes + index_bytes + code_bytes:
        raise ValueError("DCT basis component length mismatch")
    scales = np.frombuffer(raw[:scale_bytes], dtype="<f4")
    indices = np.frombuffer(raw[scale_bytes : scale_bytes + index_bytes], dtype="<u2").reshape(plane_count, keep)
    codes = np.frombuffer(raw[scale_bytes + index_bytes :], dtype="<i2").reshape(plane_count, keep)
    transformed = np.zeros((plane_count, 768), dtype=np.float64)
    for plane in range(plane_count):
        if np.unique(indices[plane]).size != keep or np.any(indices[plane] >= 768):
            raise ValueError("DCT basis indices are invalid")
        transformed[plane, indices[plane]] = codes[plane] * scales[plane]
    return idctn(transformed.reshape(BASIS_SHAPE), axes=(-2, -1), norm="ortho").astype(np.float32)


def _quantize_factor_columns(value: np.ndarray, bits: int, axis: int) -> tuple[np.ndarray, np.ndarray]:
    limit = (1 << (bits - 1)) - 1
    maximum = np.max(np.abs(value.astype(np.float64)), axis=axis)
    scales = np.maximum(maximum / limit, 1e-12).astype("<f4")
    shape = [1] * value.ndim
    for index, size in enumerate(scales.shape):
        shape[index if axis != 0 else index + 1] = size
    if axis == 0:
        divisor = scales[None, :]
    elif axis == 1:
        divisor = scales[:, None]
    else:
        raise ValueError("unsupported factor quantizer axis")
    codes = np.rint(value / divisor).clip(-limit, limit).astype("<i2")
    return codes, scales


def encode_low_rank_basis_component(basis: np.ndarray, rank: int, bits: int) -> tuple[bytes, np.ndarray]:
    matrix = basis.reshape(DIM * 3, -1).astype(np.float64)
    means = matrix.mean(axis=1).astype("<f4")
    u, singular, vt = np.linalg.svd(matrix - means[:, None], full_matrices=False)
    left = u[:, :rank] * singular[:rank]
    right = vt[:rank]
    left_codes, left_scales = _quantize_factor_columns(left, bits, axis=0)
    right_codes, right_scales = _quantize_factor_columns(right, bits, axis=1)
    restored = (
        np.einsum(
            "ir,rj->ij",
            left_codes.astype(np.float64) * left_scales.astype(np.float64)[None],
            right_codes.astype(np.float64) * right_scales.astype(np.float64)[:, None],
        )
        + means.astype(np.float64)[:, None]
    )
    raw = b"".join(
        [
            means.tobytes(),
            left_scales.tobytes(),
            right_scales.tobytes(),
            left_codes.tobytes(),
            right_codes.tobytes(),
        ]
    )
    return (
        struct.pack("<BBH", rank, bits, 0) + zlib.compress(raw, level=9),
        restored.reshape(BASIS_SHAPE).astype(np.float32),
    )


def decode_low_rank_basis_component(component: bytes) -> np.ndarray:
    rank, bits, reserved = struct.unpack_from("<BBH", component)
    if reserved or not 1 <= rank <= 36 or not 3 <= bits <= 15:
        raise ValueError("invalid low-rank basis header")
    raw = zlib.decompress(component[4:])
    cursor = 0
    means = np.frombuffer(raw[cursor : cursor + 36 * 4], dtype="<f4")
    cursor += 36 * 4
    left_scales = np.frombuffer(raw[cursor : cursor + rank * 4], dtype="<f4")
    cursor += rank * 4
    right_scales = np.frombuffer(raw[cursor : cursor + rank * 4], dtype="<f4")
    cursor += rank * 4
    left_count = 36 * rank
    left = np.frombuffer(raw[cursor : cursor + left_count * 2], dtype="<i2").reshape(36, rank)
    cursor += left_count * 2
    right_count = rank * 768
    right = np.frombuffer(raw[cursor:], dtype="<i2")
    if right.size != right_count:
        raise ValueError("low-rank basis factor length mismatch")
    right = right.reshape(rank, 768)
    restored = np.einsum(
        "ir,rj->ij",
        left.astype(np.float64) * left_scales.astype(np.float64)[None],
        right.astype(np.float64) * right_scales.astype(np.float64)[:, None],
    )
    restored += means.astype(np.float64)[:, None]
    return restored.reshape(BASIS_SHAPE).astype(np.float32)


def encode_per_plane_basis_component(basis: np.ndarray, bits: int, percentile: float) -> tuple[bytes, np.ndarray]:
    limit = (1 << (bits - 1)) - 1
    planes = basis.reshape(36, 768).astype(np.float64)
    thresholds = np.percentile(np.abs(planes), percentile, axis=1)
    scales = np.maximum(thresholds / limit, 1e-12).astype("<f4")
    codes = np.rint(planes / scales[:, None]).clip(-limit, limit).astype(np.int8)
    component = (
        struct.pack("<BBHf", bits, 36, 0, float(percentile))
        + scales.tobytes()
        + zlib.compress(codes.tobytes(), level=9)
    )
    return component, (codes.astype(np.float32) * scales[:, None]).reshape(BASIS_SHAPE)


def decode_per_plane_basis_component(component: bytes) -> np.ndarray:
    bits, planes, reserved, percentile = struct.unpack_from("<BBHf", component)
    if reserved or planes != 36 or not 2 <= bits <= 5 or not 0 < percentile <= 100:
        raise ValueError("invalid per-plane basis header")
    scales = np.frombuffer(component[8 : 8 + 36 * 4], dtype="<f4").copy()
    raw = zlib.decompress(component[8 + 36 * 4 :])
    if len(raw) != math.prod(BASIS_SHAPE):
        raise ValueError("per-plane basis code length mismatch")
    codes = np.frombuffer(raw, dtype=np.int8).reshape(36, 768)
    limit = (1 << (bits - 1)) - 1
    if codes.min() < -limit or codes.max() > limit:
        raise ValueError("per-plane basis code exceeds declared precision")
    return (codes.astype(np.float32) * scales[:, None]).reshape(BASIS_SHAPE)


def _round_divide(numerator: np.ndarray, denominator: int) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.int64)
    return np.where(
        numerator >= 0,
        (numerator + denominator // 2) // denominator,
        -((-numerator + denominator // 2) // denominator),
    )


def predictor_parameters(codes: np.ndarray, kind: int, order: int, stride: int) -> bytes:
    if kind == PRED_AR:
        x = codes.astype(np.float64)
        rows = []
        for t in range(order, N):
            rows.append(np.concatenate([[1.0], x[t - order : t][::-1, 0]]))
        # Fit independently because each dimension has a different trajectory.
        coefficient = np.empty((DIM, order), dtype="<i2")
        intercept = np.empty(DIM, dtype="<i4")
        for dimension in range(DIM):
            design = np.column_stack(
                [
                    np.ones(N - order),
                    *[x[order - lag - 1 : N - lag - 1, dimension] for lag in range(order)],
                ]
            )
            fit, *_ = np.linalg.lstsq(design, x[order:, dimension], rcond=None)
            intercept[dimension] = int(np.rint(fit[0] * 256.0))
            coefficient[dimension] = np.rint(fit[1:] * 256.0).clip(-32768, 32767).astype(np.int16)
        return intercept.tobytes() + coefficient.tobytes()
    if kind in (PRED_LINEAR_KNOT, PRED_CUBIC_KNOT):
        positions = list(range(0, N, stride))
        if positions[-1] != N - 1:
            positions.append(N - 1)
        return np.asarray(codes[positions], dtype="<i2").tobytes()
    return b""


def predict_codes(
    reconstructed: np.ndarray,
    kind: int,
    order: int,
    stride: int,
    parameters: bytes,
) -> np.ndarray:
    prediction = np.zeros((N, DIM), dtype=np.int32)
    if kind == PRED_FIRST:
        prediction[1:] = reconstructed[:-1]
    elif kind == PRED_SECOND:
        prediction[1] = reconstructed[0]
        prediction[2:] = signed_mod12(2 * reconstructed[1:-1].astype(np.int64) - reconstructed[:-2].astype(np.int64))
    elif kind == PRED_AR:
        intercept_bytes = DIM * 4
        intercept = np.frombuffer(parameters[:intercept_bytes], dtype="<i4")
        coefficient = np.frombuffer(parameters[intercept_bytes:], dtype="<i2")
        if coefficient.size != DIM * order:
            raise ValueError("AR parameter length mismatch")
        coefficient = coefficient.reshape(DIM, order)
        for t in range(N):
            if t < order:
                prediction[t] = reconstructed[t - 1] if t else 0
                continue
            history = reconstructed[t - order : t][::-1].T.astype(np.int64)
            numerator = intercept.astype(np.int64) + np.sum(coefficient.astype(np.int64) * history, axis=1)
            prediction[t] = signed_mod12(_round_divide(numerator, 256))
    elif kind in (PRED_LINEAR_KNOT, PRED_CUBIC_KNOT):
        positions = list(range(0, N, stride))
        if positions[-1] != N - 1:
            positions.append(N - 1)
        knots = np.frombuffer(parameters, dtype="<i2")
        if knots.size != len(positions) * DIM:
            raise ValueError("knot parameter length mismatch")
        knots = knots.reshape(len(positions), DIM).astype(np.float64)
        for t in range(N):
            right = int(np.searchsorted(positions, t, side="right"))
            left = max(0, right - 1)
            right = min(right, len(positions) - 1)
            if left == right:
                value = knots[left]
            else:
                alpha = (t - positions[left]) / (positions[right] - positions[left])
                if kind == PRED_LINEAR_KNOT:
                    value = (1.0 - alpha) * knots[left] + alpha * knots[right]
                else:
                    p0 = knots[max(0, left - 1)]
                    p1 = knots[left]
                    p2 = knots[right]
                    p3 = knots[min(len(knots) - 1, right + 1)]
                    a2 = alpha * alpha
                    a3 = a2 * alpha
                    value = 0.5 * (
                        2 * p1
                        + (-p0 + p2) * alpha
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * a2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * a3
                    )
            prediction[t] = signed_mod12(np.rint(value).astype(np.int64))
    else:
        raise ValueError("unknown coefficient predictor")
    return prediction


def encode_predictor_coefficient_component(
    codes: np.ndarray,
    scales: np.ndarray,
    kind: int,
    order: int = 0,
    stride: int = 0,
) -> tuple[bytes, np.ndarray]:
    parameters = predictor_parameters(codes, kind, order, stride)
    reconstructed = np.zeros_like(codes, dtype=np.int32)
    residual = np.zeros_like(codes, dtype=np.int32)
    for t in range(N):
        prediction = predict_codes(reconstructed, kind, order, stride, parameters)[t]
        residual[t] = signed_mod12(codes[t].astype(np.int64) - prediction)
        reconstructed[t] = signed_mod12(prediction.astype(np.int64) + residual[t])
    if not np.array_equal(reconstructed, codes):
        raise AssertionError("predictor residual failed exact int12 reconstruction")
    zigzag = (((residual.astype(np.int64) << 1) ^ (residual.astype(np.int64) >> 63)) & 0xFFF).astype("<u2")
    component = b"".join(
        [
            struct.pack("<BBHI", kind, order, stride, len(parameters)),
            np.asarray(scales, dtype="<f4").tobytes(),
            parameters,
            zlib.compress(zigzag.tobytes(), level=9),
        ]
    )
    return component, reconstructed.astype(np.float32) * np.asarray(scales)[None]


def decode_predictor_coefficient_component(component: bytes) -> np.ndarray:
    kind, order, stride, parameter_bytes = struct.unpack_from("<BBHI", component)
    cursor = 8
    scales = np.frombuffer(component[cursor : cursor + DIM * 4], dtype="<f4").copy()
    cursor += DIM * 4
    parameters = component[cursor : cursor + parameter_bytes]
    cursor += parameter_bytes
    raw = zlib.decompress(component[cursor:])
    if len(raw) != N * DIM * 2:
        raise ValueError("predictor residual length mismatch")
    encoded = np.frombuffer(raw, dtype="<u2").reshape(N, DIM).astype(np.int64)
    residual = ((encoded >> 1) ^ -(encoded & 1)).astype(np.int32)
    reconstructed = np.zeros((N, DIM), dtype=np.int32)
    for t in range(N):
        prediction = predict_codes(reconstructed, kind, order, stride, parameters)[t]
        reconstructed[t] = signed_mod12(prediction.astype(np.int64) + residual[t])
    return reconstructed.astype(np.float32) * scales[None]


def encode_low_rank_coefficient_component(
    codes: np.ndarray,
    scales: np.ndarray,
    rank: int,
    factor_bits: int,
    residual_step: int,
) -> tuple[bytes, np.ndarray]:
    value = codes.astype(np.float64)
    means = np.rint(value.mean(axis=0)).astype("<i2")
    u, singular, vt = np.linalg.svd(value - means[None], full_matrices=False)
    scores = u[:, :rank] * singular[:rank]
    loadings = vt[:rank]
    score_codes, score_scales = _quantize_factor_columns(scores, factor_bits, axis=0)
    loading_codes, loading_scales = _quantize_factor_columns(loadings, factor_bits, axis=1)
    prediction = np.rint(
        np.einsum(
            "nr,rd->nd",
            score_codes.astype(np.float64) * score_scales.astype(np.float64)[None],
            loading_codes.astype(np.float64) * loading_scales.astype(np.float64)[:, None],
        )
        + means.astype(np.float64)[None]
    ).astype(np.int32)
    residual = np.rint((codes.astype(np.int64) - prediction) / residual_step)
    residual = residual.clip(-32768, 32767).astype("<i2")
    reconstructed_codes = np.clip(
        prediction.astype(np.int64) + residual.astype(np.int64) * residual_step,
        -2047,
        2047,
    ).astype(np.int32)
    raw = b"".join(
        [
            np.asarray(scales, dtype="<f4").tobytes(),
            means.tobytes(),
            score_scales.tobytes(),
            loading_scales.tobytes(),
            score_codes.tobytes(),
            loading_codes.tobytes(),
            residual.tobytes(),
        ]
    )
    component = struct.pack("<BBHI", rank, factor_bits, residual_step, len(raw)) + zlib.compress(raw, level=9)
    return component, reconstructed_codes.astype(np.float32) * np.asarray(scales)[None]


def decode_low_rank_coefficient_component(component: bytes) -> np.ndarray:
    rank, factor_bits, residual_step, raw_bytes = struct.unpack_from("<BBHI", component)
    if not 4 <= rank <= 11 or not 3 <= factor_bits <= 15 or residual_step < 1:
        raise ValueError("invalid low-rank coefficient header")
    raw = zlib.decompress(component[8:])
    if len(raw) != raw_bytes:
        raise ValueError("low-rank coefficient payload length mismatch")
    cursor = 0
    scales = np.frombuffer(raw[cursor : cursor + DIM * 4], dtype="<f4").copy()
    cursor += DIM * 4
    means = np.frombuffer(raw[cursor : cursor + DIM * 2], dtype="<i2").copy()
    cursor += DIM * 2
    score_scales = np.frombuffer(raw[cursor : cursor + rank * 4], dtype="<f4").copy()
    cursor += rank * 4
    loading_scales = np.frombuffer(raw[cursor : cursor + rank * 4], dtype="<f4").copy()
    cursor += rank * 4
    score_count = N * rank
    score_codes = np.frombuffer(raw[cursor : cursor + score_count * 2], dtype="<i2").reshape(N, rank)
    cursor += score_count * 2
    loading_count = rank * DIM
    loading_codes = np.frombuffer(raw[cursor : cursor + loading_count * 2], dtype="<i2").reshape(rank, DIM)
    cursor += loading_count * 2
    residual = np.frombuffer(raw[cursor:], dtype="<i2")
    if residual.size != N * DIM:
        raise ValueError("low-rank coefficient residual length mismatch")
    residual = residual.reshape(N, DIM)
    prediction = np.rint(
        np.einsum(
            "nr,rd->nd",
            score_codes.astype(np.float64) * score_scales.astype(np.float64)[None],
            loading_codes.astype(np.float64) * loading_scales.astype(np.float64)[:, None],
        )
        + means.astype(np.float64)[None]
    ).astype(np.int32)
    reconstructed = np.clip(
        prediction.astype(np.int64) + residual.astype(np.int64) * residual_step,
        -2047,
        2047,
    ).astype(np.int32)
    return reconstructed.astype(np.float32) * scales[None]


def decode_overlay_carrier(blob: bytes, codec: Any) -> tuple[np.ndarray, np.ndarray]:
    if len(blob) < OVERLAY_HEADER.size:
        raise ValueError("truncated PK2R carrier")
    magic, version, basis_mode, coefficient_mode, reserved, basis_bytes, coefficient_bytes = OVERLAY_HEADER.unpack_from(
        blob
    )
    if magic != OVERLAY_MAGIC or version != OVERLAY_VERSION or reserved:
        raise ValueError("unsupported PK2R carrier header")
    expected = OVERLAY_HEADER.size + basis_bytes + coefficient_bytes
    if len(blob) != expected:
        raise ValueError("PK2R carrier length mismatch")
    basis_component = blob[OVERLAY_HEADER.size : OVERLAY_HEADER.size + basis_bytes]
    coefficient_component = blob[OVERLAY_HEADER.size + basis_bytes :]
    basis_decoders = {
        BASIS_CPR1: lambda value: decode_cpr1_basis_component(value, codec),
        BASIS_HAAR: decode_haar_basis_component,
        BASIS_DCT: decode_dct_basis_component,
        BASIS_LOW_RANK: decode_low_rank_basis_component,
        BASIS_PER_PLANE: decode_per_plane_basis_component,
    }
    coefficient_decoders = {
        COEFF_CPR1: lambda value: decode_cpr1_coefficient_component(value, codec),
        COEFF_PREDICTOR: decode_predictor_coefficient_component,
        COEFF_LOW_RANK_RESIDUAL: decode_low_rank_coefficient_component,
    }
    try:
        basis = basis_decoders[basis_mode](basis_component)
        coefficients = coefficient_decoders[coefficient_mode](coefficient_component)
    except KeyError as error:
        raise ValueError("unknown PK2R component mode") from error
    if basis.shape != BASIS_SHAPE or coefficients.shape != (N, DIM):
        raise ValueError("PK2R decoder produced the wrong declared arrays")
    return basis.astype(np.float32), coefficients.astype(np.float32)


def carrier_component_pair(blob: bytes, codec: Any) -> tuple[int, bytes, int, bytes]:
    if blob[:4] == b"CPR1":
        basis_component, coefficient_component = split_cpr1_components(blob, codec)
        return BASIS_CPR1, basis_component, COEFF_CPR1, coefficient_component
    if blob[:4] != OVERLAY_MAGIC or len(blob) < OVERLAY_HEADER.size:
        raise ValueError("unsupported carrier for component composition")
    magic, version, basis_mode, coefficient_mode, reserved, basis_bytes, coefficient_bytes = OVERLAY_HEADER.unpack_from(
        blob
    )
    if magic != OVERLAY_MAGIC or version != OVERLAY_VERSION or reserved:
        raise ValueError("invalid overlay carrier for composition")
    if len(blob) != OVERLAY_HEADER.size + basis_bytes + coefficient_bytes:
        raise ValueError("overlay composition source length mismatch")
    basis_component = blob[OVERLAY_HEADER.size : OVERLAY_HEADER.size + basis_bytes]
    coefficient_component = blob[OVERLAY_HEADER.size + basis_bytes :]
    return basis_mode, basis_component, coefficient_mode, coefficient_component


def quantize_basis(
    basis: np.ndarray, bits: int, percentile: float = 100.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bits < 2 or bits > 5:
        raise ValueError("effective basis precision must be 2..5")
    limit = (1 << (bits - 1)) - 1
    threshold = np.percentile(np.abs(basis.astype(np.float64)), percentile, axis=(1, 2, 3))
    scales = np.maximum(threshold / limit, 1e-12).astype("<f4")
    codes = np.rint(basis / scales[:, None, None, None]).clip(-limit, limit).astype(np.int32)
    restored = codes.astype(np.float64) * scales[:, None, None, None]
    return restored.astype(np.float32), codes, scales


def quantize_coefficients(
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scales = np.maximum(
        np.max(np.abs(coefficients.astype(np.float64)), axis=0) / 2047.0,
        1e-12,
    ).astype("<f4")
    codes = np.rint(coefficients / scales[None]).clip(-2047, 2047).astype(np.int32)
    restored = codes.astype(np.float64) * scales[None]
    return restored.astype(np.float32), codes, scales


def carrier_product_mse(
    basis_a: np.ndarray,
    coeff_a: np.ndarray,
    basis_b: np.ndarray,
    coeff_b: np.ndarray,
) -> float:
    basis_a_2d = np.ascontiguousarray(basis_a.reshape(DIM, -1), dtype=np.float64)
    basis_b_2d = np.ascontiguousarray(basis_b.reshape(DIM, -1), dtype=np.float64)
    coeff_a_2d = np.ascontiguousarray(coeff_a, dtype=np.float64)
    coeff_b_2d = np.ascontiguousarray(coeff_b, dtype=np.float64)
    a = np.einsum("nd,dp->np", coeff_a_2d, basis_a_2d, optimize=False)
    b = np.einsum("nd,dp->np", coeff_b_2d, basis_b_2d, optimize=False)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite reconstructed carrier in MSE diagnostic")
    delta = a - b
    return float(np.einsum("ij,ij->", delta, delta) / delta.size)


def joint_delta_s(
    *,
    baseline_d_pose: float,
    candidate_d_pose: float,
    baseline_d_seg: float,
    candidate_d_seg: float,
    baseline_archive_bytes: int,
    candidate_archive_bytes: int,
) -> dict[str, float]:
    seg = 100.0 * (candidate_d_seg - baseline_d_seg)
    pose = math.sqrt(10.0 * candidate_d_pose) - math.sqrt(10.0 * baseline_d_pose)
    rate = 25.0 * (candidate_archive_bytes - baseline_archive_bytes) / REFERENCE_BYTES
    return {"seg": seg, "pose": pose, "rate": rate, "total": seg + pose + rate}


def measured_dimension_order(rows: list[dict[str, Any]]) -> list[int]:
    singles = [row for row in rows if row["name"].startswith("capacity_drop_dim")]
    if len(singles) != DIM:
        raise ValueError(f"expected {DIM} measured single-dimension rows")
    ranked: list[tuple[float, int]] = []
    for row in singles:
        saved = -int(row["delta_archive_bytes"])
        if saved <= 0:
            raise ValueError(f"{row['name']} did not save archive bytes")
        distortion_cost = float(row["delta_s_seg"]) + float(row["delta_s_pose"])
        dimension = int(row["name"].rsplit("dim", 1)[1])
        ranked.append((distortion_cost / saved, dimension))
    return [dimension for _, dimension in sorted(ranked)]


def coefficient_diagnostics(coefficients: np.ndarray) -> dict[str, Any]:
    value = coefficients.astype(np.float64)
    centered = value - value.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.cumsum(singular**2) / np.sum(singular**2)
    lag1 = []
    lag2 = []
    low_frequency_energy = []
    for dimension in range(DIM):
        column = value[:, dimension]
        lag1.append(float(np.corrcoef(column[:-1], column[1:])[0, 1]))
        lag2.append(float(np.corrcoef(column[:-2], column[2:])[0, 1]))
        spectrum = np.abs(np.fft.rfft(column - column.mean())) ** 2
        split = max(1, len(spectrum) // 10)
        low_frequency_energy.append(float(spectrum[:split].sum() / spectrum.sum()))
    return {
        "lag1_per_dimension": lag1,
        "lag2_per_dimension": lag2,
        "low_frequency_first_decile_energy_per_dimension": low_frequency_energy,
        "cross_dimension_singular_values": singular.tolist(),
        "cross_dimension_cumulative_energy": energy.tolist(),
    }


def basis_diagnostics(basis: np.ndarray) -> dict[str, Any]:
    value = basis.astype(np.float64)
    transformed = dctn(value, axes=(-2, -1), norm="ortho")
    squared = transformed.reshape(DIM * 3, -1) ** 2
    ordered = np.sort(squared, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) / np.sum(ordered, axis=1, keepdims=True)
    plane_matrix = value.reshape(DIM * 3, -1)
    plane_matrix -= plane_matrix.mean(axis=1, keepdims=True)
    singular = np.linalg.svd(plane_matrix, compute_uv=False)
    plane_energy = np.cumsum(singular**2) / np.sum(singular**2)
    horizontal_lag1 = []
    vertical_lag1 = []
    for plane in value.reshape(DIM * 3, value.shape[-2], value.shape[-1]):
        horizontal_lag1.append(float(np.corrcoef(plane[:, :-1].reshape(-1), plane[:, 1:].reshape(-1))[0, 1]))
        vertical_lag1.append(float(np.corrcoef(plane[:-1].reshape(-1), plane[1:].reshape(-1))[0, 1]))
    counts = {}
    for level in (0.90, 0.95, 0.99):
        needed = np.argmax(cumulative >= level, axis=1) + 1
        counts[str(level)] = {
            "median": float(np.median(needed)),
            "max": int(needed.max()),
            "min": int(needed.min()),
        }
    return {
        "dct_coefficients_for_energy": counts,
        "horizontal_lag1_per_plane": horizontal_lag1,
        "vertical_lag1_per_plane": vertical_lag1,
        "horizontal_lag1_median": float(np.median(horizontal_lag1)),
        "vertical_lag1_median": float(np.median(vertical_lag1)),
        "cross_plane_singular_values": singular.tolist(),
        "cross_plane_cumulative_energy": plane_energy.tolist(),
    }


def dct_reconstruct(basis: np.ndarray, keep_fraction: float) -> np.ndarray:
    transformed = dctn(basis.astype(np.float64), axes=(-2, -1), norm="ortho")
    output = np.zeros_like(transformed)
    flat = transformed.reshape(DIM * 3, -1)
    out_flat = output.reshape(DIM * 3, -1)
    keep = max(1, round(flat.shape[1] * keep_fraction))
    for plane in range(flat.shape[0]):
        indices = np.argpartition(np.abs(flat[plane]), -keep)[-keep:]
        out_flat[plane, indices] = flat[plane, indices]
    return idctn(output, axes=(-2, -1), norm="ortho").astype(np.float32)


def plane_rank_reconstruct(basis: np.ndarray, rank: int) -> np.ndarray:
    matrix = basis.reshape(DIM * 3, -1).astype(np.float64)
    means = matrix.mean(axis=1, keepdims=True)
    u, singular, vt = np.linalg.svd(matrix - means, full_matrices=False)
    left = u[:, :rank] * singular[:rank]
    restored = np.einsum("ir,rp->ip", left, vt[:rank], optimize=False) + means
    if not np.isfinite(restored).all():
        raise ValueError("non-finite plane-rank reconstruction")
    return restored.reshape(BASIS_SHAPE).astype(np.float32)


def coefficient_rank_reconstruct(coefficients: np.ndarray, rank: int) -> np.ndarray:
    value = coefficients.astype(np.float64)
    means = value.mean(axis=0, keepdims=True)
    u, singular, vt = np.linalg.svd(value - means, full_matrices=False)
    left = u[:, :rank] * singular[:rank]
    restored = np.einsum("nr,rd->nd", left, vt[:rank], optimize=False) + means
    if not np.isfinite(restored).all():
        raise ValueError("non-finite coefficient-rank reconstruction")
    return restored.astype(np.float32)


def orthogonal_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(DIM, DIM)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs[None]


def materialize_candidate(
    *,
    name: str,
    arm: str,
    scope: str,
    bundle: Bundle,
    codec: Any,
    inflate: Any,
    basis: np.ndarray,
    coefficients: np.ndarray,
    basis_codes: np.ndarray | None = None,
    basis_scales: np.ndarray | None = None,
    coefficient_codes: np.ndarray | None = None,
    coefficient_scales: np.ndarray | None = None,
    baseline_basis: np.ndarray,
    baseline_coefficients: np.ndarray,
    out_dir: Path,
    score: bool,
    notes: str = "",
) -> dict[str, Any]:
    if basis_codes is None or basis_scales is None:
        basis, basis_codes, basis_scales = quantize_basis(basis, 5)
    if coefficient_codes is None or coefficient_scales is None:
        coefficients, coefficient_codes, coefficient_scales = quantize_coefficients(coefficients)
    encoded = encode_absolute_codes(coefficient_codes)
    carrier = codec.encode_compact_carrier(
        basis_scales,
        basis_codes.reshape(-1),
        coefficient_scales,
        encoded,
    )
    decoded = codec.decode_compact_carrier(carrier, basis_count=math.prod(BASIS_SHAPE), frames=N, dimensions=DIM)
    decoded_basis = decoded[1].reshape(BASIS_SHAPE).astype(np.float32)
    decoded_basis *= decoded[0][:, None, None, None]
    decoded_codes = decode_absolute_codes(decoded[3])
    decoded_coefficients = decoded_codes.astype(np.float32) * decoded[2][None]
    if not np.array_equal(decoded[1].reshape(BASIS_SHAPE), basis_codes):
        raise RuntimeError(f"{name}: real codec changed basis codes")
    if not np.array_equal(decoded_codes, coefficient_codes):
        raise RuntimeError(f"{name}: real codec changed coefficient codes")
    semantic_pose = struct.pack("<II", len(bundle.semantic), len(carrier))
    semantic_pose += bundle.semantic + carrier
    _, inflate_basis, inflate_coeff = inflate.unpack_semantic_pose(semantic_pose)
    if not np.array_equal(inflate_basis.numpy(), decoded_basis):
        raise RuntimeError(f"{name}: real inflate changed basis")
    if not np.array_equal(inflate_coeff.numpy(), decoded_coefficients):
        raise RuntimeError(f"{name}: real inflate changed coefficients")
    archive_blob = replace_carrier(bundle, carrier)
    candidate_dir = out_dir / "candidates" / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    archive_path = candidate_dir / "archive.zip"
    archive_path.write_bytes(archive_blob)
    parsed = extract_bundle(archive_path)
    if parsed.carrier != carrier or parsed.semantic != bundle.semantic:
        raise RuntimeError(f"{name}: candidate archive parse-back mismatch")
    return {
        "name": name,
        "arm": arm,
        "scope": scope,
        "score_requested": bool(score),
        "notes": notes,
        "carrier_bytes": len(carrier),
        "carrier_sha256": sha256_bytes(carrier),
        "archive_path": str(archive_path),
        "archive_bytes": len(archive_blob),
        "archive_sha256": sha256_bytes(archive_blob),
        "parse_back": {
            "real_carrier_codec": True,
            "real_inflate_unpack": True,
            "basis_array_sha256": array_sha256(decoded_basis),
            "coefficient_array_sha256": array_sha256(decoded_coefficients),
        },
        "carrier_product_mse_vs_baseline": carrier_product_mse(
            decoded_basis,
            decoded_coefficients,
            baseline_basis,
            baseline_coefficients,
        ),
    }


def materialize_overlay_candidate(
    *,
    name: str,
    arm: str,
    scope: str,
    bundle: Bundle,
    codec: Any,
    basis_mode: int,
    basis_component: bytes,
    coefficient_mode: int,
    coefficient_component: bytes,
    declared_basis: np.ndarray,
    declared_coefficients: np.ndarray,
    baseline_basis: np.ndarray,
    baseline_coefficients: np.ndarray,
    out_dir: Path,
    score: bool,
    notes: str = "",
) -> dict[str, Any]:
    carrier = encode_overlay(
        basis_mode,
        basis_component,
        coefficient_mode,
        coefficient_component,
    )
    decoded_basis, decoded_coefficients = decode_overlay_carrier(carrier, codec)
    if not np.array_equal(decoded_basis, np.asarray(declared_basis, dtype=np.float32)):
        raise RuntimeError(f"{name}: overlay decoder did not reconstruct declared basis")
    if not np.array_equal(decoded_coefficients, np.asarray(declared_coefficients, dtype=np.float32)):
        raise RuntimeError(f"{name}: overlay decoder did not reconstruct declared coefficients")
    archive_blob = replace_carrier(bundle, carrier)
    candidate_dir = out_dir / "candidates" / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    archive_path = candidate_dir / "archive.zip"
    archive_path.write_bytes(archive_blob)
    parsed = extract_bundle(archive_path)
    if parsed.carrier != carrier or parsed.semantic != bundle.semantic:
        raise RuntimeError(f"{name}: overlay archive parse-back mismatch")
    reparsed_basis, reparsed_coefficients = decode_overlay_carrier(parsed.carrier, codec)
    if not np.array_equal(reparsed_basis, decoded_basis) or not np.array_equal(
        reparsed_coefficients, decoded_coefficients
    ):
        raise RuntimeError(f"{name}: archive receiver parse-back changed arrays")
    return {
        "name": name,
        "arm": arm,
        "scope": scope,
        "score_requested": bool(score),
        "notes": notes,
        "carrier_bytes": len(carrier),
        "carrier_sha256": sha256_bytes(carrier),
        "archive_path": str(archive_path),
        "archive_bytes": len(archive_blob),
        "archive_sha256": sha256_bytes(archive_blob),
        "parse_back": {
            "real_carrier_codec_delegated_for_unchanged_components": True,
            "experiment_receiver_overlay": True,
            "receiver_magic": OVERLAY_MAGIC.decode("ascii"),
            "basis_mode": int(basis_mode),
            "coefficient_mode": int(coefficient_mode),
            "metadata_counted_inside_carrier": True,
            "basis_array_sha256": array_sha256(decoded_basis),
            "coefficient_array_sha256": array_sha256(decoded_coefficients),
        },
        "carrier_product_mse_vs_baseline": carrier_product_mse(
            decoded_basis,
            decoded_coefficients,
            baseline_basis,
            baseline_coefficients,
        ),
    }


def analyze(args: argparse.Namespace) -> int:
    out = validate_out_dir(args.out_dir)
    preflight = checked_free_space(out, args.required_free_bytes)
    runner_sha = sha256_file(Path(__file__))
    if args.resume_from.exists():
        try:
            prior_state = json.loads(args.resume_from.read_text(encoding="utf-8"))
            prior_path = Path(prior_state["analysis_receipt"])
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            outputs_valid = all(
                Path(row["archive_path"]).is_file()
                and Path(row["archive_path"]).stat().st_size == row["archive_bytes"]
                and sha256_file(Path(row["archive_path"])) == row["archive_sha256"]
                for row in prior["candidates"]
            )
            inputs_valid = (
                sha256_file(args.archive) == EXPECTED_ARCHIVE_SHA
                and sha256_file(PR130_CODE / "carrier_codec.py") == EXPECTED_CODEC_SHA
                and sha256_file(PR130_CODE / "inflate.py") == EXPECTED_INFLATE_SHA
                and prior.get("pins", {}).get("archive", {}).get("sha256") == EXPECTED_ARCHIVE_SHA
                and prior.get("pins", {}).get("carrier_codec", {}).get("sha256") == EXPECTED_CODEC_SHA
                and prior.get("pins", {}).get("inflate", {}).get("sha256") == EXPECTED_INFLATE_SHA
            )
            if (
                prior_state.get("analysis_complete")
                and prior.get("workspace", {}).get("runner_sha256") == runner_sha
                and outputs_valid
                and inputs_valid
            ):
                repaired_state = {
                    "schema": "ddm_pk2_pose_carrier_representation_state.v1",
                    "analysis_complete": True,
                    "analysis_receipt": str(prior_path),
                    "analysis_receipt_sha256": sha256_file(prior_path),
                    "score_complete": False,
                    "updated_at_utc": utcnow(),
                }
                atomic_json(args.resume_from, repaired_state)
                atomic_json(
                    out / "checkpoints" / "stage_analysis_complete.json",
                    repaired_state,
                )
                print(
                    json.dumps(
                        {
                            "stage": "analyze",
                            "status": "skipped_valid_resume",
                            "receipt": str(prior_path),
                            "receipt_sha256": sha256_file(prior_path),
                        },
                        indent=2,
                    )
                )
                return 0
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            pass
    archive_sha = sha256_file(args.archive)
    codec_sha = sha256_file(PR130_CODE / "carrier_codec.py")
    inflate_sha = sha256_file(PR130_CODE / "inflate.py")
    if archive_sha != EXPECTED_ARCHIVE_SHA:
        raise RuntimeError(f"archive pin mismatch: {archive_sha}")
    if codec_sha != EXPECTED_CODEC_SHA:
        raise RuntimeError(f"carrier codec pin mismatch: {codec_sha}")
    if inflate_sha != EXPECTED_INFLATE_SHA:
        raise RuntimeError(f"inflate pin mismatch: {inflate_sha}")
    codec, inflate = setup_imports()
    if codec.MAGIC != b"CPR1" or codec.HEADER.format != "<4sII":
        raise RuntimeError("pinned CPR1 schema does not match charter")
    if (inflate.CARRIER_DIM, inflate.CARRIER_H, inflate.CARRIER_W) != (12, 24, 32):
        raise RuntimeError("pinned inflate carrier dimensions do not match charter")
    bundle = extract_bundle(args.archive)
    decoded = codec.decode_compact_carrier(bundle.carrier, basis_count=math.prod(BASIS_SHAPE), frames=N, dimensions=DIM)
    baseline_reencoded = codec.encode_compact_carrier(*decoded)
    if baseline_reencoded != bundle.carrier:
        raise RuntimeError("baseline carrier did not byte-round-trip")
    basis_scales, basis_codes_flat, coefficient_scales, encoded = decoded
    basis_codes = basis_codes_flat.reshape(BASIS_SHAPE)
    coefficient_codes = decode_absolute_codes(encoded)
    basis = basis_codes.astype(np.float32) * basis_scales[:, None, None, None]
    coefficients = coefficient_codes.astype(np.float32) * coefficient_scales[None]
    baseline_basis_component, baseline_coefficient_component = split_cpr1_components(bundle.carrier, codec)
    if not np.array_equal(decode_cpr1_basis_component(baseline_basis_component, codec), basis) or not np.array_equal(
        decode_cpr1_coefficient_component(baseline_coefficient_component, codec),
        coefficients,
    ):
        raise RuntimeError("CPR1 component split changed the baseline arrays")
    magic, basis_bits, coefficient_bits = codec.HEADER.unpack_from(bundle.carrier)
    if magic != b"CPR1":
        raise RuntimeError("unexpected carrier magic")

    candidates: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        candidates.append(
            materialize_candidate(
                bundle=bundle,
                codec=codec,
                inflate=inflate,
                baseline_basis=basis,
                baseline_coefficients=coefficients,
                out_dir=out,
                **kwargs,
            )
        )

    def add_overlay(**kwargs: Any) -> None:
        candidates.append(
            materialize_overlay_candidate(
                bundle=bundle,
                codec=codec,
                baseline_basis=basis,
                baseline_coefficients=coefficients,
                out_dir=out,
                **kwargs,
            )
        )

    add(
        name="baseline_cpr1_int5",
        arm="control",
        scope="MEASURED_CONTROL",
        basis=basis,
        coefficients=coefficients,
        basis_codes=basis_codes,
        basis_scales=basis_scales,
        coefficient_codes=coefficient_codes,
        coefficient_scales=coefficient_scales,
        score=True,
        notes="Deployed signed-int5 basis and int12 coefficient control.",
    )
    baseline_candidate = candidates[-1]
    if baseline_candidate["archive_sha256"] != archive_sha:
        raise RuntimeError("deterministic baseline materialization changed archive bytes")

    # Exact reversible coefficient representations.  Predictor parameters,
    # knots, scales, and residuals are all inside the counted PK2R section.
    predictor_specs = [
        ("first", PRED_FIRST, 0, 0),
        ("second", PRED_SECOND, 0, 0),
        *[(f"ar{order}", PRED_AR, order, 0) for order in range(1, 5)],
        ("linear_knot30", PRED_LINEAR_KNOT, 0, 30),
        ("linear_knot60", PRED_LINEAR_KNOT, 0, 60),
        ("cubic_knot30", PRED_CUBIC_KNOT, 0, 30),
        ("cubic_knot60", PRED_CUBIC_KNOT, 0, 60),
    ]
    predictor_rows: list[dict[str, Any]] = []
    for label, kind, order, stride in predictor_specs:
        component, restored = encode_predictor_coefficient_component(
            coefficient_codes,
            coefficient_scales,
            kind,
            order=order,
            stride=stride,
        )
        add_overlay(
            name=f"coeff_predictor_{label}",
            arm="A_reversible_predictor",
            scope="MEASURED_EXACT_REPRESENTATION",
            basis_mode=BASIS_CPR1,
            basis_component=baseline_basis_component,
            coefficient_mode=COEFF_PREDICTOR,
            coefficient_component=component,
            declared_basis=basis,
            declared_coefficients=restored,
            score=False,
            notes=(
                "Exact modulo-int12 predictor residual; all predictor parameters "
                "and residual bytes are counted in PK2R. Array equality makes this "
                "a byte-only row, not a scorer row."
            ),
        )
        predictor_rows.append(candidates[-1])

    # Genuine low-rank + residual packets: ranks 4..11, multiple factor
    # quantizers, and both exact and lossy residual steps.  Unlike the toy SVD
    # brackets below, these packets store factors and a residual directly.
    low_rank_rows: list[dict[str, Any]] = []
    for rank in range(4, 12):
        for factor_bits in (6, 8, 10):
            for residual_step in (1, 4, 16):
                component, restored = encode_low_rank_coefficient_component(
                    coefficient_codes,
                    coefficient_scales,
                    rank,
                    factor_bits,
                    residual_step,
                )
                restored = decode_low_rank_coefficient_component(component)
                add_overlay(
                    name=(f"coeff_lowrank_r{rank:02d}_q{factor_bits:02d}_res{residual_step:02d}"),
                    arm="A_low_rank_plus_residual",
                    scope="MEASURED_OPTIMAL_FORM_PACKET",
                    basis_mode=BASIS_CPR1,
                    basis_component=baseline_basis_component,
                    coefficient_mode=COEFF_LOW_RANK_RESIDUAL,
                    coefficient_component=component,
                    declared_basis=basis,
                    declared_coefficients=restored,
                    score=False,
                    notes=(
                        "Directly stored quantized SVD factors plus counted residual; "
                        "residual_step=1 is exact, larger steps are lossy."
                    ),
                )
                candidates[-1]["packet_parameters"] = {
                    "rank": rank,
                    "factor_bits": factor_bits,
                    "residual_step": residual_step,
                }
                low_rank_rows.append(candidates[-1])

    # Reversible lifting/Haar packet: a real transform representation whose
    # exact decoded basis equals CPR1's deployed signed-int5 tensor.
    haar_component = encode_haar_basis_component(basis_codes, basis_scales)
    add_overlay(
        name="basis_haar_lifting_exact",
        arm="B_reversible_spatial_transform",
        scope="MEASURED_EXACT_REPRESENTATION",
        basis_mode=BASIS_HAAR,
        basis_component=haar_component,
        coefficient_mode=COEFF_CPR1,
        coefficient_component=baseline_coefficient_component,
        declared_basis=basis,
        declared_coefficients=coefficients,
        score=False,
        notes="Exact integer lifting transform plus counted entropy payload; byte-only because decoded arrays are identical.",
    )

    dct_packet_rows: list[dict[str, Any]] = []
    for keep in (64, 128, 256):
        for packet_bits in (8, 10):
            component, restored = encode_dct_basis_component(basis, keep, packet_bits)
            restored = decode_dct_basis_component(component)
            add_overlay(
                name=f"basis_dct_packet_k{keep:03d}_q{packet_bits:02d}",
                arm="B_spatial_DCT_packet",
                scope="MEASURED_OPTIMAL_FORM_PACKET",
                basis_mode=BASIS_DCT,
                basis_component=component,
                coefficient_mode=COEFF_CPR1,
                coefficient_component=baseline_coefficient_component,
                declared_basis=restored,
                declared_coefficients=coefficients,
                score=False,
                notes="Sparse per-plane DCT coefficients, indices, scales, and codes stored directly in PK2R.",
            )
            candidates[-1]["packet_parameters"] = {
                "keep_per_plane": keep,
                "coefficient_bits": packet_bits,
            }
            dct_packet_rows.append(candidates[-1])

    basis_low_rank_rows: list[dict[str, Any]] = []
    for rank in (6, 12, 18):
        for factor_bits in (8, 10):
            component, restored = encode_low_rank_basis_component(basis, rank, factor_bits)
            restored = decode_low_rank_basis_component(component)
            add_overlay(
                name=f"basis_lowrank_packet_r{rank:02d}_q{factor_bits:02d}",
                arm="B_cross_plane_low_rank_packet",
                scope="MEASURED_OPTIMAL_FORM_PACKET",
                basis_mode=BASIS_LOW_RANK,
                basis_component=component,
                coefficient_mode=COEFF_CPR1,
                coefficient_component=baseline_coefficient_component,
                declared_basis=restored,
                declared_coefficients=coefficients,
                score=False,
                notes="Directly stored cross-plane low-rank factors with counted per-factor scales.",
            )
            candidates[-1]["packet_parameters"] = {
                "rank": rank,
                "factor_bits": factor_bits,
            }
            basis_low_rank_rows.append(candidates[-1])

    per_plane_rows: list[dict[str, Any]] = []
    for bits, percentile in ((4, 100.0), (4, 99.5), (3, 100.0), (3, 99.0)):
        component, restored = encode_per_plane_basis_component(basis, bits, percentile)
        restored = decode_per_plane_basis_component(component)
        add_overlay(
            name=f"basis_perplane_int{bits}_p{str(percentile).replace('.', '_')}",
            arm="B_per_plane_precision",
            scope="MEASURED_FORMULATION",
            basis_mode=BASIS_PER_PLANE,
            basis_component=component,
            coefficient_mode=COEFF_CPR1,
            coefficient_component=baseline_coefficient_component,
            declared_basis=restored,
            declared_coefficients=coefficients,
            score=True,
            notes="All 36 per-plane scales are counted and parsed by PK2R.",
        )
        per_plane_rows.append(candidates[-1])

    for bits in (4, 3):
        for percentile in (100.0, 99.5) if bits == 4 else (100.0, 99.0):
            restored, codes, scales = quantize_basis(basis, bits, percentile)
            add(
                name=f"basis_int{bits}_p{str(percentile).replace('.', '_')}",
                arm="B_precision",
                scope="MEASURED_FORMULATION",
                basis=restored,
                coefficients=coefficients,
                basis_codes=codes,
                basis_scales=scales,
                coefficient_codes=coefficient_codes,
                coefficient_scales=coefficient_scales,
                score=True,
                notes="Per-dimension scale with percentile outlier clipping; unchanged CPR1 receiver.",
            )

    for keep in (0.75, 0.50, 0.25):
        reconstructed = dct_reconstruct(basis, keep)
        restored, codes, scales = quantize_basis(reconstructed, 5)
        add(
            name=f"basis_dct_keep{int(keep * 100):02d}",
            arm="B_spatial_DCT",
            scope="TOY_BRACKET_FULL_ARRAY_CPR1",
            basis=restored,
            coefficients=coefficients,
            basis_codes=codes,
            basis_scales=scales,
            coefficient_codes=coefficient_codes,
            coefficient_scales=coefficient_scales,
            score=True,
            notes="DCT projection is real; storing reconstructed full CPR1 arrays is not optimal transform-packet form.",
        )

    for rank in (12, 18, 24):
        reconstructed = plane_rank_reconstruct(basis, rank)
        restored, codes, scales = quantize_basis(reconstructed, 5)
        add(
            name=f"basis_plane_rank{rank:02d}",
            arm="B_cross_plane_rank",
            scope="TOY_BRACKET_FULL_ARRAY_CPR1",
            basis=restored,
            coefficients=coefficients,
            basis_codes=codes,
            basis_scales=scales,
            coefficient_codes=coefficient_codes,
            coefficient_scales=coefficient_scales,
            score=True,
            notes="Low-rank plane projection is real; factors are not stored directly.",
        )

    for rank in (4, 6, 8, 10, 11):
        reconstructed = coefficient_rank_reconstruct(coefficients, rank)
        restored, codes, scales = quantize_coefficients(reconstructed)
        add(
            name=f"coeff_rank{rank:02d}",
            arm="A_low_rank",
            scope="TOY_BRACKET_FULL_ARRAY_CPR1",
            basis=basis,
            coefficients=restored,
            basis_codes=basis_codes,
            basis_scales=basis_scales,
            coefficient_codes=codes,
            coefficient_scales=scales,
            score=rank in (8, 10, 11),
            notes="Genuine SVD projection plus residual loss, but full reconstructed coefficients are stored; not optimal factor-packet form.",
        )

    single_drop_proxy: list[tuple[float, int]] = []
    for dimension in range(DIM):
        dropped_basis = basis.copy()
        dropped_coefficients = coefficients.copy()
        dropped_basis[dimension] = 0
        dropped_coefficients[:, dimension] = 0
        row_name = f"capacity_drop_dim{dimension:02d}"
        add(
            name=row_name,
            arm="C_capacity",
            scope="MEASURED_EXISTING_CARRIER_RESPONSE",
            basis=dropped_basis,
            coefficients=dropped_coefficients,
            score=True,
            notes="Existing trained carrier response only; no smaller-carrier retraining.",
        )
        single_drop_proxy.append((candidates[-1]["carrier_product_mse_vs_baseline"], dimension))

    ordered_dimensions = [dimension for _, dimension in sorted(single_drop_proxy)]
    for count in (2, 3):
        dropped_basis = basis.copy()
        dropped_coefficients = coefficients.copy()
        for dimension in ordered_dimensions[:count]:
            dropped_basis[dimension] = 0
            dropped_coefficients[:, dimension] = 0
        add(
            name=f"capacity_drop_nested{count:02d}",
            arm="C_capacity",
            scope="MEASURED_EXISTING_CARRIER_RESPONSE",
            basis=dropped_basis,
            coefficients=dropped_coefficients,
            score=True,
            notes=f"Nested proxy ordering {ordered_dimensions[:count]}; scorer reorders by joint value after measurement.",
        )

    gauge_rows = []
    for offset in range(args.gauge_trials):
        rotation = orthogonal_rotation(args.seed + offset)
        rotated_coefficients = np.einsum(
            "nd,dk->nk",
            coefficients.astype(np.float64),
            rotation,
            optimize=False,
        )
        rotated_basis = np.einsum("ij,jchw->ichw", rotation.T, basis.astype(np.float64))
        restored_basis, codes_basis, scales_basis = quantize_basis(rotated_basis, 5)
        restored_coeff, codes_coeff, scales_coeff = quantize_coefficients(rotated_coefficients)
        encoded_coeff = encode_absolute_codes(codes_coeff)
        blob = codec.encode_compact_carrier(scales_basis, codes_basis.reshape(-1), scales_coeff, encoded_coeff)
        gauge_rows.append(
            {
                "seed": args.seed + offset,
                "carrier_bytes": len(blob),
                "carrier_product_mse_vs_baseline": carrier_product_mse(
                    restored_basis, restored_coeff, basis, coefficients
                ),
            }
        )
    admissible_gauge = sorted(
        gauge_rows,
        key=lambda row: (row["carrier_bytes"], row["carrier_product_mse_vs_baseline"]),
    )[0]
    rotation = orthogonal_rotation(admissible_gauge["seed"])
    rotated_coefficients = np.einsum(
        "nd,dk->nk",
        coefficients.astype(np.float64),
        rotation,
        optimize=False,
    )
    rotated_basis = np.einsum("ij,jchw->ichw", rotation.T, basis.astype(np.float64))
    restored_basis, codes_basis, scales_basis = quantize_basis(rotated_basis, 5)
    restored_coeff, codes_coeff, scales_coeff = quantize_coefficients(rotated_coefficients)
    add(
        name="gauge_rotation_best",
        arm="beyond_seed_gauge_rotation",
        scope="MEASURED_FORMULATION_SEEDED_SEARCH",
        basis=restored_basis,
        coefficients=restored_coeff,
        basis_codes=codes_basis,
        basis_scales=scales_basis,
        coefficient_codes=codes_coeff,
        coefficient_scales=scales_coeff,
        score=True,
        notes=f"Best of {args.gauge_trials} seeded orthogonal gauge rotations; seed {admissible_gauge['seed']}.",
    )

    # Pre-register non-additive compositions of useful coefficient and basis rows.
    for rank in (10, 11):
        coeff_reconstructed = coefficient_rank_reconstruct(coefficients, rank)
        coeff_restored, coeff_codes, coeff_scales = quantize_coefficients(coeff_reconstructed)
        basis_restored, basis_low_codes, basis_low_scales = quantize_basis(basis, 4, 100.0)
        add(
            name=f"compose_coeff_rank{rank:02d}_basis_int4",
            arm="A_plus_B_composed",
            scope="TOY_BRACKET_FULL_ARRAY_CPR1",
            basis=basis_restored,
            coefficients=coeff_restored,
            basis_codes=basis_low_codes,
            basis_scales=basis_low_scales,
            coefficient_codes=coeff_codes,
            coefficient_scales=coeff_scales,
            score=True,
            notes="Composed and materialized toy full-array control; direct factor-packet compositions are selected after measured A/B scoring.",
        )

    def mark_proxy_pareto_for_score(rows: list[dict[str, Any]], maximum: int) -> list[str]:
        frontier: list[dict[str, Any]] = []
        best_mse = float("inf")
        for row in sorted(
            rows,
            key=lambda value: (
                value["archive_bytes"],
                value["carrier_product_mse_vs_baseline"],
            ),
        ):
            mse = float(row["carrier_product_mse_vs_baseline"])
            if mse < best_mse:
                frontier.append(row)
                best_mse = mse
        if len(frontier) > maximum:
            indices = np.linspace(0, len(frontier) - 1, maximum, dtype=int)
            frontier = [frontier[index] for index in sorted(set(indices.tolist()))]
        for row in frontier:
            row["score_requested"] = True
            row["shortlist_reason"] = "byte/product-MSE Pareto frontier"
        return [row["name"] for row in frontier]

    packet_shortlists = {
        "A_low_rank_plus_residual": mark_proxy_pareto_for_score(low_rank_rows, 6),
        "B_spatial_DCT_packet": mark_proxy_pareto_for_score(dct_packet_rows, 3),
        "B_cross_plane_low_rank_packet": mark_proxy_pareto_for_score(basis_low_rank_rows, 3),
    }

    baseline_archive_bytes = next(row["archive_bytes"] for row in candidates if row["name"] == "baseline_cpr1_int5")
    baseline_carrier_bytes = len(bundle.carrier)
    for row in candidates:
        row["delta_archive_bytes"] = row["archive_bytes"] - baseline_archive_bytes
        row["delta_carrier_bytes"] = row["carrier_bytes"] - baseline_carrier_bytes

    receipt = {
        "schema": "ddm_pk2_pose_carrier_representation_analysis.v1",
        "created_at_utc": utcnow(),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "workspace": {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "runner_sha256": runner_sha,
        },
        "storage_preflight": preflight,
        "pins": {
            "archive": {"path": str(args.archive), "bytes": len(bundle.archive_blob), "sha256": archive_sha},
            "carrier_codec": {"path": str(PR130_CODE / "carrier_codec.py"), "sha256": codec_sha},
            "inflate": {"path": str(PR130_CODE / "inflate.py"), "sha256": inflate_sha},
            "pose_film": {
                "path": str(SRC / "tac/torch_vehicle/pose_film.py"),
                "sha256": sha256_file(SRC / "tac/torch_vehicle/pose_film.py"),
            },
        },
        "baseline": {
            "archive_bytes": len(bundle.archive_blob),
            "member_bytes": len(bundle.member),
            "models_compressed_bytes": len(bundle.models_compressed),
            "models_raw_bytes": len(bundle.models_raw),
            "semantic_bytes": len(bundle.semantic),
            "semantic_sha256": sha256_bytes(bundle.semantic),
            "carrier_bytes": len(bundle.carrier),
            "carrier_sha256": sha256_bytes(bundle.carrier),
            "hpac_bytes": len(bundle.hpac),
            "hpac_sha256": sha256_bytes(bundle.hpac),
            "token_bytes": len(bundle.tokens),
            "token_sha256": sha256_bytes(bundle.tokens),
            "basis_bit_count": int(basis_bits),
            "basis_payload_bytes": (int(basis_bits) + 7) // 8,
            "basis_symbols": int(basis_codes.size),
            "basis_bits_per_value": float(basis_bits / basis_codes.size),
            "basis_source_precision_bits": 5,
            "basis_code_range": [int(basis_codes.min()), int(basis_codes.max())],
            "coefficient_bit_count": int(coefficient_bits),
            "coefficient_payload_bytes": (int(coefficient_bits) + 7) // 8,
            "coefficient_symbols": int(coefficient_codes.size),
            "coefficient_bits_per_value": float(coefficient_bits / coefficient_codes.size),
            "coefficient_shape": list(coefficient_codes.shape),
            "coefficient_code_range": [int(coefficient_codes.min()), int(coefficient_codes.max())],
            "lossless_carrier_byte_roundtrip": True,
        },
        "charter_corrections": {
            "coefficient_denominator": "600x12=7200, not 1200x12=14400",
            "coefficient_bits_per_value": float(coefficient_bits / coefficient_codes.size),
            "basis_precision": "already signed int5 in the deployed CPR1 archive; int8 is only the decoded container dtype",
        },
        "diagnostics": {
            "coefficients": coefficient_diagnostics(coefficients),
            "basis": basis_diagnostics(basis),
        },
        "exact_predictor_controls": {
            "status": "MEASURED_BYTE_ONLY_EXACT_ARRAYS",
            "rows": [row["name"] for row in predictor_rows],
            "reason": (
                "All predictors use an exact modulo-int12 residual and a counted "
                "PK2R receiver dispatch. Bare recoding of unchanged CPR1 bytes was "
                "not raced."
            ),
        },
        "gauge_search": gauge_rows,
        "packet_shortlists": packet_shortlists,
        "candidates": candidates,
        "score_queue": [row["name"] for row in candidates if row["score_requested"]],
        "unrun_optimal_form": [
            "retrained smaller carrier",
            "full n600 promotion of a non-control row; not fired because no n120 non-control row beat CPR1",
        ],
    }
    receipt_path = out / "analysis_receipt.json"
    atomic_json(receipt_path, receipt)
    state = {
        "schema": "ddm_pk2_pose_carrier_representation_state.v1",
        "analysis_complete": True,
        "analysis_receipt": str(receipt_path),
        "analysis_receipt_sha256": sha256_file(receipt_path),
        "score_complete": False,
        "updated_at_utc": utcnow(),
    }
    atomic_json(args.resume_from, state)
    atomic_json(out / "checkpoints" / "stage_analysis_complete.json", state)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "candidate_count": len(candidates),
                "score_queue_count": len(receipt["score_queue"]),
            },
            indent=2,
        )
    )
    return 0


def load_target_cache(path: Path) -> dict[str, torch.Tensor]:
    raw = lzma.decompress(path.read_bytes()) if path.suffix == ".xz" else path.read_bytes()
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or set(value) != {"pose", "seg"}:
        raise ValueError("target cache must contain exactly pose and seg")
    if tuple(value["pose"].shape) != (N, 6):
        raise ValueError("target pose shape mismatch")
    if tuple(value["seg"].shape) != (N, 384, 512):
        raise ValueError("target seg shape mismatch")
    if tensor_sha256(value["seg"]) != EXPECTED_SEG_SHA:
        raise RuntimeError("target seg tensor is not the decoded PR130 token map")
    return value


def augment_capacity(args: argparse.Namespace) -> int:
    out = validate_out_dir(args.out_dir)
    state = json.loads(args.resume_from.read_text(encoding="utf-8"))
    analysis_path = Path(state["analysis_receipt"])
    score_path = Path(state["score_receipt"])
    if sha256_file(analysis_path) != state["analysis_receipt_sha256"]:
        raise RuntimeError("analysis receipt hash drift")
    if sha256_file(score_path) != state["score_receipt_sha256"]:
        raise RuntimeError("score receipt hash drift")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    score_receipt = json.loads(score_path.read_text(encoding="utf-8"))
    order = measured_dimension_order(score_receipt["rows"])

    codec, inflate = setup_imports()
    bundle = extract_bundle(args.archive)
    decoded = codec.decode_compact_carrier(bundle.carrier, basis_count=math.prod(BASIS_SHAPE), frames=N, dimensions=DIM)
    basis_scales, basis_codes_flat, coefficient_scales, encoded = decoded
    basis_codes = basis_codes_flat.reshape(BASIS_SHAPE)
    coefficient_codes = decode_absolute_codes(encoded)
    basis = basis_codes.astype(np.float32) * basis_scales[:, None, None, None]
    coefficients = coefficient_codes.astype(np.float32) * coefficient_scales[None]
    names = {"capacity_drop_measured_nested02", "capacity_drop_measured_nested03"}
    analysis["candidates"] = [row for row in analysis["candidates"] if row["name"] not in names]
    for count in (2, 3):
        dropped_basis = basis.copy()
        dropped_coefficients = coefficients.copy()
        for dimension in order[:count]:
            dropped_basis[dimension] = 0
            dropped_coefficients[:, dimension] = 0
        row = materialize_candidate(
            name=f"capacity_drop_measured_nested{count:02d}",
            arm="C_capacity",
            scope="MEASURED_EXISTING_CARRIER_RESPONSE",
            bundle=bundle,
            codec=codec,
            inflate=inflate,
            basis=dropped_basis,
            coefficients=dropped_coefficients,
            baseline_basis=basis,
            baseline_coefficients=coefficients,
            out_dir=out,
            score=True,
            notes=(
                "Nested by measured n120 joint distortion cost per full-archive byte "
                f"saved; dimensions {order[:count]}. Existing trained carrier only."
            ),
        )
        row["delta_archive_bytes"] = row["archive_bytes"] - analysis["baseline"]["archive_bytes"]
        row["delta_carrier_bytes"] = row["carrier_bytes"] - analysis["baseline"]["carrier_bytes"]
        analysis["candidates"].append(row)
    analysis["measured_capacity_order"] = {
        "metric": "(delta_s_seg + delta_s_pose) / full_archive_bytes_saved",
        "selection_scope": score_receipt["scope"],
        "dimensions": order,
    }
    analysis["score_queue"] = [row["name"] for row in analysis["candidates"] if row["score_requested"]]
    atomic_json(analysis_path, analysis)
    state.update(
        {
            "analysis_receipt_sha256": sha256_file(analysis_path),
            "score_complete": False,
            "updated_at_utc": utcnow(),
        }
    )
    atomic_json(args.resume_from, state)
    receipt = {
        "schema": "ddm_pk2_measured_capacity_materialization.v1",
        "created_at_utc": utcnow(),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "measured_dimension_order": order,
        "materialized": sorted(names),
        "analysis_receipt": str(analysis_path),
        "analysis_receipt_sha256": state["analysis_receipt_sha256"],
    }
    receipt_path = out / "checkpoints" / "stage_measured_capacity_materialized.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))
    return 0


def augment_composition(args: argparse.Namespace) -> int:
    out = validate_out_dir(args.out_dir)
    state = json.loads(args.resume_from.read_text(encoding="utf-8"))
    analysis_path = Path(state["analysis_receipt"])
    score_path = Path(state["score_receipt"])
    if sha256_file(analysis_path) != state["analysis_receipt_sha256"]:
        raise RuntimeError("analysis receipt hash drift")
    if sha256_file(score_path) != state["score_receipt_sha256"]:
        raise RuntimeError("score receipt hash drift")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    score_receipt = json.loads(score_path.read_text(encoding="utf-8"))
    scored = score_receipt["rows"]
    a_rows = [row for row in scored if row["arm"] == "A_low_rank_plus_residual"]
    b_rows = [
        row
        for row in scored
        if row["arm"]
        in {
            "B_spatial_DCT_packet",
            "B_cross_plane_low_rank_packet",
            "B_per_plane_precision",
            "B_precision",
        }
    ]
    if not a_rows or not b_rows:
        raise RuntimeError("composition requires independently scored optimal-form A and B rows")
    best_a = min(a_rows, key=lambda row: row["delta_s"])
    best_b = min(b_rows, key=lambda row: row["delta_s"])
    by_name = {row["name"]: row for row in analysis["candidates"]}
    source_a = by_name[best_a["name"]]
    source_b = by_name[best_b["name"]]
    codec, _ = setup_imports()
    a_bundle = extract_bundle(Path(source_a["archive_path"]))
    b_bundle = extract_bundle(Path(source_b["archive_path"]))
    a_basis_mode, a_basis_component, a_coeff_mode, a_coeff_component = carrier_component_pair(a_bundle.carrier, codec)
    b_basis_mode, b_basis_component, b_coeff_mode, b_coeff_component = carrier_component_pair(b_bundle.carrier, codec)
    del a_basis_mode, a_basis_component, b_coeff_mode, b_coeff_component
    declared_basis, _ = decode_overlay_carrier(
        encode_overlay(
            b_basis_mode,
            b_basis_component,
            a_coeff_mode,
            a_coeff_component,
        ),
        codec,
    )
    _, declared_coefficients = decode_overlay_carrier(
        encode_overlay(
            b_basis_mode,
            b_basis_component,
            a_coeff_mode,
            a_coeff_component,
        ),
        codec,
    )
    baseline = extract_bundle(args.archive)
    baseline_basis, baseline_coefficients = decode_overlay_carrier(
        encode_overlay(
            *carrier_component_pair(baseline.carrier, codec)[:2],
            *carrier_component_pair(baseline.carrier, codec)[2:],
        ),
        codec,
    )
    name = "compose_best_measured_A_B"
    analysis["candidates"] = [row for row in analysis["candidates"] if row["name"] != name]
    row = materialize_overlay_candidate(
        name=name,
        arm="A_plus_B_composed_optimal_packet",
        scope="MEASURED_OPTIMAL_FORM_PACKET_COMPOSITION",
        bundle=baseline,
        codec=codec,
        basis_mode=b_basis_mode,
        basis_component=b_basis_component,
        coefficient_mode=a_coeff_mode,
        coefficient_component=a_coeff_component,
        declared_basis=declared_basis,
        declared_coefficients=declared_coefficients,
        baseline_basis=baseline_basis,
        baseline_coefficients=baseline_coefficients,
        out_dir=out,
        score=True,
        notes=(
            f"Materialized composition of independently measured A={best_a['name']} "
            f"and B={best_b['name']}; deltas are not added arithmetically."
        ),
    )
    row["delta_archive_bytes"] = row["archive_bytes"] - analysis["baseline"]["archive_bytes"]
    row["delta_carrier_bytes"] = row["carrier_bytes"] - analysis["baseline"]["carrier_bytes"]
    row["composition_sources"] = {"A": best_a["name"], "B": best_b["name"]}
    analysis["candidates"].append(row)
    analysis["composition"] = row["composition_sources"]
    analysis["score_queue"] = [value["name"] for value in analysis["candidates"] if value["score_requested"]]
    atomic_json(analysis_path, analysis)
    state.update(
        {
            "analysis_receipt_sha256": sha256_file(analysis_path),
            "score_complete": False,
            "updated_at_utc": utcnow(),
        }
    )
    atomic_json(args.resume_from, state)
    receipt = {
        "schema": "ddm_pk2_measured_composition_materialization.v1",
        "created_at_utc": utcnow(),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "source_A": best_a,
        "source_B": best_b,
        "materialized": row,
        "analysis_receipt_sha256": state["analysis_receipt_sha256"],
    }
    receipt_path = out / "checkpoints/stage_composition_materialized.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))
    return 0


def finalize(args: argparse.Namespace) -> int:
    out = validate_out_dir(args.out_dir)
    state = json.loads(args.resume_from.read_text(encoding="utf-8"))
    if not state.get("score_complete"):
        raise RuntimeError("cannot finalize before the full score queue completes")
    analysis_path = Path(state["analysis_receipt"])
    score_path = Path(state["score_receipt"])
    if sha256_file(analysis_path) != state["analysis_receipt_sha256"]:
        raise RuntimeError("analysis receipt hash drift")
    if sha256_file(score_path) != state["score_receipt_sha256"]:
        raise RuntimeError("score receipt hash drift")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    score_receipt = json.loads(score_path.read_text(encoding="utf-8"))
    scored_names = {row["name"] for row in score_receipt["rows"]}
    if "compose_best_measured_A_B" not in scored_names:
        raise RuntimeError("winning A/B packet composition has not been scored")
    byte_only = [
        {
            "name": row["name"],
            "arm": row["arm"],
            "scope": row["scope"],
            "archive_bytes": row["archive_bytes"],
            "delta_archive_bytes": row["delta_archive_bytes"],
            "carrier_bytes": row["carrier_bytes"],
            "archive_sha256": row["archive_sha256"],
            "parse_back": row["parse_back"],
        }
        for row in analysis["candidates"]
        if row["name"] not in scored_names
    ]
    best = min(score_receipt["rows"], key=lambda row: row["delta_s"])
    composition = next(row for row in score_receipt["rows"] if row["name"] == "compose_best_measured_A_B")
    unrun_rungs = [
        row
        for row in analysis["unrun_optimal_form"]
        if not row.startswith("full n600")
    ]
    unrun_rungs.append(
        "full n600 promotion of a non-control row; not fired because no n120 non-control row beat CPR1"
    )
    scoped_verdicts = [
        {
            "scope": "INSTANCE",
            "subject": "seeded stratified-random n120 PR130 CPR1 archive",
            "verdict": ("ADVISORY_WIN" if best["delta_s"] < 0 else "NO_ADVISORY_WIN"),
            "reason": (
                f"best measured row {best['name']} delta_S={best['delta_s']:.12g}; not n600 and not contest authority"
            ),
        },
        {
            "scope": "FORMULATION",
            "subject": "bare outer-coder race on unchanged CPR1 carrier bytes",
            "verdict": "DEAD_END_NOT_RERUN",
            "reason": "settled +4 B prior result and current byte-only ladder changes representations instead",
        },
        {
            "scope": "FAMILY",
            "subject": "retrained smaller PR130 carrier",
            "verdict": "UNRUN_OUT_OF_SCOPE",
            "reason": "only the existing trained-carrier capacity response was measured",
        },
    ]
    final_receipt = {
        "schema": "ddm_pk2_pose_carrier_representation_final.v1",
        "created_at_utc": utcnow(),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "finalizer_runner_sha256": sha256_file(Path(__file__)),
        "analysis_receipt": {
            "path": str(analysis_path),
            "sha256": sha256_file(analysis_path),
        },
        "score_receipt": {
            "path": str(score_path),
            "sha256": sha256_file(score_path),
        },
        "input_pins": analysis["pins"],
        "actual_cpr1_counts": analysis["baseline"],
        "charter_corrections": analysis["charter_corrections"],
        "selection": score_receipt["selection"],
        "diagnostics": analysis["diagnostics"],
        "ranked_scorer_rows": score_receipt["rows"],
        "byte_only_rows": sorted(byte_only, key=lambda row: row["archive_bytes"]),
        "best_measured_row": best,
        "composed_best": composition,
        "unrun_rungs": unrun_rungs,
        "scoped_verdicts": scoped_verdicts,
        "follow_ons": [
            {
                "id": "ddm_pk2_rate_aware_gauge_qat",
                "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                "owner": "future PR130 pose-carrier training owner",
                "consumer_store": ".omx/state/codex_arm_queue.next_if_resumed.jsonl",
                "fire_trigger": (
                    "A scorer-free, counted PK2R preflight must project at least "
                    "2000 full-archive bytes saved while keeping carrier-product "
                    "MSE below 2.5e-6; only then launch resumable quantization-aware "
                    "training and a seeded stratified n120 scorer row."
                ),
            },
            {
                "id": "ddm_pk2_full_n600_promotion",
                "disposition": "FOLDED",
                "owner": "ddm_pk2",
                "consumer_store": "/Volumes/VertigoDataTier/pact/ddm_pk2_20260809/FINAL_RECEIPT.json",
                "fire_trigger": "None in this run: no non-control n120 row had delta_S < 0.",
            },
        ],
        "recall_evidence": {
            "sources_searched": [
                ".omx/research/ full-text queries CPR1, compact_carrier, pose carrier, low-rank pose, acceleration matching, gauge rotation, signed int5",
                ".omx/research/CANONICAL_RESEARCH_INDEX_20260629.md",
                ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md",
                ".omx/state/canonical_equations_registry.jsonl via tools/list_canonical_equations.py --json",
                ".omx/state/main_hot_state.md and tools/codex_arm_queue.py status",
            ],
            "beyond_charter": [
                "#140 rank-4/511 was Pareto-dominant on a different 600x6 FiLM pose object; method reused, number not transferred",
                "AM1 already identifies exact delta, second-order, and acceleration/control residual packet races as the correct smooth-trajectory test",
                "ddm_na2 measured pose-prefix bias 2.54-4.21x harder, which requires representative n120 selection",
                "the 20260808 ddm_pk2 directory is an earlier Candidate-A surface-fit task with the same short identifier, not this PR130 pose-carrier experiment",
            ],
            "plan_changes": [
                "implemented direct factor-plus-residual packets instead of treating projected full arrays as family evidence",
                "implemented counted exact predictor and transform receiver modes instead of rerunning the closed unchanged-byte coder cell",
                "required an explicit live-ledger confirmation before scorer launch",
            ],
        },
        "commands": [
            ".venv/bin/python -m pytest -q experiments/tests/test_ddm_pk2_pose_carrier_representation.py",
            ".venv/bin/python -m py_compile experiments/ddm_pk2_pose_carrier_representation.py",
            ".venv/bin/python experiments/ddm_pk2_pose_carrier_representation.py --help",
            ".venv/bin/python experiments/ddm_pk2_pose_carrier_representation.py --out-dir /Volumes/VertigoDataTier/pact/ddm_pk2_20260809 --resume-from /Volumes/VertigoDataTier/pact/ddm_pk2_20260809/progress.json analyze",
            ".venv/bin/python experiments/ddm_pk2_pose_carrier_representation.py --out-dir /Volumes/VertigoDataTier/pact/ddm_pk2_20260809 --resume-from /Volumes/VertigoDataTier/pact/ddm_pk2_20260809/progress.json --seed 20260809 score --n 120 --scorer-slot-confirmed-free",
            ".venv/bin/python experiments/ddm_pk2_pose_carrier_representation.py --out-dir /Volumes/VertigoDataTier/pact/ddm_pk2_20260809 --resume-from /Volumes/VertigoDataTier/pact/ddm_pk2_20260809/progress.json augment-capacity",
            ".venv/bin/python experiments/ddm_pk2_pose_carrier_representation.py --out-dir /Volumes/VertigoDataTier/pact/ddm_pk2_20260809 --resume-from /Volumes/VertigoDataTier/pact/ddm_pk2_20260809/progress.json augment-composition",
            ".venv/bin/python experiments/ddm_pk2_pose_carrier_representation.py --out-dir /Volumes/VertigoDataTier/pact/ddm_pk2_20260809 --resume-from /Volumes/VertigoDataTier/pact/ddm_pk2_20260809/progress.json finalize",
        ],
    }
    RESEARCH_OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = RESEARCH_OUT / "FINAL_RECEIPT.json"
    atomic_json(receipt_path, final_receipt)
    scorer_lines = [
        "| rank | row | arm | archive B | delta B | d_pose | d_seg | delta S | scope |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(score_receipt["rows"], start=1):
        scorer_lines.append(
            f"| {rank} | `{row['name']}` | {row['arm']} | {row['archive_bytes']} | "
            f"{row['delta_archive_bytes']:+d} | {row['d_pose']:.9g} | "
            f"{row['d_seg']:.9g} | {row['delta_s']:+.9g} | {row['scope']} |"
        )
    byte_lines = [
        "| row | arm | archive B | delta B | parse-back | scope |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in sorted(byte_only, key=lambda value: value["archive_bytes"]):
        byte_lines.append(
            f"| `{row['name']}` | {row['arm']} | {row['archive_bytes']} | "
            f"{row['delta_archive_bytes']:+d} | PASS | {row['scope']} |"
        )
    markdown = (
        "\n".join(
            [
                "# DDM PK2 pose-carrier representation results",
                "",
                f"All numbers below are {AXIS}, `score_claim=false`. The exact contest pointer did not move.",
                "",
                "## Measured result",
                "",
                f"Best n120 row: `{best['name']}` at {best['archive_bytes']} B, d_pose {best['d_pose']:.12g}, d_seg {best['d_seg']:.12g}, delta S {best['delta_s']:+.12g}.",
                f"Composed A+B row: {composition['archive_bytes']} B, d_pose {composition['d_pose']:.12g}, d_seg {composition['d_seg']:.12g}, delta S {composition['delta_s']:+.12g}.",
                "",
                "## Ranked scorer table",
                "",
                *scorer_lines,
                "",
                "## Byte-only table",
                "",
                "These rows have real archive bytes and receiver parse-back, but were not scorer rows.",
                "",
                *byte_lines,
                "",
                "## RECALL EVIDENCE",
                "",
                "Full-corpus recall found the #140 low-rank method on a different object, AM1's exact residual-predictor crosswalk, ddm_na2's measured pose-prefix bias, and a 20260808 Candidate-A surface-fit task that reused the short ddm_pk2 identifier but is not this PR130 experiment. That changed the implementation to direct counted factor/residual and transform packets, while retaining seeded stratified n120 selection. No ancestor result was transferred as a PR130 number.",
                "",
                "## Boundaries",
                "",
                "- Not measured: contest CPU/CUDA or a retrained smaller carrier.",
                "- No non-control row advanced to n600: the n120 gate selected unchanged CPR1, so firing a full scorer on the losing composed packet would not be a promotion measurement.",
                "- Measured: exact CPR1 anatomy, real candidate archives, receiver parse-back, seeded stratified n120 frozen CPU-torch scorer rows.",
                "- Unchanged-byte outer-coder races were not repeated.",
                "",
                "## NEXT_IF_RESUMED",
                "",
                "- `ddm_pk2_rate_aware_gauge_qat`; disposition=QUEUED-WITH-A-FIRE-ORDER; owner=future PR130 pose-carrier training owner; consumer_store=.omx/state/codex_arm_queue.next_if_resumed.jsonl; fire_trigger=a scorer-free counted PK2R preflight projects at least 2000 full-archive bytes saved with carrier-product MSE below 2.5e-6, after which only a resumable quantization-aware training run and seeded stratified n120 row may fire.",
                "- `ddm_pk2_full_n600_promotion`; disposition=FOLDED; owner=ddm_pk2; consumer_store=/Volumes/VertigoDataTier/pact/ddm_pk2_20260809/FINAL_RECEIPT.json; fire_trigger=none in this run because no non-control n120 row had delta S below zero.",
                "",
                "## LIVE-HYPOTHESES",
                "",
                "- A learned rate-aware gauge can still outperform the 64 random rotations: C @ B is invariant before quantization, while the random search optimized neither entropy nor scorer sensitivity. The preflight threshold above prevents this from becoming another unconstrained search.",
                "- Quantization-aware retraining may make per-plane low precision usable because per-plane scaling reduced post-hoc int4 pose damage relative to the shared-scale row. This remains weak evidence: the measured absolute distortion is still far outside break-even.",
                "",
                "## DEAD-ENDS",
                "",
                "- Bare Brotli/LZMA/Zstd/ANS recoding of unchanged CPR1 pose bytes: the prior real-byte race was +4 B and the current runner did not reopen it.",
                "- Calling int7/int6/int5 a precision ladder: deployed source values are already signed int5, so those labels are one control row.",
                "- Treating projected full arrays as a low-rank or transform family verdict: those rows remain toy brackets and direct factor packets carry the actual verdict surface.",
                "- Exact first/second/AR/spline coefficient predictors on this instance: all reconstructed the deployed coefficients exactly but enlarged the archive by 1804 to 2232 bytes.",
                "- Direct low-rank coefficient factors plus residuals on this instance: the exact row was 4316 bytes larger, and every lossy shortlisted row worsened both rate or pose enough to lose.",
                "- Post-hoc dimension dropping on the existing carrier: every single dimension and both measured-order nested drops were decisively outside break-even.",
                "- The seeded 64-rotation random gauge formulation: its best materialized row was larger and had d_pose 0.442203.",
            ]
        )
        + "\n"
    )
    results_path = RESEARCH_OUT / "RESULTS.md"
    results_path.write_text(markdown, encoding="utf-8")
    final_state = {
        **state,
        "finalize_complete": True,
        "final_receipt": str(receipt_path),
        "final_receipt_sha256": sha256_file(receipt_path),
        "results_markdown": str(results_path),
        "results_markdown_sha256": sha256_file(results_path),
        "updated_at_utc": utcnow(),
    }
    atomic_json(args.resume_from, final_state)
    atomic_json(out / "checkpoints/stage_finalize_complete.json", final_state)
    print(
        json.dumps(
            {
                "final_receipt": str(receipt_path),
                "final_receipt_sha256": sha256_file(receipt_path),
                "results": str(results_path),
                "results_sha256": sha256_file(results_path),
                "best": best["name"],
                "best_delta_s": best["delta_s"],
            },
            indent=2,
        )
    )
    return 0


def pose_target_center_energy(targets: np.ndarray) -> np.ndarray:
    centered = targets.astype(np.float64) - targets.astype(np.float64).mean(axis=0, keepdims=True)
    return np.mean(centered * centered, axis=1)


def selection(n: int, seed: int, targets: torch.Tensor) -> tuple[list[int], dict[str, Any]]:
    if n < 120:
        raise ValueError("score stage requires n>=120")
    from tac.subset_selection import MODE_STRATIFIED, select

    governing = pose_target_center_energy(targets.numpy()).tolist()
    selected = select(
        n,
        N,
        mode=MODE_STRATIFIED,
        seed=seed,
        block_count=10,
        governing=governing,
        governing_name="pose_target_center_energy",
    )
    result = selected.provenance()
    if result.get("pair_selection") != MODE_STRATIFIED:
        raise RuntimeError("selector did not preserve stratified mode")
    return list(selected.indices), result


def load_scorers(device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    import modules

    posenet = modules.PoseNet().eval().to(device)
    segnet = modules.SegNet().eval().to(device)
    posenet.load_state_dict(load_file(str(UPSTREAM / "models/posenet.safetensors"), device=str(device)))
    segnet.load_state_dict(load_file(str(UPSTREAM / "models/segnet.safetensors"), device=str(device)))
    for parameter in list(posenet.parameters()) + list(segnet.parameters()):
        parameter.requires_grad_(False)
    return posenet, segnet


@torch.inference_mode()
def render_masters(
    semantic: torch.nn.Module,
    token_maps: torch.Tensor,
    indices: list[int],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    semantic = semantic.eval().to(device)
    rendered: list[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        idx = torch.tensor(current, dtype=torch.long, device=device)
        tokens = token_maps[current].long().to(device)
        master_eval = semantic(tokens, idx)
        master = (
            F.interpolate(
                master_eval,
                size=(CAMERA_H, CAMERA_W),
                mode="bilinear",
                align_corners=False,
            )
            .clamp(0.0, 255.0)
            .round()
            .to(torch.uint8)
            .cpu()
        )
        rendered.append(master)
    return torch.cat(rendered, dim=0)


@torch.inference_mode()
def score_seg(
    segnet: torch.nn.Module,
    masters: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, list[float]]:
    errors: list[torch.Tensor] = []
    per_pair: list[float] = []
    for start in range(0, len(masters), batch_size):
        batch = masters[start : start + batch_size].float().to(device)
        resized = F.interpolate(batch, size=(384, 512), mode="bilinear")
        prediction = segnet(resized).argmax(dim=1).cpu()
        current = (prediction != targets[start : start + len(batch)]).float()
        errors.append(current)
        per_pair.extend(current.mean(dim=(1, 2)).tolist())
    return float(torch.cat(errors).mean().item()), per_pair


@torch.inference_mode()
def render_slaves(
    inflate: Any,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
    selected_indices: list[int],
    batch_size: int,
    device: torch.device,
) -> Iterable[tuple[int, torch.Tensor]]:
    normalized = inflate.normalized_basis(basis.to(device))
    coefficients = coefficients.to(device)
    for start in range(0, len(selected_indices), batch_size):
        current = selected_indices[start : start + batch_size]
        coeff = coefficients[current]
        carrier = torch.einsum("bk,kchw->bchw", coeff, normalized)
        carrier = carrier / math.sqrt(DIM)
        slave_eval = (127.5 + inflate.CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0).round()
        slave = (
            F.interpolate(
                slave_eval,
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            .clamp(0.0, 255.0)
            .round()
            .to(torch.uint8)
            .cpu()
        )
        yield start, slave


@torch.inference_mode()
def score_pose_candidate(
    *,
    posenet: torch.nn.Module,
    inflate: Any,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
    masters: torch.Tensor,
    pose_targets: torch.Tensor,
    selected_indices: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[float, list[float]]:
    per_pair: list[float] = []
    for start, slaves in render_slaves(
        inflate,
        basis,
        coefficients,
        selected_indices,
        batch_size,
        device,
    ):
        count = len(slaves)
        master = masters[start : start + count]
        pair = torch.stack([slaves, master], dim=1).float().to(device)
        prepared = posenet.preprocess_input(pair)
        predicted = posenet(prepared)["pose"][:, :6].cpu()
        target = pose_targets[start : start + count]
        current = (predicted - target).square().mean(dim=1)
        per_pair.extend(current.tolist())
    return float(np.mean(per_pair)), per_pair


def candidate_arrays(archive_path: Path, inflate: Any) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    bundle = extract_bundle(archive_path)
    if bundle.carrier[:4] == OVERLAY_MAGIC:
        codec, _ = setup_imports()
        basis_array, coefficient_array = decode_overlay_carrier(bundle.carrier, codec)
        return (
            torch.from_numpy(basis_array),
            torch.from_numpy(coefficient_array),
            bundle.semantic,
        )
    semantic_pose = struct.pack("<II", len(bundle.semantic), len(bundle.carrier))
    semantic_pose += bundle.semantic + bundle.carrier
    _, basis, coefficients = inflate.unpack_semantic_pose(semantic_pose)
    return basis, coefficients, bundle.semantic


def score(args: argparse.Namespace) -> int:
    if not args.scorer_slot_confirmed_free:
        raise RuntimeError(
            "score stage requires primary-agent confirmation that the live scorer "
            "ledger and governed process receipts are free"
        )
    out = validate_out_dir(args.out_dir)
    preflight = checked_free_space(out, args.required_free_bytes)
    state = json.loads(args.resume_from.read_text(encoding="utf-8"))
    if not state.get("analysis_complete"):
        raise RuntimeError("analysis stage is not complete")
    analysis_path = Path(state["analysis_receipt"])
    if sha256_file(analysis_path) != state["analysis_receipt_sha256"]:
        raise RuntimeError("analysis receipt hash drift")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    targets = load_target_cache(args.target_cache)
    indices, selection_receipt = selection(args.n, args.seed, targets["pose"])
    device = torch.device("cpu")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _, inflate = setup_imports()
    baseline_row = next(row for row in analysis["candidates"] if row["name"] == "baseline_cpr1_int5")
    baseline_bundle = extract_bundle(Path(baseline_row["archive_path"]))
    semantic_pose = (
        struct.pack("<II", len(baseline_bundle.semantic), len(baseline_bundle.carrier))
        + baseline_bundle.semantic
        + baseline_bundle.carrier
    )
    semantic, _, _ = inflate.unpack_semantic_pose(semantic_pose)

    master_path = out / "checkpoints" / f"masters_n{args.n}_seed{args.seed}.pt"
    master_receipt_path = master_path.with_suffix(".json")
    if master_path.exists() and master_receipt_path.exists():
        master_receipt = json.loads(master_receipt_path.read_text(encoding="utf-8"))
        if master_receipt.get("selection_indices") != indices:
            raise RuntimeError("cached master selection does not match score selection")
        masters = torch.load(master_path, map_location="cpu", weights_only=True)
        if tensor_sha256(masters) != master_receipt["tensor_sha256"]:
            raise RuntimeError("cached master tensor hash mismatch")
    else:
        masters = render_masters(
            semantic,
            targets["seg"],
            indices,
            args.render_batch_size,
            device,
        )
        master_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = master_path.with_suffix(".tmp")
        torch.save(masters, temporary)
        os.replace(temporary, master_path)
        master_receipt = {
            "schema": "ddm_pk2_master_cache.v1",
            "created_at_utc": utcnow(),
            "selection_indices": indices,
            "shape": list(masters.shape),
            "dtype": str(masters.dtype),
            "tensor_sha256": tensor_sha256(masters),
            "semantic_sha256": sha256_bytes(baseline_bundle.semantic),
            "source_token_tensor_sha256": tensor_sha256(targets["seg"]),
        }
        atomic_json(master_receipt_path, master_receipt)

    posenet, segnet = load_scorers(device)
    selected_seg_targets = targets["seg"][indices]
    selected_pose_targets = targets["pose"][indices]
    d_seg, per_pair_seg = score_seg(
        segnet,
        masters,
        selected_seg_targets,
        args.score_batch_size,
        device,
    )

    score_checkpoint = out / "checkpoints" / f"score_n{args.n}_seed{args.seed}.json"
    if score_checkpoint.exists():
        scored = json.loads(score_checkpoint.read_text(encoding="utf-8")).get("rows", [])
    else:
        scored = []
    current_candidates = {row["name"]: row for row in analysis["candidates"]}
    scored = [row for row in scored if row["name"] in current_candidates]
    for scored_row in scored:
        candidate = current_candidates[scored_row["name"]]
        if scored_row["archive_bytes"] != candidate["archive_bytes"]:
            raise RuntimeError(f"{scored_row['name']}: scorer checkpoint archive size drift")
        scored_row.update(
            {
                "archive_sha256": candidate["archive_sha256"],
                "carrier_sha256": candidate["carrier_sha256"],
                "frame1_semantic_sha256": sha256_bytes(baseline_bundle.semantic),
                "baseline_master_tensor_sha256": tensor_sha256(masters),
            }
        )
    completed = {row["name"] for row in scored}
    requested = [row for row in analysis["candidates"] if row.get("score_requested")]
    if args.only:
        wanted = set(args.only)
        requested = [row for row in requested if row["name"] in wanted]
        missing = wanted.difference(row["name"] for row in requested)
        if missing:
            raise ValueError(f"unknown or unrequested score candidates: {sorted(missing)}")
    for row in requested:
        archive_path = Path(row["archive_path"])
        if (
            not archive_path.is_file()
            or archive_path.stat().st_size != row["archive_bytes"]
            or sha256_file(archive_path) != row["archive_sha256"]
        ):
            raise RuntimeError(f"{row['name']}: candidate archive hash drift")
    for position, row in enumerate(requested, start=1):
        if row["name"] in completed:
            continue
        started = time.time()
        basis, coefficients, semantic_bytes = candidate_arrays(Path(row["archive_path"]), inflate)
        if sha256_bytes(semantic_bytes) != sha256_bytes(baseline_bundle.semantic):
            raise RuntimeError(f"{row['name']}: frame1 semantic bytes changed")
        d_pose, per_pair_pose = score_pose_candidate(
            posenet=posenet,
            inflate=inflate,
            basis=basis,
            coefficients=coefficients,
            masters=masters,
            pose_targets=selected_pose_targets,
            selected_indices=indices,
            batch_size=args.score_batch_size,
            device=device,
        )
        score_row = {
            "name": row["name"],
            "arm": row["arm"],
            "scope": row["scope"],
            "archive_bytes": row["archive_bytes"],
            "archive_sha256": row["archive_sha256"],
            "delta_archive_bytes": row["delta_archive_bytes"],
            "carrier_bytes": row["carrier_bytes"],
            "carrier_sha256": row["carrier_sha256"],
            "delta_carrier_bytes": row["delta_carrier_bytes"],
            "d_pose": d_pose,
            "d_seg": d_seg,
            "frame1_byte_identity": True,
            "frame1_semantic_sha256": sha256_bytes(baseline_bundle.semantic),
            "baseline_master_tensor_sha256": tensor_sha256(masters),
            "per_pair_pose": per_pair_pose,
            "per_pair_seg": per_pair_seg,
            "elapsed_seconds": time.time() - started,
            "measured_at_utc": utcnow(),
            "axis": AXIS,
            "score_claim": SCORE_CLAIM,
        }
        scored.append(score_row)
        completed.add(row["name"])
        atomic_json(
            score_checkpoint,
            {
                "schema": "ddm_pk2_pose_carrier_score_checkpoint.v1",
                "selection": selection_receipt,
                "rows": scored,
                "updated_at_utc": utcnow(),
            },
        )
        print(
            f"scored {position}/{len(requested)} {row['name']}: d_pose={d_pose:.12g} d_seg={d_seg:.12g}",
            flush=True,
        )

    baseline_score = next(row for row in scored if row["name"] == "baseline_cpr1_int5")
    for row in scored:
        terms = joint_delta_s(
            baseline_d_pose=baseline_score["d_pose"],
            candidate_d_pose=row["d_pose"],
            baseline_d_seg=baseline_score["d_seg"],
            candidate_d_seg=row["d_seg"],
            baseline_archive_bytes=baseline_score["archive_bytes"],
            candidate_archive_bytes=row["archive_bytes"],
        )
        delta_seg = row["d_seg"] - baseline_score["d_seg"]
        row["delta_d_pose"] = row["d_pose"] - baseline_score["d_pose"]
        row["delta_d_seg"] = delta_seg
        row["delta_s_seg"] = terms["seg"]
        row["delta_s_pose"] = terms["pose"]
        row["delta_s_rate"] = terms["rate"]
        row["delta_s"] = terms["total"]

    scored.sort(key=lambda row: row["delta_s"])
    receipt = {
        "schema": "ddm_pk2_pose_carrier_representation_score.v1",
        "created_at_utc": utcnow(),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "scope": f"seeded stratified-random n={args.n}; not n600; not contest authority",
        "selection": selection_receipt,
        "selection_indices": indices,
        "target_cache": {
            "path": str(args.target_cache),
            "file_sha256": sha256_file(args.target_cache),
            "seg_tensor_sha256": tensor_sha256(targets["seg"]),
            "pose_tensor_sha256": tensor_sha256(targets["pose"]),
            "lineage": "frozen RTX 2000 Ada DALI cache; candidate scorers run on macOS CPU",
        },
        "master_cache": {
            "path": str(master_path),
            "bytes": master_path.stat().st_size,
            "sha256": sha256_file(master_path),
            "tensor_sha256": tensor_sha256(masters),
        },
        "storage_preflight": preflight,
        "baseline": baseline_score,
        "rows": scored,
        "ranked_names": [row["name"] for row in scored],
        "completion": {
            "invocation_requested_count": len(requested),
            "full_queue_count": len(analysis["score_queue"]),
            "full_queue_completed_count": sum(name in completed for name in analysis["score_queue"]),
            "all_requested_complete": all(name in completed for name in analysis["score_queue"]),
        },
    }
    receipt_path = out / f"score_receipt_n{args.n}_seed{args.seed}.json"
    atomic_json(receipt_path, receipt)
    state.update(
        {
            "score_complete": receipt["completion"]["all_requested_complete"],
            "score_receipt": str(receipt_path),
            "score_receipt_sha256": sha256_file(receipt_path),
            "updated_at_utc": utcnow(),
        }
    )
    atomic_json(args.resume_from, state)
    atomic_json(out / "checkpoints" / f"stage_score_n{args.n}_complete.json", state)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "best": scored[0]["name"] if scored else None,
                "best_delta_s": scored[0]["delta_s"] if scored else None,
                "rows": len(scored),
            },
            indent=2,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    result.add_argument("--target-cache", type=Path, default=DEFAULT_TARGET_CACHE)
    result.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    result.add_argument(
        "--resume-from",
        type=Path,
        default=DEFAULT_OUT / "progress.json",
    )
    result.add_argument("--seed", type=int, default=20260809)
    result.add_argument("--required-free-bytes", type=int, default=2 << 30)
    subcommands = result.add_subparsers(dest="command", required=True)
    analyze_parser = subcommands.add_parser("analyze")
    analyze_parser.add_argument("--gauge-trials", type=int, default=64)
    analyze_parser.set_defaults(function=analyze)
    capacity_parser = subcommands.add_parser("augment-capacity")
    capacity_parser.set_defaults(function=augment_capacity)
    composition_parser = subcommands.add_parser("augment-composition")
    composition_parser.set_defaults(function=augment_composition)
    score_parser = subcommands.add_parser("score")
    score_parser.add_argument("--n", type=int, default=120)
    score_parser.add_argument("--cpu-threads", type=int, default=8)
    score_parser.add_argument("--render-batch-size", type=int, default=2)
    score_parser.add_argument("--score-batch-size", type=int, default=4)
    score_parser.add_argument("--only", action="append", default=[])
    score_parser.add_argument(
        "--scorer-slot-confirmed-free",
        action="store_true",
        help=(
            "fail-closed primary-agent assertion made only after checking the live "
            "arm ledger and governed scorer receipts"
        ),
    )
    score_parser.set_defaults(function=score)
    finalize_parser = subcommands.add_parser("finalize")
    finalize_parser.set_defaults(function=finalize)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
