#!/usr/bin/env python3
"""Retained conditional-entropy ceiling measurement for the pinned DX2 field.

This is a scorer-free diagnostic.  It reuses TO2's exact decoded token array,
measures explicitly named causal context families, and prices the video-derived
static count model separately from the ideal data term.  It also reports the
zero-transmitted-state KT prequential term for the same contexts so an in-sample
plug-in entropy cannot masquerade as available bytes.

Every count histogram and every serialized model variant is retained.  No
receiver, archive, scorer, upstream file, Modal job, or Metal job is modified or
invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np
from scipy.special import gammaln


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cx3_context_axis_ceiling/measurement_v4"
)
TO2_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_to2_token_ordering_race/measurement_v1"
)
TOKENS = TO2_ROOT / "retained/input/dx2_tokens_decoded.u8"
TOKEN_STREAM = TO2_ROOT / "retained/input/dx2_token_stream_rc64.bin"
CHECKPOINT_RECEIPT = TO2_ROOT / "retained/input/tokens_cpu_stage_complete.json"
DX2_ARCHIVE = TO2_ROOT / "retained/input/archive.zip"
TO2_RESULT = TO2_ROOT / "RESULT.json"
DX2_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")
FS2_RESULT = Path("/Volumes/APDataStore/pact/ddm_fs2/FS2_TOKEN_RD_REPLAY.json")
PREDICTOR_ARGMAX = Path(
    "/Volumes/APDataStore/pact/ddm_fs2/retained/token_rd/argmax_field.npy"
)
PREDICTOR_U_INDEX = Path(
    "/Volumes/APDataStore/pact/ddm_fs2/retained/token_rd/u_index_field.npy"
)

EXPECTED: dict[Path, tuple[int | None, str]] = {
    TOKENS: (
        117_964_800,
        "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
    ),
    TOKEN_STREAM: (
        113_777,
        "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5",
    ),
    CHECKPOINT_RECEIPT: (
        3_511,
        "c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9",
    ),
    DX2_ARCHIVE: (
        180_368,
        "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",
    ),
    FS2_RESULT: (
        3_014,
        "edf65f114cd01d468109a42bccba284b96365cd1aac77ad609cc5dfaddec8863",
    ),
    PREDICTOR_ARGMAX: (
        117_964_928,
        "93cdf71daedd39505c5031aca7cf8524a6358fc862ce838acfbcc1cc73dcae33",
    ),
    PREDICTOR_U_INDEX: (
        235_929_728,
        "74470f44a5333b27b131fcd0cf5d17fd41d82cc219d5fbd1b0557feb8825295f",
    ),
}

T, H, W = 600, 384, 512
SITES = H * W
SYMBOLS = T * SITES
ALPHABET = 5
SENTINEL = ALPHABET
CONTEXT_BASE = ALPHABET + 1
INCUMBENT_STREAM_BYTES = 113_777
INCUMBENT_MODEL_BYTES = 13_515
INCUMBENT_ARCHIVE_BYTES = 180_368
INCUMBENT_SCORE = 0.14821987563243377
DEMAND_BYTES = 42_382
TOKEN_TARGET_BYTES = 71_395
MODEL_MAGIC = b"CX3M1"
MODEL_HEADER = struct.Struct("<5sBQQ")
CHECKPOINT_INTERVAL = 120
# Ten percent of the campaign's required cut is a decision-relevant material
# result; smaller hindsight gaps do not justify a group-partition refit.
GROUP_REFIT_TRIGGER_SAVING = math.ceil(DEMAND_BYTES * 0.10)
MIN_STORAGE_BYTES = 8 * (1 << 30)
LZMA_FILTERS = [
    {
        "id": lzma.FILTER_LZMA1,
        "dict_size": 1 << 20,
        "lc": 3,
        "lp": 0,
        "pb": 2,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 128,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]

SOURCE_FILES = (
    DX2_RUNTIME / "runtime/residual_archive.py",
    DX2_RUNTIME / "runtime/rr4_free_corrector.py",
    DX2_RUNTIME / "runtime/fx1_logistic_mixer_corrector.py",
    DX2_RUNTIME / "runtime/fx2_model_axis_corrector.py",
    DX2_RUNTIME / "runtime/free_corrector.py",
    DX2_RUNTIME / "runtime/native_free_corrector.py",
    DX2_RUNTIME / "runtime/f26_corrector_native.c",
    DX2_RUNTIME / "cpr1/hpac_integer.py",
    DX2_RUNTIME / "cpr1/hpac_integer_sparse.py",
)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()

INCUMBENT_FAMILIES: dict[str, str] = {
    "shipped_joint": "predicted class x surprise bin x previous-frame agreement t-1/t-2 x same-site run length x previous-frame boundary distance",
    "temporal_spatial": "predicted class x previous-frame agreement t-1/t-2 x causal left/up agreement with prediction",
    "surprise_only": "predicted class x surprise bin",
    "spatial_surprise": "predicted class x causal left/up agreement with prediction x surprise bin",
    "spatial_boundary": "predicted class x causal left/up agreement with prediction x previous-frame boundary distance",
    "run_surprise": "predicted class x same-site temporal run length x surprise bin",
    "boundary_surprise": "predicted class x previous-frame boundary distance x surprise bin",
    "temporal_surprise": "predicted class x previous-frame agreement t-1/t-2 x surprise bin",
    "shipped_fast256": "shipped_joint context with its counts halved above 256 observations",
    "shipped_fast4096": "shipped_joint context with its counts halved above 4096 observations",
    "surprise_fast256": "surprise_only context with its counts halved above 256 observations",
    "spatial4_surprise": "predicted class x agreement count across causal left/up/up-right/up-left x surprise bin",
    "homog_surprise": "predicted class x number of distinct causal-neighbour classes x surprise bin",
    "homog_boundary_surprise": "predicted class x distinct causal-neighbour classes x previous-frame boundary distance x surprise bin",
    "spatial4_boundary": "predicted class x four-neighbour agreement count x previous-frame boundary distance",
    "homog_spatial4": "predicted class x distinct causal-neighbour classes x four-neighbour agreement count",
    "spatial4_temporal": "predicted class x previous-frame agreement t-1/t-2 x four-neighbour agreement count",
    "homog_surprise_fast256": "homog_surprise context with its counts halved above 256 observations",
    "spatial4_surprise_fast256": "spatial4_surprise context with its counts halved above 256 observations",
}


@dataclass(frozen=True)
class ContextSpec:
    name: str
    kind: str
    domain: int
    conditioning: str
    scope_reduction: str
    patch: int = 64
    delta: int = 2


BASE_SPECS = (
    ContextSpec(
        "order0",
        "order0",
        1,
        "no conditioning",
        "memoryless diagnostic",
    ),
    ContextSpec(
        "previous_decode_symbol",
        "previous_decode_symbol",
        CONTEXT_BASE,
        "immediately previous token in the incumbent frame->group->raster event order; frame-start sentinel",
        "one-symbol event-order context only",
    ),
    ContextSpec(
        "incumbent_group190",
        "group",
        190,
        "public group g=(x mod 64)+2*(y mod 64)",
        "group identity only; not the learned HPAC output",
    ),
    ContextSpec(
        "spatial_causal_left_up_r1",
        "spatial_lu1",
        CONTEXT_BASE**2,
        "causally available left and up tokens under the incumbent group wavefront; unavailable sentinel",
        "two spatial neighbours only",
    ),
    ContextSpec(
        "spatial_causal_left_up_r1_r2",
        "spatial_lu12",
        CONTEXT_BASE**4,
        "causally available left/up tokens at radii 1 and 2 under the incumbent group wavefront",
        "four axial spatial neighbours only",
    ),
    ContextSpec(
        "temporal_same_site_prev1",
        "temporal1",
        CONTEXT_BASE,
        "same raster site in previous frame; frame-0 sentinel",
        "one temporal frame only",
    ),
    ContextSpec(
        "temporal_same_site_prev1_prev2",
        "temporal2",
        CONTEXT_BASE**2,
        "same raster site in previous two frames; sentinels before history exists",
        "two temporal frames only",
    ),
    ContextSpec(
        "joint_left_up_prev1",
        "joint_lu_prev1",
        CONTEXT_BASE**3,
        "causal left/up tokens x same-site previous-frame token",
        "one spatial radius and one temporal frame",
    ),
    ContextSpec(
        "joint_left_up_prev1_prev2",
        "joint_lu_prev12",
        CONTEXT_BASE**4,
        "causal left/up tokens x same-site previous two frames",
        "one spatial radius and two temporal frames",
    ),
    ContextSpec(
        "incumbent_group190_prev1",
        "group_prev1",
        190 * CONTEXT_BASE,
        "incumbent group identity x same-site previous-frame token",
        "group plus one temporal symbol; not learned HPAC",
    ),
    ContextSpec(
        "incumbent_group190_left_up_prev1",
        "group_lu_prev1",
        190 * CONTEXT_BASE**3,
        "incumbent group identity x causal left/up tokens x same-site previous-frame token",
        "token-only proxy for the inherited group partition",
    ),
    ContextSpec(
        "joint_spatial4_prev1_prev2",
        "joint_spatial4_prev12",
        CONTEXT_BASE**6,
        "causal left/up/up-right/up-left tokens x same-site previous two frames",
        "deliberately rich token-only causal context; no coordinates or neural logits",
    ),
    ContextSpec(
        "hpac_argmax",
        "hpac_argmax",
        ALPHABET,
        "same-token-field RC2 replay's corrected coding-row argmax; its HPAC logit/CDF digests equal DX2",
        "learned predictor symbol only; no confidence or causal token feature",
    ),
    ContextSpec(
        "hpac_argmax_u64",
        "hpac_argmax_u64",
        ALPHABET * 64,
        "corrected coding-row argmax x 64 half-bit confidence bins derived from retained u-index",
        "learned predictor symbol and coarse confidence only",
    ),
    ContextSpec(
        "hpac_argmax_u64_group190",
        "hpac_group",
        190 * ALPHABET * 64,
        "incumbent group identity x corrected coding-row argmax x 64 confidence bins",
        "learned predictor summary plus inherited group identity",
    ),
    ContextSpec(
        "hpac_argmax_u64_left_up_prev1",
        "hpac_lu_prev1",
        ALPHABET * 64 * CONTEXT_BASE**3,
        "corrected coding-row argmax/confidence x causal left/up tokens x previous-frame same-site token",
        "learned predictor summary plus one spatial radius and one temporal frame",
    ),
    ContextSpec(
        "hpac_full_corrector_feature_union",
        "hpac_feature_union",
        ALPHABET * 64 * 6 * 6 * 4 * 8 * 5,
        "corrected coding-row argmax/confidence x four-neighbour agreement x neighbour homogeneity x t-1/t-2 agreement x run x previous-frame boundary bucket",
        "union of the public 19-member corrector feature axes, but not its continuous five-class probability vector",
    ),
    ContextSpec(
        "overrich_site_prev1_prev2",
        "site_prev12",
        SITES * CONTEXT_BASE**2,
        "exact public raster site x same-site previous two tokens",
        "deliberately over-rich upper-bound probe; site-specific empirical laws",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_receipt(receipt: dict[str, Any]) -> None:
    path = Path(receipt["path"])
    actual = file_receipt(path)
    if actual["bytes"] != receipt["bytes"] or actual["sha256"] != receipt["sha256"]:
        raise RuntimeError(f"retained artifact drifted: {path}")


def verify_pin(path: Path, expected_bytes: int | None, expected_sha: str) -> dict[str, Any]:
    receipt = file_receipt(path)
    if expected_bytes is not None and receipt["bytes"] != expected_bytes:
        raise RuntimeError(f"pinned size drift: {path}")
    if receipt["sha256"] != expected_sha:
        raise RuntimeError(f"pinned sha drift: {path}")
    return receipt


def fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        receipt = file_receipt(path)
        if receipt["bytes"] != len(payload) or receipt["sha256"] != digest:
            raise RuntimeError(f"refusing to overwrite differing retained payload: {path}")
        return receipt
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_parent(path)
    return {"path": str(path), "bytes": len(payload), "sha256": digest}


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = np.load(path, mmap_mode="r")
        if existing.shape != array.shape or existing.dtype != array.dtype or not np.array_equal(existing, array):
            raise RuntimeError(f"refusing to overwrite differing retained array: {path}")
        return file_receipt(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_parent(path)
    return file_receipt(path)


def encode_uleb(value: int) -> bytes:
    if value < 0:
        raise ValueError("ULEB requires nonnegative input")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(output)


def decode_uleb(payload: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if cursor >= len(payload) or shift > 63:
            raise ValueError("invalid ULEB stream")
        byte = payload[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7


def serialize_sparse_model(domain: int, active: np.ndarray, counts: np.ndarray) -> bytes:
    payload = bytearray(MODEL_HEADER.pack(MODEL_MAGIC, ALPHABET, domain, len(active)))
    previous = -1
    for context, row in zip(active.tolist(), counts.tolist(), strict=True):
        payload.extend(encode_uleb(int(context) - previous - 1))
        for value in row:
            payload.extend(encode_uleb(int(value)))
        previous = int(context)
    return bytes(payload)


def parse_sparse_model(payload: bytes) -> tuple[int, np.ndarray, np.ndarray]:
    if len(payload) < MODEL_HEADER.size:
        raise ValueError("truncated CX3 model")
    magic, alphabet, domain, count = MODEL_HEADER.unpack_from(payload)
    if magic != MODEL_MAGIC or alphabet != ALPHABET or domain <= 0:
        raise ValueError("invalid CX3 model header")
    active = np.empty(count, dtype=np.uint64)
    rows = np.empty((count, ALPHABET), dtype=np.uint64)
    cursor = MODEL_HEADER.size
    previous = -1
    for index in range(count):
        delta, cursor = decode_uleb(payload, cursor)
        context = previous + 1 + delta
        if context >= domain:
            raise ValueError("CX3 context outside domain")
        active[index] = context
        for symbol in range(ALPHABET):
            rows[index, symbol], cursor = decode_uleb(payload, cursor)
        previous = context
    if cursor != len(payload):
        raise ValueError("CX3 model has trailing bytes")
    return int(domain), active, rows


def compress_model(raw: bytes, coder: str) -> bytes:
    if coder == "brotli_q11":
        return bytes(brotli.compress(raw, quality=11))
    if coder == "zlib9":
        return zlib.compress(raw, level=9)
    if coder == "lzma1_1m":
        return lzma.compress(raw, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    raise ValueError(coder)


def decompress_model(payload: bytes, coder: str) -> bytes:
    if coder == "brotli_q11":
        return bytes(brotli.decompress(payload))
    if coder == "zlib9":
        return zlib.decompress(payload)
    if coder == "lzma1_1m":
        return lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    raise ValueError(coder)


def group_map(patch: int, delta: int) -> tuple[np.ndarray, int]:
    if H % patch or W % patch or patch <= 0 or delta <= 0:
        raise ValueError("group partition must tile the public plane and use positive delta")
    y = np.repeat(np.arange(H, dtype=np.int64), W)
    x = np.tile(np.arange(W, dtype=np.int64), H)
    groups = (x % patch) + delta * (y % patch)
    count = (1 + delta) * patch - delta
    if int(groups.min()) != 0 or int(groups.max()) != count - 1:
        raise RuntimeError("group domain is not dense")
    return groups, count


def neighbour_plan(groups: np.ndarray, patch: int, delta: int) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    y = np.repeat(np.arange(H, dtype=np.int64), W)
    x = np.tile(np.arange(W, dtype=np.int64), H)
    plans: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for dx, dy in ((-1, 0), (0, -1), (-2, 0), (0, -2), (1, -1), (-1, -1)):
        nx = x + dx
        ny = y + dy
        inside = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
        index = np.clip(ny, 0, H - 1) * W + np.clip(nx, 0, W - 1)
        # A whole group is decoded before the next.  Same-group neighbours are
        # unavailable even if their raster index is smaller.
        available = inside & (groups[index] < groups)
        plans[(dx, dy)] = (index.astype(np.int64), available)
    return plans


def pack_context(parts: tuple[np.ndarray, ...]) -> np.ndarray:
    if not parts:
        raise ValueError("context needs at least one part")
    result = np.asarray(parts[0], dtype=np.uint64)
    for part in parts[1:]:
        result = result * CONTEXT_BASE + np.asarray(part, dtype=np.uint64)
    return result


def neighbour_values(current: np.ndarray, plan: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    index, available = plan
    return np.where(available, current[index], SENTINEL).astype(np.uint64)


def boundary_buckets(previous: np.ndarray, max_distance: int = 4) -> np.ndarray:
    plane = np.asarray(previous).reshape(H, W)
    edge = np.zeros((H, W), dtype=bool)
    edge[1:] |= plane[1:] != plane[:-1]
    edge[:-1] |= plane[:-1] != plane[1:]
    edge[:, 1:] |= plane[:, 1:] != plane[:, :-1]
    edge[:, :-1] |= plane[:, :-1] != plane[:, 1:]
    result = np.full((H, W), max_distance, dtype=np.uint64)
    active = edge.copy()
    result[active] = 0
    for distance in range(1, max_distance):
        grown = active.copy()
        grown[1:] |= active[:-1]
        grown[:-1] |= active[1:]
        grown[:, 1:] |= active[:, :-1]
        grown[:, :-1] |= active[:, 1:]
        active = grown
        result[(result == max_distance) & active] = distance
    return result.reshape(-1)


def run_before_frame(frame: int, tokens: np.memmap) -> np.ndarray:
    run = np.zeros(SITES, dtype=np.uint64)
    if frame < 2:
        return run
    reference = np.asarray(tokens[frame - 1]).reshape(-1)
    active = np.ones(SITES, dtype=bool)
    for lag in range(2, min(frame + 1, 9)):
        active &= reference == np.asarray(tokens[frame - lag]).reshape(-1)
        run += active
    return run


def contexts_for_frame(
    spec: ContextSpec,
    frame: int,
    tokens: np.memmap,
    groups: np.ndarray,
    order: np.ndarray,
    plans: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    predictor_argmax: np.ndarray,
    predictor_u_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(tokens[frame]).reshape(-1)
    prev1 = np.asarray(tokens[frame - 1]).reshape(-1) if frame >= 1 else None
    prev2 = np.asarray(tokens[frame - 2]).reshape(-1) if frame >= 2 else None
    temporal1 = (
        prev1.astype(np.uint64, copy=False)
        if prev1 is not None
        else np.full(SITES, SENTINEL, dtype=np.uint64)
    )
    temporal2 = (
        prev2.astype(np.uint64, copy=False)
        if prev2 is not None
        else np.full(SITES, SENTINEL, dtype=np.uint64)
    )
    predicted = np.asarray(predictor_argmax[frame], dtype=np.uint64).reshape(-1)
    u64 = np.minimum(
        np.asarray(predictor_u_index[frame], dtype=np.uint64).reshape(-1) // 4,
        63,
    )
    prediction = predicted * 64 + u64
    if spec.kind == "order0":
        context = np.zeros(SITES, dtype=np.uint64)
    elif spec.kind == "previous_decode_symbol":
        symbols = current[order]
        context = np.empty(SITES, dtype=np.uint64)
        context[0] = SENTINEL
        context[1:] = symbols[:-1]
        return context, symbols
    elif spec.kind == "group":
        context = groups.astype(np.uint64, copy=False)
    elif spec.kind == "spatial_lu1":
        context = pack_context(
            (neighbour_values(current, plans[(-1, 0)]), neighbour_values(current, plans[(0, -1)]))
        )
    elif spec.kind == "spatial_lu12":
        context = pack_context(
            (
                neighbour_values(current, plans[(-1, 0)]),
                neighbour_values(current, plans[(0, -1)]),
                neighbour_values(current, plans[(-2, 0)]),
                neighbour_values(current, plans[(0, -2)]),
            )
        )
    elif spec.kind == "temporal1":
        context = temporal1
    elif spec.kind == "temporal2":
        context = pack_context((temporal1, temporal2))
    elif spec.kind == "joint_lu_prev1":
        context = pack_context(
            (
                neighbour_values(current, plans[(-1, 0)]),
                neighbour_values(current, plans[(0, -1)]),
                temporal1,
            )
        )
    elif spec.kind == "joint_lu_prev12":
        context = pack_context(
            (
                neighbour_values(current, plans[(-1, 0)]),
                neighbour_values(current, plans[(0, -1)]),
                temporal1,
                temporal2,
            )
        )
    elif spec.kind == "group_prev1":
        context = groups.astype(np.uint64) * CONTEXT_BASE + temporal1
    elif spec.kind == "group_lu_prev1":
        subcontext = pack_context(
            (
                neighbour_values(current, plans[(-1, 0)]),
                neighbour_values(current, plans[(0, -1)]),
                temporal1,
            )
        )
        context = groups.astype(np.uint64) * (CONTEXT_BASE**3) + subcontext
    elif spec.kind == "joint_spatial4_prev12":
        context = pack_context(
            (
                neighbour_values(current, plans[(-1, 0)]),
                neighbour_values(current, plans[(0, -1)]),
                neighbour_values(current, plans[(1, -1)]),
                neighbour_values(current, plans[(-1, -1)]),
                temporal1,
                temporal2,
            )
        )
    elif spec.kind == "hpac_argmax":
        context = predicted
    elif spec.kind == "hpac_argmax_u64":
        context = prediction
    elif spec.kind == "hpac_group":
        context = groups.astype(np.uint64) * (ALPHABET * 64) + prediction
    elif spec.kind == "hpac_lu_prev1":
        token_context = pack_context(
            (
                neighbour_values(current, plans[(-1, 0)]),
                neighbour_values(current, plans[(0, -1)]),
                temporal1,
            )
        )
        context = prediction * (CONTEXT_BASE**3) + token_context
    elif spec.kind == "hpac_feature_union":
        neighbours = np.stack(
            [
                neighbour_values(current, plans[offset])
                for offset in ((-1, 0), (0, -1), (1, -1), (-1, -1))
            ]
        )
        available = neighbours != SENTINEL
        agreeing = ((neighbours == predicted[None, :]) & available).sum(axis=0)
        spatial4 = np.where(available.any(axis=0), np.minimum(agreeing + 1, 5), 0)
        homogeneity = np.zeros(SITES, dtype=np.uint64)
        for symbol in range(ALPHABET):
            homogeneity += np.any((neighbours == symbol) & available, axis=0)
        homogeneity = np.minimum(homogeneity, 5)
        agree1 = ((temporal1 != SENTINEL) & (temporal1 == predicted)).astype(np.uint64)
        # The shipped corrector initializes prev2 to class 0 and flips
        # have_prev after frame 0.  Therefore frame 1 compares against that
        # zero plane; only frame 0 suppresses both temporal agreement bits.
        if frame == 0:
            agree2 = np.zeros(SITES, dtype=np.uint64)
        elif frame == 1:
            agree2 = (predicted == 0).astype(np.uint64)
        else:
            agree2 = (temporal2 == predicted).astype(np.uint64)
        temporal_agreement = agree1 * 2 + agree2
        run = run_before_frame(frame, tokens)
        boundary = (
            boundary_buckets(prev1)
            if prev1 is not None
            else np.full(SITES, 4, dtype=np.uint64)
        )
        context = prediction
        for part, base in (
            (spatial4, 6),
            (homogeneity, 6),
            (temporal_agreement, 4),
            (run, 8),
            (boundary, 5),
        ):
            context = context * base + part
    elif spec.kind == "site_prev12":
        site = np.arange(SITES, dtype=np.uint64)
        context = site * (CONTEXT_BASE**2) + pack_context((temporal1, temporal2))
    else:
        raise ValueError(f"unknown context kind {spec.kind}")
    return np.asarray(context, dtype=np.uint64), current


def plugin_entropy_bits(counts: np.ndarray) -> float:
    rows = np.asarray(counts, dtype=np.float64)
    totals = rows.sum(axis=1)
    positive = rows > 0
    terms = np.zeros_like(rows)
    np.log2(rows, out=terms, where=positive)
    total_logs = np.zeros_like(totals)
    np.log2(totals, out=total_logs, where=totals > 0)
    return float(np.sum(rows * (total_logs[:, None] - terms), where=positive))


def kt_prequential_bits(counts: np.ndarray) -> float:
    rows = np.asarray(counts, dtype=np.float64)
    totals = rows.sum(axis=1)
    active = totals > 0
    rows = rows[active]
    totals = totals[active]
    alpha = 0.5
    log_probability = (
        gammaln(ALPHABET * alpha)
        - gammaln(totals + ALPHABET * alpha)
        + np.sum(gammaln(rows + alpha) - gammaln(alpha), axis=1)
    )
    return float(-np.sum(log_probability) / math.log(2.0))


def latest_checkpoint(row_root: Path, spec: ContextSpec) -> tuple[int, np.ndarray] | None:
    receipts = sorted(row_root.glob("checkpoint_frame_*.json"))
    if not receipts:
        return None
    receipt_path = receipts[-1]
    checkpoint = json.loads(receipt_path.read_text())
    if (
        checkpoint.get("schema") != "ddm.cx3.context_checkpoint.v1"
        or checkpoint.get("context_name") != spec.name
        or checkpoint.get("domain") != spec.domain
        or checkpoint.get("measurement_script_sha256") != sha256_file(Path(__file__))
    ):
        raise RuntimeError(f"checkpoint identity drifted: {receipt_path}")
    verify_receipt(checkpoint["counts"])
    counts = np.load(checkpoint["counts"]["path"])
    return int(checkpoint["next_frame"]), np.asarray(counts, dtype=np.uint64)


def write_checkpoint(row_root: Path, spec: ContextSpec, next_frame: int, counts: np.ndarray) -> None:
    counts_path = row_root / f"checkpoint_frame_{next_frame:04d}.counts.npy"
    receipt = atomic_npy(counts_path, counts)
    atomic_json(
        row_root / f"checkpoint_frame_{next_frame:04d}.json",
        {
            "schema": "ddm.cx3.context_checkpoint.v1",
            "context_name": spec.name,
            "domain": spec.domain,
            "next_frame": next_frame,
            "counts": receipt,
            "source_token_sha256": EXPECTED[TOKENS][1],
            "measurement_script_sha256": sha256_file(Path(__file__)),
        },
    )


def retain_model_variants(row_root: Path, raw: bytes) -> dict[str, Any]:
    raw_receipt = atomic_bytes(row_root / "model.cx3m.raw", raw)
    variants = []
    for coder in ("brotli_q11", "lzma1_1m", "zlib9"):
        coded = compress_model(raw, coder)
        repeat = compress_model(raw, coder)
        if coded != repeat or decompress_model(coded, coder) != raw:
            raise RuntimeError(f"model coder failed deterministic round trip: {coder}")
        receipt = atomic_bytes(row_root / f"model.{coder}.bin", coded)
        repeat_receipt = atomic_bytes(row_root / f"model.{coder}.repeat.bin", repeat)
        variants.append(
            {
                "coder": coder,
                "bytes": len(coded),
                "payload": receipt,
                "repeat": repeat_receipt,
            }
        )
    winner = min(variants, key=lambda row: (row["bytes"], row["coder"]))
    return {"raw": raw_receipt, "variants": variants, "winner": winner}


def measure_spec(
    output: Path,
    spec: ContextSpec,
    tokens: np.memmap,
    predictor_argmax: np.ndarray,
    predictor_u_index: np.ndarray,
) -> dict[str, Any]:
    row_root = output / "retained/rows" / spec.name
    result_path = row_root / "ROW.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("schema") != "ddm.cx3.context_row.v1" or result.get("name") != spec.name:
            raise RuntimeError(f"retained row identity drifted: {result_path}")
        if result.get("measurement_script_sha256") != sha256_file(Path(__file__)):
            raise RuntimeError(f"retained row was produced by a different measurement script: {result_path}")
        for receipt in result["receipts"]:
            verify_receipt(receipt)
        return result

    started = time.time()
    groups, group_count = group_map(spec.patch, spec.delta)
    if spec.kind.startswith("group") and spec.domain not in {
        group_count,
        group_count * CONTEXT_BASE,
        group_count * CONTEXT_BASE**3,
    }:
        raise RuntimeError(f"group domain mismatch for {spec.name}")
    if spec.kind == "hpac_group" and spec.domain != group_count * ALPHABET * 64:
        raise RuntimeError(f"HPAC group domain mismatch for {spec.name}")
    plans = neighbour_plan(groups, spec.patch, spec.delta)
    order = np.argsort(groups, kind="stable")

    resumed = latest_checkpoint(row_root, spec)
    if resumed is None:
        next_frame = 0
        counts = np.zeros((spec.domain, ALPHABET), dtype=np.uint64)
    else:
        next_frame, counts = resumed
    flat_counts = counts.reshape(-1)
    for frame in range(next_frame, T):
        contexts, symbols = contexts_for_frame(
            spec,
            frame,
            tokens,
            groups,
            order,
            plans,
            predictor_argmax,
            predictor_u_index,
        )
        if contexts.shape != symbols.shape or contexts.size != SITES:
            raise RuntimeError(f"context shape mismatch for {spec.name} frame {frame}")
        if int(contexts.min()) < 0 or int(contexts.max()) >= spec.domain:
            raise RuntimeError(f"context outside domain for {spec.name} frame {frame}")
        keys = contexts * ALPHABET + symbols.astype(np.uint64, copy=False)
        if flat_counts.size <= 2_000_000:
            flat_counts += np.bincount(keys.astype(np.int64), minlength=flat_counts.size).astype(
                np.uint64
            )
        else:
            unique, frequency = np.unique(keys, return_counts=True)
            flat_counts[unique] += frequency.astype(np.uint64)
        next_value = frame + 1
        if next_value % CHECKPOINT_INTERVAL == 0 or next_value == T:
            write_checkpoint(row_root, spec, next_value, counts)

    if int(counts.sum()) != SYMBOLS:
        raise RuntimeError(f"count denominator mismatch for {spec.name}")
    active = np.flatnonzero(counts.sum(axis=1)).astype(np.uint64)
    active_counts = counts[active].astype(np.uint64, copy=True)
    active_receipt = atomic_npy(row_root / "active_context_ids.u64.npy", active)
    histogram_receipt = atomic_npy(row_root / "active_symbol_counts.u64.npy", active_counts)
    final_counts_receipt = atomic_npy(row_root / "final_dense_counts.u64.npy", counts)

    raw_model = serialize_sparse_model(spec.domain, active, active_counts)
    parsed_domain, parsed_active, parsed_counts = parse_sparse_model(raw_model)
    if (
        parsed_domain != spec.domain
        or not np.array_equal(parsed_active, active)
        or not np.array_equal(parsed_counts, active_counts)
    ):
        raise RuntimeError(f"serialized model does not invert for {spec.name}")
    model = retain_model_variants(row_root, raw_model)

    plugin_bits = plugin_entropy_bits(active_counts)
    kt_bits = kt_prequential_bits(active_counts)
    plugin_bytes = math.ceil(plugin_bits / 8.0)
    kt_bytes = math.ceil(kt_bits / 8.0)
    model_bytes = int(model["winner"]["bytes"])
    static_total = plugin_bytes + model_bytes
    ideal_best = min(static_total, kt_bytes)
    group_histogram = np.bincount(groups, minlength=group_count).astype(np.uint64)
    group_hist_receipt = atomic_npy(row_root / "group_cell_counts.u64.npy", group_histogram)
    receipts = [
        active_receipt,
        histogram_receipt,
        final_counts_receipt,
        group_hist_receipt,
        model["raw"],
        *[variant["payload"] for variant in model["variants"]],
        *[variant["repeat"] for variant in model["variants"]],
    ]
    result = {
        "schema": "ddm.cx3.context_row.v1",
        "name": spec.name,
        "kind": spec.kind,
        "conditioning": spec.conditioning,
        "scope_reduction": spec.scope_reduction,
        "symbols": SYMBOLS,
        "alphabet": ALPHABET,
        "domain_contexts": spec.domain,
        "active_contexts": int(len(active)),
        "patch": spec.patch,
        "delta": spec.delta,
        "group_count": group_count,
        "group_collision_summary": {
            "patch_cells": spec.patch * spec.patch,
            "groups": group_count,
            "collisions": spec.patch * spec.patch - group_count,
            "min_cells_per_group": int(group_histogram.min()),
            "max_cells_per_group": int(group_histogram.max()),
        },
        "plugin_ideal_data_bits": plugin_bits,
        "plugin_ideal_data_ceiling_bytes": plugin_bytes,
        "static_model_description": model,
        "static_model_description_bytes": model_bytes,
        "plugin_plus_model_bytes": static_total,
        "kt_prequential_ideal_bits": kt_bits,
        "kt_prequential_ideal_ceiling_bytes": kt_bytes,
        "kt_transmitted_model_bytes": 0,
        "best_ideal_model_cost_inclusive_bytes": ideal_best,
        "delta_best_ideal_vs_incumbent_stream_bytes": ideal_best - INCUMBENT_STREAM_BYTES,
        "bits_per_symbol_plugin": plugin_bits / SYMBOLS,
        "bits_per_symbol_kt": kt_bits / SYMBOLS,
        "source_token_sha256": EXPECTED[TOKENS][1],
        "measurement_script_sha256": sha256_file(Path(__file__)),
        "causality": (
            "all current-frame neighbours are admitted only when neighbour_group < current_group; "
            "temporal values read only earlier frames; previous_decode_symbol reads the prior event"
        ),
        "bound_scope": (
            "The plug-in row is an in-sample ideal data term plus an exact counted model payload. "
            "The KT row is an ideal prequential marginal length with zero transmitted state. "
            "Neither is a finite-precision coded token payload or a universal lower bound."
        ),
        "elapsed_seconds": time.time() - started,
        "receipts": receipts,
    }
    atomic_json(result_path, result)
    return result


def group_refit_specs() -> tuple[ContextSpec, ...]:
    specs = []
    for patch in (16, 32, 64, 128):
        for delta in (1, 2, 3, 4):
            group_count = (1 + delta) * patch - delta
            specs.append(
                ContextSpec(
                    f"group_refit_p{patch}_d{delta}_hpac_argmax_u64",
                    "hpac_group",
                    group_count * ALPHABET * 64,
                    f"candidate group g=(x mod {patch})+{delta}*(y mod {patch}) x retained learned-predictor argmax/confidence",
                    "borrowed-constant diagnostic with the predictor trace fixed; not a changed-order HPAC forward",
                    patch=patch,
                    delta=delta,
                )
            )
    return tuple(specs)


def retain_source_and_pins(output: Path) -> dict[str, Any]:
    pins = {
        str(path): verify_pin(path, expected_bytes, expected_sha)
        for path, (expected_bytes, expected_sha) in EXPECTED.items()
    }
    to2 = json.loads(TO2_RESULT.read_text())
    if to2["anatomy"]["token_stream_bytes"] != INCUMBENT_STREAM_BYTES:
        raise RuntimeError("TO2 incumbent token size drifted")
    if to2["anatomy"]["hpac_bytes"] != INCUMBENT_MODEL_BYTES:
        raise RuntimeError("TO2 incumbent model size drifted")
    fs2 = json.loads(FS2_RESULT.read_text())
    checkpoint = json.loads(CHECKPOINT_RECEIPT.read_text())
    if not fs2["positive_control"]["instrument_valid"]:
        raise RuntimeError("FS2 learned-predictor trace failed its original positive control")
    for key in ("corrected_quantized_logit_sha256", "corrected_cdf_input_sha256"):
        if fs2["positive_control"][key] != checkpoint["token_decoder"][key]:
            raise RuntimeError(f"FS2 learned-predictor trace does not match DX2 {key}")
    sources = []
    for source in SOURCE_FILES:
        payload = source.read_bytes()
        sources.append(
            {
                "original": file_receipt(source),
                "snapshot": atomic_bytes(output / "retained/source" / source.name, payload),
            }
        )
    inputs = []
    for source in (TO2_RESULT, CHECKPOINT_RECEIPT, FS2_RESULT):
        inputs.append(atomic_bytes(output / "retained/input" / source.name, source.read_bytes()))
    incumbent = {
        "archive_bytes": INCUMBENT_ARCHIVE_BYTES,
        "score": INCUMBENT_SCORE,
        "token_stream_bytes": INCUMBENT_STREAM_BYTES,
        "token_stream_bits_per_symbol": INCUMBENT_STREAM_BYTES * 8 / SYMBOLS,
        "hpac_model_blob_counted_bytes": INCUMBENT_MODEL_BYTES,
        "token_plus_hpac_bytes": INCUMBENT_STREAM_BYTES + INCUMBENT_MODEL_BYTES,
        "token_plus_hpac_bits_per_symbol": (INCUMBENT_STREAM_BYTES + INCUMBENT_MODEL_BYTES)
        * 8
        / SYMBOLS,
        "group_map": "g=(x mod 64)+2*(y mod 64), 190 groups, frame outermost, stable raster within group",
        "base_hpac_conditions": (
            "learned frame index embedding; full previous token frame through learned 3x3 convolution; "
            "already-decoded current-frame groups through masked 7x7, dilated-5x5 and dilated-3x3 "
            "patch-local convolutions; fixed local coordinates; optional learned frame scale and SPM enabled"
        ),
        "member_count": len(INCUMBENT_FAMILIES),
        "members": INCUMBENT_FAMILIES,
        "mixer_context": (
            "4,000 online weight cells: predicted class x previous-frame boundary bucket x "
            "same-site agreement t-1/t-2 x causal-neighbour homogeneity x 8-bin surprise bucket"
        ),
        "within_miss_context": (
            "separate MA1 relative-law table keyed by causal up, up-right, left and same-site previous-frame token; 6^4=1,296 cells"
        ),
        "adaptive_state_cost": (
            "zero transmitted bytes; empty generic tables are rebuilt causally by both encoder and receiver"
        ),
        "learned_predictor_trace": {
            "scope": (
                "FS2 was replayed on the identical cc10... token field and its pre-corrector "
                "HPAC logit/CDF digests equal DX2. Its retained coding-row argmax/confidence "
                "summaries predate DX2's final 70-byte corrector improvement, so they are a "
                "conditioning coordinate, not a claim to reproduce DX2's final probability row."
            ),
            "argmax": file_receipt(PREDICTOR_ARGMAX),
            "u_index": file_receipt(PREDICTOR_U_INDEX),
        },
    }
    receipt = {
        "schema": "ddm.cx3.inherited_state.v1",
        "pins": pins,
        "source_snapshots": sources,
        "input_snapshots": inputs,
        "incumbent": incumbent,
    }
    atomic_json(output / "INHERITED_STATE.json", receipt)
    return receipt


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    if usage.free < MIN_STORAGE_BYTES:
        raise RuntimeError(
            f"Vertigo storage preflight failed: {usage.free} free < {MIN_STORAGE_BYTES} required"
        )
    report = {
        "schema": "ddm.cx3.storage_preflight.v1",
        "tier": "/Volumes/VertigoDataTier/pact",
        "output": str(output),
        "free_bytes": usage.free,
        "required_free_bytes": MIN_STORAGE_BYTES,
        "apdatastore_for_new_receipts": False,
        "cleanup_policy": "retain all stage checkpoints, histograms, models, and losing variants",
    }
    attempt_root = output / "retained/preflight"
    attempt = len(tuple(attempt_root.glob("attempt_*.json"))) + 1
    report["receipt"] = atomic_json(
        attempt_root / f"attempt_{attempt:04d}.json", report
    )
    return report


def manifest_tree(output: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            rows.append(file_receipt(path))
    manifest = {
        "schema": "ddm.cx3.retention_manifest.v1",
        "root": str(output),
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    atomic_json(output / "MANIFEST.json", manifest)
    return manifest


def run_self_test() -> None:
    active = np.array([0, 4, 99], dtype=np.uint64)
    counts = np.array([[1, 2, 3, 4, 5], [0, 0, 7, 0, 1], [9, 0, 0, 0, 0]], dtype=np.uint64)
    payload = serialize_sparse_model(100, active, counts)
    domain, rebuilt_active, rebuilt_counts = parse_sparse_model(payload)
    if domain != 100 or not np.array_equal(active, rebuilt_active) or not np.array_equal(counts, rebuilt_counts):
        raise RuntimeError("CX3 model self-test failed")
    for coder in ("brotli_q11", "lzma1_1m", "zlib9"):
        if decompress_model(compress_model(payload, coder), coder) != payload:
            raise RuntimeError(f"CX3 coder self-test failed: {coder}")
    groups, count = group_map(64, 2)
    if count != 190 or int(groups[0]) != 0 or int(groups[2]) != 2 or int(groups[W]) != 2:
        raise RuntimeError("incumbent group-map self-test failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    run_self_test()
    if args.self_test:
        print(json.dumps({"self_test": "pass"}, sort_keys=True))
        return 0

    output = args.resume_from if args.resume_from is not None else args.output_root
    if output != DEFAULT_OUTPUT:
        raise SystemExit(f"CX3 output is pinned to {DEFAULT_OUTPUT}")
    if (output / "RESULT.json").exists() and (output / "MANIFEST.json").exists():
        manifest = json.loads((output / "MANIFEST.json").read_text())
        for receipt in manifest["artifacts"]:
            verify_receipt(receipt)
        result = json.loads((output / "RESULT.json").read_text())
        print(
            json.dumps(
                {
                    "resume_status": "already_complete_all_manifest_receipts_verified",
                    "result": file_receipt(output / "RESULT.json"),
                    "manifest": file_receipt(output / "MANIFEST.json"),
                    "best": result["best_ideal_model_cost_inclusive"]["name"],
                    "best_ideal_bytes": result["best_ideal_model_cost_inclusive"][
                        "best_ideal_model_cost_inclusive_bytes"
                    ],
                    "group_refit_triggered": result["group_refit_gate"]["triggered"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    started = time.time()
    storage = storage_preflight(output)
    inherited = retain_source_and_pins(output)
    tokens = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(T, H, W))
    predictor_argmax = np.load(PREDICTOR_ARGMAX, mmap_mode="r")
    predictor_u_index = np.load(PREDICTOR_U_INDEX, mmap_mode="r")
    if predictor_argmax.shape != (T, SITES) or predictor_argmax.dtype != np.uint8:
        raise RuntimeError("learned predictor argmax shape/dtype drifted")
    if predictor_u_index.shape != (T, SITES) or predictor_u_index.dtype != np.uint16:
        raise RuntimeError("learned predictor confidence shape/dtype drifted")

    rows = []
    for spec in BASE_SPECS:
        rows.append(measure_spec(output, spec, tokens, predictor_argmax, predictor_u_index))
        atomic_json(
            output / "retained/state" / f"base_ladder_{len(rows):02d}.json",
            {
                "schema": "ddm.cx3.state.v1",
                "phase": "base_ladder",
                "completed_rows": [row["name"] for row in rows],
                "next_row": BASE_SPECS[len(rows)].name if len(rows) < len(BASE_SPECS) else None,
            },
        )

    best_base = min(rows, key=lambda row: row["best_ideal_model_cost_inclusive_bytes"])
    material_saving = (
        INCUMBENT_STREAM_BYTES - best_base["best_ideal_model_cost_inclusive_bytes"]
    )
    group_refit_triggered = material_saving >= GROUP_REFIT_TRIGGER_SAVING
    refit_rows = []
    if group_refit_triggered:
        for spec in group_refit_specs():
            refit_rows.append(
                measure_spec(output, spec, tokens, predictor_argmax, predictor_u_index)
            )
            atomic_json(
                output / "retained/state" / f"group_refit_{len(refit_rows):02d}.json",
                {
                    "schema": "ddm.cx3.state.v1",
                    "phase": "group_refit",
                    "completed_rows": [row["name"] for row in rows + refit_rows],
                    "next_row": None,
                },
            )

    all_rows = rows + refit_rows
    best = min(all_rows, key=lambda row: row["best_ideal_model_cost_inclusive_bytes"])
    best_static = min(all_rows, key=lambda row: row["plugin_plus_model_bytes"])
    best_kt = min(all_rows, key=lambda row: row["kt_prequential_ideal_ceiling_bytes"])
    result = {
        "schema": "ddm.cx3.context_axis_ceiling.v1",
        "axis": "[macOS-CPU advisory / scorer-free lossless diagnostic]",
        "command": {"argv": sys.argv, "cwd": str(Path.cwd())},
        "environment": {
            "git_head_before_measurement": git_head(),
            "measurement_script": file_receipt(Path(__file__)),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "storage": storage,
        "inherited": inherited,
        "rows": all_rows,
        "best_ideal_model_cost_inclusive": best,
        "best_static_plugin_plus_counted_model": best_static,
        "best_zero_state_kt": best_kt,
        "group_refit_gate": {
            "trigger_saving_bytes": GROUP_REFIT_TRIGGER_SAVING,
            "base_best_saving_bytes": material_saving,
            "triggered": group_refit_triggered,
            "reason": (
                "ten percent of the 42,382 B campaign demand is the pre-registered materiality bar"
            ),
        },
        "demand_adjudication": {
            "campaign_demand_bytes": DEMAND_BYTES,
            "token_target_bytes": TOKEN_TARGET_BYTES,
            "best_ideal_total_bytes": best["best_ideal_model_cost_inclusive_bytes"],
            "best_ideal_saving_vs_113777": INCUMBENT_STREAM_BYTES
            - best["best_ideal_model_cost_inclusive_bytes"],
            "fraction_of_demand": (
                INCUMBENT_STREAM_BYTES - best["best_ideal_model_cost_inclusive_bytes"]
            )
            / DEMAND_BYTES,
            "falsifier_71395_reached_by_ideal_probe": best[
                "best_ideal_model_cost_inclusive_bytes"
            ]
            <= TOKEN_TARGET_BYTES,
            "finite_precision_payload_built": False,
            "scope": (
                "No entropy row is admitted as a shippable candidate: every row is an ideal "
                "log-loss diagnostic even though its static model bytes are exact retained payloads."
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(
        output / "STATE.json",
        {
            "schema": "ddm.cx3.state.v1",
            "phase": "complete",
            "completed_rows": [row["name"] for row in all_rows],
            "result": file_receipt(output / "RESULT.json"),
        },
    )
    manifest = manifest_tree(output)
    # Rehash every manifest-listed artifact after the complete result exists.
    for receipt in manifest["artifacts"]:
        verify_receipt(receipt)
    print(
        json.dumps(
            {
                "result": file_receipt(output / "RESULT.json"),
                "manifest": file_receipt(output / "MANIFEST.json"),
                "best": best["name"],
                "best_ideal_bytes": best["best_ideal_model_cost_inclusive_bytes"],
                "group_refit_triggered": group_refit_triggered,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
