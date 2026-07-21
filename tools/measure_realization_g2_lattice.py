#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit and measure the real seed-compose G2 RGB-lattice handoff.

The factor-2 integer solver consumes an RGB scorer plane.  The merged
seed-compose campaign preserved class-ID fields and cache-replay cell/tube
checks only.  This tool first walks the real n16/n64/n600 stages and records
that input-domain distinction.  Its optional real control then uses the exact
source-derived uint8 scorer planes that the canonical support-fill actually
accepts, realizes both camera frames, and measures them through native CPU
Torch.  The control is deliberately charged for both dense planes and remains
semantically unbound to the seed's class field; it can prove the downstream
lattice while refusing to counterfeit the missing zero-byte cells-to-RGB map.

Every pair stage and chunk checkpoint is immutable/resumable.  No camera or
plane tensor is persisted: the source cache is ZIP_STORED-mapped, one pair is
materialized at a time, and only hashes/metrics reach durable SSD evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math import warp_real_luma_frame0 as g1_warp  # noqa: E402
from tac.optimization.predict_project_receiver import (  # noqa: E402
    PROJECTED_RGB_PLANE_CUSTODY_SCHEMA,
    predict_cell_field,
    projected_plane_array_sha256,
    realize_projected_rgb_plane_camera_uint8,
)
from tac.optimization.predict_project_schema import (  # noqa: E402
    parse_constraint_seed,
    serialize_constraint_seed,
)
from tac.optimization.predictor_r2_missdelta import (  # noqa: E402
    AdaptiveStream,
    AdaptiveStreamDecoder,
)
from tac.optimization.predictor_upgrade_xi_chart import (  # noqa: E402
    load_g1_worldsheet_motion,
    relative_adjacent_xi,
)
from tac.optimization.resize_full_kernel import FullResizeKernel  # noqa: E402
from tac.optimization.seed_compose_b2 import GT_CACHE_SHA256  # noqa: E402

SCHEMA: Final = "realization_g2_lattice_receipt.v2"
PAIR_STAGE_SCHEMA: Final = "predict_project_pair_stage.v0"
HARD_ORACLE_SCHEMA: Final = "predict_project_hard_oracle_pair.v0"
SOURCE_CONTROL_STAGE_SCHEMA: Final = "realization_g2b_source_plane_pair.v1"
SOURCE_CONTROL_CONFIG_SCHEMA: Final = "realization_g2b_source_plane_config.v1"
INTERIOR_STAGE_SCHEMA: Final = "realization_g2c_interior_pair.v1"
INTERIOR_CONFIG_SCHEMA: Final = "realization_g2c_interior_config.v1"
INTERIOR_RECEIPT_SCHEMA: Final = "realization_g2c_interior_receipt.v1"
CONTEXTUAL_STAGE_SCHEMA: Final = "realization_g2d_predict_base_pair.v1"
CONTEXTUAL_CONFIG_SCHEMA: Final = "realization_g2d_predict_base_config.v1"
CONTEXTUAL_RECEIPT_SCHEMA: Final = "realization_g2d_predict_base_receipt.v1"
PREFIXES: Final = (16, 64, 600)
RGB_REALIZATION_FIELDS: Final = frozenset(
    {
        "projected_rgb_sha256",
        "camera_uint8_sha256",
        "factor2_verification",
        "projection_custody",
    }
)
POINTER: Final = "0.1910828242 [contest-CPU] UNMOVED"
AXIS: Final = "[macOS-CPU advisory]"
SCORER_HW: Final = (384, 512)
CAMERA_HW: Final = (874, 1164)
PAIR_COUNT: Final = 600
POSE_Q_SCALE: Final = 1_048_576

R1_FIXED_MAGNITUDE_PALETTE: Final = np.asarray(
    (
        (192, 192, 64),
        (64, 192, 192),
        (64, 192, 64),
        (192, 64, 192),
        (64, 64, 192),
    ),
    dtype=np.uint8,
)
R2_MAX_MARGIN_PALETTE: Final = np.asarray(
    (
        (153, 255, 51),
        (51, 255, 204),
        (0, 153, 0),
        (102, 204, 51),
        (0, 255, 153),
    ),
    dtype=np.uint8,
)
R2_CONTEXT_FREE_MARGINS: Final = (
    0.7793469429016113,
    -5.11375105381012,
    9.181995153427124,
    2.989253133535385,
    2.9624489545822144,
)
R3_MEMORY_PROTOTYPES: Final = np.asarray(
    (
        ((153, 255, 51), (51, 51, 0), (51, 0, 0), (51, 51, 51)),
        ((51, 255, 204), (153, 153, 153), (204, 204, 102), (204, 153, 102)),
        ((0, 153, 0), (0, 204, 51), (0, 255, 51), (102, 255, 0)),
        ((102, 204, 51), (204, 51, 255), (153, 51, 204), (255, 51, 255)),
        ((0, 255, 153), (153, 255, 102), (0, 0, 204), (51, 51, 255)),
    ),
    dtype=np.float64,
)
INTERIOR_RUNG_IDS: Final = (
    "R1_FIXED_MAGNITUDE",
    "R2_MAX_MARGIN",
    "R3_HOPFIELD_MEMORY_PROX",
    "R4_DYING_WRITE_EXCEPTIONS",
)
CONTEXTUAL_PROJECTION_BANDS: Final = (16, 32, 64, 128, 255)
CONTEXTUAL_LANE_ID: Final = "lane_realization_g2d_predict_base_578_20260721"
CONTEXTUAL_EXCEPTION_MAGIC: Final = b"G2DX1"
CONTEXTUAL_EXCEPTION_HEADER: Final = struct.Struct(">5sHIII")
CONTEXTUAL_SEED_BASELINE_BYTES: Final = 78_969
CONTEXTUAL_TARGET_BOX_BYTES: Final = 216_222


class RealizationAuditError(ValueError):
    """Missing, mixed, or falsely promoted G2 audit evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealizationAuditError(f"cannot read JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise RealizationAuditError(f"evidence must be one JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RealizationAuditError(f"{label} must be an exact nonnegative integer")
    return value


def stored_npy_memmap(npz_path: Path, key: str) -> np.memmap:
    """Map one unencrypted ZIP_STORED NPY member without inflating the cache."""

    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as exc:
            raise RealizationAuditError(f"cache lacks {member}") from exc
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise RealizationAuditError(f"cache member must be unencrypted ZIP_STORED: {member}")
        offset = int(info.header_offset)
    with npz_path.open("rb") as handle:
        handle.seek(offset)
        header = handle.read(30)
        if len(header) != 30:
            raise RealizationAuditError(f"truncated ZIP header: {member}")
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise RealizationAuditError(f"invalid ZIP local header: {member}")
        handle.seek(offset + 30 + int(fields[-2]) + int(fields[-1]))
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise RealizationAuditError(f"unsupported NPY version for {member}: {version}")
        data_offset = handle.tell()
    return np.memmap(
        npz_path,
        mode="r",
        dtype=dtype,
        shape=shape,
        offset=data_offset,
        order="F" if fortran else "C",
    )


def _load_real_cache(path: Path) -> dict[str, np.memmap]:
    if _sha256(path) != GT_CACHE_SHA256:
        raise RealizationAuditError("real n600 GT-cache SHA-256 mismatch")
    fields = {key: stored_npy_memmap(path, key) for key in ("n_pairs", "gt_f0", "gt_f1", "lstars", "gt_poses")}
    if int(np.asarray(fields["n_pairs"]).reshape(())) != PAIR_COUNT:
        raise RealizationAuditError("source-plane control requires exact real n600 cache")
    if fields["gt_f0"].shape != (PAIR_COUNT, *CAMERA_HW, 3) or fields["gt_f1"].shape != (
        PAIR_COUNT,
        *CAMERA_HW,
        3,
    ):
        raise RealizationAuditError("GT-cache camera geometry mismatch")
    if fields["lstars"].shape != (PAIR_COUNT, *SCORER_HW) or fields["gt_poses"].shape != (
        PAIR_COUNT,
        6,
    ):
        raise RealizationAuditError("GT-cache scorer/pose geometry mismatch")
    return fields


def _load_distortion_net(upstream: Path, threads: int) -> tuple[Any, Any, dict[str, Any]]:
    if threads < 1 or not (upstream / "modules.py").is_file():
        raise RealizationAuditError("native CPU-Torch scorer custody is unavailable")
    sys.path.insert(0, str(upstream))
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch.set_num_threads(threads)
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    net = DistortionNet().eval().to("cpu")
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    custody = {
        "implementation": "upstream.modules.DistortionNet.native_cpu_torch",
        "modules_path": str(upstream / "modules.py"),
        "modules_sha256": _sha256(upstream / "modules.py"),
        "segnet_weights_path": str(Path(segnet_sd_path)),
        "segnet_weights_sha256": _sha256(Path(segnet_sd_path)),
        "posenet_weights_path": str(Path(posenet_sd_path)),
        "posenet_weights_sha256": _sha256(Path(posenet_sd_path)),
        "threads": threads,
        "seed": 1234,
        "deterministic_algorithms": True,
    }
    return net, torch, custody


def _represented_cells(seed: Mapping[str, Any], pair_index: int) -> np.ndarray:
    represented = predict_cell_field(seed, pair_index)
    for row in seed["constraint_seeds"]:
        if row["time"] == pair_index and row["frame_index"] == 1:
            represented[row["y"], row["x"]] = row["cell_id"]
    return represented


def _exact_source_target_plane(operator: Any, camera: np.ndarray) -> np.ndarray:
    numerators, denominator = operator.apply_numerators(camera.astype(np.int64, copy=False))
    return np.clip(np.rint(numerators.astype(np.float64) / denominator), 0, 255).astype(np.uint8)


def _source_plane_custody(
    *,
    seed_sha256: str,
    rgb: np.ndarray,
    represented_cells: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema": PROJECTED_RGB_PLANE_CUSTODY_SCHEMA,
        "source_kind": "encoder_supplied_counted",
        "generator_id": "exact_rational_round_of_source_camera_control_not_cells_to_rgb",
        "seed_sha256": seed_sha256,
        "projected_rgb_sha256": projected_plane_array_sha256(rgb),
        "projected_cells_sha256": projected_plane_array_sha256(represented_cells),
        "additional_seed_bytes": int(rgb.nbytes),
        "decoder_scorer_invocations": 0,
    }


def _decoder_plane_custody(
    *,
    seed_sha256: str,
    rgb: np.ndarray,
    represented_cells: np.ndarray,
    rung_id: str,
    additional_seed_bytes: int = 0,
) -> dict[str, Any]:
    source_kind = "decoder_derived_from_seed" if additional_seed_bytes == 0 else "encoder_supplied_counted"
    return {
        "schema": PROJECTED_RGB_PLANE_CUSTODY_SCHEMA,
        "source_kind": source_kind,
        "generator_id": f"realization_g2c_{rung_id.lower()}_v1",
        "seed_sha256": seed_sha256,
        "projected_rgb_sha256": projected_plane_array_sha256(rgb),
        "projected_cells_sha256": projected_plane_array_sha256(represented_cells),
        "additional_seed_bytes": additional_seed_bytes,
        "decoder_scorer_invocations": 0,
    }


def _local_cell_context(cells: np.ndarray) -> np.ndarray:
    """Return three deterministic seed-cell context channels in [-1, 1]."""

    value = np.asarray(cells, dtype=np.float64)
    scale = 1.0 / 4.0
    horizontal = (np.roll(value, -1, axis=1) - np.roll(value, 1, axis=1)) * scale
    vertical = (np.roll(value, -1, axis=0) - np.roll(value, 1, axis=0)) * scale
    checker = (((np.indices(value.shape).sum(axis=0) & 1) * 2) - 1).astype(np.float64)
    return np.stack((horizontal, vertical, checker), axis=-1)


def interior_rgb_plane(cells: np.ndarray, rung_id: str) -> np.ndarray:
    """Decode one zero-byte, scorer-free RGB-plane formulation from class cells."""

    represented = np.asarray(cells)
    if represented.dtype != np.uint8 or represented.shape != SCORER_HW:
        raise RealizationAuditError(f"cells must be uint8 {SCORER_HW}")
    if np.any(represented >= len(R1_FIXED_MAGNITUDE_PALETTE)):
        raise RealizationAuditError("cells contain an out-of-range class ID")
    if rung_id == "R1_FIXED_MAGNITUDE":
        return R1_FIXED_MAGNITUDE_PALETTE[represented]
    if rung_id == "R2_MAX_MARGIN":
        return R2_MAX_MARGIN_PALETTE[represented]
    if rung_id != "R3_HOPFIELD_MEMORY_PROX":
        raise RealizationAuditError(f"unsupported zero-byte interior rung: {rung_id}")

    # One modern-Hopfield retrieval step.  The memories are frozen-scorer,
    # video-independent constant-tile probes; the query is the fixed-magnitude
    # code plus local cell context.  This is behaviorally distinct from either
    # palette and remains a generic decoder procedure with zero payload bytes.
    base = R1_FIXED_MAGNITUDE_PALETTE[represented].astype(np.float64)
    query = np.clip(base + 24.0 * _local_cell_context(represented), 0.0, 255.0)
    out = np.empty_like(query)
    beta = 8.0
    for class_id in range(R3_MEMORY_PROTOTYPES.shape[0]):
        mask = represented == class_id
        if not np.any(mask):
            continue
        memories = R3_MEMORY_PROTOTYPES[class_id]
        memory_unit = (memories - 127.5) / 127.5
        query_unit = (query[mask] - 127.5) / 127.5
        logits = beta * np.einsum("nc,kc->nk", query_unit, memory_unit) / 3.0
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        out[mask] = np.einsum("nk,kc->nc", weights, memories)
    return np.rint(np.clip(out, 0.0, 255.0)).astype(np.uint8)


def encode_dying_write_exceptions(
    constraints: Sequence[Mapping[str, Any]],
    survival_rows: Sequence[Mapping[str, Any]],
    source_rgb_plane: np.ndarray,
) -> bytes:
    """Encode only R3 dying-write ordinals and their counted RGB triplets."""

    if len(constraints) != len(survival_rows):
        raise RealizationAuditError("constraint/survival cardinality mismatch")
    dying = [index for index, row in enumerate(survival_rows) if row["survives"] is False]
    if not dying:
        return b""
    if len(dying) > 0xFFFF or any(index > 0xFFFF for index in dying):
        raise RealizationAuditError("dying-write exception stream exceeds u16 ordinal range")
    payload = bytearray(struct.pack(">H", len(dying)))
    for index in dying:
        row = constraints[index]
        rgb = np.asarray(source_rgb_plane[row["y"], row["x"]], dtype=np.uint8)
        payload.extend(struct.pack(">HBBB", index, int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return bytes(payload)


def apply_dying_write_exceptions(
    base_rgb_plane: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
    payload: bytes,
) -> tuple[np.ndarray, list[int]]:
    """Strict parse-back of the R4 ordinal/RGB stream."""

    out = np.asarray(base_rgb_plane, dtype=np.uint8).copy()
    if not payload:
        return out, []
    if len(payload) < 2:
        raise RealizationAuditError("truncated dying-write exception stream")
    count = struct.unpack_from(">H", payload, 0)[0]
    if len(payload) != 2 + 5 * count:
        raise RealizationAuditError("dying-write exception stream length mismatch")
    ordinals: list[int] = []
    for offset in range(count):
        ordinal, r, g, b = struct.unpack_from(">HBBB", payload, 2 + 5 * offset)
        if ordinal >= len(constraints) or (ordinals and ordinal <= ordinals[-1]):
            raise RealizationAuditError("dying-write exception ordinals are noncanonical")
        row = constraints[ordinal]
        out[row["y"], row["x"]] = (r, g, b)
        ordinals.append(ordinal)
    return out, ordinals


def _encode_varint_adaptive(stream: AdaptiveStream, value: int) -> None:
    if value < 0:
        raise RealizationAuditError("adaptive varint must be nonnegative")
    index = 0
    while True:
        byte = value & 0x7F
        value >>= 7
        stream.encode(byte | (0x80 if value else 0), min(index, 2))
        index += 1
        if not value:
            return


def _decode_varint_adaptive(stream: AdaptiveStreamDecoder) -> int:
    value = 0
    shift = 0
    for index in range(10):
        byte = stream.decode(min(index, 2))
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value
        shift += 7
    raise RealizationAuditError("adaptive varint exceeds uint64")


def encode_contextual_rgb_exceptions(
    base_rgb_plane: np.ndarray,
    projected_rgb_plane: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
) -> tuple[bytes, list[int]]:
    """Encode changed constraint ordinals and RGB values with the #557 coder.

    Locations are receiver-known from the canonical seed constraint order.  The
    stream therefore pays only ordinal gaps and final RGB symbols, not absolute
    coordinates.  Re-encoding is deterministic because both adaptive models
    start from Laplace-one counts and use only causal decoded context.
    """

    base = np.asarray(base_rgb_plane)
    projected = np.asarray(projected_rgb_plane)
    if base.dtype != np.uint8 or projected.dtype != np.uint8 or base.shape != (*SCORER_HW, 3):
        raise RealizationAuditError("contextual exception planes must be uint8 scorer RGB")
    if projected.shape != base.shape:
        raise RealizationAuditError("contextual exception plane geometry mismatch")
    changed: list[int] = []
    seen_sites: set[tuple[int, int]] = set()
    for ordinal, row in enumerate(constraints):
        site = (int(row["y"]), int(row["x"]))
        if site in seen_sites:
            raise RealizationAuditError("contextual constraints contain duplicate RGB sites")
        seen_sites.add(site)
        if not np.array_equal(base[site], projected[site]):
            changed.append(ordinal)
    gap_stream = AdaptiveStream(256)
    rgb_stream = AdaptiveStream(256)
    previous = -1
    for ordinal in changed:
        _encode_varint_adaptive(gap_stream, ordinal - previous - 1)
        row = constraints[ordinal]
        class_id = int(row["cell_id"])
        y, x = int(row["y"]), int(row["x"])
        for channel, value in enumerate(projected[y, x]):
            rgb_stream.encode(int(value), class_id * 3 + channel)
        previous = ordinal
    gap_payload = gap_stream.finish()
    rgb_payload = rgb_stream.finish()
    checksum = zlib.crc32(gap_payload + rgb_payload) & 0xFFFFFFFF
    return (
        CONTEXTUAL_EXCEPTION_HEADER.pack(
            CONTEXTUAL_EXCEPTION_MAGIC,
            len(changed),
            len(gap_payload),
            len(rgb_payload),
            checksum,
        )
        + gap_payload
        + rgb_payload,
        changed,
    )


def apply_contextual_rgb_exceptions(
    base_rgb_plane: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
    payload: bytes,
) -> tuple[np.ndarray, list[int]]:
    """Strictly parse one contextual ordinal/RGB exception stream."""

    base = np.asarray(base_rgb_plane)
    if base.dtype != np.uint8 or base.shape != (*SCORER_HW, 3):
        raise RealizationAuditError("contextual exception base must be uint8 scorer RGB")
    if len(payload) < CONTEXTUAL_EXCEPTION_HEADER.size:
        raise RealizationAuditError("contextual exception stream is truncated")
    magic, count, gap_size, rgb_size, checksum = CONTEXTUAL_EXCEPTION_HEADER.unpack_from(payload)
    if magic != CONTEXTUAL_EXCEPTION_MAGIC:
        raise RealizationAuditError("contextual exception magic mismatch")
    expected = CONTEXTUAL_EXCEPTION_HEADER.size + gap_size + rgb_size
    if len(payload) != expected:
        raise RealizationAuditError("contextual exception stream length mismatch")
    gap_payload = payload[CONTEXTUAL_EXCEPTION_HEADER.size : CONTEXTUAL_EXCEPTION_HEADER.size + gap_size]
    rgb_payload = payload[CONTEXTUAL_EXCEPTION_HEADER.size + gap_size :]
    if zlib.crc32(gap_payload + rgb_payload) & 0xFFFFFFFF != checksum:
        raise RealizationAuditError("contextual exception checksum mismatch")
    if count and (not gap_payload or not rgb_payload):
        raise RealizationAuditError("nonempty contextual exception lacks coded payload")
    if not count and (gap_payload or rgb_payload):
        raise RealizationAuditError("empty contextual exception has trailing coded payload")
    out = base.copy()
    if not count:
        return out, []
    gaps = AdaptiveStreamDecoder(gap_payload, 256)
    rgbs = AdaptiveStreamDecoder(rgb_payload, 256)
    ordinals: list[int] = []
    previous = -1
    for _ in range(count):
        ordinal = previous + 1 + _decode_varint_adaptive(gaps)
        if ordinal >= len(constraints):
            raise RealizationAuditError("contextual exception ordinal is out of range")
        row = constraints[ordinal]
        class_id = int(row["cell_id"])
        y, x = int(row["y"]), int(row["x"])
        out[y, x] = [rgbs.decode(class_id * 3 + channel) for channel in range(3)]
        ordinals.append(ordinal)
        previous = ordinal
    return out, ordinals


def contextual_advected_rgb_plane(
    previous_decoded_rgb_plane: np.ndarray,
    xi: np.ndarray,
    geom: g1_warp.GroundHomographyGeom,
) -> np.ndarray:
    """G1-advect previously decoded real RGB, then apply the uint8 knife edge."""

    previous = np.asarray(previous_decoded_rgb_plane)
    if previous.dtype != np.uint8 or previous.shape != (*SCORER_HW, 3):
        raise RealizationAuditError("contextual predictor requires previous decoded scorer RGB")
    warped = g1_warp.warp_frame0_native_numpy(previous, np.asarray(xi, dtype=np.float64), geom)
    return np.clip(np.rint(warped), 0, 255).astype(np.uint8)


def contextual_banded_projection(
    base_rgb_plane: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
    violated_ordinals: Sequence[int],
    band: int,
) -> np.ndarray:
    """Move only violated sites toward their max-margin prototypes by L-inf band."""

    if band not in CONTEXTUAL_PROJECTION_BANDS:
        raise RealizationAuditError("contextual projection band is not preregistered")
    out = np.asarray(base_rgb_plane, dtype=np.uint8).copy()
    if out.shape != (*SCORER_HW, 3):
        raise RealizationAuditError("contextual projection base geometry mismatch")
    if list(violated_ordinals) != sorted({int(value) for value in violated_ordinals}):
        raise RealizationAuditError("violated ordinals must be sorted and unique")
    for ordinal in violated_ordinals:
        if not 0 <= ordinal < len(constraints):
            raise RealizationAuditError("violated ordinal is out of range")
        row = constraints[ordinal]
        class_id = int(row["cell_id"])
        if not 0 <= class_id < len(R2_MAX_MARGIN_PALETTE):
            raise RealizationAuditError("contextual projection class ID is out of range")
        y, x = int(row["y"]), int(row["x"])
        base = out[y, x].astype(np.int16)
        target = R2_MAX_MARGIN_PALETTE[class_id].astype(np.int16)
        out[y, x] = np.clip(base + np.clip(target - base, -band, band), 0, 255).astype(np.uint8)
    return out


def _margin_bucket(value: float) -> str:
    if value <= 0.0:
        return "nonpositive"
    if value <= 1.0:
        return "positive_le_1"
    if value <= 4.0:
        return "positive_1_to_4"
    return "positive_gt_4"


def _hard_oracle_interior(
    net: Any,
    torch: Any,
    frame0: np.ndarray,
    frame1: np.ndarray,
    target_cells: np.ndarray,
    represented_cells: np.ndarray,
    target_pose: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    """Hard oracle plus target-class margins at each declared semantic write."""

    pair = np.stack((frame0, frame1), axis=0)[None]
    tensor = torch.from_numpy(np.ascontiguousarray(pair)).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        logits_tensor = net.segnet(net.segnet.preprocess_input(tensor))[0]
        argmax = logits_tensor.argmax(dim=0).cpu().numpy().astype(np.uint8)
        logits = logits_tensor.cpu().numpy().astype(np.float64)
        pose_output = net.posenet(net.posenet.preprocess_input(tensor))
        pose_tensor = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
        pose6 = pose_tensor[0, :6].cpu().numpy().astype(np.float64)
    pose_q = np.rint(pose6 * POSE_Q_SCALE).astype(np.int64)
    tubes = [row["pose_tube"] for row in constraints if row["pose_tube"] is not None]
    if not tubes:
        raise RealizationAuditError("pair has no declared pose tube")
    outside = []
    for tube in tubes:
        lower = np.asarray(tube["lower_q"], dtype=np.int64)
        upper = np.asarray(tube["upper_q"], dtype=np.int64)
        outside.append(np.maximum(lower - pose_q, 0) + np.maximum(pose_q - upper, 0))
    best_outside = min(outside, key=lambda value: float(np.sum(value.astype(np.float64) ** 2)))

    writes: list[dict[str, Any]] = []
    for ordinal, constraint in enumerate(constraints):
        class_id = int(constraint["cell_id"])
        y, x = int(constraint["y"]), int(constraint["x"])
        rival = float(np.max(np.delete(logits[:, y, x], class_id)))
        margin = float(logits[class_id, y, x] - rival)
        writes.append(
            {
                "ordinal": ordinal,
                "class_id": class_id,
                "stratum": str(constraint["stratum"]),
                "survives": int(argmax[y, x]) == class_id,
                "target_logit_margin": margin,
                "margin_bucket": _margin_bucket(margin),
            }
        )
    hard = {
        "d_seg_realized_vs_frozen_target": float(np.mean(argmax != target_cells)),
        "d_seg_description_vs_frozen_target": float(np.mean(represented_cells != target_cells)),
        "d_seg_realized_argmax_vs_description": float(np.mean(argmax != represented_cells)),
        "realized_argmax_equals_description": bool(np.array_equal(argmax, represented_cells)),
        "all_declared_writes_survive": all(row["survives"] is True for row in writes),
        "d_pose_realized_vs_frozen_target": float(np.mean((pose6 - target_pose) ** 2)),
        "d_pose_realized_outside_declared_tube": float(np.mean((best_outside.astype(np.float64) / POSE_Q_SCALE) ** 2)),
        "pose_within_declared_tube": bool(np.all(best_outside == 0)),
        "pose6": pose6.tolist(),
        "realized_argmax_sha256": projected_plane_array_sha256(argmax),
    }
    return hard, argmax, writes


def _seg_write_oracle(
    net: Any,
    torch: Any,
    frame0: np.ndarray,
    frame1: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Frozen SegNet-only encoder oracle for contextual band selection."""

    pair = np.stack((frame0, frame1), axis=0)[None]
    tensor = torch.from_numpy(np.ascontiguousarray(pair)).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        logits_tensor = net.segnet(net.segnet.preprocess_input(tensor))[0]
        argmax = logits_tensor.argmax(dim=0).cpu().numpy().astype(np.uint8)
        logits = logits_tensor.cpu().numpy().astype(np.float64)
    writes: list[dict[str, Any]] = []
    for ordinal, constraint in enumerate(constraints):
        class_id = int(constraint["cell_id"])
        y, x = int(constraint["y"]), int(constraint["x"])
        rival = float(np.max(np.delete(logits[:, y, x], class_id)))
        margin = float(logits[class_id, y, x] - rival)
        writes.append(
            {
                "ordinal": ordinal,
                "class_id": class_id,
                "stratum": str(constraint["stratum"]),
                "survives": int(argmax[y, x]) == class_id,
                "target_logit_margin": margin,
                "margin_bucket": _margin_bucket(margin),
            }
        )
    return argmax, writes


def _contextual_candidate_rank(
    writes: Sequence[Mapping[str, Any]],
    *,
    changed_sites: int,
    band: int,
) -> tuple[int, int, float, float, int, int]:
    margins = [float(row["target_logit_margin"]) for row in writes]
    surviving = sum(row["survives"] is True for row in writes)
    positive = sum(value > 0.0 for value in margins)
    minimum = min(margins, default=float("inf"))
    return (
        surviving,
        positive,
        minimum,
        float(sum(margins)),
        -changed_sites,
        -band,
    )


def _hard_oracle(
    net: Any,
    torch: Any,
    frame0: np.ndarray,
    frame1: np.ndarray,
    target_cells: np.ndarray,
    represented_cells: np.ndarray,
    target_pose: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    pair = np.stack((frame0, frame1), axis=0)[None]
    tensor = torch.from_numpy(np.ascontiguousarray(pair)).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        logits = net.segnet(net.segnet.preprocess_input(tensor))[0]
        argmax = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
        pose_output = net.posenet(net.posenet.preprocess_input(tensor))
        pose_tensor = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
        pose6 = pose_tensor[0, :6].cpu().numpy().astype(np.float64)
    pose_q = np.rint(pose6 * POSE_Q_SCALE).astype(np.int64)
    tubes = [row["pose_tube"] for row in constraints if row["pose_tube"] is not None]
    if not tubes:
        raise RealizationAuditError("pair has no declared pose tube")
    outside = []
    for tube in tubes:
        lower = np.asarray(tube["lower_q"], dtype=np.int64)
        upper = np.asarray(tube["upper_q"], dtype=np.int64)
        outside.append(np.maximum(lower - pose_q, 0) + np.maximum(pose_q - upper, 0))
    best_outside = min(outside, key=lambda value: float(np.sum(value.astype(np.float64) ** 2)))
    return {
        "d_seg_realized_vs_frozen_target": float(np.mean(argmax != target_cells)),
        "d_seg_description_vs_frozen_target": float(np.mean(represented_cells != target_cells)),
        "d_seg_realized_argmax_vs_description": float(np.mean(argmax != represented_cells)),
        "realized_argmax_equals_description": bool(np.array_equal(argmax, represented_cells)),
        "d_pose_realized_vs_frozen_target": float(np.mean((pose6 - target_pose) ** 2)),
        "d_pose_realized_outside_declared_tube": float(np.mean((best_outside.astype(np.float64) / POSE_Q_SCALE) ** 2)),
        "pose_within_declared_tube": bool(np.all(best_outside == 0)),
        "pose6": pose6.tolist(),
        "realized_argmax_sha256": projected_plane_array_sha256(argmax),
    }, argmax


def _write_survival_rows(
    constraints: Sequence[Mapping[str, Any]],
    argmax: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for constraint in constraints:
        survives = int(argmax[constraint["y"], constraint["x"]]) == int(constraint["cell_id"])
        rows.append(
            {
                "class_id": int(constraint["cell_id"]),
                "stratum": str(constraint["stratum"]),
                "survives": survives,
            }
        )
    return rows


def _load_source_control_stages(stage_dir: Path, config_sha256: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not stage_dir.exists():
        return rows
    for path in sorted(stage_dir.glob("pair_*.json")):
        row = _load_json(path)
        pair_index = row.get("pair_index")
        if (
            row.get("schema") != SOURCE_CONTROL_STAGE_SCHEMA
            or row.get("config_sha256") != config_sha256
            or isinstance(pair_index, bool)
            or not isinstance(pair_index, int)
            or pair_index in rows
        ):
            raise RealizationAuditError(f"source-control resume custody mismatch: {path}")
        rows[pair_index] = row
    return rows


def summarize_source_control_prefix(prefix: int, stage_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one measured real source-plane prefix without semantic promotion."""

    if prefix not in PREFIXES or len(stage_rows) != prefix:
        raise RealizationAuditError("source-control prefix is not n16/n64/n600")
    if [row.get("pair_index") for row in stage_rows] != list(range(prefix)):
        raise RealizationAuditError("source-control prefix is not contiguous from zero")
    if any(row.get("schema") != SOURCE_CONTROL_STAGE_SCHEMA for row in stage_rows):
        raise RealizationAuditError("source-control stage schema mismatch")

    by_class: Counter[int] = Counter()
    by_class_survives: Counter[int] = Counter()
    by_stratum: Counter[str] = Counter()
    by_stratum_survives: Counter[str] = Counter()
    for stage in stage_rows:
        for write in stage["declared_write_survival"]:
            class_id, stratum = int(write["class_id"]), str(write["stratum"])
            by_class[class_id] += 1
            by_stratum[stratum] += 1
            if write["survives"] is True:
                by_class_survives[class_id] += 1
                by_stratum_survives[stratum] += 1

    def survival_rows(total: Counter[Any], surviving: Counter[Any], key: str) -> list[dict[str, Any]]:
        return [
            {
                key: identity,
                "declared_writes": count,
                "surviving_writes": surviving[identity],
                "dying_writes": count - surviving[identity],
                "survival_fraction": surviving[identity] / count,
            }
            for identity, count in sorted(total.items(), key=lambda item: str(item[0]))
        ]

    hard = [row["hard_oracle"] for row in stage_rows]
    timing_keys = tuple(stage_rows[0]["timings_seconds"])
    timing_sums = {key: float(sum(float(row["timings_seconds"][key]) for row in stage_rows)) for key in timing_keys}
    return {
        "schema": "realization_g2b_source_plane_prefix.v1",
        "n": prefix,
        "pair_count": prefix,
        "uint8_factor2_exact_pair_count": sum(row["uint8_factor2_exact"] is True for row in stage_rows),
        "uint8_factor2_exact_fraction": float(np.mean([row["uint8_factor2_exact"] for row in stage_rows])),
        "double_decode_identical_pair_count": sum(row["double_decode_identical"] is True for row in stage_rows),
        "semantic_cells_to_rgb_exact_pair_count": sum(
            row["hard_oracle"]["realized_argmax_equals_description"] is True for row in stage_rows
        ),
        "mean_d_seg_realized_vs_frozen_target": float(
            np.mean([row["d_seg_realized_vs_frozen_target"] for row in hard])
        ),
        "mean_d_seg_description_vs_frozen_target": float(
            np.mean([row["d_seg_description_vs_frozen_target"] for row in hard])
        ),
        "mean_d_seg_realized_argmax_vs_description": float(
            np.mean([row["d_seg_realized_argmax_vs_description"] for row in hard])
        ),
        "mean_d_pose_realized_vs_frozen_target": float(
            np.mean([row["d_pose_realized_vs_frozen_target"] for row in hard])
        ),
        "mean_d_pose_realized_outside_declared_tube": float(
            np.mean([row["d_pose_realized_outside_declared_tube"] for row in hard])
        ),
        "pose_within_declared_tube_pair_count": sum(row["pose_within_declared_tube"] is True for row in hard),
        "additional_seed_bytes_per_pair": int(stage_rows[0]["additional_seed_bytes"]),
        "additional_seed_bytes_total": int(sum(row["additional_seed_bytes"] for row in stage_rows)),
        "zero_added_seed_byte_target_met": all(row["additional_seed_bytes"] == 0 for row in stage_rows),
        "by_class": survival_rows(by_class, by_class_survives, "class_id"),
        "by_stratum": survival_rows(by_stratum, by_stratum_survives, "stratum"),
        "timings_seconds_sum": timing_sums,
        "timings_seconds_mean_per_pair": {key: value / prefix for key, value in timing_sums.items()},
        "status": "MEASURED_SOURCE_RGB_CONTROL_NOT_SEED_RECEIVER",
        "verdict_scope": (
            "exact source-derived two-plane RGB control; proves RGB-plane support-fill/lattice and native scorer "
            "only, not a decoder-derived cells-to-RGB seed path"
        ),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }


def run_source_plane_control(
    *,
    seed_path: Path,
    gt_cache_path: Path,
    upstream: Path,
    output_root: Path,
    chunk_size: int,
    threads: int,
) -> dict[str, Any]:
    """Run/resume the charged source-RGB control for all 600 real pairs."""

    if chunk_size < 1 or threads < 1:
        raise RealizationAuditError("chunk size and CPU threads must be positive")
    seed_bytes = seed_path.read_bytes()
    seed = parse_constraint_seed(seed_bytes)
    if serialize_constraint_seed(seed) != seed_bytes:
        raise RealizationAuditError("seed is not canonical on parse-back")
    seed_sha256 = hashlib.sha256(seed_bytes).hexdigest()
    cache = _load_real_cache(gt_cache_path)
    net, torch, scorer_custody = _load_distortion_net(upstream, threads)
    kernel = FullResizeKernel.build()
    implementation_paths = (
        REPO / "src/tac/optimization/predict_project_receiver.py",
        REPO / "tools/measure_realization_g2_lattice.py",
    )
    config = {
        "schema": SOURCE_CONTROL_CONFIG_SCHEMA,
        "seed_sha256": seed_sha256,
        "gt_cache_sha256": GT_CACHE_SHA256,
        "scorer_custody": scorer_custody,
        "implementation_sources": {str(path.relative_to(REPO)): _sha256(path) for path in implementation_paths},
        "chunk_size": chunk_size,
        "pair_count": PAIR_COUNT,
        "axis": AXIS,
        "control_input": "exact_rational_round_of_source_gt_f0_gt_f1",
        "semantic_cells_to_rgb_claim": False,
    }
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    root = output_root / "source_plane_control"
    stage_dir = root / "stages"
    rows = _load_source_control_stages(stage_dir, config_sha256)
    resumed_pairs = len(rows)
    constraints_by_pair: dict[int, list[Mapping[str, Any]]] = {pair: [] for pair in range(PAIR_COUNT)}
    for constraint in seed["constraint_seeds"]:
        if constraint["frame_index"] == 1:
            constraints_by_pair[int(constraint["time"])].append(constraint)

    for chunk_begin in range(0, PAIR_COUNT, chunk_size):
        chunk_end = min(PAIR_COUNT, chunk_begin + chunk_size)
        for pair_index in range(chunk_begin, chunk_end):
            if pair_index in rows:
                continue
            started = time.perf_counter()
            clock = time.perf_counter()
            source0 = np.asarray(cache["gt_f0"][pair_index], dtype=np.uint8).copy()
            source1 = np.asarray(cache["gt_f1"][pair_index], dtype=np.uint8).copy()
            target_cells = np.asarray(cache["lstars"][pair_index], dtype=np.uint8).copy()
            target_pose = np.asarray(cache["gt_poses"][pair_index], dtype=np.float64).copy()
            load_seconds = time.perf_counter() - clock

            clock = time.perf_counter()
            represented = _represented_cells(seed, pair_index)
            cell_decode_seconds = time.perf_counter() - clock

            clock = time.perf_counter()
            plane0 = _exact_source_target_plane(kernel.operator, source0)
            plane1 = _exact_source_target_plane(kernel.operator, source1)
            plane_projection_seconds = time.perf_counter() - clock

            clock = time.perf_counter()
            realized0 = realize_projected_rgb_plane_camera_uint8(
                plane0,
                represented,
                _source_plane_custody(seed_sha256=seed_sha256, rgb=plane0, represented_cells=represented),
                kernel=kernel,
            )
            realized1 = realize_projected_rgb_plane_camera_uint8(
                plane1,
                represented,
                _source_plane_custody(seed_sha256=seed_sha256, rgb=plane1, represented_cells=represented),
                kernel=kernel,
            )
            second0 = realize_projected_rgb_plane_camera_uint8(
                plane0,
                represented,
                _source_plane_custody(seed_sha256=seed_sha256, rgb=plane0, represented_cells=represented),
                kernel=kernel,
            )
            second1 = realize_projected_rgb_plane_camera_uint8(
                plane1,
                represented,
                _source_plane_custody(seed_sha256=seed_sha256, rgb=plane1, represented_cells=represented),
                kernel=kernel,
            )
            double_equal = bool(
                np.array_equal(realized0["frame"], second0["frame"])
                and np.array_equal(realized1["frame"], second1["frame"])
            )
            lattice_seconds = time.perf_counter() - clock

            clock = time.perf_counter()
            hard, actual_argmax = _hard_oracle(
                net,
                torch,
                realized0["frame"],
                realized1["frame"],
                target_cells,
                represented,
                target_pose,
                constraints_by_pair[pair_index],
            )
            hard_seconds = time.perf_counter() - clock
            pair_exact = bool(
                realized0["factor2_verification"]["certified_exact"]
                and realized1["factor2_verification"]["certified_exact"]
            )
            if not pair_exact or not double_equal:
                raise RealizationAuditError(f"pair {pair_index} lost exact/deterministic lattice custody")
            timings = {
                "source_cache_load": load_seconds,
                "seed_cell_decode": cell_decode_seconds,
                "source_plane_projection": plane_projection_seconds,
                "lattice_double_decode": lattice_seconds,
                "native_cpu_torch_hard_oracle": hard_seconds,
                "total": time.perf_counter() - started,
            }
            row = {
                "schema": SOURCE_CONTROL_STAGE_SCHEMA,
                "config_sha256": config_sha256,
                "pair_index": pair_index,
                "projected_rgb_frame0_sha256": realized0["projected_rgb_sha256"],
                "projected_rgb_frame1_sha256": realized1["projected_rgb_sha256"],
                "projected_cells_sha256": realized1["projected_cells_sha256"],
                "camera_frame0_sha256": realized0["camera_uint8_sha256"],
                "camera_frame1_sha256": realized1["camera_uint8_sha256"],
                "uint8_factor2_exact": pair_exact,
                "double_decode_identical": double_equal,
                "additional_seed_bytes": int(plane0.nbytes + plane1.nbytes),
                "dense_plane_payload_convention": "frame0_u8_HWC_C_order_then_frame1_u8_HWC_C_order_no_header_fixed_geometry",
                "semantic_binding": "UNBOUND_SOURCE_RGB_CONTROL_NOT_DERIVED_FROM_PROJECTED_CELLS",
                "hard_oracle": hard,
                "declared_write_survival": _write_survival_rows(constraints_by_pair[pair_index], actual_argmax),
                "timings_seconds": timings,
                "authority": f"MEASURED {AXIS}",
                "score_claim": False,
                "promotion_eligible": False,
            }
            path = stage_dir / f"pair_{pair_index:04d}.json"
            _atomic_json(path, row)
            rows[pair_index] = row
            del source0, source1, plane0, plane1, represented, target_cells
            del realized0, realized1, second0, second1, actual_argmax

        checkpoint = {
            "schema": "realization_g2b_source_plane_chunk_checkpoint.v1",
            "config_sha256": config_sha256,
            "completed_through_exclusive": chunk_end,
            "completed_pairs": len(rows),
            "all_pair_stages_preserved": True,
            "resumed_pairs_at_invocation_start": resumed_pairs,
        }
        _atomic_json(root / "checkpoints" / f"chunk_{chunk_begin:04d}_{chunk_end:04d}.json", checkpoint)

    ordered = [rows[index] for index in range(PAIR_COUNT)]
    prefixes = [summarize_source_control_prefix(prefix, ordered[:prefix]) for prefix in PREFIXES]
    for row in prefixes:
        path = root / "checkpoints" / f"prefix_n{row['n']}.json"
        _atomic_json(path, row)
        row["checkpoint_path"] = str(path)
        row["checkpoint_sha256"] = _sha256(path)
    control_receipt = {
        "schema": "realization_g2b_source_plane_control_receipt.v1",
        "config": config,
        "config_sha256": config_sha256,
        "resumed_pairs_at_invocation_start": resumed_pairs,
        "prefix_ladder": prefixes,
        "stage_root": str(stage_dir),
        "stage_count": len(ordered),
        "stage_first_sha256": _sha256(stage_dir / "pair_0000.json"),
        "stage_last_sha256": _sha256(stage_dir / "pair_0599.json"),
        "automatic_disk_hygiene": (
            "ZIP_STORED mmap plus one-pair camera tensors; atomic temp JSON removed after replace; "
            "no RGB/camera bulk persisted"
        ),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }
    _atomic_json(root / "receipt.json", control_receipt)
    return control_receipt


def _load_interior_stages(
    stage_dir: Path,
    *,
    rung_id: str,
    config_sha256: str,
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not stage_dir.exists():
        return rows
    for path in sorted(stage_dir.glob("pair_*.json")):
        row = _load_json(path)
        pair_index = row.get("pair_index")
        if (
            row.get("schema") != INTERIOR_STAGE_SCHEMA
            or row.get("rung_id") != rung_id
            or row.get("config_sha256") != config_sha256
            or isinstance(pair_index, bool)
            or not isinstance(pair_index, int)
            or pair_index in rows
        ):
            raise RealizationAuditError(f"interior-rung resume custody mismatch: {path}")
        rows[pair_index] = row
    return rows


def _aggregate_survival(
    writes: Sequence[Mapping[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    totals: Counter[Any] = Counter()
    surviving: Counter[Any] = Counter()
    for row in writes:
        identity = row[key]
        totals[identity] += 1
        if row["survives"] is True:
            surviving[identity] += 1
    return [
        {
            key: identity,
            "declared_writes": count,
            "surviving_writes": surviving[identity],
            "dying_writes": count - surviving[identity],
            "survival_fraction": surviving[identity] / count,
        }
        for identity, count in sorted(totals.items(), key=lambda item: str(item[0]))
    ]


def summarize_interior_prefix(
    prefix: int,
    stage_rows: Sequence[Mapping[str, Any]],
    *,
    rung_id: str,
) -> dict[str, Any]:
    """Aggregate one real interior-fill prefix and its dying-write anatomy."""

    if prefix not in PREFIXES or len(stage_rows) != prefix:
        raise RealizationAuditError("interior prefix is not n16/n64/n600")
    if [row.get("pair_index") for row in stage_rows] != list(range(prefix)):
        raise RealizationAuditError("interior prefix is not contiguous from zero")
    if any(row.get("schema") != INTERIOR_STAGE_SCHEMA or row.get("rung_id") != rung_id for row in stage_rows):
        raise RealizationAuditError("interior stage schema/rung mismatch")
    writes = [write for row in stage_rows for write in row["declared_write_survival"]]
    hard = [row["hard_oracle"] for row in stage_rows]
    timing_keys = tuple(stage_rows[0]["timings_seconds"])
    timing_sums = {key: float(sum(float(row["timings_seconds"][key]) for row in stage_rows)) for key in timing_keys}
    total_writes = len(writes)
    surviving_writes = sum(row["survives"] is True for row in writes)
    exception_records = [record for row in stage_rows for record in row["exception_stream"]["records"]]
    exception_by_stratum: Counter[str] = Counter(str(row["stratum"]) for row in exception_records)
    exception_headers = sum(2 for row in stage_rows if row["exception_stream"]["record_count"] > 0)
    return {
        "schema": "realization_g2c_interior_prefix.v1",
        "rung_id": rung_id,
        "n": prefix,
        "pair_count": prefix,
        "uint8_factor2_exact_pair_count": sum(row["uint8_factor2_exact"] is True for row in stage_rows),
        "uint8_factor2_exact_fraction": float(np.mean([row["uint8_factor2_exact"] for row in stage_rows])),
        "double_decode_identical_pair_count": sum(row["double_decode_identical"] is True for row in stage_rows),
        "semantic_cells_to_rgb_exact_pair_count": sum(
            row["realized_argmax_equals_description"] is True for row in hard
        ),
        "semantic_cells_to_rgb_exact_fraction": float(
            np.mean([row["realized_argmax_equals_description"] for row in hard])
        ),
        "all_declared_writes_survive_pair_count": sum(row["all_declared_writes_survive"] is True for row in hard),
        "all_declared_writes_survive_pair_fraction": float(
            np.mean([row["all_declared_writes_survive"] for row in hard])
        ),
        "declared_write_count": total_writes,
        "surviving_write_count": surviving_writes,
        "dying_write_count": total_writes - surviving_writes,
        "declared_write_survival_fraction": (surviving_writes / total_writes if total_writes else 1.0),
        "mean_d_seg_realized_vs_frozen_target": float(
            np.mean([row["d_seg_realized_vs_frozen_target"] for row in hard])
        ),
        "mean_d_seg_description_vs_frozen_target": float(
            np.mean([row["d_seg_description_vs_frozen_target"] for row in hard])
        ),
        "mean_d_seg_realized_argmax_vs_description": float(
            np.mean([row["d_seg_realized_argmax_vs_description"] for row in hard])
        ),
        "mean_d_pose_realized_vs_frozen_target": float(
            np.mean([row["d_pose_realized_vs_frozen_target"] for row in hard])
        ),
        "mean_d_pose_realized_outside_declared_tube": float(
            np.mean([row["d_pose_realized_outside_declared_tube"] for row in hard])
        ),
        "pose_within_declared_tube_pair_count": sum(row["pose_within_declared_tube"] is True for row in hard),
        "additional_seed_bytes_total": int(sum(row["additional_seed_bytes"] for row in stage_rows)),
        "zero_added_seed_byte_target_met": all(row["additional_seed_bytes"] == 0 for row in stage_rows),
        "by_class": _aggregate_survival(writes, "class_id"),
        "by_stratum": _aggregate_survival(writes, "stratum"),
        "by_margin_bucket": _aggregate_survival(writes, "margin_bucket"),
        "R4_exception_pricing": {
            "record_count": len(exception_records),
            "record_bytes": 5 * len(exception_records),
            "nonempty_pair_header_bytes": exception_headers,
            "total_bytes": 5 * len(exception_records) + exception_headers,
            "by_stratum": [
                {
                    "stratum": stratum,
                    "records": count,
                    "record_bytes": 5 * count,
                }
                for stratum, count in sorted(exception_by_stratum.items())
            ],
        },
        "timings_seconds_sum": timing_sums,
        "timings_seconds_mean_per_pair": {key: value / prefix for key, value in timing_sums.items()},
        "status": "MEASURED_RECEIVER_INTERIOR_RUNG",
        "verdict_scope": (
            "this exact generic decoder formulation and seed instance; textured, learned, "
            "and higher-order contextual interior-fill families remain open"
        ),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _rung_rank(prefix: Mapping[str, Any]) -> tuple[int, int, int, float, int, int]:
    return (
        int(prefix["semantic_cells_to_rgb_exact_pair_count"]),
        int(prefix["all_declared_writes_survive_pair_count"]),
        int(prefix["surviving_write_count"]),
        -float(prefix["mean_d_seg_realized_argmax_vs_description"]),
        int(prefix["pose_within_declared_tube_pair_count"]),
        -int(prefix["additional_seed_bytes_total"]),
    )


def run_interior_rungs(
    *,
    seed_path: Path,
    gt_cache_path: Path,
    upstream: Path,
    output_root: Path,
    chunk_size: int,
    threads: int,
    stop_after_prefix: int,
) -> dict[str, Any]:
    """Run/resume R1-R4 through the real factor-2 and native scorer path."""

    if chunk_size < 1 or threads < 1 or stop_after_prefix not in PREFIXES:
        raise RealizationAuditError("invalid interior-rung chunk/thread/prefix setting")
    seed_bytes = seed_path.read_bytes()
    seed = parse_constraint_seed(seed_bytes)
    if serialize_constraint_seed(seed) != seed_bytes:
        raise RealizationAuditError("seed is not canonical on parse-back")
    seed_sha256 = hashlib.sha256(seed_bytes).hexdigest()
    cache = _load_real_cache(gt_cache_path)
    net, torch, scorer_custody = _load_distortion_net(upstream, threads)
    kernel = FullResizeKernel.build()
    implementation_path = REPO / "tools/measure_realization_g2_lattice.py"
    config = {
        "schema": INTERIOR_CONFIG_SCHEMA,
        "seed_sha256": seed_sha256,
        "gt_cache_sha256": GT_CACHE_SHA256,
        "scorer_custody": scorer_custody,
        "implementation_source": {str(implementation_path.relative_to(REPO)): _sha256(implementation_path)},
        "rung_ids": list(INTERIOR_RUNG_IDS),
        "r1_fixed_magnitude_palette": R1_FIXED_MAGNITUDE_PALETTE.tolist(),
        "r2_max_margin_palette": R2_MAX_MARGIN_PALETTE.tolist(),
        "r2_context_free_margins": list(R2_CONTEXT_FREE_MARGINS),
        "r2_geometry_probe": "frozen SegNet 6^3 constant-tile RGB cube; grid values 0,51,...,255",
        "r3_memory_prototypes": R3_MEMORY_PROTOTYPES.astype(int).tolist(),
        "r3_hopfield_beta": 8.0,
        "r4_base_rung": "R2_MAX_MARGIN",
        "frame0_policy": "same decoded pair cell field; seed lacks an intra-pair appearance carrier",
        "pair_count": PAIR_COUNT,
        "axis": AXIS,
    }
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    root = output_root / "interior_rungs"
    stage_dirs = {rung: root / rung / "stages" for rung in INTERIOR_RUNG_IDS}
    rows = {
        rung: _load_interior_stages(stage_dirs[rung], rung_id=rung, config_sha256=config_sha256)
        for rung in INTERIOR_RUNG_IDS
    }
    resumed = {rung: len(rung_rows) for rung, rung_rows in rows.items()}
    constraints_by_pair: dict[int, list[Mapping[str, Any]]] = {pair: [] for pair in range(PAIR_COUNT)}
    for constraint in seed["constraint_seeds"]:
        if constraint["frame_index"] == 1:
            constraints_by_pair[int(constraint["time"])].append(constraint)

    for chunk_begin in range(0, stop_after_prefix, chunk_size):
        chunk_end = min(stop_after_prefix, chunk_begin + chunk_size)
        for pair_index in range(chunk_begin, chunk_end):
            started = time.perf_counter()
            clock = time.perf_counter()
            source1 = np.asarray(cache["gt_f1"][pair_index], dtype=np.uint8).copy()
            target_cells = np.asarray(cache["lstars"][pair_index], dtype=np.uint8).copy()
            target_pose = np.asarray(cache["gt_poses"][pair_index], dtype=np.float64).copy()
            load_seconds = time.perf_counter() - clock
            clock = time.perf_counter()
            represented = _represented_cells(seed, pair_index)
            cell_decode_seconds = time.perf_counter() - clock
            base_planes = {rung: interior_rgb_plane(represented, rung) for rung in INTERIOR_RUNG_IDS[:3]}
            constraints = constraints_by_pair[pair_index]

            for rung_id in INTERIOR_RUNG_IDS:
                if pair_index in rows[rung_id]:
                    continue
                rung_started = time.perf_counter()
                exception_payload = b""
                exception_records: list[dict[str, Any]] = []
                plane0 = base_planes["R2_MAX_MARGIN" if rung_id == "R4_DYING_WRITE_EXCEPTIONS" else rung_id]
                plane1 = plane0
                clock = time.perf_counter()
                if rung_id == "R4_DYING_WRITE_EXCEPTIONS":
                    r2 = rows["R2_MAX_MARGIN"].get(pair_index)
                    if r2 is None:
                        raise RealizationAuditError("R4 requires the preserved R2 stage")
                    source_plane1 = _exact_source_target_plane(kernel.operator, source1)
                    exception_payload = encode_dying_write_exceptions(
                        constraints,
                        r2["declared_write_survival"],
                        source_plane1,
                    )
                    plane1, ordinals = apply_dying_write_exceptions(
                        plane0,
                        constraints,
                        exception_payload,
                    )
                    exception_records = [
                        {
                            "ordinal": ordinal,
                            "class_id": int(constraints[ordinal]["cell_id"]),
                            "stratum": str(constraints[ordinal]["stratum"]),
                        }
                        for ordinal in ordinals
                    ]
                plane_decode_seconds = time.perf_counter() - clock

                clock = time.perf_counter()
                realized0 = realize_projected_rgb_plane_camera_uint8(
                    plane0,
                    represented,
                    _decoder_plane_custody(
                        seed_sha256=seed_sha256,
                        rgb=plane0,
                        represented_cells=represented,
                        rung_id=rung_id,
                    ),
                    kernel=kernel,
                )
                realized1 = realize_projected_rgb_plane_camera_uint8(
                    plane1,
                    represented,
                    _decoder_plane_custody(
                        seed_sha256=seed_sha256,
                        rgb=plane1,
                        represented_cells=represented,
                        rung_id=rung_id,
                        additional_seed_bytes=len(exception_payload),
                    ),
                    kernel=kernel,
                )
                second0 = realize_projected_rgb_plane_camera_uint8(
                    plane0,
                    represented,
                    _decoder_plane_custody(
                        seed_sha256=seed_sha256,
                        rgb=plane0,
                        represented_cells=represented,
                        rung_id=rung_id,
                    ),
                    kernel=kernel,
                )
                second1 = realize_projected_rgb_plane_camera_uint8(
                    plane1,
                    represented,
                    _decoder_plane_custody(
                        seed_sha256=seed_sha256,
                        rgb=plane1,
                        represented_cells=represented,
                        rung_id=rung_id,
                        additional_seed_bytes=len(exception_payload),
                    ),
                    kernel=kernel,
                )
                double_equal = bool(
                    np.array_equal(realized0["frame"], second0["frame"])
                    and np.array_equal(realized1["frame"], second1["frame"])
                )
                lattice_seconds = time.perf_counter() - clock
                clock = time.perf_counter()
                hard, actual_argmax, writes = _hard_oracle_interior(
                    net,
                    torch,
                    realized0["frame"],
                    realized1["frame"],
                    target_cells,
                    represented,
                    target_pose,
                    constraints,
                )
                hard_seconds = time.perf_counter() - clock
                pair_exact = bool(
                    realized0["factor2_verification"]["certified_exact"]
                    and realized1["factor2_verification"]["certified_exact"]
                )
                if not pair_exact or not double_equal:
                    raise RealizationAuditError(f"pair {pair_index} rung {rung_id} lost exact/deterministic custody")
                stage = {
                    "schema": INTERIOR_STAGE_SCHEMA,
                    "config_sha256": config_sha256,
                    "rung_id": rung_id,
                    "pair_index": pair_index,
                    "projected_rgb_frame0_sha256": realized0["projected_rgb_sha256"],
                    "projected_rgb_frame1_sha256": realized1["projected_rgb_sha256"],
                    "projected_cells_sha256": realized1["projected_cells_sha256"],
                    "camera_frame0_sha256": realized0["camera_uint8_sha256"],
                    "camera_frame1_sha256": realized1["camera_uint8_sha256"],
                    "uint8_factor2_exact": pair_exact,
                    "double_decode_identical": double_equal,
                    "receiver_derived_rgb": True,
                    "additional_seed_bytes": len(exception_payload),
                    "hard_oracle": hard,
                    "declared_write_survival": writes,
                    "exception_stream": {
                        "schema": "realization_g2c_dying_write_exceptions.v1",
                        "payload_sha256": hashlib.sha256(exception_payload).hexdigest(),
                        "payload_bytes": len(exception_payload),
                        "record_count": len(exception_records),
                        "records": exception_records,
                        "parseback_exact": True,
                    },
                    "timings_seconds": {
                        "source_cache_load_shared": load_seconds,
                        "seed_cell_decode_shared": cell_decode_seconds,
                        "rgb_plane_decode": plane_decode_seconds,
                        "lattice_double_decode": lattice_seconds,
                        "native_cpu_torch_hard_oracle": hard_seconds,
                        "rung_total": time.perf_counter() - rung_started,
                        "pair_total_to_stage": time.perf_counter() - started,
                    },
                    "authority": f"MEASURED {AXIS}",
                    "score_claim": False,
                    "promotion_eligible": False,
                }
                path = stage_dirs[rung_id] / f"pair_{pair_index:04d}.json"
                _atomic_json(path, stage)
                rows[rung_id][pair_index] = stage
                del realized0, realized1, second0, second1, actual_argmax

        checkpoint = {
            "schema": "realization_g2c_interior_chunk_checkpoint.v1",
            "config_sha256": config_sha256,
            "completed_through_exclusive": chunk_end,
            "completed_pairs_by_rung": {rung: len(rung_rows) for rung, rung_rows in rows.items()},
            "all_pair_stages_preserved": True,
            "resumed_pairs_at_invocation_start": resumed,
        }
        _atomic_json(
            root / "checkpoints" / f"chunk_{chunk_begin:04d}_{chunk_end:04d}.json",
            checkpoint,
        )

    ladders: dict[str, list[dict[str, Any]]] = {}
    for rung_id in INTERIOR_RUNG_IDS:
        ordered = [rows[rung_id][index] for index in range(stop_after_prefix)]
        ladders[rung_id] = []
        for prefix in PREFIXES:
            if prefix > stop_after_prefix:
                continue
            summary = summarize_interior_prefix(
                prefix,
                ordered[:prefix],
                rung_id=rung_id,
            )
            path = root / rung_id / "checkpoints" / f"prefix_n{prefix}.json"
            _atomic_json(path, summary)
            summary["checkpoint_path"] = str(path)
            summary["checkpoint_sha256"] = _sha256(path)
            ladders[rung_id].append(summary)

    final_prefix = {rung: ladder[-1] for rung, ladder in ladders.items()}
    winner = max(INTERIOR_RUNG_IDS, key=lambda rung: _rung_rank(final_prefix[rung]))
    winning = final_prefix[winner]
    if stop_after_prefix == PAIR_COUNT:
        from tac.canonical_equations.predict_project_realization_admissibility_20260721 import (
            predict_project_realization_certificate,
        )

        admission = predict_project_realization_certificate(
            pair_count=PAIR_COUNT,
            uint8_factor2_exact_fraction=winning["uint8_factor2_exact_fraction"],
            double_decode_identical_pair_count=winning["double_decode_identical_pair_count"],
            semantic_cells_to_rgb_exact_pair_count=winning["semantic_cells_to_rgb_exact_pair_count"],
            pose_within_declared_tube_pair_count=winning["pose_within_declared_tube_pair_count"],
            additional_seed_bytes=winning["additional_seed_bytes_total"],
            receiver_derived_rgb=True,
        )
    else:
        admission = {
            "accepted": False,
            "status": "PARTIAL_PREFIX_NOT_N600",
            "failed_predicates": ("n600",),
            "score_claim": False,
            "promotion_eligible": False,
        }
    receipt = {
        "schema": INTERIOR_RECEIPT_SCHEMA,
        "lane_id": "lane_realization_g2c_interior_578_20260721",
        "task_id": "578",
        "config": config,
        "config_sha256": config_sha256,
        "completed_prefix": stop_after_prefix,
        "resumed_pairs_at_invocation_start": resumed,
        "D1_semantic_exact_ladders": ladders,
        "winning_rung": winner,
        "winning_prefix": winning,
        "D2_added_seed_bytes": {
            rung: {
                "additional_seed_bytes_total": final_prefix[rung]["additional_seed_bytes_total"],
                "zero_added_seed_byte_target_met": final_prefix[rung]["zero_added_seed_byte_target_met"],
                "R4_exception_pricing": final_prefix[rung]["R4_exception_pricing"],
            }
            for rung in INTERIOR_RUNG_IDS
        },
        "D3_winning_hard_oracle": {
            "rung_id": winner,
            "mean_d_seg_realized_vs_frozen_target": winning["mean_d_seg_realized_vs_frozen_target"],
            "mean_d_seg_description_vs_frozen_target": winning["mean_d_seg_description_vs_frozen_target"],
            "mean_d_pose_realized_vs_frozen_target": winning["mean_d_pose_realized_vs_frozen_target"],
            "mean_d_pose_realized_outside_declared_tube": winning["mean_d_pose_realized_outside_declared_tube"],
            "pose_within_declared_tube_pair_count": winning["pose_within_declared_tube_pair_count"],
            "status": "MEASURED_HARD_ORACLE",
        },
        "D4_admissibility": admission,
        "verdict": (
            "INTERIOR_RUNG_ADMISSIBLE" if admission["accepted"] else "MEASURED_INTERIOR_FORMULATIONS_NOT_ADMISSIBLE"
        ),
        "verdict_scope": (
            "R1 fixed-magnitude, R2 constant-tile max-margin, R3 frozen-prototype "
            "Hopfield prox, and R4 R2-residual dying-write RGB exceptions on seed_compose_b2; "
            "textured, learned, and exact spatial SegNet-cell optimizers remain untested"
        ),
        "storage": {
            "stage_root": str(root),
            "automatic_disk_hygiene": (
                "ZIP_STORED mmap plus one-pair/rung tensors; atomic JSON temp files removed; "
                "no RGB, camera, logits, or exception payload bulk persisted"
            ),
        },
        "authority": {
            "axis": AXIS,
            "pointer": POINTER,
            "pointer_moved": False,
            "score_claim": False,
            "promotion_eligible": False,
            "main_landing_review_required": True,
        },
    }
    _atomic_json(root / "receipt.json", receipt)
    return receipt


def _load_contextual_stages(
    stage_dir: Path,
    sidecar_dir: Path,
    config_sha256: str,
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not stage_dir.exists():
        return rows
    for path in sorted(stage_dir.glob("pair_*.json")):
        row = _load_json(path)
        pair_index = row.get("pair_index")
        if (
            row.get("schema") != CONTEXTUAL_STAGE_SCHEMA
            or row.get("config_sha256") != config_sha256
            or isinstance(pair_index, bool)
            or not isinstance(pair_index, int)
            or pair_index in rows
        ):
            raise RealizationAuditError(f"contextual resume custody mismatch: {path}")
        sidecar = sidecar_dir / f"pair_{pair_index:04d}.g2dx"
        if not sidecar.is_file() or _sha256(sidecar) != row["exception_stream"]["sha256"]:
            raise RealizationAuditError(f"contextual exception sidecar custody mismatch: {sidecar}")
        if sidecar.stat().st_size != row["exception_stream"]["bytes"]:
            raise RealizationAuditError(f"contextual exception sidecar size mismatch: {sidecar}")
        rows[pair_index] = row
    return rows


def summarize_contextual_prefix(
    prefix: int,
    stage_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_bytes: int,
) -> dict[str, Any]:
    """Aggregate one contiguous contextual realization prefix."""

    if prefix not in PREFIXES or len(stage_rows) != prefix:
        raise RealizationAuditError("contextual prefix is not n16/n64/n600")
    if [row.get("pair_index") for row in stage_rows] != list(range(prefix)):
        raise RealizationAuditError("contextual prefix is not contiguous from zero")
    if any(row.get("schema") != CONTEXTUAL_STAGE_SCHEMA for row in stage_rows):
        raise RealizationAuditError("contextual stage schema mismatch")
    writes = [write for row in stage_rows for write in row["declared_write_survival"]]
    hard = [row["hard_oracle"] for row in stage_rows]
    exception_bytes = int(sum(row["exception_stream"]["bytes"] for row in stage_rows))
    changed = int(sum(row["exception_stream"]["record_count"] for row in stage_rows))
    total_bytes = CONTEXTUAL_SEED_BASELINE_BYTES + bootstrap_bytes + exception_bytes
    positive_survives = sum(row["survives"] is True and float(row["target_logit_margin"]) > 0.0 for row in writes)
    positive_total = sum(float(row["target_logit_margin"]) > 0.0 for row in writes)
    nonpositive_survives = sum(row["survives"] is True and float(row["target_logit_margin"]) <= 0.0 for row in writes)
    nonpositive_total = len(writes) - positive_total
    return {
        "schema": "realization_g2d_predict_base_prefix.v1",
        "n": prefix,
        "pair_count": prefix,
        "frame_count": 2 * prefix,
        "uint8_factor2_exact_pair_count": sum(row["uint8_factor2_exact"] is True for row in stage_rows),
        "uint8_factor2_exact_fraction": float(np.mean([row["uint8_factor2_exact"] for row in stage_rows])),
        "double_decode_identical_pair_count": sum(row["double_decode_identical"] is True for row in stage_rows),
        "semantic_cells_to_rgb_exact_pair_count": sum(
            row["hard_oracle"]["realized_argmax_equals_description"] is True for row in stage_rows
        ),
        "semantic_cells_to_rgb_exact_fraction": float(
            np.mean([row["hard_oracle"]["realized_argmax_equals_description"] for row in stage_rows])
        ),
        "all_declared_writes_survive_pair_count": sum(
            row["hard_oracle"]["all_declared_writes_survive"] is True for row in stage_rows
        ),
        "declared_write_count": len(writes),
        "surviving_write_count": sum(row["survives"] is True for row in writes),
        "declared_write_survival_fraction": (float(np.mean([row["survives"] for row in writes])) if writes else 1.0),
        "by_class": _aggregate_survival(writes, "class_id"),
        "by_stratum": _aggregate_survival(writes, "stratum"),
        "by_margin_bucket": _aggregate_survival(writes, "margin_bucket"),
        "margin_survival_contingency": {
            "positive_survives": positive_survives,
            "positive_total": positive_total,
            "nonpositive_survives": nonpositive_survives,
            "nonpositive_total": nonpositive_total,
        },
        "mean_d_seg_realized_vs_frozen_target": float(
            np.mean([row["d_seg_realized_vs_frozen_target"] for row in hard])
        ),
        "mean_d_seg_description_vs_frozen_target": float(
            np.mean([row["d_seg_description_vs_frozen_target"] for row in hard])
        ),
        "mean_d_seg_realized_argmax_vs_description": float(
            np.mean([row["d_seg_realized_argmax_vs_description"] for row in hard])
        ),
        "mean_d_pose_realized_vs_frozen_target": float(
            np.mean([row["d_pose_realized_vs_frozen_target"] for row in hard])
        ),
        "mean_d_pose_realized_outside_declared_tube": float(
            np.mean([row["d_pose_realized_outside_declared_tube"] for row in hard])
        ),
        "pose_within_declared_tube_pair_count": sum(row["pose_within_declared_tube"] is True for row in hard),
        "byte_accounting": {
            "seed_baseline_bytes": CONTEXTUAL_SEED_BASELINE_BYTES,
            "frame0_bootstrap_brotli11_bytes": bootstrap_bytes,
            "per_frame_exception_container_bytes": exception_bytes,
            "exception_record_count": changed,
            "contextual_total_bytes": total_bytes,
            "target_box_bytes": CONTEXTUAL_TARGET_BOX_BYTES,
            "headroom_vs_target_box_bytes": CONTEXTUAL_TARGET_BOX_BYTES - total_bytes,
            "fits_target_box": total_bytes <= CONTEXTUAL_TARGET_BOX_BYTES,
        },
        "projection_band_histogram": dict(
            sorted(Counter(str(row["projection"]["selected_band"]) for row in stage_rows).items())
        ),
        "status": "MEASURED_CONTEXTUAL_PREDICT_BASE_PREFIX",
        "verdict_scope": (
            "one exact Brotli bootstrap, canonical G1 ground-homography RGB prediction, "
            "one-round class-prototype L-infinity band projection, and #557 ordinal/RGB exceptions"
        ),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _contextual_plane_custody(
    *,
    seed_sha256: str,
    rgb: np.ndarray,
    represented_cells: np.ndarray,
    additional_seed_bytes: int,
    generator_id: str,
) -> dict[str, Any]:
    return {
        "schema": PROJECTED_RGB_PLANE_CUSTODY_SCHEMA,
        "source_kind": "encoder_supplied_counted" if additional_seed_bytes else "decoder_derived_from_seed",
        "generator_id": generator_id,
        "seed_sha256": seed_sha256,
        "projected_rgb_sha256": projected_plane_array_sha256(rgb),
        "projected_cells_sha256": projected_plane_array_sha256(represented_cells),
        "additional_seed_bytes": additional_seed_bytes,
        "decoder_scorer_invocations": 0,
    }


def _contextual_realize(
    plane: np.ndarray,
    represented: np.ndarray,
    *,
    seed_sha256: str,
    additional_seed_bytes: int,
    generator_id: str,
    kernel: FullResizeKernel,
) -> dict[str, Any]:
    return realize_projected_rgb_plane_camera_uint8(
        plane,
        represented,
        _contextual_plane_custody(
            seed_sha256=seed_sha256,
            rgb=plane,
            represented_cells=represented,
            additional_seed_bytes=additional_seed_bytes,
            generator_id=generator_id,
        ),
        kernel=kernel,
    )


def _replay_contextual_sequence(
    *,
    bootstrap_path: Path,
    prefix: int,
    rows: Mapping[int, Mapping[str, Any]],
    sidecar_dir: Path,
    constraints_by_pair: Mapping[int, Sequence[Mapping[str, Any]]],
    represented_by_pair: Mapping[int, np.ndarray],
    cross_xi: np.ndarray,
    within_xi: np.ndarray,
    geom: g1_warp.GroundHomographyGeom,
    seed_sha256: str,
    kernel: FullResizeKernel,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw = brotli.decompress(bootstrap_path.read_bytes())
    if len(raw) != int(np.prod((*SCORER_HW, 3))):
        raise RealizationAuditError("contextual bootstrap decoded byte count mismatch")
    bootstrap = np.frombuffer(raw, dtype=np.uint8).reshape(*SCORER_HW, 3).copy()
    frame_hashes: list[str] = []
    previous_plane: np.ndarray | None = None
    for pair_index in range(prefix):
        represented = represented_by_pair[pair_index]
        plane0 = (
            bootstrap.copy()
            if pair_index == 0
            else contextual_advected_rgb_plane(previous_plane, cross_xi[pair_index], geom)
        )
        frame0 = _contextual_realize(
            plane0,
            represented,
            seed_sha256=seed_sha256,
            additional_seed_bytes=bootstrap_path.stat().st_size if pair_index == 0 else 0,
            generator_id="g2d_bootstrap_brotli11" if pair_index == 0 else "g2d_cross_pair_g1_advected_rgb",
            kernel=kernel,
        )
        base1 = contextual_advected_rgb_plane(plane0, within_xi[pair_index], geom)
        payload = (sidecar_dir / f"pair_{pair_index:04d}.g2dx").read_bytes()
        plane1, ordinals = apply_contextual_rgb_exceptions(base1, constraints_by_pair[pair_index], payload)
        if ordinals != rows[pair_index]["exception_stream"]["ordinals"]:
            raise RealizationAuditError("contextual replay ordinal custody mismatch")
        frame1 = _contextual_realize(
            plane1,
            represented,
            seed_sha256=seed_sha256,
            additional_seed_bytes=len(payload),
            generator_id="g2d_within_pair_g1_advected_rgb_plus_margin_projection",
            kernel=kernel,
        )
        if (
            frame0["camera_uint8_sha256"] != rows[pair_index]["camera_frame0_sha256"]
            or frame1["camera_uint8_sha256"] != rows[pair_index]["camera_frame1_sha256"]
        ):
            raise RealizationAuditError("contextual sequential replay camera hash mismatch")
        frame_hashes.extend((frame0["camera_uint8_sha256"], frame1["camera_uint8_sha256"]))
        previous_plane = plane1
    return {
        "schema": "realization_g2d_sequential_decode_wallclock.v1",
        "pair_count": prefix,
        "frame_count": 2 * prefix,
        "wall_seconds": time.perf_counter() - started,
        "frame_hash_tree_sha256": hashlib.sha256(json.dumps(frame_hashes, separators=(",", ":")).encode()).hexdigest(),
        "decoder_scorer_invocations": 0,
        "wallclock_is_gate": False,
        "measurement_label": f"MEASURED {AXIS}",
    }


def run_contextual_predict_base(
    *,
    seed_path: Path,
    gt_cache_path: Path,
    upstream: Path,
    output_root: Path,
    chunk_size: int,
    threads: int,
    stop_after_prefix: int,
) -> dict[str, Any]:
    """Run/resume the G2d sequential contextual RGB realization ladder."""

    if chunk_size < 1 or threads < 1 or stop_after_prefix not in PREFIXES:
        raise RealizationAuditError("invalid contextual chunk/thread/prefix setting")
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    if usage.free < 1 << 30:
        raise RealizationAuditError("contextual storage preflight requires at least 1 GiB free")
    seed_bytes = seed_path.read_bytes()
    seed = parse_constraint_seed(seed_bytes)
    if serialize_constraint_seed(seed) != seed_bytes:
        raise RealizationAuditError("seed is not canonical on parse-back")
    seed_sha256 = hashlib.sha256(seed_bytes).hexdigest()
    cache = _load_real_cache(gt_cache_path)
    net, torch, scorer_custody = _load_distortion_net(upstream, threads)
    kernel = FullResizeKernel.build()
    s_t, s_r, pitch_rad, motion_custody = load_g1_worldsheet_motion(REPO)
    poses = np.asarray(cache["gt_poses"], dtype=np.float64)
    cross_xi, cross_custody = relative_adjacent_xi(poses, s_t=s_t, s_r=s_r, pitch_rad=pitch_rad)
    within_xi = np.stack(
        [g1_warp.xi_from_pose_calibration(pose, s_t=s_t, s_r=s_r, pitch=pitch_rad) for pose in poses],
        axis=0,
    )
    geom = g1_warp.GroundHomographyGeom.eon(native_hw=SCORER_HW, pitch=pitch_rad)
    source0 = np.asarray(cache["gt_f0"][0], dtype=np.uint8).copy()
    bootstrap = _exact_source_target_plane(kernel.operator, source0)
    bootstrap_payload = brotli.compress(bootstrap.tobytes(order="C"), quality=11)
    root = output_root / "contextual_predict_base"
    bootstrap_path = root / "bootstrap_frame0.brotli"
    if bootstrap_path.exists() and bootstrap_path.read_bytes() != bootstrap_payload:
        raise RealizationAuditError("contextual bootstrap resume custody mismatch")
    if not bootstrap_path.exists():
        bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".bootstrap_frame0.", suffix=".tmp", dir=bootstrap_path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(bootstrap_payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, bootstrap_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    implementation_paths = (
        REPO / "tools/measure_realization_g2_lattice.py",
        REPO / "src/tac/optimization/predict_project_receiver.py",
        REPO / "src/tac/optimization/uint8_lattice_feasibility.py",
        REPO / "src/tac/boundary_math/warp_real_luma_frame0.py",
        REPO / "src/tac/optimization/predictor_upgrade_xi_chart.py",
        REPO / "src/tac/optimization/predictor_r2_missdelta.py",
        REPO / "src/tac/canonical_equations/predict_project_realization_admissibility_20260721.py",
    )
    config = {
        "schema": CONTEXTUAL_CONFIG_SCHEMA,
        "seed_sha256": seed_sha256,
        "gt_cache_sha256": GT_CACHE_SHA256,
        "scorer_custody": scorer_custody,
        "motion_custody": {**motion_custody, **cross_custody},
        "implementation_sources": {str(path.relative_to(REPO)): _sha256(path) for path in implementation_paths},
        "decode_order": "frame0_0_bootstrap_then_frame1_0_then_frame0_1...frame1_599",
        "cross_pair_motion": "gt_poses[t] direct G1 nearest-target-pair proxy",
        "within_pair_motion": "gt_poses[t] exact banked target for pair",
        "projection_bands_linf_u8": list(CONTEXTUAL_PROJECTION_BANDS),
        "projection_target": "R2 class max-margin prototype from contextual base",
        "projection_scope": "initially violated declared constraint sites only",
        "projection_rounds": 1,
        "bootstrap": {
            "raw_bytes": bootstrap.nbytes,
            "raw_sha256": hashlib.sha256(bootstrap.tobytes()).hexdigest(),
            "brotli11_bytes": len(bootstrap_payload),
            "brotli11_sha256": hashlib.sha256(bootstrap_payload).hexdigest(),
            "coder_custody": "#557 settled complete classical coder choice: Brotli-11",
            "video_derived_counted": True,
        },
        "exception_coder": "#557 adaptive range coder with ordinal-gap and class-channel RGB contexts",
        "pair_count": PAIR_COUNT,
        "axis": AXIS,
    }
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    stage_dir = root / "stages"
    sidecar_dir = root / "exceptions"
    rows = _load_contextual_stages(stage_dir, sidecar_dir, config_sha256)
    resumed_pairs = len(rows)
    if rows and sorted(rows) != list(range(max(rows) + 1)):
        raise RealizationAuditError("contextual resume stages are not a contiguous prefix")
    constraints_by_pair: dict[int, list[Mapping[str, Any]]] = {pair: [] for pair in range(PAIR_COUNT)}
    for constraint in seed["constraint_seeds"]:
        if constraint["frame_index"] == 1:
            constraints_by_pair[int(constraint["time"])].append(constraint)
    represented_by_pair = {pair: _represented_cells(seed, pair) for pair in range(stop_after_prefix)}
    previous_plane: np.ndarray | None = None
    for chunk_begin in range(0, stop_after_prefix, chunk_size):
        chunk_end = min(stop_after_prefix, chunk_begin + chunk_size)
        for pair_index in range(chunk_begin, chunk_end):
            represented = represented_by_pair[pair_index]
            constraints = constraints_by_pair[pair_index]
            predict0_clock = time.perf_counter()
            plane0 = (
                bootstrap.copy()
                if pair_index == 0
                else contextual_advected_rgb_plane(previous_plane, cross_xi[pair_index], geom)
            )
            predict0_seconds = time.perf_counter() - predict0_clock
            predict1_clock = time.perf_counter()
            base1 = contextual_advected_rgb_plane(plane0, within_xi[pair_index], geom)
            predict1_seconds = time.perf_counter() - predict1_clock
            sidecar_path = sidecar_dir / f"pair_{pair_index:04d}.g2dx"
            if pair_index in rows:
                previous_plane, _ = apply_contextual_rgb_exceptions(base1, constraints, sidecar_path.read_bytes())
                continue
            started = time.perf_counter()
            target_cells = np.asarray(cache["lstars"][pair_index], dtype=np.uint8).copy()
            target_pose = np.asarray(cache["gt_poses"][pair_index], dtype=np.float64).copy()
            lattice0_clock = time.perf_counter()
            realized0 = _contextual_realize(
                plane0,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=len(bootstrap_payload) if pair_index == 0 else 0,
                generator_id="g2d_bootstrap_brotli11" if pair_index == 0 else "g2d_cross_pair_g1_advected_rgb",
                kernel=kernel,
            )
            realized_base1 = _contextual_realize(
                base1,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=0,
                generator_id="g2d_within_pair_g1_advected_rgb_base",
                kernel=kernel,
            )
            base_lattice_seconds = time.perf_counter() - lattice0_clock
            encoder_clock = time.perf_counter()
            _, initial_writes = _seg_write_oracle(net, torch, realized0["frame"], realized_base1["frame"], constraints)
            violated = [row["ordinal"] for row in initial_writes if row["survives"] is False]
            selected_plane = base1
            selected_writes = initial_writes
            selected_band = 0
            selected_rank = _contextual_candidate_rank(initial_writes, changed_sites=0, band=0)
            candidate_rows = []
            for band in CONTEXTUAL_PROJECTION_BANDS:
                candidate_plane = contextual_banded_projection(base1, constraints, violated, band)
                candidate_realized = _contextual_realize(
                    candidate_plane,
                    represented,
                    seed_sha256=seed_sha256,
                    additional_seed_bytes=0,
                    generator_id=f"g2d_margin_projection_band_{band}",
                    kernel=kernel,
                )
                _, candidate_writes = _seg_write_oracle(
                    net, torch, realized0["frame"], candidate_realized["frame"], constraints
                )
                rank = _contextual_candidate_rank(candidate_writes, changed_sites=len(violated), band=band)
                candidate_rows.append(
                    {
                        "band": band,
                        "surviving_writes": sum(row["survives"] is True for row in candidate_writes),
                        "positive_margin_writes": sum(
                            float(row["target_logit_margin"]) > 0.0 for row in candidate_writes
                        ),
                        "minimum_margin": min(
                            (float(row["target_logit_margin"]) for row in candidate_writes),
                            default=None,
                        ),
                    }
                )
                if rank > selected_rank:
                    selected_rank = rank
                    selected_plane = candidate_plane
                    selected_writes = candidate_writes
                    selected_band = band
            encoder_projection_seconds = time.perf_counter() - encoder_clock
            exception_payload, changed_ordinals = encode_contextual_rgb_exceptions(base1, selected_plane, constraints)
            parsed_plane, parsed_ordinals = apply_contextual_rgb_exceptions(base1, constraints, exception_payload)
            if (
                parsed_ordinals != changed_ordinals
                or not np.array_equal(parsed_plane, selected_plane)
                or encode_contextual_rgb_exceptions(base1, parsed_plane, constraints)[0] != exception_payload
            ):
                raise RealizationAuditError("contextual exception parse-back/re-encode failed")
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=f".pair_{pair_index:04d}.", suffix=".tmp", dir=sidecar_dir)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(exception_payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, sidecar_path)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            final_lattice_clock = time.perf_counter()
            realized1 = _contextual_realize(
                parsed_plane,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=len(exception_payload),
                generator_id="g2d_within_pair_g1_advected_rgb_plus_margin_projection",
                kernel=kernel,
            )
            second0 = _contextual_realize(
                plane0,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=len(bootstrap_payload) if pair_index == 0 else 0,
                generator_id="g2d_double_decode_frame0",
                kernel=kernel,
            )
            second1 = _contextual_realize(
                parsed_plane,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=len(exception_payload),
                generator_id="g2d_double_decode_frame1",
                kernel=kernel,
            )
            final_lattice_seconds = time.perf_counter() - final_lattice_clock
            hard_clock = time.perf_counter()
            hard, actual_argmax, writes = _hard_oracle_interior(
                net,
                torch,
                realized0["frame"],
                realized1["frame"],
                target_cells,
                represented,
                target_pose,
                constraints,
            )
            hard_seconds = time.perf_counter() - hard_clock
            if writes != selected_writes:
                raise RealizationAuditError("contextual final scorer row drifted from selected band")
            double_equal = bool(
                np.array_equal(realized0["frame"], second0["frame"])
                and np.array_equal(realized1["frame"], second1["frame"])
            )
            pair_exact = bool(
                realized0["factor2_verification"]["certified_exact"]
                and realized1["factor2_verification"]["certified_exact"]
            )
            if not pair_exact or not double_equal:
                raise RealizationAuditError(f"contextual pair {pair_index} lost exact/deterministic lattice custody")
            previous_roundtrip = _exact_source_target_plane(kernel.operator, realized1["frame"])
            if not np.array_equal(previous_roundtrip, parsed_plane):
                raise RealizationAuditError("contextual previous-decoded RGB plane round-trip drift")
            stage = {
                "schema": CONTEXTUAL_STAGE_SCHEMA,
                "config_sha256": config_sha256,
                "pair_index": pair_index,
                "projected_rgb_frame0_sha256": realized0["projected_rgb_sha256"],
                "projected_rgb_frame1_sha256": realized1["projected_rgb_sha256"],
                "projected_cells_sha256": realized1["projected_cells_sha256"],
                "camera_frame0_sha256": realized0["camera_uint8_sha256"],
                "camera_frame1_sha256": realized1["camera_uint8_sha256"],
                "uint8_factor2_exact": pair_exact,
                "double_decode_identical": double_equal,
                "previous_decoded_plane_roundtrip_exact": True,
                "receiver_derived_rgb": True,
                "hard_oracle": hard,
                "declared_write_survival": writes,
                "projection": {
                    "initial_violated_ordinals": violated,
                    "initial_surviving_writes": sum(row["survives"] is True for row in initial_writes),
                    "selected_band": selected_band,
                    "candidate_rows": candidate_rows,
                    "changed_site_count": len(changed_ordinals),
                    "scope": "initially violated declared constraint sites only",
                },
                "exception_stream": {
                    "schema": "realization_g2d_contextual_rgb_exceptions.v1",
                    "path": str(sidecar_path),
                    "bytes": len(exception_payload),
                    "sha256": hashlib.sha256(exception_payload).hexdigest(),
                    "record_count": len(changed_ordinals),
                    "ordinals": changed_ordinals,
                    "parseback_exact": True,
                    "reencode_byte_identical": True,
                    "coder": "#557 adaptive range coder",
                },
                "timings_seconds": {
                    "decoder_cross_pair_rgb_prediction": predict0_seconds,
                    "decoder_within_pair_rgb_prediction": predict1_seconds,
                    "decoder_base_lattice": base_lattice_seconds,
                    "encoder_margin_band_selection": encoder_projection_seconds,
                    "decoder_final_lattice_and_double_decode": final_lattice_seconds,
                    "native_cpu_torch_final_hard_oracle": hard_seconds,
                    "pair_total": time.perf_counter() - started,
                },
                "authority": f"MEASURED {AXIS}",
                "score_claim": False,
                "promotion_eligible": False,
            }
            _atomic_json(stage_dir / f"pair_{pair_index:04d}.json", stage)
            rows[pair_index] = stage
            previous_plane = parsed_plane
            del target_cells, target_pose, actual_argmax
        _atomic_json(
            root / "checkpoints" / f"chunk_{chunk_begin:04d}_{chunk_end:04d}.json",
            {
                "schema": "realization_g2d_predict_base_chunk_checkpoint.v1",
                "config_sha256": config_sha256,
                "completed_through_exclusive": chunk_end,
                "completed_pairs": len(rows),
                "all_pair_stages_preserved": True,
                "resumed_pairs_at_invocation_start": resumed_pairs,
            },
        )
    ordered = [rows[index] for index in range(stop_after_prefix)]
    prefixes = []
    for prefix in PREFIXES:
        if prefix > stop_after_prefix:
            continue
        summary = summarize_contextual_prefix(prefix, ordered[:prefix], bootstrap_bytes=len(bootstrap_payload))
        path = root / "checkpoints" / f"prefix_n{prefix}.json"
        _atomic_json(path, summary)
        summary["checkpoint_path"] = str(path)
        summary["checkpoint_sha256"] = _sha256(path)
        prefixes.append(summary)
    final = prefixes[-1]
    replay = _replay_contextual_sequence(
        bootstrap_path=bootstrap_path,
        prefix=stop_after_prefix,
        rows=rows,
        sidecar_dir=sidecar_dir,
        constraints_by_pair=constraints_by_pair,
        represented_by_pair=represented_by_pair,
        cross_xi=cross_xi,
        within_xi=within_xi,
        geom=geom,
        seed_sha256=seed_sha256,
        kernel=kernel,
    )
    if stop_after_prefix == PAIR_COUNT:
        from tac.canonical_equations.predict_project_realization_admissibility_20260721 import (
            predict_project_realization_certificate,
        )

        admission = predict_project_realization_certificate(
            pair_count=PAIR_COUNT,
            uint8_factor2_exact_fraction=final["uint8_factor2_exact_fraction"],
            double_decode_identical_pair_count=final["double_decode_identical_pair_count"],
            semantic_cells_to_rgb_exact_pair_count=final["semantic_cells_to_rgb_exact_pair_count"],
            pose_within_declared_tube_pair_count=final["pose_within_declared_tube_pair_count"],
            additional_seed_bytes=(
                final["byte_accounting"]["frame0_bootstrap_brotli11_bytes"]
                + final["byte_accounting"]["per_frame_exception_container_bytes"]
            ),
            receiver_derived_rgb=True,
        )
    else:
        admission = {
            "accepted": False,
            "status": "PARTIAL_PREFIX_NOT_N600",
            "failed_predicates": ("n600",),
            "score_claim": False,
            "promotion_eligible": False,
        }
    receipt = {
        "schema": CONTEXTUAL_RECEIPT_SCHEMA,
        "lane_id": CONTEXTUAL_LANE_ID,
        "task_id": "578",
        "config": config,
        "config_sha256": config_sha256,
        "completed_prefix": stop_after_prefix,
        "resumed_pairs_at_invocation_start": resumed_pairs,
        "D1_semantic_exact_ladder": prefixes,
        "D2_pose_real_motion": {
            "mean_d_pose_realized_vs_frozen_target": final["mean_d_pose_realized_vs_frozen_target"],
            "mean_d_pose_realized_outside_declared_tube": final["mean_d_pose_realized_outside_declared_tube"],
            "pose_within_declared_tube_pair_count": final["pose_within_declared_tube_pair_count"],
            "intra_pair_frames_identical_by_design": False,
            "status": f"MEASURED {AXIS}",
        },
        "D3_byte_accounting": final["byte_accounting"],
        "D4_sequential_decode_wallclock": replay,
        "D5_admissibility": admission,
        "verdict": (
            "CONTEXTUAL_PREDICT_BASE_ADMISSIBLE"
            if admission["accepted"]
            else "MEASURED_CONTEXTUAL_PREDICT_BASE_NOT_ADMISSIBLE"
        ),
        "verdict_scope": (
            "one exact Brotli-11 frame0 scorer-RGB bootstrap, canonical G1 ground-plane "
            "homography for within/cross transitions, one-round R2-prototype L-infinity "
            "projection at initially violated declared sites, and #557 ordinal/RGB exceptions; "
            "not a negative on learned spatial contextual generators, secant/QP projection, "
            "curvelet/shearlet action families, or exact cross-pair motion"
        ),
        "storage": {
            "root": str(root),
            "free_bytes_at_preflight": usage.free,
            "automatic_disk_hygiene": (
                "ZIP_STORED cache memmaps, one pair of RGB/camera tensors at a time, atomic JSON/bin "
                "replace, and no persisted camera/logit tensors"
            ),
        },
        "authority": {
            "axis": AXIS,
            "pointer": POINTER,
            "pointer_moved": False,
            "score_claim": False,
            "promotion_eligible": False,
            "main_landing_review_required": True,
        },
    }
    _atomic_json(root / "receipt.json", receipt)
    _atomic_json(output_root / "receipt.json", receipt)
    return receipt


def audit_prefix(
    *,
    prefix: int,
    stage_rows: Sequence[Mapping[str, Any]],
    constraints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify one real prefix without inventing an RGB projection."""

    if prefix not in PREFIXES or len(stage_rows) != prefix:
        raise RealizationAuditError("prefix rows do not match the canonical n16/n64/n600 ladder")
    if [row.get("pair_index") for row in stage_rows] != list(range(prefix)):
        raise RealizationAuditError("pair stages are not contiguous from zero")
    label_only_pairs = 0
    for index, row in enumerate(stage_rows):
        if row.get("schema") != PAIR_STAGE_SCHEMA:
            raise RealizationAuditError(f"pair {index} stage schema mismatch")
        hard = row.get("hard_oracle")
        if not isinstance(hard, Mapping) or hard.get("schema") != HARD_ORACLE_SCHEMA:
            raise RealizationAuditError(f"pair {index} lacks the real seed-compose hard row")
        if hard.get("cell_exact") is not True or hard.get("pose_within_tube") is not True:
            raise RealizationAuditError(f"pair {index} lost the settled plane-level cell/tube invariant")
        if hard.get("uint8_factor2_exact") is not False:
            raise RealizationAuditError(f"pair {index} no longer matches the seed-compose G2 blocker")
        if RGB_REALIZATION_FIELDS.intersection(hard) or RGB_REALIZATION_FIELDS.intersection(row):
            raise RealizationAuditError(f"pair {index} unexpectedly carries unaudited RGB realization fields")
        label_only_pairs += 1

    selected = []
    for row in constraints:
        time_value = _exact_nonnegative_int(row.get("time"), "constraint time")
        if time_value < prefix:
            selected.append(row)
    by_class = Counter(int(row["cell_id"]) for row in selected)
    by_stratum = Counter(str(row["stratum"]) for row in selected)

    def blocked_rows(counter: Counter[Any], name: str) -> list[dict[str, Any]]:
        return [
            {
                name: key,
                "declared_writes": count,
                "surviving_writes": 0,
                "dying_writes": 0,
                "not_attempted_missing_rgb_projection": count,
                "exact_fraction": None,
            }
            for key, count in sorted(counter.items(), key=lambda item: str(item[0]))
        ]

    hard_rows = [row["hard_oracle"] for row in stage_rows]
    return {
        "schema": "realization_g2_prefix_audit.v1",
        "n": prefix,
        "pair_count": prefix,
        "label_only_pair_count": label_only_pairs,
        "rgb_projection_pair_count": 0,
        "lattice_attempted_pair_count": 0,
        "uint8_factor2_exact_pair_count": 0,
        "uint8_factor2_exact_fraction": None,
        "declared_constraint_count": len(selected),
        "by_class": blocked_rows(by_class, "class_id"),
        "by_stratum": blocked_rows(by_stratum, "stratum"),
        "plane_level_cache_replay": {
            "cell_exact_pairs": sum(row["cell_exact"] is True for row in hard_rows),
            "pose_within_tube_pairs": sum(row["pose_within_tube"] is True for row in hard_rows),
            "mean_d_seg_description": sum(float(row["d_seg"]) for row in hard_rows) / prefix,
            "mean_d_pose_tube_debt": sum(float(row["d_pose"]) for row in hard_rows) / prefix,
            "authority": "MEASURED_EXISTING_LABEL_AND_TUBE_CACHE_REPLAY_NOT_REALIZED_RGB",
        },
        "status": "BLOCKED_INPUT_DOMAIN_LABEL_FIELD_IS_NOT_RGB_PLANE",
        "verdict_scope": "seed_compose_b2 projected class IDs -> composed RGB lattice handoff",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }


def run_audit(
    *,
    seed_path: Path,
    stage_root: Path,
    m2_receipt_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    seed_bytes = seed_path.read_bytes()
    seed = parse_constraint_seed(seed_bytes)
    constraints = seed["constraint_seeds"]
    stage_paths = sorted(stage_root.glob("pair_*.json"))
    if len(stage_paths) != 600:
        raise RealizationAuditError(f"expected 600 preserved pair stages, found {len(stage_paths)}")
    stage_rows = [_load_json(path) for path in stage_paths]
    m2 = _load_json(m2_receipt_path)
    if m2.get("schema") != "m2_live_target_selection_receipt.v1":
        raise RealizationAuditError("M2 existence comparator schema mismatch")

    prefix_rows = []
    checkpoints = output_root / "checkpoints"
    for prefix in PREFIXES:
        row = audit_prefix(prefix=prefix, stage_rows=stage_rows[:prefix], constraints=constraints)
        checkpoint_path = checkpoints / f"prefix_n{prefix}.json"
        _atomic_json(checkpoint_path, row)
        row["checkpoint_path"] = str(checkpoint_path)
        row["checkpoint_sha256"] = _sha256(checkpoint_path)
        prefix_rows.append(row)

    implementation_paths = (
        REPO / "src/tac/optimization/predict_project_receiver.py",
        REPO / "tools/measure_realization_g2_lattice.py",
        REPO / "src/tac/tests/test_predict_project_receiver.py",
    )
    implementation = {str(path.relative_to(REPO)): _sha256(path) for path in implementation_paths}
    receipt = {
        "schema": SCHEMA,
        "lane_id": "lane_realization_g2_lattice_578_20260721",
        "task_id": "578",
        "verdict": "COMPOSED_RGB_LATTICE_BUILT_SEED_TO_RGB_PROJECTION_PREMISE_FALSIFIED",
        "verdict_scope": (
            "the current seed_compose_b2 class-ID projection cannot enter the RGB factor-2 lattice; "
            "this is a formulation handoff gap, not a realization-family negative"
        ),
        "D1_prefix_ladder": prefix_rows,
        "D1_implementation": {
            "status": "BUILT_STRICT_RGB_INPUT_CONTRACT",
            "callable": "tac.optimization.predict_project_receiver.realize_projected_rgb_plane_camera_uint8",
            "structural_fixture_factor2_exact": True,
            "real_seed_rgb_input_status": "ABSENT",
            "real_n600_uint8_factor2_exact": None,
            "reason": "the real seed and preserved stages contain 2D uint8 class IDs, no HxWx3 uint8 projected RGB plane",
        },
        "D2_cost": {
            "added_decode_seconds_per_pair": None,
            "additional_seed_bytes": None,
            "zero_byte_target_met": False,
            "status": "UNMEASURED_NO_COMPOSED_RGB_FRAMES",
            "M2_existence_comparator": {
                "receipt_path": str(m2_receipt_path),
                "receipt_sha256": _sha256(m2_receipt_path),
                "archive_bytes": m2["candidate"]["archive_bytes"],
                "d_seg": m2["candidate"]["d_seg"],
                "d_pose": m2["candidate"]["d_pose"],
                "interpretation": "exact realization exists when source RGB target values are counted; this does not supply the missing zero-byte seed-to-RGB map",
            },
        },
        "D3_pose": {
            "realized_frame_d_pose": None,
            "plane_level_tube_debt_d_pose": prefix_rows[-1]["plane_level_cache_replay"]["mean_d_pose_tube_debt"],
            "status": "BLOCKED_NO_COMPOSED_REALIZED_FRAMES",
            "transfer_forbidden": True,
        },
        "D4_equation": {
            "registered": False,
            "blocker": "D1_REAL_N600_COMPOSED_RGB_LATTICE_ANCHOR_ABSENT",
        },
        "reuse_manifest": {
            "seed_schema_and_parser": "tac.optimization.predict_project_schema",
            "receiver_project_stage_extended": "tac.optimization.predict_project_receiver",
            "uint8_lattice": "tac.optimization.uint8_lattice_feasibility",
            "joint_interval_solver": "tac.optimization.joint_seg_pose_rate",
            "full_kernel": "tac.optimization.resize_full_kernel",
            "seed_compose_measurement": str(stage_root.parent / "receipt.json"),
            "M2_realization_existence_anchor": str(m2_receipt_path),
            "new_with_justification": (
                "one bounded audit CLI is required because the existing measurement runner cannot distinguish "
                "a projected class-ID field from an RGB lattice input"
            ),
        },
        "input_custody": {
            "seed_path": str(seed_path),
            "seed_bytes": len(seed_bytes),
            "seed_sha256": hashlib.sha256(seed_bytes).hexdigest(),
            "stage_root": str(stage_root),
            "stage_count": len(stage_paths),
            "stage_first_sha256": _sha256(stage_paths[0]),
            "stage_last_sha256": _sha256(stage_paths[-1]),
        },
        "implementation_sources": implementation,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
            ).stdout.strip(),
            "storage_free_bytes_at_start": usage.free,
            "mmap_or_chunk_policy": "600 preserved JSON stages read in bounded pair order; no camera tensor materialized",
            "automatic_cleanup": "atomic temporary JSON files removed after replace; no rebuildable bulk produced",
        },
        "authority": {
            "axis": AXIS,
            "pointer": POINTER,
            "pointer_moved": False,
            "score_claim": False,
            "promotion_eligible": False,
            "main_landing_review_required": True,
        },
    }
    _atomic_json(output_root / "receipt.json", receipt)
    return receipt


def compose_source_control_receipt(
    audit_receipt: Mapping[str, Any],
    source_control: Mapping[str, Any],
    *,
    gt_cache_path: Path,
) -> dict[str, Any]:
    """Attach the measured charged control while preserving the seed blocker."""

    receipt = dict(audit_receipt)
    prefixes = source_control["prefix_ladder"]
    n600 = prefixes[-1]
    receipt.update(
        {
            "schema": SCHEMA,
            "lane_id": "lane_realization_g2b_supportfill_578_20260721",
            "verdict": "SOURCE_RGB_CONTROL_EXACT_ZERO_BYTE_CELLS_TO_RGB_PREMISE_FALSIFIED",
            "verdict_scope": (
                "real n16/n64/n600 source-derived RGB-plane control proves the downstream canonical support-fill "
                "and factor-2 lattice, but no decoder-derived cells-to-RGB or frame0 pose synthesis exists; "
                "this is a handoff-premise negative, not a lattice or learned-generator family negative"
            ),
            "D1_source_plane_control_ladder": prefixes,
            "D1_implementation": {
                **receipt["D1_implementation"],
                "status": "MEASURED_SOURCE_RGB_CONTROL_EXACT_SEED_PATH_STILL_BLOCKED",
                "real_n600_uint8_factor2_exact": n600["uint8_factor2_exact_fraction"],
                "real_seed_rgb_input_status": "ABSENT_DECODER_DERIVED_CELLS_TO_RGB",
                "source_rgb_control_status": "PRESENT_ENCODER_SUPPLIED_COUNTED",
                "semantic_cells_to_rgb_exact_pairs_n600": n600["semantic_cells_to_rgb_exact_pair_count"],
            },
            "D2_cost": {
                **receipt["D2_cost"],
                "additional_seed_bytes": n600["additional_seed_bytes_total"],
                "additional_seed_bytes_per_pair": n600["additional_seed_bytes_per_pair"],
                "zero_byte_target_met": n600["zero_added_seed_byte_target_met"],
                "source_control_total_decode_seconds": n600["timings_seconds_sum"]["lattice_double_decode"],
                "source_control_mean_decode_seconds_per_pair": n600["timings_seconds_mean_per_pair"][
                    "lattice_double_decode"
                ],
                "status": "MEASURED_CHARGED_SOURCE_RGB_CONTROL_NOT_SEED_RECEIVER",
            },
            "D3_pose": {
                "realized_frame_d_pose": n600["mean_d_pose_realized_vs_frozen_target"],
                "realized_outside_declared_tube_d_pose": n600["mean_d_pose_realized_outside_declared_tube"],
                "pose_within_declared_tube_pairs": n600["pose_within_declared_tube_pair_count"],
                "status": "MEASURED_SOURCE_RGB_CONTROL_ONLY",
                "transfer_forbidden": True,
            },
            "D4_equation": {
                "registered": False,
                "blocker": "D2_ZERO_BYTE_SEMANTIC_CELLS_TO_RGB_ADMISSION_FALSE",
                "source_control_anchor_ready": True,
                "required_evaluator": "predict_project_realization_admissibility_v1",
            },
            "source_control": {
                "receipt_path": str(Path(source_control["stage_root"]).parent / "receipt.json"),
                "receipt_sha256": _sha256(Path(source_control["stage_root"]).parent / "receipt.json"),
                "config_sha256": source_control["config_sha256"],
                "gt_cache_path": str(gt_cache_path),
                "gt_cache_sha256": GT_CACHE_SHA256,
            },
            "authority": {
                **receipt["authority"],
                "main_landing_review_required": True,
            },
        }
    )
    receipt["reuse_manifest"] = {
        **receipt["reuse_manifest"],
        "canonical_support_fill_actual_direction": (
            "tac.optimization.uint8_lattice_feasibility.realize_factor2_uint8_scorer_plane: "
            "uint8 RGB scorer plane to camera RGB"
        ),
        "tie_aware_exactness": "tac.optimization.tie_aware_preimage",
        "source_plane_custody_chain": "#547/#549 exact rational rounded gt_f0/gt_f1 range(A) control",
        "cells_to_rgb_decoder": None,
        "new_with_justification": (
            "the predecessor audit is extended in place with a resumable charged source-plane control; "
            "no palette, source table in decoder code, or forked measurement CLI is introduced"
        ),
    }
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/seeds/seed_compose_b2_loose.ppcs"),
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/hard_oracle_n600/stages"),
    )
    parser.add_argument(
        "--m2-receipt",
        type=Path,
        default=REPO / ".omx/research/m2_live_target_selection_20260720T1548Z.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/realization_g2b_20260721"),
    )
    parser.add_argument(
        "--gt-cache",
        type=Path,
        default=Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"),
    )
    default_upstream = REPO / "upstream"
    if not default_upstream.is_dir():
        default_upstream = Path("/Users/adpena/Projects/pact/upstream")
    parser.add_argument("--upstream", type=Path, default=default_upstream)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--interior-rungs",
        action="store_true",
        help="measure the zero-byte R1-R3 and counted R4 cell-interior receiver ladder",
    )
    parser.add_argument(
        "--contextual-predict-base",
        action="store_true",
        help="measure sequential G1-advected RGB prediction plus margin-banded projection",
    )
    parser.add_argument(
        "--stop-after-prefix",
        type=int,
        choices=PREFIXES,
        default=PAIR_COUNT,
        help="preserve stages and stop after n16, n64, or n600",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="write only the inherited label-vs-RGB audit; skip native scorer measurement",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sum((args.audit_only, args.interior_rungs, args.contextual_predict_base)) > 1:
        raise RealizationAuditError(
            "--audit-only, --interior-rungs, and --contextual-predict-base are mutually exclusive"
        )
    if args.contextual_predict_base:
        receipt = run_contextual_predict_base(
            seed_path=args.seed.resolve(),
            gt_cache_path=args.gt_cache.resolve(),
            upstream=args.upstream.resolve(),
            output_root=args.output_root.resolve(),
            chunk_size=args.chunk_size,
            threads=args.threads,
            stop_after_prefix=args.stop_after_prefix,
        )
        print(
            json.dumps(
                {
                    "receipt": str(args.output_root / "receipt.json"),
                    "verdict": receipt["verdict"],
                    "completed_prefix": receipt["completed_prefix"],
                },
                sort_keys=True,
            )
        )
        return 0
    receipt = run_audit(
        seed_path=args.seed.resolve(),
        stage_root=args.stage_root.resolve(),
        m2_receipt_path=args.m2_receipt.resolve(),
        output_root=args.output_root.resolve(),
    )
    if args.interior_rungs:
        receipt = run_interior_rungs(
            seed_path=args.seed.resolve(),
            gt_cache_path=args.gt_cache.resolve(),
            upstream=args.upstream.resolve(),
            output_root=args.output_root.resolve(),
            chunk_size=args.chunk_size,
            threads=args.threads,
            stop_after_prefix=args.stop_after_prefix,
        )
        _atomic_json(args.output_root.resolve() / "receipt.json", receipt)
    elif not args.audit_only:
        source_control = run_source_plane_control(
            seed_path=args.seed.resolve(),
            gt_cache_path=args.gt_cache.resolve(),
            upstream=args.upstream.resolve(),
            output_root=args.output_root.resolve(),
            chunk_size=args.chunk_size,
            threads=args.threads,
        )
        receipt = compose_source_control_receipt(
            receipt,
            source_control,
            gt_cache_path=args.gt_cache.resolve(),
        )
        _atomic_json(args.output_root.resolve() / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "receipt": str(args.output_root / "receipt.json"),
                "verdict": receipt["verdict"],
                "completed_prefix": (receipt.get("completed_prefix") if args.interior_rungs else PAIR_COUNT),
                "winning_rung": receipt.get("winning_rung"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
