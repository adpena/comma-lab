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
import math
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
    load_lane_chart,
    parse_static_charts,
    relative_adjacent_xi,
    render_lane_mask,
)
from tac.optimization.predictor_upgrade_xi_chart import (  # noqa: E402
    predict_cell_field as predict_chart_cell_field,
)
from tac.optimization.realized_secant_custody import (  # noqa: E402
    BIDIRECTIONAL_RECEIPT_SCHEMA,
    BidirectionalRungObservation,
    SecantObservation,
    WriteSecantObservation,
    build_bidirectional_rung_observation,
    build_bidirectional_trust_region_custody,
    build_pair_trust_region_custody,
    build_trust_regions,
    decode_coefficient_packet,
    encode_coefficient_packet,
    select_best_bidirectional_rungs,
    solve_minimal_norm_inequalities,
    validate_bidirectional_receipt,
)
from tac.optimization.realized_secant_custody import (  # noqa: E402
    RECEIPT_SCHEMA as SECANT_RECEIPT_SCHEMA,
)
from tac.optimization.realized_secant_custody import (  # noqa: E402
    PairSolveStatus as SecantPairSolveStatus,
)
from tac.optimization.realized_secant_custody import (  # noqa: E402
    QPStatus as SecantQPStatus,
)
from tac.optimization.realized_secant_custody import (  # noqa: E402
    canonical_sha256 as secant_canonical_sha256,
)
from tac.optimization.realized_secant_custody import (  # noqa: E402
    validate_receipt as validate_secant_receipt,
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
FRAME0_PRIOR_RACE_SCHEMA: Final = "realization_g2d_frame0_prior_race_receipt.v1"
SECANT_STAGE_SCHEMA: Final = "realization_g2e_secant_pair.v1"
SECANT_CONFIG_SCHEMA: Final = "realization_g2e_secant_config.v1"
AMPLITUDE_STAGE_SCHEMA: Final = "realization_g2f_bidirectional_amplitude_pair.v1"
AMPLITUDE_CONFIG_SCHEMA: Final = "realization_g2f_bidirectional_amplitude_config.v1"
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
SECANT_LANE_ID: Final = "lane_g2e_secant_custody_578_20260721"
SECANT_SIGNED_AMPLITUDES: Final = (4.0, -4.0, 8.0, -8.0)
SECANT_RELATIVE_RESIDUAL_TOLERANCE: Final = 0.35
SECANT_REQUIRED_MARGIN: Final = 1e-4
SECANT_CHART_RANK: Final = 4
AMPLITUDE_LANE_ID: Final = "lane_g2f_bidirectional_amplitude_ladder_578_20260721"
G2E_PRIOR_RECEIPT_SHA256: Final = "e89157bb8dfc6b11b20aecccd4dbe82113ea706c9a1eb054de989c26a740dbc4"
AMPLITUDE_LSB_LAWREF_ID: Final = "witness_realization_lsb_regime_v1"
AMPLITUDE_R_OPERATOR_LAWREF_ID: Final = "separable_resize_full_kernel_direct_sum_v1"
FRAME0_PALETTE_MAGIC: Final = b"G2PAL1"
FRAME0_STATIC_CHART_SHA256: Final = "2b3665c47f7a404e7ac8ea1b30cad768d4ce2a84fd998e167d230b522e18ba43"
DEFAULT_FRAME0_STATIC_CHART: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/predictor_upgrade_20260721/"
    "canonical_g1_d4_fixed_20260721/charts/static_charts_n64.pxch"
)
DEFAULT_FRAME0_LANE_CHART: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/boundary_inverse_20260721/"
    "run_20260721T052100Z_threshold0p5/coherent_slot_none_dash.lbnd2"
)
DEFAULT_VJP_CAMPAIGN: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/extension_n600_20260720/campaign_receipt.json"
)
DEFAULT_M1_BAND_RECEIPT: Final = Path("/Volumes/VertigoDataTier/pact/evidence/m1_band_manifest_20260720/receipt.json")
DEFAULT_M1_INNER_JACOBIAN: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/m1_band_manifest_20260720/records/inner_jacobian_secant_qp.json"
)
DEFAULT_RANK4_PROTOTYPE_RECEIPT: Final = (
    REPO / ".omx/research/prereq_surfaces_flush_20260720/surface_2_rank4_prototype_bank.json"
)
DEFAULT_G2E_PRIOR_RECEIPT: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/g2e_secant_20260721/final_hardened/receipt.json"
)


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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Create or verify one immutable byte artifact."""

    if path.exists():
        if path.read_bytes() != payload:
            raise RealizationAuditError(f"immutable byte artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


def serialize_frozen_scorer_palette() -> bytes:
    """Serialize the counted frozen-scorer class-to-RGB constants."""

    return FRAME0_PALETTE_MAGIC + R2_MAX_MARGIN_PALETTE.tobytes(order="C")


def parse_frozen_scorer_palette(payload: bytes) -> np.ndarray:
    """Parse the tiny counted class-to-RGB packet with canonical round-trip."""

    expected = len(FRAME0_PALETTE_MAGIC) + R2_MAX_MARGIN_PALETTE.nbytes
    if len(payload) != expected or payload[: len(FRAME0_PALETTE_MAGIC)] != FRAME0_PALETTE_MAGIC:
        raise RealizationAuditError("frozen-scorer palette packet mismatch")
    palette = (
        np.frombuffer(
            payload,
            dtype=np.uint8,
            count=R2_MAX_MARGIN_PALETTE.size,
            offset=len(FRAME0_PALETTE_MAGIC),
        )
        .reshape(R2_MAX_MARGIN_PALETTE.shape)
        .copy()
    )
    if not np.array_equal(palette, R2_MAX_MARGIN_PALETTE):
        raise RealizationAuditError("frozen-scorer palette constants drifted")
    if serialize_frozen_scorer_palette() != payload:
        raise RealizationAuditError("frozen-scorer palette parse-back is noncanonical")
    return palette


def protect_seed_class_sites(
    cells: np.ndarray,
    seed: Mapping[str, Any],
    *,
    radius: int = 1,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Apply the counted #208 per-class site protection to a class prior."""

    if radius < 0:
        raise RealizationAuditError("site-protection radius must be nonnegative")
    out = np.asarray(cells, dtype=np.uint8).copy()
    if out.shape != SCORER_HW or np.any(out >= len(R2_MAX_MARGIN_PALETTE)):
        raise RealizationAuditError("site-protection class field mismatch")
    quantum = seed["ground_chart"]["coordinate_quantum"]
    scale = float(quantum["numerator"]) / float(quantum["denominator"])
    protected: list[dict[str, int]] = []
    for site in seed["ground_chart"]["cells"]:
        class_id = int(site["class_id"])
        y = round(int(site["site_y_q"]) * scale)
        x = round(int(site["site_x_q"]) * scale)
        y0, y1 = max(0, y - radius), min(out.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(out.shape[1], x + radius + 1)
        if y0 >= y1 or x0 >= x1 or not 0 <= class_id < len(R2_MAX_MARGIN_PALETTE):
            raise RealizationAuditError("ground-chart protected site is outside canonical geometry")
        out[y0:y1, x0:x1] = class_id
        protected.append(
            {
                "class_id": class_id,
                "y": y,
                "x": x,
                "protected_pixels": (y1 - y0) * (x1 - x0),
            }
        )
    return out, protected


def openpilot_frame0_class_prior(
    *,
    seed: Mapping[str, Any],
    static_charts: Any,
    lane_mask: np.ndarray,
    geom: g1_warp.GroundHomographyGeom,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Compose the existing per-class geometric solve for pair-0 initialization."""

    cells = predict_chart_cell_field(
        pair_index=0,
        prior_decoded_field=None,
        charts=static_charts,
        relative_xi=np.zeros(6, dtype=np.float64),
        worldsheet_geom=geom,
        lane_mask=np.asarray(lane_mask, dtype=np.bool_),
        movable_tracks=seed["movable_tracks"],
    )
    return protect_seed_class_sites(np.asarray(cells, dtype=np.uint8), seed)


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


def _head_patch_144(feature: np.ndarray, row: int, col: int) -> tuple[float, ...]:
    """Gather the zero-padded 16x3x3 input patch to segmentation_head[0]."""

    if feature.ndim != 3 or feature.shape[0] != 16:
        raise RealizationAuditError("candidate-state head input must have shape 16xHxW")
    padded = np.pad(feature, ((0, 0), (1, 1), (1, 1)), mode="constant")
    patch = padded[:, row : row + 3, col : col + 3]
    if patch.shape != (16, 3, 3) or not np.isfinite(patch).all():
        raise RealizationAuditError("candidate-state 144D head patch is malformed")
    return tuple(float(value) for value in patch.reshape(-1))


def _candidate_seg_state(
    net: Any,
    torch: Any,
    frame0: np.ndarray,
    frame1: np.ndarray,
    scorer_plane: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
    *,
    fixed_rivals: Sequence[int] | None = None,
    with_jacobian: bool,
) -> dict[str, Any]:
    """Capture fresh candidate logits/head input and optional local RGB pullback."""

    if not constraints:
        raise RealizationAuditError("realized-secant custody requires declared writes")
    pair = np.stack((frame0, frame1), axis=0)[None]
    camera_tensor = torch.from_numpy(np.ascontiguousarray(pair)).permute(0, 1, 4, 2, 3).contiguous().float()
    scorer_input = net.segnet.preprocess_input(camera_tensor).detach()
    expected = torch.from_numpy(np.ascontiguousarray(scorer_plane)).permute(2, 0, 1).unsqueeze(0).float()
    receiver_input_maxabs = float(torch.max(torch.abs(scorer_input - expected)).item())
    if receiver_input_maxabs > 1e-4:
        raise RealizationAuditError(f"real receiver/scorer input drift {receiver_input_maxabs} exceeds 1e-4")
    scorer_input.requires_grad_(with_jacobian)
    captured: dict[str, Any] = {}

    def head_pre_hook(_module: Any, inputs: Any) -> None:
        captured["head_input"] = inputs[0]

    head = net.segnet.segmentation_head[0]
    handle = head.register_forward_pre_hook(head_pre_hook)
    try:
        logits_tensor = net.segnet(scorer_input)[0]
    finally:
        handle.remove()
    if "head_input" not in captured or tuple(logits_tensor.shape) != (5, *SCORER_HW):
        raise RealizationAuditError("candidate-state SegNet capture geometry mismatch")
    logits = logits_tensor.detach().cpu().numpy().astype(np.float64)
    head_input = captured["head_input"][0].detach().cpu().numpy().astype(np.float64)
    argmax = logits.argmax(axis=0).astype(np.uint8)
    margins: list[float] = []
    rivals: list[int] = []
    current_classes: list[int] = []
    patches: list[tuple[float, ...]] = []
    margin_tensors: list[Any] = []
    for ordinal, constraint in enumerate(constraints):
        target = int(constraint["cell_id"])
        row, col = int(constraint["y"]), int(constraint["x"])
        current = int(argmax[row, col])
        if fixed_rivals is None:
            if current != target:
                rival = current
            else:
                rival_values = logits[:, row, col].copy()
                rival_values[target] = -np.inf
                rival = int(np.argmax(rival_values))
        else:
            rival = int(fixed_rivals[ordinal])
            if not 0 <= rival < 5 or rival == target:
                raise RealizationAuditError("fixed candidate-state rival is invalid")
        margins.append(float(logits[target, row, col] - logits[rival, row, col]))
        rivals.append(rival)
        current_classes.append(current)
        patches.append(_head_patch_144(head_input, row, col))
        margin_tensors.append(logits_tensor[target, row, col] - logits_tensor[rival, row, col])

    jacobian: np.ndarray | None = None
    if with_jacobian:
        site_count = len(constraints)
        jacobian = np.empty((site_count, 3 * site_count), dtype=np.float64)
        for row_index, margin_tensor in enumerate(margin_tensors):
            gradient = torch.autograd.grad(
                margin_tensor,
                scorer_input,
                retain_graph=row_index + 1 < site_count,
                create_graph=False,
            )[0][0]
            local = []
            for constraint in constraints:
                local.extend(
                    float(value)
                    for value in gradient[:, int(constraint["y"]), int(constraint["x"])].detach().cpu().tolist()
                )
            jacobian[row_index] = local
        if not np.isfinite(jacobian).all():
            raise RealizationAuditError("candidate-state local margin Jacobian is nonfinite")
    scorer_array = scorer_input.detach().cpu().numpy().astype(np.float32, copy=False)
    return {
        "logits": logits,
        "argmax": argmax,
        "margins": np.asarray(margins, dtype=np.float64),
        "rivals": tuple(rivals),
        "current_classes": tuple(current_classes),
        "feature_patches": tuple(patches),
        "local_jacobian": jacobian,
        "receiver_input_maxabs": receiver_input_maxabs,
        "segnet_input_sha256": hashlib.sha256(np.ascontiguousarray(scorer_array).tobytes()).hexdigest(),
        "logits_sha256": hashlib.sha256(np.ascontiguousarray(logits.astype("<f4")).tobytes()).hexdigest(),
    }


def _rank4_chart_directions(local_jacobian: np.ndarray) -> np.ndarray:
    """Build four deterministic response columns, zero-padding unavailable directions."""

    jacobian = np.asarray(local_jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] == 0 or jacobian.shape[1] == 0:
        raise RealizationAuditError("candidate-state chart Jacobian geometry is too small")
    if not np.isfinite(jacobian).all():
        raise RealizationAuditError("candidate-state chart Jacobian is nonfinite")
    _, _, vh = np.linalg.svd(jacobian, full_matrices=True)
    available_directions = min(SECANT_CHART_RANK, vh.shape[0])
    directions = np.zeros((jacobian.shape[1], SECANT_CHART_RANK), dtype=np.float64)
    directions[:, :available_directions] = vh[:available_directions].T
    for column in range(available_directions):
        vector = directions[:, column]
        scale = float(np.max(np.abs(vector), initial=0.0))
        if scale == 0.0 or not math.isfinite(scale):
            raise RealizationAuditError("candidate-state chart produced a zero/nonfinite column")
        vector /= scale
        response_sum = float(np.sum(jacobian @ vector))
        pivot = int(np.argmax(np.abs(vector)))
        if response_sum < 0.0 or (response_sum == 0.0 and vector[pivot] < 0.0):
            vector *= -1.0
    return directions


def _local_rgb_values(
    plane: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    sites = [(int(row["y"]), int(row["x"])) for row in constraints]
    if len(sites) != len(set(sites)):
        raise RealizationAuditError("declared writes contain duplicate RGB sites")
    return np.asarray([plane[row, col] for row, col in sites], dtype=np.float64).reshape(-1)


def _apply_local_chart_delta(
    baseline: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Round one local chart move once and apply uint8 bounds explicitly."""

    base_values = _local_rgb_values(baseline, constraints)
    raw_delta = np.asarray(delta, dtype=np.float64)
    if raw_delta.shape != base_values.shape or not np.isfinite(raw_delta).all():
        raise RealizationAuditError("local chart delta geometry/value mismatch")
    rounded = np.rint(raw_delta)
    unbounded = base_values + rounded
    saturation = int(np.count_nonzero((unbounded < 0.0) | (unbounded > 255.0)))
    bounded = np.clip(unbounded, 0.0, 255.0).astype(np.uint8)
    candidate = np.asarray(baseline, dtype=np.uint8).copy()
    for ordinal, constraint in enumerate(constraints):
        candidate[int(constraint["y"]), int(constraint["x"])] = bounded[3 * ordinal : 3 * ordinal + 3]
    return candidate, bounded.astype(np.float64) - base_values, saturation


def derive_bidirectional_amplitude_ladder(
    kernel: FullResizeKernel,
    g2e_prior_receipt: Mapping[str, Any],
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Derive a dyadic scorer-plane ladder from exact ``R`` gain and G2e extent."""

    denominators = {
        int(row.denominator) * int(col.denominator)
        for row in kernel.operator.row_supports
        for col in kernel.operator.col_supports
    }
    if len(denominators) != 1:
        raise RealizationAuditError("G2f R operator lacks one exact denominator")
    denominator = denominators.pop()
    gain_numerators = [
        sum(
            int(row_numerator) * int(col_numerator)
            for row_numerator in row.numerators
            for col_numerator in col.numerators
        )
        for row in kernel.operator.row_supports
        for col in kernel.operator.col_supports
    ]
    max_gain_numerator = max(gain_numerators)
    if max_gain_numerator <= 0 or any(value != max_gain_numerator for value in gain_numerators):
        raise RealizationAuditError("G2f exact R induced-Linf gain is not spatially uniform")
    induced_linf_gain = max_gain_numerator / denominator
    if not math.isclose(induced_linf_gain, 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise RealizationAuditError("G2f exact convex R gain drifted from unity")
    half_lsb_amplitude = denominator / (2.0 * max_gain_numerator)
    prior_rows = g2e_prior_receipt.get("secant_observations")
    if not isinstance(prior_rows, list) or len(prior_rows) != 64:
        raise RealizationAuditError("G2f requires the complete 64-row G2e rung-0 prior")
    prior_abs_amplitudes = sorted({abs(float(row["signed_amplitude"])) for row in prior_rows})
    if not prior_abs_amplitudes or prior_abs_amplitudes[0] <= 0.0:
        raise RealizationAuditError("G2e prior amplitudes are malformed")
    upper_bracket = 2.0 * prior_abs_amplitudes[-1]
    amplitudes: list[float] = []
    amplitude = half_lsb_amplitude
    while amplitude <= upper_bracket * (1.0 + 1e-12):
        amplitudes.append(float(amplitude))
        amplitude *= 2.0
    if amplitudes[-1] < upper_bracket:
        amplitudes.append(float(amplitude))
    custody = {
        "schema": "g2f_amplitude_ladder_derivation.v1",
        "lsb_lawref_id": AMPLITUDE_LSB_LAWREF_ID,
        "r_operator_lawref_id": AMPLITUDE_R_OPERATOR_LAWREF_ID,
        "exact_r_common_denominator": denominator,
        "exact_r_max_abs_row_sum_numerator": max_gain_numerator,
        "exact_r_induced_linf_gain": induced_linf_gain,
        "sub_lsb_boundary": "1/(2*||R||_inf)",
        "half_lsb_amplitude": half_lsb_amplitude,
        "g2e_prior_absolute_amplitudes": prior_abs_amplitudes,
        "upper_bracket_rule": "one dyadic octave above maximum G2e prior amplitude",
        "upper_bracket_amplitude": upper_bracket,
        "amplitudes": amplitudes,
        "constant_guessed": False,
    }
    return tuple(amplitudes), custody


def _effective_chart_direction_count(chart_directions: np.ndarray) -> int:
    active = np.max(np.abs(np.asarray(chart_directions, dtype=np.float64)), axis=0) > 0.0
    count = int(np.count_nonzero(active))
    if count < 1 or count > SECANT_CHART_RANK or not np.all(active[:count]) or np.any(active[count:]):
        raise RealizationAuditError("G2f effective chart directions are not a contiguous rank prefix")
    return count


def _measure_or_reuse_secant_branch(
    *,
    pair_index: int,
    direction_index: int,
    amplitude: float,
    base1: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
    chart_direction: np.ndarray,
    local_jacobian: np.ndarray,
    baseline_state: Mapping[str, Any],
    realized0_frame: np.ndarray,
    represented: np.ndarray,
    seed_sha256: str,
    kernel: FullResizeKernel,
    net: Any,
    torch: Any,
    reused_prior: SecantObservation | None,
) -> tuple[SecantObservation, np.ndarray]:
    """Measure one signed branch or verify/reuse its immutable G2e observation."""

    probe_plane, actual_delta, saturation = _apply_local_chart_delta(
        base1,
        constraints,
        amplitude * chart_direction,
    )
    predicted = local_jacobian @ actual_delta
    if reused_prior is not None:
        if (
            reused_prior.pair_index != pair_index
            or reused_prior.column_index != direction_index
            or not math.isclose(reused_prior.signed_amplitude, amplitude, rel_tol=0.0, abs_tol=1e-12)
            or reused_prior.uint8_saturation_count != saturation
            or not math.isclose(
                reused_prior.applied_rgb_l2,
                float(np.linalg.norm(actual_delta)),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or not math.isclose(
                reused_prior.applied_rgb_linf,
                float(np.max(np.abs(actual_delta), initial=0.0)),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise RealizationAuditError("G2f reconstructed G2e branch geometry drifted")
        for ordinal, (constraint, prior_write) in enumerate(zip(constraints, reused_prior.writes, strict=True)):
            if (
                prior_write.ordinal != ordinal
                or prior_write.target_class != int(constraint["cell_id"])
                or prior_write.current_class != int(baseline_state["current_classes"][ordinal])
                or prior_write.margin_bucket != _margin_bucket(float(baseline_state["margins"][ordinal]))
                or not math.isclose(
                    prior_write.pre_margin,
                    float(baseline_state["margins"][ordinal]),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )
                or not math.isclose(
                    prior_write.predicted_margin_delta,
                    float(predicted[ordinal]),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )
            ):
                raise RealizationAuditError("G2f G2e prior write no longer matches candidate chart")
        return reused_prior, actual_delta

    probe_realized = _contextual_realize(
        probe_plane,
        represented,
        seed_sha256=seed_sha256,
        additional_seed_bytes=0,
        generator_id=f"g2f_bidirectional_direction_{direction_index}_amplitude_{amplitude:+.12g}",
        kernel=kernel,
    )
    probe_state = _candidate_seg_state(
        net,
        torch,
        realized0_frame,
        probe_realized["frame"],
        probe_plane,
        constraints,
        fixed_rivals=baseline_state["rivals"],
        with_jacobian=False,
    )
    realized_delta = np.asarray(probe_state["margins"], dtype=np.float64) - np.asarray(
        baseline_state["margins"], dtype=np.float64
    )
    writes = []
    for ordinal, constraint in enumerate(constraints):
        predicted_value = float(predicted[ordinal])
        realized_value = float(realized_delta[ordinal])
        writes.append(
            WriteSecantObservation(
                ordinal=ordinal,
                target_class=int(constraint["cell_id"]),
                current_class=int(baseline_state["current_classes"][ordinal]),
                pre_margin=float(baseline_state["margins"][ordinal]),
                margin_bucket=_margin_bucket(float(baseline_state["margins"][ordinal])),
                expected_sign=1 if predicted_value >= 0.0 else -1,
                feature_displacement=tuple(
                    float(after - before)
                    for after, before in zip(
                        probe_state["feature_patches"][ordinal],
                        baseline_state["feature_patches"][ordinal],
                        strict=True,
                    )
                ),
                predicted_margin_delta=predicted_value,
                realized_margin_delta=realized_value,
                secant_ratio=realized_value / amplitude,
            )
        )
    return (
        SecantObservation(
            pair_index=pair_index,
            column_index=direction_index,
            signed_amplitude=amplitude,
            applied_rgb_l2=float(np.linalg.norm(actual_delta)),
            applied_rgb_linf=float(np.max(np.abs(actual_delta), initial=0.0)),
            uint8_saturation_count=saturation,
            writes=tuple(writes),
        ),
        actual_delta,
    )


def _exception_parseback(
    baseline: np.ndarray,
    constraints: Sequence[Mapping[str, Any]],
    candidate: np.ndarray,
) -> tuple[bytes, np.ndarray, list[int]]:
    """Use #557 for nonempty residuals and represent no payload as zero bytes."""

    if np.array_equal(baseline, candidate):
        return b"", np.asarray(baseline, dtype=np.uint8).copy(), []
    payload, ordinals = encode_contextual_rgb_exceptions(baseline, candidate, constraints)
    parsed, parsed_ordinals = apply_contextual_rgb_exceptions(baseline, constraints, payload)
    if (
        parsed_ordinals != ordinals
        or not np.array_equal(parsed, candidate)
        or encode_contextual_rgb_exceptions(baseline, parsed, constraints)[0] != payload
    ):
        raise RealizationAuditError("G2e #557 parse-back/re-encode custody failed")
    return payload, parsed, ordinals


def _load_secant_stages(stage_dir: Path, sidecar_dir: Path, config_sha256: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not stage_dir.exists():
        return rows
    for path in sorted(stage_dir.glob("pair_*.json")):
        row = _load_json(path)
        pair_index = row.get("pair_index")
        if (
            row.get("schema") != SECANT_STAGE_SCHEMA
            or row.get("config_sha256") != config_sha256
            or isinstance(pair_index, bool)
            or not isinstance(pair_index, int)
            or pair_index in rows
        ):
            raise RealizationAuditError(f"G2e resume custody mismatch: {path}")
        sidecar = sidecar_dir / f"pair_{pair_index:04d}.g2dx"
        if not sidecar.is_file() or _sha256(sidecar) != row["correction_packet"]["sha256"]:
            raise RealizationAuditError(f"G2e correction sidecar custody mismatch: {sidecar}")
        if sidecar.stat().st_size != row["correction_packet"]["bytes"]:
            raise RealizationAuditError(f"G2e correction sidecar size mismatch: {sidecar}")
        rows[pair_index] = row
    return rows


def _load_amplitude_stages(
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
            row.get("schema") != AMPLITUDE_STAGE_SCHEMA
            or row.get("config_sha256") != config_sha256
            or isinstance(pair_index, bool)
            or not isinstance(pair_index, int)
            or pair_index in rows
        ):
            raise RealizationAuditError(f"G2f resume custody mismatch: {path}")
        sidecar = sidecar_dir / f"pair_{pair_index:04d}.g2dx"
        if not sidecar.is_file() or _sha256(sidecar) != row["correction_packet"]["sha256"]:
            raise RealizationAuditError(f"G2f correction sidecar custody mismatch: {sidecar}")
        if sidecar.stat().st_size != row["correction_packet"]["bytes"]:
            raise RealizationAuditError(f"G2f correction sidecar size mismatch: {sidecar}")
        rows[pair_index] = row
    return rows


def _response_summary_rows(
    observations: Sequence[BidirectionalRungObservation],
    *,
    key: str,
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[Any]] = {}
    for observation in observations:
        for write in observation.writes:
            identity = getattr(write, key)
            grouped.setdefault(identity, []).append(write)
    rows = []
    for identity, writes in sorted(grouped.items(), key=lambda item: str(item[0])):
        residuals = [
            abs(write.odd_realized_secant - write.odd_predicted_secant)
            / max(abs(write.odd_realized_secant), abs(write.odd_predicted_secant), 1e-12)
            for write in writes
        ]
        sign_consistent = [
            all(
                abs(predicted) > 1e-12 and abs(realized) > 1e-12 and predicted * realized > 0.0
                for predicted, realized in (
                    (
                        write.positive_predicted_margin_delta,
                        write.positive_realized_margin_delta,
                    ),
                    (
                        write.negative_predicted_margin_delta,
                        write.negative_realized_margin_delta,
                    ),
                    (write.odd_predicted_secant, write.odd_realized_secant),
                )
            )
            for write in writes
        ]
        rows.append(
            {
                key: identity,
                "response_count": len(writes),
                "sign_consistent_count": sum(sign_consistent),
                "sign_consistent_fraction": float(np.mean(sign_consistent)),
                "max_relative_residual": max(residuals),
                "mean_relative_residual": float(np.mean(residuals)),
                "mean_abs_odd_realized_secant": float(np.mean([abs(write.odd_realized_secant) for write in writes])),
                "mean_abs_even_realized_secant": float(np.mean([abs(write.even_realized_secant) for write in writes])),
            }
        )
    return rows


def summarize_bidirectional_amplitude_prefix(
    prefix: int,
    stage_rows: Sequence[Mapping[str, Any]],
    *,
    base_bytes: int,
) -> dict[str, Any]:
    """Aggregate G2f response-curve, trust, receiver, rate, and Pose custody."""

    if prefix not in PREFIXES or len(stage_rows) != prefix:
        raise RealizationAuditError("G2f prefix must be exactly n16/n64/n600")
    if [row.get("pair_index") for row in stage_rows] != list(range(prefix)):
        raise RealizationAuditError("G2f prefix is not contiguous from zero")
    if any(row.get("schema") != AMPLITUDE_STAGE_SCHEMA for row in stage_rows):
        raise RealizationAuditError("G2f prefix stage schema mismatch")
    observations = [
        BidirectionalRungObservation.from_dict(raw)
        for stage in stage_rows
        for raw in stage["bidirectional_observations"]
    ]
    trust_regions = [row for stage in stage_rows for row in stage["amplitude_trust_regions"]]
    amplitudes = sorted({observation.amplitude for observation in observations})
    pair_direction_rung: dict[tuple[int, int, int], list[Mapping[str, Any]]] = {}
    for row in trust_regions:
        key = (int(row["pair_index"]), int(row["direction_index"]), int(row["rung_index"]))
        pair_direction_rung.setdefault(key, []).append(row)
    usable_by_rung = []
    for rung_index, amplitude in enumerate(amplitudes):
        groups = [
            rows
            for (pair, direction, rung), rows in sorted(pair_direction_rung.items())
            if pair < prefix and direction >= 0 and rung == rung_index
        ]
        usable_groups = sum(all(row["usable"] is True for row in rows) for rows in groups)
        rung_regions = [row for rows in groups for row in rows]
        usable_by_rung.append(
            {
                "rung_index": rung_index,
                "amplitude": amplitude,
                "pair_direction_count": len(groups),
                "usable_pair_direction_count": usable_groups,
                "usable_pair_direction_fraction": usable_groups / len(groups),
                "trust_region_count": len(rung_regions),
                "usable_trust_region_count": sum(row["usable"] is True for row in rung_regions),
                "positive_uint8_saturation_count": sum(
                    int(row["positive_uint8_saturation_count"]) for row in rung_regions
                ),
                "negative_uint8_saturation_count": sum(
                    int(row["negative_uint8_saturation_count"]) for row in rung_regions
                ),
            }
        )
    max_usable_fraction = max(row["usable_pair_direction_fraction"] for row in usable_by_rung)
    knee = (
        min(
            (row for row in usable_by_rung if row["usable_pair_direction_fraction"] == max_usable_fraction),
            key=lambda row: row["amplitude"],
        )
        if max_usable_fraction > 0.0
        else None
    )
    refusal_reasons = Counter(
        reason for row in trust_regions if row["usable"] is False for reason in row["refusal_reasons"]
    )
    failed_regions = [row for row in trust_regions if row["usable"] is False]
    writes = [write for row in stage_rows for write in row["declared_write_survival"]]
    hard = [row["hard_oracle"] for row in stage_rows]
    correction_bytes = int(sum(row["correction_packet"]["bytes"] for row in stage_rows))
    total_bytes = base_bytes + correction_bytes
    factor2_exact_count = sum(row["uint8_factor2_exact"] is True for row in stage_rows)
    double_decode_count = sum(row["double_decode_identical"] is True for row in stage_rows)
    whole_exact_count = sum(row["realized_argmax_equals_description"] is True for row in hard)
    tube_count = sum(row["pose_within_declared_tube"] is True for row in hard)
    from tac.canonical_equations.predict_project_realization_admissibility_20260721 import (
        predict_project_realization_certificate,
    )

    admissibility = predict_project_realization_certificate(
        pair_count=prefix,
        uint8_factor2_exact_fraction=factor2_exact_count / prefix,
        double_decode_identical_pair_count=double_decode_count,
        semantic_cells_to_rgb_exact_pair_count=whole_exact_count,
        pose_within_declared_tube_pair_count=tube_count,
        additional_seed_bytes=(base_bytes - CONTEXTUAL_SEED_BASELINE_BYTES + correction_bytes),
        receiver_derived_rgb=False,
    )
    return {
        "schema": "realization_g2f_bidirectional_amplitude_prefix.v1",
        "n": prefix,
        "D1_response_curve": {
            "bidirectional_rung_observation_count": len(observations),
            "signed_branch_count": 2 * len(observations),
            "g2e_prior_branch_reuse_count": sum(
                observation.positive_source == "REUSED_G2E_RUNG0_PRIOR" for observation in observations
            )
            + sum(observation.negative_source == "REUSED_G2E_RUNG0_PRIOR" for observation in observations),
            "usable_fraction_vs_rung": usable_by_rung,
            "knee_rung": knee,
            "trust_region_count": len(trust_regions),
            "usable_trust_region_count": sum(row["usable"] is True for row in trust_regions),
            "refusal_reason_counts": dict(sorted(refusal_reasons.items())),
            "by_class": _response_summary_rows(observations, key="target_class"),
            "by_margin_bucket": _response_summary_rows(observations, key="margin_bucket"),
            "by_stratum": _response_summary_rows(observations, key="stratum"),
        },
        "D2_selected_trust": {
            "all_directions_usable_pair_count": sum(
                all(selection["selected"] is True for selection in row["selected_rungs"]) for row in stage_rows
            ),
            "pair_count": prefix,
            "failed_trust_region_count": len(failed_regions),
            "residual_failed_region_count": sum(
                "RELATIVE_SECANT_RESIDUAL" in row["refusal_reasons"] for row in failed_regions
            ),
            "sign_failed_region_count": sum(
                "BIDIRECTIONAL_SIGN_OR_ZERO" in row["refusal_reasons"] for row in failed_regions
            ),
            "saturation_associated_failed_region_count": sum(
                row["saturation_associated"] is True for row in failed_regions
            ),
        },
        "D3_receiver_closed_qp": {
            "admitted_pair_count": sum(row["pair_solve"]["admitted"] is True for row in stage_rows),
            "solve_status_histogram": dict(sorted(Counter(row["pair_solve"]["status"] for row in stage_rows).items())),
            "hard_oracle_is_admission_authority": True,
        },
        "uint8_factor2_exact_pair_count": factor2_exact_count,
        "double_decode_identical_pair_count": double_decode_count,
        "whole_description_exact_pair_count": whole_exact_count,
        "all_declared_writes_survive_pair_count": sum(row["all_declared_writes_survive"] is True for row in hard),
        "declared_write_count": len(writes),
        "surviving_write_count": sum(row["survives"] is True for row in writes),
        "declared_write_survival_fraction": float(np.mean([row["survives"] for row in writes])),
        "by_class": _aggregate_survival(writes, "class_id"),
        "by_stratum": _aggregate_survival(writes, "stratum"),
        "by_margin_bucket": _aggregate_survival(writes, "margin_bucket"),
        "mean_d_seg_realized_vs_frozen_target": float(
            np.mean([row["d_seg_realized_vs_frozen_target"] for row in hard])
        ),
        "mean_d_pose_realized_vs_frozen_target": float(
            np.mean([row["d_pose_realized_vs_frozen_target"] for row in hard])
        ),
        "mean_d_pose_declared_tube_debt": float(
            np.mean([row["d_pose_realized_outside_declared_tube"] for row in hard])
        ),
        "tube_contained_pair_count": tube_count,
        "predict_project_realization_admissibility_v1": admissibility,
        "byte_accounting": {
            "base_bytes": base_bytes,
            "correction_bytes": correction_bytes,
            "total_bytes": total_bytes,
            "target_box_bytes": CONTEXTUAL_TARGET_BOX_BYTES,
            "headroom_vs_target_box_bytes": CONTEXTUAL_TARGET_BOX_BYTES - total_bytes,
            "fits_target_box": total_bytes <= CONTEXTUAL_TARGET_BOX_BYTES,
            "rate_break_even": "25/37,545,489 score units per correction byte",
        },
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }


def summarize_secant_prefix(prefix: int, stage_rows: Sequence[Mapping[str, Any]], *, base_bytes: int) -> dict[str, Any]:
    """Aggregate G2e whole-description, write, trust, pose, and rate custody."""

    if prefix not in PREFIXES or len(stage_rows) != prefix:
        raise RealizationAuditError("G2e prefix must be exactly n16/n64/n600")
    if [row.get("pair_index") for row in stage_rows] != list(range(prefix)):
        raise RealizationAuditError("G2e prefix is not contiguous from zero")
    if any(row.get("schema") != SECANT_STAGE_SCHEMA for row in stage_rows):
        raise RealizationAuditError("G2e prefix stage schema mismatch")
    writes = [write for row in stage_rows for write in row["declared_write_survival"]]
    hard = [row["hard_oracle"] for row in stage_rows]
    correction_bytes = int(sum(row["correction_packet"]["bytes"] for row in stage_rows))
    total_bytes = base_bytes + correction_bytes
    factor2_exact_count = sum(row["uint8_factor2_exact"] is True for row in stage_rows)
    double_decode_count = sum(row["double_decode_identical"] is True for row in stage_rows)
    whole_exact_count = sum(row["realized_argmax_equals_description"] is True for row in hard)
    tube_count = sum(row["pose_within_declared_tube"] is True for row in hard)
    from tac.canonical_equations.predict_project_realization_admissibility_20260721 import (
        predict_project_realization_certificate,
    )

    admissibility = predict_project_realization_certificate(
        pair_count=prefix,
        uint8_factor2_exact_fraction=factor2_exact_count / prefix,
        double_decode_identical_pair_count=double_decode_count,
        semantic_cells_to_rgb_exact_pair_count=whole_exact_count,
        pose_within_declared_tube_pair_count=tube_count,
        additional_seed_bytes=(base_bytes - CONTEXTUAL_SEED_BASELINE_BYTES + correction_bytes),
        receiver_derived_rgb=False,
    )
    return {
        "schema": "realization_g2e_secant_prefix.v1",
        "n": prefix,
        "pair_count": prefix,
        "uint8_factor2_exact_pair_count": factor2_exact_count,
        "double_decode_identical_pair_count": double_decode_count,
        "whole_description_exact_pair_count": whole_exact_count,
        "all_declared_writes_survive_pair_count": sum(row["all_declared_writes_survive"] is True for row in hard),
        "declared_write_count": len(writes),
        "surviving_write_count": sum(row["survives"] is True for row in writes),
        "declared_write_survival_fraction": (float(np.mean([row["survives"] for row in writes])) if writes else 1.0),
        "by_class": _aggregate_survival(writes, "class_id"),
        "by_stratum": _aggregate_survival(writes, "stratum"),
        "by_margin_bucket": _aggregate_survival(writes, "margin_bucket"),
        "admitted_pair_count": sum(row["pair_solve"]["admitted"] is True for row in stage_rows),
        "solve_status_histogram": dict(sorted(Counter(row["pair_solve"]["status"] for row in stage_rows).items())),
        "mean_d_seg_realized_vs_frozen_target": float(
            np.mean([row["d_seg_realized_vs_frozen_target"] for row in hard])
        ),
        "mean_d_pose_realized_vs_frozen_target": float(
            np.mean([row["d_pose_realized_vs_frozen_target"] for row in hard])
        ),
        "mean_d_pose_declared_tube_debt": float(
            np.mean([row["d_pose_realized_outside_declared_tube"] for row in hard])
        ),
        "tube_contained_pair_count": tube_count,
        "predict_project_realization_admissibility_v1": admissibility,
        "byte_accounting": {
            "base_bytes": base_bytes,
            "correction_bytes": correction_bytes,
            "total_bytes": total_bytes,
            "target_box_bytes": CONTEXTUAL_TARGET_BOX_BYTES,
            "headroom_vs_target_box_bytes": CONTEXTUAL_TARGET_BOX_BYTES - total_bytes,
            "fits_target_box": total_bytes <= CONTEXTUAL_TARGET_BOX_BYTES,
            "rate_break_even": "25/37,545,489 score units per correction byte",
        },
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }


def run_frame0_prior_race(
    *,
    seed_path: Path,
    gt_cache_path: Path,
    upstream: Path,
    output_root: Path,
    static_chart_path: Path,
    lane_chart_path: Path,
    vjp_campaign_path: Path,
    m1_band_receipt_path: Path,
    m1_inner_jacobian_path: Path,
    rank4_prototype_receipt_path: Path,
    threads: int,
) -> dict[str, Any]:
    """Measure three frame-0 priors and refuse an unclosed RGB projection."""

    if threads < 1:
        raise RealizationAuditError("frame0-prior race requires positive CPU threads")
    if output_root.exists() and any(output_root.iterdir()):
        raise RealizationAuditError("frame0-prior race output root is preserved and nonempty; choose a new root")
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    if usage.free < 1 << 30:
        raise RealizationAuditError("frame0-prior race storage preflight requires 1 GiB free")

    seed_bytes = seed_path.read_bytes()
    seed = parse_constraint_seed(seed_bytes)
    if serialize_constraint_seed(seed) != seed_bytes:
        raise RealizationAuditError("frame0-prior seed is not canonical on parse-back")
    seed_sha256 = hashlib.sha256(seed_bytes).hexdigest()
    cache = _load_real_cache(gt_cache_path)
    net, torch, scorer_custody = _load_distortion_net(upstream, threads)
    kernel = FullResizeKernel.build()
    s_t, s_r, pitch_rad, motion_custody = load_g1_worldsheet_motion(REPO)
    geom = g1_warp.GroundHomographyGeom.eon(native_hw=SCORER_HW, pitch=pitch_rad)
    pose0 = np.asarray(cache["gt_poses"][0], dtype=np.float64).copy()
    within_xi = g1_warp.xi_from_pose_calibration(pose0, s_t=s_t, s_r=s_r, pitch=pitch_rad)
    target_cells = np.asarray(cache["lstars"][0], dtype=np.uint8).copy()
    source0 = np.asarray(cache["gt_f0"][0], dtype=np.uint8).copy()
    represented = _represented_cells(seed, 0)
    constraints = [row for row in seed["constraint_seeds"] if row["time"] == 0 and row["frame_index"] == 1]

    static_raw = static_chart_path.read_bytes()
    if hashlib.sha256(static_raw).hexdigest() != FRAME0_STATIC_CHART_SHA256:
        raise RealizationAuditError("frame0 static-chart SHA-256 custody mismatch")
    static_charts = parse_static_charts(static_raw)
    static_zlib = zlib.compress(static_raw, 9)
    parsed_static = parse_static_charts(zlib.decompress(static_zlib))
    if (
        not np.array_equal(parsed_static.road_undrivable, static_charts.road_undrivable)
        or not np.array_equal(parsed_static.hood, static_charts.hood)
        or parsed_static.adjacency != static_charts.adjacency
    ):
        raise RealizationAuditError("frame0 static-chart compressed parse-back mismatch")
    lane_pairs, lane_config, lane_custody = load_lane_chart(lane_chart_path)
    lane_raw = lane_chart_path.read_bytes()
    lane_brotli = brotli.compress(lane_raw, quality=11)
    if brotli.decompress(lane_brotli) != lane_raw:
        raise RealizationAuditError("frame0 lane-chart compressed parse-back mismatch")
    lane_mask = render_lane_mask(lane_pairs[0], lane_config, h=SCORER_HW[0], w=SCORER_HW[1])
    openpilot_cells, protected_sites = openpilot_frame0_class_prior(
        seed=seed,
        static_charts=static_charts,
        lane_mask=lane_mask,
        geom=geom,
    )

    palette_payload = serialize_frozen_scorer_palette()
    palette = parse_frozen_scorer_palette(palette_payload)
    exact_plane = _exact_source_target_plane(kernel.operator, source0)
    exact_iframe = brotli.compress(exact_plane.tobytes(order="C"), quality=11)
    if brotli.decompress(exact_iframe) != exact_plane.tobytes(order="C"):
        raise RealizationAuditError("exact frame0 I-frame Brotli parse-back mismatch")
    _atomic_bytes(output_root / "payloads" / "frozen_scorer_palette.g2pal", palette_payload)
    _atomic_bytes(output_root / "payloads" / "static_charts_n64.zlib9", static_zlib)
    _atomic_bytes(output_root / "payloads" / "lane_chart.brotli11", lane_brotli)
    _atomic_bytes(output_root / "payloads" / "exact_iframe.brotli11", exact_iframe)

    candidates = (
        {
            "candidate_id": "exact_source_iframe_control",
            "plane0": exact_plane,
            "description": target_cells,
            "payload_bytes": len(exact_iframe),
            "payload_sections": {"exact_iframe_brotli11": len(exact_iframe)},
            "source_scope": "counted source-derived exact scorer-RGB control",
        },
        {
            "candidate_id": "keyframe_class_description",
            "plane0": palette[represented],
            "description": represented,
            "payload_bytes": len(palette_payload),
            "payload_sections": {"frozen_scorer_palette": len(palette_payload)},
            "source_scope": "counted seed chart plus counted frozen-scorer palette",
        },
        {
            "candidate_id": "openpilot_per_class_geometric_solve",
            "plane0": palette[openpilot_cells],
            "description": openpilot_cells,
            "payload_bytes": len(palette_payload) + len(static_zlib) + len(lane_brotli),
            "payload_sections": {
                "frozen_scorer_palette": len(palette_payload),
                "n64_static_chart_zlib9": len(static_zlib),
                "lane_polynomial_chart_brotli11": len(lane_brotli),
            },
            "source_scope": (
                "#138/#145 G1 and lane geometry plus #139 static hood, #208 protected "
                "class sites, #234 movable tracks, and counted frozen-scorer palette"
            ),
        },
    )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        started = time.perf_counter()
        plane0 = np.asarray(candidate["plane0"], dtype=np.uint8)
        description = np.asarray(candidate["description"], dtype=np.uint8)
        plane1 = contextual_advected_rgb_plane(plane0, within_xi, geom)
        realized0 = _contextual_realize(
            plane0,
            description,
            seed_sha256=seed_sha256,
            additional_seed_bytes=int(candidate["payload_bytes"]),
            generator_id=f"frame0_prior_race_{candidate['candidate_id']}",
            kernel=kernel,
        )
        realized1 = _contextual_realize(
            plane1,
            description,
            seed_sha256=seed_sha256,
            additional_seed_bytes=0,
            generator_id=f"frame0_prior_race_{candidate['candidate_id']}_g1_refine",
            kernel=kernel,
        )
        repeated0 = _contextual_realize(
            plane0,
            description,
            seed_sha256=seed_sha256,
            additional_seed_bytes=int(candidate["payload_bytes"]),
            generator_id=f"frame0_prior_race_{candidate['candidate_id']}_repeat",
            kernel=kernel,
        )
        repeated1 = _contextual_realize(
            plane1,
            description,
            seed_sha256=seed_sha256,
            additional_seed_bytes=0,
            generator_id=f"frame0_prior_race_{candidate['candidate_id']}_g1_refine_repeat",
            kernel=kernel,
        )
        hard, actual_argmax, writes = _hard_oracle_interior(
            net,
            torch,
            realized0["frame"],
            realized1["frame"],
            target_cells,
            description,
            pose0,
            constraints,
        )
        violated = [int(row["ordinal"]) for row in writes if row["survives"] is False]
        full_prototype = contextual_banded_projection(plane1, constraints, violated, 255)
        upper_payload, upper_ordinals = encode_contextual_rgb_exceptions(
            plane1,
            full_prototype,
            constraints,
        )
        parsed_upper, parsed_ordinals = apply_contextual_rgb_exceptions(
            plane1,
            constraints,
            upper_payload,
        )
        if parsed_ordinals != upper_ordinals or not np.array_equal(parsed_upper, full_prototype):
            raise RealizationAuditError("frame0-prior syntactic exception upper bound lost parse-back")
        upper_path = output_root / "payloads" / f"{candidate['candidate_id']}.prototype_upper.g2dx"
        _atomic_bytes(upper_path, upper_payload)
        double_decode = bool(
            np.array_equal(realized0["frame"], repeated0["frame"])
            and np.array_equal(realized1["frame"], repeated1["frame"])
        )
        factor2_exact = bool(
            realized0["factor2_verification"]["certified_exact"]
            and realized1["factor2_verification"]["certified_exact"]
        )
        if not factor2_exact or not double_decode:
            raise RealizationAuditError("frame0-prior receiver lost exact deterministic custody")
        base_total = CONTEXTUAL_SEED_BASELINE_BYTES + int(candidate["payload_bytes"])
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_scope": candidate["source_scope"],
                "payload_sections": candidate["payload_sections"],
                "base_counted_bytes": base_total,
                "base_headroom_vs_target_box_bytes": CONTEXTUAL_TARGET_BOX_BYTES - base_total,
                "base_fits_target_box": base_total <= CONTEXTUAL_TARGET_BOX_BYTES,
                "syntactic_full_prototype_exception_upper_bound": {
                    "bytes": len(upper_payload),
                    "record_count": len(upper_ordinals),
                    "path": str(upper_path),
                    "sha256": hashlib.sha256(upper_payload).hexdigest(),
                    "parseback_exact": True,
                    "semantic_admission": False,
                    "scope": (
                        "#557 byte upper bound for non-minimal full-prototype writes at currently "
                        "violated declared sites; not the required rank4/secant/QP projection"
                    ),
                    "total_counted_bytes_if_carried": base_total + len(upper_payload),
                },
                "hard_oracle": hard,
                "declared_write_count": len(writes),
                "declared_write_surviving_count": sum(row["survives"] is True for row in writes),
                "declared_write_positive_margin_count": sum(float(row["target_logit_margin"]) > 0.0 for row in writes),
                "candidate_winner_vs_source_arrangement_disagreement_pixels": int(
                    np.count_nonzero(actual_argmax != target_cells)
                ),
                "uint8_factor2_exact": factor2_exact,
                "double_decode_identical": double_decode,
                "decoder_scorer_invocations": 0,
                "wall_seconds": time.perf_counter() - started,
            }
        )

    vjp_campaign = _load_json(vjp_campaign_path)
    if (
        vjp_campaign.get("status") != "COMPLETE_N600"
        or vjp_campaign.get("final_completed_count") != PAIR_COUNT
        or vjp_campaign.get("still_missing_pair_ids") != []
        or vjp_campaign.get("refused_pair_ids") != []
    ):
        raise RealizationAuditError("active VJP campaign is not terminal n600")
    inner = _load_json(m1_inner_jacobian_path)
    m1 = _load_json(m1_band_receipt_path)
    rank4 = _load_json(rank4_prototype_receipt_path)
    if inner.get("blocker") != "R1B2_RANK4_FIRST_ORDER_REALIZED_SECANT_CUSTODY_ABSENT":
        raise RealizationAuditError("inner-Jacobian/secant blocker drifted")
    if rank4.get("rank") != 4 or rank4.get("canonical_equation") != "segnet_head_rank4_linear_flipdist_v1":
        raise RealizationAuditError("rank4 prototype receipt custody mismatch")
    projection_blocker = {
        "schema": "realization_g2d_rank4_projection_compatibility.v1",
        "rank4_head": {
            "status": "BUILT_EXACT_IN_144D_PENULTIMATE_FEATURE_QUOTIENT",
            "receipt_path": str(rank4_prototype_receipt_path),
            "receipt_sha256": _sha256(rank4_prototype_receipt_path),
            "rank": rank4["rank"],
            "prototype_labels": rank4["prototype_labels"],
            "closed_form_feature_flip_distance": "abs(margin)/norm(delta_w)",
        },
        "source_vjp": {
            "status": vjp_campaign["status"],
            "completed_pair_count": vjp_campaign["final_completed_count"],
            "campaign_path": str(vjp_campaign_path),
            "campaign_sha256": _sha256(vjp_campaign_path),
            "scope": "source/native winner-rival arrangements, not these generated candidate arrangements",
        },
        "candidate_arrangement_disagreement_pixels": {
            row["candidate_id"]: row["candidate_winner_vs_source_arrangement_disagreement_pixels"] for row in rows
        },
        "m1_first_order_band": {
            "receipt_path": str(m1_band_receipt_path),
            "receipt_sha256": _sha256(m1_band_receipt_path),
            "vjp_sidecars_rehashed": m1["vjp_sidecars_rehashed"],
            "selected_pixel_count": m1["selected_pixel_count"],
            "readiness": m1["readiness"],
        },
        "missing_closure": {
            "blocker": inner["blocker"],
            "record_path": str(m1_inner_jacobian_path),
            "record_sha256": _sha256(m1_inner_jacobian_path),
            "realized_backbone_secants": inner["realized_backbone_secants"],
            "qp_receiver_closure": inner["qp_receiver_closure"],
            "rounding_ball_radius_application": "BLOCKED_WITHOUT_CANDIDATE_ARRANGEMENT_SECANT_QP",
        },
        "disposition": "BLOCKED_FAIL_CLOSED_NO_RANK4_RGB_EXCEPTION_STREAM_EMITTED",
        "verdict_scope": (
            "the exact head-space hyperplane solve is built and the source-arrangement first-order "
            "RGB pullback is complete n600; the candidate-arrangement nonlinear trunk secants and "
            "receiver-closed QP are absent. This is not a negative on computed rank4 projection."
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }
    _atomic_json(output_root / "projection_blocker.json", projection_blocker)

    ranked = sorted(
        rows,
        key=lambda row: (
            row["hard_oracle"]["d_seg_realized_vs_frozen_target"],
            -row["declared_write_surviving_count"],
            row["base_counted_bytes"],
        ),
    )
    receipt = {
        "schema": FRAME0_PRIOR_RACE_SCHEMA,
        "lane_id": CONTEXTUAL_LANE_ID,
        "task_id": "578",
        "measurement": f"MEASURED {AXIS}",
        "candidate_rows": rows,
        "base_only_rank": [row["candidate_id"] for row in ranked],
        "base_only_winner": ranked[0]["candidate_id"],
        "openpilot_composition": {
            "policy": "xi_advected_prior_per_class_charts.v2 pair0",
            "lane_custody": lane_custody,
            "protected_class_sites": protected_sites,
            "static_chart_raw_sha256": FRAME0_STATIC_CHART_SHA256,
            "static_chart_raw_bytes": len(static_raw),
            "static_chart_zlib9_bytes_counted_here": len(static_zlib),
            "static_chart_zlib_parseback_exact": True,
            "lane_chart_brotli11_bytes_counted_here": len(lane_brotli),
            "lane_chart_brotli_parseback_exact": True,
        },
        "projection_blocker": {
            "path": str(output_root / "projection_blocker.json"),
            "sha256": _sha256(output_root / "projection_blocker.json"),
            "disposition": projection_blocker["disposition"],
        },
        "scorer_custody": scorer_custody,
        "motion_custody": motion_custody,
        "source_custody": {
            "seed": {"path": str(seed_path), "bytes": len(seed_bytes), "sha256": seed_sha256},
            "gt_cache": {"path": str(gt_cache_path), "sha256": GT_CACHE_SHA256},
            "tool": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        },
        "storage": {
            "root": str(output_root),
            "free_bytes_at_preflight": usage.free,
            "automatic_disk_hygiene": "small immutable compressed packets and JSON only; no camera/logit tensor persistence",
        },
        "verdict": "MEASURED_FRAME0_PRIOR_RACE_PROJECTION_BLOCKED",
        "verdict_scope": (
            "pair0 frame0 prior followed by one G1 within-pair refinement only; no exception stream "
            "is admitted until candidate-arrangement rank4 first-order plus realized secant QP closes"
        ),
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


def run_realized_secant_custody(
    *,
    seed_path: Path,
    gt_cache_path: Path,
    upstream: Path,
    output_root: Path,
    rank4_prototype_receipt_path: Path,
    static_chart_path: Path,
    lane_chart_path: Path,
    chunk_size: int,
    threads: int,
    stop_after_prefix: int,
) -> dict[str, Any]:
    """Run/resume receiver-closed candidate-state secants and per-pair QPs."""

    if chunk_size < 1 or threads < 1 or stop_after_prefix not in PREFIXES:
        raise RealizationAuditError("invalid G2e chunk/thread/prefix setting")
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    if usage.free < 1 << 30:
        raise RealizationAuditError("G2e storage preflight requires at least 1 GiB free")
    seed_bytes = seed_path.read_bytes()
    seed = parse_constraint_seed(seed_bytes)
    if serialize_constraint_seed(seed) != seed_bytes:
        raise RealizationAuditError("G2e seed is not canonical on parse-back")
    seed_sha256 = hashlib.sha256(seed_bytes).hexdigest()
    rank4 = _load_json(rank4_prototype_receipt_path)
    if (
        rank4.get("schema") != "rank4_valid_cell_prototypes_v1"
        or rank4.get("rank") != SECANT_CHART_RANK
        or rank4.get("canonical_equation") != "segnet_head_rank4_linear_flipdist_v1"
    ):
        raise RealizationAuditError("G2e rank-4 prototype receipt custody mismatch")
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
    static_raw = static_chart_path.read_bytes()
    if hashlib.sha256(static_raw).hexdigest() != FRAME0_STATIC_CHART_SHA256:
        raise RealizationAuditError("G2e frame0 static-chart SHA-256 custody mismatch")
    static_charts = parse_static_charts(static_raw)
    static_zlib = zlib.compress(static_raw, 9)
    parsed_static = parse_static_charts(zlib.decompress(static_zlib))
    if (
        not np.array_equal(parsed_static.road_undrivable, static_charts.road_undrivable)
        or not np.array_equal(parsed_static.hood, static_charts.hood)
        or parsed_static.adjacency != static_charts.adjacency
    ):
        raise RealizationAuditError("G2e static-chart compressed parse-back mismatch")
    lane_pairs, lane_config, lane_custody = load_lane_chart(lane_chart_path)
    lane_raw = lane_chart_path.read_bytes()
    lane_brotli = brotli.compress(lane_raw, quality=11)
    if brotli.decompress(lane_brotli) != lane_raw:
        raise RealizationAuditError("G2e lane-chart compressed parse-back mismatch")
    lane_mask = render_lane_mask(lane_pairs[0], lane_config, h=SCORER_HW[0], w=SCORER_HW[1])
    openpilot_cells, protected_sites = openpilot_frame0_class_prior(
        seed=seed,
        static_charts=static_charts,
        lane_mask=lane_mask,
        geom=geom,
    )
    palette_payload = serialize_frozen_scorer_palette()
    palette = parse_frozen_scorer_palette(palette_payload)
    bootstrap = palette[openpilot_cells]
    bootstrap_counted_bytes = len(palette_payload) + len(static_zlib) + len(lane_brotli)
    root = output_root / "realized_secant_custody_openpilot"
    _atomic_bytes(root / "base_packets" / "frozen_scorer_palette.g2pal", palette_payload)
    _atomic_bytes(root / "base_packets" / "static_charts_n64.zlib9", static_zlib)
    _atomic_bytes(root / "base_packets" / "lane_chart.brotli11", lane_brotli)

    implementation_paths = (
        REPO / "tools/measure_realization_g2_lattice.py",
        REPO / "src/tac/optimization/realized_secant_custody.py",
        REPO / "src/tac/optimization/predict_project_receiver.py",
        REPO / "src/tac/optimization/uint8_lattice_feasibility.py",
        REPO / "src/tac/boundary_math/warp_real_luma_frame0.py",
        REPO / "src/tac/optimization/predictor_upgrade_xi_chart.py",
        REPO / "src/tac/optimization/predictor_r2_missdelta.py",
        REPO / "src/tac/canonical_equations/predict_project_realization_admissibility_20260721.py",
    )
    config = {
        "schema": SECANT_CONFIG_SCHEMA,
        "seed": 1234,
        "seed_sha256": seed_sha256,
        "gt_cache_sha256": GT_CACHE_SHA256,
        "scorer_custody": scorer_custody,
        "motion_custody": {**motion_custody, **cross_custody},
        "rank4_prototype_receipt": {
            "path": str(rank4_prototype_receipt_path),
            "sha256": _sha256(rank4_prototype_receipt_path),
            "rank": rank4["rank"],
            "prototype_sha256": rank4["prototype_sha256"],
            "quotient_basis_sha256": rank4["quotient_basis_sha256"],
        },
        "implementation_sources": {str(path.relative_to(REPO)): _sha256(path) for path in implementation_paths},
        "base": ("G2d openpilot per-class geometric frame0 prior plus canonical within/cross G1 advected RGB"),
        "openpilot_base_custody": {
            "palette_bytes": len(palette_payload),
            "palette_sha256": hashlib.sha256(palette_payload).hexdigest(),
            "static_chart_raw_sha256": FRAME0_STATIC_CHART_SHA256,
            "static_chart_zlib9_bytes": len(static_zlib),
            "static_chart_zlib9_sha256": hashlib.sha256(static_zlib).hexdigest(),
            "lane_chart_raw_sha256": hashlib.sha256(lane_raw).hexdigest(),
            "lane_chart_brotli11_bytes": len(lane_brotli),
            "lane_chart_brotli11_sha256": hashlib.sha256(lane_brotli).hexdigest(),
            "lane_custody": lane_custody,
            "protected_class_sites": protected_sites,
            "total_counted_bytes": bootstrap_counted_bytes,
        },
        "candidate_response": (
            "fresh candidate-state native CPU-Torch SegNet, exact input to segmentation_head[0], "
            "local declared-write RGB pullback"
        ),
        "chart_rank": SECANT_CHART_RANK,
        "signed_amplitudes": list(SECANT_SIGNED_AMPLITUDES),
        "relative_secant_residual_tolerance": SECANT_RELATIVE_RESIDUAL_TOLERANCE,
        "required_positive_margin": SECANT_REQUIRED_MARGIN,
        "qp": "deterministic complete active-set enumeration in rank<=4 with uint8 box inequalities",
        "correction_codec": "#557 adaptive range coder; zero changed records carry zero bytes",
        "pair_count": PAIR_COUNT,
        "axis": AXIS,
    }
    config_sha256 = secant_canonical_sha256(config)
    stage_dir = root / "stages"
    sidecar_dir = root / "corrections"
    tested_dir = root / "tested_corrections"
    rows = _load_secant_stages(stage_dir, sidecar_dir, config_sha256)
    resumed_pairs = len(rows)
    if rows and sorted(rows) != list(range(max(rows) + 1)):
        raise RealizationAuditError("G2e resume stages are not a contiguous prefix")
    constraints_by_pair: dict[int, list[Mapping[str, Any]]] = {pair: [] for pair in range(PAIR_COUNT)}
    for constraint in seed["constraint_seeds"]:
        if constraint["frame_index"] == 1:
            constraints_by_pair[int(constraint["time"])].append(constraint)
    represented_by_pair = {pair: _represented_cells(seed, pair) for pair in range(stop_after_prefix)}
    previous_plane: np.ndarray | None = None
    rate_lambda = 25.0 / 37_545_489.0

    for chunk_begin in range(0, stop_after_prefix, chunk_size):
        chunk_end = min(stop_after_prefix, chunk_begin + chunk_size)
        for pair_index in range(chunk_begin, chunk_end):
            represented = represented_by_pair[pair_index]
            constraints = constraints_by_pair[pair_index]
            if not constraints:
                raise RealizationAuditError(f"G2e pair {pair_index} has no declared writes")
            plane0 = (
                bootstrap.copy()
                if pair_index == 0
                else contextual_advected_rgb_plane(previous_plane, cross_xi[pair_index], geom)
            )
            base1 = contextual_advected_rgb_plane(plane0, within_xi[pair_index], geom)
            sidecar_path = sidecar_dir / f"pair_{pair_index:04d}.g2dx"
            if pair_index in rows:
                payload = sidecar_path.read_bytes()
                if payload:
                    previous_plane, _ = apply_contextual_rgb_exceptions(base1, constraints, payload)
                else:
                    previous_plane = base1
                continue

            started = time.perf_counter()
            target_cells = np.asarray(cache["lstars"][pair_index], dtype=np.uint8).copy()
            target_pose = np.asarray(cache["gt_poses"][pair_index], dtype=np.float64).copy()
            realized0 = _contextual_realize(
                plane0,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=bootstrap_counted_bytes if pair_index == 0 else 0,
                generator_id=(
                    "g2e_openpilot_per_class_geometric_frame0_prior"
                    if pair_index == 0
                    else "g2e_cross_pair_g1_advected_rgb"
                ),
                kernel=kernel,
            )
            realized_base1 = _contextual_realize(
                base1,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=0,
                generator_id="g2e_within_pair_g1_advected_rgb_base",
                kernel=kernel,
            )
            baseline_state = _candidate_seg_state(
                net,
                torch,
                realized0["frame"],
                realized_base1["frame"],
                base1,
                constraints,
                with_jacobian=True,
            )
            baseline_hard, _, baseline_writes = _hard_oracle_interior(
                net,
                torch,
                realized0["frame"],
                realized_base1["frame"],
                target_cells,
                represented,
                target_pose,
                constraints,
            )
            local_jacobian = np.asarray(baseline_state["local_jacobian"], dtype=np.float64)
            chart_directions = _rank4_chart_directions(local_jacobian)
            secant_rows: list[SecantObservation] = []
            applied_columns: list[np.ndarray] = []
            realized_columns: list[np.ndarray] = []
            for column, amplitude in enumerate(SECANT_SIGNED_AMPLITUDES):
                probe_plane, actual_delta, saturation = _apply_local_chart_delta(
                    base1, constraints, amplitude * chart_directions[:, column]
                )
                probe_realized = _contextual_realize(
                    probe_plane,
                    represented,
                    seed_sha256=seed_sha256,
                    additional_seed_bytes=0,
                    generator_id=f"g2e_signed_secant_column_{column}",
                    kernel=kernel,
                )
                probe_state = _candidate_seg_state(
                    net,
                    torch,
                    realized0["frame"],
                    probe_realized["frame"],
                    probe_plane,
                    constraints,
                    fixed_rivals=baseline_state["rivals"],
                    with_jacobian=False,
                )
                predicted = local_jacobian @ actual_delta
                realized_delta = probe_state["margins"] - baseline_state["margins"]
                write_rows = []
                for ordinal, constraint in enumerate(constraints):
                    predicted_value = float(predicted[ordinal])
                    realized_value = float(realized_delta[ordinal])
                    write_rows.append(
                        WriteSecantObservation(
                            ordinal=ordinal,
                            target_class=int(constraint["cell_id"]),
                            current_class=int(baseline_state["current_classes"][ordinal]),
                            pre_margin=float(baseline_state["margins"][ordinal]),
                            margin_bucket=_margin_bucket(float(baseline_state["margins"][ordinal])),
                            expected_sign=1 if predicted_value >= 0.0 else -1,
                            feature_displacement=tuple(
                                float(after - before)
                                for after, before in zip(
                                    probe_state["feature_patches"][ordinal],
                                    baseline_state["feature_patches"][ordinal],
                                    strict=True,
                                )
                            ),
                            predicted_margin_delta=predicted_value,
                            realized_margin_delta=realized_value,
                            secant_ratio=realized_value / amplitude,
                        )
                    )
                secant_rows.append(
                    SecantObservation(
                        pair_index=pair_index,
                        column_index=column,
                        signed_amplitude=amplitude,
                        applied_rgb_l2=float(np.linalg.norm(actual_delta)),
                        applied_rgb_linf=float(np.max(np.abs(actual_delta), initial=0.0)),
                        uint8_saturation_count=saturation,
                        writes=tuple(write_rows),
                    )
                )
                applied_columns.append(actual_delta / amplitude)
                realized_columns.append(realized_delta / amplitude)

            trust_regions = build_trust_regions(
                secant_rows,
                relative_residual_tolerance=SECANT_RELATIVE_RESIDUAL_TOLERANCE,
            )
            direction_model = np.column_stack(applied_columns)
            secant_model = np.column_stack(realized_columns)
            solve = None
            tested_plane = base1
            tested_payload = b""
            tested_ordinals: list[int] = []
            coefficient_packet = b""
            status = SecantPairSolveStatus.TRUST_REGION_REFUSED
            if all(region.usable for region in trust_regions):
                solve = solve_minimal_norm_inequalities(
                    secant_model,
                    SECANT_REQUIRED_MARGIN - baseline_state["margins"],
                    direction_model,
                    _local_rgb_values(base1, constraints),
                )
                if solve.status is SecantQPStatus.SOLVED:
                    coefficient_packet = encode_coefficient_packet(solve.coefficients)
                    decoded_once = decode_coefficient_packet(coefficient_packet)
                    decoded_twice = decode_coefficient_packet(coefficient_packet)
                    if decoded_once != decoded_twice or encode_coefficient_packet(decoded_once) != coefficient_packet:
                        raise RealizationAuditError("G2e coefficient packet double-decode drift")
                    tested_plane, _, _ = _apply_local_chart_delta(
                        base1,
                        constraints,
                        direction_model @ np.asarray(decoded_once, dtype=np.float64),
                    )
                    tested_payload, tested_plane, tested_ordinals = _exception_parseback(
                        base1, constraints, tested_plane
                    )
                else:
                    status = SecantPairSolveStatus.QP_INFEASIBLE

            tested_path = tested_dir / f"pair_{pair_index:04d}.g2dx"
            _atomic_bytes(tested_path, tested_payload)
            tested_realized = _contextual_realize(
                tested_plane,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=len(tested_payload),
                generator_id="g2e_tested_rank4_secant_qp_correction",
                kernel=kernel,
            )
            tested_second = _contextual_realize(
                tested_plane,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=len(tested_payload),
                generator_id="g2e_tested_rank4_secant_qp_correction_repeat",
                kernel=kernel,
            )
            tested_hard, _, tested_writes = _hard_oracle_interior(
                net,
                torch,
                realized0["frame"],
                tested_realized["frame"],
                target_cells,
                represented,
                target_pose,
                constraints,
            )
            tested_state = _candidate_seg_state(
                net,
                torch,
                realized0["frame"],
                tested_realized["frame"],
                tested_plane,
                constraints,
                fixed_rivals=baseline_state["rivals"],
                with_jacobian=False,
            )
            double_decode = bool(
                np.array_equal(tested_realized["frame"], tested_second["frame"])
                and tested_realized["camera_uint8_sha256"] == tested_second["camera_uint8_sha256"]
            )
            positive_fixed_margins = bool(np.all(tested_state["margins"] > SECANT_REQUIRED_MARGIN))
            hard_positive = bool(
                tested_hard["all_declared_writes_survive"]
                and all(float(row["target_logit_margin"]) > 0.0 for row in tested_writes)
            )
            semantic_score_delta = 100.0 * (
                float(baseline_hard["d_seg_realized_vs_frozen_target"])
                - float(tested_hard["d_seg_realized_vs_frozen_target"])
            )
            marginal_score_units_per_byte = (
                semantic_score_delta / len(tested_payload)
                if tested_payload
                else (math.inf if semantic_score_delta > 0.0 else 0.0)
            )
            rate_admissible = bool(not tested_payload or marginal_score_units_per_byte > rate_lambda)
            kkt_clean = bool(
                solve is not None
                and solve.status is SecantQPStatus.SOLVED
                and solve.max_primal_violation <= 1e-8
                and solve.stationarity_residual <= 1e-8
                and solve.min_active_multiplier is not None
                and solve.min_active_multiplier >= -1e-9
            )
            admitted = bool(
                kkt_clean and positive_fixed_margins and hard_positive and double_decode and rate_admissible
            )
            if solve is not None and solve.status is SecantQPStatus.SOLVED:
                if not positive_fixed_margins or not hard_positive:
                    status = SecantPairSolveStatus.NEGATIVE_REALIZED_HARD_ORACLE_REFUSED
                elif not rate_admissible:
                    status = SecantPairSolveStatus.RATE_BREAK_EVEN_REFUSED
                elif not kkt_clean:
                    status = SecantPairSolveStatus.KKT_RESIDUAL_REFUSED
                elif not double_decode:
                    status = SecantPairSolveStatus.DOUBLE_DECODE_REFUSED
                else:
                    status = SecantPairSolveStatus.ADMITTED_RECEIVER_CLOSED
            admitted_payload = tested_payload if admitted else b""
            admitted_plane = tested_plane if admitted else base1
            _atomic_bytes(sidecar_path, admitted_payload)
            if admitted:
                final_realized = tested_realized
                hard, writes = tested_hard, tested_writes
            else:
                final_realized = realized_base1
                hard, writes = baseline_hard, baseline_writes
            previous_plane = admitted_plane
            stage = {
                "schema": SECANT_STAGE_SCHEMA,
                "config_sha256": config_sha256,
                "pair_index": pair_index,
                "candidate_state": {
                    "segnet_input_sha256": baseline_state["segnet_input_sha256"],
                    "logits_sha256": baseline_state["logits_sha256"],
                    "receiver_input_maxabs": baseline_state["receiver_input_maxabs"],
                    "head_input": "exact 16-channel input to segmentation_head[0]",
                    "head_patch_dimension": 144,
                    "rivals": list(baseline_state["rivals"]),
                    "fresh_candidate_response": True,
                    "source_arrangement_vjp_used_for_solve": False,
                },
                "secant_observations": [row.as_dict() for row in secant_rows],
                "trust_regions": [row.as_dict() for row in trust_regions],
                "pair_solve": {
                    "pair_index": pair_index,
                    "status": status.value,
                    "admitted": admitted,
                    "qp": solve.as_dict() if solve is not None else None,
                    "coefficient_packet": {
                        "bytes": len(coefficient_packet),
                        "sha256": hashlib.sha256(coefficient_packet).hexdigest(),
                        "double_decode_identical": bool(coefficient_packet),
                    },
                    "tested_fixed_rival_margins": tested_state["margins"].tolist(),
                    "tested_positive_fixed_rival_margins": positive_fixed_margins,
                    "tested_hard_positive_declared_writes": hard_positive,
                    "tested_uint8_saturation_count": int(sum(row.uint8_saturation_count for row in secant_rows)),
                    "tested_exception_ordinals": tested_ordinals,
                    "tested_packet_bytes": len(tested_payload),
                    "tested_packet_sha256": hashlib.sha256(tested_payload).hexdigest(),
                    "tested_packet_path": str(tested_path),
                    "double_decode_identical": double_decode,
                    "semantic_score_units_delta": semantic_score_delta,
                    "marginal_score_units_per_correction_byte": marginal_score_units_per_byte,
                    "rate_break_even": rate_lambda,
                    "rate_admissible": rate_admissible,
                    "baseline_hard_oracle": baseline_hard,
                    "tested_hard_oracle": tested_hard,
                    "pose_delta_from_semantic_correction": float(
                        tested_hard["d_pose_realized_vs_frozen_target"]
                        - baseline_hard["d_pose_realized_vs_frozen_target"]
                    ),
                },
                "correction_packet": {
                    "codec": "#557 adaptive range coder" if admitted_payload else "none",
                    "path": str(sidecar_path),
                    "bytes": len(admitted_payload),
                    "sha256": hashlib.sha256(admitted_payload).hexdigest(),
                    "video_derived_payload_carried": bool(admitted_payload),
                    "parseback_exact": True,
                    "reencode_byte_identical": True,
                },
                "hard_oracle": hard,
                "declared_write_survival": writes,
                "camera_frame0_sha256": realized0["camera_uint8_sha256"],
                "camera_frame1_sha256": final_realized["camera_uint8_sha256"],
                "uint8_factor2_exact": bool(
                    realized0["factor2_verification"]["certified_exact"]
                    and final_realized["factor2_verification"]["certified_exact"]
                ),
                "double_decode_identical": double_decode,
                "wall_seconds": time.perf_counter() - started,
                "authority": f"MEASURED {AXIS}",
                "score_claim": False,
                "promotion_eligible": False,
            }
            _atomic_json(stage_dir / f"pair_{pair_index:04d}.json", stage)
            rows[pair_index] = stage
            del target_cells, target_pose
        _atomic_json(
            root / "checkpoints" / f"chunk_{chunk_begin:04d}_{chunk_end:04d}.json",
            {
                "schema": "realization_g2e_secant_chunk_checkpoint.v1",
                "config_sha256": config_sha256,
                "completed_through_exclusive": chunk_end,
                "completed_pairs": len(rows),
                "all_pair_stages_preserved": True,
                "resumed_pairs_at_invocation_start": resumed_pairs,
            },
        )

    ordered = [rows[index] for index in range(stop_after_prefix)]
    base_bytes = CONTEXTUAL_SEED_BASELINE_BYTES + bootstrap_counted_bytes
    prefixes = []
    for prefix in PREFIXES:
        if prefix > stop_after_prefix:
            continue
        summary = summarize_secant_prefix(prefix, ordered[:prefix], base_bytes=base_bytes)
        checkpoint = root / "checkpoints" / f"prefix_n{prefix}.json"
        _atomic_json(checkpoint, summary)
        summary["checkpoint_path"] = str(checkpoint)
        summary["checkpoint_sha256"] = _sha256(checkpoint)
        prefixes.append(summary)
    final = prefixes[-1]
    admission = final["predict_project_realization_admissibility_v1"]
    secant_observations = [observation for row in ordered for observation in row["secant_observations"]]
    pair_trust_regions = list(
        build_pair_trust_region_custody(
            [SecantObservation.from_dict(row) for row in secant_observations],
            pair_count=stop_after_prefix,
            relative_residual_tolerance=config["relative_secant_residual_tolerance"],
        )
    )
    receipt: dict[str, Any] = {
        "schema": SECANT_RECEIPT_SCHEMA,
        "lane_id": SECANT_LANE_ID,
        "task_id": "578",
        "config": config,
        "config_sha256": config_sha256,
        "completed_prefix": stop_after_prefix,
        "column_indices": list(range(SECANT_CHART_RANK)),
        "secant_observations": secant_observations,
        "pair_trust_regions": pair_trust_regions,
        "pair_solves": [row["pair_solve"] for row in ordered],
        "D1_realized_secant_ladder": prefixes,
        "D2_receiver_closed_solve": {
            "admitted_pair_count": final["admitted_pair_count"],
            "solve_status_histogram": final["solve_status_histogram"],
            "hard_oracle_is_admission_authority": True,
        },
        "D3_semantic_rate_ladder": final,
        "D4_pose_scope": {
            "mean_realized_d_pose": final["mean_d_pose_realized_vs_frozen_target"],
            "mean_declared_tube_debt": final["mean_d_pose_declared_tube_debt"],
            "tube_contained_pair_count": final["tube_contained_pair_count"],
            "status": f"MEASURED_CORRECTED_FRAME_PAIRS {AXIS}",
            "nearest_target_cross_pair_proxy_exact": False,
            "blocker": (
                "the contextual cross-pair base uses the declared nearest-target G1 proxy; "
                "no exact pose-factorized conclusion transfers from this semantic arm"
            ),
            "pose_factorized_child_open": True,
        },
        "admissibility": admission,
        "verdict": (
            "MEASURED_G2E_SECANT_PREFIX_N600"
            if stop_after_prefix == PAIR_COUNT
            else f"MEASURED_G2E_SECANT_PREFIX_N{stop_after_prefix}_FAMILY_OPEN"
        ),
        "verdict_scope": (
            f"exact contiguous n{stop_after_prefix} prefix only; candidate-state local declared-write "
            "rank-4 chart with signed finite secants and receiver-closed per-pair correction; "
            "no n600 claim unless completed_prefix is 600"
        ),
        "storage": {
            "root": str(root),
            "free_bytes_at_preflight": usage.free,
            "automatic_disk_hygiene": (
                "ZIP_STORED cache memmaps, current pair/column scorer tensors only, atomic immutable "
                "JSON and #557 sidecars, no persisted camera/logit/feature tensors"
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
    receipt["receipt_sha256"] = secant_canonical_sha256(receipt)
    validate_secant_receipt(receipt, expected_pair_count=stop_after_prefix)
    _atomic_json(root / "receipt.json", receipt)
    _atomic_json(output_root / "receipt.json", receipt)
    if _load_json(root / "receipt.json")["receipt_sha256"] != receipt["receipt_sha256"]:
        raise RealizationAuditError("G2e receipt JSON parse-back drift")
    return receipt


def run_bidirectional_amplitude_ladder(
    *,
    seed_path: Path,
    gt_cache_path: Path,
    upstream: Path,
    output_root: Path,
    g2e_prior_receipt_path: Path,
    rank4_prototype_receipt_path: Path,
    static_chart_path: Path,
    lane_chart_path: Path,
    chunk_size: int,
    threads: int,
    stop_after_prefix: int,
) -> dict[str, Any]:
    """Run/resume G2f paired amplitude curves and trust-gated receiver QPs."""

    if chunk_size < 1 or threads < 1 or stop_after_prefix not in PREFIXES:
        raise RealizationAuditError("invalid G2f chunk/thread/prefix setting")
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    if usage.free < 1 << 30:
        raise RealizationAuditError("G2f storage preflight requires at least 1 GiB free")
    if _sha256(g2e_prior_receipt_path) != G2E_PRIOR_RECEIPT_SHA256:
        raise RealizationAuditError("G2f G2e rung-0 prior receipt SHA-256 mismatch")
    g2e_prior = _load_json(g2e_prior_receipt_path)
    if validate_secant_receipt(g2e_prior, expected_pair_count=16) != g2e_prior["receipt_sha256"]:
        raise RealizationAuditError("G2f G2e rung-0 prior strict validation failed")
    prior_observations = [SecantObservation.from_dict(row) for row in g2e_prior["secant_observations"]]
    prior_by_pair_direction = {(row.pair_index, row.column_index): row for row in prior_observations}
    if len(prior_by_pair_direction) != 64:
        raise RealizationAuditError("G2f G2e rung-0 prior identity coverage mismatch")
    prior_stage_root = Path(g2e_prior["storage"]["root"]) / "stages"
    prior_stages = {pair: _load_json(prior_stage_root / f"pair_{pair:04d}.json") for pair in range(16)}

    seed_bytes = seed_path.read_bytes()
    seed = parse_constraint_seed(seed_bytes)
    if serialize_constraint_seed(seed) != seed_bytes:
        raise RealizationAuditError("G2f seed is not canonical on parse-back")
    seed_sha256 = hashlib.sha256(seed_bytes).hexdigest()
    rank4 = _load_json(rank4_prototype_receipt_path)
    if (
        rank4.get("schema") != "rank4_valid_cell_prototypes_v1"
        or rank4.get("rank") != SECANT_CHART_RANK
        or rank4.get("canonical_equation") != "segnet_head_rank4_linear_flipdist_v1"
    ):
        raise RealizationAuditError("G2f rank-4 prototype receipt custody mismatch")
    cache = _load_real_cache(gt_cache_path)
    net, torch, scorer_custody = _load_distortion_net(upstream, threads)
    kernel = FullResizeKernel.build()
    amplitudes, amplitude_derivation = derive_bidirectional_amplitude_ladder(kernel, g2e_prior)
    s_t, s_r, pitch_rad, motion_custody = load_g1_worldsheet_motion(REPO)
    poses = np.asarray(cache["gt_poses"], dtype=np.float64)
    cross_xi, cross_custody = relative_adjacent_xi(poses, s_t=s_t, s_r=s_r, pitch_rad=pitch_rad)
    within_xi = np.stack(
        [g1_warp.xi_from_pose_calibration(pose, s_t=s_t, s_r=s_r, pitch=pitch_rad) for pose in poses],
        axis=0,
    )
    geom = g1_warp.GroundHomographyGeom.eon(native_hw=SCORER_HW, pitch=pitch_rad)
    static_raw = static_chart_path.read_bytes()
    if hashlib.sha256(static_raw).hexdigest() != FRAME0_STATIC_CHART_SHA256:
        raise RealizationAuditError("G2f frame0 static-chart SHA-256 custody mismatch")
    static_charts = parse_static_charts(static_raw)
    static_zlib = zlib.compress(static_raw, 9)
    parsed_static = parse_static_charts(zlib.decompress(static_zlib))
    if (
        not np.array_equal(parsed_static.road_undrivable, static_charts.road_undrivable)
        or not np.array_equal(parsed_static.hood, static_charts.hood)
        or parsed_static.adjacency != static_charts.adjacency
    ):
        raise RealizationAuditError("G2f static-chart compressed parse-back mismatch")
    lane_pairs, lane_config, lane_custody = load_lane_chart(lane_chart_path)
    lane_raw = lane_chart_path.read_bytes()
    lane_brotli = brotli.compress(lane_raw, quality=11)
    if brotli.decompress(lane_brotli) != lane_raw:
        raise RealizationAuditError("G2f lane-chart compressed parse-back mismatch")
    lane_mask = render_lane_mask(lane_pairs[0], lane_config, h=SCORER_HW[0], w=SCORER_HW[1])
    openpilot_cells, protected_sites = openpilot_frame0_class_prior(
        seed=seed,
        static_charts=static_charts,
        lane_mask=lane_mask,
        geom=geom,
    )
    palette_payload = serialize_frozen_scorer_palette()
    palette = parse_frozen_scorer_palette(palette_payload)
    bootstrap = palette[openpilot_cells]
    bootstrap_counted_bytes = len(palette_payload) + len(static_zlib) + len(lane_brotli)
    root = output_root / "bidirectional_amplitude_ladder_openpilot"
    _atomic_bytes(root / "base_packets" / "frozen_scorer_palette.g2pal", palette_payload)
    _atomic_bytes(root / "base_packets" / "static_charts_n64.zlib9", static_zlib)
    _atomic_bytes(root / "base_packets" / "lane_chart.brotli11", lane_brotli)

    implementation_paths = (
        REPO / "tools/measure_realization_g2_lattice.py",
        REPO / "src/tac/optimization/realized_secant_custody.py",
        REPO / "src/tac/optimization/predict_project_receiver.py",
        REPO / "src/tac/optimization/uint8_lattice_feasibility.py",
        REPO / "src/tac/optimization/resize_full_kernel.py",
        REPO / "src/tac/boundary_math/warp_real_luma_frame0.py",
        REPO / "src/tac/optimization/predictor_upgrade_xi_chart.py",
        REPO / "src/tac/optimization/predictor_r2_missdelta.py",
        REPO / "src/tac/canonical_equations/predict_project_realization_admissibility_20260721.py",
    )
    config = {
        "schema": AMPLITUDE_CONFIG_SCHEMA,
        "seed": 1234,
        "seed_sha256": seed_sha256,
        "gt_cache_sha256": GT_CACHE_SHA256,
        "scorer_custody": scorer_custody,
        "motion_custody": {**motion_custody, **cross_custody},
        "rank4_prototype_receipt": {
            "path": str(rank4_prototype_receipt_path),
            "sha256": _sha256(rank4_prototype_receipt_path),
            "rank": rank4["rank"],
            "prototype_sha256": rank4["prototype_sha256"],
            "quotient_basis_sha256": rank4["quotient_basis_sha256"],
        },
        "g2e_rung0_prior": {
            "path": str(g2e_prior_receipt_path),
            "receipt_file_sha256": G2E_PRIOR_RECEIPT_SHA256,
            "canonical_receipt_sha256": g2e_prior["receipt_sha256"],
            "secant_observation_count": len(prior_observations),
            "remeasured": False,
        },
        "implementation_sources": {str(path.relative_to(REPO)): _sha256(path) for path in implementation_paths},
        "base": "G2d openpilot per-class geometric frame0 prior plus canonical within/cross G1 advected RGB",
        "openpilot_base_custody": {
            "palette_bytes": len(palette_payload),
            "palette_sha256": hashlib.sha256(palette_payload).hexdigest(),
            "static_chart_raw_sha256": FRAME0_STATIC_CHART_SHA256,
            "static_chart_zlib9_bytes": len(static_zlib),
            "static_chart_zlib9_sha256": hashlib.sha256(static_zlib).hexdigest(),
            "lane_chart_raw_sha256": hashlib.sha256(lane_raw).hexdigest(),
            "lane_chart_brotli11_bytes": len(lane_brotli),
            "lane_chart_brotli11_sha256": hashlib.sha256(lane_brotli).hexdigest(),
            "lane_custody": lane_custody,
            "protected_class_sites": protected_sites,
            "total_counted_bytes": bootstrap_counted_bytes,
        },
        "amplitude_ladder": list(amplitudes),
        "amplitude_ladder_derivation": amplitude_derivation,
        "relative_secant_residual_tolerance": SECANT_RELATIVE_RESIDUAL_TOLERANCE,
        "required_positive_margin": SECANT_REQUIRED_MARGIN,
        "trust_rule": "both signed branches plus central odd secant sign-consistent; odd relative residual <= tolerance",
        "qp": "deterministic complete active-set enumeration in rank<=4 with uint8 box inequalities",
        "correction_codec": "#557 adaptive range coder; zero changed records carry zero bytes",
        "pair_count": PAIR_COUNT,
        "axis": AXIS,
    }
    config_sha256 = secant_canonical_sha256(config)
    stage_dir = root / "stages"
    sidecar_dir = root / "corrections"
    tested_dir = root / "tested_corrections"
    rows = _load_amplitude_stages(stage_dir, sidecar_dir, config_sha256)
    resumed_pairs = len(rows)
    if rows and sorted(rows) != list(range(max(rows) + 1)):
        raise RealizationAuditError("G2f resume stages are not a contiguous prefix")
    constraints_by_pair: dict[int, list[Mapping[str, Any]]] = {pair: [] for pair in range(PAIR_COUNT)}
    for constraint in seed["constraint_seeds"]:
        if constraint["frame_index"] == 1:
            constraints_by_pair[int(constraint["time"])].append(constraint)
    represented_by_pair = {pair: _represented_cells(seed, pair) for pair in range(stop_after_prefix)}
    previous_plane: np.ndarray | None = None
    rate_lambda = 25.0 / 37_545_489.0

    for chunk_begin in range(0, stop_after_prefix, chunk_size):
        chunk_end = min(stop_after_prefix, chunk_begin + chunk_size)
        for pair_index in range(chunk_begin, chunk_end):
            represented = represented_by_pair[pair_index]
            constraints = constraints_by_pair[pair_index]
            if not constraints:
                raise RealizationAuditError(f"G2f pair {pair_index} has no declared writes")
            plane0 = (
                bootstrap.copy()
                if pair_index == 0
                else contextual_advected_rgb_plane(previous_plane, cross_xi[pair_index], geom)
            )
            base1 = contextual_advected_rgb_plane(plane0, within_xi[pair_index], geom)
            sidecar_path = sidecar_dir / f"pair_{pair_index:04d}.g2dx"
            if pair_index in rows:
                payload = sidecar_path.read_bytes()
                if payload:
                    previous_plane, _ = apply_contextual_rgb_exceptions(base1, constraints, payload)
                else:
                    previous_plane = base1
                continue

            started = time.perf_counter()
            target_cells = np.asarray(cache["lstars"][pair_index], dtype=np.uint8).copy()
            target_pose = np.asarray(cache["gt_poses"][pair_index], dtype=np.float64).copy()
            realized0 = _contextual_realize(
                plane0,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=bootstrap_counted_bytes if pair_index == 0 else 0,
                generator_id=(
                    "g2f_openpilot_per_class_geometric_frame0_prior"
                    if pair_index == 0
                    else "g2f_cross_pair_g1_advected_rgb"
                ),
                kernel=kernel,
            )
            realized_base1 = _contextual_realize(
                base1,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=0,
                generator_id="g2f_within_pair_g1_advected_rgb_base",
                kernel=kernel,
            )
            baseline_state = _candidate_seg_state(
                net,
                torch,
                realized0["frame"],
                realized_base1["frame"],
                base1,
                constraints,
                with_jacobian=True,
            )
            if pair_index < 16:
                prior_candidate = prior_stages[pair_index]["candidate_state"]
                if (
                    baseline_state["segnet_input_sha256"] != prior_candidate["segnet_input_sha256"]
                    or baseline_state["logits_sha256"] != prior_candidate["logits_sha256"]
                    or list(baseline_state["rivals"]) != prior_candidate["rivals"]
                ):
                    raise RealizationAuditError("G2f candidate state drifted from G2e rung-0 prior")
            baseline_hard, _, baseline_writes = _hard_oracle_interior(
                net,
                torch,
                realized0["frame"],
                realized_base1["frame"],
                target_cells,
                represented,
                target_pose,
                constraints,
            )
            local_jacobian = np.asarray(baseline_state["local_jacobian"], dtype=np.float64)
            chart_directions = _rank4_chart_directions(local_jacobian)
            effective_direction_count = _effective_chart_direction_count(chart_directions)
            bidirectional_rows: list[BidirectionalRungObservation] = []
            for direction_index in range(effective_direction_count):
                prior = prior_by_pair_direction.get((pair_index, direction_index))
                for rung_index, amplitude in enumerate(amplitudes):
                    branches: dict[int, tuple[SecantObservation, np.ndarray, str]] = {}
                    for sign in (1, -1):
                        signed_amplitude = sign * amplitude
                        reused = (
                            prior
                            if prior is not None
                            and math.isclose(
                                prior.signed_amplitude,
                                signed_amplitude,
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            )
                            else None
                        )
                        observation, actual_delta = _measure_or_reuse_secant_branch(
                            pair_index=pair_index,
                            direction_index=direction_index,
                            amplitude=signed_amplitude,
                            base1=base1,
                            constraints=constraints,
                            chart_direction=chart_directions[:, direction_index],
                            local_jacobian=local_jacobian,
                            baseline_state=baseline_state,
                            realized0_frame=realized0["frame"],
                            represented=represented,
                            seed_sha256=seed_sha256,
                            kernel=kernel,
                            net=net,
                            torch=torch,
                            reused_prior=reused,
                        )
                        branches[sign] = (
                            observation,
                            actual_delta,
                            "REUSED_G2E_RUNG0_PRIOR" if reused is not None else "MEASURED_G2F",
                        )
                    bidirectional_rows.append(
                        build_bidirectional_rung_observation(
                            positive=branches[1][0],
                            negative=branches[-1][0],
                            rung_index=rung_index,
                            strata=tuple(str(constraint["stratum"]) for constraint in constraints),
                            positive_source=branches[1][2],
                            negative_source=branches[-1][2],
                            positive_applied_rgb_delta=branches[1][1],
                            negative_applied_rgb_delta=branches[-1][1],
                        )
                    )
            expected_prior_reuse = 1 if pair_index < 16 else 0
            actual_prior_reuse_by_direction = Counter(
                observation.direction_index
                for observation in bidirectional_rows
                if observation.positive_source == "REUSED_G2E_RUNG0_PRIOR"
                or observation.negative_source == "REUSED_G2E_RUNG0_PRIOR"
            )
            if any(
                actual_prior_reuse_by_direction[direction] != expected_prior_reuse
                for direction in range(effective_direction_count)
            ):
                raise RealizationAuditError("G2f did not consume each available G2e rung-0 row exactly once")
            current_trust = list(
                build_bidirectional_trust_region_custody(
                    bidirectional_rows,
                    relative_residual_tolerance=SECANT_RELATIVE_RESIDUAL_TOLERANCE,
                )
            )
            prefix_trust = [
                trust for prior_pair in range(pair_index) for trust in rows[prior_pair]["amplitude_trust_regions"]
            ] + current_trust
            prefix_counts = [int(rows[prior_pair]["effective_direction_count"]) for prior_pair in range(pair_index)] + [
                effective_direction_count
            ]
            current_selections = [
                selection
                for selection in select_best_bidirectional_rungs(
                    prefix_trust,
                    effective_direction_count_by_pair=prefix_counts,
                )
                if selection["pair_index"] == pair_index
            ]
            if len(current_selections) != effective_direction_count:
                raise RealizationAuditError("G2f selected-rung coverage mismatch")
            selected_lookup = {
                (observation.direction_index, observation.rung_index): observation for observation in bidirectional_rows
            }
            solve = None
            tested_plane = base1
            tested_payload = b""
            tested_ordinals: list[int] = []
            coefficient_packet = b""
            status = SecantPairSolveStatus.TRUST_REGION_REFUSED
            selected_observations: list[BidirectionalRungObservation] = []
            if all(selection["selected"] is True for selection in current_selections):
                selected_observations = [
                    selected_lookup[(int(selection["direction_index"]), int(selection["selected_rung_index"]))]
                    for selection in current_selections
                ]
                direction_model = np.column_stack(
                    [
                        (
                            np.asarray(observation.positive_applied_rgb_delta, dtype=np.float64)
                            - np.asarray(observation.negative_applied_rgb_delta, dtype=np.float64)
                        )
                        / (2.0 * observation.amplitude)
                        for observation in selected_observations
                    ]
                )
                secant_model = np.column_stack(
                    [
                        np.asarray(
                            [write.odd_realized_secant for write in observation.writes],
                            dtype=np.float64,
                        )
                        for observation in selected_observations
                    ]
                )
                solve = solve_minimal_norm_inequalities(
                    secant_model,
                    SECANT_REQUIRED_MARGIN - baseline_state["margins"],
                    direction_model,
                    _local_rgb_values(base1, constraints),
                )
                if solve.status is SecantQPStatus.SOLVED:
                    coefficient_packet = encode_coefficient_packet(solve.coefficients)
                    decoded_once = decode_coefficient_packet(coefficient_packet)
                    decoded_twice = decode_coefficient_packet(coefficient_packet)
                    if decoded_once != decoded_twice or encode_coefficient_packet(decoded_once) != coefficient_packet:
                        raise RealizationAuditError("G2f coefficient packet double-decode drift")
                    tested_plane, _, _ = _apply_local_chart_delta(
                        base1,
                        constraints,
                        direction_model @ np.asarray(decoded_once, dtype=np.float64),
                    )
                    tested_payload, tested_plane, tested_ordinals = _exception_parseback(
                        base1,
                        constraints,
                        tested_plane,
                    )
                else:
                    status = SecantPairSolveStatus.QP_INFEASIBLE

            tested_path = tested_dir / f"pair_{pair_index:04d}.g2dx"
            _atomic_bytes(tested_path, tested_payload)
            tested_realized = _contextual_realize(
                tested_plane,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=len(tested_payload),
                generator_id="g2f_tested_bidirectional_secant_qp_correction",
                kernel=kernel,
            )
            tested_second = _contextual_realize(
                tested_plane,
                represented,
                seed_sha256=seed_sha256,
                additional_seed_bytes=len(tested_payload),
                generator_id="g2f_tested_bidirectional_secant_qp_correction_repeat",
                kernel=kernel,
            )
            tested_hard, _, tested_writes = _hard_oracle_interior(
                net,
                torch,
                realized0["frame"],
                tested_realized["frame"],
                target_cells,
                represented,
                target_pose,
                constraints,
            )
            tested_state = _candidate_seg_state(
                net,
                torch,
                realized0["frame"],
                tested_realized["frame"],
                tested_plane,
                constraints,
                fixed_rivals=baseline_state["rivals"],
                with_jacobian=False,
            )
            double_decode = bool(
                np.array_equal(tested_realized["frame"], tested_second["frame"])
                and tested_realized["camera_uint8_sha256"] == tested_second["camera_uint8_sha256"]
            )
            positive_fixed_margins = bool(np.all(tested_state["margins"] > SECANT_REQUIRED_MARGIN))
            hard_positive = bool(
                tested_hard["all_declared_writes_survive"]
                and all(float(row["target_logit_margin"]) > 0.0 for row in tested_writes)
            )
            semantic_score_delta = 100.0 * (
                float(baseline_hard["d_seg_realized_vs_frozen_target"])
                - float(tested_hard["d_seg_realized_vs_frozen_target"])
            )
            marginal_score_units_per_byte = (
                semantic_score_delta / len(tested_payload)
                if tested_payload
                else (math.inf if semantic_score_delta > 0.0 else 0.0)
            )
            rate_admissible = bool(not tested_payload or marginal_score_units_per_byte > rate_lambda)
            kkt_clean = bool(
                solve is not None
                and solve.status is SecantQPStatus.SOLVED
                and solve.max_primal_violation <= 1e-8
                and solve.stationarity_residual <= 1e-8
                and solve.min_active_multiplier is not None
                and solve.min_active_multiplier >= -1e-9
            )
            admitted = bool(
                kkt_clean and positive_fixed_margins and hard_positive and double_decode and rate_admissible
            )
            if solve is not None and solve.status is SecantQPStatus.SOLVED:
                if not positive_fixed_margins or not hard_positive:
                    status = SecantPairSolveStatus.NEGATIVE_REALIZED_HARD_ORACLE_REFUSED
                elif not rate_admissible:
                    status = SecantPairSolveStatus.RATE_BREAK_EVEN_REFUSED
                elif not kkt_clean:
                    status = SecantPairSolveStatus.KKT_RESIDUAL_REFUSED
                elif not double_decode:
                    status = SecantPairSolveStatus.DOUBLE_DECODE_REFUSED
                else:
                    status = SecantPairSolveStatus.ADMITTED_RECEIVER_CLOSED
            admitted_payload = tested_payload if admitted else b""
            admitted_plane = tested_plane if admitted else base1
            _atomic_bytes(sidecar_path, admitted_payload)
            if admitted:
                final_realized = tested_realized
                hard, writes = tested_hard, tested_writes
            else:
                final_realized = realized_base1
                hard, writes = baseline_hard, baseline_writes
            previous_plane = admitted_plane
            stage = {
                "schema": AMPLITUDE_STAGE_SCHEMA,
                "config_sha256": config_sha256,
                "pair_index": pair_index,
                "effective_direction_count": effective_direction_count,
                "candidate_state": {
                    "segnet_input_sha256": baseline_state["segnet_input_sha256"],
                    "logits_sha256": baseline_state["logits_sha256"],
                    "receiver_input_maxabs": baseline_state["receiver_input_maxabs"],
                    "head_input": "exact 16-channel input to segmentation_head[0]",
                    "head_patch_dimension": 144,
                    "rivals": list(baseline_state["rivals"]),
                    "fresh_candidate_response": True,
                    "source_arrangement_vjp_used_for_solve": False,
                },
                "bidirectional_observations": [row.as_dict() for row in bidirectional_rows],
                "amplitude_trust_regions": current_trust,
                "selected_rungs": current_selections,
                "pair_solve": {
                    "pair_index": pair_index,
                    "status": status.value,
                    "admitted": admitted,
                    "qp": solve.as_dict() if solve is not None else None,
                    "selected_direction_rungs": [
                        {
                            "direction_index": observation.direction_index,
                            "rung_index": observation.rung_index,
                            "amplitude": observation.amplitude,
                        }
                        for observation in selected_observations
                    ],
                    "coefficient_packet": {
                        "bytes": len(coefficient_packet),
                        "sha256": hashlib.sha256(coefficient_packet).hexdigest(),
                        "double_decode_identical": bool(coefficient_packet),
                    },
                    "tested_fixed_rival_margins": tested_state["margins"].tolist(),
                    "tested_positive_fixed_rival_margins": positive_fixed_margins,
                    "tested_hard_positive_declared_writes": hard_positive,
                    "tested_uint8_saturation_count": int(
                        sum(
                            observation.positive.uint8_saturation_count + observation.negative.uint8_saturation_count
                            for observation in selected_observations
                        )
                    ),
                    "tested_exception_ordinals": tested_ordinals,
                    "tested_packet_bytes": len(tested_payload),
                    "tested_packet_sha256": hashlib.sha256(tested_payload).hexdigest(),
                    "tested_packet_path": str(tested_path),
                    "double_decode_identical": double_decode,
                    "semantic_score_units_delta": semantic_score_delta,
                    "marginal_score_units_per_correction_byte": marginal_score_units_per_byte,
                    "rate_break_even": rate_lambda,
                    "rate_admissible": rate_admissible,
                    "baseline_hard_oracle": baseline_hard,
                    "tested_hard_oracle": tested_hard,
                    "pose_delta_from_semantic_correction": float(
                        tested_hard["d_pose_realized_vs_frozen_target"]
                        - baseline_hard["d_pose_realized_vs_frozen_target"]
                    ),
                },
                "correction_packet": {
                    "codec": "#557 adaptive range coder" if admitted_payload else "none",
                    "path": str(sidecar_path),
                    "bytes": len(admitted_payload),
                    "sha256": hashlib.sha256(admitted_payload).hexdigest(),
                    "video_derived_payload_carried": bool(admitted_payload),
                    "parseback_exact": True,
                    "reencode_byte_identical": True,
                },
                "hard_oracle": hard,
                "declared_write_survival": writes,
                "camera_frame0_sha256": realized0["camera_uint8_sha256"],
                "camera_frame1_sha256": final_realized["camera_uint8_sha256"],
                "uint8_factor2_exact": bool(
                    realized0["factor2_verification"]["certified_exact"]
                    and final_realized["factor2_verification"]["certified_exact"]
                ),
                "double_decode_identical": double_decode,
                "wall_seconds": time.perf_counter() - started,
                "authority": f"MEASURED {AXIS}",
                "score_claim": False,
                "promotion_eligible": False,
            }
            _atomic_json(stage_dir / f"pair_{pair_index:04d}.json", stage)
            rows[pair_index] = stage
            del target_cells, target_pose
        _atomic_json(
            root / "checkpoints" / f"chunk_{chunk_begin:04d}_{chunk_end:04d}.json",
            {
                "schema": "realization_g2f_bidirectional_amplitude_chunk_checkpoint.v1",
                "config_sha256": config_sha256,
                "completed_through_exclusive": chunk_end,
                "completed_pairs": len(rows),
                "all_pair_stages_preserved": True,
                "resumed_pairs_at_invocation_start": resumed_pairs,
            },
        )

    ordered = [rows[index] for index in range(stop_after_prefix)]
    base_bytes = CONTEXTUAL_SEED_BASELINE_BYTES + bootstrap_counted_bytes
    prefixes = []
    for prefix in PREFIXES:
        if prefix > stop_after_prefix:
            continue
        summary = summarize_bidirectional_amplitude_prefix(prefix, ordered[:prefix], base_bytes=base_bytes)
        checkpoint = root / "checkpoints" / f"prefix_n{prefix}.json"
        _atomic_json(checkpoint, summary)
        summary["checkpoint_path"] = str(checkpoint)
        summary["checkpoint_sha256"] = _sha256(checkpoint)
        prefixes.append(summary)
    final = prefixes[-1]
    all_observations = [
        BidirectionalRungObservation.from_dict(raw) for row in ordered for raw in row["bidirectional_observations"]
    ]
    all_trust = list(
        build_bidirectional_trust_region_custody(
            all_observations,
            relative_residual_tolerance=SECANT_RELATIVE_RESIDUAL_TOLERANCE,
        )
    )
    effective_counts = [int(row["effective_direction_count"]) for row in ordered]
    all_selections = list(
        select_best_bidirectional_rungs(
            all_trust,
            effective_direction_count_by_pair=effective_counts,
        )
    )
    if all_trust != [trust for row in ordered for trust in row["amplitude_trust_regions"]]:
        raise RealizationAuditError("G2f final trust rederivation drift")
    if all_selections != [selection for row in ordered for selection in row["selected_rungs"]]:
        raise RealizationAuditError("G2f final selected-rung rederivation drift")
    admission = final["predict_project_realization_admissibility_v1"]
    all_directions_usable_pairs = final["D2_selected_trust"]["all_directions_usable_pair_count"]
    admitted_pairs = final["D3_receiver_closed_qp"]["admitted_pair_count"]
    if all_directions_usable_pairs == 0:
        verdict = f"MEASURED_G2F_BIDIRECTIONAL_TRUST_EMPTY_N{stop_after_prefix}_FAMILY_OPEN"
        blocker = f"R1B2_RANK4_BIDIRECTIONAL_LOCAL_LINEAR_TRUST_REGION_EMPTY_N{stop_after_prefix}_OPENPILOT"
    elif admitted_pairs == 0:
        verdict = f"MEASURED_G2F_BIDIRECTIONAL_QP_NO_ADMISSION_N{stop_after_prefix}_FAMILY_OPEN"
        blocker = f"R1B2_RANK4_BIDIRECTIONAL_RECEIVER_QP_NO_ADMISSION_N{stop_after_prefix}_OPENPILOT"
    else:
        verdict = f"MEASURED_G2F_BIDIRECTIONAL_RECEIVER_ADMISSIONS_N{stop_after_prefix}"
        blocker = None
    if admitted_pairs:
        d4 = {
            "status": f"MEASURED_D3_ADMITTED {AXIS}",
            "admitted_pair_count": admitted_pairs,
            "semantic_score_units_delta_sum": float(
                sum(row["pair_solve"]["semantic_score_units_delta"] for row in ordered if row["pair_solve"]["admitted"])
            ),
            "correction_bytes": final["byte_accounting"]["correction_bytes"],
            "route": "#598 r5 witness-anchor waterfill after MAIN review",
        }
    else:
        d4 = {
            "status": "NOT_RUN_D3_NO_ADMISSION",
            "admitted_pair_count": 0,
            "semantic_score_units_delta_sum": 0.0,
            "correction_bytes": 0,
            "route": None,
        }
    receipt: dict[str, Any] = {
        "schema": BIDIRECTIONAL_RECEIPT_SCHEMA,
        "lane_id": AMPLITUDE_LANE_ID,
        "task_id": "578",
        "config": config,
        "config_sha256": config_sha256,
        "completed_prefix": stop_after_prefix,
        "effective_direction_count_by_pair": effective_counts,
        "g2e_rung0_prior_observation_hashes": [row["row_sha256"] for row in g2e_prior["secant_observations"]],
        "bidirectional_observations": [row.as_dict() for row in all_observations],
        "pair_direction_rung_trust_regions": all_trust,
        "selected_rungs": all_selections,
        "pair_solves": [row["pair_solve"] for row in ordered],
        "D1_bidirectional_amplitude_ladder": final["D1_response_curve"],
        "D2_rebuilt_trust_regions": {
            **final["D2_selected_trust"],
            "active_blocker": blocker,
            "verdict_scope": (
                f"local-linear bidirectional rank-4 realization charts on exact contiguous n{stop_after_prefix} "
                "openpilot-base pairs; exact fp32 rank-4 quotient geometry and integer-lattice families remain open"
            ),
            "next_formulation_if_empty": (
                "#586 saturated integer-lattice/NFS-style exact preimage formulation"
                if all_directions_usable_pairs == 0
                else None
            ),
        },
        "D3_receiver_closed_qp": {
            **final["D3_receiver_closed_qp"],
            "byte_accounting": final["byte_accounting"],
            "admissibility": admission,
        },
        "D4_openpilot_base_delta": d4,
        "admissibility": admission,
        "verdict": verdict,
        "verdict_scope": (
            f"exact contiguous n{stop_after_prefix} prefix only; paired +/- amplitude ladder derived from exact "
            "R gain, candidate-state local rank-4 charts, and receiver-closed QP; no n600 claim unless "
            "completed_prefix is 600; no rank-4 quotient-family or integer-lattice-family negative"
        ),
        "storage": {
            "root": str(root),
            "free_bytes_at_preflight": usage.free,
            "automatic_disk_hygiene": (
                "ZIP_STORED cache memmaps, current pair/rung scorer tensors only, atomic immutable JSON and "
                "#557 sidecars, no persisted camera/logit/feature tensors; certify-or-block before deletion"
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
    receipt["receipt_sha256"] = secant_canonical_sha256(receipt)
    validate_bidirectional_receipt(receipt, expected_pair_count=stop_after_prefix)
    _atomic_json(root / "receipt.json", receipt)
    _atomic_json(output_root / "receipt.json", receipt)
    if _load_json(root / "receipt.json")["receipt_sha256"] != receipt["receipt_sha256"]:
        raise RealizationAuditError("G2f receipt JSON parse-back drift")
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
        "--frame0-prior-race",
        action="store_true",
        help="race exact I-frame, seed keyframe classes, and the openpilot per-class geometric prior",
    )
    parser.add_argument(
        "--realized-secant-custody",
        action="store_true",
        help="measure candidate-state rank-4 secants and receiver-closed per-pair QP corrections",
    )
    parser.add_argument(
        "--bidirectional-amplitude-ladder",
        action="store_true",
        help="measure paired +/- rank-4 secants across an R-derived amplitude ladder",
    )
    parser.add_argument(
        "--secant-output-root",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/g2e_secant_20260721"),
        help="durable SSD root for --realized-secant-custody stages and receipts",
    )
    parser.add_argument(
        "--amplitude-output-root",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/g2f_amplitude_20260721"),
        help="durable SSD root for --bidirectional-amplitude-ladder stages and receipts",
    )
    parser.add_argument("--g2e-prior-receipt", type=Path, default=DEFAULT_G2E_PRIOR_RECEIPT)
    parser.add_argument("--static-chart", type=Path, default=DEFAULT_FRAME0_STATIC_CHART)
    parser.add_argument("--lane-chart", type=Path, default=DEFAULT_FRAME0_LANE_CHART)
    parser.add_argument("--vjp-campaign", type=Path, default=DEFAULT_VJP_CAMPAIGN)
    parser.add_argument("--m1-band-receipt", type=Path, default=DEFAULT_M1_BAND_RECEIPT)
    parser.add_argument("--m1-inner-jacobian", type=Path, default=DEFAULT_M1_INNER_JACOBIAN)
    parser.add_argument("--rank4-prototype-receipt", type=Path, default=DEFAULT_RANK4_PROTOTYPE_RECEIPT)
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
    if (
        sum(
            (
                args.audit_only,
                args.interior_rungs,
                args.contextual_predict_base,
                args.frame0_prior_race,
                args.realized_secant_custody,
                args.bidirectional_amplitude_ladder,
            )
        )
        > 1
    ):
        raise RealizationAuditError(
            "--audit-only, --interior-rungs, --contextual-predict-base, --frame0-prior-race, "
            "--realized-secant-custody, and --bidirectional-amplitude-ladder are mutually exclusive"
        )
    if args.bidirectional_amplitude_ladder:
        receipt = run_bidirectional_amplitude_ladder(
            seed_path=args.seed.resolve(),
            gt_cache_path=args.gt_cache.resolve(),
            upstream=args.upstream.resolve(),
            output_root=args.amplitude_output_root.resolve(),
            g2e_prior_receipt_path=args.g2e_prior_receipt.resolve(),
            rank4_prototype_receipt_path=args.rank4_prototype_receipt.resolve(),
            static_chart_path=args.static_chart.resolve(),
            lane_chart_path=args.lane_chart.resolve(),
            chunk_size=args.chunk_size,
            threads=args.threads,
            stop_after_prefix=args.stop_after_prefix,
        )
        print(
            json.dumps(
                {
                    "receipt": str(args.amplitude_output_root.resolve() / "receipt.json"),
                    "verdict": receipt["verdict"],
                    "completed_prefix": receipt["completed_prefix"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.realized_secant_custody:
        receipt = run_realized_secant_custody(
            seed_path=args.seed.resolve(),
            gt_cache_path=args.gt_cache.resolve(),
            upstream=args.upstream.resolve(),
            output_root=args.secant_output_root.resolve(),
            rank4_prototype_receipt_path=args.rank4_prototype_receipt.resolve(),
            static_chart_path=args.static_chart.resolve(),
            lane_chart_path=args.lane_chart.resolve(),
            chunk_size=args.chunk_size,
            threads=args.threads,
            stop_after_prefix=args.stop_after_prefix,
        )
        print(
            json.dumps(
                {
                    "receipt": str(args.secant_output_root.resolve() / "receipt.json"),
                    "verdict": receipt["verdict"],
                    "completed_prefix": receipt["completed_prefix"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.frame0_prior_race:
        receipt = run_frame0_prior_race(
            seed_path=args.seed.resolve(),
            gt_cache_path=args.gt_cache.resolve(),
            upstream=args.upstream.resolve(),
            output_root=args.output_root.resolve(),
            static_chart_path=args.static_chart.resolve(),
            lane_chart_path=args.lane_chart.resolve(),
            vjp_campaign_path=args.vjp_campaign.resolve(),
            m1_band_receipt_path=args.m1_band_receipt.resolve(),
            m1_inner_jacobian_path=args.m1_inner_jacobian.resolve(),
            rank4_prototype_receipt_path=args.rank4_prototype_receipt.resolve(),
            threads=args.threads,
        )
        print(
            json.dumps(
                {
                    "receipt": str(args.output_root.resolve() / "receipt.json"),
                    "verdict": receipt["verdict"],
                    "base_only_winner": receipt["base_only_winner"],
                },
                sort_keys=True,
            )
        )
        return 0
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
