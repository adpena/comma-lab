# SPDX-License-Identifier: MIT
"""Distilled dashcam-statistical codebook — 5-10 KB tiny prior.

The codebook is a compact, FROZEN structure of driving-distribution primitives
distilled OFFLINE from publicly-available dashcam datasets (Comma2k19 MIT;
BDD100K BSD-3 code + dataset-images research-only opt-in). At inflate time the
codebook is loaded once, never trained against the contest video, and used as a
soft prior that biases the renderer toward dashcam-plausible outputs.

Codebook structure (target ~5-10 KB total after brotli):

* ``road_plane_basis`` — 8 PCA components of road-surface RGB statistics
  (sampled from Comma2k19 highway segments), shape ``(8, 16, 24, 3)`` at
  log-polar foveated resolution, int8-quantized ≈ 8*16*24*3 = 9.2 KB raw,
  brotli ≈ 3 KB.
* ``sky_horizon_band`` — 1D vertical profile of sky-to-horizon-to-road
  brightness/chroma transition, shape ``(64, 3)`` int8 ≈ 192 B, brotli ≈ 80 B.
* ``lane_curvature_pca`` — 8 PCA components of lane-marker parametric forms
  (B-spline control points), shape ``(8, 6)`` float16 ≈ 96 B.
* ``vehicle_appearance_basis`` — 4 PCA components of vehicle-silhouette
  templates at distance bins, shape ``(4, 12, 16, 3)`` int8 ≈ 2.3 KB,
  brotli ≈ 800 B.
* ``codebook_metadata`` — JSON with dataset provenance, distillation
  parameters, dataset SHA-256s, dataset license tags.

**Strategic context** (operator 2026-05-13): the substrate is dual-purpose —
contest-score side-lane AND production-deployment alignment. The codebook's
shape is designed so future Comma edge devices can compute LOCAL deltas + push
upstream (federated-learning architecture; per-vehicle prior refinement). The
inflate-time consumer is identical whether the codebook came from the offline
distillation or a federated aggregation update.

**Predicted score impact** (operator 2026-05-13, my honest analysis): the
contest scorer (FastViT-T12 + EfficientNet-B2) was ALREADY trained on driving
data, so it implicitly contains the dashcam prior. Adding ANOTHER prior on top
is partially redundant. Predicted Δ contest-CPU score: **-0.005 to -0.012**
(NOT the -0.020 to -0.030 the 4th-team memo §2.3 floated — that prediction
assumed independent prior signal, which I judge overstated). This substrate
ranks MEDIUM-EV at the contest, but HIGH-EV for production-deployment-shaped
contest entry where the prior IS the architecture that scales.

**CLAUDE.md compliance:**

* No archive bytes mutated at inflate beyond the codebook lookup
* No scorer load at inflate (per "Strict scorer rule" + Catalog #6)
* No /tmp paths (per "Forbidden /tmp paths")
* Deterministic (fixed PCA basis, fixed int8 quantization, fixed brotli quality)
* Per HNeRV parity discipline L1: codebook is OFFLINE training-data feature,
  NOT trained against the contest video — score-aware loss handles
  contest-specific adaptation via the per-pair residual
* Per HNeRV parity discipline L2: export-first design — the codebook archive
  grammar is declared in :mod:`tac.substrates.pretrained_driving_prior.archive`
* Per HNeRV parity discipline L7: bolt-on LOC ≤ 350 (this module ~250 LOC)

License attribution:

* Comma2k19 — MIT (https://github.com/commaai/comma2k19); commercial-OK
* BDD100K code — BSD-3-Clause (https://github.com/bdd100k/bdd100k); commercial-OK
* BDD100K dataset images — UC Berkeley research/academic terms; OPT-IN flag
  ``--allow-bdd100k-dataset-images`` required; the scaffold default uses
  Comma2k19 only to stay commercial-clean
* Waymo Open Dataset — non-commercial only; **SKIPPED by design**

Per CLAUDE.md Catalog #124 archive-grammar 8 fields (inline so AST walker sees):

* ``archive_grammar``: monolithic single-file ``0.bin`` (substrate-engineering
  waiver) with 4 length-prefixed codebook sections + contest renderer +
  per-pair residual
* ``parser_section_manifest``: DP1 magic + version + per-section offsets
* ``inflate_runtime_loc_budget``: <= 200 LOC substantive
* ``runtime_dep_closure``: torch + brotli only (HNeRV parity L4)
* ``export_format``: DP1 monolithic single-zip-member ``0.bin``
* ``score_aware_loss``: rate(codebook) + rate(residual) + d_seg + d_pose with
  eval-roundtrip + Atick-Redlich cooperative-receiver
* ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV parity L7)
* ``no_op_detector_planned``: emit/parse roundtrip preserves bytes; archive
  payload is structurally consumed by every section of inflate.py
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

# Codebook target shapes — tuned for ~5-10 KB after int8 + brotli.
ROAD_PLANE_BASIS_SHAPE: tuple[int, int, int, int] = (8, 16, 24, 3)
"""8 PCA components x log-polar 16x24 grid x 3 RGB channels."""

SKY_HORIZON_PROFILE_SHAPE: tuple[int, int] = (64, 3)
"""64 vertical samples x 3 RGB channels for sky-to-road transition."""

LANE_CURVATURE_PCA_SHAPE: tuple[int, int] = (8, 6)
"""8 lane-shape PCA components x 6 B-spline control coords."""

VEHICLE_APPEARANCE_BASIS_SHAPE: tuple[int, int, int, int] = (4, 12, 16, 3)
"""4 components x 12x16 silhouette x 3 RGB."""

CODEBOOK_TOTAL_TARGET_BYTES_MIN: int = 5_000
"""Minimum acceptable codebook size after brotli (rate-axis floor)."""

CODEBOOK_TOTAL_TARGET_BYTES_MAX: int = 10_000
"""Maximum acceptable codebook size after brotli (rate-axis ceiling)."""

_BROTLI_QUALITY: int = 9
"""Deterministic brotli quality (matches SIREN / TT5L)."""

# Codebook section magic (precedes each blob in archive grammar).
_SECTION_ROAD_PLANE: bytes = b"RP"
_SECTION_SKY_HORIZON: bytes = b"SH"
_SECTION_LANE_CURV: bytes = b"LC"
_SECTION_VEHICLE: bytes = b"VH"
_SECTION_META: bytes = b"MT"


@dataclass(frozen=True)
class DashcamCodebook:
    """Distilled dashcam-statistical codebook — the inflate-time data contract.

    All arrays are int8 (or fp16 for lane_curvature_pca where the dynamic
    range is wider). Floating-point scales for dequantization live in
    :attr:`metadata`.
    """

    road_plane_basis: np.ndarray
    """Int8 array shape ``ROAD_PLANE_BASIS_SHAPE``; dequant via metadata['road_plane_scale']."""

    sky_horizon_profile: np.ndarray
    """Int8 array shape ``SKY_HORIZON_PROFILE_SHAPE``; dequant via metadata['sky_horizon_scale']."""

    lane_curvature_pca: np.ndarray
    """Float16 array shape ``LANE_CURVATURE_PCA_SHAPE`` — no quant needed."""

    vehicle_appearance_basis: np.ndarray
    """Int8 array shape ``VEHICLE_APPEARANCE_BASIS_SHAPE``; dequant via metadata['vehicle_scale']."""

    metadata: dict[str, object]
    """JSON dict with dataset provenance, scales, license tags, distillation params."""


def _validate_array(name: str, arr: np.ndarray, expected_shape: tuple, dtype: np.dtype) -> None:
    """Fail-loud array shape/dtype validation."""
    if arr.shape != expected_shape:
        raise ValueError(
            f"codebook {name}: shape {arr.shape} != expected {expected_shape}"
        )
    if arr.dtype != dtype:
        raise ValueError(
            f"codebook {name}: dtype {arr.dtype} != expected {dtype}"
        )


def validate_codebook(book: DashcamCodebook) -> None:
    """Validate every codebook section shape/dtype and metadata invariants.

    Raises ValueError on any structural mismatch. Caller must wrap in
    try/except if it wants soft failure; the scaffold prefers fail-loud
    per CLAUDE.md "fail-fast validation at every boundary".
    """
    _validate_array(
        "road_plane_basis", book.road_plane_basis,
        ROAD_PLANE_BASIS_SHAPE, np.dtype(np.int8),
    )
    _validate_array(
        "sky_horizon_profile", book.sky_horizon_profile,
        SKY_HORIZON_PROFILE_SHAPE, np.dtype(np.int8),
    )
    _validate_array(
        "lane_curvature_pca", book.lane_curvature_pca,
        LANE_CURVATURE_PCA_SHAPE, np.dtype(np.float16),
    )
    _validate_array(
        "vehicle_appearance_basis", book.vehicle_appearance_basis,
        VEHICLE_APPEARANCE_BASIS_SHAPE, np.dtype(np.int8),
    )
    required_meta_keys = {
        "road_plane_scale",
        "sky_horizon_scale",
        "vehicle_scale",
        "dataset_provenance",
        "distillation_version",
        "license_tags",
    }
    missing = required_meta_keys - set(book.metadata.keys())
    if missing:
        raise ValueError(f"codebook metadata missing required keys: {sorted(missing)}")


def serialize_codebook(book: DashcamCodebook) -> bytes:
    """Serialize codebook to deterministic length-prefixed bytes.

    Layout (each section is brotli-compressed except where noted):

        SECTION_TAG(2) + LEN(4) + brotli(payload)

    The metadata JSON is sorted-keys for byte-determinism. Total target
    after brotli: 5-10 KB.
    """
    validate_codebook(book)
    buf = io.BytesIO()

    def _write_section(tag: bytes, payload: bytes) -> None:
        if len(tag) != 2:
            raise ValueError(f"section tag must be 2 bytes; got {tag!r}")
        compressed = brotli.compress(payload, quality=_BROTLI_QUALITY)
        buf.write(tag)
        buf.write(struct.pack("<I", len(compressed)))
        buf.write(compressed)

    _write_section(_SECTION_ROAD_PLANE, book.road_plane_basis.tobytes(order="C"))
    _write_section(_SECTION_SKY_HORIZON, book.sky_horizon_profile.tobytes(order="C"))
    _write_section(
        _SECTION_LANE_CURV, book.lane_curvature_pca.tobytes(order="C")
    )
    _write_section(
        _SECTION_VEHICLE, book.vehicle_appearance_basis.tobytes(order="C")
    )
    meta_json = json.dumps(book.metadata, sort_keys=True).encode("utf-8")
    _write_section(_SECTION_META, meta_json)

    return buf.getvalue()


def parse_codebook(data: bytes) -> DashcamCodebook:
    """Parse codebook bytes back into a :class:`DashcamCodebook`.

    Raises ValueError on any structural inconsistency (missing section,
    short read, wrong magic, shape mismatch).
    """
    cursor = 0
    sections: dict[bytes, bytes] = {}

    while cursor < len(data):
        if cursor + 6 > len(data):
            raise ValueError(
                f"codebook truncated at offset {cursor}; need 6 bytes for header"
            )
        tag = data[cursor : cursor + 2]
        length = struct.unpack("<I", data[cursor + 2 : cursor + 6])[0]
        cursor += 6
        if cursor + length > len(data):
            raise ValueError(
                f"codebook section {tag!r} declared length {length} exceeds remaining {len(data) - cursor}"
            )
        compressed = data[cursor : cursor + length]
        cursor += length
        try:
            payload = brotli.decompress(compressed)
        except Exception as exc:
            raise ValueError(
                f"codebook section {tag!r} brotli decompress failed: {exc}"
            ) from exc
        sections[tag] = payload

    required = {
        _SECTION_ROAD_PLANE,
        _SECTION_SKY_HORIZON,
        _SECTION_LANE_CURV,
        _SECTION_VEHICLE,
        _SECTION_META,
    }
    missing = required - set(sections.keys())
    if missing:
        raise ValueError(f"codebook missing required sections: {sorted(missing)}")

    road_plane = np.frombuffer(
        sections[_SECTION_ROAD_PLANE], dtype=np.int8
    ).reshape(ROAD_PLANE_BASIS_SHAPE)
    sky_horizon = np.frombuffer(
        sections[_SECTION_SKY_HORIZON], dtype=np.int8
    ).reshape(SKY_HORIZON_PROFILE_SHAPE)
    lane_curv = np.frombuffer(
        sections[_SECTION_LANE_CURV], dtype=np.float16
    ).reshape(LANE_CURVATURE_PCA_SHAPE)
    vehicle = np.frombuffer(
        sections[_SECTION_VEHICLE], dtype=np.int8
    ).reshape(VEHICLE_APPEARANCE_BASIS_SHAPE)
    meta = json.loads(sections[_SECTION_META].decode("utf-8"))

    book = DashcamCodebook(
        road_plane_basis=road_plane.copy(),  # frombuffer returns read-only
        sky_horizon_profile=sky_horizon.copy(),
        lane_curvature_pca=lane_curv.copy(),
        vehicle_appearance_basis=vehicle.copy(),
        metadata=meta,
    )
    validate_codebook(book)
    return book


def deterministic_zero_codebook() -> DashcamCodebook:
    """Construct a deterministic zero-filled codebook for scaffold tests.

    Used by tests and the L0 readiness path before real distillation runs.
    Production code MUST distill from real data (see :mod:`distillation`).
    """
    return DashcamCodebook(
        road_plane_basis=np.zeros(ROAD_PLANE_BASIS_SHAPE, dtype=np.int8),
        sky_horizon_profile=np.zeros(SKY_HORIZON_PROFILE_SHAPE, dtype=np.int8),
        lane_curvature_pca=np.zeros(LANE_CURVATURE_PCA_SHAPE, dtype=np.float16),
        vehicle_appearance_basis=np.zeros(VEHICLE_APPEARANCE_BASIS_SHAPE, dtype=np.int8),
        metadata={
            "road_plane_scale": 64.0,
            "sky_horizon_scale": 64.0,
            "vehicle_scale": 64.0,
            "dataset_provenance": "scaffold_zero_codebook",
            "distillation_version": "v0_scaffold",
            "license_tags": ["scaffold-only"],
        },
    )


def codebook_to_torch_tensors(
    book: DashcamCodebook, *, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Convert codebook arrays into dequantized float32 torch tensors.

    The inflate-time renderer + score-aware loss consume float32 tensors;
    the int8/fp16 storage is for archive compactness only. Dequantization
    multiplies by the per-section scale from :attr:`DashcamCodebook.metadata`.
    """
    validate_codebook(book)
    return {
        "road_plane_basis": (
            torch.from_numpy(book.road_plane_basis).float() / float(book.metadata["road_plane_scale"])
        ).to(device),
        "sky_horizon_profile": (
            torch.from_numpy(book.sky_horizon_profile).float() / float(book.metadata["sky_horizon_scale"])
        ).to(device),
        "lane_curvature_pca": torch.from_numpy(book.lane_curvature_pca).float().to(device),
        "vehicle_appearance_basis": (
            torch.from_numpy(book.vehicle_appearance_basis).float() / float(book.metadata["vehicle_scale"])
        ).to(device),
    }


__all__ = [
    "CODEBOOK_TOTAL_TARGET_BYTES_MAX",
    "CODEBOOK_TOTAL_TARGET_BYTES_MIN",
    "LANE_CURVATURE_PCA_SHAPE",
    "ROAD_PLANE_BASIS_SHAPE",
    "SKY_HORIZON_PROFILE_SHAPE",
    "VEHICLE_APPEARANCE_BASIS_SHAPE",
    "DashcamCodebook",
    "codebook_to_torch_tensors",
    "deterministic_zero_codebook",
    "parse_codebook",
    "serialize_codebook",
    "validate_codebook",
]
