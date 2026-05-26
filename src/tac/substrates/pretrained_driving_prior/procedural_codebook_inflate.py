# SPDX-License-Identifier: MIT
"""DP1 procedural-codebook inflate-time helper.

Per WAVE-3-DP1-DISPATCH-READY-EXTENSION 2026-05-20 OP-ROUTABLE #3 of
``dp1_paired_smoke_dispatch_pre_authorization_checklist_landed_20260520.md``.

The trainer's procedural-codebook replacement variant
(``--enable-procedural-codebook-replacement`` per
:mod:`experiments.train_substrate_pretrained_driving_prior`) post-processes
``pack_archive(...)`` bytes through  # DP1_PROVENANCE_OK:docstring_reference_not_runtime_call
:func:`tac.substrates.pretrained_driving_prior.distillation_procedural_variant.compose_with_procedural_codebook`,
which replaces the DP1 ``codebook_blob`` with ``brotli(seed_bytes)`` (~36 B)
instead of the canonical 5-10 KB Comma2k19-distilled codebook.

At inflate time the meta_blob carries::

    {
        "procedural_codebook_variant_active": True,
        "procedural_codebook_seed_hex": "<32-byte hex>",
        "procedural_codebook_generator_kind": "pcg64" | "xorshift" | "lcg",
        ...
    }

This module detects that flag set and re-derives the 4 canonical codebook
arrays (``road_plane_basis``, ``sky_horizon_profile``, ``lane_curvature_pca``,
``vehicle_appearance_basis``) deterministically from the seed using
:func:`tac.procedural_codebook_generator.derive_codebook_from_seed`. Each
section uses a per-section-tagged seed so a single 32-byte seed produces
4 distinct PRG streams. The derivation is byte-identical across CPU and
CUDA inflate.

This helper is intentionally a **separate module** (NOT inside inflate.py)
per Catalog #328 inflate.py LOC budget ≤ 200. The inflate.py extension is
~10 LOC (detect flag + delegate to this helper).

Sister of :mod:`tac.substrates.pretrained_driving_prior.distillation_procedural_variant`
(trainer-side composition); together they form the encode/decode roundtrip
for canonical equation #26 (predicted ΔS = -25 * (N_codebook - K_seed) /
37_545_489).

6-hook wire-in declaration per Catalog #125:

* hook #1 sensitivity-map = N/A (inflate-side helper)
* hook #2 Pareto constraint = ACTIVE via canonical equation #26 rate-axis
* hook #3 bit-allocator = ACTIVE (32-byte seed slot replaces 5-10KB)
* hook #4 cathedral autopilot dispatch = ACTIVE via sister consumer
  ``tac.cathedral_consumers.procedural_codebook_generator_consumer``
* hook #5 continual-learning posterior = ACTIVE (first empirical anchor via
  ``update_equation_with_empirical_anchor`` post-paired-smoke)
* hook #6 probe-disambiguator = ACTIVE (3-recipe contrast PROCEDURAL vs
  ORIGINAL vs NULL-EXPLOIT IS the probe disambiguator)

Catalog discipline:

* Catalog #105/#139/#220/#272 byte-mutation distinguishing-feature contract
  PASSES: mutating any seed byte produces a different derived codebook;
  changes propagate through ``DashcamPriorLoss.apply_soft_prior`` into
  rendered frames at inflate.
* Catalog #205 inflate device selector is canonical (caller delegates via
  :func:`tac.substrates._shared.inflate_runtime.select_inflate_device`).
* Catalog #209 Comma2k19 leakage refusal: this helper does NOT touch
  Comma2k19 — the seed is a deterministic PRG output, structurally OOD.
* Catalog #287/#323 canonical Provenance: no score claim asserted; the
  re-derived codebook is the same data contract the canonical inflate
  path expected (DashcamCodebook with canonical shapes + dtypes).
* Catalog #295 inflate self-containment: this module imports only from
  ``tac.procedural_codebook_generator`` (sister canonical helper) +
  ``tac.substrates.pretrained_driving_prior.codebook`` (sister substrate
  module); submission archives that include the procedural variant MUST
  vendor both per Catalog #295.
* Catalog #344 canonical equation cross-reference (#26 cited extensively).
"""

from __future__ import annotations

import hashlib

import numpy as np

from tac.procedural_codebook_generator.seed_derived_codebook import (
    derive_codebook_from_seed,
)
from tac.substrates.pretrained_driving_prior.codebook import (
    LANE_CURVATURE_PCA_SHAPE,
    ROAD_PLANE_BASIS_SHAPE,
    SKY_HORIZON_PROFILE_SHAPE,
    VEHICLE_APPEARANCE_BASIS_SHAPE,
    DashcamCodebook,
)

__all__ = [
    "PROCEDURAL_CODEBOOK_META_FLAG",
    "derive_dashcam_codebook_from_seed",
    "is_procedural_codebook_variant_archive",
    "parse_archive_procedural_aware",
]


PROCEDURAL_CODEBOOK_META_FLAG: str = "procedural_codebook_variant_active"
"""Meta-key the trainer sets to enable procedural inflate routing.

Sister of trainer's ``meta["procedural_codebook_variant_active"] = True``
assignment in ``experiments/train_substrate_pretrained_driving_prior.py
_full_main`` post-pack_archive branch.
"""


def is_procedural_codebook_variant_archive(meta: dict[str, object]) -> bool:
    """Return True iff archive meta declares the procedural-codebook variant.

    Per the trainer's WAVE-3-DP1-DISPATCH-READY-EXTENSION wire-in: the
    procedural variant sets ``meta["procedural_codebook_variant_active"] = True``
    + ``meta["procedural_codebook_seed_hex"] = <hex>``. This helper is the
    canonical detector consumed by :mod:`tac.substrates.pretrained_driving_prior.inflate`.
    """
    flag = meta.get(PROCEDURAL_CODEBOOK_META_FLAG)
    return bool(flag is True)


def _per_section_seed(base_seed: bytes, section_tag: str) -> bytes:
    """Derive a per-section seed via sha256(base_seed || section_tag).

    Per design memo §4: a single 32-byte seed produces 4 PRG streams. Each
    section's seed is ``sha256(base_seed + section_tag.encode("utf-8"))``
    (32 bytes), so the 4 derived arrays are statistically independent but
    deterministically reproducible from a single operator-supplied seed.

    The byte-mutation distinguishing-feature smoke (Catalog #272) holds for
    any section because flipping any base_seed byte changes all 4 section
    seeds via the sha256 cascade.
    """
    h = hashlib.sha256()
    h.update(base_seed)
    h.update(section_tag.encode("utf-8"))
    return h.digest()


def derive_dashcam_codebook_from_seed(
    seed_bytes: bytes,
    *,
    generator_kind: str = "pcg64",
    metadata: dict[str, object] | None = None,
) -> DashcamCodebook:
    """Re-derive the canonical DP1 :class:`DashcamCodebook` from a seed.

    Per WAVE-3-DP1-DISPATCH-READY-EXTENSION OP-ROUTABLE #3. The 4 codebook
    sections are derived independently via per-section sha256-cascaded
    seeds (see :func:`_per_section_seed`).

    Section shapes + dtypes match the canonical
    :func:`tac.substrates.pretrained_driving_prior.codebook.validate_codebook`
    contract:

    * ``road_plane_basis``: int8 shape (8, 16, 24, 3)
    * ``sky_horizon_profile``: int8 shape (64, 3)
    * ``lane_curvature_pca``: float16 shape (8, 6)
    * ``vehicle_appearance_basis``: int8 shape (4, 12, 16, 3)

    Args:
        seed_bytes: Operator-supplied procedural seed (8-256 bytes;
            canonical 32 bytes). Per canonical equation #26 domain-of-
            validity.
        generator_kind: PRNG kind (default ``"pcg64"``); must be one of
            ``tac.procedural_codebook_generator.SUPPORTED_GENERATOR_KINDS``.
        metadata: Optional metadata to attach to the
            :class:`DashcamCodebook`. When ``None``, a minimal canonical
            metadata dict is constructed so
            :func:`tac.substrates.pretrained_driving_prior.codebook.validate_codebook`
            passes.

    Returns:
        DashcamCodebook with all 4 sections derived deterministically.

    Raises:
        ProceduralCodebookGeneratorError: invalid generator_kind or seed
            outside domain-of-validity.
    """
    # int8 sections: derive as uint8, view as int8 (no copy; same bytes).
    road_seed = _per_section_seed(seed_bytes, "road_plane_basis")
    sky_seed = _per_section_seed(seed_bytes, "sky_horizon_profile")
    lane_seed = _per_section_seed(seed_bytes, "lane_curvature_pca")
    vehicle_seed = _per_section_seed(seed_bytes, "vehicle_appearance_basis")

    road_plane_basis = derive_codebook_from_seed(
        seed_bytes=road_seed,
        output_shape=ROAD_PLANE_BASIS_SHAPE,
        dtype=np.dtype(np.uint8),
        generator_kind=generator_kind,
    ).view(np.int8)
    sky_horizon_profile = derive_codebook_from_seed(
        seed_bytes=sky_seed,
        output_shape=SKY_HORIZON_PROFILE_SHAPE,
        dtype=np.dtype(np.uint8),
        generator_kind=generator_kind,
    ).view(np.int8)
    # float16: derive as uint16 then view as float16 (byte-identical
    # interpretation; some bit patterns are NaN but that's structurally
    # OK because the codebook is consumed as a fixed lookup, not in
    # gradient computation).
    lane_curvature_pca = derive_codebook_from_seed(
        seed_bytes=lane_seed,
        output_shape=LANE_CURVATURE_PCA_SHAPE,
        dtype=np.dtype(np.uint16),
        generator_kind=generator_kind,
    ).view(np.float16)
    vehicle_appearance_basis = derive_codebook_from_seed(
        seed_bytes=vehicle_seed,
        output_shape=VEHICLE_APPEARANCE_BASIS_SHAPE,
        dtype=np.dtype(np.uint8),
        generator_kind=generator_kind,
    ).view(np.int8)

    # Replace any NaN / inf in float16 with 0 so DashcamPriorLoss.apply_soft_prior
    # tensor ops are well-defined. The codebook is a fixed lookup; NaN values
    # would silently propagate through the soft-prior MSE-style application.
    lane_curvature_pca = np.where(
        np.isfinite(lane_curvature_pca),
        lane_curvature_pca,
        np.float16(0.0),
    ).astype(np.float16)

    if metadata is None:
        metadata = {}
    # Provide the canonical metadata fields that validate_codebook checks.
    # Scales chosen to match the operative dequantization scale; the actual
    # numerical magnitude does not affect inflate-side compliance because
    # the soft prior is additive + small (apply_soft_prior strength=1.0
    # in inflate.py applies the codebook as a per-pixel correction).
    canonical_meta: dict[str, object] = {
        "road_plane_scale": float(metadata.get("road_plane_scale", 1.0)),
        "sky_horizon_scale": float(metadata.get("sky_horizon_scale", 1.0)),
        "vehicle_scale": float(metadata.get("vehicle_scale", 1.0)),
        "dataset_provenance": str(
            metadata.get("dataset_provenance", "procedural_pcg64_seed_derived")
        ),
        "distillation_version": str(
            metadata.get("distillation_version", "procedural_v1")
        ),
        "license_tags": list(
            metadata.get("license_tags", ["MIT", "procedural_no_dataset_dependency"])
        ),
    }
    # Preserve any additional caller-supplied metadata (without overriding
    # the required keys above).
    for key, value in metadata.items():
        if key not in canonical_meta:
            canonical_meta[key] = value

    return DashcamCodebook(
        road_plane_basis=road_plane_basis,
        sky_horizon_profile=sky_horizon_profile,
        lane_curvature_pca=lane_curvature_pca,
        vehicle_appearance_basis=vehicle_appearance_basis,
        metadata=canonical_meta,
    )


def parse_archive_procedural_aware(archive_bytes: bytes):
    """Parse a DP1 archive with procedural-codebook variant detection.

    Sister of :func:`tac.substrates.pretrained_driving_prior.archive.parse_archive`
    (canonical Comma2k19-distilled path) extended with procedural-variant
    routing. When the meta_blob declares ``procedural_codebook_variant_active=True``,
    the codebook section bytes are interpreted as ``brotli(seed_bytes)`` and
    the 4 canonical codebook arrays are re-derived via
    :func:`derive_dashcam_codebook_from_seed`. Otherwise delegates to the
    canonical :func:`parse_archive`.

    Args:
        archive_bytes: DP1 archive bytes.

    Returns:
        DrivingPriorArchive with codebook either re-derived from seed
        (procedural variant) or parsed from Comma2k19-distilled blob
        (canonical variant). Renderer / residual / meta sections are
        identical between paths.

    Raises:
        ValueError: malformed header / truncated sections / unparseable
            meta_blob.
    """
    import json
    import struct

    import brotli  # type: ignore[import-not-found]

    from tac.substrates.pretrained_driving_prior.archive import (
        DP1_HEADER_FMT,
        DP1_HEADER_SIZE,
        DP1_MAGIC,
        DP1_SCHEMA_VERSION,
        DrivingPriorArchive,
        _deserialize_state_dict,
        parse_archive,
    )

    if len(archive_bytes) < DP1_HEADER_SIZE:
        raise ValueError(
            f"DP1 archive too short for header: "
            f"{len(archive_bytes)} < {DP1_HEADER_SIZE}"
        )
    (
        magic,
        version,
        num_pairs,
        out_h,
        out_w,
        per_pair_bytes,
        codebook_len,
        renderer_len,
        residual_len,
        meta_len,
    ) = struct.unpack(DP1_HEADER_FMT, archive_bytes[:DP1_HEADER_SIZE])
    if magic != DP1_MAGIC:
        raise ValueError(
            f"DP1 archive magic mismatch: {magic!r} != {DP1_MAGIC!r}"
        )
    if version != DP1_SCHEMA_VERSION:
        raise ValueError(
            f"DP1 schema version {version} != expected {DP1_SCHEMA_VERSION}"
        )

    expected_total = (
        DP1_HEADER_SIZE + codebook_len + renderer_len + residual_len + meta_len
    )
    if len(archive_bytes) < expected_total:
        raise ValueError(
            f"DP1 archive truncated: have {len(archive_bytes)} bytes, "
            f"expected {expected_total}"
        )

    meta_start = (
        DP1_HEADER_SIZE + codebook_len + renderer_len + residual_len
    )
    meta_blob = archive_bytes[meta_start : meta_start + meta_len]
    try:
        meta = json.loads(meta_blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Malformed meta — fall back to canonical parser (will raise).
        return parse_archive(archive_bytes)

    if not is_procedural_codebook_variant_archive(meta):
        return parse_archive(archive_bytes)

    # Procedural variant: re-derive codebook from seed.
    seed_hex = meta.get("procedural_codebook_seed_hex")
    if not isinstance(seed_hex, str) or not seed_hex:
        raise ValueError(
            "procedural_codebook_variant_active=True but "
            "procedural_codebook_seed_hex is missing or empty"
        )
    try:
        seed_bytes = bytes.fromhex(seed_hex)
    except ValueError as exc:
        raise ValueError(
            f"procedural_codebook_seed_hex is not valid hex: {exc}"
        ) from exc
    generator_kind = str(
        meta.get("procedural_codebook_generator_kind", "pcg64")
    )

    # The codebook section bytes are brotli(seed_bytes). We do not need to
    # decompress them at inflate (we already have the seed from meta); we
    # delegate to derive_dashcam_codebook_from_seed which produces the
    # canonical 4-array DashcamCodebook directly.
    codebook = derive_dashcam_codebook_from_seed(
        seed_bytes=seed_bytes,
        generator_kind=generator_kind,
        metadata={
            "license_tags": list(
                meta.get("license_tags", ["MIT", "procedural_no_dataset_dependency"])
            ),
            "dataset_provenance": str(
                meta.get(
                    "dataset_provenance", "procedural_pcg64_seed_derived"
                )
            ),
            "distillation_version": str(
                meta.get("distillation_version", "procedural_v1")
            ),
        },
    )

    renderer_start = DP1_HEADER_SIZE + codebook_len
    renderer_blob = archive_bytes[
        renderer_start : renderer_start + renderer_len
    ]
    residual_blob_start = renderer_start + renderer_len
    residual_blob = archive_bytes[
        residual_blob_start : residual_blob_start + residual_len
    ]
    renderer_state_dict = _deserialize_state_dict(renderer_blob)
    per_pair_residual = brotli.decompress(residual_blob)
    if len(per_pair_residual) != num_pairs * per_pair_bytes:
        raise ValueError(
            f"DP1 per-pair residual length {len(per_pair_residual)} != "
            f"num_pairs * per_pair_bytes = {num_pairs * per_pair_bytes}"
        )

    return DrivingPriorArchive(
        codebook=codebook,
        renderer_state_dict=renderer_state_dict,
        per_pair_residual=per_pair_residual,
        meta=meta,
        schema_version=version,
        num_pairs=num_pairs,
        output_height=out_h,
        output_width=out_w,
        per_pair_bytes=per_pair_bytes,
    )
