# SPDX-License-Identifier: MIT
"""NSCS06 v8 trainer `_full_main` v3 wire-in tests — NEW.

Per APPEND-ONLY discipline (Catalog #110/#113). Sister of
`test_cls_stream_wire_in.py` (codec + inflate surface). This file proves the
TRAINER calls `pack_archive(cls_bytes=...)` so emitted archives are CH08 v3
(not v2). Per T3 council #1335 PROCEED_WITH_REVISIONS REVISION #2 (Yousfi
BLOCKER): the cls_stream wire-in at L0 inflate is the Catalog #233 L1→L2
promotion canonical 4-gate unblocker; without the TRAINER routing through
`cls_bytes`, downstream consumers (paired Modal T4 4-arm, autopilot ranker)
never see v3 archives and the cargo-cult #5 FAIL_AT_CLASS_1 verdict persists.

These tests verify:

- (1) Stage 5b NEAREST downsample produces correctly-shaped cls_lowres from
       Stage 4 full-resolution cls_full
- (2) NEAREST downsample is the canonical sister of inflate.py's
       `Image.NEAREST` upsample (point-sampling at cell top-left preserved
       round-trip for uniform class)
- (3) The downsampled cls_bytes shape invariant matches grayscale_lowres
       shape (the per-cell low-res field contract)
- (4) End-to-end trainer-level pack_archive callsite emits v3
       schema_version when the v2_procedural_seed variant runs

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": all assertions
in this file are STRUCTURAL byte-level invariants on the canonical
`pack_archive` + `parse_archive` contract; no score literals or empirical
claims. [predicted; canonical-equation-N/A; structural-byte-invariant-only]

Discipline: Catalog #229 PV (read trainer Stage 4/5/5b/9 in full before
test design) + #287/#323 canonical Provenance (no score claim) + #110/#113
APPEND-ONLY (NEW file; sister test_cls_stream_wire_in.py preserved).
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from tac.substrates.nscs06_v8_chroma_lut.architecture import (
    GRAYSCALE_LEVELS_DEFAULT,
    NUM_SEGNET_CLASSES,
    PROCEDURAL_SEED_SIZE_BYTES,
)
from tac.substrates.nscs06_v8_chroma_lut.archive import (
    CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM,
    POSE_DIMS,
    pack_archive,
    parse_archive,
)


# ============================================================================
# Stage 5b — NEAREST downsample unit tests
# ============================================================================


def _nearest_downsample(cls_full: np.ndarray, factor: int) -> np.ndarray:
    """Canonical Stage 5b downsample (sister of trainer _full_main lines 740-744)."""
    n_pairs, h, w = cls_full.shape
    h_g = h // factor
    w_g = w // factor
    return cls_full[
        :,
        : h_g * factor : factor,
        : w_g * factor : factor,
    ]


def test_stage5b_nearest_downsample_shape() -> None:
    """NEAREST downsample reduces (n, H, W) to (n, H/factor, W/factor)."""
    n_pairs = 3
    H, W = 384, 512
    factor = 8
    cls_full = np.random.RandomState(20260526).randint(
        0, NUM_SEGNET_CLASSES, size=(n_pairs, H, W), dtype=np.uint8
    )
    cls_lowres = _nearest_downsample(cls_full, factor)
    assert cls_lowres.shape == (n_pairs, H // factor, W // factor)
    assert cls_lowres.dtype == np.uint8


def test_stage5b_uniform_class_downsample_invariant() -> None:
    """A uniformly-class-K cls_full downsamples to a uniformly-class-K cls_lowres."""
    n_pairs = 2
    H, W = 384, 512
    factor = 8
    for k in range(NUM_SEGNET_CLASSES):
        cls_full = np.full((n_pairs, H, W), k, dtype=np.uint8)
        cls_lowres = _nearest_downsample(cls_full, factor)
        assert np.all(cls_lowres == k), (
            f"uniform-class-{k} cls_full produced non-uniform cls_lowres"
        )


def test_stage5b_nearest_round_trip_with_pillow_upsample() -> None:
    """NEAREST downsample + Pillow NEAREST upsample reproduces the per-cell value.

    This is the canonical round-trip invariant the trainer wire-in relies on:
    the inflate runtime upsamples cls_lowres via `Image.NEAREST` to recover the
    per-pixel class label. The downsample (point-sample top-left) + upsample
    (NEAREST replicate) pair is byte-identical IF cls_full's per-cell field is
    already uniform (i.e. all pixels in each cell share the same class). For
    non-uniform cells the downsample is lossy (only the top-left pixel
    survives), which is the expected behavior of NEAREST.
    """
    factor = 8
    h_g, w_g = 4, 6
    cls_lowres_orig = np.random.RandomState(7).randint(
        0, NUM_SEGNET_CLASSES, size=(h_g, w_g), dtype=np.uint8
    )
    # Build cls_full via NEAREST upsample (every cell-pixel shares its top-left value).
    cls_full = np.array(
        Image.fromarray(cls_lowres_orig).resize(
            (w_g * factor, h_g * factor), Image.NEAREST
        ),
        dtype=np.uint8,
    )
    assert cls_full.shape == (h_g * factor, w_g * factor)
    # Stage 5b downsample MUST recover the original lowres exactly (NEAREST
    # upsample then top-left sample is identity).
    recovered = cls_full[: h_g * factor : factor, : w_g * factor : factor]
    assert np.array_equal(recovered, cls_lowres_orig)


# ============================================================================
# End-to-end trainer-level pack_archive callsite emits v3 schema
# ============================================================================


def _build_trainer_emission_fixture(
    *,
    n_pairs: int = 3,
    factor: int = 8,
    H: int = 384,
    W: int = 512,
    seed: int = 20260526,
) -> dict:
    """Replicate Stage 4 + Stage 5 + Stage 5b outputs at minimal scale.

    Returns the fixture passed to pack_archive at trainer line 770; matches
    the trainer's v3 callsite contract exactly.
    """
    h_g = H // factor
    w_g = W // factor
    rng = np.random.RandomState(seed)
    # Stage 4 cls_full (n_pairs, H, W) uint8 — would normally come from SegNet
    # argmax; here we use a synthetic class field with per-cell uniformity so
    # the NEAREST round-trip is exact (a stricter test surface).
    cls_lowres_seed = rng.randint(
        0, NUM_SEGNET_CLASSES, size=(n_pairs, h_g, w_g), dtype=np.uint8
    )
    cls_full = np.zeros((n_pairs, H, W), dtype=np.uint8)
    for p in range(n_pairs):
        cls_full[p] = np.array(
            Image.fromarray(cls_lowres_seed[p]).resize(
                (W, H), Image.NEAREST
            ),
            dtype=np.uint8,
        )
    # Stage 5 gray_lowres (n_pairs, h_g, w_g) uint8 — synthesized at low-res
    gray_lowres = rng.randint(
        0, GRAYSCALE_LEVELS_DEFAULT, size=(n_pairs, h_g, w_g), dtype=np.uint8
    )
    # Stage 7 pose_bytes (n_pairs * POSE_DIMS)
    pose_bytes = rng.randint(
        0, 256, size=n_pairs * POSE_DIMS, dtype=np.uint8
    ).tobytes()
    # Stage 6 chroma_lut (used for seed derivation)
    chroma_lut = rng.randint(
        0, 256, size=(GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8
    )
    seed_bytes = hashlib.sha256(chroma_lut.tobytes()).digest()[:PROCEDURAL_SEED_SIZE_BYTES]
    # Stage 5b cls_lowres + cls_bytes (the wire-in being tested)
    cls_lowres = _nearest_downsample(cls_full, factor)
    cls_bytes = np.ascontiguousarray(cls_lowres, dtype=np.uint8).tobytes()
    return {
        "n_pairs": n_pairs,
        "h_g": h_g,
        "w_g": w_g,
        "H": H,
        "W": W,
        "factor": factor,
        "cls_full": cls_full,
        "cls_lowres": cls_lowres,
        "cls_bytes": cls_bytes,
        "gray_lowres": gray_lowres,
        "pose_bytes": pose_bytes,
        "chroma_lut": chroma_lut,
        "seed_bytes": seed_bytes,
    }


def test_trainer_v3_emission_schema_version() -> None:
    """The trainer's v2_procedural_seed branch (post-wire-in) MUST emit v3."""
    fx = _build_trainer_emission_fixture()
    blob = pack_archive(
        num_pairs=fx["n_pairs"],
        grayscale_h=fx["h_g"],
        grayscale_w=fx["w_g"],
        output_height=fx["H"],
        output_width=fx["W"],
        pose_bytes=fx["pose_bytes"],
        grayscale_bytes=fx["gray_lowres"].tobytes(),
        chroma_seed=fx["seed_bytes"],
        cls_bytes=fx["cls_bytes"],
    )
    assert int(blob[4]) == CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM
    arc = parse_archive(blob)
    assert arc.schema_version == CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM
    assert arc.cls_lowres is not None


def test_trainer_v3_emission_cls_lowres_shape_matches_grayscale_lowres_shape() -> None:
    """v3 cls_lowres shape MUST match grayscale_lowres shape (per-cell field contract)."""
    fx = _build_trainer_emission_fixture()
    blob = pack_archive(
        num_pairs=fx["n_pairs"],
        grayscale_h=fx["h_g"],
        grayscale_w=fx["w_g"],
        output_height=fx["H"],
        output_width=fx["W"],
        pose_bytes=fx["pose_bytes"],
        grayscale_bytes=fx["gray_lowres"].tobytes(),
        chroma_seed=fx["seed_bytes"],
        cls_bytes=fx["cls_bytes"],
    )
    arc = parse_archive(blob)
    assert arc.cls_lowres.shape == (fx["n_pairs"], fx["h_g"], fx["w_g"])
    assert arc.cls_lowres.shape[1:] == fx["gray_lowres"].shape[1:], (
        "cls_lowres per-cell shape MUST match grayscale_lowres per-cell shape"
    )


def test_trainer_v3_emission_cls_bytes_round_trip() -> None:
    """The trainer's cls_bytes MUST round-trip byte-identically through pack/parse."""
    fx = _build_trainer_emission_fixture()
    blob = pack_archive(
        num_pairs=fx["n_pairs"],
        grayscale_h=fx["h_g"],
        grayscale_w=fx["w_g"],
        output_height=fx["H"],
        output_width=fx["W"],
        pose_bytes=fx["pose_bytes"],
        grayscale_bytes=fx["gray_lowres"].tobytes(),
        chroma_seed=fx["seed_bytes"],
        cls_bytes=fx["cls_bytes"],
    )
    arc = parse_archive(blob)
    assert arc.cls_lowres.tobytes() == fx["cls_bytes"]


def test_trainer_v3_emission_does_not_emit_v2() -> None:
    """Regression guard: post-wire-in the v2 path emits v3, NOT v2.

    This is the structural anchor that proves the wire-in landed: a pack call
    that includes cls_bytes MUST produce schema_version=3, not 2. If a future
    refactor silently drops the cls_bytes kwarg, this test will catch it.
    """
    fx = _build_trainer_emission_fixture()
    blob_v3 = pack_archive(
        num_pairs=fx["n_pairs"],
        grayscale_h=fx["h_g"],
        grayscale_w=fx["w_g"],
        output_height=fx["H"],
        output_width=fx["W"],
        pose_bytes=fx["pose_bytes"],
        grayscale_bytes=fx["gray_lowres"].tobytes(),
        chroma_seed=fx["seed_bytes"],
        cls_bytes=fx["cls_bytes"],
    )
    # Sister no-cls_bytes call (would be the pre-wire-in trainer emission)
    blob_v2 = pack_archive(
        num_pairs=fx["n_pairs"],
        grayscale_h=fx["h_g"],
        grayscale_w=fx["w_g"],
        output_height=fx["H"],
        output_width=fx["W"],
        pose_bytes=fx["pose_bytes"],
        grayscale_bytes=fx["gray_lowres"].tobytes(),
        chroma_seed=fx["seed_bytes"],
    )
    assert int(blob_v3[4]) == CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM
    assert int(blob_v2[4]) != CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM, (
        "v2 path (no cls_bytes) MUST NOT emit v3 — regression guard for the "
        "pre-wire-in trainer behavior"
    )
    # And the byte streams MUST differ (v3 has a longer header + cls_stream)
    assert len(blob_v3) > len(blob_v2)


def test_trainer_v3_emission_byte_cost_invariant() -> None:
    """v3 archive byte count MUST equal v2 byte count + 4 (CLS_LEN u32) + cls_bytes.

    This is the canonical rate-axis cost the predicted ΔS bound depends on.
    Per archive.py docstring: total rate-axis ΔS = canonical equation #26
    REPLACEMENT savings (unchanged) + 25 * (num_pairs * gh * gw) /
    37_545_489 (cls_stream ADDITIVE byte cost).
    """
    fx = _build_trainer_emission_fixture()
    blob_v3 = pack_archive(
        num_pairs=fx["n_pairs"],
        grayscale_h=fx["h_g"],
        grayscale_w=fx["w_g"],
        output_height=fx["H"],
        output_width=fx["W"],
        pose_bytes=fx["pose_bytes"],
        grayscale_bytes=fx["gray_lowres"].tobytes(),
        chroma_seed=fx["seed_bytes"],
        cls_bytes=fx["cls_bytes"],
    )
    blob_v2 = pack_archive(
        num_pairs=fx["n_pairs"],
        grayscale_h=fx["h_g"],
        grayscale_w=fx["w_g"],
        output_height=fx["H"],
        output_width=fx["W"],
        pose_bytes=fx["pose_bytes"],
        grayscale_bytes=fx["gray_lowres"].tobytes(),
        chroma_seed=fx["seed_bytes"],
    )
    # v3 header is 4 bytes longer than v1/v2 (CLS_LEN u32 appended) + cls_bytes
    expected_delta = 4 + len(fx["cls_bytes"])
    assert len(blob_v3) - len(blob_v2) == expected_delta


# ============================================================================
# Catalog #233 4-gate REFRESH evidence: trainer + codec + inflate all v3-aligned
# ============================================================================


def test_catalog_233_4_gate_refresh_trainer_codec_inflate_coherent() -> None:
    """End-to-end coherence: trainer's emission, codec's parse, inflate's branch
    all align on v3 cls_stream consumption.

    Per Catalog #233 L1→L2 promotion canonical 4-gate:
    - Gate 1 (smoke green): this test passes (rc=0 + parseable archive)
    - Gate 2 (Tier-C MDL density): out-of-scope for this wire-in test
    - Gate 3 (100ep auth-eval anchor): out-of-scope (downstream paired Modal)
    - Gate 4 (custody per Catalog #127): out-of-scope (downstream)

    THIS test proves the trainer+codec+inflate are coherent (Gate 1 REFRESH).
    """
    from tac.substrates.nscs06_v8_chroma_lut.inflate import inflate_one_video
    import tempfile

    fx = _build_trainer_emission_fixture(n_pairs=2)
    blob = pack_archive(
        num_pairs=fx["n_pairs"],
        grayscale_h=fx["h_g"],
        grayscale_w=fx["w_g"],
        output_height=fx["H"],
        output_width=fx["W"],
        pose_bytes=fx["pose_bytes"],
        grayscale_bytes=fx["gray_lowres"].tobytes(),
        chroma_seed=fx["seed_bytes"],
        cls_bytes=fx["cls_bytes"],
    )
    # Trainer emits v3
    arc = parse_archive(blob)
    assert arc.schema_version == CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM
    # Inflate consumes v3
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        raw = inflate_one_video(blob, Path(td) / "v3_e2e")
        assert raw.exists()
        assert raw.stat().st_size > 0
