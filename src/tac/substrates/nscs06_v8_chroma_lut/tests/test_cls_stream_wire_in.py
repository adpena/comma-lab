# SPDX-License-Identifier: MIT
"""CH08 v3 cls_stream wire-in tests — NEW (APPEND-ONLY per Catalog #110/#113).

Per T3 council #1335 PROCEED_WITH_REVISIONS REVISION #2 (Yousfi BLOCKER):
the cls_stream wire-in at L0 inflate is the L1→L2 promotion canonical 4-gate
unblocker per Catalog #233. These tests verify the wire-in lands cleanly:

- (1) v3 pack/parse roundtrip is byte-stable
- (2) v3 archive carries cls_lowres on parse
- (3) v1/v2 archives carry cls_lowres=None (backward compat)
- (4) inflate consumes cls_lowres at the cargo-cult #5 site
- (5) v3 vs v2 inflate produces DIFFERENT frames given identical seed +
       distinct cls_stream (the structural proof that cargo-cult #5 is
       UNWOUND)

Sister of `tests/test_path_3_c_prime_cargo_cult_unwinds.py` cargo-cult #5
verdict-dataclass test_pattern (per `distinguishing_feature_smoke.py`
canonical helper). The PASS_PER_CLASS verdict transition is the runtime
witness; this test file is the per-archive-format wire-in proof.

Discipline: Catalog #229 PV (sister tests inspected before wiring) +
#287/#323 canonical Provenance (no score claim asserted) + #110/#113
APPEND-ONLY (NEW file; sister tests preserved).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.substrates.nscs06_v8_chroma_lut.architecture import (
    CHROMA_LUT_BYTES_DEFAULT,
    GRAYSCALE_LEVELS_DEFAULT,
    NUM_SEGNET_CLASSES,
    PROCEDURAL_SEED_SIZE_BYTES,
)
from tac.substrates.nscs06_v8_chroma_lut.archive import (
    CH08_HEADER_SIZE,
    CH08_HEADER_SIZE_V3,
    CH08_SCHEMA_VERSION_INLINE_LUT,
    CH08_SCHEMA_VERSION_PROCEDURAL_SEED,
    CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM,
    POSE_DIMS,
    pack_archive,
    parse_archive,
)
from tac.substrates.nscs06_v8_chroma_lut.inflate import inflate_one_video


def _common_fixture(
    *, num_pairs: int = 4, gh: int = 4, gw: int = 6
) -> dict:
    """Build the minimal fixture shared by all cls_stream wire-in tests."""
    pose_bytes = bytes(np.random.RandomState(7).randint(
        0, 256, num_pairs * POSE_DIMS, dtype=np.uint8
    ))
    grayscale_bytes = bytes(np.random.RandomState(8).randint(
        0, GRAYSCALE_LEVELS_DEFAULT, num_pairs * gh * gw, dtype=np.uint8
    ))
    seed_bytes = bytes(np.random.RandomState(9).randint(
        0, 256, PROCEDURAL_SEED_SIZE_BYTES, dtype=np.uint8
    ))
    return {
        "num_pairs": num_pairs,
        "grayscale_h": gh,
        "grayscale_w": gw,
        "output_height": 12,
        "output_width": 18,
        "pose_bytes": pose_bytes,
        "grayscale_bytes": grayscale_bytes,
        "chroma_seed": seed_bytes,
    }


# ============================================================================
# Header layout invariants
# ============================================================================

def test_v3_header_size_invariant() -> None:
    """v3 header MUST be exactly 39 bytes (v1/v2 35 + appended CLS_LEN u32)."""
    assert CH08_HEADER_SIZE_V3 == 39
    assert CH08_HEADER_SIZE_V3 == CH08_HEADER_SIZE + 4


def test_v3_schema_version_constant() -> None:
    """v3 schema version constant MUST be 3 and DISTINCT from v1/v2."""
    assert CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM == 3
    assert CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM != (
        CH08_SCHEMA_VERSION_INLINE_LUT
    )
    assert CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM != (
        CH08_SCHEMA_VERSION_PROCEDURAL_SEED
    )


# ============================================================================
# pack_archive v3 cls_stream wiring
# ============================================================================

def test_pack_archive_v3_with_cls_bytes_produces_v3_schema() -> None:
    fixture = _common_fixture()
    n = fixture["num_pairs"] * fixture["grayscale_h"] * fixture["grayscale_w"]
    cls_bytes = bytes(np.random.RandomState(10).randint(
        0, NUM_SEGNET_CLASSES, n, dtype=np.uint8
    ))
    blob = pack_archive(**fixture, cls_bytes=cls_bytes)
    # Version byte at struct position 4 must be v3.
    assert int(blob[4]) == CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM


def test_pack_archive_v2_path_unchanged_when_cls_bytes_absent() -> None:
    """Backward compat: no cls_bytes => v2 PROCEDURAL_SEED schema selected."""
    fixture = _common_fixture()
    blob = pack_archive(**fixture)
    assert int(blob[4]) == CH08_SCHEMA_VERSION_PROCEDURAL_SEED


def test_pack_archive_v1_path_unchanged_when_inline_lut_supplied() -> None:
    """Backward compat: inline chroma_lut path selects v1 INLINE_LUT schema."""
    fixture = _common_fixture()
    del fixture["chroma_seed"]
    fixture["chroma_lut"] = np.random.RandomState(11).randint(
        0, 256, (GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8
    )
    blob = pack_archive(**fixture)
    assert int(blob[4]) == CH08_SCHEMA_VERSION_INLINE_LUT


def test_pack_archive_v3_rejects_cls_bytes_with_v1_inline_lut() -> None:
    """v3 + inline LUT is a forbidden combination (v3 implies procedural seed)."""
    fixture = _common_fixture()
    del fixture["chroma_seed"]
    fixture["chroma_lut"] = np.random.RandomState(12).randint(
        0, 256, (GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8
    )
    n = fixture["num_pairs"] * fixture["grayscale_h"] * fixture["grayscale_w"]
    cls_bytes = bytes(np.zeros(n, dtype=np.uint8))
    with pytest.raises(ValueError, match="cls_bytes supplied but schema_version"):
        pack_archive(**fixture, cls_bytes=cls_bytes)


def test_pack_archive_v3_rejects_wrong_cls_length() -> None:
    fixture = _common_fixture()
    bad_cls = bytes(np.zeros(1, dtype=np.uint8))
    with pytest.raises(ValueError, match="cls_bytes length"):
        pack_archive(**fixture, cls_bytes=bad_cls)


def test_pack_archive_v3_rejects_cls_label_at_or_above_num_classes() -> None:
    fixture = _common_fixture()
    n = fixture["num_pairs"] * fixture["grayscale_h"] * fixture["grayscale_w"]
    cls_bytes = bytes(np.full(n, NUM_SEGNET_CLASSES, dtype=np.uint8))
    blob = pack_archive(**fixture, cls_bytes=cls_bytes)
    with pytest.raises(ValueError, match="cls_stream label"):
        parse_archive(blob)


# ============================================================================
# parse_archive v3 cls_lowres propagation
# ============================================================================

def test_parse_archive_v3_returns_cls_lowres_shape() -> None:
    fixture = _common_fixture()
    n = fixture["num_pairs"] * fixture["grayscale_h"] * fixture["grayscale_w"]
    cls_bytes = bytes(np.random.RandomState(13).randint(
        0, NUM_SEGNET_CLASSES, n, dtype=np.uint8
    ))
    blob = pack_archive(**fixture, cls_bytes=cls_bytes)
    arc = parse_archive(blob)
    assert arc.cls_lowres is not None
    assert arc.cls_lowres.dtype == np.uint8
    assert arc.cls_lowres.shape == (
        fixture["num_pairs"],
        fixture["grayscale_h"],
        fixture["grayscale_w"],
    )


def test_parse_archive_v3_cls_lowres_byte_stable_roundtrip() -> None:
    fixture = _common_fixture()
    n = fixture["num_pairs"] * fixture["grayscale_h"] * fixture["grayscale_w"]
    cls_bytes = bytes(np.random.RandomState(14).randint(
        0, NUM_SEGNET_CLASSES, n, dtype=np.uint8
    ))
    blob = pack_archive(**fixture, cls_bytes=cls_bytes)
    arc = parse_archive(blob)
    assert arc.cls_lowres.tobytes() == cls_bytes


def test_parse_archive_v2_returns_cls_lowres_none() -> None:
    """Backward compat: v2 archives carry cls_lowres=None."""
    fixture = _common_fixture()
    blob = pack_archive(**fixture)
    arc = parse_archive(blob)
    assert arc.cls_lowres is None
    assert arc.schema_version == CH08_SCHEMA_VERSION_PROCEDURAL_SEED


def test_parse_archive_v1_returns_cls_lowres_none() -> None:
    """Backward compat: v1 archives carry cls_lowres=None."""
    fixture = _common_fixture()
    del fixture["chroma_seed"]
    fixture["chroma_lut"] = np.random.RandomState(15).randint(
        0, 256, (GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8
    )
    blob = pack_archive(**fixture)
    arc = parse_archive(blob)
    assert arc.cls_lowres is None
    assert arc.schema_version == CH08_SCHEMA_VERSION_INLINE_LUT


def test_parse_archive_v3_byte_stable_pack_parse_pack() -> None:
    """Byte-stability invariant: pack -> parse -> pack reproduces identical bytes."""
    fixture = _common_fixture()
    n = fixture["num_pairs"] * fixture["grayscale_h"] * fixture["grayscale_w"]
    cls_bytes = bytes(np.random.RandomState(16).randint(
        0, NUM_SEGNET_CLASSES, n, dtype=np.uint8
    ))
    blob1 = pack_archive(**fixture, cls_bytes=cls_bytes)
    arc = parse_archive(blob1)
    # Re-pack from parsed archive's fields.
    blob2 = pack_archive(
        num_pairs=arc.num_pairs,
        grayscale_h=arc.grayscale_h,
        grayscale_w=arc.grayscale_w,
        output_height=arc.output_height,
        output_width=arc.output_width,
        pose_bytes=arc.pose_bytes,
        grayscale_bytes=arc.grayscale_bytes,
        pose_quant_scale=arc.pose_quant_scale,
        chroma_seed=arc.chroma_seed,
        generator_kind=arc.generator_kind,
        cls_bytes=arc.cls_lowres.tobytes(),
    )
    assert blob1 == blob2


# ============================================================================
# inflate consumes cls_lowres at the cargo-cult #5 site
# ============================================================================

def test_inflate_v3_vs_v2_produces_different_frames_proves_cls_consumption(
    tmp_path: Path,
) -> None:
    """The structural proof that cargo-cult #5 is UNWOUND.

    Build a v2 archive (cls=0 uniform at inflate) and a v3 archive with the
    SAME seed + grayscale + pose but a NON-UNIFORM cls_stream (e.g. half
    the cells = class 0, half = class 1). The inflated frames MUST differ
    because cls_full at inflate is sourced from cls_lowres (v3 path) vs
    np.zeros (v2 legacy path).
    """
    fixture = _common_fixture()
    n = fixture["num_pairs"] * fixture["grayscale_h"] * fixture["grayscale_w"]
    # cls_stream: alternating 0/1 per cell so half use class 0 anchor (matches
    # v2 cls=0 uniform) and half use class 1 anchor (DIFFERENT from v2).
    cls_array = np.zeros(n, dtype=np.uint8)
    cls_array[1::2] = 1
    cls_bytes = bytes(cls_array)

    blob_v2 = pack_archive(**fixture)
    blob_v3 = pack_archive(**fixture, cls_bytes=cls_bytes)

    arc_v2 = parse_archive(blob_v2)
    arc_v3 = parse_archive(blob_v3)
    assert arc_v2.schema_version == CH08_SCHEMA_VERSION_PROCEDURAL_SEED
    assert arc_v3.schema_version == CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM
    assert arc_v2.cls_lowres is None
    assert arc_v3.cls_lowres is not None

    out_v2 = tmp_path / "v2"
    out_v3 = tmp_path / "v3"
    raw_v2 = inflate_one_video(blob_v2, out_v2)
    raw_v3 = inflate_one_video(blob_v3, out_v3)
    bytes_v2 = raw_v2.read_bytes()
    bytes_v3 = raw_v3.read_bytes()

    # CRITICAL PROOF: v3 frames MUST differ from v2 frames.
    # If cargo-cult #5 were still active, both would use cls=0 uniform and
    # produce byte-identical frames despite the different schema_version.
    assert bytes_v2 != bytes_v3, (
        "v3 inflate did NOT consume cls_stream — cargo-cult #5 STILL ACTIVE"
    )


def test_inflate_v2_legacy_cls_zero_uniform_preserved(tmp_path: Path) -> None:
    """Backward compat: v2 inflate MUST still use cls=0 uniform (legacy path)."""
    fixture = _common_fixture()
    blob_v2 = pack_archive(**fixture)
    arc_v2 = parse_archive(blob_v2)
    assert arc_v2.cls_lowres is None
    # v2 inflate runs without raising; produces a non-empty raw file.
    raw = inflate_one_video(blob_v2, tmp_path / "v2_legacy")
    assert raw.exists()
    assert raw.stat().st_size > 0


def test_inflate_v3_with_uniform_class_matches_v2(tmp_path: Path) -> None:
    """When cls_stream is uniformly 0, v3 MUST produce byte-identical output to v2.

    This invariant proves the wire-in does NOT introduce a bug at the boundary:
    given identical SEMANTIC cls assignment (all-zero), the byte output MUST
    be identical regardless of whether the class labels came from the cls_stream
    (v3 path) or from np.zeros (v2 legacy path).
    """
    fixture = _common_fixture()
    n = fixture["num_pairs"] * fixture["grayscale_h"] * fixture["grayscale_w"]
    cls_bytes_zero = bytes(np.zeros(n, dtype=np.uint8))

    blob_v2 = pack_archive(**fixture)
    blob_v3 = pack_archive(**fixture, cls_bytes=cls_bytes_zero)

    raw_v2 = inflate_one_video(blob_v2, tmp_path / "v2")
    raw_v3 = inflate_one_video(blob_v3, tmp_path / "v3_uniform_zero")

    assert raw_v2.read_bytes() == raw_v3.read_bytes(), (
        "v3 with cls=0 uniform does NOT match v2 — wire-in introduced "
        "spurious difference at the cls=0 boundary"
    )


# ============================================================================
# Catalog #233 4-gate evidence: parser-section-manifest-consistent for v3
# ============================================================================

def test_v3_archive_total_size_invariant() -> None:
    """v3 archive size = header (39) + seed (32) + pose + grayscale + cls."""
    fixture = _common_fixture()
    n = fixture["num_pairs"] * fixture["grayscale_h"] * fixture["grayscale_w"]
    cls_bytes = bytes(np.random.RandomState(17).randint(
        0, NUM_SEGNET_CLASSES, n, dtype=np.uint8
    ))
    blob = pack_archive(**fixture, cls_bytes=cls_bytes)
    expected = (
        CH08_HEADER_SIZE_V3
        + PROCEDURAL_SEED_SIZE_BYTES
        + len(fixture["pose_bytes"])
        + len(fixture["grayscale_bytes"])
        + n
    )
    assert len(blob) == expected
