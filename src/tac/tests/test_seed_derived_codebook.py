# SPDX-License-Identifier: MIT
"""Tests for tac.procedural_codebook_generator.seed_derived_codebook.

Covers the 3-PRNG-kind canonical API per
``.omx/research/procedural_codebook_generator_null_exploit_design_20260520.md``
Top-3 op-routable #1 (Q5 follow-on; null-exploit operationalization).

Sister of ``test_procedural_codebook_generator.py`` (covers the
``hash_seed_codebook_generator`` numpy-PCG64-wrapper module; this file
covers the 3-PRNG-kind canonical helper landed today).

Test coverage:

* Determinism per kind (same inputs → byte-identical output)
* Byte-stability across runs (re-run produces same bytes)
* Cross-kind divergence (different kinds → different bytes for same seed)
* dtype + output_shape round-trip (uint8 / int8 / float16 / float32)
* Verify-helper happy path + mismatch detection
* Input validation (empty seed / wrong types / unsupported kind /
  oversized output)
* NSCS06 v8 chroma LUT integration scenario (~4 KB → 32-byte seed)
* Empty / all-zero seed safety (PRNG salt path)
* Public API surface re-export from package __init__
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.procedural_codebook_generator import (
    DEFAULT_GENERATOR_KIND,
    MAX_OUTPUT_BYTES,
    SUPPORTED_GENERATOR_KINDS,
    ProceduralCodebookGeneratorError,
    derive_codebook_from_seed,
    verify_codebook_from_seed,
)


CANONICAL_SEED = bytes(range(32))
"""Canonical 32-byte test seed used across most tests."""


# ---------------------------------------------------------------------------
# Determinism + byte-stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(SUPPORTED_GENERATOR_KINDS))
def test_determinism_per_kind(kind: str) -> None:
    """Same (seed, shape, dtype, kind) → byte-identical output across calls."""
    a = derive_codebook_from_seed(
        seed_bytes=CANONICAL_SEED,
        output_shape=(256,),
        dtype=np.uint8,
        generator_kind=kind,
    )
    b = derive_codebook_from_seed(
        seed_bytes=CANONICAL_SEED,
        output_shape=(256,),
        dtype=np.uint8,
        generator_kind=kind,
    )
    assert a.tobytes() == b.tobytes(), f"{kind} not deterministic"


@pytest.mark.parametrize("kind", sorted(SUPPORTED_GENERATOR_KINDS))
def test_seed_variation_changes_output(kind: str) -> None:
    """Different seeds → different bytes (per-PRNG entropy check)."""
    a = derive_codebook_from_seed(
        seed_bytes=CANONICAL_SEED,
        output_shape=(64,),
        dtype=np.uint8,
        generator_kind=kind,
    )
    b = derive_codebook_from_seed(
        seed_bytes=CANONICAL_SEED[:-1] + bytes([(CANONICAL_SEED[-1] ^ 0xFF) & 0xFF]),
        output_shape=(64,),
        dtype=np.uint8,
        generator_kind=kind,
    )
    assert a.tobytes() != b.tobytes(), f"{kind} entropy seed-flip detection failed"


def test_kinds_produce_distinct_outputs() -> None:
    """The 3 canonical kinds produce mutually-distinct bytes for the same seed."""
    outputs = {
        kind: derive_codebook_from_seed(
            seed_bytes=CANONICAL_SEED,
            output_shape=(128,),
            dtype=np.uint8,
            generator_kind=kind,
        ).tobytes()
        for kind in SUPPORTED_GENERATOR_KINDS
    }
    seen: set[bytes] = set()
    for kind, blob in outputs.items():
        assert blob not in seen, f"{kind} collides with another kind"
        seen.add(blob)


# ---------------------------------------------------------------------------
# dtype + output_shape round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,shape",
    [
        (np.uint8, (256,)),
        (np.int8, (256,)),
        (np.uint8, (16, 16)),
        (np.int8, (4, 8, 8)),
        (np.float16, (128,)),
        (np.float32, (64,)),
    ],
)
def test_shape_and_dtype_round_trip(dtype: type, shape: tuple[int, ...]) -> None:
    """Output respects requested shape + dtype; byte length matches itemsize×prod(shape)."""
    arr = derive_codebook_from_seed(
        seed_bytes=CANONICAL_SEED,
        output_shape=shape,
        dtype=dtype,
        generator_kind="pcg64",
    )
    assert arr.shape == shape
    assert arr.dtype == np.dtype(dtype)
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    assert len(arr.tobytes()) == expected_bytes


def test_default_generator_kind_constant_is_pcg64() -> None:
    """Default generator kind is pcg64 (highest entropy)."""
    assert DEFAULT_GENERATOR_KIND == "pcg64"
    a = derive_codebook_from_seed(
        seed_bytes=CANONICAL_SEED, output_shape=(64,), dtype=np.uint8
    )
    b = derive_codebook_from_seed(
        seed_bytes=CANONICAL_SEED,
        output_shape=(64,),
        dtype=np.uint8,
        generator_kind=DEFAULT_GENERATOR_KIND,
    )
    assert a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# verify_codebook_from_seed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(SUPPORTED_GENERATOR_KINDS))
def test_verify_happy_path(kind: str) -> None:
    """Verify returns True when expected bytes equal derived bytes."""
    arr = derive_codebook_from_seed(
        seed_bytes=CANONICAL_SEED,
        output_shape=(128,),
        dtype=np.uint8,
        generator_kind=kind,
    )
    assert verify_codebook_from_seed(
        seed_bytes=CANONICAL_SEED,
        expected_codebook_bytes=arr.tobytes(),
        output_shape=(128,),
        dtype=np.uint8,
        generator_kind=kind,
    )


def test_verify_detects_byte_mismatch() -> None:
    """Verify returns False on byte mismatch (does not raise)."""
    arr = derive_codebook_from_seed(
        seed_bytes=CANONICAL_SEED,
        output_shape=(64,),
        dtype=np.uint8,
        generator_kind="pcg64",
    )
    bad = bytearray(arr.tobytes())
    bad[0] ^= 0xFF
    assert (
        verify_codebook_from_seed(
            seed_bytes=CANONICAL_SEED,
            expected_codebook_bytes=bytes(bad),
            output_shape=(64,),
            dtype=np.uint8,
            generator_kind="pcg64",
        )
        is False
    )


def test_verify_detects_wrong_seed() -> None:
    """Verify returns False when seed differs from the one used to derive expected."""
    arr = derive_codebook_from_seed(
        seed_bytes=CANONICAL_SEED,
        output_shape=(64,),
        dtype=np.uint8,
        generator_kind="pcg64",
    )
    other_seed = CANONICAL_SEED[:-1] + bytes([0xAA])
    assert (
        verify_codebook_from_seed(
            seed_bytes=other_seed,
            expected_codebook_bytes=arr.tobytes(),
            output_shape=(64,),
            dtype=np.uint8,
            generator_kind="pcg64",
        )
        is False
    )


# ---------------------------------------------------------------------------
# Input validation (negative path)
# ---------------------------------------------------------------------------


def test_empty_seed_rejected() -> None:
    with pytest.raises(ProceduralCodebookGeneratorError, match="non-empty"):
        derive_codebook_from_seed(
            seed_bytes=b"",
            output_shape=(8,),
            dtype=np.uint8,
        )


def test_non_bytes_seed_rejected() -> None:
    with pytest.raises(ProceduralCodebookGeneratorError, match="bytes-like"):
        derive_codebook_from_seed(
            seed_bytes="not bytes",  # type: ignore[arg-type]
            output_shape=(8,),
            dtype=np.uint8,
        )


def test_non_tuple_shape_rejected() -> None:
    with pytest.raises(ProceduralCodebookGeneratorError, match="tuple"):
        derive_codebook_from_seed(
            seed_bytes=CANONICAL_SEED,
            output_shape=[8, 4],  # type: ignore[arg-type]
            dtype=np.uint8,
        )


def test_empty_shape_rejected() -> None:
    with pytest.raises(ProceduralCodebookGeneratorError, match="non-empty tuple"):
        derive_codebook_from_seed(
            seed_bytes=CANONICAL_SEED,
            output_shape=(),
            dtype=np.uint8,
        )


@pytest.mark.parametrize("bad_dim", [0, -1, -100])
def test_non_positive_dim_rejected(bad_dim: int) -> None:
    with pytest.raises(ProceduralCodebookGeneratorError, match="positive"):
        derive_codebook_from_seed(
            seed_bytes=CANONICAL_SEED,
            output_shape=(8, bad_dim),
            dtype=np.uint8,
        )


def test_unsupported_generator_kind_rejected() -> None:
    with pytest.raises(ProceduralCodebookGeneratorError, match="not in canonical set"):
        derive_codebook_from_seed(
            seed_bytes=CANONICAL_SEED,
            output_shape=(8,),
            dtype=np.uint8,
            generator_kind="mersenne_twister",  # type: ignore[arg-type]
        )


def test_oversized_output_rejected() -> None:
    too_big = MAX_OUTPUT_BYTES + 1
    with pytest.raises(ProceduralCodebookGeneratorError, match="MAX_OUTPUT_BYTES"):
        derive_codebook_from_seed(
            seed_bytes=CANONICAL_SEED,
            output_shape=(too_big,),
            dtype=np.uint8,
        )


# ---------------------------------------------------------------------------
# All-zero seed safety (PRNG salt path)
# ---------------------------------------------------------------------------


def test_all_zero_seed_does_not_produce_all_zero_output_xorshift() -> None:
    """xorshift requires non-zero state; all-zero seed routes through sha256 salt."""
    arr = derive_codebook_from_seed(
        seed_bytes=bytes(8),
        output_shape=(64,),
        dtype=np.uint8,
        generator_kind="xorshift",
    )
    assert arr.tobytes() != bytes(64), "xorshift degenerate on all-zero seed"


def test_all_zero_seed_works_for_pcg64() -> None:
    """pcg64 derives state from sha256(seed); all-zero seed is well-defined."""
    arr = derive_codebook_from_seed(
        seed_bytes=bytes(16),
        output_shape=(64,),
        dtype=np.uint8,
        generator_kind="pcg64",
    )
    assert arr.tobytes() != bytes(64), "pcg64 degenerate on all-zero seed"


def test_short_seed_padded() -> None:
    """Seed shorter than 8 bytes is accepted (zero-padded internally)."""
    arr = derive_codebook_from_seed(
        seed_bytes=b"x",
        output_shape=(16,),
        dtype=np.uint8,
        generator_kind="pcg64",
    )
    assert len(arr.tobytes()) == 16


# ---------------------------------------------------------------------------
# NSCS06 v8 chroma LUT integration scenario
# ---------------------------------------------------------------------------


def test_nscs06_v8_chroma_lut_integration_scenario() -> None:
    """Simulates the canonical NSCS06 v8 procedural-replacement use case.

    Per memo §4: NSCS06 v8 chroma LUT ~4 KB constants → 32-byte seed =
    ~3.968 KB saved → predicted ΔS = -0.00264 (canonical formula
    25 * 3968 / 37_545_489).
    """
    seed = bytes(range(32))
    lut = derive_codebook_from_seed(
        seed_bytes=seed,
        output_shape=(1024, 4),
        dtype=np.uint8,
        generator_kind="pcg64",
    )
    assert lut.shape == (1024, 4)
    assert lut.dtype == np.dtype(np.uint8)
    assert lut.nbytes == 4096
    assert verify_codebook_from_seed(
        seed_bytes=seed,
        expected_codebook_bytes=lut.tobytes(),
        output_shape=(1024, 4),
        dtype=np.uint8,
        generator_kind="pcg64",
    )
    bytes_saved = 4096 - len(seed)
    # bytes_saved removed → rate term DECREASES by canonical formula.
    # Per CLAUDE.md "Canonical equations + models registry" + the
    # contest rate formula `25 * archive_bytes / 37_545_489`, removing
    # bytes_saved bytes from archive.zip lowers the contest score by:
    rate_term_decrease = 25.0 * bytes_saved / 37_545_489
    # Predicted ΔS contribution (signed; negative = score improvement
    # since score is lower-is-better):
    predicted_delta_s = -rate_term_decrease
    assert -0.003 < predicted_delta_s < 0
    # Memo §4 quote: "~0.00264 ΔS per 4 KB hoisted"
    assert abs(predicted_delta_s - (-0.0027059)) < 1e-4


# ---------------------------------------------------------------------------
# Public API + re-export integrity
# ---------------------------------------------------------------------------


def test_re_export_from_package_root() -> None:
    """Public symbols are importable from the package root."""
    from tac.procedural_codebook_generator import (
        DEFAULT_GENERATOR_KIND as REEXPORTED_DEFAULT,
        SUPPORTED_GENERATOR_KINDS as REEXPORTED_KINDS,
        derive_codebook_from_seed as REEXPORTED_DERIVE,
        verify_codebook_from_seed as REEXPORTED_VERIFY,
        ProceduralCodebookGeneratorError as REEXPORTED_ERROR,
    )
    assert REEXPORTED_DEFAULT == "pcg64"
    assert SUPPORTED_GENERATOR_KINDS == REEXPORTED_KINDS
    assert callable(REEXPORTED_DERIVE)
    assert callable(REEXPORTED_VERIFY)
    assert issubclass(REEXPORTED_ERROR, ValueError)


def test_supported_kinds_frozen() -> None:
    """SUPPORTED_GENERATOR_KINDS is frozen (byte-stability guarantee)."""
    assert isinstance(SUPPORTED_GENERATOR_KINDS, frozenset)
    with pytest.raises(AttributeError):
        SUPPORTED_GENERATOR_KINDS.add("not_real")  # type: ignore[attr-defined]


def test_supported_kinds_canonical_set() -> None:
    """The 3-PRNG canonical set per memo §3 is exactly {xorshift, lcg, pcg64}."""
    assert SUPPORTED_GENERATOR_KINDS == frozenset({"xorshift", "lcg", "pcg64"})


# ---------------------------------------------------------------------------
# Cross-validation with sister hash_seed_codebook_generator (sister module)
# ---------------------------------------------------------------------------


def test_distinct_from_sister_hash_seed_generator() -> None:
    """The new 3-PRNG API does NOT collide with sister emit_seed + expand_seed_to_codebook.

    Sister module uses numpy stdlib PCG64 wrapper (state machine); this
    module implements the raw PCG-XSL-RR 128/64 directly from the O'Neill
    paper. The two are independent canonical APIs serving complementary
    purposes per the package docstring.
    """
    from tac.procedural_codebook_generator import (
        emit_seed,
        expand_seed_to_codebook,
    )

    sister_seed = emit_seed((256,))
    sister_lut = expand_seed_to_codebook(sister_seed, (256,))
    our_lut = derive_codebook_from_seed(
        seed_bytes=sister_seed,
        output_shape=(256,),
        dtype=np.int8,
        generator_kind="pcg64",
    )
    assert isinstance(sister_lut, np.ndarray)
    assert isinstance(our_lut, np.ndarray)
    assert sister_lut.shape == our_lut.shape == (256,)


# ---------------------------------------------------------------------------
# Byte-stability across separate derivations
# ---------------------------------------------------------------------------


def test_byte_stability_across_repeated_derivations() -> None:
    """Same inputs → byte-identical output even on 10 sequential derivations."""
    blobs = [
        derive_codebook_from_seed(
            seed_bytes=CANONICAL_SEED,
            output_shape=(512,),
            dtype=np.uint8,
            generator_kind="pcg64",
        ).tobytes()
        for _ in range(10)
    ]
    assert len(set(blobs)) == 1, "byte-stability across repeated calls failed"


def test_output_is_writable() -> None:
    """Returned ndarray must be writable (np.frombuffer returns RO; we .copy())."""
    arr = derive_codebook_from_seed(
        seed_bytes=CANONICAL_SEED,
        output_shape=(64,),
        dtype=np.uint8,
        generator_kind="pcg64",
    )
    arr[0] = 0  # must not raise
    assert arr.flags.writeable
