"""Regression guards for the ddm_sa2 compensation solve and compile path.

The controls that actually caught bugs in this arm are encoded here:

* the LSB-first packed-CAP1 metadata inverse (a wrong bit order silently produced
  a section the receiver rejected);
* the CAP1 field REORDER between the blob and the archive body (a wrong order read
  +68 B of field-order damage as if it were compensation cost);
* the Rice cost model against the shipped bit count;
* the integer-descent termination contract.

Tests that need the external stores skip when those stores are not mounted, so the
suite stays runnable on a bare checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_sa2_compile_candidate as compile_mod

STORES = (
    compile_mod.RR4_RUNTIME.is_dir()
    and compile_mod.BOOK_SRC.is_dir()
    and compile_mod.S2_ARCHIVE.is_file()
)
needs_stores = pytest.mark.skipif(not STORES, reason="sa2 external stores not mounted")


# --------------------------------------------------------------------------
# packed-CAP1 metadata inverse
# --------------------------------------------------------------------------


def _unpack_reference(raw: bytes, count: int, bits: int) -> np.ndarray:
    """Independent re-derivation of the reader's LSB-first unpack."""
    out = np.empty(count, dtype=np.int64)
    for index in range(count):
        offset = index * bits
        byte, shift = divmod(offset, 8)
        word = raw[byte]
        if byte + 1 < len(raw):
            word |= raw[byte + 1] << 8
        out[index] = (word >> shift) & ((1 << bits) - 1)
    return out


@pytest.mark.parametrize(
    ("count", "bits"), [(12, 7), (12, 6), (32, 4), (12, 1), (5, 3), (1, 8)]
)
def test_pack_unsigned_round_trips_lsb_first(count: int, bits: int) -> None:
    rng = np.random.default_rng(count * 100 + bits)
    values = rng.integers(0, 1 << bits, size=count)
    packed = compile_mod.pack_unsigned(values, count, bits)
    assert len(packed) == (count * bits + 7) // 8
    assert np.array_equal(_unpack_reference(packed, count, bits), values)


def test_pack_unsigned_zero_pads_the_tail() -> None:
    packed = compile_mod.pack_unsigned([127] * 12, 12, 7)
    assert len(packed) == 11
    assert packed[-1] >> (12 * 7 % 8) == 0


def test_pack_unsigned_refuses_out_of_domain() -> None:
    with pytest.raises(compile_mod.SA2CompileError):
        compile_mod.pack_unsigned([1 << 7], 1, 7)


def _canonical_metadata(factors, biases, lengths, ks, rest=b"\x00" * 4) -> bytes:
    return (
        bytes(102)
        + np.asarray(factors, dtype="<i2").tobytes()
        + np.asarray(biases, dtype=np.int8).tobytes()
        + np.asarray(lengths, dtype=np.uint8).tobytes()
        + np.asarray(ks, dtype=np.uint8).tobytes()
        + rest
    )


def test_pack_cap1_metadata_refuses_wide_predictor_spread() -> None:
    canonical = _canonical_metadata(
        [100] * 11 + [400], [0] * 12, [1] * 32, [9] * 12
    )
    with pytest.raises(compile_mod.SA2CompileError, match="predictor factor spread"):
        compile_mod.pack_cap1_metadata(canonical)


def test_pack_cap1_metadata_refuses_wide_rice_spread() -> None:
    canonical = _canonical_metadata(
        [150] * 12, [0] * 12, [1] * 32, [8] * 11 + [11]
    )
    with pytest.raises(compile_mod.SA2CompileError, match="rice k spread"):
        compile_mod.pack_cap1_metadata(canonical)


def test_pack_cap1_metadata_refuses_out_of_domain_bias() -> None:
    canonical = _canonical_metadata([150] * 12, [17] * 12, [1] * 32, [9] * 12)
    with pytest.raises(compile_mod.SA2CompileError, match="canonical domains"):
        compile_mod.pack_cap1_metadata(canonical)


def test_pack_cap1_metadata_preserves_the_trailing_payload() -> None:
    rest = bytes(range(64))
    canonical = _canonical_metadata([150] * 12, [-3] * 12, [2] * 32, [9] * 12, rest)
    packed = compile_mod.pack_cap1_metadata(canonical)
    assert packed.endswith(rest)
    assert len(packed) == len(canonical) - 40


# --------------------------------------------------------------------------
# the solve module's pure surfaces
# --------------------------------------------------------------------------


solve_mod = pytest.importorskip("experiments.ddm_sa2_compensated_semantic_edit")


def test_rice_model_matches_a_hand_computed_stream() -> None:
    encoded = np.array([[0, 1], [512, 513]], dtype=np.int64)
    parameters = np.array([9, 9], dtype=np.uint8)
    # per value: (value >> k) unary + 1 stop bit + k remainder bits
    expected = (0 + 1 + 0 + 1) + 4 * (1 + 9)
    assert solve_mod.rice_bits(encoded, parameters) == expected


def test_zigzag_is_injective_and_nonnegative() -> None:
    values = np.arange(-2048, 2048, dtype=np.int64)
    codes = solve_mod.zigzag(values)
    assert codes.min() >= 0
    assert len(set(codes.tolist())) == len(values)


def test_carrier_coefficient_bits_refuses_out_of_domain() -> None:
    codes = np.zeros((4, solve_mod.DIMENSIONS), dtype=np.int64)
    parameters = np.full(solve_mod.DIMENSIONS, 9, dtype=np.uint8)
    assert solve_mod.carrier_coefficient_bits(codes, parameters) > 0


def test_damped_least_squares_solves_a_well_posed_system() -> None:
    rng = np.random.default_rng(5)
    jacobian = rng.normal(size=(6, 12))
    target = rng.normal(size=6)
    update, rank, condition = solve_mod.damped_least_squares(jacobian, target, 1e-9)
    assert rank == 6
    assert condition < 1e6
    assert np.allclose(jacobian @ update, target, atol=1e-5)


def test_damped_least_squares_is_stable_on_a_rank_deficient_system() -> None:
    jacobian = np.zeros((6, 12))
    jacobian[0, 0] = 1.0
    update, rank, _ = solve_mod.damped_least_squares(
        jacobian, np.ones(6), solve_mod.GN_DAMPING
    )
    assert rank == 1
    assert np.all(np.isfinite(update))


class _QuadraticEvaluator:
    """Stand-in with a known integer optimum, to test descent termination."""

    def __init__(self, optimum: np.ndarray) -> None:
        self.optimum = np.asarray(optimum, dtype=np.int64)
        self.calls = 0

    def objectives(self, codes):
        self.calls += 1
        values = np.array(
            [float(np.sum((np.asarray(c, dtype=np.int64) - self.optimum) ** 2)) for c in codes]
        )
        return np.zeros((len(codes), 6)), values


def test_multiscale_descent_reaches_the_integer_optimum() -> None:
    optimum = np.zeros(solve_mod.DIMENSIONS, dtype=np.int64)
    optimum[0] = 68
    optimum[3] = -20
    start = np.zeros(solve_mod.DIMENSIONS, dtype=np.int32)
    evaluator = _QuadraticEvaluator(optimum)
    final, value, trace = solve_mod.multiscale_descent(
        evaluator, start, float(np.sum(optimum**2)), solve_mod.DESCENT_LADDER
    )
    assert np.array_equal(final.astype(np.int64), optimum)
    assert value == 0.0
    assert [row["step"] for row in trace] == list(solve_mod.DESCENT_LADDER)
    assert all(row["full_passes"] >= 1 for row in trace)


def test_multiscale_descent_stops_when_no_step_improves() -> None:
    optimum = np.zeros(solve_mod.DIMENSIONS, dtype=np.int64)
    evaluator = _QuadraticEvaluator(optimum)
    start = np.zeros(solve_mod.DIMENSIONS, dtype=np.int32)
    final, value, trace = solve_mod.multiscale_descent(
        evaluator, start, 0.0, solve_mod.DESCENT_LADDER
    )
    assert np.array_equal(final, start)
    assert value == 0.0
    # one non-improving pass per ladder rung, and nothing more
    assert [row["full_passes"] for row in trace] == [1] * len(solve_mod.DESCENT_LADDER)


def test_descent_respects_the_int12_domain() -> None:
    optimum = np.full(solve_mod.DIMENSIONS, 5000, dtype=np.int64)
    evaluator = _QuadraticEvaluator(optimum)
    start = np.full(solve_mod.DIMENSIONS, 2000, dtype=np.int32)
    final, _value, _trace = solve_mod.multiscale_descent(
        evaluator, start, 1e18, solve_mod.DESCENT_LADDER
    )
    assert final.max() <= solve_mod.INT12_MAX
    assert final.min() >= solve_mod.INT12_MIN


# --------------------------------------------------------------------------
# controls that need the external stores
# --------------------------------------------------------------------------


@needs_stores
def test_rice_model_reproduces_the_shipped_coefficient_bit_count() -> None:
    surface = solve_mod.RR4Frame0Surface.load()
    measured = solve_mod.carrier_coefficient_bits(
        surface.raw_codes, surface.rice_parameters
    )
    assert measured == 79_020


@needs_stores
def test_reencoding_the_unchanged_lattice_is_byte_identical() -> None:
    mod = compile_mod._imports()
    surface = compile_mod.carrier_surface(compile_mod.BASE_ARCHIVE, mod)
    encoded = compile_mod.encode_carrier_body(
        surface["codes"], surface, mod, surface["overlay"]
    )
    assert encoded["packed_form"] == "packed"
    shipped = (
        mod.ra._restore_packed_cap1_metadata(
            mod.ra._decompress_brotli(surface["carrier_stream"])[
                : mod.ra.PACKED_CAP1_SECTION_BYTES
            ]
        )
        + surface["overlay"]
    )
    mine = mod.ra._restore_packed_cap1_metadata(encoded["section"]) + surface["overlay"]
    assert mine == shipped


@needs_stores
def test_frame0_is_disjoint_from_the_semantic_edit() -> None:
    """The structural premise of the whole arm, asserted against real bytes."""
    mod = compile_mod._imports()
    base = compile_mod.carrier_surface(compile_mod.BASE_ARCHIVE, mod)
    edited = compile_mod.carrier_surface(compile_mod.S2_ARCHIVE, mod)
    assert base["parts"].carrier_blob == edited["parts"].carrier_blob
    assert base["parts"].semantic_blob != edited["parts"].semantic_blob
    for name in ("hpac_blob", "token_stream", "residual_payload"):
        assert getattr(base["parts"], name) == getattr(edited["parts"], name)
