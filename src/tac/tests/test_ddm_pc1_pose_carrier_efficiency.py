"""Tests for ``experiments/ddm_pc1_pose_carrier_efficiency`` pure surfaces.

These cover the arithmetic and the coders' input contracts -- the parts that
decide a byte count or a verdict.  The scorer-bound modes are not tested here:
they need the frontier archive, its receiver copy and its 3.66 GB parse-back,
none of which belong in a unit test.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "experiments" / "ddm_pc1_pose_carrier_efficiency.py"


def _load_module():
    if str(REPO / "experiments") not in sys.path:
        sys.path.insert(0, str(REPO / "experiments"))
    spec = importlib.util.spec_from_file_location(
        "ddm_pc1_pose_carrier_efficiency", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: ``@dataclass`` resolves annotations through
    # ``sys.modules[cls.__module__]`` and raises on an unregistered module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pc1 = _load_module()


# --------------------------------------------------------------------------
# score arithmetic
# --------------------------------------------------------------------------


def test_break_even_at_zero_rate_change_is_the_base_itself():
    base = 6.134076e-06
    assert pc1.break_even_d_pose(base, 0.0) == pytest.approx(base)


def test_break_even_tightens_below_the_base_for_a_byte_adding_edit():
    base = 6.134076e-06
    assert 0.0 < pc1.break_even_d_pose(base, +1e-5) < base


def test_break_even_refuses_a_nonpositive_base():
    with pytest.raises(pc1.Pc1Error):
        pc1.break_even_d_pose(0.0, -1e-3)


def test_break_even_d_pose_is_the_exact_zero_of_the_score_delta():
    base = 6.134076407345324e-06
    rate = -1.20787e-03
    even = pc1.break_even_d_pose(base, rate)
    net = rate + (pc1.pose_leg(even) - pc1.pose_leg(base))
    assert abs(net) < 1e-15


def test_break_even_grows_with_the_size_of_the_rate_saving():
    base = 6.134076e-06
    small = pc1.break_even_d_pose(base, -2.6102e-04)
    large = pc1.break_even_d_pose(base, -8.01894e-03)
    assert base < small < large


def test_composed_score_matches_the_contest_definition():
    got = pc1.composed_score(0.00029229, 6.14e-06, 179_982)
    want = (
        100.0 * 0.00029229
        + math.sqrt(10.0 * 6.14e-06)
        + 25.0 * 179_982 / 37_545_489.0
    )
    assert got == pytest.approx(want, rel=0, abs=1e-15)


def test_byte_to_score_is_the_contest_rate_slope():
    assert pc1.BYTE_TO_SCORE == 25.0 / 37_545_489.0


def test_ft1_pose_ceiling_is_below_v4_break_even_so_the_ceiling_binds():
    """The ordering that decides which cap applies to the generated basis."""
    base = 6.134076407345324e-06
    v4_break_even = pc1.break_even_d_pose(base, -12_043 * pc1.BYTE_TO_SCORE)
    assert v4_break_even > pc1.FT1_POSE_CEILING


# --------------------------------------------------------------------------
# the basis symbol alphabet
# --------------------------------------------------------------------------


def test_basis_symbols_are_the_zigzag_of_the_signed_codes():
    codes = np.array([0, -1, 1, -2, 2, -15, 15], dtype=np.int32)
    got = pc1.basis_symbols_from_codes(codes)
    assert got.tolist() == [0, 1, 2, 3, 4, 29, 30]


def test_shipped_code_range_fits_the_five_bit_alphabet():
    codes = np.arange(-16, 16, dtype=np.int32)
    symbols = pc1.basis_symbols_from_codes(codes)
    assert symbols.min() >= 0
    assert symbols.max() < 32


# --------------------------------------------------------------------------
# quantisation -- the optimal-form contract
# --------------------------------------------------------------------------


def _atoms(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = rng.integers(
        -15, 16, size=(pc1.CARRIER_DIM, pc1.BASIS_PLANES, pc1.CARRIER_H, pc1.CARRIER_W)
    )
    return grid.astype(np.int32)


def test_quantize_basis_respects_the_target_alphabet():
    out = pc1.quantize_basis(_atoms(), 4, grid_points=6)
    assert out.min() >= -8 and out.max() <= 7
    out3 = pc1.quantize_basis(_atoms(), 3, grid_points=6)
    assert out3.min() >= -4 and out3.max() <= 3


def test_quantize_basis_refuses_an_unsupported_depth():
    with pytest.raises(pc1.Pc1Error):
        pc1.quantize_basis(_atoms(), 6, grid_points=4)
    with pytest.raises(pc1.Pc1Error):
        pc1.quantize_basis(_atoms(), 1, grid_points=4)


def test_quantize_basis_output_shape_is_the_carrier_grid():
    out = pc1.quantize_basis(_atoms(), 4, grid_points=6)
    assert out.shape == (
        pc1.CARRIER_DIM, pc1.BASIS_PLANES, pc1.CARRIER_H, pc1.CARRIER_W
    )


def test_quantize_basis_is_deterministic():
    first = pc1.quantize_basis(_atoms(7), 4, grid_points=6)
    second = pc1.quantize_basis(_atoms(7), 4, grid_points=6)
    assert np.array_equal(first, second)


def test_quantize_basis_beats_the_naive_global_step_it_replaces():
    """The optimal-form claim, as an executable contract.

    A per-atom searched step must be at least as faithful as the global
    ``codes >> (5 - bits)`` first pass on every atom, or the search is not
    doing what the module says it does.
    """
    codes = _atoms(3)
    optimal = pc1.quantize_basis(codes, 4, grid_points=24)
    naive = np.clip(np.rint(codes.astype(np.float64) / 2.0), -8, 7)

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = a.reshape(-1).astype(np.float64)
        b = b.reshape(-1).astype(np.float64)
        a = (a - a.mean()) / max(a.std(), 1e-12)
        b = (b - b.mean()) / max(b.std(), 1e-12)
        return float(np.dot(a, b) / a.size)

    for k in range(pc1.CARRIER_DIM):
        assert cosine(codes[k], optimal[k]) >= cosine(codes[k], naive[k]) - 1e-9


# --------------------------------------------------------------------------
# the generated basis
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["luma", "planar", "opponent"])
def test_dct_basis_shape_and_nonconstant_atoms(mode: str):
    atoms = pc1.dct_basis_2d(
        pc1.CARRIER_DIM, planes=pc1.BASIS_PLANES,
        height=pc1.CARRIER_H, width=pc1.CARRIER_W, mode=mode,
    )
    assert atoms.shape == (
        pc1.CARRIER_DIM, pc1.BASIS_PLANES, pc1.CARRIER_H, pc1.CARRIER_W
    )
    for k in range(pc1.CARRIER_DIM):
        assert atoms[k].std() > 0.0, f"{mode} atom {k} is constant"


def test_dct_basis_is_deterministic_and_depends_on_nothing_but_the_shape():
    first = pc1.dct_basis_2d(12, planes=3, height=24, width=32, mode="luma")
    second = pc1.dct_basis_2d(12, planes=3, height=24, width=32, mode="luma")
    assert np.array_equal(first, second)


def test_dct_luma_atoms_are_identical_on_every_plane():
    atoms = pc1.dct_basis_2d(12, planes=3, height=24, width=32, mode="luma")
    for k in range(12):
        assert np.allclose(atoms[k, 0], atoms[k, 1])
        assert np.allclose(atoms[k, 0], atoms[k, 2])


def test_dct_planar_atoms_live_on_one_plane_each():
    atoms = pc1.dct_basis_2d(12, planes=3, height=24, width=32, mode="planar")
    for k in range(12):
        live = [d for d in range(3) if np.abs(atoms[k, d]).max() > 0]
        assert len(live) == 1


def test_dct_basis_refuses_an_unknown_mode():
    with pytest.raises(pc1.Pc1Error):
        pc1.dct_basis_2d(12, planes=3, height=24, width=32, mode="wavelet")


def test_dct_basis_refuses_a_count_not_divisible_by_the_plane_count():
    with pytest.raises(pc1.Pc1Error):
        pc1.dct_basis_2d(11, planes=3, height=24, width=32, mode="planar")


def test_quantize_generated_refuses_an_identically_zero_atom():
    atoms = np.zeros((2, 3, 4, 4), dtype=np.float64)
    with pytest.raises(pc1.Pc1Error):
        pc1.quantize_generated(atoms)


# --------------------------------------------------------------------------
# the coefficient pre-transform the Rice coder consumes
# --------------------------------------------------------------------------


def test_zigzag_delta_round_trips_through_the_receiver_s_cumsum():
    """Encoder/decoder identity against ``cpr1/inflate.py:236-244`` read forwards."""
    rng = np.random.default_rng(11)
    codes = rng.integers(-2048, 2048, size=(pc1.N_PAIRS, pc1.CARRIER_DIM)).astype(
        np.int32
    )
    zig = pc1._zigzag_delta_along_pairs(codes)
    delta = (zig.astype(np.int64) >> 1) ^ -(zig.astype(np.int64) & 1)
    recovered = np.cumsum(delta, axis=0) & 0xFFF
    recovered = np.where(recovered >= 0x800, recovered - 0x1000, recovered)
    assert np.array_equal(recovered.astype(np.int32), codes)


def test_zigzag_delta_output_is_inside_the_unsigned_int12_domain():
    rng = np.random.default_rng(5)
    codes = rng.integers(-2048, 2048, size=(pc1.N_PAIRS, pc1.CARRIER_DIM)).astype(
        np.int32
    )
    zig = pc1._zigzag_delta_along_pairs(codes)
    assert zig.min() >= 0 and zig.max() < 4096


def test_zigzag_delta_refuses_a_wrong_shaped_lattice():
    with pytest.raises(pc1.Pc1Error):
        pc1._zigzag_delta_along_pairs(np.zeros((10, 12), dtype=np.int32))


# --------------------------------------------------------------------------
# the variant table
# --------------------------------------------------------------------------


def test_every_variant_name_matches_its_spec_name():
    for name, spec in pc1.VARIANTS.items():
        assert spec.name == name


def test_base_variant_is_the_identity_on_both_axes():
    base = pc1.VARIANTS["v0_base"]
    assert base.basis_bits is None
    assert base.generated_mode is None
    assert base.svd_rank is None
    assert base.lattice_factor == 1


def test_generated_variants_store_no_basis():
    for name in ("v4_dct_luma", "v4_dct_planar", "v4_dct_opponent"):
        assert pc1.VARIANTS[name].generated_mode is not None
        assert pc1.VARIANTS[name].basis_bits is None
