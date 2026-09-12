"""Tests for ``experiments/ddm_cb1_carrier_basis_refit``.

The heavy tests are skipped when move 49's tree is not mounted; the light ones are
not, because they guard the arithmetic that a wrong answer would be silent about.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

cb1 = pytest.importorskip("ddm_cb1_carrier_basis_refit")

POINTER_MOUNTED = cb1.POINTER_ARCHIVE.exists()
needs_pointer = pytest.mark.skipif(
    not POINTER_MOUNTED, reason="move 49's tree is not mounted"
)


def test_zigzag_matches_the_receivers_inverse():
    """``symbol = zigzag(code)`` must invert to the receiver's own unzigzag."""
    codes = np.arange(cb1.BASIS_CODE_MIN, cb1.BASIS_CODE_MAX + 1, dtype=np.int64)
    symbols = cb1.zigzag_symbols(codes)
    # carrier_codec.py:187-190, verbatim.
    back = (symbols.astype(np.int64) >> 1) ^ -(symbols.astype(np.int64) & 1)
    assert np.array_equal(back, codes)
    assert symbols.min() >= 0
    assert symbols.max() <= 31


def test_zigzag_refuses_a_code_outside_the_five_bit_domain():
    with pytest.raises(cb1.Cb1Error):
        cb1.zigzag_symbols(np.array([16], dtype=np.int64))
    with pytest.raises(cb1.Cb1Error):
        cb1.zigzag_symbols(np.array([-17], dtype=np.int64))


def test_composed_score_reproduces_the_pointer_from_components():
    """The identity control: the arithmetic must land on move 49's banked row."""
    score = cb1.composed_score(
        cb1.POINTER_D_SEG_T4, cb1.POINTER_D_POSE_T4, cb1.POINTER_ARCHIVE_BYTES
    )
    assert abs(score - cb1.POINTER_SCORE_T4) < 1e-12


def test_principal_angles_are_zero_on_an_identical_span():
    rng = np.random.default_rng(7)
    basis = rng.standard_normal((40, 6))
    # a different basis for the SAME span must read zero degrees.
    remixed = basis @ rng.standard_normal((6, 6))
    angles = cb1.principal_angles(basis, remixed)
    assert angles.max() < 1e-6


def test_principal_angles_are_ninety_on_orthogonal_spans():
    identity = np.eye(20)
    angles = cb1.principal_angles(identity[:, :5], identity[:, 5:10])
    assert abs(angles.min() - 90.0) < 1e-6


def test_quantize_atom_stays_inside_the_shipped_alphabet():
    rng = np.random.default_rng(11)
    target = rng.standard_normal(
        (cb1.BASIS_PLANES, cb1.CARRIER_H, cb1.CARRIER_W)
    ).reshape(-1)
    codes, cosine = cb1.quantize_atom(target)
    assert codes.shape == (cb1.BASIS_PLANES, cb1.CARRIER_H, cb1.CARRIER_W)
    assert codes.min() >= cb1.BASIS_CODE_MIN
    assert codes.max() <= cb1.BASIS_CODE_MAX
    assert 0.0 < cosine <= 1.0


def test_quantize_atom_is_near_exact_on_a_smooth_target():
    """A band-limited smooth atom is what the carrier actually holds."""
    rows = np.linspace(-1.0, 1.0, cb1.CARRIER_H)[:, None]
    columns = np.linspace(-1.0, 1.0, cb1.CARRIER_W)[None, :]
    plane = np.cos(2.0 * rows) * np.sin(3.0 * columns)
    target = np.stack([plane, 0.5 * plane, -0.25 * plane]).reshape(-1)
    _codes, cosine = cb1.quantize_atom(target)
    assert cosine > 0.99


def test_quantize_atom_beats_the_naive_single_step():
    """The searched per-atom step must not be worse than a fixed one (pc1 section 5)."""
    rng = np.random.default_rng(3)
    target = rng.standard_normal(cb1.BASIS_PLANES * cb1.CARRIER_H * cb1.CARRIER_W)
    centred = target - target.mean()
    _codes, searched = cb1.quantize_atom(target)
    naive = np.clip(
        np.rint(centred / np.abs(centred).max() * 4.0),
        cb1.BASIS_CODE_MIN,
        cb1.BASIS_CODE_MAX,
    )
    naive = naive - naive.mean()
    naive_cos = float(
        (naive * centred).sum() / (np.linalg.norm(naive) * np.linalg.norm(centred))
    )
    assert searched >= naive_cos


def test_quantize_atom_refuses_a_zero_atom():
    with pytest.raises(cb1.Cb1Error):
        cb1.quantize_atom(np.zeros(cb1.BASIS_PLANES * cb1.CARRIER_H * cb1.CARRIER_W))


def test_jsonable_refuses_an_unknown_type():
    with pytest.raises(TypeError):
        cb1._jsonable(object())


def test_pose_base_gate_is_ten_times_the_measured_band():
    assert pytest.approx(10.0 * cb1.POSE_BATCH_BAND_ABS) == cb1.POSE_BASE_GATE_ABS
    # and the band is pp1's MEASURED maximum, not a chosen epsilon.
    assert pytest.approx(2.586e-09, rel=0.0) == cb1.POSE_BATCH_BAND_ABS


@needs_pointer
def test_pointer_identity_holds():
    receipts = cb1.assert_pointer()
    assert receipts["archive_sha256"] == cb1.POINTER_ARCHIVE_SHA256
    assert receipts["archive_bytes"] == cb1.POINTER_ARCHIVE_BYTES
    assert receipts["gt_lineage"] == cb1.up2.LINEAGE_DALI


@needs_pointer
def test_encode_basis_fields_round_trips_through_the_receivers_decoder():
    rng = np.random.default_rng(5)
    codes = rng.integers(
        cb1.BASIS_CODE_MIN,
        cb1.BASIS_CODE_MAX + 1,
        size=(cb1.CARRIER_DIM, cb1.BASIS_PLANES, cb1.CARRIER_H, cb1.CARRIER_W),
    )
    fields = cb1.encode_basis_fields(codes)
    rr5 = cb1._rr5()
    back = rr5.huffman_decode(
        np.asarray(fields["lengths"], dtype=np.int64),
        fields["payload"],
        fields["bits"],
        rr5.BASIS_SYMBOLS,
    )
    assert np.array_equal(back, cb1.zigzag_symbols(codes))
    assert fields["huffman_bytes"] > 0
    assert fields["rider_arith_bytes"] > 0


@needs_pointer
def test_with_basis_on_the_shipped_codes_reproduces_the_shipped_geometry():
    """Feeding the shipped codes back in must be a no-op on the span geometry."""
    inst = cb1.load_instrument()
    shipped = cb1.shipped_basis_codes(inst)
    sister = cb1.with_basis(inst, shipped)
    assert np.allclose(
        inst.blow.numpy(), sister.blow.numpy(), rtol=0.0, atol=1e-6
    )
    assert np.allclose(inst.gram.numpy(), sister.gram.numpy(), rtol=0.0, atol=1e-6)


@needs_pointer
def test_with_basis_refuses_codes_outside_the_five_bit_domain():
    inst = cb1.load_instrument()
    bad = cb1.shipped_basis_codes(inst).copy()
    bad[0, 0, 0, 0] = 31
    with pytest.raises(cb1.Cb1Error):
        cb1.with_basis(inst, bad)


@needs_pointer
def test_with_basis_refuses_a_wrong_shape():
    inst = cb1.load_instrument()
    with pytest.raises(cb1.Cb1Error):
        cb1.with_basis(inst, np.zeros((11, 3, 24, 32), dtype=np.int64))


def test_break_even_is_the_scoring_functions_own_arithmetic():
    """A byte saving must buy exactly the pose the score says it buys."""
    base = 4.543568679770593e-06
    # zero byte delta: with no rate saving the pose must IMPROVE by at least the bar,
    # so the break-even sits BELOW the base.  (The first version of this assertion
    # said `>` and passed against the sign error it was written to catch.)
    at_zero = cb1.break_even_d_pose(0, base)
    assert at_zero < base
    # recompose: the net at the break-even d_pose must equal the admit bar.
    for delta in (0, -209, -1277, +92):
        allowed = cb1.break_even_d_pose(delta, base)
        net = (
            math.sqrt(10.0 * allowed)
            - math.sqrt(10.0 * base)
            + delta * cb1.RATE_PER_BYTE
        )
        assert net == pytest.approx(cb1.ADMIT_BAR, rel=1e-9)


def test_break_even_refuses_a_byte_delta_larger_than_the_whole_pose_term():
    base = 4.543568679770593e-06
    with pytest.raises(cb1.Cb1Error):
        cb1.break_even_d_pose(1_000_000, base)


def test_a_byte_saving_raises_the_break_even_and_a_cost_lowers_it():
    base = 4.543568679770593e-06
    assert cb1.break_even_d_pose(-1277, base) > cb1.break_even_d_pose(-209, base)
    assert cb1.break_even_d_pose(+92, base) < cb1.break_even_d_pose(0, base)


def test_build_partial_refuses_an_alpha_outside_the_unit_segment():
    class _Stub:
        pass

    with pytest.raises(cb1.Cb1Error):
        cb1.build_partial(_Stub(), np.zeros(4), 0, 1.5)
    with pytest.raises(cb1.Cb1Error):
        cb1.build_partial(_Stub(), np.zeros(4), 0, -0.1)


@needs_pointer
def test_build_partial_at_alpha_zero_is_the_incumbent_atom():
    """alpha = 0 must reproduce the shipped atom's DIRECTION, so the ladder starts null."""
    inst = cb1.load_instrument()
    directions = np.load(
        Path("/Volumes/VertigoDataTier/pact/ddm_cb1/subspace/subspace_directions.npz")
    ) if Path(
        "/Volumes/VertigoDataTier/pact/ddm_cb1/subspace/subspace_directions.npz"
    ).exists() else None
    if directions is None:
        pytest.skip("subspace directions not measured in this workspace")
    _codes, detail = cb1.build_partial(inst, directions["residual_v"][:, 0], 6, 0.0)
    assert detail["realized_cosine_to_incumbent"] > 0.999


@needs_pointer
def test_build_partial_moves_the_atom_as_alpha_grows():
    inst = cb1.load_instrument()
    path = Path(
        "/Volumes/VertigoDataTier/pact/ddm_cb1/subspace/subspace_directions.npz"
    )
    if not path.exists():
        pytest.skip("subspace directions not measured in this workspace")
    directions = np.load(path)
    cosines = [
        cb1.build_partial(inst, directions["residual_v"][:, 0], 6, alpha)[1][
            "realized_cosine_to_incumbent"
        ]
        for alpha in (0.0, 0.5, 1.0)
    ]
    assert cosines[0] > cosines[1] > cosines[2]
