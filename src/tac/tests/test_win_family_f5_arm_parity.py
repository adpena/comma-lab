# SPDX-License-Identifier: MIT
"""F5 CONSUMER PROOF: the canonical instrument is equivalent to the arm-local ones.

WHY THIS FILE EXISTS.  A canonical surface with no consumer is an orphan (#936).  The two
live arms that own the pose and seg gates -- ``ddm_up2`` and ``ddm_jg1`` -- cannot be
edited while they are in flight, so adoption cannot be proved by migrating their imports
today.  It CAN be proved the other way round: this file imports both arms and checks that
:mod:`tac.local_contest_instruments` returns the SAME answers on the SAME inputs.

That makes the canonical a proven drop-in rather than an assertion, and it turns any
future divergence between the arms and the canonical into a failing test instead of a
silent fork.  The adoption rows for MAIN are listed in
``.omx/research/ddm_cw1_win_family_canonicalization_20260819.md``.

The arms are IMPORTED, never modified.  Both import cleanly without torch (their scorer
loads are inside functions), so this suite stays hermetic and fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

from tac import gt_lineage  # noqa: E402
from tac import local_contest_instruments as lci  # noqa: E402

up2 = pytest.importorskip("ddm_up2_shipping_pose_solve")
jg1 = pytest.importorskip("ddm_jg1_seg_solve")


# --- the axis <-> lineage table -------------------------------------------------

#: up2 spells the lineages with its own local strings; this is the intended mapping onto
#: the content-addressed registry's constants.
_UP2_TO_CANONICAL = {
    up2.LINEAGE_DALI: gt_lineage.DALI_NVDEC,
    up2.LINEAGE_AV_PYAV: gt_lineage.PYAV_YUV420_TO_RGB,
}
#: up2's axis keys, mapped onto the canonical axis labels.
_UP2_AXIS_TO_CANONICAL = {
    "contest_cuda": lci.AXIS_CONTEST_CUDA,
    "contest_cpu": lci.AXIS_CONTEST_CPU,
    "macos_cpu_advisory": lci.AXIS_MACOS_CPU_ADVISORY,
}


def test_canonical_covers_every_axis_up2_knows():
    assert set(_UP2_AXIS_TO_CANONICAL.values()) == set(lci.AXIS_GT_LINEAGE)


def test_axis_lineage_table_agrees_with_up2():
    for up2_axis, canonical_axis in _UP2_AXIS_TO_CANONICAL.items():
        expected = _UP2_TO_CANONICAL[up2.AXIS_GT_LINEAGE[up2_axis]]
        assert lci.required_lineage_for_axis(canonical_axis) == expected


def test_unknown_axis_refuses_in_both():
    with pytest.raises(up2.Up2Error):
        up2.required_lineage_for_axis("contest_tpu")
    with pytest.raises(lci.InstrumentRefusal):
        lci.required_lineage_for_axis("contest-TPU")


# --- score arithmetic ------------------------------------------------------------

_ROWS = [
    (0.00030309, 7.77e-06, 176_420),
    (0.0052766, 1.4e-04, 250_898),
    (0.00389011, 2.5e-05, 569_996),
]


@pytest.mark.parametrize(("d_seg", "d_pose", "archive_bytes"), _ROWS)
def test_score_matches_up2_exactly(d_seg, d_pose, archive_bytes):
    assert lci.contest_score_from_legs(d_seg, d_pose, archive_bytes) == pytest.approx(
        up2._score_from(d_pose, d_seg, archive_bytes), abs=1e-15
    )


@pytest.mark.parametrize(("_seg", "d_pose", "_bytes"), _ROWS)
def test_pose_leg_matches_up2_exactly(_seg, d_pose, _bytes):
    assert lci.pose_leg(d_pose) == pytest.approx(up2.pose_leg(d_pose), abs=1e-15)


@pytest.mark.parametrize(("_seg", "d_pose", "_bytes"), _ROWS)
def test_pose_report_bound_matches_up2_exactly(_seg, d_pose, _bytes):
    assert lci.pose_report_bound(d_pose) == pytest.approx(
        up2.pose_report_bound(d_pose), abs=1e-18
    )


def test_pose_report_bound_at_zero_matches_up2():
    assert lci.pose_report_bound(0.0) == pytest.approx(up2.pose_report_bound(0.0))


def test_resolvable_floor_matches_up2():
    assert lci.resolvable_d_pose_floor() == up2.resolvable_d_pose_floor()


def test_byte_to_score_matches_up2():
    assert lci.rate_leg(1) == pytest.approx(up2.BYTE_TO_SCORE, abs=1e-18)


def test_pointer_score_matches_up2s_recorded_constant():
    """EXECUTED: the canonical arithmetic reproduces up2's own recorded pointer score."""
    assert lci.contest_score_from_legs(
        up2.POINTER_D_SEG_T4, up2.POINTER_D_POSE_T4, up2.POINTER_ARCHIVE_BYTES
    ) == pytest.approx(up2.POINTER_SCORE, abs=1e-15)


# --- population selection ---------------------------------------------------------


@pytest.mark.parametrize("pairs", [8, 96, 120, 599, 600, 1200])
def test_select_pairs_matches_up2_exactly(pairs):
    assert np.array_equal(
        lci.select_pairs(pairs, seed=20260819), up2.select_pairs(pairs, 20260819)
    )


@pytest.mark.parametrize("seed", [1, 7, 20260819])
def test_select_pairs_matches_up2_across_seeds(seed):
    assert np.array_equal(lci.select_pairs(96, seed=seed), up2.select_pairs(96, seed))


def test_both_refuse_a_prefix_at_sub_n600():
    """Neither may hand back [0..n): the prefix bias inverts sign per axis."""
    assert not np.array_equal(lci.select_pairs(96, seed=1), np.arange(96))
    assert not np.array_equal(up2.select_pairs(96, 1), np.arange(96))


# --- the seg leg ------------------------------------------------------------------


def test_d_seg_per_pair_matches_jg1_exactly():
    rng = np.random.default_rng(11)
    argmax = rng.integers(0, 5, size=(4, 8, 8)).astype(np.uint8)
    gt = rng.integers(0, 5, size=(4, 8, 8)).astype(np.uint8)
    assert np.array_equal(lci.d_seg_per_pair(argmax, gt), jg1.d_seg_per_pair(argmax, gt))


def test_d_seg_per_pair_matches_jg1_on_identity():
    argmax = np.zeros((3, 4, 4), dtype=np.uint8)
    assert np.array_equal(
        lci.d_seg_per_pair(argmax, argmax.copy()), jg1.d_seg_per_pair(argmax, argmax.copy())
    )


def test_both_refuse_a_seg_shape_mismatch():
    with pytest.raises(jg1.Jg1Error):
        jg1.d_seg_per_pair(np.zeros((2, 2, 2)), np.zeros((2, 2, 3)))
    with pytest.raises(lci.InstrumentRefusal):
        lci.d_seg_per_pair(np.zeros((2, 2, 2)), np.zeros((2, 2, 3)))


def test_seg_cell_exchange_rate_matches_jg1():
    """jg1's S_PER_SEG_CELL is 100/117,964,800; the canonical seg leg must agree."""
    one_cell_d_seg = 1.0 / jg1.SEG_CELLS_TOTAL
    assert lci.seg_leg(one_cell_d_seg) == pytest.approx(jg1.S_PER_SEG_CELL, rel=1e-12)


def test_archive_byte_exchange_rate_matches_jg1():
    assert lci.rate_leg(1) == pytest.approx(jg1.S_PER_ARCHIVE_BYTE, rel=1e-12)


def test_jg1_published_seg_legs_bracket_the_lineage_factor():
    """jg1's two published legs differ by the pi2 seg factor, within the arm's own spread."""
    ratio = jg1.POINTER_D_SEG_AV / jg1.POINTER_D_SEG_DALI
    assert ratio == pytest.approx(lci.ADVISORY_SEG_MULTIPLICATIVE_FACTOR, rel=0.05)


# --- the gap the canonical closes ---------------------------------------------------


def test_up2_resolves_lineage_by_filename_and_the_canonical_does_not():
    """The defect this canonicalisation fixes, asserted as behaviour.

    ``ddm_up2.load_gt_poses`` infers lineage from a filename substring.  ``ddm_gl1``
    measured seven files named ``gt_argmax_n600.npy`` across three distinct sha256 and
    BOTH lineages, so a name cannot be identity.  The canonical path resolves by content
    through :mod:`tac.gt_lineage` and refuses an unregistered artifact outright.
    """
    source = Path(up2.__file__).read_text(encoding="utf-8")
    assert 'LINEAGE_DALI if "dali" in path.name.lower()' in source, (
        "up2's filename-based lineage inference changed; re-check whether this "
        "canonicalisation rationale still holds"
    )
    with pytest.raises(gt_lineage.GtLineageError):
        lci.assert_axis_lineage(
            REPO / "pyproject.toml", axis=lci.AXIS_CONTEST_CUDA, instrument="parity_test"
        )
