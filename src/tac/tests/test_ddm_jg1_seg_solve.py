"""Tests for ``experiments/ddm_jg1_seg_solve.py`` -- the ddm_jg1 seg instrument.

The controls that matter here are the ones that would still pass if the module were
replaced by a stub returning constants.  Each test below is written so that it would
FAIL against such a stub: the accounting identities are checked against independently
computed values, the refusals are exercised with inputs that must be rejected, and the
exchange rate is recomputed from the score definition rather than read back.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
EXPERIMENTS = REPO / "experiments"


def _load_module():
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location(
        "ddm_jg1_seg_solve", EXPERIMENTS / "ddm_jg1_seg_solve.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ddm_jg1_seg_solve"] = module
    spec.loader.exec_module(module)
    return module


jg1 = _load_module()


# ---------------------------------------------------------------------------------
# Geometry and the score arithmetic
# ---------------------------------------------------------------------------------


def test_seg_cell_total_matches_the_token_field():
    """The d_seg denominator IS the token grid; if these ever diverge the exchange
    rate silently becomes wrong, which is the kind of error that reads as a result."""
    assert jg1.SEG_CELLS_TOTAL == jg1.N_PAIRS * jg1.EVAL_H * jg1.EVAL_W
    assert jg1.SEG_CELLS_TOTAL == 117_964_800


def test_exchange_rate_recomputed_from_the_score_definition():
    """S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489.

    One repaired cell moves d_seg by 1/SEG_CELLS_TOTAL, hence 100/SEG_CELLS_TOTAL in S.
    Recomputed here from the definition, not read back from the module's own constant.
    """
    s_cell = 100.0 / (600 * 384 * 512)
    s_byte = 25.0 / 37_545_489
    assert pytest.approx(s_cell, rel=1e-12) == jg1.S_PER_SEG_CELL
    assert pytest.approx(s_byte, rel=1e-12) == jg1.S_PER_ARCHIVE_BYTE
    assert pytest.approx(s_cell / s_byte, rel=1e-12) == jg1.BYTES_PER_SEG_CELL
    # The governing number, quoted in the memo, is ~1.27 bytes per repaired cell.
    assert 1.27 < jg1.BYTES_PER_SEG_CELL < 1.28


def test_class_names_are_canonical_comma10k_order_not_the_luma_sort():
    """The luma sort of comma10k ``class_values=[41,76,90,124,161]`` gives
    ``[Road, Lane, MyCar, Undrivable, Movable]`` and is WRONG.  It has bitten the
    campaign three times, so it gets a test rather than a comment."""
    assert jg1.CLASS_NAMES == ("Road", "Lane", "Undrivable", "Movable", "MyCar")
    assert jg1.CLASS_NAMES[2] != "MyCar"
    assert jg1.CLASS_NAMES[4] == "MyCar"
    assert jg1.NUM_CLASSES == len(jg1.CLASS_NAMES) == 5


# ---------------------------------------------------------------------------------
# d_seg -- the contest's own definition
# ---------------------------------------------------------------------------------


def test_d_seg_per_pair_matches_modules_compute_distortion_semantics():
    """``modules.py:111-113`` is ``(a.argmax != b.argmax).float().mean(spatial)``.

    Built here from a hand-countable field so the expected value is arithmetic, not a
    second implementation of the same loop.
    """
    argmax = np.zeros((3, 4, 5), dtype=np.uint8)
    gt = np.zeros((3, 4, 5), dtype=np.uint8)
    gt[0, 0, 0] = 1  # 1 of 20 cells in pair 0
    gt[1, :2, :2] = 2  # 4 of 20 cells in pair 1
    # pair 2 identical -> 0
    out = jg1.d_seg_per_pair(argmax, gt)
    assert out.shape == (3,)
    assert out[0] == pytest.approx(1 / 20)
    assert out[1] == pytest.approx(4 / 20)
    assert out[2] == 0.0
    assert out.dtype == np.float64


def test_d_seg_per_pair_refuses_shape_mismatch():
    with pytest.raises(jg1.Jg1Error, match="disagree in shape"):
        jg1.d_seg_per_pair(
            np.zeros((2, 4, 5), dtype=np.uint8), np.zeros((3, 4, 5), dtype=np.uint8)
        )


def test_d_seg_is_sensitive_to_every_changed_cell():
    """A stub that ignored its input would return a constant.  Ramp the disagreement
    one cell at a time and require a strictly monotone response."""
    gt = np.zeros((1, 8, 8), dtype=np.uint8)
    previous = -1.0
    for k in range(0, 9):
        argmax = np.zeros((1, 8, 8), dtype=np.uint8)
        argmax.reshape(-1)[:k] = 3
        value = float(jg1.d_seg_per_pair(argmax, gt)[0])
        assert value > previous
        previous = value
    assert previous == pytest.approx(8 / 64)


# ---------------------------------------------------------------------------------
# The decomposition that decides the strategy
# ---------------------------------------------------------------------------------


def test_token_vs_gt_agreement_accounting_closes():
    tokens = np.zeros((2, 3, 3), dtype=np.uint8)
    gt = np.zeros((2, 3, 3), dtype=np.uint8)
    gt[0, 0, 0] = 1
    gt[0, 1, 1] = 2
    gt[1, 2, 2] = 4
    report = jg1.token_vs_gt_agreement(tokens, gt, np.array([0, 1]))
    assert report["cells"] == 18
    assert report["token_gt_disagreeing_cells"] == 3
    assert report["token_gt_disagreement_rate"] == pytest.approx(3 / 18)
    confusion = np.array(report["confusion_gt_rows_token_cols"])
    # every cell lands in exactly one confusion bin
    assert confusion.sum() == 18
    # gt=1 was stored as token 0 exactly once
    assert confusion[1, 0] == 1
    assert confusion[2, 0] == 1
    assert confusion[4, 0] == 1
    assert confusion[0, 0] == 15


def test_flip_ledger_partitions_every_flip_exactly_once():
    """The edge counts must sum to the flip count, and the token split must too.
    Both are partitions; if either leaked, a decomposition quoted in the memo would
    silently not add up."""
    rng = np.random.default_rng(20260819)
    gt = rng.integers(0, 5, size=(4, 16, 16)).astype(np.uint8)
    argmax = gt.copy()
    tokens = gt.copy()
    # 30 deliberate argmax flips, 10 of which also have a wrong stored token
    flat_arg = argmax.reshape(-1)
    flat_tok = tokens.reshape(-1)
    picks = rng.choice(flat_arg.size, size=30, replace=False)
    for n, cell in enumerate(picks):
        flat_arg[cell] = (flat_arg[cell] + 1) % 5
        if n < 10:
            flat_tok[cell] = (flat_tok[cell] + 2) % 5
    ledger = jg1.flip_ledger(argmax, gt, tokens)
    assert ledger["flips"] == 30
    edges = np.array(ledger["edge_counts_gt_rows_ours_cols"])
    assert edges.sum() == 30
    assert np.trace(edges) == 0  # a flip never has gt == ours
    assert (
        ledger["flips_where_stored_token_already_equals_gt"]
        + ledger["flips_where_stored_token_is_wrong"]
        == 30
    )
    assert ledger["flips_where_stored_token_is_wrong"] == 10
    assert sum(ledger["token_class_at_flip"]) == 30


def test_flip_ledger_is_empty_when_there_is_nothing_to_explain():
    gt = np.full((2, 5, 5), 2, dtype=np.uint8)
    ledger = jg1.flip_ledger(gt.copy(), gt, gt.copy())
    assert ledger["flips"] == 0
    assert np.array(ledger["edge_counts_gt_rows_ours_cols"]).sum() == 0
    assert ledger["flips_where_stored_token_is_wrong"] == 0


# ---------------------------------------------------------------------------------
# Refusals -- fail closed, loudly
# ---------------------------------------------------------------------------------


def test_load_tokens_refuses_a_wrong_sized_field(tmp_path):
    bad = tmp_path / "short.u8"
    bad.write_bytes(b"\x00" * 1024)
    with pytest.raises(jg1.Jg1Error, match="expected"):
        jg1.load_tokens(bad)


def test_load_tokens_refuses_a_value_outside_the_embedding_domain(tmp_path):
    """A token >= NUM_CLASSES would index past ``nn.Embedding(5, width)``.  Such a
    field is not a class map, and silently accepting it would produce a 'seg result'
    from a decode that the receiver could not even run."""
    field = np.zeros(jg1.N_PAIRS * jg1.EVAL_H * jg1.EVAL_W, dtype=np.uint8)
    field[7] = 5
    path = tmp_path / "bad_domain.u8"
    field.tofile(path)
    with pytest.raises(jg1.Jg1Error, match="outside the embedding domain"):
        jg1.load_tokens(path)


def test_load_tokens_refuses_a_missing_file(tmp_path):
    with pytest.raises(jg1.Jg1Error, match="does not exist"):
        jg1.load_tokens(tmp_path / "absent.u8")


def test_load_gt_seg_labels_refuses_an_unknown_lineage():
    with pytest.raises(jg1.Jg1Error, match="unknown GT lineage"):
        jg1.load_gt_seg_labels("mps_whatever")


def test_lineage_gate_is_up2s_and_still_fails_closed():
    """The seg axis inherits the GT-lineage gate rather than re-implementing it.
    ddm_up3 measured the two lineages 1.43x apart ON THE SEG AXIS, so a contest-CUDA
    seg claim scored against PyAV GT is simply the wrong number."""
    from ddm_up2_shipping_pose_solve import (
        LINEAGE_AV_PYAV,
        LINEAGE_DALI,
        Up2Error,
        verify_gt_lineage,
    )

    assert verify_gt_lineage(axis="contest_cuda", declared_lineage=LINEAGE_DALI)[
        "status"
    ] == "VERIFIED"
    with pytest.raises(Up2Error, match="lineage mismatch"):
        verify_gt_lineage(axis="contest_cuda", declared_lineage=LINEAGE_AV_PYAV)


def test_seg_leg_carries_its_lineage_into_the_serialised_form():
    """A number without its lineage is unquotable; the dataclass must not be able to
    emit one."""
    leg = jg1.SegLeg(
        lineage="dali",
        pairs=4,
        sampling="seeded_random",
        d_seg=0.0003,
        cells_compared=100,
        cells_disagreeing=3,
        per_pair=(0.0, 0.0, 0.0, 0.0012),
    )
    blob = leg.to_json()
    assert blob["lineage"] == "dali"
    assert "axis_note" in blob


def test_resolve_indices_never_returns_a_prefix_below_n600():
    """ddm_bp2/ddm_na2 measured a contiguous prefix as a DIFFERENT POPULATION.  The
    selector is up2's precisely so this arm cannot re-introduce the bug."""
    picked = jg1._resolve_indices(48, 20260819)
    assert len(picked) == 48
    assert not np.array_equal(picked, np.arange(48))
    assert len(np.unique(picked)) == 48
    assert picked.max() < jg1.N_PAIRS
    # deterministic for a fixed seed
    assert np.array_equal(picked, jg1._resolve_indices(48, 20260819))


def test_resolve_indices_full_field_is_the_whole_field():
    picked = jg1._resolve_indices(jg1.N_PAIRS, 1)
    assert np.array_equal(picked, np.arange(jg1.N_PAIRS))
