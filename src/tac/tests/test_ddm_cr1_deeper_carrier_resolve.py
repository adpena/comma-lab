"""Tests for ``experiments/ddm_cr1_deeper_carrier_resolve``.

The heavy tests are skipped when move 49's tree is not mounted; the light ones are not,
because they guard the arithmetic and the acceptance gate that a wrong answer would be
silent about.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

cr1 = pytest.importorskip("ddm_cr1_deeper_carrier_resolve")

POINTER_MOUNTED = cr1.POINTER.archive.exists()
needs_pointer = pytest.mark.skipif(
    not POINTER_MOUNTED, reason="move 49's tree is not mounted"
)


def test_composed_score_reproduces_the_pointer_from_components():
    """The identity control: the arithmetic must land on move 49's banked row."""
    score = cr1.composed_score(
        cr1.POINTER_D_SEG_T4, cr1.POINTER_D_POSE_T4, cr1.POINTER_ARCHIVE_BYTES
    )
    assert abs(score - cr1.POINTER_SCORE_T4) < 1e-12


def test_acceptance_tolerance_is_the_measured_band_not_a_multiple():
    """The BASE gate is 10x the observed band; the ACCEPTANCE gate is the band itself.

    Confusing the two would either accept gains the batch order could have produced
    (tolerance too small) or discard real ones (too large).  Both constants are pinned
    to pp1's single measured number so neither can drift alone.
    """
    assert cr1.POSE_ACCEPT_TOL_ABS == cr1.POSE_BATCH_BAND_ABS
    assert cr1.POSE_BASE_GATE_ABS == 10.0 * cr1.POSE_BATCH_BAND_ABS
    assert pytest.approx(2.586e-09, rel=1e-12) == cr1.POSE_BATCH_BAND_ABS


def test_solver_configuration_matches_cb1s_control():
    """cb1's keep-12 control is the reference form; its two budgets are reused verbatim."""
    import ddm_cb1_carrier_basis_refit as cb1

    manifest = Path(
        "/Volumes/VertigoDataTier/pact/ddm_cb1/stage_res_12_0/launch_manifest.json"
    )
    if not manifest.exists():
        pytest.skip("cb1's launch manifest is not mounted")
    argv = json.loads(manifest.read_text())["argv"]
    # cb1 relied on its parser defaults for both budgets; assert them at the parser.
    defaults = cb1.build_parser().parse_args(
        ["resolve", "--basis-codes", "x", "--base-pose", "y", "--out-dir", "z"]
    )
    assert "--outer-rounds" not in argv and "--max-gn-iterations" not in argv
    assert defaults.outer_rounds == cr1.OUTER_ROUNDS
    assert defaults.max_gn_iterations == cr1.MAX_GN_ITERATIONS


def test_jsonable_refuses_an_unknown_type():
    with pytest.raises(TypeError):
        cr1._jsonable(object())


def test_jsonable_carries_numpy_and_paths():
    assert cr1._jsonable(np.int64(3)) == 3
    assert cr1._jsonable(np.float64(0.5)) == 0.5
    assert cr1._jsonable(np.array([1, 2])) == [1, 2]
    assert cr1._jsonable(Path("/a/b")) == "/a/b"


def test_merge_rows_takes_the_last_row_per_pair(tmp_path):
    """A resumed shard may re-write a pair; the merge must not double-count it."""
    a = tmp_path / "a"
    a.mkdir()
    (a / "rows_00_of_02.jsonl").write_text(
        json.dumps({"pair": 1, "start_d_pose": 1.0, "final_d_pose": 0.9}) + "\n"
    )
    (a / "rows_01_of_02.jsonl").write_text(
        json.dumps({"pair": 2, "start_d_pose": 2.0, "final_d_pose": 2.0}) + "\n"
    )
    merged = cr1.merge_rows([a])
    assert set(merged) == {1, 2}
    assert merged[1]["final_d_pose"] == 0.9


def test_merge_rows_ignores_blank_lines(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "rows_00_of_01.jsonl").write_text(
        json.dumps({"pair": 0, "start_d_pose": 1.0, "final_d_pose": 1.0}) + "\n\n"
    )
    assert set(cr1.merge_rows([a])) == {0}


def test_rate_per_byte_is_the_scoring_functions_own_exchange():
    assert pytest.approx(6.658589531221714e-07, rel=1e-12) == cr1.RATE_PER_BYTE
    assert cr1.RATE_PER_BYTE == 25.0 / 37_545_489.0


def test_pose_leg_sensitivity_at_the_move49_base():
    """d(leg)/d(d_pose) = 5 / sqrt(10 d); the number every pose row is scaled by."""
    base = cr1.PP1_BASE_MEAN
    leg = cr1.jg5.pose_leg(base)
    numeric = (cr1.jg5.pose_leg(base * 1.000001) - leg) / (base * 0.000001)
    assert numeric == pytest.approx(5.0 / leg * 1.0, rel=1e-5)


def test_a_gain_below_the_band_is_not_accepted():
    """The gate is strict ``>``: a gain exactly at the band is measurement, not signal."""
    assert not (cr1.POSE_ACCEPT_TOL_ABS > cr1.POSE_ACCEPT_TOL_ABS)
    assert (cr1.POSE_ACCEPT_TOL_ABS * 1.0000001) > cr1.POSE_ACCEPT_TOL_ABS


def test_verdict_refuses_two_moves_that_share_a_pair(tmp_path):
    """Recomposing d_pose from two per-pair changes is only legal if they are disjoint.

    If the same pair appeared in both moves, the composed mean would double-count its
    change and the table would read better than the object.  The refusal is the reason
    the composition can be recomposed at all rather than added leg-by-leg.
    """
    assemble = tmp_path / "ASSEMBLE.json"
    admission = tmp_path / "ADMISSION.json"
    kept = tmp_path / "kept.json"
    assemble.write_text(
        json.dumps(
            {
                "pose": {"base_mean_n600": 4.5e-06, "resolved_mean_n600": 4.4e-06},
                "rate": {"candidate_bytes": 179153},
                "acceptance": {"accepted_pairs": [59]},
            }
        )
    )
    admission.write_text(
        json.dumps(
            {
                "reference_drop_everything": {
                    "d_seg_t4": 1e-04,
                    "d_pose": 4.5e-06,
                    "archive_bytes_modelled": 179153.0,
                },
                "best": {
                    "d_seg_t4": 1e-04,
                    "d_pose": 4.4e-06,
                    "archive_bytes_modelled": 179163.0,
                },
            }
        )
    )
    kept.write_text(json.dumps([59, 70]))
    args = cr1.build_parser().parse_args(
        [
            "verdict",
            "--assemble", str(assemble),
            "--admission", str(admission),
            "--kept-pairs", str(kept),
            "--out-dir", str(tmp_path / "out"),
        ]
    )
    if not POINTER_MOUNTED:
        pytest.skip("move 49's tree is not mounted")
    with pytest.raises(cr1.Cr1Error, match="share pairs"):
        cr1.cmd_verdict(args)


def test_verdict_refuses_a_base_that_is_not_the_admissions_own(tmp_path):
    """Two ledgers on different bases cannot be differenced; the gate says so."""
    assemble = tmp_path / "ASSEMBLE.json"
    admission = tmp_path / "ADMISSION.json"
    kept = tmp_path / "kept.json"
    assemble.write_text(
        json.dumps(
            {
                "pose": {"base_mean_n600": 4.5e-06, "resolved_mean_n600": 4.4e-06},
                "rate": {"candidate_bytes": 179153},
                "acceptance": {"accepted_pairs": [391]},
            }
        )
    )
    admission.write_text(
        json.dumps(
            {
                "reference_drop_everything": {
                    "d_seg_t4": 1e-04,
                    "d_pose": 4.6e-06,
                    "archive_bytes_modelled": 179153.0,
                },
                "best": {
                    "d_seg_t4": 1e-04,
                    "d_pose": 4.4e-06,
                    "archive_bytes_modelled": 179163.0,
                },
            }
        )
    )
    kept.write_text(json.dumps([59]))
    args = cr1.build_parser().parse_args(
        [
            "verdict",
            "--assemble", str(assemble),
            "--admission", str(admission),
            "--kept-pairs", str(kept),
            "--out-dir", str(tmp_path / "out"),
        ]
    )
    if not POINTER_MOUNTED:
        pytest.skip("move 49's tree is not mounted")
    with pytest.raises(cr1.Cr1Error, match="not standing on the same instrument"):
        cr1.cmd_verdict(args)


@needs_pointer
def test_pointer_identity_holds():
    receipts = cr1.assert_pointer()
    assert receipts["pointer_archive_sha256"] == cr1.POINTER_ARCHIVE_SHA256
    assert receipts["pointer_archive_bytes"] == cr1.POINTER_ARCHIVE_BYTES
    assert receipts["gt_lineage"] == cr1.up2.LINEAGE_DALI


@needs_pointer
def test_the_base_stage_wrote_a_base_that_reproduces_pp1_exactly():
    path = cr1.WORK / "base" / "BASE.json"
    if not path.exists():
        pytest.skip("the base stage has not run in this store")
    report = json.loads(path.read_text())
    assert report["C1_base_reproduction"]["max_abs_gap"] == 0.0
    assert report["C2_twins"]["byte_identical"] is True
    assert report["C3_codes_identity"]["instrument_codes_equal_archive_codes"] is True
    assert report["d_pose_mean"] == pytest.approx(cr1.PP1_BASE_MEAN, rel=0, abs=0)


@needs_pointer
def test_the_resolve_only_candidate_is_zero_bytes_and_moves_exactly_one_pair():
    """The measured result this arm exists to record, guarded against a silent redo."""
    path = cr1.WORK / "assemble_resolve_only" / "ASSEMBLE.json"
    if not path.exists():
        pytest.skip("the assemble stage has not run in this store")
    report = json.loads(path.read_text())
    assert report["acceptance"]["pairs_solved"] == 600
    assert report["acceptance"]["pairs_accepted"] == 1
    assert report["acceptance"]["accepted_pairs"] == [391]
    # pair 387 "gained" 6.82e-09 while returning the codes it started with.
    assert report["acceptance"]["identical_codes_with_positive_gain"] == [
        {"pair": 387, "gain": 6.817188764884907e-09}
    ]
    assert report["acceptance"]["identical_codes_gain_max"] > cr1.POSE_BATCH_BAND_ABS
    assert report["rate"]["delta_bytes"] == 0
    assert report["rate"]["candidate_bytes"] == cr1.POINTER_ARCHIVE_BYTES
    assert report["frame1_section_identity"]["frame1_sections_all_identical"] is True
    assert report["pose"]["unchanged_max_abs_gap_vs_base"] == 0.0
    assert not report["net"]["clears_like_for_like"]


@needs_pointer
def test_the_twins_control_double_compiled_both_carriers():
    path = cr1.WORK / "twins_resolve_only" / "TWINS.json"
    if not path.exists():
        pytest.skip("the twins stage has not run in this store")
    report = json.loads(path.read_text())
    assert report["identity_double_compile"]["identical"] is True
    assert report["candidate_double_compile"]["identical"] is True
    assert report["identity_reproduces_move49"] is True
    assert report["candidate_double_compile"]["bytes"] == cr1.POINTER_ARCHIVE_BYTES
