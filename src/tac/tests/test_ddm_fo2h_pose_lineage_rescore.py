"""Guard the fo2h PyAV-vs-DALI pose re-scorer.

The instrument exists to answer one question: does the pose-null seg edit's measured pose
degradation (aggregate ratio 1.3725 on PyAV GT) survive onto the DALI GT lineage the contest
actually scores?  Everything that could make its answer a lie is pinned here:

* the aggregate must be a ratio of MEANS (the `evaluate.py` aggregation), never a mean of ratios;
* the eta-gate control must FAIL when the PyAV column drifts, because that column is the only
  evidence the instrument is measuring the same object as the gate;
* the sign-flip verdict must key off which side of 1.0 each aggregate lands on;
* a non-DALI cache must be refused rather than silently scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[3] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

rs = pytest.importorskip("ddm_fo2h_pose_lineage_rescore")


def _row(pair: int, pb: float, pa: float, db: float, da: float) -> dict:
    return {"pair": pair,
            "pyav_d_pose_before": pb, "pyav_d_pose_after": pa, "pyav_ratio": pa / pb,
            "dali_d_pose_before": db, "dali_d_pose_after": da, "dali_ratio": da / db}


# --------------------------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------------------------
def test_aggregate_is_ratio_of_means_not_mean_of_ratios() -> None:
    """One heavy pair improving slightly must outweigh a light pair worsening hugely."""
    rows = [_row(0, 100.0, 90.0, 1.0, 1.0), _row(1, 1.0, 10.0, 1.0, 1.0)]
    # ratio of means = (90+10)/(100+1) = 0.990 ; mean of ratios = (0.9 + 10)/2 = 5.45
    assert rs.aggregate(rows, "pyav") == pytest.approx(100.0 / 101.0)


def test_aggregate_handles_the_measured_direction() -> None:
    rows = [_row(0, 1.0e-4, 1.4e-4, 7.0e-6, 9.0e-6)]
    assert rs.aggregate(rows, "pyav") == pytest.approx(1.4)
    assert rs.aggregate(rows, "dali") == pytest.approx(9.0 / 7.0)


def test_aggregate_zero_denominator_is_nan_not_a_crash() -> None:
    """A degenerate pair must not take the whole summary down, and must not read as 1.0."""
    import math
    rows = [{"pyav_d_pose_before": 0.0, "pyav_d_pose_after": 0.0}]
    assert math.isnan(rs.aggregate(rows, "pyav"))


# --------------------------------------------------------------------------------------------
# the eta-gate control -- the receipt that this is the same object
# --------------------------------------------------------------------------------------------
def _gate_file(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "ETA_GATE_ROWS.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_control_passes_when_pyav_column_reproduces_the_gate(tmp_path: Path) -> None:
    gate = _gate_file(tmp_path, [{"pair": 5, "d_pose_before": 1.0e-4, "d_pose_after": 1.4e-4}])
    out = rs.control_vs_eta_gate([_row(5, 1.0e-4, 1.4e-4, 7e-6, 9e-6)], gate)
    assert out["reproduces_eta_gate"] is True
    assert out["n_compared"] == 1


def test_control_FAILS_on_a_drifted_pyav_column(tmp_path: Path) -> None:
    """A 1% drift means the geometry differs; the instrument must refuse to look trustworthy."""
    gate = _gate_file(tmp_path, [{"pair": 5, "d_pose_before": 1.0e-4, "d_pose_after": 1.4e-4}])
    out = rs.control_vs_eta_gate([_row(5, 1.01e-4, 1.4e-4, 7e-6, 9e-6)], gate)
    assert out["reproduces_eta_gate"] is False
    assert out["worst_rel_err_before"] > 1e-3


def test_control_reports_unchecked_when_no_gate_rows_exist(tmp_path: Path) -> None:
    out = rs.control_vs_eta_gate([_row(5, 1e-4, 1.4e-4, 7e-6, 9e-6)],
                                 tmp_path / "absent.jsonl")
    assert out["checked"] is False


def test_control_with_no_overlapping_pairs_does_not_claim_reproduction(tmp_path: Path) -> None:
    gate = _gate_file(tmp_path, [{"pair": 99, "d_pose_before": 1e-4, "d_pose_after": 1.4e-4}])
    out = rs.control_vs_eta_gate([_row(5, 1e-4, 1.4e-4, 7e-6, 9e-6)], gate)
    assert out["n_compared"] == 0
    assert out["reproduces_eta_gate"] is False


# --------------------------------------------------------------------------------------------
# retained-frame discovery
# --------------------------------------------------------------------------------------------
def test_retained_pairs_parses_and_sorts_pair_ids(tmp_path: Path) -> None:
    for name in ("cam_edit_pair0015.npy", "cam_edit_pair0003.npy", "cam_edit_pair0134.npy"):
        (tmp_path / name).touch()
    (tmp_path / "not_a_frame.npy").touch()
    assert rs.retained_pairs(tmp_path) == [3, 15, 134]


def test_retained_pairs_empty_dir_is_empty(tmp_path: Path) -> None:
    assert rs.retained_pairs(tmp_path) == []


# --------------------------------------------------------------------------------------------
# fail-closed contracts
# --------------------------------------------------------------------------------------------
def test_open_raw_refuses_a_wrong_sized_file(tmp_path: Path) -> None:
    p = tmp_path / "0.raw"
    p.write_bytes(b"\x00" * 1024)
    with pytest.raises(rs.Fo2hLineageError, match="expected"):
        rs.open_raw(p)


def test_open_raw_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(rs.Fo2hLineageError, match="does not exist"):
        rs.open_raw(tmp_path / "nope.raw")


def test_frame_geometry_matches_the_camera_lattice() -> None:
    """874x1164x3 is the camera frame the eta gate edits; a mismatch silently rescales pose."""
    assert (rs.H, rs.W, rs.C) == (874, 1164, 3)
    assert rs.FRAME_B == 874 * 1164 * 3
    assert rs.N_PAIRS == 600


def test_default_gt_cache_is_the_dali_one_not_the_av_one() -> None:
    """up1's CLI defaulted to the AV cache and that is the phantom-19x trap; ours must not."""
    assert "dali" in rs.DEFAULT_GT_DALI.name.lower()
    assert "gt_cache_av" not in rs.DEFAULT_GT_DALI.name.lower()


def test_identity_edit_is_degenerate_not_a_sign_flip() -> None:
    """The identity control (`cam_edit = dec1`) gives 1.0/1.0; calling that a FLIP is backwards.

    This is the bug the identity control actually caught: a bare product-of-signs test returns 0,
    which is not > 0, so it fell through to SIGN FLIPS.
    """
    assert rs.lineage_verdict(1.0, 1.0) == "DEGENERATE-NO-POSE-CHANGE"


def test_lineage_verdict_agrees_when_both_worsen() -> None:
    assert "both WORSEN" in rs.lineage_verdict(1.3725, 1.12)


def test_lineage_verdict_agrees_when_both_improve() -> None:
    assert "both IMPROVE" in rs.lineage_verdict(0.79, 0.93)


def test_lineage_verdict_flags_the_flip_this_arm_is_hunting() -> None:
    """PyAV says the edit hurts pose, DALI says it helps -- the case that would rescue the channel."""
    assert rs.lineage_verdict(1.3725, 0.88) == "SIGN FLIPS ACROSS LINEAGES"


def test_lineage_verdict_nan_is_undetermined_never_agreement() -> None:
    assert rs.lineage_verdict(float("nan"), 1.2) == "UNDETERMINED-NAN-AGGREGATE"


def test_default_raw_matches_the_eta_gate_raw() -> None:
    """The eta-gate reproduction control is only meaningful if both read the SAME decode.

    A first draft of this module pointed at a plausible-looking but different raw path; the run
    failed closed on a missing file, but had that path existed it would have produced a silently
    different `dec0`/`dec1` and a broken control.
    """
    gate = pytest.importorskip("ddm_rt1_eta_gate_pose_constrained")
    assert rs.DEFAULT_RAW == gate.DEFAULT_RAW
