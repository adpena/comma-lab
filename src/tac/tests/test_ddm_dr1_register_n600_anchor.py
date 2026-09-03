# SPDX-License-Identifier: MIT
"""Tests for the ddm_dr1 n600 delta_R anchor builder.

These pin the two things that are easy to get silently wrong: the pre-registered
falsifier arithmetic (an off-by-a-sign here would report a fired falsifier as
passed) and the residual semantics (putting the n96->n600 input change into
``EmpiricalAnchor.residual`` would poison the law's posterior with a law error
that does not exist).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "experiments" / "ddm_dr1_register_n600_anchor.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("_dr1_anchor", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_script()


@pytest.fixture
def report():
    # delta_R chosen so the DERIVED minimum integer headroom is 2:
    # ceil(0.0371 / 0.0196) == 2.
    return {
        "measurement": "delta_R_noise_floor",
        "gt_npz": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        "n_frames": 600,
        "band": 1.0,
        "annulus_area_frac": 0.0256,
        "delta_R": 0.0196,
        "cross_check_full_R_vs_gt_direct": {"annulus": {"p95": 0.0371}},
    }


@pytest.fixture
def receipts():
    return {
        "torch_num_threads": 4,
        "per_class_annulus_pooled": {
            "Road": {"n_px": 10, "p95": 0.02},
            "Lane": {"n_px": 5, "p95": 0.01},
            "Undrivable": {"n_px": 5, "p95": 0.015},
            "Movable": {"n_px": 2, "p95": 0.03},
            "MyCar": None,
            # degenerate: an all-zero perturbation must not crash registration
            "Degenerate": {"n_px": 1, "p95": 0.0},
        },
        "sub_band_sensitivity": {
            "band_1": {"n_px": 22, "p95": 0.0196},
            "band_0.5": {"n_px": 11, "p95": 0.018},
            "band_0.25": None,
        },
    }


# ------------------------------------------------------------------- falsifier


def test_falsifier_passes_inside_the_band(mod):
    v = mod.falsifier_verdict(mod.N96_DELTA_R)
    assert v["falsifier_fired"] is False
    assert v["relative_deviation"] == 0.0
    assert v["ratio_n600_over_n96"] == 1.0


@pytest.mark.parametrize("scale", [1.099, 0.901, 1.0, 1.05, 0.95])
def test_falsifier_does_not_fire_inside(mod, scale):
    assert mod.falsifier_verdict(mod.N96_DELTA_R * scale)["falsifier_fired"] is False


@pytest.mark.parametrize("scale", [1.101, 0.899, 2.0, 0.5])
def test_falsifier_fires_outside_in_both_directions(mod, scale):
    v = mod.falsifier_verdict(mod.N96_DELTA_R * scale)
    assert v["falsifier_fired"] is True
    assert (v["relative_deviation"] > 0) == (scale > 1.0)


def test_falsifier_band_matches_the_pre_registered_charter_numbers(mod):
    v = mod.falsifier_verdict(mod.N96_DELTA_R)
    lo, hi = v["band"]
    # charter (commit 4870d475c) pre-registered 0.01763-0.02155
    assert lo == pytest.approx(0.017631, abs=1e-6)
    assert hi == pytest.approx(0.021549, abs=1e-6)
    assert mod.FALSIFIER_REL_TOL == 0.10


# ---------------------------------------------------------------------- anchor


def test_anchor_residual_is_zero_because_the_law_is_exact_arithmetic(mod, report, receipts):
    anchor = mod.build_n600_anchor(
        report, receipts, report_path="reports/x.json", measurement_utc="2026-09-04T00:00:00Z"
    )
    assert anchor.residual == 0.0
    # the input change is recorded as DATA, not as law error
    assert anchor.empirical_output["prefix_bias_check"]["n600_delta_R"] == 0.0196


def test_anchor_derives_headroom_and_m_safe_from_the_report(mod, report, receipts):
    anchor = mod.build_n600_anchor(
        report, receipts, report_path="reports/x.json", measurement_utc="2026-09-04T00:00:00Z"
    )
    out = anchor.empirical_output
    assert out["derived_headroom"] == 2.0  # ceil(0.0371 / 0.0196)
    assert out["derived_m_safe"] == pytest.approx(2.0 * 0.0196, rel=1e-12)
    assert out["per_class_derived_m_safe"]["Lane"] == pytest.approx(2.0 * 0.01, rel=1e-12)
    assert out["per_class_derived_m_safe"]["MyCar"] is None
    assert out["per_class_derived_m_safe"]["Degenerate"] is None
    assert out["sub_band_delta_R"]["band_0.5"] == 0.018
    assert out["sub_band_delta_R"]["band_0.25"] is None
    assert out["per_class_annulus_p95"]["Degenerate"] == 0.0
    assert out["per_class_annulus_p95"]["MyCar"] is None


def test_anchor_id_does_not_collide_with_the_n96_anchor(mod, report):
    anchor = mod.build_n600_anchor(
        report, None, report_path="reports/x.json", measurement_utc="2026-09-04T00:00:00Z"
    )
    assert anchor.anchor_id == "margin_band_delta_r_noise_floor_n600_20260904"
    assert anchor.anchor_id != "margin_band_delta_r_noise_floor_n96_20260708"


def test_anchor_records_the_pyav_lineage_and_full_cohort(mod, report, receipts):
    anchor = mod.build_n600_anchor(
        report, receipts, report_path="reports/x.json",
        measurement_utc="2026-09-04T00:00:00Z", gt_npz_sha256="deadbeef",
    )
    assert anchor.inputs["n_frames"] == 600
    assert "not a prefix" in anchor.inputs["cohort"]
    assert "PyAV" in anchor.inputs["gt_frame_lineage"]
    assert anchor.inputs["gt_npz_sha256"] == "deadbeef"
    assert anchor.inputs["torch_num_threads"] == 4
    assert "NON-PROMOTABLE" in anchor.provenance.measurement_axis


def test_anchor_tolerates_missing_receipts(mod, report):
    anchor = mod.build_n600_anchor(
        report, None, report_path="reports/x.json", measurement_utc="2026-09-04T00:00:00Z"
    )
    assert anchor.empirical_output["per_class_annulus_p95"] == {}
    assert anchor.inputs["torch_num_threads"] is None
