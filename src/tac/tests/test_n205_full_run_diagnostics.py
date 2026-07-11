# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("n205_diag", ROOT / "tools/n205_full_run_diagnostics.py")
assert SPEC and SPEC.loader
diag = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diag
SPEC.loader.exec_module(diag)


def verdict(epoch: int, road: float, undriv: float, d_seg: float = 0.03, d_pose: float = 10.0):
    per = [road, 0.2, undriv, 0.01, 0.001]
    return {"stage": "verdict", "epoch": epoch, "d_seg": d_seg, "d_pose": d_pose,
            "d_seg_by_class": per, "flip_share_by_class": per, "seg_form": "unify_tau"}


def annulus(epoch: int, road: float, undriv: float):
    return {"stage": "annulus_convergence", "epoch": epoch,
            "threshold": {"per_class_annulus_flip_frac": {
                "0": road, "1": 0.2, "2": undriv, "3": 0.01, "4": 0.001}}}


def reversal_rows(post_values):
    rows = []
    for ep, v in zip(diag.EP450_PRE, (0.10, 0.11, 0.12), strict=True):
        rows.extend((verdict(ep, v, v), annulus(ep, v, v)))
    for ep, v in zip(diag.EP450_POST, post_values, strict=True):
        rows.extend((verdict(ep, v, v), annulus(ep, v, v)))
    return rows


def test_ep450_pass_requires_mean_drop_and_negative_slope():
    got = diag.ep450_reversal(reversal_rows((0.10, 0.09, 0.08)))
    assert got["status"] == "PASS_REVERSAL"


def test_ep450_fail_is_implementation_scoped():
    got = diag.ep450_reversal(reversal_rows((0.12, 0.13, 0.14)))
    assert got["status"] == "FAIL_IMPLEMENTATION_FALSIFIED"
    assert "NOT family/paradigm" in got["verdict_scope_on_fail"]


def test_ep450_missing_rows_stays_pending():
    got = diag.ep450_reversal(reversal_rows((0.10, 0.09, 0.08))[:-2])
    assert got["status"] == "PRE_REGISTERED_PENDING"
    assert 525 in got["missing_epochs"]


def basin_rows(values, p10=0.03, cond=90_000.0, start=694):
    return [{"stage": "jacobian_basin", "epoch": start + 4 * i,
             "median_sigma_min": value, "p10_sigma_min": p10, "median_cond": cond}
            for i, value in enumerate(values)]


def test_pose_ready_from_last_eight_preboundary_rows():
    rows = basin_rows([0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108])
    got = diag.pose_finish_readiness(rows)
    assert got["status"] == "READY"


def test_pose_degenerate_collapse_guard():
    rows = basin_rows([0.14, 0.14, 0.14, 0.14, 0.05, 0.05, 0.05, 0.05])
    got = diag.pose_finish_readiness(rows)
    assert got["status"] == "DEGENERATE_NOT_READY"
    assert got["last4_over_prior4"] < 0.75


def test_stage_attribution_reports_per_class_delta():
    rows = [verdict(425, 0.12, 0.06, 0.04), annulus(425, 0.5, 0.4),
            verdict(475, 0.10, 0.05, 0.03), annulus(475, 0.4, 0.3)]
    item = diag.stage_attribution(rows)["boundaries"][0]
    assert item["boundary_epoch"] == 450
    assert np.isclose(item["delta_d_seg"], -0.01)
    assert np.isclose(item["per_class_delta_within_flip"]["Road"], -0.02)


def test_post_pose_requires_pose_gain_without_seg_harm():
    rows = basin_rows([0.101] * 8)
    rows.extend((verdict(725, 0.1, 0.1, d_seg=0.03, d_pose=10.0),
                 verdict(825, 0.1, 0.1, d_seg=0.0305, d_pose=8.5)))
    got = diag.pose_finish_readiness(rows)
    assert got["post_switch"]["status"] == "SUCCESS"


def test_refuse_any_output_beneath_run_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    inside = run / "diagnostic.json"
    try:
        diag.refuse_run_write(inside, run, "--out-json")
    except ValueError as exc:
        assert "sacred read-only run dir" in str(exc)
    else:
        raise AssertionError("write beneath run dir was not refused")
