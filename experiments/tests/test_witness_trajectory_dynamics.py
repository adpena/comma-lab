# SPDX-License-Identifier: MIT
"""Tests for the READ-ONLY witness trajectory-dynamics instrument
(tools/render_witness_trajectory_dynamics.py, landed 2026-06-27).

The TEMPORAL sister of the spatial stage-diff: it parses the level-set witness
verdict log(s) and extracts per-stage time-constants (descent rate / plateau /
dead-tail / time-to-demonstrate) that the A/B campaign uses to budget arm length.

These tests build SYNTHETIC verdict series with KNOWN dynamics and assert the
extractor recovers them, parse a real verdict log line, cover the empty /
single-verdict edge cases, and — when the real taualone log is present — assert
the tool reproduces the hand-measured anchors (tau time-to-best ~375 @ ~675,
dead-tail ~200, l7 descent ~2.8x tau) since the tool is the oracle for them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.render_witness_trajectory_dynamics import (  # noqa: E402
    assign_stage,
    compute_dynamics,
    compute_stage_dynamics,
    group_stages,
    merge_rows,
    parse_verdicts,
    resolve_logs,
)

_REAL_TAUALONE = _REPO / ".omx/tmp/levelset_amort_deconf_taualone_20260627T194432Z.log"


def _v(epoch, d_seg, seg_form=None, d_pose=None, blob_bytes=70000, implied_S=None):
    row = {"stage": "verdict", "epoch": epoch, "d_seg": d_seg, "blob_bytes": blob_bytes}
    if seg_form is not None:
        row["seg_form"] = seg_form
    if d_pose is not None:
        row["d_pose"] = d_pose
    if implied_S is not None:
        row["implied_S"] = implied_S
    return row


def _linear_stage(form, onset, step, dsegs, d_pose_const=0.001):
    """Build a stage of verdict rows with given d_seg values at onset, onset+step,…"""
    return [
        _v(onset + i * step, dseg, seg_form=form, d_pose=d_pose_const)
        for i, dseg in enumerate(dsegs)
    ]


# ───────────────────────── parse + merge ─────────────────────────
def test_parse_real_verdict_line(tmp_path):
    """A real verdict JSON line parses to the expected fields."""
    real = (
        '{"stage": "verdict", "epoch": 675, "seg_form": "tau_softplus", '
        '"d_seg": 0.003755, "d_pose": 0.000602, "blob_bytes": 74000, '
        '"implied_S": 0.5017, "async": true}'
    )
    p = tmp_path / "real.log"
    p.write_text("some preamble\n" + real + "\n")
    rows = parse_verdicts(p)
    assert len(rows) == 1
    assert rows[0]["epoch"] == 675 and rows[0]["seg_form"] == "tau_softplus"
    assert rows[0]["d_seg"] == pytest.approx(0.003755)


def test_parse_skips_nonverdict_and_malformed(tmp_path):
    p = tmp_path / "mixed.log"
    p.write_text(
        "\n".join(
            [
                '[durable-daemon] launching: ...',
                '{"stage": "gt", "n_pairs": 200}',
                '{"stage": "verdict", "epoch": 0, "d_seg": 0.5, "blob_bytes": 1}',
                '{"stage": "verdict", "epoch": BROKEN',  # malformed JSON
                '{"stage": "verdict", "d_seg": 0.1}',  # no epoch
                '{"stage": "verdict", "epoch": 25, "seg_form": "ce", "d_seg": 0.02, "blob_bytes": 2}',
            ]
        )
    )
    rows = parse_verdicts(p)
    assert [r["epoch"] for r in rows] == [0, 25]


def test_parse_missing_file_returns_empty(tmp_path):
    assert parse_verdicts(tmp_path / "nope.log") == []


def test_merge_rows_last_log_wins_per_epoch():
    a = [_v(300, 0.0056, "tau_softplus"), _v(325, 0.0061, "tau_softplus")]
    b = [_v(325, 0.0049, "tau_softplus"), _v(350, 0.0048, "tau_softplus")]
    merged = merge_rows([a, b])
    assert [r["epoch"] for r in merged] == [300, 325, 350]
    # epoch 325 overlap: later list (b) wins
    assert merged[1]["d_seg"] == pytest.approx(0.0049)


# ───────────────────────── stage assignment ─────────────────────────
def test_assign_stage_uses_seg_form_when_present():
    assert assign_stage(_v(500, 0.004, "tau_softplus"), 300, 900, None) == "tau_softplus"


def test_assign_stage_infers_formless_from_curriculum():
    assert assign_stage(_v(0, 0.5), 300, 900, None) == "ce"
    assert assign_stage(_v(500, 0.004), 300, 900, None) == "tau_softplus"
    assert assign_stage(_v(950, 0.003), 300, 900, None) == "l7_softplus"


def test_assign_stage_muon_relabel():
    assert assign_stage(_v(1100, 0.003, "l7_softplus"), 300, 900, 1000) == "muon"
    assert assign_stage(_v(950, 0.003, "l7_softplus"), 300, 900, 1000) == "l7_softplus"


def test_group_stages_chronological_order():
    rows = (
        _linear_stage("l7_softplus", 900, 25, [0.0044, 0.0039])
        + _linear_stage("ce", 0, 25, [0.5, 0.02])
        + _linear_stage("tau_softplus", 300, 25, [0.0059, 0.0050])
    )
    groups = group_stages(rows, 300, 900, None)
    assert list(groups.keys()) == ["ce", "tau_softplus", "l7_softplus"]


# ───────────────────────── per-stage dynamics ─────────────────────────
def test_known_descent_rate():
    stage = _linear_stage("tau_softplus", 100, 10, [0.010, 0.008, 0.006, 0.004])
    d = compute_stage_dynamics("tau_softplus", stage, plateau_k=3, demonstrate_frac=0.5)
    assert d["onset_epoch"] == 100 and d["best_epoch"] == 130
    assert d["time_to_best"] == 30
    # Delta d_seg/epoch = (0.004 - 0.010)/30 = -2.0e-4
    assert d["descent_rate_dseg_per_epoch"] == pytest.approx(-2.0e-4)
    assert d["improvement_per_epoch_dseg"] == pytest.approx(2.0e-4)


def test_known_plateau_and_dead_tail():
    # improves to best@130, then 3 non-improving verdicts -> plateau@130
    stage = _linear_stage(
        "tau_softplus", 100, 10, [0.010, 0.008, 0.006, 0.004, 0.0041, 0.0042, 0.0043]
    )
    d = compute_stage_dynamics("tau_softplus", stage, plateau_k=3, demonstrate_frac=0.5)
    assert d["best_epoch"] == 130
    assert d["plateau_onset_epoch"] == 130
    assert d["dead_tail_epochs"] == 30  # 160 - 130
    assert d["dead_tail_verdicts"] == 3
    assert d["dead_tail_net_delta_dseg"] == pytest.approx(0.0003)  # 0.0043 - 0.004


def test_plateau_not_fired_when_streak_below_k():
    # only 2 non-improving verdicts after best -> no plateau at K=3
    stage = _linear_stage("tau_softplus", 100, 10, [0.010, 0.006, 0.004, 0.0041, 0.0042])
    d = compute_stage_dynamics("tau_softplus", stage, plateau_k=3, demonstrate_frac=0.5)
    assert d["plateau_onset_epoch"] is None
    assert d["dead_tail_epochs"] == 0


def test_known_time_to_demonstrate():
    # start 0.010, best 0.004, total drop 0.006, 50% target = 0.007;
    # first epoch <= 0.007 is 0.006 @ ep120 -> 20 epochs into the stage
    stage = _linear_stage("tau_softplus", 100, 10, [0.010, 0.008, 0.006, 0.004])
    d = compute_stage_dynamics("tau_softplus", stage, plateau_k=3, demonstrate_frac=0.5)
    assert d["time_to_demonstrate_dseg"] == 20


def test_volatility_band_post_best():
    stage = _linear_stage("tau_softplus", 100, 10, [0.010, 0.004, 0.0042, 0.0044])
    d = compute_stage_dynamics("tau_softplus", stage, plateau_k=3, demonstrate_frac=0.5)
    # post-best (epoch>=110): [0.004, 0.0042, 0.0044]
    assert d["vol_post_best_min"] == pytest.approx(0.004)
    assert d["vol_post_best_max"] == pytest.approx(0.0044)
    assert d["vol_post_best_std"] > 0.0


def test_net_deltas_seg_and_pose():
    stage = [
        _v(100, 0.010, "tau_softplus", d_pose=0.002),
        _v(110, 0.006, "tau_softplus", d_pose=0.0015),
        _v(120, 0.005, "tau_softplus", d_pose=0.001),
    ]
    d = compute_stage_dynamics("tau_softplus", stage, plateau_k=3, demonstrate_frac=0.5)
    assert d["net_delta_dseg"] == pytest.approx(-0.005)  # 0.005 - 0.010
    assert d["net_delta_dpose"] == pytest.approx(-0.001)  # 0.001 - 0.002


def test_single_verdict_stage_edge_case():
    stage = [_v(299, 0.0059, "ce")]
    d = compute_stage_dynamics("ce", stage, plateau_k=3, demonstrate_frac=0.5)
    assert d["n_verdicts"] == 1
    assert d["best_epoch"] == 299 and d["time_to_best"] == 0
    assert d["descent_rate_dseg_per_epoch"] is None  # no productive window
    assert d["plateau_onset_epoch"] is None
    assert d["time_to_demonstrate_dseg"] is None


# ───────────────────────── cross-stage + top level ─────────────────────────
def test_cross_stage_ranking_orders_by_descent():
    rows = (
        _linear_stage("tau_softplus", 300, 100, [0.006, 0.0055])  # slow: 5e-6/ep
        + _linear_stage("l7_softplus", 900, 50, [0.0044, 0.00362])  # fast: ~1.56e-5/ep
    )
    dyn = compute_dynamics(rows, tau=300, l7=900, muon=None, plateau_k=3, demonstrate_frac=0.5)
    ranking = dyn["cross_stage_ranking"]
    assert ranking[0]["stage"] == "l7_softplus"
    assert ranking[1]["stage"] == "tau_softplus"
    assert ranking[0]["ratio_to_slowest"] > 1.0


def test_compute_dynamics_empty_rows():
    dyn = compute_dynamics([], tau=300, l7=900, muon=None, plateau_k=3, demonstrate_frac=0.5)
    assert dyn["n_verdicts"] == 0
    assert dyn["stages"] == []
    assert dyn["cross_stage_ranking"] == []
    assert dyn["epoch_range"] is None


# ───────────────────────── resolve_logs ─────────────────────────
def test_resolve_logs_explicit_wins(tmp_path):
    a = tmp_path / "a.log"
    a.write_text("x")
    out = resolve_logs([str(a)], ".omx/tmp/levelset_amort_*.log")
    assert out == [a]


def test_resolve_logs_glob_newest_verdict_bearing(tmp_path):
    old = tmp_path / "old.log"
    new = tmp_path / "new.log"
    nonv = tmp_path / "nonverdict.log"
    old.write_text('{"stage": "verdict", "epoch": 0, "d_seg": 0.5, "blob_bytes": 1}')
    new.write_text('{"stage": "verdict", "epoch": 25, "d_seg": 0.02, "blob_bytes": 1}')
    nonv.write_text('{"stage": "gt", "n_pairs": 200}')
    import os
    import time

    t = time.time()
    os.utime(old, (t - 100, t - 100))
    os.utime(new, (t - 10, t - 10))
    os.utime(nonv, (t, t))  # newest mtime but NOT verdict-bearing -> must be skipped
    out = resolve_logs(None, str(tmp_path / "*.log"))
    assert out == [new]


# ───────────────────────── real-log anchor oracle ─────────────────────────
@pytest.mark.skipif(not _REAL_TAUALONE.exists(), reason="real taualone log not present")
def test_real_log_reproduces_hand_measured_anchors():
    rows = parse_verdicts(_REAL_TAUALONE)
    assert rows, "expected verdict rows in the real taualone log"
    dyn = compute_dynamics(rows, tau=300, l7=900, muon=None, plateau_k=3, demonstrate_frac=0.5)
    ac = dyn["anchor_check"]
    # tau: ~375 epochs to best @ ~675, ~200-ep dead tail
    assert ac["tau_time_to_best"] == 375
    assert ac["tau_best_epoch"] == 675
    assert ac["tau_dead_tail_epochs"] == 200
    assert ac["tau_time_to_best_matches_375"] is True
    assert ac["tau_dead_tail_matches_200"] is True
    # l7 descent ~2.8x tau
    assert ac["l7_over_tau_descent_ratio"] == pytest.approx(2.77, abs=0.2)
    assert ac["l7_descent_ratio_matches_2p8"] is True
