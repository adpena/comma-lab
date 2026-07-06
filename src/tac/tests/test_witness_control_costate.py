"""Tests for tac.witness_control — the θ* costate shadow controller (task #303 Phase A).

Covers: estimator math (slopes + stderr propagation), analytic costates, refusal
discipline (UNIDENTIFIABLE over guessing), the canonical-classifier integration,
backtests on REAL #205 log rows (inline snippets + the real log when present),
the NEVER-REGRESS guard, JSONL emission, and the STRUCTURAL no-actuation invariant.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tac.witness_control.costate_estimator import (
    ANALYTIC,
    MEASURED,
    PARTIAL,
    UNIDENTIFIABLE,
    analytic_costates,
    chain_ds_depoch,
    cross_run_lever_costate,
    rollback_gain,
    slope_with_stderr,
    stage_epoch_costates,
    sweep_finite_difference,
    transition_jump_costate,
)
from tac.witness_control.shadow_controller import (
    ACTUATION,
    build_shadow_report,
    load_run_inputs,
    parse_launch_sh_flags,
    write_shadow_row,
)

REPO = Path(__file__).resolve().parents[3]
#: the completed #205 reference run (backtest substrate; real n600 advisory verdicts)
RUN_205 = REPO / "experiments/results/levelset_n600_witness_20260703T120444Z"

# ── REAL row snippets from the #205 run.log (verbatim values; the CE descent, the
#    tau-onset spike at ep300->325, and the sustained tau creep) ──
CE_ROWS = [
    {"stage": "verdict", "epoch": 200, "seg_form": "ce", "d_seg": 0.005088,
     "d_pose": 0.003081, "blob_bytes": 100526, "implied_S": 0.7513, "ep_loss": 521.239},
    {"stage": "verdict", "epoch": 225, "seg_form": "ce", "d_seg": 0.004964,
     "d_pose": 0.002489, "blob_bytes": 100233, "implied_S": 0.7209, "ep_loss": 517.161},
    {"stage": "verdict", "epoch": 250, "seg_form": "ce", "d_seg": 0.004835,
     "d_pose": 0.002761, "blob_bytes": 99760, "implied_S": 0.7161, "ep_loss": 514.665},
    {"stage": "verdict", "epoch": 275, "seg_form": "ce", "d_seg": 0.004763,
     "d_pose": 0.002447, "blob_bytes": 99709, "implied_S": 0.6991, "ep_loss": 513.461},
]
TAU_ROWS = [
    {"stage": "verdict", "epoch": 300, "seg_form": "tau_softplus", "d_seg": 0.004752,
     "d_pose": 0.002099, "blob_bytes": 99550, "implied_S": 0.6864, "ep_loss": 148.553},
    {"stage": "verdict", "epoch": 325, "seg_form": "tau_softplus", "d_seg": 0.005923,
     "d_pose": 0.003826, "blob_bytes": 99580, "implied_S": 0.8542, "ep_loss": 147.512},
    {"stage": "verdict", "epoch": 350, "seg_form": "tau_softplus", "d_seg": 0.006267,
     "d_pose": 0.002529, "blob_bytes": 99489, "implied_S": 0.8519, "ep_loss": 141.335},
    {"stage": "verdict", "epoch": 375, "seg_form": "tau_softplus", "d_seg": 0.006424,
     "d_pose": 0.002411, "blob_bytes": 99365, "implied_S": 0.8638, "ep_loss": 137.598},
    {"stage": "verdict", "epoch": 400, "seg_form": "tau_softplus", "d_seg": 0.006568,
     "d_pose": 0.002277, "blob_bytes": 98995, "implied_S": 0.8736, "ep_loss": 134.122},
]


# ─────────────────────────── slope math ───────────────────────────
def test_slope_exact_linear_zero_stderr():
    fit = slope_with_stderr([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    assert fit.slope == pytest.approx(2.0)
    assert fit.stderr == pytest.approx(0.0, abs=1e-12)
    assert fit.n == 4


def test_slope_noisy_has_positive_stderr():
    fit = slope_with_stderr([0, 1, 2, 3, 4], [0.0, 1.2, 1.8, 3.3, 3.7])
    assert fit.slope == pytest.approx(0.96, abs=0.05)
    assert fit.stderr is not None and fit.stderr > 0.0


def test_slope_insufficient_points_refuses_stderr():
    assert slope_with_stderr([1.0], [2.0]).stderr is None
    two = slope_with_stderr([1.0, 2.0], [2.0, 4.0])
    assert two.slope == pytest.approx(2.0)
    assert two.stderr is None  # n=2 < MIN_ROWS_FOR_SLOPE: no residual dof


def test_slope_zero_x_spread():
    fit = slope_with_stderr([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
    assert fit.slope == 0.0 and fit.stderr is None


# ─────────────────────────── analytic costates ───────────────────────────
def test_analytic_lambda_seg_and_rate_exact():
    by_name = {c.name: c for c in analytic_costates(0.0025)}
    assert by_name["lambda_d_seg"].value == 100.0
    assert by_name["lambda_d_seg"].status == ANALYTIC
    assert by_name["lambda_bytes"].value == pytest.approx(25.0 / 37_545_489.0)


def test_analytic_lambda_pose_at_operating_point():
    by_name = {c.name: c for c in analytic_costates(0.0025)}
    # 5/sqrt(10*0.0025) = 5/sqrt(0.025) = 31.6227...
    assert by_name["lambda_d_pose"].value == pytest.approx(5.0 / math.sqrt(0.025))
    assert by_name["lambda_d_pose"].status == ANALYTIC


def test_analytic_lambda_pose_unidentifiable_at_zero():
    by_name = {c.name: c for c in analytic_costates(0.0)}
    assert by_name["lambda_d_pose"].status == UNIDENTIFIABLE
    assert by_name["lambda_d_pose"].value is None
    by_name = {c.name: c for c in analytic_costates(None)}
    assert by_name["lambda_d_pose"].status == UNIDENTIFIABLE


# ─────────────────────── trajectory costates (real snippets) ───────────────────────
def test_stage_epoch_costates_ce_descent():
    fits = stage_epoch_costates(CE_ROWS, "ce")
    assert fits["d_seg"].slope < 0.0            # CE was descending (measured)
    assert fits["d_seg"].stderr is not None
    assert fits["ep_loss"].slope < 0.0


def test_chain_ds_depoch_measured_with_band():
    fits = stage_epoch_costates(CE_ROWS, "ce")
    est = chain_ds_depoch(fits, d_pose_latest=0.002447, stage="ce", evidence=("t",))
    assert est.status == MEASURED
    assert est.value is not None and est.value < 0.0   # improving S per epoch
    band = est.band()
    assert band is not None and band[0] < est.value < band[1]


def test_chain_ds_depoch_unidentifiable_two_rows():
    fits = stage_epoch_costates(CE_ROWS[:2], "ce")
    est = chain_ds_depoch(fits, 0.0025, "ce", ("t",))
    assert est.status == UNIDENTIFIABLE
    assert est.value is None
    assert any("need" in e for e in est.evidence)


def test_transition_jump_ce_to_tau_positive():
    est = transition_jump_costate(CE_ROWS + TAU_ROWS, "ce", "tau_softplus")
    assert est.status == MEASURED
    # jump = 100*(0.004752 - 0.004763) = -0.0011 in S units (tau's FIRST verdict was
    # still below CE-last; the spike shows at ep325). Verify the exact arithmetic.
    assert est.value == pytest.approx(100.0 * (0.004752 - 0.004763), abs=1e-9)


def test_transition_jump_missing_stage_unidentifiable():
    est = transition_jump_costate(CE_ROWS, "ce", "l7_softplus")
    assert est.status == UNIDENTIFIABLE
    assert est.value is None


def test_rollback_gain_measured_on_tau_creep():
    rows = CE_ROWS + TAU_ROWS
    est = rollback_gain(rows)
    assert est.status == MEASURED
    # best d_seg = ep300 (0.004752, S 0.6864); latest = ep400 (S 0.8736)
    assert est.value == pytest.approx(0.8736 - 0.6864, abs=1e-6)
    assert any("ep300" in e for e in est.evidence)


def test_rollback_gain_insufficient_rows():
    assert rollback_gain(CE_ROWS[:1]).status == UNIDENTIFIABLE
    assert rollback_gain([]).status == UNIDENTIFIABLE


# ─────────────────────── cross-run A/B honesty ───────────────────────
def test_cross_run_single_lever_measured():
    est = cross_run_lever_costate({"lane-prior-phi1-mode": ("replace", "paint")},
                                  "runA", "runB", d_seg_a=0.0068, d_seg_b=0.0059,
                                  epoch=75)
    assert est.status == MEASURED
    assert est.value == pytest.approx(100.0 * (0.0059 - 0.0068))


def test_cross_run_confounded_unidentifiable():
    est = cross_run_lever_costate(
        {"lane-prior-phi1-mode": ("replace", "paint"), "mod-dim": ("32", "19")},
        "205", "fresh", d_seg_a=0.0068, d_seg_b=0.1275, epoch=75)
    assert est.status == UNIDENTIFIABLE
    assert est.value is None
    assert any("single-lever A/B" in e for e in est.evidence)


def test_sweep_partial_status_focal_exemplar():
    # focal γ-sweep exemplar (ep75 island grad-share; focal_boundary_calibration_20260705)
    pts = [(0.0, 0.0354), (0.5, 0.0354), (1.0, 0.0350), (2.0, 0.0339), (3.0, 0.0328)]
    est = sweep_finite_difference("dgrad_share_dgamma[island,ep75]", pts,
                                  units="grad-share per γ", evidence=("focal memo",),
                                  chained_to_S=False)
    assert est.status == PARTIAL          # internal observable: chain to S NOT measured
    assert est.value is not None and est.value < 0.0   # anti-island at ep75 (measured)


def test_sweep_insufficient_points():
    est = sweep_finite_difference("x", [(1.0, 2.0)], "u", ("e",))
    assert est.status == UNIDENTIFIABLE


# ─────────────────────── launch.sh flag parsing ───────────────────────
def test_parse_launch_flags_basic():
    flags = parse_launch_sh_flags(
        "PYTHONPATH=src .venv/bin/python trainer.py \\\n"
        "  --mod-dim 19 --seed-islands --eikonal-weight=0.05 \\\n"
        "  --lane-prior-phi1-mode paint --closed-loop-control\n")
    assert flags["mod-dim"] == "19"
    assert flags["seed-islands"] is True
    assert flags["eikonal-weight"] == "0.05"
    assert flags["lane-prior-phi1-mode"] == "paint"
    assert flags["closed-loop-control"] is True


def test_parse_launch_flags_parity_with_dashboard():
    """Regression: the replica must match tools/render_levelset_dashboard's parser on
    a REAL launch.sh (that module imports matplotlib, hence test-time-only load)."""
    launch = RUN_205 / "launch.sh"
    if not launch.is_file():
        pytest.skip("reference run launch.sh not present on this checkout")
    try:
        import importlib
        import sys as _sys
        tools_dir = str(REPO / "tools")
        if tools_dir not in _sys.path:
            _sys.path.insert(0, tools_dir)
        rld = importlib.import_module("render_levelset_dashboard")
    except ImportError as exc:  # matplotlib absent etc.
        pytest.skip(f"dashboard unimportable here: {exc}")
    text = launch.read_text(errors="replace")
    assert parse_launch_sh_flags(text) == rld._parse_launch_sh_flags(text)


# ─────────────────────── shadow controller on synthetic run dirs ───────────────────────
def _make_run_dir(tmp_path: Path, rows: list[dict], flags_line: str = "--mod-dim 19") -> Path:
    d = tmp_path / "run"
    d.mkdir()
    (d / "run.log").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "launch.sh").write_text(f".venv/bin/python trainer.py {flags_line}\n")
    return d


def test_shadow_report_converging_continue(tmp_path):
    d = _make_run_dir(tmp_path, CE_ROWS)
    rep = build_shadow_report(load_run_inputs(d))
    assert rep.classification["classification"] == "converging"
    actions = [r["action"] for r in rep.recommendations]
    assert "CONTINUE_STAGE" in actions
    top = rep.recommendations[0]
    assert top["predicted_dS"] < 0.0
    assert top["evidence"]                      # Rudin readback: never silent
    assert top["predicted_dS_band"] is not None


def test_shadow_report_diverging_rollback(tmp_path):
    d = _make_run_dir(tmp_path, CE_ROWS + TAU_ROWS)
    rep = build_shadow_report(load_run_inputs(d))
    assert rep.classification["classification"] == "diverging_erasing"
    actions = [r["action"] for r in rep.recommendations]
    assert "ROLLBACK_TO_BEST_CHECKPOINT" in actions
    rb = next(r for r in rep.recommendations if r["action"] == "ROLLBACK_TO_BEST_CHECKPOINT")
    assert rb["predicted_dS"] == pytest.approx(-(0.8736 - 0.6864), abs=1e-6)


def test_shadow_as_of_truncation(tmp_path):
    d = _make_run_dir(tmp_path, CE_ROWS + TAU_ROWS)
    rep = build_shadow_report(load_run_inputs(d, as_of_epoch=325))
    assert rep.epoch_latest == 325
    # 2 tau verdicts into the stage => recoverable-transient WATCH, not erosion
    assert rep.classification["classification"] == "transition_transient"
    assert any(r["action"] == "WATCH_NO_ACTION" for r in rep.recommendations)


def test_never_regress_refuses_positive_ds(tmp_path):
    # d_seg gently converging BUT d_pose rising hard -> chained dS/dep central > 0:
    # CONTINUE_STAGE must be REFUSED by the POWERPLAY guard, not ranked.
    rows = [
        {"stage": "verdict", "epoch": 0, "seg_form": "ce", "d_seg": 0.010000,
         "d_pose": 0.001, "blob_bytes": 100000, "ep_loss": 100.0},
        {"stage": "verdict", "epoch": 25, "seg_form": "ce", "d_seg": 0.009900,
         "d_pose": 0.010, "blob_bytes": 100000, "ep_loss": 99.0},
        {"stage": "verdict", "epoch": 50, "seg_form": "ce", "d_seg": 0.009800,
         "d_pose": 0.030, "blob_bytes": 100000, "ep_loss": 98.0},
        {"stage": "verdict", "epoch": 75, "seg_form": "ce", "d_seg": 0.009700,
         "d_pose": 0.060, "blob_bytes": 100000, "ep_loss": 97.0},
    ]
    d = _make_run_dir(tmp_path, rows)
    rep = build_shadow_report(load_run_inputs(d))
    assert rep.classification["classification"] == "converging"
    assert not any(r["action"] == "CONTINUE_STAGE" for r in rep.recommendations)
    refused = [r for r in rep.refused if r["action"] == "CONTINUE_STAGE"]
    assert refused and "NEVER_REGRESS" in refused[0]["refusal_reason"]


def test_shadow_empty_log_refuses_honestly(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "run.log").write_text("not json\n")
    rep = build_shadow_report(load_run_inputs(d))
    assert rep.recommendations == []
    assert any(p["costate"] == "ALL_TRAJECTORY_COSTATES" for p in rep.probe_queue)


def test_write_shadow_row_appends_actuation_none(tmp_path):
    d = _make_run_dir(tmp_path, CE_ROWS)
    rep = build_shadow_report(load_run_inputs(d))
    out = write_shadow_row(d, rep)
    out2 = write_shadow_row(d, rep)
    assert out == out2 == d / "costate_shadow.jsonl"
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    for ln in lines:
        row = json.loads(ln)
        assert row["actuation"] == "NONE"
        assert "advisory" in row["axis"]
        assert "0.19110" in row["pointer"]
        assert ACTUATION == "NONE"


def test_uncertainty_propagation_formula():
    """Chained stderr must equal sqrt(sum (lambda_i * se_i)^2) for the included channels."""
    fits = stage_epoch_costates(CE_ROWS, "ce")
    est = chain_ds_depoch(fits, d_pose_latest=0.002447, stage="ce", evidence=("t",))
    lam_pose = 5.0 / math.sqrt(10.0 * 0.002447)
    lam_bytes = 25.0 / 37_545_489.0
    expected = math.sqrt((100.0 * fits["d_seg"].stderr) ** 2
                         + (lam_pose * fits["d_pose"].stderr) ** 2
                         + (lam_bytes * fits["blob_bytes"].stderr) ** 2)
    assert est.stderr == pytest.approx(expected)


# ─────────────────────── STRUCTURAL no-actuation invariant ───────────────────────
def test_no_actuation_capability():
    """CONTAINMENT is structural: no module in the package (nor the CLI) may contain
    any process-spawning / signaling surface. Source-scan, not a config flag."""
    forbidden = ("import subprocess", "from subprocess", "os.system(", "os.exec",
                 "os.spawn", "os.popen(", "os.kill(", "import signal", "pty.spawn",
                 "Popen(")
    files = list((REPO / "src/tac/witness_control").glob("*.py"))
    files.append(REPO / "tools/costate_shadow_report.py")
    assert len(files) >= 3
    for f in files:
        src = f.read_text()
        for tok in forbidden:
            assert tok not in src, f"actuation-capable token {tok!r} in {f}"


# ─────────────────────── backtest on the REAL #205 log ───────────────────────
needs_205 = pytest.mark.skipif(not (RUN_205 / "run.log").is_file(),
                               reason="#205 reference run.log not on this checkout")


@needs_205
def test_backtest_205_ep325_watch_not_act():
    """As-of ep325 (one tau verdict after the spike): the controller must say WATCH —
    the tau-onset spike is indistinguishable from a recoverable transient yet."""
    rep = build_shadow_report(load_run_inputs(RUN_205, as_of_epoch=325))
    assert rep.classification["classification"] == "transition_transient"
    assert any(r["action"] == "WATCH_NO_ACTION" for r in rep.recommendations)


@needs_205
def test_backtest_205_ep450_diverging_rollback():
    """As-of ep450: sustained tau-creep must classify as erosion, and the top ranked
    recommendation must be the rollback (largest identifiable |ΔS|), with negative
    predicted ΔS (an improvement)."""
    rep = build_shadow_report(load_run_inputs(RUN_205, as_of_epoch=450))
    assert rep.classification["classification"] == "diverging_erasing"
    assert rep.recommendations
    top = rep.recommendations[0]
    assert top["action"] == "ROLLBACK_TO_BEST_CHECKPOINT"
    assert top["predicted_dS"] < 0.0
    assert any("verdict@ep" in e for e in top["evidence"])


@needs_205
def test_backtest_205_l7_muon_unidentifiable():
    """The l7/Muon stages never ran in the #205 log (ends ~ep525 < l7@1000/muon@726
    reach): their costates MUST come back UNIDENTIFIABLE, not guessed."""
    rep = build_shadow_report(load_run_inputs(RUN_205))
    stages = rep.state.get("stages_seen") or []
    assert "l7_softplus" not in stages and "muon" not in stages
    named = {c.name for c in rep.costates}
    assert not any("l7" in n or "muon" in n for n in named if "dS_depoch" in n)


# --- #247 SENSE→DECIDE: activation-ledger duty_to_measure surface -----------
def test_shadow_report_surfaces_duty_to_measure_never_fired_levers():
    """The controller SURFACES the DSL levers OWED a measurement (the 'off is a tracked queue'
    apparatus) so the operator never has to remember. NO-FAKE: never-fired levers carry an honest
    'why' and NO fabricated predicted ΔS (they have no measured costate). Ordered never-fired-first."""
    from tac.witness_control.shadow_controller import RunInputs
    rep = build_shadow_report(RunInputs(run_dir="/tmp/duty_smoke", verdicts=[], stage_rows={}, flags={}))
    dtm = rep.duty_to_measure
    assert dtm, "empty ledger should honestly surface every registered DSL lever as owed"
    assert all(r["default"] == "off" for r in dtm)                       # score-affecting levers default off
    assert all("why" in r and "predicted_dS" not in r for r in dtm)      # NO-FAKE: no invented ΔS
    assert dtm[0]["state"] in ("never-fired", "fired-unmeasured")        # unmeasured surfaced first
    assert "duty_to_measure" in rep.to_row()                            # persisted in the JSONL row
