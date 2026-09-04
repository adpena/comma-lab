"""Tests for ddm_ng5 -- the first TWO-LEVER burn cell (ng3's tau band x ng4's carried duals).

Every test here asserts BEHAVIOUR: that the composition is the control plus exactly the two
levers, that each half comes from the parent it claims, that the tau collision between the two
parents is MEASURED rather than asserted, and that no decimal either parent measured is retyped
into executable code here.  A test that only checked constants would pass against a stub.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

ng5 = pytest.importorskip("experiments.ddm_ng5_composition_cell")
qbr1 = pytest.importorskip("experiments.ddm_qbr1_born_fairform_burn_prep")

REPO = Path(__file__).resolve().parents[3]
ARM_SOURCE = REPO / "experiments/ddm_ng5_composition_cell.py"

pytestmark = pytest.mark.skipif(
    not qbr1.QBR_INITIAL_STATE.is_file(),
    reason="the QBR1 pinned initial state is not mounted on this host",
)


@pytest.fixture(scope="module")
def compiled():
    cell, control, audit = ng5.compile_composition_cell()
    return cell, control, audit


# ── the composition is the control plus exactly two levers ──────────────────────────────────────
def test_the_cell_validates_through_the_burn_preps_own_validator(compiled):
    cell, _control, _audit = compiled
    qbr1.validate_config(cell, require_launch_authority=False)


def test_the_tau_leg_is_ng3s_law_resolved_band(compiled):
    cell, _control, _audit = compiled
    assert cell["tau_band"]["mode"] == "msafe_band"
    assert cell["tau_band"]["law"] == "margin_band_satisficing_threshold_v1"


def test_the_dual_leg_is_ng4s_r10_continuation(compiled):
    cell, _control, _audit = compiled
    assert cell["margin_dual"]["mode"] == qbr1.R10_CONTINUATION_MODE
    assert cell["margin_dual"]["law"] == qbr1.R10_CONTINUATION_LAW


def test_the_trainer_read_scalars_equal_the_band_blocks_endpoints(compiled):
    cell, _control, _audit = compiled
    assert cell["expected_flip_tau_start"] == cell["tau_band"]["start"]
    assert cell["expected_flip_tau_end"] == cell["tau_band"]["end"]


def test_the_executable_multipliers_equal_the_dual_blocks(compiled):
    cell, _control, _audit = compiled
    assert (
        cell["margin_constraints"]["initial_lambdas"] == cell["margin_dual"]["initial_lambdas"]
    )


def test_the_executable_seeder_reads_the_carried_multipliers(compiled):
    cell, control, _audit = compiled
    bounds = {k: float(v) for k, v in control["margin_constraints"]["bounds"].items()}
    assert qbr1.initial_margin_constraint_lambdas(cell, bounds) == {
        k: float(v) for k, v in cell["margin_constraints"]["initial_lambdas"].items()
    }
    assert set(qbr1.initial_margin_constraint_lambdas(control, bounds).values()) == {0.0}


def test_only_the_allowed_keys_move_against_a_freshly_compiled_control(compiled):
    cell, control, _audit = compiled
    moved = {k for k in set(cell) | set(control) if cell.get(k) != control.get(k)}
    assert moved <= ng5.ALLOWED_COMPOSITION_MUTATIONS


def test_the_cell_is_a_cold_optimizer_transition(compiled):
    cell, _control, _audit = compiled
    assert cell["resume_from"] is None
    assert cell.get("area_cap") is None


def test_the_seal_leaves_the_cell_unauthorized_and_the_lanes_unbound(compiled):
    cell, _control, _audit = compiled
    assert cell["launch_authorized"] is False
    for lane in ("scorer_lane", "metal_lane"):
        assert cell[lane]["claimed"] is False
        assert cell[lane]["claim_id"] is None


def test_validate_returns_both_legs_with_their_parent_attribution(compiled):
    cell, control, _audit = compiled
    report = ng5.validate_composition_cell(cell, control)
    assert report["tau_leg"]["source"] == "ddm_ng3"
    assert report["dual_leg"]["source"] == "ddm_ng4"
    assert report["margin_constraints_fields_moved"] == ["initial_lambdas"]


# ── the validator refuses every degenerate composition ──────────────────────────────────────────
def test_validate_refuses_a_cell_that_took_ng4s_held_band_instead_of_ng3s(compiled):
    cell, control, _audit = compiled
    broken = copy.deepcopy(cell)
    broken["tau_band"] = {"mode": qbr1.R10_CONTINUATION_MODE}
    with pytest.raises(ng5.NG5Error, match="tau leg"):
        ng5.validate_composition_cell(broken, control)


def test_validate_refuses_a_cell_whose_dual_leg_is_missing(compiled):
    cell, control, _audit = compiled
    broken = copy.deepcopy(cell)
    broken["margin_dual"] = {"mode": "something_else"}
    with pytest.raises(ng5.NG5Error, match="dual leg"):
        ng5.validate_composition_cell(broken, control)


def test_validate_refuses_a_third_lever(compiled):
    cell, control, _audit = compiled
    broken = copy.deepcopy(cell)
    broken["area_cap"] = {"mode": "born_rare_class"}
    with pytest.raises(ng5.NG5Error, match="area cap"):
        ng5.validate_composition_cell(broken, control)


def test_validate_refuses_a_warm_optimizer(compiled):
    cell, control, _audit = compiled
    broken = copy.deepcopy(cell)
    broken["resume_from"] = "/some/checkpoint.pt"
    with pytest.raises(ng5.NG5Error, match="COLD-OPTIMIZER"):
        ng5.validate_composition_cell(broken, control)


def test_validate_refuses_a_widened_key_surface(compiled):
    cell, control, _audit = compiled
    broken = copy.deepcopy(cell)
    broken["learning_rate"] = float(control["learning_rate"]) * 2.0
    with pytest.raises(ng5.NG5Error):
        ng5.validate_composition_cell(broken, control)


def test_validate_refuses_a_dual_leg_that_also_moved_the_bounds(compiled):
    cell, control, _audit = compiled
    broken = copy.deepcopy(cell)
    broken["margin_constraints"]["eta_lambda"] = 1.0
    with pytest.raises(ng5.NG5Error, match="initial_lambdas"):
        ng5.validate_composition_cell(broken, control)


# ── the tau collision is measured, not asserted ─────────────────────────────────────────────────
def test_the_tau_collision_audit_records_the_premise_as_false(compiled):
    _cell, _control, audit = compiled
    assert audit["premise_holds"] is False
    assert audit["composition_takes"] == "ng3_msafe_band"


def test_the_composition_narrows_the_entry_where_the_control_widens_it(compiled):
    _cell, _control, audit = compiled
    assert audit["entry_step_control_widens_by"] > 2.5
    assert audit["entry_step_composition_narrows_by"] > 1.0
    assert audit["entry_step_composition_narrows_by"] < audit["entry_step_control_widens_by"]


def test_the_third_geometry_is_named_and_not_taken(compiled):
    _cell, _control, audit = compiled
    held = audit["ng4_held_band"]["start"]
    band_end = audit["ng3_msafe_band"]["end"]
    assert audit["third_geometry_not_taken"]["band"] == [held, band_end]


# ── comparability and provenance ────────────────────────────────────────────────────────────────
def test_the_no_pin_movement_receipt_reads_every_lever_surface_file():
    receipt = ng5.no_pin_movement_receipt()
    assert set(receipt["files"]) == set(ng5.NO_PIN_MOVEMENT_SURFACE)
    assert receipt["no_pin_movement"] == all(
        row["identical"] for row in receipt["files"].values()
    )


def test_the_parent_step1_states_come_from_the_parents_own_receipts():
    if not (ng5.NG3_SMOKE_RESULT.is_file() and ng5.NG4_SMOKE_RESULT.is_file()):
        pytest.skip("a parent's bounded-smoke receipt is not mounted on this host")
    states = ng5.parent_step1_states()
    ng3 = json.loads(ng5.NG3_SMOKE_RESULT.read_text(encoding="utf-8"))
    ng4 = json.loads(ng5.NG4_SMOKE_RESULT.read_text(encoding="utf-8"))
    assert states["ng3_tau_band_live_state_sha256"] == (
        ng3["no_op_detector"]["tau_band_live_state_sha256"]
    )
    assert states["ng4_continuous_live_state_sha256"] == (
        ng4["no_op_detector"]["continuous_live_state_sha256"]
    )
    assert states["ng3_tau_band_live_state_sha256"] != states["ng4_continuous_live_state_sha256"]


def test_the_two_parents_step1_states_differ_from_the_shared_control():
    if not (ng5.NG3_SMOKE_RESULT.is_file() and ng5.NG4_SMOKE_RESULT.is_file()):
        pytest.skip("a parent's bounded-smoke receipt is not mounted on this host")
    states = ng5.parent_step1_states()
    shared = states["shared_control_live_state_sha256"]
    assert states["ng3_tau_band_live_state_sha256"] != shared
    assert states["ng4_continuous_live_state_sha256"] != shared


def test_milestone_rows_refuses_a_required_run_that_lacks_a_milestone(tmp_path):
    with pytest.raises(ng5.NG5Error, match="lacks milestone"):
        ng5.milestone_rows(tmp_path, "empty", required=True)


def test_milestone_rows_reports_absent_steps_rather_than_inventing_them(tmp_path):
    rows = ng5.milestone_rows(tmp_path, "empty", required=False)
    assert rows["steps_present"] == []
    assert rows["rows"] == {}


def test_the_verdict_words_are_exactly_the_three_pre_registered_ones():
    read = ng5.pre_registered_read()
    assert read["verdict_words"] == ["BELOW-BOTH", "REDUNDANT", "ANTAGONISTIC"]
    assert set(read["rule"]) == set(read["verdict_words"])


def test_the_smoke_arm_names_and_receipt_names_are_distinct_from_every_parents():
    assert set(ng5.SMOKE_ARMS) == {"composition", "control"}
    text = ARM_SOURCE.read_text(encoding="utf-8")
    assert "ng5_composition_DONE.json" in text
    for reserved in ("ng4_continuous_DONE.json", "NG3_CELL_DONE.json", "ng2_area_cap_r2_DONE.json"):
        assert reserved not in text


# ── no-fake guards ──────────────────────────────────────────────────────────────────────────────
def _executable_literals(path: Path) -> set[str]:
    """Every numeric/string literal in the file with docstrings and comments stripped."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    literals: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # a bare string expression is a docstring
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
            literals.add(repr(node.value))
    return literals


def test_no_executable_source_retypes_either_parents_measured_decimals():
    """The band and the duals must be RE-DERIVED at compile time, never copied in."""
    parent_decimals = (
        0.04376363754272461,       # ng3 m_safe = 2*delta_R
        0.021881818771362305,      # ng3 delta_R
        0.005040981907324784,      # ng4 r10 terminal lambda_Lane
        0.017331143732962344,      # ng4 r10 terminal lambda_Movable
        0.05000000074505806,       # r10's terminal tau as a float32 image
    )
    literals = _executable_literals(ARM_SOURCE)
    for value in parent_decimals:
        assert repr(value) not in literals, f"{value} is retyped in executable code"


def test_the_arm_cannot_write_an_authorized_config_or_touch_the_claims_ledger():
    text = ARM_SOURCE.read_text(encoding="utf-8")
    # the queue spec NAMES the authorized path for the driver; the arm never writes it
    assert 'atomic_json(ARM_ROOT / "authorized_configs"' not in text
    assert "active_lane_dispatch_claims" not in text


def test_the_arm_names_no_device_literal():
    """The cell's device comes from the compiled config; this arm never picks one."""
    literals = _executable_literals(ARM_SOURCE)
    assert repr("mps") not in literals
    assert repr("cuda") not in literals


def test_the_queue_spec_declares_the_peak_from_the_measured_ledger():
    text = ARM_SOURCE.read_text(encoding="utf-8")
    assert '"measured_peak_rss_gib": "from_ledger"' in text
    assert '"peak_family"' in text


def test_the_cross_arm_agreement_reads_both_parents_recorded_values():
    """Both parents recorded the neutralized loss under DIFFERENT key names; both must be read."""
    if not (ng5.NG3_SMOKE_RESULT.is_file() and ng5.NG4_SMOKE_RESULT.is_file()):
        pytest.skip("a parent's bounded-smoke receipt is not mounted on this host")
    ng3 = json.loads(ng5.NG3_SMOKE_RESULT.read_text(encoding="utf-8"))
    recorded = float(
        ng3["differential_at_a_shared_tau"]["control_loss_total_at_shared_tau"]
    )
    agree = ng5._parent_neutralized_loss_totals(recorded)
    assert agree["ng3"] == recorded
    assert agree["ng4"] == recorded  # the two arms measured the same quantity independently
    assert agree["agrees_with_every_parent_that_measured_it"] is True


def test_the_cross_arm_agreement_refuses_a_value_neither_parent_measured():
    if not (ng5.NG3_SMOKE_RESULT.is_file() and ng5.NG4_SMOKE_RESULT.is_file()):
        pytest.skip("a parent's bounded-smoke receipt is not mounted on this host")
    agree = ng5._parent_neutralized_loss_totals(1.0)
    assert agree["agrees_with_every_parent_that_measured_it"] is False


def test_milestone_rows_returns_the_four_canonical_milestone_keys():
    if not (COLD := ng5.COLD_CONTROL_RUN).is_dir():
        pytest.skip("the cold control of record is not mounted on this host")
    rows = ng5.milestone_rows(COLD, "cold", required=True)
    assert set(rows["rows"]["5000"]) == {
        "S_hat", "d_seg_hat", "d_pose_hat", "archive_bytes_exact"
    }


def test_the_bands_start_is_strictly_below_r10s_terminal_tau(compiled):
    """The 'ng4's tau half is SUBSUMED' claim must be a MEASUREMENT, not a sentence."""
    _cell, _control, audit = compiled
    assert audit["ng3_msafe_band"]["start"] < audit["ng4_held_band"]["start"]
    assert audit["ng3_msafe_band"]["start"] < audit["control_band"][0]


def test_the_queue_falsifier_thresholds_separate_the_cell_from_the_control(compiled):
    """Vacuity==pass ([[m50]]): a falsifier that could never fire is not a falsifier.

    Both queue falsifiers are inert-lever detectors, so each must FIRE on a run that did not get
    its lever.  The control's own values stand in for that run: it seeds duals from zero and
    trains at the legacy tau, so ``lambda_Lane < carried`` and ``tau > band_start`` both hold.
    """
    cell, control, _audit = compiled
    carried_lane = float(cell["margin_dual"]["initial_lambdas"]["Lane"])
    band_start = float(cell["expected_flip_tau_start"])
    control_lane_seed = float(
        qbr1.initial_margin_constraint_lambdas(
            control, {k: float(v) for k, v in control["margin_constraints"]["bounds"].items()}
        )["Lane"]
    )
    control_tau = float(control["expected_flip_tau_start"])
    # op "lt" against the carried multiplier fires on a run that re-warmed from zero
    assert control_lane_seed < carried_lane
    # op "gt" against the band start fires on a run that trained at the legacy temperature
    assert control_tau > band_start
