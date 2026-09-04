# SPDX-License-Identifier: MIT
"""ddm_ng4 — the CONTINUOUS-OBJECTIVE cell: gates, seeding, and the derivation of the carried state.

Every test here is about BEHAVIOUR, not constants: the three that matter most are
``test_absent_blocks_seed_the_duals_from_zero_exactly_as_before`` (the control stays
byte-identical), ``test_the_carried_duals_reach_both_executable_paths`` (a lever the smoke
cannot reach is an inert lever), and
``test_dsl_linear_anneal_equals_the_trainers_tau_for_step`` (the DSL's re-implementation of the
anneal cannot drift from the trainer's).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

qbr1 = pytest.importorskip("experiments.ddm_qbr1_born_fairform_burn_prep")
qbt = pytest.importorskip("experiments.ddm_qbt1_qbflow_trainer")

from tac.witness_dsl.curriculum_dsl import (  # noqa: E402
    QBR1_CONTINUED_OBJECTIVE_STATES,
    QBR1_LEGACY_TAU_BAND,
    QBR1_TAU_BAND_MODES,
    ContinuousObjectiveFromR10,
    _linear_anneal_tau,
    compile_qbr1_continuous_objective_config,
)

_R10_AVAILABLE = qbr1.R10_CONFIG.is_file() and qbr1.R10_CHECKPOINT.is_file()
needs_r10 = pytest.mark.skipif(not _R10_AVAILABLE, reason="r10's pinned artifacts are not mounted")


# ---------------------------------------------------------------------------
# 1. the anneal geometry contract
# ---------------------------------------------------------------------------
def test_dsl_linear_anneal_equals_the_trainers_tau_for_step():
    """The DSL cannot import qbt (qbt imports the DSL), so the duplication must be guarded."""

    for start, end in ((0.15, 0.05), (0.05, 0.05), (0.0437, 0.0218)):
        for total in (2, 5, 5_000, 10_000):
            for step in (0, 1, total // 2, total - 1):
                assert _linear_anneal_tau(step, total, start, end) == qbt.tau_for_step(
                    step, total, start, end
                )


def test_tau_for_step_admits_a_held_band_and_holds_it_flat():
    held = float(QBR1_LEGACY_TAU_BAND[1])
    values = [qbt.tau_for_step(step, 5_000, held, held) for step in (0, 1, 2_500, 4_999)]
    assert values == [held] * 4


def test_tau_for_step_still_refuses_an_increasing_or_nonpositive_band():
    with pytest.raises(qbt.QBT1Error):
        qbt.tau_for_step(0, 5_000, 0.05, 0.15)
    with pytest.raises(qbt.QBT1Error):
        qbt.tau_for_step(0, 5_000, 0.0, 0.0)
    with pytest.raises(qbt.QBT1Error):
        qbt.tau_for_step(5_000, 5_000, 0.05, 0.05)
    with pytest.raises(ValueError):
        _linear_anneal_tau(0, 5_000, 0.05, 0.15)


def test_the_admissible_band_set_gains_exactly_the_held_continuation_band():
    bands = qbt.admissible_expected_flip_tau_bands()
    held = float(QBR1_LEGACY_TAU_BAND[1])
    assert QBR1_LEGACY_TAU_BAND in bands
    assert (held, held) in bands
    assert len(bands) == 3
    assert len(set(bands)) == 3


def test_the_held_band_is_not_a_new_literal_it_is_the_legacy_end():
    """The continuation band reuses the end this lineage already ships, so no decimal is typed."""

    held = [band for band in qbt.admissible_expected_flip_tau_bands() if band[0] == band[1]]
    assert held == [(QBR1_LEGACY_TAU_BAND[1], QBR1_LEGACY_TAU_BAND[1])]


def test_r10_continuation_is_a_declared_mode():
    assert qbr1.R10_CONTINUATION_MODE in QBR1_TAU_BAND_MODES
    assert set(QBR1_CONTINUED_OBJECTIVE_STATES) == {"expected_flip_tau", "margin_dual"}


# ---------------------------------------------------------------------------
# 2. the dual seeding — the control must stay byte-identical
# ---------------------------------------------------------------------------
def _bounds() -> dict[str, float]:
    return {"Lane": 0.12, "Movable": 0.009}


def test_absent_blocks_seed_the_duals_from_zero_exactly_as_before():
    """This is what keeps every QBR1 cell sealed before ng4 byte-identical."""

    bounds = _bounds()
    assert qbr1.initial_margin_constraint_lambdas({}, bounds) == dict.fromkeys(bounds, 0.0)
    assert qbr1.initial_margin_constraint_lambdas(
        {"margin_constraints": {"bounds": bounds}}, bounds
    ) == dict.fromkeys(bounds, 0.0)


def test_declared_multipliers_are_seeded_verbatim():
    bounds = _bounds()
    declared = {"Lane": 0.005040981907324784, "Movable": 0.017331143732962344}
    config = {"margin_constraints": {"bounds": bounds, "initial_lambdas": declared}}
    assert qbr1.initial_margin_constraint_lambdas(config, bounds) == declared


def test_a_partial_multiplier_set_is_refused():
    bounds = _bounds()
    config = {"margin_constraints": {"bounds": bounds, "initial_lambdas": {"Lane": 0.1}}}
    with pytest.raises(qbr1.QBR1Error):
        qbr1.initial_margin_constraint_lambdas(config, bounds)


def test_the_carried_duals_reach_both_executable_paths():
    """A lever the smoke cannot reach is an inert lever; both loops must call the seeder."""

    source = Path(qbr1.__file__).read_text(encoding="utf-8")
    assert source.count("initial_margin_constraint_lambdas(") >= 3
    assert "lambdas = initial_margin_constraint_lambdas(config, bounds)" in source
    assert "lambdas = initial_margin_constraint_lambdas(smoke_config, bounds)" in source
    assert "dict.fromkeys(bounds, 0.0)" in source  # the control path still exists, inside the seeder


# ---------------------------------------------------------------------------
# 3. the r10 derivation
# ---------------------------------------------------------------------------
@needs_r10
def test_r10_terminal_tau_is_re_derived_through_the_anneal_not_read_off_the_endpoint():
    live = qbr1.r10_terminal_tau()
    assert live["r10_terminal_step_index"] == live["r10_margin_steps"] - 1
    assert live["r10_terminal_tau"] == qbt.tau_for_step(
        live["r10_terminal_step_index"], live["r10_margin_steps"],
        live["r10_expected_flip_tau_start"], live["r10_expected_flip_tau_end"],
    )
    # MEASURED 2026-09-04: r10 annealed to exactly the band end this lineage ships.
    assert live["r10_terminal_tau"] == QBR1_LEGACY_TAU_BAND[1]


@needs_r10
def test_r10_terminal_duals_match_the_pinned_checkpoints_curriculum_state():
    live = qbr1.r10_terminal_duals()
    assert set(live["initial_lambdas"]) == set(live["r10_bounds"])
    for value in live["initial_lambdas"].values():
        assert 0.0 <= value <= qbt.MARGIN_CONSTRAINT_LAMBDA_MAX
    assert live["r10_constraint_mode"] == qbt.MARGIN_CONSTRAINT_LANE_MOVABLE
    # the dual LAW is identical on both sides, which is what makes a multiplier transferable.
    pins = qbt.MARGIN_CONSTRAINT_MODE_PINS[qbt.MARGIN_CONSTRAINT_LANE_MOVABLE]
    assert live["r10_bounds"] == {k: float(v) for k, v in pins["bounds"].items()}
    assert live["r10_eta_lambda"] == float(pins["eta_lambda"])


@needs_r10
def test_the_lever_emits_both_namespaces_and_nothing_else():
    lever = ContinuousObjectiveFromR10(qbr1.R10_CONFIG, qbr1.R10_CHECKPOINT)
    namespaces = {key.split(".", 1)[0] for key in lever.overrides}
    assert namespaces == set(QBR1_CONTINUED_OBJECTIVE_STATES)
    tau, start, end, dual, lambdas = compile_qbr1_continuous_objective_config(lever)
    assert start == end == qbr1.r10_terminal_tau()["r10_terminal_tau"]
    assert lambdas == qbr1.r10_terminal_duals()["initial_lambdas"]
    assert tau["mode"] == dual["mode"] == qbr1.R10_CONTINUATION_MODE


@needs_r10
def test_the_lever_is_byte_stable_across_compiles():
    """No observation timestamp is carried, so a sealed sha can be quoted and verified."""

    first = ContinuousObjectiveFromR10(qbr1.R10_CONFIG, qbr1.R10_CHECKPOINT)
    second = ContinuousObjectiveFromR10(qbr1.R10_CONFIG, qbr1.R10_CHECKPOINT)
    assert json.dumps(first.overrides, sort_keys=True) == json.dumps(second.overrides, sort_keys=True)


def test_compile_refuses_an_override_outside_the_two_namespaces():
    from tac.witness_dsl.curriculum_dsl import Lever

    bad = Lever("bad", overrides={"area_cap.lambdas": {}})
    with pytest.raises(ValueError):
        compile_qbr1_continuous_objective_config(bad)


# ---------------------------------------------------------------------------
# 4. the two QBR1 gates
# ---------------------------------------------------------------------------
@needs_r10
def test_a_control_config_still_validates_unchanged():
    initial_state = qbt.file_fact(qbr1.QBR_INITIAL_STATE)
    control = qbr1.compile_cell(20260902, "control_native100", initial_state)
    assert "margin_dual" not in control
    assert "initial_lambdas" not in control["margin_constraints"]
    assert (control["expected_flip_tau_start"], control["expected_flip_tau_end"]) == QBR1_LEGACY_TAU_BAND
    qbr1.validate_config(control, require_launch_authority=False)


@needs_r10
def test_the_continuation_cell_validates_and_moves_only_its_lever():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    cell, control = ng4.compile_continuous_objective_cell()
    diff = ng4.validate_continuous_objective_cell(cell, control)
    assert diff["margin_constraints_fields_moved"] == ["initial_lambdas"]
    assert set(diff["differing_keys"]) <= ng4.ALLOWED_CONTINUATION_MUTATIONS
    assert cell["resume_from"] is None
    assert cell.get("area_cap") is None


@needs_r10
def test_a_hand_edited_held_temperature_is_refused():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    cell, _control = ng4.compile_continuous_objective_cell()
    tampered = copy.deepcopy(cell)
    tampered["expected_flip_tau_start"] = 0.06
    tampered["expected_flip_tau_end"] = 0.06
    tampered["tau_band"]["start"] = 0.06
    tampered["tau_band"]["end"] = 0.06
    with pytest.raises(qbr1.QBR1Error):
        qbr1.validate_tau_band_block(tampered)


@needs_r10
def test_a_held_band_whose_provenance_disagrees_with_r10_is_refused():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    cell, _control = ng4.compile_continuous_objective_cell()
    tampered = copy.deepcopy(cell)
    tampered["tau_band"]["r10_margin_steps"] = 9_999
    with pytest.raises(qbr1.QBR1Error):
        qbr1.validate_tau_band_block(tampered)


@needs_r10
def test_a_held_band_that_cites_the_margin_band_law_is_refused():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    cell, _control = ng4.compile_continuous_objective_cell()
    tampered = copy.deepcopy(cell)
    tampered["tau_band"]["law"] = "margin_band_satisficing_threshold_v1"
    with pytest.raises(qbr1.QBR1Error):
        qbr1.validate_tau_band_block(tampered)


@needs_r10
def test_hand_edited_multipliers_are_refused():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    cell, _control = ng4.compile_continuous_objective_cell()
    tampered = copy.deepcopy(cell)
    tampered["margin_constraints"]["initial_lambdas"]["Lane"] = 0.5
    with pytest.raises(qbr1.QBR1Error):
        qbr1.validate_margin_dual_block(tampered)


@needs_r10
def test_multipliers_without_their_provenance_block_are_refused():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    cell, _control = ng4.compile_continuous_objective_cell()
    tampered = copy.deepcopy(cell)
    tampered.pop("margin_dual")
    with pytest.raises(qbr1.QBR1Error):
        qbr1.validate_margin_dual_block(tampered)


@needs_r10
def test_a_provenance_block_without_executable_multipliers_is_refused():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    cell, _control = ng4.compile_continuous_objective_cell()
    tampered = copy.deepcopy(cell)
    tampered["margin_constraints"].pop("initial_lambdas")
    with pytest.raises(qbr1.QBR1Error):
        qbr1.validate_margin_dual_block(tampered)


@needs_r10
def test_carrying_duals_across_a_changed_bound_is_refused():
    """A multiplier whose constraint moved is a number whose meaning moved — the ng4 genus."""

    from experiments import ddm_ng4_continuous_objective_cell as ng4

    cell, _control = ng4.compile_continuous_objective_cell()
    tampered = copy.deepcopy(cell)
    tampered["margin_constraints"]["bounds"]["Lane"] = 0.11
    with pytest.raises(qbr1.QBR1Error):
        qbr1.validate_margin_dual_block(tampered)


@needs_r10
def test_carrying_duals_across_a_changed_step_size_is_refused():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    cell, _control = ng4.compile_continuous_objective_cell()
    tampered = copy.deepcopy(cell)
    tampered["margin_constraints"]["eta_lambda"] = 0.2
    with pytest.raises(qbr1.QBR1Error):
        qbr1.validate_margin_dual_block(tampered)


# ---------------------------------------------------------------------------
# 5. the audit's own claims
# ---------------------------------------------------------------------------
@needs_r10
def test_the_discontinuity_audit_carries_exactly_two_states():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    _cell, control = ng4.compile_continuous_objective_cell()
    audit = ng4.discontinuity_audit(control)
    carried = {name for name, row in audit.items() if row.get("carried_by_this_cell")}
    assert carried == {"expected_flip_tau", "margin_dual"}


@needs_r10
def test_batch_geometry_was_never_discontinuous():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    _cell, control = ng4.compile_continuous_objective_cell()
    row = ng4.discontinuity_audit(control)["batch_geometry"]
    assert row["pair_ids_identical"] is True
    assert row["r10_chunk_pairs"] == row["cell_chunk_pairs"] == 16
    assert row["discontinuous"] is False


@needs_r10
def test_the_ema_executed_rate_is_continuous_and_the_ema_is_measurement_only():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    _cell, control = ng4.compile_continuous_objective_cell()
    row = ng4.discontinuity_audit(control)["ema"]
    assert row["relative_gap"] < 1.0e-4
    assert row["discontinuous"] is False
    # and the mechanism claim behind it: the milestone reads the shadow, the loop never does.
    source = Path(qbr1.__file__).read_text(encoding="utf-8")
    assert "with qbt.ema_scope(model, ema), torch.no_grad():" in source
    assert "ema.update(model)" in source


@needs_r10
def test_the_dual_law_is_identical_on_both_sides_so_the_multipliers_transfer():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    _cell, control = ng4.compile_continuous_objective_cell()
    assert ng4.discontinuity_audit(control)["margin_dual"]["dual_law_identical"] is True


# ---------------------------------------------------------------------------
# 6. the arm may not authorize, fire, or write near a live run
# ---------------------------------------------------------------------------
def test_the_arm_never_writes_an_authorized_config_or_touches_the_claims_ledger():
    source = Path(REPO / "experiments/ddm_ng4_continuous_objective_cell.py").read_text(encoding="utf-8")
    assert "authorized_configs" not in source
    assert "active_lane_dispatch_claims" not in source
    assert "launch_authorized\": True" not in source
    assert '"mps"' not in source


@needs_r10
def test_the_seal_leaves_the_cell_unauthorized_and_unbound():
    from experiments import ddm_ng4_continuous_objective_cell as ng4

    cell, _control = ng4.compile_continuous_objective_cell()
    assert cell["launch_authorized"] is False
    assert cell["scorer_lane"] == {"claimed": False, "claim_id": None}
    assert cell["metal_lane"] == {"claimed": False, "claim_id": None}
    assert cell["score_claim"] is False
    assert cell["promotion_eligible"] is False


def test_no_executable_source_retypes_r10s_terminal_decimals():
    """r10's duals, target decay and terminal tau are READ from pinned artifacts, never typed.

    Scope is EXECUTABLE code: docstrings are stripped first, because a docstring quoting a
    MEASURED value is documentation, while the same decimal in a literal would be a second
    source of truth that could silently disagree with the artifact it came from.
    """

    import ast

    forbidden = ("0.005040981907324784", "0.017331143732962344", "0.9995405077759483",
                 "0.9991017964071857")
    for relative in ("experiments/ddm_ng4_continuous_objective_cell.py",
                     "experiments/ddm_qbr1_born_fairform_burn_prep.py",
                     "src/tac/witness_dsl/curriculum_dsl.py"):
        tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or id(node) in docstrings:
                continue
            if isinstance(node.value, str):
                for decimal in forbidden:
                    assert decimal not in node.value, f"{relative} retypes {decimal}"
            elif isinstance(node.value, float):
                assert repr(node.value) not in forbidden, f"{relative} retypes {node.value}"
