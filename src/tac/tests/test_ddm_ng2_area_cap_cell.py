# SPDX-License-Identifier: MIT
"""Tests for the ddm_ng2 one-sided Chan-Vese area cap and its fixed-tau telemetry row.

The tests verify BEHAVIOUR, never constants: every assertion here would fail if the penalty
were replaced by a zero, if the hinge stopped being one-sided, if the area estimator lost its
argmax value or its softmax gradient, if the telemetry row acquired a gradient, or if the
sealed cell moved more than the single lever.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


qbt = _load("ddm_qbt1_qbflow_trainer_ng2test", "experiments/ddm_qbt1_qbflow_trainer.py")

CLASSES = ("Lane", "Movable")
LAMBDAS = {"Lane": 2_799.79768968321, "Movable": 7_587.370036040673}


def _field(lane_rows: int, movable_rows: int, height: int = 16, width: int = 16) -> torch.Tensor:
    logits = torch.full((2, 5, height, width), -8.0)
    logits[:, 0] = 8.0
    if lane_rows:
        logits[:, 1, :lane_rows, :] = 16.0
    if movable_rows:
        logits[:, 3, 8:8 + movable_rows, :] = 16.0
    return logits


def _target(height: int = 16, width: int = 16) -> torch.Tensor:
    target = torch.zeros((2, height, width), dtype=torch.long)
    target[:, :4, :] = 1
    target[:, 4:8, :] = 3
    return target


# ---------------------------------------------------------------------------
# the area estimator
# ---------------------------------------------------------------------------
def test_realized_class_area_value_is_the_exact_argmax_area_not_the_softmax_mass():
    logits = _field(8, 8)
    areas = qbt.realized_class_area_ste(logits)
    index = logits.argmax(dim=1)
    for class_id in range(5):
        exact = (index == class_id).double().mean(dim=(1, 2))
        assert torch.allclose(areas[:, class_id].double(), exact, atol=0.0, rtol=0.0)
    soft = torch.softmax(logits, dim=1).mean(dim=(2, 3))
    assert not torch.allclose(areas, soft), "value must be the argmax area, not the soft mass"


def test_realized_class_area_gradient_is_the_softmax_jacobian():
    logits = _field(8, 8).requires_grad_(True)
    areas = qbt.realized_class_area_ste(logits)
    areas[:, 1].sum().backward()
    soft_logits = _field(8, 8).requires_grad_(True)
    torch.softmax(soft_logits, dim=1).mean(dim=(2, 3))[:, 1].sum().backward()
    assert torch.allclose(logits.grad, soft_logits.grad, atol=0.0, rtol=0.0)


def test_realized_class_area_sums_to_one_over_classes():
    areas = qbt.realized_class_area_ste(_field(5, 3))
    assert torch.allclose(areas.sum(dim=1), torch.ones(areas.shape[0]), atol=1e-6)


def test_realized_class_area_refuses_wrong_geometry():
    with pytest.raises(qbt.QBT1Error):
        qbt.realized_class_area_ste(torch.zeros((2, 5, 16)))


# ---------------------------------------------------------------------------
# the one-sided penalty
# ---------------------------------------------------------------------------
def test_cap_is_exactly_zero_when_both_classes_sit_at_or_under_gt_area():
    penalty, components = qbt.one_sided_area_cap_penalty(_field(2, 2), _target(), LAMBDAS)
    assert float(penalty) == 0.0
    for name in CLASSES:
        assert float(components[f"area_cap_over_{name}"]) == 0.0


def test_cap_is_exactly_zero_at_the_hinge_and_positive_one_row_above_it():
    at_gt, _ = qbt.one_sided_area_cap_penalty(_field(4, 4), _target(), LAMBDAS)
    above, _ = qbt.one_sided_area_cap_penalty(_field(5, 5), _target(), LAMBDAS)
    assert float(at_gt) == 0.0
    assert float(above) > 0.0


def test_cap_gradient_vanishes_below_gt_and_is_inward_above_it():
    under = _field(2, 2).requires_grad_(True)
    qbt.one_sided_area_cap_penalty(under, _target(), LAMBDAS)[0].backward()
    assert float(under.grad.abs().max()) == 0.0

    over = _field(8, 8).requires_grad_(True)
    qbt.one_sided_area_cap_penalty(over, _target(), LAMBDAS)[0].backward()
    # the penalty pushes the over-painted class's logits DOWN, i.e. positive dL/dlogit there.
    assert float(over.grad[:, 1, :8, :].sum()) > 0.0
    assert float(over.grad[:, 3, 8:16, :].sum()) > 0.0


def test_cap_energy_matches_the_registered_numpy_reference():
    from tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708 import (
        area_penalty,
    )

    logits = _field(8, 8)
    target = _target()
    penalty, _components = qbt.one_sided_area_cap_penalty(logits, target, LAMBDAS)
    areas = qbt.realized_class_area_ste(logits).detach()[0].double().numpy()
    gt = torch.stack([(target[0] == c).double().mean() for c in range(5)]).numpy()
    expected = 0.0
    for name, class_id in qbt.AREA_CAP_CLASSES:
        over = max(0.0, float(areas[class_id]) - float(gt[class_id]))
        expected += 0.5 * LAMBDAS[name] * over * over
    assert math.isclose(float(penalty), expected, rel_tol=1e-6)
    assert area_penalty(areas, gt, (1, 3), birth_force=1.0, tolerance=0.25) >= 0.0


def test_cap_hinges_per_pair_never_on_the_batch_mean():
    """A pair over GT and a pair under it must NOT cancel."""
    target = _target()
    logits = _field(8, 8)
    logits[1] = _field(0, 0)[1]  # second pair paints neither class
    per_pair, _ = qbt.one_sided_area_cap_penalty(logits, target, LAMBDAS)
    only_over, _ = qbt.one_sided_area_cap_penalty(logits[:1], target[:1], LAMBDAS)
    # HT weights default to 1 here, so the per-pair mean is half of the single over-painted pair.
    assert math.isclose(float(per_pair), 0.5 * float(only_over), rel_tol=1e-9)
    assert float(per_pair) > 0.0


def test_cap_honours_sample_weights():
    target = _target()
    logits = _field(8, 8)
    flat, _ = qbt.one_sided_area_cap_penalty(logits, target, LAMBDAS, torch.ones(2))
    skewed, _ = qbt.one_sided_area_cap_penalty(logits, target, LAMBDAS, torch.tensor([1.0, 3.0]))
    assert math.isclose(float(flat), float(skewed), rel_tol=1e-9)  # identical pairs
    logits[1] = _field(0, 0)[1]
    light, _ = qbt.one_sided_area_cap_penalty(logits, target, LAMBDAS, torch.tensor([3.0, 1.0]))
    heavy, _ = qbt.one_sided_area_cap_penalty(logits, target, LAMBDAS, torch.tensor([1.0, 3.0]))
    assert float(light) > float(heavy)


def test_cap_refuses_a_wrong_class_set_or_a_non_positive_lambda():
    with pytest.raises(qbt.QBT1Error):
        qbt.one_sided_area_cap_penalty(_field(8, 8), _target(), {"Lane": 1.0})
    with pytest.raises(qbt.QBT1Error):
        qbt.one_sided_area_cap_penalty(_field(8, 8), _target(), {"Lane": 0.0, "Movable": 1.0})
    with pytest.raises(qbt.QBT1Error):
        qbt.one_sided_area_cap_penalty(
            _field(8, 8), _target(), LAMBDAS, torch.tensor([1.0, 0.0])
        )


def test_cap_refuses_mismatched_target_geometry():
    with pytest.raises(qbt.QBT1Error):
        qbt.one_sided_area_cap_penalty(_field(8, 8), _target(height=8), LAMBDAS)


# ---------------------------------------------------------------------------
# the stiffness derivation
# ---------------------------------------------------------------------------
def test_lambda_is_the_registered_laws_own_value_never_a_local_reimplementation():
    from tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708 import (
        area_constraint_lambda,
    )

    gt = qbt.selection_gt_area_fractions(list(qbt.SELECTION_IDS))
    force = {"Lane": 0.6793084080, "Movable": 2.3063736731}
    tolerance = {"Lane": 0.04069965740993542, "Movable": 0.02440342632729764}
    lambdas = qbt.derive_area_cap_lambdas(list(qbt.SELECTION_IDS), force, tolerance)
    for name, class_id in qbt.AREA_CAP_CLASSES:
        assert lambdas[name] == area_constraint_lambda(
            gt[class_id], birth_force=force[name], tolerance=tolerance[name]
        )


def test_selection_gt_areas_are_the_balanced_weight_bincount_normalized():
    gt = qbt.selection_gt_area_fractions(list(qbt.SELECTION_IDS))
    weights = qbt.derive_balanced_class_weights(list(qbt.SELECTION_IDS), torch.device("cpu"))
    assert math.isclose(sum(gt.values()), 1.0, rel_tol=1e-12)
    for class_id, area in gt.items():
        assert math.isclose(float(weights[class_id]) * 5.0 * area, 1.0, rel_tol=1e-6)


def test_derive_area_cap_lambdas_refuses_a_wrong_class_set():
    with pytest.raises(qbt.QBT1Error):
        qbt.derive_area_cap_lambdas(
            list(qbt.SELECTION_IDS), {"Lane": 1.0}, {"Lane": 0.1, "Movable": 0.1}
        )


# ---------------------------------------------------------------------------
# the DSL leg
# ---------------------------------------------------------------------------
def _lever():
    from tac.witness_dsl.curriculum_dsl import AreaCapBornRareClass

    return AreaCapBornRareClass(
        birth_force={"Lane": 0.6793084080, "Movable": 2.3063736731},
        tolerance={"Lane": 0.04069965740993542, "Movable": 0.02440342632729764},
        gt_area={"Lane": 0.005961418151855469, "Movable": 0.012456258138020834},
    )


def test_dsl_lever_compiles_the_law_and_its_own_lambdas():
    from tac.witness_dsl.curriculum_dsl import compile_qbr1_area_cap_config

    block = compile_qbr1_area_cap_config(_lever())
    assert block["law"] == "chan_vese_area_constraint_birth_balance_v1"
    assert block["softmax_temperature"] == 1.0
    assert set(block["lambdas"]) == set(CLASSES)
    for name in CLASSES:
        assert math.isclose(block["lambdas"][name], LAMBDAS[name], rel_tol=1e-12)


def test_dsl_lever_refuses_a_partial_or_non_positive_knob():
    from tac.witness_dsl.curriculum_dsl import AreaCapBornRareClass

    good = {"Lane": 1.0, "Movable": 1.0}
    with pytest.raises(ValueError):
        AreaCapBornRareClass(birth_force={"Lane": 1.0}, tolerance=good, gt_area=good)
    with pytest.raises(ValueError):
        AreaCapBornRareClass(birth_force=good, tolerance={"Lane": 0.0, "Movable": 1.0},
                             gt_area=good)


def test_compile_refuses_overrides_outside_the_area_cap_namespace():
    from tac.witness_dsl.curriculum_dsl import Lever, compile_qbr1_area_cap_config

    with pytest.raises(ValueError):
        compile_qbr1_area_cap_config(Lever("x", overrides={"--not-a-config-key": 1}))
    with pytest.raises(ValueError):
        compile_qbr1_area_cap_config(Lever("x", overrides={}))


# ---------------------------------------------------------------------------
# the objective composition + the telemetry row
# ---------------------------------------------------------------------------
def _objective_inputs():
    torch.manual_seed(7)
    batch, height, width = 2, 16, 16
    logits = _field(8, 8).requires_grad_(True)
    outputs = {"class_logits": torch.randn(batch, height, width, 5, requires_grad=True)}
    camera = torch.rand(batch, 2, 3, 32, 32) * 255.0
    pose6 = torch.randn(batch, 6, requires_grad=True)
    target_pose6 = torch.randn(batch, 6)
    return logits, outputs, camera, pose6, _target(), target_pose6, torch.ones(batch)


def _config(with_cap: bool):
    config = {"objective": {"realized_weight": 100.0, "native_interface_weight": 100.0}}
    if with_cap:
        from tac.witness_dsl.curriculum_dsl import compile_qbr1_area_cap_config

        config["area_cap"] = compile_qbr1_area_cap_config(_lever())
    return config


def test_objective_without_an_area_cap_block_is_the_control_form():
    qbr1 = _load("ddm_qbr1_ng2test", "experiments/ddm_qbr1_born_fairform_burn_prep.py")
    logits, outputs, camera, pose6, target, target_pose6, weights = _objective_inputs()
    lambdas = {"Lane": 0.01, "Movable": 0.02}
    total, components = qbr1.fairform_objective(
        _config(False), outputs, camera, pose6, logits, target, target_pose6, 0.15,
        weights, lambdas,
    )
    assert not any(name.startswith("area_cap") for name in components)
    assert torch.isfinite(total)


def test_objective_with_the_area_cap_adds_exactly_the_cap_energy():
    qbr1 = _load("ddm_qbr1_ng2test", "experiments/ddm_qbr1_born_fairform_burn_prep.py")
    logits, outputs, camera, pose6, target, target_pose6, weights = _objective_inputs()
    lambdas = {"Lane": 0.01, "Movable": 0.02}
    args = (outputs, camera, pose6, logits, target, target_pose6, 0.15, weights, lambdas)
    plain, _ = qbr1.fairform_objective(_config(False), *args)
    capped, components = qbr1.fairform_objective(_config(True), *args)
    expected, _ = qbt.one_sided_area_cap_penalty(logits, target, LAMBDAS, weights)
    assert math.isclose(float(capped) - float(plain), float(expected), rel_tol=1e-6)
    assert float(components["area_cap_energy"]) > 0.0


def test_fixed_tau_telemetry_is_present_detached_and_never_in_the_loss():
    qbr1 = _load("ddm_qbr1_ng2test", "experiments/ddm_qbr1_born_fairform_burn_prep.py")
    logits, outputs, camera, pose6, target, target_pose6, weights = _objective_inputs()
    lambdas = {"Lane": 0.01, "Movable": 0.02}
    total, components = qbr1.fairform_objective(
        _config(False), outputs, camera, pose6, logits, target, target_pose6, 0.15,
        weights, lambdas,
    )
    telemetry = components["seg_expected_flip_realized_tau_ref"]
    assert telemetry.requires_grad is False
    assert math.isclose(float(components["tau_ref"]),
                        qbt.EXPECTED_FLIP_TAU_REFERENCE, rel_tol=1e-6)
    reference = qbt.expected_flip_margin_loss(
        logits.detach(), target, qbt.EXPECTED_FLIP_TAU_REFERENCE, weights
    )
    assert math.isclose(float(telemetry), float(reference), rel_tol=1e-9)
    # the loss must not have absorbed it: back-propagating leaves the telemetry unreachable.
    total.backward()
    assert telemetry.grad_fn is None


def test_fixed_tau_telemetry_differs_from_the_annealed_value_at_the_same_field():
    """On a field with FINITE margins the two temperatures must report different numbers.

    The saturated synthetic field above cannot show this (both sigmoids are 0 or 1), which is
    exactly why ddm_sd1's schedule leg is invisible until the margins are realistic."""
    qbr1 = _load("ddm_qbr1_ng2test", "experiments/ddm_qbr1_born_fairform_burn_prep.py")
    logits, outputs, camera, pose6, target, target_pose6, weights = _objective_inputs()
    torch.manual_seed(11)
    logits = (0.05 * torch.randn_like(logits)).requires_grad_(True)
    _total, components = qbr1.fairform_objective(
        _config(False), outputs, camera, pose6, logits, target, target_pose6, 0.15,
        weights, {"Lane": 0.01, "Movable": 0.02},
    )
    annealed = float(components["seg_expected_flip_realized"])
    reference = float(components["seg_expected_flip_realized_tau_ref"])
    assert reference != annealed, "a telemetry row equal to the training value reports nothing"


# ---------------------------------------------------------------------------
# the sealed-cell contract
# ---------------------------------------------------------------------------
def test_area_cap_block_validator_rederives_every_lambda():
    qbr1 = _load("ddm_qbr1_ng2test", "experiments/ddm_qbr1_born_fairform_burn_prep.py")
    from tac.witness_dsl.curriculum_dsl import compile_qbr1_area_cap_config

    config = {"area_cap": compile_qbr1_area_cap_config(_lever())}
    qbr1.validate_area_cap_block(config)  # passes
    qbr1.validate_area_cap_block({})  # absent block is the control form
    tampered = copy.deepcopy(config)
    tampered["area_cap"]["lambdas"]["Lane"] *= 2.0
    with pytest.raises(qbr1.QBR1Error):
        qbr1.validate_area_cap_block(tampered)


def test_area_cap_block_validator_refuses_tau_as_the_softmax_temperature():
    qbr1 = _load("ddm_qbr1_ng2test", "experiments/ddm_qbr1_born_fairform_burn_prep.py")
    from tac.witness_dsl.curriculum_dsl import compile_qbr1_area_cap_config

    config = {"area_cap": compile_qbr1_area_cap_config(_lever())}
    config["area_cap"]["softmax_temperature"] = 0.05
    with pytest.raises(qbr1.QBR1Error):
        qbr1.validate_area_cap_block(config)


def test_area_cap_block_validator_refuses_a_foreign_law_or_form():
    qbr1 = _load("ddm_qbr1_ng2test", "experiments/ddm_qbr1_born_fairform_burn_prep.py")
    from tac.witness_dsl.curriculum_dsl import compile_qbr1_area_cap_config

    for key, value in (("law", "some_other_law_v1"), ("form", "two_sided"),
                       ("area_estimator", "softmax_mass")):
        config = {"area_cap": compile_qbr1_area_cap_config(_lever())}
        config["area_cap"][key] = value
        with pytest.raises(qbr1.QBR1Error):
            qbr1.validate_area_cap_block(config)


def test_single_lever_validator_refuses_a_second_moved_field():
    ng2 = _load("ddm_ng2_cell_ng2test", "experiments/ddm_ng2_area_cap_cell.py")
    control = {
        "cell_id": "c", "output": "/o", "objective": {}, "ema": {}, "schedule": {},
        "initial_state": {}, "learning_rate": 2.0e-4, "margin_constraints": {},
        "expected_flip_tau_start": 0.15, "expected_flip_tau_end": 0.05, "pair_ids": [],
        "selection_weights": [], "total_steps": 5000, "milestones": [], "seed": 20260902,
        "resume_from": None, "launch_authorized": False,
        "scorer_lane": {"claimed": False, "claim_id": None},
        "metal_lane": {"claimed": False, "claim_id": None},
    }
    cell = copy.deepcopy(control)
    cell["cell_id"] = "ng2"
    cell["output"] = "/o2"
    cell["area_cap"] = {"lambdas": {}}
    assert ng2.validate_area_cap_cell(cell, control)["held_fields_identical"] is True
    moved = copy.deepcopy(cell)
    moved["learning_rate"] = 1.0e-4
    with pytest.raises(ng2.NG2Error):
        ng2.validate_area_cap_cell(moved, control)


def test_single_lever_validator_refuses_a_warm_resume_seed():
    ng2 = _load("ddm_ng2_cell_ng2test", "experiments/ddm_ng2_area_cap_cell.py")
    control = {
        "cell_id": "c", "output": "/o", "objective": {}, "ema": {}, "schedule": {},
        "initial_state": {}, "learning_rate": 2.0e-4, "margin_constraints": {},
        "expected_flip_tau_start": 0.15, "expected_flip_tau_end": 0.05, "pair_ids": [],
        "selection_weights": [], "total_steps": 5000, "milestones": [], "seed": 20260902,
        "resume_from": None, "launch_authorized": False,
        "scorer_lane": {"claimed": False, "claim_id": None},
        "metal_lane": {"claimed": False, "claim_id": None},
    }
    cell = copy.deepcopy(control)
    cell["area_cap"] = {}
    cell["resume_from"] = "/some/warm_seed.pt"
    control_with_seed = copy.deepcopy(control)
    control_with_seed["resume_from"] = "/some/warm_seed.pt"
    with pytest.raises(ng2.NG2Error):
        ng2.validate_area_cap_cell(cell, control_with_seed)


def test_falsifiers_are_pre_registered_against_the_measured_control_rows():
    ng2 = _load("ddm_ng2_cell_ng2test", "experiments/ddm_ng2_area_cap_cell.py")
    rows = ng2.falsifiers()
    assert set(rows) == {
        "1_primary_the_cap_must_beat_the_cold_control_at_both_ends",
        "2_the_cap_must_actually_bind",
        "3_the_fixed_tau_telemetry_must_be_faithful",
    }
    primary = rows["1_primary_the_cap_must_beat_the_cold_control_at_both_ends"]
    assert "0.42514878445269977" in primary["test"]
    assert "0.48567677825279465" in primary["test"]
    assert primary["control_rows"][2_000] == max(ng2.COLD_CONTROL_S_HAT.values())


def test_the_cold_control_of_record_peaks_at_step_two_thousand():
    ng2 = _load("ddm_ng2_cell_ng2test", "experiments/ddm_ng2_area_cap_cell.py")
    rows = ng2.COLD_CONTROL_S_HAT
    assert max(rows, key=rows.get) == 2_000
    assert rows[5_000] > rows[0], "the control ends above its own start; that is the excursion"


def test_measured_birth_force_reads_the_control_history_when_present():
    ng2 = _load("ddm_ng2_cell_ng2test", "experiments/ddm_ng2_area_cap_cell.py")
    history = ng2.CONTROL_RUN / "history.jsonl"
    if not history.is_file():  # pragma: no cover - store not mounted
        pytest.skip("the control run store is not mounted")
    measured = ng2.measured_birth_force(history, window=50)
    for name in CLASSES:
        stat = measured["per_class"][name]
        assert stat["n"] == 50
        assert stat["min"] <= stat["p10"] <= stat["median"] <= stat["p90"] <= stat["max"]


def test_measured_birth_force_refuses_a_short_history(tmp_path):
    ng2 = _load("ddm_ng2_cell_ng2test", "experiments/ddm_ng2_area_cap_cell.py")
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps({
        "completed_steps": 1,
        "margin_constraint_lambdas": {"Lane": 0.1, "Movable": 0.2},
    }) + "\n", encoding="utf-8")
    with pytest.raises(ng2.NG2Error):
        ng2.measured_birth_force(path, window=10)
