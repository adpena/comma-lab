#!/usr/bin/env python3
"""Focused tests for the ddm_ng1 warm-transition seal.

These exercise the mechanism claims the design memo makes, not just field shapes: that a
fresh AdamW really is cold, that the warm seed really populates optimizer state through the
SEALED loader, that the one-lever discipline is enforced, and that the cold control's
measured excursion decomposes back to its own S_hat arithmetic.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ddm_ng1_warm_transition_burn_prep as ng1
import ddm_qbr1_born_fairform_burn_prep as qbr1
import ddm_qbt1_qbflow_trainer as qbt

AP_PRESENT = qbr1.R10_CHECKPOINT.is_file()
requires_r10 = pytest.mark.skipif(not AP_PRESENT, reason="r10 checkpoint not mounted")


def test_a_fresh_adamw_is_cold_and_takes_a_full_lr_sign_step() -> None:
    """The mechanism claim: step 1 of a fresh AdamW moves every parameter by ~lr."""

    weight = torch.nn.Parameter(torch.zeros(64))
    lr = 2.0e-4
    optimizer = torch.optim.AdamW([weight], lr=lr)
    assert optimizer.state_dict()["state"] == {}
    weight.grad = torch.full_like(weight, 3.7e-3)
    optimizer.step()
    moved = weight.detach().abs()
    assert torch.allclose(moved, torch.full_like(moved, lr), rtol=2e-3)


def test_a_warm_adamw_first_step_is_scaled_by_its_second_moment() -> None:
    """With saturated moments the same gradient produces a far smaller first step."""

    lr = 2.0e-4
    cold_w = torch.nn.Parameter(torch.zeros(64))
    warm_w = torch.nn.Parameter(torch.zeros(64))
    cold = torch.optim.AdamW([cold_w], lr=lr)
    warm = torch.optim.AdamW([warm_w], lr=lr)
    state = warm.state_dict()
    state["state"] = {
        0: {
            "step": torch.tensor(10_010.0),
            "exp_avg": torch.zeros(64),
            "exp_avg_sq": torch.full((64,), 1.0),
        }
    }
    warm.load_state_dict(state)
    grad = torch.full((64,), 3.7e-3)
    cold_w.grad = grad.clone()
    warm_w.grad = grad.clone()
    cold.step()
    warm.step()
    assert warm_w.detach().abs().max() < cold_w.detach().abs().max() / 100


def test_the_trainer_and_burn_prep_carry_no_lr_scheduler() -> None:
    """The only annealed quantity is tau; the LR is constant by construction."""

    for module_path in (
        Path(qbt.__file__),
        Path(qbr1.__file__),
    ):
        source = module_path.read_text(encoding="utf-8")
        for token in ("lr_scheduler", "LambdaLR", "CosineAnnealing", "OneCycleLR", "set_lr("):
            assert token not in source, f"{module_path.name} unexpectedly schedules the LR: {token}"


def test_tau_is_linear_and_is_the_only_schedule() -> None:
    start, end, total = 0.15, 0.05, qbr1.TOTAL_STEPS
    assert qbt.tau_for_step(0, total, start, end) == pytest.approx(start)
    assert qbt.tau_for_step(total - 1, total, start, end) == pytest.approx(end)
    midpoint = qbt.tau_for_step(total // 2, total, start, end)
    assert midpoint == pytest.approx((start + end) / 2, rel=1e-3)


def test_config_identity_ignores_resume_from_so_the_lever_is_scientifically_neutral() -> None:
    base = {
        "seed": 1,
        "resume_from": None,
        "output": "/a",
        "action": "train",
        "device": "mps",
        "launch_authorized": False,
        "scorer_lane": {"claimed": False, "claim_id": None},
        "metal_lane": {"claimed": False, "claim_id": None},
    }
    warm = dict(base, resume_from="/warm.pt", output="/b", action="resume_smoke")
    assert qbr1.config_identity(base) == qbr1.config_identity(warm)
    # MAIN binds authority and live claims at fire time; that must NOT invalidate the seed.
    authorized = dict(
        warm,
        launch_authorized=True,
        scorer_lane={"claimed": True, "claim_id": "live"},
        metal_lane={"claimed": True, "claim_id": "live"},
    )
    assert qbr1.config_identity(authorized) == qbr1.config_identity(warm)
    # Device is scientific: a cpu seed cannot be loaded into the mps burn.
    assert qbr1.config_identity(dict(base, device="cpu")) != qbr1.config_identity(base)


@requires_r10
def test_r10_warm_state_is_saturated_and_index_compatible() -> None:
    warm = ng1.r10_warm_state()
    assert warm["parameters"] == len(list(ng1._reference_model().parameters()))
    assert warm["optimizer_step_counters"] == [ng1.R10_EXPECTED_STEP]
    assert warm["source_step"] == 10_010
    assert 0.0 < warm["shadow_vs_live_relative_distance"] < 0.05


@requires_r10
def test_r10_terminal_lr_equals_the_cell_lr_so_lr_is_not_a_second_lever() -> None:
    warm = ng1.r10_warm_state()
    config = ng1.compile_warm_cell()
    report = ng1.assert_adamw_hyperparameters_match(config, warm)
    assert report["identical"] is True
    assert report["lr_is_object_tail"] == float(config["learning_rate"]) == 2.0e-4


@requires_r10
def test_warm_cell_differs_from_the_cold_control_only_in_the_transition() -> None:
    warm_config = ng1.compile_warm_cell()
    cold_config = json.loads(ng1.COLD_CONTROL_CONFIG.read_text(encoding="utf-8"))
    differing = {
        key
        for key in set(warm_config) | set(cold_config)
        if warm_config.get(key) != cold_config.get(key)
    }
    assert differing <= ng1.ALLOWED_WARM_MUTATIONS, f"warm cell moved too much: {differing}"
    assert warm_config["source_pins"] == cold_config["source_pins"]
    assert warm_config["source_revision"] == cold_config["source_revision"]
    assert warm_config["objective"] == cold_config["objective"]
    assert warm_config["schedule"] == cold_config["schedule"]
    assert warm_config["ema"] == cold_config["ema"]
    assert warm_config["learning_rate"] == cold_config["learning_rate"]
    assert warm_config["initial_state"] == cold_config["initial_state"]
    assert warm_config["launch_authorized"] is False
    assert warm_config["scorer_lane"]["claim_id"] is None
    assert warm_config["metal_lane"]["claim_id"] is None


@requires_r10
def test_warm_seed_round_trips_through_the_sealed_loader(tmp_path: Path) -> None:
    warm = ng1.r10_warm_state()
    config = ng1.compile_warm_cell()
    run_output = tmp_path / "run"
    run_output.mkdir()
    built = ng1.build_warm_seed(
        config, warm, device="cpu", run_output=run_output, seed_path=tmp_path / "seed.pt"
    )
    verification = built["verification"]
    assert verification["loaded_completed_steps"] == 0
    assert verification["optimizer_state_entries"] == warm["optimizer_state_entries"]
    assert verification["optimizer_step_counters_min"] == ng1.R10_EXPECTED_STEP
    assert verification["ema_num_updates"] == 0
    assert verification["ema_law_matched"] is True
    assert all(value == 0.0 for value in verification["margin_constraint_lambdas"].values())


@requires_r10
def test_warm_seed_identity_is_device_bound(tmp_path: Path) -> None:
    """A cpu-identity seed must not be loadable under the mps burn identity."""

    warm = ng1.r10_warm_state()
    config = ng1.compile_warm_cell()
    run_output = tmp_path / "run"
    run_output.mkdir()
    ng1.build_warm_seed(
        config, warm, device="cpu", run_output=run_output, seed_path=tmp_path / "seed.pt"
    )
    with pytest.raises(qbr1.QBR1Error, match="identity differs"):
        ng1.verify_warm_seed(config, tmp_path / "seed.pt", device="mps")


@requires_r10
def test_warm_seed_must_not_move_the_start_weights(tmp_path: Path) -> None:
    """The warm cell must begin at the control's exact weights; only moments may differ."""

    warm = ng1.r10_warm_state()
    config = ng1.compile_warm_cell()
    run_output = tmp_path / "run"
    run_output.mkdir()
    history = qbt.atomic_bytes(run_output / "history.jsonl", b"")
    payload = ng1.warm_seed_payload(config, warm, device="cpu", history_fact=history)
    name = next(iter(payload["live_state_dict"]))
    payload["live_state_dict"][name] = payload["live_state_dict"][name] + 1.0
    qbt.atomic_torch(tmp_path / "seed.pt", payload)
    with pytest.raises(ng1.NG1Error, match="changed the start weights"):
        ng1.verify_warm_seed(config, tmp_path / "seed.pt", device="cpu")


@requires_r10
def test_warm_seed_refuses_to_carry_a_second_lever(tmp_path: Path) -> None:
    warm = ng1.r10_warm_state()
    config = ng1.compile_warm_cell()
    run_output = tmp_path / "run"
    run_output.mkdir()
    history = qbt.atomic_bytes(run_output / "history.jsonl", b"")
    payload = ng1.warm_seed_payload(config, warm, device="cpu", history_fact=history)
    payload["margin_constraint_lambdas"] = {"Lane": 0.005, "Movable": 0.017}
    qbt.atomic_torch(tmp_path / "seed.pt", payload)
    with pytest.raises(ng1.NG1Error, match="margin-constraint multipliers"):
        ng1.verify_warm_seed(config, tmp_path / "seed.pt", device="cpu")


@requires_r10
def test_cold_control_excursion_decomposes_back_to_its_own_s_hat() -> None:
    """Recompute from components; never trust the summary field (operating manual sec.4)."""

    control = ng1.cold_control_receipt()
    parts = control["endpoint_excess_decomposition"]
    assert sum(parts.values()) == pytest.approx(control["endpoint_excess_over_warm_start"], rel=1e-9)
    assert control["endpoint_excess_over_warm_start"] > 0.0
    assert control["peak_step"] == 2000
    assert parts["d_seg"] / control["endpoint_excess_over_warm_start"] > 0.85


@requires_r10
def test_the_surrogate_lives_in_history_not_in_the_milestone_file() -> None:
    """Falsifier 2 must name a field that actually exists where it says it does."""

    milestone = json.loads(
        (ng1.COLD_CONTROL_RUN / "milestones/step_002000/MILESTONE.json").read_text(encoding="utf-8")
    )
    assert "seg_expected_flip_realized" not in milestone
    surrogate = ng1.surrogate_at_milestones(ng1.COLD_CONTROL_RUN / "history.jsonl")
    assert set(surrogate) == set(qbr1.MILESTONES) - {0}
    ordered = [surrogate[step] for step in sorted(surrogate)]
    assert ordered == sorted(ordered, reverse=True), "the control surrogate is monotone falling"


@requires_r10
def test_falsifiers_are_preregistered_against_the_measured_control() -> None:
    control = ng1.cold_control_receipt()
    rows = ng1.falsifiers(control)
    assert rows["secondary_free_read"]["cold_control_surrogate_by_step"]["5000"] < (
        rows["secondary_free_read"]["cold_control_surrogate_by_step"]["1000"]
    )
    assert rows["secondary_free_read"]["cold_control_d_seg_by_step"]["5000"] > (
        control["milestones"][0]["d_seg_hat"]
    )
    assert rows["primary"]["warm_start_S_hat"] == control["warm_start_S_hat"]
    assert set(rows["primary"]["cold_control_by_step"]) == {
        str(step) for step in qbr1.MILESTONES
    }
    assert "no_op_detector" in rows


def test_seal_refuses_to_truncate_a_started_run(tmp_path: Path) -> None:
    run_output = tmp_path / "run"
    run_output.mkdir()
    ng1.refuse_if_the_run_has_already_started(run_output)
    (run_output / "history.jsonl").write_bytes(b"")
    ng1.refuse_if_the_run_has_already_started(run_output)
    (run_output / "history.jsonl").write_bytes(b'{"completed_steps": 1}\n')
    with pytest.raises(ng1.NG1Error, match="started run"):
        ng1.refuse_if_the_run_has_already_started(run_output)
    (run_output / "history.jsonl").write_bytes(b"")
    (run_output / "RESULT.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ng1.NG1Error, match="completed run"):
        ng1.refuse_if_the_run_has_already_started(run_output)


def test_run_root_is_an_authorized_custody_root_and_not_the_live_burn() -> None:
    resolved = ng1.RUN_OUTPUT.resolve()
    assert qbt.QBR1_RETENTION_ROOT.resolve() in resolved.parents
    assert "ddm_wc3_qbr1_ema_law_cure" not in str(resolved)
    assert "ddm_wc3_qbr1_ema_law_cure" not in str(ng1.SMOKE_ROOT.resolve())


def test_displacement_is_a_real_l2_over_the_live_state(tmp_path: Path) -> None:
    reference = {"a": torch.zeros(4), "b": torch.zeros(3)}
    payload = {"live_state_dict": {"a": torch.full((4,), 0.5), "b": torch.zeros(3)}}
    torch.save(payload, tmp_path / "ck.pt")
    assert ng1._displacement(reference, str(tmp_path / "ck.pt")) == pytest.approx(
        math.sqrt(4 * 0.25)
    )


@requires_r10
def test_validate_warm_cell_refuses_any_second_mutation() -> None:
    config = ng1.compile_warm_cell()
    control = json.loads(ng1.COLD_CONTROL_CONFIG.read_text(encoding="utf-8"))
    report = ng1.validate_warm_cell(config, control)
    assert set(report["differing_keys"]) <= ng1.ALLOWED_WARM_MUTATIONS
    mutated = copy.deepcopy(config)
    mutated["learning_rate"] = 5.0e-5
    with pytest.raises(ng1.NG1Error, match="more than the transition"):
        ng1.validate_warm_cell(mutated, control)
    # Authorizing the cell trips the diff guard first; the explicit guard is the second line
    # of defence, reached only when the control is itself authorized.
    authorized = copy.deepcopy(config)
    authorized["launch_authorized"] = True
    with pytest.raises(ng1.NG1Error, match="more than the transition"):
        ng1.validate_warm_cell(authorized, control)
    authorized_control = copy.deepcopy(control)
    authorized_control["launch_authorized"] = True
    with pytest.raises(ng1.NG1Error, match="unauthorized"):
        ng1.validate_warm_cell(authorized, authorized_control)
    claimed = copy.deepcopy(config)
    claimed["scorer_lane"] = {"claimed": True, "claim_id": "x"}
    claimed_control = copy.deepcopy(control)
    claimed_control["scorer_lane"] = {"claimed": True, "claim_id": "x"}
    with pytest.raises(ng1.NG1Error, match="unbound for MAIN"):
        ng1.validate_warm_cell(claimed, claimed_control)


@requires_r10
def test_inherited_pins_are_verified_against_their_own_recorded_paths() -> None:
    config = ng1.compile_warm_cell()
    rows = ng1.verify_inherited_pins(config)
    assert set(rows) == set(config["source_pins"])
    broken = copy.deepcopy(config)
    broken["source_pins"]["qbt_trainer"]["sha256"] = "0" * 64
    with pytest.raises(ng1.NG1Error, match="drifted on disk"):
        ng1.verify_inherited_pins(broken)


def test_the_packet_schema_pin_drift_that_forced_pin_inheritance_is_cured() -> None:
    """Documents WHY the warm cell inherits pins, and that the reason has since been cured.

    At ng1's seal the working tree could not compile a QBR1-lineage cell: the packet-schema
    memo had drifted from its pin (worktree 7fe5285f6... vs pinned 5405ccd49...), so a fresh
    compile would have re-pinned the warm cell away from its own control.  MAIN's 4a7ae5ca0
    re-pinned the trainer to the memo's current bytes (eq1's addendum was append-only), so the
    working tree verifies again.  ng1's SEALED cell keeps the inherited pins either way -- a
    sealed config is never re-derived -- but the stated blocker is now HISTORY, and a test that
    still asserted the drift would be asserting a cured condition.
    """

    import hashlib

    sealed_tree_pin = "5405ccd499d14d28230874059e47d47f1f2818038519f1b27c97ed9377f132aa"
    path = Path(qbt.PIN_PATHS["packet_schema"])
    live = hashlib.sha256(path.read_bytes()).hexdigest()
    assert qbt.PINNED_SHA256["packet_schema"] != sealed_tree_pin, (
        "4a7ae5ca0's re-pin is expected in the working tree; the sealed QBR1 tree keeps the old one"
    )
    assert live == qbt.PINNED_SHA256["packet_schema"], (
        "the working tree's packet schema must match its own pin, else no cell compiles here"
    )
    assert qbt.verify_pins()["packet_schema"]["sha256"] == live
