from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools/probe_p0_costate_reuse_k2.py"


def _load_tool():
    name = "_test_probe_p0_costate_reuse_k2"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_tool()


@dataclass(frozen=True)
class _Assignment:
    pair_index: int
    checkpoint_name: str = "v9"

    def to_dict(self) -> dict:
        return {"pair_index": self.pair_index, "checkpoint_name": self.checkpoint_name}


def _row(pair_index: int, *, accepted: bool) -> dict:
    return {
        "assignment": {"pair_index": pair_index, "checkpoint_name": "v9"},
        "status": "REUSE_GUARD_ACCEPT" if accepted else "REUSE_GUARD_FALLBACK",
        "eligible_for_k2": True,
        "reuse_guard_accept": accepted,
        "reuse_guard": {
            "ce_strict_descent": accepted,
            "d_seg_nonworsening": True,
            "d_pose_nonworsening": True,
        },
        "costate_fidelity": {"cosine_fp32": 0.9, "relative_l2_error_fp32": 0.4},
        "renderer_gradient_fidelity": {
            "cosine_fp32": 0.95,
            "relative_l2_error_fp32": 0.3,
        },
        "stale_minus_exact_regret": {
            "ce": -0.01 if accepted else 0.02,
            "d_seg": 0.0 if accepted else 0.01,
            "d_pose": 0.0,
        },
    }


def _sealed_row(assignment: _Assignment, contract_sha256: str) -> dict:
    row = {
        "schema": probe.PAIR_SCHEMA,
        "run_contract_sha256": contract_sha256,
        "assignment": assignment.to_dict(),
        "eligible_for_k2": False,
        "reuse_guard_accept": False,
        "status": "TERMINAL_OR_BLOCKED_AT_ANCHOR",
    }
    row["record_content_sha256"] = probe._canonical_sha256(row)
    return row


def _terminal_row(pair_index: int) -> dict:
    return {
        "assignment": {"pair_index": pair_index, "checkpoint_name": "v9"},
        "status": "TERMINAL_OR_BLOCKED_AT_ANCHOR",
        "eligible_for_k2": False,
        "reuse_guard_accept": False,
    }


def test_exact_call_amortization_counts_fallback_without_fake_speedup() -> None:
    calls = probe.exact_call_amortization(calibration_states=4, accepted_reuses=3)
    assert calls["baseline_exact_costate_calls"] == 8
    assert calls["guarded_k2_exact_costate_calls"] == 5
    assert calls["exact_call_amortization_x"] == pytest.approx(1.6)
    assert calls["backward_call_reduction_fraction"] == pytest.approx(3 / 8)


def test_raw_probe_timing_routing_requires_corrected_adjudication_without_go() -> None:
    routing = probe.raw_probe_timing_routing()
    assert routing == {
        "status": probe.RAW_PROBE_TIMING_STATUS,
        "corrected_adjudication_required": True,
        "operator_go_request_eligible": False,
        "operator_go_granted": False,
    }
    surfaces = probe.raw_probe_timing_surfaces()
    assert surfaces == {
        "fidelity_gate": {
            "in_loop_timing": routing["status"],
            "timing_routing": routing,
        },
        "authority": {
            "whole_epoch_speedup": routing["status"],
            "operator_go_request_eligible": False,
            "operator_go_granted": False,
        },
    }


def test_aggregate_charges_forward_guard_and_only_labels_accepted_regret() -> None:
    aggregate = probe.aggregate_records([_row(0, accepted=True), _row(1, accepted=False)])
    forward_share = probe.DIAGNOSTIC_FORWARD_SHARE
    expected = 2.0 / (2.0 + forward_share - 0.5)
    assert aggregate["diagnostic_teacher_slice_economics"]["conditional_speedup_x"] == pytest.approx(expected)
    rejection = aggregate["diagnostic_teacher_slice_economics"]["rejected_second_step_charge"]
    assert rejection["guard_forward"] == pytest.approx(forward_share)
    assert rejection["rollback_exact_forward_plus_backward_refresh"] == 1.0
    assert rejection["total"] == pytest.approx(1.0 + forward_share)
    assert aggregate["diagnostic_teacher_slice_economics"]["whole_epoch_speedup"] == (probe.RAW_PROBE_TIMING_STATUS)
    assert aggregate["accepted_stale_minus_exact_regret"]["ce"]["mean"] == pytest.approx(-0.01)
    assert aggregate["all_eligible_stale_minus_exact_regret"]["ce"]["mean"] == pytest.approx(0.005)


def test_legacy_v2_rederivation_reproduces_superseded_equation_from_rows() -> None:
    rows = [_row(0, accepted=True), _row(1, accepted=False)]
    legacy = probe._aggregate_records_legacy_v2(rows)
    corrected = probe.aggregate_records(rows)
    alpha = probe.DIAGNOSTIC_FORWARD_SHARE
    legacy_economics = legacy["diagnostic_teacher_slice_economics"]
    corrected_economics = corrected["diagnostic_teacher_slice_economics"]

    assert legacy_economics["formula"] == "2/(1+forward_share+fallback_rate*(1-forward_share))"
    assert legacy_economics["whole_epoch_speedup"] == probe.HISTORICAL_EMBEDDED_WHOLE_EPOCH_LABEL
    assert legacy_economics["conditional_speedup_x"] == pytest.approx(2.0 / (1.0 + alpha + 0.5 * (1 - alpha)))
    assert legacy_economics["required_accept_fraction_strict_gt"] == pytest.approx(2 * alpha / (1 - alpha))
    assert "guarded_expected_cost" not in legacy_economics
    assert "rejected_second_step_charge" not in legacy_economics
    assert corrected_economics["formula"] == "2/(2+forward_share-accept_fraction)"
    assert corrected_economics["conditional_speedup_x"] != legacy_economics["conditional_speedup_x"]
    assert probe._canonical_sha256(probe.LEGACY_V2_ADMISSION_SPEC) == (probe.EXPECTED_LEGACY_ADMISSION_SPEC_SHA256)


def test_admission_requires_rate_above_derived_amdahl_gate_and_all_fidelity() -> None:
    threshold = probe.diagnostic_admission_threshold(probe.DIAGNOSTIC_FORWARD_SHARE)
    assert threshold["required_accept_fraction_strict_gt"] == pytest.approx(3.0 * probe.DIAGNOSTIC_FORWARD_SHARE)

    passing = probe.aggregate_records([_row(index, accepted=index < 322) for index in range(600)])
    gate = probe.evaluate_admission_gate(passing, complete_n600=True)
    assert gate["measured_accept_fraction"] == pytest.approx(322 / 600)
    assert gate["passed"] is True
    assert gate["diagnostic_teacher_slice_speedup_x"] > gate["forward_elimination_amdahl_ceiling_x"]

    below_rate = probe.aggregate_records([_row(index, accepted=index < 321) for index in range(600)])
    assert probe.evaluate_admission_gate(below_rate, complete_n600=True)["passed"] is False

    one_of_six_hundred = probe.aggregate_records(
        [_row(0, accepted=True)] + [_terminal_row(index) for index in range(1, 600)]
    )
    one_gate = probe.evaluate_admission_gate(one_of_six_hundred, complete_n600=True)
    assert one_gate["measured_accept_fraction"] == pytest.approx(1 / 600)
    assert one_gate["passed"] is False

    bad_gradient_rows = [_row(index, accepted=index < 322) for index in range(600)]
    bad_gradient_rows[0]["renderer_gradient_fidelity"]["relative_l2_error_fp32"] = 1.0
    bad_gradient_gate = probe.evaluate_admission_gate(probe.aggregate_records(bad_gradient_rows), complete_n600=True)
    assert bad_gradient_gate["passed"] is False
    assert bad_gradient_gate["predicates"]["all_accepted_gradient_relative_l2_strict_lt_one"] is False

    bad_regret_rows = [_row(index, accepted=index < 322) for index in range(600)]
    bad_regret_rows[0]["stale_minus_exact_regret"]["d_seg"] = 1e-6
    bad_regret_gate = probe.evaluate_admission_gate(probe.aggregate_records(bad_regret_rows), complete_n600=True)
    assert bad_regret_gate["passed"] is False
    assert bad_regret_gate["predicates"]["all_accepted_stale_d_seg_regret_lte_exact"] is False


@pytest.mark.parametrize(
    ("p_expression", "positive", "beats_ceiling"),
    [
        (lambda alpha: 0.0, False, False),
        (lambda alpha: alpha, False, False),
        (lambda alpha: 3.0 * alpha, True, False),
        (lambda alpha: 1.0, True, True),
    ],
)
def test_corrected_economics_strict_boundaries(p_expression, positive: bool, beats_ceiling: bool) -> None:
    alpha = probe.DIAGNOSTIC_FORWARD_SHARE
    p = p_expression(alpha)
    expected_cost = 2.0 + alpha - p
    threshold = probe.diagnostic_admission_threshold(alpha)
    assert (p > alpha) is positive
    assert 2.0 / expected_cost == pytest.approx(2.0 / (2.0 + alpha - p))
    assert (p > threshold["required_accept_fraction_strict_gt"]) is beats_ceiling


def test_stage_manifest_detects_tampered_row_and_manifest(tmp_path: Path) -> None:
    contract_sha256 = "a" * 64
    assignments = [_Assignment(0), _Assignment(1)]
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    for assignment in assignments:
        probe._atomic_json(
            probe._pair_path(tmp_path, assignment.pair_index),
            _sealed_row(assignment, contract_sha256),
        )
    manifest = probe._stage_manifest(tmp_path, "v9", assignments, contract_sha256)
    assert manifest["run_contract_sha256"] == contract_sha256
    assert manifest["state_count"] == 2

    row_path = probe._pair_path(tmp_path, 0)
    tampered_row = json.loads(row_path.read_text())
    tampered_row["status"] = "TAMPERED"
    probe._atomic_json(row_path, tampered_row)
    with pytest.raises(probe.ProbeError, match="pair record content custody drift"):
        probe._verify_stage_manifest(tmp_path, "v9", assignments, contract_sha256)

    probe._atomic_json(row_path, _sealed_row(assignments[0], contract_sha256))
    manifest_path = tmp_path / "stage_v9_complete.json"
    tampered_manifest = json.loads(manifest_path.read_text())
    tampered_manifest["tree_sha256"] = "0" * 64
    probe._atomic_json(manifest_path, tampered_manifest)
    with pytest.raises(probe.ProbeError, match="manifest custody drift at tree_sha256"):
        probe._verify_stage_manifest(tmp_path, "v9", assignments, contract_sha256)


def test_row_resume_and_load_refuse_stale_contract(tmp_path: Path) -> None:
    assignment = _Assignment(0)
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    probe._atomic_json(probe._pair_path(tmp_path, 0), _sealed_row(assignment, "a" * 64))
    assert len(probe._load_records(tmp_path, [assignment], "a" * 64)) == 1
    with pytest.raises(probe.ProbeError, match="run contract drift"):
        probe._load_records(tmp_path, [assignment], "b" * 64)


def test_admission_content_binds_objective_scorers_records_and_aggregate() -> None:
    aggregate = probe.aggregate_records([_row(0, accepted=True)])
    gate = probe.evaluate_admission_gate(aggregate, complete_n600=False)
    contract = {
        "sha256": "a" * 64,
        "payload": {"objective_sha256": "b" * 64, "scorer_sha256": "c" * 64},
    }
    manifests = [{"tree_sha256": "d" * 64, "sha256": "e" * 64}]
    content = probe.build_admission_content(
        run_contract=contract,
        stage_manifest_custody=manifests,
        aggregate=aggregate,
        admission_gate=gate,
        admission_verdict="NOT_ADMITTED",
    )
    assert content["run_contract_sha256"] == "a" * 64
    assert content["objective_sha256"] == "b" * 64
    assert content["scorer_sha256"] == "c" * 64
    assert content["stage_manifest_custody"] == manifests
    original = probe._canonical_sha256(content)
    content["objective_sha256"] = "f" * 64
    assert probe._canonical_sha256(content) != original


def test_storage_plan_must_be_unblocked_and_bound_to_output(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    plan = tmp_path / "plan.json"
    payload = {
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "selected_tier": "local",
        "selected_workload_root": str(output),
        "requested_bytes": 1,
        "selected_free_bytes": 2,
        "operator_storage_policy": {"local_disk_enabled": True},
    }
    plan.write_text(json.dumps(payload), encoding="utf-8")
    assert probe._validate_storage_plan(plan, output)["selected_tier"] == "local"

    payload["blockers"] = ["selected_workload_root_missing"]
    plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(probe.ProbeError, match="remains blocked"):
        probe._validate_storage_plan(plan, output)


def test_storage_plan_requires_integer_capacity_or_bound_selected_tier(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    plan = tmp_path / "plan.json"
    payload = {
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "selected_tier": "local",
        "selected_workload_root": str(output),
        "requested_bytes": 2,
        "operator_storage_policy": {"local_disk_enabled": True},
        "tiers": [
            {
                "name": "local",
                "workload_root": str(output),
                "eligible": True,
                "requested_bytes": 2,
                "free_bytes": 3,
            }
        ],
    }
    plan.write_text(json.dumps(payload), encoding="utf-8")
    assert probe._validate_storage_plan(plan, output)["selected_free_bytes"] == 3

    payload["tiers"] = []
    plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(probe.ProbeError, match="selected_free_bytes"):
        probe._validate_storage_plan(plan, output)

    payload["selected_free_bytes"] = 1
    payload["requested_bytes"] = 2
    plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(probe.ProbeError, match="does not reserve"):
        probe._validate_storage_plan(plan, output)

    payload["selected_free_bytes"] = 3
    payload["requested_bytes"] = "2"
    plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(probe.ProbeError, match="requested_bytes"):
        probe._validate_storage_plan(plan, output)


def _contract(payload: dict) -> dict:
    semantic = {key: value for key, value in payload.items() if key != "git_head_at_launch"}
    return {
        "sha256": probe._canonical_sha256(semantic),
        "payload": payload,
        "launch_provenance_sha256": probe._canonical_sha256(payload),
    }


def _completed_receipt_fixture(tmp_path: Path, *, routing_class: str) -> dict:
    if routing_class == probe.CURRENT_TIMING_ROUTING_CLASS:
        admission_spec = probe.ADMISSION_SPEC
    elif routing_class == probe.LEGACY_TIMING_ROUTING_CLASS:
        admission_spec = probe.LEGACY_V2_ADMISSION_SPEC
    else:
        raise AssertionError(f"unsupported fixture routing class: {routing_class}")
    contract = _contract(
        {
            "schema": probe.SCHEMA,
            "git_head_at_launch": "a" * 40,
            "output_dir": str(tmp_path.resolve()),
            "source_custody": {},
            "objective_sha256": "b" * 64,
            "scorer_sha256": "c" * 64,
            "admission_spec": admission_spec,
            "admission_spec_sha256": probe._canonical_sha256(admission_spec),
            "constants": {"n_pairs": 1},
            "max_pairs": None,
        }
    )
    assignment = _Assignment(0)
    (tmp_path / "pairs").mkdir()
    probe._atomic_json(probe._pair_path(tmp_path, 0), _sealed_row(assignment, contract["sha256"]))
    manifest = probe._stage_manifest(tmp_path, "v9", [assignment], contract["sha256"])
    manifest_path = tmp_path / "stage_v9_complete.json"
    stage_custody = [
        {
            "checkpoint_name": "v9",
            "run_contract_sha256": contract["sha256"],
            "state_count": 1,
            "tree_sha256": manifest["tree_sha256"],
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": probe._sha256(manifest_path),
        }
    ]
    records = probe._load_records(tmp_path, [assignment], contract["sha256"])
    if routing_class == probe.LEGACY_TIMING_ROUTING_CLASS:
        aggregate = probe._aggregate_records_legacy_v2(records)
        gate = probe._evaluate_legacy_v2_admission_gate(aggregate, complete_n600=False)
    else:
        aggregate = probe.aggregate_records(records)
        gate = probe.evaluate_admission_gate(aggregate, complete_n600=False)
    admission_content = probe.build_admission_content(
        run_contract=contract,
        stage_manifest_custody=stage_custody,
        aggregate=aggregate,
        admission_gate=gate,
        admission_verdict="NOT_ADMITTED",
    )
    fidelity_gate = {
        "complete_n600": False,
        "calibration_admission_gate": gate,
        "admission": "NOT_ADMITTED",
        "live_trainer_activation": False,
        "runtime_exact_gradient_access": False,
    }
    authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    if routing_class == probe.CURRENT_TIMING_ROUTING_CLASS:
        timing_surfaces = probe.raw_probe_timing_surfaces()
        fidelity_gate.update(timing_surfaces["fidelity_gate"])
        authority.update(timing_surfaces["authority"])
    elif routing_class == probe.LEGACY_TIMING_ROUTING_CLASS:
        fidelity_gate["in_loop_timing"] = probe.HISTORICAL_SOURCE_IN_LOOP_TIMING_LABEL
        authority["whole_epoch_speedup"] = probe.HISTORICAL_EMBEDDED_WHOLE_EPOCH_LABEL
    receipt = {
        "schema": probe.SCHEMA,
        "status": "completed",
        "n_pairs": 1,
        "run_contract": contract,
        "objective_sha256": "b" * 64,
        "scorer_sha256": "c" * 64,
        "stage_manifest_custody": stage_custody,
        "measurement": aggregate,
        "admission_verdict": "NOT_ADMITTED",
        "admission_content": admission_content,
        "admission_content_sha256": probe._canonical_sha256(admission_content),
        "fidelity_gate": fidelity_gate,
        "authority": authority,
    }
    receipt_path = tmp_path / "measurement_receipt.json"
    complete_path = tmp_path / "complete.json"
    run_contract_path = tmp_path / "run_contract.json"
    probe._atomic_json(run_contract_path, contract)
    probe._atomic_json(receipt_path, receipt)
    probe._atomic_json(
        complete_path,
        {
            "schema": "p0_costate_reuse_k2_complete.v2",
            "receipt": "measurement_receipt.json",
            "receipt_bytes": receipt_path.stat().st_size,
            "receipt_sha256": probe._sha256(receipt_path),
        },
    )
    return {
        "contract": contract,
        "assignment": assignment,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "complete_path": complete_path,
        "run_contract_path": run_contract_path,
    }


def _legacy_roots_for_fixture(fixture: dict) -> object:
    contract = fixture["contract"]
    return probe._ExpectedLegacyReceiptRoots(
        output_dir=fixture["receipt_path"].parent.resolve(),
        receipt_sha256=probe._sha256(fixture["receipt_path"]),
        complete_sha256=probe._sha256(fixture["complete_path"]),
        run_contract_file_sha256=probe._sha256(fixture["run_contract_path"]),
        run_contract_sha256=contract["sha256"],
        objective_sha256=contract["payload"]["objective_sha256"],
        scorer_sha256=contract["payload"]["scorer_sha256"],
        state_count=1,
    )


def _load_completed_fixture(tmp_path: Path, fixture: dict, *, expected_legacy_roots=None) -> dict:
    loaded = probe._load_completed_receipt(
        tmp_path,
        run_contract=fixture["contract"],
        assignments=[fixture["assignment"]],
        checkpoint_names=["v9"],
        complete_n600=False,
        expected_legacy_roots=expected_legacy_roots,
    )
    assert loaded is not None
    return loaded


def _reseal_completed_fixture(fixture: dict, receipt: dict) -> None:
    receipt_path = fixture["receipt_path"]
    probe._atomic_json(receipt_path, receipt)
    complete = json.loads(fixture["complete_path"].read_text())
    complete["receipt_bytes"] = receipt_path.stat().st_size
    complete["receipt_sha256"] = probe._sha256(receipt_path)
    probe._atomic_json(fixture["complete_path"], complete)


def _patch_public_resume_fixture(monkeypatch, tmp_path: Path, fixture: dict, *, expected_roots=None):
    module_loads = []

    class _Round2:
        CHECKPOINTS = (("v9", Path("unused"), 0),)

        @staticmethod
        def deterministic_replay_assignments(**_kwargs):
            return [fixture["assignment"]]

    class _Torch:
        set_num_threads = staticmethod(lambda _value: None)
        set_num_interop_threads = staticmethod(lambda _value: None)
        manual_seed = staticmethod(lambda _value: None)
        use_deterministic_algorithms = staticmethod(lambda _value: None)

    def fake_load_module(name: str, relative: str):
        module_loads.append((name, relative))
        return _Round2 if relative == "tools/probe_frozen_replay_convex_head.py" else object()

    monkeypatch.setattr(probe, "REPO", tmp_path)
    monkeypatch.setattr(probe, "_validate_storage_plan", lambda *_args: {"fixture": True})
    monkeypatch.setattr(probe, "_acquire_lock", lambda _output_dir: -1)
    monkeypatch.setattr(probe, "_release_lock", lambda _descriptor: None)
    monkeypatch.setattr(probe, "_load_module", fake_load_module)
    monkeypatch.setitem(sys.modules, "torch", _Torch)
    if expected_roots is not None:
        monkeypatch.setattr(probe, "_public_expected_legacy_receipt_roots", lambda: expected_roots)
    args = probe.argparse.Namespace(
        output_dir=fixture["receipt_path"].parent,
        storage_plan=tmp_path / "storage.json",
        max_pairs=None,
        resume=True,
    )
    return module_loads, args


def test_resume_contract_ignores_only_launch_head_provenance() -> None:
    prior = _contract(
        {
            "git_head_at_launch": "a" * 40,
            "source_custody": {"probe.py": {"sha256": "b" * 64}},
            "input_custody": {"gt": {"sha256": "c" * 64}},
        }
    )
    current = _contract(
        {
            "git_head_at_launch": "d" * 40,
            "source_custody": {"probe.py": {"sha256": "b" * 64}},
            "input_custody": {"gt": {"sha256": "c" * 64}},
        }
    )
    assert probe._validate_resume_contract(prior, current) is prior

    drifted = _contract(
        {
            "git_head_at_launch": "d" * 40,
            "source_custody": {"probe.py": {"sha256": "e" * 64}},
            "input_custody": {"gt": {"sha256": "c" * 64}},
        }
    )
    with pytest.raises(probe.ProbeError, match="resume run contract changed"):
        probe._validate_resume_contract(prior, drifted)


def test_completed_receipt_is_immutable_and_contract_bound(tmp_path: Path) -> None:
    fixture = _completed_receipt_fixture(tmp_path, routing_class=probe.CURRENT_TIMING_ROUTING_CLASS)
    receipt = fixture["receipt"]
    receipt_path = fixture["receipt_path"]
    assert _load_completed_fixture(tmp_path, fixture) == receipt

    tampered = dict(receipt)
    tampered["measurement"] = {"n": 599}
    probe._atomic_json(receipt_path, tampered)
    with pytest.raises(probe.ProbeError, match="receipt custody changed"):
        _load_completed_fixture(tmp_path, fixture)


def test_current_completed_receipt_contains_no_historical_timer_routing(tmp_path: Path) -> None:
    fixture = _completed_receipt_fixture(tmp_path, routing_class=probe.CURRENT_TIMING_ROUTING_CLASS)
    serialized = json.dumps(fixture["receipt"], sort_keys=True)

    assert probe.HISTORICAL_EMBEDDED_WHOLE_EPOCH_LABEL not in serialized
    assert probe.HISTORICAL_SOURCE_IN_LOOP_TIMING_LABEL not in serialized
    assert probe.RAW_PROBE_TIMING_STATUS in serialized
    assert fixture["receipt"]["fidelity_gate"]["calibration_admission_gate"]["whole_epoch_speedup"] == (
        probe.RAW_PROBE_TIMING_STATUS
    )


def test_legacy_completed_receipt_is_public_normalized_without_source_byte_mutation(tmp_path: Path) -> None:
    fixture = _completed_receipt_fixture(tmp_path, routing_class=probe.LEGACY_TIMING_ROUTING_CLASS)
    expected_roots = _legacy_roots_for_fixture(fixture)
    receipt_path = fixture["receipt_path"]
    source_receipt_bytes = receipt_path.read_bytes()
    source_receipt_sha256 = probe._sha256(receipt_path)
    source_complete_bytes = fixture["complete_path"].read_bytes()

    normalized = _load_completed_fixture(tmp_path, fixture, expected_legacy_roots=expected_roots)
    current = probe.raw_probe_timing_surfaces()
    assert normalized["fidelity_gate"]["in_loop_timing"] == current["fidelity_gate"]["in_loop_timing"]
    assert normalized["fidelity_gate"]["timing_routing"] == current["fidelity_gate"]["timing_routing"]
    for field, value in current["authority"].items():
        assert normalized["authority"][field] == value
    assert normalized["source_receipt_sha256"] == source_receipt_sha256
    historical = normalized["historical_source_routing"]
    assert historical["classification"] == probe.LEGACY_TIMING_ROUTING_CLASS
    assert historical["control_routing_authority"] is False
    assert normalized["measurement"] == fixture["receipt"]["measurement"]
    assert (
        normalized["fidelity_gate"]["calibration_admission_gate"]
        == (fixture["receipt"]["fidelity_gate"]["calibration_admission_gate"])
    )
    economics = normalized["historical_embedded_economics"]
    assert economics["status"] == "SUPERSEDED_LEGACY_V2_ECONOMICS_NON_AUTHORITY"
    assert economics["embedded_historical_economics_preserved"] is True
    assert economics["corrected_adjudication_required"] is True
    assert economics["promotion_authority"] is False
    assert economics["legacy_diagnostic"]["formula"] == ("2/(1+forward_share+fallback_rate*(1-forward_share))")
    assert economics["corrected_rederived_diagnostic"]["formula"] == ("2/(2+forward_share-accept_fraction)")
    assert economics["legacy_aggregate_sha256"] == probe._canonical_sha256(fixture["receipt"]["measurement"])
    assert economics["corrected_rederived_aggregate_sha256"] != economics["legacy_aggregate_sha256"]
    assert receipt_path.read_bytes() == source_receipt_bytes
    assert probe._sha256(receipt_path) == source_receipt_sha256
    assert fixture["complete_path"].read_bytes() == source_complete_bytes
    assert fixture["receipt"]["fidelity_gate"]["in_loop_timing"] == "OWED_OPERATOR_GO"
    assert "timing_routing" not in fixture["receipt"]["fidelity_gate"]


def test_public_resume_returns_pinned_legacy_normalization_before_current_source_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "experiments/results/legacy"
    output_dir.mkdir(parents=True)
    fixture = _completed_receipt_fixture(output_dir, routing_class=probe.LEGACY_TIMING_ROUTING_CLASS)
    expected_roots = _legacy_roots_for_fixture(fixture)
    receipt_bytes = fixture["receipt_path"].read_bytes()
    complete_bytes = fixture["complete_path"].read_bytes()
    module_loads, args = _patch_public_resume_fixture(
        monkeypatch,
        tmp_path,
        fixture,
        expected_roots=expected_roots,
    )

    normalized = probe.run(args)
    assert normalized["fidelity_gate"]["in_loop_timing"] == probe.RAW_PROBE_TIMING_STATUS
    assert normalized["authority"]["operator_go_request_eligible"] is False
    assert normalized["authority"]["operator_go_granted"] is False
    assert normalized["source_receipt_sha256"] == expected_roots.receipt_sha256
    assert normalized["measurement"]["diagnostic_teacher_slice_economics"]["formula"] == (
        "2/(1+forward_share+fallback_rate*(1-forward_share))"
    )
    assert normalized["historical_embedded_economics"]["promotion_authority"] is False
    assert module_loads == [("_p0_k2_round2", "tools/probe_frozen_replay_convex_head.py")]
    assert fixture["receipt_path"].read_bytes() == receipt_bytes
    assert fixture["complete_path"].read_bytes() == complete_bytes


@pytest.mark.parametrize("completed", (False, True))
def test_public_resume_current_or_incomplete_receipt_still_enforces_current_contract(
    tmp_path: Path,
    monkeypatch,
    completed: bool,
) -> None:
    output_dir = tmp_path / "experiments/results/current_or_incomplete"
    output_dir.mkdir(parents=True)
    fixture = _completed_receipt_fixture(output_dir, routing_class=probe.CURRENT_TIMING_ROUTING_CLASS)
    if not completed:
        fixture["complete_path"].unlink()
    _module_loads, args = _patch_public_resume_fixture(monkeypatch, tmp_path, fixture)
    monkeypatch.setattr(
        probe,
        "_run_contract",
        lambda **_kwargs: (_ for _ in ()).throw(probe.ProbeError("CURRENT_CONTRACT_ENFORCED")),
    )

    with pytest.raises(probe.ProbeError, match="CURRENT_CONTRACT_ENFORCED"):
        probe.run(args)


def test_unpinned_legacy_label_receipt_is_refused_even_after_outer_reseal(tmp_path: Path) -> None:
    fixture = _completed_receipt_fixture(tmp_path, routing_class=probe.LEGACY_TIMING_ROUTING_CLASS)
    expected_roots = _legacy_roots_for_fixture(fixture)
    with pytest.raises(probe.ProbeError, match="not bound to reviewed immutable roots"):
        _load_completed_fixture(tmp_path, fixture)

    tampered = json.loads(json.dumps(fixture["receipt"]))
    tampered["axis"] = "RESEALED_UNPINNED_SYNTHETIC"
    _reseal_completed_fixture(fixture, tampered)
    with pytest.raises(probe.ProbeError, match="not the pinned immutable receipt"):
        probe._pinned_legacy_receipt_present(tmp_path, expected_roots)


def test_public_resume_rederives_actual_pinned_legacy_receipt_read_only_when_present(monkeypatch) -> None:
    roots = probe._public_expected_legacy_receipt_roots()
    required = tuple(
        roots.output_dir / name for name in ("measurement_receipt.json", "complete.json", "run_contract.json")
    )
    if not all(path.is_file() for path in required):
        pytest.skip("ignored local sealed K2 receipt is unavailable")

    def sealed_tree_hashes() -> dict[str, str]:
        return {
            str(path.relative_to(roots.output_dir)): probe._sha256(path)
            for path in roots.output_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        }

    before = sealed_tree_hashes()
    module_loads: list[str] = []
    real_load_module = probe._load_module

    def read_only_load_module(name: str, relative: str):
        module_loads.append(relative)
        assert relative == "tools/probe_frozen_replay_convex_head.py"
        return real_load_module(name, relative)

    monkeypatch.setattr(probe, "_validate_storage_plan", lambda *_args: {"read_only_test": True})
    monkeypatch.setattr(probe, "_acquire_lock", lambda _output_dir: -1)
    monkeypatch.setattr(probe, "_release_lock", lambda _descriptor: None)
    monkeypatch.setattr(probe, "_load_module", read_only_load_module)
    normalized = probe.run(
        probe.argparse.Namespace(
            output_dir=roots.output_dir,
            storage_plan=probe.DEFAULT_STORAGE_PLAN,
            max_pairs=None,
            resume=True,
        )
    )

    economics = normalized["historical_embedded_economics"]
    assert normalized["fidelity_gate"]["in_loop_timing"] == probe.RAW_PROBE_TIMING_STATUS
    assert normalized["authority"]["operator_go_request_eligible"] is False
    assert normalized["authority"]["operator_go_granted"] is False
    assert economics["legacy_diagnostic"]["conditional_speedup_x"] == pytest.approx(1.4538672169368423)
    assert economics["corrected_rederived_diagnostic"]["conditional_speedup_x"] == pytest.approx(1.4099643443401577)
    assert economics["corrected_adjudication_required"] is True
    assert economics["promotion_authority"] is False
    assert module_loads == ["tools/probe_frozen_replay_convex_head.py"]
    assert sealed_tree_hashes() == before


@pytest.mark.parametrize(
    "tamper_case",
    ("mixed_legacy_status", "missing_routing", "arbitrary_authority", "go_enabled"),
)
def test_hash_consistent_completed_receipt_timing_tamper_is_refused(tmp_path: Path, tamper_case: str) -> None:
    fixture = _completed_receipt_fixture(tmp_path, routing_class=probe.CURRENT_TIMING_ROUTING_CLASS)
    tampered = json.loads(json.dumps(fixture["receipt"]))
    if tamper_case == "mixed_legacy_status":
        tampered["fidelity_gate"]["in_loop_timing"] = probe.HISTORICAL_SOURCE_IN_LOOP_TIMING_LABEL
    elif tamper_case == "missing_routing":
        tampered["fidelity_gate"].pop("timing_routing")
    elif tamper_case == "arbitrary_authority":
        tampered["authority"]["whole_epoch_speedup"] = "ARBITRARY"
    elif tamper_case == "go_enabled":
        tampered["authority"]["operator_go_request_eligible"] = True
    else:
        raise AssertionError(tamper_case)
    _reseal_completed_fixture(fixture, tampered)

    with pytest.raises(probe.ProbeError, match="timing routing changed"):
        _load_completed_fixture(tmp_path, fixture)


def test_hash_consistent_incomplete_completed_receipt_is_refused(tmp_path: Path) -> None:
    contract = _contract(
        {
            "git_head_at_launch": "a" * 40,
            "objective_sha256": "b" * 64,
            "scorer_sha256": "c" * 64,
        }
    )
    receipt_path = tmp_path / "measurement_receipt.json"
    probe._atomic_json(receipt_path, {"status": "completed", "run_contract": contract})
    probe._atomic_json(
        tmp_path / "complete.json",
        {
            "schema": "p0_costate_reuse_k2_complete.v2",
            "receipt": "measurement_receipt.json",
            "receipt_bytes": receipt_path.stat().st_size,
            "receipt_sha256": probe._sha256(receipt_path),
        },
    )
    with pytest.raises(probe.ProbeError, match="receipt status changed"):
        probe._load_completed_receipt(
            tmp_path,
            run_contract=contract,
            assignments=[_Assignment(0)],
            checkpoint_names=["v9"],
            complete_n600=False,
        )


def test_objective_identity_is_stable_and_content_derived() -> None:
    first = probe._objective_sha256()
    assert first == probe._canonical_sha256(probe.OBJECTIVE_SPEC)
    assert len(first) == 64
