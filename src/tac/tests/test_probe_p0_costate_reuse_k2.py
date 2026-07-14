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


def test_aggregate_charges_forward_guard_and_only_labels_accepted_regret() -> None:
    aggregate = probe.aggregate_records([_row(0, accepted=True), _row(1, accepted=False)])
    forward_share = probe.DIAGNOSTIC_FORWARD_SHARE
    expected = 2.0 / (1.0 + forward_share + 0.5 * (1.0 - forward_share))
    assert aggregate["diagnostic_teacher_slice_economics"]["conditional_speedup_x"] == pytest.approx(
        expected
    )
    assert aggregate["accepted_stale_minus_exact_regret"]["ce"]["mean"] == pytest.approx(-0.01)
    assert aggregate["all_eligible_stale_minus_exact_regret"]["ce"]["mean"] == pytest.approx(0.005)


def test_admission_requires_rate_above_derived_amdahl_gate_and_all_fidelity() -> None:
    threshold = probe.diagnostic_admission_threshold(probe.DIAGNOSTIC_FORWARD_SHARE)
    assert threshold["required_accept_fraction_strict_gt"] == pytest.approx(
        2.0 * probe.DIAGNOSTIC_FORWARD_SHARE / (1.0 - probe.DIAGNOSTIC_FORWARD_SHARE)
    )

    passing = probe.aggregate_records(
        [_row(index, accepted=index < 264) for index in range(600)]
    )
    gate = probe.evaluate_admission_gate(passing, complete_n600=True)
    assert gate["measured_accept_fraction"] == pytest.approx(0.44)
    assert gate["passed"] is True
    assert gate["diagnostic_teacher_slice_speedup_x"] > gate[
        "forward_elimination_amdahl_ceiling_x"
    ]

    below_rate = probe.aggregate_records(
        [_row(index, accepted=index < 258) for index in range(600)]
    )
    assert probe.evaluate_admission_gate(below_rate, complete_n600=True)["passed"] is False

    one_of_six_hundred = probe.aggregate_records(
        [_row(0, accepted=True)] + [_terminal_row(index) for index in range(1, 600)]
    )
    one_gate = probe.evaluate_admission_gate(one_of_six_hundred, complete_n600=True)
    assert one_gate["measured_accept_fraction"] == pytest.approx(1 / 600)
    assert one_gate["passed"] is False

    bad_gradient_rows = [_row(index, accepted=index < 264) for index in range(600)]
    bad_gradient_rows[0]["renderer_gradient_fidelity"]["relative_l2_error_fp32"] = 1.0
    bad_gradient_gate = probe.evaluate_admission_gate(
        probe.aggregate_records(bad_gradient_rows), complete_n600=True
    )
    assert bad_gradient_gate["passed"] is False
    assert (
        bad_gradient_gate["predicates"]["all_accepted_gradient_relative_l2_strict_lt_one"]
        is False
    )

    bad_regret_rows = [_row(index, accepted=index < 264) for index in range(600)]
    bad_regret_rows[0]["stale_minus_exact_regret"]["d_seg"] = 1e-6
    bad_regret_gate = probe.evaluate_admission_gate(
        probe.aggregate_records(bad_regret_rows), complete_n600=True
    )
    assert bad_regret_gate["passed"] is False
    assert (
        bad_regret_gate["predicates"]["all_accepted_stale_d_seg_regret_lte_exact"]
        is False
    )


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
    probe._atomic_json(
        probe._pair_path(tmp_path, 0), _sealed_row(assignment, "a" * 64)
    )
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
    contract = _contract(
        {
            "git_head_at_launch": "a" * 40,
            "source_custody": {},
            "objective_sha256": "b" * 64,
            "scorer_sha256": "c" * 64,
        }
    )
    assignment = _Assignment(0)
    (tmp_path / "pairs").mkdir()
    probe._atomic_json(
        probe._pair_path(tmp_path, 0), _sealed_row(assignment, contract["sha256"])
    )
    manifest = probe._stage_manifest(
        tmp_path, "v9", [assignment], contract["sha256"]
    )
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
    aggregate = probe.aggregate_records(records)
    gate = probe.evaluate_admission_gate(aggregate, complete_n600=False)
    admission_content = probe.build_admission_content(
        run_contract=contract,
        stage_manifest_custody=stage_custody,
        aggregate=aggregate,
        admission_gate=gate,
        admission_verdict="NOT_ADMITTED",
    )
    receipt_path = tmp_path / "measurement_receipt.json"
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
        "fidelity_gate": {
            "complete_n600": False,
            "calibration_admission_gate": gate,
            "admission": "NOT_ADMITTED",
            "live_trainer_activation": False,
            "runtime_exact_gradient_access": False,
        },
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
        },
    }
    probe._atomic_json(receipt_path, receipt)
    probe._atomic_json(
        tmp_path / "complete.json",
        {
            "schema": "p0_costate_reuse_k2_complete.v2",
            "receipt": "measurement_receipt.json",
            "receipt_bytes": receipt_path.stat().st_size,
            "receipt_sha256": probe._sha256(receipt_path),
        },
    )
    assert probe._load_completed_receipt(
        tmp_path,
        run_contract=contract,
        assignments=[assignment],
        checkpoint_names=["v9"],
        complete_n600=False,
    ) == receipt

    tampered = dict(receipt)
    tampered["measurement"] = {"n": 599}
    probe._atomic_json(receipt_path, tampered)
    with pytest.raises(probe.ProbeError, match="receipt custody changed"):
        probe._load_completed_receipt(
            tmp_path,
            run_contract=contract,
            assignments=[assignment],
            checkpoint_names=["v9"],
            complete_n600=False,
        )


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
