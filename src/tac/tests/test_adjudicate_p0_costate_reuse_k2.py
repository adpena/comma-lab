from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools/adjudicate_p0_costate_reuse_k2.py"


def _load_tool() -> Any:
    name = "_test_adjudicate_p0_costate_reuse_k2"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _file_custody(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    rendered_path = str(path if relative_to is None else path.relative_to(relative_to))
    return {
        "path": rendered_path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _source_bytes(source_dir: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(source_dir)): path.read_bytes()
        for path in source_dir.rglob("*")
        if path.is_file() and path.name not in {tool.OUTPUT_NAME, ".probe.lock"}
    }


def _fixture_roots(source_dir: Path) -> Any:
    contract = json.loads((source_dir / "run_contract.json").read_text())
    return tool._ExpectedSourceRoots(
        run_contract_sha256=contract["sha256"],
        objective_sha256=contract["payload"]["objective_sha256"],
        scorer_sha256=contract["payload"]["scorer_sha256"],
        measurement_receipt_sha256=_file_custody(source_dir / "measurement_receipt.json")["sha256"],
        complete_sha256=_file_custody(source_dir / "complete.json")["sha256"],
    )


def _adjudicate_fixture(source_dir: Path, *, expected_state_count: int, expected_stage_count: int) -> dict[str, Any]:
    return tool._adjudicate_with_expected_roots(
        source_dir,
        expected_state_count=expected_state_count,
        expected_stage_count=expected_stage_count,
        expected_roots=_fixture_roots(source_dir),
    )


def _rewrite_receipt_and_complete(source_dir: Path, receipt: dict[str, Any]) -> None:
    receipt_path = source_dir / "measurement_receipt.json"
    _write_json(receipt_path, receipt)
    receipt_custody = _file_custody(receipt_path, relative_to=source_dir)
    complete_path = source_dir / "complete.json"
    complete = json.loads(complete_path.read_text())
    complete["receipt_bytes"] = receipt_custody["bytes"]
    complete["receipt_sha256"] = receipt_custody["sha256"]
    _write_json(complete_path, complete)


def _rebind_stage_and_receipt_custody(source_dir: Path) -> None:
    receipt_path = source_dir / "measurement_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    stage_custody: list[dict[str, Any]] = []
    for prior in receipt["stage_manifest_custody"]:
        checkpoint_name = prior["checkpoint_name"]
        manifest_path = source_dir / f"stage_{checkpoint_name}_complete.json"
        manifest = json.loads(manifest_path.read_text())
        records: list[dict[str, Any]] = []
        for record in manifest["records"]:
            pair_index = record["pair_index"]
            records.append(
                {
                    "pair_index": pair_index,
                    **_file_custody(
                        source_dir / f"pairs/pair_{pair_index:04d}.json",
                        relative_to=source_dir,
                    ),
                }
            )
        manifest["records"] = records
        manifest["tree_sha256"] = tool.canonical_sha256(records)
        _write_json(manifest_path, manifest)
        stage_custody.append(
            {
                "checkpoint_name": checkpoint_name,
                "run_contract_sha256": manifest["run_contract_sha256"],
                "state_count": manifest["state_count"],
                "tree_sha256": manifest["tree_sha256"],
                **_file_custody(manifest_path),
            }
        )
    receipt["stage_manifest_custody"] = stage_custody
    receipt["admission_content"]["stage_manifest_custody"] = stage_custody
    receipt["admission_content_sha256"] = tool.canonical_sha256(receipt["admission_content"])
    _rewrite_receipt_and_complete(source_dir, receipt)


def _make_fixture(
    source_dir: Path,
    *,
    state_count: int = 4,
    stage_count: int = 2,
    accepted_count: int = 3,
    bad_gradient: bool = False,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    source_dir.mkdir(parents=True)
    (source_dir / ".probe.lock").touch()
    alpha = tool.EXPECTED_FORWARD_SHARE
    objective_spec = {
        "schema": "p0_costate_reuse_k2_objective.v2",
        "fallback": tool.EXPECTED_FALLBACK_SEMANTICS,
    }
    admission_spec = {
        "schema": "p0_costate_reuse_k2_admission.v1",
        "superseded_formula": "2-p*(1-alpha)",
    }
    input_custody = {
        "upstream/models/segnet.safetensors": {
            "path": "upstream/models/segnet.safetensors",
            "bytes": 11,
            "sha256": "1" * 64,
        },
        "upstream/models/posenet.safetensors": {
            "path": "upstream/models/posenet.safetensors",
            "bytes": 13,
            "sha256": "2" * 64,
        },
    }
    payload = {
        "schema": tool.SOURCE_SCHEMA,
        "git_head_at_launch": "a" * 40,
        "output_dir": str(source_dir),
        "objective_spec": objective_spec,
        "objective_sha256": tool.canonical_sha256(objective_spec),
        "admission_spec": admission_spec,
        "admission_spec_sha256": tool.canonical_sha256(admission_spec),
        "input_custody": input_custody,
        "scorer_sha256": tool.canonical_sha256(input_custody),
        "constants": {
            "n_pairs": state_count,
            "checkpoint_count": stage_count,
            "K_max": 2,
            "diagnostic_forward_share": alpha,
            "holdout_period": 5,
            "seed": 455,
        },
        "max_pairs": None,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    semantic_payload = {key: value for key, value in payload.items() if key != "git_head_at_launch"}
    contract = {
        "sha256": tool.canonical_sha256(semantic_payload),
        "payload": payload,
        "launch_provenance_sha256": tool.canonical_sha256(payload),
    }
    _write_json(source_dir / "run_contract.json", contract)

    stage_names = [f"v9_stage_{index}" for index in range(stage_count)]
    rows: list[dict[str, Any]] = []
    for pair_index in range(state_count):
        checkpoint_index = pair_index % stage_count
        accepted = pair_index < accepted_count
        relative_l2 = 1.0 if bad_gradient and pair_index == 0 else 0.25
        current_metrics = {"ce": 1.0, "d_seg": 0.2, "d_pose": 0.3}
        exact_second_metrics = {"ce": 0.85, "d_seg": 0.2, "d_pose": 0.3}
        stale_second_metrics = (
            {"ce": 0.9, "d_seg": 0.2, "d_pose": 0.3} if accepted else {"ce": 1.1, "d_seg": 0.21, "d_pose": 0.3}
        )
        assignment = {
            "pair_index": pair_index,
            "checkpoint_index": checkpoint_index,
            "checkpoint_name": stage_names[checkpoint_index],
            "split": "heldout" if pair_index % 5 == 455 % 5 else "train",
        }
        row = {
            "schema": tool.PAIR_SCHEMA,
            "run_contract_sha256": contract["sha256"],
            "assignment": assignment,
            "status": "REUSE_GUARD_ACCEPT" if accepted else "REUSE_GUARD_FALLBACK",
            "eligible_for_k2": True,
            "reuse_guard_accept": accepted,
            "reuse_guard": {
                "ce_strict_descent": stale_second_metrics["ce"] < current_metrics["ce"],
                "d_seg_nonworsening": (stale_second_metrics["d_seg"] <= current_metrics["d_seg"]),
                "d_pose_nonworsening": (stale_second_metrics["d_pose"] <= current_metrics["d_pose"]),
            },
            "current_metrics": current_metrics,
            "exact_second_metrics": exact_second_metrics,
            "stale_second_metrics": stale_second_metrics,
            "costate_fidelity": {
                "cosine_fp32": 0.9 - 0.01 * pair_index,
                "relative_l2_error_fp32": 0.4 + 0.01 * pair_index,
            },
            "renderer_gradient_fidelity": {
                "cosine_fp32": 0.95,
                "relative_l2_error_fp32": relative_l2,
            },
            "stale_minus_exact_regret": {
                "ce": float(stale_second_metrics["ce"] - exact_second_metrics["ce"]),
                "d_seg": float(stale_second_metrics["d_seg"] - exact_second_metrics["d_seg"]),
                "d_pose": float(stale_second_metrics["d_pose"] - exact_second_metrics["d_pose"]),
            },
        }
        row["record_content_sha256"] = tool.canonical_sha256(row)
        _write_json(source_dir / f"pairs/pair_{pair_index:04d}.json", row)
        rows.append(row)

    stage_manifests: list[dict[str, Any]] = []
    stage_receipt_custody: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint_name in enumerate(stage_names):
        indices = list(range(checkpoint_index, state_count, stage_count))
        record_custody = [
            {
                "pair_index": pair_index,
                **_file_custody(
                    source_dir / f"pairs/pair_{pair_index:04d}.json",
                    relative_to=source_dir,
                ),
            }
            for pair_index in indices
        ]
        manifest = {
            "schema": tool.STAGE_SCHEMA,
            "completed_at_utc": f"2026-07-14T00:00:0{checkpoint_index}Z",
            "run_contract_sha256": contract["sha256"],
            "checkpoint_name": checkpoint_name,
            "state_count": len(record_custody),
            "records": record_custody,
            "tree_sha256": tool.canonical_sha256(record_custody),
        }
        manifest_path = source_dir / f"stage_{checkpoint_name}_complete.json"
        _write_json(manifest_path, manifest)
        stage_manifests.append(manifest)
        stage_receipt_custody.append(
            {
                "checkpoint_name": checkpoint_name,
                "run_contract_sha256": contract["sha256"],
                "state_count": len(record_custody),
                "tree_sha256": manifest["tree_sha256"],
                **_file_custody(manifest_path),
            }
        )

    eligible_count = state_count
    fallback_count = state_count - accepted_count
    p = accepted_count / state_count
    accepted_rel_l2_passes = accepted_count - (1 if bad_gradient else 0)
    fixture_distributions = tool._accepted_row_metrics(rows)["fidelity_distributions"]
    costate_distribution = fixture_distributions["costate_fidelity"]
    renderer_distribution = fixture_distributions["renderer_gradient_fidelity"]

    def without_counts(distributions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            key: {name: value for name, value in distribution.items() if name != "row_count"}
            for key, distribution in distributions.items()
        }

    measurement = {
        "state_count": state_count,
        "unique_pair_count": state_count,
        "checkpoint_counts": dict(Counter(row["assignment"]["checkpoint_name"] for row in rows)),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "eligible_state_count": eligible_count,
        "terminal_or_blocked_state_count": 0,
        "reuse_guard_accept_count": accepted_count,
        "behavioral_full_facet_accept_count": accepted_count,
        "inconsistent_accept_flag_count": 0,
        "reuse_guard_fallback_count": fallback_count,
        "reuse_guard_accept_fraction": p,
        "calibration_fallback_count": fallback_count,
        "calibration_accept_fraction": p,
        "exact_costate_call_economics": {
            "calibration_two_step_opportunities": state_count,
            "accepted_reuses": accepted_count,
            "fallback_refreshes": fallback_count,
            "baseline_exact_costate_calls": 2 * state_count,
            "guarded_k2_exact_costate_calls": 2 * state_count - accepted_count,
            "exact_costate_calls_saved": accepted_count,
            "exact_call_amortization_x": 2 * state_count / (2 * state_count - accepted_count),
            "backward_call_reduction_fraction": accepted_count / (2 * state_count),
        },
        "costate_fidelity": {
            "cosine_fp32": costate_distribution["cosine_fp32"],
            "relative_l2_error_fp32": costate_distribution["relative_l2_error_fp32"],
        },
        "renderer_gradient_fidelity": {
            "cosine_fp32": renderer_distribution["cosine_fp32"],
            "relative_l2_error_fp32": renderer_distribution["relative_l2_error_fp32"],
            "accepted_calibration_row_count": accepted_count,
            "accepted_calibration_fidelity_present_count": accepted_count,
            "accepted_calibration_relative_l2_lt_one_count": accepted_rel_l2_passes,
            "accepted_calibration_relative_l2_threshold": 1.0,
            "accepted_calibration_relative_l2_comparator": "strict_lt",
        },
        "accepted_d_seg_regret_gate": {
            "accepted_calibration_row_count": accepted_count,
            "accepted_calibration_regret_present_count": accepted_count,
            "accepted_calibration_d_seg_regret_lte_zero_count": accepted_count,
            "threshold": 0.0,
            "comparator": "lte",
        },
        "accepted_stale_minus_exact_regret": without_counts(fixture_distributions["accepted_stale_minus_exact_regret"]),
        "all_eligible_stale_minus_exact_regret": without_counts(
            fixture_distributions["all_eligible_stale_minus_exact_regret"]
        ),
        "diagnostic_teacher_slice_economics": {
            "forward_share_alpha": alpha,
            "fallback_rate": fallback_count / state_count,
            "formula": "2/(1+alpha+q*(1-alpha))",
        },
    }
    old_gate_passed = not bad_gradient
    old_gate_spec = dict(admission_spec)
    old_gate = {
        "schema": admission_spec["schema"],
        "spec": old_gate_spec,
        "spec_sha256": tool.canonical_sha256(old_gate_spec),
        "passed": old_gate_passed,
    }
    old_verdict = "ADMIT_K2_GUARDED_REUSE" if old_gate_passed else "NOT_ADMITTED"
    admission_content = {
        "run_contract_sha256": contract["sha256"],
        "objective_sha256": payload["objective_sha256"],
        "scorer_sha256": payload["scorer_sha256"],
        "admission_spec_sha256": old_gate["spec_sha256"],
        "stage_manifest_custody": stage_receipt_custody,
        "aggregate_sha256": tool.canonical_sha256(measurement),
        "admission_verdict": old_verdict,
    }
    receipt = {
        "schema": tool.SOURCE_SCHEMA,
        "completed_at_utc": "2026-07-14T00:01:00Z",
        "axis": tool.EXPECTED_AXIS,
        "lane_id": tool.EXPECTED_LANE_ID,
        "run_contract": contract,
        "status": "completed",
        "admission_verdict": old_verdict,
        "n_pairs": state_count,
        "objective_sha256": payload["objective_sha256"],
        "scorer_sha256": payload["scorer_sha256"],
        "stage_manifest_custody": stage_receipt_custody,
        "admission_content": admission_content,
        "admission_content_sha256": tool.canonical_sha256(admission_content),
        "measurement": measurement,
        "fidelity_gate": {
            "complete_n600": True,
            "calibration_admission_gate": old_gate,
            "admission": old_verdict,
            "live_trainer_activation": False,
            "runtime_exact_gradient_access": False,
        },
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "whole_epoch_speedup": tool.WHOLE_EPOCH_OWED,
            "contest_cpu_cuda": "NOT_MEASURED",
        },
        "host": {"numpy": tool.np.__version__},
    }
    receipt_path = source_dir / "measurement_receipt.json"
    _write_json(receipt_path, receipt)
    receipt_custody = _file_custody(receipt_path, relative_to=source_dir)
    complete = {
        "schema": tool.COMPLETE_SCHEMA,
        "receipt": "measurement_receipt.json",
        "receipt_bytes": receipt_custody["bytes"],
        "receipt_sha256": receipt_custody["sha256"],
        "completed_at_utc": "2026-07-14T00:01:01Z",
    }
    _write_json(source_dir / "complete.json", complete)
    return {
        "contract": contract,
        "receipt": receipt,
        "stage_manifests": stage_manifests,
    }


def test_corrected_economics_boundary_algebra_is_strict() -> None:
    alpha = tool.EXPECTED_FORWARD_SHARE
    at_zero = tool.derive_corrected_economics(0.0, alpha)
    assert at_zero["guarded_expected_cost"] == pytest.approx(2.0 + alpha)
    assert at_zero["rejected_cycle_cost"] == pytest.approx(2.0 + alpha)
    assert at_zero["rejected_second_step_charge"]["total"] == pytest.approx(1.0 + alpha)
    assert at_zero["positive_speedup_strict"] is False

    at_alpha = tool.derive_corrected_economics(alpha, alpha)
    assert at_alpha["corrected_teacher_slice_speedup_x"] == pytest.approx(1.0)
    assert at_alpha["positive_speedup_strict"] is False

    threshold = 3.0 * alpha
    at_threshold = tool.derive_corrected_economics(threshold, alpha)
    assert at_threshold["beats_forward_elimination_amdahl_ceiling_strict"] is False
    assert at_threshold["corrected_teacher_slice_speedup_x"] == pytest.approx(
        at_threshold["forward_elimination_amdahl_ceiling_x"]
    )
    just_above = tool.derive_corrected_economics(math.nextafter(threshold, math.inf), alpha)
    assert just_above["beats_forward_elimination_amdahl_ceiling_strict"] is True

    at_one = tool.derive_corrected_economics(1.0, alpha)
    assert at_one["guarded_expected_cost"] == pytest.approx(1.0 + alpha)
    assert at_one["exact_backward_call_amortization_x"] == pytest.approx(2.0)
    assert at_one["exact_backward_call_reduction_fraction"] == pytest.approx(0.5)


def test_public_source_semantic_roots_are_exactly_pinned() -> None:
    assert tool.EXPECTED_RUN_CONTRACT_SHA256 == "e9c4a6629bcbc91876d2476b0bef051dfe56fe27d93076fa79f7225a5b62d56f"
    assert tool.EXPECTED_OBJECTIVE_SHA256 == "af5ae342f3987b82c2d3ee5bdb12dcfca1ecab07631fd545a9e723c15cb7c9e7"
    assert tool.EXPECTED_SCORER_SHA256 == "584f711dfb85163c38caf8976ebeda87698baefb45f9f5979539a8c176b6b73e"


def test_adjudicator_recursively_binds_source_and_is_idempotent(tmp_path: Path) -> None:
    source_dir = tmp_path / "sealed"
    _make_fixture(source_dir)
    before = _source_bytes(source_dir)

    wrapper = _adjudicate_fixture(source_dir, expected_state_count=4, expected_stage_count=2)
    output = source_dir / tool.OUTPUT_NAME
    first_output_bytes = output.read_bytes()
    assert wrapper["schema"] == tool.WRAPPER_SCHEMA
    assert wrapper["source_adjudication"]["original_admission_verdict_status"] == tool.SUPERSEDED_LABEL
    assert wrapper["corrected_admission_verdict"] == "ADMIT_K2_GUARDED_REUSE"
    assert wrapper["corrected_diagnostic_economics"]["guarded_expected_cost"] == pytest.approx(
        2.0 + tool.EXPECTED_FORWARD_SHARE - 0.75
    )
    assert wrapper["corrected_diagnostic_economics"]["corrected_teacher_slice_speedup_x"] == pytest.approx(
        2.0 / (2.0 + tool.EXPECTED_FORWARD_SHARE - 0.75)
    )
    assert len(wrapper["source_custody"]["pair_records"]) == 4
    assert len(wrapper["source_custody"]["stage_manifests"]) == 2
    distributions = wrapper["measurement_rederived_from_sealed_rows"]["fidelity_distributions"]
    assert distributions["costate_fidelity"]["row_count"] == 4
    assert distributions["costate_fidelity"]["relative_l2_error_fp32"]["median"] == pytest.approx(0.415)
    assert distributions["accepted_renderer_gradient_fidelity"]["row_count"] == 3
    assert distributions["accepted_renderer_gradient_fidelity"]["relative_l2_error_fp32"]["max"] == pytest.approx(0.25)
    assert distributions["accepted_stale_minus_exact_regret"]["d_seg"]["max"] == pytest.approx(0.0)
    assert wrapper["execution"]["teacher_calls"] == 0
    assert wrapper["execution"]["scorer_calls"] == 0
    assert wrapper["quantile_replay_environment"] == {
        "source_numpy_version": tool.np.__version__,
        "current_numpy_version": tool.np.__version__,
        "exact_version_match": True,
        "quantile_schema": "min/p10/median/mean/p90/max; NumPy default linear method",
    }
    assert wrapper["authority"]["promotion_eligible"] is False
    unsigned = dict(wrapper)
    claimed_content_hash = unsigned.pop("adjudication_content_sha256")
    assert claimed_content_hash == tool.canonical_sha256(unsigned)
    assert _source_bytes(source_dir) == before

    repeated = _adjudicate_fixture(source_dir, expected_state_count=4, expected_stage_count=2)
    assert repeated == wrapper
    assert output.read_bytes() == first_output_bytes
    assert _source_bytes(source_dir) == before


def test_adjudicator_emits_nonadmission_when_accepted_fidelity_fails(tmp_path: Path) -> None:
    source_dir = tmp_path / "bad_fidelity"
    _make_fixture(source_dir, bad_gradient=True)
    wrapper = _adjudicate_fixture(source_dir, expected_state_count=4, expected_stage_count=2)
    gate = wrapper["corrected_admission_gate"]
    assert gate["predicates"]["all_accepted_gradient_relative_l2_strict_lt_one"] is False
    assert gate["passed"] is False
    assert wrapper["corrected_admission_verdict"] == "NOT_ADMITTED"
    assert wrapper["authority"]["live_trainer_activation"] is False


def test_adjudicator_uses_source_fallback_operation_order(tmp_path: Path) -> None:
    source_dir = tmp_path / "fallback_float_order"
    _make_fixture(source_dir, state_count=3, stage_count=1, accepted_count=1)
    wrapper = _adjudicate_fixture(source_dir, expected_state_count=3, expected_stage_count=1)
    assert wrapper["measurement_rederived_from_sealed_rows"]["calibration_fallback_count"] == 2
    assert wrapper["corrected_diagnostic_economics"]["fallback_fraction_q"] == pytest.approx(1.0 - 1.0 / 3.0)


def test_adjudicator_rederives_guard_booleans_after_all_hashes_are_rebound(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "semantic_guard_tamper"
    _make_fixture(source_dir)
    row_path = source_dir / "pairs/pair_0000.json"
    row = json.loads(row_path.read_text())
    row["reuse_guard"]["ce_strict_descent"] = False
    row.pop("record_content_sha256")
    row["record_content_sha256"] = tool.canonical_sha256(row)
    _write_json(row_path, row)
    _rebind_stage_and_receipt_custody(source_dir)

    with pytest.raises(tool.AdjudicationError, match="guard predicates disagree with sealed metrics"):
        _adjudicate_fixture(source_dir, expected_state_count=4, expected_stage_count=2)
    assert not (source_dir / tool.OUTPUT_NAME).exists()


def test_adjudicator_binds_original_gate_spec_to_run_contract_after_rehash(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "gate_spec_tamper"
    _make_fixture(source_dir)
    receipt_path = source_dir / "measurement_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    original_gate = receipt["fidelity_gate"]["calibration_admission_gate"]
    original_gate["spec"]["tampered"] = True
    original_gate["spec_sha256"] = tool.canonical_sha256(original_gate["spec"])
    receipt["admission_content"]["admission_spec_sha256"] = original_gate["spec_sha256"]
    receipt["admission_content_sha256"] = tool.canonical_sha256(receipt["admission_content"])
    _rewrite_receipt_and_complete(source_dir, receipt)

    with pytest.raises(tool.AdjudicationError, match="run-contract admission spec"):
        _adjudicate_fixture(source_dir, expected_state_count=4, expected_stage_count=2)
    assert not (source_dir / tool.OUTPUT_NAME).exists()


def _tamper_row(source_dir: Path) -> None:
    path = source_dir / "pairs/pair_0000.json"
    payload = json.loads(path.read_text())
    payload["status"] = "TAMPERED"
    _write_json(path, payload)


def _tamper_manifest(source_dir: Path) -> None:
    path = source_dir / "stage_v9_stage_0_complete.json"
    payload = json.loads(path.read_text())
    payload["tree_sha256"] = "0" * 64
    _write_json(path, payload)


def _tamper_receipt(source_dir: Path) -> None:
    path = source_dir / "measurement_receipt.json"
    payload = json.loads(path.read_text())
    payload["objective_sha256"] = "0" * 64
    _write_json(path, payload)


def _tamper_complete(source_dir: Path) -> None:
    path = source_dir / "complete.json"
    payload = json.loads(path.read_text())
    payload["receipt_sha256"] = "0" * 64
    _write_json(path, payload)


def _tamper_contract(source_dir: Path) -> None:
    path = source_dir / "run_contract.json"
    payload = json.loads(path.read_text())
    payload["payload"]["objective_spec"]["fallback"] = "guard_forward_plus_backward_only"
    _write_json(path, payload)


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("row", _tamper_row),
        ("manifest", _tamper_manifest),
        ("receipt", _tamper_receipt),
        ("complete", _tamper_complete),
        ("contract", _tamper_contract),
    ],
)
def test_adjudicator_rejects_nested_mutation_without_output(
    tmp_path: Path,
    name: str,
    mutate: Callable[[Path], None],
) -> None:
    source_dir = tmp_path / name
    _make_fixture(source_dir)
    mutate(source_dir)
    with pytest.raises(tool.AdjudicationError):
        _adjudicate_fixture(source_dir, expected_state_count=4, expected_stage_count=2)
    assert not (source_dir / tool.OUTPUT_NAME).exists()


def test_adjudicator_requires_completion_seal_and_exact_pair_coverage(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    _make_fixture(incomplete)
    incomplete_roots = _fixture_roots(incomplete)
    (incomplete / "complete.json").unlink()
    with pytest.raises(tool.AdjudicationError, match="completion seal"):
        tool._adjudicate_with_expected_roots(
            incomplete,
            expected_state_count=4,
            expected_stage_count=2,
            expected_roots=incomplete_roots,
        )
    assert not (incomplete / tool.OUTPUT_NAME).exists()

    missing_pair = tmp_path / "missing_pair"
    _make_fixture(missing_pair)
    (missing_pair / "pairs/pair_0003.json").unlink()
    with pytest.raises(tool.AdjudicationError, match=r"missing=.*pair_0003"):
        _adjudicate_fixture(missing_pair, expected_state_count=4, expected_stage_count=2)
    assert not (missing_pair / tool.OUTPUT_NAME).exists()


def test_public_postseal_byte_roots_are_exact_and_clearance_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tool.EXPECTED_SOURCE_RECEIPT_SHA256 == "4c84c1f80ae7fc1b4ee76d28395405834e3eecd439155e4ebd79d4e81530506c"
    assert tool.EXPECTED_SOURCE_COMPLETE_SHA256 == "45ccbccee780d26bf350442ddf5551d62d483957c591b706fe5eb746dfbea34c"
    monkeypatch.setattr(tool, "EXPECTED_SOURCE_RECEIPT_SHA256", None)
    with pytest.raises(tool.AdjudicationError, match="blocked until reviewed receipt"):
        tool.adjudicate(tmp_path / "must_not_be_read")


def test_public_adjudicator_rejects_self_consistent_fake_n600_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "fake_n600"
    _make_fixture(source_dir, state_count=600, stage_count=3, accepted_count=400)
    fake_roots = _fixture_roots(source_dir)
    monkeypatch.setattr(
        tool,
        "EXPECTED_SOURCE_RECEIPT_SHA256",
        fake_roots.measurement_receipt_sha256,
    )
    monkeypatch.setattr(tool, "EXPECTED_SOURCE_COMPLETE_SHA256", fake_roots.complete_sha256)
    with pytest.raises(tool.AdjudicationError, match="code-reviewed v3 root"):
        tool.adjudicate(source_dir)
    assert not (source_dir / tool.OUTPUT_NAME).exists()


@pytest.mark.parametrize("extra_kind", ["json", "pair", "stage", "symlink"])
def test_adjudicator_rejects_every_unbound_source_entry(
    tmp_path: Path,
    extra_kind: str,
) -> None:
    source_dir = tmp_path / f"extra_{extra_kind}"
    _make_fixture(source_dir)
    if extra_kind == "json":
        _write_json(source_dir / "unbound.json", {"unbound": True})
    elif extra_kind == "pair":
        _write_json(source_dir / "pairs/pair_9999.json", {"unbound": True})
    elif extra_kind == "stage":
        _write_json(source_dir / "stage_unbound_complete.json", {"unbound": True})
    else:
        (source_dir / "unbound_link").symlink_to(source_dir / "run_contract.json")
    with pytest.raises(tool.AdjudicationError, match=r"unbound symlink|inventory changed"):
        _adjudicate_fixture(source_dir, expected_state_count=4, expected_stage_count=2)
    assert not (source_dir / tool.OUTPUT_NAME).exists()


def test_adjudicator_refuses_nonidentical_existing_wrapper(tmp_path: Path) -> None:
    source_dir = tmp_path / "nonidentical_wrapper"
    _make_fixture(source_dir)
    output = source_dir / tool.OUTPUT_NAME
    output.write_text("{}\n")
    with pytest.raises(tool.AdjudicationError, match="overwrite non-identical"):
        _adjudicate_fixture(source_dir, expected_state_count=4, expected_stage_count=2)
    assert output.read_text() == "{}\n"


def test_adjudicator_rejects_numpy_version_drift_before_quantile_comparison(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "numpy_version_drift"
    _make_fixture(source_dir)
    receipt_path = source_dir / "measurement_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["host"]["numpy"] = "0.0.synthetic-drift"
    _rewrite_receipt_and_complete(source_dir, receipt)
    with pytest.raises(tool.AdjudicationError, match="NumPy version drift"):
        _adjudicate_fixture(source_dir, expected_state_count=4, expected_stage_count=2)
    assert not (source_dir / tool.OUTPUT_NAME).exists()


def test_adjudicator_refuses_source_while_probe_lock_is_held(tmp_path: Path) -> None:
    source_dir = tmp_path / "locked"
    _make_fixture(source_dir)
    descriptor = os.open(source_dir / ".probe.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(tool.AdjudicationError, match="still running"):
            _adjudicate_fixture(source_dir, expected_state_count=4, expected_stage_count=2)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert not (source_dir / tool.OUTPUT_NAME).exists()
