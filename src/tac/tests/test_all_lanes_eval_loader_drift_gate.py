from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ALL_LANES = REPO / "tools" / "all_lanes_preflight.py"
NON_PROMOTABLE_FIELDS = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}


def _valid_device_axis_custody() -> dict[str, object]:
    return {
        "score_axis": "diagnostic_loader_drift",
        "claimed_score_axes": [],
        "score_claim_axis": "none",
        "contest_cpu_axis_claim": False,
        "contest_cuda_axis_claim": False,
        "contest_cuda_claim": False,
        "contest_cpu_claim": False,
        "macos_cpu_advisory_claim": False,
        "mps_claim": False,
        "promotion_eligible": False,
        "score_claim_valid": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
    }


def _valid_custody_labels() -> dict[str, object]:
    return {
        "artifact_kind": "diagnostic_loader_forward_drift_probe",
        "score_path": "not_run",
        "score_claim_axis": "none",
        "contest_cpu_axis_claim": False,
        "contest_cuda_axis_claim": False,
        "dispatch_attempted": False,
        "dispatch_claim_required_before_remote_run": True,
        "diagnostic_non_promotable": True,
    }


def _load_all_lanes_module() -> Any:
    spec = importlib.util.spec_from_file_location("all_lanes_eval_loader_drift_test", ALL_LANES)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_intended_cells() -> list[dict[str, object]]:
    rows = []
    for cell_id in (
        "cpu_av",
        "cuda_dali",
        "cuda_av_shared_input",
        "cpu_dali",
    ):
        rows.append(
            {
                **NON_PROMOTABLE_FIELDS,
                "cell_id": cell_id,
                "available": cell_id == "cpu_av",
                "unsupported_codes": [] if cell_id == "cpu_av" else ["cuda_available"],
            }
        )
    return rows


def _valid_cell_discriminator_plan() -> list[dict[str, object]]:
    return [
        {
            **NON_PROMOTABLE_FIELDS,
            "comparison_id": comparison_id,
            "available": False,
            "unavailable_codes": ["cuda_dali:cuda_available"],
        }
        for comparison_id in (
            "raw_decoder_input_byte_drift_pre_network",
            "forward_kernel_drift_fixed_pyav_input",
            "forward_kernel_drift_fixed_dali_input",
            "decoder_effect_fixed_cpu_forward",
            "decoder_effect_fixed_cuda_forward",
        )
    ]


def _probe_contract_fields() -> dict[str, object]:
    return {
        **NON_PROMOTABLE_FIELDS,
        "score_axis": "diagnostic_loader_drift",
        "device_axis_custody": _valid_device_axis_custody(),
        "custody_labels": _valid_custody_labels(),
        "intended_cells": _valid_intended_cells(),
        "cell_discriminator_plan": _valid_cell_discriminator_plan(),
        "forward_matrix_complete": False,
        "forward_matrix_summary": {
            **NON_PROMOTABLE_FIELDS,
            "requested": False,
            "complete": False,
            "status": "not_requested",
            "required_cell_ids": [
                "cpu_av",
                "cuda_dali",
                "cuda_av_shared_input",
                "cpu_dali",
            ],
            "unavailable_cell_ids": [
                "cuda_dali",
                "cuda_av_shared_input",
                "cpu_dali",
            ],
            "forward_row_count": 0,
        },
        "local_prerequisite_summary": {
            "cuda_available": False,
            "dali_available": False,
            "cuda_dali_runtime_available": None,
            "missing_cuda_dali_prerequisite_codes": ["cuda_available"],
            "missing_cuda_dali_prerequisite_reasons": ["cuda_available=false"],
        },
        "future_remote_run_contract": {
            **NON_PROMOTABLE_FIELDS,
            "requires_dispatch_claim_before_remote_gpu_run": True,
            "diagnostic_command": [
                ".venv/bin/python",
                "tools/probe_eval_loader_drift.py",
                "--run-forward-cells",
            ],
            "claim_command_template": ["tools/claim_lane_dispatch.py", "claim"],
        },
    }


def _probe_payload(
    *,
    unavailable_class: str,
    unavailable_codes: list[str],
    unavailable_reason: str,
) -> dict[str, object]:
    return {
        **_probe_contract_fields(),
        "comparison_available": False,
        "comparison_unavailable_class": unavailable_class,
        "comparison_unavailable_reason": unavailable_reason,
        "comparison_unavailable_reasons": [unavailable_reason],
        "comparison_unavailable_codes": unavailable_codes,
    }


def _stub_probe_run(module: Any, monkeypatch: Any, payload: dict[str, object], *, returncode: int) -> None:
    class Result:
        stdout = ""
        stderr = ""

        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def fake_run(args: list[str], *, capture_output: bool, text: bool) -> Result:
        assert capture_output is True
        assert text is True
        output_path = Path(args[args.index("--json-out") + 1])
        output_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return Result(returncode)

    monkeypatch.setattr(module.subprocess, "run", fake_run)


def test_eval_loader_drift_gate_rejects_probe_runtime_error_json(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = _probe_payload(
        unavailable_class=module.EVAL_LOADER_DRIFT_PROBE_RUNTIME_ERROR_CLASS,
        unavailable_codes=[module.EVAL_LOADER_DRIFT_PROBE_RUNTIME_ERROR_CLASS],
        unavailable_reason="probe_runtime_error: RuntimeError: decoder failed",
    )
    _stub_probe_run(module, monkeypatch, payload, returncode=2)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert "non-prerequisite reason" in output
    assert "probe_runtime_error" in output


def test_eval_loader_drift_gate_accepts_known_missing_prereq_json(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = _probe_payload(
        unavailable_class=module.EVAL_LOADER_DRIFT_MISSING_PREREQ_CLASS,
        unavailable_codes=["cuda_available"],
        unavailable_reason="cuda_available=false",
    )
    _stub_probe_run(module, monkeypatch, payload, returncode=2)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is True
    assert "known missing prerequisite(s): cuda_available" in output


def test_eval_loader_drift_gate_rejects_malformed_comparison_rows(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = {
        **_probe_contract_fields(),
        "comparison_available": True,
        "comparison_rows": [42],
    }
    _stub_probe_run(module, monkeypatch, payload, returncode=0)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert "comparison schema invalid" in output
    assert "row 0: expected object" in output


def test_eval_loader_drift_gate_rejects_sequence_or_shape_mismatch(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = {
        **_probe_contract_fields(),
        "comparison_available": True,
        "comparison_rows": [
            {
                "path_match": True,
                "sequence_index_match": False,
                "comparison": {
                    "shape_match": False,
                    "numel": 100,
                    "max_abs_lsb": 2.0,
                    "mean_abs_lsb": 0.5,
                    "rms_abs_lsb": 0.7,
                    "nonzero_fraction": 0.2,
                },
            }
        ],
    }
    _stub_probe_run(module, monkeypatch, payload, returncode=0)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert "sequence index mismatch" in output
    assert "comparison shape mismatch" in output


def test_eval_loader_drift_gate_rejects_missing_drift_metrics(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = {
        **_probe_contract_fields(),
        "comparison_available": True,
        "comparison_rows": [
            {
                "path_match": True,
                "sequence_index_match": True,
                "comparison": {"shape_match": True, "numel": 100},
            }
        ],
    }
    _stub_probe_run(module, monkeypatch, payload, returncode=0)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert "max_abs_lsb must be numeric" in output
    assert "rms_abs_lsb must be numeric" in output


def test_eval_loader_drift_gate_rejects_nonfinite_drift_metrics(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = {
        **_probe_contract_fields(),
        "comparison_available": True,
        "comparison_rows": [
            {
                "path_match": True,
                "sequence_index_match": True,
                "comparison": {
                    "shape_match": True,
                    "numel": 100,
                    "max_abs_lsb": 2.0,
                    "mean_abs_lsb": 0.5,
                    "rms_abs_lsb": float("nan"),
                    "nonzero_fraction": 0.2,
                },
            }
        ],
    }
    _stub_probe_run(module, monkeypatch, payload, returncode=0)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert "rms_abs_lsb must be finite" in output


def test_eval_loader_drift_gate_accepts_complete_comparison_row(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = {
        **_probe_contract_fields(),
        "comparison_available": True,
        "comparison_rows": [
            {
                "path_match": True,
                "sequence_index_match": True,
                "comparison": {
                    "shape_match": True,
                    "numel": 100,
                    "max_abs_lsb": 2.0,
                    "mean_abs_lsb": 0.5,
                    "rms_abs_lsb": 0.7,
                    "nonzero_fraction": 0.2,
                },
            }
        ],
    }
    _stub_probe_run(module, monkeypatch, payload, returncode=0)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is True
    assert "DALI-vs-PyAV decoded-RGB comparison emitted" in output


def test_eval_loader_drift_gate_rejects_score_valid_diagnostic_payload(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = _probe_payload(
        unavailable_class=module.EVAL_LOADER_DRIFT_MISSING_PREREQ_CLASS,
        unavailable_codes=["cuda_available"],
        unavailable_reason="cuda_available=false",
    )
    payload["score_claim_valid"] = True
    _stub_probe_run(module, monkeypatch, payload, returncode=2)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert "score_claim_valid" in output


def test_eval_loader_drift_gate_rejects_dispatch_ready_diagnostic_payload(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = _probe_payload(
        unavailable_class=module.EVAL_LOADER_DRIFT_MISSING_PREREQ_CLASS,
        unavailable_codes=["cuda_available"],
        unavailable_reason="cuda_available=false",
    )
    payload["ready_for_exact_eval_dispatch"] = True
    _stub_probe_run(module, monkeypatch, payload, returncode=2)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert "ready_for_exact_eval_dispatch=false" in output


def test_eval_loader_drift_gate_rejects_dispatch_attempted_diagnostic_payload(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = _probe_payload(
        unavailable_class=module.EVAL_LOADER_DRIFT_MISSING_PREREQ_CLASS,
        unavailable_codes=["cuda_available"],
        unavailable_reason="cuda_available=false",
    )
    payload["dispatch_attempted"] = True
    _stub_probe_run(module, monkeypatch, payload, returncode=2)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert "dispatch_attempted=false" in output


def test_eval_loader_drift_gate_rejects_dispatch_attempted_2x2_plan(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = _probe_payload(
        unavailable_class=module.EVAL_LOADER_DRIFT_MISSING_PREREQ_CLASS,
        unavailable_codes=["cuda_available"],
        unavailable_reason="cuda_available=false",
    )
    intended_cells = payload["intended_cells"]
    assert isinstance(intended_cells, list)
    intended_cells[0]["dispatch_attempted"] = True
    _stub_probe_run(module, monkeypatch, payload, returncode=2)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert "2x2 plan schema invalid" in output
    assert "cell cpu_av: dispatch_attempted must be false" in output


def test_eval_loader_drift_gate_rejects_contest_axis_diagnostic_payload(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = _probe_payload(
        unavailable_class=module.EVAL_LOADER_DRIFT_MISSING_PREREQ_CLASS,
        unavailable_codes=["cuda_available"],
        unavailable_reason="cuda_available=false",
    )
    payload["score_axis"] = "contest_cuda"
    _stub_probe_run(module, monkeypatch, payload, returncode=2)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert "diagnostic_loader_drift axis" in output


def test_eval_loader_drift_gate_rejects_device_axis_claims(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = _probe_payload(
        unavailable_class=module.EVAL_LOADER_DRIFT_MISSING_PREREQ_CLASS,
        unavailable_codes=["cuda_available"],
        unavailable_reason="cuda_available=false",
    )
    payload["device_axis_custody"] = _valid_device_axis_custody()
    payload["device_axis_custody"]["contest_cuda_claim"] = True
    _stub_probe_run(module, monkeypatch, payload, returncode=2)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert "contest_cuda_claim=false" in output


def test_eval_loader_drift_gate_rejects_axis_alias_claims(monkeypatch) -> None:
    module = _load_all_lanes_module()
    payload = _probe_payload(
        unavailable_class=module.EVAL_LOADER_DRIFT_MISSING_PREREQ_CLASS,
        unavailable_codes=["cuda_available"],
        unavailable_reason="cuda_available=false",
    )
    device_axis_custody = payload["device_axis_custody"]
    assert isinstance(device_axis_custody, dict)
    device_axis_custody["contest_cuda_axis_claim"] = True
    device_axis_custody["claimed_score_axes"] = ["contest_cuda"]
    device_axis_custody["mps_claim"] = True
    custody_labels = payload["custody_labels"]
    assert isinstance(custody_labels, dict)
    custody_labels["contest_cuda_axis_claim"] = True
    _stub_probe_run(module, monkeypatch, payload, returncode=2)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert (
        "contest_cuda_axis_claim=false" in output
        or "claimed_score_axes must stay empty" in output
    )
