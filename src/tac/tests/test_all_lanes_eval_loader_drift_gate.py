from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ALL_LANES = REPO / "tools" / "all_lanes_preflight.py"


def _load_all_lanes_module() -> Any:
    spec = importlib.util.spec_from_file_location("all_lanes_eval_loader_drift_test", ALL_LANES)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _probe_payload(
    *,
    unavailable_class: str,
    unavailable_codes: list[str],
    unavailable_reason: str,
) -> dict[str, object]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "score_axis": "diagnostic_loader_drift",
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
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "score_axis": "diagnostic_loader_drift",
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
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "score_axis": "diagnostic_loader_drift",
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
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "score_axis": "diagnostic_loader_drift",
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
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "score_axis": "diagnostic_loader_drift",
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
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "score_axis": "diagnostic_loader_drift",
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
    assert "exact-eval dispatch-ready" in output


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
    payload["device_axis_custody"] = {
        "contest_cuda_claim": True,
        "contest_cpu_claim": False,
        "macos_cpu_advisory_claim": False,
        "promotion_eligible": False,
        "score_claim_valid": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    _stub_probe_run(module, monkeypatch, payload, returncode=2)

    passed, output = module._run_eval_loader_drift_probe_gate()

    assert passed is False
    assert "contest_cuda_claim=false" in output
