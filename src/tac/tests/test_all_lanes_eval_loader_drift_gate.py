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
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
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
