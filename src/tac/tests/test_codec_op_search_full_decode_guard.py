from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]


def _load_tool(filename: str, module_name: str):
    tool_path = REPO / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cma_tool = _load_tool("codec_op_cma_search.py", "codec_op_cma_guard_test")
optuna_tool = _load_tool(
    "codec_op_optuna_search.py",
    "codec_op_optuna_guard_test",
)


class _Result:
    def __init__(self, *, bytes_out: int, op_state: dict[str, Any]) -> None:
        self.blob = b"fixture"
        self.op_state = op_state
        self.bytes_out = bytes_out


class _DroppingCodecOp:
    def __init__(self, **params: Any) -> None:
        del params

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> _Result:
        del context
        return _Result(bytes_out=123, op_state={"x": state_dict["x"]})

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        del blob, context
        return {"x": op_state["x"]}


class _ShapeMismatchCodecOp:
    def __init__(self, **params: Any) -> None:
        del params

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> _Result:
        del context
        return _Result(bytes_out=124, op_state={"x": state_dict["x"]})

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        del blob, context
        return {"x": op_state["x"][:1].reshape(1, 1)}


class _IdentityCodecOp:
    def __init__(self, *, scale: float = 0.0, quality: int = 1) -> None:
        self.scale = float(scale)
        self.quality = int(quality)

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> _Result:
        del context
        bytes_out = 100 + 10 * self.quality + round(abs(self.scale) * 1_000)
        return _Result(bytes_out=bytes_out, op_state={"x": state_dict["x"]})

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        del blob, context
        return {"x": op_state["x"] + self.scale}


def _state_dict() -> dict[str, torch.Tensor]:
    return {
        "x": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "y": torch.tensor([3.0], dtype=torch.float32),
    }


def _specs(tool: Any) -> list[Any]:
    return [
        tool.ParamSpec("scale", "float", 0.0, 0.2, 0.0),
        tool.ParamSpec("quality", "int", 1, 5, 3),
    ]


@pytest.mark.parametrize("tool", [cma_tool, optuna_tool])
def test_full_decode_guard_rejects_missing_tensor_keys_by_default(tool: Any) -> None:
    ev = tool._evaluate(_DroppingCodecOp, {}, _state_dict(), eval_idx=0)
    tool.annotate_pareto_frontier([ev])

    assert ev.error == "ValueError: CodecOp decode did not reconstruct every input tensor key"
    assert ev.decode_coverage_required is True
    assert ev.partial_decode_waived is False
    assert ev.decode_coverage_status == "failed"
    assert ev.expected_tensor_count == 2
    assert ev.matched_tensor_count == 1
    assert ev.missing_tensor_keys == ["y"]
    assert ev.matched_tensor_keys == ["x"]
    assert ev.fitness is None
    assert ev.pareto_frontier is False


@pytest.mark.parametrize("tool", [cma_tool, optuna_tool])
def test_full_decode_guard_rejects_shape_mismatch_by_default(tool: Any) -> None:
    ev = tool._evaluate(_ShapeMismatchCodecOp, {}, _state_dict(), eval_idx=0)
    tool.annotate_pareto_frontier([ev])

    assert ev.error == "ValueError: CodecOp decode did not reconstruct every input tensor key"
    assert ev.shape_mismatch_tensor_keys == ["x"]
    assert ev.missing_tensor_keys == ["y"]
    assert ev.matched_tensor_count == 0
    assert ev.fitness is None
    assert ev.pareto_frontier is False


@pytest.mark.parametrize("tool", [cma_tool, optuna_tool])
def test_partial_decode_requires_explicit_waiver_reason(tool: Any) -> None:
    with pytest.raises(SystemExit, match="waiver-reason is required"):
        tool._validate_partial_decode_waiver(True, None)

    ev = tool._evaluate(
        _DroppingCodecOp,
        {},
        _state_dict(),
        eval_idx=0,
        require_full_decode=False,
    )
    assert ev.error == "ValueError: partial decode waiver requires a reason"
    assert ev.decode_coverage_status == "waiver_invalid"


@pytest.mark.parametrize("tool", [cma_tool, optuna_tool])
def test_partial_decode_waiver_is_audited_and_remains_planning_only(
    tool: Any,
    tmp_path: Path,
) -> None:
    reason = "fixture subset search over x tensor only"
    ev = tool._evaluate(
        _DroppingCodecOp,
        {},
        _state_dict(),
        eval_idx=0,
        require_full_decode=False,
        partial_decode_waiver_reason=reason,
    )
    tool.annotate_pareto_frontier([ev])

    assert ev.error is None
    assert ev.decode_coverage_required is False
    assert ev.partial_decode_waived is True
    assert ev.partial_decode_waiver_reason == reason
    assert ev.decode_coverage_status == "partial_waived"
    assert ev.missing_tensor_keys == ["y"]
    assert ev.fitness is not None
    assert ev.pareto_frontier is True

    ledger_path = tmp_path / "ledger.jsonl"
    tool.append_atom_ledger_rows(
        ledger_path,
        [ev],
        op_module="fixture.module",
        op_class="DroppingCodecOp",
        substrate_label="fixture",
    )
    row = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert row["decode_coverage_required"] is False
    assert row["partial_decode_waived"] is True
    assert row["partial_decode_waiver_reason"] == reason
    assert row["decode_coverage_status"] == "partial_waived"
    assert row["dispatchable"] is False
    assert row["promotion_eligible"] is False
    assert row["score_claim"] is False
    assert row["exact_cuda_auth_eval"] is False
    assert "partial_decode_coverage_waived" in row["dispatch_blockers"]
    assert "missing_exact_cuda_auth_eval" in row["dispatch_blockers"]


def test_cma_es_search_hard_caps_eval_budget_with_partial_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("numpy")
    instances: list[Any] = []

    class FakeCMA:
        population_size = 4

        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.tell_calls: list[Any] = []
            instances.append(self)

        def should_stop(self) -> bool:
            return False

        def ask(self) -> list[float]:
            return [0.0, 3.0]

        def tell(self, solutions: list[Any]) -> None:
            self.tell_calls.append(solutions)

    fake_cmaes = types.ModuleType("cmaes")
    fake_cmaes.CMA = FakeCMA
    monkeypatch.setitem(sys.modules, "cmaes", fake_cmaes)

    evaluations = cma_tool.cma_es_search(
        _IdentityCodecOp,
        {"x": torch.tensor([1.0], dtype=torch.float32)},
        _specs(cma_tool),
        max_evals=3,
        seed=99,
    )

    assert len(evaluations) == 3
    assert evaluations[0].params == {"scale": 0.0, "quality": 3}
    assert instances
    assert instances[0].tell_calls == []

    report = cma_tool._build_search_report(
        evaluations,
        op_module="fixture.module",
        op_class="IdentityCodecOp",
        optimizer_name="cma_es",
        seed=99,
        specs=_specs(cma_tool),
        requested_max_evals=3,
        state_dict={"x": torch.tensor([1.0], dtype=torch.float32)},
    )
    payload = asdict(report)
    assert payload["n_evaluations"] == 3
    assert payload["requested_max_evals"] == 3
    assert payload["max_eval_semantics"] == "hard_cap_no_overshoot"
    assert payload["baseline_eval_idx"] == 0
    assert payload["baseline_params"] == {"scale": 0.0, "quality": 3}
    assert payload["baseline_status"] == "evaluated"
    assert payload["score_claim"] is False
    assert payload["dispatchable"] is False
    assert payload["exact_cuda_auth_eval"] is False


def test_optuna_init_baseline_and_custody_fields_are_reported(
    tmp_path: Path,
) -> None:
    pytest.importorskip("optuna")
    state_dict = {"x": torch.tensor([1.0], dtype=torch.float32)}
    state_path = tmp_path / "state.pt"
    torch.save(state_dict, state_path)

    evaluations = optuna_tool.optuna_tpe_search(
        _IdentityCodecOp,
        state_dict,
        _specs(optuna_tool),
        max_evals=2,
        seed=7,
    )
    report = optuna_tool._build_search_report(
        evaluations,
        op_module="fixture.module",
        op_class="IdentityCodecOp",
        seed=7,
        specs=_specs(optuna_tool),
        requested_max_evals=2,
        state_dict_path=state_path,
        state_dict=state_dict,
    )
    payload = asdict(report)

    assert evaluations[0].params == {"scale": 0.0, "quality": 3}
    assert payload["baseline_eval_idx"] == 0
    assert payload["baseline_params"] == {"scale": 0.0, "quality": 3}
    assert payload["baseline_status"] == "evaluated"
    assert payload["state_dict_path"] == str(state_path)
    assert payload["state_dict_bytes"] == state_path.stat().st_size
    assert len(payload["state_dict_sha256"]) == 64
    assert payload["tensor_contract"] == [
        {"key": "x", "shape": [1], "dtype": "torch.float32", "numel": 1}
    ]
    assert payload["evidence_grade"] == "[CPU-prep+optuna_tpe]"
    assert payload["target_modes"] == ["contest_exact_eval_planning"]
    assert payload["score_claim"] is False
    assert payload["dispatchable"] is False
    assert payload["promotion_eligible"] is False
    assert payload["exact_cuda_auth_eval"] is False
    assert payload["archive_sha256"] is None
    assert "missing_exact_cuda_auth_eval" in payload["dispatch_blockers"]


@pytest.mark.parametrize(
    ("tool", "tool_name"),
    [(cma_tool, "cma"), (optuna_tool, "optuna")],
)
def test_emit_evidence_rows_are_explicitly_planning_only(
    tool: Any,
    tool_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = f"fixture_emit_evidence_{tool_name}"
    fixture_module = tmp_path / f"{module_name}.py"
    fixture_module.write_text(
        """
class Result:
    def __init__(self, bytes_out, op_state):
        self.blob = b"fixture"
        self.op_state = op_state
        self.bytes_out = bytes_out


class FixtureCodecOp:
    def __init__(self, scale=0.0, quality=1):
        self.scale = float(scale)
        self.quality = int(quality)

    def encode(self, state_dict, context):
        return Result(100 + 10 * self.quality, {"x": state_dict["x"]})

    def decode(self, blob, *, op_state, context):
        return {"x": op_state["x"] + self.scale}
""",
        encoding="utf-8",
    )
    state_path = tmp_path / f"{tool_name}_state.pt"
    torch.save({"x": torch.tensor([1.0], dtype=torch.float32)}, state_path)
    report_path = tmp_path / f"{tool_name}_report.json"
    evidence_path = tmp_path / f"{tool_name}_evidence.jsonl"

    if tool_name == "optuna":

        def fake_optuna_search(
            op_cls: Any,
            state_dict: dict[str, torch.Tensor],
            specs: list[Any],
            max_evals: int,
            seed: int = 0,
            **kwargs: Any,
        ) -> list[Any]:
            del op_cls, max_evals, seed, kwargs
            return [
                optuna_tool._evaluate(
                    _IdentityCodecOp,
                    optuna_tool._init_params(specs),
                    state_dict,
                    eval_idx=0,
                )
            ]

        monkeypatch.setattr(tool, "optuna_tpe_search", fake_optuna_search)

    argv = [
        "--module",
        module_name,
        "--class",
        "FixtureCodecOp",
        "--state-dict-path",
        str(state_path),
        "--param-spec",
        json.dumps(
            {
                "scale": {"type": "float", "low": 0.0, "high": 0.2, "init": 0.0},
                "quality": {"type": "int", "low": 1, "high": 5, "init": 3},
            }
        ),
        "--max-evals",
        "1",
        "--seed",
        "5",
        "--output",
        str(report_path),
        "--emit-evidence",
        str(evidence_path),
    ]
    if tool_name == "cma":
        argv.extend(["--optimizer", "random"])

    sys.path.insert(0, str(tmp_path))
    try:
        rc = tool.main(argv)
    finally:
        sys.path.remove(str(tmp_path))

    assert rc == 0
    row = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert row["score_claim"] is False
    assert row["dispatchable"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_affecting_payload_changed"] is False
    assert row["charged_bits_changed"] is False
    assert row["exact_cuda_auth_eval"] is False
    assert row["archive_sha256"] is None
    assert row["archive_bytes"] is None
    assert row["promotion_eligible"] is False
    assert row["target_modes"] == ["contest_exact_eval_planning"]
    assert row["evidence_semantics"] == "cpu_codec_op_search_forensic"
    assert "missing_exact_cuda_auth_eval" in row["dispatch_blockers"]
    assert "codec_op_bytes_out_not_archive_bytes" in row["evidence_limitations"]
