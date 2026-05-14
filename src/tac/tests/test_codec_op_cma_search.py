# SPDX-License-Identifier: MIT
from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools" / "codec_op_cma_search.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("codec_op_cma_search_under_test", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


class _Result:
    def __init__(self, *, bytes_out: int, op_state: dict[str, Any]) -> None:
        self.blob = b"fixture"
        self.op_state = op_state
        self.bytes_out = bytes_out


class _ToyCodecOp:
    def __init__(self, *, scale: float = 0.0, quality: int = 1) -> None:
        self.scale = float(scale)
        self.quality = int(quality)

    def encode(self, state_dict: dict[str, torch.Tensor], context: dict[str, Any]) -> _Result:
        del context
        bytes_out = 100 + 10 * self.quality + round(abs(self.scale) * 1_000)
        return _Result(bytes_out=bytes_out, op_state={"x": state_dict["x"]})

    def decode(self, blob: bytes, *, op_state: dict[str, Any], context: dict[str, Any]) -> dict[str, torch.Tensor]:
        del blob, context
        return {"x": op_state["x"] + self.scale}


class _FailingCodecOp:
    def __init__(self, **params: Any) -> None:
        del params

    def encode(self, state_dict: dict[str, torch.Tensor], context: dict[str, Any]) -> _Result:
        del state_dict, context
        raise RuntimeError("fixture encode failure")

    def decode(self, blob: bytes, *, op_state: dict[str, Any], context: dict[str, Any]) -> dict[str, torch.Tensor]:
        del blob, op_state, context
        return {}


class _DroppingCodecOp:
    def __init__(self, **params: Any) -> None:
        del params

    def encode(self, state_dict: dict[str, torch.Tensor], context: dict[str, Any]) -> _Result:
        del context
        return _Result(bytes_out=123, op_state={"x": state_dict["x"]})

    def decode(self, blob: bytes, *, op_state: dict[str, Any], context: dict[str, Any]) -> dict[str, torch.Tensor]:
        del blob, context
        return {"x": op_state["x"]}


def _state_dict() -> dict[str, torch.Tensor]:
    return {"x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}


def _projection(evaluations: list[Any]) -> list[tuple[dict[str, Any], int, float | None, float | None, str | None]]:
    return [
        (
            ev.params,
            ev.bytes_out,
            ev.reconstruction_rms,
            ev.fitness,
            ev.error,
        )
        for ev in evaluations
    ]


def test_random_search_is_deterministic_and_starts_with_init() -> None:
    specs = [
        tool.ParamSpec("scale", "float", 0.0, 0.2, 0.0),
        tool.ParamSpec("quality", "int", 1, 5, 3),
    ]

    first = tool.random_search(_ToyCodecOp, _state_dict(), specs, max_evals=5, seed=17)
    second = tool.random_search(_ToyCodecOp, _state_dict(), specs, max_evals=5, seed=17)

    assert _projection(first) == _projection(second)
    assert first[0].params == {"scale": 0.0, "quality": 3}
    assert len(first) == 5


def test_cma_es_missing_dependency_uses_same_deterministic_random_path(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = [
        tool.ParamSpec("scale", "float", 0.0, 0.2, 0.0),
        tool.ParamSpec("quality", "int", 1, 5, 3),
    ]
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cmaes":
            raise ImportError("forced missing cmaes")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    fallback = tool.cma_es_search(_ToyCodecOp, _state_dict(), specs, max_evals=4, seed=23)
    direct = tool.random_search(_ToyCodecOp, _state_dict(), specs, max_evals=4, seed=23)

    assert _projection(fallback) == _projection(direct)


def test_parameter_spec_validation_and_coercion() -> None:
    int_spec = tool.ParamSpec("n", "int", 1, 3, 2)
    float_spec = tool.ParamSpec("alpha", "float", -1.0, 1.0, 0.0)

    assert tool._coerce(0.1, int_spec) == 1
    assert tool._coerce(2.6, int_spec) == 3
    assert tool._coerce(7.0, int_spec) == 3
    assert tool._coerce(-2.0, float_spec) == -1.0
    assert tool._coerce(0.25, float_spec) == pytest.approx(0.25)

    parsed = tool._parse_param_spec(
        json.dumps({"alpha": {"type": "float", "low": 0.1, "high": 1.0, "log": True}})
    )
    assert parsed == [tool.ParamSpec("alpha", "float", 0.1, 1.0, 0.55, True)]

    with pytest.raises(SystemExit, match="unsupported type"):
        tool._parse_param_spec('{"bad": {"type": "str", "low": 0, "high": 1}}')
    with pytest.raises(SystemExit, match="high must be >= low"):
        tool._parse_param_spec('{"bad": {"low": 2, "high": 1}}')
    with pytest.raises(SystemExit, match="log range must be > 0"):
        tool._parse_param_spec('{"bad": {"low": 0, "high": 1, "log": true}}')
    with pytest.raises(ValueError, match="unsupported parameter type"):
        tool._coerce(1.0, tool.ParamSpec("bad", "str", 0, 1, 0))


def test_failed_evaluation_is_json_safe_and_not_pareto_ranked() -> None:
    ev = tool._evaluate(_FailingCodecOp, {"scale": 0.0}, _state_dict(), eval_idx=0)
    tool.annotate_pareto_frontier([ev])

    assert ev.bytes_out == -1
    assert ev.reconstruction_rms is None
    assert ev.fitness is None
    assert ev.error == "RuntimeError: fixture encode failure"
    assert ev.pareto_frontier is False
    json.dumps(asdict(ev), allow_nan=False)


def test_partial_decode_coverage_is_failed_not_ranked() -> None:
    state_dict = {
        "x": torch.tensor([1.0]),
        "y": torch.tensor([2.0]),
    }

    ev = tool._evaluate(_DroppingCodecOp, {}, state_dict, eval_idx=0)
    tool.annotate_pareto_frontier([ev])

    assert ev.bytes_out == 123
    assert ev.error == "ValueError: CodecOp decode did not reconstruct every input tensor key"
    assert ev.expected_tensor_count == 2
    assert ev.matched_tensor_count == 1
    assert ev.missing_tensor_keys == ["y"]
    assert ev.fitness is None
    assert ev.pareto_frontier is False


def test_materialization_waits_for_decode_coverage(tmp_path: Path) -> None:
    state_dict = {
        "x": torch.tensor([1.0]),
        "y": torch.tensor([2.0]),
    }

    ev = tool._evaluate(
        _DroppingCodecOp,
        {},
        state_dict,
        eval_idx=0,
        materialized_payload_output_dir=tmp_path / "payloads",
    )

    assert ev.error == "ValueError: CodecOp decode did not reconstruct every input tensor key"
    assert ev.materialized_payload_path is None
    assert not (tmp_path / "payloads").exists()


def test_append_atom_ledger_rows_uses_planning_only_target_metadata(tmp_path: Path) -> None:
    evaluations = [
        tool.Evaluation(
            eval_idx=0,
            params={"scale": 0.0, "quality": 1},
            bytes_out=100,
            reconstruction_rms=0.0,
            fitness=100.0,
            timestamp_utc="2026-05-07T00:00:00Z",
            expected_tensor_count=1,
            matched_tensor_count=1,
        ),
        tool.Evaluation(
            eval_idx=1,
            params={"scale": 0.1, "quality": 1},
            bytes_out=100,
            reconstruction_rms=0.1,
            fitness=100_100.0,
            timestamp_utc="2026-05-07T00:00:01Z",
            expected_tensor_count=1,
            matched_tensor_count=1,
        ),
        tool.Evaluation(
            eval_idx=2,
            params={"scale": 0.2, "quality": 1},
            bytes_out=-1,
            reconstruction_rms=None,
            fitness=None,
            timestamp_utc="2026-05-07T00:00:02Z",
            error="RuntimeError: fixture",
        ),
    ]
    ledger = tmp_path / "ledger.jsonl"

    tool.append_atom_ledger_rows(
        ledger,
        evaluations,
        op_module="fixture.module",
        op_class="FixtureCodecOp",
        substrate_label="fixture_search",
    )

    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    first, dominated, failed = rows
    assert first["schema"] == "codec_op_cma_search_eval_v1"
    assert first["tool"] == "tools/codec_op_cma_search.py"
    assert first["target_modes"] == ["contest_exact_eval_planning"]
    assert "contest_exact_eval" not in first["target_modes"]
    assert first["deployment_target"] == "desktop_research"
    assert first["ready_for_exact_eval_dispatch"] is False
    assert first["score_claim"] is False
    assert first["score_affecting_payload_changed"] is False
    assert first["charged_bits_changed"] is False
    assert first["evidence_semantics"] == "cpu_codec_op_search_forensic"
    assert "missing_exact_cuda_auth_eval" in first["dispatch_blockers"]
    assert first["byte_delta"] == 0
    assert first["expected_tensor_count"] == 1
    assert first["matched_tensor_count"] == 1
    assert first["pareto_frontier"] is True
    assert dominated["pareto_frontier"] is False
    assert dominated["pareto_dominated_by"] == [0]
    assert failed["planning_objectives"]["fitness"] is None
    assert "evaluation_failed" in failed["dispatch_blockers"]


def test_main_writes_strict_report_schema_and_ledger(tmp_path: Path) -> None:
    fixture_module = tmp_path / "fixture_codec_op.py"
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
        return Result(
            100 + 10 * self.quality + int(round(abs(self.scale) * 1000)),
            {"x": state_dict["x"]},
        )

    def decode(self, blob, *, op_state, context):
        return {"x": op_state["x"] + self.scale}
""",
        encoding="utf-8",
    )
    state_path = tmp_path / "state.pt"
    torch.save(_state_dict(), state_path)
    report_path = tmp_path / "report.json"
    ledger_path = tmp_path / "ledger.jsonl"
    sys.path.insert(0, str(tmp_path))
    try:
        rc = tool.main(
            [
                "--module",
                "fixture_codec_op",
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
                "3",
                "--seed",
                "9",
                "--optimizer",
                "random",
                "--output",
                str(report_path),
                "--atom-ledger-output",
                str(ledger_path),
            ]
        )
    finally:
        sys.path.remove(str(tmp_path))

    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "codec_op_cma_search_report_v1"
    assert payload["tool"] == "tools/codec_op_cma_search.py"
    assert payload["optimizer"] == "random_search"
    assert payload["n_evaluations"] == 3
    assert payload["n_successful"] == 3
    assert payload["n_failed"] == 0
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["score_claim"] is False
    assert payload["target_modes"] == ["contest_exact_eval_planning"]
    assert payload["score_affecting_payload_changed"] is False
    assert payload["charged_bits_changed"] is False
    assert payload["best_eval"]["pareto_frontier"] is True
    assert payload["parameter_space"][0]["name"] == "scale"
    assert len(ledger_path.read_text(encoding="utf-8").splitlines()) == 3


def test_main_materializes_successful_payloads_for_bridge(tmp_path: Path) -> None:
    fixture_module = tmp_path / "fixture_materialized_codec_op.py"
    fixture_module.write_text(
        """
class Result:
    def __init__(self, blob, op_state):
        self.blob = blob
        self.op_state = op_state
        self.bytes_out = len(blob)


class FixtureMaterializedCodecOp:
    def __init__(self, quality=1):
        self.quality = int(quality)

    def encode(self, state_dict, context):
        return Result(b"payload-" + bytes([self.quality]), {"x": state_dict["x"]})

    def decode(self, blob, *, op_state, context):
        return {"x": op_state["x"]}
""",
        encoding="utf-8",
    )
    state_path = tmp_path / "state.pt"
    report_path = tmp_path / "report.json"
    ledger_path = tmp_path / "ledger.jsonl"
    payload_dir = tmp_path / "payloads"
    torch.save(_state_dict(), state_path)
    sys.path.insert(0, str(tmp_path))
    try:
        rc = tool.main(
            [
                "--module",
                "fixture_materialized_codec_op",
                "--class",
                "FixtureMaterializedCodecOp",
                "--state-dict-path",
                str(state_path),
                "--param-spec",
                json.dumps({"quality": {"type": "int", "low": 1, "high": 3, "init": 2}}),
                "--max-evals",
                "1",
                "--seed",
                "5",
                "--optimizer",
                "random",
                "--output",
                str(report_path),
                "--atom-ledger-output",
                str(ledger_path),
                "--materialized-payload-output-dir",
                str(payload_dir),
                "--materialized-payload-contract",
                "pr106_decoder_packed_brotli",
            ]
        )
    finally:
        sys.path.remove(str(tmp_path))

    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    best = report["best_eval"]
    materialized_path = Path(best["materialized_payload_path"])
    assert materialized_path.is_file()
    assert materialized_path.read_bytes() == b"payload-\x02"
    assert best["materialized_payload_bytes"] == len(b"payload-\x02")
    assert best["materialized_payload_sha256"]
    assert best["materialized_payload_contract"] == "pr106_decoder_packed_brotli"

    row = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["materialized_payload_path"] == best["materialized_payload_path"]
    assert row["materialized_payload_sha256"] == best["materialized_payload_sha256"]
    assert row["ready_for_exact_eval_dispatch"] is False
