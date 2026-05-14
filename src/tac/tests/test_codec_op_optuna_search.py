# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools" / "codec_op_optuna_search.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "codec_op_optuna_search_under_test", TOOL_PATH
    )
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

    def encode(
        self, state_dict: dict[str, torch.Tensor], context: dict[str, Any]
    ) -> _Result:
        del context
        bytes_out = 100 + 10 * self.quality + round(abs(self.scale) * 1_000)
        return _Result(bytes_out=bytes_out, op_state={"x": state_dict["x"]})

    def decode(
        self, blob: bytes, *, op_state: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        del blob, context
        return {"x": op_state["x"] + self.scale}


class _FailingCodecOp:
    def __init__(self, **params: Any) -> None:
        del params

    def encode(
        self, state_dict: dict[str, torch.Tensor], context: dict[str, Any]
    ) -> _Result:
        del state_dict, context
        raise RuntimeError("fixture encode failure")

    def decode(
        self, blob: bytes, *, op_state: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        del blob, op_state, context
        return {}


class _DroppingCodecOp:
    def __init__(self, **params: Any) -> None:
        del params

    def encode(
        self, state_dict: dict[str, torch.Tensor], context: dict[str, Any]
    ) -> _Result:
        del context
        return _Result(bytes_out=123, op_state={"x": state_dict["x"]})

    def decode(
        self, blob: bytes, *, op_state: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        del blob, context
        return {"x": op_state["x"]}


def _state_dict() -> dict[str, torch.Tensor]:
    return {"x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}


def _projection(
    evaluations: list[Any],
) -> list[tuple[dict[str, Any], int, float | None, float | None, str | None]]:
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


def test_optuna_tpe_search_is_deterministic_for_fixed_seed() -> None:
    pytest.importorskip("optuna")
    specs = [
        tool.ParamSpec("scale", "float", 0.0, 0.2, 0.0),
        tool.ParamSpec("quality", "int", 1, 5, 3),
    ]

    first = tool.optuna_tpe_search(
        _ToyCodecOp, _state_dict(), specs, max_evals=5, seed=17
    )
    second = tool.optuna_tpe_search(
        _ToyCodecOp, _state_dict(), specs, max_evals=5, seed=17
    )

    assert _projection(first) == _projection(second)
    assert first[0].params == {"scale": 0.0, "quality": 3}
    assert len(first) == 5
    assert all(ev.error is None for ev in first)


def test_parameter_spec_validation() -> None:
    parsed = tool._parse_param_spec(
        json.dumps(
            {
                "alpha": {
                    "type": "float",
                    "low": 0.1,
                    "high": 1.0,
                    "init": 0.3,
                    "log": True,
                }
            }
        )
    )
    assert parsed == [tool.ParamSpec("alpha", "float", 0.1, 1.0, 0.3, True)]

    with pytest.raises(SystemExit, match="unsupported type"):
        tool._parse_param_spec('{"bad": {"type": "str", "low": 0, "high": 1}}')
    with pytest.raises(SystemExit, match="high must be >= low"):
        tool._parse_param_spec('{"bad": {"low": 2, "high": 1}}')
    with pytest.raises(SystemExit, match="int bounds must be integers"):
        tool._parse_param_spec(
            '{"bad": {"type": "int", "low": 1.5, "high": 3}}'
        )
    with pytest.raises(SystemExit, match="log range must be > 0"):
        tool._parse_param_spec('{"bad": {"low": 0, "high": 1, "log": true}}')
    with pytest.raises(SystemExit, match="must contain at least one parameter"):
        tool._parse_param_spec("{}")


def test_load_state_dict_requires_tensors_or_explicit_tensor_key(tmp_path: Path) -> None:
    dict_path = tmp_path / "dict.pt"
    torch.save(_state_dict(), dict_path)
    assert list(tool._load_state_dict(dict_path, None)) == ["x"]

    tensor_path = tmp_path / "tensor.pt"
    torch.save(torch.tensor([1.0]), tensor_path)
    assert list(tool._load_state_dict(tensor_path, "single")) == ["single"]

    bad_path = tmp_path / "bad.pt"
    torch.save({"x": 1}, bad_path)
    with pytest.raises(SystemExit, match="is not a Tensor"):
        tool._load_state_dict(bad_path, None)


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


def test_append_atom_ledger_rows_uses_planning_only_metadata(tmp_path: Path) -> None:
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
            materialized_payload_path="payloads/eval_00000.section",
            materialized_payload_bytes=100,
            materialized_payload_sha256="a" * 64,
            materialized_payload_contract="pr106_decoder_packed_brotli",
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
    assert first["schema"] == "codec_op_optuna_search_eval_v1"
    assert first["tool"] == "tools/codec_op_optuna_search.py"
    assert first["target_modes"] == ["contest_exact_eval_planning"]
    assert first["ready_for_exact_eval_dispatch"] is False
    assert first["field_selection_ready_for_exact_eval_dispatch"] is False
    assert first["dispatchable"] is False
    assert first["score_claim"] is False
    assert first["score_affecting_payload_changed"] is False
    assert first["charged_bits_changed"] is False
    assert "missing_exact_cuda_auth_eval" in first["dispatch_blockers"]
    assert first["byte_delta"] == 0
    assert first["expected_tensor_count"] == 1
    assert first["matched_tensor_count"] == 1
    assert first["materialized_payload_path"] == "payloads/eval_00000.section"
    assert first["materialized_payload_bytes"] == 100
    assert first["materialized_payload_sha256"] == "a" * 64
    assert first["materialized_payload_contract"] == "pr106_decoder_packed_brotli"
    assert first["pareto_frontier"] is True
    assert dominated["pareto_frontier"] is False
    assert dominated["pareto_dominated_by"] == [0]
    assert failed["planning_objectives"]["fitness"] is None
    assert "evaluation_failed" in failed["dispatch_blockers"]


def test_main_writes_strict_report_schema_and_ledger(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    fixture_module = tmp_path / "fixture_codec_op.py"
    fixture_module.write_text(
        """
class Result:
    def __init__(self, bytes_out, op_state):
        repeats = (bytes_out // len(b"fixture")) + 1
        self.blob = (b"fixture" * repeats)[:bytes_out]
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
    payload_dir = tmp_path / "payloads"
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
                        "scale": {
                            "type": "float",
                            "low": 0.0,
                            "high": 0.2,
                            "init": 0.0,
                        },
                        "quality": {"type": "int", "low": 1, "high": 5, "init": 3},
                    }
                ),
                "--max-evals",
                "3",
                "--seed",
                "9",
                "--output",
                str(report_path),
                "--atom-ledger-output",
                str(ledger_path),
                "--substrate-label",
                "fixture_search",
                "--materialized-payload-output-dir",
                str(payload_dir),
                "--materialized-payload-contract",
                "pr106_decoder_packed_brotli",
            ]
        )
    finally:
        sys.path.remove(str(tmp_path))

    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "codec_op_optuna_search_report_v1"
    assert payload["tool"] == "tools/codec_op_optuna_search.py"
    assert payload["optimizer"] == "optuna_tpe"
    assert payload["n_evaluations"] == 3
    assert payload["n_successful"] == 3
    assert payload["n_failed"] == 0
    assert payload["pareto_frontier_count"] >= 1
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["score_claim"] is False
    assert payload["target_modes"] == ["contest_exact_eval_planning"]
    assert payload["score_affecting_payload_changed"] is False
    assert payload["charged_bits_changed"] is False
    assert payload["best_eval"]["pareto_frontier"] is True
    materialized_path = Path(payload["best_eval"]["materialized_payload_path"])
    materialized_payload = materialized_path.read_bytes()
    assert materialized_path.is_file()
    assert payload["best_eval"]["materialized_payload_bytes"] == len(
        materialized_payload
    )
    assert payload["best_eval"]["materialized_payload_sha256"] == hashlib.sha256(
        materialized_payload
    ).hexdigest()
    assert (
        payload["best_eval"]["materialized_payload_contract"]
        == "pr106_decoder_packed_brotli"
    )
    assert payload["optuna_version"] is not None
    assert payload["parameter_space"][0]["name"] == "scale"
    ledger_rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(ledger_rows) == 3
    assert all(row["materialized_payload_path"] for row in ledger_rows)
    assert all(
        row["materialized_payload_contract"] == "pr106_decoder_packed_brotli"
        for row in ledger_rows
    )
