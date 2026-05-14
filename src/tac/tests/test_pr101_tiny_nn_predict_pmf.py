# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]


def _load_tool() -> Any:
    path = REPO / "tools" / "pr101_tiny_nn_predict_pmf.py"
    spec = importlib.util.spec_from_file_location("pr101_tiny_nn_predict_pmf_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_state_dict(path: Path) -> Path:
    state_dict = {
        "a": torch.tensor([0.0, 1.0, 1.0, -1.0, 0.0, 1.0], dtype=torch.float32),
        "b": torch.tensor([2.0, 2.0, 0.0, 0.0, -2.0, -2.0], dtype=torch.float32),
    }
    torch.save(state_dict, path)
    return path


def _install_tiny_schema(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    monkeypatch.setattr(module, "FIXED_STATE_SCHEMA", (("a", (6,)), ("b", (6,))))

    def fake_quantize(name: str, tensor: torch.Tensor, *, n_quant: int) -> Any:
        del name, n_quant
        return SimpleNamespace(
            q_i8=tensor.detach().cpu().numpy().astype(np.int8),
            shape=tuple(int(v) for v in tensor.shape),
            scale=1.0,
        )

    monkeypatch.setattr(module, "_quantize_tensor", fake_quantize)


def _run_tiny_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")
    output = tmp_path / "tiny_nn.json"

    assert tool.main([
        "--state-dict-path",
        str(state_dict_path),
        "--output",
        str(output),
        "--variants",
        "tensor_only,tensor_prev_symbol",
        "--rank",
        "3",
        "--epochs",
        "2",
        "--batch-size",
        "4",
        "--learning-rate",
        "0.03",
        "--seed",
        "123",
        "--torch-threads",
        "1",
    ]) == 0
    return tool, json.loads(output.read_text())


def test_tiny_nn_manifest_is_planning_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tool, manifest = _run_tiny_probe(tmp_path, monkeypatch)

    assert manifest["score_claim"] is False
    assert manifest["score_affecting_payload_changed"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "no_actual_range_or_ans_bitstream" in manifest["dispatch_blockers"]
    assert "no_runtime_model_serializer_or_decoder" in manifest["dispatch_blockers"]
    assert manifest["input_state_dict_sha256"]
    assert manifest["comparison_brotli_optuna_archive_bytes"] == 178_144
    assert manifest["comparison_iid_per_tensor_floor_archive_bytes"] == 175_916
    assert manifest["comparison_per_tensor_aac_archive_bytes"] == 178_181
    assert {row["variant"] for row in manifest["variants"]} == {
        "tensor_only",
        "tensor_prev_symbol",
    }


def test_tiny_nn_probe_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    manifests = []
    for output in outputs:
        assert tool.main([
            "--state-dict-path",
            str(state_dict_path),
            "--output",
            str(output),
            "--variants",
            "tensor_only,tensor_prev_symbol",
            "--rank",
            "3",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.03",
            "--seed",
            "123",
            "--torch-threads",
            "1",
        ]) == 0
        manifests.append(json.loads(output.read_text()))
    first, second = manifests
    assert first == second


def test_tiny_nn_manifest_charges_model_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tool, manifest = _run_tiny_probe(tmp_path, monkeypatch)
    for row in manifest["variants"]:
        estimates = row["model_parameter_byte_estimate"]
        assert estimates["parameter_count"] > 0
        assert estimates["raw_fp16_bytes"] > 0
        assert estimates["brotli_int8_symmetric_bytes"] > 0
        assert row["primary_model_parameter_bytes"] == estimates["brotli_int8_symmetric_bytes"]
        assert row["estimated_archive_bytes_reference_accounting"] >= (
            row["int8_dequantized_model_payload_bytes"]
            + row["primary_model_parameter_bytes"]
        )


def test_tiny_nn_probe_rejects_missing_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = tmp_path / "state.pt"
    torch.save({"a": torch.ones(6)}, state_dict_path)

    with pytest.raises(SystemExit, match="missing tensor"):
        tool.build_tiny_nn_pmf_report(
            state_dict_path,
            config=tool.TinyNnConfig(epochs=0),
            variants=["tensor_only"],
        )
