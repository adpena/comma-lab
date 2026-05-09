from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


def _load_tool() -> Any:
    path = REPO / "tools" / "pr101_charm_real_substrate_probe.py"
    spec = importlib.util.spec_from_file_location("pr101_charm_probe_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_state_dict(path: Path) -> Path:
    state_dict = {
        "a": torch.tensor([0.0, 1.0, 1.0, -1.0], dtype=torch.float32),
        "b": torch.tensor([2.0, 2.0, 0.0, 0.0], dtype=torch.float32),
    }
    torch.save(state_dict, path)
    return path


def _install_tiny_schema(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    monkeypatch.setattr(module, "FIXED_STATE_SCHEMA", (("a", (4,)), ("b", (4,))))

    def fake_quantize(_name: str, tensor: torch.Tensor, *, n_quant: int) -> Any:
        del n_quant
        return SimpleNamespace(q_i8=tensor.numpy().astype("int8"), scale=1.0)

    monkeypatch.setattr(module, "_quantize_tensor", fake_quantize)


def test_charm_real_substrate_probe_is_fail_closed_and_roundtrips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")
    output = tmp_path / "charm_probe.json"

    assert tool.main([
        "--state-dict-path",
        str(state_dict_path),
        "--output",
        str(output),
        "--chunk-symbols",
        "3",
    ]) == 0

    manifest = json.loads(output.read_text())
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "range_coder_not_wired_into_inflate_runtime" in manifest[
        "dispatch_blockers"
    ]
    assert "reactivation_required_before_new_dispatch" in manifest[
        "dispatch_blockers"
    ]
    assert manifest["reactivation_criteria"]
    assert manifest["family_falsified"] is False
    assert manifest["method_family_retired"] is False
    assert manifest["n_symbols_total"] == 8
    assert {row["model"] for row in manifest["models"]} == {
        "tensor_gaussian",
        "previous_symbol_gaussian",
        "delta_zero_gaussian",
    }
    assert all(row["roundtrip_exact"] is True for row in manifest["models"])
    assert all(row["archive_estimate_bytes"] > 0 for row in manifest["models"])
