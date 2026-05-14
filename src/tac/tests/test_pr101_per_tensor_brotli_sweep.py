# SPDX-License-Identifier: MIT
"""Tests for tools/pr101_per_tensor_brotli_sweep.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import brotli
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "pr101_per_tensor_brotli_sweep.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pr101_per_tensor_brotli_sweep", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pr101_per_tensor_brotli_sweep"] = module
    spec.loader.exec_module(module)
    return module


def test_sweep_tensor_records_best_and_failures() -> None:
    tool = _load_tool()
    raw = bytes([0, 1, 1, 2, 3, 5, 8, 13])

    result = tool._sweep_tensor(
        raw,
        qualities=[4, 5],
        lgwins=[10],
        lgblocks=[16],
    )

    expected = min(
        len(brotli.compress(raw, quality=q, lgwin=10, lgblock=16))
        for q in (4, 5)
    )
    assert result["bytes_out"] == expected
    assert result["n_evals"] == 2
    assert result["n_failed"] == 0
    assert result["raw_bytes_len"] == len(raw)


def test_per_tensor_sweep_tiny_state_dict(monkeypatch, tmp_path: Path) -> None:
    tool = _load_tool()
    monkeypatch.setattr(tool, "FIXED_STATE_SCHEMA", (("w", (4,)),))
    state_path = tmp_path / "state.pt"
    torch.save({"w": torch.tensor([0.0, 1.0, -1.0, 0.5])}, state_path)

    manifest = tool.per_tensor_sweep(
        state_path,
        qualities=[4],
        lgwins=[10],
        lgblocks=[16],
    )

    assert manifest["schema"] == tool.SCHEMA_VERSION
    assert manifest["n_tensors"] == 1
    assert manifest["n_total_evals"] == 1
    assert manifest["total_raw_bytes"] == 4
    assert manifest["total_per_tensor_optimum_bytes"] > 0
    assert manifest["per_tensor_results"][0]["name"] == "w"
    assert manifest["per_tensor_results"][0]["raw_sha256"]
