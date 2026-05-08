"""Focused tests for PR101 A1 CPU-only anchor tools."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool(name: str):
    path = REPO_ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_lossy_int4_manifest_is_fail_closed(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool("pr101_lossy_int4_block_sweep.py")
    monkeypatch.setattr(tool, "FIXED_STATE_SCHEMA", (("a", (4,)), ("b", (3,))))

    state_dict = {
        "a": torch.tensor([0.0, 1.0, -1.0, 2.0], dtype=torch.float32),
        "b": torch.tensor([3.0, -3.0, 0.5], dtype=torch.float32),
    }
    state_path = tmp_path / "state.pt"
    torch.save(state_dict, state_path)

    manifest = tool.sweep_block_sizes(state_path, [2, 4])

    assert manifest["schema"] == "pr101_lossy_int4_block_sweep.v2"
    assert manifest["evidence_semantics"].endswith("_no_decode_no_score")
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "missing_exact_cuda_auth_eval" in manifest["dispatch_blockers"]
    assert manifest["input_state_dict_sha256"]
    assert manifest["best_archive_bytes"] == min(r["archive_bytes"] for r in manifest["rows"])


def test_kalle_fold_manifest_is_fail_closed(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool("pr101_kalle_fold_mixture_codec.py")
    monkeypatch.setattr(tool, "FIXED_STATE_SCHEMA", (("a", (8,)),))

    q_i8 = np.array([0, 0, 0, 1, -1, 2, -2, 3], dtype=np.int8)

    def fake_quantize(_name, _tensor, *, n_quant):
        assert n_quant == tool.N_QUANT
        return SimpleNamespace(q_i8=q_i8, scale=0.125)

    monkeypatch.setattr(tool, "_quantize_tensor", fake_quantize)
    state_path = tmp_path / "state.pt"
    torch.save({"a": torch.zeros(8, dtype=torch.float32)}, state_path)

    manifest = tool.run_codec(state_path)

    assert manifest["schema"] == "pr101_kalle_fold_mixture_codec.v2"
    assert manifest["evidence_semantics"].endswith("_no_decoder_no_score")
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "mixture_decoder_not_wired_into_runtime_packet" in manifest["dispatch_blockers"]
    assert manifest["input_state_dict_sha256"]
    assert manifest["n_tensors"] == 1
    assert manifest["archive_bytes"] >= manifest["archive_overhead_bytes"]


def test_kalle_fold_evidence_row_keeps_autopilot_contract(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool("pr101_kalle_fold_mixture_codec.py")
    monkeypatch.setattr(tool, "FIXED_STATE_SCHEMA", (("a", (4,)),))
    q_i8 = np.array([0, 0, 1, -1], dtype=np.int8)
    monkeypatch.setattr(
        tool,
        "_quantize_tensor",
        lambda _name, _tensor, *, n_quant: SimpleNamespace(q_i8=q_i8, scale=1.0),
    )

    state_path = tmp_path / "state.pt"
    torch.save({"a": torch.zeros(4, dtype=torch.float32)}, state_path)
    output = tmp_path / "manifest.json"
    evidence = tmp_path / "evidence.jsonl"

    assert tool.main([
        "--state-dict", str(state_path),
        "--output-json", str(output),
        "--output-evidence", str(evidence),
    ]) == 0
    row = json.loads(evidence.read_text(encoding="utf-8"))
    assert row["technique"] == "kalle_fold_mixture_canonical_shapes"
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
