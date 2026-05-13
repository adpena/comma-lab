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


def _assert_proxy_contract(payload: dict) -> None:
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["proxy_row"] is True
    assert "missing_exact_cuda_auth_eval" in payload["dispatch_blockers"]
    assert "requires_exact_cuda_auth_eval_before_any_score_use" in payload["dispatch_blockers"]


def _assert_no_family_falsification(payload: dict) -> None:
    assert payload["family_falsified"] is False
    assert payload["falsification_scope"] in {
        "measured_configuration_only",
        "none_proxy_anchor_only",
    }


def test_lossy_int4_roundtrip_proxy_never_dispatches_or_falsifies_family(
    tmp_path: Path, monkeypatch
) -> None:
    tool = _load_tool("pr101_lossy_int4_roundtrip_test.py")
    monkeypatch.setattr(tool, "FIXED_STATE_SCHEMA", (("a", (4,)),))
    state_path = tmp_path / "state.pt"
    torch.save({"a": torch.tensor([1.0, 2.0, -1.0, -2.0])}, state_path)

    manifest = tool.measure_full_roundtrip(state_path, block_size=2)

    _assert_proxy_contract(manifest)
    _assert_no_family_falsification(manifest)
    assert manifest["evidence_semantics"].endswith("_no_score")


def test_lossy_int4_per_channel_proxy_scope_is_measured_config_only(
    tmp_path: Path, monkeypatch
) -> None:
    tool = _load_tool("pr101_lossy_int4_per_channel_scales.py")
    monkeypatch.setattr(tool, "FIXED_STATE_SCHEMA", (("a", (2, 2)),))
    state_path = tmp_path / "state.pt"
    torch.save({"a": torch.tensor([[1.0, 2.0], [-1.0, -2.0]])}, state_path)

    manifest = tool.measure_full(state_path)

    _assert_proxy_contract(manifest)
    _assert_no_family_falsification(manifest)
    assert manifest["verdict"] != "STILL-FALSIFIED"


def test_lossy_int4_mixed_precision_dominated_point_stays_proxy_closed(
    tmp_path: Path, monkeypatch
) -> None:
    tool = _load_tool("pr101_lossy_mixed_precision_int4_int8.py")
    monkeypatch.setattr(tool, "FIXED_STATE_SCHEMA", (("a", (4,)),))
    monkeypatch.setattr(tool, "PR101_BROTLI_BASELINE_BYTES", 1)
    state_path = tmp_path / "state.pt"
    torch.save({"a": torch.tensor([1.0, 2.0, -1.0, -2.0])}, state_path)

    manifest = tool.measure_full(state_path, target_rel_err_pct=5.0)

    _assert_proxy_contract(manifest)
    _assert_no_family_falsification(manifest)
    assert manifest["verdict"] == "MEASURED_CONFIG_DOMINATED_BY_PR101_BROTLI_BASELINE"
    assert "archive_bytes_not_below_pr101_brotli_baseline" in manifest["dispatch_blockers"]
    assert manifest["raw_payload_bytes_estimate_matches_packed"] is True


def test_lossy_coarsening_proxy_anchor_does_not_kill_neural_codecs(
    tmp_path: Path, monkeypatch
) -> None:
    tool = _load_tool("pr101_lossy_coarsening_analytical.py")
    monkeypatch.setattr(tool, "FIXED_STATE_SCHEMA", (("a", (8,)),))
    monkeypatch.setattr(
        tool,
        "_quantize_tensor",
        lambda _name, _tensor, *, n_quant: SimpleNamespace(
            q_i8=np.array([0, 0, 1, -1, 2, -2, 3, -3], dtype=np.int8),
        ),
    )
    state_path = tmp_path / "state.pt"
    torch.save({"a": torch.zeros(8)}, state_path)
    output_dir = tmp_path / "coarsening"
    evidence = tmp_path / "evidence.jsonl"

    assert tool.main([
        "--state-dict", str(state_path),
        "--budgets", "0.05",
        "--uniform-Ks", "1",
        "--output-dir", str(output_dir),
        "--evidence-jsonl", str(evidence),
    ]) == 0

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    _assert_proxy_contract(manifest)
    _assert_no_family_falsification(manifest)
    row = json.loads(evidence.read_text(encoding="utf-8"))
    _assert_proxy_contract(row)
    _assert_no_family_falsification(row)
    assert "not a neural-codec family kill" in row["source"]


def test_arch_shrink_post_hoc_manifest_rows_and_evidence_are_proxy_closed(
    tmp_path: Path, monkeypatch
) -> None:
    tool = _load_tool("pr101_arch_shrink_post_hoc_sweep.py")
    monkeypatch.setattr(tool, "FIXED_STATE_SCHEMA", (("a", (4, 2)),))

    state_path = tmp_path / "state.pt"
    torch.save(
        {"a": torch.arange(8, dtype=torch.float32).reshape(4, 2)},
        state_path,
    )
    manifest = tool.sweep_arch_shrink(state_path, [0.4, 1.0])

    _assert_proxy_contract(manifest)
    assert manifest["score_affecting_payload_changed"] is True
    assert all(row["proxy_row"] is True for row in manifest["rows"])
    assert all(row["rank_or_kill_eligible"] is False for row in manifest["rows"])

    output = tmp_path / "manifest.json"
    evidence = tmp_path / "evidence.jsonl"
    assert tool.main([
        "--state-dict", str(state_path),
        "--ratios", "0.4", "1.0",
        "--output-json", str(output),
        "--output-evidence", str(evidence),
    ]) == 0
    row = json.loads(evidence.read_text(encoding="utf-8"))
    _assert_proxy_contract(row)


def test_kalle_fold_8comp_manifest_and_evidence_are_proxy_closed(
    tmp_path: Path, monkeypatch
) -> None:
    tool = _load_tool("pr101_kalle_fold_8comp_hierarchical_codec.py")
    monkeypatch.setattr(tool, "FIXED_STATE_SCHEMA", (("a", (8,)),))
    q_i8 = np.array([0, 0, 0, 1, -1, 2, -2, 3], dtype=np.int8)
    monkeypatch.setattr(
        tool,
        "_quantize_tensor",
        lambda _name, _tensor, *, n_quant: SimpleNamespace(q_i8=q_i8, scale=1.0),
    )
    monkeypatch.setattr(tool, "fit_mixture_8", lambda _pmf: (np.zeros(14), 0.0))
    monkeypatch.setattr(tool, "encode_tensor_with_mixture", lambda _symbols, _params: b"payload")

    state_path = tmp_path / "state.pt"
    torch.save({"a": torch.zeros(8, dtype=torch.float32)}, state_path)
    manifest = tool.run_codec(state_path)

    _assert_proxy_contract(manifest)
    assert manifest["evidence_semantics"].endswith("_no_decoder_no_score")
    assert manifest["n_tensors"] == 1

    output = tmp_path / "manifest.json"
    evidence = tmp_path / "evidence.jsonl"
    assert tool.main([
        "--state-dict", str(state_path),
        "--output-json", str(output),
        "--output-evidence", str(evidence),
    ]) == 0
    row = json.loads(evidence.read_text(encoding="utf-8"))
    _assert_proxy_contract(row)


def test_remaining_cpu_mps_generators_define_proxy_contracts() -> None:
    targets = {
        "tools/pr101_compressai_balle_hyperprior.py": 2,
        "tools/pr101_compressai_balle_hyperprior_full.py": 5,
        "tools/pr101_compressai_factorized_prior.py": 2,
        "experiments/arch_shrink_quantizr_class_mps_overnight.py": 2,
        "tools/pr101_kalle_fold_8comp_hierarchical_codec.py": 2,
        "tools/pr101_arch_shrink_post_hoc_sweep.py": 3,
    }
    required_literals = [
        '"score_claim": False',
        '"promotion_eligible": False',
        '"rank_or_kill_eligible": False',
        '"ready_for_exact_eval_dispatch": False',
        '"dispatch_attempted": False',
        '"proxy_row": True',
        "missing_exact_cuda_auth_eval",
        "requires_exact_cuda_auth_eval_before_any_score_use",
    ]

    for rel_path, min_uses in targets.items():
        source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "def proxy_evidence_contract()" in source
        for literal in required_literals:
            assert literal in source, rel_path
        assert source.count("**proxy_evidence_contract()") >= min_uses, rel_path


def test_tiny_nn_200_param_tool_marks_gaussian_pmf_as_partial() -> None:
    tool = _load_tool("pr101_tiny_nn_200_param_faithful.py")

    fidelity = tool.model_spec_fidelity(n_params=188, model_brotli_bytes=435)

    assert fidelity["capacity_constraint_match"] is True
    assert fidelity["model_overhead_match"] is True
    assert fidelity["distribution_contract_match"] is False
    assert fidelity["1:1_fidelity"] is False
    assert "parametric_gaussian" in fidelity["model_spec_drift"][0]
