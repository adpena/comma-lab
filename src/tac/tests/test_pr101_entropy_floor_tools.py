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


def _load_tool(filename: str) -> Any:
    path = REPO / "tools" / filename
    spec = importlib.util.spec_from_file_location(f"{filename}_under_test", path)
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
        return SimpleNamespace(q_i8=tensor.numpy().astype(np.int8), scale=1.0)

    monkeypatch.setattr(module, "_quantize_tensor", fake_quantize)


def test_adaptive_ac_manifest_is_planning_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool("pr101_adaptive_arithmetic_coding.py")
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")
    output = tmp_path / "adaptive.json"

    assert tool.main([
        "--state-dict-path",
        str(state_dict_path),
        "--output",
        str(output),
    ]) == 0

    manifest = json.loads(output.read_text())
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "no_actual_adaptive_coder_bitstream" in manifest["dispatch_blockers"]
    assert manifest["input_state_dict_sha256"]
    assert len(manifest["per_tensor_results"]) == 2


def test_adaptive_ac_rejects_missing_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool("pr101_adaptive_arithmetic_coding.py")
    _install_tiny_schema(monkeypatch, tool)

    with pytest.raises(SystemExit):
        tool._separate_per_tensor_aac({"a": torch.ones(4)})


def test_pca_pmf_manifest_records_falsification_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool("pr101_pmf_pca_compression.py")
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")

    manifest = tool.pca_compress(state_dict_path, k_values=[1, 2])

    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "no_actual_pca_pmf_decoder_bitstream" in manifest["dispatch_blockers"]
    assert manifest["input_state_dict_sha256"]
    assert manifest["best_k_by_total_archive_with_brotli_overhead"]["k"] in {1, 2}
    assert len(manifest["k_sweep"]) == 2


def test_pca_pmf_rejects_invalid_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool("pr101_pmf_pca_compression.py")
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")

    with pytest.raises(SystemExit):
        tool.pca_compress(state_dict_path, k_values=[0])
    with pytest.raises(SystemExit):
        tool.pca_compress(state_dict_path, k_values=[3])


def test_entropy_floor_ladder_includes_context_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool("pr101_provable_optimal_floor.py")
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")
    output = tmp_path / "floors.json"

    assert tool.main([
        "--state-dict-path",
        str(state_dict_path),
        "--output",
        str(output),
    ]) == 0

    manifest = json.loads(output.read_text())
    floor_names = {row["name"] for row in manifest["provable_floors"]}
    assert {"iid_per_tensor", "markov1_per_tensor", "markov2_per_tensor"} <= floor_names
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "model_table_overhead_omitted_for_context_rows" in manifest["dispatch_blockers"]
    assert all(
        row["model_table_overhead_included"] is False
        for row in manifest["provable_floors"]
    )


def test_markov_context_floor_does_not_exceed_iid_for_repeated_pattern() -> None:
    tool = _load_tool("pr101_provable_optimal_floor.py")
    tensor_data = [np.array([127, 128, 127, 128, 127, 128], dtype=np.int32)]

    iid_bits = tool.floor_iid_per_tensor(tensor_data)
    markov1_bits = tool.floor_markov1_per_tensor(tensor_data)
    markov2_bits = tool.floor_markov2_per_tensor(tensor_data)

    assert markov1_bits <= iid_bits
    assert markov2_bits <= markov1_bits


def test_context_transform_probe_is_planning_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool("pr101_context_transform_floor_probe.py")
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")

    manifest = tool.build_transform_floor_report(state_dict_path)

    transforms = {row["transform"]: row for row in manifest["transforms"]}
    assert {"identity", "nibble_split", "zero_mask_nonzero_value"} <= set(transforms)
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "no_actual_transform_coder_bitstream" in manifest["dispatch_blockers"]
    assert transforms["identity"]["metadata_bytes_charged"] == 0
    assert transforms["identity"]["n_symbols_total"] == 8
    assert transforms["nibble_split"]["n_symbols_total"] == 16


def test_context_delta_transform_preserves_symbol_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool("pr101_context_transform_floor_probe.py")
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")

    manifest = tool.build_transform_floor_report(state_dict_path)
    transforms = {row["transform"]: row for row in manifest["transforms"]}

    assert transforms["delta_mod255"]["n_symbols_total"] == transforms["identity"][
        "n_symbols_total"
    ]
    assert transforms["delta_mod255"]["invertible_fixed_transform"] is True


def test_transition_table_cost_probe_is_planning_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool("pr101_markov_transition_table_cost.py")
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")

    manifest = tool.measure_transition_table_cost(state_dict_path)

    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "transition_table_not_wired_into_decoder" in manifest["dispatch_blockers"]
    assert manifest["n_nonzero_transition_pairs"] > 0
    assert manifest["recomputed_markov1_oracle_payload_bytes"] > 0
    serialization_names = {row["name"] for row in manifest["serializations"]}
    assert {"dense_u16", "dense_u32", "sparse_u16", "sparse_varint"} <= serialization_names


def test_markov1_aac_roundtrip_separates_oracle_floor_from_implemented_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool("pr101_markov1_aac_codec.py")
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")
    output = tmp_path / "markov.json"

    encoded = b"implemented-markov-payload"
    decoded = {
        "a": np.array([127, 128, 128, 126], dtype=np.int32),
        "b": np.array([129, 129, 127, 127], dtype=np.int32),
    }
    monkeypatch.setattr(
        tool,
        "encode_archive_markov1",
        lambda state_dict, *, alpha: (encoded, [{"n_symbols": 4}, {"n_symbols": 4}]),
    )
    monkeypatch.setattr(
        tool,
        "decode_archive_markov1",
        lambda archive: (decoded, []),
    )

    assert tool.main([
        "round-trip",
        "--state-dict-path",
        str(state_dict_path),
        "--output-summary",
        str(output),
    ]) == 0

    manifest = json.loads(output.read_text())
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "markov1_aac_archive_is_byte_negative_vs_brotli_anchor" in manifest[
        "dispatch_blockers"
    ]
    assert manifest["implemented_payload_bytes"] == len(encoded)
    assert manifest["oracle_markov1_floor_payload_bytes"] == 152106
    assert manifest["oracle_floor_is_implemented_codec_result"] is False


def test_per_tensor_shannon_cli_emits_fail_closed_dispatch_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool("per_tensor_shannon_analysis.py")
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")
    out_dir = tmp_path / "out"

    assert tool.main([
        "--state-dict",
        str(state_dict_path),
        "--output-dir",
        str(out_dir),
        "--no-stdout-table",
    ]) == 0

    manifest = json.loads((out_dir / "per_tensor_shannon.json").read_text())
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "shannon_floor_analysis_is_not_codec_bitstream" in manifest[
        "dispatch_blockers"
    ]
