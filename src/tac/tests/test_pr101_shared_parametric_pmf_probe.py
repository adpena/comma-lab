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
    path = REPO / "tools" / "pr101_shared_parametric_pmf_probe.py"
    spec = importlib.util.spec_from_file_location("pr101_shared_parametric_pmf_probe_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_tiny_schema(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    schema: tuple[tuple[str, tuple[int, ...]], ...],
) -> None:
    monkeypatch.setattr(module, "FIXED_STATE_SCHEMA", schema)

    def fake_quantize(name: str, tensor: torch.Tensor, *, n_quant: int) -> Any:
        del name, n_quant
        return SimpleNamespace(
            q_i8=tensor.detach().cpu().numpy().astype(np.int8),
            shape=tuple(int(v) for v in tensor.shape),
        )

    monkeypatch.setattr(module, "_quantize_tensor", fake_quantize)


def _write_state(path: Path, arrays: dict[str, list[int]]) -> Path:
    torch.save(
        {name: torch.tensor(values, dtype=torch.float32) for name, values in arrays.items()},
        path,
    )
    return path


def test_shared_pmf_manifest_is_planning_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    schema = (("a", (8,)), ("b", (8,)), ("c", (8,)))
    _install_tiny_schema(monkeypatch, tool, schema)
    state_dict_path = _write_state(
        tmp_path / "state.pt",
        {
            "a": [0, 0, 1, 1, -1, -1, 2, -2],
            "b": [0, 0, 0, 1, 1, -1, -1, -2],
            "c": [5, 5, 6, 6, 5, 6, 7, 7],
        },
    )

    manifest = tool.build_shared_model_report(
        state_dict_path,
        cluster_k_values=[1, 2, 3],
        scale_grid_size=2,
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["score_affecting_payload_changed"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "no_actual_range_or_ans_bitstream" in manifest["dispatch_blockers"]
    assert "no_runtime_model_serializer_or_decoder" in manifest["dispatch_blockers"]
    assert manifest["input_state_dict_sha256"]
    assert manifest["n_tensors"] == 3
    assert manifest["n_symbols"] == 24
    assert manifest["comparison_brotli_optuna_archive_bytes"] == 178_144
    assert manifest["comparison_per_tensor_aac_archive_bytes"] == 178_181
    assert manifest["comparison_iid_per_tensor_floor_archive_bytes"] == 175_916

    result_names = {row["name"] for row in manifest["model_results"]}
    assert "shared_parametric_spike_laplace_gaussian_identity" in result_names
    assert "shared_parametric_spike_laplace_gaussian_delta_mod255" in result_names
    assert "shared_empirical_temperature_spike_identity" in result_names
    assert "shared_canonical_pmf_clusters_identity_k1" in result_names
    for row in manifest["model_results"]:
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["rank_or_kill_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["model_parameter_bytes"] > 0
        assert "delta_vs_178144" in row
        assert "delta_vs_178181" in row
        assert "delta_vs_175916" in row


def test_shared_pmf_probe_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    schema = (("a", (10,)), ("b", (10,)), ("c", (10,)))
    _install_tiny_schema(monkeypatch, tool, schema)
    state_dict_path = _write_state(
        tmp_path / "state.pt",
        {
            "a": [0, 0, 0, 1, 1, 1, -1, -1, 2, -2],
            "b": [0, 1, 0, 1, 0, 1, 2, 2, -2, -2],
            "c": [9, 9, 9, 8, 8, 10, 10, 10, 7, 7],
        },
    )

    first = tool.build_shared_model_report(
        state_dict_path,
        cluster_k_values=[1, 2],
        scale_grid_size=2,
    )
    second = tool.build_shared_model_report(
        state_dict_path,
        cluster_k_values=[1, 2],
        scale_grid_size=2,
    )

    assert first == second


def test_shared_canonical_cluster_helps_identical_synthetic_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    schema = tuple((f"t{i}", (64,)) for i in range(6))
    _install_tiny_schema(monkeypatch, tool, schema)
    shared_values = [0] * 28 + [1] * 16 + [-1] * 12 + [2] * 4 + [-2] * 4
    state_dict_path = _write_state(
        tmp_path / "state.pt",
        {name: shared_values for name, _shape in schema},
    )

    manifest = tool.build_shared_model_report(
        state_dict_path,
        cluster_k_values=[1],
        scale_grid_size=1,
    )
    shared = next(
        row
        for row in manifest["model_results"]
        if row["name"] == "shared_canonical_pmf_clusters_identity_k1"
    )

    baseline = manifest["baseline_per_tensor_empirical_pmf"]
    assert shared["used_models"] == 1
    assert shared["archive_estimate_bytes"] < baseline["archive_estimate_bytes_raw_fp16_table"]
    assert shared["sharing_gain_vs_per_tensor_raw_fp16_table_bytes"] > 0


def test_shared_pmf_probe_rejects_missing_tensor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    schema = (("a", (4,)), ("b", (4,)))
    _install_tiny_schema(monkeypatch, tool, schema)
    state_dict_path = _write_state(tmp_path / "state.pt", {"a": [0, 1, -1, 2]})

    with pytest.raises(SystemExit, match="missing tensor"):
        tool.build_shared_model_report(state_dict_path, cluster_k_values=[1], scale_grid_size=1)


def test_shared_pmf_probe_writes_non_promotable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    schema = (("a", (8,)), ("b", (8,)))
    _install_tiny_schema(monkeypatch, tool, schema)
    state_dict_path = _write_state(
        tmp_path / "state.pt",
        {
            "a": [0, 0, 1, 1, -1, -1, 2, -2],
            "b": [0, 1, 0, 1, 0, -1, 2, -2],
        },
    )
    output = tmp_path / "manifest.json"
    evidence = tmp_path / "evidence.jsonl"

    assert tool.main([
        "--state-dict-path", str(state_dict_path),
        "--cluster-k-values", "1,2",
        "--scale-grid-size", "1",
        "--output", str(output),
        "--output-evidence", str(evidence),
    ]) == 0

    row = json.loads(evidence.read_text(encoding="utf-8"))
    assert row["technique"] == "shared_canonical_pmf_clusters"
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["rank_or_kill_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
