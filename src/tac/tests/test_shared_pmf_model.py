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


def _rows() -> list[Any]:
    from tac.shared_pmf_model import coerce_symbol_stream

    return [
        coerce_symbol_stream("a", [0, 0, 0, 1, 1, 2, 2, 2], n_categories=4),
        coerce_symbol_stream("b", [0, 0, 1, 1, 1, 2, 2, 3], n_categories=4),
        coerce_symbol_stream("c", [3, 3, 3, 2, 2, 2, 1, 1], n_categories=4),
        coerce_symbol_stream("d", [3, 3, 2, 2, 2, 1, 1, 0], n_categories=4),
    ]


def _load_tool() -> Any:
    path = REPO / "tools" / "pr101_shared_model_pmf_probe.py"
    spec = importlib.util.spec_from_file_location("pr101_shared_model_pmf_probe_under_test", path)
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


def test_shared_pmf_model_serializes_and_range_roundtrips() -> None:
    from tac.shared_pmf_model import (
        SharedPMFConfig,
        build_shared_pmf_probe_result,
        compress_model,
        decompress_model,
        serialize_model,
    )

    result = build_shared_pmf_probe_result(
        _rows(),
        SharedPMFConfig(n_models=2, n_categories=4, total_frequency=1024, alpha=0.5, seed=123),
        archive_overhead_bytes=17,
    )

    assert result.model_roundtrip_ok is True
    assert result.compressed_model_roundtrip_ok is True
    assert result.payload_roundtrip_ok is True
    assert result.source_symbol_sha256 == result.reconstructed_symbol_sha256
    assert result.model_brotli_bytes > 0
    assert result.encoded_payload_bytes > 0
    assert result.archive_estimate_bytes == result.encoded_payload_bytes + result.model_brotli_bytes + 17
    assert serialize_model(decompress_model(compress_model(result.model))) == serialize_model(result.model)


def test_shared_pmf_model_is_deterministic() -> None:
    from tac.shared_pmf_model import SharedPMFConfig, build_shared_pmf_probe_result

    config = SharedPMFConfig(n_models=2, n_categories=4, total_frequency=1024, alpha=0.5, seed=999)
    first = build_shared_pmf_probe_result(_rows(), config, archive_overhead_bytes=0)
    second = build_shared_pmf_probe_result(_rows(), config, archive_overhead_bytes=0)

    assert first.model_raw_sha256 == second.model_raw_sha256
    assert first.model_brotli_sha256 == second.model_brotli_sha256
    assert first.payload_sha256 == second.payload_sha256
    assert first.archive_estimate_bytes == second.archive_estimate_bytes
    assert first.model.assignments == second.model.assignments


def test_shared_pmf_model_rejects_out_of_range_symbols() -> None:
    from tac.shared_pmf_model import coerce_symbol_stream

    with pytest.raises(ValueError, match="outside"):
        coerce_symbol_stream("bad", [0, 1, 4], n_categories=4)


def test_pr101_shared_model_probe_manifest_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    schema = (("a", (8,)), ("b", (8,)), ("c", (8,)), ("d", (8,)))
    _install_tiny_schema(monkeypatch, tool, schema)
    state_dict_path = _write_state(
        tmp_path / "state.pt",
        {
            "a": [-1, -1, 0, 0, 0, 1, 1, 2],
            "b": [-1, 0, 0, 0, 1, 1, 1, 2],
            "c": [5, 5, 5, 4, 4, 4, 3, 3],
            "d": [5, 5, 4, 4, 4, 3, 3, 2],
        },
    )

    manifest = tool.build_probe_manifest(
        state_dict_path,
        k_values=[1, 2],
        seed=20260507,
        total_frequency=1024,
    )
    best = manifest["best_model_by_archive_estimate"]

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "missing_exact_cuda_auth_eval" in manifest["dispatch_blockers"]
    assert manifest["source_state_dict_sha256"]
    assert manifest["deterministic_seed"] == 20260507
    assert best["model_bytes"] > 0
    assert best["encoded_payload_bytes"] > 0
    assert best["table_model_overhead_bytes"] == best["model_bytes"]
    assert best["archive_estimate_bytes"] == (
        best["encoded_payload_bytes"] + best["model_bytes"] + manifest["archive_overhead_bytes"]
    )
    assert best["roundtrip"]["exact_reconstruction_ok"] is True
    assert manifest["artifact_disposition_detail"]["negative_result_policy"]


def test_pr101_shared_model_probe_records_negative_loss_precisely(
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

    manifest = tool.build_probe_manifest(
        state_dict_path,
        k_values=[1],
        seed=7,
        total_frequency=1024,
        archive_overhead_bytes=tool.REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
    )
    best = manifest["best_model_by_archive_estimate"]

    assert manifest["disposition"] == "negative_loses_to_brotli_and_per_tensor_aac_after_model_and_payload_bytes"
    assert manifest["artifact_disposition_detail"]["beats_brotli_optuna"] is False
    assert manifest["artifact_disposition_detail"]["beats_per_tensor_aac"] is False
    assert best["delta_vs_brotli_optuna_archive_bytes"] == (
        best["archive_estimate_bytes"] - tool.REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES
    )


def test_pr101_shared_model_probe_cli_writes_manifest(
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

    assert tool.main([
        "--state-dict-path",
        str(state_dict_path),
        "--k-values",
        "1",
        "--total-frequency",
        "1024",
        "--output",
        str(output),
    ]) == 0

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["best_model_by_archive_estimate"]["roundtrip"]["exact_reconstruction_ok"] is True
