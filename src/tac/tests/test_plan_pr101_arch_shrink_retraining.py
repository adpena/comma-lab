# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]


def _load_tool() -> Any:
    path = REPO / "tools" / "plan_pr101_arch_shrink_retraining.py"
    spec = importlib.util.spec_from_file_location("plan_pr101_arch_shrink_under_test", path)
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


def _tiny_entropy_report(path: Path) -> Path:
    payload = {
        "archive_overhead_bytes": 4,
        "empirical_encoders": [
            {
                "name": "brotli_optuna_optimum",
                "bytes_archive": 14,
                "bytes_payload": 10,
            }
        ],
        "provable_floors": [
            {"name": "iid_per_tensor", "bits": 24.0, "bytes_payload": 3},
            {"name": "markov1_per_tensor", "bits": 16.0, "bytes_payload": 2},
            {"name": "markov2_per_tensor", "bits": 8.0, "bytes_payload": 1},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _install_tiny_schema(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    monkeypatch.setattr(module, "FIXED_STATE_SCHEMA", (("a", (4,)), ("b", (4,))))

    def fake_quantize(name: str, tensor: torch.Tensor, *, n_quant: int) -> Any:
        del name, n_quant
        return SimpleNamespace(q_i8=tensor.numpy().astype("int8"), scale=1.0)

    monkeypatch.setattr(module, "_quantize_tensor", fake_quantize)
    monkeypatch.setattr(module, "encode_decoder_compact", lambda *_args, **_kwargs: b"0123456789")


def test_plan_separates_evidence_and_blocks_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")
    report_path = _tiny_entropy_report(tmp_path / "floor.json")
    scenarios = [
        tool.Scenario(
            name="control_current_pr101_int8",
            element_retention=1.0,
            quant_bits=8.0,
            entropy_ratio=1.0,
        ),
        tool.Scenario(
            name="tiny_shrink",
            element_retention=0.5,
            quant_bits=4.0,
            entropy_ratio=0.75,
            side_info_bytes=1,
        ),
    ]

    plan = tool.build_plan(
        state_dict_path=state_dict_path,
        entropy_floor_report=report_path,
        scenarios=scenarios,
        started_at_utc="2026-05-07T00:00:00Z",
    )

    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert set(plan["evidence_partition"]) == {"empirical", "derivation", "prediction"}
    assert plan["baseline"]["current_compact_decoder_payload_bytes"] == 10
    assert plan["baseline"]["iid_per_tensor_floor_payload_bytes"] == 3
    assert plan["scenarios"][0]["name"] == "tiny_shrink"
    assert plan["scenarios"][0]["evidence_grade"] == "prediction"
    assert plan["scenarios"][0]["byte_estimate"]["expected_archive_bytes"] < 14
    assert "no_exact_cuda_auth_eval" in plan["scenarios"][0]["dispatch_blockers"]


def test_unstructured_sparse_mask_overhead_is_charged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")
    report_path = _tiny_entropy_report(tmp_path / "floor.json")
    scenario = tool.Scenario(
        name="sparse_unstructured",
        element_retention=1.0,
        sparsity=0.5,
        quant_bits=4.0,
        entropy_ratio=1.0,
        sparse_mask_mode="unstructured_bitmask_brotli_estimate",
    )

    plan = tool.build_plan(
        state_dict_path=state_dict_path,
        entropy_floor_report=report_path,
        scenarios=[scenario],
        started_at_utc="2026-05-07T00:00:00Z",
    )

    row = plan["scenarios"][0]
    assert row["byte_estimate"]["mask_overhead_bytes"] > 0
    assert "unstructured_sparsity_mask_overhead_is_prediction_only" in row["dispatch_blockers"]


def test_scenario_validation_rejects_invalid_values() -> None:
    tool = _load_tool()

    with pytest.raises(ValueError, match="element_retention"):
        tool.Scenario(
            name="bad_retention",
            element_retention=0.0,
            quant_bits=8.0,
            entropy_ratio=1.0,
        ).validate()

    with pytest.raises(ValueError, match="quant_bits"):
        tool.Scenario(
            name="bad_bits",
            element_retention=1.0,
            quant_bits=9.0,
            entropy_ratio=1.0,
        ).validate()


def test_rendered_outputs_keep_score_claim_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "state.pt")
    report_path = _tiny_entropy_report(tmp_path / "floor.json")
    output_dir = tmp_path / "out"

    assert tool.main(
        [
            "--state-dict-path",
            str(state_dict_path),
            "--entropy-floor-report",
            str(report_path),
            "--output-dir",
            str(output_dir),
            "--started-at-utc",
            "2026-05-07T00:00:00Z",
        ]
    ) == 0

    manifest = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "plan.md").read_text(encoding="utf-8")
    assert manifest["score_claim"] is False
    assert "score_claim=false" in markdown
