# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tac.cathedral_consumers import (
    cross_substrate_master_gradient_analyzer_consumer as consumer,
)
from tac.cross_substrate_master_gradient_analyzer import (
    CANONICAL_EQUATION_ID,
    CrossSubstrateMasterGradientAnalysisCorruptError,
    analyze_cross_substrate_master_gradients,
    append_analysis_locked,
    load_analyses_strict,
)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "tools" / "cross_substrate_master_gradient_cli.py").exists():
            return parent
    raise AssertionError("repo root not found")


def _load_cli_module():
    path = _repo_root() / "tools" / "cross_substrate_master_gradient_cli.py"
    spec = importlib.util.spec_from_file_location(
        "cross_substrate_master_gradient_cli_under_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _input(sha_char: str, gradient: np.ndarray, *, authoritative: bool = True) -> dict:
    return {
        "gradient_array": gradient,
        "archive_sha256": sha_char * 64,
        "measurement_axis": "[contest-CUDA]",
        "measurement_hardware": "linux_x86_64_t4",
        "measurement_call_id": f"fc-{sha_char}",
        "is_authoritative": authoritative,
    }


def test_analyzer_ranks_cross_substrate_axis_leverage_and_stays_false_authority():
    analysis = analyze_cross_substrate_master_gradients(
        [
            _input(
                "a",
                np.array(
                    [
                        [3.0, 0.0, 0.0],
                        [4.0, 0.0, 0.0],
                    ]
                ),
            ),
            _input(
                "b",
                np.array(
                    [
                        [0.0, 6.0, 0.0],
                        [0.0, 8.0, 0.0],
                    ]
                ),
            ),
        ],
        target_axes=("seg", "pose"),
        top_k_per_axis=1,
        top_n_opportunities=2,
        measurement_utc="2026-05-26T00:00:00+00:00",
    )

    assert analysis.axis_tag == "[predicted]"
    assert analysis.score_claim is False
    assert analysis.promotable is False
    assert analysis.canonical_equation_id == CANONICAL_EQUATION_ID
    assert analysis.canonical_equation_status == "FORMALIZATION_PENDING"
    assert analysis.canonical_helper_invocation == (
        "tac.cross_substrate_master_gradient_analyzer."
        "analyze_cross_substrate_master_gradients"
    )
    assert analysis.cauchy_schwarz_aggregate_upper_bound == pytest.approx(15.0)

    top = analysis.ranked_opportunities[0]
    assert top.archive_sha256 == "b" * 64
    assert top.axis == "pose"
    assert top.top_byte_indices == (1,)
    assert top.per_byte_leverage == pytest.approx(10.0 / np.sqrt(2.0))


def test_analyzer_rejects_invalid_gradient_shape():
    with pytest.raises(ValueError, match="shape"):
        analyze_cross_substrate_master_gradients(
            [_input("a", np.array([1.0, 2.0, 3.0]))],
        )


def test_analysis_ledger_append_and_strict_load(tmp_path: Path):
    analysis = analyze_cross_substrate_master_gradients(
        [_input("a", np.array([[1.0, 2.0, 3.0]]))],
        measurement_utc="2026-05-26T00:00:00+00:00",
    )
    ledger = tmp_path / "cross_substrate_master_gradient_analyses.jsonl"

    written = append_analysis_locked(analysis, path=ledger)
    rows = load_analyses_strict(ledger)

    assert rows == [written]
    assert rows[0]["score_claim"] is False
    assert rows[0]["promotable"] is False

    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(CrossSubstrateMasterGradientAnalysisCorruptError):
        load_analyses_strict(corrupt)


def test_cli_loader_keeps_advisory_rows_as_planning_not_contest_authority(
    tmp_path: Path,
):
    cli = _load_cli_module()
    gradient_path = tmp_path / "gradient.npy"
    np.save(gradient_path, np.array([[1.0, 2.0, 3.0]], dtype=np.float64))
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "archive_sha256": "c" * 64,
                "gradient_array_path": str(gradient_path),
                "measurement_axis": "[macOS-CPU advisory]",
                "measurement_hardware": "darwin_arm64_m5_max_macos_cpu_advisory",
                "measurement_call_id": "macos_local_test",
                "tensor_kind": "aggregate",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = cli._load_substrate_inputs(ledger)

    assert len(rows) == 1
    assert rows[0]["archive_sha256"] == "c" * 64
    assert rows[0]["is_authoritative"] is False
    assert np.array_equal(rows[0]["gradient_array"](), np.array([[1.0, 2.0, 3.0]]))


def test_cathedral_consumer_reports_ranked_observability_without_score_authority(
    monkeypatch: pytest.MonkeyPatch,
):
    analysis = analyze_cross_substrate_master_gradients(
        [
            _input("a", np.array([[1.0, 0.0, 0.0]])),
            _input("b", np.array([[0.0, 4.0, 0.0]])),
        ],
        target_axes=("pose",),
        top_k_per_axis=1,
        top_n_opportunities=1,
        measurement_utc="2026-05-26T00:00:00+00:00",
    ).as_dict()
    monkeypatch.setattr(consumer, "_load_latest_analysis", lambda: analysis)

    result = consumer.consume_candidate({"archive_sha256": "b" * 64})

    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"
    annotation = result["annotation"]
    assert annotation["cross_substrate_analysis_status"] == "RANKED"
    assert annotation["candidate_ranks_per_axis"] == [
        {
            "rank": 1,
            "axis": "pose",
            "per_byte_leverage": 4.0,
            "is_authoritative": True,
        }
    ]


def test_cathedral_consumer_missing_ledger_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(consumer, "_load_latest_analysis", lambda: None)

    result = consumer.consume_candidate({"archive_sha256": "a" * 64})

    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"
    assert result["annotation"]["cross_substrate_analysis_status"] == (
        "MISSING_ANALYSIS_LEDGER"
    )
