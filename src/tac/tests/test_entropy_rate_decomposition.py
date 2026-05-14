# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.entropy_rate_decomposition import (
    DISPATCH_BLOCKERS,
    EntropyRateDecompositionError,
    build_entropy_rate_decomposition,
    entropy_bits_per_symbol,
    render_markdown,
)
from tac.repo_io import json_text

REPO = Path(__file__).resolve().parents[3]


def test_entropy_rate_decomposition_output_is_deterministic() -> None:
    streams = _streams()

    first = build_entropy_rate_decomposition(streams, source_label="synthetic-hnerv")
    second = build_entropy_rate_decomposition(list(reversed(streams)), source_label="synthetic-hnerv")

    assert json_text(first) == json_text(second)
    assert first["planning_only"] is True
    assert first["score_claim"] is False
    assert first["score_evidence_grade"] == "invalid"
    assert first["ready_for_exact_eval_dispatch"] is False
    assert first["stream_count"] == 3
    assert [row["label"] for row in first["streams"]] == [
        "hnerv_decoder",
        "mask_tokens",
        "pose_delta_signs",
    ]
    assert first["opportunity_ranking"][0]["label"] == "hnerv_decoder"
    assert "Entropy-Rate Decomposition" in render_markdown(first)


def test_conditional_grouping_reduces_categorical_entropy_floor() -> None:
    manifest = build_entropy_rate_decomposition(
        [
            {
                "label": "mask_tokens",
                "stream_kind": "categorical",
                "actual_bytes": 4,
                "symbol_counts": {"0": 2, "1": 2},
                "conditional_groups": {
                    "prev_token": {
                        "prev_0": {"0": 2, "1": 0},
                        "prev_1": {"0": 0, "1": 2},
                    }
                },
            }
        ]
    )

    stream = manifest["streams"][0]
    model = stream["conditional_models"][0]
    assert entropy_bits_per_symbol({"0": 2, "1": 2}) == pytest.approx(1.0)
    assert stream["entropy_floor_bits"] == pytest.approx(4.0)
    assert stream["entropy_floor_bytes"] == pytest.approx(0.5)
    assert model["conditional_entropy_bits_per_symbol"] == pytest.approx(0.0)
    assert model["conditional_entropy_floor_bytes"] == pytest.approx(0.0)
    assert stream["best_conditional_model_label"] == "prev_token"
    assert stream["conditional_gain_over_unconditional_bytes"] == pytest.approx(0.5)
    assert stream["gap_to_best_conditional_floor_bytes"] == pytest.approx(4.0)


def test_invalid_counts_fail_closed() -> None:
    base = {
        "label": "bad",
        "stream_kind": "categorical",
        "actual_bytes": 10,
        "symbol_counts": {"0": 1, "1": 1},
    }

    with pytest.raises(EntropyRateDecompositionError, match="non-negative integer"):
        build_entropy_rate_decomposition([{**base, "symbol_counts": {"0": -1, "1": 2}}])

    with pytest.raises(EntropyRateDecompositionError, match="positive total"):
        build_entropy_rate_decomposition([{**base, "symbol_counts": {"0": 0, "1": 0}}])

    with pytest.raises(EntropyRateDecompositionError, match="non-negative integer"):
        build_entropy_rate_decomposition([{**base, "symbol_counts": {"0": 1.5, "1": 2}}])

    with pytest.raises(EntropyRateDecompositionError, match="conditional counts must sum"):
        build_entropy_rate_decomposition(
            [
                {
                    **base,
                    "conditional_groups": {
                        "bad_context": {
                            "g0": {"0": 1, "1": 0},
                            "g1": {"0": 0, "1": 2},
                        }
                    },
                }
            ]
        )


def test_manifest_and_streams_are_never_dispatch_ready() -> None:
    manifest = build_entropy_rate_decomposition(_streams())

    assert manifest["dispatch_attempted"] is False
    assert manifest["gpu_required"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "requires_exact_cuda_auth_eval" in manifest["dispatch_blockers"]
    assert "requires_byte_equivalent_codec_transform" in manifest["dispatch_blockers"]
    for stream in manifest["streams"]:
        assert stream["score_claim"] is False
        assert stream["dispatch_attempted"] is False
        assert stream["ready_for_exact_eval_dispatch"] is False
        assert stream["dispatch_blockers"] == DISPATCH_BLOCKERS
        for model in stream["conditional_models"]:
            assert model["ready_for_exact_eval_dispatch"] is False
            assert model["dispatch_blockers"] == DISPATCH_BLOCKERS
    for row in manifest["opportunity_ranking"]:
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["dispatch_blockers"] == DISPATCH_BLOCKERS


def test_audit_entropy_rate_decomposition_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    input_path = tmp_path / "streams.json"
    json_out = tmp_path / "entropy.json"
    md_out = tmp_path / "entropy.md"
    input_path.write_text(json.dumps({"streams": _streams()}, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_entropy_rate_decomposition.py"),
            "--input",
            str(input_path),
            "--source-label",
            "cli-fixture",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["source_label"] == "cli-fixture"
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "mask_tokens" in md_out.read_text(encoding="utf-8")


def _streams() -> list[dict]:
    return [
        {
            "label": "pose_delta_signs",
            "stream_kind": "pose",
            "actual_bytes": 12,
            "symbol_counts": {"neg": 1, "zero": 6, "pos": 1},
            "conditional_groups": {
                "axis": {
                    "x": {"neg": 1, "zero": 1, "pos": 0},
                    "y": {"neg": 0, "zero": 2, "pos": 0},
                    "z": {"neg": 0, "zero": 3, "pos": 1},
                }
            },
        },
        {
            "label": "hnerv_decoder",
            "stream_kind": "hnerv",
            "actual_bytes": 128,
            "symbol_counts": {"0": 8, "1": 8, "2": 16},
            "evidence_grade": "empirical",
        },
        {
            "label": "mask_tokens",
            "stream_kind": "categorical",
            "actual_bytes": 16,
            "symbol_counts": [6, 2, 2, 2],
            "conditional_groups": {
                "left_neighbor": {
                    "edge": [2, 0, 0, 0],
                    "flat": [4, 0, 2, 0],
                    "turn": [0, 2, 0, 2],
                }
            },
        },
    ]
