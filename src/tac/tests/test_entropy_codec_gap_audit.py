from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.entropy_codec_gap_audit import (
    DISPATCH_BLOCKERS,
    EntropyCodecGapAuditError,
    build_entropy_codec_gap_audit,
    render_markdown,
)
from tac.repo_io import json_text

REPO = Path(__file__).resolve().parents[3]


def test_entropy_codec_gap_audit_is_deterministic_and_non_promotable() -> None:
    first = build_entropy_codec_gap_audit(_streams(), source_label="synthetic")
    second = build_entropy_codec_gap_audit(list(reversed(_streams())), source_label="synthetic")

    assert json_text(first) == json_text(second)
    assert first["planning_only"] is True
    assert first["score_claim"] is False
    assert first["score_evidence_grade"] == "invalid"
    assert first["ready_for_exact_eval_dispatch"] is False
    assert first["stream_count"] == 2
    assert first["total_actual_bytes"] == 140
    assert "requires_roundtrip_decode_validation" in first["dispatch_blockers"]
    assert "Entropy Codec Gap Audit" in render_markdown(first)
    for row in first["streams"]:
        assert row["score_claim"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["dispatch_blockers"] == DISPATCH_BLOCKERS


def test_huffman_floor_matches_known_binary_and_skewed_cases() -> None:
    manifest = build_entropy_codec_gap_audit(
        [
            {
                "label": "balanced_bits",
                "actual_bytes": 8,
                "symbol_counts": {"0": 4, "1": 4},
            },
            {
                "label": "skewed_ternary",
                "actual_bytes": 8,
                "symbol_counts": {"a": 6, "b": 1, "c": 1},
            },
        ]
    )
    rows = {row["label"]: row for row in manifest["streams"]}

    balanced = rows["balanced_bits"]
    assert balanced["entropy_bits_per_symbol"] == pytest.approx(1.0)
    assert balanced["huffman_bits_per_symbol"] == pytest.approx(1.0)
    assert balanced["huffman_code_lengths"] == {"0": 1, "1": 1}

    skewed = rows["skewed_ternary"]
    assert skewed["huffman_code_lengths"] == {"a": 1, "b": 2, "c": 2}
    assert skewed["huffman_payload_bits"] == pytest.approx(10.0)
    assert skewed["huffman_payload_bits"] >= skewed["entropy_floor_bits"]


def test_sparse_aqc1_floor_beats_dense_aqv1_for_large_sparse_alphabet() -> None:
    manifest = build_entropy_codec_gap_audit(
        [
            {
                "label": "large_sparse_qints",
                "actual_bytes": 4096,
                "symbol_counts": [100, *([0] * 509), 12, 0, 7, *([0] * 511)],
                "codec_surface": "tac.arithmetic_qint_codec.AQc1",
            }
        ]
    )

    row = manifest["streams"][0]
    assert row["alphabet_size"] == 1024
    assert row["positive_symbol_count"] == 3
    assert row["aq_floor_applicable"] is True
    assert row["best_static_arithmetic_container_kind"] == "AQc1"
    assert row["aqc1_sparse_model_floor_bytes"] < row["aqv1_static_model_floor_bytes"]
    assert row["gap_to_best_static_arithmetic_container_floor_bytes"] > 0
    assert manifest["opportunity_ranking"][0]["label"] == "large_sparse_qints"


def test_single_symbol_stream_has_zero_bit_huffman_but_no_aq_floor() -> None:
    manifest = build_entropy_codec_gap_audit(
        [{"label": "constant", "actual_bytes": 12, "symbol_counts": {"0": 1200}}]
    )

    row = manifest["streams"][0]
    assert row["entropy_floor_bits"] == pytest.approx(0.0)
    assert row["huffman_payload_bits"] == pytest.approx(0.0)
    assert row["huffman_degenerate_zero_bit_single_symbol"] is True
    assert row["aq_floor_applicable"] is False
    assert row["best_static_arithmetic_container_floor_bytes"] is None


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(EntropyCodecGapAuditError, match="nonempty"):
        build_entropy_codec_gap_audit([])
    with pytest.raises(EntropyCodecGapAuditError, match="non-negative integer"):
        build_entropy_codec_gap_audit(
            [{"label": "bad", "actual_bytes": -1, "symbol_counts": {"0": 1}}]
        )
    with pytest.raises(EntropyCodecGapAuditError, match="positive total"):
        build_entropy_codec_gap_audit(
            [{"label": "bad", "actual_bytes": 1, "symbol_counts": {"0": 0}}]
        )
    with pytest.raises(EntropyCodecGapAuditError, match="ASCII"):
        build_entropy_codec_gap_audit(
            [{"label": "bad", "actual_bytes": 1, "symbol_counts": {"\u03bc": 1, "a": 1}}],
        )


def test_audit_entropy_codec_gap_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    input_path = tmp_path / "streams.json"
    json_out = tmp_path / "gap.json"
    md_out = tmp_path / "gap.md"
    input_path.write_text(json.dumps({"streams": _streams()}, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_entropy_codec_gap.py"),
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
        cwd=REPO,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["source_label"] == "cli-fixture"
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["tool_run_manifest"]["tool"] == "tools/audit_entropy_codec_gap.py"
    assert "mask_tokens" in md_out.read_text(encoding="utf-8")


def _streams() -> list[dict]:
    return [
        {
            "label": "mask_tokens",
            "actual_bytes": 128,
            "symbol_counts": {"0": 90, "1": 6, "2": 1, "3": 1},
            "codec_surface": "src/tac/mask_entropy_coder.py",
        },
        {
            "label": "qint_signs",
            "actual_bytes": 12,
            "symbol_counts": {"neg": 1, "zero": 6, "pos": 1},
            "codec_surface": "src/tac/arithmetic_qint_codec.py",
        },
    ]
