# SPDX-License-Identifier: MIT
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
    assert manifest["entropy_overhead_target_ranking"][0]["label"] == "large_sparse_qints"
    assert manifest["entropy_overhead_target_ranking"][0]["target_kind"] == "static_arithmetic_container_gap"
    assert manifest["entropy_overhead_target_ranking"][0]["ready_for_exact_eval_dispatch"] is False


def test_aq_floor_accounts_for_local_arithmetic_payload_termination_byte() -> None:
    manifest = build_entropy_codec_gap_audit(
        [
            {
                "label": "single_positive_large_alphabet",
                "actual_bytes": 64,
                "symbol_counts": {"0": 1200, "1": 0},
            }
        ]
    )

    row = manifest["streams"][0]
    assert row["entropy_floor_bits"] == pytest.approx(0.0)
    assert row["static_arithmetic_entropy_payload_floor_bytes"] == 0
    assert row["static_arithmetic_payload_floor_bytes"] == 1
    assert row["best_static_arithmetic_container_kind"] == "AQv1"
    assert row["best_static_arithmetic_container_floor_bytes"] == 37


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


def test_record_sequence_counts_and_exact_huffman_bits_are_deterministic() -> None:
    manifest = build_entropy_codec_gap_audit(
        [
            {
                "label": "records",
                "actual_bytes": 8,
                "symbol_counts": [
                    {"symbol": "b", "count": 1},
                    {"symbol": "a", "count": 3},
                ],
            }
        ]
    )

    row = manifest["streams"][0]
    assert row["symbol_counts"] == [
        {"symbol": "a", "count": 3},
        {"symbol": "b", "count": 1},
    ]
    assert row["huffman_payload_bits_exact"] == 4
    assert manifest["total_huffman_payload_bits_exact"] == 4


def test_known_overhead_accounting_ranks_model_gap_without_dispatch() -> None:
    manifest = build_entropy_codec_gap_audit(
        [
            {
                "label": "hdc2_prev_symbol_contexts",
                "actual_bytes": 221_381,
                "encoded_payload_bytes": 180_429,
                "model_overhead_bytes": 40_840,
                "container_overhead_bytes": 112,
                "symbol_counts": {"0": 16, "1": 8, "2": 4, "3": 4},
                "codec_surface": "src/tac/hnerv_decoder_recode.py",
            },
            {
                "label": "small_side_context",
                "actual_bytes": 64,
                "encoded_payload_bytes": 48,
                "model_overhead_bytes": 12,
                "container_overhead_bytes": 4,
                "symbol_counts": {"0": 15, "1": 1},
            },
        ]
    )

    overhead = manifest["known_overhead_accounting"]
    assert overhead["streams_with_known_accounting"] == 2
    assert overhead["complete_stream_accounting_count"] == 2
    assert overhead["total_known_model_overhead_bytes"] == 40_852
    assert overhead["total_known_container_overhead_bytes"] == 116
    assert overhead["largest_known_overhead_streams"][0]["label"] == "hdc2_prev_symbol_contexts"
    assert overhead["largest_known_overhead_streams"][0]["ready_for_exact_eval_dispatch"] is False
    targets = manifest["entropy_overhead_target_ranking"]
    assert targets[0]["label"] == "hdc2_prev_symbol_contexts"
    assert targets[0]["target_kind"] == "known_payload_entropy_gap"
    assert targets[0]["target_bytes_field"] == "known_payload_gap_to_entropy_floor_bytes"
    assert targets[0]["required_next_artifact"] == "roundtrip_payload_recode_manifest"
    assert "byte_equivalent_payload_entropy_recode_manifest" in targets[0][
        "exact_next_artifact_requirements"
    ]
    assert "missing_decoded_output_byte_equivalence_report" in targets[0][
        "byte_equivalence_blockers"
    ]
    assert targets[0]["score_claim"] is False
    assert targets[0]["ready_for_byte_closed_candidate_build"] is False
    assert targets[0]["ready_for_meta_lagrangian_atom_export"] is False
    assert targets[0]["ready_for_archive_preflight"] is False
    assert targets[0]["ready_for_exact_eval_dispatch"] is False
    assert "requires_exact_cuda_auth_eval" in targets[0]["dispatch_blockers"]
    atom_export = targets[0]["meta_lagrangian_atom_export"]
    assert atom_export["export_ready"] is False
    assert "archive_manifest_sha256" in atom_export["required_fields_before_export"]
    assert "missing_candidate_archive_manifest" in atom_export["export_blockers"]
    atom_template = atom_export["atom_template"]
    assert atom_template["atom_id"] == "hdc2_prev_symbol_contexts:known_payload_entropy_gap"
    assert atom_template["family"] == "hnerv_payload_entropy_recode"
    assert atom_template["family_group"] == "hnerv_rate_equivalent_recode"
    assert atom_template["byte_delta"] == -int(targets[0]["target_bytes"])
    assert atom_template["confidence"] == 0.0
    assert atom_template["raw_equal"] is False
    model_target = next(row for row in targets if row["target_kind"] == "known_model_overhead")
    assert model_target["target_bytes"] == 40_840
    rendered = render_markdown(manifest)
    assert "Known Overhead Accounting" in rendered
    assert "Entropy-Overhead Target Ranking" in rendered
    assert "hdc2_prev_symbol_contexts:known_payload_entropy_gap" in rendered


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
    with pytest.raises(EntropyCodecGapAuditError, match="positive integer"):
        build_entropy_codec_gap_audit(
            [{"label": "bad", "actual_bytes": 0, "symbol_counts": {"0": 1}}]
        )
    with pytest.raises(EntropyCodecGapAuditError, match="ASCII"):
        build_entropy_codec_gap_audit(
            [{"label": "bad", "actual_bytes": 1, "symbol_counts": {"\u03bc": 1, "a": 1}}],
        )
    with pytest.raises(EntropyCodecGapAuditError, match="symbol and count"):
        build_entropy_codec_gap_audit(
            [{"label": "bad", "actual_bytes": 1, "symbol_counts": [{"symbol": "0"}]}]
        )
    with pytest.raises(EntropyCodecGapAuditError, match="exceeds actual_bytes"):
        build_entropy_codec_gap_audit(
            [
                {
                    "label": "bad",
                    "actual_bytes": 4,
                    "encoded_payload_bytes": 4,
                    "model_overhead_bytes": 1,
                    "symbol_counts": {"0": 1},
                }
            ]
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


def test_audit_entropy_codec_gap_cli_fails_closed_on_bad_input(tmp_path: Path) -> None:
    input_path = tmp_path / "bad_streams.json"
    input_path.write_text(
        json.dumps({"streams": [{"label": "bad", "actual_bytes": 0, "symbol_counts": {"0": 1}}]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_entropy_codec_gap.py"),
            "--input",
            str(input_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "FATAL: entropy codec gap audit input rejected" in result.stderr


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
