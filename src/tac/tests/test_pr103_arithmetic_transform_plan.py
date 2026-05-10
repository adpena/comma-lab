from __future__ import annotations

from pathlib import Path

import pytest

from tac.pr103_arithmetic_transform_plan import (
    PLAN_SCHEMA,
    Pr103ArithmeticTransformPlanError,
    build_pr103_arithmetic_transform_plan,
)
from tac.repo_io import write_json


def test_pr103_transform_plan_defaults_to_top_target_and_blocks_dispatch() -> None:
    manifest = _manifest()

    plan = build_pr103_arithmetic_transform_plan(schema_manifest=manifest)

    assert plan["schema"] == PLAN_SCHEMA
    assert plan["planning_only"] is True
    assert plan["score_claim"] is False
    assert plan["dispatch_attempted"] is False
    assert plan["ready_for_archive_preflight"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["target_stream"]["label"] == "stem.weight"
    assert plan["byte_accounting"]["expected_savings_bytes_upper_bound"] == 46
    assert plan["byte_accounting"]["expected_rate_score_delta_upper_bound"] < 0
    assert plan["byte_accounting"]["estimate_is_score_claim"] is False
    assert "candidate_runtime_adapter_missing" in plan["readiness_blockers"]
    assert "exact_cuda_auth_eval_missing" in plan["dispatch_blockers"]


def test_pr103_transform_plan_selects_target_by_rank_or_label() -> None:
    manifest = _manifest()

    by_rank = build_pr103_arithmetic_transform_plan(
        schema_manifest=manifest,
        target_rank=2,
    )
    by_label = build_pr103_arithmetic_transform_plan(
        schema_manifest=manifest,
        target_label="blocks.1.weight",
    )

    assert by_rank["target_stream"]["label"] == "blocks.1.weight"
    assert by_label["target_stream"]["label"] == "blocks.1.weight"
    assert by_rank["target_stream"]["model_gap_bytes_estimate"] == 45
    assert by_label["target_selection"]["label"] == "blocks.1.weight"


def test_pr103_transform_plan_carries_schema_manifest_blockers() -> None:
    manifest = _manifest()
    manifest["ready_for_schema_review"] = False
    manifest["merged_arithmetic_stream"]["reencoded_byte_identical"] = False

    plan = build_pr103_arithmetic_transform_plan(schema_manifest=manifest)

    assert "source_schema_manifest_not_ready_for_schema_review" in plan["readiness_blockers"]
    assert "source_merged_stream_reencode_not_byte_identical" in plan["readiness_blockers"]
    assert plan["ready_for_archive_preflight"] is False


def test_pr103_transform_plan_rejects_unknown_target() -> None:
    with pytest.raises(Pr103ArithmeticTransformPlanError, match="target label not found"):
        build_pr103_arithmetic_transform_plan(
            schema_manifest=_manifest(),
            target_label="missing.weight",
        )


def test_pr103_transform_plan_reads_manifest_from_path(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json(path, _manifest())

    plan = build_pr103_arithmetic_transform_plan(schema_manifest=path, repo_root=tmp_path)

    assert plan["source_schema_manifest"]["path"] == "manifest.json"
    assert plan["source_schema_manifest"]["sha256"]


def _manifest() -> dict:
    return {
        "planning_only": True,
        "score_claim": False,
        "ready_for_schema_review": True,
        "source_archive": {
            "path": "experiments/results/pr103/archive.zip",
            "bytes": 178223,
            "sha256": "a" * 64,
            "member_name": "x",
            "member_bytes": 178111,
            "member_sha256": "b" * 64,
        },
        "merged_arithmetic_stream": {
            "source_bytes": 153856,
            "source_sha256": "c" * 64,
            "decoded_symbol_count": 237561,
            "decoder_maybe_exhausted": True,
            "reencoded_byte_identical": True,
        },
        "next_arithmetic_schema_targets": [
            {
                "label": "stem.weight",
                "role": "ac_weight_tensor",
                "schema_index": 0,
                "symbol_count": 48384,
                "alphabet_size": 256,
                "decoded_symbols_sha256": "d" * 64,
                "observed_entropy_bits_per_symbol": 5.22,
                "model_cross_entropy_bits_per_symbol": 5.23,
                "observed_entropy_bytes_floor": 31582,
                "model_cross_entropy_bytes_floor": 31627,
                "model_gap_bytes_estimate": 46,
                "required_next_artifact": "byte_different_archive_manifest",
            },
            {
                "label": "blocks.1.weight",
                "role": "ac_weight_tensor",
                "schema_index": 4,
                "symbol_count": 46656,
                "alphabet_size": 256,
                "decoded_symbols_sha256": "e" * 64,
                "observed_entropy_bits_per_symbol": 5.56,
                "model_cross_entropy_bits_per_symbol": 5.57,
                "observed_entropy_bytes_floor": 32434,
                "model_cross_entropy_bytes_floor": 32478,
                "model_gap_bytes_estimate": 45,
            },
        ],
    }
