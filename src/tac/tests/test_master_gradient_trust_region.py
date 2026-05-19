# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tac.master_gradient_trust_region import (
    SCHEMA,
    TRUST_REGION_MODES,
    build_master_gradient_trust_region_candidates,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_master_gradient_trust_region_candidates.py"


def _operator_manifest(tmp_path: Path) -> dict[str, object]:
    sidecar = {
        "schema": "master_gradient_consumer_pose_axis_dominant_bytes_v1",
        "score_axis_dominance": {
            "selected": [
                {
                    "rank": 1,
                    "diagnostic_gradient_subject_byte_index": 10,
                    "pose_axis_share": 1.0,
                    "pose_axis_abs_score_contribution": 4.0,
                    "seg_axis_abs_score_contribution": 0.0,
                    "rate_axis_abs_score_contribution": 0.0,
                },
                {
                    "rank": 2,
                    "diagnostic_gradient_subject_byte_index": 20,
                    "pose_axis_share": 0.75,
                    "pose_axis_abs_score_contribution": 2.0,
                    "seg_axis_abs_score_contribution": 1.0,
                    "rate_axis_abs_score_contribution": 0.0,
                },
                {
                    "rank": 3,
                    "diagnostic_gradient_subject_byte_index": 30,
                    "pose_axis_share": 0.95,
                    "pose_axis_abs_score_contribution": 3.0,
                    "seg_axis_abs_score_contribution": 0.0,
                    "rate_axis_abs_score_contribution": 0.0,
                },
            ]
        },
    }
    sidecar_path = tmp_path / "selector.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    specs = []
    rows = []
    for rank in (1, 2, 3):
        spec_id = f"pose_axis_candidate::fixture::{rank:04d}::decoder"
        specs.append(
            {
                "spec_id": spec_id,
                "operator_id": f"op-{rank}",
                "score_claim": False,
                "ready_for_provider_dispatch": False,
            }
        )
        rows.append(
            {
                "rank": rank,
                "spec_id": spec_id,
                "diagnostic_gradient_subject_byte_index": rank * 10,
                "section_name": "decoder",
                "section_role": "brotli_streams_int8",
                "section_relative_offset": rank * 100,
                "mutation_operator": "decoder_codec_coordinate_response",
            }
        )
    return {
        "schema": "pose_byte_hoist_op7_manifest_v1",
        "archive_sha256": "a" * 64,
        "selector_sidecar_path": sidecar_path.as_posix(),
        "source_anchor": {"n_pairs_total": 600},
        "candidate_modification_specs": specs,
        "grammar_aware_operator_candidate_resolution": {
            "resolved_pose_axis_candidates": rows,
        },
        "blockers": ["packet_proofs_missing"],
    }


def _independence_report() -> dict[str, object]:
    return {
        "schema": "pair_independence_diagnostic_v1",
        "aggregate_verdict": "independence_assumption_blocked",
        "parameters": {"max_lag": 32},
        "cross_series_dependence": {"verdict": "cross_vector_dependence_blocked"},
        "series_reports": [
            {
                "serial_effective_sample_size": 21.25,
                "max_abs_autocorrelation": 0.6032,
                "lag_autocorrelation": {"1": 0.6, "32": 0.2},
            },
            {
                "serial_effective_sample_size": 40.22,
                "max_abs_autocorrelation": 0.4854,
                "lag_autocorrelation": {"1": 0.48, "32": 0.21},
            },
        ],
    }


def _score_response_outcome() -> dict[str, object]:
    return {
        "probe_id": "pr101_op7_raw_delta_exact_score_response_20260519T110500Z",
        "verdict": "DEFER",
        "source_archive_sha256": "a" * 64,
        "candidate_archive_sha256": "b" * 64,
        "contest_cpu_total_delta": 0.0016,
        "contest_cuda_total_delta": 0.0013,
    }


def test_trust_region_manifest_is_block_aware_and_non_promotional(tmp_path: Path) -> None:
    payload = build_master_gradient_trust_region_candidates(
        operator_manifest=_operator_manifest(tmp_path),
        repo_root=REPO_ROOT,
        independence_report=_independence_report(),
        score_response_outcome=_score_response_outcome(),
        max_rows_per_candidate=2,
    )

    assert payload["schema"] == SCHEMA
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["candidate_count"] == len(TRUST_REGION_MODES)
    assert payload["block_aware_pair_model"]["iid_assumption_allowed"] is False
    assert payload["block_aware_pair_model"]["temporal_block_size_pairs"] == 33
    assert payload["source_negative_control"]["same_length_raw_delta_regressed"] is True
    assert "packet_proofs_missing" in payload["blockers"]
    for candidate in payload["trust_region_candidates"]:
        assert candidate["score_claim"] is False
        assert candidate["ready_for_provider_dispatch"] is False
        assert candidate["selected_row_count"] == 2
        assert candidate["same_length_raw_delta_regression_guard"] is True


def test_segnet_boundary_mode_prefers_zero_segnet_rows(tmp_path: Path) -> None:
    payload = build_master_gradient_trust_region_candidates(
        operator_manifest=_operator_manifest(tmp_path),
        repo_root=REPO_ROOT,
        independence_report=_independence_report(),
        score_response_outcome=_score_response_outcome(),
        modes=("segnet_boundary_preserving",),
        max_rows_per_candidate=2,
    )

    rows = payload["trust_region_candidates"][0]["rows"]
    assert [row["rank"] for row in rows] == [1, 3]
    assert all(row["seg_axis_abs_score_contribution"] == 0.0 for row in rows)


def test_unknown_mode_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown trust-region modes"):
        build_master_gradient_trust_region_candidates(
            operator_manifest=_operator_manifest(tmp_path),
            repo_root=REPO_ROOT,
            modes=("bogus",),  # type: ignore[arg-type]
        )


def test_cli_writes_manifest(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("build_master_gradient_trust_region_candidates", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    operator_path = tmp_path / "operator.json"
    operator_path.write_text(json.dumps(_operator_manifest(tmp_path)), encoding="utf-8")
    independence_path = tmp_path / "independence.json"
    independence_path.write_text(json.dumps(_independence_report()), encoding="utf-8")
    probe_path = tmp_path / "probe.jsonl"
    probe_path.write_text(json.dumps(_score_response_outcome()) + "\n", encoding="utf-8")
    output_path = tmp_path / "trust_region.json"

    rc = module.main(
        [
            "--operator-manifest",
            str(operator_path),
            "--independence-report",
            str(independence_path),
            "--probe-outcome-jsonl",
            str(probe_path),
            "--probe-id",
            "pr101_op7_raw_delta_exact_score_response_20260519T110500Z",
            "--output-json",
            str(output_path),
            "--max-rows-per-candidate",
            "2",
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["candidate_count"] == len(TRUST_REGION_MODES)
