# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.local_acceleration import EVIDENCE_TAG_MLX
from tac.optimization.mlx_dynamic_sweep_observations import (
    MLXDynamicSweepObservationError,
    append_observation_row,
    build_observation_row,
    file_sha256,
    load_observation_rows,
    summarize_observation_file,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha(char: str) -> str:
    return char * 64


def _base_row(source_artifact: Path | None = None) -> dict:
    kwargs = {}
    if source_artifact is not None:
        kwargs["source_artifact_path"] = source_artifact
    return build_observation_row(
        candidate_id="prefix_k032",
        sweep_config_id="mlx_local_response",
        optimization_pass_id="smoke",
        family="decoder_q_selective_dqs1",
        observed_axis="[macOS-MLX research-signal]",
        evidence_tag=EVIDENCE_TAG_MLX,
        observed_score_or_delta=-0.00002,
        archive_sha256=_sha("a"),
        runtime_sha256=_sha("b"),
        raw_output_or_cache_sha256=_sha("c"),
        component_deltas={
            "segnet_delta": -0.0001,
            "posenet_delta": 0.0003,
            "rate_delta": -0.0002,
        },
        observed_at_utc="2026-05-22T14:00:00Z",
        **kwargs,
    )


def test_append_observation_row_jsonl_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"ok": true}\n', encoding="utf-8")
    output = tmp_path / "observations.jsonl"

    appended = append_observation_row(_base_row(source), output_path=output)

    assert output.is_file()
    rows = load_observation_rows(output)
    assert rows == [appended]
    assert appended["schema"] == "mlx_dynamic_sweep_observation.v1"
    assert appended["source_artifact_path"] == str(source)
    assert appended["source_artifact_sha256"] == file_sha256(source)
    assert appended["score_claim"] is False
    assert appended["promotion_eligible"] is False
    assert appended["rank_or_kill_eligible"] is False
    assert appended["ready_for_exact_eval_dispatch"] is False
    assert appended["promotable"] is False
    assert appended["component_deltas"]["segnet_delta"] == pytest.approx(-0.0001)


def test_summary_groups_by_candidate_config_pass_and_family(tmp_path: Path) -> None:
    output = tmp_path / "observations.jsonl"
    append_observation_row(_base_row(), output_path=output)
    second = _base_row()
    second["candidate_id"] = "prefix_k016"
    second["optimization_pass_id"] = "micro"
    second["observed_score_or_delta"] = -0.00001
    second["segnet_delta"] = -0.00005
    second["component_deltas"]["segnet_delta"] = -0.00005
    append_observation_row(second, output_path=output)

    summary = summarize_observation_file(output)

    assert summary["schema"] == "mlx_dynamic_sweep_observations.v1"
    assert summary["row_count"] == 2
    assert summary["by_candidate_id"]["prefix_k032"]["row_count"] == 1
    assert summary["by_sweep_config_id"]["mlx_local_response"]["row_count"] == 2
    assert summary["by_optimization_pass_id"]["smoke"]["row_count"] == 1
    assert summary["by_family"]["decoder_q_selective_dqs1"]["row_count"] == 2
    combo_key = "prefix_k016|mlx_local_response|micro|decoder_q_selective_dqs1"
    assert summary["by_candidate_config_pass_family"][combo_key]["row_count"] == 1
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False


def test_summary_preserves_exact_axis_evidence_labels(tmp_path: Path) -> None:
    output = tmp_path / "observations.jsonl"
    exact = _base_row()
    exact["observed_axis"] = "contest_cpu"
    exact["evidence_tag"] = "[contest-CPU]"
    exact["evidence_grade"] = "contest-CPU"

    append_observation_row(exact, output_path=output)
    summary = summarize_observation_file(output)

    assert summary["evidence_grade"] == "contest-CPU"
    assert summary["evidence_tag"] == "[contest-CPU]"
    assert summary["evidence_grades"] == ["contest-CPU"]
    assert summary["evidence_tags"] == ["[contest-CPU]"]


@pytest.mark.parametrize(
    "authority_field",
    [
        "score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
        "promotable",
    ],
)
def test_authority_fields_rejected(authority_field: str) -> None:
    row = _base_row()
    row[authority_field] = True

    with pytest.raises(MLXDynamicSweepObservationError, match=authority_field):
        build_observation_row(**_row_to_builder_kwargs(row))


def test_required_identity_and_hash_fields_are_enforced() -> None:
    row = _base_row()
    del row["candidate_id"]
    with pytest.raises(MLXDynamicSweepObservationError, match="candidate_id"):
        append_observation_row(row, output_path=Path("unused.jsonl"))

    row = _base_row()
    row["archive_sha256"] = "not-a-sha"
    with pytest.raises(MLXDynamicSweepObservationError, match="archive_sha256"):
        append_observation_row(row, output_path=Path("unused.jsonl"))

    row = _base_row()
    del row["raw_output_or_cache_sha256"]
    with pytest.raises(MLXDynamicSweepObservationError, match="raw_output_or_cache_sha256"):
        append_observation_row(row, output_path=Path("unused.jsonl"))


def test_cli_appends_and_prints_json_summary(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"source": true}\n', encoding="utf-8")
    output = tmp_path / "observations.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "append_mlx_dynamic_sweep_observation.py"),
            "--jsonl",
            str(output),
            "--candidate-id",
            "prefix_k032",
            "--sweep-config-id",
            "mlx_local_response",
            "--optimization-pass-id",
            "smoke",
            "--family",
            "decoder_q_selective_dqs1",
            "--observed-axis",
            "[macOS-MLX research-signal]",
            "--evidence-grade",
            "macOS-MLX-research-signal",
            "--observed-score-or-delta",
            "-0.00002",
            "--archive-sha256",
            _sha("a"),
            "--runtime-sha256",
            _sha("b"),
            "--raw-output-or-cache-sha256",
            _sha("c"),
            "--segnet-delta",
            "-0.0001",
            "--posenet-delta",
            "0.0003",
            "--rate-delta",
            "-0.0002",
            "--source-artifact",
            str(source),
            "--observed-at-utc",
            "2026-05-22T14:30:00Z",
            "--print-summary",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["schema"] == "mlx_dynamic_sweep_observations.v1"
    assert summary["row_count"] == 1
    assert summary["by_candidate_id"]["prefix_k032"]["row_count"] == 1
    stored = json.loads(output.read_text(encoding="utf-8").strip())
    assert stored["source_artifact_sha256"] == file_sha256(source)
    assert stored["evidence_grade"] == "macOS-MLX-research-signal"
    assert stored["score_claim"] is False
    assert stored["promotable"] is False


def _row_to_builder_kwargs(row: dict) -> dict:
    return {
        "candidate_id": row["candidate_id"],
        "sweep_config_id": row["sweep_config_id"],
        "optimization_pass_id": row["optimization_pass_id"],
        "family": row["family"],
        "observed_axis": row["observed_axis"],
        "evidence_tag": row["evidence_tag"],
        "observed_score_or_delta": row["observed_score_or_delta"],
        "archive_sha256": row["archive_sha256"],
        "runtime_sha256": row["runtime_sha256"],
        "raw_output_or_cache_sha256": row["raw_output_or_cache_sha256"],
        "component_deltas": row["component_deltas"],
        "observed_at_utc": row["observed_at_utc"],
        "extra": {
            key: value
            for key, value in row.items()
            if key
            in {
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
                "ready_for_exact_eval_dispatch",
                "promotable",
            }
        },
    }
