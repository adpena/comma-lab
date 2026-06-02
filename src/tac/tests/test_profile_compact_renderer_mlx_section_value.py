# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.archive_byte_profile import contest_rate_term
from tac.substrates.hprc.representation_spine import (
    HprcRepresentationFamily,
    build_generic_neural_spine_packet,
    write_representation_spine_projection,
)
from tac.substrates.hprc.spine_acquisition import build_spine_acquisition_report
from tac.substrates.hprc.spine_bounded_runner import build_spine_bounded_runner_plan

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.profile_compact_renderer_mlx_section_value import (  # noqa: E402
    HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
    build_compact_renderer_mlx_section_value_profile,
)

TOOL = REPO / "tools/profile_compact_renderer_mlx_section_value.py"


def test_build_profile_records_baseline_score_and_section_byte_blockers(
    tmp_path: Path,
) -> None:
    archive, projection = _archive_and_projection(tmp_path)
    runner, response, cache = _reports(tmp_path, archive, projection)

    profile = build_compact_renderer_mlx_section_value_profile(
        compact_runner_report_path=runner,
        mlx_response_path=response,
        cache_report_path=cache,
        repo_root=REPO,
    )

    assert profile["schema"] == HPRC_MLX_COMPONENT_PROFILE_SCHEMA
    assert profile["axis_tag"] == "[macOS-MLX research-signal]"
    assert profile["score_claim"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False
    assert profile["candidate_archive"]["bytes"] == archive.stat().st_size
    assert profile["candidate_archive"]["observed_sha256"] == profile["candidate_archive"]["sha256"]
    assert profile["score_components"]["avg_segnet_dist"] == 0.5
    assert profile["score_components"]["avg_posenet_dist"] == 1.6
    assert profile["score_components"]["segnet_score_component"] == 50.0
    assert profile["score_components"]["posenet_score_component"] == 4.0
    assert profile["score_components"]["nonrate_score"] == 54.0
    assert profile["score_components"]["rate_term"] == contest_rate_term(archive.stat().st_size)
    assert profile["training_context"]["evidence_role"] == "custody_rate_replay_smoke"
    assert profile["training_context"]["total_epochs_completed"] == 1
    assert profile["training_context"]["family_demote_eligible"] is False
    assert (
        profile["training_context"]["next_training_action"]
        == "continue_many_epoch_training_or_import_long_checkpoint_before_family_demotion"
    )
    assert profile["section_value_rows"] == [
        {
            **profile["section_value_rows"][0],
            "variant_id": "baseline",
            "neutralized_section": "none",
            "axis_tag": "[macOS-MLX research-signal]",
            "archive_zip_bytes": archive.stat().st_size,
            "delta_nonrate_score": 0.0,
        }
    ]
    sections = {row["section_name"]: row for row in profile["section_byte_records"]}
    assert sections["decoder_qw"]["section_bytes"] == 11
    assert sections["selectors_rc"]["section_bytes"] == 5
    assert sections["decoder_qw"]["value_status"] == (
        "blocked_missing_section_neutralization_or_ablation_evidence"
    )
    admission = profile["byte_price_admission_plan"]
    assert admission["schema"] == "compact_nerv_byte_price_controller.v1"
    assert admission["source_schema"] == profile["schema"]
    assert admission["input_row_count"] == 1
    assert admission["decision_rows"][0]["row_id"] == "baseline"
    assert admission["decision_rows"][0]["decision"] == "demote"
    assert "[macOS-MLX research-signal]" in admission["decision_rows"][0][
        "axis_labels"
    ]
    assert "axis_label_missing" not in admission["decision_rows"][0]["blockers"]
    assert admission["score_claim"] is False
    assert admission["ready_for_exact_eval_dispatch"] is False
    assert "compact_base_long_training_required_before_family_demote" in profile["blockers"]
    assert "section_neutralization_or_ablation_replay_missing" in profile["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in profile["blockers"]


def test_profile_is_bounded_runner_ingestable_without_false_section_evidence(
    tmp_path: Path,
) -> None:
    archive, projection = _archive_and_projection(tmp_path)
    runner, response, cache = _reports(tmp_path, archive, projection)
    profile = build_compact_renderer_mlx_section_value_profile(
        compact_runner_report_path=runner,
        mlx_response_path=response,
        cache_report_path=cache,
        repo_root=REPO,
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[projection],
        hard_byte_ceilings=[100_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
        mlx_profile_paths=[profile_path],
    )

    sections = {row["section_name"]: row for row in plan["section_value_rows"]}
    assert sections["decoder_qw"]["evidence_status"] == "missing"
    assert sections["selectors_rc"]["evidence_status"] == "missing"
    assert "some_sections_missing_value_per_byte_measurement" in plan["blockers"]
    assert "full_video_mlx_scorer_replay_not_attached" not in plan["blockers"]


def test_cli_writes_profile_and_refuses_overwrite_without_force(tmp_path: Path) -> None:
    archive, projection = _archive_and_projection(tmp_path)
    runner, response, cache = _reports(tmp_path, archive, projection)
    output = tmp_path / "out.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--compact-runner-report",
            str(runner),
            "--mlx-response",
            str(response),
            "--cache-report",
            str(cache),
            "--output",
            str(output),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    stdout = json.loads(completed.stdout)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["profile"] == str(output)
    assert stdout["section_byte_record_count"] == 3
    assert payload["source_reports"]["cache_report"]["path"] == str(cache)

    refused = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--compact-runner-report",
            str(runner),
            "--mlx-response",
            str(response),
            "--cache-report",
            str(cache),
            "--output",
            str(output),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "output exists" in refused.stderr


def test_profile_blocks_missing_projection_section_attribution(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"zip")
    runner, response, cache = _reports(tmp_path, archive, None)

    profile = build_compact_renderer_mlx_section_value_profile(
        compact_runner_report_path=runner,
        mlx_response_path=response,
        cache_report_path=cache,
        repo_root=REPO,
    )

    assert profile["section_byte_records"] == []
    assert "projection_manifest_path_missing" in profile["blockers"]
    assert "projection_section_byte_attribution_missing" in profile["blockers"]
    assert profile["scope_status"]["section"] == "blocked_missing_projection_section_manifest"


def _archive_and_projection(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"fake-zip-payload")
    spine = build_generic_neural_spine_packet(
        family=HprcRepresentationFamily.PACT_NERV_VQ,
        decoder_blob=b"d" * 11,
        selectors_blob=b"s" * 5,
        manifest_extra={
            "num_pairs": 600,
            "archive_bytes_are_authority_for_rate": True,
            "source_payload_kind": "test",
        },
    )
    spine.manifest["representation_spine"]["source"].update(
        {
            "archive_zip_path": archive.as_posix(),
            "archive_zip_bytes": archive.stat().st_size,
            "archive_zip_sha256": _sha256_file(archive),
        }
    )
    projection = write_representation_spine_projection(
        output_dir=tmp_path / "projection",
        spine=spine,
        basename="pact_nerv_vq_representation_spine",
    )
    return archive, Path(projection["manifest_path"])


def _reports(
    tmp_path: Path,
    archive: Path,
    projection: Path | None,
) -> tuple[Path, Path, Path]:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_manifest = cache_dir / "manifest.json"
    cache_manifest.write_text(json.dumps({"schema": "cache_manifest.v1"}), encoding="utf-8")
    inflated_manifest = tmp_path / "inflated_outputs_manifest.json"
    inflated_manifest.write_text(
        json.dumps({"schema": "inflated_manifest.v1"}),
        encoding="utf-8",
    )
    runner = tmp_path / "runner.json"
    runner.write_text(
        json.dumps(
            {
                "schema": "compact_renderer_mlx_spine_runner.v1",
                "execute_family": "pact_nerv_vq",
                "archive_path": archive.as_posix(),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": _sha256_file(archive),
                "projection_manifest_paths": ([] if projection is None else [projection.as_posix()]),
                "num_pairs": 600,
                "training_artifact": {
                    "config_snapshot": {"epochs": 1},
                    "per_epoch_metrics_count": 1,
                    "total_epochs_completed": 1,
                    "total_wall_clock_seconds": 0.25,
                },
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "schema_version": "mlx_scorer_response_payload.v1",
                "response_family": "pact_nerv_vq_test",
                "archive_size_bytes": archive.stat().st_size,
                "archive_sha256": _sha256_file(archive),
                "avg_segnet_dist": 0.5,
                "avg_posenet_dist": 1.6,
                "canonical_score": 54.0 + contest_rate_term(archive.stat().st_size),
                "score_rate_contribution": contest_rate_term(archive.stat().st_size),
                "batch_pairs": 1,
                "max_pairs": 600,
                "n_samples": 600,
                "candidate_cache_pairs": 600,
                "reference_cache_pairs": 600,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache_report.json"
    cache.write_text(
        json.dumps(
            {
                "schema": "mlx_scorer_cache_from_submission_inflate_only.v1",
                "archive": {
                    "path": archive.as_posix(),
                    "bytes": archive.stat().st_size,
                    "sha256": _sha256_file(archive),
                },
                "cache_manifest": cache_manifest.as_posix(),
                "inflated_outputs_manifest": inflated_manifest.as_posix(),
                "cached_pair_count": 600,
                "pair_count": 600,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    return runner, response, cache


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
