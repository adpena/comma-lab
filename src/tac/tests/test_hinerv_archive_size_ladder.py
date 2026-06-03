# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from comma_lab.storage_tiers import StorageTierError
from tac.analysis.hinerv_archive_size_ladder import (
    HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
    attach_hinerv_archive_ladder_score_rows,
    build_hinerv_archive_size_ladder,
    hinerv_modelsize_increment_section_value_rows,
    render_hinerv_archive_size_ladder_markdown,
)
from tac.analysis.nerv_decoder_weight_waterfill import (
    NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
)
from tac.analysis.nerv_modelsize_budget import (
    analyze_hinerv_modelsize_candidate,
    tag_hinerv_target_modelsize_candidate,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    DEMOTE,
    NERV_BYTE_PRICE_CONTROLLER_SCHEMA,
    build_nerv_byte_price_plan,
)
from tools import attach_hinerv_archive_ladder_scores as attach_cli
from tools import build_hinerv_archive_size_ladder as ladder_cli

REPO_ROOT = Path(__file__).resolve().parents[3]

try:
    import mlx.core  # noqa: F401

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


def _hinerv_budget_candidate(
    *,
    num_pairs: int = 1,
    latent_dim: int = 4,
    embed_dim: int = 4,
    decoder_channel: int = 4,
    decoder_codec: str = "int2_mixed",
    hard_byte_ceiling: int = 36_000,
    use_hierarchical_feature_grid: bool = False,
    use_convnext_blocks: bool = False,
    target_modelsize_mparams: float | None = None,
    **overrides,
) -> dict:
    row = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=hard_byte_ceiling,
        num_pairs=num_pairs,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        decoder_channel=decoder_channel,
        decoder_codec=decoder_codec,
        use_hierarchical_feature_grid=use_hierarchical_feature_grid,
        use_convnext_blocks=use_convnext_blocks,
    )
    if target_modelsize_mparams is not None:
        row = tag_hinerv_target_modelsize_candidate(
            row,
            target_modelsize_mparams=target_modelsize_mparams,
        )
    payload = row.as_dict()
    payload.update(overrides)
    return payload


def test_hinerv_archive_size_ladder_exports_one_tiny_row(tmp_path: Path) -> None:
    output_dir = tmp_path / "archive_ladder"
    report = build_hinerv_archive_size_ladder(
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        num_pairs=1,
        row_ids=("hi_nerv_local_tiny",),
        emit_receiver_proof=False,
        allow_local_output_dir=True,
        storage_reserve_free_gb=0.0,
        emit_decoder_weight_waterfill_plan=True,
        decoder_weight_waterfill_action_bits=(0, 2, 32),
    )

    assert report["schema"] == HINERV_ARCHIVE_SIZE_LADDER_SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["row_count"] == 1
    assert report["objective_authority"]["objective"] == "contest_auth_eval_scorer_only"
    assert "LPIPS" in report["objective_authority"]["forbidden_selection_terms"]
    assert report["local_output_explicitly_allowed"] is True
    assert report["storage_reserve_free_gb"] == 0.0
    assert report["storage_preflight"]["selected_workload_root"] == str(
        output_dir.resolve(strict=False)
    )
    assert report["storage_preflight"]["score_claim"] is False
    expected_backend = "mlx" if _MLX_AVAILABLE else "pytorch_portable_fallback"
    assert report["archive_export_backend_counts"] == {expected_backend: 1}
    assert "durable_evidence_on_selected_storage" in report[
        "artifact_retention_policy"
    ]
    assert "waterfill_group_bits_against_fixed_contest_byte_price" in report[
        "required_allocator_bindings"
    ]
    assert "receiver_proof_not_executed_for_archive_size_ladder" in report["blockers"]
    row = report["archive_rows"][0]
    assert row["row_id"] == "hi_nerv_local_tiny"
    assert row["modelsize_scale_source"] == "legacy_modelsize_scale"
    assert row["modelsize_scale_unit"] == "relative_local_ladder_multiplier"
    assert "not modelsize_mparams" in row["modelsize_receiver_contract"][
        "modelsize_scale_semantics"
    ]
    assert row["archive_export_backend"] == expected_backend
    if _MLX_AVAILABLE:
        assert row["backend_claim_blockers"] == []
    else:
        assert row["backend_claim_blockers"] == ["archive_export_backend_not_mlx"]
        assert "archive_export_backend_not_mlx" in row["blockers"]
        assert "archive_export_backend_not_mlx" in report["blockers"]
    assert row["archive_bytes"] == Path(row["archive_path"]).stat().st_size
    assert len(row["archive_sha256"]) == 64
    assert row["archive_rate_score_at_contest_price"] > 0.0
    assert row["spine_manifest_path"] is not None
    assert row["state_npz_manifest_path"] is not None
    assert row["decoder_weight_waterfill_plan_path"] is not None
    assert Path(row["decoder_weight_waterfill_plan_path"]).is_file()
    assert row["decoder_weight_waterfill_summary"]["schema"] == (
        NERV_DECODER_WEIGHT_WATERFILL_SCHEMA
    )
    assert "decoder_weight_saliency_missing_for_some_groups" in row[
        "decoder_weight_waterfill_summary"
    ]["blockers"]
    assert row["receiver_proof_executed"] is False
    assert row["runtime_consumption_proof_ready"] is None
    assert "adaptive_quantization_by_decoder_weight_group" in row[
        "required_allocator_bindings"
    ]
    assert "hinerv_archive_size_row_has_no_nonrate_score" in row["blockers"]
    assert report["section_value_rows"] == []
    assert report["byte_price_plan"]["schema"] == NERV_BYTE_PRICE_CONTROLLER_SCHEMA
    assert report["byte_price_plan"]["input_row_count"] == 0

    markdown = render_hinerv_archive_size_ladder_markdown(report)
    assert "HiNeRV archive-size ladder" in markdown
    assert "Decoder codec policy" in markdown
    assert "hi_nerv_local_tiny" in markdown


def test_hinerv_archive_size_ladder_exports_modelsize_budget_candidate_waterfill(
    tmp_path: Path,
) -> None:
    candidate_id = "hinerv_np1_ld4_ed8_dc4_mi1fi4_cnx_lg2c4_cx2k7_int4_mixed_ceil36000_tgtmp0p01"
    report = build_hinerv_archive_size_ladder(
        output_dir=tmp_path / "archive_ladder",
        repo_root=REPO_ROOT,
        num_pairs=1,
        row_ids=(candidate_id,),
        hinerv_modelsize_budget={
            "schema": "nerv_modelsize_budget.v1",
            "selected_candidates": [
                _hinerv_budget_candidate(
                    embed_dim=8,
                    decoder_codec="int4_mixed",
                    use_convnext_blocks=True,
                    target_modelsize_mparams=0.01,
                    nominal_total_payload_bytes=20_000,
                )
            ],
        },
        decoder_codec="int8_mixed",
        emit_receiver_proof=False,
        allow_local_output_dir=True,
        storage_reserve_free_gb=0.0,
        emit_decoder_weight_waterfill_plan=True,
        decoder_weight_waterfill_action_bits=(0, 2, 4, 8, 16, 32),
    )

    assert report["row_count"] == 1
    assert report["decoder_codec_policy"] == (
        "modelsize_budget_candidate_decoder_codec_overrides_top_level_default"
    )
    row = report["archive_rows"][0]
    assert row["row_id"] == candidate_id
    assert row["decoder_codec"] == "int4_mixed"
    assert row["modelsize_candidate"]["candidate_id"] == candidate_id
    assert row["modelsize_scale"] == pytest.approx(0.01)
    assert row["modelsize_scale_source"] == "target_modelsize_mparams"
    assert row["modelsize_scale_unit"] == "mparams"
    assert row["modelsize_receiver_contract"]["modelsize_scale_semantics"] == (
        "nearest local parameter-count target in millions of parameters"
    )
    assert row["modelsize_receiver_contract"][
        "modelsize_mparams_is_official_upstream_flag"
    ] is False
    assert row["modelsize_receiver_contract"][
        "modelsize_mparams_caps_archive_zip_bytes"
    ] is False
    assert row["nominal_total_payload_bytes"] == 20_000
    assert row["measured_minus_nominal_bytes"] == (
        row["archive_bytes"] - row["nominal_total_payload_bytes"]
    )
    assert row["config"]["latent_dim_coarse"] == 2
    assert row["config"]["latent_dim_mid"] == 4
    assert row["config"]["latent_dim_fine"] == 8
    assert row["config"]["embed_dim"] == 8
    assert row["config"]["decoder_channels"] == [4, 4, 4, 4, 4, 4, 4]
    assert row["config"]["use_convnext_blocks"] is True
    assert row["config"]["mid_injection_block_index"] == 1
    waterfill_path = Path(row["decoder_weight_waterfill_plan_path"])
    assert waterfill_path.is_file()
    waterfill = json.loads(waterfill_path.read_text(encoding="utf-8"))
    assert waterfill["schema"] == NERV_DECODER_WEIGHT_WATERFILL_SCHEMA
    assert waterfill["candidate_id"] == candidate_id
    assert waterfill["group_count"] > 0
    assert waterfill["score_claim"] is False


def test_hinerv_archive_size_ladder_honors_full_video_saliency_metadata(
    tmp_path: Path,
) -> None:
    candidate_id = "hinerv_np1_ld4_ed8_dc4_mi1fi4_cnx_lg2c4_cx2k7_int4_mixed_ceil36000_tgtmp0p01"
    saliency_path = tmp_path / "saliency_replay.json"
    saliency_path.write_text(
        json.dumps(
            {
                "schema": "hinerv_decoder_weight_saliency_replay.v1",
                "full_video_coverage": True,
                "pair_schedule": {"max_pairs": 1, "start_pair": 0, "pair_stride": 1},
                "saliency_by_name": {"head_rgb_0.bias": 1.0},
                "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    report = build_hinerv_archive_size_ladder(
        output_dir=tmp_path / "archive_ladder",
        repo_root=REPO_ROOT,
        num_pairs=1,
        row_ids=(candidate_id,),
        hinerv_modelsize_budget={
            "schema": "nerv_modelsize_budget.v1",
            "selected_candidates": [
                _hinerv_budget_candidate(
                    embed_dim=8,
                    decoder_codec="int4_mixed",
                    use_convnext_blocks=True,
                    target_modelsize_mparams=0.01,
                    nominal_total_payload_bytes=20_000,
                )
            ],
        },
        decoder_codec="int8_mixed",
        emit_receiver_proof=False,
        allow_local_output_dir=True,
        storage_reserve_free_gb=0.0,
        emit_decoder_weight_waterfill_plan=True,
        decoder_weight_saliency_json=saliency_path,
    )

    assert report["decoder_weight_saliency_metadata"]["full_video_coverage"] is True
    row = report["archive_rows"][0]
    assert row["decoder_weight_saliency_full_video_coverage"] is True
    waterfill = json.loads(
        Path(row["decoder_weight_waterfill_plan_path"]).read_text(encoding="utf-8")
    )
    assert "full_video_coverage_missing" not in waterfill["blockers"]
    assert all(
        "full_video_coverage_missing" not in waterfill_row["blockers"]
        for waterfill_row in waterfill["rows"]
    )
    assert waterfill["score_claim"] is False
    assert waterfill["ready_for_exact_eval_dispatch"] is False


def test_hinerv_archive_size_ladder_rejects_declared_full_without_schedule_proof(
    tmp_path: Path,
) -> None:
    saliency_path = tmp_path / "saliency_replay_declared_only.json"
    saliency_path.write_text(
        json.dumps(
            {
                "schema": "hinerv_decoder_weight_saliency_replay.v1",
                "full_video_coverage": True,
                "saliency_by_name": {"head_rgb_0.bias": 1.0},
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    report = build_hinerv_archive_size_ladder(
        output_dir=tmp_path / "archive_ladder",
        repo_root=REPO_ROOT,
        num_pairs=1,
        row_ids=("hi_nerv_local_tiny",),
        emit_receiver_proof=False,
        allow_local_output_dir=True,
        storage_reserve_free_gb=0.0,
        emit_decoder_weight_waterfill_plan=True,
        decoder_weight_saliency_json=saliency_path,
        decoder_weight_waterfill_action_bits=(0, 2, 32),
    )

    metadata = report["decoder_weight_saliency_metadata"]
    assert metadata["declared_full_video_coverage"] is True
    assert metadata["schedule_full_video_coverage"] is False
    assert metadata["full_video_coverage"] is False
    assert "decoder_weight_saliency_declared_full_without_schedule_proof" in metadata[
        "coverage_blockers"
    ]
    row = report["archive_rows"][0]
    assert row["decoder_weight_saliency_full_video_coverage"] is False
    waterfill = json.loads(
        Path(row["decoder_weight_waterfill_plan_path"]).read_text(encoding="utf-8")
    )
    assert "full_video_coverage_missing" in waterfill["blockers"]
    assert any(
        "full_video_coverage_missing" in waterfill_row["blockers"]
        for waterfill_row in waterfill["rows"]
    )
    assert waterfill["score_claim"] is False


def test_hinerv_archive_size_ladder_reports_missing_requested_row(tmp_path: Path) -> None:
    report = build_hinerv_archive_size_ladder(
        output_dir=tmp_path / "archive_ladder",
        repo_root=REPO_ROOT,
        num_pairs=1,
        row_ids=("does_not_exist",),
        allow_local_output_dir=True,
        storage_reserve_free_gb=0.0,
    )

    assert report["row_count"] == 0
    assert report["missing_requested_row_ids"] == ["does_not_exist"]
    assert "hinerv_archive_size_ladder_requested_rows_missing" in report["blockers"]


def test_build_hinerv_archive_size_ladder_cli_records_invocation(
    tmp_path: Path,
    capsys,
) -> None:
    output_json = tmp_path / "ladder.json"
    output_md = tmp_path / "ladder.md"
    candidate_id = "hinerv_np1_ld4_ed4_dc4_mi1fi4_int2_mixed_ceil36000"
    budget_json = tmp_path / "budget.json"
    budget_json.write_text(
        json.dumps(
            {
                "schema": "nerv_modelsize_budget.v1",
                "selected_candidates": [
                    _hinerv_budget_candidate()
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rc = ladder_cli.main(
        [
            "--output-dir",
            str(tmp_path / "archive_ladder"),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--repo-root",
            str(REPO_ROOT),
            "--num-pairs",
            "1",
            "--row-id",
            candidate_id,
            "--hinerv-modelsize-budget-json",
            str(budget_json),
            "--allow-local-output-dir",
            "--storage-reserve-free-gb",
            "0",
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["schema"] == HINERV_ARCHIVE_SIZE_LADDER_SCHEMA
    assert payload["tool_invocation"]["schema"] == (
        "hinerv_archive_size_ladder_tool_invocation.v1"
    )
    assert payload["tool_invocation"]["tool"] == (
        "tools/build_hinerv_archive_size_ladder.py"
    )
    assert "--row-id" in payload["tool_invocation"]["argv"]
    provenance = payload["tool_invocation"]["input_files"][
        "hinerv_modelsize_budget_json"
    ]
    assert provenance["path"] == budget_json.as_posix()
    assert provenance["exists"] is True
    assert provenance["bytes"] == budget_json.stat().st_size
    assert len(provenance["sha256"]) == 64
    assert provenance["score_claim"] is False
    assert provenance["ready_for_exact_eval_dispatch"] is False
    assert payload["tool_invocation"]["score_claim"] is False
    assert payload["markdown_report_path"] == output_md.as_posix()
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    markdown = output_md.read_text(encoding="utf-8")
    assert "Decoder codec policy" in markdown
    assert "Modelsize receiver contract" in markdown


def test_hinerv_archive_size_ladder_consumes_modelsize_budget_candidate(
    tmp_path: Path,
) -> None:
    candidate_id = "hinerv_np1_ld4_ed4_dc4_mi1fi4_int2_mixed_ceil36000"
    report = build_hinerv_archive_size_ladder(
        output_dir=tmp_path / "archive_ladder",
        repo_root=REPO_ROOT,
        num_pairs=1,
        row_ids=(candidate_id,),
        hinerv_modelsize_budget={
            "schema": "nerv_modelsize_budget.v1",
            "selected_candidates": [
                _hinerv_budget_candidate()
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        decoder_codec="int8_mixed",
        allow_local_output_dir=True,
        storage_reserve_free_gb=0.0,
    )

    assert report["row_count"] == 1
    assert report["decoder_codec"] == "int8_mixed"
    assert report["decoder_codec_policy"] == (
        "modelsize_budget_candidate_decoder_codec_overrides_top_level_default"
    )
    assert report["hinerv_modelsize_budget_schema"] == "nerv_modelsize_budget.v1"
    row = report["archive_rows"][0]
    assert row["row_id"] == candidate_id
    assert row["decoder_codec"] == "int2_mixed"
    assert row["modelsize_candidate"]["candidate_id"] == candidate_id
    assert row["modelsize_candidate"]["decoder_codec"] == "int2_mixed"
    assert row["modelsize_scale_source"] == "modelsize_mparams"
    assert row["modelsize_scale_unit"] == "mparams"
    assert row["modelsize_receiver_contract"]["archive_bytes_authority"] == (
        "archive_rows[].archive_bytes"
    )
    assert row["modelsize_receiver_contract"][
        "modelsize_mparams_is_official_upstream_flag"
    ] is False
    assert row["modelsize_receiver_contract"][
        "modelsize_mparams_caps_archive_zip_bytes"
    ] is False
    assert row["nominal_total_payload_bytes"] > 0
    assert row["measured_minus_nominal_bytes"] == (
        row["archive_bytes"] - row["nominal_total_payload_bytes"]
    )
    assert row["config"]["mid_injection_block_index"] == 1
    assert row["config"]["fine_injection_block_index"] == 4
    assert Path(row["archive_path"]).is_file()
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_hinerv_archive_size_ladder_rejects_mismatched_budget_config(
    tmp_path: Path,
) -> None:
    candidate_id = "hinerv_np1_ld4_ed4_dc4_mi1fi4_int2_mixed_ceil36000"
    with pytest.raises(ValueError, match="modelsize candidate config mismatch"):
        build_hinerv_archive_size_ladder(
            output_dir=tmp_path / "archive_ladder",
            repo_root=REPO_ROOT,
            num_pairs=1,
            row_ids=(candidate_id,),
            hinerv_modelsize_budget={
                "schema": "nerv_modelsize_budget.v1",
                "selected_candidates": [
                    _hinerv_budget_candidate(
                        decoder_channels=[5, 5, 5, 5, 5, 5, 5]
                    )
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            decoder_codec="int8_mixed",
            allow_local_output_dir=True,
            storage_reserve_free_gb=0.0,
        )


def test_hinerv_archive_size_ladder_rejects_true_nested_authority_flags(
    tmp_path: Path,
) -> None:
    candidate_id = "hinerv_np1_ld4_ed4_dc4_mi1fi4_int2_mixed_ceil36000"
    with pytest.raises(ValueError, match="forbidden true authority flags"):
        build_hinerv_archive_size_ladder(
            output_dir=tmp_path / "archive_ladder",
            repo_root=REPO_ROOT,
            num_pairs=1,
            row_ids=(candidate_id,),
            hinerv_modelsize_budget={
                "schema": "nerv_modelsize_budget.v1",
                "selected_candidates": [
                    _hinerv_budget_candidate(score_claim=True)
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            decoder_codec="int8_mixed",
            allow_local_output_dir=True,
            storage_reserve_free_gb=0.0,
        )


def test_hinerv_archive_size_ladder_rejects_missing_modelsize_contract(
    tmp_path: Path,
) -> None:
    candidate_id = "hinerv_np1_ld4_ed4_dc4_mi1fi4_int2_mixed_ceil36000"
    candidate = _hinerv_budget_candidate()
    candidate.pop("modelsize_control_contract")

    with pytest.raises(ValueError, match="modelsize_control_contract"):
        build_hinerv_archive_size_ladder(
            output_dir=tmp_path / "archive_ladder",
            repo_root=REPO_ROOT,
            num_pairs=1,
            row_ids=(candidate_id,),
            hinerv_modelsize_budget={
                "schema": "nerv_modelsize_budget.v1",
                "selected_candidates": [candidate],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            decoder_codec="int8_mixed",
            allow_local_output_dir=True,
            storage_reserve_free_gb=0.0,
        )


def test_hinerv_archive_size_ladder_rejects_ambiguous_target_modelsize_contract(
    tmp_path: Path,
) -> None:
    candidate_id = "hinerv_np1_ld4_ed8_dc4_mi1fi4_cnx_lg2c4_cx2k7_int4_mixed_ceil36000_tgtmp0p01"
    candidate = _hinerv_budget_candidate(
        embed_dim=8,
        decoder_codec="int4_mixed",
        use_convnext_blocks=True,
        target_modelsize_mparams=0.01,
    )
    candidate["modelsize_control_contract"] = {
        **candidate["modelsize_control_contract"],
        "control_semantics": "official_upstream_modelsize",
        "shared_target_modelsize_mparams_consumed_as": (
            "official_upstream_modelsize"
        ),
        "modelsize_mparams_is_official_upstream_flag": True,
        "modelsize_mparams_caps_archive_zip_bytes": True,
    }

    with pytest.raises(
        ValueError,
        match="modelsize_mparams_must_not_be_official_upstream_flag",
    ):
        build_hinerv_archive_size_ladder(
            output_dir=tmp_path / "archive_ladder",
            repo_root=REPO_ROOT,
            num_pairs=1,
            row_ids=(candidate_id,),
            hinerv_modelsize_budget={
                "schema": "nerv_modelsize_budget.v1",
                "selected_candidates": [candidate],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            decoder_codec="int8_mixed",
            allow_local_output_dir=True,
            storage_reserve_free_gb=0.0,
        )


def test_hinerv_archive_size_ladder_rejects_candidate_id_control_mismatch(
    tmp_path: Path,
) -> None:
    candidate_id = "hinerv_np1_ld4_ed4_dc5_mi1fi4_int2_mixed_ceil36000"
    with pytest.raises(ValueError, match="candidate_id source controls mismatch"):
        build_hinerv_archive_size_ladder(
            output_dir=tmp_path / "archive_ladder",
            repo_root=REPO_ROOT,
            num_pairs=1,
            row_ids=(candidate_id,),
            hinerv_modelsize_budget={
                "schema": "nerv_modelsize_budget.v1",
                "selected_candidates": [
                    _hinerv_budget_candidate(candidate_id=candidate_id)
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            decoder_codec="int8_mixed",
            allow_local_output_dir=True,
            storage_reserve_free_gb=0.0,
        )


def test_hinerv_archive_size_ladder_rejects_mismatched_budget_candidate_id(
    tmp_path: Path,
) -> None:
    candidate_id = "hinerv_np1_ld4_ed32_dc4_mi1fi4_int2_mixed_ceil36000"
    with pytest.raises(ValueError, match="candidate_id source controls mismatch"):
        build_hinerv_archive_size_ladder(
            output_dir=tmp_path / "archive_ladder",
            repo_root=REPO_ROOT,
            num_pairs=1,
            row_ids=(candidate_id,),
            hinerv_modelsize_budget={
                "schema": "nerv_modelsize_budget.v1",
                "selected_candidates": [
                    _hinerv_budget_candidate(candidate_id=candidate_id)
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            decoder_codec="int8_mixed",
            allow_local_output_dir=True,
            storage_reserve_free_gb=0.0,
        )


def test_hinerv_archive_size_ladder_rejects_local_output_by_default(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "archive_ladder"

    with pytest.raises(StorageTierError, match="local_disk_tier_disabled"):
        build_hinerv_archive_size_ladder(
            output_dir=output_dir,
            repo_root=REPO_ROOT,
            num_pairs=1,
            row_ids=("does_not_exist",),
        )

    assert not output_dir.exists()


def test_hinerv_modelsize_increment_rows_feed_byte_price_controller() -> None:
    rows = hinerv_modelsize_increment_section_value_rows(
        [
            {
                "from_row_id": "tiny",
                "to_row_id": "small",
                "bytes_added": 4096,
                "required_nonrate_score_improvement": 0.0125,
            }
        ]
    )

    assert rows[0]["section_id"] == "hinerv_modelsize_increment:tiny->small"
    assert rows[0]["row_kind"] == "new_residual_or_sidecar"
    assert rows[0]["byte_delta"] == 4096
    assert rows[0]["delta_nonrate_score"] is None
    assert rows[0]["required_nonrate_score_improvement"] == 0.0125
    assert rows[0]["score_claim"] is False

    plan = build_nerv_byte_price_plan(
        {
            "schema": HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
            "family": "hi_nerv",
            "axis_tag": "[planning/control]",
            "section_value_rows": rows,
        }
    )
    plan_row = plan["decision_rows"][0]
    assert plan["schema"] == NERV_BYTE_PRICE_CONTROLLER_SCHEMA
    assert plan_row["decision"] == DEMOTE
    assert plan_row["delta_rate_score"] > 0.0
    assert plan_row["delta_total_score"] is None
    assert "delta_nonrate_score_missing" in plan_row["blockers"]
    assert "receiver_proof_not_satisfied" in plan_row["blockers"]


def test_hinerv_archive_ladder_score_attachment_prices_measured_increments() -> None:
    ladder = _ladder_for_score_attachment()

    attached = attach_hinerv_archive_ladder_score_rows(
        ladder,
        {
            "schema": "hinerv_full_video_mlx_score_rows.v1",
            "axis_tag": "[macOS-MLX research-signal]",
            "score_rows": [
                {
                    "row_id": "tiny",
                    "nonrate_score": 0.230,
                    "avg_segnet_dist": 0.001,
                    "avg_posenet_dist": 0.00169,
                    "num_pairs": 600,
                    "source_report_sha256": "a" * 64,
                },
                {
                    "row_id": "small",
                    "nonrate_score": 0.180,
                    "avg_segnet_dist": 0.0007,
                    "avg_posenet_dist": 0.00121,
                    "num_pairs": 600,
                    "source_report_sha256": "b" * 64,
                },
            ],
        },
        score_source_path="/Volumes/VertigoDataTier/pact/hinerv_scores.json",
    )

    rows = {row["row_id"]: row for row in attached["archive_rows"]}
    assert rows["tiny"]["nonrate_score"] == pytest.approx(0.230)
    assert rows["small"]["measured_score_full_video_coverage"] is True
    assert rows["small"]["measured_score_custody_trusted"] is True
    assert "hinerv_archive_size_row_has_no_nonrate_score" not in rows["tiny"][
        "blockers"
    ]
    assert attached["score_attachment"]["matched_archive_row_count"] == 2
    assert attached["score_attachment"]["matched_full_video_row_count"] == 2
    assert attached["score_attachment"]["trusted_score_row_count"] == 2
    section = attached["section_value_rows"][0]
    assert section["section_id"] == "hinerv_modelsize_increment:tiny->small"
    assert section["delta_nonrate_score"] == pytest.approx(-0.05)
    assert section["byte_delta"] == 10_000
    assert section["receiver_proof_status"] == "runtime_consumption_proof_ready"
    assert section["full_video_coverage"] is True
    assert section["measured_score_custody_trusted"] is True
    assert section["blockers"] == []
    plan_row = attached["byte_price_plan"]["decision_rows"][0]
    assert plan_row["delta_nonrate_score"] == pytest.approx(-0.05)
    assert plan_row["delta_rate_score"] > 0.0
    assert plan_row["economic_decision"] in {"admit", "retrain"}
    assert plan_row["decision"] == DEMOTE
    assert "advisory_or_proxy_axis_not_promotion_authority" in plan_row["blockers"]
    assert attached["score_claim"] is False
    assert attached["ready_for_exact_eval_dispatch"] is False


def test_hinerv_archive_ladder_score_attachment_blocks_untrusted_score_rows() -> None:
    ladder = _ladder_for_score_attachment()

    attached = attach_hinerv_archive_ladder_score_rows(
        ladder,
        {
            "schema": "hinerv_full_video_mlx_score_rows.v1",
            "axis_tag": "[macOS-MLX research-signal]",
            "score_rows": [
                {"row_id": "tiny", "nonrate_score": 0.230, "num_pairs": 600},
                {"row_id": "small", "nonrate_score": 0.180, "num_pairs": 600},
            ],
        },
    )

    rows = {row["row_id"]: row for row in attached["archive_rows"]}
    assert rows["tiny"]["measured_score_custody_trusted"] is False
    assert "hinerv_archive_size_row_measured_score_untrusted" in rows["tiny"][
        "blockers"
    ]
    assert attached["score_attachment"]["matched_archive_row_count"] == 2
    assert attached["score_attachment"]["trusted_score_row_count"] == 0
    assert "hinerv_archive_size_ladder_measured_scores_untrusted" in attached[
        "blockers"
    ]
    section = attached["section_value_rows"][0]
    assert "hinerv_modelsize_increment_measured_score_untrusted" in section[
        "blockers"
    ]
    assert attached["byte_price_plan"]["decision_rows"][0]["decision"] == DEMOTE


def test_hinerv_archive_ladder_score_attachment_blocks_partial_scores() -> None:
    ladder = _ladder_for_score_attachment()

    attached = attach_hinerv_archive_ladder_score_rows(
        ladder,
        [
            {
                "row_id": "tiny",
                "d_seg": 0.001,
                "d_pose": 0.001,
                "num_pairs": 16,
                "axis_tag": "[macOS-MLX research-signal]",
            }
        ],
    )

    rows = {row["row_id"]: row for row in attached["archive_rows"]}
    assert rows["tiny"]["nonrate_score"] == pytest.approx(0.1 + (0.01 ** 0.5))
    assert "hinerv_archive_size_row_measured_score_not_full_video" in rows["tiny"][
        "blockers"
    ]
    assert "hinerv_archive_size_row_measured_score_untrusted" in rows["tiny"][
        "blockers"
    ]
    assert "hinerv_archive_size_row_measured_score_missing" in rows["small"][
        "blockers"
    ]
    assert "hinerv_archive_size_ladder_full_video_scores_incomplete" in attached[
        "blockers"
    ]
    section = attached["section_value_rows"][0]
    assert section["delta_nonrate_score"] is None
    assert "hinerv_modelsize_increment_measured_nonrate_missing" in section[
        "blockers"
    ]
    assert "hinerv_modelsize_increment_full_video_score_missing" in section[
        "blockers"
    ]
    assert "hinerv_modelsize_increment_measured_score_untrusted" in section[
        "blockers"
    ]
    plan_row = attached["byte_price_plan"]["decision_rows"][0]
    assert plan_row["decision"] == DEMOTE
    assert "delta_nonrate_score_missing" in plan_row["blockers"]


def test_hinerv_archive_ladder_score_attachment_blocks_untrusted_full_video_scores() -> None:
    ladder = _ladder_for_score_attachment()

    attached = attach_hinerv_archive_ladder_score_rows(
        ladder,
        {
            "schema": "hinerv_full_video_mlx_score_rows.v1",
            "axis_tag": "[macOS-MLX research-signal]",
            "score_rows": [
                {"row_id": "tiny", "nonrate_score": 0.230, "num_pairs": 600},
                {"row_id": "small", "nonrate_score": 0.180, "num_pairs": 600},
            ],
        },
    )

    rows = {row["row_id"]: row for row in attached["archive_rows"]}
    assert rows["tiny"]["measured_score_full_video_coverage"] is True
    assert rows["tiny"]["measured_score_custody_trusted"] is False
    assert rows["tiny"]["measured_score_trust_blockers"] == [
        "score_row_provenance_hash_missing"
    ]
    assert "hinerv_archive_size_row_measured_score_untrusted" in rows["tiny"][
        "blockers"
    ]
    assert "hinerv_archive_size_row_has_no_nonrate_score" in rows["tiny"][
        "blockers"
    ]
    assert attached["score_attachment"]["matched_archive_row_count"] == 2
    assert attached["score_attachment"]["matched_full_video_row_count"] == 0
    assert attached["score_attachment"]["trusted_score_row_count"] == 0
    assert "hinerv_archive_size_ladder_measured_scores_untrusted" in attached[
        "blockers"
    ]
    section = attached["section_value_rows"][0]
    assert section["delta_nonrate_score"] == pytest.approx(-0.05)
    assert "hinerv_modelsize_increment_measured_score_untrusted" in section[
        "blockers"
    ]


def test_attach_hinerv_archive_ladder_scores_cli_writes_artifact(tmp_path: Path) -> None:
    ladder_path = tmp_path / "ladder.json"
    score_path = tmp_path / "scores.json"
    output_path = tmp_path / "attached.json"
    ladder_path.write_text(json.dumps(_ladder_for_score_attachment()), encoding="utf-8")
    score_path.write_text(
        json.dumps(
            {
                "schema": "hinerv_full_video_mlx_score_rows.v1",
                "score_rows": [
                    {
                        "row_id": "tiny",
                        "nonrate_score": 0.23,
                        "num_pairs": 600,
                        "source_report_sha256": "a" * 64,
                    },
                    {
                        "row_id": "small",
                        "nonrate_score": 0.18,
                        "num_pairs": 600,
                        "source_report_sha256": "b" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = attach_cli.main(
        [
            "--ladder-json",
            str(ladder_path),
            "--score-json",
            str(score_path),
            "--output-json",
            str(output_path),
        ]
    )

    assert rc == 0
    attached = json.loads(output_path.read_text(encoding="utf-8"))
    assert attached["score_attachment"]["matched_archive_row_count"] == 2
    assert attached["section_value_rows"][0]["delta_nonrate_score"] == pytest.approx(
        -0.05
    )
    assert attached["score_claim"] is False


def _ladder_for_score_attachment() -> dict:
    return {
        "schema": HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
        "family": "hi_nerv",
        "axis_tag": "[planning/control]",
        "archive_rows": [
            {
                "family": "hi_nerv",
                "row_id": "tiny",
                "modelsize_scale": 0.25,
                "archive_bytes": 20_000,
                "archive_sha256": "a" * 64,
                "runtime_consumption_proof_ready": True,
                "blockers": [
                    "hinerv_archive_size_row_has_no_nonrate_score",
                    "contest_cpu_cuda_exact_eval_not_executed",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            {
                "family": "hi_nerv",
                "row_id": "small",
                "modelsize_scale": 0.5,
                "archive_bytes": 30_000,
                "archive_sha256": "b" * 64,
                "runtime_consumption_proof_ready": True,
                "blockers": [
                    "hinerv_archive_size_row_has_no_nonrate_score",
                    "contest_cpu_cuda_exact_eval_not_executed",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ],
        "blockers": [
            "hinerv_archive_size_ladder_false_authority_no_nonrate_score",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
