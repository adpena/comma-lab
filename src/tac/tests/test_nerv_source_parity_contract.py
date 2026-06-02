# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from tac.analysis.nerv_source_parity_contract import (
    build_nerv_source_parity_contract,
    render_nerv_source_parity_markdown,
    write_nerv_source_parity_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_source_parity_contract_is_false_authority_and_family_scoped() -> None:
    report = build_nerv_source_parity_contract(repo_root=REPO_ROOT)

    assert report["schema"] == "nerv_source_parity_contract.v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert set(report["families"]) == {"hi_nerv", "snerv"}
    assert report["required_for_long_training_ready"] is False
    assert report["blockers"]


def test_hinerv_generic_resize_path_is_no_longer_a_source_parity_blocker() -> None:
    report = build_nerv_source_parity_contract(repo_root=REPO_ROOT, families=("hi_nerv",))
    rows = {row["feature_id"]: row for row in report["feature_rows"]}

    assert rows["hi_nerv_generic_resolution_path"]["status"] == "implemented_or_bound"
    assert "hi_nerv_generic_resolution_path_missing" not in report["blockers"]


def test_hinerv_official_source_parity_gaps_block_long_training() -> None:
    report = build_nerv_source_parity_contract(
        repo_root=REPO_ROOT,
        families=("hi_nerv",),
    )
    rows = {row["feature_id"]: row for row in report["feature_rows"]}

    assert report["required_for_long_training_ready"] is False
    assert "hi_nerv_official_feature_grid_convnext_trilinear_missing" in report[
        "blockers"
    ]
    assert "hi_nerv_prune_quantnoise_torchac_pipeline_missing" in report["blockers"]
    assert rows["hi_nerv_official_feature_grid_convnext_trilinear"]["status"] == (
        "missing_or_partial"
    )
    assert rows["hi_nerv_official_prune_quantnoise_torchac_pipeline"]["status"] == (
        "missing_or_partial"
    )


def test_snerv_official_mfu_hfr_and_fc_dim_controls_still_block_long_training() -> None:
    report = build_nerv_source_parity_contract(repo_root=REPO_ROOT, families=("snerv",))

    assert "snerv_mfu_hfr_stride_stack_missing" in report["blockers"]
    assert "snerv_fc_dim_modelsize_control_missing" in report["blockers"]
    assert "snerv_scorer_loop_decoder_qat_missing" in report["blockers"]
    assert "snerv_lf_quant_intn_codec_missing" not in report["blockers"]
    assert "snerv_qat_receiver_codec_pricing_missing" in report["blockers"]
    assert "snerv_official_haar_mode_missing" in report["blockers"]
    assert "snerv_receiver_dependency_custody_missing" in report["blockers"]
    rows = {row["feature_id"]: row for row in report["feature_rows"]}
    missing_symbols = {
        symbol["symbol"]
        for symbol in rows["snerv_official_mfu_hfr_stride_stack"]["symbol_rows"]
        if symbol["status"] == "missing"
    }
    assert {
        "MultiResolutionFusionUnit",
        "HighFrequencyRestorer",
        "SnervTemporalExtension",
    }.issubset(missing_symbols)
    controls = {row["control_id"]: row for row in report["control_rows"]}
    assert controls["snerv_lf_stepmap_and_intN_control"]["status"] == (
        "implemented_or_declared"
    )
    assert controls["snerv_lf_stepmap_and_intN_control"]["missing_markers"] == []


def test_source_parity_contract_writes_json_and_markdown(tmp_path: Path) -> None:
    json_path = tmp_path / "source_parity.json"
    md_path = tmp_path / "source_parity.md"

    report = write_nerv_source_parity_contract(
        repo_root=REPO_ROOT,
        output_json=json_path,
        output_md=md_path,
    )

    assert json_path.is_file()
    assert md_path.is_file()
    assert "NeRV Source-Parity Contract" in md_path.read_text(encoding="utf-8")
    assert render_nerv_source_parity_markdown(report).startswith(
        "# NeRV Source-Parity Contract"
    )
