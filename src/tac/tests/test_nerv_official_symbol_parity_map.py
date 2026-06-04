# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tac.analysis.nerv_official_symbol_parity_map import (
    SCHEMA,
    build_nerv_official_symbol_parity_map,
)

REPO = Path(__file__).resolve().parents[3]


def test_official_symbol_parity_map_binds_hinerv_and_snerv_without_authority() -> None:
    report = build_nerv_official_symbol_parity_map(repo_root=REPO)

    assert report["schema"] == SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["local_symbol_map_ready"] is True
    families = {row["family"]: row for row in report["family_rows"]}
    assert families["snerv"]["local_symbol_map_ready"] is True
    assert families["hi_nerv"]["local_symbol_map_ready"] is True
    rows = {row["feature_id"]: row for row in report["symbol_rows"]}
    assert {
        "snerv_modelsize_fc_dim_solver",
        "snerv_mfu_hfr_tub_official_primitives",
        "snerv_quantized_checkpoint_payload",
        "hi_nerv_core_hierarchical_renderer",
        "hi_nerv_convnext_patch_bitstream_pipeline",
        "hi_nerv_modelsize_config_family",
    } == set(rows)
    assert rows["snerv_mfu_hfr_tub_official_primitives"][
        "local_symbols_present"
    ] is True
    assert (
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
        not in rows["snerv_mfu_hfr_tub_official_primitives"]["still_blocked_by"]
    )
    assert "snerv_official_mfu_hfr_tub_source_forward_replay_missing" in rows[
        "snerv_mfu_hfr_tub_official_primitives"
    ]["still_blocked_by"]
    assert "snerv_official_modelsize_fc_dim_not_architecture_authoritative" in rows[
        "snerv_mfu_hfr_tub_official_primitives"
    ]["still_blocked_by"]
    assert rows["hi_nerv_core_hierarchical_renderer"][
        "local_symbols_present"
    ] is True
    assert "hi_nerv_tiny_forward_parity_against_oss_missing" in rows[
        "hi_nerv_core_hierarchical_renderer"
    ]["still_blocked_by"]
    assert all(row["score_claim"] is False for row in report["symbol_rows"])


def test_official_symbol_parity_map_keeps_source_pins_unverified_when_checkout_missing(
    tmp_path: Path,
) -> None:
    report = build_nerv_official_symbol_parity_map(
        repo_root=REPO,
        families=("snerv",),
        source_roots={"snerv": tmp_path / "missing_snerv_checkout"},
    )

    assert report["local_symbol_map_ready"] is True
    assert report["source_pins_verified"] is False
    assert report["score_claim"] is False
    assert all(
        row["source_marker_status"] == "source_checkout_unavailable"
        for row in report["symbol_rows"]
    )
