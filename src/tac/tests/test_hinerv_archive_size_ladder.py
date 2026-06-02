# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from comma_lab.storage_tiers import StorageTierError
from tac.analysis.hinerv_archive_size_ladder import (
    HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
    build_hinerv_archive_size_ladder,
    hinerv_modelsize_increment_section_value_rows,
    render_hinerv_archive_size_ladder_markdown,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    DEMOTE,
    NERV_BYTE_PRICE_CONTROLLER_SCHEMA,
    build_nerv_byte_price_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

try:
    import mlx.core  # noqa: F401

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX required for archive export")
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
    assert "durable_evidence_on_selected_storage" in report[
        "artifact_retention_policy"
    ]
    assert "waterfill_group_bits_against_fixed_contest_byte_price" in report[
        "required_allocator_bindings"
    ]
    assert "receiver_proof_not_executed_for_archive_size_ladder" in report["blockers"]
    row = report["archive_rows"][0]
    assert row["row_id"] == "hi_nerv_local_tiny"
    assert row["archive_bytes"] == Path(row["archive_path"]).stat().st_size
    assert len(row["archive_sha256"]) == 64
    assert row["archive_rate_score_at_contest_price"] > 0.0
    assert row["spine_manifest_path"] is not None
    assert row["state_npz_manifest_path"] is not None
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
    assert "hi_nerv_local_tiny" in markdown


def test_hinerv_archive_size_ladder_reports_missing_requested_row(tmp_path: Path) -> None:
    if not _MLX_AVAILABLE:
        pytest.skip("MLX required for archive export")
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
