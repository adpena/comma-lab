# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.analysis.hinerv_archive_ladder_waterfill import (
    HINERV_ARCHIVE_LADDER_WATERFILL_SCHEMA,
    HinervArchiveLadderWaterfillError,
    build_hinerv_archive_ladder_waterfill,
)
from tac.repo_io import sha256_file
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    NERV_BYTE_PRICE_CONTROLLER_SCHEMA,
)
from tools.build_hinerv_archive_ladder_waterfill import (
    _global_saliency,
    _report_blockers,
    _row_blockers,
)
from tools.build_hinerv_archive_ladder_waterfill import (
    main as tool_main,
)


def test_hinerv_archive_ladder_waterfill_consumes_state_npz_manifest(
    tmp_path: Path,
) -> None:
    ladder = _ladder(tmp_path, row_id="tiny", saliency_ready=True)

    report = build_hinerv_archive_ladder_waterfill(
        ladder,
        global_saliency_by_name={"blocks.0.weight": 0.0},
        action_bits=(0, 2, 32),
        candidate_id="candidate_a",
    )

    assert report["schema"] == HINERV_ARCHIVE_LADDER_WATERFILL_SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["row_count"] == 1
    assert report["rows"][0]["row_id"] == "tiny"
    assert report["rows"][0]["waterfill_summary"]["group_count"] == 1
    assert report["rows"][0]["waterfill_summary"]["total_selected_byte_delta"] < 0
    assert report["section_value_rows"][0]["archive_ladder_row_id"] == "tiny"
    assert report["byte_price_plan"]["schema"] == NERV_BYTE_PRICE_CONTROLLER_SCHEMA
    assert "contest_cpu_cuda_exact_eval_not_executed" in report["blockers"]
    assert (
        "decoder_weight_saliency_json_path_missing_for_replay_command"
        in report["rows"][0]["blockers"]
    )
    assert report["rows"][0]["archive_ladder_replay_command_argv"] is None


def test_hinerv_archive_ladder_waterfill_fails_closed_on_bad_manifest_sha(
    tmp_path: Path,
) -> None:
    ladder = _ladder(tmp_path, row_id="tiny", saliency_ready=True)
    manifest_path = Path(ladder["archive_rows"][0]["state_npz_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = build_hinerv_archive_ladder_waterfill(ladder)

    assert report["rows"][0]["waterfill_plan"] is None
    assert "state_npz_artifact_sha256_mismatch" in report["rows"][0]["blockers"]
    assert "state_npz_artifact_sha256_mismatch" in report["blockers"]
    assert report["section_value_rows"] == []


def test_hinerv_archive_ladder_waterfill_carries_saliency_replay_blockers(
    tmp_path: Path,
) -> None:
    ladder = _ladder(tmp_path, row_id="tiny", saliency_ready=True)

    report = build_hinerv_archive_ladder_waterfill(
        ladder,
        global_saliency_by_name={"blocks.0.weight": 0.0},
        saliency_report_blockers=("full_video_coverage_missing",),
        saliency_row_blockers_by_id={
            "tiny": ("score_loss_proxy_outside_allocator_linearization_basin",),
        },
        action_bits=(0, 2, 32),
        candidate_id="candidate_a",
    )

    row_blockers = report["rows"][0]["blockers"]
    assert "decoder_weight_saliency_replay_has_blockers" in report["blockers"]
    assert "full_video_coverage_missing" in report["blockers"]
    assert "full_video_coverage_missing" in row_blockers
    assert "score_loss_proxy_outside_allocator_linearization_basin" in row_blockers
    assert (
        "decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin"
        in row_blockers
    )


def test_hinerv_archive_ladder_waterfill_rejects_wrong_source_schema(
    tmp_path: Path,
) -> None:
    ladder = _ladder(tmp_path, row_id="tiny", saliency_ready=False)
    ladder["schema"] = "wrong"

    with pytest.raises(
        HinervArchiveLadderWaterfillError,
        match=r"expected hinerv_archive_size_ladder\.v1",
    ):
        build_hinerv_archive_ladder_waterfill(ladder)


def test_build_hinerv_archive_ladder_waterfill_cli_smoke(tmp_path: Path) -> None:
    ladder = _ladder(tmp_path, row_id="tiny", saliency_ready=True)
    ladder_path = tmp_path / "ladder.json"
    saliency_path = tmp_path / "saliency.json"
    output_json = tmp_path / "waterfill.json"
    output_md = tmp_path / "waterfill.md"
    ladder_path.write_text(json.dumps(ladder), encoding="utf-8")
    saliency_path.write_text(
        json.dumps({"saliency_by_row": {"tiny": {"blocks.0.weight": 0.0}}}),
        encoding="utf-8",
    )

    rc = tool_main(
        [
            "--archive-ladder-json",
            str(ladder_path),
            "--saliency-json",
            str(saliency_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--action-bits",
            "0,2,32",
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["row_count"] == 1
    assert payload["section_value_rows"][0]["archive_ladder_row_id"] == "tiny"
    row = payload["rows"][0]
    assert row["archive_ladder_replay_command_axis_tag"] == (
        "[planning/control:false-authority]"
    )
    assert row["archive_ladder_replay_output_dir"] == (
        "/Volumes/VertigoDataTier/pact/hinerv_archive_ladder_waterfill_replay/tiny"
    )
    assert row["archive_ladder_replay_command_argv"] == [
        ".venv/bin/python",
        "tools/build_hinerv_archive_size_ladder.py",
        "--output-dir",
        "/Volumes/VertigoDataTier/pact/hinerv_archive_ladder_waterfill_replay/tiny",
        "--output-json",
        ".omx/research/hinerv_archive_size_ladder_replay_tiny_false_authority.json",
        "--output-md",
        ".omx/research/hinerv_archive_size_ladder_replay_tiny_false_authority.md",
        "--num-pairs",
        "600",
        "--row-id",
        "tiny",
        "--decoder-codec",
        "int8_mixed",
        "--emit-receiver-proof",
        "--emit-decoder-weight-waterfill-plan",
        "--decoder-weight-saliency-json",
        str(saliency_path),
        "--decoder-weight-waterfill-action-bits",
        "0,2,32",
    ]
    assert row["archive_ladder_replay_command_hint"] == " ".join(
        row["archive_ladder_replay_command_argv"]
    )
    assert "HiNeRV archive ladder decoder waterfill" in output_md.read_text(
        encoding="utf-8"
    )


def test_hinerv_archive_ladder_waterfill_keeps_row_saliency_row_scoped() -> None:
    payload = {
        "saliency_by_row": {
            "hi_nerv_local_tiny": {"head_rgb_1.weight": 0.25},
        },
        "saliency_by_name": {"head_rgb_1.weight": 0.25},
    }

    assert _global_saliency(payload) == {}


def test_hinerv_archive_ladder_waterfill_cli_extracts_saliency_blockers() -> None:
    payload = {
        "blockers": ["full_video_coverage_missing"],
        "rows": [
            {
                "row_id": "hi_nerv_local_tiny",
                "blockers": [
                    "score_loss_proxy_outside_allocator_linearization_basin"
                ],
            }
        ],
    }

    assert _report_blockers(payload) == ("full_video_coverage_missing",)
    assert _row_blockers(payload) == {
        "hi_nerv_local_tiny": (
            "score_loss_proxy_outside_allocator_linearization_basin",
        )
    }


def _ladder(tmp_path: Path, *, row_id: str, saliency_ready: bool) -> dict:
    state_path = tmp_path / f"{row_id}.npz"
    np.savez(
        state_path,
        **{
            "blocks.0.weight": np.asarray([0.25, -0.5, 1.0], dtype=np.float32),
            "latents_coarse": np.asarray([999.0], dtype=np.float32),
        },
    )
    manifest_path = tmp_path / f"{row_id}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "framework_agnostic_npz_bridge_manifest.v1",
                "artifact_path": str(state_path),
                "artifact_sha256": sha256_file(state_path),
                "tensor_count": 2,
                "consumption_recommended": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    return {
        "schema": "hinerv_archive_size_ladder.v1",
        "family": "hi_nerv",
        "axis_tag": "[planning/control]",
        "num_pairs": 600 if saliency_ready else 1,
        "report_path": str(tmp_path / "source_ladder.json"),
        "archive_rows": [
            {
                "row_id": row_id,
                "archive_bytes": 1234,
                "archive_sha256": "a" * 64,
                "state_npz_manifest_path": str(manifest_path),
                "runtime_consumption_proof_ready": True,
                "decoder_codec": "int8_mixed",
            }
        ],
        "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
