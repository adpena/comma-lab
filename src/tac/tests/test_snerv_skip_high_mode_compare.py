# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.snerv_skip_high_mode_compare import (
    build_skip_high_mode_comparison,
    render_markdown_report,
)


def _write_binary_profile(
    path: Path,
    *,
    archive_bytes: int,
    codec: str,
    stored_raw_bytes: int,
    stored_shape: list[int],
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "snerv_binary_profile.v1",
                "charged_archive_bytes": archive_bytes,
                "snar1_packet_bytes": archive_bytes - 1000,
                "section_summary": {
                    "largest_section": "decoder_payload",
                    "largest_section_bytes": archive_bytes - 2000,
                },
                "decoder_payload_header": {
                    "skip_high_storage": {
                        "schema": "snerv_official_skip_high_storage.v1",
                        "codec": codec,
                        "stored_shape": stored_shape,
                        "source_shape": [1200, 3, 192, 256],
                        "stored_raw_bytes": stored_raw_bytes,
                        "source_raw_bytes": 1_415_577_600,
                        "raw_byte_savings": 1_415_577_600 - stored_raw_bytes,
                        "receiver_expands_skip_high": True,
                        "lossless_relative_to_source_skip_high": False,
                    }
                },
                "blockers": ["snerv_binary_profile_is_rate_only_not_score_authority"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_prefilter(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "archive_bytes": 93_620,
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": "executed"},
                "score_components": {
                    "canonical_score": 90.8,
                    "seg_term": 50.4,
                    "pose_term": 40.3,
                    "rate_term": 0.06,
                },
                "scorer_input_diagnosis": {
                    "schema": "mlx_renderer_prefilter_scorer_input_diagnosis.v1",
                    "verdict": "SCORER_INPUT_OUT_OF_DISTRIBUTION",
                    "candidate_output_out_of_distribution": True,
                    "blockers": [
                        "mlx_renderer_prefilter_scorer_input_out_of_distribution"
                    ],
                },
                "blockers": ["scorer_input_segnet_last_rgb_mean_absdiff_gt_50"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_skip_high_comparison_finds_rate_vs_value_domain_crux(tmp_path: Path) -> None:
    scalar = tmp_path / "scalar.json"
    shared = tmp_path / "shared.json"
    prefilter = tmp_path / "prefilter.json"
    _write_binary_profile(
        scalar,
        archive_bytes=91_445,
        codec="scalar_mean_float64",
        stored_raw_bytes=8,
        stored_shape=[1, 1, 1, 1],
    )
    _write_binary_profile(
        shared,
        archive_bytes=436_084,
        codec="shared_mean_float64",
        stored_raw_bytes=1_179_648,
        stored_shape=[1, 3, 192, 256],
    )
    _write_prefilter(prefilter)

    payload = build_skip_high_mode_comparison(
        binary_profiles={"scalar": scalar, "shared": shared},
        prefilter_profiles={"scalar_epoch003199": prefilter},
        hard_byte_ceiling=178_000,
    )

    assert payload["verdict"] == "NO_CURRENT_SKIP_HIGH_MODE_READY_FOR_EXACT_EVAL"
    assert payload["best_rate_row"]["label"] == "scalar"
    assert payload["best_non_scalar_skip_high_row"]["label"] == "shared"
    assert "no_skip_high_mode_with_both_byte_cap_and_non_scalar_storage" in payload[
        "blockers"
    ]
    assert "skip_high_prefilter_scorer_input_out_of_distribution" in payload[
        "blockers"
    ]
    rows = {row["label"]: row for row in payload["binary_profile_rows"]}
    assert rows["scalar"]["scalar_collapse_risk"] is True
    assert rows["shared"]["scalar_collapse_risk"] is False
    assert rows["shared"]["under_hard_byte_ceiling"] is False
    assert payload["prefilter_profile_rows"][0]["local_replay_admissible"] is False

    md = render_markdown_report(payload)
    assert "rate-admissible scalar skip-high" in md
    assert "FUNDAMENTAL" not in md
