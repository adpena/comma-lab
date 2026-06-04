# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.analysis.hinerv_official_control_parity_guard import (
    PRUNE_QUANT_TORCHAC_CONTROL_ID,
    SCHEMA,
    HiNervOfficialControlParityGuardError,
    build_hinerv_official_control_parity_guard,
    main,
    require_hinerv_official_control_parity,
)


def test_guard_blocks_local_bitstream_roundtrip_without_official_torchac() -> None:
    report = build_hinerv_official_control_parity_guard(
        source_audit_report=_source_audit(component_proven=False),
        local_bitstream_report=_local_bitstream_roundtrip(),
    )

    row = report["control_rows"][0]
    assert report["schema"] == SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert row["local_receiver_control_real"] is True
    assert row["local_entropy_torchac_encode_decode_bound"] is False
    assert row["official_control_parity_proven"] is False
    assert "hinerv_prune_quant_codec_source_forward_replay_missing" in row[
        "blockers"
    ]
    assert "hinerv_official_torchac_encode_decode_not_bound" in row["blockers"]
    assert "hinerv_official_torchac_byte_streams_not_present" in row["blockers"]
    assert report["official_control_parity_proven"] is False


def test_guard_rejects_boolean_only_prune_quant_component() -> None:
    report = build_hinerv_official_control_parity_guard(
        source_audit_report=_source_audit(component_proven=True),
        forward_parity_artifact={
            "schema": "hinerv_official_forward_parity.v1",
            "official_forward_parity_passed": True,
            "component_rows": [
                {
                    "component_id": "prune_quant_codec",
                    "source_forward_parity_proven": True,
                }
            ],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    blockers = report["control_rows"][0]["blockers"]
    assert "numeric_max_abs_error_missing:prune_quant_codec" in blockers
    assert "official_output_sha256_missing:prune_quant_codec" in blockers
    assert "hinerv_official_torchac_encode_decode_not_bound" in blockers
    assert "hinerv_official_torchac_stream_sha256_missing" in blockers


def test_guard_accepts_hash_backed_official_torchac_replay() -> None:
    source_audit = _source_audit(component_proven=True)
    source_audit["component_state_rows"][0]["forward_parity_artifact_component"] = (
        _official_torchac_component()
    )

    report = build_hinerv_official_control_parity_guard(
        source_audit_report=source_audit,
        forward_parity_artifact={
            "schema": "hinerv_official_forward_parity.v1",
            "official_forward_parity_passed": True,
            "component_rows": [_official_torchac_component()],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    row = report["control_rows"][0]
    assert row["official_torchac_encode_decode_bound"] is True
    assert row["official_torchac_byte_streams_present"] is True
    assert row["official_torchac_roundtrip_max_abs_error"] == 0.0
    assert row["blockers"] == []
    assert row["official_control_parity_proven"] is True
    assert report["official_control_parity_proven"] is True
    assert report["blockers"] == []
    assert report["source_faithful_stack_claim"] is False


def test_require_guard_raises_with_exact_blocker_names() -> None:
    with pytest.raises(
        HiNervOfficialControlParityGuardError,
        match="hinerv_official_torchac_encode_decode_not_bound",
    ):
        require_hinerv_official_control_parity(
            source_audit_report=_source_audit(component_proven=False),
            local_bitstream_report=_local_bitstream_roundtrip(),
        )


def test_guard_can_load_json_inputs_from_paths(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    artifact_path = tmp_path / "artifact.json"
    source_path.write_text(
        json.dumps(_source_audit(component_proven=True), sort_keys=True),
        encoding="utf-8",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "schema": "hinerv_official_forward_parity.v1",
                "component_rows": [_official_torchac_component()],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = build_hinerv_official_control_parity_guard(
        source_audit_report_path=source_path,
        forward_parity_artifact_path=artifact_path,
        claimed_control_ids=(PRUNE_QUANT_TORCHAC_CONTROL_ID,),
    )

    assert report["blockers"] == []
    assert report["control_rows"][0]["official_control_parity_proven"] is True


def test_guard_cli_emits_report_and_require_blocks(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    local_path = tmp_path / "local.json"
    out_path = tmp_path / "guard.json"
    source_path.write_text(
        json.dumps(_source_audit(component_proven=False), sort_keys=True),
        encoding="utf-8",
    )
    local_path.write_text(
        json.dumps(_local_bitstream_roundtrip(), sort_keys=True),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--source-audit-report",
            str(source_path),
            "--local-bitstream-report",
            str(local_path),
            "--out",
            str(out_path),
            "--require",
        ]
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["schema"] == SCHEMA
    assert payload["score_claim"] is False
    assert "hinerv_official_torchac_encode_decode_not_bound" in payload[
        "blockers"
    ]


def _source_audit(*, component_proven: bool) -> dict[str, object]:
    return {
        "schema": "hinerv_official_source_parity_audit.v1",
        "authority": "false_authority_source_audit_no_score_claim",
        "official_marker_group_rows": [
            {
                "group_id": "official_quant_prune_torchac_bitstream",
                "all_markers_present": True,
            }
        ],
        "component_state_rows": [
            {
                "component_id": "prune_quant_codec",
                "source_forward_parity_proven": component_proven,
                "source_forward_parity_falsified": not component_proven,
                "blockers": []
                if component_proven
                else ["hinerv_prune_quant_codec_source_forward_replay_missing"],
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _local_bitstream_roundtrip() -> dict[str, object]:
    return {
        "schema": "hi_nerv_bitstream_roundtrip_measurement.v1",
        "proof": "receiver_visible_prune_quantnoise_decoder_codec_roundtrip_v1",
        "rows": [
            {
                "decoder_codec_requested": "int8_mixed",
                "blob_bytes": 128,
                "roundtrip_error": {"max_abs_error": 0.0},
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "official_entropy_receiver_consumption": {
            "schema": "hi_nerv_official_entropy_receiver_consumption.v1",
            "torchac_encode_decode_bound": False,
            "torchac_byte_streams_present": False,
            "blockers": [
                "hinerv_official_torchac_encode_decode_not_bound",
                "hinerv_official_torchac_cdf_streams_not_serialized",
            ],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "blockers": [
            "hi_nerv_bitstream_roundtrip_is_local_rate_distortion_evidence_only",
            "hinerv_official_torchac_encode_decode_not_bound",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _official_torchac_component() -> dict[str, object]:
    return {
        "component_id": "prune_quant_codec",
        "source_forward_parity_proven": True,
        "max_abs_error": 0.0,
        "tolerance": 1.0e-6,
        "input_sha256": "1" * 64,
        "official_output_sha256": "2" * 64,
        "portable_output_sha256": "2" * 64,
        "official_weight_sha256": "3" * 64,
        "official_torchac_encode_decode_bound": True,
        "torchac_byte_streams_present": True,
        "official_torchac_stream_sha256": "4" * 64,
        "torchac_roundtrip_max_abs_error": 0.0,
        "torchac_roundtrip_tolerance": 1.0e-6,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
