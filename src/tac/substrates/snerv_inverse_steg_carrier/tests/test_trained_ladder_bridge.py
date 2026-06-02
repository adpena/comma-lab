# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_SPECTRA_PRESERVING_ADAPTER,
)
from tac.substrates.snerv_inverse_steg_carrier.trained_ladder_bridge import (
    build_snerv_trained_ladder_row_from_advisory,
)


def test_bridge_emits_strict_false_authority_row_from_real_packet(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-real-packet"
    packet_path = tmp_path / "candidate.snar"
    packet_path.write_bytes(packet)
    advisory = _advisory(packet)

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=advisory,
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        repo_root=tmp_path,
    )

    row = payload["rows"][0]
    assert payload["status"] == "trained_ladder_row_blocked"
    assert payload["archive_custody"]["computed_from_file"] is True
    assert row["archive_bytes"] == len(packet)
    assert row["archive_sha256"] == hashlib.sha256(packet).hexdigest()
    assert row["fc_dim"] == 12
    assert row["official_controls"]["fc_dim"] == 12
    assert row["official_controls"]["emb_size"] == 4
    assert row["receiver_archive_replay_verified"] is True
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "sample_pair_count_below_full600" in payload["blockers"]
    assert "required_emission_field_missing:official_controls.--modelsize" in payload[
        "blockers"
    ]
    assert (
        "required_emission_field_false:official_controls.source_faithful_stack"
        in payload["blockers"]
    )
    assert "required_emission_field_false:official_controls.mfu_enabled" in payload[
        "blockers"
    ]
    assert "required_emission_field_false:official_controls.hfr_enabled" in payload[
        "blockers"
    ]
    assert row["official_controls"]["source_faithful_stack"] is False
    assert row["official_controls"]["official_parity_status"] == (
        "blocked_official_mfu_hfr_not_implemented"
    )
    local_modelsize = row["official_controls"]["local_modelsize_analogue"]
    assert local_modelsize["schema"] == "snerv_local_modelsize_analogue.v1"
    assert local_modelsize["fc_dim"] == 12
    assert local_modelsize["emb_size"] == 4
    assert local_modelsize["adapter"] == "snerv_inverse_steg_principled_fork"
    assert local_modelsize["decoder_feature_count"] == 16
    assert local_modelsize["official_modelsize_authority"] is False
    assert "required_emission_field_missing:official_controls.fc_dim" not in payload[
        "blockers"
    ]
    assert "modelsize_or_fc_dim_missing" not in payload["blockers"]
    assert "required_emission_field_missing:qat_bits" in payload["blockers"]


def test_bridge_rejects_unknown_archive_path_kind(tmp_path: Path) -> None:
    packet_path = tmp_path / "candidate.snar"
    packet_path.write_bytes(b"packet")

    with pytest.raises(ValueError, match="archive_path_kind"):
        build_snerv_trained_ladder_row_from_advisory(
            advisory_result=_advisory(b"packet"),
            archive_path=packet_path,
            archive_path_kind="unknown",
            repo_root=tmp_path,
        )


def test_bridge_records_receiver_visible_mfu_hfr_adapter_controls(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-adapter-packet"
    packet_path = tmp_path / "adapter.snar"
    packet_path.write_bytes(packet)

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=_advisory(
            packet,
            snerv_model_size_adapter=SNERV_SPECTRA_PRESERVING_ADAPTER,
            snerv_mfu_scales=(1, 3),
            snerv_hfr_gain=0.25,
        ),
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        repo_root=tmp_path,
    )

    controls = payload["rows"][0]["official_controls"]
    assert controls["source_faithful_stack"] is False
    assert controls["adapter"] == SNERV_SPECTRA_PRESERVING_ADAPTER
    assert controls["mfu_enabled"] is True
    assert controls["hfr_enabled"] is True
    assert controls["snerv_t_enabled"] is False
    assert controls["mfu_scales"] == [1, 3]
    assert controls["hfr_gain"] == 0.25
    assert controls["official_parity_status"] == (
        "receiver_safe_mfu_hfr_temporal_adapter_present__official_oss_"
        "parity_still_required"
    )
    assert "required_emission_field_false:official_controls.mfu_enabled" not in payload[
        "blockers"
    ]
    assert "required_emission_field_false:official_controls.hfr_enabled" not in payload[
        "blockers"
    ]


def _advisory(packet: bytes, **overrides: object) -> SimpleNamespace:
    values = {
        "snerv_model_size_adapter": "snerv_inverse_steg_principled_fork",
        "snerv_mfu_scales": (),
        "snerv_hfr_gain": 0.0,
        "snerv_temporal_context": 0,
    }
    values.update(overrides)
    return SimpleNamespace(
        n_pairs=2,
        levels=1,
        wavelet="haar",
        snerv_fc_dim=12,
        snerv_emb_size=4,
        snerv_patch_radius=1,
        decoder_feature_count=16,
        receiver_archive_packet_bytes=len(packet),
        receiver_archive_sha256=hashlib.sha256(packet).hexdigest(),
        receiver_archive_replay_verified=True,
        decoder_payload_codec="mixed_magnitude_symmetric",
        linf_steps_payload_codec="snerv_step_map_coder.v1",
        hf_decoder_fit_mode="least_squares",
        hf_decoder_saliency_component="combined",
        d_seg_mean_linf=0.01,
        d_pose_mean_linf=0.04,
        score_linf=2.0,
        axis_tag="[macOS-CPU advisory]",
        **values,
    )
