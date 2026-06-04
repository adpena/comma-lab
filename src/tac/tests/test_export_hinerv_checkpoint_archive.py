# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.analysis.nerv_modelsize_budget import build_hinerv_config_from_size_knobs
from tools.export_hinerv_checkpoint_archive import (
    _blockers,
    _export_hard_byte_ceiling,
    _modelsize_integrity_profile,
    _resolve_decoder_codec,
)


def _state() -> dict[str, np.ndarray]:
    return {
        "latents_coarse": np.zeros((2, 4), dtype=np.float32),
        "latents_mid": np.zeros((2, 8), dtype=np.float32),
        "latents_fine": np.zeros((2, 16), dtype=np.float32),
        "blocks.0.conv.weight": np.zeros((24, 3, 3, 8), dtype=np.float32),
        "feature_grids.0.grids.0": np.zeros((2, 2, 2, 4), dtype=np.float32),
        "convnext_blocks.0.dwconv.weight": np.zeros((6, 7, 7, 1), dtype=np.float32),
    }


def test_hinerv_checkpoint_decoder_codec_propagates_candidate_over_default_runner_codec() -> None:
    resolution = _resolve_decoder_codec(
        explicit_arg=None,
        command_args={"compact_decoder_codec": "portfolio_auto"},
        candidate={"decoder_codec": "int7_mixed"},
    )

    assert resolution["resolved"] == "int7_mixed"
    assert resolution["resolution_source"] == "modelsize_candidate_decoder_codec"
    assert resolution["candidate_codec_takes_precedence_over_runner_default"] is True
    assert resolution["modelsize_candidate_decoder_codec_propagates_to_export"] is True
    assert resolution["modelsize_candidate_decoder_codec_is_capacity_authority"] is True
    assert resolution["blockers"] == []


def test_hinerv_checkpoint_decoder_codec_keeps_explicit_arg_authority() -> None:
    resolution = _resolve_decoder_codec(
        explicit_arg="int4_mixed",
        command_args={"compact_decoder_codec": "portfolio_auto"},
        candidate={"decoder_codec": "int7_mixed"},
    )

    assert resolution["resolved"] == "int4_mixed"
    assert resolution["resolution_source"] == "explicit_arg"
    assert resolution["modelsize_candidate_decoder_codec"] == "int7_mixed"
    assert resolution["modelsize_candidate_decoder_codec_propagates_to_export"] is False
    assert resolution["modelsize_candidate_decoder_codec_is_capacity_authority"] is False
    assert resolution["blockers"] == ["candidate_decoder_codec_not_export_authority"]


def test_hinerv_checkpoint_export_blocks_candidate_decoder_codec_override() -> None:
    resolution = _resolve_decoder_codec(
        explicit_arg="int4_mixed",
        command_args={"compact_decoder_codec": "portfolio_auto"},
        candidate={"decoder_codec": "int7_mixed"},
    )

    blockers = _blockers(
        archive_bytes=120_000,
        hard_byte_ceilings=[178_000],
        receiver_proof={"runtime_consumption_proof_ready": True},
        receiver_proof_requested=True,
        modelsize_integrity={"blockers": []},
        decoder_codec_resolution=resolution,
    )

    assert "candidate_decoder_codec_not_export_authority" in blockers


def test_hinerv_checkpoint_export_uses_strictest_candidate_or_startup_byte_ceiling() -> None:
    assert (
        _export_hard_byte_ceiling(
            candidate={"hard_byte_ceiling": 178_000},
            hard_byte_ceilings=[216_000, 285_000],
        )
        == 178_000
    )
    assert (
        _export_hard_byte_ceiling(
            candidate={"hard_byte_ceiling": 285_000},
            hard_byte_ceilings=[216_000],
        )
        == 216_000
    )
    assert (
        _export_hard_byte_ceiling(
            candidate={},
            hard_byte_ceilings=[0, "178000"],
        )
        == 178_000
    )


def test_hinerv_checkpoint_decoder_codec_keeps_nondefault_runner_codec() -> None:
    resolution = _resolve_decoder_codec(
        explicit_arg=None,
        command_args={"compact_decoder_codec": "int8_mixed"},
        candidate={"decoder_codec": "int7_mixed"},
    )

    assert resolution["resolved"] == "int8_mixed"
    assert resolution["resolution_source"] == "runner_compact_decoder_codec"
    assert resolution["modelsize_candidate_decoder_codec_propagates_to_export"] is False
    assert resolution["modelsize_candidate_decoder_codec_is_capacity_authority"] is False
    assert resolution["blockers"] == ["candidate_decoder_codec_not_export_authority"]


def test_hinerv_modelsize_integrity_matches_candidate_controls() -> None:
    cfg = build_hinerv_config_from_size_knobs(
        num_pairs=2,
        latent_dim=8,
        embed_dim=8,
        decoder_channel=6,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
    )
    state = _state()
    total = sum(int(value.size) for value in state.values())
    profile = _modelsize_integrity_profile(
        state,
        candidate={
            "candidate_id": "hinerv_test",
            "num_pairs": 2,
            "decoder_channel": 6,
            "total_trainable_params": total,
            "modelsize_mparams": total / 1_000_000.0,
            "use_hierarchical_feature_grid": True,
            "use_convnext_blocks": True,
        },
        cfg=cfg,
    )

    assert profile["matches_candidate_controls"] is True
    assert profile["blockers"] == []
    assert profile["observed_total_trainable_params"] == total


def test_hinerv_modelsize_integrity_blocks_fake_dimension_metadata() -> None:
    cfg = build_hinerv_config_from_size_knobs(
        num_pairs=2,
        latent_dim=8,
        embed_dim=8,
        decoder_channel=6,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
    )
    profile = _modelsize_integrity_profile(
        _state(),
        candidate={
            "candidate_id": "hinerv_bad",
            "num_pairs": 2,
            "decoder_channel": 12,
            "total_trainable_params": 1,
            "modelsize_mparams": 0.5,
            "use_hierarchical_feature_grid": False,
            "use_convnext_blocks": True,
        },
        cfg=cfg,
    )

    assert profile["matches_candidate_controls"] is False
    assert "hinerv_modelsize_decoder_channel_mismatch" in profile["blockers"]
    assert "hinerv_modelsize_total_trainable_params_mismatch" in profile["blockers"]
    assert "hinerv_modelsize_mparams_metadata_mismatch" in profile["blockers"]
    assert (
        "hinerv_modelsize_unexpected_tensor_prefix_present:feature_grids."
        in profile["blockers"]
    )
