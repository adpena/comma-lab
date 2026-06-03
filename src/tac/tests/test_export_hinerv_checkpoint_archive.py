# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.analysis.nerv_modelsize_budget import build_hinerv_config_from_size_knobs
from tools.export_hinerv_checkpoint_archive import (
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


def test_hinerv_checkpoint_decoder_codec_prefers_candidate_over_runner_default() -> None:
    resolution = _resolve_decoder_codec(
        explicit_arg=None,
        command_args={"compact_decoder_codec": "portfolio_auto"},
        candidate={"decoder_codec": "int7_mixed"},
    )

    assert resolution["resolved"] == "int7_mixed"
    assert resolution["resolution_source"] == "modelsize_candidate_decoder_codec"
    assert resolution["candidate_codec_takes_precedence_over_runner_default"] is True


def test_hinerv_checkpoint_decoder_codec_keeps_explicit_arg_authority() -> None:
    resolution = _resolve_decoder_codec(
        explicit_arg="int4_mixed",
        command_args={"compact_decoder_codec": "portfolio_auto"},
        candidate={"decoder_codec": "int7_mixed"},
    )

    assert resolution["resolved"] == "int4_mixed"
    assert resolution["resolution_source"] == "explicit_arg"


def test_hinerv_checkpoint_decoder_codec_keeps_nondefault_runner_codec() -> None:
    resolution = _resolve_decoder_codec(
        explicit_arg=None,
        command_args={"compact_decoder_codec": "int8_mixed"},
        candidate={"decoder_codec": "int7_mixed"},
    )

    assert resolution["resolved"] == "int8_mixed"
    assert resolution["resolution_source"] == "runner_compact_decoder_codec"


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
