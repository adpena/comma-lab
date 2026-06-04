# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import numpy as np

from tac.analysis.nerv_modelsize_budget import build_hinerv_config_from_size_knobs
from tools import export_hinerv_checkpoint_archive as export_mod
from tools.export_hinerv_checkpoint_archive import (
    _blockers,
    _export_hard_byte_ceiling,
    _hinerv_source_pair_indices,
    _maybe_write_receiver_raw_cache_mlx_prefilter,
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


def test_hinerv_checkpoint_export_writes_report_before_mlx_prefilter(
    tmp_path,
    monkeypatch,
) -> None:
    startup = tmp_path / "startup.json"
    state_path = tmp_path / "state.npsd"
    meta = tmp_path / "checkpoint.meta.json"
    output_json = tmp_path / "export" / "hinerv_checkpoint_archive_export.json"
    state_path.write_bytes(b"state")
    candidate = {
        "candidate_id": "hinerv_pending_report_test",
        "num_pairs": 2,
        "latent_dim": 8,
        "embed_dim": 8,
        "decoder_channel": 6,
        "use_hierarchical_feature_grid": True,
        "use_convnext_blocks": True,
        "hard_byte_ceiling": 178_000,
        "nominal_total_payload_bytes": 120_000,
        "decoder_codec": "int7_mixed",
    }
    startup.write_text(
        json.dumps(
            {
                "modelsize_candidate": candidate,
                "command_args": {"num_pairs": 2, "compact_decoder_codec": "portfolio_auto"},
                "hard_byte_ceilings": [178_000],
            }
        ),
        encoding="utf-8",
    )
    meta.write_text(
        json.dumps(
            {
                "global_epoch": 7,
                "ema_shadow_state_path": state_path.as_posix(),
                "live_state_path": state_path.as_posix(),
            }
        ),
        encoding="utf-8",
    )

    class FakeAdapter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def import_state_dict(self, *_args, **_kwargs) -> None:
            pass

    def fake_export(_model, out_dir, **_kwargs):
        out = out_dir
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        archive.write_bytes(b"archive")
        proof_dir = out / "receiver_proof"
        proof_dir.mkdir(parents=True, exist_ok=True)
        proof = proof_dir / "hi_nerv_mlx_receiver_proof.json"
        proof.write_text(
            json.dumps(
                {
                    "runtime_consumption_proof_ready": True,
                    "receiver_archive_replay_verified": True,
                    "blockers": [],
                }
            ),
            encoding="utf-8",
        )
        return archive, export_mod.sha256_file(archive), archive.stat().st_size

    def fake_prefilter(**_kwargs):
        pending = json.loads(output_json.read_text(encoding="utf-8"))
        assert pending["report_status"] == (
            "archive_receiver_proof_written_prefilter_pending"
        )
        assert pending["archive_bytes"] == len(b"archive")
        assert pending["receiver_proof_ready"] is True
        assert pending["local_mlx_prefilter_written"] is False
        return {
            "schema": "hinerv_checkpoint_receiver_raw_mlx_prefilter_request.v1",
            "written": True,
            "profile_path": (tmp_path / "prefilter.json").as_posix(),
            "blockers": [],
            **export_mod.FALSE_AUTHORITY,
        }

    monkeypatch.setattr(export_mod, "HinervSubstrateMLX", lambda _cfg: object())
    monkeypatch.setattr(export_mod, "MlxScoreAwareAdapter", FakeAdapter)
    monkeypatch.setattr(export_mod, "unpack_state_dict_numpy", lambda _path: _state())
    monkeypatch.setattr(export_mod, "export_hi_nerv_mlx_archive", fake_export)
    monkeypatch.setattr(
        export_mod,
        "_maybe_write_receiver_raw_cache_mlx_prefilter",
        fake_prefilter,
    )

    report = export_mod.export_checkpoint_archive(
        startup_json=startup,
        checkpoint_meta=meta,
        output_dir=tmp_path / "export",
        output_json=output_json,
        emit_receiver_proof=True,
        retain_receiver_proof_output=True,
        write_mlx_prefilter_profile=True,
    )

    final = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["report_status"] == "complete"
    assert final["report_status"] == "complete"
    assert final["local_mlx_prefilter_written"] is True


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


def test_hinerv_checkpoint_export_records_measurement_bypass_for_over_cap() -> None:
    blockers = _blockers(
        archive_bytes=214_187,
        hard_byte_ceiling=178_000,
        hard_byte_ceilings=[178_000, 216_000],
        receiver_proof={"runtime_consumption_proof_ready": True},
        receiver_proof_requested=True,
        modelsize_integrity={"blockers": []},
        decoder_codec_resolution={"blockers": []},
        hard_byte_ceiling_measurement_bypass_enabled=True,
    )

    assert "archive_bytes_exceed_tightest_hard_ceiling" in blockers
    assert "hard_byte_ceiling_export_bypassed_for_measurement" in blockers


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


def test_hinerv_checkpoint_prefilter_not_requested_is_false_authority_blocker(
    tmp_path,
) -> None:
    profile = _maybe_write_receiver_raw_cache_mlx_prefilter(
        requested=False,
        output_dir=tmp_path,
        receiver_proof={},
        startup={},
        command_args={},
        candidate={"num_pairs": 2},
        archive_bytes=123,
        archive_sha256="a" * 64,
        source_video_path=None,
        scorer_upstream_dir=None,
        scorer_device="cpu",
        scorer_batch_pairs=1,
        progress_every=0,
        repo_root=tmp_path,
    )

    assert profile["written"] is False
    assert profile["blockers"] == ["hinerv_checkpoint_mlx_prefilter_not_requested"]
    assert profile["score_claim"] is False


def test_hinerv_checkpoint_prefilter_normalizes_non_singleton_batch_pairs(
    tmp_path,
) -> None:
    profile = _maybe_write_receiver_raw_cache_mlx_prefilter(
        requested=False,
        output_dir=tmp_path,
        receiver_proof={},
        startup={},
        command_args={},
        candidate={"num_pairs": 2},
        archive_bytes=123,
        archive_sha256="a" * 64,
        source_video_path=None,
        scorer_upstream_dir=None,
        scorer_device="cpu",
        scorer_batch_pairs=4,
        progress_every=0,
        repo_root=tmp_path,
    )

    assert profile["scorer_batch_pairs_requested"] == 4
    assert profile["scorer_batch_pairs_effective"] == 1
    assert profile["scorer_batch_pairs_normalized_to_singleton"] is True
    assert (
        profile["scorer_batch_pairs_normalization"]["reason"]
        == "production_mlx_scorer_response_uses_singleton_batches_after_recorded_segnet_batch_shape_drift"
    )


def test_hinerv_checkpoint_prefilter_requires_prefix_source_pair_indices(
    tmp_path,
) -> None:
    raw = tmp_path / "candidate.raw"
    raw.write_bytes(b"raw")
    profile = _maybe_write_receiver_raw_cache_mlx_prefilter(
        requested=True,
        output_dir=tmp_path / "prefilter",
        receiver_proof={
            "receiver_output_path": raw.as_posix(),
            "receiver_output_retained": True,
        },
        startup={},
        command_args={"prioritized_pair_indices": "2,4"},
        candidate={"num_pairs": 2},
        archive_bytes=123,
        archive_sha256="a" * 64,
        source_video_path=None,
        scorer_upstream_dir=None,
        scorer_device="cpu",
        scorer_batch_pairs=1,
        progress_every=0,
        repo_root=tmp_path,
    )

    assert profile["written"] is False
    assert profile["source_pair_indices"] == [2, 4]
    assert profile["blockers"] == [
        "hinerv_checkpoint_mlx_prefilter_requires_prefix_source_pair_indices"
    ]


def test_hinerv_checkpoint_source_pair_indices_default_to_prefix() -> None:
    assert _hinerv_source_pair_indices(candidate={"num_pairs": 3}, command_args={}) == (
        0,
        1,
        2,
    )
    assert _hinerv_source_pair_indices(
        candidate={},
        command_args={"prioritized_pair_indices": "0,1"},
    ) == (0, 1)


def test_hinerv_checkpoint_blockers_switch_when_mlx_prefilter_written() -> None:
    blockers = _blockers(
        archive_bytes=120_000,
        hard_byte_ceilings=[178_000],
        receiver_proof={"runtime_consumption_proof_ready": True},
        receiver_proof_requested=True,
        modelsize_integrity={"blockers": []},
        decoder_codec_resolution={"blockers": []},
        mlx_prefilter_profile={
            "written": True,
            "blockers": ["mlx_local_replay_not_contest_auth_axis"],
        },
    )

    assert "full_video_scorer_replay_not_executed" not in blockers
    assert "mlx_local_replay_not_contest_auth_axis" in blockers


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
