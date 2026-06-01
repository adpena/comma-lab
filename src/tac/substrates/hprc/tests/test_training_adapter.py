# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import tac.substrates.hprc.archive_candidate as hprc_archive_candidate
from tac.substrates.hprc.archive import HprcSectionKind, parse_hprc_packet
from tac.substrates.hprc.learned_receiver import (
    COMPACT_RECEIVER_MODE,
    compact_receiver_reconstruction_metrics,
    decode_compact_receiver_packet,
)
from tac.substrates.hprc.training_adapter import (
    HPRC_LONG_TRAINING_SUBSTRATE_ID,
    HprcCompactReceiverLongTrainingAdapter,
)
from tac.training.long_training_canonical import (
    CurriculumStage,
    LongTrainingConfig,
    run_long_training,
    validate_substrate_adapter,
)
from tools import run_hprc_compact_receiver_training as hprc_training_tool


def _frames() -> np.ndarray:
    y = np.arange(8, dtype=np.float32)[:, None]
    x = np.arange(10, dtype=np.float32)[None, :]
    frames = []
    for frame_index in range(6):
        frame = np.empty((8, 10, 3), dtype=np.float32)
        frame[:, :, 0] = 32 + frame_index * 4 + x
        frame[:, :, 1] = 48 + frame_index * 2 + y
        frame[:, :, 2] = 64 + frame_index + x + y
        frames.append(frame)
    return np.stack(frames, axis=0).clip(0, 255).astype(np.uint8)


def test_hprc_training_adapter_conforms_and_exports_packet(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hprc_archive_candidate, "HPRC_RECEIVER_PROOF_SCRATCH_BYTES", 1)
    adapter = HprcCompactReceiverLongTrainingAdapter(
        _frames(),
        basis_count=4,
        residual_grid_h=4,
        residual_grid_w=5,
        initial_latent_gain=0.0,
        initial_residual_gain=0.0,
        initial_receiver_state_gain=0.0,
        emit_archive_bound_candidate_package=False,
        source_manifest={"unit_test": "hprc_training_adapter"},
    )

    validate_substrate_adapter(adapter)
    batch = adapter.sample_batch(batch_size=2, seed=123)
    before = adapter.loss_fn(adapter.model, batch, {"recon": 1.0})["total"]
    for _ in range(12):
        adapter.train_step(batch, learning_rate=0.01, loss_weights={"recon": 1.0})
    after = adapter.loss_fn(adapter.model, batch, {"recon": 1.0})["total"]
    assert after < before

    archive_path, archive_sha, archive_bytes = adapter.export_archive(adapter.model, tmp_path)
    assert archive_path.is_file()
    assert len(archive_sha) == 64
    assert archive_bytes == archive_path.stat().st_size
    export_manifest = json.loads((tmp_path / "hprc_compact_receiver_training_export.json").read_text())
    assert export_manifest["receiver_proof_requested"] is False
    assert export_manifest["score_claim"] is False

    packet = parse_hprc_packet(
        (tmp_path / "hprc_compact_receiver_archive_export" / "0.bin").read_bytes()
    )
    compact = decode_compact_receiver_packet(packet)
    assert compact.manifest["hprc_receiver_mode"] == COMPACT_RECEIVER_MODE
    assert compact.manifest["trained_renderer_export_ready"] is True
    rdo = json.loads(packet.section_map()[HprcSectionKind.RDO_PLAN])
    assert rdo["output_resize"] == "bilinear"
    assert rdo["output_resize_alignment"] == "bilinear_align_corners_false"
    assert rdo["residual_grid_h"] == 4
    assert rdo["residual_grid_w"] == 5
    metrics = compact_receiver_reconstruction_metrics(compact, _frames())
    assert metrics["score_claim"] is False
    assert metrics["mse_rgb255"] < before


def test_hprc_training_adapter_runs_canonical_long_training(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hprc_archive_candidate, "HPRC_RECEIVER_PROOF_SCRATCH_BYTES", 1)
    adapter = HprcCompactReceiverLongTrainingAdapter(
        _frames(),
        basis_count=3,
        residual_grid_h=4,
        residual_grid_w=5,
        initial_latent_gain=0.0,
        initial_residual_gain=0.0,
        initial_receiver_state_gain=0.0,
        emit_archive_bound_candidate_package=False,
        source_manifest={"unit_test": "canonical_long_training"},
    )
    config = LongTrainingConfig(
        substrate_id=HPRC_LONG_TRAINING_SUBSTRATE_ID,
        lane_id="lane_hprc_compact_receiver_training_unit_20260531",
        epochs=3,
        batch_pair_indices_per_step=2,
        curriculum_stages=(
            CurriculumStage(
                name="gain_fit",
                start_epoch=0,
                end_epoch=3,
                loss_weights={"recon": 1.0},
            ),
        ),
        checkpoint_interval_epochs=1,
        early_stopping_patience=10,
        learning_rate=0.01,
        output_dir=tmp_path / "hprc_long_training",
        notes="Unit smoke for HPRC compact receiver train/export adapter.",
    )

    artifact = run_long_training(adapter, config)

    assert artifact.substrate_id == HPRC_LONG_TRAINING_SUBSTRATE_ID
    assert artifact.archive_path is not None
    assert artifact.archive_sha256 is not None
    assert artifact.archive_bytes is not None and artifact.archive_bytes > 0
    assert artifact.score_claim is False
    assert artifact.ready_for_exact_eval_dispatch is False
    assert artifact.substrate_artifact_metadata["receiver_mode"] == COMPACT_RECEIVER_MODE
    assert Path(artifact.telemetry_path).is_file()


def test_hprc_native_rate_aware_training_shrinks_unprotected_residual_tokens() -> None:
    protection = np.zeros((6, 4, 5, 3), dtype=np.float32)
    adapter = HprcCompactReceiverLongTrainingAdapter(
        _frames(),
        basis_count=3,
        residual_grid_h=4,
        residual_grid_w=5,
        initial_latent_gain=0.0,
        initial_residual_gain=0.0,
        initial_receiver_state_gain=0.0,
        native_rate_aware=True,
        rate_aware_residual_prox_weight=1.0,
        residual_protection=protection,
        emit_archive_bound_candidate_package=False,
    )
    batch = {"frame_indices": np.arange(6, dtype=np.int32)}
    adapter.model.residual[:] = 0.5
    before = float(np.mean(np.abs(adapter.model.residual)))

    metrics = adapter.train_step(
        batch,
        learning_rate=0.1,
        loss_weights={"recon": 0.0, "residual_rate_prox": 1.0},
    )

    after = float(np.mean(np.abs(adapter.model.residual)))
    assert after < before
    assert metrics["native_rate_residual_mean_abs_delta"] > 0.0


def test_hprc_native_rate_protection_blocks_residual_shrink() -> None:
    protection = np.ones((6, 4, 5, 3), dtype=np.float32)
    adapter = HprcCompactReceiverLongTrainingAdapter(
        _frames(),
        basis_count=3,
        residual_grid_h=4,
        residual_grid_w=5,
        initial_latent_gain=0.0,
        initial_residual_gain=0.0,
        initial_receiver_state_gain=0.0,
        native_rate_aware=True,
        rate_aware_residual_prox_weight=1.0,
        residual_protection=protection,
        emit_archive_bound_candidate_package=False,
    )
    batch = {"frame_indices": np.arange(6, dtype=np.int32)}
    adapter.model.residual[:] = 0.5
    before = adapter.model.residual.copy()

    adapter.train_step(
        batch,
        learning_rate=0.1,
        loss_weights={"recon": 0.0, "residual_rate_prox": 1.0},
    )

    np.testing.assert_allclose(adapter.model.residual, before)


def test_hprc_mlx_training_backend_exports_numpy_portable_packet() -> None:
    pytest.importorskip("mlx.core")
    adapter = HprcCompactReceiverLongTrainingAdapter(
        _frames(),
        basis_count=3,
        residual_grid_h=4,
        residual_grid_w=5,
        initial_latent_gain=0.0,
        initial_residual_gain=0.0,
        initial_receiver_state_gain=0.0,
        native_rate_aware=True,
        rate_aware_residual_prox_weight=0.25,
        residual_protection=np.zeros((6, 4, 5, 3), dtype=np.float32),
        emit_archive_bound_candidate_package=False,
        training_backend="mlx",
    )
    batch = {"frame_indices": np.arange(6, dtype=np.int32)}

    metrics = adapter.train_step(
        batch,
        learning_rate=0.01,
        loss_weights={"recon": 1.0, "residual_rate_prox": 0.25},
    )

    assert adapter.effective_training_backend == "mlx"
    assert metrics["loss_backend_is_mlx"] == 1.0
    packet = parse_hprc_packet(adapter.model.packet_bytes())
    compact = decode_compact_receiver_packet(packet)
    assert compact.manifest["portable_runtime"] == "numpy"
    assert compact.manifest["training_backend"]["effective_training_backend"] == "mlx"
    assert compact.manifest["training_backend"]["contest_runtime_requires_mlx"] is False
    assert compact.rdo_plan["training_backend"]["portable_runtime"] == "numpy"


def test_hprc_gain_l2_gradient_matches_reweighted_loss() -> None:
    adapter = HprcCompactReceiverLongTrainingAdapter(
        _frames(),
        basis_count=3,
        residual_grid_h=4,
        residual_grid_w=5,
        initial_latent_gain=0.2,
        initial_residual_gain=0.3,
        initial_receiver_state_gain=0.1,
        emit_archive_bound_candidate_package=False,
    )
    batch = {"frame_indices": np.arange(6, dtype=np.int32)}
    weights = {"recon": 0.25, "gain_l2": 0.5}
    analytic = adapter.loss_fn(adapter.model, batch, weights)["latent_gain_grad"]
    original = float(adapter.model.latent_gain)
    eps = 1e-2
    adapter.model.latent_gain = original + eps
    plus = adapter.loss_fn(adapter.model, batch, weights)["total"]
    adapter.model.latent_gain = original - eps
    minus = adapter.loss_fn(adapter.model, batch, weights)["total"]
    adapter.model.latent_gain = original
    numeric = (plus - minus) / (2.0 * eps)

    assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-4)


def test_hprc_training_cli_materializes_storage_custody_result(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(hprc_archive_candidate, "HPRC_RECEIVER_PROOF_SCRATCH_BYTES", 1)
    frames_path = tmp_path / "frames.npy"
    np.save(frames_path, _frames())

    exit_code = hprc_training_tool.main(
        [
            "--frames-npy",
            frames_path.as_posix(),
            "--output-dir",
            (tmp_path / "hprc_cli_out").as_posix(),
            "--epochs",
            "2",
            "--batch-pair-indices-per-step",
            "2",
            "--learning-rate",
            "0.01",
            "--training-backend",
            "numpy",
            "--skip-runtime-consumption-proof",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == hprc_training_tool.HPRC_LONG_TRAINING_RESULT_SCHEMA
    assert payload["runtime_consumption_proof_requested"] is False
    assert payload["training_backend"]["effective"] == "numpy"
    assert payload["artifact"]["archive_path"]
    assert Path(payload["result_path"]).is_file()
    assert payload["score_claim"] is False


def test_hprc_training_cli_consumes_native_rate_protection_surface(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(hprc_archive_candidate, "HPRC_RECEIVER_PROOF_SCRATCH_BYTES", 1)
    frames_path = tmp_path / "frames.npy"
    protection_path = tmp_path / "protection.npy"
    np.save(frames_path, _frames())
    np.save(protection_path, np.zeros((6, 4, 5, 3), dtype=np.float32))

    exit_code = hprc_training_tool.main(
        [
            "--frames-npy",
            frames_path.as_posix(),
            "--output-dir",
            (tmp_path / "hprc_cli_rate_out").as_posix(),
            "--epochs",
            "1",
            "--batch-pair-indices-per-step",
            "3",
            "--learning-rate",
            "0.01",
            "--training-backend",
            "numpy",
            "--residual-grid-h",
            "4",
            "--residual-grid-w",
            "5",
            "--native-rate-aware",
            "--rate-aware-residual-prox-weight",
            "0.5",
            "--rate-aware-residual-protection-npy",
            protection_path.as_posix(),
            "--skip-runtime-consumption-proof",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["residual_protection_manifest"]["shape"] == [6, 4, 5, 3]
    artifact_metadata = payload["artifact"]["substrate_artifact_metadata"]
    assert artifact_metadata["native_rate_aware_training"]["enabled"] is True
    assert artifact_metadata["native_rate_aware_training"]["residual_protection_present"] is True
    assert artifact_metadata["training_backend"]["portable_runtime"] == "numpy"


def test_hprc_training_cli_decodes_real_video_source_via_canonical_helper(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    monkeypatch.setattr(hprc_archive_candidate, "HPRC_RECEIVER_PROOF_SCRATCH_BYTES", 1)
    video_path = tmp_path / "0.mkv"
    video_path.write_bytes(b"fake-video")

    def fake_decode_real_pairs(video_path_arg, **kwargs):
        assert Path(video_path_arg) == video_path
        assert kwargs["substrate_tag"] == "hprc_compact_receiver"
        values = torch.arange(2 * 2 * 3 * 4 * 5, dtype=torch.float32)
        return values.reshape(2, 2, 3, 4, 5)

    monkeypatch.setattr(hprc_training_tool, "decode_real_pairs", fake_decode_real_pairs)

    exit_code = hprc_training_tool.main(
        [
            "--video-path",
            video_path.as_posix(),
            "--decode-pairs",
            "2",
            "--decode-height",
            "4",
            "--decode-width",
            "5",
            "--output-dir",
            (tmp_path / "hprc_video_cli_out").as_posix(),
            "--output-manifest",
            (tmp_path / "hprc_video_cli_out" / "result.json").as_posix(),
            "--epochs",
            "1",
            "--batch-pair-indices-per-step",
            "2",
            "--training-backend",
            "numpy",
            "--skip-runtime-consumption-proof",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_manifest"]["source_kind"] == "contest_video_decode"
    assert payload["source_manifest"]["decoded_frame_count"] == 4
    assert payload["source_manifest"]["frames_shape"] == [4, 4, 5, 3]
    assert Path(payload["result_path"]).name == "result.json"
    assert payload["score_claim"] is False
