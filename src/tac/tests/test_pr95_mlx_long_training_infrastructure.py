# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from comma_lab.scheduler.local_training_queue import build_local_training_execution_queue
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.pr95_hnerv_mlx_long_training import (
    PR95_MLX_LONG_TRAINING_EXACT_READINESS_BLOCKERS,
    PR95_MLX_LONG_TRAINING_EXPORT_PLACEHOLDER_SCHEMA,
    PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY,
    PR95_MLX_LONG_TRAINING_FIDELITY_CLASS,
    PR95_MLX_LONG_TRAINING_FIDELITY_STATUS,
    PR95_MLX_LONG_TRAINING_PLAN_SCHEMA,
    PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS,
    PR95_MLX_LONG_TRAINING_TELEMETRY_SCHEMA,
    LongTrainingConfig,
    MLXLongTrainingPipeline,
    MLXPairIterator,
    StageHyperparameters,
    StageTelemetryRow,
    TrainingTelemetry,
    _LongTrainingBundleMLX,
    build_long_training_plan_report,
    list_initial_substrate_adapter_registry,
    register_canonical_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_long_training_provenance_is_mlx_false_authority() -> None:
    config = LongTrainingConfig(
        source_video_path=Path("upstream/videos/0.mkv"),
        smoke_mode=True,
        smoke_epochs_per_stage=1,
        checkpoint_every_epochs=1,
        operator_run_label="unit-test-plan",
    )

    provenance = register_canonical_provenance(
        config,
        source_video_sha256="a" * 64,
        source_video_frame_count=1200,
    )

    assert provenance["schema"] == "pr95_mlx_long_training_provenance.v1"
    assert provenance["evidence_grade"] == EVIDENCE_GRADE_MLX
    assert provenance["evidence_tag"] == EVIDENCE_TAG_MLX
    assert provenance["axis_tag"] == EVIDENCE_TAG_MLX
    assert provenance["score_axis"] == EVIDENCE_TAG_MLX
    assert provenance["source_video_frame_count_scope"] == "full_video_decode"
    assert provenance["max_frames"] is None
    assert provenance["training_fidelity_class"] == PR95_MLX_LONG_TRAINING_FIDELITY_CLASS
    assert provenance["training_fidelity_status"] == PR95_MLX_LONG_TRAINING_FIDELITY_STATUS
    assert (
        provenance["reproduction_equivalence_class"]
        == PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS
    )
    assert provenance["exact_readiness_refusal"]["ready"] is False
    assert provenance["exact_readiness_refusal"]["blockers"] == list(
        PR95_MLX_LONG_TRAINING_EXACT_READINESS_BLOCKERS
    )
    for key, value in PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY.items():
        assert provenance[key] is value


def test_long_training_plan_report_is_queue_observable() -> None:
    config = LongTrainingConfig(
        source_video_path=Path("upstream/videos/0.mkv"),
        telemetry_path=Path(".omx/state/pr95_mlx_long_training_test.jsonl"),
        smoke_mode=True,
        smoke_epochs_per_stage=2,
        checkpoint_every_epochs=1,
        max_frames=16,
    )

    report = build_long_training_plan_report(
        config,
        output_report_path=Path(".omx/research/pr95_mlx_long_training_test_plan.json"),
        source_video_sha256="b" * 64,
        source_video_frame_count=16,
        telemetry_path=config.telemetry_path,
        command=["tools/run_pr95_mlx_long_training.py", "--smoke-mode"],
    )

    assert report["schema"] == PR95_MLX_LONG_TRAINING_PLAN_SCHEMA
    assert report["mode"] == "plan_only"
    assert report["training_fidelity_class"] == PR95_MLX_LONG_TRAINING_FIDELITY_CLASS
    assert report["training_fidelity_status"] == PR95_MLX_LONG_TRAINING_FIDELITY_STATUS
    assert report["source_video_frame_count_scope"] == "max_frames_cap"
    assert report["max_frames"] == 16
    assert report["reproduction_equivalence"] is False
    assert report["reproduction_claim"] is False
    assert (
        report["reproduction_equivalence_class"]
        == PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS
    )
    assert report["candidate_registry_count"] >= 6
    assert report["artifact_paths"] == [config.telemetry_path.as_posix()]
    assert report["readiness_blockers"] == list(
        PR95_MLX_LONG_TRAINING_EXACT_READINESS_BLOCKERS
    )
    assert report["exact_readiness_refusal"]["ready"] is False
    postconditions = report["recommended_execution"]["extra_artifact_postconditions"]
    assert {
        "type": "json_equals",
        "path": ".omx/research/pr95_mlx_long_training_test_plan.json",
        "key": "reproduction_equivalence",
        "equals": False,
    } in postconditions
    assert {
        "type": "json_false_authority",
        "path": ".omx/research/pr95_mlx_long_training_test_plan.json",
        "required_false": sorted(PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY),
        "false_or_missing": [],
    } in postconditions
    for key, value in PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY.items():
        assert report[key] is value


def test_long_training_plan_report_marks_unknown_frame_count() -> None:
    config = LongTrainingConfig(
        source_video_path=Path("upstream/videos/0.mkv"),
        smoke_mode=True,
        smoke_epochs_per_stage=1,
        checkpoint_every_epochs=1,
    )

    report = build_long_training_plan_report(
        config,
        output_report_path=Path(".omx/research/pr95_mlx_long_training_plan.json"),
        source_video_sha256="b" * 64,
        source_video_frame_count=None,
        command=["tools/run_pr95_mlx_long_training.py"],
    )

    assert report["source_video_frame_count"] is None
    assert report["source_video_frame_count_scope"] == "not_decoded"
    assert report["canonical_provenance"]["source_video_frame_count"] is None
    assert (
        report["canonical_provenance"]["source_video_frame_count_scope"]
        == "not_decoded"
    )


def test_long_training_plan_report_compiles_to_experiment_queue(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "long_training_plan.json"
    telemetry_path = tmp_path / "telemetry.jsonl"
    config = LongTrainingConfig(
        source_video_path=Path("upstream/videos/0.mkv"),
        telemetry_path=telemetry_path,
        checkpoint_root=tmp_path / "checkpoints",
        smoke_mode=True,
        smoke_epochs_per_stage=1,
        checkpoint_every_epochs=1,
    )
    report = build_long_training_plan_report(
        config,
        output_report_path=report_path,
        telemetry_path=telemetry_path,
        command=[
            ".venv/bin/python",
            "tools/run_pr95_mlx_long_training.py",
            "--output-report",
            report_path.as_posix(),
            "--telemetry-path",
            telemetry_path.as_posix(),
            "--smoke-mode",
        ],
    )

    queue = build_local_training_execution_queue(
        [report],
        queue_id="pr95_mlx_long_training_queue_fixture",
        repo_root=REPO_ROOT,
        local_mlx_concurrency=1,
    )

    step = queue["experiments"][0]["steps"][0]
    assert queue["schema"] == "experiment_queue.v1"
    assert queue["queue_id"] == "pr95_mlx_long_training_queue_fixture"
    assert step["command"][1] == "tools/run_pr95_mlx_long_training.py"
    assert step["telemetry"]["artifact_paths"][0].endswith(
        "long_training_plan.json"
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["key"] == "training_fidelity_class"
        and condition["equals"] == PR95_MLX_LONG_TRAINING_FIDELITY_CLASS
        for condition in step["postconditions"]
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["key"] == "reproduction_equivalence"
        and condition["equals"] is False
        for condition in step["postconditions"]
    )
    assert queue["experiments"][0]["metadata"]["ready_for_exact_eval_dispatch"] is False
    assert queue["experiments"][0]["metadata"]["reproduction_equivalence"] is False


def test_long_training_telemetry_header_persists_false_authority(
    tmp_path: Path,
) -> None:
    telemetry = TrainingTelemetry(
        lane_id="lane_pr95_mlx_long_training_test",
        source_video_sha256="c" * 64,
        source_video_frame_count=2,
        source_video_frame_count_scope="max_frames_cap",
        max_frames=2,
        canonical_citation=".omx/research/pr95_8stage_curriculum_forensic_20260513.md",
        run_started_utc="2026-05-25T18:00:00Z",
    )
    telemetry.append_row(
        StageTelemetryRow(
            stage_index=1,
            stage_name="smoke",
            epoch_within_stage=1,
            global_epoch=1,
            loss=0.25,
            learning_rate=1e-4,
            batch_size=1,
            wall_clock_seconds=0.01,
            mlx_peak_memory_bytes=0,
            timestamp_utc="2026-05-25T18:00:01Z",
        )
    )
    path = tmp_path / "telemetry.jsonl"

    telemetry.persist(path)

    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert header["schema_version"] == PR95_MLX_LONG_TRAINING_TELEMETRY_SCHEMA
    assert header["source_video_frame_count_scope"] == "max_frames_cap"
    assert header["max_frames"] == 2
    assert header["evidence_grade"] == EVIDENCE_GRADE_MLX
    assert header["evidence_tag"] == EVIDENCE_TAG_MLX
    assert header["training_fidelity_class"] == PR95_MLX_LONG_TRAINING_FIDELITY_CLASS
    assert header["training_fidelity_status"] == PR95_MLX_LONG_TRAINING_FIDELITY_STATUS
    assert (
        header["reproduction_equivalence_class"]
        == PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS
    )
    assert header["row_count"] == 1
    for key, value in PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY.items():
        assert header[key] is value


def test_initial_substrate_registry_has_operator_routable_ids() -> None:
    registry = list_initial_substrate_adapter_registry()
    ids = {adapter.candidate_id for adapter in registry}

    assert "hinton_distilled_scorer_surrogate_top1" in ids
    assert "uniward_per_instance_x_wavelet_db4_probe9" in ids
    assert "pretrained_driving_prior_dp1" in ids
    assert all(adapter.as_dict()["operator_routable_summary"] for adapter in registry)


def test_mlx_pair_iterator_uses_adjacent_frame_pairs_for_pr95_latents() -> None:
    pytest.importorskip("mlx.core")

    class _FrameSource:
        video_path = Path("synthetic_4_frame_source.mkv")

        def iter_frames(self):
            for value in range(4):
                yield np.full((2, 3, 3), value, dtype=np.uint8)

    iterator = MLXPairIterator(_FrameSource(), random_seed=0)
    indices, targets = iterator.sample_batch(2)

    indices_np = np.asarray(indices)
    targets_np = np.asarray(targets)
    assert iterator.frame_count == 4
    assert iterator.pair_count == 2
    assert targets_np.shape == (2, 2, 2, 3, 3)
    for batch_index, pair_index in enumerate(indices_np):
        expected_f0 = int(pair_index) * 2 / 255.0
        expected_f1 = (int(pair_index) * 2 + 1) / 255.0
        assert targets_np[batch_index, 0, 0, 0, 0] == pytest.approx(expected_f0)
        assert targets_np[batch_index, 1, 0, 0, 0] == pytest.approx(expected_f1)


def test_long_training_step_updates_trainable_latents() -> None:
    mx = pytest.importorskip("mlx.core")
    optim = pytest.importorskip("mlx.optimizers")

    config = LongTrainingConfig(
        source_video_path=Path("upstream/videos/0.mkv"),
        base_channels=4,
        eval_size=(384, 512),
        random_seed=7,
    )
    pipeline = MLXLongTrainingPipeline(config)
    pipeline._bundle = _LongTrainingBundleMLX(
        latent_count=2,
        latent_dim=28,
        base_channels=4,
        eval_size=(384, 512),
        seed=7,
    )
    pipeline._decoder = pipeline._bundle.decoder
    pipeline._latents_full = pipeline._bundle.latents
    pipeline._optimizer = optim.Adam(learning_rate=1e-3)
    before = np.asarray(pipeline._bundle.latents).copy()

    loss = pipeline.training_step(
        mx.array([0], dtype=mx.int32),
        mx.zeros((1, 2, 384, 512, 3), dtype=mx.float32),
    )
    mx.eval(pipeline._bundle.latents)
    after = np.asarray(pipeline._bundle.latents)

    assert loss >= 0.0
    assert not np.array_equal(before, after)


def test_long_training_checkpoint_exports_trained_latents_in_pt(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mlx.core")
    pytest.importorskip("torch")

    config = LongTrainingConfig(
        source_video_path=Path("upstream/videos/0.mkv"),
        checkpoint_root=tmp_path,
        base_channels=4,
        eval_size=(384, 512),
        random_seed=9,
    )
    pipeline = MLXLongTrainingPipeline(config)
    pipeline._bundle = _LongTrainingBundleMLX(
        latent_count=2,
        latent_dim=28,
        base_channels=4,
        eval_size=(384, 512),
        seed=9,
    )
    pipeline._decoder = pipeline._bundle.decoder
    pipeline._latents_full = pipeline._bundle.latents
    pipeline._source_video_sha256 = "d" * 64
    artifact = pipeline._persist_checkpoint(
        StageHyperparameters(
            stage_index=1,
            name="unit",
            epochs=1,
            learning_rate=1e-4,
            batch_size=1,
        ),
        loss_at_checkpoint=0.125,
    )

    import torch

    loaded = torch.load(artifact.pytorch_state_dict_path, weights_only=True)
    assert artifact.trained_latents_exported is True
    assert artifact.pytorch_export_succeeded is True
    assert artifact.pytorch_export_manifest_path is not None
    assert artifact.pytorch_export_manifest_path.is_file()
    assert artifact.latents_path is not None
    assert artifact.latents_path.is_file()
    assert "latents" in loaded
    assert list(loaded["latents"].shape) == [2, 28]
    assert "blocks.0.weight" in loaded
    assert "blocks.0.conv.weight" not in loaded
    assert "refine.0.weight" in loaded
    assert "refine0.weight" not in loaded
    artifact_dict = artifact.as_dict()
    assert artifact_dict["exact_readiness_refusal"]["ready"] is False
    assert artifact_dict["reproduction_equivalence"] is False
    assert (
        artifact_dict["reproduction_equivalence_class"]
        == PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS
    )


def test_long_training_checkpoint_deferred_export_placeholder_is_false_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mlx.core")
    import tac.local_acceleration.mlx_to_pytorch_export as export_bridge

    def _raise_export(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("forced export failure")

    monkeypatch.setattr(
        export_bridge,
        "export_mlx_state_dict_to_torch_pt",
        _raise_export,
    )
    config = LongTrainingConfig(
        source_video_path=Path("upstream/videos/0.mkv"),
        checkpoint_root=tmp_path,
        base_channels=4,
        eval_size=(384, 512),
        random_seed=11,
    )
    pipeline = MLXLongTrainingPipeline(config)
    pipeline._bundle = _LongTrainingBundleMLX(
        latent_count=2,
        latent_dim=28,
        base_channels=4,
        eval_size=(384, 512),
        seed=11,
    )
    pipeline._decoder = pipeline._bundle.decoder
    pipeline._latents_full = pipeline._bundle.latents
    pipeline._source_video_sha256 = "e" * 64

    artifact = pipeline._persist_checkpoint(
        StageHyperparameters(
            stage_index=1,
            name="unit",
            epochs=1,
            learning_rate=1e-4,
            batch_size=1,
        ),
        loss_at_checkpoint=0.25,
    )

    assert artifact.pytorch_export_succeeded is False
    assert artifact.trained_latents_exported is False
    assert not artifact.pytorch_state_dict_path.exists()
    assert artifact.pytorch_export_deferred_path is not None
    placeholder = json.loads(
        artifact.pytorch_export_deferred_path.read_text(encoding="utf-8")
    )
    assert placeholder["schema"] == PR95_MLX_LONG_TRAINING_EXPORT_PLACEHOLDER_SCHEMA
    assert placeholder["placeholder_is_not_pytorch_state_dict"] is True
    assert placeholder["pytorch_export_succeeded"] is False
    assert placeholder["pytorch_state_dict_exists"] is False
    assert placeholder["ready_for_exact_eval_dispatch"] is False
    assert placeholder["reproduction_equivalence"] is False
    assert placeholder["reproduction_claim"] is False
    assert placeholder["exact_readiness_refusal"]["ready"] is False
    assert (
        placeholder["reproduction_equivalence_class"]
        == PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS
    )
    for key, value in PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY.items():
        assert placeholder[key] is value


def test_run_pr95_mlx_long_training_plan_cli_writes_report(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "long_training_plan.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_pr95_mlx_long_training.py"),
            "--output-report",
            str(report_path),
            "--source-video-path",
            "upstream/videos/0.mkv",
            "--telemetry-path",
            str(tmp_path / "telemetry.jsonl"),
            "--smoke-mode",
            "--smoke-epochs-per-stage",
            "1",
            "--checkpoint-every-epochs",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["mode"] == "plan_only"
    assert report["schema"] == PR95_MLX_LONG_TRAINING_PLAN_SCHEMA
    assert report["recommended_execution"]["tool"] == (
        "tools/run_pr95_mlx_long_training.py"
    )
    assert report["recommended_execution"]["resource_kind"] == "local_cpu"
    assert report["recommended_execution"]["authority_kind"] == (
        "macos_mlx_research_signal"
    )
    assert report["recommended_execution"]["output_manifest"] == str(report_path)
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["reproduction_equivalence"] is False
    assert report["exact_readiness_refusal"]["ready"] is False
    assert report["telemetry_path"].endswith("telemetry.jsonl")
    for key, value in PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY.items():
        assert report[key] is value


def test_run_pr95_mlx_long_training_cli_has_explicit_full_execute_mode(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "long_training_execute_plan.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_pr95_mlx_long_training.py"),
            "--output-report",
            str(report_path),
            "--source-video-path",
            "upstream/videos/0.mkv",
            "--telemetry-path",
            str(tmp_path / "telemetry.jsonl"),
            "--max-frames",
            "1200",
            "--checkpoint-every-epochs",
            "400",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    command = report["recommended_execution"]["python_command_args"]
    assert "--execute" not in command
    assert "--execute-smoke" not in command
    assert report["total_epochs"] == 3000
    assert report["recommended_execution"]["resource_kind"] == "local_cpu"

    conflict = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_pr95_mlx_long_training.py"),
            "--output-report",
            str(tmp_path / "conflict.json"),
            "--execute",
            "--execute-smoke",
            "--smoke-mode",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert conflict.returncode != 0
    assert "pass only one of --execute or --execute-smoke" in conflict.stderr

    ambiguous_smoke = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_pr95_mlx_long_training.py"),
            "--output-report",
            str(tmp_path / "ambiguous_smoke.json"),
            "--execute",
            "--smoke-mode",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert ambiguous_smoke.returncode != 0
    assert "--execute runs the configured long curriculum" in ambiguous_smoke.stderr
