# SPDX-License-Identifier: MIT
"""Tests for the MLX-first compact renderer spine runner."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import signal
import struct
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.analysis.snerv_step_map_coder import encode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_SPECTRA_PRESERVING_ADAPTER,
    HfGenerationDecoder,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.run_compact_renderer_mlx_spine_runner as runner_mod  # noqa: E402
from tools.run_compact_renderer_mlx_spine_runner import (  # noqa: E402
    COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
    CompactRendererMlxSpineRunnerError,
    _parse_args,
    _require_scorer_upstream_dir_for_distillation,
    _resolve_execute_modelsize_candidate,
    _resolve_source_video_path,
    adapt_pr95_mlx_report_to_spine,
    adapt_pr95_stage8_report_to_spine,
    build_plan_only_report,
    execute_hi_nerv_mlx_scoreaware_and_adapt,
    execute_pact_nerv_selector_v4_mlx_smoke_and_adapt,
    execute_pr95_hnerv_mlx_scoreaware_and_adapt,
    execute_snerv_inverse_steg_advisory_and_adapt,
)

try:
    import mlx.core as _mx  # noqa: F401

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

try:
    import av as _av  # noqa: F401

    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False


def _write_synthetic_pr95_archive(path: Path, *, pairs: int = 600) -> Path:
    chunks = []
    for payload in (f'{{"pairs":{pairs}}}'.encode(), b"decoder", b"latents"):
        chunks.append(struct.pack("<I", len(payload)))
        chunks.append(payload)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", b"".join(chunks))
    return path


def _write_hinerv_receiver_proof(
    path: Path,
    *,
    archive_bytes: int,
    archive_sha256: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_mlx_generated_receiver_proof.v1",
                "runtime_consumption_proof_ready": True,
                "receiver_archive_replay_verified": True,
                "archive_bytes": int(archive_bytes),
                "archive_sha256": archive_sha256,
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _synthetic_snerv_packet(*, pairs: int = 2) -> bytes:
    plane_count = int(pairs) * 2 * 3
    lf_planes = [
        (np.arange(48, dtype=np.int64).reshape(6, 8) + idx) % 17
        for idx in range(plane_count)
    ]
    step_maps = [
        np.full((6, 8), 1.0 + idx * 0.01, dtype=np.float32)
        for idx in range(plane_count)
    ]
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(
            lf_zero_points=np.zeros(plane_count, dtype=np.float32),
        ),
        lf_payload=encode_lf_quant_payload(lf_planes),
        decoder_payload=encode_decoder_payload(HfGenerationDecoder.zeros(levels=2)),
        step_map_packet=encode_step_maps(step_maps, bins=4).packet,
        metadata={
            "n_pairs": int(pairs),
            "frames_per_pair": 2,
            "channels": 3,
            "height": 12,
            "width": 16,
            "lf_plane_count": plane_count,
            "levels": 2,
            "wavelet": "db2",
        },
    )
    return archive.packet


def test_write_decoder_weight_saliency_artifact_for_waterfill(tmp_path: Path) -> None:
    artifact = {
        "substrate_artifact_metadata": {
            "decoder_weight_gradient_saliency": {
                "schema": "mlx_decoder_weight_gradient_saliency.v1",
                "row_count": 1,
                "rows": [
                    {
                        "group_name": "decoder.blocks.0.weight",
                        "saliency": 3.25,
                        "sample_count": 2,
                        "numel": 16,
                    }
                ],
                "saliency_by_name": {"decoder.blocks.0.weight": 3.25},
                "blockers": [],
                "authority": "macos_mlx_research_signal_false_authority",
            }
        }
    }

    out = runner_mod._write_decoder_weight_saliency_artifact(
        artifact_dict=artifact,
        output_dir=tmp_path / "hi_nerv_mlx_training",
        family="hi_nerv",
    )

    path = Path(out["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert out["written"] is True
    assert out["row_count"] == 1
    assert out["sha256"] == runner_mod._sha256_file(path)
    assert payload["artifact_schema"] == (
        "compact_runner_decoder_weight_saliency_artifact.v1"
    )
    assert payload["schema"] == "mlx_decoder_weight_gradient_saliency.v1"
    assert payload["family"] == "hi_nerv"
    assert payload["rows"][0]["group_name"] == "decoder.blocks.0.weight"


def test_compact_scoreaware_stage_loss_weights_feed_curriculum() -> None:
    weights = runner_mod._compact_scoreaware_stage_loss_weights(
        recon=0.25,
        segnet=2.0,
        pose=1.5,
    )
    stages = runner_mod._compact_scoreaware_curriculum_stages(
        substrate_id="unit_hi_nerv",
        epochs=9,
        loss_weights=weights,
    )

    assert weights == {"recon": 0.25, "distill": 2.0, "pose_distill": 1.5}
    assert len(stages) == 1
    assert stages[0].start_epoch == 0
    assert stages[0].end_epoch == 9
    assert dict(stages[0].loss_weights) == weights
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="finite"):
        runner_mod._compact_scoreaware_stage_loss_weights(
            recon=-0.1,
            segnet=1.0,
            pose=1.0,
        )


def test_compact_family_interrupted_report_preserves_false_authority_evidence(
    tmp_path: Path,
) -> None:
    out = tmp_path / "snerv_run"
    train_dir = out / "snerv_mlx_native_export"
    train_dir.mkdir(parents=True)
    startup = out / runner_mod.COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    telemetry = train_dir / "telemetry.jsonl"
    startup.write_text('{"schema":"compact_carrier_startup_marker.v1"}\n')
    telemetry.write_text('{"epoch":0,"loss":1.0}\n')
    source_video = tmp_path / "0.mkv"
    source_video.write_bytes(b"fake video")
    args = _parse_args(
        [
            "--execute-family",
            "snerv",
            "--planner-row-id",
            "snerv::candidate::optimizer",
            "--output-dir",
            out.as_posix(),
            "--source-video-path",
            source_video.as_posix(),
            "--allow-manual-compact-family-launch",
        ]
    )

    report = runner_mod._write_compact_family_interrupted_report(
        output_dir=out,
        args=args,
        source_video_path=source_video,
        hard_byte_ceilings=(178_000, 216_000, 285_000),
        modelsize_candidate={"candidate_id": "snerv_test", "nominal_bytes": 123},
        signum=signal.SIGTERM,
        reason="unit_test",
    )

    report_path = Path(report["report_path"])
    payload = json.loads(report_path.read_text())
    assert payload["mode"] == "interrupted_compact_family_run"
    assert payload["signal_name"] == "SIGTERM"
    assert payload["training_executed"] is False
    assert payload["training_started"] is True
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "snerv_training_interrupted_before_export" in payload["blockers"]
    evidence_paths = {row["path"] for row in payload["evidence_files"]}
    assert startup.as_posix() in evidence_paths
    assert telemetry.as_posix() in evidence_paths
    assert all("sha256" in row for row in payload["evidence_files"])
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_recover_interrupted_report_from_startup_marker_summarizes_telemetry(
    tmp_path: Path,
) -> None:
    out = tmp_path / "hi_nerv_run"
    train_dir = out / "hi_nerv_mlx_training"
    train_dir.mkdir(parents=True)
    startup = out / runner_mod.COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    startup.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_startup_marker.v1",
                "pid": 123,
                "execute_family": "hi_nerv",
                "planner_row_id": "hi_nerv::hinerv_np600::adamw",
                "modelsize_candidate_id": "hinerv_np600",
                "modelsize_candidate": {"candidate_id": "hinerv_np600"},
                "output_dir": out.as_posix(),
                "source_video_path": "/Volumes/VertigoDataTier/pact/upstream/videos/0.mkv",
                "hard_byte_ceilings": [178_000, 285_000],
                "command_args": {"execute_family": "hi_nerv"},
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    telemetry = train_dir / "telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            [
                json.dumps({"epoch": 0, "loss": 9.0}, sort_keys=True),
                json.dumps(
                    {
                        "captured_at_utc": "2026-06-03T06:19:19Z",
                        "epoch": 26805,
                        "learning_rate": 2.7e-5,
                        "loss": 25.21,
                        "loss_components": {
                            "loss_part_distill": 5.64,
                            "loss_part_pose_distill": 2.52,
                            "loss_part_pr95_c1a_entropy": 6.68,
                            "pr95_stage_index": 8.0,
                            "pr95_stage_uses_muon": 1.0,
                        },
                        "per_axis_decomposition": {
                            "pose": 2.28,
                            "seg": 5.65,
                        },
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_dir = train_dir / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_meta = checkpoint_dir / "epoch026805_20260603T061919Z.meta.json"
    checkpoint_live = checkpoint_dir / "epoch026805_20260603T061919Z.live.state.npsd"
    checkpoint_ema = (
        checkpoint_dir / "epoch026805_20260603T061919Z.ema_shadow.state.npsd"
    )
    checkpoint_meta.write_text(
        json.dumps(
            {
                "schema_version": "long_training_canonical_checkpoint.v1",
                "global_epoch": 26805,
                "live_state_path": checkpoint_live.as_posix(),
                "ema_shadow_state_path": checkpoint_ema.as_posix(),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_live.write_bytes(b"live-state")
    checkpoint_ema.write_bytes(b"ema-state")

    report = runner_mod._write_compact_family_interrupted_report_from_startup_marker(
        output_dir=out,
        reason="unit_test_recovery",
    )

    payload = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    telemetry_summary = payload["telemetry_summary"]
    assert payload["mode"] == "recovered_interrupted_compact_family_run"
    assert payload["recovered"] is True
    assert payload["execute_family"] == "hi_nerv"
    assert payload["training_started"] is True
    assert payload["training_executed"] is True
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "hi_nerv_training_interrupted_before_export" in payload["blockers"]
    assert telemetry_summary["row_count"] == 2
    assert telemetry_summary["last_epoch"] == 26805
    assert telemetry_summary["last_loss_components"]["pr95_stage_index"] == 8.0
    assert telemetry_summary["last_loss_components"]["pr95_stage_uses_muon"] == 1.0
    assert telemetry_summary["last_per_axis_decomposition"]["pose"] == 2.28
    evidence_paths = {row["path"] for row in payload["evidence_files"]}
    assert startup.as_posix() in evidence_paths
    assert telemetry.as_posix() in evidence_paths
    assert checkpoint_meta.as_posix() in evidence_paths
    assert checkpoint_live.as_posix() in evidence_paths
    assert checkpoint_ema.as_posix() in evidence_paths


def test_write_decoder_weight_saliency_artifact_missing_fails_closed(
    tmp_path: Path,
) -> None:
    out = runner_mod._write_decoder_weight_saliency_artifact(
        artifact_dict={},
        output_dir=tmp_path / "hi_nerv_mlx_training",
        family="hi_nerv",
    )

    assert out["written"] is False
    assert out["reason"] == "decoder_weight_gradient_saliency_missing"
    assert out["authority"] == "macos_mlx_research_signal_false_authority"


def _write_mlx_prefilter_profile(
    path: Path,
    *,
    pairs: int,
    batch_pairs: int,
    score: float,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "producer": "tac.local_acceleration.mlx_renderer_prefilter_profile",
                "max_pairs": int(pairs),
                "num_pairs": int(pairs),
                "n_samples": int(pairs),
                "scorer_batch_pairs": int(batch_pairs),
                "scope_status": {"full_video": "executed"},
                "score_components": {"canonical_score": float(score)},
                "mlx_response_summary": {
                    "batch_pairs": int(batch_pairs),
                    "max_pairs": int(pairs),
                    "n_samples": int(pairs),
                },
                "section_value_rows": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_verified_joint_recon_weight(
    root: Path,
    *,
    pairs: int,
    name: str,
) -> tuple[Path, Path]:
    out = root / "experiments" / "results" / name
    out.mkdir(parents=True)
    weight = np.ones((pairs, 2, 384, 512, 1), dtype=np.float32)
    weight_path = out / "joint_p18_p19_recon_pixel_weight.npz"
    np.savez_compressed(weight_path, weight=weight)
    sha = runner_mod._sha256_file(weight_path)
    manifest_path = out / "joint_p18_p19_recon_pixel_weight_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "joint_p18_p19_recon_pixel_weight_manifest.v1",
                "weight_path": weight_path.as_posix(),
                "weight_sha256": sha,
                "config": {
                    "num_pairs": pairs,
                    "scorer_hw": [384, 512],
                },
                "metadata": {
                    "schema": "joint_p18_p19_recon_pixel_weight.v1",
                    "blockers": [],
                    "training_consumption_recommended": True,
                    "gradient_health": {
                        "schema": "joint_recon_pixel_weight_gradient_health.v1",
                        "status": "pass_finite",
                        "component_count": 1,
                        "components_with_nonfinite": 0,
                        "total_nonfinite_values": 0,
                        "consumption_recommended": True,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return weight_path, manifest_path


def test_planner_row_launch_gate_rejects_manual_hinerv_without_row_id(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        execute_family="hi_nerv",
        planner_row_id="",
        allow_manual_compact_family_launch=False,
    )

    blockers = runner_mod._planner_row_launch_blockers(args)
    assert blockers == [
        "hi_nerv_planner_row_id_missing",
        (
            "top_priority_compact_carrier_launch_must_come_from_"
            "nerv_long_training_campaign_plan"
        ),
    ]

    report = runner_mod._write_planner_row_launch_refusal(
        output_dir=tmp_path,
        args=args,
        blockers=blockers,
        hard_byte_ceilings=(178_000, 216_000, 285_000),
        repo_root=REPO_ROOT,
    )
    payload = json.loads(
        Path(report["report_path"]).read_text(encoding="utf-8")
    )

    assert payload["mode"] == "compact_carrier_planner_row_launch_refused"
    assert payload["trainer_launch_allowed"] is False
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["planner_launch_contract"]["planner_row_id"] is None
    assert payload["planner_launch_contract"]["allow_manual_compact_family_launch"] is False
    assert payload["blockers"] == blockers


def _write_planner_row_queue_artifact(
    path: Path,
    *,
    schema: str = "experiment_queue.v1",
    family: str = "hi_nerv",
    row_id: str = "hi_nerv::candidate::adamw",
    status: str = "queued",
    blocked: bool = False,
    runnable_contract: bool = True,
    command_extra: list[str] | None = None,
) -> Path:
    payload = {
        "schema": schema,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "experiments": [
            {
                "id": "unit_planner_row",
                "family": family,
                "status": status,
                "blocked": blocked,
                "launch_authority_contract": {
                    "schema": "nerv_long_training_queue_launch_authority_contract.v1",
                    "queue_status_is_local_mlx_plan": True,
                    "queue_status_is_runnable_plan": runnable_contract,
                    "queue_launch_blockers": []
                    if runnable_contract
                    else ["unit_not_runnable"],
                    "queue_status_is_receiver_proof": False,
                    "queue_status_is_cpu_replay_proof": False,
                    "queue_status_is_exact_eval_authority": False,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "steps": [
                    {
                        "id": "run_mlx_first_campaign_row",
                        "command": [
                            "python",
                            "tools/run_compact_renderer_mlx_spine_runner.py",
                            "--execute-family",
                            family,
                            "--planner-row-id",
                            row_id,
                            *(command_extra or []),
                        ],
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_planner_row_launch_gate_requires_queue_artifact_for_planner_row() -> None:
    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="hi_nerv",
            planner_row_id="hi_nerv::candidate::adamw",
            planner_row_queue_artifact=[],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            repo_root=REPO_ROOT,
        )
    )

    assert "hi_nerv_planner_row_queue_artifact_missing" in blockers
    assert "planner_row_queue_artifact_required_for_planner_row_launch" in blockers
    assert "bounded_planner_row_timing_smoke_waiver_missing" in blockers


def test_planner_row_launch_gate_allows_queued_runnable_artifact(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(tmp_path / "queue.json")
    args = SimpleNamespace(
        execute_family="hi_nerv",
        planner_row_id="hi_nerv::candidate::adamw",
        planner_row_queue_artifact=[queue_path],
        allow_bounded_planner_row_timing_smoke_waiver=False,
        allow_manual_compact_family_launch=False,
        num_pairs=600,
        epochs=29650,
        repo_root=REPO_ROOT,
    )

    assert runner_mod._planner_row_launch_blockers(args) == []
    guard = runner_mod._planner_row_launch_guard(args)
    record = guard["queue_artifact_status"]["artifact_records"][0]
    assert record["bytes"] == queue_path.stat().st_size
    assert record["sha256"] == runner_mod._sha256_file(queue_path)


def test_planner_row_launch_gate_rejects_fake_queue_schema(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(
        tmp_path / "queue.json",
        schema="fake_queue.v1",
    )

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="hi_nerv",
            planner_row_id="hi_nerv::candidate::adamw",
            planner_row_queue_artifact=[queue_path],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            repo_root=REPO_ROOT,
        )
    )

    assert "planner_row_queue_artifact_schema_not_allowed" in blockers
    assert "hi_nerv_planner_row_id_not_found_in_queue_artifact" in blockers


def test_planner_row_launch_gate_rejects_stale_command_controls(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(
        tmp_path / "queue.json",
        command_extra=[
            "--num-pairs",
            "600",
            "--epochs",
            "29650",
            "--modelsize-candidate-id",
            "queued-candidate",
        ],
    )

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="hi_nerv",
            planner_row_id="hi_nerv::candidate::adamw",
            planner_row_queue_artifact=[queue_path],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=16,
            modelsize_candidate_id="different-candidate",
            repo_root=REPO_ROOT,
        )
    )

    assert "planner_row_command_mismatch:--epochs" in blockers
    assert "planner_row_command_mismatch:--modelsize-candidate-id" in blockers


def test_planner_row_launch_gate_tracks_snerv_official_modelsize_controls(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(
        tmp_path / "queue.json",
        family="snerv",
        row_id="snerv::candidate::native_rate_aware_training",
        command_extra=[
            "--num-pairs",
            "600",
            "--epochs",
            "29650",
            "--modelsize-candidate-id",
            "auto",
            "--snerv-official-modelsize-mparams",
            "0.05",
            "--snerv-modelsize-control-profile",
            "official_cli_default",
            "--snerv-official-enc-strds",
            "1,2,2",
            "--snerv-official-dec-strds",
            "2,2,1",
        ],
    )

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="snerv",
            planner_row_id="snerv::candidate::native_rate_aware_training",
            planner_row_queue_artifact=[queue_path],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            modelsize_candidate_id="auto",
            snerv_official_modelsize_mparams=[0.07],
            snerv_modelsize_control_profile="contest_receiver_profile",
            snerv_official_enc_strds=(1, 3, 2),
            snerv_official_dec_strds=(2, 2, 1),
            repo_root=REPO_ROOT,
        )
    )

    assert "planner_row_command_mismatch:--snerv-official-modelsize-mparams" in blockers
    assert "planner_row_command_mismatch:--snerv-modelsize-control-profile" in blockers
    assert "planner_row_command_mismatch:--snerv-official-enc-strds" in blockers
    assert "planner_row_command_mismatch:--snerv-official-dec-strds" not in blockers


def test_planner_row_launch_gate_accepts_numeric_equivalent_snerv_hfr_gain(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(
        tmp_path / "queue.json",
        family="snerv",
        row_id="snerv::candidate::native_rate_aware_training",
        command_extra=[
            "--num-pairs",
            "600",
            "--epochs",
            "29650",
            "--modelsize-candidate-id",
            "candidate",
            "--snerv-hfr-gain",
            "0",
        ],
    )

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="snerv",
            planner_row_id="snerv::candidate::native_rate_aware_training",
            planner_row_queue_artifact=[queue_path],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            modelsize_candidate_id="candidate",
            snerv_hfr_gain=0.0,
            repo_root=REPO_ROOT,
        )
    )

    assert "planner_row_command_mismatch:--snerv-hfr-gain" not in blockers


def test_planner_row_launch_gate_rejects_nonrunnable_queue_artifact(
    tmp_path: Path,
) -> None:
    queue_path = _write_planner_row_queue_artifact(
        tmp_path / "queue.json",
        status="disabled",
        blocked=True,
        runnable_contract=False,
    )

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="hi_nerv",
            planner_row_id="hi_nerv::candidate::adamw",
            planner_row_queue_artifact=[queue_path],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            repo_root=REPO_ROOT,
        )
    )

    assert "hi_nerv_planner_row_queue_artifact_not_queued_or_runnable" in blockers
    assert "bounded_planner_row_timing_smoke_waiver_missing" in blockers


def test_planner_row_launch_gate_allows_only_bounded_timing_smoke_waiver() -> None:
    assert runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="snerv",
            planner_row_id="snerv::candidate::adamw",
            planner_row_queue_artifact=[],
            allow_bounded_planner_row_timing_smoke_waiver=True,
            allow_manual_compact_family_launch=False,
            num_pairs=2,
            epochs=1,
            repo_root=REPO_ROOT,
        )
    ) == []

    blockers = runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="snerv",
            planner_row_id="snerv::candidate::adamw",
            planner_row_queue_artifact=[],
            allow_bounded_planner_row_timing_smoke_waiver=True,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            repo_root=REPO_ROOT,
        )
    )
    assert "snerv_bounded_timing_smoke_waiver_exceeds_limits" in blockers


def test_planner_row_launch_gate_allows_explicit_manual_without_row() -> None:
    assert runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="snerv",
            planner_row_id="",
            planner_row_queue_artifact=[],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=True,
            num_pairs=600,
            epochs=29650,
            repo_root=REPO_ROOT,
        )
    ) == []


def test_planner_row_launch_gate_allows_non_required_family() -> None:
    assert runner_mod._planner_row_launch_blockers(
        SimpleNamespace(
            execute_family="pr95_hnerv",
            planner_row_id="",
            planner_row_queue_artifact=[],
            allow_bounded_planner_row_timing_smoke_waiver=False,
            allow_manual_compact_family_launch=False,
            num_pairs=600,
            epochs=29650,
            repo_root=REPO_ROOT,
        )
    ) == []


def test_compact_family_startup_marker_records_mlx_custody(
    tmp_path: Path,
    monkeypatch,
) -> None:
    weight_path = tmp_path.parent / "joint_p18_p19_recon_pixel_weight.npz"
    weight_path.write_bytes(b"joint-weight")

    def fake_discover_joint_recon_pixel_weight_path(*, repo_root, num_pairs):
        return weight_path, {
            "schema": "compact_auto_joint_recon_pixel_weight_discovery.v1",
            "num_pairs": int(num_pairs),
        }

    monkeypatch.setattr(
        runner_mod,
        "_discover_joint_recon_pixel_weight_path",
        fake_discover_joint_recon_pixel_weight_path,
    )
    args = SimpleNamespace(
        execute_family="hi_nerv",
        planner_row_id="hi_nerv::candidate::adamw",
        modelsize_candidate_id="candidate",
        target_modelsize_mparams=[],
        hinerv_target_modelsize_mparams=[],
        snerv_official_modelsize_mparams=[],
        auto_joint_recon_pixel_weight=True,
        distillation_device="mps",
        requested_distillation_device="gpu",
        mlx_prefilter_scorer_device="gpu",
        mlx_prefilter_scorer_batch_pairs=8,
        mlx_prefilter_progress_every=10,
        num_pairs=600,
        output_dir=tmp_path,
        repo_root=Path("/repo"),
    )

    path = runner_mod._write_compact_family_startup_marker(
        output_dir=tmp_path,
        args=args,
        source_video_path=Path("/Volumes/VertigoDataTier/pact/source/0.mkv"),
        hard_byte_ceilings=(178_000, 216_000),
        modelsize_candidate={"candidate_id": "candidate", "num_pairs": 600},
    )
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == "compact_carrier_startup_marker.v1"
    assert payload["execute_family"] == "hi_nerv"
    assert payload["planner_row_id"] == "hi_nerv::candidate::adamw"
    assert payload["distillation_device"] == "mps"
    assert payload["requested_distillation_device"] == "gpu"
    assert payload["mlx_prefilter_scorer_device"] == "gpu"
    assert payload["mlx_prefilter_scorer_batch_pairs"] == 8
    assert payload["auto_joint_recon_pixel_weight_path"] == weight_path.as_posix()
    assert payload["auto_joint_recon_pixel_weight_sha256"] == runner_mod._sha256_file(
        weight_path
    )
    assert payload["auto_joint_recon_pixel_weight_error"] is None
    assert payload["modelsize_target_binding"]["schema"] == (
        "compact_startup_modelsize_target_binding.v1"
    )
    assert payload["modelsize_target_binding"]["inverse_target_requested"] is False
    assert payload["modelsize_target_binding"]["selected_from_inverse_target"] is False
    assert payload["byte_cap_binding"]["schema"] == (
        "compact_startup_byte_cap_binding.v1"
    )
    assert payload["byte_cap_binding"]["hard_byte_cap_requested"] is True
    assert payload["byte_cap_binding"]["tightest_hard_byte_ceiling"] == 178_000
    assert payload["byte_cap_binding"]["authority_surface"] == (
        "measured_archive_zip_bytes_after_receiver_export"
    )
    assert "byte_cap_requires_measured_archive_zip_export" in payload[
        "byte_cap_binding"
    ]["blockers"]
    assert payload["campaign_identity"]["auto_joint_recon_pixel_weight_path"] == (
        weight_path.as_posix()
    )
    assert payload["score_claim"] is False
    assert payload["frontier_score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["false_authority_flags"] == [
        "macos_mlx_research_signal_until_archive_receiver_and_exact_eval",
        "startup_marker_before_trained_export_or_full_video_replay",
    ]
    assert (
        runner_mod._has_disallowed_existing_output_artifacts(
            tmp_path,
            allow_startup_marker_only=True,
        )
        is False
    )

    (tmp_path / "foreign_artifact.json").write_text("{}", encoding="utf-8")
    assert (
        runner_mod._has_disallowed_existing_output_artifacts(
            tmp_path,
            allow_startup_marker_only=True,
        )
        is True
    )


def test_compact_family_startup_marker_records_inverse_modelsize_target(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        execute_family="hi_nerv",
        planner_row_id="hi_nerv::targeted::adamw",
        modelsize_candidate_id="auto",
        target_modelsize_mparams=[0.178],
        hinerv_target_modelsize_mparams=[],
        snerv_official_modelsize_mparams=[],
        auto_joint_recon_pixel_weight=False,
        distillation_device="gpu",
        requested_distillation_device="gpu",
        mlx_prefilter_scorer_device="gpu",
        mlx_prefilter_scorer_batch_pairs=4,
        mlx_prefilter_progress_every=50,
        num_pairs=600,
        output_dir=tmp_path,
        repo_root=Path("/repo"),
    )

    path = runner_mod._write_compact_family_startup_marker(
        output_dir=tmp_path,
        args=args,
        source_video_path=Path("/Volumes/VertigoDataTier/pact/source/0.mkv"),
        hard_byte_ceilings=(178_000,),
        modelsize_candidate={
            "candidate_id": "hinerv_np600_target",
            "capacity_source": "local_hinerv_target_modelsize",
            "target_modelsize_mparams": 0.178,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 164_000,
            "byte_headroom": 14_000,
            "nominal_under_ceiling": True,
            "modelsize_control_contract": {
                "control_semantics": "local_receiver_visible_grid_search_nearest_target",
                "shared_target_modelsize_mparams_consumed_as": (
                    "nearest_local_param_count_target"
                ),
                "modelsize_mparams_is_official_upstream_flag": False,
                "modelsize_mparams_caps_archive_zip_bytes": False,
                "archive_bytes_authority_required": True,
            },
        },
    )

    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    binding = payload["modelsize_target_binding"]
    assert binding["inverse_target_requested"] is True
    assert binding["selected_from_inverse_target"] is True
    assert binding["effective_requested_targets_for_family"] == [0.178]
    assert binding["selected_capacity_source"] == "local_hinerv_target_modelsize"
    assert binding["control_semantics"] == (
        "local_receiver_visible_grid_search_nearest_target"
    )
    assert binding["blockers"] == []
    byte_binding = payload["byte_cap_binding"]
    assert byte_binding["hard_byte_cap_requested"] is True
    assert byte_binding["tightest_hard_byte_ceiling"] == 178_000
    assert byte_binding["selected_candidate_hard_byte_ceiling"] == 178_000
    assert byte_binding["selected_candidate_nominal_total_payload_bytes"] == 164_000
    assert byte_binding["nominal_headroom_against_tightest_hard_ceiling"] == 14_000
    assert byte_binding["selected_candidate_nominal_under_tightest_hard_ceiling"] is True
    assert byte_binding["selected_candidate_nominal_under_own_hard_ceiling"] is True
    assert byte_binding["calibrated_predicted_archive_bytes"] is None
    assert byte_binding["modelsize_mparams_caps_archive_zip_bytes"] is False
    assert byte_binding["archive_bytes_authority_required"] is True
    assert byte_binding["blockers"] == [
        "byte_cap_requires_measured_archive_zip_export"
    ]


def test_compact_family_interrupted_report_preserves_false_authority_custody(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "hi_nerv_mlx_training" / "telemetry.jsonl"
    telemetry.parent.mkdir()
    telemetry.write_text('{"epoch":0,"loss":1.0}\n', encoding="utf-8")
    startup = tmp_path / runner_mod.COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    startup.write_text(
        '{"schema":"compact_carrier_startup_marker.v1"}\n',
        encoding="utf-8",
    )
    args = SimpleNamespace(
        execute_family="hi_nerv",
        planner_row_id="hi_nerv::candidate::adamw",
        modelsize_candidate_id="candidate",
        allow_duplicate_campaign=False,
        output_dir=tmp_path,
        overwrite=False,
        repo_root=Path("/repo"),
        num_pairs=600,
        auto_joint_recon_pixel_weight=False,
        distillation_device="mps",
        mlx_prefilter_scorer_device="gpu",
    )

    report = runner_mod._write_compact_family_interrupted_report(
        output_dir=tmp_path,
        args=args,
        source_video_path=Path("/Volumes/VertigoDataTier/pact/source/0.mkv"),
        hard_byte_ceilings=(178_000,),
        modelsize_candidate={"candidate_id": "candidate", "num_pairs": 600},
        signum=15,
        reason="unit_test_signal",
    )

    assert report["schema"] == COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA
    assert report["mode"] == "interrupted_compact_family_run"
    assert report["signal_name"] == "SIGTERM"
    assert report["interruption_reason"] == "unit_test_signal"
    assert report["execute_family"] == "hi_nerv"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "hi_nerv_training_interrupted_before_export" in report["blockers"]
    evidence_by_name = {
        Path(row["path"]).name: row for row in report["evidence_files"]
    }
    assert "telemetry.jsonl" in evidence_by_name
    assert evidence_by_name["telemetry.jsonl"]["sha256"] == runner_mod._sha256_file(
        telemetry
    )
    report_path = tmp_path / "compact_renderer_mlx_spine_runner_report.json"
    assert report_path.is_file()


def test_hi_nerv_source_faithfulness_classifies_local_adaptation() -> None:
    report = runner_mod._hi_nerv_source_faithfulness_report(
        cfg=SimpleNamespace(
            use_hierarchical_feature_grid=False,
            use_convnext_blocks=False,
        ),
        decoder_codec="portfolio_auto",
    )

    assert report["schema"] == "hi_nerv_source_faithfulness.v1"
    assert report["classification"] == "local_hiv1_adaptation_not_official_hinerv"
    assert report["source_faithful_official_hinerv"] is False
    assert report["local_hiv1_adaptation"] is True
    assert "hinerv_official_hierarchical_feature_grid_not_enabled" in report[
        "blockers"
    ]
    assert "hinerv_pr95_pr101_latent_delta_brotli_codec_missing" in report[
        "blockers"
    ]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_hi_nerv_source_faithfulness_separates_official_controls_from_pr95_gaps() -> None:
    report = runner_mod._hi_nerv_source_faithfulness_report(
        cfg=SimpleNamespace(
            use_hierarchical_feature_grid=True,
            use_convnext_blocks=True,
            local_grid_levels=2,
            local_grid_channels=4,
            convnext_mlp_ratio=2,
            convnext_kernel_size=3,
        ),
        decoder_codec="portfolio_auto",
    )

    assert report["classification"] == (
        "official_hinerv_control_candidate_source_parity_bound_"
        "pr95_better_gaps"
    )
    assert report["official_hinerv_control"] is True
    assert report["source_faithful_official_hinerv"] is False
    assert report["official_source_parity_proof_required"] is True
    assert report["official_source_parity_proof_attached"] is True
    assert report["local_hiv1_adaptation"] is True
    assert report["official_hinerv_blockers"] == []
    assert report["source_parity_blockers"] == []
    assert report["source_parity_binding"]["required_for_long_training_ready"] is True
    assert report["source_parity_binding"]["feature_statuses"][
        "hi_nerv_official_patch_index_path"
    ] == "implemented_or_bound"
    assert "hinerv_pr95_pixelshuffle_bilinear_skip_refine_path_missing" in report[
        "pr95_better_blockers"
    ]
    assert report["local_grid_levels"] == 2
    assert report["convnext_kernel_size"] == 3
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_hi_nerv_source_faithfulness_metadata_strips_authority_keys() -> None:
    metadata = runner_mod._hi_nerv_source_faithfulness_metadata(
        cfg=SimpleNamespace(
            use_hierarchical_feature_grid=True,
            use_convnext_blocks=True,
            local_grid_levels=2,
            local_grid_channels=4,
            convnext_mlp_ratio=2,
            convnext_kernel_size=3,
        ),
        decoder_codec="portfolio_auto",
    )

    assert metadata["schema"] == "hi_nerv_source_faithfulness.v1"
    assert metadata["official_hinerv_control"] is True
    assert metadata["source_faithful_official_hinerv"] is False
    assert metadata["official_source_parity_proof_required"] is True
    for forbidden in (
        "score_claim",
        "frontier_score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        assert forbidden not in metadata


def test_hinerv_refuses_non_official_control_candidate_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_train(**_kwargs):
        raise AssertionError("local HiV1-style candidate must refuse before training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_local_control_refusal",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        modelsize_candidate={
            "schema": "hinerv_modelsize_candidate.v1",
            "family": "hi_nerv",
            "candidate_id": "local-hiv1-unit-candidate",
            "latent_dim": 4,
            "embed_dim": 4,
            "decoder_channel": 4,
            "decoder_codec": "int4_mixed",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 100_000,
            "nominal_under_ceiling": True,
            "use_hierarchical_feature_grid": False,
            "use_convnext_blocks": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "hi_nerv_official_control_launch_refused"
    assert out["training_executed"] is False
    assert out["trainer_launch_allowed"] is False
    assert out["hi_nerv_source_faithfulness"]["official_hinerv_control"] is False
    assert out["hi_nerv_control_precedence"]["schema"] == (
        "hi_nerv_launch_control_precedence.v1"
    )
    assert out["hi_nerv_control_precedence"][
        "more_finely_grained_child_rules_take_priority"
    ] is True
    assert out["hi_nerv_control_precedence"][
        "parent_rules_remain_required_guardrails"
    ] is True
    assert "hinerv_official_hierarchical_feature_grid_not_enabled" in out[
        "hi_nerv_control_precedence"
    ]["source_base_blockers"]
    assert "hinerv_official_control_required_for_top_priority_launch" in out[
        "blockers"
    ]
    assert "hinerv_official_hierarchical_feature_grid_not_enabled" in out[
        "blockers"
    ]
    assert "hinerv_official_convnext_blocks_not_enabled" in out["blockers"]


def test_startup_marker_only_output_dir_is_not_dirty(tmp_path: Path) -> None:
    out = tmp_path / "candidate"
    out.mkdir()
    marker = out / runner_mod.COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    marker.write_text("{}", encoding="utf-8")

    assert (
        runner_mod._has_disallowed_existing_output_artifacts(
            out,
            allow_startup_marker_only=True,
        )
        is False
    )

    (out / "hi_nerv_mlx_training").mkdir()
    assert (
        runner_mod._has_disallowed_existing_output_artifacts(
            out,
            allow_startup_marker_only=True,
        )
        is False
    )

    (out / "hi_nerv_mlx_training" / "telemetry.jsonl").write_text(
        "{}",
        encoding="utf-8",
    )
    assert (
        runner_mod._has_disallowed_existing_output_artifacts(
            out,
            allow_startup_marker_only=True,
        )
        is True
    )


def test_torch_scorer_device_alias_resolves_gpu_to_concrete_backend() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True),
        ),
    )

    assert (
        runner_mod._resolve_torch_scorer_device_alias(
            "gpu",
            torch_module=fake_torch,
        )
        == "mps"
    )

    fake_cuda = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True),
        ),
    )
    assert (
        runner_mod._resolve_torch_scorer_device_alias(
            "gpu",
            torch_module=fake_cuda,
        )
        == "cuda"
    )
    assert runner_mod._resolve_torch_scorer_device_alias("metal") == "mps"


def test_torch_scorer_device_alias_fails_closed_without_gpu() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False),
        ),
    )

    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match=r"neither torch\.cuda nor torch\.backends\.mps",
    ):
        runner_mod._resolve_torch_scorer_device_alias(
            "gpu",
            torch_module=fake_torch,
        )


def test_hinerv_execute_allows_runner_startup_marker_only_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "hinerv_marker_only"
    output_dir.mkdir()
    marker = output_dir / runner_mod.COMPACT_FAMILY_STARTUP_MARKER_FILENAME
    marker.write_text("{}", encoding="utf-8")
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=output_dir,
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        distillation_device="mps",
        requested_distillation_device="gpu",
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs
    assert captured_train_kwargs["distillation_device"] == "mps"
    assert captured_train_kwargs["requested_distillation_device"] == "gpu"
    assert captured_train_kwargs["prioritized_pair_indices"] == ()
    assert captured_train_kwargs["use_hierarchical_feature_grid"] is True
    assert captured_train_kwargs["use_convnext_blocks"] is True
    assert out["execute_family"] == "hi_nerv"
    assert out["training_executed"] is True
    assert marker.is_file()
    embedded_plan = out["nerv_long_training_campaign_plan"]
    assert embedded_plan["planner_row_queue_artifact_path"].endswith(
        "/compact_renderer_mlx_spine_runner_report.json"
    )
    embedded_command = embedded_plan["campaign_rows"][0]["command_argv"]
    assert "--planner-row-queue-artifact" in embedded_command


def test_hinerv_execute_forwards_prioritized_pair_indices(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=4)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "score_aware_training": {
                    "decoder_weight_gradient_saliency_artifact": None,
                }
            },
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_priority",
        num_pairs=4,
        epochs=1,
        batch_pair_indices_per_step=2,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        checkpoint_interval_epochs=7,
        checkpoint_dir=tmp_path / "external_checkpoints",
        resume_from_checkpoint=tmp_path / "external_checkpoints/epoch000006.meta.json",
        prioritized_pair_indices=(3, 1, 3),
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["prioritized_pair_indices"] == (3, 1)
    assert captured_train_kwargs["use_hierarchical_feature_grid"] is True
    assert captured_train_kwargs["use_convnext_blocks"] is True
    assert captured_train_kwargs["checkpoint_interval_epochs"] == 7
    assert captured_train_kwargs["checkpoint_dir"] == (
        tmp_path / "external_checkpoints"
    ).resolve(strict=False)
    assert captured_train_kwargs["resume_from_checkpoint"] == (
        tmp_path / "external_checkpoints/epoch000006.meta.json"
    ).resolve(strict=False)
    prioritized = out["score_aware_training"]["prioritized_pair_training"]
    assert prioritized["enabled"] is True
    assert prioritized["pair_indices"] == [3, 1]
    assert (
        prioritized["pair_index_domain"]
        == "decoded_prefix_pair_indices_0_to_num_pairs_minus_1"
    )
    assert prioritized["arbitrary_source_pair_hydration"] is False
    assert prioritized["target_hydration_pair_indices_consumed"] is False
    assert prioritized["requires_num_pairs_covering_pair_ids"] is True
    assert prioritized["score_claim"] is False
    assert prioritized["promotion_eligible"] is False
    assert out["score_claim"] is False


def test_parse_prioritized_pair_indices_arg() -> None:
    assert runner_mod._parse_nonnegative_int_csv("") == ()
    assert runner_mod._parse_nonnegative_int_csv("3,1,3,0") == (3, 1, 0)
    assert runner_mod._normalize_nonnegative_int_sequence(None) == ()
    assert runner_mod._normalize_nonnegative_int_sequence("3,1,3") == (3, 1)
    with pytest.raises(CompactRendererMlxSpineRunnerError):
        runner_mod._parse_nonnegative_int_csv("2,-1")


def test_parse_prioritized_pair_indices_file_arg(tmp_path: Path) -> None:
    pair_file = tmp_path / "feedback.json"
    pair_file.write_text(
        '{"sample_generalization_gate":{"hard_pair_coverage":'
        '{"prioritized_pair_indices":[8,2,8]}}}',
        encoding="utf-8",
    )
    args = runner_mod._parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--num-pairs",
            "10",
            "--prioritized-pair-indices",
            "3,2",
            "--prioritized-pair-indices-file",
            str(pair_file),
        ]
    )

    assert runner_mod._prioritized_pair_indices_from_args(args) == (3, 2, 8)


def test_parse_prioritized_pair_indices_allows_source_ids_above_num_pairs() -> None:
    args = runner_mod._parse_args(
        [
            "--execute-family",
            "snerv",
            "--num-pairs",
            "4",
            "--prioritized-pair-indices",
            "3,4",
        ]
    )

    assert runner_mod._prioritized_pair_indices_from_args(args) == (3, 4)


def test_hinerv_execute_rejects_out_of_range_prioritized_pairs(
    tmp_path: Path,
) -> None:
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="out-of-range"):
        execute_hi_nerv_mlx_scoreaware_and_adapt(
            output_dir=tmp_path / "hinerv_out_of_range_priority",
            num_pairs=4,
            epochs=1,
            batch_pair_indices_per_step=2,
            learning_rate=1e-3,
            source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
            hard_byte_ceilings=(178_000,),
            latent_dim=4,
            embed_dim=4,
            decoder_channel=4,
            segnet_distillation_weight=1.0,
            pose_distillation_weight=1.0,
            prioritized_pair_indices=(3, 4),
            repo_root=REPO_ROOT,
        )


def test_hinerv_private_smoke_rejects_out_of_range_prioritized_pairs(
    tmp_path: Path,
) -> None:
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="out-of-range"):
        runner_mod._run_hi_nerv_mlx_scoreaware_smoke(
            output_dir=tmp_path / "private_smoke_out_of_range_priority",
            num_pairs=4,
            epochs=1,
            batch_pair_indices_per_step=2,
            learning_rate=1e-3,
            source_video_path=tmp_path / "not_loaded_when_pair_guard_fails.mkv",
            latent_dim=4,
            embed_dim=4,
            decoder_channel=4,
            use_hierarchical_feature_grid=False,
            use_convnext_blocks=False,
            local_grid_levels=2,
            local_grid_channels=4,
            convnext_mlp_ratio=2,
            convnext_kernel_size=7,
            mid_injection_block_index=1,
            fine_injection_block_index=4,
            decoder_codec="portfolio_auto",
            ema_decay=0.9,
            segnet_distillation_weight=1.0,
            pose_distillation_weight=1.0,
            pose_distillation_loss="mse",
            pose_distillation_huber_delta=1.0,
            recon_loss_stage_weight=1.0,
            segnet_loss_stage_weight=1.0,
            pose_loss_stage_weight=1.0,
            segnet_distillation_objective="kl_t2",
            distillation_temperature=2.0,
            segnet_tau_boundary=1.0,
            segnet_hinge_margin=1.0,
            distillation_device="cpu",
            requested_distillation_device=None,
            allow_segnet_only_research=False,
            coder_aware_qat=False,
            coder_qat_quant_bits=8,
            coder_qat_quant_residual_weight=0.0,
            coder_qat_magnitude_weight=0.0,
            coder_qat_delta_weight=0.0,
            coder_qat_c1a_entropy_weight=0.0,
            coder_qat_c1a_sigma=0.0,
            coder_qat_c1a_sample_size=0,
            recon_pixel_weight_path=None,
            decoder_weight_waterfill_plan=None,
            recon_pixel_weight_auto_discovery=None,
            auto_segnet_boundary_recon_weight=False,
            recon_pixel_weight_tau=1.0,
            recon_pixel_weight_normalize="mean",
            mlx_prefilter_scorer_device=None,
            mlx_prefilter_scorer_batch_pairs=1,
            mlx_prefilter_progress_every=50,
            telemetry_flush_interval_epochs=1,
            checkpoint_interval_epochs=1,
            checkpoint_dir=None,
            resume_from_checkpoint=None,
            optimizer_kind="adamw",
            hi_nerv_optimizer_policy={},
            optimizer_controls={},
            prioritized_pair_indices=(4,),
            random_seed=0,
            scorer_upstream_dir=REPO_ROOT / "upstream",
            repo_root=REPO_ROOT,
        )


def test_adapt_pr95_mlx_report_emits_spine_acquisition_and_runner(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "stage08.pt"
    latents = tmp_path / "stage08.latents.npy"
    manifest = tmp_path / "stage08.pt.export_manifest.json"
    weights.write_bytes(b"trained-mlx-weights")
    latents.write_bytes(b"trained-mlx-latents")
    manifest.write_text(
        json.dumps({"schema": "pr95_mlx_long_training_pytorch_export_manifest.v1"}),
        encoding="utf-8",
    )
    report_path = tmp_path / "pr95_mlx_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "pr95_mlx_long_training_plan.v1",
                "mode": "executed_smoke",
                "evidence_grade": "[macOS-MLX research-signal]",
                "source_video_frame_count": 4,
                "max_frames": 4,
                "checkpoint_artifacts": [
                    {
                        "stage_index": 8,
                        "global_epoch": 8,
                        "pytorch_state_dict_path": weights.as_posix(),
                        "latents_path": latents.as_posix(),
                        "pytorch_export_manifest_path": manifest.as_posix(),
                        "pytorch_export_succeeded": True,
                        "trained_latents_exported": True,
                        "evidence_grade": "[macOS-MLX research-signal]",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    followup_path = tmp_path / "hprc_queue_followup_report.json"
    followup_path.write_text(
        json.dumps(
            {
                "schema": "hprc_queue_followup_report.v1",
                "training_result_path": "prior_hprc_rate_collapse.json",
                "archive": {
                    "archive_zip_sha256": "f" * 64,
                    "archive_zip_bytes": 217365,
                },
                "planner_learning_signals": [
                    {
                        "schema": "hprc_planner_learning_signal.v1",
                        "signal_id": (
                            "hprc_rate_feasible_but_resolution_distortion_bound"
                        ),
                        "status": "route_to_pose_geometry_or_predictive_redesign",
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    out = adapt_pr95_mlx_report_to_spine(
        pr95_mlx_report_path=report_path,
        output_dir=tmp_path / "out",
        hard_byte_ceilings=(178_000,),
        hprc_queue_followup_report_paths=(followup_path,),
        repo_root=REPO_ROOT,
    )

    assert out["schema"] == COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA
    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert Path(out["spine_adapter_report_path"]).is_file()
    acquisition = json.loads(Path(out["acquisition_report_path"]).read_text())
    assert acquisition["rows"][0]["family"] == "pr95_hnerv"
    assert acquisition["rows"][0]["coverage"]["declared_pairs"] == 2
    assert acquisition["rows"][0]["coverage"]["valid_for_base_comparison"] is False
    runner = json.loads(Path(out["bounded_runner_plan_path"]).read_text())
    assert runner["selected_runner_rows"] == []
    assert "no_full_coverage_compact_base_candidate" in runner["blockers"]
    assert (
        "implementation_readiness_blocked_fake_or_incomplete_candidate"
        in runner["blockers"]
    )
    assert runner["hprc_queue_followup_signal_rows"][0]["signal_id"] == (
        "hprc_rate_feasible_but_resolution_distortion_bound"
    )
    assert out["hprc_queue_followup_report_paths"] == [followup_path.as_posix()]
    assert "mlx_local_report_is_advisory_not_score_authority" in out["blockers"]


def test_plan_only_report_keeps_all_compact_families_false_authority(
    tmp_path: Path,
) -> None:
    report = build_plan_only_report(
        output_dir=tmp_path / "plan",
        hard_byte_ceilings=(178_000, 216_000),
        repo_root=REPO_ROOT,
    )

    assert report["nerv_oss_flag_audit"]["schema"] == "nerv_oss_flag_audit.v1"
    assert "--modelsize" in report["nerv_oss_flag_audit"]["hnerv_high_ev_flags"]
    assert "--modelsize" in report["nerv_oss_flag_audit"]["snerv_high_ev_flags"]
    assert "--quant-level" in report["nerv_oss_flag_audit"]["hinerv_high_ev_flags"]
    assert report["hinerv_modelsize_budget"]["schema"] == "nerv_modelsize_budget.v1"
    assert report["hinerv_modelsize_budget"]["selected_candidate_count"] > 0
    assert report["hinerv_modelsize_budget"]["score_claim"] is False
    assert report["snerv_modelsize_budget"]["schema"] == "snerv_modelsize_budget.v1"
    assert report["snerv_modelsize_budget"]["selected_candidate_count"] > 0
    assert report["snerv_modelsize_budget"]["score_claim"] is False
    campaign_plan = report["nerv_long_training_campaign_plan"]
    assert campaign_plan["schema"] == "nerv_long_training_campaign_plan.v1"
    assert campaign_plan["score_claim"] is False
    assert campaign_plan["ready_for_exact_eval_dispatch"] is False
    assert campaign_plan["family_counts"]["hi_nerv"] > 0
    assert campaign_plan["family_counts"]["snerv"] > 0
    launchable_local_rows = [
        row
        for row in campaign_plan["campaign_rows"]
        if row["local_mlx_launch_command_ready"]
    ]
    assert campaign_plan["launchable_local_row_count"] == len(launchable_local_rows)
    assert launchable_local_rows
    assert any(
        row["experiment_queue_entry"]["launch_authority_contract"][
            "queue_status_is_local_mlx_plan"
        ]
        is True
        for row in campaign_plan["campaign_rows"]
    )
    assert all(
        row["experiment_queue_entry"]["launch_authority_contract"][
            "queue_status_is_runnable_plan"
        ]
        is row["local_mlx_launch_command_ready"]
        for row in campaign_plan["campaign_rows"]
    )
    assert all(
        row["experiment_queue_entry"]["launch_authority_contract"][
            "queue_status_is_receiver_proof"
        ]
        is False
        and row["experiment_queue_entry"]["launch_authority_contract"][
            "queue_status_is_cpu_replay_proof"
        ]
        is False
        and row["experiment_queue_entry"]["launch_authority_contract"][
            "queue_status_is_exact_eval_authority"
        ]
        is False
        for row in campaign_plan["campaign_rows"]
    )
    assert campaign_plan["experiment_queue"]["schema"] == "experiment_queue.v1"
    assert campaign_plan["experiment_queue_experiment_count"] == campaign_plan[
        "campaign_row_count"
    ]
    ready_families = {
        row["family"]
        for row in launchable_local_rows
    }
    assert ready_families == {"snerv"}
    assert any(
        row["family"] == "hi_nerv"
        and "requires_verified_joint_p18_p19_recon_pixel_weight_artifact"
        in row["blockers"]
        for row in campaign_plan["campaign_rows"]
    )
    snerv_campaign_rows = [
        row for row in campaign_plan["campaign_rows"] if row["family"] == "snerv"
    ]
    assert snerv_campaign_rows
    assert all(row["local_mlx_launch_command_ready"] is True for row in snerv_campaign_rows)
    assert all(
        row["score_lowering_gate"]["command_materialized"] is True
        and row["score_lowering_gate"]["local_mlx_executable"] is True
        and row["score_lowering_gate"]["prelaunch_allowed"] is True
        and row["score_lowering_gate"]["promotion_prelaunch_allowed"] is False
        and row["score_lowering_gate"]["cpu_replay_ready"] is False
        and row["score_lowering_gate"]["exact_gate_ready"] is False
        and row["score_lowering_gate"]["score_claim"] is False
        and row["score_lowering_gate"]["score_claim_valid"] is False
        and row["score_lowering_gate"]["promotion_eligible"] is False
        and row["score_lowering_gate"]["ready_for_exact_eval_dispatch"] is False
        for row in snerv_campaign_rows
    )
    assert all(row["score_lowering_gate"]["launch_blockers"] == [] for row in snerv_campaign_rows)
    assert any(
        "snerv_byte_closed_archive_export_missing"
        in row["score_lowering_gate"]["promotion_blockers"]
        for row in snerv_campaign_rows
    )
    assert report["nerv_stack_synergy_audit"]["schema"] == (
        "nerv_stack_synergy_audit.v1"
    )
    assert report["nerv_stack_synergy_audit"]["score_claim"] is False
    assert {row["stack_id"] for row in report["nerv_stack_synergy_audit"]["stacks"]} == {
        "hi_nerv",
        "snerv",
    }

    families = {row["family"]: row for row in report["target_family_rows"]}
    assert "pr95_hnerv" in families
    assert "hi_nerv" in families
    assert "snerv" in families
    assert "rnerv" in families
    assert "sr_nerv" in families
    assert "boostnerv" in families
    assert "pvq_nerv" in families
    assert "rt_vq_nerv" in families
    assert "pact_nerv_selector_v4" in families
    assert families["pr95_hnerv"]["status"] == (
        "executable_mlx_archive_export_control_arm"
    )
    assert "not a PR95-faithful reproduction" in families["pr95_hnerv"][
        "execution_scope"
    ]
    assert families["pact_nerv_vq"]["status"] == "executable_mlx_backend_available"
    assert families["pact_nerv_selector_v4"]["status"] == (
        "executable_mlx_backend_available"
    )
    assert families["pact_nerv_selector_v4"]["section_value_profiler"] == (
        "tools/profile_pact_nerv_selector_v4_mlx_section_value.py"
    )
    assert "pact_nerv_selector_v4" in families["pact_nerv_selector_v4"][
        "trainer_entrypoint"
    ]
    assert families["pvq_nerv"]["status"] == "executable_via_pact_nerv_vq_adapter"
    assert families["hi_nerv"]["status"] == (
        "mlx_archive_export_adapter_available_"
        "distortion_fit_actuator_pending"
    )
    assert families["hi_nerv"]["trainer_entrypoint"].endswith(
        "--execute-family hi_nerv"
    )
    assert families["hi_nerv"]["archive_exporter"] == (
        "tac.substrates.hi_nerv.archive_candidate.export_hi_nerv_mlx_archive"
    )
    assert families["hi_nerv"]["stack_role"] == "primary_carrier"
    assert "super-small-rate-by-design" in families["hi_nerv"]["rate_axis_evidence"]
    assert "cheap bytes alone cannot promote" in families["hi_nerv"][
        "distortion_fit_blocker"
    ]
    hinerv_plan = families["hi_nerv"]["score_aware_carrier_training_plan"]
    assert hinerv_plan["planner_action"] == (
        "run_receiver_closed_modelsize_ladder_before_score_aware_training"
    )
    assert hinerv_plan["carrier_fit_status"] == "unusable"
    assert hinerv_plan["allocator_target_surface"] == "decoder_weights"
    assert hinerv_plan["score_claim"] is False
    assert hinerv_plan["ready_for_exact_eval_dispatch"] is False
    assert "carrier_fit_unusable_d_seg" in hinerv_plan["dispatch_blockers"]
    assert (
        "latent_posthoc_allocator_demoted_low_leverage"
        in hinerv_plan["dispatch_blockers"]
    )
    assert families["snerv"]["status"] == (
        "executable_cpu_advisory_plus_mlx_native_export_adapter_available"
    )
    assert families["snerv"]["trainer_kind"] == (
        "mlx_native_target_hydration_receiver_export_available_"
        "scoreaware_long_training_missing"
    )
    assert families["snerv"]["next_action"].startswith(
        "bind_learned_mlx_scoreaware_decoder_training_to_snerv_native_export"
    )
    assert "optional MLX-native target-hydration/export/receiver-proof attachment" in (
        families["snerv"]["execution_scope"]
    )
    assert families["snerv"]["archive_exporter"] == (
        "tac.substrates.snerv_inverse_steg_carrier.archive_candidate."
        "export_snerv_archive_bound_candidate_package"
    )
    assert families["snerv"]["stack_role"] == "primary_carrier"
    assert families["snerv"]["score_aware_carrier_training_plan"][
        "score_aware_training_ready"
    ] is False
    assert "missing_training_stack:real_segnet_teacher" in families["snerv"][
        "score_aware_carrier_training_plan"
    ]["dispatch_blockers"]
    assert (
        "sr_nerv_lowres_encode_superresolve_resolution_deadzone"
        in families["snerv"]["allowed_enhancers"]
    )
    assert "ffnerv_flow_pose_channel" in families["snerv"]["allowed_enhancers"]
    assert families["sr_nerv"]["stack_role"] == (
        "resolution_axis_enhancer_or_design_knob"
    )
    assert families["sr_nerv"]["enhancer_priority"] > families["boostnerv"][
        "carrier_priority"
    ]
    assert families["rnerv"]["status"] == "migration_required"
    assert families["rnerv"]["stack_role"] == "enhancer_or_search_prior"
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_plan_only_report_routes_backend_rows_by_real_executability(
    tmp_path: Path,
) -> None:
    report = build_plan_only_report(
        output_dir=tmp_path / "plan",
        hard_byte_ceilings=(178_000,),
        repo_root=REPO_ROOT,
    )

    rows = {
        (row["family"], row["hard_byte_ceiling"]): row
        for row in report["compact_base_campaign_rows"]
    }
    assert rows[("pact_nerv_vq", 178_000)]["route_status"] == (
        "queued_for_mlx_training_archive_export_receiver_proof"
    )
    assert rows[("pr95_hnerv", 178_000)]["route_status"] == (
        "queued_for_mlx_training_archive_export_receiver_proof"
    )
    assert rows[("pvq_nerv", 178_000)]["canonical_family"] == "pact_nerv_vq"
    assert rows[("pvq_nerv", 178_000)]["route_status"] == (
        "queued_for_mlx_training_archive_export_receiver_proof"
    )
    assert rows[("pact_nerv_selector_v4", 178_000)]["route_status"] == (
        "queued_for_mlx_training_archive_export_receiver_proof"
    )
    assert rows[("pact_nerv_selector_v4", 178_000)]["section_value_profiler"] == (
        "tools/profile_pact_nerv_selector_v4_mlx_section_value.py"
    )
    assert rows[("hi_nerv", 178_000)]["route_status"] == (
        "queued_for_mlx_archive_adapter_smoke_scoreaware_training_pending"
    )
    assert rows[("hi_nerv", 178_000)]["stack_role"] == "primary_carrier"
    assert "export_hi_nerv_mlx_archive" in rows[("hi_nerv", 178_000)][
        "archive_exporter"
    ]
    modelsize_budget = rows[("hi_nerv", 178_000)]["modelsize_budget"]
    assert modelsize_budget["schema"] == "nerv_modelsize_budget.v1"
    assert modelsize_budget["hard_byte_ceilings"] == [178_000]
    assert modelsize_budget["selected_candidate_count"] > 0
    assert modelsize_budget["ready_for_exact_eval_dispatch"] is False
    snerv_budget = rows[("snerv", 178_000)]["modelsize_budget"]
    assert snerv_budget["schema"] == "snerv_modelsize_budget.v1"
    assert snerv_budget["hard_byte_ceilings"] == [178_000]
    assert snerv_budget["selected_candidate_count"] > 0
    assert snerv_budget["ready_for_exact_eval_dispatch"] is False
    campaign_plan = rows[("hi_nerv", 178_000)][
        "score_aware_carrier_training_plan"
    ]
    assert campaign_plan["planner_action"] == (
        "run_receiver_closed_modelsize_ladder_before_score_aware_training"
    )
    assert campaign_plan["linf_latent_posthoc_status"] == "demoted"
    assert campaign_plan["promotion_eligible"] is False
    assert rows[("snerv", 178_000)]["route_status"] == (
        "migration_required_before_runner_execution"
    )
    assert rows[("snerv", 178_000)]["stack_role"] == "primary_carrier"
    assert rows[("sr_nerv", 178_000)]["route_status"] == (
        "migration_required_before_runner_execution"
    )
    assert rows[("sr_nerv", 178_000)]["stack_role"] == (
        "resolution_axis_enhancer_or_design_knob"
    )
    assert "scorer_mirror_check" in rows[("sr_nerv", 178_000)]["next_action"]
    assert rows[("boostnerv", 178_000)]["route_status"] == (
        "migration_required_before_runner_execution"
    )
    assert rows[("boostnerv", 178_000)]["stack_role"] == "enhancer_bolt_on"
    assert rows[("rnerv", 178_000)]["trainer_entrypoint"] is None
    assert rows[("rnerv", 178_000)]["stack_role"] == "enhancer_or_search_prior"
    assert rows[("pact_nerv_vq", 178_000)]["score_claim"] is False
    assert rows[("pact_nerv_vq", 178_000)]["ready_for_exact_eval_dispatch"] is False
    assert rows[("pact_nerv_selector_v4", 178_000)]["score_claim"] is False
    assert (
        rows[("pact_nerv_selector_v4", 178_000)]["ready_for_exact_eval_dispatch"]
        is False
    )


def test_execute_modelsize_candidate_auto_uses_tightest_viable_byte_ceiling() -> None:
    hi = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000, 285_000),
    )
    sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000, 285_000),
    )

    assert hi is not None
    assert hi["family"] == "hi_nerv"
    assert hi["hard_byte_ceiling"] == 178_000
    assert hi["nominal_under_ceiling"] is True
    assert hi["use_hierarchical_feature_grid"] is True
    assert hi["use_convnext_blocks"] is True
    hi_precedence = hi["modelsize_control_contract"]["control_precedence"]
    assert hi_precedence["more_finely_grained_child_rules_take_priority"] is True
    assert hi_precedence[
        "official_controls_are_base_constraints_not_rate_optimizer_overrides"
    ] is True
    target_hi = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000,),
        num_pairs=17,
        hinerv_target_modelsize_mparams=(0.03,),
    )
    assert target_hi is not None
    assert target_hi["family"] == "hi_nerv"
    assert target_hi["capacity_source"] == "local_hinerv_target_modelsize"
    assert target_hi["use_hierarchical_feature_grid"] is True
    assert target_hi["use_convnext_blocks"] is True
    assert target_hi["target_modelsize_mparams"] == 0.03
    target_hi_precedence = target_hi["modelsize_control_contract"][
        "control_precedence"
    ]
    assert target_hi_precedence["highest_specificity_active_layer"] == (
        "pact_target_modelsize_child_rule"
    )
    assert target_hi_precedence[
        "more_finely_grained_child_rules_take_priority"
    ] is True
    assert target_hi["modelsize_error_mparams"] == pytest.approx(
        abs(target_hi["modelsize_mparams"] - 0.03)
    )
    assert target_hi["candidate_id"].endswith("_tgtmp0p03")
    reparsed_target_hi = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id=target_hi["candidate_id"],
        hard_byte_ceilings=(178_000,),
        num_pairs=17,
    )
    assert reparsed_target_hi == target_hi
    assert target_hi["ready_for_exact_eval_dispatch"] is False
    assert sn is not None
    assert sn["family"] == "snerv"
    assert sn["hard_byte_ceiling"] == 285_000
    assert sn["nominal_under_ceiling"] is True
    official_sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000,),
        snerv_official_modelsize_mparams=(0.05,),
    )
    assert official_sn is not None
    assert official_sn["family"] == "snerv"
    assert official_sn["capacity_source"] == "official_snerv_modelsize"
    assert official_sn["modelsize_mparams"] == 0.05
    assert official_sn["official_modelsize_solution"]["fc_dim"] == (
        official_sn["fc_dim"]
    )
    assert official_sn["ready_for_exact_eval_dispatch"] is False
    target_hi = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(36_000,),
        hinerv_target_modelsize_mparams=(0.02,),
    )
    assert target_hi is not None
    assert target_hi["family"] == "hi_nerv"
    assert target_hi["capacity_source"] == "local_hinerv_target_modelsize"
    assert target_hi["use_hierarchical_feature_grid"] is True
    assert target_hi["use_convnext_blocks"] is True
    assert target_hi["target_modelsize_mparams"] == 0.02
    assert target_hi["modelsize_error_mparams"] is not None
    assert "_tgtmp0p02" in target_hi["candidate_id"]
    assert target_hi["ready_for_exact_eval_dispatch"] is False
    shared_target_hi = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000,),
        num_pairs=17,
        target_modelsize_mparams=(0.03,),
    )
    assert shared_target_hi is not None
    assert shared_target_hi["family"] == "hi_nerv"
    assert shared_target_hi["capacity_source"] == "local_hinerv_target_modelsize"
    assert shared_target_hi["use_hierarchical_feature_grid"] is True
    assert shared_target_hi["use_convnext_blocks"] is True
    assert shared_target_hi["target_modelsize_mparams"] == 0.03
    assert shared_target_hi["modelsize_error_mparams"] == pytest.approx(
        abs(shared_target_hi["modelsize_mparams"] - 0.03)
    )
    assert shared_target_hi["candidate_id"].endswith("_tgtmp0p03")
    shared_hi_contract = shared_target_hi["modelsize_control_contract"]
    assert shared_hi_contract["schema"] == "nerv_modelsize_control_contract.v1"
    assert shared_hi_contract["control_semantics"] == (
        "local_receiver_visible_grid_search_nearest_target"
    )
    assert shared_hi_contract["shared_target_modelsize_mparams_consumed_as"] == (
        "nearest_local_param_count_target"
    )
    assert shared_hi_contract["modelsize_mparams_is_official_upstream_flag"] is False
    assert shared_hi_contract["archive_bytes_authority_required"] is True
    shared_target_sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000,),
        target_modelsize_mparams=(0.05,),
    )
    assert shared_target_sn is not None
    assert shared_target_sn["family"] == "snerv"
    assert shared_target_sn["capacity_source"] == "official_snerv_modelsize"
    assert shared_target_sn["modelsize_mparams"] == 0.05
    assert shared_target_sn["official_modelsize_solution"]["fc_dim"] == (
        shared_target_sn["fc_dim"]
    )
    assert shared_target_sn["ready_for_exact_eval_dispatch"] is False
    shared_sn_contract = shared_target_sn["modelsize_control_contract"]
    assert shared_sn_contract["schema"] == "nerv_modelsize_control_contract.v1"
    assert shared_sn_contract["control_semantics"] == (
        "official_snerv_modelsize_quadratic_fc_dim_solve"
    )
    assert shared_sn_contract["shared_target_modelsize_mparams_consumed_as"] == (
        "official_snerv_modelsize_quadratic_fc_dim_solve"
    )
    assert shared_sn_contract["modelsize_mparams_is_official_upstream_flag"] is True
    assert shared_sn_contract["modelsize_mparams_caps_archive_zip_bytes"] is False
    assert shared_sn_contract["mutates_receiver_visible_fc_dim"] is True
    assert shared_sn_contract["archive_bytes_authority_required"] is True
    explicit_shared_target_sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=shared_target_sn["candidate_id"],
        hard_byte_ceilings=(178_000,),
    )
    assert explicit_shared_target_sn is not None
    assert explicit_shared_target_sn["candidate_id"] == shared_target_sn["candidate_id"]
    assert explicit_shared_target_sn["capacity_source"] == "official_snerv_modelsize"
    assert explicit_shared_target_sn["modelsize_mparams"] == 0.05
    assert explicit_shared_target_sn["official_modelsize_solution"]["fc_dim"] == (
        explicit_shared_target_sn["fc_dim"]
    )
    explicit = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id=hi["candidate_id"],
        hard_byte_ceilings=(178_000, 285_000),
    )
    assert explicit == hi
    assert (
        _resolve_execute_modelsize_candidate(
            family="hi_nerv",
            candidate_id="manual",
            hard_byte_ceilings=(178_000,),
        )
        is None
    )
    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError):
        _resolve_execute_modelsize_candidate(
            family="snerv",
            candidate_id="missing-candidate",
            hard_byte_ceilings=(178_000,),
        )


def test_byte_cap_controller_uses_measured_archive_feedback() -> None:
    attached = runner_mod._attach_byte_cap_controller_predictions(
        [
            {
                "candidate_id": "small",
                "family": "hi_nerv",
                "hard_byte_ceiling": 100,
                "nominal_total_payload_bytes": 60,
                "decoder_codec": "int4_scale_bundled",
                "nominal_under_ceiling": True,
            },
            {
                "candidate_id": "large",
                "family": "hi_nerv",
                "hard_byte_ceiling": 100,
                "nominal_total_payload_bytes": 80,
                "decoder_codec": "int4_scale_bundled",
                "nominal_under_ceiling": True,
            },
        ],
        family="hi_nerv",
        byte_cap_feedback_rows=[
            {
                "row_id": "measured_export",
                "family": "hi_nerv",
                "nominal_total_payload_bytes": 60,
                "measured_archive_bytes": 90,
                "decoder_codec": "int4_scale_bundled",
                "receiver_proof_passed": True,
            }
        ],
    )

    by_id = {row["candidate_id"]: row for row in attached}
    assert by_id["small"]["byte_cap_controller"]["predicted_archive_bytes"] == 90
    assert by_id["small"]["byte_cap_controller"][
        "predicted_under_hard_byte_ceiling"
    ] is True
    assert by_id["large"]["byte_cap_controller"]["predicted_archive_bytes"] == 120
    assert by_id["large"]["byte_cap_controller"][
        "predicted_under_hard_byte_ceiling"
    ] is False
    assert runner_mod._byte_cap_controller_predicts_under_ceiling(by_id["small"])
    assert not runner_mod._byte_cap_controller_predicts_under_ceiling(by_id["large"])


def test_byte_cap_controller_exploits_measured_compression_headroom() -> None:
    attached = runner_mod._attach_byte_cap_controller_predictions(
        [
            {
                "candidate_id": "larger",
                "family": "hi_nerv",
                "hard_byte_ceiling": 100,
                "nominal_total_payload_bytes": 140,
                "decoder_codec": "portfolio_auto",
                "nominal_under_ceiling": False,
            }
        ],
        family="hi_nerv",
        byte_cap_feedback_rows=[
            {
                "row_id": "compressed_export",
                "family": "hi_nerv",
                "nominal_total_payload_bytes": 100,
                "measured_archive_bytes": 50,
                "decoder_codec": "portfolio_auto",
                "receiver_proof_ready": True,
            }
        ],
    )

    controller = attached[0]["byte_cap_controller"]
    assert controller["prediction_rule"] == (
        "max_observed_archive_to_nominal_ratio_or_additive_overhead"
    )
    assert controller["predicted_archive_bytes"] == 90
    assert controller["predicted_under_hard_byte_ceiling"] is True


def test_byte_cap_controller_ignores_unproven_feedback_rows() -> None:
    attached = runner_mod._attach_byte_cap_controller_predictions(
        [
            {
                "candidate_id": "candidate",
                "family": "hi_nerv",
                "hard_byte_ceiling": 100,
                "nominal_total_payload_bytes": 80,
                "decoder_codec": "int4_scale_bundled",
                "nominal_under_ceiling": True,
            }
        ],
        family="hi_nerv",
        byte_cap_feedback_rows=[
            {
                "row_id": "optimistic_unproven",
                "family": "hi_nerv",
                "nominal_total_payload_bytes": 100,
                "measured_archive_bytes": 50,
                "decoder_codec": "int4_scale_bundled",
                "receiver_proof_passed": False,
            }
        ],
    )

    controller = attached[0]["byte_cap_controller"]
    assert controller["predicted_archive_bytes"] == 80
    assert controller["prediction_rule"] == "nominal_payload_bytes_uncalibrated"
    assert controller["calibration_observation_count"] == 0
    assert "byte_cap_controller_measured_archive_feedback_missing" in controller[
        "blockers"
    ]


def test_execute_modelsize_auto_uses_byte_cap_feedback_to_avoid_overcap() -> None:
    nominal = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000,),
    )
    assert nominal is not None
    assert nominal["nominal_total_payload_bytes"] <= 178_000

    calibrated = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="auto",
        hard_byte_ceilings=(178_000,),
        byte_cap_feedback_rows=[
            {
                "family": "hi_nerv",
                "row_id": "prior_export",
                "decoder_codec": nominal["decoder_codec"],
                "nominal_total_payload_bytes": nominal[
                    "nominal_total_payload_bytes"
                ],
                "measured_archive_bytes": nominal[
                    "nominal_total_payload_bytes"
                ]
                + 20_000,
                "receiver_proof_passed": True,
            }
        ],
    )

    assert calibrated is not None
    assert calibrated["hard_byte_ceiling"] == 178_000
    assert calibrated["nominal_total_payload_bytes"] < nominal[
        "nominal_total_payload_bytes"
    ]
    controller = calibrated["byte_cap_controller"]
    assert controller["calibration_observation_count"] == 1
    assert controller["predicted_under_hard_byte_ceiling"] is True
    assert controller["predicted_archive_bytes"] <= 178_000


def test_execute_modelsize_auto_fails_closed_when_calibrated_feedback_has_no_under_cap_candidate() -> None:
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="hi_nerv_modelsize_auto_no_calibrated_candidate_under_hard_byte_ceiling",
    ):
        _resolve_execute_modelsize_candidate(
            family="hi_nerv",
            candidate_id="auto",
            hard_byte_ceilings=(178_000,),
            byte_cap_feedback_rows=[
                {
                    "family": "hi_nerv",
                    "row_id": "oversized_export",
                    "decoder_codec": "portfolio_auto",
                    "nominal_total_payload_bytes": 1,
                    "measured_archive_bytes": 1_000_000,
                    "receiver_proof_passed": True,
                }
            ],
        )


def test_modelsize_byte_cap_feedback_loader_accepts_checkpoint_exports(
    tmp_path: Path,
) -> None:
    path = tmp_path / "export.json"
    path.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_checkpoint_archive_export.v1",
                "family": "hi_nerv",
                "candidate_id": "hinerv-row",
                "archive_bytes": 123_456,
                "decoder_codec": "int4_mixed",
                "receiver_proof_ready": True,
                "modelsize_candidate": {
                    "candidate_id": "hinerv-row",
                    "family": "hi_nerv",
                    "hard_byte_ceiling": 178_000,
                    "nominal_total_payload_bytes": 111_111,
                },
            }
        ),
        encoding="utf-8",
    )

    rows = runner_mod._load_modelsize_byte_cap_feedback_rows([path])

    assert rows == [
        {
            "family": "hi_nerv",
            "row_id": "hinerv-row",
            "candidate_id": "hinerv-row",
            "measured_archive_bytes": 123_456,
            "nominal_total_payload_bytes": 111_111,
            "hard_byte_ceiling": 178_000,
            "decoder_codec": "int4_mixed",
            "receiver_closed": True,
            "source_path": path.resolve(strict=False).as_posix(),
        }
    ]


def test_modelsize_byte_cap_feedback_loader_prefers_nested_candidate_nominal_over_packet_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snerv_export.json"
    path.write_text(
        json.dumps(
            {
                "schema": "snerv_checkpoint_archive_export.v1",
                "family": "snerv",
                "archive_bytes": 444_036,
                "packet_bytes": 2_347_396,
                "decoder_codec": "int8_symmetric",
                "receiver_proof_passed": True,
                "receiver_contract_satisfied": True,
                "modelsize_candidate": {
                    "candidate_id": "snerv-row",
                    "family": "snerv",
                    "hard_byte_ceiling": 216_000,
                    "nominal_total_payload_bytes": 188_854,
                    "decoder_payload_codec": "int8_symmetric",
                },
            }
        ),
        encoding="utf-8",
    )

    rows = runner_mod._load_modelsize_byte_cap_feedback_rows([path])

    assert rows == [
        {
            "family": "snerv",
            "candidate_id": "snerv-row",
            "measured_archive_bytes": 444_036,
            "nominal_total_payload_bytes": 188_854,
            "hard_byte_ceiling": 216_000,
            "decoder_codec": "int8_symmetric",
            "receiver_closed": True,
            "source_path": path.resolve(strict=False).as_posix(),
        }
    ]


def test_modelsize_byte_cap_feedback_loader_reads_startup_candidate_fallback(
    tmp_path: Path,
) -> None:
    startup = tmp_path / "startup.json"
    startup.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_startup_marker.v1",
                "modelsize_candidate": {
                    "candidate_id": "hinerv-fallback",
                    "family": "hi_nerv",
                    "hard_byte_ceiling": 178_000,
                    "nominal_total_payload_bytes": 117_000,
                    "decoder_codec": "int7_mixed",
                },
            }
        ),
        encoding="utf-8",
    )
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            {
                "schema": "hinerv_checkpoint_archive_export.v1",
                "family": "hi_nerv",
                "candidate_id": "hinerv-fallback",
                "archive_bytes": 122_074,
                "receiver_proof_ready": True,
                "startup_json_path": startup.as_posix(),
            }
        ),
        encoding="utf-8",
    )

    rows = runner_mod._load_modelsize_byte_cap_feedback_rows([export])

    assert rows[0]["candidate_id"] == "hinerv-fallback"
    assert rows[0]["measured_archive_bytes"] == 122_074
    assert rows[0]["nominal_total_payload_bytes"] == 117_000
    assert rows[0]["hard_byte_ceiling"] == 178_000
    assert rows[0]["decoder_codec"] == "int7_mixed"
    assert rows[0]["receiver_closed"] is True


def test_snerv_official_modelsize_controls_require_candidate_resolution() -> None:
    manual = _parse_args(
        [
            "--execute-family",
            "snerv",
            "--modelsize-candidate-id",
            "manual",
            "--snerv-fc-dim",
            "13",
        ]
    )
    blocked = _parse_args(
        [
            "--execute-family",
            "snerv",
            "--modelsize-candidate-id",
            "manual",
            "--snerv-official-modelsize-mparams",
            "0.05",
            "--snerv-official-enc-strds",
            "1,2,2",
        ]
    )

    assert runner_mod._snerv_official_modelsize_candidate_resolution_blockers(
        manual
    ) == []
    blockers = runner_mod._snerv_official_modelsize_candidate_resolution_blockers(
        blocked
    )
    assert (
        "snerv_official_modelsize_control_requires_candidate_resolution:"
        "--snerv-official-modelsize-mparams"
    ) in blockers
    assert (
        "snerv_official_modelsize_control_requires_candidate_resolution:"
        "--snerv-official-enc-strds"
    ) in blockers
    with pytest.raises(SystemExit, match="SNeRV official modelsize controls require"):
        runner_mod.main(
            [
                "--execute-family",
                "snerv",
                "--modelsize-candidate-id",
                "manual",
                "--snerv-official-modelsize-mparams",
                "0.05",
            ]
        )


def test_execute_modelsize_candidate_resolves_self_describing_queue_ids() -> None:
    hi = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id="hinerv_np600_ld4_ed12_dc12_int4_mixed_ceil36000",
        hard_byte_ceilings=(178_000,),
    )
    hi_official = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id=(
            "hinerv_np600_ld4_ed12_dc12_mi1fi4_hfg_cnx_lg2c4_cx2k7_"
            "int4_mixed_ceil36000"
        ),
        hard_byte_ceilings=(178_000,),
    )
    hi_target = _resolve_execute_modelsize_candidate(
        family="hi_nerv",
        candidate_id=(
            "hinerv_np600_ld4_ed12_dc12_mi1fi4_hfg_cnx_lg2c4_cx2k7_"
            "int4_mixed_ceil36000_tgtmp0p02"
        ),
        hard_byte_ceilings=(178_000,),
    )
    sn = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id="snerv_np600_lv2_lfb1p5_stepb0p5_int2_symmetric_ceil36000",
        hard_byte_ceilings=(178_000,),
    )
    sn_rich = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=(
            "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p1_"
            "mfu1-3_hfr0p25_t1_adbase_int2_symmetric_ceil36000"
        ),
        hard_byte_ceilings=(178_000,),
    )
    sn_spectra = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=(
            "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p3_"
            "mfu1-5_hfr0p375_t2_adspectra_int2_symmetric_ceil36000"
        ),
        hard_byte_ceilings=(178_000,),
    )
    sn_temporal = _resolve_execute_modelsize_candidate(
        family="snerv",
        candidate_id=(
            "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p3_"
            "mfu1-5_hfr0p375_t2_tmhaar1_adspectra_int2_symmetric_ceil36000"
        ),
        hard_byte_ceilings=(178_000,),
    )

    assert hi is not None
    assert hi["candidate_id"] == (
        "hinerv_np600_ld4_ed12_dc12_int4_mixed_ceil36000"
    )
    assert hi_target is not None
    assert hi_target["capacity_source"] == "local_hinerv_target_modelsize"
    assert hi_target["target_modelsize_mparams"] == 0.02
    assert hi_target["candidate_id"].endswith("_tgtmp0p02")
    assert hi["num_pairs"] == 600
    assert hi["hard_byte_ceiling"] == 36_000
    assert hi["decoder_codec"] == "int4_mixed"
    assert hi["use_hierarchical_feature_grid"] is False
    assert hi["mid_injection_block_index"] == 1
    assert hi["fine_injection_block_index"] == 4
    assert hi_official is not None
    assert hi_official["candidate_id"] == (
        "hinerv_np600_ld4_ed12_dc12_mi1fi4_hfg_cnx_lg2c4_cx2k7_"
        "int4_mixed_ceil36000"
    )
    assert hi_official["use_hierarchical_feature_grid"] is True
    assert hi_official["use_convnext_blocks"] is True
    assert hi_official["local_grid_levels"] == 2
    assert hi_official["local_grid_channels"] == 4
    assert hi_official["convnext_mlp_ratio"] == 2
    assert hi_official["convnext_kernel_size"] == 7
    assert hi_official["mid_injection_block_index"] == 1
    assert hi_official["fine_injection_block_index"] == 4
    assert hi_official["hard_byte_ceiling"] == 36_000
    assert sn is not None
    assert sn["candidate_id"] == (
        "snerv_np600_lv2_lfb1p5_stepb0p5_int2_symmetric_ceil36000"
    )
    assert sn["legacy_candidate_id"] is True
    assert sn["num_pairs"] == 600
    assert sn["hard_byte_ceiling"] == 36_000
    assert sn["decoder_payload_codec"] == "int2_symmetric"
    assert sn_rich is not None
    assert sn_rich["candidate_id"] == (
        "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p1_"
        "mfu1-3_hfr0p25_t1_adbase_int2_symmetric_ceil36000"
    )
    assert sn_rich["wavelet"] == "haar"
    assert sn_rich["fc_dim"] == 11
    assert sn_rich["emb_size"] == 2
    assert sn_rich["patch_radius"] == 1
    assert sn_rich["mfu_scales"] == [1, 3]
    assert sn_rich["hfr_gain"] == 0.25
    assert sn_rich["temporal_context"] == 1
    assert sn_rich["snerv_model_size_adapter"] == (
        "snerv_fc_dim_emb_size_adapter_v1"
    )
    assert sn_spectra is not None
    assert sn_spectra["candidate_id"] == (
        "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p3_"
        "mfu1-5_hfr0p375_t2_adspectra_int2_symmetric_ceil36000"
    )
    assert sn_spectra["patch_radius"] == 3
    assert sn_spectra["mfu_scales"] == [1, 5]
    assert sn_spectra["hfr_gain"] == pytest.approx(0.375)
    assert sn_spectra["temporal_context"] == 2
    assert sn_spectra["snerv_model_size_adapter"] == (
        SNERV_SPECTRA_PRESERVING_ADAPTER
    )
    assert sn_temporal is not None
    assert sn_temporal["temporal_context"] == 2
    assert sn_temporal["temporal_mode"] == "official_haar_dwt1d_lowpass"
    assert sn_temporal["candidate_id"].find("_tmhaar1_") >= 0
    with pytest.raises(CompactRendererMlxSpineRunnerError):
        _resolve_execute_modelsize_candidate(
            family="snerv",
            candidate_id=(
                "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p3_"
                "mfu1-5_hfr0p3750_t2_adspectra_int2_symmetric_ceil36000"
            ),
            hard_byte_ceilings=(178_000,),
        )


def test_pose_instability_epoch_monitor_requires_sustained_bad_pose() -> None:
    monitor = runner_mod._PoseInstabilityEpochMonitor(
        min_epoch=2,
        consecutive_bad_epochs=2,
        pose_loss_threshold=100.0,
        pose_axis_threshold=100.0,
    )
    ok = SimpleNamespace(
        epoch=2,
        loss_components={"loss_part_pose_distill": 99.0},
        per_axis_decomposition={"pose": 99.0},
    )
    bad_a = SimpleNamespace(
        epoch=3,
        loss_components={"loss_part_pose_distill": 101.0},
        per_axis_decomposition={"pose": 10.0},
    )
    bad_b = SimpleNamespace(
        epoch=4,
        loss_components={"loss_part_pose_distill": 101.0},
        per_axis_decomposition={"pose": 101.0},
    )

    monitor(ok)
    monitor(bad_a)
    with pytest.raises(runner_mod.LongTrainingStopRequested):
        monitor(bad_b)


def test_active_campaign_lock_identity_excludes_output_dir(tmp_path: Path) -> None:
    weight = tmp_path / "weights.npz"
    np.savez_compressed(weight, weight=np.ones((1,), dtype=np.float32))
    source = tmp_path / "0.mkv"
    source.write_bytes(b"video")
    args_a = _parse_args([
        "--execute-family",
        "hi_nerv",
        "--output-dir",
        str(tmp_path / "a"),
        "--num-pairs",
        "600",
        "--epochs",
        "8",
        "--recon-pixel-weight-path",
        str(weight),
    ])
    args_b = _parse_args([
        "--execute-family",
        "hi_nerv",
        "--output-dir",
        str(tmp_path / "b"),
        "--num-pairs",
        "600",
        "--epochs",
        "8",
        "--recon-pixel-weight-path",
        str(weight),
    ])

    payload_a = runner_mod._active_campaign_lock_payload(
        args_a,
        source_video_path=source,
        hard_byte_ceilings=(178_000,),
    )
    payload_b = runner_mod._active_campaign_lock_payload(
        args_b,
        source_video_path=source,
        hard_byte_ceilings=(178_000,),
    )

    assert payload_a == payload_b
    assert payload_a["recon_pixel_weight_sha256"]
    assert runner_mod._campaign_lock_digest(payload_a) == (
        runner_mod._campaign_lock_digest(payload_b)
    )


def test_active_campaign_lock_refuses_duplicate_active_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "0.mkv"
    source.write_bytes(b"video")
    args = _parse_args([
        "--execute-family",
        "hi_nerv",
        "--num-pairs",
        "600",
        "--epochs",
        "8",
    ])
    monkeypatch.setattr(runner_mod, "_active_family_campaign_processes", lambda **_: [])

    lock_path = runner_mod._acquire_active_campaign_lock(
        output_dir=tmp_path / "a",
        args=args,
        source_video_path=source,
        hard_byte_ceilings=(178_000,),
    )
    assert lock_path is not None
    assert lock_path.is_file()
    manifest = json.loads(lock_path.read_text(encoding="utf-8"))
    assert manifest["family"] == "hi_nerv"
    assert manifest["planner_row_id"] is None
    assert manifest["modelsize_candidate_id"] == "auto"
    assert manifest["score_claim"] is False
    try:
        with pytest.raises(SystemExit, match="duplicate active"):
            runner_mod._acquire_active_campaign_lock(
                output_dir=tmp_path / "b",
                args=args,
                source_video_path=source,
                hard_byte_ceilings=(178_000,),
            )
    finally:
        runner_mod._release_active_campaign_lock(lock_path, os.getpid())
    assert not lock_path.exists()


def test_active_campaign_lock_refuses_same_family_snerv_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "0.mkv"
    source.write_bytes(b"video")
    args = _parse_args([
        "--execute-family",
        "snerv",
        "--num-pairs",
        "32",
        "--epochs",
        "1",
    ])
    monkeypatch.setattr(
        runner_mod,
        "_active_process_table_rows",
        lambda: [
            {
                "pid": 424242,
                "ppid": 1,
                "elapsed": "03:55",
                "command": (
                    ".venv/bin/python tools/run_snerv_inverse_steg_advisory.py "
                    "--n-pairs 600 --packet-out /Volumes/VertigoDataTier/pact/x.snar"
                ),
            }
        ],
    )
    monkeypatch.setattr(runner_mod, "_pid_is_alive", lambda pid: pid == 424242)

    with pytest.raises(SystemExit, match="active same-family"):
        runner_mod._acquire_active_campaign_lock(
            output_dir=tmp_path / "snerv_run",
            args=args,
            source_video_path=source,
            hard_byte_ceilings=(178_000,),
        )

    refusal_paths = sorted(
        (tmp_path / ".active_compact_renderer_campaign_locks").glob(
            "family_process_refusal_snerv_*.json"
        )
    )
    assert len(refusal_paths) == 1
    payload = json.loads(refusal_paths[0].read_text(encoding="utf-8"))
    assert payload["schema"] == runner_mod.ACTIVE_FAMILY_PROCESS_REFUSAL_SCHEMA
    assert payload["family"] == "snerv"
    assert payload["active_processes"][0]["pid"] == 424242
    assert payload["score_claim"] is False


def test_active_family_process_detection_ignores_current_process_ancestors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_mod, "_pid_is_alive", lambda pid: pid in {10, 11, 12})
    rows = [
        {
            "pid": 10,
            "ppid": 9,
            "elapsed": "00:01",
            "command": (
                "/bin/zsh -lc python tools/run_compact_renderer_mlx_spine_runner.py "
                "--execute-family snerv"
            ),
        },
        {
            "pid": 11,
            "ppid": 10,
            "elapsed": "00:01",
            "command": (
                "/Users/adpena/Projects/pact/.venv/bin/python "
                "tools/run_compact_renderer_mlx_spine_runner.py --execute-family snerv"
            ),
        },
        {
            "pid": 12,
            "ppid": 1,
            "elapsed": "03:55",
            "command": (
                ".venv/bin/python tools/run_snerv_inverse_steg_advisory.py "
                "--n-pairs 600"
            ),
        },
    ]

    matches = runner_mod._active_family_campaign_processes(
        family="snerv",
        current_pid=11,
        process_rows=rows,
    )

    assert [row["pid"] for row in matches] == [12]


def test_active_family_process_detection_ignores_hinerv_pytest_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_mod, "_pid_is_alive", lambda pid: pid == 424242)
    rows = [
        {
            "pid": 424242,
            "ppid": 1,
            "elapsed": "01:25",
            "command": (
                ".venv/bin/python -m pytest -q "
                "src/tac/tests/test_hinerv_archive_size_ladder.py "
                "src/tac/tests/test_compact_renderer_mlx_spine_runner.py"
            ),
        },
    ]

    matches = runner_mod._active_family_campaign_processes(
        family="hi_nerv",
        current_pid=1,
        process_rows=rows,
    )

    assert matches == []


def test_active_campaign_lock_allow_duplicate_skips_family_process_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "0.mkv"
    source.write_bytes(b"video")
    args = _parse_args([
        "--execute-family",
        "snerv",
        "--num-pairs",
        "32",
        "--epochs",
        "1",
        "--allow-duplicate-campaign",
    ])
    monkeypatch.setattr(
        runner_mod,
        "_active_process_table_rows",
        lambda: [
            {
                "pid": 424242,
                "ppid": 1,
                "elapsed": "03:55",
                "command": ".venv/bin/python tools/run_snerv_inverse_steg_advisory.py",
            }
        ],
    )
    monkeypatch.setattr(runner_mod, "_pid_is_alive", lambda pid: pid == 424242)

    lock_path = runner_mod._acquire_active_campaign_lock(
        output_dir=tmp_path / "snerv_run",
        args=args,
        source_video_path=source,
        hard_byte_ceilings=(178_000,),
    )

    assert lock_path is None
    assert not (tmp_path / ".active_compact_renderer_campaign_locks").exists()


@pytest.mark.parametrize(
    "function_name",
    ("_run_pact_nerv_vq_mlx_smoke", "_run_pact_nerv_selector_v4_mlx_smoke"),
)
def test_pact_compact_runners_forward_optimizer_and_shared_qat_metadata(
    function_name: str,
) -> None:
    source = Path(runner_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    target_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == function_name
    )
    calls = [
        node
        for node in ast.walk(target_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_mlx_score_aware_full_main"
    ]
    assert len(calls) == 1
    kw_names = {kw.arg for kw in calls[0].keywords if kw.arg is not None}
    assert "pr95_faithful_curriculum_enabled" in kw_names
    assert "pr95_curriculum_total_epochs" in kw_names
    assert "grad_clip_max_norm" in kw_names
    assert "weight_decay" in kw_names
    assert "optimizer_kind" in kw_names
    assert "warmup_epochs" in kw_names
    assert "warmup_steps_per_epoch" in kw_names
    assert "cosine_decay_enabled" in kw_names
    assert "cosine_decay_total_epochs" in kw_names
    assert "cosine_decay_min_lr_ratio" in kw_names
    target_source = ast.get_source_segment(source, target_fn) or ""
    assert '"optimizer_policy": strip_candidate_curriculum_authority_fields' in (
        target_source
    )
    assert '"optimizer_controls": strip_candidate_curriculum_authority_fields' in (
        target_source
    )
    assert '"coder_aware_qat": coder_qat_metadata_row' in target_source


def test_mlx_optimizer_controls_default_to_pact_muon_adamw() -> None:
    controls = runner_mod._resolve_mlx_score_aware_optimizer_controls(
        optimizer_kind=None,
        requested_weight_decay=None,
        grad_clip_max_norm=1.0,
        warmup_epochs=0,
        warmup_steps_per_epoch=1,
        cosine_decay_enabled=False,
        cosine_decay_total_epochs=None,
        cosine_decay_min_lr_ratio=1e-2,
        run_epochs=128,
    )

    assert controls["optimizer_kind"] == runner_mod.DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
    assert controls["weight_decay_effective"] == pytest.approx(1.0e-4)
    assert controls["weight_decay_defaulted"] is True
    assert controls["grad_clip_max_norm"] == pytest.approx(1.0)
    assert controls["borrowed_pr95_partition_rule"] is True
    assert controls["score_claim"] is False
    assert controls["ready_for_exact_eval_dispatch"] is False


def test_compact_runner_parser_defaults_to_shared_optimizer_kind() -> None:
    args = _parse_args(["--execute-family", "hi_nerv", "--num-pairs", "1"])

    assert args.optimizer_kind == runner_mod.DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
    assert (
        args.checkpoint_interval_epochs
        == runner_mod.DEFAULT_COMPACT_FAMILY_CHECKPOINT_INTERVAL_EPOCHS
    )
    assert args.checkpoint_dir is None
    assert args.resume_from_checkpoint is None


def test_compact_runner_checkpoint_controls_parse_and_validate() -> None:
    args = _parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--num-pairs",
            "1",
            "--checkpoint-interval-epochs",
            "17",
            "--checkpoint-dir",
            "ssd/checkpoints",
            "--resume-from-checkpoint",
            "ssd/checkpoints/epoch000016.meta.json",
        ]
    )

    assert args.checkpoint_interval_epochs == 17
    assert args.checkpoint_dir == Path("ssd/checkpoints")
    assert args.resume_from_checkpoint == Path("ssd/checkpoints/epoch000016.meta.json")
    assert runner_mod._resolve_checkpoint_interval_epochs(17, epochs=100) == 17
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="positive integer"):
        runner_mod._resolve_checkpoint_interval_epochs(True, epochs=100)
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="> 0"):
        runner_mod._resolve_checkpoint_interval_epochs(0, epochs=100)


def test_mlx_optimizer_controls_reject_weight_decay_for_no_decay_kind() -> None:
    with pytest.raises(CompactRendererMlxSpineRunnerError, match="weight-decay"):
        runner_mod._resolve_mlx_score_aware_optimizer_controls(
            optimizer_kind="adam",
            requested_weight_decay=1.0e-4,
            grad_clip_max_norm=1.0,
            warmup_epochs=0,
            warmup_steps_per_epoch=1,
            cosine_decay_enabled=False,
            cosine_decay_total_epochs=None,
            cosine_decay_min_lr_ratio=1e-2,
            run_epochs=128,
        )


def test_mlx_optimizer_controls_default_cosine_total_to_run_epochs() -> None:
    controls = runner_mod._resolve_mlx_score_aware_optimizer_controls(
        optimizer_kind="pact_muon_adamw",
        requested_weight_decay=2.0e-4,
        grad_clip_max_norm=0.5,
        warmup_epochs=5,
        warmup_steps_per_epoch=7,
        cosine_decay_enabled=True,
        cosine_decay_total_epochs=None,
        cosine_decay_min_lr_ratio=5.0e-2,
        run_epochs=128,
    )

    assert controls["weight_decay_effective"] == pytest.approx(2.0e-4)
    assert controls["warmup_epochs"] == 5
    assert controls["warmup_steps_per_epoch"] == 7
    assert controls["cosine_decay_enabled"] is True
    assert controls["cosine_decay_total_epochs"] == 128
    assert controls["cosine_decay_total_epochs_defaulted_to_run_epochs"] is True
    assert controls["cosine_decay_min_lr_ratio"] == pytest.approx(5.0e-2)


@pytest.mark.parametrize(
    "execute_fn",
    (
        runner_mod.execute_pact_nerv_vq_mlx_smoke_and_adapt,
        runner_mod.execute_pact_nerv_selector_v4_mlx_smoke_and_adapt,
    ),
)
def test_pact_compact_execute_refuses_unsupported_weight_decay(
    tmp_path: Path,
    execute_fn,
) -> None:
    with pytest.raises(
        CompactRendererMlxSpineRunnerError,
        match="optimizer-weight-decay is only supported",
    ):
        execute_fn(
            output_dir=tmp_path / execute_fn.__name__,
            num_pairs=1,
            epochs=1,
            batch_pair_indices_per_step=1,
            learning_rate=1e-3,
            source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
            optimizer_kind="adam",
            optimizer_weight_decay=1.0e-4,
            repo_root=REPO_ROOT,
        )


def test_default_source_video_resolves_from_external_upstream(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "clean_worktree"
    upstream = tmp_path / "canonical_upstream"
    video = upstream / "videos" / "0.mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    repo_root.mkdir()

    resolved = _resolve_source_video_path(
        runner_mod.DEFAULT_SOURCE_VIDEO_PATH,
        base=repo_root,
        upstream_dir=upstream,
    )

    assert resolved == video.resolve(strict=False)


def test_default_scorer_upstream_resolves_from_canonical_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "clean_worktree"
    repo_root.mkdir()
    canonical_upstream = tmp_path / "canonical_upstream"
    (canonical_upstream / "models").mkdir(parents=True)
    (canonical_upstream / "modules.py").write_text("# upstream\n", encoding="utf-8")
    (canonical_upstream / "models" / "posenet.safetensors").write_bytes(b"pose")
    (canonical_upstream / "models" / "segnet.safetensors").write_bytes(b"seg")
    monkeypatch.setattr(
        runner_mod,
        "DEFAULT_UPSTREAM_DIR",
        repo_root / "upstream",
    )
    monkeypatch.setattr(
        runner_mod,
        "CANONICAL_UPSTREAM_FALLBACK_DIR",
        canonical_upstream,
    )

    resolved = runner_mod._resolve_scorer_upstream_dir(
        repo_root,
        runner_mod.DEFAULT_UPSTREAM_DIR,
    )

    assert resolved == canonical_upstream.resolve(strict=False)


def test_explicit_scorer_upstream_does_not_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "clean_worktree"
    repo_root.mkdir()
    explicit_upstream = tmp_path / "explicit_upstream"
    (explicit_upstream / "models").mkdir(parents=True)
    (explicit_upstream / "modules.py").write_text("# incomplete\n", encoding="utf-8")
    canonical_upstream = tmp_path / "canonical_upstream"
    (canonical_upstream / "models").mkdir(parents=True)
    (canonical_upstream / "modules.py").write_text("# upstream\n", encoding="utf-8")
    (canonical_upstream / "models" / "posenet.safetensors").write_bytes(b"pose")
    (canonical_upstream / "models" / "segnet.safetensors").write_bytes(b"seg")
    monkeypatch.setattr(
        runner_mod,
        "DEFAULT_UPSTREAM_DIR",
        repo_root / "upstream",
    )
    monkeypatch.setattr(
        runner_mod,
        "CANONICAL_UPSTREAM_FALLBACK_DIR",
        canonical_upstream,
    )

    resolved = runner_mod._resolve_scorer_upstream_dir(repo_root, explicit_upstream)

    assert resolved == explicit_upstream.resolve(strict=False)


def test_auto_joint_recon_pixel_weight_discovers_verified_pair_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write_verified_joint_recon_weight(root, pairs=32, name="weight_32")
    weight_600, manifest_600 = _write_verified_joint_recon_weight(
        root,
        pairs=600,
        name="weight_600",
    )
    monkeypatch.setattr(runner_mod, "DEFAULT_SSD_ROOTS", ())

    discovered, metadata = runner_mod._discover_joint_recon_pixel_weight_path(
        repo_root=root,
        num_pairs=600,
    )

    assert discovered == weight_600.resolve(strict=False)
    assert metadata["selected_manifest_path"] == manifest_600.as_posix()
    assert metadata["selected_weight_sha256"] == runner_mod._sha256_file(weight_600)
    assert metadata["candidate_count"] == 1
    assert metadata["score_claim"] is False


def test_auto_joint_recon_pixel_weight_refuses_missing_pair_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write_verified_joint_recon_weight(root, pairs=32, name="weight_32")
    monkeypatch.setattr(runner_mod, "DEFAULT_SSD_ROOTS", ())

    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError, match="600"):
        runner_mod._discover_joint_recon_pixel_weight_path(
            repo_root=root,
            num_pairs=600,
        )


def test_pact_vq_execute_parser_exposes_real_scorer_binding_flags() -> None:
    args = _parse_args(
        [
            "--execute-family",
            "pact_nerv_vq",
            "--upstream-dir",
            "canonical_upstream",
            "--segnet-distillation-weight",
            "0.25",
            "--pose-distillation-weight",
            "0.75",
            "--pose-distillation-loss",
            "huber",
            "--pose-distillation-huber-delta",
            "2.5",
            "--segnet-distillation-objective",
            "boundary_argmax_hinge",
            "--distillation-temperature",
            "1.5",
            "--segnet-tau-boundary",
            "0.8",
            "--segnet-hinge-margin",
            "1.25",
            "--distillation-device",
            "cpu",
            "--compact-decoder-codec",
            "int8_scale_bundled",
            "--coder-aware-qat",
            "--coder-qat-quant-bits",
            "4",
            "--coder-qat-quant-residual-weight",
            "0.001",
            "--coder-qat-magnitude-weight",
            "0.0001",
            "--coder-qat-delta-weight",
            "0.0002",
            "--coder-qat-c1a-entropy-weight",
            "0.0003",
            "--coder-qat-c1a-sigma",
            "0.35",
            "--coder-qat-c1a-sample-size",
            "64",
            "--hprc-queue-followup-report",
            "hprc_queue_followup_report.json",
        ]
    )

    assert args.segnet_distillation_weight == 0.25
    assert args.upstream_dir == Path("canonical_upstream")
    assert args.pose_distillation_weight == 0.75
    assert args.pose_distillation_loss == "huber"
    assert args.pose_distillation_huber_delta == 2.5
    assert args.segnet_distillation_objective == "boundary_argmax_hinge"
    assert args.distillation_temperature == 1.5
    assert args.segnet_tau_boundary == 0.8
    assert args.segnet_hinge_margin == 1.25
    assert args.distillation_device == "cpu"
    assert args.compact_decoder_codec == "int8_scale_bundled"
    assert args.coder_aware_qat is True
    assert args.coder_qat_quant_bits == 4
    assert args.coder_qat_quant_residual_weight == 0.001
    assert args.coder_qat_magnitude_weight == 0.0001
    assert args.coder_qat_delta_weight == 0.0002
    assert args.coder_qat_c1a_entropy_weight == 0.0003
    assert args.coder_qat_c1a_sigma == 0.35
    assert args.coder_qat_c1a_sample_size == 64
    assert args.hprc_queue_followup_report == [
        Path("hprc_queue_followup_report.json")
    ]
    assert args.allow_segnet_only_research is False


def test_real_scorer_distillation_requires_complete_upstream_snapshot(
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "upstream"
    (incomplete / "models").mkdir(parents=True)
    (incomplete / "modules.py").write_text("# synthetic upstream\n", encoding="utf-8")

    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError) as exc:
        _require_scorer_upstream_dir_for_distillation(
            upstream_dir=incomplete,
            segnet_distillation_weight=0.1,
            pose_distillation_weight=0.0,
        )

    msg = str(exc.value)
    assert "--upstream-dir" in msg
    assert "posenet.safetensors" in msg
    assert "segnet.safetensors" in msg


def test_selector_v4_execute_parser_exposes_real_family_controls() -> None:
    args = _parse_args(
        [
            "--execute-family",
            "pact_nerv_selector_v4",
            "--compact-selector-palette-size",
            "32",
            "--num-pairs",
            "600",
            "--segnet-distillation-weight",
            "0.2",
            "--pose-distillation-weight",
            "0.4",
            "--mlx-profile",
            "selector_section_value_profile.json",
        ]
    )

    assert args.execute_family == "pact_nerv_selector_v4"
    assert args.compact_selector_palette_size == 32
    assert args.num_pairs == 600
    assert args.segnet_distillation_weight == 0.2
    assert args.pose_distillation_weight == 0.4
    assert args.mlx_profile == [Path("selector_section_value_profile.json")]


def test_pr95_hnerv_execute_parser_exposes_public_archive_seed() -> None:
    args = _parse_args(
        [
            "--execute-family",
            "pr95_hnerv",
            "--pr95-source-archive",
            "public_pr95/archive.zip",
            "--num-pairs",
            "600",
            "--epochs",
            "2",
            "--run-receiver-proof",
            "--keep-receiver-proof-output",
            "--receiver-proof-timeout-seconds",
            "17",
        ]
    )

    assert args.execute_family == "pr95_hnerv"
    assert args.pr95_source_archive == Path("public_pr95/archive.zip")
    assert args.num_pairs == 600
    assert args.epochs == 2
    assert args.run_receiver_proof is True
    assert args.keep_receiver_proof_output is True
    assert args.receiver_proof_timeout_seconds == 17


def test_hinerv_snerv_execute_parser_accepts_planner_gated_families() -> None:
    hi = _parse_args(
        [
            "--execute-family",
            "hi_nerv",
            "--num-pairs",
            "32",
            "--run-local-cpu-replay",
            "--keep-local-replay-inflated",
            "--retain-failed-local-replay-scratch",
            "--recon-pixel-weight-path",
            "weights.npz",
            "--auto-joint-recon-pixel-weight",
            "--auto-segnet-boundary-recon-weight",
            "--recon-pixel-weight-tau",
            "0.75",
            "--recon-pixel-weight-normalize",
            "none",
            "--recon-loss-stage-weight",
            "0.25",
            "--segnet-loss-stage-weight",
            "2.0",
            "--pose-loss-stage-weight",
            "1.5",
            "--mlx-prefilter-scorer-batch-pairs",
            "8",
            "--mlx-prefilter-scorer-device",
            "gpu",
            "--mlx-prefilter-progress-every",
            "10",
            "--telemetry-flush-interval-epochs",
            "1",
            "--run-post-export-materializers",
            "--post-export-materializer-max-steps",
            "3",
            "--post-export-materializer-max-parallel",
            "2",
            "--post-export-materializer-max-experiments",
            "1",
            "--modelsize-candidate-id",
            "manual",
            "--planner-row-id",
            "hi_nerv::manual::lion",
            "--optimizer-kind",
            "lion",
        ]
    )
    sn = _parse_args(
        [
            "--execute-family",
            "snerv",
            "--num-pairs",
            "128",
            "--coder-aware-qat",
            "--coder-qat-quant-bits",
            "4",
            "--snerv-scorer-loop-max-trials",
            "5",
            "--snerv-scorer-loop-search-mode",
            "learned_random_subspace",
            "--snerv-scorer-loop-byte-pressure-multiplier",
            "1.25",
            "--snerv-scorer-loop-section-value-pressure-multiplier",
            "1.75",
            "--snerv-scorer-loop-lf-payload-codec",
            "auto",
            "--snerv-scorer-loop-pose-slack",
            "0.001",
            "--snerv-scorer-loop-seg-slack",
            "0.002",
            "--snerv-scorer-loop-pair-stride",
            "3",
            "--snerv-scorer-loop-start-pair",
            "7",
            "--snerv-spectra-preserving-adapter",
            "--snerv-mfu-scales",
            "1,3",
            "--snerv-model-size-adapter",
            "snerv_manual_unit_adapter",
            "--snerv-fc-dim",
            "13",
            "--snerv-emb-size",
            "5",
            "--snerv-patch-radius",
            "2",
            "--snerv-hfr-gain",
            "0.25",
            "--snerv-temporal-context",
            "4",
            "--planner-row-id",
            "snerv::manual::native_rate_aware_training",
            "--skip-snerv-native-mlx-export",
            "--snerv-native-mlx-receiver-proof-timeout",
            "123",
            "--snerv-native-mlx-decoder-train-steps",
            "7",
            "--snerv-native-mlx-decoder-train-lr",
            "0.0003",
            "--snerv-native-mlx-decoder-train-ridge",
            "0.000004",
            "--snerv-native-mlx-decoder-train-optimizer",
            "lion",
            "--snerv-score-aware-long-training-epochs",
            "17",
            "--snerv-score-aware-long-training-lr",
            "0.0025",
            "--snerv-score-aware-long-training-batch-pairs",
            "6",
            "--snerv-score-aware-long-training-optimizer",
            "lion",
            "--snerv-score-aware-long-training-grad-clip-max-norm",
            "0.75",
            "--snerv-score-aware-long-training-weight-decay",
            "-1",
            "--snerv-score-aware-long-training-eval-roundtrip-ste",
            "--snerv-score-aware-long-training-pr95-muon-policy",
            "every_stage",
        ]
    )

    assert hi.execute_family == "hi_nerv"
    assert hi.num_pairs == 32
    assert hi.run_local_cpu_replay is True
    assert hi.keep_local_replay_inflated is True
    assert hi.retain_failed_local_replay_scratch is True
    assert hi.recon_pixel_weight_path == Path("weights.npz")
    assert hi.auto_joint_recon_pixel_weight is True
    assert hi.auto_segnet_boundary_recon_weight is True
    assert hi.recon_pixel_weight_tau == 0.75
    assert hi.recon_pixel_weight_normalize == "none"
    assert hi.recon_loss_stage_weight == 0.25
    assert hi.segnet_loss_stage_weight == 2.0
    assert hi.pose_loss_stage_weight == 1.5
    assert hi.mlx_prefilter_scorer_batch_pairs == 8
    assert hi.mlx_prefilter_scorer_device == "gpu"
    assert hi.mlx_prefilter_progress_every == 10
    assert hi.telemetry_flush_interval_epochs == 1
    assert hi.run_post_export_materializers is True
    assert hi.post_export_materializer_max_steps == 3
    assert hi.post_export_materializer_max_parallel == 2
    assert hi.post_export_materializer_max_experiments == 1
    assert hi.modelsize_candidate_id == "manual"
    assert hi.planner_row_id == "hi_nerv::manual::lion"
    assert hi.optimizer_kind == "lion"
    assert sn.execute_family == "snerv"
    assert sn.num_pairs == 128
    assert sn.coder_aware_qat is True
    assert sn.coder_qat_quant_bits == 4
    assert sn.snerv_scorer_loop_max_trials == 5
    assert sn.snerv_scorer_loop_search_mode == "learned_random_subspace"
    assert sn.snerv_scorer_loop_byte_pressure_multiplier == 1.25
    assert sn.snerv_scorer_loop_section_value_pressure_multiplier == 1.75
    assert sn.snerv_scorer_loop_lf_payload_codec == "auto"
    assert sn.snerv_scorer_loop_pose_slack == 0.001
    assert sn.snerv_scorer_loop_seg_slack == 0.002
    assert sn.snerv_scorer_loop_pair_stride == 3
    assert sn.snerv_scorer_loop_start_pair == 7
    assert sn.snerv_spectra_preserving_adapter is True
    assert sn.snerv_mfu_scales == "1,3"
    assert sn.snerv_model_size_adapter == "snerv_manual_unit_adapter"
    assert sn.snerv_fc_dim == 13
    assert sn.snerv_emb_size == 5
    assert sn.snerv_patch_radius == 2
    assert sn.snerv_hfr_gain == 0.25
    assert sn.snerv_temporal_context == 4
    assert sn.planner_row_id == "snerv::manual::native_rate_aware_training"
    assert sn.skip_snerv_native_mlx_export is True
    assert sn.snerv_native_mlx_receiver_proof_timeout == 123
    assert sn.snerv_native_mlx_decoder_train_steps == 7
    assert sn.snerv_native_mlx_decoder_train_lr == 0.0003
    assert sn.snerv_native_mlx_decoder_train_ridge == 0.000004
    assert sn.snerv_native_mlx_decoder_train_optimizer == "lion"
    assert sn.snerv_score_aware_long_training_epochs == 17
    assert sn.snerv_score_aware_long_training_lr == 0.0025
    assert sn.snerv_score_aware_long_training_batch_pairs == 6
    assert sn.snerv_score_aware_long_training_optimizer == "lion"
    assert sn.snerv_score_aware_long_training_grad_clip_max_norm == 0.75
    assert sn.snerv_score_aware_long_training_weight_decay == -1.0
    assert sn.snerv_score_aware_long_training_eval_roundtrip_ste is True
    assert sn.snerv_score_aware_long_training_pr95_muon_policy == "every_stage"


def test_top_priority_family_cli_refuses_missing_planner_row(
    tmp_path: Path,
) -> None:
    out = tmp_path / "hinerv_missing_planner"
    rc = runner_mod.main(
        [
            "--execute-family",
            "hi_nerv",
            "--num-pairs",
            "32",
            "--epochs",
            "29650",
            "--output-dir",
            out.as_posix(),
            "--repo-root",
            REPO_ROOT.as_posix(),
        ]
    )

    assert rc == 0
    report_path = out / "compact_renderer_mlx_spine_runner_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "compact_carrier_planner_row_launch_refused"
    assert report["execute_family"] == "hi_nerv"
    assert report["training_executed"] is False
    assert report["trainer_launch_allowed"] is False
    assert "hi_nerv_planner_row_id_missing" in report["blockers"]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_main_from_snerv_advisory_forwards_cleanup_scratch_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "snerv_advisory.json"
    report_path.write_text(json.dumps({"schema": "unit.snerv"}), encoding="utf-8")
    source_video = tmp_path / "0.mkv"
    source_video.write_bytes(b"video")
    calls: list[dict[str, object]] = []

    def fake_adapt_snerv_advisory_report_to_spine(**kwargs):
        calls.append(dict(kwargs))
        out = Path(kwargs["output_dir"])
        runner_report = out / "compact_renderer_mlx_spine_runner_report.json"
        runner_report.parent.mkdir(parents=True, exist_ok=True)
        runner_report.write_text("{}", encoding="utf-8")
        return {
            "mode": "adapted_snerv_advisory_report_to_spine",
            "report_path": runner_report.as_posix(),
            "blockers": ["unit_not_exact_ready"],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(runner_mod, "_acquire_active_campaign_lock", lambda **_: None)
    monkeypatch.setattr(
        runner_mod,
        "adapt_snerv_advisory_report_to_spine",
        fake_adapt_snerv_advisory_report_to_spine,
    )

    base_argv = [
        "--from-snerv-advisory-report",
        report_path.as_posix(),
        "--source-video-path",
        source_video.as_posix(),
        "--upstream-dir",
        (tmp_path / "upstream").as_posix(),
        "--repo-root",
        tmp_path.as_posix(),
    ]

    assert (
        runner_mod.main(
            [
                *base_argv,
                "--output-dir",
                (tmp_path / "default_cleanup").as_posix(),
            ]
        )
        == 0
    )
    assert calls[-1]["cleanup_failed_local_replay_scratch"] is True
    assert json.loads(capsys.readouterr().out)["ready_for_exact_eval_dispatch"] is False

    assert (
        runner_mod.main(
            [
                *base_argv,
                "--output-dir",
                (tmp_path / "retain_cleanup").as_posix(),
                "--retain-failed-local-replay-scratch",
            ]
        )
        == 0
    )
    assert calls[-1]["cleanup_failed_local_replay_scratch"] is False


def test_post_export_materializer_executor_runs_output_scoped_queue(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "queue_artifact.json"
    queue_path = tmp_path / "experiment_queue.json"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "unit_post_export_materializer",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
        "experiments": [
            {
                "id": "exp",
                "lane_id": "unit",
                "metadata": {
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "steps": [
                    {
                        "id": "write_artifact",
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import json, pathlib; "
                                f"pathlib.Path({artifact.as_posix()!r}).write_text("
                                "json.dumps({'schema':'post-export-unit.v1'}), "
                                "encoding='utf-8')"
                            ),
                        ],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": artifact.as_posix(),
                                "key": "schema",
                                "equals": "post-export-unit.v1",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    queue_path.write_text(json.dumps(queue, sort_keys=True), encoding="utf-8")
    queue_root = tmp_path / "queue_root"
    plan = {
        "schema": "compact_carrier_post_export_materializer_plan.v1",
        "compiled": True,
        "queue_id": "unit_post_export_materializer",
        "experiment_queue_path": queue_path.as_posix(),
        "queue_output_dir": queue_root.as_posix(),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    result = runner_mod._execute_carrier_post_export_materializer_plan(
        plan=plan,
        requested=True,
        max_steps=1,
        max_parallel=1,
        repo_root=REPO_ROOT,
    )

    assert result["schema"] == "compact_carrier_post_export_materializer_execution.v1"
    assert result["requested"] is True
    assert result["executed"] is True
    assert result["blockers"] == []
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["worker"]["success_count"] == 1
    assert result["worker"]["failure_count"] == 0
    assert result["worker"]["steps_started"] == 1
    assert Path(result["state_path"]).is_file()
    assert Path(result["log_root"]).is_dir()
    assert Path(result["execution_path"]).is_file()
    assert artifact.is_file()
    persisted = json.loads(Path(result["execution_path"]).read_text(encoding="utf-8"))
    assert persisted["state_path"] == result["state_path"]
    assert persisted["worker"]["success_count"] == 1


def test_post_export_materializer_executor_can_focus_one_chain(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    other = tmp_path / "other.json"
    queue_path = tmp_path / "experiment_queue.json"
    queue_root = tmp_path / "queue_root"
    handoff_dir = queue_root / "archive_zip_repack_v1_exact_eval_handoff"
    source_queue = handoff_dir / "source_queue.json"
    harvest_report = handoff_dir / "harvest_report.json"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "unit_post_export_chain_focus",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
        "experiments": [
            {
                "id": "archive_zip_repack_chain",
                "steps": [
                    {
                        "id": "materialize_local_proof_chain",
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import json, pathlib; "
                                f"pathlib.Path({first.as_posix()!r}).write_text("
                                "json.dumps({'schema':'first.v1'}), "
                                "encoding='utf-8')"
                            ),
                        ],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": first.as_posix(),
                                "key": "schema",
                                "equals": "first.v1",
                            }
                        ],
                    },
                    {
                        "id": "harvest_materializer_chains",
                        "requires": ["materialize_local_proof_chain"],
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import json, pathlib; "
                                f"pathlib.Path({handoff_dir.as_posix()!r}).mkdir("
                                "parents=True, exist_ok=True); "
                                f"pathlib.Path({second.as_posix()!r}).write_text("
                                "json.dumps({'schema':'second.v1'}), "
                                "encoding='utf-8'); "
                                f"pathlib.Path({source_queue.as_posix()!r}).write_text("
                                "json.dumps({'schema':'optimizer_candidate_queue_v1'}), "
                                "encoding='utf-8'); "
                                f"pathlib.Path({harvest_report.as_posix()!r}).write_text("
                                "json.dumps({'schema':'materializer_chain_harvest_report.v1', "
                                "'score_claim': False, "
                                "'ready_for_exact_eval_dispatch': False}), "
                                "encoding='utf-8')"
                            ),
                        ],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": second.as_posix(),
                                "key": "schema",
                                "equals": "second.v1",
                            }
                        ],
                    },
                ],
            },
            {
                "id": "packet_member_zip_header_elide_chain",
                "steps": [
                    {
                        "id": "materialize_local_proof_chain",
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import json, pathlib; "
                                f"pathlib.Path({other.as_posix()!r}).write_text("
                                "json.dumps({'schema':'other.v1'}), "
                                "encoding='utf-8')"
                            ),
                        ],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": other.as_posix(),
                                "key": "schema",
                                "equals": "other.v1",
                            }
                        ],
                    }
                ],
            },
        ],
    }
    queue_path.write_text(json.dumps(queue, sort_keys=True), encoding="utf-8")
    plan = {
        "schema": "compact_carrier_post_export_materializer_plan.v1",
        "compiled": True,
        "queue_id": "unit_post_export_chain_focus",
        "experiment_queue_path": queue_path.as_posix(),
        "experiment_queue_state_path": (queue_root / "experiment_queue.sqlite").as_posix(),
        "queue_output_dir": queue_root.as_posix(),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    result = runner_mod._execute_carrier_post_export_materializer_plan(
        plan=plan,
        requested=True,
        max_steps=2,
        max_parallel=1,
        max_experiments=1,
        repo_root=REPO_ROOT,
    )

    assert result["blockers"] == []
    assert result["max_experiments"] == 1
    assert result["worker"]["steps_started"] == 2
    assert result["worker"]["started_experiment_ids"] == ["archive_zip_repack_chain"]
    assert [row["ready_step"]["step_id"] for row in result["worker"]["step_results"]] == [
        "materialize_local_proof_chain",
        "harvest_materializer_chains",
    ]
    assert first.is_file()
    assert second.is_file()
    assert not other.exists()
    handoff_summary = result["handoff_summary"]
    assert handoff_summary["handoff_count"] == 1
    handoff = handoff_summary["rows"][0]
    assert handoff["target_kind"] == "archive_zip_repack_v1"
    assert handoff["source_queue_path"] == source_queue.as_posix()
    assert handoff["harvest_report_path"] == harvest_report.as_posix()
    assert handoff["source_queue_schema"] == "optimizer_candidate_queue_v1"
    assert handoff["harvest_report_schema"] == "materializer_chain_harvest_report.v1"
    assert handoff["ready_for_exact_eval_dispatch"] is False


def test_post_export_materializer_sweep_feedback_keeps_byte_saving_atoms(
    tmp_path: Path,
) -> None:
    positive = tmp_path / "archive_zip_repack_v1" / "sweep.json"
    positive.parent.mkdir(parents=True)
    positive.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_materializer_empirical_sweep.v1",
                "target_kind": "archive_zip_repack_v1",
                "observation_count": 1,
                "rate_positive_count": 1,
                "rate_nonpositive_count": 0,
                "max_saved_bytes": 979,
                "total_positive_saved_bytes": 979,
                "planner_feedback": {
                    "recommended_acquisition_rule": (
                        "rank_rate_positive_materializer_after_inflate_parity"
                    )
                },
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    zero = tmp_path / "packet_member_zip_header_elide_v1" / "sweep.json"
    zero.parent.mkdir(parents=True)
    zero.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_materializer_empirical_sweep.v1",
                "target_kind": "packet_member_zip_header_elide_v1",
                "observation_count": 1,
                "rate_positive_count": 0,
                "rate_nonpositive_count": 1,
                "max_saved_bytes": 0,
                "total_positive_saved_bytes": 0,
                "planner_feedback": {
                    "recommended_acquisition_rule": (
                        "demote_packet_member_zip_header_elide_v1_for_matching_archive_class"
                    )
                },
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    summary = runner_mod._post_export_materializer_sweep_feedback_summary(tmp_path)

    assert summary["schema"] == (
        "compact_carrier_post_export_sweep_feedback_summary.v1"
    )
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["byte_saving_sweep_count"] == 1
    assert summary["zero_save_sweep_count"] == 1
    assert summary["total_positive_saved_bytes"] == 979
    assert summary["retain_target_kinds"] == ["archive_zip_repack_v1"]
    assert summary["zero_save_target_kinds"] == [
        "packet_member_zip_header_elide_v1"
    ]
    assert summary["recommended_global_rule"] == (
        "retain_and_order_byte_saving_atoms_before_demoting_full_lane"
    )
    dispositions = {
        row["target_kind"]: row["full_stack_chain_disposition"]
        for row in summary["rows"]
    }
    assert dispositions["archive_zip_repack_v1"] == (
        "retain_byte_saving_atom_for_ordered_chain_solver"
    )
    assert dispositions["packet_member_zip_header_elide_v1"] == (
        "demote_only_matching_zero_save_archive_class"
    )


def test_recon_pixel_weight_loader_records_file_custody(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "joint_p18_p19_weight.npz"
    np.savez(
        weight_path,
        weight=np.ones((384, 512, 1), dtype=np.float32),
    )

    weight, metadata = runner_mod._load_recon_pixel_weight(
        weight_path,
        base=tmp_path,
        normalize="mean",
    )

    assert weight.shape == (384, 512, 1)
    assert metadata["schema"] == "compact_recon_pixel_weight.v1"
    assert metadata["enabled"] is True
    assert metadata["source_kind"] == "file"
    assert metadata["path"] == weight_path.as_posix()
    assert metadata["sha256"] == runner_mod._sha256_file(weight_path)
    assert metadata["npz_key"] == "weight"
    assert metadata["normalize"] == "mean"
    assert metadata["scorer_terms"] == {
        "p18_segnet": "caller_supplied",
        "p19_posenet": "caller_supplied",
    }
    assert metadata["stats"]["shape"] == [384, 512, 1]
    assert metadata["stats"]["nonzero_fraction"] == 1.0
    assert metadata["producer_manifest"]["status"] == (
        "not_found_unverified_manual_or_legacy_weight"
    )
    assert metadata["producer_manifest"]["consumption_certified"] is False
    assert metadata["authority"] == "false_macos_mlx_research_signal"


def test_recon_pixel_weight_loader_accepts_pair_frame_map(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "joint_p18_p19_pair_frame_weight.npz"
    weight = np.ones((3, 2, 384, 512, 1), dtype=np.float32)
    weight[1, 0, :, :, 0] = 2.0
    np.savez_compressed(weight_path, weight=weight)

    loaded, metadata = runner_mod._load_recon_pixel_weight(
        weight_path,
        base=tmp_path,
        expected_pairs=3,
        normalize="mean",
    )

    assert loaded.shape == (3, 2, 384, 512, 1)
    assert metadata["schema"] == "compact_recon_pixel_weight.v1"
    assert metadata["enabled"] is True
    assert metadata["source_kind"] == "file"
    assert metadata["expected_pairs"] == 3
    assert metadata["stats"]["shape"] == [3, 2, 384, 512, 1]
    assert metadata["sha256"] == runner_mod._sha256_file(weight_path)


def test_recon_pixel_weight_loader_carries_verified_gradient_manifest(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "joint_p18_p19_recon_pixel_weight.npz"
    np.savez_compressed(
        weight_path,
        weight=np.ones((2, 2, 384, 512, 1), dtype=np.float32),
    )
    health = {
        "schema": "joint_recon_pixel_weight_gradient_health.v1",
        "surface_generation_backend": "torch_exact_cpu_scorer_vjp.v1",
        "component_count": 14,
        "components_with_nonfinite": 0,
        "total_nonfinite_values": 0,
        "sanitized_components": [],
        "status": "pass_finite",
        "consumption_recommended": True,
    }
    (tmp_path / "joint_p18_p19_recon_pixel_weight_manifest.json").write_text(
        json.dumps(
            {
                "schema": "joint_p18_p19_recon_pixel_weight_manifest.v1",
                "weight_path": weight_path.as_posix(),
                "weight_sha256": runner_mod._sha256_file(weight_path),
                "metadata": {
                    "schema": "joint_p18_p19_recon_pixel_weight.v1",
                    "surface_generation_backend": "torch_exact_cpu_scorer_vjp.v1",
                    "gradient_health": health,
                    "blockers": [],
                    "training_consumption_recommended": True,
                },
            }
        ),
        encoding="utf-8",
    )

    _, metadata = runner_mod._load_recon_pixel_weight(
        weight_path,
        base=tmp_path,
        expected_pairs=2,
        normalize="mean",
    )

    producer = metadata["producer_manifest"]
    assert producer["status"] == "verified_finite_gradient_manifest"
    assert producer["consumption_certified"] is True
    assert producer["weight_sha256"] == runner_mod._sha256_file(weight_path)
    assert producer["gradient_health"] == health


def test_recon_pixel_weight_loader_refuses_stale_manifest_without_gradient_health(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "joint_p18_p19_recon_pixel_weight.npz"
    np.savez_compressed(
        weight_path,
        weight=np.ones((2, 2, 384, 512, 1), dtype=np.float32),
    )
    (tmp_path / "joint_p18_p19_recon_pixel_weight_manifest.json").write_text(
        json.dumps(
            {
                "schema": "joint_p18_p19_recon_pixel_weight_manifest.v1",
                "weight_path": weight_path.as_posix(),
                "weight_sha256": runner_mod._sha256_file(weight_path),
                "metadata": {
                    "schema": "joint_p18_p19_recon_pixel_weight.v1",
                    "blockers": [],
                    "training_consumption_recommended": True,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError) as exc:
        runner_mod._load_recon_pixel_weight(
            weight_path,
            base=tmp_path,
            expected_pairs=2,
            normalize="mean",
        )

    assert "missing gradient_health" in str(exc.value)


def test_recon_pixel_weight_loader_rejects_pair_count_mismatch(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "joint_p18_p19_pair_frame_weight.npy"
    np.save(weight_path, np.ones((2, 2, 384, 512, 1), dtype=np.float32))

    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError) as exc:
        runner_mod._load_recon_pixel_weight(
            weight_path,
            base=tmp_path,
            expected_pairs=3,
            normalize="mean",
        )

    assert "pair count" in str(exc.value)


def test_recon_pixel_weight_loader_fails_closed_on_bad_shape(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "wrong_shape.npy"
    np.save(weight_path, np.ones((192, 256), dtype=np.float32))

    with pytest.raises(runner_mod.CompactRendererMlxSpineRunnerError) as exc:
        runner_mod._load_recon_pixel_weight(
            weight_path,
            base=tmp_path,
            normalize="mean",
        )

    assert "spatial shape" in str(exc.value)


def test_hinerv_full_coverage_execute_runs_local_cpu_replay_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    staged_calls: list[dict[str, object]] = []

    def fake_stage_local_replay_submission(**kwargs):
        staged_calls.append(kwargs)
        staged = Path(kwargs["output_dir"]) / "submission"
        staged.mkdir(parents=True)
        (staged / "archive.zip").write_bytes(b"archive")
        (staged / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return staged

    class FakeReplaySummary:
        def to_json(self) -> str:
            return json.dumps(
                {
                    "schema": "local_submission_replay.v1",
                    "evaluation_passed": True,
                    "device": "cpu",
                    "axis_tag": "[macOS-CPU advisory]",
                    "local_score_estimate": 1.234,
                    "blockers": [],
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            )

    replay_calls: list[dict[str, object]] = []

    def fake_run_local_submission_replay(**kwargs):
        replay_calls.append(kwargs)
        return FakeReplaySummary()

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "stage_local_replay_submission",
        fake_stage_local_replay_submission,
    )
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        fake_run_local_submission_replay,
    )
    mlx_profile_path = tmp_path / "full_video_mlx_profile.json"
    mlx_profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "max_pairs": 600,
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": "executed"},
                "score_components": {"canonical_score": 0.1},
                "section_value_rows": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_gate",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        mlx_profile_paths=(mlx_profile_path,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        mlx_prefilter_scorer_batch_pairs=4,
        mlx_prefilter_progress_every=7,
        optimizer_kind="lion",
        upstream_dir=tmp_path / "canonical_upstream",
        repo_root=REPO_ROOT,
    )

    assert Path(captured_train_kwargs["scorer_upstream_dir"]) == (
        tmp_path / "canonical_upstream"
    )
    assert captured_train_kwargs["mlx_prefilter_scorer_batch_pairs"] == 4
    assert captured_train_kwargs["mlx_prefilter_progress_every"] == 7
    assert captured_train_kwargs["optimizer_kind"] == "lion"
    optimizer_controls = captured_train_kwargs["optimizer_controls"]
    assert optimizer_controls["optimizer_kind"] == "lion"
    assert optimizer_controls["weight_decay_effective"] == pytest.approx(1.0e-4)
    assert optimizer_controls["weight_decay_defaulted"] is True
    assert optimizer_controls["grad_clip_max_norm"] == pytest.approx(1.0)
    optimizer_policy = captured_train_kwargs["hi_nerv_optimizer_policy"]
    assert optimizer_policy["resolved_policy"] == "native_optimizer"
    assert optimizer_policy["optimizer_kind_consumed_by_native_mlx"] is True
    assert optimizer_policy["pr95_faithful_curriculum_enabled"] is False
    assert out["score_aware_training"]["optimizer_policy"] == optimizer_policy
    assert out["score_aware_training"]["optimizer_controls"] == optimizer_controls
    assert out["score_aware_training"]["local_mlx_prefilter"] == {
        "schema": "compact_hi_nerv_local_mlx_prefilter_config.v1",
        "scorer_device": "cpu",
        "scorer_batch_pairs": 4,
        "progress_every": 7,
        "singleton_required_for_local_cpu_replay_unlock": True,
        "gpu_profiles_are_prefilter_only": False,
        "batched_profiles_are_prefilter_only": True,
        "authority": "macos_mlx_research_signal_false_authority",
    }
    assert staged_calls
    assert replay_calls
    assert out["local_cpu_replay_gate"]["executed"] is True
    assert out["local_cpu_replay_gate"]["default_enabled_for_full_coverage"] is True
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is True
    assert out["local_cpu_replay_gate"]["local_replay_mlx_prefilter_passed"] is True
    assert out["mlx_prefilter_coverage"]["has_full_video_mlx_prefilter"] is True
    assert out["mlx_prefilter_coverage"]["local_replay_mlx_prefilter_passed"] is True
    assert out["mlx_prefilter_coverage"]["blockers"] == []
    assert out["local_cpu_replay_summary"]["axis_tag"] == "[macOS-CPU advisory]"
    assert out["local_cpu_replay_summary"]["score_claim"] is False
    assert out["local_cpu_replay_summary_paths"]
    assert Path(out["local_cpu_replay_summary_paths"][0]).is_file()
    post_export = out["post_export_materializer_plan"]
    assert post_export["schema"] == "compact_carrier_post_export_materializer_plan.v1"
    assert post_export["compiled"] is True
    assert post_export["queue_launch_executed"] is False
    assert post_export["experiment_count"] > 0
    assert Path(post_export["experiment_queue_path"]).is_file()
    assert post_export["archive_record"]["source_runtime_dir"].endswith(
        "/hi_nerv_mlx_training/submission"
    )
    assert post_export["archive_record"]["source_inflate_sh_path"].endswith(
        "/hi_nerv_mlx_training/submission/inflate.sh"
    )
    contexts = json.loads(Path(post_export["materializer_contexts_path"]).read_text())
    first_context = contexts["rows"][0]["context"]
    assert first_context["source_runtime_dir"].endswith(
        "/hi_nerv_mlx_training/submission"
    )
    assert first_context["packet_member_merge_source_runtime_dir"].endswith(
        "/hi_nerv_mlx_training/submission"
    )
    queue = json.loads(Path(post_export["experiment_queue_path"]).read_text())
    assert queue["schema"] == "experiment_queue.v1"
    assert post_export["experiment_queue_state_path"].endswith(
        "/experiment_queue.sqlite"
    )
    harvest_step = queue["experiments"][0]["steps"][1]
    state_arg_index = harvest_step["command"].index("--state") + 1
    assert harvest_step["command"][state_arg_index] == post_export[
        "experiment_queue_state_path"
    ]
    assert post_export["score_claim"] is False
    assert post_export["ready_for_exact_eval_dispatch"] is False
    post_export_execution = out["post_export_materializer_execution"]
    assert post_export_execution["requested"] is False
    assert post_export_execution["executed"] is False
    assert post_export_execution["mode"] == "compile_only_execution_not_requested"
    assert Path(post_export_execution["execution_path"]).is_file()
    assert "local_cpu_replay_not_executed" not in out["blockers"]
    assert "local_cpu_replay_not_run_partial_pair_coverage" not in out["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in out["blockers"]


def test_hinerv_long_campaign_refuses_when_pr95_prelaunch_gate_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_train(**_kwargs):
        raise AssertionError("long campaign should refuse before MLX training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_refusal",
        num_pairs=600,
        epochs=8,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "hi_nerv_pr95_binding_prelaunch_refused"
    assert out["training_executed"] is False
    assert "pr95_long_campaign_prelaunch_gate_failed" in out["blockers"]
    assert "hi_nerv_modelsize_archive_budget_missing" in out["blockers"]
    gate = out["candidate_curriculum_plan"]["long_campaign_prelaunch_gate"]
    assert gate["launch_allowed"] is False
    assert "receiver_proof" in gate["post_run_requirements_excluded"]
    assert out["candidate_feedback"]["row"][
        "long_campaign_prelaunch_launch_allowed"
    ] is False
    assert Path(out["report_path"]).is_file()


def test_hinerv_trained_archive_byte_oracle_writes_receiver_closed_ladder(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"hi-nerv-byte-closed-candidate")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    proof = tmp_path / "receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_receiver_proof.v1",
                "receiver_archive_replay_verified": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    oracle = runner_mod._write_hi_nerv_trained_archive_byte_oracle(
        output_dir=tmp_path,
        artifact_dict={
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
        },
        modelsize_candidate={
            "candidate_id": "hi_nerv_oracle_tiny",
            "modelsize_mparams": 0.02,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 40_000,
        },
        hard_byte_ceilings=(178_000,),
        num_pairs=2,
        receiver_proof_path=proof,
        local_cpu_replay_summary={
            "axis_tag": "[macOS-CPU advisory]",
            "score_claim": False,
            "blockers": [],
        },
        mlx_prefilter_coverage={"has_full_video_mlx_prefilter": False},
        repo_root=REPO_ROOT,
    )

    oracle_path = Path(oracle["path"])
    ladder_path = Path(oracle["receiver_closed_modelsize_ladder_path"])
    assert oracle_path.is_file()
    assert ladder_path.is_file()
    assert oracle["schema"] == "hi_nerv_trained_archive_byte_oracle.v1"
    assert oracle["row"]["receiver_proof_passed"] is True
    assert oracle["row"]["measured_archive_bytes"] == archive.stat().st_size
    assert oracle["row"]["archive_sha256"] == archive_sha
    assert oracle["measured_byte_cap_report"]["archive_zip_bytes"] == archive.stat().st_size
    assert (
        oracle["measured_byte_cap_report"][
            "delta_bytes_vs_tightest_hard_byte_ceiling"
        ]
        == archive.stat().st_size - 178_000
    )
    assert (
        oracle["row"]["measured_archive_bytes_under_tightest_hard_ceiling"]
        is True
    )
    assert "hi_nerv_trained_archive_byte_oracle_partial_pair_scope" in oracle[
        "blockers"
    ]
    assert "hi_nerv_local_cpu_replay_not_contest_auth_axis" in oracle["blockers"]
    assert oracle["receiver_closed_modelsize_ladder"]["schema"] == (
        "nerv_receiver_closed_modelsize_ladder.v1"
    )
    assert oracle["score_claim"] is False
    assert oracle["ready_for_exact_eval_dispatch"] is False


def test_hinerv_trained_archive_byte_oracle_blocks_measured_over_cap(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"x" * 64)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    proof = tmp_path / "receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_receiver_proof.v1",
                "receiver_archive_replay_verified": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    oracle = runner_mod._write_hi_nerv_trained_archive_byte_oracle(
        output_dir=tmp_path,
        artifact_dict={
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
        },
        modelsize_candidate={
            "candidate_id": "hi_nerv_oracle_overcap",
            "modelsize_mparams": 0.02,
            "hard_byte_ceiling": 32,
            "nominal_total_payload_bytes": 16,
        },
        hard_byte_ceilings=(32,),
        num_pairs=600,
        receiver_proof_path=proof,
        local_cpu_replay_summary={
            "axis_tag": "[contest-CPU]",
            "score_claim": False,
            "blockers": [],
        },
        mlx_prefilter_coverage={"has_full_video_mlx_prefilter": True},
        repo_root=REPO_ROOT,
    )

    report = oracle["measured_byte_cap_report"]
    assert report["archive_zip_bytes"] == 64
    assert report["delta_bytes_vs_tightest_hard_byte_ceiling"] == 32
    assert report["under_tightest_hard_byte_ceiling"] is False
    assert "measured_archive_bytes_exceed_tightest_hard_ceiling" in report[
        "blockers"
    ]
    assert "measured_archive_bytes_exceed_tightest_hard_ceiling" in oracle[
        "blockers"
    ]
    assert oracle["feedback_ready"] is False


def test_hinerv_refuses_unscored_launch_but_consumes_modelsize_ladder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_train(**_kwargs):
        raise AssertionError("unscored HiNeRV launch must refuse before training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)
    ladder = tmp_path / "hinerv_receiver_closed_ladder.json"
    ladder.write_text(
        json.dumps(
            {
                "schema": "nerv_receiver_closed_modelsize_ladder.v1",
                "carrier_id": "hi_nerv",
                "modelsize_budget_rows": [
                    {
                        "row_id": "tiny",
                        "archive_bytes": 40_000,
                        "archive_sha256": "a" * 64,
                        "nonrate_score": 95.0,
                        "modelsize_mparams": 0.02,
                        "fc_dim": 8,
                        "receiver_closed": True,
                        "receiver_proof_passed": True,
                        "receiver_archive_replay_verified": True,
                        "receiver_proof_path": "proofs/tiny.json",
                        "receiver_proof_sha256": "b" * 64,
                        "axis_tag": "[macOS-CPU advisory]",
                        "num_pairs": 600,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    },
                    {
                        "row_id": "small",
                        "archive_bytes": 80_000,
                        "archive_sha256": "c" * 64,
                        "nonrate_score": 80.0,
                        "modelsize_mparams": 0.04,
                        "fc_dim": 16,
                        "receiver_closed": True,
                        "receiver_proof_passed": True,
                        "receiver_archive_replay_verified": True,
                        "receiver_proof_path": "proofs/small.json",
                        "receiver_proof_sha256": "d" * 64,
                        "axis_tag": "[macOS-CPU advisory]",
                        "num_pairs": 600,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    },
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_refusal",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        modelsize_budget_json_paths=(ladder,),
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "hi_nerv_mlx_scoreaware_launch_refused"
    assert out["training_executed"] is False
    assert "hi_nerv_real_segnet_teacher_missing" in out["blockers"]
    assert "hi_nerv_real_posenet_teacher_missing" in out["blockers"]
    assert out["modelsize_budget_evidence"]["row_count"] == 2
    assert out["modelsize_budget_evidence"]["sources"][0]["rows_added"] == 2
    plan = out["score_aware_carrier_training_plan"]
    assert plan["planner_action"] == "run_score_aware_decoder_weight_training_full_main"
    assert plan["modelsize_budget_receiver_closed_ready"] is True
    assert plan["evidence_summary"]["receiver_closed_selected_modelsize_archive_bytes"] == 80_000
    assert out["score_aware_training_config_gate"]["launch_allowed"] is False
    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert Path(out["report_path"]).is_file()


def test_hinerv_refusal_filters_modelsize_rows_without_nonrate_score(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_train(**_kwargs):
        raise AssertionError("invalid modelsize evidence must refuse before training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)
    bad_ladder = tmp_path / "hinerv_waterfill_without_nonrate.json"
    bad_ladder.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_ladder_waterfill.v1",
                "rows": [
                    {
                        "row_id": "hi_nerv_local_tiny",
                        "archive_bytes": 134_938,
                        "archive_sha256": "a" * 64,
                        "waterfill_summary": {"group_count": 24},
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_bad_modelsize_refusal",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        modelsize_budget_json_paths=(bad_ladder,),
        repo_root=REPO_ROOT,
    )

    source = out["modelsize_budget_evidence"]["sources"][0]
    assert out["mode"] == "hi_nerv_mlx_scoreaware_launch_refused"
    assert out["training_executed"] is False
    assert out["modelsize_budget_evidence"]["row_count"] == 0
    assert source["rows_seen"] == 1
    assert source["rows_added"] == 0
    assert source["rows_rejected"] == 1
    assert "modelsize_budget_json_rows_rejected" in source["blockers"]
    rejected = set(source["rejected_rows"][0]["blockers"])
    assert {
        "receiver_closed_modelsize_ladder_schema_required",
        "receiver_closed_byte_proof_missing",
        "receiver_proof_path_missing",
        "receiver_proof_sha256_missing_or_invalid",
        "receiver_proof_axis_tag_missing",
        "receiver_proof_full_sample_count_missing",
        "source_bound_modelsize_or_fc_dim_missing",
        "modelsize_budget_row_missing_nonrate_score",
    }.issubset(rejected)
    assert "hi_nerv_real_segnet_teacher_missing" in out["blockers"]
    assert "hi_nerv_real_posenet_teacher_missing" in out["blockers"]
    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False


def test_hinerv_refusal_rejects_raw_modelsize_budget_selected_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_train(**_kwargs):
        raise AssertionError("raw planning artifact must refuse before training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)
    raw_budget = tmp_path / "nerv_modelsize_budget.json"
    raw_budget.write_text(
        json.dumps(
            {
                "schema": "nerv_modelsize_budget.v1",
                "family": "hi_nerv",
                "selected_candidates": [
                    {
                        "candidate_id": "hinerv_np600_ld4_ed8_dc8_int4_mixed_ceil178000",
                        "archive_bytes": 72_000,
                        "nonrate_score": 0.21,
                        "modelsize_mparams": 0.032,
                        "receiver_closed": True,
                        "receiver_proof_passed": True,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
                "selected_candidate_count": 1,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_raw_modelsize_refusal",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        modelsize_budget_json_paths=(raw_budget,),
        repo_root=REPO_ROOT,
    )

    source = out["modelsize_budget_evidence"]["sources"][0]
    assert out["mode"] == "hi_nerv_mlx_scoreaware_launch_refused"
    assert out["training_executed"] is False
    assert out["modelsize_budget_evidence"]["row_count"] == 0
    assert source["source_schema"] == "nerv_modelsize_budget.v1"
    assert source["authority"] == (
        "planning_artifact_only_not_receiver_closed_ladder_evidence"
    )
    assert source["rows_seen"] == 1
    assert source["rows_added"] == 0
    assert source["rows_rejected"] == 1
    assert source["score_claim"] is False
    assert source["promotion_eligible"] is False
    assert source["ready_for_exact_eval_dispatch"] is False
    assert "raw_nerv_modelsize_budget_artifact_not_receiver_closed_ladder" in source[
        "blockers"
    ]
    assert "receiver_closed_modelsize_ladder_schema_required" in source["blockers"]
    assert source["rejected_rows"][0]["blockers"] == [
        "selected_candidates_are_planning_rows_not_receiver_closed_ladder",
        "receiver_closed_byte_proof_missing",
        "measured_receiver_archive_bytes_missing",
    ]
    plan = out["score_aware_carrier_training_plan"]
    assert plan["modelsize_budget_receiver_closed_ready"] is False
    assert "receiver_closed_modelsize_budget_ladder_missing" in plan[
        "dispatch_blockers"
    ]
    assert out["score_claim"] is False
    assert out["promotion_eligible"] is False
    assert out["ready_for_exact_eval_dispatch"] is False


def test_hinerv_refusal_rejects_canonical_score_only_modelsize_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_train(**_kwargs):
        raise AssertionError("score-only modelsize evidence must refuse before training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)
    score_only_ladder = tmp_path / "hinerv_score_only_ladder.json"
    score_only_ladder.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_modelsize_budget_plan.v2",
                "modelsize_budget_rows": [
                    {
                        "row_id": "score_only",
                        "archive_bytes": 120_000,
                        "canonical_score": 0.5,
                        "receiver_closed": True,
                        "receiver_proof_passed": True,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_score_only_refusal",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        modelsize_budget_json_paths=(score_only_ladder,),
        repo_root=REPO_ROOT,
    )

    source = out["modelsize_budget_evidence"]["sources"][0]
    assert out["mode"] == "hi_nerv_mlx_scoreaware_launch_refused"
    assert out["modelsize_budget_evidence"]["row_count"] == 0
    assert source["rows_seen"] == 1
    assert source["rows_rejected"] == 1
    rejected = set(source["rejected_rows"][0]["blockers"])
    assert {
        "receiver_closed_modelsize_ladder_schema_required",
        "modelsize_budget_row_missing_nonrate_score",
        "receiver_proof_path_missing",
        "receiver_proof_sha256_missing_or_invalid",
        "archive_sha256_missing_or_invalid",
        "receiver_proof_axis_tag_missing",
        "receiver_proof_full_sample_count_missing",
        "source_bound_modelsize_or_fc_dim_missing",
    }.issubset(rejected)


def test_hinerv_auto_mlx_prefilter_profile_unlocks_local_cpu_replay_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (out / "local_mlx_prefilter_profile.json").write_text(
            json.dumps(
                {
                    "schema": "hprc_mlx_component_neutralization_profile.v1",
                    "producer": "tac.local_acceleration.mlx_renderer_prefilter_profile",
                    "max_pairs": 600,
                    "num_pairs": 600,
                    "n_samples": 600,
                    "scorer_batch_pairs": 1,
                    "scope_status": {"full_video": "executed"},
                    "score_components": {"canonical_score": 0.1},
                    "mlx_response_summary": {
                        "batch_pairs": 1,
                        "max_pairs": 600,
                        "n_samples": 600,
                    },
                    "section_value_rows": [],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    replay_calls: list[dict[str, object]] = []

    def fake_stage_local_replay_submission(**kwargs):
        staged = Path(kwargs["output_dir"]) / "submission"
        staged.mkdir(parents=True)
        (staged / "archive.zip").write_bytes(b"archive")
        (staged / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return staged

    class FakeReplaySummary:
        def to_json(self) -> str:
            return json.dumps(
                {
                    "schema": "local_submission_replay.v1",
                    "evaluation_passed": True,
                    "device": "cpu",
                    "axis_tag": "[macOS-CPU advisory]",
                    "local_score_estimate": 1.234,
                    "blockers": [],
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            )

    def fake_run_local_submission_replay(**kwargs):
        replay_calls.append(kwargs)
        return FakeReplaySummary()

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "stage_local_replay_submission",
        fake_stage_local_replay_submission,
    )
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        fake_run_local_submission_replay,
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_gate",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert replay_calls
    assert out["auto_mlx_prefilter_profile_path"].endswith(
        "local_mlx_prefilter_profile.json"
    )
    assert out["mlx_profile_paths"] == [out["auto_mlx_prefilter_profile_path"]]
    assert out["local_cpu_replay_gate"]["executed"] is True
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is True
    assert out["local_cpu_replay_gate"]["local_replay_mlx_prefilter_passed"] is True
    assert "full_video_mlx_scorer_replay_not_attached" not in out["blockers"]
    assert "hi_nerv_full_video_local_prefilter_missing" not in out["blockers"]
    assert "hi_nerv_local_cpu_replay_gate_missing" not in out["blockers"]
    assert "hi_nerv_full_video_local_prefilter_missing" not in out[
        "candidate_feedback"
    ]["row"]["pr95_stack_binding_blockers"]
    assert "hi_nerv_local_cpu_replay_gate_missing" not in out["candidate_feedback"][
        "row"
    ]["pr95_stack_binding_blockers"]


def test_hinerv_execute_emits_trained_archive_byte_oracle_feedback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate = {
        "candidate_id": "unit_hinerv_candidate",
        "family": "hi_nerv",
        "num_pairs": 600,
        "latent_dim": 4,
        "embed_dim": 8,
        "decoder_channel": 8,
        "decoder_codec": "int4_mixed",
        "hard_byte_ceiling": 178_000,
        "nominal_total_payload_bytes": 70_000,
        "modelsize_mparams": 0.031,
        "use_hierarchical_feature_grid": True,
        "use_convnext_blocks": True,
    }

    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        archive_sha = runner_mod._sha256_file(archive)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        _write_hinerv_receiver_proof(
            out / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json",
            archive_bytes=archive.stat().st_size,
            archive_sha256=archive_sha,
        )
        (out / "local_mlx_prefilter_profile.json").write_text(
            json.dumps(
                {
                    "schema": "hprc_mlx_component_neutralization_profile.v1",
                    "producer": "tac.local_acceleration.mlx_renderer_prefilter_profile",
                    "max_pairs": 600,
                    "num_pairs": 600,
                    "n_samples": 600,
                    "scorer_batch_pairs": 1,
                    "scope_status": {"full_video": "executed"},
                    "score_components": {"canonical_score": 0.1},
                    "mlx_response_summary": {
                        "batch_pairs": 1,
                        "max_pairs": 600,
                        "n_samples": 600,
                    },
                    "section_value_rows": [],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
        }

    def fake_stage_local_replay_submission(**kwargs):
        staged = Path(kwargs["output_dir"]) / "submission"
        staged.mkdir(parents=True)
        (staged / "archive.zip").write_bytes(b"archive")
        (staged / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return staged

    class FakeReplaySummary:
        def to_json(self) -> str:
            return json.dumps(
                {
                    "schema": "local_submission_replay.v1",
                    "evaluation_passed": True,
                    "device": "cpu",
                    "axis_tag": "[macOS-CPU advisory]",
                    "local_score_estimate": 1.234,
                    "blockers": [],
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            )

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "stage_local_replay_submission",
        fake_stage_local_replay_submission,
    )
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        lambda **_kwargs: FakeReplaySummary(),
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_byte_oracle",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        modelsize_candidate=candidate,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    oracle = out["trained_archive_byte_oracle"]
    assert oracle["schema"] == "hi_nerv_trained_archive_byte_oracle.v1"
    assert oracle["measured_archive_bytes"] == out["archive_bytes"]
    assert oracle["row"]["receiver_proof_passed"] is True
    assert oracle["row"]["archive_bytes"] == out["archive_bytes"]
    assert oracle["row"]["nominal_total_payload_bytes"] == 70_000
    assert Path(oracle["path"]).is_file()
    assert Path(oracle["receiver_closed_modelsize_ladder_path"]).is_file()
    byte_feedback = out["candidate_curriculum_plan"]["byte_oracle_logging"]
    assert byte_feedback["byte_feedback_source"] == (
        "hi_nerv_trained_archive_byte_oracle"
    )
    assert byte_feedback["measured_archive_bytes"] == out["archive_bytes"]
    assert byte_feedback["trained_archive_byte_oracle_path"] == oracle["path"]
    assert out["candidate_feedback"]["row"]["measured_archive_bytes"] == out[
        "archive_bytes"
    ]
    assert out["score_claim"] is False
    assert out["promotion_eligible"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert "contest_cpu_cuda_exact_eval_not_executed" in out["blockers"]
    assert "hi_nerv_local_cpu_replay_not_contest_auth_axis" in out["blockers"]


def test_hinerv_auto_joint_recon_weight_flows_to_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight_path, _manifest_path = _write_verified_joint_recon_weight(
        tmp_path / "repo",
        pairs=2,
        name="weight_2",
    )
    discovery = {
        "schema": "compact_auto_joint_recon_pixel_weight_discovery.v1",
        "status": "selected_verified_joint_p18_p19_weight",
        "num_pairs": 2,
        "selected_weight_path": weight_path.as_posix(),
        "selected_weight_sha256": runner_mod._sha256_file(weight_path),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    captured: dict[str, object] = {}

    def fake_discover(**kwargs):
        assert kwargs["num_pairs"] == 2
        return weight_path, discovery

    def fake_train(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "score_aware_training": {
                    "recon_pixel_weight": {
                        "schema": "compact_recon_pixel_weight.v1",
                        "enabled": True,
                        "source_kind": "auto_discovered_joint_p18_p19_file",
                        "path": weight_path.as_posix(),
                        "auto_discovery": discovery,
                        "authority": "false_macos_mlx_research_signal",
                    }
                }
            },
        }

    monkeypatch.setattr(runner_mod, "_discover_joint_recon_pixel_weight_path", fake_discover)
    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_auto_joint",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        auto_joint_recon_pixel_weight=True,
        recon_loss_stage_weight=0.25,
        segnet_loss_stage_weight=2.0,
        pose_loss_stage_weight=1.5,
        mlx_prefilter_scorer_device="gpu",
        repo_root=REPO_ROOT,
    )

    assert captured["recon_pixel_weight_path"] == weight_path
    assert captured["recon_pixel_weight_auto_discovery"] == discovery
    assert captured["auto_segnet_boundary_recon_weight"] is False
    assert captured["recon_loss_stage_weight"] == 0.25
    assert captured["segnet_loss_stage_weight"] == 2.0
    assert captured["pose_loss_stage_weight"] == 1.5
    assert captured["mlx_prefilter_scorer_device"] == "gpu"
    assert out["score_aware_training"]["stage_loss_weights"] == {
        "distill": 2.0,
        "pose_distill": 1.5,
        "recon": 0.25,
    }
    assert out["score_aware_training"]["recon_pixel_weight"]["source_kind"] == (
        "auto_discovered_joint_p18_p19_file"
    )
    assert "hinerv_candidate_curriculum_recon_pixel_weight_missing" not in out[
        "blockers"
    ]


def test_hinerv_sampled_mlx_profile_does_not_unlock_default_cpu_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    replay_calls: list[dict[str, object]] = []

    def fake_run_local_submission_replay(**kwargs):
        replay_calls.append(kwargs)
        raise AssertionError("sampled MLX profile must not run default CPU replay")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        fake_run_local_submission_replay,
    )
    sampled_profile = tmp_path / "sampled_mlx_profile.json"
    sampled_profile.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "mlx_response_summary": {"max_pairs": 128, "n_samples": 128},
                "scorer_batch_pairs": 1,
                "scope_status": {
                    "full_video": "sampled_prefix_requires_full_video_rerun"
                },
                "section_value_rows": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_gate",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        mlx_profile_paths=(sampled_profile,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert replay_calls == []
    assert out["local_cpu_replay_gate"]["executed"] is False
    assert out["local_cpu_replay_gate"]["default_enabled_for_full_coverage"] is False
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is False
    assert out["mlx_prefilter_coverage"]["has_full_video_mlx_prefilter"] is False
    assert "local_cpu_replay_waiting_for_full_video_mlx_prefilter" in out[
        "blockers"
    ]
    assert "full_video_mlx_scorer_replay_not_attached" in out["blockers"]
    assert "sampled_mlx_prefilter_requires_full_video_rerun" in out["blockers"]


def test_hinerv_full_video_bad_mlx_score_does_not_unlock_default_cpu_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    replay_calls: list[dict[str, object]] = []

    def fake_run_local_submission_replay(**kwargs):
        replay_calls.append(kwargs)
        raise AssertionError("bad MLX prefilter score must not run default CPU replay")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        fake_run_local_submission_replay,
    )
    bad_profile = tmp_path / "bad_full_video_mlx_profile.json"
    bad_profile.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "max_pairs": 600,
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": "executed"},
                "score_components": {"canonical_score": 90.0},
                "section_value_rows": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_gate",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        mlx_profile_paths=(bad_profile,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert replay_calls == []
    assert out["local_cpu_replay_gate"]["executed"] is False
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is True
    assert out["local_cpu_replay_gate"]["local_replay_mlx_prefilter_passed"] is False
    assert out["mlx_prefilter_coverage"]["best_full_video_mlx_score"] == 90.0
    assert "local_cpu_replay_blocked_by_mlx_prefilter_score" in out["blockers"]
    assert "mlx_prefilter_score_not_below_local_replay_threshold" in out["blockers"]
    assert "mlx_score_above_hard_demote_threshold" in out["blockers"]


def test_hinerv_batched_full_video_mlx_prefilter_feeds_acquisition_not_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        _write_mlx_prefilter_profile(
            out / "local_mlx_prefilter_profile.json",
            pairs=600,
            batch_pairs=8,
            score=91.0,
        )
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    replay_calls: list[dict[str, object]] = []

    def fake_run_local_submission_replay(**kwargs):
        replay_calls.append(kwargs)
        raise AssertionError("batched MLX acquisition profile must not run CPU replay")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)
    monkeypatch.setattr(
        runner_mod,
        "run_local_submission_replay",
        fake_run_local_submission_replay,
    )

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_batched_gpu_gate",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert replay_calls == []
    assert out["local_cpu_replay_gate"]["executed"] is False
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is True
    assert out["local_cpu_replay_gate"]["local_replay_mlx_prefilter_passed"] is False
    assert out["mlx_prefilter_coverage"]["has_full_video_mlx_prefilter"] is True
    assert out["mlx_prefilter_coverage"]["local_replay_profile_paths"] == []
    assert out["mlx_prefilter_coverage"]["blockers"] == [
        "mlx_profile_batch_pairs_not_singleton"
    ]
    assert "full_video_mlx_scorer_replay_not_attached" not in out["blockers"]
    assert "hi_nerv_full_video_local_prefilter_missing" not in out["blockers"]
    assert "local_cpu_replay_waiting_for_full_video_mlx_prefilter" not in out[
        "blockers"
    ]
    assert "hi_nerv_local_cpu_replay_gate_missing" in out["blockers"]
    feedback_blockers = out["candidate_feedback"]["row"][
        "pr95_stack_binding_blockers"
    ]
    assert "hi_nerv_full_video_local_prefilter_missing" not in feedback_blockers
    assert "hi_nerv_local_cpu_replay_gate_missing" in feedback_blockers
    feedback_row = out["candidate_feedback"]["row"]
    assert feedback_row["mlx_prefilter_has_full_video"] is True
    assert feedback_row["mlx_prefilter_local_replay_passed"] is False
    assert feedback_row["mlx_prefilter_blockers"] == [
        "mlx_profile_batch_pairs_not_singleton"
    ]
    assert feedback_row["local_cpu_replay_gate_has_full_video_mlx_prefilter"] is True
    assert (
        feedback_row["local_cpu_replay_gate_local_replay_mlx_prefilter_passed"]
        is False
    )
    assert feedback_row["local_cpu_replay_gate_executed"] is False


def test_hinerv_full_coverage_waits_for_mlx_prefilter_before_default_cpu_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=600)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_gate",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        repo_root=REPO_ROOT,
    )

    assert out["local_cpu_replay_gate"]["executed"] is False
    assert out["local_cpu_replay_gate"]["default_enabled_for_full_coverage"] is False
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is False
    assert "local_cpu_replay_waiting_for_full_video_mlx_prefilter" in out[
        "blockers"
    ]


def test_hinerv_execute_threads_coder_qat_and_reads_verified_waterfill_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}
    weight_path = tmp_path / "joint_p18_p19_weight.npy"
    np.save(weight_path, np.ones((384, 512), dtype=np.float32))
    waterfill_plan_path = tmp_path / "hinerv_decoder_waterfill.json"
    waterfill_plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "hinerv-unit-candidate",
        "group_count": 1,
        "full_video_coverage": True,
        "receiver_proof_status": "runtime_consumption_proof_ready",
        "rows": [
            {
                "group_name": "head_rgb_1.bias",
                "shape": [3],
                "numel": 3,
                "selected_bits": 0,
                "selected_action": "zero_rle",
                "blockers": [],
            }
        ],
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    waterfill_plan_path.write_text(
        json.dumps(waterfill_plan, sort_keys=True),
        encoding="utf-8",
    )

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "score_aware_training": {
                    "schema": "mlx_score_aware_training_objective.v1",
                    "segnet_distillation_weight": 0.0,
                    "pose_distillation_weight": 0.0,
                },
                "substrate_supplied_score_aware_training": {
                    "schema": "compact_hi_nerv_score_aware_training.v1",
                    "coder_aware_qat": {
                        "schema": "coder_aware_decoder_qat.v1",
                        "enabled": True,
                        "quant_bits": 4,
                        "quant_residual_weight": 0.001,
                        "magnitude_weight": 0.0001,
                        "delta_weight": 0.0002,
                        "c1a_entropy_weight": 0.0003,
                        "c1a_sigma": 0.35,
                        "c1a_sample_size": 64,
                        "c1a_source": (
                            "PR95 cat_entropy_v2 soft categorical entropy "
                            "adapted to selected decoder weights"
                        ),
                        "authority": "false_macos_mlx_research_signal",
                    },
                    "recon_pixel_weight": {
                        "schema": "compact_recon_pixel_weight.v1",
                        "enabled": True,
                        "source_kind": "file",
                        "path": Path(
                            kwargs["recon_pixel_weight_path"]
                        ).as_posix(),
                        "sha256": runner_mod._sha256_file(
                            Path(kwargs["recon_pixel_weight_path"])
                        ),
                        "npz_key": None,
                        "normalize": kwargs["recon_pixel_weight_normalize"],
                        "scorer_terms": {
                            "p18_segnet": "caller_supplied",
                            "p19_posenet": "caller_supplied",
                        },
                        "authority": "false_macos_mlx_research_signal",
                    },
                },
            },
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_qat",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        decoder_codec="int2_scale_bundled",
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        modelsize_candidate={
            "schema": "hinerv_modelsize_candidate.v1",
            "family": "hi_nerv",
            "candidate_id": "hinerv-unit-candidate",
            "latent_dim": 12,
            "embed_dim": 16,
            "decoder_channel": 6,
            "mid_injection_block_index": 2,
            "fine_injection_block_index": 5,
            "decoder_codec": "int4_mixed",
            "hi_nerv_latent_codec": "int16_brotli_q11",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 160_000,
            "nominal_under_ceiling": True,
            "use_hierarchical_feature_grid": True,
            "use_convnext_blocks": True,
            "modelsize_control_contract": {
                "schema": "nerv_modelsize_control_contract.v1",
                "family": "hi_nerv",
                "control_semantics": (
                    "local_receiver_visible_grid_search_nearest_target"
                ),
                "archive_bytes_authority_required": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0001,
        coder_qat_delta_weight=0.0002,
        coder_qat_c1a_entropy_weight=0.0003,
        coder_qat_c1a_sigma=0.35,
        coder_qat_c1a_sample_size=64,
        decoder_weight_waterfill_plan_json=waterfill_plan_path,
        recon_pixel_weight_path=weight_path,
        auto_segnet_boundary_recon_weight=False,
        recon_pixel_weight_tau=0.5,
        recon_pixel_weight_normalize="mean",
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["coder_aware_qat"] is True
    assert captured_train_kwargs["coder_qat_quant_bits"] == 4
    assert captured_train_kwargs["candidate_curriculum_plan"]["coder_pressure"][
        "quant_bits"
    ] == 4
    assert captured_train_kwargs["latent_dim"] == 12
    assert captured_train_kwargs["embed_dim"] == 16
    assert captured_train_kwargs["decoder_channel"] == 6
    assert captured_train_kwargs["decoder_codec"] == "int4_mixed"
    assert captured_train_kwargs["hi_nerv_latent_codec"] == "int16_brotli_q11"
    assert captured_train_kwargs["coder_qat_quant_residual_weight"] == 0.001
    assert captured_train_kwargs["coder_qat_magnitude_weight"] == 0.0001
    assert captured_train_kwargs["coder_qat_delta_weight"] == 0.0002
    assert captured_train_kwargs["coder_qat_c1a_entropy_weight"] == 0.0003
    assert captured_train_kwargs["coder_qat_c1a_sigma"] == 0.35
    assert captured_train_kwargs["coder_qat_c1a_sample_size"] == 64
    captured_waterfill = captured_train_kwargs["decoder_weight_waterfill_plan"]
    assert captured_waterfill["schema"] == waterfill_plan["schema"]
    assert captured_waterfill["rows"] == waterfill_plan["rows"]
    launch_custody = captured_waterfill["compact_runner_launch_custody"]
    assert launch_custody["schema"] == (
        "compact_hi_nerv_decoder_weight_waterfill_launch_custody.v1"
    )
    assert launch_custody["path"] == waterfill_plan_path.as_posix()
    assert launch_custody["sha256"] == runner_mod._sha256_file(waterfill_plan_path)
    assert launch_custody["source_schema"] == "nerv_decoder_weight_waterfill.v1"
    assert launch_custody["score_claim"] is False
    assert (
        captured_waterfill["receiver_proof_status"]
        == "runtime_consumption_proof_ready"
    )
    assert captured_waterfill["full_video_coverage"] is True
    assert captured_waterfill["blockers"] == []
    assert captured_train_kwargs["recon_pixel_weight_path"] == weight_path
    assert captured_train_kwargs["auto_segnet_boundary_recon_weight"] is False
    assert captured_train_kwargs["recon_pixel_weight_tau"] == 0.5
    assert captured_train_kwargs["recon_pixel_weight_normalize"] == "mean"
    coder_qat = out["score_aware_training"]["coder_aware_qat"]
    expected_coder_qat = {
        "schema": "coder_aware_decoder_qat.v1",
        "enabled": True,
        "quant_bits": 4,
        "quant_residual_weight": 0.001,
        "magnitude_weight": 0.0001,
        "delta_weight": 0.0002,
        "c1a_entropy_weight": 0.0003,
        "c1a_sigma": 0.35,
        "c1a_sample_size": 64,
        "c1a_source": (
            "PR95 cat_entropy_v2 soft categorical entropy adapted to selected "
            "decoder weights"
        ),
        "authority": "false_macos_mlx_research_signal",
    }
    for key, expected in expected_coder_qat.items():
        assert coder_qat[key] == expected
    selection = out["modelsize_candidate_selection"]
    assert selection["selection_mode"] == "planner_candidate"
    assert selection["candidate"]["candidate_id"] == "hinerv-unit-candidate"
    assert selection["modelsize_control_contract"]["family"] == "hi_nerv"
    assert selection["modelsize_control_contract"]["control_semantics"] == (
        "local_receiver_visible_grid_search_nearest_target"
    )
    assert (
        selection["modelsize_control_contract"]["archive_bytes_authority_required"]
        is True
    )
    assert selection["modelsize_control_contract"]["control_precedence"][
        "child_rules_override_parent_defaults"
    ] is True
    precedence = out["hi_nerv_control_precedence"]
    assert precedence["more_finely_grained_child_rules_take_priority"] is True
    assert precedence["pact_controls_take_priority_inside_source_faithful_subset"] is True
    assert precedence["highest_specificity_active_layer"] == (
        "promotion_and_exact_eval_gates"
    )
    assert precedence["modelsize_control_precedence"][
        "highest_specificity_active_layer"
    ] == "pact_receiver_visible_modelsize_child_rule"
    assert selection["candidate_curriculum_plan"]["coder_pressure"]["enabled"] is True
    assert selection["candidate_curriculum_plan"]["coder_pressure"][
        "quant_bits"
    ] == 4
    assert selection["launch_latent_dim"] == 12
    assert selection["launch_embed_dim"] == 16
    assert selection["launch_decoder_channel"] == 6
    assert selection["launch_decoder_codec"] == "int4_mixed"
    assert selection["launch_hi_nerv_latent_codec"] == "int16_brotli_q11"
    assert out["score_aware_training"]["decoder_codec"] == "int4_mixed"
    assert out["score_aware_training"]["hi_nerv_latent_codec"] == (
        "int16_brotli_q11"
    )
    waterfill = out["score_aware_training"]["decoder_weight_waterfill_plan"]
    assert waterfill["attached"] is True
    assert waterfill["path"] == waterfill_plan_path.as_posix()
    assert waterfill["sha256"] == runner_mod._sha256_file(waterfill_plan_path)
    assert waterfill["launch_custody"] == launch_custody
    assert waterfill["source_schema"] == "nerv_decoder_weight_waterfill.v1"
    assert waterfill["row_count"] == 1
    assert waterfill["source_blockers"] == []
    assert waterfill["active"] is True
    assert waterfill["validated"] is True
    assert waterfill["validated_rows"] == [
        {
            "group_name": "head_rgb_1.bias",
            "expected_shape": [3],
            "expected_numel": 3,
            "shape_checked": True,
            "numel_checked": True,
        }
    ]
    assert waterfill["score_claim"] is False
    feedback = out["candidate_curriculum_plan"]["byte_oracle_logging"]
    assert feedback["candidate_num_pairs"] == 600
    assert feedback["measured_num_pairs"] == 2
    assert feedback["feedback_scope"] == "partial_pair_advisory"
    assert feedback["scope_matches_candidate"] is False
    assert feedback["feedback_ready"] is False
    assert "partial_pair_byte_feedback_only" in out["blockers"]
    candidate_feedback = out["candidate_feedback"]
    assert Path(candidate_feedback["row_path"]).is_file()
    assert Path(candidate_feedback["ledger_path"]).is_file()
    assert candidate_feedback["row"]["candidate_id"] == "hinerv-unit-candidate"
    assert candidate_feedback["row"]["candidate_num_pairs"] == 600
    assert candidate_feedback["row"]["measured_num_pairs"] == 2
    assert candidate_feedback["row"]["feedback_ready"] is False
    assert candidate_feedback["score_claim"] is False
    assert out["score_aware_training"]["recon_pixel_weight"][
        "source_kind"
    ] == "file"
    assert out["score_aware_training"]["recon_pixel_weight"]["path"] == (
        weight_path.as_posix()
    )
    assert out["score_aware_training"]["recon_pixel_weight"]["scorer_terms"] == {
        "p18_segnet": "caller_supplied",
        "p19_posenet": "caller_supplied",
    }


def _hinerv_waterfill_modelsize_candidate(
    candidate_id: str = "hinerv-unit-candidate",
) -> dict[str, object]:
    return {
        "schema": "hinerv_modelsize_candidate.v1",
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "latent_dim": 12,
        "embed_dim": 16,
        "decoder_channel": 6,
        "mid_injection_block_index": 2,
        "fine_injection_block_index": 5,
        "decoder_codec": "int4_mixed",
        "num_pairs": 600,
        "hard_byte_ceiling": 178_000,
        "nominal_total_payload_bytes": 160_000,
        "nominal_under_ceiling": True,
        "use_hierarchical_feature_grid": True,
        "use_convnext_blocks": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _write_hinerv_waterfill_plan(
    path: Path,
    *,
    candidate_id: str = "hinerv-unit-candidate",
    group_name: str = "head_rgb_1.bias",
    shape: list[int] | None = None,
    numel: int | None = None,
) -> Path:
    row: dict[str, object] = {
        "group_name": group_name,
        "selected_bits": 0,
        "selected_action": "zero_rle",
    }
    if shape is not None:
        row["shape"] = shape
    if numel is not None:
        row["numel"] = numel
    path.write_text(
        json.dumps(
            {
                "schema": "nerv_decoder_weight_waterfill.v1",
                "family": "hi_nerv",
                "candidate_id": candidate_id,
                "group_count": 1,
                "rows": [row],
                "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _execute_hinerv_waterfill_validation_probe(
    tmp_path: Path,
    monkeypatch,
    *,
    plan_path: Path,
    candidate_id: str = "hinerv-unit-candidate",
) -> tuple[dict[str, object], dict[str, object]]:
    captured_train_kwargs: dict[str, object] = {}

    def fail_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        raise AssertionError("invalid waterfill plan must refuse before training")

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fail_train)
    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / f"hinerv_waterfill_refusal_{candidate_id}",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        modelsize_candidate=_hinerv_waterfill_modelsize_candidate(candidate_id),
        decoder_weight_waterfill_plan_json=plan_path,
        repo_root=REPO_ROOT,
    )
    return out, captured_train_kwargs


def test_hinerv_waterfill_plan_candidate_mismatch_refuses_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan_path = _write_hinerv_waterfill_plan(
        tmp_path / "candidate_mismatch_waterfill.json",
        candidate_id="different-hinerv-candidate",
        shape=[3],
        numel=3,
    )

    out, captured = _execute_hinerv_waterfill_validation_probe(
        tmp_path,
        monkeypatch,
        plan_path=plan_path,
    )

    assert captured == {}
    assert out["mode"] == "hi_nerv_decoder_weight_waterfill_plan_launch_refused"
    assert out["training_executed"] is False
    waterfill = out["score_aware_training"]["decoder_weight_waterfill_plan"]
    assert waterfill["attached"] is False
    assert waterfill["active"] is False
    assert waterfill["validated"] is False
    assert "decoder_weight_waterfill_candidate_id_mismatch:different-hinerv-candidate" in (
        out["blockers"]
    )
    assert waterfill["candidate_match"]["matched"] is False
    assert waterfill["score_claim"] is False


def test_hinerv_waterfill_plan_unknown_group_refuses_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan_path = _write_hinerv_waterfill_plan(
        tmp_path / "unknown_group_waterfill.json",
        group_name="missing_decoder_group.weight",
        shape=[1],
        numel=1,
    )

    out, captured = _execute_hinerv_waterfill_validation_probe(
        tmp_path,
        monkeypatch,
        plan_path=plan_path,
    )

    assert captured == {}
    waterfill = out["score_aware_training"]["decoder_weight_waterfill_plan"]
    assert waterfill["attached"] is False
    assert waterfill["active"] is False
    assert "decoder_weight_waterfill_group_missing:missing_decoder_group.weight" in (
        out["blockers"]
    )


def test_hinerv_waterfill_plan_shape_mismatch_refuses_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan_path = _write_hinerv_waterfill_plan(
        tmp_path / "shape_mismatch_waterfill.json",
        group_name="head_rgb_1.bias",
        shape=[4],
        numel=3,
    )

    out, captured = _execute_hinerv_waterfill_validation_probe(
        tmp_path,
        monkeypatch,
        plan_path=plan_path,
    )

    assert captured == {}
    waterfill = out["score_aware_training"]["decoder_weight_waterfill_plan"]
    assert waterfill["attached"] is False
    assert waterfill["active"] is False
    assert "decoder_weight_waterfill_shape_mismatch:head_rgb_1.bias" in out[
        "blockers"
    ]
    assert waterfill["validated_rows"][0]["declared_shape"] == [4]


def test_hinerv_waterfill_plan_compiles_train_time_fake_quant_bits() -> None:
    plan = {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": "unit",
        "rows": [
            {"group_name": "head_rgb_1.weight", "selected_bits": 0},
            {"group_name": "blocks.0.conv.weight", "selected_bits": 4},
            {"group_name": "blocks.1.conv.weight", "selected_bits": 6},
            {"group_name": "blocks.2.conv.weight", "selected_bits": 7},
            {"group_name": "latent_embed.bias", "selected_bits": 32},
        ],
    }

    assert runner_mod._decoder_weight_waterfill_fake_quant_bits_by_name(plan) == {
        "head_rgb_1.weight": 0,
        "blocks.0.conv.weight": 4,
        "blocks.1.conv.weight": 6,
        "blocks.2.conv.weight": 7,
        "latent_embed.bias": 32,
    }

    invalid = {**plan, "rows": [{"group_name": "x", "selected_bits": 3}]}
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="selected_bits",
    ):
        runner_mod._decoder_weight_waterfill_fake_quant_bits_by_name(invalid)


def test_hinerv_modelsize_launch_auto_binds_joint_scorer_pressure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive, pairs=2)
        submission = out / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    monkeypatch.setattr(runner_mod, "_run_hi_nerv_mlx_scoreaware_smoke", fake_train)

    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_auto_score_pressure",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        modelsize_candidate={
            "schema": "hinerv_modelsize_candidate.v1",
            "family": "hi_nerv",
            "candidate_id": "hinerv-auto-score-pressure",
            "latent_dim": 12,
            "embed_dim": 16,
            "decoder_channel": 6,
            "mid_injection_block_index": 2,
            "fine_injection_block_index": 5,
            "decoder_codec": "int4_mixed",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 160_000,
            "nominal_under_ceiling": True,
            "use_hierarchical_feature_grid": True,
            "use_convnext_blocks": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        pose_distillation_loss="huber",
        pose_distillation_huber_delta=2.5,
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        optimizer_kind="adafactor",
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["optimizer_kind"] == "adafactor"
    assert captured_train_kwargs["optimizer_controls"]["optimizer_kind"] == "adafactor"
    assert captured_train_kwargs["optimizer_controls"]["weight_decay_effective"] == pytest.approx(1.0e-4)
    assert captured_train_kwargs["segnet_distillation_weight"] == 1.0
    assert captured_train_kwargs["pose_distillation_weight"] == 1.0
    assert captured_train_kwargs["mid_injection_block_index"] == 2
    assert captured_train_kwargs["fine_injection_block_index"] == 5
    assert captured_train_kwargs["pose_distillation_loss"] == "huber"
    assert captured_train_kwargs["pose_distillation_huber_delta"] == 2.5
    assert captured_train_kwargs["coder_aware_qat"] is True
    assert captured_train_kwargs["coder_qat_quant_bits"] == 4
    assert captured_train_kwargs["hi_nerv_optimizer_policy"]["resolved_policy"] == (
        "native_optimizer"
    )
    assert captured_train_kwargs["hi_nerv_optimizer_policy"][
        "optimizer_kind_consumed_by_native_mlx"
    ] is True
    plan = captured_train_kwargs["candidate_curriculum_plan"]
    assert "hinerv_candidate_curriculum_requires_real_segnet_teacher" not in plan[
        "blockers"
    ]
    assert "hinerv_candidate_curriculum_requires_real_posenet_teacher" not in plan[
        "blockers"
    ]
    binding = out["hi_nerv_modelsize_launch_pressure"]
    assert binding["source"] == "modelsize_candidate_minimum_joint_scorer_pressure"
    assert {row["field"] for row in binding["mutations"]} == {
        "segnet_distillation_weight",
        "pose_distillation_weight",
    }
    assert out["score_aware_training"]["requested_segnet_distillation_weight"] == 0.0
    assert out["score_aware_training"]["requested_pose_distillation_weight"] == 0.0
    assert out["score_aware_training"]["segnet_distillation_weight"] == 1.0
    assert out["score_aware_training"]["pose_distillation_weight"] == 1.0
    assert out["score_aware_training"]["optimizer_kind"] == "adafactor"
    assert out["score_aware_training"]["optimizer_controls"]["optimizer_kind"] == (
        "adafactor"
    )
    assert out["score_aware_training"]["optimizer_policy"]["resolved_policy"] == (
        "native_optimizer"
    )
    assert out["score_aware_training_config_gate"]["frontier_targeting"] is True
    assert "hi_nerv_real_segnet_posenet_teachers_not_both_attached" not in out[
        "blockers"
    ]
    assert "hinerv_candidate_curriculum_requires_real_segnet_teacher" not in out[
        "blockers"
    ]
    assert "hinerv_candidate_curriculum_requires_real_posenet_teacher" not in out[
        "blockers"
    ]


def test_hinerv_optimizer_policy_refuses_pr95_curriculum_swallowing_non_adamw() -> None:
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="non-PR95-compatible --optimizer-kind would be ignored",
    ):
        runner_mod._resolve_hi_nerv_optimizer_policy(
            requested_policy="pr95_curriculum",
            epochs=29_650,
            optimizer_kind="lion",
        )

    native = runner_mod._resolve_hi_nerv_optimizer_policy(
        requested_policy="auto",
        epochs=29_650,
        optimizer_kind="lion",
    )
    assert native["resolved_policy"] == "native_optimizer"
    assert native["optimizer_kind_consumed_by_native_mlx"] is True

    pr95 = runner_mod._resolve_hi_nerv_optimizer_policy(
        requested_policy="auto",
        epochs=29_650,
        optimizer_kind="adamw",
    )
    assert pr95["resolved_policy"] == "pr95_curriculum"
    assert pr95["pr95_faithful_curriculum_enabled"] is True
    assert pr95["pr95_muon_policy"] == "faithful_stage8_only"


def test_hinerv_auto_policy_binds_long_pact_muon_adamw_to_pr95_every_stage() -> None:
    default_policy = runner_mod._resolve_hi_nerv_optimizer_policy(
        requested_policy="auto",
        epochs=29_650,
        optimizer_kind="pact_muon_adamw",
    )

    assert default_policy["resolved_policy"] == "pr95_curriculum"
    assert default_policy["optimizer_kind"] == "pact_muon_adamw"
    assert default_policy["pr95_muon_policy"] == "every_stage"
    assert (
        default_policy["effective_optimizer_label"]
        == "pr95_8stage_muon_adamw_every_stage"
    )
    assert default_policy["optimizer_kind_consumed_by_native_mlx"] is False
    assert default_policy["optimizer_kind_consumed_by_pr95_curriculum"] is True
    assert default_policy["pr95_faithful_curriculum_enabled"] is True


def test_hinerv_auto_policy_keeps_short_pact_muon_adamw_smokes_native() -> None:
    smoke_policy = runner_mod._resolve_hi_nerv_optimizer_policy(
        requested_policy="auto",
        epochs=2,
        optimizer_kind="pact_muon_adamw",
    )

    assert smoke_policy["resolved_policy"] == "native_optimizer"
    assert smoke_policy["optimizer_kind"] == "pact_muon_adamw"
    assert smoke_policy["pr95_muon_policy"] is None
    assert smoke_policy["effective_optimizer_label"] == "pact_muon_adamw"
    assert smoke_policy["optimizer_kind_consumed_by_native_mlx"] is True
    assert smoke_policy["optimizer_kind_consumed_by_pr95_curriculum"] is False
    assert smoke_policy["pr95_faithful_curriculum_enabled"] is False


def test_hinerv_optimizer_controls_default_to_pact_muon_adamw() -> None:
    controls = runner_mod._resolve_mlx_score_aware_optimizer_controls(
        optimizer_kind="pact_muon_adamw",
        requested_weight_decay=None,
        grad_clip_max_norm=1.0,
        warmup_epochs=0,
        warmup_steps_per_epoch=1,
        cosine_decay_enabled=False,
        cosine_decay_total_epochs=None,
        cosine_decay_min_lr_ratio=1e-2,
        run_epochs=29_650,
    )

    assert controls["optimizer_kind"] == "pact_muon_adamw"
    assert controls["weight_decay_effective"] == pytest.approx(1.0e-4)
    assert controls["weight_decay_defaulted"] is True
    assert controls["borrowed_pr95_partition_rule"] is True
    assert controls["original_pact_default_optimizer"] is True


def test_hinerv_optimizer_controls_refuse_silent_decay_drop() -> None:
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="optimizer-weight-decay is only supported",
    ):
        runner_mod._resolve_mlx_score_aware_optimizer_controls(
            optimizer_kind="adamax",
            requested_weight_decay=1.0e-4,
            grad_clip_max_norm=1.0,
            warmup_epochs=0,
            warmup_steps_per_epoch=1,
            cosine_decay_enabled=False,
            cosine_decay_total_epochs=None,
            cosine_decay_min_lr_ratio=1e-2,
            run_epochs=64,
        )


@pytest.mark.skipif(
    not _MLX_AVAILABLE or not _AV_AVAILABLE,
    reason="MLX and PyAV runtime video decode are required for HiNeRV adapter smoke",
)
def test_hinerv_execute_runs_training_archive_and_receiver_proof(
    tmp_path: Path,
) -> None:
    out = execute_hi_nerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "hinerv_gate",
        num_pairs=1,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        allow_unscored_research_smoke=True,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "executed_hi_nerv_mlx_scoreaware_and_exported"
    assert out["execute_family"] == "hi_nerv"
    assert out["training_executed"] is True
    assert out["adapter_smoke_only"] is False
    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert Path(out["archive_path"]).is_file()
    assert out["archive_bytes"] == Path(out["archive_path"]).stat().st_size
    assert len(out["archive_sha256"]) == 64
    assert out["scorer_upstream_snapshot"]["modules_py_exists"] is True
    assert out["scorer_upstream_snapshot"]["posenet_safetensors_exists"] is True
    assert out["scorer_upstream_snapshot"]["segnet_safetensors_exists"] is True
    assert out["projection_manifest_paths"]
    assert out["receiver_proof_report_paths"]
    proof = json.loads(Path(out["receiver_proof_report_paths"][0]).read_text())
    assert proof["runtime_consumption_proof_ready"] is True
    assert proof["receiver_contract_satisfied"] is True
    assert "hi_nerv_real_segnet_posenet_teachers_not_both_attached" in out[
        "blockers"
    ]
    assert out["score_aware_training"]["optimizer_kind"] == "pact_muon_adamw"
    assert out["score_aware_training"]["optimizer_policy"]["resolved_policy"] == (
        "native_optimizer"
    )
    assert out["score_aware_training"]["optimizer_controls"][
        "weight_decay_effective"
    ] == pytest.approx(1.0e-4)
    assert "hi_nerv_pr95_faithful_curriculum_requires_min_8_epochs" not in out[
        "blockers"
    ]
    assert "hi_nerv_receiver_proof_missing" not in out["blockers"]
    assert "local_cpu_replay_not_run_partial_pair_coverage" in out["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in out["blockers"]
    assert Path(out["report_path"]).is_file()


def test_snerv_execution_writes_archive_bound_report_and_reusable_hooks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet = _synthetic_snerv_packet(pairs=2)
    captured_advisory_kwargs: dict[str, object] = {}

    def fake_run_snerv_advisory(**kwargs):
        captured_advisory_kwargs.update(kwargs)
        assert kwargs["n_pairs"] == 2

        def as_jsonable() -> dict[str, object]:
            return {
                "schema": "fake_snerv_advisory.v1",
                "receiver_archive_packet": {
                    "bytes": len(packet),
                    "sha256": "0" * 64,
                    "redacted": True,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

        return SimpleNamespace(
            n_pairs=2,
            source_pair_indices=(7, 2),
            receiver_archive_packet=packet,
            as_jsonable=as_jsonable,
            levels=int(kwargs["levels"]),
            wavelet=str(kwargs["wavelet"]),
            score_linf=12.0,
            score_l2=13.0,
            d_seg_mean_linf=0.1,
            d_pose_mean_linf=0.01,
            archive_bytes_total=len(packet),
            snerv_fc_dim=9,
            snerv_emb_size=0,
            snerv_patch_radius=1,
            decoder_feature_count=9,
            beats_frontier_rate=True,
            receiver_archive_replay_verified=True,
        )

    def fake_export_snerv_archive_bound_candidate_package(**kwargs):
        package_dir = Path(kwargs["output_dir"])
        package_dir.mkdir(parents=True, exist_ok=True)
        archive = package_dir / "archive.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", bytes(kwargs["packet"]))
        submission = package_dir / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        proof = package_dir / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
        proof.parent.mkdir()
        proof.write_text(
            json.dumps(
                {
                    "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
                    "runtime_consumption_proof_ready": True,
                    "receiver_contract_satisfied": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )
        package_path = package_dir / "archive_bound_candidate_adapter_package.json"
        row = {
            "candidate_archive_path": archive.as_posix(),
            "candidate_archive_bytes": archive.stat().st_size,
            "candidate_archive_sha256": runner_mod._sha256_file(archive),
            "runtime_consumption_proof_ready": True,
            "receiver_contract_satisfied": True,
            "blockers": ["snerv_packet_not_full_600_pairs"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        package_path.write_text(
            json.dumps({"candidate_rows": [row]}, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "archive_bound_candidate_adapter_package": {
                "candidate_rows": [row],
            },
            "receiver_proof": {
                "proof_path": proof.as_posix(),
                "runtime_consumption_proof_ready": True,
                "receiver_contract_satisfied": True,
                "blockers": [],
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod
    import tac.substrates.snerv_inverse_steg_carrier.archive_candidate as package_mod

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fake_run_snerv_advisory)
    monkeypatch.setattr(
        package_mod,
        "export_snerv_archive_bound_candidate_package",
        fake_export_snerv_archive_bound_candidate_package,
    )

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_gate",
        num_pairs=2,
        epochs=3,
        hard_byte_ceilings=(178_000, 216_000),
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-unit-candidate",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "mfu_scales": (1, 5),
            "hfr_gain": 0.375,
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
            "nominal_under_ceiling": True,
            "modelsize_control_contract": {
                "schema": "nerv_modelsize_control_contract.v1",
                "family": "snerv",
                "control_semantics": (
                    "manual_receiver_visible_fc_dim_feature_basis"
                ),
                "archive_bytes_authority_required": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        prioritized_pair_indices=(7, 2, 7),
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "executed_snerv_archive_bound_advisory_and_exported"
    assert out["execute_family"] == "snerv"
    assert out["training_executed"] is False
    assert captured_advisory_kwargs["levels"] == 2
    assert captured_advisory_kwargs["target_bits_per_coeff"] == 1.5
    assert captured_advisory_kwargs["step_map_coder_mode"] == "waterfill"
    assert captured_advisory_kwargs["step_map_waterfill_bits_per_coeff"] == 0.5
    assert captured_advisory_kwargs["decoder_payload_codec"] == "int2_symmetric"
    assert captured_advisory_kwargs["snerv_mfu_scales"] == (1, 5)
    assert captured_advisory_kwargs["snerv_hfr_gain"] == 0.375
    assert captured_advisory_kwargs["pair_indices"] == (7, 2)
    selection = out["modelsize_candidate_selection"]
    assert selection["selection_mode"] == "planner_candidate"
    assert selection["candidate"]["candidate_id"] == "snerv-unit-candidate"
    assert selection["modelsize_control_contract"]["family"] == "snerv"
    assert selection["modelsize_control_contract"]["control_semantics"] == (
        "manual_receiver_visible_fc_dim_feature_basis"
    )
    assert (
        selection["modelsize_control_contract"]["archive_bytes_authority_required"]
        is True
    )
    assert selection["launch_levels"] == 2
    assert selection["launch_bits_per_coeff"] == 1.5
    assert selection["launch_decoder_payload_codec"] == "int2_symmetric"
    assert selection["candidate_curriculum_plan"]["receiver_grammar_controls"][
        "step_map_coder_mode"
    ] == "waterfill"
    assert Path(out["archive_path"]).is_file()
    assert out["archive_bytes"] == Path(out["archive_path"]).stat().st_size
    assert Path(out["receiver_archive_packet_path"]).read_bytes() == packet
    assert Path(out["advisory_report_path"]).is_file()
    assert Path(out["runtime_package_path"]).is_file()
    assert Path(out["trained_ladder_row_payload_path"]).is_file()
    assert out["trained_ladder_row_payload"]["schema"] == (
        "nerv_trained_ladder_row_payload.v1"
    )
    assert out["trained_ladder_row_payload"]["status"] == (
        "trained_ladder_row_blocked"
    )
    assert out["trained_ladder_row_payload"]["archive_path_kind"] == (
        "contest_archive_zip"
    )
    assert "sample_pair_count_below_full600" in out["trained_ladder_row_payload"][
        "blockers"
    ]
    assert out["receiver_proof_report_paths"]
    planner = out["score_aware_carrier_training_plan"]
    assert planner["score_aware_training_ready"] is False
    native_contract = out["snerv_mlx_native_adapter_contract"]
    assert native_contract["schema"] == "snerv_mlx_native_adapter_contract.v1"
    assert native_contract["surfaces_ready"] is True
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in (
        native_contract["blockers"]
    )
    assert out["score_aware_training"]["status"] == (
        "executed_cpu_advisory_mlx_native_training_missing"
    )
    assert out["score_aware_training"]["target_bits_per_coeff"] == 1.5
    assert out["score_aware_training"]["step_map_coder_mode"] == "waterfill"
    assert out["score_aware_training"]["decoder_payload_codec"] == "int2_symmetric"
    assert out["score_aware_training"]["source_pair_indices"] == [7, 2]
    assert out["score_aware_training"]["prioritized_pair_training"]["enabled"] is True
    assert out["score_aware_training"]["prioritized_pair_training"]["pair_indices"] == [
        7,
        2,
    ]
    assert (
        out["score_aware_training"]["prioritized_pair_training"][
            "consumed_by_cpu_advisory"
        ]
        is True
    )
    assert (
        out["score_aware_training"]["prioritized_pair_training"][
            "consumed_by_mlx_native_export"
        ]
        is False
    )
    assert (
        out["score_aware_training"]["prioritized_pair_training"]["score_claim"]
        is False
    )
    assert (
        out["score_aware_training"]["prioritized_pair_training"][
            "promotion_eligible"
        ]
        is False
    )
    assert (
        out["score_aware_training"]["prioritized_pair_training"][
            "ready_for_exact_eval_dispatch"
        ]
        is False
    )
    feedback = out["candidate_curriculum_plan"]["byte_oracle_logging"]
    assert feedback["candidate_num_pairs"] == 600
    assert feedback["measured_num_pairs"] == 2
    assert feedback["feedback_scope"] == "partial_pair_advisory"
    assert feedback["scope_matches_candidate"] is False
    assert feedback["feedback_ready"] is False
    assert feedback["measured_payload_bytes"] == len(packet)
    assert feedback["measured_archive_bytes"] == Path(out["archive_path"]).stat().st_size
    assert "partial_pair_byte_feedback_only" in out["blockers"]
    candidate_feedback = out["candidate_feedback"]
    assert Path(candidate_feedback["row_path"]).is_file()
    assert Path(candidate_feedback["ledger_path"]).is_file()
    assert candidate_feedback["row"]["candidate_id"] == "snerv-unit-candidate"
    assert candidate_feedback["row"]["candidate_num_pairs"] == 600
    assert candidate_feedback["row"]["measured_num_pairs"] == 2
    assert candidate_feedback["row"]["feedback_ready"] is False
    assert candidate_feedback["score_claim"] is False
    binary_profile = out["snerv_binary_profile"]
    assert binary_profile["profile_written"] is True
    assert Path(binary_profile["profile_path"]).is_file()
    assert binary_profile["snar1_packet_bytes"] == len(packet)
    assert binary_profile["lf_payload_bytes"] > 0
    assert binary_profile["score_claim"] is False
    assert out["score_aware_training"]["beats_frontier_rate"] is True
    assert out["reusable_optimization_followups"][
        "applies_after_byte_closed_export"
    ] is True
    assert "final_rate_attack_and_repair_materializers" in out[
        "reusable_optimization_followups"
    ]["required_hooks"]
    post_export = out["post_export_materializer_plan"]
    assert post_export["schema"] == "compact_carrier_post_export_materializer_plan.v1"
    assert post_export["compiled"] is True
    assert post_export["queue_launch_executed"] is False
    assert post_export["experiment_count"] > 0
    assert Path(post_export["experiment_queue_path"]).is_file()
    assert post_export["archive_record"]["source_runtime_dir"].endswith(
        "/snerv_archive_bound_package/submission"
    )
    assert post_export["archive_record"]["source_inflate_sh_path"].endswith(
        "/snerv_archive_bound_package/submission/inflate.sh"
    )
    contexts = json.loads(Path(post_export["materializer_contexts_path"]).read_text())
    first_context = contexts["rows"][0]["context"]
    assert first_context["source_runtime_dir"].endswith(
        "/snerv_archive_bound_package/submission"
    )
    assert first_context["packet_member_merge_source_runtime_dir"].endswith(
        "/snerv_archive_bound_package/submission"
    )
    queue = json.loads(Path(post_export["experiment_queue_path"]).read_text())
    harvest_step = queue["experiments"][0]["steps"][1]
    state_arg_index = harvest_step["command"].index("--state") + 1
    assert harvest_step["command"][state_arg_index] == post_export[
        "experiment_queue_state_path"
    ]
    post_export_execution = out["post_export_materializer_execution"]
    assert post_export_execution["requested"] is False
    assert post_export_execution["executed"] is False
    assert Path(post_export_execution["execution_path"]).is_file()
    assert out["reusable_optimization_followups"][
        "post_export_experiment_queue_path"
    ] == post_export["experiment_queue_path"]
    assert out["candidate_curriculum_plan"]["training_plan"][
        "receiver_proof_attached"
    ] is True
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in out[
        "blockers"
    ]
    assert out["snerv_mlx_native_adapter_contract"]["surfaces_ready"] is True
    assert "snerv_receiver_proof_missing" not in out["blockers"]
    assert "full_video_mlx_scorer_replay_not_attached" in out["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in out["blockers"]


def test_execute_snerv_attaches_native_mlx_export_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet = _synthetic_snerv_packet(pairs=2)
    native_calls: list[dict[str, object]] = []

    def fake_run_snerv_advisory(**kwargs):
        def as_jsonable() -> dict[str, object]:
            return {
                "schema": "fake_snerv_advisory.v1",
                "receiver_archive_packet": {
                    "bytes": len(packet),
                    "sha256": "0" * 64,
                    "redacted": True,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

        return SimpleNamespace(
            n_pairs=int(kwargs["n_pairs"]),
            receiver_archive_packet=packet,
            as_jsonable=as_jsonable,
            levels=int(kwargs["levels"]),
            wavelet="db2",
            score_linf=12.0,
            score_l2=13.0,
            d_seg_mean_linf=0.1,
            d_pose_mean_linf=0.01,
            archive_bytes_total=len(packet),
            snerv_fc_dim=kwargs["snerv_fc_dim"],
            snerv_emb_size=kwargs["snerv_emb_size"],
            snerv_patch_radius=kwargs["snerv_patch_radius"],
            snerv_model_size_adapter=kwargs["snerv_model_size_adapter"],
            snerv_mfu_scales=kwargs["snerv_mfu_scales"],
            snerv_hfr_gain=kwargs["snerv_hfr_gain"],
            snerv_temporal_context=kwargs["snerv_temporal_context"],
            snerv_temporal_mode=kwargs["snerv_temporal_mode"],
            decoder_feature_count=(
                int(kwargs["snerv_fc_dim"]) + int(kwargs["snerv_emb_size"])
            ),
            beats_frontier_rate=True,
            receiver_archive_replay_verified=True,
        )

    def fake_export_snerv_archive_bound_candidate_package(**kwargs):
        package_dir = Path(kwargs["output_dir"])
        package_dir.mkdir(parents=True, exist_ok=True)
        archive = package_dir / "archive.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", bytes(kwargs["packet"]))
        submission = package_dir / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        proof = package_dir / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
        proof.parent.mkdir()
        proof.write_text(
            json.dumps(
                {
                    "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
                    "runtime_consumption_proof_ready": True,
                    "runtime_consumption_proof_passed": True,
                    "receiver_contract_satisfied": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )
        row = {
            "candidate_archive_path": archive.as_posix(),
            "candidate_archive_bytes": archive.stat().st_size,
            "candidate_archive_sha256": runner_mod._sha256_file(archive),
            "runtime_consumption_proof_ready": True,
            "receiver_contract_satisfied": True,
            "blockers": ["snerv_packet_not_full_600_pairs"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        (package_dir / "archive_bound_candidate_adapter_package.json").write_text(
            json.dumps({"candidate_rows": [row]}, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "archive_bound_candidate_adapter_package": {"candidate_rows": [row]},
            "receiver_proof": {
                "proof_path": proof.as_posix(),
                "runtime_consumption_proof_ready": True,
                "runtime_consumption_proof_passed": True,
                "receiver_contract_satisfied": True,
                "blockers": [],
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    def fake_train_export_snerv_mlx_native(**kwargs):
        native_calls.append(dict(kwargs))
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        packet_path = out / "snerv_mlx_native_packet.snar"
        packet_path.write_bytes(b"native-snar")
        archive = out / "archive.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", b"native-snar")
        proof = out / "receiver_proof.json"
        proof.write_text(
            json.dumps({"receiver_contract_satisfied": True}),
            encoding="utf-8",
        )
        report = out / "snerv_mlx_native_train_export.json"
        long_training_executed = int(
            kwargs.get("score_aware_long_training_epochs") or 0
        ) > 0
        real_teachers_bound = bool(
            float(kwargs.get("segnet_distillation_weight") or 0.0) > 0.0
            and float(kwargs.get("pose_distillation_weight") or 0.0) > 0.0
        )
        payload = {
            "schema": "snerv_mlx_native_train_export.v1",
            "report_path": report.as_posix(),
            "packet_path": packet_path.as_posix(),
            "packet_bytes": packet_path.stat().st_size,
            "packet_sha256": runner_mod._sha256_file(packet_path),
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "receiver_proof_path": proof.as_posix(),
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "native_mlx_training_executed": int(
                kwargs.get("native_mlx_decoder_train_steps") or 0
            )
            > 0
            or long_training_executed,
            "native_mlx_training_kind": (
                "snerv_mlx_score_aware_haar_renderer"
                if long_training_executed
                else "full_batch_hf_decoder_gradient_descent"
            ),
            "score_aware_long_training_executed": long_training_executed,
            "score_aware_long_training_real_teachers_bound": real_teachers_bound,
            "score_aware_long_training_has_real_segnet_teacher": real_teachers_bound,
            "score_aware_long_training_has_real_posenet_teacher": real_teachers_bound,
            "score_aware_long_training_kind": (
                "snerv_mlx_score_aware_haar_renderer"
                if long_training_executed
                else "none"
            ),
            "score_aware_long_training": {
                "schema": "snerv_mlx_score_aware_long_training_attachment.v1",
                "executed": long_training_executed,
                "requested_epochs": int(
                    kwargs.get("score_aware_long_training_epochs") or 0
                ),
                "learning_rate": float(
                    kwargs.get("score_aware_long_training_lr") or 0.0
                ),
                "batch_pairs": int(
                    kwargs.get("score_aware_long_training_batch_pairs") or 0
                ),
                "optimizer_kind": str(
                    kwargs.get("score_aware_long_training_optimizer") or ""
                ),
                "eval_roundtrip_ste_enabled": bool(
                    kwargs.get("score_aware_long_training_eval_roundtrip_ste")
                ),
                "has_real_segnet_teacher": real_teachers_bound,
                "has_real_posenet_teacher": real_teachers_bound,
                "teacher_binding": {
                    "schema": "snerv_mlx_real_scorer_teacher_binding.v1",
                    "requested": real_teachers_bound,
                    "segnet_distillation_weight": float(
                        kwargs.get("segnet_distillation_weight") or 0.0
                    ),
                    "pose_distillation_weight": float(
                        kwargs.get("pose_distillation_weight") or 0.0
                    ),
                    "pose_distillation_loss": str(
                        kwargs.get("pose_distillation_loss") or "mse"
                    ),
                    "pose_distillation_huber_delta": float(
                        kwargs.get("pose_distillation_huber_delta") or 1.0
                    ),
                    "has_real_segnet_teacher": real_teachers_bound,
                    "has_real_posenet_teacher": real_teachers_bound,
                    "allow_segnet_only_research": bool(
                        kwargs.get("allow_segnet_only_research")
                    ),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "native_mlx_hf_decoder_training": {
                "schema": "snerv_native_mlx_hf_decoder_training.v1",
                "requested_steps": int(
                    kwargs.get("native_mlx_decoder_train_steps") or 0
                ),
                "learning_rate": float(
                    kwargs.get("native_mlx_decoder_train_lr") or 0.0
                ),
                "ridge": float(kwargs.get("native_mlx_decoder_train_ridge") or 0.0),
                "executed": int(kwargs.get("native_mlx_decoder_train_steps") or 0)
                > 0,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "scorer_loop_qat": {
                "executed": bool(kwargs.get("run_scorer_loop_qat")),
                "receiver_contract_satisfied": bool(kwargs.get("run_scorer_loop_qat")),
                "ready_for_pose_guard_gate": bool(kwargs.get("run_scorer_loop_qat")),
                "accepted_improvement": bool(kwargs.get("run_scorer_loop_qat")),
                "emitted_packet_uses_scorer_loop_best_decoder": False,
            },
            "num_pairs": int(kwargs["num_pairs"]),
            "source_pair_indices": [
                int(value)
                for value in (
                    kwargs.get("pair_indices") or tuple(range(int(kwargs["num_pairs"])))
                )
            ],
            "blockers": [
                *(
                    []
                    if long_training_executed
                    else ["snerv_mlx_score_aware_long_training_not_executed"]
                ),
                "contest_cpu_cuda_exact_eval_not_executed",
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        report.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return payload

    class FakeQatResult:
        def as_jsonable(self) -> dict[str, object]:
            return {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 2,
                "scorer_loop_evaluations": 1,
                "accepted_improvement": True,
                "receiver_contract_satisfied": True,
                "ready_for_pose_guard_gate": True,
                "baseline": {
                    "archive_bytes": len(packet),
                    "archive_sha256": "3" * 64,
                    "score_linf": 12.0,
                },
                "best": {
                    "archive_bytes": len(packet) - 1,
                    "archive_sha256": "4" * 64,
                    "score_linf": 11.0,
                },
                "blockers": ["local_smoke_only_not_full_600_pairs"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod
    import tac.substrates.snerv_inverse_steg_carrier.archive_candidate as package_mod
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as native_mod
    import tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat as qat_mod

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fake_run_snerv_advisory)
    monkeypatch.setattr(
        package_mod,
        "export_snerv_archive_bound_candidate_package",
        fake_export_snerv_archive_bound_candidate_package,
    )
    monkeypatch.setattr(
        native_mod,
        "train_export_snerv_mlx_native",
        fake_train_export_snerv_mlx_native,
    )
    monkeypatch.setattr(
        qat_mod,
        "run_snerv_scorer_loop_decoder_qat_smoke",
        lambda **_kwargs: FakeQatResult(),
    )
    recon_weight_path = tmp_path / "joint_recon_weight.npy"
    np.save(recon_weight_path, np.ones((384, 512), dtype=np.float32))
    recon_weight_manifest_path = tmp_path / "joint_recon_weight_manifest.json"
    recon_weight_manifest_path.write_text("{}", encoding="utf-8")
    recon_weight_discovery = {
        "schema": "compact_auto_joint_recon_pixel_weight_discovery.v1",
        "status": "selected_verified_joint_p18_p19_weight",
        "num_pairs": 2,
        "selected_manifest_path": recon_weight_manifest_path.as_posix(),
        "selected_manifest_sha256": runner_mod._sha256_file(recon_weight_manifest_path),
        "selected_weight_path": recon_weight_path.as_posix(),
        "selected_weight_sha256": runner_mod._sha256_file(recon_weight_path),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    def fake_discover_joint_recon_pixel_weight_path(**kwargs):
        assert kwargs["num_pairs"] == 2
        return recon_weight_path, recon_weight_discovery

    monkeypatch.setattr(
        runner_mod,
        "_discover_joint_recon_pixel_weight_path",
        fake_discover_joint_recon_pixel_weight_path,
    )

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_native_gate",
        num_pairs=2,
        epochs=3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-native-smoke",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
            "nominal_under_ceiling": True,
        },
        run_native_mlx_export=True,
        run_scorer_loop_qat=True,
        snerv_scorer_loop_max_trials=1,
        snerv_scorer_loop_search_mode="top_weight_coordinate",
        snerv_scorer_loop_qat_bits=4,
        snerv_scorer_loop_component_guard_mode="pose_seg_hard",
        auto_joint_recon_pixel_weight=True,
        recon_pixel_weight_normalize="none",
        snerv_native_mlx_decoder_train_steps=11,
        snerv_native_mlx_decoder_train_lr=0.004,
        snerv_native_mlx_decoder_train_ridge=0.0003,
        snerv_native_mlx_decoder_train_optimizer="adam",
        snerv_score_aware_long_training_epochs=13,
        snerv_score_aware_long_training_lr=0.002,
        snerv_score_aware_long_training_batch_pairs=2,
        snerv_score_aware_long_training_optimizer="lion",
        snerv_score_aware_long_training_grad_clip_max_norm=0.5,
        snerv_score_aware_long_training_weight_decay=None,
        snerv_score_aware_long_training_eval_roundtrip_ste=True,
        segnet_distillation_weight=0.025,
        pose_distillation_weight=0.0025,
        pose_distillation_loss="huber",
        pose_distillation_huber_delta=2.25,
        segnet_distillation_objective="boundary_decision_tckd",
        distillation_temperature=3.0,
        segnet_tau_boundary=0.75,
        segnet_hinge_margin=1.25,
        prioritized_pair_indices=(7, 2, 7),
        repo_root=REPO_ROOT,
    )

    assert native_calls
    assert native_calls[0]["num_pairs"] == 2
    assert native_calls[0]["pair_indices"] is None
    assert native_calls[0]["prioritized_pair_indices"] == (7, 2)
    assert native_calls[0]["run_scorer_loop_qat"] is True
    assert native_calls[0]["scorer_loop_qat_max_trials"] == 1
    assert native_calls[0]["scorer_loop_qat_search_mode"] == "top_weight_coordinate"
    assert native_calls[0]["scorer_loop_qat_qat_bits"] == 4
    assert native_calls[0]["scorer_loop_qat_component_guard_mode"] == "pose_seg_hard"
    assert native_calls[0]["scorer_loop_qat_decoder_payload_codec"] == (
        "int2_symmetric"
    )
    assert native_calls[0]["scorer_loop_qat_lf_payload_codec"] == "portfolio_auto"
    assert native_calls[0]["native_mlx_decoder_train_steps"] == 11
    assert native_calls[0]["native_mlx_decoder_train_lr"] == pytest.approx(0.004)
    assert native_calls[0]["native_mlx_decoder_train_ridge"] == pytest.approx(0.0003)
    assert native_calls[0]["native_mlx_decoder_train_optimizer"] == "adam"
    assert native_calls[0]["score_aware_long_training_epochs"] == 13
    assert native_calls[0]["score_aware_long_training_lr"] == pytest.approx(0.002)
    assert native_calls[0]["score_aware_long_training_batch_pairs"] == 2
    assert native_calls[0]["score_aware_long_training_optimizer"] == "lion"
    assert native_calls[0]["score_aware_long_training_grad_clip_max_norm"] == 0.5
    assert native_calls[0]["score_aware_long_training_weight_decay"] is None
    assert native_calls[0]["score_aware_long_training_eval_roundtrip_ste"] is True
    assert native_calls[0]["segnet_distillation_weight"] == pytest.approx(0.025)
    assert native_calls[0]["pose_distillation_weight"] == pytest.approx(0.0025)
    assert native_calls[0]["pose_distillation_loss"] == "huber"
    assert native_calls[0]["pose_distillation_huber_delta"] == pytest.approx(2.25)
    assert native_calls[0]["segnet_distillation_objective"] == (
        "boundary_decision_tckd"
    )
    assert native_calls[0]["distillation_temperature"] == pytest.approx(3.0)
    assert native_calls[0]["segnet_tau_boundary"] == pytest.approx(0.75)
    assert native_calls[0]["segnet_hinge_margin"] == pytest.approx(1.25)
    assert native_calls[0]["distillation_device"] == "cpu"
    assert native_calls[0]["allow_segnet_only_research"] is False
    assert native_calls[0]["modelsize_candidate"][
        "snerv_score_aware_long_training_epochs"
    ] == 13
    assert native_calls[0]["modelsize_candidate"][
        "snerv_segnet_distillation_weight"
    ] == pytest.approx(0.025)
    assert native_calls[0]["modelsize_candidate"][
        "snerv_pose_distillation_weight"
    ] == pytest.approx(0.0025)
    assert Path(native_calls[0]["recon_pixel_weight_path"]) == recon_weight_path
    assert (
        Path(native_calls[0]["recon_pixel_weight_manifest_path"])
        == recon_weight_manifest_path
    )
    assert native_calls[0]["recon_pixel_weight_normalize"] == "none"
    native = out["snerv_mlx_native_export"]
    assert native["executed"] is True
    assert native["source_pair_indices"] == [0, 1]
    assert native["prioritized_pair_training"]["enabled"] is True
    assert native["prioritized_pair_training"]["pair_indices"] == [7, 2]
    assert native["prioritized_pair_training"]["sampling_scope"] == (
        "score_aware_training_batches_not_target_hydration"
    )
    assert (
        native["prioritized_pair_training"]["consumed_by_native_mlx_train_export"]
        is True
    )
    assert native["receiver_proof_passed"] is True
    assert native["receiver_contract_satisfied"] is True
    assert native["native_mlx_training_executed"] is True
    assert native["native_mlx_training_kind"] == (
        "snerv_mlx_score_aware_haar_renderer"
    )
    assert native["score_aware_long_training_executed"] is True
    assert native["score_aware_long_training_real_teachers_bound"] is True
    assert native["score_aware_long_training_has_real_segnet_teacher"] is True
    assert native["score_aware_long_training_has_real_posenet_teacher"] is True
    assert native["native_mlx_hf_decoder_training"]["requested_steps"] == 11
    assert native["native_mlx_hf_decoder_training"]["learning_rate"] == pytest.approx(
        0.004
    )
    assert native["native_mlx_hf_decoder_training"]["ridge"] == pytest.approx(0.0003)
    assert native["scorer_loop_qat_attached"] is True
    assert native["scorer_loop_qat_receiver_contract_satisfied"] is True
    assert native["scorer_loop_qat_ready_for_pose_guard_gate"] is True
    assert native["scorer_loop_qat_accepted_improvement"] is True
    assert native["scorer_loop_qat_best_materialized"] is False
    assert Path(native["artifact_report_path"]).is_file()
    recon_weight = out["snerv_recon_pixel_weight"]
    assert recon_weight["requested"] is True
    assert recon_weight["manifest_path"] == recon_weight_manifest_path.as_posix()
    assert recon_weight["auto_discovery"] == recon_weight_discovery
    assert recon_weight["enabled"] is False
    assert recon_weight["native_export_consumed"] is False
    assert recon_weight["primary_archive_consumed"] is False
    assert recon_weight["primary_archive_source"] == "snerv_native_mlx_export_direct"
    assert out["score_aware_training"]["mlx_native_train_export_attached"] is True
    assert out["score_aware_training"]["mlx_native_receiver_proof_passed"] is True
    assert out["score_aware_training"]["mlx_native_full600_export_verified"] is False
    top_prioritized = out["score_aware_training"]["prioritized_pair_training"]
    assert top_prioritized["consumed_by_mlx_native_export"] is True
    assert top_prioritized["mlx_native_export_blocker"] is None
    assert (
        out["score_aware_training"]["mlx_native_file_backed_export_evidence"][
            "file_backed_export_proof_passed"
        ]
        is True
    )
    assert (
        out["score_aware_training"]["mlx_native_file_backed_export_evidence"][
            "required_pair_file_backed_export_proof_passed"
        ]
        is False
    )
    plan = out["candidate_curriculum_plan"]
    assert plan["training_plan"]["native_mlx_train_export_attached"] is True
    assert plan["training_plan"]["native_mlx_receiver_proof_passed"] is True
    assert plan["training_plan"]["native_mlx_file_backed_export_proof_passed"] is False
    assert plan["training_plan"]["native_mlx_scorer_loop_qat_attached"] is True
    assert plan["training_plan"]["scorer_loop_qat_attached"] is True
    assert plan["training_plan"]["native_mlx_real_segnet_teacher_bound"] is True
    assert plan["training_plan"]["native_mlx_real_posenet_teacher_bound"] is True
    assert plan["training_plan"]["native_mlx_joint_real_teachers_bound"] is True
    assert plan["training_plan"]["native_mlx_eval_roundtrip_ste_bound"] is True
    assert (
        plan["training_plan"]["native_mlx_differentiable_pose_preprocess_bound"]
        is True
    )
    assert "snerv_scorer_loop_qat_not_attached" not in plan["blockers"]
    assert "snerv_real_segnet_teacher_missing" not in plan["blockers"]
    assert "snerv_real_posenet_teacher_missing" not in plan["blockers"]
    assert "snerv_qat_forward_missing" in plan["blockers"]
    assert "snerv_coder_aware_regularizer_missing" in plan["blockers"]
    assert plan["pr95_stack_binding"]["complete"] is False
    assert "snerv_native_scorer_loop_best_packet_not_materialized" in plan[
        "blockers"
    ]
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in plan[
        "blockers"
    ]
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in out[
        "blockers"
    ]
    assert "snerv_mlx_native_receiver_proof_missing_or_failed" not in out["blockers"]
    assert "snerv_mlx_native_export_partial_pair_coverage" in out["blockers"]
    assert "snerv_mlx_native_arbitrary_pair_hydration_not_implemented" not in out[
        "blockers"
    ]
    assert "snerv_mlx_native_prioritized_pair_hydration_not_consumed" not in out[
        "blockers"
    ]
    assert (
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        not in plan["blockers"]
    )
    assert "snerv_mlx_native_longer_staged_training_not_executed" not in out[
        "blockers"
    ]


def test_snerv_coder_aware_qat_executes_receiver_priced_scorer_loop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet = _synthetic_snerv_packet(pairs=2)
    captured_advisory_kwargs: dict[str, object] = {}
    captured_qat_kwargs: dict[str, object] = {}

    def fake_run_snerv_advisory(**kwargs):
        captured_advisory_kwargs.update(kwargs)

        def as_jsonable() -> dict[str, object]:
            return {
                "schema": "fake_snerv_advisory.v1",
                "receiver_archive_packet": {
                    "bytes": len(packet),
                    "sha256": "0" * 64,
                    "redacted": True,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

        return SimpleNamespace(
            n_pairs=int(kwargs["n_pairs"]),
            receiver_archive_packet=packet,
            as_jsonable=as_jsonable,
            levels=int(kwargs["levels"]),
            wavelet="db2",
            score_linf=12.0,
            score_l2=13.0,
            d_seg_mean_linf=0.1,
            d_pose_mean_linf=0.01,
            archive_bytes_total=len(packet),
            snerv_fc_dim=9,
            snerv_emb_size=0,
            snerv_patch_radius=1,
            snerv_model_size_adapter=kwargs["snerv_model_size_adapter"],
            snerv_mfu_scales=kwargs["snerv_mfu_scales"],
            snerv_hfr_gain=kwargs["snerv_hfr_gain"],
            snerv_temporal_context=0,
            snerv_temporal_mode=kwargs["snerv_temporal_mode"],
            decoder_feature_count=9,
            beats_frontier_rate=True,
            receiver_archive_replay_verified=True,
        )

    def fake_export_snerv_archive_bound_candidate_package(**kwargs):
        package_dir = Path(kwargs["output_dir"])
        package_dir.mkdir(parents=True, exist_ok=True)
        archive = package_dir / "archive.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", bytes(kwargs["packet"]))
        submission = package_dir / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        proof = package_dir / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
        proof.parent.mkdir()
        proof.write_text(
            json.dumps(
                {
                    "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
                    "runtime_consumption_proof_ready": True,
                    "receiver_contract_satisfied": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )
        row = {
            "candidate_archive_path": archive.as_posix(),
            "candidate_archive_bytes": archive.stat().st_size,
            "candidate_archive_sha256": runner_mod._sha256_file(archive),
            "runtime_consumption_proof_ready": True,
            "receiver_contract_satisfied": True,
            "blockers": ["snerv_packet_not_full_600_pairs"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        (package_dir / "archive_bound_candidate_adapter_package.json").write_text(
            json.dumps({"candidate_rows": [row]}, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "archive_bound_candidate_adapter_package": {
                "candidate_rows": [row],
            },
            "receiver_proof": {
                "proof_path": proof.as_posix(),
                "runtime_consumption_proof_ready": True,
                "receiver_contract_satisfied": True,
                "blockers": [],
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    def fake_run_snerv_scorer_loop_decoder_qat(**kwargs):
        captured_qat_kwargs.update(kwargs)
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback(
                SimpleNamespace(
                    as_jsonable=lambda: {
                        "label": "fake_progress_eval",
                        "archive_bytes": 123,
                        "score_linf": 4.5,
                    }
                )
            )

        def as_jsonable() -> dict[str, object]:
            return {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "accepted_improvement": True,
                "receiver_contract_satisfied": True,
                "ready_for_pose_guard_gate": True,
                "improvement_score_delta": -0.01,
                "improvement_d_pose_delta": 0.0,
                "improvement_d_seg_delta": -0.001,
                "scorer_loop_evaluations": 3,
                "pair_robust_admission": {
                    "schema": "snerv_pair_robust_admission.v1",
                    "n_pairs": 2,
                    "min_score_improved_fraction": 0.5,
                    "max_pose_worsened_fraction": 0.25,
                    "pose_slack": 0.001,
                    "score_improved_fraction": 1.0,
                    "pose_worsened_fraction": 0.0,
                    "permissive_guard": False,
                    "passed": True,
                    "blockers": [],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

        return SimpleNamespace(as_jsonable=as_jsonable)

    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod
    import tac.substrates.snerv_inverse_steg_carrier.archive_candidate as package_mod
    import tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat as qat_mod

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fake_run_snerv_advisory)
    monkeypatch.setattr(
        package_mod,
        "export_snerv_archive_bound_candidate_package",
        fake_export_snerv_archive_bound_candidate_package,
    )
    monkeypatch.setattr(
        qat_mod,
        "run_snerv_scorer_loop_decoder_qat",
        fake_run_snerv_scorer_loop_decoder_qat,
    )

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_qat_gate",
        num_pairs=2,
        epochs=3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-q-smoke",
            "wavelet": "haar",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "fc_dim": 11,
            "emb_size": 2,
            "patch_radius": 1,
            "temporal_context": 1,
            "temporal_mode": "official_haar_dwt1d_lowpass",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
            "nominal_under_ceiling": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        snerv_spectra_preserving_adapter=True,
        snerv_mfu_scales=(1, 3),
        snerv_hfr_gain=0.25,
        run_scorer_loop_qat=True,
        snerv_scorer_loop_qat_bits=4,
        snerv_scorer_loop_max_trials=5,
        snerv_scorer_loop_search_mode="learned_random_subspace",
        snerv_scorer_loop_step_map_bins=8,
        snerv_scorer_loop_lf_payload_codec="auto",
        snerv_scorer_loop_perturb_scale=0.03,
        snerv_scorer_loop_byte_pressure_multiplier=1.25,
        snerv_scorer_loop_max_archive_byte_growth=77,
        snerv_scorer_loop_pose_slack=0.001,
        snerv_scorer_loop_seg_slack=0.002,
        snerv_scorer_loop_pair_stride=3,
        snerv_scorer_loop_start_pair=7,
        snerv_scorer_loop_pair_guard_min_score_improved_fraction=0.5,
        snerv_scorer_loop_pair_guard_max_pose_worsened_fraction=0.25,
        snerv_scorer_loop_component_guard_mode="pose_hard",
        random_seed=123,
        repo_root=REPO_ROOT,
    )

    assert captured_advisory_kwargs["snerv_model_size_adapter"] == (
        SNERV_SPECTRA_PRESERVING_ADAPTER
    )
    assert captured_advisory_kwargs["wavelet"] == "haar"
    assert captured_advisory_kwargs["target_bits_per_coeff"] == 1.5
    assert captured_advisory_kwargs["snerv_fc_dim"] == 11
    assert captured_advisory_kwargs["snerv_emb_size"] == 2
    assert captured_advisory_kwargs["snerv_temporal_context"] == 1
    assert (
        captured_advisory_kwargs["snerv_temporal_mode"]
        == "official_haar_dwt1d_lowpass"
    )
    assert captured_advisory_kwargs["snerv_mfu_scales"] == (1, 3)
    assert captured_advisory_kwargs["snerv_hfr_gain"] == 0.25
    assert captured_qat_kwargs["n_pairs"] == 2
    assert captured_qat_kwargs["levels"] == 2
    assert captured_qat_kwargs["wavelet"] == "haar"
    assert captured_qat_kwargs["target_bits_per_coeff"] == 1.5
    assert captured_qat_kwargs["snerv_spectra_preserving_adapter"] is True
    assert captured_qat_kwargs["snerv_model_size_adapter"] == (
        SNERV_SPECTRA_PRESERVING_ADAPTER
    )
    assert captured_qat_kwargs["snerv_fc_dim"] == 11
    assert captured_qat_kwargs["snerv_emb_size"] == 2
    assert captured_qat_kwargs["snerv_temporal_context"] == 1
    assert (
        captured_qat_kwargs["snerv_temporal_mode"]
        == "official_haar_dwt1d_lowpass"
    )
    assert captured_qat_kwargs["snerv_mfu_scales"] == (1, 3)
    assert captured_qat_kwargs["snerv_hfr_gain"] == 0.25
    assert captured_qat_kwargs["qat_bits"] == 4
    assert captured_qat_kwargs["decoder_payload_codec"] == "int2_symmetric"
    assert captured_qat_kwargs["lf_payload_codec"] == "auto"
    assert captured_qat_kwargs["max_trials"] == 5
    assert captured_qat_kwargs["search_mode"] == "learned_random_subspace"
    assert captured_qat_kwargs["step_map_bins"] == 8
    assert captured_qat_kwargs["byte_pressure_multiplier"] == 1.25
    assert captured_qat_kwargs["max_archive_byte_growth"] == 77
    assert captured_qat_kwargs["pose_slack"] == 0.001
    assert captured_qat_kwargs["seg_slack"] == 0.002
    assert captured_qat_kwargs["pair_stride"] == 3
    assert captured_qat_kwargs["start_pair"] == 7
    assert captured_qat_kwargs["pair_guard_min_score_improved_fraction"] == 0.5
    assert captured_qat_kwargs["pair_guard_max_pose_worsened_fraction"] == 0.25
    assert captured_qat_kwargs["component_guard_mode"] == "pose_hard"
    assert captured_qat_kwargs["seed"] == 123
    assert callable(captured_qat_kwargs["progress_callback"])

    qat = out["snerv_scorer_loop_qat"]
    assert qat["executed"] is True
    assert qat["component_guard_mode"] == "pose_hard"
    assert qat["pair_robust_admission"]["schema"] == (
        "snerv_pair_robust_admission.v1"
    )
    assert qat["pair_robust_admission"]["passed"] is True
    assert qat["lf_payload_codec"] == "auto"
    assert qat["accepted_improvement"] is True
    assert qat["receiver_contract_satisfied"] is True
    assert qat["ready_for_pose_guard_gate"] is True
    assert Path(qat["result_path"]).is_file()
    progress_path = Path(qat["progress_jsonl_path"])
    assert progress_path.is_file()
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress_payload["schema"] == "snerv_scorer_loop_decoder_qat_progress.v1"
    assert progress_payload["score_claim"] is False
    assert progress_payload["row"]["label"] == "fake_progress_eval"
    assert len(qat["progress_jsonl_sha256"]) == 64
    assert out["score_aware_training"]["scorer_loop_qat"]["executed"] is True
    assert (
        out["score_aware_training"]["scorer_loop_component_guard_mode"]
        == "pose_hard"
    )
    assert out["score_aware_training"]["status"] == (
        "executed_cpu_advisory_plus_receiver_priced_scorer_loop_qat_"
        "mlx_native_training_missing"
    )
    plan = out["candidate_curriculum_plan"]
    assert plan["training_plan"]["scorer_loop_qat_attached"] is True
    assert plan["training_plan"]["scorer_loop_qat_receiver_contract_satisfied"] is True
    assert plan["training_plan"]["receiver_proof_attached"] is True
    assert "snerv_scorer_loop_qat_not_attached" not in plan["blockers"]
    assert "snerv_real_segnet_teacher_missing" in plan["blockers"]
    assert "snerv_real_posenet_teacher_missing" in plan["blockers"]
    assert "snerv_qat_forward_missing" in plan["blockers"]
    assert "snerv_coder_aware_regularizer_missing" in plan["blockers"]
    assert "snerv_receiver_proof_missing" not in plan["blockers"]
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in out[
        "blockers"
    ]
    assert "snerv_mlx_native_longer_staged_training_not_executed" in out[
        "blockers"
    ]
    assert "snerv_longer_staged_score_aware_training_not_executed" not in out[
        "blockers"
    ]


def test_snerv_runner_refuses_conflicting_candidate_adapter_flag(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        runner_mod.CompactRendererMlxSpineRunnerError,
        match="modelsize candidate adapter conflicts",
    ):
        execute_snerv_inverse_steg_advisory_and_adapt(
            output_dir=tmp_path / "snerv_conflict",
            num_pairs=2,
            epochs=3,
            source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
            modelsize_candidate={
                "schema": "snerv_modelsize_candidate.v1",
                "family": "snerv",
                "candidate_id": "snerv-conflict",
                "wavelet": "haar",
                "levels": 2,
                "bits_per_coeff": 1.5,
                "step_map_bits_per_coeff": 0.5,
                "decoder_payload_codec": "int2_symmetric",
                "snerv_model_size_adapter": "snerv_fc_dim_emb_size_adapter_v1",
                "fc_dim": 11,
                "emb_size": 2,
                "num_pairs": 600,
                "hard_byte_ceiling": 178_000,
                "nominal_total_payload_bytes": 150_000,
                "nominal_under_ceiling": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            snerv_spectra_preserving_adapter=True,
            repo_root=REPO_ROOT,
        )


def test_snerv_batched_full_video_mlx_prefilter_feeds_acquisition_not_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet = _synthetic_snerv_packet(pairs=600)

    def fake_run_snerv_advisory(**kwargs):
        assert kwargs["n_pairs"] == 600

        def as_jsonable() -> dict[str, object]:
            return {
                "schema": "fake_snerv_advisory.v1",
                "receiver_archive_packet": {
                    "bytes": len(packet),
                    "sha256": "0" * 64,
                    "redacted": True,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

        return SimpleNamespace(
            n_pairs=600,
            receiver_archive_packet=packet,
            as_jsonable=as_jsonable,
            levels=int(kwargs["levels"]),
            wavelet="db2",
            score_linf=91.0,
            score_l2=92.0,
            d_seg_mean_linf=0.5,
            d_pose_mean_linf=160.0,
            archive_bytes_total=len(packet),
            snerv_fc_dim=9,
            snerv_emb_size=0,
            snerv_patch_radius=1,
            decoder_feature_count=9,
            beats_frontier_rate=True,
            receiver_archive_replay_verified=True,
        )

    def fake_export_snerv_archive_bound_candidate_package(**kwargs):
        package_dir = Path(kwargs["output_dir"])
        package_dir.mkdir(parents=True, exist_ok=True)
        archive = package_dir / "archive.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", bytes(kwargs["packet"]))
        submission = package_dir / "submission"
        submission.mkdir()
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        proof = package_dir / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
        proof.parent.mkdir()
        proof.write_text(
            json.dumps(
                {
                    "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
                    "runtime_consumption_proof_ready": True,
                    "receiver_contract_satisfied": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )
        row = {
            "candidate_archive_path": archive.as_posix(),
            "candidate_archive_bytes": archive.stat().st_size,
            "candidate_archive_sha256": runner_mod._sha256_file(archive),
            "runtime_consumption_proof_ready": True,
            "receiver_contract_satisfied": True,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        (package_dir / "archive_bound_candidate_adapter_package.json").write_text(
            json.dumps({"candidate_rows": [row]}, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "archive_bound_candidate_adapter_package": {
                "candidate_rows": [row],
            },
            "receiver_proof": {
                "proof_path": proof.as_posix(),
                "runtime_consumption_proof_ready": True,
                "receiver_contract_satisfied": True,
                "blockers": [],
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod
    import tac.substrates.snerv_inverse_steg_carrier.archive_candidate as package_mod

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fake_run_snerv_advisory)
    monkeypatch.setattr(
        package_mod,
        "export_snerv_archive_bound_candidate_package",
        fake_export_snerv_archive_bound_candidate_package,
    )
    batched_profile = _write_mlx_prefilter_profile(
        tmp_path / "snerv_batched_mlx_full600.json",
        pairs=600,
        batch_pairs=8,
        score=91.0,
    )

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_batched_gpu_gate",
        num_pairs=600,
        epochs=3,
        hard_byte_ceilings=(178_000, 216_000),
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        mlx_profile_paths=(batched_profile,),
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-full600-batched-profile",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
            "nominal_under_ceiling": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        repo_root=REPO_ROOT,
    )

    assert out["local_cpu_replay_gate"]["executed"] is False
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is True
    assert out["local_cpu_replay_gate"]["local_replay_mlx_prefilter_passed"] is False
    assert out["mlx_prefilter_coverage"]["has_full_video_mlx_prefilter"] is True
    assert out["mlx_prefilter_coverage"]["local_replay_profile_paths"] == []
    assert out["mlx_prefilter_coverage"]["blockers"] == [
        "mlx_profile_batch_pairs_not_singleton"
    ]
    assert "full_video_mlx_scorer_replay_not_attached" not in out["blockers"]
    assert "local_cpu_replay_waiting_for_full_video_mlx_prefilter" not in out[
        "blockers"
    ]
    assert "local_cpu_replay_blocked_by_mlx_prefilter_score" in out["blockers"]
    feedback_blockers = out["candidate_feedback"]["row"][
        "pr95_stack_binding_blockers"
    ]
    assert "full_video_mlx_scorer_replay_not_attached" not in feedback_blockers
    feedback_row = out["candidate_feedback"]["row"]
    assert "local_cpu_replay_blocked_by_mlx_prefilter_score" in feedback_row[
        "blockers"
    ]
    assert feedback_row["mlx_prefilter_has_full_video"] is True
    assert feedback_row["mlx_prefilter_local_replay_passed"] is False
    assert feedback_row["mlx_prefilter_blockers"] == [
        "mlx_profile_batch_pairs_not_singleton"
    ]
    assert feedback_row["local_cpu_replay_gate_has_full_video_mlx_prefilter"] is True
    assert (
        feedback_row["local_cpu_replay_gate_local_replay_mlx_prefilter_passed"]
        is False
    )
    assert feedback_row["local_cpu_replay_gate_executed"] is False


def test_snerv_long_campaign_refuses_when_pr95_prelaunch_gate_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod

    def fail_run_snerv_advisory(**_kwargs):
        raise AssertionError("long campaign should refuse before SNeRV advisory")

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fail_run_snerv_advisory)

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_refusal",
        num_pairs=600,
        epochs=8,
        hard_byte_ceilings=(178_000, 216_000),
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-long-candidate",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
        },
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "snerv_pr95_binding_prelaunch_refused"
    assert out["training_executed"] is False
    assert "pr95_long_campaign_prelaunch_gate_failed" in out["blockers"]
    assert "snerv_pr95_staged_curriculum_missing" in out["blockers"]
    assert "snerv_real_segnet_teacher_missing" in out["blockers"]
    gate = out["candidate_curriculum_plan"]["long_campaign_prelaunch_gate"]
    assert gate["launch_allowed"] is False
    assert "local_cpu_replay_gate" in gate["post_run_requirements_excluded"]
    assert out["candidate_feedback"]["row"]["candidate_id"] == (
        "snerv-long-candidate"
    )
    assert out["candidate_feedback"]["row"][
        "long_campaign_prelaunch_launch_allowed"
    ] is False
    assert Path(out["report_path"]).is_file()


def test_snerv_native_export_bypasses_pr95_prelaunch_only_for_local_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.advisory as advisory_mod
    import tac.substrates.snerv_inverse_steg_carrier.archive_candidate as package_mod
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as native_mod

    class FakeAdvisory:
        receiver_archive_packet = b"SNAR1-unit"
        archive_bytes_total = 10
        levels = 2
        wavelet = "haar"
        score_linf = 1.0
        score_l2 = 1.1
        d_seg_mean_linf = 0.01
        d_pose_mean_linf = 0.02
        beats_frontier_rate = True
        receiver_archive_replay_verified = True

        def as_jsonable(self) -> dict:
            return {
                "schema": "snerv_inverse_steg_advisory.v1",
                "n_pairs": 600,
                "archive_bytes_total": 10,
                "levels": 2,
                "wavelet": "haar",
                "score_linf": 1.0,
                "score_l2": 1.1,
                "d_seg_mean_linf": 0.01,
                "d_pose_mean_linf": 0.02,
                "beats_frontier_rate": True,
                "receiver_archive_replay_verified": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

    def fake_run_snerv_advisory(**_kwargs):
        return FakeAdvisory()

    def fake_package(**_kwargs):
        archive = tmp_path / "archive.zip"
        archive.write_bytes(b"zip")
        return {
            "schema": "snerv_inverse_steg_archive_bound_adapter_package.v1",
            "archive_bound_candidate_adapter_package": {
                "candidate_rows": [
                    {
                        "candidate_archive_path": archive.as_posix(),
                        "candidate_archive_bytes": archive.stat().st_size,
                        "candidate_archive_sha256": "a" * 64,
                    }
                ],
            },
            "receiver_proof": {
                "proof_path": (tmp_path / "proof.json").as_posix(),
                "receiver_contract_satisfied": True,
            },
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": "a" * 64,
            "submission_dir": (tmp_path / "submission").as_posix(),
            "receiver_proof_path": (tmp_path / "proof.json").as_posix(),
            "receiver_contract_satisfied": True,
            "blockers": ["paired_contest_cpu_cuda_auth_eval_missing"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    def fake_native_export(**_kwargs):
        report = tmp_path / "native_report.json"
        packet = tmp_path / "native_packet.snar"
        archive = tmp_path / "native_archive.zip"
        proof = tmp_path / "native_receiver_proof.json"
        report.write_text(
            '{"schema":"snerv_mlx_native_train_export.v1"}\n',
            encoding="utf-8",
        )
        packet.write_bytes(b"native packet")
        archive.write_bytes(b"native archive")
        proof.write_text(
            json.dumps(
                {
                    "receiver_contract_satisfied": True,
                    "runtime_consumption_proof_passed": True,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "schema": "snerv_mlx_native_train_export.v1",
            "num_pairs": 600,
            "artifact_report_path": report.as_posix(),
            "packet_path": packet.as_posix(),
            "packet_bytes": packet.stat().st_size,
            "packet_sha256": runner_mod._sha256_file(packet),
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "receiver_proof_path": proof.as_posix(),
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "native_mlx_full600_campaign_ready": True,
            "scorer_loop_qat": {
                "requested": False,
                "executed": False,
                "blockers": ["snerv_scorer_loop_qat_not_requested"],
            },
            "blockers": ["snerv_mlx_score_aware_long_training_not_executed"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(advisory_mod, "run_snerv_advisory", fake_run_snerv_advisory)
    monkeypatch.setattr(
        package_mod,
        "export_snerv_archive_bound_candidate_package",
        fake_package,
    )
    monkeypatch.setattr(native_mod, "train_export_snerv_mlx_native", fake_native_export)

    out = execute_snerv_inverse_steg_advisory_and_adapt(
        output_dir=tmp_path / "snerv_native_proof",
        num_pairs=600,
        epochs=8,
        hard_byte_ceilings=(178_000, 216_000),
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        modelsize_candidate={
            "schema": "snerv_modelsize_candidate.v1",
            "family": "snerv",
            "candidate_id": "snerv-long-candidate",
            "levels": 2,
            "bits_per_coeff": 1.5,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int2_symmetric",
            "num_pairs": 600,
            "hard_byte_ceiling": 178_000,
            "nominal_total_payload_bytes": 150_000,
        },
        run_native_mlx_export=True,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "executed_snerv_archive_bound_advisory_and_exported"
    assert out["execute_family"] == "snerv"
    assert "pr95_long_campaign_prelaunch_gate_failed" in out["blockers"]
    assert "snerv_pr95_staged_curriculum_missing" in out["blockers"]
    assert out["candidate_curriculum_plan"]["training_plan"][
        "native_mlx_long_training_bound"
    ] is False
    assert (
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        in out["candidate_curriculum_plan"]["blockers"]
    )
    native = out["snerv_mlx_native_export"]
    assert native["native_mlx_full600_export_proof_ready"] is True
    assert native["native_mlx_full600_campaign_ready"] is False
    assert "snerv_mlx_native_export_closed_form_not_training" in native["blockers"]
    assert (
        "snerv_mlx_native_full600_not_campaign_ready_without_learned_training"
        in native["blockers"]
    )
    assert out["score_aware_training"]["mlx_native_training_required_next"] is True
    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False


def test_adapt_snerv_advisory_report_consumes_existing_runtime_package(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "snerv_package"
    submission = package_dir / "submission"
    submission.mkdir(parents=True)
    (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (submission / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    archive = _write_synthetic_pr95_archive(package_dir / "archive.zip", pairs=600)
    archive_sha = runner_mod._sha256_file(archive)
    archive_bytes = archive.stat().st_size
    proof_path = package_dir / "snerv_inverse_steg_receiver_proof.json"
    proof_payload = {
        "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
        "proof_path": proof_path.as_posix(),
        "runtime_consumption_proof_passed": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    proof_path.write_text(json.dumps(proof_payload), encoding="utf-8")
    report_path = tmp_path / "snerv_advisory.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "snerv_inverse_steg_advisory.v1",
                "n_pairs": 600,
                "runtime_package_dir": package_dir.as_posix(),
                "runtime_package": {
                    "archive_bound_candidate_adapter_package": {
                        "candidate_rows": [
                            {
                                "candidate_archive_path": archive.as_posix(),
                                "candidate_archive_sha256": archive_sha,
                                "candidate_archive_bytes": archive_bytes,
                                "blockers": [
                                    "paired_contest_cpu_cuda_auth_eval_missing"
                                ],
                                "score_claim": False,
                                "promotion_eligible": False,
                                "ready_for_exact_eval_dispatch": False,
                            }
                        ]
                    },
                    "receiver_proof": proof_payload,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    out = runner_mod.adapt_snerv_advisory_report_to_spine(
        snerv_advisory_report_path=report_path,
        output_dir=tmp_path / "adapted",
        hard_byte_ceilings=(178_000,),
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "adapted_snerv_advisory_report_to_spine"
    assert out["source_snerv_advisory_report_sha256"] == runner_mod._sha256_file(
        report_path
    )
    post_export = out["post_export_materializer_plan"]
    assert post_export["compiled"] is True
    assert post_export["archive_record"]["source_runtime_dir"].endswith(
        "/snerv_package/submission"
    )
    contexts = json.loads(Path(post_export["materializer_contexts_path"]).read_text())
    first_context = contexts["rows"][0]["context"]
    assert first_context["source_runtime_dir"].endswith("/snerv_package/submission")
    assert "--source-runtime-dir" in json.loads(
        Path(post_export["materializer_work_queue_path"]).read_text()
    )["rows"][0]["command"]
    assert out["post_export_materializer_execution"]["requested"] is False
    assert "paired_contest_cpu_cuda_auth_eval_missing" in out["blockers"]
    assert out["source_parity_contract"]["schema"] == (
        "nerv_source_parity_contract.v1"
    )
    assert out["source_parity_required_for_long_training_ready"] is True
    assert out["source_parity_blockers"] == []
    assert "source_parity:snerv_official_mfu_hfr_tub_parity_missing" in out[
        "source_parity_nonblocking_gaps"
    ]
    legacy_contract = out["legacy_advisory_ingest_contract"]
    assert legacy_contract["schema"] == "snerv_legacy_advisory_ingest_contract.v1"
    assert legacy_contract["source_parity_consumed"] is True
    assert legacy_contract["legacy_advisory_is_not_score_authority"] is True
    assert legacy_contract["score_claim"] is False
    assert legacy_contract["ready_for_exact_eval_dispatch"] is False


def test_adapt_snerv_advisory_report_uses_package_dir_archive_fallback(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "snerv_package"
    submission = package_dir / "submission"
    submission.mkdir(parents=True)
    (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (submission / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    _write_synthetic_pr95_archive(package_dir / "archive.zip", pairs=600)
    proof_path = package_dir / "snerv_inverse_steg_receiver_proof.json"
    proof_payload = {
        "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
        "proof_path": proof_path.as_posix(),
        "runtime_consumption_proof_passed": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    proof_path.write_text(json.dumps(proof_payload), encoding="utf-8")
    report_path = tmp_path / "snerv_advisory.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "snerv_inverse_steg_advisory.v1",
                "n_pairs": 600,
                "runtime_package_dir": package_dir.as_posix(),
                "runtime_package": {
                    "archive_bound_candidate_adapter_package": {
                        "candidate_rows": []
                    },
                    "receiver_proof": proof_payload,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    out = runner_mod.adapt_snerv_advisory_report_to_spine(
        snerv_advisory_report_path=report_path,
        output_dir=tmp_path / "adapted",
        hard_byte_ceilings=(178_000,),
        repo_root=REPO_ROOT,
    )

    post_export = out["post_export_materializer_plan"]
    assert post_export["compiled"] is True
    assert post_export["archive_record"]["absolute_path"].endswith(
        "/snerv_package/archive.zip"
    )
    assert "local_cpu_replay_waiting_for_full_video_mlx_prefilter" in out[
        "blockers"
    ]


def test_adapt_snerv_advisory_report_accepts_archive_package_manifest(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "snerv_package"
    submission = package_dir / "submission"
    submission.mkdir(parents=True)
    (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (submission / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    archive = _write_synthetic_pr95_archive(package_dir / "archive.zip", pairs=600)
    archive_sha = runner_mod._sha256_file(archive)
    archive_bytes = archive.stat().st_size
    proof_path = package_dir / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
    proof_path.parent.mkdir()
    proof_payload = {
        "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
        "proof_path": proof_path.as_posix(),
        "runtime_consumption_proof_passed": True,
        "runtime_consumption_proof_ready": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    proof_path.write_text(json.dumps(proof_payload), encoding="utf-8")
    package_manifest = package_dir / "archive_bound_candidate_adapter_package.json"
    package_manifest.write_text(
        json.dumps(
            {
                "schema": "snerv_inverse_steg_archive_bound_candidate_package.v1",
                "archive_bound_candidate_adapter_package": {
                    "candidate_rows": [
                        {
                            "candidate_archive_path": archive.as_posix(),
                            "candidate_archive_sha256": archive_sha,
                            "candidate_archive_bytes": archive_bytes,
                            "runtime_adapter_manifest": {"n_pairs": 600},
                            "blockers": [
                                "paired_contest_cpu_cuda_auth_eval_missing"
                            ],
                            "score_claim": False,
                            "promotion_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                        }
                    ]
                },
                "receiver_proof": proof_payload,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    out = runner_mod.adapt_snerv_advisory_report_to_spine(
        snerv_advisory_report_path=package_manifest,
        output_dir=tmp_path / "adapted",
        hard_byte_ceilings=(178_000,),
        repo_root=REPO_ROOT,
    )

    assert out["source_snerv_advisory_report_sha256"] == runner_mod._sha256_file(
        package_manifest
    )
    assert out["runtime_package_dir"].endswith("/snerv_package")
    assert out["archive_sha256"] == archive_sha
    assert out["receiver_proof_report_paths"] == [proof_path.as_posix()]
    assert "snerv_packet_not_full_600_pairs" not in out["blockers"]
    assert "paired_contest_cpu_cuda_auth_eval_missing" in out["blockers"]
    assert out["post_export_materializer_plan"]["archive_record"][
        "source_runtime_dir"
    ].endswith("/snerv_package/submission")


def test_pr95_stage8_execute_parser_exposes_source_lane_controls() -> None:
    args = _parse_args(
        [
            "--execute-pr95-stage8-source",
            "--stage8-epochs",
            "25",
            "--stage8-eval-every",
            "5",
            "--stage8-batch-size",
            "4",
            "--stage8-device",
            "cpu",
            "--stage8-muon-weight-decay",
            "0.0007",
            "--stage8-target-cache-path",
            "targets.pt",
            "--stage8-no-build-target-cache",
        ]
    )

    assert args.execute_pr95_stage8_source is True
    assert args.stage8_epochs == 25
    assert args.stage8_eval_every == 5
    assert args.stage8_batch_size == 4
    assert args.stage8_device == "cpu"
    assert args.stage8_muon_weight_decay == 0.0007
    assert args.stage8_target_cache_path == Path("targets.pt")
    assert args.stage8_no_build_target_cache is True


def test_adapt_pr95_stage8_report_emits_spine_runner_and_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = _write_synthetic_pr95_archive(tmp_path / "stage8_archive.zip")
    stage8_report_path = tmp_path / "pr95_stage8_report.json"
    stage8_report = {
        "schema": "pr95_stage8_from_public_archive_lane.v1",
        "mode": "execute",
        "report_path": stage8_report_path.as_posix(),
        "source_archive_zip": archive.as_posix(),
        "candidate_archive_zip_path": archive.as_posix(),
        "candidate_archive_zip_bytes": archive.stat().st_size,
        "candidate_archive_zip_sha256": runner_mod._sha256_file(archive),
        "local_training_result": {
            "raw_result": {"public_stage8_train_stage_called": True}
        },
        "exact_gate": {
            "schema": "exact_gate_blocker.v1",
            "blockers": ["contest_cpu_cuda_exact_eval_missing"],
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    stage8_report_path.write_text(
        json.dumps(stage8_report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    def fake_receiver_proof(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        proof_path = out / "pr95_hnerv_receiver_proof.json"
        report = {
            "schema": "pr95_hnerv_receiver_proof.v1",
            "report_path": proof_path.as_posix(),
            "archive_zip_path": Path(kwargs["archive_zip"]).as_posix(),
            "archive_zip_sha256": runner_mod._sha256_file(Path(kwargs["archive_zip"])),
            "runtime_consumption_proof_passed": True,
            "receiver_contract_satisfied": True,
            "receiver_output_kind": "contest_raw_rgb_interleaved",
            "receiver_output_bytes": 1,
            "receiver_proof_valid": True,
            "blockers": [],
            "exact_readiness_refusal": {
                "schema": "exact_readiness_refusal.v1",
                "ready": False,
                "blockers": ["runtime_consumption_smoke_is_not_score_authority"],
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        proof_path.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(runner_mod, "run_pr95_hnerv_receiver_proof", fake_receiver_proof)

    out = adapt_pr95_stage8_report_to_spine(
        pr95_stage8_report_path=stage8_report_path,
        output_dir=tmp_path / "adapted",
        hard_byte_ceilings=(178_000,),
        run_receiver_proof=True,
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "adapted_pr95_stage8_public_archive_report"
    assert out["score_claim"] is False
    assert out["stage8_source_faithfulness"]["public_stage8_train_stage_called"] is True
    assert out["stage8_source_faithfulness"]["source_faithful_training_complete"] is True
    assert out["receiver_proof_report_paths"]
    runner = json.loads(Path(out["bounded_runner_plan_path"]).read_text())
    row = runner["selected_runner_rows"][0]
    assert row["receiver_proof_observed"] is True
    assert row["receiver_proof_passed"] is True
    assert "contest_cpu_cuda_exact_eval_missing" in out["blockers"]
    assert "runtime_consumption_smoke_is_not_score_authority" in out["blockers"]


def test_adapt_pr95_stage8_report_reuses_embedded_package_proof(
    tmp_path: Path,
) -> None:
    archive = _write_synthetic_pr95_archive(tmp_path / "stage8_archive.zip")
    proof_path = tmp_path / "embedded_receiver_proof.json"
    proof = {
        "schema": "pr95_mlx_pytorch_package_receiver_proof.v1",
        "proof_path": proof_path.as_posix(),
        "archive_path": archive.as_posix(),
        "archive_sha256": runner_mod._sha256_file(archive),
        "runtime_consumption_proof_passed": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    proof_path.write_text(
        json.dumps(proof, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    stage8_report_path = tmp_path / "pr95_stage8_report.json"
    stage8_report_path.write_text(
        json.dumps(
            {
                "schema": "pr95_stage8_from_public_archive_lane.v1",
                "mode": "execute",
                "candidate_archive_zip_path": archive.as_posix(),
                "candidate_archive_zip_bytes": archive.stat().st_size,
                "candidate_archive_zip_sha256": runner_mod._sha256_file(archive),
                "package_report": {
                    "archive_bound_candidate_receiver_proof": proof,
                },
                "local_training_result": {
                    "raw_result": {"public_stage8_train_stage_called": False}
                },
                "exact_gate": {"blockers": []},
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    out = adapt_pr95_stage8_report_to_spine(
        pr95_stage8_report_path=stage8_report_path,
        output_dir=tmp_path / "adapted",
        hard_byte_ceilings=(178_000,),
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    assert out["receiver_proof_report_paths"] == [proof_path.as_posix()]
    assert "receiver_proof_not_executed" not in out["blockers"]
    runner = json.loads(Path(out["bounded_runner_plan_path"]).read_text())
    row = runner["selected_runner_rows"][0]
    assert row["receiver_proof_observed"] is True
    assert row["receiver_proof_passed"] is True


def test_pr95_hnerv_execute_arm_emits_runner_and_fail_closed_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "pr95_public_archive.zip"
        _write_synthetic_pr95_archive(archive)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
        }

    def fake_receiver_proof(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = Path(kwargs["archive_zip"])
        report = {
            "schema": "pr95_hnerv_receiver_proof.v1",
            "proof_path": (out / "pr95_hnerv_receiver_proof.json").as_posix(),
            "archive_path": archive.as_posix(),
            "archive_sha256": runner_mod._sha256_file(archive),
            "runtime_consumption_proof_passed": True,
            "receiver_contract_satisfied": True,
            "receiver_output_kind": "contest_raw_rgb_interleaved",
            "receiver_output_bytes": 1,
            "receiver_proof_valid": True,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "exact_readiness_refusal": {
                "schema": "exact_readiness_refusal.v1",
                "ready": False,
                "blockers": ["runtime_consumption_smoke_is_not_score_authority"],
            },
        }
        Path(report["proof_path"]).write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(
        runner_mod,
        "_run_pr95_hnerv_mlx_scoreaware_smoke",
        fake_train,
    )
    monkeypatch.setattr(runner_mod, "run_pr95_hnerv_receiver_proof", fake_receiver_proof)

    out = execute_pr95_hnerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "run",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-5,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        source_archive_zip=tmp_path / "synthetic_source_pr95.zip",
        hard_byte_ceilings=(178_000,),
        run_receiver_proof=True,
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert out["projection_manifest_paths"]
    assert out["receiver_proof_report_paths"]
    assert Path(out["acquisition_report_path"]).is_file()
    runner = json.loads(Path(out["bounded_runner_plan_path"]).read_text())
    row = runner["selected_runner_rows"][0]
    assert row["family"] == "pr95_hnerv"
    assert row["receiver_proof_observed"] is True
    assert row["receiver_proof_passed"] is True
    post_export = out["post_export_materializer_plan"]
    assert post_export["schema"] == "compact_carrier_post_export_materializer_plan.v1"
    assert post_export["compiled"] is True
    assert Path(post_export["experiment_queue_path"]).is_file()
    assert out["post_export_materializer_execution"]["requested"] is False
    assert out["post_export_materializer_execution"]["executed"] is False
    assert out["control_arm_scope"]["source_faithful_pr95_reproduction"] is False
    assert (
        "pr95_hnerv_mlx_archive_export_control_arm_not_pr95_faithful_reproduction"
        in out["blockers"]
    )
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in out["blockers"]


def test_pr95_hnerv_execute_arm_uses_real_scorer_binding_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "pr95_public_archive.zip"
        _write_synthetic_pr95_archive(archive)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "score_aware_training": {
                    "schema": "mlx_score_aware_training_objective.v1",
                    "segnet_distillation_weight": 0.05,
                    "pose_distillation_weight": 0.0005,
                    "has_real_segnet_teacher": True,
                    "has_real_posenet_teacher": True,
                    "allow_mock_scorer_teacher": False,
                    "allow_segnet_only_research": False,
                }
            },
        }

    def fake_receiver_proof(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = Path(kwargs["archive_zip"])
        report = {
            "schema": "pr95_hnerv_receiver_proof.v1",
            "proof_path": (out / "pr95_hnerv_receiver_proof.json").as_posix(),
            "archive_path": archive.as_posix(),
            "archive_sha256": runner_mod._sha256_file(archive),
            "runtime_consumption_proof_passed": True,
            "receiver_proof_valid": True,
            "blockers": [],
            "exact_readiness_refusal": {
                "schema": "exact_readiness_refusal.v1",
                "ready": False,
                "blockers": ["runtime_consumption_smoke_is_not_score_authority"],
            },
        }
        Path(report["proof_path"]).write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(
        runner_mod,
        "_run_pr95_hnerv_mlx_scoreaware_smoke",
        fake_train,
    )
    monkeypatch.setattr(runner_mod, "run_pr95_hnerv_receiver_proof", fake_receiver_proof)

    out = execute_pr95_hnerv_mlx_scoreaware_and_adapt(
        output_dir=tmp_path / "run",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-5,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        source_archive_zip=tmp_path / "synthetic_source_pr95.zip",
        hard_byte_ceilings=(178_000,),
        segnet_distillation_weight=0.05,
        pose_distillation_weight=0.0005,
        run_receiver_proof=True,
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    assert "pr95_segnet_posenet_network_loss_not_wired_to_mlx" not in out["blockers"]
    assert (
        "pr95_source_video_rgb_yuv6_preprocess_loss_is_not_full_scorer_loss"
        not in out["blockers"]
    )
    assert (
        "pr95_hnerv_default_scorer_distillation_weights_are_zero_unless_cli_overridden"
        not in out["blockers"]
    )
    assert (
        "pr95_mlx_scoreaware_teacher_distillation_is_advisory_not_exact_contest_loss"
        in out["blockers"]
    )
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in out["blockers"]


def test_vq_execute_forwards_optimizer_controls_and_shared_qat_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": runner_mod._sha256_file(archive),
            "substrate_artifact_metadata": {
                "family": "pact_nerv_vq",
            },
        }

    monkeypatch.setattr(runner_mod, "_run_pact_nerv_vq_mlx_smoke", fake_train)

    out = runner_mod.execute_pact_nerv_vq_mlx_smoke_and_adapt(
        output_dir=tmp_path / "vq_run",
        num_pairs=2,
        epochs=5,
        batch_pair_indices_per_step=1,
        learning_rate=1e-4,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        latent_dim=8,
        embed_dim=8,
        codebook_size=16,
        decoder_channel=8,
        decoder_codec="int4_scale_bundled",
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0001,
        coder_qat_delta_weight=0.0002,
        coder_qat_c1a_entropy_weight=0.0003,
        coder_qat_c1a_sigma=0.35,
        coder_qat_c1a_sample_size=64,
        optimizer_kind="lion",
        optimizer_grad_clip_max_norm=0.75,
        optimizer_weight_decay=2.0e-4,
        optimizer_warmup_epochs=2,
        optimizer_warmup_steps_per_epoch=3,
        optimizer_cosine_decay_enabled=True,
        optimizer_cosine_decay_total_epochs=5,
        optimizer_cosine_decay_min_lr_ratio=0.025,
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    optimizer_controls = captured_train_kwargs["optimizer_controls"]
    assert optimizer_controls["optimizer_kind"] == "lion"
    assert optimizer_controls["grad_clip_max_norm"] == pytest.approx(0.75)
    assert optimizer_controls["weight_decay_effective"] == pytest.approx(2.0e-4)
    assert optimizer_controls["warmup_epochs"] == 2
    assert optimizer_controls["warmup_steps_per_epoch"] == 3
    assert optimizer_controls["cosine_decay_enabled"] is True
    assert optimizer_controls["cosine_decay_total_epochs"] == 5
    assert optimizer_controls["cosine_decay_min_lr_ratio"] == pytest.approx(0.025)
    optimizer_policy = captured_train_kwargs["optimizer_policy"]
    assert optimizer_policy["schema"] == "compact_pact_native_optimizer_policy.v1"
    assert optimizer_policy["resolved_policy"] == "native_optimizer"
    assert optimizer_policy["optimizer_kind_consumed_by_native_mlx"] is True
    assert optimizer_policy["optimizer_kind_consumed_by_pr95_curriculum"] is False
    assert optimizer_policy["pr95_faithful_curriculum_enabled"] is False
    assert captured_train_kwargs["optimizer_kind"] == "lion"
    assert captured_train_kwargs["coder_aware_qat"] is True
    assert captured_train_kwargs["coder_qat_quant_bits"] == 4

    report_training = out["score_aware_training"]
    assert report_training["optimizer_policy"]["schema"] == optimizer_policy["schema"]
    assert report_training["optimizer_policy"]["resolved_policy"] == (
        optimizer_policy["resolved_policy"]
    )
    assert report_training["optimizer_policy"][
        "optimizer_kind_consumed_by_native_mlx"
    ] is True
    assert report_training["optimizer_policy"][
        "optimizer_kind_consumed_by_pr95_curriculum"
    ] is False
    assert report_training["optimizer_controls"]["optimizer_kind"] == (
        optimizer_controls["optimizer_kind"]
    )
    assert report_training["optimizer_controls"]["grad_clip_max_norm"] == (
        optimizer_controls["grad_clip_max_norm"]
    )
    assert report_training["optimizer_controls"]["weight_decay_effective"] == (
        optimizer_controls["weight_decay_effective"]
    )
    assert report_training["optimizer_kind"] == "lion"
    assert report_training["native_optimizer_active"] is True
    assert report_training["effective_weight_decay"] == pytest.approx(2.0e-4)
    qat = report_training["coder_aware_qat"]
    assert qat["schema"] == "coder_aware_decoder_qat.v1"
    assert qat["enabled"] is True
    assert qat["quant_bits"] == 4
    assert qat["authority"] == "false_macos_mlx_research_signal"
    assert qat["authority_status"] == (
        "advisory_training_loss_only_not_archive_or_score_authority"
    )
    assert "quantizer_geometry" in qat
    assert "include_substrings" in qat
    assert "exclude_substrings" in qat
    assert "pact_nerv_vq_spine_projection_manifest_missing" in out["blockers"]
    assert out["score_claim"] is False
    assert out["ready_for_exact_eval_dispatch"] is False


def test_selector_v4_execute_arm_emits_runner_and_fail_closed_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}

    def fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        _write_synthetic_pr95_archive(archive)
        archive_sha = runner_mod._sha256_file(archive)
        archive_bytes = archive.stat().st_size
        receiver_dir = out / "receiver_proof"
        receiver_dir.mkdir(parents=True, exist_ok=True)
        receiver_proof = {
            "schema": "pact_nerv_selector_v4_mlx_generated_receiver_proof.v1",
            "proof_path": (
                receiver_dir / "pact_nerv_selector_v4_mlx_receiver_proof.json"
            ).as_posix(),
            "archive_path": archive.as_posix(),
            "archive_zip_path": archive.as_posix(),
            "archive_sha256": archive_sha,
            "archive_zip_sha256": archive_sha,
            "runtime_consumption_proof_passed": True,
            "receiver_contract_satisfied": True,
            "receiver_proof_valid": True,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "exact_readiness_refusal": {
                "schema": "exact_readiness_refusal.v1",
                "ready": False,
                "blockers": ["runtime_consumption_smoke_is_not_score_authority"],
            },
        }
        Path(receiver_proof["proof_path"]).write_text(
            json.dumps(receiver_proof, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        projection = {
            "schema": "hprc_representation_spine_projection.v1",
            "family": "pact_nerv",
            "hprc_bin_bytes": 100,
            "manifest": {
                "representation_spine": {
                    "schema": "hprc_representation_spine_manifest.v1",
                    "family": "pact_nerv",
                    "hprc_bin_bytes": 100,
                    "source": {
                        "kind": "pact_nerv_selector_v4_export_payload",
                        "archive_zip_path": archive.as_posix(),
                        "archive_zip_sha256": archive_sha,
                        "bytes": archive_bytes,
                        "sha256": archive_sha,
                    },
                    "manifest_extra": {
                        "emitted_by": "export_pact_nerv_selector_v4_mlx_archive",
                        "num_pairs": 600,
                        "num_frames": 1200,
                        "coverage_source": "test_selector_v4_execute_arm",
                        "selector_codec": "run_length_varint_selector",
                    },
                    "sections": [
                        {
                            "name": "decoder_qw",
                            "role": "charged_decoder_or_program_weights",
                            "bytes": 60,
                            "sha256": "a" * 64,
                        },
                        {
                            "name": "latents_rc",
                            "role": "charged_per_pair_latents",
                            "bytes": 40,
                            "sha256": "b" * 64,
                        },
                    ],
                }
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        manifest = (
            out / "hprc_representation_spine_pact_nerv_selector_v4_manifest.json"
        )
        runner_mod._write_json(manifest, projection)
        return {
            "archive_path": archive.as_posix(),
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha,
            "substrate_artifact_metadata": {
                "family": "pact_nerv_selector_v4",
                "selector_codec": "run_length_varint_selector",
            },
        }

    monkeypatch.setattr(
        runner_mod,
        "_run_pact_nerv_selector_v4_mlx_smoke",
        fake_train,
    )
    projection_manifest_path = (
        tmp_path
        / "run"
        / "pact_nerv_selector_v4_mlx_training"
        / "hprc_representation_spine_pact_nerv_selector_v4_manifest.json"
    )
    mlx_profile_path = tmp_path / "selector_section_value_profile.json"
    mlx_profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "family": "pact_nerv",
                "max_pairs": 600,
                "scorer_batch_pairs": 1,
                "projection_manifest_path": projection_manifest_path.as_posix(),
                "scope_status": {
                    "full_video": True,
                    "axis": "[macOS-MLX research-signal]",
                },
                "section_value_rows": [
                    {
                        "variant_id": "neutralize_decoder_qw",
                        "neutralized_section": "decoder_qw",
                        "archive_bytes_removed_vs_baseline": 60,
                        "delta_nonrate_score": 0.25,
                        "delta_total_mlx_score_advisory": 0.25,
                        "family": "pact_nerv",
                        "projection_manifest_path": (
                            projection_manifest_path.as_posix()
                        ),
                        "marginal_status": "measured_full_video_mlx_advisory",
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    out = execute_pact_nerv_selector_v4_mlx_smoke_and_adapt(
        output_dir=tmp_path / "run",
        num_pairs=600,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-5,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=8,
        embed_dim=8,
        selector_palette_size=16,
        decoder_channel=8,
        decoder_codec="int4_scale_bundled",
        mlx_profile_paths=(mlx_profile_path,),
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0001,
        coder_qat_delta_weight=0.0002,
        coder_qat_c1a_entropy_weight=0.0003,
        coder_qat_c1a_sigma=0.35,
        coder_qat_c1a_sample_size=64,
        optimizer_kind="adafactor",
        optimizer_grad_clip_max_norm=0.5,
        optimizer_weight_decay=3.0e-4,
        optimizer_warmup_epochs=1,
        optimizer_warmup_steps_per_epoch=2,
        optimizer_cosine_decay_enabled=True,
        optimizer_cosine_decay_total_epochs=3,
        optimizer_cosine_decay_min_lr_ratio=0.05,
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "executed_pact_nerv_selector_v4_mlx_smoke_and_exported"
    assert out["execute_family"] == "pact_nerv_selector_v4"
    assert out["score_claim"] is False
    assert out["promotion_eligible"] is False
    assert out["ready_for_exact_eval_dispatch"] is False
    assert out["selector_v4_archive_surface"]["selector_codec"] == (
        "run_length_varint_selector"
    )
    assert out["score_aware_training"]["scorer_coupled_rd"][
        "fixed_marginal_byte_price"
    ] == "25/uncompressed_total"
    optimizer_controls = captured_train_kwargs["optimizer_controls"]
    assert optimizer_controls["optimizer_kind"] == "adafactor"
    assert optimizer_controls["grad_clip_max_norm"] == pytest.approx(0.5)
    assert optimizer_controls["weight_decay_effective"] == pytest.approx(3.0e-4)
    assert optimizer_controls["warmup_epochs"] == 1
    assert optimizer_controls["warmup_steps_per_epoch"] == 2
    assert optimizer_controls["cosine_decay_enabled"] is True
    assert optimizer_controls["cosine_decay_total_epochs"] == 3
    assert optimizer_controls["cosine_decay_min_lr_ratio"] == pytest.approx(0.05)
    optimizer_policy = captured_train_kwargs["optimizer_policy"]
    assert optimizer_policy["schema"] == "compact_pact_native_optimizer_policy.v1"
    assert optimizer_policy["family"] == "pact_nerv_selector_v4"
    assert optimizer_policy["resolved_policy"] == "native_optimizer"
    assert optimizer_policy["optimizer_kind_consumed_by_native_mlx"] is True
    assert optimizer_policy["optimizer_kind_consumed_by_pr95_curriculum"] is False
    assert optimizer_policy["pr95_faithful_curriculum_enabled"] is False
    assert out["score_aware_training"]["optimizer_policy"]["schema"] == (
        optimizer_policy["schema"]
    )
    assert out["score_aware_training"]["optimizer_policy"]["resolved_policy"] == (
        optimizer_policy["resolved_policy"]
    )
    assert out["score_aware_training"]["optimizer_policy"][
        "optimizer_kind_consumed_by_native_mlx"
    ] is True
    assert out["score_aware_training"]["optimizer_policy"][
        "optimizer_kind_consumed_by_pr95_curriculum"
    ] is False
    assert out["score_aware_training"]["optimizer_controls"]["optimizer_kind"] == (
        optimizer_controls["optimizer_kind"]
    )
    assert out["score_aware_training"]["optimizer_controls"][
        "grad_clip_max_norm"
    ] == optimizer_controls["grad_clip_max_norm"]
    assert out["score_aware_training"]["optimizer_controls"][
        "weight_decay_effective"
    ] == optimizer_controls["weight_decay_effective"]
    assert out["score_aware_training"]["optimizer_kind"] == "adafactor"
    assert out["score_aware_training"]["native_optimizer_active"] is True
    assert out["score_aware_training"]["effective_weight_decay"] == pytest.approx(
        3.0e-4
    )
    qat = out["score_aware_training"]["coder_aware_qat"]
    assert qat["schema"] == "coder_aware_decoder_qat.v1"
    assert qat["enabled"] is True
    assert qat["quant_bits"] == 4
    assert qat["quant_residual_weight"] == pytest.approx(0.001)
    assert qat["magnitude_weight"] == pytest.approx(0.0001)
    assert qat["delta_weight"] == pytest.approx(0.0002)
    assert qat["c1a_entropy_weight"] == pytest.approx(0.0003)
    assert qat["c1a_sigma"] == pytest.approx(0.35)
    assert qat["c1a_sample_size"] == 64
    assert qat["authority"] == "false_macos_mlx_research_signal"
    assert qat["authority_status"] == (
        "advisory_training_loss_only_not_archive_or_score_authority"
    )
    assert "quantizer_geometry" in qat
    assert "include_substrings" in qat
    assert "exclude_substrings" in qat
    assert out["score_aware_training"]["decoder_codec"] == "int4_scale_bundled"
    assert captured_train_kwargs["optimizer_kind"] == "adafactor"
    assert captured_train_kwargs["coder_aware_qat"] is True
    assert captured_train_kwargs["decoder_codec"] == "int4_scale_bundled"
    assert captured_train_kwargs["coder_qat_quant_bits"] == 4
    assert captured_train_kwargs["coder_qat_quant_residual_weight"] == 0.001
    assert captured_train_kwargs["coder_qat_magnitude_weight"] == 0.0001
    assert captured_train_kwargs["coder_qat_delta_weight"] == 0.0002
    assert captured_train_kwargs["coder_qat_c1a_entropy_weight"] == 0.0003
    assert captured_train_kwargs["coder_qat_c1a_sigma"] == 0.35
    assert captured_train_kwargs["coder_qat_c1a_sample_size"] == 64
    assert out["projection_manifest_paths"]
    assert out["receiver_proof_report_paths"]
    assert out["mlx_profile_paths"] == [mlx_profile_path.as_posix()]
    assert Path(out["acquisition_report_path"]).is_file()
    runner = json.loads(Path(out["bounded_runner_plan_path"]).read_text())
    assert runner["mlx_profile_paths"] == [mlx_profile_path.as_posix()]
    row = runner["selected_runner_rows"][0]
    assert row["family"] == "pact_nerv"
    assert row["receiver_proof_observed"] is True
    assert row["receiver_proof_passed"] is True
    value_rows = {item["section_name"]: item for item in runner["section_value_rows"]}
    assert value_rows["decoder_qw"]["evidence_status"] == "measured_mlx_advisory"
    assert value_rows["decoder_qw"]["admission_status"] == (
        "admit_section_bytes_for_receiver_proof"
    )
    post_export = out["post_export_materializer_plan"]
    assert post_export["schema"] == "compact_carrier_post_export_materializer_plan.v1"
    assert post_export["compiled"] is True
    assert Path(post_export["experiment_queue_path"]).is_file()
    assert out["post_export_materializer_execution"]["requested"] is False
    assert out["post_export_materializer_execution"]["executed"] is False
    assert "full_video_mlx_scorer_replay_not_attached" not in out["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in out["blockers"]
    assert "pact_nerv_selector_v4_spine_projection_manifest_missing" not in out[
        "blockers"
    ]


def test_selector_v4_execute_arm_threads_render_quality_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_train(**_kwargs):
        return {
            "archive_path": None,
            "archive_bytes": None,
            "archive_sha256": None,
            "substrate_artifact_metadata": {
                "selector_v4_render_quality": {
                    "schema": "pact_nerv_selector_v4_mlx_render_quality.v1",
                    "verdict": "RENDER_OUTPUT_DEGENERATE_BLOCK_ARCHIVE_PROFILE",
                    "export_blocked_recommended": True,
                    "blockers": [
                        "selector_v4_render_segnet_last_frame_std_too_low",
                    ],
                },
            },
        }

    monkeypatch.setattr(
        runner_mod,
        "_run_pact_nerv_selector_v4_mlx_smoke",
        fake_train,
    )

    out = execute_pact_nerv_selector_v4_mlx_smoke_and_adapt(
        output_dir=tmp_path / "run",
        num_pairs=2,
        epochs=1,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        allow_overwrite=True,
        repo_root=REPO_ROOT,
    )

    assert out["execute_family"] == "pact_nerv_selector_v4"
    assert out["selector_v4_render_quality"]["export_blocked_recommended"] is True
    assert "selector_v4_render_segnet_last_frame_std_too_low" in out["blockers"]
    assert "pact_nerv_selector_v4_render_quality_gate_failed" in out["blockers"]
    assert "pact_nerv_selector_v4_archive_export_missing" in out["blockers"]
