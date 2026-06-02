# SPDX-License-Identifier: MIT
"""Tests for the MLX-first compact renderer spine runner."""

from __future__ import annotations

import ast
import json
import os
import struct
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.run_compact_renderer_mlx_spine_runner as runner_mod  # noqa: E402
from tools.run_compact_renderer_mlx_spine_runner import (  # noqa: E402
    COMPACT_RENDERER_MLX_SPINE_RUNNER_SCHEMA,
    _parse_args,
    _require_scorer_upstream_dir_for_distillation,
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


def _write_synthetic_pr95_archive(path: Path, *, pairs: int = 600) -> Path:
    chunks = []
    for payload in (f'{{"pairs":{pairs}}}'.encode(), b"decoder", b"latents"):
        chunks.append(struct.pack("<I", len(payload)))
        chunks.append(payload)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", b"".join(chunks))
    return path


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
        "executable_archive_bound_cpu_advisory_mlx_migration_required"
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


def test_active_campaign_lock_refuses_duplicate_active_pid(tmp_path: Path) -> None:
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

    lock_path = runner_mod._acquire_active_campaign_lock(
        output_dir=tmp_path / "a",
        args=args,
        source_video_path=source,
        hard_byte_ceilings=(178_000,),
    )
    assert lock_path is not None
    assert lock_path.is_file()
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


def test_pact_vq_runner_forwards_pr95_curriculum_kwargs() -> None:
    source = Path(runner_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    target_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_pact_nerv_vq_mlx_smoke"
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
            "--hprc-queue-followup-report",
            "hprc_queue_followup_report.json",
        ]
    )

    assert args.segnet_distillation_weight == 0.25
    assert args.upstream_dir == Path("canonical_upstream")
    assert args.pose_distillation_weight == 0.75
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
            "--auto-segnet-boundary-recon-weight",
            "--recon-pixel-weight-tau",
            "0.75",
            "--recon-pixel-weight-normalize",
            "none",
            "--mlx-prefilter-scorer-batch-pairs",
            "8",
            "--mlx-prefilter-progress-every",
            "10",
            "--run-post-export-materializers",
            "--post-export-materializer-max-steps",
            "3",
            "--post-export-materializer-max-parallel",
            "2",
            "--post-export-materializer-max-experiments",
            "1",
        ]
    )
    sn = _parse_args(["--execute-family", "snerv", "--num-pairs", "128"])

    assert hi.execute_family == "hi_nerv"
    assert hi.num_pairs == 32
    assert hi.run_local_cpu_replay is True
    assert hi.keep_local_replay_inflated is True
    assert hi.retain_failed_local_replay_scratch is True
    assert hi.recon_pixel_weight_path == Path("weights.npz")
    assert hi.auto_segnet_boundary_recon_weight is True
    assert hi.recon_pixel_weight_tau == 0.75
    assert hi.recon_pixel_weight_normalize == "none"
    assert hi.mlx_prefilter_scorer_batch_pairs == 8
    assert hi.mlx_prefilter_progress_every == 10
    assert hi.run_post_export_materializers is True
    assert hi.post_export_materializer_max_steps == 3
    assert hi.post_export_materializer_max_parallel == 2
    assert hi.post_export_materializer_max_experiments == 1
    assert sn.execute_family == "snerv"
    assert sn.num_pairs == 128
    assert sn.post_export_materializer_max_experiments == 1


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
        epochs=8,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        mlx_profile_paths=(mlx_profile_path,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        mlx_prefilter_scorer_batch_pairs=4,
        mlx_prefilter_progress_every=7,
        upstream_dir=tmp_path / "canonical_upstream",
        repo_root=REPO_ROOT,
    )

    assert Path(captured_train_kwargs["scorer_upstream_dir"]) == (
        tmp_path / "canonical_upstream"
    )
    assert captured_train_kwargs["mlx_prefilter_scorer_batch_pairs"] == 4
    assert captured_train_kwargs["mlx_prefilter_progress_every"] == 7
    assert out["score_aware_training"]["local_mlx_prefilter"] == {
        "schema": "compact_hi_nerv_local_mlx_prefilter_config.v1",
        "scorer_batch_pairs": 4,
        "progress_every": 7,
        "singleton_required_for_local_cpu_replay_unlock": True,
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
        epochs=8,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
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
        epochs=8,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        mlx_profile_paths=(sampled_profile,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
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
        epochs=8,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        mlx_profile_paths=(bad_profile,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
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
        epochs=8,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        repo_root=REPO_ROOT,
    )

    assert out["local_cpu_replay_gate"]["executed"] is False
    assert out["local_cpu_replay_gate"]["default_enabled_for_full_coverage"] is False
    assert out["local_cpu_replay_gate"]["has_full_video_mlx_prefilter"] is False
    assert "local_cpu_replay_waiting_for_full_video_mlx_prefilter" in out[
        "blockers"
    ]


def test_hinerv_execute_threads_coder_qat_and_reads_substrate_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_train_kwargs: dict[str, object] = {}
    weight_path = tmp_path / "joint_p18_p19_weight.npy"
    np.save(weight_path, np.ones((384, 512), dtype=np.float32))

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
        epochs=8,
        batch_pair_indices_per_step=1,
        learning_rate=1e-3,
        source_video_path=REPO_ROOT / "upstream/videos/0.mkv",
        hard_byte_ceilings=(178_000,),
        latent_dim=4,
        embed_dim=4,
        decoder_channel=4,
        decoder_codec="int2_scale_bundled",
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        coder_qat_quant_residual_weight=0.001,
        coder_qat_magnitude_weight=0.0001,
        coder_qat_delta_weight=0.0002,
        recon_pixel_weight_path=weight_path,
        auto_segnet_boundary_recon_weight=False,
        recon_pixel_weight_tau=0.5,
        recon_pixel_weight_normalize="mean",
        repo_root=REPO_ROOT,
    )

    assert captured_train_kwargs["coder_aware_qat"] is True
    assert captured_train_kwargs["decoder_codec"] == "int2_scale_bundled"
    assert captured_train_kwargs["coder_qat_quant_bits"] == 4
    assert captured_train_kwargs["coder_qat_quant_residual_weight"] == 0.001
    assert captured_train_kwargs["coder_qat_magnitude_weight"] == 0.0001
    assert captured_train_kwargs["coder_qat_delta_weight"] == 0.0002
    assert captured_train_kwargs["recon_pixel_weight_path"] == weight_path
    assert captured_train_kwargs["auto_segnet_boundary_recon_weight"] is False
    assert captured_train_kwargs["recon_pixel_weight_tau"] == 0.5
    assert captured_train_kwargs["recon_pixel_weight_normalize"] == "mean"
    assert out["score_aware_training"]["coder_aware_qat"] == {
        "schema": "coder_aware_decoder_qat.v1",
        "enabled": True,
        "quant_bits": 4,
        "quant_residual_weight": 0.001,
        "magnitude_weight": 0.0001,
        "delta_weight": 0.0002,
        "authority": "false_macos_mlx_research_signal",
    }
    assert out["score_aware_training"]["decoder_codec"] == "int2_scale_bundled"
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


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX required for HiNeRV adapter smoke")
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
    assert out["projection_manifest_paths"]
    assert out["receiver_proof_report_paths"]
    proof = json.loads(Path(out["receiver_proof_report_paths"][0]).read_text())
    assert proof["runtime_consumption_proof_ready"] is True
    assert proof["receiver_contract_satisfied"] is True
    assert "hi_nerv_real_segnet_posenet_teachers_not_both_attached" in out[
        "blockers"
    ]
    assert "hi_nerv_pr95_faithful_curriculum_requires_min_8_epochs" in out[
        "blockers"
    ]
    assert "local_cpu_replay_not_run_partial_pair_coverage" in out["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in out["blockers"]
    assert Path(out["report_path"]).is_file()


def test_snerv_execution_writes_archive_bound_report_and_reusable_hooks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet = b"SNERVPACKET"

    def fake_run_snerv_advisory(**kwargs):
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
            receiver_archive_packet=packet,
            as_jsonable=as_jsonable,
            levels=3,
            wavelet="db2",
            score_linf=12.0,
            score_l2=13.0,
            d_seg_mean_linf=0.1,
            d_pose_mean_linf=0.01,
            archive_bytes_total=len(packet),
            beats_frontier_rate=True,
            receiver_archive_replay_verified=True,
        )

    def fake_export_snerv_archive_bound_candidate_package(**kwargs):
        package_dir = Path(kwargs["output_dir"])
        package_dir.mkdir(parents=True, exist_ok=True)
        archive = package_dir / "archive.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("0.bin", b"snerv-archive")
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
        repo_root=REPO_ROOT,
    )

    assert out["mode"] == "executed_snerv_archive_bound_advisory_and_exported"
    assert out["execute_family"] == "snerv"
    assert out["training_executed"] is False
    assert Path(out["archive_path"]).is_file()
    assert out["archive_bytes"] == Path(out["archive_path"]).stat().st_size
    assert Path(out["receiver_archive_packet_path"]).read_bytes() == packet
    assert Path(out["advisory_report_path"]).is_file()
    assert Path(out["runtime_package_path"]).is_file()
    assert out["receiver_proof_report_paths"]
    planner = out["score_aware_carrier_training_plan"]
    assert planner["score_aware_training_ready"] is False
    assert out["score_aware_training"]["status"] == (
        "executed_cpu_advisory_mlx_native_training_missing"
    )
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
    assert "snerv_mlx_native_train_export_archive_adapter_missing" in out[
        "blockers"
    ]
    assert "full_video_mlx_scorer_replay_not_attached" in out["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in out["blockers"]


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
    assert out["score_aware_training"]["coder_aware_qat"] == {
        "enabled": True,
        "quant_bits": 4,
        "quant_residual_weight": 0.001,
        "magnitude_weight": 0.0001,
        "delta_weight": 0.0002,
        "authority": "false_macos_mlx_research_signal",
    }
    assert out["score_aware_training"]["decoder_codec"] == "int4_scale_bundled"
    assert captured_train_kwargs["coder_aware_qat"] is True
    assert captured_train_kwargs["decoder_codec"] == "int4_scale_bundled"
    assert captured_train_kwargs["coder_qat_quant_bits"] == 4
    assert captured_train_kwargs["coder_qat_quant_residual_weight"] == 0.001
    assert captured_train_kwargs["coder_qat_magnitude_weight"] == 0.0001
    assert captured_train_kwargs["coder_qat_delta_weight"] == 0.0002
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
