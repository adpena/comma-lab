#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a queue-owned HPRC compact-receiver train/export campaign."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import normalize_queue_definition  # noqa: E402
from comma_lab.scheduler.local_training_queue import (  # noqa: E402
    build_local_training_execution_queue,
)
from comma_lab.storage_tiers import (  # noqa: E402
    DEFAULT_RESERVE_FREE_GB,
    StorageTierError,
    parse_storage_tier_specs,
    plan_experiment_storage,
    require_selected_storage,
)
from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact  # noqa: E402
from tac.substrates.hprc.archive_candidate import (  # noqa: E402
    FALSE_AUTHORITY,
    HPRC_RECEIVER_PROOF_SCRATCH_BYTES,
)

HPRC_TRAINING_PLAN_SCHEMA = "hprc_compact_receiver_training_plan.v1"
HPRC_TRAINING_PLAN_SUITE_SCHEMA = "hprc_compact_receiver_training_plan_suite.v1"
HPRC_TRAINING_QUEUE_BUILD_SCHEMA = "hprc_compact_receiver_training_queue_build.v1"
DEFAULT_HPRC_TRAINING_QUEUE_WORKLOAD_SUBDIR = "experiments/results/hprc_compact_receiver_training_queue"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--queue-id")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--video-path", type=Path, default=Path("upstream/videos/0.mkv"))
    parser.add_argument("--decode-pairs", type=int, default=8)
    parser.add_argument(
        "--campaign-pairs",
        action="append",
        type=int,
        default=[],
        help=(
            "Repeat to build a multi-scale campaign, e.g. 32/128/600. "
            "When omitted, --decode-pairs is used."
        ),
    )
    parser.add_argument("--decode-max-pairs", type=int)
    parser.add_argument("--decode-height", type=int, default=96)
    parser.add_argument("--decode-width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-pair-indices-per-step", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--basis-count", type=int, default=3)
    parser.add_argument("--residual-grid-h", type=int, default=24)
    parser.add_argument("--residual-grid-w", type=int, default=32)
    parser.add_argument(
        "--training-backend",
        choices=("auto", "mlx", "numpy"),
        default="auto",
        help=(
            "Local trainer backend. Default is MLX/Metal when available; exported "
            "archives remain numpy-portable for contest CPU/T4 auth eval."
        ),
    )
    parser.add_argument(
        "--enable-native-rate-aware-hprc",
        action="store_true",
        help="Train HPRC residual tokens with native rate pressure before archive export.",
    )
    parser.add_argument("--native-rate-residual-l1-weight", type=float, default=0.0)
    parser.add_argument("--native-rate-residual-prox-weight", type=float, default=0.0)
    parser.add_argument(
        "--native-rate-residual-protection-npy",
        type=Path,
        help="Precomputed residual protection surface consumed by native rate-aware training.",
    )
    parser.add_argument(
        "--native-rate-p19-posenet-null-pairs",
        type=Path,
        help="Optional P19 artifact used to build the native HPRC train-time protection surface.",
    )
    parser.add_argument(
        "--native-rate-p18-segnet-region-waterfill",
        type=Path,
        help="Optional P18 artifact used to protect SegNet-sensitive residual cells.",
    )
    parser.add_argument("--native-rate-default-protection", type=float, default=1.0)
    parser.add_argument("--native-rate-p19-null-protection", type=float, default=0.15)
    parser.add_argument("--native-rate-p18-region-protection", type=float, default=1.0)
    parser.add_argument("--local-cpu-concurrency", type=int, default=1)
    parser.add_argument("--local-mlx-concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--full-replay-min-pairs",
        type=int,
        default=600,
        help="Campaigns at or above this pair count get full local CPU replay and exact-auth gating.",
    )
    parser.add_argument("--local-replay-device", default="cpu", choices=("cpu",))
    parser.add_argument("--auth-frontier-score", type=float)
    parser.add_argument("--local-baseline-score", type=float)
    parser.add_argument("--min-local-improvement", type=float, default=0.0)
    parser.add_argument("--exact-auth-axis", default="[contest-CPU]")
    parser.add_argument(
        "--disable-rate-prefilter-before-local-replay",
        action="store_true",
        help=(
            "Run local replay even when archive rate alone cannot beat the target. "
            "Default keeps the queue mathematically fail-closed and avoids wasting CPU."
        ),
    )
    parser.add_argument(
        "--disable-hprc-rate-collapse",
        action="store_true",
        help="Skip the lossless HPRC section entropy transcode before replay/rate gates.",
    )
    parser.add_argument(
        "--hprc-rate-collapse-sections",
        action="append",
        default=[],
        help=(
            "Comma/space separated HPRC sections to entropy-wrap before replay. "
            "Default: decoder_qw,latents_rc,selectors_rc,residual_rc,receiver_state."
        ),
    )
    parser.add_argument("--hprc-rate-collapse-brotli-quality", type=int, default=11)
    parser.add_argument(
        "--hprc-rate-collapse-residual-collapse-schedule",
        action="append",
        default=[],
        help=(
            "Repeat or comma-separate residual-token collapse specs passed through "
            "to the HPRC transcode materializer, e.g. dz0_qd10."
        ),
    )
    parser.add_argument(
        "--hprc-rate-collapse-residual-importance-npy",
        type=Path,
        help="Optional residual-token importance .npy consumed by the rate-collapse materializer.",
    )
    parser.add_argument(
        "--hprc-rate-collapse-p19-posenet-null-pairs",
        type=Path,
        help="Optional P19 PoseNet-null artifact consumed by the rate-collapse materializer.",
    )
    parser.add_argument(
        "--hprc-rate-collapse-p18-segnet-region-waterfill",
        type=Path,
        help="Optional P18 SegNet-region artifact consumed by the rate-collapse materializer.",
    )
    parser.add_argument("--hprc-rate-collapse-importance-coarsen-quantile", type=float)
    parser.add_argument(
        "--hprc-rate-collapse-importance-selection-domain",
        choices=("global_weighted", "eligible_low"),
    )
    parser.add_argument("--hprc-rate-collapse-importance-protected-spec")
    parser.add_argument(
        "--hprc-rate-collapse-waterfill-low-spec",
        default="dz0_qd10",
        help=(
            "Default low-importance residual collapse used when P18/P19 artifacts "
            "are provided but no explicit residual-collapse schedule is set."
        ),
    )
    parser.add_argument(
        "--hprc-rate-collapse-waterfill-high-spec",
        default="dz0_qd1",
        help=(
            "Default high-importance residual collapse for P18/P19 waterfill; "
            "keeps boundaries/pose-sensitive cells near-lossless."
        ),
    )
    parser.add_argument(
        "--hprc-rate-collapse-waterfill-coarsen-quantile",
        type=float,
        default=0.25,
        help=(
            "Default fraction of explicit low-importance P19-null cells to coarsen "
            "when P18/P19 artifacts are present."
        ),
    )
    parser.add_argument(
        "--hprc-rate-collapse-distortion-reserve",
        type=float,
        default=0.04,
        help=(
            "Require this much score headroom after rate before local replay. "
            "This keeps HPRC from replaying candidates whose rate term barely "
            "clears the frontier but leaves no plausible SegNet/PoseNet budget."
        ),
    )
    parser.add_argument(
        "--strict-exact-auth-gate-returncode",
        action="store_true",
        help="Let a blocked exact-auth gate fail the queue instead of writing a durable blocked report.",
    )
    parser.add_argument(
        "--z8-archive-bin",
        type=Path,
        help="Optional Z8 0.bin to bind full-video P18/P19 allocator follow-ups.",
    )
    parser.add_argument("--z8-surface", type=Path)
    parser.add_argument("--z8-reference-pairs-npy", type=Path)
    parser.add_argument("--z8-pair-chunk-size", type=int, default=64)
    parser.add_argument("--z8-joint-weight-quantile", type=float, default=0.35)
    parser.add_argument("--z8-coefficient-deadzone-quantile", type=float, default=0.50)
    parser.add_argument("--z8-quantization-step", type=float, default=1.0 / 255.0)
    parser.add_argument("--z8-max-pairs", type=int)
    parser.add_argument("--z8-emit-receiver-proof", action="store_true")
    parser.add_argument(
        "--disable-z8-entropy-code-quantized-details",
        action="store_true",
        help="Do not enable the v2 detail entropy codec on Z8 materializer follow-ups.",
    )
    parser.add_argument("--skip-runtime-consumption-proof", action="store_true")
    parser.add_argument("--retain-receiver-output", action="store_true")
    parser.add_argument("--storage-tier", action="append", default=[])
    parser.add_argument(
        "--storage-workload-subdir",
        default=DEFAULT_HPRC_TRAINING_QUEUE_WORKLOAD_SUBDIR,
    )
    parser.add_argument("--storage-reserve-free-gb", type=float, default=DEFAULT_RESERVE_FREE_GB)
    parser.add_argument(
        "--storage-expected-bytes",
        type=int,
        default=HPRC_RECEIVER_PROOF_SCRATCH_BYTES,
    )
    parser.add_argument("--allow-local-output-dir", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--expected-plan-output-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve(strict=False)
    run_id = args.run_id or _utc_run_id()
    run_root, storage_plan_path = _select_output_dir(args, repo_root=repo_root, run_id=run_id)
    campaign_pairs = _campaign_pairs(args)
    plan_output = (
        run_root / "hprc_compact_receiver_training_plan.json"
        if args.plan_output is None
        else _resolve_path(args.plan_output, repo_root=repo_root)
    )

    plans: list[dict[str, Any]] = []
    native_surface_steps: list[dict[str, Any]] = []
    for pairs in campaign_pairs:
        campaign_output_dir = (
            run_root
            if len(campaign_pairs) == 1
            else run_root / f"pairs_{int(pairs):04d}"
        )
        campaign_output_dir.mkdir(parents=True, exist_ok=True)
        output_manifest = (
            campaign_output_dir / "hprc_compact_receiver_training_run_result.json"
        )
        native_surface_step = _native_rate_surface_step_config(
            args=args,
            repo_root=repo_root,
            output_dir=campaign_output_dir,
            decode_pairs=int(pairs),
        )
        if native_surface_step is not None:
            native_surface_steps.append(native_surface_step)
        plans.append(
            build_hprc_compact_receiver_training_plan(
                repo_root=repo_root,
                run_id=f"{run_id}_pairs{int(pairs):04d}",
                output_dir=campaign_output_dir,
                output_manifest=output_manifest,
                storage_plan_path=storage_plan_path,
                video_path=_resolve_path(args.video_path, repo_root=repo_root),
                decode_pairs=int(pairs),
                decode_max_pairs=args.decode_max_pairs,
                decode_height=int(args.decode_height),
                decode_width=int(args.decode_width),
                epochs=int(args.epochs),
                batch_pair_indices_per_step=int(args.batch_pair_indices_per_step),
                learning_rate=float(args.learning_rate),
                basis_count=int(args.basis_count),
                residual_grid_h=int(args.residual_grid_h),
                residual_grid_w=int(args.residual_grid_w),
                training_backend=str(args.training_backend),
                native_rate_aware=bool(args.enable_native_rate_aware_hprc),
                native_rate_residual_l1_weight=float(args.native_rate_residual_l1_weight),
                native_rate_residual_prox_weight=float(args.native_rate_residual_prox_weight),
                native_rate_residual_protection_npy=_native_rate_protection_npy_for_plan(
                    args=args,
                    repo_root=repo_root,
                    output_dir=campaign_output_dir,
                ),
                skip_runtime_consumption_proof=(
                    bool(args.skip_runtime_consumption_proof)
                    or not bool(args.disable_hprc_rate_collapse)
                ),
                retain_receiver_output=bool(args.retain_receiver_output),
            )
        )
    queue = build_local_training_execution_queue(
        plans,
        queue_id=args.queue_id or f"hprc_compact_receiver_campaign_{run_id}",
        repo_root=repo_root,
        lane_id="lane_hprc_compact_receiver_training",
        local_cpu_concurrency=int(args.local_cpu_concurrency),
        local_mlx_concurrency=int(args.local_mlx_concurrency),
        timeout_seconds=int(args.timeout_seconds),
    )
    _append_campaign_followup_steps(
        queue,
        plans=plans,
        native_surface_steps=native_surface_steps,
        args=args,
        repo_root=repo_root,
        full_replay_min_pairs=int(args.full_replay_min_pairs),
        timeout_seconds=int(args.timeout_seconds),
    )
    queue = normalize_queue_definition(queue)
    if len(plans) == 1 and not native_surface_steps:
        plan_artifact: dict[str, Any] = plans[0]
    else:
        plan_artifact = {
            "schema": HPRC_TRAINING_PLAN_SUITE_SCHEMA,
            "run_id": run_id,
            "campaign_pairs": campaign_pairs,
            "plan_count": len(plans),
            "plans": plans,
            "native_rate_surface_steps": native_surface_steps,
            **FALSE_AUTHORITY,
        }
    _write_json(
        plan_output,
        plan_artifact,
        allow_overwrite=bool(args.allow_overwrite) or args.expected_plan_output_sha256 is not None,
        expected_existing_sha256=args.expected_plan_output_sha256,
    )
    queue_output = _resolve_path(args.output, repo_root=repo_root)
    _write_json(
        queue_output,
        queue,
        allow_overwrite=bool(args.allow_overwrite) or args.expected_output_sha256 is not None,
        expected_existing_sha256=args.expected_output_sha256,
    )
    print(
        json.dumps(
            {
                "schema": HPRC_TRAINING_QUEUE_BUILD_SCHEMA,
                "queue_path": queue_output.as_posix(),
                "plan_path": plan_output.as_posix(),
                "queue_id": queue["queue_id"],
                "run_id": run_id,
                "campaign_pairs": campaign_pairs,
                "output_dir": run_root.as_posix(),
                "storage_plan_path": storage_plan_path.as_posix(),
                "experiment_count": len(queue["experiments"]),
                "full_replay_experiment_count": sum(
                    1 for pairs in campaign_pairs if int(pairs) >= int(args.full_replay_min_pairs)
                ),
                **FALSE_AUTHORITY,
            },
            sort_keys=True,
        )
    )
    return 0


def build_hprc_compact_receiver_training_plan(
    *,
    repo_root: Path,
    run_id: str,
    output_dir: Path,
    output_manifest: Path,
    storage_plan_path: Path,
    video_path: Path,
    decode_pairs: int,
    decode_max_pairs: int | None,
    decode_height: int,
    decode_width: int,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float,
    basis_count: int,
    residual_grid_h: int,
    residual_grid_w: int,
    training_backend: str,
    native_rate_aware: bool,
    native_rate_residual_l1_weight: float,
    native_rate_residual_prox_weight: float,
    native_rate_residual_protection_npy: Path | None,
    skip_runtime_consumption_proof: bool,
    retain_receiver_output: bool,
) -> dict[str, Any]:
    video_sha = sha256_file(video_path)
    command = [
        ".venv/bin/python",
        "tools/run_hprc_compact_receiver_training.py",
        "--video-path",
        _repo_rel_or_abs(video_path, repo_root),
        "--decode-pairs",
        str(int(decode_pairs)),
        "--decode-height",
        str(int(decode_height)),
        "--decode-width",
        str(int(decode_width)),
        "--output-dir",
        _repo_rel_or_abs(output_dir, repo_root),
        "--output-manifest",
        _repo_rel_or_abs(output_manifest, repo_root),
        "--storage-plan-path",
        _repo_rel_or_abs(storage_plan_path, repo_root),
        "--epochs",
        str(int(epochs)),
        "--batch-pair-indices-per-step",
        str(int(batch_pair_indices_per_step)),
        "--learning-rate",
        repr(float(learning_rate)),
        "--basis-count",
        str(int(basis_count)),
        "--residual-grid-h",
        str(int(residual_grid_h)),
        "--residual-grid-w",
        str(int(residual_grid_w)),
        "--training-backend",
        str(training_backend),
    ]
    if native_rate_aware:
        command.append("--native-rate-aware")
        command.extend(
            [
                "--rate-aware-residual-l1-weight",
                repr(float(native_rate_residual_l1_weight)),
                "--rate-aware-residual-prox-weight",
                repr(float(native_rate_residual_prox_weight)),
            ]
        )
        if native_rate_residual_protection_npy is not None:
            command.extend(
                [
                    "--rate-aware-residual-protection-npy",
                    _repo_rel_or_abs(native_rate_residual_protection_npy, repo_root),
                ]
            )
    if decode_max_pairs is not None:
        command.extend(["--decode-max-pairs", str(int(decode_max_pairs))])
    if skip_runtime_consumption_proof:
        command.append("--skip-runtime-consumption-proof")
    if retain_receiver_output:
        command.append("--retain-receiver-output")
    extra_postconditions: list[dict[str, Any]] = [
        {
            "type": "path_exists",
            "path": (output_dir / "training_artifact.json").as_posix(),
        },
        {
            "type": "json_false_authority",
            "path": (output_dir / "training_artifact.json").as_posix(),
        },
        {
            "type": "path_exists",
            "path": (output_dir / "hprc_compact_receiver_training_export.json").as_posix(),
        },
        {
            "type": "json_false_authority",
            "path": (output_dir / "hprc_compact_receiver_training_export.json").as_posix(),
        },
        {
            "type": "json_equals",
            "path": output_manifest.as_posix(),
            "key": "schema",
            "equals": "hprc_compact_receiver_training_run_result.v1",
        },
        {
            "type": "json_equals",
            "path": output_manifest.as_posix(),
            "key": "source_manifest.source_kind",
            "equals": "contest_video_decode",
        },
    ]
    if not skip_runtime_consumption_proof:
        receiver_proof = (
            output_dir
            / "hprc_compact_receiver_archive_export"
            / "receiver_proof"
            / "hprc_receiver_proof.json"
        )
        adapter_package = (
            output_dir
            / "hprc_compact_receiver_archive_export"
            / "archive_bound_candidate_adapter_package.json"
        )
        extra_postconditions.extend(
            [
                {"type": "path_exists", "path": receiver_proof.as_posix()},
                {
                    "type": "json_equals",
                    "path": receiver_proof.as_posix(),
                    "key": "schema",
                    "equals": "hprc_generated_receiver_proof.v1",
                },
                {
                    "type": "json_equals",
                    "path": receiver_proof.as_posix(),
                    "key": "score_claim",
                    "equals": False,
                },
                {
                    "type": "json_equals",
                    "path": receiver_proof.as_posix(),
                    "key": "promotion_eligible",
                    "equals": False,
                },
                {
                    "type": "json_equals",
                    "path": receiver_proof.as_posix(),
                    "key": "ready_for_exact_eval_dispatch",
                    "equals": False,
                },
                {"type": "path_exists", "path": adapter_package.as_posix()},
                {"type": "json_false_authority", "path": adapter_package.as_posix()},
            ]
        )
    return {
        "schema": HPRC_TRAINING_PLAN_SCHEMA,
        "candidate_id": f"hprc_compact_receiver_{run_id}",
        "lane_id": "lane_hprc_compact_receiver_training",
        "representation_family": "hprc",
        "substrate_family": "hierarchical_predictive_coding",
        "source_dir": _repo_rel_or_abs(video_path, repo_root),
        "training_signal_kind": "real_contest_video_lowres_frame_fit",
        "candidate_params": {
            "run_id": run_id,
            "source_video_sha256": video_sha,
            "decode_pairs": int(decode_pairs),
            "decode_max_pairs": None if decode_max_pairs is None else int(decode_max_pairs),
            "decode_height": int(decode_height),
            "decode_width": int(decode_width),
            "epochs": int(epochs),
            "basis_count": int(basis_count),
            "residual_grid_h": int(residual_grid_h),
            "residual_grid_w": int(residual_grid_w),
            "training_backend": str(training_backend),
            "portable_runtime": "numpy",
            "native_rate_aware": bool(native_rate_aware),
            "native_rate_residual_l1_weight": float(native_rate_residual_l1_weight),
            "native_rate_residual_prox_weight": float(native_rate_residual_prox_weight),
            "native_rate_residual_protection_npy": None
            if native_rate_residual_protection_npy is None
            else _repo_rel_or_abs(native_rate_residual_protection_npy, repo_root),
        },
        "recommended_execution": {
            "schema": "hprc_compact_receiver_training_recommended_execution.v1",
            "tool": "tools/run_hprc_compact_receiver_training.py",
            "training_backend": str(training_backend),
            "device": "auto" if str(training_backend) == "auto" else str(training_backend),
            "resource_kind": "local_mlx"
            if str(training_backend) in {"auto", "mlx"}
            else "local_cpu",
            "portable_runtime": "numpy",
            "contest_runtime_requires_mlx": False,
            "output_manifest": output_manifest.as_posix(),
            "python_command_args": command,
            "extra_artifact_postconditions": extra_postconditions,
            **FALSE_AUTHORITY,
        },
        **FALSE_AUTHORITY,
    }


def _append_campaign_followup_steps(
    queue: dict[str, Any],
    *,
    plans: list[dict[str, Any]],
    native_surface_steps: list[dict[str, Any]],
    args: argparse.Namespace,
    repo_root: Path,
    full_replay_min_pairs: int,
    timeout_seconds: int,
) -> None:
    surface_by_output = {
        str(step["training_output_dir"]): step
        for step in native_surface_steps
    }
    for plan, experiment in zip(plans, queue["experiments"], strict=True):
        params = plan["candidate_params"]
        output_manifest = Path(plan["recommended_execution"]["output_manifest"])
        output_dir = output_manifest.parent
        decode_pairs = int(params["decode_pairs"])
        export_dir = output_dir / "hprc_compact_receiver_archive_export"
        rate_collapse_dir = output_dir / "hprc_rate_collapse"
        rate_collapse_report = rate_collapse_dir / "hprc_rate_collapse_report.json"
        candidate_result = output_manifest
        candidate_export_dir = export_dir
        if not bool(args.disable_hprc_rate_collapse):
            candidate_result = rate_collapse_report
            candidate_export_dir = rate_collapse_dir / "best_archive_export"
        rate_gate = output_dir / "archive_rate_local_replay_gate.json"
        local_replay_summary = output_dir / "local_cpu_replay" / "local_submission_replay_summary.json"
        exact_gate = output_dir / "exact_auth_gate_cpu.json"
        followup_report = output_dir / "hprc_queue_followup_report.json"
        post_replay_report = output_dir / "hprc_queue_post_replay_report.json"
        native_surface_step = surface_by_output.get(output_dir.as_posix())

        if native_surface_step is not None:
            experiment["steps"].insert(
                0,
                _native_rate_surface_queue_step(
                    config=native_surface_step,
                    timeout_seconds=timeout_seconds,
                ),
            )
            train_requires = ["build_hprc_native_rate_residual_protection_surface"]
            for step in experiment["steps"]:
                if step.get("id") == "run_local_training":
                    step["requires"] = train_requires
                    step["telemetry"]["input_artifact_paths"].append(
                        str(native_surface_step["output_npy"])
                    )
                    break

        if not bool(args.disable_hprc_rate_collapse):
            experiment["steps"].append(
                _hprc_rate_collapse_step(
                    step_id="transcode_hprc_rate_collapse",
                    training_result=output_manifest,
                    output_dir=rate_collapse_dir,
                    report_path=rate_collapse_report,
                    args=args,
                    timeout_seconds=timeout_seconds,
                    skip_receiver_proof=True,
                    requires=["run_local_training"],
                )
            )

        if decode_pairs >= full_replay_min_pairs:
            followup_requires = [
                "run_local_training"
                if bool(args.disable_hprc_rate_collapse)
                else "transcode_hprc_rate_collapse"
            ]
            followup_local_summary = local_replay_summary
            followup_gate = exact_gate
        else:
            followup_requires = [
                "run_local_training"
                if bool(args.disable_hprc_rate_collapse)
                else "transcode_hprc_rate_collapse"
            ]
            followup_local_summary = None
            followup_gate = None

        experiment["steps"].append(
            _hprc_followup_report_step(
                step_id="write_hprc_campaign_followup_report",
                training_result=candidate_result,
                report_path=followup_report,
                decode_pairs=decode_pairs,
                full_replay_min_pairs=full_replay_min_pairs,
                local_replay_summary=followup_local_summary,
                exact_gate=followup_gate,
                z8_archive_bin=args.z8_archive_bin,
                z8_surface=args.z8_surface,
                z8_reference_pairs_npy=args.z8_reference_pairs_npy,
                repo_root=repo_root,
                requires=followup_requires,
                timeout_seconds=timeout_seconds,
            )
        )

        if decode_pairs >= full_replay_min_pairs:
            if not bool(args.disable_rate_prefilter_before_local_replay):
                experiment["steps"].append(
                    _archive_rate_gate_step(
                        training_result=candidate_result,
                        gate_json=rate_gate,
                        args=args,
                        timeout_seconds=timeout_seconds,
                        requires=["write_hprc_campaign_followup_report"],
                    )
                )
                proof_requires = ["gate_archive_rate_before_local_replay"]
            else:
                proof_requires = ["write_hprc_campaign_followup_report"]
            if not bool(args.disable_hprc_rate_collapse):
                experiment["steps"].append(
                    _hprc_rate_collapse_step(
                        step_id="prove_hprc_rate_collapsed_receiver",
                        training_result=output_manifest,
                        output_dir=rate_collapse_dir,
                        report_path=rate_collapse_report,
                        args=args,
                        timeout_seconds=timeout_seconds,
                        skip_receiver_proof=False,
                        requires=proof_requires,
                    )
                )
                replay_requires = ["prove_hprc_rate_collapsed_receiver"]
            else:
                replay_requires = proof_requires
            experiment["steps"].append(
                _local_replay_step(
                    output_dir=output_dir,
                    export_dir=candidate_export_dir,
                    summary_json=local_replay_summary,
                    device=str(args.local_replay_device),
                    timeout_seconds=timeout_seconds,
                    requires=replay_requires,
                )
            )
            experiment["steps"].append(
                _exact_auth_gate_step(
                    replay_summary_json=local_replay_summary,
                    gate_json=exact_gate,
                    args=args,
                    timeout_seconds=timeout_seconds,
                )
            )
            experiment["steps"].append(
                _hprc_followup_report_step(
                    step_id="write_hprc_campaign_post_replay_report",
                    training_result=candidate_result,
                    report_path=post_replay_report,
                    decode_pairs=decode_pairs,
                    full_replay_min_pairs=full_replay_min_pairs,
                    local_replay_summary=local_replay_summary,
                    exact_gate=exact_gate,
                    z8_archive_bin=args.z8_archive_bin,
                    z8_surface=args.z8_surface,
                    z8_reference_pairs_npy=args.z8_reference_pairs_npy,
                    repo_root=repo_root,
                    requires=["gate_exact_cpu_after_local_replay"],
                    timeout_seconds=timeout_seconds,
                )
            )
        if args.z8_archive_bin is not None:
            experiment["steps"].append(
                _z8_allocator_plan_step(
                    archive_bin=_resolve_path(args.z8_archive_bin, repo_root=repo_root),
                    output_dir=output_dir / "z8_full_video_p18_p19_allocator_plan",
                    pair_chunk_size=int(args.z8_pair_chunk_size),
                    timeout_seconds=timeout_seconds,
                    requires=["write_hprc_campaign_followup_report"],
                )
            )
        if args.z8_archive_bin is not None and args.z8_surface is not None:
            experiment["steps"].append(
                _z8_materializer_step(
                    archive_bin=_resolve_path(args.z8_archive_bin, repo_root=repo_root),
                    surface=_resolve_path(args.z8_surface, repo_root=repo_root),
                    output_dir=output_dir / "z8_joint_p18_p19_materialized_candidate",
                    args=args,
                    timeout_seconds=timeout_seconds,
                    requires=["build_z8_full_video_p18_p19_allocator_plan"],
                )
            )


def _native_rate_protection_npy_for_plan(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    output_dir: Path,
) -> Path | None:
    if not bool(args.enable_native_rate_aware_hprc):
        return None
    if args.native_rate_residual_protection_npy is not None:
        return _resolve_path(args.native_rate_residual_protection_npy, repo_root=repo_root)
    if args.native_rate_p19_posenet_null_pairs is not None:
        return output_dir / "hprc_native_rate_surface" / "residual_protection.npy"
    return None


def _native_rate_surface_step_config(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    output_dir: Path,
    decode_pairs: int,
) -> dict[str, Any] | None:
    if not bool(args.enable_native_rate_aware_hprc):
        return None
    if args.native_rate_residual_protection_npy is not None:
        return None
    if args.native_rate_p19_posenet_null_pairs is None:
        return None
    effective_pairs = int(decode_pairs)
    if args.decode_max_pairs is not None:
        effective_pairs = min(effective_pairs, int(args.decode_max_pairs))
    surface_dir = output_dir / "hprc_native_rate_surface"
    output_npy = surface_dir / "residual_protection.npy"
    report_path = surface_dir / "hprc_native_rate_residual_protection_surface.json"
    command = [
        ".venv/bin/python",
        "tools/build_hprc_native_rate_surface.py",
        "--p19-posenet-null-pairs",
        _repo_rel_or_abs(
            _resolve_path(args.native_rate_p19_posenet_null_pairs, repo_root=repo_root),
            repo_root,
        ),
        "--frames",
        str(effective_pairs * 2),
        "--residual-grid-h",
        str(int(args.residual_grid_h)),
        "--residual-grid-w",
        str(int(args.residual_grid_w)),
        "--gop-size",
        "2",
        "--default-protection",
        repr(float(args.native_rate_default_protection)),
        "--p19-null-protection",
        repr(float(args.native_rate_p19_null_protection)),
        "--p18-region-protection",
        repr(float(args.native_rate_p18_region_protection)),
        "--output-npy",
        _repo_rel_or_abs(output_npy, repo_root),
        "--out-json",
        _repo_rel_or_abs(report_path, repo_root),
        "--repo-root",
        repo_root.as_posix(),
        "--allow-overwrite",
    ]
    if args.native_rate_p18_segnet_region_waterfill is not None:
        command.extend(
            [
                "--p18-segnet-region-waterfill",
                _repo_rel_or_abs(
                    _resolve_path(args.native_rate_p18_segnet_region_waterfill, repo_root=repo_root),
                    repo_root,
                ),
            ]
        )
    return {
        "schema": "hprc_native_rate_surface_step_config.v1",
        "training_output_dir": output_dir.as_posix(),
        "output_npy": output_npy.as_posix(),
        "report_path": report_path.as_posix(),
        "decode_pairs": int(decode_pairs),
        "effective_pairs": int(effective_pairs),
        "command": command,
        **FALSE_AUTHORITY,
    }


def _native_rate_surface_queue_step(
    *,
    config: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    output_npy = Path(str(config["output_npy"]))
    report_path = Path(str(config["report_path"]))
    return {
        "id": "build_hprc_native_rate_residual_protection_surface",
        "kind": "command",
        "command": list(config["command"]),
        "resources": {"kind": "local_cpu"},
        "timeout_seconds": timeout_seconds,
        "postconditions": [
            {"type": "path_exists", "path": output_npy.as_posix()},
            {"type": "path_exists", "path": report_path.as_posix()},
            {"type": "json_false_authority", "path": report_path.as_posix()},
            {
                "type": "json_equals",
                "path": report_path.as_posix(),
                "key": "schema",
                "equals": "hprc_native_rate_residual_protection_surface.v1",
            },
        ],
        "telemetry": {
            "artifact_paths": [output_npy.as_posix(), report_path.as_posix()],
            "input_artifact_paths": [
                part
                for index, part in enumerate(config["command"])
                if index > 0 and config["command"][index - 1] in {
                    "--p19-posenet-null-pairs",
                    "--p18-segnet-region-waterfill",
                }
            ],
        },
    }


def _local_replay_step(
    *,
    output_dir: Path,
    export_dir: Path,
    summary_json: Path,
    device: str,
    timeout_seconds: int,
    requires: list[str],
) -> dict[str, Any]:
    replay_dir = output_dir / "local_cpu_replay"
    return {
        "id": "run_local_cpu_replay",
        "kind": "command",
        "requires": requires,
        "command": [
            ".venv/bin/python",
            "tools/run_local_submission_replay.py",
            "--runtime-submission-dir",
            (export_dir / "submission").as_posix(),
            "--archive-zip",
            (export_dir / "archive.zip").as_posix(),
            "--output-dir",
            replay_dir.as_posix(),
            "--summary-json",
            summary_json.as_posix(),
            "--device",
            device,
            "--force",
        ],
        "resources": {"kind": "local_cpu"},
        "timeout_seconds": timeout_seconds,
        "postconditions": [
            {"type": "path_exists", "path": summary_json.as_posix()},
            {"type": "json_false_authority", "path": summary_json.as_posix()},
            {
                "type": "json_equals",
                "path": summary_json.as_posix(),
                "key": "schema",
                "equals": "local_submission_replay.v1",
            },
        ],
        "telemetry": {
            "artifact_paths": [summary_json.as_posix()],
            "input_artifact_paths": [(export_dir / "archive.zip").as_posix()],
            "recursive": True,
            "max_recursive_entries": 256,
        },
    }


def _archive_rate_gate_step(
    *,
    training_result: Path,
    gate_json: Path,
    args: argparse.Namespace,
    timeout_seconds: int,
    requires: list[str],
) -> dict[str, Any]:
    command = [
        ".venv/bin/python",
        "tools/gate_archive_rate_for_local_replay.py",
        "--training-result",
        training_result.as_posix(),
        "--min-local-improvement",
        repr(float(_rate_gate_margin(args))),
        "--out-json",
        gate_json.as_posix(),
        "--allow-overwrite",
        "--success-on-blocked",
    ]
    if args.auth_frontier_score is not None:
        command.extend(["--auth-frontier-score", repr(float(args.auth_frontier_score))])
    if args.local_baseline_score is not None:
        command.extend(["--local-baseline-score", repr(float(args.local_baseline_score))])
    return {
        "id": "gate_archive_rate_before_local_replay",
        "kind": "command",
        "requires": requires,
        "command": command,
        "resources": {"kind": "local_cpu"},
        "timeout_seconds": timeout_seconds,
        "on_postcondition_failure": "skipped",
        "postconditions": [
            {"type": "path_exists", "path": gate_json.as_posix()},
            {"type": "json_false_authority", "path": gate_json.as_posix()},
            {
                "type": "json_equals",
                "path": gate_json.as_posix(),
                "key": "schema",
                "equals": "archive_rate_local_replay_gate.v1",
            },
            {
                "type": "json_equals",
                "path": gate_json.as_posix(),
                "key": "local_replay_recommended",
                "equals": True,
            },
        ],
        "telemetry": {
            "artifact_paths": [gate_json.as_posix()],
            "input_artifact_paths": [training_result.as_posix()],
        },
    }


def _hprc_rate_collapse_step(
    *,
    step_id: str,
    training_result: Path,
    output_dir: Path,
    report_path: Path,
    args: argparse.Namespace,
    timeout_seconds: int,
    skip_receiver_proof: bool,
    requires: list[str],
) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve(strict=False)
    command = [
        ".venv/bin/python",
        "tools/transcode_hprc_compact_receiver_rate_collapse.py",
        "--training-result",
        training_result.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--repo-root",
        repo_root.as_posix(),
        "--brotli-quality",
        str(int(args.hprc_rate_collapse_brotli_quality)),
        "--out-json",
        report_path.as_posix(),
        "--allow-overwrite",
    ]
    for raw in args.hprc_rate_collapse_sections or []:
        command.extend(["--sections", str(raw)])
    for raw in args.hprc_rate_collapse_residual_collapse_schedule or []:
        command.extend(["--residual-collapse-schedule", str(raw)])
    if (
        _rate_collapse_requests_structured_waterfill(args)
        and not args.hprc_rate_collapse_residual_collapse_schedule
    ):
        command.extend(
            [
                "--residual-collapse-schedule",
                str(args.hprc_rate_collapse_waterfill_low_spec),
            ]
        )
    if _rate_collapse_requests_lossy_residual(args):
        command.append("--enable-lossy-residual-collapse")
    if args.hprc_rate_collapse_residual_importance_npy is not None:
        importance_npy = _resolve_path(
            args.hprc_rate_collapse_residual_importance_npy,
            repo_root=repo_root,
        )
        command.extend(
            [
                "--residual-importance-npy",
                _repo_rel_or_abs(importance_npy, repo_root),
            ]
        )
    p19_posenet_null_pairs = _rate_collapse_p19_posenet_null_pairs(args)
    if p19_posenet_null_pairs is not None:
        p19_path = _resolve_path(
            p19_posenet_null_pairs,
            repo_root=repo_root,
        )
        command.extend(
            [
                "--p19-posenet-null-pairs",
                _repo_rel_or_abs(p19_path, repo_root),
            ]
        )
    p18_segnet_region_waterfill = _rate_collapse_p18_segnet_region_waterfill(args)
    if p18_segnet_region_waterfill is not None:
        p18_path = _resolve_path(
            p18_segnet_region_waterfill,
            repo_root=repo_root,
        )
        command.extend(
            [
                "--p18-segnet-region-waterfill",
                _repo_rel_or_abs(p18_path, repo_root),
            ]
        )
    coarsen_quantile = args.hprc_rate_collapse_importance_coarsen_quantile
    if coarsen_quantile is None and _rate_collapse_requests_structured_waterfill(args):
        coarsen_quantile = float(args.hprc_rate_collapse_waterfill_coarsen_quantile)
    if coarsen_quantile is not None:
        command.extend(
            [
                "--importance-coarsen-quantile",
                repr(float(coarsen_quantile)),
            ]
        )
    selection_domain = args.hprc_rate_collapse_importance_selection_domain
    if selection_domain is None and _rate_collapse_requests_structured_waterfill(args):
        selection_domain = "eligible_low"
    if selection_domain is not None:
        command.extend(
            [
                "--importance-selection-domain",
                str(selection_domain),
            ]
        )
    protected_spec = args.hprc_rate_collapse_importance_protected_spec
    if protected_spec is None and _rate_collapse_requests_structured_waterfill(args):
        protected_spec = str(args.hprc_rate_collapse_waterfill_high_spec)
    if protected_spec is not None:
        command.extend(
            [
                "--importance-protected-spec",
                str(protected_spec),
            ]
        )
    if skip_receiver_proof:
        command.append("--skip-receiver-proof")
    target_rate_term = _rate_collapse_target_rate_term(args)
    if target_rate_term is not None:
        command.extend(["--target-rate-term", repr(float(target_rate_term))])
    if bool(args.retain_receiver_output):
        command.append("--retain-receiver-output")
    postconditions: list[dict[str, Any]] = [
        {"type": "path_exists", "path": report_path.as_posix()},
        {"type": "json_false_authority", "path": report_path.as_posix()},
        {
            "type": "json_equals",
            "path": report_path.as_posix(),
            "key": "schema",
            "equals": "hprc_rate_collapse_report.v1",
        },
    ]
    if not skip_receiver_proof:
        receiver_proof = output_dir / "best_archive_export" / "receiver_proof" / "hprc_receiver_proof.json"
        postconditions.extend(
            [
                {"type": "path_exists", "path": receiver_proof.as_posix()},
                {"type": "json_false_authority", "path": receiver_proof.as_posix()},
                {
                    "type": "json_equals",
                    "path": receiver_proof.as_posix(),
                    "key": "schema",
                    "equals": "hprc_generated_receiver_proof.v1",
                },
                {
                    "type": "json_equals",
                    "path": receiver_proof.as_posix(),
                    "key": "receiver_contract_satisfied",
                    "equals": True,
                },
                {
                    "type": "json_equals",
                    "path": receiver_proof.as_posix(),
                    "key": "runtime_consumption_proof_ready",
                    "equals": True,
                },
                {
                    "type": "json_equals",
                    "path": receiver_proof.as_posix(),
                    "key": "blockers",
                    "equals": [],
                },
                {
                    "type": "json_equals",
                    "path": report_path.as_posix(),
                    "key": "artifact.receiver_proof_present",
                    "equals": True,
                },
            ]
        )
    return {
        "id": step_id,
        "kind": "command",
        "requires": requires,
        "command": command,
        "resources": {"kind": "local_cpu"},
        "timeout_seconds": timeout_seconds,
        "postconditions": postconditions,
        "telemetry": {
            "artifact_paths": [output_dir.as_posix(), report_path.as_posix()],
            "input_artifact_paths": [training_result.as_posix()],
            "recursive": True,
            "max_recursive_entries": 256,
        },
    }


def _rate_collapse_requests_lossy_residual(args: argparse.Namespace) -> bool:
    return bool(
        args.hprc_rate_collapse_residual_collapse_schedule
        or args.hprc_rate_collapse_residual_importance_npy is not None
        or _rate_collapse_p19_posenet_null_pairs(args) is not None
        or _rate_collapse_p18_segnet_region_waterfill(args) is not None
    )


def _rate_collapse_requests_structured_waterfill(args: argparse.Namespace) -> bool:
    return bool(
        _rate_collapse_p19_posenet_null_pairs(args) is not None
        or _rate_collapse_p18_segnet_region_waterfill(args) is not None
        or args.hprc_rate_collapse_residual_importance_npy is not None
    )


def _rate_collapse_p19_posenet_null_pairs(args: argparse.Namespace) -> Path | None:
    if args.hprc_rate_collapse_p19_posenet_null_pairs is not None:
        return args.hprc_rate_collapse_p19_posenet_null_pairs
    if args.hprc_rate_collapse_residual_importance_npy is not None:
        return None
    return args.native_rate_p19_posenet_null_pairs


def _rate_collapse_p18_segnet_region_waterfill(args: argparse.Namespace) -> Path | None:
    if args.hprc_rate_collapse_p18_segnet_region_waterfill is not None:
        return args.hprc_rate_collapse_p18_segnet_region_waterfill
    if args.hprc_rate_collapse_residual_importance_npy is not None:
        return None
    return args.native_rate_p18_segnet_region_waterfill


def _rate_gate_margin(args: argparse.Namespace) -> float:
    return max(
        float(args.min_local_improvement),
        float(args.hprc_rate_collapse_distortion_reserve),
        0.0,
    )


def _rate_collapse_target_rate_term(args: argparse.Namespace) -> float | None:
    targets = [
        float(value)
        for value in (args.auth_frontier_score, args.local_baseline_score)
        if value is not None
    ]
    if not targets:
        return None
    return min(targets) - _rate_gate_margin(args)


def _exact_auth_gate_step(
    *,
    replay_summary_json: Path,
    gate_json: Path,
    args: argparse.Namespace,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [
        ".venv/bin/python",
        "tools/gate_local_candidate_for_exact_auth.py",
        "--local-replay-summary-json",
        replay_summary_json.as_posix(),
        "--exact-auth-axis",
        str(args.exact_auth_axis),
        "--min-local-improvement",
        repr(float(args.min_local_improvement)),
        "--out-json",
        gate_json.as_posix(),
    ]
    if args.auth_frontier_score is not None:
        command.extend(["--auth-frontier-score", repr(float(args.auth_frontier_score))])
    if args.local_baseline_score is not None:
        command.extend(["--local-baseline-score", repr(float(args.local_baseline_score))])
    if not bool(args.strict_exact_auth_gate_returncode):
        command.append("--success-on-blocked")
    return {
        "id": "gate_exact_cpu_after_local_replay",
        "kind": "command",
        "requires": ["run_local_cpu_replay"],
        "command": command,
        "resources": {"kind": "local_cpu"},
        "timeout_seconds": timeout_seconds,
        "postconditions": [
            {"type": "path_exists", "path": gate_json.as_posix()},
            {"type": "json_false_authority", "path": gate_json.as_posix()},
            {
                "type": "json_equals",
                "path": gate_json.as_posix(),
                "key": "schema",
                "equals": "local_candidate_exact_auth_gate.v1",
            },
        ],
        "telemetry": {
            "artifact_paths": [gate_json.as_posix()],
            "input_artifact_paths": [replay_summary_json.as_posix()],
        },
    }


def _hprc_followup_report_step(
    *,
    step_id: str,
    training_result: Path,
    report_path: Path,
    decode_pairs: int,
    full_replay_min_pairs: int,
    local_replay_summary: Path | None,
    exact_gate: Path | None,
    z8_archive_bin: Path | None,
    z8_surface: Path | None,
    z8_reference_pairs_npy: Path | None,
    repo_root: Path,
    requires: list[str],
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [
        ".venv/bin/python",
        "tools/write_hprc_queue_followup_report.py",
        "--training-result",
        training_result.as_posix(),
        "--decode-pairs",
        str(int(decode_pairs)),
        "--full-replay-min-pairs",
        str(int(full_replay_min_pairs)),
        "--repo-root",
        repo_root.as_posix(),
        "--out-json",
        report_path.as_posix(),
        "--allow-overwrite",
    ]
    if local_replay_summary is not None:
        command.extend(["--local-replay-summary-json", local_replay_summary.as_posix()])
    if exact_gate is not None:
        command.extend(["--exact-auth-gate-json", exact_gate.as_posix()])
    if z8_archive_bin is not None:
        command.extend(["--z8-archive-bin", z8_archive_bin.as_posix()])
    if z8_surface is not None:
        command.extend(["--z8-surface", z8_surface.as_posix()])
    if z8_reference_pairs_npy is not None:
        command.extend(["--z8-reference-pairs-npy", z8_reference_pairs_npy.as_posix()])
    return {
        "id": step_id,
        "kind": "command",
        "requires": requires,
        "command": command,
        "resources": {"kind": "local_cpu"},
        "timeout_seconds": timeout_seconds,
        "postconditions": [
            {"type": "path_exists", "path": report_path.as_posix()},
            {"type": "json_false_authority", "path": report_path.as_posix()},
            {
                "type": "json_equals",
                "path": report_path.as_posix(),
                "key": "schema",
                "equals": "hprc_queue_followup_report.v1",
            },
        ],
        "telemetry": {
            "artifact_paths": [report_path.as_posix()],
            "input_artifact_paths": [training_result.as_posix()],
        },
    }


def _z8_allocator_plan_step(
    *,
    archive_bin: Path,
    output_dir: Path,
    pair_chunk_size: int,
    timeout_seconds: int,
    requires: list[str],
) -> dict[str, Any]:
    plan_path = output_dir / "z8_full_video_vjp_acquisition_plan.json"
    return {
        "id": "build_z8_full_video_p18_p19_allocator_plan",
        "kind": "command",
        "requires": requires,
        "command": [
            ".venv/bin/python",
            "tools/build_z8_full_video_vjp_surface_bundle.py",
            "--archive-bin",
            archive_bin.as_posix(),
            "--output-dir",
            output_dir.as_posix(),
            "--pair-chunk-size",
            str(int(pair_chunk_size)),
            "--overwrite",
        ],
        "resources": {"kind": "local_mlx"},
        "timeout_seconds": timeout_seconds,
        "postconditions": [
            {"type": "path_exists", "path": plan_path.as_posix()},
            {"type": "json_false_authority", "path": plan_path.as_posix()},
            {
                "type": "json_equals",
                "path": plan_path.as_posix(),
                "key": "schema",
                "equals": "z8_full_video_vjp_acquisition_plan.v1",
            },
        ],
        "telemetry": {
            "artifact_paths": [output_dir.as_posix()],
            "input_artifact_paths": [archive_bin.as_posix()],
            "recursive": True,
            "max_recursive_entries": 128,
        },
    }


def _z8_materializer_step(
    *,
    archive_bin: Path,
    surface: Path,
    output_dir: Path,
    args: argparse.Namespace,
    timeout_seconds: int,
    requires: list[str],
) -> dict[str, Any]:
    manifest_path = output_dir / "z8_joint_p18_p19_deadzone_manifest.json"
    command = [
        ".venv/bin/python",
        "tools/materialize_z8_joint_p18_p19_deadzone_candidate.py",
        "--archive-bin",
        archive_bin.as_posix(),
        "--surface",
        surface.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--joint-weight-quantile",
        repr(float(args.z8_joint_weight_quantile)),
        "--coefficient-deadzone-quantile",
        repr(float(args.z8_coefficient_deadzone_quantile)),
        "--quantization-step",
        repr(float(args.z8_quantization_step)),
    ]
    if args.z8_max_pairs is not None:
        command.extend(["--max-pairs", str(int(args.z8_max_pairs))])
    if args.z8_emit_receiver_proof:
        command.append("--emit-receiver-proof")
    if not args.disable_z8_entropy_code_quantized_details:
        command.append("--entropy-code-quantized-details")
    return {
        "id": "materialize_z8_p18_p19_allocator_candidate",
        "kind": "command",
        "requires": requires,
        "command": command,
        "resources": {"kind": "local_mlx"},
        "timeout_seconds": timeout_seconds,
        "postconditions": [
            {"type": "path_exists", "path": manifest_path.as_posix()},
            {"type": "json_false_authority", "path": manifest_path.as_posix()},
            {
                "type": "json_equals",
                "path": manifest_path.as_posix(),
                "key": "schema",
                "equals": "z8_joint_p18_p19_coefficient_deadzone_candidate.v1",
            },
        ],
        "telemetry": {
            "artifact_paths": [output_dir.as_posix()],
            "input_artifact_paths": [archive_bin.as_posix(), surface.as_posix()],
            "recursive": True,
            "max_recursive_entries": 512,
        },
    }


def _campaign_pairs(args: argparse.Namespace) -> list[int]:
    raw = [int(value) for value in (args.campaign_pairs or [])]
    if not raw:
        raw = [int(args.decode_pairs)]
    out = sorted(dict.fromkeys(raw))
    if any(value < 1 for value in out):
        raise ValueError("campaign pair counts must be >= 1")
    return out


def _select_output_dir(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    run_id: str,
) -> tuple[Path, Path]:
    tiers = parse_storage_tier_specs(
        list(args.storage_tier),
        repo_root=repo_root,
        reserve_free_gb=float(args.storage_reserve_free_gb),
        allow_local_disk=bool(args.allow_local_output_dir),
    )
    plan = plan_experiment_storage(
        tiers,
        workload_subdir=str(args.storage_workload_subdir),
        requested_bytes=int(args.storage_expected_bytes),
        create=True,
    )
    workload_root = require_selected_storage(plan)
    output_dir = workload_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_plan_path = output_dir / "hprc_compact_receiver_training_storage_plan.json"
    expected_storage_plan_sha = (
        sha256_file(storage_plan_path)
        if storage_plan_path.is_file() and bool(args.allow_overwrite)
        else None
    )
    _write_json(
        storage_plan_path,
        {
            "schema": "hprc_compact_receiver_training_storage_plan.v1",
            "storage_plan": plan.to_dict(),
            "selected_training_output_dir": output_dir.as_posix(),
            **FALSE_AUTHORITY,
        },
        allow_overwrite=bool(args.allow_overwrite),
        expected_existing_sha256=expected_storage_plan_sha,
    )
    return output_dir, storage_plan_path


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    out = Path(path).expanduser()
    return out if out.is_absolute() else repo_root / out


def _repo_rel_or_abs(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(
    path: Path,
    payload: object,
    *,
    allow_overwrite: bool,
    expected_existing_sha256: str | None,
) -> None:
    if expected_existing_sha256 is None and allow_overwrite and path.is_file():
        expected_existing_sha256 = sha256_file(path)
    write_json_artifact(
        path,
        payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_existing_sha256,
    )


def _utc_run_id() -> str:
    return time.strftime("hprc_compact_receiver_%Y%m%dT%H%M%SZ", time.gmtime())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactWriteError, StorageTierError, ValueError) as exc:
        print(f"build_hprc_compact_receiver_training_queue failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
