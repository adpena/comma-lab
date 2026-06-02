#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""HiNeRV MLX-local score-aware trainer.

This is a real MLX harness binding for the current local HiNeRV archive family.
It is still false-authority: MLX/local training artifacts may guide iteration,
but contest CPU/CUDA replay is the only score/rank surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.operator_storage_waterfall import (
    operator_storage_policy_payload,
    operator_storage_tier_cli_specs,
)
from comma_lab.storage_tiers import (
    DEFAULT_RESERVE_FREE_GB,
    StorageTierError,
    parse_storage_tier_specs,
    plan_experiment_storage,
    require_selected_storage,
)
from tac.analysis.nerv_modelsize_ladder import (
    hi_nerv_modelsize_config_rows,
)
from tac.repo_io import write_json
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    FALSE_AUTHORITY,
)

TRAINER_SCHEMA = "hi_nerv_mlx_score_aware_trainer.v1"
TRAINER_AUTHORITY = "false_authority_macos_mlx_training_no_contest_score_claim"
DEFAULT_WORKLOAD_SUBDIR = "hinerv_mlx_local_training"
MODEL_SIZE_ROWS = tuple(
    row["row_id"] for row in hi_nerv_modelsize_config_rows(num_pairs=600)
)


def _full_main(args: argparse.Namespace) -> int:
    """Run canonical MLX score-aware training for the current HiNeRV carrier."""

    from tac.substrates._shared.mlx_score_aware import (
        RendererBundle,
        build_mlx_posenet_pair_teacher,
        build_mlx_segnet_pair_teacher,
        decode_mlx_targets,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates._shared.mlx_score_aware.coder_qat import (
        build_decoder_coder_qat_terms,
        coder_qat_loss_weights,
        coder_qat_metadata,
    )
    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX
    from tac.substrates.hinton_distilled_scorer_surrogate import (
        DEFAULT_POSE_DIMS,
        DEFAULT_SEGNET_CLASSES,
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )

    output_dir, storage_payload = _resolve_output_dir(args)
    cfg = _config_from_args(args)
    model = HinervSubstrateMLX(cfg)
    if args.decoder_fake_quant_forward:
        model.configure_decoder_fake_quant_forward(
            enabled=True,
            quant_bits=int(args.decoder_fake_quant_bits),
        )
    coder_qat_cfg = _coder_qat_config_from_args(args)
    extra_loss_terms = None
    if coder_qat_cfg.enabled:

        def _extra_loss_terms(model_obj: Any, _idx: Any) -> dict[str, Any]:
            return dict(build_decoder_coder_qat_terms(model_obj, coder_qat_cfg))

        extra_loss_terms = _extra_loss_terms

    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        args.video_path,
        num_pairs=int(cfg.num_pairs),
        output_height=int(cfg.output_height),
        output_width=int(cfg.output_width),
    )

    scorer_teacher = None
    pose_scorer_teacher = None
    learnable_student_head = None
    learnable_pose_student_head = None
    pose_distillation_weight = 0.0
    if float(args.distillation_weight) > 0.0 and not args.allow_mock_scorer_teacher:
        bundle_no_teacher = RendererBundle(
            model=model,
            target_rgb_0=target_rgb_0,
            target_rgb_1=target_rgb_1,
            num_pairs=int(cfg.num_pairs),
            forward_convention="call_b2chw_255",
            distillation_weight=0.0,
            pose_distillation_weight=0.0,
            pose_dims=DEFAULT_POSE_DIMS,
        )
        scorer_teacher = build_mlx_segnet_pair_teacher(
            bundle_no_teacher,
            upstream_dir=str(args.upstream_dir),
            device="cpu",
        )
        pose_scorer_teacher = build_mlx_posenet_pair_teacher(
            bundle_no_teacher,
            upstream_dir=str(args.upstream_dir),
            device="cpu",
        )
        learnable_student_head = build_learnable_student_head(
            num_classes=DEFAULT_SEGNET_CLASSES,
            in_channels=3,
            seed=int(args.seed),
        )
        learnable_pose_student_head = build_learnable_pose_student_head(
            pose_dims=DEFAULT_POSE_DIMS,
            seed=int(args.seed),
        )
        pose_distillation_weight = float(args.pose_distillation_weight)

    bundle = RendererBundle(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=int(cfg.num_pairs),
        forward_convention="call_b2chw_255",
        extra_loss_terms=extra_loss_terms,
        extra_loss_weights=coder_qat_loss_weights(coder_qat_cfg),
        distillation_weight=float(args.distillation_weight),
        scorer_teacher=scorer_teacher,
        learnable_student_head=learnable_student_head,
        pose_distillation_weight=pose_distillation_weight,
        pose_scorer_teacher=pose_scorer_teacher,
        learnable_pose_student_head=learnable_pose_student_head,
        pose_dims=DEFAULT_POSE_DIMS,
        allow_mock_scorer_teacher=bool(args.allow_mock_scorer_teacher),
        allow_segnet_only_research=bool(args.allow_segnet_only_research),
        export_archive_fn=lambda model_obj, out_dir: export_hi_nerv_mlx_archive(
            model_obj,
            out_dir,
            repo_root=REPO_ROOT,
            decoder_codec=str(args.decoder_codec),
            source_backend="mlx",
        ),
        substrate_artifact_metadata={
            "schema": TRAINER_SCHEMA,
            "authority": TRAINER_AUTHORITY,
            "family": "hi_nerv",
            "source_fidelity_status": "local_hi_nerv_fork_not_official_hinerv_parity",
            "modelsize_row": args.modelsize_row,
            "config": _config_snapshot(cfg),
            "decoder_codec": str(args.decoder_codec),
            "decoder_fake_quant_forward": {
                "enabled": bool(args.decoder_fake_quant_forward),
                "quant_bits": int(args.decoder_fake_quant_bits),
            },
            "coder_qat": coder_qat_metadata(coder_qat_cfg),
            "eval_roundtrip_ste_enabled": bool(args.eval_roundtrip_ste),
            "storage_preflight": _metadata_safe(storage_payload),
            "blockers": [
                "contest_cpu_cuda_exact_eval_not_executed",
                "official_hinerv_feature_grid_parity_not_proven",
            ],
        },
        eval_roundtrip_ste_enabled=bool(args.eval_roundtrip_ste),
        pose_student_input_preprocess=str(args.pose_student_input_preprocess),
    )
    write_json(
        output_dir / "hi_nerv_mlx_training_launch_preflight.json",
        {
            "schema": TRAINER_SCHEMA,
            "authority": TRAINER_AUTHORITY,
            "output_dir": output_dir.as_posix(),
            "storage_preflight": storage_payload,
            "command": sys.argv,
            **FALSE_AUTHORITY,
        },
    )
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="hi_nerv_mlx_local",
        lane_id="lane_hi_nerv_mlx_score_aware_local_20260602",
        output_dir=output_dir,
        epochs=int(args.epochs),
        batch_pair_indices_per_step=min(int(args.batch_pairs), int(cfg.num_pairs)),
        learning_rate=float(args.full_lr),
        seed=int(args.seed),
        checkpoint_interval_epochs=int(args.checkpoint_interval_epochs),
        pr95_faithful_curriculum_enabled=bool(args.pr95_faithful_curriculum),
        pr95_curriculum_total_epochs=args.pr95_curriculum_total_epochs,
        grad_clip_max_norm=args.grad_clip_max_norm,
        warmup_epochs=int(args.warmup_epochs),
        warmup_steps_per_epoch=1,
        weight_decay=args.weight_decay,
        optimizer_kind=str(args.optimizer_kind),
        cosine_decay_enabled=bool(args.cosine_decay),
        cosine_decay_total_epochs=args.cosine_decay_total_epochs,
        cosine_decay_min_lr_ratio=float(args.cosine_decay_min_lr_ratio),
        ema_archive_selection_enabled=bool(args.ema_archive_selection),
        notes=(
            "HiNeRV MLX-local score-aware training through the canonical "
            "mlx_score_aware harness, with optional real SegNet/PoseNet teacher "
            "binding, PR95 faithful curriculum, coder-aware QAT, eval-roundtrip "
            "STE, and archive export. False-authority until contest CPU/CUDA replay."
        ),
    )
    post_export_quality = _maybe_write_post_export_receiver_cache_quality(
        args=args,
        output_dir=output_dir,
        archive_path=getattr(artifact, "archive_path", None),
    )
    if post_export_quality is not None:
        _attach_post_export_receiver_cache_quality_to_training_artifact(
            output_dir=output_dir,
            report=post_export_quality,
        )
    print(
        json.dumps(
            {
                "schema": TRAINER_SCHEMA,
                "output_dir": output_dir.as_posix(),
                "epochs": artifact.total_epochs_completed,
                "archive_bytes": getattr(artifact, "archive_bytes", None),
                "post_export_receiver_cache_quality_report": (
                    post_export_quality.get("report_path")
                    if post_export_quality is not None
                    else None
                ),
                "post_export_receiver_cache_quality_passed": (
                    bool(post_export_quality.get("quality_gate_passed"))
                    if post_export_quality is not None
                    else False
                ),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _smoke_main(args: argparse.Namespace) -> int:
    """Run a small real MLX forward/export smoke for the HiNeRV binding."""

    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: MLX import failed: {exc!r}", file=sys.stderr)
        return 2
    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive
    from tac.substrates.hi_nerv.mlx_renderer import MLX_EVIDENCE_GRADE, HinervSubstrateMLX

    output_dir, storage_payload = _resolve_output_dir(args)
    cfg = _config_from_args(args)
    model = HinervSubstrateMLX(cfg)
    if args.decoder_fake_quant_forward:
        model.configure_decoder_fake_quant_forward(
            enabled=True,
            quant_bits=int(args.decoder_fake_quant_bits),
        )
    idx = mx.array(list(range(min(2, int(cfg.num_pairs)))), dtype=mx.int32)
    output = model(idx)
    mx.eval(output)
    archive_path = archive_sha256 = None
    archive_bytes = None
    if args.smoke_export_archive:
        archive_path_obj, archive_sha256, archive_bytes = export_hi_nerv_mlx_archive(
            model,
            output_dir / "smoke_archive_export",
            repo_root=REPO_ROOT,
            decoder_codec=str(args.decoder_codec),
            source_backend="mlx",
        )
        archive_path = archive_path_obj.as_posix()
    post_export_quality = _maybe_write_post_export_receiver_cache_quality(
        args=args,
        output_dir=output_dir,
        archive_path=Path(archive_path) if archive_path else None,
    )
    manifest = {
        "schema": "hi_nerv_mlx_trainer_smoke.v1",
        "authority": TRAINER_AUTHORITY,
        "axis_tag": MLX_EVIDENCE_GRADE,
        "family": "hi_nerv",
        "source_fidelity_status": "local_hi_nerv_fork_not_official_hinerv_parity",
        "output_dir": output_dir.as_posix(),
        "storage_preflight": storage_payload,
        "modelsize_row": args.modelsize_row,
        "config": _config_snapshot(cfg),
        "num_parameters": int(model.num_parameters()),
        "forward_convention": "call_b2chw_255",
        "forward_smoke": {
            "input_indices": [int(v) for v in idx.tolist()],
            "output_shape": [int(v) for v in output.shape],
            "output_min": float(mx.min(output)),
            "output_max": float(mx.max(output)),
            "output_mean": float(mx.mean(output)),
        },
        "decoder_codec": str(args.decoder_codec),
        "archive_path": archive_path,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "post_export_receiver_cache_quality": (
            _receiver_cache_quality_manifest_summary(post_export_quality)
            if post_export_quality is not None
            else None
        ),
        "blockers": [
            "contest_cpu_cuda_exact_eval_not_executed",
            "hi_nerv_smoke_no_training_score",
            "official_hinerv_feature_grid_parity_not_proven",
        ],
        **FALSE_AUTHORITY,
    }
    write_json(output_dir / "smoke_manifest.json", manifest)
    print(json.dumps({"smoke_manifest": (output_dir / "smoke_manifest.json").as_posix(), **FALSE_AUTHORITY}, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-local-output-dir", action="store_true")
    parser.add_argument("--storage-workload-subdir", default=DEFAULT_WORKLOAD_SUBDIR)
    parser.add_argument("--storage-expected-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--storage-reserve-free-gb", type=float, default=DEFAULT_RESERVE_FREE_GB)
    parser.add_argument("--num-pairs", type=int, default=600)
    parser.add_argument("--modelsize-row", choices=MODEL_SIZE_ROWS, default="hi_nerv_local_tiny")
    parser.add_argument("--latent-dim-coarse", type=int, default=None)
    parser.add_argument("--latent-dim-mid", type=int, default=None)
    parser.add_argument("--latent-dim-fine", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--decoder-channels", default=None)
    parser.add_argument("--output-height", type=int, default=384)
    parser.add_argument("--output-width", type=int, default=512)
    parser.add_argument("--sin-frequency", type=float, default=None)
    parser.add_argument("--video-path", type=Path, default=Path("upstream/videos/0.mkv"))
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument("--full-lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval-epochs", type=int, default=25)
    parser.add_argument("--decoder-codec", default="int8_mixed")
    parser.add_argument("--decoder-fake-quant-forward", action="store_true")
    parser.add_argument("--decoder-fake-quant-bits", type=int, default=8)
    parser.add_argument("--coder-qat", action="store_true")
    parser.add_argument("--coder-qat-bits", type=int, default=8)
    parser.add_argument("--coder-qat-quant-residual-weight", type=float, default=1.0e-4)
    parser.add_argument("--coder-qat-magnitude-weight", type=float, default=0.0)
    parser.add_argument("--coder-qat-delta-weight", type=float, default=0.0)
    parser.add_argument("--distillation-weight", type=float, default=0.0)
    parser.add_argument("--pose-distillation-weight", type=float, default=1.0)
    parser.add_argument("--allow-mock-scorer-teacher", action="store_true")
    parser.add_argument("--allow-segnet-only-research", action="store_true")
    parser.add_argument("--eval-roundtrip-ste", action="store_true")
    parser.add_argument("--pose-student-input-preprocess", choices=("rgb", "pr95_yuv6"), default="pr95_yuv6")
    parser.add_argument("--pr95-faithful-curriculum", action="store_true")
    parser.add_argument("--pr95-curriculum-total-epochs", type=int, default=None)
    parser.add_argument("--grad-clip-max-norm", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--optimizer-kind", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--cosine-decay", action="store_true")
    parser.add_argument("--cosine-decay-total-epochs", type=int, default=None)
    parser.add_argument("--cosine-decay-min-lr-ratio", type=float, default=1.0e-2)
    parser.add_argument("--ema-archive-selection", action="store_true")
    parser.add_argument("--smoke-export-archive", action="store_true")
    parser.add_argument("--post-export-receiver-cache-quality-gate", action="store_true")
    parser.add_argument(
        "--receiver-cache-quality-reference-cache-dir",
        type=Path,
        default=Path("experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600"),
    )
    parser.add_argument("--receiver-cache-quality-max-pairs", type=int, default=1)
    parser.add_argument("--receiver-cache-quality-batch-pairs", type=int, default=1)
    parser.add_argument("--receiver-cache-quality-min-segnet-std", type=float, default=1.0)
    parser.add_argument(
        "--receiver-cache-quality-min-segnet-dynamic-range",
        type=float,
        default=16.0,
    )
    parser.add_argument(
        "--receiver-cache-quality-max-segnet-mae-vs-reference-for-fit-gate",
        type=float,
        default=64.0,
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> Any:
    rows = {
        str(row["row_id"]): row["config"]
        for row in hi_nerv_modelsize_config_rows(num_pairs=int(args.num_pairs))
    }
    cfg = rows[str(args.modelsize_row)]
    updates: dict[str, Any] = {
        "num_pairs": int(args.num_pairs),
        "output_height": int(args.output_height),
        "output_width": int(args.output_width),
    }
    for attr in (
        "latent_dim_coarse",
        "latent_dim_mid",
        "latent_dim_fine",
        "embed_dim",
        "sin_frequency",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            updates[attr] = value
    if args.decoder_channels:
        updates["decoder_channels"] = tuple(
            int(part) for part in str(args.decoder_channels).split(",") if part
        )
    return replace(cfg, **updates)


def _coder_qat_config_from_args(args: argparse.Namespace) -> Any:
    from tac.substrates._shared.mlx_score_aware.coder_qat import CoderAwareQATConfig

    return CoderAwareQATConfig(
        enabled=bool(args.coder_qat),
        quant_bits=int(args.coder_qat_bits),
        quant_residual_weight=float(args.coder_qat_quant_residual_weight),
        magnitude_weight=float(args.coder_qat_magnitude_weight),
        delta_weight=float(args.coder_qat_delta_weight),
    ).validated()


def _resolve_output_dir(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.output_dir is not None:
        output = args.output_dir.expanduser()
        if not output.is_absolute():
            output = REPO_ROOT / output
        output = output.resolve(strict=False)
        if _looks_local(output) and not bool(args.allow_local_output_dir):
            raise StorageTierError(
                "hi_nerv_mlx_trainer_output_storage_preflight_failed: "
                "local_disk_tier_disabled"
            )
        output.mkdir(parents=True, exist_ok=True)
        return output, {
            "schema": "hi_nerv_mlx_trainer_explicit_output_preflight.v1",
            "selected_workload_root": output.as_posix(),
            "explicit_output_dir": True,
            "local_output_explicitly_allowed": bool(args.allow_local_output_dir),
            "operator_storage_policy": operator_storage_policy_payload(),
            "blockers": [],
            **FALSE_AUTHORITY,
        }

    tiers = parse_storage_tier_specs(
        operator_storage_tier_cli_specs(()),
        repo_root=REPO_ROOT,
        reserve_free_gb=float(args.storage_reserve_free_gb),
        allow_local_disk=False,
    )
    subdir = (
        f"{str(args.storage_workload_subdir).strip('/')}/"
        f"{args.modelsize_row!s}_{int(args.num_pairs)}pairs"
    )
    plan = plan_experiment_storage(
        tiers,
        workload_subdir=subdir,
        requested_bytes=int(args.storage_expected_bytes),
        min_free_bytes=0,
        create=True,
        probe_writable=True,
    )
    output = require_selected_storage(plan)
    payload = plan.to_dict()
    payload["operator_storage_policy"] = operator_storage_policy_payload()
    payload["selected_workload_root"] = output.as_posix()
    payload.update(FALSE_AUTHORITY)
    return output, payload


def _config_snapshot(cfg: Any) -> dict[str, Any]:
    return {
        "latent_dim_coarse": int(cfg.latent_dim_coarse),
        "latent_dim_mid": int(cfg.latent_dim_mid),
        "latent_dim_fine": int(cfg.latent_dim_fine),
        "embed_dim": int(cfg.embed_dim),
        "decoder_channels": [int(v) for v in cfg.decoder_channels],
        "sin_frequency": float(cfg.sin_frequency),
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "num_pairs": int(cfg.num_pairs),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
    }


def _looks_local(path: Path) -> bool:
    return not path.resolve(strict=False).as_posix().startswith("/Volumes/")


_METADATA_FORBIDDEN_AUTHORITY_KEYS = frozenset(
    {
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
        "promotable",
        "score_claim_valid",
    }
)


def _metadata_safe(value: Any) -> Any:
    """Drop nested authority keys before passing data into RendererBundle metadata."""

    if isinstance(value, dict):
        return {
            str(key): _metadata_safe(child)
            for key, child in value.items()
            if str(key) not in _METADATA_FORBIDDEN_AUTHORITY_KEYS
        }
    if isinstance(value, list):
        return [_metadata_safe(child) for child in value]
    return value


def _maybe_write_post_export_receiver_cache_quality(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    archive_path: str | Path | None,
) -> dict[str, Any] | None:
    if not bool(args.post_export_receiver_cache_quality_gate):
        return None
    if archive_path is None:
        return _write_post_export_receiver_cache_quality_refusal(
            output_dir=output_dir,
            blockers=["hi_nerv_archive_export_missing_for_receiver_cache_quality"],
        )
    archive = Path(archive_path).expanduser().resolve(strict=False)
    reference = args.receiver_cache_quality_reference_cache_dir.expanduser()
    if not reference.is_absolute():
        reference = (REPO_ROOT / reference).resolve(strict=False)
    if not archive.is_file():
        return _write_post_export_receiver_cache_quality_refusal(
            output_dir=output_dir,
            blockers=["hi_nerv_archive_export_path_missing_for_receiver_cache_quality"],
            archive_path=archive,
            reference_cache_dir=reference,
        )
    if not reference.is_dir():
        return _write_post_export_receiver_cache_quality_refusal(
            output_dir=output_dir,
            blockers=["hi_nerv_reference_cache_missing_for_receiver_cache_quality"],
            archive_path=archive,
            reference_cache_dir=reference,
        )
    from tac.substrates.hi_nerv.receiver_cache_quality import (
        write_hi_nerv_receiver_cache_quality_report,
    )

    return write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=output_dir / "post_export_receiver_cache_quality",
        reference_cache_dir=reference,
        max_pairs=int(args.receiver_cache_quality_max_pairs),
        batch_pairs=int(args.receiver_cache_quality_batch_pairs),
        sample_pairs=int(args.receiver_cache_quality_max_pairs),
        min_segnet_std=float(args.receiver_cache_quality_min_segnet_std),
        min_segnet_dynamic_range=float(
            args.receiver_cache_quality_min_segnet_dynamic_range
        ),
        max_segnet_mae_vs_reference_for_fit_gate=float(
            args.receiver_cache_quality_max_segnet_mae_vs_reference_for_fit_gate
        ),
    )


def _write_post_export_receiver_cache_quality_refusal(
    *,
    output_dir: Path,
    blockers: list[str],
    archive_path: Path | None = None,
    reference_cache_dir: Path | None = None,
) -> dict[str, Any]:
    report = {
        "schema": "hi_nerv_receiver_cache_quality_report.v1",
        "output_dir": (output_dir / "post_export_receiver_cache_quality").as_posix(),
        "archive_path": archive_path.as_posix() if archive_path is not None else None,
        "reference_cache_dir": (
            reference_cache_dir.as_posix() if reference_cache_dir is not None else None
        ),
        "quality_gate": None,
        "quality_gate_passed": False,
        "blockers": [
            "hi_nerv_receiver_cache_quality_is_false_authority",
            *[str(blocker) for blocker in blockers],
        ],
        **FALSE_AUTHORITY,
    }
    out = output_dir / "post_export_receiver_cache_quality"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "hi_nerv_receiver_cache_quality_report.json"
    report["report_path"] = path.as_posix()
    write_json(path, report)
    return report


def _attach_post_export_receiver_cache_quality_to_training_artifact(
    *,
    output_dir: Path,
    report: dict[str, Any],
) -> None:
    artifact_path = output_dir / "training_artifact.json"
    if not artifact_path.is_file():
        return
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata = dict(artifact.get("substrate_artifact_metadata") or {})
    metadata["post_export_receiver_cache_quality"] = (
        _receiver_cache_quality_manifest_summary(report)
    )
    artifact["substrate_artifact_metadata"] = metadata
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _receiver_cache_quality_manifest_summary(
    report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if report is None:
        return None
    gate = report.get("quality_gate") if isinstance(report, dict) else None
    gate_stats = gate.get("stats") if isinstance(gate, dict) else None
    return {
        "schema": "hi_nerv_receiver_cache_quality_summary.v1",
        "report_path": report.get("report_path"),
        "archive_path": report.get("archive_path"),
        "archive_sha256": report.get("archive_sha256"),
        "candidate_cache_dir": report.get("candidate_cache_dir"),
        "quality_gate_path": report.get("quality_gate_path"),
        "quality_gate_verdict": gate.get("verdict") if isinstance(gate, dict) else None,
        "quality_gate_passed": bool(report.get("quality_gate_passed")),
        "candidate_segnet_last_rgb_stats": (
            gate_stats.get("candidate_segnet_last_rgb")
            if isinstance(gate_stats, dict)
            else None
        ),
        "distance_to_reference": (
            gate.get("distance_to_reference") if isinstance(gate, dict) else None
        ),
        "blockers": [str(blocker) for blocker in report.get("blockers") or []],
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.full:
        return _full_main(args)
    return _smoke_main(args)


__all__ = [
    "TRAINER_SCHEMA",
    "_build_parser",
    "_coder_qat_config_from_args",
    "_config_from_args",
    "_metadata_safe",
    "_resolve_output_dir",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
