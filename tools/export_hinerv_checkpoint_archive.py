#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export a byte-closed HiNeRV archive from a live MLX checkpoint.

This is a mid-run rate-feedback tool. It restores a canonical `.npsd`
checkpoint into the exact HiNeRV MLX model described by the runner startup
artifact, then delegates archive construction to
`tac.substrates.hi_nerv.archive_candidate.export_hi_nerv_mlx_archive`.
It does not claim score authority.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_modelsize_budget import build_hinerv_config_from_size_knobs  # noqa: E402
from tac.local_acceleration.mlx_preprocess import (  # noqa: E402
    write_scorer_input_cache_from_raw_file,
    write_scorer_input_cache_from_video_file,
)
from tac.local_acceleration.mlx_scorer_response import (  # noqa: E402
    MANIFEST_CACHE_INTEGRITY_MODE,
    build_mlx_scorer_response_payload,
)
from tac.repo_io import sha256_file, write_json_artifact  # noqa: E402
from tac.substrates._shared.mlx_score_aware import RendererBundle  # noqa: E402
from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter  # noqa: E402
from tac.substrates._shared.numpy_portable_inflate import unpack_state_dict_numpy  # noqa: E402
from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive  # noqa: E402
from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX  # noqa: E402

FALSE_AUTHORITY = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
HINERV_CHECKPOINT_EXPORT_DEFAULT_DECODER_CODEC = "portfolio_auto"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-json", required=True, type=Path)
    parser.add_argument("--checkpoint-meta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--state-kind",
        choices=("ema", "live"),
        default="ema",
        help="Checkpoint state to export. EMA is the normal archive-selection surface.",
    )
    parser.add_argument("--decoder-codec", default=None)
    parser.add_argument("--latent-codec", default=None)
    parser.add_argument("--emit-receiver-proof", action="store_true")
    parser.add_argument("--retain-receiver-proof-output", action="store_true")
    parser.add_argument(
        "--allow-over-hard-byte-ceiling-for-measurement",
        action="store_true",
        help=(
            "Do not fail before receiver proof when the measured archive exceeds "
            "the active hard byte ceiling. The report still records the over-cap "
            "blocker and remains false-authority."
        ),
    )
    parser.add_argument(
        "--write-mlx-prefilter-profile",
        action="store_true",
        help=(
            "Build a false-authority full-video MLX scorer replay from the "
            "retained receiver raw output."
        ),
    )
    parser.add_argument("--mlx-prefilter-scorer-device", default="cpu")
    parser.add_argument("--mlx-prefilter-scorer-batch-pairs", default=1, type=int)
    parser.add_argument("--mlx-prefilter-progress-every", default=50, type=int)
    parser.add_argument("--source-video-path", default=None, type=Path)
    parser.add_argument("--scorer-upstream-dir", default=None, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--output-json", default=None, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_json = args.output_json or args.output_dir / "hinerv_checkpoint_archive_export.json"
    report = export_checkpoint_archive(
        startup_json=args.startup_json,
        checkpoint_meta=args.checkpoint_meta,
        output_dir=args.output_dir,
        output_json=output_json,
        state_kind=args.state_kind,
        decoder_codec=args.decoder_codec,
        latent_codec=args.latent_codec,
        emit_receiver_proof=bool(args.emit_receiver_proof),
        retain_receiver_proof_output=bool(args.retain_receiver_proof_output),
        allow_over_hard_byte_ceiling_for_measurement=bool(
            args.allow_over_hard_byte_ceiling_for_measurement
        ),
        write_mlx_prefilter_profile=bool(args.write_mlx_prefilter_profile),
        mlx_prefilter_scorer_device=str(args.mlx_prefilter_scorer_device),
        mlx_prefilter_scorer_batch_pairs=int(args.mlx_prefilter_scorer_batch_pairs),
        mlx_prefilter_progress_every=int(args.mlx_prefilter_progress_every),
        source_video_path=args.source_video_path,
        scorer_upstream_dir=args.scorer_upstream_dir,
        repo_root=args.repo_root,
    )
    print(json.dumps(_summary(report), indent=2, sort_keys=True))
    return 0


def export_checkpoint_archive(
    *,
    startup_json: str | Path,
    checkpoint_meta: str | Path,
    output_dir: str | Path,
    output_json: str | Path | None = None,
    state_kind: str = "ema",
    decoder_codec: str | None = None,
    latent_codec: str | None = None,
    emit_receiver_proof: bool = False,
    retain_receiver_proof_output: bool = False,
    allow_over_hard_byte_ceiling_for_measurement: bool = False,
    write_mlx_prefilter_profile: bool = False,
    mlx_prefilter_scorer_device: str = "cpu",
    mlx_prefilter_scorer_batch_pairs: int = 1,
    mlx_prefilter_progress_every: int = 50,
    source_video_path: str | Path | None = None,
    scorer_upstream_dir: str | Path | None = None,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    startup_path = Path(startup_json).expanduser().resolve(strict=False)
    meta_path = Path(checkpoint_meta).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    root = Path(repo_root).expanduser().resolve(strict=False)
    startup = _read_json(startup_path)
    meta = _read_json(meta_path)
    candidate = _require_mapping(startup.get("modelsize_candidate"), "modelsize_candidate")
    command_args = _require_mapping(startup.get("command_args"), "command_args")

    cfg = build_hinerv_config_from_size_knobs(
        num_pairs=int(candidate.get("num_pairs") or command_args.get("num_pairs") or 600),
        latent_dim=int(candidate["latent_dim"]),
        embed_dim=int(candidate["embed_dim"]),
        decoder_channel=int(candidate["decoder_channel"]),
        use_hierarchical_feature_grid=bool(candidate.get("use_hierarchical_feature_grid")),
        use_convnext_blocks=bool(candidate.get("use_convnext_blocks")),
        local_grid_levels=int(candidate.get("local_grid_levels") or 2),
        local_grid_channels=int(candidate.get("local_grid_channels") or 4),
        convnext_mlp_ratio=int(candidate.get("convnext_mlp_ratio") or 2),
        convnext_kernel_size=int(candidate.get("convnext_kernel_size") or 7),
        mid_injection_block_index=int(candidate.get("mid_injection_block_index") or 1),
        fine_injection_block_index=int(candidate.get("fine_injection_block_index") or 4),
    )
    model = HinervSubstrateMLX(cfg)
    adapter = MlxScoreAwareAdapter(
        RendererBundle(
            model=model,
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=int(cfg.num_pairs),
            forward_convention="call_b2chw_255",
        ),
        substrate_id="compact_runner_hi_nerv_mlx_checkpoint_export",
    )
    state_path = _checkpoint_state_path(meta, state_kind=state_kind)
    state = unpack_state_dict_numpy(state_path.read_bytes())
    modelsize_integrity = _modelsize_integrity_profile(
        state,
        candidate=candidate,
        cfg=cfg,
    )
    adapter.import_state_dict(model, state_path)
    decoder_codec_resolution = _resolve_decoder_codec(
        explicit_arg=decoder_codec,
        command_args=command_args,
        candidate=candidate,
    )
    resolved_decoder_codec = str(decoder_codec_resolution["resolved"])
    resolved_latent_codec = str(
        latent_codec or command_args.get("hi_nerv_latent_codec") or "int16_raw"
    )
    hard_byte_ceiling = _export_hard_byte_ceiling(
        candidate=candidate,
        hard_byte_ceilings=startup.get("hard_byte_ceilings") or [],
    )
    enforced_hard_byte_ceiling = (
        None
        if allow_over_hard_byte_ceiling_for_measurement
        else hard_byte_ceiling
    )
    archive_path, archive_sha256, archive_bytes = export_hi_nerv_mlx_archive(
        model,
        out,
        repo_root=root,
        emit_archive_bound_candidate_package=bool(emit_receiver_proof),
        retain_receiver_proof_output=bool(retain_receiver_proof_output),
        decoder_codec=resolved_decoder_codec,
        latent_codec=resolved_latent_codec,
        source_backend="mlx_checkpoint_npsd",
        mlx_triage_argv=[
            "tools/export_hinerv_checkpoint_archive.py",
            "--startup-json",
            startup_path.as_posix(),
            "--checkpoint-meta",
            meta_path.as_posix(),
        ],
        hard_byte_ceiling=enforced_hard_byte_ceiling,
    )
    receiver_proof_path = out / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json"
    receiver_proof = _read_json(receiver_proof_path) if receiver_proof_path.is_file() else {}
    section_profile = _section_profile(out, archive_bytes=int(archive_bytes))
    report_path = (
        Path(output_json)
        if output_json is not None
        else out / "hinerv_checkpoint_archive_export.json"
    ).expanduser().resolve(strict=False)
    pending_mlx_prefilter_profile = {
        "schema": "hinerv_checkpoint_mlx_prefilter_profile.v1",
        "written": False,
        "profile_path": None,
        "blockers": (
            ["hinerv_checkpoint_mlx_prefilter_pending"]
            if write_mlx_prefilter_profile
            else ["hinerv_checkpoint_mlx_prefilter_not_requested"]
        ),
        **FALSE_AUTHORITY,
    }
    preliminary_report = {
        "schema": "hinerv_checkpoint_archive_export.v1",
        "family": "hi_nerv",
        "report_status": "archive_receiver_proof_written_prefilter_pending",
        "candidate_id": candidate.get("candidate_id"),
        "modelsize_candidate": candidate,
        "checkpoint_meta_path": meta_path.as_posix(),
        "checkpoint_meta_sha256": sha256_file(meta_path),
        "checkpoint_epoch": meta.get("global_epoch"),
        "checkpoint_state_kind": state_kind,
        "checkpoint_state_path": state_path.as_posix(),
        "checkpoint_state_sha256": sha256_file(state_path),
        "modelsize_integrity": modelsize_integrity,
        "startup_json_path": startup_path.as_posix(),
        "startup_json_sha256": sha256_file(startup_path),
        "output_dir": out.as_posix(),
        "archive_path": Path(archive_path).as_posix(),
        "archive_sha256": archive_sha256,
        "archive_bytes": int(archive_bytes),
        "hard_byte_ceiling_enforced_by_export": enforced_hard_byte_ceiling,
        "hard_byte_ceiling_requested_by_candidate_or_startup": hard_byte_ceiling,
        "hard_byte_ceiling_measurement_bypass_enabled": bool(
            allow_over_hard_byte_ceiling_for_measurement
        ),
        "rate_byte_profile": section_profile,
        "hard_byte_ceilings": list(startup.get("hard_byte_ceilings") or []),
        "decoder_codec": resolved_decoder_codec,
        "decoder_codec_resolution": decoder_codec_resolution,
        "latent_codec": resolved_latent_codec,
        "receiver_proof_path": (
            receiver_proof_path.as_posix() if receiver_proof_path.is_file() else None
        ),
        "receiver_proof_sha256": (
            sha256_file(receiver_proof_path) if receiver_proof_path.is_file() else None
        ),
        "receiver_proof_ready": bool(
            receiver_proof.get("runtime_consumption_proof_ready")
        ),
        "local_mlx_prefilter_profile": pending_mlx_prefilter_profile,
        "local_mlx_prefilter_profile_path": None,
        "local_mlx_prefilter_written": False,
        "report_path": report_path.as_posix(),
        "modelsize_byte_cap_feedback_row": _modelsize_byte_cap_feedback_row(
            candidate=candidate,
            archive_bytes=int(archive_bytes),
            hard_byte_ceilings=startup.get("hard_byte_ceilings") or [],
            decoder_codec=resolved_decoder_codec,
            receiver_proof_ready=bool(
                receiver_proof.get("runtime_consumption_proof_ready")
            ),
            archive_path=Path(archive_path),
            archive_sha256=archive_sha256,
            hard_byte_ceiling_enforced_by_export=enforced_hard_byte_ceiling,
            hard_byte_ceiling_measurement_bypass_enabled=bool(
                allow_over_hard_byte_ceiling_for_measurement
            ),
        ),
        "blockers": _blockers(
            archive_bytes=int(archive_bytes),
            hard_byte_ceiling=hard_byte_ceiling,
            hard_byte_ceilings=startup.get("hard_byte_ceilings") or [],
            receiver_proof=receiver_proof,
            receiver_proof_requested=bool(emit_receiver_proof),
            modelsize_integrity=modelsize_integrity,
            decoder_codec_resolution=decoder_codec_resolution,
            mlx_prefilter_profile=pending_mlx_prefilter_profile,
            hard_byte_ceiling_measurement_bypass_enabled=bool(
                allow_over_hard_byte_ceiling_for_measurement
            ),
        ),
        **FALSE_AUTHORITY,
    }
    preliminary_write = write_json_artifact(report_path, preliminary_report)
    mlx_prefilter_profile = _maybe_write_receiver_raw_cache_mlx_prefilter(
        requested=bool(write_mlx_prefilter_profile),
        output_dir=out / "local_mlx_prefilter",
        receiver_proof=receiver_proof,
        startup=startup,
        command_args=command_args,
        candidate=candidate,
        archive_bytes=int(archive_bytes),
        archive_sha256=str(archive_sha256),
        source_video_path=source_video_path,
        scorer_upstream_dir=scorer_upstream_dir,
        scorer_device=_canonical_mlx_prefilter_device(mlx_prefilter_scorer_device),
        scorer_batch_pairs=int(mlx_prefilter_scorer_batch_pairs),
        progress_every=int(mlx_prefilter_progress_every),
        repo_root=root,
    )
    report = {
        "schema": "hinerv_checkpoint_archive_export.v1",
        "family": "hi_nerv",
        "report_status": "complete",
        "candidate_id": candidate.get("candidate_id"),
        "modelsize_candidate": candidate,
        "checkpoint_meta_path": meta_path.as_posix(),
        "checkpoint_meta_sha256": sha256_file(meta_path),
        "checkpoint_epoch": meta.get("global_epoch"),
        "checkpoint_state_kind": state_kind,
        "checkpoint_state_path": state_path.as_posix(),
        "checkpoint_state_sha256": sha256_file(state_path),
        "modelsize_integrity": modelsize_integrity,
        "startup_json_path": startup_path.as_posix(),
        "startup_json_sha256": sha256_file(startup_path),
        "output_dir": out.as_posix(),
        "archive_path": Path(archive_path).as_posix(),
        "archive_sha256": archive_sha256,
        "archive_bytes": int(archive_bytes),
        "hard_byte_ceiling_enforced_by_export": enforced_hard_byte_ceiling,
        "hard_byte_ceiling_requested_by_candidate_or_startup": hard_byte_ceiling,
        "hard_byte_ceiling_measurement_bypass_enabled": bool(
            allow_over_hard_byte_ceiling_for_measurement
        ),
        "rate_byte_profile": section_profile,
        "hard_byte_ceilings": list(startup.get("hard_byte_ceilings") or []),
        "decoder_codec": resolved_decoder_codec,
        "decoder_codec_resolution": decoder_codec_resolution,
        "latent_codec": resolved_latent_codec,
        "receiver_proof_path": receiver_proof_path.as_posix() if receiver_proof_path.is_file() else None,
        "receiver_proof_sha256": sha256_file(receiver_proof_path) if receiver_proof_path.is_file() else None,
        "receiver_proof_ready": bool(receiver_proof.get("runtime_consumption_proof_ready")),
        "local_mlx_prefilter_profile": mlx_prefilter_profile,
        "local_mlx_prefilter_profile_path": mlx_prefilter_profile.get("profile_path"),
        "local_mlx_prefilter_written": mlx_prefilter_profile.get("written") is True,
        "report_path": report_path.as_posix(),
        "modelsize_byte_cap_feedback_row": _modelsize_byte_cap_feedback_row(
            candidate=candidate,
            archive_bytes=int(archive_bytes),
            hard_byte_ceilings=startup.get("hard_byte_ceilings") or [],
            decoder_codec=resolved_decoder_codec,
            receiver_proof_ready=bool(
                receiver_proof.get("runtime_consumption_proof_ready")
            ),
            archive_path=Path(archive_path),
            archive_sha256=archive_sha256,
            hard_byte_ceiling_enforced_by_export=enforced_hard_byte_ceiling,
            hard_byte_ceiling_measurement_bypass_enabled=bool(
                allow_over_hard_byte_ceiling_for_measurement
            ),
        ),
        "blockers": _blockers(
            archive_bytes=int(archive_bytes),
            hard_byte_ceiling=hard_byte_ceiling,
            hard_byte_ceilings=startup.get("hard_byte_ceilings") or [],
            receiver_proof=receiver_proof,
            receiver_proof_requested=bool(emit_receiver_proof),
            modelsize_integrity=modelsize_integrity,
            decoder_codec_resolution=decoder_codec_resolution,
            mlx_prefilter_profile=mlx_prefilter_profile,
            hard_byte_ceiling_measurement_bypass_enabled=bool(
                allow_over_hard_byte_ceiling_for_measurement
            ),
        ),
        **FALSE_AUTHORITY,
    }
    write_json_artifact(
        report_path,
        report,
        allow_overwrite=True,
        expected_existing_sha256=preliminary_write.sha256,
    )
    return report


def _modelsize_byte_cap_feedback_row(
    *,
    candidate: dict[str, Any],
    archive_bytes: int,
    hard_byte_ceilings: Any,
    decoder_codec: str,
    receiver_proof_ready: bool,
    archive_path: Path,
    archive_sha256: str,
    hard_byte_ceiling_enforced_by_export: int | None,
    hard_byte_ceiling_measurement_bypass_enabled: bool,
) -> dict[str, Any]:
    ceiling = _export_hard_byte_ceiling(
        candidate=candidate,
        hard_byte_ceilings=hard_byte_ceilings,
    )
    nominal = (
        _optional_int(candidate.get("nominal_total_payload_bytes"))
        or _optional_int(candidate.get("total_payload_bytes"))
        or _optional_int(candidate.get("estimated_total_payload_bytes"))
    )
    archive_minus_nominal = (
        None if nominal is None else int(archive_bytes) - int(nominal)
    )
    archive_to_nominal = (
        None
        if nominal is None or int(nominal) <= 0
        else float(archive_bytes) / float(nominal)
    )
    overrun = (
        None if ceiling is None else max(0, int(archive_bytes) - int(ceiling))
    )
    required_nominal_max = None
    if (
        ceiling is not None
        and nominal is not None
        and int(archive_bytes) > 0
    ):
        required_nominal_max = math.floor(
            float(ceiling) * float(nominal) / float(archive_bytes)
        )
    return {
        "schema": "nerv_modelsize_byte_cap_feedback_row.v1",
        "family": "hi_nerv",
        "candidate_id": candidate.get("candidate_id"),
        "codec": str(decoder_codec),
        "decoder_codec": str(decoder_codec),
        "modelsize_candidate": candidate,
        "hard_byte_ceiling": ceiling,
        "hard_byte_ceiling_enforced_by_export": hard_byte_ceiling_enforced_by_export,
        "hard_byte_ceiling_measurement_bypass_enabled": bool(
            hard_byte_ceiling_measurement_bypass_enabled
        ),
        "nominal_total_payload_bytes": nominal,
        "measured_archive_bytes": int(archive_bytes),
        "archive_bytes": int(archive_bytes),
        "archive_path": archive_path.as_posix(),
        "archive_sha256": archive_sha256,
        "archive_minus_nominal_bytes": archive_minus_nominal,
        "archive_to_nominal_ratio": archive_to_nominal,
        "calibrated_archive_overrun_bytes": overrun,
        "required_nominal_payload_bytes_max": required_nominal_max,
        "receiver_proof_ready": bool(receiver_proof_ready),
        "receiver_closed": bool(receiver_proof_ready),
        "authority_surface": "measured_archive_zip_bytes_after_receiver_export",
        **FALSE_AUTHORITY,
    }


def _resolve_decoder_codec(
    *,
    explicit_arg: Any,
    command_args: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Resolve checkpoint export codec without letting runner defaults mask candidates."""

    explicit = _optional_nonempty_str(explicit_arg)
    runner = _optional_nonempty_str(command_args.get("compact_decoder_codec"))
    candidate_codec = _optional_nonempty_str(candidate.get("decoder_codec"))
    runner_default_like = (
        runner is None or runner == HINERV_CHECKPOINT_EXPORT_DEFAULT_DECODER_CODEC
    )
    if explicit is not None:
        resolved = explicit
        source = "explicit_arg"
    elif candidate_codec is not None and runner_default_like:
        resolved = candidate_codec
        source = "modelsize_candidate_decoder_codec"
    elif runner is not None:
        resolved = runner
        source = "runner_compact_decoder_codec"
    else:
        resolved = HINERV_CHECKPOINT_EXPORT_DEFAULT_DECODER_CODEC
        source = "checkpoint_export_default"
    candidate_propagates = source == "modelsize_candidate_decoder_codec"
    candidate_ignored = candidate_codec is not None and not candidate_propagates
    blockers = (
        ["candidate_decoder_codec_not_export_authority"] if candidate_ignored else []
    )
    return {
        "schema": "hinerv_checkpoint_decoder_codec_resolution.v1",
        "explicit_arg": explicit_arg,
        "runner_compact_decoder_codec": runner,
        "runner_compact_decoder_codec_default_like": runner_default_like,
        "modelsize_candidate_decoder_codec": candidate_codec,
        "resolved": resolved,
        "resolution_source": source,
        "candidate_codec_takes_precedence_over_runner_default": bool(
            source == "modelsize_candidate_decoder_codec" and runner_default_like
        ),
        "modelsize_candidate_decoder_codec_propagates_to_export": candidate_propagates,
        "modelsize_candidate_decoder_codec_is_capacity_authority": candidate_propagates,
        "candidate_codec_advisory_reason": (
            "modelsize candidate decoder_codec propagated through checkpoint export "
            "because runner codec was absent/default-like"
            if candidate_propagates
            else "modelsize candidates describe graph capacity; decoder codec promotion is "
            "measured by the resolved export codec"
        ),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _checkpoint_state_path(meta: dict[str, Any], *, state_kind: str) -> Path:
    key = "ema_shadow_state_path" if state_kind == "ema" else "live_state_path"
    value = meta.get(key)
    if not value:
        raise ValueError(f"checkpoint meta missing {key}")
    path = Path(str(value)).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint state not found: {path}")
    return path


def _blockers(
    *,
    archive_bytes: int,
    hard_byte_ceilings: Any,
    hard_byte_ceiling: int | None = None,
    receiver_proof: dict[str, Any],
    receiver_proof_requested: bool,
    modelsize_integrity: dict[str, Any],
    decoder_codec_resolution: dict[str, Any],
    mlx_prefilter_profile: dict[str, Any] | None = None,
    hard_byte_ceiling_measurement_bypass_enabled: bool = False,
) -> list[str]:
    blockers = [
        "macos_mlx_checkpoint_export_false_authority",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    mlx_prefilter_profile = dict(mlx_prefilter_profile or {})
    if mlx_prefilter_profile.get("written") is True:
        blockers.extend(
            str(blocker)
            for blocker in mlx_prefilter_profile.get("blockers") or ()
            if str(blocker)
        )
    else:
        blockers.append("full_video_scorer_replay_not_executed")
        blockers.extend(
            str(blocker)
            for blocker in mlx_prefilter_profile.get("blockers") or ()
            if str(blocker)
        )
    ceiling = hard_byte_ceiling or _min_positive_int(hard_byte_ceilings)
    if ceiling is not None and int(archive_bytes) > int(ceiling):
        blockers.append("archive_bytes_exceed_tightest_hard_ceiling")
        if hard_byte_ceiling_measurement_bypass_enabled:
            blockers.append("hard_byte_ceiling_export_bypassed_for_measurement")
    if not receiver_proof_requested:
        blockers.append("receiver_proof_not_requested")
    elif receiver_proof.get("runtime_consumption_proof_ready") is not True:
        blockers.append("receiver_proof_not_ready")
    blockers.extend(str(v) for v in modelsize_integrity.get("blockers") or [])
    blockers.extend(str(v) for v in decoder_codec_resolution.get("blockers") or [])
    return list(dict.fromkeys(blockers))


def _modelsize_integrity_profile(
    state: dict[str, Any],
    *,
    candidate: dict[str, Any],
    cfg: Any,
) -> dict[str, Any]:
    """Prove candidate size controls are bound to checkpoint tensor shapes."""

    total_params = int(sum(int(getattr(value, "size", 0)) for value in state.values()))
    expected_total = _optional_int(candidate.get("total_trainable_params"))
    expected_decoder_channel = int(candidate.get("decoder_channel") or cfg.decoder_channels[-1])
    expected_num_pairs = int(candidate.get("num_pairs") or cfg.num_pairs)
    expected_latents = {
        "latents_coarse": (expected_num_pairs, int(cfg.latent_dim_coarse)),
        "latents_mid": (expected_num_pairs, int(cfg.latent_dim_mid)),
        "latents_fine": (expected_num_pairs, int(cfg.latent_dim_fine)),
    }
    expected_required_prefixes: dict[str, bool] = {
        "feature_grids.": bool(candidate.get("use_hierarchical_feature_grid")),
        "convnext_blocks.": bool(candidate.get("use_convnext_blocks")),
    }
    blockers: list[str] = []
    observed_latents: dict[str, list[int] | None] = {}
    for key, expected_shape in expected_latents.items():
        value = state.get(key)
        observed = None if value is None else [int(v) for v in value.shape]
        observed_latents[key] = observed
        if observed != [int(v) for v in expected_shape]:
            blockers.append(f"hinerv_modelsize_latent_shape_mismatch:{key}")
    if expected_total is not None and int(expected_total) != total_params:
        blockers.append("hinerv_modelsize_total_trainable_params_mismatch")
    conv_weight = state.get("blocks.0.conv.weight")
    observed_decoder_channel_raw = (
        int(conv_weight.shape[0])
        if conv_weight is not None and getattr(conv_weight, "ndim", 0) >= 1
        else None
    )
    observed_decoder_channel = (
        observed_decoder_channel_raw // 4
        if observed_decoder_channel_raw is not None
        else None
    )
    if observed_decoder_channel != expected_decoder_channel:
        blockers.append("hinerv_modelsize_decoder_channel_mismatch")
    prefix_presence: dict[str, bool] = {}
    for prefix, required in expected_required_prefixes.items():
        present = any(str(key).startswith(prefix) for key in state)
        prefix_presence[prefix] = bool(present)
        if required and not present:
            blockers.append(f"hinerv_modelsize_required_tensor_prefix_missing:{prefix}")
        if not required and present:
            blockers.append(f"hinerv_modelsize_unexpected_tensor_prefix_present:{prefix}")
    modelsize_mparams = candidate.get("modelsize_mparams")
    observed_mparams = float(total_params) / 1_000_000.0
    if modelsize_mparams is not None and abs(float(modelsize_mparams) - observed_mparams) > 1e-6:
        blockers.append("hinerv_modelsize_mparams_metadata_mismatch")
    return {
        "schema": "hinerv_checkpoint_modelsize_integrity.v1",
        "profile_ready": True,
        "candidate_id": candidate.get("candidate_id"),
        "control_semantics": (
            (candidate.get("modelsize_control_contract") or {}).get("control_semantics")
            if isinstance(candidate.get("modelsize_control_contract"), dict)
            else None
        ),
        "modelsize_mparams_caps_archive_zip_bytes": False,
        "archive_byte_authority_surface": "measured_archive_zip_after_export",
        "expected_total_trainable_params": expected_total,
        "observed_total_trainable_params": total_params,
        "expected_modelsize_mparams": modelsize_mparams,
        "observed_modelsize_mparams": observed_mparams,
        "expected_decoder_channel": expected_decoder_channel,
        "observed_decoder_channel": observed_decoder_channel,
        "observed_first_block_conv_out_channels": observed_decoder_channel_raw,
        "decoder_channel_shape_rule": "blocks.0.conv.weight_out_channels_div_4_after_pixelshuffle",
        "expected_latent_shapes": {
            key: [int(v) for v in shape] for key, shape in expected_latents.items()
        },
        "observed_latent_shapes": observed_latents,
        "prefix_presence": prefix_presence,
        "matches_candidate_controls": not blockers,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _min_positive_int(values: Any) -> int | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        values = [values]
    try:
        ints = [int(value) for value in values if int(value) > 0]
    except (TypeError, ValueError):
        return None
    return min(ints) if ints else None


def _export_hard_byte_ceiling(
    *,
    candidate: dict[str, Any],
    hard_byte_ceilings: Any,
) -> int | None:
    """Return the tightest active measured archive-byte ceiling.

    Modelsize candidates and runner startup rows can both carry ceilings.  The
    materializer must honor the strictest positive value; otherwise a looser
    runner default can silently weaken a candidate's byte contract.
    """

    values: list[int] = []
    candidate_ceiling = _optional_int(candidate.get("hard_byte_ceiling"))
    if candidate_ceiling is not None:
        values.append(candidate_ceiling)
    startup_ceiling = _min_positive_int(hard_byte_ceilings)
    if startup_ceiling is not None:
        values.append(startup_ceiling)
    positives = [int(value) for value in values if int(value) > 0]
    return min(positives) if positives else None


def _optional_nonempty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _section_profile(output_dir: Path, *, archive_bytes: int) -> dict[str, Any]:
    manifest_path = output_dir / "hprc_representation_spine_hi_nerv_manifest.json"
    if not manifest_path.is_file():
        return {
            "schema": "hinerv_checkpoint_archive_rate_byte_profile.v1",
            "profile_ready": False,
            "blockers": ["hprc_spine_manifest_missing"],
        }
    manifest = _read_json(manifest_path)
    spine = (
        manifest.get("manifest", {})
        .get("representation_spine", {})
    )
    sections = []
    for section in spine.get("sections") or []:
        if not isinstance(section, dict):
            continue
        byte_count = int(section.get("bytes") or 0)
        sections.append(
            {
                "name": str(section.get("name") or ""),
                "role": str(section.get("role") or ""),
                "bytes": byte_count,
                "archive_fraction": byte_count / float(archive_bytes) if archive_bytes > 0 else None,
                "sha256": section.get("sha256"),
            }
        )
    section_bytes = sum(int(section["bytes"]) for section in sections)
    overhead_bytes = int(archive_bytes) - section_bytes
    return {
        "schema": "hinerv_checkpoint_archive_rate_byte_profile.v1",
        "profile_ready": bool(sections),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "hprc_bin_bytes": int(manifest.get("hprc_bin_bytes") or spine.get("hprc_bin_bytes") or 0),
        "archive_bytes": int(archive_bytes),
        "section_payload_bytes": int(section_bytes),
        "archive_overhead_and_manifest_bytes": int(overhead_bytes),
        "sections": sections,
        "dominant_sections": sorted(sections, key=lambda row: int(row["bytes"]), reverse=True)[:4],
        "blockers": [] if sections else ["hprc_spine_sections_missing"],
    }


def _maybe_write_receiver_raw_cache_mlx_prefilter(
    *,
    requested: bool,
    output_dir: str | Path,
    receiver_proof: dict[str, Any],
    startup: dict[str, Any],
    command_args: dict[str, Any],
    candidate: dict[str, Any],
    archive_bytes: int | None,
    archive_sha256: str | None,
    source_video_path: str | Path | None,
    scorer_upstream_dir: str | Path | None,
    scorer_device: str,
    scorer_batch_pairs: int,
    progress_every: int,
    repo_root: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir).expanduser().resolve(strict=False)
    batch_control = _canonical_mlx_prefilter_batch_pairs(scorer_batch_pairs)
    effective_batch_pairs = int(batch_control["effective_batch_pairs"])
    base = {
        "schema": "hinerv_checkpoint_receiver_raw_mlx_prefilter_request.v1",
        "requested": bool(requested),
        "profile_path": (out / "local_mlx_prefilter_profile.json").as_posix(),
        "progress_path": None,
        "receiver_decoded_archive_raw_required": True,
        "scorer_batch_pairs_requested": int(batch_control["requested_batch_pairs"]),
        "scorer_batch_pairs_effective": effective_batch_pairs,
        "scorer_batch_pairs_normalized_to_singleton": bool(
            batch_control["normalized_to_singleton"]
        ),
        "scorer_batch_pairs_normalization": batch_control,
        **FALSE_AUTHORITY,
    }
    if not requested:
        return {
            **base,
            "written": False,
            "blockers": ["hinerv_checkpoint_mlx_prefilter_not_requested"],
        }
    source_pair_indices = _hinerv_source_pair_indices(
        candidate=candidate,
        command_args=command_args,
    )
    n_pairs = len(source_pair_indices)
    if n_pairs < 1:
        return {
            **base,
            "written": False,
            "blockers": ["hinerv_checkpoint_mlx_prefilter_pair_count_missing"],
        }
    if source_pair_indices != tuple(range(n_pairs)):
        return {
            **base,
            "written": False,
            "source_pair_indices": [int(value) for value in source_pair_indices],
            "blockers": [
                "hinerv_checkpoint_mlx_prefilter_requires_prefix_source_pair_indices"
            ],
        }
    raw_path = _resolve_receiver_raw_path(
        receiver_proof.get("receiver_output_path"),
        repo_root=repo_root,
    )
    if raw_path is None or not raw_path.is_file() or raw_path.stat().st_size <= 0:
        return {
            **base,
            "written": False,
            "blockers": ["hinerv_checkpoint_mlx_prefilter_receiver_raw_missing"],
        }
    if receiver_proof.get("receiver_output_retained") is not True:
        return {
            **base,
            "written": False,
            "receiver_output_path": raw_path.as_posix(),
            "blockers": [
                "hinerv_checkpoint_mlx_prefilter_receiver_raw_not_retained"
            ],
        }
    if archive_bytes is None or archive_sha256 is None:
        return {
            **base,
            "written": False,
            "receiver_output_path": raw_path.as_posix(),
            "blockers": ["hinerv_checkpoint_mlx_prefilter_archive_identity_missing"],
        }
    source_video = _resolve_source_video_path(
        explicit=source_video_path,
        startup=startup,
        command_args=command_args,
        repo_root=repo_root,
    )
    upstream_dir = _resolve_scorer_upstream_dir(
        explicit=scorer_upstream_dir,
        command_args=command_args,
        repo_root=repo_root,
    )
    profile_path = out / "local_mlx_prefilter_profile.json"
    reference_cache_dir = out / "scorer_input_caches" / "reference_source_video"
    candidate_cache_dir = out / "scorer_input_caches" / "candidate_receiver_raw"
    components_dir = out / "scorer_input_caches" / "components"
    try:
        reference_manifest = _ensure_video_scorer_cache(
            source_video,
            reference_cache_dir,
            pair_count=n_pairs,
            batch_pairs=effective_batch_pairs,
        )
        candidate_manifest = _ensure_raw_scorer_cache(
            raw_path,
            candidate_cache_dir,
            archive_sha256=str(archive_sha256),
            inflated_outputs_aggregate_sha256=str(
                receiver_proof.get("receiver_output_sha256") or ""
            )
            or None,
            pair_count=n_pairs,
            batch_pairs=effective_batch_pairs,
        )
        response = build_mlx_scorer_response_payload(
            reference_cache_dir=reference_cache_dir,
            candidate_cache_dir=candidate_cache_dir,
            archive_size_bytes=int(archive_bytes),
            repo_root=repo_root,
            upstream_dir=upstream_dir,
            batch_pairs=effective_batch_pairs,
            device_type=str(scorer_device),
            components_dir=components_dir,
            progress_every=max(0, int(progress_every)),
            allow_gpu_research_signal=str(scorer_device) == "gpu",
            allow_unaudited_candidate_cache_debug=True,
            cache_integrity_mode=MANIFEST_CACHE_INTEGRITY_MODE,
            response_family="hi_nerv",
        )
        response = {
            **response,
            "schema": "mlx_scorer_response.v1",
            "schema_version": "mlx_scorer_response.v1",
            "hinerv_receiver_raw_cache_prefilter": {
                "schema": "hinerv_receiver_raw_cache_prefilter.v1",
                "receiver_output_path": raw_path.as_posix(),
                "receiver_output_sha256": receiver_proof.get(
                    "receiver_output_sha256"
                ),
                "reference_cache_manifest": reference_manifest,
                "candidate_cache_manifest": candidate_manifest,
                "source_pair_indices_alignment": "prefix_source_pair_indices",
                "scorer_batch_pairs_requested": int(
                    batch_control["requested_batch_pairs"]
                ),
                "scorer_batch_pairs_effective": effective_batch_pairs,
                "scorer_batch_pairs_normalized_to_singleton": bool(
                    batch_control["normalized_to_singleton"]
                ),
                "scorer_batch_pairs_normalization": batch_control,
                **FALSE_AUTHORITY,
            },
        }
        write_json_artifact(profile_path, response)
        reference_cleanup = _cleanup_rebuildable_scorer_cache_arrays(
            reference_cache_dir,
            reference_manifest,
            reason="hinerv_checkpoint_receiver_raw_prefilter_reference_cache_rebuildable",
        )
        candidate_cleanup = _cleanup_rebuildable_scorer_cache_arrays(
            candidate_cache_dir,
            candidate_manifest,
            reason="hinerv_checkpoint_receiver_raw_prefilter_candidate_cache_rebuildable",
        )
        return {
            **base,
            "written": True,
            "profile_schema": "mlx_scorer_response.v1",
            "profile_path": profile_path.as_posix(),
            "profile_sha256": sha256_file(profile_path),
            "cache_backed": True,
            "candidate_cache_dir": candidate_cache_dir.as_posix(),
            "reference_cache_dir": reference_cache_dir.as_posix(),
            "components_dir": components_dir.as_posix(),
            "scorer_response_schema": response.get("schema"),
            "n_samples": response.get("n_samples"),
            "score_recomputed_from_components": response.get(
                "score_recomputed_from_components"
            ),
            "avg_segnet_dist": response.get("avg_segnet_dist"),
            "avg_posenet_dist": response.get("avg_posenet_dist"),
            "cache_cleanup": {
                "schema": "hinerv_receiver_raw_cache_prefilter_cleanup.v1",
                "cleanup_policy": "delete_certified_rebuildable_cache_arrays_leave_manifests",
                "reference_cache_cleanup": reference_cleanup,
                "candidate_cache_cleanup": candidate_cleanup,
                **FALSE_AUTHORITY,
            },
            "blockers": [
                "mlx_local_replay_not_contest_auth_axis",
                "hinerv_receiver_raw_cache_prefilter_false_authority",
            ],
            **FALSE_AUTHORITY,
        }
    except Exception as exc:
        failure = {
            "schema": "hinerv_receiver_raw_cache_prefilter_failure.v1",
            "requested": True,
            "receiver_output_path": raw_path.as_posix(),
            "failure": repr(exc),
            "blockers": ["hinerv_receiver_raw_cache_prefilter_failed"],
            **FALSE_AUTHORITY,
        }
        write_json_artifact(profile_path, failure)
        return {
            **base,
            "written": False,
            "profile_schema": failure["schema"],
            "profile_path": profile_path.as_posix(),
            "profile_sha256": sha256_file(profile_path),
            "failure": repr(exc),
            "blockers": ["hinerv_receiver_raw_cache_prefilter_failed"],
            **FALSE_AUTHORITY,
        }


def _ensure_video_scorer_cache(
    video_path: Path,
    output_dir: Path,
    *,
    pair_count: int,
    batch_pairs: int,
) -> dict[str, Any]:
    return _ensure_scorer_cache(
        output_dir,
        writer=lambda: write_scorer_input_cache_from_video_file(
            video_path,
            output_dir,
            max_pairs=int(pair_count),
            batch_pairs=int(batch_pairs),
        ),
    )


def _ensure_raw_scorer_cache(
    raw_path: Path,
    output_dir: Path,
    *,
    archive_sha256: str,
    inflated_outputs_aggregate_sha256: str | None,
    pair_count: int,
    batch_pairs: int,
) -> dict[str, Any]:
    return _ensure_scorer_cache(
        output_dir,
        writer=lambda: write_scorer_input_cache_from_raw_file(
            raw_path,
            output_dir,
            archive_sha256=archive_sha256,
            inflated_outputs_aggregate_sha256=inflated_outputs_aggregate_sha256,
            max_pairs=int(pair_count),
            batch_pairs=int(batch_pairs),
        ),
    )


def _ensure_scorer_cache(
    output_dir: Path,
    *,
    writer: Any,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    cache_files = (
        output_dir / "segnet_last_rgb.npy",
        output_dir / "posenet_yuv6_pair.npy",
        output_dir / "pair_indices.npy",
    )
    if manifest_path.is_file():
        return _read_json(manifest_path)
    existing = [path.as_posix() for path in cache_files if path.exists()]
    if existing:
        raise ValueError(
            "refusing partial scorer-input cache without manifest: "
            + ", ".join(existing)
        )
    return writer()


def _cleanup_rebuildable_scorer_cache_arrays(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    cleanup = {
        "schema": "hinerv_checkpoint_scorer_cache_array_cleanup.v1",
        "cache_dir": output_dir.as_posix(),
        "reason": str(reason),
        "deleted_files": [],
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        cleanup["blockers"] = ["scorer_cache_cleanup_manifest_artifacts_missing"]
        return cleanup
    deleted: list[dict[str, Any]] = []
    blockers: list[str] = []
    for key, record in artifacts.items():
        if not isinstance(record, dict):
            blockers.append(f"scorer_cache_cleanup_artifact_record_invalid:{key}")
            continue
        path_value = record.get("path")
        path = Path(str(path_value)).expanduser().resolve(strict=False) if path_value else None
        if path is None or path.suffix != ".npy" or not path.is_file():
            continue
        expected_sha = str(record.get("sha256") or "")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            blockers.append(f"scorer_cache_cleanup_sha256_mismatch:{key}")
            continue
        file_record = {
            "artifact_id": str(key),
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "delete_certified_rebuildable": True,
        }
        path.unlink()
        deleted.append(file_record)
    cleanup["deleted_files"] = deleted
    cleanup["blockers"] = blockers
    return cleanup


def _resolve_source_video_path(
    *,
    explicit: str | Path | None,
    startup: dict[str, Any],
    command_args: dict[str, Any],
    repo_root: str | Path,
) -> Path:
    value = (
        explicit
        or startup.get("source_video_path")
        or command_args.get("source_video_path")
        or Path(repo_root) / "upstream" / "videos" / "0.mkv"
    )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve(strict=False)


def _resolve_scorer_upstream_dir(
    *,
    explicit: str | Path | None,
    command_args: dict[str, Any],
    repo_root: str | Path,
) -> Path:
    value = explicit or command_args.get("scorer_upstream_dir") or Path(repo_root) / "upstream"
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve(strict=False)


def _resolve_receiver_raw_path(value: Any, *, repo_root: str | Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve(strict=False)


def _hinerv_source_pair_indices(
    *,
    candidate: dict[str, Any],
    command_args: dict[str, Any],
) -> tuple[int, ...]:
    raw = (
        candidate.get("source_pair_indices")
        or command_args.get("source_pair_indices")
        or command_args.get("prioritized_pair_indices")
    )
    if raw is None or raw == "":
        n_pairs = int(candidate.get("num_pairs") or command_args.get("num_pairs") or 0)
        return tuple(range(n_pairs))
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
        return tuple(int(value) for value in values)
    if isinstance(raw, list | tuple):
        return tuple(int(value) for value in raw)
    raise ValueError(f"unsupported HiNeRV source_pair_indices payload: {raw!r}")


def _canonical_mlx_prefilter_device(value: Any) -> str:
    device = str(value or "cpu").strip().lower()
    aliases = {
        "": "cpu",
        "cpu": "cpu",
        "gpu": "gpu",
        "metal": "gpu",
        "mps": "gpu",
        "mlx-gpu": "gpu",
    }
    if device not in aliases:
        raise ValueError(
            "mlx prefilter scorer device must be one of "
            "cpu, gpu, metal, or mps; got "
            f"{value!r}"
        )
    return aliases[device]


def _canonical_mlx_prefilter_batch_pairs(value: Any) -> dict[str, Any]:
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = 1
    requested = max(1, requested)
    effective = 1
    normalized = requested != effective
    return {
        "schema": "mlx_prefilter_batch_pairs_control.v1",
        "requested_batch_pairs": requested,
        "effective_batch_pairs": effective,
        "normalized_to_singleton": normalized,
        "reason": (
            "production_mlx_scorer_response_uses_singleton_batches_after_"
            "recorded_segnet_batch_shape_drift"
            if normalized
            else "production_mlx_scorer_response_singleton_batch"
        ),
        **FALSE_AUTHORITY,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"startup JSON missing object {name!r}")
    return dict(value)


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": report["schema"],
        "candidate_id": report.get("candidate_id"),
        "checkpoint_epoch": report.get("checkpoint_epoch"),
        "checkpoint_state_kind": report.get("checkpoint_state_kind"),
        "archive_path": report.get("archive_path"),
        "archive_bytes": report.get("archive_bytes"),
        "hard_byte_ceiling_enforced_by_export": report.get(
            "hard_byte_ceiling_enforced_by_export"
        ),
        "hard_byte_ceiling_requested_by_candidate_or_startup": report.get(
            "hard_byte_ceiling_requested_by_candidate_or_startup"
        ),
        "hard_byte_ceiling_measurement_bypass_enabled": report.get(
            "hard_byte_ceiling_measurement_bypass_enabled"
        ),
        "rate_byte_profile": report.get("rate_byte_profile"),
        "receiver_proof_ready": report.get("receiver_proof_ready"),
        "blockers": report.get("blockers"),
        "score_claim": report.get("score_claim"),
        "ready_for_exact_eval_dispatch": report.get("ready_for_exact_eval_dispatch"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
