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

from tac.analysis.mlx_cache_quality_gate import build_mlx_cache_quality_gate  # noqa: E402
from tac.analysis.nerv_modelsize_budget import build_hinerv_config_from_size_knobs  # noqa: E402
from tac.local_acceleration.mlx_preprocess import (  # noqa: E402
    CAMERA_HW,
    recover_scorer_input_cache_manifest_from_existing_arrays,
    write_scorer_input_cache_from_raw_file,
    write_scorer_input_cache_from_video_file,
)
from tac.local_acceleration.mlx_scorer_response import (  # noqa: E402
    MANIFEST_CACHE_INTEGRITY_MODE,
    attach_cache_quality_gate_to_mlx_scorer_response,
    build_mlx_scorer_response_payload,
)
from tac.repo_io import sha256_file, write_json_artifact  # noqa: E402
from tac.substrates._shared.mlx_score_aware import RendererBundle  # noqa: E402
from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter  # noqa: E402
from tac.substrates._shared.numpy_portable_inflate import unpack_state_dict_numpy  # noqa: E402
from tac.substrates.hi_nerv.archive import build_archive_section_telemetry  # noqa: E402
from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive  # noqa: E402
from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX  # noqa: E402

FALSE_AUTHORITY = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
HINERV_CHECKPOINT_EXPORT_DEFAULT_DECODER_CODEC = "portfolio_auto"
HINERV_CHECKPOINT_FIT_SCALE_GUARD_SCHEMA = "hinerv_checkpoint_fit_scale_guard.v1"
HINERV_CHECKPOINT_FIT_SCALE_MAX_LAST_FRAME_MAE = 64.0
HINERV_CHECKPOINT_FIT_SCALE_MAX_MEAN_DELTA = 32.0
HINERV_CHECKPOINT_FIT_SCALE_MAX_STD_DELTA = 64.0
HINERV_CHECKPOINT_FIT_SCALE_MAX_POSENET_YUV6_MAE = 64.0
HINERV_CHECKPOINT_FIT_SCALE_MAX_POSENET_YUV6_MEAN_DELTA = 32.0
HINERV_CHECKPOINT_FIT_SCALE_MAX_POSENET_YUV6_STD_DELTA = 64.0
HINERV_SCORER_RGB_HW = (384, 512)


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
    receiver_fit_scale_guard = _build_checkpoint_receiver_fit_scale_guard(
        model,
        cfg=cfg,
        source_video_path=source_video_path or command_args.get("source_video_path"),
    )
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
    receiver_proof_ready = bool(receiver_proof.get("runtime_consumption_proof_ready"))
    receiver_proof_passed = bool(receiver_proof.get("runtime_consumption_proof_passed"))
    receiver_contract_satisfied = bool(receiver_proof.get("receiver_contract_satisfied"))
    receiver_closed = bool(
        receiver_proof_ready
        and receiver_proof_passed
        and receiver_contract_satisfied
        and not receiver_proof.get("blockers")
    )
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
        "receiver_fit_scale_guard": receiver_fit_scale_guard,
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
        "receiver_proof_passed": receiver_proof_passed,
        "receiver_contract_satisfied": receiver_contract_satisfied,
        "receiver_closed": receiver_closed,
        "local_mlx_prefilter_profile": pending_mlx_prefilter_profile,
        "local_mlx_prefilter_profile_path": None,
        "local_mlx_prefilter_written": False,
        "report_path": report_path.as_posix(),
        "modelsize_byte_cap_feedback_row": _modelsize_byte_cap_feedback_row(
            candidate=candidate,
            archive_bytes=int(archive_bytes),
            hard_byte_ceilings=startup.get("hard_byte_ceilings") or [],
            decoder_codec=resolved_decoder_codec,
            latent_codec=resolved_latent_codec,
            archive_section_telemetry=section_profile.get(
                "archive_section_telemetry"
            ),
            receiver_proof_ready=receiver_proof_ready,
            receiver_proof_passed=receiver_proof_passed,
            receiver_contract_satisfied=receiver_contract_satisfied,
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
            receiver_fit_scale_guard=receiver_fit_scale_guard,
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
        "receiver_fit_scale_guard": receiver_fit_scale_guard,
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
        "receiver_proof_ready": receiver_proof_ready,
        "receiver_proof_passed": receiver_proof_passed,
        "receiver_contract_satisfied": receiver_contract_satisfied,
        "receiver_closed": receiver_closed,
        "local_mlx_prefilter_profile": mlx_prefilter_profile,
        "local_mlx_prefilter_profile_path": mlx_prefilter_profile.get("profile_path"),
        "local_mlx_prefilter_written": mlx_prefilter_profile.get("written") is True,
        "report_path": report_path.as_posix(),
        "modelsize_byte_cap_feedback_row": _modelsize_byte_cap_feedback_row(
            candidate=candidate,
            archive_bytes=int(archive_bytes),
            hard_byte_ceilings=startup.get("hard_byte_ceilings") or [],
            decoder_codec=resolved_decoder_codec,
            latent_codec=resolved_latent_codec,
            archive_section_telemetry=section_profile.get(
                "archive_section_telemetry"
            ),
            receiver_proof_ready=receiver_proof_ready,
            receiver_proof_passed=receiver_proof_passed,
            receiver_contract_satisfied=receiver_contract_satisfied,
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
            receiver_fit_scale_guard=receiver_fit_scale_guard,
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
    latent_codec: str,
    archive_section_telemetry: Any,
    receiver_proof_ready: bool,
    receiver_proof_passed: bool,
    receiver_contract_satisfied: bool,
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
    receiver_closed = bool(
        receiver_proof_ready and receiver_proof_passed and receiver_contract_satisfied
    )
    return {
        "schema": "nerv_modelsize_byte_cap_feedback_row.v1",
        "family": "hi_nerv",
        "candidate_id": candidate.get("candidate_id"),
        "codec": str(decoder_codec),
        "decoder_codec": str(decoder_codec),
        "latent_codec": str(latent_codec),
        "archive_section_telemetry": archive_section_telemetry,
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
        "receiver_proof_passed": bool(receiver_proof_passed),
        "receiver_contract_satisfied": bool(receiver_contract_satisfied),
        "receiver_closed": receiver_closed,
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


def _sample_fit_scale_guard_pair_indices(
    *,
    num_pairs: int,
    max_pair_samples: int = 6,
) -> list[int]:
    total = int(num_pairs)
    count = min(max(1, int(max_pair_samples)), total)
    if count == 1:
        return [0]
    return [
        round(index * (total - 1) / float(count - 1))
        for index in range(count)
    ]


def _build_checkpoint_receiver_fit_scale_guard(
    model: Any,
    *,
    cfg: Any,
    source_video_path: str | Path | None,
    max_pair_samples: int = 6,
) -> dict[str, Any]:
    """Cheap sampled fit/domain guard for HiNeRV checkpoint exports.

    The receiver proof proves byte/runtime consumption, not that the trained
    checkpoint is still in the same RGB domain as the reference video. This
    guard samples the loaded MLX checkpoint before archive packing and compares
    its byte-range frame-1 statistics against the decoded source-video targets.
    """

    source = (
        Path(source_video_path).expanduser().resolve(strict=False)
        if source_video_path is not None
        else None
    )
    base: dict[str, Any] = {
        "schema": HINERV_CHECKPOINT_FIT_SCALE_GUARD_SCHEMA,
        "guard_kind": "sampled_loaded_checkpoint_vs_source_video_rgb_stats",
        "guard_ready": False,
        "gate_passed": False,
        "source_video_path": source.as_posix() if source is not None else None,
        "thresholds": {
            "max_last_frame_mae": HINERV_CHECKPOINT_FIT_SCALE_MAX_LAST_FRAME_MAE,
            "max_last_frame_mean_abs_delta": (
                HINERV_CHECKPOINT_FIT_SCALE_MAX_MEAN_DELTA
            ),
            "max_last_frame_std_abs_delta": (
                HINERV_CHECKPOINT_FIT_SCALE_MAX_STD_DELTA
            ),
            "max_posenet_yuv6_pair_mae": HINERV_CHECKPOINT_FIT_SCALE_MAX_POSENET_YUV6_MAE,
            "max_posenet_yuv6_pair_mean_abs_delta": (
                HINERV_CHECKPOINT_FIT_SCALE_MAX_POSENET_YUV6_MEAN_DELTA
            ),
            "max_posenet_yuv6_pair_std_abs_delta": (
                HINERV_CHECKPOINT_FIT_SCALE_MAX_POSENET_YUV6_STD_DELTA
            ),
        },
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    if source is None or not source.is_file():
        return {
            **base,
            "guard_status": "not_run_source_video_missing",
            "reason": "source_video_path not supplied or not readable",
        }
    try:
        import mlx.core as mx
        import numpy as np

        from tac.substrates._shared.mlx_score_aware.targets import (
            decode_mlx_targets,
        )

        pair_indices = _sample_fit_scale_guard_pair_indices(
            num_pairs=int(cfg.num_pairs),
            max_pair_samples=int(max_pair_samples),
        )
        idx = mx.array(pair_indices, dtype=mx.int32)
        candidate_b2chw_255 = model(idx)
        mx.eval(candidate_b2chw_255)
        candidate = np.asarray(candidate_b2chw_255, dtype=np.float32).transpose(
            0,
            1,
            3,
            4,
            2,
        )
        target_0, target_1 = decode_mlx_targets(
            source,
            num_pairs=len(pair_indices),
            output_height=int(cfg.output_height),
            output_width=int(cfg.output_width),
            pair_indices=pair_indices,
        )
        reference = (
            np.stack(
                [
                    np.asarray(target_0, dtype=np.float32),
                    np.asarray(target_1, dtype=np.float32),
                ],
                axis=1,
            )
            * 255.0
        )
        candidate_last = candidate[:, 1]
        reference_last = reference[:, 1]
        last_frame_mae = float(np.mean(np.abs(candidate_last - reference_last)))
        candidate_yuv6 = _rgb_pairs_nhwc255_to_posenet_yuv6_pair(candidate)
        reference_yuv6 = _rgb_pairs_nhwc255_to_posenet_yuv6_pair(reference)
        yuv6_pair_mae = float(np.mean(np.abs(candidate_yuv6 - reference_yuv6)))
        candidate_mean = float(np.mean(candidate_last))
        candidate_std = float(np.std(candidate_last))
        reference_mean = float(np.mean(reference_last))
        reference_std = float(np.std(reference_last))
        mean_delta = abs(candidate_mean - reference_mean)
        std_delta = abs(candidate_std - reference_std)
        candidate_yuv6_mean = float(np.mean(candidate_yuv6))
        candidate_yuv6_std = float(np.std(candidate_yuv6))
        reference_yuv6_mean = float(np.mean(reference_yuv6))
        reference_yuv6_std = float(np.std(reference_yuv6))
        yuv6_mean_delta = abs(candidate_yuv6_mean - reference_yuv6_mean)
        yuv6_std_delta = abs(candidate_yuv6_std - reference_yuv6_std)
        yuv6_saturation_fraction = float(
            np.mean((candidate_yuv6 <= 1.0) | (candidate_yuv6 >= 254.0))
        )
        saturation_fraction = float(
            np.mean((candidate_last <= 1.0) | (candidate_last >= 254.0))
        )
        gate_passed = (
            last_frame_mae <= HINERV_CHECKPOINT_FIT_SCALE_MAX_LAST_FRAME_MAE
            and mean_delta <= HINERV_CHECKPOINT_FIT_SCALE_MAX_MEAN_DELTA
            and std_delta <= HINERV_CHECKPOINT_FIT_SCALE_MAX_STD_DELTA
            and yuv6_pair_mae <= HINERV_CHECKPOINT_FIT_SCALE_MAX_POSENET_YUV6_MAE
            and yuv6_mean_delta
            <= HINERV_CHECKPOINT_FIT_SCALE_MAX_POSENET_YUV6_MEAN_DELTA
            and yuv6_std_delta
            <= HINERV_CHECKPOINT_FIT_SCALE_MAX_POSENET_YUV6_STD_DELTA
        )
        blockers = [] if gate_passed else ["hinerv_checkpoint_fit_scale_gate_failed"]
        return {
            **base,
            "guard_ready": True,
            "gate_passed": bool(gate_passed),
            "guard_status": "passed" if gate_passed else "failed",
            "pair_indices": pair_indices,
            "sampled_pair_count": len(pair_indices),
            "candidate_last_rgb_stats": {
                "mean": candidate_mean,
                "std": candidate_std,
                "min": float(np.min(candidate_last)),
                "max": float(np.max(candidate_last)),
                "saturation_fraction_le1_or_ge254": saturation_fraction,
            },
            "candidate_posenet_yuv6_pair_stats": {
                "mean": candidate_yuv6_mean,
                "std": candidate_yuv6_std,
                "min": float(np.min(candidate_yuv6)),
                "max": float(np.max(candidate_yuv6)),
                "saturation_fraction_le1_or_ge254": yuv6_saturation_fraction,
                "shape": [int(v) for v in candidate_yuv6.shape],
                "channel_order_12": [
                    "frame0_y00",
                    "frame0_y10",
                    "frame0_y01",
                    "frame0_y11",
                    "frame0_U",
                    "frame0_V",
                    "frame1_y00",
                    "frame1_y10",
                    "frame1_y01",
                    "frame1_y11",
                    "frame1_U",
                    "frame1_V",
                ],
            },
            "reference_last_rgb_stats": {
                "mean": reference_mean,
                "std": reference_std,
                "min": float(np.min(reference_last)),
                "max": float(np.max(reference_last)),
            },
            "reference_posenet_yuv6_pair_stats": {
                "mean": reference_yuv6_mean,
                "std": reference_yuv6_std,
                "min": float(np.min(reference_yuv6)),
                "max": float(np.max(reference_yuv6)),
                "shape": [int(v) for v in reference_yuv6.shape],
            },
            "fit_distance": {
                "last_frame_rgb_mae": last_frame_mae,
                "last_frame_mean_abs_delta": mean_delta,
                "last_frame_std_abs_delta": std_delta,
                "posenet_yuv6_pair_mae": yuv6_pair_mae,
                "posenet_yuv6_pair_mean_abs_delta": yuv6_mean_delta,
                "posenet_yuv6_pair_std_abs_delta": yuv6_std_delta,
            },
            "blockers": blockers,
        }
    except Exception as exc:  # pragma: no cover - depends on local MLX/video stack.
        return {
            **base,
            "guard_status": "execution_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "blockers": ["hinerv_checkpoint_fit_scale_guard_execution_failed"],
        }


def _resize_rgb_nhwc255_for_scorer(rgb: Any) -> Any:
    import numpy as np

    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected NHWC RGB tensor, got shape {arr.shape}")
    target_h, target_w = HINERV_SCORER_RGB_HW
    if arr.shape[1:3] == (target_h, target_w):
        return arr
    from PIL import Image

    out = np.empty((arr.shape[0], target_h, target_w, 3), dtype=np.float32)
    for idx, frame in enumerate(arr):
        frame_u8 = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
        resized = Image.fromarray(frame_u8, mode="RGB").resize(
            (target_w, target_h),
            Image.Resampling.BILINEAR,
        )
        out[idx] = np.asarray(resized, dtype=np.float32)
    return out


def _rgb_pairs_nhwc255_to_posenet_yuv6_pair(rgb_pairs: Any) -> Any:
    """Convert sampled RGB pairs to upstream PoseNet's 12-channel YUV6 cache."""

    import numpy as np

    from tac.framework_agnostic.backend import Backend
    from tac.framework_agnostic.canonical_kernels import rgb_to_yuv6

    pairs = np.asarray(rgb_pairs, dtype=np.float32)
    if pairs.ndim != 5 or pairs.shape[1] != 2 or pairs.shape[-1] != 3:
        raise ValueError(
            "expected RGB pairs with shape (pairs, 2, H, W, 3); "
            f"got {pairs.shape}"
        )
    flat = pairs.reshape((-1, pairs.shape[2], pairs.shape[3], 3))
    flat = _resize_rgb_nhwc255_for_scorer(flat)
    nchw = np.transpose(flat, (0, 3, 1, 2))
    yuv6 = rgb_to_yuv6(nchw, backend=Backend.NUMPY, value_range=255.0)
    yuv6 = np.asarray(yuv6, dtype=np.float32)
    return yuv6.reshape((pairs.shape[0], 2, 6, yuv6.shape[-2], yuv6.shape[-1])).reshape(
        pairs.shape[0],
        12,
        yuv6.shape[-2],
        yuv6.shape[-1],
    )


def _blockers(
    *,
    archive_bytes: int,
    hard_byte_ceilings: Any,
    hard_byte_ceiling: int | None = None,
    receiver_proof: dict[str, Any],
    receiver_proof_requested: bool,
    modelsize_integrity: dict[str, Any],
    decoder_codec_resolution: dict[str, Any],
    receiver_fit_scale_guard: dict[str, Any] | None = None,
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
        cache_quality_gate = mlx_prefilter_profile.get("cache_quality_gate")
        if not isinstance(cache_quality_gate, dict):
            blockers.append("hinerv_receiver_raw_cache_quality_gate_missing")
            blockers.append("mlx_scorer_response_cache_quality_gate_failed")
        else:
            if cache_quality_gate.get("fit_gate_passed") is not True:
                blockers.append("mlx_scorer_response_cache_quality_gate_failed")
            if cache_quality_gate.get("candidate_cache_nondegenerate") is not True:
                blockers.append("mlx_scorer_response_candidate_cache_degenerate")
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
    else:
        if receiver_proof.get("runtime_consumption_proof_ready") is not True:
            blockers.append("receiver_proof_not_ready")
        if receiver_proof.get("runtime_consumption_proof_passed") is not True:
            blockers.append("runtime_consumption_proof_not_passed")
        if receiver_proof.get("receiver_contract_satisfied") is not True:
            blockers.append("receiver_contract_not_satisfied")
        if receiver_proof.get("blockers"):
            blockers.append("receiver_proof_blockers_present")
    blockers.extend(str(v) for v in modelsize_integrity.get("blockers") or [])
    guard = dict(receiver_fit_scale_guard or {})
    blockers.extend(str(v) for v in guard.get("blockers") or [])
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
    telemetry_path = output_dir / "hi_nerv_archive_section_telemetry.json"
    if telemetry_path.is_file():
        telemetry = _read_json(telemetry_path)
        telemetry_sha256 = sha256_file(telemetry_path)
    else:
        bin_path = output_dir / "0.bin"
        if bin_path.is_file():
            telemetry = build_archive_section_telemetry(
                bin_path.read_bytes(),
                archive_zip_bytes=int(archive_bytes),
            )
            telemetry_sha256 = None
        else:
            telemetry = None
            telemetry_sha256 = None
    if isinstance(telemetry, dict):
        sections = []
        for section in telemetry.get("sections") or []:
            if not isinstance(section, dict):
                continue
            byte_count = int(section.get("bytes") or 0)
            sections.append(
                {
                    "name": str(section.get("name") or ""),
                    "role": str(section.get("role") or ""),
                    "bytes": byte_count,
                    "archive_fraction": (
                        byte_count / float(archive_bytes)
                        if archive_bytes > 0
                        else None
                    ),
                    "sha256": section.get("sha256"),
                    "offset": section.get("offset"),
                    "end_offset": section.get("end_offset"),
                    "codec": section.get("codec"),
                    "scale": section.get("scale"),
                    "raw_bytes": section.get("raw_bytes"),
                    "coded_to_raw_ratio": section.get("coded_to_raw_ratio"),
                }
            )
        section_bytes = sum(int(section["bytes"]) for section in sections)
        overhead_bytes = int(archive_bytes) - int(
            telemetry.get("inner_payload_bytes") or section_bytes
        )
        return {
            "schema": "hinerv_checkpoint_archive_rate_byte_profile.v1",
            "profile_ready": bool(sections),
            "profile_source": "hiv1_archive_section_telemetry",
            "archive_section_telemetry_path": (
                telemetry_path.as_posix() if telemetry_path.is_file() else None
            ),
            "archive_section_telemetry_sha256": telemetry_sha256,
            "archive_section_telemetry": telemetry,
            "hprc_bin_bytes": int(
                telemetry.get("hprc_bin_bytes")
                or telemetry.get("inner_payload_bytes")
                or section_bytes
            ),
            "archive_bytes": int(archive_bytes),
            "section_payload_bytes": int(section_bytes),
            "archive_overhead_and_manifest_bytes": int(overhead_bytes),
            "sections": sections,
            "dominant_sections": sorted(
                sections,
                key=lambda row: int(row["bytes"]),
                reverse=True,
            )[:4],
            "blockers": [] if sections else ["hiv1_archive_sections_missing"],
        }

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
        cache_quality_gate_path = out / "cache_quality_gate.json"
        cache_quality_gate = _build_receiver_raw_cache_quality_gate(
            candidate_cache_dir=candidate_cache_dir,
            reference_cache_dir=reference_cache_dir,
            output_path=cache_quality_gate_path,
            pair_count=n_pairs,
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
        response = attach_cache_quality_gate_to_mlx_scorer_response(
            response,
            cache_quality_gate,
            source_path=cache_quality_gate_path,
        )
        response_blockers = [
            str(blocker)
            for blocker in response.get("blockers") or ()
            if str(blocker)
        ]
        if not isinstance(response.get("cache_quality_gate"), dict):
            response_blockers.extend(
                [
                    "hinerv_receiver_raw_cache_quality_gate_missing",
                    "mlx_scorer_response_cache_quality_gate_failed",
                ]
            )
        response = {
            **response,
            "blockers": list(dict.fromkeys(response_blockers)),
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
                "cache_quality_gate": response.get("cache_quality_gate"),
                "cache_quality_gate_path": cache_quality_gate_path.as_posix(),
                "cache_quality_gate_sha256": sha256_file(cache_quality_gate_path),
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
            "cache_quality_gate": response.get("cache_quality_gate"),
            "cache_quality_gate_path": cache_quality_gate_path.as_posix(),
            "cache_quality_gate_sha256": sha256_file(cache_quality_gate_path),
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
                *[
                    str(blocker)
                    for blocker in response.get("blockers") or ()
                    if str(blocker)
                ],
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


def _build_receiver_raw_cache_quality_gate(
    *,
    candidate_cache_dir: Path,
    reference_cache_dir: Path,
    output_path: Path,
    pair_count: int,
) -> dict[str, Any]:
    sample_pairs = max(1, min(32, int(pair_count)))
    try:
        gate = build_mlx_cache_quality_gate(
            candidate_cache_dir=candidate_cache_dir,
            reference_cache_dir=reference_cache_dir,
            sample_pairs=sample_pairs,
        )
    except Exception as exc:
        gate = {
            "schema": "mlx_cache_quality_gate.v1",
            "verdict": "CACHE_QUALITY_GATE_FAILED",
            "candidate_cache_nondegenerate": None,
            "fit_gate_passed": False,
            "sample_pairs": sample_pairs,
            "candidate_cache_dir": candidate_cache_dir.as_posix(),
            "reference_cache_dir": reference_cache_dir.as_posix(),
            "failure": repr(exc),
            "blockers": [
                "mlx_cache_quality_gate_failed",
                f"mlx_cache_quality_gate_exception:{type(exc).__name__}",
            ],
            "recommended_next_actions": [
                "preserve_receiver_proof_but_block_exact_gate_until_cache_quality_gate_reruns"
            ],
            **FALSE_AUTHORITY,
        }
    write_json_artifact(output_path, gate)
    return gate


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
        recoverer=lambda: recover_scorer_input_cache_manifest_from_existing_arrays(
            output_dir,
            source=video_path,
            source_kind="video",
            frame_shape_hwc=(*CAMERA_HW, 3),
            streaming_batch_pairs=None,
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
        recoverer=lambda: recover_scorer_input_cache_manifest_from_existing_arrays(
            output_dir,
            source=raw_path,
            source_kind="raw",
            archive_sha256=archive_sha256,
            inflated_outputs_aggregate_sha256=inflated_outputs_aggregate_sha256,
            raw_sha256=sha256_file(raw_path),
            frame_shape_hwc=(*CAMERA_HW, 3),
            streaming_batch_pairs=int(batch_pairs),
        ),
    )


def _ensure_scorer_cache(
    output_dir: Path,
    *,
    writer: Any,
    recoverer: Any,
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
        if len(existing) == len(cache_files):
            return recoverer()
        raise ValueError(
            "refusing incomplete scorer-input cache without manifest: "
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
        "receiver_proof_passed": report.get("receiver_proof_passed"),
        "receiver_contract_satisfied": report.get("receiver_contract_satisfied"),
        "receiver_closed": report.get("receiver_closed"),
        "blockers": report.get("blockers"),
        "score_claim": report.get("score_claim"),
        "ready_for_exact_eval_dispatch": report.get("ready_for_exact_eval_dispatch"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
