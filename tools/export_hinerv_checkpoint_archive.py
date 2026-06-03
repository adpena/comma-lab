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
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_modelsize_budget import build_hinerv_config_from_size_knobs  # noqa: E402
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
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--output-json", default=None, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = export_checkpoint_archive(
        startup_json=args.startup_json,
        checkpoint_meta=args.checkpoint_meta,
        output_dir=args.output_dir,
        state_kind=args.state_kind,
        decoder_codec=args.decoder_codec,
        latent_codec=args.latent_codec,
        emit_receiver_proof=bool(args.emit_receiver_proof),
        retain_receiver_proof_output=bool(args.retain_receiver_proof_output),
        repo_root=args.repo_root,
    )
    output_json = args.output_json or args.output_dir / "hinerv_checkpoint_archive_export.json"
    report["report_path"] = output_json.expanduser().resolve(strict=False).as_posix()
    write_json_artifact(output_json, report)
    print(json.dumps(_summary(report), indent=2, sort_keys=True))
    return 0


def export_checkpoint_archive(
    *,
    startup_json: str | Path,
    checkpoint_meta: str | Path,
    output_dir: str | Path,
    state_kind: str = "ema",
    decoder_codec: str | None = None,
    latent_codec: str | None = None,
    emit_receiver_proof: bool = False,
    retain_receiver_proof_output: bool = False,
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
    )
    receiver_proof_path = out / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json"
    receiver_proof = _read_json(receiver_proof_path) if receiver_proof_path.is_file() else {}
    section_profile = _section_profile(out, archive_bytes=int(archive_bytes))
    report = {
        "schema": "hinerv_checkpoint_archive_export.v1",
        "family": "hi_nerv",
        "candidate_id": candidate.get("candidate_id"),
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
        "rate_byte_profile": section_profile,
        "hard_byte_ceilings": list(startup.get("hard_byte_ceilings") or []),
        "decoder_codec": resolved_decoder_codec,
        "decoder_codec_resolution": decoder_codec_resolution,
        "latent_codec": resolved_latent_codec,
        "receiver_proof_path": receiver_proof_path.as_posix() if receiver_proof_path.is_file() else None,
        "receiver_proof_sha256": sha256_file(receiver_proof_path) if receiver_proof_path.is_file() else None,
        "receiver_proof_ready": bool(receiver_proof.get("runtime_consumption_proof_ready")),
        "blockers": _blockers(
            archive_bytes=int(archive_bytes),
            hard_byte_ceilings=startup.get("hard_byte_ceilings") or [],
            receiver_proof=receiver_proof,
            receiver_proof_requested=bool(emit_receiver_proof),
            modelsize_integrity=modelsize_integrity,
        ),
        **FALSE_AUTHORITY,
    }
    return report


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
    elif candidate_codec is not None:
        resolved = candidate_codec
        source = "modelsize_candidate_decoder_codec"
    else:
        resolved = HINERV_CHECKPOINT_EXPORT_DEFAULT_DECODER_CODEC
        source = "checkpoint_export_default"
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
    receiver_proof: dict[str, Any],
    receiver_proof_requested: bool,
    modelsize_integrity: dict[str, Any],
) -> list[str]:
    blockers = [
        "macos_mlx_checkpoint_export_false_authority",
        "contest_cpu_cuda_exact_eval_not_executed",
        "full_video_scorer_replay_not_executed",
    ]
    ceilings = [int(v) for v in hard_byte_ceilings if int(v) > 0]
    if ceilings and int(archive_bytes) > min(ceilings):
        blockers.append("archive_bytes_exceed_tightest_hard_ceiling")
    if not receiver_proof_requested:
        blockers.append("receiver_proof_not_requested")
    elif receiver_proof.get("runtime_consumption_proof_ready") is not True:
        blockers.append("receiver_proof_not_ready")
    blockers.extend(str(v) for v in modelsize_integrity.get("blockers") or [])
    return blockers


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
        "rate_byte_profile": report.get("rate_byte_profile"),
        "receiver_proof_ready": report.get("receiver_proof_ready"),
        "blockers": report.get("blockers"),
        "score_claim": report.get("score_claim"),
        "ready_for_exact_eval_dispatch": report.get("ready_for_exact_eval_dispatch"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
