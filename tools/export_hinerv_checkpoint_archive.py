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
from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive  # noqa: E402
from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX  # noqa: E402

FALSE_AUTHORITY = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


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
    adapter.import_state_dict(model, state_path)
    resolved_decoder_codec = str(
        decoder_codec
        or command_args.get("compact_decoder_codec")
        or candidate.get("decoder_codec")
        or "portfolio_auto"
    )
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
        "startup_json_path": startup_path.as_posix(),
        "startup_json_sha256": sha256_file(startup_path),
        "output_dir": out.as_posix(),
        "archive_path": Path(archive_path).as_posix(),
        "archive_sha256": archive_sha256,
        "archive_bytes": int(archive_bytes),
        "rate_byte_profile": section_profile,
        "hard_byte_ceilings": list(startup.get("hard_byte_ceilings") or []),
        "decoder_codec": resolved_decoder_codec,
        "decoder_codec_resolution": {
            "explicit_arg": decoder_codec,
            "runner_compact_decoder_codec": command_args.get("compact_decoder_codec"),
            "modelsize_candidate_decoder_codec": candidate.get("decoder_codec"),
            "resolved": resolved_decoder_codec,
        },
        "latent_codec": resolved_latent_codec,
        "receiver_proof_path": receiver_proof_path.as_posix() if receiver_proof_path.is_file() else None,
        "receiver_proof_sha256": sha256_file(receiver_proof_path) if receiver_proof_path.is_file() else None,
        "receiver_proof_ready": bool(receiver_proof.get("runtime_consumption_proof_ready")),
        "blockers": _blockers(
            archive_bytes=int(archive_bytes),
            hard_byte_ceilings=startup.get("hard_byte_ceilings") or [],
            receiver_proof=receiver_proof,
            receiver_proof_requested=bool(emit_receiver_proof),
        ),
        **FALSE_AUTHORITY,
    }
    return report


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
    return blockers


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
