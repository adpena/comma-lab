#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a compact learned HPRC receiver from low-resolution frame pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

import numpy as np  # noqa: E402

from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES, contest_rate_term, profile_archive  # noqa: E402
from tac.repo_io import sha256_file, write_json  # noqa: E402
from tac.substrates.hprc.archive import parse_hprc_packet  # noqa: E402
from tac.substrates.hprc.archive_candidate import (  # noqa: E402
    HPRC_SUB019_ZERO_DISTORTION_BYTE_CEILING,
    export_hprc_archive_bytes,
)
from tac.substrates.hprc.learned_receiver import (  # noqa: E402
    build_compact_receiver_packet_from_lowres_frames,
    compact_receiver_reconstruction_metrics,
    compact_receiver_section_byte_profile,
    compact_receiver_section_value_profile,
    decode_compact_receiver_packet,
)

DEFAULT_REFERENCE_PAIRS_NPY = (
    REPO_ROOT
    / ".omx/research/z8_full_video_mlx_vjp_live_20260531T181115Z/reference_pairs_rgb255.npy"
)
SCHEMA = "hprc_compact_receiver_candidate_materialization_result.v1"
COMPONENT_BUDGET_SCHEMA = "hprc_compact_receiver_component_budget_profile.v1"


def _resolve(path: Path, *, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def _json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _zero_distortion_byte_ceiling(target_score: float) -> int:
    return max(0, int(float(target_score) * CONTEST_ORIGINAL_BYTES / 25.0))


def _compression_gap_rows(archive_bytes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_score in (0.19, 0.18, 0.15, 0.10):
        ceiling = _zero_distortion_byte_ceiling(target_score)
        rows.append(
            {
                "target_score": target_score,
                "zero_distortion_archive_byte_ceiling": ceiling,
                "bytes_over_zero_distortion_ceiling": max(0, int(archive_bytes) - ceiling),
                "compression_factor_needed_from_current": (
                    float(int(archive_bytes) / ceiling) if ceiling else float("inf")
                ),
                "authority": "rate_only_zero_distortion_ceiling_not_score_claim",
            }
        )
    return rows


def _component_budget_profile(
    *,
    archive_bytes: int,
    archive_profile: dict[str, Any],
    section_byte_profile: dict[str, Any],
    section_value_profile: dict[str, Any],
) -> dict[str, Any]:
    value_rows = {
        str(row["section"]): row
        for row in section_value_profile.get("section_rows", [])
        if isinstance(row, dict) and "section" in row
    }
    section_rows = [
        row
        for row in section_byte_profile.get("section_rows", [])
        if isinstance(row, dict) and int(row.get("bytes", 0)) > 0
    ]
    payload_bytes = max(
        int(section_byte_profile.get("hprc_payload_section_bytes") or 0),
        sum(int(row.get("bytes", 0)) for row in section_rows),
        1,
    )
    target_ceiling = int(HPRC_SUB019_ZERO_DISTORTION_BYTE_CEILING)
    target_section_payload_budget = int(payload_bytes * target_ceiling / max(int(archive_bytes), 1))
    rows: list[dict[str, Any]] = []
    for row in section_rows:
        section = str(row["section"])
        section_bytes = int(row.get("bytes", 0))
        value = dict(value_rows.get(section, {}))
        proxy_value_per_kib = float(value.get("delta_mse_per_kib", 0.0) or 0.0)
        proportional_budget = int(section_bytes * target_section_payload_budget / payload_bytes)
        scorer_value_available = section in value_rows
        rows.append(
            {
                "section": section,
                "pipeline_stage": {
                    "decoder_qw": "pre_entropy_learned_renderer_weights",
                    "latents_rc": "pre_entropy_frame_latent_stream",
                    "selectors_rc": "pre_entropy_frame_selector_stream",
                    "residual_rc": "pre_entropy_scorer_weighted_residual_tokens",
                    "rdo_plan": "allocator_contract_metadata",
                    "receiver_state": "decode_time_temporal_state_stream",
                    "manifest_json": "custody_and_provenance_metadata",
                }.get(section, "unknown"),
                "entropy_position": {
                    "decoder_qw": "before_entropy_coder",
                    "latents_rc": "before_entropy_coder",
                    "selectors_rc": "before_entropy_coder",
                    "residual_rc": "before_entropy_coder",
                    "rdo_plan": "after_transform_metadata",
                    "receiver_state": "before_entropy_coder",
                    "manifest_json": "after_transform_metadata",
                }.get(section, "unknown"),
                "current_payload_bytes": section_bytes,
                "share_of_hprc_payload": row.get("share_of_hprc_payload"),
                "zero_distortion_sub019_proportional_budget_bytes": proportional_budget,
                "bytes_to_remove_at_proportional_budget": max(
                    0, section_bytes - proportional_budget
                ),
                "proportional_budget_basis": (
                    "current archive.zip-to-hprc-payload compression ratio; not an "
                    "optimal allocation"
                ),
                "decoder_grid_proxy_value_available": scorer_value_available,
                "decoder_grid_proxy_delta_mse_per_kib": proxy_value_per_kib,
                "decoder_grid_proxy_delta_mse": value.get("delta_mse_rgb255"),
                "allocation_status": "requires_scorer_rd_curve_and_lagrangian_lambda",
            }
        )
    rows.sort(key=lambda item: -float(item["current_payload_bytes"]))
    return {
        "schema": COMPONENT_BUDGET_SCHEMA,
        "archive_zip_bytes": int(archive_bytes),
        "contest_rate_term": contest_rate_term(int(archive_bytes)),
        "sub019_zero_distortion_archive_byte_ceiling": target_ceiling,
        "bytes_over_sub019_zero_distortion_ceiling": max(0, int(archive_bytes) - target_ceiling),
        "compression_factor_needed_for_sub019_zero_distortion": float(
            int(archive_bytes) / max(target_ceiling, 1)
        ),
        "hprc_payload_section_bytes": int(payload_bytes),
        "target_section_payload_budget_at_current_archive_compression_ratio": (
            target_section_payload_budget
        ),
        "archive_member_profile": {
            "zip_overhead_bytes": archive_profile.get("zip_overhead_bytes"),
            "members": archive_profile.get("members"),
            "profile_by_top_level": archive_profile.get("profile_by_top_level"),
        },
        "section_budget_rows": rows,
        "mathematical_authority": (
            "component profile is an allocator input; final optimal bytes require "
            "MLX/exact SegNet-PoseNet-rate RD curves and Lagrangian lambda solve"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--reference-pairs-npy", type=Path, default=DEFAULT_REFERENCE_PAIRS_NPY)
    parser.add_argument("--basis-count", type=int, default=5)
    parser.add_argument("--residual-grid-h", type=int, default=24)
    parser.add_argument("--residual-grid-w", type=int, default=32)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--retain-receiver-output", action="store_true")
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve(strict=False)
    output_dir = _resolve(args.output_dir, repo_root=repo_root)
    if output_dir.exists() and not args.allow_existing_output_dir:
        print(f"FATAL: output dir exists: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = _resolve(args.reference_pairs_npy, repo_root=repo_root)
    if not source_path.is_file():
        print(f"FATAL: missing reference pairs npy: {source_path}", file=sys.stderr)
        return 2
    source = np.load(source_path, mmap_mode="r")
    if source.ndim != 5 or source.shape[1] != 2 or source.shape[-1] != 3:
        print(
            "FATAL: reference pairs must have shape pairs x 2 x H x W x RGB; "
            f"got {source.shape}",
            file=sys.stderr,
        )
        return 2
    pair_count = int(source.shape[0])
    if args.max_pairs is not None:
        pair_count = min(pair_count, max(1, int(args.max_pairs)))
    frames = np.asarray(source[:pair_count], dtype=np.float32)
    source_manifest = {
        "schema": "hprc_compact_receiver_source_manifest.v1",
        "source_reference_pairs_npy": source_path.as_posix(),
        "source_reference_pairs_bytes": int(source_path.stat().st_size),
        "source_reference_pairs_sha256": sha256_file(source_path),
        "source_shape": [int(v) for v in source.shape],
        "materialized_pair_count": int(pair_count),
        "materialized_frame_count": int(pair_count * 2),
        "decoder_grid_height": int(source.shape[2]),
        "decoder_grid_width": int(source.shape[3]),
        "contest_output_height": 874,
        "contest_output_width": 1164,
        "basis_count": int(args.basis_count),
        "residual_grid_h": int(args.residual_grid_h),
        "residual_grid_w": int(args.residual_grid_w),
        "resolution_authority": "contest_output_raw_bytes_only_after_inflate_sh_replay",
        "score_claim": False,
        "promotion_eligible": False,
    }
    write_json(output_dir / "hprc_compact_receiver_source_manifest.json", source_manifest)
    packet = build_compact_receiver_packet_from_lowres_frames(
        frames,
        basis_count=int(args.basis_count),
        residual_grid_h=int(args.residual_grid_h),
        residual_grid_w=int(args.residual_grid_w),
        source_manifest=source_manifest,
    )
    reconstruction_metrics = compact_receiver_reconstruction_metrics(
        decode_compact_receiver_packet(parse_hprc_packet(packet)),
        frames,
    )
    compact = decode_compact_receiver_packet(parse_hprc_packet(packet))
    section_byte_profile = compact_receiver_section_byte_profile(compact.packet)
    section_value_profile = compact_receiver_section_value_profile(compact, frames)
    write_json(
        output_dir / "hprc_compact_receiver_decoder_grid_reconstruction_metrics.json",
        reconstruction_metrics,
    )
    write_json(
        output_dir / "hprc_compact_receiver_section_byte_profile.json",
        section_byte_profile,
    )
    write_json(
        output_dir / "hprc_compact_receiver_section_value_profile.json",
        section_value_profile,
    )
    archive_zip_path, archive_sha256, archive_bytes = export_hprc_archive_bytes(
        packet,
        output_dir,
        repo_root=repo_root,
        retain_receiver_proof_output=bool(args.retain_receiver_output),
        mlx_triage_argv=list(sys.argv if argv is None else [sys.argv[0], *argv]),
    )
    archive_profile = profile_archive(archive_zip_path)
    component_budget_profile = _component_budget_profile(
        archive_bytes=int(archive_bytes),
        archive_profile=archive_profile,
        section_byte_profile=section_byte_profile,
        section_value_profile=section_value_profile,
    )
    write_json(output_dir / "hprc_compact_receiver_archive_zip_profile.json", archive_profile)
    write_json(
        output_dir / "hprc_compact_receiver_component_budget_profile.json",
        component_budget_profile,
    )
    result = {
        "schema": SCHEMA,
        "archive_zip_path": archive_zip_path.as_posix(),
        "archive_zip_sha256": archive_sha256,
        "archive_zip_bytes": int(archive_bytes),
        "hprc_0bin_path": (output_dir / "0.bin").as_posix(),
        "hprc_0bin_bytes": int((output_dir / "0.bin").stat().st_size),
        "source_manifest_path": (output_dir / "hprc_compact_receiver_source_manifest.json").as_posix(),
        "decoder_grid_reconstruction_metrics_path": (
            output_dir / "hprc_compact_receiver_decoder_grid_reconstruction_metrics.json"
        ).as_posix(),
        "decoder_grid_reconstruction_metrics": reconstruction_metrics,
        "section_byte_profile_path": (
            output_dir / "hprc_compact_receiver_section_byte_profile.json"
        ).as_posix(),
        "section_byte_profile": section_byte_profile,
        "section_value_profile_path": (
            output_dir / "hprc_compact_receiver_section_value_profile.json"
        ).as_posix(),
        "section_value_profile": section_value_profile,
        "archive_zip_profile_path": (
            output_dir / "hprc_compact_receiver_archive_zip_profile.json"
        ).as_posix(),
        "component_budget_profile_path": (
            output_dir / "hprc_compact_receiver_component_budget_profile.json"
        ).as_posix(),
        "component_budget_profile": component_budget_profile,
        "zero_distortion_compression_gap_rows": _compression_gap_rows(int(archive_bytes)),
        "byte_ledger_path": (output_dir / "hprc_archive_byte_ledger.json").as_posix(),
        "receiver_proof_path": (output_dir / "receiver_proof/hprc_receiver_proof.json").as_posix(),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "exact_axis_blocker": "contest_cpu_cuda_exact_eval_not_executed",
    }
    write_json(output_dir / "hprc_compact_receiver_materialization_result.json", result)
    print(_json_text(result), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
