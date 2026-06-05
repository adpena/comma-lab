#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export a byte-closed SNeRV archive directly from an MLX checkpoint.

This is the SNeRV sister of ``tools/export_hinerv_checkpoint_archive.py``.  It
exists because long SNeRV runs can be interrupted after useful checkpoints but
before the normal terminal export path runs.  The exporter packetizes the
checkpoint's receiver-visible LF planes and decoder kernels directly; it does
not re-fit a fresh decoder from rendered pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.mlx_cache_quality_gate import build_mlx_cache_quality_gate  # noqa: E402
from tac.analysis.snerv_official_source_forward_harness import (  # noqa: E402
    build_snerv_official_trained_checkpoint_mapping_manifest,
)
from tac.analysis.snerv_step_map_coder import encode_step_maps_waterfill  # noqa: E402
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
from tac.substrates._shared.mlx_score_aware.targets import decode_mlx_targets  # noqa: E402
from tac.substrates._shared.numpy_portable_inflate import unpack_state_dict_numpy  # noqa: E402
from tac.substrates.snerv_inverse_steg_carrier.archive import (  # noqa: E402
    SnervArchivePacket,
    decode_snerv_archive_frames,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    inspect_lf_quant_payload_header,
    pack_snerv_archive,
    resolve_decoder_payload_codec,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (  # noqa: E402
    _DETAIL_KEYS,
    HfGenerationDecoder,
    SnervModelSizeConfig,
    quantize_lf,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_payload_codec import (  # noqa: E402
    selected_lf_payload_codec_label,
)
from tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export import (  # noqa: E402
    SCORER_HW,
    _build_official_mfu_hfr_tub_packet_from_components,
    _model_size_from_candidate,
    _official_passthrough_mfu,
    _official_receiver_tensor_map_from_packet,
    _selected_packet_official_payload_authority,
    _write_snerv_native_receiver_decoded_mlx_prefilter,
    export_snerv_mlx_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.official_hfr import (  # noqa: E402
    OfficialConv2dNchw,
    OfficialHfrConvBlock,
    OfficialHfrHeads,
)

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
    parser.add_argument("--lf-payload-codec", default=None)
    parser.add_argument("--emit-receiver-proof", action="store_true")
    parser.add_argument("--retain-receiver-proof-output", action="store_true")
    parser.add_argument("--receiver-proof-timeout-seconds", default=1800, type=int)
    parser.add_argument(
        "--allow-over-hard-byte-ceiling-for-measurement",
        action="store_true",
        help=(
            "Keep exporting/reporting even when measured archive.zip bytes exceed "
            "the active hard byte ceiling. The report records an over-cap blocker "
            "and remains false-authority."
        ),
    )
    parser.add_argument(
        "--write-mlx-prefilter-profile",
        action="store_true",
        help=(
            "Decode the selected SNAR1 packet through the receiver and write a "
            "false-authority full-video MLX scorer prefilter profile."
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
    report = export_snerv_checkpoint_archive(
        startup_json=args.startup_json,
        checkpoint_meta=args.checkpoint_meta,
        output_dir=args.output_dir,
        state_kind=args.state_kind,
        decoder_codec=args.decoder_codec,
        lf_payload_codec=args.lf_payload_codec,
        emit_receiver_proof=bool(args.emit_receiver_proof),
        retain_receiver_proof_output=bool(args.retain_receiver_proof_output),
        receiver_proof_timeout_seconds=int(args.receiver_proof_timeout_seconds),
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
    output_json = args.output_json or args.output_dir / "snerv_checkpoint_archive_export.json"
    report["report_path"] = output_json.expanduser().resolve(strict=False).as_posix()
    write_json_artifact(output_json, report)
    print(json.dumps(_summary(report), indent=2, sort_keys=True))
    return 0


def export_snerv_checkpoint_archive(
    *,
    startup_json: str | Path,
    checkpoint_meta: str | Path,
    output_dir: str | Path,
    state_kind: str = "ema",
    decoder_codec: str | None = None,
    lf_payload_codec: str | None = None,
    emit_receiver_proof: bool = False,
    retain_receiver_proof_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
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
    state_path = _checkpoint_state_path(meta, state_kind=state_kind)
    checkpoint_meta_sha256 = sha256_file(meta_path)
    checkpoint_state_sha256 = sha256_file(state_path)
    startup_json_sha256 = sha256_file(startup_path)
    state = unpack_state_dict_numpy(state_path.read_bytes())
    checkpoint_state_key_count = len(state)
    checkpoint_trained_state_exportable = checkpoint_state_key_count > 0
    hard_byte_ceiling = _export_hard_byte_ceiling(
        candidate=candidate,
        hard_byte_ceilings=startup.get("hard_byte_ceilings") or [],
    )

    levels = int(candidate.get("levels") or candidate.get("snerv_levels") or command_args.get("levels") or 3)
    wavelet = str(candidate.get("wavelet") or command_args.get("wavelet") or "haar")
    target_bits_per_coeff = float(
        candidate.get("bits_per_coeff")
        or candidate.get("target_bits_per_coeff")
        or command_args.get("target_bits_per_coeff")
        or 2.5
    )
    step_map_bits_per_coeff = float(
        candidate.get("step_map_bits_per_coeff")
        or candidate.get("snerv_step_map_bits_per_coeff")
        or command_args.get("step_map_waterfill_bits_per_coeff")
        or 4.0
    )
    requested_decoder_codec = str(
        decoder_codec
        or candidate.get("decoder_payload_codec")
        or command_args.get("decoder_payload_codec")
        or "mixed_magnitude_symmetric"
    )
    resolved_decoder_codec = resolve_decoder_payload_codec(requested_decoder_codec)
    resolved_lf_codec = str(
        lf_payload_codec
        or candidate.get("lf_payload_codec")
        or command_args.get("lf_payload_codec")
        or "portfolio_auto"
    )
    model_size = _model_size_from_candidate(candidate)
    metadata_extra = {
        "checkpoint_export_schema": "snerv_checkpoint_archive_export.v1",
        "checkpoint_meta_path": meta_path.as_posix(),
        "checkpoint_epoch": meta.get("global_epoch"),
        "checkpoint_state_kind": state_kind,
        "checkpoint_state_sha256": checkpoint_state_sha256,
        "checkpoint_state_key_count": checkpoint_state_key_count,
        "checkpoint_trained_state_exportable": checkpoint_trained_state_exportable,
        "startup_json_sha256": startup_json_sha256,
        "native_mlx_training_executed": True,
        "score_aware_long_training_executed": True,
        "score_aware_long_training_trained_state_exportable": (
            checkpoint_trained_state_exportable
        ),
        "score_aware_long_training": {
            "schema": "snerv_checkpoint_export_score_aware_long_training.v1",
            "executed": True,
            "trained_state_exportable": checkpoint_trained_state_exportable,
            "training_kind": "checkpoint_harvest_interrupted_run",
            "checkpoint_state_kind": state_kind,
            "checkpoint_state_key_count": checkpoint_state_key_count,
            **FALSE_AUTHORITY,
        },
        "score_aware_long_training_kind": "checkpoint_harvest_interrupted_run",
        "hard_byte_ceiling": hard_byte_ceiling,
        **FALSE_AUTHORITY,
    }
    if model_size.official_mfu_hfr_tub_numeric_primitives_requested:
        packet = build_snerv_official_checkpoint_packet(
            state,
            model_size=model_size,
            metadata_extra=metadata_extra,
        )
    else:
        packet = build_snerv_checkpoint_packet(
            state,
            levels=levels,
            wavelet=wavelet,
            target_bits_per_coeff=target_bits_per_coeff,
            step_map_bits_per_coeff=step_map_bits_per_coeff,
            decoder_payload_codec=resolved_decoder_codec,
            lf_payload_codec=resolved_lf_codec,
            model_size=model_size,
            metadata_extra={
                **metadata_extra,
                "native_mlx_training_kind": "checkpoint_direct_lf_decoder_packetization",
            },
        )
    out.mkdir(parents=True, exist_ok=True)
    packet_path = out / "snerv_checkpoint_packet.bin"
    packet_path.write_bytes(packet.packet)
    official_checkpoint_export_binding = _official_checkpoint_export_binding(
        packet,
        model_size=model_size,
        checkpoint_state=state,
    )
    package: dict[str, Any] | None = None
    if emit_receiver_proof:
        package = export_snerv_mlx_archive(
            {"packet_path": packet_path.as_posix(), "packet_sha256": _sha256_bytes(packet.packet)},
            out / "snerv_checkpoint_archive_bound_package",
            repo_root=root,
            retain_receiver_output=bool(retain_receiver_proof_output),
            receiver_proof_timeout_seconds=int(receiver_proof_timeout_seconds),
            hard_byte_ceiling=hard_byte_ceiling,
            allow_over_hard_byte_ceiling_for_measurement=bool(
                allow_over_hard_byte_ceiling_for_measurement
            ),
        )
    receiver_proof = dict(package.get("receiver_proof") or {}) if package else {}
    archive_path = receiver_proof.get("archive_path") if receiver_proof else None
    archive_bytes = receiver_proof.get("archive_bytes") if receiver_proof else None
    archive_sha256 = receiver_proof.get("archive_sha256") if receiver_proof else None
    mlx_prefilter_profile = _maybe_write_receiver_decoded_mlx_prefilter(
        requested=bool(write_mlx_prefilter_profile),
        output_dir=out / "local_mlx_prefilter",
        packet=packet,
        receiver_proof=receiver_proof,
        startup=startup,
        command_args=command_args,
        archive_bytes=int(archive_bytes) if archive_bytes is not None else None,
        archive_sha256=str(archive_sha256) if archive_sha256 else None,
        source_video_path=source_video_path,
        scorer_upstream_dir=scorer_upstream_dir,
        scorer_device=_canonical_mlx_prefilter_device(mlx_prefilter_scorer_device),
        scorer_batch_pairs=int(mlx_prefilter_scorer_batch_pairs),
        progress_every=int(mlx_prefilter_progress_every),
        repo_root=root,
    )
    report = {
        "schema": "snerv_checkpoint_archive_export.v1",
        "family": "snerv",
        "candidate_id": candidate.get("candidate_id"),
        "checkpoint_meta_path": meta_path.as_posix(),
        "checkpoint_meta_sha256": checkpoint_meta_sha256,
        "checkpoint_meta_present_at_report_write": meta_path.is_file(),
        "checkpoint_epoch": meta.get("global_epoch"),
        "checkpoint_state_kind": state_kind,
        "checkpoint_state_path": state_path.as_posix(),
        "checkpoint_state_sha256": checkpoint_state_sha256,
        "checkpoint_state_key_count": checkpoint_state_key_count,
        "checkpoint_state_present_at_report_write": state_path.is_file(),
        "checkpoint_trained_state_exportable": checkpoint_trained_state_exportable,
        "score_aware_long_training_executed": (
            packet.metadata.get("score_aware_long_training_executed") is True
        ),
        "score_aware_long_training_trained_state_exportable": (
            packet.metadata.get("score_aware_long_training_trained_state_exportable")
            is True
        ),
        "score_aware_long_training": dict(
            packet.metadata.get("score_aware_long_training") or {}
        ),
        "modelsize_candidate": candidate,
        "startup_json_path": startup_path.as_posix(),
        "startup_json_sha256": startup_json_sha256,
        "output_dir": out.as_posix(),
        "packet_path": packet_path.as_posix(),
        "packet_bytes": int(packet.total_bytes),
        "packet_sha256": _sha256_bytes(packet.packet),
        "packet_section_bytes": dict(packet.section_bytes),
        "packet_section_sha256": dict(packet.section_sha256),
        "packet_section_reports": dict(packet.section_reports),
        "packet_section_report_summary": _packet_section_report_summary(packet),
        "packet_metadata_summary": _packet_metadata_summary(packet),
        "official_checkpoint_export_binding": official_checkpoint_export_binding,
        "archive_path": str(archive_path) if archive_path else None,
        "archive_bytes": int(archive_bytes) if archive_bytes is not None else None,
        "archive_sha256": str(archive_sha256) if archive_sha256 else None,
        "hard_byte_ceiling_requested_by_candidate_or_startup": hard_byte_ceiling,
        "hard_byte_ceiling_checked_after_export": bool(
            hard_byte_ceiling is not None and archive_bytes is not None
        ),
        "hard_byte_ceiling_measurement_bypass_enabled": bool(
            allow_over_hard_byte_ceiling_for_measurement
        ),
        "decoder_codec": resolved_decoder_codec,
        "decoder_codec_requested": requested_decoder_codec,
        "lf_payload_codec": packet.metadata.get("lf_payload_codec"),
        "lf_payload_codec_requested": (
            packet.metadata.get("lf_payload_codec_requested") or resolved_lf_codec
        ),
        "lf_payload_codec_selected": packet.metadata.get("lf_payload_codec_selected"),
        "lf_payload_codec_selection_report": packet.metadata.get(
            "lf_payload_codec_selection_report"
        ),
        "target_bits_per_coeff": float(target_bits_per_coeff),
        "step_map_bits_per_coeff": float(step_map_bits_per_coeff),
        "model_size": model_size.as_jsonable(),
        "receiver_proof_path": receiver_proof.get("proof_path") if receiver_proof else None,
        "receiver_proof_sha256": (
            sha256_file(receiver_proof["proof_path"])
            if receiver_proof.get("proof_path")
            and Path(str(receiver_proof["proof_path"])).is_file()
            else None
        ),
        "receiver_proof_passed": receiver_proof.get("runtime_consumption_proof_passed") is True,
        "receiver_contract_satisfied": receiver_proof.get("receiver_contract_satisfied") is True,
        "modelsize_byte_cap_feedback_row": _modelsize_byte_cap_feedback_row(
            candidate=candidate,
            archive_bytes=int(archive_bytes) if archive_bytes is not None else None,
            hard_byte_ceiling=hard_byte_ceiling,
            packet_bytes=int(packet.total_bytes),
            decoder_codec=resolved_decoder_codec,
            receiver_proof_passed=receiver_proof.get("runtime_consumption_proof_passed") is True,
            receiver_contract_satisfied=receiver_proof.get("receiver_contract_satisfied") is True,
            archive_path=str(archive_path) if archive_path else None,
            archive_sha256=str(archive_sha256) if archive_sha256 else None,
            measurement_bypass_enabled=bool(
                allow_over_hard_byte_ceiling_for_measurement
            ),
        ),
        "local_mlx_prefilter_profile": mlx_prefilter_profile,
        "local_mlx_prefilter_profile_path": mlx_prefilter_profile.get("profile_path"),
        "local_mlx_prefilter_progress_path": mlx_prefilter_profile.get("progress_path"),
        "local_mlx_prefilter_written": mlx_prefilter_profile.get("written") is True,
        "blockers": _blockers(
            archive_bytes=int(archive_bytes) if archive_bytes is not None else None,
            hard_byte_ceiling=hard_byte_ceiling,
            receiver_proof=receiver_proof,
            receiver_proof_requested=bool(emit_receiver_proof),
            mlx_prefilter_profile=mlx_prefilter_profile,
            official_checkpoint_export_binding=official_checkpoint_export_binding,
            hard_byte_ceiling_measurement_bypass_enabled=bool(
                allow_over_hard_byte_ceiling_for_measurement
            ),
        ),
        **FALSE_AUTHORITY,
    }
    return report


def build_snerv_checkpoint_packet(
    state: dict[str, np.ndarray],
    *,
    levels: int,
    wavelet: str,
    target_bits_per_coeff: float,
    step_map_bits_per_coeff: float,
    decoder_payload_codec: str,
    lf_payload_codec: str,
    model_size: SnervModelSizeConfig,
    metadata_extra: dict[str, Any] | None = None,
) -> SnervArchivePacket:
    lf = np.asarray(state.get("latents_lf_planes"), dtype=np.float32)
    if lf.ndim != 5 or lf.shape[1] != 2 or lf.shape[2] != 3:
        raise ValueError(f"latents_lf_planes must be shaped (pairs,2,3,H,W); got {tuple(lf.shape)}")
    n_pairs, frames_per_pair, channels, lf_h, lf_w = (int(v) for v in lf.shape)
    orig_hw = (int(lf_h) * (1 << int(levels)), int(lf_w) * (1 << int(levels)))
    decoder = _decoder_from_state(state, levels=int(levels), model_size=model_size)
    n_levels = max(2, round(2.0 ** float(target_bits_per_coeff)))
    lf_quant_planes: list[np.ndarray] = []
    lf_zero_points: list[float] = []
    step_maps: list[np.ndarray] = []
    allocation_rows: list[dict[str, Any]] = []
    for pair_idx in range(n_pairs):
        for frame_idx in range(frames_per_pair):
            for channel_idx in range(channels):
                plane = np.asarray(lf[pair_idx, frame_idx, channel_idx], dtype=np.float64)
                q_uniform, scale, _zero_unused = quantize_lf(plane, n_levels=n_levels)
                step = np.full(plane.shape, float(scale), dtype=np.float32)
                q, _scale, zero = quantize_lf(plane, per_element_steps=step)
                lf_quant_planes.append(q)
                lf_zero_points.append(float(zero))
                step_maps.append(step)
                allocation_rows.append(
                    {
                        "schema": "snerv_checkpoint_lf_step_allocation_row.v1",
                        "pair_idx": int(pair_idx),
                        "source_pair_idx": int(pair_idx),
                        "frame_idx": int(frame_idx),
                        "channel_idx": int(channel_idx),
                        "mode": "uniform_checkpoint_lf_quantization",
                        "uniform_step": float(scale),
                        "target_bits_per_coeff": float(target_bits_per_coeff),
                        **FALSE_AUTHORITY,
                    }
                )
                if q.shape != q_uniform.shape:
                    raise AssertionError("internal quantization shape mismatch")
    step_packet = encode_step_maps_waterfill(
        step_maps,
        map_importance=np.ones((len(step_maps),), dtype=np.float64),
        target_bits_per_coeff=float(step_map_bits_per_coeff),
    )
    lf_payload_codec_requested = str(lf_payload_codec)
    lf_payload = encode_lf_quant_payload(
        lf_quant_planes,
        codec=lf_payload_codec_requested,
    )
    lf_payload_codec_report = inspect_lf_quant_payload_header(lf_payload)
    lf_payload_codec_selected = selected_lf_payload_codec_label(
        lf_payload_codec_report,
        requested_codec=lf_payload_codec_requested,
    )
    metadata = {
        "n_pairs": n_pairs,
        "frames_per_pair": frames_per_pair,
        "channels": channels,
        "levels": int(levels),
        "wavelet": str(wavelet),
        "carrier_hw": [int(orig_hw[0]), int(orig_hw[1])],
        "orig_hw": [int(orig_hw[0]), int(orig_hw[1])],
        "source_pair_indices": [int(i) for i in range(n_pairs)],
        "source_pair_indices_preserved": True,
        "pair_index_alignment_mode": "prefix_source_pair_indices",
        "lf_plane_count": len(lf_quant_planes),
        "lf_coeff_count_total": int(sum(int(plane.size) for plane in lf_quant_planes)),
        "lf_zero_dtype": "float32_le",
        "lf_scale_mode": "implicit_per_element_steps_scale_1",
        "lf_payload_codec": lf_payload_codec_selected,
        "lf_payload_codec_requested": lf_payload_codec_requested,
        "lf_payload_codec_selected": lf_payload_codec_selected,
        "lf_payload_codec_selection_report": lf_payload_codec_report,
        "step_map_packet_schema": step_packet.schema,
        "step_map_coder_mode": "checkpoint_uniform_step_map_waterfill",
        "step_map_coder_groups": [dict(group) for group in step_packet.groups],
        "step_map_waterfill_bits_per_coeff": float(step_map_bits_per_coeff),
        "target_bits_per_coeff": float(target_bits_per_coeff),
        "uniform_quantization_levels": int(n_levels),
        "allocation_mode": "checkpoint_direct_lf_decoder_uniform_quantization",
        "lf_step_allocation_mode": "uniform_checkpoint_lf_quantization",
        "lf_step_allocation_rows": allocation_rows,
        "hf_decoder_fit_mode": "trained_mlx_checkpoint_decoder_kernels",
        "native_mlx_hf_decoder_training": {
            "schema": "snerv_native_mlx_hf_decoder_training.v1",
            "executed": True,
            "source": "checkpoint_decoder_kernels",
            **FALSE_AUTHORITY,
        },
        "native_mlx_training_executed": True,
        "native_mlx_training_kind": "checkpoint_direct_lf_decoder_packetization",
        "score_aware_long_training_executed": True,
        "score_aware_long_training_kind": "checkpoint_harvest_interrupted_run",
        "native_mlx_training_export_guard": {
            "schema": "snerv_mlx_native_training_export_guard.v1",
            "native_mlx_training_executed": True,
            "blockers": [],
            **FALSE_AUTHORITY,
        },
        "decoder_payload_codec": resolve_decoder_payload_codec(decoder_payload_codec),
        "decoder_payload_codec_requested": str(decoder_payload_codec),
        "snerv_fc_dim": int(model_size.fc_dim),
        "snerv_emb_size": int(model_size.emb_size),
        "snerv_patch_radius": int(model_size.patch_radius),
        "snerv_model_size_adapter": model_size.adapter,
        "snerv_spectra_preserving_adapter_enabled": bool(
            model_size.adapter == "snerv_spectra_preserving_mfu_hfr_tub_adapter_v1"
        ),
        "snerv_mfu_scales": [int(v) for v in model_size.mfu_scales],
        "snerv_hfr_gain": float(model_size.hfr_gain),
        "snerv_temporal_context": int(model_size.temporal_context),
        "snerv_temporal_mode": model_size.temporal_mode,
        "decoder_feature_count": int(model_size.feature_count),
        **dict(metadata_extra or {}),
        **FALSE_AUTHORITY,
    }
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=lf_zero_points),
        lf_payload=lf_payload,
        decoder_payload=encode_decoder_payload(
            decoder,
            codec=resolve_decoder_payload_codec(decoder_payload_codec),
        ),
        step_map_packet=step_packet.packet,
        metadata=metadata,
    )
    frames = decode_snerv_archive_frames(archive.packet)
    if tuple(frames.shape) != (n_pairs, frames_per_pair, channels, *orig_hw):
        raise ValueError(
            "receiver decode shape mismatch: "
            f"got {tuple(frames.shape)}, expected {(n_pairs, frames_per_pair, channels, *orig_hw)}"
        )
    return archive


def build_snerv_official_checkpoint_packet(
    state: dict[str, np.ndarray],
    *,
    model_size: SnervModelSizeConfig,
    metadata_extra: dict[str, Any] | None = None,
) -> SnervArchivePacket:
    """Packetize an interrupted official MFU/HFR/TUB MLX checkpoint.

    Official SNeRV long-training checkpoints store receiver atoms under the
    official renderer state names (`low`, `skip_mid`, `skip_high`, `hfr_*`).
    When the selected adapter is official, exporting through the local LF/kernel
    grammar loses that signal, so this path reconstructs the official receiver
    payload directly and fails closed if those atoms are absent.
    """

    components = _official_components_from_checkpoint_state(
        state,
        model_size=model_size,
    )
    n_frames = int(components["skip_high_full_shape"][0])
    if n_frames <= 0 or n_frames % 2:
        raise ValueError(
            f"official checkpoint skip_high frame count must be positive/even, got {n_frames}"
        )
    return _build_official_mfu_hfr_tub_packet_from_components(
        components,
        source_pair_indices=tuple(range(n_frames // 2)),
        model_size=model_size,
        metadata_extra={
            "native_mlx_training_kind": "checkpoint_official_mfu_hfr_tub_receiver_packetization",
            "score_aware_long_training_kind": "checkpoint_official_mfu_hfr_tub_harvest_interrupted_run",
            "checkpoint_packetization_mode": "official_mfu_hfr_tub_receiver_payload",
            "allocation_mode": "official_mfu_hfr_tub_checkpoint_receiver_payload",
            "hf_decoder_fit_mode": "trained_official_hfr_heads_from_mlx_checkpoint",
            "official_checkpoint_state_keys_verified": True,
            **dict(metadata_extra or {}),
        },
    )


def _official_components_from_checkpoint_state(
    state: dict[str, np.ndarray],
    *,
    model_size: SnervModelSizeConfig,
) -> dict[str, Any]:
    low = _checkpoint_state_array(state, "low")
    skip_mid = _checkpoint_state_array(state, "skip_mid")
    skip_high = _checkpoint_state_array(state, "skip_high")
    low = np.asarray(low, dtype=np.float64)
    skip_mid = np.asarray(skip_mid, dtype=np.float64)
    skip_high = np.asarray(skip_high, dtype=np.float64)
    _validate_official_checkpoint_tensor("low", low)
    _validate_official_checkpoint_tensor("skip_mid", skip_mid)
    _validate_official_checkpoint_tensor("skip_high", skip_high)
    full_shape = _infer_official_skip_high_full_shape(
        low=low,
        skip_mid=skip_mid,
        skip_high=skip_high,
        official_skip_high_mode=str(model_size.official_skip_high_mode),
    )
    n_frames, channels, ll_h, ll_w = (int(v) for v in full_shape)
    h = int(ll_h) * 2
    w = int(ll_w) * 2
    tub_zero = np.zeros((channels, h, w), dtype=np.float64)
    tub_temporal_encoder_concat, tub_output2_raw = (
        _official_tub_output2_payload_from_checkpoint_state(state)
    )
    default_temporal_shape = (1, 4, max(1, ll_h // 2), max(1, ll_w // 2))
    default_output2_shape = (2, 8, max(1, ll_h // 2), max(1, ll_w // 2))
    temporal_encoder_output_shape = (
        tuple(int(v) for v in tub_temporal_encoder_concat.shape)
        if tub_temporal_encoder_concat is not None
        else default_temporal_shape
    )
    output2_decoder_output_shape = (
        tuple(int(v) for v in tub_output2_raw.shape)
        if tub_output2_raw is not None
        else default_output2_shape
    )
    return {
        "mfu": _official_passthrough_mfu(channels=channels),
        "hfr_heads": _official_hfr_heads_from_checkpoint_state(state),
        "low": low,
        "skip_mid": skip_mid,
        "skip_high": skip_high,
        "skip_high_mode": str(model_size.official_skip_high_mode),
        "skip_high_full_shape": full_shape,
        "skip_high_export_storage_shape": tuple(int(v) for v in skip_high.shape),
        "skip_high_export_is_compact_train_state": tuple(int(v) for v in skip_high.shape)
        != full_shape,
        "tub_current": tub_zero,
        "tub_previous": tub_zero,
        "tub_next_frame": tub_zero,
        "temporal_encoder_output_shape": temporal_encoder_output_shape,
        "fc_hw": (2, 2),
        "output2_decoder_output_shape": output2_decoder_output_shape,
        **(
            {
                "tub_temporal_encoder_concat": tub_temporal_encoder_concat,
                "tub_output2_raw": tub_output2_raw,
                "official_checkpoint_tub_output2_payload_preserved": True,
            }
            if tub_temporal_encoder_concat is not None
            and tub_output2_raw is not None
            else {"official_checkpoint_tub_output2_payload_preserved": False}
        ),
        "n_pairs": n_frames // 2,
        "frames_per_pair": 2,
        "channels": channels,
        "h": h,
        "w": w,
        "model_size": model_size.as_jsonable(),
    }


def _official_hfr_heads_from_checkpoint_state(
    state: dict[str, np.ndarray],
) -> OfficialHfrHeads:
    return OfficialHfrHeads(
        lh_head=_official_hfr_head_from_checkpoint_state(state, "lh"),
        hl_head=_official_hfr_head_from_checkpoint_state(state, "hl"),
        hh_head=_official_hfr_head_from_checkpoint_state(state, "hh"),
    )


def _official_hfr_head_from_checkpoint_state(
    state: dict[str, np.ndarray],
    name: str,
) -> OfficialHfrConvBlock:
    return OfficialHfrConvBlock(
        conv1=OfficialConv2dNchw(
            _checkpoint_state_array(
                state,
                f"hfr.{name}.conv1.weight",
                f"hfr_{name}_conv1_weight",
            ),
            _checkpoint_state_array(
                state,
                f"hfr.{name}.conv1.bias",
                f"hfr_{name}_conv1_bias",
            ),
            padding=0,
        ),
        conv2=OfficialConv2dNchw(
            _checkpoint_state_array(
                state,
                f"hfr.{name}.conv2.weight",
                f"hfr_{name}_conv2_weight",
            ),
            _checkpoint_state_array(
                state,
                f"hfr.{name}.conv2.bias",
                f"hfr_{name}_conv2_bias",
            ),
            padding=1,
        ),
    )


def _official_tub_output2_payload_from_checkpoint_state(
    state: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    temporal = _checkpoint_state_optional_array(
        state,
        "tub.temporal_encoder_concat",
        "tub_temporal_encoder_concat",
    )
    raw = _checkpoint_state_optional_array(
        state,
        "tub.output2_raw",
        "tub_output2_raw",
    )
    if (temporal is None) != (raw is None):
        raise ValueError(
            "official checkpoint TUB output2 payload requires both "
            "tub.temporal_encoder_concat and tub.output2_raw"
        )
    if temporal is None or raw is None:
        return None, None
    temporal = np.asarray(temporal, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    _validate_official_checkpoint_tensor("tub.temporal_encoder_concat", temporal)
    _validate_official_checkpoint_tensor("tub.output2_raw", raw)
    return temporal, raw


def _checkpoint_state_array(
    state: dict[str, np.ndarray],
    *keys: str,
) -> np.ndarray:
    for key in keys:
        if key in state:
            return np.asarray(state[key])
    joined = ", ".join(keys)
    raise ValueError(f"official checkpoint state missing any of: {joined}")


def _checkpoint_state_optional_array(
    state: dict[str, np.ndarray],
    *keys: str,
) -> np.ndarray | None:
    for key in keys:
        if key in state:
            return np.asarray(state[key])
    return None


def _validate_official_checkpoint_tensor(name: str, value: np.ndarray) -> None:
    if value.ndim != 4:
        raise ValueError(f"official checkpoint tensor {name} must be NCHW, got {value.shape}")
    if any(int(dim) <= 0 for dim in value.shape):
        raise ValueError(f"official checkpoint tensor {name} has non-positive shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"official checkpoint tensor {name} contains non-finite values")


def _infer_official_skip_high_full_shape(
    *,
    low: np.ndarray,
    skip_mid: np.ndarray,
    skip_high: np.ndarray,
    official_skip_high_mode: str,
) -> tuple[int, int, int, int]:
    if tuple(low.shape[:2]) != tuple(skip_mid.shape[:2]):
        raise ValueError(
            f"official checkpoint low/skip_mid batch-channel mismatch: {low.shape} vs {skip_mid.shape}"
        )
    n_frames, channels, mid_h, mid_w = (int(v) for v in skip_mid.shape)
    if tuple(int(v) for v in low.shape[-2:]) != (max(1, mid_h // 2), max(1, mid_w // 2)):
        raise ValueError(
            "official checkpoint low spatial shape must be half skip_mid; "
            f"low={low.shape[-2:]} skip_mid={skip_mid.shape[-2:]}"
        )
    full_shape = (n_frames, channels, mid_h * 2, mid_w * 2)
    mode = str(official_skip_high_mode).strip().lower()
    expected_shapes = {
        "full": full_shape,
        "shared_mean": (1, channels, full_shape[2], full_shape[3]),
        "channel_mean": (1, channels, 1, 1),
        "scalar_mean": (1, 1, 1, 1),
    }
    expected = expected_shapes.get(mode)
    if expected is None:
        raise ValueError(f"unsupported official checkpoint skip_high mode: {mode!r}")
    if tuple(int(v) for v in skip_high.shape) != expected:
        raise ValueError(
            "official checkpoint skip_high shape does not match selected compact mode; "
            f"mode={mode} got={tuple(skip_high.shape)} expected={expected}"
        )
    return full_shape


def _decoder_from_state(
    state: dict[str, np.ndarray],
    *,
    levels: int,
    model_size: SnervModelSizeConfig,
) -> HfGenerationDecoder:
    kernels: dict[int, dict[str, np.ndarray]] = {}
    for lvl in range(int(levels)):
        row: dict[str, np.ndarray] = {}
        for subband in _DETAIL_KEYS:
            key = f"decoder_kernels.{lvl}.{subband}"
            if key not in state:
                raise ValueError(f"checkpoint state missing {key}")
            arr = np.asarray(state[key], dtype=np.float64).reshape(-1)
            if arr.size != int(model_size.feature_count):
                raise ValueError(
                    f"{key} has {arr.size} values, expected {model_size.feature_count}"
                )
            row[subband] = arr
        kernels[lvl] = row
    return HfGenerationDecoder(
        kernels=kernels,
        levels=int(levels),
        model_size=model_size,
    )


def _checkpoint_state_path(meta: dict[str, Any], *, state_kind: str) -> Path:
    key = "ema_shadow_state_path" if state_kind == "ema" else "live_state_path"
    value = meta.get(key)
    if not value:
        raise ValueError(f"checkpoint meta missing {key}")
    path = Path(str(value)).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint state not found: {path}")
    return path


def _maybe_write_receiver_decoded_mlx_prefilter(
    *,
    requested: bool,
    output_dir: str | Path,
    packet: SnervArchivePacket,
    receiver_proof: dict[str, Any],
    startup: dict[str, Any],
    command_args: dict[str, Any],
    archive_bytes: int | None,
    archive_sha256: str | None,
    source_video_path: str | Path | None,
    scorer_upstream_dir: str | Path | None,
    scorer_device: str,
    scorer_batch_pairs: int,
    progress_every: int,
    repo_root: str | Path,
) -> dict[str, Any]:
    metadata = dict(packet.metadata)
    out = Path(output_dir).expanduser().resolve(strict=False)
    batch_control = _canonical_mlx_prefilter_batch_pairs(scorer_batch_pairs)
    effective_batch_pairs = int(batch_control["effective_batch_pairs"])
    base = {
        "schema": "snerv_checkpoint_receiver_decoded_mlx_prefilter_request.v1",
        "requested": bool(requested),
        "profile_path": (out / "local_mlx_prefilter_profile.json").as_posix(),
        "progress_path": (out / "local_mlx_prefilter_progress.jsonl").as_posix(),
        "packet_sha256": _sha256_bytes(packet.packet),
        "packet_bytes": int(packet.total_bytes),
        "receiver_decoded_selected_packet": True,
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
            "blockers": ["snerv_checkpoint_mlx_prefilter_not_requested"],
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
    source_pair_indices = _packet_source_pair_indices(metadata)
    cache_profile = _maybe_write_receiver_raw_cache_mlx_prefilter(
        requested=bool(requested),
        output_dir=out,
        receiver_proof=receiver_proof,
        source_pair_indices=source_pair_indices,
        source_video_path=source_video,
        scorer_upstream_dir=upstream_dir,
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        scorer_device=scorer_device,
        scorer_batch_pairs=effective_batch_pairs,
        progress_every=progress_every,
        repo_root=repo_root,
        base=base,
    )
    if cache_profile is not None:
        return cache_profile
    output_hw = _packet_output_hw(metadata)
    if archive_bytes is None or archive_sha256 is None:
        return _attach_prefilter_batch_control(
            _write_snerv_native_receiver_decoded_mlx_prefilter(
                requested=True,
                output_dir=out,
                selected_packet=packet.packet,
                target0_np=np.empty((0, output_hw[0], output_hw[1], 3), dtype=np.float32),
                target1_np=np.empty((0, output_hw[0], output_hw[1], 3), dtype=np.float32),
                archive_bytes=archive_bytes,
                archive_sha256=archive_sha256,
                source_video_path=source_video,
                scorer_upstream_dir=upstream_dir,
                scorer_device=scorer_device,
                scorer_batch_pairs=effective_batch_pairs,
                progress_every=progress_every,
                allow_overwrite=False,
            ),
            base=base,
        )
    target0_mlx, target1_mlx = decode_mlx_targets(
        source_video,
        num_pairs=len(source_pair_indices),
        output_height=int(output_hw[0]),
        output_width=int(output_hw[1]),
        pair_indices=source_pair_indices,
    )
    target0_np = np.asarray(target0_mlx, dtype=np.float32)
    target1_np = np.asarray(target1_mlx, dtype=np.float32)
    return _attach_prefilter_batch_control(
        _write_snerv_native_receiver_decoded_mlx_prefilter(
            requested=True,
            output_dir=out,
            selected_packet=packet.packet,
            target0_np=target0_np,
            target1_np=target1_np,
            archive_bytes=archive_bytes,
            archive_sha256=archive_sha256,
            source_video_path=source_video,
            scorer_upstream_dir=upstream_dir,
            scorer_device=scorer_device,
            scorer_batch_pairs=effective_batch_pairs,
            progress_every=progress_every,
            allow_overwrite=False,
        ),
        base=base,
    )


def _attach_prefilter_batch_control(
    profile: dict[str, Any],
    *,
    base: dict[str, Any],
) -> dict[str, Any]:
    return {
        **base,
        **dict(profile),
        "scorer_batch_pairs_requested": base.get("scorer_batch_pairs_requested"),
        "scorer_batch_pairs_effective": base.get("scorer_batch_pairs_effective"),
        "scorer_batch_pairs_normalized_to_singleton": base.get(
            "scorer_batch_pairs_normalized_to_singleton"
        ),
        "scorer_batch_pairs_normalization": base.get(
            "scorer_batch_pairs_normalization"
        ),
    }


def _maybe_write_receiver_raw_cache_mlx_prefilter(
    *,
    requested: bool,
    output_dir: Path,
    receiver_proof: dict[str, Any],
    source_pair_indices: tuple[int, ...],
    source_video_path: Path,
    scorer_upstream_dir: Path,
    archive_bytes: int | None,
    archive_sha256: str | None,
    scorer_device: str,
    scorer_batch_pairs: int,
    progress_every: int,
    repo_root: str | Path,
    base: dict[str, Any],
) -> dict[str, Any] | None:
    if not requested:
        return None
    n_pairs = len(source_pair_indices)
    if n_pairs < 1 or source_pair_indices != tuple(range(n_pairs)):
        return None
    raw_value = receiver_proof.get("receiver_output_path")
    raw_path = Path(str(raw_value)).expanduser().resolve(strict=False) if raw_value else None
    if raw_path is None or not raw_path.is_file() or raw_path.stat().st_size <= 0:
        return None
    if archive_bytes is None or archive_sha256 is None:
        return None
    profile_path = output_dir / "local_mlx_prefilter_profile.json"
    reference_cache_dir = output_dir / "scorer_input_caches" / "reference_source_video"
    candidate_cache_dir = output_dir / "scorer_input_caches" / "candidate_receiver_raw"
    components_dir = output_dir / "scorer_input_caches" / "components"
    try:
        reference_manifest = _ensure_video_scorer_cache(
            source_video_path,
            reference_cache_dir,
            pair_count=n_pairs,
            batch_pairs=int(scorer_batch_pairs),
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
            batch_pairs=int(scorer_batch_pairs),
        )
        cache_quality_gate_path = output_dir / "cache_quality_gate.json"
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
            upstream_dir=scorer_upstream_dir,
            batch_pairs=int(scorer_batch_pairs),
            device_type=str(scorer_device),
            components_dir=components_dir,
            progress_every=max(0, int(progress_every)),
            allow_gpu_research_signal=str(scorer_device) == "gpu",
            allow_unaudited_candidate_cache_debug=True,
            cache_integrity_mode=MANIFEST_CACHE_INTEGRITY_MODE,
            response_family="snerv",
        )
        response = attach_cache_quality_gate_to_mlx_scorer_response(
            response,
            cache_quality_gate,
            source_path=cache_quality_gate_path,
        )
        response = {
            **response,
            "schema": "mlx_scorer_response.v1",
            "schema_version": "mlx_scorer_response.v1",
            "snerv_receiver_raw_cache_prefilter": {
                "schema": "snerv_receiver_raw_cache_prefilter.v1",
                "receiver_output_path": raw_path.as_posix(),
                "receiver_output_sha256": receiver_proof.get("receiver_output_sha256"),
                "reference_cache_manifest": reference_manifest,
                "candidate_cache_manifest": candidate_manifest,
                "cache_quality_gate": response.get("cache_quality_gate"),
                "cache_quality_gate_path": cache_quality_gate_path.as_posix(),
                "cache_quality_gate_sha256": sha256_file(cache_quality_gate_path),
                "source_pair_indices_alignment": "prefix_source_pair_indices",
                "scorer_batch_pairs_requested": base.get(
                    "scorer_batch_pairs_requested"
                ),
                "scorer_batch_pairs_effective": base.get(
                    "scorer_batch_pairs_effective"
                ),
                "scorer_batch_pairs_normalized_to_singleton": base.get(
                    "scorer_batch_pairs_normalized_to_singleton"
                ),
                "scorer_batch_pairs_normalization": base.get(
                    "scorer_batch_pairs_normalization"
                ),
                **FALSE_AUTHORITY,
            },
        }
        write_json_artifact(profile_path, response)
        reference_cleanup = _cleanup_rebuildable_scorer_cache_arrays(
            reference_cache_dir,
            reference_manifest,
            reason="snerv_checkpoint_receiver_raw_prefilter_reference_cache_rebuildable",
        )
        candidate_cleanup = _cleanup_rebuildable_scorer_cache_arrays(
            candidate_cache_dir,
            candidate_manifest,
            reason="snerv_checkpoint_receiver_raw_prefilter_candidate_cache_rebuildable",
        )
        return {
            **base,
            "written": True,
            "profile_schema": "mlx_scorer_response.v1",
            "profile_path": profile_path.as_posix(),
            "profile_sha256": sha256_file(profile_path),
            "progress_path": None,
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
                "schema": "snerv_receiver_raw_cache_prefilter_cleanup.v1",
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
                "snerv_receiver_raw_cache_prefilter_false_authority",
            ],
            **FALSE_AUTHORITY,
        }
    except Exception as exc:
        failure = {
            "schema": "snerv_receiver_raw_cache_prefilter_failure.v1",
            "requested": True,
            "receiver_output_path": raw_path.as_posix(),
            "failure": repr(exc),
            "blockers": ["snerv_receiver_raw_cache_prefilter_failed"],
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
            "blockers": [
                "snerv_receiver_raw_cache_prefilter_failed",
            ],
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
        "schema": "snerv_checkpoint_scorer_cache_array_cleanup.v1",
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
        if path is None or path.suffix != ".npy":
            continue
        if not path.is_file():
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


def _packet_source_pair_indices(metadata: dict[str, Any]) -> tuple[int, ...]:
    n_pairs = int(metadata.get("n_pairs") or 0)
    raw = metadata.get("source_pair_indices")
    if raw is None:
        return tuple(range(n_pairs))
    if not isinstance(raw, list | tuple):
        raise ValueError("SNAR1 metadata source_pair_indices must be a list when present")
    out = tuple(int(value) for value in raw)
    if len(out) != n_pairs:
        raise ValueError(
            "SNAR1 metadata source_pair_indices length mismatch: "
            f"{len(out)} != {n_pairs}"
        )
    return out


def _packet_output_hw(metadata: dict[str, Any]) -> tuple[int, int]:
    raw = metadata.get("carrier_hw") or metadata.get("orig_hw")
    if isinstance(raw, list | tuple) and len(raw) == 2:
        height, width = (int(raw[0]), int(raw[1]))
        if height > 0 and width > 0:
            return height, width
    return int(SCORER_HW[0]), int(SCORER_HW[1])


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


def _blockers(
    *,
    archive_bytes: int | None,
    hard_byte_ceiling: int | None,
    receiver_proof: dict[str, Any],
    receiver_proof_requested: bool,
    mlx_prefilter_profile: dict[str, Any],
    official_checkpoint_export_binding: dict[str, Any] | None = None,
    hard_byte_ceiling_measurement_bypass_enabled: bool = False,
) -> list[str]:
    blockers = [
        "macos_mlx_checkpoint_export_false_authority",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
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
    if not receiver_proof_requested:
        blockers.append("receiver_proof_not_requested")
    elif receiver_proof.get("runtime_consumption_proof_passed") is not True:
        blockers.append("receiver_proof_not_passed")
    if receiver_proof_requested and receiver_proof.get("receiver_contract_satisfied") is not True:
        blockers.append("receiver_contract_not_satisfied")
    if (
        hard_byte_ceiling is not None
        and archive_bytes is not None
        and int(archive_bytes) > int(hard_byte_ceiling)
    ):
        blockers.append("archive_bytes_exceed_tightest_hard_ceiling")
        if hard_byte_ceiling_measurement_bypass_enabled:
            blockers.append("hard_byte_ceiling_export_bypassed_for_measurement")
    official_binding = (
        official_checkpoint_export_binding
        if isinstance(official_checkpoint_export_binding, dict)
        else {}
    )
    if (
        official_binding.get("official_tub_output2_activation_payload_bound") is True
        and official_binding.get("official_tub_output2_score_causal_receiver_frame_bound")
        is not True
    ):
        blockers.append(
            "snerv_official_tub_output2_non_score_causal_bytes_present"
        )
        blockers.append(
            "snerv_official_tub_output2_elide_or_bind_source_faithful_frame_decode"
        )
    return list(dict.fromkeys(blockers))


def _modelsize_byte_cap_feedback_row(
    *,
    candidate: dict[str, Any],
    archive_bytes: int | None,
    hard_byte_ceiling: int | None,
    packet_bytes: int,
    decoder_codec: str,
    receiver_proof_passed: bool,
    receiver_contract_satisfied: bool,
    archive_path: str | None,
    archive_sha256: str | None,
    measurement_bypass_enabled: bool,
) -> dict[str, Any]:
    nominal = (
        _optional_int(candidate.get("nominal_total_payload_bytes"))
        or _optional_int(candidate.get("total_payload_bytes"))
        or _optional_int(candidate.get("estimated_total_payload_bytes"))
        or int(packet_bytes)
    )
    measured = int(archive_bytes) if archive_bytes is not None else None
    archive_minus_nominal = (
        None if measured is None else int(measured) - int(nominal)
    )
    archive_to_nominal = (
        None
        if measured is None or int(nominal) <= 0
        else float(measured) / float(nominal)
    )
    overrun = (
        None
        if measured is None or hard_byte_ceiling is None
        else max(0, int(measured) - int(hard_byte_ceiling))
    )
    required_nominal_max = None
    if (
        measured is not None
        and hard_byte_ceiling is not None
        and int(measured) > 0
        and int(nominal) > 0
    ):
        required_nominal_max = math.floor(
            float(hard_byte_ceiling) * float(nominal) / float(measured)
        )
    receiver_closed = bool(receiver_proof_passed and receiver_contract_satisfied)
    return {
        "schema": "nerv_modelsize_byte_cap_feedback_row.v1",
        "family": "snerv",
        "candidate_id": candidate.get("candidate_id"),
        "codec": str(decoder_codec),
        "decoder_codec": str(decoder_codec),
        "modelsize_candidate": candidate,
        "hard_byte_ceiling": hard_byte_ceiling,
        "hard_byte_ceiling_measurement_bypass_enabled": bool(measurement_bypass_enabled),
        "nominal_total_payload_bytes": int(nominal),
        "measured_archive_bytes": measured,
        "archive_bytes": measured,
        "packet_bytes": int(packet_bytes),
        "archive_path": archive_path,
        "archive_sha256": archive_sha256,
        "archive_minus_nominal_bytes": archive_minus_nominal,
        "archive_to_nominal_ratio": archive_to_nominal,
        "calibrated_archive_overrun_bytes": overrun,
        "required_nominal_payload_bytes_max": required_nominal_max,
        "receiver_proof_passed": bool(receiver_proof_passed),
        "receiver_contract_satisfied": bool(receiver_contract_satisfied),
        "receiver_closed": receiver_closed,
        "authority_surface": "measured_archive_zip_bytes_after_receiver_export",
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
    values: list[int] = []
    candidate_ceiling = _optional_int(
        candidate.get("hard_byte_ceiling", candidate.get("snerv_hard_byte_ceiling"))
    )
    if candidate_ceiling is not None:
        values.append(candidate_ceiling)
    startup_ceiling = _min_positive_int(hard_byte_ceilings)
    if startup_ceiling is not None:
        values.append(startup_ceiling)
    positives = [int(value) for value in values if int(value) > 0]
    return min(positives) if positives else None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"startup JSON missing object {name}")
    return dict(value)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _official_checkpoint_export_binding(
    packet: SnervArchivePacket,
    *,
    model_size: SnervModelSizeConfig,
    checkpoint_state: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Classify official checkpoint packet binding without claiming source parity."""

    requested = bool(model_size.official_mfu_hfr_tub_numeric_primitives_requested)
    selected_authority = _selected_packet_official_payload_authority(packet.packet)
    receiver_tensor_map = _official_receiver_tensor_map_from_packet(packet.packet)
    packet_metadata = _packet_metadata_from_snerv_packet(packet.packet)
    tensor_category_counts = dict(receiver_tensor_map.get("category_counts") or {})
    tensor_names = {
        str(row.get("name") or "")
        for row in receiver_tensor_map.get("rows") or ()
        if isinstance(row, dict)
    }
    tub_output2_storage = packet_metadata.get("official_tub_output2_storage")
    if not isinstance(tub_output2_storage, dict):
        tub_output2_storage = {}
    tub_output2_activation_payload_bound = bool(
        selected_authority.get("official_decoder_payload_selected") is True
        and receiver_tensor_map.get("receiver_tensor_map_verified") is True
        and tensor_category_counts.get("official_tub_output2_payload", 0) >= 2
        and {"tub.temporal_encoder_concat", "tub.output2_raw"}.issubset(tensor_names)
        and tub_output2_storage.get("stored") is True
        and tub_output2_storage.get("receiver_executes_output2_fusion_from_payload")
        is True
        and packet_metadata.get("official_tub_output2_receiver_executed") is True
    )
    tub_output2_score_causal_frame_bound = bool(
        tub_output2_activation_payload_bound
        and tub_output2_storage.get("receiver_frame_decode_consumes_output2") is True
        and tub_output2_storage.get("scored_pixel_render_bound") is True
    )
    tub_output2_store_requested = bool(
        getattr(
            model_size,
            "official_tub_output2_store_for_receiver_proof_requested",
            model_size.official_tub_output2_store_for_receiver_proof,
        )
    )
    tub_output2_store_honored = bool(
        model_size.official_tub_output2_store_for_receiver_proof
    )
    tub_output2_source_raw_bytes = int(
        tub_output2_storage.get("source_raw_bytes") or 0
    )
    tub_output2_selected_raw_bytes = int(
        tub_output2_storage.get("stored_raw_bytes") or 0
    )
    native_checkpoint_export_bound = bool(
        requested
        and selected_authority.get("frame_producing_official_export") is True
        and receiver_tensor_map.get("receiver_tensor_map_verified") is True
    )
    official_mapping_state = _official_state_dict_slice_for_mapping(
        checkpoint_state or {}
    )
    official_state_manifest = build_snerv_official_trained_checkpoint_mapping_manifest(
        official_mapping_state or None,
        state_dict_kind=(
            "checkpoint_export_upstream_official_state_dict_slice"
            if official_mapping_state
            else "checkpoint_export_has_receiver_atoms_not_upstream_official_state_dict"
        ),
        source="export_snerv_checkpoint_archive.official_checkpoint_export_binding",
    )
    mfu_hfr_mapping_proven = bool(
        official_state_manifest.get(
            "official_mfu_hfr_trained_checkpoint_weight_mapping_proven"
        )
        is True
    )
    tub_mapping_proven = bool(
        official_state_manifest.get(
            "official_tub_temporal_encoder_weight_mapping_proven"
        )
        is True
    )
    preserved_blockers = [
        *(
            []
            if mfu_hfr_mapping_proven
            else [
                "snerv_official_mfu_hfr_tub_weight_mapping_missing",
                "snerv_official_trained_checkpoint_state_dict_mapping_missing",
            ]
        ),
        "snerv_official_trained_checkpoint_source_forward_replay_missing",
        *(
            []
            if tub_mapping_proven
            else [
                "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
            ]
        ),
    ]
    return {
        "schema": "snerv_official_checkpoint_export_binding.v1",
        "requested": requested,
        "native_checkpoint_export_bound_to_official_payload": (
            native_checkpoint_export_bound
        ),
        "official_receiver_payload_bound": bool(
            selected_authority.get("official_decoder_payload_selected") is True
        ),
        "official_receiver_tensor_map_verified": bool(
            receiver_tensor_map.get("receiver_tensor_map_verified") is True
        ),
        "selected_packet_status": selected_authority.get("status"),
        "selected_packet_decoded_frame_shape": selected_authority.get(
            "decoded_frame_shape"
        ),
        "selected_packet_authority": selected_authority,
        "official_receiver_tensor_map": receiver_tensor_map,
        "official_tensor_category_bytes": dict(
            receiver_tensor_map.get("category_bytes") or {}
        ),
        "official_tensor_category_counts": dict(
            tensor_category_counts
        ),
        "official_tub_output2_storage": tub_output2_storage,
        "official_tub_output2_export_mode": (
            model_size.official_tub_output2_export_mode
        ),
        "official_tub_output2_store_for_receiver_proof_requested": (
            tub_output2_store_requested
        ),
        "official_tub_output2_store_for_receiver_proof_honored": (
            tub_output2_store_honored
        ),
        "official_tub_output2_auto_elided_for_score_candidate": bool(
            tub_output2_store_requested
            and not tub_output2_store_honored
            and model_size.official_tub_output2_export_mode == "auto_elide"
        ),
        "official_tub_output2_source_raw_bytes": tub_output2_source_raw_bytes,
        "official_tub_output2_selected_runtime_raw_bytes": (
            tub_output2_selected_raw_bytes
        ),
        "official_tub_output2_elided_raw_bytes": max(
            0,
            tub_output2_source_raw_bytes - tub_output2_selected_raw_bytes,
        ),
        "official_tub_output2_receiver_executed": bool(
            packet_metadata.get("official_tub_output2_receiver_executed") is True
        ),
        "official_tub_output2_activation_payload_bound": (
            tub_output2_activation_payload_bound
        ),
        "official_tub_receiver_activation_mapping_proven": (
            tub_output2_activation_payload_bound
        ),
        "official_tub_receiver_activation_mapping_semantics": (
            "receiver_executes_stored_temporal_encoder_concat_and_output2_raw_payload"
            if tub_output2_activation_payload_bound
            else "requires_tub_temporal_encoder_concat_and_output2_raw_receiver_payload"
        ),
        "official_tub_output2_score_causal_receiver_frame_bound": (
            tub_output2_score_causal_frame_bound
        ),
        "official_tub_output2_byte_cap_admission": (
            "admit_as_score_causal_tub_activation"
            if tub_output2_score_causal_frame_bound
            else (
                "elide_until_receiver_frame_decode_consumes_output2_or_scored_delta_positive"
                if tub_output2_activation_payload_bound
                else "not_present"
            )
        ),
        "official_trained_checkpoint_mapping_manifest": official_state_manifest,
        "trained_state_exportable": bool(native_checkpoint_export_bound),
        "official_trained_state_exportable": bool(native_checkpoint_export_bound),
        "official_trained_checkpoint_state_dict_slice_present": bool(
            official_mapping_state
        ),
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": (
            mfu_hfr_mapping_proven
        ),
        "official_tub_temporal_encoder_weight_mapping_proven": tub_mapping_proven,
        "official_trained_checkpoint_state_dict_mapping_verified": bool(
            mfu_hfr_mapping_proven and tub_mapping_proven
        ),
        "official_export_bound": False,
        "official_export_bound_semantics": (
            "checkpoint_export_receiver_payload_binding_not_source_forward_parity"
        ),
        "source_forward_replay_authority": False,
        "source_forward_replay_bound": False,
        "preserved_blockers": preserved_blockers,
        "blockers": (
            []
            if native_checkpoint_export_bound
            else [
                *(
                    str(blocker)
                    for blocker in selected_authority.get("blockers", ())
                    if str(blocker)
                ),
                *(
                    str(blocker)
                    for blocker in receiver_tensor_map.get("blockers", ())
                    if str(blocker)
                ),
            ]
        ),
        **FALSE_AUTHORITY,
    }


def _packet_metadata_from_snerv_packet(packet: bytes) -> dict[str, Any]:
    try:
        decoded = unpack_snerv_archive(packet)
    except Exception:
        return {}
    metadata = decoded.metadata
    return dict(metadata) if isinstance(metadata, dict) else {}


def _official_state_dict_slice_for_mapping(
    state: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        str(key): value
        for key, value in state.items()
        if str(key).startswith(("decoder.", "encoder."))
    }


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    packet_metadata = dict(report.get("packet_metadata_summary") or {})
    section_summary = dict(report.get("packet_section_report_summary") or {})
    lf_summary = dict(section_summary.get("lf_payload_codec_report") or {})
    return {
        "schema": report.get("schema"),
        "checkpoint_epoch": report.get("checkpoint_epoch"),
        "packet_bytes": report.get("packet_bytes"),
        "checkpoint_packetization_mode": packet_metadata.get(
            "checkpoint_packetization_mode"
        ),
        "decoder_payload_codec": packet_metadata.get("decoder_payload_codec"),
        "archive_bytes": report.get("archive_bytes"),
        "hard_byte_ceiling_requested_by_candidate_or_startup": report.get(
            "hard_byte_ceiling_requested_by_candidate_or_startup"
        ),
        "hard_byte_ceiling_checked_after_export": report.get(
            "hard_byte_ceiling_checked_after_export"
        ),
        "hard_byte_ceiling_measurement_bypass_enabled": report.get(
            "hard_byte_ceiling_measurement_bypass_enabled"
        ),
        "receiver_proof_passed": report.get("receiver_proof_passed"),
        "receiver_contract_satisfied": report.get("receiver_contract_satisfied"),
        "checkpoint_trained_state_exportable": report.get(
            "checkpoint_trained_state_exportable"
        ),
        "score_aware_long_training_executed": report.get(
            "score_aware_long_training_executed"
        ),
        "score_aware_long_training_trained_state_exportable": report.get(
            "score_aware_long_training_trained_state_exportable"
        ),
        "local_mlx_prefilter_written": report.get("local_mlx_prefilter_written"),
        "local_mlx_prefilter_profile_path": report.get("local_mlx_prefilter_profile_path"),
        "lf_payload_report_status": lf_summary.get("report_status"),
        "lf_payload_packet_schema": lf_summary.get("schema"),
        "lf_payload_mode_histogram": lf_summary.get("mode_histogram"),
        "lf_payload_section_bytes": lf_summary.get("section_bytes"),
        "blockers": report.get("blockers"),
        "score_claim": report.get("score_claim"),
    }


def _packet_section_report_summary(packet: SnervArchivePacket) -> dict[str, Any]:
    lf = dict(packet.section_reports.get("lf_payload_codec_report") or {})
    keys = (
        "schema",
        "report_status",
        "section_name",
        "section_bytes",
        "packet_bytes",
        "raw_i64_bytes",
        "payload_bytes",
        "plane_count",
        "mode_histogram",
        "wrapper_histogram",
        "blockers",
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
    )
    return {
        "schema": "snerv_checkpoint_packet_section_report_summary.v1",
        "lf_payload_codec_report": {key: lf.get(key) for key in keys if key in lf},
        **FALSE_AUTHORITY,
    }


def _packet_metadata_summary(packet: SnervArchivePacket) -> dict[str, Any]:
    keys = (
        "decoder_payload_codec",
        "checkpoint_packetization_mode",
        "allocation_mode",
        "hf_decoder_fit_mode",
        "lf_payload_codec",
        "lf_payload_codec_requested",
        "lf_payload_codec_selected",
        "lf_payload_codec_selection_report",
        "snerv_model_size_adapter",
        "snerv_official_mfu_hfr_tub_numeric_primitives_requested",
        "snerv_official_mfu_hfr_tub_export_bound",
        "snerv_official_mfu_hfr_tub_export_bound_semantics",
        "snerv_official_mfu_hfr_tub_receiver_payload_bound",
        "snerv_official_mfu_hfr_tub_source_forward_replay_bound",
        "snerv_official_mfu_hfr_tub_source_forward_replay_authority",
        "snerv_official_mfu_hfr_tub_frame_producing_export",
        "official_skip_high_mode",
        "official_skip_high_full_shape",
        "official_skip_high_export_storage_shape",
        "official_skip_high_export_is_compact_train_state",
        "checkpoint_state_key_count",
        "checkpoint_trained_state_exportable",
        "source_faithful_stack",
        "official_source_parity_blockers",
        "score_aware_long_training",
        "score_aware_long_training_executed",
        "score_aware_long_training_trained_state_exportable",
        "native_mlx_training_kind",
        "score_aware_long_training_kind",
        "checkpoint_export_schema",
    )
    return {
        "schema": "snerv_checkpoint_packet_metadata_summary.v1",
        **{key: packet.metadata.get(key) for key in keys if key in packet.metadata},
        **FALSE_AUTHORITY,
    }


if __name__ == "__main__":
    raise SystemExit(main())
