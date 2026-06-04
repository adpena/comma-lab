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
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_step_map_coder import encode_step_maps_waterfill  # noqa: E402
from tac.repo_io import sha256_file, write_json_artifact  # noqa: E402
from tac.substrates._shared.mlx_score_aware.targets import decode_mlx_targets  # noqa: E402
from tac.substrates._shared.numpy_portable_inflate import unpack_state_dict_numpy  # noqa: E402
from tac.substrates.snerv_inverse_steg_carrier.archive import (  # noqa: E402
    SnervArchivePacket,
    decode_snerv_archive_frames,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
    resolve_decoder_payload_codec,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (  # noqa: E402
    _DETAIL_KEYS,
    HfGenerationDecoder,
    SnervModelSizeConfig,
    quantize_lf,
)
from tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export import (  # noqa: E402
    SCORER_HW,
    _build_official_mfu_hfr_tub_packet_from_components,
    _model_size_from_candidate,
    _official_passthrough_mfu,
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
    state = unpack_state_dict_numpy(state_path.read_bytes())

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
        "checkpoint_state_sha256": sha256_file(state_path),
        "startup_json_sha256": sha256_file(startup_path),
        "native_mlx_training_executed": True,
        "score_aware_long_training_executed": True,
        "score_aware_long_training_kind": "checkpoint_harvest_interrupted_run",
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
    package: dict[str, Any] | None = None
    if emit_receiver_proof:
        package = export_snerv_mlx_archive(
            {"packet_path": packet_path.as_posix(), "packet_sha256": _sha256_bytes(packet.packet)},
            out / "snerv_checkpoint_archive_bound_package",
            repo_root=root,
            retain_receiver_output=bool(retain_receiver_proof_output),
            receiver_proof_timeout_seconds=int(receiver_proof_timeout_seconds),
        )
    receiver_proof = dict(package.get("receiver_proof") or {}) if package else {}
    archive_path = receiver_proof.get("archive_path") if receiver_proof else None
    archive_bytes = receiver_proof.get("archive_bytes") if receiver_proof else None
    archive_sha256 = receiver_proof.get("archive_sha256") if receiver_proof else None
    mlx_prefilter_profile = _maybe_write_receiver_decoded_mlx_prefilter(
        requested=bool(write_mlx_prefilter_profile),
        output_dir=out / "local_mlx_prefilter",
        packet=packet,
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
        "checkpoint_meta_sha256": sha256_file(meta_path),
        "checkpoint_epoch": meta.get("global_epoch"),
        "checkpoint_state_kind": state_kind,
        "checkpoint_state_path": state_path.as_posix(),
        "checkpoint_state_sha256": sha256_file(state_path),
        "modelsize_candidate": candidate,
        "startup_json_path": startup_path.as_posix(),
        "startup_json_sha256": sha256_file(startup_path),
        "output_dir": out.as_posix(),
        "packet_path": packet_path.as_posix(),
        "packet_bytes": int(packet.total_bytes),
        "packet_sha256": _sha256_bytes(packet.packet),
        "packet_section_bytes": dict(packet.section_bytes),
        "packet_section_sha256": dict(packet.section_sha256),
        "packet_metadata_summary": _packet_metadata_summary(packet),
        "archive_path": str(archive_path) if archive_path else None,
        "archive_bytes": int(archive_bytes) if archive_bytes is not None else None,
        "archive_sha256": str(archive_sha256) if archive_sha256 else None,
        "decoder_codec": resolved_decoder_codec,
        "decoder_codec_requested": requested_decoder_codec,
        "lf_payload_codec": resolved_lf_codec,
        "target_bits_per_coeff": float(target_bits_per_coeff),
        "step_map_bits_per_coeff": float(step_map_bits_per_coeff),
        "model_size": model_size.as_jsonable(),
        "receiver_proof_path": receiver_proof.get("proof_path") if receiver_proof else None,
        "receiver_proof_passed": receiver_proof.get("runtime_consumption_proof_passed") is True,
        "receiver_contract_satisfied": receiver_proof.get("receiver_contract_satisfied") is True,
        "local_mlx_prefilter_profile": mlx_prefilter_profile,
        "local_mlx_prefilter_profile_path": mlx_prefilter_profile.get("profile_path"),
        "local_mlx_prefilter_progress_path": mlx_prefilter_profile.get("progress_path"),
        "local_mlx_prefilter_written": mlx_prefilter_profile.get("written") is True,
        "blockers": _blockers(
            receiver_proof=receiver_proof,
            receiver_proof_requested=bool(emit_receiver_proof),
            mlx_prefilter_profile=mlx_prefilter_profile,
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
        "lf_payload_codec": str(lf_payload_codec),
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
        lf_payload=encode_lf_quant_payload(lf_quant_planes, codec=lf_payload_codec),
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
        "temporal_encoder_output_shape": (1, 4, max(1, ll_h // 2), max(1, ll_w // 2)),
        "fc_hw": (2, 2),
        "output2_decoder_output_shape": (2, 8, max(1, ll_h // 2), max(1, ll_w // 2)),
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


def _checkpoint_state_array(
    state: dict[str, np.ndarray],
    *keys: str,
) -> np.ndarray:
    for key in keys:
        if key in state:
            return np.asarray(state[key])
    joined = ", ".join(keys)
    raise ValueError(f"official checkpoint state missing any of: {joined}")


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
    base = {
        "schema": "snerv_checkpoint_receiver_decoded_mlx_prefilter_request.v1",
        "requested": bool(requested),
        "profile_path": (out / "local_mlx_prefilter_profile.json").as_posix(),
        "progress_path": (out / "local_mlx_prefilter_progress.jsonl").as_posix(),
        "packet_sha256": _sha256_bytes(packet.packet),
        "packet_bytes": int(packet.total_bytes),
        "receiver_decoded_selected_packet": True,
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
    output_hw = _packet_output_hw(metadata)
    if archive_bytes is None or archive_sha256 is None:
        return _write_snerv_native_receiver_decoded_mlx_prefilter(
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
            scorer_batch_pairs=scorer_batch_pairs,
            progress_every=progress_every,
            allow_overwrite=False,
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
    return _write_snerv_native_receiver_decoded_mlx_prefilter(
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
        scorer_batch_pairs=scorer_batch_pairs,
        progress_every=progress_every,
        allow_overwrite=False,
    )


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


def _blockers(
    *,
    receiver_proof: dict[str, Any],
    receiver_proof_requested: bool,
    mlx_prefilter_profile: dict[str, Any],
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
    return list(dict.fromkeys(blockers))


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


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    packet_metadata = dict(report.get("packet_metadata_summary") or {})
    return {
        "schema": report.get("schema"),
        "checkpoint_epoch": report.get("checkpoint_epoch"),
        "packet_bytes": report.get("packet_bytes"),
        "checkpoint_packetization_mode": packet_metadata.get(
            "checkpoint_packetization_mode"
        ),
        "decoder_payload_codec": packet_metadata.get("decoder_payload_codec"),
        "archive_bytes": report.get("archive_bytes"),
        "receiver_proof_passed": report.get("receiver_proof_passed"),
        "receiver_contract_satisfied": report.get("receiver_contract_satisfied"),
        "local_mlx_prefilter_written": report.get("local_mlx_prefilter_written"),
        "local_mlx_prefilter_profile_path": report.get("local_mlx_prefilter_profile_path"),
        "blockers": report.get("blockers"),
        "score_claim": report.get("score_claim"),
    }


def _packet_metadata_summary(packet: SnervArchivePacket) -> dict[str, Any]:
    keys = (
        "decoder_payload_codec",
        "checkpoint_packetization_mode",
        "allocation_mode",
        "hf_decoder_fit_mode",
        "snerv_model_size_adapter",
        "snerv_official_mfu_hfr_tub_numeric_primitives_requested",
        "snerv_official_mfu_hfr_tub_export_bound",
        "snerv_official_mfu_hfr_tub_frame_producing_export",
        "official_skip_high_mode",
        "official_skip_high_full_shape",
        "official_skip_high_export_storage_shape",
        "official_skip_high_export_is_compact_train_state",
        "source_faithful_stack",
        "official_source_parity_blockers",
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
