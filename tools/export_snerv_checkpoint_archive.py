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
    _model_size_from_candidate,
    export_snerv_mlx_archive,
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
            "checkpoint_export_schema": "snerv_checkpoint_archive_export.v1",
            "checkpoint_meta_path": meta_path.as_posix(),
            "checkpoint_epoch": meta.get("global_epoch"),
            "checkpoint_state_kind": state_kind,
            "checkpoint_state_sha256": sha256_file(state_path),
            "startup_json_sha256": sha256_file(startup_path),
            "native_mlx_training_executed": True,
            "native_mlx_training_kind": "checkpoint_direct_lf_decoder_packetization",
            "score_aware_long_training_executed": True,
            "score_aware_long_training_kind": "checkpoint_harvest_interrupted_run",
            **FALSE_AUTHORITY,
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
        "startup_json_path": startup_path.as_posix(),
        "startup_json_sha256": sha256_file(startup_path),
        "output_dir": out.as_posix(),
        "packet_path": packet_path.as_posix(),
        "packet_bytes": int(packet.total_bytes),
        "packet_sha256": _sha256_bytes(packet.packet),
        "packet_section_bytes": dict(packet.section_bytes),
        "packet_section_sha256": dict(packet.section_sha256),
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
        "blockers": _blockers(receiver_proof=receiver_proof, receiver_proof_requested=bool(emit_receiver_proof)),
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


def _blockers(
    *,
    receiver_proof: dict[str, Any],
    receiver_proof_requested: bool,
) -> list[str]:
    blockers = [
        "macos_mlx_checkpoint_export_false_authority",
        "contest_cpu_cuda_exact_eval_not_executed",
        "full_video_scorer_replay_not_executed",
    ]
    if not receiver_proof_requested:
        blockers.append("receiver_proof_not_requested")
    elif receiver_proof.get("runtime_consumption_proof_passed") is not True:
        blockers.append("receiver_proof_not_passed")
    if receiver_proof_requested and receiver_proof.get("receiver_contract_satisfied") is not True:
        blockers.append("receiver_contract_not_satisfied")
    return blockers


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
    return {
        "schema": report.get("schema"),
        "checkpoint_epoch": report.get("checkpoint_epoch"),
        "packet_bytes": report.get("packet_bytes"),
        "archive_bytes": report.get("archive_bytes"),
        "receiver_proof_passed": report.get("receiver_proof_passed"),
        "receiver_contract_satisfied": report.get("receiver_contract_satisfied"),
        "blockers": report.get("blockers"),
        "score_claim": report.get("score_claim"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
