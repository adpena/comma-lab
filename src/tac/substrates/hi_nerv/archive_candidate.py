# SPDX-License-Identifier: MIT
"""Byte-closed HiNeRV archive export helpers for MLX/local training artifacts.

This module is the receiver/bundling half of the MLX HiNeRV adapter.  It
bridges the MLX renderer's PyTorch-layout ``export_state_dict()`` into the HIV1
archive grammar, writes a contest-shaped ``archive.zip``, projects the payload
into the HPRC representation spine for byte-value accounting, and emits the
shared archive-bound receiver proof/package.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tac.framework_agnostic.helpers import (
    npz_to_numpy_primitives,
    write_npz_bridge_artifact,
)
from tac.local_acceleration.mlx_numpy_portability_contract import (
    build_mlx_numpy_portability_contract,
)
from tac.optimization.archive_bound_candidate_runtime_bridge import (
    emit_archive_bound_candidate_runtime_package,
)
from tac.repo_io import sha256_file
from tac.submission_archive import (
    MINIMAL_SINGLE_MEMBER_NAME,
    build_minimal_single_member_archive_bytes,
)
from tac.substrates._shared.inflate_runtime import CAMERA_HW, rgb_pair_to_uint8_frames
from tac.substrates._shared.pact_nerv_full_main import write_contest_runtime
from tac.substrates.hi_nerv.architecture import (
    HinervConfig,
    HinervSubstrate,
    validate_decoder_state_dict,
)
from tac.substrates.hi_nerv.archive import (
    build_archive_section_telemetry,
    pack_archive,
    parse_archive,
)
from tac.substrates.hi_nerv.bitstream import (
    prepare_hi_nerv_decoder_bitstream_state,
)
from tac.substrates.hi_nerv.target_region_actions import (
    TARGET_REGION_ACTION_META_KEY,
    decode_target_region_actions_from_meta,
    target_region_action_payload_codec,
    target_region_action_section_telemetry_for_payload,
    wrap_model_with_target_region_actions,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.hprc.representation_spine import (
    build_hi_nerv_spine_from_archive_payload,
    write_representation_spine_projection,
)

HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "hi_nerv_mlx_archive_bound_adapter_package.v1"
)
HI_NERV_MLX_RECEIVER_PROOF_SCHEMA = "hi_nerv_mlx_generated_receiver_proof.v1"
HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_ID = "hi_nerv_mlx_archive_export"
HI_NERV_MLX_ARCHIVE_CANDIDATE_FAMILY = "hi_nerv_mlx"
HI_NERV_MLX_ARCHIVE_TRANSFORM_KIND = "hi_nerv_mlx_archive"
HI_NERV_DECODER_RENDERED_PIXEL_PROOF_SCHEMA = (
    "hi_nerv_decoder_preparation_rendered_pixel_proof.v1"
)
HI_NERV_MLX_LIVE_RECEIVER_EXPORT_PARITY_PROOF_SCHEMA = (
    "hi_nerv_mlx_live_receiver_export_parity_proof.v1"
)
HI_NERV_TARGET_REGION_ACTION_PARSEBACK_SURVIVAL_SCHEMA = (
    "hi_nerv_target_region_action_parseback_survival.v1"
)

_LATENT_KEYS = ("latents_coarse", "latents_mid", "latents_fine")
_STATE_NPZ_NAME = "hi_nerv_mlx_exported_state.npz"
_STATE_NPZ_MANIFEST_NAME = "hi_nerv_mlx_exported_state_npz_manifest.json"
_BITSTREAM_PREPARATION_REPORT_NAME = "hi_nerv_bitstream_preparation.json"
_ARCHIVE_SECTION_TELEMETRY_NAME = "hi_nerv_archive_section_telemetry.json"
_LIVE_RECEIVER_EXPORT_PARITY_NAME = "hi_nerv_mlx_live_receiver_export_parity.json"
_LIVE_RECEIVER_CODEC_PORTFOLIO_SELECTION_NAME = (
    "hi_nerv_live_receiver_codec_portfolio_selection.json"
)
_LIVE_RECEIVER_CODEC_PORTFOLIO_SELECTION_SCHEMA = (
    "hi_nerv_live_receiver_codec_portfolio_selection.v1"
)
_PORTFOLIO_AUTO_CODEC_ALIASES = frozenset({"auto", "portfolio_auto", "int8_auto"})
_LIVE_RECEIVER_CODEC_PORTFOLIO_CANDIDATES = (
    "fp16_enveloped",
    "int8_mixed",
    "int8_scale_bundled",
    "int7_mixed",
    "int7_scale_bundled",
    "int6_mixed",
    "int6_scale_bundled",
    "int4_mixed",
    "int4_scale_bundled",
    "int2_mixed",
    "int2_scale_bundled",
)


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_hi_nerv_archive_replay_components(
    archive_path: str | Path,
    batch: Any,
    *,
    target_rgb_0: Any,
    target_rgb_1: Any,
    scorer_teacher: Any | None = None,
    pose_scorer_teacher: Any | None = None,
    candidate_kind: str = "",
) -> dict[str, float]:
    """Return receiver-parse-back components for live/EMA archive selection.

    This is local false-authority evidence, but the tensor source is the
    charged ``archive.zip`` bytes parsed through the HiNeRV receiver grammar.
    Score-unit components use the upstream evaluator coefficients when the
    real scorer teachers are attached; health-only metrics are prefixed with
    ``selection_health_`` so the shared selector can sort by non-collapse
    without double-counting them in the proxy score.
    """

    path = Path(archive_path).expanduser().resolve(strict=False)
    payload = _read_hiv1_payload_from_archive_zip(path)
    arc = parse_archive(payload)
    cfg = _hinerv_config_from_archive_meta(arc)
    pair_indices = _local_pair_indices_from_batch(batch)
    if pair_indices.size == 0:
        raise ValueError("HiNeRV archive replay requires at least one pair index")
    max_pair = int(pair_indices.max())
    if max_pair >= int(arc.latents_fine.shape[0]):
        raise ValueError(
            "HiNeRV archive replay pair index exceeds archive latent rows: "
            f"max_pair={max_pair} num_pairs={int(arc.latents_fine.shape[0])}"
        )
    target0 = _target_rows_nhwc01(target_rgb_0, pair_indices, "target_rgb_0")
    target1 = _target_rows_nhwc01(target_rgb_1, pair_indices, "target_rgb_1")
    receiver_model = _load_receiver_model_for_pixel_proof(
        cfg=cfg,
        decoder_state=arc.decoder_state_dict,
        latents_coarse=arc.latents_coarse,
        latents_mid=arc.latents_mid,
        latents_fine=arc.latents_fine,
        meta=arc.meta,
    )
    torch_pair_indices = torch.tensor(pair_indices.tolist(), dtype=torch.long)
    with torch.no_grad():
        receiver_pixels = _render_receiver_pixels(receiver_model, torch_pair_indices)
    receiver = np.asarray(receiver_pixels.detach().cpu(), dtype=np.float32)
    receiver_nhwc = np.transpose(receiver, (0, 1, 3, 4, 2))
    target_pair = np.stack([target0, target1], axis=1).astype(np.float32, copy=False)
    if receiver_nhwc.shape != target_pair.shape:
        raise ValueError(
            "HiNeRV archive replay target/receiver shape mismatch: "
            f"receiver={receiver_nhwc.shape} target={target_pair.shape}"
        )
    diff = receiver_nhwc - target_pair
    frame0_mse = float(np.mean(diff[:, 0] * diff[:, 0]))
    frame1_mse = float(np.mean(diff[:, 1] * diff[:, 1]))
    pair_mse = 0.5 * (frame0_mse + frame1_mse)
    temporal_delta = receiver_nhwc[:, 1] - receiver_nhwc[:, 0]
    out: dict[str, float] = {
        "archive_replay_pair_count": float(pair_indices.size),
        "archive_replay_archive_bytes": float(path.stat().st_size),
        "archive_replay_payload_bytes": float(len(payload)),
        "archive_replay_candidate_is_ema": (
            1.0 if str(candidate_kind).lower() == "ema" else 0.0
        ),
        "parseback_rgb_pair_mse": pair_mse,
        "parseback_rgb_frame0_mse": frame0_mse,
        "parseback_rgb_frame1_mse": frame1_mse,
        "selection_health_parseback_rgb_std": float(np.std(receiver_nhwc)),
        "selection_health_parseback_rgb_dynamic_range": float(
            np.max(receiver_nhwc) - np.min(receiver_nhwc)
        ),
        "selection_health_parseback_rgb_temporal_delta_std": float(
            np.std(temporal_delta)
        ),
        "selection_health_parseback_rgb_temporal_delta_mean_abs": float(
            np.mean(np.abs(temporal_delta))
        ),
    }
    _attach_segnet_archive_replay_components(
        out,
        receiver_nhwc[:, 1],
        scorer_teacher=scorer_teacher,
        pair_indices=pair_indices,
        batch=batch,
    )
    _attach_posenet_archive_replay_components(
        out,
        receiver_nhwc,
        pose_scorer_teacher=pose_scorer_teacher,
        pair_indices=pair_indices,
        batch=batch,
    )
    return {key: value for key, value in out.items() if math.isfinite(float(value))}


def build_hi_nerv_target_region_action_parseback_survival(
    archive_path: str | Path,
    *,
    expected_program_base64: str | None = None,
    expected_support_sha256: str | None = None,
    expected_payload_bytes: int | None = None,
    inflated_raw_path: str | Path | None = None,
) -> dict[str, Any]:
    """Prove charged target-region action survival through HIV1 parse-back.

    This parses the exported archive, decodes the charged action sidecar,
    renders the same receiver with and without the sidecar, and verifies that
    every encoded support pixel is overwritten with the exact uint8 RGB action
    value. It is receiver evidence, not score authority.
    """

    path = Path(archive_path).expanduser().resolve(strict=False)

    def _blocked(blocker: str, **extra: Any) -> dict[str, Any]:
        return {
            "schema": HI_NERV_TARGET_REGION_ACTION_PARSEBACK_SURVIVAL_SCHEMA,
            "surface": "parseback_mlx",
            "archive_path": path.as_posix(),
            "survived": False,
            "fakequant_survived": False,
            "parseback_survived": False,
            "inflate_survived": False,
            "blockers": [blocker],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            **extra,
            **FALSE_AUTHORITY,
        }

    if not path.is_file():
        return _blocked("target_region_action_archive_zip_missing")
    archive_sha256 = sha256_file(path)
    try:
        payload = _read_hiv1_payload_from_archive_zip(path)
        arc = parse_archive(payload)
    except Exception as exc:
        return _blocked(
            "target_region_action_archive_parseback_failed",
            archive_sha256=archive_sha256,
            archive_bytes=int(path.stat().st_size),
            failure=f"{type(exc).__name__}:{exc}",
        )

    meta = dict(getattr(arc, "meta", {}) or {})
    raw_b64 = meta.get(TARGET_REGION_ACTION_META_KEY)
    if not isinstance(raw_b64, str) or not raw_b64:
        return _blocked(
            "target_region_action_meta_missing",
            archive_sha256=archive_sha256,
            archive_bytes=int(path.stat().st_size),
        )
    try:
        stored_payload = base64.b64decode(raw_b64.encode("ascii"), validate=True)
        actions = decode_target_region_actions_from_meta(meta)
    except Exception as exc:
        return _blocked(
            "target_region_action_meta_decode_failed",
            archive_sha256=archive_sha256,
            archive_bytes=int(path.stat().st_size),
            failure=f"{type(exc).__name__}:{exc}",
        )
    if not actions:
        return _blocked(
            "target_region_action_empty",
            archive_sha256=archive_sha256,
            archive_bytes=int(path.stat().st_size),
        )
    stored_program_sha256 = hashlib.sha256(raw_b64.encode("ascii")).hexdigest()

    telemetry = target_region_action_section_telemetry_for_payload(actions, stored_payload)
    telemetry["program_base64_sha256"] = stored_program_sha256
    telemetry["payload_codec"] = target_region_action_payload_codec(stored_payload)
    telemetry["base64_text_bytes"] = len(raw_b64.encode("ascii"))
    blockers: list[str] = []
    blockers.extend(
        str(blocker)
        for blocker in telemetry.get("interpretation_blockers", [])
        if str(blocker)
    )
    expected_payload_sha256: str | None = None
    if expected_program_base64:
        try:
            expected_payload = base64.b64decode(
                str(expected_program_base64).encode("ascii"),
                validate=True,
            )
            expected_payload_sha256 = hashlib.sha256(expected_payload).hexdigest()
        except Exception as exc:
            expected_payload = b""
            blockers.append(
                f"target_region_action_expected_program_decode_failed:{type(exc).__name__}"
            )
        if expected_payload and expected_payload != stored_payload:
            blockers.append("target_region_action_payload_mismatch")
    if expected_support_sha256 and str(expected_support_sha256) != str(
        telemetry.get("support_sha256") or ""
    ):
        blockers.append("target_region_action_support_sha256_mismatch")
    if expected_payload_bytes is not None and int(expected_payload_bytes) != len(stored_payload):
        blockers.append("target_region_action_payload_bytes_mismatch")

    pair_indices = torch.tensor(
        sorted({int(action.pair_index) for action in actions}),
        dtype=torch.long,
    )
    pair_to_batch = {int(pair): index for index, pair in enumerate(pair_indices.tolist())}
    try:
        cfg = _hinerv_config_from_archive_meta(arc)
        action_model = _load_receiver_model_for_pixel_proof(
            cfg=cfg,
            decoder_state=arc.decoder_state_dict,
            latents_coarse=arc.latents_coarse,
            latents_mid=arc.latents_mid,
            latents_fine=arc.latents_fine,
            meta=meta,
        )
        base_meta = dict(meta)
        base_meta.pop(TARGET_REGION_ACTION_META_KEY, None)
        base_model = _load_receiver_model_for_pixel_proof(
            cfg=cfg,
            decoder_state=arc.decoder_state_dict,
            latents_coarse=arc.latents_coarse,
            latents_mid=arc.latents_mid,
            latents_fine=arc.latents_fine,
            meta=base_meta,
        )
        with torch.no_grad():
            action_pixels = _render_receiver_pixels(action_model, pair_indices)
            base_pixels = _render_receiver_pixels(base_model, pair_indices)
    except Exception as exc:
        return _blocked(
            "target_region_action_receiver_render_failed",
            archive_sha256=archive_sha256,
            archive_bytes=int(path.stat().st_size),
            target_region_actions=telemetry,
            failure=f"{type(exc).__name__}:{exc}",
        )

    action_np = action_pixels.detach().cpu().numpy().astype(np.float32, copy=False)
    base_np = base_pixels.detach().cpu().numpy().astype(np.float32, copy=False)
    total_pixels = 0
    exact_applied_pixels = 0
    changed_pixels = 0
    max_abs_error = 0.0
    max_abs_base_delta = 0.0
    for action in actions:
        batch_index = pair_to_batch[int(action.pair_index)]
        frame_index = int(action.frame_index)
        y = action.yx[:, 0].astype(np.int64, copy=False)
        x = action.yx[:, 1].astype(np.int64, copy=False)
        observed = action_np[batch_index, frame_index, :, y, x]
        base_values = base_np[batch_index, frame_index, :, y, x]
        if observed.shape[0] == 3 and observed.ndim == 2:
            observed = np.transpose(observed, (1, 0))
            base_values = np.transpose(base_values, (1, 0))
        expected = action.rgb_u8.astype(np.float32) / 255.0
        abs_error = np.abs(observed - expected)
        base_delta = np.abs(observed - base_values)
        total_pixels += int(action.pixel_count)
        exact_applied_pixels += int(np.count_nonzero(np.all(abs_error <= 1.0e-7, axis=1)))
        changed_pixels += int(np.count_nonzero(np.any(base_delta > 0.0, axis=1)))
        if abs_error.size:
            max_abs_error = max(max_abs_error, float(abs_error.max()))
        if base_delta.size:
            max_abs_base_delta = max(max_abs_base_delta, float(base_delta.max()))

    parseback_survived = bool(total_pixels > 0 and exact_applied_pixels == total_pixels)
    if not parseback_survived:
        blockers.append("target_region_action_parseback_survival_failed")
    inflate = _target_region_action_inflated_raw_survival(
        action_model=action_model,
        base_model=base_model,
        pair_indices=pair_indices,
        cfg=cfg,
        inflated_raw_path=inflated_raw_path,
    )
    blockers.extend(str(blocker) for blocker in inflate.get("blockers") or [])
    inflate_survived = bool(inflate.get("inflate_survived") is True)
    return {
        "schema": HI_NERV_TARGET_REGION_ACTION_PARSEBACK_SURVIVAL_SCHEMA,
        "surface": "inflate_raw" if inflate.get("inflated_raw_checked") else "parseback_mlx",
        "archive_path": path.as_posix(),
        "archive_sha256": archive_sha256,
        "archive_bytes": int(path.stat().st_size),
        "hiv1_payload_bytes": len(payload),
        "target_region_actions": telemetry,
        "support_sha256": telemetry.get("support_sha256"),
        "decoded_support_sha256": telemetry.get("decoded_support_sha256"),
        "decoded_action_sha256": telemetry.get("decoded_action_sha256"),
        "encoded_program_sha256": telemetry.get("encoded_program_sha256"),
        "action_count": len(actions),
        "pair_indices": [int(value) for value in pair_indices.tolist()],
        "total_action_pixels": int(total_pixels),
        "exact_uint8_action_pixels_applied": int(exact_applied_pixels),
        "receiver_changed_action_pixels": int(changed_pixels),
        "max_abs_action_rgb_error": float(max_abs_error),
        "max_abs_receiver_delta_vs_no_action": float(max_abs_base_delta),
        "expected_payload_sha256": expected_payload_sha256,
        "stored_payload_sha256": hashlib.sha256(stored_payload).hexdigest(),
        "target_region_action_program_sha256": stored_program_sha256,
        "expected_support_sha256": expected_support_sha256,
        "expected_payload_bytes": expected_payload_bytes,
        "survived": parseback_survived,
        "fakequant_survived": parseback_survived,
        "parseback_survived": parseback_survived,
        **inflate,
        "inflate_survived": inflate_survived,
        "blockers": _dedupe_strings(
            [
                *blockers,
                *([] if inflate_survived else ["target_region_action_inflate_survival_missing"]),
            ]
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _target_region_action_camera_uint8_pairs(
    model: torch.nn.Module,
    pair_indices: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        rgb_0, rgb_1 = model(pair_indices)
    frames: list[np.ndarray] = []
    for index in range(int(pair_indices.numel())):
        frames.append(
            rgb_pair_to_uint8_frames(
                rgb_0[index : index + 1],
                rgb_1[index : index + 1],
                input_range="unit",
            )
        )
    if not frames:
        return np.empty((0, 2, CAMERA_HW[0], CAMERA_HW[1], 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def _target_region_action_inflated_raw_survival(
    *,
    action_model: torch.nn.Module,
    base_model: torch.nn.Module,
    pair_indices: torch.Tensor,
    cfg: HinervConfig,
    inflated_raw_path: str | Path | None,
) -> dict[str, Any]:
    if inflated_raw_path is None:
        return {
            "inflated_raw_checked": False,
            "inflate_survived": False,
            "blockers": ["target_region_action_inflate_survival_missing"],
        }
    raw_path = Path(inflated_raw_path).expanduser().resolve(strict=False)
    if not raw_path.is_file():
        return {
            "inflated_raw_checked": True,
            "inflated_raw_path": raw_path.as_posix(),
            "inflate_survived": False,
            "blockers": ["target_region_action_inflated_raw_missing"],
        }
    frame_bytes = int(CAMERA_HW[0]) * int(CAMERA_HW[1]) * 3
    expected_bytes = int(cfg.num_pairs) * 2 * frame_bytes
    actual_bytes = int(raw_path.stat().st_size)
    if actual_bytes != expected_bytes:
        return {
            "inflated_raw_checked": True,
            "inflated_raw_path": raw_path.as_posix(),
            "inflated_raw_sha256": sha256_file(raw_path),
            "inflated_raw_bytes": actual_bytes,
            "expected_inflated_raw_bytes": expected_bytes,
            "inflate_survived": False,
            "blockers": ["target_region_action_inflated_raw_size_mismatch"],
        }
    try:
        action_pairs = _target_region_action_camera_uint8_pairs(action_model, pair_indices)
        base_pairs = _target_region_action_camera_uint8_pairs(base_model, pair_indices)
        raw = np.memmap(
            raw_path,
            dtype=np.uint8,
            mode="r",
            shape=(int(cfg.num_pairs) * 2, int(CAMERA_HW[0]), int(CAMERA_HW[1]), 3),
        )
        raw_pairs = np.stack(
            [
                np.stack(
                    [
                        np.asarray(raw[int(pair) * 2], dtype=np.uint8),
                        np.asarray(raw[int(pair) * 2 + 1], dtype=np.uint8),
                    ],
                    axis=0,
                )
                for pair in pair_indices.tolist()
            ],
            axis=0,
        )
    except Exception as exc:
        return {
            "inflated_raw_checked": True,
            "inflated_raw_path": raw_path.as_posix(),
            "inflated_raw_sha256": sha256_file(raw_path),
            "inflated_raw_bytes": actual_bytes,
            "inflate_survived": False,
            "failure": f"{type(exc).__name__}:{exc}",
            "blockers": ["target_region_action_inflated_raw_compare_failed"],
        }
    abs_error = np.abs(
        raw_pairs.astype(np.int16, copy=False) - action_pairs.astype(np.int16, copy=False)
    )
    action_delta = np.not_equal(action_pairs, base_pairs)
    raw_matches_action = bool(np.array_equal(raw_pairs, action_pairs))
    changed_values = int(np.count_nonzero(action_delta))
    changed_pixels = int(np.count_nonzero(np.any(action_delta, axis=-1)))
    inflate_survived = bool(raw_matches_action and changed_pixels > 0)
    blockers: list[str] = []
    if not raw_matches_action:
        blockers.append("target_region_action_inflated_raw_mismatch")
    if changed_pixels <= 0:
        blockers.append("target_region_action_inflated_raw_no_action_delta")
    return {
        "inflated_raw_checked": True,
        "inflated_raw_path": raw_path.as_posix(),
        "inflated_raw_sha256": sha256_file(raw_path),
        "inflated_raw_bytes": actual_bytes,
        "expected_inflated_raw_bytes": expected_bytes,
        "inflated_raw_pair_indices": [int(pair) for pair in pair_indices.tolist()],
        "inflated_raw_matches_action_receiver": raw_matches_action,
        "inflated_raw_action_changed_values": changed_values,
        "inflated_raw_action_changed_pixels": changed_pixels,
        "inflated_raw_total_pair_pixels": int(raw_pairs.shape[0] * raw_pairs.shape[1] * raw_pairs.shape[2] * raw_pairs.shape[3]),
        "inflated_raw_max_abs_error_vs_action_receiver": int(abs_error.max()) if abs_error.size else 0,
        "inflate_survived": inflate_survived,
        "blockers": blockers,
    }


def hi_nerv_mlx_numpy_portability_contract(
    *,
    canonical_npz_bridge_used: bool = True,
    training_backend: str = "mlx",
    latent_codec: str = "int16_raw",
) -> dict[str, Any]:
    """Return the honest portability contract for the current HiNeRV receiver."""

    backend = str(training_backend)
    normalized_latent_codec = str(latent_codec)
    receiver_dependencies = ["torch", "brotli", "python_stdlib"]
    notes = (
        "HiNeRV MLX export is NumPy-array backed, but the contest receiver "
        "currently decodes with PyTorch. This is contest-compliant when "
        "dependency closure passes, but not pure NumPy inflate."
    )
    if normalized_latent_codec == "int16_hi_ac_brotli_q11":
        receiver_dependencies.insert(2, "constriction")
        notes += (
            " The selected high-byte arithmetic latent codec also requires "
            "the constriction range-coder dependency at inflate time."
        )
    if normalized_latent_codec.startswith(("int8_", "int4_", "int2_")):
        notes += (
            " The selected lower-bit latent codec is receiver-bound and lossy "
            "relative to the int16 latent quantizer; scorer-domain replay is "
            "required before promotion."
        )
    return build_mlx_numpy_portability_contract(
        substrate_id="hi_nerv",
        training_backend=backend,
        exported_state_kind=f"pytorch_layout_numpy_arrays_from_{backend}_model",
        archive_payload_kind="hiv1_monolithic_0_bin",
        receiver_runtime_kind="torch_decode_receiver",
        receiver_dependencies=tuple(receiver_dependencies),
        numpy_array_export=True,
        canonical_npz_bridge_used=canonical_npz_bridge_used,
        pure_numpy_inflate=False,
        notes=notes,
    )


def _expected_receiver_output_bytes(cfg: HinervConfig) -> int:
    return int(cfg.num_pairs) * 2 * int(CAMERA_HW[0]) * int(CAMERA_HW[1]) * 3


def hi_nerv_meta_from_config(cfg: HinervConfig) -> dict[str, object]:
    """Return the minimal receiver metadata needed to rebuild the decoder."""

    return {
        "embed_dim": int(cfg.embed_dim),
        "initial_grid_h": int(cfg.initial_grid_h),
        "initial_grid_w": int(cfg.initial_grid_w),
        "decoder_channels": [int(value) for value in cfg.decoder_channels],
        "sin_frequency": float(cfg.sin_frequency),
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "mid_injection_block_index": int(cfg.mid_injection_block_index),
        "fine_injection_block_index": int(cfg.fine_injection_block_index),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
        "use_hierarchical_feature_grid": bool(cfg.use_hierarchical_feature_grid),
        "use_convnext_blocks": bool(cfg.use_convnext_blocks),
        "local_grid_levels": int(cfg.local_grid_levels),
        "local_grid_channels": int(cfg.local_grid_channels),
        "convnext_mlp_ratio": int(cfg.convnext_mlp_ratio),
        "convnext_kernel_size": int(cfg.convnext_kernel_size),
        "init_seed": int(getattr(cfg, "init_seed", 0)),
    }


def _hi_nerv_meta_with_target_region_actions(
    cfg: HinervConfig,
    *,
    target_region_action_program_base64: str | None = None,
) -> dict[str, object]:
    meta = hi_nerv_meta_from_config(cfg)
    if target_region_action_program_base64 in (None, ""):
        return meta
    if not isinstance(target_region_action_program_base64, str):
        raise TypeError(
            "target_region_action_program_base64 must be base64 text when supplied"
        )
    meta[TARGET_REGION_ACTION_META_KEY] = target_region_action_program_base64
    # Validate against the receiver grammar before bytes are packed.
    decode_target_region_actions_from_meta(dict(meta))
    return meta


def _read_hiv1_payload_from_archive_zip(archive_zip_path: Path) -> bytes:
    if not archive_zip_path.is_file():
        raise FileNotFoundError(f"HiNeRV archive.zip missing: {archive_zip_path}")
    with zipfile.ZipFile(archive_zip_path, "r") as zf:
        file_names = [name for name in zf.namelist() if not name.endswith("/")]
        receiver_names = [
            name
            for name in file_names
            if name in {"0.bin", MINIMAL_SINGLE_MEMBER_NAME}
        ]
        candidates = receiver_names or file_names
        if len(candidates) != 1:
            raise ValueError(
                "HiNeRV archive replay requires exactly one receiver payload "
                f"member; got {file_names}"
            )
        return zf.read(candidates[0])


def _hinerv_config_from_archive_meta(arc: Any) -> HinervConfig:
    meta = dict(getattr(arc, "meta", {}) or {})
    decoder_channels = meta.get("decoder_channels")
    if not isinstance(decoder_channels, Sequence) or isinstance(
        decoder_channels,
        (str, bytes),
    ):
        raise ValueError("HiNeRV archive replay meta missing decoder_channels")
    return HinervConfig(
        latent_dim_coarse=int(arc.latents_coarse.shape[1]),
        latent_dim_mid=int(arc.latents_mid.shape[1]),
        latent_dim_fine=int(arc.latents_fine.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(value) for value in decoder_channels),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        mid_injection_block_index=int(meta["mid_injection_block_index"]),
        fine_injection_block_index=int(meta["fine_injection_block_index"]),
        num_pairs=int(arc.latents_fine.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
        use_hierarchical_feature_grid=bool(
            meta.get("use_hierarchical_feature_grid", False)
        ),
        use_convnext_blocks=bool(meta.get("use_convnext_blocks", False)),
        local_grid_levels=int(meta.get("local_grid_levels", 1)),
        local_grid_channels=int(meta.get("local_grid_channels", 0)),
        convnext_mlp_ratio=int(meta.get("convnext_mlp_ratio", 2)),
        convnext_kernel_size=int(meta.get("convnext_kernel_size", 3)),
        init_seed=int(meta.get("init_seed", 0)),
    )


def _local_pair_indices_from_batch(batch: Any) -> np.ndarray:
    source = batch
    if isinstance(batch, Mapping):
        source = _first_mapping_value(
            batch,
            ("local_pair_indices", "pair_indices", "indices"),
        )
        if source is None:
            raise ValueError(
                "HiNeRV archive replay batch mapping must contain one of "
                "local_pair_indices, pair_indices, or indices"
            )
    arr = np.asarray(source, dtype=np.int64).reshape(-1)
    if np.any(arr < 0):
        raise ValueError(f"HiNeRV archive replay pair indices must be >= 0: {arr}")
    return np.ascontiguousarray(arr.astype(np.int64, copy=False))


def _target_rows_nhwc01(target: Any, pair_indices: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim != 4 or int(arr.shape[-1]) != 3:
        raise ValueError(f"{name} must be NHWC RGB; got shape={arr.shape}")
    max_pair = int(pair_indices.max()) if pair_indices.size else -1
    if max_pair >= int(arr.shape[0]):
        raise ValueError(
            f"{name} has {arr.shape[0]} rows but replay requested pair {max_pair}"
        )
    rows = np.ascontiguousarray(arr[pair_indices])
    if not np.all(np.isfinite(rows)):
        raise ValueError(f"{name} contains non-finite replay rows")
    return rows


def _attach_segnet_archive_replay_components(
    out: dict[str, float],
    candidate_rgb_1_nhwc01: np.ndarray,
    *,
    scorer_teacher: Any | None,
    pair_indices: np.ndarray,
    batch: Any,
) -> None:
    if scorer_teacher is None:
        return
    live_logits_fn = getattr(
        scorer_teacher,
        "teacher_logits_for_frames_nhwc01",
        None,
    )
    if not callable(live_logits_fn):
        return
    import mlx.core as mx

    candidate = mx.array(candidate_rgb_1_nhwc01, dtype=mx.float32)
    candidate_logits = live_logits_fn(candidate)
    mx.eval(candidate_logits)
    candidate_argmax = np.asarray(mx.argmax(candidate_logits, axis=-1), dtype=np.int64)
    target_argmax = _segnet_target_argmax_for_batch(
        scorer_teacher,
        batch=batch,
        pair_indices=pair_indices,
    )
    if target_argmax is None:
        return
    if tuple(candidate_argmax.shape) != tuple(target_argmax.shape):
        raise ValueError(
            "HiNeRV archive replay SegNet argmax shape mismatch: "
            f"candidate={candidate_argmax.shape} target={target_argmax.shape}"
        )
    num_classes = int(
        getattr(scorer_teacher, "num_classes", int(candidate_logits.shape[-1]))
    )
    d_seg = float(np.mean(candidate_argmax != target_argmax))
    out["parseback_segnet_argmax_disagreement_score_units"] = 100.0 * d_seg
    out["selection_health_segnet_direct_live_argmax_disagreement_rate"] = d_seg
    out[
        "selection_health_segnet_direct_live_candidate_occupied_class_fraction"
    ] = _occupied_class_fraction(candidate_argmax, num_classes=num_classes)
    coverage = _target_class_coverage(
        candidate_argmax,
        target_argmax,
        num_classes=num_classes,
    )
    out[
        "selection_health_segnet_direct_live_candidate_target_class_coverage_fraction"
    ] = coverage["coverage_fraction"]
    out[
        "selection_health_segnet_direct_live_candidate_target_any_class_coverage_fraction"
    ] = coverage["any_coverage_fraction"]
    out[
        "selection_health_segnet_direct_live_candidate_target_class_min_ratio"
    ] = coverage["min_ratio"]


def _segnet_target_argmax_for_batch(
    scorer_teacher: Any,
    *,
    batch: Any,
    pair_indices: np.ndarray,
) -> np.ndarray | None:
    import mlx.core as mx

    idx = batch
    if isinstance(idx, Mapping):
        idx = _first_mapping_value(idx, ("local_pair_indices", "pair_indices"))
        if idx is None:
            idx = pair_indices
    idx_mx = mx.array(np.asarray(idx, dtype=np.int32).reshape(-1), dtype=mx.int32)
    argmax_fn = getattr(scorer_teacher, "teacher_argmax_for_indices", None)
    if callable(argmax_fn):
        target = argmax_fn(idx_mx)
        mx.eval(target)
        return np.asarray(target, dtype=np.int64)
    logits_fn = getattr(scorer_teacher, "teacher_logits_for_indices", None)
    if callable(logits_fn):
        logits = logits_fn(idx_mx)
        mx.eval(logits)
        return np.asarray(mx.argmax(logits, axis=-1), dtype=np.int64)
    return None


def _occupied_class_fraction(values: np.ndarray, *, num_classes: int) -> float:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    if flat.size == 0 or num_classes <= 0:
        return 0.0
    counts = np.bincount(flat, minlength=num_classes)[:num_classes]
    min_pixels = max(2, math.ceil(float(flat.size) * 1.0e-3))
    return float(np.count_nonzero(counts >= min_pixels)) / float(num_classes)


def _target_class_coverage(
    candidate: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    ratios: list[float] = []
    any_hits = 0
    for cls in range(max(0, int(num_classes))):
        mask = target == cls
        total = int(np.count_nonzero(mask))
        if total <= 0:
            continue
        hit_count = int(np.count_nonzero((candidate == cls) & mask))
        ratio = float(hit_count) / float(total)
        ratios.append(ratio)
        if hit_count > 0:
            any_hits += 1
    if not ratios:
        return {
            "coverage_fraction": 0.0,
            "any_coverage_fraction": 0.0,
            "min_ratio": 0.0,
        }
    positive = [value for value in ratios if value > 0.0]
    return {
        "coverage_fraction": float(len(positive)) / float(len(ratios)),
        "any_coverage_fraction": float(any_hits) / float(len(ratios)),
        "min_ratio": float(min(ratios)),
    }


def _attach_posenet_archive_replay_components(
    out: dict[str, float],
    candidate_pair_nhwc01: np.ndarray,
    *,
    pose_scorer_teacher: Any | None,
    pair_indices: np.ndarray,
    batch: Any,
) -> None:
    if pose_scorer_teacher is None:
        return
    live_pose_fn = getattr(
        pose_scorer_teacher,
        "teacher_pose_for_yuv6_pair_nhwc",
        None,
    )
    target_pose_fn = getattr(pose_scorer_teacher, "teacher_pose_for_indices", None)
    if not callable(live_pose_fn) or not callable(target_pose_fn):
        return
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    rgb0 = mx.array(candidate_pair_nhwc01[:, 0], dtype=mx.float32)
    rgb1 = mx.array(candidate_pair_nhwc01[:, 1], dtype=mx.float32)
    yuv6_pair = mx.concatenate(
        [rgb_to_yuv6_mlx(rgb0 * 255.0), rgb_to_yuv6_mlx(rgb1 * 255.0)],
        axis=-1,
    )
    idx = batch
    if isinstance(idx, Mapping):
        idx = _first_mapping_value(idx, ("local_pair_indices", "pair_indices"))
        if idx is None:
            idx = pair_indices
    idx_mx = mx.array(np.asarray(idx, dtype=np.int32).reshape(-1), dtype=mx.int32)
    candidate_pose = live_pose_fn(yuv6_pair)
    target_pose = target_pose_fn(idx_mx)
    raw_diff = candidate_pose - target_pose
    raw_mse = mx.mean(raw_diff * raw_diff)
    score = mx.sqrt(10.0 * raw_mse + 1.0e-12)
    yuv6_delta = yuv6_pair[..., 6:12] - yuv6_pair[..., 0:6]
    mx.eval(raw_mse, score, yuv6_pair, yuv6_delta)
    raw_mse_f = float(raw_mse.item())
    out["parseback_posenet_direct_live_score_term"] = float(score.item())
    out["selection_health_parseback_posenet_direct_live_raw_mse"] = raw_mse_f
    out["selection_health_parseback_posenet_yuv6_pair_std"] = float(
        mx.std(yuv6_pair).item()
    )
    out["selection_health_parseback_posenet_yuv6_temporal_delta_std"] = float(
        mx.std(yuv6_delta).item()
    )


def _first_mapping_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _state_bridge_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / _STATE_NPZ_NAME, out_dir / _STATE_NPZ_MANIFEST_NAME


def _write_and_reload_exported_state_via_numpy_bridge(
    *,
    exported_state_dict: dict[str, np.ndarray],
    output_dir: Path,
    source_backend: str = "mlx",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Persist and reload the exact NumPy bridge consumed by the packer."""

    npz_path, manifest_path = _state_bridge_paths(output_dir)
    manifest = write_npz_bridge_artifact(
        exported_state_dict,
        npz_path,
        source_backend=str(source_backend),
        bridge_kind=f"hi_nerv_{source_backend}_export_state_dict_to_npz",
        manifest_path=manifest_path,
        require_finite=True,
    )
    if manifest.get("consumption_recommended") is not True:
        raise ValueError(
            "HiNeRV MLX export NPZ bridge is not consumption-recommended: "
            f"{manifest.get('blockers')}"
        )
    return npz_to_numpy_primitives(npz_path.read_bytes()), manifest


def _require_exported_tensor(
    exported_state_dict: dict[str, np.ndarray],
    key: str,
) -> torch.Tensor:
    if key not in exported_state_dict:
        raise ValueError(f"exported_state_dict missing {key!r}")
    return torch.from_numpy(np.asarray(exported_state_dict[key]).copy()).to(
        dtype=torch.float32
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().to("cpu").contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.asarray(arr.shape, dtype="<i8").tobytes())
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _decoder_state_sha256(decoder_state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(decoder_state):
        h.update(str(name).encode("utf-8"))
        h.update(b"\0")
        h.update(_tensor_sha256(decoder_state[name]).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _changed_decoder_tensors(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> list[str]:
    changed: list[str] = []
    for name in sorted(set(before) | set(after)):
        before_tensor = before.get(name)
        after_tensor = after.get(name)
        if before_tensor is None or after_tensor is None:
            changed.append(name)
            continue
        if _tensor_sha256(before_tensor) != _tensor_sha256(after_tensor):
            changed.append(name)
    return changed


def _load_receiver_model_for_pixel_proof(
    *,
    cfg: HinervConfig,
    decoder_state: Mapping[str, torch.Tensor],
    latents_coarse: torch.Tensor,
    latents_mid: torch.Tensor,
    latents_fine: torch.Tensor,
    meta: Mapping[str, Any] | None = None,
) -> torch.nn.Module:
    model = HinervSubstrate(cfg).eval()
    state = {
        name: tensor.detach().clone().to(dtype=torch.float32, device="cpu")
        for name, tensor in decoder_state.items()
    }
    state.update(
        {
            "latents_coarse": latents_coarse.detach()
            .clone()
            .to(dtype=torch.float32, device="cpu"),
            "latents_mid": latents_mid.detach()
            .clone()
            .to(dtype=torch.float32, device="cpu"),
            "latents_fine": latents_fine.detach()
            .clone()
            .to(dtype=torch.float32, device="cpu"),
        }
    )
    model.load_state_dict(state, strict=True)
    return wrap_model_with_target_region_actions(model, dict(meta or {}))


def _render_receiver_pixels(
    model: torch.nn.Module,
    pair_indices: torch.Tensor,
) -> torch.Tensor:
    rgb_0, rgb_1 = model(pair_indices)
    return (
        torch.stack((rgb_0, rgb_1), dim=1)
        .detach()
        .to(dtype=torch.float32, device="cpu")
        .contiguous()
    )


def _sample_pair_indices_for_pixel_proof(
    *,
    num_pairs: int,
    max_pair_samples: int,
) -> torch.Tensor:
    """Return deterministic spread samples for rendered-pixel mutation proof."""

    total = int(num_pairs)
    requested = int(max_pair_samples)
    if total <= 0:
        raise ValueError("HiNeRV rendered-pixel proof requires num_pairs > 0")
    if requested <= 0:
        raise ValueError("HiNeRV rendered-pixel proof requires max_pair_samples > 0")
    pair_count = min(requested, total)
    if pair_count == total:
        values = list(range(total))
    elif pair_count == 1:
        values = [0]
    else:
        values = [
            round(index * (total - 1) / float(pair_count - 1))
            for index in range(pair_count)
        ]
    return torch.tensor(values, dtype=torch.long)


def _render_mlx_live_pixels_unit(model: Any, pair_indices: torch.Tensor) -> np.ndarray:
    """Render the live MLX model as ``B,2,3,H,W`` unit RGB for export parity."""

    import mlx.core as mx

    idx = mx.array(np.asarray(pair_indices.tolist(), dtype=np.int32), dtype=mx.int32)
    live = model(idx)
    mx.eval(live)
    return (np.asarray(live, dtype=np.float32) / 255.0).copy()


def _build_mlx_live_receiver_export_parity_proof(
    *,
    model: Any,
    archive_bytes: bytes,
    cfg: HinervConfig,
    source_backend: str,
    latent_codec: str = "int16_raw",
    max_pair_samples: int = 4,
    max_mean_abs_delta: float = 1.0e-3,
    max_max_abs_delta: float = 5.0e-3,
) -> dict[str, Any]:
    """Compare live MLX pixels to parsed HIV1 receiver pixels after export."""

    pair_indices = _sample_pair_indices_for_pixel_proof(
        num_pairs=int(cfg.num_pairs),
        max_pair_samples=int(max_pair_samples),
    )
    proof: dict[str, Any] = {
        "schema": HI_NERV_MLX_LIVE_RECEIVER_EXPORT_PARITY_PROOF_SCHEMA,
        "proof_kind": "sampled_live_mlx_vs_parsed_hiv1_receiver_pixels",
        "latent_codec": str(latent_codec),
        "lossy_latent_codec": bool(_latent_codec_is_lossy(latent_codec)),
        "pair_indices": [int(value) for value in pair_indices.tolist()],
        "sampled_pair_count": int(pair_indices.numel()),
        "max_mean_abs_delta": float(max_mean_abs_delta),
        "max_max_abs_delta": float(max_max_abs_delta),
        "blockers": [
            "sampled_live_receiver_export_parity_not_full_video",
            "contest_cpu_cuda_exact_eval_not_executed",
            "scorer_replay_not_executed",
        ],
        **FALSE_AUTHORITY,
    }
    if source_backend != "mlx":
        proof.update(
            {
                "passed": False,
                "receiver_decode_passed": False,
                "proof_status": "not_applicable_non_mlx_source_backend",
                "source_backend": source_backend,
                "blockers": [
                    *proof["blockers"],
                    "hi_nerv_mlx_live_receiver_export_parity_not_applicable_non_mlx_source_backend",
                ],
            }
        )
        return proof
    try:
        live_pixels = _render_mlx_live_pixels_unit(model, pair_indices)
        arc = parse_archive(archive_bytes)
        receiver_model = _load_receiver_model_for_pixel_proof(
            cfg=cfg,
            decoder_state=arc.decoder_state_dict,
            latents_coarse=arc.latents_coarse,
            latents_mid=arc.latents_mid,
            latents_fine=arc.latents_fine,
            meta=arc.meta,
        )
        with torch.no_grad():
            receiver_pixels = _render_receiver_pixels(receiver_model, pair_indices)
        receiver_np = receiver_pixels.detach().cpu().numpy().astype(np.float32)
        if tuple(live_pixels.shape) != tuple(receiver_np.shape):
            proof.update(
                {
                    "passed": False,
                    "receiver_decode_passed": False,
                    "proof_status": "live_receiver_shape_mismatch",
                    "live_tensor_shape": [int(value) for value in live_pixels.shape],
                    "receiver_tensor_shape": [
                        int(value) for value in receiver_np.shape
                    ],
                    "blockers": [
                        *proof["blockers"],
                        "hi_nerv_mlx_live_receiver_export_shape_mismatch",
                    ],
                }
            )
            return proof
        delta = np.abs(live_pixels - receiver_np)
        max_abs_delta = float(delta.max()) if delta.size else 0.0
        mean_abs_delta = float(delta.mean()) if delta.size else 0.0
        passed = bool(
            mean_abs_delta <= float(max_mean_abs_delta)
            and max_abs_delta <= float(max_max_abs_delta)
        )
        lossy_latent_codec = bool(_latent_codec_is_lossy(latent_codec))
        measured_lossy_delta = bool(lossy_latent_codec and not passed)
        proof.update(
            {
                "passed": passed,
                "receiver_decode_passed": True,
                "lossy_latent_delta_measured": measured_lossy_delta,
                "proof_status": (
                    "sampled_live_receiver_export_parity_passed"
                    if passed
                    else "sampled_live_receiver_export_lossy_latent_delta_measured"
                    if measured_lossy_delta
                    else "sampled_live_receiver_export_parity_failed"
                ),
                "live_tensor_shape": [int(value) for value in live_pixels.shape],
                "receiver_tensor_shape": [int(value) for value in receiver_np.shape],
                "max_abs_delta": max_abs_delta,
                "mean_abs_delta": mean_abs_delta,
                "changed_element_count": int(np.count_nonzero(delta > 0.0)),
                "live_tensor_sha256": _sha256_numpy_array(live_pixels),
                "receiver_tensor_sha256": _sha256_numpy_array(receiver_np),
                "blockers": [
                    *proof["blockers"],
                    *(
                        []
                        if passed or measured_lossy_delta
                        else ["hi_nerv_mlx_live_receiver_export_parity_failed"]
                    ),
                ],
            }
        )
        return proof
    except Exception as exc:
        proof.update(
            {
                "passed": False,
                "receiver_decode_passed": False,
                "proof_status": "sampled_live_receiver_export_parity_error",
                "failure": repr(exc),
                "blockers": [
                    *proof["blockers"],
                    "hi_nerv_mlx_live_receiver_export_parity_error",
                ],
            }
        )
        return proof


def _sha256_numpy_array(array: np.ndarray) -> str:
    arr = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.asarray(arr.shape, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return h.hexdigest()


def _live_receiver_export_parity_extra_blockers(
    proof: Mapping[str, Any],
) -> list[str]:
    if proof.get("proof_status") == "sampled_live_receiver_export_parity_passed":
        return []
    return [
        str(blocker)
        for blocker in proof.get("blockers") or []
        if str(blocker).startswith("hi_nerv_mlx_live_receiver_export_")
    ]


def _live_receiver_codec_portfolio_extra_blockers(
    selection: Mapping[str, Any],
) -> list[str]:
    selected = selection.get("selected_row")
    selected_passed = (
        isinstance(selected, Mapping)
        and selected.get("live_receiver_export_parity_passed") is True
    )
    if selected_passed:
        return []
    return [
        str(blocker)
        for blocker in selection.get("blockers") or []
        if str(blocker).startswith("hi_nerv_live_receiver_codec_portfolio_")
    ]


def _build_decoder_rendered_pixel_proof(
    *,
    decoder_state_before: Mapping[str, torch.Tensor],
    decoder_state_after: Mapping[str, torch.Tensor],
    latents_coarse: torch.Tensor,
    latents_mid: torch.Tensor,
    latents_fine: torch.Tensor,
    cfg: HinervConfig,
    max_pair_samples: int = 3,
) -> dict[str, Any]:
    changed_names = _changed_decoder_tensors(decoder_state_before, decoder_state_after)
    pair_indices = _sample_pair_indices_for_pixel_proof(
        num_pairs=int(cfg.num_pairs),
        max_pair_samples=int(max_pair_samples),
    )
    proof: dict[str, Any] = {
        "schema": HI_NERV_DECODER_RENDERED_PIXEL_PROOF_SCHEMA,
        "proof_kind": "sampled_receiver_rendered_pixel_delta",
        "pair_indices": [int(value) for value in pair_indices.tolist()],
        "sampled_pair_count": int(pair_indices.numel()),
        "decoder_tensor_count": len(decoder_state_after),
        "changed_decoder_tensor_count": len(changed_names),
        "changed_decoder_tensor_names": changed_names,
        "decoder_state_sha256_before": _decoder_state_sha256(decoder_state_before),
        "decoder_state_sha256_after": _decoder_state_sha256(decoder_state_after),
        "blockers": [
            "sampled_rendered_pixel_proof_not_full_video",
            "contest_cpu_cuda_exact_eval_not_executed",
            "scorer_replay_not_executed",
        ],
        **FALSE_AUTHORITY,
    }
    if not changed_names:
        proof.update(
            {
                "proof_status": "not_required_no_decoder_state_change",
                "decoder_state_changed": False,
                "rendered_pixels_changed": False,
                "changed_rendered_pixel_count": 0,
                "max_abs_rendered_pixel_delta": 0.0,
                "mean_abs_rendered_pixel_delta": 0.0,
            }
        )
        return proof

    before_model = _load_receiver_model_for_pixel_proof(
        cfg=cfg,
        decoder_state=decoder_state_before,
        latents_coarse=latents_coarse,
        latents_mid=latents_mid,
        latents_fine=latents_fine,
    )
    after_model = _load_receiver_model_for_pixel_proof(
        cfg=cfg,
        decoder_state=decoder_state_after,
        latents_coarse=latents_coarse,
        latents_mid=latents_mid,
        latents_fine=latents_fine,
    )
    with torch.no_grad():
        before_pixels = _render_receiver_pixels(before_model, pair_indices)
        after_pixels = _render_receiver_pixels(after_model, pair_indices)
    delta = torch.abs(after_pixels - before_pixels)
    max_abs_delta = float(delta.max().item()) if delta.numel() else 0.0
    changed_pixel_count = int(torch.count_nonzero(delta > 0.0).item())
    rendered_pixels_changed = bool(changed_pixel_count > 0 and max_abs_delta > 0.0)
    proof.update(
        {
            "proof_status": (
                "sampled_rendered_pixels_changed"
                if rendered_pixels_changed
                else "sampled_rendered_pixels_no_change"
            ),
            "decoder_state_changed": True,
            "rendered_pixels_changed": rendered_pixels_changed,
            "changed_rendered_pixel_count": changed_pixel_count,
            "max_abs_rendered_pixel_delta": max_abs_delta,
            "mean_abs_rendered_pixel_delta": (
                float(delta.mean().item()) if delta.numel() else 0.0
            ),
            "rendered_tensor_shape": [int(value) for value in after_pixels.shape],
            "rendered_tensor_sha256_before": _tensor_sha256(before_pixels),
            "rendered_tensor_sha256_after": _tensor_sha256(after_pixels),
        }
    )
    return proof


def _bitstream_report_with_rendered_pixel_proof(
    *,
    prepared_report: Mapping[str, Any],
    decoder_state_before: Mapping[str, torch.Tensor],
    decoder_state_after: Mapping[str, torch.Tensor],
    latents_coarse: torch.Tensor,
    latents_mid: torch.Tensor,
    latents_fine: torch.Tensor,
    cfg: HinervConfig,
) -> dict[str, Any]:
    report = copy.deepcopy(dict(prepared_report))
    proof = _build_decoder_rendered_pixel_proof(
        decoder_state_before=decoder_state_before,
        decoder_state_after=decoder_state_after,
        latents_coarse=latents_coarse,
        latents_mid=latents_mid,
        latents_fine=latents_fine,
        cfg=cfg,
    )
    if proof["decoder_state_changed"] and not proof["rendered_pixels_changed"]:
        raise ValueError(
            "HiNeRV decoder bitstream preparation changed decoder tensors but "
            "sampled receiver rendered pixels did not change"
        )
    report["decoder_rendered_pixel_proof"] = proof
    waterfill = report.get("decoder_weight_waterfill")
    if isinstance(waterfill, dict):
        waterfill["rendered_pixel_proof"] = proof
        waterfill["rendered_pixel_proof_status"] = proof["proof_status"]
    return report


def pack_archive_from_exported_state_dict(
    *,
    exported_state_dict: dict[str, np.ndarray],
    cfg: HinervConfig,
    decoder_codec: str = "int8_mixed",
    pruning_ratio: float = 0.0,
    quant_noise_bits: int | None = None,
    quant_noise_scale: float = 0.0,
    quant_noise_seed: int = 0,
    decoder_weight_waterfill_plan: Mapping[str, Any] | None = None,
    latent_codec: str = "int16_raw",
    target_region_action_program_base64: str | None = None,
    return_bitstream_report: bool = False,
) -> bytes | tuple[bytes, dict[str, Any]]:
    """Pack PyTorch-layout exported MLX tensors into HIV1 ``0.bin`` bytes."""

    latents_coarse = _require_exported_tensor(exported_state_dict, "latents_coarse")
    latents_mid = _require_exported_tensor(exported_state_dict, "latents_mid")
    latents_fine = _require_exported_tensor(exported_state_dict, "latents_fine")
    expected_shapes = {
        "latents_coarse": (int(cfg.num_pairs), int(cfg.latent_dim_coarse)),
        "latents_mid": (int(cfg.num_pairs), int(cfg.latent_dim_mid)),
        "latents_fine": (int(cfg.num_pairs), int(cfg.latent_dim_fine)),
    }
    for key, tensor in (
        ("latents_coarse", latents_coarse),
        ("latents_mid", latents_mid),
        ("latents_fine", latents_fine),
    ):
        if tuple(int(v) for v in tensor.shape) != expected_shapes[key]:
            raise ValueError(
                f"{key} shape {tuple(tensor.shape)} != {expected_shapes[key]}"
            )

    decoder_state: dict[str, torch.Tensor] = {}
    for name, arr in exported_state_dict.items():
        if name in _LATENT_KEYS:
            continue
        decoder_state[name] = torch.from_numpy(np.asarray(arr).copy()).to(
            dtype=torch.float32
        )
    validate_decoder_state_dict(
        decoder_state,
        cfg,
        context="hi_nerv_exported_decoder_state",
    )
    prepared = prepare_hi_nerv_decoder_bitstream_state(
        decoder_state,
        pruning_ratio=pruning_ratio,
        quant_noise_bits=quant_noise_bits,
        quant_noise_scale=quant_noise_scale,
        quant_noise_seed=quant_noise_seed,
        decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
    )
    bitstream_report = _bitstream_report_with_rendered_pixel_proof(
        prepared_report=prepared.report,
        decoder_state_before=decoder_state,
        decoder_state_after=prepared.state_dict,
        latents_coarse=latents_coarse,
        latents_mid=latents_mid,
        latents_fine=latents_fine,
        cfg=cfg,
    )

    blob = pack_archive(
        prepared.state_dict,
        latents_coarse,
        latents_mid,
        latents_fine,
        _hi_nerv_meta_with_target_region_actions(
            cfg,
            target_region_action_program_base64=target_region_action_program_base64,
        ),
        decoder_codec=decoder_codec,
        latent_codec=latent_codec,
    )
    if return_bitstream_report:
        return blob, bitstream_report
    return blob


def _normalize_decoder_codec(codec: str) -> str:
    return str(codec).strip().lower()


def _latent_codec_is_lossy(codec: str) -> bool:
    return str(codec).startswith(("int8_", "int4_", "int2_"))


def _selection_row_float(row: Mapping[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _build_single_codec_portfolio_selection_report(
    *,
    requested_decoder_codec: str,
    selected_decoder_codec: str,
    latent_codec: str,
    archive_bytes: int,
    payload_bytes: int,
    archive_zip_build: Mapping[str, Any],
    live_receiver_export_parity: Mapping[str, Any],
    hard_byte_ceiling: int | None,
) -> dict[str, Any]:
    ceiling = None if hard_byte_ceiling is None else int(hard_byte_ceiling)
    receiver_survived = bool(
        live_receiver_export_parity.get("receiver_decode_passed")
        or live_receiver_export_parity.get("passed")
    )
    lossy_latent_codec = _latent_codec_is_lossy(latent_codec)
    row = {
        "decoder_codec_requested": str(requested_decoder_codec),
        "decoder_codec_emitted": str(selected_decoder_codec),
        "latent_codec": str(latent_codec),
        "lossy_latent_codec": bool(lossy_latent_codec),
        "payload_bytes": int(payload_bytes),
        "archive_bytes": int(archive_bytes),
        "archive_sha256": archive_zip_build.get("archive_sha256"),
        "archive_zip_build": dict(archive_zip_build),
        "under_hard_byte_ceiling": (
            None if ceiling is None else bool(int(archive_bytes) <= ceiling)
        ),
        "live_receiver_export_parity_passed": bool(
            live_receiver_export_parity.get("passed")
        ),
        "live_receiver_export_receiver_survived": bool(receiver_survived),
        "live_receiver_export_parity_status": live_receiver_export_parity.get(
            "proof_status"
        ),
        "mean_abs_delta": live_receiver_export_parity.get("mean_abs_delta"),
        "max_abs_delta": live_receiver_export_parity.get("max_abs_delta"),
        "blockers": list(live_receiver_export_parity.get("blockers") or []),
        **FALSE_AUTHORITY,
    }
    blockers: list[str] = [
        "hi_nerv_live_receiver_codec_portfolio_is_sampled_false_authority",
        "contest_cpu_cuda_exact_eval_not_executed",
        "full_video_scorer_value_replay_not_executed",
    ]
    if not bool(row["live_receiver_export_receiver_survived"]):
        blockers.append(
            "hi_nerv_live_receiver_codec_portfolio_selected_not_receiver_surviving"
        )
    if (
        not bool(row["live_receiver_export_parity_passed"])
        and not bool(lossy_latent_codec and receiver_survived)
    ):
        blockers.append(
            "hi_nerv_live_receiver_codec_portfolio_selected_codec_failed_parity"
        )
    if ceiling is not None and int(archive_bytes) > ceiling:
        blockers.append(
            "hi_nerv_live_receiver_codec_portfolio_selected_codec_over_hard_byte_ceiling"
        )
    return {
        "schema": _LIVE_RECEIVER_CODEC_PORTFOLIO_SELECTION_SCHEMA,
        "requested_decoder_codec": str(requested_decoder_codec),
        "selected_decoder_codec": str(selected_decoder_codec),
        "selected_decoder_codec_requested": str(requested_decoder_codec),
        "selected_decoder_codec_effective": str(selected_decoder_codec),
        "selected_decoder_codec_source": "archive_section_telemetry",
        "selection_mode": "single_codec_requested",
        "hard_byte_ceiling": ceiling,
        "candidate_count": 1,
        "measured_candidate_count": 1,
        "parity_passing_candidate_count": int(
            bool(row["live_receiver_export_parity_passed"])
        ),
        "receiver_surviving_candidate_count": int(
            bool(row["live_receiver_export_receiver_survived"])
        ),
        "selected_row": row,
        "rows": [row],
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _select_live_receiver_portfolio_archive(
    *,
    model: Any,
    exported_state_dict: Mapping[str, np.ndarray],
    cfg: HinervConfig,
    requested_decoder_codec: str,
    source_backend: str,
    pruning_ratio: float,
    quant_noise_bits: int | None,
    quant_noise_scale: float,
    quant_noise_seed: int,
    decoder_weight_waterfill_plan: Mapping[str, Any] | None,
    latent_codec: str,
    target_region_action_program_base64: str | None,
    hard_byte_ceiling: int | None,
) -> tuple[bytes, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Pick the cheapest receiver-pixel-preserving codec for ``portfolio_auto``.

    Rate-only ``portfolio_auto`` can choose int4/int2 packets that are tiny but
    destroy the live scorer state after receiver decode.  This archive-bound
    selector measures charged ZIP bytes and the existing live-MLX vs parsed-HIV1
    receiver pixel proof for each candidate, then selects the cheapest
    parity-passing codec under the byte ceiling.  If none passes, it selects the
    lowest-error diagnostic row under the ceiling so the run remains inspectable
    but carries explicit blockers.
    """

    ceiling = None if hard_byte_ceiling is None else int(hard_byte_ceiling)
    rows: list[dict[str, Any]] = []
    candidate_payloads: dict[str, tuple[bytes, dict[str, Any], dict[str, Any]]] = {}
    for codec in _LIVE_RECEIVER_CODEC_PORTFOLIO_CANDIDATES:
        try:
            bin_bytes, bitstream_report = pack_archive_from_exported_state_dict(
                exported_state_dict=dict(exported_state_dict),
                cfg=cfg,
                decoder_codec=codec,
                pruning_ratio=pruning_ratio,
                quant_noise_bits=quant_noise_bits,
                quant_noise_scale=quant_noise_scale,
                quant_noise_seed=quant_noise_seed,
                decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
                latent_codec=latent_codec,
                target_region_action_program_base64=target_region_action_program_base64,
                return_bitstream_report=True,
            )
            archive_zip_bytes, archive_zip_build = (
                build_minimal_single_member_archive_bytes(bin_bytes)
            )
            live_receiver_export_parity = _build_mlx_live_receiver_export_parity_proof(
                model=model,
                archive_bytes=bin_bytes,
                cfg=cfg,
                source_backend=source_backend,
                latent_codec=latent_codec,
            )
            archive_section_telemetry = build_archive_section_telemetry(
                bin_bytes,
                archive_zip_bytes=len(archive_zip_bytes),
            )
            emitted_codec = str(archive_section_telemetry.get("decoder_codec") or codec)
            receiver_survived = bool(
                live_receiver_export_parity.get("receiver_decode_passed")
                or live_receiver_export_parity.get("passed")
            )
            row = {
                "decoder_codec_requested": codec,
                "decoder_codec_emitted": emitted_codec,
                "latent_codec": str(latent_codec),
                "lossy_latent_codec": bool(_latent_codec_is_lossy(latent_codec)),
                "status": "measured",
                "payload_bytes": len(bin_bytes),
                "archive_bytes": len(archive_zip_bytes),
                "archive_sha256": archive_zip_build.get("archive_sha256"),
                "archive_zip_build": dict(archive_zip_build),
                "under_hard_byte_ceiling": (
                    None
                    if ceiling is None
                    else bool(len(archive_zip_bytes) <= ceiling)
                ),
                "live_receiver_export_parity_passed": bool(
                    live_receiver_export_parity.get("passed")
                ),
                "live_receiver_export_receiver_survived": bool(receiver_survived),
                "live_receiver_export_parity_status": (
                    live_receiver_export_parity.get("proof_status")
                ),
                "mean_abs_delta": live_receiver_export_parity.get("mean_abs_delta"),
                "max_abs_delta": live_receiver_export_parity.get("max_abs_delta"),
                "parity_blockers": list(
                    live_receiver_export_parity.get("blockers") or []
                ),
                **FALSE_AUTHORITY,
            }
            rows.append(row)
            candidate_payloads[codec] = (
                bin_bytes,
                bitstream_report,
                live_receiver_export_parity,
            )
        except Exception as exc:
            rows.append(
                {
                    "decoder_codec_requested": codec,
                    "status": "failed",
                    "failure": repr(exc),
                    "live_receiver_export_parity_passed": False,
                    "under_hard_byte_ceiling": None,
                    **FALSE_AUTHORITY,
                }
            )

    measured = [row for row in rows if row.get("status") == "measured"]
    under_ceiling = [
        row
        for row in measured
        if ceiling is None or int(row["archive_bytes"]) <= ceiling
    ]
    eligible_scope = under_ceiling or measured
    parity_passing = [
        row for row in eligible_scope if bool(row.get("live_receiver_export_parity_passed"))
    ]
    receiver_surviving = [
        row
        for row in eligible_scope
        if bool(row.get("live_receiver_export_receiver_survived"))
    ]
    if parity_passing:
        selected_row = min(parity_passing, key=lambda row: int(row["archive_bytes"]))
        selection_mode = "cheapest_live_receiver_parity_passing_codec"
    elif receiver_surviving and _latent_codec_is_lossy(latent_codec):
        selected_row = min(
            receiver_surviving,
            key=lambda row: (
                _selection_row_float(row, "mean_abs_delta", float("inf")),
                _selection_row_float(row, "max_abs_delta", float("inf")),
                int(row["archive_bytes"]),
            ),
        )
        selection_mode = "lowest_error_receiver_surviving_lossy_latent_codec"
    elif eligible_scope:
        selected_row = min(
            eligible_scope,
            key=lambda row: (
                _selection_row_float(row, "mean_abs_delta", float("inf")),
                _selection_row_float(row, "max_abs_delta", float("inf")),
                int(row["archive_bytes"]),
            ),
        )
        selection_mode = "lowest_error_diagnostic_codec_no_parity_pass"
    else:
        selected_row = None
        selection_mode = "no_measured_codec"

    blockers: list[str] = [
        "hi_nerv_live_receiver_codec_portfolio_is_sampled_false_authority",
        "contest_cpu_cuda_exact_eval_not_executed",
        "full_video_scorer_value_replay_not_executed",
    ]
    if not measured:
        blockers.append("hi_nerv_live_receiver_codec_portfolio_no_measured_codec")
    if ceiling is not None and measured and not under_ceiling:
        blockers.append(
            "hi_nerv_live_receiver_codec_portfolio_no_candidate_under_hard_byte_ceiling"
        )
    if (
        measured
        and not parity_passing
        and not (_latent_codec_is_lossy(latent_codec) and receiver_surviving)
    ):
        blockers.append(
            "hi_nerv_live_receiver_codec_portfolio_no_parity_passing_codec"
        )
    if selected_row is not None and not bool(
        selected_row.get("live_receiver_export_receiver_survived")
    ):
        blockers.append(
            "hi_nerv_live_receiver_codec_portfolio_selected_not_receiver_surviving"
        )
    if (
        selected_row is not None
        and not bool(selected_row.get("live_receiver_export_parity_passed"))
        and not bool(
            _latent_codec_is_lossy(latent_codec)
            and selected_row.get("live_receiver_export_receiver_survived")
        )
    ):
        blockers.append(
            "hi_nerv_live_receiver_codec_portfolio_selected_codec_failed_parity"
        )
    selected_requested_codec = (
        str(selected_row["decoder_codec_requested"]) if selected_row is not None else None
    )
    selected_effective_codec = (
        str(selected_row.get("decoder_codec_emitted") or selected_requested_codec)
        if selected_row is not None
        else None
    )
    if (
        selected_requested_codec is None
        or selected_requested_codec not in candidate_payloads
        or selected_effective_codec is None
    ):
        raise ValueError(
            "HiNeRV live-receiver codec portfolio could not select a measured codec"
        )
    bin_bytes, bitstream_report, live_receiver_export_parity = candidate_payloads[
        selected_requested_codec
    ]
    archive_zip_bytes, archive_zip_build = build_minimal_single_member_archive_bytes(
        bin_bytes
    )
    selection_report = {
        "schema": _LIVE_RECEIVER_CODEC_PORTFOLIO_SELECTION_SCHEMA,
        "requested_decoder_codec": str(requested_decoder_codec),
        "selected_decoder_codec": selected_effective_codec,
        "selected_decoder_codec_requested": selected_requested_codec,
        "selected_decoder_codec_effective": selected_effective_codec,
        "selected_decoder_codec_source": "archive_section_telemetry",
        "selection_mode": selection_mode,
        "hard_byte_ceiling": ceiling,
        "candidate_count": len(rows),
        "measured_candidate_count": len(measured),
        "candidate_under_hard_byte_ceiling_count": len(under_ceiling),
        "parity_passing_candidate_count": len(parity_passing),
        "receiver_surviving_candidate_count": len(receiver_surviving),
        "selected_row": dict(selected_row),
        "rows": rows,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }
    return (
        bin_bytes,
        bitstream_report,
        live_receiver_export_parity,
        archive_zip_build,
        selection_report,
    )


def export_hi_nerv_mlx_archive(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
    decoder_codec: str = "int8_mixed",
    source_backend: str = "mlx",
    pruning_ratio: float = 0.0,
    quant_noise_bits: int | None = None,
    quant_noise_scale: float = 0.0,
    quant_noise_seed: int = 0,
    decoder_weight_waterfill_plan: Mapping[str, Any] | None = None,
    latent_codec: str = "int16_raw",
    target_region_action_program_base64: str | None = None,
    hard_byte_ceiling: int | None = None,
) -> tuple[Path, str, int]:
    """Export an MLX HiNeRV model as a contest-shaped ``archive.zip``."""

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = model.cfg
    exported_state_dict, npz_bridge_manifest = (
        _write_and_reload_exported_state_via_numpy_bridge(
            exported_state_dict=model.export_state_dict(),
            output_dir=out_dir,
            source_backend=source_backend,
        )
    )
    requested_decoder_codec = str(decoder_codec)
    if (
        _normalize_decoder_codec(requested_decoder_codec)
        in _PORTFOLIO_AUTO_CODEC_ALIASES
        and str(source_backend) == "mlx"
    ):
        (
            bin_bytes,
            bitstream_report,
            live_receiver_export_parity,
            archive_zip_build,
            live_receiver_codec_portfolio_selection,
        ) = _select_live_receiver_portfolio_archive(
            model=model,
            exported_state_dict=exported_state_dict,
            cfg=cfg,
            requested_decoder_codec=requested_decoder_codec,
            source_backend=source_backend,
            pruning_ratio=pruning_ratio,
            quant_noise_bits=quant_noise_bits,
            quant_noise_scale=quant_noise_scale,
            quant_noise_seed=quant_noise_seed,
            decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
            latent_codec=latent_codec,
            target_region_action_program_base64=target_region_action_program_base64,
            hard_byte_ceiling=hard_byte_ceiling,
        )
        effective_decoder_codec = str(
            live_receiver_codec_portfolio_selection["selected_decoder_codec"]
        )
    else:
        bin_bytes, bitstream_report = pack_archive_from_exported_state_dict(
            exported_state_dict=exported_state_dict,
            cfg=cfg,
            decoder_codec=requested_decoder_codec,
            pruning_ratio=pruning_ratio,
            quant_noise_bits=quant_noise_bits,
            quant_noise_scale=quant_noise_scale,
            quant_noise_seed=quant_noise_seed,
            decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
            latent_codec=latent_codec,
            target_region_action_program_base64=target_region_action_program_base64,
            return_bitstream_report=True,
        )
        live_receiver_export_parity = _build_mlx_live_receiver_export_parity_proof(
            model=model,
            archive_bytes=bin_bytes,
            cfg=cfg,
            source_backend=source_backend,
            latent_codec=latent_codec,
        )
        archive_zip_bytes, archive_zip_build = build_minimal_single_member_archive_bytes(
            bin_bytes
        )
        archive_section_probe = build_archive_section_telemetry(
            bin_bytes,
            archive_zip_bytes=len(archive_zip_bytes),
        )
        effective_decoder_codec = str(
            archive_section_probe.get("decoder_codec") or requested_decoder_codec
        )
        live_receiver_codec_portfolio_selection = (
            _build_single_codec_portfolio_selection_report(
                requested_decoder_codec=requested_decoder_codec,
                selected_decoder_codec=effective_decoder_codec,
                latent_codec=latent_codec,
                archive_bytes=len(archive_zip_bytes),
                payload_bytes=len(bin_bytes),
                archive_zip_build=archive_zip_build,
                live_receiver_export_parity=live_receiver_export_parity,
                hard_byte_ceiling=hard_byte_ceiling,
            )
        )
    bitstream_report = {
        **dict(bitstream_report),
        "decoder_codec": effective_decoder_codec,
        "latent_codec": latent_codec,
        "requested_decoder_codec": requested_decoder_codec,
        "decoder_codec_requested_by_export": requested_decoder_codec,
        "decoder_codec_selected_by_export": effective_decoder_codec,
        "live_receiver_codec_portfolio_selection": (
            live_receiver_codec_portfolio_selection
        ),
        "live_receiver_codec_portfolio_selection_schema": (
            _LIVE_RECEIVER_CODEC_PORTFOLIO_SELECTION_SCHEMA
        ),
    }
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)
    bitstream_report_path = out_dir / _BITSTREAM_PREPARATION_REPORT_NAME
    bitstream_report_path.write_text(
        json.dumps(bitstream_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    live_receiver_export_parity_path = out_dir / _LIVE_RECEIVER_EXPORT_PARITY_NAME
    live_receiver_export_parity_path.write_text(
        json.dumps(live_receiver_export_parity, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    live_receiver_codec_portfolio_selection_path = (
        out_dir / _LIVE_RECEIVER_CODEC_PORTFOLIO_SELECTION_NAME
    )
    live_receiver_codec_portfolio_selection_path.write_text(
        json.dumps(
            live_receiver_codec_portfolio_selection,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    submission_dir = out_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="hi_nerv",
        repo_root=root,
        runtime_module_files=(
            "architecture.py",
            "archive.py",
            "inflate.py",
            "target_region_actions.py",
        ),
        vendor_shared_inflate_runtime=True,
    )
    (submission_dir / "0.bin").write_bytes(bin_bytes)
    archive_zip_path = out_dir / "archive.zip"
    archive_zip_bytes, archive_zip_build_fresh = build_minimal_single_member_archive_bytes(
        bin_bytes
    )
    if archive_zip_build_fresh.get("archive_sha256") != archive_zip_build.get(
        "archive_sha256"
    ):
        archive_zip_build = archive_zip_build_fresh
    archive_zip_path.write_bytes(archive_zip_bytes)
    archive_sha256 = sha256_file(archive_zip_path)
    archive_bytes = archive_zip_path.stat().st_size
    archive_zip_build = {
        **archive_zip_build,
        "path": archive_zip_path.as_posix(),
        "bytes": int(archive_bytes),
        "sha256": archive_sha256,
    }
    archive_section_telemetry = build_archive_section_telemetry(
        bin_bytes,
        archive_zip_bytes=int(archive_bytes),
    )
    archive_section_decoder_codec = str(
        archive_section_telemetry.get("decoder_codec") or ""
    )
    if archive_section_decoder_codec != str(effective_decoder_codec):
        raise ValueError(
            "HiNeRV effective decoder codec custody mismatch: "
            f"selection/report={effective_decoder_codec!r} "
            f"archive_section_telemetry={archive_section_decoder_codec!r}"
        )
    archive_section_telemetry_path = out_dir / _ARCHIVE_SECTION_TELEMETRY_NAME
    archive_section_telemetry_path.write_text(
        json.dumps(archive_section_telemetry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if hard_byte_ceiling is not None:
        ceiling = int(hard_byte_ceiling)
        if ceiling <= 0:
            raise ValueError("hard_byte_ceiling must be positive when supplied")
        if int(archive_bytes) > ceiling:
            raise ValueError(
                "HiNeRV archive exceeds hard_byte_ceiling: "
                f"{archive_bytes} > {ceiling}"
            )
    write_representation_spine_projection(
        output_dir=out_dir,
        spine=build_hi_nerv_spine_from_archive_payload(
            bin_bytes,
            source={
                "kind": "hi_nerv_export_payload",
                "archive_zip_path": archive_zip_path.as_posix(),
                "archive_zip_sha256": archive_sha256,
                "archive_zip_bytes": int(archive_bytes),
            },
            manifest_extra={
                "emitted_by": "export_hi_nerv_mlx_archive",
                "archive_bytes_are_authority_for_rate": True,
                "archive_zip_build": archive_zip_build,
                "archive_zip_payload_only": True,
                "runtime_source_outside_archive_zip": True,
                "upstream_evaluate_rate_uses_archive_zip_stat_only": True,
                "decoder_codec": effective_decoder_codec,
                "requested_decoder_codec": requested_decoder_codec,
                "latent_codec": latent_codec,
                "archive_section_telemetry": archive_section_telemetry,
                "archive_section_telemetry_path": (
                    archive_section_telemetry_path.as_posix()
                ),
                "hi_nerv_live_receiver_codec_portfolio_selection": (
                    live_receiver_codec_portfolio_selection
                ),
                "hi_nerv_live_receiver_codec_portfolio_selection_path": (
                    live_receiver_codec_portfolio_selection_path.as_posix()
                ),
                "hi_nerv_bitstream_preparation": bitstream_report,
                "hi_nerv_bitstream_preparation_path": (
                    bitstream_report_path.as_posix()
                ),
                "hi_nerv_mlx_live_receiver_export_parity": (
                    live_receiver_export_parity
                ),
                "hi_nerv_mlx_live_receiver_export_parity_path": (
                    live_receiver_export_parity_path.as_posix()
                ),
                "num_pairs": int(cfg.num_pairs),
                "state_npz_bridge": {
                    "artifact_path": npz_bridge_manifest["artifact_path"],
                    "artifact_sha256": npz_bridge_manifest["artifact_sha256"],
                    "manifest_path": npz_bridge_manifest["manifest_path"],
                    "tensor_count": npz_bridge_manifest["tensor_count"],
                },
                "export_source_backend": str(source_backend),
            },
        ),
        basename="hprc_representation_spine_hi_nerv",
    )
    if emit_archive_bound_candidate_package:
        emit_archive_bound_candidate_runtime_package(
            adapter_id=HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_ID,
            candidate_family=HI_NERV_MLX_ARCHIVE_CANDIDATE_FAMILY,
            candidate_id_prefix="hi_nerv_mlx",
            transform_kind=HI_NERV_MLX_ARCHIVE_TRANSFORM_KIND,
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            receiver_contract_kind="hi_nerv_mlx_generated_inflate_sh_decode_only_receiver",
            proof_schema=HI_NERV_MLX_RECEIVER_PROOF_SCHEMA,
            proof_filename="hi_nerv_mlx_receiver_proof.json",
            candidate_label="hi_nerv",
            expected_receiver_output_name="0.raw",
            expected_receiver_output_bytes=_expected_receiver_output_bytes(cfg),
            retain_receiver_output=retain_receiver_proof_output,
            runtime_adapter_manifest_extra={
                "schema": "hi_nerv_mlx_runtime_adapter_manifest.v1",
                "latent_pyramid": ["coarse", "mid", "fine"],
                "decoder_codec": effective_decoder_codec,
                "requested_decoder_codec": requested_decoder_codec,
                "latent_codec": latent_codec,
                "archive_zip_build": archive_zip_build,
                "archive_zip_payload_only": True,
                "runtime_source_outside_archive_zip": True,
                "upstream_evaluate_rate_uses_archive_zip_stat_only": True,
                "archive_section_telemetry": archive_section_telemetry,
                "archive_section_telemetry_path": (
                    archive_section_telemetry_path.as_posix()
                ),
                "hi_nerv_live_receiver_codec_portfolio_selection": (
                    live_receiver_codec_portfolio_selection
                ),
                "hi_nerv_live_receiver_codec_portfolio_selection_path": (
                    live_receiver_codec_portfolio_selection_path.as_posix()
                ),
                "hi_nerv_bitstream_preparation": bitstream_report,
                "hi_nerv_bitstream_preparation_path": (
                    bitstream_report_path.as_posix()
                ),
                "hi_nerv_mlx_live_receiver_export_parity": (
                    live_receiver_export_parity
                ),
                "hi_nerv_mlx_live_receiver_export_parity_path": (
                    live_receiver_export_parity_path.as_posix()
                ),
                "num_pairs": int(cfg.num_pairs),
                "state_npz_bridge_manifest": npz_bridge_manifest,
                "mlx_numpy_portability_contract": (
                    hi_nerv_mlx_numpy_portability_contract(
                        training_backend=source_backend,
                        latent_codec=latent_codec,
                    )
                ),
            },
            candidate_row_schema="hi_nerv_mlx_archive_bound_candidate_row.v1",
            wrapper_schema=HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
            extra_blockers=[
                *_live_receiver_export_parity_extra_blockers(
                    live_receiver_export_parity
                ),
                *_live_receiver_codec_portfolio_extra_blockers(
                    live_receiver_codec_portfolio_selection
                ),
            ],
            mlx_triage_argv=mlx_triage_argv,
        )
    return (archive_zip_path, archive_sha256, archive_bytes)


def export_hi_nerv_mlx_archive_bound_candidate_package(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
    decoder_codec: str = "int8_mixed",
    source_backend: str = "mlx",
    pruning_ratio: float = 0.0,
    quant_noise_bits: int | None = None,
    quant_noise_scale: float = 0.0,
    quant_noise_seed: int = 0,
    decoder_weight_waterfill_plan: Mapping[str, Any] | None = None,
    latent_codec: str = "int16_raw",
    target_region_action_program_base64: str | None = None,
    hard_byte_ceiling: int | None = None,
) -> dict[str, Any]:
    """Export HiNeRV MLX bytes and emit the shared candidate package."""

    archive_zip_path, archive_sha256, archive_bytes = export_hi_nerv_mlx_archive(
        model,
        output_dir,
        repo_root=repo_root,
        emit_archive_bound_candidate_package=False,
        decoder_codec=decoder_codec,
        source_backend=source_backend,
        pruning_ratio=pruning_ratio,
        quant_noise_bits=quant_noise_bits,
        quant_noise_scale=quant_noise_scale,
        quant_noise_seed=quant_noise_seed,
        decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
        latent_codec=latent_codec,
        target_region_action_program_base64=target_region_action_program_base64,
        hard_byte_ceiling=hard_byte_ceiling,
    )
    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    cfg = model.cfg
    _, npz_bridge_manifest_path = _state_bridge_paths(out_dir)
    npz_bridge_manifest = json.loads(
        npz_bridge_manifest_path.read_text(encoding="utf-8")
    )
    bitstream_report_path = out_dir / _BITSTREAM_PREPARATION_REPORT_NAME
    bitstream_report = json.loads(bitstream_report_path.read_text(encoding="utf-8"))
    live_receiver_export_parity_path = out_dir / _LIVE_RECEIVER_EXPORT_PARITY_NAME
    live_receiver_export_parity = json.loads(
        live_receiver_export_parity_path.read_text(encoding="utf-8")
    )
    live_receiver_codec_portfolio_selection_path = (
        out_dir / _LIVE_RECEIVER_CODEC_PORTFOLIO_SELECTION_NAME
    )
    live_receiver_codec_portfolio_selection = json.loads(
        live_receiver_codec_portfolio_selection_path.read_text(encoding="utf-8")
    )
    archive_section_telemetry_path = out_dir / _ARCHIVE_SECTION_TELEMETRY_NAME
    archive_section_telemetry = json.loads(
        archive_section_telemetry_path.read_text(encoding="utf-8")
    )
    effective_decoder_codec = str(
        archive_section_telemetry.get("decoder_codec")
        or live_receiver_codec_portfolio_selection.get("selected_decoder_codec")
        or decoder_codec
    )
    requested_decoder_codec = str(
        live_receiver_codec_portfolio_selection.get("requested_decoder_codec")
        or decoder_codec
    )
    return emit_archive_bound_candidate_runtime_package(
        adapter_id=HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_ID,
        candidate_family=HI_NERV_MLX_ARCHIVE_CANDIDATE_FAMILY,
        candidate_id_prefix="hi_nerv_mlx",
        transform_kind=HI_NERV_MLX_ARCHIVE_TRANSFORM_KIND,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=out_dir / "submission",
        output_dir=out_dir,
        repo_root=root,
        receiver_contract_kind="hi_nerv_mlx_generated_inflate_sh_decode_only_receiver",
        proof_schema=HI_NERV_MLX_RECEIVER_PROOF_SCHEMA,
        proof_filename="hi_nerv_mlx_receiver_proof.json",
        candidate_label="hi_nerv",
        expected_receiver_output_name="0.raw",
        expected_receiver_output_bytes=_expected_receiver_output_bytes(cfg),
        retain_receiver_output=retain_receiver_proof_output,
        runtime_adapter_manifest_extra={
            "schema": "hi_nerv_mlx_runtime_adapter_manifest.v1",
            "latent_pyramid": ["coarse", "mid", "fine"],
            "decoder_codec": effective_decoder_codec,
            "requested_decoder_codec": requested_decoder_codec,
            "latent_codec": latent_codec,
            "archive_section_telemetry": archive_section_telemetry,
            "archive_section_telemetry_path": archive_section_telemetry_path.as_posix(),
            "hi_nerv_live_receiver_codec_portfolio_selection": (
                live_receiver_codec_portfolio_selection
            ),
            "hi_nerv_live_receiver_codec_portfolio_selection_path": (
                live_receiver_codec_portfolio_selection_path.as_posix()
            ),
            "hi_nerv_bitstream_preparation": bitstream_report,
            "hi_nerv_bitstream_preparation_path": bitstream_report_path.as_posix(),
            "hi_nerv_mlx_live_receiver_export_parity": live_receiver_export_parity,
            "hi_nerv_mlx_live_receiver_export_parity_path": (
                live_receiver_export_parity_path.as_posix()
            ),
            "num_pairs": int(cfg.num_pairs),
            "state_npz_bridge_manifest": npz_bridge_manifest,
            "mlx_numpy_portability_contract": (
                hi_nerv_mlx_numpy_portability_contract(
                    training_backend=source_backend,
                    latent_codec=latent_codec,
                )
            ),
        },
        candidate_row_schema="hi_nerv_mlx_archive_bound_candidate_row.v1",
        wrapper_schema=HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        extra_blockers=[
            *_live_receiver_export_parity_extra_blockers(live_receiver_export_parity),
            *_live_receiver_codec_portfolio_extra_blockers(
                live_receiver_codec_portfolio_selection
            ),
        ],
        mlx_triage_argv=mlx_triage_argv,
    )


__all__ = [
    "HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_ID",
    "HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "HI_NERV_MLX_ARCHIVE_CANDIDATE_FAMILY",
    "HI_NERV_MLX_ARCHIVE_TRANSFORM_KIND",
    "HI_NERV_TARGET_REGION_ACTION_PARSEBACK_SURVIVAL_SCHEMA",
    "build_hi_nerv_target_region_action_parseback_survival",
    "export_hi_nerv_mlx_archive",
    "export_hi_nerv_mlx_archive_bound_candidate_package",
    "hi_nerv_meta_from_config",
    "hi_nerv_mlx_numpy_portability_contract",
    "pack_archive_from_exported_state_dict",
]
