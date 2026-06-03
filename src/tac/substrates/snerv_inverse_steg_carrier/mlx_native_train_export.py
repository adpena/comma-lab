# SPDX-License-Identifier: MIT
"""Native MLX target-hydration to SNeRV SNAR1 archive export.

This is the first real adapter surface behind
``mlx_native_adapter_contract.py``.  It is deliberately honest about its
scope: MLX owns real-video target hydration and bridge custody; the current
decoder fit is the existing deterministic NumPy closed-form SNeRV fit, because
that is the receiver-portable training primitive already used by the advisory
lane.  Longer scorer-in-loop MLX optimization remains blocked until the runner
attaches real SegNet/PoseNet teachers and byte pressure inside the training
loop.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.adaptation.hard_pair_indices import normalize_pair_indices
from tac.analysis.snerv_step_map_coder import encode_step_maps_waterfill
from tac.contest_eval_contract import build_upstream_eval_contract
from tac.optimization.archive_bound_candidate_runtime_bridge import (
    run_generated_inflate_receiver_proof,
)
from tac.repo_io import sha256_file, write_bytes_artifact, write_json
from tac.substrates._shared.mlx_score_aware.bridge_drift import (
    build_mlx_numpy_bridge_drift_bundle,
    mlx_numpy_bridge_drift_report,
)
from tac.substrates._shared.mlx_score_aware.targets import decode_mlx_targets
from tac.substrates.snerv_inverse_steg_carrier.allocation import (
    LfSaliency,
    allocate_lf_linf,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    SnervArchivePacket,
    decode_snerv_archive_frames,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.archive_candidate import (
    CAMERA_HW,
    SNERV_RECEIVER_PROOF_SCHEMA,
    expected_receiver_output_bytes_from_metadata,
    export_snerv_archive_bound_candidate_package,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    _DETAIL_KEYS,
    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
    SNERV_SPECTRA_PRESERVING_ADAPTER,
    HfGenerationDecoder,
    SnervModelSizeConfig,
    _decoder_features,
    _hfr_for_model_size,
    _kernel_storage_shape,
    _upsample_nn,
    encode_frame_lf,
    fit_hf_decoder_least_squares,
    fit_hf_decoder_weighted_least_squares,
    official_snerv_modelsize_to_fc_dim,
    quantize_lf,
)
from tac.substrates.snerv_inverse_steg_carrier.dwt import (
    WaveletPyramid,
    dwt2_native_synthesis_adjoint,
    idwt2_multilevel,
)
from tac.substrates.snerv_inverse_steg_carrier.official_hfr import (
    OFFICIAL_SNERV_HFR_SOURCE_CONTRACT,
    OFFICIAL_SNERV_HFR_SOURCE_SHA,
    SNERV_OFFICIAL_HFR_CONVBLOCK_NUMPY_PROOF,
)
from tac.substrates.snerv_inverse_steg_carrier.official_mfu import (
    OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS,
    OFFICIAL_SNERV_MFU_SOURCE,
    OFFICIAL_SNERV_RB_SOURCE,
)
from tac.substrates.snerv_inverse_steg_carrier.official_tub import (
    OFFICIAL_SNERV_T_SOURCE_SHA,
    OFFICIAL_SNERV_T_TUB_SCHEMA,
    OFFICIAL_SNERV_T_TUB_SOURCE_CONTRACT,
)

SNERV_MLX_NATIVE_TRAIN_EXPORT_SCHEMA = "snerv_mlx_native_train_export.v1"
SNERV_MLX_NATIVE_PREFILTER_PROFILE_SCHEMA = "snerv_mlx_native_prefilter_profile.v1"
SNERV_MLX_NATIVE_STORAGE_PREFLIGHT_SCHEMA = "snerv_mlx_native_storage_preflight.v1"
SNERV_MLX_NATIVE_PACKET_FILENAME = "snerv_mlx_native_packet.snar"
SNERV_MLX_NATIVE_REPORT_FILENAME = "snerv_mlx_native_train_export.json"
SNERV_DWT_ADJOINT_SALIENCY_WEIGHTED_FIT_MODE = "joint_p18_p19_dwt_adjoint_saliency_weighted_least_squares"
SCORER_HW = (384, 512)

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
NATIVE_MLX_DECODER_LOSS_WORSEN_REL_TOL = 1.0e-7


class SnervMlxNativeExportError(ValueError):
    """Raised when the native MLX SNeRV adapter cannot build a valid export."""


@dataclass(frozen=True)
class SnervMlxNativeArtifact:
    """File-backed output of the native MLX SNeRV adapter."""

    schema: str
    output_dir: str
    packet_path: str
    packet_bytes: int
    packet_sha256: str
    num_pairs: int
    source_pair_indices: tuple[int, ...]
    levels: int
    wavelet: str
    target_bits_per_coeff: float
    step_map_bits_per_coeff: float
    step_map_packet_schema: str
    step_map_coder_mode: str
    step_map_coder_groups: tuple[dict[str, Any], ...]
    decoder_payload_codec: str
    lf_payload_codec: str
    model_size: dict[str, Any]
    bridge_drift: dict[str, Any]
    scorer_custody: dict[str, Any]
    scorer_loop_qat: dict[str, Any]
    storage_preflight: dict[str, Any]
    archive_package: dict[str, Any] | None
    archive_path: str | None
    archive_bytes: int | None
    archive_sha256: str | None
    receiver_proof_path: str | None
    receiver_proof_passed: bool
    receiver_contract_satisfied: bool
    report_path: str | None
    blockers: tuple[str, ...]
    score_claim: bool = False
    frontier_score_claim: bool = False
    rank_or_kill_eligible: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_pair_indices"] = [int(value) for value in self.source_pair_indices]
        payload["step_map_coder_groups"] = [dict(group) for group in self.step_map_coder_groups]
        payload["blockers"] = [str(value) for value in self.blockers]
        return payload


def train_export_snerv_mlx_native(
    output_dir: str | Path,
    num_pairs: int,
    source_video_path: str | Path,
    modelsize_candidate: Mapping[str, Any] | None,
    scorer_upstream_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    output_height: int = SCORER_HW[0],
    output_width: int = SCORER_HW[1],
    run_archive_export: bool = True,
    retain_receiver_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
    run_scorer_loop_qat: bool = False,
    native_mlx_decoder_train_steps: int = 0,
    native_mlx_decoder_train_lr: float = 1.0e-5,
    native_mlx_decoder_train_ridge: float = 1.0e-6,
    scorer_loop_qat_max_trials: int = 0,
    scorer_loop_qat_search_mode: str = "random_signed",
    scorer_loop_qat_qat_bits: int = 8,
    scorer_loop_qat_decoder_payload_codec: str | None = None,
    scorer_loop_qat_lf_payload_codec: str | None = None,
    scorer_loop_qat_component_guard_mode: str = "score_primary",
    scorer_loop_qat_device: str = "cpu",
    recon_pixel_weight_path: str | Path | None = None,
    recon_pixel_weight_manifest_path: str | Path | None = None,
    recon_pixel_weight_normalize: str = "mean",
    pair_indices: Sequence[Any] | str | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Hydrate real targets on MLX, export a NumPy-portable SNAR1 archive.

    The function name is intentionally the contract surface.  Its payload keeps
    the current training scope precise: ``mlx_target_hydration`` plus the
    existing deterministic closed-form SNeRV decoder fit.  It does not claim
    full score-aware long training.
    """

    started = time.monotonic()
    root = _repo_root(repo_root)
    scorer_custody = build_upstream_eval_contract(
        repo_root=root,
        upstream_dir=Path(scorer_upstream_dir).expanduser(),
    )
    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    candidate = dict(modelsize_candidate or {})
    levels = int(candidate.get("levels", candidate.get("snerv_levels", 3)))
    wavelet = str(candidate.get("wavelet", "haar"))
    target_bits_per_coeff = float(candidate.get("bits_per_coeff", 2.5))
    step_map_bits_per_coeff = float(
        candidate.get(
            "step_map_bits_per_coeff",
            candidate.get("snerv_step_map_bits_per_coeff", 4.0),
        )
    )
    decoder_payload_codec = str(candidate.get("decoder_payload_codec", "mixed_magnitude_symmetric"))
    active_decoder_payload_codec = (
        str(scorer_loop_qat_decoder_payload_codec) if scorer_loop_qat_decoder_payload_codec else decoder_payload_codec
    )
    lf_payload_codec = str(candidate.get("lf_payload_codec", "portfolio_auto"))
    active_lf_payload_codec = (
        str(scorer_loop_qat_lf_payload_codec) if scorer_loop_qat_lf_payload_codec else lf_payload_codec
    )
    model_size = _model_size_from_candidate(candidate)
    source_pair_indices = _source_pair_indices_for_native_export(
        int(num_pairs),
        pair_indices=pair_indices,
    )
    effective_num_pairs = len(source_pair_indices)
    explicit_pair_indices = pair_indices is not None
    official_primitives_requested = bool(model_size.official_mfu_hfr_tub_numeric_primitives_requested)
    official_primitives_blockers = list(model_size.official_mfu_hfr_tub_export_blockers)
    official_binding = (
        _official_primitives_export_binding(
            model_size=model_size,
            candidate=candidate,
            blockers=official_primitives_blockers,
        )
        if official_primitives_requested
        else None
    )

    target0_mlx, target1_mlx = decode_mlx_targets(
        source_video_path,
        num_pairs=effective_num_pairs,
        output_height=int(output_height),
        output_width=int(output_width),
        pair_indices=source_pair_indices if explicit_pair_indices else None,
    )
    target0_np = np.asarray(target0_mlx, dtype=np.float32)
    target1_np = np.asarray(target1_mlx, dtype=np.float32)
    bridge = build_mlx_numpy_bridge_drift_bundle(
        (
            mlx_numpy_bridge_drift_report(
                label="target_frame0_nhwc01",
                mlx_array=target0_mlx,
                numpy_array=target0_np,
                atol=0.0,
                rtol=0.0,
            ),
            mlx_numpy_bridge_drift_report(
                label="target_frame1_nhwc01",
                mlx_array=target1_mlx,
                numpy_array=target1_np,
                atol=0.0,
                rtol=0.0,
            ),
        ),
        bundle_id="snerv_mlx_native_target_bridge",
    )
    if bridge["allclose"] is not True:
        raise SnervMlxNativeExportError("MLX target bridge drift failed: " + ",".join(bridge["blockers"]))

    pairs_nchw255 = _target_pairs_to_nchw255(target0_np, target1_np)
    recon_weight, recon_weight_metadata = _load_recon_pixel_weight_for_native_export(
        recon_pixel_weight_path,
        manifest_path=recon_pixel_weight_manifest_path,
        expected_pairs=effective_num_pairs,
        expected_hw=(int(output_height), int(output_width)),
        normalize=str(recon_pixel_weight_normalize),
    )
    closed_form_archive = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs_nchw255,
        levels=levels,
        wavelet=wavelet,
        target_bits_per_coeff=target_bits_per_coeff,
        step_map_bits_per_coeff=step_map_bits_per_coeff,
        decoder_payload_codec=active_decoder_payload_codec,
        lf_payload_codec=active_lf_payload_codec,
        model_size=model_size,
        source_pair_indices=source_pair_indices,
        recon_pixel_weight=recon_weight,
        recon_pixel_weight_metadata=recon_weight_metadata,
        hf_decoder_saliency_gain=float(
            candidate.get(
                "hf_decoder_saliency_gain",
                candidate.get("snerv_hf_decoder_saliency_gain", 1.0),
            )
        ),
        native_mlx_decoder_train_steps=int(
            candidate.get(
                "native_mlx_decoder_train_steps",
                candidate.get("snerv_native_mlx_decoder_train_steps", native_mlx_decoder_train_steps),
            )
        ),
        native_mlx_decoder_train_lr=float(
            candidate.get(
                "native_mlx_decoder_train_lr",
                candidate.get("snerv_native_mlx_decoder_train_lr", native_mlx_decoder_train_lr),
            )
        ),
        native_mlx_decoder_train_ridge=float(
            candidate.get(
                "native_mlx_decoder_train_ridge",
                candidate.get("snerv_native_mlx_decoder_train_ridge", native_mlx_decoder_train_ridge),
            )
        ),
        metadata_extra={
            "source_video_path": Path(source_video_path).as_posix(),
            "pair_index_alignment_mode": (
                "explicit_source_pair_indices" if explicit_pair_indices else "prefix_source_pair_indices"
            ),
            "human_visual_fidelity_objective": False,
            "contest_scorer_distortion_objective": (
                _recon_pixel_weight_metadata_is_verified_gradient_manifest(recon_weight_metadata)
                if recon_weight is not None
                else False
            ),
            **(
                _official_primitives_packet_metadata(
                    official_binding,
                    blockers=official_primitives_blockers,
                )
                if official_binding is not None
                else {}
            ),
        },
    )
    scorer_loop_qat = _run_scorer_loop_qat_attachment(
        requested=bool(run_scorer_loop_qat),
        output_dir=out / "snerv_scorer_loop_qat",
        num_pairs=effective_num_pairs,
        pair_indices=source_pair_indices if explicit_pair_indices else None,
        source_video_path=source_video_path,
        scorer_upstream_dir=scorer_upstream_dir,
        levels=levels,
        wavelet=wavelet,
        target_bits_per_coeff=target_bits_per_coeff,
        model_size=model_size,
        max_trials=int(scorer_loop_qat_max_trials),
        search_mode=str(scorer_loop_qat_search_mode),
        qat_bits=int(scorer_loop_qat_qat_bits),
        decoder_payload_codec=active_decoder_payload_codec,
        lf_payload_codec=active_lf_payload_codec,
        component_guard_mode=str(scorer_loop_qat_component_guard_mode),
        device=str(scorer_loop_qat_device),
        allow_overwrite=allow_overwrite,
    )
    selected_packet = bytes(closed_form_archive.packet)
    selected_packet_source = _packet_source_from_snerv_native_metadata(closed_form_archive.metadata)
    scorer_loop_qat_public = {key: value for key, value in scorer_loop_qat.items() if key != "_best_packet_bytes"}
    best_packet = scorer_loop_qat.get("_best_packet_bytes")
    best_packet_ready = (
        isinstance(best_packet, bytes)
        and best_packet
        and scorer_loop_qat.get("accepted_improvement") is True
        and scorer_loop_qat.get("receiver_contract_satisfied") is True
    )
    best_packet_source_pair_indices = (
        _packet_source_pair_indices(best_packet) if isinstance(best_packet, bytes) and best_packet else None
    )
    best_packet_preserves_source_pairs = _packet_preserves_source_pair_indices(
        best_packet if isinstance(best_packet, bytes) else b"",
        expected_source_pair_indices=source_pair_indices,
        explicit_pair_indices=explicit_pair_indices,
    )
    best_packet_preserves_recon_weight = _packet_preserves_recon_weight_binding(
        best_packet if isinstance(best_packet, bytes) else b"",
        recon_weight_metadata,
    )
    if best_packet_ready and best_packet_preserves_recon_weight and best_packet_preserves_source_pairs:
        selected_packet = bytes(best_packet)
        selected_packet_source = "scorer_loop_qat_best_receiver_packet"
        scorer_loop_qat_public["emitted_packet_uses_scorer_loop_best_decoder"] = True
        scorer_loop_qat_public["emitted_packet_bytes"] = len(selected_packet)
        scorer_loop_qat_public["emitted_packet_sha256"] = _sha256_bytes(selected_packet)
        scorer_loop_qat_public["blockers"] = _ordered_unique(
            str(blocker)
            for blocker in scorer_loop_qat_public.get("blockers") or []
            if str(blocker) != "snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export"
        )
        report = scorer_loop_qat_public.get("report_path")
        if report:
            _payload_for_disk = {key: value for key, value in scorer_loop_qat_public.items() if key != "report_path"}
            write_json(report, _payload_for_disk)
    elif best_packet_ready and (recon_weight_metadata is not None or explicit_pair_indices):
        scorer_loop_qat_public["emitted_packet_uses_scorer_loop_best_decoder"] = False
        extra_blockers: list[str] = []
        if recon_weight_metadata is not None and not best_packet_preserves_recon_weight:
            scorer_loop_qat_public["recon_weight_binding_required"] = True
            scorer_loop_qat_public["recon_weight_binding_preserved"] = False
            scorer_loop_qat_public["recon_weight_expected_sha256"] = recon_weight_metadata.get("sha256")
            extra_blockers.append("snerv_scorer_loop_qat_best_packet_rejected_recon_weight_binding_mismatch")
        if explicit_pair_indices and not best_packet_preserves_source_pairs:
            scorer_loop_qat_public["source_pair_indices_binding_required"] = True
            scorer_loop_qat_public["source_pair_indices_binding_preserved"] = False
            scorer_loop_qat_public["source_pair_indices_expected"] = [int(value) for value in source_pair_indices]
            scorer_loop_qat_public["source_pair_indices_actual"] = (
                [int(value) for value in best_packet_source_pair_indices]
                if best_packet_source_pair_indices is not None
                else None
            )
            extra_blockers.append("snerv_scorer_loop_qat_best_packet_rejected_source_pair_indices_mismatch")
        if extra_blockers:
            extra_blockers.append("snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export")
        scorer_loop_qat_public["blockers"] = _ordered_unique(
            [
                *(str(blocker) for blocker in scorer_loop_qat_public.get("blockers") or [] if str(blocker)),
                *extra_blockers,
            ]
        )
        report = scorer_loop_qat_public.get("report_path")
        if report:
            _payload_for_disk = {key: value for key, value in scorer_loop_qat_public.items() if key != "report_path"}
            write_json(report, _payload_for_disk)

    packet_path = out / SNERV_MLX_NATIVE_PACKET_FILENAME
    write_bytes_artifact(
        packet_path,
        selected_packet,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=(sha256_file(packet_path) if allow_overwrite and packet_path.is_file() else None),
    )

    storage_preflight = build_snerv_mlx_native_storage_preflight(
        output_dir=out,
        n_pairs=effective_num_pairs,
        packet_bytes=len(selected_packet),
    )
    package: dict[str, Any] | None = None
    blockers = [
        "snerv_mlx_score_aware_long_training_not_executed",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if official_primitives_requested:
        blockers.extend(official_primitives_blockers)
    if scorer_custody.get("contract_valid") is not True:
        blockers.append("snerv_mlx_native_scorer_custody_contract_invalid")
    if recon_weight_metadata is not None and recon_weight_metadata.get("producer_manifest_verified") is not True:
        blockers.append("snerv_recon_pixel_weight_verified_gradient_manifest_not_bound_to_native_export")
    blockers.extend(
        _scorer_loop_qat_blockers(
            scorer_loop_qat_public,
            num_pairs=effective_num_pairs,
        )
    )
    if run_archive_export:
        if storage_preflight["preflight_passed"] is not True:
            blockers.append("snerv_mlx_native_receiver_proof_storage_preflight_failed")
        else:
            package = export_snerv_mlx_archive(
                model_or_artifact={
                    "packet_path": packet_path.as_posix(),
                    "packet_sha256": _sha256_bytes(selected_packet),
                },
                output_dir=out / "snerv_mlx_native_archive_bound_package",
                repo_root=root,
                retain_receiver_output=retain_receiver_output,
                receiver_proof_timeout_seconds=receiver_proof_timeout_seconds,
            )
    receiver_proof = dict(package.get("receiver_proof") or {}) if package else {}
    selected_archive_metadata = unpack_snerv_archive(selected_packet).metadata

    artifact = SnervMlxNativeArtifact(
        schema=SNERV_MLX_NATIVE_TRAIN_EXPORT_SCHEMA,
        output_dir=out.as_posix(),
        packet_path=packet_path.as_posix(),
        packet_bytes=len(selected_packet),
        packet_sha256=_sha256_bytes(selected_packet),
        num_pairs=effective_num_pairs,
        source_pair_indices=tuple(int(value) for value in source_pair_indices),
        levels=levels,
        wavelet=wavelet,
        target_bits_per_coeff=target_bits_per_coeff,
        step_map_bits_per_coeff=step_map_bits_per_coeff,
        step_map_packet_schema=str(selected_archive_metadata.get("step_map_packet_schema") or ""),
        step_map_coder_mode=str(selected_archive_metadata.get("step_map_coder_mode") or ""),
        step_map_coder_groups=tuple(
            dict(group) for group in selected_archive_metadata.get("step_map_coder_groups") or ()
        ),
        decoder_payload_codec=active_decoder_payload_codec,
        lf_payload_codec=active_lf_payload_codec,
        model_size=model_size.as_jsonable(),
        bridge_drift=bridge,
        scorer_custody=scorer_custody,
        scorer_loop_qat=scorer_loop_qat_public,
        storage_preflight=storage_preflight,
        archive_package=package,
        archive_path=(str(receiver_proof.get("archive_path")) if receiver_proof.get("archive_path") else None),
        archive_bytes=(
            int(receiver_proof["archive_bytes"]) if receiver_proof.get("archive_bytes") is not None else None
        ),
        archive_sha256=(str(receiver_proof.get("archive_sha256")) if receiver_proof.get("archive_sha256") else None),
        receiver_proof_path=(str(receiver_proof.get("proof_path")) if receiver_proof.get("proof_path") else None),
        receiver_proof_passed=(receiver_proof.get("runtime_consumption_proof_passed") is True),
        receiver_contract_satisfied=(receiver_proof.get("receiver_contract_satisfied") is True),
        report_path=None,
        blockers=tuple(dict.fromkeys(blockers)),
    )
    payload = artifact.as_jsonable()
    payload["executed"] = True
    payload["packet_source"] = selected_packet_source
    selected_recon_metadata = selected_archive_metadata.get("recon_pixel_weight_metadata")
    selected_recon_consumed = selected_archive_metadata.get("recon_pixel_weight_consumed") is True and isinstance(
        selected_recon_metadata, Mapping
    )
    payload["score_aware_hf_decoder_fit_executed"] = bool(selected_recon_consumed)
    payload["score_aware_long_training_executed"] = False
    payload["native_mlx_training_executed"] = bool(
        selected_archive_metadata.get("native_mlx_training_executed") is True
    )
    payload["native_mlx_training_kind"] = str(selected_archive_metadata.get("native_mlx_training_kind") or "none")
    payload["native_mlx_hf_decoder_training"] = dict(
        selected_archive_metadata.get("native_mlx_hf_decoder_training") or {}
    )
    if selected_recon_consumed:
        payload["recon_pixel_weight"] = {
            **dict(selected_recon_metadata),
            "selected_packet_consumed": True,
        }
    else:
        payload["recon_pixel_weight"] = {
            "schema": "snerv_mlx_native_recon_pixel_weight_consumption.v1",
            "enabled": False,
            "requested": recon_weight_metadata is not None,
            "selected_packet_consumed": False,
            "source_kind": "disabled",
            "hf_decoder_fit_mode": "least_squares_numpy_closed_form",
            "requested_metadata": dict(recon_weight_metadata or {}),
            **FALSE_AUTHORITY,
        }
    if official_binding is not None:
        payload["official_primitive_binding"] = _receiver_bound_official_primitives_export_binding(
            official_binding,
            packet_path=packet_path,
            packet_bytes=len(selected_packet),
            packet_sha256=_sha256_bytes(selected_packet),
            selected_archive_metadata=selected_archive_metadata,
            package=package,
            receiver_proof=receiver_proof,
        )
        payload["snerv_official_mfu_hfr_tub_numeric_primitives_requested"] = True
        payload["snerv_official_mfu_hfr_tub_export_bound"] = False
        payload["snerv_official_mfu_hfr_tub_receiver_bound_surrogate_export"] = True
        payload["snerv_official_mfu_hfr_tub_export_blockers"] = list(official_primitives_blockers)
    payload["wall_seconds"] = round(time.monotonic() - started, 6)
    report_path = out / SNERV_MLX_NATIVE_REPORT_FILENAME
    if report_path.exists() and not allow_overwrite:
        raise SnervMlxNativeExportError(f"refusing to overwrite existing artifact: {report_path}")
    payload["report_path"] = report_path.as_posix()
    write_json(report_path, payload)
    return payload


def export_snerv_mlx_archive(
    model_or_artifact: Any,
    output_dir: str | Path,
    repo_root: str | Path,
    *,
    retain_receiver_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Export SNAR1 packet bytes through the canonical archive-bound package."""

    root = _repo_root(repo_root)
    out = Path(output_dir).expanduser().resolve(strict=False)
    packet = _packet_bytes_from_artifact(model_or_artifact)
    decoded = unpack_snerv_archive(packet)
    storage = build_snerv_mlx_native_storage_preflight(
        output_dir=out,
        n_pairs=int(decoded.metadata.get("n_pairs", 0)),
        packet_bytes=len(packet),
    )
    if storage["preflight_passed"] is not True:
        raise SnervMlxNativeExportError(
            "receiver proof storage preflight failed: "
            f"free={storage['free_bytes']} required={storage['required_bytes']}"
        )
    package = export_snerv_archive_bound_candidate_package(
        packet=packet,
        output_dir=out,
        repo_root=root,
        retain_receiver_output=retain_receiver_output,
        receiver_proof_timeout_seconds=receiver_proof_timeout_seconds,
    )
    package["snerv_mlx_native_storage_preflight"] = storage
    return package


def write_snerv_mlx_receiver_proof(
    archive_zip_path: str | Path,
    runtime_submission_dir: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    retain_receiver_output: bool = False,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run the generated SNeRV receiver proof for an existing archive package."""

    root = _repo_root(repo_root)
    archive_zip = Path(archive_zip_path).expanduser().resolve(strict=False)
    submission_dir = Path(runtime_submission_dir).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    packet_path = submission_dir / "0.bin"
    if not packet_path.is_file():
        raise SnervMlxNativeExportError(f"runtime submission dir is missing 0.bin: {submission_dir}")
    decoded = unpack_snerv_archive(packet_path.read_bytes())
    expected_bytes = expected_receiver_output_bytes_from_metadata(decoded.metadata)
    storage = build_snerv_mlx_native_storage_preflight(
        output_dir=out,
        n_pairs=int(decoded.metadata.get("n_pairs", 0)),
        packet_bytes=packet_path.stat().st_size,
    )
    if storage["preflight_passed"] is not True:
        raise SnervMlxNativeExportError(
            "receiver proof storage preflight failed: "
            f"free={storage['free_bytes']} required={storage['required_bytes']}"
        )
    proof = run_generated_inflate_receiver_proof(
        archive_zip_path=archive_zip,
        archive_sha256=sha256_file(archive_zip),
        archive_bytes=archive_zip.stat().st_size,
        submission_dir=submission_dir,
        archive_dir_for_inflate=submission_dir,
        output_dir=out,
        repo_root=root,
        proof_schema=SNERV_RECEIVER_PROOF_SCHEMA,
        proof_filename="snerv_mlx_native_receiver_proof.json",
        candidate_label="snerv_mlx_native",
        expected_receiver_output_name="0.raw",
        expected_receiver_output_bytes=expected_bytes,
        retain_receiver_output=retain_receiver_output,
        timeout_seconds=int(timeout_seconds),
    )
    proof["snerv_mlx_native_storage_preflight"] = storage
    return proof


def _run_scorer_loop_qat_attachment(
    *,
    requested: bool,
    output_dir: str | Path,
    num_pairs: int,
    pair_indices: tuple[int, ...] | None,
    source_video_path: str | Path,
    scorer_upstream_dir: str | Path,
    levels: int,
    wavelet: str,
    target_bits_per_coeff: float,
    model_size: SnervModelSizeConfig,
    max_trials: int,
    search_mode: str,
    qat_bits: int,
    decoder_payload_codec: str,
    lf_payload_codec: str,
    component_guard_mode: str,
    device: str,
    allow_overwrite: bool,
) -> dict[str, Any]:
    """Attach real SegNet/PoseNet scorer-loop evidence to a native export."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "snerv_scorer_loop_qat_attachment.json"
    if report_path.exists() and not allow_overwrite:
        raise SnervMlxNativeExportError(f"refusing to overwrite existing scorer-loop artifact: {report_path}")
    if not requested:
        payload = {
            "schema": "snerv_mlx_native_scorer_loop_qat_attachment.v1",
            "requested": False,
            "executed": False,
            "source_pair_indices": [int(value) for value in pair_indices or ()],
            "component_guard_mode": str(component_guard_mode),
            "decoder_payload_codec": str(decoder_payload_codec),
            "lf_payload_codec": str(lf_payload_codec),
            "receiver_contract_satisfied": False,
            "accepted_improvement": False,
            "full_video_coverage": False,
            "emitted_packet_uses_scorer_loop_best_decoder": False,
            "blockers": ["snerv_scorer_loop_qat_not_requested"],
            **FALSE_AUTHORITY,
        }
        write_json(report_path, payload)
        return {**payload, "report_path": report_path.as_posix()}

    try:
        from tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat import (
            run_snerv_scorer_loop_decoder_qat_smoke,
        )

        result = run_snerv_scorer_loop_decoder_qat_smoke(
            n_pairs=int(num_pairs),
            pair_indices=pair_indices,
            levels=int(levels),
            wavelet=str(wavelet),
            target_bits_per_coeff=float(target_bits_per_coeff),
            upstream_dir=str(scorer_upstream_dir),
            video_path=str(source_video_path),
            device=str(device),
            snerv_model_size_adapter=model_size.adapter,
            snerv_fc_dim=int(model_size.fc_dim),
            snerv_emb_size=int(model_size.emb_size),
            snerv_patch_radius=int(model_size.patch_radius),
            snerv_mfu_scales=tuple(int(v) for v in model_size.mfu_scales),
            snerv_hfr_gain=float(model_size.hfr_gain),
            snerv_temporal_context=int(model_size.temporal_context),
            snerv_temporal_mode=model_size.temporal_mode,
            decoder_payload_codec=str(decoder_payload_codec),
            lf_payload_codec=str(lf_payload_codec),
            qat_bits=int(qat_bits),
            max_trials=int(max_trials),
            search_mode=str(search_mode),
            component_guard_mode=str(component_guard_mode),
        )
        result_payload = result.as_jsonable()
        best_packet = getattr(result, "best_packet", b"")
        if best_packet:
            best_packet = bytes(best_packet)
        best_packet_path = out / "best_packet.snar"
        best_packet_materialized = False
        best_packet_path_str: str | None = None
        best_packet_path_sha256: str | None = None
        if best_packet:
            write_bytes_artifact(
                best_packet_path,
                best_packet,
                allow_overwrite=allow_overwrite,
                expected_existing_sha256=(
                    sha256_file(best_packet_path) if allow_overwrite and best_packet_path.is_file() else None
                ),
            )
            best_packet_materialized = True
            best_packet_path_str = best_packet_path.as_posix()
            best_packet_path_sha256 = sha256_file(best_packet_path)
        blockers = [str(blocker) for blocker in result_payload.get("blockers") or [] if blocker]
        if bool(result_payload.get("receiver_contract_satisfied")) and not (
            best_packet_materialized and best_packet_path_sha256 == result_payload.get("best_packet_sha256")
        ):
            blockers.append("snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export")
        payload = {
            "schema": "snerv_mlx_native_scorer_loop_qat_attachment.v1",
            "requested": True,
            "executed": True,
            "source_schema": result_payload.get("schema"),
            "axis_tag": result_payload.get("axis_tag"),
            "num_pairs": int(result_payload.get("n_pairs") or num_pairs),
            "source_pair_indices": [
                int(value)
                for value in (
                    result_payload.get("source_pair_indices")
                    or pair_indices
                    or tuple(range(int(result_payload.get("n_pairs") or num_pairs)))
                )
            ],
            "full_video_coverage": int(result_payload.get("n_pairs") or 0) == 600,
            "scorer_loop_evaluations": int(result_payload.get("scorer_loop_evaluations") or 0),
            "baseline_archive_bytes": _nested(result_payload, "baseline", "archive_bytes"),
            "best_archive_bytes": _nested(result_payload, "best", "archive_bytes"),
            "baseline_archive_sha256": _nested(result_payload, "baseline", "archive_sha256"),
            "best_archive_sha256": _nested(result_payload, "best", "archive_sha256"),
            "baseline_score_linf": _nested(result_payload, "baseline", "score_linf"),
            "best_score_linf": _nested(result_payload, "best", "score_linf"),
            "best_packet_bytes": int(result_payload.get("best_packet_bytes") or 0),
            "best_packet_sha256": result_payload.get("best_packet_sha256"),
            "best_packet_path": best_packet_path_str,
            "best_packet_path_sha256": best_packet_path_sha256,
            "best_packet_materialized": best_packet_materialized,
            "decoder_payload_codec": str(result_payload.get("decoder_payload_codec") or decoder_payload_codec),
            "lf_payload_codec": str(result_payload.get("lf_payload_codec") or lf_payload_codec),
            "component_guard_mode": str(result_payload.get("component_guard_mode") or component_guard_mode),
            "pair_robust_admission": result_payload.get("pair_robust_admission"),
            "accepted_improvement": bool(result_payload.get("accepted_improvement")),
            "receiver_contract_satisfied": bool(result_payload.get("receiver_contract_satisfied")),
            "ready_for_pose_guard_gate": bool(result_payload.get("ready_for_pose_guard_gate")),
            "emitted_packet_uses_scorer_loop_best_decoder": False,
            "result": result_payload,
            "blockers": _ordered_unique(blockers),
            **FALSE_AUTHORITY,
        }
        if best_packet_materialized and best_packet_path_sha256 == result_payload.get("best_packet_sha256"):
            payload["_best_packet_bytes"] = best_packet
    except Exception as exc:
        payload = {
            "schema": "snerv_mlx_native_scorer_loop_qat_attachment.v1",
            "requested": True,
            "executed": False,
            "failure": repr(exc),
            "source_pair_indices": [int(value) for value in pair_indices or ()],
            "receiver_contract_satisfied": False,
            "accepted_improvement": False,
            "decoder_payload_codec": str(decoder_payload_codec),
            "lf_payload_codec": str(lf_payload_codec),
            "component_guard_mode": str(component_guard_mode),
            "full_video_coverage": False,
            "emitted_packet_uses_scorer_loop_best_decoder": False,
            "blockers": ["snerv_scorer_loop_qat_attachment_failed"],
            **FALSE_AUTHORITY,
        }
    payload_for_disk = {key: value for key, value in payload.items() if key != "_best_packet_bytes"}
    write_json(report_path, payload_for_disk)
    return {**payload, "report_path": report_path.as_posix()}


def _scorer_loop_qat_blockers(
    scorer_loop_qat: Mapping[str, Any],
    *,
    num_pairs: int,
) -> list[str]:
    blockers = [str(blocker) for blocker in scorer_loop_qat.get("blockers") or [] if str(blocker)]
    if scorer_loop_qat.get("executed") is not True:
        blockers.append("snerv_real_segnet_posenet_teacher_loop_not_attached")
        return _ordered_unique(blockers)
    if scorer_loop_qat.get("receiver_contract_satisfied") is not True:
        blockers.append("snerv_real_segnet_posenet_teacher_loop_receiver_unproven")
    if scorer_loop_qat.get("accepted_improvement") is not True:
        blockers.append("snerv_scorer_loop_qat_no_accepted_improvement")
    if int(num_pairs) != 600 or scorer_loop_qat.get("full_video_coverage") is not True:
        blockers.append("snerv_scorer_loop_qat_not_full_video")
    if scorer_loop_qat.get("emitted_packet_uses_scorer_loop_best_decoder") is not True:
        blockers.append("snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export")
    return _ordered_unique(blockers)


def write_snerv_mlx_prefilter_profile(
    artifact: Any,
    archive_bytes: int,
    archive_sha256: str,
    output_path: str | Path,
    upstream_dir: str | Path,
    *,
    component_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a false-authority local MLX prefilter profile for queue gates."""

    art = _artifact_mapping(artifact)
    component = dict(component_profile or {})
    blockers = []
    artifact_blockers = [str(item) for item in art.get("blockers") or [] if item]
    if not component:
        blockers.append("snerv_mlx_prefilter_component_scorers_not_attached")
    if int(art.get("num_pairs") or 0) != 600:
        blockers.append("snerv_mlx_prefilter_not_full_video")
    if artifact_blockers:
        blockers.append("snerv_mlx_prefilter_artifact_has_blockers")
    bridge = art.get("bridge_drift") if isinstance(art.get("bridge_drift"), Mapping) else {}
    if bridge.get("allclose") is not True:
        blockers.append("snerv_mlx_prefilter_bridge_drift_unproven")
    if not art.get("archive_path") or not art.get("archive_sha256"):
        blockers.append("snerv_mlx_prefilter_archive_package_missing")
    elif str(art.get("archive_sha256")) != str(archive_sha256):
        blockers.append("snerv_mlx_prefilter_archive_sha_mismatch")
    if int(art.get("archive_bytes") or 0) and int(art.get("archive_bytes") or 0) != int(archive_bytes):
        blockers.append("snerv_mlx_prefilter_archive_bytes_mismatch")
    if art.get("receiver_proof_passed") is not True:
        blockers.append("snerv_mlx_prefilter_receiver_proof_missing")
    if art.get("receiver_contract_satisfied") is not True:
        blockers.append("snerv_mlx_prefilter_receiver_contract_unsatisfied")
    ready_for_cpu_replay = bool(component) and not blockers
    payload = {
        "schema": SNERV_MLX_NATIVE_PREFILTER_PROFILE_SCHEMA,
        "artifact_schema": art.get("schema"),
        "artifact_report_path": art.get("report_path"),
        "packet_path": art.get("packet_path"),
        "num_pairs": int(art.get("num_pairs") or 0),
        "archive_bytes": int(archive_bytes),
        "archive_sha256": str(archive_sha256),
        "upstream_dir": Path(upstream_dir).as_posix(),
        "component_profile": component or None,
        "prefilter_ready_for_cpu_replay": ready_for_cpu_replay,
        "artifact_blockers": artifact_blockers,
        "blockers": [b for b in dict.fromkeys(blockers) if b],
        **FALSE_AUTHORITY,
    }
    write_json(output_path, payload)
    return payload


def build_snerv_mlx_native_packet_from_numpy_pairs(
    pairs_nchw255: np.ndarray,
    *,
    levels: int,
    wavelet: str,
    target_bits_per_coeff: float,
    decoder_payload_codec: str,
    step_map_bits_per_coeff: float = 4.0,
    lf_payload_codec: str = "portfolio_auto",
    model_size: SnervModelSizeConfig | None = None,
    source_pair_indices: Sequence[Any] | str | None = None,
    recon_pixel_weight: np.ndarray | None = None,
    recon_pixel_weight_metadata: Mapping[str, Any] | None = None,
    hf_decoder_saliency_gain: float = 1.0,
    native_mlx_decoder_train_steps: int = 0,
    native_mlx_decoder_train_lr: float = 1.0e-5,
    native_mlx_decoder_train_ridge: float = 1.0e-6,
    metadata_extra: Mapping[str, Any] | None = None,
) -> SnervArchivePacket:
    """Build an SNAR1 packet from NumPy pair frames using existing SNeRV codecs."""

    pairs = np.asarray(pairs_nchw255, dtype=np.float32)
    if pairs.ndim != 5 or pairs.shape[1] != 2 or pairs.shape[2] != 3:
        raise SnervMlxNativeExportError(f"pairs_nchw255 must be shaped (pairs, 2, 3, H, W); got {tuple(pairs.shape)}")
    if not np.isfinite(pairs).all():
        raise SnervMlxNativeExportError("pairs_nchw255 contains nonfinite values")
    n_pairs, _frames, channels, h, w = (int(v) for v in pairs.shape)
    source_pair_indices_tuple = _source_pair_indices_for_packet(
        n_pairs,
        source_pair_indices=source_pair_indices,
    )
    model_size = model_size or SnervModelSizeConfig()
    weighted_fit_enabled = recon_pixel_weight is not None
    recon_weight = (
        _normalize_recon_pixel_weight_array(
            recon_pixel_weight,
            expected_pairs=n_pairs,
            expected_hw=(h, w),
            normalize="none",
        )
        if weighted_fit_enabled
        else None
    )
    pyramids = []
    weight_pyramids: list[WaveletPyramid] | None = [] if weighted_fit_enabled else None
    records: list[tuple[int, int, int, Any]] = []
    for pair_idx in range(n_pairs):
        for frame_idx in range(2):
            for channel_idx in range(channels):
                pyr = encode_frame_lf(
                    pairs[pair_idx, frame_idx, channel_idx],
                    levels=int(levels),
                    wavelet=str(wavelet),
                )
                pyramids.append(pyr)
                if weight_pyramids is not None and recon_weight is not None:
                    weight_channel = 0 if int(recon_weight.shape[-1]) == 1 else channel_idx
                    weight_pyramids.append(
                        dwt2_native_synthesis_adjoint(
                            recon_weight[pair_idx, frame_idx, :, :, weight_channel],
                            levels=int(levels),
                            wavelet=str(wavelet),
                        )
                    )
                records.append((pair_idx, frame_idx, channel_idx, pyr))
    if weight_pyramids is None:
        decoder = fit_hf_decoder_least_squares(
            pyramids,
            levels=int(levels),
            model_size=model_size,
            temporal_group_count=channels,
        )
        hf_decoder_fit_mode = "least_squares_numpy_closed_form"
    else:
        decoder = fit_hf_decoder_weighted_least_squares(
            pyramids,
            levels=int(levels),
            detail_weight_pyramids=weight_pyramids,
            saliency_gain=float(hf_decoder_saliency_gain),
            model_size=model_size,
            temporal_group_count=channels,
        )
        hf_decoder_fit_mode = SNERV_DWT_ADJOINT_SALIENCY_WEIGHTED_FIT_MODE
    mlx_training_report: dict[str, Any] = {
        "schema": "snerv_native_mlx_hf_decoder_training.v1",
        "requested_steps": int(native_mlx_decoder_train_steps),
        "executed": False,
        "blockers": (
            ["snerv_native_mlx_decoder_training_not_requested"] if int(native_mlx_decoder_train_steps) <= 0 else []
        ),
        **FALSE_AUTHORITY,
    }
    if int(native_mlx_decoder_train_steps) > 0:
        trained_decoder, mlx_training_report = _fit_hf_decoder_mlx_full_batch_gradient_descent(
            pyramids,
            levels=int(levels),
            initial_decoder=decoder,
            detail_weight_pyramids=weight_pyramids,
            saliency_gain=float(hf_decoder_saliency_gain),
            temporal_group_count=channels,
            steps=int(native_mlx_decoder_train_steps),
            learning_rate=float(native_mlx_decoder_train_lr),
            ridge=float(native_mlx_decoder_train_ridge),
        )
        if mlx_training_report.get("executed") is True:
            decoder = trained_decoder
            hf_decoder_fit_mode = f"native_mlx_full_batch_gradient_descent_from_{hf_decoder_fit_mode}"

    lf_quant_planes: list[np.ndarray] = []
    lf_zero_points: list[float] = []
    step_maps: list[np.ndarray] = []
    lf_step_allocation_rows: list[dict[str, Any]] = []
    step_map_importance_values: list[float] = []
    n_levels = max(2, round(2.0 ** float(target_bits_per_coeff)))
    for record_index, (pair_idx, frame_idx, channel_idx, pyr) in enumerate(records):
        source_pair_idx = int(source_pair_indices_tuple[pair_idx])
        q_uniform, scale, zero = quantize_lf(pyr.lf, n_levels=n_levels)
        step = np.full(q_uniform.shape, float(scale), dtype=np.float32)
        allocation_row: dict[str, Any] = {
            "schema": "snerv_lf_step_allocation_row.v1",
            "pair_idx": int(pair_idx),
            "source_pair_idx": source_pair_idx,
            "frame_idx": int(frame_idx),
            "channel_idx": int(channel_idx),
            "mode": "uniform_l2_baseline",
            "uniform_step": float(scale),
            "target_bits_per_coeff": float(target_bits_per_coeff),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        if weight_pyramids is not None:
            weight_lf = np.abs(np.asarray(weight_pyramids[record_index].lf, dtype=np.float64))
            if weight_lf.shape != pyr.lf.shape:
                raise SnervMlxNativeExportError(
                    f"DWT-adjoint LF saliency shape {weight_lf.shape} != LF shape {pyr.lf.shape}"
                )
            dynamic_range = max(float(np.max(pyr.lf) - np.min(pyr.lf)), 1.0e-9)
            min_step = max(float(scale) / 16.0, 1.0e-9)
            max_step = max(dynamic_range, float(scale) * 16.0, min_step * 2.0)
            alloc = allocate_lf_linf(
                LfSaliency(
                    lf_saliency=weight_lf,
                    lf_shape=(int(pyr.lf.shape[0]), int(pyr.lf.shape[1])),
                    pixel_seg_mass=0.0,
                    pixel_pose_mass=0.0,
                ),
                target_bits=float(pyr.lf.size) * float(target_bits_per_coeff),
                dynamic_range=dynamic_range,
                min_step=min_step,
                max_step=max_step,
            )
            step = alloc.steps.reshape(pyr.lf.shape).astype(np.float32)
            allocation_row.update(
                {
                    "mode": "joint_p18_p19_dwt_adjoint_lf_reverse_waterfill",
                    "dynamic_range": dynamic_range,
                    "target_bits_total": float(pyr.lf.size) * float(target_bits_per_coeff),
                    "realized_bits_total": float(alloc.total_bits),
                    "water_level": float(alloc.water_level),
                    "min_step": float(alloc.min_step),
                    "max_step": float(alloc.max_step),
                    "lf_saliency_mean": float(np.mean(weight_lf)),
                    "lf_saliency_max": float(np.max(weight_lf)),
                }
            )
        q, _scale, zero = quantize_lf(pyr.lf, per_element_steps=step)
        lf_quant_planes.append(q)
        lf_zero_points.append(float(zero))
        step_maps.append(step)
        step_map_importance_values.append(max(float(allocation_row.get("lf_saliency_mean", 1.0)), 1.0e-12))
        lf_step_allocation_rows.append(allocation_row)

    step_map_importance = np.asarray(step_map_importance_values, dtype=np.float64)
    step_packet = encode_step_maps_waterfill(
        step_maps,
        map_importance=step_map_importance,
        target_bits_per_coeff=float(step_map_bits_per_coeff),
    )
    decoder_payload = encode_decoder_payload(decoder, codec=decoder_payload_codec)
    verified_recon_weight = _recon_pixel_weight_metadata_is_verified_gradient_manifest(recon_pixel_weight_metadata)
    metadata = {
        "n_pairs": n_pairs,
        "frames_per_pair": 2,
        "channels": channels,
        "levels": int(levels),
        "wavelet": str(wavelet),
        "carrier_hw": [h, w],
        "orig_hw": [h, w],
        "source_pair_indices": [int(value) for value in source_pair_indices_tuple],
        "source_pair_indices_preserved": True,
        "pair_index_alignment_mode": (
            "prefix_source_pair_indices"
            if source_pair_indices_tuple == tuple(range(n_pairs))
            else "explicit_source_pair_indices"
        ),
        "lf_plane_count": len(lf_quant_planes),
        "lf_coeff_count_total": int(sum(int(p.lf.size) for p in pyramids)),
        "lf_zero_dtype": "float32_le",
        "lf_scale_mode": "implicit_per_element_steps_scale_1",
        "lf_payload_codec": str(lf_payload_codec),
        "step_map_packet_schema": step_packet.schema,
        "step_map_coder_mode": (
            "joint_p18_p19_lf_step_map_waterfill"
            if weighted_fit_enabled
            else "waterfill_mlx_native_uniform_importance_bridge"
        ),
        "step_map_coder_bins": None,
        "step_map_waterfill_bits_per_coeff": float(step_map_bits_per_coeff),
        "step_map_coder_groups": [dict(group) for group in step_packet.groups],
        "target_bits_per_coeff": float(target_bits_per_coeff),
        "uniform_quantization_levels": int(n_levels),
        "allocation_mode": (
            "joint_p18_p19_lf_waterfill_plus_hf_dwt_adjoint_saliency"
            if weighted_fit_enabled
            else "uniform_mlx_native_closed_form_export"
        ),
        "lf_step_allocation_mode": (
            "joint_p18_p19_dwt_adjoint_lf_reverse_waterfill" if weighted_fit_enabled else "uniform_l2_baseline"
        ),
        "lf_step_allocation_rows": lf_step_allocation_rows,
        "hf_decoder_fit_mode": hf_decoder_fit_mode,
        "hf_decoder_saliency_gain": float(hf_decoder_saliency_gain),
        "native_mlx_hf_decoder_training": mlx_training_report,
        "hf_decoder_weight_domain": (
            "dwt_adjoint_detail_saliency_diagonal" if weighted_fit_enabled else "unweighted_least_squares"
        ),
        "weight_domain": (
            "dwt_adjoint_detail_saliency_diagonal" if weighted_fit_enabled else "unweighted_least_squares"
        ),
        "exact_pixel_weighted_objective": False,
        "recon_pixel_weight_consumed": bool(weighted_fit_enabled),
        "recon_pixel_weight_metadata": dict(recon_pixel_weight_metadata or {}),
        "recon_pixel_weight_verified_gradient_manifest": verified_recon_weight,
        "contest_scorer_distortion_objective": bool(weighted_fit_enabled and verified_recon_weight),
        "score_aware_hf_decoder_fit_executed": bool(weighted_fit_enabled),
        "score_aware_long_training_executed": False,
        "native_mlx_training_executed": bool(mlx_training_report.get("executed") is True),
        "native_mlx_training_kind": (
            "hf_decoder_full_batch_gradient_descent" if mlx_training_report.get("executed") is True else "none"
        ),
        "decoder_payload_codec": str(decoder_payload_codec),
        "snerv_fc_dim": int(model_size.fc_dim),
        "snerv_emb_size": int(model_size.emb_size),
        "snerv_patch_radius": int(model_size.patch_radius),
        "snerv_model_size_adapter": model_size.adapter,
        "snerv_official_mfu_hfr_tub_numeric_primitives_requested": bool(
            model_size.official_mfu_hfr_tub_numeric_primitives_requested
        ),
        "snerv_official_mfu_hfr_tub_export_bound": False,
        "snerv_official_mfu_hfr_tub_export_blockers": list(model_size.official_mfu_hfr_tub_export_blockers),
        "snerv_spectra_preserving_adapter_enabled": (model_size.adapter == SNERV_SPECTRA_PRESERVING_ADAPTER),
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
        decoder_payload=decoder_payload,
        step_map_packet=step_packet.packet,
        metadata=metadata,
    )
    _verify_receiver_frame_decode(archive, reference_shape=pairs.shape)
    return archive


def _fit_hf_decoder_mlx_full_batch_gradient_descent(
    pyramids: list[WaveletPyramid],
    *,
    levels: int,
    initial_decoder: HfGenerationDecoder,
    detail_weight_pyramids: list[WaveletPyramid] | None,
    saliency_gain: float,
    temporal_group_count: int,
    steps: int,
    learning_rate: float,
    ridge: float,
) -> tuple[HfGenerationDecoder, dict[str, Any]]:
    if int(steps) <= 0:
        raise SnervMlxNativeExportError("native MLX decoder train steps must be positive")
    if float(learning_rate) <= 0.0:
        raise SnervMlxNativeExportError("native MLX decoder train lr must be positive")
    if float(ridge) < 0.0:
        raise SnervMlxNativeExportError("native MLX decoder train ridge must be non-negative")
    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - exercised on non-MLX hosts.
        raise SnervMlxNativeExportError(f"MLX import failed for SNeRV decoder training: {exc!s}") from exc

    matrices = _hf_decoder_training_matrices(
        pyramids,
        levels=int(levels),
        model_size=initial_decoder.model_size,
        detail_weight_pyramids=detail_weight_pyramids,
        saliency_gain=float(saliency_gain),
        temporal_group_count=int(temporal_group_count),
    )
    kernels: dict[int, dict[str, np.ndarray]] = {}
    loss_rows: list[dict[str, Any]] = []
    for lvl in range(int(levels)):
        kernels[lvl] = {}
        for subband in _DETAIL_KEYS:
            row = matrices[(lvl, subband)]
            x = mx.array(row["features"], dtype=mx.float32)
            y = mx.array(row["target"], dtype=mx.float32)
            weights = mx.array(row["weights"], dtype=mx.float32)
            denom = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=mx.float32))
            k = mx.array(
                np.asarray(initial_decoder.kernels[lvl][subband], dtype=np.float32).reshape(-1),
                dtype=mx.float32,
            )

            initial_loss = float(
                _mlx_weighted_linear_loss(
                    mx,
                    x=x,
                    y=y,
                    weights=weights,
                    denom=denom,
                    vec=k,
                    ridge=float(ridge),
                ).item()
            )
            for _step in range(int(steps)):
                pred = mx.matmul(x, k)
                residual = pred - y
                grad = (2.0 * mx.matmul(mx.transpose(x), weights * residual) / denom) + (2.0 * float(ridge) * k)
                k = k - (float(learning_rate) * grad)
                mx.eval(k)
            final_loss = float(
                _mlx_weighted_linear_loss(
                    mx,
                    x=x,
                    y=y,
                    weights=weights,
                    denom=denom,
                    vec=k,
                    ridge=float(ridge),
                ).item()
            )
            kernels[lvl][subband] = np.asarray(k, dtype=np.float64).reshape(
                _kernel_storage_shape(initial_decoder.model_size)
            )
            loss_rows.append(
                {
                    "level": int(lvl),
                    "subband": subband,
                    "sample_count": int(row["features"].shape[0]),
                    "feature_count": int(row["features"].shape[1]),
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "loss_delta": final_loss - initial_loss,
                }
            )
    all_final_losses_finite = bool(all(np.isfinite(float(row["final_loss"])) for row in loss_rows))
    worsened_rows = [
        row
        for row in loss_rows
        if _native_mlx_loss_worsened(
            initial=float(row["initial_loss"]),
            final=float(row["final_loss"]),
        )
    ]
    blockers = [
        "snerv_native_mlx_decoder_final_loss_nonfinite" if not all_final_losses_finite else "",
        "snerv_native_mlx_decoder_loss_worsened" if worsened_rows else "",
    ]
    blockers = [blocker for blocker in blockers if blocker]
    accepted = all_final_losses_finite and not blockers
    trained_decoder = HfGenerationDecoder(
        kernels=kernels,
        levels=int(levels),
        model_size=initial_decoder.model_size,
    )
    return (
        trained_decoder if accepted else initial_decoder,
        {
            "schema": "snerv_native_mlx_hf_decoder_training.v1",
            "attempted": True,
            "executed": bool(accepted),
            "accepted": bool(accepted),
            "optimizer": "full_batch_gradient_descent",
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "ridge": float(ridge),
            "saliency_gain": float(saliency_gain),
            "level_subband_rows": loss_rows,
            "mean_initial_loss": float(np.mean([row["initial_loss"] for row in loss_rows])),
            "mean_final_loss": float(np.mean([row["final_loss"] for row in loss_rows])),
            "all_final_losses_finite": all_final_losses_finite,
            "any_loss_worsened": bool(worsened_rows),
            "worsened_level_subband_rows": [
                {
                    "level": int(row["level"]),
                    "subband": str(row["subband"]),
                    "initial_loss": float(row["initial_loss"]),
                    "final_loss": float(row["final_loss"]),
                    "loss_delta": float(row["loss_delta"]),
                }
                for row in worsened_rows
            ],
            "loss_worsen_relative_tolerance": NATIVE_MLX_DECODER_LOSS_WORSEN_REL_TOL,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        },
    )


def _hf_decoder_training_matrices(
    pyramids: list[WaveletPyramid],
    *,
    levels: int,
    model_size: SnervModelSizeConfig,
    detail_weight_pyramids: list[WaveletPyramid] | None,
    saliency_gain: float,
    temporal_group_count: int,
) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    if not pyramids:
        raise SnervMlxNativeExportError("native MLX decoder training needs pyramids")
    if detail_weight_pyramids is not None and len(detail_weight_pyramids) != len(pyramids):
        raise SnervMlxNativeExportError("detail weight pyramid count mismatch")
    feature_count = int(model_size.feature_count)
    lf_sequence_all = [np.asarray(pyr.lf, dtype=np.float64) for pyr in pyramids]
    rows: dict[tuple[int, str], dict[str, list[np.ndarray]]] = {
        (lvl, subband): {"features": [], "target": [], "weights": []}
        for lvl in range(int(levels))
        for subband in _DETAIL_KEYS
    }
    for pyr_idx, pyr in enumerate(pyramids):
        temporal_sequence = None
        temporal_index = None
        if int(model_size.temporal_context) > 0:
            temporal_sequence, temporal_index = _flat_temporal_group_for_native_export(
                lf_sequence_all,
                flat_index=pyr_idx,
                group_count=int(temporal_group_count),
            )
        approx = np.asarray(pyr.lf, dtype=np.float64)
        for lvl, (lh, hl, hh) in enumerate(pyr.details):
            target_hw = (int(lh.shape[0]), int(lh.shape[1]))
            up = _upsample_nn(approx, target_hw)
            features = _decoder_features(
                up,
                model_size,
                lf_sequence=temporal_sequence,
                sequence_index=temporal_index,
            ).reshape(-1, feature_count)
            for subband, detail in (("LH", lh), ("HL", hl), ("HH", hh)):
                correction = _hfr_for_model_size(model_size).correction(
                    up,
                    subband=subband,
                    target_hw=tuple(int(v) for v in detail.shape),
                )
                target = np.asarray(detail, dtype=np.float64) - correction
                weights = _native_export_detail_weights(
                    detail_weight_pyramids,
                    pyramid_index=pyr_idx,
                    level=lvl,
                    subband=subband,
                    expected_shape=detail.shape,
                    saliency_gain=float(saliency_gain),
                )
                bucket = rows[(lvl, subband)]
                bucket["features"].append(features.astype(np.float32))
                bucket["target"].append(target.reshape(-1).astype(np.float32))
                bucket["weights"].append(weights.reshape(-1).astype(np.float32))
            approx = idwt2_multilevel(
                WaveletPyramid(
                    coeffs=[approx, (lh, hl, hh)],
                    levels=1,
                    wavelet=pyr.wavelet,
                    orig_hw=(target_hw[0] * 2, target_hw[1] * 2),
                )
            )
    return {
        key: {
            "features": np.concatenate(value["features"], axis=0),
            "target": np.concatenate(value["target"], axis=0),
            "weights": np.concatenate(value["weights"], axis=0),
        }
        for key, value in rows.items()
    }


def _native_export_detail_weights(
    detail_weight_pyramids: list[WaveletPyramid] | None,
    *,
    pyramid_index: int,
    level: int,
    subband: str,
    expected_shape: tuple[int, ...],
    saliency_gain: float,
) -> np.ndarray:
    if detail_weight_pyramids is None:
        return np.ones(expected_shape, dtype=np.float32)
    subband_idx = _DETAIL_KEYS.index(subband)
    raw = np.abs(
        np.asarray(
            detail_weight_pyramids[int(pyramid_index)].details[int(level)][subband_idx],
            dtype=np.float64,
        )
    )
    if raw.shape != expected_shape:
        raise SnervMlxNativeExportError(f"detail weight shape {raw.shape} != expected {expected_shape}")
    if not np.isfinite(raw).all():
        raise SnervMlxNativeExportError("detail weights contain nonfinite values")
    mean = float(np.mean(raw))
    scaled = np.ones(expected_shape, dtype=np.float64) if mean <= 0.0 else raw / mean
    return np.maximum(1.0e-3, 1.0 + float(saliency_gain) * (scaled - 1.0)).astype(np.float32)


def _native_mlx_loss_worsened(*, initial: float, final: float) -> bool:
    if not np.isfinite(float(initial)) or not np.isfinite(float(final)):
        return True
    tolerance = NATIVE_MLX_DECODER_LOSS_WORSEN_REL_TOL * max(
        1.0,
        abs(float(initial)),
    )
    return float(final) > float(initial) + tolerance


def _recon_pixel_weight_metadata_is_verified_gradient_manifest(
    metadata: Mapping[str, Any] | None,
) -> bool:
    """Return True only for file-backed, finite-gradient P18/P19 custody."""

    if not isinstance(metadata, Mapping):
        return False
    producer = metadata.get("producer_manifest")
    if not isinstance(producer, Mapping):
        return False
    return (
        metadata.get("producer_manifest_verified") is True
        and metadata.get("verification_status") == "verified_finite_gradient_manifest"
        and producer.get("status") == "verified_finite_gradient_manifest"
        and producer.get("consumption_certified") is True
        and bool(producer.get("weight_sha256"))
        and bool(metadata.get("sha256"))
        and str(producer.get("weight_sha256")) == str(metadata.get("sha256"))
    )


def _mlx_weighted_linear_loss(
    mx: Any,
    *,
    x: Any,
    y: Any,
    weights: Any,
    denom: Any,
    vec: Any,
    ridge: float,
) -> Any:
    residual = mx.matmul(x, vec) - y
    data = mx.sum(weights * residual * residual) / denom
    reg = float(ridge) * mx.sum(vec * vec)
    return data + reg


def _packet_source_from_snerv_native_metadata(metadata: Mapping[str, Any]) -> str:
    if metadata.get("native_mlx_training_executed") is True:
        return str(metadata.get("hf_decoder_fit_mode") or "native_mlx_hf_decoder_training")
    if metadata.get("recon_pixel_weight_consumed") is True:
        return "mlx_target_hydration_numpy_joint_p18_p19_dwt_adjoint_saliency_weighted_decoder_fit"
    return "mlx_target_hydration_numpy_closed_form_decoder_fit"


def _flat_temporal_group_for_native_export(
    lf_sequence_all: Sequence[np.ndarray],
    *,
    flat_index: int,
    group_count: int,
) -> tuple[Sequence[np.ndarray], int]:
    group_count = int(group_count)
    if group_count < 1:
        raise SnervMlxNativeExportError("temporal group count must be positive")
    group_start = (int(flat_index) // group_count) * group_count
    group = lf_sequence_all[group_start : group_start + group_count]
    if len(group) != group_count:
        group = lf_sequence_all[max(0, len(lf_sequence_all) - group_count) :]
    if not group:
        raise SnervMlxNativeExportError("empty temporal group")
    return group, min(int(flat_index) - group_start, len(group) - 1)


def build_snerv_mlx_native_storage_preflight(
    *,
    output_dir: str | Path,
    n_pairs: int,
    packet_bytes: int,
    extra_margin_bytes: int = 512 * 1024 * 1024,
) -> dict[str, Any]:
    """Return free-space readiness for receiver proof and archive export."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    raw_bytes = int(n_pairs) * 2 * 3 * int(CAMERA_HW[0]) * int(CAMERA_HW[1])
    required = raw_bytes + int(packet_bytes) + int(extra_margin_bytes)
    free = shutil.disk_usage(out).free
    return {
        "schema": SNERV_MLX_NATIVE_STORAGE_PREFLIGHT_SCHEMA,
        "path": out.as_posix(),
        "n_pairs": int(n_pairs),
        "expected_receiver_raw_bytes": raw_bytes,
        "packet_bytes": int(packet_bytes),
        "extra_margin_bytes": int(extra_margin_bytes),
        "required_bytes": int(required),
        "free_bytes": int(free),
        "preflight_passed": bool(int(free) >= int(required)),
        **FALSE_AUTHORITY,
    }


def _target_pairs_to_nchw255(target0_np: np.ndarray, target1_np: np.ndarray) -> np.ndarray:
    if target0_np.shape != target1_np.shape:
        raise SnervMlxNativeExportError(f"target frame shapes differ: {target0_np.shape} vs {target1_np.shape}")
    if target0_np.ndim != 4 or target0_np.shape[-1] != 3:
        raise SnervMlxNativeExportError(f"targets must be NHWC RGB shaped (pairs, H, W, 3); got {target0_np.shape}")
    pair = np.stack([target0_np, target1_np], axis=1)
    pair = np.clip(pair.astype(np.float32), 0.0, 1.0) * 255.0
    return np.transpose(pair, (0, 1, 4, 2, 3)).astype(np.float32)


def _source_pair_indices_for_native_export(
    num_pairs: int,
    *,
    pair_indices: Sequence[Any] | str | None,
) -> tuple[int, ...]:
    if pair_indices is None:
        if int(num_pairs) < 1:
            raise SnervMlxNativeExportError(f"num_pairs must be >= 1; got {num_pairs}")
        return tuple(range(int(num_pairs)))
    normalized = normalize_pair_indices(
        pair_indices,
        field="snerv_mlx_native_pair_indices",
    )
    if not normalized:
        raise SnervMlxNativeExportError("pair_indices must contain at least one pair")
    return tuple(int(value) for value in normalized)


def _source_pair_indices_for_packet(
    n_pairs: int,
    *,
    source_pair_indices: Sequence[Any] | str | None,
) -> tuple[int, ...]:
    if source_pair_indices is None:
        return tuple(range(int(n_pairs)))
    normalized = normalize_pair_indices(
        source_pair_indices,
        field="snerv_mlx_native_source_pair_indices",
    )
    if len(normalized) != int(n_pairs):
        raise SnervMlxNativeExportError(
            f"source_pair_indices length {len(normalized)} does not match packet pair count {int(n_pairs)}"
        )
    return tuple(int(value) for value in normalized)


def _packet_source_pair_indices(packet: bytes) -> tuple[int, ...] | None:
    if not packet:
        return None
    try:
        metadata = unpack_snerv_archive(packet).metadata
    except Exception:
        return None
    raw = metadata.get("source_pair_indices")
    if raw is None:
        return None
    try:
        return normalize_pair_indices(
            raw,
            field="snerv_mlx_native_packet_source_pair_indices",
        )
    except Exception:
        return None


def _packet_preserves_source_pair_indices(
    packet: bytes,
    *,
    expected_source_pair_indices: Sequence[int],
    explicit_pair_indices: bool,
) -> bool:
    actual = _packet_source_pair_indices(packet)
    expected = tuple(int(value) for value in expected_source_pair_indices)
    if actual is None:
        return (not explicit_pair_indices) and expected == tuple(range(len(expected)))
    return tuple(actual) == expected


def _packet_preserves_recon_weight_binding(
    packet: bytes,
    recon_weight_metadata: Mapping[str, Any] | None,
) -> bool:
    if recon_weight_metadata is None:
        return True
    expected_sha = recon_weight_metadata.get("sha256")
    if not expected_sha:
        return False
    try:
        metadata = unpack_snerv_archive(packet).metadata
    except Exception:
        return False
    if metadata.get("recon_pixel_weight_consumed") is not True:
        return False
    actual = metadata.get("recon_pixel_weight_metadata")
    if not isinstance(actual, Mapping):
        return False
    return str(actual.get("sha256") or "") == str(expected_sha)


def _load_recon_pixel_weight_for_native_export(
    path: str | Path | None,
    *,
    manifest_path: str | Path | None = None,
    expected_pairs: int,
    expected_hw: tuple[int, int],
    normalize: str,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    if normalize not in {"mean", "none"}:
        raise SnervMlxNativeExportError("recon_pixel_weight_normalize must be 'mean' or 'none'")
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise SnervMlxNativeExportError(f"recon_pixel_weight_path is not a file: {resolved}")
    npz_key: str | None = None
    if resolved.suffix == ".npz":
        with np.load(resolved) as data:
            keys = sorted(str(key) for key in data.files)
            if not keys:
                raise SnervMlxNativeExportError(f"recon_pixel_weight npz is empty: {resolved}")
            npz_key = "weight" if "weight" in data.files else keys[0]
            raw = np.asarray(data[npz_key], dtype=np.float32)
    else:
        raw = np.asarray(np.load(resolved), dtype=np.float32)
    weight = _normalize_recon_pixel_weight_array(
        raw,
        expected_pairs=int(expected_pairs),
        expected_hw=expected_hw,
        normalize=normalize,
    )
    source_sha256 = sha256_file(resolved)
    producer_manifest = _load_recon_pixel_weight_producer_manifest_for_native_export(
        resolved,
        expected_weight_sha256=source_sha256,
        explicit_manifest_path=manifest_path,
        expected_pairs=int(expected_pairs),
        expected_hw=expected_hw,
    )
    producer_manifest_verified = (
        producer_manifest.get("consumption_certified") is True
        and producer_manifest.get("status") == "verified_finite_gradient_manifest"
    )
    metadata = {
        "schema": "snerv_mlx_native_recon_pixel_weight_consumption.v1",
        "enabled": True,
        "source_kind": "file",
        "path": resolved.as_posix(),
        "sha256": source_sha256,
        "npz_key": npz_key,
        "normalize": normalize,
        "expected_pairs": int(expected_pairs),
        "expected_hw": [int(expected_hw[0]), int(expected_hw[1])],
        "consumed_shape": [int(v) for v in weight.shape],
        "stats": _array_stats(weight),
        "hf_decoder_fit_mode": SNERV_DWT_ADJOINT_SALIENCY_WEIGHTED_FIT_MODE,
        "weight_domain": "dwt_adjoint_detail_saliency_diagonal",
        "exact_pixel_weighted_objective": False,
        "producer_manifest": producer_manifest,
        "producer_manifest_verified": producer_manifest_verified,
        "verification_status": (
            "verified_finite_gradient_manifest"
            if producer_manifest_verified
            else "manual_file_sha_only_no_verified_gradient_manifest"
        ),
        "authority": "false_macos_mlx_research_signal",
        **FALSE_AUTHORITY,
    }
    return weight, metadata


def _load_recon_pixel_weight_producer_manifest_for_native_export(
    weight_path: Path,
    *,
    expected_weight_sha256: str,
    explicit_manifest_path: str | Path | None,
    expected_pairs: int,
    expected_hw: tuple[int, int],
) -> dict[str, Any]:
    """Validate finite-gradient producer custody for a native-export weight.

    The compact runner can auto-discover the newest joint P18/P19 recon-weight,
    but the native exporter is the archive-producing boundary. It must re-check
    the producer manifest itself so manifest custody survives direct calls and
    cannot be forged by runner metadata alone.
    """

    if explicit_manifest_path is None:
        manifest_path = weight_path.with_name("joint_p18_p19_recon_pixel_weight_manifest.json")
        if not manifest_path.is_file():
            return {
                "schema": "snerv_mlx_native_recon_pixel_weight_producer_manifest.v1",
                "status": "not_found_unverified_manual_or_legacy_weight",
                "path": manifest_path.as_posix(),
                "consumption_certified": False,
            }
    else:
        manifest_path = _resolve_recon_pixel_weight_manifest_path(
            explicit_manifest_path,
            base=weight_path.parent,
        )
        if not manifest_path.is_file():
            raise SnervMlxNativeExportError(f"recon_pixel_weight_manifest_path is not a file: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SnervMlxNativeExportError(
            f"recon pixel weight producer manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise SnervMlxNativeExportError(f"recon pixel weight producer manifest must be an object: {manifest_path}")

    manifest_weight_path = manifest.get("weight_path")
    if manifest_weight_path is None:
        raise SnervMlxNativeExportError("recon pixel weight producer manifest is missing weight_path")
    manifest_resolved = _resolve_recon_pixel_weight_manifest_path(
        manifest_weight_path,
        base=manifest_path.parent,
    )
    if manifest_resolved != weight_path:
        raise SnervMlxNativeExportError(
            "recon pixel weight producer manifest points at a different "
            f"weight file: {manifest_resolved} != {weight_path}"
        )

    manifest_weight_sha = str(manifest.get("weight_sha256") or "")
    if manifest_weight_sha != expected_weight_sha256:
        raise SnervMlxNativeExportError(
            "recon pixel weight producer manifest SHA does not match loaded "
            f"weight file: {manifest_weight_sha} != {expected_weight_sha256}"
        )

    config = manifest.get("config")
    if isinstance(config, Mapping):
        if config.get("num_pairs") is not None and int(config["num_pairs"]) != int(expected_pairs):
            raise SnervMlxNativeExportError(
                "recon pixel weight producer manifest num_pairs mismatch: "
                f"{config.get('num_pairs')} != {int(expected_pairs)}"
            )
        scorer_hw = config.get("scorer_hw")
        if scorer_hw is not None and [int(v) for v in scorer_hw] != [
            int(expected_hw[0]),
            int(expected_hw[1]),
        ]:
            raise SnervMlxNativeExportError(
                "recon pixel weight producer manifest scorer_hw mismatch: "
                f"{scorer_hw} != {[int(expected_hw[0]), int(expected_hw[1])]}"
            )

    producer_metadata = manifest.get("metadata")
    if not isinstance(producer_metadata, dict):
        raise SnervMlxNativeExportError("recon pixel weight producer manifest is missing metadata object")
    gradient_health = producer_metadata.get("gradient_health")
    if not isinstance(gradient_health, dict):
        raise SnervMlxNativeExportError(
            "recon pixel weight producer manifest is missing gradient_health; "
            "regenerate the surface with the finite-gradient producer"
        )
    blockers = [str(blocker) for blocker in producer_metadata.get("blockers") or []]
    consumption_recommended = bool(producer_metadata.get("training_consumption_recommended", False))
    if gradient_health.get("status") != "pass_finite":
        raise SnervMlxNativeExportError(
            f"recon pixel weight producer manifest did not pass finite-gradient health: {gradient_health.get('status')}"
        )
    if not consumption_recommended or blockers:
        raise SnervMlxNativeExportError(
            f"recon pixel weight producer manifest is not recommended for training consumption; blockers={blockers}"
        )

    return {
        "schema": "snerv_mlx_native_recon_pixel_weight_producer_manifest.v1",
        "status": "verified_finite_gradient_manifest",
        "path": manifest_path.as_posix(),
        "sha256": sha256_file(manifest_path),
        "producer_schema": manifest.get("schema"),
        "producer_metadata_schema": producer_metadata.get("schema"),
        "weight_path": manifest_resolved.as_posix(),
        "weight_sha256": expected_weight_sha256,
        "config": dict(config) if isinstance(config, Mapping) else None,
        "gradient_health": dict(gradient_health),
        "blockers": blockers,
        "training_consumption_recommended": consumption_recommended,
        "consumption_certified": True,
    }


def _resolve_recon_pixel_weight_manifest_path(path: str | Path, *, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _normalize_recon_pixel_weight_array(
    value: Any,
    *,
    expected_pairs: int,
    expected_hw: tuple[int, int],
    normalize: str,
) -> np.ndarray:
    if normalize not in {"mean", "none"}:
        raise SnervMlxNativeExportError("recon_pixel_weight normalize must be 'mean' or 'none'")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        h, w = arr.shape
        if (int(h), int(w)) != expected_hw:
            raise SnervMlxNativeExportError(f"recon_pixel_weight spatial shape {(int(h), int(w))} != {expected_hw}")
        arr = arr[None, None, :, :, None]
    elif arr.ndim == 3:
        h, w, channels = arr.shape
        if (int(h), int(w)) != expected_hw:
            raise SnervMlxNativeExportError(f"recon_pixel_weight spatial shape {(int(h), int(w))} != {expected_hw}")
        if int(channels) not in (1, 3):
            raise SnervMlxNativeExportError("recon_pixel_weight channel count must be 1 or 3")
        arr = arr[None, None, :, :, :]
    elif arr.ndim == 4:
        leading, h, w, channels = arr.shape
        if int(leading) not in (1, int(expected_pairs)):
            raise SnervMlxNativeExportError(
                "4D recon_pixel_weight leading dimension must be 1 or "
                f"expected_pairs={int(expected_pairs)}; got {int(leading)}"
            )
        if (int(h), int(w)) != expected_hw:
            raise SnervMlxNativeExportError(f"recon_pixel_weight spatial shape {(int(h), int(w))} != {expected_hw}")
        if int(channels) not in (1, 3):
            raise SnervMlxNativeExportError("recon_pixel_weight channel count must be 1 or 3")
        arr = arr[:, None, :, :, :]
    elif arr.ndim == 5:
        pairs, frames, h, w, channels = arr.shape
        if int(pairs) not in (1, int(expected_pairs)):
            raise SnervMlxNativeExportError(
                "5D recon_pixel_weight pair dimension must be 1 or "
                f"expected_pairs={int(expected_pairs)}; got {int(pairs)}"
            )
        if int(frames) != 2:
            raise SnervMlxNativeExportError("5D recon_pixel_weight frame dimension must be 2")
        if (int(h), int(w)) != expected_hw:
            raise SnervMlxNativeExportError(f"recon_pixel_weight spatial shape {(int(h), int(w))} != {expected_hw}")
        if int(channels) not in (1, 3):
            raise SnervMlxNativeExportError("recon_pixel_weight channel count must be 1 or 3")
    else:
        raise SnervMlxNativeExportError(
            "recon_pixel_weight must be shaped (H,W), (H,W,1/3), (1|N,H,W,1/3), or (1|N,2,H,W,1/3)"
        )
    if not np.isfinite(arr).all():
        raise SnervMlxNativeExportError("recon_pixel_weight must be finite")
    if float(np.min(arr)) < 0.0:
        raise SnervMlxNativeExportError("recon_pixel_weight must be non-negative")
    if float(np.mean(arr)) <= 0.0:
        raise SnervMlxNativeExportError("recon_pixel_weight must have positive total mass")
    arr = np.broadcast_to(
        arr,
        (
            int(expected_pairs),
            2,
            int(expected_hw[0]),
            int(expected_hw[1]),
            int(arr.shape[-1]),
        ),
    ).astype(np.float32, copy=True)
    if normalize == "mean":
        mean = float(np.mean(arr))
        if mean <= 0.0 or not np.isfinite(mean):
            raise SnervMlxNativeExportError("recon_pixel_weight mean is not positive finite")
        arr = arr / mean
    return np.ascontiguousarray(arr, dtype=np.float32)


def _array_stats(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "shape": [int(v) for v in arr.shape],
        "dtype": str(np.asarray(value).dtype),
        "min": float(np.min(finite)) if finite.size else 0.0,
        "max": float(np.max(finite)) if finite.size else 0.0,
        "mean": float(np.mean(finite)) if finite.size else 0.0,
        "std": float(np.std(finite)) if finite.size else 0.0,
        "nonzero_fraction": float(np.count_nonzero(arr) / max(int(arr.size), 1)),
        "nonfinite_count": int(arr.size - finite.size),
    }


def _verify_receiver_frame_decode(
    archive: SnervArchivePacket,
    *,
    reference_shape: Sequence[int],
) -> None:
    decoded = decode_snerv_archive_frames(archive.packet)
    if tuple(int(v) for v in decoded.shape) != tuple(int(v) for v in reference_shape):
        raise SnervMlxNativeExportError(f"receiver decode shape {decoded.shape} != reference {tuple(reference_shape)}")
    if not np.isfinite(decoded).all():
        raise SnervMlxNativeExportError("receiver decode produced nonfinite values")


def _official_primitives_packet_metadata(
    official_binding: Mapping[str, Any] | None,
    *,
    blockers: Sequence[str],
) -> dict[str, Any]:
    if official_binding is None:
        return {}
    return {
        "snerv_official_mfu_hfr_tub_receiver_bound_surrogate_export": True,
        "snerv_official_mfu_hfr_tub_receiver_bound_surrogate_kind": (
            "snar1_linear_hf_generation_decoder_not_official_neural_graph"
        ),
        "snerv_official_mfu_hfr_tub_export_bound": False,
        "snerv_official_mfu_hfr_tub_export_blockers": [str(blocker) for blocker in blockers],
        "snerv_official_mfu_hfr_tub_binding_schema": official_binding.get("schema"),
        "source_faithful_stack": False,
        **FALSE_AUTHORITY,
    }


def _receiver_bound_official_primitives_export_binding(
    official_binding: Mapping[str, Any],
    *,
    packet_path: Path,
    packet_bytes: int,
    packet_sha256: str,
    selected_archive_metadata: Mapping[str, Any],
    package: Mapping[str, Any] | None,
    receiver_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the official-primitives request to a real receiver packet, fail-closed."""

    blockers = [str(blocker) for blocker in official_binding.get("blockers") or []]
    proof_passed = receiver_proof.get("runtime_consumption_proof_passed") is True
    receiver_satisfied = receiver_proof.get("receiver_contract_satisfied") is True
    archive_path = receiver_proof.get("archive_path")
    archive_sha256 = receiver_proof.get("archive_sha256")
    archive_bytes = receiver_proof.get("archive_bytes")
    out = dict(official_binding)
    out["schema"] = "snerv_official_mfu_hfr_tub_export_binding.v3"
    out["export_bound_to_receiver_packet"] = True
    out["official_export_bound"] = False
    out["surrogate_receiver_payload_contract_emitted"] = True
    out["official_receiver_payload_contract_emitted"] = False
    out["receiver_bound_surrogate_export"] = {
        "schema": "snerv_official_receiver_bound_surrogate_export.v1",
        "kind": "snar1_linear_hf_generation_decoder_not_official_neural_graph",
        "packet_path": packet_path.as_posix(),
        "packet_bytes": int(packet_bytes),
        "packet_sha256": str(packet_sha256),
        "packet_source_faithful_stack": False,
        "packet_decoder_payload_codec": selected_archive_metadata.get("decoder_payload_codec"),
        "packet_decoder_payload_schema": out.get("current_snar_decoder_payload_schema"),
        "packet_receiver_decode_verified_by_builder": True,
        "archive_path": str(archive_path) if archive_path else None,
        "archive_bytes": int(archive_bytes) if archive_bytes is not None else None,
        "archive_sha256": str(archive_sha256) if archive_sha256 else None,
        "receiver_proof_path": (str(receiver_proof.get("proof_path")) if receiver_proof.get("proof_path") else None),
        "surrogate_receiver_contract_satisfied": bool(receiver_satisfied),
        "surrogate_runtime_consumption_proof_passed": bool(proof_passed),
        "archive_package_present": package is not None,
        **FALSE_AUTHORITY,
    }
    out["blocker_evidence"] = _official_primitives_blocker_evidence(
        blockers,
        surrogate_receiver_runtime_decode_passed=bool(proof_passed),
        surrogate_receiver_contract_satisfied=bool(receiver_satisfied),
    )
    out["unclosed_official_blockers"] = blockers
    out["blockers"] = blockers
    out["export_consumed_official_mfu"] = False
    out["export_consumed_official_hfr"] = False
    out["export_consumed_official_tub"] = False
    out["source_forward_replay_authority"] = False
    out["receiver_runtime_decode_authority"] = False
    out.update(FALSE_AUTHORITY)
    return out


def _official_primitives_blocker_evidence(
    blockers: Sequence[str],
    *,
    surrogate_receiver_runtime_decode_passed: bool,
    surrogate_receiver_contract_satisfied: bool,
) -> list[dict[str, Any]]:
    specs = {
        "snerv_official_neural_decoder_payload_grammar_missing": {
            "missing_artifact": ("receiver-visible official decoder graph topology and section grammar"),
            "current_evidence": ("SNAR1 decoder_payload stores HfGenerationDecoder linear kernels only"),
            "closure_test": (
                "decode official graph payload into executable MFU/HFR/TUB/T-output "
                "receiver objects without scorer or torch sidecars"
            ),
        },
        "snerv_official_mfu_hfr_tub_weight_mapping_missing": {
            "missing_artifact": ("official state_dict-to-receiver tensor map with per-tensor bytes and sha256"),
            "current_evidence": ("portable MFU/HFR/TUB primitives exist, but export does not consume official weights"),
            "closure_test": (
                "map official encoder/MFU/HFR/TUB/residual-block tensors into the "
                "receiver payload and verify every mapped tensor hash"
            ),
        },
        "snerv_official_mfu_hfr_tub_source_forward_replay_missing": {
            "missing_artifact": ("same-input official torch forward replay against the portable receiver graph"),
            "current_evidence": ("surrogate SNAR1 packet decodes, but no official source graph output is compared"),
            "closure_test": (
                "run official source forward and portable receiver forward on the "
                "same frames/weights and record max error plus output sha256"
            ),
        },
        "snerv_official_receiver_runtime_decode_missing": {
            "missing_artifact": ("inflate runtime that consumes official neural graph sections"),
            "current_evidence": ("existing runtime consumes linear SNAR1 sections only"),
            "closure_test": (
                "generated inflate.sh decodes official graph payload sections into 0.raw with matching receiver proof"
            ),
        },
    }
    rows = []
    for blocker in blockers:
        spec = specs.get(str(blocker), {})
        rows.append(
            {
                "blocker": str(blocker),
                "closed": False,
                "missing_artifact": spec.get("missing_artifact", "unspecified"),
                "current_evidence": spec.get("current_evidence", "unspecified"),
                "closure_test": spec.get("closure_test", "unspecified"),
                "surrogate_receiver_runtime_decode_passed": bool(surrogate_receiver_runtime_decode_passed),
                "surrogate_receiver_contract_satisfied": bool(surrogate_receiver_contract_satisfied),
                "official_authority": False,
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _blocked_official_primitives_native_export(
    *,
    output_dir: Path,
    model_size: SnervModelSizeConfig,
    candidate: Mapping[str, Any],
    source_video_path: str | Path,
    scorer_upstream_dir: str | Path,
    source_pair_indices: Sequence[int],
    explicit_pair_indices: bool,
    num_pairs: int,
    started_monotonic: float,
) -> dict[str, Any]:
    blockers = list(model_size.official_mfu_hfr_tub_export_blockers)
    official_binding = _official_primitives_export_binding(
        model_size=model_size,
        candidate=candidate,
        blockers=blockers,
    )
    payload = {
        "schema": "snerv_mlx_native_train_export.v1",
        "output_dir": output_dir.as_posix(),
        "executed": False,
        "failure": "official_mfu_hfr_tub_numeric_primitives_requested_but_export_not_bound",
        "family": "snerv",
        "source_video_path": Path(source_video_path).as_posix(),
        "scorer_upstream_dir": Path(scorer_upstream_dir).as_posix(),
        "num_pairs": int(num_pairs),
        "source_pair_indices": [int(value) for value in source_pair_indices],
        "pair_index_alignment_mode": (
            "explicit_source_pair_indices" if explicit_pair_indices else "prefix_source_pair_indices"
        ),
        "candidate_id": candidate.get("candidate_id"),
        "model_size": model_size.as_jsonable(),
        "snerv_model_size_adapter": model_size.adapter,
        "snerv_official_mfu_hfr_tub_numeric_primitives_requested": True,
        "snerv_official_mfu_hfr_tub_export_bound": False,
        "snerv_official_mfu_hfr_tub_export_blockers": blockers,
        "official_primitive_binding": official_binding,
        "packet_path": None,
        "packet_bytes": None,
        "packet_sha256": None,
        "archive_path": None,
        "archive_bytes": None,
        "archive_sha256": None,
        "receiver_proof_path": None,
        "receiver_proof_passed": False,
        "receiver_contract_satisfied": False,
        "native_mlx_training_executed": False,
        "native_mlx_training_kind": "none",
        "native_mlx_hf_decoder_training": {
            "schema": "snerv_native_mlx_hf_decoder_training.v1",
            "requested_steps": 0,
            "executed": False,
            "blockers": ["snerv_official_mfu_hfr_tub_export_blocked_before_decoder_training"],
            **FALSE_AUTHORITY,
        },
        "blockers": blockers,
        "elapsed_seconds": float(time.monotonic() - float(started_monotonic)),
        **FALSE_AUTHORITY,
    }
    report_path = output_dir / SNERV_MLX_NATIVE_REPORT_FILENAME
    write_json(report_path, payload)
    payload["report_path"] = report_path.as_posix()
    payload["report_sha256"] = sha256_file(report_path)
    write_json(report_path, payload)
    return payload


def _official_primitives_export_binding(
    *,
    model_size: SnervModelSizeConfig,
    candidate: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    """Describe the exact official-SNeRV export gap without granting authority."""

    required_sections = (
        "official_encoder_embedding_payload",
        "official_mfu_weight_payload",
        "official_hfr_weight_payload",
        "official_tub_weight_payload",
        "official_idwt_or_wavelet_payload",
        "official_decoder_graph_topology_payload",
    )
    consumed_controls = {
        "fc_dim": int(model_size.fc_dim),
        "emb_size": int(model_size.emb_size),
        "patch_radius": int(model_size.patch_radius),
        "mfu_scales": [int(v) for v in model_size.mfu_scales],
        "hfr_gain": float(model_size.hfr_gain),
        "temporal_context": int(model_size.temporal_context),
        "temporal_mode": str(model_size.temporal_mode),
        "candidate_id": candidate.get("candidate_id"),
        "official_modelsize_solution": candidate.get("official_modelsize_solution"),
    }
    return {
        "schema": "snerv_official_mfu_hfr_tub_export_binding.v2",
        "adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        "primitive_modules_available": True,
        "current_snar_decoder_payload_schema": "linear_hf_generation_decoder_only",
        "linear_hf_generation_decoder_compatible_with_official_neural_graph": False,
        "receiver_payload_contract_emitted": True,
        "required_receiver_payload_sections": list(required_sections),
        "missing_receiver_payload_sections": list(required_sections),
        "source_pins": {
            "official_head_sha": OFFICIAL_SNERV_HFR_SOURCE_SHA,
            "official_mfu_source": OFFICIAL_SNERV_MFU_SOURCE,
            "official_residual_block_source": OFFICIAL_SNERV_RB_SOURCE,
            "official_hfr_source_contract": OFFICIAL_SNERV_HFR_SOURCE_CONTRACT,
            "official_hfr_numeric_proof": SNERV_OFFICIAL_HFR_CONVBLOCK_NUMPY_PROOF,
            "official_tub_source_sha": OFFICIAL_SNERV_T_SOURCE_SHA,
            "official_tub_source_contract": OFFICIAL_SNERV_T_TUB_SOURCE_CONTRACT,
            "official_tub_schema": OFFICIAL_SNERV_T_TUB_SCHEMA,
        },
        "numeric_parity_blockers": list(OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS),
        "candidate_controls_consumed": {key: value for key, value in consumed_controls.items() if value is not None},
        "export_consumed_official_mfu": False,
        "export_consumed_official_hfr": False,
        "export_consumed_official_tub": False,
        "source_forward_replay_authority": False,
        "receiver_runtime_decode_authority": False,
        "blockers": list(blockers),
        **FALSE_AUTHORITY,
    }


def _model_size_from_candidate(candidate: Mapping[str, Any]) -> SnervModelSizeConfig:
    scales = candidate.get("mfu_scales", candidate.get("snerv_mfu_scales", (1, 2, 4)))
    if isinstance(scales, str):
        scales_tuple = tuple(int(v) for v in scales.split(",") if v.strip())
    else:
        scales_tuple = tuple(int(v) for v in scales)
    adapter = str(
        candidate.get(
            "model_size_adapter",
            candidate.get("snerv_model_size_adapter", "snerv_fc_dim_emb_size_adapter_v1"),
        )
    )
    if bool(
        candidate.get(
            "snerv_official_mfu_hfr_tub_numeric_primitives",
            candidate.get("use_official_snerv_primitives", False),
        )
    ):
        adapter = SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER
    fc_dim = _fc_dim_from_candidate(candidate)
    return SnervModelSizeConfig(
        fc_dim=fc_dim,
        emb_size=int(candidate.get("emb_size", candidate.get("snerv_emb_size", 0))),
        patch_radius=int(candidate.get("patch_radius", candidate.get("snerv_patch_radius", 1))),
        mfu_scales=scales_tuple,
        hfr_gain=float(candidate.get("hfr_gain", candidate.get("snerv_hfr_gain", 0.0))),
        temporal_context=int(
            candidate.get(
                "temporal_context",
                candidate.get("snerv_temporal_context", 0),
            )
        ),
        temporal_mode=str(
            candidate.get(
                "temporal_mode",
                candidate.get("snerv_temporal_mode", "delta"),
            )
        ),
        adapter=adapter,
    )


def _fc_dim_from_candidate(candidate: Mapping[str, Any]) -> int:
    if candidate.get("fc_dim") is not None:
        return int(candidate["fc_dim"])
    if candidate.get("snerv_fc_dim") is not None:
        return int(candidate["snerv_fc_dim"])
    solution = candidate.get("official_modelsize_solution")
    if isinstance(solution, Mapping) and solution.get("fc_dim") is not None:
        return int(solution["fc_dim"])
    modelsize = candidate.get("modelsize_mparams", candidate.get("official_modelsize_mparams"))
    if modelsize is not None:
        full_data_length = candidate.get("full_data_length")
        final_size = candidate.get("final_size")
        enc_strds = candidate.get("enc_strds", candidate.get("official_enc_strds"))
        dec_strds = candidate.get("dec_strds", candidate.get("official_dec_strds"))
        if full_data_length is not None and final_size is not None and enc_strds is not None and dec_strds is not None:
            return int(
                official_snerv_modelsize_to_fc_dim(
                    modelsize_mparams=float(modelsize),
                    full_data_length=int(full_data_length),
                    final_size=int(final_size),
                    enc_strds=tuple(int(v) for v in enc_strds),
                    dec_strds=tuple(int(v) for v in dec_strds),
                    ks=_int_tuple_or_default(candidate.get("ks"), (0, 1, 5)),
                    enc_dim=_float_tuple_or_default(
                        candidate.get("enc_dim"),
                        (64.0, 16.0),
                    ),
                    emb_size=int(candidate.get("emb_size", candidate.get("snerv_emb_size", 0))),
                    reduce=float(candidate.get("reduce", 1.2)),
                    lower_width=int(candidate.get("lower_width", 12)),
                    saturate_stages=int(candidate.get("saturate_stages", -1)),
                ).fc_dim
            )
    return 9


def _int_tuple_or_default(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(int(v) for v in value.split(",") if v.strip())
    return tuple(int(v) for v in value)


def _float_tuple_or_default(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(float(v) for v in value.split(",") if v.strip())
    return tuple(float(v) for v in value)


def _packet_bytes_from_artifact(model_or_artifact: Any) -> bytes:
    if isinstance(model_or_artifact, bytes):
        return bytes(model_or_artifact)
    if isinstance(model_or_artifact, (str, Path)):
        return Path(model_or_artifact).read_bytes()
    mapping = _artifact_mapping(model_or_artifact)
    if isinstance(mapping.get("packet"), bytes):
        return bytes(mapping["packet"])
    for key in ("packet_path", "receiver_archive_packet_path"):
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return Path(value).expanduser().read_bytes()
    raise SnervMlxNativeExportError(
        "model_or_artifact must be packet bytes, a packet path, or a mapping "
        "with packet_path/receiver_archive_packet_path"
    )


def _artifact_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "as_jsonable") and callable(value.as_jsonable):
        return dict(value.as_jsonable())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return [str(value) for value in dict.fromkeys(str(v) for v in values if str(v))]


def _repo_root(repo_root: str | Path | None) -> Path:
    return (
        Path(repo_root).expanduser().resolve(strict=False)
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "FALSE_AUTHORITY",
    "SCORER_HW",
    "SNERV_MLX_NATIVE_PACKET_FILENAME",
    "SNERV_MLX_NATIVE_PREFILTER_PROFILE_SCHEMA",
    "SNERV_MLX_NATIVE_REPORT_FILENAME",
    "SNERV_MLX_NATIVE_STORAGE_PREFLIGHT_SCHEMA",
    "SNERV_MLX_NATIVE_TRAIN_EXPORT_SCHEMA",
    "SnervMlxNativeArtifact",
    "SnervMlxNativeExportError",
    "build_snerv_mlx_native_packet_from_numpy_pairs",
    "build_snerv_mlx_native_storage_preflight",
    "export_snerv_mlx_archive",
    "train_export_snerv_mlx_native",
    "write_snerv_mlx_prefilter_profile",
    "write_snerv_mlx_receiver_proof",
]
