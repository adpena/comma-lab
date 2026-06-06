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
import math
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.adaptation.hard_pair_indices import normalize_pair_indices
from tac.analysis.nerv_candidate_curriculum import (
    strip_candidate_curriculum_authority_fields,
)
from tac.analysis.snerv_official_primitive_replay import (
    build_snerv_official_receiver_runtime_decode_contract,
)
from tac.analysis.snerv_official_source_forward_harness import (
    build_snerv_official_trained_checkpoint_mapping_manifest,
)
from tac.analysis.snerv_official_tub_source_forward_replay import (
    TUB_PRESERVED_BLOCKERS,
    build_snerv_official_tub_source_forward_replay_artifact,
)
from tac.analysis.snerv_step_map_coder import encode_step_maps_waterfill
from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES
from tac.contest_eval_contract import build_upstream_eval_contract
from tac.optimization.archive_bound_candidate_runtime_bridge import (
    run_generated_inflate_receiver_proof,
)
from tac.repo_io import sha256_file, write_bytes_artifact, write_json
from tac.substrates._shared.mlx_score_aware.bridge_drift import (
    build_mlx_numpy_bridge_drift_bundle,
    mlx_numpy_bridge_drift_report,
)
from tac.substrates._shared.mlx_score_aware.curriculum import (
    build_scoreaware_curriculum_stages,
    coerce_scoreaware_stage_loss_weights,
)
from tac.substrates._shared.mlx_score_aware.dual_ascent import safe_dual_metric_key
from tac.substrates._shared.mlx_score_aware.targets import decode_mlx_targets
from tac.substrates.snerv_inverse_steg_carrier.allocation import (
    LfSaliency,
    allocate_lf_linf,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA,
    OFFICIAL_MFU_HFR_TUB_REQUIRED_TENSOR_KEYS,
    SECTION_ORDER,
    SNERV_ARCHIVE_SCHEMA,
    SNERV_ARCHIVE_SCHEMA_V2,
    SnervArchivePacket,
    decode_official_mfu_hfr_tub_decoder_payload,
    decode_snerv_archive_frames,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    encode_official_mfu_hfr_tub_decoder_payload,
    execute_official_mfu_hfr_tub_decoder_payload,
    inspect_decoder_payload_header,
    inspect_lf_quant_payload_header,
    pack_snerv_archive,
    pack_snerv_archive_snar2,
    resolve_decoder_payload_codec,
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
    SnervCarrierError,
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
    dwt2_multilevel,
    dwt2_native_synthesis_adjoint,
    idwt2_multilevel,
)
from tac.substrates.snerv_inverse_steg_carrier.inflate import _resize_nchw_bilinear
from tac.substrates.snerv_inverse_steg_carrier.lf_payload_codec import (
    selected_lf_payload_codec_label,
)
from tac.substrates.snerv_inverse_steg_carrier.mlx_native_adapter_contract import (
    build_snerv_mlx_native_training_export_guard,
)
from tac.substrates.snerv_inverse_steg_carrier.official_hfr import (
    OFFICIAL_SNERV_HFR_SOURCE_CONTRACT,
    OFFICIAL_SNERV_HFR_SOURCE_SHA,
    SNERV_OFFICIAL_HFR_CONVBLOCK_NUMPY_PROOF,
    OfficialConv2dNchw,
    OfficialHfrConvBlock,
    OfficialHfrHeads,
)
from tac.substrates.snerv_inverse_steg_carrier.official_mfu import (
    OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS,
    OFFICIAL_SNERV_MFU_SOURCE,
    OFFICIAL_SNERV_RB_SOURCE,
    OfficialConvTranspose2dNchw,
    OfficialResidualBlocksWithInputConv,
    OfficialSnervMfu,
    OfficialSnervMfuSpec,
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
SNERV_CONTEST_RATE_SCORE_PER_BYTE = 25.0 / float(ORIGINAL_VIDEO_BYTES)

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
SNERV_OFFICIAL_TRAINED_CHECKPOINT_STATE_DICT_MAPPING_BLOCKER = (
    "snerv_official_trained_checkpoint_state_dict_mapping_missing"
)
SNERV_OFFICIAL_MFU_HFR_TUB_WEIGHT_MAPPING_BLOCKER = (
    "snerv_official_mfu_hfr_tub_weight_mapping_missing"
)
SNERV_OFFICIAL_TRAINED_CHECKPOINT_SOURCE_FORWARD_BLOCKER = (
    "snerv_official_trained_checkpoint_source_forward_replay_missing"
)
SNERV_OFFICIAL_TUB_BATCHED_TEMPORAL_CONTEXT_SOURCE_BLOCKER = (
    "snerv_official_tub_batched_temporal_context_source_forward_replay_missing"
)
SNERV_OFFICIAL_TUB_SOURCE_FIXTURE_REPLAY_MISSING_BLOCKER = (
    "snerv_official_tub_source_fixture_replay_missing"
)
SNERV_OFFICIAL_PACKET_SOURCE_PARITY_BLOCKERS = (
    "snerv_official_bootstrap_stores_haar_ll_as_mfu_skip_high",
    "snerv_official_encoder_mfu_skip_hierarchy_source_forward_replay_missing",
    SNERV_OFFICIAL_TUB_BATCHED_TEMPORAL_CONTEXT_SOURCE_BLOCKER,
)
SNERV_SCORE_AWARE_CHECKPOINT_SELECTION_SCHEMA = (
    "snerv_score_aware_checkpoint_selection_policy.v1"
)
SNERV_ARCHIVE_SECTION_QAT_WEIGHT_POLICY_SCHEMA = (
    "snerv_archive_section_qat_weight_policy.v1"
)
SNERV_RECEIVER_FRAME_RECONSTRUCTION_PROFILE_SCHEMA = (
    "snerv_receiver_frame_reconstruction_profile.v1"
)
NATIVE_MLX_DECODER_LOSS_WORSEN_REL_TOL = 1.0e-7
SNERV_OFFICIAL_LONG_TRAINING_REPLAY_MAX_PAIRS = 16
SNERV_OFFICIAL_HFR_BOOTSTRAP_LS_MAX_ROWS = 262_144
SNERV_SCORE_AWARE_CHECKPOINT_RETENTION_KEEP_LAST_N_DEFAULT = 4
DEFAULT_SNERV_SCORER_INPUT_DISTRIBUTION_GUARD_WEIGHT = 2.0
SNERV_RECEIVER_RECON_MIN_STD_RATIO = 0.05
SNERV_RECEIVER_RECON_MAX_STD_RATIO = 20.0
SNERV_RECEIVER_RECON_MAX_SATURATION_DELTA = 0.35
SNERV_RECEIVER_RECON_MAX_RMSE_NCHW255 = 96.0
SNERV_RECEIVER_RECON_MAX_MAE_NCHW255 = 64.0
SNERV_LIVE_SECTION_BYTE_OFFICIAL_COMPONENTS_MISSING_BLOCKER = (
    "snerv_live_section_byte_official_components_current_state_missing"
)
SNERV_LIVE_SECTION_BYTE_OFFICIAL_FALLBACK_BLOCKER = (
    "snerv_live_section_byte_official_metrics_fallback_used"
)
SNERV_LIVE_SECTION_BYTE_OFFICIAL_NEVER_REFRESHED_BLOCKER = (
    "snerv_live_section_byte_official_metrics_never_refreshed_current_components"
)
OFFICIAL_TUB_OUTPUT2_PAYLOAD_TENSOR_NAMES = (
    "tub.temporal_encoder_concat",
    "tub.output2_raw",
)
OFFICIAL_TUB_OUTPUT2_RECEIVER_OUTPUT_TENSOR_NAMES = (
    "tub.output2_decoder_input",
    "tub.output2_fused",
)
OFFICIAL_TUB_OUTPUT2_BINDING_SIGNATURE_FIELDS = (
    "stored",
    "source_payload_present",
    "proof_only_elided_from_selected_runtime_packet",
    "proof_only_false_authority_metadata",
    "receiver_executes_output2_fusion_from_payload",
    "receiver_frame_decode_consumes_output2",
    "receiver_frame_decode_binding_status",
    "receiver_output2_frame_shape_match",
    "train_time_loss_coupled",
    "scored_pixel_render_bound",
    "source_raw_bytes",
    "stored_raw_bytes",
    "temporal_encoder_output_shape",
    "output2_decoder_output_shape",
    "fc_hw",
    "payload_tensor_names",
)


class SnervMlxNativeExportError(ValueError):
    """Raised when the native MLX SNeRV adapter cannot build a valid export."""


def _resolve_torch_scorer_device_alias(
    requested_device: str,
    *,
    torch_module: Any | None = None,
) -> str:
    """Resolve direct SNeRV scorer-teacher device aliases to PyTorch devices."""

    requested = str(requested_device or "cpu").strip().lower()
    if requested in {"cpu", "cuda", "mps"}:
        return requested
    if requested == "metal":
        return "mps"
    if requested != "gpu":
        raise SnervMlxNativeExportError(
            f"unsupported scorer distillation device: {requested_device!r}"
        )
    torch = torch_module
    if torch is None:
        try:
            import torch as torch  # type: ignore[no-redef]
        except Exception as exc:  # pragma: no cover - import failure is environment.
            raise SnervMlxNativeExportError(
                "distillation_device='gpu' requires PyTorch to resolve a concrete "
                "scorer teacher device"
            ) from exc
    cuda = getattr(torch, "cuda", None)
    if bool(getattr(cuda, "is_available", lambda: False)()):
        return "cuda"
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if bool(getattr(mps, "is_available", lambda: False)()):
        return "mps"
    raise SnervMlxNativeExportError(
        "distillation_device='gpu' requested, but neither torch.cuda nor "
        "torch.backends.mps is available"
    )


def _candidate_first_non_null(
    candidate: Mapping[str, Any],
    keys: Sequence[str],
    fallback: Any,
) -> Any:
    """Return the first explicit non-null candidate override.

    Candidate rows often carry JSON ``null`` for fields they do not own.  For
    launch controls such as checkpoint retention, treating that null as an
    override silently disables the caller's safe default.  Null therefore means
    "no override"; an intentional preserve-all retention run uses ``-1``.
    """

    for key in keys:
        if key in candidate and candidate[key] is not None:
            return candidate[key]
    return fallback


def _coerce_state_dict_value_to_numpy(value: Any, *, key: str) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    try:
        return np.asarray(value)
    except Exception as exc:
        raise SnervMlxNativeExportError(
            f"official trained checkpoint state value {key!r} is not array-like"
        ) from exc


def _coerce_state_dict_to_numpy_mapping(raw: Any) -> dict[str, np.ndarray]:
    if isinstance(raw, Mapping):
        for nested_key in ("state_dict", "model_state_dict", "decoder_state_dict"):
            nested = raw.get(nested_key)
            if isinstance(nested, Mapping):
                raw = nested
                break
    if not isinstance(raw, Mapping):
        raise SnervMlxNativeExportError(
            "official trained checkpoint state_dict must be a mapping"
        )
    out: dict[str, np.ndarray] = {}
    for raw_key, raw_value in raw.items():
        key = str(raw_key)
        if not key:
            raise SnervMlxNativeExportError(
                "official trained checkpoint state_dict keys must be non-empty"
            )
        out[key] = _coerce_state_dict_value_to_numpy(raw_value, key=key)
    return out


def _load_official_trained_checkpoint_state_dict_path(
    path: str | Path,
) -> dict[str, np.ndarray]:
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise SnervMlxNativeExportError(
            "official trained checkpoint state_dict path does not exist: "
            f"{resolved.as_posix()}"
        )
    suffix = resolved.suffix.lower()
    if suffix == ".npz":
        with np.load(resolved, allow_pickle=False) as npz:
            return {str(key): np.asarray(npz[key]) for key in npz.files}
    if suffix == ".json":
        return _coerce_state_dict_to_numpy_mapping(
            json.loads(resolved.read_text(encoding="utf-8"))
        )
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is optional locally.
        raise SnervMlxNativeExportError(
            "loading non-NPZ/JSON official trained checkpoints requires torch"
        ) from exc
    loaded = torch.load(resolved, map_location="cpu")
    return _coerce_state_dict_to_numpy_mapping(loaded)


def _official_trained_checkpoint_mapping_manifest_from_inputs(
    *,
    state_dict: Mapping[str, Any] | None,
    state_dict_path: str | Path | None,
    decoder_len: int | None,
    state_dict_kind: str,
) -> dict[str, Any]:
    if state_dict is not None and state_dict_path is not None:
        raise SnervMlxNativeExportError(
            "provide only one of official_trained_checkpoint_state_dict and "
            "official_trained_checkpoint_state_dict_path"
        )
    if state_dict_path is not None:
        resolved = Path(state_dict_path).expanduser().resolve(strict=False)
        normalized = _load_official_trained_checkpoint_state_dict_path(resolved)
        source = resolved.as_posix()
    elif state_dict is not None:
        normalized = _coerce_state_dict_to_numpy_mapping(state_dict)
        source = "in_memory_official_trained_checkpoint_state_dict"
    else:
        normalized = None
        source = "snerv_mlx_native_train_export"
    return build_snerv_official_trained_checkpoint_mapping_manifest(
        normalized,
        decoder_len=decoder_len,
        state_dict_kind=state_dict_kind
        if normalized is not None
        else "missing_upstream_official_checkpoint_for_mlx_long_training",
        source=source,
    )


def _coerce_official_checkpoint_mapping_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    source: str,
) -> dict[str, Any]:
    if isinstance(manifest, Mapping) and manifest.get(
        "schema"
    ) == "snerv_official_trained_checkpoint_state_dict_mapping_manifest.v1":
        return dict(manifest)
    return build_snerv_official_trained_checkpoint_mapping_manifest(
        None,
        state_dict_kind="missing_upstream_official_checkpoint_for_mlx_long_training",
        source=source,
    )


def _official_checkpoint_component_mapping_verified(
    manifest: Mapping[str, Any],
    component_id: str,
) -> bool:
    for row in manifest.get("component_rows") or ():
        if (
            isinstance(row, Mapping)
            and str(row.get("component_id") or "") == str(component_id)
        ):
            return bool(row.get("trained_checkpoint_weight_mapping_proven") is True)
    return False


def _official_checkpoint_full_mapping_verified(
    manifest: Mapping[str, Any],
) -> bool:
    return bool(
        manifest.get("official_mfu_hfr_trained_checkpoint_weight_mapping_proven")
        is True
        and manifest.get("official_tub_temporal_encoder_weight_mapping_proven")
        is True
    )


def _coerce_gradient_multiplier_by_name(value: Any) -> dict[str, float]:
    """Normalize exact parameter-name gradient multipliers for shared MLX runs."""

    if value is None:
        return {}
    items: Iterable[tuple[Any, Any]]
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parsed: list[tuple[Any, Any]] = []
        for row in value:
            if isinstance(row, Mapping):
                if "name" in row and "value" in row:
                    parsed.append((row["name"], row["value"]))
                elif "parameter" in row and "multiplier" in row:
                    parsed.append((row["parameter"], row["multiplier"]))
                else:
                    raise SnervMlxNativeExportError(
                        "gradient multiplier rows must contain name/value or "
                        "parameter/multiplier"
                    )
            elif isinstance(row, Sequence) and not isinstance(
                row, (str, bytes, bytearray)
            ):
                pair = list(row)
                if len(pair) != 2:
                    raise SnervMlxNativeExportError(
                        "gradient multiplier sequence rows must be pairs"
                    )
                parsed.append((pair[0], pair[1]))
            else:
                raise SnervMlxNativeExportError(
                    "gradient multipliers must be a mapping or sequence of pairs"
                )
        items = parsed
    else:
        raise SnervMlxNativeExportError(
            "gradient multipliers must be a mapping or sequence of pairs"
        )
    out: dict[str, float] = {}
    for raw_name, raw_multiplier in items:
        name = str(raw_name)
        if not name:
            raise SnervMlxNativeExportError(
                "gradient multiplier parameter names must be non-empty"
            )
        try:
            multiplier = float(raw_multiplier)
        except (TypeError, ValueError) as exc:
            raise SnervMlxNativeExportError(
                f"gradient multiplier {name!r} must be a finite float"
            ) from exc
        if not math.isfinite(multiplier) or multiplier < 0.0:
            raise SnervMlxNativeExportError(
                f"gradient multiplier {name!r} must be finite and >= 0"
            )
        out[name] = multiplier
    return out


def _snerv_score_aware_checkpoint_selection_policy(
    *,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    pose_direct_live_distillation_weight: float = 0.0,
    segnet_direct_live_distillation_weight: float = 0.0,
    segnet_direct_live_class_histogram_weight: float = 0.0,
    segnet_direct_live_class_balanced_hinge_weight: float = 0.0,
    segnet_direct_live_class_balanced_ce_weight: float = 0.0,
    segnet_direct_live_class_balanced_squared_hinge_weight: float = 0.0,
    segnet_direct_live_class_region_recon_weight: float = 0.0,
    segnet_direct_live_rare_class_logit_weight: float = 0.0,
    segnet_direct_live_target_mass_floor_weight: float = 0.0,
    segnet_direct_live_target_min_ratio_floor_weight: float = 0.0,
    scorer_input_distribution_guard_weight: float = 0.0,
    scorer_input_contrast_floor_weight: float = 0.0,
    scorer_input_shape_tether_weight: float = 0.0,
    posenet_yuv6_geometry_tether_weight: float = 0.0,
    posenet_temporal_signal_floor_weight: float = 0.0,
    has_real_segnet_teacher: bool,
    has_real_posenet_teacher: bool,
    coder_aware_qat_bound: bool,
    coder_qat_loss_weight_map: Mapping[str, float] | None,
    pr95_faithful_curriculum_enabled: bool,
) -> dict[str, Any]:
    """Choose the SNeRV long-training checkpoint-selection metric.

    Raw reconstruction MSE is a safe fallback only for runs that have no scorer
    teachers, no coder-aware QAT pressure, and no PR95 scorer curriculum. Once
    any of those score-aware surfaces is active, checkpoint selection must use
    the same composite loss family that training optimizes; otherwise a long
    run can throw away a state that improves the contest objective because its
    human/reconstruction MSE is slightly worse.
    """

    seg_weight = float(segnet_distillation_weight)
    direct_live_weight = float(segnet_direct_live_distillation_weight)
    direct_live_subcontrol_weights = {
        "class_histogram": float(segnet_direct_live_class_histogram_weight),
        "class_balanced_hinge": float(segnet_direct_live_class_balanced_hinge_weight),
        "class_balanced_ce": float(segnet_direct_live_class_balanced_ce_weight),
        "class_balanced_squared_hinge": float(
            segnet_direct_live_class_balanced_squared_hinge_weight
        ),
        "class_region_recon": float(segnet_direct_live_class_region_recon_weight),
        "rare_class_logit": float(segnet_direct_live_rare_class_logit_weight),
        "target_mass_floor": float(segnet_direct_live_target_mass_floor_weight),
        "target_min_ratio_floor": float(
            segnet_direct_live_target_min_ratio_floor_weight
        ),
    }
    active_direct_live_subcontrols = {
        name: weight
        for name, weight in direct_live_subcontrol_weights.items()
        if weight > 0.0
    }
    pose_weight = float(pose_distillation_weight)
    pose_direct_live_weight = float(pose_direct_live_distillation_weight)
    guard_weight = float(scorer_input_distribution_guard_weight)
    contrast_floor_weight = float(scorer_input_contrast_floor_weight)
    shape_tether_weight = float(scorer_input_shape_tether_weight)
    geometry_tether_weight = float(posenet_yuv6_geometry_tether_weight)
    temporal_floor_weight = float(posenet_temporal_signal_floor_weight)
    weighted_qat_terms = {
        str(name): float(weight)
        for name, weight in dict(coder_qat_loss_weight_map or {}).items()
        if float(weight) != 0.0
    }
    active_surfaces: list[str] = []
    required_loss_parts = ["recon"]
    blockers: list[str] = []
    if seg_weight > 0.0:
        active_surfaces.append("real_segnet_teacher_distillation")
        required_loss_parts.append("distill")
        if not has_real_segnet_teacher:
            blockers.append(
                "snerv_score_aware_checkpoint_selection_segnet_teacher_missing"
            )
    if direct_live_weight > 0.0:
        active_surfaces.append("real_segnet_direct_live_distillation")
        required_loss_parts.append("segnet_direct_live_distill")
        if not has_real_segnet_teacher:
            blockers.append(
                "snerv_score_aware_checkpoint_selection_segnet_teacher_missing"
            )
    if active_direct_live_subcontrols:
        active_surfaces.append("real_segnet_direct_live_subcontrols")
        required_loss_parts.append("segnet_direct_live_distill")
        if active_direct_live_subcontrols.get("class_region_recon", 0.0) > 0.0:
            required_loss_parts.append(
                "segnet_direct_live_class_region_recon_loss"
            )
        if active_direct_live_subcontrols.get("rare_class_logit", 0.0) > 0.0:
            required_loss_parts.append(
                "segnet_direct_live_rare_class_logit_loss"
            )
        if active_direct_live_subcontrols.get("target_mass_floor", 0.0) > 0.0:
            required_loss_parts.append(
                "segnet_direct_live_target_mass_floor_loss"
            )
        if active_direct_live_subcontrols.get("target_min_ratio_floor", 0.0) > 0.0:
            required_loss_parts.append(
                "segnet_direct_live_target_min_ratio_floor_loss"
            )
        if not has_real_segnet_teacher:
            blockers.append(
                "snerv_score_aware_checkpoint_selection_segnet_teacher_missing"
            )
    if pose_weight > 0.0:
        active_surfaces.append("real_posenet_teacher_distillation")
        required_loss_parts.append("pose_score_term")
        if not has_real_posenet_teacher:
            blockers.append(
                "snerv_score_aware_checkpoint_selection_posenet_teacher_missing"
            )
    if pose_direct_live_weight > 0.0:
        active_surfaces.append("real_posenet_direct_live_distillation")
        required_loss_parts.append("pose_direct_live_score_term")
        if not has_real_posenet_teacher:
            blockers.append(
                "snerv_score_aware_checkpoint_selection_posenet_teacher_missing"
            )
    if guard_weight > 0.0:
        active_surfaces.append("scorer_input_distribution_guard")
        required_loss_parts.append("scorer_input_distribution_guard")
    if contrast_floor_weight > 0.0:
        active_surfaces.append("scorer_input_contrast_floor")
        required_loss_parts.append("scorer_input_contrast_floor")
    if shape_tether_weight > 0.0:
        active_surfaces.append("scorer_input_shape_tether")
        required_loss_parts.append("scorer_input_shape_tether")
    if geometry_tether_weight > 0.0:
        active_surfaces.append("posenet_yuv6_geometry_tether")
        required_loss_parts.append("posenet_yuv6_geometry_tether")
    if temporal_floor_weight > 0.0:
        active_surfaces.append("posenet_temporal_signal_floor")
        required_loss_parts.append("posenet_temporal_signal_floor")
    if bool(coder_aware_qat_bound):
        active_surfaces.append("coder_aware_qat")
        if not weighted_qat_terms:
            blockers.append(
                "snerv_score_aware_checkpoint_selection_coder_qat_terms_missing"
            )
        else:
            required_loss_parts.extend(sorted(weighted_qat_terms))
    if bool(pr95_faithful_curriculum_enabled):
        active_surfaces.append("pr95_faithful_curriculum")
        required_loss_parts.append("pr95_stage_scorer_surrogate")

    uses_score_aware = bool(active_surfaces)
    return {
        "schema": SNERV_SCORE_AWARE_CHECKPOINT_SELECTION_SCHEMA,
        "selection_metric": (
            "score_aware_composite_full_video_surrogate"
            if uses_score_aware
            else "full_reconstruction_mse_nchw255"
        ),
        "selection_metric_value_key": (
            "score_aware_composite_loss"
            if uses_score_aware
            else "recon_mse_nchw255"
        ),
        "uses_score_aware_composite": uses_score_aware,
        "mse_fallback": not uses_score_aware,
        "mse_fallback_reason": (
            "no_scorer_teacher_no_coder_qat_no_pr95_curriculum"
            if not uses_score_aware
            else None
        ),
        "active_score_surfaces": active_surfaces,
        "required_loss_parts": _ordered_unique(required_loss_parts),
        "segnet_direct_live_distillation_weight": direct_live_weight,
        "segnet_direct_live_subcontrol_weights": active_direct_live_subcontrols,
        "pose_direct_live_distillation_weight": pose_direct_live_weight,
        "pose_selection_loss_part": (
            "pose_direct_live_score_term"
            if pose_direct_live_weight > 0.0
            else ("pose_score_term" if pose_weight > 0.0 else None)
        ),
        "scorer_input_distribution_guard_weight": guard_weight,
        "scorer_input_contrast_floor_weight": contrast_floor_weight,
        "scorer_input_shape_tether_weight": shape_tether_weight,
        "posenet_yuv6_geometry_tether_weight": geometry_tether_weight,
        "posenet_temporal_signal_floor_weight": temporal_floor_weight,
        "weighted_coder_qat_terms": weighted_qat_terms,
        "fail_closed_on_missing_parts": uses_score_aware,
        "full_reduction": (
            "deterministic_pair_chunks_no_update_before_reduction"
            if uses_score_aware
            else "full_reconstruction_mse_full_pairs"
        ),
        "blockers": _ordered_unique(blockers),
        "human_visual_fidelity_objective": False,
        **FALSE_AUTHORITY,
    }


def _snerv_checkpoint_selection_row_is_better(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    *,
    metric_value_key: str,
    epsilon: float = 1.0e-9,
) -> bool:
    candidate_support = _snerv_checkpoint_selection_support_tuple(
        candidate,
        metric_value_key=metric_value_key,
    )
    incumbent_support = _snerv_checkpoint_selection_support_tuple(
        incumbent,
        metric_value_key=metric_value_key,
    )
    if candidate_support is not None or incumbent_support is not None:
        if candidate_support is None:
            if incumbent_support is not None and incumbent_support[0] <= 0.0:
                return _snerv_checkpoint_selection_scalar_row_is_better(
                    candidate,
                    incumbent,
                    metric_value_key=metric_value_key,
                    epsilon=epsilon,
                )
            return False
        if incumbent_support is None:
            return candidate_support[0] > 0.0
        for candidate_value, incumbent_value in zip(
            candidate_support,
            incumbent_support,
            strict=True,
        ):
            if candidate_value > incumbent_value + float(epsilon):
                return True
            if candidate_value < incumbent_value - float(epsilon):
                return False
        return False

    return _snerv_checkpoint_selection_scalar_row_is_better(
        candidate,
        incumbent,
        metric_value_key=metric_value_key,
        epsilon=epsilon,
    )


def _snerv_checkpoint_selection_scalar_row_is_better(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    *,
    metric_value_key: str,
    epsilon: float = 1.0e-9,
) -> bool:
    try:
        candidate_value = float(candidate[metric_value_key])
    except (KeyError, TypeError, ValueError):
        return False
    try:
        incumbent_value = float(incumbent[metric_value_key])
    except (KeyError, TypeError, ValueError):
        incumbent_value = float("inf")
    return bool(
        np.isfinite(candidate_value)
        and (
            not np.isfinite(incumbent_value)
            or candidate_value < incumbent_value - float(epsilon)
        )
    )


_SNERV_SELECTION_SUPPORT_ALIASES: dict[str, tuple[str, ...]] = {
    "segnet_direct_live_candidate_occupied_class_fraction": (
        "segnet_direct_live_candidate_occupied_class_fraction",
        "loss_part_segnet_direct_live_candidate_occupied_class_fraction",
        "loss_part_pr95_stage_segnet_direct_live_candidate_occupied_class_fraction",
    ),
    "segnet_direct_live_candidate_target_class_coverage_fraction": (
        "segnet_direct_live_candidate_target_class_coverage_fraction",
        "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction",
        "loss_part_pr95_stage_segnet_direct_live_candidate_target_class_coverage_fraction",
    ),
    "segnet_direct_live_candidate_target_class_min_ratio": (
        "segnet_direct_live_candidate_target_class_min_ratio",
        "loss_part_segnet_direct_live_candidate_target_class_min_ratio",
        "loss_part_pr95_stage_segnet_direct_live_candidate_target_class_min_ratio",
    ),
    "segnet_direct_live_argmax_disagreement": (
        "segnet_direct_live_argmax_disagreement",
        "loss_part_segnet_direct_live_argmax_disagreement",
        "loss_part_pr95_stage_segnet_direct_live_argmax_disagreement",
    ),
}


def _snerv_checkpoint_selection_attach_support_metrics(row: dict[str, Any]) -> None:
    for output_key, aliases in _SNERV_SELECTION_SUPPORT_ALIASES.items():
        value = _snerv_checkpoint_selection_support_value(row, aliases)
        if value is not None:
            row[output_key] = float(value)
    row["support_aware_selection_axis"] = (
        "segnet_last_frame_target_class_support_then_scalar_surrogate"
        if any(
            output_key in row
            for output_key in _SNERV_SELECTION_SUPPORT_ALIASES
        )
        else "scalar_surrogate_only"
    )


def _snerv_checkpoint_selection_support_tuple(
    row: Mapping[str, Any],
    *,
    metric_value_key: str,
) -> tuple[float, ...] | None:
    coverage = _snerv_checkpoint_selection_support_value(
        row,
        _SNERV_SELECTION_SUPPORT_ALIASES[
            "segnet_direct_live_candidate_target_class_coverage_fraction"
        ],
    )
    min_ratio = _snerv_checkpoint_selection_support_value(
        row,
        _SNERV_SELECTION_SUPPORT_ALIASES[
            "segnet_direct_live_candidate_target_class_min_ratio"
        ],
    )
    occupied = _snerv_checkpoint_selection_support_value(
        row,
        _SNERV_SELECTION_SUPPORT_ALIASES[
            "segnet_direct_live_candidate_occupied_class_fraction"
        ],
    )
    argmax = _snerv_checkpoint_selection_support_value(
        row,
        _SNERV_SELECTION_SUPPORT_ALIASES["segnet_direct_live_argmax_disagreement"],
    )
    if not any(
        value is not None
        for value in (coverage, min_ratio, occupied, argmax)
    ):
        return None
    metric = _snerv_checkpoint_selection_metric_value(row, metric_value_key)
    return (
        1.0 if _snerv_checkpoint_selection_blocker_free(row) else 0.0,
        -1.0 if coverage is None else float(coverage),
        -1.0 if min_ratio is None else float(min_ratio),
        -1.0 if occupied is None else float(occupied),
        float("-inf") if argmax is None else -float(argmax),
        float("-inf") if metric is None else -float(metric),
    )


def _snerv_checkpoint_selection_blocker_free(row: Mapping[str, Any]) -> bool:
    return not any(
        str(blocker)
        for blocker in row.get("score_aware_checkpoint_selection_blockers", ())
    )


def _snerv_checkpoint_selection_metric_value(
    row: Mapping[str, Any],
    metric_value_key: str,
) -> float | None:
    try:
        value = float(row[metric_value_key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _snerv_checkpoint_selection_support_value(
    row: Mapping[str, Any],
    aliases: Sequence[str],
) -> float | None:
    search_rows: list[Mapping[str, Any]] = [row]
    parts = row.get("score_aware_composite_parts")
    if isinstance(parts, Mapping):
        search_rows.append(parts)
    for mapping in search_rows:
        for alias in aliases:
            for key in (alias, f"raw_{alias}", f"weighted_{alias}"):
                value = mapping.get(key)
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(scalar):
                    return scalar
    return None


def _snerv_score_aware_long_training_telemetry_contract(
    telemetry_path: str | Path,
    *,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    segnet_student_live_calibration_weight: float,
    segnet_direct_live_distillation_weight: float = 0.0,
    pose_direct_live_distillation_weight: float = 0.0,
    segnet_direct_live_class_histogram_weight: float = 0.0,
    segnet_direct_live_class_balanced_hinge_weight: float = 0.0,
    segnet_direct_live_class_balanced_ce_weight: float = 0.0,
    segnet_direct_live_class_balanced_squared_hinge_weight: float = 0.0,
    segnet_direct_live_class_region_recon_weight: float = 0.0,
    segnet_direct_live_rare_class_logit_weight: float = 0.0,
    segnet_direct_live_target_mass_floor_weight: float = 0.0,
    segnet_direct_live_target_min_ratio_floor_weight: float = 0.0,
    pr95_faithful_curriculum_enabled: bool,
    coder_aware_qat_bound: bool,
    train_time_section_byte_control_bound: bool,
    scorer_input_distribution_guard_weight: float,
    scorer_input_contrast_floor_weight: float = 0.0,
    scorer_input_shape_tether_weight: float = 0.0,
    posenet_yuv6_geometry_tether_weight: float = 0.0,
    posenet_temporal_signal_floor_weight: float = 0.0,
    gradient_multiplier_controls_requested: bool = False,
    scorer_space_step_guard_enabled: bool = False,
) -> dict[str, Any]:
    """Validate that a SNeRV long run actually drove score-aware controls."""

    path = Path(telemetry_path).expanduser()
    blockers: list[str] = []
    expected_seg = float(segnet_distillation_weight) > 0.0
    expected_pose = float(pose_distillation_weight) > 0.0
    expected_pose_direct_live = float(pose_direct_live_distillation_weight) > 0.0
    expected_live_calibration = bool(
        expected_seg and float(segnet_student_live_calibration_weight) > 0.0
    )
    expected_direct_live_region_recon = (
        float(segnet_direct_live_class_region_recon_weight) > 0.0
    )
    expected_direct_live_rare_class_logit = (
        float(segnet_direct_live_rare_class_logit_weight) > 0.0
    )
    expected_direct_live_target_mass_floor = (
        float(segnet_direct_live_target_mass_floor_weight) > 0.0
    )
    expected_direct_live_target_min_ratio_floor = (
        float(segnet_direct_live_target_min_ratio_floor_weight) > 0.0
    )
    expected_direct_live_subcontrol = any(
        float(value) > 0.0
        for value in (
            segnet_direct_live_class_histogram_weight,
            segnet_direct_live_class_balanced_hinge_weight,
            segnet_direct_live_class_balanced_ce_weight,
            segnet_direct_live_class_balanced_squared_hinge_weight,
            segnet_direct_live_class_region_recon_weight,
            segnet_direct_live_rare_class_logit_weight,
            segnet_direct_live_target_mass_floor_weight,
            segnet_direct_live_target_min_ratio_floor_weight,
        )
    )
    expected_direct_live = (
        float(segnet_direct_live_distillation_weight) > 0.0
        or expected_direct_live_subcontrol
    )
    expected_section = bool(coder_aware_qat_bound and train_time_section_byte_control_bound)
    expected_guard = float(scorer_input_distribution_guard_weight) > 0.0
    expected_contrast_floor = float(scorer_input_contrast_floor_weight) > 0.0
    expected_shape_tether = float(scorer_input_shape_tether_weight) > 0.0
    expected_geometry_tether = float(posenet_yuv6_geometry_tether_weight) > 0.0
    expected_temporal_floor = float(posenet_temporal_signal_floor_weight) > 0.0
    expected_gradient_multiplier = bool(gradient_multiplier_controls_requested)
    expected_scorer_space_step_guard = bool(scorer_space_step_guard_enabled)
    expected_any = bool(
        expected_seg
        or expected_pose
        or expected_pose_direct_live
        or expected_live_calibration
        or expected_direct_live
        or expected_section
        or expected_guard
        or expected_contrast_floor
        or expected_shape_tether
        or expected_geometry_tether
        or expected_temporal_floor
        or expected_gradient_multiplier
        or expected_scorer_space_step_guard
    )
    row_count = 0
    malformed_rows = 0
    seg_dual_observed = False
    pose_dual_observed = False
    guard_dual_observed = False
    seg_loss_observed = False
    pose_loss_observed = False
    pose_direct_live_loss_observed = False
    pose_direct_live_raw_mse_observed = False
    pose_direct_live_score_term_observed = False
    archive_rate_observed = False
    section_rate_observed = False
    guard_loss_observed = False
    contrast_floor_loss_observed = False
    contrast_floor_segnet_ratio_observed = False
    contrast_floor_posenet_ratio_observed = False
    shape_tether_loss_observed = False
    shape_tether_segnet_observed = False
    shape_tether_posenet_pair_observed = False
    shape_tether_posenet_delta_observed = False
    geometry_tether_loss_observed = False
    geometry_tether_pair_observed = False
    geometry_tether_delta_observed = False
    temporal_floor_loss_observed = False
    temporal_floor_std_ratio_observed = False
    temporal_floor_mean_abs_ratio_observed = False
    live_calibration_active_observed = False
    live_calibration_loss_observed = False
    direct_live_loss_observed = False
    direct_live_argmax_observed = False
    direct_live_class_occupancy_observed = False
    direct_live_class_region_recon_observed = False
    direct_live_rare_class_logit_observed = False
    direct_live_target_mass_floor_observed = False
    direct_live_target_min_ratio_floor_observed = False
    direct_live_target_mass_floor_dual_observed = False
    direct_live_target_mass_floor_dual_lambda_active_observed = False
    direct_live_target_min_ratio_floor_dual_observed = False
    direct_live_target_min_ratio_floor_dual_lambda_active_observed = False
    direct_live_max_candidate_occupied_class_fraction: float | None = None
    direct_live_target_class_coverage_observed = False
    direct_live_max_candidate_target_class_coverage_fraction: float | None = None
    pr95_seg_loss_observed = False
    pr95_pose_loss_observed = False
    seg_dual_lambda_active_observed = False
    pose_dual_lambda_active_observed = False
    archive_dual_lambda_active_observed = False
    archive_dual_positive_violation_observed = False
    archive_dual_update_observed = False
    section_dual_lambda_active_observed = False
    section_dual_positive_violation_observed = False
    section_dual_update_observed = False
    archive_dual_weight_applied_observed = False
    section_dual_weight_applied_observed = False
    section_dual_zero_base_masked_observed = False
    pr95_seg_effective_weight_seen = False
    pr95_seg_effective_weight_active = False
    pr95_pose_effective_weight_seen = False
    pr95_pose_effective_weight_active = False
    gradient_multiplier_requested_observed = False
    gradient_multiplier_applied_observed = False
    gradient_multiplier_missing_requested_observed = False
    gradient_multiplier_noop_observed = False
    scorer_space_step_guard_config_observed = False
    scorer_space_step_guard_metric_observed = False
    scorer_space_step_guard_intervention_observed = False
    if not path.is_file():
        blockers = (
            ["snerv_score_aware_long_training_telemetry_missing"]
            if expected_any
            else []
        )
        return {
            "schema": "snerv_score_aware_long_training_telemetry_contract.v1",
            "telemetry_path": path.as_posix(),
            "telemetry_exists": False,
            "row_count": 0,
            "passed": not blockers,
            "blockers": blockers,
        }
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
            if not isinstance(row, Mapping):
                malformed_rows += 1
                continue
            row_count += 1
            seg_loss_observed = seg_loss_observed or _finite_number_in_row(
                row,
                "loss_part_distill",
            )
            pose_loss_observed = pose_loss_observed or _finite_number_in_row(
                row,
                "loss_part_pose_distill",
            )
            pose_direct_live_loss_observed = (
                pose_direct_live_loss_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_pose_direct_live_distill",
                        "raw_pose_direct_live_distill",
                    )
                )
            )
            pose_direct_live_raw_mse_observed = (
                pose_direct_live_raw_mse_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_pose_direct_live_raw_mse",
                        "raw_pose_direct_live_raw_mse",
                    )
                )
            )
            pose_direct_live_score_term_observed = (
                pose_direct_live_score_term_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_pose_direct_live_score_term",
                        "raw_pose_direct_live_score_term",
                    )
                )
            )
            pr95_seg_loss_observed = pr95_seg_loss_observed or _finite_number_in_row(
                row,
                "loss_part_pr95_stage_seg_surrogate",
            )
            pr95_pose_loss_observed = pr95_pose_loss_observed or _finite_number_in_row(
                row,
                "loss_part_pr95_stage_pose_surrogate",
            )
            pr95_seg_effective_weight_seen = (
                pr95_seg_effective_weight_seen
                or _finite_number_in_row(row, "loss_part_pr95_stage_effective_seg_weight")
            )
            pr95_seg_effective_weight_active = (
                pr95_seg_effective_weight_active
                or _row_nonzero_finite_number(
                    row,
                    "loss_part_pr95_stage_effective_seg_weight",
                )
            )
            pr95_pose_effective_weight_seen = (
                pr95_pose_effective_weight_seen
                or _finite_number_in_row(row, "loss_part_pr95_stage_effective_pose_weight")
            )
            pr95_pose_effective_weight_active = (
                pr95_pose_effective_weight_active
                or _row_nonzero_finite_number(
                    row,
                    "loss_part_pr95_stage_effective_pose_weight",
                )
            )
            guard_loss_observed = guard_loss_observed or any(
                _finite_number_in_row(row, key)
                for key in (
                    "loss_part_scorer_input_distribution_guard",
                    "loss_part_pr95_stage_scorer_input_distribution_guard",
                )
            )
            contrast_floor_loss_observed = contrast_floor_loss_observed or any(
                _finite_number_in_row(row, key)
                for key in (
                    "loss_part_scorer_input_contrast_floor",
                    "loss_part_pr95_stage_scorer_input_contrast_floor",
                )
            )
            contrast_floor_segnet_ratio_observed = (
                contrast_floor_segnet_ratio_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio",
                        "loss_part_pr95_stage_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio",
                    )
                )
            )
            contrast_floor_posenet_ratio_observed = (
                contrast_floor_posenet_ratio_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio",
                        "loss_part_pr95_stage_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio",
                    )
                )
            )
            shape_tether_loss_observed = shape_tether_loss_observed or any(
                _finite_number_in_row(row, key)
                for key in (
                    "loss_part_scorer_input_shape_tether",
                    "loss_part_pr95_stage_scorer_input_shape_tether",
                )
            )
            shape_tether_segnet_observed = (
                shape_tether_segnet_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_scorer_input_shape_tether_segnet_last_rgb",
                        "loss_part_pr95_stage_scorer_input_shape_tether_segnet_last_rgb",
                    )
                )
            )
            shape_tether_posenet_pair_observed = (
                shape_tether_posenet_pair_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_scorer_input_shape_tether_posenet_yuv6_pair",
                        "loss_part_pr95_stage_scorer_input_shape_tether_posenet_yuv6_pair",
                    )
                )
            )
            shape_tether_posenet_delta_observed = (
                shape_tether_posenet_delta_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta",
                        "loss_part_pr95_stage_scorer_input_shape_tether_posenet_yuv6_temporal_delta",
                    )
                )
            )
            geometry_tether_loss_observed = geometry_tether_loss_observed or any(
                _finite_number_in_row(row, key)
                for key in (
                    "loss_part_posenet_yuv6_geometry_tether",
                    "loss_part_pr95_stage_posenet_yuv6_geometry_tether",
                )
            )
            geometry_tether_pair_observed = (
                geometry_tether_pair_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_posenet_yuv6_geometry_tether_pair",
                        "loss_part_pr95_stage_posenet_yuv6_geometry_tether_pair",
                    )
                )
            )
            geometry_tether_delta_observed = (
                geometry_tether_delta_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_posenet_yuv6_geometry_tether_temporal_delta",
                        "loss_part_pr95_stage_posenet_yuv6_geometry_tether_temporal_delta",
                    )
                )
            )
            temporal_floor_loss_observed = temporal_floor_loss_observed or any(
                _finite_number_in_row(row, key)
                for key in (
                    "loss_part_posenet_temporal_signal_floor",
                    "loss_part_pr95_stage_posenet_temporal_signal_floor",
                )
            )
            temporal_floor_std_ratio_observed = (
                temporal_floor_std_ratio_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_posenet_temporal_signal_floor_mean_std_ratio",
                        "loss_part_pr95_stage_posenet_temporal_signal_floor_mean_std_ratio",
                    )
                )
            )
            temporal_floor_mean_abs_ratio_observed = (
                temporal_floor_mean_abs_ratio_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_posenet_temporal_signal_floor_mean_abs_ratio",
                        "loss_part_pr95_stage_posenet_temporal_signal_floor_mean_abs_ratio",
                    )
                )
            )
            live_calibration_active_observed = (
                live_calibration_active_observed
                or _row_float_equals(
                    row,
                    "segnet_student_live_calibration_active",
                    1.0,
                )
            )
            live_calibration_loss_observed = (
                live_calibration_loss_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_segnet_student_live_calibration",
                        "loss_part_weighted_segnet_student_live_calibration",
                    )
                )
            )
            direct_live_loss_observed = (
                direct_live_loss_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_segnet_direct_live_distill",
                        "loss_part_pr95_stage_segnet_direct_live_distill",
                        "loss_part_weighted_pr95_stage_segnet_direct_live_distill",
                    )
                )
            )
            direct_live_class_region_recon_observed = (
                direct_live_class_region_recon_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_segnet_direct_live_class_region_recon_loss",
                        "loss_part_pr95_stage_segnet_direct_live_class_region_recon_loss",
                    )
                )
            )
            direct_live_rare_class_logit_observed = (
                direct_live_rare_class_logit_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_segnet_direct_live_rare_class_logit_loss",
                        "loss_part_pr95_stage_segnet_direct_live_rare_class_logit_loss",
                    )
                )
            )
            direct_live_target_mass_floor_observed = (
                direct_live_target_mass_floor_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_segnet_direct_live_target_mass_floor_loss",
                        "loss_part_pr95_stage_segnet_direct_live_target_mass_floor_loss",
                    )
                )
            )
            direct_live_target_min_ratio_floor_observed = (
                direct_live_target_min_ratio_floor_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_segnet_direct_live_target_min_ratio_floor_loss",
                        "loss_part_pr95_stage_segnet_direct_live_target_min_ratio_floor_loss",
                    )
                )
            )
            direct_live_argmax_observed = (
                direct_live_argmax_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "loss_part_segnet_direct_live_argmax_disagreement",
                        "loss_part_pr95_stage_segnet_direct_live_argmax_disagreement",
                    )
                )
            )
            for key in (
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction",
                "loss_part_pr95_stage_segnet_direct_live_candidate_occupied_class_fraction",
            ):
                value = _telemetry_row_value(row, key)
                if _finite_number(value):
                    direct_live_class_occupancy_observed = True
                    fraction = float(value)
                    if (
                        direct_live_max_candidate_occupied_class_fraction is None
                        or fraction
                        > direct_live_max_candidate_occupied_class_fraction
                    ):
                        direct_live_max_candidate_occupied_class_fraction = fraction
            for key in (
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction",
                "loss_part_pr95_stage_segnet_direct_live_candidate_target_class_coverage_fraction",
            ):
                value = _telemetry_row_value(row, key)
                if _finite_number(value):
                    direct_live_target_class_coverage_observed = True
                    fraction = float(value)
                    if (
                        direct_live_max_candidate_target_class_coverage_fraction
                        is None
                        or fraction
                        > direct_live_max_candidate_target_class_coverage_fraction
                    ):
                        direct_live_max_candidate_target_class_coverage_fraction = (
                            fraction
                        )
            section_rate_observed = section_rate_observed or any(
                str(key).startswith("train_time_section_rate_score__")
                and _finite_number(value)
                for key, value in _telemetry_row_items(row)
            )
            archive_rate_observed = archive_rate_observed or _finite_number_in_row(
                row,
                "train_time_archive_rate_score",
            )
            archive_dual_lambda_active_observed = (
                archive_dual_lambda_active_observed
                or _row_nonzero_finite_number(
                    row,
                    "dual_ascent_lambda__snerv_archive_total_bytes",
                )
            )
            archive_violation = _telemetry_row_value(
                row,
                "dual_ascent_violation__snerv_archive_total_bytes",
            )
            archive_dual_positive_violation_observed = (
                archive_dual_positive_violation_observed
                or (
                    _finite_number(archive_violation)
                    and float(archive_violation) > 0.0
                )
            )
            archive_dual_update_observed = (
                archive_dual_update_observed
                or _row_nonzero_finite_number(
                    row,
                    "dual_ascent_update_count__snerv_archive_total_bytes",
                )
            )
            archive_dual_weight_applied_observed = (
                archive_dual_weight_applied_observed
                or _row_nonzero_finite_number(
                    row,
                    "dual_ascent_weight_applied__snerv_archive_total_bytes",
                )
                or _row_nonzero_finite_number(
                    row,
                    "dual_ascent_effective_loss_weight__snerv_archive_total_bytes",
                )
            )
            section_dual_lambda_active_observed = (
                section_dual_lambda_active_observed
                or any(
                    str(key).startswith("dual_ascent_lambda__snerv_")
                    and str(key).endswith("_section_bytes")
                    and _finite_number(value)
                    and abs(float(value)) > 0.0
                    for key, value in _telemetry_row_items(row)
                )
            )
            section_dual_positive_violation_observed = (
                section_dual_positive_violation_observed
                or any(
                    str(key).startswith("dual_ascent_violation__snerv_")
                    and str(key).endswith("_section_bytes")
                    and _finite_number(value)
                    and float(value) > 0.0
                    for key, value in _telemetry_row_items(row)
                )
            )
            section_dual_update_observed = (
                section_dual_update_observed
                or any(
                    str(key).startswith("dual_ascent_update_count__snerv_")
                    and str(key).endswith("_section_bytes")
                    and _finite_number(value)
                    and abs(float(value)) > 0.0
                    for key, value in _telemetry_row_items(row)
                )
            )
            section_dual_weight_applied_observed = (
                section_dual_weight_applied_observed
                or any(
                    (
                        str(key).startswith("dual_ascent_weight_applied__snerv_")
                        or str(key).startswith(
                            "dual_ascent_effective_loss_weight__snerv_"
                        )
                    )
                    and str(key).endswith("_section_bytes")
                    and _finite_number(value)
                    and abs(float(value)) > 0.0
                    for key, value in _telemetry_row_items(row)
                )
            )
            section_dual_zero_base_masked_observed = (
                section_dual_zero_base_masked_observed
                or any(
                    str(key).startswith("dual_ascent_zero_base_masked__snerv_")
                    and str(key).endswith("_section_bytes")
                    and _finite_number(value)
                    and abs(float(value)) > 0.0
                    for key, value in _telemetry_row_items(row)
                )
            )
            seg_dual_observed = seg_dual_observed or _row_float_equals(
                row,
                "dual_ascent_missing_metric__snerv_segnet_last_frame_distill",
                0.0,
            )
            direct_live_target_mass_floor_dual_observed = (
                direct_live_target_mass_floor_dual_observed
                or _row_float_equals(
                    row,
                    "dual_ascent_missing_metric__snerv_segnet_direct_live_target_mass_floor",
                    0.0,
                )
            )
            direct_live_target_min_ratio_floor_dual_observed = (
                direct_live_target_min_ratio_floor_dual_observed
                or _row_float_equals(
                    row,
                    "dual_ascent_missing_metric__snerv_segnet_direct_live_target_min_ratio_floor",
                    0.0,
                )
            )
            pose_dual_observed = pose_dual_observed or _row_float_equals(
                row,
                "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill",
                0.0,
            )
            guard_dual_observed = guard_dual_observed or _row_float_equals(
                row,
                "dual_ascent_missing_metric__snerv_scorer_input_distribution_guard",
                0.0,
            )
            seg_dual_lambda_active_observed = (
                seg_dual_lambda_active_observed
                or _row_nonzero_finite_number(
                    row,
                    "dual_ascent_lambda__snerv_segnet_last_frame_distill",
                )
            )
            direct_live_target_mass_floor_dual_lambda_active_observed = (
                direct_live_target_mass_floor_dual_lambda_active_observed
                or _row_nonzero_finite_number(
                    row,
                    "dual_ascent_lambda__snerv_segnet_direct_live_target_mass_floor",
                )
            )
            direct_live_target_min_ratio_floor_dual_lambda_active_observed = (
                direct_live_target_min_ratio_floor_dual_lambda_active_observed
                or _row_nonzero_finite_number(
                    row,
                    "dual_ascent_lambda__snerv_segnet_direct_live_target_min_ratio_floor",
                )
            )
            pose_dual_lambda_active_observed = (
                pose_dual_lambda_active_observed
                or _row_nonzero_finite_number(
                    row,
                    "dual_ascent_lambda__snerv_posenet_yuv6_pair_distill",
                )
            )
            gradient_multiplier_requested_observed = (
                gradient_multiplier_requested_observed
                or _row_nonzero_finite_number(
                    row,
                    "gradient_multiplier_requested_control_count",
                )
            )
            gradient_multiplier_applied_observed = (
                gradient_multiplier_applied_observed
                or _row_nonzero_finite_number(
                    row,
                    "gradient_multiplier_applied_leaf_count",
                )
            )
            gradient_multiplier_missing_requested_observed = (
                gradient_multiplier_missing_requested_observed
                or _row_nonzero_finite_number(
                    row,
                    "gradient_multiplier_missing_requested_count",
                )
            )
            gradient_multiplier_noop_observed = (
                gradient_multiplier_noop_observed
                or _row_float_equals(
                    row,
                    "gradient_multiplier_requested_but_unapplied",
                    1.0,
                )
            )
            scorer_space_step_guard_config_observed = (
                scorer_space_step_guard_config_observed
                or _row_float_equals(row, "scorer_space_step_guard_enabled", 1.0)
            )
            scorer_space_step_guard_metric_observed = (
                scorer_space_step_guard_metric_observed
                or any(
                    _finite_number_in_row(row, key)
                    for key in (
                        "scorer_space_step_guard_eligible",
                        "scorer_space_step_guard_rejected",
                        "scorer_space_step_guard_effective_optimizer_learning_rate",
                        "scorer_space_step_guard_optimizer_learning_rate_scale",
                    )
                )
            )
            scorer_space_step_guard_intervention_observed = (
                scorer_space_step_guard_intervention_observed
                or _row_nonzero_finite_number(row, "scorer_space_step_guard_rejected")
                or _row_nonzero_finite_number(
                    row,
                    "scorer_space_step_guard_intervened",
                )
                or _row_nonzero_finite_number(
                    row,
                    "scorer_space_step_guard_backtracking_accepted",
                )
            )
    if row_count <= 0:
        blockers.append("snerv_score_aware_long_training_telemetry_empty")
    if malformed_rows:
        blockers.append("snerv_score_aware_long_training_telemetry_malformed_rows")
    if expected_seg and not seg_loss_observed:
        blockers.append("snerv_score_aware_long_training_segnet_loss_metric_missing")
    if expected_pose and not pose_loss_observed:
        blockers.append("snerv_score_aware_long_training_posenet_loss_metric_missing")
    if expected_pose_direct_live and not pose_direct_live_loss_observed:
        blockers.append(
            "snerv_score_aware_long_training_direct_live_posenet_loss_missing"
        )
    if expected_pose_direct_live and not pose_direct_live_score_term_observed:
        blockers.append(
            "snerv_score_aware_long_training_direct_live_posenet_score_term_metric_missing"
        )
    if expected_pose_direct_live and not pose_direct_live_raw_mse_observed:
        blockers.append(
            "snerv_score_aware_long_training_direct_live_posenet_raw_mse_metric_missing"
        )
    if expected_seg and not seg_dual_observed:
        blockers.append("snerv_score_aware_long_training_dual_segnet_metric_never_observed")
    if expected_pose and not pose_dual_observed:
        blockers.append("snerv_score_aware_long_training_dual_posenet_metric_never_observed")
    if expected_seg and not seg_dual_lambda_active_observed:
        blockers.append("snerv_score_aware_long_training_dual_segnet_lambda_never_active")
    if expected_pose and not pose_dual_lambda_active_observed:
        blockers.append("snerv_score_aware_long_training_dual_posenet_lambda_never_active")
    if expected_section and not section_rate_observed:
        blockers.append("snerv_score_aware_long_training_section_rate_metric_missing")
    if (
        expected_section
        and section_dual_positive_violation_observed
        and not section_dual_lambda_active_observed
    ):
        blockers.append(
            "snerv_score_aware_long_training_section_byte_dual_lambda_never_active"
        )
    if (
        expected_section
        and section_dual_lambda_active_observed
        and section_dual_positive_violation_observed
        and not section_dual_weight_applied_observed
        and not (section_dual_update_observed and row_count <= 2)
    ):
        blockers.append(
            "snerv_score_aware_long_training_section_byte_dual_weight_never_applied"
        )
    if expected_section and section_dual_zero_base_masked_observed:
        blockers.append(
            "snerv_score_aware_long_training_section_byte_dual_zero_base_masked"
        )
    if expected_section and not archive_rate_observed:
        blockers.append("snerv_score_aware_long_training_archive_rate_metric_missing")
    if (
        expected_section
        and archive_dual_positive_violation_observed
        and not archive_dual_lambda_active_observed
    ):
        blockers.append(
            "snerv_score_aware_long_training_archive_byte_dual_lambda_never_active"
        )
    if (
        expected_section
        and archive_dual_lambda_active_observed
        and archive_dual_positive_violation_observed
        and not archive_dual_weight_applied_observed
        and not (archive_dual_update_observed and row_count <= 2)
    ):
        blockers.append(
            "snerv_score_aware_long_training_archive_byte_dual_weight_never_applied"
        )
    if expected_guard and not guard_loss_observed:
        blockers.append("snerv_score_aware_long_training_scorer_input_guard_metric_missing")
    if expected_guard and not guard_dual_observed:
        blockers.append(
            "snerv_score_aware_long_training_dual_scorer_input_guard_metric_never_observed"
        )
    if expected_contrast_floor and not contrast_floor_loss_observed:
        blockers.append(
            "snerv_score_aware_long_training_scorer_input_contrast_floor_metric_missing"
        )
    if expected_contrast_floor and not contrast_floor_segnet_ratio_observed:
        blockers.append(
            "snerv_score_aware_long_training_scorer_input_contrast_floor_segnet_ratio_metric_missing"
        )
    if expected_contrast_floor and not contrast_floor_posenet_ratio_observed:
        blockers.append(
            "snerv_score_aware_long_training_scorer_input_contrast_floor_posenet_ratio_metric_missing"
        )
    if expected_shape_tether and not shape_tether_loss_observed:
        blockers.append(
            "snerv_score_aware_long_training_scorer_input_shape_tether_metric_missing"
        )
    if expected_shape_tether and not shape_tether_segnet_observed:
        blockers.append(
            "snerv_score_aware_long_training_scorer_input_shape_tether_segnet_metric_missing"
        )
    if expected_shape_tether and not shape_tether_posenet_pair_observed:
        blockers.append(
            "snerv_score_aware_long_training_scorer_input_shape_tether_posenet_pair_metric_missing"
        )
    if expected_shape_tether and not shape_tether_posenet_delta_observed:
        blockers.append(
            "snerv_score_aware_long_training_scorer_input_shape_tether_posenet_delta_metric_missing"
        )
    if expected_geometry_tether and not geometry_tether_loss_observed:
        blockers.append(
            "snerv_score_aware_long_training_posenet_yuv6_geometry_tether_metric_missing"
        )
    if expected_geometry_tether and not geometry_tether_pair_observed:
        blockers.append(
            "snerv_score_aware_long_training_posenet_yuv6_geometry_tether_pair_metric_missing"
        )
    if expected_geometry_tether and not geometry_tether_delta_observed:
        blockers.append(
            "snerv_score_aware_long_training_posenet_yuv6_geometry_tether_delta_metric_missing"
        )
    if expected_temporal_floor and not temporal_floor_loss_observed:
        blockers.append(
            "snerv_score_aware_long_training_posenet_temporal_signal_floor_metric_missing"
        )
    if expected_temporal_floor and not temporal_floor_std_ratio_observed:
        blockers.append(
            "snerv_score_aware_long_training_posenet_temporal_signal_floor_std_ratio_metric_missing"
        )
    if expected_temporal_floor and not temporal_floor_mean_abs_ratio_observed:
        blockers.append(
            "snerv_score_aware_long_training_posenet_temporal_signal_floor_mean_abs_ratio_metric_missing"
        )
    if expected_live_calibration and not live_calibration_active_observed:
        blockers.append(
            "snerv_score_aware_long_training_live_segnet_calibration_never_active"
        )
    if expected_live_calibration and not live_calibration_loss_observed:
        blockers.append(
            "snerv_score_aware_long_training_live_segnet_calibration_loss_missing"
        )
    if expected_direct_live and not direct_live_loss_observed:
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_loss_missing"
        )
    if expected_direct_live_region_recon and not direct_live_class_region_recon_observed:
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_class_region_recon_metric_missing"
        )
    if expected_direct_live_rare_class_logit and not direct_live_rare_class_logit_observed:
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_rare_class_logit_metric_missing"
        )
    if expected_direct_live_target_mass_floor and not direct_live_target_mass_floor_observed:
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_target_mass_floor_metric_missing"
        )
    if (
        expected_direct_live_target_mass_floor
        and not direct_live_target_mass_floor_dual_observed
    ):
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_target_mass_floor_dual_metric_never_observed"
        )
    if (
        expected_direct_live_target_mass_floor
        and not direct_live_target_mass_floor_dual_lambda_active_observed
    ):
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_target_mass_floor_dual_lambda_never_active"
        )
    if (
        expected_direct_live_target_min_ratio_floor
        and not direct_live_target_min_ratio_floor_observed
    ):
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_target_min_ratio_floor_metric_missing"
        )
    if (
        expected_direct_live_target_min_ratio_floor
        and not direct_live_target_min_ratio_floor_dual_observed
    ):
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_target_min_ratio_floor_dual_metric_never_observed"
        )
    if (
        expected_direct_live_target_min_ratio_floor
        and not direct_live_target_min_ratio_floor_dual_lambda_active_observed
    ):
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_target_min_ratio_floor_dual_lambda_never_active"
        )
    if expected_direct_live and not direct_live_argmax_observed:
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_argmax_metric_missing"
        )
    if expected_direct_live and not direct_live_class_occupancy_observed:
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_class_occupancy_metric_missing"
        )
    if expected_direct_live and not direct_live_target_class_coverage_observed:
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_target_class_coverage_metric_missing"
        )
    if (
        expected_direct_live
        and direct_live_class_occupancy_observed
        and float(direct_live_max_candidate_occupied_class_fraction or 0.0)
        < 0.400001
    ):
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_candidate_argmax_collapsed"
        )
    if (
        expected_direct_live
        and direct_live_target_class_coverage_observed
        and float(direct_live_max_candidate_target_class_coverage_fraction or 0.0)
        < 0.8
    ):
        blockers.append(
            "snerv_score_aware_long_training_direct_live_segnet_target_class_coverage_collapsed"
        )
    pr95_seg_alias_required = bool(
        pr95_seg_loss_observed
        and (pr95_seg_effective_weight_active or not pr95_seg_effective_weight_seen)
    )
    pr95_pose_alias_required = bool(
        pr95_pose_loss_observed
        and (pr95_pose_effective_weight_active or not pr95_pose_effective_weight_seen)
    )
    if pr95_faithful_curriculum_enabled and pr95_seg_alias_required and not seg_loss_observed:
        blockers.append("snerv_score_aware_long_training_pr95_seg_alias_missing")
    if pr95_faithful_curriculum_enabled and pr95_pose_alias_required and not pose_loss_observed:
        blockers.append("snerv_score_aware_long_training_pr95_pose_alias_missing")
    if expected_gradient_multiplier and not gradient_multiplier_requested_observed:
        blockers.append(
            "snerv_score_aware_long_training_gradient_multiplier_metric_missing"
        )
    if expected_gradient_multiplier and not gradient_multiplier_applied_observed:
        blockers.append(
            "snerv_score_aware_long_training_gradient_multiplier_never_applied"
        )
    if expected_gradient_multiplier and gradient_multiplier_missing_requested_observed:
        blockers.append(
            "snerv_score_aware_long_training_gradient_multiplier_missing_requested_leaf"
        )
    if expected_gradient_multiplier and gradient_multiplier_noop_observed:
        blockers.append(
            "snerv_score_aware_long_training_gradient_multiplier_requested_but_unapplied"
        )
    if expected_scorer_space_step_guard and not scorer_space_step_guard_config_observed:
        blockers.append(
            "snerv_score_aware_long_training_scorer_space_step_guard_config_missing"
        )
    if expected_scorer_space_step_guard and not scorer_space_step_guard_metric_observed:
        blockers.append(
            "snerv_score_aware_long_training_scorer_space_step_guard_metric_missing"
        )
    blockers = _ordered_unique(blockers)
    return {
        "schema": "snerv_score_aware_long_training_telemetry_contract.v1",
        "telemetry_path": path.as_posix(),
        "telemetry_exists": True,
        "row_count": int(row_count),
        "malformed_row_count": int(malformed_rows),
        "expected_segnet_dual": bool(expected_seg),
        "expected_posenet_dual": bool(expected_pose),
        "expected_posenet_direct_live_distillation": bool(expected_pose_direct_live),
        "expected_segnet_live_calibration": bool(expected_live_calibration),
        "expected_segnet_direct_live_distillation": bool(expected_direct_live),
        "expected_segnet_direct_live_class_region_recon": bool(
            expected_direct_live_region_recon
        ),
        "expected_segnet_direct_live_rare_class_logit": bool(
            expected_direct_live_rare_class_logit
        ),
        "expected_segnet_direct_live_target_mass_floor": bool(
            expected_direct_live_target_mass_floor
        ),
        "expected_segnet_direct_live_target_min_ratio_floor": bool(
            expected_direct_live_target_min_ratio_floor
        ),
        "expected_section_rate_metrics": bool(expected_section),
        "expected_section_byte_dual_lambda": bool(expected_section),
        "expected_archive_rate_metric": bool(expected_section),
        "expected_archive_byte_dual_lambda": bool(expected_section),
        "expected_scorer_input_guard_metric": bool(expected_guard),
        "expected_scorer_input_contrast_floor_metric": bool(
            expected_contrast_floor
        ),
        "expected_scorer_input_shape_tether_metric": bool(expected_shape_tether),
        "expected_posenet_yuv6_geometry_tether_metric": bool(
            expected_geometry_tether
        ),
        "expected_posenet_temporal_signal_floor_metric": bool(
            expected_temporal_floor
        ),
        "expected_gradient_multiplier_controls": bool(expected_gradient_multiplier),
        "expected_scorer_space_step_guard": bool(expected_scorer_space_step_guard),
        "segnet_loss_metric_observed": bool(seg_loss_observed),
        "posenet_loss_metric_observed": bool(pose_loss_observed),
        "posenet_direct_live_loss_observed": bool(pose_direct_live_loss_observed),
        "posenet_direct_live_raw_mse_observed": bool(
            pose_direct_live_raw_mse_observed
        ),
        "posenet_direct_live_score_term_observed": bool(
            pose_direct_live_score_term_observed
        ),
        "segnet_dual_metric_observed": bool(seg_dual_observed),
        "posenet_dual_metric_observed": bool(pose_dual_observed),
        "segnet_dual_lambda_active_observed": bool(seg_dual_lambda_active_observed),
        "posenet_dual_lambda_active_observed": bool(pose_dual_lambda_active_observed),
        "archive_rate_metric_observed": bool(archive_rate_observed),
        "archive_byte_dual_lambda_active_observed": bool(
            archive_dual_lambda_active_observed
        ),
        "archive_byte_dual_positive_violation_observed": bool(
            archive_dual_positive_violation_observed
        ),
        "archive_byte_dual_update_observed": bool(archive_dual_update_observed),
        "archive_byte_dual_pending_weight_after_short_update": bool(
            archive_dual_lambda_active_observed
            and archive_dual_positive_violation_observed
            and archive_dual_update_observed
            and not archive_dual_weight_applied_observed
            and row_count <= 2
        ),
        "archive_byte_dual_weight_applied_observed": bool(
            archive_dual_weight_applied_observed
        ),
        "section_rate_metric_observed": bool(section_rate_observed),
        "section_byte_dual_lambda_active_observed": bool(
            section_dual_lambda_active_observed
        ),
        "section_byte_dual_positive_violation_observed": bool(
            section_dual_positive_violation_observed
        ),
        "section_byte_dual_update_observed": bool(section_dual_update_observed),
        "section_byte_dual_pending_weight_after_short_update": bool(
            section_dual_lambda_active_observed
            and section_dual_positive_violation_observed
            and section_dual_update_observed
            and not section_dual_weight_applied_observed
            and row_count <= 2
        ),
        "section_byte_dual_weight_applied_observed": bool(
            section_dual_weight_applied_observed
        ),
        "section_byte_dual_zero_base_masked_observed": bool(
            section_dual_zero_base_masked_observed
        ),
        "scorer_input_guard_metric_observed": bool(guard_loss_observed),
        "scorer_input_guard_dual_metric_observed": bool(guard_dual_observed),
        "scorer_input_contrast_floor_metric_observed": bool(
            contrast_floor_loss_observed
        ),
        "scorer_input_contrast_floor_segnet_ratio_metric_observed": bool(
            contrast_floor_segnet_ratio_observed
        ),
        "scorer_input_contrast_floor_posenet_ratio_metric_observed": bool(
            contrast_floor_posenet_ratio_observed
        ),
        "scorer_input_shape_tether_metric_observed": bool(
            shape_tether_loss_observed
        ),
        "scorer_input_shape_tether_segnet_metric_observed": bool(
            shape_tether_segnet_observed
        ),
        "scorer_input_shape_tether_posenet_pair_metric_observed": bool(
            shape_tether_posenet_pair_observed
        ),
        "scorer_input_shape_tether_posenet_delta_metric_observed": bool(
            shape_tether_posenet_delta_observed
        ),
        "posenet_yuv6_geometry_tether_metric_observed": bool(
            geometry_tether_loss_observed
        ),
        "posenet_yuv6_geometry_tether_pair_metric_observed": bool(
            geometry_tether_pair_observed
        ),
        "posenet_yuv6_geometry_tether_delta_metric_observed": bool(
            geometry_tether_delta_observed
        ),
        "posenet_temporal_signal_floor_metric_observed": bool(
            temporal_floor_loss_observed
        ),
        "posenet_temporal_signal_floor_std_ratio_metric_observed": bool(
            temporal_floor_std_ratio_observed
        ),
        "posenet_temporal_signal_floor_mean_abs_ratio_metric_observed": bool(
            temporal_floor_mean_abs_ratio_observed
        ),
        "segnet_live_calibration_active_observed": bool(
            live_calibration_active_observed
        ),
        "segnet_live_calibration_loss_observed": bool(
            live_calibration_loss_observed
        ),
        "segnet_direct_live_distillation_loss_observed": bool(
            direct_live_loss_observed
        ),
        "segnet_direct_live_class_region_recon_metric_observed": bool(
            direct_live_class_region_recon_observed
        ),
        "segnet_direct_live_rare_class_logit_metric_observed": bool(
            direct_live_rare_class_logit_observed
        ),
        "segnet_direct_live_target_mass_floor_metric_observed": bool(
            direct_live_target_mass_floor_observed
        ),
        "segnet_direct_live_target_mass_floor_dual_metric_observed": bool(
            direct_live_target_mass_floor_dual_observed
        ),
        "segnet_direct_live_target_mass_floor_dual_lambda_active_observed": bool(
            direct_live_target_mass_floor_dual_lambda_active_observed
        ),
        "segnet_direct_live_target_min_ratio_floor_metric_observed": bool(
            direct_live_target_min_ratio_floor_observed
        ),
        "segnet_direct_live_target_min_ratio_floor_dual_metric_observed": bool(
            direct_live_target_min_ratio_floor_dual_observed
        ),
        "segnet_direct_live_target_min_ratio_floor_dual_lambda_active_observed": bool(
            direct_live_target_min_ratio_floor_dual_lambda_active_observed
        ),
        "segnet_direct_live_argmax_metric_observed": bool(
            direct_live_argmax_observed
        ),
        "segnet_direct_live_class_occupancy_metric_observed": bool(
            direct_live_class_occupancy_observed
        ),
        "segnet_direct_live_max_candidate_occupied_class_fraction": (
            direct_live_max_candidate_occupied_class_fraction
        ),
        "segnet_direct_live_target_class_coverage_metric_observed": bool(
            direct_live_target_class_coverage_observed
        ),
        "segnet_direct_live_max_candidate_target_class_coverage_fraction": (
            direct_live_max_candidate_target_class_coverage_fraction
        ),
        "pr95_stage_seg_loss_observed": bool(pr95_seg_loss_observed),
        "pr95_stage_pose_loss_observed": bool(pr95_pose_loss_observed),
        "pr95_stage_seg_effective_weight_active_observed": bool(
            pr95_seg_effective_weight_active
        ),
        "pr95_stage_pose_effective_weight_active_observed": bool(
            pr95_pose_effective_weight_active
        ),
        "gradient_multiplier_requested_observed": bool(
            gradient_multiplier_requested_observed
        ),
        "gradient_multiplier_applied_observed": bool(
            gradient_multiplier_applied_observed
        ),
        "gradient_multiplier_missing_requested_observed": bool(
            gradient_multiplier_missing_requested_observed
        ),
        "gradient_multiplier_noop_observed": bool(gradient_multiplier_noop_observed),
        "scorer_space_step_guard_config_observed": bool(
            scorer_space_step_guard_config_observed
        ),
        "scorer_space_step_guard_metric_observed": bool(
            scorer_space_step_guard_metric_observed
        ),
        "scorer_space_step_guard_intervention_observed": bool(
            scorer_space_step_guard_intervention_observed
        ),
        "passed": not blockers,
        "blockers": blockers,
    }


def _bind_snerv_scorer_tether_dual_targets(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require SNeRV scorer-tether duals to activate instead of self-normalizing."""

    tether_ids = {
        "snerv_segnet_last_frame_distill",
        "snerv_posenet_yuv6_pair_distill",
    }
    out = dict(config)
    bound_constraints: list[dict[str, Any]] = []
    for raw in config.get("constraints") or []:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        if str(row.get("constraint_id") or "") in tether_ids:
            row["target"] = 0.0
            row.pop("target_fraction_of_initial", None)
            row["scorer_tether_launch_gate_target_bound"] = True
        bound_constraints.append(row)
    out["constraints"] = bound_constraints
    out["snerv_scorer_tether_launch_gate_target_policy"] = {
        "schema": "snerv_scorer_tether_dual_target_policy.v1",
        "target": 0.0,
        "target_fraction_of_initial_removed": True,
        "constraint_ids": sorted(
            str(row.get("constraint_id") or "")
            for row in bound_constraints
            if row.get("scorer_tether_launch_gate_target_bound") is True
        ),
        "rationale": (
            "Short prelaunch SNeRV long-training smokes must prove scorer "
            "tether lambdas activate before scalar/shared/spectra successors launch."
        ),
        **FALSE_AUTHORITY,
    }
    return out


def _finite_number_in_row(row: Mapping[str, Any], key: str) -> bool:
    value = _telemetry_row_value(row, key)
    return _finite_number(value)


def _telemetry_row_value(row: Mapping[str, Any], key: str) -> Any:
    """Read canonical telemetry keys from flat or nested harness rows."""

    if key in row:
        return row.get(key)
    loss_components = row.get("loss_components")
    if isinstance(loss_components, Mapping) and key in loss_components:
        return loss_components.get(key)
    return None


def _telemetry_row_items(row: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Return flat + nested loss-component telemetry for contract probes."""

    items: list[tuple[str, Any]] = [(str(key), value) for key, value in row.items()]
    loss_components = row.get("loss_components")
    if isinstance(loss_components, Mapping):
        items.extend((str(key), value) for key, value in loss_components.items())
    return tuple(items)


def _latest_snerv_score_aware_training_metrics(
    telemetry_path: str | Path,
) -> dict[str, Any]:
    """Extract the last valid scorer/rate-control row for agent-first reports."""

    path = Path(telemetry_path).expanduser()
    base: dict[str, Any] = {
        "schema": "snerv_score_aware_latest_training_metrics.v1",
        "telemetry_path": path.as_posix(),
        "telemetry_exists": path.is_file(),
        "row_count": 0,
        "latest_epoch": None,
        "latest_loss": None,
        "blockers": [],
    }
    if not path.is_file():
        return {**base, "blockers": ["snerv_score_aware_latest_telemetry_missing"]}

    last_row: Mapping[str, Any] | None = None
    malformed_rows = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
            if not isinstance(row, Mapping):
                malformed_rows += 1
                continue
            base["row_count"] = int(base["row_count"]) + 1
            last_row = row

    blockers: list[str] = []
    if malformed_rows:
        blockers.append("snerv_score_aware_latest_telemetry_malformed_rows")
    if last_row is None:
        blockers.append("snerv_score_aware_latest_telemetry_empty")
        return {**base, "malformed_rows": malformed_rows, "blockers": blockers}

    def _float_metric(key: str) -> float | None:
        value = _telemetry_row_value(last_row, key)
        if not _finite_number(value):
            return None
        return float(value)

    def _selected_metrics(keys: Sequence[str]) -> dict[str, float]:
        return {
            key: value
            for key in keys
            if (value := _float_metric(key)) is not None
        }

    def _delta(pre_key: str, post_key: str) -> dict[str, float | bool | None]:
        pre = _float_metric(pre_key)
        post = _float_metric(post_key)
        delta = None if pre is None or post is None else post - pre
        return {
            "pre": pre,
            "post": post,
            "delta": delta,
            "improved_or_equal": (None if delta is None else delta <= 0.0),
        }

    section_bytes = {
        key.removeprefix("train_time_section_bytes__"): float(value)
        for key, value in _telemetry_row_items(last_row)
        if str(key).startswith("train_time_section_bytes__")
        and _finite_number(value)
    }
    guard_metrics = _selected_metrics(
        (
            "scorer_space_step_guard_enabled",
            "scorer_space_step_guard_eligible",
            "scorer_space_step_guard_rejected",
            "scorer_space_step_guard_backtracking_accepted",
            "scorer_space_step_guard_backtracking_attempt_count",
            "scorer_space_step_guard_optimizer_learning_rate_scale",
            "scorer_space_step_guard_effective_optimizer_learning_rate",
            "scorer_space_step_guard_max_direct_nonrate_score_worsening",
            "scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening",
            "scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening",
        )
    )
    dynamics_metrics = _selected_metrics(
        (
            "dynamics_gradient_all_l2",
            "dynamics_param_delta_all_l2",
            "gradient_multiplier_applied_leaf_count",
            "gradient_multiplier_missing_requested_leaf_count",
        )
    )
    scorer_deltas = {
        "direct_nonrate_score": _delta(
            "dynamics_pre_update_loss_part_direct_nonrate_score",
            "loss_part_direct_nonrate_score",
        ),
        "pose_direct_live_score_term": _delta(
            "dynamics_pre_update_loss_part_pose_direct_live_score_term",
            "loss_part_pose_direct_live_score_term",
        ),
        "pose_score_term": _delta(
            "dynamics_pre_update_loss_part_pose_score_term",
            "loss_part_pose_score_term",
        ),
        "segnet_direct_live_argmax_disagreement": _delta(
            "dynamics_pre_update_loss_part_segnet_direct_live_argmax_disagreement",
            "loss_part_segnet_direct_live_argmax_disagreement",
        ),
        "segnet_direct_live_target_class_coverage_fraction": _delta(
            "dynamics_pre_update_loss_part_segnet_direct_live_candidate_target_class_coverage_fraction",
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction",
        ),
        "segnet_direct_live_occupied_class_fraction": _delta(
            "dynamics_pre_update_loss_part_segnet_direct_live_candidate_occupied_class_fraction",
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction",
        ),
    }
    return {
        **base,
        "malformed_rows": malformed_rows,
        "latest_epoch": _telemetry_row_value(last_row, "epoch"),
        "latest_loss": _float_metric("loss"),
        "train_time_archive_bytes": _float_metric("train_time_archive_bytes"),
        "train_time_section_bytes": dict(sorted(section_bytes.items())),
        "scorer_space_step_guard": guard_metrics,
        "scorer_deltas": scorer_deltas,
        "dynamics": dynamics_metrics,
        "blockers": blockers,
    }


def _finite_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric))


def _row_float_equals(row: Mapping[str, Any], key: str, expected: float) -> bool:
    value = _telemetry_row_value(row, key)
    if not _finite_number(value):
        return False
    return float(value) == float(expected)


def _row_nonzero_finite_number(
    row: Mapping[str, Any],
    key: str,
    *,
    epsilon: float = 1.0e-12,
) -> bool:
    value = _telemetry_row_value(row, key)
    if not _finite_number(value):
        return False
    return bool(abs(float(value)) > float(epsilon))


SNERV_OFFICIAL_CODER_QAT_ADDITIONAL_INCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "low",
    "skip_mid",
    "skip_high",
    "tub_temporal_encoder_concat",
    "tub_output2_raw",
)


def _snerv_official_coder_qat_include_substrings(
    base_include_substrings: Sequence[str],
) -> tuple[str, ...]:
    """Return QAT selectors that cover official frame-producing payload atoms."""

    out: list[str] = []
    for token in (
        *tuple(str(item) for item in base_include_substrings),
        *SNERV_OFFICIAL_CODER_QAT_ADDITIONAL_INCLUDE_SUBSTRINGS,
    ):
        text = str(token)
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _build_snerv_pretraining_archive_section_qat_weight_policy(
    *,
    pairs_nchw255: np.ndarray,
    model_size: SnervModelSizeConfig,
    levels: int,
    wavelet: str,
    source_pair_indices: tuple[int, ...],
    target_bits_per_coeff: float,
    step_map_bits_per_coeff: float,
    decoder_payload_codec: str,
    lf_payload_codec: str,
    recon_pixel_weight: np.ndarray | None,
    recon_pixel_weight_metadata: Mapping[str, Any] | None,
    hf_decoder_saliency_gain: float,
    hard_byte_ceiling: int | None,
    base_qat_weights: Mapping[str, float],
    max_section_multiplier: float = 8.0,
) -> dict[str, Any]:
    """Price exact baseline SNAR1 sections before SNeRV long training.

    The long-training renderer has two distinct train-time actuation surfaces:
    decoder/HFR/MFU tensors and LF latents.  The receiver packet tells us which
    surface is actually byte-heavy for this candidate.  This policy therefore
    scales ``coder_qat_*`` terms from ``decoder_payload`` and emits separate
    ``latent_qat_*`` weights from ``lf_payload`` instead of pretending all rate
    pressure lives in one generic decoder blob.
    """

    base_weights = {
        str(key): float(value)
        for key, value in dict(base_qat_weights or {}).items()
        if float(value) >= 0.0
    }
    base: dict[str, Any] = {
        "schema": SNERV_ARCHIVE_SECTION_QAT_WEIGHT_POLICY_SCHEMA,
        "attached": bool(base_weights),
        "active": False,
        "baseline_packet_bytes": None,
        "baseline_packet_sha256": None,
        "hard_byte_ceiling": hard_byte_ceiling,
        "max_section_multiplier": float(max_section_multiplier),
        "base_loss_weights": base_weights,
        "extra_loss_weights": dict(base_weights),
        "affected_loss_weight_keys": [],
        "section_rows": [],
        "official_decoder_payload_component_rows": [],
        "applied_section_operators": [],
        "pending_section_operators": [],
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    if not base_weights:
        return {
            **base,
            "blockers": ["snerv_archive_section_qat_base_weights_empty"],
        }
    if not any(float(value) > 0.0 for value in base_weights.values()):
        return {
            **base,
            "blockers": ["snerv_archive_section_qat_base_weights_all_zero"],
        }
    try:
        baseline_packet = build_snerv_mlx_native_packet_from_numpy_pairs(
            pairs_nchw255,
            levels=int(levels),
            wavelet=str(wavelet),
            target_bits_per_coeff=float(target_bits_per_coeff),
            step_map_bits_per_coeff=float(step_map_bits_per_coeff),
            decoder_payload_codec=str(decoder_payload_codec),
            lf_payload_codec=str(lf_payload_codec),
            model_size=model_size,
            source_pair_indices=source_pair_indices,
            recon_pixel_weight=recon_pixel_weight,
            recon_pixel_weight_metadata=recon_pixel_weight_metadata,
            hf_decoder_saliency_gain=float(hf_decoder_saliency_gain),
            native_mlx_decoder_train_steps=0,
            metadata_extra={
                "section_qat_pretraining_baseline": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        )
        decoded = unpack_snerv_archive(baseline_packet.packet)
    except Exception as exc:
        return {
            **base,
            "blockers": [
                "snerv_archive_section_qat_pretraining_packet_failed",
                f"snerv_archive_section_qat_pretraining_packet_exception_{type(exc).__name__}",
            ],
            "failure": repr(exc),
        }

    section_bytes = {str(name): len(blob) for name, blob in decoded.sections.items()}
    section_rows = _snerv_archive_section_pressure_rows(
        section_bytes,
        packet_bytes=len(baseline_packet.packet),
        archive_bytes=None,
        hard_byte_ceiling=hard_byte_ceiling,
    )
    official_component_rows = (
        _snerv_official_decoder_component_pressure_rows(
            _official_receiver_tensor_map_from_packet(baseline_packet.packet),
            decoder_payload_bytes=section_bytes.get("decoder_payload"),
            packet_bytes=len(baseline_packet.packet),
            archive_bytes=None,
            hard_byte_ceiling=hard_byte_ceiling,
        )
        if bool(model_size.official_mfu_hfr_tub_numeric_primitives_requested)
        else []
    )
    decoder_row = _section_row(section_rows, "decoder_payload")
    lf_row = _section_row(section_rows, "lf_payload")
    if decoder_row is None:
        return {
            **base,
            "baseline_packet_bytes": len(baseline_packet.packet),
            "baseline_packet_sha256": _sha256_bytes(baseline_packet.packet),
            "section_rows": section_rows,
            "official_decoder_payload_component_rows": official_component_rows,
            "blockers": ["snerv_archive_section_qat_decoder_payload_missing"],
        }

    official_payload_requested = bool(
        model_size.official_mfu_hfr_tub_numeric_primitives_requested
    )
    lf_train_time_actuated = not official_payload_requested
    lf_non_actuation_reason = (
        "official_payload_frame_decode_uses_decoder_payload_dummy_lf_member"
        if official_payload_requested
        else ""
    )
    decoder_multiplier = _snerv_qat_multiplier_from_pressure_row(
        decoder_row,
        max_section_multiplier=float(max_section_multiplier),
    )
    lf_multiplier = (
        _snerv_qat_multiplier_from_pressure_row(
            lf_row,
            max_section_multiplier=float(max_section_multiplier),
        )
        if lf_row is not None
        else 1.0
    )
    decoder_weights = {
        key: float(value) * decoder_multiplier for key, value in base_weights.items()
    }
    latent_weights = (
        {
            f"latent_qat_{key.removeprefix('coder_qat_')}": (
                float(value) * lf_multiplier
            )
            for key, value in base_weights.items()
            if lf_row is not None and int(lf_row.get("bytes") or 0) > 0
        }
        if lf_train_time_actuated
        else {}
    )
    scaled_weights = {**decoder_weights, **latent_weights}
    pending = []
    for row in section_rows:
        name = str(row.get("name") or "")
        if name == "decoder_payload":
            continue
        if name == "lf_payload":
            if latent_weights:
                continue
            pending.append(
                {
                    "section_name": name,
                    "bytes": int(row.get("bytes") or 0),
                    "fraction_of_packet": float(row.get("fraction_of_packet") or 0.0),
                    "required_operator": (
                        "none_official_dummy_lf_payload_not_train_time_actuated"
                        if lf_non_actuation_reason
                        else "lf_latent_qat_or_representation_operator"
                    ),
                    "current_status": (
                        "not_train_time_actuated"
                        if lf_non_actuation_reason
                        else "priced_not_yet_applied"
                    ),
                    "pending_reason": (
                        lf_non_actuation_reason
                        or "lf_payload_active_latent_qat_loss_key_missing"
                    ),
                    **FALSE_AUTHORITY,
                }
            )
            continue
        pending.append(
            {
                "section_name": name,
                "bytes": int(row.get("bytes") or 0),
                "fraction_of_packet": float(row.get("fraction_of_packet") or 0.0),
                "required_operator": (
                    "step_map_waterfill_operator"
                    if name == "step_map_packet"
                    else "packet_layout_or_metadata_codec_operator"
                ),
                "current_status": "priced_not_yet_applied",
                **FALSE_AUTHORITY,
            }
        )
    return {
        **base,
        "active": any(float(value) > 0.0 for value in scaled_weights.values()),
        "baseline_packet_bytes": len(baseline_packet.packet),
        "baseline_packet_sha256": _sha256_bytes(baseline_packet.packet),
        "extra_loss_weights": scaled_weights,
        "affected_loss_weight_keys": sorted(scaled_weights),
        "section_rows": section_rows,
        "official_decoder_payload_component_rows": official_component_rows,
        "decoder_section_bytes": int(decoder_row["bytes"]),
        "decoder_section_fraction": float(decoder_row.get("fraction_of_packet") or 0.0),
        "decoder_pressure_multiplier": float(decoder_multiplier),
        "lf_section_bytes": int(lf_row.get("bytes") or 0) if lf_row is not None else 0,
        "lf_section_fraction": (
            float(lf_row.get("fraction_of_packet") or 0.0) if lf_row is not None else 0.0
        ),
        "lf_pressure_multiplier": float(lf_multiplier),
        "applied_section_operators": [
            {
                "section_name": "decoder_payload",
                "operator": "decoder_coder_qat_loss_weight_scaling",
                "loss_weight_keys": sorted(decoder_weights),
                "multiplier": float(decoder_multiplier),
                **FALSE_AUTHORITY,
            },
            *(
                [
                    {
                        "section_name": "lf_payload",
                        "operator": "lf_latent_coder_qat_loss_weight_scaling",
                        "loss_weight_keys": sorted(latent_weights),
                        "multiplier": float(lf_multiplier),
                        **FALSE_AUTHORITY,
                    }
                ]
                if latent_weights
                else []
            ),
        ],
        "pending_section_operators": pending,
        "non_actuated_section_names": (
            ["lf_payload"] if lf_non_actuation_reason else []
        ),
        "non_actuated_section_reasons": (
            {"lf_payload": lf_non_actuation_reason}
            if lf_non_actuation_reason
            else {}
        ),
        "blockers": [],
    }


def _build_snerv_train_time_section_byte_control(
    archive_section_qat_policy: Mapping[str, Any],
    active_loss_weights: Mapping[str, float],
    *,
    hard_byte_ceiling: int | None,
) -> dict[str, Any]:
    """Turn SNAR section telemetry into real train-time byte-cap constraints."""

    loss_weights = {
        str(key): float(value)
        for key, value in dict(active_loss_weights or {}).items()
        if float(value) >= 0.0
    }
    section_rows = [
        dict(row)
        for row in archive_section_qat_policy.get("section_rows") or []
        if isinstance(row, Mapping) and int(row.get("bytes") or 0) > 0
    ]
    baseline_packet_bytes = _positive_int_or_none(
        archive_section_qat_policy.get("baseline_packet_bytes")
    )
    section_bytes = {
        str(row.get("name") or f"section_{idx:04d}"): int(row.get("bytes") or 0)
        for idx, row in enumerate(section_rows)
    }
    metrics_payload = (
        {
            "schema": "snerv_train_time_section_byte_metrics.v1",
            "archive_bytes": baseline_packet_bytes or sum(section_bytes.values()),
            "section_bytes": dict(sorted(section_bytes.items())),
            "rate_score_per_byte": float(SNERV_CONTEST_RATE_SCORE_PER_BYTE),
            "section_rate_scores": {
                name: float(nbytes) * float(SNERV_CONTEST_RATE_SCORE_PER_BYTE)
                for name, nbytes in sorted(section_bytes.items())
            },
            **{
                f"train_time_section_rate_score__{safe_dual_metric_key(name)}": (
                    float(nbytes) * float(SNERV_CONTEST_RATE_SCORE_PER_BYTE)
                )
                for name, nbytes in sorted(section_bytes.items())
            },
            "authority": "macos_mlx_research_signal_false_authority",
        }
        if section_bytes
        else None
    )
    base: dict[str, Any] = {
        "schema": "snerv_train_time_section_byte_control.v1",
        "attached": bool(section_rows),
        "active": False,
        "hard_byte_ceiling": hard_byte_ceiling,
        "baseline_packet_bytes": baseline_packet_bytes,
        "section_bytes": dict(sorted(section_bytes.items())),
        "section_byte_budgets": {},
        "section_byte_loss_weight_key_map": {},
        "section_byte_loss_weight_scale_map": {},
        "budget_rows": [],
        "pending_section_rows": [],
        "metrics_payload": metrics_payload,
        "controlled_section_count": 0,
        "pending_section_count": 0,
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    policy_blockers = [
        str(blocker)
        for blocker in archive_section_qat_policy.get("blockers") or []
        if str(blocker)
    ]
    non_actuated_sections: dict[str, str] = {
        str(row.get("section_name") or ""): str(
            row.get("pending_reason") or row.get("current_status") or ""
        )
        for row in archive_section_qat_policy.get("pending_section_operators") or []
        if isinstance(row, Mapping)
        and str(row.get("section_name") or "")
        and str(row.get("current_status") or "") == "not_train_time_actuated"
    }
    if policy_blockers:
        return {
            **base,
            "blockers": _ordered_unique(
                [
                    *policy_blockers,
                    "snerv_train_time_section_byte_archive_policy_blocked",
                ]
            ),
        }
    if hard_byte_ceiling is None:
        return {
            **base,
            "pending_section_rows": _snerv_pending_train_time_section_rows(
                section_rows,
                reason="hard_byte_ceiling_not_configured",
                rate_score_per_byte=SNERV_CONTEST_RATE_SCORE_PER_BYTE,
            ),
            "pending_section_count": len(section_rows),
            "blockers": ["snerv_train_time_section_byte_hard_ceiling_missing"],
        }
    if not section_rows or baseline_packet_bytes is None:
        return {
            **base,
            "blockers": ["snerv_train_time_section_byte_section_rows_missing"],
        }
    budgets: dict[str, int] = {}
    loss_key_map: dict[str, str] = {}
    loss_scale_map: dict[str, float] = {}
    budget_rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    denominator = max(int(baseline_packet_bytes), 1)
    for row in section_rows:
        name = str(row.get("name") or "")
        nbytes = int(row.get("bytes") or 0)
        budget = max(1, int(np.floor(float(hard_byte_ceiling) * nbytes / denominator)))
        loss_key = _snerv_train_time_loss_key_for_section(
            name=name,
            loss_weights=loss_weights,
        )
        if loss_key is None:
            non_actuated_reason = non_actuated_sections.get(name)
            pending_rows.append(
                {
                    "section_name": name,
                    "bytes": nbytes,
                    "budget_bytes_if_actuated": budget,
                    "rate_score": float(nbytes)
                    * float(SNERV_CONTEST_RATE_SCORE_PER_BYTE),
                    "budget_rate_score_if_actuated": float(budget)
                    * float(SNERV_CONTEST_RATE_SCORE_PER_BYTE),
                    "pending_reason": (
                        non_actuated_reason
                        if non_actuated_reason
                        else (
                        "active_qat_loss_key_missing"
                        if name in {"decoder_payload", "lf_payload"}
                        else "non_differentiable_archive_section"
                        )
                    ),
                    "current_status": (
                        "not_train_time_actuated"
                        if non_actuated_reason
                        else "pending_train_time_operator"
                    ),
                    **FALSE_AUTHORITY,
                }
            )
            if name in {"decoder_payload", "lf_payload"} and not non_actuated_reason:
                blockers.append(
                    f"snerv_train_time_section_byte_{name}_active_loss_key_missing"
                )
            continue
        budgets[name] = budget
        loss_key_map[name] = loss_key
        loss_scale_map[name] = 1.0
        budget_rows.append(
            {
                "section_name": name,
                "bytes": nbytes,
                "budget_bytes": budget,
                "rate_score": float(nbytes)
                * float(SNERV_CONTEST_RATE_SCORE_PER_BYTE),
                "budget_rate_score": float(budget)
                * float(SNERV_CONTEST_RATE_SCORE_PER_BYTE),
                "loss_weight_key": loss_key,
                "operator": (
                    "decoder_coder_qat_dual_ascent"
                    if name == "decoder_payload"
                    else "lf_latent_coder_qat_dual_ascent"
                ),
                **FALSE_AUTHORITY,
            }
        )
    if not budgets:
        blockers.append("snerv_train_time_section_byte_no_actuated_sections")
    return {
        **base,
        "active": bool(budgets and not blockers),
        "section_byte_budgets": dict(sorted(budgets.items())),
        "section_byte_loss_weight_key_map": dict(sorted(loss_key_map.items())),
        "section_byte_loss_weight_scale_map": dict(sorted(loss_scale_map.items())),
        "budget_rows": budget_rows,
        "pending_section_rows": pending_rows,
        "controlled_section_count": len(budget_rows),
        "pending_section_count": len(pending_rows),
        "blockers": _ordered_unique(blockers),
    }


def _snerv_train_time_section_byte_metric_payload_from_packet(
    packet: SnervArchivePacket,
    *,
    packet_source: str,
    packet_builder_scope: str,
    submission_archive_format: str,
    refresh_call: int,
    refresh_every_steps: int,
) -> dict[str, Any]:
    """Return live submission-format section bytes in the loss-adapter shape."""

    section_bytes = {
        str(name): int(value)
        for name, value in dict(packet.section_bytes).items()
        if int(value) > 0
    }
    if not section_bytes:
        raise SnervMlxNativeExportError(
            "live SNeRV train-time packet emitted no positive section bytes"
        )
    submission_packet, submission_repack = _snerv_submission_packet_for_export(
        packet.packet,
        submission_archive_format=submission_archive_format,
    )
    archive_bytes = len(submission_packet)
    section_total = int(sum(section_bytes.values()))
    return {
        "schema": "snerv_live_train_time_section_byte_metrics.v1",
        "archive_bytes": int(archive_bytes),
        "section_bytes": dict(sorted(section_bytes.items())),
        "rate_score_per_byte": float(SNERV_CONTEST_RATE_SCORE_PER_BYTE),
        "section_rate_scores": {
            name: float(nbytes) * float(SNERV_CONTEST_RATE_SCORE_PER_BYTE)
            for name, nbytes in sorted(section_bytes.items())
        },
        **{
            f"train_time_section_rate_score__{safe_dual_metric_key(name)}": (
                float(nbytes) * float(SNERV_CONTEST_RATE_SCORE_PER_BYTE)
            )
            for name, nbytes in sorted(section_bytes.items())
        },
        "authority": "macos_mlx_research_signal_false_authority",
        "byte_basis": "current_receiver_submission_packet_sections",
        "submission_archive_format": str(submission_archive_format),
        "live_profile": {
            "packet_source": str(packet_source),
            "packet_builder_scope": str(packet_builder_scope),
            "refresh_call": int(refresh_call),
            "refresh_every_steps": int(refresh_every_steps),
            "packet_sha256": _sha256_bytes(packet.packet),
            "packet_schema": packet.schema,
            "packet_bytes_before_submission_repack": int(
                _positive_int_or_none(packet.total_bytes) or len(packet.packet)
            ),
            "submission_packet_sha256": _sha256_bytes(submission_packet),
            "submission_packet_schema": submission_repack.get("output_packet_schema"),
            "submission_packet_bytes": int(archive_bytes),
            "submission_packet_header_bytes": int(max(archive_bytes - section_total, 0)),
            "submission_repack": submission_repack,
        },
        "blockers": [
            "live_metrics_are_submission_packet_bytes_until_export_zip_replay"
        ],
    }


def _build_snerv_live_train_time_section_byte_metrics_callback(
    *,
    model_size: SnervModelSizeConfig,
    levels: int,
    wavelet: str,
    source_pair_indices: tuple[int, ...],
    target_bits_per_coeff: float,
    step_map_bits_per_coeff: float,
    decoder_payload_codec: str,
    lf_payload_codec: str,
    recon_pixel_weight: np.ndarray | None,
    recon_pixel_weight_metadata: Mapping[str, Any] | None,
    hf_decoder_saliency_gain: float,
    train_time_section_byte_control: Mapping[str, Any],
    batch_size: int,
    refresh_every_steps: int | None,
    submission_archive_format: str = "snar2",
) -> tuple[
    Callable[[Any, Any, Mapping[str, float]], Mapping[str, Any] | None] | None,
    dict[str, Any],
]:
    """Build a live current-state receiver byte callback for MLX dual ascent."""

    static_payload_obj = train_time_section_byte_control.get("metrics_payload")
    static_payload = (
        dict(static_payload_obj) if isinstance(static_payload_obj, Mapping) else None
    )
    refresh_every = max(1, int(refresh_every_steps or 25))
    official_metrics_require_current_components = bool(
        model_size.official_mfu_hfr_tub_numeric_primitives_requested
    )
    explicit_diagnostic_payload = train_time_section_byte_control.get("active") is False
    byte_control_requires_live_refresh = bool(
        not explicit_diagnostic_payload
        and (
            static_payload is not None
            or train_time_section_byte_control.get("active")
            or train_time_section_byte_control.get("section_byte_budgets")
            or train_time_section_byte_control.get("section_byte_loss_weight_key_map")
        )
    )
    metadata: dict[str, Any] = {
        "schema": "snerv_live_train_time_section_byte_metrics_callback.v1",
        "attached": static_payload is not None,
        "active": False,
        "refresh_every_steps": int(refresh_every),
        "refresh_calls": 0,
        "cache_hits": 0,
        "fallback_count": 0,
        "uses_current_renderer_state": True,
        "requires_current_official_components": (
            official_metrics_require_current_components
        ),
        "requires_live_refresh_for_active_byte_control": (
            byte_control_requires_live_refresh
        ),
        "writes_artifacts": False,
        "authority_class": "macos_mlx_research_signal_false_authority",
        "submission_archive_format": str(submission_archive_format or "snar2"),
        "packet_builder_scope": (
            "official_mfu_hfr_tub_current_component_packet"
            if official_metrics_require_current_components
            else "rendered_pairs_numpy_portable_receiver_rebuild"
        ),
        "blockers": [],
    }
    if static_payload is None:
        metadata["blockers"] = [
            "snerv_live_section_byte_callback_requires_startup_metrics_payload"
        ]
        return None, metadata

    cache: dict[str, Any] = {}

    def _fallback_payload(reason: str) -> Mapping[str, Any] | None:
        metadata["fallback_count"] = int(metadata.get("fallback_count", 0)) + 1
        metadata["last_fallback_reason"] = str(reason)
        fallback_blockers = [str(reason)]
        if byte_control_requires_live_refresh:
            fallback_blockers.append(
                "snerv_live_section_byte_active_control_static_fallback_forbidden"
            )
        if official_metrics_require_current_components:
            fallback_blockers.extend(
                (
                    SNERV_LIVE_SECTION_BYTE_OFFICIAL_COMPONENTS_MISSING_BLOCKER,
                    SNERV_LIVE_SECTION_BYTE_OFFICIAL_FALLBACK_BLOCKER,
                )
            )
        metadata["blockers"] = _ordered_unique(
            [
                *(str(blocker) for blocker in metadata.get("blockers", ()) if str(blocker)),
                *fallback_blockers,
            ]
        )
        if byte_control_requires_live_refresh:
            return None
        payload = dict(static_payload)
        payload["schema"] = "snerv_train_time_section_byte_metrics_fallback.v1"
        payload["live_profile"] = {
            "fallback_reason": str(reason),
            "refresh_every_steps": int(refresh_every),
            "packet_builder_scope": metadata["packet_builder_scope"],
        }
        payload["blockers"] = _ordered_unique(
            [
                *(
                    str(blocker)
                    for blocker in payload.get("blockers", ())
                    if str(blocker)
                ),
                *fallback_blockers,
            ]
        )
        return payload

    def _build_live_payload(model_obj: Any) -> Mapping[str, Any] | None:
        call_index = int(metadata.get("refresh_calls", 0)) + 1
        metadata["refresh_calls"] = call_index
        if bool(model_size.official_mfu_hfr_tub_numeric_primitives_requested):
            export_components = getattr(model_obj, "export_official_components", None)
            if not callable(export_components):
                return _fallback_payload(
                    "snerv_official_live_section_byte_export_components_missing"
                )
            packet = _build_official_mfu_hfr_tub_packet_from_components(
                export_components(),
                source_pair_indices=source_pair_indices,
                model_size=model_size,
                metadata_extra={
                    "live_train_time_section_byte_probe": True,
                    "score_aware_long_training_packet_probe": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
            )
            packet_source = "current_official_renderer_components"
        else:
            render_pairs = getattr(model_obj, "render_pairs_nchw255", None)
            if not callable(render_pairs):
                return _fallback_payload(
                    "snerv_live_section_byte_render_pairs_missing"
                )
            rendered_pairs = np.asarray(
                render_pairs(batch_size=max(1, int(batch_size))),
                dtype=np.float32,
            )
            packet = build_snerv_mlx_native_packet_from_numpy_pairs(
                rendered_pairs,
                levels=int(levels),
                wavelet=str(wavelet),
                target_bits_per_coeff=float(target_bits_per_coeff),
                step_map_bits_per_coeff=float(step_map_bits_per_coeff),
                decoder_payload_codec=str(decoder_payload_codec),
                lf_payload_codec=str(lf_payload_codec),
                model_size=model_size,
                source_pair_indices=source_pair_indices,
                recon_pixel_weight=recon_pixel_weight,
                recon_pixel_weight_metadata=recon_pixel_weight_metadata,
                hf_decoder_saliency_gain=float(hf_decoder_saliency_gain),
                native_mlx_decoder_train_steps=0,
                metadata_extra={
                    "live_train_time_section_byte_probe": True,
                    "score_aware_long_training_packet_probe": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
            )
            packet_source = "current_renderer_rendered_pairs"
        payload = _snerv_train_time_section_byte_metric_payload_from_packet(
            packet,
            packet_source=packet_source,
            packet_builder_scope=str(metadata["packet_builder_scope"]),
            submission_archive_format=str(metadata["submission_archive_format"]),
            refresh_call=call_index,
            refresh_every_steps=refresh_every,
        )
        metadata["active"] = True
        metadata["last_archive_bytes"] = int(payload["archive_bytes"])
        metadata["last_section_bytes"] = dict(payload["section_bytes"])
        metadata["last_packet_source"] = packet_source
        cache.clear()
        cache.update(payload)
        return dict(payload)

    def _callback(
        model_obj: Any,
        _idx: Any,
        _loss_weights: Mapping[str, float],
    ) -> Mapping[str, Any] | None:
        call_count = int(metadata.get("callback_calls", 0)) + 1
        metadata["callback_calls"] = call_count
        should_refresh = not cache or ((call_count - 1) % refresh_every == 0)
        if not should_refresh:
            metadata["cache_hits"] = int(metadata.get("cache_hits", 0)) + 1
            return dict(cache)
        try:
            return _build_live_payload(model_obj)
        except Exception as exc:
            metadata["last_exception"] = f"{type(exc).__name__}: {exc}"
            return _fallback_payload(
                f"snerv_live_section_byte_refresh_failed_{type(exc).__name__}"
            )

    return _callback, metadata


def _snerv_live_section_byte_metrics_blockers(
    metadata: Mapping[str, Any],
    *,
    train_time_section_byte_control_bound: bool,
) -> list[str]:
    """Return blockers that make live section-byte pressure untrustworthy."""

    if not train_time_section_byte_control_bound:
        return []
    blockers = [
        str(blocker)
        for blocker in metadata.get("blockers") or ()
        if str(blocker)
    ]
    if metadata.get("requires_current_official_components") is True:
        if int(metadata.get("fallback_count") or 0) > 0:
            blockers.append(SNERV_LIVE_SECTION_BYTE_OFFICIAL_FALLBACK_BLOCKER)
        if metadata.get("active") is not True:
            blockers.append(
                SNERV_LIVE_SECTION_BYTE_OFFICIAL_NEVER_REFRESHED_BLOCKER
            )
    if metadata.get("requires_live_refresh_for_active_byte_control") is True:
        if int(metadata.get("fallback_count") or 0) > 0:
            blockers.append(
                "snerv_live_section_byte_active_control_static_fallback_forbidden"
            )
        if metadata.get("active") is not True:
            blockers.append(
                "snerv_live_section_byte_active_control_never_refreshed_current_packet"
            )
    return _ordered_unique(blockers)


def _positive_int_or_none(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _snerv_pending_train_time_section_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    reason: str,
    rate_score_per_byte: float = SNERV_CONTEST_RATE_SCORE_PER_BYTE,
) -> list[dict[str, Any]]:
    return [
        {
            "section_name": str(row.get("name") or ""),
            "bytes": int(row.get("bytes") or 0),
            "rate_score": float(int(row.get("bytes") or 0))
            * float(rate_score_per_byte),
            "pending_reason": reason,
            **FALSE_AUTHORITY,
        }
        for row in rows
    ]


def _snerv_train_time_loss_key_for_section(
    *,
    name: str,
    loss_weights: Mapping[str, float],
) -> str | None:
    if name == "decoder_payload":
        candidates = (
            "coder_qat_c1a_entropy",
            "coder_qat_quant_residual",
            "coder_qat_delta",
            "coder_qat_magnitude",
        )
    elif name == "lf_payload":
        candidates = (
            "latent_qat_c1a_entropy",
            "latent_qat_quant_residual",
            "latent_qat_delta",
            "latent_qat_magnitude",
        )
    else:
        return None
    for key in candidates:
        if float(loss_weights.get(key, 0.0)) > 0.0:
            return key
    return None


def _section_row(
    rows: Sequence[Mapping[str, Any]],
    name: str,
) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("name") or "") == str(name):
            return dict(row)
    return None


def _snerv_qat_multiplier_from_pressure_row(
    row: Mapping[str, Any],
    *,
    max_section_multiplier: float,
) -> float:
    fraction = max(0.0, float(row.get("fraction_of_packet") or 0.0))
    over_cap = row.get("exceeds_hard_byte_ceiling")
    multiplier = 1.0 + fraction
    if over_cap is True:
        multiplier += 1.0
    return min(max(1.0, multiplier), max(1.0, float(max_section_multiplier)))


def _coerce_checkpoint_keep_last(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SnervMlxNativeExportError(
            "checkpoint_retention_keep_last_n must be integer, -1, or None; "
            f"got bool {value!r}"
        )
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise SnervMlxNativeExportError(
            "checkpoint_retention_keep_last_n must be integer, -1, or None; "
            f"got {value!r}"
        ) from exc
    if resolved == -1:
        return None
    if resolved < 0:
        raise SnervMlxNativeExportError(
            "checkpoint_retention_keep_last_n must be >= -1; "
            f"got {resolved!r}"
        )
    return resolved


def _coerce_checkpoint_keep_best(value: Any) -> int:
    if isinstance(value, bool):
        raise SnervMlxNativeExportError(
            "checkpoint_retention_keep_best_n must be non-negative integer; "
            f"got bool {value!r}"
        )
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise SnervMlxNativeExportError(
            "checkpoint_retention_keep_best_n must be non-negative integer; "
            f"got {value!r}"
        ) from exc
    if resolved < 0:
        raise SnervMlxNativeExportError(
            "checkpoint_retention_keep_best_n must be non-negative; "
            f"got {resolved!r}"
        )
    return resolved


def _coerce_checkpoint_keep_every(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SnervMlxNativeExportError(
            "checkpoint_retention_keep_every_n_epochs must be positive integer "
            f"or None; got bool {value!r}"
        )
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise SnervMlxNativeExportError(
            "checkpoint_retention_keep_every_n_epochs must be positive integer "
            f"or None; got {value!r}"
        ) from exc
    if resolved <= 0:
        raise SnervMlxNativeExportError(
            "checkpoint_retention_keep_every_n_epochs must be > 0; "
            f"got {resolved!r}"
        )
    return resolved


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
    step_map_waterfill_bits_per_coeff: float | None
    step_map_coder_groups: tuple[dict[str, Any], ...]
    decoder_payload_codec: str
    lf_payload_codec: str
    model_size: dict[str, Any]
    bridge_drift: dict[str, Any]
    receiver_target_reconstruction_profile: dict[str, Any]
    receiver_export_reconstruction_profile: dict[str, Any]
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
    native_mlx_decoder_train_optimizer: str = "pact_guarded_adamw",
    score_aware_long_training_epochs: int = 0,
    score_aware_long_training_lr: float = 1.0e-3,
    score_aware_long_training_batch_pairs: int = 2,
    score_aware_long_training_section_byte_refresh_every_steps: int = 25,
    score_aware_long_training_optimizer: str = "pact_muon_adamw",
    score_aware_long_training_grad_clip_max_norm: float | None = 1.0,
    score_aware_long_training_weight_decay: float | None = 1.0e-4,
    score_aware_long_training_eval_roundtrip_ste: bool = False,
    score_aware_long_training_scorer_input_distribution_guard_weight: float = (
        DEFAULT_SNERV_SCORER_INPUT_DISTRIBUTION_GUARD_WEIGHT
    ),
    score_aware_long_training_scorer_input_distribution_guard_saturation_margin: float = 0.02,
    score_aware_long_training_scorer_input_distribution_guard_temperature: float = 0.01,
    score_aware_long_training_scorer_input_contrast_floor_weight: float = 0.0,
    score_aware_long_training_scorer_input_contrast_floor_segnet_min_std_ratio: float = 0.5,
    score_aware_long_training_scorer_input_contrast_floor_posenet_yuv6_min_std_ratio: float = 0.5,
    score_aware_long_training_scorer_input_shape_tether_weight: float = 0.0,
    score_aware_long_training_posenet_yuv6_geometry_tether_weight: float = 0.0,
    score_aware_long_training_posenet_temporal_signal_floor_weight: float = 0.0,
    score_aware_long_training_posenet_temporal_signal_min_std_ratio: float = 0.25,
    score_aware_long_training_posenet_temporal_signal_min_mean_abs_ratio: float = 0.25,
    score_aware_long_training_scorer_space_step_guard_enabled: bool = True,
    score_aware_long_training_scorer_space_step_guard_min_pre_segnet_occupied_class_fraction: float = 0.4,
    score_aware_long_training_scorer_space_step_guard_min_post_segnet_occupied_class_fraction: float = 0.4,
    score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction: float
    | None = 0.8,
    score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_post_segnet_contrast_ratio: float
    | None = 4.25,
    score_aware_long_training_scorer_space_step_guard_max_post_segnet_distribution_mae: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_post_segnet_argmax_disagreement: float
    | None = 0.5,
    score_aware_long_training_scorer_space_step_guard_max_post_pose_score_term: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_post_pose_direct_live_score_term: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_pose_score_term_relative_worsening: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_pose_score_term_absolute_worsening: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_direct_nonrate_score_worsening: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening: float
    | None = 5.0,
    score_aware_long_training_scorer_space_step_guard_backtracking_steps: int = 6,
    score_aware_long_training_scorer_space_step_guard_backtracking_shrink: float = 0.5,
    score_aware_long_training_loss_weights: Mapping[str, float] | None = None,
    score_aware_long_training_pose_warmup_epochs: int = 0,
    score_aware_long_training_scorer_input_shape_warmup_epochs: int = 0,
    score_aware_long_training_segnet_direct_live_escape_warmup_epochs: int = 0,
    score_aware_long_training_segnet_direct_live_escape_class_multiplier: float = 1.0,
    score_aware_long_training_checkpoint_retention_keep_last_n: int | None = (
        SNERV_SCORE_AWARE_CHECKPOINT_RETENTION_KEEP_LAST_N_DEFAULT
    ),
    score_aware_long_training_checkpoint_retention_keep_best_n: int = 1,
    score_aware_long_training_checkpoint_retention_keep_every_n_epochs: int | None = None,
    score_aware_long_training_checkpoint_retention_cold_store_roots: tuple[Path, ...] = (),
    segnet_distillation_weight: float = 0.0,
    pose_distillation_weight: float = 0.0,
    pose_direct_live_distillation_weight: float = 0.0,
    pose_distillation_loss: str = "mse",
    pose_distillation_huber_delta: float = 1.0,
    segnet_distillation_objective: str = "kl_t2",
    distillation_temperature: float = 2.0,
    segnet_student_live_calibration_weight: float = 1.0,
    segnet_direct_live_distillation_weight: float = 0.0,
    segnet_direct_live_base_loss_weight: float = 1.0,
    segnet_direct_live_class_histogram_weight: float = 0.0,
    segnet_direct_live_class_balanced_hinge_weight: float = 0.0,
    segnet_direct_live_class_balanced_ce_weight: float = 0.0,
    segnet_direct_live_class_balanced_squared_hinge_weight: float = 0.0,
    segnet_direct_live_class_region_recon_weight: float = 0.0,
    segnet_direct_live_rare_class_logit_weight: float = 0.0,
    segnet_direct_live_target_mass_floor_weight: float = 0.0,
    segnet_direct_live_target_min_ratio_floor_weight: float = 0.0,
    segnet_tau_boundary: float = 1.0,
    segnet_hinge_margin: float = 1.0,
    distillation_device: str = "cpu",
    allow_segnet_only_research: bool = False,
    coder_aware_qat: bool = False,
    coder_qat_quant_bits: int = 8,
    coder_qat_quant_residual_weight: float = 1.0e-3,
    coder_qat_magnitude_weight: float = 1.0e-4,
    coder_qat_delta_weight: float = 2.0e-4,
    coder_qat_c1a_entropy_weight: float = 1.0e-4,
    coder_qat_c1a_sigma: float = 0.2,
    coder_qat_c1a_sample_size: int = 512,
    score_aware_long_training_pr95_faithful_curriculum: bool = False,
    score_aware_long_training_pr95_muon_policy: str = "every_stage",
    score_aware_long_training_pr95_source_weight_amplification: bool = False,
    score_aware_long_training_gradient_multiplier_by_name: Mapping[str, float] | Sequence[Any] | None = None,
    score_aware_long_training_bias_gradient_multiplier: float | None = None,
    score_aware_long_training_output_head_bias_gradient_multiplier: float = 1.0,
    official_trained_checkpoint_state_dict: Mapping[str, Any] | None = None,
    official_trained_checkpoint_state_dict_path: str | Path | None = None,
    official_trained_checkpoint_decoder_len: int | None = None,
    official_trained_checkpoint_state_dict_kind: str = (
        "official_trained_checkpoint_state_dict"
    ),
    write_mlx_prefilter_profile: bool = False,
    mlx_prefilter_scorer_device: str = "cpu",
    mlx_prefilter_scorer_batch_pairs: int = 1,
    mlx_prefilter_progress_every: int = 50,
    scorer_loop_qat_max_trials: int = 0,
    scorer_loop_qat_search_mode: str = "random_signed",
    scorer_loop_qat_qat_bits: int = 8,
    scorer_loop_qat_decoder_payload_codec: str | None = None,
    scorer_loop_qat_lf_payload_codec: str | None = None,
    scorer_loop_qat_component_guard_mode: str = "pose_seg_hard",
    scorer_loop_qat_pair_guard_min_score_improved_fraction: float = 1.0,
    scorer_loop_qat_pair_guard_max_pose_worsened_fraction: float = 0.0,
    scorer_loop_qat_device: str = "cpu",
    scorer_loop_qat_perturb_scale: float = 0.02,
    scorer_loop_qat_byte_pressure_multiplier: float = 1.0,
    scorer_loop_qat_section_value_pressure_multiplier: float = 1.0,
    scorer_loop_qat_max_archive_byte_growth: int | None = None,
    scorer_loop_qat_byte_growth_admission_mode: str = "hard_cap",
    scorer_loop_qat_pose_slack: float = 0.0,
    scorer_loop_qat_seg_slack: float = 0.0,
    scorer_loop_qat_seed: int = 1337,
    recon_pixel_weight_path: str | Path | None = None,
    recon_pixel_weight_manifest_path: str | Path | None = None,
    recon_pixel_weight_normalize: str = "mean",
    scorer_error_pair_sampling_weights: Mapping[int, float] | None = None,
    scorer_error_pair_curriculum: Mapping[str, Any] | None = None,
    pair_indices: Sequence[Any] | str | None = None,
    prioritized_pair_indices: Sequence[Any] | str | None = None,
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
    hard_byte_ceiling = _hard_byte_ceiling_from_candidate(candidate)
    source_pair_indices = _source_pair_indices_for_native_export(
        int(num_pairs),
        pair_indices=pair_indices,
    )
    priority_pair_indices = normalize_pair_indices(
        prioritized_pair_indices,
        field="prioritized_pair_indices",
    )
    scorer_error_pair_sampling_weights = {
        int(pair): float(weight)
        for pair, weight in dict(scorer_error_pair_sampling_weights or {}).items()
    }
    scorer_error_pair_curriculum = dict(scorer_error_pair_curriculum or {})
    effective_num_pairs = len(source_pair_indices)
    explicit_pair_indices = pair_indices is not None
    official_primitives_requested = bool(model_size.official_mfu_hfr_tub_numeric_primitives_requested)
    official_primitives_blockers = list(model_size.official_mfu_hfr_tub_export_blockers)
    official_checkpoint_controls_requested = bool(
        official_trained_checkpoint_state_dict is not None
        or official_trained_checkpoint_state_dict_path is not None
        or official_trained_checkpoint_decoder_len is not None
    )
    if official_checkpoint_controls_requested and not official_primitives_requested:
        raise SnervMlxNativeExportError(
            "official trained checkpoint controls require "
            "snerv_model_size_adapter="
            f"{SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER!r}"
        )
    official_trained_checkpoint_mapping_manifest = (
        _official_trained_checkpoint_mapping_manifest_from_inputs(
            state_dict=official_trained_checkpoint_state_dict,
            state_dict_path=official_trained_checkpoint_state_dict_path,
            decoder_len=official_trained_checkpoint_decoder_len,
            state_dict_kind=official_trained_checkpoint_state_dict_kind,
        )
    )
    official_checkpoint_mapping_verified = _official_checkpoint_full_mapping_verified(
        official_trained_checkpoint_mapping_manifest
    )
    if official_checkpoint_mapping_verified:
        official_primitives_blockers = [
            str(blocker)
            for blocker in official_primitives_blockers
            if str(blocker)
            not in {
                SNERV_OFFICIAL_MFU_HFR_TUB_WEIGHT_MAPPING_BLOCKER,
                SNERV_OFFICIAL_TRAINED_CHECKPOINT_STATE_DICT_MAPPING_BLOCKER,
            }
        ]
    official_binding = (
        _official_primitives_export_binding(
            repo_root=root,
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
    checkpoint_retention_keep_last_n = _coerce_checkpoint_keep_last(
        _candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_checkpoint_retention_keep_last_n",
                "snerv_score_aware_long_training_checkpoint_retention_keep_last_n",
            ),
            score_aware_long_training_checkpoint_retention_keep_last_n,
        )
    )
    checkpoint_retention_keep_best_n = _coerce_checkpoint_keep_best(
        _candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_checkpoint_retention_keep_best_n",
                "snerv_score_aware_long_training_checkpoint_retention_keep_best_n",
            ),
            score_aware_long_training_checkpoint_retention_keep_best_n,
        )
    )
    checkpoint_retention_keep_every_n_epochs = _coerce_checkpoint_keep_every(
        _candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_checkpoint_retention_keep_every_n_epochs",
                "snerv_score_aware_long_training_checkpoint_retention_keep_every_n_epochs",
            ),
            score_aware_long_training_checkpoint_retention_keep_every_n_epochs,
        )
    )
    score_aware_long_training = _run_score_aware_long_training_attachment(
        requested_epochs=int(
            candidate.get(
                "score_aware_long_training_epochs",
                candidate.get(
                    "snerv_score_aware_long_training_epochs",
                    score_aware_long_training_epochs,
                ),
            )
        ),
        output_dir=out / "snerv_score_aware_long_training",
        pairs_nchw255=pairs_nchw255,
        model_size=model_size,
        levels=levels,
        wavelet=wavelet,
        source_pair_indices=source_pair_indices,
        target_bits_per_coeff=target_bits_per_coeff,
        step_map_bits_per_coeff=step_map_bits_per_coeff,
        decoder_payload_codec=active_decoder_payload_codec,
        lf_payload_codec=active_lf_payload_codec,
        recon_pixel_weight=recon_weight,
        recon_pixel_weight_metadata=recon_weight_metadata,
        hf_decoder_saliency_gain=float(
            candidate.get(
                "hf_decoder_saliency_gain",
                candidate.get("snerv_hf_decoder_saliency_gain", 1.0),
            )
        ),
        hard_byte_ceiling=hard_byte_ceiling,
        learning_rate=float(
            candidate.get(
                "score_aware_long_training_lr",
                candidate.get(
                    "snerv_score_aware_long_training_lr",
                    score_aware_long_training_lr,
                ),
            )
        ),
        batch_pairs=int(
            candidate.get(
                "score_aware_long_training_batch_pairs",
                candidate.get(
                    "snerv_score_aware_long_training_batch_pairs",
                    score_aware_long_training_batch_pairs,
                ),
            )
        ),
        section_byte_refresh_every_steps=int(
            candidate.get(
                "score_aware_long_training_section_byte_refresh_every_steps",
                candidate.get(
                    "snerv_score_aware_long_training_section_byte_refresh_every_steps",
                    candidate.get(
                        "train_time_section_byte_refresh_every_steps",
                        score_aware_long_training_section_byte_refresh_every_steps,
                    ),
                ),
            )
        ),
        optimizer_kind=str(
            candidate.get(
                "score_aware_long_training_optimizer",
                candidate.get(
                    "snerv_score_aware_long_training_optimizer",
                    score_aware_long_training_optimizer,
                ),
            )
        ),
        grad_clip_max_norm=(
            None
            if candidate.get(
                "score_aware_long_training_grad_clip_max_norm",
                candidate.get(
                    "snerv_score_aware_long_training_grad_clip_max_norm",
                    score_aware_long_training_grad_clip_max_norm,
                ),
            )
            is None
            else float(
                candidate.get(
                    "score_aware_long_training_grad_clip_max_norm",
                    candidate.get(
                        "snerv_score_aware_long_training_grad_clip_max_norm",
                        score_aware_long_training_grad_clip_max_norm,
                    ),
                )
            )
        ),
        weight_decay=(
            None
            if candidate.get(
                "score_aware_long_training_weight_decay",
                candidate.get(
                    "snerv_score_aware_long_training_weight_decay",
                    score_aware_long_training_weight_decay,
                ),
            )
            is None
            else float(
                candidate.get(
                    "score_aware_long_training_weight_decay",
                    candidate.get(
                        "snerv_score_aware_long_training_weight_decay",
                        score_aware_long_training_weight_decay,
                    ),
                )
            )
        ),
        eval_roundtrip_ste=bool(
            candidate.get(
                "score_aware_long_training_eval_roundtrip_ste",
                candidate.get(
                    "snerv_score_aware_long_training_eval_roundtrip_ste",
                    score_aware_long_training_eval_roundtrip_ste,
                ),
            )
        ),
        scorer_input_distribution_guard_weight=float(
            candidate.get(
                "score_aware_long_training_scorer_input_distribution_guard_weight",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_input_distribution_guard_weight",
                    score_aware_long_training_scorer_input_distribution_guard_weight,
                ),
            )
        ),
        scorer_input_distribution_guard_saturation_margin=float(
            candidate.get(
                "score_aware_long_training_scorer_input_distribution_guard_saturation_margin",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_input_distribution_guard_saturation_margin",
                    score_aware_long_training_scorer_input_distribution_guard_saturation_margin,
                ),
            )
        ),
        scorer_input_distribution_guard_temperature=float(
            candidate.get(
                "score_aware_long_training_scorer_input_distribution_guard_temperature",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_input_distribution_guard_temperature",
                    score_aware_long_training_scorer_input_distribution_guard_temperature,
                ),
            )
        ),
        scorer_input_contrast_floor_weight=float(
            candidate.get(
                "score_aware_long_training_scorer_input_contrast_floor_weight",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_input_contrast_floor_weight",
                    score_aware_long_training_scorer_input_contrast_floor_weight,
                ),
            )
        ),
        scorer_input_contrast_floor_segnet_min_std_ratio=float(
            candidate.get(
                "score_aware_long_training_scorer_input_contrast_floor_segnet_min_std_ratio",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_input_contrast_floor_segnet_min_std_ratio",
                    score_aware_long_training_scorer_input_contrast_floor_segnet_min_std_ratio,
                ),
            )
        ),
        scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=float(
            candidate.get(
                "score_aware_long_training_scorer_input_contrast_floor_posenet_yuv6_min_std_ratio",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_input_contrast_floor_posenet_yuv6_min_std_ratio",
                    score_aware_long_training_scorer_input_contrast_floor_posenet_yuv6_min_std_ratio,
                ),
            )
        ),
        scorer_input_shape_tether_weight=float(
            candidate.get(
                "score_aware_long_training_scorer_input_shape_tether_weight",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_input_shape_tether_weight",
                    score_aware_long_training_scorer_input_shape_tether_weight,
                ),
            )
        ),
        posenet_yuv6_geometry_tether_weight=float(
            candidate.get(
                "score_aware_long_training_posenet_yuv6_geometry_tether_weight",
                candidate.get(
                    "snerv_score_aware_long_training_posenet_yuv6_geometry_tether_weight",
                    score_aware_long_training_posenet_yuv6_geometry_tether_weight,
                ),
            )
        ),
        posenet_temporal_signal_floor_weight=float(
            candidate.get(
                "score_aware_long_training_posenet_temporal_signal_floor_weight",
                candidate.get(
                    "snerv_score_aware_long_training_posenet_temporal_signal_floor_weight",
                    score_aware_long_training_posenet_temporal_signal_floor_weight,
                ),
            )
        ),
        posenet_temporal_signal_min_std_ratio=float(
            candidate.get(
                "score_aware_long_training_posenet_temporal_signal_min_std_ratio",
                candidate.get(
                    "snerv_score_aware_long_training_posenet_temporal_signal_min_std_ratio",
                    score_aware_long_training_posenet_temporal_signal_min_std_ratio,
                ),
            )
        ),
        posenet_temporal_signal_min_mean_abs_ratio=float(
            candidate.get(
                "score_aware_long_training_posenet_temporal_signal_min_mean_abs_ratio",
                candidate.get(
                    "snerv_score_aware_long_training_posenet_temporal_signal_min_mean_abs_ratio",
                    score_aware_long_training_posenet_temporal_signal_min_mean_abs_ratio,
                ),
            )
        ),
        score_aware_long_training_scorer_space_step_guard_enabled=bool(
            candidate.get(
                "score_aware_long_training_scorer_space_step_guard_enabled",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_space_step_guard_enabled",
                    score_aware_long_training_scorer_space_step_guard_enabled,
                ),
            )
        ),
        score_aware_long_training_scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=float(
            candidate.get(
                "score_aware_long_training_scorer_space_step_guard_min_pre_segnet_occupied_class_fraction",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_space_step_guard_min_pre_segnet_occupied_class_fraction",
                    score_aware_long_training_scorer_space_step_guard_min_pre_segnet_occupied_class_fraction,
                ),
            )
        ),
        score_aware_long_training_scorer_space_step_guard_min_post_segnet_occupied_class_fraction=float(
            candidate.get(
                "score_aware_long_training_scorer_space_step_guard_min_post_segnet_occupied_class_fraction",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_space_step_guard_min_post_segnet_occupied_class_fraction",
                    score_aware_long_training_scorer_space_step_guard_min_post_segnet_occupied_class_fraction,
                ),
            )
        ),
        score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction",
                "snerv_score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction",
            ),
            score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction,
        ),
        score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio",
                "snerv_score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio",
            ),
            score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio,
        ),
        score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop",
            ),
            score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop,
        ),
        score_aware_long_training_scorer_space_step_guard_max_post_segnet_contrast_ratio=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_post_segnet_contrast_ratio",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_post_segnet_contrast_ratio",
            ),
            score_aware_long_training_scorer_space_step_guard_max_post_segnet_contrast_ratio,
        ),
        score_aware_long_training_scorer_space_step_guard_max_post_segnet_distribution_mae=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_post_segnet_distribution_mae",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_post_segnet_distribution_mae",
            ),
            score_aware_long_training_scorer_space_step_guard_max_post_segnet_distribution_mae,
        ),
        score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae",
            ),
            score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae,
        ),
        score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio",
            ),
            score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio,
        ),
        score_aware_long_training_scorer_space_step_guard_max_post_segnet_argmax_disagreement=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_post_segnet_argmax_disagreement",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_post_segnet_argmax_disagreement",
            ),
            score_aware_long_training_scorer_space_step_guard_max_post_segnet_argmax_disagreement,
        ),
        score_aware_long_training_scorer_space_step_guard_max_post_pose_score_term=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_post_pose_score_term",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_post_pose_score_term",
            ),
            score_aware_long_training_scorer_space_step_guard_max_post_pose_score_term,
        ),
        score_aware_long_training_scorer_space_step_guard_max_post_pose_direct_live_score_term=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_post_pose_direct_live_score_term",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_post_pose_direct_live_score_term",
            ),
            score_aware_long_training_scorer_space_step_guard_max_post_pose_direct_live_score_term,
        ),
        score_aware_long_training_scorer_space_step_guard_max_pose_score_term_relative_worsening=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_pose_score_term_relative_worsening",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_pose_score_term_relative_worsening",
            ),
            score_aware_long_training_scorer_space_step_guard_max_pose_score_term_relative_worsening,
        ),
        score_aware_long_training_scorer_space_step_guard_max_pose_score_term_absolute_worsening=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_pose_score_term_absolute_worsening",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_pose_score_term_absolute_worsening",
            ),
            score_aware_long_training_scorer_space_step_guard_max_pose_score_term_absolute_worsening,
        ),
        score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening",
            ),
            score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening,
        ),
        score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening",
            ),
            score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening,
        ),
        score_aware_long_training_scorer_space_step_guard_max_direct_nonrate_score_worsening=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_direct_nonrate_score_worsening",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_direct_nonrate_score_worsening",
            ),
            score_aware_long_training_scorer_space_step_guard_max_direct_nonrate_score_worsening,
        ),
        score_aware_long_training_scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening",
                "snerv_score_aware_long_training_scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening",
            ),
            score_aware_long_training_scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening,
        ),
        score_aware_long_training_scorer_space_step_guard_backtracking_steps=int(
            candidate.get(
                "score_aware_long_training_scorer_space_step_guard_backtracking_steps",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_space_step_guard_backtracking_steps",
                    score_aware_long_training_scorer_space_step_guard_backtracking_steps,
                ),
            )
        ),
        score_aware_long_training_scorer_space_step_guard_backtracking_shrink=float(
            candidate.get(
                "score_aware_long_training_scorer_space_step_guard_backtracking_shrink",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_space_step_guard_backtracking_shrink",
                    score_aware_long_training_scorer_space_step_guard_backtracking_shrink,
                ),
            )
        ),
        score_aware_long_training_loss_weights=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_stage_loss_weights",
                "snerv_score_aware_long_training_stage_loss_weights",
                "score_aware_long_training_loss_weights",
                "snerv_score_aware_long_training_loss_weights",
            ),
            score_aware_long_training_loss_weights,
        ),
        score_aware_long_training_pose_warmup_epochs=int(
            candidate.get(
                "score_aware_long_training_pose_warmup_epochs",
                candidate.get(
                    "snerv_score_aware_long_training_pose_warmup_epochs",
                    score_aware_long_training_pose_warmup_epochs,
                ),
            )
        ),
        score_aware_long_training_scorer_input_shape_warmup_epochs=int(
            candidate.get(
                "score_aware_long_training_scorer_input_shape_warmup_epochs",
                candidate.get(
                    "snerv_score_aware_long_training_scorer_input_shape_warmup_epochs",
                    score_aware_long_training_scorer_input_shape_warmup_epochs,
                ),
            )
        ),
        score_aware_long_training_segnet_direct_live_escape_warmup_epochs=int(
            candidate.get(
                "score_aware_long_training_segnet_direct_live_escape_warmup_epochs",
                candidate.get(
                    "snerv_score_aware_long_training_segnet_direct_live_escape_warmup_epochs",
                    score_aware_long_training_segnet_direct_live_escape_warmup_epochs,
                ),
            )
        ),
        score_aware_long_training_segnet_direct_live_escape_class_multiplier=float(
            candidate.get(
                "score_aware_long_training_segnet_direct_live_escape_class_multiplier",
                candidate.get(
                    "snerv_score_aware_long_training_segnet_direct_live_escape_class_multiplier",
                    score_aware_long_training_segnet_direct_live_escape_class_multiplier,
                ),
            )
        ),
        checkpoint_retention_keep_last_n=checkpoint_retention_keep_last_n,
        checkpoint_retention_keep_best_n=checkpoint_retention_keep_best_n,
        checkpoint_retention_keep_every_n_epochs=checkpoint_retention_keep_every_n_epochs,
        checkpoint_retention_cold_store_roots=tuple(
            Path(root)
            for root in score_aware_long_training_checkpoint_retention_cold_store_roots
        ),
        scorer_upstream_dir=scorer_upstream_dir,
        segnet_distillation_weight=float(
            candidate.get(
                "segnet_distillation_weight",
                candidate.get(
                    "snerv_segnet_distillation_weight",
                    segnet_distillation_weight,
                ),
            )
        ),
        pose_distillation_weight=float(
            candidate.get(
                "pose_distillation_weight",
                candidate.get(
                    "snerv_pose_distillation_weight",
                    pose_distillation_weight,
                ),
            )
        ),
        pose_direct_live_distillation_weight=float(
            candidate.get(
                "pose_direct_live_distillation_weight",
                candidate.get(
                    "snerv_pose_direct_live_distillation_weight",
                    pose_direct_live_distillation_weight,
                ),
            )
        ),
        pose_distillation_loss=str(
            candidate.get(
                "pose_distillation_loss",
                candidate.get("snerv_pose_distillation_loss", pose_distillation_loss),
            )
        ),
        pose_distillation_huber_delta=float(
            candidate.get(
                "pose_distillation_huber_delta",
                candidate.get(
                    "snerv_pose_distillation_huber_delta",
                    pose_distillation_huber_delta,
                ),
            )
        ),
        segnet_distillation_objective=str(
            candidate.get(
                "segnet_distillation_objective",
                candidate.get(
                    "snerv_segnet_distillation_objective",
                    segnet_distillation_objective,
                ),
            )
        ),
        distillation_temperature=float(
            candidate.get(
                "distillation_temperature",
                candidate.get("snerv_distillation_temperature", distillation_temperature),
            )
        ),
        segnet_student_live_calibration_weight=float(
            candidate.get(
                "segnet_student_live_calibration_weight",
                candidate.get(
                    "snerv_segnet_student_live_calibration_weight",
                    segnet_student_live_calibration_weight,
                ),
            )
        ),
        segnet_direct_live_distillation_weight=float(
            candidate.get(
                "segnet_direct_live_distillation_weight",
                candidate.get(
                    "snerv_segnet_direct_live_distillation_weight",
                    segnet_direct_live_distillation_weight,
                ),
            )
        ),
        segnet_direct_live_base_loss_weight=float(
            candidate.get(
                "segnet_direct_live_base_loss_weight",
                candidate.get(
                    "snerv_segnet_direct_live_base_loss_weight",
                    segnet_direct_live_base_loss_weight,
                ),
            )
        ),
        segnet_direct_live_class_histogram_weight=float(
            candidate.get(
                "segnet_direct_live_class_histogram_weight",
                candidate.get(
                    "snerv_segnet_direct_live_class_histogram_weight",
                    segnet_direct_live_class_histogram_weight,
                ),
            )
        ),
        segnet_direct_live_class_balanced_hinge_weight=float(
            candidate.get(
                "segnet_direct_live_class_balanced_hinge_weight",
                candidate.get(
                    "snerv_segnet_direct_live_class_balanced_hinge_weight",
                    segnet_direct_live_class_balanced_hinge_weight,
                ),
            )
        ),
        segnet_direct_live_class_balanced_ce_weight=float(
            candidate.get(
                "segnet_direct_live_class_balanced_ce_weight",
                candidate.get(
                    "snerv_segnet_direct_live_class_balanced_ce_weight",
                    segnet_direct_live_class_balanced_ce_weight,
                ),
            )
        ),
        segnet_direct_live_class_balanced_squared_hinge_weight=float(
            candidate.get(
                "segnet_direct_live_class_balanced_squared_hinge_weight",
                candidate.get(
                    "snerv_segnet_direct_live_class_balanced_squared_hinge_weight",
                    segnet_direct_live_class_balanced_squared_hinge_weight,
                ),
            )
        ),
        segnet_direct_live_class_region_recon_weight=float(
            candidate.get(
                "segnet_direct_live_class_region_recon_weight",
                candidate.get(
                    "snerv_segnet_direct_live_class_region_recon_weight",
                    segnet_direct_live_class_region_recon_weight,
                ),
            )
        ),
        segnet_direct_live_rare_class_logit_weight=float(
            candidate.get(
                "segnet_direct_live_rare_class_logit_weight",
                candidate.get(
                    "snerv_segnet_direct_live_rare_class_logit_weight",
                    segnet_direct_live_rare_class_logit_weight,
                ),
            )
        ),
        segnet_direct_live_target_mass_floor_weight=float(
            candidate.get(
                "segnet_direct_live_target_mass_floor_weight",
                candidate.get(
                    "snerv_segnet_direct_live_target_mass_floor_weight",
                    segnet_direct_live_target_mass_floor_weight,
                ),
            )
        ),
        segnet_direct_live_target_min_ratio_floor_weight=float(
            candidate.get(
                "segnet_direct_live_target_min_ratio_floor_weight",
                candidate.get(
                    "snerv_segnet_direct_live_target_min_ratio_floor_weight",
                    segnet_direct_live_target_min_ratio_floor_weight,
                ),
            )
        ),
        segnet_tau_boundary=float(
            candidate.get(
                "segnet_tau_boundary",
                candidate.get("snerv_segnet_tau_boundary", segnet_tau_boundary),
            )
        ),
        segnet_hinge_margin=float(
            candidate.get(
                "segnet_hinge_margin",
                candidate.get("snerv_segnet_hinge_margin", segnet_hinge_margin),
            )
        ),
        distillation_device=str(
            candidate.get(
                "distillation_device",
                candidate.get("snerv_distillation_device", distillation_device),
            )
        ),
        allow_segnet_only_research=bool(
            candidate.get(
                "allow_segnet_only_research",
                candidate.get(
                    "snerv_allow_segnet_only_research",
                    allow_segnet_only_research,
                ),
            )
        ),
        coder_aware_qat=bool(
            candidate.get(
                "coder_aware_qat",
                candidate.get("snerv_coder_aware_qat", coder_aware_qat),
            )
        ),
        coder_qat_quant_bits=int(
            candidate.get(
                "coder_qat_quant_bits",
                candidate.get("snerv_coder_qat_quant_bits", coder_qat_quant_bits),
            )
        ),
        coder_qat_quant_residual_weight=float(
            candidate.get(
                "coder_qat_quant_residual_weight",
                candidate.get(
                    "snerv_coder_qat_quant_residual_weight",
                    coder_qat_quant_residual_weight,
                ),
            )
        ),
        coder_qat_magnitude_weight=float(
            candidate.get(
                "coder_qat_magnitude_weight",
                candidate.get(
                    "snerv_coder_qat_magnitude_weight",
                    coder_qat_magnitude_weight,
                ),
            )
        ),
        coder_qat_delta_weight=float(
            candidate.get(
                "coder_qat_delta_weight",
                candidate.get("snerv_coder_qat_delta_weight", coder_qat_delta_weight),
            )
        ),
        coder_qat_c1a_entropy_weight=float(
            candidate.get(
                "coder_qat_c1a_entropy_weight",
                candidate.get(
                    "snerv_coder_qat_c1a_entropy_weight",
                    coder_qat_c1a_entropy_weight,
                ),
            )
        ),
        coder_qat_c1a_sigma=float(
            candidate.get(
                "coder_qat_c1a_sigma",
                candidate.get("snerv_coder_qat_c1a_sigma", coder_qat_c1a_sigma),
            )
        ),
        coder_qat_c1a_sample_size=int(
            candidate.get(
                "coder_qat_c1a_sample_size",
                candidate.get(
                    "snerv_coder_qat_c1a_sample_size",
                    coder_qat_c1a_sample_size,
                ),
            )
        ),
        pr95_faithful_curriculum_enabled=bool(
            candidate.get(
                "score_aware_long_training_pr95_faithful_curriculum",
                candidate.get(
                    "snerv_score_aware_long_training_pr95_faithful_curriculum",
                    score_aware_long_training_pr95_faithful_curriculum,
                ),
            )
        ),
        pr95_muon_policy=str(
            candidate.get(
                "score_aware_long_training_pr95_muon_policy",
                candidate.get(
                    "snerv_score_aware_long_training_pr95_muon_policy",
                    score_aware_long_training_pr95_muon_policy,
                ),
            )
        ),
        pr95_stage_source_weight_amplification_enabled=bool(
            candidate.get(
                "score_aware_long_training_pr95_source_weight_amplification",
                candidate.get(
                    "snerv_score_aware_long_training_pr95_source_weight_amplification",
                    score_aware_long_training_pr95_source_weight_amplification,
                ),
            )
        ),
        gradient_multiplier_by_name=_coerce_gradient_multiplier_by_name(
            _candidate_first_non_null(
                candidate,
                (
                    "score_aware_long_training_gradient_multiplier_by_name",
                    "snerv_score_aware_long_training_gradient_multiplier_by_name",
                ),
                score_aware_long_training_gradient_multiplier_by_name,
            )
        ),
        bias_gradient_multiplier=_candidate_first_non_null(
            candidate,
            (
                "score_aware_long_training_bias_gradient_multiplier",
                "snerv_score_aware_long_training_bias_gradient_multiplier",
            ),
            score_aware_long_training_bias_gradient_multiplier,
        ),
        output_head_bias_gradient_multiplier=float(
            _candidate_first_non_null(
                candidate,
                (
                    "score_aware_long_training_output_head_bias_gradient_multiplier",
                    "snerv_score_aware_long_training_output_head_bias_gradient_multiplier",
                ),
                score_aware_long_training_output_head_bias_gradient_multiplier,
            )
        ),
        official_trained_checkpoint_mapping_manifest=(
            official_trained_checkpoint_mapping_manifest
        ),
        prioritized_pair_indices=priority_pair_indices,
        scorer_error_pair_sampling_weights=scorer_error_pair_sampling_weights,
        scorer_error_pair_curriculum=scorer_error_pair_curriculum,
        allow_overwrite=allow_overwrite,
    )
    trained_pairs_candidate = score_aware_long_training.get("_trained_pairs_nchw255")
    trained_state_exportable = bool(
        isinstance(trained_pairs_candidate, np.ndarray)
        and score_aware_long_training.get("trained_state_exportable") is True
    )
    pairs_for_packet = (
        np.asarray(trained_pairs_candidate, dtype=np.float32)
        if trained_state_exportable
        else pairs_nchw255
    )
    trained_official_packet = score_aware_long_training.get("_trained_official_packet")
    score_aware_long_training_public = {
        key: value
        for key, value in score_aware_long_training.items()
        if key not in {"_trained_pairs_nchw255", "_trained_official_packet"}
    }
    packet_metadata_extra = {
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
        "score_aware_long_training": score_aware_long_training_public,
        "score_aware_long_training_executed": bool(
            score_aware_long_training_public.get("executed") is True
        ),
        "score_aware_long_training_trained_state_exportable": (
            trained_state_exportable
        ),
        "score_aware_long_training_kind": str(
            score_aware_long_training_public.get("training_kind") or "none"
        ),
        "score_aware_long_training_optimizer": str(
            score_aware_long_training_public.get("optimizer_kind") or "none"
        ),
        "score_aware_long_training_scorer_input_distribution_guard_bound": bool(
            score_aware_long_training_public.get(
                "scorer_input_distribution_guard_bound"
            )
            is True
        ),
        "hard_byte_ceiling": hard_byte_ceiling,
        **(
            {
                "native_mlx_training_executed": True,
                "native_mlx_training_kind": str(
                    score_aware_long_training_public.get("training_kind")
                    or "snerv_mlx_score_aware_haar_renderer"
                ),
            }
            if trained_state_exportable
            else {}
        ),
        **(
            _official_primitives_packet_metadata(
                official_binding,
                blockers=official_primitives_blockers,
            )
            if official_binding is not None
            else {}
        ),
    }
    if isinstance(trained_official_packet, SnervArchivePacket):
        closed_form_archive = trained_official_packet
    else:
        closed_form_archive = build_snerv_mlx_native_packet_from_numpy_pairs(
            pairs_for_packet,
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
            native_mlx_decoder_train_optimizer=str(
                candidate.get(
                    "native_mlx_decoder_train_optimizer",
                    candidate.get("snerv_native_mlx_decoder_train_optimizer", native_mlx_decoder_train_optimizer),
                )
            ),
            metadata_extra=packet_metadata_extra,
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
        pair_guard_min_score_improved_fraction=float(
            scorer_loop_qat_pair_guard_min_score_improved_fraction
        ),
        pair_guard_max_pose_worsened_fraction=float(
            scorer_loop_qat_pair_guard_max_pose_worsened_fraction
        ),
        device=str(scorer_loop_qat_device),
        perturb_scale=float(scorer_loop_qat_perturb_scale),
        byte_pressure_multiplier=float(scorer_loop_qat_byte_pressure_multiplier),
        section_value_pressure_multiplier=float(
            scorer_loop_qat_section_value_pressure_multiplier
        ),
        max_archive_byte_growth=scorer_loop_qat_max_archive_byte_growth,
        byte_growth_admission_mode=str(scorer_loop_qat_byte_growth_admission_mode),
        pose_slack=float(scorer_loop_qat_pose_slack),
        seg_slack=float(scorer_loop_qat_seg_slack),
        seed=int(scorer_loop_qat_seed),
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
        and scorer_loop_qat.get("ready_for_pose_guard_gate") is True
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
    best_packet_preserves_official_decoder_payload = (
        _packet_preserves_official_decoder_payload(
            best_packet if isinstance(best_packet, bytes) else b"",
        )
        if official_primitives_requested
        else True
    )
    official_tub_output2_binding_report = (
        _official_tub_output2_binding_preservation_report(
            best_packet if isinstance(best_packet, bytes) else b"",
            base_packet=closed_form_archive.packet,
        )
        if official_primitives_requested
        else {
            "schema": "snerv_official_tub_output2_binding_preservation.v1",
            "preserved": True,
            "mismatched_fields": [],
            **FALSE_AUTHORITY,
        }
    )
    best_packet_preserves_official_tub_output2_binding = bool(
        official_tub_output2_binding_report.get("preserved") is True
    )
    if official_primitives_requested:
        scorer_loop_qat_public["official_decoder_payload_binding_required"] = True
        scorer_loop_qat_public["official_decoder_payload_binding_preserved"] = bool(
            best_packet_preserves_official_decoder_payload
        )
        scorer_loop_qat_public["official_tub_output2_binding_required"] = True
        scorer_loop_qat_public["official_tub_output2_binding_preserved"] = bool(
            best_packet_preserves_official_tub_output2_binding
        )
        scorer_loop_qat_public["official_tub_output2_binding_report"] = (
            official_tub_output2_binding_report
        )
    if (
        best_packet_ready
        and best_packet_preserves_recon_weight
        and best_packet_preserves_source_pairs
        and best_packet_preserves_official_decoder_payload
        and best_packet_preserves_official_tub_output2_binding
    ):
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
    elif best_packet_ready and (
        recon_weight_metadata is not None
            or explicit_pair_indices
            or official_primitives_requested
    ):
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
        if official_primitives_requested and not best_packet_preserves_official_decoder_payload:
            scorer_loop_qat_public["official_decoder_payload_binding_required"] = True
            scorer_loop_qat_public["official_decoder_payload_binding_preserved"] = False
            extra_blockers.append("snerv_scorer_loop_qat_best_packet_rejected_official_payload_mismatch")
        if official_primitives_requested and not best_packet_preserves_official_tub_output2_binding:
            scorer_loop_qat_public["official_tub_output2_binding_required"] = True
            scorer_loop_qat_public["official_tub_output2_binding_preserved"] = False
            scorer_loop_qat_public["official_tub_output2_binding_report"] = (
                official_tub_output2_binding_report
            )
            extra_blockers.extend(
                str(blocker)
                for blocker in official_tub_output2_binding_report.get("blockers") or ()
                if str(blocker)
            )
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

    selected_metadata_continuity: dict[str, Any] = {
        "schema": "snerv_selected_packet_metadata_continuity.v1",
        "selected_packet_source": str(selected_packet_source),
        "inherited_field_count": 0,
        "inherited_fields": [],
        "metadata_only_repack": False,
        **FALSE_AUTHORITY,
    }
    selected_packet_report_metadata: dict[str, Any] | None = None
    selected_metadata_continuity_blockers: list[str] = []
    if selected_packet_source == "scorer_loop_qat_best_receiver_packet":
        try:
            (
                selected_packet,
                selected_packet_report_metadata,
                selected_metadata_continuity,
            ) = _repack_selected_packet_with_section_metadata_continuity(
                selected_packet,
                base_packet=closed_form_archive.packet,
                selected_packet_source=selected_packet_source,
            )
        except Exception as exc:
            selected_metadata_continuity = {
                **selected_metadata_continuity,
                "failure": f"{type(exc).__name__}: {exc}",
                "blockers": [
                    "snerv_selected_packet_metadata_continuity_failed"
                ],
                **FALSE_AUTHORITY,
            }
            selected_metadata_continuity_blockers.append(
                "snerv_selected_packet_metadata_continuity_failed"
            )
        scorer_loop_qat_public["selected_packet_metadata_continuity"] = (
            selected_metadata_continuity
        )
        scorer_loop_qat_public["emitted_packet_bytes"] = len(selected_packet)
        scorer_loop_qat_public["emitted_packet_sha256"] = _sha256_bytes(
            selected_packet
        )

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
    blockers.extend(selected_metadata_continuity_blockers)
    if official_primitives_requested:
        blockers.extend(official_primitives_blockers)
    if scorer_custody.get("contract_valid") is not True:
        blockers.append("snerv_mlx_native_scorer_custody_contract_invalid")
    if recon_weight_metadata is not None and recon_weight_metadata.get("producer_manifest_verified") is not True:
        blockers.append("snerv_recon_pixel_weight_verified_gradient_manifest_not_bound_to_native_export")
    if score_aware_long_training_public.get("executed") is True:
        blockers = [
            blocker
            for blocker in blockers
            if blocker != "snerv_mlx_score_aware_long_training_not_executed"
        ]
    long_training_joint_real_teachers_bound = bool(
        score_aware_long_training_public.get("has_real_segnet_teacher") is True
        and score_aware_long_training_public.get("has_real_posenet_teacher") is True
    )
    blockers.extend(
        str(blocker)
        for blocker in score_aware_long_training_public.get("blockers") or ()
        if str(blocker)
    )
    long_training_official_replay = score_aware_long_training_public.get(
        "official_mfu_hfr_tub_source_forward_replay"
    )
    official_tub_source_fixture_binding = (
        _official_tub_source_fixture_binding(long_training_official_replay)
        if isinstance(long_training_official_replay, Mapping)
        else _official_tub_source_fixture_binding(None)
    )
    if isinstance(long_training_official_replay, Mapping):
        if (
            official_tub_source_fixture_binding.get(
                "official_tub_temporal_encoder_output2_source_fixture_replay_passed"
            )
            is True
        ):
            blockers = [
                str(blocker)
                for blocker in blockers
                if str(blocker)
                != "snerv_official_mfu_hfr_tub_source_forward_replay_missing"
            ]
            official_primitives_blockers = [
                str(blocker)
                for blocker in official_primitives_blockers
                if str(blocker)
                != "snerv_official_mfu_hfr_tub_source_forward_replay_missing"
            ]
        blockers.extend(
            str(blocker)
            for blocker in long_training_official_replay.get("blockers") or ()
            if str(blocker)
        )
    blockers.extend(
        _scorer_loop_qat_blockers(
            scorer_loop_qat_public,
            num_pairs=effective_num_pairs,
        )
    )
    if long_training_joint_real_teachers_bound:
        blockers = [
            str(blocker)
            for blocker in blockers
            if str(blocker) != "snerv_real_segnet_posenet_teacher_loop_not_attached"
        ]
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
                allow_overwrite=allow_overwrite,
            )
    receiver_proof = dict(package.get("receiver_proof") or {}) if package else {}
    selected_packet_sha256 = _sha256_bytes(selected_packet)
    selected_archive = unpack_snerv_archive(selected_packet)
    selected_packet_schema = selected_archive.schema
    selected_packet_wire_format = _snerv_packet_wire_format_from_schema(
        selected_archive.schema
    )
    selected_packet_contest_submission_wire_format_ready = (
        selected_packet_wire_format == "snar2"
    )
    if scorer_loop_qat_public.get("emitted_packet_uses_scorer_loop_best_decoder") is True:
        scorer_loop_qat_public["emitted_packet_schema"] = selected_packet_schema
        scorer_loop_qat_public["emitted_packet_wire_format"] = selected_packet_wire_format
        scorer_loop_qat_public["emitted_packet_contest_submission_wire_format_ready"] = (
            bool(selected_packet_contest_submission_wire_format_ready)
        )
        report = scorer_loop_qat_public.get("report_path")
        if report:
            _payload_for_disk = {
                key: value
                for key, value in scorer_loop_qat_public.items()
                if key != "report_path"
            }
            write_json(report, _payload_for_disk)
    selected_archive_metadata = _selected_archive_metadata_for_report(
        decoded_metadata=selected_archive.metadata,
        selected_packet_sha256=selected_packet_sha256,
        selected_packet_source=selected_packet_source,
        closed_form_archive=closed_form_archive,
    )
    if selected_packet_report_metadata is not None:
        selected_archive_metadata = {
            **selected_archive_metadata,
            **dict(selected_packet_report_metadata),
            "receiver_packet_metadata": dict(selected_archive.metadata),
            "receiver_packet_report_metadata_source": (
                "snar2_compact_wire_metadata_plus_selected_packet_report_metadata"
            ),
            "receiver_packet_report_metadata_source_packet": str(
                selected_packet_source
            ),
            **FALSE_AUTHORITY,
        }
    selected_section_bytes = {
        str(name): len(blob) for name, blob in selected_archive.sections.items()
    }
    selected_decoder_payload_codec = str(
        selected_archive_metadata.get("decoder_payload_codec")
        or active_decoder_payload_codec
    )
    selected_lf_payload_codec = str(
        selected_archive_metadata.get("lf_payload_codec_selected")
        or selected_archive_metadata.get("lf_payload_codec")
        or active_lf_payload_codec
    )
    receiver_target_profile = _snerv_receiver_frame_reconstruction_profile(
        selected_packet,
        reference_pairs_nchw255=pairs_nchw255,
        source_pair_indices=source_pair_indices,
        profile_id="selected_packet_vs_source_targets",
        reference_kind="source_targets_nchw255",
        packet_source=selected_packet_source,
    )
    receiver_export_profile = _snerv_receiver_frame_reconstruction_profile(
        selected_packet,
        reference_pairs_nchw255=pairs_for_packet,
        source_pair_indices=source_pair_indices,
        profile_id="selected_packet_vs_export_reference",
        reference_kind=(
            "score_aware_long_training_selected_pairs_nchw255"
            if score_aware_long_training_public.get("executed") is True
            else "source_targets_nchw255"
        ),
        packet_source=selected_packet_source,
    )
    blockers.extend(
        str(blocker)
        for profile in (receiver_target_profile, receiver_export_profile)
        for blocker in profile.get("blockers") or ()
        if str(blocker)
    )
    official_skip_high_value_domain = _snerv_official_skip_high_value_domain_gate(
        selected_archive_metadata,
        receiver_target_profile=receiver_target_profile,
        receiver_export_profile=receiver_export_profile,
    )
    blockers.extend(
        str(blocker)
        for blocker in official_skip_high_value_domain.get("blockers") or ()
        if str(blocker)
    )
    byte_cap_control = _build_snerv_mlx_native_byte_cap_control(
        candidate=candidate,
        hard_byte_ceiling=hard_byte_ceiling,
        packet_source=selected_packet_source,
        packet_sha256=selected_packet_sha256,
        packet_bytes=len(selected_packet),
        section_bytes=selected_section_bytes,
        decoder_payload_codec=selected_decoder_payload_codec,
        lf_payload_codec=selected_lf_payload_codec,
        official_receiver_tensor_map=(
            _official_receiver_tensor_map_from_packet(selected_packet)
            if official_primitives_requested
            else None
        ),
        archive_bytes=(
            int(receiver_proof["archive_bytes"])
            if receiver_proof.get("archive_bytes") is not None
            else None
        ),
        archive_sha256=(
            str(receiver_proof.get("archive_sha256"))
            if receiver_proof.get("archive_sha256")
            else None
        ),
        receiver_proof_passed=(
            receiver_proof.get("runtime_consumption_proof_passed") is True
        ),
        receiver_contract_satisfied=(
            receiver_proof.get("receiver_contract_satisfied") is True
        ),
        run_archive_export=bool(run_archive_export),
    )
    blockers.extend(str(blocker) for blocker in byte_cap_control.get("blockers") or ())
    mlx_prefilter_profile = _write_snerv_native_receiver_decoded_mlx_prefilter(
        requested=bool(write_mlx_prefilter_profile),
        output_dir=out / "local_mlx_prefilter",
        selected_packet=selected_packet,
        target0_np=target0_np,
        target1_np=target1_np,
        archive_bytes=(
            int(receiver_proof["archive_bytes"])
            if receiver_proof.get("archive_bytes") is not None
            else None
        ),
        archive_sha256=(
            str(receiver_proof.get("archive_sha256"))
            if receiver_proof.get("archive_sha256")
            else None
        ),
        source_video_path=source_video_path,
        scorer_upstream_dir=scorer_upstream_dir,
        scorer_device=str(mlx_prefilter_scorer_device),
        scorer_batch_pairs=int(mlx_prefilter_scorer_batch_pairs),
        progress_every=int(mlx_prefilter_progress_every),
        allow_overwrite=allow_overwrite,
    )
    selected_official_authority = (
        _selected_packet_official_payload_authority(selected_packet)
        if official_primitives_requested
        else None
    )
    selected_official_tensor_map = (
        _official_receiver_tensor_map_from_packet(selected_packet)
        if official_primitives_requested
        else None
    )
    if (
        official_primitives_requested
        and selected_official_authority is not None
        and selected_official_authority.get("frame_producing_official_export") is True
    ):
        official_primitives_blockers = [
            str(blocker)
            for blocker in official_primitives_blockers
            if str(blocker)
            != "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
        ]
        blockers = [
            str(blocker)
            for blocker in blockers
            if str(blocker)
            != "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
        ]
        if (
            selected_official_tensor_map is not None
            and selected_official_tensor_map.get("receiver_tensor_map_verified")
            is True
        ):
            weight_mapping_blockers = (
                []
                if official_checkpoint_mapping_verified
                else [SNERV_OFFICIAL_MFU_HFR_TUB_WEIGHT_MAPPING_BLOCKER]
            )
            selected_official_tensor_map = {
                **dict(selected_official_tensor_map),
                "official_state_dict_mapping_verified": (
                    official_checkpoint_mapping_verified
                ),
                "official_weight_mapping_blocker_closed": (
                    official_checkpoint_mapping_verified
                ),
                "official_weight_mapping_scope": (
                    "receiver_payload_tensor_hashes_plus_upstream_state_dict_mapping"
                    if official_checkpoint_mapping_verified
                    else "receiver_payload_tensor_hashes_only_not_upstream_state_dict_mapping"
                ),
                "official_weight_mapping_blockers": weight_mapping_blockers,
                "official_trained_checkpoint_mapping_manifest": (
                    official_trained_checkpoint_mapping_manifest
                ),
            }
        if official_binding is not None:
            official_binding = dict(official_binding)
            official_binding["blockers"] = list(official_primitives_blockers)
    native_training_export_guard = build_snerv_mlx_native_training_export_guard(
        {
            "native_mlx_training_executed": selected_archive_metadata.get(
                "native_mlx_training_executed"
            ),
            "native_mlx_hf_decoder_training": selected_archive_metadata.get(
                "native_mlx_hf_decoder_training"
            ),
            "packet_source": selected_packet_source,
        }
    )
    blockers.extend(
        str(blocker)
        for blocker in native_training_export_guard.get("blockers") or ()
    )

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
        step_map_waterfill_bits_per_coeff=(
            float(selected_archive_metadata["step_map_waterfill_bits_per_coeff"])
            if selected_archive_metadata.get("step_map_waterfill_bits_per_coeff")
            is not None
            else None
        ),
        step_map_coder_groups=tuple(
            dict(group) for group in selected_archive_metadata.get("step_map_coder_groups") or ()
        ),
        decoder_payload_codec=selected_decoder_payload_codec,
        lf_payload_codec=selected_lf_payload_codec,
        model_size=model_size.as_jsonable(),
        bridge_drift=bridge,
        receiver_target_reconstruction_profile=receiver_target_profile,
        receiver_export_reconstruction_profile=receiver_export_profile,
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
    payload["packet_schema"] = selected_packet_schema
    payload["packet_wire_format"] = selected_packet_wire_format
    payload["packet_contest_submission_wire_format_ready"] = (
        bool(selected_packet_contest_submission_wire_format_ready)
    )
    payload["hard_byte_ceiling"] = hard_byte_ceiling
    payload["byte_cap_control"] = byte_cap_control
    payload["official_skip_high_value_domain_gate"] = official_skip_high_value_domain
    payload["local_mlx_prefilter_profile"] = mlx_prefilter_profile
    payload["local_mlx_prefilter_profile_path"] = mlx_prefilter_profile.get(
        "profile_path"
    )
    payload["local_mlx_prefilter_progress_path"] = mlx_prefilter_profile.get(
        "progress_path"
    )
    payload["executed"] = True
    payload["packet_source"] = selected_packet_source
    for key, value in _selected_metadata_report_fields(
        selected_archive_metadata
    ).items():
        payload.setdefault(key, value)
    selected_recon_metadata = selected_archive_metadata.get("recon_pixel_weight_metadata")
    selected_recon_consumed = selected_archive_metadata.get("recon_pixel_weight_consumed") is True and isinstance(
        selected_recon_metadata, Mapping
    )
    payload["score_aware_hf_decoder_fit_executed"] = bool(selected_recon_consumed)
    payload["score_aware_long_training"] = score_aware_long_training_public
    payload["score_aware_long_training_executed"] = bool(
        selected_archive_metadata.get("score_aware_long_training_executed") is True
    )
    payload["score_aware_long_training_trained_state_exportable"] = bool(
        selected_archive_metadata.get(
            "score_aware_long_training_trained_state_exportable"
        )
        is True
    )
    payload["score_aware_long_training_real_teachers_bound"] = (
        long_training_joint_real_teachers_bound
    )
    payload["score_aware_long_training_has_real_segnet_teacher"] = bool(
        score_aware_long_training_public.get("has_real_segnet_teacher") is True
    )
    payload["score_aware_long_training_has_real_posenet_teacher"] = bool(
        score_aware_long_training_public.get("has_real_posenet_teacher") is True
    )
    payload["score_aware_long_training_coder_qat_bound"] = bool(
        score_aware_long_training_public.get("coder_aware_qat_bound") is True
    )
    payload["score_aware_long_training_scorer_input_distribution_guard_bound"] = bool(
        score_aware_long_training_public.get(
            "scorer_input_distribution_guard_bound"
        )
        is True
    )
    payload["score_aware_long_training_pr95_curriculum_bound"] = bool(
        score_aware_long_training_public.get("pr95_faithful_curriculum_enabled")
        is True
    )
    payload["score_aware_long_training_pr95_muon_policy"] = str(
        score_aware_long_training_public.get("pr95_muon_policy")
        or "every_stage"
    )
    payload["score_aware_long_training_pr95_source_weight_amplification_bound"] = bool(
        score_aware_long_training_public.get(
            "pr95_stage_source_weight_amplification_enabled"
        )
        is True
    )
    payload["score_aware_long_training_pr95_faithful_optimizer_schedule_bound"] = bool(
        score_aware_long_training_public.get("pr95_faithful_curriculum_enabled")
        is True
    )
    payload["score_aware_long_training_muon_adamw_partition_bound"] = bool(
        score_aware_long_training_public.get("muon_adamw_partition_bound") is True
    )
    payload["score_aware_long_training_pact_native_muon_adamw_partition_bound"] = bool(
        score_aware_long_training_public.get("pact_native_muon_adamw_partition_bound")
        is True
    )
    payload["score_aware_long_training_kind"] = str(
        selected_archive_metadata.get("score_aware_long_training_kind") or "none"
    )
    payload["native_mlx_training_executed"] = bool(
        selected_archive_metadata.get("native_mlx_training_executed") is True
    )
    payload["native_mlx_training_kind"] = str(selected_archive_metadata.get("native_mlx_training_kind") or "none")
    payload["native_mlx_hf_decoder_training"] = dict(
        selected_archive_metadata.get("native_mlx_hf_decoder_training") or {}
    )
    payload["native_mlx_training_export_guard"] = native_training_export_guard
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
        selected_tub_source_fixture_binding = (
            _official_tub_source_fixture_binding_from_metadata(
                selected_archive_metadata
            )
        )
        payload["official_primitive_binding"] = _receiver_bound_official_primitives_export_binding(
            official_binding,
            packet_path=packet_path,
            packet_bytes=len(selected_packet),
            packet_sha256=_sha256_bytes(selected_packet),
            selected_packet=selected_packet,
            selected_archive_metadata=selected_archive_metadata,
            package=package,
            receiver_proof=receiver_proof,
            official_trained_checkpoint_mapping_manifest=(
                official_trained_checkpoint_mapping_manifest
            ),
        )
        payload["snerv_official_mfu_hfr_tub_numeric_primitives_requested"] = True
        payload["snerv_official_mfu_hfr_tub_export_bound"] = bool(
            selected_archive_metadata.get("snerv_official_mfu_hfr_tub_export_bound")
            is True
        )
        payload["snerv_official_mfu_hfr_tub_export_bound_semantics"] = str(
            selected_archive_metadata.get(
                "snerv_official_mfu_hfr_tub_export_bound_semantics"
            )
            or "receiver_payload_bound_not_source_forward_parity"
        )
        payload["snerv_official_mfu_hfr_tub_receiver_payload_bound"] = bool(
            selected_archive_metadata.get(
                "snerv_official_mfu_hfr_tub_receiver_payload_bound"
            )
            is True
        )
        payload["snerv_official_mfu_hfr_tub_source_forward_replay_bound"] = bool(
            selected_archive_metadata.get(
                "snerv_official_mfu_hfr_tub_source_forward_replay_bound"
            )
            is True
        )
        payload["snerv_official_mfu_hfr_tub_source_forward_replay_authority"] = bool(
            selected_archive_metadata.get(
                "snerv_official_mfu_hfr_tub_source_forward_replay_authority"
            )
            is True
        )
        payload["snerv_official_tub_source_fixture_binding"] = (
            selected_tub_source_fixture_binding
        )
        payload["snerv_official_tub_source_fixture_replay_bound"] = bool(
            _official_tub_source_fixture_replay_bound(
                selected_tub_source_fixture_binding
            )
        )
        payload["snerv_official_tub_source_fixture_replay_passed"] = bool(
            selected_tub_source_fixture_binding.get(
                "official_tub_temporal_encoder_output2_source_fixture_replay_passed"
            )
            is True
        )
        payload["snerv_official_tub_source_forward_fixture_bound"] = bool(
            selected_archive_metadata.get(
                "snerv_official_tub_source_forward_fixture_bound"
            )
            is True
        )
        payload["official_source_parity_blockers"] = [
            str(blocker)
            for blocker in selected_archive_metadata.get(
                "official_source_parity_blockers"
            )
            or _official_packet_source_parity_blockers(
                selected_tub_source_fixture_binding
            )
        ]
        payload["snerv_official_mfu_hfr_tub_receiver_bound_surrogate_export"] = not bool(
            payload["snerv_official_mfu_hfr_tub_export_bound"]
        )
        payload["snerv_official_mfu_hfr_tub_frame_producing_export"] = bool(
            selected_archive_metadata.get(
                "snerv_official_mfu_hfr_tub_frame_producing_export"
            )
            is True
        )
        payload["snerv_official_mfu_hfr_tub_export_blockers"] = list(official_primitives_blockers)
    payload["wall_seconds"] = round(time.monotonic() - started, 6)
    report_path = out / SNERV_MLX_NATIVE_REPORT_FILENAME
    if report_path.exists() and not allow_overwrite:
        raise SnervMlxNativeExportError(f"refusing to overwrite existing artifact: {report_path}")
    payload["report_path"] = report_path.as_posix()
    write_json(report_path, payload)
    return payload


def _write_snerv_native_receiver_decoded_mlx_prefilter(
    *,
    requested: bool,
    output_dir: str | Path,
    selected_packet: bytes,
    target0_np: np.ndarray,
    target1_np: np.ndarray,
    archive_bytes: int | None,
    archive_sha256: str | None,
    source_video_path: str | Path,
    scorer_upstream_dir: str | Path,
    scorer_device: str,
    scorer_batch_pairs: int,
    progress_every: int,
    allow_overwrite: bool,
) -> dict[str, Any]:
    """Write a false-authority MLX scorer prefilter for the selected SNAR1 bytes.

    The profile is intentionally receiver-decoded, not trainer-tensor decoded:
    ``selected_packet`` is first unpacked through the SNeRV receiver path, then
    scored against the hydrated target frames. This makes the local replay
    unlock depend on the exact packet bytes that inflate.sh will consume.
    """

    out = Path(output_dir).expanduser().resolve(strict=False)
    profile_path = out / "local_mlx_prefilter_profile.json"
    progress_path = out / "local_mlx_prefilter_progress.jsonl"
    base = {
        "schema": SNERV_MLX_NATIVE_PREFILTER_PROFILE_SCHEMA,
        "requested": bool(requested),
        "profile_path": profile_path.as_posix(),
        "progress_path": progress_path.as_posix(),
        "receiver_decoded_selected_packet": True,
        "archive_bytes": int(archive_bytes) if archive_bytes is not None else None,
        "archive_sha256": archive_sha256,
        "scorer_device": str(scorer_device),
        "scorer_batch_pairs": int(scorer_batch_pairs),
        "progress_every": int(progress_every),
        **FALSE_AUTHORITY,
    }
    if not requested:
        return {
            **base,
            "written": False,
            "blockers": ["snerv_native_mlx_prefilter_not_requested"],
        }
    out.mkdir(parents=True, exist_ok=True)
    if profile_path.exists() and not allow_overwrite:
        raise SnervMlxNativeExportError(
            f"refusing to overwrite existing MLX prefilter profile: {profile_path}"
        )
    try:
        if archive_bytes is None or archive_sha256 is None:
            raise SnervMlxNativeExportError(
                "archive_bytes/archive_sha256 missing; cannot build SNeRV MLX prefilter"
            )
        import mlx.core as mx

        from tac.local_acceleration.mlx_renderer_prefilter_profile import (
            write_mlx_renderer_prefilter_profile,
        )
        from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle

        decoded_pairs = decode_snerv_archive_frames(selected_packet).astype(
            np.float32,
            copy=False,
        )
        if decoded_pairs.shape[0] != int(target0_np.shape[0]):
            raise SnervMlxNativeExportError(
                "receiver-decoded pair count does not match target pair count: "
                f"{decoded_pairs.shape[0]} != {target0_np.shape[0]}"
            )

        class _ReceiverDecodedPacketModel:
            def __init__(self, pairs_b2chw255: np.ndarray) -> None:
                self._pairs = mx.array(pairs_b2chw255, dtype=mx.float32)

            def __call__(self, idx: Any) -> Any:
                return mx.take(self._pairs, idx, axis=0)

        bundle = RendererBundle(
            model=_ReceiverDecodedPacketModel(decoded_pairs),
            target_rgb_0=mx.array(target0_np, dtype=mx.float32),
            target_rgb_1=mx.array(target1_np, dtype=mx.float32),
            num_pairs=int(decoded_pairs.shape[0]),
            forward_convention="call_b2chw_255",
            substrate_artifact_metadata={
                "schema": "snerv_receiver_decoded_packet_prefilter_bundle.v1",
                "receiver_decoded_selected_packet": True,
                "packet_sha256": _sha256_bytes(selected_packet),
                "archive_sha256": str(archive_sha256),
                "human_visual_fidelity_objective": False,
                "contest_scorer_prefilter_only": True,
            },
        )
        profile = write_mlx_renderer_prefilter_profile(
            bundle=bundle,
            output_path=profile_path,
            archive_bytes=int(archive_bytes),
            archive_sha256=str(archive_sha256),
            upstream_dir=scorer_upstream_dir,
            scorer_device=str(scorer_device),
            scorer_batch_pairs=int(scorer_batch_pairs),
            run_id="snerv_native_receiver_decoded_packet_prefilter",
            source_video_path=source_video_path,
            progress_jsonl_path=progress_path,
            progress_every=int(progress_every),
        )
        return {
            **base,
            "written": True,
            "profile_schema": profile.get("schema"),
            "profile_sha256": sha256_file(profile_path),
            "blockers": list(profile.get("blockers") or []),
        }
    except Exception as exc:
        try:
            from tac.local_acceleration.mlx_renderer_prefilter_profile import (
                write_mlx_renderer_prefilter_failure_profile,
            )

            failure = write_mlx_renderer_prefilter_failure_profile(
                output_path=profile_path,
                archive_bytes=archive_bytes,
                archive_sha256=archive_sha256,
                num_pairs=int(target0_np.shape[0]),
                failure=repr(exc),
                run_id="snerv_native_receiver_decoded_packet_prefilter",
            )
            return {
                **base,
                "written": False,
                "profile_schema": failure.get("schema"),
                "profile_sha256": sha256_file(profile_path),
                "failure": repr(exc),
                "blockers": [
                    "snerv_native_mlx_prefilter_failed",
                    *list(failure.get("blockers") or []),
                ],
            }
        except Exception:
            return {
                **base,
                "written": False,
                "failure": repr(exc),
                "blockers": ["snerv_native_mlx_prefilter_failed"],
            }


def export_snerv_mlx_archive(
    model_or_artifact: Any,
    output_dir: str | Path,
    repo_root: str | Path,
    *,
    retain_receiver_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
    hard_byte_ceiling: int | None = None,
    allow_over_hard_byte_ceiling_for_measurement: bool = False,
    submission_archive_format: str = "snar2",
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Export receiver packet bytes through the canonical archive-bound package.

    Training packets keep rich JSON metadata for analysis.  Submission packets
    default to SNAR2 so fixed receiver grammar and non-signal labels are paid in
    ``inflate.py`` code instead of ``archive.zip`` bytes.
    """

    root = _repo_root(repo_root)
    out = Path(output_dir).expanduser().resolve(strict=False)
    input_packet = _packet_bytes_from_artifact(model_or_artifact)
    decoded = unpack_snerv_archive(input_packet)
    export_hard_byte_ceiling = _snerv_export_hard_byte_ceiling(
        explicit=hard_byte_ceiling,
        metadata=decoded.metadata,
    )
    packet, submission_repack = _snerv_submission_packet_for_export(
        input_packet,
        submission_archive_format=submission_archive_format,
    )
    export_decoded = unpack_snerv_archive(packet)
    storage = build_snerv_mlx_native_storage_preflight(
        output_dir=out,
        n_pairs=int(export_decoded.metadata.get("n_pairs", decoded.metadata.get("n_pairs", 0))),
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
        allow_overwrite=allow_overwrite,
    )
    package["snerv_mlx_native_storage_preflight"] = storage
    package["snerv_submission_archive_repack"] = submission_repack
    _enforce_snerv_mlx_archive_hard_byte_ceiling(
        package,
        output_dir=out,
        hard_byte_ceiling=export_hard_byte_ceiling,
        measurement_bypass_enabled=bool(
            allow_over_hard_byte_ceiling_for_measurement
        ),
    )
    return package


def _snerv_submission_packet_for_export(
    packet: bytes,
    *,
    submission_archive_format: str,
) -> tuple[bytes, dict[str, Any]]:
    """Return receiver packet bytes for archive-bound export.

    ``snar2`` is a lossless container transform: the four receiver sections are
    byte-identical, but the outer self-describing JSON header is replaced by a
    fixed binary header whose constants live in receiver code.
    """

    requested = str(submission_archive_format or "snar2").strip().lower()
    decoded = unpack_snerv_archive(packet)
    base = {
        "schema": "snerv_submission_archive_repack.v1",
        "requested_archive_format": requested,
        "input_packet_schema": decoded.schema,
        "input_packet_bytes": len(packet),
        "input_packet_sha256": _sha256_bytes(packet),
        "section_sha256": {
            name: _sha256_bytes(decoded.sections[name]) for name in SECTION_ORDER
        },
        "lossless_receiver_section_transform": True,
        **FALSE_AUTHORITY,
    }
    if requested in {"none", "input", "preserve", "snar1"}:
        return bytes(packet), {
            **base,
            "output_packet_schema": decoded.schema,
            "output_packet_bytes": len(packet),
            "output_packet_sha256": _sha256_bytes(packet),
            "repacked": False,
            "bytes_saved": 0,
            "blockers": [],
        }
    if requested not in {"snar2", "compact", "fixed_header"}:
        raise SnervMlxNativeExportError(
            f"unsupported SNeRV submission archive format: {submission_archive_format!r}"
        )
    compact = pack_snerv_archive_snar2(
        metadata_payload=decoded.sections["metadata_payload"],
        lf_payload=decoded.sections["lf_payload"],
        decoder_payload=decoded.sections["decoder_payload"],
        step_map_packet=decoded.sections["step_map_packet"],
        metadata=decoded.metadata,
    ).packet
    compact_decoded = unpack_snerv_archive(compact)
    for name in SECTION_ORDER:
        if compact_decoded.sections[name] != decoded.sections[name]:
            raise SnervMlxNativeExportError(
                f"SNAR2 repack mutated receiver section {name!r}"
            )
    return compact, {
        **base,
        "output_packet_schema": compact_decoded.schema,
        "output_packet_bytes": len(compact),
        "output_packet_sha256": _sha256_bytes(compact),
        "repacked": compact != packet,
        "bytes_saved": len(packet) - len(compact),
        "blockers": [],
    }


def _snerv_export_hard_byte_ceiling(
    *,
    explicit: int | None,
    metadata: Mapping[str, Any],
) -> int | None:
    value = explicit
    if value is None:
        value = metadata.get(
            "hard_byte_ceiling",
            metadata.get("hard_byte_ceiling_requested_by_candidate_or_startup"),
        )
    if value is None:
        return None
    try:
        ceiling = int(value)
    except (TypeError, ValueError) as exc:
        raise SnervMlxNativeExportError(
            "SNeRV hard_byte_ceiling must be an integer"
        ) from exc
    if ceiling <= 0:
        raise SnervMlxNativeExportError("SNeRV hard_byte_ceiling must be positive")
    return ceiling


def _enforce_snerv_mlx_archive_hard_byte_ceiling(
    package: dict[str, Any],
    *,
    output_dir: Path,
    hard_byte_ceiling: int | None,
    measurement_bypass_enabled: bool,
) -> None:
    receiver_proof = dict(package.get("receiver_proof") or {})
    archive_bytes = _positive_int_or_none(receiver_proof.get("archive_bytes"))
    archive_path = (
        str(receiver_proof.get("archive_path"))
        if receiver_proof.get("archive_path")
        else None
    )
    archive_sha256 = (
        str(receiver_proof.get("archive_sha256"))
        if receiver_proof.get("archive_sha256")
        else None
    )
    blockers: list[str] = []
    overrun: int | None = None
    measurement_bypass_applies = False
    if hard_byte_ceiling is not None:
        if archive_bytes is None:
            blockers.append(
                "snerv_mlx_native_hard_byte_ceiling_archive_bytes_missing"
            )
        else:
            overrun = int(archive_bytes) - int(hard_byte_ceiling)
            if overrun > 0:
                blockers.extend(
                    [
                        "snerv_mlx_native_archive_exceeds_hard_byte_ceiling",
                        "archive_bytes_exceed_tightest_hard_ceiling",
                    ]
                )
                if measurement_bypass_enabled:
                    blockers.append("hard_byte_ceiling_export_bypassed_for_measurement")
                    measurement_bypass_applies = True
    blockers = _ordered_unique(blockers)
    gate = {
        "schema": "snerv_mlx_archive_hard_byte_ceiling_export_gate.v1",
        "hard_byte_ceiling": int(hard_byte_ceiling)
        if hard_byte_ceiling is not None
        else None,
        "archive_bytes": int(archive_bytes) if archive_bytes is not None else None,
        "archive_path": archive_path,
        "archive_sha256": archive_sha256,
        "archive_overrun_bytes": (
            max(0, int(overrun)) if overrun is not None else None
        ),
        "checked": bool(hard_byte_ceiling is not None and archive_bytes is not None),
        "strict_export_enforced": bool(
            hard_byte_ceiling is not None and not measurement_bypass_enabled
        ),
        "measurement_bypass_enabled": bool(measurement_bypass_enabled),
        "measurement_bypass_applies": bool(measurement_bypass_applies),
        "export_allowed": not blockers or bool(measurement_bypass_applies),
        "passed": not blockers,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }
    package["hard_byte_ceiling_export_gate"] = gate
    if blockers:
        package["blockers"] = _ordered_unique(
            [
                *(str(blocker) for blocker in package.get("blockers") or [] if blocker),
                *blockers,
            ]
        )
    if blockers and not measurement_bypass_applies:
        manifest_path = output_dir / "snerv_hard_byte_ceiling_export_blocker.json"
        write_json(
            manifest_path,
            {
                **gate,
                "blocker_manifest_path": manifest_path.as_posix(),
                "failure_class": "strict_hard_byte_ceiling_export_refusal",
            },
        )
        raise SnervMlxNativeExportError(
            "SNeRV archive export exceeds the hard byte ceiling; "
            f"archive_bytes={archive_bytes} hard_byte_ceiling={hard_byte_ceiling} "
            f"blockers={','.join(blockers)} "
            f"blocker_manifest={manifest_path.as_posix()}"
        )


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


def _run_score_aware_long_training_attachment(
    *,
    requested_epochs: int,
    output_dir: str | Path,
    pairs_nchw255: np.ndarray,
    model_size: SnervModelSizeConfig,
    levels: int,
    wavelet: str,
    source_pair_indices: tuple[int, ...],
    target_bits_per_coeff: float,
    step_map_bits_per_coeff: float,
    decoder_payload_codec: str,
    lf_payload_codec: str,
    recon_pixel_weight: np.ndarray | None,
    recon_pixel_weight_metadata: Mapping[str, Any] | None,
    hf_decoder_saliency_gain: float,
    hard_byte_ceiling: int | None,
    learning_rate: float,
    batch_pairs: int,
    section_byte_refresh_every_steps: int,
    optimizer_kind: str,
    grad_clip_max_norm: float | None,
    weight_decay: float | None,
    eval_roundtrip_ste: bool,
    scorer_input_distribution_guard_weight: float,
    scorer_input_distribution_guard_saturation_margin: float,
    scorer_input_distribution_guard_temperature: float,
    scorer_input_contrast_floor_weight: float,
    scorer_input_contrast_floor_segnet_min_std_ratio: float,
    scorer_input_contrast_floor_posenet_yuv6_min_std_ratio: float,
    scorer_input_shape_tether_weight: float = 0.0,
    posenet_yuv6_geometry_tether_weight: float = 0.0,
    posenet_temporal_signal_floor_weight: float = 0.0,
    posenet_temporal_signal_min_std_ratio: float = 0.25,
    posenet_temporal_signal_min_mean_abs_ratio: float = 0.25,
    score_aware_long_training_scorer_space_step_guard_enabled: bool = True,
    score_aware_long_training_scorer_space_step_guard_min_pre_segnet_occupied_class_fraction: float = 0.4,
    score_aware_long_training_scorer_space_step_guard_min_post_segnet_occupied_class_fraction: float = 0.4,
    score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction: float
    | None = 0.8,
    score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_post_segnet_contrast_ratio: float
    | None = 4.25,
    score_aware_long_training_scorer_space_step_guard_max_post_segnet_distribution_mae: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_post_segnet_argmax_disagreement: float
    | None = 0.5,
    score_aware_long_training_scorer_space_step_guard_max_post_pose_score_term: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_post_pose_direct_live_score_term: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_pose_score_term_relative_worsening: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_pose_score_term_absolute_worsening: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_direct_nonrate_score_worsening: float
    | None = None,
    score_aware_long_training_scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening: float
    | None = 5.0,
    score_aware_long_training_scorer_space_step_guard_backtracking_steps: int = 6,
    score_aware_long_training_scorer_space_step_guard_backtracking_shrink: float = 0.5,
    score_aware_long_training_loss_weights: Mapping[str, float] | None = None,
    score_aware_long_training_pose_warmup_epochs: int = 0,
    score_aware_long_training_scorer_input_shape_warmup_epochs: int = 0,
    score_aware_long_training_segnet_direct_live_escape_warmup_epochs: int = 0,
    score_aware_long_training_segnet_direct_live_escape_class_multiplier: float = 1.0,
    checkpoint_retention_keep_last_n: int | None,
    checkpoint_retention_keep_best_n: int,
    checkpoint_retention_keep_every_n_epochs: int | None,
    checkpoint_retention_cold_store_roots: tuple[Path, ...],
    scorer_upstream_dir: str | Path,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    pose_direct_live_distillation_weight: float = 0.0,
    pose_distillation_loss: str,
    pose_distillation_huber_delta: float,
    segnet_distillation_objective: str,
    distillation_temperature: float,
    segnet_student_live_calibration_weight: float,
    segnet_direct_live_distillation_weight: float = 0.0,
    segnet_direct_live_base_loss_weight: float = 1.0,
    segnet_direct_live_class_histogram_weight: float = 0.0,
    segnet_direct_live_class_balanced_hinge_weight: float = 0.0,
    segnet_direct_live_class_balanced_ce_weight: float = 0.0,
    segnet_direct_live_class_balanced_squared_hinge_weight: float = 0.0,
    segnet_direct_live_class_region_recon_weight: float = 0.0,
    segnet_direct_live_rare_class_logit_weight: float = 0.0,
    segnet_direct_live_target_mass_floor_weight: float = 0.0,
    segnet_direct_live_target_min_ratio_floor_weight: float = 0.0,
    segnet_tau_boundary: float,
    segnet_hinge_margin: float,
    distillation_device: str,
    allow_segnet_only_research: bool,
    coder_aware_qat: bool,
    coder_qat_quant_bits: int,
    coder_qat_quant_residual_weight: float,
    coder_qat_magnitude_weight: float,
    coder_qat_delta_weight: float,
    coder_qat_c1a_entropy_weight: float,
    coder_qat_c1a_sigma: float,
    coder_qat_c1a_sample_size: int,
    pr95_faithful_curriculum_enabled: bool,
    pr95_muon_policy: str,
    pr95_stage_source_weight_amplification_enabled: bool = False,
    gradient_multiplier_by_name: Mapping[str, float] | None = None,
    bias_gradient_multiplier: float | None = None,
    output_head_bias_gradient_multiplier: float = 1.0,
    official_trained_checkpoint_mapping_manifest: Mapping[str, Any] | None = None,
    prioritized_pair_indices: tuple[int, ...],
    scorer_error_pair_sampling_weights: Mapping[int, float] | None,
    scorer_error_pair_curriculum: Mapping[str, Any] | None,
    allow_overwrite: bool,
) -> dict[str, Any]:
    """Run real SNeRV MLX long training before NumPy-portable SNAR export."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "snerv_score_aware_long_training.json"
    if report_path.exists() and not allow_overwrite:
        raise SnervMlxNativeExportError(
            f"refusing to overwrite existing score-aware long-training report: {report_path}"
        )
    seg_weight = float(segnet_distillation_weight)
    pose_weight = float(pose_distillation_weight)
    pose_direct_live_weight = float(pose_direct_live_distillation_weight)
    guard_weight = float(scorer_input_distribution_guard_weight)
    guard_saturation_margin = float(scorer_input_distribution_guard_saturation_margin)
    guard_temperature = float(scorer_input_distribution_guard_temperature)
    contrast_floor_weight = float(scorer_input_contrast_floor_weight)
    contrast_floor_segnet_ratio = float(
        scorer_input_contrast_floor_segnet_min_std_ratio
    )
    contrast_floor_posenet_ratio = float(
        scorer_input_contrast_floor_posenet_yuv6_min_std_ratio
    )
    shape_tether_weight = float(scorer_input_shape_tether_weight)
    geometry_tether_weight = float(posenet_yuv6_geometry_tether_weight)
    temporal_floor_weight = float(posenet_temporal_signal_floor_weight)
    temporal_floor_std_ratio = float(posenet_temporal_signal_min_std_ratio)
    temporal_floor_mean_abs_ratio = float(
        posenet_temporal_signal_min_mean_abs_ratio
    )
    scorer_space_step_guard_enabled = bool(
        score_aware_long_training_scorer_space_step_guard_enabled
    )
    scorer_space_step_guard_min_pre_segnet_occupied_class_fraction = float(
        score_aware_long_training_scorer_space_step_guard_min_pre_segnet_occupied_class_fraction
    )
    scorer_space_step_guard_min_post_segnet_occupied_class_fraction = float(
        score_aware_long_training_scorer_space_step_guard_min_post_segnet_occupied_class_fraction
    )
    scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction = (
        None
        if score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction
        is None
        else float(
            score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction
        )
    )
    scorer_space_step_guard_min_post_segnet_target_class_min_ratio = (
        None
        if score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio
        is None
        else float(
            score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio
        )
    )
    scorer_space_step_guard_max_post_segnet_target_class_ratio_drop = (
        None
        if score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop
        is None
        else float(
            score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop
        )
    )
    scorer_space_step_guard_backtracking_steps = int(
        score_aware_long_training_scorer_space_step_guard_backtracking_steps
    )
    scorer_space_step_guard_backtracking_shrink = float(
        score_aware_long_training_scorer_space_step_guard_backtracking_shrink
    )
    live_calibration_weight = float(segnet_student_live_calibration_weight)
    direct_live_weight = float(segnet_direct_live_distillation_weight)
    direct_live_base_loss_weight = float(segnet_direct_live_base_loss_weight)
    direct_live_histogram_weight = float(segnet_direct_live_class_histogram_weight)
    direct_live_balanced_hinge_weight = float(
        segnet_direct_live_class_balanced_hinge_weight
    )
    direct_live_balanced_ce_weight = float(
        segnet_direct_live_class_balanced_ce_weight
    )
    direct_live_balanced_squared_hinge_weight = float(
        segnet_direct_live_class_balanced_squared_hinge_weight
    )
    direct_live_class_region_recon_weight = float(
        segnet_direct_live_class_region_recon_weight
    )
    direct_live_rare_class_logit_weight = float(
        segnet_direct_live_rare_class_logit_weight
    )
    direct_live_target_mass_floor_weight = float(
        segnet_direct_live_target_mass_floor_weight
    )
    direct_live_target_min_ratio_floor_weight = float(
        segnet_direct_live_target_min_ratio_floor_weight
    )
    escape_class_multiplier = float(
        score_aware_long_training_segnet_direct_live_escape_class_multiplier
    )
    if not math.isfinite(escape_class_multiplier) or escape_class_multiplier <= 0.0:
        raise SnervMlxNativeExportError(
            "score_aware_long_training_segnet_direct_live_escape_class_multiplier "
            "must be finite and > 0"
        )
    direct_live_segnet_subcontrol_active = any(
        value > 0.0
        for value in (
            direct_live_histogram_weight,
            direct_live_balanced_hinge_weight,
            direct_live_balanced_ce_weight,
            direct_live_balanced_squared_hinge_weight,
            direct_live_class_region_recon_weight,
            direct_live_rare_class_logit_weight,
            direct_live_target_mass_floor_weight,
            direct_live_target_min_ratio_floor_weight,
        )
    )
    direct_live_segnet_active = (
        direct_live_weight > 0.0 or direct_live_segnet_subcontrol_active
    )
    pose_loss = str(pose_distillation_loss)
    pose_huber_delta = float(pose_distillation_huber_delta)
    scorer_error_pair_sampling_weights = {
        int(pair): float(weight)
        for pair, weight in dict(scorer_error_pair_sampling_weights or {}).items()
    }
    scorer_error_pair_curriculum = dict(scorer_error_pair_curriculum or {})
    gradient_multiplier_by_name = _coerce_gradient_multiplier_by_name(
        gradient_multiplier_by_name
    )
    if bias_gradient_multiplier is not None:
        bias_gradient_multiplier = float(bias_gradient_multiplier)
        if not math.isfinite(bias_gradient_multiplier) or bias_gradient_multiplier < 0.0:
            raise SnervMlxNativeExportError(
                "bias_gradient_multiplier must be finite and >= 0"
            )
    output_head_bias_gradient_multiplier = float(output_head_bias_gradient_multiplier)
    if (
        not math.isfinite(output_head_bias_gradient_multiplier)
        or output_head_bias_gradient_multiplier < 0.0
    ):
        raise SnervMlxNativeExportError(
            "output_head_bias_gradient_multiplier must be finite and >= 0"
        )
    resolved_scorer_upstream_dir = Path(scorer_upstream_dir).expanduser().resolve(
        strict=False
    )
    distillation_requested = bool(
        seg_weight > 0.0
        or pose_weight > 0.0
        or pose_direct_live_weight > 0.0
        or direct_live_segnet_active
    )
    requested_distillation_device = str(distillation_device)
    resolved_distillation_device = (
        _resolve_torch_scorer_device_alias(requested_distillation_device)
        if distillation_requested
        else requested_distillation_device
    )
    official_training_requested = bool(
        model_size.official_mfu_hfr_tub_numeric_primitives_requested
    )
    trained_checkpoint_mapping_manifest = (
        dict(official_trained_checkpoint_mapping_manifest)
        if isinstance(official_trained_checkpoint_mapping_manifest, Mapping)
        else build_snerv_official_trained_checkpoint_mapping_manifest(
            None,
            state_dict_kind=(
                "missing_upstream_official_checkpoint_for_mlx_long_training"
            ),
            source="snerv_mlx_native_train_export",
        )
    )
    official_checkpoint_loaded = bool(
        trained_checkpoint_mapping_manifest.get("official_trained_checkpoint_loaded")
        is True
    )
    official_checkpoint_mapping_verified = bool(
        trained_checkpoint_mapping_manifest.get(
            "official_mfu_hfr_trained_checkpoint_weight_mapping_proven"
        )
        is True
        and trained_checkpoint_mapping_manifest.get(
            "official_tub_temporal_encoder_weight_mapping_proven"
        )
        is True
    )
    training_kind = (
        "snerv_mlx_official_mfu_hfr_tub_score_renderer"
        if official_training_requested
        else "snerv_mlx_score_aware_haar_renderer"
    )
    try:
        stage_loss_weights = coerce_scoreaware_stage_loss_weights(
            score_aware_long_training_loss_weights
        )
        if int(requested_epochs) > 0:
            score_aware_curriculum_stages = build_scoreaware_curriculum_stages(
                substrate_id="snerv_inverse_steg_carrier",
                epochs=int(requested_epochs),
                loss_weights=stage_loss_weights,
                pose_distillation_warmup_epochs=int(
                    score_aware_long_training_pose_warmup_epochs
                ),
                scorer_input_shape_warmup_epochs=int(
                    score_aware_long_training_scorer_input_shape_warmup_epochs
                ),
                segnet_direct_live_escape_warmup_epochs=int(
                    score_aware_long_training_segnet_direct_live_escape_warmup_epochs
                ),
                segnet_direct_live_escape_class_multiplier=escape_class_multiplier,
            )
        else:
            score_aware_curriculum_stages = ()
    except ValueError as exc:
        raise SnervMlxNativeExportError(str(exc)) from exc
    base_payload = {
        "schema": "snerv_mlx_score_aware_long_training_attachment.v1",
        "requested_epochs": int(requested_epochs),
        "executed": False,
        "training_kind": training_kind,
        "optimizer_kind": str(optimizer_kind),
        "stage_loss_weights": stage_loss_weights,
        "gradient_multiplier_controls": {
            "schema": "snerv_score_aware_long_training_gradient_multiplier_controls.v1",
            "enabled": bool(
                any(float(value) != 1.0 for value in gradient_multiplier_by_name.values())
                or (
                    bias_gradient_multiplier is not None
                    and float(bias_gradient_multiplier) != 1.0
                )
                or float(output_head_bias_gradient_multiplier) != 1.0
            ),
            "exact_active_name_count": sum(
                1
                for value in gradient_multiplier_by_name.values()
                if float(value) != 1.0
            ),
            "bias_gradient_multiplier": bias_gradient_multiplier,
            "output_head_bias_gradient_multiplier": float(
                output_head_bias_gradient_multiplier
            ),
            "authority": "macos_mlx_training_lagrangian_false_authority",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "curriculum_warmup_epochs": {
            "pose_distillation_warmup_epochs": int(
                score_aware_long_training_pose_warmup_epochs
            ),
            "scorer_input_shape_warmup_epochs": int(
                score_aware_long_training_scorer_input_shape_warmup_epochs
            ),
            "segnet_direct_live_escape_warmup_epochs": int(
                score_aware_long_training_segnet_direct_live_escape_warmup_epochs
            ),
            "segnet_direct_live_escape_class_multiplier": escape_class_multiplier,
        },
        "curriculum_stage_count": len(score_aware_curriculum_stages),
        "curriculum_stage_names": [
            str(getattr(stage, "name", "")) for stage in score_aware_curriculum_stages
        ],
        "section_byte_refresh_every_steps": int(section_byte_refresh_every_steps),
        "checkpoint_retention": {
            "schema": "snerv_mlx_score_aware_checkpoint_retention.v1",
            "keep_last_n": checkpoint_retention_keep_last_n,
            "keep_best_n": int(checkpoint_retention_keep_best_n),
            "keep_every_n_epochs": checkpoint_retention_keep_every_n_epochs,
            "cold_store_roots": [
                root.as_posix() for root in checkpoint_retention_cold_store_roots
            ],
            "hot_directory_scope": "periodic_checkpoints_only_final_always_kept",
            **FALSE_AUTHORITY,
        },
        "levels": int(levels),
        "wavelet": str(wavelet),
        "source_pair_indices": [int(value) for value in source_pair_indices],
        "model_size": model_size.as_jsonable(),
        "human_visual_fidelity_objective": False,
        "contest_scorer_distillation_objective": distillation_requested,
        "coder_aware_qat_requested": bool(coder_aware_qat),
        "pr95_faithful_curriculum_enabled": bool(
            pr95_faithful_curriculum_enabled
        ),
        "pr95_muon_policy": str(pr95_muon_policy),
        "scorer_input_distribution_guard": {
            "schema": "snerv_mlx_score_aware_scorer_input_distribution_guard.v1",
            "requested": guard_weight > 0.0,
            "enabled": guard_weight > 0.0,
            "bound_to_renderer_bundle": False,
            "weight": guard_weight,
            "saturation_margin": guard_saturation_margin,
            "temperature": guard_temperature,
            "target_surface": "decoded_rgb01_vs_target_rgb01_mean_std_soft_saturation",
            "score_authority": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "scorer_input_distribution_guard_bound": False,
        "scorer_input_contrast_floor": {
            "schema": "snerv_mlx_score_aware_scorer_input_contrast_floor.v1",
            "requested": contrast_floor_weight > 0.0,
            "enabled": contrast_floor_weight > 0.0,
            "bound_to_renderer_bundle": False,
            "weight": contrast_floor_weight,
            "segnet_last_rgb_min_std_ratio": contrast_floor_segnet_ratio,
            "posenet_yuv6_pair_min_std_ratio": contrast_floor_posenet_ratio,
            "target_surface": (
                "segnet_last_frame_rgb_and_posenet_two_frame_yuv6_std_ratio"
            ),
            "human_visual_fidelity_objective": False,
            "score_authority": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "scorer_input_contrast_floor_bound": False,
        "scorer_input_shape_tether": {
            "schema": "snerv_mlx_score_aware_scorer_input_shape_tether.v1",
            "requested": shape_tether_weight > 0.0,
            "enabled": shape_tether_weight > 0.0,
            "bound_to_renderer_bundle": False,
            "weight": shape_tether_weight,
            "target_surface": (
                "segnet_last_frame_rgb_plus_posenet_yuv6_pair_and_temporal_delta_"
                "centered_reference_variance_normalized_fit"
            ),
            "human_visual_fidelity_objective": False,
            "score_authority": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "scorer_input_shape_tether_bound": False,
        "posenet_yuv6_geometry_tether": {
            "schema": "snerv_mlx_score_aware_posenet_yuv6_geometry_tether.v1",
            "requested": geometry_tether_weight > 0.0,
            "enabled": geometry_tether_weight > 0.0,
            "bound_to_renderer_bundle": False,
            "weight": geometry_tether_weight,
            "target_surface": (
                "exact_upstream_posenet_yuv6_pair_spatial_gradient_and_"
                "temporal_delta_geometry"
            ),
            "human_visual_fidelity_objective": False,
            "score_authority": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "posenet_yuv6_geometry_tether_bound": False,
        "posenet_temporal_signal_floor": {
            "schema": "snerv_mlx_score_aware_posenet_temporal_signal_floor.v1",
            "requested": temporal_floor_weight > 0.0,
            "enabled": temporal_floor_weight > 0.0,
            "bound_to_renderer_bundle": False,
            "weight": temporal_floor_weight,
            "min_std_ratio": temporal_floor_std_ratio,
            "min_mean_abs_ratio": temporal_floor_mean_abs_ratio,
            "target_surface": (
                "exact_upstream_posenet_yuv6_frame1_minus_frame0_temporal_signal"
            ),
            "human_visual_fidelity_objective": False,
            "score_authority": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "posenet_temporal_signal_floor_bound": False,
        "scorer_space_step_guard": {
            "schema": "snerv_mlx_score_aware_scorer_space_step_guard.v1",
            "requested": scorer_space_step_guard_enabled,
            "enabled": scorer_space_step_guard_enabled,
            "bound_to_shared_mlx_adapter": False,
            "min_pre_segnet_occupied_class_fraction": (
                scorer_space_step_guard_min_pre_segnet_occupied_class_fraction
            ),
            "min_post_segnet_occupied_class_fraction": (
                scorer_space_step_guard_min_post_segnet_occupied_class_fraction
            ),
            "min_post_segnet_target_class_coverage_fraction": (
                scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction
            ),
            "min_post_segnet_target_class_min_ratio": (
                scorer_space_step_guard_min_post_segnet_target_class_min_ratio
            ),
            "max_post_segnet_target_class_ratio_drop": (
                scorer_space_step_guard_max_post_segnet_target_class_ratio_drop
            ),
            "max_post_segnet_contrast_ratio": (
                None
                if score_aware_long_training_scorer_space_step_guard_max_post_segnet_contrast_ratio
                is None
                else float(
                    score_aware_long_training_scorer_space_step_guard_max_post_segnet_contrast_ratio
                )
            ),
            "max_post_segnet_argmax_disagreement": (
                None
                if score_aware_long_training_scorer_space_step_guard_max_post_segnet_argmax_disagreement
                is None
                else float(
                    score_aware_long_training_scorer_space_step_guard_max_post_segnet_argmax_disagreement
                )
            ),
            "backtracking_steps": scorer_space_step_guard_backtracking_steps,
            "backtracking_shrink": scorer_space_step_guard_backtracking_shrink,
            "target_surface": (
                "shared_mlx_real_scorer_post_update_trust_region_for_segnet_"
                "last_frame_rgb_and_posenet_yuv6_pair"
            ),
            "human_visual_fidelity_objective": False,
            "score_authority": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "scorer_space_step_guard_bound": False,
        "prioritized_pair_training": {
            "schema": "snerv_mlx_score_aware_long_training_priority_pairs.v1",
            "enabled": bool(prioritized_pair_indices),
            "pair_indices": [int(value) for value in prioritized_pair_indices],
            "pair_count": len(prioritized_pair_indices),
            "sampling_scope": "score_aware_training_batches_not_target_hydration",
            **FALSE_AUTHORITY,
        },
        "scorer_error_pair_curriculum": {
            **strip_candidate_curriculum_authority_fields(
                scorer_error_pair_curriculum
            ),
            "schema": "snerv_mlx_score_aware_scorer_error_pair_curriculum.v1",
            "consumed_by_shared_mlx_sampler": bool(
                scorer_error_pair_sampling_weights
            ),
            "weighted_pair_count": len(scorer_error_pair_sampling_weights),
            "sampling_scope": "score_aware_training_batches_not_target_hydration",
            "canonical_authority_surface": (
                "TrainingArtifact top-level false-authority fields"
            ),
        },
        "official_mfu_hfr_tub_train_export": {
            "schema": "snerv_official_mfu_hfr_tub_train_export_binding.v1",
            "requested": official_training_requested,
            "train_renderer_bound": False,
            "trained_receiver_payload_exported": False,
            "trained_receiver_state_bound": False,
            "trained_receiver_state_mapping_scope": "none",
            "trained_weight_mapping_to_long_training_bound": False,
            "official_trained_checkpoint_state_dict_loaded": official_checkpoint_loaded,
            "official_trained_checkpoint_state_dict_mapping_verified": (
                official_checkpoint_mapping_verified
            ),
            "official_trained_checkpoint_mapping_manifest": (
                trained_checkpoint_mapping_manifest
            ),
            "official_trained_checkpoint_source_forward_replay_verified": False,
            "source_forward_replay_authority": False,
            **FALSE_AUTHORITY,
        },
        "contest_scorer_distortion_objective": bool(
            _recon_pixel_weight_metadata_is_verified_gradient_manifest(
                recon_pixel_weight_metadata
            )
        ),
        "teacher_binding": {
            "schema": "snerv_mlx_real_scorer_teacher_binding.v1",
            "requested": distillation_requested,
            "segnet_distillation_weight": seg_weight,
            "pose_distillation_weight": pose_weight,
            "pose_direct_live_distillation_weight": pose_direct_live_weight,
            "pose_distillation_loss": pose_loss,
            "pose_distillation_huber_delta": pose_huber_delta,
            "segnet_distillation_objective": str(segnet_distillation_objective),
            "distillation_temperature": float(distillation_temperature),
            "segnet_student_live_calibration_weight": live_calibration_weight,
            "segnet_direct_live_distillation_weight": direct_live_weight,
            "segnet_direct_live_base_loss_weight": direct_live_base_loss_weight,
            "segnet_direct_live_class_histogram_weight": direct_live_histogram_weight,
            "segnet_direct_live_class_balanced_hinge_weight": (
                direct_live_balanced_hinge_weight
            ),
            "segnet_direct_live_class_balanced_ce_weight": (
                direct_live_balanced_ce_weight
            ),
            "segnet_direct_live_class_balanced_squared_hinge_weight": (
                direct_live_balanced_squared_hinge_weight
            ),
            "segnet_direct_live_class_region_recon_weight": (
                direct_live_class_region_recon_weight
            ),
            "segnet_direct_live_rare_class_logit_weight": (
                direct_live_rare_class_logit_weight
            ),
            "segnet_direct_live_target_mass_floor_weight": (
                direct_live_target_mass_floor_weight
            ),
            "segnet_direct_live_target_min_ratio_floor_weight": (
                direct_live_target_min_ratio_floor_weight
            ),
            "segnet_tau_boundary": float(segnet_tau_boundary),
            "segnet_hinge_margin": float(segnet_hinge_margin),
            "requested_distillation_device": requested_distillation_device,
            "distillation_device": resolved_distillation_device,
            "distillation_device_resolution": {
                "schema": "snerv_native_torch_scorer_device_resolution.v1",
                "requested": requested_distillation_device,
                "resolved": resolved_distillation_device,
                "scope": "real_pytorch_segnet_posenet_teacher_cache",
            },
            "scorer_upstream_dir": resolved_scorer_upstream_dir.as_posix(),
            "has_real_segnet_teacher": False,
            "has_real_posenet_teacher": False,
            "allow_segnet_only_research": bool(allow_segnet_only_research),
            "pose_student_input_preprocess": (
                "pr95_yuv6"
                if pose_weight > 0.0 or pose_direct_live_weight > 0.0
                else None
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "has_real_segnet_teacher": False,
        "has_real_posenet_teacher": False,
        **FALSE_AUTHORITY,
    }
    validation_blockers: list[str] = []
    if seg_weight < 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_distillation_weight_negative"
        )
    if pose_weight < 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_pose_distillation_weight_negative"
        )
    if pose_direct_live_weight < 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_pose_direct_live_distillation_weight_negative"
        )
    if live_calibration_weight < 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_student_live_calibration_weight_negative"
        )
    if direct_live_weight < 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_direct_live_distillation_weight_negative"
        )
    if (
        not np.isfinite(direct_live_base_loss_weight)
        or direct_live_base_loss_weight < 0.0
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_direct_live_base_loss_weight_invalid"
        )
    if not np.isfinite(direct_live_histogram_weight) or direct_live_histogram_weight < 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_direct_live_class_histogram_weight_invalid"
        )
    if (
        not np.isfinite(direct_live_balanced_hinge_weight)
        or direct_live_balanced_hinge_weight < 0.0
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_direct_live_class_balanced_hinge_weight_invalid"
        )
    if not np.isfinite(direct_live_balanced_ce_weight) or direct_live_balanced_ce_weight < 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_direct_live_class_balanced_ce_weight_invalid"
        )
    if (
        not np.isfinite(direct_live_balanced_squared_hinge_weight)
        or direct_live_balanced_squared_hinge_weight < 0.0
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_direct_live_class_balanced_squared_hinge_weight_invalid"
        )
    if (
        not np.isfinite(direct_live_class_region_recon_weight)
        or direct_live_class_region_recon_weight < 0.0
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_direct_live_class_region_recon_weight_invalid"
        )
    if (
        not np.isfinite(direct_live_rare_class_logit_weight)
        or direct_live_rare_class_logit_weight < 0.0
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_direct_live_rare_class_logit_weight_invalid"
        )
    if (
        not np.isfinite(direct_live_target_mass_floor_weight)
        or direct_live_target_mass_floor_weight < 0.0
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_direct_live_target_mass_floor_weight_invalid"
        )
    if (
        not np.isfinite(direct_live_target_min_ratio_floor_weight)
        or direct_live_target_min_ratio_floor_weight < 0.0
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_direct_live_target_min_ratio_floor_weight_invalid"
        )
    if guard_weight < 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_input_distribution_guard_weight_negative"
        )
    if not (0.0 < guard_saturation_margin < 0.5):
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_input_distribution_guard_saturation_margin_invalid"
        )
    if guard_temperature <= 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_input_distribution_guard_temperature_nonpositive"
        )
    if not np.isfinite(contrast_floor_weight) or contrast_floor_weight < 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_input_contrast_floor_weight_invalid"
        )
    if not np.isfinite(contrast_floor_segnet_ratio) or contrast_floor_segnet_ratio <= 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_input_contrast_floor_segnet_ratio_invalid"
        )
    if not np.isfinite(contrast_floor_posenet_ratio) or contrast_floor_posenet_ratio <= 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_input_contrast_floor_posenet_ratio_invalid"
        )
    if not np.isfinite(shape_tether_weight) or shape_tether_weight < 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_input_shape_tether_weight_invalid"
        )
    if not np.isfinite(geometry_tether_weight) or geometry_tether_weight < 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_posenet_yuv6_geometry_tether_weight_invalid"
        )
    if not np.isfinite(temporal_floor_weight) or temporal_floor_weight < 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_posenet_temporal_signal_floor_weight_invalid"
        )
    if not np.isfinite(temporal_floor_std_ratio) or temporal_floor_std_ratio <= 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_posenet_temporal_signal_floor_std_ratio_invalid"
        )
    if (
        not np.isfinite(temporal_floor_mean_abs_ratio)
        or temporal_floor_mean_abs_ratio <= 0.0
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_posenet_temporal_signal_floor_mean_abs_ratio_invalid"
        )
    if not np.isfinite(scorer_space_step_guard_min_pre_segnet_occupied_class_fraction):
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_space_step_guard_min_pre_segnet_occupied_invalid"
        )
    if not np.isfinite(scorer_space_step_guard_min_post_segnet_occupied_class_fraction):
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_space_step_guard_min_post_segnet_occupied_invalid"
        )
    if (
        scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction
        is not None
        and (
            not np.isfinite(
                scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction
            )
            or scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction
            < 0.0
            or scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction
            > 1.0
        )
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_invalid"
        )
    if (
        scorer_space_step_guard_min_post_segnet_target_class_min_ratio is not None
        and (
            not np.isfinite(
                scorer_space_step_guard_min_post_segnet_target_class_min_ratio
            )
            or scorer_space_step_guard_min_post_segnet_target_class_min_ratio < 0.0
            or scorer_space_step_guard_min_post_segnet_target_class_min_ratio > 1.0
        )
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio_invalid"
        )
    if (
        scorer_space_step_guard_max_post_segnet_target_class_ratio_drop is not None
        and (
            not np.isfinite(
                scorer_space_step_guard_max_post_segnet_target_class_ratio_drop
            )
            or scorer_space_step_guard_max_post_segnet_target_class_ratio_drop < 0.0
            or scorer_space_step_guard_max_post_segnet_target_class_ratio_drop > 1.0
        )
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop_invalid"
        )
    if scorer_space_step_guard_backtracking_steps < 0:
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_space_step_guard_backtracking_steps_invalid"
        )
    if (
        not np.isfinite(scorer_space_step_guard_backtracking_shrink)
        or scorer_space_step_guard_backtracking_shrink <= 0.0
        or scorer_space_step_guard_backtracking_shrink >= 1.0
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_scorer_space_step_guard_backtracking_shrink_invalid"
        )
    if pose_loss not in {"mse", "huber"}:
        validation_blockers.append(
            "snerv_score_aware_long_training_pose_distillation_loss_invalid"
        )
    if pose_huber_delta <= 0.0:
        validation_blockers.append(
            "snerv_score_aware_long_training_pose_huber_delta_nonpositive"
        )
    if (
        (seg_weight > 0.0 or direct_live_segnet_active)
        and pose_weight <= 0.0
        and not allow_segnet_only_research
    ):
        validation_blockers.append(
            "snerv_score_aware_long_training_segnet_requires_posenet_teacher"
        )
    if distillation_requested:
        required = [resolved_scorer_upstream_dir / "modules.py"]
        if seg_weight > 0.0 or direct_live_segnet_active:
            required.append(
                resolved_scorer_upstream_dir / "models" / "segnet.safetensors"
            )
        if pose_weight > 0.0:
            required.append(
                resolved_scorer_upstream_dir / "models" / "posenet.safetensors"
            )
        missing = [path.as_posix() for path in required if not path.is_file()]
        if missing:
            validation_blockers.append(
                "snerv_score_aware_long_training_real_teacher_upstream_missing"
            )
    if validation_blockers:
        payload = {
            **base_payload,
            "validation_failures": _ordered_unique(validation_blockers),
            "blockers": _ordered_unique(validation_blockers),
        }
        write_json(report_path, payload)
        return {**payload, "report_path": report_path.as_posix()}
    if int(requested_epochs) <= 0:
        payload = {
            **base_payload,
            "blockers": ["snerv_mlx_score_aware_long_training_not_requested"],
        }
        write_json(report_path, payload)
        return {**payload, "report_path": report_path.as_posix()}
    pairs = np.asarray(pairs_nchw255, dtype=np.float32)
    official_source_forward_replay: dict[str, Any] | None = None
    if official_training_requested:
        if int(pairs.shape[0]) > SNERV_OFFICIAL_LONG_TRAINING_REPLAY_MAX_PAIRS:
            source_forward_replay = (
                _build_deferred_official_mfu_hfr_tub_long_training_replay_contract(
                    output_dir=out,
                    pair_count=int(pairs.shape[0]),
                    source_pair_indices=source_pair_indices,
                    official_trained_checkpoint_mapping_manifest=(
                        trained_checkpoint_mapping_manifest
                    ),
                    allow_overwrite=allow_overwrite,
                )
            )
        else:
            source_forward_replay = _build_official_mfu_hfr_tub_long_training_replay_contract(
                output_dir=out,
                pairs_nchw255=pairs,
                model_size=model_size,
                source_pair_indices=source_pair_indices,
                official_trained_checkpoint_mapping_manifest=(
                    trained_checkpoint_mapping_manifest
                ),
                allow_overwrite=allow_overwrite,
            )
        official_source_forward_replay = (
            _official_long_training_replay_with_renderer_binding(
                source_forward_replay,
                official_trained_checkpoint_mapping_manifest=(
                    trained_checkpoint_mapping_manifest
                ),
            )
        )
        official_validation_blockers: list[str] = []
        if int(levels) != 1:
            official_validation_blockers.append(
                "snerv_score_aware_long_training_official_mfu_hfr_tub_requires_haar_j1"
            )
        if str(wavelet).strip().lower() not in {"haar", "db1"}:
            official_validation_blockers.append(
                "snerv_score_aware_long_training_official_mfu_hfr_tub_requires_haar_j1"
            )
        if official_validation_blockers:
            payload = {
                **base_payload,
                "official_mfu_hfr_tub_source_forward_replay": official_source_forward_replay,
                "blockers": _ordered_unique(official_validation_blockers),
            }
            write_json(report_path, payload)
            return {**payload, "report_path": report_path.as_posix()}
    if (not official_training_requested) and str(wavelet).strip().lower() not in {"haar", "db1"}:
        payload = {
            **base_payload,
            "blockers": [
                "snerv_score_aware_long_training_requires_receiver_safe_haar"
            ],
        }
        write_json(report_path, payload)
        return {**payload, "report_path": report_path.as_posix()}

    try:
        import mlx.core as mx

        from tac.local_acceleration.pr95_hnerv_mlx import (
            partition_pr95_mlx_parameter_names,
        )
        from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle
        from tac.substrates._shared.mlx_score_aware.coder_qat import (
            DEFAULT_DECODER_INCLUDE_SUBSTRINGS,
            CoderAwareQATConfig,
            build_decoder_coder_qat_terms,
            coder_qat_loss_weights,
            coder_qat_metadata,
        )
        from tac.substrates._shared.mlx_score_aware.dual_ascent import (
            build_default_nerv_train_time_dual_ascent_config,
        )
        from tac.substrates._shared.mlx_score_aware.harness import (
            run_mlx_score_aware_full_main,
        )
        from tac.substrates._shared.mlx_score_aware.loss import (
            build_mlx_posenet_pair_teacher,
            build_mlx_segnet_pair_teacher,
            score_aware_loss,
        )
        from tac.substrates.hinton_distilled_scorer_surrogate import (
            build_learnable_pose_student_head,
            build_learnable_student_head,
        )
        from tac.substrates.snerv_inverse_steg_carrier.mlx_renderer import (
            SNERV_MLX_OFFICIAL_MFU_HFR_TUB_RENDERER_SCHEMA,
            SNERV_MLX_RENDERER_SCHEMA,
            SnervMlxHaarScoreRenderer,
            SnervMlxOfficialMfuHfrTubScoreRenderer,
        )

        if official_training_requested:
            official_components = _official_mfu_hfr_tub_bootstrap_components_from_pairs(
                pairs,
                model_size=model_size,
            )
            official_tub_output2_kwargs = _official_tub_output2_renderer_kwargs(
                official_components
            )
            model = SnervMlxOfficialMfuHfrTubScoreRenderer(
                mfu=official_components["mfu"],
                hfr_heads=official_components["hfr_heads"],
                low=np.asarray(official_components["low"], dtype=np.float32),
                skip_mid=np.asarray(
                    official_components["skip_mid"],
                    dtype=np.float32,
                ),
                skip_high=np.asarray(
                    official_components["skip_high"],
                    dtype=np.float32,
                ),
                output_hw=(int(pairs.shape[-2]), int(pairs.shape[-1])),
                model_size=model_size,
                skip_high_mode=model_size.official_skip_high_mode,
                tub_current=np.asarray(
                    official_components["tub_current"],
                    dtype=np.float32,
                ),
                tub_previous=np.asarray(
                    official_components["tub_previous"],
                    dtype=np.float32,
                ),
                tub_next_frame=np.asarray(
                    official_components["tub_next_frame"],
                    dtype=np.float32,
                ),
                **official_tub_output2_kwargs,
            )
            renderer_schema = SNERV_MLX_OFFICIAL_MFU_HFR_TUB_RENDERER_SCHEMA
        else:
            model = SnervMlxHaarScoreRenderer.from_numpy_pairs(
                pairs,
                levels=int(levels),
                wavelet="haar",
                model_size=model_size,
            )
            renderer_schema = SNERV_MLX_RENDERER_SCHEMA
        pr95_optimizer_split = partition_pr95_mlx_parameter_names(
            model.parameters()
        )
        pr95_optimizer_coverage = {
            "schema": "snerv_pr95_optimizer_parameter_coverage.v1",
            "pr95_faithful_curriculum_enabled": bool(
                pr95_faithful_curriculum_enabled
            ),
            "pr95_muon_policy_requested": str(pr95_muon_policy),
            "pr95_muon_policy": str(pr95_muon_policy),
            "muon_tensor_count": len(pr95_optimizer_split.get("muon") or []),
            "adamw_tensor_count": len(pr95_optimizer_split.get("adamw") or []),
            "muon_parameter_names": list(pr95_optimizer_split.get("muon") or []),
            "adamw_parameter_names": list(pr95_optimizer_split.get("adamw") or []),
            "vector_decoder_kernels_are_not_matrix_muon_targets": not official_training_requested,
            "official_mfu_hfr_tub_matrix_payload_atoms_bound": official_training_requested,
            **FALSE_AUTHORITY,
        }
        effective_pr95_muon_policy = str(pr95_muon_policy)
        if (
            pr95_faithful_curriculum_enabled
            and str(pr95_muon_policy) == "every_stage"
            and int(pr95_optimizer_coverage["muon_tensor_count"]) <= 0
        ):
            effective_pr95_muon_policy = "faithful_stage8_only"
            pr95_optimizer_coverage = {
                **pr95_optimizer_coverage,
                "pr95_muon_policy": effective_pr95_muon_policy,
                "muon_policy_fallback_applied": True,
                "muon_policy_fallback_reason": (
                    "requested_every_stage_muon_but_snerv_renderer_has_no_"
                    "eligible_matrix_tensors"
                ),
                "score_lane_blocker": False,
            }
        else:
            pr95_optimizer_coverage = {
                **pr95_optimizer_coverage,
                "muon_policy_fallback_applied": False,
                "score_lane_blocker": False,
            }
        coder_qat_cfg = CoderAwareQATConfig(
            enabled=bool(coder_aware_qat),
            quant_bits=int(coder_qat_quant_bits),
            quant_residual_weight=float(coder_qat_quant_residual_weight),
            magnitude_weight=float(coder_qat_magnitude_weight),
            delta_weight=float(coder_qat_delta_weight),
            c1a_entropy_weight=float(coder_qat_c1a_entropy_weight),
            c1a_sigma=float(coder_qat_c1a_sigma),
            c1a_sample_size=int(coder_qat_c1a_sample_size),
            include_substrings=(
                _snerv_official_coder_qat_include_substrings(
                    DEFAULT_DECODER_INCLUDE_SUBSTRINGS
                )
                if official_training_requested
                else DEFAULT_DECODER_INCLUDE_SUBSTRINGS
            ),
        ).validated()
        coder_qat_loss_weight_map = coder_qat_loss_weights(coder_qat_cfg)
        archive_section_qat_policy = (
            _build_snerv_pretraining_archive_section_qat_weight_policy(
                pairs_nchw255=pairs,
                model_size=model_size,
                levels=int(levels),
                wavelet=str(wavelet),
                source_pair_indices=source_pair_indices,
                target_bits_per_coeff=float(target_bits_per_coeff),
                step_map_bits_per_coeff=float(step_map_bits_per_coeff),
                decoder_payload_codec=str(decoder_payload_codec),
                lf_payload_codec=str(lf_payload_codec),
                recon_pixel_weight=recon_pixel_weight,
                recon_pixel_weight_metadata=recon_pixel_weight_metadata,
                hf_decoder_saliency_gain=float(hf_decoder_saliency_gain),
                hard_byte_ceiling=hard_byte_ceiling,
                base_qat_weights=coder_qat_loss_weight_map,
            )
            if coder_qat_cfg.enabled
            else {
                "schema": SNERV_ARCHIVE_SECTION_QAT_WEIGHT_POLICY_SCHEMA,
                "attached": False,
                "active": False,
                "blockers": ["snerv_archive_section_qat_not_requested"],
                **FALSE_AUTHORITY,
            }
        )
        if coder_qat_cfg.enabled and not archive_section_qat_policy.get("blockers"):
            coder_qat_loss_weight_map = {
                str(key): float(value)
                for key, value in dict(
                    archive_section_qat_policy.get("extra_loss_weights") or {}
                ).items()
                if float(value) >= 0.0
            }
        latent_qat_loss_weight_map = {
            str(key): float(value)
            for key, value in coder_qat_loss_weight_map.items()
            if str(key).startswith("latent_qat_") and float(value) > 0.0
        }
        latent_qat_cfg = CoderAwareQATConfig(
            enabled=bool(latent_qat_loss_weight_map),
            quant_bits=int(coder_qat_cfg.quant_bits),
            quant_residual_weight=(
                1.0 if latent_qat_loss_weight_map.get("latent_qat_quant_residual") else 0.0
            ),
            magnitude_weight=(
                1.0 if latent_qat_loss_weight_map.get("latent_qat_magnitude") else 0.0
            ),
            delta_weight=(
                1.0 if latent_qat_loss_weight_map.get("latent_qat_delta") else 0.0
            ),
            c1a_entropy_weight=(
                1.0 if latent_qat_loss_weight_map.get("latent_qat_c1a_entropy") else 0.0
            ),
            c1a_sigma=float(coder_qat_cfg.c1a_sigma),
            c1a_sample_size=int(coder_qat_cfg.c1a_sample_size),
            include_substrings=("latents_lf_planes",),
            exclude_substrings=(),
        ).validated()
        train_time_section_byte_control = _build_snerv_train_time_section_byte_control(
            archive_section_qat_policy,
            coder_qat_loss_weight_map,
            hard_byte_ceiling=hard_byte_ceiling,
        )
        (
            live_train_time_section_byte_metrics,
            live_train_time_section_byte_metrics_metadata,
        ) = _build_snerv_live_train_time_section_byte_metrics_callback(
            model_size=model_size,
            levels=int(levels),
            wavelet=str(wavelet),
            source_pair_indices=source_pair_indices,
            target_bits_per_coeff=float(target_bits_per_coeff),
            step_map_bits_per_coeff=float(step_map_bits_per_coeff),
            decoder_payload_codec=str(decoder_payload_codec),
            lf_payload_codec=str(lf_payload_codec),
            recon_pixel_weight=recon_pixel_weight,
            recon_pixel_weight_metadata=recon_pixel_weight_metadata,
            hf_decoder_saliency_gain=float(hf_decoder_saliency_gain),
            train_time_section_byte_control=train_time_section_byte_control,
            batch_size=max(1, int(batch_pairs)),
            refresh_every_steps=int(section_byte_refresh_every_steps),
        )
        archive_section_qat_blockers = [
            str(blocker)
            for blocker in archive_section_qat_policy.get("blockers") or ()
            if str(blocker) and str(blocker) != "snerv_archive_section_qat_not_requested"
        ]
        if coder_qat_cfg.enabled and archive_section_qat_blockers:
            payload = {
                **base_payload,
                "coder_aware_qat": coder_qat_metadata(coder_qat_cfg),
                "coder_aware_qat_bound": False,
                "archive_section_qat_weight_policy": archive_section_qat_policy,
                "archive_section_qat_weight_policy_bound": False,
                "latent_qat_bound": False,
                "blockers": _ordered_unique(archive_section_qat_blockers),
            }
            write_json(report_path, payload)
            return {**payload, "report_path": report_path.as_posix()}
        train_time_dual_ascent_config = (
            build_default_nerv_train_time_dual_ascent_config(
                family="snerv",
                segnet_distillation_weight=seg_weight,
                segnet_direct_live_distillation_weight=direct_live_weight,
                segnet_direct_live_class_histogram_weight=(
                    direct_live_histogram_weight
                ),
                segnet_direct_live_class_balanced_hinge_weight=(
                    direct_live_balanced_hinge_weight
                ),
                segnet_direct_live_class_balanced_ce_weight=(
                    direct_live_balanced_ce_weight
                ),
                segnet_direct_live_class_balanced_squared_hinge_weight=(
                    direct_live_balanced_squared_hinge_weight
                ),
                segnet_direct_live_class_region_recon_weight=(
                    direct_live_class_region_recon_weight
                ),
                segnet_direct_live_rare_class_logit_weight=(
                    direct_live_rare_class_logit_weight
                ),
                segnet_direct_live_target_mass_floor_weight=(
                    direct_live_target_mass_floor_weight
                ),
                segnet_direct_live_target_min_ratio_floor_weight=(
                    direct_live_target_min_ratio_floor_weight
                ),
                pose_distillation_weight=pose_weight,
                pose_direct_live_distillation_weight=pose_direct_live_weight,
                scorer_input_distribution_guard_weight=guard_weight,
                scorer_input_contrast_floor_weight=contrast_floor_weight,
                scorer_input_shape_tether_weight=shape_tether_weight,
                posenet_yuv6_geometry_tether_weight=geometry_tether_weight,
                posenet_temporal_signal_floor_weight=temporal_floor_weight,
                coder_qat_loss_weight_map=coder_qat_loss_weight_map,
                archive_byte_budget=train_time_section_byte_control.get(
                    "hard_byte_ceiling"
                ),
                section_byte_budgets=train_time_section_byte_control.get(
                    "section_byte_budgets"
                ),
                section_byte_loss_weight_key_map=train_time_section_byte_control.get(
                    "section_byte_loss_weight_key_map"
                ),
                section_byte_loss_weight_scale_map=train_time_section_byte_control.get(
                    "section_byte_loss_weight_scale_map"
                ),
            )
        )
        train_time_dual_ascent_config = _bind_snerv_scorer_tether_dual_targets(
            train_time_dual_ascent_config
        )

        def _extra_loss_terms(model_obj: Any, _idx: Any) -> dict[str, Any]:
            terms = dict(build_decoder_coder_qat_terms(model_obj, coder_qat_cfg))
            if latent_qat_cfg.enabled:
                for key, value in build_decoder_coder_qat_terms(
                    model_obj,
                    latent_qat_cfg,
                ).items():
                    suffix = str(key).removeprefix("coder_qat_")
                    terms[f"latent_qat_{suffix}"] = value
            return terms

        def _train_time_section_byte_metrics(
            _model_obj: Any,
            _idx: Any,
            _loss_weights: Mapping[str, float],
        ) -> Mapping[str, Any] | None:
            payload = train_time_section_byte_control.get("metrics_payload")
            return dict(payload) if isinstance(payload, Mapping) else None

        initial_pairs = model.render_pairs_nchw255(batch_size=max(1, int(batch_pairs)))
        initial_mse = float(np.mean((initial_pairs - pairs) ** 2))
        best_state = model.export_state_dict()
        best_selection: dict[str, Any] = {}
        selection_history: list[dict[str, Any]] = []
        selection_failures: list[str] = []
        selection_interval_epochs = max(1, min(100, int(requested_epochs)))

        target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0, dtype=mx.float32)
        target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0, dtype=mx.float32)
        recon_weight_mlx = (
            None
            if recon_pixel_weight is None
            else mx.array(np.asarray(recon_pixel_weight, dtype=np.float32), dtype=mx.float32)
        )
        metadata_forbidden_authority_keys = {
            "score_claim",
            "promotion_eligible",
            "ready_for_exact_eval_dispatch",
            "rank_or_kill_eligible",
            "promotable",
            "score_claim_valid",
        }
        teacher_binding_metadata = {
            key: value
            for key, value in dict(base_payload["teacher_binding"]).items()
            if key not in metadata_forbidden_authority_keys
        }
        guard_metadata = {
            key: value
            for key, value in dict(
                base_payload["scorer_input_distribution_guard"]
            ).items()
            if key not in metadata_forbidden_authority_keys
        }
        contrast_floor_metadata = {
            key: value
            for key, value in dict(base_payload["scorer_input_contrast_floor"]).items()
            if key not in metadata_forbidden_authority_keys
        }
        temporal_floor_metadata = {
            key: value
            for key, value in dict(base_payload["posenet_temporal_signal_floor"]).items()
            if key not in metadata_forbidden_authority_keys
        }
        pr95_optimizer_coverage_metadata = {
            key: value
            for key, value in dict(pr95_optimizer_coverage).items()
            if key not in metadata_forbidden_authority_keys
        }
        bundle_kwargs: dict[str, Any] = {
            "model": model,
            "target_rgb_0": target0,
            "target_rgb_1": target1,
            "num_pairs": int(pairs.shape[0]),
            "source_pair_indices": tuple(int(value) for value in source_pair_indices),
            "forward_convention": "reconstruct_pair_nchw01",
            "extra_loss_terms": _extra_loss_terms if coder_qat_cfg.enabled else None,
            "extra_loss_weights": coder_qat_loss_weight_map,
            "recon_pixel_weight": recon_weight_mlx,
            "recon_pixel_weight_normalize": "mean",
            "eval_roundtrip_ste_enabled": bool(eval_roundtrip_ste),
            "pose_student_input_preprocess": "pr95_yuv6"
            if pose_weight > 0.0 or pose_direct_live_weight > 0.0
            else "rgb",
            "substrate_artifact_metadata": {
                "schema": "snerv_mlx_score_aware_renderer_bundle.v1",
                "renderer_schema": renderer_schema,
                "source_pair_indices": [int(value) for value in source_pair_indices],
                "receiver_export_path": "SNAR1_numpy_portable_packet_after_training",
                "human_visual_fidelity_objective": False,
                "contest_scorer_distillation_objective": distillation_requested,
                "coder_aware_qat": coder_qat_metadata(coder_qat_cfg),
                "archive_section_qat_weight_policy": (
                    strip_candidate_curriculum_authority_fields(
                        archive_section_qat_policy
                    )
                ),
                "train_time_dual_ascent": (
                    strip_candidate_curriculum_authority_fields(
                        train_time_dual_ascent_config
                    )
                ),
                "train_time_section_byte_control": (
                    strip_candidate_curriculum_authority_fields(
                        train_time_section_byte_control
                    )
                ),
                "live_train_time_section_byte_metrics": (
                    live_train_time_section_byte_metrics_metadata
                ),
                "pr95_faithful_curriculum_enabled": bool(
                    pr95_faithful_curriculum_enabled
                ),
                "pr95_muon_policy_requested": str(pr95_muon_policy),
                "pr95_muon_policy": effective_pr95_muon_policy,
                "pr95_optimizer_coverage": pr95_optimizer_coverage_metadata,
                "scorer_input_distribution_guard": {
                    **guard_metadata,
                    "bound_to_renderer_bundle": guard_weight > 0.0,
                },
                "scorer_input_contrast_floor": {
                    **contrast_floor_metadata,
                    "bound_to_renderer_bundle": contrast_floor_weight > 0.0,
                },
                "scorer_input_shape_tether": {
                    **dict(base_payload["scorer_input_shape_tether"]),
                    "bound_to_renderer_bundle": shape_tether_weight > 0.0,
                },
                "posenet_yuv6_geometry_tether": {
                    **dict(base_payload["posenet_yuv6_geometry_tether"]),
                    "bound_to_renderer_bundle": geometry_tether_weight > 0.0,
                },
                "posenet_temporal_signal_floor": {
                    **temporal_floor_metadata,
                    "bound_to_renderer_bundle": temporal_floor_weight > 0.0,
                },
                "contest_scorer_distortion_objective": bool(
                    _recon_pixel_weight_metadata_is_verified_gradient_manifest(
                        recon_pixel_weight_metadata
                    )
                ),
                "teacher_binding": teacher_binding_metadata,
            },
        }
        if train_time_section_byte_control.get("metrics_payload") is not None:
            bundle_kwargs["train_time_section_byte_metrics"] = (
                live_train_time_section_byte_metrics
                if live_train_time_section_byte_metrics is not None
                else _train_time_section_byte_metrics
            )
        bundle_kwargs["substrate_artifact_metadata"] = (
            strip_candidate_curriculum_authority_fields(
                bundle_kwargs["substrate_artifact_metadata"]
            )
        )
        teacher_probe_bundle = RendererBundle(**bundle_kwargs)
        scorer_teacher = None
        learnable_student_head = None
        pose_scorer_teacher = None
        learnable_pose_student_head = None
        if seg_weight > 0.0 or direct_live_segnet_active:
            scorer_teacher = build_mlx_segnet_pair_teacher(
                teacher_probe_bundle,
                upstream_dir=resolved_scorer_upstream_dir,
                device=resolved_distillation_device,
            )
            learnable_student_head = build_learnable_student_head(
                num_classes=int(scorer_teacher.num_classes),
                seed=0,
            ) if seg_weight > 0.0 else None
        if pose_weight > 0.0:
            pose_scorer_teacher = build_mlx_posenet_pair_teacher(
                teacher_probe_bundle,
                upstream_dir=resolved_scorer_upstream_dir,
                device=resolved_distillation_device,
            )
            learnable_pose_student_head = build_learnable_pose_student_head(
                pose_dims=int(pose_scorer_teacher.pose_dims),
                input_channels=6,
                seed=1,
            )
        teacher_binding = {
            **dict(base_payload["teacher_binding"]),
            "has_real_segnet_teacher": scorer_teacher is not None,
            "has_real_posenet_teacher": pose_scorer_teacher is not None,
            "segnet_teacher_num_classes": (
                int(scorer_teacher.num_classes) if scorer_teacher is not None else None
            ),
            "posenet_teacher_pose_dims": (
                int(pose_scorer_teacher.pose_dims)
                if pose_scorer_teacher is not None
                else None
            ),
            "learnable_student_head_bound": learnable_student_head is not None,
            "learnable_pose_student_head_bound": (
                learnable_pose_student_head is not None
            ),
        }
        bundle_kwargs["substrate_artifact_metadata"] = {
            **dict(bundle_kwargs["substrate_artifact_metadata"]),
            "teacher_binding": {
                key: value
                for key, value in teacher_binding.items()
                if key not in metadata_forbidden_authority_keys
            },
        }
        bundle_kwargs["substrate_artifact_metadata"] = (
            strip_candidate_curriculum_authority_fields(
                bundle_kwargs["substrate_artifact_metadata"]
            )
        )
        bundle = RendererBundle(
            **bundle_kwargs,
            distillation_weight=seg_weight,
            scorer_teacher=scorer_teacher,
            learnable_student_head=learnable_student_head,
            distillation_temperature=float(distillation_temperature),
            segnet_distillation_objective=str(segnet_distillation_objective),
            segnet_student_live_calibration_weight=live_calibration_weight,
            segnet_direct_live_distillation_weight=direct_live_weight,
            segnet_direct_live_base_loss_weight=direct_live_base_loss_weight,
            segnet_direct_live_class_histogram_weight=direct_live_histogram_weight,
            segnet_direct_live_class_balanced_hinge_weight=(
                direct_live_balanced_hinge_weight
            ),
            segnet_direct_live_class_balanced_ce_weight=(
                direct_live_balanced_ce_weight
            ),
            segnet_direct_live_class_balanced_squared_hinge_weight=(
                direct_live_balanced_squared_hinge_weight
            ),
            segnet_direct_live_class_region_recon_weight=(
                direct_live_class_region_recon_weight
            ),
            segnet_direct_live_rare_class_logit_weight=(
                direct_live_rare_class_logit_weight
            ),
            segnet_direct_live_target_mass_floor_weight=(
                direct_live_target_mass_floor_weight
            ),
            segnet_direct_live_target_min_ratio_floor_weight=(
                direct_live_target_min_ratio_floor_weight
            ),
            segnet_tau_boundary=float(segnet_tau_boundary),
            segnet_hinge_margin=float(segnet_hinge_margin),
            distillation_num_classes=(
                int(scorer_teacher.num_classes) if scorer_teacher is not None else 5
            ),
            pose_distillation_weight=pose_weight,
            pose_direct_live_distillation_weight=pose_direct_live_weight,
            pose_distillation_loss=pose_loss,
            pose_distillation_huber_delta=pose_huber_delta,
            pose_scorer_teacher=pose_scorer_teacher,
            learnable_pose_student_head=learnable_pose_student_head,
            pose_dims=(
                int(pose_scorer_teacher.pose_dims)
                if pose_scorer_teacher is not None
                else 6
            ),
            allow_segnet_only_research=bool(allow_segnet_only_research),
            scorer_input_distribution_guard_weight=guard_weight,
            scorer_input_distribution_guard_saturation_margin=guard_saturation_margin,
            scorer_input_distribution_guard_temperature=guard_temperature,
            scorer_input_contrast_floor_weight=contrast_floor_weight,
            scorer_input_contrast_floor_segnet_min_std_ratio=contrast_floor_segnet_ratio,
            scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=(
                contrast_floor_posenet_ratio
            ),
            scorer_input_shape_tether_weight=shape_tether_weight,
            posenet_yuv6_geometry_tether_weight=geometry_tether_weight,
            posenet_temporal_signal_floor_weight=temporal_floor_weight,
            posenet_temporal_signal_min_std_ratio=temporal_floor_std_ratio,
            posenet_temporal_signal_min_mean_abs_ratio=(
                temporal_floor_mean_abs_ratio
            ),
        )
        recon_selection_stage_weight = float(stage_loss_weights.get("recon", 1.0))
        seg_selection_stage_weight = float(stage_loss_weights.get("distill", 1.0))
        direct_live_selection_stage_weight = float(
            stage_loss_weights.get("segnet_direct_live_distill", seg_selection_stage_weight)
        )
        pose_selection_stage_weight = float(
            stage_loss_weights.get("pose_distill", 1.0)
        )
        pose_direct_live_selection_stage_weight = float(
            stage_loss_weights.get(
                "pose_direct_live_distill",
                pose_selection_stage_weight,
            )
        )
        guard_selection_stage_weight = float(
            stage_loss_weights.get("scorer_input_guard", 1.0)
        )
        contrast_floor_selection_stage_weight = float(
            stage_loss_weights.get(
                "scorer_input_contrast_floor",
                guard_selection_stage_weight,
            )
        )
        shape_tether_selection_stage_weight = float(
            stage_loss_weights.get(
                "scorer_input_shape_tether",
                guard_selection_stage_weight,
            )
        )
        geometry_tether_selection_stage_weight = float(
            stage_loss_weights.get(
                "posenet_yuv6_geometry_tether",
                guard_selection_stage_weight,
            )
        )
        temporal_floor_selection_stage_weight = float(
            stage_loss_weights.get(
                "posenet_temporal_signal_floor",
                guard_selection_stage_weight,
            )
        )
        selection_policy = _snerv_score_aware_checkpoint_selection_policy(
            segnet_distillation_weight=seg_weight * seg_selection_stage_weight,
            segnet_direct_live_distillation_weight=(
                direct_live_weight * direct_live_selection_stage_weight
            ),
            segnet_direct_live_class_histogram_weight=(
                direct_live_histogram_weight * direct_live_selection_stage_weight
            ),
            segnet_direct_live_class_balanced_hinge_weight=(
                direct_live_balanced_hinge_weight * direct_live_selection_stage_weight
            ),
            segnet_direct_live_class_balanced_ce_weight=(
                direct_live_balanced_ce_weight * direct_live_selection_stage_weight
            ),
            segnet_direct_live_class_balanced_squared_hinge_weight=(
                direct_live_balanced_squared_hinge_weight
                * direct_live_selection_stage_weight
            ),
            segnet_direct_live_class_region_recon_weight=(
                direct_live_class_region_recon_weight
                * direct_live_selection_stage_weight
            ),
            segnet_direct_live_rare_class_logit_weight=(
                direct_live_rare_class_logit_weight
                * direct_live_selection_stage_weight
            ),
            segnet_direct_live_target_mass_floor_weight=(
                direct_live_target_mass_floor_weight
                * direct_live_selection_stage_weight
            ),
            segnet_direct_live_target_min_ratio_floor_weight=(
                direct_live_target_min_ratio_floor_weight
                * direct_live_selection_stage_weight
            ),
            pose_distillation_weight=pose_weight * pose_selection_stage_weight,
            pose_direct_live_distillation_weight=(
                pose_direct_live_weight * pose_direct_live_selection_stage_weight
            ),
            scorer_input_distribution_guard_weight=(
                guard_weight * guard_selection_stage_weight
            ),
            scorer_input_contrast_floor_weight=(
                contrast_floor_weight * contrast_floor_selection_stage_weight
            ),
            scorer_input_shape_tether_weight=(
                shape_tether_weight * shape_tether_selection_stage_weight
            ),
            posenet_yuv6_geometry_tether_weight=(
                geometry_tether_weight * geometry_tether_selection_stage_weight
            ),
            posenet_temporal_signal_floor_weight=(
                temporal_floor_weight * temporal_floor_selection_stage_weight
            ),
            has_real_segnet_teacher=scorer_teacher is not None,
            has_real_posenet_teacher=pose_scorer_teacher is not None,
            coder_aware_qat_bound=bool(coder_qat_cfg.enabled),
            coder_qat_loss_weight_map=coder_qat_loss_weight_map,
            pr95_faithful_curriculum_enabled=bool(
                pr95_faithful_curriculum_enabled
            ),
        )
        metric_value_key = str(selection_policy["selection_metric_value_key"])
        selection_chunk_size = max(1, int(batch_pairs))

        def _mlx_scalar_to_float(value: Any) -> float:
            mx.eval(value)
            if hasattr(value, "item"):
                return float(value.item())
            return float(np.asarray(value).item())

        def _score_aware_selection_parts() -> tuple[float, dict[str, float], list[str]]:
            """Evaluate the full-video selector composite without updating params."""

            raw_parts: dict[str, float] = {}
            weighted_parts: dict[str, float] = {}
            blockers_for_row: list[str] = []
            n_pairs = int(pairs.shape[0])
            if n_pairs <= 0:
                return (
                    float("nan"),
                    {},
                    ["snerv_score_aware_checkpoint_selection_no_pairs"],
                )
            coder_terms = {
                str(name): float(weight)
                for name, weight in dict(coder_qat_loss_weight_map).items()
                if float(weight) != 0.0
            }
            coder_raw_parts: dict[str, float] = {}
            for start in range(0, n_pairs, selection_chunk_size):
                stop = min(n_pairs, start + selection_chunk_size)
                idx_np = np.arange(start, stop, dtype=np.int32)
                idx = mx.array(idx_np, dtype=mx.int32)
                try:
                    _total, parts = score_aware_loss(
                        bundle,
                        idx,
                        loss_weights=stage_loss_weights,
                    )
                except Exception as exc:  # fail closed; selection must not silently fall back.
                    return (
                        float("nan"),
                        {},
                        [
                            "snerv_score_aware_checkpoint_selection_loss_eval_failed",
                            f"snerv_score_aware_checkpoint_selection_exception_{type(exc).__name__}",
                        ],
                    )
                chunk_weight = float(stop - start) / float(n_pairs)
                for name, value in parts.items():
                    if name == "total":
                        continue
                    scalar = _mlx_scalar_to_float(value)
                    if not np.isfinite(scalar):
                        blockers_for_row.append(
                            "snerv_score_aware_checkpoint_selection_loss_part_nonfinite"
                        )
                        continue
                    if name in coder_terms:
                        coder_raw_parts.setdefault(str(name), scalar)
                    else:
                        raw_parts[str(name)] = raw_parts.get(str(name), 0.0) + (
                            chunk_weight * scalar
                        )
            for name, value in coder_raw_parts.items():
                raw_parts[name] = float(value)

            missing = [
                str(name)
                for name in selection_policy["required_loss_parts"]
                if str(name) not in raw_parts
                and str(name) != "pr95_stage_scorer_surrogate"
            ]
            if missing:
                blockers_for_row.extend(
                    [
                        "snerv_score_aware_checkpoint_selection_required_parts_missing",
                        *(
                            f"snerv_score_aware_checkpoint_selection_missing_{name}"
                            for name in missing
                        ),
                    ]
                )

            total = recon_selection_stage_weight * float(raw_parts.get("recon", 0.0))
            weighted_parts["recon"] = total
            if "distill" in raw_parts:
                weighted_parts["distill"] = (
                    seg_weight
                    * seg_selection_stage_weight
                    * raw_parts["distill"]
                )
                total += weighted_parts["distill"]
            if "segnet_direct_live_distill" in raw_parts:
                weighted_parts["segnet_direct_live_distill"] = (
                    direct_live_weight
                    * direct_live_selection_stage_weight
                    * raw_parts["segnet_direct_live_distill"]
                )
                total += weighted_parts["segnet_direct_live_distill"]
            if "pose_score_term" in raw_parts:
                weighted_parts["pose_score_term"] = (
                    pose_weight
                    * pose_selection_stage_weight
                    * raw_parts["pose_score_term"]
                )
                total += weighted_parts["pose_score_term"]
            elif pose_weight > 0.0 and "pose_distill" in raw_parts:
                blockers_for_row.append(
                    "snerv_score_aware_checkpoint_selection_pose_score_term_missing_raw_pose_mse_not_used"
                )
            if "pose_direct_live_score_term" in raw_parts:
                weighted_parts["pose_direct_live_score_term"] = (
                    pose_direct_live_weight
                    * pose_direct_live_selection_stage_weight
                    * raw_parts["pose_direct_live_score_term"]
                )
                total += weighted_parts["pose_direct_live_score_term"]
            elif pose_direct_live_weight > 0.0 and "pose_direct_live_distill" in raw_parts:
                weighted_parts["pose_direct_live_score_term"] = (
                    pose_direct_live_weight
                    * pose_direct_live_selection_stage_weight
                    * raw_parts["pose_direct_live_distill"]
                )
                total += weighted_parts["pose_direct_live_score_term"]
            elif pose_direct_live_weight > 0.0:
                blockers_for_row.append(
                    "snerv_score_aware_checkpoint_selection_pose_direct_live_score_term_missing"
                )
            if "scorer_input_distribution_guard" in raw_parts:
                weighted_parts["scorer_input_distribution_guard"] = (
                    float(bundle.scorer_input_distribution_guard_weight)
                    * guard_selection_stage_weight
                    * raw_parts["scorer_input_distribution_guard"]
                )
                total += weighted_parts["scorer_input_distribution_guard"]
            if "scorer_input_contrast_floor" in raw_parts:
                weighted_parts["scorer_input_contrast_floor"] = (
                    float(bundle.scorer_input_contrast_floor_weight)
                    * contrast_floor_selection_stage_weight
                    * raw_parts["scorer_input_contrast_floor"]
                )
                total += weighted_parts["scorer_input_contrast_floor"]
            if "scorer_input_shape_tether" in raw_parts:
                weighted_parts["scorer_input_shape_tether"] = (
                    float(bundle.scorer_input_shape_tether_weight)
                    * shape_tether_selection_stage_weight
                    * raw_parts["scorer_input_shape_tether"]
                )
                total += weighted_parts["scorer_input_shape_tether"]
            if "posenet_yuv6_geometry_tether" in raw_parts:
                weighted_parts["posenet_yuv6_geometry_tether"] = (
                    float(bundle.posenet_yuv6_geometry_tether_weight)
                    * geometry_tether_selection_stage_weight
                    * raw_parts["posenet_yuv6_geometry_tether"]
                )
                total += weighted_parts["posenet_yuv6_geometry_tether"]
            if "posenet_temporal_signal_floor" in raw_parts:
                weighted_parts["posenet_temporal_signal_floor"] = (
                    float(bundle.posenet_temporal_signal_floor_weight)
                    * temporal_floor_selection_stage_weight
                    * raw_parts["posenet_temporal_signal_floor"]
                )
                total += weighted_parts["posenet_temporal_signal_floor"]
            for name, weight in coder_terms.items():
                if name in raw_parts:
                    weighted_parts[name] = float(weight) * raw_parts[name]
                    total += weighted_parts[name]

            if "pr95_stage_scorer_surrogate" in selection_policy[
                "required_loss_parts"
            ]:
                stage_surrogate = float(total)
                if np.isfinite(stage_surrogate):
                    raw_parts["pr95_stage_scorer_surrogate"] = stage_surrogate
                    weighted_parts["pr95_stage_scorer_surrogate"] = stage_surrogate
                else:
                    blockers_for_row.append(
                        "snerv_score_aware_checkpoint_selection_pr95_stage_surrogate_nonfinite"
                    )

            if not np.isfinite(total):
                blockers_for_row.append(
                    "snerv_score_aware_checkpoint_selection_composite_nonfinite"
                )
            return (
                total,
                {
                    **{f"raw_{name}": float(value) for name, value in raw_parts.items()},
                    **{
                        f"weighted_{name}": float(value)
                        for name, value in weighted_parts.items()
                    },
                },
                _ordered_unique(blockers_for_row),
            )

        def _selection_metric_row(
            *,
            epoch: int,
            training_loss: float | None,
            state_source: str,
        ) -> dict[str, Any]:
            rendered = model.render_pairs_nchw255(batch_size=max(1, int(batch_pairs)))
            recon_mse = float(np.mean((rendered - pairs) ** 2))
            scorer_distortion = _snerv_scorer_domain_distortion_anatomy(
                rendered,
                pairs,
                source_pair_indices=source_pair_indices,
                worst_pair_count=0,
                comparison_domain="train_renderer_native_geometry",
            )
            row: dict[str, Any] = {
                "epoch": int(epoch),
                "state_source": str(state_source),
                "selection_metric": selection_policy["selection_metric"],
                "selection_metric_value_key": metric_value_key,
                "recon_mse_nchw255": recon_mse,
                "segnet_frame1_rgb_mse_nchw255": scorer_distortion[
                    "segnet_frame1_rgb_mse_nchw255"
                ],
                "posenet_yuv6_pair_mse": scorer_distortion["posenet_yuv6_pair_mse"],
                "posenet_yuv6_temporal_delta_mse": scorer_distortion[
                    "posenet_yuv6_temporal_delta_mse"
                ],
                "scorer_domain_distortion_anatomy": scorer_distortion,
                "training_loss": (
                    None if training_loss is None else float(training_loss)
                ),
                "selected_as_best": False,
            }
            if selection_policy["uses_score_aware_composite"]:
                composite, parts, blockers_for_row = _score_aware_selection_parts()
                blockers_for_row = _ordered_unique(
                    [
                        *(
                            str(blocker)
                            for blocker in selection_policy.get("blockers", ())
                            if str(blocker)
                        ),
                        *blockers_for_row,
                    ]
                )
                row.update(
                    {
                        "score_aware_composite_loss": composite,
                        "score_aware_composite_parts": parts,
                        "score_aware_checkpoint_selection_blockers": blockers_for_row,
                    }
                )
            else:
                row["score_aware_checkpoint_selection_blockers"] = []
            _snerv_checkpoint_selection_attach_support_metrics(row)
            if not np.isfinite(float(row.get(metric_value_key, float("nan")))):
                row["score_aware_checkpoint_selection_blockers"] = _ordered_unique(
                    [
                        *(
                            str(blocker)
                            for blocker in row.get(
                                "score_aware_checkpoint_selection_blockers",
                                (),
                            )
                            if str(blocker)
                        ),
                        "snerv_score_aware_checkpoint_selection_metric_nonfinite",
                    ]
                )
            return row

        initial_selection = _selection_metric_row(
            epoch=-1,
            training_loss=None,
            state_source="initial_closed_form_renderer",
        )
        initial_selection["selected_as_best"] = True
        best_selection = dict(initial_selection)
        selection_history.append(dict(initial_selection))
        initial_selection_blockers = _ordered_unique(
            [
                *(
                    str(blocker)
                    for blocker in selection_policy.get("blockers") or ()
                    if str(blocker)
                ),
                *(
                    str(blocker)
                    for blocker in initial_selection.get(
                        "score_aware_checkpoint_selection_blockers",
                        (),
                    )
                    if str(blocker)
                ),
            ]
        )
        if selection_policy["uses_score_aware_composite"] and initial_selection_blockers:
            selection_failures.extend(initial_selection_blockers)

        def _maybe_select_current_renderer(
            *,
            epoch: int,
            training_loss: float | None,
            state_source: str,
        ) -> dict[str, Any]:
            nonlocal best_selection, best_state
            row = _selection_metric_row(
                epoch=epoch,
                training_loss=training_loss,
                state_source=state_source,
            )
            row_blockers = [
                str(blocker)
                for blocker in row.get("score_aware_checkpoint_selection_blockers", ())
                if str(blocker)
            ]
            if selection_policy["uses_score_aware_composite"] and row_blockers:
                selection_failures.extend(row_blockers)
            if _snerv_checkpoint_selection_row_is_better(
                row,
                best_selection,
                metric_value_key=metric_value_key,
            ):
                best_state = model.export_state_dict()
                row["selected_as_best"] = True
                best_selection = dict(row)
            selection_history.append(row)
            return row

        def _on_epoch_end(metrics: Any) -> None:
            epoch = int(metrics.epoch)
            if (epoch + 1) % selection_interval_epochs != 0 and (
                epoch + 1
            ) < int(requested_epochs):
                return
            _maybe_select_current_renderer(
                epoch=epoch,
                training_loss=float(metrics.loss),
                state_source="post_epoch_interval_or_final",
            )
        artifact = run_mlx_score_aware_full_main(
            bundle=bundle,
            substrate_id="snerv_inverse_steg_carrier",
            lane_id="lane_snerv_mlx_score_aware_train_export",
            output_dir=out / "long_training",
            epochs=int(requested_epochs),
            batch_pair_indices_per_step=max(1, int(batch_pairs)),
            learning_rate=float(learning_rate),
            seed=0,
            checkpoint_interval_epochs=max(1, min(100, int(requested_epochs))),
            checkpoint_retention_keep_last_n=checkpoint_retention_keep_last_n,
            checkpoint_retention_keep_best_n=checkpoint_retention_keep_best_n,
            checkpoint_retention_keep_every_n_epochs=checkpoint_retention_keep_every_n_epochs,
            checkpoint_retention_cold_store_roots=checkpoint_retention_cold_store_roots,
            curriculum_stages=score_aware_curriculum_stages,
            telemetry_flush_interval_epochs=1,
            grad_clip_max_norm=grad_clip_max_norm,
            weight_decay=weight_decay,
            optimizer_kind=str(optimizer_kind),
            pr95_faithful_curriculum_enabled=bool(
                pr95_faithful_curriculum_enabled
            ),
            pr95_curriculum_total_epochs=(
                max(8, int(requested_epochs))
                if pr95_faithful_curriculum_enabled
                else None
            ),
            pr95_muon_policy=effective_pr95_muon_policy,
            pr95_stage_source_weight_amplification_enabled=bool(
                pr95_stage_source_weight_amplification_enabled
            ),
            pr95_force_weighted_extra_qat_when_stage_inactive=bool(
                coder_qat_cfg.enabled
                and train_time_section_byte_control.get("active") is True
            ),
            prioritized_pair_indices=tuple(
                int(value) for value in prioritized_pair_indices
            ),
            pair_sampling_weights=scorer_error_pair_sampling_weights,
            pair_sampling_default_weight=float(
                scorer_error_pair_curriculum.get("default_weight", 1.0)
            ),
            gradient_multiplier_by_name=gradient_multiplier_by_name,
            bias_gradient_multiplier=bias_gradient_multiplier,
            output_head_bias_gradient_multiplier=float(
                output_head_bias_gradient_multiplier
            ),
            scorer_space_step_guard_enabled=scorer_space_step_guard_enabled,
            scorer_space_step_guard_min_pre_segnet_occupied_class_fraction=(
                scorer_space_step_guard_min_pre_segnet_occupied_class_fraction
            ),
            scorer_space_step_guard_min_post_segnet_occupied_class_fraction=(
                scorer_space_step_guard_min_post_segnet_occupied_class_fraction
            ),
            scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction=(
                scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction
            ),
            scorer_space_step_guard_min_post_segnet_target_class_min_ratio=(
                scorer_space_step_guard_min_post_segnet_target_class_min_ratio
            ),
            scorer_space_step_guard_max_post_segnet_target_class_ratio_drop=(
                scorer_space_step_guard_max_post_segnet_target_class_ratio_drop
            ),
            scorer_space_step_guard_max_post_segnet_contrast_ratio=(
                score_aware_long_training_scorer_space_step_guard_max_post_segnet_contrast_ratio
            ),
            scorer_space_step_guard_max_post_segnet_distribution_mae=(
                score_aware_long_training_scorer_space_step_guard_max_post_segnet_distribution_mae
            ),
            scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae=(
                score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae
            ),
            scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio=(
                score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio
            ),
            scorer_space_step_guard_max_post_segnet_argmax_disagreement=(
                score_aware_long_training_scorer_space_step_guard_max_post_segnet_argmax_disagreement
            ),
            scorer_space_step_guard_max_post_pose_score_term=(
                score_aware_long_training_scorer_space_step_guard_max_post_pose_score_term
            ),
            scorer_space_step_guard_max_post_pose_direct_live_score_term=(
                score_aware_long_training_scorer_space_step_guard_max_post_pose_direct_live_score_term
            ),
            scorer_space_step_guard_max_pose_score_term_relative_worsening=(
                score_aware_long_training_scorer_space_step_guard_max_pose_score_term_relative_worsening
            ),
            scorer_space_step_guard_max_pose_score_term_absolute_worsening=(
                score_aware_long_training_scorer_space_step_guard_max_pose_score_term_absolute_worsening
            ),
            scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening=(
                score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening
            ),
            scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening=(
                score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening
            ),
            scorer_space_step_guard_max_direct_nonrate_score_worsening=(
                score_aware_long_training_scorer_space_step_guard_max_direct_nonrate_score_worsening
            ),
            scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening=(
                score_aware_long_training_scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening
            ),
            scorer_space_step_guard_backtracking_steps=(
                scorer_space_step_guard_backtracking_steps
            ),
            scorer_space_step_guard_backtracking_shrink=(
                scorer_space_step_guard_backtracking_shrink
            ),
            train_time_dual_ascent_config=train_time_dual_ascent_config,
            notes=(
                "SNeRV MLX score-aware train/export attachment: train LF "
                "latents or official MFU/HFR/TUB receiver payload atoms with "
                "the canonical shared MLX harness, then export trained bytes "
                "through the NumPy-portable SNAR1 receiver packet builder."
            ),
            on_epoch_end=_on_epoch_end,
        )
        live_final_pairs = model.render_pairs_nchw255(batch_size=max(1, int(batch_pairs)))
        live_final_mse = float(np.mean((live_final_pairs - pairs) ** 2))
        live_final_selection = _maybe_select_current_renderer(
            epoch=int(artifact.as_dict().get("total_epochs_completed") or 0) - 1,
            training_loss=None,
            state_source="live_final_post_training",
        )
        model.import_state_dict(best_state)
        trained_pairs = model.render_pairs_nchw255(batch_size=max(1, int(batch_pairs)))
        final_mse = float(np.mean((trained_pairs - pairs) ** 2))
        artifact_dict = artifact.as_dict()
        telemetry_path = str(artifact_dict.get("telemetry_path") or "")
        live_checkpoint_path = str(artifact_dict.get("live_checkpoint_path") or "")
        ema_checkpoint_path = str(
            artifact_dict.get("ema_shadow_checkpoint_path") or ""
        )
        latest_training_metrics = _latest_snerv_score_aware_training_metrics(
            telemetry_path
        )
        blockers: list[str] = []
        training_telemetry_contract = (
            _snerv_score_aware_long_training_telemetry_contract(
                telemetry_path,
                segnet_distillation_weight=seg_weight,
                pose_distillation_weight=pose_weight,
                pose_direct_live_distillation_weight=pose_direct_live_weight,
                segnet_student_live_calibration_weight=live_calibration_weight,
                segnet_direct_live_distillation_weight=direct_live_weight,
                segnet_direct_live_class_histogram_weight=(
                    direct_live_histogram_weight
                ),
                segnet_direct_live_class_balanced_hinge_weight=(
                    direct_live_balanced_hinge_weight
                ),
                segnet_direct_live_class_balanced_ce_weight=(
                    direct_live_balanced_ce_weight
                ),
                segnet_direct_live_class_balanced_squared_hinge_weight=(
                    direct_live_balanced_squared_hinge_weight
                ),
                segnet_direct_live_class_region_recon_weight=(
                    direct_live_class_region_recon_weight
                ),
                segnet_direct_live_rare_class_logit_weight=(
                    direct_live_rare_class_logit_weight
                ),
                segnet_direct_live_target_mass_floor_weight=(
                    direct_live_target_mass_floor_weight
                ),
                segnet_direct_live_target_min_ratio_floor_weight=(
                    direct_live_target_min_ratio_floor_weight
                ),
                pr95_faithful_curriculum_enabled=bool(
                    pr95_faithful_curriculum_enabled
                ),
                coder_aware_qat_bound=bool(coder_qat_cfg.enabled),
                train_time_section_byte_control_bound=bool(
                    train_time_section_byte_control.get("active") is True
                ),
                scorer_input_distribution_guard_weight=guard_weight,
                scorer_input_contrast_floor_weight=contrast_floor_weight,
                scorer_input_shape_tether_weight=shape_tether_weight,
                posenet_yuv6_geometry_tether_weight=geometry_tether_weight,
                posenet_temporal_signal_floor_weight=temporal_floor_weight,
                gradient_multiplier_controls_requested=bool(
                    base_payload["gradient_multiplier_controls"]["enabled"]
                ),
                scorer_space_step_guard_enabled=scorer_space_step_guard_enabled,
            )
        )
        blockers.extend(training_telemetry_contract.get("blockers") or ())
        blockers.extend(
            _snerv_live_section_byte_metrics_blockers(
                live_train_time_section_byte_metrics_metadata,
                train_time_section_byte_control_bound=bool(
                    train_time_section_byte_control.get("active") is True
                ),
            )
        )
        if not np.isfinite(final_mse):
            blockers.append("snerv_score_aware_long_training_selected_mse_nonfinite")
        if selection_policy["uses_score_aware_composite"]:
            selection_blockers = _ordered_unique(
                [
                    *(
                        str(blocker)
                        for blocker in selection_policy.get("blockers") or ()
                        if str(blocker)
                    ),
                    *selection_failures,
                    *(
                        str(blocker)
                        for blocker in best_selection.get(
                            "score_aware_checkpoint_selection_blockers",
                            (),
                        )
                        if str(blocker)
                    ),
                ]
            )
            blockers.extend(selection_blockers)
        if int(artifact_dict.get("total_epochs_completed") or 0) < int(requested_epochs):
            blockers.append("snerv_score_aware_long_training_early_stopped")
        selection_warnings = []
        if (
            np.isfinite(live_final_mse)
            and np.isfinite(final_mse)
            and live_final_mse > final_mse + 1.0e-9
        ):
            selection_warnings.append(
                "snerv_score_aware_long_training_live_final_worse_than_selected"
            )
        if _snerv_checkpoint_selection_row_is_better(
            best_selection,
            live_final_selection,
            metric_value_key=metric_value_key,
        ):
            selection_warnings.append(
                "snerv_score_aware_long_training_live_final_selection_metric_worse_than_selected"
            )
        if str(best_selection.get("state_source")) == "initial_closed_form_renderer":
            selection_warnings.append(
                "snerv_score_aware_long_training_selected_initial_no_improvement"
            )
        trained_state_exportable = bool(
            np.isfinite(final_mse)
            and isinstance(trained_pairs, np.ndarray)
            and tuple(int(dim) for dim in trained_pairs.shape)
            == tuple(int(dim) for dim in pairs.shape)
        )
        official_packet: SnervArchivePacket | None = None
        official_train_export = dict(base_payload["official_mfu_hfr_tub_train_export"])
        if official_training_requested:
            official_train_export.update(
                {
                    "train_renderer_bound": True,
                    "trained_receiver_payload_exported": False,
                    "trained_receiver_state_bound": False,
                    "trained_receiver_state_mapping_scope": (
                        "train_renderer_bound_before_receiver_payload_export"
                    ),
                    "trained_weight_mapping_to_long_training_bound": (
                        official_checkpoint_mapping_verified
                    ),
                    "official_trained_checkpoint_state_dict_loaded": (
                        official_checkpoint_loaded
                    ),
                    "official_trained_checkpoint_state_dict_mapping_verified": (
                        official_checkpoint_mapping_verified
                    ),
                    "official_trained_checkpoint_mapping_manifest": (
                        trained_checkpoint_mapping_manifest
                    ),
                    "official_trained_checkpoint_source_forward_replay_verified": False,
                    "train_renderer_schema": renderer_schema,
                    "source_forward_replay_authority": False,
                }
            )
        payload = {
            **base_payload,
            "executed": True,
            "training_completed": True,
            "blocker_free_execution": not blockers,
            "trained_state_exportable": trained_state_exportable,
            "trained_pairs_materialized": trained_state_exportable,
            "epochs_completed": int(artifact_dict.get("total_epochs_completed") or 0),
            "learning_rate": float(learning_rate),
            "batch_pairs": max(1, int(batch_pairs)),
            "grad_clip_max_norm": (
                None if grad_clip_max_norm is None else float(grad_clip_max_norm)
            ),
            "weight_decay": None if weight_decay is None else float(weight_decay),
            "eval_roundtrip_ste_enabled": bool(eval_roundtrip_ste),
            "scorer_input_distribution_guard": {
                **dict(base_payload["scorer_input_distribution_guard"]),
                "bound_to_renderer_bundle": guard_weight > 0.0,
            },
            "scorer_input_distribution_guard_bound": guard_weight > 0.0,
            "scorer_input_contrast_floor": {
                **dict(base_payload["scorer_input_contrast_floor"]),
                "bound_to_renderer_bundle": contrast_floor_weight > 0.0,
            },
            "scorer_input_contrast_floor_bound": contrast_floor_weight > 0.0,
            "scorer_input_shape_tether": {
                **dict(base_payload["scorer_input_shape_tether"]),
                "bound_to_renderer_bundle": shape_tether_weight > 0.0,
            },
            "scorer_input_shape_tether_bound": shape_tether_weight > 0.0,
            "posenet_yuv6_geometry_tether": {
                **dict(base_payload["posenet_yuv6_geometry_tether"]),
                "bound_to_renderer_bundle": geometry_tether_weight > 0.0,
            },
            "posenet_yuv6_geometry_tether_bound": geometry_tether_weight > 0.0,
            "posenet_temporal_signal_floor": {
                **dict(base_payload["posenet_temporal_signal_floor"]),
                "bound_to_renderer_bundle": temporal_floor_weight > 0.0,
            },
            "posenet_temporal_signal_floor_bound": temporal_floor_weight > 0.0,
            "scorer_space_step_guard": {
                **dict(base_payload["scorer_space_step_guard"]),
                "bound_to_shared_mlx_adapter": scorer_space_step_guard_enabled,
            },
            "scorer_space_step_guard_bound": scorer_space_step_guard_enabled,
            "pr95_faithful_curriculum_enabled": bool(
                pr95_faithful_curriculum_enabled
            ),
            "pr95_muon_policy_requested": str(pr95_muon_policy),
            "pr95_muon_policy": effective_pr95_muon_policy,
            "pr95_stage_source_weight_amplification_enabled": bool(
                pr95_stage_source_weight_amplification_enabled
            ),
            "pr95_optimizer_coverage": pr95_optimizer_coverage,
            "optimizer_binding_mode": (
                "pr95_faithful_curriculum"
                if pr95_faithful_curriculum_enabled
                else str(optimizer_kind)
            ),
            "pr95_faithful_optimizer_schedule_bound": bool(
                pr95_faithful_curriculum_enabled
            ),
            "pact_native_muon_adamw_partition_bound": bool(
                str(optimizer_kind) == "pact_muon_adamw"
                and not pr95_faithful_curriculum_enabled
            ),
            "muon_adamw_partition_bound": bool(
                str(optimizer_kind) == "pact_muon_adamw"
                or pr95_faithful_curriculum_enabled
            ),
            "coder_aware_qat": coder_qat_metadata(coder_qat_cfg),
            "coder_aware_qat_bound": bool(coder_qat_cfg.enabled),
            "archive_section_qat_weight_policy": archive_section_qat_policy,
            "archive_section_qat_weight_policy_bound": bool(
                archive_section_qat_policy.get("active") is True
            ),
            "train_time_section_byte_control": train_time_section_byte_control,
            "train_time_section_byte_control_bound": bool(
                train_time_section_byte_control.get("active") is True
            ),
            "training_telemetry_contract": training_telemetry_contract,
            "latest_training_metrics": latest_training_metrics,
            "telemetry_path": telemetry_path,
            "live_checkpoint_path": live_checkpoint_path,
            "ema_shadow_checkpoint_path": ema_checkpoint_path,
            "train_time_archive_bytes": latest_training_metrics.get(
                "train_time_archive_bytes"
            ),
            "train_time_section_bytes": latest_training_metrics.get(
                "train_time_section_bytes"
            ),
            "latest_scorer_space_step_guard": latest_training_metrics.get(
                "scorer_space_step_guard"
            ),
            "latest_scorer_deltas": latest_training_metrics.get("scorer_deltas"),
            "live_train_time_section_byte_metrics": (
                live_train_time_section_byte_metrics_metadata
            ),
            "latent_qat_bound": bool(latent_qat_cfg.enabled),
            "teacher_binding": teacher_binding,
            "official_mfu_hfr_tub_train_export": official_train_export,
            **(
                {
                    "official_mfu_hfr_tub_source_forward_replay": official_source_forward_replay,
                }
                if official_source_forward_replay is not None
                else {}
            ),
            "has_real_segnet_teacher": scorer_teacher is not None,
            "has_real_posenet_teacher": pose_scorer_teacher is not None,
            "renderer": model.metadata(),
            "initial_recon_mse_nchw255": initial_mse,
            "live_final_recon_mse_nchw255": live_final_mse,
            "final_recon_mse_nchw255": final_mse,
            "loss_delta_nchw255": final_mse - initial_mse,
            "checkpoint_selection_policy": selection_policy,
            "best_checkpoint_selection": best_selection,
            "selection_interval_epochs": int(selection_interval_epochs),
            "selection_history_tail": selection_history[-8:],
            "selection_failures": _ordered_unique(selection_failures),
            "selection_warnings": selection_warnings,
            "training_artifact": {
                "schema": artifact_dict.get("schema"),
                "substrate_id": artifact_dict.get("substrate_id"),
                "lane_id": artifact_dict.get("lane_id"),
                "total_epochs_completed": artifact_dict.get("total_epochs_completed"),
                "total_wall_clock_seconds": artifact_dict.get("total_wall_clock_seconds"),
                "telemetry_path": telemetry_path,
                "live_checkpoint_path": live_checkpoint_path,
                "ema_shadow_checkpoint_path": ema_checkpoint_path,
                "checkpoint_selection": artifact_dict.get("checkpoint_selection"),
                "archive_path": artifact_dict.get("archive_path"),
                "archive_bytes": artifact_dict.get("archive_bytes"),
                "score_claim": artifact_dict.get("score_claim"),
                "promotion_eligible": artifact_dict.get("promotion_eligible"),
                "ready_for_exact_eval_dispatch": artifact_dict.get(
                    "ready_for_exact_eval_dispatch"
                ),
            },
            "blockers": _ordered_unique(blockers),
        }
        if official_training_requested and trained_state_exportable:
            trained_official_train_export = {
                **official_train_export,
                "trained_receiver_payload_exported": True,
                "trained_receiver_state_bound": True,
                "trained_receiver_state_mapping_scope": (
                    "upstream_official_state_dict_bound_to_mlx_receiver_component_state"
                    if official_checkpoint_mapping_verified
                    else "mlx_receiver_component_state_not_upstream_official_state_dict"
                ),
                "trained_weight_mapping_to_long_training_bound": (
                    official_checkpoint_mapping_verified
                ),
                "official_trained_checkpoint_state_dict_loaded": (
                    official_checkpoint_loaded
                ),
                "official_trained_checkpoint_state_dict_mapping_verified": (
                    official_checkpoint_mapping_verified
                ),
                "official_trained_checkpoint_mapping_manifest": (
                    trained_checkpoint_mapping_manifest
                ),
                "official_trained_checkpoint_source_forward_replay_verified": False,
                "authority_blockers": [
                    *(
                        []
                        if official_checkpoint_mapping_verified
                        else [SNERV_OFFICIAL_MFU_HFR_TUB_WEIGHT_MAPPING_BLOCKER]
                    ),
                    *(
                        []
                        if official_checkpoint_mapping_verified
                        else [SNERV_OFFICIAL_TRAINED_CHECKPOINT_STATE_DICT_MAPPING_BLOCKER]
                    ),
                    SNERV_OFFICIAL_TRAINED_CHECKPOINT_SOURCE_FORWARD_BLOCKER,
                ],
            }
            payload_for_packet = {
                **payload,
                "official_mfu_hfr_tub_train_export": trained_official_train_export,
            }
            official_packet = _build_official_mfu_hfr_tub_packet_from_components(
                model.export_official_components(),
                source_pair_indices=source_pair_indices,
                model_size=model_size,
                metadata_extra={
                    "source_pair_indices": [int(value) for value in source_pair_indices],
                    "source_pair_indices_preserved": True,
                    "allocation_mode": "official_mfu_hfr_tub_trained_mlx_receiver_payload",
                    "hf_decoder_fit_mode": "official_mfu_hfr_tub_mlx_trained_payload_atoms",
                    "score_aware_long_training": payload_for_packet,
                    "score_aware_long_training_executed": bool(
                        payload.get("executed") is True
                    ),
                    "score_aware_long_training_trained_state_exportable": True,
                    "score_aware_long_training_kind": training_kind,
                    "score_aware_long_training_optimizer": str(optimizer_kind),
                    "native_mlx_training_executed": True,
                    "native_mlx_training_kind": training_kind,
                    **(
                        {
                            "official_mfu_hfr_tub_source_forward_replay": (
                                official_source_forward_replay
                            ),
                            "snerv_official_tub_source_fixture_binding": (
                                _official_tub_source_fixture_binding(
                                    official_source_forward_replay
                                )
                            ),
                        }
                        if official_source_forward_replay is not None
                        else {}
                    ),
                    "official_mfu_hfr_tub_train_export": trained_official_train_export,
                },
            )
            official_packet_export_metadata = {
                str(key): value
                for key, value in official_packet.metadata.items()
                if str(key).startswith("official_tub_output2_")
                or str(key).startswith("snerv_official_tub_source_")
                or str(key) == "official_source_parity_blockers"
            }
            payload["official_mfu_hfr_tub_train_export"] = {
                **trained_official_train_export,
                "trained_packet_bytes": int(official_packet.total_bytes),
                "trained_packet_sha256": _sha256_bytes(official_packet.packet),
                **official_packet_export_metadata,
            }
        write_json(report_path, payload)
        result = {
            **payload,
            "report_path": report_path.as_posix(),
            "_trained_pairs_nchw255": trained_pairs,
        }
        if official_packet is not None:
            result["_trained_official_packet"] = official_packet
        return result
    except Exception as exc:
        payload = {
            **base_payload,
            "training_completed": False,
            "failure": repr(exc),
            "blockers": ["snerv_score_aware_long_training_failed"],
        }
        write_json(report_path, payload)
        return {**payload, "report_path": report_path.as_posix()}


def _build_official_mfu_hfr_tub_long_training_replay_contract(
    *,
    output_dir: str | Path,
    pairs_nchw255: np.ndarray,
    model_size: SnervModelSizeConfig,
    source_pair_indices: Sequence[int],
    official_trained_checkpoint_mapping_manifest: Mapping[str, Any] | None = None,
    allow_overwrite: bool,
) -> dict[str, Any]:
    """Write the executable official-payload replay proof for long-training refusal.

    This is intentionally narrower than official SNeRV parity: it proves that
    the native export can materialize and replay a frame-producing official
    MFU/HFR/TUB receiver payload for the requested targets, while keeping the
    differentiable MLX training graph and upstream-source forward replay as
    explicit blockers.
    """

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / "snerv_official_mfu_hfr_tub_source_forward_replay_contract.json"
    if artifact_path.exists() and not allow_overwrite:
        raise SnervMlxNativeExportError(
            f"refusing to overwrite existing official source-forward replay contract: {artifact_path}"
        )
    pairs = np.asarray(pairs_nchw255, dtype=np.float32)
    trained_checkpoint_mapping_manifest = _coerce_official_checkpoint_mapping_manifest(
        official_trained_checkpoint_mapping_manifest,
        source="snerv_mlx_native_train_export_replay_contract",
    )
    official_checkpoint_loaded = bool(
        trained_checkpoint_mapping_manifest.get("official_trained_checkpoint_loaded")
        is True
    )
    official_checkpoint_mapping_verified = bool(
        trained_checkpoint_mapping_manifest.get(
            "official_mfu_hfr_trained_checkpoint_weight_mapping_proven"
        )
        is True
        and trained_checkpoint_mapping_manifest.get(
            "official_tub_temporal_encoder_weight_mapping_proven"
        )
        is True
    )
    base: dict[str, Any] = {
        "schema": "snerv_official_mfu_hfr_tub_source_forward_replay_contract.v1",
        "family": "snerv",
        "adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        "requested_pair_count": int(pairs.shape[0]) if pairs.ndim >= 1 else 0,
        "source_pair_indices": [int(value) for value in source_pair_indices],
        "source_forward_replay_bound": False,
        "source_forward_replay_verified": False,
        "source_forward_replay_authority": False,
        "score_aware_long_training_renderer_bound": False,
        "train_renderer_bound": False,
        "trained_receiver_state_bound": False,
        "trained_receiver_state_mapping_scope": "none",
        "trained_weight_mapping_to_long_training_bound": False,
        "official_trained_checkpoint_loaded": official_checkpoint_loaded,
        "official_trained_checkpoint_state_dict_mapping_verified": (
            official_checkpoint_mapping_verified
        ),
        "official_trained_checkpoint_mapping_manifest": (
            trained_checkpoint_mapping_manifest
        ),
        "official_trained_checkpoint_source_forward_replay_verified": False,
        "receiver_official_payload_forward_replay_passed": False,
        "official_torch_source_forward_replay_passed": False,
        "component_rows": [],
        **FALSE_AUTHORITY,
    }
    try:
        tub_fixture_replay = _build_official_tub_fixture_replay_for_long_training()
        packet = _build_official_mfu_hfr_tub_packet_from_numpy_pairs(
            pairs,
            source_pair_indices=tuple(int(value) for value in source_pair_indices),
            model_size=model_size,
            metadata_extra={
                "source_forward_replay_contract_probe": True,
                "source_faithful_stack": False,
            },
        )
        selected_authority = _selected_packet_official_payload_authority(packet.packet)
        receiver_frame_replay = selected_authority.get(
            "official_receiver_payload_frame_replay"
        )
        receiver_frame_replay = (
            dict(receiver_frame_replay)
            if isinstance(receiver_frame_replay, Mapping)
            else {}
        )
        tensor_map = _official_receiver_tensor_map_from_packet(packet.packet)
        decoded = unpack_snerv_archive(packet.packet)
        primitive_proof = execute_official_mfu_hfr_tub_decoder_payload(
            decoded.sections["decoder_payload"]
        )
        frames = decode_snerv_archive_frames(packet.packet)
        shape_matches = tuple(int(value) for value in frames.shape) == tuple(
            int(value) for value in pairs.shape
        )
        finite = bool(np.isfinite(frames).all())
        max_abs_error = (
            float(np.max(np.abs(np.asarray(frames, dtype=np.float32) - pairs)))
            if shape_matches and finite
            else None
        )
        replay_passed = bool(
            selected_authority.get("frame_producing_official_export") is True
            and selected_authority.get("receiver_payload_frame_replay_passed") is True
            and receiver_frame_replay.get("receiver_payload_frame_replay_passed") is True
            and tensor_map.get("receiver_tensor_map_verified") is True
            and primitive_proof.get("receiver_runtime_decode_proven") is True
            and shape_matches
            and finite
            and max_abs_error is not None
            and max_abs_error <= 5.0e-2
        )
        component_rows = _official_long_training_replay_component_rows(
            primitive_proof=primitive_proof,
            tensor_map=tensor_map,
            receiver_replay_passed=replay_passed,
            tub_fixture_replay=tub_fixture_replay,
        )
        source_forward_blockers = _filter_official_source_forward_blockers_for_mapping(
            _official_source_forward_blockers_from_tub_fixture(tub_fixture_replay),
            trained_checkpoint_mapping_manifest,
        )
        blockers = [
            "" if replay_passed else "snerv_official_mfu_hfr_tub_receiver_payload_replay_failed",
            "snerv_score_aware_long_training_official_mfu_hfr_tub_differentiable_mlx_renderer_missing",
            "snerv_official_mfu_hfr_tub_trained_weight_mapping_to_long_training_missing",
            ""
            if official_checkpoint_mapping_verified
            else SNERV_OFFICIAL_TRAINED_CHECKPOINT_STATE_DICT_MAPPING_BLOCKER,
            SNERV_OFFICIAL_TRAINED_CHECKPOINT_SOURCE_FORWARD_BLOCKER,
            *source_forward_blockers,
        ]
        payload = {
            **base,
            "packet_bytes": len(packet.packet),
            "packet_sha256": _sha256_bytes(packet.packet),
            "selected_packet_authority": selected_authority,
            "official_receiver_payload_frame_replay": receiver_frame_replay,
            "receiver_payload_frame_replay_passed": bool(
                receiver_frame_replay.get("receiver_payload_frame_replay_passed")
                is True
            ),
            "official_receiver_tensor_map": tensor_map,
            "official_receiver_runtime_decode_proof": primitive_proof,
            "official_tub_source_forward_fixture_replay": tub_fixture_replay,
            "official_tub_fixture_source_forward_replay_proven": _official_tub_fixture_replay_passed(
                tub_fixture_replay
            ),
            "decoded_frame_shape": [int(value) for value in frames.shape],
            "target_frame_shape": [int(value) for value in pairs.shape],
            "decoded_frames_finite": finite,
            "frame_shape_matches": shape_matches,
            "max_abs_error_nchw255": max_abs_error,
            "receiver_official_payload_forward_replay_passed": replay_passed,
            "component_rows": component_rows,
            "blockers": _ordered_unique(str(blocker) for blocker in blockers if blocker),
        }
    except Exception as exc:
        tub_fixture_replay = _build_official_tub_fixture_replay_for_long_training()
        payload = {
            **base,
            "failure": f"{type(exc).__name__}: {exc}",
            "official_tub_source_forward_fixture_replay": tub_fixture_replay,
            "official_tub_fixture_source_forward_replay_proven": _official_tub_fixture_replay_passed(
                tub_fixture_replay
            ),
            "blockers": [
                "snerv_official_mfu_hfr_tub_receiver_payload_replay_failed",
                "snerv_score_aware_long_training_official_mfu_hfr_tub_differentiable_mlx_renderer_missing",
                *_filter_official_source_forward_blockers_for_mapping(
                    _official_source_forward_blockers_from_tub_fixture(
                        tub_fixture_replay
                    ),
                    trained_checkpoint_mapping_manifest,
                ),
            ],
        }
    payload.update(FALSE_AUTHORITY)
    write_json(artifact_path, payload)
    payload["artifact_path"] = artifact_path.as_posix()
    payload["artifact_sha256"] = sha256_file(artifact_path)
    write_json(artifact_path, payload)
    return payload


def _build_deferred_official_mfu_hfr_tub_long_training_replay_contract(
    *,
    output_dir: str | Path,
    pair_count: int,
    source_pair_indices: Sequence[int],
    official_trained_checkpoint_mapping_manifest: Mapping[str, Any] | None = None,
    allow_overwrite: bool,
) -> dict[str, Any]:
    """Write a fail-closed replay contract without blocking full-video training."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / "snerv_official_mfu_hfr_tub_source_forward_replay_contract.json"
    if artifact_path.exists() and not allow_overwrite:
        raise SnervMlxNativeExportError(
            f"refusing to overwrite existing official source-forward replay contract: {artifact_path}"
        )
    blocker = "snerv_official_mfu_hfr_tub_receiver_payload_replay_deferred_full_video"
    tub_fixture_replay = _build_official_tub_fixture_replay_for_long_training()
    trained_checkpoint_mapping_manifest = _coerce_official_checkpoint_mapping_manifest(
        official_trained_checkpoint_mapping_manifest,
        source="snerv_mlx_native_train_export_deferred_replay_contract",
    )
    official_checkpoint_loaded = bool(
        trained_checkpoint_mapping_manifest.get("official_trained_checkpoint_loaded")
        is True
    )
    official_checkpoint_mapping_verified = bool(
        trained_checkpoint_mapping_manifest.get(
            "official_mfu_hfr_trained_checkpoint_weight_mapping_proven"
        )
        is True
        and trained_checkpoint_mapping_manifest.get(
            "official_tub_temporal_encoder_weight_mapping_proven"
        )
        is True
    )
    source_forward_blockers = _filter_official_source_forward_blockers_for_mapping(
        _official_source_forward_blockers_from_tub_fixture(tub_fixture_replay),
        trained_checkpoint_mapping_manifest,
    )
    tub_fixture_passed = _official_tub_fixture_replay_passed(tub_fixture_replay)
    tub_source_blockers = (
        _official_tub_fixture_preserved_blockers(tub_fixture_replay)
        if tub_fixture_passed
        else ["snerv_official_mfu_hfr_tub_source_forward_replay_missing"]
    )
    payload: dict[str, Any] = {
        "schema": "snerv_official_mfu_hfr_tub_source_forward_replay_contract.v1",
        "family": "snerv",
        "adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        "requested_pair_count": int(pair_count),
        "source_pair_indices": [int(value) for value in source_pair_indices],
        "source_forward_replay_bound": False,
        "source_forward_replay_verified": False,
        "source_forward_replay_authority": False,
        "score_aware_long_training_renderer_bound": False,
        "train_renderer_bound": False,
        "trained_receiver_state_bound": False,
        "trained_receiver_state_mapping_scope": "none",
        "trained_weight_mapping_to_long_training_bound": False,
        "official_trained_checkpoint_loaded": official_checkpoint_loaded,
        "official_trained_checkpoint_state_dict_mapping_verified": (
            official_checkpoint_mapping_verified
        ),
        "official_trained_checkpoint_mapping_manifest": (
            trained_checkpoint_mapping_manifest
        ),
        "official_trained_checkpoint_source_forward_replay_verified": False,
        "receiver_official_payload_forward_replay_passed": False,
        "official_torch_source_forward_replay_passed": False,
        "official_tub_source_forward_fixture_replay": tub_fixture_replay,
        "official_tub_fixture_source_forward_replay_proven": tub_fixture_passed,
        "deferred_for_full_video_training_start": True,
        "defer_threshold_pairs": int(SNERV_OFFICIAL_LONG_TRAINING_REPLAY_MAX_PAIRS),
        "defer_reason": (
            "full_video_score_aware_training_must_not_block_on_pretraining_"
            "receiver_payload_replay"
        ),
        "selected_packet_authority": {
            "status": "deferred_until_export_receiver_replay",
            "frame_producing_official_export": False,
            **FALSE_AUTHORITY,
        },
        "official_receiver_tensor_map": {
            "schema": "snerv_official_receiver_tensor_map.v1",
            "receiver_tensor_map_verified": False,
            "deferred_for_full_video_training_start": True,
            "blockers": [blocker],
            **FALSE_AUTHORITY,
        },
        "official_receiver_runtime_decode_proof": {
            "schema": "snerv_official_receiver_runtime_decode_contract.v1",
            "receiver_runtime_decode_proven": False,
            "deferred_for_full_video_training_start": True,
            "blockers": [blocker],
            **FALSE_AUTHORITY,
        },
        "component_rows": [
            {
                "schema": "snerv_official_mfu_hfr_tub_long_training_replay_component.v1",
                "component_id": component_id,
                "receiver_payload_forward_replay_proven": False,
                "receiver_tensor_payload_present": False,
                "official_source_forward_parity_proven": False,
                "score_aware_long_training_renderer_bound": False,
                "train_renderer_bound": False,
                "trained_receiver_state_bound": False,
                "official_trained_checkpoint_state_dict_mapping_verified": False,
                "deferred_for_full_video_training_start": True,
                "blockers": [
                    "snerv_score_aware_long_training_official_mfu_hfr_tub_differentiable_mlx_renderer_missing",
                    blocker,
                    *(
                        tub_source_blockers
                        if component_id == "tub"
                        else [source_blocker]
                    ),
                ],
                **FALSE_AUTHORITY,
            }
            for component_id, source_blocker in (
                (
                    "mfu",
                    "snerv_mfu_source_forward_replay_requires_upstream_torch_state_dict_mapping",
                ),
                (
                    "hfr",
                    "snerv_hfr_source_forward_replay_requires_upstream_torch_state_dict_mapping",
                ),
                (
                    "tub",
                    "snerv_tub_full_source_forward_replay_requires_temporal_encoder_decoder_fusion_mapping",
                ),
            )
        ],
        "blockers": _ordered_unique(
            (
                blocker,
                "snerv_score_aware_long_training_official_mfu_hfr_tub_differentiable_mlx_renderer_missing",
                "snerv_official_mfu_hfr_tub_trained_weight_mapping_to_long_training_missing",
                ""
                if official_checkpoint_mapping_verified
                else SNERV_OFFICIAL_TRAINED_CHECKPOINT_STATE_DICT_MAPPING_BLOCKER,
                SNERV_OFFICIAL_TRAINED_CHECKPOINT_SOURCE_FORWARD_BLOCKER,
                *source_forward_blockers,
            )
        ),
        **FALSE_AUTHORITY,
    }
    write_json(artifact_path, payload)
    payload["artifact_path"] = artifact_path.as_posix()
    payload["artifact_sha256"] = sha256_file(artifact_path)
    write_json(artifact_path, payload)
    return payload


def _official_long_training_replay_component_rows(
    *,
    primitive_proof: Mapping[str, Any],
    tensor_map: Mapping[str, Any],
    receiver_replay_passed: bool,
    tub_fixture_replay: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    executed = primitive_proof.get("executed_components")
    executed = dict(executed) if isinstance(executed, Mapping) else {}
    category_counts = tensor_map.get("category_counts")
    category_counts = dict(category_counts) if isinstance(category_counts, Mapping) else {}
    component_specs = (
        (
            "mfu",
            "official_mfu",
            ("official_mfu_weight_payload", "official_mfu_input_payload"),
            "snerv_mfu_source_forward_replay_requires_upstream_torch_state_dict_mapping",
        ),
        (
            "hfr",
            "official_hfr",
            ("official_hfr_weight_payload",),
            "snerv_hfr_source_forward_replay_requires_upstream_torch_state_dict_mapping",
        ),
        (
            "tub",
            "official_tub",
            ("official_tub_input_payload",),
            "snerv_tub_full_source_forward_replay_requires_temporal_encoder_decoder_fusion_mapping",
        ),
    )
    rows: list[dict[str, Any]] = []
    for component_id, proof_key, categories, source_blocker in component_specs:
        tensor_payload_present = any(int(category_counts.get(category) or 0) > 0 for category in categories)
        receiver_component_passed = bool(
            receiver_replay_passed
            and executed.get(proof_key) is True
            and tensor_payload_present
        )
        tub_fixture_passed = bool(
            component_id == "tub"
            and _official_tub_fixture_replay_passed(tub_fixture_replay)
        )
        source_blockers = (
            _official_tub_fixture_preserved_blockers(tub_fixture_replay)
            if tub_fixture_passed
            else [source_blocker]
        )
        rows.append(
            {
                "schema": "snerv_official_mfu_hfr_tub_long_training_replay_component.v1",
                "component_id": component_id,
                "receiver_payload_forward_replay_proven": receiver_component_passed,
                "receiver_tensor_payload_present": tensor_payload_present,
                "official_source_forward_parity_proven": False,
                "official_tub_fixture_source_forward_replay_proven": tub_fixture_passed,
                "official_tub_fixture_closed_blockers": (
                    list(tub_fixture_replay.get("closed_blockers") or ())
                    if tub_fixture_passed and isinstance(tub_fixture_replay, Mapping)
                    else []
                ),
                "score_aware_long_training_renderer_bound": False,
                "train_renderer_bound": False,
                "trained_receiver_state_bound": False,
                "official_trained_checkpoint_state_dict_mapping_verified": False,
                "blockers": [
                    "snerv_score_aware_long_training_official_mfu_hfr_tub_differentiable_mlx_renderer_missing",
                    *source_blockers,
                ],
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _build_official_tub_fixture_replay_for_long_training() -> dict[str, Any]:
    """Run or fail closed the cheap SNeRV_T output_2 source fixture replay."""

    try:
        payload = build_snerv_official_tub_source_forward_replay_artifact()
    except Exception as exc:  # pragma: no cover - defensive fail-closed path.
        return {
            "schema": "snerv_official_tub_source_forward_replay.v1",
            "family": "snerv",
            "component_id": "tub",
            "source_forward_replay_executed": False,
            "official_tub_temporal_encoder_output2_source_fixture_replay_passed": False,
            "failure": f"{type(exc).__name__}: {exc}",
            "blockers": [
                "snerv_official_tub_temporal_encoder_output2_fixture_failed",
                "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
            ],
            **FALSE_AUTHORITY,
        }
    return dict(payload)


def _official_tub_fixture_replay_passed(
    replay: Mapping[str, Any] | None,
) -> bool:
    return bool(
        isinstance(replay, Mapping)
        and replay.get(
            "official_tub_temporal_encoder_output2_source_fixture_replay_passed"
        )
        is True
    )


def _official_tub_fixture_preserved_blockers(
    replay: Mapping[str, Any] | None,
) -> list[str]:
    if isinstance(replay, Mapping):
        preserved = replay.get("preserved_blockers")
        if isinstance(preserved, Sequence) and not isinstance(preserved, (str, bytes)):
            return _ordered_unique([str(value) for value in preserved])
    return list(TUB_PRESERVED_BLOCKERS)


def _official_source_forward_blockers_from_tub_fixture(
    replay: Mapping[str, Any] | None,
) -> list[str]:
    blockers = [
        "snerv_mfu_source_forward_replay_requires_upstream_torch_state_dict_mapping",
        "snerv_hfr_source_forward_replay_requires_upstream_torch_state_dict_mapping",
    ]
    if _official_tub_fixture_replay_passed(replay):
        blockers.extend(_official_tub_fixture_preserved_blockers(replay))
    else:
        blockers.append("snerv_official_mfu_hfr_tub_source_forward_replay_missing")
    return _ordered_unique(blockers)


def _filter_official_source_forward_blockers_for_mapping(
    blockers: Sequence[Any],
    mapping_manifest: Mapping[str, Any] | None,
) -> list[str]:
    manifest = mapping_manifest if isinstance(mapping_manifest, Mapping) else {}
    loaded = manifest.get("official_trained_checkpoint_loaded") is True
    mfu_hfr_mapped = (
        manifest.get("official_mfu_hfr_trained_checkpoint_weight_mapping_proven")
        is True
    )
    tub_temporal_mapped = (
        manifest.get("official_tub_temporal_encoder_weight_mapping_proven") is True
    )
    tub_output2_mapped = (
        manifest.get("official_tub_output2_decoder_weight_mapping_proven") is True
    )
    closed: set[str] = set()
    if loaded:
        closed.add("snerv_official_trained_checkpoint_state_dict_not_loaded")
    if mfu_hfr_mapped:
        closed.add(SNERV_OFFICIAL_MFU_HFR_TUB_WEIGHT_MAPPING_BLOCKER)
    if tub_temporal_mapped:
        closed.update(
            {
                "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_encoder_decoder_weights_not_loaded",
                "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
            }
        )
    if tub_output2_mapped:
        closed.add("snerv_official_tub_portable_output2_decoder_weight_mapping_missing")
    return _ordered_unique(str(blocker) for blocker in blockers if str(blocker) not in closed)


def _official_tub_fixture_replay_from_contract(
    replay: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(replay, Mapping):
        return None
    fixture = replay.get("official_tub_source_forward_fixture_replay")
    if isinstance(fixture, Mapping):
        return fixture
    if str(replay.get("schema") or "") == "snerv_official_tub_source_forward_replay.v1":
        return replay
    return None


def _official_tub_source_fixture_binding(
    replay: Mapping[str, Any] | None,
) -> dict[str, Any]:
    fixture = _official_tub_fixture_replay_from_contract(replay)
    fixture_passed = _official_tub_fixture_replay_passed(fixture)
    replay_mapping = dict(replay or {}) if isinstance(replay, Mapping) else {}
    renderer_bound = bool(
        replay_mapping.get("score_aware_long_training_renderer_bound") is True
    )
    train_renderer_bound = bool(replay_mapping.get("train_renderer_bound") is True)
    trained_receiver_state_bound = bool(
        replay_mapping.get("trained_receiver_state_bound") is True
    )
    receiver_payload_replay_passed = bool(
        replay_mapping.get("receiver_official_payload_forward_replay_passed") is True
    )
    source_fixture_bound = bool(
        fixture_passed
        and renderer_bound
        and train_renderer_bound
        and trained_receiver_state_bound
        and receiver_payload_replay_passed
    )
    closed_blockers = (
        [
            str(blocker)
            for blocker in (fixture or {}).get("closed_blockers") or ()
            if str(blocker)
        ]
        if fixture_passed
        else []
    )
    preserved_blockers = (
        _official_tub_fixture_preserved_blockers(fixture) if fixture_passed else []
    )
    blockers = (
        list(preserved_blockers)
        if source_fixture_bound
        else [SNERV_OFFICIAL_TUB_SOURCE_FIXTURE_REPLAY_MISSING_BLOCKER]
    )
    return {
        "schema": "snerv_official_tub_source_fixture_binding.v1",
        "component_id": "tub",
        "source_fixture_replay_bound": source_fixture_bound,
        "source_fixture_replay_bound_semantics": (
            "official_snerv_t_output2_fixture_plus_mlx_receiver_state_binding_"
            "not_full_trained_checkpoint_source_forward_parity"
        ),
        "official_tub_temporal_encoder_output2_source_fixture_replay_passed": (
            fixture_passed
        ),
        "official_tub_fixture_source_forward_replay_proven": fixture_passed,
        "score_aware_long_training_renderer_bound": renderer_bound,
        "train_renderer_bound": train_renderer_bound,
        "trained_receiver_state_bound": trained_receiver_state_bound,
        "receiver_official_payload_forward_replay_passed": receiver_payload_replay_passed,
        "closed_source_parity_blockers": _ordered_unique(closed_blockers),
        "preserved_source_parity_blockers": _ordered_unique(preserved_blockers),
        "full_tub_source_forward_parity_proven": False,
        "source_forward_parity_proven": False,
        "source_forward_replay_authority": False,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _official_tub_source_fixture_binding_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(metadata, Mapping):
        direct = metadata.get("snerv_official_tub_source_fixture_binding")
        if isinstance(direct, Mapping):
            return dict(direct)
        legacy_direct = metadata.get("official_tub_source_fixture_binding")
        if isinstance(legacy_direct, Mapping):
            return dict(legacy_direct)
        replay = metadata.get("official_mfu_hfr_tub_source_forward_replay")
        if isinstance(replay, Mapping):
            return _official_tub_source_fixture_binding(replay)
        long_training = metadata.get("score_aware_long_training")
        if isinstance(long_training, Mapping):
            replay = long_training.get("official_mfu_hfr_tub_source_forward_replay")
            if isinstance(replay, Mapping):
                return _official_tub_source_fixture_binding(replay)
    return _official_tub_source_fixture_binding(None)


def _official_tub_source_fixture_replay_bound(
    binding: Mapping[str, Any] | None,
) -> bool:
    return bool(
        isinstance(binding, Mapping)
        and binding.get("source_fixture_replay_bound") is True
        and binding.get(
            "official_tub_temporal_encoder_output2_source_fixture_replay_passed"
        )
        is True
    )


def _official_packet_source_parity_blockers(
    binding: Mapping[str, Any] | None,
) -> list[str]:
    blockers = list(SNERV_OFFICIAL_PACKET_SOURCE_PARITY_BLOCKERS)
    if _official_tub_source_fixture_replay_bound(binding):
        blockers = [
            blocker
            for blocker in blockers
            if blocker != SNERV_OFFICIAL_TUB_BATCHED_TEMPORAL_CONTEXT_SOURCE_BLOCKER
        ]
    return _ordered_unique(blockers)


def _official_long_training_replay_with_renderer_binding(
    replay: Mapping[str, Any],
    *,
    official_trained_checkpoint_mapping_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Mark receiver replay custody as consumed by a real train-time renderer."""

    payload = dict(replay)
    trained_checkpoint_mapping_manifest = _coerce_official_checkpoint_mapping_manifest(
        official_trained_checkpoint_mapping_manifest,
        source="snerv_mlx_native_train_export_replay_binding",
    )
    official_checkpoint_loaded = bool(
        trained_checkpoint_mapping_manifest.get("official_trained_checkpoint_loaded")
        is True
    )
    official_checkpoint_mapping_verified = bool(
        trained_checkpoint_mapping_manifest.get(
            "official_mfu_hfr_trained_checkpoint_weight_mapping_proven"
        )
        is True
        and trained_checkpoint_mapping_manifest.get(
            "official_tub_temporal_encoder_weight_mapping_proven"
        )
        is True
    )
    payload["score_aware_long_training_renderer_bound"] = True
    payload["train_renderer_bound"] = True
    payload["trained_receiver_state_bound"] = True
    payload["trained_receiver_state_mapping_scope"] = (
        "mlx_receiver_payload_components_bound_to_training_state"
    )
    payload["trained_weight_mapping_to_long_training_bound"] = (
        official_checkpoint_mapping_verified
    )
    payload["official_trained_checkpoint_loaded"] = official_checkpoint_loaded
    payload["official_trained_checkpoint_state_dict_mapping_verified"] = (
        official_checkpoint_mapping_verified
    )
    payload["official_trained_checkpoint_mapping_manifest"] = (
        trained_checkpoint_mapping_manifest
    )
    payload["official_trained_checkpoint_source_forward_replay_verified"] = False
    payload["blockers"] = _ordered_unique(
        str(blocker)
        for blocker in payload.get("blockers") or ()
        if str(blocker)
        not in {
            "snerv_score_aware_long_training_official_mfu_hfr_tub_differentiable_mlx_renderer_missing",
            *(
                {
                    SNERV_OFFICIAL_MFU_HFR_TUB_WEIGHT_MAPPING_BLOCKER,
                    SNERV_OFFICIAL_TRAINED_CHECKPOINT_STATE_DICT_MAPPING_BLOCKER,
                    "snerv_mfu_source_forward_replay_requires_upstream_torch_state_dict_mapping",
                    "snerv_hfr_source_forward_replay_requires_upstream_torch_state_dict_mapping",
                }
                if official_checkpoint_mapping_verified
                else set()
            ),
        }
    )
    rows: list[dict[str, Any]] = []
    for raw_row in payload.get("component_rows") or ():
        row = dict(raw_row)
        row["score_aware_long_training_renderer_bound"] = True
        row["train_renderer_bound"] = True
        row["trained_receiver_state_bound"] = True
        row["official_trained_checkpoint_state_dict_mapping_verified"] = (
            _official_checkpoint_component_mapping_verified(
                trained_checkpoint_mapping_manifest,
                str(row.get("component_id") or ""),
            )
        )
        row["trained_weight_mapping_to_long_training_bound"] = bool(
            row["official_trained_checkpoint_state_dict_mapping_verified"]
        )
        closed_row_mapping_blockers: set[str] = set()
        if row["official_trained_checkpoint_state_dict_mapping_verified"]:
            component_id = str(row.get("component_id") or "")
            if component_id == "mfu":
                closed_row_mapping_blockers.add(
                    "snerv_mfu_source_forward_replay_requires_upstream_torch_state_dict_mapping"
                )
            elif component_id == "hfr":
                closed_row_mapping_blockers.add(
                    "snerv_hfr_source_forward_replay_requires_upstream_torch_state_dict_mapping"
                )
        row["blockers"] = [
            str(blocker)
            for blocker in row.get("blockers") or ()
            if str(blocker)
            not in {
                "snerv_score_aware_long_training_official_mfu_hfr_tub_differentiable_mlx_renderer_missing",
                *closed_row_mapping_blockers,
            }
        ]
        rows.append(row)
    payload["component_rows"] = rows
    artifact_path = payload.get("artifact_path")
    if isinstance(artifact_path, str) and artifact_path:
        path = Path(artifact_path).expanduser().resolve(strict=False)
        if path.is_file():
            payload.pop("artifact_sha256", None)
            write_json(path, payload)
            payload["artifact_sha256"] = sha256_file(path)
            write_json(path, payload)
    return payload


def _snerv_packet_wire_format_from_schema(schema: Any) -> str | None:
    text = str(schema or "").strip()
    if text == SNERV_ARCHIVE_SCHEMA:
        return "snar1"
    if text == SNERV_ARCHIVE_SCHEMA_V2:
        return "snar2"
    return None


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
    pair_guard_min_score_improved_fraction: float,
    pair_guard_max_pose_worsened_fraction: float,
    device: str,
    perturb_scale: float,
    byte_pressure_multiplier: float,
    section_value_pressure_multiplier: float,
    max_archive_byte_growth: int | None,
    byte_growth_admission_mode: str,
    pose_slack: float,
    seg_slack: float,
    seed: int,
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
            "pair_guard_min_score_improved_fraction": float(
                pair_guard_min_score_improved_fraction
            ),
            "pair_guard_max_pose_worsened_fraction": float(
                pair_guard_max_pose_worsened_fraction
            ),
            "perturb_scale": float(perturb_scale),
            "byte_pressure_multiplier": float(byte_pressure_multiplier),
            "section_value_pressure_multiplier": float(
                section_value_pressure_multiplier
            ),
            "max_archive_byte_growth": max_archive_byte_growth,
            "byte_growth_admission_mode": str(byte_growth_admission_mode),
            "pose_slack": float(pose_slack),
            "seg_slack": float(seg_slack),
            "seed": int(seed),
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
            perturb_scale=float(perturb_scale),
            byte_pressure_multiplier=float(byte_pressure_multiplier),
            section_value_pressure_multiplier=float(section_value_pressure_multiplier),
            max_archive_byte_growth=max_archive_byte_growth,
            byte_growth_admission_mode=str(byte_growth_admission_mode),
            pose_slack=float(pose_slack),
            seg_slack=float(seg_slack),
            component_guard_mode=str(component_guard_mode),
            pair_guard_min_score_improved_fraction=float(
                pair_guard_min_score_improved_fraction
            ),
            pair_guard_max_pose_worsened_fraction=float(
                pair_guard_max_pose_worsened_fraction
            ),
            seed=int(seed),
        )
        result_payload = result.as_jsonable()
        best_packet = getattr(result, "best_packet", b"")
        if best_packet:
            best_packet = bytes(best_packet)
        best_packet_path = out / "best_packet.snar"
        best_packet_materialized = False
        best_packet_path_str: str | None = None
        best_packet_path_sha256: str | None = None
        best_packet_lf_payload_codec: str | None = None
        best_packet_lf_payload_codec_requested: str | None = None
        best_packet_lf_payload_codec_selected: str | None = None
        best_packet_lf_payload_codec_selection_report: dict[str, Any] | None = None
        best_packet_schema: str | None = None
        best_packet_wire_format: str | None = None
        best_packet_contest_submission_wire_format_ready = False
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
            try:
                best_archive = unpack_snerv_archive(best_packet)
                best_packet_schema = best_archive.schema
                best_packet_wire_format = _snerv_packet_wire_format_from_schema(
                    best_archive.schema
                )
                best_packet_contest_submission_wire_format_ready = (
                    best_packet_wire_format == "snar2"
                )
                best_metadata = best_archive.metadata
                best_packet_lf_payload_codec_selected = str(
                    best_metadata.get("lf_payload_codec_selected")
                    or best_metadata.get("lf_payload_codec")
                    or ""
                ) or None
                best_packet_lf_payload_codec_requested = str(
                    best_metadata.get("lf_payload_codec_requested")
                    or result_payload.get("lf_payload_codec_requested")
                    or lf_payload_codec
                )
                best_packet_lf_payload_codec = str(
                    best_packet_lf_payload_codec_selected
                    or result_payload.get("lf_payload_codec")
                    or lf_payload_codec
                )
                if isinstance(
                    best_metadata.get("lf_payload_codec_selection_report"),
                    Mapping,
                ):
                    best_packet_lf_payload_codec_selection_report = dict(
                        best_metadata["lf_payload_codec_selection_report"]
                    )
            except Exception:
                best_packet_lf_payload_codec = None
        blockers = [str(blocker) for blocker in result_payload.get("blockers") or [] if blocker]
        if bool(result_payload.get("receiver_contract_satisfied")) and not (
            best_packet_materialized and best_packet_path_sha256 == result_payload.get("best_packet_sha256")
        ):
            blockers.append("snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export")

        def _payload_value(key: str, default: Any) -> Any:
            value = result_payload.get(key)
            return default if value is None else value

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
                "best_packet_schema": best_packet_schema,
                "best_packet_wire_format": best_packet_wire_format,
                "best_packet_contest_submission_wire_format_ready": (
                    bool(best_packet_contest_submission_wire_format_ready)
                ),
                "decoder_payload_codec": str(result_payload.get("decoder_payload_codec") or decoder_payload_codec),
            "lf_payload_codec": str(
                best_packet_lf_payload_codec
                or result_payload.get("lf_payload_codec_selected")
                or result_payload.get("lf_payload_codec")
                or lf_payload_codec
            ),
            "lf_payload_codec_requested": str(
                best_packet_lf_payload_codec_requested
                or result_payload.get("lf_payload_codec_requested")
                or lf_payload_codec
            ),
            "lf_payload_codec_selected": str(
                best_packet_lf_payload_codec_selected
                or result_payload.get("lf_payload_codec_selected")
                or result_payload.get("lf_payload_codec")
                or lf_payload_codec
            ),
            "lf_payload_codec_selection_report": (
                best_packet_lf_payload_codec_selection_report
                or result_payload.get("lf_payload_codec_selection_report")
            ),
            "component_guard_mode": str(result_payload.get("component_guard_mode") or component_guard_mode),
            "pair_guard_min_score_improved_fraction": float(
                pair_guard_min_score_improved_fraction
            ),
            "pair_guard_max_pose_worsened_fraction": float(
                pair_guard_max_pose_worsened_fraction
            ),
            "perturb_scale": float(_payload_value("perturb_scale", perturb_scale)),
            "byte_pressure_multiplier": float(
                _payload_value("byte_pressure_multiplier", byte_pressure_multiplier)
            ),
            "section_value_pressure_multiplier": float(
                _payload_value(
                    "section_value_pressure_multiplier",
                    section_value_pressure_multiplier,
                )
            ),
            "max_archive_byte_growth": _payload_value(
                "max_archive_byte_growth", max_archive_byte_growth
            ),
            "byte_growth_admission_mode": str(
                _payload_value(
                    "byte_growth_admission_mode", byte_growth_admission_mode
                )
            ),
            "pose_slack": float(_payload_value("pose_slack", pose_slack)),
            "seg_slack": float(_payload_value("seg_slack", seg_slack)),
            "seed": int(seed),
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
            "pair_guard_min_score_improved_fraction": float(
                pair_guard_min_score_improved_fraction
            ),
            "pair_guard_max_pose_worsened_fraction": float(
                pair_guard_max_pose_worsened_fraction
            ),
            "perturb_scale": float(perturb_scale),
            "byte_pressure_multiplier": float(byte_pressure_multiplier),
            "section_value_pressure_multiplier": float(
                section_value_pressure_multiplier
            ),
            "max_archive_byte_growth": max_archive_byte_growth,
            "byte_growth_admission_mode": str(byte_growth_admission_mode),
            "pose_slack": float(pose_slack),
            "seg_slack": float(seg_slack),
            "seed": int(seed),
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
    elif not (
        scorer_loop_qat.get("emitted_packet_contest_submission_wire_format_ready")
        is True
        or str(scorer_loop_qat.get("emitted_packet_wire_format") or "").lower()
        == "snar2"
    ):
        blockers.append(
            "snerv_scorer_loop_qat_emitted_packet_snar2_repack_required"
            if str(scorer_loop_qat.get("emitted_packet_wire_format") or "").lower()
            == "snar1"
            else "snerv_scorer_loop_qat_emitted_packet_wire_format_missing"
        )
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
    native_mlx_decoder_train_optimizer: str = "pact_guarded_adamw",
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
    if model_size.official_mfu_hfr_tub_numeric_primitives_requested:
        return _build_official_mfu_hfr_tub_packet_from_numpy_pairs(
            pairs,
            source_pair_indices=source_pair_indices_tuple,
            model_size=model_size,
            metadata_extra=metadata_extra,
        )
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
        "requested_optimizer": str(native_mlx_decoder_train_optimizer),
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
            optimizer_name=str(native_mlx_decoder_train_optimizer),
        )
        if mlx_training_report.get("executed") is True:
            decoder = trained_decoder
            optimizer_tag = _native_mlx_decoder_optimizer_tag(mlx_training_report)
            hf_decoder_fit_mode = f"native_mlx_{optimizer_tag}_from_{hf_decoder_fit_mode}"

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
    decoder_payload_codec_requested = str(decoder_payload_codec)
    decoder_payload_codec_resolved = resolve_decoder_payload_codec(decoder_payload_codec_requested)
    decoder_payload = encode_decoder_payload(decoder, codec=decoder_payload_codec_resolved)
    lf_payload_codec_requested = str(lf_payload_codec)
    lf_payload = encode_lf_quant_payload(lf_quant_planes, codec=lf_payload_codec_requested)
    lf_payload_codec_report = inspect_lf_quant_payload_header(lf_payload)
    lf_payload_codec_selected = selected_lf_payload_codec_label(
        lf_payload_codec_report,
        requested_codec=lf_payload_codec_requested,
    )
    native_training_export_guard = build_snerv_mlx_native_training_export_guard(
        {
            "native_mlx_training_executed": bool(mlx_training_report.get("executed") is True),
            "native_mlx_hf_decoder_training": mlx_training_report,
        }
    )
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
        "lf_payload_codec": lf_payload_codec_selected,
        "lf_payload_codec_requested": lf_payload_codec_requested,
        "lf_payload_codec_selected": lf_payload_codec_selected,
        "lf_payload_codec_selection_report": lf_payload_codec_report,
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
        "native_mlx_training_export_guard": native_training_export_guard,
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
            f"hf_decoder_{_native_mlx_decoder_optimizer_tag(mlx_training_report)}"
            if mlx_training_report.get("executed") is True
            else "none"
        ),
        "decoder_payload_codec": decoder_payload_codec_resolved,
        "decoder_payload_codec_requested": decoder_payload_codec_requested,
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
        lf_payload=lf_payload,
        decoder_payload=decoder_payload,
        step_map_packet=step_packet.packet,
        metadata=metadata,
    )
    _verify_receiver_frame_decode(archive, reference_shape=pairs.shape)
    return archive


def _build_official_mfu_hfr_tub_packet_from_numpy_pairs(
    pairs: np.ndarray,
    *,
    source_pair_indices: Sequence[int],
    model_size: SnervModelSizeConfig,
    metadata_extra: Mapping[str, Any] | None,
) -> SnervArchivePacket:
    """Build a receiver-rendered official MFU/HFR packet from target pixels.

    This bridge keeps official MFU/HFR primitives in the archive path while
    covering every requested frame. It fits shared official HFR heads over the
    full requested batch in one-level Haar space, stores those official tensors
    plus batched MFU inputs, and lets the receiver render the normal
    ``(pairs, frames, channels, H, W)`` tensor from the official payload path.
    """

    components = _official_mfu_hfr_tub_bootstrap_components_from_pairs(
        pairs,
        model_size=model_size,
    )
    return _build_official_mfu_hfr_tub_packet_from_components(
        components,
        source_pair_indices=source_pair_indices,
        model_size=model_size,
        metadata_extra={
            "hf_decoder_fit_mode": "official_hfr_heads_least_squares_from_haar_ll",
            "allocation_mode": "official_mfu_hfr_tub_frame_producing_bootstrap",
            **dict(metadata_extra or {}),
        },
    )


def _official_mfu_hfr_tub_bootstrap_components_from_pairs(
    pairs: np.ndarray,
    *,
    model_size: SnervModelSizeConfig,
) -> dict[str, Any]:
    """Initialize official MFU/HFR/TUB payload atoms from target pair frames."""

    if pairs.shape[0] < 1 or pairs.shape[1] < 2 or pairs.shape[2] != 3:
        raise SnervMlxNativeExportError(
            "official MFU/HFR/TUB export requires at least one RGB frame pair"
        )
    n_pairs = int(pairs.shape[0])
    frames_per_pair = int(pairs.shape[1])
    target_frames = np.asarray(
        pairs.reshape(n_pairs * frames_per_pair, pairs.shape[2], pairs.shape[3], pairs.shape[4]),
        dtype=np.float64,
    )
    target_chw = np.asarray(pairs[0, 1], dtype=np.float64)
    previous_chw = np.asarray(pairs[0, 0], dtype=np.float64)
    channels, h, w = (int(v) for v in target_chw.shape)
    if channels != 3:
        raise SnervMlxNativeExportError("official MFU/HFR/TUB export requires RGB targets")
    if h % 8 or w % 8:
        raise SnervMlxNativeExportError(
            "official MFU/HFR/TUB bootstrap export requires H/W divisible by 8"
        )

    ll_rows: list[np.ndarray] = []
    lh_rows: list[np.ndarray] = []
    hl_rows: list[np.ndarray] = []
    hh_rows: list[np.ndarray] = []
    for frame_chw in target_frames:
        pyramids = [
            dwt2_multilevel(frame_chw[channel], levels=1, wavelet="haar")
            for channel in range(channels)
        ]
        ll_rows.append(np.stack([pyr.lf for pyr in pyramids], axis=0))
        lh_rows.append(np.stack([pyr.details[0][0] for pyr in pyramids], axis=0))
        hl_rows.append(np.stack([pyr.details[0][1] for pyr in pyramids], axis=0))
        hh_rows.append(np.stack([pyr.details[0][2] for pyr in pyramids], axis=0))
    ll = np.stack(ll_rows, axis=0)
    lh = np.stack(lh_rows, axis=0)
    hl = np.stack(hl_rows, axis=0)
    hh = np.stack(hh_rows, axis=0)
    ll_h, ll_w = (int(v) for v in ll.shape[-2:])

    mfu = _official_passthrough_mfu(channels=channels)
    skip_high_mode = str(model_size.official_skip_high_mode)
    hfr_fit_ll = ll
    if skip_high_mode == "shared_mean":
        hfr_fit_ll = np.broadcast_to(
            np.mean(ll, axis=0, keepdims=True, dtype=np.float64),
            ll.shape,
        ).copy()
    hfr_heads = OfficialHfrHeads(
        lh_head=_fit_official_hfr_head_from_ll(hfr_fit_ll, lh),
        hl_head=_fit_official_hfr_head_from_ll(hfr_fit_ll, hl),
        hh_head=_fit_official_hfr_head_from_ll(hfr_fit_ll, hh),
    )
    low = np.zeros((target_frames.shape[0], channels, ll_h // 4, ll_w // 4), dtype=np.float64)
    skip_mid = np.zeros((target_frames.shape[0], channels, ll_h // 2, ll_w // 2), dtype=np.float64)
    skip_high = ll
    return {
        "mfu": mfu,
        "hfr_heads": hfr_heads,
        "low": low,
        "skip_mid": skip_mid,
        "skip_high": skip_high,
        "skip_high_mode": skip_high_mode,
        "skip_high_full_shape": tuple(int(v) for v in skip_high.shape),
        "tub_current": target_chw,
        "tub_previous": previous_chw,
        "tub_next_frame": target_chw,
        "temporal_encoder_output_shape": (1, 4, max(1, ll_h // 2), max(1, ll_w // 2)),
        "fc_hw": (2, 2),
        "output2_decoder_output_shape": (2, 8, max(1, ll_h // 2), max(1, ll_w // 2)),
        "n_pairs": n_pairs,
        "frames_per_pair": frames_per_pair,
        "channels": channels,
        "h": h,
        "w": w,
        "model_size": model_size.as_jsonable(),
        "official_hfr_bootstrap": {
            "schema": "snerv_official_hfr_bootstrap_fit.v1",
            "fit": "deterministic_bounded_row_least_squares",
            "total_rows_per_head": int(ll.shape[0]) * int(ll_h) * int(ll_w),
            "max_rows_per_head": int(SNERV_OFFICIAL_HFR_BOOTSTRAP_LS_MAX_ROWS),
            "fit_rows_per_head": min(
                int(SNERV_OFFICIAL_HFR_BOOTSTRAP_LS_MAX_ROWS),
                int(ll.shape[0]) * int(ll_h) * int(ll_w),
            ),
            "sampled": (
                int(ll.shape[0]) * int(ll_h) * int(ll_w)
                > int(SNERV_OFFICIAL_HFR_BOOTSTRAP_LS_MAX_ROWS)
            ),
            "sample_policy": "linspace_flat_nhw_deterministic",
            "human_visual_fidelity_objective": False,
            "official_skip_high_mode": skip_high_mode,
            **FALSE_AUTHORITY,
        },
    }


def _official_tub_output2_renderer_kwargs(
    components: Mapping[str, Any],
) -> dict[str, Any]:
    has_temporal = "tub_temporal_encoder_concat" in components
    has_raw = "tub_output2_raw" in components
    if has_temporal != has_raw:
        raise SnervMlxNativeExportError(
            "official TUB output2 renderer payload requires both "
            "tub_temporal_encoder_concat and tub_output2_raw"
        )
    if not has_temporal:
        return {}
    return {
        "tub_temporal_encoder_concat": np.asarray(
            components["tub_temporal_encoder_concat"],
            dtype=np.float32,
        ),
        "tub_output2_raw": np.asarray(
            components["tub_output2_raw"],
            dtype=np.float32,
        ),
        "tub_output2_fc_hw": tuple(
            int(v) for v in components.get("fc_hw") or (2, 2)
        ),
    }


def _official_tub_output2_rows_from_manifest(
    rows: Any,
    *,
    names: Sequence[str],
) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    wanted = {str(name) for name in names}
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("name")) in wanted:
            out.append(dict(row))
    return out


def _official_tub_output2_manifest_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str | None:
    if not rows:
        return None
    return _sha256_bytes(
        json.dumps(
            [dict(row) for row in rows],
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _official_tub_output2_packet_metadata(
    *,
    payload_header: Mapping[str, Any],
    payload_proof: Mapping[str, Any],
) -> dict[str, Any]:
    storage = dict(payload_header.get("tub_output2_storage") or {})
    executed_components = payload_proof.get("executed_components")
    executed_components = (
        dict(executed_components) if isinstance(executed_components, Mapping) else {}
    )
    payload_rows = _official_tub_output2_rows_from_manifest(
        payload_header.get("tensor_manifest") or (),
        names=OFFICIAL_TUB_OUTPUT2_PAYLOAD_TENSOR_NAMES,
    )
    receiver_rows = _official_tub_output2_rows_from_manifest(
        payload_proof.get("output_tensors") or (),
        names=OFFICIAL_TUB_OUTPUT2_RECEIVER_OUTPUT_TENSOR_NAMES,
    )
    return {
        "official_tub_output2_storage": storage,
        "official_tub_output2_payload_export_bound": bool(
            storage.get("stored") is True
        ),
        "official_tub_output2_payload_source_available": bool(
            storage.get("source_payload_present") is True
        ),
        "official_tub_output2_payload_proof_only_elided": bool(
            storage.get("proof_only_elided_from_selected_runtime_packet") is True
        ),
        "official_tub_output2_payload_false_authority_metadata_bound": bool(
            storage.get("proof_only_false_authority_metadata") is True
        ),
        "official_tub_output2_payload_selected_runtime_bytes": int(
            storage.get("stored_raw_bytes") or 0
        ),
        "official_tub_output2_payload_source_raw_bytes": int(
            storage.get("source_raw_bytes") or 0
        ),
        "official_tub_output2_receiver_executed": bool(
            executed_components.get("official_tub_output2_fusion") is True
        ),
        "official_tub_output2_receiver_frame_bound": bool(
            storage.get("receiver_frame_decode_consumes_output2") is True
        ),
        "official_tub_output2_payload_loss_coupled": bool(
            storage.get("train_time_loss_coupled") is True
        ),
        "official_tub_output2_payload_tensor_names": [
            str(row.get("name")) for row in payload_rows
        ],
        "official_tub_output2_payload_tensor_count": len(payload_rows),
        "official_tub_output2_payload_tensor_manifest": payload_rows,
        "official_tub_output2_payload_tensor_manifest_sha256": (
            _official_tub_output2_manifest_sha256(payload_rows)
        ),
        "official_tub_output2_receiver_output_tensor_names": [
            str(row.get("name")) for row in receiver_rows
        ],
        "official_tub_output2_receiver_output_tensor_count": len(receiver_rows),
        "official_tub_output2_receiver_output_tensor_manifest": receiver_rows,
        "official_tub_output2_receiver_output_tensor_manifest_sha256": (
            _official_tub_output2_manifest_sha256(receiver_rows)
        ),
    }


def _build_official_mfu_hfr_tub_packet_from_components(
    components: Mapping[str, Any],
    *,
    source_pair_indices: Sequence[int],
    model_size: SnervModelSizeConfig,
    metadata_extra: Mapping[str, Any] | None,
) -> SnervArchivePacket:
    """Pack trained official MFU/HFR/TUB atoms into the receiver SNAR grammar."""

    low = np.asarray(components["low"], dtype=np.float64)
    skip_mid = np.asarray(components["skip_mid"], dtype=np.float64)
    skip_high = np.asarray(components["skip_high"], dtype=np.float64)
    skip_high_mode = str(components.get("skip_high_mode") or model_size.official_skip_high_mode)
    skip_high_full_shape = tuple(
        int(v) for v in components.get("skip_high_full_shape") or skip_high.shape
    )
    n_frames = int(skip_high_full_shape[0])
    if n_frames <= 0 or n_frames % 2:
        raise SnervMlxNativeExportError(
            f"official MFU/HFR/TUB components require even frame count; got {n_frames}"
        )
    n_pairs = n_frames // 2
    frames_per_pair = 2
    channels = int(skip_high_full_shape[1])
    h = int(skip_high_full_shape[-2]) * 2
    w = int(skip_high_full_shape[-1]) * 2
    tub_temporal_encoder_concat = components.get("tub_temporal_encoder_concat")
    tub_output2_raw = components.get("tub_output2_raw")
    mfu_input_codec = _official_mfu_input_codec_for_export(low, skip_mid)
    official_payload = encode_official_mfu_hfr_tub_decoder_payload(
        mfu=components["mfu"],
        hfr_heads=components["hfr_heads"],
        low=low,
        skip_mid=skip_mid,
        skip_high=skip_high,
        tub_current=np.asarray(components["tub_current"], dtype=np.float64),
        tub_previous=np.asarray(components["tub_previous"], dtype=np.float64),
        tub_next_frame=np.asarray(components["tub_next_frame"], dtype=np.float64),
        temporal_encoder_output_shape=tuple(
            int(v) for v in components.get("temporal_encoder_output_shape") or (1, 4, max(1, h // 4), max(1, w // 4))
        ),
        fc_hw=tuple(int(v) for v in components.get("fc_hw") or (2, 2)),
        output2_decoder_output_shape=tuple(
            int(v) for v in components.get("output2_decoder_output_shape") or (2, 8, max(1, h // 4), max(1, w // 4))
        ),
        mfu_input_codec=mfu_input_codec,
        skip_high_codec=skip_high_mode,
        skip_high_source_shape=skip_high_full_shape,
        tub_input_codec="unused_synthetic",
        tub_temporal_encoder_concat=(
            None
            if tub_temporal_encoder_concat is None
            else np.asarray(tub_temporal_encoder_concat, dtype=np.float64)
        ),
        tub_output2_raw=(
            None if tub_output2_raw is None else np.asarray(tub_output2_raw, dtype=np.float64)
        ),
        store_tub_output2_for_receiver_proof=bool(
            model_size.official_tub_output2_store_for_receiver_proof
        ),
    )
    official_payload_proof = execute_official_mfu_hfr_tub_decoder_payload(official_payload)
    official_payload_header = decode_official_mfu_hfr_tub_decoder_payload(official_payload).header
    official_tub_output2_metadata = _official_tub_output2_packet_metadata(
        payload_header=official_payload_header,
        payload_proof=official_payload_proof,
    )
    metadata_extra_payload = dict(metadata_extra or {})
    tub_source_fixture_binding = _official_tub_source_fixture_binding_from_metadata(
        metadata_extra_payload
    )
    official_source_parity_blockers = _official_packet_source_parity_blockers(
        tub_source_fixture_binding
    )
    step_packet = encode_step_maps_waterfill(
        [np.ones((1, 1), dtype=np.float32)],
        map_importance=np.ones((1,), dtype=np.float64),
        target_bits_per_coeff=1.0,
    )
    official_lf_payload_codec_requested = "spatial_delta_zigzag_leb128_lzma"
    official_lf_payload = encode_lf_quant_payload(
        [np.zeros((1, 1), dtype=np.int64)],
        codec=official_lf_payload_codec_requested,
    )
    official_lf_payload_codec_report = inspect_lf_quant_payload_header(
        official_lf_payload
    )
    official_lf_payload_codec_selected = selected_lf_payload_codec_label(
        official_lf_payload_codec_report,
        requested_codec=official_lf_payload_codec_requested,
    )
    metadata = {
        "n_pairs": n_pairs,
        "frames_per_pair": frames_per_pair,
        "channels": channels,
        "levels": 1,
        "wavelet": "haar",
        "carrier_hw": [h, w],
        "orig_hw": [h, w],
        "source_pair_indices": [int(value) for value in source_pair_indices],
        "source_pair_indices_preserved": True,
        "pair_index_alignment_mode": "official_batched_requested_pairs",
        "lf_plane_count": 1,
        "lf_payload_codec": official_lf_payload_codec_selected,
        "lf_payload_codec_requested": official_lf_payload_codec_requested,
        "lf_payload_codec_selected": official_lf_payload_codec_selected,
        "lf_payload_codec_selection_report": official_lf_payload_codec_report,
        "lf_payload_receiver_usage": (
            "unused_dummy_zero_official_payload_frame_decode_uses_decoder_payload"
        ),
        "step_map_packet_schema": step_packet.schema,
        "step_map_coder_mode": "official_payload_unused_dummy_step",
        "step_map_coder_groups": [dict(group) for group in step_packet.groups],
        "allocation_mode": "official_mfu_hfr_tub_frame_producing_bootstrap",
        "hf_decoder_fit_mode": "official_hfr_heads_least_squares_from_haar_ll",
        "official_skip_high_mode": skip_high_mode,
        "official_skip_high_full_shape": [int(v) for v in skip_high_full_shape],
        "official_skip_high_export_storage_shape": [
            int(v) for v in skip_high.shape
        ],
        "official_skip_high_export_is_compact_train_state": bool(
            tuple(int(v) for v in skip_high.shape) != skip_high_full_shape
        ),
        "official_mfu_input_storage_mode": (
            mfu_input_codec if mfu_input_codec is not None else "full"
        ),
        "official_tub_input_storage_mode": "unused_synthetic",
        **(
            {"official_hfr_bootstrap": components["official_hfr_bootstrap"]}
            if isinstance(components.get("official_hfr_bootstrap"), Mapping)
            else {}
        ),
        "decoder_payload_codec": DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA,
        **metadata_extra_payload,
        **official_tub_output2_metadata,
        "snerv_official_tub_source_fixture_binding": tub_source_fixture_binding,
        "snerv_official_tub_source_fixture_replay_bound": (
            _official_tub_source_fixture_replay_bound(tub_source_fixture_binding)
        ),
        "snerv_official_tub_source_fixture_replay_passed": bool(
            tub_source_fixture_binding.get(
                "official_tub_temporal_encoder_output2_source_fixture_replay_passed"
            )
            is True
        ),
        "snerv_official_tub_source_fixture_closed_blockers": list(
            tub_source_fixture_binding.get("closed_source_parity_blockers") or []
        ),
        "snerv_official_tub_source_fixture_preserved_blockers": list(
            tub_source_fixture_binding.get("preserved_source_parity_blockers") or []
        ),
        "snerv_official_tub_source_forward_fixture_bound": (
            _official_tub_source_fixture_replay_bound(tub_source_fixture_binding)
        ),
        "snerv_model_size_adapter": model_size.adapter,
        "snerv_official_mfu_hfr_tub_numeric_primitives_requested": True,
        "snerv_official_mfu_hfr_tub_export_bound": True,
        "snerv_official_mfu_hfr_tub_export_bound_semantics": (
            "receiver_payload_bound_not_source_forward_parity"
        ),
        "snerv_official_mfu_hfr_tub_receiver_payload_bound": True,
        "snerv_official_mfu_hfr_tub_source_forward_replay_bound": False,
        "snerv_official_mfu_hfr_tub_source_forward_replay_authority": False,
        "snerv_official_mfu_hfr_tub_frame_producing_export": True,
        "source_faithful_stack": False,
        "official_source_parity_blockers": official_source_parity_blockers,
        **FALSE_AUTHORITY,
    }
    archive = pack_snerv_archive_snar2(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=[0.0]),
        lf_payload=official_lf_payload,
        decoder_payload=official_payload,
        step_map_packet=step_packet.packet,
        metadata=metadata,
    )
    _verify_receiver_frame_decode(
        archive,
        reference_shape=(n_pairs, frames_per_pair, channels, h, w),
    )
    return archive


def _official_mfu_input_codec_for_export(
    low: np.ndarray,
    skip_mid: np.ndarray,
) -> str | None:
    """Use receiver-synthetic MFU inputs only when the source tensors are exactly zero."""

    if np.count_nonzero(low) == 0 and np.count_nonzero(skip_mid) == 0:
        return "zero_synthetic"
    return None


def _official_passthrough_mfu(*, channels: int) -> OfficialSnervMfu:
    spec = OfficialSnervMfuSpec(
        low_channels=int(channels),
        mid_channels=int(channels),
        high_channels=int(channels),
        mid_stride=2,
        high_stride=2,
        num_blocks=0,
    )
    zero_up = np.zeros((channels, channels, 2, 2), dtype=np.float64)
    zero_bias = np.zeros((channels,), dtype=np.float64)
    rb_mid_weight = np.zeros((channels, channels * 2, 3, 3), dtype=np.float64)
    rb_high_weight = np.zeros((channels, channels * 2, 3, 3), dtype=np.float64)
    for channel in range(channels):
        rb_high_weight[channel, channels + channel, 1, 1] = 1.0
    return OfficialSnervMfu(
        spec=spec,
        upsample_mid=OfficialConvTranspose2dNchw(zero_up, zero_bias, stride=2),
        rb_mid=OfficialResidualBlocksWithInputConv(
            input_conv=OfficialConv2dNchw(rb_mid_weight, zero_bias, padding=1),
            residual_blocks=(),
        ),
        upsample_high=OfficialConvTranspose2dNchw(zero_up, zero_bias, stride=2),
        rb_high=OfficialResidualBlocksWithInputConv(
            input_conv=OfficialConv2dNchw(rb_high_weight, zero_bias, padding=1),
            residual_blocks=(),
        ),
    )


def _fit_official_hfr_head_from_ll(
    ll_chw: np.ndarray,
    detail_chw: np.ndarray,
    *,
    max_rows: int = SNERV_OFFICIAL_HFR_BOOTSTRAP_LS_MAX_ROWS,
) -> OfficialHfrConvBlock:
    ll = np.asarray(ll_chw, dtype=np.float64)
    detail = np.asarray(detail_chw, dtype=np.float64)
    if ll.ndim == 3:
        ll = ll[np.newaxis, :, :, :]
    if detail.ndim == 3:
        detail = detail[np.newaxis, :, :, :]
    if ll.shape != detail.shape or ll.ndim != 4:
        raise SnervMlxNativeExportError(
            f"official HFR fit expects matching NCHW LL/detail, got {ll.shape} and {detail.shape}"
        )
    batch, channels, h, w = (int(v) for v in ll.shape)
    conv1_weight = np.zeros((channels, channels, 1, 1), dtype=np.float64)
    for channel in range(channels):
        conv1_weight[channel, channel, 0, 0] = 1.0
    conv1_bias = np.zeros((channels,), dtype=np.float64)
    hidden = np.where(ll >= 0.0, ll, 0.1 * ll)
    padded = np.pad(hidden, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    total_rows = int(batch) * int(h) * int(w)
    row_budget = int(max_rows)
    if row_budget > 0 and total_rows > row_budget:
        flat_indices = np.linspace(
            0,
            total_rows - 1,
            num=row_budget,
            dtype=np.int64,
        )
        frame_idx = flat_indices // (h * w)
        rem = flat_indices % (h * w)
        y_idx = rem // w
        x_idx = rem % w
        columns = [
            padded[frame_idx, :, y_idx + dy, x_idx + dx]
            for dy in range(3)
            for dx in range(3)
        ]
        design = np.stack(columns, axis=-1).reshape(row_budget, channels * 9)
        target = detail[frame_idx, :, y_idx, x_idx]
    else:
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (3, 3),
            axis=(2, 3),
        )
        design = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
            total_rows,
            channels * 9,
        )
        target = detail.transpose(0, 2, 3, 1).reshape(total_rows, channels)
    design = np.concatenate(
        [design, np.ones((int(design.shape[0]), 1), dtype=np.float64)],
        axis=1,
    )
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    conv2_weight = beta[:-1, :].T.reshape(channels, channels, 3, 3)
    conv2_bias = beta[-1, :]
    return OfficialHfrConvBlock(
        conv1=OfficialConv2dNchw(conv1_weight, conv1_bias, padding=0),
        conv2=OfficialConv2dNchw(conv2_weight, conv2_bias, padding=1),
    )


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
    optimizer_name: str = "pact_guarded_adamw",
) -> tuple[HfGenerationDecoder, dict[str, Any]]:
    if int(steps) <= 0:
        raise SnervMlxNativeExportError("native MLX decoder train steps must be positive")
    if float(learning_rate) <= 0.0:
        raise SnervMlxNativeExportError("native MLX decoder train lr must be positive")
    if float(ridge) < 0.0:
        raise SnervMlxNativeExportError("native MLX decoder train ridge must be non-negative")
    optimizer_tag = _normalize_native_mlx_decoder_optimizer(optimizer_name)
    try:
        import mlx.core as mx
        import mlx.optimizers as optim
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
            optimizer_used = optimizer_tag
            guard_action = "none"
            k = _run_native_mlx_decoder_optimizer_steps(
                mx,
                optim,
                x=x,
                y=y,
                weights=weights,
                denom=denom,
                initial_vec=k,
                ridge=float(ridge),
                steps=int(steps),
                learning_rate=float(learning_rate),
                optimizer=(
                    "adamw" if optimizer_tag == "pact_guarded_adamw" else optimizer_tag
                ),
            )
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
            if optimizer_tag == "pact_guarded_adamw" and _native_mlx_loss_worsened(
                initial=initial_loss,
                final=final_loss,
            ):
                fallback = _run_native_mlx_decoder_optimizer_steps(
                    mx,
                    optim,
                    x=x,
                    y=y,
                    weights=weights,
                    denom=denom,
                    initial_vec=mx.array(
                        np.asarray(initial_decoder.kernels[lvl][subband], dtype=np.float32).reshape(-1),
                        dtype=mx.float32,
                    ),
                    ridge=float(ridge),
                    steps=int(steps),
                    learning_rate=float(learning_rate),
                    optimizer="full_batch_gradient_descent",
                )
                fallback_loss = float(
                    _mlx_weighted_linear_loss(
                        mx,
                        x=x,
                        y=y,
                        weights=weights,
                        denom=denom,
                        vec=fallback,
                        ridge=float(ridge),
                    ).item()
                )
                if not _native_mlx_loss_worsened(
                    initial=initial_loss,
                    final=fallback_loss,
                ):
                    k = fallback
                    final_loss = fallback_loss
                    optimizer_used = "full_batch_gradient_descent"
                    guard_action = "adamw_worsened_used_full_batch_gradient_descent"
                else:
                    k = mx.array(
                        np.asarray(initial_decoder.kernels[lvl][subband], dtype=np.float32).reshape(-1),
                        dtype=mx.float32,
                    )
                    final_loss = initial_loss
                    optimizer_used = "closed_form_initial"
                    guard_action = "adamw_and_gradient_descent_worsened_kept_initial"
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
                    "optimizer_used": optimizer_used,
                    "guard_action": guard_action,
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
            "optimizer": optimizer_tag,
            "optimizer_backend": (
                "mlx.optimizers+guarded_manual_fallback"
                if optimizer_tag == "pact_guarded_adamw"
                else (
                    "mlx.optimizers"
                    if optimizer_tag != "full_batch_gradient_descent"
                    else "manual_mlx"
                )
            ),
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "ridge": float(ridge),
            "saliency_gain": float(saliency_gain),
            "level_subband_rows": loss_rows,
            "optimizer_used_counts": {
                str(name): sum(1 for row in loss_rows if row["optimizer_used"] == name)
                for name in sorted({str(row["optimizer_used"]) for row in loss_rows})
            },
            "guarded_fallback_count": sum(
                1 for row in loss_rows if str(row["guard_action"]) != "none"
            ),
            "kept_initial_count": sum(
                1 for row in loss_rows if row["optimizer_used"] == "closed_form_initial"
            ),
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


def _normalize_native_mlx_decoder_optimizer(name: str) -> str:
    normalized = str(name or "adamw").strip().lower().replace("-", "_")
    aliases = {
        "gd": "full_batch_gradient_descent",
        "gradient_descent": "full_batch_gradient_descent",
        "manual_gradient_descent": "full_batch_gradient_descent",
        "full_batch_gd": "full_batch_gradient_descent",
        "full_batch_gradient_descent": "full_batch_gradient_descent",
        "pact": "pact_guarded_adamw",
        "pact_adamw": "pact_guarded_adamw",
        "pact_guarded_adamw": "pact_guarded_adamw",
        "guarded_adamw": "pact_guarded_adamw",
        "adamw": "adamw",
        "adam": "adam",
        "lion": "lion",
        "sgd": "sgd",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise SnervMlxNativeExportError(
            "unsupported native MLX decoder optimizer "
            f"{name!r}; expected one of {sorted(aliases)}"
        ) from exc


def _build_native_mlx_decoder_optimizer(
    optim: Any,
    *,
    optimizer: str,
    learning_rate: float,
) -> Any:
    if optimizer in ("full_batch_gradient_descent", "pact_guarded_adamw"):
        return None
    if optimizer == "adamw":
        return optim.AdamW(learning_rate=float(learning_rate), weight_decay=0.0)
    if optimizer == "adam":
        return optim.Adam(learning_rate=float(learning_rate))
    if optimizer == "lion":
        return optim.Lion(learning_rate=float(learning_rate), weight_decay=0.0)
    if optimizer == "sgd":
        return optim.SGD(learning_rate=float(learning_rate), momentum=0.9)
    raise SnervMlxNativeExportError(f"unsupported native MLX decoder optimizer {optimizer!r}")


def _run_native_mlx_decoder_optimizer_steps(
    mx: Any,
    optim: Any,
    *,
    x: Any,
    y: Any,
    weights: Any,
    denom: Any,
    initial_vec: Any,
    ridge: float,
    steps: int,
    learning_rate: float,
    optimizer: str,
) -> Any:
    opt = _build_native_mlx_decoder_optimizer(
        optim,
        optimizer=optimizer,
        learning_rate=float(learning_rate),
    )
    params = {"k": initial_vec}
    for _step in range(int(steps)):
        k = params["k"]
        pred = mx.matmul(x, k)
        residual = pred - y
        grad = (2.0 * mx.matmul(mx.transpose(x), weights * residual) / denom) + (
            2.0 * float(ridge) * k
        )
        if optimizer == "full_batch_gradient_descent":
            params["k"] = k - (float(learning_rate) * grad)
            mx.eval(params["k"])
        else:
            opt.update(params, {"k": grad})
            mx.eval(params, opt.state)
    return params["k"]


def _native_mlx_decoder_optimizer_tag(report: Mapping[str, Any]) -> str:
    return _normalize_native_mlx_decoder_optimizer(
        str(
            report.get("optimizer")
            or report.get("requested_optimizer")
            or "pact_guarded_adamw"
        )
    )


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


def _selected_archive_metadata_for_report(
    *,
    decoded_metadata: Mapping[str, Any],
    selected_packet_sha256: str,
    selected_packet_source: str,
    closed_form_archive: SnervArchivePacket,
) -> dict[str, Any]:
    """Keep SNAR2 wire metadata compact while report/control metadata stays rich."""

    receiver_metadata = dict(decoded_metadata)
    metadata = dict(receiver_metadata)
    if str(selected_packet_sha256) == _sha256_bytes(closed_form_archive.packet):
        metadata = {
            **metadata,
            **dict(closed_form_archive.metadata),
            "receiver_packet_metadata": receiver_metadata,
            "receiver_packet_report_metadata_source": (
                "snar2_compact_wire_metadata_plus_in_memory_export_metadata"
            ),
            "receiver_packet_report_metadata_source_packet": str(
                selected_packet_source
            ),
            **FALSE_AUTHORITY,
        }
    return metadata


def _selected_metadata_report_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Expose non-authority packet telemetry in reports without charging it in SNAR2."""

    prefixes = (
        "official_",
        "snerv_official_",
        "lf_payload_",
        "step_map_",
        "receiver_packet_report_",
    )
    exact = {
        "official_source_parity_blockers",
        "receiver_packet_metadata",
        "source_faithful_stack",
    }
    return {
        str(key): value
        for key, value in metadata.items()
        if str(key).startswith(prefixes) or str(key) in exact
    }


_SNERV_SELECTED_PACKET_SECTION_METADATA_CONTINUITY_FIELDS = {
    "lf_payload": (
        "lf_payload_codec",
        "lf_payload_codec_requested",
        "lf_payload_codec_selected",
        "lf_payload_codec_selection_report",
        "allocation_mode",
        "lf_step_allocation_mode",
        "lf_step_allocation_rows",
        "contest_scorer_distortion_objective",
        "recon_pixel_weight_consumed",
        "recon_pixel_weight_metadata",
    ),
    "step_map_packet": (
        "step_map_packet_schema",
        "step_map_coder_mode",
        "step_map_coder_bins",
        "step_map_waterfill_bits_per_coeff",
        "step_map_coder_groups",
    ),
}


def _snerv_metadata_value_present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _repack_selected_packet_with_section_metadata_continuity(
    selected_packet: bytes,
    *,
    base_packet: bytes,
    selected_packet_source: str,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Preserve receiver-grammar metadata when QAT swaps only payload sections."""

    selected_archive = unpack_snerv_archive(selected_packet)
    base_archive = unpack_snerv_archive(base_packet)
    selected_metadata = dict(selected_archive.metadata)
    inherited: list[dict[str, Any]] = []
    for section_name, fields in (
        _SNERV_SELECTED_PACKET_SECTION_METADATA_CONTINUITY_FIELDS.items()
    ):
        if selected_archive.sections.get(section_name) != base_archive.sections.get(
            section_name
        ):
            continue
        for field in fields:
            if _snerv_metadata_value_present(selected_metadata.get(field)):
                continue
            base_value = base_archive.metadata.get(field)
            if not _snerv_metadata_value_present(base_value):
                continue
            selected_metadata[field] = base_value
            inherited.append({"section_name": section_name, "field": field})
    continuity = {
        "schema": "snerv_selected_packet_metadata_continuity.v1",
        "selected_packet_source": str(selected_packet_source),
        "base_packet_sha256": _sha256_bytes(base_packet),
        "input_selected_packet_sha256": _sha256_bytes(selected_packet),
        "inherited_field_count": len(inherited),
        "inherited_fields": inherited,
        "metadata_only_repack": bool(inherited),
        **FALSE_AUTHORITY,
    }
    section_metadata_packet = bytes(selected_packet)
    if inherited:
        selected_metadata["selected_packet_metadata_continuity"] = continuity
        section_metadata_packet = pack_snerv_archive(
            metadata_payload=selected_archive.sections["metadata_payload"],
            lf_payload=selected_archive.sections["lf_payload"],
            decoder_payload=selected_archive.sections["decoder_payload"],
            step_map_packet=selected_archive.sections["step_map_packet"],
            metadata=selected_metadata,
        ).packet
    repacked, submission_repack = _snerv_submission_packet_for_export(
        section_metadata_packet,
        submission_archive_format="snar2",
    )
    continuity["container_repacked_to_submission_format"] = bool(
        submission_repack.get("repacked")
    )
    continuity["output_packet_schema"] = submission_repack.get(
        "output_packet_schema"
    )
    continuity["output_packet_wire_format"] = _snerv_packet_wire_format_from_schema(
        submission_repack.get("output_packet_schema")
    )
    continuity["contest_submission_wire_format_ready"] = (
        continuity["output_packet_wire_format"] == "snar2"
    )
    continuity["output_selected_packet_sha256"] = _sha256_bytes(repacked)
    selected_metadata["selected_packet_metadata_continuity"] = continuity
    return repacked, selected_metadata, continuity


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
    for label, arr in (("target0", target0_np), ("target1", target1_np)):
        if not np.isfinite(arr).all():
            raise SnervMlxNativeExportError(f"{label} contains nonfinite values")
        min_value = float(np.min(arr)) if arr.size else 0.0
        max_value = float(np.max(arr)) if arr.size else 0.0
        if min_value < -1.0e-6 or max_value > 1.0 + 1.0e-6:
            raise SnervMlxNativeExportError(
                f"{label} must be normalized RGB in [0,1] before NCHW255 "
                f"conversion; got min={min_value:.6g} max={max_value:.6g}"
            )
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


def _packet_preserves_official_decoder_payload(packet: bytes) -> bool:
    if not packet:
        return False
    try:
        decoded = unpack_snerv_archive(packet)
        header = inspect_decoder_payload_header(decoded.sections["decoder_payload"])
    except Exception:
        return False
    return str(header.get("schema") or "") == DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA


def _official_tub_output2_binding_signature_from_packet(packet: bytes) -> dict[str, Any]:
    if not packet:
        return {
            "official_decoder_payload_selected": False,
            "blockers": ["snerv_official_tub_output2_binding_packet_missing"],
            **FALSE_AUTHORITY,
        }
    try:
        decoded = unpack_snerv_archive(packet)
        header = inspect_decoder_payload_header(decoded.sections["decoder_payload"])
    except Exception as exc:
        return {
            "official_decoder_payload_selected": False,
            "failure": f"{type(exc).__name__}: {exc}",
            "blockers": ["snerv_official_tub_output2_binding_packet_unreadable"],
            **FALSE_AUTHORITY,
        }
    if str(header.get("schema") or "") != DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA:
        return {
            "official_decoder_payload_selected": False,
            "blockers": ["snerv_official_tub_output2_binding_non_official_decoder_payload"],
            **FALSE_AUTHORITY,
        }
    storage = dict(header.get("tub_output2_storage") or {})
    tub_config = dict(header.get("tub_config") or {})
    payload_tensor_names = sorted(
        str(row.get("name"))
        for row in header.get("tensor_manifest") or ()
        if isinstance(row, Mapping)
        and str(row.get("name") or "") in OFFICIAL_TUB_OUTPUT2_PAYLOAD_TENSOR_NAMES
    )
    return {
        "official_decoder_payload_selected": True,
        "stored": bool(storage.get("stored")),
        "source_payload_present": bool(storage.get("source_payload_present")),
        "proof_only_elided_from_selected_runtime_packet": bool(
            storage.get("proof_only_elided_from_selected_runtime_packet")
        ),
        "proof_only_false_authority_metadata": bool(
            storage.get("proof_only_false_authority_metadata")
        ),
        "receiver_executes_output2_fusion_from_payload": bool(
            storage.get("receiver_executes_output2_fusion_from_payload")
        ),
        "receiver_frame_decode_consumes_output2": bool(
            storage.get("receiver_frame_decode_consumes_output2")
        ),
        "receiver_frame_decode_binding_status": str(
            storage.get("receiver_frame_decode_binding_status") or ""
        ),
        "receiver_output2_frame_shape_match": bool(
            storage.get("receiver_output2_frame_shape_match")
        ),
        "train_time_loss_coupled": bool(storage.get("train_time_loss_coupled")),
        "scored_pixel_render_bound": bool(storage.get("scored_pixel_render_bound")),
        "source_raw_bytes": int(storage.get("source_raw_bytes") or 0),
        "stored_raw_bytes": int(storage.get("stored_raw_bytes") or 0),
        "temporal_encoder_output_shape": (
            list(tub_config.get("temporal_encoder_output_shape"))
            if tub_config.get("temporal_encoder_output_shape") is not None
            else None
        ),
        "output2_decoder_output_shape": (
            list(tub_config.get("output2_decoder_output_shape"))
            if tub_config.get("output2_decoder_output_shape") is not None
            else None
        ),
        "fc_hw": (
            list(tub_config.get("fc_hw"))
            if tub_config.get("fc_hw") is not None
            else None
        ),
        "payload_tensor_names": payload_tensor_names,
        "blockers": [],
        **FALSE_AUTHORITY,
    }


def _official_tub_output2_binding_preservation_report(
    packet: bytes,
    *,
    base_packet: bytes,
) -> dict[str, Any]:
    expected = _official_tub_output2_binding_signature_from_packet(base_packet)
    actual = _official_tub_output2_binding_signature_from_packet(packet)
    mismatched_fields: list[str] = []
    if (
        expected.get("official_decoder_payload_selected") is True
        and actual.get("official_decoder_payload_selected") is True
    ):
        for field in OFFICIAL_TUB_OUTPUT2_BINDING_SIGNATURE_FIELDS:
            if expected.get(field) != actual.get(field):
                mismatched_fields.append(str(field))
    else:
        mismatched_fields.append("official_decoder_payload_selected")
    preserved = bool(
        expected.get("official_decoder_payload_selected") is True
        and actual.get("official_decoder_payload_selected") is True
        and not mismatched_fields
    )
    return {
        "schema": "snerv_official_tub_output2_binding_preservation.v1",
        "preserved": preserved,
        "mismatched_fields": mismatched_fields,
        "expected": expected,
        "actual": actual,
        "blockers": (
            []
            if preserved
            else ["snerv_scorer_loop_qat_best_packet_rejected_official_tub_output2_binding_mismatch"]
        ),
        **FALSE_AUTHORITY,
    }


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
    archive_view = unpack_snerv_archive(archive.packet)
    header = inspect_decoder_payload_header(archive_view.sections["decoder_payload"])
    if str(header.get("schema") or "") == DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA:
        proof = _selected_packet_official_payload_frame_replay(archive.packet)
        if proof.get("receiver_payload_frame_replay_passed") is not True:
            raise SnervMlxNativeExportError(
                "official MFU/HFR/TUB receiver payload did not prove frame replay: "
                + ",".join(str(blocker) for blocker in proof.get("blockers") or ())
            )


def _snerv_receiver_frame_reconstruction_profile(
    packet: bytes,
    *,
    reference_pairs_nchw255: np.ndarray,
    source_pair_indices: Sequence[int],
    profile_id: str,
    reference_kind: str,
    packet_source: str,
    worst_pair_count: int = 16,
) -> dict[str, Any]:
    """Profile receiver-decoded pixels against the named reference tensor.

    This is distortion evidence, not score authority and not source-forward
    parity.  It exists in the native export report because the scorer only ever
    sees receiver-decoded pixels; long-run selection and byte-cap controls need
    this profile before they spend hours optimizing a train-side tensor that the
    SNAR1 receiver materializes differently.
    """

    reference = np.asarray(reference_pairs_nchw255, dtype=np.float32)
    reference_shape = tuple(int(value) for value in reference.shape)
    profile: dict[str, Any] = {
        "schema": SNERV_RECEIVER_FRAME_RECONSTRUCTION_PROFILE_SCHEMA,
        "profile_id": str(profile_id),
        "reference_kind": str(reference_kind),
        "packet_source": str(packet_source),
        "source_pair_indices": [int(value) for value in source_pair_indices],
        "worst_pair_count": int(max(0, worst_pair_count)),
        "receiver_decoded_selected_packet": True,
        "shape_matches": False,
        "receiver_frames_finite": False,
        "reference_frames_finite": bool(np.isfinite(reference).all()),
        "score_claim": False,
        "frontier_score_claim": False,
        "rank_or_kill_eligible": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [],
    }
    if (
        reference.ndim != 5
        or int(reference.shape[1]) != 2
        or int(reference.shape[2]) != 3
    ):
        return {
            **profile,
            "reference_shape": [int(value) for value in reference_shape],
            "blockers": [
                "snerv_receiver_frame_reconstruction_reference_not_nchw_pair_tensor"
            ],
        }
    try:
        decoded = decode_snerv_archive_frames(packet).astype(np.float32, copy=False)
    except Exception as exc:
        return {
            **profile,
            "failure": f"{type(exc).__name__}: {exc}",
            "blockers": [
                "snerv_receiver_frame_reconstruction_decode_failed",
                f"snerv_receiver_frame_reconstruction_exception_{type(exc).__name__}",
            ],
        }

    decoded_shape = tuple(int(v) for v in decoded.shape)
    raw_shape_matches = decoded_shape == reference_shape
    comparison_decoded = decoded
    comparison_domain = "receiver_native_geometry"
    scorer_geometry_resize_applied = False
    shape_matches = raw_shape_matches
    receiver_finite = bool(np.isfinite(decoded).all())
    blockers: list[str] = []
    if (
        not raw_shape_matches
        and len(decoded_shape) == 5
        and len(reference_shape) == 5
        and decoded_shape[:3] == reference_shape[:3]
        and reference_shape[-2:] == SCORER_HW
    ):
        flat_decoded = decoded.reshape(
            int(np.prod(decoded_shape[:2])),
            int(decoded_shape[2]),
            int(decoded_shape[3]),
            int(decoded_shape[4]),
        )
        comparison_decoded = _resize_nchw_bilinear(flat_decoded, out_hw=SCORER_HW)
        comparison_decoded = comparison_decoded.reshape(reference_shape)
        comparison_domain = "upstream_scorer_geometry_bilinear"
        scorer_geometry_resize_applied = True
        shape_matches = True
    if not shape_matches:
        blockers.append("snerv_receiver_frame_reconstruction_shape_mismatch")
    if len(decoded_shape) != 5:
        blockers.append(
            "snerv_receiver_frame_reconstruction_decoded_not_nchw_pair_tensor"
        )
    if not receiver_finite:
        blockers.append("snerv_receiver_frame_reconstruction_nonfinite_receiver")
    if profile["reference_frames_finite"] is not True:
        blockers.append("snerv_receiver_frame_reconstruction_nonfinite_reference")

    profile.update(
        {
            "decoded_shape": [int(value) for value in decoded.shape],
            "raw_decoded_shape": [int(value) for value in decoded.shape],
            "reference_shape": [int(value) for value in reference.shape],
            "comparison_decoded_shape": [
                int(value) for value in comparison_decoded.shape
            ],
            "comparison_domain": comparison_domain,
            "raw_shape_matches": raw_shape_matches,
            "shape_matches": shape_matches,
            "scorer_geometry_resize_applied": scorer_geometry_resize_applied,
            "receiver_frames_finite": receiver_finite,
            "blockers": blockers,
        }
    )
    if blockers:
        return profile

    decoded = comparison_decoded
    diff = decoded - reference
    abs_diff = np.abs(diff)
    scorer_distortion = _snerv_scorer_domain_distortion_anatomy(
        decoded,
        reference,
        source_pair_indices=source_pair_indices,
        worst_pair_count=worst_pair_count,
        comparison_domain=comparison_domain,
    )
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(abs_diff))
    max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
    decoded_std = float(np.std(decoded))
    reference_std = float(np.std(reference))
    std_ratio = decoded_std / max(reference_std, 1.0e-12)
    decoded_dynamic_range = float(np.max(decoded) - np.min(decoded)) if decoded.size else 0.0
    reference_dynamic_range = (
        float(np.max(reference) - np.min(reference)) if reference.size else 0.0
    )
    decoded_low_sat = float(np.mean(decoded <= 1.0)) if decoded.size else 0.0
    decoded_high_sat = float(np.mean(decoded >= 254.0)) if decoded.size else 0.0
    reference_low_sat = float(np.mean(reference <= 1.0)) if reference.size else 0.0
    reference_high_sat = float(np.mean(reference >= 254.0)) if reference.size else 0.0
    saturation_delta = abs(decoded_low_sat - reference_low_sat) + abs(
        decoded_high_sat - reference_high_sat
    )
    value_domain_blockers: list[str] = []
    rmse = float(np.sqrt(max(mse, 0.0)))
    if decoded_dynamic_range <= 1.0:
        value_domain_blockers.append(
            "snerv_receiver_frame_reconstruction_decoded_dynamic_range_degenerate"
        )
    if reference_std > 1.0e-6 and std_ratio < SNERV_RECEIVER_RECON_MIN_STD_RATIO:
        value_domain_blockers.append(
            "snerv_receiver_frame_reconstruction_decoded_std_collapsed"
        )
    if reference_std > 1.0e-6 and std_ratio > SNERV_RECEIVER_RECON_MAX_STD_RATIO:
        value_domain_blockers.append(
            "snerv_receiver_frame_reconstruction_decoded_std_exploded"
        )
    if saturation_delta > SNERV_RECEIVER_RECON_MAX_SATURATION_DELTA:
        value_domain_blockers.append(
            "snerv_receiver_frame_reconstruction_saturation_delta_excessive"
        )
    if rmse > SNERV_RECEIVER_RECON_MAX_RMSE_NCHW255:
        value_domain_blockers.append(
            "snerv_receiver_frame_reconstruction_rmse_exceeds_value_domain_gate"
        )
    if mae > SNERV_RECEIVER_RECON_MAX_MAE_NCHW255:
        value_domain_blockers.append(
            "snerv_receiver_frame_reconstruction_mae_exceeds_value_domain_gate"
        )
    per_pair_mse = np.mean(diff * diff, axis=(1, 2, 3, 4))
    per_pair_mae = np.mean(abs_diff, axis=(1, 2, 3, 4))
    per_pair_max_abs = np.max(abs_diff, axis=(1, 2, 3, 4))
    order = np.argsort(-per_pair_mse, kind="stable")
    worst_rows: list[dict[str, Any]] = []
    source_indices = [int(value) for value in source_pair_indices]
    for rank, pair_idx_np in enumerate(order[: max(0, int(worst_pair_count))]):
        pair_idx = int(pair_idx_np)
        source_pair_idx = (
            source_indices[pair_idx] if pair_idx < len(source_indices) else pair_idx
        )
        worst_rows.append(
            {
                "rank": int(rank),
                "pair_idx": pair_idx,
                "source_pair_idx": int(source_pair_idx),
                "mse_nchw255": float(per_pair_mse[pair_idx]),
                "mae_nchw255": float(per_pair_mae[pair_idx]),
                "max_abs_nchw255": float(per_pair_max_abs[pair_idx]),
            }
        )
    profile.update(
        {
            "mse_nchw255": mse,
            "mae_nchw255": mae,
            "max_abs_nchw255": max_abs,
            "rmse_nchw255": rmse,
            "segnet_frame1_rgb_mse_nchw255": scorer_distortion[
                "segnet_frame1_rgb_mse_nchw255"
            ],
            "posenet_yuv6_pair_mse": scorer_distortion["posenet_yuv6_pair_mse"],
            "posenet_yuv6_temporal_delta_mse": scorer_distortion[
                "posenet_yuv6_temporal_delta_mse"
            ],
            "scorer_domain_distortion_anatomy": scorer_distortion,
            "receiver_value_domain_gate": {
                "schema": "snerv_receiver_frame_reconstruction_value_domain_gate.v1",
                "decoded_std": decoded_std,
                "reference_std": reference_std,
                "std_ratio": std_ratio,
                "decoded_dynamic_range": decoded_dynamic_range,
                "reference_dynamic_range": reference_dynamic_range,
                "decoded_low_saturation_fraction": decoded_low_sat,
                "decoded_high_saturation_fraction": decoded_high_sat,
                "reference_low_saturation_fraction": reference_low_sat,
                "reference_high_saturation_fraction": reference_high_sat,
                "saturation_delta": saturation_delta,
                "min_std_ratio": SNERV_RECEIVER_RECON_MIN_STD_RATIO,
                "max_std_ratio": SNERV_RECEIVER_RECON_MAX_STD_RATIO,
                "max_saturation_delta": SNERV_RECEIVER_RECON_MAX_SATURATION_DELTA,
                "max_rmse_nchw255": SNERV_RECEIVER_RECON_MAX_RMSE_NCHW255,
                "max_mae_nchw255": SNERV_RECEIVER_RECON_MAX_MAE_NCHW255,
                "passed": not value_domain_blockers,
                "blockers": value_domain_blockers,
                **FALSE_AUTHORITY,
            },
            "worst_pairs_by_mse": worst_rows,
            "blockers": value_domain_blockers,
        }
    )
    return profile


def _snerv_scorer_domain_distortion_anatomy(
    decoded_pairs_nchw255: np.ndarray,
    reference_pairs_nchw255: np.ndarray,
    *,
    source_pair_indices: Sequence[int] = (),
    worst_pair_count: int = 0,
    comparison_domain: str,
) -> dict[str, Any]:
    """Split receiver-pixel error by the actual upstream scorer domains."""

    decoded = np.asarray(decoded_pairs_nchw255, dtype=np.float32)
    reference = np.asarray(reference_pairs_nchw255, dtype=np.float32)
    if (
        decoded.shape != reference.shape
        or decoded.ndim != 5
        or int(decoded.shape[1]) != 2
        or int(decoded.shape[2]) != 3
    ):
        raise ValueError(
            "SNeRV scorer-domain distortion anatomy expects matching "
            f"(pairs,2,3,H,W) tensors, got decoded={decoded.shape} "
            f"reference={reference.shape}"
        )
    diff = decoded - reference
    pair_rgb_mse = float(np.mean(diff * diff))
    frame0_mse = float(np.mean(diff[:, 0] * diff[:, 0]))
    frame1_mse = float(np.mean(diff[:, 1] * diff[:, 1]))
    decoded_yuv6 = _snerv_rgb_nchw255_to_yuv6_numpy(
        decoded.reshape(-1, 3, int(decoded.shape[-2]), int(decoded.shape[-1]))
    ).reshape(
        int(decoded.shape[0]),
        2,
        6,
        int(decoded.shape[-2]) // 2,
        int(decoded.shape[-1]) // 2,
    )
    reference_yuv6 = _snerv_rgb_nchw255_to_yuv6_numpy(
        reference.reshape(
            -1,
            3,
            int(reference.shape[-2]),
            int(reference.shape[-1]),
        )
    ).reshape(
        int(reference.shape[0]),
        2,
        6,
        int(reference.shape[-2]) // 2,
        int(reference.shape[-1]) // 2,
    )
    yuv6_diff = decoded_yuv6 - reference_yuv6
    yuv6_pair_mse = float(np.mean(yuv6_diff * yuv6_diff))
    temporal_delta_diff = (
        decoded_yuv6[:, 1] - decoded_yuv6[:, 0]
    ) - (
        reference_yuv6[:, 1] - reference_yuv6[:, 0]
    )
    temporal_delta_mse = float(np.mean(temporal_delta_diff * temporal_delta_diff))
    source_indices = [int(value) for value in source_pair_indices]
    worst_rows: list[dict[str, Any]] = []
    if worst_pair_count > 0:
        per_pair_frame1 = np.mean(diff[:, 1] * diff[:, 1], axis=(1, 2, 3))
        per_pair_yuv6 = np.mean(yuv6_diff * yuv6_diff, axis=(1, 2, 3, 4))
        per_pair_temporal = np.mean(
            temporal_delta_diff * temporal_delta_diff,
            axis=(1, 2, 3),
        )
        order = np.argsort(-per_pair_frame1, kind="stable")
        for rank, pair_idx_np in enumerate(order[: max(0, int(worst_pair_count))]):
            pair_idx = int(pair_idx_np)
            source_pair_idx = (
                source_indices[pair_idx] if pair_idx < len(source_indices) else pair_idx
            )
            worst_rows.append(
                {
                    "rank": int(rank),
                    "pair_idx": pair_idx,
                    "source_pair_idx": int(source_pair_idx),
                    "segnet_frame1_rgb_mse_nchw255": float(
                        per_pair_frame1[pair_idx]
                    ),
                    "posenet_yuv6_pair_mse": float(per_pair_yuv6[pair_idx]),
                    "posenet_yuv6_temporal_delta_mse": float(
                        per_pair_temporal[pair_idx]
                    ),
                }
            )
    return {
        "schema": "snerv_scorer_domain_distortion_anatomy.v1",
        "comparison_domain": str(comparison_domain),
        "scorer_geometry": {
            "segnet": "last_frame_rgb_before_segnet_resize_or_after_profile_resize",
            "posenet": "two_frame_upstream_rgb_to_yuv6_pair",
            "human_visual_fidelity_objective": False,
        },
        "human_visual_fidelity_objective": False,
        "pair_rgb_mse_nchw255": pair_rgb_mse,
        "frame0_rgb_mse_nchw255": frame0_mse,
        "segnet_frame1_rgb_mse_nchw255": frame1_mse,
        "segnet_frame1_to_pair_rgb_mse_ratio": frame1_mse
        / max(pair_rgb_mse, 1.0e-12),
        "posenet_yuv6_pair_mse": yuv6_pair_mse,
        "posenet_yuv6_temporal_delta_mse": temporal_delta_mse,
        "posenet_temporal_to_pair_yuv6_mse_ratio": temporal_delta_mse
        / max(yuv6_pair_mse, 1.0e-12),
        "worst_pairs_by_segnet_frame1_mse": worst_rows,
        **FALSE_AUTHORITY,
    }


def _snerv_rgb_nchw255_to_yuv6_numpy(rgb_nchw255: np.ndarray) -> np.ndarray:
    """NumPy equivalent of upstream frame_utils.rgb_to_yuv6."""

    rgb = np.asarray(rgb_nchw255, dtype=np.float32)
    if rgb.ndim != 4 or int(rgb.shape[1]) != 3:
        raise ValueError(f"rgb_nchw255 must be NCHW with 3 channels, got {rgb.shape}")
    h2 = int(rgb.shape[-2]) // 2
    w2 = int(rgb.shape[-1]) // 2
    if h2 <= 0 or w2 <= 0:
        raise ValueError(f"rgb_nchw255 spatial dims must be at least 2x2, got {rgb.shape}")
    rgb = rgb[:, :, : 2 * h2, : 2 * w2]
    red = rgb[:, 0]
    green = rgb[:, 1]
    blue = rgb[:, 2]
    y = np.clip(0.299 * red + 0.587 * green + 0.114 * blue, 0.0, 255.0)
    u = np.clip((blue - y) / 1.772 + 128.0, 0.0, 255.0)
    v = np.clip((red - y) / 1.402 + 128.0, 0.0, 255.0)
    u_sub = (
        u[:, 0::2, 0::2]
        + u[:, 1::2, 0::2]
        + u[:, 0::2, 1::2]
        + u[:, 1::2, 1::2]
    ) * 0.25
    v_sub = (
        v[:, 0::2, 0::2]
        + v[:, 1::2, 0::2]
        + v[:, 0::2, 1::2]
        + v[:, 1::2, 1::2]
    ) * 0.25
    y00 = y[:, 0::2, 0::2]
    y10 = y[:, 1::2, 0::2]
    y01 = y[:, 0::2, 1::2]
    y11 = y[:, 1::2, 1::2]
    return np.stack([y00, y10, y01, y11, u_sub, v_sub], axis=1).astype(
        np.float32,
        copy=False,
    )


def _snerv_official_skip_high_value_domain_gate(
    metadata: Mapping[str, Any],
    *,
    receiver_target_profile: Mapping[str, Any],
    receiver_export_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when compact official skip-high storage breaks pixels.

    Compact skip-high modes are byte attractive, but they are only useful for a
    long run when the receiver-decoded frames remain in the scorer's value
    domain.  This gate binds that decision to decoded-pixel evidence instead of
    letting a tiny scalar/channel/shared mean payload masquerade as progress.
    """

    mode = str(metadata.get("official_skip_high_mode") or "none")
    compact = mode in {"shared_mean", "channel_mean", "scalar_mean"}
    blockers: list[str] = []
    target_gate = (
        receiver_target_profile.get("receiver_value_domain_gate")
        if isinstance(receiver_target_profile, Mapping)
        else None
    )
    export_gate = (
        receiver_export_profile.get("receiver_value_domain_gate")
        if isinstance(receiver_export_profile, Mapping)
        else None
    )
    target_passed = isinstance(target_gate, Mapping) and target_gate.get("passed") is True
    export_passed = isinstance(export_gate, Mapping) and export_gate.get("passed") is True
    if compact:
        if not target_passed:
            blockers.append(
                "snerv_official_compact_skip_high_target_value_domain_not_passed"
            )
        if not export_passed:
            blockers.append(
                "snerv_official_compact_skip_high_export_value_domain_not_passed"
            )
        if mode == "scalar_mean" and blockers:
            blockers.append("snerv_official_scalar_mean_skip_high_collapse_risk")
        elif mode == "channel_mean" and blockers:
            blockers.append("snerv_official_channel_mean_skip_high_collapse_risk")
        elif mode == "shared_mean" and blockers:
            blockers.append("snerv_official_shared_mean_skip_high_collapse_risk")
    return {
        "schema": "snerv_official_skip_high_value_domain_gate.v1",
        "official_skip_high_mode": mode,
        "compact_skip_high_mode": compact,
        "receiver_target_value_domain_passed": target_passed,
        "receiver_export_value_domain_passed": export_passed,
        "passed": not blockers,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


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
        "snerv_official_mfu_hfr_tub_export_bound_semantics": (
            "receiver_payload_bound_not_source_forward_parity"
        ),
        "snerv_official_mfu_hfr_tub_receiver_payload_bound": False,
        "snerv_official_mfu_hfr_tub_source_forward_replay_bound": False,
        "snerv_official_mfu_hfr_tub_source_forward_replay_authority": False,
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
    selected_packet: bytes,
    selected_archive_metadata: Mapping[str, Any],
    package: Mapping[str, Any] | None,
    receiver_proof: Mapping[str, Any],
    official_trained_checkpoint_mapping_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the official-primitives request to a real receiver packet, fail-closed."""

    blockers = [str(blocker) for blocker in official_binding.get("blockers") or []]
    trained_checkpoint_mapping_manifest = _coerce_official_checkpoint_mapping_manifest(
        official_trained_checkpoint_mapping_manifest,
        source="snerv_mlx_native_receiver_bound_export_binding",
    )
    official_checkpoint_mapping_verified = _official_checkpoint_full_mapping_verified(
        trained_checkpoint_mapping_manifest
    )
    if official_checkpoint_mapping_verified:
        blockers = [
            blocker
            for blocker in blockers
            if blocker
            not in {
                SNERV_OFFICIAL_MFU_HFR_TUB_WEIGHT_MAPPING_BLOCKER,
                SNERV_OFFICIAL_TRAINED_CHECKPOINT_STATE_DICT_MAPPING_BLOCKER,
            }
        ]
    selected_authority = _selected_packet_official_payload_authority(selected_packet)
    tensor_map = _official_receiver_tensor_map_from_packet(selected_packet)
    tub_source_fixture_binding = _official_tub_source_fixture_binding_from_metadata(
        selected_archive_metadata
    )
    receiver_contract_proven = bool(
        dict(official_binding.get("official_receiver_runtime_decode_contract") or {}).get(
            "receiver_runtime_decode_proven"
        )
        is True
    )
    tensor_map = {
        **tensor_map,
        "receiver_runtime_decode_contract_proven": receiver_contract_proven,
        "receiver_runtime_decode_authority": False,
        "receiver_runtime_decode_authority_scope": (
            "tensor_map_only_selected_packet_frame_decode_required"
        ),
    }
    if bool(tensor_map.get("receiver_tensor_map_verified")):
        weight_mapping_blockers = (
            []
            if official_checkpoint_mapping_verified
            else [SNERV_OFFICIAL_MFU_HFR_TUB_WEIGHT_MAPPING_BLOCKER]
        )
        tensor_map = {
            **tensor_map,
            "official_state_dict_mapping_verified": (
                official_checkpoint_mapping_verified
            ),
            "official_weight_mapping_blocker_closed": (
                official_checkpoint_mapping_verified
            ),
            "official_weight_mapping_scope": (
                "receiver_payload_tensor_hashes_plus_upstream_state_dict_mapping"
                if official_checkpoint_mapping_verified
                else "receiver_payload_tensor_hashes_only_not_upstream_state_dict_mapping"
            ),
            "official_weight_mapping_blockers": weight_mapping_blockers,
            "official_trained_checkpoint_mapping_manifest": (
                trained_checkpoint_mapping_manifest
            ),
        }
    proof_passed = receiver_proof.get("runtime_consumption_proof_passed") is True
    receiver_satisfied = receiver_proof.get("receiver_contract_satisfied") is True
    archive_path = receiver_proof.get("archive_path")
    archive_sha256 = receiver_proof.get("archive_sha256")
    archive_bytes = receiver_proof.get("archive_bytes")
    out = dict(official_binding)
    out["schema"] = "snerv_official_mfu_hfr_tub_export_binding.v3"
    out["export_bound_to_receiver_packet"] = True
    out["receiver_native_export_bound"] = bool(
        selected_authority["frame_producing_official_export"]
    )
    out["official_export_bound"] = False
    out["official_export_bound_semantics"] = (
        "requires_receiver_export_native_mlx_export_and_source_forward_replay"
    )
    out["official_receiver_payload_bound"] = bool(
        selected_authority["official_decoder_payload_selected"]
    )
    out["official_source_forward_replay_bound"] = False
    out["source_forward_replay_bound_by_export"] = False
    out["official_tub_source_fixture_binding"] = tub_source_fixture_binding
    out["official_tub_source_fixture_replay_bound"] = bool(
        _official_tub_source_fixture_replay_bound(tub_source_fixture_binding)
    )
    out["official_tub_source_fixture_replay_passed"] = bool(
        tub_source_fixture_binding.get(
            "official_tub_temporal_encoder_output2_source_fixture_replay_passed"
        )
        is True
    )
    out["official_tub_source_forward_fixture_bound"] = bool(
        selected_archive_metadata.get(
            "snerv_official_tub_source_forward_fixture_bound"
        )
        is True
    )
    out["official_source_parity_blockers"] = [
        str(blocker)
        for blocker in selected_archive_metadata.get("official_source_parity_blockers")
        or _official_packet_source_parity_blockers(tub_source_fixture_binding)
    ]
    out["surrogate_receiver_payload_contract_emitted"] = not bool(
        selected_authority["frame_producing_official_export"]
    )
    out["official_receiver_payload_contract_emitted"] = bool(
        selected_authority["official_decoder_payload_selected"]
    )
    out["selected_packet_authority"] = selected_authority
    out["official_receiver_tensor_map"] = tensor_map
    out["receiver_bound_surrogate_export"] = {
        "schema": "snerv_official_receiver_bound_surrogate_export.v1",
        "kind": "snar1_linear_hf_generation_decoder_not_official_neural_graph",
        "packet_path": packet_path.as_posix(),
        "packet_bytes": int(packet_bytes),
        "packet_sha256": str(packet_sha256),
        "packet_source_faithful_stack": False,
        "packet_decoder_payload_codec": selected_archive_metadata.get("decoder_payload_codec"),
        "packet_decoder_payload_schema": selected_authority.get("decoder_payload_schema"),
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
        selected_packet_authority=selected_authority,
        surrogate_receiver_runtime_decode_passed=bool(proof_passed),
        surrogate_receiver_contract_satisfied=bool(receiver_satisfied),
    )
    out["official_receiver_payload_frame_replay"] = selected_authority.get(
        "official_receiver_payload_frame_replay"
    )
    out["unclosed_official_blockers"] = blockers
    out["blockers"] = blockers
    out["export_consumed_official_mfu"] = False
    out["export_consumed_official_hfr"] = False
    out["export_consumed_official_tub"] = False
    if selected_authority["frame_producing_official_export"]:
        out["export_consumed_official_mfu"] = True
        out["export_consumed_official_hfr"] = True
        out["export_consumed_official_tub"] = True
    out["source_forward_replay_authority"] = False
    out["official_receiver_runtime_decode_contract_proven"] = receiver_contract_proven
    out["receiver_runtime_decode_authority"] = bool(
        out["official_receiver_runtime_decode_contract_proven"]
        and selected_authority.get("frame_decode_succeeded") is True
        and selected_authority.get("receiver_payload_frame_replay_passed") is True
    )
    out["selected_packet_official_payload_runtime_decode_authority"] = bool(
        selected_authority["official_payload_runtime_decode_authority"]
    )
    out["selected_packet_frame_producing_official_export"] = bool(
        selected_authority["frame_producing_official_export"]
    )
    out.update(FALSE_AUTHORITY)
    return out


def _official_receiver_tensor_map_from_packet(packet: bytes) -> dict[str, Any]:
    """Extract receiver-bound official tensor custody from selected bytes."""

    base = {
        "schema": "snerv_official_mfu_hfr_tub_receiver_tensor_map.v1",
        "receiver_tensor_map_verified": False,
        "official_decoder_payload_selected": False,
        "row_count": 0,
        "total_tensor_bytes": 0,
        "category_counts": {},
        "category_bytes": {},
        "receiver_runtime_decode_contract_proven": False,
        "receiver_runtime_decode_authority": False,
        "receiver_runtime_decode_authority_scope": (
            "tensor_map_only_selected_packet_frame_decode_required"
        ),
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    try:
        decoded = unpack_snerv_archive(packet)
        header = inspect_decoder_payload_header(decoded.sections["decoder_payload"])
    except Exception as exc:
        return {
            **base,
            "blockers": ["snerv_official_receiver_tensor_map_packet_parse_failed"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    if str(header.get("schema") or "") != DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA:
        return {
            **base,
            "blockers": ["snerv_official_receiver_tensor_map_payload_not_official"],
            "decoder_payload_schema": str(header.get("schema") or ""),
        }
    rows = []
    tensor_names: set[str] = set()
    category_counts: dict[str, int] = {}
    category_bytes: dict[str, int] = {}
    for raw_row in header.get("tensor_manifest") or ():
        if not isinstance(raw_row, Mapping):
            return {
                **base,
                "official_decoder_payload_selected": True,
                "blockers": ["snerv_official_receiver_tensor_map_malformed_row"],
            }
        name = str(raw_row.get("name") or "")
        if not name:
            return {
                **base,
                "official_decoder_payload_selected": True,
                "blockers": ["snerv_official_receiver_tensor_map_malformed_row"],
                "error": "official tensor manifest row missing name",
            }
        try:
            nbytes, byte_key = _official_receiver_tensor_manifest_nbytes(raw_row)
            shape = _official_receiver_tensor_manifest_shape(raw_row)
        except (TypeError, ValueError) as exc:
            return {
                **base,
                "official_decoder_payload_selected": True,
                "blockers": ["snerv_official_receiver_tensor_map_invalid_tensor_bytes"],
                "error": str(exc),
            }
        dtype = str(raw_row.get("dtype") or "")
        if dtype != "float64_le":
            return {
                **base,
                "official_decoder_payload_selected": True,
                "blockers": ["snerv_official_receiver_tensor_map_invalid_tensor_dtype"],
                "error": f"official tensor {name!r} dtype {dtype!r} is not float64_le",
            }
        expected_nbytes = int(np.prod(shape)) * np.dtype("<f8").itemsize
        if expected_nbytes != int(nbytes):
            return {
                **base,
                "official_decoder_payload_selected": True,
                "blockers": ["snerv_official_receiver_tensor_map_invalid_tensor_bytes"],
                "error": (
                    f"official tensor {name!r} shape implies {expected_nbytes} "
                    f"bytes but manifest records {nbytes}"
                ),
            }
        sha256 = str(raw_row.get("sha256") or "")
        if not _official_receiver_tensor_manifest_sha256_valid(sha256):
            return {
                **base,
                "official_decoder_payload_selected": True,
                "blockers": ["snerv_official_receiver_tensor_map_invalid_tensor_sha256"],
                "error": f"official tensor {name!r} missing valid sha256",
            }
        category = _official_receiver_tensor_category(name)
        row = {
            "name": name,
            "category": category,
            "shape": [int(value) for value in shape],
            "dtype": dtype,
            "bytes": nbytes,
            "manifest_byte_key": byte_key,
            "sha256": sha256,
        }
        rows.append(row)
        tensor_names.add(name)
        category_counts[category] = category_counts.get(category, 0) + 1
        category_bytes[category] = category_bytes.get(category, 0) + nbytes
    manifest_sha = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    try:
        required_tensor_names = _official_receiver_required_tensor_keys_from_header(
            header,
            present_tensor_names=tensor_names,
        )
    except (TypeError, ValueError) as exc:
        return {
            **base,
            "official_decoder_payload_selected": True,
            "row_count": len(rows),
            "total_tensor_bytes": int(sum(int(row["bytes"]) for row in rows)),
            "category_counts": dict(sorted(category_counts.items())),
            "category_bytes": dict(sorted(category_bytes.items())),
            "tensor_manifest_sha256": manifest_sha,
            "rows": rows,
            "blockers": ["snerv_official_receiver_tensor_map_required_key_contract_invalid"],
            "error": str(exc),
        }
    missing_required = sorted(required_tensor_names.difference(tensor_names))
    blockers = []
    if not rows:
        blockers.append("snerv_official_receiver_tensor_map_rows_missing")
    if missing_required:
        blockers.append("snerv_official_receiver_tensor_map_missing_required_tensors")
    return {
        **base,
        "receiver_tensor_map_verified": bool(rows) and not blockers,
        "official_decoder_payload_selected": True,
        "decoder_payload_schema": str(header.get("schema") or ""),
        "decoder_payload_codec": str(header.get("codec") or ""),
        "row_count": len(rows),
        "total_tensor_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "category_counts": dict(sorted(category_counts.items())),
        "category_bytes": dict(sorted(category_bytes.items())),
        "required_tensor_key_count": len(required_tensor_names),
        "missing_required_tensor_keys": missing_required,
        "tensor_manifest_sha256": manifest_sha,
        "rows": rows,
        "blockers": blockers,
    }


def _official_receiver_tensor_manifest_nbytes(
    raw_row: Mapping[str, Any],
) -> tuple[int, str]:
    """Read official tensor byte count across manifest dialects.

    The active receiver encoder emits ``bytes``.  Earlier source-parity helpers
    and some audit artifacts used ``nbytes``.  The tensor map is a custody
    profiler, so it must accept either dialect while refusing ambiguous or
    missing accounting.
    """

    has_bytes = raw_row.get("bytes") is not None
    has_nbytes = raw_row.get("nbytes") is not None
    if not has_bytes and not has_nbytes:
        raise ValueError("official tensor manifest row missing bytes/nbytes")
    bytes_value = int(raw_row["bytes"]) if has_bytes else None
    nbytes_value = int(raw_row["nbytes"]) if has_nbytes else None
    if bytes_value is not None and nbytes_value is not None:
        if bytes_value != nbytes_value:
            raise ValueError(
                "official tensor manifest row has mismatched bytes and nbytes"
            )
        value = bytes_value
        key = "bytes+nbytes"
    elif bytes_value is not None:
        value = bytes_value
        key = "bytes"
    else:
        value = int(nbytes_value)
        key = "nbytes"
    if value <= 0:
        raise ValueError("official tensor manifest row has non-positive byte count")
    return int(value), key


def _official_receiver_tensor_manifest_shape(
    raw_row: Mapping[str, Any],
) -> tuple[int, ...]:
    shape = tuple(int(value) for value in raw_row.get("shape") or ())
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("official tensor manifest row has invalid shape")
    return shape


def _official_receiver_tensor_manifest_sha256_valid(value: str) -> bool:
    digest = str(value or "").strip().lower()
    return len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest)


def _official_receiver_required_tensor_keys_from_header(
    header: Mapping[str, Any],
    *,
    present_tensor_names: set[str],
) -> set[str]:
    """Return tensor names required for executable official receiver replay."""

    raw_spec = header.get("mfu_spec")
    if not isinstance(raw_spec, Mapping):
        raise ValueError("official receiver tensor map missing mfu_spec")
    num_blocks = int(raw_spec.get("num_blocks", 0))
    if num_blocks < 0:
        raise ValueError("official receiver tensor map has negative num_blocks")
    required = set(OFFICIAL_MFU_HFR_TUB_REQUIRED_TENSOR_KEYS)
    for prefix in ("mfu.rb_mid", "mfu.rb_high"):
        for idx in range(num_blocks):
            for conv in ("conv1", "conv2"):
                required.add(f"{prefix}.block{idx}.{conv}.weight")
                required.add(f"{prefix}.block{idx}.{conv}.bias")
    output2_names = {"tub.temporal_encoder_concat", "tub.output2_raw"}
    storage = header.get("tub_output2_storage")
    output2_required = False
    if isinstance(storage, Mapping):
        output2_required = bool(storage.get("stored"))
    present_output2 = present_tensor_names.intersection(output2_names)
    if output2_required or present_output2:
        required.update(output2_names)
    return required


def _official_receiver_tensor_category(name: str) -> str:
    if name.startswith("mfu."):
        return "official_mfu_weight_payload"
    if name.startswith("hfr."):
        return "official_hfr_weight_payload"
    if name.startswith("inputs.tub."):
        return "official_tub_input_payload"
    if name.startswith("inputs.mfu."):
        return "official_mfu_input_payload"
    if name in {"tub.temporal_encoder_concat", "tub.output2_raw"}:
        return "official_tub_output2_payload"
    if name.startswith("tub."):
        return "official_tub_weight_payload"
    return "official_decoder_graph_topology_payload"


def _selected_packet_official_payload_frame_replay(packet: bytes) -> dict[str, Any]:
    """Prove selected official payload bytes are the archive frame producer."""

    out: dict[str, Any] = {
        "schema": "snerv_official_mfu_hfr_tub_receiver_payload_frame_replay.v1",
        "packet_sha256": _sha256_bytes(packet),
        "packet_bytes": len(packet),
        "official_decoder_payload_selected": False,
        "receiver_payload_frame_replay_attempted": True,
        "official_receiver_runtime_decode_proven": False,
        "payload_frame_decode_succeeded": False,
        "archive_frame_decode_succeeded": False,
        "frame_decode_succeeded": False,
        "payload_archive_frames_match": False,
        "receiver_payload_frame_replay_passed": False,
        "source_forward_replay_bound_by_frame_replay": False,
        "source_forward_replay_verified": False,
        "source_forward_replay_authority": False,
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    try:
        decoded = unpack_snerv_archive(packet)
        decoder_payload = decoded.sections["decoder_payload"]
        header = inspect_decoder_payload_header(decoder_payload)
        schema = str(header.get("schema") or "")
        out["decoder_payload_schema"] = schema
        out["decoder_payload_codec"] = str(header.get("codec") or "")
        out["official_decoder_payload_selected"] = (
            schema == DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA
        )
        if out["official_decoder_payload_selected"] is not True:
            return {
                **out,
                "blockers": [
                    "snerv_official_mfu_hfr_tub_decoder_payload_not_selected"
                ],
            }

        payload_obj = decode_official_mfu_hfr_tub_decoder_payload(decoder_payload)
        primitive_proof = payload_obj.execute()
        payload_flat = np.asarray(payload_obj.decode_frames(), dtype=np.float32)
        archive_pairs = np.asarray(decode_snerv_archive_frames(packet), dtype=np.float32)
        if archive_pairs.ndim != 5:
            raise SnervMlxNativeExportError(
                "official archive frame replay expected pair tensor rank 5, "
                f"got {archive_pairs.shape}"
            )
        archive_flat = archive_pairs.reshape(
            int(archive_pairs.shape[0]) * int(archive_pairs.shape[1]),
            int(archive_pairs.shape[2]),
            int(archive_pairs.shape[3]),
            int(archive_pairs.shape[4]),
        )
        payload_finite = bool(np.isfinite(payload_flat).all())
        archive_finite = bool(np.isfinite(archive_pairs).all())
        shape_matches = tuple(int(v) for v in payload_flat.shape) == tuple(
            int(v) for v in archive_flat.shape
        )
        payload_frame_sha = _sha256_frame_array(payload_flat)
        archive_flat_sha = _sha256_frame_array(archive_flat)
        frame_hashes_match = bool(shape_matches and payload_frame_sha == archive_flat_sha)
        passed = bool(
            primitive_proof.get("receiver_runtime_decode_proven") is True
            and payload_finite
            and archive_finite
            and shape_matches
            and frame_hashes_match
        )
        blockers = []
        if primitive_proof.get("receiver_runtime_decode_proven") is not True:
            blockers.append(
                "snerv_official_mfu_hfr_tub_receiver_runtime_decode_not_proven"
            )
        if not payload_finite:
            blockers.append("snerv_official_mfu_hfr_tub_payload_frames_nonfinite")
        if not archive_finite:
            blockers.append("snerv_official_mfu_hfr_tub_archive_frames_nonfinite")
        if not shape_matches:
            blockers.append(
                "snerv_official_mfu_hfr_tub_payload_archive_frame_shape_mismatch"
            )
        if shape_matches and not frame_hashes_match:
            blockers.append(
                "snerv_official_mfu_hfr_tub_payload_archive_frame_hash_mismatch"
            )
        return {
            **out,
            "decoder_payload_sha256": _sha256_bytes(decoder_payload),
            "decoder_payload_bytes": len(decoder_payload),
            "official_receiver_runtime_decode_proven": bool(
                primitive_proof.get("receiver_runtime_decode_proven") is True
            ),
            "official_receiver_runtime_decode_proof": primitive_proof,
            "payload_frame_decode_succeeded": True,
            "archive_frame_decode_succeeded": True,
            "frame_decode_succeeded": bool(passed),
            "payload_frame_shape": [int(v) for v in payload_flat.shape],
            "archive_frame_shape": [int(v) for v in archive_pairs.shape],
            "archive_flat_frame_shape": [int(v) for v in archive_flat.shape],
            "payload_frames_finite": payload_finite,
            "archive_frames_finite": archive_finite,
            "payload_archive_frame_shape_matches": shape_matches,
            "payload_flat_frame_sha256": payload_frame_sha,
            "archive_pair_frame_sha256": _sha256_frame_array(archive_pairs),
            "archive_flat_frame_sha256": archive_flat_sha,
            "payload_archive_frames_match": frame_hashes_match,
            "receiver_payload_frame_replay_passed": passed,
            "blockers": _ordered_unique(blockers),
        }
    except Exception as exc:
        return {
            **out,
            "failure": f"{type(exc).__name__}: {exc}",
            "blockers": [
                "snerv_official_mfu_hfr_tub_receiver_payload_frame_replay_failed",
                f"snerv_official_mfu_hfr_tub_receiver_payload_frame_replay_exception_{type(exc).__name__}",
            ],
        }


def _selected_packet_official_payload_authority(packet: bytes) -> dict[str, Any]:
    """Classify selected receiver bytes; intent metadata is not authority."""

    out: dict[str, Any] = {
        "schema": "snerv_selected_packet_official_payload_authority.v1",
        "packet_sha256": _sha256_bytes(packet),
        "packet_bytes": len(packet),
        "decoder_payload_schema": None,
        "decoder_payload_codec": None,
        "official_decoder_payload_selected": False,
        "linear_surrogate_decoder_selected": False,
        "frame_decode_attempted": False,
        "frame_decode_succeeded": False,
        "receiver_payload_frame_replay_passed": False,
        "official_payload_runtime_decode_authority": False,
        "frame_producing_official_export": False,
        "status": "unclassified",
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    try:
        decoded = unpack_snerv_archive(packet)
        header = inspect_decoder_payload_header(decoded.sections["decoder_payload"])
        schema = str(header.get("schema") or "")
        out["decoder_payload_schema"] = schema
        out["decoder_payload_codec"] = str(header.get("codec") or "")
        out["official_decoder_payload_selected"] = (
            schema == DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA
        )
        out["linear_surrogate_decoder_selected"] = not bool(
            out["official_decoder_payload_selected"]
        )
        out["frame_decode_attempted"] = True
        if out["official_decoder_payload_selected"]:
            frame_replay = _selected_packet_official_payload_frame_replay(packet)
            out["official_receiver_payload_frame_replay"] = frame_replay
            out["receiver_payload_frame_replay_passed"] = bool(
                frame_replay.get("receiver_payload_frame_replay_passed") is True
            )
            out["frame_decode_succeeded"] = bool(
                frame_replay.get("archive_frame_decode_succeeded") is True
            )
            if frame_replay.get("archive_frame_shape") is not None:
                out["decoded_frame_shape"] = list(frame_replay["archive_frame_shape"])
            if frame_replay.get("receiver_payload_frame_replay_passed") is True:
                out["status"] = "frame_producing_official_export"
                out["official_payload_runtime_decode_authority"] = True
                out["frame_producing_official_export"] = True
            else:
                out["status"] = "official_payload_selected_not_frame_producing"
                out["blockers"] = _ordered_unique(
                    [
                        "snerv_official_mfu_hfr_tub_selected_payload_not_frame_producing",
                        *(
                            str(blocker)
                            for blocker in frame_replay.get("blockers") or ()
                        ),
                    ]
                )
            return out
        try:
            frames = decode_snerv_archive_frames(packet)
        except Exception as exc:
            out["frame_decode_error"] = f"{type(exc).__name__}: {exc}"
            if out["official_decoder_payload_selected"]:
                out["status"] = "official_payload_selected_not_frame_producing"
                out["blockers"] = [
                    "snerv_official_mfu_hfr_tub_selected_payload_not_frame_producing"
                ]
            else:
                out["status"] = "surrogate_packet_frame_decode_failed"
                out["blockers"] = ["snerv_selected_surrogate_packet_frame_decode_failed"]
        else:
            out["frame_decode_succeeded"] = True
            out["decoded_frame_shape"] = [int(value) for value in frames.shape]
            if out["official_decoder_payload_selected"]:
                out["status"] = "frame_producing_official_export"
                out["official_payload_runtime_decode_authority"] = True
                out["frame_producing_official_export"] = True
            else:
                out["status"] = "surrogate_linear_decoder_frame_producing"
                out["blockers"] = [
                    "snerv_selected_packet_uses_linear_surrogate_decoder_payload"
                ]
    except Exception as exc:
        out["status"] = "packet_parse_failed"
        out["packet_parse_error"] = f"{type(exc).__name__}: {exc}"
        out["blockers"] = ["snerv_selected_packet_official_payload_authority_parse_failed"]
    return out


def _official_primitives_blocker_evidence(
    blockers: Sequence[str],
    *,
    selected_packet_authority: Mapping[str, Any],
    surrogate_receiver_runtime_decode_passed: bool,
    surrogate_receiver_contract_satisfied: bool,
) -> list[dict[str, Any]]:
    specs = {
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload": {
            "missing_artifact": ("native MLX train/export selected packet with official MFU/HFR/TUB decoder payload"),
            "current_evidence": (
                "receiver-visible official decoder payload grammar exists; selected "
                f"packet status is {selected_packet_authority.get('status')}"
            ),
            "closure_test": (
                "train/export writes decoder_payload schema "
                f"{DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA} and generated "
                "inflate consumes it into 0.raw"
            ),
        },
        "snerv_official_mfu_hfr_tub_weight_mapping_missing": {
            "missing_artifact": ("official state_dict-to-receiver tensor map with per-tensor bytes and sha256"),
            "current_evidence": (
                "portable MFU/HFR/TUB primitives and receiver payload grammar exist, "
                "but native export does not consume trained official weights"
            ),
            "closure_test": (
                "map official encoder/MFU/HFR/TUB/residual-block tensors into the "
                "receiver payload and verify every mapped tensor hash"
            ),
        },
        "snerv_official_mfu_hfr_tub_source_forward_replay_missing": {
            "missing_artifact": ("same-input official torch forward replay against the portable receiver graph"),
            "current_evidence": (
                "selected receiver payload frame replay may pass, but no official "
                "source graph output is compared"
            ),
            "closure_test": (
                "run official source forward and portable receiver forward on the "
                "same frames/weights and record max error plus output sha256"
            ),
        },
    }
    rows = []
    for blocker in blockers:
        spec = specs.get(str(blocker), {})
        receiver_payload_binding_closed = bool(
            str(blocker)
            == "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
            and selected_packet_authority.get("frame_producing_official_export") is True
            and selected_packet_authority.get("receiver_payload_frame_replay_passed")
            is True
        )
        rows.append(
            {
                "blocker": str(blocker),
                "closed": receiver_payload_binding_closed,
                "missing_artifact": spec.get("missing_artifact", "unspecified"),
                "current_evidence": spec.get("current_evidence", "unspecified"),
                "closure_test": spec.get("closure_test", "unspecified"),
                "selected_packet_status": selected_packet_authority.get("status"),
                "selected_packet_decoder_payload_schema": selected_packet_authority.get(
                    "decoder_payload_schema"
                ),
                "selected_packet_official_decoder_payload_selected": bool(
                    selected_packet_authority.get("official_decoder_payload_selected")
                    is True
                ),
                "selected_packet_frame_producing_official_export": bool(
                    selected_packet_authority.get("frame_producing_official_export")
                    is True
                ),
                "selected_packet_receiver_payload_frame_replay_passed": bool(
                    selected_packet_authority.get("receiver_payload_frame_replay_passed")
                    is True
                ),
                "surrogate_receiver_runtime_decode_passed": bool(surrogate_receiver_runtime_decode_passed),
                "surrogate_receiver_contract_satisfied": bool(surrogate_receiver_contract_satisfied),
                "receiver_payload_binding_authority": bool(
                    receiver_payload_binding_closed
                ),
                "source_forward_authority": False,
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
        repo_root=_repo_root(None),
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
    repo_root: str | Path,
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
    receiver_contract = build_snerv_official_receiver_runtime_decode_contract(
        repo_root=repo_root,
    )
    return {
        "schema": "snerv_official_mfu_hfr_tub_export_binding.v2",
        "adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        "primitive_modules_available": True,
        "current_snar_decoder_payload_schema": "linear_hf_generation_decoder_only",
        "available_official_decoder_payload_schema": DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA,
        "linear_hf_generation_decoder_compatible_with_official_neural_graph": False,
        "receiver_payload_contract_emitted": True,
        "official_receiver_payload_contract_available": bool(
            receiver_contract.get("receiver_archive_payload_bound") is True
        ),
        "official_receiver_runtime_decode_contract": receiver_contract,
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
        "receiver_runtime_decode_contract_proven": bool(
            receiver_contract.get("receiver_runtime_decode_proven") is True
        ),
        "receiver_runtime_decode_authority": False,
        "receiver_runtime_decode_authority_scope": (
            "tensor_map_only_selected_packet_frame_decode_required"
        ),
        "native_mlx_export_bound_to_official_payload": False,
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
    fc_dim, fc_dim_source = _fc_dim_resolution_from_candidate(candidate)
    requested_tub_output2_store = bool(
        candidate.get(
            "official_tub_output2_store_for_receiver_proof",
            candidate.get("snerv_official_tub_output2_store_for_receiver_proof", False),
        )
    )
    tub_output2_export_mode = str(
        candidate.get(
            "official_tub_output2_export_mode",
            candidate.get(
                "snerv_official_tub_output2_export_mode",
                "auto_elide",
            ),
        )
    ).strip().lower()
    if tub_output2_export_mode not in {"auto_elide", "proof_only"}:
        raise SnervCarrierError(
            "official_tub_output2_export_mode must be one of "
            "['auto_elide', 'proof_only']"
        )
    # Candidate parsing is the train/export automation boundary. TUB output_2
    # is source-parity useful, but it is not frame-decode score-causal in the
    # current receiver. Only an explicit proof-only candidate may pay for it.
    store_tub_output2 = bool(
        requested_tub_output2_store and tub_output2_export_mode == "proof_only"
    )
    return SnervModelSizeConfig(
        fc_dim=fc_dim,
        fc_dim_source=fc_dim_source,
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
        official_skip_high_mode=str(
            candidate.get(
                "official_skip_high_mode",
                candidate.get("snerv_official_skip_high_mode", "full"),
            )
        ),
        official_tub_output2_store_for_receiver_proof=store_tub_output2,
        official_tub_output2_export_mode=tub_output2_export_mode,
        official_tub_output2_store_for_receiver_proof_requested=(
            requested_tub_output2_store
        ),
        adapter=adapter,
    )


def _hard_byte_ceiling_from_candidate(candidate: Mapping[str, Any]) -> int | None:
    value = candidate.get("hard_byte_ceiling", candidate.get("snerv_hard_byte_ceiling"))
    if value is None:
        return None
    try:
        ceiling = int(value)
    except (TypeError, ValueError) as exc:
        raise SnervCarrierError("hard_byte_ceiling must be an integer") from exc
    if ceiling <= 0:
        raise SnervCarrierError("hard_byte_ceiling must be positive")
    return ceiling


def _build_snerv_mlx_native_byte_cap_control(
    *,
    candidate: Mapping[str, Any],
    hard_byte_ceiling: int | None,
    packet_source: str,
    packet_sha256: str,
    packet_bytes: int,
    section_bytes: Mapping[str, int] | None = None,
    decoder_payload_codec: str | None = None,
    lf_payload_codec: str | None = None,
    official_receiver_tensor_map: Mapping[str, Any] | None = None,
    archive_bytes: int | None,
    archive_sha256: str | None,
    receiver_proof_passed: bool,
    receiver_contract_satisfied: bool,
    run_archive_export: bool,
) -> dict[str, Any]:
    controller = candidate.get("byte_cap_controller")
    controller_payload = dict(controller) if isinstance(controller, Mapping) else None
    measured_section_bytes = {
        str(name): int(value)
        for name, value in (section_bytes or {}).items()
        if int(value) >= 0
    }
    lf_payload_bytes = int(measured_section_bytes.get("lf_payload", 0))
    largest_section_name: str | None = None
    largest_section_bytes: int | None = None
    if measured_section_bytes:
        largest_section_name, largest_section_bytes = max(
            measured_section_bytes.items(),
            key=lambda item: int(item[1]),
        )
    packet_denominator = max(int(packet_bytes), 1)
    archive_denominator = (
        max(int(archive_bytes), 1) if archive_bytes is not None else None
    )
    section_pressure_rows = _snerv_archive_section_pressure_rows(
        measured_section_bytes,
        packet_bytes=int(packet_bytes),
        archive_bytes=archive_bytes,
        hard_byte_ceiling=hard_byte_ceiling,
    )
    official_component_rows = _snerv_official_decoder_component_pressure_rows(
        official_receiver_tensor_map,
        decoder_payload_bytes=measured_section_bytes.get("decoder_payload"),
        packet_bytes=int(packet_bytes),
        archive_bytes=archive_bytes,
        hard_byte_ceiling=hard_byte_ceiling,
    )
    proof_only_component_rows = [
        row
        for row in official_component_rows
        if row.get("receiver_frame_decode_bound") is False
    ]
    non_score_causal_component_rows = [
        row
        for row in official_component_rows
        if row.get("train_time_loss_coupled") is False
        or row.get("receiver_frame_decode_bound") is False
    ]
    largest_pressure_row = max(
        section_pressure_rows + official_component_rows,
        key=lambda row: int(row.get("bytes", 0)),
        default=None,
    )
    base = {
        "schema": "snerv_mlx_native_hard_byte_ceiling_control.v1",
        "attached": hard_byte_ceiling is not None,
        "hard_byte_ceiling": hard_byte_ceiling,
        "packet_source": str(packet_source),
        "packet_sha256": str(packet_sha256),
        "packet_bytes": int(packet_bytes),
        "decoder_payload_codec": (
            str(decoder_payload_codec) if decoder_payload_codec else None
        ),
        "lf_payload_codec": str(lf_payload_codec) if lf_payload_codec else None,
        "section_bytes": measured_section_bytes,
        "section_pressure_rows": section_pressure_rows,
        "official_decoder_payload_component_rows": official_component_rows,
        "official_decoder_payload_component_bytes": {
            str(row["name"]): int(row["bytes"])
            for row in official_component_rows
        },
        "official_decoder_payload_proof_only_component_bytes": {
            str(row["name"]): int(row["bytes"])
            for row in proof_only_component_rows
        },
        "official_decoder_payload_proof_only_component_total_bytes": int(
            sum(int(row["bytes"]) for row in proof_only_component_rows)
        ),
        "official_decoder_payload_non_score_causal_component_bytes": {
            str(row["name"]): int(row["bytes"])
            for row in non_score_causal_component_rows
        },
        "official_decoder_payload_non_score_causal_component_total_bytes": int(
            sum(int(row["bytes"]) for row in non_score_causal_component_rows)
        ),
        "official_decoder_payload_non_score_causal_byte_cap_action": (
            "elide_or_implement_source_faithful_receiver_frame_decode_before_score_candidate"
            if non_score_causal_component_rows
            else "none"
        ),
        "official_decoder_payload_component_pressure_bound": bool(
            official_component_rows
        ),
        "largest_pressure_scope": (
            str(largest_pressure_row.get("scope")) if largest_pressure_row else None
        ),
        "largest_pressure_name": (
            str(largest_pressure_row.get("name")) if largest_pressure_row else None
        ),
        "largest_pressure_bytes": (
            int(largest_pressure_row.get("bytes", 0)) if largest_pressure_row else None
        ),
        "lf_payload_bytes": lf_payload_bytes,
        "lf_payload_fraction_of_packet": float(lf_payload_bytes / packet_denominator),
        "lf_payload_fraction_of_archive": (
            float(lf_payload_bytes / archive_denominator)
            if archive_denominator is not None
            else None
        ),
        "largest_section_name": largest_section_name,
        "largest_section_bytes": (
            int(largest_section_bytes) if largest_section_bytes is not None else None
        ),
        "lf_payload_is_largest_section": largest_section_name == "lf_payload",
        "archive_bytes": int(archive_bytes) if archive_bytes is not None else None,
        "archive_sha256": str(archive_sha256) if archive_sha256 else None,
        "run_archive_export": bool(run_archive_export),
        "receiver_proof_passed": bool(receiver_proof_passed),
        "receiver_contract_satisfied": bool(receiver_contract_satisfied),
        "archive_bytes_authoritative": bool(
            run_archive_export
            and archive_bytes is not None
            and receiver_proof_passed
            and receiver_contract_satisfied
        ),
        "byte_cap_controller": controller_payload,
        "authority": (
            "measured_receiver_proven_archive_zip_bytes"
            if (
                run_archive_export
                and archive_bytes is not None
                and receiver_proof_passed
                and receiver_contract_satisfied
            )
            else "archive_bytes_not_authoritative_until_receiver_proof_passes"
        ),
        **FALSE_AUTHORITY,
    }
    if hard_byte_ceiling is None:
        return {
            **base,
            "under_hard_byte_ceiling": None,
            "delta_bytes_vs_hard_byte_ceiling": None,
            "enforced": False,
            "archive_overrun_bytes": None,
            "lf_payload_exceeds_hard_byte_ceiling": None,
            "lf_payload_can_cover_archive_overrun": None,
            "blockers": [],
        }

    blockers: list[str] = []
    under: bool | None = None
    delta: int | None = None
    archive_overrun_bytes: int | None = None
    lf_payload_exceeds_hard_byte_ceiling = bool(
        lf_payload_bytes > int(hard_byte_ceiling)
    )
    lf_payload_can_cover_archive_overrun: bool | None = None
    if not run_archive_export:
        blockers.append("snerv_mlx_native_hard_byte_ceiling_not_enforced_archive_export_disabled")
    elif archive_bytes is None:
        blockers.append("snerv_mlx_native_hard_byte_ceiling_archive_bytes_missing")
    elif not receiver_proof_passed or not receiver_contract_satisfied:
        blockers.append(
            "snerv_mlx_native_hard_byte_ceiling_receiver_proof_missing_or_failed"
        )
    else:
        delta = int(archive_bytes) - int(hard_byte_ceiling)
        under = delta <= 0
        if not under:
            archive_overrun_bytes = int(delta)
            lf_payload_can_cover_archive_overrun = bool(
                lf_payload_bytes >= archive_overrun_bytes
            )
            blockers.append("snerv_mlx_native_archive_exceeds_hard_byte_ceiling")
            if lf_payload_exceeds_hard_byte_ceiling:
                blockers.append("snerv_lf_payload_exceeds_hard_byte_ceiling")
            if largest_section_name == "lf_payload":
                blockers.append(
                    "snerv_lf_payload_is_largest_section_on_over_ceiling_export"
                )
            if largest_section_name == "decoder_payload":
                blockers.append(
                    "snerv_decoder_payload_is_largest_section_on_over_ceiling_export"
                )
            if (
                lf_payload_exceeds_hard_byte_ceiling
                or lf_payload_can_cover_archive_overrun
            ):
                blockers.append(
                    "snerv_lf_payload_recode_or_representation_change_required_for_hard_ceiling"
                )
            if (
                largest_section_name == "decoder_payload"
                and measured_section_bytes.get("decoder_payload", 0)
                >= archive_overrun_bytes
            ):
                blockers.append(
                    "snerv_decoder_payload_component_recode_or_modelsize_change_required_for_hard_ceiling"
                )
                if official_component_rows:
                    blockers.append(
                        "snerv_official_mfu_hfr_tub_component_byte_pressure_requires_modelsize_waterfill"
                    )
                if proof_only_component_rows:
                    blockers.append(
                        "snerv_official_mfu_hfr_tub_proof_only_component_bytes_require_ablation_before_modelsize_growth"
                    )

    return {
        **base,
        "under_hard_byte_ceiling": under,
        "delta_bytes_vs_hard_byte_ceiling": delta,
        "archive_overrun_bytes": archive_overrun_bytes,
        "lf_payload_exceeds_hard_byte_ceiling": lf_payload_exceeds_hard_byte_ceiling,
        "lf_payload_can_cover_archive_overrun": lf_payload_can_cover_archive_overrun,
        "enforced": bool(
            run_archive_export
            and archive_bytes is not None
            and receiver_proof_passed
            and receiver_contract_satisfied
        ),
        "blockers": _ordered_unique(blockers),
    }


def _snerv_archive_section_pressure_rows(
    section_bytes: Mapping[str, int],
    *,
    packet_bytes: int,
    archive_bytes: int | None,
    hard_byte_ceiling: int | None,
) -> list[dict[str, Any]]:
    """Build exact SNAR1 section rows for byte-cap/modelsize controllers."""

    packet_denominator = max(int(packet_bytes), 1)
    archive_denominator = (
        max(int(archive_bytes), 1) if archive_bytes is not None else None
    )
    largest_name: str | None = None
    if section_bytes:
        largest_name = max(section_bytes.items(), key=lambda item: int(item[1]))[0]
    rows: list[dict[str, Any]] = []
    for name, nbytes in sorted(
        section_bytes.items(), key=lambda item: (-int(item[1]), str(item[0]))
    ):
        bytes_int = int(nbytes)
        rows.append(
            {
                "scope": "snar_archive_section",
                "name": str(name),
                "bytes": bytes_int,
                "byte_basis": "exact_receiver_packet_section_bytes",
                "fraction_of_packet": float(bytes_int / packet_denominator),
                "fraction_of_archive": (
                    float(bytes_int / archive_denominator)
                    if archive_denominator is not None
                    else None
                ),
                "is_largest_section": str(name) == largest_name,
                "exceeds_hard_byte_ceiling": (
                    bool(bytes_int > int(hard_byte_ceiling))
                    if hard_byte_ceiling is not None
                    else None
                ),
            }
        )
    return rows


def _snerv_official_decoder_component_pressure_rows(
    official_receiver_tensor_map: Mapping[str, Any] | None,
    *,
    decoder_payload_bytes: int | None,
    packet_bytes: int,
    archive_bytes: int | None,
    hard_byte_ceiling: int | None,
) -> list[dict[str, Any]]:
    """Expose official MFU/HFR/TUB raw tensor byte pressure inside decoder_payload.

    The current official payload stores all receiver tensors under one LZMA
    decoder section, so component rows are not exact ZIP byte spans.  They are
    real receiver-manifest tensor byte masses, which is the right control
    surface for modelsize, QAT, and waterfilling before a future packet compiler
    splits or reorders the stream.
    """

    if not isinstance(official_receiver_tensor_map, Mapping):
        return []
    if official_receiver_tensor_map.get("receiver_tensor_map_verified") is not True:
        return []
    category_bytes_raw = official_receiver_tensor_map.get("category_bytes")
    if not isinstance(category_bytes_raw, Mapping):
        return []
    packet_denominator = max(int(packet_bytes), 1)
    archive_denominator = (
        max(int(archive_bytes), 1) if archive_bytes is not None else None
    )
    decoder_denominator = (
        max(int(decoder_payload_bytes), 1)
        if decoder_payload_bytes is not None
        else None
    )
    total_tensor_bytes = max(
        int(official_receiver_tensor_map.get("total_tensor_bytes") or 0),
        1,
    )
    rows: list[dict[str, Any]] = []
    for name, value in sorted(
        category_bytes_raw.items(), key=lambda item: (-int(item[1]), str(item[0]))
    ):
        bytes_int = int(value)
        binding = _snerv_official_decoder_component_render_binding(str(name))
        rows.append(
            {
                "scope": "official_mfu_hfr_tub_decoder_payload_category",
                "name": str(name),
                "bytes": bytes_int,
                "byte_basis": (
                    "receiver_tensor_manifest_raw_float64_bytes_inside_single_lzma_decoder_payload"
                ),
                "fraction_of_official_raw_tensor_bytes": float(
                    bytes_int / total_tensor_bytes
                ),
                "fraction_of_decoder_payload_section": (
                    float(bytes_int / decoder_denominator)
                    if decoder_denominator is not None
                    else None
                ),
                "fraction_of_packet": float(bytes_int / packet_denominator),
                "fraction_of_archive": (
                    float(bytes_int / archive_denominator)
                    if archive_denominator is not None
                    else None
                ),
                "exceeds_hard_byte_ceiling": (
                    bool(bytes_int > int(hard_byte_ceiling))
                    if hard_byte_ceiling is not None
                    else None
                ),
                **binding,
            }
        )
    return rows


def _snerv_official_decoder_component_render_binding(name: str) -> dict[str, Any]:
    """Classify official payload categories by scored-frame decode causality.

    The official packet can carry tensors for receiver primitive proofs that are
    not consumed by ``decode_frames()``.  Hard byte-cap/modelsize control must
    not protect those bytes as if they could move SegNet/PoseNet output.
    """

    category = str(name)
    render_bound = category in {
        "official_mfu_weight_payload",
        "official_hfr_weight_payload",
        "official_mfu_input_payload",
    }
    receiver_activation_not_frame_bound = category == "official_tub_output2_payload"
    proof_only = category in {
        "official_tub_input_payload",
        "official_tub_output2_payload",
        "official_tub_weight_payload",
    }
    if render_bound:
        action = "protect_quantize_or_waterfill_by_scorer_gradient"
        admission_class = "score_causal_receiver_frame_decode_atom"
    elif proof_only:
        action = (
            "elide_unless_receiver_frame_decode_bound_or_scored_delta_positive"
            if receiver_activation_not_frame_bound
            else "zero_or_elide_until_receiver_frame_decode_bound"
        )
        admission_class = (
            "receiver_activation_not_frame_decode_bound_rate_liability"
            if receiver_activation_not_frame_bound
            else "proof_only_rate_liability"
        )
    else:
        action = "inspect_before_protecting_under_byte_cap"
        admission_class = "unknown_or_graph_topology_atom"
    return {
        "receiver_frame_decode_bound": bool(render_bound),
        "train_time_loss_coupled": bool(render_bound),
        "score_causal_without_source_forward_tub": bool(render_bound),
        "receiver_activation_payload_bound": bool(receiver_activation_not_frame_bound),
        "receiver_activation_payload_score_causal": False,
        "proof_only_receiver_payload": bool(proof_only),
        "byte_cap_action": action,
        "waterfill_admission_class": admission_class,
    }


def _fc_dim_from_candidate(candidate: Mapping[str, Any]) -> int:
    return _fc_dim_resolution_from_candidate(candidate)[0]


def _fc_dim_resolution_from_candidate(candidate: Mapping[str, Any]) -> tuple[int, str]:
    solution = candidate.get("official_modelsize_solution")
    if isinstance(solution, Mapping) and solution.get("fc_dim") is not None:
        return int(solution["fc_dim"]), "official_modelsize_solution"
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
            ), "official_modelsize_formula"
        missing = [
            key
            for key, value in (
                ("full_data_length", full_data_length),
                ("final_size", final_size),
                ("enc_strds/official_enc_strds", enc_strds),
                ("dec_strds/official_dec_strds", dec_strds),
            )
            if value is None
        ]
        raise SnervCarrierError(
            "modelsize_mparams requires official_modelsize_solution or "
            "official formula inputs; missing " + ", ".join(missing)
        )
    if candidate.get("fc_dim") is not None:
        return int(candidate["fc_dim"]), "explicit_fc_dim"
    if candidate.get("snerv_fc_dim") is not None:
        return int(candidate["snerv_fc_dim"]), "explicit_snerv_fc_dim"
    return 9, "fallback_default_missing_official_modelsize_inputs"


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


def _sha256_frame_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return _sha256_bytes(arr.tobytes())


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
