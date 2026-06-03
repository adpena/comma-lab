# SPDX-License-Identifier: MIT
"""Bridge SNeRV receiver exports into NeRV trained-ladder row payloads."""

from __future__ import annotations

from collections.abc import Mapping
from math import sqrt
from pathlib import Path
from typing import Any

from tac.analysis.nerv_trained_ladder_row_emitter import (
    build_nerv_trained_ladder_row_payload,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
    SNERV_SPECTRA_PRESERVING_ADAPTER,
)

SNERV_ADVISORY_TRAINED_LADDER_BRIDGE_SCHEMA = (
    "snerv_advisory_trained_ladder_bridge.v1"
)
SNERV_ADVISORY_TRAINER_METADATA_SCHEMA = (
    "snerv_advisory_receiver_export_trainer_metadata.v1"
)
SNERV_ADVISORY_SCORER_EVAL_SCHEMA = "snerv_advisory_component_eval.v1"
SNERV_PACKET_RECEIVER_PROOF_SCHEMA = "snerv_advisory_packet_receiver_replay.v1"
SNERV_OFFICIAL_MFU_HFR_TUB_EXPORT_BLOCKERS = (
    "snerv_official_neural_decoder_payload_grammar_missing",
    "snerv_official_mfu_hfr_tub_weight_mapping_missing",
    "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
    "snerv_official_receiver_runtime_decode_missing",
)
PROTECTED_OFFICIAL_CONTROL_FIELDS = frozenset(
    (
        "source_faithful_stack",
        "official_parity_status",
        "official_mfu_hfr_tub_numeric_primitives_requested",
        "official_mfu_hfr_tub_primitives_present",
        "official_mfu_hfr_tub_export_bound",
        "official_mfu_hfr_tub_export_blockers",
    )
)


def build_snerv_trained_ladder_row_from_advisory(
    *,
    advisory_result: Any,
    archive_path: str | Path,
    archive_path_kind: str,
    receiver_proof: Mapping[str, Any] | None = None,
    target_bits_per_coeff: float | None = None,
    qat_bits: int | None = None,
    official_controls: Mapping[str, Any] | None = None,
    row_id: str | None = None,
    full_pair_count: int = 600,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the strict trained-row payload for a SNeRV receiver export."""

    kind = str(archive_path_kind).strip()
    if kind not in {"receiver_snar_packet", "contest_archive_zip"}:
        raise ValueError(
            "archive_path_kind must be receiver_snar_packet or contest_archive_zip"
        )
    controls = _actual_controls(advisory_result)
    if official_controls is not None:
        controls = _merge_official_controls_fail_closed(
            controls,
            official_controls,
        )

    proof = dict(receiver_proof or {})
    if not proof:
        proof = _packet_receiver_proof(advisory_result)

    payload = build_nerv_trained_ladder_row_payload(
        family="snerv",
        archive_path=archive_path,
        trainer_metadata=_trainer_metadata(
            advisory_result,
            archive_path_kind=kind,
            target_bits_per_coeff=target_bits_per_coeff,
            qat_bits=qat_bits,
            official_controls=controls,
        ),
        receiver_proof=proof,
        scorer_eval=_scorer_eval(advisory_result),
        official_controls=controls,
        row_id=row_id,
        full_pair_count=int(full_pair_count),
        repo_root=repo_root,
    )
    payload["bridge_schema"] = SNERV_ADVISORY_TRAINED_LADDER_BRIDGE_SCHEMA
    payload["archive_path_kind"] = kind
    payload["bridge_notes"] = (
        "SNeRV advisory/export row: byte custody and local receiver replay are "
        "real, but score/exact/frontier authority remains false until full600 "
        "paired contest CPU/CUDA and source-faithful SNeRV controls close."
    )
    return payload


def _actual_controls(advisory_result: Any) -> dict[str, Any]:
    adapter = str(
        _attr(advisory_result, "snerv_model_size_adapter")
        or "snerv_inverse_steg_principled_fork"
    )
    hfr_gain = float(_attr(advisory_result, "snerv_hfr_gain") or 0.0)
    temporal_context = int(_attr(advisory_result, "snerv_temporal_context") or 0)
    temporal_mode = str(_attr(advisory_result, "snerv_temporal_mode") or "delta")
    official_primitives_requested = (
        adapter == SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER
    )
    receiver_safe_mfu_adapter_present = (
        adapter == SNERV_SPECTRA_PRESERVING_ADAPTER
        or "mfu_hfr_temporal" in adapter
    )
    mfu_enabled = bool(
        receiver_safe_mfu_adapter_present or official_primitives_requested
    )
    if official_primitives_requested:
        official_parity_status = (
            "official_mfu_hfr_tub_numeric_primitives_present__receiver_export_"
            "and_source_forward_replay_required"
        )
    elif receiver_safe_mfu_adapter_present:
        official_parity_status = (
            "receiver_safe_mfu_hfr_temporal_adapter_present__official_oss_"
            "parity_still_required"
        )
    else:
        official_parity_status = "blocked_official_mfu_hfr_not_implemented"
    controls: dict[str, Any] = {
        "source_faithful_stack": False,
        "official_parity_status": official_parity_status,
        "adapter": adapter,
        "mfu_enabled": bool(mfu_enabled),
        "receiver_safe_mfu_adapter_present": bool(receiver_safe_mfu_adapter_present),
        "official_mfu_hfr_tub_numeric_primitives_requested": bool(
            official_primitives_requested
        ),
        "official_mfu_hfr_tub_primitives_present": bool(
            official_primitives_requested
        ),
        "official_mfu_hfr_tub_export_bound": False,
        "official_mfu_hfr_tub_export_blockers": (
            list(SNERV_OFFICIAL_MFU_HFR_TUB_EXPORT_BLOCKERS)
            if official_primitives_requested
            else []
        ),
        "hfr_enabled": bool(hfr_gain > 0.0),
        "snerv_t_enabled": bool(temporal_context > 0),
        "snerv_temporal_mode": temporal_mode,
    }
    wavelet = _attr(advisory_result, "wavelet")
    levels = _attr(advisory_result, "levels")
    fc_dim = _int_attr(advisory_result, "snerv_fc_dim")
    emb_size = _int_attr(advisory_result, "snerv_emb_size")
    patch_radius = _int_attr(advisory_result, "snerv_patch_radius")
    feature_count = _int_attr(advisory_result, "decoder_feature_count")
    mfu_scales = _attr(advisory_result, "snerv_mfu_scales") or ()
    official_solution = _attr(advisory_result, "official_modelsize_solution")
    if wavelet is not None:
        controls["wavelet"] = wavelet
    if levels is not None:
        controls["levels"] = int(levels)
    if fc_dim is not None:
        controls["fc_dim"] = fc_dim
    if emb_size is not None:
        controls["emb_size"] = emb_size
    if patch_radius is not None:
        controls["patch_radius"] = patch_radius
    if feature_count is not None:
        controls["decoder_feature_count"] = feature_count
    controls["mfu_scales"] = [int(v) for v in mfu_scales]
    controls["hfr_gain"] = hfr_gain
    controls["temporal_context"] = temporal_context
    if isinstance(official_solution, Mapping):
        modelsize_mparams = _float_mapping_value(
            official_solution,
            "modelsize_mparams",
        )
        solved_fc_dim = _int_mapping_value(official_solution, "fc_dim")
        if modelsize_mparams is not None:
            controls["--modelsize"] = modelsize_mparams
        controls["official_modelsize_solution"] = dict(official_solution)
        controls["source_bound_modelsize_control"] = {
            "schema": "snerv_source_bound_modelsize_control.v1",
            "--modelsize": modelsize_mparams,
            "fc_dim": solved_fc_dim,
            "official_modelsize_solution": dict(official_solution),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        controls["modelsize_control_source"] = "official_snerv_modelsize"
    if fc_dim is not None or emb_size is not None or feature_count is not None:
        controls["local_modelsize_analogue"] = {
            "schema": "snerv_local_modelsize_analogue.v1",
            "fc_dim": fc_dim,
            "emb_size": emb_size,
            "patch_radius": patch_radius,
            "mfu_scales": [int(v) for v in mfu_scales],
            "hfr_gain": hfr_gain,
            "temporal_context": temporal_context,
            "adapter": adapter,
            "decoder_feature_count": feature_count,
            "official_modelsize_authority": bool(isinstance(official_solution, Mapping)),
            "official_modelsize_solution": (
                dict(official_solution)
                if isinstance(official_solution, Mapping)
                else None
            ),
            "authority_note": (
                "local fc_dim/emb_size/MFU/HFR controls alter receiver decoder "
                "features and bytes, but are not official SNeRV --modelsize "
                "stack authority until upstream OSS parity closes"
            ),
        }
    return controls


def _merge_official_controls_fail_closed(
    actual_controls: Mapping[str, Any],
    supplied_controls: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge enrichment fields without allowing caller-supplied authority flags."""

    merged = dict(actual_controls)
    protected_overrides: dict[str, Any] = {}
    for key, value in dict(supplied_controls).items():
        if key in PROTECTED_OFFICIAL_CONTROL_FIELDS:
            protected_overrides[key] = value
            continue
        merged[key] = value
    if protected_overrides:
        merged["official_control_override_guard"] = {
            "schema": "snerv_official_control_override_guard.v1",
            "protected_fields": sorted(PROTECTED_OFFICIAL_CONTROL_FIELDS),
            "ignored_overrides": protected_overrides,
            "source_faithful_stack": bool(merged.get("source_faithful_stack")) is True,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    return merged


def _trainer_metadata(
    advisory_result: Any,
    *,
    archive_path_kind: str,
    target_bits_per_coeff: float | None,
    qat_bits: int | None,
    official_controls: Mapping[str, Any],
) -> dict[str, Any]:
    lf_payload_codec = _attr(advisory_result, "lf_payload_codec")
    metadata: dict[str, Any] = {
        "schema": SNERV_ADVISORY_TRAINER_METADATA_SCHEMA,
        "n_pairs": _int_attr(advisory_result, "n_pairs"),
        "official_controls": dict(official_controls),
        "fc_dim": _int_attr(advisory_result, "snerv_fc_dim"),
        "snerv_emb_size": _int_attr(advisory_result, "snerv_emb_size"),
        "snerv_patch_radius": _int_attr(advisory_result, "snerv_patch_radius"),
        "snerv_model_size_adapter": _attr(
            advisory_result,
            "snerv_model_size_adapter",
        ),
        "snerv_mfu_scales": list(_attr(advisory_result, "snerv_mfu_scales") or ()),
        "snerv_hfr_gain": _attr(advisory_result, "snerv_hfr_gain"),
        "snerv_temporal_context": _attr(
            advisory_result,
            "snerv_temporal_context",
        ),
        "snerv_temporal_mode": _attr(advisory_result, "snerv_temporal_mode"),
        "decoder_feature_count": _int_attr(advisory_result, "decoder_feature_count"),
        "receiver_codec_mode": archive_path_kind,
        "lf_payload_codec": lf_payload_codec,
        "decoder_precision_mode": _attr(advisory_result, "decoder_payload_codec"),
        "step_map_codec": _attr(advisory_result, "linf_steps_payload_codec"),
        "target_bits_per_coeff": target_bits_per_coeff,
        "hf_decoder_fit_mode": _attr(advisory_result, "hf_decoder_fit_mode"),
        "hf_decoder_saliency_component": _attr(
            advisory_result,
            "hf_decoder_saliency_component",
        ),
        "source": "snerv_advisory_receiver_export",
    }
    if qat_bits is not None:
        metadata["qat_bits"] = int(qat_bits)
    official_solution = _attr(advisory_result, "official_modelsize_solution")
    if isinstance(official_solution, Mapping):
        metadata["official_modelsize_solution"] = dict(official_solution)
        modelsize_mparams = _float_mapping_value(
            official_solution,
            "modelsize_mparams",
        )
        if modelsize_mparams is not None:
            metadata["modelsize_mparams"] = modelsize_mparams
    if archive_path_kind == "receiver_snar_packet":
        packet_bytes = _int_attr(
            advisory_result,
            "receiver_archive_packet_bytes",
            fallback_attr="archive_bytes_total",
        )
        packet_sha = _attr(advisory_result, "receiver_archive_sha256")
        if packet_bytes is not None:
            metadata["archive_bytes"] = packet_bytes
        if packet_sha is not None:
            metadata["archive_sha256"] = packet_sha
    return {key: value for key, value in metadata.items() if value is not None}


def _packet_receiver_proof(advisory_result: Any) -> dict[str, Any]:
    replay = bool(_attr(advisory_result, "receiver_archive_replay_verified") is True)
    proof: dict[str, Any] = {
        "schema": SNERV_PACKET_RECEIVER_PROOF_SCHEMA,
        "proof_source": "advisory_result_fields_only",
        "receiver_proof_identity_bound": False,
        "receiver_archive_replay_verified": replay,
        "receiver_contract_satisfied": replay,
        "runtime_consumption_proof_ready": replay,
        "archive_bytes": _int_attr(
            advisory_result,
            "receiver_archive_packet_bytes",
            fallback_attr="archive_bytes_total",
        ),
        "archive_sha256": _attr(advisory_result, "receiver_archive_sha256"),
        "receiver_archive_replay_error": _attr(
            advisory_result,
            "receiver_archive_replay_error",
        ),
    }
    return {key: value for key, value in proof.items() if value is not None}


def _scorer_eval(advisory_result: Any) -> dict[str, Any]:
    d_seg = _float_attr(advisory_result, "d_seg_mean_linf")
    d_pose = _float_attr(advisory_result, "d_pose_mean_linf")
    eval_payload: dict[str, Any] = {
        "schema": SNERV_ADVISORY_SCORER_EVAL_SCHEMA,
        "axis_tag": _attr(advisory_result, "axis_tag"),
        "d_seg_mean_linf": d_seg,
        "d_pose_mean_linf": d_pose,
        "score_linf": _float_attr(advisory_result, "score_linf"),
    }
    if d_seg is not None and d_pose is not None:
        eval_payload["nonrate_score"] = 100.0 * d_seg + sqrt(10.0 * max(d_pose, 0.0))
    return {key: value for key, value in eval_payload.items() if value is not None}


def _attr(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _int_attr(value: Any, name: str, *, fallback_attr: str | None = None) -> int | None:
    raw = _attr(value, name)
    if raw is None and fallback_attr is not None:
        raw = _attr(value, fallback_attr)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _float_attr(value: Any, name: str) -> float | None:
    raw = _attr(value, name)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _float_mapping_value(value: Mapping[str, Any], name: str) -> float | None:
    try:
        return float(value.get(name))
    except (TypeError, ValueError):
        return None


def _int_mapping_value(value: Mapping[str, Any], name: str) -> int | None:
    try:
        return int(value.get(name))
    except (TypeError, ValueError):
        return None


__all__ = [
    "SNERV_ADVISORY_TRAINED_LADDER_BRIDGE_SCHEMA",
    "build_snerv_trained_ladder_row_from_advisory",
]
