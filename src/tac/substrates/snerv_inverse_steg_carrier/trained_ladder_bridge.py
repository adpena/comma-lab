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
FALSE_AUTHORITY = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
SNERV_OFFICIAL_MFU_HFR_TUB_EXPORT_BLOCKERS = (
    "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload",
    "snerv_official_mfu_hfr_tub_weight_mapping_missing",
    "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
)
SNERV_OFFICIAL_MFU_HFR_TUB_POST_EXPORT_BLOCKERS = (
    "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
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
    official_export_bound = _official_mfu_hfr_tub_export_bound(advisory_result)
    official_tensor_map_custody = _official_receiver_tensor_map_custody(
        advisory_result
    )
    official_export_blockers = _official_mfu_hfr_tub_export_blockers(
        advisory_result,
        export_bound=official_export_bound,
    )
    receiver_safe_mfu_adapter_present = (
        adapter == SNERV_SPECTRA_PRESERVING_ADAPTER
        or "mfu_hfr_temporal" in adapter
    )
    mfu_enabled = bool(
        receiver_safe_mfu_adapter_present or official_primitives_requested
    )
    if official_primitives_requested:
        if official_export_bound:
            official_parity_status = (
                "official_mfu_hfr_tub_receiver_payload_bound__source_forward_"
                "replay_required"
            )
        else:
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
        "official_mfu_hfr_tub_export_bound": bool(official_export_bound),
        "official_mfu_hfr_tub_export_blockers": (
            official_export_blockers if official_primitives_requested else []
        ),
        "hfr_enabled": bool(hfr_gain > 0.0),
        "snerv_t_enabled": bool(temporal_context > 0),
        "snerv_temporal_mode": temporal_mode,
    }
    if official_primitives_requested:
        controls["official_receiver_tensor_map_verified"] = bool(
            official_tensor_map_custody["receiver_tensor_map_verified"]
        )
        controls["official_receiver_tensor_map_custody"] = (
            official_tensor_map_custody
        )
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


def _official_mfu_hfr_tub_export_bound(advisory_result: Any) -> bool:
    """Return whether advisory evidence proves receiver-bound official payload."""

    binding = _attr(advisory_result, "official_primitive_binding")
    if isinstance(binding, Mapping):
        return bool(
            binding.get("official_export_bound") is True
            and binding.get("export_bound_to_receiver_packet") is True
            and binding.get("official_receiver_payload_contract_emitted") is True
            and binding.get("receiver_runtime_decode_authority") is True
        )
    return False


def _official_mfu_hfr_tub_export_blockers(
    advisory_result: Any,
    *,
    export_bound: bool,
) -> list[str]:
    """Project official-payload blockers from actual export evidence, fail closed."""

    raw = _attr(advisory_result, "blockers")
    observed = [str(value) for value in raw] if isinstance(raw, (list, tuple)) else []
    official_observed = [
        value
        for value in observed
        if value.startswith("snerv_official_mfu_hfr_tub_")
    ]
    tensor_map_verified = _official_receiver_tensor_map_verified(advisory_result)
    if export_bound:
        blockers = [
            value
            for value in official_observed
            if value
            != "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
        ]
        if tensor_map_verified:
            blockers = [
                value
                for value in blockers
                if value != "snerv_official_mfu_hfr_tub_weight_mapping_missing"
            ]
        for fallback in SNERV_OFFICIAL_MFU_HFR_TUB_POST_EXPORT_BLOCKERS:
            if fallback not in blockers:
                blockers.append(fallback)
        return blockers
    return list(SNERV_OFFICIAL_MFU_HFR_TUB_EXPORT_BLOCKERS)


def _official_receiver_tensor_map_verified(advisory_result: Any) -> bool:
    return bool(
        _official_receiver_tensor_map_custody(advisory_result)[
            "receiver_tensor_map_verified"
        ]
    )


def _official_receiver_tensor_map_custody(advisory_result: Any) -> dict[str, Any]:
    """Return row/hash-backed official receiver tensor-map custody evidence."""

    binding = _attr(advisory_result, "official_primitive_binding")
    if not isinstance(binding, Mapping):
        return _tensor_map_custody_row(
            blockers=["snerv_official_receiver_tensor_map_binding_missing"]
        )
    tensor_map = binding.get("official_receiver_tensor_map")
    if not isinstance(tensor_map, Mapping):
        return _tensor_map_custody_row(
            blockers=["snerv_official_receiver_tensor_map_missing"]
        )

    blockers: list[str] = []
    raw_blockers = tensor_map.get("blockers")
    if isinstance(raw_blockers, (list, tuple)):
        blockers.extend(str(value) for value in raw_blockers if str(value))
    elif raw_blockers:
        blockers.append(str(raw_blockers))

    rows_raw = tensor_map.get("rows")
    rows = list(rows_raw) if isinstance(rows_raw, (list, tuple)) else []
    manifest_sha = str(tensor_map.get("tensor_manifest_sha256") or "")
    row_count = _int_mapping_value(tensor_map, "row_count") or 0
    total_tensor_bytes = _int_mapping_value(tensor_map, "total_tensor_bytes") or 0
    official_payload_selected = bool(
        tensor_map.get("official_decoder_payload_selected") is True
    )

    if tensor_map.get("receiver_tensor_map_verified") is not True:
        blockers.append("snerv_official_receiver_tensor_map_verified_flag_false")
    if not official_payload_selected:
        blockers.append("snerv_official_receiver_tensor_map_payload_not_official")
    if not rows:
        blockers.append("snerv_official_receiver_tensor_map_rows_missing")
    if row_count <= 0:
        blockers.append("snerv_official_receiver_tensor_map_row_count_missing")
    elif rows and row_count != len(rows):
        blockers.append("snerv_official_receiver_tensor_map_row_count_mismatch")
    if total_tensor_bytes <= 0:
        blockers.append("snerv_official_receiver_tensor_map_bytes_missing")
    if not _is_sha256_hex(manifest_sha):
        blockers.append("snerv_official_receiver_tensor_map_manifest_sha_missing")

    row_names: list[str] = []
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            blockers.append("snerv_official_receiver_tensor_map_malformed_row")
            continue
        name = str(raw_row.get("name") or "")
        if name:
            row_names.append(name)
        row_bytes = _int_mapping_value(raw_row, "bytes") or 0
        if row_bytes <= 0:
            blockers.append("snerv_official_receiver_tensor_map_row_bytes_missing")
        if not _is_sha256_hex(str(raw_row.get("sha256") or "")):
            blockers.append("snerv_official_receiver_tensor_map_row_sha_missing")

    verified = not _unique_texts(blockers)
    return _tensor_map_custody_row(
        receiver_tensor_map_verified=verified,
        official_decoder_payload_selected=official_payload_selected,
        row_count=row_count,
        total_tensor_bytes=total_tensor_bytes,
        tensor_manifest_sha256=manifest_sha if _is_sha256_hex(manifest_sha) else None,
        row_names=row_names,
        category_counts=tensor_map.get("category_counts"),
        category_bytes=tensor_map.get("category_bytes"),
        blockers=blockers,
    )


def _tensor_map_custody_row(
    *,
    receiver_tensor_map_verified: bool = False,
    official_decoder_payload_selected: bool = False,
    row_count: int = 0,
    total_tensor_bytes: int = 0,
    tensor_manifest_sha256: str | None = None,
    row_names: list[str] | None = None,
    category_counts: Any = None,
    category_bytes: Any = None,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "snerv_official_receiver_tensor_map_custody.v1",
        "receiver_tensor_map_verified": bool(receiver_tensor_map_verified),
        "official_decoder_payload_selected": bool(official_decoder_payload_selected),
        "row_count": int(row_count),
        "total_tensor_bytes": int(total_tensor_bytes),
        "tensor_manifest_sha256": tensor_manifest_sha256,
        "row_names": list(row_names or []),
        "category_counts": dict(category_counts)
        if isinstance(category_counts, Mapping)
        else {},
        "category_bytes": dict(category_bytes)
        if isinstance(category_bytes, Mapping)
        else {},
        "blockers": _unique_texts(blockers or []),
        **FALSE_AUTHORITY,
    }


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


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
