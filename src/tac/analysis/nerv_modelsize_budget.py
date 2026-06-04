# SPDX-License-Identifier: MIT
"""NeRV-family model-size budget planner.

The HNeRV/SNeRV upstreams expose ``--modelsize`` as a parameter-budget knob.
For this contest, the useful inverse is stricter: pick architecture capacity
from a charged archive-byte ceiling, then verify with the real archive exporter.

This module is deliberately planner-grade, not score authority. It produces
machine-readable size candidates and OSS flag coverage so the queue runner can
stop launching one arbitrary compact HiNeRV point.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from math import ceil
from typing import Any

from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_BASE_MODEL_SIZE_ADAPTER,
    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
    SNERV_SPECTRA_PRESERVING_ADAPTER,
    SNERV_TEMPORAL_MODES,
    SnervCarrierError,
    normalize_snerv_model_size_adapter,
    official_snerv_modelsize_to_fc_dim,
)

CONTEST_RATE_DENOM_BYTES = 37_545_489
CONTEST_RATE_MULTIPLIER = 25.0
RATE_SCORE_PER_BYTE = CONTEST_RATE_MULTIPLIER / CONTEST_RATE_DENOM_BYTES

SCHEMA = "nerv_modelsize_budget.v1"
HINERV_COMPACT_MID_INJECTION_BLOCK_INDEX = 1
HINERV_COMPACT_FINE_INJECTION_BLOCK_INDEX = 4
OSS_FLAG_AUDIT_SCHEMA = "nerv_oss_flag_audit.v1"
MODELSIZE_RATE_AUTHORITY_SURFACE = (
    "archive_zip_bytes_after_receiver_export_and_inflate_proof"
)
MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS = (
    "nominal_payload_bytes_are_planner_prior_only",
    "nominal_under_ceiling_is_not_promotion_authority",
    "receiver_closed_archive_bytes_required_for_under_ceiling_claim",
    "trained_archive_export_required_for_score_or_rate_claim",
    "archive_bytes_authority_required",
)
MODELSIZE_CONTROL_AUTHORITY_SPLIT_SCHEMA = "nerv_modelsize_control_authority_split.v1"


def modelsize_control_authority_split(
    *,
    family: str,
    control_semantics: str,
    modelsize_mparams_is_official_upstream_flag: bool,
    target_modelsize_mparams_present: bool,
    invalid_control_row: bool = False,
) -> dict[str, Any]:
    """Make size-control semantics machine-checkable and fail-closed."""

    if modelsize_mparams_is_official_upstream_flag:
        mparams_semantics = "official_upstream_parameter_budget_control"
    elif target_modelsize_mparams_present:
        mparams_semantics = "local_nearest_parameter_count_target"
    else:
        mparams_semantics = "absent_or_measured_parameter_count_metadata"
    return {
        "schema": MODELSIZE_CONTROL_AUTHORITY_SPLIT_SCHEMA,
        "family": str(family),
        "control_semantics": str(control_semantics),
        "modelsize_mparams_semantics": mparams_semantics,
        "target_modelsize_mparams_present": bool(target_modelsize_mparams_present),
        "modelsize_mparams_is_official_upstream_flag": bool(
            modelsize_mparams_is_official_upstream_flag
        ),
        "modelsize_mparams_caps_archive_zip_bytes": False,
        "archive_byte_authority_surface": MODELSIZE_RATE_AUTHORITY_SURFACE,
        "archive_byte_cap_semantics": (
            "hard_byte_ceiling_is_filter_only_until_receiver_export_measures_bytes"
        ),
        "same_numeric_target_can_feed_family_specific_controls": True,
        "invalid_control_row": bool(invalid_control_row),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def modelsize_control_precedence_contract(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the modelsize control cascade without granting score authority."""

    family = str(candidate.get("family") or "")
    candidate_id = str(candidate.get("candidate_id") or "")
    capacity_source = str(candidate.get("capacity_source") or "")
    contract = candidate.get("modelsize_control_contract")
    control_semantics = (
        str(contract.get("control_semantics") or "")
        if isinstance(contract, Mapping)
        else ""
    )
    receiver_visible_capacity_active = bool(capacity_source or control_semantics) or any(
        candidate.get(key) is not None
        for key in (
            "latent_dim",
            "embed_dim",
            "decoder_channel",
            "fc_dim",
            "lower_width",
            "enc_dim",
            "modelsize_mparams",
        )
    )
    target_modelsize = candidate.get("target_modelsize_mparams")
    official_source_base_active = (
        family == "hi_nerv"
        and bool(candidate.get("use_hierarchical_feature_grid"))
        and bool(candidate.get("use_convnext_blocks"))
    )
    if family == "snerv":
        official_source_base_active = str(
            candidate.get("snerv_model_size_adapter") or ""
        ).startswith("snerv_")
    child_layers: list[dict[str, Any]] = [
        {
            "layer_id": "parent_source_or_oss_defaults",
            "specificity": 10,
            "active": True,
            "role": "base_rule",
        },
        {
            "layer_id": "official_source_control_base",
            "specificity": 30,
            "active": bool(official_source_base_active),
            "role": "required_guardrail",
        },
        {
            "layer_id": "pact_hard_byte_ceiling",
            "specificity": 70,
            "active": bool(candidate.get("hard_byte_ceiling")),
            "role": "receiver_archive_budget_filter",
        },
        {
            "layer_id": "pact_receiver_visible_modelsize_child_rule",
            "specificity": 80,
            "active": receiver_visible_capacity_active,
            "role": "receiver_visible_capacity_rule",
        },
    ]
    if target_modelsize is not None:
        child_layers.append(
            {
                "layer_id": "pact_target_modelsize_child_rule",
                "specificity": 90,
                "active": True,
                "role": "nearest_receiver_visible_capacity_target",
            }
        )
    if candidate.get("official_modelsize_solution") is not None:
        child_layers.append(
            {
                "layer_id": "official_snerv_modelsize_fc_dim_child_rule",
                "specificity": 90,
                "active": True,
                "role": "receiver_visible_fc_dim_solution",
            }
        )
    active_layers = [row for row in child_layers if row["active"]]
    highest = max(active_layers, key=lambda row: int(row["specificity"]))
    return {
        "schema": "nerv_control_precedence_contract.v1",
        "family": family,
        "candidate_id": candidate_id,
        "cascade_model": "css_like_specificity_with_fail_closed_authority_gates",
        "more_finely_grained_child_rules_take_priority": True,
        "child_rules_override_parent_defaults": True,
        "parent_rules_remain_required_guardrails": True,
        "official_controls_are_base_constraints_not_rate_optimizer_overrides": True,
        "highest_specificity_active_layer": highest["layer_id"],
        "layers_low_to_high_specificity": sorted(
            child_layers,
            key=lambda row: int(row["specificity"]),
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

DEFAULT_HINERV_LATENT_DIMS = (4, 8, 12, 16, 24, 28)
DEFAULT_HINERV_EMBED_DIMS = (8, 12, 16, 24, 32)
DEFAULT_HINERV_DECODER_CHANNELS = (4, 6, 8, 12, 16, 24, 32)
DEFAULT_HINERV_DECODER_CODECS = (
    "portfolio_auto",
    "int8_mixed",
    "int7_mixed",
    "int6_mixed",
    "int4_mixed",
    "int2_mixed",
)
DEFAULT_SNERV_LEVELS = (2, 3, 4, 5)
SNERV_OFFICIAL_MFU_HFR_TUB_LEVELS = (1,)
DEFAULT_SNERV_BITS_PER_COEFF = (1.5, 2.0, 2.5, 3.0, 4.0)
DEFAULT_SNERV_STEP_MAP_BITS_PER_COEFF = (0.5, 1.0, 2.0, 4.0)
DEFAULT_SNERV_DECODER_CODECS = (
    "mixed_magnitude_symmetric",
    "int8_symmetric",
    "int4_symmetric",
    "int2_symmetric",
)
DEFAULT_SNERV_MODEL_SIZE_ADAPTER = SNERV_BASE_MODEL_SIZE_ADAPTER
DEFAULT_SNERV_TEMPORAL_MODE = "delta"
SNERV_OFFICIAL_CLI_DEFAULT_PROFILE_ID = "official_cli_default"
SNERV_CONTEST_RECEIVER_PROFILE_ID = "contest_receiver_profile"
SNERV_PACT_RECEIVER_PROFILE_ID = SNERV_CONTEST_RECEIVER_PROFILE_ID
SNERV_MANUAL_STRIDE_OVERRIDE_PROFILE_ID = "manual_stride_override"
SNERV_MODELSIZE_CONTROL_PROFILES: dict[str, dict[str, Any]] = {
    SNERV_OFFICIAL_CLI_DEFAULT_PROFILE_ID: {
        "profile_id": SNERV_OFFICIAL_CLI_DEFAULT_PROFILE_ID,
        "source": "upstream_train_snerv_parser_defaults",
        "enc_strds": [],
        "dec_strds": [5, 4, 3, 2, 2],
        "modelsize_solve_supported": False,
        "blockers": ["official_cli_default_enc_strds_empty_requires_source_context"],
    },
    SNERV_CONTEST_RECEIVER_PROFILE_ID: {
        "profile_id": SNERV_CONTEST_RECEIVER_PROFILE_ID,
        "source": "pact_receiver_closed_snar1_profile",
        "source_family": "PACT SNeRV receiver adapter",
        "source_notes": (
            "Not an upstream README default; selected for PACT contest receiver "
            "custody with explicit measured-archive authority required."
        ),
        "enc_strds": [5, 4, 2, 2, 2],
        "dec_strds": [5, 4, 2, 2, 2],
        "modelsize_solve_supported": True,
        "blockers": [],
    },
}
DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID = SNERV_CONTEST_RECEIVER_PROFILE_ID
DEFAULT_SNERV_MODELSIZE_ENC_STRDS = tuple(
    int(v)
    for v in SNERV_MODELSIZE_CONTROL_PROFILES[DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID][
        "enc_strds"
    ]
)
DEFAULT_SNERV_MODELSIZE_DEC_STRDS = tuple(
    int(v)
    for v in SNERV_MODELSIZE_CONTROL_PROFILES[DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID][
        "dec_strds"
    ]
)
# Backwards-compatible import aliases.  New code should read the named
# ``modelsize_control_profile_id`` instead of treating these as upstream CLI
# defaults.
DEFAULT_SNERV_OFFICIAL_ENC_STRDS = DEFAULT_SNERV_MODELSIZE_ENC_STRDS
DEFAULT_SNERV_OFFICIAL_DEC_STRDS = DEFAULT_SNERV_MODELSIZE_DEC_STRDS
_SNERV_ADAPTER_TO_ID_TOKEN = {
    DEFAULT_SNERV_MODEL_SIZE_ADAPTER: "base",
    SNERV_SPECTRA_PRESERVING_ADAPTER: "spectra",
    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER: "official",
}
_SNERV_ID_TOKEN_TO_ADAPTER = {
    value: key for key, value in _SNERV_ADAPTER_TO_ID_TOKEN.items()
}
_SNERV_TEMPORAL_MODE_TO_ID_TOKEN = {
    "delta": "d",
    "official_haar_dwt1d_lowpass": "haar1",
}
_SNERV_ID_TOKEN_TO_TEMPORAL_MODE = {
    value: key for key, value in _SNERV_TEMPORAL_MODE_TO_ID_TOKEN.items()
}


class NervModelSizeBudgetError(ValueError):
    """Raised when a model-size budget request is malformed."""


def snerv_modelsize_control_profile(profile_id: str) -> dict[str, Any]:
    """Return a copy of a named SNeRV modelsize control profile."""

    key = str(profile_id or DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID).strip()
    profile = SNERV_MODELSIZE_CONTROL_PROFILES.get(key)
    if profile is None:
        valid = ", ".join(sorted(SNERV_MODELSIZE_CONTROL_PROFILES))
        raise NervModelSizeBudgetError(
            f"unknown SNeRV modelsize control profile {key!r}; expected one of {valid}"
        )
    return {
        **profile,
        "enc_strds": [int(v) for v in profile.get("enc_strds") or []],
        "dec_strds": [int(v) for v in profile.get("dec_strds") or []],
        "blockers": [str(v) for v in profile.get("blockers") or []],
    }


def _snerv_profile_for_strides(
    *,
    profile_id: str,
    enc_strds: tuple[int, ...],
    dec_strds: tuple[int, ...],
) -> dict[str, Any]:
    profile = snerv_modelsize_control_profile(profile_id)
    profile_enc = tuple(int(v) for v in profile["enc_strds"])
    profile_dec = tuple(int(v) for v in profile["dec_strds"])
    enc = tuple(int(v) for v in enc_strds)
    dec = tuple(int(v) for v in dec_strds)
    if enc == profile_enc and dec == profile_dec:
        return profile
    return {
        "profile_id": SNERV_MANUAL_STRIDE_OVERRIDE_PROFILE_ID,
        "source": "operator_or_tool_explicit_stride_override",
        "base_profile_id": str(profile.get("profile_id")),
        "enc_strds": [int(v) for v in enc],
        "dec_strds": [int(v) for v in dec],
        "modelsize_solve_supported": bool(enc and dec),
        "blockers": ([] if enc and dec else ["manual_stride_override_missing_enc_or_dec_strds"]),
    }


def snerv_model_size_adapter_id_token(adapter: str) -> str:
    """Encode a SNeRV adapter string into a lossless candidate-id token."""

    normalized = normalize_snerv_model_size_adapter(str(adapter))
    known = _SNERV_ADAPTER_TO_ID_TOKEN.get(normalized)
    if known is not None:
        return known
    return "hx" + normalized.encode("utf-8").hex()


def snerv_model_size_adapter_from_id_token(token: str) -> str:
    """Decode a SNeRV adapter candidate-id token."""

    normalized = str(token)
    known = _SNERV_ID_TOKEN_TO_ADAPTER.get(normalized)
    if known is not None:
        return known
    if normalized.startswith("hx"):
        try:
            return normalize_snerv_model_size_adapter(
                bytes.fromhex(normalized[2:]).decode("utf-8")
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise NervModelSizeBudgetError(
                f"invalid SNeRV adapter id token: {token!r}"
            ) from exc
    raise NervModelSizeBudgetError(f"unknown SNeRV adapter id token: {token!r}")


def snerv_temporal_mode_id_token(temporal_mode: str) -> str:
    """Encode a receiver-visible SNeRV temporal mode into a candidate-id token."""

    normalized = str(temporal_mode)
    if normalized not in SNERV_TEMPORAL_MODES:
        raise NervModelSizeBudgetError(f"unknown SNeRV temporal mode: {temporal_mode!r}")
    token = _SNERV_TEMPORAL_MODE_TO_ID_TOKEN.get(normalized)
    if token is not None:
        return token
    return "hx" + normalized.encode("utf-8").hex()


def snerv_temporal_mode_from_id_token(token: str) -> str:
    """Decode a SNeRV temporal-mode candidate-id token."""

    normalized = str(token)
    known = _SNERV_ID_TOKEN_TO_TEMPORAL_MODE.get(normalized)
    if known is not None:
        return known
    if normalized.startswith("hx"):
        try:
            temporal_mode = bytes.fromhex(normalized[2:]).decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise NervModelSizeBudgetError(
                f"invalid SNeRV temporal-mode id token: {token!r}"
            ) from exc
        if temporal_mode in SNERV_TEMPORAL_MODES:
            return temporal_mode
    raise NervModelSizeBudgetError(f"unknown SNeRV temporal-mode id token: {token!r}")


def snerv_modelsize_candidate_id_from_controls(
    *,
    num_pairs: int,
    wavelet: str,
    levels: int,
    bits_per_coeff: float,
    step_map_bits_per_coeff: float,
    fc_dim: int,
    emb_size: int,
    patch_radius: int,
    mfu_scales: tuple[int, ...],
    hfr_gain: float,
    temporal_context: int,
    temporal_mode: str = DEFAULT_SNERV_TEMPORAL_MODE,
    snerv_model_size_adapter: str,
    decoder_payload_codec: str,
    hard_byte_ceiling: int,
    official_modelsize_mparams: float | None = None,
) -> str:
    """Build the canonical self-describing SNeRV model-size candidate id."""

    bits_label = _float_id_token(bits_per_coeff)
    step_label = _float_id_token(step_map_bits_per_coeff)
    hfr_label = _float_id_token(hfr_gain)
    official_label = (
        ""
        if official_modelsize_mparams is None
        else f"_oms{_float_id_token(float(official_modelsize_mparams))}"
    )
    wavelet_label = str(wavelet).replace(".", "p").replace("_", "")
    feature_label = f"fc{int(fc_dim)}e{int(emb_size)}"
    mfu_label = "-".join(str(int(value)) for value in mfu_scales)
    adapter_label = snerv_model_size_adapter_id_token(snerv_model_size_adapter)
    temporal_mode_label = (
        ""
        if str(temporal_mode) == DEFAULT_SNERV_TEMPORAL_MODE
        else f"_tm{snerv_temporal_mode_id_token(str(temporal_mode))}"
    )
    return (
        f"snerv_np{int(num_pairs)}_{wavelet_label}_lv{int(levels)}_"
        f"lfb{bits_label}_stepb{step_label}_{feature_label}_"
        f"p{int(patch_radius)}_mfu{mfu_label}_"
        f"hfr{hfr_label}_t{int(temporal_context)}{temporal_mode_label}_"
        f"ad{adapter_label}{official_label}_"
        f"{decoder_payload_codec}_ceil{int(hard_byte_ceiling)}"
    )


def _float_id_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def hinerv_modelsize_candidate_id_from_controls(
    *,
    num_pairs: int,
    latent_dim: int,
    embed_dim: int,
    decoder_channel: int,
    decoder_codec: str,
    hard_byte_ceiling: int,
    use_hierarchical_feature_grid: bool,
    use_convnext_blocks: bool,
    local_grid_levels: int,
    local_grid_channels: int,
    convnext_mlp_ratio: int,
    convnext_kernel_size: int,
    mid_injection_block_index: int,
    fine_injection_block_index: int,
) -> str:
    """Build the lossless self-describing HiNeRV model-size candidate id.

    Older ids encoded only ``_hfg``/``_cnx`` booleans. That collided for
    materially different receiver graphs with different grid, ConvNeXt, or
    latent-injection controls. New ids include every architecture knob that the
    executor must reconstruct before training/export.
    """

    control_label = ""
    if bool(use_hierarchical_feature_grid):
        control_label += "_hfg"
    if bool(use_convnext_blocks):
        control_label += "_cnx"
    if bool(use_hierarchical_feature_grid) or bool(use_convnext_blocks):
        control_label += (
            f"_lg{int(local_grid_levels)}c{int(local_grid_channels)}"
            f"_cx{int(convnext_mlp_ratio)}k{int(convnext_kernel_size)}"
        )
    return (
        f"hinerv_np{int(num_pairs)}_ld{int(latent_dim)}_ed{int(embed_dim)}_"
        f"dc{int(decoder_channel)}_mi{int(mid_injection_block_index)}"
        f"fi{int(fine_injection_block_index)}{control_label}_"
        f"{decoder_codec}_ceil{int(hard_byte_ceiling)}"
    )


@dataclass(frozen=True)
class HinervModelSizeCandidate:
    """One local HiNeRV capacity point, priced before real training/export."""

    schema: str
    family: str
    candidate_id: str
    num_pairs: int
    hard_byte_ceiling: int
    latent_dim: int
    embed_dim: int
    decoder_channel: int
    decoder_channels: tuple[int, ...]
    mid_injection_block_index: int
    fine_injection_block_index: int
    decoder_codec: str
    use_hierarchical_feature_grid: bool
    use_convnext_blocks: bool
    local_grid_levels: int
    local_grid_channels: int
    convnext_mlp_ratio: int
    convnext_kernel_size: int
    latent_dim_coarse: int
    latent_dim_mid: int
    latent_dim_fine: int
    total_trainable_params: int
    decoder_trainable_params: int
    latent_trainable_params: int
    latent_int16_payload_bytes: int
    raw_fp32_total_param_bytes: int
    nominal_decoder_payload_bytes: int
    nominal_total_payload_bytes: int
    nominal_rate_score: float
    nominal_under_ceiling: bool
    byte_headroom: int
    modelsize_mparams: float
    capacity_source: str
    target_modelsize_mparams: float | None
    modelsize_error_mparams: float | None
    upstream_modelsize_analogue: str
    requires_archive_byte_oracle: bool
    score_claim: bool
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decoder_channels"] = list(self.decoder_channels)
        payload["modelsize_control_contract"] = {
            "schema": "nerv_modelsize_control_contract.v1",
            "family": "hi_nerv",
            "control_semantics": (
                "local_receiver_visible_grid_search_nearest_target"
                if self.capacity_source == "local_hinerv_target_modelsize"
                else "manual_receiver_visible_architecture_knobs"
            ),
            "shared_target_modelsize_mparams_consumed_as": (
                "nearest_local_param_count_target"
                if self.target_modelsize_mparams is not None
                else None
            ),
            "modelsize_mparams_is_official_upstream_flag": False,
            "modelsize_mparams_caps_archive_zip_bytes": False,
            "authority_split": modelsize_control_authority_split(
                family="hi_nerv",
                control_semantics=(
                    "local_receiver_visible_grid_search_nearest_target"
                    if self.capacity_source == "local_hinerv_target_modelsize"
                    else "manual_receiver_visible_architecture_knobs"
                ),
                modelsize_mparams_is_official_upstream_flag=False,
                target_modelsize_mparams_present=(
                    self.target_modelsize_mparams is not None
                ),
            ),
            **dict.fromkeys(MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS, True),
            "rate_authority_surface": MODELSIZE_RATE_AUTHORITY_SURFACE,
            "mutates_receiver_visible_architecture": True,
            "mutates_trained_parameter_count": True,
            "hard_byte_ceiling_is_archive_budget_filter": True,
            "control_precedence": modelsize_control_precedence_contract(payload),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        return payload


@dataclass(frozen=True)
class SnervModelSizeCandidate:
    """One SNeRV LF/HF receiver-grammar budget point, before real replay."""

    schema: str
    family: str
    candidate_id: str
    num_pairs: int
    hard_byte_ceiling: int
    carrier_hw: tuple[int, int]
    wavelet: str
    levels: int
    bits_per_coeff: float
    step_map_bits_per_coeff: float
    decoder_payload_codec: str
    snerv_model_size_adapter: str
    capacity_source: str
    modelsize_mparams: float | None
    modelsize_control_profile_id: str
    modelsize_control_profile: dict[str, Any]
    official_modelsize_solution: dict[str, Any] | None
    fc_dim: int
    emb_size: int
    patch_radius: int
    mfu_scales: tuple[int, ...]
    hfr_gain: float
    temporal_context: int
    temporal_mode: str
    decoder_feature_count: int
    lf_coeffs_per_plane: int
    lf_plane_count: int
    lf_coeff_count_total: int
    hf_decoder_weight_count: int
    nominal_lf_payload_bytes: int
    nominal_step_map_payload_bytes: int
    nominal_decoder_payload_bytes: int
    nominal_metadata_payload_bytes: int
    nominal_header_overhead_bytes: int
    nominal_total_payload_bytes: int
    nominal_rate_score: float
    nominal_under_ceiling: bool
    byte_headroom: int
    upstream_modelsize_analogue: str
    requires_snAR1_archive_byte_oracle: bool
    score_claim: bool
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["carrier_hw"] = list(self.carrier_hw)
        payload["mfu_scales"] = list(self.mfu_scales)
        official = self.official_modelsize_solution is not None
        payload["modelsize_control_contract"] = {
            "schema": "nerv_modelsize_control_contract.v1",
            "family": "snerv",
            "control_semantics": (
                "official_snerv_modelsize_quadratic_fc_dim_solve"
                if official
                else "manual_receiver_visible_fc_dim_feature_basis"
            ),
            "shared_target_modelsize_mparams_consumed_as": (
                "official_snerv_modelsize_quadratic_fc_dim_solve"
                if official
                else None
            ),
            "modelsize_mparams_is_official_upstream_flag": bool(official),
            "modelsize_control_profile_id": self.modelsize_control_profile_id,
            "modelsize_control_profile": dict(self.modelsize_control_profile),
            "modelsize_mparams_caps_archive_zip_bytes": False,
            "authority_split": modelsize_control_authority_split(
                family="snerv",
                control_semantics=(
                    "official_snerv_modelsize_quadratic_fc_dim_solve"
                    if official
                    else "manual_receiver_visible_fc_dim_feature_basis"
                ),
                modelsize_mparams_is_official_upstream_flag=bool(official),
                target_modelsize_mparams_present=bool(official),
            ),
            **dict.fromkeys(MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS, True),
            "rate_authority_surface": MODELSIZE_RATE_AUTHORITY_SURFACE,
            "mutates_receiver_visible_architecture": True,
            "mutates_receiver_visible_fc_dim": True,
            "mutates_receiver_visible_temporal_basis": (
                self.temporal_context > 0
                and self.temporal_mode != DEFAULT_SNERV_TEMPORAL_MODE
            ),
            "mutates_trained_parameter_count": True,
            "hard_byte_ceiling_is_archive_budget_filter": True,
            "control_precedence": modelsize_control_precedence_contract(payload),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        return payload


def decoder_codec_nominal_bits(codec: str) -> int:
    """Return the nominal per-weight bit width for planner byte priors."""

    normalized = str(codec).strip().lower()
    if normalized in {"fp16", "fp16_enveloped", "fp16_brotli_legacy", "legacy"}:
        return 16
    if normalized.startswith("int8") or normalized in {"auto", "portfolio_auto"}:
        return 8
    if normalized.startswith("int7"):
        return 7
    if normalized.startswith("int6"):
        return 6
    if normalized.startswith("int4"):
        return 4
    if normalized.startswith("int2"):
        return 2
    raise NervModelSizeBudgetError(f"unsupported decoder codec: {codec!r}")


def snerv_decoder_codec_nominal_bits(codec: str) -> int:
    """Return a nominal per-weight bit width for SNeRV HF decoder codecs."""

    normalized = str(codec).strip().lower()
    if normalized in {"float32_lzma", "fp32_lzma", "float32", "legacy"}:
        return 32
    if normalized in {"mixed_magnitude_symmetric", "mixed_symmetric"}:
        return 4
    if normalized.startswith("int8"):
        return 8
    if normalized.startswith("int4"):
        return 4
    if normalized.startswith("int2"):
        return 2
    raise NervModelSizeBudgetError(f"unsupported SNeRV decoder codec: {codec!r}")


def build_hinerv_config_from_size_knobs(
    *,
    num_pairs: int,
    latent_dim: int,
    embed_dim: int,
    decoder_channel: int,
    use_hierarchical_feature_grid: bool = False,
    use_convnext_blocks: bool = False,
    local_grid_levels: int = 2,
    local_grid_channels: int = 4,
    convnext_mlp_ratio: int = 2,
    convnext_kernel_size: int = 7,
    mid_injection_block_index: int = HINERV_COMPACT_MID_INJECTION_BLOCK_INDEX,
    fine_injection_block_index: int = HINERV_COMPACT_FINE_INJECTION_BLOCK_INDEX,
):
    """Build the current local HiNeRV config from compact size knobs."""

    from tac.substrates.hi_nerv.architecture import HinervConfig

    if num_pairs <= 0:
        raise NervModelSizeBudgetError("num_pairs must be positive")
    if latent_dim <= 0 or embed_dim <= 0 or decoder_channel <= 0:
        raise NervModelSizeBudgetError(
            "latent_dim, embed_dim, and decoder_channel must be positive"
        )
    if local_grid_levels <= 0 or local_grid_channels <= 0:
        raise NervModelSizeBudgetError(
            "local_grid_levels and local_grid_channels must be positive"
        )
    if convnext_mlp_ratio <= 0:
        raise NervModelSizeBudgetError("convnext_mlp_ratio must be positive")
    if convnext_kernel_size <= 0 or convnext_kernel_size % 2 == 0:
        raise NervModelSizeBudgetError(
            "convnext_kernel_size must be positive and odd"
        )
    return HinervConfig(
        latent_dim_coarse=max(1, int(latent_dim) // 2),
        latent_dim_mid=max(1, int(latent_dim)),
        latent_dim_fine=max(1, int(latent_dim) * 2),
        embed_dim=max(1, int(embed_dim)),
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=tuple([max(1, int(decoder_channel))] * 7),
        mid_injection_block_index=int(mid_injection_block_index),
        fine_injection_block_index=int(fine_injection_block_index),
        num_pairs=int(num_pairs),
        output_height=384,
        output_width=512,
        use_hierarchical_feature_grid=bool(use_hierarchical_feature_grid),
        use_convnext_blocks=bool(use_convnext_blocks),
        local_grid_levels=int(local_grid_levels),
        local_grid_channels=int(local_grid_channels),
        convnext_mlp_ratio=int(convnext_mlp_ratio),
        convnext_kernel_size=int(convnext_kernel_size),
    )


def _count_hinerv_params(cfg: Any) -> tuple[int, int, int]:
    """Closed-form parameter count for ``tac.substrates.hi_nerv.architecture``."""

    latent = int(cfg.num_pairs) * (
        int(cfg.latent_dim_coarse)
        + int(cfg.latent_dim_mid)
        + int(cfg.latent_dim_fine)
    )
    embed_out = int(cfg.embed_dim) * int(cfg.initial_grid_h) * int(cfg.initial_grid_w)
    decoder = int(cfg.latent_dim_coarse) * embed_out + embed_out
    channels = [int(cfg.embed_dim), *[int(v) for v in cfg.decoder_channels]]
    for i in range(int(cfg.num_upsample_blocks)):
        in_ch = channels[i]
        out_ch = channels[i + 1]
        # _UpBlock Conv2d(in_ch, out_ch * 4, kernel=3, bias=True)
        decoder += (out_ch * 4) * in_ch * 3 * 3 + (out_ch * 4)
        if bool(getattr(cfg, "use_hierarchical_feature_grid", False)):
            level_channels: list[int] = []
            for level in range(int(cfg.local_grid_levels)):
                time_bins = max(2, ceil(int(cfg.num_pairs) / float(2**level)))
                grid_ch = max(
                    1,
                    (int(cfg.local_grid_channels) * (2**level))
                    // max(1, i + 1),
                )
                level_channels.append(int(grid_ch))
                decoder += time_bins * 2 * 2 * grid_ch
            decoder += int(out_ch) * sum(level_channels) + int(out_ch)
        if bool(getattr(cfg, "use_convnext_blocks", False)):
            kernel = int(cfg.convnext_kernel_size)
            hidden = max(int(out_ch), int(out_ch) * max(1, int(cfg.convnext_mlp_ratio)))
            # depthwise conv + LayerNorm2d + two pointwise convs + gamma.
            decoder += int(out_ch) * 1 * kernel * kernel + int(out_ch)
            decoder += 2 * int(out_ch)
            decoder += hidden * int(out_ch) + hidden
            decoder += int(out_ch) * hidden + int(out_ch)
            decoder += int(out_ch)
    mid_ch = channels[int(cfg.mid_injection_block_index) + 1]
    fine_ch = channels[int(cfg.fine_injection_block_index) + 1]
    decoder += int(cfg.latent_dim_mid) * mid_ch + mid_ch
    decoder += int(cfg.latent_dim_fine) * fine_ch + fine_ch
    final_ch = channels[int(cfg.num_upsample_blocks)]
    # Two RGB heads Conv2d(final_ch, 3, kernel=3, bias=True).
    decoder += 2 * (3 * final_ch * 3 * 3 + 3)
    return int(decoder + latent), int(decoder), int(latent)


def analyze_hinerv_modelsize_candidate(
    *,
    hard_byte_ceiling: int,
    num_pairs: int,
    latent_dim: int,
    embed_dim: int,
    decoder_channel: int,
    decoder_codec: str,
    use_hierarchical_feature_grid: bool = False,
    use_convnext_blocks: bool = False,
    local_grid_levels: int = 2,
    local_grid_channels: int = 4,
    convnext_mlp_ratio: int = 2,
    convnext_kernel_size: int = 7,
    mid_injection_block_index: int = HINERV_COMPACT_MID_INJECTION_BLOCK_INDEX,
    fine_injection_block_index: int = HINERV_COMPACT_FINE_INJECTION_BLOCK_INDEX,
) -> HinervModelSizeCandidate:
    """Analyze one local HiNeRV size point against an archive-byte ceiling."""

    if hard_byte_ceiling <= 0:
        raise NervModelSizeBudgetError("hard_byte_ceiling must be positive")
    cfg = build_hinerv_config_from_size_knobs(
        num_pairs=num_pairs,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        decoder_channel=decoder_channel,
        use_hierarchical_feature_grid=use_hierarchical_feature_grid,
        use_convnext_blocks=use_convnext_blocks,
        local_grid_levels=local_grid_levels,
        local_grid_channels=local_grid_channels,
        convnext_mlp_ratio=convnext_mlp_ratio,
        convnext_kernel_size=convnext_kernel_size,
        mid_injection_block_index=mid_injection_block_index,
        fine_injection_block_index=fine_injection_block_index,
    )
    total_params, decoder_params, latent_params = _count_hinerv_params(cfg)
    latent_payload = int(
        2
        * int(num_pairs)
        * (
            int(cfg.latent_dim_coarse)
            + int(cfg.latent_dim_mid)
            + int(cfg.latent_dim_fine)
        )
    )
    bits = decoder_codec_nominal_bits(decoder_codec)
    nominal_decoder_payload = int((decoder_params * bits + 7) // 8)
    nominal_total_payload = int(latent_payload + nominal_decoder_payload)
    headroom = int(hard_byte_ceiling) - nominal_total_payload
    return HinervModelSizeCandidate(
        schema="hinerv_modelsize_candidate.v1",
        family="hi_nerv",
        candidate_id=hinerv_modelsize_candidate_id_from_controls(
            num_pairs=int(num_pairs),
            latent_dim=int(latent_dim),
            embed_dim=int(embed_dim),
            decoder_channel=int(decoder_channel),
            decoder_codec=str(decoder_codec),
            hard_byte_ceiling=int(hard_byte_ceiling),
            use_hierarchical_feature_grid=bool(cfg.use_hierarchical_feature_grid),
            use_convnext_blocks=bool(cfg.use_convnext_blocks),
            local_grid_levels=int(cfg.local_grid_levels),
            local_grid_channels=int(cfg.local_grid_channels),
            convnext_mlp_ratio=int(cfg.convnext_mlp_ratio),
            convnext_kernel_size=int(cfg.convnext_kernel_size),
            mid_injection_block_index=int(cfg.mid_injection_block_index),
            fine_injection_block_index=int(cfg.fine_injection_block_index),
        ),
        num_pairs=int(num_pairs),
        hard_byte_ceiling=int(hard_byte_ceiling),
        latent_dim=int(latent_dim),
        embed_dim=int(embed_dim),
        decoder_channel=int(decoder_channel),
        decoder_channels=tuple(int(v) for v in cfg.decoder_channels),
        mid_injection_block_index=int(cfg.mid_injection_block_index),
        fine_injection_block_index=int(cfg.fine_injection_block_index),
        decoder_codec=str(decoder_codec),
        use_hierarchical_feature_grid=bool(cfg.use_hierarchical_feature_grid),
        use_convnext_blocks=bool(cfg.use_convnext_blocks),
        local_grid_levels=int(cfg.local_grid_levels),
        local_grid_channels=int(cfg.local_grid_channels),
        convnext_mlp_ratio=int(cfg.convnext_mlp_ratio),
        convnext_kernel_size=int(cfg.convnext_kernel_size),
        latent_dim_coarse=int(cfg.latent_dim_coarse),
        latent_dim_mid=int(cfg.latent_dim_mid),
        latent_dim_fine=int(cfg.latent_dim_fine),
        total_trainable_params=total_params,
        decoder_trainable_params=decoder_params,
        latent_trainable_params=latent_params,
        latent_int16_payload_bytes=latent_payload,
        raw_fp32_total_param_bytes=int(total_params * 4),
        nominal_decoder_payload_bytes=nominal_decoder_payload,
        nominal_total_payload_bytes=nominal_total_payload,
        nominal_rate_score=float(nominal_total_payload * RATE_SCORE_PER_BYTE),
        nominal_under_ceiling=bool(headroom >= 0),
        byte_headroom=headroom,
        modelsize_mparams=float(total_params / 1_000_000.0),
        capacity_source="manual_local_knobs",
        target_modelsize_mparams=None,
        modelsize_error_mparams=None,
        upstream_modelsize_analogue=(
            "local repeated-channel HiNeRV analogue of upstream "
            "HNeRV/SNeRV --modelsize parameter budget; real authority is the "
            "measured archive exporter byte oracle"
        ),
        requires_archive_byte_oracle=True,
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
    )


def tag_hinerv_target_modelsize_candidate(
    row: HinervModelSizeCandidate,
    *,
    target_modelsize_mparams: float,
) -> HinervModelSizeCandidate:
    """Attach explicit inverse-target provenance to a real HiNeRV knob row."""

    target = float(target_modelsize_mparams)
    if target <= 0.0:
        raise NervModelSizeBudgetError("target_modelsize_mparams must be positive")
    return replace(
        row,
        candidate_id=f"{row.candidate_id}_tgtmp{_float_id_token(target)}",
        capacity_source="local_hinerv_target_modelsize",
        target_modelsize_mparams=target,
        modelsize_error_mparams=abs(float(row.modelsize_mparams) - target),
        upstream_modelsize_analogue=(
            row.upstream_modelsize_analogue
            + "; selected by local inverse target-modelsize search over "
            "receiver-visible HiNeRV knobs, not an upstream official equation"
        ),
    )


def _select_hinerv_target_modelsize_rows(
    base_rows: list[HinervModelSizeCandidate],
    *,
    target_modelsize_mparams: tuple[float, ...],
) -> list[HinervModelSizeCandidate]:
    """Pick nearest executable local HiNeRV rows for each target mparam budget."""

    tagged: list[HinervModelSizeCandidate] = []
    targets = tuple(float(value) for value in target_modelsize_mparams)
    for target in targets:
        if target <= 0.0:
            raise NervModelSizeBudgetError(
                "target_modelsize_mparams entries must be positive"
            )
    if not targets:
        return tagged
    ceilings = sorted({row.hard_byte_ceiling for row in base_rows})
    codecs = sorted({row.decoder_codec for row in base_rows})
    for ceiling in ceilings:
        for target in targets:
            for codec in codecs:
                rows = [
                    row
                    for row in base_rows
                    if row.hard_byte_ceiling == ceiling and row.decoder_codec == codec
                ]
                if not rows:
                    continue
                chosen = min(
                    rows,
                    key=lambda row: (
                        abs(float(row.modelsize_mparams) - target),
                        not bool(row.nominal_under_ceiling),
                        abs(int(row.byte_headroom)),
                        -int(row.nominal_total_payload_bytes),
                        -int(row.total_trainable_params),
                    ),
                )
                tagged.append(
                    tag_hinerv_target_modelsize_candidate(
                        chosen,
                        target_modelsize_mparams=target,
                    )
                )
    return tagged


def enumerate_hinerv_modelsize_candidates(
    *,
    hard_byte_ceilings: tuple[int, ...],
    num_pairs: int,
    latent_dims: tuple[int, ...] = DEFAULT_HINERV_LATENT_DIMS,
    embed_dims: tuple[int, ...] = DEFAULT_HINERV_EMBED_DIMS,
    decoder_channels: tuple[int, ...] = DEFAULT_HINERV_DECODER_CHANNELS,
    decoder_codecs: tuple[str, ...] = DEFAULT_HINERV_DECODER_CODECS,
    use_hierarchical_feature_grid_options: tuple[bool, ...] = (False, True),
    use_convnext_blocks_options: tuple[bool, ...] = (False, True),
    local_grid_levels: int = 2,
    local_grid_channels: int = 4,
    convnext_mlp_ratio: int = 2,
    convnext_kernel_size: int = 7,
    target_modelsize_mparams: tuple[float, ...] = (),
) -> list[HinervModelSizeCandidate]:
    """Enumerate local HiNeRV capacity points for queue planning."""

    rows: list[HinervModelSizeCandidate] = []
    for ceiling in sorted({int(v) for v in hard_byte_ceilings if int(v) > 0}):
        for latent_dim in latent_dims:
            for embed_dim in embed_dims:
                for decoder_channel in decoder_channels:
                    for decoder_codec in decoder_codecs:
                        for use_grid in tuple(
                            bool(v) for v in use_hierarchical_feature_grid_options
                        ):
                            for use_convnext in tuple(
                                bool(v) for v in use_convnext_blocks_options
                            ):
                                rows.append(
                                    analyze_hinerv_modelsize_candidate(
                                        hard_byte_ceiling=ceiling,
                                        num_pairs=num_pairs,
                                        latent_dim=latent_dim,
                                        embed_dim=embed_dim,
                                        decoder_channel=decoder_channel,
                                        decoder_codec=decoder_codec,
                                        use_hierarchical_feature_grid=use_grid,
                                        use_convnext_blocks=use_convnext,
                                        local_grid_levels=local_grid_levels,
                                        local_grid_channels=local_grid_channels,
                                        convnext_mlp_ratio=convnext_mlp_ratio,
                                        convnext_kernel_size=convnext_kernel_size,
                                    )
                                )
    return [
        *rows,
        *_select_hinerv_target_modelsize_rows(
            rows,
            target_modelsize_mparams=target_modelsize_mparams,
        ),
    ]


def select_hinerv_modelsize_candidates(
    candidates: list[HinervModelSizeCandidate],
    *,
    per_ceiling_limit: int = 8,
) -> list[HinervModelSizeCandidate]:
    """Pick byte-plausible candidates without collapsing the quantization ladder."""

    if per_ceiling_limit <= 0:
        raise NervModelSizeBudgetError("per_ceiling_limit must be positive")
    selected: list[HinervModelSizeCandidate] = []
    ceilings = sorted({row.hard_byte_ceiling for row in candidates})
    for ceiling in ceilings:
        group = [row for row in candidates if row.hard_byte_ceiling == ceiling]
        under = [row for row in group if row.nominal_under_ceiling]
        chosen: dict[str, HinervModelSizeCandidate] = {}
        target_rows = [
            row for row in group if row.capacity_source == "local_hinerv_target_modelsize"
        ]
        target_rows.sort(
            key=lambda row: (
                float(row.target_modelsize_mparams or 0.0),
                float(row.modelsize_error_mparams or 0.0),
                not bool(row.nominal_under_ceiling),
                abs(int(row.byte_headroom)),
                -int(row.nominal_total_payload_bytes),
            )
        )
        for row in target_rows:
            if len(chosen) >= per_ceiling_limit:
                break
            chosen[row.candidate_id] = row
        for codec in DEFAULT_HINERV_DECODER_CODECS:
            codec_rows = [row for row in under if row.decoder_codec == codec]
            if not codec_rows:
                continue
            best_for_codec = max(
                codec_rows,
                key=lambda row: (
                    row.total_trainable_params,
                    row.decoder_trainable_params,
                    row.nominal_total_payload_bytes,
                ),
            )
            chosen[best_for_codec.candidate_id] = best_for_codec
        if under:
            tightest_under = max(under, key=lambda row: row.nominal_total_payload_bytes)
            chosen[tightest_under.candidate_id] = tightest_under
        under.sort(
            key=lambda row: (
                row.total_trainable_params,
                row.decoder_trainable_params,
                row.nominal_total_payload_bytes,
            ),
            reverse=True,
        )
        for row in under:
            if len(chosen) >= per_ceiling_limit:
                break
            chosen.setdefault(row.candidate_id, row)
        selected.extend(list(chosen.values())[:per_ceiling_limit])
        if not under:
            group.sort(key=lambda row: abs(row.byte_headroom))
            selected.extend(group[: min(2, per_ceiling_limit)])
    return selected


def _dedupe_snerv_modelsize_candidates(
    rows: list[SnervModelSizeCandidate],
    *,
    key_fn: Callable[[SnervModelSizeCandidate], tuple[float, ...]],
) -> list[SnervModelSizeCandidate]:
    """Keep one row per receiver candidate id, preserving strongest provenance."""

    best: dict[str, SnervModelSizeCandidate] = {}
    order: list[str] = []
    for row in rows:
        current = best.get(row.candidate_id)
        if current is None:
            order.append(row.candidate_id)
            best[row.candidate_id] = row
        elif key_fn(row) > key_fn(current):
            best[row.candidate_id] = row
    return [best[candidate_id] for candidate_id in order]


def build_hinerv_modelsize_budget_report(
    *,
    hard_byte_ceilings: tuple[int, ...],
    num_pairs: int,
    per_ceiling_limit: int = 8,
    use_hierarchical_feature_grid_options: tuple[bool, ...] = (False, True),
    use_convnext_blocks_options: tuple[bool, ...] = (False, True),
    target_modelsize_mparams: tuple[float, ...] = (),
    official_controls_only: bool = False,
) -> dict[str, Any]:
    """Return a planner-safe local HiNeRV model-size budget report."""

    if official_controls_only:
        use_hierarchical_feature_grid_options = (True,)
        use_convnext_blocks_options = (True,)
    candidates = enumerate_hinerv_modelsize_candidates(
        hard_byte_ceilings=hard_byte_ceilings,
        num_pairs=num_pairs,
        use_hierarchical_feature_grid_options=use_hierarchical_feature_grid_options,
        use_convnext_blocks_options=use_convnext_blocks_options,
        target_modelsize_mparams=target_modelsize_mparams,
    )
    selected = select_hinerv_modelsize_candidates(
        candidates,
        per_ceiling_limit=per_ceiling_limit,
    )
    by_ceiling: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        by_ceiling.setdefault(str(row.hard_byte_ceiling), []).append(row.as_dict())
    return {
        "schema": SCHEMA,
        "family": "hi_nerv",
        "num_pairs": int(num_pairs),
        "hard_byte_ceilings": sorted({int(v) for v in hard_byte_ceilings}),
        "use_hierarchical_feature_grid_options": [
            bool(v) for v in use_hierarchical_feature_grid_options
        ],
        "use_convnext_blocks_options": [bool(v) for v in use_convnext_blocks_options],
        "official_controls_only": bool(official_controls_only),
        "target_modelsize_mparams": [float(v) for v in target_modelsize_mparams],
        "candidate_count": len(candidates),
        "selected_candidate_count": len(selected),
        "selected_candidates": [row.as_dict() for row in selected],
        "selected_candidates_by_ceiling": by_ceiling,
        "budget_math": {
            "contest_rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "nominal_payload_is_not_authority": True,
            "modelsize_control_contract_required_true_fields": list(
                MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS
            ),
            "rate_authority_surface": MODELSIZE_RATE_AUTHORITY_SURFACE,
            "selection_strategy": (
                "for each byte ceiling, retain the best byte-plausible point "
                "from every available decoder codec family before filling with "
                "highest-capacity candidates; when target_modelsize_mparams is "
                "provided, first retain the nearest executable local HiNeRV knob "
                "row per codec; exact archive bytes and scorer replay arbitrate "
                "the quantization ladder"
            ),
            "official_controls_only": bool(official_controls_only),
            "real_authority": (
                "trained byte-closed archive.zip bytes measured after "
                "export_hi_nerv_mlx_archive plus receiver proof"
            ),
            "marginal_rule": (
                "admit extra latent/channel/coder capacity only when the "
                "full-video scorer distortion reduction per added byte exceeds "
                "the contest rate price"
            ),
        },
        "blockers": [
            "hinerv_modelsize_candidates_require_trained_archive_byte_oracle",
            "hinerv_modelsize_candidates_require_full_video_mlx_prefilter",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def analyze_snerv_modelsize_candidate(
    *,
    hard_byte_ceiling: int,
    num_pairs: int,
    carrier_hw: tuple[int, int] = (384, 512),
    wavelet: str = "db2",
    levels: int,
    bits_per_coeff: float,
    step_map_bits_per_coeff: float,
    decoder_payload_codec: str,
    snerv_model_size_adapter: str = DEFAULT_SNERV_MODEL_SIZE_ADAPTER,
    official_modelsize_mparams: float | None = None,
    official_enc_strds: tuple[int, ...] = DEFAULT_SNERV_MODELSIZE_ENC_STRDS,
    official_dec_strds: tuple[int, ...] = DEFAULT_SNERV_MODELSIZE_DEC_STRDS,
    modelsize_control_profile_id: str = DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID,
    fc_dim: int = 9,
    emb_size: int = 0,
    patch_radius: int = 1,
    mfu_scales: tuple[int, ...] = (1, 2, 4),
    hfr_gain: float = 0.0,
    temporal_context: int = 0,
    temporal_mode: str = DEFAULT_SNERV_TEMPORAL_MODE,
) -> SnervModelSizeCandidate:
    """Analyze one SNeRV receiver-grammar point against an archive ceiling."""

    if hard_byte_ceiling <= 0:
        raise NervModelSizeBudgetError("hard_byte_ceiling must be positive")
    if num_pairs <= 0:
        raise NervModelSizeBudgetError("num_pairs must be positive")
    if levels <= 0:
        raise NervModelSizeBudgetError("levels must be positive")
    if bits_per_coeff <= 0 or step_map_bits_per_coeff < 0:
        raise NervModelSizeBudgetError(
            "bits_per_coeff must be positive and step_map_bits_per_coeff non-negative"
        )
    from tac.substrates.snerv_inverse_steg_carrier.carrier import (
        SnervModelSizeConfig,
    )
    from tac.substrates.snerv_inverse_steg_carrier.dwt import lf_coeff_count

    modelsize_control_profile = _snerv_profile_for_strides(
        profile_id=modelsize_control_profile_id,
        enc_strds=tuple(int(v) for v in official_enc_strds),
        dec_strds=tuple(int(v) for v in official_dec_strds),
    )
    official_modelsize_solution = None
    if official_modelsize_mparams is not None:
        official_modelsize_solution_obj = official_snerv_modelsize_to_fc_dim(
            modelsize_mparams=float(official_modelsize_mparams),
            full_data_length=int(num_pairs) * 2,
            final_size=int(carrier_hw[0]) * int(carrier_hw[1]),
            enc_strds=tuple(int(v) for v in official_enc_strds),
            dec_strds=tuple(int(v) for v in official_dec_strds),
            emb_size=int(emb_size),
        )
        official_modelsize_solution = official_modelsize_solution_obj.as_jsonable()
        fc_dim = int(official_modelsize_solution_obj.fc_dim)

    lf_per_plane = lf_coeff_count(carrier_hw, levels=levels, wavelet=wavelet)
    plane_count = int(num_pairs) * 2 * 3
    lf_total = int(lf_per_plane * plane_count)
    model_size = SnervModelSizeConfig(
        fc_dim=int(fc_dim),
        emb_size=int(emb_size),
        patch_radius=int(patch_radius),
        mfu_scales=tuple(int(v) for v in mfu_scales),
        hfr_gain=float(hfr_gain),
        temporal_context=int(temporal_context),
        temporal_mode=str(temporal_mode),
        adapter=str(snerv_model_size_adapter),
    )
    snerv_model_size_adapter = str(model_size.adapter)
    decoder_weight_count = int(levels) * 3 * int(model_size.feature_count)
    decoder_bits = snerv_decoder_codec_nominal_bits(decoder_payload_codec)
    lf_payload = ceil(lf_total * float(bits_per_coeff) / 8.0)
    step_payload = ceil(lf_total * float(step_map_bits_per_coeff) / 8.0)
    decoder_payload = ceil(decoder_weight_count * decoder_bits / 8.0) + int(levels) * 12
    metadata_payload = int(plane_count * 4)
    header_overhead = 1024
    total_payload = (
        lf_payload + step_payload + decoder_payload + metadata_payload + header_overhead
    )
    headroom = int(hard_byte_ceiling) - int(total_payload)
    candidate_id = snerv_modelsize_candidate_id_from_controls(
        num_pairs=int(num_pairs),
        wavelet=str(wavelet),
        levels=int(levels),
        bits_per_coeff=float(bits_per_coeff),
        step_map_bits_per_coeff=float(step_map_bits_per_coeff),
        fc_dim=int(model_size.fc_dim),
        emb_size=int(model_size.emb_size),
        patch_radius=int(model_size.patch_radius),
        mfu_scales=tuple(int(value) for value in model_size.mfu_scales),
        hfr_gain=float(model_size.hfr_gain),
        temporal_context=int(model_size.temporal_context),
        temporal_mode=str(model_size.temporal_mode),
        snerv_model_size_adapter=str(model_size.adapter),
        decoder_payload_codec=str(decoder_payload_codec),
        hard_byte_ceiling=int(hard_byte_ceiling),
        official_modelsize_mparams=official_modelsize_mparams,
    )
    return SnervModelSizeCandidate(
        schema="snerv_modelsize_candidate.v1",
        family="snerv",
        candidate_id=candidate_id,
        num_pairs=int(num_pairs),
        hard_byte_ceiling=int(hard_byte_ceiling),
        carrier_hw=(int(carrier_hw[0]), int(carrier_hw[1])),
        wavelet=str(wavelet),
        levels=int(levels),
        bits_per_coeff=float(bits_per_coeff),
        step_map_bits_per_coeff=float(step_map_bits_per_coeff),
        decoder_payload_codec=str(decoder_payload_codec),
        snerv_model_size_adapter=str(model_size.adapter),
        capacity_source=(
            "manual_fc_dim"
            if official_modelsize_mparams is None
            else "official_snerv_modelsize"
        ),
        modelsize_mparams=(
            None
            if official_modelsize_mparams is None
            else float(official_modelsize_mparams)
        ),
        modelsize_control_profile_id=str(modelsize_control_profile["profile_id"]),
        modelsize_control_profile=modelsize_control_profile,
        official_modelsize_solution=official_modelsize_solution,
        fc_dim=int(model_size.fc_dim),
        emb_size=int(model_size.emb_size),
        patch_radius=int(model_size.patch_radius),
        mfu_scales=tuple(int(v) for v in model_size.mfu_scales),
        hfr_gain=float(model_size.hfr_gain),
        temporal_context=int(model_size.temporal_context),
        temporal_mode=str(model_size.temporal_mode),
        decoder_feature_count=int(model_size.feature_count),
        lf_coeffs_per_plane=lf_per_plane,
        lf_plane_count=plane_count,
        lf_coeff_count_total=lf_total,
        hf_decoder_weight_count=decoder_weight_count,
        nominal_lf_payload_bytes=lf_payload,
        nominal_step_map_payload_bytes=step_payload,
        nominal_decoder_payload_bytes=decoder_payload,
        nominal_metadata_payload_bytes=metadata_payload,
        nominal_header_overhead_bytes=header_overhead,
        nominal_total_payload_bytes=total_payload,
        nominal_rate_score=float(total_payload * RATE_SCORE_PER_BYTE),
        nominal_under_ceiling=bool(headroom >= 0),
        byte_headroom=headroom,
        upstream_modelsize_analogue=(
            "SNeRV local analogue of upstream --modelsize: LF resolution/bit "
            "budget plus receiver-visible fc_dim/emb_size/MFU/HFR/temporal "
            "decoder feature payload define archive capacity; real authority "
            "is measured SNAR1 archive bytes"
        ),
        requires_snAR1_archive_byte_oracle=True,
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
    )


def enumerate_snerv_modelsize_candidates(
    *,
    hard_byte_ceilings: tuple[int, ...],
    num_pairs: int,
    carrier_hw: tuple[int, int] = (384, 512),
    wavelet: str = "haar",
    levels: tuple[int, ...] = DEFAULT_SNERV_LEVELS,
    bits_per_coeffs: tuple[float, ...] = DEFAULT_SNERV_BITS_PER_COEFF,
    step_map_bits_per_coeffs: tuple[float, ...] = DEFAULT_SNERV_STEP_MAP_BITS_PER_COEFF,
    decoder_codecs: tuple[str, ...] = DEFAULT_SNERV_DECODER_CODECS,
    snerv_model_size_adapter: str = DEFAULT_SNERV_MODEL_SIZE_ADAPTER,
    official_modelsize_mparams: tuple[float, ...] = (),
    official_enc_strds: tuple[int, ...] = DEFAULT_SNERV_MODELSIZE_ENC_STRDS,
    official_dec_strds: tuple[int, ...] = DEFAULT_SNERV_MODELSIZE_DEC_STRDS,
    modelsize_control_profile_id: str = DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID,
    fc_dims: tuple[int, ...] = (9,),
    emb_sizes: tuple[int, ...] = (0,),
    patch_radius: int = 1,
    mfu_scales: tuple[int, ...] = (1, 2, 4),
    hfr_gain: float = 0.0,
    temporal_context: int = 0,
    temporal_modes: tuple[str, ...] = (DEFAULT_SNERV_TEMPORAL_MODE,),
    invalid_candidate_rows: list[dict[str, Any]] | None = None,
) -> list[SnervModelSizeCandidate]:
    """Enumerate SNeRV LF/HF receiver-grammar capacity points."""

    rows: list[SnervModelSizeCandidate] = []
    canonical_adapter = normalize_snerv_model_size_adapter(snerv_model_size_adapter)
    effective_levels = tuple(int(v) for v in levels)
    if canonical_adapter == SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER:
        effective_levels = SNERV_OFFICIAL_MFU_HFR_TUB_LEVELS
    for ceiling in sorted({int(v) for v in hard_byte_ceilings if int(v) > 0}):
        for lvl in effective_levels:
            for bits in bits_per_coeffs:
                for step_bits in step_map_bits_per_coeffs:
                    for decoder_codec in decoder_codecs:
                        for temporal_mode in temporal_modes:
                            for fc_dim in fc_dims:
                                for emb_size in emb_sizes:
                                    rows.append(
                                        analyze_snerv_modelsize_candidate(
                                            hard_byte_ceiling=ceiling,
                                            num_pairs=num_pairs,
                                            carrier_hw=carrier_hw,
                                            wavelet=wavelet,
                                            levels=lvl,
                                            bits_per_coeff=bits,
                                            step_map_bits_per_coeff=step_bits,
                                            decoder_payload_codec=decoder_codec,
                                            snerv_model_size_adapter=(
                                                canonical_adapter
                                            ),
                                            fc_dim=int(fc_dim),
                                            emb_size=int(emb_size),
                                            patch_radius=int(patch_radius),
                                            mfu_scales=tuple(int(v) for v in mfu_scales),
                                            hfr_gain=float(hfr_gain),
                                            temporal_context=int(temporal_context),
                                            temporal_mode=str(temporal_mode),
                                        )
                                    )
                        for modelsize_mparams in official_modelsize_mparams:
                            for temporal_mode in temporal_modes:
                                for emb_size in emb_sizes:
                                    try:
                                        rows.append(
                                            analyze_snerv_modelsize_candidate(
                                                hard_byte_ceiling=ceiling,
                                                num_pairs=num_pairs,
                                                carrier_hw=carrier_hw,
                                                wavelet=wavelet,
                                                levels=lvl,
                                                bits_per_coeff=bits,
                                                step_map_bits_per_coeff=step_bits,
                                                decoder_payload_codec=decoder_codec,
                                                snerv_model_size_adapter=(
                                                    canonical_adapter
                                                ),
                                                official_modelsize_mparams=float(
                                                    modelsize_mparams
                                                ),
                                                official_enc_strds=official_enc_strds,
                                                official_dec_strds=official_dec_strds,
                                                modelsize_control_profile_id=(
                                                    modelsize_control_profile_id
                                                ),
                                                emb_size=int(emb_size),
                                                patch_radius=int(patch_radius),
                                                mfu_scales=tuple(int(v) for v in mfu_scales),
                                                hfr_gain=float(hfr_gain),
                                                temporal_context=int(temporal_context),
                                                temporal_mode=str(temporal_mode),
                                            )
                                        )
                                    except (
                                        NervModelSizeBudgetError,
                                        SnervCarrierError,
                                        ValueError,
                                    ) as exc:
                                        if invalid_candidate_rows is not None:
                                            invalid_candidate_rows.append(
                                                _invalid_snerv_official_modelsize_row(
                                                    hard_byte_ceiling=ceiling,
                                                    num_pairs=num_pairs,
                                                    carrier_hw=carrier_hw,
                                                    wavelet=wavelet,
                                                    levels=lvl,
                                                    bits_per_coeff=bits,
                                                    step_map_bits_per_coeff=step_bits,
                                                    decoder_payload_codec=decoder_codec,
                                                    snerv_model_size_adapter=(
                                                        canonical_adapter
                                                    ),
                                                    official_modelsize_mparams=float(
                                                        modelsize_mparams
                                                    ),
                                                    official_enc_strds=official_enc_strds,
                                                    official_dec_strds=official_dec_strds,
                                                    modelsize_control_profile_id=(
                                                        modelsize_control_profile_id
                                                    ),
                                                    emb_size=int(emb_size),
                                                    patch_radius=int(patch_radius),
                                                    mfu_scales=tuple(
                                                        int(v) for v in mfu_scales
                                                    ),
                                                    hfr_gain=float(hfr_gain),
                                                    temporal_context=int(temporal_context),
                                                    temporal_mode=str(temporal_mode),
                                                    error=exc,
                                                )
                                            )
    return rows


def _invalid_snerv_official_modelsize_row(
    *,
    hard_byte_ceiling: int,
    num_pairs: int,
    carrier_hw: tuple[int, int],
    wavelet: str,
    levels: int,
    bits_per_coeff: float,
    step_map_bits_per_coeff: float,
    decoder_payload_codec: str,
    snerv_model_size_adapter: str,
    official_modelsize_mparams: float,
    official_enc_strds: tuple[int, ...],
    official_dec_strds: tuple[int, ...],
    modelsize_control_profile_id: str,
    emb_size: int,
    patch_radius: int,
    mfu_scales: tuple[int, ...],
    hfr_gain: float,
    temporal_context: int,
    temporal_mode: str,
    error: BaseException,
) -> dict[str, Any]:
    """Record an unsatisfiable official SNeRV ``--modelsize`` point."""

    profile = _snerv_profile_for_strides(
        profile_id=modelsize_control_profile_id,
        enc_strds=tuple(int(v) for v in official_enc_strds),
        dec_strds=tuple(int(v) for v in official_dec_strds),
    )
    contract_payload = {
        "family": "snerv",
        "candidate_id": None,
        "capacity_source": "invalid_official_snerv_modelsize",
        "snerv_model_size_adapter": str(snerv_model_size_adapter),
        "modelsize_mparams": float(official_modelsize_mparams),
        "modelsize_control_profile_id": str(profile["profile_id"]),
        "hard_byte_ceiling": int(hard_byte_ceiling),
    }
    return {
        "schema": "snerv_invalid_official_modelsize_candidate.v1",
        "family": "snerv",
        "status": "invalid_official_modelsize_candidate",
        "candidate_id": None,
        "hard_byte_ceiling": int(hard_byte_ceiling),
        "num_pairs": int(num_pairs),
        "carrier_hw": [int(carrier_hw[0]), int(carrier_hw[1])],
        "wavelet": str(wavelet),
        "levels": int(levels),
        "bits_per_coeff": float(bits_per_coeff),
        "step_map_bits_per_coeff": float(step_map_bits_per_coeff),
        "decoder_payload_codec": str(decoder_payload_codec),
        "snerv_model_size_adapter": str(snerv_model_size_adapter),
        "official_modelsize_mparams": float(official_modelsize_mparams),
        "modelsize_control_profile_id": str(profile["profile_id"]),
        "modelsize_control_profile": profile,
        "official_enc_strds": [int(v) for v in official_enc_strds],
        "official_dec_strds": [int(v) for v in official_dec_strds],
        "emb_size": int(emb_size),
        "patch_radius": int(patch_radius),
        "mfu_scales": [int(v) for v in mfu_scales],
        "hfr_gain": float(hfr_gain),
        "temporal_context": int(temporal_context),
        "temporal_mode": str(temporal_mode),
        "error_type": type(error).__name__,
        "blockers": [
            "snerv_official_modelsize_quadratic_unsatisfied",
            f"snerv_official_modelsize_error:{type(error).__name__}",
        ],
        "modelsize_control_contract": {
            "schema": "nerv_modelsize_control_contract.v1",
            "family": "snerv",
            "control_semantics": (
                "invalid_official_snerv_modelsize_quadratic_fc_dim_solve"
            ),
            "shared_target_modelsize_mparams_consumed_as": (
                "official_snerv_modelsize_quadratic_fc_dim_solve"
            ),
            "modelsize_mparams_is_official_upstream_flag": True,
            "modelsize_control_profile_id": str(profile["profile_id"]),
            "modelsize_control_profile": profile,
            "modelsize_mparams_caps_archive_zip_bytes": False,
            "authority_split": modelsize_control_authority_split(
                family="snerv",
                control_semantics=(
                    "invalid_official_snerv_modelsize_quadratic_fc_dim_solve"
                ),
                modelsize_mparams_is_official_upstream_flag=True,
                target_modelsize_mparams_present=True,
                invalid_control_row=True,
            ),
            **dict.fromkeys(MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS, True),
            "rate_authority_surface": MODELSIZE_RATE_AUTHORITY_SURFACE,
            "mutates_receiver_visible_architecture": False,
            "mutates_receiver_visible_fc_dim": False,
            "invalid_control_row": True,
            "control_resolution_status": (
                "failed_before_receiver_visible_fc_dim_candidate"
            ),
            "control_precedence": modelsize_control_precedence_contract(
                contract_payload
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "error": str(error),
        "score_claim": False,
        "score_claim_valid": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "production_hardened_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def select_snerv_modelsize_candidates(
    candidates: list[SnervModelSizeCandidate],
    *,
    per_ceiling_limit: int = 8,
) -> list[SnervModelSizeCandidate]:
    """Pick SNeRV points while preserving level and precision diversity."""

    if per_ceiling_limit <= 0:
        raise NervModelSizeBudgetError("per_ceiling_limit must be positive")
    selected: list[SnervModelSizeCandidate] = []
    ceilings = sorted({row.hard_byte_ceiling for row in candidates})

    def _selection_key(row: SnervModelSizeCandidate) -> tuple[float, ...]:
        return (
            float(row.nominal_total_payload_bytes),
            1.0 if row.official_modelsize_solution is not None else 0.0,
            float(row.modelsize_mparams or 0.0),
            float(row.bits_per_coeff),
            float(row.step_map_bits_per_coeff),
        )

    for ceiling in ceilings:
        group = [row for row in candidates if row.hard_byte_ceiling == ceiling]
        under = [row for row in group if row.nominal_under_ceiling]
        chosen: dict[str, SnervModelSizeCandidate] = {}

        def choose(
            chosen_by_id: dict[str, SnervModelSizeCandidate],
            row: SnervModelSizeCandidate,
        ) -> None:
            current = chosen_by_id.get(row.candidate_id)
            if current is None or _selection_key(row) > _selection_key(current):
                chosen_by_id[row.candidate_id] = row

        for lvl in DEFAULT_SNERV_LEVELS:
            lvl_rows = [row for row in under if row.levels == lvl]
            if lvl_rows:
                best_for_level = max(
                    lvl_rows,
                    key=_selection_key,
                )
            else:
                level_blockers = [row for row in group if row.levels == lvl]
                if not level_blockers:
                    continue
                best_for_level = min(level_blockers, key=lambda row: abs(row.byte_headroom))
            choose(chosen, best_for_level)
        for temporal_mode in sorted({row.temporal_mode for row in group}):
            mode_rows = [row for row in under if row.temporal_mode == temporal_mode]
            if mode_rows:
                best_for_mode = max(mode_rows, key=_selection_key)
            else:
                mode_blockers = [
                    row for row in group if row.temporal_mode == temporal_mode
                ]
                if not mode_blockers:
                    continue
                best_for_mode = min(
                    mode_blockers,
                    key=lambda row: abs(row.byte_headroom),
                )
            choose(chosen, best_for_mode)
        for bits in DEFAULT_SNERV_BITS_PER_COEFF:
            bit_rows = [row for row in under if row.bits_per_coeff == bits]
            if bit_rows:
                best_for_bits = max(bit_rows, key=_selection_key)
            else:
                bit_blockers = [row for row in group if row.bits_per_coeff == bits]
                if not bit_blockers:
                    continue
                best_for_bits = min(bit_blockers, key=lambda row: abs(row.byte_headroom))
            choose(chosen, best_for_bits)
        under.sort(
            key=lambda row: (
                *_selection_key(row),
                row.levels,
            ),
            reverse=True,
        )
        for row in under:
            if len(chosen) >= per_ceiling_limit:
                break
            choose(chosen, row)
        selected.extend(
            _dedupe_snerv_modelsize_candidates(
                list(chosen.values()),
                key_fn=_selection_key,
            )[:per_ceiling_limit]
        )
        if not under:
            group.sort(key=lambda row: abs(row.byte_headroom))
            selected.extend(
                _dedupe_snerv_modelsize_candidates(group, key_fn=_selection_key)[
                    : min(2, per_ceiling_limit)
                ]
            )
    return _dedupe_snerv_modelsize_candidates(selected, key_fn=_selection_key)


def build_snerv_modelsize_budget_report(
    *,
    hard_byte_ceilings: tuple[int, ...],
    num_pairs: int,
    per_ceiling_limit: int = 8,
    carrier_hw: tuple[int, int] = (384, 512),
    wavelet: str = "haar",
    fc_dims: tuple[int, ...] = (9,),
    emb_sizes: tuple[int, ...] = (0,),
    official_modelsize_mparams: tuple[float, ...] = (),
    official_enc_strds: tuple[int, ...] = DEFAULT_SNERV_MODELSIZE_ENC_STRDS,
    official_dec_strds: tuple[int, ...] = DEFAULT_SNERV_MODELSIZE_DEC_STRDS,
    modelsize_control_profile_id: str = DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID,
    snerv_model_size_adapter: str = DEFAULT_SNERV_MODEL_SIZE_ADAPTER,
    patch_radius: int = 1,
    mfu_scales: tuple[int, ...] = (1, 2, 4),
    hfr_gain: float = 0.0,
    temporal_context: int = 0,
    temporal_modes: tuple[str, ...] = (DEFAULT_SNERV_TEMPORAL_MODE,),
) -> dict[str, Any]:
    """Return a planner-safe SNeRV model-size budget report."""

    modelsize_control_profile = _snerv_profile_for_strides(
        profile_id=modelsize_control_profile_id,
        enc_strds=tuple(int(v) for v in official_enc_strds),
        dec_strds=tuple(int(v) for v in official_dec_strds),
    )
    invalid_candidate_rows: list[dict[str, Any]] = []
    candidates = enumerate_snerv_modelsize_candidates(
        hard_byte_ceilings=hard_byte_ceilings,
        num_pairs=num_pairs,
        carrier_hw=carrier_hw,
        wavelet=wavelet,
        fc_dims=fc_dims,
        emb_sizes=emb_sizes,
        official_modelsize_mparams=official_modelsize_mparams,
        official_enc_strds=official_enc_strds,
        official_dec_strds=official_dec_strds,
        modelsize_control_profile_id=str(modelsize_control_profile["profile_id"]),
        snerv_model_size_adapter=snerv_model_size_adapter,
        patch_radius=patch_radius,
        mfu_scales=mfu_scales,
        hfr_gain=hfr_gain,
        temporal_context=temporal_context,
        temporal_modes=temporal_modes,
        invalid_candidate_rows=invalid_candidate_rows,
    )
    selected = select_snerv_modelsize_candidates(
        candidates,
        per_ceiling_limit=per_ceiling_limit,
    )
    by_ceiling: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        by_ceiling.setdefault(str(row.hard_byte_ceiling), []).append(row.as_dict())
    return {
        "schema": "snerv_modelsize_budget.v1",
        "family": "snerv",
        "num_pairs": int(num_pairs),
        "carrier_hw": [int(carrier_hw[0]), int(carrier_hw[1])],
        "wavelet": str(wavelet),
        "snerv_model_size_adapter": str(snerv_model_size_adapter),
        "fc_dims": [int(v) for v in fc_dims],
        "official_modelsize_mparams": [
            float(v) for v in official_modelsize_mparams
        ],
        "modelsize_control_profile_id": str(modelsize_control_profile["profile_id"]),
        "modelsize_control_profile": modelsize_control_profile,
        "official_enc_strds": [int(v) for v in official_enc_strds],
        "official_dec_strds": [int(v) for v in official_dec_strds],
        "emb_sizes": [int(v) for v in emb_sizes],
        "patch_radius": int(patch_radius),
        "mfu_scales": [int(v) for v in mfu_scales],
        "hfr_gain": float(hfr_gain),
        "temporal_context": int(temporal_context),
        "temporal_modes": [str(v) for v in temporal_modes],
        "hard_byte_ceilings": sorted({int(v) for v in hard_byte_ceilings}),
        "candidate_count": len(candidates),
        "invalid_candidate_count": len(invalid_candidate_rows),
        "invalid_candidates": invalid_candidate_rows,
        "selected_candidate_count": len(selected),
        "selected_candidates": [row.as_dict() for row in selected],
        "selected_candidates_by_ceiling": by_ceiling,
        "budget_math": {
            "contest_rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "nominal_payload_is_not_authority": True,
            "modelsize_control_contract_required_true_fields": list(
                MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS
            ),
            "rate_authority_surface": MODELSIZE_RATE_AUTHORITY_SURFACE,
            "selection_strategy": (
                "retain diverse DWT levels and LF/step precision points under "
                "each byte ceiling, then let real SNAR1 archive bytes and "
                "full-video scorer replay arbitrate"
            ),
            "real_authority": (
                "receiver-visible SNAR1 packet or packaged archive.zip bytes "
                "from run_snerv_advisory/export_snerv_archive_bound_candidate_package"
            ),
            "marginal_rule": (
                "admit LF precision, step-map precision, or HF decoder payload "
                "only when full-video P18/P19 distortion reduction per byte "
                "exceeds the contest rate price"
            ),
        },
        "blockers": [
            "snerv_modelsize_candidates_require_real_snAR1_archive_byte_oracle",
            "snerv_modelsize_candidates_require_full_video_mlx_prefilter",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def official_nerv_oss_flag_audit() -> dict[str, Any]:
    """Return the upstream flag families that should inform local adapters."""

    return {
        "schema": OSS_FLAG_AUDIT_SCHEMA,
        "sources": {
            "hnerv": "https://github.com/haochen-rye/HNeRV",
            "snerv": "https://github.com/qwertja/SNeRV",
            "hinerv": "https://github.com/hmkx/HiNeRV",
            "local_research_basis": "tac.optimization.research_basis",
            "local_priority_seam": "tac.analysis.nerv_top_priority_stack_seam",
        },
        "hnerv_high_ev_flags": [
            "--modelsize",
            "--ks",
            "--reduce",
            "--lower_width",
            "--enc_strds",
            "--dec_strds",
            "--enc_dim",
            "--fc_hw",
            "--conv_type",
            "--num_blks",
            "--quant_model_bit",
            "--quant_embed_bit",
            "--quant_axis",
            "--loss",
            "--lr_type",
            "--out_bias",
        ],
        "snerv_high_ev_flags": [
            "--modelsize",
            "--fc_dim",
            "--emb_size",
            "--num_blocks",
            "--enc_strds",
            "--dec_strds",
            "--quant_model_bit",
            "--quant_embed_bit",
            "--quant_embed2_bit",
            "--quant_axis",
            "--model snerv_t",
            "--model snerv_t_2d",
            "--grad_max_norm",
            "--loss",
            "--lr_type",
        ],
        "hinerv_high_ev_flags": [
            "--channels",
            "--channels-reduce",
            "--channels-reduce-base",
            "--channels-min",
            "--depths",
            "--exps",
            "--kernels",
            "--scales-t",
            "--scales-hw",
            "--base-grid-size",
            "--base-grid-level",
            "--base-grid-level-scale",
            "--enc-grid-size",
            "--enc-grid-level",
            "--enc-grid-level-scale",
            "--enc-grid-depth-scale",
            "--upsample-type",
            "--eval-patch-size",
            "--prune-ratio",
            "--prune-weight",
            "--quant-level",
            "--quant-noise",
            "--quant-ste",
            "--bitstream",
            "--bitstream-q",
        ],
        "local_gap": (
            "Current local HiNeRV execution exposes latent_dim/embed_dim/"
            "decoder_channel/decoder_codec. It does not yet expose upstream "
            "stride/kernel/reduction/fc/grid/prune schedules as executable "
            "capacity atoms."
        ),
        "control_to_local_consumer_map": [
            {
                "control_family": "archive_byte_capacity",
                "upstream_flags": [
                    "--modelsize",
                    "--fc_dim",
                    "--channels",
                    "--channels-reduce",
                    "--depths",
                    "--base-grid-size",
                ],
                "local_consumers": [
                    "tac.analysis.nerv_modelsize_budget",
                    "tools/run_compact_renderer_mlx_spine_runner.py",
                    "tac.substrates.hprc.spine_acquisition",
                    "tac.substrates.hprc.spine_bounded_runner",
                ],
                "exploit": (
                    "invert target archive bytes into capacity candidates, train "
                    "only byte-plausible points, then promote by measured "
                    "archive bytes"
                ),
            },
            {
                "control_family": "scorer_saliency_and_inverse_steganalysis",
                "upstream_flags": [
                    "--loss",
                    "--reg",
                    "--eval-metric",
                    "--quant-noise",
                    "--quant-ste",
                ],
                "local_consumers": [
                    "tac.analysis.score_exact_saliency",
                    "tac.analysis.hinerv_latent_linf_allocation",
                    "tac.analysis.inverse_steganalysis_linf_vs_l2_gate",
                    "tac.master_gradient_pose_vulnerability",
                    "tac.sensitivity_map",
                    "experiments/build_component_response_plan_from_sensitivity_artifacts.py",
                ],
                "exploit": (
                    "replace PSNR/MSE-only capacity allocation with joint "
                    "P18/P19 scorer-gradient pricing over decoder weights, "
                    "latents, regions, and pair incidence"
                ),
            },
            {
                "control_family": "quantization_and_entropy_payload",
                "upstream_flags": [
                    "--quant_model_bit",
                    "--quant_embed_bit",
                    "--quant_embed2_bit",
                    "--quant_axis",
                    "--quant-level",
                    "--bitstream-q",
                ],
                "local_consumers": [
                    "tac.substrates._shared.decoder_state_codec",
                    "tools/probe_snerv_decoder_mode_assignments.py",
                    "src/tac/analysis/snerv_step_map_coder.py",
                    "tools/profile_pact_nerv_selector_v3_mlx_section_value.py",
                    "tools/profile_pact_nerv_selector_v4_mlx_section_value.py",
                ],
                "exploit": (
                    "choose int2/int4/int8/fp16/codebook/bitplane payloads per "
                    "tensor or token from scorer value, not uniform bit depth"
                ),
            },
            {
                "control_family": "resolution_patch_and_pose_pathway",
                "upstream_flags": [
                    "--resize_list",
                    "--input-size",
                    "--patch-size",
                    "--eval-patch-size",
                    "--scales-t",
                    "--scales-hw",
                    "--upsample-type",
                ],
                "local_consumers": [
                    "tac.substrates.hprc.resolution_contract",
                    "materialize_mlx_scorer_cache_from_submission.py",
                    "tac.analysis.nerv_top_priority_stack_seam",
                    "experiments/gt_sparse_tto.py",
                    "experiments/renderer_tto.py",
                ],
                "exploit": (
                    "encode near scorer input resolution, protect pose-critical "
                    "pathways, and spend high-res rate only where PoseNet/SegNet "
                    "actually observe it"
                ),
            },
            {
                "control_family": "synergy_and_component_attribution",
                "upstream_flags": [
                    "--eval_freq",
                    "--eval-epochs",
                    "--profile",
                    "--bitstream",
                    "--bitstream-q",
                ],
                "local_consumers": [
                    "tools/run_mlx_scorer_response_cache.py",
                    "tools/profile_mlx_scorer_response_cache.py",
                    "tools/build_byte_shaving_signal_surface.py",
                    "tools/canvas_multiop_composition_closed_form_prediction_sweep.py",
                    "tools/check_substrate_dykstra_feasibility.py",
                ],
                "exploit": (
                    "turn every trained/exported candidate into component "
                    "movement, section value, Venn/synergy, and Dykstra feasibility "
                    "signal before exact auth spend"
                ),
            },
        ],
        "cross_variant_design_priors": [
            {
                "variant": "HNeRV / PR95-HNeRV",
                "role": "control arm and cheap-by-construction reference",
                "transfer_to_hinerv": [
                    "archive-byte budget should be input, not afterthought",
                    "model weights are the dominant score-bearing payload",
                    "8-stage/curriculum/Muon/QAT/coder-aware regularization are rate levers too",
                ],
                "transfer_to_snerv": [
                    "do not let generated-HF novelty replace byte-closed archive discipline",
                    "train under quantized payload pressure, not just posthoc LF storage",
                ],
            },
            {
                "variant": "SR-NeRV",
                "role": "resolution-axis enhancer",
                "transfer_to_hinerv": [
                    "encode at scorer-observed resolution first",
                    "super-resolve only for receiver output compliance",
                    "protect PoseNet geometry separately when low-res content is insufficient",
                ],
                "transfer_to_snerv": [
                    "LF/HF split should be paired with scorer-resolution dead-zone checks",
                    "HF generation should target SegNet boundary and PoseNet geometry, not perceptual detail",
                ],
            },
            {
                "variant": "RNeRV / E-NeRV",
                "role": "spatial-temporal disentanglement and config-search prior",
                "transfer_to_hinerv": [
                    "separate content amortized in decoder from per-pair motion or latent channels",
                    "search capacity distribution across temporal/spatial blocks",
                ],
                "transfer_to_snerv": [
                    "use temporal side information where pair geometry dominates PoseNet",
                    "consider temporal SNeRV_T/SNeRV_T_2D branches as pose-path enhancers",
                ],
            },
            {
                "variant": "FFNeRV",
                "role": "flow-guided pose-channel enhancer",
                "transfer_to_hinerv": [
                    "add flow/ego-motion-conditioned pathway only if byte-priced and receiver-closed",
                    "use PoseNet Fisher signal to decide where flow conditioning buys bytes",
                ],
                "transfer_to_snerv": [
                    "generated detail can be flow-conditioned for pose-critical regions",
                ],
            },
            {
                "variant": "BoostNeRV",
                "role": "conditional decoder / temporal affine bolt-on",
                "transfer_to_hinerv": [
                    "use as a byte-priced enhancer after base capacity is scorer-fit",
                    "regularize conditional parameters for entropy coding",
                ],
                "transfer_to_snerv": [
                    "test as HF-generator conditioner, not as a separate carrier",
                ],
            },
            {
                "variant": "HiNeRV upstream",
                "role": "hierarchical grid, patch, prune, quant bitstream prior",
                "transfer_to_hinerv": [
                    "promote pruning/QAT/bitstream stages into queue-owned schedule",
                    "map grid/channel/depth controls to archive-byte candidates",
                ],
                "transfer_to_snerv": [
                    "use patch/grid scheduling for full-video training throughput and scorer cache reuse",
                ],
            },
        ],
        "top_priority": (
            "Implement archive-byte inversion first: target byte ceiling -> "
            "capacity candidates -> real train/export archive byte oracle -> "
            "full-video MLX prefilter -> local CPU replay."
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


__all__ = [
    "DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID",
    "DEFAULT_SNERV_MODELSIZE_DEC_STRDS",
    "DEFAULT_SNERV_MODELSIZE_ENC_STRDS",
    "DEFAULT_SNERV_MODEL_SIZE_ADAPTER",
    "DEFAULT_SNERV_OFFICIAL_DEC_STRDS",
    "DEFAULT_SNERV_OFFICIAL_ENC_STRDS",
    "DEFAULT_SNERV_TEMPORAL_MODE",
    "MODELSIZE_CONTROL_AUTHORITY_SPLIT_SCHEMA",
    "SNERV_CONTEST_RECEIVER_PROFILE_ID",
    "SNERV_MANUAL_STRIDE_OVERRIDE_PROFILE_ID",
    "SNERV_MODELSIZE_CONTROL_PROFILES",
    "SNERV_OFFICIAL_CLI_DEFAULT_PROFILE_ID",
    "SNERV_OFFICIAL_MFU_HFR_TUB_LEVELS",
    "SNERV_PACT_RECEIVER_PROFILE_ID",
    "HinervModelSizeCandidate",
    "NervModelSizeBudgetError",
    "SnervModelSizeCandidate",
    "analyze_hinerv_modelsize_candidate",
    "analyze_snerv_modelsize_candidate",
    "build_hinerv_config_from_size_knobs",
    "build_hinerv_modelsize_budget_report",
    "build_snerv_modelsize_budget_report",
    "decoder_codec_nominal_bits",
    "enumerate_hinerv_modelsize_candidates",
    "enumerate_snerv_modelsize_candidates",
    "modelsize_control_authority_split",
    "modelsize_control_precedence_contract",
    "official_nerv_oss_flag_audit",
    "select_hinerv_modelsize_candidates",
    "select_snerv_modelsize_candidates",
    "snerv_decoder_codec_nominal_bits",
    "snerv_model_size_adapter_from_id_token",
    "snerv_model_size_adapter_id_token",
    "snerv_modelsize_candidate_id_from_controls",
    "snerv_modelsize_control_profile",
    "snerv_temporal_mode_from_id_token",
    "snerv_temporal_mode_id_token",
    "tag_hinerv_target_modelsize_candidate",
]
