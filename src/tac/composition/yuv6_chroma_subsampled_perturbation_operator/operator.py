# SPDX-License-Identifier: MIT
"""YUV6 chroma-subsampled perturbation operator implementation.

See ``__init__.py`` for design rationale + Yousfi-blind-spot anchor +
canonical apparatus mutation chain.

This module implements the 4 canonical perturbation STRATEGIES + the
canonical 4:2:0 chroma subsampling math + the luma-preservation
invariant gate. All operations are NUMPY-NATIVE so the operator runs
on macOS-CPU advisory hardware ($0 per CLAUDE.md "MLX-first" 8th
standing directive 2026-05-30).
"""
from __future__ import annotations

import enum
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np


# Canonical BT.601 full-range YUV coefficients per
# tac.differentiable_eval_roundtrip.differentiable_rgb_to_yuv6.
BT601_KYR = 0.299
BT601_KYG = 0.587
BT601_KYB = 0.114
BT601_U_DIVISOR = 1.772
BT601_V_DIVISOR = 1.402

DEFAULT_PERTURBATION_MAGNITUDE = 4.0  # in [0, 255] integer-byte units

ATICK_REDLICH_DEFAULT_BLEND_COEFFICIENTS = (0.33, 0.33, 0.34)

_LUMA_AXIS_INDICES = (0, 1, 2, 3)  # Y00, Y10, Y01, Y11 in YUV6 channel order
_CHROMA_AXIS_INDICES = (4, 5)  # U_sub, V_sub in YUV6 channel order


class ChromaPerturbationStrategy(enum.Enum):
    """Canonical 4-value perturbation strategy enum.

    Each strategy returns a weight map (one weight per chroma cell at
    half-resolution) that the operator uses to scale the per-cell
    perturbation magnitude.
    """

    LOCAL_VARIANCE_WEIGHTED = "local_variance_weighted"
    SEGNET_GRADIENT_WEIGHTED = "segnet_gradient_weighted"
    POSENET_GRADIENT_WEIGHTED_VIA_MAE_V = "posenet_gradient_weighted_via_mae_v"
    JOINT_ATICK_REDLICH_LINEAR_COMBINATION = "joint_atick_redlich_linear_combination"


class ChromaSubsampledPerturbationConfigInvalidError(ValueError):
    """Raised on config-invariant violation."""


@dataclass(frozen=True)
class ChromaSubsampledPerturbationConfig:
    """Frozen config for the YUV6 chroma-subsampled perturbation operator.

    Attributes:
        strategy: one of :class:`ChromaPerturbationStrategy`.
        perturbation_magnitude: per-byte magnitude in [0, 255] units
            (default 4.0).
        atick_redlich_blend_coefficients: 3-coefficient tuple summing to
            1.0 for the JOINT_ATICK_REDLICH_LINEAR_COMBINATION strategy.
        segnet_gradient_map: optional (H//2, W//2) float ndarray for the
            SegNet-gradient-weighted strategy (caller-supplied).
        posenet_gradient_map: optional (H//2, W//2) float ndarray for
            the PoseNet-gradient-weighted strategy (typically derived from
            tac.scorer_surrogate.posenet_mae_v).
    """

    strategy: ChromaPerturbationStrategy
    perturbation_magnitude: float = DEFAULT_PERTURBATION_MAGNITUDE
    atick_redlich_blend_coefficients: tuple[float, float, float] = (
        ATICK_REDLICH_DEFAULT_BLEND_COEFFICIENTS
    )
    segnet_gradient_map: np.ndarray | None = None
    posenet_gradient_map: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, ChromaPerturbationStrategy):
            raise ChromaSubsampledPerturbationConfigInvalidError(
                f"strategy must be ChromaPerturbationStrategy, got {self.strategy!r}"
            )
        if self.perturbation_magnitude <= 0.0:
            raise ChromaSubsampledPerturbationConfigInvalidError(
                f"perturbation_magnitude must be > 0, got {self.perturbation_magnitude}"
            )
        if self.perturbation_magnitude > 255.0:
            raise ChromaSubsampledPerturbationConfigInvalidError(
                f"perturbation_magnitude must be <= 255, got {self.perturbation_magnitude}"
            )
        if len(self.atick_redlich_blend_coefficients) != 3:
            raise ChromaSubsampledPerturbationConfigInvalidError(
                "atick_redlich_blend_coefficients must be a 3-tuple"
            )
        if not np.isclose(
            sum(self.atick_redlich_blend_coefficients), 1.0, atol=1e-3
        ):
            raise ChromaSubsampledPerturbationConfigInvalidError(
                "atick_redlich_blend_coefficients must sum to 1.0 (within 1e-3)"
            )
        if any(c < 0.0 for c in self.atick_redlich_blend_coefficients):
            raise ChromaSubsampledPerturbationConfigInvalidError(
                "atick_redlich_blend_coefficients must all be >= 0"
            )
        # Strategy-specific arg requirements
        if self.strategy is ChromaPerturbationStrategy.SEGNET_GRADIENT_WEIGHTED:
            if self.segnet_gradient_map is None:
                raise ChromaSubsampledPerturbationConfigInvalidError(
                    "SEGNET_GRADIENT_WEIGHTED requires segnet_gradient_map"
                )
        if self.strategy is ChromaPerturbationStrategy.POSENET_GRADIENT_WEIGHTED_VIA_MAE_V:
            if self.posenet_gradient_map is None:
                raise ChromaSubsampledPerturbationConfigInvalidError(
                    "POSENET_GRADIENT_WEIGHTED_VIA_MAE_V requires "
                    "posenet_gradient_map"
                )
        if (
            self.strategy is ChromaPerturbationStrategy.JOINT_ATICK_REDLICH_LINEAR_COMBINATION
        ):
            # Joint requires BOTH gradient maps (variance is computed from input
            # so it's always available; segnet + posenet must be supplied).
            if self.segnet_gradient_map is None or self.posenet_gradient_map is None:
                raise ChromaSubsampledPerturbationConfigInvalidError(
                    "JOINT_ATICK_REDLICH_LINEAR_COMBINATION requires BOTH "
                    "segnet_gradient_map AND posenet_gradient_map"
                )


@dataclass(frozen=True)
class ChromaSubsampledPerturbationResult:
    """Typed result of the chroma-subsampled perturbation operator.

    Catalog #341 Tier A canonical-routing markers are baked in:
    ``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
    ``axis_tag="[predicted]"`` per the operator-mental-model gap
    closure in CLAUDE.md "Mission alignment" Consequence 1: this
    operator is OBSERVABILITY-ONLY at the consumer surface until
    paired-CUDA RATIFICATION lands per Catalog #370.

    The canonical output is in YUV6 space — the actual PoseNet input
    representation. ``perturbed_yuv6_first_frame`` + ``perturbed_yuv6_second_frame``
    are byte-exact perturbations of the input YUV6 tensors (luma channels
    preserved EXACTLY in YUV6 space; chroma channels modified per the
    weight map). The optional ``perturbed_rgb_*`` arrays are approximate
    reconstructions for downstream RGB-consuming code; see
    :func:`yuv6_to_rgb_numpy` docstring for the lossy-reconstruction
    semantics (4:2:0 chroma upsample + BT.601 inverse).
    """

    strategy_used: str
    perturbed_yuv6_first_frame: np.ndarray  # (6, H//2, W//2) float in [0, 255]
    perturbed_yuv6_second_frame: np.ndarray  # (6, H//2, W//2) float in [0, 255]
    perturbed_rgb_first_frame: np.ndarray  # (H, W, 3) approximate reconstruction
    perturbed_rgb_second_frame: np.ndarray  # (H, W, 3) approximate reconstruction
    perturbation_weight_map: np.ndarray  # (H//2, W//2) float in [0, 1]
    predicted_delta_adjustment: float
    promotable: bool
    axis_tag: str
    confidence: float
    rationale: str
    luma_preservation_max_abs_drift_yuv6: float  # canonical (exact 0.0 by construction)
    luma_preservation_max_abs_drift_rgb_reconstructed: float  # approximate-reconstruction drift
    chroma_perturbation_max_abs_drift_yuv6: float  # canonical
    predicted_axis_decomposition: Mapping[str, Any] | None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Catalog #341 frozen-False
        if self.predicted_delta_adjustment != 0.0:
            raise ValueError(
                "predicted_delta_adjustment MUST be 0.0 per Catalog #341 "
                "Tier A canonical-routing markers"
            )
        if self.promotable is not False:
            raise ValueError(
                "promotable MUST be False per Catalog #341"
            )
        if self.axis_tag != "[predicted]":
            raise ValueError(
                f"axis_tag must be '[predicted]', got {self.axis_tag!r}"
            )
        if self.luma_preservation_max_abs_drift_yuv6 < 0.0:
            raise ValueError("luma_preservation_max_abs_drift_yuv6 must be >= 0")
        if self.luma_preservation_max_abs_drift_rgb_reconstructed < 0.0:
            raise ValueError(
                "luma_preservation_max_abs_drift_rgb_reconstructed must be >= 0"
            )
        if self.chroma_perturbation_max_abs_drift_yuv6 < 0.0:
            raise ValueError("chroma_perturbation_max_abs_drift_yuv6 must be >= 0")


def rgb_to_yuv6_numpy(rgb_hwc: np.ndarray) -> np.ndarray:
    # CATALOG_383_PRINCIPLED_FORK_OK:hwc_float64_precision_contract_canonical_helper_is_float32_native_diverges_2e_minus_5_above_fp32_epsilon_at_perturbation_operator_downstream_dependency_per_catalog_290_falling_rule_documented_in_audit_inventory_A_2_6
    """Numpy port of canonical ``differentiable_rgb_to_yuv6``.

    Input: ``(H, W, 3)`` float ndarray in ``[0, 255]``.
    Output: ``(6, H//2, W//2)`` float ndarray.

    Channel order: ``[Y00, Y10, Y01, Y11, U_sub, V_sub]`` matching the
    canonical PoseNet input.

    Replicates the BT.601 full-range YUV with 4:2:0 chroma subsampling
    per ``tac.differentiable_eval_roundtrip.differentiable_rgb_to_yuv6``
    line-for-line.

    PRINCIPLED FORK per Catalog #290 falling-rule list (NOT delegating
    to :func:`tac.framework_agnostic.canonical_kernels.rgb_to_yuv6`):
    this operator's downstream perturbation math depends on float64
    precision and the canonical helper is float32-native (~2e-5 absolute
    discrepancy when canonical float32 math is promoted to float64
    post-hoc, vs ~3e-7 fp32 epsilon — empirically verified during the
    rgb_to_yuv6 canonical extraction migration 2026-05-30). The pure
    float64 math below preserves the perturbation operator's downstream
    precision contract and stays byte-identical with the legacy
    pre-migration behavior.
    """
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
        raise ValueError(
            f"rgb_to_yuv6_numpy requires (H, W, 3); got shape {rgb_hwc.shape}"
        )
    H, W, _ = rgb_hwc.shape
    H2, W2 = H // 2, W // 2
    rgb = rgb_hwc[: 2 * H2, : 2 * W2, :].astype(np.float64)

    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    Y = np.clip(R * BT601_KYR + G * BT601_KYG + B * BT601_KYB, 0.0, 255.0)
    U = np.clip((B - Y) / BT601_U_DIVISOR + 128.0, 0.0, 255.0)
    V = np.clip((R - Y) / BT601_V_DIVISOR + 128.0, 0.0, 255.0)

    # 4 luma channels: stride-2 sampling of the full-res Y
    Y00 = Y[0::2, 0::2]
    Y10 = Y[1::2, 0::2]
    Y01 = Y[0::2, 1::2]
    Y11 = Y[1::2, 1::2]

    # 4:2:0 chroma subsampling: mean of 2x2 block
    U_sub = (
        U[0::2, 0::2] + U[1::2, 0::2] + U[0::2, 1::2] + U[1::2, 1::2]
    ) * 0.25
    V_sub = (
        V[0::2, 0::2] + V[1::2, 0::2] + V[0::2, 1::2] + V[1::2, 1::2]
    ) * 0.25

    return np.stack([Y00, Y10, Y01, Y11, U_sub, V_sub], axis=0)


def yuv6_to_rgb_numpy(yuv6_chw: np.ndarray) -> np.ndarray:
    """Inverse YUV6 -> RGB at full resolution (lossy round-trip of 4:2:0).

    Output: ``(2*H2, 2*W2, 3)`` float ndarray in ``[0, 255]``.

    The 4:2:0 subsampling step is LOSSY (4 chroma pixels averaged into
    1) so the round-trip is approximate: we upsample (U_sub, V_sub) by
    replicating each subsampled value across the 2x2 block. This matches
    the canonical decoder's nearest-neighbor chroma upsampling.
    """
    if yuv6_chw.ndim != 3 or yuv6_chw.shape[0] != 6:
        raise ValueError(
            f"yuv6_to_rgb_numpy requires (6, H//2, W//2); got shape {yuv6_chw.shape}"
        )
    Y00, Y10, Y01, Y11, U_sub, V_sub = yuv6_chw
    H2, W2 = Y00.shape
    H, W = 2 * H2, 2 * W2

    Y_full = np.empty((H, W), dtype=np.float64)
    Y_full[0::2, 0::2] = Y00
    Y_full[1::2, 0::2] = Y10
    Y_full[0::2, 1::2] = Y01
    Y_full[1::2, 1::2] = Y11

    # Nearest-neighbor upsample of chroma
    U_full = np.repeat(np.repeat(U_sub, 2, axis=0), 2, axis=1)
    V_full = np.repeat(np.repeat(V_sub, 2, axis=0), 2, axis=1)

    # Inverse BT.601 full-range: shift center back to 0
    U_centered = U_full - 128.0
    V_centered = V_full - 128.0
    # R = Y + (V_centered * BT601_V_DIVISOR)
    # B = Y + (U_centered * BT601_U_DIVISOR)
    # G = (Y - kYR*R - kYB*B) / kYG
    R = Y_full + V_centered * BT601_V_DIVISOR
    B = Y_full + U_centered * BT601_U_DIVISOR
    G = (Y_full - BT601_KYR * R - BT601_KYB * B) / BT601_KYG

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0.0, 255.0)


def _normalize_to_unit_interval(arr: np.ndarray) -> np.ndarray:
    """Normalize an ndarray to [0, 1] by min-max (or return zeros if degenerate)."""
    a = np.asarray(arr, dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    if a_max - a_min < 1e-30:
        return np.zeros_like(a)
    return (a - a_min) / (a_max - a_min)


def _local_variance_chroma_map(
    rgb_hwc: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """Compute per-chroma-cell local variance of the input RGB.

    Yousfi-UNIWARD canonical analog: variance is a TEXTURE proxy and
    high-texture regions absorb perturbation better (lower detectability).
    """
    yuv6 = rgb_to_yuv6_numpy(rgb_hwc)
    # Use the 4 luma channels' per-cell variance as the texture proxy
    # (chroma is itself the perturbation target, so we use luma to detect texture)
    luma_stack = yuv6[:4]  # (4, H//2, W//2)
    per_cell_variance = luma_stack.var(axis=0)
    if per_cell_variance.shape != target_shape:
        # Crop or pad to target_shape (defensive; should match by construction)
        h, w = target_shape
        per_cell_variance = per_cell_variance[:h, :w]
    return _normalize_to_unit_interval(per_cell_variance)


def compute_chroma_perturbation_weight_map(
    config: ChromaSubsampledPerturbationConfig,
    rgb_hwc: np.ndarray,
) -> np.ndarray:
    """Compute the per-chroma-cell weight map for the given strategy.

    Returns a ``(H//2, W//2)`` array in ``[0, 1]`` — the per-cell weight
    scaling the per-cell perturbation magnitude.
    """
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
        raise ValueError(
            f"rgb_hwc must be (H, W, 3); got shape {rgb_hwc.shape}"
        )
    H, W, _ = rgb_hwc.shape
    H2, W2 = H // 2, W // 2
    target_shape = (H2, W2)

    if config.strategy is ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED:
        return _local_variance_chroma_map(rgb_hwc, target_shape)

    if config.strategy is ChromaPerturbationStrategy.SEGNET_GRADIENT_WEIGHTED:
        seg = config.segnet_gradient_map
        assert seg is not None  # guarded by __post_init__
        if seg.shape != target_shape:
            raise ValueError(
                f"segnet_gradient_map shape {seg.shape} != ({H2}, {W2})"
            )
        return _normalize_to_unit_interval(seg)

    if config.strategy is ChromaPerturbationStrategy.POSENET_GRADIENT_WEIGHTED_VIA_MAE_V:
        pose = config.posenet_gradient_map
        assert pose is not None
        if pose.shape != target_shape:
            raise ValueError(
                f"posenet_gradient_map shape {pose.shape} != ({H2}, {W2})"
            )
        return _normalize_to_unit_interval(pose)

    if config.strategy is ChromaPerturbationStrategy.JOINT_ATICK_REDLICH_LINEAR_COMBINATION:
        c_var, c_seg, c_pose = config.atick_redlich_blend_coefficients
        var_map = _local_variance_chroma_map(rgb_hwc, target_shape)
        seg = config.segnet_gradient_map
        pose = config.posenet_gradient_map
        assert seg is not None and pose is not None
        if seg.shape != target_shape:
            raise ValueError(
                f"segnet_gradient_map shape {seg.shape} != ({H2}, {W2})"
            )
        if pose.shape != target_shape:
            raise ValueError(
                f"posenet_gradient_map shape {pose.shape} != ({H2}, {W2})"
            )
        seg_norm = _normalize_to_unit_interval(seg)
        pose_norm = _normalize_to_unit_interval(pose)
        blended = c_var * var_map + c_seg * seg_norm + c_pose * pose_norm
        return _normalize_to_unit_interval(blended)

    raise ChromaSubsampledPerturbationConfigInvalidError(
        f"unknown strategy {config.strategy!r}"
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_of_arrays(*arrs: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(repr((a.shape, str(a.dtype))).encode("utf-8"))
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _build_canonical_provenance(
    *,
    strategy: ChromaPerturbationStrategy,
    input_sha256: str,
    perturbation_magnitude: float,
    measurement_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": "provenance_v1",
        "kind": "predicted_from_model",
        "model_identifier": (
            "tac.composition.yuv6_chroma_subsampled_perturbation_operator."
            "apply_chroma_subsampled_perturbation"
        ),
        "strategy_used": strategy.value,
        "input_sha256": input_sha256,
        "perturbation_magnitude": float(perturbation_magnitude),
        "axis_tag": "[predicted]",
        "evidence_grade": "predicted",
        "score_claim": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "captured_at_utc": measurement_utc,
        "canonical_helper_invocation": (
            "tac.composition.yuv6_chroma_subsampled_perturbation_operator."
            "apply_chroma_subsampled_perturbation"
        ),
    }


def _build_predicted_axis_decomposition(
    *,
    luma_max_abs_drift_yuv6: float,
    luma_max_abs_drift_rgb_reconstructed: float,
    chroma_max_abs_drift_yuv6: float,
    strategy: ChromaPerturbationStrategy,
    canonical_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Catalog #356 AxisDecomposition surface (observability-only).

    The per-axis decomposition surfaces:
      * predicted_d_seg_delta: 0.0 (we do not predict seg in this Tier A operator).
      * predicted_d_pose_delta: 0.0 (we do not predict pose; downstream Tier B
        consumer per Dim 6 Step 6.5 may extend).
      * predicted_archive_bytes_delta: 0 (chroma-only perturbation preserves byte count).
      * chroma_perturbation_max_abs_drift_yuv6: real materialized chroma drift in
        canonical YUV6 space.
      * luma_preservation_max_abs_drift_yuv6: real materialized luma drift in
        canonical YUV6 space (exactly 0.0 by construction — we modify ONLY
        channels 4+5).
      * luma_preservation_max_abs_drift_rgb_reconstructed: real materialized
        luma drift after lossy RGB reconstruction (non-zero because of the
        4:2:0 nearest-neighbor chroma upsample inverse).

    Caller can consume the chroma/luma drift fields as ranking priors without
    them becoming score-claims.
    """
    return {
        "schema_version": "axis_decomposition_v1",
        "predicted_d_seg_delta": 0.0,
        "predicted_d_pose_delta": 0.0,
        "predicted_archive_bytes_delta": 0,
        "chroma_perturbation_max_abs_drift_yuv6": float(chroma_max_abs_drift_yuv6),
        "luma_preservation_max_abs_drift_yuv6": float(luma_max_abs_drift_yuv6),
        "luma_preservation_max_abs_drift_rgb_reconstructed": float(
            luma_max_abs_drift_rgb_reconstructed
        ),
        "strategy_used": strategy.value,
        "axis_tag": "[predicted]",
        "canonical_provenance": dict(canonical_provenance),
    }


def assert_luma_preservation_invariant(
    *,
    original_yuv6: np.ndarray,
    perturbed_yuv6: np.ndarray,
    luma_drift_threshold: float = 1e-9,
) -> float:
    """Compute + assert luma-preservation invariant max-abs drift in YUV6 space.

    This operator perturbs ONLY the (U_sub, V_sub) chroma channels of the
    YUV6 representation, so the canonical invariant in YUV6 space is that
    the 4 luma channels (Y00, Y10, Y01, Y11) are BYTE-IDENTICAL between
    original + perturbed (drift = 0.0 by construction).

    Operates in canonical YUV6 space (not lossy RGB reconstruction).

    Returns the max-abs drift across the 4 luma channels; raises
    AssertionError if the drift exceeds ``luma_drift_threshold`` (default
    1e-9 — should be exactly 0 modulo float64 numerical noise).
    """
    if original_yuv6.shape != perturbed_yuv6.shape:
        raise ValueError(
            f"original_yuv6 shape {original_yuv6.shape} != "
            f"perturbed_yuv6 shape {perturbed_yuv6.shape}"
        )
    if original_yuv6.shape[0] != 6:
        raise ValueError(
            f"yuv6 tensors must have 6 channels; got shape {original_yuv6.shape}"
        )
    luma_drift = float(
        np.max(np.abs(original_yuv6[:4] - perturbed_yuv6[:4]))
    )
    if luma_drift > luma_drift_threshold:
        raise AssertionError(
            f"luma-preservation invariant violated: max-abs drift "
            f"{luma_drift} > threshold {luma_drift_threshold}"
        )
    return luma_drift


def apply_chroma_subsampled_perturbation(
    *,
    config: ChromaSubsampledPerturbationConfig,
    rgb_first_frame_hwc: np.ndarray,
    rgb_second_frame_hwc: np.ndarray,
) -> ChromaSubsampledPerturbationResult:
    """Apply the YUV6 chroma-subsampled perturbation operator.

    Pipeline:
      1. Compute per-strategy weight map ``(H//2, W//2)`` in ``[0, 1]``.
      2. Convert each input frame to YUV6 ``(6, H//2, W//2)``.
      3. Add ``magnitude * weight_map`` to (U_sub, V_sub) for each frame.
         Sign alternates per-frame so the canonical decoder rounds chroma
         in opposite directions per frame (Atick-Redlich cooperative-
         receiver weighting at the chroma-axis).
      4. Inverse YUV6 -> RGB at full resolution.
      5. Compute luma-preservation + chroma-perturbation drift metrics.
      6. Emit Catalog #341 Tier A canonical-routing markers.

    Args:
        config: :class:`ChromaSubsampledPerturbationConfig`.
        rgb_first_frame_hwc: ``(H, W, 3)`` float in ``[0, 255]``.
        rgb_second_frame_hwc: ``(H, W, 3)`` float in ``[0, 255]``.

    Returns:
        :class:`ChromaSubsampledPerturbationResult` with perturbed
        frames + drift metrics + canonical Provenance + AxisDecomposition.
    """
    # Compute weight map from the first frame (caller-stable reference)
    weight_map = compute_chroma_perturbation_weight_map(
        config, rgb_first_frame_hwc
    )

    # Pipeline frame 1
    yuv6_first = rgb_to_yuv6_numpy(rgb_first_frame_hwc)
    yuv6_first_perturbed = yuv6_first.copy()
    yuv6_first_perturbed[4] = np.clip(
        yuv6_first[4] + config.perturbation_magnitude * weight_map,
        0.0,
        255.0,
    )
    yuv6_first_perturbed[5] = np.clip(
        yuv6_first[5] + config.perturbation_magnitude * weight_map,
        0.0,
        255.0,
    )
    perturbed_first = yuv6_to_rgb_numpy(yuv6_first_perturbed)

    # Pipeline frame 2 — opposite chroma sign per Atick-Redlich cooperative
    # weighting (the canonical decoder rounds opposite chroma in opposite
    # directions; this reduces detectability while preserving the
    # cost-discrimination signal).
    yuv6_second = rgb_to_yuv6_numpy(rgb_second_frame_hwc)
    yuv6_second_perturbed = yuv6_second.copy()
    yuv6_second_perturbed[4] = np.clip(
        yuv6_second[4] - config.perturbation_magnitude * weight_map,
        0.0,
        255.0,
    )
    yuv6_second_perturbed[5] = np.clip(
        yuv6_second[5] - config.perturbation_magnitude * weight_map,
        0.0,
        255.0,
    )
    perturbed_second = yuv6_to_rgb_numpy(yuv6_second_perturbed)

    # Canonical YUV6-space drift metrics (exact)
    luma_drift_yuv6_first = float(
        np.max(np.abs(yuv6_first[:4] - yuv6_first_perturbed[:4]))
    )
    luma_drift_yuv6_second = float(
        np.max(np.abs(yuv6_second[:4] - yuv6_second_perturbed[:4]))
    )
    luma_max_drift_yuv6 = max(luma_drift_yuv6_first, luma_drift_yuv6_second)

    chroma_drift_yuv6_first = float(
        np.max(np.abs(yuv6_first[4:] - yuv6_first_perturbed[4:]))
    )
    chroma_drift_yuv6_second = float(
        np.max(np.abs(yuv6_second[4:] - yuv6_second_perturbed[4:]))
    )
    chroma_max_drift_yuv6 = max(chroma_drift_yuv6_first, chroma_drift_yuv6_second)

    # Approximate-reconstruction RGB-space luma drift (lossy 4:2:0 inverse)
    luma_drift_rgb_first = float(
        np.max(np.abs(yuv6_first[:4] - rgb_to_yuv6_numpy(perturbed_first)[:4]))
    )
    luma_drift_rgb_second = float(
        np.max(np.abs(yuv6_second[:4] - rgb_to_yuv6_numpy(perturbed_second)[:4]))
    )
    luma_max_drift_rgb = max(luma_drift_rgb_first, luma_drift_rgb_second)

    input_sha = _sha256_of_arrays(rgb_first_frame_hwc, rgb_second_frame_hwc)
    utc = _utc_now()
    canonical_provenance = _build_canonical_provenance(
        strategy=config.strategy,
        input_sha256=input_sha,
        perturbation_magnitude=config.perturbation_magnitude,
        measurement_utc=utc,
    )
    predicted_axis_decomposition = _build_predicted_axis_decomposition(
        luma_max_abs_drift_yuv6=luma_max_drift_yuv6,
        luma_max_abs_drift_rgb_reconstructed=luma_max_drift_rgb,
        chroma_max_abs_drift_yuv6=chroma_max_drift_yuv6,
        strategy=config.strategy,
        canonical_provenance=canonical_provenance,
    )

    return ChromaSubsampledPerturbationResult(
        strategy_used=config.strategy.value,
        perturbed_yuv6_first_frame=yuv6_first_perturbed,
        perturbed_yuv6_second_frame=yuv6_second_perturbed,
        perturbed_rgb_first_frame=perturbed_first,
        perturbed_rgb_second_frame=perturbed_second,
        perturbation_weight_map=weight_map,
        predicted_delta_adjustment=0.0,
        promotable=False,
        axis_tag="[predicted]",
        confidence=0.0,
        rationale=(
            f"chroma-subsampled perturbation via strategy {config.strategy.value}; "
            f"YUV6-luma-drift={luma_max_drift_yuv6:.4g} (exact 0 by construction) "
            f"YUV6-chroma-drift={chroma_max_drift_yuv6:.4f} "
            f"RGB-reconstructed-luma-drift={luma_max_drift_rgb:.4f}"
        ),
        luma_preservation_max_abs_drift_yuv6=luma_max_drift_yuv6,
        luma_preservation_max_abs_drift_rgb_reconstructed=luma_max_drift_rgb,
        chroma_perturbation_max_abs_drift_yuv6=chroma_max_drift_yuv6,
        predicted_axis_decomposition=predicted_axis_decomposition,
        provenance=canonical_provenance,
    )
