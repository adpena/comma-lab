# SPDX-License-Identifier: MIT
"""Typed primitives for the DDM PT1 continuous-paint ceiling experiment.

PT1 deliberately keeps receiver mechanisms separate:

``hard_camera_placement``
    Evaluate a continuous class partition at camera-pixel centres, choose one
    class, and write its full-amplitude uint8 prototype.  No blended value is
    quantized.  Sub-cell placement is expressed by the evaluator-owned
    bilinear camera-to-scorer resize.

``analytic_coverage_blend``
    Integrate the same signed-distance fields analytically at the render grid
    and blend prototypes before uint8.  This is the v14-adverse control.  It is
    never silently composed with the hard-placement arm.

``global_amplitude_statistics_match``
    Keep class geometry fixed while matching the global per-channel moments.
    This isolates a BN/SE-sensitive amplitude-statistics mechanism.

``stratum_spectrum_match``
    Keep class geometry fixed while fitting per-stratum coefficients over the
    existing measured-passband texture-trunk basis.  This is a diagnostic for
    local-spectrum/region-ERF sensitivity and is not silently identified with
    the analytic-coverage control.

The module contains no scorer and makes no score claim.  A verdict exists only
after the exact frozen scorer has consumed the returned camera bytes.  Small
fixture tests exercise behavior; they are not empirical evidence.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import ndimage

from tac.boundary_math.phase_primitives import gt_tie_targets_numpy
from tac.boundary_math.texture_trunk import (
    build_gabor_bank_numpy,
    default_band_spec,
)
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    head_flip_distance_feature_space,
)
from tac.optimization.ddm_dv2_sdwl1 import (
    SentenceLayout,
    SentenceOptions,
    TemporalMode,
    extract_fact_inventory,
    measure_serialization,
)
from tac.through_r.resolution_chain import CAMERA_HW, SEG_HW

N_CLASSES: Final = 5
CLASS_ORDER: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
MECHANISM_ORDER: Final = (
    "sub_cell_placement",
    "bn_se_amplitude_statistics",
    "texture_prior_or_region_erf",
    "class_interaction",
)


class ContinuousPaintError(ValueError):
    """Raised when PT1 inputs would make a mechanism or custody claim ambiguous."""


@dataclass(frozen=True, slots=True)
class SDWL1DescriptionDebt:
    """Exact parse-backed charge for freshly fitted target geometry."""

    bytes: int
    sha256: str
    exact_parseback: bool
    semantic_sha256: str
    described_scalar_facts: int


@dataclass(frozen=True, slots=True)
class MechanismDecomposition:
    """Disjoint operational attribution of corrected baseline errors."""

    baseline_errors: int
    corrected_total_primary: int
    corrected_total_statistics_only: int
    corrected_total_secondary_only: int
    sub_cell_placement: int
    bn_se_amplitude_statistics: int
    texture_prior_or_region_erf: int
    class_interaction: int
    residual_errors_primary: int
    placement_attributable: int
    placement_recovered: int
    measured_survival_wall_fraction: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "baseline_errors": self.baseline_errors,
            "bn_se_amplitude_statistics": self.bn_se_amplitude_statistics,
            "class_interaction": self.class_interaction,
            "corrected_total_primary": self.corrected_total_primary,
            "corrected_total_secondary_only": self.corrected_total_secondary_only,
            "corrected_total_statistics_only": self.corrected_total_statistics_only,
            "measured_survival_wall_fraction": self.measured_survival_wall_fraction,
            "placement_attributable": self.placement_attributable,
            "placement_recovered": self.placement_recovered,
            "residual_errors_primary": self.residual_errors_primary,
            "sub_cell_placement": self.sub_cell_placement,
            "texture_prior_or_region_erf": self.texture_prior_or_region_erf,
        }


def _labels(labels: np.ndarray) -> np.ndarray:
    value = np.asarray(labels)
    if value.ndim != 3 or value.shape[1:] != SEG_HW:
        raise ContinuousPaintError(
            f"labels must be [pairs,{SEG_HW[0]},{SEG_HW[1]}], got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.integer):
        raise ContinuousPaintError("labels must use an integer dtype")
    if value.size == 0 or int(value.min()) < 0 or int(value.max()) >= N_CLASSES:
        raise ContinuousPaintError("labels must contain class IDs in [0,5)")
    return np.ascontiguousarray(value, dtype=np.uint8)


def _palette(palette_rgb_u8: np.ndarray) -> np.ndarray:
    value = np.asarray(palette_rgb_u8)
    if (
        value.shape != (N_CLASSES, 3)
        or not np.issubdtype(value.dtype, np.integer)
        or np.any((value < 0) | (value > 255))
    ):
        raise ContinuousPaintError("palette must be uint8-compatible [5,3]")
    return np.ascontiguousarray(value, dtype=np.uint8)


def signed_distance_fields(
    labels: np.ndarray,
    *,
    margins: np.ndarray | None = None,
) -> np.ndarray:
    """Fit deterministic local sub-pixel fields to a scorer-grid partition.

    For each class, the field is ``distance(inside)-distance(outside)`` with
    the zero crossing halfway between opposing scorer-cell centres.  This is an
    encode-side fit to the supplied partition, not a decoder-free fact.  Its
    bytes must therefore be charged by :func:`measure_fitted_geometry_sdwl1`.
    """

    value = _labels(labels)
    margin_values: np.ndarray | None = None
    if margins is not None:
        margin_values = np.asarray(margins, dtype=np.float32)
        if (
            margin_values.shape != value.shape
            or not np.all(np.isfinite(margin_values))
            or np.any(margin_values < 0.0)
        ):
            raise ContinuousPaintError(
                "sub-pixel localization margins must be finite, nonnegative, "
                "and share the label shape"
            )
    fields = np.empty((*value.shape, N_CLASSES), dtype=np.float32)
    saturation = float(np.hypot(*SEG_HW))
    for pair in range(value.shape[0]):
        for class_id in range(N_CLASSES):
            mask = value[pair] == class_id
            if not np.any(mask):
                fields[pair, :, :, class_id] = -saturation
                continue
            if np.all(mask):
                fields[pair, :, :, class_id] = saturation
                continue
            inside = ndimage.distance_transform_edt(mask)
            outside = ndimage.distance_transform_edt(~mask)
            fields[pair, :, :, class_id] = (inside - outside).astype(np.float32)
        if margin_values is not None:
            tie, direction, active = gt_tie_targets_numpy(
                value[pair],
                margin_values[pair],
                band=float("inf"),
            )
            rows, columns = np.nonzero(active)
            down = direction[rows, columns] >= 0.5
            partner_rows = rows + down.astype(np.int64)
            partner_columns = columns + (~down).astype(np.int64)
            source_classes = value[pair, rows, columns].astype(np.int64)
            partner_classes = value[
                pair,
                partner_rows,
                partner_columns,
            ].astype(np.int64)
            q_difference = (
                fields[
                    pair,
                    partner_rows,
                    partner_columns,
                    source_classes,
                ]
                - fields[
                    pair,
                    partner_rows,
                    partner_columns,
                    partner_classes,
                ]
            )
            clipped_tie = np.clip(tie[rows, columns], 1.0e-4, 1.0 - 1.0e-4)
            desired_difference = (
                -clipped_tie / (1.0 - clipped_tie) * q_difference
            )
            midpoint = 0.5 * (
                fields[pair, rows, columns, source_classes]
                + fields[pair, rows, columns, partner_classes]
            )
            fields[pair, rows, columns, source_classes] = (
                midpoint + 0.5 * desired_difference
            )
            fields[pair, rows, columns, partner_classes] = (
                midpoint - 0.5 * desired_difference
            )
    if not np.array_equal(np.argmax(fields, axis=-1), value):
        raise ContinuousPaintError(
            "margin-localized signed-distance fields changed the source partition"
        )
    return fields


def resample_fields_at_pixel_centres(
    fields: np.ndarray,
    *,
    output_hw: tuple[int, int] = CAMERA_HW,
) -> np.ndarray:
    """Bilinearly evaluate scorer-grid fields at output pixel centres.

    The coordinate law matches ``align_corners=False``:
    ``source=(output+0.5)*source_size/output_size-0.5`` with border clamping.
    This evaluates geometry; it does not supersample or box-average RGB.
    """

    value = np.asarray(fields)
    if value.ndim != 4 or value.shape[1:3] != SEG_HW or value.shape[-1] != N_CLASSES:
        raise ContinuousPaintError(
            f"fields must be [pairs,{SEG_HW[0]},{SEG_HW[1]},5], got {value.shape}"
        )
    output_h, output_w = output_hw
    if output_h <= 0 or output_w <= 0:
        raise ContinuousPaintError("output geometry must be positive")
    source_h, source_w = SEG_HW
    rows = (np.arange(output_h, dtype=np.float64) + 0.5) * source_h / output_h - 0.5
    columns = (np.arange(output_w, dtype=np.float64) + 0.5) * source_w / output_w - 0.5
    rr, cc = np.meshgrid(rows, columns, indexing="ij")
    coordinates = np.stack((rr, cc))
    output = np.empty(
        (value.shape[0], output_h, output_w, N_CLASSES),
        dtype=np.float32,
    )
    for pair in range(value.shape[0]):
        for class_id in range(N_CLASSES):
            output[pair, :, :, class_id] = ndimage.map_coordinates(
                value[pair, :, :, class_id],
                coordinates,
                order=1,
                mode="nearest",
                prefilter=False,
            )
    return output


def render_hard_camera_placement(
    camera_fields: np.ndarray,
    palette_rgb_u8: np.ndarray,
) -> np.ndarray:
    """Primary arm: hard full-amplitude prototypes at camera resolution."""

    fields = np.asarray(camera_fields)
    palette = _palette(palette_rgb_u8)
    if (
        fields.ndim != 4
        or fields.shape[1:3] != CAMERA_HW
        or fields.shape[-1] != N_CLASSES
    ):
        raise ContinuousPaintError(
            f"camera fields must be [pairs,{CAMERA_HW[0]},{CAMERA_HW[1]},5]"
        )
    if np.any(np.all(~np.isfinite(fields), axis=-1)):
        raise ContinuousPaintError("every camera pixel needs at least one finite class field")
    classes = np.argmax(fields, axis=-1)
    camera = palette[classes]
    if not np.all(np.any(np.all(camera[..., None, :] == palette, axis=-1), axis=-1)):
        raise ContinuousPaintError("hard placement emitted a non-prototype RGB value")
    return np.ascontiguousarray(camera, dtype=np.uint8)


def render_analytic_coverage_blend(
    camera_fields: np.ndarray,
    palette_rgb_u8: np.ndarray,
    *,
    softness: float = 1.0,
) -> np.ndarray:
    """Secondary arm: analytic SDF coverage blend, then round once to uint8.

    ``coverage_c = clip(0.5 + phi_c/softness, 0, 1)`` is normalized across
    classes.  Pixels whose truncated coverages all vanish fall back to the
    hard winning class.  No supersampling is used.
    """

    fields = np.asarray(camera_fields, dtype=np.float32)
    palette = _palette(palette_rgb_u8).astype(np.float32)
    if (
        fields.ndim != 4
        or fields.shape[1:3] != CAMERA_HW
        or fields.shape[-1] != N_CLASSES
    ):
        raise ContinuousPaintError(
            f"camera fields must be [pairs,{CAMERA_HW[0]},{CAMERA_HW[1]},5]"
        )
    if not np.isfinite(softness) or softness <= 0.0:
        raise ContinuousPaintError("analytic coverage softness must be positive")
    coverage = np.clip(0.5 + fields / float(softness), 0.0, 1.0)
    total = coverage.sum(axis=-1, keepdims=True)
    missing = total[..., 0] == 0.0
    if np.any(missing):
        hard = np.argmax(fields[missing], axis=-1)
        coverage[missing] = np.eye(N_CLASSES, dtype=np.float32)[hard]
        total = coverage.sum(axis=-1, keepdims=True)
    rgb = np.einsum("nhwk,kc->nhwc", coverage / total, palette, optimize=True)
    return np.ascontiguousarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8))


def fit_global_channel_statistics(
    source_rgb_u8: np.ndarray,
    target_rgb_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one global per-channel affine map from source to target moments."""

    source = np.asarray(source_rgb_u8)
    target = np.asarray(target_rgb_u8)
    if (
        source.shape != target.shape
        or source.ndim != 4
        or source.shape[-1] != 3
        or source.dtype != np.uint8
        or target.dtype != np.uint8
    ):
        raise ContinuousPaintError(
            "global statistics require same-shape NHWC uint8 source and target"
        )
    axes = (0, 1, 2)
    source_float = source.astype(np.float64)
    target_float = target.astype(np.float64)
    source_mean = source_float.mean(axis=axes)
    target_mean = target_float.mean(axis=axes)
    source_std = source_float.std(axis=axes)
    target_std = target_float.std(axis=axes)
    if np.any(source_std < 1.0e-9):
        raise ContinuousPaintError("source channel variance is zero")
    scale = target_std / source_std
    offset = target_mean - scale * source_mean
    return scale.astype(np.float32), offset.astype(np.float32)


def apply_global_channel_statistics(
    source_rgb_u8: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Apply a fitted global channel affine without changing spatial geometry."""

    source = np.asarray(source_rgb_u8)
    scale_value = np.asarray(scale, dtype=np.float32)
    offset_value = np.asarray(offset, dtype=np.float32)
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ContinuousPaintError("statistics source must be NHWC uint8")
    if scale_value.shape != (3,) or offset_value.shape != (3,):
        raise ContinuousPaintError("statistics scale and offset must be three-vectors")
    if not np.all(np.isfinite(scale_value)) or not np.all(np.isfinite(offset_value)):
        raise ContinuousPaintError("statistics affine must be finite")
    matched = source.astype(np.float32) * scale_value + offset_value
    return np.ascontiguousarray(np.clip(np.rint(matched), 0, 255).astype(np.uint8))


def encode_global_channel_statistics(
    scale: np.ndarray,
    offset: np.ndarray,
) -> bytes:
    """Canonical 30-byte payload: magic plus six little-endian float32 values."""

    scale_value = np.asarray(scale, dtype="<f4")
    offset_value = np.asarray(offset, dtype="<f4")
    if scale_value.shape != (3,) or offset_value.shape != (3,):
        raise ContinuousPaintError("statistics payload requires two three-vectors")
    payload = b"PT1AS1" + struct.pack("<6f", *(scale_value.tolist() + offset_value.tolist()))
    if len(payload) != 30:
        raise ContinuousPaintError("statistics payload byte accounting changed")
    decoded = np.frombuffer(payload[6:], dtype="<f4")
    if not np.array_equal(decoded[:3], scale_value) or not np.array_equal(
        decoded[3:], offset_value
    ):
        raise ContinuousPaintError("statistics payload parse-back differs")
    return payload


def stratum_spectrum_components(
    *,
    seed: int,
) -> np.ndarray:
    """Return deterministic unit-RMS passband components ``[K,C,P,H,W]``.

    The periods and features come from the existing texture-trunk bank, whose
    support is restricted to the measured through-R/stem-surviving passband.
    Seeded signs choose a reproducible mixture of orientations and phases; the
    video-derived, counted values are only the fitted per-stratum coefficients.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContinuousPaintError("spectrum seed must be an integer")
    spec = default_band_spec()
    features_per_period = len(spec.orientations_deg) * spec.n_phase
    bank = build_gabor_bank_numpy(*SEG_HW, spec).reshape(
        *SEG_HW,
        len(spec.periods),
        features_per_period,
    )
    components = np.empty(
        (N_CLASSES, 3, len(spec.periods), *SEG_HW),
        dtype=np.float32,
    )
    feature_index = np.arange(features_per_period, dtype=np.uint32)
    for class_id in range(N_CLASSES):
        for channel in range(3):
            for period_index in range(len(spec.periods)):
                key = (
                    np.uint32(seed & 0xFFFFFFFF)
                    ^ np.uint32(((class_id + 1) * 0x9E3779B1) & 0xFFFFFFFF)
                    ^ np.uint32(((channel + 1) * 0x85EBCA77) & 0xFFFFFFFF)
                    ^ np.uint32(((period_index + 1) * 0xC2B2AE3D) & 0xFFFFFFFF)
                    ^ feature_index * np.uint32(0x27D4EB2F)
                )
                signs = np.where((key & np.uint32(1)) == 0, -1.0, 1.0)
                component = np.einsum(
                    "hwf,f->hw",
                    bank[:, :, period_index, :],
                    signs,
                    optimize=True,
                ).astype(np.float64)
                component -= component.mean()
                rms = float(np.sqrt(np.mean(np.square(component))))
                if rms <= 1.0e-12:
                    raise ContinuousPaintError(
                        "spectrum component has zero energy"
                    )
                components[class_id, channel, period_index] = (
                    component / rms
                ).astype(np.float32)
    return components


def stratum_spectrum_normal_equations(
    render_labels: np.ndarray,
    target_render_rgb: np.ndarray,
    palette_rgb_u8: np.ndarray,
    *,
    seed: int,
    components: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate per-stratum least-squares equations for passband power."""

    labels = _labels(render_labels)
    target = np.asarray(target_render_rgb)
    palette = _palette(palette_rgb_u8)
    if target.shape != (*labels.shape, 3) or not np.all(np.isfinite(target)):
        raise ContinuousPaintError(
            "spectrum fitting target must be finite [pairs,384,512,3]"
        )
    component_value = (
        stratum_spectrum_components(seed=seed)
        if components is None
        else np.asarray(components, dtype=np.float32)
    )
    expected_component_shape = (
        N_CLASSES,
        3,
        len(default_band_spec().periods),
        *SEG_HW,
    )
    if component_value.shape != expected_component_shape:
        raise ContinuousPaintError("spectrum component bank geometry differs")
    period_count = component_value.shape[2]
    gram = np.zeros(
        (N_CLASSES, 3, period_count, period_count),
        dtype=np.float64,
    )
    rhs = np.zeros((N_CLASSES, 3, period_count), dtype=np.float64)
    residual = target.astype(np.float64) - palette[labels].astype(np.float64)
    for class_id in range(N_CLASSES):
        mask = labels == class_id
        if not np.any(mask):
            continue
        for channel in range(3):
            design = np.broadcast_to(
                component_value[class_id, channel].transpose(1, 2, 0),
                (*labels.shape, period_count),
            )[mask]
            response = residual[..., channel][mask]
            gram[class_id, channel] = design.T @ design
            rhs[class_id, channel] = design.T @ response
    return gram, rhs


def solve_stratum_spectrum_coefficients(
    gram: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve the global per-stratum passband coefficients deterministically."""

    spec = default_band_spec()
    period_count = len(spec.periods)
    gram_value = np.asarray(gram, dtype=np.float64)
    rhs_value = np.asarray(rhs, dtype=np.float64)
    if gram_value.shape != (N_CLASSES, 3, period_count, period_count):
        raise ContinuousPaintError("spectrum Gram geometry differs")
    if rhs_value.shape != (N_CLASSES, 3, period_count):
        raise ContinuousPaintError("spectrum RHS geometry differs")
    output = np.empty((N_CLASSES, 3, period_count), dtype=np.float32)
    for class_id in range(N_CLASSES):
        for channel in range(3):
            matrix = gram_value[class_id, channel]
            ridge = max(float(np.trace(matrix)), 1.0) * 1.0e-12
            output[class_id, channel] = np.linalg.solve(
                matrix + ridge * np.eye(period_count),
                rhs_value[class_id, channel],
            ).astype(np.float32)
    return output


def render_stratum_spectrum_match(
    render_labels: np.ndarray,
    palette_rgb_u8: np.ndarray,
    coefficients: np.ndarray,
    *,
    seed: int,
    components: np.ndarray | None = None,
) -> np.ndarray:
    """Render the deterministic per-stratum passband spectrum at scorer grid."""

    labels = _labels(render_labels)
    palette = _palette(palette_rgb_u8)
    component_value = (
        stratum_spectrum_components(seed=seed)
        if components is None
        else np.asarray(components, dtype=np.float32)
    )
    coeff = np.asarray(coefficients, dtype=np.float32)
    if coeff.shape != component_value.shape[:3]:
        raise ContinuousPaintError("spectrum coefficients differ from bank shape")
    texture_by_class = np.einsum(
        "kcp,kcphw->kchw",
        coeff,
        component_value,
        optimize=True,
    )
    output = palette[labels].astype(np.float32)
    for class_id in range(N_CLASSES):
        mask = labels == class_id
        for channel in range(3):
            output[..., channel][mask] += np.broadcast_to(
                texture_by_class[class_id, channel],
                labels.shape,
            )[mask]
    return np.ascontiguousarray(
        np.clip(np.rint(output), 0, 255).astype(np.uint8)
    )


def encode_stratum_spectrum_coefficients(coefficients: np.ndarray) -> bytes:
    """Canonical 186-byte payload: magic plus 45 little-endian float32 values."""

    spec = default_band_spec()
    value = np.asarray(coefficients, dtype="<f4")
    expected_shape = (N_CLASSES, 3, len(spec.periods))
    if value.shape != expected_shape or not np.all(np.isfinite(value)):
        raise ContinuousPaintError(
            f"spectrum coefficients must be finite {expected_shape}"
        )
    payload = b"PT1SP1" + value.tobytes(order="C")
    if len(payload) != 186:
        raise ContinuousPaintError("spectrum payload byte accounting changed")
    decoded = np.frombuffer(payload[6:], dtype="<f4").reshape(expected_shape)
    if not np.array_equal(decoded, value):
        raise ContinuousPaintError("spectrum payload parse-back differs")
    return payload


def advect_camera_texture(
    texture_rgb_u8: np.ndarray,
    flow_yx: np.ndarray,
) -> np.ndarray:
    """Deterministically advect amplitude structure by a custodied camera flow."""

    texture = np.asarray(texture_rgb_u8)
    flow = np.asarray(flow_yx, dtype=np.float32)
    if (
        texture.ndim != 4
        or texture.shape[1:3] != CAMERA_HW
        or texture.shape[-1] != 3
        or texture.dtype != np.uint8
    ):
        raise ContinuousPaintError("texture must be [pairs,874,1164,3] uint8")
    if flow.shape != (texture.shape[0], *CAMERA_HW, 2) or not np.all(
        np.isfinite(flow)
    ):
        raise ContinuousPaintError("flow_yx must be finite [pairs,874,1164,2]")
    rows, columns = np.meshgrid(
        np.arange(CAMERA_HW[0], dtype=np.float32),
        np.arange(CAMERA_HW[1], dtype=np.float32),
        indexing="ij",
    )
    output = np.empty_like(texture)
    for pair_index in range(texture.shape[0]):
        coordinates = np.stack(
            (
                rows - flow[pair_index, :, :, 0],
                columns - flow[pair_index, :, :, 1],
            )
        )
        for channel in range(3):
            output[pair_index, :, :, channel] = np.clip(
                np.rint(
                    ndimage.map_coordinates(
                        texture[pair_index, :, :, channel].astype(np.float32),
                        coordinates,
                        order=1,
                        mode="nearest",
                        prefilter=False,
                    )
                ),
                0,
                255,
            ).astype(np.uint8)
    return np.ascontiguousarray(output)


def target_boundary_band(labels: np.ndarray, *, dilation: int = 1) -> np.ndarray:
    """Return a deterministic target-label boundary band."""

    value = _labels(labels)
    if isinstance(dilation, bool) or not isinstance(dilation, int) or dilation < 0:
        raise ContinuousPaintError("boundary dilation must be a nonnegative integer")
    band = np.zeros(value.shape, dtype=bool)
    band[:, :, 1:] |= value[:, :, 1:] != value[:, :, :-1]
    band[:, :, :-1] |= value[:, :, 1:] != value[:, :, :-1]
    band[:, 1:, :] |= value[:, 1:, :] != value[:, :-1, :]
    band[:, :-1, :] |= value[:, 1:, :] != value[:, :-1, :]
    if dilation:
        structure = np.ones((1, 2 * dilation + 1, 2 * dilation + 1), dtype=bool)
        band = ndimage.binary_dilation(band, structure=structure)
    return np.ascontiguousarray(band)


def split_curve_provenance(
    *,
    target_labels: np.ndarray,
    described_curve_mask: np.ndarray,
    dilation: int = 1,
) -> dict[str, np.ndarray]:
    """Split target boundary sites into reused-description and fitted geometry.

    A target boundary site is ``already_described`` only if the decoded,
    already-counted curve mask has a boundary within the same dilation band.
    Every remaining target boundary site is charged to the encode-side fit.
    """

    target_band = target_boundary_band(target_labels, dilation=dilation)
    described = np.asarray(described_curve_mask)
    if described.shape != target_band.shape or described.dtype != np.bool_:
        raise ContinuousPaintError("described_curve_mask must be bool with target-label shape")
    described_band = np.zeros(described.shape, dtype=bool)
    described_band[:, :, 1:] |= described[:, :, 1:] != described[:, :, :-1]
    described_band[:, :, :-1] |= described[:, :, 1:] != described[:, :, :-1]
    described_band[:, 1:, :] |= described[:, 1:, :] != described[:, :-1, :]
    described_band[:, :-1, :] |= described[:, 1:, :] != described[:, :-1, :]
    if dilation:
        structure = np.ones((1, 2 * dilation + 1, 2 * dilation + 1), dtype=bool)
        described_band = ndimage.binary_dilation(described_band, structure=structure)
    reused = target_band & described_band
    fitted = target_band & ~reused
    if np.any(reused & fitted) or not np.array_equal(reused | fitted, target_band):
        raise ContinuousPaintError("curve-provenance masks do not partition target boundary")
    return {
        "already_described_curve_sites": np.ascontiguousarray(reused),
        "freshly_fitted_curve_sites": np.ascontiguousarray(fitted),
        "target_boundary_sites": target_band,
    }


def measure_fitted_geometry_sdwl1(
    labels: np.ndarray,
    margins: np.ndarray,
    gt_poses: np.ndarray,
) -> SDWL1DescriptionDebt:
    """Charge fresh target-derived geometry through complete SDWL1 parse-back."""

    inventory = extract_fact_inventory(_labels(labels), margins, gt_poses)
    measured = measure_serialization(
        inventory,
        options=SentenceOptions(
            layout=SentenceLayout.TYPED_SECTION,
            temporal_mode=TemporalMode.CAUSAL_DELTA,
        ),
    )
    if not measured.exact_parseback or measured.outer_deflate_bytes <= 0:
        raise ContinuousPaintError("fitted SDWL1 geometry lacks exact positive-byte custody")
    return SDWL1DescriptionDebt(
        bytes=measured.outer_deflate_bytes,
        sha256=measured.outer_deflate_sha256,
        exact_parseback=measured.exact_parseback,
        semantic_sha256=inventory.semantic_sha256,
        described_scalar_facts=measured.described_scalar_fact_count,
    )


def rank4_flip_distance(margins: np.ndarray, pair_norms: np.ndarray) -> np.ndarray:
    """Vectorized rank-4 head distance ``|margin|/||Delta w||``."""

    margin_values = np.asarray(margins, dtype=np.float64)
    norm_values = np.asarray(pair_norms, dtype=np.float64)
    if margin_values.shape != norm_values.shape or np.any(norm_values <= 0.0):
        raise ContinuousPaintError("margin and positive pair-normal arrays must share shape")
    flat = [
        head_flip_distance_feature_space(float(margin), float(norm))
        for margin, norm in zip(margin_values.ravel(), norm_values.ravel(), strict=True)
    ]
    return np.asarray(flat, dtype=np.float64).reshape(margin_values.shape)


def decompose_mechanisms(
    *,
    target: np.ndarray,
    baseline: np.ndarray,
    primary_hard: np.ndarray,
    statistics_control: np.ndarray,
    statistics_matched: np.ndarray,
    texture_probe: np.ndarray,
    boundary_band: np.ndarray,
) -> MechanismDecomposition:
    """Disjointly attribute observed corrections without claiming causal Shapley value."""

    arrays = [
        np.asarray(value)
        for value in (
            target,
            baseline,
            primary_hard,
            statistics_control,
            statistics_matched,
            texture_probe,
        )
    ]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ContinuousPaintError("all mechanism argmax arrays must share shape")
    band = np.asarray(boundary_band)
    if band.shape != arrays[0].shape or band.dtype != np.bool_:
        raise ContinuousPaintError("boundary_band must be bool with argmax-array shape")
    wrong = arrays[1] != arrays[0]
    primary_correct = arrays[2] == arrays[0]
    statistics_control_wrong = arrays[3] != arrays[0]
    statistics_correct = arrays[4] == arrays[0]
    texture_correct = arrays[5] == arrays[0]
    primary_changed = arrays[2] != arrays[1]
    sub_cell = wrong & primary_correct & band
    class_interaction = wrong & primary_correct & ~band
    amplitude = (
        wrong
        & ~primary_correct
        & statistics_control_wrong
        & statistics_correct
    )
    texture = wrong & ~primary_correct & ~statistics_correct & texture_correct
    if np.any(sub_cell & class_interaction) or np.any(
        (sub_cell | class_interaction) & (amplitude | texture)
    ) or np.any(
        amplitude & texture
    ):
        raise ContinuousPaintError("mechanism attribution is not disjoint")
    placement_attributable = wrong & band & primary_changed
    attributable_count = int(np.count_nonzero(placement_attributable))
    recovered_count = int(np.count_nonzero(placement_attributable & primary_correct))
    wall = (
        1.0 - recovered_count / attributable_count
        if attributable_count
        else 1.0
    )
    return MechanismDecomposition(
        baseline_errors=int(np.count_nonzero(wrong)),
        corrected_total_primary=int(np.count_nonzero(wrong & primary_correct)),
        corrected_total_statistics_only=int(np.count_nonzero(amplitude)),
        corrected_total_secondary_only=int(np.count_nonzero(texture)),
        sub_cell_placement=int(np.count_nonzero(sub_cell)),
        bn_se_amplitude_statistics=int(np.count_nonzero(amplitude)),
        texture_prior_or_region_erf=int(np.count_nonzero(texture)),
        class_interaction=int(np.count_nonzero(class_interaction)),
        residual_errors_primary=int(np.count_nonzero(arrays[2] != arrays[0])),
        placement_attributable=attributable_count,
        placement_recovered=recovered_count,
        measured_survival_wall_fraction=wall,
    )


def scorer_native_divergence_rows(
    *,
    candidate: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    margins: np.ndarray,
) -> list[dict[str, object]]:
    """Emit per-layer static and temporal scorer-native divergence rows.

    Activations are expected in NCHW or ND form.  Four-dimensional rows use
    categorical-Fisher weights ``0.5*sech(margin/2)^2`` resampled to the
    layer's spatial grid.  Temporal stability is the relative error between
    candidate and target first differences along the ordered pair axis.
    """

    if set(candidate) != set(target) or not candidate:
        raise ContinuousPaintError("candidate/target activation layers must match")
    margin_values = np.asarray(margins, dtype=np.float64)
    if margin_values.ndim != 3 or margin_values.shape[1:] != SEG_HW:
        raise ContinuousPaintError("scorer-native margins must be [pairs,384,512]")
    rows: list[dict[str, object]] = []
    for layer in candidate:
        cand = np.asarray(candidate[layer], dtype=np.float64)
        truth = np.asarray(target[layer], dtype=np.float64)
        if cand.shape != truth.shape or cand.shape[0] != margin_values.shape[0]:
            raise ContinuousPaintError(f"activation custody differs at layer {layer}")
        if not np.all(np.isfinite(cand)) or not np.all(np.isfinite(truth)):
            raise ContinuousPaintError(
                f"activation profile is non-finite at layer {layer}"
            )
        delta = cand - truth
        delta_norm = float(np.linalg.norm(delta.ravel()))
        target_norm = float(np.linalg.norm(truth.ravel()))
        relative = delta_norm / max(target_norm, 1.0e-12)
        fisher_weighted: float | None = None
        spatial_band = "global"
        channel_group = "all"
        if cand.ndim == 4:
            height, width = cand.shape[-2:]
            zoom = (1.0, height / SEG_HW[0], width / SEG_HW[1])
            resized_margin = ndimage.zoom(
                margin_values,
                zoom,
                order=1,
                mode="nearest",
                prefilter=False,
            )
            resized_margin = resized_margin[:, :height, :width]
            fisher = 0.5 / np.cosh(np.clip(resized_margin / 2.0, -40.0, 40.0)) ** 2
            fisher_weighted = float(
                np.sqrt(
                    np.mean(
                        np.square(delta)
                        * fisher[:, None, :, :]
                    )
                )
            )
            spatial_band = f"{height}x{width}"
            channel_group = f"all_{cand.shape[1]}ch"
        if cand.shape[0] >= 2:
            candidate_trajectory = np.diff(cand, axis=0)
            target_trajectory = np.diff(truth, axis=0)
            trajectory_delta = candidate_trajectory - target_trajectory
            trajectory_relative = float(
                np.linalg.norm(trajectory_delta.ravel())
                / max(np.linalg.norm(target_trajectory.ravel()), 1.0e-12)
            )
        else:
            trajectory_relative = None
        rows.append(
            {
                "channel_group": channel_group,
                "delta_norm": delta_norm,
                "delta_norm_relative": relative,
                "fisher_weighted_delta": fisher_weighted,
                "layer": layer,
                "spatial_band": spatial_band,
                "trajectory_delta_norm_relative": trajectory_relative,
            }
        )
    return rows


def stage_transition(
    *,
    before: np.ndarray,
    after: np.ndarray,
    target: np.ndarray,
    owner_mask: np.ndarray | None = None,
) -> dict[str, int]:
    """Exact ``after=before+introduced-corrected`` argmax accounting."""

    before_value = np.asarray(before)
    after_value = np.asarray(after)
    target_value = np.asarray(target)
    if before_value.shape != after_value.shape or before_value.shape != target_value.shape:
        raise ContinuousPaintError("stage transition arrays must share shape")
    owner = (
        np.ones(before_value.shape, dtype=bool)
        if owner_mask is None
        else np.asarray(owner_mask)
    )
    if owner.shape != before_value.shape or owner.dtype != np.bool_:
        raise ContinuousPaintError("owner_mask must be bool with transition-array shape")
    before_wrong = (before_value != target_value) & owner
    after_wrong = (after_value != target_value) & owner
    introduced = ~before_wrong & after_wrong & owner
    corrected = before_wrong & ~after_wrong & owner
    row = {
        "argmax_diff_from_previous": int(
            np.count_nonzero((before_value != after_value) & owner)
        ),
        "errors_after": int(np.count_nonzero(after_wrong)),
        "errors_before": int(np.count_nonzero(before_wrong)),
        "errors_corrected": int(np.count_nonzero(corrected)),
        "errors_introduced": int(np.count_nonzero(introduced)),
        "errors_persisting": int(np.count_nonzero(before_wrong & after_wrong)),
        "owner_sites": int(np.count_nonzero(owner)),
    }
    if (
        row["errors_after"]
        != row["errors_before"] + row["errors_introduced"] - row["errors_corrected"]
    ):
        raise ContinuousPaintError("stage error-flow conservation failed")
    return row


def sha256_array(value: np.ndarray) -> str:
    """Content hash including dtype and shape, for immutable batch custody."""

    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{array.shape}".encode("ascii")
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


__all__ = [
    "CLASS_ORDER",
    "MECHANISM_ORDER",
    "N_CLASSES",
    "ContinuousPaintError",
    "MechanismDecomposition",
    "SDWL1DescriptionDebt",
    "advect_camera_texture",
    "apply_global_channel_statistics",
    "decompose_mechanisms",
    "encode_global_channel_statistics",
    "encode_stratum_spectrum_coefficients",
    "fit_global_channel_statistics",
    "measure_fitted_geometry_sdwl1",
    "rank4_flip_distance",
    "render_analytic_coverage_blend",
    "render_hard_camera_placement",
    "render_stratum_spectrum_match",
    "resample_fields_at_pixel_centres",
    "scorer_native_divergence_rows",
    "sha256_array",
    "signed_distance_fields",
    "solve_stratum_spectrum_coefficients",
    "split_curve_provenance",
    "stage_transition",
    "stratum_spectrum_components",
    "stratum_spectrum_normal_equations",
    "target_boundary_band",
]
