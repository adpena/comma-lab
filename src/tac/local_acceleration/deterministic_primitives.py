# SPDX-License-Identifier: MIT
"""Canonical drift-characterization primitives for MLX/PyTorch parity engineering.

PR95-MLX-PYTORCH-DRIFT-MITIGATION-ENGINEERING 2026-05-25 self-protection per
operator critical insights *"if the MLX numpy weights are determined to be
portable or portable with acceptable drift that must be taken into consideration
as well"* + *"continue engineering and fixing drift away"*.

## Empirical drift characterization (anchor 2026-05-25)

Per-op isolated drift at random-init float32 on Apple Silicon (MLX CPU vs
PyTorch CPU on macOS) measured by
``tools/measure_pr95_mlx_pytorch_per_op_drift.py``:

| Operation                  | max_abs   | mean_abs  | Classification             |
| -------------------------- | --------- | --------- | -------------------------- |
| bilinear_resize_2x         | 2.38e-07  | 1.83e-08  | BYTE_STABLE_BY_DEFAULT     |
| sin                        | 5.96e-08  | 5.13e-09  | BYTE_STABLE_BY_DEFAULT     |
| sigmoid                    | 1.19e-07  | 1.24e-08  | BYTE_STABLE_BY_DEFAULT     |
| pixel_shuffle_2x           | 0.00e+00  | 0.00e+00  | BYTE_STABLE_BY_DEFAULT     |
| Linear (28 -> 1728)        | 0.00e+00  | 0.00e+00  | BYTE_STABLE_BY_DEFAULT     |
| Conv2d 3x3 (36 -> 144)     | 1.43e-06  | 1.38e-07  | NUMERIC_TOLERANCE_INHERENT |
| HNeRVDecoder (full, 6-stg) | 3.05e-05  | 4.80e-06  | NUMERIC_TOLERANCE_INHERENT |

The full-decoder drift matches the Slot 1 trained-checkpoint anchor exactly
(3.05e-05 max_abs at random init and at the trained Stage 8 checkpoint, with
small mean_abs variation). The Stage 8 decoder-boundary trace localizes the
visible cliff to the final RGB heads: pre-head features remain around 1e-7,
then ``rgb_0``/``rgb_1`` produce the 3.05e-05 output delta. That makes the
mechanism framework-arithmetic drift in conv/head execution plus sigmoid and
255 scaling, NOT sin/bilinear/pixel-shuffle substitution.

## Engineering verdict: naive substitution is refuted

Initial Slot 4 hypothesis: substituting bilinear_resize + sin + sigmoid with
deterministic numpy primitives would reduce aggregate drift by >50%.

Empirical refutation:
- bilinear_resize_2x is ALREADY BYTE_STABLE_BY_DEFAULT (max_abs=2.4e-7)
- sin is ALREADY BYTE_STABLE_BY_DEFAULT (max_abs=6e-8)
- sigmoid is ALREADY BYTE_STABLE_BY_DEFAULT (max_abs=1.2e-7)
- pixel_shuffle_2x is FULLY DETERMINISTIC (max_abs=0.0; reshape+transpose)

The measured single Conv2d drift is ~1.4e-6 in this tiny probe, while the
trained decoder trace keeps the stem, upsample blocks, refine residual, and
feature tensor below ~2.8e-7. The first >1e-5 boundary is the RGB head output,
where final head arithmetic is passed through sigmoid and scaled by 255.

Substituting Conv2d with naive numpy einsum *also* produces ~1.8e-6 drift vs
PyTorch in the measured probe. We therefore do not claim a portable
byte-identical Conv2d primitive yet; the production contract is an attested
tolerance with exact-auth escalation, not a silent coercion.

## Canonical engineering primitive: ATTESTED-TOLERANCE PORTABILITY

Per CLAUDE.md "Beauty, simplicity, and developer experience" + "Mission alignment"
operational consequence 4 (frontier-breaking moves DOMINATE rigor budget): rather
than chase drift elimination, this module CANONICALIZES the drift property as a
PORTABILITY ATTESTATION the operator can rely on:

- ``classify_operation_drift(op_name)`` -> ``DriftClassification`` enum
- ``validate_mlx_pytorch_parity_within_tolerance(...)`` runtime helper with
  canonical (max_abs, mean_abs) attested bands per operation class
- ``portability_attestation_for_state_dict(state_dict, expected_class)`` per-tensor
  attestation surface for ``mlx_to_pytorch_export.py`` consumers
- ``canonical_drift_bands_for_pr95_hnerv_decoder()`` returns the empirical
  reference bands from this module's anchor measurement, queryable from
  ``tools/export_pr95_mlx_to_pytorch_state_dict.py`` VERDICT upgrade

Per Catalog #344: the canonical equation
``mlx_numpy_weights_portability_to_pytorch_with_drift_class_v1`` is
FORMALIZATION_PENDING; this module is the implementation surface that will
feed the equation's empirical anchors via
``tac.canonical_equations.update_equation_with_empirical_anchor``.

## Sister discipline cross-references

- CLAUDE.md "MPS auth eval is NOISE" - MLX is NOT MPS; MLX is the Apple Silicon
  native framework with deterministic-with-tolerance arithmetic vs PyTorch MPS
  which has 23x drift on PoseNet. The MLX/PyTorch drift of 3.05e-5 at full
  decoder is 6 orders of magnitude smaller than MPS/CUDA drift on PoseNet (0.245
  vs 0.0107 = 23x). MLX is a **tractable** parity story; MPS is NOT.
- CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
  HARDWARE" non-negotiable: this module does NOT change the
  ``evidence_grade="MLX-research-signal"`` tag; the drift attestation is a
  PORTABILITY claim, not a contest-axis promotion. Exact CUDA T4 + Linux x86_64
  CPU eval is still required for score, frontier, promotion, or submission.
- Catalog #1 (no MPS fallback default) + Catalog #192 (macOS-CPU advisory not
  promoted) + Catalog #317 (one-arg local dispatch evidence-grade stamping):
  every primitive in this module preserves non-promotable markers and emits
  ``score_claim=False`` / ``promotion_eligible=False`` / ``axis_tag=[predicted]``
  in every return-value attestation.
- Catalog #205 (canonical select_inflate_device): per the operator's "fix all
  warnings and issues and bugs" non-negotiable, this module preserves
  device-deterministic semantics - MLX device selection (cpu vs gpu) is
  honored via ``mx.default_device()`` snapshot/restore pattern mirroring the
  existing ``compare_pr95_public_archive_forward_with_pytorch`` helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

__all__ = [
    "ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR",
    "PR95_HNERV_DECODER_CANONICAL_DRIFT_BANDS",
    "PR95_HNERV_DECODER_PER_OP_DRIFT_ANCHORS",
    "ActiveExplorationPathVerdict",
    "ActiveExplorationThreadResult",
    "CudnnReferenceMeasurementResult",
    "DriftAttestation",
    "DriftClassification",
    "MlxDeterministicInvestigationResult",
    "canonical_drift_bands_for_pr95_hnerv_decoder",
    "classify_operation_drift",
    "classify_reduction_percent",
    "fp64_intermediate_conv2d_3x3",
    "kahan_compensated_sum",
    "kahan_conv2d_3x3",
    "portability_attestation_for_state_dict",
    "validate_mlx_pytorch_parity_within_tolerance",
]


class DriftClassification(StrEnum):
    """Per-operation drift classification per the empirical anchor 2026-05-25.

    Three structural classes (Catalog #344 canonical taxonomy):

- BYTE_STABLE_BY_DEFAULT: max_abs <= 1e-6, mean_abs <= 1e-7. The operation
      is byte-stable across frameworks in float32 at the measured scale.
      Substitution provides ZERO drift reduction.

- NUMERIC_TOLERANCE_INHERENT: max_abs <= 1e-4, mean_abs <= 1e-5. Drift is
      intrinsic framework arithmetic (BLAS accumulator order; intrinsic SIMD
      vectorization differences). PORTABLE within attested tolerance;
      substitution to numpy reference does NOT close the gap because numpy
      itself differs from optimized BLAS.

- FRAMEWORK_DIFFERENT: max_abs > 1e-4 OR mean_abs > 1e-5. Drift exceeds
      attested tolerance band - substrate-class behavior differs materially
      (e.g., different activation function semantics). Requires per-operation
      design-time fix before parity claim is valid.
    """

    BYTE_STABLE_BY_DEFAULT = "byte_stable_by_default"
    NUMERIC_TOLERANCE_INHERENT = "numeric_tolerance_inherent"
    FRAMEWORK_DIFFERENT = "framework_different"


@dataclass(frozen=True)
class DriftAttestation:
    """Canonical attestation for a per-operation or aggregate drift measurement.

    Per CLAUDE.md "Apples-to-apples evidence discipline": every drift claim
    carries (operation_name, measured_max_abs, measured_mean_abs, expected_class,
    actual_class, attested_band, attested_within_band) so the operator-facing
    audit surface can decide whether the drift is structurally acceptable
    or requires engineering intervention.

    Per Catalog #287/#323 canonical Provenance: every attestation defaults
    score_claim=False + promotable=False + axis_tag="[predicted]" because
    drift attestations are LOCAL-PROBE evidence, not contest-axis claims.
    """

    operation_name: str
    measured_max_abs: float
    measured_mean_abs: float
    expected_class: DriftClassification
    actual_class: DriftClassification
    attested_max_abs_band: float
    attested_mean_abs_band: float
    attested_within_band: bool
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "mlx_pytorch_drift_attestation.v1",
            "operation_name": self.operation_name,
            "measured_max_abs": self.measured_max_abs,
            "measured_mean_abs": self.measured_mean_abs,
            "expected_class": self.expected_class.value,
            "actual_class": self.actual_class.value,
            "attested_max_abs_band": self.attested_max_abs_band,
            "attested_mean_abs_band": self.attested_mean_abs_band,
            "attested_within_band": self.attested_within_band,
            "notes": self.notes,
            # Canonical Provenance markers per Catalog #287/#323/#192/#1.
            "evidence_grade": EVIDENCE_GRADE_MLX,
            "evidence_tag": EVIDENCE_TAG_MLX,
            "axis_tag": "[predicted]",
            "score_claim": False,
            "promotion_eligible": False,
            "promotable": False,
            "ready_for_exact_eval_dispatch": False,
            "blockers": [
                "local_mlx_pytorch_drift_attestation_is_not_contest_auth_eval",
                "requires_paired_cuda_t4_or_linux_x86_64_eval_for_promotion",
            ],
        }


# Canonical drift bands per the BYTE_STABLE / NUMERIC_TOLERANCE / FRAMEWORK_DIFFERENT
# classification thresholds. The bands are inclusive upper bounds - a measurement
# at exactly the band edge is INSIDE the band.
_BYTE_STABLE_MAX_ABS = 1e-6
_BYTE_STABLE_MEAN_ABS = 1e-7
_NUMERIC_TOLERANCE_MAX_ABS = 1e-4
_NUMERIC_TOLERANCE_MEAN_ABS = 1e-5


# Per-operation empirical anchors measured 2026-05-25 (Apple Silicon M5 Max,
# float32, MLX CPU device vs PyTorch CPU on macOS). Each entry maps
# to (canonical_class, attested_max_abs_band, attested_mean_abs_band,
# anchor_max_abs, anchor_mean_abs). The attested bands are the CLASS bounds;
# the anchor values are the actual empirical measurement that justifies the
# class assignment.
PR95_HNERV_DECODER_PER_OP_DRIFT_ANCHORS: dict[str, dict[str, Any]] = {
    "bilinear_resize_2x_align_corners_false_nhwc": {
        "canonical_class": DriftClassification.BYTE_STABLE_BY_DEFAULT,
        "attested_max_abs_band": _BYTE_STABLE_MAX_ABS,
        "attested_mean_abs_band": _BYTE_STABLE_MEAN_ABS,
        "anchor_max_abs": 2.3842e-07,
        "anchor_mean_abs": 1.8309e-08,
        "anchor_input_shape_nchw": (1, 36, 6, 8),
        "anchor_output_shape_nchw": (1, 36, 12, 16),
        "anchor_dtype": "float32",
        "notes": "MLX manual NHWC reshape/concat path matches PyTorch F.interpolate(mode='bilinear', align_corners=False) byte-stably at 2.4e-7.",
    },
    "sin": {
        "canonical_class": DriftClassification.BYTE_STABLE_BY_DEFAULT,
        "attested_max_abs_band": _BYTE_STABLE_MAX_ABS,
        "attested_mean_abs_band": _BYTE_STABLE_MEAN_ABS,
        "anchor_max_abs": 5.9605e-08,
        "anchor_mean_abs": 5.1331e-09,
        "anchor_input_shape_nchw": (1, 36, 12, 16),
        "anchor_dtype": "float32",
        "notes": "MLX mx.sin matches PyTorch torch.sin byte-stably at 6e-8.",
    },
    "sigmoid": {
        "canonical_class": DriftClassification.BYTE_STABLE_BY_DEFAULT,
        "attested_max_abs_band": _BYTE_STABLE_MAX_ABS,
        "attested_mean_abs_band": _BYTE_STABLE_MEAN_ABS,
        "anchor_max_abs": 1.1921e-07,
        "anchor_mean_abs": 1.2397e-08,
        "anchor_input_shape_nchw": (1, 36, 12, 16),
        "anchor_dtype": "float32",
        "notes": "MLX mx.sigmoid matches PyTorch torch.sigmoid byte-stably at 1.2e-7.",
    },
    "pixel_shuffle_2x_nhwc": {
        "canonical_class": DriftClassification.BYTE_STABLE_BY_DEFAULT,
        "attested_max_abs_band": _BYTE_STABLE_MAX_ABS,
        "attested_mean_abs_band": _BYTE_STABLE_MEAN_ABS,
        "anchor_max_abs": 0.0,
        "anchor_mean_abs": 0.0,
        "anchor_input_shape_nchw": (1, 144, 6, 8),
        "anchor_output_shape_nchw": (1, 36, 12, 16),
        "anchor_dtype": "float32",
        "notes": "MLX NHWC reshape/transpose pipeline is fully deterministic vs PyTorch F.pixel_shuffle.",
    },
    "linear_stem": {
        "canonical_class": DriftClassification.BYTE_STABLE_BY_DEFAULT,
        "attested_max_abs_band": _BYTE_STABLE_MAX_ABS,
        "attested_mean_abs_band": _BYTE_STABLE_MEAN_ABS,
        "anchor_max_abs": 0.0,
        "anchor_mean_abs": 0.0,
        "anchor_input_shape": (1, 28),
        "anchor_output_shape": (1, 1728),
        "anchor_dtype": "float32",
        "notes": "MLX nn.Linear matches PyTorch nn.Linear at this size byte-stably (small matmul fits in single SIMD register).",
    },
    "conv2d_3x3_pad1": {
        "canonical_class": DriftClassification.NUMERIC_TOLERANCE_INHERENT,
        "attested_max_abs_band": _NUMERIC_TOLERANCE_MAX_ABS,
        "attested_mean_abs_band": _NUMERIC_TOLERANCE_MEAN_ABS,
        "anchor_max_abs": 1.4305e-06,
        "anchor_mean_abs": 1.3792e-07,
        "anchor_input_shape_nchw": (1, 36, 6, 8),
        "anchor_output_shape_nchw": (1, 144, 6, 8),
        "anchor_dtype": "float32",
        "notes": "MLX Conv2d and PyTorch Conv2d follow different optimized float32 accumulation paths on macOS. A naive numpy einsum reference also drifts from PyTorch at this scale. PORTABLE within attested tolerance.",
    },
    "hnerv_decoder_full": {
        "canonical_class": DriftClassification.NUMERIC_TOLERANCE_INHERENT,
        "attested_max_abs_band": _NUMERIC_TOLERANCE_MAX_ABS,
        "attested_mean_abs_band": _NUMERIC_TOLERANCE_MEAN_ABS,
        "anchor_max_abs": 3.0518e-05,
        "anchor_mean_abs": 4.7964e-06,
        "anchor_input_shape": (2, 28),
        "anchor_output_shape_n2chw": (2, 2, 3, 384, 512),
        "anchor_dtype": "float32",
        "notes": "Full PR95 HNeRV decoder forward. Stage 8 trace keeps pre-head features near 1e-7; the visible 3.05e-05 cliff appears at rgb_0/rgb_1 after final head conv, sigmoid, and 255 scaling. Empirically matches the trained-checkpoint max_abs anchor.",
    },
}


# Canonical aggregate bands for the full PR95 HNeRV decoder forward pass.
# Returned by ``canonical_drift_bands_for_pr95_hnerv_decoder()`` for
# operator-facing consumers (Slot 1 export bridge VERDICT upgrade).
PR95_HNERV_DECODER_CANONICAL_DRIFT_BANDS: dict[str, Any] = {
    "schema": "pr95_hnerv_decoder_canonical_drift_bands.v1",
    "anchor_utc": "2026-05-25T17:43:00Z",
    "anchor_substrate": "PR95 HNeRV decoder (latent_dim=28, base_channels=36, eval_size=384x512)",
    "framework_pair": "MLX CPU vs PyTorch CPU on macOS",
    "hardware": "Apple Silicon M5 Max (macOS)",
    "attested_max_abs": _NUMERIC_TOLERANCE_MAX_ABS,
    "attested_mean_abs": _NUMERIC_TOLERANCE_MEAN_ABS,
    "canonical_class": DriftClassification.NUMERIC_TOLERANCE_INHERENT.value,
    "anchor_max_abs": 3.0518e-05,
    "anchor_mean_abs": 4.7964e-06,
    "rtol_recommendation_for_export_bridge_verdict_upgrade": 1e-4,
    "atol_recommendation_for_export_bridge_verdict_upgrade": 1e-4,
    "verdict_classification": "PORTABLE_WITH_ATTESTED_TOLERANCE",
    "operator_routable_summary": (
        "Slot 1 export bridge VERDICT can upgrade from NUMERIC_TOLERANCE_RTOL_1e-2 "
        "to PORTABLE_WITH_ATTESTED_TOLERANCE_RTOL_1e-4. Drift is BOUNDED and "
        "REPRODUCIBLE; engineering substitution of bilinear/sin/sigmoid yields ZERO "
        "reduction (those ops are already BYTE_STABLE_BY_DEFAULT at 2.4e-7 / 6e-8 / "
        "1.2e-7). Stage 8 traces localize the visible cliff to the final RGB heads, "
        "so PORTABILITY via attested tolerance plus exact-auth escalation is the "
        "canonical engineering primitive."
    ),
}


def classify_operation_drift(
    measured_max_abs: float,
    measured_mean_abs: float,
) -> DriftClassification:
    """Classify a measured drift pair into BYTE_STABLE / NUMERIC_TOLERANCE / FRAMEWORK_DIFFERENT.

    Per the empirical 2026-05-25 thresholds:
    - BYTE_STABLE_BY_DEFAULT: max_abs <= 1e-6 AND mean_abs <= 1e-7
    - NUMERIC_TOLERANCE_INHERENT: max_abs <= 1e-4 AND mean_abs <= 1e-5
    - FRAMEWORK_DIFFERENT: anything outside

    Raises:
        ValueError: if either measured value is negative or non-finite.
    """
    if not np.isfinite(measured_max_abs) or measured_max_abs < 0:
        raise ValueError(f"measured_max_abs must be non-negative finite; got {measured_max_abs!r}")
    if not np.isfinite(measured_mean_abs) or measured_mean_abs < 0:
        raise ValueError(f"measured_mean_abs must be non-negative finite; got {measured_mean_abs!r}")

    if (
        measured_max_abs <= _BYTE_STABLE_MAX_ABS
        and measured_mean_abs <= _BYTE_STABLE_MEAN_ABS
    ):
        return DriftClassification.BYTE_STABLE_BY_DEFAULT
    if (
        measured_max_abs <= _NUMERIC_TOLERANCE_MAX_ABS
        and measured_mean_abs <= _NUMERIC_TOLERANCE_MEAN_ABS
    ):
        return DriftClassification.NUMERIC_TOLERANCE_INHERENT
    return DriftClassification.FRAMEWORK_DIFFERENT


def canonical_drift_bands_for_pr95_hnerv_decoder() -> dict[str, Any]:
    """Return the canonical aggregate drift bands for the PR95 HNeRV decoder.

    Operator-facing surface for the Slot 1 export bridge VERDICT upgrade:
    ``tools/export_pr95_mlx_to_pytorch_state_dict.py`` can consume the returned
    dict's ``rtol_recommendation_for_export_bridge_verdict_upgrade`` field as
    the canonical tolerance for the parity assertion.
    """
    return dict(PR95_HNERV_DECODER_CANONICAL_DRIFT_BANDS)


def validate_mlx_pytorch_parity_within_tolerance(
    *,
    operation_name: str,
    mlx_output: np.ndarray,
    pytorch_output: np.ndarray,
    expected_class: DriftClassification | None = None,
) -> DriftAttestation:
    """Validate MLX vs PyTorch outputs are within their canonical drift tolerance.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323
    canonical Provenance: the returned attestation carries the full evidence
    triple (operation_name, measured drift, expected vs actual class).

    Args:
        operation_name: canonical operation key (e.g. ``"conv2d_3x3_pad1"``,
            ``"hnerv_decoder_full"``). Used to look up the canonical attested
            band from ``PR95_HNERV_DECODER_PER_OP_DRIFT_ANCHORS``.
        mlx_output: MLX output as numpy array.
        pytorch_output: PyTorch output as numpy array (matching shape).
        expected_class: optional override. If None, derived from the canonical
            anchors dict; if the operation_name is not anchored, the
            band defaults to NUMERIC_TOLERANCE_INHERENT (the safer band).

    Returns:
        DriftAttestation with ``attested_within_band=True`` iff measured
        drift is within the (max_abs_band, mean_abs_band) for the expected
        class.

    Raises:
        ValueError: if outputs have mismatched shape or NaN/inf.
    """
    if mlx_output.shape != pytorch_output.shape:
        raise ValueError(
            f"shape mismatch: mlx_output={mlx_output.shape} vs pytorch_output={pytorch_output.shape}"
        )
    if not np.all(np.isfinite(mlx_output)):
        raise ValueError(f"mlx_output for {operation_name!r} contains NaN/inf")
    if not np.all(np.isfinite(pytorch_output)):
        raise ValueError(f"pytorch_output for {operation_name!r} contains NaN/inf")

    diff = np.abs(mlx_output - pytorch_output)
    measured_max_abs = float(diff.max()) if diff.size else 0.0
    measured_mean_abs = float(diff.mean()) if diff.size else 0.0
    actual_class = classify_operation_drift(measured_max_abs, measured_mean_abs)

    anchor = PR95_HNERV_DECODER_PER_OP_DRIFT_ANCHORS.get(operation_name)
    if expected_class is None:
        if anchor is not None:
            expected_class = anchor["canonical_class"]
            attested_max_abs_band = float(anchor["attested_max_abs_band"])
            attested_mean_abs_band = float(anchor["attested_mean_abs_band"])
        else:
            # Default to NUMERIC_TOLERANCE for unknown operations (safer).
            expected_class = DriftClassification.NUMERIC_TOLERANCE_INHERENT
            attested_max_abs_band = _NUMERIC_TOLERANCE_MAX_ABS
            attested_mean_abs_band = _NUMERIC_TOLERANCE_MEAN_ABS
    else:
        if expected_class == DriftClassification.BYTE_STABLE_BY_DEFAULT:
            attested_max_abs_band = _BYTE_STABLE_MAX_ABS
            attested_mean_abs_band = _BYTE_STABLE_MEAN_ABS
        elif expected_class == DriftClassification.NUMERIC_TOLERANCE_INHERENT:
            attested_max_abs_band = _NUMERIC_TOLERANCE_MAX_ABS
            attested_mean_abs_band = _NUMERIC_TOLERANCE_MEAN_ABS
        else:
            # FRAMEWORK_DIFFERENT is a refusal class, not a wider tolerance
            # band. Accepting it here would let arbitrary parity drift become
            # an attested success.
            attested_max_abs_band = 0.0
            attested_mean_abs_band = 0.0

    attested_within_band = (
        expected_class != DriftClassification.FRAMEWORK_DIFFERENT
        and measured_max_abs <= attested_max_abs_band
        and measured_mean_abs <= attested_mean_abs_band
    )

    notes = ""
    if anchor is not None:
        anchor_max = float(anchor.get("anchor_max_abs", 0.0))
        anchor_mean = float(anchor.get("anchor_mean_abs", 0.0))
        notes = (
            f"canonical anchor max_abs={anchor_max:.4e} mean_abs={anchor_mean:.4e}; "
            f"measured max_abs={measured_max_abs:.4e} mean_abs={measured_mean_abs:.4e}"
        )

    return DriftAttestation(
        operation_name=operation_name,
        measured_max_abs=measured_max_abs,
        measured_mean_abs=measured_mean_abs,
        expected_class=expected_class,
        actual_class=actual_class,
        attested_max_abs_band=attested_max_abs_band,
        attested_mean_abs_band=attested_mean_abs_band,
        attested_within_band=attested_within_band,
        notes=notes,
    )


def portability_attestation_for_state_dict(
    *,
    mlx_state_dict_np: dict[str, np.ndarray],
    pytorch_state_dict_np: dict[str, np.ndarray],
    substrate_id: str = "pr95_hnerv_decoder",
) -> dict[str, Any]:
    """Per-tensor MLX-numpy-weights portability attestation for export bridge consumers.

    Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable:
    *"every MLX-derived row must carry explicit false authority"*. This helper
    builds the canonical per-tensor + aggregate attestation that
    ``mlx_to_pytorch_export.py`` consumers can route through for the operator-
    facing portability claim.

    The state_dict-level attestation answers: are the MLX numpy weights
    byte-stable round-trip via the export bridge? (Answer: YES; state_dict
    bridge IS byte-stable per Slot 1; the drift surfaces only at forward-pass
    execution.)

    Args:
        mlx_state_dict_np: dict mapping parameter name -> numpy array (MLX layout,
            already converted via ``pytorch_state_dict_from_mlx``).
        pytorch_state_dict_np: dict mapping parameter name -> numpy array
            (PyTorch reference layout; typically the original state_dict before
            MLX load).
        substrate_id: canonical substrate identifier for the attestation.

    Returns:
        Canonical attestation dict with per-tensor + aggregate drift, ready for
        ``json.dump`` to ``.omx/state/`` via canonical fcntl-locked writers.
    """
    if not mlx_state_dict_np:
        raise ValueError("mlx_state_dict_np must contain at least one parameter")

    mlx_keys = set(mlx_state_dict_np.keys())
    pytorch_keys = set(pytorch_state_dict_np.keys())
    if mlx_keys != pytorch_keys:
        only_mlx = sorted(mlx_keys - pytorch_keys)
        only_pytorch = sorted(pytorch_keys - mlx_keys)
        raise KeyError(
            f"state_dict key mismatch: only in MLX={only_mlx[:5]}; "
            f"only in PyTorch={only_pytorch[:5]}"
        )

    per_tensor: dict[str, dict[str, Any]] = {}
    aggregate_max_abs = 0.0
    aggregate_sum_abs = 0.0
    aggregate_count = 0
    all_byte_stable = True

    for name in sorted(mlx_keys):
        mlx_arr = mlx_state_dict_np[name]
        pytorch_arr = pytorch_state_dict_np[name]
        if mlx_arr.shape != pytorch_arr.shape:
            raise ValueError(
                f"shape mismatch for {name!r}: MLX={mlx_arr.shape} vs PyTorch={pytorch_arr.shape}"
            )
        diff = np.abs(mlx_arr.astype(np.float32) - pytorch_arr.astype(np.float32))
        max_abs = float(diff.max()) if diff.size else 0.0
        mean_abs = float(diff.mean()) if diff.size else 0.0
        cls = classify_operation_drift(max_abs, mean_abs)
        per_tensor[name] = {
            "shape": [int(x) for x in mlx_arr.shape],
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "classification": cls.value,
            "byte_stable": cls == DriftClassification.BYTE_STABLE_BY_DEFAULT,
        }
        aggregate_max_abs = max(aggregate_max_abs, max_abs)
        aggregate_sum_abs += float(diff.sum()) if diff.size else 0.0
        aggregate_count += int(diff.size)
        if cls != DriftClassification.BYTE_STABLE_BY_DEFAULT:
            all_byte_stable = False

    aggregate_mean_abs = (
        aggregate_sum_abs / aggregate_count if aggregate_count > 0 else 0.0
    )

    return {
        "schema": "mlx_pytorch_state_dict_portability_attestation.v1",
        "substrate_id": substrate_id,
        "tensor_count": len(mlx_keys),
        "per_tensor": per_tensor,
        "aggregate_max_abs": aggregate_max_abs,
        "aggregate_mean_abs": aggregate_mean_abs,
        "all_tensors_byte_stable": all_byte_stable,
        "verdict": (
            "BYTE_STABLE_PER_TENSOR_STATE_DICT_BRIDGE"
            if all_byte_stable
            else "STATE_DICT_BRIDGE_DRIFT_EXCEEDS_BYTE_STABLE_THRESHOLD"
        ),
        # Canonical Provenance per Catalog #287/#323/#192/#1.
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [
            "local_mlx_pytorch_state_dict_attestation_is_not_contest_auth_eval",
            "state_dict_bridge_byte_stability_does_not_imply_forward_pass_byte_stability",
            "forward_pass_carries_intrinsic_framework_arithmetic_drift_per_conv2d",
            "requires_paired_cuda_t4_or_linux_x86_64_eval_for_promotion",
        ],
        "operator_routable_summary": (
            "MLX numpy state_dict export is byte-stable per Slot 1 bridge; "
            "forward-pass parity carries bounded ~3e-5 RGB-head/framework "
            "arithmetic drift per the canonical bands. Use "
            "validate_mlx_pytorch_parity_within_tolerance(...) for runtime "
            "forward-pass attestation."
        ),
    }


# ============================================================================
# T3-GRAND-COUNCIL-ACTIVE-EXPLORATION-CONV2D-DRIFT-UNEXPLORED-PATHS extensions
# ============================================================================
# APPEND-ONLY 2026-05-25 per Catalog #110/#113 HISTORICAL_PROVENANCE +
# operator NON-NEGOTIABLE *"the grand council should explore the unexplored
# and address the unaddressed"*.
#
# Slot 2 declared the 4 paths "NOT FIXABLE"; T3 grand council active
# exploration EMPIRICALLY MEASURED the 4 paths via sister codex
# ``build_mlx_conv2d_accumulation_probe_manifest`` + ``mlx_runtime_determinism_contract``
# and produced revised per-path verdicts:
#
# Thread 1 (Kahan compensated summation): PARTIALLY_FIXABLE_MARGINAL
#   per-scale empirical reductions: 0.0% / 10.3% / 22.4% (3 PR95 stages)
#   Carmack MVP-first 5/5 step 2 falsification: predicted >50% reduction;  # DOCSTRING_PERCENT_CLAIM_OK:empirically_measured_pr95_3stage_falsification_anchor_Higham2002_theoretical_bound_REFUTED_per_carmack_mvp_first_5of5_methodology_predicted_vs_measured_documented_inline
#   max observed 22.4% — Higham 2002 theoretical bound REFUTED at MLX/PyTorch
#   CPU framework boundary because the dominant drift floor is NOT summation
#   precision; it is per-stage kernel vectorization/SIMD order differences.
#
# Thread 2 (FP64 intermediate accumulation): PARTIALLY_FIXABLE_MARGINAL
#   per-scale empirical reductions: 6.2% / 10.3% / 22.4% (same 3 stages)
#   Carmack MVP-first 5/5 step 2 falsification: predicted >50% reduction;  # DOCSTRING_PERCENT_CLAIM_OK:empirically_measured_pr95_3stage_falsification_anchor_FP64_intermediate_advantage_REFUTED_per_carmack_mvp_first_5of5_methodology_predicted_vs_measured_documented_inline
#   max observed 22.4% — 29-extra-mantissa-bit FP64 advantage REFUTED at the
#   measured scale because Conv2d MLX/PyTorch divergence is NOT at the
#   LSB-rounding-of-mul-add layer.
#
# Thread 3 (MLX-side deterministic-reduction): NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL
#   MLX 0.31.2 exposes ZERO public deterministic-reduction flags in core or
#   metal namespaces. Classification: framework_different_no_public_deterministic_reduction_flag.
#   The PyTorch-side pinning via _torch_backend_options is asymmetric; MLX
#   equivalent does not exist in the public API.
#
# Thread 4 (cuDNN reference Conv2d 3x3): DEFERRED_PENDING_PAID_DISPATCH
#   macOS Apple Silicon cannot execute PyTorch cuDNN (no NVIDIA GPU). Per
#   CLAUDE.md "MPS auth eval is NOISE" non-negotiable + Catalog #1, MPS is NOT
#   a valid substitute for cuDNN measurement (23x drift on PoseNet documented).
#   Operator-decision gate per Catalog #199 paired-env: $2-5 Modal A100 or
#   Vast.ai 4090 paired dispatch required for empirical measurement.


class ActiveExplorationPathVerdict(StrEnum):
    """Per-path active-exploration verdict for the 4 unexplored mitigation paths.

    Per Catalog #307 paradigm-vs-implementation classification + Carmack
    MVP-first 5/5 step 2 falsification thresholds:

    - FIXABLE: empirical reduction >= 50% (operationally meaningful, matches
      predicted theoretical band; substitution becomes the canonical primitive)

    - PARTIALLY_FIXABLE_MARGINAL: 10% <= reduction < 50% (measurable but below
      Carmack MVP-first 5/5 step 2 predicted band; substitution becomes a
      conditional engineering primitive at larger-spatial scales only)

    - NOT_FIXABLE_SUBSTITUTION_ONLY: 0 <= reduction < 10% (substitution does
      not move the drift floor; the bug class is non-summation; alternative
      paths required)

    - NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL: < 0 reduction OR no public API
      surface (substitution makes drift worse OR framework does not expose
      the canonical surface; requires upstream framework feature change)

    - DEFERRED_PENDING_PAID_DISPATCH: empirical measurement cannot run on
      local macOS (per CLAUDE.md "MPS auth eval is NOISE" non-negotiable);
      operator-decision gate required for $2-5 paid dispatch
    """

    FIXABLE = "FIXABLE"
    PARTIALLY_FIXABLE_MARGINAL = "PARTIALLY_FIXABLE_MARGINAL"
    NOT_FIXABLE_SUBSTITUTION_ONLY = "NOT_FIXABLE_SUBSTITUTION_ONLY"
    NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL = "NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL"
    DEFERRED_PENDING_PAID_DISPATCH = "DEFERRED_PENDING_PAID_DISPATCH"


@dataclass(frozen=True)
class ActiveExplorationThreadResult:
    """Per-thread typed result for the 4-thread active exploration.

    Per Catalog #287/#323 canonical Provenance: every thread result carries
    score_claim=False + promotable=False + axis_tag="[predicted]" because
    active-exploration measurements are LOCAL-PROBE evidence (Catalog #192
    macOS-MLX advisory), not contest-axis claims.
    """

    thread_id: int
    thread_name: str
    path_verdict: ActiveExplorationPathVerdict
    max_observed_reduction_percent: float
    predicted_reduction_percent_lower_bound: float
    carmack_mvp_first_falsified: bool
    per_scale_observations: tuple[dict[str, Any], ...]
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "active_exploration_thread_result.v1",
            "thread_id": self.thread_id,
            "thread_name": self.thread_name,
            "path_verdict": self.path_verdict.value,
            "max_observed_reduction_percent": self.max_observed_reduction_percent,
            "predicted_reduction_percent_lower_bound": self.predicted_reduction_percent_lower_bound,
            "carmack_mvp_first_falsified": self.carmack_mvp_first_falsified,
            "per_scale_observations": list(self.per_scale_observations),
            "notes": self.notes,
            # Canonical Provenance per Catalog #287/#323/#192/#1.
            "evidence_grade": EVIDENCE_GRADE_MLX,
            "evidence_tag": EVIDENCE_TAG_MLX,
            "axis_tag": "[predicted]",
            "score_claim": False,
            "promotion_eligible": False,
            "promotable": False,
            "ready_for_exact_eval_dispatch": False,
        }


@dataclass(frozen=True)
class MlxDeterministicInvestigationResult:
    """Thread 3 typed result for MLX deterministic-reduction enforcement.

    Empirical anchor 2026-05-25 on MLX 0.31.2:
    - public_core_deterministic_attrs: []
    - public_metal_deterministic_attrs: []
    - classification: framework_different_no_public_deterministic_reduction_flag
    """

    mlx_version: str
    deterministic_reduction_flag_available: bool
    public_core_deterministic_attrs: tuple[str, ...]
    public_metal_deterministic_attrs: tuple[str, ...]
    classification: str
    path_verdict: ActiveExplorationPathVerdict

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "mlx_deterministic_investigation_result.v1",
            "mlx_version": self.mlx_version,
            "deterministic_reduction_flag_available": self.deterministic_reduction_flag_available,
            "public_core_deterministic_attrs": list(self.public_core_deterministic_attrs),
            "public_metal_deterministic_attrs": list(self.public_metal_deterministic_attrs),
            "classification": self.classification,
            "path_verdict": self.path_verdict.value,
            "evidence_grade": EVIDENCE_GRADE_MLX,
            "evidence_tag": EVIDENCE_TAG_MLX,
            "axis_tag": "[predicted]",
            "score_claim": False,
            "promotion_eligible": False,
            "promotable": False,
            "ready_for_exact_eval_dispatch": False,
        }


@dataclass(frozen=True)
class CudnnReferenceMeasurementResult:
    """Thread 4 typed result for cuDNN reference Conv2d 3x3 measurement.

    On macOS Apple Silicon: torch.cuda.is_available()=False therefore the
    measurement is DEFERRED_PENDING_PAID_DISPATCH. MPS substitution is FORBIDDEN
    per CLAUDE.md "MPS auth eval is NOISE" non-negotiable (23x drift on
    PoseNet documented).
    """

    cuda_locally_available: bool
    cudnn_locally_available: bool
    mps_available: bool
    path_verdict: ActiveExplorationPathVerdict
    estimated_paid_dispatch_cost_usd: float | None
    mps_not_substitute_rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "cudnn_reference_measurement_result.v1",
            "cuda_locally_available": self.cuda_locally_available,
            "cudnn_locally_available": self.cudnn_locally_available,
            "mps_available": self.mps_available,
            "path_verdict": self.path_verdict.value,
            "estimated_paid_dispatch_cost_usd": self.estimated_paid_dispatch_cost_usd,
            "mps_not_substitute_rationale": self.mps_not_substitute_rationale,
            "evidence_grade": EVIDENCE_GRADE_MLX,
            "evidence_tag": EVIDENCE_TAG_MLX,
            "axis_tag": "[predicted]",
            "score_claim": False,
            "promotion_eligible": False,
            "promotable": False,
            "ready_for_exact_eval_dispatch": False,
        }


def classify_reduction_percent(
    reduction_percent: float,
) -> ActiveExplorationPathVerdict:
    """Classify a per-path drift reduction into ActiveExplorationPathVerdict.

    Per Carmack MVP-first 5/5 falsification thresholds:
    - >= 50%: FIXABLE (operationally meaningful drift reduction; matches
      predicted theoretical band; substitution becomes canonical primitive)
    - 10-50%: PARTIALLY_FIXABLE_MARGINAL (measurable but below predicted band;
      substitution becomes conditional engineering primitive at larger scales)
    - 0-10%: NOT_FIXABLE_SUBSTITUTION_ONLY (drift floor is non-summation)
    - <0%: NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL (substitute makes drift worse)

    Raises:
        ValueError: if reduction_percent is NaN or inf.
    """
    if not np.isfinite(reduction_percent):
        raise ValueError(f"reduction_percent must be finite, got {reduction_percent!r}")
    if reduction_percent >= 50.0:
        return ActiveExplorationPathVerdict.FIXABLE
    if reduction_percent >= 10.0:
        return ActiveExplorationPathVerdict.PARTIALLY_FIXABLE_MARGINAL
    if reduction_percent >= 0.0:
        return ActiveExplorationPathVerdict.NOT_FIXABLE_SUBSTITUTION_ONLY
    return ActiveExplorationPathVerdict.NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL


def kahan_compensated_sum(values: np.ndarray) -> np.float32:
    """Kahan compensated summation per Higham 2002 'Accuracy and Stability of
    Numerical Algorithms' Ch.4 Algorithm 4.2.

    Provably reduces summation drift from O(N * eps) to O(eps^2) where
    eps = float32 machine epsilon ~= 1.19e-7.

    Reference implementation in pure numpy for the canonical drift-attestation
    surface. The sister codex MLX implementation lives at
    ``tac.local_acceleration.mlx_scorer_adapters.MLXConv2dReference``
    accumulation_mode="kahan_fp32" which uses native MLX arrays + compensation
    term for full-substrate execution.

    Args:
        values: 1D array of float32 values to sum.

    Returns:
        Kahan-compensated sum as np.float32.
    """
    if values.size == 0:
        return np.float32(0.0)
    if values.ndim != 1:
        raise ValueError(f"kahan_compensated_sum expects 1D array; got {values.shape}")
    sum_acc = np.float32(0.0)
    c = np.float32(0.0)  # compensation term
    for v in values:
        y = np.float32(v) - c
        t = sum_acc + y
        c = (t - sum_acc) - y
        sum_acc = t
    return sum_acc


def kahan_conv2d_3x3(
    input_nhwc: np.ndarray,
    kernel_hwio: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    padding: int = 1,
    stride: int = 1,
) -> np.ndarray:
    """Conv2d 3x3 with Kahan compensated summation per output position.

    Reference numpy implementation. The sister codex MLX implementation at
    ``tac.local_acceleration.mlx_scorer_adapters.MLXConv2dReference``
    (accumulation_mode="kahan_fp32") is the canonical production surface and
    runs natively on MLX arrays. This pure-numpy variant is the design-time
    falsifiability surface for unit tests.

    Per active exploration empirical anchor 2026-05-25: Kahan summation
    reduces Conv2d 3x3 max_abs drift by 0.0% / 10.3% / 22.4% across 3 PR95
    HNeRV decoder stages. The Carmack MVP-first 5/5 step 2 prediction
    (Higham 2002 >50% reduction) is REFUTED at the MLX/PyTorch CPU boundary.  # DOCSTRING_PERCENT_CLAIM_OK:empirically_measured_pr95_3stage_anchor_2026_05_25_Higham_2002_theoretical_bound_REFUTED_documented_inline_per_carmack_mvp_first_5of5_falsification

    Args:
        input_nhwc: (N, H, W, C_in) float32 array.
        kernel_hwio: (kH, kW, C_in, C_out) float32 array.
        bias: optional (C_out,) float32 bias.
        padding: SAME-style padding (1 for kH=kW=3 produces same H,W).
        stride: convolution stride.

    Returns:
        (N, H_out, W_out, C_out) float32 array.
    """
    if input_nhwc.ndim != 4:
        raise ValueError(f"input_nhwc must be rank-4 NHWC; got {input_nhwc.shape}")
    if kernel_hwio.ndim != 4:
        raise ValueError(f"kernel_hwio must be rank-4 HWIO; got {kernel_hwio.shape}")

    n, h_in, w_in, c_in = input_nhwc.shape
    kh, kw, kc_in, c_out = kernel_hwio.shape
    if kc_in != c_in:
        raise ValueError(f"kernel C_in {kc_in} != input C_in {c_in}")
    if bias is not None and bias.shape != (c_out,):
        raise ValueError(f"bias must be ({c_out},); got {bias.shape}")
    if stride < 1 or padding < 0:
        raise ValueError(f"stride={stride}, padding={padding} must satisfy stride>=1, padding>=0")

    h_padded = h_in + 2 * padding
    w_padded = w_in + 2 * padding
    padded = np.zeros((n, h_padded, w_padded, c_in), dtype=np.float32)
    padded[:, padding : padding + h_in, padding : padding + w_in, :] = input_nhwc

    h_out = (h_padded - kh) // stride + 1
    w_out = (w_padded - kw) // stride + 1
    out = np.zeros((n, h_out, w_out, c_out), dtype=np.float32)

    for batch_idx in range(n):
        for out_y in range(h_out):
            for out_x in range(w_out):
                receptive_field = padded[
                    batch_idx,
                    out_y * stride : out_y * stride + kh,
                    out_x * stride : out_x * stride + kw,
                    :,
                ]  # (kH, kW, C_in)
                for oc in range(c_out):
                    terms = (receptive_field * kernel_hwio[:, :, :, oc]).reshape(-1).astype(np.float32)
                    out[batch_idx, out_y, out_x, oc] = kahan_compensated_sum(terms)
                if bias is not None:
                    out[batch_idx, out_y, out_x, :] += bias
    return out


def fp64_intermediate_conv2d_3x3(
    input_nhwc: np.ndarray,
    kernel_hwio: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    padding: int = 1,
    stride: int = 1,
) -> np.ndarray:
    """Conv2d 3x3 with FP64 intermediate accumulation; FP32 input/output.

    Casts FP32 inputs to FP64 for the accumulation arithmetic then casts back
    to FP32 at the output. FP64 has 53 mantissa bits vs FP32's 23 - the 29
    extra bits cast conv accumulation drift far below FP32 LSB at the
    individual mul-add layer.

    Reference numpy implementation. The sister codex MLX implementation at
    ``tac.local_acceleration.mlx_scorer_adapters.MLXConv2dReference``
    (accumulation_mode="fixed_fp64") is the canonical production surface.

    Per active exploration empirical anchor 2026-05-25: FP64 intermediate
    accumulation reduces Conv2d 3x3 max_abs drift by 6.2% / 10.3% / 22.4%
    across 3 PR95 HNeRV decoder stages. The Carmack MVP-first 5/5 step 2
    prediction (29-extra-mantissa-bit >99% reduction) is REFUTED because the  # DOCSTRING_PERCENT_CLAIM_OK:empirically_measured_pr95_3stage_anchor_2026_05_25_29_extra_mantissa_bit_predicted_advantage_REFUTED_per_carmack_mvp_first_5of5_falsification_documented_inline
    dominant drift floor is NOT at the LSB-mul-add layer.

    Args:
        input_nhwc: (N, H, W, C_in) float32 array.
        kernel_hwio: (kH, kW, C_in, C_out) float32 array.
        bias: optional (C_out,) float32 bias.
        padding: SAME-style padding.
        stride: convolution stride.

    Returns:
        (N, H_out, W_out, C_out) float32 array (FP64 internally, downcast).
    """
    if input_nhwc.ndim != 4:
        raise ValueError(f"input_nhwc must be rank-4 NHWC; got {input_nhwc.shape}")
    if kernel_hwio.ndim != 4:
        raise ValueError(f"kernel_hwio must be rank-4 HWIO; got {kernel_hwio.shape}")
    n, h_in, w_in, c_in = input_nhwc.shape
    kh, kw, kc_in, c_out = kernel_hwio.shape
    if kc_in != c_in:
        raise ValueError(f"kernel C_in {kc_in} != input C_in {c_in}")
    if bias is not None and bias.shape != (c_out,):
        raise ValueError(f"bias must be ({c_out},); got {bias.shape}")
    if stride < 1 or padding < 0:
        raise ValueError(f"stride={stride}, padding={padding} must satisfy stride>=1, padding>=0")

    input_fp64 = input_nhwc.astype(np.float64)
    kernel_fp64 = kernel_hwio.astype(np.float64)
    bias_fp64 = None if bias is None else bias.astype(np.float64)

    h_padded = h_in + 2 * padding
    w_padded = w_in + 2 * padding
    padded = np.zeros((n, h_padded, w_padded, c_in), dtype=np.float64)
    padded[:, padding : padding + h_in, padding : padding + w_in, :] = input_fp64

    h_out = (h_padded - kh) // stride + 1
    w_out = (w_padded - kw) // stride + 1
    out_fp64 = np.zeros((n, h_out, w_out, c_out), dtype=np.float64)

    for batch_idx in range(n):
        for out_y in range(h_out):
            for out_x in range(w_out):
                receptive_field = padded[
                    batch_idx,
                    out_y * stride : out_y * stride + kh,
                    out_x * stride : out_x * stride + kw,
                    :,
                ]
                for oc in range(c_out):
                    out_fp64[batch_idx, out_y, out_x, oc] = (
                        receptive_field * kernel_fp64[:, :, :, oc]
                    ).sum()
                if bias_fp64 is not None:
                    out_fp64[batch_idx, out_y, out_x, :] += bias_fp64
    return out_fp64.astype(np.float32)


# Canonical empirical anchor for T3 grand council active exploration 2026-05-25.
# Per Catalog #287/#323 canonical Provenance every claim carries
# evidence_grade=macOS-MLX-research-signal + axis_tag=[predicted] + score_claim=False.
ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR: dict[str, Any] = {
    "schema": "active_exploration_conv2d_drift_unexplored_paths_anchor.v1",
    "anchor_utc": "2026-05-25T20:08:34Z",
    "anchor_substrate": "PR95 HNeRV decoder Conv2d 3x3 stages (3 scales tested)",
    "framework_pair": "MLX CPU 0.31.2 vs PyTorch CPU 2.11.0 on macOS",
    "hardware": "Apple Silicon M5 Max (macOS)",
    "evidence_grade": EVIDENCE_GRADE_MLX,
    "evidence_tag": EVIDENCE_TAG_MLX,
    "axis_tag": "[predicted]",
    "score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "ready_for_exact_eval_dispatch": False,
    "thread_1_kahan_compensated_summation": {
        "path_verdict": ActiveExplorationPathVerdict.PARTIALLY_FIXABLE_MARGINAL.value,
        "max_observed_reduction_percent": 22.4,
        "predicted_reduction_percent_lower_bound": 50.0,
        "carmack_mvp_first_falsified": True,
        "per_scale_reductions": {
            "pr95_stage2_36_to_144_6x8": 0.0,
            "pr95_midstage_144_to_144_24x32": 10.3,
            "pr95_final_head_class_256_to_256_48x64": 22.4,
        },
        "notes": (
            "Kahan summation (Higham 2002) reduces drift at larger spatial scales "
            "(10-22%) but does NOT achieve the >50% Carmack MVP-first 5/5 step 2 "
            "predicted band. The dominant drift floor at small scales is NOT "
            "summation precision; at larger scales Kahan does help marginally. "
            "Sister codex implementation at tac.local_acceleration.mlx_scorer_adapters.MLXConv2dReference "
            "(accumulation_mode='kahan_fp32')."
        ),
    },
    "thread_2_fp64_intermediate_accumulation": {
        "path_verdict": ActiveExplorationPathVerdict.PARTIALLY_FIXABLE_MARGINAL.value,
        "max_observed_reduction_percent": 22.4,
        "predicted_reduction_percent_lower_bound": 50.0,
        "carmack_mvp_first_falsified": True,
        "per_scale_reductions": {
            "pr95_stage2_36_to_144_6x8": 6.2,
            "pr95_midstage_144_to_144_24x32": 10.3,
            "pr95_final_head_class_256_to_256_48x64": 22.4,
        },
        "notes": (
            "FP64 intermediate accumulation (29 extra mantissa bits) reduces drift "
            "by 6-22% across scales but does NOT achieve the >50% Carmack MVP-first "
            "5/5 step 2 predicted band. FP64 ~= Kahan at all measured scales (no "
            "additional FP64 advantage over Kahan). Sister codex implementation at "
            "tac.local_acceleration.mlx_scorer_adapters.MLXConv2dReference "
            "(accumulation_mode='fixed_fp64')."
        ),
    },
    "thread_3_mlx_deterministic_reduction_enforcement": {
        "path_verdict": ActiveExplorationPathVerdict.NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL.value,
        "mlx_version_tested": "0.31.2",
        "public_core_deterministic_attrs": [],
        "public_metal_deterministic_attrs": [],
        "classification": "framework_different_no_public_deterministic_reduction_flag",
        "carmack_mvp_first_falsified": False,
        "notes": (
            "MLX 0.31.2 exposes ZERO public deterministic-reduction flags in core "
            "or metal namespaces. PyTorch-side pinning via _torch_backend_options is "
            "ASYMMETRIC; the equivalent MLX surface does not exist in the public API. "
            "Verdict NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL_NO_PUBLIC_API until upstream "
            "MLX exposes the flag. Sister codex investigation at "
            "tac.local_acceleration.mlx_scorer_torch_parity.mlx_runtime_determinism_contract."
        ),
    },
    "thread_4_cudnn_reference_conv2d_3x3_measurement": {
        "path_verdict": ActiveExplorationPathVerdict.DEFERRED_PENDING_PAID_DISPATCH.value,
        "cuda_locally_available": False,
        "cudnn_locally_available": False,
        "mps_available": True,
        "mps_not_substitute_rationale": (
            "MPS is NOT a substitute for cuDNN per CLAUDE.md 'MPS auth eval is "
            "NOISE' non-negotiable (23x drift on PoseNet documented)"
        ),
        "estimated_paid_dispatch_cost_usd": 2.0,
        "carmack_mvp_first_falsified": False,
        "notes": (
            "macOS Apple Silicon torch.cuda.is_available()=False. Per CLAUDE.md "
            "'MPS auth eval is NOISE' + Catalog #1, MPS substitution is FORBIDDEN. "
            "Operator-decision gate per Catalog #199 paired-env: $2-5 paid Modal "
            "A100 / Vast.ai 4090 dispatch required for empirical cuDNN reference "
            "measurement. Sister codex infrastructure at "
            "tac.local_acceleration.mlx_scorer_torch_parity.build_mlx_conv2d_accumulation_probe_manifest "
            "supports --torch-device cuda parameter; CLI invocation deferred."
        ),
    },
    "aggregate_path_verdict_summary": {
        "fixable_count": 0,
        "partially_fixable_marginal_count": 2,
        "not_fixable_framework_fundamental_count": 1,
        "deferred_pending_paid_dispatch_count": 1,
        "overall_verdict": "PROCEED_WITH_REVISIONS",
    },
    "operator_routable_summary": (
        "Slot 2 'NOT FIXABLE' verdict was structurally over-stated. Empirical "
        "exploration shows: (1) Kahan partially fixable at larger scales (max 22.4%); "
        "(2) FP64 partially fixable at larger scales (max 22.4%); (3) MLX-side "
        "deterministic-reduction not fixable at framework boundary; (4) cuDNN "
        "reference deferred pending paid dispatch. Revised canonical primitive: "
        "ATTESTED-TOLERANCE PORTABILITY remains correct AT SMALL SCALES; substitute "
        "Kahan or FP64 accumulation_mode reduces drift 10-22% at larger spatial "
        "scales. Slot 1 export bridge VERDICT can claim NUMERIC_TOLERANCE 3.05e-5 "
        "as the canonical PR95 HNeRV decoder band (unchanged); per-stage "
        "substitution remains optional + measurably beneficial only at "
        ">= 144-channel 24x32 spatial scales."
    ),
    "carmack_mvp_first_step_2_falsification_summary": {
        "thread_1_predicted_lower_bound_percent": 50.0,
        "thread_1_max_observed_percent": 22.4,
        "thread_1_falsified": True,
        "thread_2_predicted_lower_bound_percent": 50.0,
        "thread_2_max_observed_percent": 22.4,
        "thread_2_falsified": True,
        "thread_3_predicted_verdict_lower_bound": "FIXABLE_OR_FRAMEWORK_DIFFERENT",
        "thread_3_actual_verdict": "NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL_NO_PUBLIC_API",
        "thread_4_predicted_verdict_lower_bound": "LOCAL_OR_DEFERRED",
        "thread_4_actual_verdict": "DEFERRED_PENDING_PAID_DISPATCH",
    },
}
