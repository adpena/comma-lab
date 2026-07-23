# SPDX-License-Identifier: MIT
"""Typed Phase-0 substrate for the DDM scorer analytic atlas.

This module modernizes the old ``evaluator_response_atlas`` index without
pretending that the execution-disabled AT1 lane materialized the expensive
600-pair tensors.  It provides:

* frozen-weight closed-form factor builders;
* hash-fresh factor records, stage checkpoints, and exact n600 coverage gates;
* exact Jacobian composition and gaze pullback reference operations;
* a loss-accounted SDWL1 description-fact <-> E2 runtime bridge;
* the single producer for the live DDM pair/site lambda bundle.

The full gaze and Jacobian artifacts remain external tensor references.  A
factor is usable only while every stamped input hash still matches.  Tables
without a named consumer are counted but inert.

Authority: research-only, ``score_claim=false``, ``execution_allowed=false``,
``[macOS-CPU frozen-scorer advisory]``.  Nothing here launches a scorer, a
trainer, a provider, or an evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tac.ddm_costate_law import ddm_joint_costate, realized_pair_distortion_delta

SCHEMA = "ddm_scorer_analytic_atlas.v2"
FACTOR_SCHEMA = "ddm_scorer_analytic_factor.v1"
CHECKPOINT_SCHEMA = "ddm_scorer_analytic_atlas_checkpoint.v1"
BRIDGE_SCHEMA = "sdwl1_e2_coordinate_bridge.v1"
LAMBDA_SCHEMA = "ddm_scorer_analytic_lambda_bundle.v1"
PAIR_COUNT = 600
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
SHA256_HEX_LENGTH = 64
FP32_AGGREGATION_SCHEMA = "ddm_scorer_fp32_aggregation_order_envelope.v1"

# These are code pointers, not copied results.  A materializer must stamp the
# actual source bytes and scorer/model inputs it consumed.
CURRENT_PORTS: tuple[str, ...] = (
    "tac.scorer.load_default_scorers",
    "tac.scorer.make_scorers_differentiable",
    "tac.canonical_equations.segnet_head_rank4_flipdist_20260715",
    "tac.optimization.resize_full_kernel.FullResizeKernel",
    "tac.witness_control.exact_costate_reuse",
    "tac.witness_control.pose_verdict_gate",
    "tac.optimization.ddm_runtime_receiver",
)


class AnalyticAtlasError(ValueError):
    """A factor, freshness, completeness, or bridge contract failed closed."""


class FactorStatus(StrEnum):
    DERIVED = "DERIVED"
    MEASURED = "MEASURED"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"


class ConsumptionStatus(StrEnum):
    CONSUMED = "CONSUMED"
    COUNTED_INERT = "COUNTED_INERT"
    WAITING_CONSUMER = "WAITING_CONSUMER"


def _is_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalyticAtlasError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise AnalyticAtlasError(f"{name} must be finite")
    return result


def fp32_sequential_sum(values: Sequence[float]) -> float:
    """Reproduce the evaluator's zero-dimensional fp32 ``+=`` accumulation."""

    if not values:
        raise AnalyticAtlasError("fp32 accumulation requires at least one value")
    accumulator = np.float32(0.0)
    for index, value in enumerate(values):
        scalar = np.float32(_finite(value, name=f"values[{index}]"))
        accumulator = np.float32(accumulator + scalar)
    return float(accumulator)


def fp32_aggregation_order_envelope(
    *,
    pose_batch_sums: Sequence[float],
    seg_batch_sums: Sequence[float],
    total_pairs: int = PAIR_COUNT,
) -> dict[str, Any]:
    """Quantify reduction-order sensitivity of evaluator-shaped batch sums.

    Inputs are already the per-batch sums emitted by the two distortion heads:
    PoseNet contributes the sum of per-pair six-coordinate MSE values and
    SegNet contributes the sum of uniform-site per-pair disagreement means.
    This function changes *only* the order of the evaluator's sequential fp32
    batch accumulator.  It does not claim that all orders occur on one hardware
    axis.
    """

    if (
        isinstance(total_pairs, bool)
        or not isinstance(total_pairs, int)
        or total_pairs <= 0
    ):
        raise AnalyticAtlasError("total_pairs must be a positive integer")
    if len(pose_batch_sums) != len(seg_batch_sums) or not pose_batch_sums:
        raise AnalyticAtlasError("Pose/Seg batch-sum vectors must be nonempty peers")

    def summarize(values: Sequence[float], *, network: str) -> dict[str, Any]:
        array = np.asarray(
            [_finite(value, name=f"{network}_batch_sum") for value in values],
            dtype=np.float32,
        )
        if np.any(array < 0.0):
            raise AnalyticAtlasError(f"{network} batch sums must be nonnegative")
        orders = {
            "forward": array,
            "reverse": array[::-1],
            "ascending": np.sort(array),
            "descending": np.sort(array)[::-1],
        }
        raw = {
            name: fp32_sequential_sum(order.tolist())
            for name, order in orders.items()
        }
        fp32_pair_count = np.float32(total_pairs)
        if network == "posenet":
            score_terms = {
                name: math.sqrt(
                    10.0 * float(np.float32(value) / fp32_pair_count)
                )
                for name, value in raw.items()
            }
        else:
            score_terms = {
                name: 100.0 * float(np.float32(value) / fp32_pair_count)
                for name, value in raw.items()
            }
        return {
            "network": network,
            "batch_count": int(array.size),
            "float64_sum_of_fp32_batch_scalars": float(
                array.astype(np.float64).sum()
            ),
            "fp32_accumulator_by_order": raw,
            "raw_accumulator_span": max(raw.values()) - min(raw.values()),
            "score_term_by_order": score_terms,
            "score_term_span": max(score_terms.values()) - min(score_terms.values()),
        }

    pose = summarize(pose_batch_sums, network="posenet")
    seg = summarize(seg_batch_sums, network="segnet")
    return {
        "schema": FP32_AGGREGATION_SCHEMA,
        "first_rung": True,
        "pair_count": total_pairs,
        "scope": (
            "batch-reduction order only; forward pass, pair order, device, "
            "checkpoint, and receiver bytes held fixed"
        ),
        "pose": pose,
        "seg": seg,
        "score_span_upper_bound_if_term_extrema_cooccur": (
            pose["score_term_span"] + seg["score_term_span"]
        ),
        "score_claim": False,
    }


@dataclass(frozen=True)
class SourceHashStamp:
    """One producer input identity and its consumption-time validity rule."""

    source_id: str
    path: str
    sha256: str
    bytes: int
    validity_horizon: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.path or not self.validity_horizon:
            raise AnalyticAtlasError("source stamp text fields must be nonempty")
        if not _is_sha256(self.sha256):
            raise AnalyticAtlasError(f"{self.source_id}: invalid SHA-256")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise AnalyticAtlasError(f"{self.source_id}: bytes must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TensorArtifactRef:
    """A durable external tensor; large arrays never live in the JSON manifest."""

    path: str
    sha256: str
    bytes: int
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.path or not self.dtype or not _is_sha256(self.sha256):
            raise AnalyticAtlasError("tensor artifact identity is incomplete")
        if (
            isinstance(self.bytes, bool)
            or not isinstance(self.bytes, int)
            or self.bytes <= 0
            or not self.shape
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.shape
            )
        ):
            raise AnalyticAtlasError("tensor artifact geometry/bytes are invalid")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data


@dataclass(frozen=True)
class AnalyticFactor:
    """One typed factor with mandatory first-rung and freshness custody."""

    factor_id: str
    factor_kind: str
    status: FactorStatus
    first_rung: bool
    pair_start: int
    pair_stop: int
    source_hashes: tuple[SourceHashStamp, ...]
    content_sha256: str
    payload: Mapping[str, Any]
    network: str | None = None
    layer_id: str | None = None
    consumer: str | None = None
    consumption_status: ConsumptionStatus = ConsumptionStatus.COUNTED_INERT
    nonadditive_pool_id: str | None = None
    uint8_surviving_projection: Mapping[str, Any] | None = None
    tensor: TensorArtifactRef | None = None

    def __post_init__(self) -> None:
        if not self.factor_id or not self.factor_kind or not self.source_hashes:
            raise AnalyticAtlasError("factor identity and source hashes are required")
        if self.first_rung is not True:
            raise AnalyticAtlasError("every factor row must carry FIRST-RUNG=true")
        if not (0 <= self.pair_start < self.pair_stop <= PAIR_COUNT):
            raise AnalyticAtlasError("factor pair interval must lie inside exact n600")
        if len({stamp.source_id for stamp in self.source_hashes}) != len(
            self.source_hashes
        ):
            raise AnalyticAtlasError("factor source IDs must be unique")
        if not _is_sha256(self.content_sha256):
            raise AnalyticAtlasError("factor content SHA-256 is invalid")
        if self.content_sha256 != _payload_sha256(dict(self.payload)):
            raise AnalyticAtlasError("factor payload hash mismatch")
        if self.consumption_status is ConsumptionStatus.CONSUMED and not self.consumer:
            raise AnalyticAtlasError("a consumed factor requires a named consumer")
        if self.factor_kind == "axis_projection.amplitude":
            projection = self.uint8_surviving_projection
            required = {"projection_kind", "survives_uint8_r", "artifact_sha256"}
            if not isinstance(projection, Mapping) or set(projection) != required:
                raise AnalyticAtlasError(
                    "every amplitude factor requires one complete uint8-surviving projection"
                )
            if not isinstance(projection["survives_uint8_r"], bool) or not _is_sha256(
                str(projection["artifact_sha256"])
            ):
                raise AnalyticAtlasError("amplitude uint8 projection custody is invalid")

    def verify_fresh(self, current_hashes: Mapping[str, str]) -> None:
        changed = [
            stamp.source_id
            for stamp in self.source_hashes
            if current_hashes.get(stamp.source_id) != stamp.sha256
        ]
        if changed:
            raise AnalyticAtlasError(
                "factor inputs are stale; re-derive instead of consuming: "
                + ",".join(sorted(changed))
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FACTOR_SCHEMA,
            "factor_id": self.factor_id,
            "factor_kind": self.factor_kind,
            "status": self.status.value,
            "first_rung": self.first_rung,
            "pair_range": [self.pair_start, self.pair_stop],
            "source_hashes": [row.to_dict() for row in self.source_hashes],
            "content_sha256": self.content_sha256,
            "payload": dict(self.payload),
            "network": self.network,
            "layer_id": self.layer_id,
            "consumer": self.consumer,
            "consumption_status": self.consumption_status.value,
            "nonadditive_pool_id": self.nonadditive_pool_id,
            "uint8_surviving_projection": (
                dict(self.uint8_surviving_projection)
                if self.uint8_surviving_projection is not None
                else None
            ),
            "tensor": self.tensor.to_dict() if self.tensor else None,
        }


def build_factor(
    *,
    factor_id: str,
    factor_kind: str,
    status: FactorStatus,
    payload: Mapping[str, Any],
    source_hashes: Sequence[SourceHashStamp],
    pair_start: int = 0,
    pair_stop: int = PAIR_COUNT,
    network: str | None = None,
    layer_id: str | None = None,
    consumer: str | None = None,
    consumption_status: ConsumptionStatus = ConsumptionStatus.COUNTED_INERT,
    nonadditive_pool_id: str | None = None,
    uint8_surviving_projection: Mapping[str, Any] | None = None,
    tensor: TensorArtifactRef | None = None,
) -> AnalyticFactor:
    canonical_payload = dict(payload)
    return AnalyticFactor(
        factor_id=factor_id,
        factor_kind=factor_kind,
        status=status,
        first_rung=True,
        pair_start=pair_start,
        pair_stop=pair_stop,
        source_hashes=tuple(source_hashes),
        content_sha256=_payload_sha256(canonical_payload),
        payload=canonical_payload,
        network=network,
        layer_id=layer_id,
        consumer=consumer,
        consumption_status=consumption_status,
        nonadditive_pool_id=nonadditive_pool_id,
        uint8_surviving_projection=uint8_surviving_projection,
        tensor=tensor,
    )


def derive_batchnorm_expected_stats(
    *,
    layer_id: str,
    running_mean: np.ndarray,
    running_variance: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float,
    source_hashes: Sequence[SourceHashStamp],
) -> AnalyticFactor:
    """Derive the frozen eval-BN affine map exactly from checkpoint values."""

    mean = np.asarray(running_mean, dtype=np.float64)
    variance = np.asarray(running_variance, dtype=np.float64)
    scale_weight = np.asarray(gamma, dtype=np.float64)
    bias = np.asarray(beta, dtype=np.float64)
    eps = _finite(epsilon, name="epsilon")
    if eps <= 0.0 or not (
        mean.ndim == variance.ndim == scale_weight.ndim == bias.ndim == 1
        and mean.shape == variance.shape == scale_weight.shape == bias.shape
        and mean.size > 0
    ):
        raise AnalyticAtlasError("BN checkpoint arrays or epsilon are invalid")
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(variance).all()
        or not np.isfinite(scale_weight).all()
        or not np.isfinite(bias).all()
        or np.any(variance < 0.0)
    ):
        raise AnalyticAtlasError("BN checkpoint arrays must be finite with variance >= 0")
    affine_scale = scale_weight / np.sqrt(variance + eps)
    affine_offset = bias - affine_scale * mean
    payload = {
        "formula": "y=gamma*(x-running_mean)/sqrt(running_variance+epsilon)+beta",
        "channels": int(mean.size),
        "epsilon": eps,
        "running_mean": mean.tolist(),
        "running_variance": variance.tolist(),
        "gamma": scale_weight.tolist(),
        "beta": bias.tolist(),
        "affine_scale": affine_scale.tolist(),
        "affine_offset": affine_offset.tolist(),
    }
    return build_factor(
        factor_id=f"bn_expected_stats:{layer_id}",
        factor_kind="closed_form.batchnorm_expected_stats",
        status=FactorStatus.DERIVED,
        payload=payload,
        source_hashes=source_hashes,
        layer_id=layer_id,
    )


def evaluate_batchnorm_factor(factor: AnalyticFactor, value: np.ndarray) -> np.ndarray:
    if factor.factor_kind != "closed_form.batchnorm_expected_stats":
        raise AnalyticAtlasError("factor is not a BN expected-statistics table")
    scale = np.asarray(factor.payload["affine_scale"], dtype=np.float64)
    offset = np.asarray(factor.payload["affine_offset"], dtype=np.float64)
    x = np.asarray(value, dtype=np.float64)
    if x.shape[-1] != scale.size:
        raise AnalyticAtlasError("BN input channel count changed")
    return x * scale + offset


def derive_se_gate_closed_form(
    *,
    layer_id: str,
    reduce_weight: np.ndarray,
    reduce_bias: np.ndarray,
    expand_weight: np.ndarray,
    expand_bias: np.ndarray,
    source_hashes: Sequence[SourceHashStamp],
) -> AnalyticFactor:
    """Record ``sigmoid(W2*silu(W1*GAP(z)+b1)+b2)`` from frozen weights."""

    w1 = np.asarray(reduce_weight, dtype=np.float64)
    b1 = np.asarray(reduce_bias, dtype=np.float64)
    w2 = np.asarray(expand_weight, dtype=np.float64)
    b2 = np.asarray(expand_bias, dtype=np.float64)
    if (
        w1.ndim != 2
        or w2.ndim != 2
        or b1.shape != (w1.shape[0],)
        or b2.shape != (w2.shape[0],)
        or w2.shape[1] != w1.shape[0]
        or any(not np.isfinite(row).all() for row in (w1, b1, w2, b2))
    ):
        raise AnalyticAtlasError("SE gate weight geometry is invalid")
    payload = {
        "formula": "gate=sigmoid(W_expand*silu(W_reduce*GAP(z)+b_reduce)+b_expand)",
        "reduce_weight": w1.tolist(),
        "reduce_bias": b1.tolist(),
        "expand_weight": w2.tolist(),
        "expand_bias": b2.tolist(),
        "input_channels": int(w1.shape[1]),
        "squeeze_channels": int(w1.shape[0]),
        "output_channels": int(w2.shape[0]),
    }
    return build_factor(
        factor_id=f"se_gate:{layer_id}",
        factor_kind="closed_form.squeeze_excite_gate",
        status=FactorStatus.DERIVED,
        payload=payload,
        source_hashes=source_hashes,
        layer_id=layer_id,
    )


def evaluate_se_gate_factor(
    factor: AnalyticFactor, global_average: np.ndarray
) -> np.ndarray:
    if factor.factor_kind != "closed_form.squeeze_excite_gate":
        raise AnalyticAtlasError("factor is not an SE gate")
    x = np.asarray(global_average, dtype=np.float64)
    w1 = np.asarray(factor.payload["reduce_weight"], dtype=np.float64)
    b1 = np.asarray(factor.payload["reduce_bias"], dtype=np.float64)
    w2 = np.asarray(factor.payload["expand_weight"], dtype=np.float64)
    b2 = np.asarray(factor.payload["expand_bias"], dtype=np.float64)
    if x.shape[-1] != w1.shape[1]:
        raise AnalyticAtlasError("SE input channel count changed")
    hidden = x @ w1.T + b1
    hidden = hidden / (1.0 + np.exp(-hidden))
    logits = hidden @ w2.T + b2
    return 1.0 / (1.0 + np.exp(-logits))


def derive_kernel_dft_bank(
    *,
    layer_id: str,
    kernels: np.ndarray,
    source_hashes: Sequence[SourceHashStamp],
) -> AnalyticFactor:
    """Derive per-kernel complex DFT magnitude and phase from frozen weights."""

    weights = np.asarray(kernels, dtype=np.float64)
    if weights.ndim != 4 or not np.isfinite(weights).all():
        raise AnalyticAtlasError("convolution kernels must be finite OIHW weights")
    spectrum = np.fft.fft2(weights, axes=(-2, -1))
    payload = {
        "formula": "DFT2(weight[o,i,:,:])",
        "weight_shape": list(weights.shape),
        "magnitude": np.abs(spectrum).tolist(),
        "phase_radians": np.angle(spectrum).tolist(),
        "frequency_axis": "native_kernel_cycles_per_sample",
        "carrier_basis_authority": (
            "analysis_only; residual/carrier selection remains curvelet/shearlet"
        ),
    }
    return build_factor(
        factor_id=f"kernel_dft:{layer_id}",
        factor_kind="closed_form.kernel_dft_frequency_phase",
        status=FactorStatus.DERIVED,
        payload=payload,
        source_hashes=source_hashes,
        layer_id=layer_id,
    )


def derive_bn_silu_contrast(
    *,
    layer_id: str,
    bn_factor: AnalyticFactor,
) -> AnalyticFactor:
    """Compose the frozen BN affine row with exact SiLU contrast response."""

    if bn_factor.factor_kind != "closed_form.batchnorm_expected_stats":
        raise AnalyticAtlasError("BN x SiLU composition requires a BN factor")
    payload = {
        "formula": "silu(a*x+b)=(a*x+b)*sigmoid(a*x+b)",
        "bn_factor_id": bn_factor.factor_id,
        "affine_scale": list(bn_factor.payload["affine_scale"]),
        "affine_offset": list(bn_factor.payload["affine_offset"]),
        "contrast_derivative": (
            "a*(sigmoid(u)+u*sigmoid(u)*(1-sigmoid(u))),u=a*x+b"
        ),
    }
    return build_factor(
        factor_id=f"bn_silu_contrast:{layer_id}",
        factor_kind="closed_form.bn_silu_contrast",
        status=FactorStatus.DERIVED,
        payload=payload,
        source_hashes=bn_factor.source_hashes,
        layer_id=layer_id,
    )


def build_r_null_band_certificate(
    *,
    resize_authority: SourceHashStamp,
    requested_band_ids: Sequence[str],
) -> AnalyticFactor:
    """Reuse #580 and refuse a global DFT-null claim it does not establish.

    #580 proves the complete *spatial* kernel of the exact resize.  The shared
    resize is not represented as a circular convolution, so that receipt alone
    does not certify named global DFT bands as exactly dead.  Until a
    SHA-stamped diagonalization lands, the correct dead-band set is empty and
    the DR2b admission guard must refuse free truncation.
    """

    from tac.canonical_equations.resize_full_kernel_structure_20260720 import (
        full_resize_kernel_direct_sum,
    )

    spatial = full_resize_kernel_direct_sum()
    payload = {
        "resize_authority": (
            "tac.canonical_equations.resize_full_kernel_structure_20260720:"
            "full_resize_kernel_direct_sum"
        ),
        "spatial_kernel": spatial,
        "requested_band_ids": list(requested_band_ids),
        "exact_dead_band_ids": [],
        "frequency_certificate_status": (
            "BLOCKED_NO_EXACT_GLOBAL_DFT_DIAGONALIZATION_FROM_580"
        ),
        "frequency_band_admission": "REFUSE_ZERO_BYTE_TRUNCATION",
        "verdict_scope": (
            "#580 spatial-kernel authority only; spectral families remain open"
        ),
    }
    return build_factor(
        factor_id="r_null_band_certificate:#580",
        factor_kind="closed_form.r_null_band_certificate",
        status=FactorStatus.BLOCKED,
        payload=payload,
        source_hashes=(resize_authority,),
        consumer="tac.optimization.ddm_dr2b_tolerance_costate.frequency_band_admission",
        consumption_status=ConsumptionStatus.WAITING_CONSUMER,
    )


def build_gaze_factor(
    *,
    network: Literal["segnet", "posenet"],
    layer_id: str,
    pair_start: int,
    pair_stop: int,
    tensor: TensorArtifactRef,
    source_hashes: Sequence[SourceHashStamp],
    vjp_count_per_pair: int,
    head_pullback_rank: int | None,
) -> AnalyticFactor:
    """Register an externally materialized exact ``dS/dz_k`` tensor shard."""

    if network == "posenet" and vjp_count_per_pair != 6:
        raise AnalyticAtlasError("PoseNet gaze requires exactly six VJPs per pair")
    if network == "segnet" and head_pullback_rank != 4:
        raise AnalyticAtlasError("SegNet gaze requires the exact rank-4 head pullback")
    payload = {
        "definition": "lambda_k=dS/dz_k",
        "network": network,
        "layer_id": layer_id,
        "pair_range": [pair_start, pair_stop],
        "vjp_count_per_pair": vjp_count_per_pair,
        "head_pullback_rank": head_pullback_rank,
        "tensor": tensor.to_dict(),
        "input_layer_restriction": (
            "dS/dx is the layer-0 restriction of this same factor chain"
        ),
    }
    return build_factor(
        factor_id=f"gaze:{network}:{layer_id}:{pair_start}:{pair_stop}",
        factor_kind="gaze.exact_layer_costate",
        status=FactorStatus.MEASURED,
        payload=payload,
        source_hashes=source_hashes,
        pair_start=pair_start,
        pair_stop=pair_stop,
        network=network,
        layer_id=layer_id,
        tensor=tensor,
    )


def require_total_gaze_coverage(
    factors: Sequence[AnalyticFactor],
    *,
    expected_layers: Mapping[str, Sequence[str]],
) -> None:
    """Require gap-free exact n600 coverage for every declared scorer layer."""

    for network, layers in expected_layers.items():
        for layer_id in layers:
            intervals = sorted(
                (factor.pair_start, factor.pair_stop)
                for factor in factors
                if factor.factor_kind == "gaze.exact_layer_costate"
                and factor.network == network
                and factor.layer_id == layer_id
            )
            cursor = 0
            for start, stop in intervals:
                if start != cursor:
                    raise AnalyticAtlasError(
                        f"{network}:{layer_id} gaze coverage gap/overlap at {cursor}"
                    )
                cursor = stop
            if cursor != PAIR_COUNT:
                raise AnalyticAtlasError(
                    f"{network}:{layer_id} gaze is not n600-complete ({cursor}/600)"
                )


def compose_jacobian_factors(factors: Sequence[np.ndarray]) -> np.ndarray:
    """Compose forward Jacobians ``J_n @ ... @ J_1`` exactly in float64."""

    if not factors:
        raise AnalyticAtlasError("at least one Jacobian factor is required")
    matrices = [np.asarray(value, dtype=np.float64) for value in factors]
    if any(matrix.ndim != 2 or not np.isfinite(matrix).all() for matrix in matrices):
        raise AnalyticAtlasError("Jacobian factors must be finite matrices")
    result = matrices[0]
    for matrix in matrices[1:]:
        if matrix.shape[1] != result.shape[0]:
            raise AnalyticAtlasError("Jacobian factor dimensions do not compose")
        result = matrix @ result
    return result


def pull_back_gaze(
    terminal_gaze: np.ndarray, forward_factors: Sequence[np.ndarray]
) -> np.ndarray:
    """Compute the input-layer restriction ``J_1.T ... J_n.T lambda_n``."""

    gaze = np.asarray(terminal_gaze, dtype=np.float64)
    if gaze.ndim != 1 or not np.isfinite(gaze).all():
        raise AnalyticAtlasError("terminal gaze must be one finite vector")
    composed = compose_jacobian_factors(forward_factors)
    if composed.shape[0] != gaze.size:
        raise AnalyticAtlasError("terminal gaze does not match composed Jacobian output")
    return composed.T @ gaze


def project_gaze_onto_axis(
    *,
    factor_id: str,
    gaze: np.ndarray,
    basis: np.ndarray,
    axis_name: Literal[
        "amplitude",
        "frequency",
        "phase",
        "contrast",
        "channel_energy",
        "texture_statistics",
    ],
    source_hashes: Sequence[SourceHashStamp],
    uint8_surviving_projection: Mapping[str, Any] | None = None,
    nonadditive_pool_id: str | None = None,
) -> AnalyticFactor:
    """Project one gaze vector onto a declared analysis axis."""

    vector = np.asarray(gaze, dtype=np.float64)
    axes = np.asarray(basis, dtype=np.float64)
    if (
        vector.ndim != 1
        or axes.ndim != 2
        or axes.shape[1] != vector.size
        or not np.isfinite(vector).all()
        or not np.isfinite(axes).all()
    ):
        raise AnalyticAtlasError("gaze/basis geometry is invalid")
    coefficients = axes @ vector
    payload = {
        "axis": axis_name,
        "definition": "basis @ gaze",
        "basis_shape": list(axes.shape),
        "coefficients": coefficients.tolist(),
        "nonadditive_pool_id": nonadditive_pool_id,
    }
    return build_factor(
        factor_id=factor_id,
        factor_kind=f"axis_projection.{axis_name}",
        status=FactorStatus.DERIVED,
        payload=payload,
        source_hashes=source_hashes,
        nonadditive_pool_id=nonadditive_pool_id,
        uint8_surviving_projection=uint8_surviving_projection,
    )


@dataclass(frozen=True)
class NonAdditivePool:
    pool_id: str
    member_factor_ids: tuple[str, ...]
    kkt_constraint: str
    source_hashes: tuple[SourceHashStamp, ...]

    def __post_init__(self) -> None:
        if (
            not self.pool_id
            or not self.member_factor_ids
            or len(set(self.member_factor_ids)) != len(self.member_factor_ids)
            or not self.kkt_constraint
            or not self.source_hashes
        ):
            raise AnalyticAtlasError("non-additive pool contract is incomplete")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "member_factor_ids": list(self.member_factor_ids),
            "kkt_constraint": self.kkt_constraint,
            "source_hashes": [row.to_dict() for row in self.source_hashes],
        }


def build_manifest(
    *,
    factors: Sequence[AnalyticFactor],
    pools: Sequence[NonAdditivePool],
    materialization_status: str,
) -> dict[str, Any]:
    """Build a no-fake manifest; unconsumed factors stay counted but inert."""

    factor_ids = [factor.factor_id for factor in factors]
    if len(set(factor_ids)) != len(factor_ids):
        raise AnalyticAtlasError("factor IDs must be unique")
    pool_members = {member for pool in pools for member in pool.member_factor_ids}
    missing = sorted(pool_members - set(factor_ids))
    if missing:
        raise AnalyticAtlasError(f"non-additive pool references unknown factors: {missing}")
    inert = [
        factor.factor_id
        for factor in factors
        if factor.consumption_status is not ConsumptionStatus.CONSUMED
    ]
    return {
        "schema": SCHEMA,
        "pair_count_required": PAIR_COUNT,
        "n600_or_not_evidence": True,
        "materialization_status": materialization_status,
        "current_port_pointers": list(CURRENT_PORTS),
        "factors": [factor.to_dict() for factor in factors],
        "nonadditive_pools": [pool.to_dict() for pool in pools],
        "factor_count": len(factors),
        "unconsumed_counted_inert_factor_ids": inert,
        "authority": {
            "research_only": True,
            "execution_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "axis": EVIDENCE_AXIS,
        },
    }


@dataclass(frozen=True)
class AtlasCheckpoint:
    """Small resumable stage state; every restored source hash must still match."""

    stage_id: str
    completed_factor_ids: tuple[str, ...]
    source_hashes: tuple[SourceHashStamp, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.stage_id
            or len(set(self.completed_factor_ids)) != len(self.completed_factor_ids)
            or not self.source_hashes
            or not _is_sha256(self.manifest_sha256)
        ):
            raise AnalyticAtlasError("atlas checkpoint contract is incomplete")

    def verify_fresh(self, current_hashes: Mapping[str, str]) -> None:
        changed = [
            row.source_id
            for row in self.source_hashes
            if current_hashes.get(row.source_id) != row.sha256
        ]
        if changed:
            raise AnalyticAtlasError(
                "checkpoint inputs are stale; re-derive stage: "
                + ",".join(sorted(changed))
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "stage_id": self.stage_id,
            "completed_factor_ids": list(self.completed_factor_ids),
            "source_hashes": [row.to_dict() for row in self.source_hashes],
            "manifest_sha256": self.manifest_sha256,
        }


def write_stage_checkpoint(path: Path, checkpoint: AtlasCheckpoint) -> None:
    """Atomically publish one preserved stage checkpoint; never overwrite."""

    destination = Path(path)
    if destination.exists():
        raise AnalyticAtlasError(f"stage checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(checkpoint.to_dict()) + b"\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


@dataclass(frozen=True)
class CoordinateRelation:
    relation_id: str
    fact_coordinates: str
    runtime_coordinates: str
    direction: str
    invertibility: str
    fact_scalar_count: int
    runtime_coordinate_count: int
    loss_account: Mapping[str, Any]
    first_rung: bool = True

    def __post_init__(self) -> None:
        if (
            not self.relation_id
            or self.first_rung is not True
            or self.fact_scalar_count < 0
            or self.runtime_coordinate_count < 0
            or not self.loss_account
        ):
            raise AnalyticAtlasError("coordinate relation contract is incomplete")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_sdwl1_e2_coordinate_bridge(
    *, source_hashes: Sequence[SourceHashStamp]
) -> dict[str, Any]:
    """Build the complete explicit loss ledger between current SDWL1 and E2.

    This is intentionally not called invertible.  SDWL1 contains 45,600
    aggregate scalar facts, while E2 carries a 117,964,800-cell role plane and
    702,000 int16 chart coordinates.  E2's compact pose member is explicitly
    absent.  The bridge therefore keeps fact-side and runtime-side prices in
    one typed object but forbids transferring a tolerance across a lossy row.
    """

    if not source_hashes:
        raise AnalyticAtlasError("bridge requires SHA-stamped SDWL1/E2 sources")
    source_ids = {row.source_id for row in source_hashes}
    if not {"sdwl1", "e2_manifest"}.issubset(source_ids):
        raise AnalyticAtlasError("bridge requires sdwl1 and e2_manifest source stamps")

    relations = (
        CoordinateRelation(
            relation_id="semantic_role_plane_to_partition_moments",
            fact_coordinates="pair[600] x partition_cell[class=5] x fields[8]",
            runtime_coordinates="semantic/composed.dds[600,384,512]",
            direction="runtime_to_fact_partial",
            invertibility="MANY_TO_ONE_AND_BACKGROUND_UNRESOLVED",
            fact_scalar_count=600 * 5 * 8,
            runtime_coordinate_count=600 * 384 * 512,
            loss_account={
                "runtime_role_plane_explicit_cells": 117_964_800,
                "inverse_enumeration": "NOT_AVAILABLE",
                "background_role_code": 0,
                "background_to_sdwl1_class": "UNRESOLVED",
                "price_transfer": "FORBIDDEN",
            },
        ),
        CoordinateRelation(
            relation_id="semantic_role_plane_to_separatrix_geometry",
            fact_coordinates="pair[600] x separatrix[class=5] x fields[6]",
            runtime_coordinates="semantic/composed.dds[600,384,512]",
            direction="runtime_to_fact_partial",
            invertibility="MANY_TO_ONE;MARGIN_BANDS_REQUIRE_SCORER_MARGIN_INPUT",
            fact_scalar_count=600 * 5 * 6,
            runtime_coordinate_count=600 * 384 * 512,
            loss_account={
                "horizontal_vertical_cuts": "DERIVABLE_ON_RESOLVED_ROLE_CELLS",
                "margin_band_counts": "ABSENT_FROM_E2_PACKET",
                "missing_margin_fact_scalars": 600 * 5 * 4,
                "price_transfer": "FORBIDDEN",
            },
        ),
        CoordinateRelation(
            relation_id="pair_screw_to_pose_runtime",
            fact_coordinates="pair[600] x pair_screw[6] float64 bit patterns",
            runtime_coordinates="E2 counted pose member",
            direction="none",
            invertibility="ABSENT",
            fact_scalar_count=600 * 6,
            runtime_coordinate_count=0,
            loss_account={
                "fact_pose_scalars": 3_600,
                "e2_packet_pose_bytes": 0,
                "nested_pose6_status": "CONSUMED_BEFORE_EXPORT_NOT_COUNTED_MEMBER",
                "xi_to_chart_jacobian": "ABSENT",
                "price_transfer": "FORBIDDEN",
            },
        ),
        CoordinateRelation(
            relation_id="e2_chart_to_sdwl1",
            fact_coordinates="no declared SDWL1 chart coordinate",
            runtime_coordinates=(
                "base/chart.ddb anchors[3600]+gradients[7200]+residuals[691200]"
            ),
            direction="none",
            invertibility="ABSENT",
            fact_scalar_count=0,
            runtime_coordinate_count=702_000,
            loss_account={
                "unmapped_runtime_chart_coordinates": 702_000,
                "price_transfer": "FORBIDDEN",
            },
        ),
        CoordinateRelation(
            relation_id="e2_palette_prosody_to_sdwl1",
            fact_coordinates="no emitted SDWL1 amplitude/contrast/channel-energy field",
            runtime_coordinates="manifest.output.palette_rgb_u8[6,3]",
            direction="none",
            invertibility="ABSENT",
            fact_scalar_count=0,
            runtime_coordinate_count=18,
            loss_account={
                "unmapped_runtime_palette_coordinates": 18,
                "uint8_surviving_projection_required": True,
                "price_transfer": "FORBIDDEN",
            },
        ),
    )
    return {
        "schema": BRIDGE_SCHEMA,
        "status": "LOSS_ACCOUNTED_NOT_INVERTIBLE",
        "first_rung": True,
        "pair_count": PAIR_COUNT,
        "source_hashes": [row.to_dict() for row in source_hashes],
        "sdwl1": {
            "semantic_shape": [600, 11, 8],
            "declared_scalar_fact_count": 45_600,
            "padding_scalars": 7_200,
        },
        "e2": {
            "semantic_role_plane_shape": [600, 384, 512],
            "semantic_role_coordinates": 117_964_800,
            "chart_coordinates": 702_000,
            "counted_pose_member_coordinates": 0,
        },
        "relations": [row.to_dict() for row in relations],
        "pricing": {
            "fact_side": "SDWL1 exact/member bytes and fact-local costates",
            "runtime_side": "receiver-closed E2 coordinate costates",
            "cross_side": (
                "null unless relation invertibility=EXACT and all source hashes match"
            ),
            "u1_ladder": "BLOCKED_ON_LOSSY_RELATIONS",
            "mode_rerace": "BLOCKED_ON_LOSSY_RELATIONS",
            "xi_direction": "BLOCKED_E2_POSE_MEMBER_AND_XI_TO_CHART_JACOBIAN_ABSENT",
        },
        "verdict_scope": (
            "current SDWL1 aggregate grammar versus E2 runtime packet only; "
            "description and compact-inverse families remain open"
        ),
    }


def e2_role_codes_to_partial_sdwl1_classes(
    role_codes: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Map resolved E2 role codes to SDWL1 class IDs and count code-0 loss."""

    codes = np.asarray(role_codes)
    if codes.ndim != 3 or codes.shape != (PAIR_COUNT, 384, 512):
        raise AnalyticAtlasError("E2 role plane must have exact shape (600,384,512)")
    if not np.issubdtype(codes.dtype, np.integer) or (
        codes.size and (int(codes.min()) < 0 or int(codes.max()) > 5)
    ):
        raise AnalyticAtlasError("E2 role codes must be integer values 0..5")
    # E2 paint order: background, Undrivable, Road, Lane, Movable, MyCar.
    lookup = np.asarray([-1, 2, 0, 1, 3, 4], dtype=np.int8)
    mapped = lookup[codes]
    return mapped, int(np.count_nonzero(mapped < 0))


def _number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AnalyticAtlasError("non-finite lambda input")
    return result


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return sum(
        a * b for a, b in zip(left_centered, right_centered, strict=True)
    ) / denominator


def _ndcg(predicted: Sequence[float], relevance: Sequence[float], k: int) -> float | None:
    if not predicted or len(predicted) != len(relevance):
        return None
    limit = min(k, len(predicted))

    def dcg(order: Iterable[int]) -> float:
        return sum(
            relevance[index] / math.log2(rank + 2.0)
            for rank, index in enumerate(list(order)[:limit])
        )

    predicted_order = sorted(
        range(len(predicted)), key=lambda index: predicted[index], reverse=True
    )
    ideal_order = sorted(
        range(len(relevance)), key=lambda index: relevance[index], reverse=True
    )
    ideal = dcg(ideal_order)
    return None if ideal == 0.0 else dcg(predicted_order) / ideal


def build_ddm_lambda_bundle(
    *,
    atlas: Mapping[int, Mapping[str, Any]],
    v19: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Produce the only live DDM pair/site lambda rows from exact joined factors.

    The g3 atlas must be complete for all 600 pairs even though the extant v19
    exact receiver replay covers eight selected pairs.  Missing v19 rows remain
    explicit, counted, and inert; they are not filled from the legacy organ.
    """

    expected = set(range(PAIR_COUNT))
    observed = set(atlas)
    if observed != expected:
        missing = len(expected - observed)
        extra = len(observed - expected)
        raise AnalyticAtlasError(
            f"lambda producer requires exact n600 g3 atlas; missing={missing} extra={extra}"
        )
    for required in ("g3", "v19", "g3_full_atlas"):
        if not _is_sha256(str(source_hashes.get(required, ""))):
            raise AnalyticAtlasError(f"lambda producer lacks fresh {required} hash")

    pairs: list[dict[str, Any]] = []
    sites: list[dict[str, Any]] = []
    measured_rows = list(v19["pair_recursion_ledger"]["rows"])
    for measured in measured_rows:
        pair_id = int(measured["source_pair_id"])
        g3 = atlas[pair_id]
        gap = _number(g3["costate_signal"]["lambda_proxy_score_debt"])
        allocated = _number(g3["allocated_bytes"]["allocated_bytes"])
        geometry = g3["evaluator_response_geometry"][
            "joint_cone_summary_diagnostic_only"
        ]
        visibility = max(
            0.0, min(1.0, 1.0 - _number(geometry["empty_cone_fraction"]))
        )
        changed = max(1, int(measured["changed_argmax_cells"]))
        realizability = max(
            0.0, min(1.0, int(measured["helpful_flips"]) / changed)
        )
        delta_s = realized_pair_distortion_delta(
            d_seg_before=_number(measured["d_seg_before"]),
            d_seg_after=_number(measured["d_seg_after"]),
            d_pose_before=_number(measured["d_pose_before"]),
            d_pose_after=_number(measured["d_pose_after"]),
        )
        realized_closure = max(0.0, -delta_s)
        d2 = min(1.0, realized_closure / gap) if gap > 0.0 else 0.0
        byte_price = 1.0 / max(allocated, 1.0)
        pre_d2 = ddm_joint_costate(
            gap, visibility, realizability, byte_price, 1.0
        )
        value = ddm_joint_costate(gap, visibility, realizability, byte_price, d2)
        pairs.append(
            {
                "pair_index": pair_id,
                "exact_gap": gap,
                "visibility": visibility,
                "uint8_realizability": realizability,
                "allocated_baseline_bytes": allocated,
                "candidate_shared_bytes": measured.get("per_pair_byte_allocation"),
                "candidate_shared_byte_status": "OWED_NOT_INVENTED",
                "byte_price": byte_price,
                "realized_distortion_delta_s": delta_s,
                "dual_tolerance_d2": d2,
                "lambda_pre_d2": pre_d2,
                "lambda_d2": value,
                "validity_radius": d2,
                "validity_kind": (
                    "MEASURED_REALIZED_CLOSURE_FRACTION_NOT_UNIVERSAL"
                ),
                "epistemic_status": "MEASURED_PLUS_DERIVED",
                "first_rung": True,
            }
        )

        flips = g3["segmentation"]["class_flip_counts"]
        total_flips = max(1, sum(int(count) for count in flips.values()))
        for stratum, row in measured["per_stratum"].items():
            stratum_gap = gap * int(flips.get(stratum, 0)) / total_flips
            before = max(1, int(row["errors_before"]))
            closure = max(
                0, int(row["errors_before"]) - int(row["errors_after"])
            )
            site_realizability = min(1.0, closure / before)
            sites.append(
                {
                    "pair_index": pair_id,
                    "stratum": stratum,
                    "exact_gap": stratum_gap,
                    "visibility": visibility,
                    "uint8_realizability": site_realizability,
                    "byte_price": byte_price,
                    "dual_tolerance_d2": d2,
                    "lambda_d2": ddm_joint_costate(
                        stratum_gap,
                        visibility,
                        site_realizability,
                        byte_price,
                        d2,
                    ),
                    "errors_before": int(row["errors_before"]),
                    "errors_after": int(row["errors_after"]),
                    "allocation_status": (
                        "G3_CLASS_FLIP_SHARE_DERIVED; "
                        "V19_SHARED_BYTES_UNALLOCATED"
                    ),
                    "first_rung": True,
                }
            )

    pairs.sort(key=lambda row: row["lambda_d2"], reverse=True)
    sites.sort(key=lambda row: row["lambda_d2"], reverse=True)
    predicted = [row["lambda_pre_d2"] for row in pairs]
    realized = [max(0.0, -row["realized_distortion_delta_s"]) for row in pairs]
    spearman = _pearson(_rankdata(predicted), _rankdata(realized))
    missing_pair_ids = sorted(expected - {int(row["pair_index"]) for row in pairs})
    backtest = {
        "schema": "ddm_factorized_adjoint_backtest.v2",
        "n_pairs": len(pairs),
        "predictor": (
            "g3_gap * g3_usable_support * v19_helpful/changed "
            "* 1/g3_allocated_bytes"
        ),
        "target": (
            "positive exact v19 Seg/Pose distortion closure; shared rate bytes excluded"
        ),
        "spearman_rho": spearman,
        "ndcg_at_4": _ndcg(predicted, realized, 4),
        "positive_realized_pairs": sum(value > 0.0 for value in realized),
        "verdict": (
            "FACTORIZED_ADJOINT_VALID_ON_THIS_EIGHT_PAIR_BACKTEST"
            if spearman is not None and spearman >= 0.5
            else "FACTORIZED_ADJOINT_INVALID_OR_UNIDENTIFIABLE"
        ),
        "verdict_scope": "INSTANCE:V19_EIGHT_PAIR_EXACT_RECEIVER_REPLAY_X_G3_ATLAS",
        "first_rung": True,
    }
    payload = {
        "pair_rows": pairs,
        "site_rows": sites,
        "backtest": backtest,
        "n600_g3_input_complete": True,
        "v19_exact_pair_count": len(pairs),
        "missing_exact_pair_lambda_count": len(missing_pair_ids),
    }
    return {
        "schema": LAMBDA_SCHEMA,
        "producer": (
            "tac.optimization.scorer_analytic_atlas.build_ddm_lambda_bundle"
        ),
        "status": (
            "PARTIAL_EXACT_V19_BACKTEST"
            if missing_pair_ids
            else "N600_EXACT_COMPLETE"
        ),
        "pair_count_required": PAIR_COUNT,
        "n600_g3_input_complete": True,
        "pair_rows": pairs,
        "site_rows": sites,
        "backtest": backtest,
        "missing_exact_pair_lambda_count": len(missing_pair_ids),
        "missing_exact_pair_lambda_ids_sha256": _payload_sha256(missing_pair_ids),
        "unconsumed_missing_pairs_counted_inert": True,
        "input_hashes": {
            key: str(value) for key, value in sorted(source_hashes.items())
        },
        "content_sha256": _payload_sha256(payload),
        "first_rung": True,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "actuation": "NONE",
        "evidence_axis": EVIDENCE_AXIS,
    }


__all__ = [
    "BRIDGE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CURRENT_PORTS",
    "EVIDENCE_AXIS",
    "FACTOR_SCHEMA",
    "LAMBDA_SCHEMA",
    "PAIR_COUNT",
    "SCHEMA",
    "AnalyticAtlasError",
    "AnalyticFactor",
    "AtlasCheckpoint",
    "ConsumptionStatus",
    "CoordinateRelation",
    "FactorStatus",
    "NonAdditivePool",
    "SourceHashStamp",
    "TensorArtifactRef",
    "build_ddm_lambda_bundle",
    "build_factor",
    "build_gaze_factor",
    "build_manifest",
    "build_r_null_band_certificate",
    "build_sdwl1_e2_coordinate_bridge",
    "compose_jacobian_factors",
    "derive_batchnorm_expected_stats",
    "derive_bn_silu_contrast",
    "derive_kernel_dft_bank",
    "derive_se_gate_closed_form",
    "e2_role_codes_to_partial_sdwl1_classes",
    "evaluate_batchnorm_factor",
    "evaluate_se_gate_factor",
    "project_gaze_onto_axis",
    "pull_back_gaze",
    "require_total_gaze_coverage",
    "write_stage_checkpoint",
]
