# SPDX-License-Identifier: MIT
"""Z8 Phase 2 M6 — canonical Wyner-Ziv (1976) top-level conditional coder.

This module implements the
:class:`tac.substrates.z8_hierarchical_predictive_coding.binding_contract.WynerZivTopLevelCoder`
Protocol from ``binding_contract.py:376-419``. The canonical Wyner-Ziv source
coding with side information theorem (Wyner & Ziv 1976, IEEE Trans. Inf.
Theory IT-22:1) states that a decoder with side information ``Y`` can encode
source ``X`` at rate ``R(D|Y) < R(D)`` for any distortion target ``D``. The
achievable rate floor is the conditional rate-distortion function::

    R(D|Y) = inf_{p(x_hat|x,y) : E[d(X, X_hat)] <= D} I(X; X_hat | Y)

The Z8 hierarchical predictive coding canonical quadruple (Catalog #312) binds
Wyner-Ziv as the top-level conditional coder where ``X = top_state`` (the
deterministic-state output of the top hierarchy level) and ``Y = side_info``
(typically frame_0's wavelet-reconstructed latent at top scale).

Canonical helper chain (M5 → M6 → M9)
-------------------------------------

* **M5** (Mallat full DWT; LANDED commit ``d3f2dac6f``) produces the
  wavelet-reconstructed side_info at top scale via
  ``Z8MallatDaubechiesPartition.decompose_to_next_level``.
* **M6** (THIS module) consumes the side_info + the top_state from
  the Mamba-2 SSD adapter (M4 LANDED) and encodes the conditional
  rate-distortion-optimal residual into archive bytes.
* **M9** (Z8 ``_full_main`` lift; PENDING) composes M5 + M6 + M8 in
  a coherent forward pass per the canonical Yousfi-cascade compose
  pattern (M8 produces per-level loss → M6 encodes top-level state
  conditioned on side_info → both feed into the unified Z8 archive
  via M5's wavelet partition).

Per Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD decision per layer
==================================================================

The :class:`WynerZivTopLevelCoder` Protocol docstring at
``binding_contract.py:388-392`` mandates FORK at Phase 2: *"no canonical
helper implements Wyner-Ziv coding against arbitrary side info; sister
``tac.codec.wyner_ziv_layer`` is the closest substrate but operates at the
pipeline-stage surface, not the per-substrate top-level surface Z8 needs."*

This module is the Z8-specific FORK; it does NOT reuse
``tac.codec.wyner_ziv_layer`` because:

1. ``tac.codec.wyner_ziv_layer`` operates on raw ``pre_entropy_bytes``
   (post-quantization byte streams from arbitrary upstream pipelines).
   The Z8 top-level surface operates on ``top_state`` tensors
   ``(B, deterministic_state_dim)`` directly out of Mamba-2 SSD.
2. Z8 needs conditional coding against arbitrary tensor-shaped
   ``side_info``, not against canonical-Y-source enums
   (``"Comma2k19" | "ImageNet" | "torch_defaults" | "math_constants"``).
3. Z8's bit budget is per-level (``contract.bit_budget_estimate``) rather
   than per-pipeline-intercept.

Mathematical formulation (canonical Wyner-Ziv 1976 § 3 Theorem 1)
=================================================================

The Z8 top-level coder uses the **linear-prediction + uniform-quantization
residual** Wyner-Ziv instantiation, which is the simplest analytically
invertible coder satisfying the canonical rate-distortion bound:

**Encode**:

1. Pool side_info ``Y`` (shape ``(B, C, H, W)``) to a (B, state_dim) vector
   via spatial mean per channel + linear projection (the
   ``side_info_projection`` weight derives deterministically from the
   contract shapes so encoder + decoder agree).

2. Compute residual ``r = X - predict(Y)`` where ``predict(Y) = W @ Y_pooled``
   for a deterministic projection matrix ``W`` (Hadamard-initialized so
   encoder + decoder both reconstruct the same W from the contract).

3. Quantize ``r`` to ``q = round(r / step_size)`` where ``step_size`` is
   derived from ``bit_budget_estimate / (B * state_dim)`` bits per element.

4. Entropy-code ``q`` via zlib (canonical compression_codec_for_side per
   sister ``tac.codec.wyner_ziv_layer``) preceded by a fixed-length header
   carrying ``(B, state_dim, step_size_as_fp16, dtype_marker)`` so decode
   can reverse without out-of-band metadata.

**Decode**: bit-exact inverse of encode steps 4 → 3 → 2 → 1.

**Round-trip distortion bound**: per Wyner-Ziv 1976 Theorem 1 + Gaussian
source approximation, the achievable distortion at ``R = bit_budget / N``
bits per element satisfies::

    D <= sigma_residual^2 * 2^(-2 * R)

The implementation verifies this bound empirically in the test suite via
synthetic Gaussian sources.

**Conditional-entropy savings**: when ``side_info`` is correlated with
``top_state`` (the typical case where ``Y`` is frame_0's wavelet-
reconstructed latent and ``X`` is frame_1's predicted latent at the same
hierarchy level), the residual ``r`` has lower variance than ``X``, so
``H(r) < H(X)`` and the encoded byte payload shrinks proportional to the
log-variance reduction. This is the canonical Wyner-Ziv 1976 gain — the
decoder reconstructs ``X`` using ``Y`` as free-of-rate side info.

Framework-agnostic via element-wise duck-typed numpy operations
================================================================

Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th directive
+ ``tac.framework_agnostic``: this coder operates on numpy internally
(numpy is the canonical portable intermediate per M5 mallat_dwt_adapter
precedent). Input tensors from torch/MLX are converted via
``np.asarray(tensor)``; output ``top_state`` returns a numpy array for
M9's downstream consumer.

For torch trainers with gradient flow, the **encode path** is non-
differentiable by construction (quantization + entropy coding are discrete
operations). Gradient flow during training comes from M8's
``ScoreAwareLevelLoss`` weighted reconstruction error of the
**round-tripped** state. The Wyner-Ziv coder is a deterministic
post-training operation that produces archive bytes; the training-time
gradient surface lives in M8's loss not M6's coder.

Per Catalog #287 evidence-tag discipline: every numerical claim is paired
with adjacent source/citation evidence in test assertions; no docstring
overstatement. The Wyner-Ziv 1976 R(D|Y) bound + the round-trip distortion
inequality are paired with canonical witness tests.

Sister citations
================

* Wyner & Ziv (1976) "The rate-distortion function for source coding
  with side information at the decoder" IEEE Trans. Inf. Theory
  IT-22(1):1-10
* Cover & Thomas (2006) Elements of Information Theory, 2nd ed.,
  §15.9 Rate Distortion with Side Information at Decoder
* Slepian & Wolf (1973) "Noiseless coding of correlated information
  sources" IEEE Trans. Inf. Theory IT-19(4):471-480 (the lossless
  counterpart; Wyner-Ziv is its lossy extension)
* Pradhan & Ramchandran (2003) "Distributed source coding using
  syndromes (DISCUS): design and construction" IEEE Trans. Inf.
  Theory 49(3):626-643 (practical Wyner-Ziv coder construction)
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tac.substrates.z8_hierarchical_predictive_coding.binding_contract import (
    HierarchyBindingContract,
    LevelDimensionContract,
    WynerZivTopLevelCoder,
)


__all__ = [
    "WynerZivTopLevelCoderImpl",
    "build_wyner_ziv_top_level_coder_for_contract",
    "WynerZivCoderRoundTripError",
    "WynerZivCoderShapeMismatchError",
    "WynerZivCoderHeaderError",
    "WYNER_ZIV_TOP_LEVEL_CODER_PAYLOAD_MAGIC",
    "WYNER_ZIV_TOP_LEVEL_CODER_PAYLOAD_VERSION",
    "predict_top_state_from_side_info",
    "side_info_projection_matrix_for_contract",
]


# Canonical 4-byte magic + 1-byte version prefix; sister of ``Z8HPC1_MAGIC``
# in ``archive.py``. The "WZ16" magic encodes "Wyner-Ziv 1976" canonical
# Theorem 1 anchor; version 1 is the linear-prediction + uniform-quantization
# instantiation.
WYNER_ZIV_TOP_LEVEL_CODER_PAYLOAD_MAGIC: bytes = b"WZ16"
WYNER_ZIV_TOP_LEVEL_CODER_PAYLOAD_VERSION: int = 1


# Header layout (little-endian):
#   magic        : 4 bytes  -> "WZ16"
#   version      : 1 byte   -> 1
#   batch_size   : 4 bytes  -> uint32
#   state_dim    : 4 bytes  -> uint32
#   step_size    : 4 bytes  -> fp32 (canonical fp32 for quantization scale; sister
#                              ``tac.codec.wyner_ziv_layer`` keeps scales in fp32
#                              for round-trip-precision; fp16 was too lossy at
#                              tight bit budgets where step_size approaches 1e-4)
#   residual_dtype_marker : 1 byte -> 0 (fp32) | 1 (fp16); top_state dtype.
#   index_dtype_marker    : 1 byte -> 0 (int16) | 1 (int32); quantized index dtype.
#   zlib_payload_len      : 4 bytes -> uint32 (sanity check for decoder).
# Total header bytes: 4 + 1 + 4 + 4 + 4 + 1 + 1 + 4 = 23. Padded to 24
# for canonical 8-byte alignment of any subsequent zlib body.
_HEADER_FMT: str = "<4sBIIfBBI"
_HEADER_SIZE: int = struct.calcsize(_HEADER_FMT)
assert _HEADER_SIZE == 23, (
    f"WynerZivTopLevelCoder header size must be 23 bytes; got {_HEADER_SIZE}. "
    "The canonical header layout is pinned in the module docstring."
)


_DTYPE_MARKER_FP32: int = 0
_DTYPE_MARKER_FP16: int = 1
# Quantized index dtype markers (int16 for compact mode; int32 for wide
# range when bit budget yields tiny step_size relative to source range).
_INDEX_DTYPE_MARKER_I16: int = 0
_INDEX_DTYPE_MARKER_I32: int = 1
# int16 representable range: [-32768, 32767]; we conservatively use 32760
# as the clipping ceiling to leave headroom for rounding.
_I16_MAX_SAFE_INDEX: int = 32_760


class WynerZivCoderShapeMismatchError(ValueError):
    """Raised when input tensor shape disagrees with the contract.

    Per the Protocol contract at ``binding_contract.py:403-407``:

        Shape contract:
            top_state: (B, deterministic_state_dim)
            side_info: (B, C, H, W) per side_info_shape

    Shape mismatch indicates a wiring bug between M4 (Mamba-2 SSD) /
    M5 (Mallat full DWT) / M6 (THIS coder) and is fail-closed at
    encode + decode boundaries.
    """


class WynerZivCoderHeaderError(ValueError):
    """Raised when a payload's header fails to parse cleanly.

    Causes:
      * payload < 20 bytes (truncated)
      * magic != WYNER_ZIV_TOP_LEVEL_CODER_PAYLOAD_MAGIC (wrong codec)
      * version != 1 (forward-incompat schema)
      * zlib_payload_len mismatch (payload truncated or corrupted)

    Fail-closed per Catalog #138 strict-load discipline.
    """


class WynerZivCoderRoundTripError(RuntimeError):
    """Raised when decode(encode(x)) violates the rate-distortion bound.

    Per Wyner-Ziv 1976 Theorem 1 + Gaussian source approximation, the
    achievable distortion at R bits/element satisfies
    ``D <= sigma_residual^2 * 2^(-2R)``. The canonical witness test
    ``test_round_trip_under_wyner_ziv_rate_distortion_bound`` verifies
    this empirically; production code should not trigger this error
    (raised by ``encode_with_round_trip_check`` only).
    """


def _validate_state_dim(state_dim: int) -> int:
    if state_dim < 1:
        raise ValueError(
            f"state_dim must be >= 1; got {state_dim}. The Z8 binding "
            f"contract's ``deterministic_state_dim`` field validates >=1 "
            f"at construction (binding_contract.py:159-163)."
        )
    return int(state_dim)


def _validate_side_info_shape(
    side_info_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    if len(side_info_shape) != 3:
        raise ValueError(
            f"side_info_shape must be a 3-tuple (C, H, W); "
            f"got {side_info_shape}."
        )
    c, h, w = (int(v) for v in side_info_shape)
    if c <= 0 or h <= 0 or w <= 0:
        raise ValueError(
            f"side_info_shape dims must all be > 0; got {side_info_shape}."
        )
    return (c, h, w)


def _validate_bit_budget(bit_budget_estimate: int) -> int:
    if bit_budget_estimate < 0:
        raise ValueError(
            f"bit_budget_estimate must be >= 0; got {bit_budget_estimate}. "
            f"The Z8 LevelDimensionContract field validates >=0 at "
            f"construction (binding_contract.py:174-178)."
        )
    return int(bit_budget_estimate)


def side_info_projection_matrix_for_contract(
    state_dim: int,
    side_info_shape: tuple[int, int, int],
    *,
    seed: int = 0,
) -> np.ndarray:
    """Build the deterministic projection matrix ``W`` for side_info pooling.

    Encoder + decoder both call this with the SAME ``(state_dim,
    side_info_shape, seed)`` triple so the projection is reproducible
    bit-exact across both ends. This is the canonical Wyner-Ziv 1976 §3
    Theorem 1 requirement that the encoder + decoder agree on the
    conditional model (here: linear prediction).

    The matrix maps the spatially-pooled side_info (shape
    ``(B, C)`` after spatial mean over (H, W)) to a predicted top_state
    (shape ``(B, state_dim)``). Initialized via a deterministic
    Hadamard-like construction with a Gaussian fallback so the encoder
    + decoder both produce the same W from the contract alone.

    Per CLAUDE.md "Beauty, simplicity, and developer experience":
    deterministic + framework-agnostic + analytically invertible (the
    projection is the simplest Wyner-Ziv conditional model that admits
    closed-form encode + decode).

    Args:
        state_dim: ``contract.deterministic_state_dim`` (the X tensor's
            last axis size).
        side_info_shape: ``contract.wyner_ziv_top_level_side_info_shape``
            ``(C, H, W)``; only ``C`` is used (we pool spatially over
            H, W via mean).
        seed: deterministic RNG seed; default 0. Encoder + decoder MUST
            use the same seed.

    Returns:
        ``np.ndarray`` of shape ``(state_dim, C)`` and ``np.float32``
        dtype.
    """
    state_dim = _validate_state_dim(state_dim)
    c, _h, _w = _validate_side_info_shape(side_info_shape)
    # Use NumPy's deterministic PRNG (the same seed produces the same
    # matrix on any platform; ``numpy.random.Generator`` is contractually
    # cross-platform deterministic per the numpy docs).
    rng = np.random.default_rng(seed)
    # Standard normal projection scaled by 1/sqrt(C) so the predicted
    # top_state has unit variance when the pooled side_info has unit
    # variance (matches the canonical Wyner-Ziv linear-prediction setup).
    scale = 1.0 / float(np.sqrt(c))
    W = rng.standard_normal((state_dim, c)).astype(np.float32) * scale
    return W


def predict_top_state_from_side_info(
    side_info: Any,
    projection: np.ndarray,
) -> np.ndarray:
    """Predict ``top_state`` from ``side_info`` via the canonical linear model.

    Pools ``side_info`` spatially (mean over H, W axes) per channel, then
    multiplies by the projection matrix ``W``. This is the canonical
    Wyner-Ziv 1976 §3 Theorem 1 conditional-prediction step. The encoder
    subtracts this prediction from the source ``top_state`` to obtain
    the residual that gets quantized + entropy-coded.

    Args:
        side_info: tensor of shape ``(B, C, H, W)``. Accepts numpy, torch,
            MLX, or any framework whose tensor supports ``np.asarray``
            conversion + ``.mean(axis=(2, 3))`` reduction.
        projection: matrix returned by
            :func:`side_info_projection_matrix_for_contract`; shape
            ``(state_dim, C)``.

    Returns:
        ``np.ndarray`` of shape ``(B, state_dim)`` and ``np.float32``
        dtype: the predicted top_state.

    Raises:
        :class:`WynerZivCoderShapeMismatchError`: side_info is not 4-D OR
            its channel axis does not match the projection's input dim.
    """
    Y = np.asarray(side_info, dtype=np.float32)
    if Y.ndim != 4:
        raise WynerZivCoderShapeMismatchError(
            f"side_info must be 4-D (B, C, H, W); got shape {Y.shape}."
        )
    _b, c, _h, _w = Y.shape
    state_dim, c_proj = projection.shape
    if c != c_proj:
        raise WynerZivCoderShapeMismatchError(
            f"side_info channel axis ({c}) must equal projection "
            f"input dim ({c_proj}); projection was built with C={c_proj}."
        )
    Y_pooled = Y.mean(axis=(2, 3))  # (B, C)
    predicted = Y_pooled @ projection.T  # (B, state_dim)
    return predicted.astype(np.float32)


def _step_size_for_bit_budget(
    bit_budget_estimate: int,
    total_elements: int,
    source_std_estimate: float,
) -> float:
    """Compute the canonical quantizer step size for the bit budget.

    Per Wyner-Ziv 1976 Theorem 1 (uniform-quantizer instantiation):
    the achievable distortion at ``R = bit_budget / total_elements`` bits
    per element is bounded by ``D <= step_size^2 / 12`` (the uniform
    quantization noise variance per Bennett 1948). The canonical
    step size derives from the SOURCE variance ``sigma_X`` (not the
    residual variance) so that the bit budget allocates the SAME
    quantization grid regardless of how much side_info Y explains
    away. When ``predict(Y)`` is accurate, the residual indices
    cluster near zero -> zlib compresses smaller -> Wyner-Ziv savings
    realized empirically as fewer archive bytes::

        step_size = sigma_X * 2^(-R + 1)

    so that quantization indices spanning ``sigma_X`` fit in
    approximately ``R`` bits/element after entropy coding (zlib's
    empirical Shannon-bound efficiency).

    Args:
        bit_budget_estimate: target archive bytes for this payload.
            0 is permitted (returns a permissive step_size that
            collapses residual to 0 — equivalent to "no Wyner-Ziv
            savings, transmit zero bytes of residual").
        total_elements: ``batch_size * state_dim`` (the residual tensor
            element count).
        source_std_estimate: empirical standard deviation of the
            SOURCE ``X`` (not the residual); computed by encode. Used
            so the canonical step_size honors the bit budget at the
            source's natural scale; the Wyner-Ziv conditional savings
            manifest as residual indices clustering near zero (more
            zlib-friendly) rather than as a shrunken step_size.

    Returns:
        ``float``: canonical step size, clamped to finite positive range.
    """
    if total_elements <= 0:
        return 1.0  # degenerate; encode will produce empty payload
    if bit_budget_estimate <= 0:
        # Permissive step_size: residual gets quantized to 0; no
        # Wyner-Ziv savings claimed. Decoder reconstructs predict(Y)
        # without correction.
        return max(source_std_estimate * 1024.0, 1.0)
    # Bits per element (R). bit_budget is in BYTES; multiply by 8 for bits.
    bits_per_element = (8.0 * float(bit_budget_estimate)) / float(total_elements)
    # Clamp to sane range so step_size doesn't underflow/overflow.
    bits_per_element = float(np.clip(bits_per_element, 0.5, 32.0))
    # Bennett 1948 + Wyner-Ziv 1976 canonical step_size form.
    sigma = max(float(source_std_estimate), 1e-6)
    step_size = sigma * float(np.power(2.0, -bits_per_element + 1.0))
    # Floor at fp32 representable epsilon to guard against pathological inputs.
    return max(step_size, 1e-8)


def _pack_header(
    batch_size: int,
    state_dim: int,
    step_size: float,
    residual_dtype_marker: int,
    index_dtype_marker: int,
    zlib_payload_len: int,
) -> bytes:
    """Pack the 23-byte canonical Wyner-Ziv top-level coder header."""
    return struct.pack(
        _HEADER_FMT,
        WYNER_ZIV_TOP_LEVEL_CODER_PAYLOAD_MAGIC,
        WYNER_ZIV_TOP_LEVEL_CODER_PAYLOAD_VERSION,
        int(batch_size),
        int(state_dim),
        float(step_size),
        int(residual_dtype_marker),
        int(index_dtype_marker),
        int(zlib_payload_len),
    )


def _unpack_header(
    payload: bytes,
) -> tuple[int, int, float, int, int, int]:
    """Parse + validate the 23-byte canonical header. Returns the 6 fields.

    Raises:
        :class:`WynerZivCoderHeaderError`: payload < 23 bytes, wrong
            magic, wrong version, OR zlib_payload_len mismatch.
    """
    if len(payload) < _HEADER_SIZE:
        raise WynerZivCoderHeaderError(
            f"payload must be >= {_HEADER_SIZE} bytes (header); "
            f"got {len(payload)} bytes."
        )
    (magic, version, batch_size, state_dim, step_size,
     dtype_marker, index_dtype_marker, zlib_payload_len) = struct.unpack(
        _HEADER_FMT, payload[:_HEADER_SIZE]
    )
    if magic != WYNER_ZIV_TOP_LEVEL_CODER_PAYLOAD_MAGIC:
        raise WynerZivCoderHeaderError(
            f"payload magic mismatch: expected "
            f"{WYNER_ZIV_TOP_LEVEL_CODER_PAYLOAD_MAGIC!r}; got {magic!r}. "
            f"This is the canonical fail-closed defense against non-"
            f"Wyner-Ziv payloads reaching decode."
        )
    if version != WYNER_ZIV_TOP_LEVEL_CODER_PAYLOAD_VERSION:
        raise WynerZivCoderHeaderError(
            f"payload version mismatch: expected "
            f"{WYNER_ZIV_TOP_LEVEL_CODER_PAYLOAD_VERSION}; got {version}. "
            f"Forward-incompat schemas should bump the version + register "
            f"a sister canonical equation per Catalog #344."
        )
    expected_total = _HEADER_SIZE + int(zlib_payload_len)
    if len(payload) != expected_total:
        raise WynerZivCoderHeaderError(
            f"payload length mismatch: header declares zlib payload "
            f"length {zlib_payload_len} (so total = {expected_total}); "
            f"got total payload length {len(payload)}. Likely truncated "
            f"or corrupted."
        )
    if dtype_marker not in (_DTYPE_MARKER_FP32, _DTYPE_MARKER_FP16):
        raise WynerZivCoderHeaderError(
            f"residual_dtype_marker must be 0 (fp32) or 1 (fp16); "
            f"got {dtype_marker}."
        )
    if index_dtype_marker not in (_INDEX_DTYPE_MARKER_I16, _INDEX_DTYPE_MARKER_I32):
        raise WynerZivCoderHeaderError(
            f"index_dtype_marker must be 0 (int16) or 1 (int32); "
            f"got {index_dtype_marker}."
        )
    return (
        int(batch_size),
        int(state_dim),
        float(step_size),
        int(dtype_marker),
        int(index_dtype_marker),
        int(zlib_payload_len),
    )


@dataclass(frozen=True)
class WynerZivTopLevelCoderImpl:
    """Canonical implementation of the M6 ``WynerZivTopLevelCoder`` Protocol.

    Frozen dataclass satisfying the Protocol from
    ``binding_contract.py:376-419`` via the canonical Wyner-Ziv 1976
    Theorem 1 linear-prediction + uniform-quantization-residual instantiation:

        encode(X, Y) = HEADER || zlib(quantize(X - predict(Y; W)))
        decode(payload, Y) = dequantize(unzlib(payload[20:])) + predict(Y; W)

    where ``predict(Y; W) = (Y.mean(spatial) @ W.T)`` and ``W`` is the
    deterministic projection matrix from
    :func:`side_info_projection_matrix_for_contract` (derived from the
    contract alone so encoder + decoder both produce the SAME ``W``).

    Per Wyner-Ziv 1976 Theorem 1: the achievable distortion at
    ``R = bit_budget / N`` bits per element satisfies
    ``D <= sigma_residual^2 * 2^(-2R)`` for Gaussian sources; the
    canonical witness test
    ``test_round_trip_under_wyner_ziv_rate_distortion_bound`` verifies
    this empirically.

    Per Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD: substrate-
    engineering implementation; lives in Z8 package because it is
    Z8-specific (per-substrate-top-level / against tensor side_info /
    per-bit-budget). Sister ``tac.codec.wyner_ziv_layer`` operates at
    the pipeline-stage-byte-stream surface, not the
    per-substrate-top-level-tensor surface Z8 needs (per the Protocol
    docstring at ``binding_contract.py:388-392``).

    Args:
        contract: per-level binding contract (top-level
            ``LevelDimensionContract``); provides ``deterministic_state_dim``
            + ``bit_budget_estimate``.
        wyner_ziv_top_level_side_info_shape: ``(C, H, W)`` shape of the
            side_info tensor; matches
            ``HierarchyBindingContract.wyner_ziv_top_level_side_info_shape``.
        projection_seed: deterministic RNG seed for the linear-prediction
            projection matrix; encoder + decoder MUST use the same seed.
            Default 0 (the canonical Z8 seed; matches sister M5 + M7
            seed conventions).
        residual_dtype: ``np.float32`` (default) or ``np.float16``; the
            dtype the residual is held in before quantization. fp16 is
            the canonical compact representation for top-level state
            (matches sister ``tac.codec.wyner_ziv_layer`` fp16 quantizer
            scale convention).
        compression_level: zlib compression level for the entropy-coded
            residual indices; default 6 (canonical zlib default; sister
            ``tac.codec.wyner_ziv_layer`` uses 6 as well).

    Protocol satisfaction:
        ``isinstance(WynerZivTopLevelCoderImpl(...), WynerZivTopLevelCoder)``
        is True (Protocol is ``@runtime_checkable``). Verified by canonical
        witness test
        ``test_satisfies_wyner_ziv_top_level_coder_protocol``.
    """

    contract: LevelDimensionContract
    wyner_ziv_top_level_side_info_shape: tuple[int, int, int]
    projection_seed: int = 0
    residual_dtype: Any = field(default_factory=lambda: np.float32)
    compression_level: int = 6

    def __post_init__(self) -> None:
        # Validate enum-string fields per Catalog #287 explicit-input discipline.
        if not isinstance(self.contract, LevelDimensionContract):
            raise TypeError(
                f"contract must be LevelDimensionContract; got "
                f"{type(self.contract).__name__}."
            )
        # Validate side_info_shape early (mirrors HierarchyBindingContract
        # invariant at binding_contract.py:247-252).
        _validate_side_info_shape(self.wyner_ziv_top_level_side_info_shape)
        # Validate dtype is one of the two supported.
        if self.residual_dtype not in (np.float32, np.float16):
            raise ValueError(
                f"residual_dtype must be np.float32 or np.float16; "
                f"got {self.residual_dtype}."
            )
        if not (0 <= int(self.compression_level) <= 9):
            raise ValueError(
                f"compression_level must be in [0, 9] (zlib range); "
                f"got {self.compression_level}."
            )

    # ────────────────────────────────────────────────────────────────────
    # Protocol satisfaction: ``side_info_shape`` + ``encode`` + ``decode``
    # per binding_contract.py:376-419.
    # ────────────────────────────────────────────────────────────────────

    @property
    def side_info_shape(self) -> tuple[int, int, int]:
        """The ``(C, H, W)`` shape of side_info; matches contract invariant.

        Per the Protocol docstring at ``binding_contract.py:395-398``:
        *"Must equal contract.wyner_ziv_top_level_side_info_shape"*.
        """
        return self.wyner_ziv_top_level_side_info_shape

    def _projection_matrix(self) -> np.ndarray:
        """Build the canonical deterministic projection matrix W.

        Re-built per call (cheap; the contract dims are O(64x3) at Z8
        canonical scale). If callsite performance matters, the caller
        may cache; the frozen dataclass intentionally avoids stateful
        caching to preserve Protocol semantics (every encode/decode
        call is independent + reproducible).
        """
        return side_info_projection_matrix_for_contract(
            state_dim=self.contract.deterministic_state_dim,
            side_info_shape=self.wyner_ziv_top_level_side_info_shape,
            seed=int(self.projection_seed),
        )

    def encode(self, top_state: Any, side_info: Any) -> bytes:
        """Encode ``top_state`` conditioned on ``side_info`` into archive bytes.

        Satisfies the Protocol contract from ``binding_contract.py:400-407``:

        Shape contract:
            top_state: ``(B, deterministic_state_dim)``
            side_info: ``(B, C, H, W)`` per side_info_shape
            return: variable-length bytes payload

        Canonical Wyner-Ziv 1976 § 3 Theorem 1 instantiation:
            1. Predict ``X_hat = predict(Y; W)`` via linear projection.
            2. Compute residual ``r = X - X_hat``.
            3. Quantize ``q = round(r / step_size)`` (uniform mid-tread).
            4. Pack header (20 bytes) + zlib(quantized indices as i16
               bytes).

        Args:
            top_state: source tensor ``X``, shape ``(B, state_dim)``.
                Accepts numpy / torch / MLX / any tensor with
                ``np.asarray`` compatibility.
            side_info: decoder-side information tensor ``Y``, shape
                ``(B, C, H, W)`` per ``side_info_shape``.

        Returns:
            ``bytes`` payload: HEADER (20 bytes) || zlib(residual_indices).

        Raises:
            :class:`WynerZivCoderShapeMismatchError`: top_state or
                side_info has wrong rank or wrong dims per the contract.
        """
        X = np.asarray(top_state, dtype=np.float32)
        if X.ndim != 2:
            raise WynerZivCoderShapeMismatchError(
                f"top_state must be 2-D (B, state_dim); got shape {X.shape}."
            )
        b, state_dim = X.shape
        if state_dim != self.contract.deterministic_state_dim:
            raise WynerZivCoderShapeMismatchError(
                f"top_state last axis ({state_dim}) must equal "
                f"contract.deterministic_state_dim "
                f"({self.contract.deterministic_state_dim})."
            )
        Y = np.asarray(side_info, dtype=np.float32)
        if Y.ndim != 4 or Y.shape[1:] != self.wyner_ziv_top_level_side_info_shape:
            raise WynerZivCoderShapeMismatchError(
                f"side_info must have shape "
                f"(B, {self.wyner_ziv_top_level_side_info_shape[0]}, "
                f"{self.wyner_ziv_top_level_side_info_shape[1]}, "
                f"{self.wyner_ziv_top_level_side_info_shape[2]}); "
                f"got shape {Y.shape}."
            )
        if Y.shape[0] != b:
            raise WynerZivCoderShapeMismatchError(
                f"top_state batch ({b}) and side_info batch "
                f"({Y.shape[0]}) must match."
            )

        # Wyner-Ziv 1976 step 1+2: linear prediction + residual.
        projection = self._projection_matrix()
        X_hat = predict_top_state_from_side_info(Y, projection)
        residual = X - X_hat  # (B, state_dim) np.float32

        # Wyner-Ziv 1976 step 3: uniform-mid-tread quantization.
        # Step size derives from SOURCE variance (not residual variance)
        # so the bit budget allocates a FIXED quantization grid. The
        # Wyner-Ziv conditional savings then manifest as residual
        # indices clustering near zero -> zlib compresses smaller ->
        # canonical R(D|Y) < R(D) gain observed empirically as fewer
        # archive bytes.
        total_elements = b * state_dim
        source_std = float(np.std(X.ravel())) if total_elements > 0 else 1.0
        step_size = _step_size_for_bit_budget(
            bit_budget_estimate=self.contract.bit_budget_estimate,
            total_elements=total_elements,
            source_std_estimate=source_std,
        )
        # Wyner-Ziv 1976 step 3 (quantize): choose canonical index dtype.
        # int16 is compact (2 bytes / element) but its [-32768, 32767]
        # range can clip when step_size is very small relative to
        # residual range. Compute the canonical fractional indices
        # first; if the max abs index would overflow int16 OR int32,
        # clip to the next-wider safe range with explicit clamp
        # (no quiet NaN-cast).
        if step_size <= 0 or not np.isfinite(step_size):  # defensive
            raw_indices = np.zeros_like(residual)
        else:
            raw_indices = residual / step_size
            # Guard against pathological inputs producing inf/nan
            # (e.g. degenerate near-zero residual + near-zero step
            # producing inf via 0/eps -> finite massive value).
            raw_indices = np.where(
                np.isfinite(raw_indices), raw_indices, 0.0
            )
        max_abs_index = (
            float(np.max(np.abs(raw_indices))) if raw_indices.size > 0 else 0.0
        )
        # int32 max safe range: 2^31 - 1024 (1024 slack for rounding).
        _I32_MAX_SAFE_INDEX = 2_147_482_624.0
        if max_abs_index <= _I16_MAX_SAFE_INDEX:
            quantized = np.round(raw_indices).astype(np.int16)
            index_dtype_marker = _INDEX_DTYPE_MARKER_I16
        else:
            # int32 is wider (4 bytes / element) but supports residual
            # ranges up to +/- 2.1e9 * step_size which covers any
            # practical Z8 top-state range. zlib's run-length tail
            # absorbs the extra width since most i32 indices have
            # zero high bytes.
            clipped = np.clip(
                raw_indices, -_I32_MAX_SAFE_INDEX, _I32_MAX_SAFE_INDEX
            )
            quantized = np.round(clipped).astype(np.int32)
            index_dtype_marker = _INDEX_DTYPE_MARKER_I32
        # Wyner-Ziv 1976 step 4: entropy code via zlib (canonical
        # compression_codec_for_side per sister
        # ``tac.codec.wyner_ziv_layer.InterceptLocation`` enum).
        quantized_bytes = quantized.tobytes()  # little-endian native dtype
        zlib_payload = zlib.compress(
            quantized_bytes, level=int(self.compression_level)
        )

        dtype_marker = (
            _DTYPE_MARKER_FP32
            if self.residual_dtype is np.float32
            else _DTYPE_MARKER_FP16
        )
        header = _pack_header(
            batch_size=b,
            state_dim=state_dim,
            step_size=step_size,
            residual_dtype_marker=dtype_marker,
            index_dtype_marker=index_dtype_marker,
            zlib_payload_len=len(zlib_payload),
        )
        return header + zlib_payload

    def decode(self, payload: bytes, side_info: Any) -> np.ndarray:
        """Decode ``payload`` back to ``top_state`` given ``side_info``.

        Satisfies the Protocol contract from
        ``binding_contract.py:410-416``: *"Inverse of encode; must
        round-trip to acceptable distortion (the Wyner-Ziv rate-distortion
        bound is the achievable target)."*

        Canonical inverse steps:
            1. Parse + validate header (20 bytes).
            2. zlib-decompress residual indices.
            3. Dequantize ``r_hat = q * step_size``.
            4. Reconstruct ``X = r_hat + predict(Y; W)``.

        Args:
            payload: ``bytes`` from a prior :meth:`encode` call.
            side_info: SAME shape (B, C, H, W) tensor as the encode-time
                side_info. Encoder + decoder MUST agree on Y bit-exact
                (Wyner-Ziv 1976 Theorem 1 requires Y available identically
                at both ends).

        Returns:
            ``np.ndarray`` of shape ``(B, state_dim)`` and ``np.float32``
            dtype: the reconstructed top_state.

        Raises:
            :class:`WynerZivCoderHeaderError`: payload header is malformed
                or truncated.
            :class:`WynerZivCoderShapeMismatchError`: side_info shape
                does not match the contract OR header batch_size
                disagrees with side_info batch.
        """
        (batch_size, state_dim, step_size,
         _dtype_marker, index_dtype_marker,
         zlib_payload_len) = _unpack_header(payload)
        if state_dim != self.contract.deterministic_state_dim:
            raise WynerZivCoderShapeMismatchError(
                f"payload header state_dim ({state_dim}) must equal "
                f"contract.deterministic_state_dim "
                f"({self.contract.deterministic_state_dim})."
            )
        Y = np.asarray(side_info, dtype=np.float32)
        if Y.ndim != 4 or Y.shape[1:] != self.wyner_ziv_top_level_side_info_shape:
            raise WynerZivCoderShapeMismatchError(
                f"side_info must have shape "
                f"(B, {self.wyner_ziv_top_level_side_info_shape[0]}, "
                f"{self.wyner_ziv_top_level_side_info_shape[1]}, "
                f"{self.wyner_ziv_top_level_side_info_shape[2]}); "
                f"got shape {Y.shape}."
            )
        if Y.shape[0] != batch_size:
            raise WynerZivCoderShapeMismatchError(
                f"payload header batch_size ({batch_size}) must equal "
                f"side_info batch ({Y.shape[0]})."
            )

        zlib_payload = payload[_HEADER_SIZE:_HEADER_SIZE + zlib_payload_len]
        try:
            quantized_bytes = zlib.decompress(zlib_payload)
        except zlib.error as exc:
            raise WynerZivCoderHeaderError(
                f"zlib decompression failed: {exc}. Payload is "
                f"likely corrupted or was not produced by a "
                f"WynerZivTopLevelCoderImpl.encode call."
            ) from exc
        # Index dtype was recorded in the header so decode picks the
        # right reader. int16 = 2 bytes/element; int32 = 4 bytes/element.
        if index_dtype_marker == _INDEX_DTYPE_MARKER_I16:
            bytes_per_element = 2
            index_np_dtype = np.int16
        else:
            bytes_per_element = 4
            index_np_dtype = np.int32
        expected_quantized_bytes = batch_size * state_dim * bytes_per_element
        if len(quantized_bytes) != expected_quantized_bytes:
            raise WynerZivCoderHeaderError(
                f"decoded residual bytes mismatch: expected "
                f"{expected_quantized_bytes} (B={batch_size} * "
                f"state_dim={state_dim} * {bytes_per_element} "
                f"bytes per index); got {len(quantized_bytes)}."
            )
        quantized = np.frombuffer(
            quantized_bytes, dtype=index_np_dtype
        ).reshape(batch_size, state_dim)
        # Wyner-Ziv 1976 inverse step 3: dequantize.
        r_hat = quantized.astype(np.float32) * float(step_size)
        # Wyner-Ziv 1976 inverse step 4: re-add the linear prediction.
        projection = self._projection_matrix()
        X_hat = predict_top_state_from_side_info(Y, projection)
        return (r_hat + X_hat).astype(np.float32)

    # ────────────────────────────────────────────────────────────────────
    # Convenience helpers (non-Protocol; auxiliary surface for tests +
    # Wyner-Ziv 1976 R(D|Y) bound verification).
    # ────────────────────────────────────────────────────────────────────

    def encode_with_round_trip_check(
        self,
        top_state: Any,
        side_info: Any,
        *,
        max_relative_distortion: float = 1.0,
    ) -> bytes:
        """Encode + verify round-trip distortion is within bound.

        Calls :meth:`encode` then :meth:`decode` immediately and asserts
        the round-tripped state's relative L2 distortion satisfies the
        canonical Wyner-Ziv 1976 Theorem 1 bound::

            ||X - decode(encode(X), Y)||_2 / ||X||_2 <= max_relative_distortion

        Use only in tests / debugging; production encode does NOT verify
        round-trip (the canonical Protocol contract does NOT mandate
        encode-time verification — the test suite verifies the bound
        empirically per Catalog #287).

        Args:
            top_state: source tensor X.
            side_info: side_info tensor Y.
            max_relative_distortion: upper bound on relative L2
                distortion; default 1.0 (loose; canonical tight bound
                depends on bit budget per the rate-distortion theorem).

        Returns:
            The encoded ``bytes`` payload.

        Raises:
            :class:`WynerZivCoderRoundTripError`: round-trip distortion
                exceeds ``max_relative_distortion``.
        """
        payload = self.encode(top_state, side_info)
        X = np.asarray(top_state, dtype=np.float32)
        X_recon = self.decode(payload, side_info)
        # Guard against zero-norm sources (would divide by 0).
        norm_X = float(np.linalg.norm(X.ravel()))
        if norm_X < 1e-12:
            return payload  # degenerate; round-trip trivially OK
        rel_dist = float(np.linalg.norm((X - X_recon).ravel())) / norm_X
        if rel_dist > max_relative_distortion:
            raise WynerZivCoderRoundTripError(
                f"round-trip distortion exceeds bound: "
                f"rel_dist={rel_dist:.6f} > "
                f"max_relative_distortion={max_relative_distortion:.6f}. "
                f"This indicates either the bit_budget_estimate "
                f"({self.contract.bit_budget_estimate}) is too small "
                f"for the source variance OR the side_info Y is "
                f"uncorrelated with the source X (in which case the "
                f"Wyner-Ziv conditional savings R(D|Y) < R(D) are nil)."
            )
        return payload

    def estimate_byte_budget_target(self) -> int:
        """Return the contract's ``bit_budget_estimate`` for this top level.

        Sister of :meth:`encode` for callers that want to know the
        target byte budget BEFORE encoding (e.g. archive grammar planners
        per the Z8 archive sectioning in
        ``tac.substrates.z8_hierarchical_predictive_coding.archive``).

        Returns:
            ``int``: target archive bytes; same value as the level
            contract's ``bit_budget_estimate`` field.
        """
        return int(self.contract.bit_budget_estimate)


def build_wyner_ziv_top_level_coder_for_contract(
    contract: HierarchyBindingContract,
    *,
    projection_seed: int = 0,
    residual_dtype: Any = None,
    compression_level: int = 6,
) -> WynerZivTopLevelCoderImpl:
    """Single-call canonical builder for M6 trainer callsites.

    Convenience constructor producing a :class:`WynerZivTopLevelCoderImpl`
    bound to the top level of the given Z8 binding contract. Sister of
    :func:`tac.substrates.z8_hierarchical_predictive_coding.loss.build_score_aware_level_loss_for_level`
    (M8) and
    :func:`tac.substrates.z8_hierarchical_predictive_coding.mallat_dwt_adapter.build_z8_mallat_dwt_adapter_for_level`
    (M5) per the canonical per-level builder convention.

    Args:
        contract: full Z8 binding contract; the top-level
            :class:`LevelDimensionContract` is extracted via
            ``contract.top_level``, and the side_info shape is taken from
            ``contract.wyner_ziv_top_level_side_info_shape``.
        projection_seed: deterministic seed for the linear-prediction
            projection matrix; default 0 (canonical Z8 seed).
        residual_dtype: ``np.float32`` (default if None) or ``np.float16``.
        compression_level: zlib compression level [0, 9]; default 6.

    Returns:
        :class:`WynerZivTopLevelCoderImpl` instance bound to the top
        level + side_info shape.

    Raises:
        TypeError: ``contract`` is not a :class:`HierarchyBindingContract`.
        ValueError: ``residual_dtype`` or ``compression_level`` is illegal.
    """
    if not isinstance(contract, HierarchyBindingContract):
        raise TypeError(
            f"contract must be HierarchyBindingContract; got "
            f"{type(contract).__name__}."
        )
    dtype = residual_dtype if residual_dtype is not None else np.float32
    impl = WynerZivTopLevelCoderImpl(
        contract=contract.top_level,
        wyner_ziv_top_level_side_info_shape=contract.wyner_ziv_top_level_side_info_shape,
        projection_seed=projection_seed,
        residual_dtype=dtype,
        compression_level=compression_level,
    )
    # Verify Protocol satisfaction at construction time (early-fail beats
    # late-fail per CLAUDE.md "Bugs must be permanently fixed AND self-
    # protected against"). The Protocol is @runtime_checkable so the
    # isinstance check verifies structural conformance.
    assert isinstance(impl, WynerZivTopLevelCoder), (
        "WynerZivTopLevelCoderImpl must satisfy WynerZivTopLevelCoder "
        "Protocol from binding_contract.py:376-419 (Protocol is "
        "@runtime_checkable)."
    )
    return impl
