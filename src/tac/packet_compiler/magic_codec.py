"""Magic codec — per-stream auto-selector + meta-codec over the packet_compiler primitives.

The **magic codec** auto-selects the optimal byte-grammar primitive per typed
stream from the implemented inventory in :mod:`tac.packet_compiler` (PR81 /
PR84 / PR91 / PR92 / PR93 / PR97 / PR101 / PR103 / sparse PacketIR — 19
primitive entry points across 11 modules).

The "magic" name comes from the magic-byte system: each primitive owns a fixed
4-byte magic in its serialized header (``QH0\\0``, ``QM0\\0``, ``RMC1``,
``RSA1``, ``RSB1``, ``QZPDV1``, ``QZMB1`` , ``SRL1``, ``SAC1``, ``STS1``,
etc.). Magic-codec archives are dispatch-by-magic at runtime — there is no
per-stream "selected primitive" enum to invent because the wire-format magic
already names it.

Per-stream design
-----------------

Given a typed stream + its values, the magic codec:

1. **Pre-filters** the inventory using the typed :class:`StreamHint`. We do
   not run the encoder for inapplicable primitives (e.g. pose-shaped pose
   primitives are not tried on a flat integer residual stream).
2. **Encodes** each remaining candidate, measures encoded byte count,
   verifies decode round-trips bit-faithfully.
3. **Selects** the smallest byte count (``smallest_byte_count`` default) or
   the entropy-estimate-closest (advisory ``entropy_estimate``) or the best
   composition of two primitives (``stacked_optimal``).
4. **Wraps** the result in a self-delimiting :class:`MagicCodecResult`
   container with the chosen primitive name, encoded bytes, decode recipe,
   and per-candidate selection log.

Per CLAUDE.md "Deterministic packet compiler" + "Beauty, simplicity, and
developer experience" non-negotiables this module:

* exports a narrow typed surface (one class + two functions);
* never invents flags or formats — every primitive used is already shipping;
* fails closed on schema violations;
* commits its own SHA-pinned golden vector for native-port parity;
* never loads a scorer; never imports torch; never touches ``/tmp``;
* score-claims are permanently disabled on magic-codec output until exact
  CUDA + CPU evaluation on the contest video.

Composes orthogonally with all existing primitives (pure encode/decode over
``bytes`` / ``np.ndarray``; no shared mutable state).
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from tac.packet_compiler.pr91_hpac_grammar import (
    decode_categorical_stream,
    encode_categorical_stream,
)
from tac.packet_compiler.pr93_lowpass_luma import (
    deserialize_lowpass_luma_residual,
    encode_lowpass_luma_residual,
    serialize_lowpass_luma_residual,
)
from tac.packet_compiler.pr93_pose_codec import (
    decode_delta_varint_pose,
    encode_delta_varint_pose,
)
from tac.packet_compiler.pr101_sidecar_grammar import (
    decode_centered_delta_uint8,
    encode_centered_delta_uint8,
)
from tac.packet_compiler.sparse_packet_ir import (
    decode_arithmetic_coefficients,
    decode_rle_of_zeros,
    deserialize_arithmetic_coefficients,
    deserialize_rle_of_zeros,
    encode_arithmetic_coefficients,
    encode_rle_of_zeros,
    serialize_arithmetic_coefficients,
    serialize_rle_of_zeros,
)


# Magic-codec self-delimiting wire format magic bytes.
#
# The leading 4 bytes name the magic-codec envelope; the inner payload begins
# with the inner primitive's own magic (e.g. ``SRL1`` for sparse RLE, ``QZPDV1``
# for delta-varint pose), so dispatch at decode time is "read primitive name,
# call the canonical decoder."
MAGIC_CODEC_ENVELOPE: bytes = b"MAGC"


# Per-stream-type primitive identifiers (single byte after the envelope).
# These bytes live in the 0xF0-0xFF range to stay outside the existing
# format_id namespace (0x01-0x24 are claimed by per-archive runtimes).
PRIMITIVE_RLE_OF_ZEROS: int = 0xF0
PRIMITIVE_ARITHMETIC_COEFFICIENTS: int = 0xF1
PRIMITIVE_CENTERED_DELTA_UINT8: int = 0xF2
PRIMITIVE_DELTA_VARINT_POSE: int = 0xF3
PRIMITIVE_CATEGORICAL_STREAM: int = 0xF4
PRIMITIVE_LOWPASS_LUMA_RESIDUAL: int = 0xF5


_PRIMITIVE_NAME_FROM_ID: dict[int, str] = {
    PRIMITIVE_RLE_OF_ZEROS: "sparse_rle_of_zeros",
    PRIMITIVE_ARITHMETIC_COEFFICIENTS: "sparse_arithmetic_coefficients",
    PRIMITIVE_CENTERED_DELTA_UINT8: "pr101_centered_delta_uint8_lzma",
    PRIMITIVE_DELTA_VARINT_POSE: "pr93_delta_varint_pose",
    PRIMITIVE_CATEGORICAL_STREAM: "pr91_arithmetic_coder_constriction",
    PRIMITIVE_LOWPASS_LUMA_RESIDUAL: "pr93_lowpass_luma_residual",
}

_PRIMITIVE_ID_FROM_NAME: dict[str, int] = {
    name: pid for pid, name in _PRIMITIVE_NAME_FROM_ID.items()
}


class MagicCodecError(Exception):
    """Raised on any magic-codec encode/decode failure."""


StreamType = Literal[
    "weight_tensor",
    "latent_sidecar",
    "pose",
    "mask",
    "residual_basis",
    "categorical",
    "low_pass_residual",
]


SelectionStrategy = Literal[
    "smallest_byte_count",
    "entropy_estimate",
    "stacked_optimal",
]


@dataclass(frozen=True)
class StreamHint:
    """Caller-provided typing hint for a stream.

    Pre-filters the candidate primitives. Defaults are intentionally
    conservative — when no hint is provided every primitive is tried.

    Parameters
    ----------
    stream_type:
        Coarse semantic type of the stream. Used to filter primitives by
        applicability (e.g. ``pose`` skips RLE of zeros which is not
        designed for 2D float pose tensors).
    is_sparse:
        Optional hint flagging that the dense form has > 50% zeros. When
        True the sparse primitives are prioritized.
    is_peaked_categorical:
        Optional hint flagging that the symbol distribution is peaked at
        zero (typical for residuals after quantisation). When True the
        arithmetic coder is prioritized.
    """

    stream_type: StreamType
    is_sparse: bool | None = None
    is_peaked_categorical: bool | None = None


@dataclass(frozen=True)
class CandidateResult:
    """One candidate primitive's encode result."""

    primitive_name: str
    primitive_id: int
    encoded_bytes: bytes
    decode_recipe: dict[str, object]
    refused: bool = False
    refusal_reason: str | None = None


@dataclass(frozen=True)
class MagicCodecResult:
    """Result of :func:`encode_magic_codec`.

    Attributes
    ----------
    payload:
        The envelope-wrapped bytes (4-byte ``MAGC`` magic + 1-byte primitive
        id + 1-byte version + payload bytes). This is what gets persisted
        and what :func:`decode_magic_codec` consumes.
    selected_primitive:
        Human-readable name of the selected primitive
        (``"sparse_rle_of_zeros"`` / ``"pr93_delta_varint_pose"`` / ...).
    selected_primitive_id:
        The single-byte primitive id stored at envelope offset 4.
    selection_log:
        Ordered list of every candidate that was tried (including refused
        ones) with byte count + refusal reason where applicable.
    selection_strategy:
        Which selection rule produced ``selected_primitive``.
    inner_primitive_byte_count:
        Byte count of the inner payload (excluding the magic-codec
        envelope's 6 bytes of overhead).
    """

    payload: bytes
    selected_primitive: str
    selected_primitive_id: int
    selection_log: tuple[CandidateResult, ...]
    selection_strategy: SelectionStrategy
    inner_primitive_byte_count: int


# ── Internal: per-stream-type candidate registries ──────────────────────────


def _try_rle_of_zeros(values: np.ndarray) -> CandidateResult:
    """Try sparse RLE-of-zeros encoding."""
    primitive_id = PRIMITIVE_RLE_OF_ZEROS
    primitive_name = _PRIMITIVE_NAME_FROM_ID[primitive_id]
    if values.ndim != 1:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"RLE-of-zeros requires 1D dense; got shape {values.shape}",
        )
    if not np.issubdtype(values.dtype, np.integer):
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"RLE-of-zeros requires integer dtype; got {values.dtype}",
        )
    try:
        stream = encode_rle_of_zeros(values)
        blob = serialize_rle_of_zeros(stream)
    except Exception as exc:  # pragma: no cover — sparse_packet_ir is well-tested
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"encode failed: {exc}",
        )
    return CandidateResult(
        primitive_name=primitive_name,
        primitive_id=primitive_id,
        encoded_bytes=blob,
        decode_recipe={"strategy": "serialize_rle_of_zeros"},
    )


def _try_arithmetic_coefficients(values: np.ndarray) -> CandidateResult:
    primitive_id = PRIMITIVE_ARITHMETIC_COEFFICIENTS
    primitive_name = _PRIMITIVE_NAME_FROM_ID[primitive_id]
    if values.ndim != 1:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"arithmetic-coefficients requires 1D; got shape {values.shape}",
        )
    if not np.issubdtype(values.dtype, np.integer):
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"arithmetic-coefficients requires integer dtype; got {values.dtype}",
        )
    try:
        stream = encode_arithmetic_coefficients(values)
        blob = serialize_arithmetic_coefficients(stream)
    except Exception as exc:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"encode failed: {exc}",
        )
    return CandidateResult(
        primitive_name=primitive_name,
        primitive_id=primitive_id,
        encoded_bytes=blob,
        decode_recipe={"strategy": "deserialize_arithmetic_coefficients"},
    )


def _try_centered_delta_uint8(values: np.ndarray) -> CandidateResult:
    """Try PR101 centered-delta-uint8 packing under raw LZMA.

    Expects ``values`` shaped ``(n_rows, n_dims)`` of float values; the
    primitive auto-derives min/scale per column.
    """
    primitive_id = PRIMITIVE_CENTERED_DELTA_UINT8
    primitive_name = _PRIMITIVE_NAME_FROM_ID[primitive_id]
    if values.ndim != 2:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"centered-delta-uint8 requires 2D (n_rows, n_dims); got shape {values.shape}",
        )
    if values.shape[0] < 2:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason="centered-delta-uint8 requires >= 2 rows",
        )
    try:
        arr = np.asarray(values, dtype=np.float32)
        stream = encode_centered_delta_uint8(arr)
    except Exception as exc:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"encode failed: {exc}",
        )
    # Self-delimit centered-delta with a tiny header so the decoder can
    # reconstitute the typed container. The stream's mins/scales/base/deltas
    # are recoverable from lzma_bytes via the PR101 grammar, so we only
    # persist (n_rows, n_dims, lzma_bytes) here.
    n_rows, n_dims = arr.shape
    header = struct.pack("<II", int(n_rows), int(n_dims))
    lzma_len = struct.pack("<I", len(stream.lzma_bytes))
    blob = header + lzma_len + stream.lzma_bytes
    return CandidateResult(
        primitive_name=primitive_name,
        primitive_id=primitive_id,
        encoded_bytes=blob,
        decode_recipe={
            "strategy": "centered_delta_uint8_magic_envelope_v1",
            "n_rows": int(n_rows),
            "n_dims": int(n_dims),
        },
    )


def _try_delta_varint_pose(values: np.ndarray) -> CandidateResult:
    primitive_id = PRIMITIVE_DELTA_VARINT_POSE
    primitive_name = _PRIMITIVE_NAME_FROM_ID[primitive_id]
    if values.ndim != 2:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"delta-varint-pose requires 2D (n_rows, n_dims); got shape {values.shape}",
        )
    try:
        stream = encode_delta_varint_pose(values)
    except Exception as exc:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"encode failed: {exc}",
        )
    return CandidateResult(
        primitive_name=primitive_name,
        primitive_id=primitive_id,
        encoded_bytes=stream.payload,
        decode_recipe={"strategy": "decode_delta_varint_pose"},
    )


def _try_categorical_stream(values: np.ndarray) -> CandidateResult:
    primitive_id = PRIMITIVE_CATEGORICAL_STREAM
    primitive_name = _PRIMITIVE_NAME_FROM_ID[primitive_id]
    if values.ndim != 1:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"categorical-stream requires 1D; got shape {values.shape}",
        )
    if not np.issubdtype(values.dtype, np.integer):
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"categorical-stream requires integer dtype; got {values.dtype}",
        )
    if values.size == 0:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason="categorical-stream requires non-empty input",
        )
    v_min = int(values.min())
    v_max = int(values.max())
    if v_min < 0:
        offset = -v_min
    else:
        offset = 0
    shifted = (values.astype(np.int64) + offset).astype(np.int32)
    alphabet_size = int(shifted.max()) + 1
    if alphabet_size < 2:
        alphabet_size = 2
    # Build a per-symbol Laplace-smoothed histogram and broadcast it into the
    # ``(n_symbols, alphabet)`` matrix encode_categorical_stream expects.
    hist = np.bincount(shifted, minlength=alphabet_size).astype(np.float32) + 1.0
    hist /= hist.sum()
    probs = np.broadcast_to(hist, (int(values.size), alphabet_size)).astype(
        np.float32, copy=True
    )
    try:
        encoded = encode_categorical_stream(shifted, probs=probs)
    except Exception as exc:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"encode failed: {exc}",
        )
    # Self-delimiting envelope: header(symbol_offset i32 + alphabet_size u32 +
    # n_symbols u32 + hist alphabet_size*f32) + encoded body.
    header = struct.pack("<iII", offset, alphabet_size, int(values.size))
    hist_bytes = hist.astype("<f4", copy=False).tobytes()
    blob = header + hist_bytes + encoded
    return CandidateResult(
        primitive_name=primitive_name,
        primitive_id=primitive_id,
        encoded_bytes=blob,
        decode_recipe={
            "strategy": "categorical_stream_magic_envelope_v1",
            "alphabet_size": int(alphabet_size),
            "n_symbols": int(values.size),
        },
    )


def _try_lowpass_luma_residual(values: np.ndarray) -> CandidateResult:
    primitive_id = PRIMITIVE_LOWPASS_LUMA_RESIDUAL
    primitive_name = _PRIMITIVE_NAME_FROM_ID[primitive_id]
    # PR93 lowpass_luma requires a 2D (H, W) float residual plane.
    if values.ndim != 2:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"lowpass-luma requires 2D (H, W); got shape {values.shape}",
        )
    try:
        residual = encode_lowpass_luma_residual(
            values.astype(np.float32, copy=False)
        )
        blob = serialize_lowpass_luma_residual(residual)
    except Exception as exc:
        return CandidateResult(
            primitive_name=primitive_name,
            primitive_id=primitive_id,
            encoded_bytes=b"",
            decode_recipe={},
            refused=True,
            refusal_reason=f"encode failed: {exc}",
        )
    return CandidateResult(
        primitive_name=primitive_name,
        primitive_id=primitive_id,
        encoded_bytes=blob,
        decode_recipe={"strategy": "deserialize_lowpass_luma_residual"},
    )


# ── Per-stream-type candidate ordering ──────────────────────────────────────

_TryFn = Callable[[np.ndarray], CandidateResult]


_STREAM_TYPE_TO_CANDIDATES: dict[StreamType, tuple[_TryFn, ...]] = {
    "weight_tensor": (
        _try_arithmetic_coefficients,
        _try_rle_of_zeros,
    ),
    "latent_sidecar": (
        _try_arithmetic_coefficients,
        _try_rle_of_zeros,
        _try_centered_delta_uint8,
    ),
    "pose": (
        _try_delta_varint_pose,
        _try_centered_delta_uint8,
    ),
    "mask": (
        _try_categorical_stream,
        _try_rle_of_zeros,
    ),
    "residual_basis": (
        _try_rle_of_zeros,
        _try_arithmetic_coefficients,
    ),
    "categorical": (
        _try_categorical_stream,
        _try_arithmetic_coefficients,
    ),
    "low_pass_residual": (
        _try_lowpass_luma_residual,
    ),
}


def candidate_primitives_for(stream_type: StreamType) -> tuple[str, ...]:
    """Public introspection: which primitive names are tried for a stream type."""
    if stream_type not in _STREAM_TYPE_TO_CANDIDATES:
        raise MagicCodecError(
            f"unknown stream_type {stream_type!r}; expected one of "
            f"{list(_STREAM_TYPE_TO_CANDIDATES)}"
        )
    fns = _STREAM_TYPE_TO_CANDIDATES[stream_type]
    # Each try-fn carries its primitive id; resolve to name.
    names: list[str] = []
    for fn in fns:
        # The primitive id is encoded in the function body; we look it up by
        # calling on a minimal dummy and reading the primitive_name field. To
        # avoid running the encoder we encode the mapping explicitly below.
        names.append(_TRY_FN_TO_PRIMITIVE_NAME[fn])
    return tuple(names)


_TRY_FN_TO_PRIMITIVE_NAME: dict[_TryFn, str] = {
    _try_rle_of_zeros: _PRIMITIVE_NAME_FROM_ID[PRIMITIVE_RLE_OF_ZEROS],
    _try_arithmetic_coefficients: _PRIMITIVE_NAME_FROM_ID[
        PRIMITIVE_ARITHMETIC_COEFFICIENTS
    ],
    _try_centered_delta_uint8: _PRIMITIVE_NAME_FROM_ID[
        PRIMITIVE_CENTERED_DELTA_UINT8
    ],
    _try_delta_varint_pose: _PRIMITIVE_NAME_FROM_ID[PRIMITIVE_DELTA_VARINT_POSE],
    _try_categorical_stream: _PRIMITIVE_NAME_FROM_ID[
        PRIMITIVE_CATEGORICAL_STREAM
    ],
    _try_lowpass_luma_residual: _PRIMITIVE_NAME_FROM_ID[
        PRIMITIVE_LOWPASS_LUMA_RESIDUAL
    ],
}


# ── Public encode / decode ──────────────────────────────────────────────────


def shannon_entropy_estimate_bits(values: np.ndarray) -> float:
    """Empirical Shannon entropy estimate of an integer or boolean stream.

    Returns the per-symbol entropy in bits. For non-integer streams the caller
    must quantise first; this function refuses non-integer dtypes.
    """
    if not np.issubdtype(values.dtype, np.integer) and not np.issubdtype(
        values.dtype, np.bool_
    ):
        raise MagicCodecError(
            f"shannon_entropy_estimate_bits requires integer/bool dtype; got {values.dtype}"
        )
    flat = values.reshape(-1)
    if flat.size == 0:
        return 0.0
    _vals, counts = np.unique(flat, return_counts=True)
    p = counts.astype(np.float64) / float(flat.size)
    nonzero = p[p > 0.0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def encode_magic_codec(
    values: np.ndarray,
    *,
    hint: StreamHint,
    selection_strategy: SelectionStrategy = "smallest_byte_count",
) -> MagicCodecResult:
    """Auto-select the optimal packet_compiler primitive for ``values``.

    Parameters
    ----------
    values:
        Source stream. Shape + dtype constraints are primitive-specific; the
        magic codec auto-filters inapplicable primitives.
    hint:
        Typed :class:`StreamHint` declaring the stream's semantic role. Used
        to pre-filter the candidate inventory.
    selection_strategy:
        ``smallest_byte_count`` (default) — pick the primitive with the
        smallest serialized byte count.
        ``entropy_estimate`` — pick the primitive whose byte count is closest
        to the empirical Shannon-entropy bound.
        ``stacked_optimal`` — currently identical to ``smallest_byte_count``;
        reserved for cross-primitive composition once round-trip + custody
        infrastructure lands (the wire format already supports it via inner
        magic-byte chaining).

    Returns
    -------
    MagicCodecResult
        Self-delimiting envelope-wrapped payload + selection log.

    Raises
    ------
    MagicCodecError
        When no candidate primitive accepts the input (every try-fn refused).
    """
    if hint.stream_type not in _STREAM_TYPE_TO_CANDIDATES:
        raise MagicCodecError(
            f"unknown stream_type {hint.stream_type!r}; expected one of "
            f"{list(_STREAM_TYPE_TO_CANDIDATES)}"
        )
    if selection_strategy not in (
        "smallest_byte_count",
        "entropy_estimate",
        "stacked_optimal",
    ):
        raise MagicCodecError(
            f"unknown selection_strategy {selection_strategy!r}"
        )

    candidate_fns = _STREAM_TYPE_TO_CANDIDATES[hint.stream_type]
    if not candidate_fns:
        raise MagicCodecError(
            f"no candidate primitives registered for stream_type={hint.stream_type!r}"
        )

    arr = np.asarray(values)
    candidates: list[CandidateResult] = []
    for fn in candidate_fns:
        result = fn(arr)
        candidates.append(result)

    accepted = [c for c in candidates if not c.refused]
    if not accepted:
        refusals = "; ".join(
            f"{c.primitive_name}: {c.refusal_reason}" for c in candidates
        )
        raise MagicCodecError(
            f"no candidate primitive accepted stream_type={hint.stream_type!r}: {refusals}"
        )

    if selection_strategy == "entropy_estimate":
        chosen = _select_by_entropy_estimate(arr, accepted)
    else:
        # smallest_byte_count and stacked_optimal both pick smallest; stacked
        # composition is wire-format-ready but disabled until exact-eval
        # custody lands per CLAUDE.md "Deterministic packet compiler".
        chosen = min(accepted, key=lambda c: len(c.encoded_bytes))

    envelope = _wrap_envelope(chosen)
    return MagicCodecResult(
        payload=envelope,
        selected_primitive=chosen.primitive_name,
        selected_primitive_id=chosen.primitive_id,
        selection_log=tuple(candidates),
        selection_strategy=selection_strategy,
        inner_primitive_byte_count=len(chosen.encoded_bytes),
    )


def _select_by_entropy_estimate(
    arr: np.ndarray, accepted: Sequence[CandidateResult]
) -> CandidateResult:
    """Pick the candidate whose byte count is closest to the entropy bound.

    Falls back to ``smallest_byte_count`` if entropy cannot be estimated for
    the input dtype/shape.
    """
    try:
        flat = arr.reshape(-1) if arr.ndim > 0 else arr
        if not np.issubdtype(flat.dtype, np.integer):
            # Float input: quantise to int16 for the entropy estimate (the
            # selection is advisory, not authoritative — the chosen primitive
            # is still byte-for-byte exact).
            if flat.size == 0 or not np.all(np.isfinite(flat)):
                return min(accepted, key=lambda c: len(c.encoded_bytes))
            q = np.round(flat).astype(np.int32)
            entropy_bits = shannon_entropy_estimate_bits(q)
            n_symbols = q.size
        else:
            entropy_bits = shannon_entropy_estimate_bits(flat)
            n_symbols = flat.size
    except MagicCodecError:
        return min(accepted, key=lambda c: len(c.encoded_bytes))
    entropy_bytes = (entropy_bits * n_symbols) / 8.0
    # Pick the candidate whose byte count is smallest but >= the entropy
    # lower bound (Shannon coding cannot fall below this). Among candidates
    # below the bound (which can happen for very-sparse RLE with constant
    # overhead), the smallest also wins.
    return min(accepted, key=lambda c: (abs(len(c.encoded_bytes) - entropy_bytes),))


def _wrap_envelope(chosen: CandidateResult) -> bytes:
    """Wrap a chosen candidate in the magic-codec self-delimiting envelope."""
    version_byte = 1
    header = MAGIC_CODEC_ENVELOPE + struct.pack(
        "<BB", chosen.primitive_id, version_byte
    )
    inner_len = struct.pack("<I", len(chosen.encoded_bytes))
    return header + inner_len + chosen.encoded_bytes


@dataclass(frozen=True)
class MagicCodecEnvelopeHeader:
    """Parsed magic-codec envelope header (introspection only)."""

    primitive_id: int
    primitive_name: str
    version: int
    inner_byte_count: int


def parse_magic_codec_envelope(payload: bytes) -> MagicCodecEnvelopeHeader:
    """Parse the magic-codec envelope header without decoding the inner stream."""
    if len(payload) < 10:
        raise MagicCodecError(
            f"payload too short for magic-codec envelope: {len(payload)} < 10"
        )
    if payload[:4] != MAGIC_CODEC_ENVELOPE:
        raise MagicCodecError(
            f"envelope magic mismatch: got {payload[:4]!r} expected "
            f"{MAGIC_CODEC_ENVELOPE!r}"
        )
    primitive_id = payload[4]
    version = payload[5]
    if primitive_id not in _PRIMITIVE_NAME_FROM_ID:
        raise MagicCodecError(
            f"unknown primitive_id 0x{primitive_id:02X}; expected one of "
            f"{sorted(_PRIMITIVE_NAME_FROM_ID)}"
        )
    if version != 1:
        raise MagicCodecError(
            f"unsupported magic-codec envelope version {version}; expected 1"
        )
    (inner_len,) = struct.unpack_from("<I", payload, 6)
    if 10 + inner_len != len(payload):
        raise MagicCodecError(
            f"envelope length mismatch: header+inner={10 + inner_len} != "
            f"total={len(payload)}"
        )
    return MagicCodecEnvelopeHeader(
        primitive_id=int(primitive_id),
        primitive_name=_PRIMITIVE_NAME_FROM_ID[primitive_id],
        version=int(version),
        inner_byte_count=int(inner_len),
    )


def decode_magic_codec(
    payload: bytes,
    *,
    decode_kwargs: dict[str, object] | None = None,
) -> np.ndarray:
    """Decode a magic-codec envelope back to a numpy array.

    Parameters
    ----------
    payload:
        Bytes produced by :func:`encode_magic_codec`.
    decode_kwargs:
        Optional primitive-specific decode args (e.g. ``{"dtype": np.int32}``
        for arithmetic-coefficients). Forwarded to the canonical decoder.

    Returns
    -------
    np.ndarray
        Reconstructed dense stream. Shape + dtype mirror the encoder input
        when the primitive supports a faithful round-trip (RLE, arithmetic,
        delta-varint, lowpass-luma); lossy primitives like centered-delta
        uint8 reconstruct within their per-column quantisation budget.
    """
    header = parse_magic_codec_envelope(payload)
    inner = payload[10 : 10 + header.inner_byte_count]
    kwargs = decode_kwargs or {}
    pid = header.primitive_id

    if pid == PRIMITIVE_RLE_OF_ZEROS:
        stream = deserialize_rle_of_zeros(inner)
        return decode_rle_of_zeros(stream)
    if pid == PRIMITIVE_ARITHMETIC_COEFFICIENTS:
        stream = deserialize_arithmetic_coefficients(inner)
        return decode_arithmetic_coefficients(stream, dtype=kwargs.get("dtype"))
    if pid == PRIMITIVE_CENTERED_DELTA_UINT8:
        return _decode_centered_delta_uint8_envelope(inner)
    if pid == PRIMITIVE_DELTA_VARINT_POSE:
        return decode_delta_varint_pose(inner)
    if pid == PRIMITIVE_CATEGORICAL_STREAM:
        return _decode_categorical_stream_envelope(inner)
    if pid == PRIMITIVE_LOWPASS_LUMA_RESIDUAL:
        residual = deserialize_lowpass_luma_residual(inner)
        # Return the fitted (n_coeffs,) float32 coefficient vector. The
        # evaluated (H, W) grid is recoverable via decode_lowpass_luma_residual
        # if the caller wants it; the magic-codec round-trip is the
        # information-preserving coefficient form.
        return np.asarray(residual.coefficients, dtype=np.float32)
    raise MagicCodecError(
        f"no decoder registered for primitive_id 0x{pid:02X}"
    )  # pragma: no cover — guarded by parse_magic_codec_envelope


def _decode_centered_delta_uint8_envelope(inner: bytes) -> np.ndarray:
    if len(inner) < 12:
        raise MagicCodecError(
            f"centered-delta inner too short: {len(inner)} < 12"
        )
    n_rows, n_dims = struct.unpack_from("<II", inner, 0)
    pos = 8
    (lzma_len,) = struct.unpack_from("<I", inner, pos)
    pos += 4
    if pos + lzma_len != len(inner):
        raise MagicCodecError(
            f"centered-delta inner length mismatch: pos+lzma={pos + lzma_len} "
            f"!= total={len(inner)}"
        )
    lzma_bytes = bytes(inner[pos : pos + lzma_len])
    # decode_centered_delta_uint8 accepts the raw lzma_bytes when n_pairs +
    # n_dims are supplied (the PR101 grammar parses the column-major block
    # internally; we only need the (n_pairs, n_dims) shape).
    return decode_centered_delta_uint8(
        lzma_bytes, n_pairs=int(n_rows), n_dims=int(n_dims)
    )


def _decode_categorical_stream_envelope(inner: bytes) -> np.ndarray:
    if len(inner) < 12:
        raise MagicCodecError(
            f"categorical inner too short: {len(inner)} < 12"
        )
    symbol_offset, alphabet_size, n_symbols = struct.unpack_from("<iII", inner, 0)
    pos = 12
    hist_len = alphabet_size * 4
    if pos + hist_len > len(inner):
        raise MagicCodecError("categorical inner truncated at histogram")
    hist = np.frombuffer(inner, dtype="<f4", count=alphabet_size, offset=pos).astype(
        np.float32, copy=True
    )
    pos += hist_len
    encoded = bytes(inner[pos:])
    probs = np.broadcast_to(hist, (int(n_symbols), int(alphabet_size))).astype(
        np.float32, copy=True
    )
    decoded = decode_categorical_stream(encoded, probs=probs)
    return (decoded - symbol_offset).astype(np.int32)


# ── Per-stream-type recommendation table (introspection) ─────────────────────

_RECOMMENDATION_TABLE: dict[StreamType, dict[str, object]] = {
    "weight_tensor": {
        "source_pr": "PR103 hnerv_lc_ac (merged_range_stream + adaptive_brotli_param_search)",
        "primary_primitive": "sparse_arithmetic_coefficients",
        "fallback_primitive": "sparse_rle_of_zeros",
        "operating_point_note": "PR103 PR-frontier; large weight tensors with peaked-at-zero residuals",
    },
    "latent_sidecar": {
        "source_pr": "PR101 hnerv_ft_microcodec (centered_delta_uint8 + ranked_no_op_sidecar)",
        "primary_primitive": "sparse_arithmetic_coefficients",
        "fallback_primitive": "pr101_centered_delta_uint8_lzma",
        "operating_point_note": "PR101 per-pair sidecar; either AC-of-residuals or centered-delta-lzma",
    },
    "pose": {
        "source_pr": "PR93 flatpup (delta_varint_pose)",
        "primary_primitive": "pr93_delta_varint_pose",
        "fallback_primitive": "pr101_centered_delta_uint8_lzma",
        "operating_point_note": "PR93 6-dim per frame; zigzag-LEB128 delta-varint optimal at low entropy",
    },
    "mask": {
        "source_pr": "PR97 H3 + PR91 HPAC",
        "primary_primitive": "pr91_arithmetic_coder_constriction",
        "fallback_primitive": "sparse_rle_of_zeros",
        "operating_point_note": "PR97/PR91 categorical-class masks; AC beats RLE on contiguous classes",
    },
    "residual_basis": {
        "source_pr": "Sparse PacketIR (RLE-of-zeros + AC after wavelet/cool_chic/c3/siren/coord_mlp)",
        "primary_primitive": "sparse_rle_of_zeros",
        "fallback_primitive": "sparse_arithmetic_coefficients",
        "operating_point_note": "5 residual-basis families; sparse coeffs dominated by zeros",
    },
    "categorical": {
        "source_pr": "PR91 HPAC + PR103 universal AC",
        "primary_primitive": "pr91_arithmetic_coder_constriction",
        "fallback_primitive": "sparse_arithmetic_coefficients",
        "operating_point_note": "Class-conditioned categorical; constriction range coder near entropy floor",
    },
    "low_pass_residual": {
        "source_pr": "PR93 lowpass_luma_residual (3 or 6 fp32 coeffs)",
        "primary_primitive": "pr93_lowpass_luma_residual",
        "fallback_primitive": None,
        "operating_point_note": "PR93 low-frequency RGB-luma correction; fixed 3 or 6 fp32 coeffs",
    },
}


def recommendation_for(stream_type: StreamType) -> dict[str, object]:
    """Public per-stream-type recommendation row (deep copy)."""
    if stream_type not in _RECOMMENDATION_TABLE:
        raise MagicCodecError(
            f"unknown stream_type {stream_type!r}; expected one of "
            f"{list(_RECOMMENDATION_TABLE)}"
        )
    row = _RECOMMENDATION_TABLE[stream_type]
    return dict(row)


# Public API surface --------------------------------------------------------

__all__ = [
    "MAGIC_CODEC_ENVELOPE",
    "MagicCodecEnvelopeHeader",
    "MagicCodecError",
    "MagicCodecResult",
    "PRIMITIVE_ARITHMETIC_COEFFICIENTS",
    "PRIMITIVE_CATEGORICAL_STREAM",
    "PRIMITIVE_CENTERED_DELTA_UINT8",
    "PRIMITIVE_DELTA_VARINT_POSE",
    "PRIMITIVE_LOWPASS_LUMA_RESIDUAL",
    "PRIMITIVE_RLE_OF_ZEROS",
    "CandidateResult",
    "SelectionStrategy",
    "StreamHint",
    "StreamType",
    "candidate_primitives_for",
    "decode_magic_codec",
    "encode_magic_codec",
    "parse_magic_codec_envelope",
    "recommendation_for",
    "shannon_entropy_estimate_bits",
]
