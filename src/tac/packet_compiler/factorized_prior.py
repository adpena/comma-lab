# SPDX-License-Identifier: MIT
"""CompressAI ``FactorizedPrior`` (Ballé 2018 baseline) packet-compiler adapter.

This module wraps :class:`compressai.models.FactorizedPrior` (Ballé, Laparra,
Simoncelli, ICLR 2017 / 2018; the *non-hyperprior* factorised baseline) as a
typed ``tac.packet_compiler`` primitive, mirroring the contract every other
primitive in this package follows:

* a frozen ``CompressedPayload`` dataclass that carries the encoded byte
  strings + shape;
* ``encode_factorized_prior(...)`` / ``decode_factorized_prior(...)``
  functions whose round-trip is bit-exact on the
  ``compressai_factorized_prior_v1`` golden vector;
* deterministic SHA-256 over the encoded byte stream so the golden vector
  is reproducible from the recipe documented in the test module;
* zero scorer-load, zero ``/tmp`` paths, zero MPS device defaults.

The CompressAI ``FactorizedPrior`` model uses a single fully-factorised
``EntropyBottleneck`` on the analysis output ``y``. There is no
hyperprior side-channel — the entropy bottleneck's learned CDF tables ARE
the side-information, and they are baked into the trained ``state_dict``.
The compressed payload is therefore a single byte string per batch element.

Wire format
===========

::

    magic       = b'CAFP'                 # 4 bytes  — "CompressAI FactorizedPrior"
    version     = uint8                   # 1 byte   — payload schema version (0x01)
    n_batch     = uint32 little-endian    # 4 bytes  — batch dimension (== 1 typical)
    shape_h     = uint32 little-endian    # 4 bytes  — y_hat spatial height
    shape_w     = uint32 little-endian    # 4 bytes  — y_hat spatial width
    string_len  = uint32 little-endian    # 4 bytes  — len(strings[0][b]) for b=0
    string_body = bytes                    # string_len bytes — the AC payload
    ... repeats string_len + string_body for each batch element ...

CompressAI's ``compress()`` returns a list-of-lists structure
``strings = [[bytes, bytes, ...]]`` where the outer list has one entry per
sub-stream (``FactorizedPrior`` has exactly **one** sub-stream — the ``y``
entropy-bottleneck stream). The inner list has one byte string per batch
element. We linearise that to a sequence of ``(string_len, string_body)``
records per batch element to keep the wire format flat.

Composition / substrate compatibility
=====================================

The CompressAI baseline operates on a feature map (typically shape
``(B, N, H, W)`` post-analysis). Within ``tac.packet_compiler`` it
composes with any substrate whose analysis output is a 4-D feature map
the model was trained on. The ``target_substrate_hint`` is intentionally
narrow (``"any_with_compatible_latent_shape"``) — the trained model's
``N`` (number of channels) and downsampling factor must match the
substrate's analysis stage.

CLAUDE.md compliance
====================

* ``score_claim=false``, ``promotion_eligible=false``,
  ``ready_for_exact_eval_dispatch=false`` per the
  ``forbidden_score_claim_with_byte_change_unless_inflate_consumes``
  non-negotiable. This is a *primitive*, not a packet.
* No scorer load, no MPS, no ``/tmp`` paths.
* CompressAI's ``model.update(force=True)`` MUST be called before any
  ``compress()`` invocation; the wrapper raises ``FactorizedPriorError``
  on uninitialised CDF tables.
* Pure functional encode/decode helpers; the only state is the trained
  ``FactorizedPrior`` model the caller passes in. The adapter does NOT
  own model weights.
* The wire format is deterministic given the trained model + the input
  tensor — any drift here would break the golden vector.

[empirical:src/tac/packet_compiler/golden_vectors/compressai_factorized_prior_v1.json]

score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


MAGIC_FACTORIZED_PRIOR = b"CAFP"
FACTORIZED_PRIOR_VERSION = 0x01
TARGET_SUBSTRATE_HINT = "any_with_compatible_latent_shape"
_HEADER_PREFIX_LEN = len(MAGIC_FACTORIZED_PRIOR) + 1 + 4 + 4 + 4  # magic+ver+n+h+w


class FactorizedPriorError(RuntimeError):
    """Raised on invalid CompressAI FactorizedPrior payloads or contracts."""


@dataclass(frozen=True)
class FactorizedPriorPayload:
    """Encoded ``FactorizedPrior`` payload.

    Attributes
    ----------
    magic
        Always ``b'CAFP'``.
    version
        Wire-format version. Currently ``0x01``.
    n_batch
        Number of batch elements in the original ``compress()`` input.
    shape
        ``(H, W)`` of the latent feature map ``y_hat`` after analysis.
        This is what ``decompress(strings, shape)`` expects.
    strings
        Per-batch-element byte strings. ``len(strings) == n_batch``.
    """

    magic: bytes
    version: int
    n_batch: int
    shape: tuple[int, int]
    strings: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if self.magic != MAGIC_FACTORIZED_PRIOR:
            raise FactorizedPriorError(
                f"magic must be {MAGIC_FACTORIZED_PRIOR!r}; got {self.magic!r}"
            )
        if self.version != FACTORIZED_PRIOR_VERSION:
            raise FactorizedPriorError(
                f"unsupported version {self.version}; "
                f"this module emits {FACTORIZED_PRIOR_VERSION}"
            )
        if self.n_batch != len(self.strings):
            raise FactorizedPriorError(
                f"n_batch ({self.n_batch}) != len(strings) ({len(self.strings)})"
            )
        if not (isinstance(self.shape, tuple) and len(self.shape) == 2):
            raise FactorizedPriorError(
                f"shape must be (H, W); got {self.shape!r}"
            )
        for i, dim in enumerate(self.shape):
            if not isinstance(dim, int) or dim < 1:
                raise FactorizedPriorError(
                    f"shape[{i}] must be a positive int; got {dim!r}"
                )


def _assert_model_updated(model: Any) -> None:
    """Refuse encode against a model that has not had ``update()`` called.

    CompressAI's CDF tables are built lazily; without a prior ``update()``
    the entropy bottleneck has no AC table and ``compress()`` either raises
    or emits a payload that won't round-trip.
    """
    eb = getattr(model, "entropy_bottleneck", None)
    if eb is None:
        raise FactorizedPriorError(
            "model has no entropy_bottleneck attribute; "
            "is this a compressai.models.FactorizedPrior?"
        )
    if not hasattr(eb, "_quantized_cdf") or eb._quantized_cdf.numel() == 0:
        raise FactorizedPriorError(
            "FactorizedPrior model has no CDF tables. "
            "Call model.update(force=True) before compress()."
        )


def encode_factorized_prior(
    model: Any,
    x: "torch.Tensor",
) -> FactorizedPriorPayload:
    """Encode a feature map ``x`` through a CompressAI FactorizedPrior.

    Parameters
    ----------
    model
        A trained ``compressai.models.FactorizedPrior`` instance. The
        model must be in ``eval()`` mode and ``model.update(force=True)``
        must have been called (so the CDF tables exist).
    x
        Input tensor of shape ``(B, 3, H, W)``. The model's analysis
        stage downsamples by a factor of 16 (4 stride-2 convs), so
        ``H % 16 == 0`` and ``W % 16 == 0`` is required.

    Returns
    -------
    FactorizedPriorPayload
        The encoded payload + metadata; pass to
        :func:`serialize_factorized_prior` to obtain the wire bytes.

    Raises
    ------
    FactorizedPriorError
        On uninitialised model, missing CompressAI dependency, or
        unsupported tensor shape.
    """
    import torch  # local import to keep module import-time cheap

    _assert_model_updated(model)
    if not isinstance(x, torch.Tensor):
        raise FactorizedPriorError(
            f"x must be a torch.Tensor; got {type(x)!r}"
        )
    if x.dim() != 4:
        raise FactorizedPriorError(
            f"x must be 4-D (B, C, H, W); got {tuple(x.shape)!r}"
        )
    b, _c, h, w = x.shape
    if h % 16 != 0 or w % 16 != 0:
        raise FactorizedPriorError(
            f"FactorizedPrior requires H and W divisible by 16; got ({h}, {w})"
        )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model.compress(x)
    finally:
        if was_training:
            model.train()
    strings = out["strings"]
    shape = tuple(int(d) for d in out["shape"])  # (H/16, W/16)
    # CompressAI FactorizedPrior has exactly one sub-stream.
    if len(strings) != 1:
        raise FactorizedPriorError(
            f"FactorizedPrior must emit exactly 1 sub-stream; "
            f"got {len(strings)} (is this really FactorizedPrior?)"
        )
    per_batch = tuple(bytes(s) for s in strings[0])
    if len(per_batch) != b:
        raise FactorizedPriorError(
            f"per-batch string count ({len(per_batch)}) != batch ({b})"
        )
    return FactorizedPriorPayload(
        magic=MAGIC_FACTORIZED_PRIOR,
        version=FACTORIZED_PRIOR_VERSION,
        n_batch=b,
        shape=(int(shape[0]), int(shape[1])),
        strings=per_batch,
    )


def decode_factorized_prior(
    model: Any,
    payload: FactorizedPriorPayload,
) -> "torch.Tensor":
    """Decode a payload back to the reconstructed ``x_hat`` tensor.

    Parameters
    ----------
    model
        The same trained ``compressai.models.FactorizedPrior`` instance
        used to encode (or a freshly loaded copy of the same weights).
        ``model.update(force=True)`` must have been called.
    payload
        The :class:`FactorizedPriorPayload` produced by
        :func:`encode_factorized_prior`.

    Returns
    -------
    torch.Tensor
        ``x_hat`` reconstruction of shape ``(n_batch, 3, H, W)`` where
        ``H = shape[0] * 16`` and ``W = shape[1] * 16``.
    """
    import torch

    _assert_model_updated(model)
    if not isinstance(payload, FactorizedPriorPayload):
        raise FactorizedPriorError(
            f"payload must be FactorizedPriorPayload; got {type(payload)!r}"
        )
    strings_for_compressai = [list(payload.strings)]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model.decompress(strings_for_compressai, payload.shape)
    finally:
        if was_training:
            model.train()
    return out["x_hat"]


def serialize_factorized_prior(payload: FactorizedPriorPayload) -> bytes:
    """Serialise a payload into deterministic wire bytes (see grammar above)."""
    parts: list[bytes] = [
        payload.magic,
        struct.pack("<B", payload.version),
        struct.pack("<I", payload.n_batch),
        struct.pack("<I", payload.shape[0]),
        struct.pack("<I", payload.shape[1]),
    ]
    for s in payload.strings:
        parts.append(struct.pack("<I", len(s)))
        parts.append(s)
    return b"".join(parts)


def deserialize_factorized_prior(blob: bytes) -> FactorizedPriorPayload:
    """Parse wire bytes back into a :class:`FactorizedPriorPayload`."""
    if not isinstance(blob, (bytes, bytearray)):
        raise FactorizedPriorError(
            f"blob must be bytes-like; got {type(blob)!r}"
        )
    blob = bytes(blob)
    if len(blob) < _HEADER_PREFIX_LEN:
        raise FactorizedPriorError(
            f"blob too short ({len(blob)} bytes); "
            f"need at least {_HEADER_PREFIX_LEN} for header"
        )
    if blob[:4] != MAGIC_FACTORIZED_PRIOR:
        raise FactorizedPriorError(
            f"bad magic; expected {MAGIC_FACTORIZED_PRIOR!r}, "
            f"got {blob[:4]!r}"
        )
    cursor = 4
    (version,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    if version != FACTORIZED_PRIOR_VERSION:
        raise FactorizedPriorError(
            f"unsupported wire version {version}; "
            f"this module reads only {FACTORIZED_PRIOR_VERSION}"
        )
    (n_batch,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    (shape_h,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    (shape_w,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    strings: list[bytes] = []
    for _ in range(n_batch):
        if cursor + 4 > len(blob):
            raise FactorizedPriorError(
                "blob truncated at string length prefix"
            )
        (slen,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        if cursor + slen > len(blob):
            raise FactorizedPriorError(
                f"blob truncated at string body (need {slen} bytes, "
                f"have {len(blob) - cursor})"
            )
        strings.append(blob[cursor : cursor + slen])
        cursor += slen
    if cursor != len(blob):
        raise FactorizedPriorError(
            f"trailing {len(blob) - cursor} bytes after last string body"
        )
    return FactorizedPriorPayload(
        magic=MAGIC_FACTORIZED_PRIOR,
        version=version,
        n_batch=n_batch,
        shape=(int(shape_h), int(shape_w)),
        strings=tuple(strings),
    )


__all__ = [
    "FACTORIZED_PRIOR_VERSION",
    "FactorizedPriorError",
    "FactorizedPriorPayload",
    "MAGIC_FACTORIZED_PRIOR",
    "TARGET_SUBSTRATE_HINT",
    "decode_factorized_prior",
    "deserialize_factorized_prior",
    "encode_factorized_prior",
    "serialize_factorized_prior",
]
