# SPDX-License-Identifier: MIT
"""CompressAI Ballé-family hyperprior packet-compiler adapter.

This module wraps the three hyperprior variants from CompressAI as typed
``tac.packet_compiler`` primitives:

* :class:`compressai.models.ScaleHyperprior`
  (Ballé 2018 hyperprior with scale-only side information)
* :class:`compressai.models.MeanScaleHyperprior`
  (Minnen, Ballé, Toderici 2018 — mean + scale conditional Gaussian)
* :class:`compressai.models.JointAutoregressiveHierarchicalPriors`
  (Minnen, Ballé, Toderici 2018 — autoregressive context model on ``y``)

All three emit a **two-sub-stream** payload: ``strings = [[y_strings],
[z_strings]]``. The wire format below is variant-agnostic — the same
serializer handles all three; the *variant tag* selects which CompressAI
model class is used to decode.

Wire format
===========

::

    magic       = b'CABH'                 # 4 bytes  — "CompressAI Balle Hyperprior"
    version     = uint8                   # 1 byte   — payload schema version (0x01)
    variant     = uint8                   # 1 byte   — 0x01=scale, 0x02=meanscale, 0x03=joint_ar
    n_batch     = uint32 little-endian    # 4 bytes
    shape_h     = uint32 little-endian    # 4 bytes  — z_hat spatial height
    shape_w     = uint32 little-endian    # 4 bytes  — z_hat spatial width
    n_substreams = uint8                  # 1 byte   — always 2 (y + z)
    # For each sub-stream (y first, z second):
    #   For each batch element:
    #     string_len = uint32 little-endian
    #     string_body = bytes (string_len)

The two sub-streams are emitted **y-then-z** for parity with CompressAI's
internal ``compress()`` ordering. We do NOT re-order or merge the streams
because doing so would couple to inflate-time CDF table state.

Variant compatibility
=====================

* ``VARIANT_SCALE`` (``0x01``) corresponds to
  :class:`compressai.models.ScaleHyperprior`.
* ``VARIANT_MEANSCALE`` (``0x02``) corresponds to
  :class:`compressai.models.MeanScaleHyperprior`.
* ``VARIANT_JOINT_AR`` (``0x03``) corresponds to
  :class:`compressai.models.JointAutoregressiveHierarchicalPriors`.

The variant tag is stored in the payload so the inflate-time decoder
knows which CompressAI model class to instantiate. Decoders that load a
mismatched variant raise :class:`BalleHyperpriorError` with a clear
``payload.variant != model.__class__.__name__`` message.

Composition / substrate compatibility
=====================================

CompressAI hyperprior models take 4-D image inputs and emit two
sub-streams: one for the analysis output ``y`` (conditional Gaussian)
and one for the hyperprior summary ``z`` (factorised). The
``target_substrate_hint`` is intentionally narrow
(``"any_with_compatible_latent_shape"``) — the trained model's ``N``,
``M``, and downsampling factor must match the substrate's analysis
stage.

Per the schema-elision / stacking discipline, these primitives DO NOT
stack with the schema-elision primitives (PR98 CD1 / PR100 schema-driven
/ PR105 size-sort) on the same byte region; CompressAI hyperpriors emit
their OWN entropy-coded byte streams.

CLAUDE.md compliance
====================

* ``score_claim=false``, ``promotion_eligible=false``,
  ``ready_for_exact_eval_dispatch=false`` per the
  ``forbidden_score_claim_with_byte_change_unless_inflate_consumes``
  non-negotiable.
* No scorer load, no MPS, no ``/tmp`` paths.
* ``model.update(force=True)`` MUST be called on the model before any
  ``compress()`` invocation — the wrapper raises if either the
  ``entropy_bottleneck`` or ``gaussian_conditional`` has uninitialised
  CDF tables.
* The wire format is deterministic given the trained model + the input
  tensor.
* Per the CLAUDE.md HNeRV parity discipline rule
  ``check_representation_lane_has_archive_grammar_at_design_time``,
  this adapter declares ``archive_grammar`` (the wire format above) +
  ``export_format`` (``"compressai_balle_hyperprior_two_substream"``)
  + ``score_aware_loss`` (``None`` — this is a primitive; downstream
  packet compilers wire the loss).

[empirical:src/tac/packet_compiler/golden_vectors/compressai_balle_hyperprior_v1.json]

score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


MAGIC_BALLE_HYPERPRIOR = b"CABH"
BALLE_HYPERPRIOR_VERSION = 0x01

VARIANT_SCALE = 0x01
VARIANT_MEANSCALE = 0x02
VARIANT_JOINT_AR = 0x03

#: Map from CompressAI class name -> variant tag.
_CLASS_TO_VARIANT: dict[str, int] = {
    "ScaleHyperprior": VARIANT_SCALE,
    "MeanScaleHyperprior": VARIANT_MEANSCALE,
    "JointAutoregressiveHierarchicalPriors": VARIANT_JOINT_AR,
}

#: Inverse map for documentation / error messages.
_VARIANT_TO_CLASS: dict[int, str] = {v: k for k, v in _CLASS_TO_VARIANT.items()}

TARGET_SUBSTRATE_HINT = "any_with_compatible_latent_shape"
_HEADER_PREFIX_LEN = (
    len(MAGIC_BALLE_HYPERPRIOR) + 1 + 1 + 4 + 4 + 4 + 1
)  # magic+ver+variant+n+h+w+nsub


class BalleHyperpriorError(RuntimeError):
    """Raised on invalid CompressAI Ballé-hyperprior payloads or contracts."""


@dataclass(frozen=True)
class BalleHyperpriorPayload:
    """Encoded CompressAI Ballé-hyperprior payload.

    Attributes
    ----------
    magic
        Always ``b'CABH'``.
    version
        Wire-format version. Currently ``0x01``.
    variant
        One of ``VARIANT_SCALE``, ``VARIANT_MEANSCALE``,
        ``VARIANT_JOINT_AR``.
    n_batch
        Batch elements in the original ``compress()`` input.
    shape
        ``(H, W)`` of the ``z_hat`` spatial extent. CompressAI's
        ``decompress(strings, shape)`` requires this for sub-pixel
        upsampling alignment.
    strings
        Tuple of two tuples: ``(y_strings, z_strings)``. Each inner
        tuple has length ``n_batch``.
    """

    magic: bytes
    version: int
    variant: int
    n_batch: int
    shape: tuple[int, int]
    strings: tuple[tuple[bytes, ...], tuple[bytes, ...]]

    def __post_init__(self) -> None:
        if self.magic != MAGIC_BALLE_HYPERPRIOR:
            raise BalleHyperpriorError(
                f"magic must be {MAGIC_BALLE_HYPERPRIOR!r}; got {self.magic!r}"
            )
        if self.version != BALLE_HYPERPRIOR_VERSION:
            raise BalleHyperpriorError(
                f"unsupported version {self.version}; "
                f"this module emits {BALLE_HYPERPRIOR_VERSION}"
            )
        if self.variant not in _VARIANT_TO_CLASS:
            raise BalleHyperpriorError(
                f"unknown variant {self.variant}; "
                f"expected one of {sorted(_VARIANT_TO_CLASS)}"
            )
        if len(self.strings) != 2:
            raise BalleHyperpriorError(
                f"strings must be a 2-tuple (y, z); "
                f"got {len(self.strings)} sub-streams"
            )
        for sub_i, sub in enumerate(self.strings):
            if len(sub) != self.n_batch:
                raise BalleHyperpriorError(
                    f"sub-stream {sub_i} has {len(sub)} entries; "
                    f"expected {self.n_batch} (n_batch)"
                )
        if not (isinstance(self.shape, tuple) and len(self.shape) == 2):
            raise BalleHyperpriorError(
                f"shape must be (H, W); got {self.shape!r}"
            )
        for i, dim in enumerate(self.shape):
            if not isinstance(dim, int) or dim < 1:
                raise BalleHyperpriorError(
                    f"shape[{i}] must be a positive int; got {dim!r}"
                )


def _variant_for_model(model: Any) -> int:
    name = model.__class__.__name__
    if name not in _CLASS_TO_VARIANT:
        raise BalleHyperpriorError(
            f"unsupported model class {name!r}; "
            f"expected one of {sorted(_CLASS_TO_VARIANT)}"
        )
    return _CLASS_TO_VARIANT[name]


def _assert_model_updated(model: Any) -> None:
    eb = getattr(model, "entropy_bottleneck", None)
    gc = getattr(model, "gaussian_conditional", None)
    if eb is None or gc is None:
        raise BalleHyperpriorError(
            "model missing entropy_bottleneck or gaussian_conditional; "
            "is this a CompressAI Ballé-family hyperprior?"
        )
    if not hasattr(eb, "_quantized_cdf") or eb._quantized_cdf.numel() == 0:
        raise BalleHyperpriorError(
            "Ballé hyperprior model has no entropy-bottleneck CDF tables. "
            "Call model.update(force=True) before compress()."
        )
    if not hasattr(gc, "_quantized_cdf") or gc._quantized_cdf.numel() == 0:
        raise BalleHyperpriorError(
            "Ballé hyperprior model has no gaussian-conditional CDF tables. "
            "Call model.update(force=True) before compress()."
        )


def encode_balle_hyperprior(
    model: Any,
    x: "torch.Tensor",
) -> BalleHyperpriorPayload:
    """Encode a feature map ``x`` through a CompressAI Ballé hyperprior.

    Parameters
    ----------
    model
        A trained ``ScaleHyperprior`` / ``MeanScaleHyperprior`` /
        ``JointAutoregressiveHierarchicalPriors`` instance with
        ``model.update(force=True)`` called.
    x
        Input tensor of shape ``(B, 3, H, W)``. Hyperprior models
        downsample by 64 (4 + 2 stride-2 convs), so the conventional
        constraint is ``H % 64 == 0`` and ``W % 64 == 0``. The wrapper
        does NOT enforce this — the user is responsible for padding —
        but the CompressAI model will fail with a shape mismatch if
        violated.

    Returns
    -------
    BalleHyperpriorPayload
    """
    import torch

    _assert_model_updated(model)
    variant = _variant_for_model(model)
    if not isinstance(x, torch.Tensor):
        raise BalleHyperpriorError(
            f"x must be a torch.Tensor; got {type(x)!r}"
        )
    if x.dim() != 4:
        raise BalleHyperpriorError(
            f"x must be 4-D (B, C, H, W); got {tuple(x.shape)!r}"
        )
    b = x.shape[0]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model.compress(x)
    finally:
        if was_training:
            model.train()
    strings = out["strings"]
    shape = tuple(int(d) for d in out["shape"])
    if len(strings) != 2:
        raise BalleHyperpriorError(
            f"hyperprior must emit exactly 2 sub-streams (y, z); "
            f"got {len(strings)} (is this really a hyperprior model?)"
        )
    y_strings = tuple(bytes(s) for s in strings[0])
    z_strings = tuple(bytes(s) for s in strings[1])
    if len(y_strings) != b or len(z_strings) != b:
        raise BalleHyperpriorError(
            f"per-batch counts mismatch: y={len(y_strings)}, "
            f"z={len(z_strings)}, batch={b}"
        )
    return BalleHyperpriorPayload(
        magic=MAGIC_BALLE_HYPERPRIOR,
        version=BALLE_HYPERPRIOR_VERSION,
        variant=variant,
        n_batch=b,
        shape=(int(shape[0]), int(shape[1])),
        strings=(y_strings, z_strings),
    )


def decode_balle_hyperprior(
    model: Any,
    payload: BalleHyperpriorPayload,
) -> "torch.Tensor":
    """Decode a Ballé-hyperprior payload back to ``x_hat``.

    Parameters
    ----------
    model
        The same trained CompressAI hyperprior model used to encode
        (or a freshly loaded copy of the same weights). The model class
        must match the payload's variant tag — passing a
        ``ScaleHyperprior`` to a ``VARIANT_MEANSCALE`` payload raises.
    payload
        The :class:`BalleHyperpriorPayload` produced by
        :func:`encode_balle_hyperprior`.

    Returns
    -------
    torch.Tensor
        ``x_hat`` reconstruction.
    """
    import torch

    _assert_model_updated(model)
    if not isinstance(payload, BalleHyperpriorPayload):
        raise BalleHyperpriorError(
            f"payload must be BalleHyperpriorPayload; got {type(payload)!r}"
        )
    expected_class = _VARIANT_TO_CLASS[payload.variant]
    if model.__class__.__name__ != expected_class:
        raise BalleHyperpriorError(
            f"variant mismatch: payload encoded with {expected_class}, "
            f"but model is {model.__class__.__name__}"
        )
    strings_for_compressai = [
        list(payload.strings[0]),
        list(payload.strings[1]),
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model.decompress(strings_for_compressai, payload.shape)
    finally:
        if was_training:
            model.train()
    return out["x_hat"]


def serialize_balle_hyperprior(payload: BalleHyperpriorPayload) -> bytes:
    """Serialise a payload to deterministic wire bytes (see grammar above)."""
    parts: list[bytes] = [
        payload.magic,
        struct.pack("<B", payload.version),
        struct.pack("<B", payload.variant),
        struct.pack("<I", payload.n_batch),
        struct.pack("<I", payload.shape[0]),
        struct.pack("<I", payload.shape[1]),
        struct.pack("<B", 2),  # n_substreams = 2 (y, z)
    ]
    for substream in payload.strings:
        for s in substream:
            parts.append(struct.pack("<I", len(s)))
            parts.append(s)
    return b"".join(parts)


def deserialize_balle_hyperprior(blob: bytes) -> BalleHyperpriorPayload:
    """Parse wire bytes back into a :class:`BalleHyperpriorPayload`."""
    if not isinstance(blob, (bytes, bytearray)):
        raise BalleHyperpriorError(
            f"blob must be bytes-like; got {type(blob)!r}"
        )
    blob = bytes(blob)
    if len(blob) < _HEADER_PREFIX_LEN:
        raise BalleHyperpriorError(
            f"blob too short ({len(blob)} bytes); "
            f"need at least {_HEADER_PREFIX_LEN}"
        )
    if blob[:4] != MAGIC_BALLE_HYPERPRIOR:
        raise BalleHyperpriorError(
            f"bad magic; expected {MAGIC_BALLE_HYPERPRIOR!r}, "
            f"got {blob[:4]!r}"
        )
    cursor = 4
    (version,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    if version != BALLE_HYPERPRIOR_VERSION:
        raise BalleHyperpriorError(
            f"unsupported wire version {version}; "
            f"this module reads only {BALLE_HYPERPRIOR_VERSION}"
        )
    (variant,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    if variant not in _VARIANT_TO_CLASS:
        raise BalleHyperpriorError(
            f"unknown variant {variant}; "
            f"expected one of {sorted(_VARIANT_TO_CLASS)}"
        )
    (n_batch,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    (shape_h,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    (shape_w,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    (n_substreams,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    if n_substreams != 2:
        raise BalleHyperpriorError(
            f"n_substreams must be 2 for Ballé hyperprior; got {n_substreams}"
        )
    sub_strings: list[list[bytes]] = []
    for _sub in range(2):
        per_batch: list[bytes] = []
        for _ in range(n_batch):
            if cursor + 4 > len(blob):
                raise BalleHyperpriorError(
                    "blob truncated at string length prefix"
                )
            (slen,) = struct.unpack_from("<I", blob, cursor)
            cursor += 4
            if cursor + slen > len(blob):
                raise BalleHyperpriorError(
                    f"blob truncated at string body (need {slen}, "
                    f"have {len(blob) - cursor})"
                )
            per_batch.append(blob[cursor : cursor + slen])
            cursor += slen
        sub_strings.append(per_batch)
    if cursor != len(blob):
        raise BalleHyperpriorError(
            f"trailing {len(blob) - cursor} bytes after last string body"
        )
    return BalleHyperpriorPayload(
        magic=MAGIC_BALLE_HYPERPRIOR,
        version=version,
        variant=variant,
        n_batch=n_batch,
        shape=(int(shape_h), int(shape_w)),
        strings=(tuple(sub_strings[0]), tuple(sub_strings[1])),
    )


__all__ = [
    "BALLE_HYPERPRIOR_VERSION",
    "BalleHyperpriorError",
    "BalleHyperpriorPayload",
    "MAGIC_BALLE_HYPERPRIOR",
    "TARGET_SUBSTRATE_HINT",
    "VARIANT_JOINT_AR",
    "VARIANT_MEANSCALE",
    "VARIANT_SCALE",
    "decode_balle_hyperprior",
    "deserialize_balle_hyperprior",
    "encode_balle_hyperprior",
    "serialize_balle_hyperprior",
]
