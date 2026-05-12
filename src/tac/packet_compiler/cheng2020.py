"""CompressAI Cheng-2020 (anchor + attention) packet-compiler adapter.

This module wraps the Cheng-Sun-Maeda-Tanaka CVPR 2020 entropy-model
architectures shipped by CompressAI as typed ``tac.packet_compiler``
primitives:

* :class:`compressai.models.Cheng2020Anchor` — residual blocks +
  attention-free joint-autoregressive hyperprior.
* :class:`compressai.models.Cheng2020Attention` — same backbone + a
  global-attention block at three pyramid levels.

Both variants emit a **two-sub-stream** payload (``y`` autoregressive
+ ``z`` factorised) identical in structure to the Ballé
:class:`compressai.models.JointAutoregressiveHierarchicalPriors`
variant — what differs is the analysis/synthesis architecture and the
context model. The wire format below is variant-agnostic; the variant
tag selects which CompressAI model class is used to decode.

Wire format
===========

::

    magic       = b'CACG'                 # 4 bytes  — "CompressAI Cheng-2020"
    version     = uint8                   # 1 byte   — payload schema version (0x01)
    variant     = uint8                   # 1 byte   — 0x01=anchor, 0x02=attention
    n_batch     = uint32 little-endian    # 4 bytes
    shape_h     = uint32 little-endian    # 4 bytes  — z_hat spatial height
    shape_w     = uint32 little-endian    # 4 bytes  — z_hat spatial width
    n_substreams = uint8                  # 1 byte   — always 2 (y + z)
    # For each sub-stream (y first, z second):
    #   For each batch element:
    #     string_len = uint32 little-endian
    #     string_body = bytes (string_len)

Variant compatibility
=====================

* ``VARIANT_ANCHOR`` (``0x01``) corresponds to
  :class:`compressai.models.Cheng2020Anchor`.
* ``VARIANT_ATTENTION`` (``0x02``) corresponds to
  :class:`compressai.models.Cheng2020Attention`.

The variant tag is stored in the payload so the inflate-time decoder
knows which CompressAI model class to instantiate. Decoders that load a
mismatched variant raise :class:`Cheng2020Error`.

Composition / substrate compatibility
=====================================

Cheng-2020 models take 4-D image inputs and emit ``y`` (joint
autoregressive Gaussian) + ``z`` (factorised) sub-streams. The
``target_substrate_hint`` is intentionally narrow
(``"any_with_compatible_latent_shape"``) — the trained model's ``N``
and downsampling factor must match the substrate's analysis stage.

CompressAI's autoregressive context model uses a serial PixelCNN-style
masked convolution at decode time, which makes inference latency
non-negligible (~seconds per 768x512 image on CPU). This is a known
property of the architecture; the wrapper itself does not introduce any
additional latency.

CLAUDE.md compliance
====================

* ``score_claim=false``, ``promotion_eligible=false``,
  ``ready_for_exact_eval_dispatch=false`` per the
  ``forbidden_score_claim_with_byte_change_unless_inflate_consumes``
  non-negotiable.
* No scorer load, no MPS, no ``/tmp`` paths.
* ``model.update(force=True)`` MUST be called on the model before any
  ``compress()`` invocation — the wrapper raises if the CDF tables are
  uninitialised.
* The wire format is deterministic given the trained model + the input
  tensor.
* Per CLAUDE.md HNeRV parity discipline check 124, the adapter
  declares ``archive_grammar`` (above) + ``export_format``
  (``"compressai_cheng2020_two_substream"``) + ``score_aware_loss``
  (``None`` — primitive; downstream packet compilers wire the loss).

[empirical:src/tac/packet_compiler/golden_vectors/compressai_cheng2020_v1.json]

score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


MAGIC_CHENG2020 = b"CACG"
CHENG2020_VERSION = 0x01

VARIANT_ANCHOR = 0x01
VARIANT_ATTENTION = 0x02

_CLASS_TO_VARIANT: dict[str, int] = {
    "Cheng2020Anchor": VARIANT_ANCHOR,
    "Cheng2020Attention": VARIANT_ATTENTION,
}
_VARIANT_TO_CLASS: dict[int, str] = {v: k for k, v in _CLASS_TO_VARIANT.items()}

TARGET_SUBSTRATE_HINT = "any_with_compatible_latent_shape"
_HEADER_PREFIX_LEN = (
    len(MAGIC_CHENG2020) + 1 + 1 + 4 + 4 + 4 + 1
)  # magic+ver+variant+n+h+w+nsub


class Cheng2020Error(RuntimeError):
    """Raised on invalid CompressAI Cheng-2020 payloads or contracts."""


@dataclass(frozen=True)
class Cheng2020Payload:
    """Encoded CompressAI Cheng-2020 payload.

    Attributes
    ----------
    magic
        Always ``b'CACG'``.
    version
        Wire-format version. Currently ``0x01``.
    variant
        One of ``VARIANT_ANCHOR``, ``VARIANT_ATTENTION``.
    n_batch
        Batch elements in the original ``compress()`` input.
    shape
        ``(H, W)`` of the ``z_hat`` spatial extent.
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
        if self.magic != MAGIC_CHENG2020:
            raise Cheng2020Error(
                f"magic must be {MAGIC_CHENG2020!r}; got {self.magic!r}"
            )
        if self.version != CHENG2020_VERSION:
            raise Cheng2020Error(
                f"unsupported version {self.version}; "
                f"this module emits {CHENG2020_VERSION}"
            )
        if self.variant not in _VARIANT_TO_CLASS:
            raise Cheng2020Error(
                f"unknown variant {self.variant}; "
                f"expected one of {sorted(_VARIANT_TO_CLASS)}"
            )
        if len(self.strings) != 2:
            raise Cheng2020Error(
                f"strings must be a 2-tuple (y, z); "
                f"got {len(self.strings)} sub-streams"
            )
        for sub_i, sub in enumerate(self.strings):
            if len(sub) != self.n_batch:
                raise Cheng2020Error(
                    f"sub-stream {sub_i} has {len(sub)} entries; "
                    f"expected {self.n_batch} (n_batch)"
                )
        if not (isinstance(self.shape, tuple) and len(self.shape) == 2):
            raise Cheng2020Error(
                f"shape must be (H, W); got {self.shape!r}"
            )
        for i, dim in enumerate(self.shape):
            if not isinstance(dim, int) or dim < 1:
                raise Cheng2020Error(
                    f"shape[{i}] must be a positive int; got {dim!r}"
                )


def _variant_for_model(model: Any) -> int:
    name = model.__class__.__name__
    if name not in _CLASS_TO_VARIANT:
        raise Cheng2020Error(
            f"unsupported model class {name!r}; "
            f"expected one of {sorted(_CLASS_TO_VARIANT)}"
        )
    return _CLASS_TO_VARIANT[name]


def _assert_model_updated(model: Any) -> None:
    eb = getattr(model, "entropy_bottleneck", None)
    gc = getattr(model, "gaussian_conditional", None)
    if eb is None or gc is None:
        raise Cheng2020Error(
            "model missing entropy_bottleneck or gaussian_conditional; "
            "is this a CompressAI Cheng-2020 model?"
        )
    if not hasattr(eb, "_quantized_cdf") or eb._quantized_cdf.numel() == 0:
        raise Cheng2020Error(
            "Cheng-2020 model has no entropy-bottleneck CDF tables. "
            "Call model.update(force=True) before compress()."
        )
    if not hasattr(gc, "_quantized_cdf") or gc._quantized_cdf.numel() == 0:
        raise Cheng2020Error(
            "Cheng-2020 model has no gaussian-conditional CDF tables. "
            "Call model.update(force=True) before compress()."
        )


def encode_cheng2020(
    model: Any,
    x: "torch.Tensor",
) -> Cheng2020Payload:
    """Encode a feature map ``x`` through a CompressAI Cheng-2020 model.

    Parameters
    ----------
    model
        A trained ``Cheng2020Anchor`` / ``Cheng2020Attention`` instance
        with ``model.update(force=True)`` called.
    x
        Input tensor of shape ``(B, 3, H, W)``. Cheng-2020 models
        downsample by 64 (3 stride-2 downsample modules + 2 stride-2
        hyperprior convs); the conventional constraint is
        ``H % 64 == 0`` and ``W % 64 == 0``. The wrapper does NOT
        enforce this — the user is responsible for padding — but the
        CompressAI model will fail with a shape mismatch if violated.

    Returns
    -------
    Cheng2020Payload
    """
    import torch

    _assert_model_updated(model)
    variant = _variant_for_model(model)
    if not isinstance(x, torch.Tensor):
        raise Cheng2020Error(
            f"x must be a torch.Tensor; got {type(x)!r}"
        )
    if x.dim() != 4:
        raise Cheng2020Error(
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
        raise Cheng2020Error(
            f"Cheng-2020 must emit exactly 2 sub-streams (y, z); "
            f"got {len(strings)}"
        )
    y_strings = tuple(bytes(s) for s in strings[0])
    z_strings = tuple(bytes(s) for s in strings[1])
    if len(y_strings) != b or len(z_strings) != b:
        raise Cheng2020Error(
            f"per-batch counts mismatch: y={len(y_strings)}, "
            f"z={len(z_strings)}, batch={b}"
        )
    return Cheng2020Payload(
        magic=MAGIC_CHENG2020,
        version=CHENG2020_VERSION,
        variant=variant,
        n_batch=b,
        shape=(int(shape[0]), int(shape[1])),
        strings=(y_strings, z_strings),
    )


def decode_cheng2020(
    model: Any,
    payload: Cheng2020Payload,
) -> "torch.Tensor":
    """Decode a Cheng-2020 payload back to ``x_hat``."""
    import torch

    _assert_model_updated(model)
    if not isinstance(payload, Cheng2020Payload):
        raise Cheng2020Error(
            f"payload must be Cheng2020Payload; got {type(payload)!r}"
        )
    expected_class = _VARIANT_TO_CLASS[payload.variant]
    if model.__class__.__name__ != expected_class:
        raise Cheng2020Error(
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


def serialize_cheng2020(payload: Cheng2020Payload) -> bytes:
    """Serialise a payload to deterministic wire bytes."""
    parts: list[bytes] = [
        payload.magic,
        struct.pack("<B", payload.version),
        struct.pack("<B", payload.variant),
        struct.pack("<I", payload.n_batch),
        struct.pack("<I", payload.shape[0]),
        struct.pack("<I", payload.shape[1]),
        struct.pack("<B", 2),
    ]
    for substream in payload.strings:
        for s in substream:
            parts.append(struct.pack("<I", len(s)))
            parts.append(s)
    return b"".join(parts)


def deserialize_cheng2020(blob: bytes) -> Cheng2020Payload:
    """Parse wire bytes back into a :class:`Cheng2020Payload`."""
    if not isinstance(blob, (bytes, bytearray)):
        raise Cheng2020Error(
            f"blob must be bytes-like; got {type(blob)!r}"
        )
    blob = bytes(blob)
    if len(blob) < _HEADER_PREFIX_LEN:
        raise Cheng2020Error(
            f"blob too short ({len(blob)} bytes); "
            f"need at least {_HEADER_PREFIX_LEN}"
        )
    if blob[:4] != MAGIC_CHENG2020:
        raise Cheng2020Error(
            f"bad magic; expected {MAGIC_CHENG2020!r}, got {blob[:4]!r}"
        )
    cursor = 4
    (version,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    if version != CHENG2020_VERSION:
        raise Cheng2020Error(
            f"unsupported wire version {version}; "
            f"this module reads only {CHENG2020_VERSION}"
        )
    (variant,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    if variant not in _VARIANT_TO_CLASS:
        raise Cheng2020Error(
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
        raise Cheng2020Error(
            f"n_substreams must be 2 for Cheng-2020; got {n_substreams}"
        )
    sub_strings: list[list[bytes]] = []
    for _sub in range(2):
        per_batch: list[bytes] = []
        for _ in range(n_batch):
            if cursor + 4 > len(blob):
                raise Cheng2020Error(
                    "blob truncated at string length prefix"
                )
            (slen,) = struct.unpack_from("<I", blob, cursor)
            cursor += 4
            if cursor + slen > len(blob):
                raise Cheng2020Error(
                    f"blob truncated at string body (need {slen}, "
                    f"have {len(blob) - cursor})"
                )
            per_batch.append(blob[cursor : cursor + slen])
            cursor += slen
        sub_strings.append(per_batch)
    if cursor != len(blob):
        raise Cheng2020Error(
            f"trailing {len(blob) - cursor} bytes after last string body"
        )
    return Cheng2020Payload(
        magic=MAGIC_CHENG2020,
        version=version,
        variant=variant,
        n_batch=n_batch,
        shape=(int(shape_h), int(shape_w)),
        strings=(tuple(sub_strings[0]), tuple(sub_strings[1])),
    )


__all__ = [
    "CHENG2020_VERSION",
    "Cheng2020Error",
    "Cheng2020Payload",
    "MAGIC_CHENG2020",
    "TARGET_SUBSTRATE_HINT",
    "VARIANT_ANCHOR",
    "VARIANT_ATTENTION",
    "decode_cheng2020",
    "deserialize_cheng2020",
    "encode_cheng2020",
    "serialize_cheng2020",
]
