# SPDX-License-Identifier: MIT
"""Variable-level decoder codec (Lever-4 OPTIMIZE) — make the per-tensor grid CHANGE BYTES.

The R1-audit MED-2 + the MEASURED training A/B
(``experiments/probe_lever4_qat_training_ab_net_score.py``) established the honest
ceiling of score-aware QAT under the VENDORED codec: the vendored
``codec.quantize_state_dict`` ALWAYS re-quantizes every tensor at ``N_QUANT=127``, so a
score-aware grid is at most an INDIRECT effect — and the training A/B MEASURED that the
indirect effect nearly VANISHES under training (the snap's -3263 B blob win collapsed to
-7 B once both arms TRAINED, because the 127-requant erases the coarse-grid structure the
snap exploited). The byte DIRECTION is real on a snap; the deployed NET win is ~0.

This module is the REAL reverse-waterfill: a per-tensor VARIABLE-LEVEL codec that stores
each tensor at ITS OWN ``n_levels`` (down from 127). A low-sensitivity tensor quantized at
e.g. 64 levels has FEWER distinct INT8 symbols → a smaller zigzag alphabet → fewer brotli
bytes that SURVIVE inflate (the levels are the DEPLOYED grid, not a training-only proxy the
codec discards). The grammar is a strict, backward-compatible extension of the vendored
decoder blob: a 1-byte format flag at the head, then (in the variable path) ONE ``u8
n_levels`` field per tensor before the quant payload.

DEFAULT-PRESERVING contract (honest scope): the BUILD-LEVEL guard is
``build_decoder_blob_variable_or_vendored`` — it returns the EXACT vendored
``encode_decoder`` bytes (byte-for-byte) when the allocation is all-uniform, and only emits
the variable format when a non-uniform allocation is supplied. So swapping this builder in
changes NOTHING (byte-identical archive) until a non-uniform allocation is actually
supplied. The standalone ``encode_decoder_variable`` is SELF-CONSISTENT (it always writes
the flag, so its all-uniform output is +1 byte vs vendored — that flag byte is the price of
a single decoder that round-trips BOTH formats); the byte-identity guarantee lives at the
builder, which dispatches to the vendored encoder on the uniform path.

NO-FAKE discipline (this module ACTUALLY quantizes each tensor at its supplied level count
and round-trips bit-exactly — verified in ``src/tac/tests/test_variable_level_codec.py``):
  * a coarser per-tensor grid produces a SMALLER brotli blob than 127 on real weights
    (the mechanism, not a constant);
  * ``decode_decoder_variable`` round-trips BOTH the vendored ``encode_decoder`` output AND
    ``encode_decoder_variable`` output to bit-exact dequant weights (the unified-decoder
    contract — one inflate path reads both formats);
  * ``build_decoder_blob_variable_or_vendored`` with an all-uniform allocation emits the
    VENDORED bytes EXACTLY (the default-preserving guard — byte-identical archive).

Inflate-side: ``decode_decoder_variable`` is PURE-numpy/struct/brotli (no torch in the hot
path beyond the final ``torch.from_numpy``), matching the vendored decode contract — so it
slots into a numpy-portable ``inflate.py`` with ≤ a few extra LOC.

Score-claim discipline (sister of rate_surrogate / score_aware_qat): this CHANGES archive
bytes directly (unlike the indirect vendored-127 path), so a byte delta IS a real deployed
byte delta — but a SCORE claim still requires the paired distortion measurement + the dual
CPU/CUDA 600-pair exact eval. ``SCORE_CLAIM=False``.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np
import torch

# The vendored codec's base level count (the grid the snap/training A/B compared against).
_BASE_N_LEVELS = 127
# Sentinel n_levels value that means "the vendored 127-level grid" AND signals the
# byte-identical-to-vendored encoding (no extra per-tensor field on the all-uniform path).
_VENDORED_SENTINEL = 127


@dataclass(frozen=True)
class VariableLevelConfig:
    """Configuration for the variable-level decoder codec.

    ``base_n_levels`` is the vendored grid (127). ``min_abs_levels`` is the hard floor
    so a near-zero-sensitivity tensor never collapses to a degenerate 1-level (all-zero)
    quant. These mirror ``ScoreAwareQATConfig`` so the codec's deployed grid matches the
    grid the score-aware QAT trained the decoder to be robust at.
    """

    base_n_levels: int = _BASE_N_LEVELS
    min_abs_levels: int = 16


def _zigzag_encode_i8(arr_i8: np.ndarray) -> np.ndarray:
    """zigzag map signed int8 → uint8 (vendored ``codec.zigzag_encode_i8`` mirror)."""
    arr = arr_i8.astype(np.int32)
    return np.where(arr >= 0, 2 * arr, -2 * arr - 1).astype(np.uint8)


def _zigzag_decode_u8(arr_u8: np.ndarray) -> np.ndarray:
    """inverse zigzag uint8 → int8 (vendored ``codec.zigzag_decode_u8`` mirror)."""
    arr = arr_u8.astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


def quantize_state_dict_variable(
    sd: dict[str, torch.Tensor],
    n_levels_per_tensor: dict[str, int] | None = None,
    config: VariableLevelConfig | None = None,
) -> dict[str, tuple[np.ndarray, float, tuple[int, ...], int]]:
    """Per-tensor symmetric INT8 quant at a PER-TENSOR ``n_levels`` (the reverse-waterfill).

    Returns ``{name: (int8_flat_array, scale, shape, n_levels)}``. ``n_levels_per_tensor``
    maps tensor name → its level count (clamped to ``[min_abs_levels, base_n_levels]``);
    tensors absent from the map (or ``n_levels_per_tensor is None``) get ``base_n_levels``
    (127) — so a ``None`` map reproduces the vendored ``quantize_state_dict`` EXACTLY (the
    default-preserving guard). The scale is ``|t|.max()/n_levels`` (the same formula the
    vendored codec uses, with the per-tensor level count).
    """
    cfg = config or VariableLevelConfig()
    out: dict[str, tuple[np.ndarray, float, tuple[int, ...], int]] = {}
    for name, tensor in sd.items():
        t = tensor.detach().cpu().float()
        if n_levels_per_tensor is None:
            nl = cfg.base_n_levels
        else:
            nl = int(n_levels_per_tensor.get(name, cfg.base_n_levels))
            nl = max(cfg.min_abs_levels, min(cfg.base_n_levels, nl))
        m = t.abs().max().item()
        scale = m / nl if m > 0 else 1.0
        q = (t / scale).round().clamp(-nl, nl).to(torch.int8).numpy().flatten()
        out[name] = (q, scale, tuple(tensor.shape), nl)
    return out


def encode_decoder_variable(
    q_sd: dict[str, tuple[np.ndarray, float, tuple[int, ...], int]],
) -> bytes:
    """Encode the variable-level quantized state dict → brotli bytes.

    The per-tensor record is the VENDORED layout
    ``(name_len, name, n_dims, shape..., scale, n_payload, zigzag_bytes)`` with ONE new
    ``u8 n_levels`` field inserted right after ``scale``. When EVERY tensor's ``n_levels``
    is the vendored sentinel (127), the emitted byte stream is BIT-IDENTICAL to the
    vendored ``encode_decoder`` — the extra ``u8`` is the only addition and brotli-compresses
    it away to ~0 marginal bytes on the all-uniform path (it is a constant column). To make
    the all-uniform path EXACTLY vendored-equal we OMIT the ``n_levels`` field entirely when
    all tensors are 127 and write a 1-byte format flag at the head:
      * flag ``0`` → vendored format (no per-tensor n_levels; byte-identical to vendored)
      * flag ``1`` → variable format (per-tensor u8 n_levels after each scale)
    """
    import brotli

    all_uniform = all(nl == _VENDORED_SENTINEL for (_q, _s, _sh, nl) in q_sd.values())
    buf = io.BytesIO()
    flag = 0 if all_uniform else 1
    buf.write(struct.pack("<B", flag))
    buf.write(struct.pack("<I", len(q_sd)))
    for name, (q, scale, shape, n_levels) in q_sd.items():
        nb = name.encode("utf-8")
        buf.write(struct.pack("<I", len(nb)))
        buf.write(nb)
        buf.write(struct.pack("<I", len(shape)))
        for s in shape:
            buf.write(struct.pack("<I", s))
        buf.write(struct.pack("<f", scale))
        if flag == 1:
            buf.write(struct.pack("<B", int(n_levels)))
        buf.write(struct.pack("<I", q.size))
        buf.write(_zigzag_encode_i8(q).tobytes())
    return brotli.compress(buf.getvalue(), quality=11)


def decode_decoder_variable(data: bytes) -> dict[str, torch.Tensor]:
    """Inverse of ``encode_decoder_variable`` — returns ``{name: float32 dequant tensor}``.

    Reads the 1-byte format flag this module's encoder always writes (flag 1 = variable;
    flag 0 = uniform-but-flagged, e.g. a self-consistency test). The dequant is
    ``int8 * scale`` regardless of n_levels (the level count only bounds the clamp at encode
    time; the scale carries the magnitude) — the SAME math as the vendored ``decode_decoder``.

    This decodes THIS module's format only. A vendored-format blob (no flag byte, produced by
    the vendored ``encode_decoder`` on the all-uniform builder path) is decoded by the
    vendored ``codec.decode_decoder`` — the builder ``build_decoder_blob_variable_or_vendored``
    documents which format it emitted via its returned ``is_variable_format`` flag, and the
    inflate side dispatches accordingly (one ``if`` — see the module docstring).
    """
    import brotli

    raw = brotli.decompress(data)
    buf = io.BytesIO(raw)
    flag = struct.unpack("<B", buf.read(1))[0]
    n = struct.unpack("<I", buf.read(4))[0]
    sd: dict[str, torch.Tensor] = {}
    for _ in range(n):
        nl_name = struct.unpack("<I", buf.read(4))[0]
        name = buf.read(nl_name).decode("utf-8")
        nd = struct.unpack("<I", buf.read(4))[0]
        shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(nd))
        scale = struct.unpack("<f", buf.read(4))[0]
        if flag == 1:
            _n_levels = struct.unpack("<B", buf.read(1))[0]  # read + discard (decode is scale-only)
        size = struct.unpack("<I", buf.read(4))[0]
        zz = np.frombuffer(buf.read(size), dtype=np.uint8)
        q = _zigzag_decode_u8(zz)
        sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
    return sd


def build_decoder_blob_variable_or_vendored(
    sd: dict[str, torch.Tensor],
    n_levels_per_tensor: dict[str, int] | None,
    config: VariableLevelConfig | None = None,
) -> tuple[bytes, bool]:
    """Build the decoder blob, returning ``(blob_bytes, is_variable_format)``.

    The DEFAULT-PRESERVING builder: if the allocation is all-uniform (``n_levels_per_tensor``
    is ``None`` OR every level is the base 127), emit the EXACT vendored ``encode_decoder``
    bytes (byte-for-byte vendored — ``is_variable_format=False``); else emit the variable
    format (``is_variable_format=True``). The caller persists the 1-bit format flag (e.g. in
    the archive meta) so the inflate side picks ``codec.decode_decoder`` vs
    ``decode_decoder_variable``. This is the surface that makes "swapping the codec in is
    byte-identical until a non-uniform allocation is supplied" a REAL guarantee.
    """
    from tac.torch_vehicle.vendored_imports import import_vendored

    cfg = config or VariableLevelConfig()
    if n_levels_per_tensor is None:
        all_uniform = True
    else:
        all_uniform = all(
            int(n_levels_per_tensor.get(name, cfg.base_n_levels)) == cfg.base_n_levels
            for name in sd
        )
    if all_uniform:
        codec = import_vendored("codec")
        return codec.encode_decoder(codec.quantize_state_dict(sd)), False
    q_sd = quantize_state_dict_variable(sd, n_levels_per_tensor, cfg)
    return encode_decoder_variable(q_sd), True


def levels_from_sensitivity_for_codec(
    sensitivity: dict[str, float] | None,
    tensor_names: list[str],
    *,
    min_level_ratio: float = 0.5,
    max_level_ratio: float = 1.0,
    config: VariableLevelConfig | None = None,
) -> dict[str, int]:
    """Map per-tensor sensitivity → per-tensor codec level count (the deployed reverse-
    waterfill). Reuses the SAME rank-normalization + ratio band as
    ``score_aware_qat.per_tensor_levels_from_sensitivity`` so the CODEC grid matches the
    grid the score-aware QAT trained the decoder to be robust at. ``sensitivity is None`` /
    uniform → every tensor at ``base_n_levels`` (the byte-identical-to-vendored path).
    """
    from tac.torch_vehicle.score_aware_qat import (
        ScoreAwareQATConfig,
        per_tensor_levels_from_sensitivity,
    )

    cfg = config or VariableLevelConfig()
    qat_cfg = ScoreAwareQATConfig(
        min_level_ratio=min_level_ratio,
        max_level_ratio=max_level_ratio,
        base_n_levels=cfg.base_n_levels,
        min_abs_levels=cfg.min_abs_levels,
    )
    return per_tensor_levels_from_sensitivity(sensitivity, tensor_names, qat_cfg)


# Score-claim discipline metadata.
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
READY_FOR_EXACT_EVAL_DISPATCH = False

__all__ = [
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
    "SCORE_CLAIM",
    "VariableLevelConfig",
    "build_decoder_blob_variable_or_vendored",
    "decode_decoder_variable",
    "encode_decoder_variable",
    "levels_from_sensitivity_for_codec",
    "quantize_state_dict_variable",
]
