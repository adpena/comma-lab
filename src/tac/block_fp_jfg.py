"""Block-FP codec for JointFrameGenerator (Wave-Ω Ω-3 transplant).

This module implements a per-block exponent-shift quantizer for the canonical
JointFrameGenerator (JFG) renderer (88K params, FiLM-conditioned). It is the
Wave-Ω Ω-3 building block per the blueprint at
``.omx/research/wave_omega_stack_composition_blueprint_20260507_claude.md``.

Origin (external design motivation)
------------------------------------
Selfcomp (PR #56) used a closely related block-FP idea on a SegMap weight blob
(per memory
``reference_pr56_selfcomp_blob_byte_layout_proper_reverse_engineering_20260501.md``).
That is design motivation only. JFG carries FiLM-conditioning layers whose
weight distributions differ from SegMap convolutions, so this module must earn
its own byte and exact-eval evidence before it can support any score claim.

Kill criterion
--------------
Per the Contrarian's note in the council review (blueprint §Component 3):
**any FiLM layer effective bpw > 2.5 → KILL the lane.** A single-FiLM-layer
validation gate (``validate_film_layer_block_fp``) must run BEFORE compress.

Strict-scorer-rule
------------------
This module is a pure weight codec. **No SegNet/PoseNet/scorer load occurs.**
Compress and decompress are pure CPU byte ↔ tensor math. Inflate-side
dispatch is wired in ``submissions/robust_current/inflate_renderer.py`` and
likewise touches no scorer.

Wire-format (BFJ1 v1)
---------------------
::

    [4]  magic = b"BFJ1"
    [4]  version = u32_LE (=1)
    [4]  block_size = u32_LE (default 64; FiLM-protected layers store
                                their weights raw at FP16, regardless)
    [4]  default_dtype_strlen = u32_LE  (bytes in the default dtype string)
    [n]  default dtype string utf8 (e.g., "float32")
    [4]  n_layers = u32_LE
    [...layers...]  layer order is DETERMINISTIC (sorted by name) for
                     byte-reproducibility.

Per-layer header::

    [2]  name_len = u16_LE
    [n]  name utf8
    [1]  shape_dim = u8
    [4*shape_dim]  shape entries u32_LE each
    [1]  dtype_str_len = u8
    [n]  dtype utf8 (e.g. "float32")
    [4]  block_size_layer = u32_LE   (=block_size for non-protected,
                                          =0 for protected/raw FP16 layers)
    [4]  actual_param_count = u32_LE  (numel pre-padding; <= padded len)
    [4]  n_blocks = u32_LE
    [4]  payload_kind = u32_LE   (0 = block-FP int8 mantissa + per-block
                                  exponent; 1 = protected FP16 raw)
    [4]  payload_nbytes = u32_LE   (size of body for this layer)
    [...payload bytes per kind...]

For ``payload_kind == 0`` (block-FP int8 + exponent):
    [int8_mantissa]   ``actual_param_count`` int8 bytes (HWOI-permuted view
                       for 4D Conv2d weights when the encoder was given
                       hwoi_permute=True)
    [block_exponents] ``n_blocks`` int8 bytes — per-block log2 scale

For ``payload_kind == 1`` (protected raw FP16):
    [fp16_data]   ``actual_param_count * 2`` bytes (little-endian half-floats)

The whole serialized envelope is then ``lzma.compress``-ed. When enabled, HWOI
permute is applied to Conv2d 4D tensors before LZMA as a deterministic layout
heuristic; any archive-level benefit must be measured on the exact payload.

Deterministic byte-order
------------------------
Layer order is ``sorted(state_dict.keys())``. All block-fp computations are
torch CPU ops. ``actual_param_count`` is recorded so the decoder can trim
trailing zero-padding.

API summary
-----------
* :class:`BlockFPConfig` — config dataclass.
* :class:`BlockFPTensor` — frozen per-layer payload representation.
* :func:`quantize_jfg_block_fp` — JFG state-dict → dict[str, BlockFPTensor].
* :func:`compress_jfg_block_fp` — dict + envelope → lzma-compressed bytes.
* :func:`decompress_jfg_block_fp` — bytes → state_dict (load-ready).
* :func:`validate_film_layer_block_fp` — Contrarian's single-FiLM-layer gate.
"""
from __future__ import annotations

import lzma
import math
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

__all__ = [
    "BFJ1_VERSION",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_PROTECT_PATTERNS",
    "MAGIC_BFJ1",
    "BlockFPConfig",
    "BlockFPTensor",
    "ValidationResult",
    "compress_jfg_block_fp",
    "decompress_jfg_block_fp",
    "is_film_protected",
    "quantize_jfg_block_fp",
    "validate_film_layer_block_fp",
]

# ── Constants ─────────────────────────────────────────────────────────────

MAGIC_BFJ1: bytes = b"BFJ1"
BFJ1_VERSION: int = 1

DEFAULT_BLOCK_SIZE: int = 64

# Default patterns matching FiLM/conditioning-flagged tensor names. A name
# whose lowercase contains ANY of these substrings is treated as
# FiLM-protected and stored raw at FP16. The patterns deliberately overlap
# with self_compress.SC_PROTECTED_NAME_PATTERNS in spirit — but Block-FP-JFG
# is a separate codec with its own protection list per the Contrarian's
# council caveat (FiLM layers in JFG differ from SegMap conv layers).
DEFAULT_PROTECT_PATTERNS: tuple[str, ...] = (
    "film",
    "cond",
    "pose_mlp",
    "gamma",
    "beta",
    "scale",
    "shift",
)

# Payload kind tags. Recorded per-layer; allows future kinds to be added
# without breaking the envelope.
_PAYLOAD_KIND_BLOCKFP: int = 0
_PAYLOAD_KIND_PROTECTED_FP16: int = 1

# IEEE-safe per-block exponent band — matches block_fp_codec.py's bounds.
_EXP_MIN: int = -32
_EXP_MAX: int = 32

# Conservative compressed-size sanity floor for JFG-class state dicts. The
# envelope plus protected FP16 reservation alone is ~10-15KB; values much
# below this are almost certainly an empty serializer bug.
_SIZE_SANITY_FLOOR_BYTES: int = 1024


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BlockFPConfig:
    """Configuration for the JFG block-FP codec.

    Attributes:
        block_size: Number of int8 mantissa entries that share a single
            int8 per-block exponent. Selfcomp PR#56 used 16 on a SegMap.
            Default 64 here is empirically the byte-floor for Conv2d
            (decoder pack overhead amortizes well for the 0.387 bpw
            target). Tests cover the small-block path explicitly.
        protect_film_layers: When True (DEFAULT per Contrarian/Hotz revisions
            of the council blueprint), tensor names matching any
            ``protect_patterns`` substring are stored raw at FP16 (kind 1).
            When False (used ONLY in the
            :func:`validate_film_layer_block_fp` gate), every tensor goes
            through block-FP. Production callers should NEVER set this False
            outside the validation gate.
        protect_patterns: Lowercase substrings; a name is protected when
            ``any(p in name.lower() for p in protect_patterns)``.
        film_block_size: When ``protect_film_layers=False`` and a FiLM layer
            still goes through block-FP (validation-gate path or aggressive
            ablation), use this smaller block size for FiLM tensors. Default
            32 — FiLM weight tensors are (cond_dim*2, cond_dim) ≈
            small-and-dense, and a smaller block adapts the per-block
            exponent to the FiLM scale band.
        hwoi_permute: When True, Conv2d 4D weights are permuted from
            (out_ch, in_ch, kH, kW) to (kH, kW, out_ch, in_ch) before block
            partitioning. Selfcomp's empirical "+~5% xz gain" comes from this
            layout (memory ref).
        lzma_preset: Preset level for ``lzma.compress``. 9 is the maximum;
            tests use a lower preset to keep wall-clock reasonable.
    """

    block_size: int = DEFAULT_BLOCK_SIZE
    protect_film_layers: bool = True
    protect_patterns: tuple[str, ...] = DEFAULT_PROTECT_PATTERNS
    film_block_size: int = 32
    hwoi_permute: bool = True
    lzma_preset: int = 9


# ── Per-layer dataclasses ─────────────────────────────────────────────────


@dataclass(frozen=True)
class BlockFPTensor:
    """Frozen per-layer block-FP payload (no torch tensors retained).

    Attributes:
        int8_mantissa: Quantized int8 entries; shape == padded numel for
            block-FP layers; shape == 0 for protected FP16 layers (the
            raw FP16 bytes go into ``fp16_payload`` instead).
        block_exponents: One int8 per block; size == ``n_blocks``. Empty
            for protected FP16 layers.
        block_size: The block_size used for THIS layer (may differ from the
            global config if FiLM layers used film_block_size).
        original_shape: Pre-permute, pre-pad torch tensor shape. The decoder
            needs this to inverse-HWOI and trim padding.
        dtype_target: String name of the layer's original dtype (e.g.
            "float32"). The decoder converts back to this dtype before
            ``model.load_state_dict``.
        actual_param_count: Pre-padding ``numel()``; <= len(int8_mantissa).
        was_hwoi_permuted: True iff this is a 4D Conv2d weight that the
            encoder permuted before quantization. Decoder must inverse-permute.
        protected: True iff this layer was stored raw at FP16.
        fp16_payload: Raw little-endian FP16 bytes for protected layers.
            Empty for block-FP layers.
    """

    int8_mantissa: bytes
    block_exponents: bytes
    block_size: int
    original_shape: tuple[int, ...]
    dtype_target: str
    actual_param_count: int
    was_hwoi_permuted: bool
    protected: bool
    fp16_payload: bytes = b""


@dataclass(frozen=True)
class ValidationResult:
    """Result of the single-FiLM-layer block-FP validation gate.

    Attributes:
        layer_name: The tensor name validated.
        roundtrip_mse: MSE between original FP32 weight and the
            block-FP-applied roundtrip. Strictly numeric (no NaN/Inf).
        threshold: The decision threshold the result was compared against.
        passed: True iff ``roundtrip_mse <= threshold``.
        kill: Convenience inverse of ``passed`` — True means lane should be
            killed.
        effective_bpw: Estimated per-weight bit cost (for council kill-floor
            comparison — Contrarian's > 2.5 bpw kill criterion).
    """

    layer_name: str
    roundtrip_mse: float
    threshold: float
    passed: bool
    kill: bool
    effective_bpw: float


# ── Helpers ───────────────────────────────────────────────────────────────


def is_film_protected(name: str, patterns: Iterable[str]) -> bool:
    """Return True iff ``name`` matches any FiLM/protect pattern."""
    lower = name.lower()
    return any(p in lower for p in patterns)


def _hwoi_permute_forward(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Permute (O, I, kH, kW) → (kH, kW, O, I) for 4D Conv2d weights.

    Returns (permuted, did_permute). Non-4D tensors pass through unchanged.
    """
    if weight.dim() == 4:
        # Selfcomp's HWOI: kH, kW, out_ch, in_ch
        permuted = weight.permute(2, 3, 0, 1).contiguous()
        return permuted, True
    return weight.contiguous(), False


def _hwoi_permute_inverse(weight: torch.Tensor) -> torch.Tensor:
    """Inverse of HWOI: (kH, kW, O, I) → (O, I, kH, kW)."""
    if weight.dim() != 4:
        raise ValueError(
            f"_hwoi_permute_inverse expects 4D, got {weight.dim()}D"
        )
    return weight.permute(2, 3, 0, 1).contiguous()


def _quantize_block_fp_int8(
    flat_weight: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a flat float tensor → (int8 mantissa, int8 per-block exponent).

    Per-block algebra::

        e_b   = clip(ceil(log2(max_abs_b / 127)), [_EXP_MIN, _EXP_MAX])
        scale = 2 ** e_b
        m     = clip(round(weight / scale), [-128, 127])
        recon = m * scale

    The mantissa is stored at int8 (1 byte/weight), the exponent at int8
    (1 byte/block). Block size 64 → ``1 + 1/64 ≈ 1.016 bytes/weight``
    raw, which after lzma over highly redundant weight distributions
    typically lands in the 0.4-1.0 bpw band (Selfcomp PR#56 measured
    0.387 bpw on a SegMap analog).

    Raises:
        ValueError: ``block_size <= 0`` or ``flat_weight.dim() != 1``.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if flat_weight.dim() != 1:
        raise ValueError(
            f"_quantize_block_fp_int8 expects 1-D, got {flat_weight.dim()}D"
        )
    if flat_weight.numel() == 0:
        return (
            torch.zeros((0,), dtype=torch.int8),
            torch.zeros((0,), dtype=torch.int8),
        )

    flat = flat_weight.detach().to(torch.float32)
    n = flat.numel()
    nb = (n + block_size - 1) // block_size

    # Pad with zeros to multiple of block_size for the mantissa storage —
    # the actual_param_count is recorded separately so the decoder trims.
    pad_len = nb * block_size - n
    if pad_len > 0:
        flat = torch.cat(
            [flat, torch.zeros(pad_len, dtype=torch.float32)], dim=0
        )

    blocks = flat.view(nb, block_size)
    max_abs = blocks.abs().amax(dim=1)  # (nb,)

    # Per-block exponent: scale = 2^e such that max_abs / scale ≈ 1.
    # We want max_abs * 2^-e to be just under 127 (int8 max), so:
    #     e = ceil(log2(max_abs / 127))
    # All-zero block → e = 0; mantissa = 0 everywhere.
    exps = torch.zeros((nb,), dtype=torch.int32)
    nonzero_mask = max_abs > 0
    if nonzero_mask.any():
        log2_maxabs = torch.log2(
            max_abs[nonzero_mask].clamp(min=torch.finfo(torch.float32).tiny)
        )
        # log2(max_abs/127) = log2(max_abs) - log2(127)
        log2_127 = math.log2(127.0)
        raw_exp = torch.ceil(log2_maxabs - log2_127).to(torch.int32)
        exps[nonzero_mask] = raw_exp.clamp(min=_EXP_MIN, max=_EXP_MAX)

    # Reconstruct per-element scale; broadcast over each block.
    scales = (2.0 ** exps.to(torch.float32)).view(nb, 1)
    scaled = blocks / scales
    rounded = torch.round(scaled).clamp(min=-128, max=127).to(torch.int8)

    return rounded.view(-1).contiguous(), exps.to(torch.int8)


def _dequantize_block_fp_int8(
    int8_mantissa: torch.Tensor,
    block_exponents: torch.Tensor,
    block_size: int,
    actual_param_count: int,
) -> torch.Tensor:
    """Inverse of :func:`_quantize_block_fp_int8`. Returns float32 tensor."""
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if int8_mantissa.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)

    nb = block_exponents.numel()
    expected_padded = nb * block_size
    if int8_mantissa.numel() != expected_padded:
        raise ValueError(
            f"int8_mantissa numel {int8_mantissa.numel()} != "
            f"n_blocks * block_size = {nb} * {block_size} = {expected_padded}"
        )

    mant_f32 = int8_mantissa.to(torch.float32).view(nb, block_size)
    scales = (2.0 ** block_exponents.to(torch.float32)).view(nb, 1)
    recon = (mant_f32 * scales).view(-1)
    return recon[:actual_param_count].contiguous()


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Map a torch.dtype to a stable string name."""
    return str(dtype).removeprefix("torch.")


def _str_to_torch_dtype(s: str) -> torch.dtype:
    """Inverse of :func:`_torch_dtype_to_str`."""
    if not hasattr(torch, s):
        raise ValueError(f"unknown torch dtype string: {s!r}")
    obj = getattr(torch, s)
    if not isinstance(obj, torch.dtype):
        raise ValueError(f"torch.{s} is not a torch.dtype")
    return obj


# ── Quantize ──────────────────────────────────────────────────────────────


def quantize_jfg_block_fp(
    state_dict: Mapping[str, torch.Tensor],
    config: BlockFPConfig,
) -> dict[str, BlockFPTensor]:
    """Quantize a JFG state dict into per-tensor block-FP payloads.

    Per the Contrarian's council caveat, FiLM-flagged layers (matching any
    of ``config.protect_patterns``) are stored raw at FP16 by default. To
    quantize a FiLM layer for the validation gate, callers must pass a
    config with ``protect_film_layers=False``.

    Strict-scorer-rule compliance: pure CPU torch math. No SegNet/PoseNet/
    scorer is loaded.

    Args:
        state_dict: A model state-dict — ``{name: tensor}``. Values must
            be float tensors (any float dtype; will be converted to f32 for
            quantization and back to ``tensor.dtype`` at decode).
        config: BlockFPConfig.

    Returns:
        A dict ``{name: BlockFPTensor}`` with one entry per tensor in the
        input state_dict. The dict iteration order is the input order;
        the on-disk encoder serializes layers sorted by name for byte
        determinism.

    Raises:
        ValueError: a non-float tensor is encountered, or block_size is
            invalid.
        TypeError: the input is not a Mapping.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"quantize_jfg_block_fp expects a state-dict Mapping, got "
            f"{type(state_dict).__name__}"
        )

    out: dict[str, BlockFPTensor] = {}
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"state_dict[{name!r}] is not a torch.Tensor "
                f"(got {type(tensor).__name__})"
            )
        if not tensor.is_floating_point():
            raise ValueError(
                f"state_dict[{name!r}] dtype={tensor.dtype} — block-FP-JFG "
                f"only supports float tensors. Convert ints to floats "
                f"before passing in."
            )
        original_shape = tuple(int(s) for s in tensor.shape)
        dtype_target = _torch_dtype_to_str(tensor.dtype)
        weight = tensor.detach().to(torch.float32).cpu()
        protected = is_film_protected(name, config.protect_patterns)

        if protected and config.protect_film_layers:
            # Store raw FP16. The decoder converts back to dtype_target.
            fp16_bytes = (
                weight.to(torch.float16)
                .contiguous()
                .numpy()
                .tobytes()
            )
            out[name] = BlockFPTensor(
                int8_mantissa=b"",
                block_exponents=b"",
                block_size=0,
                original_shape=original_shape,
                dtype_target=dtype_target,
                actual_param_count=int(weight.numel()),
                was_hwoi_permuted=False,
                protected=True,
                fp16_payload=fp16_bytes,
            )
            continue

        # Block-FP path. Determine block_size for this layer:
        block_size = (
            config.film_block_size
            if (protected and not config.protect_film_layers)
            else config.block_size
        )

        # HWOI permute (Conv2d 4D only, when enabled).
        if config.hwoi_permute:
            permuted, did_permute = _hwoi_permute_forward(weight)
        else:
            permuted, did_permute = weight.contiguous(), False

        flat = permuted.reshape(-1)
        mantissa, exponents = _quantize_block_fp_int8(flat, block_size)
        out[name] = BlockFPTensor(
            int8_mantissa=bytes(mantissa.numpy().tobytes()),
            block_exponents=bytes(exponents.numpy().tobytes()),
            block_size=block_size,
            original_shape=original_shape,
            dtype_target=dtype_target,
            actual_param_count=int(weight.numel()),
            was_hwoi_permuted=did_permute,
            protected=False,
            fp16_payload=b"",
        )
    return out


# ── Compress (envelope + lzma) ────────────────────────────────────────────


def _encode_envelope(
    quantized: dict[str, BlockFPTensor],
    *,
    block_size: int,
    default_dtype: str,
) -> bytes:
    """Pack the deterministic byte envelope. NOT lzma-compressed."""
    parts: list[bytes] = []
    parts.append(MAGIC_BFJ1)
    parts.append(struct.pack("<I", BFJ1_VERSION))
    parts.append(struct.pack("<I", int(block_size)))

    default_dtype_b = default_dtype.encode("utf-8")
    parts.append(struct.pack("<I", len(default_dtype_b)))
    parts.append(default_dtype_b)

    sorted_names = sorted(quantized.keys())
    parts.append(struct.pack("<I", len(sorted_names)))

    for name in sorted_names:
        bft = quantized[name]
        name_b = name.encode("utf-8")
        if len(name_b) > 0xFFFF:
            raise ValueError(
                f"layer name too long for u16 prefix: {len(name_b)} bytes"
            )
        parts.append(struct.pack("<H", len(name_b)))
        parts.append(name_b)

        shape_dim = len(bft.original_shape)
        if shape_dim > 255:
            raise ValueError(
                f"shape rank too large for u8 prefix: {shape_dim}"
            )
        parts.append(struct.pack("<B", shape_dim))
        for s in bft.original_shape:
            parts.append(struct.pack("<I", int(s)))

        dtype_b = bft.dtype_target.encode("utf-8")
        if len(dtype_b) > 255:
            raise ValueError(
                f"dtype string too long for u8 prefix: {len(dtype_b)} bytes"
            )
        parts.append(struct.pack("<B", len(dtype_b)))
        parts.append(dtype_b)

        parts.append(struct.pack("<I", int(bft.block_size)))
        parts.append(struct.pack("<I", int(bft.actual_param_count)))

        if bft.protected:
            kind = _PAYLOAD_KIND_PROTECTED_FP16
            n_blocks = 0
            payload = bft.fp16_payload
        else:
            kind = _PAYLOAD_KIND_BLOCKFP
            n_blocks = len(bft.block_exponents)
            payload = bft.int8_mantissa + bft.block_exponents

        # was_hwoi_permuted: pack into the high bit of payload_kind so we
        # don't need a separate byte. Kind is u32 but values are small.
        kind_word = int(kind)
        if bft.was_hwoi_permuted:
            kind_word |= 0x80000000

        parts.append(struct.pack("<I", n_blocks))
        parts.append(struct.pack("<I", kind_word))
        parts.append(struct.pack("<I", len(payload)))
        parts.append(payload)

    return b"".join(parts)


def compress_jfg_block_fp(
    quantized: dict[str, BlockFPTensor],
    *,
    block_size: int | None = None,
    hwoi_permute: bool = True,
    lzma_preset: int = 9,
    default_dtype: str = "float32",
) -> bytes:
    """Pack quantized layers into the BFJ1 envelope and lzma-compress.

    The HWOI-permute flag here is informational; whether each layer was
    permuted is already recorded in :class:`BlockFPTensor.was_hwoi_permuted`.
    The flag is retained in the signature for documentation/audit clarity.

    The default block_size in the envelope header is informational (the
    per-layer ``block_size`` field is the binding value at decode). When
    ``None``, we infer the most common per-layer block_size in the input.
    """
    del hwoi_permute
    if not quantized:
        raise ValueError("compress_jfg_block_fp: empty quantized dict")

    if block_size is None:
        sizes = [bft.block_size for bft in quantized.values() if bft.block_size > 0]
        if sizes:
            # most common
            from collections import Counter
            block_size = Counter(sizes).most_common(1)[0][0]
        else:
            block_size = DEFAULT_BLOCK_SIZE

    envelope = _encode_envelope(
        quantized,
        block_size=int(block_size),
        default_dtype=default_dtype,
    )
    # On-disk file format: outer 4-byte BFJ1 magic + lzma-compressed
    # inner envelope. The inner envelope ALSO begins with BFJ1 (the
    # canonical envelope-magic check is on the post-lzma bytes), so
    # readers can sanity-check both layers.
    inner = lzma.compress(envelope, preset=int(lzma_preset))
    return MAGIC_BFJ1 + inner


# ── Decompress ────────────────────────────────────────────────────────────


def _decode_envelope(blob: bytes) -> tuple[dict[str, BlockFPTensor], int]:
    """Inverse of :func:`_encode_envelope`. Returns (quantized, header_block_size)."""
    if blob[:4] != MAGIC_BFJ1:
        raise ValueError(
            f"BFJ1 decode: expected magic {MAGIC_BFJ1!r}, got {blob[:4]!r}"
        )
    offset = 4
    (version,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4
    if version != BFJ1_VERSION:
        raise ValueError(
            f"BFJ1 decode: unsupported version {version}, expected {BFJ1_VERSION}"
        )

    (header_block_size,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4

    (default_dtype_len,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4
    _ = blob[offset:offset + default_dtype_len].decode("utf-8")  # informational
    offset += default_dtype_len

    (n_layers,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4

    out: dict[str, BlockFPTensor] = {}
    for _layer_idx in range(n_layers):
        (name_len,) = struct.unpack("<H", blob[offset:offset + 2])
        offset += 2
        name = blob[offset:offset + name_len].decode("utf-8")
        offset += name_len

        shape_dim = blob[offset]
        offset += 1
        shape: list[int] = []
        for _ in range(shape_dim):
            (s,) = struct.unpack("<I", blob[offset:offset + 4])
            offset += 4
            shape.append(s)

        dtype_len = blob[offset]
        offset += 1
        dtype_target = blob[offset:offset + dtype_len].decode("utf-8")
        offset += dtype_len

        (block_size,) = struct.unpack("<I", blob[offset:offset + 4])
        offset += 4
        (actual_param_count,) = struct.unpack("<I", blob[offset:offset + 4])
        offset += 4
        (n_blocks,) = struct.unpack("<I", blob[offset:offset + 4])
        offset += 4
        (kind_word,) = struct.unpack("<I", blob[offset:offset + 4])
        offset += 4
        (payload_len,) = struct.unpack("<I", blob[offset:offset + 4])
        offset += 4
        payload = blob[offset:offset + payload_len]
        offset += payload_len

        was_hwoi_permuted = bool(kind_word & 0x80000000)
        kind = kind_word & 0x7FFFFFFF

        if kind == _PAYLOAD_KIND_PROTECTED_FP16:
            out[name] = BlockFPTensor(
                int8_mantissa=b"",
                block_exponents=b"",
                block_size=0,
                original_shape=tuple(shape),
                dtype_target=dtype_target,
                actual_param_count=actual_param_count,
                was_hwoi_permuted=False,
                protected=True,
                fp16_payload=payload,
            )
        elif kind == _PAYLOAD_KIND_BLOCKFP:
            mantissa_nbytes = payload_len - n_blocks
            if mantissa_nbytes < 0:
                raise ValueError(
                    f"BFJ1 decode: layer {name!r} payload_len={payload_len} "
                    f"< n_blocks={n_blocks}"
                )
            int8_mantissa = payload[:mantissa_nbytes]
            block_exponents = payload[mantissa_nbytes:]
            out[name] = BlockFPTensor(
                int8_mantissa=int8_mantissa,
                block_exponents=block_exponents,
                block_size=block_size,
                original_shape=tuple(shape),
                dtype_target=dtype_target,
                actual_param_count=actual_param_count,
                was_hwoi_permuted=was_hwoi_permuted,
                protected=False,
                fp16_payload=b"",
            )
        else:
            raise ValueError(
                f"BFJ1 decode: unknown payload_kind {kind} for layer {name!r}"
            )

    return out, header_block_size


def decompress_jfg_block_fp(
    blob: bytes,
    model_skeleton: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Decompress a BFJ1 blob → state_dict ready for ``model.load_state_dict``.

    Args:
        blob: lzma-compressed BFJ1 envelope.
        model_skeleton: Optional. When provided, the returned state_dict is
            sanity-checked to contain exactly the keys of
            ``model_skeleton.state_dict()``. Mismatch raises ValueError.

    Strict-scorer-rule compliance: pure CPU torch math.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"decompress_jfg_block_fp expects bytes-like, got "
            f"{type(blob).__name__}"
        )
    blob = bytes(blob)
    # Three accepted on-disk forms for forward compatibility:
    #   1. BFJ1 + lzma(BFJ1 + envelope)         — canonical (compress_jfg_block_fp output)
    #   2. lzma(BFJ1 + envelope)                — legacy/internal
    #   3. BFJ1 + envelope (no lzma)            — already-decompressed envelope
    if blob[:4] == MAGIC_BFJ1:
        # Canonical form: outer magic + inner lzma(BFJ1 + envelope)
        # Or already-decompressed envelope (form 3 — no lzma).
        rest = blob[4:]
        if rest[:6] == b"\xfd7zXZ\x00":
            envelope = lzma.decompress(rest)
            if envelope[:4] != MAGIC_BFJ1:
                raise ValueError(
                    f"BFJ1 decompress: lzma payload does not start with magic "
                    f"{MAGIC_BFJ1!r}"
                )
        else:
            # Form 3: outer BFJ1 then raw envelope (no lzma) — only valid if
            # the rest starts with the version field. We test by re-prepending
            # the magic and treating the whole blob as the envelope.
            envelope = blob
    else:
        # Form 2: legacy — pure lzma stream wrapping a BFJ1 envelope.
        envelope = lzma.decompress(blob)
        if envelope[:4] != MAGIC_BFJ1:
            raise ValueError(
                f"BFJ1 decompress: lzma payload does not start with magic "
                f"{MAGIC_BFJ1!r}"
            )

    quantized, _ = _decode_envelope(envelope)

    state_dict: dict[str, torch.Tensor] = {}
    for name, bft in quantized.items():
        target_dtype = _str_to_torch_dtype(bft.dtype_target)
        if bft.protected:
            # FP16 payload → numpy → torch → cast to target dtype.
            import numpy as np
            arr = np.frombuffer(bft.fp16_payload, dtype=np.float16).copy()
            if arr.size != bft.actual_param_count:
                raise ValueError(
                    f"BFJ1 decompress: protected layer {name!r} has "
                    f"{arr.size} fp16 entries, expected {bft.actual_param_count}"
                )
            tensor = torch.from_numpy(arr).to(target_dtype).reshape(
                bft.original_shape
            )
            state_dict[name] = tensor
            continue

        import numpy as np
        mant = np.frombuffer(bft.int8_mantissa, dtype=np.int8).copy()
        exps = np.frombuffer(bft.block_exponents, dtype=np.int8).copy()
        flat = _dequantize_block_fp_int8(
            torch.from_numpy(mant),
            torch.from_numpy(exps),
            bft.block_size,
            bft.actual_param_count,
        )
        # Reshape into (post-permute or natural) shape, then inverse-HWOI.
        if bft.was_hwoi_permuted and len(bft.original_shape) == 4:
            o, i, kh, kw = bft.original_shape
            permuted = flat.reshape(kh, kw, o, i)
            tensor = _hwoi_permute_inverse(permuted)
        else:
            tensor = flat.reshape(bft.original_shape)
        state_dict[name] = tensor.to(target_dtype).contiguous()

    if model_skeleton is not None:
        expected_keys = set(model_skeleton.state_dict().keys())
        got_keys = set(state_dict.keys())
        missing = expected_keys - got_keys
        extra = got_keys - expected_keys
        if missing or extra:
            raise ValueError(
                f"BFJ1 decompress: state_dict key mismatch — "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

    return state_dict


# ── Validation gate (Contrarian's Phase F precondition) ───────────────────


def validate_film_layer_block_fp(
    layer_weight: torch.Tensor,
    config: BlockFPConfig,
    *,
    layer_name: str = "<film_layer>",
    mse_threshold: float = 1e-3,
) -> ValidationResult:
    """Single-FiLM-layer block-FP validation gate.

    Computes round-trip MSE on a single FiLM-flagged layer with block-FP
    APPLIED (i.e., with ``protect_film_layers`` temporarily disabled). If
    round-trip MSE exceeds ``mse_threshold``, the FiLM weights are too
    sensitive for block-FP — the lane should be killed.

    Per the Contrarian's note in the council blueprint review, kill criterion
    is **any FiLM layer effective bpw > 2.5**. The bpw check is captured in
    :attr:`ValidationResult.effective_bpw` for downstream gating; the
    primary kill condition here is MSE exceeding ``mse_threshold``.

    Args:
        layer_weight: A FiLM-flagged tensor. Must be a float torch.Tensor.
        config: Used to derive block_size (we override
            ``protect_film_layers=False`` internally for this call).
        layer_name: Display name in the result for audit logs.
        mse_threshold: Kill threshold; default 1e-3.

    Returns:
        :class:`ValidationResult`.
    """
    if not isinstance(layer_weight, torch.Tensor):
        raise TypeError(
            f"validate_film_layer_block_fp expects torch.Tensor, got "
            f"{type(layer_weight).__name__}"
        )
    if not layer_weight.is_floating_point():
        raise ValueError(
            f"validate_film_layer_block_fp expects float tensor, got "
            f"{layer_weight.dtype}"
        )
    if layer_weight.numel() == 0:
        raise ValueError("validate_film_layer_block_fp: empty tensor")

    # Force the validation path: this layer is treated as block-FP-eligible.
    forced = BlockFPConfig(
        block_size=config.block_size,
        protect_film_layers=False,  # FORCE block-FP on this layer
        protect_patterns=config.protect_patterns,
        film_block_size=config.film_block_size,
        hwoi_permute=config.hwoi_permute,
        lzma_preset=config.lzma_preset,
    )

    sd = {layer_name: layer_weight.detach().to(torch.float32).cpu()}
    quantized = quantize_jfg_block_fp(sd, forced)
    blob = compress_jfg_block_fp(quantized, lzma_preset=config.lzma_preset)
    restored = decompress_jfg_block_fp(blob)
    recon = restored[layer_name]

    diff = (recon.to(torch.float32) - sd[layer_name]).pow(2)
    mse_val = float(diff.mean().item())
    if not math.isfinite(mse_val):
        raise ValueError(
            f"validate_film_layer_block_fp: non-finite MSE for {layer_name!r}"
        )

    # Effective bpw: total compressed bytes (envelope+lzma) per param.
    eff_bpw = float(8.0 * len(blob) / max(1, layer_weight.numel()))

    passed = mse_val <= mse_threshold
    return ValidationResult(
        layer_name=layer_name,
        roundtrip_mse=mse_val,
        threshold=float(mse_threshold),
        passed=bool(passed),
        kill=bool(not passed),
        effective_bpw=eff_bpw,
    )
