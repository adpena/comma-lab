"""Lane Ω-W-V2 — water-fill + arithmetic terminal on Selfcomp/block-FP weights.

V2 layers a static-histogram arithmetic coder ON TOP of V1's per-channel
qint allocation. The canonical stack (see memory file
``project_codec_stacking_composition_canonical_orders_20260429.md``) is:

    block-FP  →  water-fill  →  hyperprior  →  arithmetic  ← TERMINAL

V2 implements the first, second, and FOURTH layers. The hyperprior layer
is intentionally DEFERRED to V3 because the council-vetted bit-budget codex
verdict notes:

    "Hyperprior is both bit-saver AND bit-cost; charge side-info inside
     archive.zip. No 'free prior' accounting. Implementation only where
     stream size amortizes side-info."

Selfcomp's renderer at 88K params × 1.017 bpw ≈ ~11KB of qint payload —
borderline for hyperprior amortization (memory says >5KB amortizes,
but 11KB is in the "barely worth it" band). V2 ships static-histogram
arithmetic which has zero learnable overhead and dominates xz/LZMA on
low-entropy ternary streams. V3 will add a tiny scale-hyperprior IF
empirical data shows it beats V2 by >50bp after full pipeline measurement
(Carmack hard-kill rule from the same codex).

Predicted savings band [prediction]: 200-450 basis points (2-4.5% archive
shrink) on Selfcomp-class block-FP renderers; 70-90 bp on pose-only. These
numbers come from the grand-council stacking codex 2026-04-29 PM
(/tmp/codex_runs/stacking_composition_unlimited_compress.log) and have NOT
been empirically verified on a 384x512 archive yet. Empirical verification
must add an [empirical:<artifact>] tag per CLAUDE.md FORBIDDEN PATTERNS.

Pipeline (encode_omega_w_v2):
    1. Block-FP eligibility check (raise ``GateRegression`` if ineligible).
    2. Per-channel water-fill bit allocation reusing V1.
    3. Per-channel quantization to allocated qint ladder.
    4. Static-histogram arithmetic coding terminal.
    5. Write OWV2 self-describing header + arithmetic payload.

Pipeline (decode_omega_w_v2):
    1. Parse magic byte (``OWV2``) + header.
    2. Decode arithmetic payload to per-channel qint stream.
    3. Reconstruct float weights via per-channel exponent + qint * 2**exp.
    4. Return reconstructed weight tensor (CPU, float32, OIHW).

CLAUDE.md compliance gates:
    * No scorer load at decode time — pure-math byte→tensor (Check H STRICT).
    * No silent defaults — every public function arg is None or explicit
      (Check 81 STRICT).
    * Hard overhead gate — encode raises ``GateRegression`` if encoded +
      header >= V1 raw qint bytes for the SAME tensor (memory: codec stacking
      "if encoded + header >= V1, keep V1"). No silently-larger output ships.
    * No GPU dependency; encode/decode are pure CPU.
    * No mutation of V1 module; V2 is additive only.
"""
from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass

import numpy as np
import torch

from tac.arithmetic_qint_codec import (
    decode_qints_arithmetic,
    encode_qints_arithmetic,
)
from tac.block_fp_codec import _EXP_MAX, _EXP_MIN
from tac.water_filling_codec import (
    QINT_LEVELS,
    WaterFillError,
    water_fill_bit_budget,
)


# ── magic bytes (added to canonical registry — see tac.codec_magic_registry) ──
OWV2_MAGIC: bytes = b"OWV2"
"""Lane Ω-W-V2 self-describing payload magic. 4 bytes, ASCII."""

OWV2_VERSION: int = 1
"""Header version. Bumped on any wire-format change."""

OWV2_ARITH_TABLE_VERSION: int = 1
"""Static-histogram arithmetic table version. Bumped if the cumulative-frequency
schema changes."""


# ── exceptions ────────────────────────────────────────────────────────────


class GateRegression(ValueError):
    """Raised when V2 encode would produce >= bytes than V1 raw qint output.

    The hard-overhead gate exists per memory ``project_codec_stacking_
    composition_canonical_orders_20260429.md`` Carmack rule: any candidate
    stack that cannot beat its simpler baseline by 50 bp AFTER overhead
    DIES. V2 raises rather than silently shipping a regression.
    """


class BlockFPIneligible(GateRegression):
    """Raised when input weights fail block-FP eligibility checks.

    Block-FP requires a 4-D conv weight (O, I, kH, kW) with finite values.
    Linear/embedding/bias tensors are NOT eligible — they should keep their
    own codec (linear_q_per_tensor_v1) and fall through V1 unmodified.
    """


# ── eligibility (the gate the council demanded) ───────────────────────────


def _check_block_fp_eligible(weights_block_fp: torch.Tensor) -> None:
    """Raise BlockFPIneligible if the input is not a valid block-FP candidate.

    Eligibility requirements (from block_fp_codec.encode_conv_weight):
        * Tensor (not None, not scalar).
        * Rank == 4: shape (O, I, kH, kW).
        * Finite values everywhere (no NaN/Inf — see encode_conv_weight
          assertion at block_fp_codec.py:469).
        * Numel >= 1 (zero-element tensors have nothing to allocate).
        * O >= 1 (at least one output channel).
    """
    if weights_block_fp is None:
        raise BlockFPIneligible(
            "encode_omega_w_v2: weights_block_fp is None — pass the conv "
            "weight tensor explicitly, never a default. (CLAUDE.md silent-"
            "default audit class.)"
        )
    if not torch.is_tensor(weights_block_fp):
        raise BlockFPIneligible(
            f"encode_omega_w_v2: weights_block_fp must be a torch.Tensor, "
            f"got {type(weights_block_fp).__name__}. Block-FP only applies "
            f"to dense conv weights."
        )
    if weights_block_fp.dim() != 4:
        raise BlockFPIneligible(
            f"encode_omega_w_v2: block-FP requires a 4-D conv weight "
            f"(O, I, kH, kW); got rank-{weights_block_fp.dim()} shape "
            f"{tuple(weights_block_fp.shape)}. Linear / embedding / bias "
            f"tensors must use linear_q_per_tensor_v1 instead."
        )
    if weights_block_fp.numel() == 0:
        raise BlockFPIneligible(
            "encode_omega_w_v2: empty tensor (numel=0); nothing to allocate."
        )
    if weights_block_fp.shape[0] < 1:
        raise BlockFPIneligible(
            f"encode_omega_w_v2: zero output channels (shape={tuple(weights_block_fp.shape)})."
        )
    if not torch.isfinite(weights_block_fp).all():
        n_bad = int((~torch.isfinite(weights_block_fp)).sum().item())
        raise BlockFPIneligible(
            f"encode_omega_w_v2: weights contain {n_bad} non-finite value(s) "
            f"(NaN/Inf). Refuse to silently zero — fix upstream training."
        )


def _hessian_eligible(hessian: torch.Tensor, num_channels: int) -> None:
    """Validate Hessian shape + finiteness."""
    if hessian is None:
        raise GateRegression(
            "encode_omega_w_v2: hessian is None — must pass per-channel "
            "Hessian (use water_filling_codec.estimate_per_channel_hessian "
            "to compute). Silent default would let V2 ship without "
            "scorer-aware allocation."
        )
    if not torch.is_tensor(hessian):
        raise GateRegression(
            f"encode_omega_w_v2: hessian must be a torch.Tensor, "
            f"got {type(hessian).__name__}."
        )
    if hessian.dim() != 1 or int(hessian.shape[0]) != num_channels:
        raise GateRegression(
            f"encode_omega_w_v2: hessian shape {tuple(hessian.shape)} != "
            f"({num_channels},). Expect per-output-channel curvature."
        )
    if not torch.isfinite(hessian).all():
        n_bad = int((~torch.isfinite(hessian)).sum().item())
        raise GateRegression(
            f"encode_omega_w_v2: hessian contains {n_bad} non-finite value(s). "
            f"Loss is NaN/Inf — fix upstream."
        )
    if (hessian < 0).any():
        raise GateRegression(
            "encode_omega_w_v2: hessian has negative entries — H_c must be "
            "Σ(∂L/∂w)² ≥ 0. Check estimator path."
        )


# ── per-channel quantization (for V2 internal use) ────────────────────────


def _quantize_to_per_channel_qint(
    weights: torch.Tensor,
    qint_per_channel: list[int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Quantize per-channel block-FP-style.

    Mirrors block_fp_codec.encode_conv_weight algebra: per output channel c,
        Q_c = qint_per_channel[c]
        max_abs_c = max(|w[c]|)
        exp_c = ceil(log2(max_abs_c / Q_c)) (clamped to [_EXP_MIN, _EXP_MAX])
        qint[c] = round(w[c] / 2**exp_c).clamp(-Q_c, +Q_c)

    Returns:
        flat_qints: int32 array shape (O*I*kH*kW,) of per-element qints,
            stored ROW-MAJOR by output channel (so we can decode-then-reshape
            back to (O, I, kH, kW)).
        exponents: int32 array shape (O,) of per-channel exponents.
        per_channel_qmax_used: list[int] length O of the actual Q_c values.
    """
    o, i, kh, kw = weights.shape
    if len(qint_per_channel) != o:
        raise GateRegression(
            f"_quantize_to_per_channel_qint: qint_per_channel length "
            f"{len(qint_per_channel)} != O={o}"
        )
    w = weights.detach().to(torch.float32)
    qint_oihw = torch.zeros((o, i, kh, kw), dtype=torch.int32)
    exponents = torch.zeros((o,), dtype=torch.int32)
    qmax_used: list[int] = []
    for c in range(o):
        Qc = int(qint_per_channel[c])
        if Qc not in QINT_LEVELS:
            raise GateRegression(
                f"_quantize_to_per_channel_qint: Q_c={Qc} not in canonical "
                f"ladder {QINT_LEVELS}"
            )
        wc = w[c]
        max_abs = float(wc.abs().max().item())
        if max_abs == 0.0 or not math.isfinite(max_abs):
            exponents[c] = 0
            qmax_used.append(Qc)
            continue
        exp_f = math.ceil(math.log2(max_abs / Qc))
        e = max(_EXP_MIN, min(_EXP_MAX, int(exp_f)))
        exponents[c] = e
        scale = 2.0 ** e
        scaled = (wc / scale).round().clamp(-Qc, Qc)
        qint_oihw[c] = scaled.to(torch.int32)
        qmax_used.append(Qc)
    flat = qint_oihw.reshape(-1).to(torch.int32).cpu().numpy()
    exp_np = exponents.cpu().numpy()
    return flat, exp_np, qmax_used


# ── V1-baseline byte estimator (the regression gate compares against this) ──


def _v1_raw_qint_byte_estimate(
    weights: torch.Tensor,
    qint_per_channel: list[int],
) -> int:
    """Estimate V1's raw qint byte count for the same allocation.

    V1 stores qint as INT8 dense + per-channel int32 exponents in
    block_fp_codec.encode_conv_weight. The raw byte cost (BEFORE outer
    tar.xz compression) is::

        bytes_v1_raw = O * I * kH * kW * 1   # int8 qint
                     + O * 4                  # int32 exponent
                     + small fixed header

    V2 must beat this AFTER its arithmetic coder + OWV2 header. This is the
    apples-to-apples comparison: both are pre-outer-compression byte budgets
    paid by archive.zip's DEFLATE step. Outer compression typically helps
    V1 more than V2 (because xz LZMA exploits the same redundancy as the
    arithmetic coder), so the gate is conservative — we still want V2 to
    beat V1 on raw bytes, then verify post-archive that the win persists.

    Returns:
        int byte count for V1 raw representation of the same allocation.
    """
    o, i, kh, kw = weights.shape
    qint_bytes = int(o * i * kh * kw)
    exponents_bytes = int(o * 4)
    fixed_header = 32  # block_fp_codec BlockFPHeader is ~32B; conservative
    return qint_bytes + exponents_bytes + fixed_header


# ── public API: encode ────────────────────────────────────────────────────


def encode_omega_w_v2(
    weights_block_fp: torch.Tensor | None = None,
    hessian: torch.Tensor | None = None,
    *,
    total_bits: int | None = None,
    variance: torch.Tensor | None = None,
) -> bytes:
    """Encode Lane Ω-W-V2: water-fill + static-arithmetic terminal.

    Args:
        weights_block_fp: Float (O, I, kH, kW) conv weight, eligible for
            block-FP encoding. Required (no silent default per Check 81).
        hessian: Per-output-channel Hessian (O,) tensor. Required.
        total_bits: Total signed-integer bit budget across all channels.
            Required. The water-fill allocates the budget across channels
            via Shannon's reverse-water-filling formula.
        variance: Per-output-channel variance (O,) tensor. If None, computed
            from the weights themselves (unbiased var across each channel).
            Explicit None default is acceptable: the variance estimate is
            a deterministic function of the weights, so the implicit
            derivation is NOT a silent override of a profile value.

    Returns:
        bytes — the OWV2 self-describing payload.

    Raises:
        BlockFPIneligible: input fails block-FP shape/finiteness checks.
        GateRegression: V2 encoded size >= V1 raw qint estimate (no
            silently larger output ships).
        WaterFillError: water-filling itself fails (infeasible budget,
            non-finite Hessian/variance).
    """
    # ── 1. Eligibility gate (block-FP shape + finiteness checks) ─────────
    _check_block_fp_eligible(weights_block_fp)
    o, i, kh, kw = weights_block_fp.shape
    _hessian_eligible(hessian, o)
    if total_bits is None:
        raise GateRegression(
            "encode_omega_w_v2: total_bits is required (no silent default — "
            "Check 81 STRICT). Pass an explicit byte budget × 8 from your "
            "profile or compress-time configuration."
        )
    if total_bits <= 0:
        raise GateRegression(
            f"encode_omega_w_v2: total_bits={total_bits} must be > 0."
        )

    # Variance: derive from weights if not supplied (deterministic from input).
    if variance is None:
        wflat = weights_block_fp.detach().to(torch.float32).reshape(o, -1)
        variance = wflat.var(dim=1, unbiased=False).cpu()
    else:
        if not torch.is_tensor(variance) or variance.dim() != 1 or int(variance.shape[0]) != o:
            raise GateRegression(
                f"encode_omega_w_v2: variance shape {tuple(variance.shape) if torch.is_tensor(variance) else variance!r} "
                f"!= ({o},)."
            )
        if not torch.isfinite(variance).all() or (variance < 0).any():
            raise GateRegression(
                "encode_omega_w_v2: variance must be finite and non-negative."
            )

    # ── 2. Water-fill bit allocation (V1 reuse) ──────────────────────────
    pseudo_layer_name = "owv2.weight"  # internal name — never written to disk
    counts_per_channel = [int(i * kh * kw)] * int(o)
    qint_assignment = water_fill_bit_budget(
        {pseudo_layer_name: hessian.detach().to(torch.float32).cpu()},
        {pseudo_layer_name: variance.detach().to(torch.float32).cpu()},
        {pseudo_layer_name: counts_per_channel},
        int(total_bits),
    )
    qint_per_channel: list[int] = qint_assignment[pseudo_layer_name]
    if len(qint_per_channel) != o:
        raise WaterFillError(
            f"encode_omega_w_v2: water_fill returned {len(qint_per_channel)} "
            f"qints, expected {o}"
        )

    # ── 3. Per-channel quantization to allocated ladder ──────────────────
    flat_qints, exponents, qmax_used = _quantize_to_per_channel_qint(
        weights_block_fp, qint_per_channel
    )

    # ── 4. Static-histogram arithmetic terminal ──────────────────────────
    # The arithmetic coder needs a single alphabet for the whole stream.
    # We use the WIDEST per-channel Q (=> alphabet = 2*Q_max + 1) and
    # offset = Q_max so all symbols fit in [0, 2*Q_max].
    Q_max_global: int = max(qmax_used)
    num_symbols = 2 * Q_max_global + 1
    offset = Q_max_global

    # Sanity-check: every qint actually fits in the global alphabet.
    if flat_qints.size > 0:
        actual_min = int(flat_qints.min())
        actual_max = int(flat_qints.max())
        if actual_min < -Q_max_global or actual_max > Q_max_global:
            raise GateRegression(
                f"encode_omega_w_v2: per-channel quantization produced "
                f"qint outside [-{Q_max_global}, +{Q_max_global}] — "
                f"actual range [{actual_min}, {actual_max}]. Bug in "
                f"_quantize_to_per_channel_qint."
            )

    arith_payload = encode_qints_arithmetic(
        flat_qints.astype(np.int32),
        num_symbols=int(num_symbols),
        offset=int(offset),
    )

    # ── 5. OWV2 header ───────────────────────────────────────────────────
    # Layout (little-endian):
    #     magic              : 4 bytes  = b"OWV2"
    #     version            : 2 bytes  uint16
    #     arith_table_version: 2 bytes  uint16
    #     o, i, kh, kw       : 4 * 4 bytes int32  (original OIHW shape)
    #     q_max_global       : 4 bytes  int32     (alphabet anchor)
    #     n_channels         : 4 bytes  int32     (= o; redundancy guard)
    #     qint_per_channel   : n_channels * 1 byte uint8 (each Q_c, fits in [1, 31])
    #     exponents          : n_channels * 4 bytes int32
    #     payload_size       : 8 bytes  uint64
    #     payload            : payload_size bytes (AQv1 arithmetic blob)
    header = io.BytesIO()
    header.write(OWV2_MAGIC)
    header.write(struct.pack("<H", OWV2_VERSION))
    header.write(struct.pack("<H", OWV2_ARITH_TABLE_VERSION))
    header.write(struct.pack("<iiii", int(o), int(i), int(kh), int(kw)))
    header.write(struct.pack("<i", int(Q_max_global)))
    header.write(struct.pack("<i", int(o)))
    qmax_bytes = bytes(int(q) & 0xFF for q in qmax_used)
    header.write(qmax_bytes)
    header.write(exponents.astype("<i4").tobytes())
    header.write(struct.pack("<Q", len(arith_payload)))

    encoded = header.getvalue() + arith_payload

    # ── HARD OVERHEAD GATE (Carmack rule from memory) ────────────────────
    v1_estimate = _v1_raw_qint_byte_estimate(weights_block_fp, qint_per_channel)
    if len(encoded) >= v1_estimate:
        raise GateRegression(
            f"encode_omega_w_v2: OWV2 encoded {len(encoded)} B >= V1 raw "
            f"{v1_estimate} B for the same tensor + allocation. The "
            f"arithmetic coder did not amortize its header on this stream. "
            f"Per memory project_codec_stacking_composition canonical orders: "
            f"keep V1 instead of shipping a regression. (Tensor shape "
            f"{tuple(weights_block_fp.shape)}, total_bits={total_bits}.)"
        )
    return encoded


# ── public API: decode ────────────────────────────────────────────────────


@dataclass
class _OWV2Header:
    """Parsed OWV2 header fields."""

    version: int
    arith_table_version: int
    o: int
    i: int
    kh: int
    kw: int
    q_max_global: int
    qmax_per_channel: list[int]
    exponents: np.ndarray  # int32 (O,)
    payload_size: int
    header_byte_len: int


def _parse_owv2_header(blob: bytes) -> _OWV2Header:
    """Strict header parser; raises ValueError on malformed input."""
    if len(blob) < 4 + 2 + 2 + 16 + 4 + 4:
        raise ValueError(
            f"decode_omega_w_v2: blob length {len(blob)} too small for OWV2 header"
        )
    if blob[:4] != OWV2_MAGIC:
        raise ValueError(
            f"decode_omega_w_v2: bad magic {blob[:4]!r}, expected {OWV2_MAGIC!r}"
        )
    buf = io.BytesIO(blob)
    buf.read(4)  # magic consumed
    (version,) = struct.unpack("<H", buf.read(2))
    if version != OWV2_VERSION:
        raise ValueError(
            f"decode_omega_w_v2: unsupported version {version}; expected {OWV2_VERSION}"
        )
    (arith_v,) = struct.unpack("<H", buf.read(2))
    o, i, kh, kw = struct.unpack("<iiii", buf.read(16))
    (q_max_global,) = struct.unpack("<i", buf.read(4))
    (n_channels,) = struct.unpack("<i", buf.read(4))
    if n_channels != o:
        raise ValueError(
            f"decode_omega_w_v2: header n_channels={n_channels} != O={o}"
        )
    qmax_bytes = buf.read(int(o))
    if len(qmax_bytes) != int(o):
        raise ValueError(
            f"decode_omega_w_v2: truncated qmax_per_channel "
            f"(read {len(qmax_bytes)} of {o})"
        )
    qmax_per_channel = [int(b) for b in qmax_bytes]
    exp_bytes = buf.read(int(o) * 4)
    if len(exp_bytes) != int(o) * 4:
        raise ValueError(
            f"decode_omega_w_v2: truncated exponents "
            f"(read {len(exp_bytes)} of {o*4})"
        )
    exponents = np.frombuffer(exp_bytes, dtype="<i4").copy()
    (payload_size,) = struct.unpack("<Q", buf.read(8))
    header_byte_len = buf.tell()
    return _OWV2Header(
        version=version,
        arith_table_version=arith_v,
        o=int(o),
        i=int(i),
        kh=int(kh),
        kw=int(kw),
        q_max_global=int(q_max_global),
        qmax_per_channel=qmax_per_channel,
        exponents=exponents,
        payload_size=int(payload_size),
        header_byte_len=int(header_byte_len),
    )


def decode_omega_w_v2(blob: bytes | None = None) -> torch.Tensor:
    """Decode an OWV2 payload back to a float32 (O, I, kH, kW) weight tensor.

    Pure-math byte → tensor. NO scorer load (Check H STRICT). NO GPU needed.

    Args:
        blob: bytes produced by ``encode_omega_w_v2``. Required.

    Returns:
        torch.Tensor float32 (O, I, kH, kW), CPU.
    """
    if blob is None:
        raise ValueError(
            "decode_omega_w_v2: blob is required (no silent default — "
            "Check 81 STRICT)."
        )
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError(
            f"decode_omega_w_v2: blob must be bytes-like, got {type(blob).__name__}"
        )

    header = _parse_owv2_header(bytes(blob))
    payload_start = header.header_byte_len
    payload_end = payload_start + header.payload_size
    if payload_end > len(blob):
        raise ValueError(
            f"decode_omega_w_v2: declared payload {header.payload_size}B but "
            f"blob has only {len(blob) - payload_start}B remaining."
        )
    arith_payload = bytes(blob[payload_start:payload_end])

    n_elements = header.o * header.i * header.kh * header.kw
    qint_flat = decode_qints_arithmetic(arith_payload, expected_dtype=np.int32)
    if qint_flat.size != n_elements:
        raise ValueError(
            f"decode_omega_w_v2: arithmetic decode produced {qint_flat.size} "
            f"symbols, expected {n_elements} for shape "
            f"({header.o}, {header.i}, {header.kh}, {header.kw})"
        )

    qint_oihw = torch.from_numpy(
        qint_flat.reshape(header.o, header.i, header.kh, header.kw).astype(np.int32)
    ).to(torch.float32)
    exponents_t = torch.from_numpy(header.exponents.astype(np.int32)).to(torch.float32)
    scales = torch.pow(2.0, exponents_t).view(-1, 1, 1, 1)
    out = qint_oihw * scales
    return out.contiguous()


# ── compatibility helper for compress_archive wiring ──────────────────────


def is_owv2_blob(blob: bytes) -> bool:
    """Magic-byte sniff: True iff the blob starts with OWV2 magic.

    Used by renderer load paths to dispatch between OWV2 and legacy
    (block-FP / ASYM / FP4A / etc.) decoders.
    """
    return isinstance(blob, (bytes, bytearray, memoryview)) and bytes(blob[:4]) == OWV2_MAGIC


__all__ = [
    "OWV2_MAGIC",
    "OWV2_VERSION",
    "OWV2_ARITH_TABLE_VERSION",
    "GateRegression",
    "BlockFPIneligible",
    "encode_omega_w_v2",
    "decode_omega_w_v2",
    "is_owv2_blob",
]
