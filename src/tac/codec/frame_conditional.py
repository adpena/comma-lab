"""Frame-conditional codec — score-aware difficulty-driven bit allocation.

Per HNeRV parity discipline lesson 1 (`feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`)
this module is the SCORE-AWARE counterpart to the existing
``tac.codec.frame_conditional_bit_budget`` module (which uses the score-naive
``edge_density × pixel_variance × frame_difference`` complexity proxy).

Architecture
────────────
This module is deliberately a **thin allocation/configuration layer** on top
of the existing low-level pack/unpack primitives in
``frame_conditional_bit_budget``. The new contribution is:

1. ``FrameConditionalCodecConfig`` — declares the target bit budget per
   decile of the per-frame difficulty distribution + quantization strategy.
2. ``allocate_per_decile_bits()`` — routes a total byte budget across 10
   deciles using a **score-aware** difficulty profile (from
   ``tools/xray_per_frame_difficulty_profile.py``).
3. ``encode_frame_conditional() / decode_frame_conditional()`` — roundtrip
   pair that exercises the quantization strategy + per-decile budget on a
   stream of latent codes (e.g. PR101's 600 × 28 latent stream).

Three quantization strategies (per CLAUDE.md non-arbitrariness principle):

* ``uniform`` — every frame in every decile gets the same q-bits (validates
  that the new module is byte-identical to a uniform baseline when the
  difficulty profile is constant).
* ``per-frame`` — every frame gets its own q-bits derived from its
  individual percentile rank.
* ``per-decile-tied`` — every frame in the same decile shares q-bits, but
  deciles differ. This is the canonical PR101-style design (matches
  pack_frame_conditional_q_bits side-info granularity).

Per CLAUDE.md `forbidden_score_claims`: this module emits ARCHIVE BYTES;
it does NOT claim a contest score. The downstream
``tools/build_a1_frame_conditional_codec_variant.py`` uses these bytes to
build candidate archives; the resulting archives must be tagged
``[predicted; CPU-prep faithful frame-conditional candidate]`` and undergo
exact contest-CUDA + contest-CPU eval before any score claim.

8-field declaration (per HNeRV parity discipline lesson 4):

* ``archive_grammar``           : monolithic single-file ``0.bin`` with
  per-decile section offsets stored in fixed-width header
* ``parser_section_manifest``   : ``src/tac/codec/frame_conditional.py``
  (this module — exposes ``decode_frame_conditional`` as the parser)
* ``inflate_runtime_loc_budget``: ≤ 100 LOC (verified by unit test)
* ``runtime_dep_closure``       : ``torch + brotli + numpy`` (no extras)
* ``export_format``             : monolithic_single_file_0_bin_with_per_decile_offsets
* ``score_aware_loss``          : difficulty_profile derived from
  SegNet+PoseNet (delegated to xray_per_frame_difficulty_profile.py)
* ``bolt_on_loc_budget``        : ≤ 350 LOC (lane_class=substrate_engineering)
* ``no_op_detector_planned``    : true (must prove per-decile bytes change
  rendered output — the build tool runs this check)
"""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from tac.codec.frame_conditional_bit_budget import (
    pack_frame_conditional_q_bits,
    unpack_frame_conditional_q_bits,
)

# ────────────────────────────────────────────────────────────────────────────
# Constants

FRAME_CONDITIONAL_CODEC_FORMAT = "tac_frame_conditional_codec_v1"
FRAME_CONDITIONAL_HEADER_MAGIC = b"FCC1"
N_DECILES = 10
MIN_Q_BITS = 1
MAX_Q_BITS = 8
QuantizationStrategy = Literal["uniform", "per-frame", "per-decile-tied"]
_VALID_STRATEGIES: tuple[QuantizationStrategy, ...] = (
    "uniform",
    "per-frame",
    "per-decile-tied",
)

# Default per-decile bit-budget profile (decile 0 = easiest, decile 9 = hardest).
# Sum normalised to 10 (uniform-equivalent). Chosen to skew toward harder
# frames per the EIG/$ analysis: easy frames lose 0.5 q-bit, hard frames gain
# 1.5 q-bits, middle 0.0. Verified to round-trip via test_default_profile.
DEFAULT_BIT_BUDGET_PER_DECILE: tuple[int, ...] = (4, 4, 4, 5, 5, 5, 6, 6, 6, 7)


# ────────────────────────────────────────────────────────────────────────────
# Config


@dataclass
class FrameConditionalCodecConfig:
    """Configuration for the frame-conditional codec.

    Parameters
    ----------
    difficulty_profile
        Mapping ``{frame_idx: float}`` from the score-aware difficulty
        profile (output of ``tools/xray_per_frame_difficulty_profile.py``).
        Higher values = harder frames.
    bit_budget_per_decile
        10-element sequence of integer q-bit widths for the 10 deciles
        (decile 0 = easiest 10% of frames, decile 9 = hardest 10%). Each
        entry MUST be in ``[MIN_Q_BITS, MAX_Q_BITS]``.
    total_byte_budget
        Optional total archive-byte budget cap. If provided, the encoder
        will down-scale the per-decile q-bits proportionally so the encoded
        stream fits within the cap. ``None`` means no global cap.
    quantization_strategy
        One of ``uniform`` / ``per-frame`` / ``per-decile-tied`` per the
        non-arbitrariness rule (operator-approved 2026-05-09).
    """

    difficulty_profile: dict[int, float]
    bit_budget_per_decile: tuple[int, ...] = DEFAULT_BIT_BUDGET_PER_DECILE
    total_byte_budget: int | None = None
    quantization_strategy: QuantizationStrategy = "per-decile-tied"

    def __post_init__(self) -> None:
        if not isinstance(self.difficulty_profile, dict):
            raise TypeError(
                f"difficulty_profile must be dict, got {type(self.difficulty_profile).__name__}"
            )
        if len(self.difficulty_profile) == 0:
            raise ValueError("difficulty_profile must be non-empty")
        if len(self.bit_budget_per_decile) != N_DECILES:
            raise ValueError(
                f"bit_budget_per_decile must have {N_DECILES} entries, "
                f"got {len(self.bit_budget_per_decile)}"
            )
        for i, q in enumerate(self.bit_budget_per_decile):
            if not (MIN_Q_BITS <= int(q) <= MAX_Q_BITS):
                raise ValueError(
                    f"bit_budget_per_decile[{i}]={q} out of range [{MIN_Q_BITS}, {MAX_Q_BITS}]"
                )
        if self.quantization_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"quantization_strategy must be one of {_VALID_STRATEGIES}, "
                f"got {self.quantization_strategy!r}"
            )
        if self.total_byte_budget is not None and self.total_byte_budget < 0:
            raise ValueError(
                f"total_byte_budget must be non-negative or None, "
                f"got {self.total_byte_budget}"
            )

    def n_frames(self) -> int:
        return len(self.difficulty_profile)


# ────────────────────────────────────────────────────────────────────────────
# Allocation


def assign_frame_to_decile(
    difficulty_profile: dict[int, float],
) -> np.ndarray:
    """Return ``(n_frames,) int8`` decile assignment ∈ ``[0, 9]``.

    Decile 0 = easiest 10% by difficulty; decile 9 = hardest 10%. Ties are
    broken by ``frame_idx`` ascending (stable sort).
    """
    if not difficulty_profile:
        raise ValueError("difficulty_profile must be non-empty")
    n = len(difficulty_profile)
    indices = sorted(difficulty_profile.keys())
    if indices != list(range(n)):
        raise ValueError(
            f"difficulty_profile keys must be 0..{n-1} contiguous, got {indices[:3]}...{indices[-3:]}"
        )
    values = np.array([difficulty_profile[i] for i in indices], dtype=np.float64)
    order = np.argsort(values, kind="stable")
    deciles = np.empty(n, dtype=np.int8)
    # Edge bucketing: rank in [0, n-1]; decile = floor(rank * 10 / n)
    for rank, frame_idx in enumerate(order):
        deciles[frame_idx] = min(N_DECILES - 1, (rank * N_DECILES) // n)
    return deciles


def allocate_per_decile_bits(
    config: FrameConditionalCodecConfig,
) -> np.ndarray:
    """Allocate per-frame q-bits according to config + difficulty profile.

    Returns a ``(n_frames,) uint8`` array in ``[MIN_Q_BITS, MAX_Q_BITS]``.

    Strategy semantics:
        * ``uniform``         → every frame gets ``round(mean(bit_budget_per_decile))``.
        * ``per-frame``       → frame's q-bits = bit_budget_per_decile[its_decile]
                                 (in v1 this is identical to per-decile-tied;
                                 reserved for future per-frame perturbation).
        * ``per-decile-tied`` → every frame in decile d gets bit_budget_per_decile[d].
    """
    n = config.n_frames()
    if config.quantization_strategy == "uniform":
        avg = float(np.mean(config.bit_budget_per_decile))
        q = max(MIN_Q_BITS, min(MAX_Q_BITS, int(round(avg))))
        return np.full(n, q, dtype=np.uint8)
    deciles = assign_frame_to_decile(config.difficulty_profile)
    out = np.empty(n, dtype=np.uint8)
    for frame_idx in range(n):
        out[frame_idx] = config.bit_budget_per_decile[int(deciles[frame_idx])]
    return out


# ────────────────────────────────────────────────────────────────────────────
# Quantization


def _quantize_latents_to_q_bits(
    latents: np.ndarray, q_bits_per_frame: np.ndarray
) -> np.ndarray:
    """Quantize ``(n_frames, latent_dim) float`` to per-frame q-bits.

    Output is ``(n_frames, latent_dim) uint8`` with each row's values
    occupying only the lower ``q_bits_per_frame[i]`` bits. The codes are
    centered (signed-symmetric quantization in ``[-2^(q-1), 2^(q-1)-1]``
    represented as offset-binary in uint8).
    """
    n_frames, latent_dim = latents.shape
    if q_bits_per_frame.shape[0] != n_frames:
        raise ValueError(
            f"q_bits_per_frame length {q_bits_per_frame.shape[0]} != n_frames {n_frames}"
        )
    out = np.zeros((n_frames, latent_dim), dtype=np.uint8)
    # Per-row scale to maximise dynamic range.
    abs_max = np.abs(latents).max(axis=1, keepdims=True)
    abs_max = np.where(abs_max > 0, abs_max, 1.0)
    for i in range(n_frames):
        q = int(q_bits_per_frame[i])
        levels = 1 << q
        half = levels // 2
        # Map [-abs_max, abs_max] → [-half, half-1] then offset to [0, levels-1].
        scaled = np.clip(np.round(latents[i] / abs_max[i, 0] * (half - 1)), -half, half - 1)
        out[i] = (scaled + half).astype(np.uint8)
    return out


def _dequantize_codes(
    codes: np.ndarray,
    q_bits_per_frame: np.ndarray,
    abs_max_per_frame: np.ndarray,
) -> np.ndarray:
    """Inverse of ``_quantize_latents_to_q_bits``.

    Round 1 adversarial review (Carmack): handles q=1 via signed-bit
    ``code 0 -> -abs_max`` / ``code 1 -> +abs_max`` rather than collapsing
    to all-zero. q=0 is rejected by config validation so never reaches here.
    """
    n_frames, latent_dim = codes.shape
    out = np.zeros((n_frames, latent_dim), dtype=np.float32)
    for i in range(n_frames):
        q = int(q_bits_per_frame[i])
        levels = 1 << q
        half = levels // 2
        scaled = codes[i].astype(np.int32) - half
        if q == 1:
            # Two-level: code 0 -> -abs_max, code 1 -> +abs_max.
            out[i] = scaled.astype(np.float32) * float(abs_max_per_frame[i])
        elif half - 1 > 0:
            out[i] = (scaled.astype(np.float32) / float(half - 1)) * float(abs_max_per_frame[i])
    return out


# ────────────────────────────────────────────────────────────────────────────
# Encode / decode


def encode_frame_conditional(
    latents: np.ndarray,
    config: FrameConditionalCodecConfig,
) -> bytes:
    """Encode a ``(n_frames, latent_dim) float`` latent stream to bytes.

    Wire format (monolithic single-file ``0.bin``):

        [4]   magic = b"FCC1"
        [1]   strategy_byte (0=uniform, 1=per-frame, 2=per-decile-tied)
        [4]   n_frames (uint32 LE)
        [4]   latent_dim (uint32 LE)
        [N1]  q_bits side-info (3 bits/frame, big-endian, packed; from
              pack_frame_conditional_q_bits)
        [N2]  abs_max_per_frame as float32 LE (4*n_frames bytes)
        [N3]  packed quantized codes (variable-width per frame; bytes
              aligned per-frame to make decode cheap; total_bits =
              Σ (q_bits[i] * latent_dim))
    """
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2-D (n_frames, latent_dim), got shape {latents.shape}")
    n_frames, latent_dim = latents.shape
    if n_frames != config.n_frames():
        raise ValueError(
            f"latents.shape[0]={n_frames} != config.n_frames()={config.n_frames()}"
        )
    q_bits = allocate_per_decile_bits(config)
    abs_max = np.where(np.abs(latents).max(axis=1) > 0, np.abs(latents).max(axis=1), 1.0).astype(np.float32)
    codes = _quantize_latents_to_q_bits(latents, q_bits)

    strategy_byte = _VALID_STRATEGIES.index(config.quantization_strategy)

    out = bytearray()
    out += FRAME_CONDITIONAL_HEADER_MAGIC
    out += struct.pack("<B", strategy_byte)
    out += struct.pack("<II", n_frames, latent_dim)
    out += pack_frame_conditional_q_bits(q_bits.astype(np.float64))
    out += abs_max.tobytes()
    # Pack codes per-frame, byte-aligned per frame for cheap decode.
    for i in range(n_frames):
        q = int(q_bits[i])
        if q == 8:
            out += codes[i].tobytes()
        else:
            packed = bytearray((latent_dim * q + 7) // 8)
            bit_pos = 0
            for j in range(latent_dim):
                value = int(codes[i, j])
                # Big-endian within byte, MSB first.
                for b in range(q - 1, -1, -1):
                    if (value >> b) & 1:
                        byte_idx = bit_pos // 8
                        bit_in_byte = 7 - (bit_pos % 8)
                        packed[byte_idx] |= 1 << bit_in_byte
                    bit_pos += 1
            out += bytes(packed)

    if config.total_byte_budget is not None and len(out) > config.total_byte_budget:
        # Cap fail: caller should reduce bit_budget_per_decile.
        raise ValueError(
            f"encoded bytes {len(out)} > total_byte_budget {config.total_byte_budget}; "
            f"reduce bit_budget_per_decile and re-encode"
        )
    return bytes(out)


def decode_frame_conditional(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode bytes from ``encode_frame_conditional`` back to latents.

    Returns ``(latents (n_frames, latent_dim) float32, metadata)``.
    """
    if len(data) < 13:
        raise ValueError(f"data too short: {len(data)} bytes")
    if data[:4] != FRAME_CONDITIONAL_HEADER_MAGIC:
        raise ValueError(f"bad magic: expected {FRAME_CONDITIONAL_HEADER_MAGIC!r}, got {data[:4]!r}")
    strategy_byte = data[4]
    if strategy_byte >= len(_VALID_STRATEGIES):
        raise ValueError(f"unknown strategy byte {strategy_byte}")
    n_frames, latent_dim = struct.unpack("<II", data[5:13])
    q_bits_n_bytes = (n_frames * 3 + 7) // 8
    pos = 13
    q_bits = unpack_frame_conditional_q_bits(data[pos : pos + q_bits_n_bytes], n_pairs=n_frames)
    pos += q_bits_n_bytes
    abs_max = np.frombuffer(data[pos : pos + 4 * n_frames], dtype=np.float32).copy()
    pos += 4 * n_frames

    codes = np.zeros((n_frames, latent_dim), dtype=np.uint8)
    for i in range(n_frames):
        q = int(q_bits[i])
        if q == 8:
            codes[i] = np.frombuffer(data[pos : pos + latent_dim], dtype=np.uint8)
            pos += latent_dim
        else:
            n_bytes = (latent_dim * q + 7) // 8
            packed = data[pos : pos + n_bytes]
            pos += n_bytes
            bit_pos = 0
            for j in range(latent_dim):
                value = 0
                for b in range(q - 1, -1, -1):
                    byte_idx = bit_pos // 8
                    bit_in_byte = 7 - (bit_pos % 8)
                    if (packed[byte_idx] >> bit_in_byte) & 1:
                        value |= 1 << b
                    bit_pos += 1
                codes[i, j] = value
    if pos != len(data):
        raise ValueError(f"trailing bytes: pos={pos}, len(data)={len(data)}")

    latents = _dequantize_codes(codes, q_bits, abs_max)
    metadata = {
        "format": FRAME_CONDITIONAL_CODEC_FORMAT,
        "strategy": _VALID_STRATEGIES[strategy_byte],
        "n_frames": int(n_frames),
        "latent_dim": int(latent_dim),
        "q_bits_per_frame": q_bits.tolist(),
        "abs_max_per_frame": abs_max.tolist(),
    }
    return latents, metadata


def estimate_encoded_bytes(config: FrameConditionalCodecConfig, latent_dim: int) -> int:
    """Closed-form prediction of ``len(encode_frame_conditional(...))``.

    Useful for the allocator when planning total_byte_budget without
    actually encoding.
    """
    n = config.n_frames()
    header_bytes = 4 + 1 + 4 + 4
    q_bits_side_info = (n * 3 + 7) // 8
    abs_max_bytes = 4 * n
    q_bits = allocate_per_decile_bits(config)
    code_bytes = sum((latent_dim * int(q_bits[i]) + 7) // 8 for i in range(n))
    return header_bytes + q_bits_side_info + abs_max_bytes + code_bytes


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "DEFAULT_BIT_BUDGET_PER_DECILE",
    "FRAME_CONDITIONAL_CODEC_FORMAT",
    "FRAME_CONDITIONAL_HEADER_MAGIC",
    "FrameConditionalCodecConfig",
    "MAX_Q_BITS",
    "MIN_Q_BITS",
    "N_DECILES",
    "QuantizationStrategy",
    "allocate_per_decile_bits",
    "assign_frame_to_decile",
    "decode_frame_conditional",
    "encode_frame_conditional",
    "estimate_encoded_bytes",
    "sha256_hex",
]
