# SPDX-License-Identifier: MIT
"""Scorer-conditional entropy coder (range-coder variant).

Mathematical formulation
------------------------

Given a symbol stream ``x_1, x_2, ..., x_N`` over alphabet ``Σ`` of size ``K``
and an auxiliary side-information stream ``y_1, y_2, ..., y_N`` (scorer-derived
state per symbol, see module docstring), the JSCC encoder uses the conditional
probability mass function

    p(x_t | y_t)                                                            (1)

to drive a range coder. The decoder reconstructs ``y_t`` from already-decoded
symbols and a deterministic side-state model, so the side-state is NOT carried
in the archive bytes (only the symbol stream + the model parameters).

Expected coded length per symbol:

    E[L] = -E[ log2 p(x | y) ] = H(X | Y) ≤ H(X)                            (2)

The savings vs unconditional arithmetic coding equal the mutual information
``I(X; Y)``. Per the source memo (SE-4) prediction: ~10% rate-term savings
on HNeRV-family substrates where scorer-state correlates with latent symbol
distribution.

Range coder implementation
--------------------------

This is a precision-controlled integer range coder (a.k.a. arithmetic coder
with renormalization), not the floating-point arithmetic-coding implementation
in ``tac.packet_compiler.pr103_arithmetic_coding``. The 32-bit-precision form
matches the canonical Range-Coder pattern of Subbotin / Schindler.

The conditional density ``p(x | y)`` is computed by a small MLP at encode time;
at decode time, the SAME MLP (same weights, same forward pass) is recomputed
on the reconstructed side-state. Determinism is critical: the MLP runs in FP32
on the target device with ``torch.no_grad()`` and rounded-to-int frequency
tables to avoid floating-point divergence between encoder and decoder.

Cross-references
----------------
- Source memo SE-4: ``.omx/research/ancient_elder_polymath_research_20260513.md``
- Range-coder lineage: Martin 1979, Subbotin 1999, Schindler 1998.
- Sister unconditional arithmetic coder:
  ``tac.packet_compiler.pr103_arithmetic_coding``.

Lane: ``lane_implement_iglt_ternary_jscc_kc3_canonical_20260513``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = [
    "JSCC_FORMAT_VERSION",
    "JSCC_MAGIC",
    "PRECISION_BITS",
    "SE4R_MAGIC",
    "TOTAL_FREQ",
    "ScorerConditionalEntropyCoder",
    "ScorerConditionalProbabilityModel",
    "decode_jscc_stream",
    "encode_jscc_stream",
    "validate_frequency_table",
]


SE4R_MAGIC: bytes = b"SE4R"
"""Magic bytes for the SE-4 scorer-conditional range-coder section.

Distinct from the legacy Huffman variant's ``b"JSCC"`` magic
(``tac.codec.jscc.conditional_huffman._MAGIC``). Both coders live in the
same JSCC package but emit different magic so a stream-level parser can
distinguish them.
"""

JSCC_MAGIC: bytes = SE4R_MAGIC
"""Backward-compatible alias for ``SE4R_MAGIC``.

The package is named ``tac.codec.jscc`` because both scorer-conditional
coders share the JSCC lane, but this range-coder stream is specifically the
SE4R binary grammar. New code should prefer ``SE4R_MAGIC`` when the distinction
from the legacy Huffman ``b"JSCC"`` packet matters.
"""

JSCC_FORMAT_VERSION: int = 2
"""Format-version byte that follows the magic. Bumped on grammar changes."""

PRECISION_BITS: int = 16
"""Precision of frequency-table totals. Range coder uses 32-bit internal range."""

TOTAL_FREQ: int = 1 << PRECISION_BITS
"""Maximum cumulative frequency. Symbols share this budget."""


# ── ScorerConditionalProbabilityModel: small MLP density predictor ──────


class ScorerConditionalProbabilityModel(nn.Module):
    """Small MLP mapping ``side_state -> p(x | side_state)``.

    Architecture: side_dim -> hidden_dim -> alphabet_size; softmax output.
    Default hidden_dim=32 gives ~(side_dim*32 + 32*alphabet_size) params; for
    side_dim=8 and alphabet_size=256 this is ~256 + 8192 = ~8.4K params.

    Per the source memo, this model is INCLUDED in the archive bytes — the
    decoder needs identical weights to reconstruct the probability table. The
    MLP weights are quantized to int8 (~8.4 KB → ~8.4 KB after fake_quantize).
    The training-time MLP is FP32; export round-trips through int8.

    Args:
        side_dim: dimensionality of the side-state vector.
        alphabet_size: size of the symbol alphabet ``K``.
        hidden_dim: hidden-layer width. Default 32.

    Example:
        >>> model = ScorerConditionalProbabilityModel(
        ...     side_dim=8, alphabet_size=256, hidden_dim=32)
        >>> side = torch.randn(1, 8)
        >>> probs = model(side)
        >>> assert probs.shape == (1, 256)
        >>> assert torch.allclose(probs.sum(dim=-1), torch.tensor(1.0))
    """

    def __init__(
        self,
        side_dim: int,
        alphabet_size: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if side_dim <= 0:
            raise ValueError(f"side_dim must be positive, got {side_dim}")
        if alphabet_size < 2:
            raise ValueError(
                f"alphabet_size must be >= 2, got {alphabet_size}"
            )
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {hidden_dim}"
            )
        self.side_dim = int(side_dim)
        self.alphabet_size = int(alphabet_size)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(side_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, alphabet_size)

    def forward(self, side_state: torch.Tensor) -> torch.Tensor:
        """Return probability distribution per row of side_state.

        Args:
            side_state: shape ``(batch, side_dim)`` (or ``(side_dim,)``).

        Returns:
            Tensor of shape ``(batch, alphabet_size)`` with softmax-normalized
            probabilities along the last axis.
        """
        if side_state.ndim == 1:
            side_state = side_state.unsqueeze(0)
        h = torch.relu(self.fc1(side_state))
        logits = self.fc2(h)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def integer_frequency_table(
        self, side_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute the integer frequency table for a row of side_state.

        Returns a 1-D ``int64`` tensor of length ``alphabet_size`` whose
        entries sum to exactly ``TOTAL_FREQ``. The table is rounded from the
        floating-point softmax distribution with a deterministic
        residual-distribution scheme to ensure the sum equals ``TOTAL_FREQ``
        exactly. This is critical for encoder-decoder bit-exactness.

        Args:
            side_state: shape ``(side_dim,)`` for a single time step.

        Returns:
            ``int64`` tensor of shape ``(alphabet_size,)``.
        """
        probs = self.forward(side_state.detach()).squeeze(0)
        return _probs_to_integer_frequencies(probs, TOTAL_FREQ)


def _probs_to_integer_frequencies(
    probs: torch.Tensor, total_freq: int
) -> torch.Tensor:
    """Convert a floating-point distribution to an integer freq table.

    The output has integer entries that sum EXACTLY to ``total_freq``. Each
    entry is at least 1 (no zero-probability symbols — arithmetic coders
    cannot encode a zero-probability symbol). The residual rounding error is
    distributed to entries with the largest fractional remainder.

    Args:
        probs: 1-D FP tensor (any device); softmax-normalized.
        total_freq: target sum.

    Returns:
        1-D int64 tensor on the same device as ``probs``.
    """
    K = int(probs.numel())
    if total_freq < K:
        raise ValueError(
            f"total_freq={total_freq} must be >= alphabet_size={K} "
            f"(every symbol needs at least 1 count)"
        )
    # Move to CPU for deterministic rounding
    p = probs.detach().to(torch.float64).cpu()
    scaled = p * float(total_freq - K)  # reserve 1 for each symbol
    floor = scaled.floor().to(torch.int64)
    remainder = scaled - floor.to(torch.float64)
    # Distribute the difference (total_freq - K - floor.sum()) to the
    # symbols with the largest remainder.
    deficit = int(total_freq - K - floor.sum().item())
    if deficit > 0:
        # Top-k indices by remainder
        _, top_idx = torch.topk(remainder, k=deficit)
        floor[top_idx] += 1
    elif deficit < 0:
        # Should be impossible under floor() but guard anyway
        _, bot_idx = torch.topk(-remainder, k=-deficit)
        floor[bot_idx] -= 1
    # Add 1 to each symbol (the reserved budget)
    freqs = floor + 1
    # Sanity check
    if int(freqs.sum().item()) != total_freq:
        raise RuntimeError(
            f"frequency-table sum {int(freqs.sum().item())} != "
            f"expected {total_freq} (rounding logic bug)"
        )
    return freqs


def validate_frequency_table(freqs: torch.Tensor) -> None:
    """Check a frequency table is well-formed for the range coder.

    Raises:
        ValueError: if any entry < 1, sum != TOTAL_FREQ, or wrong shape/dtype.
    """
    if freqs.ndim != 1:
        raise ValueError(
            f"freqs must be 1-D, got shape {tuple(freqs.shape)}"
        )
    if freqs.dtype != torch.int64:
        raise ValueError(f"freqs must be int64, got {freqs.dtype}")
    if int(freqs.min().item()) < 1:
        raise ValueError(
            "every frequency entry must be >= 1 (no zero-probability symbols)"
        )
    if int(freqs.sum().item()) != TOTAL_FREQ:
        raise ValueError(
            f"frequency-table sum {int(freqs.sum().item())} != "
            f"TOTAL_FREQ={TOTAL_FREQ}"
        )


# ── Arithmetic coder (CACM bit-precision form, Witten/Neal/Cleary 1987) ─


# Precision bits for the arithmetic coder state. 32 is the standard form;
# we use 30 here to leave 2-bit headroom against carry-propagation overflow.
_AC_CODE_BITS: int = 32
_AC_TOP_MASK: int = (1 << _AC_CODE_BITS) - 1
_AC_HALF: int = 1 << (_AC_CODE_BITS - 1)
_AC_QUARTER: int = 1 << (_AC_CODE_BITS - 2)
_AC_THREE_Q: int = _AC_HALF + _AC_QUARTER


class _ACBitWriter:
    """Bitwise output stream for the arithmetic coder."""

    def __init__(self) -> None:
        self._byte_buf: list[int] = []
        self._cur_byte: int = 0
        self._cur_bits: int = 0

    def write_bit(self, bit: int) -> None:
        self._cur_byte = (self._cur_byte << 1) | (bit & 1)
        self._cur_bits += 1
        if self._cur_bits == 8:
            self._byte_buf.append(self._cur_byte)
            self._cur_byte = 0
            self._cur_bits = 0

    def write_bit_plus_pending(self, bit: int, pending: int) -> int:
        """Write ``bit`` then ``pending`` copies of the opposite bit.

        Used by the CACM E1/E2/E3 scaling rules. Returns 0 (pending count
        is reset by the caller).
        """
        self.write_bit(bit)
        opp = 1 - (bit & 1)
        for _ in range(pending):
            self.write_bit(opp)
        return 0

    def flush(self) -> bytes:
        if self._cur_bits > 0:
            # Pad the final byte with zeros on the right.
            self._byte_buf.append(self._cur_byte << (8 - self._cur_bits))
        return bytes(self._byte_buf)


class _ACBitReader:
    """Bitwise input stream that returns 0 past end-of-stream."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte_pos: int = 0
        self._bit_pos: int = 0

    def read_bit(self) -> int:
        if self._byte_pos >= len(self._data):
            return 0
        b = self._data[self._byte_pos]
        bit = (b >> (7 - self._bit_pos)) & 1
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._bit_pos = 0
            self._byte_pos += 1
        return bit


class _RangeEncoder:
    """CACM-style bit-level arithmetic encoder (Witten/Neal/Cleary 1987).

    Implements E1, E2, E3 scaling. Encodes one symbol at a time given its
    cumulative low/high frequencies. Bit-exact round-trip with
    ``_RangeDecoder``.
    """

    def __init__(self) -> None:
        self._low: int = 0
        self._high: int = _AC_TOP_MASK
        self._pending_bits: int = 0
        self._writer = _ACBitWriter()

    def encode_symbol(
        self, cum_low: int, cum_high: int, total_freq: int
    ) -> None:
        rng = self._high - self._low + 1
        # Update [low, high] for the symbol's sub-interval
        new_high = self._low + (rng * cum_high) // total_freq - 1
        new_low = self._low + (rng * cum_low) // total_freq
        self._low = new_low
        self._high = new_high
        # Scaling loop: emit MSBs while possible
        while True:
            if self._high < _AC_HALF:
                # E1: emit 0, shift left
                self._pending_bits = self._writer.write_bit_plus_pending(
                    0, self._pending_bits
                )
                self._low = (self._low << 1) & _AC_TOP_MASK
                self._high = ((self._high << 1) | 1) & _AC_TOP_MASK
            elif self._low >= _AC_HALF:
                # E2: emit 1, subtract HALF, shift left
                self._pending_bits = self._writer.write_bit_plus_pending(
                    1, self._pending_bits
                )
                self._low = ((self._low - _AC_HALF) << 1) & _AC_TOP_MASK
                self._high = (
                    ((self._high - _AC_HALF) << 1) | 1
                ) & _AC_TOP_MASK
            elif self._low >= _AC_QUARTER and self._high < _AC_THREE_Q:
                # E3: middle straddle, increment pending, subtract QUARTER
                self._pending_bits += 1
                self._low = ((self._low - _AC_QUARTER) << 1) & _AC_TOP_MASK
                self._high = (
                    ((self._high - _AC_QUARTER) << 1) | 1
                ) & _AC_TOP_MASK
            else:
                break

    def finish(self) -> bytes:
        """Flush remaining bits."""
        # Emit one final bit (the bit determining which interval to commit to).
        self._pending_bits += 1
        if self._low < _AC_QUARTER:
            self._writer.write_bit_plus_pending(0, self._pending_bits)
        else:
            self._writer.write_bit_plus_pending(1, self._pending_bits)
        return self._writer.flush()


class _RangeDecoder:
    """CACM-style arithmetic decoder."""

    def __init__(self, data: bytes) -> None:
        self._reader = _ACBitReader(data)
        self._low: int = 0
        self._high: int = _AC_TOP_MASK
        self._value: int = 0
        for _ in range(_AC_CODE_BITS):
            self._value = (self._value << 1) | self._reader.read_bit()

    def decode_target(self, total_freq: int) -> int:
        rng = self._high - self._low + 1
        target = ((self._value - self._low + 1) * total_freq - 1) // rng
        return target

    def advance(self, cum_low: int, cum_high: int, total_freq: int) -> None:
        rng = self._high - self._low + 1
        new_high = self._low + (rng * cum_high) // total_freq - 1
        new_low = self._low + (rng * cum_low) // total_freq
        self._low = new_low
        self._high = new_high
        while True:
            if self._high < _AC_HALF:
                pass  # both shift left, MSB=0
            elif self._low >= _AC_HALF:
                self._value -= _AC_HALF
                self._low -= _AC_HALF
                self._high -= _AC_HALF
            elif self._low >= _AC_QUARTER and self._high < _AC_THREE_Q:
                self._value -= _AC_QUARTER
                self._low -= _AC_QUARTER
                self._high -= _AC_QUARTER
            else:
                break
            self._low = (self._low << 1) & _AC_TOP_MASK
            self._high = ((self._high << 1) | 1) & _AC_TOP_MASK
            self._value = (
                ((self._value << 1) & _AC_TOP_MASK) | self._reader.read_bit()
            )


def _build_cum_table(freqs: torch.Tensor) -> list[int]:
    """Return cumulative-frequency prefix sums; length ``K + 1``."""
    cum = [0]
    running = 0
    for f in freqs.tolist():
        running += int(f)
        cum.append(running)
    return cum


# ── High-level encode / decode entry points ─────────────────────────────


def encode_jscc_stream(
    symbols: list[int],
    side_states: torch.Tensor,
    model: ScorerConditionalProbabilityModel,
) -> bytes:
    """Encode a symbol stream with scorer-conditional probabilities.

    Args:
        symbols: list of symbol indices in [0, alphabet_size).
        side_states: tensor of shape ``(N, side_dim)`` carrying the
            side-state per symbol. ``len(symbols) == N``.
        model: the conditional density predictor.

    Returns:
        Encoded byte stream. NOTE: this is the raw range-coded payload —
        wrap with ``serialize_jscc_section`` for an archive-ready section.

    Raises:
        ValueError: on shape / range mismatches.
    """
    N = len(symbols)
    if side_states.ndim != 2:
        raise ValueError(
            f"side_states must be 2-D (N, side_dim), got shape "
            f"{tuple(side_states.shape)}"
        )
    if side_states.shape[0] != N:
        raise ValueError(
            f"side_states.shape[0]={side_states.shape[0]} != "
            f"len(symbols)={N}"
        )
    if side_states.shape[1] != model.side_dim:
        raise ValueError(
            f"side_states.shape[1]={side_states.shape[1]} != "
            f"model.side_dim={model.side_dim}"
        )
    for i, sym in enumerate(symbols):
        if not (0 <= sym < model.alphabet_size):
            raise ValueError(
                f"symbol[{i}]={sym} out of range "
                f"[0, {model.alphabet_size})"
            )

    enc = _RangeEncoder()
    for i in range(N):
        freqs = model.integer_frequency_table(side_states[i])
        cum = _build_cum_table(freqs)
        sym = symbols[i]
        enc.encode_symbol(cum[sym], cum[sym + 1], TOTAL_FREQ)
    return enc.finish()


def decode_jscc_stream(
    data: bytes,
    side_states: torch.Tensor,
    model: ScorerConditionalProbabilityModel,
    n_symbols: int,
) -> list[int]:
    """Decode a JSCC-encoded symbol stream.

    Args:
        data: encoded byte stream from ``encode_jscc_stream``.
        side_states: tensor of shape ``(n_symbols, side_dim)`` — must match
            the side-states used at encode time bit-exactly.
        model: same conditional density predictor as used at encode time.
        n_symbols: number of symbols to decode.

    Returns:
        Decoded symbol list of length ``n_symbols``.
    """
    if side_states.ndim != 2:
        raise ValueError(
            f"side_states must be 2-D, got shape {tuple(side_states.shape)}"
        )
    if side_states.shape[0] != n_symbols:
        raise ValueError(
            f"side_states.shape[0]={side_states.shape[0]} != "
            f"n_symbols={n_symbols}"
        )
    dec = _RangeDecoder(data)
    out: list[int] = []
    for i in range(n_symbols):
        freqs = model.integer_frequency_table(side_states[i])
        cum = _build_cum_table(freqs)
        target = dec.decode_target(TOTAL_FREQ)
        # Find the symbol whose cum range contains target
        # Linear scan — fine for moderate alphabet sizes; binary search
        # would scale to alphabet_size >> 1000.
        sym = 0
        for s in range(model.alphabet_size):
            if cum[s] <= target < cum[s + 1]:
                sym = s
                break
        out.append(sym)
        dec.advance(cum[sym], cum[sym + 1], TOTAL_FREQ)
    return out


# ── High-level ScorerConditionalEntropyCoder class (encode + decode + state) ──


class ScorerConditionalEntropyCoder:
    """High-level scorer-conditional entropy coder.

    Wraps the ``encode_jscc_stream`` / ``decode_jscc_stream`` pair with a
    persistent ``ScorerConditionalProbabilityModel`` and an explicit
    archive-section format (magic bytes + version + payload-length prefix).

    Cross-references the ``tac.packet_compiler`` registry: the binary
    section emitted by ``.encode_section()`` can be embedded in an archive
    grammar at any byte offset (see ``tac.codec.jscc.archive_format``).

    Args:
        model: the conditional probability model.

    Example:
        >>> model = ScorerConditionalProbabilityModel(
        ...     side_dim=4, alphabet_size=16)
        >>> coder = ScorerConditionalEntropyCoder(model)
        >>> side = torch.randn(10, 4)
        >>> symbols = [int(s) for s in torch.randint(0, 16, (10,)).tolist()]
        >>> payload = coder.encode(symbols, side)
        >>> decoded = coder.decode(payload, side, n_symbols=10)
        >>> assert decoded == symbols
    """

    def __init__(self, model: ScorerConditionalProbabilityModel) -> None:
        if not isinstance(model, ScorerConditionalProbabilityModel):
            raise TypeError(
                f"model must be a ScorerConditionalProbabilityModel, got "
                f"{type(model).__name__}"
            )
        self.model = model

    def encode(
        self, symbols: list[int], side_states: torch.Tensor
    ) -> bytes:
        """Encode a symbol stream → raw range-coded bytes."""
        return encode_jscc_stream(symbols, side_states, self.model)

    def decode(
        self,
        data: bytes,
        side_states: torch.Tensor,
        n_symbols: int,
    ) -> list[int]:
        """Decode a JSCC byte stream → symbol list."""
        return decode_jscc_stream(data, side_states, self.model, n_symbols)

    def estimated_coded_bits(
        self, symbols: list[int], side_states: torch.Tensor
    ) -> float:
        """Return the theoretical coded length under the conditional model.

        ``-Σ_t log2 p(x_t | y_t)``. Useful for comparison vs an unconditional
        coder's coded length to measure the JSCC savings.

        Args:
            symbols: list of symbol indices.
            side_states: 2-D tensor ``(N, side_dim)``.

        Returns:
            Estimated coded length in bits (a real-valued upper bound for
            the integer range coder).
        """
        if side_states.shape[0] != len(symbols):
            raise ValueError(
                f"side_states.shape[0]={side_states.shape[0]} != "
                f"len(symbols)={len(symbols)}"
            )
        with torch.no_grad():
            probs_all = self.model(side_states)
            total_bits = 0.0
            for i, sym in enumerate(symbols):
                p = float(probs_all[i, sym].item())
                # Clip to avoid -log(0)
                p = max(p, 1.0 / (TOTAL_FREQ * 2.0))
                total_bits -= math.log2(p)
        return total_bits
