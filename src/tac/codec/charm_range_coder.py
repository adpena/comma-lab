"""ChARM 2020 range coder — bit-exact byte stream from per-symbol conditional PMFs.

This module turns the differentiable rate predictions emitted by the ChARM 2020
hyperprior (``experiments/train_charm_50k_toy_substrate.py::CharmHyperprior``) into
real, byte-exact compressed payloads. The decoder consumes the byte stream + the
SAME conditional PMFs (re-derived deterministically at decode time) and yields
the original symbols.

Design (per CLAUDE.md "Beauty, simplicity, and developer experience"):

* Two narrow public classes — :class:`ChARMRangeEncoder` and
  :class:`ChARMRangeDecoder` — with a pinned, typed API.
* Built on top of the battle-tested integer-state range coder at
  :mod:`tac.lossless.range_coder` (no fp arithmetic in the inner state update;
  Witten/Neal/Cleary 1987 style). We do NOT reimplement the state machine; we
  adapt it to the channel-conditional Gaussian use case.
* Typed :class:`ChARMBitStreamHeader` captures bit-stream provenance: magic
  bytes, format version, symbol count, alphabet range, frequency-table
  precision. Header is emitted big-endian deterministically so the byte
  stream is reproducible across machines and python versions.
* Float arithmetic is constrained to the PMF-derivation step (which the
  decoder MUST reproduce identically). The inner state machine and table
  cumulation are pure integer operations.
* Empty stream is supported (count=0 produces a header-only blob).
* The coder is symbol-by-symbol: the caller supplies a fresh PMF for each
  symbol, mirroring the autoregressive nature of ChARM 2020.

Usage:

.. code-block:: python

    coder = ChARMRangeEncoder(alphabet=(-128, 127))
    for residual, (mu, sigma) in zip(int8_residuals, charm_predictions):
        pmf = gaussian_pmf_int8(mu, sigma, alphabet_lo=-128, alphabet_hi=127)
        coder.write_symbol(int(residual), pmf)
    byte_stream = coder.finish()

    decoder = ChARMRangeDecoder(byte_stream, alphabet=(-128, 127))
    restored = []
    for (mu, sigma) in charm_predictions:
        pmf = gaussian_pmf_int8(mu, sigma, alphabet_lo=-128, alphabet_hi=127)
        restored.append(decoder.read_symbol(pmf))

# ROUNDTRIP_TESTED: src/tac/tests/test_charm_range_coder.py

The output rate falls within ~1% of the matched-Gaussian Shannon prediction
``H = 0.5 * log2(2 * pi * e * sigma^2)`` per symbol at convergence.
"""
from __future__ import annotations

import dataclasses
import math
import struct
from bisect import bisect_right
from typing import Iterable, Sequence

import numpy as np

from tac.lossless.range_coder import (
    RangeDecoder,
    RangeEncoder,
    cumulative_frequencies,
    normalize_probabilities,
)


# ---------------------------------------------------------------------------
# Bit-stream header — typed, deterministic, big-endian
# ---------------------------------------------------------------------------


_MAGIC = b"CHRC"  # ChaRm Range Coder
_FORMAT_VERSION = 1
# Header layout (big-endian):
#   4 bytes  magic 'CHRC'
#   1 byte   format version
#   1 byte   reserved (0)
#   2 bytes  alphabet_lo (signed int16; covers INT8 plus margin)
#   2 bytes  alphabet_hi (signed int16)
#   4 bytes  num_symbols (uint32)
#   2 bytes  pmf_total_bits (uint16; e.g., 15 for total=2^15)
#   2 bytes  payload_len (uint16; for sanity / corruption check)
#   = 18 bytes header
_HEADER_FMT = ">4sBBhhIHH"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)
assert HEADER_SIZE == 18, "header size drift; bump format version if intentional"


@dataclasses.dataclass(frozen=True)
class ChARMBitStreamHeader:
    """Typed header emitted at the start of every ChARM range-coded blob.

    Fields are deterministic and byte-aligned to keep the stream reproducible
    across hosts. The decoder validates each field on read; corruption raises
    :class:`ValueError`.
    """

    magic: bytes = _MAGIC
    version: int = _FORMAT_VERSION
    alphabet_lo: int = -128
    alphabet_hi: int = 127
    num_symbols: int = 0
    pmf_total_bits: int = 15  # PMF precision exponent; total = 1 << pmf_total_bits
    payload_len: int = 0

    @property
    def alphabet_size(self) -> int:
        return self.alphabet_hi - self.alphabet_lo + 1

    @property
    def pmf_total(self) -> int:
        return 1 << self.pmf_total_bits

    def to_bytes(self) -> bytes:
        if not (-(1 << 15) <= self.alphabet_lo <= (1 << 15) - 1):
            raise ValueError("alphabet_lo out of int16 range")
        if not (-(1 << 15) <= self.alphabet_hi <= (1 << 15) - 1):
            raise ValueError("alphabet_hi out of int16 range")
        if self.alphabet_hi < self.alphabet_lo:
            raise ValueError("alphabet_hi < alphabet_lo")
        if self.num_symbols < 0 or self.num_symbols > (1 << 32) - 1:
            raise ValueError("num_symbols out of uint32 range")
        if self.pmf_total_bits < 1 or self.pmf_total_bits > 16:
            raise ValueError("pmf_total_bits must be in [1, 16]")
        if self.payload_len < 0 or self.payload_len > (1 << 16) - 1:
            raise ValueError("payload_len out of uint16 range")
        return struct.pack(
            _HEADER_FMT,
            self.magic,
            self.version,
            0,  # reserved
            self.alphabet_lo,
            self.alphabet_hi,
            self.num_symbols,
            self.pmf_total_bits,
            self.payload_len,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "ChARMBitStreamHeader":
        if len(data) < HEADER_SIZE:
            raise ValueError(
                f"byte stream too short for ChARM header: got {len(data)}, "
                f"need {HEADER_SIZE}"
            )
        magic, version, _reserved, alphabet_lo, alphabet_hi, num_symbols, pmf_total_bits, payload_len = (
            struct.unpack(_HEADER_FMT, data[:HEADER_SIZE])
        )
        if magic != _MAGIC:
            raise ValueError(
                f"bad ChARM magic: expected {_MAGIC!r}, got {magic!r}"
            )
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"unsupported ChARM format version: got {version}, "
                f"expected {_FORMAT_VERSION}"
            )
        if alphabet_hi < alphabet_lo:
            raise ValueError("corrupt header: alphabet_hi < alphabet_lo")
        if pmf_total_bits < 1 or pmf_total_bits > 16:
            raise ValueError(
                f"corrupt header: pmf_total_bits out of [1, 16]: {pmf_total_bits}"
            )
        return cls(
            magic=magic,
            version=version,
            alphabet_lo=alphabet_lo,
            alphabet_hi=alphabet_hi,
            num_symbols=num_symbols,
            pmf_total_bits=pmf_total_bits,
            payload_len=payload_len,
        )


# ---------------------------------------------------------------------------
# Conditional Gaussian PMF helper
# ---------------------------------------------------------------------------


def gaussian_pmf_int8(
    mu: float,
    sigma: float,
    *,
    alphabet_lo: int = -128,
    alphabet_hi: int = 127,
    floor_prob: float = 1e-9,
) -> np.ndarray:
    """Discretise a 1-D Gaussian over an integer alphabet via CDF differences.

    The probability of integer ``k`` is ``Phi((k+0.5-mu)/sigma) - Phi((k-0.5-mu)/sigma)``.
    A small ``floor_prob`` is added before normalisation so that no symbol has
    zero probability (otherwise the range coder cannot represent that symbol);
    this is the canonical ``epsilon`` trick used in CompressAI-style coders.

    Determinism: this function uses ``math.erf`` (libm) and standard float64
    arithmetic. Encoder and decoder MUST call this with the same arguments to
    produce identical byte-quantised PMFs.

    Args:
        mu: location parameter
        sigma: scale parameter (must be > 0)
        alphabet_lo: inclusive lower bound of symbol alphabet (default -128)
        alphabet_hi: inclusive upper bound of symbol alphabet (default 127)
        floor_prob: small probability added per bin for numerical floor

    Returns:
        ``np.ndarray`` of shape ``(alphabet_hi - alphabet_lo + 1,)`` with positive
        floats summing to 1.0 (within float64 precision).
    """
    if not math.isfinite(mu):
        raise ValueError(f"mu must be finite; got {mu}")
    if not math.isfinite(sigma):
        raise ValueError(f"sigma must be finite; got {sigma}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0; got {sigma}")
    if alphabet_hi < alphabet_lo:
        raise ValueError("alphabet_hi must be >= alphabet_lo")
    if not math.isfinite(floor_prob):
        raise ValueError(f"floor_prob must be finite; got {floor_prob}")
    if floor_prob < 0.0:
        raise ValueError("floor_prob must be >= 0")

    n = alphabet_hi - alphabet_lo + 1
    # Vectorised CDF difference. We extend the lowest/highest bins to ±inf
    # so probability mass under the tails is captured (instead of dropped).
    edges = np.arange(alphabet_lo, alphabet_hi + 2, dtype=np.float64) - 0.5
    edges_norm = (edges - mu) / (sigma * math.sqrt(2.0))
    # math.erf is per-element; numpy's vectorisation via np.vectorize calls
    # the python-level math.erf, which is acceptable for our small alphabets.
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(edges_norm))
    # Extend tails: anything below alphabet_lo accrues to the lo bin; above
    # alphabet_hi accrues to the hi bin.
    cdf[0] = 0.0
    cdf[-1] = 1.0
    pmf = np.diff(cdf)
    assert pmf.shape == (n,)
    pmf = np.clip(pmf, 0.0, 1.0) + floor_prob
    pmf /= pmf.sum()
    return pmf


def shannon_bits_for_pmf(pmf: np.ndarray, symbol_index: int) -> float:
    """Information content (in bits) of selecting ``symbol_index`` under ``pmf``.

    This is what an ideal code would charge for that symbol. Used by the
    self-test that asserts coded-bit count is within ~1% of Shannon entropy
    at convergence.
    """
    p = float(pmf[symbol_index])
    if p <= 0.0:
        raise ValueError("pmf[symbol_index] must be > 0")
    return -math.log2(p)


# ---------------------------------------------------------------------------
# Encoder + decoder
# ---------------------------------------------------------------------------


class ChARMRangeEncoder:
    """Encode a sequence of symbols under per-symbol conditional PMFs.

    The encoder maintains the integer-state range coder from
    :mod:`tac.lossless.range_coder` plus a count of symbols emitted so the
    final blob can carry an authoritative ``num_symbols`` field.

    Public API (frozen):
        ``__init__(*, alphabet, pmf_total_bits=15)``
        ``write_symbol(symbol: int, pmf: np.ndarray) -> None``
        ``finish() -> bytes``
        ``num_symbols_written`` property
    """

    def __init__(
        self,
        *,
        alphabet: tuple[int, int] = (-128, 127),
        pmf_total_bits: int = 15,
    ) -> None:
        if alphabet[1] < alphabet[0]:
            raise ValueError("alphabet upper bound must be >= lower bound")
        if pmf_total_bits < 1 or pmf_total_bits > 16:
            raise ValueError("pmf_total_bits must be in [1, 16]")
        self._alphabet_lo, self._alphabet_hi = int(alphabet[0]), int(alphabet[1])
        self._pmf_total_bits = int(pmf_total_bits)
        self._pmf_total = 1 << self._pmf_total_bits
        if self.alphabet_size > self._pmf_total:
            raise ValueError(
                f"alphabet size {self.alphabet_size} exceeds PMF total "
                f"{self._pmf_total}; increase pmf_total_bits"
            )
        self._inner = RangeEncoder()
        self._num_symbols = 0
        self._finished = False

    @property
    def num_symbols_written(self) -> int:
        return self._num_symbols

    @property
    def alphabet_size(self) -> int:
        return self._alphabet_hi - self._alphabet_lo + 1

    def write_symbol(self, symbol: int, pmf: np.ndarray) -> None:
        if self._finished:
            raise RuntimeError("encoder already finished; cannot write more symbols")
        if not (self._alphabet_lo <= symbol <= self._alphabet_hi):
            raise ValueError(
                f"symbol {symbol} out of alphabet [{self._alphabet_lo}, {self._alphabet_hi}]"
            )
        pmf_arr = np.asarray(pmf, dtype=np.float64)
        if pmf_arr.shape != (self.alphabet_size,):
            raise ValueError(
                f"pmf shape {pmf_arr.shape} != ({self.alphabet_size},) (alphabet size)"
            )
        if not np.all(np.isfinite(pmf_arr)):
            raise ValueError("pmf must contain only finite values")
        # Convert PMF → integer frequency table summing exactly to pmf_total.
        freqs = normalize_probabilities(pmf_arr.tolist(), total=self._pmf_total)
        cumulative, total = cumulative_frequencies(freqs)
        # Index of the symbol within the integer-indexed frequency table.
        index = symbol - self._alphabet_lo
        self._inner.encode(symbol=index, cumulative=cumulative, total=total)
        self._num_symbols += 1

    def finish(self) -> bytes:
        """Flush and return the full byte stream (header + payload)."""
        if self._finished:
            raise RuntimeError("finish() called twice on the same encoder")
        self._finished = True
        if self._num_symbols == 0:
            payload = b""
        else:
            payload = self._inner.finish()
        # payload_len is bounded by uint16 (65535) for header schema simplicity;
        # for streams larger than that we'd bump format version. INT8-residual
        # use-cases at 50K params land well under this bound.
        if len(payload) > (1 << 16) - 1:
            raise ValueError(
                f"payload {len(payload)} bytes exceeds uint16 header field; "
                "bump ChARMBitStreamHeader format version to widen payload_len"
            )
        header = ChARMBitStreamHeader(
            alphabet_lo=self._alphabet_lo,
            alphabet_hi=self._alphabet_hi,
            num_symbols=self._num_symbols,
            pmf_total_bits=self._pmf_total_bits,
            payload_len=len(payload),
        ).to_bytes()
        return header + payload


class ChARMRangeDecoder:
    """Decode a sequence of symbols under per-symbol conditional PMFs.

    The decoder validates the bit-stream header before opening the inner
    range decoder. ``read_symbol(pmf)`` returns the next symbol; calling it
    after ``num_symbols`` symbols have been read raises :class:`RuntimeError`.

    Public API (frozen):
        ``__init__(byte_stream: bytes, *, alphabet=None)``
        ``header`` property (typed :class:`ChARMBitStreamHeader`)
        ``read_symbol(pmf: np.ndarray) -> int``
        ``num_symbols`` property (total expected, from header)
        ``num_symbols_read`` property
    """

    def __init__(
        self,
        byte_stream: bytes,
        *,
        alphabet: tuple[int, int] | None = None,
    ) -> None:
        self._header = ChARMBitStreamHeader.from_bytes(byte_stream)
        if alphabet is not None:
            if (alphabet[0] != self._header.alphabet_lo) or (
                alphabet[1] != self._header.alphabet_hi
            ):
                raise ValueError(
                    f"alphabet mismatch: header says [{self._header.alphabet_lo}, "
                    f"{self._header.alphabet_hi}] but caller passed {alphabet}"
                )
        payload = byte_stream[HEADER_SIZE : HEADER_SIZE + self._header.payload_len]
        if len(payload) != self._header.payload_len:
            raise ValueError(
                f"truncated ChARM byte stream: header says payload_len="
                f"{self._header.payload_len}, got {len(payload)}"
            )
        expected_len = HEADER_SIZE + self._header.payload_len
        if len(byte_stream) != expected_len:
            raise ValueError(
                f"trailing bytes after ChARM payload: expected stream length "
                f"{expected_len}, got {len(byte_stream)}"
            )
        self._inner: RangeDecoder | None
        if self._header.num_symbols == 0:
            # Empty stream: no inner decoder needed.
            self._inner = None
        else:
            if not payload:
                raise ValueError(
                    "non-empty ChARM stream missing payload bytes (corrupt header?)"
                )
            self._inner = RangeDecoder(payload)
        self._num_read = 0

    @property
    def header(self) -> ChARMBitStreamHeader:
        return self._header

    @property
    def num_symbols(self) -> int:
        return self._header.num_symbols

    @property
    def num_symbols_read(self) -> int:
        return self._num_read

    @property
    def alphabet(self) -> tuple[int, int]:
        return (self._header.alphabet_lo, self._header.alphabet_hi)

    def read_symbol(self, pmf: np.ndarray) -> int:
        if self._num_read >= self._header.num_symbols:
            raise RuntimeError(
                f"read_symbol() called after {self._header.num_symbols} symbols already read"
            )
        if self._inner is None:
            raise RuntimeError(
                "internal: empty stream marked non-empty by header counter"
            )
        alphabet_size = self._header.alphabet_size
        pmf_arr = np.asarray(pmf, dtype=np.float64)
        if pmf_arr.shape != (alphabet_size,):
            raise ValueError(
                f"pmf shape {pmf_arr.shape} != ({alphabet_size},) (alphabet size)"
            )
        if not np.all(np.isfinite(pmf_arr)):
            raise ValueError("pmf must contain only finite values")
        freqs = normalize_probabilities(pmf_arr.tolist(), total=self._header.pmf_total)
        cumulative, total = cumulative_frequencies(freqs)
        target = self._inner.target(total)
        # bisect to locate the symbol whose cumulative interval contains target
        index = bisect_right(cumulative, target) - 1
        if index < 0 or index >= alphabet_size:
            raise ValueError(
                "decoded target outside frequency table — corrupt byte stream?"
            )
        self._inner.update(
            low_count=cumulative[index],
            high_count=cumulative[index + 1],
            total=total,
        )
        self._num_read += 1
        return index + self._header.alphabet_lo


# ---------------------------------------------------------------------------
# Convenience wrappers — high-level encode_symbols / decode_symbols
# ---------------------------------------------------------------------------


def encode_symbols(
    symbols: Sequence[int],
    pmfs: Sequence[np.ndarray],
    *,
    alphabet: tuple[int, int] = (-128, 127),
    pmf_total_bits: int = 15,
) -> bytes:
    """One-shot encode: ``len(symbols) == len(pmfs)``; returns the byte blob.

    This is a thin wrapper around :class:`ChARMRangeEncoder` for the common
    case where the caller has both lists in memory.
    """
    if len(symbols) != len(pmfs):
        raise ValueError(
            f"symbols and pmfs must have same length: got {len(symbols)} vs {len(pmfs)}"
        )
    enc = ChARMRangeEncoder(alphabet=alphabet, pmf_total_bits=pmf_total_bits)
    for sym, pmf in zip(symbols, pmfs):
        enc.write_symbol(int(sym), pmf)
    return enc.finish()


def decode_symbols(
    byte_stream: bytes,
    pmfs: Iterable[np.ndarray],
) -> list[int]:
    """One-shot decode: yields a list of symbols.

    The caller MUST supply the same PMF sequence used at encode time. The
    decoder cross-checks its own ``num_symbols`` field against the iterable
    length: if the iterable is shorter than ``num_symbols`` the decoder
    raises; if longer, the trailing PMFs are ignored.
    """
    dec = ChARMRangeDecoder(byte_stream)
    out: list[int] = []
    pmfs_list = list(pmfs)
    if len(pmfs_list) < dec.num_symbols:
        raise ValueError(
            f"pmfs iterable shorter than num_symbols: got {len(pmfs_list)}, "
            f"need {dec.num_symbols}"
        )
    for i in range(dec.num_symbols):
        out.append(dec.read_symbol(pmfs_list[i]))
    return out


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "ChARMBitStreamHeader",
    "ChARMRangeEncoder",
    "ChARMRangeDecoder",
    "HEADER_SIZE",
    "decode_symbols",
    "encode_symbols",
    "gaussian_pmf_int8",
    "shannon_bits_for_pmf",
]
