"""Syndrome-trellis coding (STC) for ternary mask-delta payloads.

Background (CLAUDE.md "Tomáš Filler — Fridrich's other student"):
    > "Syndrome-trellis coding (STC); parity-check codes for per-frame mask
    >  payload"

REVIEW-SEC-COMP Filler finding:
    "STC opportunity unexploited for PARADIGM-α mask payload (per-frame mask
     deltas are ternary — canonical STC target per Filler-Fridrich-Pevný 2011)."

Reference
---------
Filler, T., Judas, J., & Fridrich, J. (2011). "Minimizing additive distortion
in steganography using syndrome-trellis codes." IEEE TIFS 6(3): 920-935.

Key idea (binary STC)
---------------------
Cover sequence ``x ∈ {0,1}^n`` carries message ``m ∈ {0,1}^k`` via stego ``y``
satisfying syndrome constraint ``H · y = m`` over GF(2). The encoder finds the
``y`` minimizing total embedding cost ``Σ ρ_i · [x_i ≠ y_i]`` subject to the
linear syndrome constraint. The submatrix ``H̄`` is repeated horizontally to
form ``H`` (block size ``h``, code length ``w``). Decoding is trivial:
``m = H · y``.

Ternary STC (this implementation) for mask deltas
-------------------------------------------------
Per-frame mask deltas are ternary (-1, 0, +1) when neighbouring frames change
class membership. We split the ternary cover into two binary streams: a
*sign-or-zero* stream (1 if delta != 0) and a *sign* stream (1 if delta == +1,
defined only where sign-or-zero == 1). Each is encoded with a binary STC under
its own embedding-cost vector, derived from a wet-paper-style rule:

* ``ρ_i = 1`` for normal pixels (changes are cheap).
* ``ρ_i = WET_COST`` (very large) for pixels we never want to flip — e.g.
  pixels at hard class boundaries that the detector keys off of.

The wet-paper construction handles the asymmetry of ternary encoding: when a
sign-or-zero bit is forced from 1 to 0 (i.e. the codec discards a real change)
the original delta value is irrelevant, so the sign-stream cost at that
position is set to wet (skipped during minimization).

Parameters
----------
``constraint_height`` (h): rows of ``H̄``. Larger ``h`` → exponentially better
distortion at the cost of trellis-state explosion (Filler 2011 recommends
``h <= 12`` for tractable encoding). We default to ``h=8``.

The implementation here is a pedagogical reference: it constructs the trellis
explicitly and uses Viterbi-style dynamic programming. It is not optimised for
the >100KB streams used in the full PARADIGM-α pipeline, but is correct on the
test vectors in ``test_syndrome_trellis_codec.py``.

Falsification scope
-------------------
``filler_stc_ternary_mask_delta_only``: only the binary-STC + ternary-split
variant is tested. Score-aware embedding costs (detector-in-loop), the
double-layered STC variant (Filler-Pevný 2010 dual STC), and the GF(q>2)
variant remain in ``reactivation_criteria_remaining``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

WET_COST: float = 1.0e9  # cost forbidding a flip at this position


# ---------------------------------------------------------------------------
# Submatrix construction
# ---------------------------------------------------------------------------


def make_submatrix(constraint_height: int, code_length: int, *, seed: int = 0) -> np.ndarray:
    """Construct a sparse-ish ``h × w`` parity-check submatrix over GF(2).

    Filler 2011 recommends pseudo-random sparse submatrices with a constant
    column weight. We use 2 ones per column (a 2-regular code), placed at
    deterministic positions seeded by ``seed`` so encode/decode remain
    reproducible.
    """
    if constraint_height <= 0 or code_length <= 0:
        raise ValueError("constraint_height and code_length must be positive")
    if constraint_height > 16:
        raise ValueError("constraint_height > 16 is intractable in this reference impl")
    rng = np.random.default_rng(seed)
    H = np.zeros((constraint_height, code_length), dtype=np.uint8)
    for j in range(code_length):
        rows = rng.choice(constraint_height, size=2, replace=False)
        H[rows, j] = 1
    # Force first row to be all-ones so column-sum parity is well-defined for
    # the trivial "all zeros" message, simplifying syndrome bookkeeping.
    H[0, :] = 1
    return H


# ---------------------------------------------------------------------------
# Binary STC core
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class STCParams:
    """Parameters for a binary syndrome-trellis code."""

    constraint_height: int
    submatrix_seed: int = 0

    def __post_init__(self) -> None:
        if self.constraint_height < 1 or self.constraint_height > 16:
            raise ValueError("constraint_height must be in [1, 16]")


def _viterbi_encode_binary(
    cover: np.ndarray, costs: np.ndarray, message: np.ndarray, H_bar: np.ndarray
) -> np.ndarray:
    """Viterbi-style trellis search for the binary STC stego sequence.

    Returns ``y`` such that ``H · y = message`` and ``Σ costs_i · [x_i != y_i]``
    is minimised. This is a pedagogical reference; for production-scale streams
    use the official C reference (Filler 2011).
    """
    n = int(cover.size)
    h = int(H_bar.shape[0])
    k = int(message.size)
    if H_bar.shape[1] != n:
        raise ValueError(
            f"H_bar shape {H_bar.shape} is incompatible with cover length {n}"
        )
    if k != h:
        raise ValueError(
            f"For block size n={n} this reference impl handles message length k={h}"
        )

    n_states = 1 << h
    INF = np.float64(np.inf)
    # dp[s] = min cost to reach state s after processing some prefix.
    dp = np.full(n_states, INF, dtype=np.float64)
    dp[0] = 0.0
    parent = np.full((n, n_states), -1, dtype=np.int32)
    chose_one = np.zeros((n, n_states), dtype=np.uint8)

    # Precompute integer column representations of H_bar.
    cols_int = np.zeros(n, dtype=np.int64)
    for j in range(n):
        v = 0
        for i in range(h):
            if H_bar[i, j]:
                v |= 1 << i
        cols_int[j] = v

    for j in range(n):
        col = int(cols_int[j])
        nxt = np.full(n_states, INF, dtype=np.float64)
        # Transition costs: y_j = 0 keeps state, y_j = 1 XORs state with col.
        cost_zero = 0.0 if cover[j] == 0 else float(costs[j])
        cost_one = 0.0 if cover[j] == 1 else float(costs[j])
        for s in range(n_states):
            base = dp[s]
            if not np.isfinite(base):
                continue
            # y_j = 0
            cand = base + cost_zero
            if cand < nxt[s]:
                nxt[s] = cand
                parent[j, s] = s
                chose_one[j, s] = 0
            # y_j = 1
            s2 = s ^ col
            cand = base + cost_one
            if cand < nxt[s2]:
                nxt[s2] = cand
                parent[j, s2] = s
                chose_one[j, s2] = 1
        dp = nxt

    target_state = 0
    for i in range(h):
        if message[i]:
            target_state |= 1 << i
    if not np.isfinite(dp[target_state]):
        raise RuntimeError("Viterbi encode found no path satisfying the syndrome")

    # Backtrack
    y = np.zeros(n, dtype=np.uint8)
    s = target_state
    for j in range(n - 1, -1, -1):
        bit = chose_one[j, s]
        y[j] = bit
        s = parent[j, s]
    return y


def stc_encode_block(
    cover: Sequence[int],
    costs: Sequence[float],
    message: Sequence[int],
    params: STCParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a single STC block.

    Returns ``(y, H)`` where ``y`` is the stego bit-vector and ``H`` is the
    parity-check matrix that satisfies ``H @ y = message`` (mod 2).
    """
    cover_arr = np.asarray(cover, dtype=np.uint8)
    costs_arr = np.asarray(costs, dtype=np.float64)
    message_arr = np.asarray(message, dtype=np.uint8)
    if cover_arr.ndim != 1 or costs_arr.ndim != 1 or message_arr.ndim != 1:
        raise ValueError("cover/costs/message must be 1-D arrays")
    if cover_arr.size != costs_arr.size:
        raise ValueError("cover and costs must have equal length")
    n = int(cover_arr.size)
    h = params.constraint_height
    if message_arr.size != h:
        raise ValueError(
            f"message length {message_arr.size} must equal constraint_height {h}"
        )

    H_bar = make_submatrix(h, n, seed=params.submatrix_seed)
    y = _viterbi_encode_binary(cover_arr, costs_arr, message_arr, H_bar)
    return y, H_bar


def stc_decode_block(stego: np.ndarray, H_bar: np.ndarray) -> np.ndarray:
    """Recover the embedded message from a stego block: ``m = H · y``."""
    stego = np.asarray(stego, dtype=np.uint8)
    if stego.ndim != 1 or stego.size != H_bar.shape[1]:
        raise ValueError("stego shape incompatible with H_bar")
    return (H_bar @ stego) % 2


# ---------------------------------------------------------------------------
# Ternary STC: split a ternary cover into sign-or-zero + sign streams
# ---------------------------------------------------------------------------


def split_ternary(
    deltas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ternary deltas (-1, 0, +1) into two binary streams.

    Returns ``(soz, sign, sign_mask)`` where:
    * ``soz[i] = 1`` iff ``deltas[i] != 0``
    * ``sign[i] = 1`` iff ``deltas[i] == +1`` (only meaningful where soz=1)
    * ``sign_mask[i] = soz[i]`` — used downstream to wet-cost positions where
      sign is irrelevant.
    """
    deltas = np.asarray(deltas, dtype=np.int8)
    if not np.all(np.isin(deltas, [-1, 0, 1])):
        raise ValueError("deltas must be ternary in {-1, 0, +1}")
    soz = (deltas != 0).astype(np.uint8)
    sign = (deltas == 1).astype(np.uint8)
    return soz, sign, soz.copy()


def join_ternary(soz: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Inverse of ``split_ternary``: reconstruct ternary deltas.

    Where ``soz == 0``, the output is 0 regardless of ``sign``.
    """
    soz = np.asarray(soz, dtype=np.uint8)
    sign = np.asarray(sign, dtype=np.uint8)
    if soz.shape != sign.shape:
        raise ValueError("soz and sign must have identical shape")
    out = np.where(soz == 0, np.int8(0), np.where(sign == 1, np.int8(1), np.int8(-1)))
    return out.astype(np.int8)


def ternary_stc_encode_message(
    deltas: np.ndarray,
    soz_costs: np.ndarray,
    sign_costs: np.ndarray,
    message: np.ndarray,
    params: STCParams,
) -> dict:
    """Embed ``message`` (binary) into a single block of ternary cover.

    The implementation maps ``message`` once into the SoZ stream — that is,
    ``soz_stego = stc_encode_block(soz, soz_costs, message)`` — and leaves the
    sign stream unchanged. (The dual-layer STC of Filler-Pevný 2010 embeds two
    independent messages into both streams; that variant is in the
    reactivation list.)

    Returns a dict with keys ``soz_stego``, ``sign_stego``, ``H_bar``,
    ``stego_deltas``, and ``cost`` (total embedding distortion).
    """
    soz, sign, _ = split_ternary(deltas)
    soz_stego, H_bar = stc_encode_block(soz, soz_costs, message, params)

    # Where soz_stego == 0 but soz == 1 we force sign to a deterministic
    # default; downstream consumers must treat sign[i] as undefined whenever
    # soz_stego[i] == 0.
    sign_stego = np.where(soz_stego == 1, sign, np.uint8(0))
    stego_deltas = join_ternary(soz_stego, sign_stego)

    flips_soz = int((soz_stego != soz).sum())
    flips_sign = int(((sign_stego != sign) & (soz_stego == 1) & (soz == 1)).sum())
    cost = float(
        ((soz_stego != soz).astype(np.float64) * soz_costs).sum()
        + (((sign_stego != sign) & (soz_stego == 1) & (soz == 1)).astype(np.float64) * sign_costs).sum()
    )

    return {
        "soz_stego": soz_stego,
        "sign_stego": sign_stego,
        "stego_deltas": stego_deltas,
        "H_bar": H_bar,
        "flips_soz": flips_soz,
        "flips_sign": flips_sign,
        "cost": cost,
    }


def ternary_stc_decode_message(stego_deltas: np.ndarray, H_bar: np.ndarray) -> np.ndarray:
    """Recover the embedded message from a ternary stego block."""
    stego_deltas = np.asarray(stego_deltas, dtype=np.int8)
    soz = (stego_deltas != 0).astype(np.uint8)
    return stc_decode_block(soz, H_bar)


# ---------------------------------------------------------------------------
# Streaming wrapper: blocked encode of a long ternary delta sequence
# ---------------------------------------------------------------------------


def ternary_stc_encode_stream(
    deltas: np.ndarray,
    costs: np.ndarray,
    *,
    block_size: int,
    params: STCParams,
    message_block_provider=None,
) -> dict:
    """Encode a long ternary stream block-by-block.

    ``costs`` is broadcast to both the SoZ and sign streams. Positions with
    ``costs[i] >= WET_COST / 2`` are forbidden flips.

    ``message_block_provider`` is a callable ``i -> np.ndarray`` returning the
    ``constraint_height``-bit message for block ``i``. If ``None``, each block
    receives the all-zero message — the encoder then minimises distortion
    subject to ``H · y = 0``, i.e. it produces the cheapest valid syndrome
    realisation. Useful as a baseline.
    """
    deltas = np.asarray(deltas, dtype=np.int8)
    costs = np.asarray(costs, dtype=np.float64)
    if deltas.ndim != 1 or deltas.shape != costs.shape:
        raise ValueError("deltas and costs must be 1-D arrays of equal length")
    if block_size < params.constraint_height:
        raise ValueError("block_size must be >= constraint_height")

    h = params.constraint_height
    n_blocks = (deltas.size + block_size - 1) // block_size
    total_cost = 0.0
    stego_blocks: list[np.ndarray] = []
    H_blocks: list[np.ndarray] = []
    flips_soz = 0
    flips_sign = 0

    for b in range(n_blocks):
        start = b * block_size
        stop = min(start + block_size, deltas.size)
        block = deltas[start:stop]
        block_costs = costs[start:stop]
        # Pad final block to ``block_size`` with wet zero cover.
        if block.size < block_size:
            pad = block_size - block.size
            block = np.concatenate([block, np.zeros(pad, dtype=np.int8)])
            block_costs = np.concatenate(
                [block_costs, np.full(pad, WET_COST, dtype=np.float64)]
            )

        if message_block_provider is None:
            message = np.zeros(h, dtype=np.uint8)
        else:
            message = np.asarray(message_block_provider(b), dtype=np.uint8)
            if message.size != h:
                raise ValueError(
                    f"message_block_provider returned wrong length for block {b}"
                )

        result = ternary_stc_encode_message(
            block,
            soz_costs=block_costs,
            sign_costs=block_costs,
            message=message,
            params=STCParams(
                constraint_height=h,
                submatrix_seed=params.submatrix_seed + b,
            ),
        )
        stego_blocks.append(result["stego_deltas"][: stop - start])
        H_blocks.append(result["H_bar"])
        total_cost += float(result["cost"])
        flips_soz += int(result["flips_soz"])
        flips_sign += int(result["flips_sign"])

    stego = np.concatenate(stego_blocks).astype(np.int8)
    return {
        "stego": stego,
        "H_blocks": H_blocks,
        "total_cost": total_cost,
        "flips_soz": flips_soz,
        "flips_sign": flips_sign,
        "n_blocks": n_blocks,
        "block_size": block_size,
    }


# ---------------------------------------------------------------------------
# Helper: ternary mask-delta extraction
# ---------------------------------------------------------------------------


def extract_mask_deltas_ternary(masks: np.ndarray) -> np.ndarray:
    """Return ternary deltas between consecutive class masks.

    For 5-class masks the raw delta ``masks[t] - masks[t-1]`` lives in
    ``[-4, +4]``. We project to ternary: ``-1`` if the raw delta is negative,
    ``+1`` if positive, ``0`` if equal. Consumers wanting the full integer
    delta should use a different codec (e.g. range coding); ternary STC is the
    canonical Filler-Fridrich-Pevný construction and is optimal when the
    payload truly is ternary.
    """
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError("masks must be (N, H, W)")
    diffs = np.diff(masks.astype(np.int16), axis=0)
    out = np.sign(diffs).astype(np.int8)
    return out.reshape(-1)
