# SPDX-License-Identifier: MIT
"""Margin-conditional seg-repair residual coder + waterfill selector (lever D, task #72).

The seg term is ``100*d_seg`` where ``d_seg = mean[argmax S(F1) != L*]`` (the per-pixel argmax-flip
rate, verified ``upstream/modules.py``).  Fixing every flip drops the seg term by ``100*d_seg``; at the
naive unconditional position cost (``log2 C(N_grid, K) ~ 1.53 B/flip``) plus class cost that EXCEEDS the
``1.27 B/flip`` break-even (closed-spec §10), so a *naive* seg-repair sidecar is net-zero-or-negative on
RATE (the #51 DEFER).

THE LEVER (this module): the decoder regenerates the SegNet margin field FOR FREE from the renderer's
own output (it has the renderer -> recompute margins).  So the sidecar need not address flips over all
``N_grid`` pixels -- only over the decoder-KNOWN low-margin boundary set ``B = {p : m(p) < tau}``.  The
CONDITIONAL position cost is ``log2 C(|B|, K)``, much smaller than ``log2 C(N_grid, K)`` because the flips
concentrate in the fragile boundary band the decoder can localize without being told.  Plus the target
class costs ``H(target | margin-context)`` bits.  This coder MEASURES that realized conditional cost and
WATERFILLS the subset that (a) codes below ``1.27 B/flip`` AND (b) has positive exact-decode net value
(flips fixed minus collateral new-bad flips) -- because a frame-1 correction pays receptive-field
collateral (the #51 / #55 finding), so gross stored-flip count is NOT the value; NET is.

Two cost components (both real, both measured -- NO FAKE):
  1. POSITION (conditional): given the decoder-known boundary set ``B``, the flips are a K-subset of
     ``|B|`` candidates -> ``log2 C(|B|, K)`` bits (combinatorial set index, the tightest achievable).
     This is the ``MarginConditionalResidualCoder.position_bits`` -- it MUST shrink when the margin
     context concentrates the flips (tested: a context that does not concentrate -> no saving).
  2. CLASS: the target class per flip, costed at the conditional entropy
     ``H(target_class | margin-bin)`` (an arithmetic/empirical model) -- ``class_bits``.

The WATERFILL (``waterfill_select``): rank flips by descending (net_value / code_cost) and admit the
prefix whose marginal clears the ``1.27 B/flip`` water level AND has ``net_value > 0``.  This is the
closed-spec §10 KKT admission applied per flip, with the receptive-field collateral priced in.

NO FAKE:
  * ``conditional_position_bits`` returns ``log2 C(|B|, K)`` -- it is BELOW the unconditional
    ``log2 C(N_grid, K)`` exactly when ``|B| < N_grid`` (tested on a real-shaped boundary set).  A coder
    that ignores the margin (``B = all pixels``) does NOT beat the unconditional (tested).
  * ``class_bits_conditional`` returns the realized conditional entropy * K -- below the unconditional
    ``H(target) * K`` iff the margin context informs the class (tested).
  * ``waterfill_select`` admits only flips with ``net_value > 0`` and marginal cost ``< 1.27`` -- a
    select-all stub FAILS the discrimination test; a constant net-value of 0 admits nothing.
  * the sidecar ``encode_residual`` -> ``decode_residual`` round-trips the admitted (position, target)
    set bit-exactly (tested).
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass

import numpy as np

# ── The water level (closed-spec §10; same constants as region_merge.py) ─────
_D_RATE_DENOM = 37_545_489
_N_SCORED_PER_FRAME = 384 * 512  # 196,608
_N_FRAMES_SCORED = 600
_N_SCORED_TOTAL = _N_FRAMES_SCORED * _N_SCORED_PER_FRAME  # 117,964,800
_SEG_WEIGHT = 100.0
# per-flip score value (global mean) and break-even byte price
SEG_VALUE_PER_FLIP = _SEG_WEIGHT / _N_SCORED_TOTAL  # 8.48e-7 score/flip
BYTES_PER_SCORE = _D_RATE_DENOM / 25.0  # bytes per unit score (inverse of 25/D)
WATERLINE_BYTES_PER_FLIP = SEG_VALUE_PER_FLIP * BYTES_PER_SCORE  # == 1.27 B/flip
N_SEG_CLASSES = 5


def log2_choose(n: int, k: int) -> float:
    """``log2 C(n, k)`` in bits (the information content of a k-subset of n)."""
    if k <= 0:
        return 0.0
    if k > n:
        raise ValueError(f"cannot choose {k} from {n}")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2.0)


def boundary_set_from_margin(margin_field_flat: np.ndarray, tau: float) -> np.ndarray:
    """The decoder-known boundary set ``B = {p : m(p) < tau}`` as a boolean flat mask.

    The decoder regenerates ``margin_field_flat`` from its own render; ``tau`` is a fixed protocol
    constant both sides agree on, so ``B`` is reproducible at decode with ZERO stored bytes.
    """
    m = np.asarray(margin_field_flat, dtype=np.float64)
    if m.ndim != 1:
        raise ValueError(f"margin field must be flat (N,); got {m.shape}")
    if tau <= 0:
        raise ValueError("tau must be > 0")
    return m < tau


def conditional_position_bits(boundary_size: int, k_in: int, n_grid: int, k_out: int) -> float:
    """Position bits to address K flips given the decoder-known boundary set.

    ``k_in`` flips fall inside ``B`` (|B| = ``boundary_size``) -> ``log2 C(|B|, k_in)``.
    ``k_out`` flips fall outside ``B`` -> addressed among the complement ``log2 C(n_grid - |B|, k_out)``.
    Returns the total conditional position bits (the tightest combinatorial set-index cost).
    """
    if boundary_size < k_in:
        raise ValueError("boundary_size < k_in (flips inside B exceed B)")
    bits_in = log2_choose(boundary_size, k_in)
    bits_out = log2_choose(n_grid - boundary_size, k_out)
    return bits_in + bits_out


def unconditional_position_bits(n_grid: int, k: int) -> float:
    """The naive position floor ``log2 C(n_grid, K)`` (the #51 number, no margin context)."""
    return log2_choose(n_grid, k)


def class_bits_conditional(target_classes: np.ndarray, margin_bin: np.ndarray | None = None) -> float:
    """Total class bits ``sum H(target | margin-bin) * count`` (realized conditional entropy).

    With no ``margin_bin`` the unconditional ``H(target) * K`` is returned.  The conditional form is
    ``<= `` the unconditional (mutual information is non-negative) -- below it iff the margin informs the
    class.  NO FAKE: returns the measured empirical entropy, not a constant.
    """
    t = np.asarray(target_classes, dtype=np.int64)
    K = len(t)
    if K == 0:
        return 0.0
    if margin_bin is None:
        v, n = np.unique(t, return_counts=True)
        p = n / n.sum()
        return K * float(-np.sum(p * np.log2(p)))
    b = np.asarray(margin_bin, dtype=np.int64)
    if len(b) != K:
        raise ValueError("margin_bin length != target_classes length")
    total = 0.0
    for ub in np.unique(b):
        sel = b == ub
        tt = t[sel]
        v, n = np.unique(tt, return_counts=True)
        p = n / n.sum()
        total += float(-np.sum(p * np.log2(p))) * sel.sum()
    return total


@dataclass(frozen=True)
class ResidualCodeCost:
    """The measured conditional code cost for a flip-set (per-flip + totals)."""

    n_flips: int
    boundary_size: int
    k_in_boundary: int
    k_out_boundary: int
    position_bits: float
    class_bits: float
    unconditional_position_bits: float
    tau: float

    @property
    def total_bits(self) -> float:
        return self.position_bits + self.class_bits

    @property
    def bytes_per_flip(self) -> float:
        return (self.total_bits / 8.0) / self.n_flips if self.n_flips else 0.0

    @property
    def unconditional_bytes_per_flip(self) -> float:
        # unconditional position + same class cost (class context is orthogonal)
        if not self.n_flips:
            return 0.0
        return ((self.unconditional_position_bits + self.class_bits) / 8.0) / self.n_flips

    @property
    def beats_waterline(self) -> bool:
        return self.bytes_per_flip < WATERLINE_BYTES_PER_FLIP

    @property
    def conditional_saving_bytes_per_flip(self) -> float:
        return self.unconditional_bytes_per_flip - self.bytes_per_flip


def measure_code_cost(
    margin_field_flat: np.ndarray,
    flip_idx: np.ndarray,
    target_classes: np.ndarray,
    *,
    tau: float = 0.5,
    margin_bins: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 1e9),
) -> ResidualCodeCost:
    """Measure the realized conditional code cost of a flip-set under the decoder's free margin field.

    NO FAKE: ``boundary_size`` is the real count of low-margin pixels the decoder regenerates;
    ``position_bits`` is the combinatorial set index conditioned on that boundary; ``class_bits`` is the
    realized conditional entropy.  All shrink (vs unconditional) iff the margin context concentrates.
    """
    m = np.asarray(margin_field_flat, dtype=np.float64)
    n_grid = int(m.size)
    idx = np.asarray(flip_idx, dtype=np.int64)
    cls = np.asarray(target_classes, dtype=np.int64)
    if len(idx) != len(cls):
        raise ValueError("flip_idx and target_classes length mismatch")
    K = len(idx)
    B = boundary_set_from_margin(m, tau)
    boundary_size = int(B.sum())
    in_B = B[idx]
    k_in = int(in_B.sum())
    k_out = K - k_in
    pos_bits = conditional_position_bits(boundary_size, k_in, n_grid, k_out)
    uncond_pos = unconditional_position_bits(n_grid, K)
    bin_idx = (np.digitize(m[idx], np.asarray(margin_bins)) - 1).clip(0, len(margin_bins) - 2)
    cls_bits = class_bits_conditional(cls, bin_idx)
    return ResidualCodeCost(
        n_flips=K, boundary_size=boundary_size, k_in_boundary=k_in, k_out_boundary=k_out,
        position_bits=pos_bits, class_bits=cls_bits, unconditional_position_bits=uncond_pos, tau=tau,
    )


@dataclass(frozen=True)
class WaterfillResult:
    """The waterfill-admitted subset + its economics."""

    admitted_local_indices: np.ndarray  # indices INTO the candidate flip arrays
    n_candidates: int
    n_admitted: int
    admitted_net_value_flips: int
    admitted_code_bytes: float
    marginal_bytes_per_net_flip: float

    @property
    def admission_fraction(self) -> float:
        return self.n_admitted / self.n_candidates if self.n_candidates else 0.0

    @property
    def net_score_delta(self) -> float:
        """Exact ΔS contribution of the admitted set: -(seg drop) + rate.

        seg drop = net_value_flips * SEG_VALUE_PER_FLIP (positive = good = lowers score).
        rate cost = code_bytes * (25 / D) (positive = raises score).
        Returns ΔS (negative = improvement).
        """
        seg_drop = self.admitted_net_value_flips * SEG_VALUE_PER_FLIP
        rate_cost = self.admitted_code_bytes * (25.0 / _D_RATE_DENOM)
        return -seg_drop + rate_cost


def waterfill_select(
    net_value_per_flip: np.ndarray,
    code_bytes_per_flip: np.ndarray,
    *,
    waterline: float = WATERLINE_BYTES_PER_FLIP,
) -> WaterfillResult:
    """Admit flips whose per-flip code cost clears the water level AND whose net value is positive.

    The closed-spec §10 KKT admission, per flip, with the receptive-field collateral priced into
    ``net_value_per_flip`` (flips fixed minus new-bad created -- can be <= 0).  A flip is admitted iff
    ``net_value > 0`` AND ``code_bytes / net_value < waterline`` (its marginal byte price per NET fixed
    flip is below the break-even).  NO FAKE: a select-all stub fails (it would admit net<=0 flips); a
    constant net=0 admits nothing.
    """
    nv = np.asarray(net_value_per_flip, dtype=np.float64)
    cb = np.asarray(code_bytes_per_flip, dtype=np.float64)
    if nv.shape != cb.shape:
        raise ValueError("net_value and code_bytes shape mismatch")
    n = len(nv)
    # admit iff net positive AND marginal byte price per net flip below the waterline
    positive = nv > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        marginal = np.where(positive, cb / np.maximum(nv, 1e-12), np.inf)
    admit = positive & (marginal < waterline)
    admitted = np.where(admit)[0].astype(np.int64)
    net_val = int(nv[admit].sum()) if admit.any() else 0
    code_b = float(cb[admit].sum()) if admit.any() else 0.0
    marg = (code_b / net_val) if net_val > 0 else float("inf")
    return WaterfillResult(
        admitted_local_indices=admitted, n_candidates=n, n_admitted=int(admit.sum()),
        admitted_net_value_flips=net_val, admitted_code_bytes=code_b,
        marginal_bytes_per_net_flip=marg,
    )


# ── The reversible sidecar (the admitted residual the inflate runtime consumes) ──
_SIDECAR_MAGIC = b"MCR1"  # margin-conditional residual v1


def encode_residual(
    pair_index: int,
    admitted_flip_idx: np.ndarray,
    admitted_target_cls: np.ndarray,
    *,
    grid_h: int = 384,
    grid_w: int = 512,
) -> bytes:
    """Encode the admitted (position, target-class) residual for one pair into a reversible blob.

    Layout: MAGIC | u32 pair_index | u32 grid_h | u32 grid_w | u32 K | zlib(delta-coded u32 idx)
            | zlib(u8 target classes).  The position deltas + zlib approximate the conditional coder's
    realized cost; the EXACT economics are measured by ``measure_code_cost`` (this is the byte-closed
    realization).  NO FAKE: ``decode_residual`` recovers (idx, cls) bit-exactly (tested).
    """
    idx = np.asarray(admitted_flip_idx, dtype=np.int64)
    cls = np.asarray(admitted_target_cls, dtype=np.int64)
    if len(idx) != len(cls):
        raise ValueError("idx/cls length mismatch")
    if len(idx) and (idx.min() < 0 or idx.max() >= grid_h * grid_w):
        raise ValueError("flip idx out of grid range")
    if len(cls) and (cls.min() < 0 or cls.max() >= N_SEG_CLASSES):
        raise ValueError("target class out of range")
    order = np.argsort(idx)
    idx_sorted = idx[order]
    cls_sorted = cls[order]
    deltas = np.diff(np.concatenate([[0], idx_sorted])).astype(np.uint32) if len(idx_sorted) else np.zeros(0, np.uint32)
    idx_blob = zlib.compress(deltas.tobytes(order="C"), 9)
    cls_blob = zlib.compress(cls_sorted.astype(np.uint8).tobytes(order="C"), 9)
    head = _SIDECAR_MAGIC + struct.pack("<IIII", pair_index, grid_h, grid_w, len(idx_sorted))
    return head + struct.pack("<I", len(idx_blob)) + idx_blob + struct.pack("<I", len(cls_blob)) + cls_blob


def decode_residual(blob: bytes) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    """Decode -> (pair_index, grid_h, grid_w, flip_idx, target_cls).  Bit-exact inverse."""
    if blob[:4] != _SIDECAR_MAGIC:
        raise ValueError("bad MCR1 magic")
    pair_index, grid_h, grid_w, K = struct.unpack("<IIII", blob[4:20])
    off = 20
    (idx_len,) = struct.unpack("<I", blob[off:off + 4])
    off += 4
    deltas = np.frombuffer(zlib.decompress(blob[off:off + idx_len]), dtype=np.uint32)
    off += idx_len
    (cls_len,) = struct.unpack("<I", blob[off:off + 4])
    off += 4
    cls = np.frombuffer(zlib.decompress(blob[off:off + cls_len]), dtype=np.uint8).astype(np.int64)
    off += cls_len
    idx = np.cumsum(deltas.astype(np.int64)) if K else np.zeros(0, np.int64)
    return int(pair_index), int(grid_h), int(grid_w), idx, cls
