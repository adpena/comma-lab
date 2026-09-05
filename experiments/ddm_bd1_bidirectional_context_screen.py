"""ddm_bd1 step B -- the $0 counting-model screen for BIDIRECTIONAL temporal context.

Question (MEASURED, scorer-free, CPU only): over the exact 600 decoded GT SegNet
argmax label fields the frontier archive codes (``inputs/tokens_null.u8``,
600 x 384 x 512 uint8, classes 0..4), how much does giving the coder the NEXT
plane at distance ``d`` reduce the conditional cost of the current plane, beyond
what the causal spatial neighbourhood + the PREVIOUS plane at distance ``d``
already supply?

The instrument is a plain adaptive-count model in ONE model class for both arms.
The arms differ ONLY by the presence of the next-plane taps, so the RELATIVE
saving is the transferable statistic; the ABSOLUTE bytes run far above the
trained mixer (dc1's 21-tap oracle coded at 144 KB against the mixer's 113 KB),
so absolutes are never transferred.

Two cost readings bracket the trained mixer:

* ``kt`` -- the exact sequential Krichevsky-Trofimov (Dirichlet-1/2) code length.
  It is order-independent, so it is computable in closed form from the final
  per-context histogram alone.  It charges the FULL per-context learning cost,
  which the bidirectional arm pays 5x or 25x more of (it has that many more
  contexts).  A trained mixer amortises the mapping into shared weights and pays
  nothing like this, so ``kt`` is a LOWER bound on the bidirectional gain.
* ``plugin`` -- the two-pass cost under the final (Dirichlet-1/2 smoothed)
  per-context distribution.  It ignores the learning cost entirely, so it is an
  UPPER bound on the gain.  It is also the only reading that attributes cost
  per-pair, which the pyramid assembly needs.

Decomposition mirrors ddm_hc1: per context the predicted class is the count
argmax; ``indicator`` bits code "was the prediction right?" and ``selection``
bits code which wrong class it was, over the wrong symbols only.

Nothing here trains, dispatches, or scores.  Axis of every number:
``[exact local byte/bit arithmetic, scorer-free]``; ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
from scipy.special import gammaln

REPO = Path(__file__).resolve().parents[1]
FIELD = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/inputs/tokens_null.u8"
)
STORE = Path("/Volumes/VertigoDataTier/pact/ddm_bd1_bidirectional_pyramid_context")

NUM_PAIRS = 600
HEIGHT = 384
WIDTH = 512
NUM_CLASSES = 5
PIXELS = HEIGHT * WIDTH

# MEASURED, cl2 LADDER_REPORT.json rung lambda_1p0 (the frontier archive).
SHIPPED_STREAM_BYTES = 113419
SHIPPED_MODEL_BYTES = 13466
SHIPPED_ARCHIVE_BYTES = 179982

LN2 = float(np.log(2.0))


class Bd1Error(RuntimeError):
    """Fail closed; never emit a partial row."""


# --------------------------------------------------------------------------
# context construction
# --------------------------------------------------------------------------


def _shift(plane: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Edge-replicated shift: value at (r+dy, c+dx), clamped into the frame."""
    rows = np.clip(np.arange(HEIGHT) + dy, 0, HEIGHT - 1)
    cols = np.clip(np.arange(WIDTH) + dx, 0, WIDTH - 1)
    return plane[np.ix_(rows, cols)]


def spatial6_code(plane: np.ndarray) -> np.ndarray:
    """Raster-causal 6-tap code W, NW, N, NE, WW, NN -> 0..15624 (int16).

    Ordering is chosen so that integer division recovers the shorter ladders:
    ``// 25`` gives the 4-tap (W, NW, N, NE) code and ``// 125`` the 3-tap
    (W, NW, N) code, so all three ladders share one precomputed array.
    """
    w = _shift(plane, 0, -1).astype(np.int16)
    nw = _shift(plane, -1, -1).astype(np.int16)
    n = _shift(plane, -1, 0).astype(np.int16)
    ne = _shift(plane, -1, 1).astype(np.int16)
    ww = _shift(plane, 0, -2).astype(np.int16)
    nn = _shift(plane, -2, 0).astype(np.int16)
    return ((((w * 5 + nw) * 5 + n) * 5 + ne) * 5 + ww) * 5 + nn


def plane25_code(plane: np.ndarray) -> np.ndarray:
    """Plane reduction to 25 states: (centre class, encroaching class).

    ``code = centre * 5 + m`` where ``m == 0`` when the 3x3 neighbourhood is
    uniform, else ``1 + rank`` of the most common NON-centre class among the 4
    classes that are not the centre (rank in ascending class index, so 0..3).
    This is the boundary-jitter signal: it says which class is pressing on this
    pixel, which is exactly what the "no" branch of the stream pays for.
    """
    counts = np.zeros((NUM_CLASSES, HEIGHT, WIDTH), dtype=np.int8)
    for k in range(NUM_CLASSES):
        member = (plane == k).astype(np.int8)
        acc = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += _shift(member, dy, dx)
        counts[k] = acc
    centre = plane.astype(np.int64)
    # Non-centre counts only: blank out the centre class row.
    non_centre = counts.astype(np.int16).copy()
    np.put_along_axis(non_centre, centre[None, :, :], -1, axis=0)
    best = non_centre.argmax(axis=0).astype(np.int64)
    best_count = non_centre.max(axis=0)
    uniform = best_count <= 0
    # rank of `best` among the four non-centre class indices, ascending
    rank = np.where(best > centre, best - 1, best)
    code = centre * 5 + np.where(uniform, 0, 1 + rank)
    return code.astype(np.uint8)


@dataclass(frozen=True)
class Ladder:
    name: str
    spatial_taps: int  # 3, 4 or 6
    plane_states: int  # 5 (centre only) or 25

    @property
    def spatial_states(self) -> int:
        return 5 ** self.spatial_taps


LADDERS = (
    Ladder("alpha_s4_p25", 4, 25),
    Ladder("beta_s6_p5", 6, 5),
    Ladder("gamma_s3_p25", 3, 25),
)


# --------------------------------------------------------------------------
# cost arithmetic
# --------------------------------------------------------------------------

_LG_HALF = float(gammaln(0.5))


def kt_bits_from_counts(counts: np.ndarray) -> float:
    """Exact sequential KT (Dirichlet-1/2) code length, in bits.

    ``counts`` is (contexts, K).  KT is exchangeable, so the sequential cost of
    every context's symbol sequence depends on its final histogram alone:

        -log P = lgamma(n + K/2) - lgamma(K/2)
                 - sum_k [ lgamma(n_k + 1/2) - lgamma(1/2) ]
    """
    k = counts.shape[1]
    n = counts.sum(axis=1)
    live = n > 0
    if not live.any():
        return 0.0
    n = n[live].astype(np.float64)
    c = counts[live].astype(np.float64)
    total = gammaln(n + k / 2.0) - float(gammaln(k / 2.0))
    total -= (gammaln(c + 0.5) - _LG_HALF).sum(axis=1)
    return float(total.sum() / LN2)


def kt_split_bits(counts: np.ndarray) -> tuple[float, float]:
    """hc1-style split: (indicator bits, selection bits).

    Predicted class per context is the count argmax.  ``indicator`` is a binary
    KT over (right, wrong); ``selection`` is a 4-ary KT over the wrong symbols.
    Both arms are split identically, so the split is comparable even though the
    factorised model is not byte-identical to the 5-ary KT total.
    """
    n = counts.sum(axis=1)
    live = n > 0
    if not live.any():
        return 0.0, 0.0
    c = counts[live].astype(np.float64)
    n = n[live].astype(np.float64)
    best = c.max(axis=1)
    wrong = n - best
    ind = gammaln(n + 1.0) - float(gammaln(1.0))
    ind -= gammaln(best + 0.5) - _LG_HALF
    ind -= gammaln(wrong + 0.5) - _LG_HALF
    arg = c.argmax(axis=1)
    c_wrong = c.copy()
    np.put_along_axis(c_wrong, arg[:, None], 0.0, axis=1)
    sel = gammaln(wrong + 2.0) - float(gammaln(2.0))
    sel -= (gammaln(c_wrong + 0.5) - _LG_HALF).sum(axis=1)
    # The argmax entry was zeroed, so its term is lgamma(0.5)-lgamma(0.5) = 0 and
    # the sum over 5 entries equals the sum over the 4 wrong classes.  A context
    # with no wrong symbols costs exactly 0 selection bits; assert that rather
    # than trusting the float arithmetic to land on zero.
    sel = np.where(wrong > 0, sel, 0.0)
    return float(ind.sum() / LN2), float(sel.sum() / LN2)


def plugin_logp_table(counts: np.ndarray) -> np.ndarray:
    """-log2 p_hat table, shape (contexts, K), Dirichlet-1/2 smoothed."""
    k = counts.shape[1]
    n = counts.sum(axis=1, keepdims=True).astype(np.float64)
    p = (counts.astype(np.float64) + 0.5) / (n + k / 2.0)
    return (-np.log2(p)).astype(np.float32)


def plugin_yes_no_bits(counts: np.ndarray) -> tuple[float, float]:
    """Split the plug-in cost by outcome, exactly as ddm_hc1 splits the stream.

    ``yes`` is the confirmation cost paid on symbols that matched the context's
    predicted (count-argmax) class; ``no`` is everything paid on symbols that
    did not.  hc1 MEASURED the shipped stream as 34,674 B yes / 76,601 B no, so
    these two are the directly comparable quantities.
    """
    table = plugin_logp_table(counts).astype(np.float64)
    live = counts.sum(axis=1) > 0
    if not live.any():
        return 0.0, 0.0
    c = counts[live].astype(np.float64)
    t = table[live]
    arg = c.argmax(axis=1)
    rows = np.arange(c.shape[0])
    yes = float((c[rows, arg] * t[rows, arg]).sum())
    total = float((c * t).sum())
    return yes, total - yes


# --------------------------------------------------------------------------
# the screen
# --------------------------------------------------------------------------


@dataclass
class Precomputed:
    field: np.ndarray  # (600, 384, 512) uint8
    spat6: np.ndarray  # (600, 384, 512) int16
    p25: np.ndarray  # (600, 384, 512) uint8


def precompute(field: np.ndarray) -> Precomputed:
    spat6 = np.empty((NUM_PAIRS, HEIGHT, WIDTH), dtype=np.int16)
    p25 = np.empty((NUM_PAIRS, HEIGHT, WIDTH), dtype=np.uint8)
    for i in range(NUM_PAIRS):
        plane = field[i]
        spat6[i] = spatial6_code(plane)
        p25[i] = plane25_code(plane)
    return Precomputed(field=field, spat6=spat6, p25=p25)


def _spatial_for(pre: Precomputed, idx: int, taps: int) -> np.ndarray:
    code = pre.spat6[idx].astype(np.int32)
    if taps == 6:
        return code
    if taps == 4:
        return code // 25
    if taps == 3:
        return code // 125
    raise Bd1Error(f"unsupported spatial tap count {taps}")


def _plane_for(pre: Precomputed, idx: int, states: int) -> np.ndarray:
    if states == 25:
        return pre.p25[idx].astype(np.int32)
    if states == 5:
        return pre.field[idx].astype(np.int32)
    raise Bd1Error(f"unsupported plane state count {states}")


def context_codes(
    pre: Precomputed, ladder: Ladder, idx: int, distance: int, bidirectional: bool
) -> np.ndarray:
    """Flat context index for one pair.  Caller guarantees the neighbours exist."""
    spat = _spatial_for(pre, idx, ladder.spatial_taps)
    prev = _plane_for(pre, idx - distance, ladder.plane_states)
    code = spat * ladder.plane_states + prev
    if bidirectional:
        nxt = _plane_for(pre, idx + distance, ladder.plane_states)
        code = code * ladder.plane_states + nxt
    return code


def n_contexts(ladder: Ladder, bidirectional: bool) -> int:
    n = ladder.spatial_states * ladder.plane_states
    if bidirectional:
        n *= ladder.plane_states
    return n


def accumulate_counts(
    pre: Precomputed,
    ladder: Ladder,
    pairs: list[int],
    distance: int,
    bidirectional: bool,
) -> np.ndarray:
    nctx = n_contexts(ladder, bidirectional)
    bins = nctx * NUM_CLASSES
    flat = np.zeros(bins, dtype=np.int64)
    chunk = 8  # amortise the minlength allocation across several pairs
    buffer: list[np.ndarray] = []
    for idx in pairs:
        code = context_codes(pre, ladder, idx, distance, bidirectional)
        buffer.append(code.ravel() * NUM_CLASSES + pre.field[idx].ravel().astype(np.int32))
        if len(buffer) == chunk:
            flat += np.bincount(np.concatenate(buffer), minlength=bins)
            buffer = []
    if buffer:
        flat += np.bincount(np.concatenate(buffer), minlength=bins)
    return flat.reshape(nctx, NUM_CLASSES)


def plugin_bits_per_pair(
    pre: Precomputed,
    ladder: Ladder,
    pairs: list[int],
    distance: int,
    bidirectional: bool,
    counts: np.ndarray,
) -> dict[int, float]:
    table = plugin_logp_table(counts).ravel()
    out: dict[int, float] = {}
    for idx in pairs:
        code = context_codes(pre, ladder, idx, distance, bidirectional)
        sym = code.ravel() * NUM_CLASSES + pre.field[idx].ravel().astype(np.int32)
        out[idx] = float(table[sym].sum(dtype=np.float64))
    return out


@dataclass
class ArmResult:
    ladder: str
    distance: int
    bidirectional: bool
    pairs: int
    symbols: int
    contexts_total: int
    contexts_used: int
    kt_bits: float
    plugin_bits: float
    indicator_bits: float
    selection_bits: float
    plugin_yes_bits: float
    plugin_no_bits: float

    @property
    def kt_bytes(self) -> float:
        return self.kt_bits / 8.0

    @property
    def plugin_bytes(self) -> float:
        return self.plugin_bits / 8.0


def measure_arm(
    pre: Precomputed,
    ladder: Ladder,
    pairs: list[int],
    distance: int,
    bidirectional: bool,
    want_per_pair: bool = False,
) -> tuple[ArmResult, dict[int, float] | None]:
    counts = accumulate_counts(pre, ladder, pairs, distance, bidirectional)
    kt = kt_bits_from_counts(counts)
    ind, sel = kt_split_bits(counts)
    per_pair = None
    if want_per_pair:
        per_pair = plugin_bits_per_pair(
            pre, ladder, pairs, distance, bidirectional, counts
        )
        plugin = float(sum(per_pair.values()))
    else:
        table = plugin_logp_table(counts)
        plugin = float((table.astype(np.float64) * counts).sum())
    yes_bits, no_bits = plugin_yes_no_bits(counts)
    used = int((counts.sum(axis=1) > 0).sum())
    result = ArmResult(
        ladder=ladder.name,
        distance=distance,
        bidirectional=bidirectional,
        pairs=len(pairs),
        symbols=len(pairs) * PIXELS,
        contexts_total=counts.shape[0],
        contexts_used=used,
        kt_bits=kt,
        plugin_bits=plugin,
        indicator_bits=ind,
        selection_bits=sel,
        plugin_yes_bits=yes_bits,
        plugin_no_bits=no_bits,
    )
    return result, per_pair


# --------------------------------------------------------------------------
# pyramid layout
# --------------------------------------------------------------------------


def pyramid_levels(gop: int = 8) -> dict[str, dict]:
    """B-pyramid over 600 pairs: keyframes every ``gop``, then halving distances.

    Keyframes sit at ``i = 0 mod gop`` and are coded P-only from distance ``gop``.
    The level at distance ``dd`` covers ``i = dd mod 2*dd`` and is coded from
    BOTH neighbours at +/- dd; every one of those references belongs to a
    strictly coarser level, so the receiver's decode order (keyframes, then
    gop/2, gop/4, ... , 1) is always satisfiable.  A pair whose reference is
    missing at the sequence edge falls back to the shipped causal-d1 cost.
    """
    if gop < 2 or gop & (gop - 1):
        raise Bd1Error(f"gop must be a power of two >= 2, got {gop}")
    levels: dict[str, dict] = {
        f"level{gop}_keyframe": {
            "pairs": [i for i in range(0, NUM_PAIRS, gop) if i - gop >= 0],
            "distance": gop,
            "bidirectional": False,
        }
    }
    dd = gop // 2
    while dd >= 1:
        levels[f"level{dd}"] = {
            "pairs": [
                i
                for i in range(dd, NUM_PAIRS, 2 * dd)
                if i - dd >= 0 and i + dd < NUM_PAIRS
            ],
            "distance": dd,
            "bidirectional": True,
        }
        dd //= 2
    covered: set[int] = set()
    for spec in levels.values():
        covered |= set(spec["pairs"])
    # Pair 0 has no predecessor under ANY scheme (the shipped coder feeds it a
    # zero previous plane), so it is charged identically by both schemes and is
    # excluded from numerator and denominator alike.
    levels["fallback_causal_d1"] = {
        "pairs": sorted(set(range(1, NUM_PAIRS)) - covered),
        "distance": 1,
        "bidirectional": False,
    }
    return levels


def assemble_pyramid(
    pre: Precomputed,
    ladder: Ladder,
    levels: dict[str, dict],
    cache: dict[tuple[int, bool], dict[int, float]],
    include_arms: bool = True,
) -> dict:
    """Price one pyramid layout against the shipped causal-d1 coder.

    Each (distance, arm) pair is fitted ONCE over every eligible pair -- a
    trained mixer amortises the context mapping into shared weights across all
    600 pairs, so the global fit is the reading that transfers -- and per-pair
    plug-in attribution then partitions that cost by pyramid level.  ``cache``
    is keyed by (distance, bidirectional) so a GOP sweep refits nothing.
    """
    arm_rows: dict[str, dict] = {}

    def arm(distance: int, bidi: bool) -> dict[int, float]:
        key = (distance, bidi)
        if key not in cache:
            lo = distance
            hi = NUM_PAIRS - distance if bidi else NUM_PAIRS
            res, per_pair = measure_arm(
                pre, ladder, list(range(lo, hi)), distance, bidi, want_per_pair=True
            )
            cache[key] = per_pair or {}
            arm_rows[f"d{distance}_{'PB' if bidi else 'P'}"] = asdict(res)
        return cache[key]

    baseline = arm(1, False)
    level_rows: dict[str, dict] = {}
    pyramid_bits = 0.0
    baseline_bits = 0.0
    for name, spec in levels.items():
        pairs = spec["pairs"]
        if not pairs:
            level_rows[name] = {"pairs": 0}
            continue
        values = arm(spec["distance"], spec["bidirectional"])
        missing = [p for p in pairs if p not in values or p not in baseline]
        if missing:
            raise Bd1Error(f"level {name} lost pairs {missing[:5]} in attribution")
        lvl_bits = sum(values[p] for p in pairs)
        base_bits = sum(baseline[p] for p in pairs)
        pyramid_bits += lvl_bits
        baseline_bits += base_bits
        level_rows[name] = {
            "pairs": len(pairs),
            "distance": spec["distance"],
            "bidirectional": spec["bidirectional"],
            "plugin_bytes_level": lvl_bits / 8.0,
            "plugin_bytes_causal_d1_same_pairs": base_bits / 8.0,
            "ratio_level_over_causal_d1": lvl_bits / base_bits,
        }
    net_ratio = pyramid_bits / baseline_bits
    out = {
        "levels": level_rows,
        "pyramid_plugin_bytes": pyramid_bits / 8.0,
        "baseline_causal_d1_plugin_bytes": baseline_bits / 8.0,
        "net_ratio_pyramid_over_causal_d1": net_ratio,
        "predicted_stream_bytes_at_shipped_113419": SHIPPED_STREAM_BYTES * net_ratio,
        "predicted_saving_bytes": SHIPPED_STREAM_BYTES * (1.0 - net_ratio),
        "predicted_saving_fraction": 1.0 - net_ratio,
    }
    if include_arms:
        out["arms"] = arm_rows
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def load_field(path: Path) -> tuple[np.ndarray, str]:
    raw = path.read_bytes()
    if len(raw) != NUM_PAIRS * PIXELS:
        raise Bd1Error(
            f"field {path} is {len(raw)} B, expected {NUM_PAIRS * PIXELS} B"
        )
    digest = hashlib.sha256(raw).hexdigest()
    field = np.frombuffer(raw, dtype=np.uint8).reshape(NUM_PAIRS, HEIGHT, WIDTH)
    if int(field.max()) >= NUM_CLASSES:
        raise Bd1Error("field carries a class index outside 0..4")
    return np.ascontiguousarray(field), digest


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    field, digest = load_field(Path(args.field))
    pre = precompute(field)
    precompute_seconds = time.time() - started

    ladders = [lad for lad in LADDERS if lad.name in args.ladders]
    if not ladders:
        raise Bd1Error("no ladder selected")

    report: dict = {
        "schema": "ddm_bd1_bidirectional_context_screen.v1",
        "axis": "[exact local bit/byte arithmetic, scorer-free]",
        "score_claim": False,
        "field_path": str(args.field),
        "field_sha256": digest,
        "field_shape": [NUM_PAIRS, HEIGHT, WIDTH],
        "shipped": {
            "stream_bytes": SHIPPED_STREAM_BYTES,
            "model_bytes": SHIPPED_MODEL_BYTES,
            "archive_bytes": SHIPPED_ARCHIVE_BYTES,
        },
        "precompute_seconds": precompute_seconds,
        "block1_common_window": {},
        "block2_pyramid": {},
    }

    # ---- BLOCK 1: the charter's per-distance table on one common window ----
    # The window is the widest index range for which EVERY measured distance has
    # both neighbours, so all rows are the same pairs and are directly comparable.
    distances = (1, 2, 4, 8, 16, 32)
    widest = max(distances)
    window = list(range(widest, NUM_PAIRS - widest))
    for ladder in ladders:
        rows = {}
        for distance in distances:
            for bidi in (False, True):
                res, _ = measure_arm(pre, ladder, window, distance, bidi)
                rows[f"d{distance}_{'PB' if bidi else 'P'}"] = asdict(res)
        table = {}
        for distance in distances:
            a = rows[f"d{distance}_P"]
            b = rows[f"d{distance}_PB"]
            table[f"d{distance}"] = {
                "kt_bytes_P": a["kt_bits"] / 8.0,
                "kt_bytes_PB": b["kt_bits"] / 8.0,
                "kt_ratio_PB_over_P": b["kt_bits"] / a["kt_bits"],
                "plugin_bytes_P": a["plugin_bits"] / 8.0,
                "plugin_bytes_PB": b["plugin_bits"] / 8.0,
                "plugin_ratio_PB_over_P": b["plugin_bits"] / a["plugin_bits"],
                "indicator_bytes_P": a["indicator_bits"] / 8.0,
                "indicator_bytes_PB": b["indicator_bits"] / 8.0,
                "indicator_ratio": b["indicator_bits"] / a["indicator_bits"],
                "selection_bytes_P": a["selection_bits"] / 8.0,
                "selection_bytes_PB": b["selection_bits"] / 8.0,
                "selection_ratio": b["selection_bits"] / a["selection_bits"],
                "plugin_yes_bytes_P": a["plugin_yes_bits"] / 8.0,
                "plugin_yes_bytes_PB": b["plugin_yes_bits"] / 8.0,
                "plugin_yes_ratio": b["plugin_yes_bits"] / a["plugin_yes_bits"],
                "plugin_no_bytes_P": a["plugin_no_bits"] / 8.0,
                "plugin_no_bytes_PB": b["plugin_no_bits"] / 8.0,
                "plugin_no_ratio": b["plugin_no_bits"] / a["plugin_no_bits"],
                "contexts_used_P": a["contexts_used"],
                "contexts_used_PB": b["contexts_used"],
            }
        report["block1_common_window"][ladder.name] = {
            "pairs": len(window),
            "raw": rows,
            "table": table,
        }

    # ---- BLOCK 2: pyramid assembly at the charter's GOP = 8 ----
    levels = pyramid_levels(8)
    for ladder in ladders:
        cache: dict[tuple[int, bool], dict[int, float]] = {}
        report["block2_pyramid"][ladder.name] = assemble_pyramid(
            pre, ladder, levels, cache
        )

    # ---- BLOCK 4: the GOP sweep.  F1 closes the FAMILY, so the family's BEST
    # member must be priced, not only the charter's GOP = 8 point. ----
    report["block4_gop_sweep"] = {}
    for ladder in ladders:
        cache = {}
        sweep = {}
        for gop in (2, 4, 8, 16, 32):
            sweep[f"gop{gop}"] = assemble_pyramid(
                pre, ladder, pyramid_levels(gop), cache, include_arms=False
            )
        best = min(sweep.items(), key=lambda kv: kv[1]["net_ratio_pyramid_over_causal_d1"])
        # The unattainable ceiling: every pair coded bidirectionally at distance 1.
        # No decode order can realise it (each pair would need both neighbours
        # already decoded), so it bounds the whole family from above.
        ceiling_pairs = list(range(1, NUM_PAIRS - 1))
        ceil_b, ceil_b_pp = measure_arm(pre, ladder, ceiling_pairs, 1, True, True)
        ceil_a, ceil_a_pp = measure_arm(pre, ladder, ceiling_pairs, 1, False, True)
        ceil_ratio = sum(ceil_b_pp.values()) / sum(ceil_a_pp.values())
        report["block4_gop_sweep"][ladder.name] = {
            "sweep": sweep,
            "best_gop": best[0],
            "best_saving_fraction": best[1]["predicted_saving_fraction"],
            "best_saving_bytes": best[1]["predicted_saving_bytes"],
            "unattainable_all_pairs_d1_bidirectional": {
                "pairs": len(ceiling_pairs),
                "plugin_bytes_bidirectional": ceil_b.plugin_bytes,
                "plugin_bytes_causal_d1": ceil_a.plugin_bytes,
                "net_ratio": ceil_ratio,
                "saving_fraction": 1.0 - ceil_ratio,
                "saving_bytes": SHIPPED_STREAM_BYTES * (1.0 - ceil_ratio),
            },
        }

    # ---- BLOCK 3: conservative per-level fit (KT, counts fit on that level only)
    # This charges the bidirectional arm the FULL per-context learning cost on a
    # small sample, which a weight-sharing mixer never pays.  It is the lower
    # bracket on the gain; block 2 is the upper.
    report["block3_perlevel_fit_kt"] = {}
    for ladder in ladders:
        rows = {}
        pyramid_bits = 0.0
        baseline_bits = 0.0
        for name, spec in levels.items():
            pairs = spec["pairs"]
            if not pairs:
                rows[name] = {"pairs": 0}
                continue
            lvl, _ = measure_arm(
                pre, ladder, pairs, spec["distance"], spec["bidirectional"]
            )
            base, _ = measure_arm(pre, ladder, pairs, 1, False)
            pyramid_bits += lvl.kt_bits
            baseline_bits += base.kt_bits
            rows[name] = {
                "pairs": len(pairs),
                "distance": spec["distance"],
                "bidirectional": spec["bidirectional"],
                "kt_bytes_level": lvl.kt_bytes,
                "kt_bytes_causal_d1_same_pairs": base.kt_bytes,
                "ratio_level_over_causal_d1": lvl.kt_bits / base.kt_bits,
                "indicator_ratio": lvl.indicator_bits / base.indicator_bits,
                "selection_ratio": lvl.selection_bits / base.selection_bits,
                "contexts_used_level": lvl.contexts_used,
                "contexts_used_causal_d1": base.contexts_used,
            }
        net_ratio = pyramid_bits / baseline_bits
        rows["_net"] = {
            "pyramid_kt_bytes": pyramid_bits / 8.0,
            "baseline_causal_d1_kt_bytes": baseline_bits / 8.0,
            "net_ratio_pyramid_over_causal_d1": net_ratio,
            "predicted_stream_bytes_at_shipped_113419": SHIPPED_STREAM_BYTES * net_ratio,
            "predicted_saving_bytes": SHIPPED_STREAM_BYTES * (1.0 - net_ratio),
            "predicted_saving_fraction": 1.0 - net_ratio,
        }
        report["block3_perlevel_fit_kt"][ladder.name] = rows

    report["elapsed_seconds"] = time.time() - started
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", default=str(FIELD))
    parser.add_argument(
        "--ladders",
        nargs="+",
        default=[lad.name for lad in LADDERS],
        choices=[lad.name for lad in LADDERS],
    )
    parser.add_argument("--output", default=str(STORE / "screen_report.json"))
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = run(args)
    payload = json.dumps(report, indent=2, sort_keys=True).encode()
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(out)
    print(
        json.dumps(
            {
                "output": str(out),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
