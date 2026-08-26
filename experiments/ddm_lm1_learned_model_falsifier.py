#!/usr/bin/env python3
"""ddm_lm1 -- the ddm_no1 row-2 falsifier: can a W-byte LEARNED PARAMETRIC model
replace the 13,515 B HPAC network on the dx2-lineage token stream, losslessly?

WHAT THIS MEASURES
------------------
The shipped DX2 token subsystem is two counted objects:

    HPAC network blob (Brotli-coded)   13,515 B      -- the probability model
    RC64 token stream                 113,777 B      -- 117,964,800 five-class symbols
    ------------------------------------------------
    counted token subsystem           127,292 B

Row 2 asks whether a W-byte learned parametric model can REPLACE the network and
recode the same field for fewer total bytes.  Because the field is coded
losslessly and bit-identically, d_seg and d_pose are exactly unchanged: this is a
pure coding measurement, no scorer, no renderer, no authority row.

THE INSTRUMENT IS EXACT, NOT A SURROGATE
----------------------------------------
`ddm_rsf1` measured the `entropy` rate surrogate ANTI-correlated with shipped
bytes (rho = -0.7235), so no entropy proxy is admissible.  This module does not
use one.  It computes, for an explicit probability model over the exact decode
alphabet, the two code lengths that a real arithmetic coder emits:

  * L_KT   -- the exact Krichevsky-Trofimov PREQUENTIAL code length of a purely
              adaptive context model.  This is achievable with ZERO stored bytes
              (W = 0): the decoder rebuilds the identical counts from the prefix
              it has already decoded, so under contest rule 118 the model is
              generic algorithm in inflate.py and only the stream is counted.
  * H_hind -- the empirical conditional entropy Sum_c Sum_t N_ct log2(N_c/N_ct).
              This is the HINDSIGHT-IDEAL: the code length an oracle with that
              context, unlimited parameters, zero generalisation gap and ZERO
              model cost would achieve.  It is a strict LOWER BOUND on what any
              model using that context can reach, so it is the closure tool --
              same instrument `ddm_cx3` used for the named-summary family.

`ddm_tb2` measured that the shipped RC64 coder's physical bytes exceed the sum of
its own per-position code lengths by 6.719391 bits = 0.839924 B on this exact
910,209.280609-bit stream.  Code length therefore IS the real byte number here to
better than 1 B, and `--real-coder` runs an actual range coder end-to-end
(encode -> decode -> byte-identity assert) as an independent control.

DECODE CAUSALITY (read at source, not assumed)
----------------------------------------------
`cpr1/inflate.py:33` HPAC_PATCH = 64, HPAC_DELTA = 2; `group_masks` iterates
`(1 + DELTA) * PATCH - DELTA = 190` groups with grid `column + DELTA * row`, and
`hpac_integer.patch_group_mask` keeps a neighbour iff
`offset = dx + DELTA*dy < 0`.  So a position at patch-local (r, c) has group
`g = c + 2r`, and the causally-available set of the CURRENT frame is every
position whose GLOBAL group index is strictly smaller.  Group order is global
across all 6x8 patches, so cross-patch neighbours in earlier groups are legal --
information the shipped network is structurally blind to, because it convolves
inside zero-padded patches.  Every previous frame is fully available.

BAR (from `.omx/research/ddm_no1_new_object_derivation_20260826.md`)
-------------------------------------------------------------------
Old total 127,292 B.  New total = stream' + W.  Therefore

    break-even (net > 0)        stream' <  127,292 - W
    demand-closing (net >= D)   stream' <= 127,292 - W - D,  D = 42,228 B

no1's r-table (29.6% / 34.0% / 42.8% at W = 5K/10K/20K) is the DEMAND-CLOSING bar:
r >= (42,228 + W - 13,515) / 113,777 rearranges to exactly stream' <= 85,064 - W.

no1's ORIGINAL prose bar read `113,777 - 13,515 + W`, inconsistent with its own
r-table by `27,030 - 2W` (stricter below W = 13,515, looser above).  ddm_lm1 raised
it, MAIN verified it independently at source, and the correction landed in commit
4257fa1006 (three in-place fixes plus an append-only CORRECTION section).  Cite
no1's CORRECTION, not the original prose.  This module reports every measured
stream against BOTH bars explicitly so no reader inherits either reading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Pins.  Every one of these is re-verified at run time; none is trusted as prose.
# ---------------------------------------------------------------------------

FIELD_DIR = Path(
    "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/retained/fields"
)
TOKEN_FIELD = "decoded_tokens_instrumented.u8"
COST_FIELD = "position_rc64_frequency_cost_bits.f64le.bin"

TOKEN_FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
TOKEN_FIELD_BYTES = 117_964_800
COST_FIELD_BITS = 910_209.280609_0603  # ddm_tb2 exact reconciliation

SHIPPED_STREAM_BYTES = 113_777  # physical RC64 suffix
SHIPPED_MODEL_BYTES = 13_515  # counted Brotli-coded HPAC blob
SHIPPED_TOTAL_BYTES = SHIPPED_STREAM_BYTES + SHIPPED_MODEL_BYTES  # 127,292
DEMAND_BYTES = 42_228  # ddm_no1 sub-0.12 gap at fixed distortion
S_PER_BYTE = 6.658590e-07  # ddm_eu2 rate exchange

H, W_FRAME, N_FRAMES, N_CLASS = 384, 512, 600, 5
HPAC_PATCH, HPAC_DELTA = 64, 2
N_GROUPS = (1 + HPAC_DELTA) * HPAC_PATCH - HPAC_DELTA  # 190
UNK = 5  # sentinel: not causally available / out of bounds
ALPHABET = 6  # 5 classes + UNK

# Causal neighbour ladder.  ("cur", dy, dx) is legal iff dx + 2*dy < 0 AND the
# neighbour's GLOBAL group index is smaller (checked per position).  ("prv", ...)
# reads frame f-1, ("prv2", ...) reads frame f-2; both are fully decoded.
LADDER: tuple[tuple[str, int, int], ...] = (
    ("prv", 0, 0),
    ("cur", 0, -1),
    ("cur", -1, 0),
    ("cur", -1, -1),
    ("cur", -1, 1),
    ("prv", -1, 0),
    ("prv", 0, -1),
    ("prv", 1, 0),
    ("prv", 0, 1),
    ("cur", 0, -2),
    ("cur", -2, 0),
    ("prv", -1, -1),
    ("prv", -1, 1),
    ("prv", 1, -1),
    ("prv", 1, 1),
    ("cur", 1, -3),
    ("prv2", 0, 0),
)
# Rungs at which the ladder is scored (number of leading features used).
DEFAULT_RUNGS: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 13, 15, 17)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance is best-effort, never fatal
        return "unknown"


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Field loading + pin verification
# ---------------------------------------------------------------------------


@dataclass
class Field:
    tokens: np.ndarray  # (600, 384, 512) uint8, values 0..4
    cost_bits: np.ndarray  # (600, 384, 512) float64, shipped per-position cost
    token_sha256: str
    shipped_bits_total: float
    class_counts: dict[int, int] = dc_field(default_factory=dict)


def load_field(field_dir: Path, verify_sha: bool = True) -> Field:
    token_path = field_dir / TOKEN_FIELD
    cost_path = field_dir / COST_FIELD
    for path in (token_path, cost_path):
        if not path.exists():
            raise SystemExit(f"missing retained field: {path}")

    sha = sha256_file(token_path) if verify_sha else "skipped"
    if verify_sha and sha != TOKEN_FIELD_SHA256:
        raise SystemExit(
            f"token field sha mismatch: got {sha}, charter pin {TOKEN_FIELD_SHA256}"
        )

    raw = np.memmap(token_path, dtype=np.uint8, mode="r")
    if raw.size != TOKEN_FIELD_BYTES:
        raise SystemExit(f"token field is {raw.size} B, pin is {TOKEN_FIELD_BYTES} B")
    tokens = np.asarray(raw).reshape(N_FRAMES, H, W_FRAME)
    if int(tokens.max()) >= N_CLASS:
        raise SystemExit(f"token field has class {int(tokens.max())} >= {N_CLASS}")

    cost = np.asarray(
        np.memmap(cost_path, dtype="<f8", mode="r")
    ).reshape(N_FRAMES, H, W_FRAME)
    total_bits = float(cost.astype(np.float64).sum())
    if abs(total_bits - COST_FIELD_BITS) > 1e-3:
        raise SystemExit(
            f"cost field sums to {total_bits!r} bits, tb2 pin {COST_FIELD_BITS!r}"
        )

    counts = {int(k): int(v) for k, v in enumerate(np.bincount(tokens.ravel(), minlength=N_CLASS))}
    return Field(tokens, cost, sha, total_bits, counts)


# ---------------------------------------------------------------------------
# Causal context construction
# ---------------------------------------------------------------------------


def group_index() -> np.ndarray:
    """Global decode-group index g = (x % P) + DELTA * (y % P), shape (H, W)."""
    yy, xx = np.mgrid[0:H, 0:W_FRAME]
    return ((xx % HPAC_PATCH) + HPAC_DELTA * (yy % HPAC_PATCH)).astype(np.int32)


def _shift(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """out[y, x] = frame[y + dy, x + dx]; out-of-bounds -> UNK."""
    out = np.full((H, W_FRAME), UNK, dtype=np.uint8)
    ys0, ys1 = max(0, -dy), min(H, H - dy)
    xs0, xs1 = max(0, -dx), min(W_FRAME, W_FRAME - dx)
    if ys0 < ys1 and xs0 < xs1:
        out[ys0:ys1, xs0:xs1] = frame[ys0 + dy : ys1 + dy, xs0 + dx : xs1 + dx]
    return out


def causal_masks(groups: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """For each current-frame offset, where the neighbour is in an EARLIER group."""
    masks: dict[tuple[int, int], np.ndarray] = {}
    for kind, dy, dx in LADDER:
        if kind != "cur":
            continue
        if dx + HPAC_DELTA * dy >= 0:
            raise SystemExit(f"ladder offset ({dy},{dx}) is not mask-A causal")
        mask = np.zeros((H, W_FRAME), dtype=bool)
        ys0, ys1 = max(0, -dy), min(H, H - dy)
        xs0, xs1 = max(0, -dx), min(W_FRAME, W_FRAME - dx)
        if ys0 < ys1 and xs0 < xs1:
            mask[ys0:ys1, xs0:xs1] = (
                groups[ys0 + dy : ys1 + dy, xs0 + dx : xs1 + dx] < groups[ys0:ys1, xs0:xs1]
            )
        masks[(dy, dx)] = mask
    return masks


def build_features(field: Field, n_feat: int, groups: np.ndarray) -> np.ndarray:
    """(n_feat, 600*H*W) uint8 causal feature planes for the leading ladder rungs."""
    masks = causal_masks(groups)
    per_frame = H * W_FRAME
    out = np.empty((n_feat, N_FRAMES * per_frame), dtype=np.uint8)
    blank = np.full((H, W_FRAME), UNK, dtype=np.uint8)
    for f in range(N_FRAMES):
        cur = field.tokens[f]
        prv = field.tokens[f - 1] if f >= 1 else blank
        prv2 = field.tokens[f - 2] if f >= 2 else blank
        lo = f * per_frame
        for j, (kind, dy, dx) in enumerate(LADDER[:n_feat]):
            if kind == "cur":
                vals = _shift(cur, dy, dx)
                vals = np.where(masks[(dy, dx)], vals, np.uint8(UNK))
            elif kind == "prv":
                vals = _shift(prv, dy, dx)
            else:
                vals = _shift(prv2, dy, dx)
            out[j, lo : lo + per_frame] = vals.ravel()
    return out


# ---------------------------------------------------------------------------
# Exact code lengths from a joint (context, token) count table
# ---------------------------------------------------------------------------


@dataclass
class RungResult:
    k: int
    features: tuple[str, ...]
    n_contexts: int
    hindsight_bits: float
    kt_bits: float
    hindsight_bytes: float
    kt_bytes: float


def _joint_counts(codes: np.ndarray, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort-based joint counting; alphabet-size independent.

    Returns (context_totals, per_context_class_counts) with rows aligned.
    """
    joint = codes * np.int64(N_CLASS) + tokens
    joint.sort(kind="stable")
    # Run boundaries over the sorted joint codes.
    change = np.empty(joint.size, dtype=bool)
    change[0] = True
    np.not_equal(joint[1:], joint[:-1], out=change[1:])
    starts = np.flatnonzero(change)
    values = joint[starts]
    counts = np.diff(np.append(starts, joint.size))
    ctx = values // N_CLASS
    cls = (values % N_CLASS).astype(np.int64)

    # Collapse to per-context rows of 5 class counts.
    ctx_change = np.empty(ctx.size, dtype=bool)
    ctx_change[0] = True
    np.not_equal(ctx[1:], ctx[:-1], out=ctx_change[1:])
    row = np.cumsum(ctx_change) - 1
    n_ctx = int(row[-1]) + 1
    table = np.zeros((n_ctx, N_CLASS), dtype=np.int64)
    table[row, cls] = counts
    return table.sum(axis=1), table


def score_rung(
    codes: np.ndarray, tokens: np.ndarray, n_class: int = N_CLASS
) -> tuple[float, float, int]:
    """Exact hindsight-ideal bits, exact KT prequential bits, context count."""
    totals, table = _joint_counts(codes, tokens)
    nz = table > 0

    # Hindsight-ideal: Sum_c Sum_t N_ct * log2(N_c / N_ct).
    n_ct = table[nz].astype(np.float64)
    n_c = np.repeat(totals.astype(np.float64), nz.sum(axis=1))
    hindsight = float(np.sum(n_ct * (np.log2(n_c) - np.log2(n_ct))))

    # KT / Dirichlet(1/2) prequential code length, exact and order independent:
    #   L = -log2 [ Gamma(K/2)/Gamma(N + K/2) * Prod_t Gamma(N_t + 1/2)/Gamma(1/2) ]
    # K is the LIVE alphabet size: under a class merge the field really does have
    # fewer symbols, and charging KT for a symbol that can never occur would
    # overstate the merged field's cost.
    half = 0.5
    k_half = n_class * half
    ln2 = np.log(2.0)
    per_ctx = (
        gammaln(totals.astype(np.float64) + k_half)
        - gammaln(k_half)
        - (gammaln(table.astype(np.float64) + half) - gammaln(half)).sum(axis=1)
    )
    kt = float(per_ctx.sum() / ln2)
    return hindsight, kt, int(totals.size)


# ---------------------------------------------------------------------------
# Real range coder control (encode -> decode -> byte identity)
# ---------------------------------------------------------------------------


class RangeEncoder:
    """Carry-less 32-bit range encoder, the classic Subbotin form."""

    TOP = 1 << 24
    BOT = 1 << 16

    def __init__(self) -> None:
        self.low = 0
        self.range = 0xFFFFFFFF
        self.out = bytearray()

    def encode(self, cum: int, freq: int, tot: int) -> None:
        r = self.range // tot
        self.low += r * cum
        self.range = r * freq
        while True:
            if (self.low ^ (self.low + self.range)) < self.TOP:
                pass
            elif self.range < self.BOT:
                self.range = (-self.low) & (self.BOT - 1)
            else:
                break
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & 0xFFFFFFFF
            self.range = (self.range << 8) & 0xFFFFFFFF

    def finish(self) -> bytes:
        for _ in range(4):
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & 0xFFFFFFFF
        return bytes(self.out)


class RangeDecoder:
    TOP = 1 << 24
    BOT = 1 << 16

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.low = 0
        self.range = 0xFFFFFFFF
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._next()) & 0xFFFFFFFF

    def _next(self) -> int:
        if self.pos < len(self.data):
            byte = self.data[self.pos]
            self.pos += 1
            return byte
        return 0

    def decode_freq(self, tot: int) -> int:
        self._r = self.range // tot
        value = (self.code - self.low) // self._r
        return min(value, tot - 1)

    def update(self, cum: int, freq: int, _tot: int) -> None:
        self.low += self._r * cum
        self.range = self._r * freq
        while True:
            if (self.low ^ (self.low + self.range)) < self.TOP:
                pass
            elif self.range < self.BOT:
                self.range = (-self.low) & (self.BOT - 1)
            else:
                break
            self.code = ((self.code << 8) | self._next()) & 0xFFFFFFFF
            self.low = (self.low << 8) & 0xFFFFFFFF
            self.range = (self.range << 8) & 0xFFFFFFFF


def real_coder_control(
    codes: np.ndarray, tokens: np.ndarray, n_positions: int, seed: int = 0
) -> dict[str, object]:
    """Encode a slice with an adaptive KT model through a REAL range coder.

    Proves code length == emitted bytes on this stream, and that the decoder
    reconstructs the tokens byte-identically.  Slice, not full field: the pure
    Python coder runs ~50k symbol/s, and the identity it proves is exact on
    whatever it codes.
    """
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(1, codes.size - n_positions)))
    sl = slice(start, start + n_positions)
    ctx = codes[sl].astype(np.int64)
    tok = tokens[sl].astype(np.int64)

    uniq, inverse = np.unique(ctx, return_inverse=True)
    counts = np.ones((uniq.size, N_CLASS), dtype=np.int64)  # Laplace(1) == integer KT-ish
    encoder = RangeEncoder()
    ideal_bits = 0.0
    for i in range(tok.size):
        row = counts[inverse[i]]
        tot = int(row.sum())
        symbol = int(tok[i])
        cum = int(row[:symbol].sum())
        freq = int(row[symbol])
        ideal_bits += -np.log2(freq / tot)
        encoder.encode(cum, freq, tot)
        row[symbol] += 1
    payload = encoder.finish()

    counts2 = np.ones((uniq.size, N_CLASS), dtype=np.int64)
    decoder = RangeDecoder(payload)
    recovered = np.empty(tok.size, dtype=np.int64)
    for i in range(tok.size):
        row = counts2[inverse[i]]
        tot = int(row.sum())
        target = decoder.decode_freq(tot)
        acc = 0
        symbol = 0
        for s in range(N_CLASS):
            if acc + int(row[s]) > target:
                symbol = s
                break
            acc += int(row[s])
        decoder.update(acc, int(row[symbol]), tot)
        recovered[i] = symbol
        row[symbol] += 1

    identical = bool(np.array_equal(recovered, tok))
    return {
        "positions": int(n_positions),
        "slice_start": start,
        "adaptive_code_length_bits": ideal_bits,
        "adaptive_code_length_bytes": ideal_bits / 8.0,
        "real_coder_payload_bytes": len(payload),
        "coder_overhead_bytes": len(payload) - ideal_bits / 8.0,
        "decode_byte_identical": identical,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------


def bars_for(stream_bytes: float, model_bytes: float) -> dict[str, object]:
    total = stream_bytes + model_bytes
    net = SHIPPED_TOTAL_BYTES - total
    return {
        "model_bytes_W": model_bytes,
        "stream_bytes": stream_bytes,
        "total_bytes": total,
        "net_vs_shipped_bytes": net,
        "break_even_bar_stream_bytes": SHIPPED_TOTAL_BYTES - model_bytes,
        "clears_break_even": bool(total < SHIPPED_TOTAL_BYTES),
        "demand_bar_stream_bytes": SHIPPED_TOTAL_BYTES - model_bytes - DEMAND_BYTES,
        "clears_demand": bool(net >= DEMAND_BYTES),
        "delta_s_if_realised": -net * S_PER_BYTE,
        "shortfall_vs_break_even_bytes": max(0.0, total - SHIPPED_TOTAL_BYTES),
        "ratio_to_shipped_stream": stream_bytes / SHIPPED_STREAM_BYTES,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--field-dir", type=Path, default=FIELD_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--rungs",
        type=int,
        nargs="+",
        default=list(DEFAULT_RUNGS),
        help="how many leading ladder features each rung uses",
    )
    parser.add_argument("--skip-sha", action="store_true", help="skip the 112 MB re-hash")
    parser.add_argument(
        "--merge",
        type=str,
        default=None,
        metavar="SRC:DST",
        help=(
            "merge class SRC into class DST across the WHOLE field before measuring "
            "(e.g. '1:0' = Lane into Road). Prices the RATE side of a field-side "
            "alphabet change (ddm_no1 row 3 / tba1-D3). NOT lossless: the decoded "
            "segmentation changes, so d_seg changes by an amount this arm does NOT "
            "measure -- no scorer runs here."
        ),
    )
    parser.add_argument("--real-coder-positions", type=int, default=200_000)
    parser.add_argument("--no-real-coder", action="store_true")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    retained = args.out_dir / "retained"
    retained.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("[lm1] loading + verifying pinned field ...", flush=True)
    field = load_field(args.field_dir, verify_sha=not args.skip_sha)
    shipped_bytes = field.shipped_bits_total / 8.0
    print(
        f"[lm1] token sha OK  shipped stream = {shipped_bytes:,.4f} B "
        f"(physical {SHIPPED_STREAM_BYTES:,} B)",
        flush=True,
    )

    merge_src = merge_dst = None
    n_class = N_CLASS
    if args.merge:
        merge_src, merge_dst = (int(v) for v in args.merge.split(":"))
        if not (0 <= merge_src < N_CLASS and 0 <= merge_dst < N_CLASS) or merge_src == merge_dst:
            raise SystemExit(f"--merge {args.merge} is not a valid class pair")
        # Relabel so the surviving classes stay contiguous 0..n_class-1: the KT
        # alphabet size must be the LIVE one, not the original 5.
        remap = np.arange(N_CLASS, dtype=np.uint8)
        remap[merge_src] = merge_dst
        survivors = sorted(set(remap.tolist()))
        compact = {old: new for new, old in enumerate(survivors)}
        remap = np.array([compact[int(v)] for v in remap], dtype=np.uint8)
        merged_counts_before = dict(field.class_counts)
        field.tokens = remap[field.tokens]
        n_class = len(survivors)
        print(
            f"[lm1] MERGE {merge_src}->{merge_dst}: alphabet {N_CLASS} -> {n_class}; "
            f"class {merge_src} was {merged_counts_before[merge_src]:,} positions "
            f"({100*merged_counts_before[merge_src]/TOKEN_FIELD_BYTES:.4f}% of field). "
            f"RATE SIDE ONLY -- d_seg changes and is NOT measured here.",
            flush=True,
        )

    groups = group_index()
    if int(groups.max()) + 1 != N_GROUPS:
        raise SystemExit(f"group count {int(groups.max())+1} != {N_GROUPS}")

    n_feat = max(args.rungs)
    print(f"[lm1] building {n_feat} causal feature planes ...", flush=True)
    feats = build_features(field, n_feat, groups)
    tokens_flat = field.tokens.reshape(-1).astype(np.int64)

    results: list[RungResult] = []
    codes = np.zeros(tokens_flat.size, dtype=np.int64)
    built = 0
    for k in sorted(args.rungs):
        while built < k:
            codes *= ALPHABET
            codes += feats[built].astype(np.int64)
            built += 1
        hind, kt, n_ctx = score_rung(codes, tokens_flat, n_class)
        names = tuple(f"{kind}{dy:+d}{dx:+d}" for kind, dy, dx in LADDER[:k])
        res = RungResult(k, names, n_ctx, hind, kt, hind / 8.0, kt / 8.0)
        results.append(res)
        print(
            f"[lm1] k={k:2d} ctx={n_ctx:>12,}  hindsight={hind/8:>12,.1f} B"
            f"  KT_prequential={kt/8:>12,.1f} B"
            f"  (shipped stream {shipped_bytes:,.1f} B)  {time.time()-t0:6.1f}s",
            flush=True,
        )

    control: dict[str, object] | None = None
    if not args.no_real_coder:
        print("[lm1] real range-coder control ...", flush=True)
        control = real_coder_control(codes, tokens_flat, args.real_coder_positions)
        payload = control.pop("payload")
        (retained / "real_coder_control.bin").write_bytes(payload)
        print(
            f"[lm1] real coder: code_length={control['adaptive_code_length_bytes']:,.2f} B"
            f"  emitted={control['real_coder_payload_bytes']:,} B"
            f"  overhead={control['coder_overhead_bytes']:,.2f} B"
            f"  byte_identical={control['decode_byte_identical']}",
            flush=True,
        )

    best = min(results, key=lambda r: r.kt_bytes)
    best_hind = min(results, key=lambda r: r.hindsight_bytes)

    per_w = {}
    for w in (0, 5_000, 10_000, 20_000):
        per_w[str(w)] = {
            "kt_prequential_best": bars_for(best.kt_bytes, w),
            "hindsight_ideal_best": bars_for(best_hind.hindsight_bytes, w),
        }

    payload = {
        "arm": "ddm_lm1",
        "question": "can a W-byte learned parametric model replace the 13,515 B HPAC tables, losslessly, netting bytes?",
        "axis": "[macOS-CPU advisory / scorer-free lossless coding measurement]",
        "score_claim": False,
        "promotable": False,
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "pins": {
            "token_field_sha256": field.token_sha256,
            "token_field_bytes": TOKEN_FIELD_BYTES,
            "cost_field_bits": field.shipped_bits_total,
            "cost_field_bits_pin": COST_FIELD_BITS,
            "class_counts": field.class_counts,
            "shipped_stream_bytes": SHIPPED_STREAM_BYTES,
            "shipped_model_bytes": SHIPPED_MODEL_BYTES,
            "shipped_total_bytes": SHIPPED_TOTAL_BYTES,
            "demand_bytes": DEMAND_BYTES,
        },
        "decode_causality": {
            "patch": HPAC_PATCH,
            "delta": HPAC_DELTA,
            "n_groups": N_GROUPS,
            "group_rule": "g = (x % 64) + 2 * (y % 64)",
            "causal_rule": "current-frame neighbour legal iff its GLOBAL group index < own",
            "note": "cross-patch earlier-group neighbours are legal and ARE used here; the shipped network is blind to them (zero-padded per-patch convolution)",
        },
        "ladder": [f"{kind}{dy:+d}{dx:+d}" for kind, dy, dx in LADDER[:n_feat]],
        "merge": (
            None
            if merge_src is None
            else {
                "src_class": merge_src,
                "dst_class": merge_dst,
                "live_alphabet": n_class,
                "lossless": False,
                "scope": "RATE SIDE ONLY -- d_seg changes and is NOT measured by this arm",
            }
        ),
        "rungs": [
            {
                "k": r.k,
                "features": list(r.features),
                "n_contexts": r.n_contexts,
                "hindsight_ideal_bytes": r.hindsight_bytes,
                "kt_prequential_bytes": r.kt_bytes,
                "hindsight_ratio_to_shipped_stream": r.hindsight_bytes / SHIPPED_STREAM_BYTES,
                "kt_ratio_to_shipped_stream": r.kt_bytes / SHIPPED_STREAM_BYTES,
            }
            for r in results
        ],
        "best_kt_rung_k": best.k,
        "best_hindsight_rung_k": best_hind.k,
        "bars": per_w,
        "real_coder_control": control,
        "elapsed_seconds": time.time() - t0,
    }
    out = args.out_dir / "RESULT.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[lm1] wrote {out}", flush=True)

    print("\n=== ddm_lm1 verdict inputs ===")
    print(f"shipped stream                 {SHIPPED_STREAM_BYTES:>12,} B")
    print(f"shipped model                  {SHIPPED_MODEL_BYTES:>12,} B")
    print(f"shipped total                  {SHIPPED_TOTAL_BYTES:>12,} B")
    print(f"best KT prequential (W=0)      {best.kt_bytes:>12,.1f} B   k={best.k}")
    print(f"best hindsight-ideal (W=0)     {best_hind.hindsight_bytes:>12,.1f} B   k={best_hind.k}")
    for w in (0, 5_000, 10_000, 20_000):
        b = per_w[str(w)]["kt_prequential_best"]
        print(
            f"W={w:>6,}  break-even bar stream < {b['break_even_bar_stream_bytes']:>10,.0f} B"
            f" | demand bar <= {b['demand_bar_stream_bytes']:>10,.0f} B"
            f" | measured {b['stream_bytes']:>12,.1f} B"
            f" | clears_break_even={b['clears_break_even']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
