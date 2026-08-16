#!/usr/bin/env python3
"""ddm_rt1 follow-on #1 -- race REAL coders on the retained flip mask. No entropy estimates.

The gate this pays: rt1 §3.3 priced a seg-correction channel on the free label boundary at a
33,082 B i.i.d. entropy FLOOR against a 44,480 B budget, and named the admission bar
**35,117 B** (break-even at sq1's eta = 0.7895).  A floor is not a coder.  This module codes the
retained mask with real coders and verifies each arithmetic payload by DECODING IT BACK through
the SAME online context machine the encoder used -- so the reported bytes belong to a stream a
real receiver could actually parse.

Support: `free_band_mask.npy` = the transmitted labels' own boundary, which the receiver
reconstructs for zero bytes.  One binary symbol per band pixel.  Every context is built only
from (a) the labels, which are free, and (b) symbols already decoded -- never from the future.

Coders raced:
  M0 raw packed bits            -- the do-nothing baseline
  M1 brotli(packed) q11         -- dumb general-purpose baseline
  M2 lzma(packed)               -- ditto
  M3 static binary AC           -- one global probability; realizes the i.i.d. floor
  M4 adaptive binary AC order-0 -- no context
  M5 CABAC raster order         -- context = edge class pair x causal neighbour flips x temporal
  M6 CABAC boundary-walk order  -- context = run state x temporal, on a 1-D traversal of the curve
  M7 CABAC boundary-walk order  -- context = edge class pair x run state x temporal

Axis `[macOS-CPU $0 desk]`; scorer-free, no launches, no Modal. `score_claim=false`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import time
from pathlib import Path

import numpy as np

FRAMES, SEG_H, SEG_W = 600, 384, 512
SCORED_PX = FRAMES * SEG_H * SEG_W
N_CLASSES = 5

# rt1 §3.3 pins -- REUSED, never re-derived (rt1 §4 do-not-re-derive list)
IID_FLOOR_B = 33082.0
BAR_B = 35117.0            # break-even carrier at eta = 0.7895
BUDGET_B = 44480.0         # break-even carrier at eta = 1.0
SEG_DS_PER_FLIP = 100.0 / SCORED_PX
RATE_DS_PER_BYTE = 25.0 / 37_545_489

DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816")
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/ep0634/retained/coders/"
    "s1p25_c1p0/decoded_spatial_tokens.rc64.bin"
)


class CoderError(RuntimeError):
    """Fail-closed error for custody or round-trip violations."""


# ============================================================================================
# a real binary range coder (LZMA-style), adaptive 11-bit probabilities
# ============================================================================================
K_BITS = 11
K_TOP = 1 << 24
K_MOVE = 5
P_INIT = 1 << (K_BITS - 1)
P_MAX = 1 << K_BITS


class RangeEncoder:
    """`prob` is P(bit == 0) scaled to 2**K_BITS."""

    def __init__(self) -> None:
        self.low = 0
        self.range = 0xFFFFFFFF
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def _shift_low(self) -> None:
        if self.low < 0xFF000000 or self.low > 0xFFFFFFFF:
            carry = self.low >> 32
            self.out.append((self.cache + carry) & 0xFF)
            while self.cache_size > 1:
                self.out.append((0xFF + carry) & 0xFF)
                self.cache_size -= 1
            self.cache = (self.low >> 24) & 0xFF
            self.cache_size = 0
        self.cache_size += 1
        self.low = (self.low << 8) & 0xFFFFFFFF

    def encode_bit(self, prob: int, bit: int) -> None:
        bound = (self.range >> K_BITS) * prob
        if bit == 0:
            self.range = bound
        else:
            self.low += bound
            self.range -= bound
        while self.range < K_TOP:
            self.range <<= 8
            self._shift_low()

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class RangeDecoder:
    """Exact mirror of RangeEncoder."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 1  # skip the encoder's leading cache byte
        self.range = 0xFFFFFFFF
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF

    def _byte(self) -> int:
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        return 0

    def decode_bit(self, prob: int) -> int:
        bound = (self.range >> K_BITS) * prob
        if self.code < bound:
            self.range = bound
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            bit = 1
        while self.range < K_TOP:
            self.range <<= 8
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF
        return bit


def adapt(prob: int, bit: int) -> int:
    if bit == 0:
        return prob + ((P_MAX - prob) >> K_MOVE)
    return prob - (prob >> K_MOVE)


# ============================================================================================
# geometry -- every one of these is a function of the LABELS only, hence free to the receiver
# ============================================================================================
def boundary(lab: np.ndarray) -> np.ndarray:
    b = np.zeros(lab.shape, dtype=bool)
    d = lab[:-1, :] != lab[1:, :]
    b[:-1, :] |= d
    b[1:, :] |= d
    d = lab[:, :-1] != lab[:, 1:]
    b[:, :-1] |= d
    b[:, 1:] |= d
    return b


PAIR_ID = {}
for _a in range(N_CLASSES):
    for _b in range(_a + 1, N_CLASSES):
        PAIR_ID[(_a, _b)] = len(PAIR_ID)
N_PAIR = len(PAIR_ID) + 1  # + 1 bucket for "more than two classes meet here"


def edge_pair_field(lab: np.ndarray) -> np.ndarray:
    """Id of the class pair meeting at each pixel; N_PAIR-1 when more than two classes meet."""
    pad = np.pad(lab, 1, mode="edge")
    neigh = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]], axis=0)
    present = np.zeros((N_CLASSES, *lab.shape), dtype=bool)
    for c in range(N_CLASSES):
        present[c] = (lab == c) | (neigh == c).any(axis=0)
    out = np.full(lab.shape, N_PAIR - 1, dtype=np.uint8)
    two = present.sum(axis=0) == 2
    if two.any():
        idx = np.argsort(~present, axis=0, kind="stable")
        lo = np.minimum(idx[0][two], idx[1][two]).astype(np.int64)
        hi = np.maximum(idx[0][two], idx[1][two]).astype(np.int64)
        ids = np.zeros(lo.shape, dtype=np.uint8)
        for (pa, pb), pid in PAIR_ID.items():
            ids[(lo == pa) & (hi == pb)] = pid
        out[two] = ids
    return out


def walk_order(band: np.ndarray) -> np.ndarray:
    """Traverse the boundary as curves: 8-connected components, deterministic DFS.

    Depends only on the label boundary, so the receiver reproduces it exactly for zero bytes.
    """
    h, w = band.shape
    flat = band.reshape(-1)
    idx = np.flatnonzero(flat)
    if idx.size == 0:
        return idx.astype(np.int64)
    seen = np.zeros(h * w, dtype=bool)
    order = np.empty(idx.size, dtype=np.int64)
    k = 0
    for start in idx.tolist():
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        while stack:
            p = stack.pop()
            order[k] = p
            k += 1
            y, x = divmod(p, w)
            for dy in (-1, 0, 1):
                ny = y + dy
                if ny < 0 or ny >= h:
                    continue
                for dx in (-1, 0, 1):
                    nx = x + dx
                    if (dy == 0 and dx == 0) or nx < 0 or nx >= w:
                        continue
                    q = ny * w + nx
                    if seen[q] or not flat[q]:
                        continue
                    seen[q] = True
                    stack.append(q)
    return order[:k]


# ============================================================================================
# the online context machine -- ONE implementation drives both encode and decode
# ============================================================================================
def code_stream(model: str, frames: int, work: Path, tokens: Path,
                payload: bytes | None = None) -> tuple[bytes, np.ndarray, int]:
    """Encode (payload=None) or decode the band-restricted flip stream under `model`.

    Both directions run this same function, so a successful decode proves the contexts are
    causal: nothing here can read a symbol the receiver has not already produced.
    """
    flip = np.load(work / "flip_mask_vs_gt.npy", mmap_mode="r")
    tok = np.memmap(tokens, dtype=np.uint8, mode="r", shape=(FRAMES, SEG_H, SEG_W))
    encoding = payload is None
    enc = RangeEncoder() if encoding else None
    dec = None if encoding else RangeDecoder(payload)

    n_ctx = {"order0": 1, "raster": N_PAIR * 8, "walk": 8, "walk_pair": N_PAIR * 8}[model]
    probs = [P_INIT] * n_ctx
    produced: list[np.ndarray] = []
    prev = np.zeros((SEG_H, SEG_W), dtype=np.uint8)
    total = 0
    for t in range(frames):
        lab = np.asarray(tok[t])
        bnd = boundary(lab)
        pair = edge_pair_field(lab) if model in ("raster", "walk_pair") else None
        truth = (np.asarray(flip[t]) & bnd).astype(np.uint8) if encoding else None
        cur = np.zeros((SEG_H, SEG_W), dtype=np.uint8)

        if model in ("walk", "walk_pair"):
            order = walk_order(bnd)
            run1 = run2 = run3 = 0
            pf = prev.reshape(-1)
            pr = pair.reshape(-1) if pair is not None else None
            curf = cur.reshape(-1)
            tf = truth.reshape(-1) if truth is not None else None
            for p in order.tolist():
                run = run1 + run2 + run3
                c = run * 2 + int(pf[p])
                if model == "walk_pair":
                    c = int(pr[p]) * 8 + c
                if encoding:
                    b = int(tf[p])
                    enc.encode_bit(probs[c], b)
                else:
                    b = dec.decode_bit(probs[c])
                probs[c] = adapt(probs[c], b)
                curf[p] = b
                run3, run2, run1 = run2, run1, b
                total += 1
        else:
            ys, xs = np.nonzero(bnd)
            for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
                if model == "order0":
                    c = 0
                else:
                    nf = 0
                    if y > 0:
                        nf += int(cur[y - 1, x])
                        if x > 0:
                            nf += int(cur[y - 1, x - 1])
                        if x + 1 < SEG_W:
                            nf += int(cur[y - 1, x + 1])
                    if x > 0:
                        nf += int(cur[y, x - 1])
                    c = int(pair[y, x]) * 8 + min(nf, 3) * 2 + int(prev[y, x])
                if encoding:
                    b = int(truth[y, x])
                    enc.encode_bit(probs[c], b)
                else:
                    b = dec.decode_bit(probs[c])
                probs[c] = adapt(probs[c], b)
                cur[y, x] = b
                total += 1
        produced.append(cur)
        prev = cur
    out = enc.finish() if encoding else payload
    return out, np.stack(produced), total


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ============================================================================================
# the race
# ============================================================================================
def race(args: argparse.Namespace) -> int:
    import brotli

    t0 = time.time()
    flip = np.load(args.work / "flip_mask_vs_gt.npy", mmap_mode="r")
    band = np.load(args.work / "free_band_mask.npy", mmap_mode="r")
    tok = np.memmap(args.tokens, dtype=np.uint8, mode="r", shape=(FRAMES, SEG_H, SEG_W))

    bits, ones, band_px, off_band = [], 0, 0, 0
    for t in range(args.frames):
        bnd = boundary(np.asarray(tok[t]))
        if not np.array_equal(bnd, np.asarray(band[t]).astype(bool)):
            raise CoderError(f"retained free_band_mask != boundary(labels) at frame {t}")
        f = np.asarray(flip[t]).astype(np.uint8)
        off_band += int(np.count_nonzero(f & ~bnd))
        fb = (f & bnd).astype(np.uint8)
        bits.append(fb[bnd])
        ones += int(fb.sum())
        band_px += int(bnd.sum())
    raster_bits = np.concatenate(bits)
    n = int(raster_bits.size)
    p1 = ones / n
    packed = np.packbits(raster_bits).tobytes()

    results: list[dict] = []
    payloads: dict[str, bytes] = {}

    def add(tag: str, name: str, payload: bytes, verified: bool, note: str = "") -> None:
        payloads[tag] = payload
        results.append({
            "tag": tag, "coder": name, "bytes": len(payload), "sha256": sha256_bytes(payload),
            "roundtrip_verified": verified, "bits_per_flip": len(payload) * 8 / ones,
            "note": note,
        })

    def unpacks_to_source(blob: bytes) -> bool:
        return bool(np.array_equal(np.unpackbits(np.frombuffer(blob, dtype=np.uint8))[:n],
                                   raster_bits))

    # Every "verified" flag below is EARNED by actually inverting the payload -- a verification
    # marker that no code paid for is exactly the fake-implementation class.
    add("M0", "raw packed band bits", packed, unpacks_to_source(packed), "no coder")
    b1 = brotli.compress(packed, quality=11)
    add("M1", "brotli(packed) q11", b1, unpacks_to_source(brotli.decompress(b1)))
    b2 = lzma.compress(packed, preset=9 | lzma.PRESET_EXTREME)
    add("M2", "lzma(packed) preset 9|EXTREME", b2, unpacks_to_source(lzma.decompress(b2)))

    # M3 -- static probability, the i.i.d. floor realized by an actual coder
    sp = max(1, min(P_MAX - 1, round((1 - p1) * P_MAX)))
    e = RangeEncoder()
    for b in raster_bits.tolist():
        e.encode_bit(sp, b)
    pay = e.finish()
    d = RangeDecoder(pay)
    back = np.fromiter((d.decode_bit(sp) for _ in range(n)), dtype=np.uint8, count=n)
    add("M3", "static binary AC (i.i.d. realized)", pay, bool(np.array_equal(back, raster_bits)),
        f"P(flip)={p1:.7f} quantised to {P_MAX - sp}/2048")

    for tag, model, name in (
        ("M4", "order0", "adaptive binary AC, order-0"),
        ("M5", "raster", "CABAC raster (pair x causal-neighbours x temporal)"),
        ("M6", "walk", "CABAC boundary-walk (run x temporal)"),
        ("M7", "walk_pair", "CABAC boundary-walk (pair x run x temporal)"),
    ):
        pay, enc_field, cnt = code_stream(model, args.frames, args.work, args.tokens)
        _, dec_field, _ = code_stream(model, args.frames, args.work, args.tokens, payload=pay)
        ok = bool(np.array_equal(enc_field, dec_field)) and cnt == n
        if ok:
            truth = np.stack([
                (np.asarray(flip[t]) & boundary(np.asarray(tok[t]))).astype(np.uint8)
                for t in range(args.frames)
            ])
            ok = bool(np.array_equal(dec_field, truth))
        n_ctx = {"order0": 1, "raster": N_PAIR * 8, "walk": 8,
                 "walk_pair": N_PAIR * 8}[model]
        add(tag, name, pay, ok, f"{n_ctx} contexts")

    for r in results:
        if not r["roundtrip_verified"]:
            raise CoderError(f"{r['tag']} {r['coder']} did not decode back to the source field")

    best = min(results, key=lambda r: r["bytes"])
    gain = ones * SEG_DS_PER_FLIP
    outdir = args.work / "coder_race"
    outdir.mkdir(parents=True, exist_ok=True)
    for tag, pay in payloads.items():  # ALWAYS KEEP THE PAYLOAD
        (outdir / f"{tag}.bin").write_bytes(pay)

    rec = {
        "schema": "ddm_rt1_flip_mask_coder_race.v1",
        "axis": "[macOS-CPU $0 desk] scorer-free",
        "score_claim": False,
        "promotable": False,
        "support": {
            "band_px": band_px, "band_flips": ones, "density": p1,
            "off_band_flips_not_addressable": off_band,
        },
        "pins_reused_not_rederived": {
            "iid_floor_B": IID_FLOOR_B, "bar_B": BAR_B, "budget_B": BUDGET_B,
            "seg_dS_per_flip": SEG_DS_PER_FLIP, "rate_dS_per_byte": RATE_DS_PER_BYTE,
        },
        "results": sorted(results, key=lambda r: r["bytes"]),
        "best": best,
        "verdict": {
            "beats_bar_35117_B": best["bytes"] < BAR_B,
            "beats_iid_floor_33082_B": best["bytes"] < IID_FLOOR_B,
            "best_over_iid_floor": best["bytes"] / IID_FLOOR_B,
            "seg_gain_if_all_band_flips_fixed_S": -gain,
            "eta_required_for_break_even": best["bytes"] * RATE_DS_PER_BYTE / gain,
            "net_S_at_eta_1p0": -gain + best["bytes"] * RATE_DS_PER_BYTE,
            "net_S_at_eta_0p7895": -0.7895 * gain + best["bytes"] * RATE_DS_PER_BYTE,
            "net_S_at_eta_0p5406": -0.5406 * gain + best["bytes"] * RATE_DS_PER_BYTE,
        },
        "retained_payloads": str(outdir),
        "frames_coded": args.frames,
        "wall_s": time.time() - t0,
    }
    (outdir / "RT1_CODER_RACE.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def analysis(args: argparse.Namespace) -> int:
    """Why the coders land where they do: clustering, free conditioning, target-class cost.

    These are IDEAL-model limits (empirical conditional entropies, no learning cost).  They
    bound what any coder using the same free context could reach; the race reports what a real
    coder actually reached.
    """
    flip = np.load(args.work / "flip_mask_vs_gt.npy", mmap_mode="r")
    tgt = np.load(args.work / "flip_target_class.npy", mmap_mode="r")
    tok = np.memmap(args.tokens, dtype=np.uint8, mode="r", shape=(FRAMES, SEG_H, SEG_W))

    cols: dict[str, list] = {k: [] for k in ("bits", "pair", "cfc", "temporal", "row", "deg")}
    tg_ctx, tg_val = [], []
    runs: dict[int, int] = {}
    pair11 = prev1 = walk_n = walk_ones = 0
    prev = np.zeros((SEG_H, SEG_W), dtype=np.uint8)
    rows_idx = np.repeat(np.arange(SEG_H)[:, None], SEG_W, axis=1) // 48
    t0 = time.time()
    for t in range(args.frames):
        lab = np.asarray(tok[t])
        bnd = boundary(lab)
        fb = (np.asarray(flip[t]) & bnd).astype(np.uint8)
        pair = edge_pair_field(lab)
        pad = np.pad(fb, 1)
        cfc = np.minimum(pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] + pad[1:-1, :-2], 3)
        pb = np.pad(bnd.astype(np.uint8), 1)
        deg = np.minimum(pb[:-2, 1:-1] + pb[2:, 1:-1] + pb[1:-1, :-2] + pb[1:-1, 2:], 4)
        cols["bits"].append(fb[bnd])
        cols["pair"].append(pair[bnd].astype(np.int64))
        cols["cfc"].append(cfc[bnd].astype(np.int64))
        cols["temporal"].append(prev[bnd].astype(np.int64))
        cols["row"].append(rows_idx[bnd].astype(np.int64))
        cols["deg"].append(deg[bnd].astype(np.int64))
        m = fb.astype(bool) & bnd
        if m.any():
            tg_ctx.append(lab[m].astype(np.int64) * N_PAIR + pair[m].astype(np.int64))
            tg_val.append(np.asarray(tgt[t])[m].astype(np.int64))
        fw = fb.reshape(-1)[walk_order(bnd)]
        walk_n += fw.size
        walk_ones += int(fw.sum())
        pv = np.concatenate(([0], fw[:-1]))
        pair11 += int(((fw == 1) & (pv == 1)).sum())
        prev1 += int((pv == 1).sum())
        d = np.diff(np.concatenate(([0], fw, [0])))
        for length in (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist():
            runs[length] = runs.get(length, 0) + 1
        prev = fb
    A = {k: np.concatenate(v) for k, v in cols.items()}
    bits = A["bits"]
    n, ones = int(bits.size), int(bits.sum())
    base = float(n * _binary_entropy(np.array(ones / n)) / 8)

    def cond_bytes(ctx: np.ndarray, k: int) -> float:
        tot = np.bincount(ctx, minlength=k).astype(np.float64)
        on = np.bincount(ctx, weights=bits.astype(np.float64), minlength=k)
        m = tot > 0
        p = np.zeros_like(tot)
        p[m] = on[m] / tot[m]
        return float((tot[m] * _binary_entropy(p[m])).sum() / 8.0)

    pc = A["pair"] * 4 + A["cfc"]
    pct = pc * 2 + A["temporal"]
    pctr = pct * 8 + A["row"]
    conditioning = {
        "iid": base,
        "pair": cond_bytes(A["pair"], N_PAIR),
        "causal_neighbours": cond_bytes(A["cfc"], 4),
        "temporal": cond_bytes(A["temporal"], 2),
        "row_band8": cond_bytes(A["row"], 8),
        "band_degree": cond_bytes(A["deg"], 5),
        "pair_x_causal": cond_bytes(pc, N_PAIR * 4),
        "pair_x_causal_x_temporal": cond_bytes(pct, N_PAIR * 4 * 2),
        "pair_x_causal_x_temporal_x_row": cond_bytes(pctr, N_PAIR * 4 * 2 * 8),
        "pair_x_causal_x_temporal_x_row_x_degree": cond_bytes(
            pctr * 5 + A["deg"], N_PAIR * 4 * 2 * 8 * 5),
    }
    tc, tv = np.concatenate(tg_ctx), np.concatenate(tg_val)
    _, cnt = np.unique(tv, return_counts=True)
    p = cnt / tv.size
    h_uncond = float(-(p * np.log2(p)).sum())
    tot_bits = 0.0
    for c in np.unique(tc):
        sub = tv[tc == c]
        _, cc = np.unique(sub, return_counts=True)
        pp = cc / cc.sum()
        tot_bits += float(-(pp * np.log2(pp)).sum()) * cc.sum()
    marg = walk_ones / walk_n
    rec = {
        "schema": "ddm_rt1_flip_mask_analysis.v1",
        "axis": "[macOS-CPU $0 desk] scorer-free",
        "score_claim": False,
        "label": "IDEAL-model limits (empirical conditional entropy); a real coder pays a "
                 "learning cost on top -- see the race receipt for realized bytes",
        "band_symbols": n, "band_flips": ones, "density": ones / n,
        "clustering_walk_order": {
            "p_flip": marg,
            "p_flip_given_prev_flip": pair11 / max(prev1, 1),
            "lift": (pair11 / max(prev1, 1)) / marg,
            "mean_run_length": walk_ones / max(sum(runs.values()), 1),
            "run_length_histogram": {str(k): runs[k] for k in sorted(runs)},
        },
        "conditional_entropy_bytes": conditioning,
        "conditional_gain_vs_iid": {
            k: 1 - v / base for k, v in conditioning.items() if k != "iid"},
        "target_class_side_channel": {
            "flips": int(tv.size),
            "H_uncond_bits": h_uncond,
            "H_given_own_label_and_pair_bits": tot_bits / tv.size,
            "bytes_given_free_context": tot_bits / 8.0,
        },
        "frames": args.frames,
        "wall_s": time.time() - t0,
    }
    outdir = args.work / "coder_race"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "RT1_MASK_ANALYSIS.json").write_text(
        json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["race", "analysis"], default="race")
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    ap.add_argument("--frames", type=int, default=FRAMES)
    args = ap.parse_args(argv)
    return race(args) if args.mode == "race" else analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
