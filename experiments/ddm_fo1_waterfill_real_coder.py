#!/usr/bin/env python3
"""ddm_fo1 -- price sr1's WATERFILLED seg-correction support with a REAL coder.

The gate this pays: sr1's `SR1_WATERFILL.json` claims 4,276 B for 6,512 described flips on a
41-cell waterfilled sub-support of rt1's free label boundary.  That number is an IDEAL
conditional-entropy limit -- sr1 says so itself.  pn2 §6 named the owed row and pre-registered
the bar BEFORE any run:

    real coded bytes <= 5,066 B  ->  the waterfilled channel is a real supplier
    real coded bytes >  5,066 B  ->  sr1's waterfill S is an ideal-entropy artifact and rt1's
                                     CLOSED verdict is restored on better evidence

The bar is `eta_projected * flips * (SEG_dS_per_flip / RATE_dS_per_byte)` with pn2's pooled
n=12 projected eta = 0.6111, 6,512 flips and 1.273108 B/flip -> 5,065.6 B.  It is FROZEN here.

What this module does, in order:
  1. RECONSTRUCT sr1's 41-cell selection from the transmitted labels alone -- every cell factor
     (own class, lowest differing 4-neighbour class, min(degree, 4), row band) is a deterministic
     function of the decoded label field, so the receiver recomputes the support for zero bytes.
     Fail-closed controls: the per-cell histograms must reproduce sr1's retained
     `cell_band_px.npy` / `cell_flip_px.npy` BYTE-IDENTICALLY, and the marginal waterfill must
     land on 41 cells / 6,512 flips / 4,276.17 B.
  2. RESTRICT rt1's retained flip mask to that support and race the SAME M0-M7 coders, each
     payload verified by DECODING IT BACK through the same online context machine.
  3. CODE THE TARGET CLASS for real too -- sr1's 4,276 B includes a 0.2253 bit/flip target-class
     term, so the honest comparison against the bar must carry a real coder for it as well.
  4. ADJUDICATE the total against the frozen 5,066 B bar.

M8 is an ADDITION beyond pn2's pre-registered M0-M7 set: it walks the FULL label boundary and
codes only the support pixels, which keeps rt1's curve geometry that a sparse sub-support
destroys.  Extra coders can only LOWER the realized bytes, so including it is the conservative
direction for a ">bar" verdict and the transparent direction for a "<=bar" one.  Both the
M0-M7 best and the overall best are reported.

Axis `[macOS-CPU advisory]`; scorer-free, $0, no launches, no Modal.  `score_claim=false`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import socket
import time
from pathlib import Path

import numpy as np

FRAMES, SEG_H, SEG_W = 600, 384, 512
SCORED_PX = FRAMES * SEG_H * SEG_W  # 117,964,800
N_CLASSES = 5

# --- exchange rates: exact contest arithmetic, REUSED from sr1/pn2, never re-derived ---------
SEG_DS_PER_FLIP = 100.0 / SCORED_PX          # 8.477105e-07
RATE_DS_PER_BYTE = 25.0 / 37_545_489         # 6.658590e-07
BYTES_PER_FLIP_BAR = SEG_DS_PER_FLIP / RATE_DS_PER_BYTE  # 1.273108 B/flip

# --- pins cited from the stores, never re-derived --------------------------------------------
BASE_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
BASE_S = 0.15959729295498598
GAP_S = BASE_S - 0.15
SR1_ETA_SELECTION = 0.6235      # the eta at which sr1's 41-cell selection is defined
PN2_ETA_PROJECTED = 0.6111      # pn2 §5 pooled n=12 projected eta -- the bar's eta
PN2_ETA_UNPROJECTED = 0.5651    # pn2 §5 pooled n=12 unprojected eta
SR1_MAX_DEGREE = 4
SR1_ROW_BANDS = 8

# The pre-registered bar.  FROZEN.  Do not recompute it from a fresher eta.
BAR_B = 5066.0

# sr1's retained waterfill row at its own selection eta -- the numbers this arm must reproduce.
SR1_IDEAL_CELLS = 41
SR1_IDEAL_FLIPS = 6512
SR1_IDEAL_BYTES = 4276.171156196069
SR1_TARGET_BITS_PER_FLIP = 0.22530701479359683

DEFAULT_RT1_WORK = Path("/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816")
DEFAULT_SR1_WORK = Path("/Volumes/APDataStore/pact/ddm_sr1_manufactured_seg_recovery_20260816")
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/ep0634/retained/coders/"
    "s1p25_c1p0/decoded_spatial_tokens.rc64.bin"
)
DEFAULT_GT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
)
DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_fo1_waterfill_real_coder")


class Fo1Error(RuntimeError):
    """Fail-closed error for custody, reconstruction, or round-trip violations."""


# ============================================================================================
# progress + payload custody
# ============================================================================================
def progress(work: Path, milestone: str, detail: dict) -> None:
    row = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm": "ddm_fo1",
        "milestone": milestone,
        "detail": detail,
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    work.mkdir(parents=True, exist_ok=True)
    with (work / "PROGRESS.jsonl").open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[fo1] {milestone}: {json.dumps(detail, sort_keys=True)}", flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_array(path: Path, arr: np.ndarray) -> dict:
    """ALWAYS KEEP THE PAYLOAD -- bytes to disk, then sha the bytes that landed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "shape": list(arr.shape), "dtype": str(arr.dtype)}


def save_blob(path: Path, blob: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return {"path": str(path), "bytes": len(blob), "sha256": sha256_bytes(blob)}


# ============================================================================================
# rt1's binary range coder -- COPIED VERBATIM so the bytes are comparable to rt1's race
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
# geometry -- every field below is a function of the TRANSMITTED LABELS only, hence free
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


PAIR_ID: dict[tuple[int, int], int] = {}
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


def cell_features(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sr1's cell features, COPIED VERBATIM from ddm_sr1_manufactured_seg_recovery.cell_features.

    Returns (own_class, partner_class, degree); partner is the lowest-indexed differing
    4-neighbour class (255 off the boundary) and degree counts distinct classes in the closed
    4-neighbourhood.  All three are functions of the transmitted labels alone.
    """
    own = lab
    present = np.zeros((N_CLASSES, *lab.shape), dtype=bool)
    for c in range(N_CLASSES):
        present[c] = lab == c
    shifted = np.zeros_like(present)
    for c in range(N_CLASSES):
        p = present[c]
        s = np.zeros_like(p)
        s[:-1, :] |= p[1:, :]
        s[1:, :] |= p[:-1, :]
        s[:, :-1] |= p[:, 1:]
        s[:, 1:] |= p[:, :-1]
        shifted[c] = s | p
    degree = shifted.sum(axis=0).astype(np.uint8)
    partner = np.full(lab.shape, 255, dtype=np.uint8)
    for c in range(N_CLASSES - 1, -1, -1):
        hit = shifted[c] & (own != c)
        partner[hit] = c
    return own, partner, degree


ROW_IDX = (np.arange(SEG_H) * SR1_ROW_BANDS // SEG_H).astype(np.int64)
ROW_PLANE = np.repeat(ROW_IDX[:, None], SEG_W, axis=1)
N_CELLS = N_CLASSES * (N_CLASSES + 1) * (SR1_MAX_DEGREE + 1) * SR1_ROW_BANDS  # 1200


def cell_key(lab: np.ndarray) -> np.ndarray:
    """sr1's cell index for every pixel -- receiver-derivable, zero archive bytes."""
    own, partner, degree = cell_features(lab)
    deg = np.minimum(degree, SR1_MAX_DEGREE)
    part = np.where(partner == 255, N_CLASSES, partner).astype(np.int64)
    return ((((own.astype(np.int64) * (N_CLASSES + 1) + part)
              * (SR1_MAX_DEGREE + 1) + deg.astype(np.int64))
             * SR1_ROW_BANDS) + ROW_PLANE)


def walk_order(mask: np.ndarray) -> np.ndarray:
    """Traverse `mask` as curves: 8-connected components, deterministic DFS (rt1 verbatim)."""
    h, w = mask.shape
    flat = mask.reshape(-1)
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


def binary_entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def open_tokens(path: Path) -> np.memmap:
    n = FRAMES * SEG_H * SEG_W
    if path.stat().st_size != n:
        raise Fo1Error(f"token file {path} is {path.stat().st_size} B, expected {n}")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(FRAMES, SEG_H, SEG_W))


# ============================================================================================
# stage 1 -- reconstruct sr1's waterfill selection from the labels alone
# ============================================================================================
def reconstruct_selection(args: argparse.Namespace) -> dict:
    t0 = time.time()
    tok = open_tokens(args.tokens)
    gt = np.load(args.gt, mmap_mode="r")
    pred = np.load(args.rt1_work / "argmax_base.npy", mmap_mode="r")
    flip_ret = np.load(args.rt1_work / "flip_mask_vs_gt.npy", mmap_mode="r")
    band_ret = np.load(args.rt1_work / "free_band_mask.npy", mmap_mode="r")
    for name, arr in (("gt", gt), ("argmax_base", pred)):
        if arr.shape != (FRAMES, SEG_H, SEG_W):
            raise Fo1Error(f"{name} shape {arr.shape} is not n600 scorer-res")

    band_px = np.zeros(N_CELLS, dtype=np.int64)
    flip_px = np.zeros(N_CELLS, dtype=np.int64)
    tgt_counts = np.zeros((N_CLASSES, N_CLASSES + 1, N_CLASSES), dtype=np.int64)
    total_band = total_flip = 0
    for t in range(args.frames):
        lab = np.asarray(tok[t])
        band = boundary(lab)
        # fail-closed: rt1's retained band mask must BE the label boundary, or the support this
        # arm restricts is not the support sr1 priced.
        if not np.array_equal(band, np.asarray(band_ret[t]).astype(bool)):
            raise Fo1Error(f"rt1 free_band_mask != boundary(labels) at frame {t}")
        flip_full = np.asarray(pred[t]) != np.asarray(gt[t])
        if not np.array_equal(flip_full, np.asarray(flip_ret[t]).astype(bool)):
            raise Fo1Error(f"rt1 flip_mask_vs_gt != (argmax_base != gt) at frame {t}")
        key = cell_key(lab)
        flip = flip_full & band
        band_px += np.bincount(key[band], minlength=N_CELLS)
        flip_px += np.bincount(key[flip], minlength=N_CELLS)
        if flip.any():
            own, partner, _ = cell_features(lab)
            part = np.where(partner == 255, N_CLASSES, partner).astype(np.int64)
            np.add.at(
                tgt_counts,
                (own[flip].astype(np.int64), part[flip],
                 np.asarray(gt[t])[flip].astype(np.int64)),
                1,
            )
        total_band += int(band.sum())
        total_flip += int(flip.sum())
        if (t + 1) % 100 == 0:
            print(f"  [selection] {t + 1}/{args.frames} band {total_band:,} flips {total_flip:,}",
                  flush=True)

    # --- control: the reconstruction must reproduce sr1's retained histograms exactly ---------
    control = {}
    for name, mine in (("cell_band_px", band_px), ("cell_flip_px", flip_px)):
        ref_path = args.sr1_work / f"{name}.npy"
        ref = np.load(ref_path)
        same = bool(np.array_equal(ref, mine))
        control[name] = {
            "sr1_path": str(ref_path), "sr1_sha256": sha256_file(ref_path),
            "byte_identical_to_my_reconstruction": same,
            "sr1_sum": int(ref.sum()), "mine_sum": int(mine.sum()),
        }
        if not same and args.frames == FRAMES:
            raise Fo1Error(f"reconstructed {name} != sr1's retained payload")

    # --- target-class term, sr1's own arithmetic ---------------------------------------------
    tot = tgt_counts.sum(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(tot > 0, tgt_counts / np.maximum(tot, 1), 0.0)
        logp = np.where(prob > 0, np.log2(np.maximum(prob, 1e-300)), 0.0)
    target_bits_total = float(-(tgt_counts * logp).sum())
    target_bits_per_flip = target_bits_total / max(total_flip, 1)

    # --- the marginal waterfill, sr1's own inclusion test -------------------------------------
    live = band_px > 0
    cell_ids = np.flatnonzero(live)
    n_r = band_px[live].astype(np.float64)
    k_r = flip_px[live].astype(np.float64)
    p_r = k_r / n_r
    bits_r = n_r * binary_entropy_bits(p_r)
    eta = args.eta
    marginal_ok = (eta * k_r * SEG_DS_PER_FLIP
                   > (bits_r / 8.0 + k_r * target_bits_per_flip / 8.0) * RATE_DS_PER_BYTE)
    selected = np.sort(cell_ids[marginal_ok])
    sel_flips = float(k_r[marginal_ok].sum())
    sel_band = float(n_r[marginal_ok].sum())
    ideal_mask_B = float((bits_r[marginal_ok] / 8.0).sum())
    ideal_tgt_B = float(sel_flips * target_bits_per_flip / 8.0)
    ideal_total_B = ideal_mask_B + ideal_tgt_B

    if args.frames == FRAMES:
        if selected.size != SR1_IDEAL_CELLS:
            raise Fo1Error(f"reconstructed {selected.size} cells, sr1 retained {SR1_IDEAL_CELLS}")
        if int(sel_flips) != SR1_IDEAL_FLIPS:
            raise Fo1Error(f"reconstructed {int(sel_flips)} flips, sr1 retained {SR1_IDEAL_FLIPS}")
        if abs(ideal_total_B - SR1_IDEAL_BYTES) > 1e-6:
            raise Fo1Error(f"reconstructed ideal {ideal_total_B} B != sr1 {SR1_IDEAL_BYTES} B")

    payloads = {
        "selected_cells": save_array(args.work / "retained" / "selected_cells.npy", selected),
        "cell_band_px": save_array(args.work / "retained" / "cell_band_px.npy", band_px),
        "cell_flip_px": save_array(args.work / "retained" / "cell_flip_px.npy", flip_px),
        "tgt_counts": save_array(args.work / "retained" / "tgt_counts.npy", tgt_counts),
    }
    rec = {
        "schema": "ddm_fo1_selection.v1",
        "axis": "[macOS-CPU advisory] scorer-free -- NEVER a score",
        "score_claim": False,
        "promotable": False,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "selection_eta": eta,
        "cell_definition": "(own class) x (lowest differing 4-neighbour class) x "
                           "(min(degree,4)) x (row band); all four are deterministic functions "
                           "of the TRANSMITTED label field -> receiver recomputes for 0 bytes",
        "receiver_derivability": {
            "own_class": "decoded label at the pixel -- free",
            "partner_class": "lowest differing 4-neighbour decoded label -- free",
            "degree": "distinct decoded labels in the closed 4-neighbourhood, capped at 4 -- free",
            "row_band": "pixel row // 48 -- free",
            "selected_cell_set": f"{selected.size} ids out of {N_CELLS}; a fixed table the "
                                 "encoder must transmit ONCE (priced below), not per pixel",
            "verdict": "every per-pixel factor is label-derived; only the cell SET costs bytes",
        },
        "totals": {"band_px": total_band, "band_flips": total_flip,
                   "cells_live": int(live.sum())},
        "reconstruction_control_vs_sr1": control,
        "target_class": {"bits_per_flip": target_bits_per_flip,
                         "sr1_bits_per_flip": SR1_TARGET_BITS_PER_FLIP,
                         "bits_total": target_bits_total},
        "selection": {
            "cells": int(selected.size),
            "band_px": sel_band,
            "flips": sel_flips,
            "density": sel_flips / max(sel_band, 1.0),
            "ideal_mask_B": ideal_mask_B,
            "ideal_target_B": ideal_tgt_B,
            "ideal_total_B": ideal_total_B,
            "sr1_ideal_total_B": SR1_IDEAL_BYTES,
            "ideal_bits_per_described_flip": ideal_total_B * 8 / max(sel_flips, 1.0),
        },
        "payloads": payloads,
        "frames": args.frames,
        "wall_s": time.time() - t0,
    }
    (args.work / "FO1_SELECTION.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    progress(args.work, "selection-reconstructed", {
        "cells": int(selected.size), "cells_expected": SR1_IDEAL_CELLS,
        "flips": int(sel_flips), "flips_expected": SR1_IDEAL_FLIPS,
        "band_px": int(sel_band), "ideal_total_B": ideal_total_B,
        "sr1_ideal_total_B": SR1_IDEAL_BYTES,
        "histograms_byte_identical_to_sr1": all(
            c["byte_identical_to_my_reconstruction"] for c in control.values()),
    })
    return rec


# ============================================================================================
# stage 2 -- precompute the per-frame restricted support (all label-derived, so free)
# ============================================================================================
def build_frames(args: argparse.Namespace, selected: np.ndarray) -> dict:
    """One pass over the labels producing every per-frame array the coders consume."""
    tok = open_tokens(args.tokens)
    flip_ret = np.load(args.rt1_work / "flip_mask_vs_gt.npy", mmap_mode="r")
    tgt_ret = np.load(args.rt1_work / "flip_target_class.npy", mmap_mode="r")
    sel_lut = np.zeros(N_CELLS, dtype=bool)
    sel_lut[selected] = True

    per: list[dict] = []
    n_sym = n_flip = 0
    for t in range(args.frames):
        lab = np.asarray(tok[t])
        band = boundary(lab)
        sup = band & sel_lut[cell_key(lab)]
        pair = edge_pair_field(lab)
        own, partner, _ = cell_features(lab)
        part = np.where(partner == 255, N_CLASSES, partner).astype(np.uint8)
        truth = (np.asarray(flip_ret[t]).astype(bool) & sup)

        raster = np.flatnonzero(sup.reshape(-1)).astype(np.int64)
        walk_sup = walk_order(sup)
        band_walk = walk_order(band)
        band_walk_sup = band_walk[sup.reshape(-1)[band_walk]]
        tflat = truth.reshape(-1)
        tg = np.asarray(tgt_ret[t]).reshape(-1)
        per.append({
            "raster": raster,
            "walk_sup": walk_sup,
            "walk_band": band_walk_sup,
            "bits_raster": tflat[raster].astype(np.uint8),
            "pair": pair.reshape(-1)[raster].astype(np.uint8),
            "own": lab.reshape(-1)[raster].astype(np.uint8),
            "part": part.reshape(-1)[raster].astype(np.uint8),
            "target_raster": tg[raster].astype(np.uint8),
        })
        n_sym += int(raster.size)
        n_flip += int(tflat[raster].sum())
        if (t + 1) % 100 == 0:
            print(f"  [frames] {t + 1}/{args.frames} symbols {n_sym:,} flips {n_flip:,}",
                  flush=True)
        # every walk traversal must cover exactly the support, or the stream is not the mask
        if walk_sup.size != raster.size or band_walk_sup.size != raster.size:
            raise Fo1Error(f"walk traversal misses support pixels at frame {t}")
    return {"per": per, "n_symbols": n_sym, "n_flips": n_flip}


# ============================================================================================
# stage 3 -- the mask coders.  ONE function drives encode and decode, so a successful decode
#            proves every context is causal.
# ============================================================================================
MASK_MODELS = {
    "order0": {"n_ctx": 1, "order": "raster"},
    "raster": {"n_ctx": N_PAIR * 8, "order": "raster"},
    "walk": {"n_ctx": 8, "order": "walk_sup"},
    "walk_pair": {"n_ctx": N_PAIR * 8, "order": "walk_sup"},
    "bandwalk_pair": {"n_ctx": N_PAIR * 8, "order": "walk_band"},
}


def code_mask(model: str, frames: list[dict], payload: bytes | None = None
              ) -> tuple[bytes, list[np.ndarray], int]:
    """Encode (payload=None) or decode the support-restricted flip stream under `model`."""
    spec = MASK_MODELS[model]
    encoding = payload is None
    enc = RangeEncoder() if encoding else None
    dec = None if encoding else RangeDecoder(payload)
    probs = [P_INIT] * spec["n_ctx"]
    produced: list[np.ndarray] = []
    prev = np.zeros(SEG_H * SEG_W, dtype=np.uint8)
    total = 0

    for fr in frames:
        raster = fr["raster"]
        cur = np.zeros(SEG_H * SEG_W, dtype=np.uint8)
        got = np.zeros(raster.size, dtype=np.uint8)
        pos_of = {}
        if spec["order"] == "raster":
            order = raster
            pair_of = fr["pair"]
        else:
            order = fr[spec["order"]]
            # map flat index -> position in the raster arrays so contexts and truth line up
            pos_of = {int(p): i for i, p in enumerate(raster.tolist())}
            pair_of = fr["pair"]
        truth = fr["bits_raster"]

        if spec["order"] == "raster":
            for i, p in enumerate(order.tolist()):
                if model == "order0":
                    c = 0
                else:
                    y, x = divmod(p, SEG_W)
                    nf = 0
                    if y > 0:
                        nf += int(cur[p - SEG_W])
                        if x > 0:
                            nf += int(cur[p - SEG_W - 1])
                        if x + 1 < SEG_W:
                            nf += int(cur[p - SEG_W + 1])
                    if x > 0:
                        nf += int(cur[p - 1])
                    c = int(pair_of[i]) * 8 + min(nf, 3) * 2 + int(prev[p])
                if encoding:
                    b = int(truth[i])
                    enc.encode_bit(probs[c], b)
                else:
                    b = dec.decode_bit(probs[c])
                probs[c] = adapt(probs[c], b)
                cur[p] = b
                got[i] = b
                total += 1
        else:
            run1 = run2 = run3 = 0
            for p in order.tolist():
                i = pos_of[int(p)]
                run = run1 + run2 + run3
                c = run * 2 + int(prev[p])
                if model in ("walk_pair", "bandwalk_pair"):
                    c = int(pair_of[i]) * 8 + c
                if encoding:
                    b = int(truth[i])
                    enc.encode_bit(probs[c], b)
                else:
                    b = dec.decode_bit(probs[c])
                probs[c] = adapt(probs[c], b)
                cur[p] = b
                got[i] = b
                run3, run2, run1 = run2, run1, b
                total += 1
        produced.append(got)
        prev = cur
    out = enc.finish() if encoding else payload
    return out, produced, total


# ============================================================================================
# stage 4 -- the target-class coder.  sr1's 4,276 B carries a 0.2253 bit/flip target term, so
#            the honest total needs a REAL coder for it too.
# ============================================================================================
# 5 symbols coded as 3 binary decisions on a fixed tree:
#   node0: c < 2 ?      node1 (c<2): c == 0 ?      node2 (c>=2): c == 2 ?   node3: c == 3 ?
TREE_NODES = 4


def code_target(frames: list[dict], mask_fields: list[np.ndarray], contextual: bool,
                payload: bytes | None = None) -> tuple[bytes, list[np.ndarray], int]:
    """Code the correct class at every DECODED flip position.

    Causality: the receiver has already decoded the mask, so it knows exactly which support
    pixels carry a correction before this stream starts.  Context is (own label, partner label),
    both label-derived and therefore free.
    """
    encoding = payload is None
    enc = RangeEncoder() if encoding else None
    dec = None if encoding else RangeDecoder(payload)
    n_ctx = (N_CLASSES * (N_CLASSES + 1)) if contextual else 1
    probs = [[P_INIT] * TREE_NODES for _ in range(n_ctx)]
    produced: list[np.ndarray] = []
    total = 0

    for fr, got in zip(frames, mask_fields, strict=True):
        sel = np.flatnonzero(got)
        own = fr["own"]
        part = fr["part"]
        tgt = fr["target_raster"]
        out = np.zeros(sel.size, dtype=np.uint8)
        for j, i in enumerate(sel.tolist()):
            ctx = (int(own[i]) * (N_CLASSES + 1) + int(part[i])) if contextual else 0
            pr = probs[ctx]
            if encoding:
                v = int(tgt[i])
                if v >= N_CLASSES:
                    raise Fo1Error("flip position carries no target class (255)")
                b0 = 1 if v < 2 else 0
                enc.encode_bit(pr[0], b0)
                pr[0] = adapt(pr[0], b0)
                if b0:
                    b1 = 1 if v == 0 else 0
                    enc.encode_bit(pr[1], b1)
                    pr[1] = adapt(pr[1], b1)
                else:
                    b2 = 1 if v == 2 else 0
                    enc.encode_bit(pr[2], b2)
                    pr[2] = adapt(pr[2], b2)
                    if not b2:
                        b3 = 1 if v == 3 else 0
                        enc.encode_bit(pr[3], b3)
                        pr[3] = adapt(pr[3], b3)
                out[j] = v
            else:
                b0 = dec.decode_bit(pr[0])
                pr[0] = adapt(pr[0], b0)
                if b0:
                    b1 = dec.decode_bit(pr[1])
                    pr[1] = adapt(pr[1], b1)
                    v = 0 if b1 else 1
                else:
                    b2 = dec.decode_bit(pr[2])
                    pr[2] = adapt(pr[2], b2)
                    if b2:
                        v = 2
                    else:
                        b3 = dec.decode_bit(pr[3])
                        pr[3] = adapt(pr[3], b3)
                        v = 3 if b3 else 4
                out[j] = v
            total += 1
        produced.append(out)
    blob = enc.finish() if encoding else payload
    return blob, produced, total


# ============================================================================================
# the race + the adjudication
# ============================================================================================
def race(args: argparse.Namespace) -> int:
    import brotli

    t0 = time.time()
    sel_rec = json.loads((args.work / "FO1_SELECTION.json").read_text())
    selected = np.load(args.work / "retained" / "selected_cells.npy")
    progress(args.work, "frames-building", {"cells": int(selected.size)})
    fb = build_frames(args, selected)
    frames = fb["per"]
    n, ones = fb["n_symbols"], fb["n_flips"]
    if args.frames == FRAMES and ones != SR1_IDEAL_FLIPS:
        raise Fo1Error(f"restricted mask has {ones} flips, sr1 priced {SR1_IDEAL_FLIPS}")
    progress(args.work, "mask-restricted", {
        "support_symbols": n, "support_flips": ones,
        "flips_expected": SR1_IDEAL_FLIPS,
        "support_density": ones / max(n, 1),
        "full_band_symbols": sel_rec["totals"]["band_px"],
        "support_share_of_band": n / max(sel_rec["totals"]["band_px"], 1),
    })

    raster_bits = np.concatenate([fr["bits_raster"] for fr in frames])
    truth_fields = [fr["bits_raster"] for fr in frames]
    packed = np.packbits(raster_bits).tobytes()
    p1 = ones / max(n, 1)

    results: list[dict] = []
    payloads: dict[str, bytes] = {}
    verified_fields: dict[str, list[np.ndarray]] = {}

    def add(tag: str, name: str, blob: bytes, verified: bool, note: str = "",
            preregistered: bool = True) -> None:
        payloads[tag] = blob
        results.append({
            "tag": tag, "coder": name, "bytes": len(blob), "sha256": sha256_bytes(blob),
            "roundtrip_verified": verified,
            "bits_per_described_flip": len(blob) * 8 / max(ones, 1),
            "preregistered_M0_M7": preregistered, "note": note,
        })

    def unpacks_to_source(blob: bytes) -> bool:
        return bool(np.array_equal(
            np.unpackbits(np.frombuffer(blob, dtype=np.uint8))[:n], raster_bits))

    add("M0", "raw packed support bits", packed, unpacks_to_source(packed), "no coder")
    b1 = brotli.compress(packed, quality=11)
    add("M1", "brotli(packed) q11", b1, unpacks_to_source(brotli.decompress(b1)))
    b2 = lzma.compress(packed, preset=9 | lzma.PRESET_EXTREME)
    add("M2", "lzma(packed) preset 9|EXTREME", b2, unpacks_to_source(lzma.decompress(b2)))

    sp = max(1, min(P_MAX - 1, round((1 - p1) * P_MAX)))
    e = RangeEncoder()
    for b in raster_bits.tolist():
        e.encode_bit(sp, b)
    pay = e.finish()
    d = RangeDecoder(pay)
    back = np.fromiter((d.decode_bit(sp) for _ in range(n)), dtype=np.uint8, count=n)
    add("M3", "static binary AC (i.i.d. realized)", pay,
        bool(np.array_equal(back, raster_bits)),
        f"P(flip)={p1:.7f} quantised to {P_MAX - sp}/2048; +2 B header not counted (rt1 parity)")

    for tag, model, name, prereg in (
        ("M4", "order0", "adaptive binary AC, order-0", True),
        ("M5", "raster", "CABAC raster (pair x causal-neighbours x temporal)", True),
        ("M6", "walk", "CABAC support-walk (run x temporal)", True),
        ("M7", "walk_pair", "CABAC support-walk (pair x run x temporal)", True),
        ("M8", "bandwalk_pair", "CABAC FULL-BAND-walk order, support-only symbols "
                                "(pair x run x temporal)", False),
    ):
        blob, enc_fields, cnt = code_mask(model, frames)
        _, dec_fields, _ = code_mask(model, frames, payload=blob)
        ok = cnt == n and all(
            np.array_equal(a, b) for a, b in zip(enc_fields, dec_fields, strict=True))
        if ok:
            ok = all(np.array_equal(a, b)
                     for a, b in zip(dec_fields, truth_fields, strict=True))
        add(tag, name, blob, ok, f"{MASK_MODELS[model]['n_ctx']} contexts", prereg)
        if ok:
            # ONLY a field the decoder actually produced goes in here.  Seeding this dict from
            # the truth field for the non-arithmetic coders would make the retained "round-trip
            # decode" artifact circular -- it would prove nothing about any decoder.
            verified_fields[tag] = dec_fields

    for r in results:
        if not r["roundtrip_verified"]:
            raise Fo1Error(f"{r['tag']} {r['coder']} did not decode back to the source field")

    # --- the target-class leg -----------------------------------------------------------------
    tgt_results: list[dict] = []
    tgt_payloads: dict[str, bytes] = {}
    tgt_truth = [fr["target_raster"][np.flatnonzero(fr["bits_raster"])] for fr in frames]
    flat_tgt = np.concatenate(tgt_truth) if ones else np.zeros(0, dtype=np.uint8)

    t_bits = np.unpackbits(flat_tgt.reshape(-1, 1), axis=1, count=3,
                           bitorder="little").reshape(-1)
    t_raw = np.packbits(t_bits).tobytes()
    t_back = np.packbits(
        np.unpackbits(np.frombuffer(t_raw, dtype=np.uint8))[:3 * ones].reshape(-1, 3),
        axis=1, bitorder="little").reshape(-1) if ones else np.zeros(0, dtype=np.uint8)
    tgt_payloads["T0"] = t_raw
    tgt_results.append({
        "tag": "T0", "coder": "raw 3 bits/flip packed", "bytes": len(t_raw),
        "sha256": sha256_bytes(t_raw),
        "roundtrip_verified": bool(np.array_equal(t_back, flat_tgt)),
        "bits_per_flip": len(t_raw) * 8 / max(ones, 1), "note": "no coder",
    })

    for tag, contextual, name in (
        ("T1", False, "adaptive binary-tree AC, order-0"),
        ("T2", True, "adaptive binary-tree AC, context = (own, partner) [both free]"),
    ):
        blob, enc_t, cnt = code_target(frames, truth_fields, contextual)
        _, dec_t, _ = code_target(frames, truth_fields, contextual, payload=blob)
        ok = cnt == ones and all(
            np.array_equal(a, b) for a, b in zip(enc_t, dec_t, strict=True))
        if ok:
            ok = all(np.array_equal(a, b) for a, b in zip(dec_t, tgt_truth, strict=True))
        tgt_payloads[tag] = blob
        tgt_results.append({
            "tag": tag, "coder": name, "bytes": len(blob), "sha256": sha256_bytes(blob),
            "roundtrip_verified": ok, "bits_per_flip": len(blob) * 8 / max(ones, 1),
            "note": f"{(N_CLASSES * (N_CLASSES + 1)) if contextual else 1} contexts x "
                    f"{TREE_NODES} tree nodes",
        })
    for r in tgt_results:
        if not r["roundtrip_verified"]:
            raise Fo1Error(f"{r['tag']} {r['coder']} did not decode back to the target field")

    progress(args.work, "roundtrip-verified", {
        "mask_coders": len(results), "target_coders": len(tgt_results),
        "all_verified": True,
    })

    # --- retain every payload (ALWAYS KEEP THE PAYLOAD) ---------------------------------------
    out = args.work / "retained"
    retained = {}
    for tag, blob in payloads.items():
        retained[f"mask_{tag}"] = save_blob(out / f"mask_{tag}.bin", blob)
    for tag, blob in tgt_payloads.items():
        retained[f"target_{tag}"] = save_blob(out / f"target_{tag}.bin", blob)
    retained["restricted_mask_bits"] = save_array(
        out / "restricted_mask_bits.npy", raster_bits)
    retained["restricted_support_index"] = save_array(
        out / "restricted_support_index.npy",
        np.concatenate([fr["raster"] for fr in frames]).astype(np.int32))
    retained["restricted_support_frame_offsets"] = save_array(
        out / "restricted_support_frame_offsets.npy",
        np.cumsum([0] + [int(fr["raster"].size) for fr in frames]).astype(np.int64))
    retained["target_values"] = save_array(out / "target_values.npy", flat_tgt)
    # The round-trip decode itself, so the verification is auditable from disk and not from a
    # flag: this file must be byte-identical to `restricted_mask_bits.npy`.  Sourced only from a
    # coder that produced it by DECODING (M4-M8), never from the truth field.
    best_mask = min(results, key=lambda r: r["bytes"])
    best_decoded_tag = min(
        (r for r in results if r["tag"] in verified_fields), key=lambda r: r["bytes"])["tag"]
    retained["roundtrip_decoded_mask_best"] = save_array(
        out / "roundtrip_decoded_mask_best.npy",
        np.concatenate(verified_fields[best_decoded_tag]))
    retained["roundtrip_decoded_mask_best"]["decoded_by"] = best_decoded_tag

    # --- the adjudication ---------------------------------------------------------------------
    best_prereg = min((r for r in results if r["preregistered_M0_M7"]), key=lambda r: r["bytes"])
    best_tgt = min(tgt_results, key=lambda r: r["bytes"])
    total_prereg = best_prereg["bytes"] + best_tgt["bytes"]
    total_overall = best_mask["bytes"] + best_tgt["bytes"]

    def net_dS(bytes_total: float, eta: float) -> float:
        return -eta * ones * SEG_DS_PER_FLIP + bytes_total * RATE_DS_PER_BYTE

    ideal = sel_rec["selection"]["ideal_total_B"]
    verdict = {
        "bar_B": BAR_B,
        "bar_provenance": "pn2 sec5/sec6: eta_projected_pooled_n12 = 0.6111 x 6512 flips x "
                          "1.273108 B per flip = 5065.6 B -- FROZEN, pre-registered before "
                          "this run, not recomputed here",
        "real_total_B_preregistered_M0_M7": total_prereg,
        "real_total_B_including_M8": total_overall,
        "ideal_total_B_sr1": ideal,
        "real_over_ideal_preregistered": total_prereg / ideal,
        "real_over_ideal_including_M8": total_overall / ideal,
        "headroom_available_pct": 100.0 * (BAR_B / ideal - 1.0),
        "meets_bar_preregistered_M0_M7": total_prereg <= BAR_B,
        "meets_bar_including_M8": total_overall <= BAR_B,
        "VERDICT": ("REAL SUPPLIER -- waterfilled channel clears the pre-registered bar"
                    if total_overall <= BAR_B else
                    "ARTIFACT -- sr1's waterfill S is an ideal-entropy artifact; rt1's CLOSED "
                    "verdict is restored on better evidence"),
        "eta_required_for_break_even_real": (
            total_overall * RATE_DS_PER_BYTE / (ones * SEG_DS_PER_FLIP)),
        "net_dS_at_eta": {
            f"{PN2_ETA_PROJECTED:.4f}": net_dS(total_overall, PN2_ETA_PROJECTED),
            f"{SR1_ETA_SELECTION:.4f}": net_dS(total_overall, SR1_ETA_SELECTION),
            f"{PN2_ETA_UNPROJECTED:.4f}": net_dS(total_overall, PN2_ETA_UNPROJECTED),
            "1.0000": net_dS(total_overall, 1.0),
        },
        "net_dS_ideal_at_eta": {
            f"{PN2_ETA_PROJECTED:.4f}": net_dS(ideal, PN2_ETA_PROJECTED),
            f"{SR1_ETA_SELECTION:.4f}": net_dS(ideal, SR1_ETA_SELECTION),
        },
        "share_of_gap_if_supplier": -net_dS(total_overall, PN2_ETA_PROJECTED) / GAP_S,
    }
    # the cell-set side info: the ONE thing in this channel the receiver cannot derive
    cell_set_bits = float(selected.size * np.log2(N_CELLS))
    verdict["cell_set_side_info_B"] = {
        "naive_index_list_B": cell_set_bits / 8.0,
        "bitmap_1200_cells_B": N_CELLS / 8.0,
        "note": "a fixed table transmitted ONCE for the whole clip; sr1 excluded it, so the "
                "bar comparison above also excludes it. Adding the cheaper of the two moves "
                "the total by this much and is reported so the reader can re-adjudicate.",
        "total_with_cheapest_side_info_B": total_overall + min(cell_set_bits / 8.0, N_CELLS / 8.0),
        "meets_bar_with_side_info": (
            total_overall + min(cell_set_bits / 8.0, N_CELLS / 8.0)) <= BAR_B,
    }

    rec = {
        "schema": "ddm_fo1_waterfill_real_coder.v1",
        "axis": "[macOS-CPU advisory] scorer-free -- NEVER a score",
        "score_claim": False,
        "promotable": False,
        "verdict_scope": "formulation -- the M0-M8 coder family on sr1's 41-cell waterfilled "
                         "support of rt1's free label boundary at n600. A better coder on this "
                         "same support could move the number; nothing here bounds the family.",
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "support": {
            "cells": int(selected.size),
            "symbols": n,
            "flips": ones,
            "density": p1,
            "full_band_symbols": sel_rec["totals"]["band_px"],
            "full_band_flips": sel_rec["totals"]["band_flips"],
        },
        "exchange_rates_reused_not_rederived": {
            "seg_dS_per_flip": SEG_DS_PER_FLIP,
            "rate_dS_per_byte": RATE_DS_PER_BYTE,
            "bytes_per_flip_bar": BYTES_PER_FLIP_BAR,
        },
        "mask_results": sorted(results, key=lambda r: r["bytes"]),
        "target_results": sorted(tgt_results, key=lambda r: r["bytes"]),
        "best_mask_preregistered_M0_M7": best_prereg,
        "best_mask_overall": best_mask,
        "best_target": best_tgt,
        "verdict": verdict,
        "retained_payloads": retained,
        "frames": args.frames,
        "wall_s": time.time() - t0,
    }
    (args.work / "FO1_CODER_RACE.json").write_text(
        json.dumps(rec, indent=2, sort_keys=True) + "\n")
    progress(args.work, "bar-adjudicated", {
        "best_mask_preregistered": [best_prereg["tag"], best_prereg["bytes"]],
        "best_mask_overall": [best_mask["tag"], best_mask["bytes"]],
        "best_target": [best_tgt["tag"], best_tgt["bytes"]],
        "real_total_B": total_overall, "bar_B": BAR_B, "ideal_B": ideal,
        "VERDICT": verdict["VERDICT"],
    })
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["selection", "race", "all"], default="all")
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--rt1-work", type=Path, default=DEFAULT_RT1_WORK)
    ap.add_argument("--sr1-work", type=Path, default=DEFAULT_SR1_WORK)
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--eta", type=float, default=SR1_ETA_SELECTION,
                    help="eta defining the waterfill SELECTION (sr1 used 0.6235)")
    args = ap.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage in ("selection", "all"):
        reconstruct_selection(args)
    if args.stage in ("race", "all"):
        return race(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
