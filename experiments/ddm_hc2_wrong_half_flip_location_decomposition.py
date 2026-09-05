"""ddm_hc2 — decompose the "WHERE are the flips" half of the shipped token stream.

Closed-form ceiling ONLY. No coder is built; every ceiling number here is
REFUSAL-ONLY (`score_claim=false`, `promotable=false`).

Object: the fs2 body's shipped HPAC token stream (113,411 B, sha
`5601d6fd...`), whose per-position coding rows were retained byte-identically
by `ddm_mc1`.

hc1 (`token-stream-is-one-binary-question`) split the stream losslessly:

    -log2(p_sel) = -log2(pmax)                            argmax right
                 = -log2(1-pmax) + -log2(p_sel/(1-pmax))  argmax wrong

and measured 97.80% of the stream to be the binary INDICATOR. The "no" branch
-- ~227,671 flips x 2.6917 bits ~ 76,600 B -- is what MAIN calls the cost of
saying WHERE the flips are, and it has never been decomposed.

This module asks: is there a representation of the flip LOCATION SET that beats
per-site indicator coding by >= 5,000 B?  Three representations are priced
(all cross-fitted, pair-level two-fold, 3 seeds), plus a family bound:

  (a) component code   -- component count + seed coordinates + per-component
                          shape (dictionary / run-length), two seed codes:
                          raster-gap (a1) and mixer-conditioned (a2)
  (b) boundary-offset  -- flips as signed offsets of the class boundary the
                          mixer's argmax draws (band derivable from the field,
                          zero side info beyond the offsets)
  (c) incumbent        -- the rows' own per-site sum (two readings: the "no"
                          branch alone, and the full indicator it must replace)
  (d) family bound     -- mi1's `q' = sigma(logit(1-pmax) + beta_cell)` with the
                          causal-neighbourhood flip pattern as the cell. Any
                          representation that exploits only local clustering of
                          flips induces a conditional law inside this family, so
                          (d) upper-bounds (a) and (b) at its context resolution.

Axis: `[macOS-CPU advisory / scorer-free EXACT byte measurement]` for the rows
control; `[model-ledger code length on the coder's own rows; REFUSAL-ONLY]` for
every ceiling number.

Stages: control | features | ceiling | verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# Canonical inputs (mc1 custody; see ddm_mc1_motion_compensated_previous_plane_20260904.md §8)
# ---------------------------------------------------------------------------

ROWS_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_mc1_motion_compensated_previous_plane/rows")
ROWS_PATH = ROWS_DIR / "coding_rows.f32.npy"
ARGMAX_PATH = ROWS_DIR / "base_argmax.u8.npy"
BUCKET_PATH = ROWS_DIR / "boundary_bucket.u8.npy"
CONTROL_STREAM = Path(
    "/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane/rows/control_stream.bin"
)
TOKENS_PATH = Path(
    "/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/identity_v1/out/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
ARCHIVE_PATH = Path(
    "/Volumes/VertigoDataTier/pact/ddm_fs2_carrier_resolve/fire_runtime_D_alternation/archive.zip"
)

EXPECTED = {
    "rows_sha256": "35ec67ca932112cfe11be31391ee784cb577bc6c7df1e0563f49f841fded67bf",
    "argmax_sha256": "5786bc245844aa2bec017755bdc71e415b423cca77e55b43d0ff2c6d6c132f34",
    "bucket_sha256": "4ec0010db23350347b8c01fedb9f9fbca76d758f733be6cfe7cf3dbba917c01e",
    "tokens_sha256": "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
    "stream_sha256": "5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3",
    "archive_sha256": "a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6",
    "stream_bytes": 113411,
    "code_bytes_ideal": 113410.85566696088,
    "live_positions": 50_009_121,
}

N_PAIRS = 600
H, W = 384, 512
N_POS = H * W  # 196,608
BITS_PER_BYTE = 8.0

STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
STRUCT8 = np.ones((3, 3), dtype=bool)

# raster-causal neighbour offsets (dy, dx); dy<0, or dy==0 and dx<0
CAUSAL4 = ((-1, -1), (-1, 0), (-1, 1), (0, -1))
CAUSAL8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (-2, 0), (-1, -2), (-1, 2), (0, -2))

SEEDS = (20260905, 777, 31337)

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def peak_rss_gib() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    if sys.platform == "darwin":
        return ru / (1024.0**3)
    return ru / (1024.0**2)


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_jsonable))
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(type(o)))


def bits_to_bytes(bits: float) -> float:
    return bits / BITS_PER_BYTE


# ---------------------------------------------------------------------------
# per-pair primitive
# ---------------------------------------------------------------------------


@dataclass
class PairArrays:
    """Everything derived from one pair's coding rows."""

    argmax: np.ndarray  # (N_POS,) uint8
    token: np.ndarray  # (N_POS,) uint8
    flip: np.ndarray  # (N_POS,) bool
    logit_flip: np.ndarray  # (N_POS,) float64  log(rest/pmax)
    ind_bits: np.ndarray  # (N_POS,) float64 incumbent indicator bits
    cond_bits: np.ndarray  # (N_POS,) float64 "which class" bits (0 off flips)
    live: np.ndarray  # (N_POS,) bool  pmax < 1.0 in float32
    d_other: np.ndarray  # (N_POS,) int16 chebyshev dist to nearest other-class argmax pixel
    d_token: np.ndarray  # (N_POS,) int16 chebyshev dist to nearest argmax pixel of the token class
    pat4: np.ndarray  # (N_POS,) uint8
    pat8: np.ndarray  # (N_POS,) uint16


def load_pair(rows_mm, argmax_mm, tokens_mm, p: int) -> PairArrays:
    """Derive one pair's flip set from the coding rows.

    The flip definition is hc1's, taken tie-safely on the CODING ROW (not on
    mc1's retained `base_argmax`, which is the argmax of the pre-corrector
    logits and therefore differs at 0.028 % of positions): a position is a flip
    iff the coded token's probability is strictly below the row maximum. That
    is the definition under which hc1's lossless split
    `-log2(p_sel) = -log2(pmax) [+ -log2(1-pmax) + -log2(p_sel/(1-pmax))]`
    is exact.
    """
    row = np.asarray(rows_mm[p], dtype=np.float64)  # (N_POS, 5)
    t = np.asarray(tokens_mm[p * N_POS : (p + 1) * N_POS])

    idx = np.arange(N_POS)
    s = row.sum(axis=1)
    pmax = row.max(axis=1)
    a = row.argmax(axis=1).astype(np.uint8)
    rest = s - pmax
    np.maximum(rest, 0.0, out=rest)

    flip = row[idx, t] < pmax
    # float32 saturation: pmax == 1.0 exactly -> indicator is free and no flip is representable
    pmax32 = np.asarray(rows_mm[p], dtype=np.float32).max(axis=1)
    live = pmax32 < np.float32(1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ind_bits = np.where(flip, -np.log2(rest / s), -np.log2(pmax / s))
        cond_bits = np.where(flip, -np.log2(np.maximum(row[idx, t], 1e-300) / np.maximum(rest, 1e-300)), 0.0)
        logit_flip = np.log(np.maximum(rest, 1e-300)) - np.log(np.maximum(pmax, 1e-300))
    ind_bits = np.where(np.isfinite(ind_bits), ind_bits, 0.0)
    cond_bits = np.where(np.isfinite(cond_bits), cond_bits, 0.0)

    A = a.reshape(H, W)
    # chessboard distance to the nearest argmax pixel of each class
    dmaps = np.empty((5, H, W), dtype=np.int32)
    for c in range(5):
        mask = c == A
        if mask.all():
            dmaps[c] = 0
        elif not mask.any():
            dmaps[c] = 1 << 20
        else:
            dmaps[c] = ndimage.distance_transform_cdt(~mask, metric="chessboard")
    d_token = dmaps[t.reshape(H, W), np.arange(H)[:, None], np.arange(W)[None, :]].reshape(-1)
    dstack = dmaps.copy()
    dstack[A, np.arange(H)[:, None], np.arange(W)[None, :]] = 1 << 20  # exclude own class
    d_other = dstack.min(axis=0).reshape(-1)

    fm = flip.reshape(H, W)
    pat4 = np.zeros((H, W), dtype=np.uint8)
    for bit, (dy, dx) in enumerate(CAUSAL4):
        pat4 |= (_shift(fm, dy, dx).astype(np.uint8)) << bit
    pat8 = np.zeros((H, W), dtype=np.uint16)
    for bit, (dy, dx) in enumerate(CAUSAL8):
        pat8 |= (_shift(fm, dy, dx).astype(np.uint16)) << bit

    return PairArrays(
        argmax=a,
        token=t,
        flip=flip,
        logit_flip=logit_flip,
        ind_bits=ind_bits,
        cond_bits=cond_bits,
        live=live,
        d_other=np.minimum(d_other, 32767).astype(np.int16),
        d_token=np.minimum(d_token, 32767).astype(np.int16),
        pat4=pat4.reshape(-1),
        pat8=pat8.reshape(-1),
    )


def _shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """m shifted so that out[r, c] == m[r + dy, c + dx] (zero outside)."""
    out = np.zeros_like(m)
    ys_dst = slice(max(0, -dy), H - max(0, dy))
    ys_src = slice(max(0, dy), H - max(0, -dy))
    xs_dst = slice(max(0, -dx), W - max(0, dx))
    xs_src = slice(max(0, dx), W - max(0, -dx))
    out[ys_dst, xs_dst] = m[ys_src, xs_src]
    return out


def components(flip: np.ndarray, structure: np.ndarray):
    """Label the flip mask; return (labels, n, sizes, seeds, first-index order)."""
    lab, n = ndimage.label(flip.reshape(H, W), structure=structure)
    lab = lab.reshape(-1)
    if n == 0:
        return lab, 0, np.zeros(0, np.int64), np.zeros(0, np.int64)
    sizes = np.bincount(lab, minlength=n + 1)[1:]
    order = np.argsort(lab, kind="stable")
    lab_sorted = lab[order]
    starts = np.searchsorted(lab_sorted, np.arange(1, n + 1), side="left")
    seeds = order[starts]  # first raster position of each label
    return lab, n, sizes, seeds


# ---------------------------------------------------------------------------
# STAGE control
# ---------------------------------------------------------------------------


def stage_control(out: Path, quick: bool) -> dict:
    t0 = time.time()
    res = {
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "promotable": False,
        "schema": "ddm_hc2_control.v1",
    }
    shas = {}
    for name, path in (
        ("control_stream", CONTROL_STREAM),
        ("argmax", ARGMAX_PATH),
        ("bucket", BUCKET_PATH),
        ("tokens", TOKENS_PATH),
        ("rows", ROWS_PATH),
    ):
        if quick and name in {"rows", "tokens", "argmax", "bucket"}:
            continue
        shas[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    res["sha256"] = shas
    res["sha_match"] = {
        k: (shas[k]["sha256"] == EXPECTED[f"{k}_sha256"]) if f"{k}_sha256" in EXPECTED else None
        for k in shas
    }
    res["sha_match"]["control_stream"] = shas["control_stream"]["sha256"] == EXPECTED["stream_sha256"]

    # byte-identity of the retained re-encode against the SHIPPED archive block
    p = zipfile.ZipFile(ARCHIVE_PATH).read("p")
    ctl = CONTROL_STREAM.read_bytes()
    off = p.find(ctl)
    res["byte_identical_reencode"] = {
        "archive": str(ARCHIVE_PATH),
        "archive_sha256_ok": sha256_file(ARCHIVE_PATH) == EXPECTED["archive_sha256"],
        "member": "p",
        "member_bytes": len(p),
        "stream_bytes": len(ctl),
        "offset_in_member": int(off),
        "verified": bool(off >= 0 and len(ctl) == EXPECTED["stream_bytes"]),
    }

    # streamed reconstruction of the ideal code length from the rows
    rows_mm = np.load(ROWS_PATH, mmap_mode="r")
    argmax_mm = np.load(ARGMAX_PATH, mmap_mode="r")
    tokens_mm = np.memmap(TOKENS_PATH, dtype=np.uint8, mode="r")
    npairs = 20 if quick else N_PAIRS
    total_bits = 0.0
    live = 0
    flips = 0
    sat_flips = 0
    argmax_ok = True
    idx = np.arange(N_POS)
    base_disagree = 0
    for p_i in range(npairs):
        row = np.asarray(rows_mm[p_i], dtype=np.float64)
        a_base = np.asarray(argmax_mm[p_i])
        t = np.asarray(tokens_mm[p_i * N_POS : (p_i + 1) * N_POS])
        a_row = row.argmax(axis=1).astype(np.uint8)
        base_disagree += int((a_row != a_base).sum())
        s = row.sum(axis=1)
        psel = row[idx, t] / s
        total_bits += float(-np.log2(psel).sum())
        pmax32 = np.asarray(rows_mm[p_i], dtype=np.float32).max(axis=1)
        lv = pmax32 < np.float32(1.0)
        live += int(lv.sum())
        f = row[idx, t] < row.max(axis=1)
        flips += int(f.sum())
        sat_flips += int((f & ~lv).sum())
    argmax_ok = base_disagree == 0
    res["reconstruction"] = {
        "pairs": npairs,
        "code_bits": total_bits,
        "code_bytes_ideal": bits_to_bytes(total_bits),
        "expected_full_bytes": EXPECTED["code_bytes_ideal"] if not quick else None,
        "matches_expected": (
            abs(bits_to_bytes(total_bits) - EXPECTED["code_bytes_ideal"]) < 0.01 if not quick else None
        ),
        "coding_row_argmax_equals_mc1_base_argmax": argmax_ok,
        "base_argmax_disagreements": base_disagree,
        "base_argmax_disagreement_rate": base_disagree / max(npairs * N_POS, 1),
        "live_positions": live,
        "expected_live_positions": EXPECTED["live_positions"] if not quick else None,
        "flips": flips,
        "saturated_flips": sat_flips,
    }
    res["elapsed_seconds"] = time.time() - t0
    res["peak_rss_gib"] = peak_rss_gib()
    atomic_write_json(out / ("CONTROL_QUICK.json" if quick else "CONTROL.json"), res)
    return res


# ---------------------------------------------------------------------------
# STAGE features
# ---------------------------------------------------------------------------


@dataclass
class FeatureStore:
    n_live: int = 0
    logit: np.ndarray = field(default=None)
    ind_bits: np.ndarray = field(default=None)
    flip: np.ndarray = field(default=None)
    pair: np.ndarray = field(default=None)
    pat4: np.ndarray = field(default=None)
    pat8: np.ndarray = field(default=None)
    d_other: np.ndarray = field(default=None)
    argmax: np.ndarray = field(default=None)
    is_seed: np.ndarray = field(default=None)
    is_covered: np.ndarray = field(default=None)


def stage_features(out: Path, npairs: int, keep: bool, touch_alloc: bool = False) -> dict:
    t0 = time.time()
    rows_mm = np.load(ROWS_PATH, mmap_mode="r")
    argmax_mm = np.load(ARGMAX_PATH, mmap_mode="r")
    tokens_mm = np.memmap(TOKENS_PATH, dtype=np.uint8, mode="r")

    cap = EXPECTED["live_positions"] + 1024
    fs = FeatureStore()
    fs.logit = np.empty(cap, np.float32)
    fs.ind_bits = np.empty(cap, np.float32)
    fs.flip = np.empty(cap, np.uint8)
    fs.pair = np.empty(cap, np.uint16)
    fs.pat4 = np.empty(cap, np.uint8)
    fs.pat8 = np.empty(cap, np.uint16)
    fs.d_other = np.empty(cap, np.int16)
    fs.argmax = np.empty(cap, np.uint8)
    fs.is_seed = np.empty(cap, np.uint8)
    fs.is_covered = np.empty(cap, np.uint8)
    if touch_alloc:
        # a dry pass must resident-fault every accumulator page, else its peak RSS
        # understates the full run (np.empty does not touch pages).
        for _n in (
            "logit", "ind_bits", "flip", "pair", "pat4", "pat8", "d_other",
            "argmax", "is_seed", "is_covered",
        ):
            getattr(fs, _n)[:] = 0

    # component tables (8-conn is the primary; 4-conn recorded for the charter)
    comp8 = {
        "pair": [],
        "size": [],
        "seed": [],
        "cost_bits": [],
        "seed_cost_bits": [],
        "bh": [],
        "bw": [],
        "shape_id": [],
    }
    comp4_sizes = Counter()
    comp4_cost = Counter()
    shape_dict: dict[bytes, int] = {}
    shape_keys: list[bytes] = []

    totals = {
        "ind_bits": 0.0,
        "cond_bits": 0.0,
        "no_branch_bits": 0.0,
        "yes_branch_bits": 0.0,
        "flips": 0,
        "positions": 0,
        "live": 0,
        "d_token_hist": Counter(),
        "band_flip_in": Counter(),
        "band_positions": Counter(),
        "band_ind_bits_out": Counter(),
    }
    per_pair = []
    seed_gaps: list[int] = []
    last_seed_global = -1

    w = 0
    for p_i in range(npairs):
        pa = load_pair(rows_mm, argmax_mm, tokens_mm, p_i)
        f = pa.flip
        totals["positions"] += N_POS
        totals["live"] += int(pa.live.sum())
        totals["flips"] += int(f.sum())
        nb = float(pa.ind_bits[f].sum())
        yb = float(pa.ind_bits[~f].sum())
        totals["no_branch_bits"] += nb
        totals["yes_branch_bits"] += yb
        totals["ind_bits"] += nb + yb
        totals["cond_bits"] += float(pa.cond_bits.sum())

        # boundary-offset diagnostics
        dt = pa.d_token[f]
        for v, c in zip(*np.unique(np.minimum(dt, 8), return_counts=True), strict=True):
            totals["d_token_hist"][int(v)] += int(c)
        for D in (1, 2, 3):
            band = pa.d_other <= D
            totals["band_positions"][D] += int(band.sum())
            totals["band_flip_in"][D] += int((f & band).sum())
            totals["band_ind_bits_out"][D] += float(pa.ind_bits[~band].sum())

        lab8, n8, sizes8, seeds8 = components(f, STRUCT8)
        _, n4, sizes4, seeds4 = components(f, STRUCT4)
        # per-component cost sums
        if n8:
            cost8 = np.bincount(lab8, weights=pa.ind_bits, minlength=n8 + 1)[1:]
            ys, xs = np.divmod(np.nonzero(f)[0], W)
            labs = lab8[f]
            order = np.argsort(labs, kind="stable")
            labs_s = labs[order]
            ys_s, xs_s = ys[order], xs[order]
            starts = np.searchsorted(labs_s, np.arange(1, n8 + 1), side="left")
            ends = np.searchsorted(labs_s, np.arange(1, n8 + 1), side="right")
            for k in range(n8):
                a_, b_ = starts[k], ends[k]
                yy = ys_s[a_:b_]
                xx = xs_s[a_:b_]
                y0, x0 = int(yy.min()), int(xx.min())
                key = (yy - yy[0]).astype(np.int16).tobytes() + b"|" + (xx - xx[0]).astype(np.int16).tobytes()
                sid = shape_dict.get(key)
                if sid is None:
                    sid = len(shape_keys)
                    shape_dict[key] = sid
                    shape_keys.append(key)
                comp8["shape_id"].append(sid)
                comp8["bh"].append(int(yy.max() - y0 + 1))
                comp8["bw"].append(int(xx.max() - x0 + 1))
            comp8["pair"].extend([p_i] * n8)
            comp8["size"].extend(sizes8.tolist())
            comp8["seed"].extend(seeds8.tolist())
            comp8["cost_bits"].extend(cost8.tolist())
            comp8["seed_cost_bits"].extend(pa.ind_bits[seeds8].tolist())
        if n4:
            cost4 = np.bincount(_relabel(f, STRUCT4), weights=pa.ind_bits, minlength=n4 + 1)[1:]
            for s_, c_ in zip(sizes4.tolist(), cost4.tolist(), strict=True):
                comp4_sizes[s_] += 1
                comp4_cost[s_] += c_

        # seed / covered flags on the full field (8-conn, raster decode order)
        is_seed_full = np.zeros(N_POS, np.uint8)
        is_cov_full = np.zeros(N_POS, np.uint8)
        if n8:
            is_seed_full[seeds8] = 1
            is_cov_full[f] = 1
            is_cov_full[seeds8] = 0
            g = seeds8 + p_i * N_POS
            g.sort()
            gaps = np.diff(np.concatenate(([last_seed_global], g)))
            seed_gaps.extend(gaps.tolist())
            last_seed_global = int(g[-1])

        lv = pa.live
        k = int(lv.sum())
        sl = slice(w, w + k)
        fs.logit[sl] = pa.logit_flip[lv].astype(np.float32)
        fs.ind_bits[sl] = pa.ind_bits[lv].astype(np.float32)
        fs.flip[sl] = f[lv].astype(np.uint8)
        fs.pair[sl] = p_i
        fs.pat4[sl] = pa.pat4[lv]
        fs.pat8[sl] = pa.pat8[lv]
        fs.d_other[sl] = pa.d_other[lv]
        fs.argmax[sl] = pa.argmax[lv]
        fs.is_seed[sl] = is_seed_full[lv]
        fs.is_covered[sl] = is_cov_full[lv]
        w += k

        per_pair.append(
            {
                "pair": p_i,
                "flips": int(f.sum()),
                "no_branch_bits": nb,
                "yes_branch_bits": yb,
                "n_comp8": int(n8),
                "n_comp4": int(n4),
            }
        )
        if (p_i + 1) % 50 == 0:
            print(
                f"[features] pair {p_i + 1}/{npairs} live={w} flips={totals['flips']} "
                f"peak_rss={peak_rss_gib():.2f} GiB t={time.time() - t0:.0f}s",
                flush=True,
            )

    fs.n_live = w
    for name in ("logit", "ind_bits", "flip", "pair", "pat4", "pat8", "d_other", "argmax", "is_seed", "is_covered"):
        setattr(fs, name, getattr(fs, name)[:w])

    comp = {k: np.asarray(v) for k, v in comp8.items()}
    summary = {
        "axis": "[model-ledger code length on the coder's own rows; REFUSAL-ONLY]",
        "score_claim": False,
        "promotable": False,
        "schema": "ddm_hc2_features.v1",
        "pairs": npairs,
        "positions": totals["positions"],
        "live_positions": int(w),
        "flips": totals["flips"],
        "indicator_bytes": bits_to_bytes(totals["ind_bits"]),
        "no_branch_bytes": bits_to_bytes(totals["no_branch_bits"]),
        "yes_branch_bytes": bits_to_bytes(totals["yes_branch_bits"]),
        "conditional_bytes": bits_to_bytes(totals["cond_bits"]),
        "stream_bytes_ideal": bits_to_bytes(totals["ind_bits"] + totals["cond_bits"]),
        "bits_per_flip_no_branch": totals["no_branch_bits"] / max(totals["flips"], 1),
        "n_components_8": len(comp["size"]),
        "n_components_4": int(sum(comp4_sizes.values())),
        "d_token_hist_over_flips": dict(sorted(totals["d_token_hist"].items())),
        "band": {
            str(D): {
                "positions": totals["band_positions"][D],
                "flips_in_band": totals["band_flip_in"][D],
                "out_of_band_indicator_bytes": bits_to_bytes(totals["band_ind_bits_out"][D]),
            }
            for D in (1, 2, 3)
        },
        "elapsed_seconds": time.time() - t0,
        "peak_rss_gib": peak_rss_gib(),
    }
    if comp["size"].size:
        sizes = comp["size"]
        summary["component_sizes_8"] = _size_summary(sizes, comp["cost_bits"])
    summary["component_sizes_4"] = _size_summary_counter(comp4_sizes, comp4_cost)

    if keep:
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "components_8conn.npz",
            pair=comp["pair"].astype(np.uint16),
            size=comp["size"].astype(np.int32),
            seed=comp["seed"].astype(np.int32),
            cost_bits=comp["cost_bits"].astype(np.float64),
            seed_cost_bits=comp["seed_cost_bits"].astype(np.float64),
            bh=comp["bh"].astype(np.int16),
            bw=comp["bw"].astype(np.int16),
            shape_id=comp["shape_id"].astype(np.int32),
        )
        np.save(out / "seed_gaps.i64.npy", np.asarray(seed_gaps, dtype=np.int64))
        np.savez(
            out / "features_live.npz",
            logit=fs.logit,
            ind_bits=fs.ind_bits,
            flip=fs.flip,
            pair=fs.pair,
            pat4=fs.pat4,
            pat8=fs.pat8,
            d_other=fs.d_other,
            argmax=fs.argmax,
            is_seed=fs.is_seed,
            is_covered=fs.is_covered,
        )
        atomic_write_json(out / "PER_PAIR.json", per_pair)
    atomic_write_json(out / "FEATURES.json", summary)
    return {
        "summary": summary,
        "fs": fs,
        "comp": comp,
        "seed_gaps": np.asarray(seed_gaps, dtype=np.int64),
        "shape_keys": shape_keys,
    }


def _relabel(flip, structure):
    lab, _ = ndimage.label(flip.reshape(H, W), structure=structure)
    return lab.reshape(-1)


def _size_summary(sizes: np.ndarray, cost_bits: np.ndarray) -> dict:
    out = {
        "count": int(sizes.size),
        "total_sites": int(sizes.sum()),
        "mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "p90": float(np.percentile(sizes, 90)),
        "p99": float(np.percentile(sizes, 99)),
        "max": int(sizes.max()),
    }
    buckets = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 10**9)]
    out["buckets"] = []
    for lo, hi in buckets:
        m = (sizes >= lo) & (sizes <= hi)
        out["buckets"].append(
            {
                "lo": lo,
                "hi": None if hi > 10**8 else hi,
                "components": int(m.sum()),
                "sites": int(sizes[m].sum()),
                "cost_bytes": bits_to_bytes(float(cost_bits[m].sum())),
            }
        )
    ge16 = sizes >= 16
    out["ge16"] = {
        "components": int(ge16.sum()),
        "component_share": float(ge16.mean()),
        "sites": int(sizes[ge16].sum()),
        "site_share": float(sizes[ge16].sum() / max(sizes.sum(), 1)),
        "cost_bytes": bits_to_bytes(float(cost_bits[ge16].sum())),
    }
    return out


def _size_summary_counter(sizes: Counter, cost: Counter) -> dict:
    if not sizes:
        return {}
    arr = np.repeat(np.asarray(sorted(sizes)), [sizes[s] for s in sorted(sizes)])
    tot = sum(cost.values())
    ge16_sites = sum(s * n for s, n in sizes.items() if s >= 16)
    return {
        "count": int(sum(sizes.values())),
        "total_sites": int(sum(s * n for s, n in sizes.items())),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": int(max(sizes)),
        "cost_bytes": bits_to_bytes(tot),
        "ge16_components": int(sum(n for s, n in sizes.items() if s >= 16)),
        "ge16_sites": int(ge16_sites),
        "ge16_cost_bytes": bits_to_bytes(sum(c for s, c in cost.items() if s >= 16)),
    }


# ---------------------------------------------------------------------------
# cross-fitted instruments
# ---------------------------------------------------------------------------


def fold_split(pairs: np.ndarray, npairs: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(npairs)
    half = npairs // 2
    fold_of_pair = np.zeros(npairs, np.uint8)
    fold_of_pair[perm[half:]] = 1
    return fold_of_pair[pairs]


def newton_beta(logit: np.ndarray, y: np.ndarray, cell: np.ndarray, ncell: int, iters: int = 40) -> np.ndarray:
    """Fit q' = sigmoid(logit + beta_cell) by Newton; mi1's family."""
    beta = np.zeros(ncell, np.float64)
    for _ in range(iters):
        z = logit + beta[cell]
        q = 1.0 / (1.0 + np.exp(-z))
        g = np.bincount(cell, weights=(q - y), minlength=ncell)
        hh = np.bincount(cell, weights=q * (1.0 - q), minlength=ncell)
        step = g / np.maximum(hh, 1e-12)
        step = np.clip(step, -2.0, 2.0)
        beta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def nll_bits(logit: np.ndarray, y: np.ndarray, beta_cell: np.ndarray) -> float:
    z = logit + beta_cell
    # -log2 p(y) with p = sigmoid(z)
    return float(np.sum(np.logaddexp(0.0, np.where(y > 0, -z, z))) / math.log(2.0))


def cell_rate_code_bits(y_tr, cell_tr, y_te, cell_te, ncell: int, alpha: float = 0.5) -> float:
    """KT-smoothed per-cell Bernoulli, fit on train, coded on test."""
    n1 = np.bincount(cell_tr, weights=y_tr.astype(np.float64), minlength=ncell)
    n = np.bincount(cell_tr, minlength=ncell).astype(np.float64)
    r = (n1 + alpha) / (n + 2 * alpha)
    r = np.clip(r, 1e-12, 1 - 1e-12)
    lp1 = -np.log2(r)
    lp0 = -np.log2(1.0 - r)
    return float(np.sum(np.where(y_te > 0, lp1[cell_te], lp0[cell_te])))


def qbins(logit: np.ndarray, k: int, seed_sample: int = 4_000_000) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = logit.size
    idx = rng.choice(n, size=min(seed_sample, n), replace=False)
    edges = np.quantile(logit[idx].astype(np.float64), np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(edges, logit).astype(np.int32)


# ---------------------------------------------------------------------------
# STAGE ceiling
# ---------------------------------------------------------------------------


def stage_ceiling(out: Path, feat: dict, npairs: int) -> dict:
    t0 = time.time()
    fs: FeatureStore = feat["fs"]
    comp = feat["comp"]
    seed_gaps = feat["seed_gaps"]
    summ = feat["summary"]

    logit = fs.logit.astype(np.float64)
    y = fs.flip.astype(np.float64)
    pair = fs.pair  # uint16; used only as an index into fold_of_pair
    ind_bits = fs.ind_bits.astype(np.float64)

    inc_full_bits = float(ind_bits.sum())
    inc_no_bits = float(ind_bits[fs.flip > 0].sum())
    res = {
        "axis": "[model-ledger code length on the coder's own rows; REFUSAL-ONLY]",
        "score_claim": False,
        "promotable": False,
        "schema": "ddm_hc2_ceiling.v1",
        "pairs": npairs,
        "live_positions": int(fs.n_live),
        "flips": int(fs.flip.sum()),
        "incumbent": {
            "no_branch_bytes": bits_to_bytes(inc_no_bits),
            "full_indicator_bytes": bits_to_bytes(inc_full_bits),
            "yes_branch_bytes": bits_to_bytes(inc_full_bits - inc_no_bits),
            "conditional_bytes": summ["conditional_bytes"],
            "stream_bytes_ideal": summ["stream_bytes_ideal"],
        },
    }

    qb32 = qbins(logit, 32)
    qb64 = qbins(logit, 64)

    # ---- (d) family bound: mi1's beta-per-cell with causal-neighbourhood cells
    # every cell is a dense small product, so the index IS the cell id (no sort needed)
    cell_specs = {
        "none": (lambda: np.zeros(fs.n_live, np.int32), 1),
        "q32": (lambda: qb32, 32),
        "pat4": (lambda: fs.pat4.astype(np.int32), 16),
        "pat8": (lambda: fs.pat8.astype(np.int32), 256),
        "q32_x_pat4": (lambda: qb32 * 16 + fs.pat4, 32 * 16),
        "q32_x_pat8": (lambda: qb32 * 256 + fs.pat8, 32 * 256),
        "d_other": (lambda: np.minimum(fs.d_other, 4).astype(np.int32), 5),
        "q32_x_dother": (lambda: qb32 * 5 + np.minimum(fs.d_other, 4).astype(np.int32), 32 * 5),
    }
    d_rows = {}
    for name, (mk, ncell) in cell_specs.items():
        cidx = np.ascontiguousarray(mk(), dtype=np.int64)
        per_seed = []
        for sd in SEEDS:
            fold = fold_split(pair, npairs, sd)
            held = 0.0
            for f_te in (0, 1):
                tr = fold != f_te
                te = ~tr
                beta = newton_beta(logit[tr], y[tr], cidx[tr], ncell)
                held += nll_bits(logit[te], y[te], beta[cidx[te]])
            per_seed.append(bits_to_bytes(inc_full_bits - held))
        d_rows[name] = {
            "cells": int(ncell),
            "held_out_bytes_saved_per_seed": per_seed,
            "held_out_bytes_saved_min": min(per_seed),
            "held_out_bytes_saved_median": float(np.median(per_seed)),
        }
        print(f"[d] {name}: cells={ncell} min={min(per_seed):.2f} B", flush=True)
        del cidx
    res["d_family_bound"] = d_rows

    # ---- (c)/(a) component-code ceiling
    sizes = comp["size"].astype(np.int64)
    shape_id = comp["shape_id"].astype(np.int64)
    comp_pair = comp["pair"].astype(np.int64)
    comp_cost = comp["cost_bits"].astype(np.float64)
    comp_seed_cost = comp["seed_cost_bits"].astype(np.float64)
    ncomp = sizes.size
    nshape = int(shape_id.max()) + 1 if ncomp else 0

    not_covered = fs.is_covered == 0
    seed_y = fs.is_seed.astype(np.float64)

    a_rows = {}
    per_seed_a2 = []
    per_seed_a2b = []
    per_seed_a1 = []
    per_seed_shape = []
    per_seed_ge16 = []
    # strongest available seed model: beta-per-cell on the FULL-resolution flip logit,
    # cell = q32 x min(d_other, 4) (the acausal boundary geometry -- generous on purpose)
    cell_seed = (qb32 * 5 + np.minimum(fs.d_other, 4).astype(np.int32)).astype(np.int64)
    for sd in SEEDS:
        fold_pos = fold_split(pair, npairs, sd)
        fold_comp = fold_split(comp_pair, npairs, sd)

        # --- (a2) mixer-conditioned seed field, KT per q-bin (64), covered positions free
        seed_bits = 0.0
        seed_bits_b = 0.0
        for f_te in (0, 1):
            tr = (fold_pos != f_te) & not_covered
            te = (fold_pos == f_te) & not_covered
            seed_bits += cell_rate_code_bits(seed_y[tr], qb64[tr], seed_y[te], qb64[te], 64)
            beta = newton_beta(logit[tr], seed_y[tr], cell_seed[tr], 32 * 5)
            seed_bits_b += nll_bits(logit[te], seed_y[te], beta[cell_seed[te]])

        # --- shape code: cross-fitted dictionary with escape
        shape_bits = 0.0
        shape_bits_per_comp = np.zeros(ncomp, np.float64)
        for f_te in (0, 1):
            tr = fold_comp != f_te
            te = fold_comp == f_te
            cnt = np.bincount(shape_id[tr], minlength=nshape).astype(np.float64)
            ntr = float(cnt.sum())
            # escape mass: Krichevsky-Trofimov style, one escape symbol
            seen = cnt > 0
            nseen = float(seen.sum())
            denom = ntr + nseen + 1.0
            p_shape = np.where(seen, (cnt + 0.0) / denom, 0.0)
            p_esc = 1.0 / denom
            b_seen = np.zeros(nshape, np.float64)
            b_seen[seen] = -np.log2(np.maximum(p_shape[seen], 1e-300))
            sid_te = shape_id[te]
            is_seen = seen[sid_te]
            bits = np.where(is_seen, b_seen[sid_te], 0.0)
            # escape -> explicit bounding-box bitmap code: -log2(p_esc) + log2 bbox area + bbox bits
            bh = comp["bh"].astype(np.float64)[te]
            bw = comp["bw"].astype(np.float64)[te]
            explicit = -math.log2(p_esc) + 2 * np.log2(np.maximum(bh, 1.0) + 1) + 2 * np.log2(
                np.maximum(bw, 1.0) + 1
            ) + bh * bw
            bits = np.where(is_seen, bits, explicit)
            shape_bits += float(bits.sum())
            shape_bits_per_comp[te] = bits

        # --- (a1) unconditioned raster-gap seed code (the "address is the tax" reading)
        gap_bits = _gap_code_bits(seed_gaps, comp_pair, fold_comp)

        per_seed_a2.append(bits_to_bytes(seed_bits + shape_bits))
        per_seed_a2b.append(bits_to_bytes(seed_bits_b + shape_bits))
        per_seed_a1.append(bits_to_bytes(gap_bits + shape_bits))
        per_seed_shape.append(bits_to_bytes(shape_bits))

        # CLUSTERING GAIN attributed per component: the representation pays one seed
        # event (matching the incumbent's flip bit at the seed) and then a shape code in
        # place of the incumbent's per-site cost at the component's remaining sites.
        ge16 = sizes >= 16
        saved_per_comp = (comp_cost - comp_seed_cost) - shape_bits_per_comp
        per_seed_ge16.append(
            {
                "all_clustering_gain_bytes": bits_to_bytes(float(saved_per_comp.sum())),
                "ge16_clustering_gain_bytes": bits_to_bytes(float(saved_per_comp[ge16].sum())),
                "ge16_share_of_gain": float(
                    saved_per_comp[ge16].sum() / saved_per_comp.sum()
                    if abs(saved_per_comp.sum()) > 1e-9
                    else 0.0
                ),
                "positive_gain_components": int((saved_per_comp > 0).sum()),
            }
        )

    a_rows["a2_mixer_conditioned_seed_field"] = {
        "bytes_per_seed": per_seed_a2,
        "bytes_max": max(per_seed_a2),
        "bytes_min": min(per_seed_a2),
        "bytes_median": float(np.median(per_seed_a2)),
    }
    a_rows["a2b_beta_per_cell_seed_field"] = {
        "bytes_per_seed": per_seed_a2b,
        "bytes_max": max(per_seed_a2b),
        "bytes_min": min(per_seed_a2b),
        "bytes_median": float(np.median(per_seed_a2b)),
    }
    a_rows["a1_raster_gap_seed_code"] = {
        "bytes_per_seed": per_seed_a1,
        "bytes_max": max(per_seed_a1),
        "bytes_median": float(np.median(per_seed_a1)),
    }
    a_rows["shape_component"] = {
        "bytes_per_seed": per_seed_shape,
        "bytes_median": float(np.median(per_seed_shape)),
    }
    a_rows["components"] = int(ncomp)
    a_rows["distinct_shapes"] = int(nshape)
    a_rows["ge16_attribution"] = per_seed_ge16
    res["a_component_code"] = a_rows

    # ---- (b) boundary-offset ceiling
    b_rows = {}
    for D in (1, 2, 3):
        band = fs.d_other <= D
        out_bits = float(ind_bits[~band].sum())
        cell_geo = (np.minimum(fs.d_other[band], D).astype(np.int64) * 5 + fs.argmax[band]).astype(np.int64)
        cell_q = cell_geo * 32 + qb32[band]
        yb = y[band]
        pb = pair[band]
        lb = logit[band]
        rows = {}
        for cname, cidx, ncell in (
            ("geometry_only", cell_geo, (D + 1) * 5),
            ("geometry_x_q32", cell_q, (D + 1) * 5 * 32),
        ):
            per_seed = []
            for sd in SEEDS:
                fold = fold_split(pb, npairs, sd)
                in_bits = 0.0
                for f_te in (0, 1):
                    tr = fold != f_te
                    te = ~tr
                    if cname == "geometry_only":
                        in_bits += cell_rate_code_bits(yb[tr], cidx[tr], yb[te], cidx[te], ncell)
                    else:
                        beta = newton_beta(lb[tr], yb[tr], cidx[tr], ncell)
                        in_bits += nll_bits(lb[te], yb[te], beta[cidx[te]])
                per_seed.append(bits_to_bytes(in_bits + out_bits))
            rows[cname] = {
                "cells": int(ncell),
                "total_bytes_per_seed": per_seed,
                "total_bytes_max": max(per_seed),
                "total_bytes_median": float(np.median(per_seed)),
            }
        b_rows[str(D)] = {
            "band_positions": int(band.sum()),
            "band_share_of_live": float(band.mean()),
            "flips_in_band": int(yb.sum()),
            "flip_share_in_band": float(yb.sum() / max(y.sum(), 1)),
            "out_of_band_indicator_bytes": bits_to_bytes(out_bits),
            "models": rows,
        }
        print(f"[b] D={D} band={int(band.sum())} flips_in={int(yb.sum())}", flush=True)
    res["b_boundary_offset"] = b_rows
    res["elapsed_seconds"] = time.time() - t0
    res["peak_rss_gib"] = peak_rss_gib()
    atomic_write_json(out / "CEILING.json", res)
    return res


def _gap_code_bits(gaps: np.ndarray, comp_pair: np.ndarray, fold_comp: np.ndarray) -> float:
    """Log-bucketed gap code, cross-fitted: bucket entropy + uniform residual."""
    if gaps.size == 0:
        return 0.0
    g = np.maximum(gaps.astype(np.int64), 1)
    bucket = np.floor(np.log2(g)).astype(np.int64)
    width = np.power(2.0, bucket)
    nb = int(bucket.max()) + 1
    total = 0.0
    for f_te in (0, 1):
        tr = fold_comp != f_te
        te = fold_comp == f_te
        cnt = np.bincount(bucket[tr], minlength=nb).astype(np.float64) + 0.5
        p = cnt / cnt.sum()
        total += float(np.sum(-np.log2(p[bucket[te]]) + np.log2(width[te])))
    return total


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["control", "features", "ceiling", "all", "dry"])
    ap.add_argument("--out", default="/Volumes/APDataStore/pact/ddm_hc2_wrong_half_decomposition")
    ap.add_argument("--pairs", type=int, default=N_PAIRS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-keep", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.stage in ("control", "all"):
        r = stage_control(out, quick=args.quick)
        print(json.dumps({k: r[k] for k in ("byte_identical_reencode", "reconstruction")}, indent=2, default=_jsonable))

    if args.stage == "dry":
        feat = stage_features(out / "dry", npairs=args.pairs, keep=False, touch_alloc=True)
        print(json.dumps(feat["summary"], indent=2, default=_jsonable))
        print(f"DRY PEAK RSS GiB {peak_rss_gib():.3f} elapsed {time.time() - t0:.1f}s")
        return 0

    if args.stage in ("features", "ceiling", "all"):
        feat = stage_features(out, npairs=args.pairs, keep=not args.no_keep)
        print(json.dumps(feat["summary"], indent=2, default=_jsonable))
        if args.stage in ("ceiling", "all"):
            res = stage_ceiling(out, feat, npairs=args.pairs)
            print(json.dumps({k: v for k, v in res.items() if k != "b_boundary_offset"}, indent=2, default=_jsonable))

    print(f"[hc2] done stage={args.stage} elapsed={time.time() - t0:.1f}s peak_rss={peak_rss_gib():.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
