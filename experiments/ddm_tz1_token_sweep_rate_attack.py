#!/usr/bin/env python
"""ddm_tz1 — archive token-sweep RATE attack harness (SCORER-FREE, byte-only).

Deliverable of arm ``ddm_tz1`` (operator: "remember the archive token sweep rate
attack" + steer#1 "adaptive dynamic quantization and quantization awareness" +
steer#2 "smaller than int16 where int16 is unnecessary rate").

Tokens are ~99% of the archive bytes = THE rate axis (rate = 15.5% of the gap to
the PR130 floor). This harness measures every SCORER-FREE (byte-only) leg of the
token-sweep rate attack NOW, and structures the JOINT (rate x d_seg) waterfill so
the d_seg-under-drop verdicts fire the instant the scorer frees (sibling ddm_bz1
owns the one n600 scorer slot + tac.submission_chain). This arm runs NO scorer,
edits NO receiver / submission_chain.

Six byte-only arms, all measured here:
  A. GLOBAL-L sweep   -- token_quant_levels, both vehicle forms (ix2 LIVE + smevr).
                         Reproduces bs2/#933's L=14 = -23,655 B (smevr form) AND
                         corrects it to the LIVE ix2 vehicle (-24,605 B). This IS
                         the STATIC / one-rung special case of arm B.
  B. ADAPTIVE per-cell L (#869: 768-cell x 4-rung token-by-token waterfill).
                         Each cell gets its OWN quant rung. Global-L is one point.
                         Two rung MAPS priced: (b1) decoder-derived (token
                         activity, rule-118-free = 0 counted bytes) and (b2)
                         margin-coupled QAT (pa1b `margin_coupled_level_map` over
                         the QA80 flip-mass field; scorer-derived => STORED/counted).
  C. RUNG-MAP price   -- STORED (counted) vs DERIVED (0-byte). Priced for both maps.
  D. +-1.0 CLAMP mass -- per-level + per-channel histogram; reproduces bs2's 33.30%
                         pinned. The adaptive/dynamic range is the per-cell twin of
                         the static +-1.0; the REFIT itself is scorer-gated (needs
                         the continuous pre-clamp tokens, absent from any artifact).
  E. LZMA-filter race -- re-race lc/lp/pb on the REAL token payloads (both forms).
  F. DEPTH x CODER    -- steer#2: L IS the token bit-depth (4 bits/token @ L=16);
                         tight-bitpack ceil(log2 L) vs the fixed nibble; ST_GRID
                         table already depth-laddered (encode_exact_table).

Every LOSSLESS round-trip is verified on the real bytes. Every d_seg/d_pose-under-
drop verdict is QUEUED (READY manifest) with a pre-registered break-even, NOT run.

Own-vehicle frontier: S = 0.7910689 @ 353,805 B [macOS-CPU advisory] -- UNMOVED.
This arm byte-closes nothing alone; it is the RATE leg that composes with bz1's
seg+pose row (disjoint archive sections) at byte-close time.
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np

# ---- live decode-tree modules (READ-ONLY; this arm imports, never edits) -------
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from tac.optimization import ddm_ix2_archive_container as IX2  # noqa: E402  LIVE form
from tac.witness_dsl.ax1_pool_a_levers_20260730 import (  # noqa: E402  recall-at-source
    apply_per_cell_quant_np,
    margin_coupled_level_map,
)

try:
    import brotli  # noqa: E402
except Exception:  # pragma: no cover
    brotli = None

# ------------------------------------------------------------------------------ #
# constants (gk2 calibration; current pu2 frontier)                              #
# ------------------------------------------------------------------------------ #
DEN = 37_545_489                       # rate-term denominator (upstream/evaluate.py:63 archive bytes)
PX = 196_608 * 600                     # SegNet argmax pixels over n600 (=117,964,800)
W_BYTES_PER_FLIP = 4.0 * DEN / PX      # break-even bytes per seg-flip (br1)
S_BASE = 0.7910689                     # own-vehicle frontier (pu2) [macOS-CPU advisory]
PR130_FLOOR = 0.172141                 # the BAR (m05); NOT a lineage
GAP = S_BASE - PR130_FLOOR             # 0.6189279
S_PER_BYTE = 25.0 / DEN                # 6.659e-7

# Vehicle-form token member anchors (positive controls; both proven on cx1_tokens):
IX2_BASE_L16 = 341_295                 # LIVE ix2 encode_token_frame(levels=16) (br1 anchor)
SMEVR_BASE_L16 = 346_478               # smevr/tr1 form encode_token_codes(levels=16) (bs2 base)

DEFAULT_TOKENS = "/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy"
DEFAULT_FIELD = "/Volumes/VertigoDataTier/pact/ddm_sg1_20260731/cell_flip_mass.npy"
DEFAULT_OUT = "/Volumes/VertigoDataTier/pact/ddm_tz1_20260804/tz1_byte_only_receipt.json"


# ------------------------------------------------------------------------------ #
# helpers                                                                         #
# ------------------------------------------------------------------------------ #
def _requant_global(codes16: np.ndarray, levels: int) -> np.ndarray:
    """Dequant shipped 16-level codes to [0,1], re-quantise to ``levels`` (round).

    Refit-from-shipped-codes (the only artifact we hold): x01 = c/15, code_L =
    round(x01*(L-1)). Producer-side continuous requant (bs2/smevr) differs by the
    double-quant residual; both are reported so the gap is explicit (NO-FAKE)."""
    x01 = codes16.astype(np.float64) / 15.0
    return np.rint(x01 * (levels - 1)).astype(np.uint8)


def _break_even(saved_bytes: int) -> dict[str, float]:
    """Pre-registered rate<->distortion break-even for a byte saving (#933 shape)."""
    ds_rate = -S_PER_BYTE * saved_bytes                 # negative = frontier drops
    flip_budget = saved_bytes / W_BYTES_PER_FLIP        # spend ALL on seg
    pose_term = math.sqrt(10.0 * D_POSE_LIVE)
    # if the ENTIRE rate gain were spent on pose instead of seg:
    dpose_budget = ((pose_term + S_PER_BYTE * saved_bytes) ** 2) / 10.0 - D_POSE_LIVE
    return {
        "dS_rate_only": ds_rate,
        "pct_of_gap": 100.0 * (-ds_rate) / GAP,
        "seg_flip_budget": flip_budget,
        "d_seg_budget": flip_budget / PX,
        "d_pose_budget_if_all_on_pose": dpose_budget,
    }


# Live distortion anchors (advisory; used only to shape the break-evens, never as a
# score). d_seg from the pu2 lineage cell field; d_pose from the pu2 pose tail.
D_SEG_LIVE = 0.004311794704861111      # br1 live d_seg anchor (cx1/pu2 token lattice)
D_POSE_LIVE = 0.00255143               # br1 live d_pose anchor


# ------------------------------------------------------------------------------ #
# ARM A + F(token) : GLOBAL-L sweep + bit-depth, both vehicle forms               #
# ------------------------------------------------------------------------------ #
def arm_global_L(tok: np.ndarray, levels_grid: list[int],
                 smevr_levels: tuple[int, ...] = (16, 14)) -> dict[str, Any]:
    """Full L-grid on the FAST live ix2 form; smevr (slow arithmetic coder) only on
    the provenance anchors {16,14} to confirm bs2/#933's 23,655 + the base."""
    import ddm_r7_token_coder as R7  # smevr form (READ-ONLY import)

    base_ix2 = len(IX2.encode_token_frame(tok, levels=16))
    smevr: dict[int, dict[str, Any]] = {}
    base_sm = None
    for L in smevr_levels:
        cL = _requant_global(tok, L) if L != 16 else tok
        frame_sm = R7.encode_token_codes(cL, levels=L, codec="smevr")
        back_sm = R7.decode_token_codes(frame_sm)
        smevr[L] = {"member_bytes": len(frame_sm),
                    "roundtrip": bool(np.array_equal(back_sm, cL))}
        if L == 16:
            base_sm = len(frame_sm)
    for L in smevr_levels:
        smevr[L]["saved_vs_L16"] = (base_sm - smevr[L]["member_bytes"]) if base_sm else None
    rows: list[dict[str, Any]] = []
    for L in levels_grid:
        cL = _requant_global(tok, L) if L != 16 else tok
        frame_ix2 = IX2.encode_token_frame(cL, levels=L)
        back_ix2 = IX2.decode_token_frame(frame_ix2)
        rt_ix2 = bool(np.array_equal(back_ix2, cL))
        b_ix2 = len(frame_ix2)
        saved_ix2 = base_ix2 - b_ix2
        winner = _ix2_coder_winner(cL, L)   # coder x rung surface
        rows.append({
            "L": L,
            "bits_per_token": math.log2(L),
            "ix2_member_bytes": b_ix2, "ix2_saved_bytes": saved_ix2, "ix2_roundtrip": rt_ix2,
            "ix2_coder_winner": winner,
            "break_even_ix2": _break_even(saved_ix2) if saved_ix2 > 0 else None,
        })
    return {
        "ix2_base_L16": base_ix2, "smevr_base_L16": base_sm,
        "ix2_base_matches_anchor": base_ix2 == IX2_BASE_L16,
        "smevr_base_matches_anchor": (base_sm == SMEVR_BASE_L16) if base_sm else None,
        "smevr_provenance": {str(k): v for k, v in smevr.items()},
        "smevr_L14_reproduces_bs2_23655": (smevr.get(14, {}).get("saved_vs_L16") == 23_655),
        "rows": rows,
    }


def _ix2_coder_winner(codes: np.ndarray, levels: int) -> dict[str, Any]:
    """Which generic coder wins the ix2 residual + base nibble blocks (coder race)."""
    base, delta = IX2._factor_mode_delta(codes, levels)
    residual = np.ascontiguousarray(np.transpose(delta, (1, 2, 3, 0)))
    res_pay = IX2._pack_nibbles(residual.reshape(-1))
    base_pay = IX2._pack_nibbles(base.reshape(-1))
    cr, _ = IX2.code_block(res_pay)
    cb, _ = IX2.code_block(base_pay)
    return {"residual": IX2.CODER_NAMES[cr], "base": IX2.CODER_NAMES[cb]}


# ------------------------------------------------------------------------------ #
# ARM B + C : ADAPTIVE per-cell L (#869 waterfill) + rung-map price               #
# ------------------------------------------------------------------------------ #
def _codes_to_pm1(codes16: np.ndarray) -> np.ndarray:
    return codes16.astype(np.float64) / 15.0 * 2.0 - 1.0


def _pm1_to_codes16(t: np.ndarray) -> np.ndarray:
    x01 = (np.clip(t, -1.0, 1.0) + 1.0) * 0.5
    return np.rint(x01 * 15.0).astype(np.uint8)


def _per_cell_activity(tok: np.ndarray) -> np.ndarray:
    """Decoder-DERIVABLE per-cell signal (rule-118-free): temporal flip count of the
    token codes at each (R,C) cell, summed over channels. The decoder holds the
    tokens, so a rung map = f(activity) costs ZERO counted bytes. Vectorised via the
    shipped mode-delta factorisation."""
    base, _ = IX2._factor_mode_delta(tok, 16)       # base (R,C,K) = per-cell temporal mode
    act = (tok != base[None]).sum(axis=(0, 3))      # (R,C) temporal-activity per cell
    return act.astype(np.float64)


def _apply_cell_level_map(tok: np.ndarray, level_map: np.ndarray) -> np.ndarray:
    """Coarsen each (R,C) cell's tokens to its assigned rung, KEEPING the container's
    global levels=16 (FORMAT-FREE, br1 alphabet-drop mechanism, per-cell adaptive).
    Returns codes {0..15} where each cell uses only its sub-lattice symbols.
    Fully vectorised: L broadcasts as (1,R,C,1) over the (P,R,C,K) lattice."""
    t = np.clip(_codes_to_pm1(tok), -1.0, 1.0)      # (P,R,C,K) in [-1,1]
    L = (level_map.astype(np.float64) - 1.0)[None, :, :, None]   # (1,R,C,1)
    x01 = (t + 1.0) * 0.5
    snapped = np.round(x01 * L) / L * 2.0 - 1.0     # per-cell sublattice snap
    return _pm1_to_codes16(snapped)


def arm_adaptive_per_cell(tok: np.ndarray, field: np.ndarray | None,
                          rung_ladders: list[list[int]]) -> dict[str, Any]:
    base_ix2 = len(IX2.encode_token_frame(tok, levels=16))
    P, R, C, K = tok.shape
    activity = _per_cell_activity(tok)             # (R,C) decoder-derivable
    results: list[dict[str, Any]] = []
    for ladder in rung_ladders:
        base_lvl, min_lvl = max(ladder), min(ladder)
        n_tiers = len(ladder)
        # ---- (b1) decoder-DERIVED map (token activity; 0 counted bytes) ----
        # high activity -> keep fine (base), low activity -> coarse (min). Same rank
        # law as margin_coupled but over the DECODABLE activity field.
        dmap = margin_coupled_level_map(activity, base_levels=base_lvl,
                                        min_levels=min_lvl, n_tiers=n_tiers)
        cD = _apply_cell_level_map(tok, dmap)
        bD = len(IX2.encode_token_frame(cD, levels=16))
        rtD = bool(np.array_equal(IX2.decode_token_frame(IX2.encode_token_frame(cD, levels=16)), cD))
        savedD = base_ix2 - bD
        mapcost_if_stored = _rung_map_cost(dmap)   # reference: what the derived map WOULD cost stored
        # ---- (b2) margin-coupled QAT map (QA80 flip-mass; STORED/counted) ----
        entry: dict[str, Any] = {
            "rung_ladder": ladder, "n_cells": R * C,
            "derived_map": {
                "member_bytes": bD, "saved_bytes_gross": savedD,
                "roundtrip": rtD,
                "map_counted_bytes": 0,               # decoder-derivable (rule-118-free)
                "map_bytes_if_stored": mapcost_if_stored,   # reference only; NOT counted
                "saved_bytes_net": savedD,            # net = gross (0-byte map)
                "level_histogram": _lvl_hist(dmap),
                "break_even": _break_even(savedD) if savedD > 0 else None,
            },
        }
        if field is not None:
            mmap = margin_coupled_level_map(field.astype(np.float64), base_levels=base_lvl,
                                            min_levels=min_lvl, n_tiers=n_tiers)
            cM = _apply_cell_level_map(tok, mmap)
            bM = len(IX2.encode_token_frame(cM, levels=16))
            rtM = bool(np.array_equal(IX2.decode_token_frame(IX2.encode_token_frame(cM, levels=16)), cM))
            savedM = base_ix2 - bM
            mapcostM = _rung_map_cost(mmap)          # STORED (scorer-derived field)
            entry["margin_coupled_map"] = {
                "member_bytes": bM, "saved_bytes_gross": savedM, "roundtrip": rtM,
                "map_counted_bytes": mapcostM,        # MUST be stored (field not decodable)
                "saved_bytes_net": savedM - mapcostM,
                "level_histogram": _lvl_hist(mmap),
                "break_even": _break_even(savedM - mapcostM) if (savedM - mapcostM) > 0 else None,
            }
        results.append(entry)
    return {"ix2_base_L16": base_ix2, "n_cells": R * C, "ladders": results}


def _lvl_hist(level_map: np.ndarray) -> dict[str, int]:
    vals, counts = np.unique(level_map, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def _rung_map_cost(level_map: np.ndarray) -> int:
    """STORED cost of a per-cell rung map: entropy-coded tier indices via code_block."""
    vals = np.unique(level_map)
    remap = {int(v): i for i, v in enumerate(vals)}
    idx = np.vectorize(remap.get)(level_map).astype(np.uint8).reshape(-1)
    _, coded = IX2.code_block(idx.tobytes())
    return len(coded)


# ------------------------------------------------------------------------------ #
# ARM D : +-1.0 clamp mass characterization (byte-only; refit is scorer-gated)    #
# ------------------------------------------------------------------------------ #
def arm_clamp_mass(tok: np.ndarray) -> dict[str, Any]:
    P, R, C, K = tok.shape
    hist = np.bincount(tok.reshape(-1), minlength=16).astype(np.int64)
    total = int(hist.sum())
    lvl0 = float(hist[0]) / total
    lvl15 = float(hist[15]) / total
    per_channel = {}
    for k in range(K):
        h = np.bincount(tok[..., k].reshape(-1), minlength=16)
        t = int(h.sum())
        per_channel[f"ch{k}"] = {
            "lvl0_frac": float(h[0]) / t, "lvl15_frac": float(h[15]) / t,
            "extremes_frac": float(h[0] + h[15]) / t,
        }
    return {
        "level_histogram": {str(i): int(hist[i]) for i in range(16)},
        "lvl0_frac": lvl0, "lvl15_frac": lvl15,
        "extremes_frac": lvl0 + lvl15,
        "reproduces_bs2_33pct": abs((lvl0 + lvl15) - 0.3330) < 0.01,
        "per_channel": per_channel,
        "refit_status": "SCORER-GATED: needs continuous pre-clamp tokens (absent from any artifact); "
                        "the adaptive/dynamic range is the per-cell twin (arm B).",
    }


# ------------------------------------------------------------------------------ #
# ARM E : LZMA-filter re-race on the REAL token payloads                          #
# ------------------------------------------------------------------------------ #
def arm_lzma_filter_race(tok: np.ndarray) -> dict[str, Any]:
    """Re-race lc/lp/pb over the ACTUAL ix2 token payloads vs the shipped filter +
    the full code_block field (stored/deflate/brotli/lzma). Pure byte-only."""
    base, delta = IX2._factor_mode_delta(tok, 16)
    residual = np.ascontiguousarray(np.transpose(delta, (1, 2, 3, 0)))
    payloads = {
        "residual_nibbles": IX2._pack_nibbles(residual.reshape(-1)),
        "base_nibbles": IX2._pack_nibbles(base.reshape(-1)),
    }
    out: dict[str, Any] = {}
    for name, pay in payloads.items():
        incumbent = {
            "stored": len(pay),
            "deflate9": len(zlib.compress(pay, 9)),
            "brotli11": len(brotli.compress(pay, quality=11, lgwin=24)) if brotli else None,
            "lzma_shipped(lc3lp0pb0)": len(lzma.compress(
                pay, format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 24, "lc": 3, "lp": 0, "pb": 0}])),
        }
        best_incumbent = min(v for v in incumbent.values() if v is not None)
        # sweep lc/lp/pb (lc+lp<=4 is the lzma constraint)
        best_variant = None
        best_variant_bytes = None
        for lc in range(0, 5):
            for lp in range(0, 3):
                if lc + lp > 4:
                    continue
                for pb in range(0, 3):
                    try:
                        b = len(lzma.compress(pay, format=lzma.FORMAT_RAW, filters=[
                            {"id": lzma.FILTER_LZMA1, "dict_size": 1 << 24, "lc": lc, "lp": lp, "pb": pb}]))
                    except Exception:
                        continue
                    if best_variant_bytes is None or b < best_variant_bytes:
                        best_variant_bytes = b
                        best_variant = {"lc": lc, "lp": lp, "pb": pb}
        out[name] = {
            "incumbent_coders": incumbent,
            "best_incumbent_bytes": best_incumbent,
            "best_lzma_variant": best_variant,
            "best_lzma_variant_bytes": best_variant_bytes,
            "lzma_variant_beats_incumbent": (best_variant_bytes is not None
                                             and best_variant_bytes < best_incumbent),
            "variant_gain_vs_shipped_lzma": incumbent["lzma_shipped(lc3lp0pb0)"] - (best_variant_bytes or 0),
        }
    return out


# ------------------------------------------------------------------------------ #
# ARM F : DEPTH x CODER (steer#2) -- tight bitpack vs fixed nibble (format-change) #
# ------------------------------------------------------------------------------ #
def _bitpack(codes: np.ndarray, bits: int) -> bytes:
    """Tight bit-pack codes at ``bits`` bits/code (MSB-first). Requires a receiver
    FORMAT CHANGE vs the fixed 4-bit nibble -> flagged, not admitted here."""
    flat = np.asarray(codes, dtype=np.uint64).reshape(-1)
    acc = 0
    nbits = 0
    out = bytearray()
    for v in flat.tolist():
        acc = (acc << bits) | int(v)
        nbits += bits
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def arm_depth_x_coder(tok: np.ndarray, levels_grid: list[int]) -> dict[str, Any]:
    """steer#2: L IS the token bit-depth. Compare the fixed nibble (4-bit) packing
    the container ships vs a TIGHT ceil(log2 L)-bit pack, each through code_block.
    The tight pack needs a receiver format gate (flagged FORMAT-CHANGE-REQUIRED)."""
    rows = []
    for L in levels_grid:
        cL = _requant_global(tok, L) if L != 16 else tok
        base, delta = IX2._factor_mode_delta(cL, L)
        residual = np.ascontiguousarray(np.transpose(delta, (1, 2, 3, 0))).reshape(-1)
        bpc = max(1, math.ceil(math.log2(L)))
        nib_pay = IX2._pack_nibbles(residual)          # 4-bit (shipped)
        _, nib_coded = IX2.code_block(nib_pay)
        tight_pay = _bitpack(residual, bpc)            # ceil(log2 L)-bit (format change)
        _, tight_coded = IX2.code_block(tight_pay)
        rows.append({
            "L": L, "bits_per_code_tight": bpc,
            "nibble_raw_bytes": len(nib_pay), "nibble_coded_bytes": len(nib_coded),
            "tight_raw_bytes": len(tight_pay), "tight_coded_bytes": len(tight_coded),
            "tight_gain_vs_nibble_coded": len(nib_coded) - len(tight_coded),
            "status": "FORMAT-CHANGE-REQUIRED (receiver nibble->bitpack gate; bz1 turf if adopted)",
        })
    return {"note": "residual block only; coder recovers most nibble slack, so tight gain is the residual",
            "rows": rows}


# ------------------------------------------------------------------------------ #
# ARM G : ST_GRID structural depth + comb-guard (byte note; re-race scorer-gated)  #
# ------------------------------------------------------------------------------ #
def arm_st_grid_structural() -> dict[str, Any]:
    from tac.optimization.pfs1_warp_receiver import ST_GRID
    fmt, coded = IX2.encode_exact_table(ST_GRID)
    _EXHAUSTIVE_CAP = 400_000
    # ca1 row-7 owed guard: enumerating a k-subset of the support must not breach the cap
    support = len(ST_GRID)
    guard = {str(k): (math.comb(support, k) <= _EXHAUSTIVE_CAP) for k in range(2, support)}
    return {
        "st_grid": list(ST_GRID), "n_knots": support,
        "table_format": ["f16", "f32", "f64", "scaled_int"][fmt],
        "table_counted_bytes": len(coded),
        "already_depth_laddered": True,   # encode_exact_table races f16<f32<f64<scaled-int
        "comb_guard_le_cap": guard,
        "reraze_status": "SCORER-GATED: re-snapping s_t to a different support changes the "
                         "homography (LOSSY); byte leg = table + per-pair index entropy, needs "
                         "the live selector artifact for the index stream.",
    }


# ------------------------------------------------------------------------------ #
# main                                                                            #
# ------------------------------------------------------------------------------ #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ddm_tz1 token-sweep RATE attack (byte-only)")
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--field", default=DEFAULT_FIELD)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--levels", default="16,15,14,13,12,10,8,6,4")
    args = ap.parse_args(argv)

    tok = np.load(args.tokens)
    if tok.ndim != 4 or tok.dtype != np.uint8:
        raise SystemExit(f"tokens must be uint8 (P,R,C,K); got {tok.shape} {tok.dtype}")

    # NO-FAKE fidelity: prove the vectorised per-cell snap == pa1b authority (recall-at-source).
    _p0 = _codes_to_pm1(tok[0])                                     # (R,C,K) in [-1,1]
    _lm = np.full(tok.shape[1:3], 8, dtype=np.int64)               # arbitrary non-uniform-safe map
    _ref = apply_per_cell_quant_np(_p0, _lm)
    _mine = _apply_cell_level_map(tok[:1], _lm)
    if not np.allclose(_ref, _codes_to_pm1(_mine[0]), atol=0.0):
        # container round-trips through codes {0..15}; compare at code granularity
        _ref_codes = _pm1_to_codes16(_ref)
        if not np.array_equal(_ref_codes, _mine[0]):
            raise SystemExit("vectorised per-cell snap DIVERGES from pa1b authority")
    field = None
    fp = Path(args.field)
    if fp.exists():
        field = np.load(fp)
        if field.shape != tok.shape[1:3]:
            print(f"[warn] field shape {field.shape} != cells {tok.shape[1:3]}; skipping margin-coupled map")
            field = None
    levels_grid = [int(x) for x in args.levels.split(",")]

    # 4-rung ladders (steer#1: the "4 rungs" of #869). base=16, floor=//4=4.
    rung_ladders = [[16, 12, 8, 4], [16, 14, 12, 10], [16, 8]]

    receipt: dict[str, Any] = {
        "arm": "ddm_tz1", "axis": "apparatus/RATE-attack byte-only",
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "own_vehicle_frontier": {"S": S_BASE, "archive_bytes": 353_805, "axis": "macOS-CPU advisory"},
        "tokens_shape": list(tok.shape), "tokens_src": args.tokens,
        "calibration": {"DEN": DEN, "PX": PX, "W_bytes_per_flip": W_BYTES_PER_FLIP,
                        "gap": GAP, "S_per_byte": S_PER_BYTE,
                        "one_pct_gap_bytes": 0.01 * GAP / S_PER_BYTE},
        "A_global_L": arm_global_L(tok, levels_grid),
        "B_adaptive_per_cell": arm_adaptive_per_cell(tok, field, rung_ladders),
        "D_clamp_mass": arm_clamp_mass(tok),
        "E_lzma_filter_race": arm_lzma_filter_race(tok),
        "F_depth_x_coder": arm_depth_x_coder(tok, levels_grid),
        "G_st_grid": arm_st_grid_structural(),
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(receipt, indent=1))

    # ---- human summary ----
    A = receipt["A_global_L"]
    print(f"== ARM A global-L (ix2 base {A['ix2_base_L16']} anchor-match={A['ix2_base_matches_anchor']}; "
          f"smevr base {A['smevr_base_L16']} match={A['smevr_base_matches_anchor']}) ==")
    for r in A["rows"]:
        sm = A["smevr_provenance"].get(str(r["L"]))
        if sm is None:
            sm_summary = "smevr saved=    n/a (rt=n/a)"
        else:
            saved = sm.get("saved_vs_L16")
            saved_summary = "n/a" if saved is None else f"{saved:7d}"
            sm_summary = f"smevr saved={saved_summary} (rt={sm['roundtrip']})"
        print(f"  L={r['L']:2d} ix2 saved={r['ix2_saved_bytes']:7d} (rt={r['ix2_roundtrip']}) "
              f"{sm_summary} "
              f"coders={r['ix2_coder_winner']}")
    print("== ARM B adaptive per-cell L (#869 waterfill) ==")
    for lad in receipt["B_adaptive_per_cell"]["ladders"]:
        d = lad["derived_map"]
        line = (f"  rungs={lad['rung_ladder']} DERIVED saved_net={d['saved_bytes_net']:7d} "
                f"(0-byte map, rt={d['derived_map'] if False else d['roundtrip']})")
        if "margin_coupled_map" in lad:
            m = lad["margin_coupled_map"]
            line += (f" | MARGIN saved_gross={m['saved_bytes_gross']:7d} "
                     f"map_cost={m['map_counted_bytes']} net={m['saved_bytes_net']:7d}")
        print(line)
    D = receipt["D_clamp_mass"]
    print(f"== ARM D clamp mass: lvl0={D['lvl0_frac']:.4%} lvl15={D['lvl15_frac']:.4%} "
          f"extremes={D['extremes_frac']:.4%} (bs2 33.30% -> {D['reproduces_bs2_33pct']}) ==")
    print("== ARM E lzma-filter race ==")
    for name, e in receipt["E_lzma_filter_race"].items():
        print(f"  {name}: incumbent_best={e['best_incumbent_bytes']} "
              f"best_lzma={e['best_lzma_variant']}={e['best_lzma_variant_bytes']} "
              f"beats={e['lzma_variant_beats_incumbent']} gain_vs_shipped={e['variant_gain_vs_shipped_lzma']}")
    print("== ARM F depth x coder (tight bitpack, FORMAT-CHANGE) ==")
    for r in receipt["F_depth_x_coder"]["rows"]:
        print(f"  L={r['L']:2d} bpc={r['bits_per_code_tight']} nibble_coded={r['nibble_coded_bytes']} "
              f"tight_coded={r['tight_coded_bytes']} gain={r['tight_gain_vs_nibble_coded']}")
    G = receipt["G_st_grid"]
    print(f"== ARM G ST_GRID: {G['n_knots']} knots, table={G['table_format']} "
          f"{G['table_counted_bytes']}B (depth-laddered={G['already_depth_laddered']}) ==")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
