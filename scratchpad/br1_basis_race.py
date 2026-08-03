#!/usr/bin/env python
"""ddm_br1 BR1-1 — the basis race on the counted coefficient lattice.

Every candidate is a REVERSIBLE (mod-16) change of basis on the (600,24,32,4)
token lattice.  Reversible => the tokens decode bit-for-bit => the rendered
frames are byte-identical => d_seg and d_pose are invariant BY CONSTRUCTION.
So this is a pure, scorer-free, byte-exact rate race.

Every candidate gets the SAME 4-coder race (stored/deflate/brotli/lzma) that the
incumbent gets, so the comparison is basis-vs-basis and not coder-vs-coder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac.optimization import ddm_ix2_archive_container as C  # noqa: E402

LEVELS = 16
MAGIC_HDR = len(C.TOKEN_FRAME_MAGIC) + C._TOKEN_HEADER.size  # 8 + 20 = 28

TOK = np.load("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy")
P, R, Cc, K = TOK.shape


def race(payload: bytes) -> tuple[int, int]:
    cid, coded = C.code_block(payload)
    return cid, len(coded)


def cost(residual: np.ndarray, layout: tuple[int, ...], side: np.ndarray | None) -> dict:
    """Total token-member bytes for this (residual, layout, side-info)."""
    perm = np.ascontiguousarray(np.transpose(residual, layout))
    cid_r, n_r = race(C._pack_nibbles(perm.reshape(-1)))
    if side is None:
        cid_b, n_b = 0, 0
    else:
        cid_b, n_b = race(C._pack_nibbles(np.ascontiguousarray(side).reshape(-1)))
    return {
        "bytes": MAGIC_HDR + n_r + n_b,
        "res_bytes": n_r,
        "side_bytes": n_b,
        "coder_res": C.CODER_NAMES[cid_r],
        "coder_side": C.CODER_NAMES[cid_b],
    }


# --------------------------------------------------------------------------- #
# PREDICTORS.  Each returns (residual, side_info_or_None).  All mod-16.        #
# Every one is causally invertible at decode from already-decoded symbols plus  #
# the (counted) side info.                                                      #
# --------------------------------------------------------------------------- #

def pred_ident(v):
    return v.copy(), None


def pred_mode(v):
    """INCUMBENT: per-cell temporal mode + mod-16 residual (the r7 factorisation)."""
    base, delta = C._factor_mode_delta(v, LEVELS)
    return delta, base


def pred_tprev(v):
    """Temporal DPCM: residual[p] = v[p] - v[p-1]."""
    d = v.astype(np.int16).copy()
    d[1:] = (v[1:].astype(np.int16) - v[:-1].astype(np.int16)) % LEVELS
    return d.astype(np.uint8), None


def pred_mode_tprev(v):
    """Mode-delta, then temporal DPCM on the delta."""
    base, delta = C._factor_mode_delta(v, LEVELS)
    d = delta.astype(np.int16).copy()
    d[1:] = (delta[1:].astype(np.int16) - delta[:-1].astype(np.int16)) % LEVELS
    return d.astype(np.uint8), base


def _spatial(v, axis_shift, base=None):
    """Generic spatial predictor: subtract the neighbour along a grid axis."""
    src = v if base is None else v
    d = src.astype(np.int16).copy()
    if axis_shift == "W":  # left neighbour, axis 2 (columns)
        d[:, :, 1:, :] = (src[:, :, 1:, :].astype(np.int16) - src[:, :, :-1, :].astype(np.int16)) % LEVELS
    elif axis_shift == "N":  # up neighbour, axis 1 (rows)
        d[:, 1:, :, :] = (src[:, 1:, :, :].astype(np.int16) - src[:, :-1, :, :].astype(np.int16)) % LEVELS
    return d.astype(np.uint8)


def pred_sw(v):
    return _spatial(v, "W"), None


def pred_sn(v):
    return _spatial(v, "N"), None


def pred_med(v):
    """LOCO-I / JPEG-LS MED predictor on the 24x32 grid, per (pair, channel)."""
    x = v.astype(np.int16)
    w = np.zeros_like(x)
    n = np.zeros_like(x)
    nw = np.zeros_like(x)
    w[:, :, 1:, :] = x[:, :, :-1, :]
    n[:, 1:, :, :] = x[:, :-1, :, :]
    nw[:, 1:, 1:, :] = x[:, :-1, :-1, :]
    mx = np.maximum(w, n)
    mn = np.minimum(w, n)
    pred = np.where(nw >= mx, mn, np.where(nw <= mn, mx, w + n - nw))
    # first row/col fall back to whatever neighbour exists (zeros elsewhere)
    pred[:, 0, 0, :] = 0
    pred[:, 0, 1:, :] = w[:, 0, 1:, :]
    pred[:, 1:, 0, :] = n[:, 1:, 0, :]
    return ((x - pred) % LEVELS).astype(np.uint8), None


def pred_chan(v):
    """Channel DPCM: k>0 predicted from k-1."""
    d = v.astype(np.int16).copy()
    d[..., 1:] = (v[..., 1:].astype(np.int16) - v[..., :-1].astype(np.int16)) % LEVELS
    return d.astype(np.uint8), None


def pred_mode_sw(v):
    base, delta = C._factor_mode_delta(v, LEVELS)
    return _spatial(delta, "W"), base


def pred_mode_chan(v):
    base, delta = C._factor_mode_delta(v, LEVELS)
    d = delta.astype(np.int16).copy()
    d[..., 1:] = (delta[..., 1:].astype(np.int16) - delta[..., :-1].astype(np.int16)) % LEVELS
    return d.astype(np.uint8), base


def pred_median_base(v):
    """Per-cell temporal MEDIAN instead of MODE (a different generic model)."""
    med = np.median(v.reshape(P, -1), axis=0).astype(np.int16)
    med = np.round(med).astype(np.int16).reshape(R, Cc, K)
    delta = ((v.astype(np.int16) - med[None]) % LEVELS).astype(np.uint8)
    return delta, med.astype(np.uint8)


PREDICTORS = {
    "IDENT": pred_ident,
    "MODE*": pred_mode,           # incumbent
    "MEDIAN": pred_median_base,
    "TPREV": pred_tprev,
    "MODE+TPREV": pred_mode_tprev,
    "SW": pred_sw,
    "SN": pred_sn,
    "MED": pred_med,
    "CHAN": pred_chan,
    "MODE+SW": pred_mode_sw,
    "MODE+CHAN": pred_mode_chan,
}

LAYOUTS = {
    "PRCK(native)": (0, 1, 2, 3),
    "RCKP(cell-major*)": (1, 2, 3, 0),
    "KRCP": (3, 1, 2, 0),
    "KPRC": (3, 0, 1, 2),
    "RCPK": (1, 2, 0, 3),
    "PKRC": (0, 3, 1, 2),
}


def entropy_bits(a: np.ndarray) -> float:
    cnt = np.bincount(a.reshape(-1), minlength=LEVELS).astype(np.float64)
    p = cnt[cnt > 0] / cnt.sum()
    return float(-(p * np.log2(p)).sum())


def verify_roundtrip(name: str, fn, v):
    """Reversibility proof for the predictors used in the headline."""
    res, side = fn(v)
    if name == "MODE*":
        rec = ((side[None].astype(np.int16) + res.astype(np.int16)) % LEVELS).astype(np.uint8)
    elif name == "TPREV":
        rec = np.cumsum(res.astype(np.int32), axis=0) % LEVELS
        rec = rec.astype(np.uint8)
    elif name == "MODE+TPREV":
        d = (np.cumsum(res.astype(np.int32), axis=0) % LEVELS).astype(np.int16)
        rec = ((side[None].astype(np.int16) + d) % LEVELS).astype(np.uint8)
    elif name == "MEDIAN":
        rec = ((side[None].astype(np.int16) + res.astype(np.int16)) % LEVELS).astype(np.uint8)
    elif name == "CHAN":
        rec = (np.cumsum(res.astype(np.int32), axis=3) % LEVELS).astype(np.uint8)
    elif name == "SW":
        rec = (np.cumsum(res.astype(np.int32), axis=2) % LEVELS).astype(np.uint8)
    elif name == "SN":
        rec = (np.cumsum(res.astype(np.int32), axis=1) % LEVELS).astype(np.uint8)
    else:
        return None
    return bool(np.array_equal(rec, v))


def main():
    print(f"lattice {TOK.shape}  symbols={TOK.size}  raw-nibble={TOK.size//2} B")
    print(f"order-0 entropy of raw tokens: {entropy_bits(TOK):.4f} bits/symbol")

    # control: reproduce the shipped frame exactly
    shipped = C.encode_token_frame(TOK, levels=LEVELS)
    print(f"CONTROL re-encode of shipped path: {len(shipped)} B")

    rows = []
    for pname, fn in PREDICTORS.items():
        res, side = fn(TOK)
        h = entropy_bits(res)
        zf = float((res == 0).mean())
        rt = verify_roundtrip(pname, fn, TOK)
        best = None
        for lname, perm in LAYOUTS.items():
            c = cost(res, perm, side)
            c.update(predictor=pname, layout=lname, H0=h, zero_frac=zf, reversible=rt)
            rows.append(c)
            if best is None or c["bytes"] < best["bytes"]:
                best = c
        print(
            f"{pname:12s} H0={h:.4f} zero={zf*100:5.2f}%  best={best['bytes']:7d} B "
            f"[{best['layout']}, {best['coder_res']}] side={best['side_bytes']}B "
            f"reversible={rt}"
        )
    out = Path("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/br1_basis_race.json")
    out.write_text(json.dumps(rows, indent=1))
    rows.sort(key=lambda r: r["bytes"])
    print("\n=== TOP 12 (predictor x layout x coder) ===")
    for r in rows[:12]:
        print(
            f"{r['bytes']:7d} B  {r['predictor']:12s} {r['layout']:18s} "
            f"res={r['coder_res']:7s} side={r['side_bytes']:5d}B"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
