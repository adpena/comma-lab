#!/usr/bin/env python
"""ddm_rs2 — build the BYTE-MATCHED ordering A/B for the queued scorer gate.

WHY THIS AND NOT `cell_drop63` ALONE
------------------------------------
`ddm_br1` queued `cell_drop63` to close `ddm_na1`'s P0-2.  But the "does dropping
more pay?" question already has TWO independent measured priors that agree it does
NOT, by about 2x:

  * `ddm_ba31` n600 drop-more: 0.6498 B/flip = 0.51x W  -> dominated 1.96x
  * `ddm_gr1` n48 cell sweep:  cell_drop50 realized d_seg 0.003947 -> cell_drop63
    0.0050128, i.e. dd_seg 0.0010658 = 125,732 n600-equivalent flips against a
    79,177 B saving = 0.630 B/flip = 0.49x W -> dominated 2.03x

Spending the one scorer slot to confirm a twice-corroborated negative is poor value.
The question that is genuinely OPEN -- and that this arm's re-rank raises -- is
whether the ORDERING is right.  So both arms are built at the SAME byte budget and
differ ONLY in which cells are dropped.  One scorer pass each answers "does a
support-correct key buy fewer flips per byte than the incumbent key?", which no
receipt answers and which prices every future waterfill rung.

Both arms are lossless-format by construction (a dropped cell is a legal lattice in
the same 16-symbol alphabet; the receiver is untouched) and both are BUILT and
byte-closed here, so the gate is one command with no build step left in it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SSD = Path("/Volumes/VertigoDataTier/pact")
RT = SSD / "ddm_v4d_20260731"
PF = SSD / "ddm_pfs1_20260729/d1/eval_root/submissions/pfs1"
WORK = SSD / "ddm_rs2_20260803"
CX1_DIR = WORK / "cx1_dir"
OUT = WORK / "ab"

for _p in ("src", str(PF), str(RT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import inflate_runner_v4d as IR  # noqa: E402

from tac.optimization import ddm_ix2_archive_container as C  # noqa: E402

LEVELS, R, Cc = 16, 24, 32
SEG_H, SEG_W = 384, 512
DEN = 37_545_489
PX = 196_608 * 600
W = 4.0 * DEN / PX
RF_HALF = 34
CX1_ARCHIVE = 353_808


def spearman(a, b):
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def rf_flip_mass(half: int) -> np.ndarray:
    at = np.load(SSD / "ddm_ru1_20260729/atlas_flat.npz")
    y, x = at["y"].astype(np.int64), at["x"].astype(np.int64)
    dense = np.bincount(y * SEG_W + x, minlength=SEG_H * SEG_W).reshape(SEG_H, SEG_W)
    ii = np.zeros((SEG_H + 1, SEG_W + 1), np.int64)
    ii[1:, 1:] = dense.cumsum(0).cumsum(1)
    out = np.zeros(R * Cc)
    for r_ in range(R):
        for c_ in range(Cc):
            r0, r1 = max(0, r_ * 16 - half), min(SEG_H, (r_ + 1) * 16 + half)
            c0, c1 = max(0, c_ * 16 - half), min(SEG_W, (c_ + 1) * 16 + half)
            out[r_ * Cc + c_] = ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    blob = CX1_DIR / IR.IX2_MEMBER
    payload = blob.read_bytes()
    bulk, sections = IR.parse_payload(payload)
    codes = C.decode_token_frame(bulk)
    base, delta = C._factor_mode_delta(codes, LEVELS)
    live = (delta != 0).any(axis=0).any(axis=2).reshape(-1)
    live_ids = np.nonzero(live)[0]
    base_tokens = len(C.encode_token_frame(codes, levels=LEVELS))

    def drop(cells: np.ndarray) -> np.ndarray:
        m = codes.copy()
        m[:, cells // Cc, cells % Cc, :] = base[cells // Cc, cells % Cc, :][None]
        return m

    def emit(cells: np.ndarray, name: str) -> dict:
        mod = drop(cells)
        new_bulk = C.encode_token_frame(mod, levels=LEVELS)
        zip_bytes = C.build_single_member_zip(C.build_payload(new_bulk, sections))
        p = OUT / f"rs2_{name}_archive.zip"
        p.write_bytes(zip_bytes)
        # PROVE the archive parses and decodes to exactly the lattice we handed it
        d = CX1_DIR.parent / f"ab_{name}_dir"
        d.mkdir(parents=True, exist_ok=True)
        (d / IR.IX2_MEMBER).write_bytes(C.build_payload(new_bulk, sections))
        b2, s2 = IR.parse_payload((d / IR.IX2_MEMBER).read_bytes())
        exact = bool(np.array_equal(C.decode_token_frame(b2), mod)) and s2 == sections
        dec = IR.Decoder(d)
        _ = dec.f1(0)  # the receiver actually renders from it
        return {
            "name": name, "path": str(p), "cells_dropped": len(cells),
            "token_bytes": len(new_bulk), "archive_bytes": len(zip_bytes),
            "bytes_saved_vs_cx1": CX1_ARCHIVE - len(zip_bytes),
            "dS_rate": -25.0 * (CX1_ARCHIVE - len(zip_bytes)) / DEN,
            "flip_budget_at_W": (CX1_ARCHIVE - len(zip_bytes)) / W,
            "lattice_roundtrip_exact": exact,
            "receiver_renders": True,
            "archive_dir": str(d),
        }

    # ---- key A: gr1's gradient key (the key that SELECTED the live base) -----
    gsum = np.load(SSD / "ddm_sg1_20260731/gr1_cell_gsum.npy").reshape(-1)
    gr1_order = np.argsort(gsum, kind="stable")
    k63 = round(0.63 * 768)
    selA = gr1_order[:k63]
    assert set(np.nonzero(~live)[0]).issubset(set(selA.tolist())), "drop63 must contain drop50"

    # ---- key B: wr1's ambient-flip key on the MEASURED support ---------------
    fmrf = rf_flip_mass(RF_HALF)
    fm16 = np.zeros(R * Cc)
    at = np.load(SSD / "ddm_ru1_20260729/atlas_flat.npz")
    fm16 = np.bincount((at["y"].astype(np.int64) // 16) * Cc + (at["x"].astype(np.int64) // 16),
                       minlength=R * Cc).astype(np.float64)
    # tie-break is wr1's OWN byte proxy, unchanged, so the ONLY difference
    # between the incumbent key and this one is the SUPPORT.
    resid = np.load(SSD / "ddm_wr1_20260729/wr1_cell_sensitivity_atlas.npz")["residual_mass"]
    dead_ids = np.nonzero(~live)[0]
    rs2_live_order = live_ids[np.lexsort((-resid[live_ids], fmrf[live_ids]))]

    rowA = emit(selA, "kA_gr1_drop63")
    target = base_tokens - rowA["token_bytes"]

    # byte-match: smallest k whose saving >= arm A's saving
    lo, hi, best = 0, len(rs2_live_order), None
    while lo <= hi:
        mid = (lo + hi) // 2
        cells = np.concatenate([dead_ids, rs2_live_order[:mid]])
        b = len(C.encode_token_frame(drop(cells), levels=LEVELS))
        saved = base_tokens - b
        if saved >= target:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    kB = best if best is not None else len(rs2_live_order)
    selB = np.concatenate([dead_ids, rs2_live_order[:kB]])
    rowB = emit(selB, "kB_rs2_rfkey_bytematched")

    rep = {
        "axis": "[byte-closed, scorer-free]", "score_claim": False,
        "promotion_eligible": False, "baseline": {"name": "cx1", "S": 0.8264972,
                                                  "archive_bytes": CX1_ARCHIVE,
                                                  "token_bytes": base_tokens},
        "W": W, "rf_half_px": RF_HALF,
        "arms": [rowA, rowB],
        "byte_match_residual_B": rowA["archive_bytes"] - rowB["archive_bytes"],
        "keys": {
            "spearman_gr1gsum_vs_wr1tile": spearman(gsum, fm16),
            "spearman_gr1gsum_vs_rs2rf": spearman(gsum, fmrf),
            "spearman_wr1tile_vs_rs2rf": spearman(fm16, fmrf),
        },
        "cells": {
            "A_gr1_drop63": sorted(int(v) for v in selA),
            "B_rs2_bytematched": sorted(int(v) for v in selB),
            "A_only": sorted(int(v) for v in np.setdiff1d(selA, selB)),
            "B_only": sorted(int(v) for v in np.setdiff1d(selB, selA)),
        },
        "ambient_flip_mass_in_measured_rf": {
            "A_gr1_drop63": float(fmrf[selA].sum()),
            "B_rs2_bytematched": float(fmrf[selB].sum()),
        },
        "br1_drop63_for_comparison": {
            "token_bytes": 268751, "archive_bytes": 281264, "saved": 72544,
            "note": "br1 used a DIFFERENT cell selection at the same k=484; "
                    "gr1's own ordering saves 6,633 B more",
        },
    }
    (WORK / "rs2_ab_build_receipt.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(json.dumps(rep, indent=2, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
