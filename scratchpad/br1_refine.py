#!/usr/bin/env python
"""ddm_br1 refinement — exact exchange rates, per-unit marginals, live-symbol entropy.

Repriced after the round-1 catch: d_seg = 0.004311794704861111 (the memo's
"seg 0.4311790" is the TERM 100*d_seg, not d_seg).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac.optimization import ddm_ix2_archive_container as C  # noqa: E402

LEVELS = 16
DEN = 37_545_489
PXPP = 196_608
NPAIRS = 600
PX = PXPP * NPAIRS
W = 4.0 * DEN / PX

D_SEG = 0.004311794704861111          # ddm_pz1_dseg_n600_cx1_20260803.json
SEG_TERM = 100.0 * D_SEG
FLIPS_NOW = D_SEG * PX

TOK = np.load("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy")
P, R, Cc, K = TOK.shape
NUNITS = R * Cc * K
BASE_BYTES = len(C.encode_token_frame(TOK, levels=LEVELS))


def H0(a):
    c = np.bincount(np.asarray(a).reshape(-1), minlength=LEVELS).astype(float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    out = {}
    base, delta = C._factor_mode_delta(TOK, LEVELS)
    act = (delta != 0).sum(axis=0).reshape(-1)
    live = act > 0
    n_live, n_dead = int(live.sum()), int((~live).sum())

    print(f"d_seg = {D_SEG!r}   seg TERM = {SEG_TERM:.7f}")
    print(f"total flips now  F = {FLIPS_NOW:,.1f}  (= d_seg * {PX:,})")
    print(f"W = {W!r} B/flip")
    print(f"units: live={n_live} dead={n_dead} of {NUNITS}")

    # ---- live vs dead symbol accounting -----------------------------------
    dflat = delta.reshape(P, -1)
    live_sym = dflat[:, live]
    dead_sym = dflat[:, ~live]
    assert dead_sym.size == 0 or int(dead_sym.max()) == 0, "dead units must be all-zero"
    h_all, h_live = H0(delta), H0(live_sym)
    zero_live = float((live_sym == 0).mean())

    # exact bytes with only the dead units present (i.e. everything live dropped)
    tok_alldrop = np.repeat(base[None], P, axis=0)
    b_floor = len(C.encode_token_frame(tok_alldrop, levels=LEVELS))
    live_payload = BASE_BYTES - b_floor

    print(f"\nresidual H0 all  = {h_all:.4f} b/sym  ({delta.size:,} symbols)")
    print(f"residual H0 live = {h_live:.4f} b/sym  ({live_sym.size:,} symbols), zeros={zero_live*100:.2f}%")
    print(f"floor (all live units dropped) = {b_floor} B ; live payload = {live_payload} B")
    print(f"shipped bits per LIVE symbol   = {live_payload*8/live_sym.size:.4f}")
    print(f"order-0 bound on live symbols  = {h_live*live_sym.size/8:,.0f} B "
          f"=> coder is {h_live*live_sym.size/8/live_payload:.4f}x order-0")
    print(f"order-0 bound on ALL symbols   = {h_all*delta.size/8:,.0f} B "
          f"=> coder is {h_all*delta.size/8/live_payload:.4f}x order-0")

    out["accounting"] = {
        "d_seg": D_SEG, "seg_term": SEG_TERM, "flips_now": FLIPS_NOW, "W": W,
        "units_live": n_live, "units_dead": n_dead,
        "base_bytes": BASE_BYTES, "floor_bytes": b_floor, "live_payload_bytes": live_payload,
        "H0_all": h_all, "H0_live": h_live, "zero_frac_live": zero_live,
        "bits_per_live_symbol": live_payload * 8 / live_sym.size,
        "order0_live_bytes": h_live * live_sym.size / 8,
        "coder_vs_order0_live": h_live * live_sym.size / 8 / live_payload,
    }

    # ---- EXACT per-unit marginal byte cost, on a stratified sample --------
    order = np.argsort(-act, kind="stable")
    live_order = order[: n_live]
    picks = np.unique(np.rint(np.linspace(0, n_live - 1, 24)).astype(int))
    marg = []
    for i in picks:
        u = int(live_order[i])
        t = TOK.copy()
        t.reshape(P, -1)[:, u] = base.reshape(-1)[u]
        b = len(C.encode_token_frame(t, levels=LEVELS))
        marg.append({
            "rank": int(i), "unit": u, "activity": int(act[u]),
            "bytes_after": b, "marginal_saved": BASE_BYTES - b,
            "flip_budget": (BASE_BYTES - b) / W,
        })
        print(f"  unit rank {i:5d} act={act[u]:4d}  marginal={BASE_BYTES-b:5d} B  "
              f"budget={(BASE_BYTES-b)/W:7.1f} flips")
    out["per_unit_marginal"] = marg
    ms = np.array([m["marginal_saved"] for m in marg], float)
    print(f"\nmarginal per live unit: min={ms.min():.0f} med={np.median(ms):.0f} "
          f"max={ms.max():.0f} mean={ms.mean():.1f} B")

    # ---- the actionable fine end of the unit-drop curve --------------------
    fine = []
    for n in (5, 10, 20, 40, 80, 160, 320):
        t = TOK.copy()
        t.reshape(P, -1)[:, live_order[:n]] = base.reshape(-1)[live_order[:n]][None, :]
        b = len(C.encode_token_frame(t, levels=LEVELS))
        s = BASE_BYTES - b
        fine.append({"n_units": n, "bytes": b, "saved": s, "per_unit": s / n,
                     "flip_budget": s / W, "budget_per_unit": s / W / n,
                     "pct_of_current_flips": 100.0 * (s / W) / FLIPS_NOW})
        print(f"  drop {n:4d} most-active live units: {b:7d} B  saved={s:6d}  "
              f"{s/n:6.1f} B/unit  budget={s/W:8.0f} flips ({100.0*(s/W)/FLIPS_NOW:5.2f}% of F)")
    out["fine_curve"] = fine

    Path("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/br1_refine.json").write_text(
        json.dumps(out, indent=1)
    )
    print("\nwrote br1_refine.json")


if __name__ == "__main__":
    main()
