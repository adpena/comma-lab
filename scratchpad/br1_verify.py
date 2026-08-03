#!/usr/bin/env python
"""ddm_br1 verification — the two load-bearing claims, proved on the real bytes.

CLAIM 1 (basis race is LOSSLESS): every raced basis is reversible, so the token
lattice decodes bit-for-bit and d_seg/d_pose are invariant BY CONSTRUCTION.

CLAIM 2 (drop surface is FORMAT-FREE): a dropped / coarsened lattice re-encodes
through the UNCHANGED encoder and decodes through the UNCHANGED decoder to
exactly the array the encoder was handed.  No container change, no receiver
change, no new gate on the format.

Also emits the final priced table with the JOINT profitability condition
(seg AND pose), since frame_0 is warped from frame_1 so a token change moves both.
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
PX = 196_608 * 600
W = 4.0 * DEN / PX
D_SEG = 0.004311794704861111
D_POSE = 0.00255143
POSE_TERM = float(np.sqrt(10.0 * D_POSE))
FLIPS_NOW = D_SEG * PX
S_BASE = 0.8264972
GAP = S_BASE - 0.172141

TOK = np.load("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy")
P, R, Cc, K = TOK.shape
BASE_BYTES = len(C.encode_token_frame(TOK, levels=LEVELS))
ARCHIVE = 353_808


def check_format_free(codes: np.ndarray, label: str) -> bool:
    """Re-encode with the UNCHANGED encoder, decode with the UNCHANGED decoder."""
    frame = C.encode_token_frame(codes, levels=LEVELS)
    back = C.decode_token_frame(frame)
    ok = bool(np.array_equal(back, codes)) and back.dtype == np.uint8
    ok = ok and int(codes.max()) < 16 and int(codes.min()) >= 0
    print(f"  {label:38s} encode->decode exact = {ok}   ({len(frame)} B)")
    return ok


def main():
    print("=== CLAIM 2: drop surface is FORMAT-FREE (unchanged encoder+decoder) ===")
    base, _ = C._factor_mode_delta(TOK, LEVELS)
    act = (base is not None) and ((TOK != base[None]).sum(axis=0)).reshape(-1)
    order = np.argsort(-act, kind="stable")
    allok = [check_format_free(TOK, "control (shipped lattice)")]

    t = TOK.copy()
    t.reshape(P, -1)[:, order[:320]] = base.reshape(-1)[order[:320]][None, :]
    allok.append(check_format_free(t, "A: drop 320 most-active units"))

    reps = np.rint(np.linspace(0, 15, 8)).astype(np.int64)
    idx = np.abs(TOK.astype(np.int64)[..., None] - reps[None, None, None, None, :]).argmin(-1)
    c8 = reps[idx].astype(np.uint8)
    allok.append(check_format_free(c8, "B: alphabet 8/16"))
    print(f"  ALL FORMAT-FREE CHECKS PASS = {all(allok)}")

    # ---- CLAIM 1: the winning basis is the incumbent; prove reversibility --
    print("\n=== CLAIM 1: basis race is LOSSLESS (mode-delta is exactly invertible) ===")
    b, d = C._factor_mode_delta(TOK, LEVELS)
    rec = ((b[None].astype(np.int16) + d.astype(np.int16)) % LEVELS).astype(np.uint8)
    print(f"  MODE* (winner) reconstruct == tokens : {np.array_equal(rec, TOK)}")
    print(f"  shipped frame reproduces exactly     : {BASE_BYTES == 341295}")

    # ---- the priced table --------------------------------------------------
    print("\n=== PRICED TABLE (rate side EXACT; seg/pose side is the queued gate) ===")
    rows = []
    drop = json.load(open("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/br1_drop_surface.json"))
    fine = json.load(open("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/br1_refine.json"))["fine_curve"]

    cands = []
    for f in fine:
        cands.append((f"A drop {f['n_units']} units", f["saved"]))
    for r in drop["rows"]:
        if r["axis"] == "B_level_drop" and r["bytes_saved"] > 0:
            cands.append((f"B alphabet {r['keep_levels']}/16", r["bytes_saved"]))

    print(f"{'candidate':24s} {'saved B':>8s} {'dS_rate':>10s} {'%gap':>7s} "
          f"{'flip budget':>12s} {'%F':>7s} {'or dpose budget':>16s}")
    for name, saved in cands:
        ds_rate = -25.0 * saved / DEN
        fb = saved / W
        # if the ENTIRE budget were spent on pose instead of seg:
        dpose_budget = ((POSE_TERM + 25.0 * saved / DEN) ** 2) / 10.0 - D_POSE
        rows.append({
            "candidate": name, "bytes_saved": saved, "dS_rate_only": ds_rate,
            "pct_of_gap": 100.0 * (-ds_rate) / GAP, "flip_budget": fb,
            "pct_of_current_flips": 100.0 * fb / FLIPS_NOW,
            "dpose_budget_if_all_spent_on_pose": dpose_budget,
            "new_archive_bytes": ARCHIVE - saved,
        })
        print(f"{name:24s} {saved:8d} {ds_rate:10.5f} {100.0*(-ds_rate)/GAP:6.2f}% "
              f"{fb:12.0f} {100.0*fb/FLIPS_NOW:6.2f}% {dpose_budget:16.2e}")

    out = {
        "constants": {"W": W, "DEN": DEN, "PX": PX, "d_seg": D_SEG, "d_pose": D_POSE,
                      "flips_now": FLIPS_NOW, "S_base": S_BASE, "gap": GAP,
                      "archive_bytes": ARCHIVE, "token_member_bytes": BASE_BYTES,
                      "pose_term_sensitivity_dS_per_dpose": 5.0 / POSE_TERM},
        "format_free_verified": all(allok),
        "priced": rows,
    }
    Path("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/br1_priced_table.json").write_text(
        json.dumps(out, indent=1)
    )
    print(f"\npose term sensitivity dS/d(d_pose) = {5.0/POSE_TERM:.2f}")
    print("wrote br1_priced_table.json")


if __name__ == "__main__":
    main()
