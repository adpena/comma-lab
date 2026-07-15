#!/usr/bin/env python3
"""PROBE #140 / mod-fold — cross-checkpoint `code` SVD eff-rank ($0 rate probe, optimal-form).

Sweep Arm A drain, 2026-07-14. NON-owned surface (NEW probe; pure numpy SVD on cached
checkpoints, NO render, NO train).

Context (DAG FEED-fl / FEED-fp, MEASURED on the v3_n600 EMA):
  The ONLY structural rate slack in the witness is the OVER-WIDE per-frame `code` (mod_dim):
  code SVD eff-rank (participation ratio) = 13.5, 90%-energy@21, 99%@31 -> mod_dim=32 is
  1.5x over-wide -> fold to ~21 saves ~12 KB (~14% of archive), DECODER-FREE. AC/range coders
  are DEAD (weights at the i.i.d. entropy floor). FEED-fl explicitly flagged as UNMEASURED:
  "(3) eff-rank(code) vs hidden-dim coupling ... the SVD was measured at h96 only" and
  "[MED, $0] mod 16-vs-21 d_seg-neutrality" -- and the mod-19-vs-mod-32 rate A/B is a
  run-gated LAUNCH item.

What this $0 probe settles (the RATE side of the run-gated A/B, without a render):
  Measure the participation-ratio eff-rank + energy-percentile ranks of the `code` matrix on
  EVERY available checkpoint (mod-dims 19/26/32), and answer:
    (a) Is a mod-32 code's 90%-energy rank <= 19 ? -> mod-19 loses <10% energy -> the rate win
        (mod 32->19, ~-40% latent table) is STRUCTURALLY SAFE on the rate axis (d_seg-neutrality
        is the run-gated arbiter).
    (b) Is a mod-19 code SATURATED (eff-rank ~ 19) -> no further mod headroom; or does it still
        carry slack (eff-rank << 19) -> mod could drop below 19 for even more rate.
    (c) Is eff-rank STABLE across checkpoints/mod-dims (resolving FEED-fl's h96-only concern)?

AUTHORITY: advisory [macOS-CPU research-signal], pure SVD of frozen cached checkpoints. This is
the RATE (energy) side only; the d_seg cost of folding is a RENDER roundtrip (run-gated arbiter).
NOT a score claim. verdict-scope: FORMULATION (rate structure), routes the run-gated A/B.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Available cached witness checkpoints carrying a `code` payload (mod-dim varies).
_DEFAULT_CKPTS = [
    "experiments/results/perclass_bitalloc_witness_20260710/mod32cap_ep650_BEST.npz",
    "experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z/levelset_witness_ema_BEST.npz",
    "experiments/results/levelset_n600_R1_storenothing_descent_ev1_20260703T004906Z/levelset_witness_ema_BEST.npz",
    "experiments/results/levelset_n600_witness_20260705T015247Z/levelset_witness_ema_BEST.npz",
    "experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_mlx.npz",
]
_ADVISORY = ("[macOS-CPU advisory research-signal] pure SVD of frozen checkpoints; RATE-energy side "
             "only (d_seg cost of folding is a render roundtrip, run-gated); NOT a score claim; "
             "routes the run-gated mod A/B; verdict-scope FORMULATION.")


def _energy_rank(sv: np.ndarray, frac: float) -> int:
    """Smallest k with cumulative energy (sigma^2) >= frac."""
    e = sv ** 2
    ce = np.cumsum(e) / e.sum()
    return int(np.searchsorted(ce, frac) + 1)


def code_effrank(code: np.ndarray) -> dict:
    """Participation-ratio eff-rank + energy-percentile ranks of the CENTERED code matrix.
    Mean is folded out at store time (FEED-fl), so eff-rank is measured on the centered matrix."""
    c = np.asarray(code, dtype=np.float64)
    c = c - c.mean(axis=0, keepdims=True)            # fold out the per-dim mean (stored separately)
    sv = np.linalg.svd(c, compute_uv=False)
    e = sv ** 2
    pr = float((e.sum() ** 2) / (np.square(e).sum() + 1e-30))   # participation ratio (eff-rank)
    return {
        "mod_dim": int(code.shape[1]),
        "n_rows": int(code.shape[0]),
        "eff_rank_participation_ratio": round(pr, 3),
        "rank_50pct_energy": _energy_rank(sv, 0.50),
        "rank_90pct_energy": _energy_rank(sv, 0.90),
        "rank_99pct_energy": _energy_rank(sv, 0.99),
        "energy_frac_top19": round(float((e[:19].sum()) / e.sum()), 5),
    }


def run_probe(ckpts: list[Path], root: Path) -> dict:
    rows = []
    for cp in ckpts:
        p = cp if cp.is_absolute() else (root / cp)
        if not p.exists():
            rows.append({"ckpt": str(cp), "status": "MISSING"})
            continue
        d = np.load(p, allow_pickle=False)
        if "code" not in d.files:
            rows.append({"ckpt": str(cp), "status": "NO_CODE_KEY"})
            continue
        r = code_effrank(d["code"])
        r["ckpt"] = os.path.basename(os.path.dirname(str(p))) or os.path.basename(str(p))
        r["status"] = "OK"
        # saturation of THIS mod_dim: how much of the width the 90%-energy rank uses.
        r["saturation_90pct"] = round(r["rank_90pct_energy"] / r["mod_dim"], 3)
        rows.append(r)

    ok = [r for r in rows if r.get("status") == "OK"]
    if not ok:
        return {"probe": "code_effrank_cross_ckpt_140", "authority": _ADVISORY,
                "rows": rows, "verdict": "NO_CHECKPOINTS", "rationale": "no code payloads found"}

    # (a) CLAIM-1 (robust across ALL checkpoints): is every mod-32 code's 90%-energy rank <= 19?
    #     If so, folding 32->19 drops <10% energy on any of them -> rate win SAFE on the rate axis.
    mod32 = [r for r in ok if r["mod_dim"] >= 32]
    max_90_mod32 = max((r["rank_90pct_energy"] for r in mod32), default=None)
    mod32to19_safe = (max_90_mod32 is not None and max_90_mod32 <= 19)
    # (b) CLAIM-2 (VEHICLE-DEPENDENT): below-mod-19 headroom. Judge on the NEWEST mod-19 code
    #     (the live vehicle), NOT the max/min across old vehicles. A code with 90%-energy rank
    #     near its width is saturated -> no sub-19 headroom; a near-rank-1 code has slack.
    mod19 = [r for r in ok if r["mod_dim"] <= 19]
    # newest-by-list-order proxy: prefer a v9/cgauge ckpt if present, else last mod19 row.
    live19 = next((r for r in mod19 if "cgauge" in r["ckpt"] or "v9" in r["ckpt"]), None)
    if live19 is None and mod19:
        live19 = mod19[-1]
    live_sat = live19["saturation_90pct"] if live19 else None
    live_effrank = live19["eff_rank_participation_ratio"] if live19 else None
    below19_headroom = (live_sat is not None and live_sat < 0.5)
    # (c) eff-rank stability across all checkpoints (resolves FEED-fl h96-only concern).
    effs = [r["eff_rank_participation_ratio"] for r in ok]
    eff_spread = (max(effs) - min(effs))

    if not mod32to19_safe:
        verdict = "MOD32TO19_RATE_RISK"
        rationale = (f"a mod-32 code needs rank>19 for 90% energy (max {max_90_mod32}) -> folding to 19 "
                     "may drop >10% energy -> the rate win carries a d_seg RISK; measure d_seg first.")
    elif below19_headroom:
        verdict = "MOD32TO19_SAFE_SUB19_HEADROOM_ON_LIVE"
        rationale = (f"mod-32 90%-energy rank max {max_90_mod32}<=19 -> 32->19 rate win SAFE (rate axis); "
                     f"AND the live mod-19 code is unsaturated (90%-energy at {live_sat:.0%} of width, "
                     f"eff-rank {live_effrank}) -> sub-19 mod has energy headroom too.")
    else:
        verdict = "MOD32TO19_SAFE_BUT_SUB19_SATURATED_ON_LIVE"
        rationale = (f"mod-32 90%-energy rank max {max_90_mod32}<=19 -> 32->19 rate win SAFE (rate axis, "
                     f"-12KB decoder-free, confirms FEED-fl across checkpoints); BUT the live mod-19 code "
                     f"is near-saturated (90%-energy at {live_sat:.0%} of its 19 width, eff-rank "
                     f"{live_effrank}) -> do NOT expect a sub-19 rate win on the live vehicle without "
                     "energy/d_seg loss. NOTE eff-rank is VEHICLE-DEPENDENT (spread "
                     f"{eff_spread:.1f} across checkpoints) -> FEED-fl's single '13.5' is not universal.")

    return {
        "probe": "code_effrank_cross_ckpt_140",
        "authority": _ADVISORY,
        "rows": rows,
        "eff_rank_spread_across_ckpts": round(eff_spread, 3),
        "max_90pct_rank_mod32": max_90_mod32,
        "live_mod19_ckpt": (live19["ckpt"] if live19 else None),
        "live_mod19_saturation_90pct": live_sat,
        "live_mod19_eff_rank": live_effrank,
        "verdict": verdict,
        "rationale": rationale,
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--root", type=Path, default=root,
                    help="repo root the default relative ckpt paths resolve against")
    ap.add_argument("--ckpt", type=Path, action="append", default=None,
                    help="override checkpoint list (repeatable)")
    args = ap.parse_args()
    ckpts = args.ckpt if args.ckpt else [Path(c) for c in _DEFAULT_CKPTS]
    out = run_probe(ckpts, args.root)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _main()
