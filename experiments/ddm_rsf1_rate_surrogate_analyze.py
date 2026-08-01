"""ddm_rsf1 analysis — rank fidelity of each in-loop rate surrogate vs REAL SMEVR bytes.

Decision statistic = Spearman rho (does the surrogate ORDER token fields the way the shipped
coder prices them?), with a percentile bootstrap CI. Pearson r on log(bytes) reported as the
magnitude-fidelity companion. Populations are reported SEPARATELY because trajectory rows are
serially correlated (effective n << nominal n) while the cross-config row set is ~independent.

Consumes the rows emitted by `experiments/ddm_rsf1_rate_surrogate_fidelity.py`.
Receipt (incl. the "what is NOT clean" section that bounds every number below):
`.omx/research/ddm_rsf1_rate_surrogate_fidelity_20260801.md`.

    .venv/bin/python experiments/ddm_rsf1_rate_surrogate_analyze.py \
        .omx/research/ddm_rsf1_rows_20260801.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SURR = [
    ("surr_entropy_bits", "entropy (LIVE DEFAULT)"),
    ("surr_smevr_surrogate_bits", "smevr_surrogate (BUILT; consec-delta)"),
    ("surr_modebase_bits", "mode-base residual (gd1 T4 derived cand.)"),
    ("surr_soft_occupancy", "soft mode-occupancy (SMEVR cost driver)"),
    ("mode_occupancy", "[oracle] hard mode-occupancy"),
]


def boot_spearman(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = len(x)
    out = np.empty(n)
    for i in range(n):
        k = rng.integers(0, m, m)
        if len(np.unique(x[k])) < 3 or len(np.unique(y[k])) < 3:
            out[i] = np.nan
            continue
        out[i] = stats.spearmanr(x[k], y[k]).statistic
    out = out[~np.isnan(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def report(rows, label, target="smevr_bytes"):
    y = np.array([r[target] for r in rows], float)
    print(f"\n--- {label}  (n={len(rows)}) ---")
    print(f"    {target}: min {y.min():,.0f}  max {y.max():,.0f}  "
          f"spread {(y.max()-y.min())/y.min()*100:.1f}% of min")
    print(f"    {'surrogate':44s} {'rho':>7s} {'95% CI':>18s} {'r(log bytes)':>13s}")
    res = {}
    for k, name in SURR:
        if k not in rows[0]:
            continue
        x = np.array([r[k] for r in rows], float)
        rho = stats.spearmanr(x, y).statistic
        lo, hi = boot_spearman(x, y)
        pr = stats.pearsonr(x, np.log(y)).statistic
        print(f"    {name:44s} {rho:+7.4f} [{lo:+6.3f},{hi:+6.3f}] {pr:+13.4f}")
        res[k] = (rho, lo, hi, pr)
    return res


def main() -> None:
    rows = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    print(f"loaded {len(rows)} fields")
    shapes = {tuple(r["shape"]) for r in rows}
    lv = {r["levels"] for r in rows}
    print(f"shapes: {shapes}   levels: {lv}   "
          f"(matched-shape => raw byte totals are directly comparable)")

    groups = {}
    for r in rows:
        for g in r["group"].split(","):
            groups.setdefault(g, []).append(r)

    report(rows, "ALL FIELDS (pooled; mixes 4 runs -> between-run scale dominates)")
    for g in sorted(groups):
        report(groups[g], g)

    # Within-trajectory concordance: the gradient-relevant question. For each trajectory,
    # does the surrogate move in the SAME DIRECTION as real bytes between adjacent states?
    print("\n\n=== WITHIN-TRAJECTORY DIRECTIONAL CONCORDANCE ===")
    print("(adjacent-checkpoint sign agreement: does the surrogate move with real SMEVR bytes?)")
    for g in sorted(groups):
        if not g.startswith(("B_", "C_", "D_")):
            continue
        rr = sorted(groups[g], key=lambda r: r["ckpt"])
        y = np.diff([r["smevr_bytes"] for r in rr])
        print(f"\n  {g} (n_steps={len(y)}; bytes rise in {int((y>0).sum())}/{len(y)} steps)")
        for k, name in SURR:
            if k not in rr[0]:
                continue
            d = np.diff([r[k] for r in rr])
            agree = float((np.sign(d) == np.sign(y)).mean())
            rho = stats.spearmanr([r[k] for r in rr], [r["smevr_bytes"] for r in rr]).statistic
            print(f"    {name:44s} sign-agree {agree*100:5.1f}%   rho(level) {rho:+.4f}")

    # SMEVR stream decomposition: which part of the shipped coder dominates the bytes?
    if "smevr_value_bytes" in rows[0]:
        print("\n\n=== SMEVR STREAM DECOMPOSITION (where the shipped bytes actually are) ===")
        for g in sorted(groups):
            rr = groups[g]
            b = np.mean([r["smevr_base_bytes"] for r in rr])
            o = np.mean([r["smevr_occupancy_bytes"] for r in rr])
            v = np.mean([r["smevr_value_bytes"] for r in rr])
            t = b + o + v
            print(f"  {g:22s} base {b:8.0f} ({100*b/t:4.1f}%)  occupancy {o:9.0f} ({100*o/t:4.1f}%)"
                  f"  value {v:9.0f} ({100*v/t:4.1f}%)")

    # The w_rate derivation's PREMISE, measured. `derive_w_rate_exchange_rate` assumes
    # "reduce the mean surrogate by 1 bit/token => save n_counted/8 bytes". Measure the
    # ACTUAL slope d(smevr_bytes)/d(surrogate) and compare.
    print("\n\n=== MEASURED bits->bytes EXCHANGE RATE vs the ASSUMED n/8 ===")
    # n_counted comes from the DSL's own geometry helper, never a retyped constant
    # (constants-are-poison). It is the BURN geometry (keep-384, c4, 600 pairs, shared_base);
    # rows from any other geometry are excluded from this block by the `mask` filter below.
    from tac.witness_dsl.spec_tr1_burn2_20260731 import burn_geometry_n_counted_tokens

    n_counted = burn_geometry_n_counted_tokens()
    print(f"  assumed slope (n_counted/8) = {n_counted/8:,.0f} bytes per 1 bit/token "
          f"[n_counted={n_counted:,}]")
    for g in sorted(groups):
        rr = [r for r in groups[g] if r.get("mask")]     # matched masked geometry only
        if len(rr) < 5:
            continue
        y = np.array([r["smevr_bytes"] for r in rr], float)
        for k, name in SURR[:3]:
            if k not in rr[0]:
                continue
            x = np.array([r[k] for r in rr], float)
            if np.ptp(x) == 0:
                continue
            sl = stats.linregress(x, y)
            print(f"  {g:22s} {name:42s} slope {sl.slope:+12,.0f} B/bit  R^2 {sl.rvalue**2:.3f}")

    # In-loop sampling noise floor: the term is a batch-8 estimate.
    print("\n\n=== IN-LOOP SAMPLING NOISE FLOOR (batch-8, the estimator the gradient sees) ===")
    for g in sorted(groups):
        rr = groups[g]
        for key, sd, nm in [("surr_entropy_bits", "batch8_entropy_std", "entropy"),
                            ("surr_smevr_surrogate_bits", "batch8_smevrsurr_std", "smevr_surrogate")]:
            spread = np.ptp([r[key] for r in rr])
            noise = float(np.mean([r[sd] for r in rr]))
            print(f"  {g:22s} {nm:16s} between-field spread {spread:.4f} bits | "
                  f"batch-8 std {noise:.4f} bits | SNR {spread/max(noise,1e-9):5.2f}x")


if __name__ == "__main__":
    main()
