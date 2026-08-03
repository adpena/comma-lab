# SPDX-License-Identifier: MIT
"""ddm_hg1 PROBE 1 -- two questions about the 2026-07-08 q1_signed_asymmetry artifact.

Q1 (INSTRUMENT VALIDITY).  The correlator's own docstring declares sR a POSITIVE-CONTROL
    sentinel: "if sR is ALSO at chance vs flips the instrument is untrusted and no verdict
    is admissible".  The landed artifact shows |rho_sR| <= 0.081 on EVERY major pair while
    |rho_margin| >= 0.24 -- i.e. the sentinel FAILED -- yet the verdict block was emitted
    anyway (the sentinel is metadata in the JSON, never consulted by the verdict code).
    Two competing explanations, and they have opposite consequences:
      H1  sR genuinely does not predict flips (the DEPLOYED replacement for the retired
          texture proxy is itself at chance) -> a real, decisive negative about sR.
      H2  sR is MISINDEXED.  The correlator asserts frame alignment for `lstars`
          (line ~198) but has NO equivalent assert for `sR`; a wrong stride would look
          exactly like "at chance".
    DISCRIMINATOR: sR is built FROM the margin geometry (fragility-weighted margin-Jacobian
    reachability).  The correctly-aligned sR frame must be structurally far more similar to
    its own frame's margin/label field than to any other frame's.  Sweep the candidate index
    maps and a random-offset control; if some map is sharply better than the deployed g=6w,
    that is H2 and the whole artifact is void.

Q2 (THE POOLING MECHANISM).  The asymmetry addendum's claim is that "an unsigned pooled
    estimator of a signed density has zero expectation when the two sides carry opposite
    signs".  The artifact appears to contain that mechanism caught in the act
    (Movable->Road rho=+0.1147 vs Movable->Undrivable rho=-0.1242 on the SAME source class
    and the SAME texture field).  Recompute the POOLED rho exactly from the directed
    accumulators and compare it to the directed ones.  Exact recomputation requires the
    running sums, so re-accumulate from the caches rather than trusting the JSON.

$0, scorer-free, cached artifacts only.  [macOS-numpy advisory . NON-PROMOTABLE]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

SEG_HW = (384, 512)
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
MAJOR = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (2, 3), (3, 2)]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    dx = float((xc * xc).sum())
    dy = float((yc * yc).sum())
    if dx <= 0.0 or dy <= 0.0:
        return float("nan")
    return float((xc * yc).sum() / np.sqrt(dx * dy))


def main() -> int:
    out: dict = {"probe": "hg1_probe1_sr_alignment_and_pooling",
                 "advisory": "[macOS-numpy advisory . NON-PROMOTABLE] pointer 0.1910828242 UNMOVED"}

    gt = np.load(REPO / "experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz", mmap_mode="r")
    lstars, margins = gt["lstars"], gt["margins"]
    sR = np.load(REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600_sR.npz", mmap_mode="r")["sR"]
    wdir = REPO / "experiments/results/witness_per_stage_attribution"
    w_argmax = np.asarray(np.load(wdir / "maps_MuonBest.npz")["argmax"])
    gt_sub = np.load(wdir / "_gt_argmax_subset.npy", mmap_mode="r")

    n_w = w_argmax.shape[0]
    out["n_witness_frames"] = int(n_w)
    out["shapes"] = {"lstars": list(lstars.shape), "margins": list(margins.shape),
                     "sR": list(sR.shape), "w_argmax": list(w_argmax.shape)}

    # ---- Q1: sR alignment discrimination -------------------------------------------------
    # For a sample of witness frames, correlate sR[candidate index] against that frame's own
    # margin field.  sR is derived from the margin geometry, so the aligned index must win.
    rng = np.random.default_rng(20260803)
    probe_w = list(range(0, n_w, 8))  # 12 frames, deterministic
    cand = {"g=6w (DEPLOYED)": lambda w: 6 * w, "g=2w": lambda w: 2 * w,
            "g=3w": lambda w: 3 * w, "g=w": lambda w: w, "g=w+1": lambda w: w + 1}
    align: dict[str, list[float]] = {k: [] for k in cand}
    align["RANDOM control"] = []
    for w in probe_w:
        s_idx = 2 * w
        m = np.asarray(margins[s_idx], dtype=np.float64)
        for k, f in cand.items():
            g = f(w)
            if 0 <= g < sR.shape[0]:
                align[k].append(pearson(np.asarray(sR[g], dtype=np.float64), m))
        g_rand = int(rng.integers(0, sR.shape[0]))
        align["RANDOM control"].append(pearson(np.asarray(sR[g_rand], dtype=np.float64), m))
    out["q1_sr_alignment_rho_sR_vs_own_margin"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        for k, v in align.items() if v}

    # sR field health: does it even have variance / dynamic range?
    st = []
    for w in probe_w:
        a = np.asarray(sR[6 * w], dtype=np.float64)
        st.append((float(a.min()), float(a.max()), float(a.mean()), float(a.std()),
                   float((a <= 0).mean()), float((a >= 1.0 - 1e-9).mean())))
    st = np.asarray(st)
    out["q1_sr_field_stats_at_deployed_index"] = {
        "min": float(st[:, 0].mean()), "max": float(st[:, 1].mean()),
        "mean": float(st[:, 2].mean()), "std": float(st[:, 3].mean()),
        "frac_le_zero": float(st[:, 4].mean()), "frac_saturated_at_1": float(st[:, 5].mean())}

    # ---- Q2: pooled-vs-directed on the SAME accumulators ---------------------------------
    from scipy.ndimage import binary_dilation
    from signed_flip_asymmetry_correlator import compute_texture_fields

    gt_f1 = gt["gt_f1"]
    R = 2
    struct = np.ones((2 * R + 1, 2 * R + 1), dtype=bool)
    # directed accumulator sums + a POOLED-BY-SOURCE-CLASS accumulator (the unsigned pooled shape)
    dir_sums: dict[tuple[int, int], dict[str, float]] = {}
    pool_sums: dict[int, dict[str, float]] = {}
    all_sums: dict[str, float] = {"n": 0.0, "sx": 0.0, "sxx": 0.0, "sy": 0.0, "syy": 0.0, "sxy": 0.0}

    def acc(d: dict[str, float], x: np.ndarray, y: np.ndarray) -> None:
        d["n"] += x.size
        d["sx"] += float(x.sum())
        d["sxx"] += float((x * x).sum())
        d["sy"] += float(y.sum())
        d["syy"] += float((y * y).sum())
        d["sxy"] += float((x * y).sum())

    def rho_of(d: dict[str, float]) -> float:
        n = d["n"]
        if n < 2:
            return float("nan")
        cov = d["sxy"] - d["sx"] * d["sy"] / n
        vx = d["sxx"] - d["sx"] ** 2 / n
        vy = d["syy"] - d["sy"] ** 2 / n
        if vx <= 0 or vy <= 0:
            return float("nan")
        return float(cov / np.sqrt(vx * vy))

    def new() -> dict[str, float]:
        return {"n": 0.0, "sx": 0.0, "sxx": 0.0, "sy": 0.0, "syy": 0.0, "sxy": 0.0}

    for w in range(n_w):
        s_idx = 2 * w
        gta = np.asarray(lstars[s_idx], dtype=np.int64)
        if not np.array_equal(gta.astype(np.int8), np.asarray(gt_sub[w])):
            raise RuntimeError(f"lstars alignment broken at witness frame {w}")
        wa = w_argmax[w].astype(np.int64)
        _tex, texprox = compute_texture_fields(gt_f1[s_idx], 4.0)
        present = np.unique(gta)
        for ci in present:
            mi = gta == ci
            for cj in present:
                if cj == ci:
                    continue
                near = binary_dilation(gta == cj, structure=struct)
                pop = mi & near & ((wa == ci) | (wa == cj))
                if not pop.any():
                    continue
                x = texprox[pop]
                y = (wa == cj)[pop].astype(np.float64)
                key = (int(ci), int(cj))
                dir_sums.setdefault(key, new())
                acc(dir_sums[key], x, y)
                pool_sums.setdefault(int(ci), new())
                acc(pool_sums[int(ci)], x, y)
                acc(all_sums, x, y)

    out["q2_directed"] = {
        f"{i}->{j}": {"from": CLASS_NAMES[i], "to": CLASS_NAMES[j], "n": int(d["n"]),
                      "flips": int(d["sy"]), "flip_rate": d["sy"] / d["n"],
                      "rho_texprox": rho_of(d), "major": (i, j) in MAJOR}
        for (i, j), d in sorted(dir_sums.items())}
    out["q2_pooled_by_source_class"] = {
        CLASS_NAMES[c]: {"n": int(d["n"]), "flips": int(d["sy"]), "rho_texprox": rho_of(d)}
        for c, d in sorted(pool_sums.items())}
    out["q2_pooled_all"] = {"n": int(all_sums["n"]), "flips": int(all_sums["sy"]),
                            "rho_texprox": rho_of(all_sums)}

    # cancellation ledger: for each source class, max directed |rho| vs the pooled |rho|
    canc = {}
    for c, d in sorted(pool_sums.items()):
        sides = {f"{c}->{j}": rho_of(dd) for (i, j), dd in dir_sums.items() if i == c}
        finite = [v for v in sides.values() if np.isfinite(v)]
        if not finite:
            continue
        canc[CLASS_NAMES[c]] = {
            "pooled_rho": rho_of(d),
            "max_abs_directed_rho": float(max(abs(v) for v in finite)),
            "directed_sides": sides,
            "sign_split": bool(max(finite) > 0 > min(finite)),
            "cancellation_factor": (float(max(abs(v) for v in finite) / abs(rho_of(d)))
                                    if abs(rho_of(d)) > 1e-12 else float("inf"))}
    out["q2_cancellation_ledger"] = canc

    dst = REPO / ".omx/research/ddm_hg1_probe1_sr_alignment_and_pooling.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items()
                      if k.startswith("q1") or k in ("q2_pooled_all", "q2_cancellation_ledger")}, indent=1))
    print(f"[done] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
