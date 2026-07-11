#!/usr/bin/env python3
"""Task #442 WeightWatcher / WW-PGD Phase-1 $0 diagnostic gate.

MEASURES per-layer HTSR spectral metrics across the #205 mod32cap curriculum
trajectory (tau_crossover frozen checkpoints at KNOWN events) and tests three
correlation hypotheses that decide whether porting WW-PGD as a witness lever is
worth building:

  H1 REGIME/SENSE : do per-layer alpha / detX / stable-rank trajectories move at
                    the known curriculum events (CE-end ep299, tau-best ep650,
                    Muon-start ep726, ep925, final ep1000) beyond layer spread?
  H2 RATE         : does a spectral metric predict weight CODABILITY (brotli-11
                    compressed size of int8-quantized weight tensors)?  The
                    counted archive bytes ARE trunk weights + `code`, so a
                    spectral<->codability law would make WW a rate-axis signal.
  H3 QUALITY      : does mean trunk alpha correlate with the measured d_seg
                    trajectory (tau_crossover verdict rows)?

Estimators are implemented directly (no pip weightwatcher/powerlaw available):
  * alpha  : Clauset-Shalizi-Newman (2009) continuous power-law MLE (Hill form)
             alpha = 1 + n_tail / sum(ln(lambda_i / xmin)); xmin chosen by
             minimizing the Kolmogorov-Smirnov distance D between the empirical
             tail CDF and the fitted power-law CDF (CSN xmin selection).
             Interpretation: Martin & Mahoney HTSR (alpha in [2,4] well-trained;
             alpha -> 2 critical/ideal; alpha >> 6 under-trained / random-like).
  * ks_D   : the KS distance at the chosen xmin = FIT QUALITY.  At width 96 the
             ESD has <=96 eigenvalues; large D flags alpha as NOISE-dominated.
  * detX_num (operational): ERG trace-log tail size = # top eigenvalues k with
             product(top-k lambda) >= 1  (sum log lambda >= 0).  Documented as an
             operational reimplementation, NOT a verified match to WW internals.
  * stable_rank = ||W||_F^2 / ||W||_2^2 = sum(lambda) / lambda_max
  * spectral_norm = sigma_max ; log_spectral_norm = log10(lambda_max)
  * alpha_weighted = alpha * log10(lambda_max)   (WW weighted-alpha)
  * spectral_entropy = -sum p_i log p_i, p_i = lambda_i / sum(lambda)

All numbers ADVISORY NON-PROMOTABLE. Pointer 0.19108282 [contest-CPU] UNMOVED.
Deterministic: pure linalg + deterministic brotli; no RNG.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    import brotli
except Exception as exc:  # pragma: no cover
    print(f"brotli required: {exc}", file=sys.stderr)
    sys.exit(2)

REPO = Path(__file__).resolve().parents[1]
TAU_DIR = REPO / "experiments/results/tau_crossover_trainflow_20260707"
TAU_JSON = TAU_DIR / "tau_crossover_trainflow_n600_20260707.json"

# curriculum-event-labeled frozen checkpoints (one parent run: mod32cap #205)
CKPTS = [
    ("ep299_CEend", 299, TAU_DIR / "frozen_ep299_CEend.npz"),
    ("ep650_tauBest", 650, TAU_DIR / "frozen_ep650_tauBest.npz"),
    ("ep726_MuonStart", 726, TAU_DIR / "frozen_ep726_MuonStart.npz"),
    ("ep925_liveEMA", 925, TAU_DIR / "frozen_ep925_liveEMA.npz"),
    ("final_ep1000", 1000, TAU_DIR / "frozen_final_ep1000.npz"),
]
# square trunk weight matrices eligible for alpha (>=~64 eigenvalues)
TRUNK_SQUARE = ["in_proj.weight", "hidden.0.weight", "hidden.1.weight",
                "hidden.2.weight", "hidden.3.weight"]
# additional matrices for codability only (too few eigenvalues for a trusted alpha)
EXTRA_CODE = ["film.weight", "code"]


@dataclass
class LayerMetrics:
    ckpt: str
    epoch: int
    layer: str
    shape: str
    n_eig: int
    alpha: float
    ks_D: float
    xmin: float
    num_pl_spikes: int
    alpha_weighted: float
    log_spectral_norm: float
    spectral_norm: float
    stable_rank: float
    detX_num: int
    spectral_entropy: float
    n_params: int
    brotli_int8_bytes: int
    bits_per_param: float


def eigvals_of_W(W: np.ndarray) -> np.ndarray:
    """ESD eigenvalues lambda = sigma^2 of X = W^T W, descending, positive."""
    # SVD is numerically stabler than forming W^T W for small matrices.
    s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    lam = s * s
    lam = np.sort(lam)[::-1]
    return lam[lam > 1e-12]


def fit_powerlaw_csn(lam: np.ndarray):
    """Clauset-Shalizi-Newman continuous power-law MLE with KS xmin selection.

    Returns (alpha, xmin, ks_D, n_tail). Fits over the ESD tail (largest lambda).
    """
    lam = np.sort(lam)  # ascending
    n = lam.size
    if n < 8:
        return math.nan, math.nan, math.nan, 0
    uniq = np.unique(lam)
    # candidate xmin: distinct values, leaving at least 4 points in the tail
    best = None
    for xmin in uniq[:-4]:
        tail = lam[lam >= xmin]
        nt = tail.size
        if nt < 4:
            continue
        logsum = np.sum(np.log(tail / xmin))
        if logsum <= 0:
            continue
        alpha = 1.0 + nt / logsum
        # empirical CDF vs power-law CDF on tail; KS distance
        srt = np.sort(tail)
        cdf_emp = np.arange(1, nt + 1) / nt
        cdf_fit = 1.0 - (srt / xmin) ** (1.0 - alpha)
        D = np.max(np.abs(cdf_emp - cdf_fit))
        if best is None or D < best[2]:
            best = (alpha, xmin, D, nt)
    if best is None:
        return math.nan, math.nan, math.nan, 0
    return best


def detX_num_erg(lam: np.ndarray) -> int:
    """Operational ERG tail size: # top eigenvalues whose product >= 1
    (sum log lambda >= 0). NOT a verified match to WW's internal detX."""
    csum = np.cumsum(np.log(lam))  # lam descending
    k = int(np.sum(csum >= 0.0))
    return k


def spectral_entropy(lam: np.ndarray) -> float:
    p = lam / np.sum(lam)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def brotli_int8(W: np.ndarray) -> int:
    """Deterministic codability proxy: per-tensor int8 quant then brotli q=11.
    Mirrors PR95-family L21/L29 (int8 codes + per-tensor fp16 scale)."""
    amax = np.max(np.abs(W))
    if amax == 0:
        q = np.zeros(W.shape, dtype=np.int8)
    else:
        scale = amax / 127.0
        q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return len(brotli.compress(q.tobytes(), quality=11))


def analyze_matrix(ckpt, epoch, layer, W) -> LayerMetrics:
    lam = eigvals_of_W(W)
    alpha, xmin, ksD, nt = fit_powerlaw_csn(lam)
    lam_max = float(lam[0])
    log_snorm = math.log10(lam_max)
    bbytes = brotli_int8(W)
    nparam = int(W.size)
    return LayerMetrics(
        ckpt=ckpt, epoch=epoch, layer=layer, shape="x".join(map(str, W.shape)),
        n_eig=int(lam.size),
        alpha=float(alpha), ks_D=float(ksD), xmin=float(xmin),
        num_pl_spikes=int(nt),
        alpha_weighted=float(alpha * log_snorm) if not math.isnan(alpha) else math.nan,
        log_spectral_norm=float(log_snorm),
        spectral_norm=float(math.sqrt(lam_max)),
        stable_rank=float(np.sum(lam) / lam_max),
        detX_num=detX_num_erg(lam),
        spectral_entropy=spectral_entropy(lam),
        n_params=nparam,
        brotli_int8_bytes=bbytes,
        bits_per_param=float(8.0 * bbytes / nparam),
    )


def pearson_spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3:
        return math.nan, math.nan, n
    # Pearson
    r = np.corrcoef(x, y)[0, 1]
    # Spearman (rank Pearson)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rho = np.corrcoef(rx, ry)[0, 1]
    return float(r), float(rho), n


def main():
    d_seg_by_label = {}
    if TAU_JSON.exists():
        tj = json.load(open(TAU_JSON))
        for r in tj.get("table", []):
            d_seg_by_label[r["label"]] = r["d_seg"]

    all_rows: list[LayerMetrics] = []
    for label, epoch, path in CKPTS:
        if not path.exists():
            print(f"MISSING {path}", file=sys.stderr)
            continue
        z = np.load(path, allow_pickle=True)
        for layer in TRUNK_SQUARE + EXTRA_CODE:
            if layer not in z.files:
                continue
            W = np.asarray(z[layer])
            if W.ndim != 2:
                continue
            all_rows.append(analyze_matrix(label, epoch, layer, W))

    rows = [asdict(r) for r in all_rows]

    # ---------------- H1: regime/sense trajectory (square trunk only) --------
    trunk = [r for r in all_rows if r.layer in TRUNK_SQUARE]
    labels = [c[0] for c in CKPTS]
    H1 = {}
    for metric in ["alpha", "stable_rank", "detX_num", "log_spectral_norm", "spectral_entropy"]:
        traj = {}
        for label, epoch, _ in CKPTS:
            vals = [getattr(r, metric) for r in trunk if r.ckpt == label
                    and math.isfinite(getattr(r, metric))]
            traj[label] = (float(np.mean(vals)) if vals else math.nan,
                           float(np.std(vals)) if vals else math.nan)
        # movement across trajectory vs within-checkpoint layer spread (noise floor)
        means = np.array([traj[l][0] for l in labels])
        spreads = np.array([traj[l][1] for l in labels])
        rng = float(np.nanmax(means) - np.nanmin(means))
        noise = float(np.nanmedian(spreads))
        H1[metric] = {"trajectory": traj, "across_ckpt_range": rng,
                      "within_ckpt_layer_spread_median": noise,
                      "range_over_noise": (rng / noise) if noise > 0 else math.nan}

    # ---------------- H2: codability (pooled layer x ckpt) -------------------
    # Pool ALL matrices (max N) then also square-trunk-only (clean alpha subset).
    def h2(subset, name):
        out = {}
        for metric in ["alpha", "stable_rank", "spectral_norm", "log_spectral_norm",
                       "spectral_entropy", "detX_num", "alpha_weighted"]:
            xs = [getattr(r, metric) for r in subset]
            ys = [r.bits_per_param for r in subset]
            r_p, rho, n = pearson_spearman(xs, ys)
            out[metric + "_vs_bits_per_param"] = {"pearson": r_p, "spearman": rho, "n": n}
        return out
    H2 = {
        "pooled_all_layers": h2(all_rows, "all"),
        "square_trunk_only": h2(trunk, "trunk"),
        "note": "bits_per_param = 8*brotli_int8_bytes/n_params (codability proxy; "
                "lower=more compressible=cheaper counted bytes)",
    }

    # ---------------- H3: quality (mean trunk alpha vs d_seg) ----------------
    H3 = {}
    xs, ys, used = [], [], []
    for label, epoch, _ in CKPTS:
        if label not in d_seg_by_label:
            continue
        a = [r.alpha for r in trunk if r.ckpt == label and math.isfinite(r.alpha)]
        if not a:
            continue
        xs.append(float(np.mean(a))); ys.append(d_seg_by_label[label]); used.append(label)
    r_p, rho, n = pearson_spearman(xs, ys)
    H3 = {"mean_trunk_alpha_vs_d_seg": {"pearson": r_p, "spearman": rho, "n": n},
          "checkpoints_used": used,
          "pairs": list(zip(used, [round(v, 4) for v in xs], [round(v, 6) for v in ys]))}

    # ---------------- fit-quality honesty ------------------------------------
    ks_trunk = [r.ks_D for r in trunk if math.isfinite(r.ks_D)]
    fit_quality = {
        "square_trunk_median_ks_D": float(np.median(ks_trunk)) if ks_trunk else math.nan,
        "square_trunk_max_ks_D": float(np.max(ks_trunk)) if ks_trunk else math.nan,
        "n_eig_square_trunk": trunk[0].n_eig if trunk else None,
        "interpretation": "KS D >~0.10 at width-96 => alpha is NOISE-dominated; "
                          "verdict scope INSTANCE/FORMULATION not FAMILY.",
    }

    result = {
        "task": "#442 WW-PGD Phase-1 spectral diagnostic gate",
        "axis_tag": "[macOS-CPU advisory] NON-PROMOTABLE",
        "pointer": "0.19108282 [contest-CPU] UNMOVED (this is MEANS)",
        "lineage": "levelset_n600_witness_mod32cap_20260706T115554Z (#205 mod32cap)",
        "known_events": "CE-end ep299, tau-best ep650, Muon-start ep726, ep925, final ep1000",
        "estimators": "CSN power-law MLE + KS xmin; ERG detX operational; Martin-Mahoney HTSR",
        "fit_quality": fit_quality,
        "H1_regime_sense": H1,
        "H2_rate_codability": H2,
        "H3_quality_dseg": H3,
        "per_layer_rows": rows,
    }
    outpath = TAU_DIR.parent.parent / "reports" / "ww_pgd_442_phase1_metrics.json"
    outpath.parent.mkdir(exist_ok=True)
    json.dump(result, open(outpath, "w"), indent=1)
    print(f"WROTE {outpath}")

    # ------- console summary -------
    print("\n=== FIT QUALITY (honesty) ===")
    print(json.dumps(fit_quality, indent=1))
    print("\n=== H1 range/noise (square trunk, mean over 5 layers) ===")
    for m, v in H1.items():
        print(f"  {m:18s} range={v['across_ckpt_range']:.4g} "
              f"noise={v['within_ckpt_layer_spread_median']:.4g} "
              f"range/noise={v['range_over_noise']:.3g}")
        for l in labels:
            mu, sd = v["trajectory"][l]
            print(f"      {l:16s} {mu:.4g} (+/-{sd:.3g})")
    print("\n=== H2 codability correlations (metric vs bits/param) ===")
    for subset_name, sub in [("pooled_all", H2["pooled_all_layers"]),
                             ("trunk_only", H2["square_trunk_only"])]:
        print(f"  [{subset_name}]")
        for k, v in sub.items():
            print(f"    {k:42s} pearson={v['pearson']:+.3f} spearman={v['spearman']:+.3f} n={v['n']}")
    print("\n=== H3 quality (mean trunk alpha vs d_seg) ===")
    print(json.dumps(H3, indent=1))


if __name__ == "__main__":
    main()
