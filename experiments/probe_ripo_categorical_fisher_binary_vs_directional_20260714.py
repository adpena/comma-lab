#!/usr/bin/env python3
"""MEASURE the RIPO categorical-Fisher trust-region falsification on REAL SegNet logits.

$0, cached-only (no training, no dispatch). Reproduces real K=5 SegNet logits from the
cached GT-frame cache (experiments/results/mlx_fleet_gt_cache/gt_n96.npz, gt_f1 = last
frame) through the AUTHORITY upstream SegNet forward, validates argmax against the cached
`lstars`, then compares the FALSE binary intake transfer  r_bin/sqrt(delta) = 1/sqrt(p_w)
against the CORRECT categorical-Fisher directional radius  r_dir/sqrt(delta_kl) = 2/sqrt(C_wr),
C_wr = p_w + p_r - (p_w - p_r)^2  (code-verified law:
src/tac/optimization/ripo_fisher_trust_region.py::winner_rival_curvature / winner_rival_radius).

Authority: [macOS-CPU advisory / NumPy-fp32; no score authority]. MEANS only; pointer UNMOVED.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "upstream"))

GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n96.npz"
OUT = REPO / ".omx/research/ripo_categorical_fisher_binary_vs_directional_measured_20260714.json"
N_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 96


def quant(a: np.ndarray, qs=(0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)) -> dict:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return {f"q{q}": float(np.quantile(a, q)) for q in qs}


def main() -> None:
    from modules import SegNet, segnet_sd_path  # authority forward
    from safetensors.torch import load_file

    d = np.load(GT_CACHE, allow_pickle=True)
    gt_f1 = d["gt_f1"][:N_PAIRS]  # (N,874,1164,3) uint8 -- the LAST frame SegNet reads
    lstars = d["lstars"][:N_PAIRS]  # (N,384,512) int64 cached SegNet argmax
    n = gt_f1.shape[0]

    torch.manual_seed(0)
    seg = SegNet().eval()
    seg.load_state_dict(load_file(str(segnet_sd_path), device="cpu"))

    pw_all, pr_all, marg_all = [], [], []
    argmax_match = 0
    argmax_total = 0
    bs = 4
    with torch.inference_mode():
        for i in range(0, n, bs):
            fr = gt_f1[i : i + bs]  # (b,H,W,3) uint8
            x = torch.from_numpy(fr.astype(np.float32))  # [0,255]
            x = x.permute(0, 3, 1, 2)[:, None, ...]  # (b,1,3,H,W) -> seq_len=1
            x = seg.preprocess_input(x)  # last frame + interpolate to (384,512)
            logits = seg(x)  # (b,5,384,512)
            am = logits.argmax(dim=1).cpu().numpy()  # (b,384,512)
            ls = lstars[i : i + bs]
            argmax_match += int((am == ls).sum())
            argmax_total += int(am.size)
            # softmax over class dim -> top-2 probs
            p = torch.softmax(logits, dim=1).cpu().numpy()  # (b,5,384,512)
            ps = np.sort(p, axis=1)  # ascending over classes
            pw = ps[:, -1, :, :]  # winner prob
            pr = ps[:, -2, :, :]  # runner-up prob
            z = logits.cpu().numpy()
            zs = np.sort(z, axis=1)
            marg = zs[:, -1, :, :] - zs[:, -2, :, :]  # winner-rival LOGIT margin
            pw_all.append(pw.ravel().astype(np.float64))
            pr_all.append(pr.ravel().astype(np.float64))
            marg_all.append(marg.ravel().astype(np.float64))
            print(f"  pairs {i}..{i+len(fr)-1} done", flush=True)

    pw = np.concatenate(pw_all)
    pr = np.concatenate(pr_all)
    marg = np.concatenate(marg_all)
    npix = pw.size
    argmax_frac = argmax_match / argmax_total

    # --- categorical-Fisher directional law (code-verified) ---
    c_wr = np.maximum(pw + pr - (pw - pr) ** 2, 0.0)
    # radii in units of sqrt(delta) (directional uses delta_kl; binary uses delta). Same budget.
    with np.errstate(divide="ignore"):
        r_dir = np.where(c_wr > 0, 2.0 / np.sqrt(c_wr), np.inf)  # |t|/sqrt(delta_kl) = 2/sqrt(C_wr)
    r_bin = 1.0 / np.sqrt(pw)  # ||dlogit||/sqrt(delta) = 1/sqrt(p_w)  (FALSE binary transfer)
    ratio = r_dir / r_bin  # = 2*sqrt(p_w / C_wr)

    finite = np.isfinite(ratio)
    # over/under admit: binary predicts a LARGER radius than directional => over-admit
    over_admit_frac = float((r_bin > r_dir).mean())  # binary radius exceeds true Fisher radius

    # --- ANNULUS vs INTERIOR (the rank-reversal falsification) ---
    # annulus = flip-prone boundary (small logit margin); interior = confident.
    # threshold on the winner-rival logit margin; boundary annulus per L66 ~ small margin.
    ann = marg < 0.5  # near-tie / boundary band
    interior = marg > 4.0  # confident interior
    def med(a):
        return float(np.median(a)) if a.size else float("nan")

    # Spearman rank correlation between the two radius fields (subsample for speed)
    rng = np.random.default_rng(0)
    idx = rng.choice(npix, size=min(200_000, npix), replace=False)
    rb, rd = r_bin[idx], r_dir[idx]
    fin = np.isfinite(rd)
    rb, rd = rb[fin], rd[fin]
    def rankdata(a):
        order = np.argsort(a, kind="stable")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a))
        return ranks
    spearman = float(np.corrcoef(rankdata(rb), rankdata(rd))[0, 1])

    report = {
        "authority": "[macOS-CPU advisory / NumPy-fp32; no score authority]",
        "means_only": True,
        "pointer_delta": 0.0,
        "law_source": "src/tac/optimization/ripo_fisher_trust_region.py::winner_rival_{curvature,radius}",
        "law_directional": "|t|/sqrt(delta_kl) = 2/sqrt(C_wr), C_wr=p_w+p_r-(p_w-p_r)^2",
        "law_binary_false": "||dlogit||/sqrt(delta) = 1/sqrt(p_w)  (RIPO Eq.10 scalar-action transfer, K=5-invalid)",
        "n_pairs": int(n),
        "n_pixels": int(npix),
        "validation_argmax_match_frac_vs_cached_lstars": argmax_frac,
        "operating_point_pw": quant(pw),
        "operating_point_pr": quant(pr),
        "operating_point_logit_margin": quant(marg),
        "C_wr": quant(c_wr),
        "r_dir_over_sqrt_delta": quant(r_dir[np.isfinite(r_dir)]),
        "r_bin_over_sqrt_delta": quant(r_bin),
        "ratio_dir_over_bin": quant(ratio[finite]),
        "ratio_min_worstcase": float(np.min(ratio[finite])),
        "ratio_max_worstcase": float(np.max(ratio[finite])),
        "binary_over_admit_frac": over_admit_frac,
        "spearman_rank_corr_rbin_vs_rdir": spearman,
        "annulus_margin_lt_0p5": {
            "n": int(ann.sum()),
            "median_pw": med(pw[ann]),
            "median_C_wr": med(c_wr[ann]),
            "median_r_bin": med(r_bin[ann]),
            "median_r_dir": med(r_dir[ann][np.isfinite(r_dir[ann])]),
        },
        "interior_margin_gt_4": {
            "n": int(interior.sum()),
            "median_pw": med(pw[interior]),
            "median_C_wr": med(c_wr[interior]),
            "median_r_bin": med(r_bin[interior]),
            "median_r_dir": med(r_dir[interior][np.isfinite(r_dir[interior])]),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print("\n=== SUMMARY ===")
    print(f"argmax match vs cached lstars: {argmax_frac:.6f}  (n_pix={npix})")
    print(f"p_w median {report['operating_point_pw']['q0.5']:.4f}  q01 {report['operating_point_pw']['q0.01']:.4f}")
    print(f"ratio r_dir/r_bin: median {report['ratio_dir_over_bin']['q0.5']:.3f}  "
          f"q01 {report['ratio_dir_over_bin']['q0.01']:.3f}  max {report['ratio_max_worstcase']:.2f}")
    print(f"binary over-admit frac (r_bin>r_dir): {over_admit_frac:.4g}")
    print(f"Spearman rank corr r_bin vs r_dir: {spearman:.4f}")
    a, it = report["annulus_margin_lt_0p5"], report["interior_margin_gt_4"]
    print(f"ANNULUS  (n={a['n']}): r_bin~{a['median_r_bin']:.3f}  r_dir~{a['median_r_dir']:.3f}")
    print(f"INTERIOR (n={it['n']}): r_bin~{it['median_r_bin']:.3f}  r_dir~{it['median_r_dir']:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
