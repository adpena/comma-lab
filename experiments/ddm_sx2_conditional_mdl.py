"""ddm_sx2 -- price ``ddm_sx1``'s S2 correctly: measure H(L* | FREE prediction), not 253,341*(1-rho).

``ddm_sx1`` §8 costed the S2 class ("generic separatrix predictor + counted placement residual") as

    cost = 253,341 B * (1 - rho)

where ``rho`` is the recall of a generic contour extractor against the GT separatrix. That formula
conflates two different quantities. ``rho`` is a recall on boundary LOCATION; the 253,341 B is a
CODE LENGTH. Knowing where the boundary runs does not tell you which of the five labels sits on
each side of it, nor its sub-pixel placement, and the two are not proportional. The quantity S2
actually needs is the CONDITIONAL code length of the label field given the free prediction.

This module measures it directly with the same instrument that produced the 253,341 B: an order-4
causal context model over ``lstars``. It reports

  1. the UNCONDITIONAL code length (a re-derivation cross-check of ``sx1``'s 253,341 B, computed
     here from scratch rather than confirmed by reading), and
  2. the CONDITIONAL code length with the free prediction admitted as an extra context symbol,

so the bits the free predictor actually buys are read off as a difference, not asserted.

Axis: [macOS-CPU advisory]. NO contest scorer forward. score_claim=false.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import cv2
import numpy as np
import torch

SCORER_H = 384
SCORER_W = 512
N_CLASSES = 5
DEFAULT_LSTARS = "experiments/results/ot_offset_n600_modal_20260709/gt_n600_lstars_slim.npz"
DEFAULT_FRAMES = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _resize_to_scorer_grid(frame_hwc_u8: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(frame_hwc_u8).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(SCORER_H, SCORER_W), mode="bilinear")
    return t.squeeze(0).permute(1, 2, 0).numpy()


_K3 = np.ones((3, 3), np.uint8)


def _canny_edges(rgb: np.ndarray, density: float) -> np.ndarray:
    """The winning generic extractor from the G2 race (ddm_sx2_g2_contour_hitrate.py)."""
    y = np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], 0, 255).astype(np.uint8)
    hi = float(np.percentile(y, 90))
    e = cv2.Canny(y, 0.4 * hi, hi, L2gradient=True) > 0
    yf = y.astype(np.float32)
    gx = cv2.Sobel(yf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(yf, cv2.CV_32F, 0, 1, ksize=3)
    score = np.hypot(gx, gy) * e
    n = score.size
    k = max(1, round(density * n))
    keep = np.argpartition(score.ravel(), n - k)[n - k :]
    flat = np.zeros(n, dtype=bool)
    flat[keep] = True
    return flat.reshape(score.shape)


def _padded(labels: np.ndarray) -> np.ndarray:
    """Pad with a 6th 'outside' symbol so border pixels have a well-defined causal context."""
    p = np.full((SCORER_H + 1, SCORER_W + 2), N_CLASSES, dtype=np.int64)
    p[1:, 1:-1] = labels
    return p


def _context_index(padded: np.ndarray) -> np.ndarray:
    """Order-4 causal context: (west, north, north-west, north-east), each in 0..5."""
    w = padded[1:, 0:-2]
    n = padded[0:-1, 1:-1]
    nw = padded[0:-1, 0:-2]
    ne = padded[0:-1, 2:]
    return ((w * 6 + n) * 6 + nw) * 6 + ne


N_CTX = 6**4


def _code_length_bits(counts: np.ndarray, n_ctx: int, n_sym: int) -> tuple[float, float]:
    """Empirical (ideal) code length + a Krichevsky-Trofimov model cost for the used contexts.

    Returns (data_bits, model_bits). The model term charges 0.5*log2(N_ctx_total) per free
    parameter over contexts that were actually used, which is the standard two-part MDL charge --
    without it a context model can 'win' by memorising and the floor is not a floor.
    """
    c = counts.reshape(n_ctx, n_sym).astype(np.float64)
    tot = c.sum(axis=1)
    used = tot > 0
    data_bits = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(c > 0, c / np.maximum(tot[:, None], 1.0), 1.0)
        data_bits = float(-(c * np.log2(p)).sum())
    n_used = int(used.sum())
    n_total = float(tot.sum())
    model_bits = 0.5 * (n_sym - 1) * n_used * math.log2(max(n_total, 2.0))
    return data_bits, model_bits


def run(n_pairs: int, lstars_path: str, frames_path: str, out_json: str) -> dict:
    lz = np.load(lstars_path)
    lstars = lz["lstars"].astype(np.int64)
    fz = np.load(frames_path)
    gt_f1 = fz["gt_f1"]
    n = int(min(n_pairs, lstars.shape[0], gt_f1.shape[0]))

    # boundary density, to give the free extractor the same budget the G2 race used
    def boundary(lab):
        b = np.zeros(lab.shape, dtype=bool)
        b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
        b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
        b[:-1, :] |= lab[:-1, :] != lab[1:, :]
        b[1:, :] |= lab[:-1, :] != lab[1:, :]
        return b

    gt_b_total = sum(int(boundary(lstars[i]).sum()) for i in range(n))
    density = gt_b_total / (n * SCORER_H * SCORER_W)

    uncond = np.zeros(N_CTX * N_CLASSES, dtype=np.int64)
    # conditional context = order-4 label context x (free predictor fired at this pixel?)
    cond = np.zeros(N_CTX * 2 * N_CLASSES, dtype=np.int64)
    # dilated-predictor variant: the predictor is only accurate to +/-1 px, so a dilated indicator
    # is the honest side-information a decoder could actually act on.
    cond_d = np.zeros(N_CTX * 2 * N_CLASSES, dtype=np.int64)

    for i in range(n):
        lab = lstars[i]
        ctx = _context_index(_padded(lab)).ravel()
        sym = lab.ravel()
        np.add.at(uncond, ctx * N_CLASSES + sym, 1)

        rgb = _resize_to_scorer_grid(gt_f1[i])
        pred = _canny_edges(rgb, density)
        pd = cv2.dilate(pred.astype(np.uint8), _K3, iterations=1).astype(bool)
        np.add.at(cond, (ctx * 2 + pred.ravel().astype(np.int64)) * N_CLASSES + sym, 1)
        np.add.at(cond_d, (ctx * 2 + pd.ravel().astype(np.int64)) * N_CLASSES + sym, 1)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    u_data, u_model = _code_length_bits(uncond, N_CTX, N_CLASSES)
    c_data, c_model = _code_length_bits(cond, N_CTX * 2, N_CLASSES)
    d_data, d_model = _code_length_bits(cond_d, N_CTX * 2, N_CLASSES)

    def pack(data_bits, model_bits, label):
        return {
            "label": label,
            "data_bytes": data_bits / 8.0,
            "model_bytes": model_bits / 8.0,
            "total_bytes": (data_bits + model_bits) / 8.0,
        }

    u = pack(u_data, u_model, "unconditional order-4")
    c = pack(c_data, c_model, "order-4 + free canny indicator")
    d = pack(d_data, d_model, "order-4 + free canny indicator, dilated +/-1px")

    best_cond = min(c["total_bytes"], d["total_bytes"])
    out = {
        "arm": "ddm_sx2",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "scorer_forwards_run": 0,
        "substrate": "lstars GT argmax + GT frame_1 through SegNet.preprocess_input (CEILING)",
        "n_frames": n,
        "predictor_density": density,
        "unconditional": u,
        "conditional_raw": c,
        "conditional_dilated": d,
        "bits_bought_by_free_predictor_bytes": u["total_bytes"] - best_cond,
        "fraction_of_description_bought": (u["total_bytes"] - best_cond) / u["total_bytes"],
        "sx1_naive_formula_bytes": None,  # filled by caller context; see memo
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--lstars", default=DEFAULT_LSTARS)
    ap.add_argument("--frames", default=DEFAULT_FRAMES)
    ap.add_argument("--out", default=".omx/research/ddm_sx2_conditional_mdl.json")
    a = ap.parse_args()
    o = run(a.n_pairs, a.lstars, a.frames, a.out)
    print(f"\nframes={o['n_frames']}  predictor density={o['predictor_density']:.6f}")
    for k in ("unconditional", "conditional_raw", "conditional_dilated"):
        r = o[k]
        print(f"{r['label']:<46}{r['data_bytes']:>12.0f}{r['model_bytes']:>11.0f}{r['total_bytes']:>12.0f} B")
    print(f"\nfree predictor buys {o['bits_bought_by_free_predictor_bytes']:.0f} B "
          f"= {o['fraction_of_description_bought'] * 100:.2f} % of the description")


if __name__ == "__main__":
    main()
