#!/usr/bin/env python
"""Reproduce every number in `.omx/research/ddm_tl1_train_least_scored_dims_20260802.md`.

$0, no scorer forward or backward. Two inputs only:

* ``upstream/models/{segnet,posenet}.safetensors`` -- frozen weights, static algebra only.
* ``experiments/results/mlx_fleet_gt_cache/gt_n600.npz`` -- cached n600 GT
  (``margins`` = source-side top1-top2 SegNet logit gap, ``lstars`` = source argmax).

Axis: ``[macOS-CPU advisory]``. ``score_claim=false``, ``promotion_eligible=false``.
Nothing here is a score; the memo's verdict scope travels with every number.

Usage::

    .venv/bin/python tools/ddm_tl1_scored_dims_reductions.py            # all sections
    .venv/bin/python tools/ddm_tl1_scored_dims_reductions.py --json     # machine-readable
    .venv/bin/python tools/ddm_tl1_scored_dims_reductions.py --weights-only  # skip the 471 MB read
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SEGNET = REPO / "upstream" / "models" / "segnet.safetensors"
POSENET = REPO / "upstream" / "models" / "posenet.safetensors"
GT_N600 = REPO / "experiments" / "results" / "mlx_fleet_gt_cache" / "gt_n600.npz"

# Task #456 receipt medians, Torch 2.12.1, ms/pair (cheaper_exact_forward_transfer_95kill_20260713.md).
T_REF_MS = 893.005052
T_FAST_MS = 302.06995825

# Equal-count margin inversion targets. d_seg values and their provenance.
DSEG_TARGETS = (
    (0.0043116, "live base dc1_fold, DERIVED from gap_decomposition_against_floor_20260802"),
    (0.0038892, "burn ep399, MEASURED"),
    (0.0002966, "PR130 demonstrated floor"),
)

PAIR78_TIE_MARGIN = 2.384185791015625e-7
TAUS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_rank_algebra() -> dict:
    """Memo section 2: the rank-4 head gauge, and its destruction under 3x3 overlap."""
    from safetensors.torch import load_file

    sd = load_file(str(SEGNET))
    weight = sd["segmentation_head.0.weight"].numpy().astype(np.float64)  # (5, 16, 3, 3)
    centered = weight - weight.mean(axis=0, keepdims=True)

    per_site = centered.reshape(5, -1)  # (5, 144)
    sv_site = np.linalg.svd(per_site, compute_uv=False)
    u, s, vt = np.linalg.svd(per_site, full_matrices=False)
    recon4 = (u[:, :4] * s[:4]) @ vt[:4]

    # One 16-channel feature LOCATION is read by 9 output sites through 9 different taps.
    stacked = np.concatenate(
        [centered[:, :, dy, dx] for dy in range(3) for dx in range(3)], axis=0
    )  # (45, 16)
    sv_overlap = np.linalg.svd(stacked, compute_uv=False)
    rank_overlap = int((sv_overlap > sv_overlap[0] * 1e-9).sum())

    # Negative control (m50): the same probe fed a single tap must be able to return rank 4.
    sv_one = np.linalg.svd(centered[:, :, 1, 1], compute_uv=False)
    rank_one = int((sv_one > sv_one[0] * 1e-9).sum())

    se_modules = sorted({key.split(".se.")[0] for key in sd if ".se." in key})
    total = sum(int(v.numel()) for v in sd.values())
    tracked = sum(int(v.numel()) for k, v in sd.items() if "num_batches_tracked" in k)
    running = sum(
        int(v.numel()) for k, v in sd.items() if "running_mean" in k or "running_var" in k
    )
    return {
        "segnet_sha256": _sha256(SEGNET),
        "head_shape": list(weight.shape),
        "centered_singular_values": [float(x) for x in sv_site],
        "rank4_reconstruction_max_abs_err": float(np.abs(recon4 - per_site).max()),
        "sigma1_over_sigma4": float(sv_site[0] / sv_site[3]),
        "rank_per_site_of_144": int((sv_site > sv_site[0] * 1e-9).sum()),
        "rank_under_3x3_overlap_of_16": rank_overlap,
        "overlap_condition_number": float(sv_overlap[0] / sv_overlap[rank_overlap - 1]),
        "control_single_tap_rank_of_16": rank_one,
        "se_bearing_mbconv_blocks": len(se_modules),
        "segnet_params_total_incl_buffers": total,
        "segnet_params_learnable": total - tracked - running,
        "segnet_head_params": sum(
            int(v.numel()) for k, v in sd.items() if "segmentation_head" in k
        ),
    }


def posenet_algebra() -> dict:
    """Memo section 2.5: the six scored pose outputs cost nothing to isolate."""
    from safetensors.torch import load_file

    sd = load_file(str(POSENET))
    total = sum(int(v.numel()) for v in sd.values())
    groups: dict[str, int] = {}
    for key, value in sd.items():
        prefix = key.split(".")[0]
        group = prefix if prefix in {"vision", "summarizer", "hydra"} else "other"
        groups[group] = groups.get(group, 0) + int(value.numel())
    return {
        "posenet_sha256": _sha256(POSENET),
        "posenet_params_total": total,
        "param_share": {k: v / total for k, v in groups.items()},
        "final_layer_pose_shape": list(sd["hydra.final_layer.pose.weight"].shape),
        "scored_outputs_of_12": 6,
        "params_saved_by_dropping_unscored_half": 6 * 32,
    }


def margin_reductions() -> dict:
    """Memo sections 3 and 6: the n600 source-margin CDF, the density law, the tie guard."""
    with np.load(GT_N600) as cache:
        margins = cache["margins"]  # (600, 384, 512) float32
        labels = cache["lstars"][0]
    pairs, height, width = margins.shape
    total_px = margins.size
    per_pair = margins.reshape(pairs, -1)

    cdf = {
        f"{t:g}": float((margins < t).mean())
        for t in (PAIR78_TIE_MARGIN, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.25, 0.5, 1.0, 2.0)
    }

    # Local density rho(t) = dP/dt, finite-differenced. Flat rho => d_seg linear in reach.
    edges = [0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.125, 0.15, 0.175, 0.2, 0.5, 1.0]
    shares = [float((margins < t).mean()) for t in edges]
    density = {
        f"[{edges[i - 1]:g},{edges[i]:g}]": (shares[i] - shares[i - 1]) / (edges[i] - edges[i - 1])
        for i in range(1, len(edges))
    }
    flat_band = [v for k, v in density.items() if float(k.split(",")[1][:-1]) <= 0.2]

    ordered = np.sort(margins.reshape(-1))
    inversion = {}
    for d_seg, provenance in DSEG_TARGETS:
        k = round(d_seg * total_px)
        t_star = float(ordered[k - 1]) if k > 0 else 0.0
        inversion[f"{d_seg:.7f}"] = {
            "provenance": provenance,
            "t_star_lower_bound_on_reach": t_star,
            "px_below_t_star": int((margins < t_star).sum()),
            "share_below_t_star": float((margins < t_star).mean()),
        }

    guard = {}
    for tau in TAUS:
        flagged = float(((per_pair < tau).sum(axis=1) > 0).mean())
        guard[f"{tau:g}"] = {
            "px_below_tau": int((margins < tau).sum()),
            "frames_flagged": round(flagged * pairs),
            "frac_pairs_flagged": flagged,
            "net_exact_speedup": T_REF_MS / (T_FAST_MS + flagged * T_REF_MS),
        }

    # Control C2: low margin must sit on a class boundary in lstars.
    boundary = np.zeros_like(labels, dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    low = margins[0] < 0.1
    return {
        "n_pairs": pairs,
        "frame_px": height * width,
        "total_px": total_px,
        "margin_min": float(margins.min()),
        "margin_max": float(margins.max()),
        "cdf": cdf,
        "density_rho": density,
        "rho_flat_band_min": min(flat_band),
        "rho_flat_band_max": max(flat_band),
        "equal_count_inversion": inversion,
        "raw_forward_ratio": T_REF_MS / T_FAST_MS,
        "tie_guard": guard,
        "control_boundary_share_all": float(boundary.mean()),
        "control_boundary_share_given_low_margin": float(boundary[low].mean()),
        "control_boundary_share_given_high_margin": float(boundary[margins[0] > 4].mean()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--weights-only", action="store_true", help="skip the 471 MB cached-margin read"
    )
    args = parser.parse_args(argv)

    for path in (SEGNET, POSENET) if args.weights_only else (SEGNET, POSENET, GT_N600):
        if not path.exists():
            print(f"MISSING INPUT: {path}", file=sys.stderr)
            return 2

    out = {
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "memo": ".omx/research/ddm_tl1_train_least_scored_dims_20260802.md",
        "head_rank_algebra": head_rank_algebra(),
        "posenet_algebra": posenet_algebra(),
    }
    if not args.weights_only:
        out["margin_reductions"] = margin_reductions()

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    head = out["head_rank_algebra"]
    print("== section 2: the rank-4 head gauge does not propagate backward ==")
    print(f"  segnet sha256                  {head['segnet_sha256']}")
    print("  head                           Conv2d(16 -> 5, k=3); weight shape "
          f"{tuple(head['head_shape'])}")
    print(f"  centered singular values       {head['centered_singular_values']}")
    print(f"  rank-4 recon max abs err       {head['rank4_reconstruction_max_abs_err']:.6e}")
    print(f"  rank per site                  {head['rank_per_site_of_144']} of 144 patch dims")
    print(
        f"  rank under 3x3 overlap         {head['rank_under_3x3_overlap_of_16']} of 16"
        f"  (cond {head['overlap_condition_number']:.4f})"
    )
    print(
        f"  CONTROL single tap             {head['control_single_tap_rank_of_16']} of 16"
        "  (probe can return low rank)"
    )
    print(f"  SE-bearing MBConv blocks       {head['se_bearing_mbconv_blocks']}")
    print(
        f"  head params                    {head['segnet_head_params']} of "
        f"{head['segnet_params_total_incl_buffers']}"
    )

    pose = out["posenet_algebra"]
    print("\n== section 2.5: pose ==")
    print(f"  posenet params                 {pose['posenet_params_total']}")
    print("  scored outputs                 6 of 12")
    print(f"  params saved by dropping half  {pose['params_saved_by_dropping_unscored_half']}")

    if args.weights_only:
        return 0

    marg = out["margin_reductions"]
    print("\n== section 3: the n600 source-margin CDF and the density law ==")
    for t, share in marg["cdf"].items():
        print(f"  P(margin < {t:>10}) = {share:.8f}  ({share * marg['total_px']:.0f} px)")
    print(
        f"  rho on [0,0.2]                 {marg['rho_flat_band_min']:.6f}"
        f" .. {marg['rho_flat_band_max']:.6f}"
    )
    for d_seg, row in marg["equal_count_inversion"].items():
        print(
            f"  d_seg {d_seg} -> t* {row['t_star_lower_bound_on_reach']:.6f}"
            f"  ({row['share_below_t_star'] * 100:.6f}% of px)  [{row['provenance']}]"
        )
    print("\n== section 6: the tie-guarded exact forward ==")
    print(f"  raw ratio                      {marg['raw_forward_ratio']:.10f}")
    for tau, row in marg["tie_guard"].items():
        print(
            f"  tau={tau:<8} px<tau {row['px_below_tau']:>6}"
            f"  frames {row['frames_flagged']:>3}/600"
            f"  net exact {row['net_exact_speedup']:.4f}x"
        )
    print("\n== control C2: `margins` is the argmax gap ==")
    print(f"  P(boundary) overall            {marg['control_boundary_share_all']:.6f}")
    print(f"  P(boundary | margin<0.1)       {marg['control_boundary_share_given_low_margin']:.6f}")
    print(f"  P(boundary | margin>4)         {marg['control_boundary_share_given_high_margin']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
