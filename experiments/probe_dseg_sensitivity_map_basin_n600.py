#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""d_seg-sensitivity map on the CONVERGED 600-pair small basis — the REAL-measured input the
byte-neutral taper waterfill solve consumes.

This is a clean variant of ``experiments/probe_dseg_sensitivity_map.py`` (the gate-2 probe) that
points ``_BASE`` at the converged n600 basin (``torch_vehicle_full_mps_basin_bc20_n600/best``)
instead of the hardcoded n96 control. The mechanics are identical: for each decoder weight tensor,
inject controlled 20%-relative-RMS Gaussian noise (fixed seed) into ONLY that tensor (rest FP32) and
measure the REAL Δd_seg via the authority ``RealScorerContext.exact_eval``. Δd_seg per tensor = its
d_seg-sensitivity; Δd_seg/param = sensitivity DENSITY (high density ⇒ under-provisioned stage ⇒
deserves more capacity in a byte-neutral reallocation = the taper waterfill).

WHY n600 not n96: the taper waterfill must be solved against the REAL operating point (600 pairs,
no overfit), and the sensitivity SHAPE — which stage carries d_seg — is what the waterfill needs.
``max_pairs=96`` here is only the eval subset for SPEED; it is the SHAPE of the map (per-stage
density) we consume, not an absolute Δd_seg claim.

NO-FAKE: the map is the EXACT scorer's measured Δd_seg, never assumed/heuristic. If the basin
decoder will not load or ``exact_eval`` errors, the probe raises (it does not fabricate a map).

AUTHORITY: ``[contest-CPU advisory] NON-PROMOTABLE``. CPU only (no MPS); does not disturb any live
run. The Δd_seg values are sensitivity MEASUREMENTS, not frontier/score claims.

Usage:
    .venv/bin/python experiments/probe_dseg_sensitivity_map_basin_n600.py [BASE_DIR] [MAX_PAIRS]

Default BASE_DIR = experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best
Default MAX_PAIRS = 96
Writes reports/dseg_sensitivity_map_n600.json (reports/ is tracked — durable signal).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_DEFAULT_BASE = Path("experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best")
_NOISE_REL = 0.20  # 20% relative-RMS perturbation per tensor (matches the gate-2 probe)
_BASE_CHANNELS = 20
_LATENT_DIM = 28
# resolution band of each weight tensor (the HNeRV upsample chain: 6×8→…→384×512;
# SegNet argmax @~192×256). Identical mapping to the gate-2 probe so the two maps are comparable.
_RES = {
    "stem.weight": ("6x8", "LOW"),
    "blocks.0.weight": ("12x16", "LOW"), "blocks.1.weight": ("24x32", "LOW"),
    "blocks.2.weight": ("48x64", "MID"), "blocks.3.weight": ("96x128", "MID"),
    "blocks.4.weight": ("192x256", "HIGH"), "blocks.5.weight": ("384x512", "HIGH"),
    "skips.2.weight": ("24x32", "MID"), "skips.3.weight": ("48x64", "MID"),
    "skips.4.weight": ("96x128", "MID"),
    "refine.0.weight": ("384x512", "HIGH"), "refine.1.weight": ("384x512", "HIGH"),
    "rgb_0.weight": ("384x512", "HIGH"), "rgb_1.weight": ("384x512", "HIGH"),
}

# Map each weight tensor → the taper STAGE index (0..6) it draws its width from, so the per-tensor
# sensitivity density can be aggregated to a per-STAGE density (the waterfill knob is per-stage
# channels[0..6]). The vendored decoder builds:
#   stem:   channels[0]               -> stage 0
#   blocks.i / skips.i (i=0..5):      Conv2d(channels[i], channels[i+1]*...) -> the OUTPUT width is
#                                     channels[i+1], so blocks.i scales with stage i+1's channels
#   refine.* / rgb_0 / rgb_1:         channels[6] (final) -> stage 6
_STAGE = {
    "stem.weight": 0,
    "blocks.0.weight": 1, "blocks.1.weight": 2, "blocks.2.weight": 3,
    "blocks.3.weight": 4, "blocks.4.weight": 5, "blocks.5.weight": 6,
    "skips.2.weight": 3, "skips.3.weight": 4, "skips.4.weight": 5,
    "refine.0.weight": 6, "refine.1.weight": 6, "rgb_0.weight": 6, "rgb_1.weight": 6,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    base = Path(argv[0]) if len(argv) >= 1 else _DEFAULT_BASE
    max_pairs = int(argv[1]) if len(argv) >= 2 else 96

    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    import_vendored_bundle()
    model_mod = import_vendored("model")
    video_path = import_vendored("data").get_default_video_path()

    dec_path = base / "best_ema_decoder.pt"
    lat_path = base / "best_ema_latents.pt"
    if not dec_path.exists() or not lat_path.exists():
        raise FileNotFoundError(
            f"basin checkpoint missing: {dec_path} / {lat_path}. "
            "Refusing to fabricate a sensitivity map (NO-FAKE)."
        )

    dec_sd = torch.load(dec_path, map_location="cpu", weights_only=False)
    dec_sd = dec_sd.get("state_dict", dec_sd) if isinstance(dec_sd, dict) and "state_dict" in dec_sd else dec_sd
    latents = torch.load(lat_path, map_location="cpu", weights_only=False)
    if isinstance(latents, dict):
        latents = latents.get("latents", next(iter(latents.values())))

    n_pairs = int(latents.shape[0]) if torch.is_tensor(latents) else 0
    print(f"[basin] {base}  latents={tuple(latents.shape)}  eval max_pairs={max_pairs}  "
          f"[contest-CPU advisory] NON-PROMOTABLE", flush=True)

    ctx = RealScorerContext(
        video_path, device="cpu", train_device="cpu", split_by_head=False,
        max_pairs=max_pairs, targets_cache=Path("experiments/results/capstone_gt_targets_cache"),
    )

    def build(perturb_key=None, gen=None):
        d = model_mod.HNeRVDecoder(latent_dim=_LATENT_DIM, base_channels=_BASE_CHANNELS).eval()
        sd = {}
        for k, v in dec_sd.items():
            if k == perturb_key and torch.is_tensor(v) and v.dim() >= 2:
                rms = v.pow(2).mean().sqrt()
                sd[k] = v + torch.randn(v.shape, generator=gen) * (_NOISE_REL * rms)
            else:
                sd[k] = v.clone() if torch.is_tensor(v) else v
        d.load_state_dict(sd)
        return d

    archive_bytes = 89136  # n600 basin best_meta.json (rate-proxy only; not used in the shape)
    # exact_eval evaluates latents.shape[0] pairs (the vendored evaluate_decoder streams the FULL
    # video for that many pairs); the converged basin has 600 latents, so evaluating all 600 on CPU
    # is ~6x slower per call. The sensitivity SHAPE (which stage carries d_seg) is what the
    # waterfill consumes, so we eval the FIRST max_pairs latents — still the REAL authority
    # exact_eval, just on a max_pairs-pair subset (NO-FAKE: the scorer is the contest SegNet, not a
    # proxy; the subset is the first contiguous pairs, matching the gate-2 n96 probe's coverage).
    eval_latents = latents[:max_pairs] if torch.is_tensor(latents) else latents
    base_d_seg = ctx.exact_eval(build(), eval_latents, archive_bytes)["seg_distortion"]
    print(f"[baseline] d_seg={base_d_seg:.6f}  (eval over {eval_latents.shape[0]} pairs)  "
          f"[contest-CPU advisory]", flush=True)

    rows = []
    for k in [kk for kk in dec_sd if kk.endswith(".weight")
              and torch.is_tensor(dec_sd[kk]) and dec_sd[kk].dim() >= 2]:
        n = int(dec_sd[k].numel())
        gen = torch.Generator().manual_seed(1234)  # SAME noise seed → comparable across tensors
        d_seg = ctx.exact_eval(build(k, gen), eval_latents, archive_bytes)["seg_distortion"]
        dd = d_seg - base_d_seg
        res, band = _RES.get(k, ("?", "?"))
        rows.append({"tensor": k, "res": res, "band": band, "stage": _STAGE.get(k, -1),
                     "params": n, "d_seg": d_seg, "delta": dd, "delta_per_kparam": 1e3 * dd / n})
        print(f"  {k:18} {res:>8} {band:4} st{_STAGE.get(k,-1)} p={n:>6}  d_seg={d_seg:.6f}  "
              f"Δ={dd:+.6f}  Δ/kparam={1e3*dd/n:+.5f}", flush=True)

    tot_delta = sum(max(r["delta"], 0.0) for r in rows) or 1e-9
    tot_param = sum(r["params"] for r in rows)
    print("\n=== d_seg-SENSITIVITY MAP (n600 basin, Φ3 test) ===")
    band_summary = {}
    for band in ("LOW", "MID", "HIGH"):
        br = [r for r in rows if r["band"] == band]
        d_share = sum(max(r["delta"], 0.0) for r in br) / tot_delta
        p_share = sum(r["params"] for r in br) / tot_param
        ratio = d_share / max(p_share, 1e-9)
        band_summary[band] = {"param_share": p_share, "sensitivity_share": d_share, "density_ratio": ratio}
        tag = ("← over-provisioned" if ratio < 0.7
               else ("← UNDER-provisioned (deserves capacity)" if ratio > 1.3 else ""))
        print(f"  {band:4}: param-share {p_share*100:4.1f}%  sensitivity-share {d_share*100:4.1f}%  "
              f"density-ratio {ratio:.2f}× {tag}")

    # Per-STAGE aggregation (the actual waterfill knob): sum positive Δd_seg by stage, and the
    # current param mass by stage, then the per-stage sensitivity DENSITY = Δ-share / param-share.
    stage_delta = dict.fromkeys(range(7), 0.0)
    stage_param = dict.fromkeys(range(7), 0)
    for r in rows:
        s = r["stage"]
        if 0 <= s <= 6:
            stage_delta[s] += max(r["delta"], 0.0)
            stage_param[s] += r["params"]
    print("\n=== PER-STAGE sensitivity DENSITY (waterfill input) ===")
    stage_summary = {}
    for s in range(7):
        d_share = stage_delta[s] / tot_delta
        p_share = stage_param[s] / max(tot_param, 1)
        dens = d_share / max(p_share, 1e-9)
        stage_summary[s] = {"delta_share": d_share, "param_share": p_share, "density": dens,
                            "delta_abs": stage_delta[s], "params": stage_param[s]}
        print(f"  stage{s}: Δ-share {d_share*100:5.1f}%  param-share {p_share*100:5.1f}%  "
              f"density {dens:6.2f}×")

    hi_d = band_summary["HIGH"]["sensitivity_share"]
    hi_p = band_summary["HIGH"]["param_share"]
    verdict = ("Φ3 CONFIRMED: HIGH-res band carries disproportionate d_seg sensitivity "
               f"(sens {hi_d*100:.0f}% on param {hi_p*100:.0f}%) → taper should widen it"
               if hi_d / max(hi_p, 1e-9) > 1.3 else
               "Φ3 NOT confirmed at this perturbation → redirect / re-probe")
    print(f"\n  VERDICT: {verdict}")
    top = sorted(rows, key=lambda r: r["delta_per_kparam"], reverse=True)[:3]
    print("  top-3 sensitivity-DENSITY tensors (capacity-hungry):", [r["tensor"] for r in top])

    Path("reports").mkdir(exist_ok=True)
    out = {
        "base_dir": str(base), "n_pairs": n_pairs, "eval_max_pairs": max_pairs,
        "base_channels": _BASE_CHANNELS, "latent_dim": _LATENT_DIM,
        "noise_rel": _NOISE_REL, "archive_bytes": archive_bytes,
        "baseline_d_seg": base_d_seg, "rows": rows,
        "band_summary": band_summary,
        "stage_summary": {str(k): v for k, v in stage_summary.items()},
        "verdict": verdict,
        "authority": "contest-CPU advisory NON-PROMOTABLE",
    }
    Path("reports/dseg_sensitivity_map_n600.json").write_text(json.dumps(out, indent=2))
    print("  wrote reports/dseg_sensitivity_map_n600.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
