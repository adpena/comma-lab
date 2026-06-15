#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""d_seg-sensitivity map (WS-C research gate 2) — WHERE does d_seg live in the decoder?

The macro audit (Φ3) hypothesised the HNeRV taper `[20,20,20,15,11,10,10]` mis-allocates: ~67% of
params sit at low resolution (stem/blocks.0-1 at 6×8–24×32) while only ~11% sit at the 192×256–384×512
band where SegNet's argmax-flips physically live (the stride-2 stem decides argmax at ~192×256). If true,
the taper should move capacity from the low-res stem → the high-res refine/rgb head.

This $0 CPU probe MEASURES it: for each decoder weight tensor, inject controlled relative-RMS Gaussian
noise (20%, fixed seed) into ONLY that tensor (rest FP32) and measure the REAL Δd_seg via the authority
exact_eval. Δd_seg per tensor = its d_seg-sensitivity; Δd_seg/param = sensitivity DENSITY (high density ⇒
under-provisioned ⇒ deserves more capacity). Grouped by resolution band → the taper-realloc decision +
(bridge to gate 1) which tensors must stay int8 under FP4-QAT.

VERDICT: if the HIGH-res band (192×256+384×512: blocks.4/5, refine, rgb — ~11% of params) carries a
DISPROPORTIONATE share of total Δd_seg (sensitivity-share ≫ param-share) ⇒ Φ3 CONFIRMED, design the taper
to widen it. Else ⇒ Φ3 refuted, redirect WS-C.

AUTHORITY: `[contest-CPU advisory] NON-PROMOTABLE`. CPU only (no MPS); does not disturb the live run.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

_BASE = Path("experiments/results/from0_ab_v2_n96/control/best")
_NOISE_REL = 0.20  # 20% relative-RMS perturbation per tensor
# resolution band of each weight tensor (the HNeRV upsample chain: 6×8→…→384×512; SegNet argmax @~192×256)
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


def main() -> int:
    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    import_vendored_bundle()
    model_mod = import_vendored("model")
    video_path = import_vendored("data").get_default_video_path()

    dec_sd = torch.load(_BASE / "best_ema_decoder.pt", map_location="cpu", weights_only=False)
    dec_sd = dec_sd.get("state_dict", dec_sd) if isinstance(dec_sd, dict) and "state_dict" in dec_sd else dec_sd
    latents = torch.load(_BASE / "best_ema_latents.pt", map_location="cpu", weights_only=False)
    if isinstance(latents, dict):
        latents = latents.get("latents", next(iter(latents.values())))

    ctx = RealScorerContext(
        video_path, device="cpu", train_device="cpu", split_by_head=False,
        max_pairs=96, targets_cache=Path("experiments/results/capstone_gt_targets_cache"),
    )

    def build(perturb_key=None, gen=None):
        d = model_mod.HNeRVDecoder(latent_dim=28, base_channels=20).eval()
        sd = {}
        for k, v in dec_sd.items():
            if k == perturb_key and torch.is_tensor(v) and v.dim() >= 2:
                rms = v.pow(2).mean().sqrt()
                sd[k] = v + torch.randn(v.shape, generator=gen) * (_NOISE_REL * rms)
            else:
                sd[k] = v.clone()
        d.load_state_dict(sd)
        return d

    base = ctx.exact_eval(build(), latents, 76592)["seg_distortion"]
    print(f"[baseline] d_seg={base:.6f}  [contest-CPU advisory]", flush=True)

    rows = []
    for k in [kk for kk in dec_sd if kk.endswith(".weight") and torch.is_tensor(dec_sd[kk]) and dec_sd[kk].dim() >= 2]:
        n = dec_sd[k].numel()
        gen = torch.Generator().manual_seed(1234)  # SAME noise seed → comparable across tensors
        d_seg = ctx.exact_eval(build(k, gen), latents, 76592)["seg_distortion"]
        dd = d_seg - base
        res, band = _RES.get(k, ("?", "?"))
        rows.append({"tensor": k, "res": res, "band": band, "params": n,
                     "d_seg": d_seg, "delta": dd, "delta_per_kparam": 1e3 * dd / n})
        print(f"  {k:18} {res:>8} {band:4} p={n:>6}  d_seg={d_seg:.6f}  Δ={dd:+.6f}  "
              f"Δ/kparam={1e3*dd/n:+.5f}", flush=True)

    tot_delta = sum(max(r["delta"], 0.0) for r in rows) or 1e-9
    tot_param = sum(r["params"] for r in rows)
    print("\n=== d_seg-SENSITIVITY MAP (Φ3 test) ===")
    for band in ("LOW", "MID", "HIGH"):
        br = [r for r in rows if r["band"] == band]
        d_share = sum(max(r["delta"], 0.0) for r in br) / tot_delta
        p_share = sum(r["params"] for r in br) / tot_param
        ratio = d_share / max(p_share, 1e-9)
        print(f"  {band:4}: param-share {p_share*100:4.1f}%  sensitivity-share {d_share*100:4.1f}%  "
              f"density-ratio {ratio:.2f}× {'← over-provisioned' if ratio<0.7 else ('← UNDER-provisioned (deserves capacity)' if ratio>1.3 else '')}")
    hi_d = sum(max(r["delta"],0.0) for r in rows if r["band"]=="HIGH")/tot_delta
    hi_p = sum(r["params"] for r in rows if r["band"]=="HIGH")/tot_param
    verdict = ("Φ3 CONFIRMED: HIGH-res band carries disproportionate d_seg sensitivity "
               f"(sens {hi_d*100:.0f}% on param {hi_p*100:.0f}%) → taper should widen it"
               if hi_d/max(hi_p,1e-9) > 1.3 else
               "Φ3 NOT confirmed at this perturbation → redirect WS-C / re-probe")
    print(f"\n  VERDICT: {verdict}")
    top = sorted(rows, key=lambda r: r["delta_per_kparam"], reverse=True)[:3]
    print("  top-3 sensitivity-DENSITY tensors (capacity-hungry):", [r["tensor"] for r in top])
    Path("reports").mkdir(exist_ok=True)
    Path("reports/dseg_sensitivity_map.json").write_text(json.dumps(
        {"baseline": base, "rows": rows, "verdict": verdict,
         "authority": "contest-CPU advisory NON-PROMOTABLE"}, indent=2))
    print("  wrote reports/dseg_sensitivity_map.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
