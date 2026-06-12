#!/usr/bin/env python3
"""R12 combined Lever-5×Lever-2 seg-gradient-floor + live-arm config-waste probe.

The R12 adversarial lens (distinct from R10): R10 measured the SURROGATE'S gradient
floor (the soft-cosine `1 - softmax(pred/T)[gt]` per-pixel gradient ∝ p·(1−p) dying
at cold T). But the live arm runs Lever-5 ON (`margin_weight_tau=2.0`), which
MULTIPLIES the per-pixel surrogate by a DETACHED weight `exp(-margin/τ) ∈ (0,1]`.
The R10 floor guard checks the surrogate alone. R12 asks two new questions the prior
rounds did NOT measure:

  Q1 (the "multiply-a-dead-gradient" case): does Lever-5's up-weighting RESCUE the
     cold-dead surrogate gradient? It CANNOT — the weight is detached and ≤ 1, so it
     can only SCALE DOWN the already-dead per-pixel gradient, never revive it. WORSE:
     `exp(-margin/τ)` is SMALLEST exactly on confidently-WRONG flip pixels (large
     margin), so on the precise pixels the cold surrogate already can't fix, Lever-5
     down-weights them further. So the COMBINED (margin × surrogate) gradient floor is
     AT LEAST as low as the surrogate's — the floor is NOT a surrogate-only artifact.
     This probe MEASURES the combined gradient floor on the REAL scorer and confirms
     `seg_anneal_temperature_is_gradient_alive` correctly bounds the COMBINED gradient.

  Q2 (the live-arm config-waste quantification the operator asked for): the live arm
     uses `seg_temperature_end=0.05`, BELOW R10's gradient floor (0.1). Over the
     per-stage cosine anneal the arm ACTUALLY uses (1.0 → 0.05 per stage), what
     FRACTION of each stage runs with the COMBINED seg gradient below the floor (dead)
     vs end=0.1? And on a real mid-basin decoder, how much COMBINED seg-gradient
     MAGNITUDE is lost on the flip pixels at end=0.05 vs end=0.1? This informs the
     operator's pending decision (change end→0.1 vs keep 0.05 + CE-blend). The probe
     does NOT change the arm config — it only QUANTIFIES the waste.

Authority: real frozen scorer (EfficientNet-B2 SegNet + FastViT PoseNet via
RealScorerContext → load_frozen_distortion_net), CPU-TRUSTED. RESEARCH-ONLY tiny
slice → [contest-CPU advisory] NON-PROMOTABLE. Gradient-MAGNITUDE claim only (not a
score claim). NO daemon touched (writes only .omx/tmp/r12_*; reads cached targets).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import (
    SEG_ANNEAL_GRADIENT_FLOOR_T,
    StageSpec,
    seg_anneal_temperature_is_gradient_alive,
    seg_temperature_for_epoch,
)
from tac.torch_vehicle.driver import (
    _EVAL_H,
    _EVAL_W,
    TorchVehicleConfig,
    TorchVehicleDriver,
    _seg_loss_for_spec,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import RealScorerContext

_VIDEO = "upstream/videos/0.mkv"


def _ce(s, t):
    return F.cross_entropy(s, t)


def _stage(**ov) -> StageSpec:
    base = dict(
        name="r12",
        epochs=2,
        seg_loss_fn=_ce,
        eval_every=1,
        batch_size=4,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=1e-3,
        muon_lr=2e-4,
        muon_weight_decay=0.0,
        latent_lr_mult=10.0,
        grad_clip=1e9,
        grad_clip_muon=1e9,
        lr_floor_ratio=5e-6,
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.0,
        cat_sigma=0.2,
        use_qat=False,
        init_latents_random=True,
    )
    base.update(ov)
    return StageSpec(**base)


def _render_roundtrip(decoder, latents, idx):
    """The driver's exact render + uint8-roundtrip pipeline (FiLM decoder)."""
    decoded_pair = decoder(latents[idx], idx)
    B = len(idx)
    flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    dc = decoded_bhwc.clamp(0, 255)
    dr = dc.round()
    return dc + (dr - dc).detach()


def _combined_seg_grad_norm(spec, seg_out, seg_targets_hard, temperature, params, render_fn):
    """The LATENT/PARAM gradient norm of the COMBINED (Lever-5 margin × Lever-2
    surrogate) seg loss at a given temperature — measured by autograd through the
    real-scorer graph. Returns (grad_norm, loss_value)."""
    frames = render_fn()
    so, _ = seg_out_fwd(frames)
    seg_l = _seg_loss_for_spec(spec, so, seg_targets_hard, temperature=temperature)
    gs = torch.autograd.grad(spec.seg_weight * seg_l, params, allow_unused=True)
    flat = torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for p, g in zip(params, gs, strict=False)
    ])
    return float(flat.norm()), float(seg_l.detach())


# bound at runtime in main() (closure over the real scorer)
seg_out_fwd = None  # type: ignore[assignment]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--out", default=".omx/tmp/r12_combined_floor")
    ap.add_argument("--margin-tau", type=float, default=2.0)  # live arm value
    ap.add_argument("--stage-epochs", type=int, default=100,
                    help="epochs per stage to model the per-stage cosine anneal")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    sc = RealScorerContext(
        _VIDEO, device="cpu", max_pairs=args.n_pairs,
        targets_cache=str(out / "targets_cache"),
    )
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=str(out / "run"),
        device="cpu", pose_film_enabled=True, seed=0,
    )
    spec_margin = _stage(seg_surrogate="soft_cosine", margin_weight_tau=args.margin_tau)
    spec_plain = _stage(seg_surrogate="soft_cosine", margin_weight_tau=None)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[spec_margin])

    decoder = driver._new_decoder(device=torch.device("cpu"))
    decoder.set_stored_pose(sc.pose_targets[: args.n_pairs].cpu())
    n = int(args.n_pairs)
    latents = torch.nn.Parameter(torch.randn(n, 28) * 0.1)
    idx = torch.arange(min(4, n))
    params = [p for p in decoder.parameters() if p.requires_grad] + [latents]

    global seg_out_fwd

    def seg_out_fwd_impl(frames):
        return sc.seg_pose_forward(frames)

    seg_out_fwd = seg_out_fwd_impl

    def render_fn():
        return _render_roundtrip(decoder, latents, idx)

    tgt = sc.seg_targets_hard[idx]

    # ----------------------------------------------------------------------
    # Q1 — the COMBINED (margin × surrogate) gradient floor on the real scorer.
    # MEASURE the param/latent gradient norm at warm (T=1.0), floor (T=0.1),
    # and the live-arm cold tail (T=0.05), for BOTH margin-ON (Lever-5) and
    # margin-OFF (plain Lever-2). The "multiply-a-dead-gradient" claim: the
    # margin-ON combined gradient is NOT larger than the plain one at cold T —
    # Lever-5 cannot rescue the dead surrogate (the weight is ≤ 1 and detached).
    # ----------------------------------------------------------------------
    temps = {"warm_1.0": 1.0, "floor_0.1": SEG_ANNEAL_GRADIENT_FLOOR_T, "cold_0.05": 0.05}
    q1 = {}
    for label, t in temps.items():
        gm, lm = _combined_seg_grad_norm(spec_margin, None, tgt, t, params, render_fn)
        gp, lp = _combined_seg_grad_norm(spec_plain, None, tgt, t, params, render_fn)
        q1[label] = {
            "temperature": t,
            "grad_norm_margin_on": gm,
            "grad_norm_margin_off": gp,
            "loss_margin_on": lm,
            "loss_margin_off": lp,
            # the guard's prediction at this temperature:
            "guard_says_alive": seg_anneal_temperature_is_gradient_alive(t),
        }

    # The headline ratios (combined gradient collapse, margin ON — the live arm):
    gn_warm = q1["warm_1.0"]["grad_norm_margin_on"]
    gn_floor = q1["floor_0.1"]["grad_norm_margin_on"]
    gn_cold = q1["cold_0.05"]["grad_norm_margin_on"]
    combined_collapse_ratio_cold = (gn_cold / gn_warm) if gn_warm > 0 else float("nan")
    combined_collapse_ratio_floor = (gn_floor / gn_warm) if gn_warm > 0 else float("nan")
    # The "multiply-a-dead-gradient" check: at cold T, margin ON is NOT bigger than OFF
    # (Lever-5 cannot rescue; weight ≤ 1 detached). Allow tiny FP slack.
    margin_does_not_rescue_cold = (
        q1["cold_0.05"]["grad_norm_margin_on"]
        <= q1["cold_0.05"]["grad_norm_margin_off"] + 1e-12
    )

    # ----------------------------------------------------------------------
    # Q2 — the live-arm config-waste quantification (end=0.05 vs end=0.1).
    # Over a per-stage cosine anneal (1.0 → end over `stage_epochs`), what
    # fraction of epochs are below the gradient floor (dead)?
    # ----------------------------------------------------------------------
    def dead_fraction(end: float, epochs: int) -> dict:
        spec = StageSpec(**{**spec_margin.__dict__, "epochs": epochs,
                            "seg_temperature": 1.0, "seg_temperature_end": end})
        temps_sched = [seg_temperature_for_epoch(spec, e) for e in range(epochs)]
        alive = sum(1 for t in temps_sched if seg_anneal_temperature_is_gradient_alive(t))
        dead = epochs - alive
        return {
            "end": end, "epochs": epochs,
            "alive_epochs": alive, "dead_epochs": dead,
            "dead_fraction_pct": 100.0 * dead / epochs,
            "min_temp_reached": min(temps_sched),
            "final_temp": temps_sched[-1],
        }

    q2_schedule = {
        "end_0.05_live": dead_fraction(0.05, args.stage_epochs),
        "end_0.10_proposed": dead_fraction(0.10, args.stage_epochs),
    }
    waste_pct_05 = q2_schedule["end_0.05_live"]["dead_fraction_pct"]
    waste_pct_10 = q2_schedule["end_0.10_proposed"]["dead_fraction_pct"]
    extra_dead_pct_from_05 = waste_pct_05 - waste_pct_10

    # Q2b — gradient MAGNITUDE lost on the flip pixels at the TAIL temperature.
    # Compare the combined seg grad norm at the end=0.05 floor temp (0.05) vs the
    # end=0.1 floor temp (0.1) on the SAME real decoder. The ratio is how many ×
    # gradient the arm gives up at its tail by choosing 0.05 over 0.1.
    grad_at_005 = q1["cold_0.05"]["grad_norm_margin_on"]
    grad_at_010 = q1["floor_0.1"]["grad_norm_margin_on"]
    tail_grad_loss_factor = (grad_at_010 / grad_at_005) if grad_at_005 > 0 else float("inf")

    verdict = {
        "scorer_class": type(sc).__name__,
        "is_real_scorer": type(sc).__name__ == "RealScorerContext",
        "n_pairs": n,
        "margin_tau_live": args.margin_tau,
        "SEG_ANNEAL_GRADIENT_FLOOR_T": SEG_ANNEAL_GRADIENT_FLOOR_T,
        # ---- Q1: combined gradient floor ----
        "q1_combined_grad_by_temp": q1,
        "q1_combined_collapse_ratio_cold_over_warm": combined_collapse_ratio_cold,
        "q1_combined_collapse_ratio_floor_over_warm": combined_collapse_ratio_floor,
        "q1_margin_does_not_rescue_cold": margin_does_not_rescue_cold,
        "q1_combined_floor_holds": (
            combined_collapse_ratio_cold < 1e-2  # cold combined grad << warm
        ),
        # ---- Q2: live-arm config-waste quantification ----
        "q2_schedule_dead_fraction": q2_schedule,
        "q2_waste_pct_end_0.05_live": waste_pct_05,
        "q2_waste_pct_end_0.10_proposed": waste_pct_10,
        "q2_extra_dead_pct_from_choosing_0.05": extra_dead_pct_from_05,
        "q2_tail_grad_loss_factor_010_over_005": tail_grad_loss_factor,
        # operator-facing one-liner:
        "operator_summary": (
            f"end=0.05 wastes the seg lever for {waste_pct_05:.1f}% of each stage "
            f"(vs {waste_pct_10:.1f}% at end=0.10, i.e. {extra_dead_pct_from_05:.1f} extra "
            f"dead percentage-points) and loses {tail_grad_loss_factor:.2e}x combined "
            f"seg-gradient on the flips at the tail vs end=0.10."
        ),
    }
    verdict["R12_CLEAN"] = bool(
        verdict["q1_margin_does_not_rescue_cold"]
        and verdict["q1_combined_floor_holds"]
    )
    (out / "r12_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
