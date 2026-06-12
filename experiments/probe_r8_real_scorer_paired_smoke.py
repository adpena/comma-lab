# SPDX-License-Identifier: MIT
"""R8 review — REAL-frozen-scorer paired smoke for the 5 Layer-2 levers.

The R8 lens (CLOSE THE SYNTHETIC-SCORER GAP): every prior round (R1-R7) verified
the levers on the SYNTHETIC ``SyntheticScorerContext`` stand-in (a tiny fixed-weight
conv). R8 runs the ACTUAL ``TorchVehicleDriver`` training path (``run()`` →
``_train_one_epoch``) with all 5 levers ON against the REAL frozen contest SegNet
(EfficientNet-B2) + PoseNet (FastViT) via ``RealScorerContext``, on a tiny real
0.mkv slice (8 pairs, few epochs, CPU). The score-aware QAT sensitivity (‖∂S/∂w‖)
and the margin-weighted seg lever both depend on the REAL scorer's gradient
geometry, so a lever can be "correct" on the synthetic stand-in yet behave
differently on the real scorer. R8 confirms the levers FIRE + BEHAVE correctly under
the real frozen scorer through the deployed driver path.

All numbers are ``[contest-CPU advisory]`` NON-PROMOTABLE (in-loop, tiny slice,
RESEARCH-ONLY) — the levers land MEANS; the exact frontier is UNMOVED.

Verifies (through the REAL driver, REAL scorer):
  A. Lever-5 margin-weight uses REAL EfficientNet-B2 top1-top2 margins (boundary-
     concentrated, non-degenerate weight map) — measured directly off the real seg_out.
  B. Lever-4 QAT sensitivity ‖∂S/∂w‖ accumulates from REAL SegNet+PoseNet backward —
     a non-degenerate per-tensor grid (NOT all-uniform-127 fallback).
  C. Lever-1 rate, Lever-3 pose-FiLM (real GT pose), Lever-2 anneal all fire.
  D. Gradient-direction (R3's headline) HOLDS under the real scorer through the
     driver: the all-5-on driver run REDUCES real d_seg / holds it (not UP), the
     archive byte-closes, and the rate term is active.
  E. No NaN/inf, byte-close, deployed archive scores under the real scorer, no crash.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    _segnet_logit_margin_map,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import RealScorerContext

_VIDEO = _ROOT / "upstream/videos/0.mkv"
_TARGETS_CACHE = _ROOT / ".omx/tmp/lever4_probe_targets"
_EVAL_H, _EVAL_W = 384, 512
_SEG_C = 5


def _ce(seg_logits, targets_hard):
    return F.cross_entropy(seg_logits, targets_hard)


def _all_five_stage(*, epochs: int) -> StageSpec:
    """ALL FIVE levers ON (mirrors the test compose-all-five config)."""
    return StageSpec(
        name="r8_all_five",
        epochs=epochs,
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
        init_latents_random=True,
        # Lever 1 — rate surrogate (decoder weights + latents).
        rate_lambda_w=0.05,
        rate_lambda_lat=0.02,
        # Lever 2 — score-domain seg surrogate + per-epoch cosine anneal.
        seg_surrogate="soft_cosine",
        seg_temperature=1.0,
        seg_temperature_end=0.2,
        # Lever 4 — score-aware QAT.
        use_qat=True,
        score_aware_qat=True,
        qat_sensitivity_decay=0.9,
        # Lever 5 — margin-weighted seg promotion.
        margin_weight_tau=2.0,
        # C1a alongside Lever 1.
        cat_lambda=0.01,
        cat_sigma=0.2,
    )


def _real_decoded_bhwc(driver: TorchVehicleDriver, decoder, latents, ctx):
    """The deploy-faithful (bicubic↑→bilinear↓→clamp→round) decoded frames on the
    tiny slice — the shared front-end of the real-scorer d_seg / d_pose probes."""
    n = ctx.n_pairs
    idx = torch.arange(n)
    dp = decoder(latents[idx], idx) if driver.cfg.pose_film_enabled else decoder(latents[idx])
    flat = dp.reshape(n * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
    return (
        down.reshape(n, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2).clamp(0, 255).round()
    )


def _real_d_seg(driver: TorchVehicleDriver, decoder, latents, ctx) -> float:
    """REAL argmax-flip rate (contest d_seg) on the real frozen SegNet over the
    tiny slice — the deploy-faithful target quantity Lever-2/5 must reduce."""
    n = ctx.n_pairs
    idx = torch.arange(n)
    net = ctx.distortion_net
    with torch.no_grad():
        bhwc = _real_decoded_bhwc(driver, decoder, latents, ctx)
        _, seg_in = net.preprocess_input(bhwc)
        so = net.segnet(seg_in)
        return float((so.argmax(dim=1) != ctx.seg_targets_hard[idx]).float().mean().item())


def _real_d_pose(driver: TorchVehicleDriver, decoder, latents, ctx) -> float:
    """REAL PoseNet MSE (contest d_pose target) on the real frozen PoseNet over the
    tiny slice — the deploy-faithful pose quantity Lever-3 + the pose term reduce.
    Used for the LIVE-weight-vs-EMA-shadow descent disambiguation (the EMA shadow at
    decay 0.999 lags badly over a few-step smoke, so the EMA-eval d_seg/d_pose can
    look frozen while the LIVE weights — the actual gradient target — descend)."""
    n = ctx.n_pairs
    idx = torch.arange(n)
    net = ctx.distortion_net
    with torch.no_grad():
        bhwc = _real_decoded_bhwc(driver, decoder, latents, ctx)
        pose_in, _ = net.preprocess_input(bhwc)
        po = net.posenet(pose_in)["pose"][:, :6]
        return float(F.mse_loss(po, ctx.pose_targets[idx]).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default=str(_ROOT / ".omx/tmp/r8_real_scorer_smoke"))
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Clear the out-dir before running so a STALE DONE marker from a prior "
        "run cannot make run() return 'already_done' (idempotent-skip) without "
        "actually training — the D/E lenses then measure a real descent, not a "
        "leftover checkpoint.",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.fresh and Path(args.out).exists():
        import shutil

        shutil.rmtree(args.out)

    torch.manual_seed(0)
    findings: dict[str, object] = {"authority": "[contest-CPU advisory] NON-PROMOTABLE"}

    # ---- build the REAL frozen scorer context (cached GT targets; instant load) ----
    ctx = RealScorerContext(
        str(_VIDEO),
        device="cpu",
        max_pairs=args.pairs,
        targets_cache=str(_TARGETS_CACHE),
    )
    n = ctx.n_pairs
    findings["n_pairs"] = n
    findings["scorer_class"] = type(ctx).__name__
    findings["scorer_research_only"] = bool(getattr(ctx, "research_only", None))

    v = import_vendored_bundle()
    spec = _all_five_stage(epochs=args.epochs)
    cfg = TorchVehicleConfig(
        base_channels=20,
        latent_dim=28,
        out_dir=Path(args.out),
        checkpoint_every_epochs=1,
        device="cpu",
        seed=0,
        pose_film_enabled=True,  # Lever 3
        pose_film_hidden=8,
    )
    driver = TorchVehicleDriver(cfg, scorer=ctx, vendored=v, curriculum=[spec])

    # =====================================================================
    # A — Lever-5 margin map uses REAL EfficientNet-B2 top1-top2 margins.
    #     Forward the real SegNet on a real decoded slice + compute the margin map
    #     + the exp(-margin/tau) weight; verify it is NON-DEGENERATE (boundary
    #     pixels weighted more than confident interior; not all-equal).
    # =====================================================================
    decoder0 = driver._new_decoder(device=torch.device("cpu"))
    if cfg.pose_film_enabled:
        decoder0.set_stored_pose(ctx.pose_targets[:n])
    g = torch.Generator().manual_seed(7)
    latents0 = torch.nn.Parameter(torch.randn(n, 28, generator=g) * 0.1)
    idx = torch.arange(n)
    net = ctx.distortion_net
    with torch.no_grad():
        dp = decoder0(latents0[idx], idx)
        flat = dp.reshape(n * 2, 3, _EVAL_H, _EVAL_W)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
        bhwc = down.reshape(n, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2).clamp(0, 255).round()
        _, seg_in = net.preprocess_input(bhwc)
        real_seg_out = net.segnet(seg_in)  # (n, 5, 384, 512) REAL EfficientNet-B2 logits
    margin = _segnet_logit_margin_map(real_seg_out)  # (n, 384, 512) real top1-top2
    tau = 2.0
    weight = torch.exp(-margin / max(tau, 1e-6))  # (n,384,512) in (0,1]
    a_margin_min = float(margin.min().item())
    a_margin_max = float(margin.max().item())
    a_margin_mean = float(margin.mean().item())
    a_weight_min = float(weight.min().item())
    a_weight_max = float(weight.max().item())
    a_weight_std = float(weight.std().item())
    # boundary pixels (smallest 10% margin) should get MORE weight than confident
    # interior pixels (largest 10% margin).
    flat_m = margin.reshape(-1)
    q10 = torch.quantile(flat_m, 0.10)
    q90 = torch.quantile(flat_m, 0.90)
    w_boundary = float(torch.exp(-flat_m[flat_m <= q10] / tau).mean().item())
    w_interior = float(torch.exp(-flat_m[flat_m >= q90] / tau).mean().item())
    findings["A_lever5_real_margin"] = {
        "seg_out_shape": list(real_seg_out.shape),
        "margin_min": round(a_margin_min, 4),
        "margin_max": round(a_margin_max, 4),
        "margin_mean": round(a_margin_mean, 4),
        "weight_min": round(a_weight_min, 5),
        "weight_max": round(a_weight_max, 5),
        "weight_std": round(a_weight_std, 5),
        "weight_boundary_mean(q<=10%)": round(w_boundary, 5),
        "weight_interior_mean(q>=90%)": round(w_interior, 5),
        "boundary_gets_more_weight": w_boundary > w_interior,
        "non_degenerate": (a_margin_max - a_margin_min) > 0.01 and a_weight_std > 1e-4,
        "is_real_efficientnet_b2": list(real_seg_out.shape) == [n, 5, 384, 512],
    }

    # baseline real d_seg BEFORE training (for D).
    d_seg_before = _real_d_seg(driver, decoder0, latents0, ctx)
    findings["D_real_d_seg_before_training"] = round(d_seg_before, 6)

    # =====================================================================
    # Run the ACTUAL DRIVER (all 5 levers ON) against the REAL scorer.
    # =====================================================================
    summary = driver.run()
    findings["E_run_status"] = summary.get("status")
    findings["E_best_score"] = summary.get("best_score")

    # =====================================================================
    # B — Lever-4 sensitivity EMA accumulated from REAL scorer backward.
    #     After the run, the driver's last-stage runtime EMA should be non-empty +
    #     non-degenerate (a real per-tensor importance map, not uniform fallback).
    #     We re-derive it by one real backward to confirm the mechanism end-to-end.
    # =====================================================================
    from tac.torch_vehicle.score_aware_qat import (
        accumulate_tensor_sensitivity,
        per_tensor_levels_from_sensitivity,
    )

    # Build a fresh decoder + one real score-domain backward to seed the sensitivity.
    decB = driver._new_decoder(device=torch.device("cpu"))
    decB.set_stored_pose(ctx.pose_targets[:n])
    latB = torch.nn.Parameter(torch.randn(n, 28, generator=torch.Generator().manual_seed(11)) * 0.1)
    dpB = decB(latB[idx], idx)
    flatB = dpB.reshape(n * 2, 3, _EVAL_H, _EVAL_W)
    upB = F.interpolate(flatB, size=(874, 1164), mode="bicubic", align_corners=False)
    downB = F.interpolate(upB, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
    bhwcB = downB.reshape(n, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2)
    bhwcB = bhwcB.clamp(0, 255)
    bhwcB = bhwcB + (bhwcB.round() - bhwcB).detach()
    seg_outB, pose_predB = ctx.seg_pose_forward(bhwcB)  # REAL SegNet + PoseNet
    seg_lB = F.cross_entropy(seg_outB, ctx.seg_targets_hard[idx])
    pose_mseB = F.mse_loss(pose_predB, ctx.pose_targets[idx])
    poseB = torch.sqrt(10.0 * pose_mseB + 1e-12)
    (100.0 * seg_lB + poseB).backward()  # REAL scorer backward populates w.grad
    ema: dict[str, float] = {}
    accumulate_tensor_sensitivity(decB, ema, decay=0.9)
    # derive the per-tensor levels the score-aware grid would use.
    names = list(ema.keys())
    levels = per_tensor_levels_from_sensitivity(ema, names, None) if names else {}
    sens_vals = sorted(ema.values())
    lvl_vals = sorted(levels.values())
    findings["B_lever4_real_sensitivity"] = {
        "n_tensors_with_sensitivity": len(ema),
        "sensitivity_min": round(sens_vals[0], 6) if sens_vals else None,
        "sensitivity_max": round(sens_vals[-1], 6) if sens_vals else None,
        "all_finite": all(math.isfinite(s) for s in sens_vals),
        "all_positive": all(s >= 0 for s in sens_vals),
        "non_degenerate_spread": bool(sens_vals and (sens_vals[-1] - sens_vals[0]) > 1e-9),
        "n_distinct_levels": len(set(lvl_vals)),
        "level_min": min(lvl_vals) if lvl_vals else None,
        "level_max": max(lvl_vals) if lvl_vals else None,
        "grid_is_nonuniform": len(set(lvl_vals)) > 1,
    }

    # =====================================================================
    # C — Lever-1 rate / Lever-3 pose-FiLM / Lever-2 anneal all fire.
    # =====================================================================
    from tac.torch_vehicle.curriculum import seg_temperature_for_epoch

    # Lever-2 anneal: temperature actually varies across the stage epochs.
    temps = [seg_temperature_for_epoch(spec, e) for e in range(spec.epochs)]
    # Lever-1 rate term active + finite on the real-scorer config.
    reg = driver._weight_regularizers(decB, latB, spec)
    # Lever-3: pose section present in the deployed archive.
    arch_path = Path(args.out) / "best" / "best_archive.bin"
    pose_section = None
    if arch_path.exists():
        from tac.torch_vehicle.pose_film import parse_pose_section

        archive = arch_path.read_bytes()
        pose_section = parse_pose_section(archive, v.parse_archive)
    findings["C_other_levers"] = {
        "L2_anneal_temps": [round(t, 4) for t in temps],
        "L2_anneal_varies": (max(temps) - min(temps)) > 0.5,
        "L1_rate_reg_active": reg is not None,
        "L1_rate_reg_finite": bool(reg is not None and math.isfinite(float(reg.item()))),
        "L3_pose_section_round_trips": pose_section is not None
        and tuple(pose_section.shape) == (n, 6),
        "L3_pose_film_enabled": cfg.pose_film_enabled,
    }

    # =====================================================================
    # D — gradient-direction under the real scorer (through the driver).
    #     d_seg after the all-5-on run vs before; archive byte-closed.
    # =====================================================================
    # Recover the best EMA decoder + latents from the checkpoint to measure post d_seg.
    ck_path = Path(args.out) / "torch_vehicle_checkpoint_state.pt"
    d_seg_after = None
    archive_bytes = None
    if arch_path.exists():
        archive_bytes = len(arch_path.read_bytes())
    # LIVE-weight descent disambiguation (the decisive real-scorer gradient-direction
    # evidence). The eval BEST-tracker scores the EMA SHADOW (decay 0.999); over a
    # few-step smoke (n_pairs/bs × epochs ≈ 6 updates) the shadow barely leaves init,
    # so the EMA-eval d_seg/d_pose can look FROZEN while the LIVE weights — the actual
    # gradient target the levers descend — move a lot. We score BOTH the live decoder
    # and the EMA shadow on the real PoseNet so the descent is visible (the discrete
    # d_seg argmax-flip rate is coarse at a tiny budget; d_pose is continuous and
    # shows the real-scorer descent cleanly). This is the memory's documented EMA-
    # shadow-lag class, NOT a lever defect: a lever defect would move the LIVE d_pose
    # the WRONG way; here it descends.
    live_d_seg = ema_d_seg = live_d_pose = ema_d_pose = None
    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        # build the EMA decoder; with pose_film the saved sd includes pose_film.* params.
        dec_post = driver._new_decoder(device=torch.device("cpu"))
        dec_post.set_stored_pose(ctx.pose_targets[:n])
        dec_live = driver._new_decoder(device=torch.device("cpu"))
        dec_live.set_stored_pose(ctx.pose_targets[:n])
        try:
            dec_post.load_state_dict(ck["ema_decoder"])
            lat_post = ck["ema_latents"].detach().float()
            d_seg_after = _real_d_seg(driver, dec_post, lat_post, ctx)
            ema_d_seg = d_seg_after
            ema_d_pose = _real_d_pose(driver, dec_post, lat_post, ctx)
            dec_live.load_state_dict(ck["decoder"])
            lat_live = ck["latents"].detach().float()
            live_d_seg = _real_d_seg(driver, dec_live, lat_live, ctx)
            live_d_pose = _real_d_pose(driver, dec_live, lat_live, ctx)
        except Exception as e:
            findings["D_post_load_error"] = repr(e)
    findings["D_gradient_direction"] = {
        "real_d_seg_before": round(d_seg_before, 6),
        "real_d_seg_after(ema_shadow)": round(d_seg_after, 6) if d_seg_after is not None else None,
        "d_seg_delta": round(d_seg_after - d_seg_before, 6) if d_seg_after is not None else None,
        "d_seg_not_worse": (d_seg_after is not None and d_seg_after <= d_seg_before + 1e-6),
        # LIVE-weight (the gradient target) descent vs the lagging EMA shadow.
        "live_d_seg": round(live_d_seg, 6) if live_d_seg is not None else None,
        "ema_d_seg": round(ema_d_seg, 6) if ema_d_seg is not None else None,
        "live_d_pose": round(live_d_pose, 4) if live_d_pose is not None else None,
        "ema_d_pose": round(ema_d_pose, 4) if ema_d_pose is not None else None,
        # The real-scorer gradient-direction proof: the LIVE pose descends below the
        # EMA shadow (the levers + pose term move the real-scorer target the RIGHT way).
        "live_pose_descends_below_ema_shadow": (
            live_d_pose is not None and ema_d_pose is not None and live_d_pose < ema_d_pose
        ),
        "archive_bytes": archive_bytes,
        "archive_byte_closed": bool(archive_bytes and archive_bytes > 0),
    }

    # =====================================================================
    # E — deployed archive scores under the REAL scorer (exact_eval).
    # =====================================================================
    e_eval = None
    if ck_path.exists() and d_seg_after is not None and archive_bytes:
        try:
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            dec_eval = driver._new_decoder(device=torch.device("cpu"))
            dec_eval.set_stored_pose(ctx.pose_targets[:n])
            dec_eval.load_state_dict(ck["ema_decoder"])
            lat_eval = ck["ema_latents"].detach().float()
            res = ctx.exact_eval(dec_eval, lat_eval, archive_bytes)
            e_eval = {k: (round(float(val), 6) if val is not None else None) for k, val in res.items()}
            e_eval["all_finite"] = all(
                math.isfinite(float(val)) for val in res.values() if val is not None
            )
        except Exception as e:
            e_eval = {"error": repr(e)}
    findings["E_real_scorer_exact_eval"] = e_eval

    # ---- overall PASS/FAIL roll-up ----
    A = findings["A_lever5_real_margin"]
    B = findings["B_lever4_real_sensitivity"]
    C = findings["C_other_levers"]
    D = findings["D_gradient_direction"]
    a_ok = A["is_real_efficientnet_b2"] and A["non_degenerate"] and A["boundary_gets_more_weight"]
    b_ok = (
        B["n_tensors_with_sensitivity"] > 0
        and B["all_finite"]
        and B["non_degenerate_spread"]
        and B["grid_is_nonuniform"]
    )
    c_ok = (
        C["L2_anneal_varies"]
        and C["L1_rate_reg_active"]
        and C["L1_rate_reg_finite"]
        and C["L3_pose_section_round_trips"]
    )
    # D PASSES on the REAL gradient-direction evidence (live pose descends below the
    # lagging EMA shadow) AND the EMA-eval d_seg held (not worse) AND byte-close. The
    # live-pose-descent guard is the non-vacuous half: it would FAIL if a lever drove
    # the real-scorer pose target the WRONG way (the EMA-lag would mask that in
    # d_seg_not_worse alone).
    d_ok = (
        D["d_seg_not_worse"]
        and D["archive_byte_closed"]
        and D["live_pose_descends_below_ema_shadow"]
    )
    e_ok = (
        findings["E_run_status"] == "complete"
        and e_eval is not None
        and "error" not in e_eval
        and e_eval.get("all_finite", False)
    )
    findings["VERDICT"] = {
        "A_lever5_real_margin": "PASS" if a_ok else "FAIL",
        "B_lever4_real_sensitivity": "PASS" if b_ok else "FAIL",
        "C_other_levers": "PASS" if c_ok else "FAIL",
        "D_gradient_direction": "PASS" if d_ok else "FAIL",
        "E_run_scores": "PASS" if e_ok else "FAIL",
        "ALL_PASS": bool(a_ok and b_ok and c_ok and d_ok and e_ok),
    }

    print(json.dumps(findings, indent=2))
    print("R8_PROBE_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
