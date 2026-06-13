#!/usr/bin/env python3
"""R13 all-5-levers-on END-TO-END multi-step descent probe (the 5 Layer-2 levers).

The R13 adversarial lens (distinct from all prior rounds): R10 measured the
single-STEP combined gradient SIGN; R5 measured determinism; R7/R11 measured
resume bit-identity; the synthetic `test_compose_all_five_levers_end_to_end`
byte-closes a SYNTHETIC-scorer run. NONE of them proved that with ALL FIVE levers
active, a REAL optimizer actually DESCENDS the COMBINED loss over MULTIPLE real
steps on the REAL frozen scorer — i.e. that the integrated multi-lever effect is a
working training signal, not just a per-lever sign that points downhill for one
step. A set of levers can each point downhill at step 0 yet fail to descend over a
trajectory (oscillation, the seg term going cold-dead and stalling, the rate/QAT
terms fighting the score terms cumulatively). R13 RUNS a real all-5-on Adam
trajectory on the real scorer and asserts the combined loss genuinely decreases.

What R13 measures (the falsifiable hypotheses):
  (1) DESCENT: over K Adam steps with all 5 levers ON at a gradient-ALIVE seg
      temperature (>= the R12-corrected floor 0.3), the COMBINED loss
      `w_seg·seg_l + w_pose·pose_l (+ rate)` at step K is < step 0 (the integrated
      multi-lever effect trains, not just points downhill once).
  (2) BYTE-CLOSE + PARSE-BACK: the trained decoder + stored pose (Lever-3) export
      to an archive that parses back (decoder state + latents + pose section).
  (3) FINITE REAL-SCORE COMPONENTS: the real scorer's exact d_seg / d_pose / rate
      on the parse-back archive are all FINITE and in-range (no NaN/Inf from any
      lever's contribution to the trained weights).

Authority: real frozen scorer (EfficientNet-B2 SegNet + FastViT PoseNet via
RealScorerContext → load_frozen_distortion_net), CPU-TRUSTED. RESEARCH-ONLY tiny
slice → [contest-CPU advisory] NON-PROMOTABLE. Descent-direction + finiteness claim
only (not a score claim). NO daemon touched (writes only .omx/tmp/r13_*).
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
from tac.torch_vehicle.pose_film import parse_pose_section
from tac.torch_vehicle.scorer_context import RealScorerContext

_VIDEO = "upstream/videos/0.mkv"


def _ce(s, t):
    return F.cross_entropy(s, t)


def _all5_stage(*, epochs: int, t_start: float, t_end: float) -> StageSpec:
    """A stage with ALL FIVE levers ON (the live-arm shape + Lever-1 rate + Lever-4
    QAT, the heaviest compose)."""
    return StageSpec(
        name="r13_all5",
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
        cat_lambda=0.01,
        cat_sigma=0.2,
        use_qat=True,
        init_latents_random=True,
        # Lever 2 (+ anneal) — seg surrogate ON.
        seg_surrogate="soft_cosine",
        seg_temperature=t_start,
        seg_temperature_end=t_end,
        # Lever 1 — rate surrogate ON.
        rate_lambda_w=0.05,
        rate_lambda_lat=0.02,
        # Lever 4 — score-aware QAT ON.
        score_aware_qat=True,
        qat_sensitivity_decay=0.9,
        # Lever 5 — margin weight ON (live value).
        margin_weight_tau=2.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=24)
    # The FAITHFUL daemon LR (the live distortion arm runs adamw_lr=1e-3 and
    # descends loss 0.68 -> 0.52 monotonically over epochs 427->676). A larger
    # probe LR (the original 5e-2) overshoots from step 1 and OSCILLATES — that is
    # an instrument artifact, NOT a lever defect (the R10 test-instrument lesson).
    # The descent claim must be measured at the config the levers actually train at.
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=".omx/tmp/r13_all5_e2e")
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
        device="cpu", pose_film_enabled=True, pose_film_hidden=8, seed=0,
    )
    # Keep seg gradient-ALIVE the whole trajectory (T anneals 1.0 -> the floor 0.3),
    # so the descent test exercises a USABLE seg gradient (per the R12 floor finding).
    spec = _all5_stage(epochs=args.steps, t_start=1.0, t_end=SEG_ANNEAL_GRADIENT_FLOOR_T)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[spec])

    decoder = driver._new_decoder(device=torch.device("cpu"))
    n = int(args.n_pairs)
    decoder.set_stored_pose(sc.pose_targets[:n].cpu())
    latents = torch.nn.Parameter(torch.randn(n, 28) * 0.1)
    idx = torch.arange(min(4, n))
    dec_params = [p for p in decoder.parameters() if p.requires_grad]
    opt = torch.optim.Adam(dec_params + [latents], lr=args.lr)

    def render():
        dp = decoder(latents[idx], idx)
        B = len(idx)
        flat = dp.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
        bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
        dc = bhwc.clamp(0, 255)
        return dc + (dc.round() - dc).detach()

    def combined_loss(epoch_in_stage: int):
        t = seg_temperature_for_epoch(spec, epoch_in_stage)
        seg_out, pose6 = sc.seg_pose_forward(render())
        seg_l = _seg_loss_for_spec(spec, seg_out, sc.seg_targets_hard[idx], temperature=t)
        pose_l = torch.sqrt(10.0 * F.mse_loss(pose6, sc.pose_targets[idx]) + 1e-12)
        loss = spec.seg_weight * seg_l + spec.pose_weight * pose_l
        reg = driver._weight_regularizers(decoder, latents, spec)
        if reg is not None:
            loss = loss + reg
        return loss, float(seg_l.detach()), float(pose_l.detach()), t

    # ---- (1) the multi-step DESCENT trajectory (all 5 levers ON) ----
    losses = []
    seg_terms = []
    pose_terms = []
    temps = []
    for step in range(args.steps):
        opt.zero_grad()
        loss, segv, posev, t = combined_loss(step)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        seg_terms.append(segv)
        pose_terms.append(posev)
        temps.append(t)

    loss_start = losses[0]
    loss_end = losses[-1]
    loss_min = min(losses)
    # ROBUST descent metric (the daemon's descent is visible as a TREND, not a
    # per-step monotone decrease — SGD on a real scorer has per-step noise). Compare
    # the mean of the LAST quartile of steps against the FIRST step: a working
    # multi-lever training signal drives the windowed loss BELOW the start (the live
    # daemon's loss fell 0.68 -> 0.52 over its run). A single-step loss_end < loss_start
    # would be noise-fragile; the windowed mean is the honest descent measure.
    tail = losses[max(1, (3 * len(losses)) // 4):]
    tail_mean = sum(tail) / max(len(tail), 1)
    descended = tail_mean < loss_start
    all_finite = all(
        torch.isfinite(torch.tensor(x)).item()
        for x in losses + seg_terms + pose_terms
    )

    # ---- (2) BYTE-CLOSE + PARSE-BACK (Lever-3 pose section round-trips) ----
    # Exercise the driver's OWN export+eval path via a short all-5-on run() on the
    # REAL scorer (the canonical archive contract: train -> EMA -> build_archive_with
    # _pose -> best_archive.bin -> parse_archive + parse_pose_section). The descent
    # trajectory above is the novel R13 multi-step measurement; this is the proven
    # byte-close path run with the real scorer + a finite real-score-component check.
    archive_ok = False
    pose_roundtrip_ok = False
    dec_parse_ok = False
    parse_err = None
    real_score_finite = None
    real_score_components = None
    try:
        run_cfg = TorchVehicleConfig(
            base_channels=20, latent_dim=28, out_dir=str(out / "run_e2e"),
            checkpoint_every_epochs=1, device="cpu",
            pose_film_enabled=True, pose_film_hidden=8, seed=0,
        )
        drv2 = TorchVehicleDriver(run_cfg, scorer=sc, vendored=v, curriculum=[spec])
        summary = drv2.run()
        arch_path = Path(run_cfg.out_dir) / "best" / "best_archive.bin"
        if arch_path.exists():
            archive = arch_path.read_bytes()
            archive_ok = len(archive) > 0
            pose = parse_pose_section(archive, drv2.v.parse_archive)
            pose_roundtrip_ok = pose is not None and tuple(pose.shape) == (n, 6)
            dec_sd, lat, _meta = drv2.v.parse_archive(archive)
            dec_parse_ok = bool(dec_sd) and lat.shape[0] == n
        # (3) the BEST/LAST exact real-score components must be FINITE. The driver
        # summary exposes them under ``last_eval`` (a dict with d_seg/d_pose/rate/score)
        # + a scalar ``best_score`` — NOT ``best_eval``. Read the canonical keys.
        ev = summary.get("last_eval") or summary.get("best_eval") or summary.get("best") or {}
        if isinstance(ev, dict) and ev:
            comps = {k: float(ev[k]) for k in ("d_seg", "d_pose", "rate", "score")
                     if k in ev}
            if "best_score" in summary and "score" not in comps:
                comps["score"] = float(summary["best_score"])
            real_score_components = comps
            real_score_finite = all(
                torch.isfinite(torch.tensor(val)).item() for val in comps.values()
            ) if comps else None
        elif "best_score" in summary:
            comps = {"score": float(summary["best_score"])}
            real_score_components = comps
            real_score_finite = torch.isfinite(torch.tensor(comps["score"])).item()
        parse_err = f"run() status={summary.get('status')}"
    except Exception as e:  # noqa: BLE001
        parse_err = repr(e)

    verdict = {
        "scorer_class": type(sc).__name__,
        "is_real_scorer": type(sc).__name__ == "RealScorerContext",
        "n_pairs": n,
        "steps": args.steps,
        "lr": args.lr,
        "seg_floor_T": SEG_ANNEAL_GRADIENT_FLOOR_T,
        # ---- (1) descent ----
        "loss_trajectory": [round(x, 6) for x in losses],
        "seg_term_trajectory": [round(x, 6) for x in seg_terms],
        "pose_term_trajectory": [round(x, 6) for x in pose_terms],
        "temp_trajectory": [round(x, 4) for x in temps],
        "loss_start": loss_start,
        "loss_end": loss_end,
        "loss_min": loss_min,
        "loss_tail_mean": tail_mean,
        "combined_loss_descended": descended,
        "combined_loss_reduction": loss_start - tail_mean,
        "all_losses_finite": all_finite,
        # ---- (2) byte-close + parse-back ----
        "archive_byte_closed": archive_ok,
        "lever3_pose_roundtrip_ok": pose_roundtrip_ok,
        "decoder_section_parse_ok": dec_parse_ok,
        # ---- (3) finite real-score components ----
        "real_score_components": real_score_components,
        "real_score_components_finite": real_score_finite,
        "parse_note": parse_err,
    }
    verdict["R13_CLEAN"] = bool(
        descended and all_finite and archive_ok and pose_roundtrip_ok
        and dec_parse_ok and (real_score_finite is True)
    )
    (out / "r13_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["R13_CLEAN"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
