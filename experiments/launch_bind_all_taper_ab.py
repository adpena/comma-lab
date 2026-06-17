#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""The BIND-ALL integrated A/B launcher (P2 of the bind-all production build-out).

This is the actuator for the FINAL production run that aims at sub-0.15 — it BINDS
EVERY production ingredient into ONE coherent config (per CLAUDE.md "HNeRV /
leaderboard-implementation parity discipline": architecture + score-aware training +
archive grammar + inflate runtime + export + curriculum + QAT + rate-attack + L3),
then runs it as a two-arm A/B that isolates the ONE structural lever still under test:
the d_seg-aware TAPER vs the vendored HNeRV taper.

The two arms are IDENTICAL in every ingredient EXCEPT ``taper_channels``:

  * ``arm_a`` (control)   — the vendored HNeRV taper ``[20,20,20,15,11,10,10]``.
  * ``arm_b`` (treatment) — the SOLVED byte-neutral taper ``[22,16,15,14,15,14,10]``
                            (gate-2 d_seg-sensitivity reallocation; −0.55% bytes,
                            feeds the d_seg-critical mid-late HIGH band; ``7dd5b5188``).

Both arms go through the SAME ``ConfigurableTaperHNeRVDecoder`` carrier (arm_a passes
the vendored schedule EXPLICITLY, not ``None``), so the A/B isolates the channel
SCHEDULE — not the decoder class. The verdict is cross-arm: arm_b best contest score
(100·d_seg + √(10·d_pose) + 25·B/N) vs arm_a, at matched bytes.

The bound ingredients (the bind-all production config, all ON):
  1. ARCHITECTURE      — solved/vendored taper (the A/B axis) via the taper carrier.
  2. WARM-START        — KD-warm-start: distill the converged VENDORED-taper basin
                         (teacher) into the re-tapered student (latents load direct;
                         decoder is frame-MSE distilled) — the wall-clock resolution
                         that lets the re-taper inherit the basin without the ~weeks-on-
                         MPS 29k-epoch from-scratch curriculum. ``--kd-warm-start-dir``.
  3. SCORE-AWARE SEG   — the OOMPH sharp soft_cosine seg lever (T 0.3→0.05 SHARP +
                         tight margin τ=0.5 + margin RENORM + seg_weight 1.5×) on the
                         REFINEMENT stages (≈ exact argmax-flip d_seg).
  4. POSE DECOUPLE     — FiLM-v2 (rgb_1 FiLM-clean → ∂d_seg/∂(FiLM)=0) + trunk-stopgrad
                         (∂d_seg/∂(pose-objective)=0 EXACTLY) + rgb_0-pose-trainable
                         (pose trains the frame-0 head SegNet never reads → strictly
                         more pose capacity at ZERO d_seg cost). The COMPLETE decoupling.
  5. POSE CADENCE      — pose EVERY epoch (``--pose-grad-every-k 1``); per "score >
                         training time" the throttle is a time-proxy, so k=1 is the
                         production default (APGC is the optional safety net).
  6. ALWAYS-ON         — eval_roundtrip (1:1 contest ``common.py``, always-on in the
                         driver) + EMA + EMA-WARMUP (the shadow tracks the short
                         post-KD fine-tune instead of staying frozen at the warm init).
  7. QAT + Muon + C1a + σ — inherited from the faithful PR95 refinement stage (Muon
                         final-stage, score-aware QAT, C1a coder-aware reg, σ noise).
  8. RATE ATTACK       — Lever-4 variable-level export codec
                         (``--rate-attack``): the EXPORT decoder blob is built at a
                         variable per-tensor INT8 grid from the ONLINE score-sensitivity
                         EMA (the SAME rank-norm band score-aware QAT trained on) — the
                         contest-OPTIMAL byte-half of Lever-4, on the SAME shared spine.

CONVERGENCE, NOT A FIXED CAP (CLAUDE.md "score > training time"): the budget is the
full PR95 schedule (``--total-epoch-budget None`` = 29,650) by default, run to
convergence with pose every epoch — the score is the arbiter, not wall-clock.

AUTHORITY (CLAUDE.md): the exact d_seg/d_pose that pick BEST run on the CPU authority
(``--device cpu``; NO MPS as a scorer). ``--train-device mps --split-by-head`` is the
gradient-only speed path (SegNet grad on MPS, PoseNet grad on the CPU authority). Every
in-loop number is ``[contest-CPU advisory]`` NON-PROMOTABLE until a byte-closed archive
runs through ``upstream/evaluate.py``.

Run ONE arm per invocation (own out-dir; resumable per-epoch; DONE-marker on exit), as a
DETACHED nohup daemon (NOT tool-background — the harness SIGURG-kills bg bash at ~3 min).
Use ``--print-plan`` first for the $0 resolved-config preview of EACH arm. Example:
    nohup bash -c '.venv/bin/python experiments/launch_bind_all_taper_ab.py \
        --arm arm_b \
        --kd-warm-start-dir experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best \
        --out-dir experiments/results/bind_all_ab_<stamp>/arm_b \
        --train-device mps --split-by-head --n-pairs 600 --rate-attack --go' \
        </dev/null >.../arm_b.outer.log 2>&1 & disown
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# The two arms differ ONLY in taper_channels. arm_a passes the vendored schedule
# EXPLICITLY (not None) so BOTH arms route through the ConfigurableTaperHNeRVDecoder
# carrier — the A/B isolates the channel SCHEDULE, not the decoder class.
_ARM_A_TAPER = [20, 20, 20, 15, 11, 10, 10]  # vendored HNeRV taper (control)
_ARM_B_TAPER = [22, 16, 15, 14, 15, 14, 10]  # solved byte-neutral taper (treatment)
_ARMS = ("arm_a", "arm_b")

# PR95 8-stage curriculum (0-indexed): 0=CE 1=tau 2=smooth 3=QAT (the COARSE phase) |
# 4=C1a 5=lambda 6=sigma 7=muon (the REFINEMENT phase). The oomph seg lever overlays the
# refinement stages (the basin/KD-warm carries the coarse structure; the lever refines).
_REFINEMENT_FIRST_STAGE = 4

# OOMPH seg lever = the sharp argmax-flip d_seg config (matches
# launch_oomph_finetune_disambiguator._OOMPH): sharp anneal T 0.3→0.05 over the first
# half then hold (→ ≈ exact argmax-flip d_seg), tight margin τ=0.5 (concentrate harder
# on the flip-prone boundary), margin RENORM (true weighted mean), seg_weight 1.5×.
_OOMPH = {
    "seg_surrogate": "soft_cosine",
    "seg_temperature": 0.3,
    "seg_temperature_end": 0.05,
    "seg_temperature_hold_frac": 0.5,
    "margin_weight_tau": 0.5,
    "margin_weight_renorm": True,
    "seg_weight_mult": 1.5,
}

_LIVE_BASIN_DIRS = (
    Path("experiments/results/forkpoints"),
    Path("experiments/results/torch_vehicle_full_mps_basin_bc20_n600"),
)


def _taper_for_arm(arm: str) -> list[int]:
    return list(_ARM_A_TAPER) if arm == "arm_a" else list(_ARM_B_TAPER)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arm", required=True, choices=_ARMS,
                   help="arm_a (vendored taper, control) | arm_b (solved taper, "
                        "treatment). IDENTICAL config except taper_channels.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="arm run dir (resumes if a checkpoint is present). MUST be NEW, "
                        "NOT the live basin/forkpoint tree.")
    p.add_argument("--base-channels", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=28)
    p.add_argument("--total-epoch-budget", type=int, default=None,
                   help="PR95 schedule budget (None = full faithful 29,650; run to "
                        "CONVERGENCE per 'score > training time'). Resumable to extend.")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--checkpoint-every-epochs", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="AUTHORITY/eval device (CPU-TRUSTED). NO mps.")
    p.add_argument("--train-device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="gradient backend. mps requires --split-by-head (PoseNet stays "
                        "on the CPU authority; MPS corrupts PoseNet 23x).")
    p.add_argument("--split-by-head", action="store_true",
                   help="MPS SegNet grad + CPU PoseNet grad (the speed lever). Required "
                        "by --pose-film-trunk-stopgrad (the split holds a separable pose "
                        "cotangent to mask).")
    p.add_argument("--async-eval", action="store_true")
    p.add_argument("--n-pairs", type=int, default=600,
                   help="MUST match the KD-warm basin's latent basis (the basin = 600).")
    p.add_argument("--seed", type=int, default=0,
                   help="SHARED across arms → same-seed-same-distribution init (a "
                        "bit-identical init is impossible across DIFFERENT architectures; "
                        "this is the cleanest apples-to-apples for a taper A/B).")
    # --- KD-WARM-START (the wall-clock resolution; new P1a flags) -----------------
    p.add_argument("--kd-warm-start-dir", type=Path, default=None,
                   help="the converged VENDORED-taper basin best/ dir (best_ema_decoder.pt "
                        "+ best_ema_latents.pt) distilled into the re-tapered student "
                        "(latents load direct; decoder frame-MSE distilled). Sets "
                        "cfg.kd_warm_start_dir. REQUIRED for the bind-all production run "
                        "(else the re-taper trains from scratch = ~weeks on MPS).")
    p.add_argument("--kd-warm-epochs", type=int, default=300,
                   help="KD warm-up PREFIX length (stage-0 epochs of frame-MSE distillation "
                        "before the score-aware curriculum continues). Sets cfg.kd_warm_epochs.")
    p.add_argument("--kd-warm-lr", type=float, default=1e-3,
                   help="KD distillation AdamW lr. Sets cfg.kd_warm_lr.")
    p.add_argument("--kd-warm-train-latents", dest="kd_warm_train_latents",
                   action="store_true", default=True,
                   help="co-train the latents during KD distillation (DEFAULT — the student "
                        "adapts the code to its own taper). Sets cfg.kd_warm_train_latents.")
    p.add_argument("--no-kd-warm-train-latents", dest="kd_warm_train_latents",
                   action="store_false",
                   help="freeze the latents during KD distillation (decoder-only distill).")
    # --- FiLM-v2 COMPLETE POSE DECOUPLING -----------------------------------------
    p.add_argument("--pose-film-v2", action="store_true",
                   help="enable pose-FiLM v2 (residual rgb_0 head; rgb_1/d_seg FiLM-CLEAN) "
                        "so the SHARP oomph seg crank has NO pose cost (pose carried by the "
                        "6 stored scalars, Wyner-Ziv ~1KB). Sets pose_film_enabled=True + "
                        "pose_film_version=2.")
    p.add_argument("--pose-film-trunk-stopgrad", action="store_true",
                   help="COMPLETE the FiLM-v2 decoupling: route the pose loss gradient ONLY "
                        "to the FiLM pose path, STOP-GRADIENT on the shared trunk + latents + "
                        "rgb_0/rgb_1 → ∂d_seg/∂(pose-objective)=0 EXACTLY. Requires "
                        "--pose-film-v2 + --split-by-head. Sets cfg.pose_film_trunk_stopgrad.")
    p.add_argument("--pose-film-rgb0-pose-trainable", action="store_true",
                   help="REFINE trunk-stopgrad: EXCLUDE rgb_0 from the seg-only-restore set "
                        "so the pose loss trains the frame-0 head (SegNet reads ONLY rgb_1, "
                        "so ∂d_seg/∂(rgb_0)=0 → free pose capacity at ZERO d_seg cost). "
                        "Requires --pose-film-trunk-stopgrad. Sets "
                        "cfg.pose_film_rgb0_pose_trainable.")
    # --- POSE CADENCE (k=1 production default; APGC safety) ------------------------
    p.add_argument("--pose-grad-every-k", type=int, default=1,
                   help="pose-throttle (1 = pose EVERY epoch = production default per "
                        "'score > training time'). Sets cfg.pose_grad_every_k.")
    p.add_argument("--pose-grad-resume-threshold", type=float, default=0.0,
                   help="force pose every epoch while pose_mse exceeds this (self-protect "
                        "guard for k>1). Sets cfg.pose_grad_resume_threshold.")
    p.add_argument("--pose-grad-on-train-device", action="store_true",
                   help="FULL-MPS unbundle: run the PoseNet head's forward+backward on "
                        "--train-device (MPS) too — the speed lever that does NOT touch "
                        "score (the MPS pose GRADIENT is per-step faithful; the full-MPS "
                        "basin reached CPU-authority d_pose=0.00034). The AUTHORITY eval "
                        "(d_seg/d_pose that pick BEST) STILL runs on --device (CPU). Keeps "
                        "the per-axis cotangent SPLIT so equimarginal + Mahalanobis + FiLM-"
                        "v2 trunk-stopgrad all keep firing. Requires --split-by-head + "
                        "--train-device mps. Sets cfg.pose_grad_on_train_device.")
    p.add_argument("--pose-grad-adaptive", action="store_true",
                   help="APGC closed-loop pose throttle (the safety net; IGNORES static "
                        "k/threshold, holds d_pose at its moving floor). Requires "
                        "--split-by-head. Sets cfg.pose_grad_adaptive.")
    p.add_argument("--pose-grad-floor-tol", type=float, default=0.08)
    p.add_argument("--pose-grad-k-max", type=int, default=8)
    p.add_argument("--pose-grad-trend-window", type=int, default=3)
    # --- SOPHISTICATED POSE TREATMENT (Levers A + C; default-OFF byte-identical) ---
    p.add_argument("--pose-equimarginal", action="store_true",
                   help="Lever A: the EQUIMARGINAL pose-weight controller — scale pose_weight per epoch so "
                        "the measured per-axis cotangent-norm ratio ‖cot_pose‖/‖cot_seg‖ tracks rho (the "
                        "score-optimal balance; the WEIGHT analogue of APGC). Tracks 5/sqrt(10·d_pose) + "
                        "RESTORES the balance the oomph w_seg×1.5 overlay breaks. Requires --split-by-head. "
                        "Sets cfg.pose_equimarginal_enabled.")
    p.add_argument("--pose-equimarginal-rho", type=float, default=1.0,
                   help="target pose:seg score-marginal ratio (1.0 = true equimarginal; <1 biases toward "
                        "seg). Sets cfg.pose_equimarginal_rho.")
    p.add_argument("--pose-equimarginal-decay", type=float, default=0.9,
                   help="EMA decay for the measured cotangent-norm ratio. Sets cfg.pose_equimarginal_decay.")
    p.add_argument("--pose-equimarginal-tol", type=float, default=0.15,
                   help="deadband half-width as a fraction of rho (no correction inside). Sets "
                        "cfg.pose_equimarginal_tol.")
    p.add_argument("--pose-dim-weights-auto", action="store_true",
                   help="Lever C: per-dim Mahalanobis (inverse-variance) weighting of the 6 scored pose "
                        "dims, MEASURED on the basin pose targets (renormalised to mean 1.0 → overall pose "
                        "scale preserved; SegNet reads only rgb_1 so ZERO d_seg / ZERO bytes). Sets "
                        "cfg.pose_dim_weights from the measured target variance.")
    # --- REFINEMENT detail levers -------------------------------------------------
    p.add_argument("--ema-warmup", dest="ema_warmup", action="store_true", default=True,
                   help="EMA-shadow warmup (DEFAULT — the shadow tracks the short post-KD "
                        "fine-tune instead of freezing at the warm init). Sets cfg.ema_warmup.")
    p.add_argument("--no-ema-warmup", dest="ema_warmup", action="store_false",
                   help="disable the EMA warmup (constant-decay shadow; only for a long "
                        "from-scratch budget where the warmup window is negligible).")
    p.add_argument("--muon-lr-floor-fix", action="store_true", default=True,
                   help="Muon gets its own cosine floor keyed to muon_lr (DEFAULT for the "
                        "scaled run). Sets cfg.muon_lr_floor_fix.")
    p.add_argument("--no-muon-lr-floor-fix", dest="muon_lr_floor_fix", action="store_false")
    p.add_argument("--oomph-seg-weight-mult", type=float, default=1.5,
                   help="the oomph seg_weight multiplier over the stage base (1.5 = "
                        "'everything cranked'; 1.0 isolates sharpness/margin/renorm).")
    # --- RATE ATTACK (Lever-4 variable-level export) ------------------------------
    p.add_argument("--rate-attack", action="store_true",
                   help="enable the Lever-4 variable-level EXPORT codec — the export decoder "
                        "blob is built at a variable per-tensor INT8 grid from the ONLINE "
                        "score-sensitivity EMA (the contest-OPTIMAL byte-half of Lever-4, on "
                        "the SAME shared spine as the score-aware QAT). Sets "
                        "cfg.lever4_variable_level_export_enabled.")
    p.add_argument("--targets-cache", type=Path,
                   default=Path("experiments/results/capstone_gt_targets_cache"))
    p.add_argument("--video-path", type=Path, default=None)
    p.add_argument("--go", action="store_true",
                   help="REQUIRED to launch the real-scorer arm (loads the heavy scorer).")
    p.add_argument("--print-plan", action="store_true",
                   help="print the resolved config + per-stage lever overlay for this arm "
                        "and exit ($0 — NO training, NO scorer load).")
    return p


def _overlay_oomph(specs: list, oomph_seg_weight_mult: float) -> list:
    """Overlay the OOMPH sharp soft_cosine seg lever on the REFINEMENT stages
    (>= _REFINEMENT_FIRST_STAGE); the coarse stages keep vendored CE (the basin/KD-warm
    carries the coarse structure). The ONLY non-architecture difference vs the vendored
    curriculum is the refinement seg loss form/sharpness/weight."""
    out = []
    for i, s in enumerate(specs):
        if i >= _REFINEMENT_FIRST_STAGE:
            out.append(
                dataclasses.replace(
                    s,
                    seg_surrogate=_OOMPH["seg_surrogate"],
                    seg_temperature=_OOMPH["seg_temperature"],
                    seg_temperature_end=_OOMPH["seg_temperature_end"],
                    seg_temperature_hold_frac=_OOMPH["seg_temperature_hold_frac"],
                    margin_weight_tau=_OOMPH["margin_weight_tau"],
                    margin_weight_renorm=_OOMPH["margin_weight_renorm"],
                    seg_weight=float(s.seg_weight) * float(oomph_seg_weight_mult),
                )
            )
        else:
            out.append(s)  # coarse stage: vendored CE, no seg lever
    return out


def _plan_stages(specs: list, oomph_seg_weight_mult: float) -> list[dict]:
    over = _overlay_oomph(specs, oomph_seg_weight_mult)
    return [
        {
            "stage": i, "name": s.name, "epochs": s.epochs,
            "use_muon": s.use_muon, "use_qat": s.use_qat,
            "seg_surrogate": s.seg_surrogate, "seg_temperature": s.seg_temperature,
            "seg_temperature_end": s.seg_temperature_end,
            "seg_temperature_hold_frac": s.seg_temperature_hold_frac,
            "margin_weight_tau": s.margin_weight_tau,
            "margin_weight_renorm": s.margin_weight_renorm,
            "seg_weight": s.seg_weight,
        }
        for i, s in enumerate(over)
    ]


def _resolve_config_dict(args) -> dict:
    """The resolved config for an arm (the dry-run preview AND the source-of-truth the
    real run reads). Pure — no scorer load, no training. Used by --print-plan and by the
    tests (so the tests verify exactly what the run uses)."""
    taper = _taper_for_arm(args.arm)
    return {
        "arm": args.arm,
        "taper_channels": taper,
        "base_channels": args.base_channels,
        "latent_dim": args.latent_dim,
        "total_epoch_budget": args.total_epoch_budget,
        "n_pairs": args.n_pairs,
        "seed": args.seed,
        "device": args.device,
        "train_device": args.train_device,
        "split_by_head": bool(args.split_by_head),
        "async_eval": bool(args.async_eval),
        # KD-warm-start (the new P1a flags → driver Config fields)
        "kd_warm_start_dir": (str(args.kd_warm_start_dir)
                              if args.kd_warm_start_dir is not None else None),
        "kd_warm_epochs": args.kd_warm_epochs,
        "kd_warm_lr": args.kd_warm_lr,
        "kd_warm_train_latents": bool(args.kd_warm_train_latents),
        # FiLM-v2 complete pose decoupling
        "pose_film_enabled": bool(args.pose_film_v2),
        "pose_film_version": 2 if args.pose_film_v2 else 1,
        "pose_film_trunk_stopgrad": bool(args.pose_film_trunk_stopgrad),
        "pose_film_rgb0_pose_trainable": bool(args.pose_film_rgb0_pose_trainable),
        # sophisticated pose treatment (Levers A + C)
        "pose_equimarginal": bool(args.pose_equimarginal),
        "pose_equimarginal_rho": args.pose_equimarginal_rho,
        "pose_equimarginal_decay": args.pose_equimarginal_decay,
        "pose_equimarginal_tol": args.pose_equimarginal_tol,
        "pose_dim_weights_auto": bool(args.pose_dim_weights_auto),
        # pose cadence
        "pose_grad_every_k": args.pose_grad_every_k,
        "pose_grad_resume_threshold": args.pose_grad_resume_threshold,
        "pose_grad_on_train_device": bool(args.pose_grad_on_train_device),
        "pose_grad_adaptive": bool(args.pose_grad_adaptive),
        "pose_grad_floor_tol": args.pose_grad_floor_tol,
        "pose_grad_k_max": args.pose_grad_k_max,
        "pose_grad_trend_window": args.pose_grad_trend_window,
        # refinement detail
        "ema_decay": args.ema_decay,
        "ema_warmup": bool(args.ema_warmup),
        "muon_lr_floor_fix": bool(args.muon_lr_floor_fix),
        "oomph_seg_weight_mult": args.oomph_seg_weight_mult,
        # rate attack
        "lever4_variable_level_export_enabled": bool(args.rate_attack),
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
    }


def _build_torch_vehicle_config(args, out_dir: Path, *, pose_dim_weights=None):
    """Build the live ``TorchVehicleConfig`` for an arm. Every field maps to a parsed arg
    (no invented flags). Imported lazily so --print-plan stays $0/import-light.

    ``pose_dim_weights`` (Lever C): the length-6 weight tuple MEASURED on the basin pose targets by the
    caller (``--pose-dim-weights-auto``), or None (uniform, byte-identical)."""
    from tac.torch_vehicle.driver import TorchVehicleConfig

    return TorchVehicleConfig(
        base_channels=args.base_channels, latent_dim=args.latent_dim, out_dir=out_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        total_epoch_budget=args.total_epoch_budget, ema_decay=args.ema_decay,
        eval_every=args.eval_every, device=args.device, train_device=args.train_device,
        split_by_head=args.split_by_head, async_eval=args.async_eval,
        # ARCHITECTURE: the A/B axis. Both arms use the configurable-taper carrier.
        taper_channels=_taper_for_arm(args.arm),
        # WARM-START: KD-warm-start (the new P1a flags).
        kd_warm_start_dir=args.kd_warm_start_dir,
        kd_warm_epochs=args.kd_warm_epochs,
        kd_warm_lr=args.kd_warm_lr,
        kd_warm_train_latents=args.kd_warm_train_latents,
        # POSE DECOUPLE: FiLM-v2 + trunk-stopgrad + rgb_0-pose-trainable.
        pose_film_enabled=bool(args.pose_film_v2),
        pose_film_version=2 if args.pose_film_v2 else 1,
        pose_film_trunk_stopgrad=bool(args.pose_film_trunk_stopgrad),
        pose_film_rgb0_pose_trainable=bool(args.pose_film_rgb0_pose_trainable),
        # SOPHISTICATED POSE TREATMENT: Lever A (equimarginal weight controller) + Lever C (per-dim weights).
        pose_equimarginal_enabled=bool(args.pose_equimarginal),
        pose_equimarginal_rho=args.pose_equimarginal_rho,
        pose_equimarginal_decay=args.pose_equimarginal_decay,
        pose_equimarginal_tol=args.pose_equimarginal_tol,
        pose_dim_weights=pose_dim_weights,
        # POSE CADENCE: k=1 production default + APGC safety.
        pose_grad_every_k=args.pose_grad_every_k,
        pose_grad_resume_threshold=args.pose_grad_resume_threshold,
        pose_grad_on_train_device=bool(args.pose_grad_on_train_device),
        pose_grad_adaptive=args.pose_grad_adaptive,
        pose_grad_floor_tol=args.pose_grad_floor_tol,
        pose_grad_k_max=args.pose_grad_k_max,
        pose_grad_trend_window=args.pose_grad_trend_window,
        # REFINEMENT detail: EMA-warmup + Muon lr-floor fix.
        ema_warmup=bool(args.ema_warmup),
        muon_lr_floor_fix=bool(args.muon_lr_floor_fix),
        # RATE ATTACK: Lever-4 variable-level export.
        lever4_variable_level_export_enabled=bool(args.rate_attack),
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    from tac.torch_vehicle.curriculum import build_curriculum

    specs = build_curriculum(
        total_epoch_budget=args.total_epoch_budget,
        ema_decay=args.ema_decay, eval_every=args.eval_every,
    )
    resolved = _resolve_config_dict(args)

    if args.print_plan:
        print(json.dumps({
            **resolved,
            "stages": _plan_stages(specs, args.oomph_seg_weight_mult),
        }, indent=2))
        return 0

    out_dir = Path(args.out_dir)
    for live in _LIVE_BASIN_DIRS:
        try:
            if live.resolve() in out_dir.resolve().parents or out_dir.resolve() == live.resolve():
                raise SystemExit(
                    f"REFUSED: --out-dir {out_dir} inside the live basin tree {live}. "
                    "Use a NEW results dir for the A/B."
                )
        except FileNotFoundError:
            pass
    if args.split_by_head and args.train_device != "mps":
        raise SystemExit("--split-by-head requires --train-device mps")
    if args.train_device == "mps" and not args.split_by_head:
        # The guard enforces the per-axis SPLIT (the substrate the equimarginal /
        # Mahalanobis / FiLM-v2 levers ride on), NOT "the pose gradient must be CPU".
        # The pose gradient MAY run on MPS via --pose-grad-on-train-device — the MPS pose
        # GRADIENT is faithful; only the SCORE may never come from MPS. The authority rule
        # is enforced separately below (--device is CPU/CUDA; __post_init__ refuses
        # device='mps'; exact_eval always runs on --device).
        raise SystemExit("--train-device mps must be used WITH --split-by-head (the per-"
                         "axis cotangent split). The pose GRADIENT may run on MPS via "
                         "--pose-grad-on-train-device; the SCORE never does (--device is "
                         "the CPU/CUDA authority).")
    # AUTHORITY INVARIANT (the rule --pose-grad-on-train-device does NOT relax): the exact
    # d_seg/d_pose that pick BEST run on --device, which MUST be CPU/CUDA, NEVER MPS. The
    # driver's __post_init__ also refuses device='mps', but fail loudly at the launcher too.
    if str(args.device).lower().startswith("mps"):
        raise SystemExit("--device (the AUTHORITY/eval device) must be cpu or cuda, NEVER "
                         "mps (CLAUDE.md 'MPS auth eval is NOISE': 23x pose drift). MPS is "
                         "the GRADIENT backend only (--train-device / "
                         "--pose-grad-on-train-device).")
    if args.pose_grad_on_train_device and not args.split_by_head:
        raise SystemExit("--pose-grad-on-train-device requires --split-by-head (it selects "
                         "the train device for the separable PoseNet head of the split "
                         "backward).")
    if args.pose_grad_on_train_device and args.train_device != "mps":
        raise SystemExit("--pose-grad-on-train-device requires --train-device mps (the "
                         "whole point is the MPS pose gradient; with a CPU train device it "
                         "is a no-op).")
    # SELF-PROTECT (regression guard, 2026-06-17): the MPS-pose UNBUNDLE
    # (cfg.pose_grad_on_train_device) already exists in the driver + this launcher. A run
    # that uses --train-device mps --split-by-head but OMITS --pose-grad-on-train-device
    # silently takes the PoseNet-backward-on-CPU path: ~133 s/ep vs ~18-40 full-MPS — a
    # 7.3x wall-clock REGRESSION for ZERO score benefit (the full_mps basin proved the MPS
    # pose GRADIENT faithful at d_pose 0.00034; the SCORE always stays on the CPU authority
    # via --device). Warn loudly so the slow path is a DELIBERATE choice (suspected MPS-pose
    # Muon chaos), never an accidental omission like the arm_b/corrected launches were.
    if (
        args.train_device == "mps"
        and args.split_by_head
        and not args.pose_grad_on_train_device
    ):
        print(
            "[WARN] CPU-PoseNet path selected (--split-by-head + --train-device mps WITHOUT "
            "--pose-grad-on-train-device): ~133 s/ep vs ~18-40 full-MPS = ~7.3x SLOWER for "
            "ZERO score gain. The MPS-pose unbundle exists (basin proved the gradient "
            "faithful; SCORE stays on the --device CPU authority). Add "
            "--pose-grad-on-train-device unless you are deliberately avoiding suspected "
            "MPS-pose Muon chaos.",
            file=sys.stderr,
            flush=True,
        )
    if not args.go:
        raise SystemExit("REFUSED without --go. Use --print-plan for the $0 resolved-config "
                         "preview.")

    out_dir.mkdir(parents=True, exist_ok=True)
    from tac.torch_vehicle.driver import TorchVehicleDriver, import_vendored_bundle
    from tac.torch_vehicle.scorer_context import RealScorerContext

    video_path = args.video_path
    if video_path is None:
        from tac.torch_vehicle.vendored_imports import import_vendored

        video_path = import_vendored("data").get_default_video_path()

    overlaid = _overlay_oomph(specs, args.oomph_seg_weight_mult)
    scorer = RealScorerContext(
        video_path, device=args.device, train_device=args.train_device,
        split_by_head=args.split_by_head, max_pairs=args.n_pairs,
        targets_cache=args.targets_cache,
    )
    # Lever C: measure the per-dim pose weights from the REAL basin pose targets (inverse-variance
    # Mahalanobis). None when --pose-dim-weights-auto is off → uniform → byte-identical.
    pose_dim_weights = None
    if bool(args.pose_dim_weights_auto):
        from tac.torch_vehicle.pose_dim_weights import (
            measure_pose_dim_weights_from_targets,
        )

        pose_dim_weights = measure_pose_dim_weights_from_targets(
            scorer.pose_targets, mode="inv_var"
        )
        print(json.dumps({"lever_c_pose_dim_weights_measured": list(pose_dim_weights),
                          "authority": "[contest-CPU advisory] NON-PROMOTABLE"}), flush=True)
    cfg = _build_torch_vehicle_config(args, out_dir, pose_dim_weights=pose_dim_weights)
    print(json.dumps({
        **resolved, "out_dir": str(out_dir),
        "stages": _plan_stages(specs, args.oomph_seg_weight_mult),
    }, indent=2), flush=True)

    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=overlaid
    )
    summary = driver.run()
    print(json.dumps({**summary, "arm": args.arm,
                      "authority": "[contest-CPU advisory] NON-PROMOTABLE"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
