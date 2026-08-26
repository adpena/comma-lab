# SPDX-License-Identifier: MIT
"""MARGIN-HINGE vs PR95 L7-SOFTPLUS — int8-QAT gradient-geometry potency probe (2026-06-22).

THE MANDATE (the watch-item from ``decisive_run_161_config_and_deepmath_optimality_audit``
section C): the decisive run applies OUR ``margin_hinge`` seg loss (``margin_target=1.0``)
across ALL stages via ``--seg-margin-hinge``. PR95's vendored stage-5 finisher uses
``l7_softplus_seg_loss`` (``tau=0.3, l7_threshold=1.0, l7_mult=4.0``). The int5 finding
(``feedback_frontier_int5_score_aware_qat_finetune_path_b_caps``) suggested margin_hinge
might "minimize the surrogate on FP but lose it post-quant" at a coarse grid. This probe
MEASURES the per-step d_seg potency of the two losses AT the real fork point
(``stage4_v332_qat_end``, post-QAT / pre-stage-5), under int8 fake-quant — the deployed
authority surface.

WHAT THIS IS (and is NOT):
  * IT IS a per-step gradient-geometry MEASUREMENT (CLAUDE.md "Measurement-first"): for
    each loss, take ONE gradient at the fork point, apply ``w' = w - lr*g`` for an LR
    sweep, and measure Δd_seg both FP and int8-fake-quant. Plus the gradient cosine.
  * IT IS NOT a curriculum-outcome verdict and NOT a kill (CLAUDE.md "Forbidden premature
    KILL"). It is a CANDIDATE-disambiguator for a future bat00 per-stage-loss A/B. The
    decisive arbiter remains the live run's Muon stage-8 d_seg slope.

DISCIPLINE (CLAUDE.md):
  * Apples-to-apples: the seg-loss FUNCTION is the ONLY variable. SAME fork-point EMA
    decoder + latents (deepcopy per arm), SAME 24 pairs, SAME LR sweep, SAME int8 quant,
    NO pose term (seg isolated). d_seg is the contest argmax-disagreement rate measured on
    the SAME frozen CPU SegNet through the SAME eval round-trip for every arm.
  * CPU ONLY (``device='cpu'``). MPS is FORBIDDEN as the d_seg surface (CLAUDE.md "MPS auth
    eval is NOISE": 2x SegNet drift) AND is the live run's device — we must not contend.
    CPU SegNet IS the d_seg authority anyway. ``OMP_NUM_THREADS`` capped by the launcher.
  * READ-ONLY on the live run: only the preserved ``stage4_v332_qat_end`` snapshot is read.
  * N=24 pairs (small; minutes on CPU). Every number ``[contest-CPU advisory]``
    NON-PROMOTABLE; pointer UNMOVED; NO score claim.

REUSE (no scorer / decoder / quant / roundtrip reimplemented):
  * ``RealScorerContext`` — frozen CPU SegNet + cached n-pair GT argmax targets.
  * ``configurable_taper_decoder.ConfigurableTaperHNeRVDecoder`` — the run's decoder.
  * ``tac.torch_vehicle.driver._seg_loss_for_spec`` (+ a margin_hinge ``StageSpec``) — the
    EXACT loss path the live ``--seg-margin-hinge`` run uses.
  * vendored ``losses.l7_softplus_seg_loss`` — ``specs[4].seg_loss_fn`` (PR95 stage-5).
  * ``tac.torch_vehicle.score_aware_qat._fake_quantize_n`` — the vendored int8 (127-level)
    per-tensor symmetric fake-quant (the stage-4 QAT quant).
  * ``probe_lensA``'s ``_roundtrip`` math (bicubic↑874→bilinear↓384→uint8-STE) is re-derived
    here inline (small, 1:1 with the driver) so the probe is self-contained.

USAGE:
  .venv/bin/python experiments/probe_margin_hinge_vs_l7_softplus_grad_geometry.py \
      --n-pairs 24 --out-json experiments/results/margin_hinge_vs_l7_probe_20260622/result.json
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from tac.torch_vehicle.driver import _seg_loss_for_spec
from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.score_aware_qat import _fake_quantize_n
from tac.torch_vehicle.scorer_context import RealScorerContext
from tac.torch_vehicle.vendored_imports import import_vendored

_EVAL_H, _EVAL_W = 384, 512
_SEG_NUM_CLASSES = 5
_RUN_DIR = Path("experiments/results/yousfi_r3_taper_marginhinge_e5_20260620")
_SNAP = _RUN_DIR / "stage_snapshots" / "stage4_v332_qat_end"
_VIDEO = Path("upstream/videos/0.mkv")
# The capstone cache already holds gt_targets_n24.pt etc. (zero recompute).
_TARGETS_CACHE = Path("experiments/results/capstone_gt_targets_cache")
# Run config (manifest): base_channels=20, latent_dim=28, taper as below.
_BASE_CH = 20
_LATENT_DIM = 28
_TAPER = [16, 16, 17, 19, 19, 14, 10]
_QAT_N_LEVELS = 127  # the vendored stage-4 int8 QAT level count


def _roundtrip(decoded_pair: torch.Tensor) -> torch.Tensor:
    """1:1 port of the driver per-step frame math: bicubic↑874→bilinear↓384→uint8-STE.

    ``decoded_pair`` is (B, 2, 3, 384, 512) float [0,255]; returns (B, 2, 384, 512, 3)
    (BHWC, the scorer's ``preprocess_input`` expects the channel-last pair frames)."""
    B = decoded_pair.shape[0]
    flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    decoded_clamped = decoded_bhwc.clamp(0, 255)
    return decoded_clamped + (decoded_clamped.round() - decoded_clamped).detach()


def _build_decoder() -> torch.nn.Module:
    from tac.torch_vehicle.configurable_taper_decoder import ConfigurableTaperHNeRVDecoder

    return ConfigurableTaperHNeRVDecoder(
        latent_dim=_LATENT_DIM,
        base_channels=_BASE_CH,
        eval_size=(_EVAL_H, _EVAL_W),
        channels=list(_TAPER),
    )


def _quantize_decoder_inplace(decoder: torch.nn.Module, n_levels: int) -> None:
    """Apply the vendored int8 per-tensor symmetric fake-quant to every Conv2d/Linear
    WEIGHT in place (the stage-4 QAT quant; biases left FP, matching the vendored QAT
    which quantizes weights only). Mutates the live ``.data`` (caller works on a copy)."""
    with torch.no_grad():
        for m in decoder.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                m.weight.data.copy_(_fake_quantize_n(m.weight.data, n_levels))


@torch.no_grad()
def _measure_dseg(decoder: torch.nn.Module, latents: torch.Tensor, scorer) -> float:
    """Contest d_seg = mean per-pixel argmax-disagreement (SegNet(render).argmax !=
    GT.argmax) over ALL n pairs, on the frozen CPU SegNet through the eval round-trip.
    This is the SAME argmax-disagreement the contest computes (NOT a proxy)."""
    decoder.eval()
    decoded_pair = decoder(latents)  # (n, 2, 3, 384, 512)
    decoded_bhwc = _roundtrip(decoded_pair)
    seg_out = scorer.seg_forward_train(decoded_bhwc)  # frozen CPU SegNet (B,5,Hs,Ws)
    pred_argmax = seg_out.argmax(dim=1)
    gt_argmax = scorer.seg_targets_hard  # (n, Hs, Ws) on CPU (device==train_device==cpu)
    return float((pred_argmax != gt_argmax).float().mean().item())


def _margin_hinge_spec() -> StageSpec:
    """A seg-only StageSpec routing through ``--seg-margin-hinge`` (margin_target=1.0) —
    the EXACT loss path the live decisive run uses for the seg term."""
    return StageSpec(
        name="probe_margin_hinge",
        epochs=1,
        seg_loss_fn=lambda s, t: torch.zeros((), device=s.device),  # unused on surrogate path
        eval_every=1,
        batch_size=1,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=3e-5,
        muon_lr=0.0,
        muon_weight_decay=0.0,
        latent_lr_mult=1.0,
        grad_clip=1.0,
        grad_clip_muon=None,
        lr_floor_ratio=1.0,
        seg_weight=1.0,
        pose_weight=0.0,
        cat_lambda=0.0,
        cat_sigma=0.0,
        use_qat=False,
        init_latents_random=False,
        seg_surrogate="margin_hinge",
        seg_temperature=1.0,
        margin_weight_tau=None,
        seg_margin_hinge_target=1.0,
    )


def _grad_for_loss(
    *, loss_name: str, decoder: torch.nn.Module, latents: torch.Tensor, scorer,
    losses_mod, mh_spec: StageSpec,
) -> tuple[dict[str, torch.Tensor], float, torch.Tensor]:
    """Compute the decoder-param gradient of the seg loss at the fork point (FP decoder).

    Returns: (grad_dict {name->grad}, scalar_loss_value, flat_grad_vector). The render is
    the SAME eval-roundtrip render both arms see; only the loss FUNCTION differs."""
    decoder.train()
    decoder.zero_grad(set_to_none=True)
    decoded_pair = decoder(latents)
    decoded_bhwc = _roundtrip(decoded_pair)
    seg_out = scorer.seg_forward_train(decoded_bhwc)  # (B,5,Hs,Ws), CPU frozen SegNet
    gt_hard = scorer.seg_targets_hard  # (B,Hs,Ws)
    if loss_name == "margin_hinge":
        seg_l = _seg_loss_for_spec(mh_spec, seg_out, gt_hard, temperature=None)
    elif loss_name == "l7_softplus":
        seg_l = losses_mod.l7_softplus_seg_loss(
            seg_out, gt_hard, tau=0.3, l7_threshold=1.0, l7_mult=4.0
        )
    else:
        raise ValueError(loss_name)
    seg_l.backward()
    grads: dict[str, torch.Tensor] = {}
    flat_parts: list[torch.Tensor] = []
    for name, p in decoder.named_parameters():
        if p.grad is not None:
            g = p.grad.detach().clone()
            grads[name] = g
            flat_parts.append(g.reshape(-1))
    flat = torch.cat(flat_parts) if flat_parts else torch.zeros(1)
    return grads, float(seg_l.item()), flat


def _step_copy_and_measure(
    *, base_state: dict, grads: dict[str, torch.Tensor], lr: float, scorer, latents,
    n_levels: int,
) -> tuple[float, float]:
    """w' = w - lr*g on a COPY; return (d_seg_fp, d_seg_quant). The latents are FROZEN at
    the fork-point value (we isolate the DECODER-weight step — the per-step seg potency of
    the loss on the deployed weights). Both d_seg use the SAME frozen CPU SegNet."""
    # FP arm
    dec_fp = _build_decoder()
    dec_fp.load_state_dict(copy.deepcopy(base_state))
    with torch.no_grad():
        for name, p in dec_fp.named_parameters():
            if name in grads:
                p.data.add_(grads[name], alpha=-lr)
    d_seg_fp = _measure_dseg(dec_fp, latents, scorer)
    # int8-fake-quant arm (the authority/deployed surface): quantize the stepped weights.
    dec_q = _build_decoder()
    dec_q.load_state_dict(dec_fp.state_dict())  # same stepped weights
    _quantize_decoder_inplace(dec_q, n_levels)
    d_seg_quant = _measure_dseg(dec_q, latents, scorer)
    return d_seg_fp, d_seg_quant


def _multistep_descent(
    *, loss_name: str, base_state: dict, base_latents: torch.Tensor, scorer, losses_mod,
    mh_spec: StageSpec, n_steps: int, eval_every: int, adamw_lr: float, grad_clip: float,
    n_levels: int,
) -> dict:
    """Multi-step AdamW seg-only descent from the fork point (the per-step single gradient
    is at the d_seg noise floor on a converged basin; a multi-step descent gives a real
    potency curve). Trains the DECODER + LATENTS (the live run trains both) on the SAME n
    pairs with the SAME LR/clip/optimizer; the ONLY variable is the seg loss FUNCTION.

    At each eval point measures d_seg FP (live weights) AND d_seg int8-fake-quant (the
    deployed authority surface) — so the quant-erosion is tracked along the whole descent,
    not just at one step. Returns the FP + quant d_seg curves + endpoints."""
    decoder = _build_decoder()
    decoder.load_state_dict(copy.deepcopy(base_state))
    decoder.train()
    latents = base_latents.clone().detach().requires_grad_(True)
    opt = torch.optim.AdamW(
        [{"params": decoder.parameters()}, {"params": [latents]}], lr=adamw_lr
    )

    def _eval_both() -> tuple[float, float]:
        d_fp = _measure_dseg(decoder, latents, scorer)
        dq = _build_decoder()
        dq.load_state_dict(decoder.state_dict())
        _quantize_decoder_inplace(dq, n_levels)
        d_q = _measure_dseg(dq, latents, scorer)
        return d_fp, d_q

    d0_fp, d0_q = _eval_both()
    curve_fp = [(0, d0_fp)]
    curve_q = [(0, d0_q)]
    loss_first = loss_last = float("nan")
    for step in range(1, n_steps + 1):
        decoder.train()
        opt.zero_grad(set_to_none=True)
        decoded_pair = decoder(latents)
        decoded_bhwc = _roundtrip(decoded_pair)
        seg_out = scorer.seg_forward_train(decoded_bhwc)
        gt_hard = scorer.seg_targets_hard
        if loss_name == "margin_hinge":
            seg_l = _seg_loss_for_spec(mh_spec, seg_out, gt_hard, temperature=None)
        else:
            seg_l = losses_mod.l7_softplus_seg_loss(
                seg_out, gt_hard, tau=0.3, l7_threshold=1.0, l7_mult=4.0
            )
        seg_l.backward()
        torch.nn.utils.clip_grad_norm_([*decoder.parameters(), latents], grad_clip)
        if step == 1:
            loss_first = float(seg_l.item())
        loss_last = float(seg_l.item())
        opt.step()
        if step % eval_every == 0 or step == n_steps:
            d_fp, d_q = _eval_both()
            curve_fp.append((step, d_fp))
            curve_q.append((step, d_q))
    return {
        "loss_first": loss_first,
        "loss_last": loss_last,
        "d_seg_fp_curve": curve_fp,
        "d_seg_quant_curve": curve_q,
        "d_seg_fp_start": curve_fp[0][1],
        "d_seg_fp_end": curve_fp[-1][1],
        "d_seg_quant_start": curve_q[0][1],
        "d_seg_quant_end": curve_q[-1][1],
        "delta_d_seg_fp_total": curve_fp[-1][1] - curve_fp[0][1],
        "delta_d_seg_quant_total": curve_q[-1][1] - curve_q[0][1],
        # quant erosion of the DESCENT = how much the int8 grid eats the FP descent.
        "quant_erosion_total": (curve_q[-1][1] - curve_q[0][1])
        - (curve_fp[-1][1] - curve_fp[0][1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument(
        "--lrs", type=str, default="1e-4,3e-4,1e-3",
        help="comma-separated LR sweep (the stage-5 adamw_lr is 3e-5; we sweep a few "
        "scales above it to make a SINGLE gradient step produce a measurable Δd_seg).",
    )
    ap.add_argument(
        "--n-steps", type=int, default=60,
        help="multi-step descent length per loss (the single-step deltas sit at the d_seg "
        "noise floor on a converged basin; a multi-step descent gives a real potency curve). "
        "0 disables the multi-step section (single-gradient geometry only).",
    )
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument(
        "--descent-lr", type=float, default=1e-3,
        help="AdamW LR for the multi-step descent (a few scales above stage-5's 3e-5 so a "
        "short window produces a measurable descent; the SAME LR for both losses).",
    )
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]

    if not _SNAP.exists():
        raise SystemExit(f"fork-point snapshot not found: {_SNAP}")
    dec_path = _SNAP / "best_ema_decoder.pt"
    lat_path = _SNAP / "best_ema_latents.pt"

    t0 = time.time()
    print(f"[probe] CPU fork-point gradient-geometry probe; n={args.n_pairs}", flush=True)
    print(f"[probe] fork point: {_SNAP}", flush=True)

    # Frozen CPU SegNet + cached n-pair GT targets (zero recompute if cache hit).
    scorer = RealScorerContext(
        _VIDEO,
        device="cpu",
        max_pairs=int(args.n_pairs),
        targets_cache=_TARGETS_CACHE,
    )
    losses_mod = import_vendored("losses")

    # Load the fork-point EMA decoder state + the FIRST n latents (CPU).
    base_state = torch.load(dec_path, map_location="cpu", weights_only=False)
    if isinstance(base_state, dict) and "state_dict" in base_state:
        base_state = base_state["state_dict"]
    all_latents = torch.load(lat_path, map_location="cpu", weights_only=False)
    latents = all_latents[: int(args.n_pairs)].detach().clone()

    # Sanity: a fresh decoder loads the fork state strictly (architecture match).
    probe_dec = _build_decoder()
    probe_dec.load_state_dict(copy.deepcopy(base_state))  # raises on mismatch

    # BASELINE d_seg: the fork-point weights, int8-quantized (the deployed/authority
    # surface), AND fp (for the erosion reference at step 0).
    base_dec_fp = _build_decoder()
    base_dec_fp.load_state_dict(copy.deepcopy(base_state))
    baseline_fp = _measure_dseg(base_dec_fp, latents, scorer)
    base_dec_q = _build_decoder()
    base_dec_q.load_state_dict(copy.deepcopy(base_state))
    _quantize_decoder_inplace(base_dec_q, _QAT_N_LEVELS)
    baseline_quant = _measure_dseg(base_dec_q, latents, scorer)
    print(
        f"[probe] baseline d_seg: fp={baseline_fp:.6f}  int8quant={baseline_quant:.6f}  "
        f"(quant-erosion @ fork = {baseline_quant - baseline_fp:+.6f})",
        flush=True,
    )

    # Per-loss gradient at the fork point (on a FRESH FP decoder per arm so the .grad is
    # clean; the render is identical so this is apples-to-apples).
    arms: dict[str, dict] = {}
    flat_grads: dict[str, torch.Tensor] = {}
    for loss_name in ("margin_hinge", "l7_softplus"):
        dec = _build_decoder()
        dec.load_state_dict(copy.deepcopy(base_state))
        grads, loss_val, flat = _grad_for_loss(
            loss_name=loss_name, decoder=dec, latents=latents, scorer=scorer,
            losses_mod=losses_mod, mh_spec=_margin_hinge_spec(),
        )
        gnorm = float(flat.norm().item())
        print(f"[probe] {loss_name}: loss={loss_val:.6e}  ||g||={gnorm:.6e}", flush=True)
        flat_grads[loss_name] = flat
        sweep = []
        for lr in lrs:
            d_fp, d_q = _step_copy_and_measure(
                base_state=base_state, grads=grads, lr=lr, scorer=scorer,
                latents=latents, n_levels=_QAT_N_LEVELS,
            )
            row = {
                "lr": lr,
                "d_seg_fp": d_fp,
                "d_seg_quant": d_q,
                "delta_d_seg_fp": d_fp - baseline_fp,
                "delta_d_seg_quant": d_q - baseline_quant,
                # quant erosion of THIS loss's step = how much the int8 grid eats the
                # FP improvement (delta_quant - delta_fp; >0 means quant erodes the win).
                "quant_erosion_of_step": (d_q - baseline_quant) - (d_fp - baseline_fp),
            }
            sweep.append(row)
            print(
                f"[probe]   lr={lr:.1e}  Δd_seg_fp={row['delta_d_seg_fp']:+.6f}  "
                f"Δd_seg_quant={row['delta_d_seg_quant']:+.6f}  "
                f"erosion={row['quant_erosion_of_step']:+.6f}",
                flush=True,
            )
        arms[loss_name] = {"loss": loss_val, "grad_norm": gnorm, "sweep": sweep}

    # Gradient alignment.
    g_mh = flat_grads["margin_hinge"]
    g_l7 = flat_grads["l7_softplus"]
    cos = float(
        F.cosine_similarity(g_mh.unsqueeze(0), g_l7.unsqueeze(0)).item()
    )
    print(f"[probe] cosine(g_margin_hinge, g_l7_softplus) = {cos:.4f}", flush=True)

    # ---- MULTI-STEP descent (the discriminating potency signal) ----
    # The single-gradient step on a CONVERGED basin sits at the d_seg noise floor (one
    # step at LR<=1e-3 barely moves a converged d_seg). A short multi-step AdamW descent
    # per loss (SAME LR/clip/optimizer/pairs; loss FUNCTION is the only variable) gives a
    # real potency curve where any divergence (incl. quant-erosion divergence) would show.
    multistep: dict = {}
    if args.n_steps > 0:
        mh_spec = _margin_hinge_spec()
        for loss_name in ("margin_hinge", "l7_softplus"):
            ms = _multistep_descent(
                loss_name=loss_name, base_state=base_state, base_latents=latents,
                scorer=scorer, losses_mod=losses_mod, mh_spec=mh_spec,
                n_steps=int(args.n_steps), eval_every=int(args.eval_every),
                adamw_lr=float(args.descent_lr), grad_clip=float(args.grad_clip),
                n_levels=_QAT_N_LEVELS,
            )
            multistep[loss_name] = ms
            print(
                f"[probe] [{int(args.n_steps)}-step descent] {loss_name}: "
                f"d_seg_fp {ms['d_seg_fp_start']:.6f}->{ms['d_seg_fp_end']:.6f} "
                f"(Δ{ms['delta_d_seg_fp_total']:+.6f})  "
                f"d_seg_quant {ms['d_seg_quant_start']:.6f}->{ms['d_seg_quant_end']:.6f} "
                f"(Δ{ms['delta_d_seg_quant_total']:+.6f})  "
                f"quant_erosion {ms['quant_erosion_total']:+.6f}",
                flush=True,
            )

    # ---- HONEST verdict (per-step, ONE checkpoint; NOT a curriculum verdict, NO kill) ----
    # Best (most negative) post-quant Δd_seg per loss across the LR sweep.
    def best_quant_delta(name: str) -> float:
        return min(r["delta_d_seg_quant"] for r in arms[name]["sweep"])

    def best_quant_erosion_at_best_lr(name: str) -> dict:
        # the erosion at the LR that gave the best (most negative) FP delta
        rows = arms[name]["sweep"]
        best = min(rows, key=lambda r: r["delta_d_seg_fp"])
        return best

    mh_best_q = best_quant_delta("margin_hinge")
    l7_best_q = best_quant_delta("l7_softplus")
    mh_best_row = best_quant_erosion_at_best_lr("margin_hinge")
    l7_best_row = best_quant_erosion_at_best_lr("l7_softplus")

    # Verdict taxonomy (NO kill / NO wall):
    #   NO_DIVERGENCE_EVIDENCE  — the two losses are within noise post-quant → margin_hinge
    #     is fine; the flat stage-5 d_seg is the predicted AdamW conditioning-crawl, not the
    #     loss. Keep margin_hinge.
    #   L7_POST_QUANT_ADVANTAGE — l7_softplus is clearly more potent post-quant AND quant
    #     erodes margin_hinge MORE → the int5 divergence is a LIVE candidate → recommend the
    #     fuller bat00 per-stage-loss A/B (do NOT switch the live run; the decisive arbiter
    #     is the Muon stage-8 read).
    #   MARGIN_HINGE_ADVANTAGE  — margin_hinge wins post-quant → keep the lever (confirms the
    #     audit's expectation that margin_hinge is the validated d_seg lever).
    noise_band = 1.0 / float(
        scorer.seg_targets_hard.numel()
    )  # ~1 pixel of d_seg = the finest measurable change
    # The MULTI-STEP descent is the discriminating signal (the single step is noise on a
    # converged basin). When available, the verdict uses the multi-step post-quant deltas;
    # the single-step gap is kept as a secondary diagnostic.
    if multistep:
        mh_q_total = multistep["margin_hinge"]["delta_d_seg_quant_total"]
        l7_q_total = multistep["l7_softplus"]["delta_d_seg_quant_total"]
        mh_eros = multistep["margin_hinge"]["quant_erosion_total"]
        l7_eros = multistep["l7_softplus"]["quant_erosion_total"]
        gap = l7_q_total - mh_q_total  # <0 ⇒ l7 descends d_seg MORE post-quant; >0 ⇒ mh
        erosion_gap = mh_eros - l7_eros  # >0 ⇒ quant erodes margin_hinge MORE than l7
        decision_basis = "multistep_descent"
        # discrimination band: ~10 pixels of d_seg over the n pairs (a real, not-noise step)
        band = max(10.0 * noise_band, 5e-5)
    else:
        gap = l7_best_q - mh_best_q
        erosion_gap = (
            mh_best_row["quant_erosion_of_step"] - l7_best_row["quant_erosion_of_step"]
        )
        decision_basis = "single_step"
        band = max(noise_band, 5e-5)
    if abs(gap) <= band:
        verdict = "NO_DIVERGENCE_EVIDENCE"
    elif gap < 0 and erosion_gap > 0:
        verdict = "L7_POST_QUANT_ADVANTAGE"
    elif gap > 0:
        verdict = "MARGIN_HINGE_ADVANTAGE"
    else:
        # l7 more potent post-quant but NOT via more margin_hinge erosion → still informative
        verdict = "L7_POST_QUANT_ADVANTAGE_NOT_EROSION_DRIVEN"

    result = {
        "schema": "margin_hinge_vs_l7_softplus_grad_geometry_v1",
        "utc": datetime.now(UTC).isoformat(),
        "authority": "contest-CPU advisory (NON-PROMOTABLE)",
        "score_claim": False,
        "promotable": False,
        "device": "cpu",
        "fork_point": str(_SNAP),
        "n_pairs": int(args.n_pairs),
        "qat_n_levels": _QAT_N_LEVELS,
        "lrs": lrs,
        "baseline_d_seg_fp": baseline_fp,
        "baseline_d_seg_quant": baseline_quant,
        "baseline_quant_erosion": baseline_quant - baseline_fp,
        "noise_band_one_pixel": noise_band,
        "arms": arms,
        "multistep": multistep,
        "multistep_n_steps": int(args.n_steps),
        "multistep_descent_lr": float(args.descent_lr),
        "gradient_cosine_mh_vs_l7": cos,
        "best_quant_delta": {"margin_hinge": mh_best_q, "l7_softplus": l7_best_q},
        "post_quant_gap_l7_minus_mh": gap,
        "erosion_gap_mh_minus_l7": erosion_gap,
        "decision_basis": decision_basis,
        "verdict": verdict,
        "verdict_is_kill": False,
        "decisive_arbiter": "the live run's Muon stage-8 d_seg slope (this probe only "
        "disambiguates a candidate; it does NOT switch the live run)",
        "seconds": time.time() - t0,
    }
    print(f"[probe] VERDICT: {verdict}  (post-quant gap l7−mh={gap:+.6f}, "
          f"erosion gap mh−l7={erosion_gap:+.6f}, cos={cos:.4f})", flush=True)
    print(f"[probe] elapsed {result['seconds']:.1f}s", flush=True)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"[probe] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
