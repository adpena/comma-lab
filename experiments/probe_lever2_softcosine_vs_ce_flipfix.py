# SPDX-License-Identifier: MIT
"""LEVER-2-vs-CE GATE probe — does soft-cosine actually FIX argmax-flips, or is it
worse than the vendored CE seg loss on exactly the pixels d_seg measures?

THE QUESTION (gates the 30h from-0 run #116). Lever-2 REPLACES the vendored CE seg
loss (``F.cross_entropy(seg_logits, gt_argmax)``) with the soft-cosine surrogate
``1 - softmax(pred/T)[gt]`` (``tac.losses.core.segnet_surrogate_per_pixel``). R10/R12
established the surrogate's GRADIENT NORM decays vs temperature (the cold-T floor at
T≈0.3). But gradient-norm-alive is NOT the same as "fixes the right pixels": the
contest d_seg is the per-pixel ARGMAX-DISAGREEMENT rate between the GT frame's SegNet
argmax and the DECODED frame's SegNet argmax (``modules.SegNet.compute_distortion``:
``(out1.argmax(1) != out2.argmax(1)).float().mean()`` on the LAST frame). The
flip-set is the confidently-WRONG pixels. The analytic concern:

  * soft-cosine per-pixel = ``1 - p_gt`` where ``p_gt = softmax(pred/T)[gt]``. Its
    gradient on the GT-class logit ``∝ p_gt·(1 - p_gt)`` → 0 as ``p_gt → 0`` (a
    confidently-wrong flip). So soft-cosine's pull toward the GT class VANISHES on
    exactly the flip pixels it should fix.
  * CE per-pixel = ``-log p_gt``. Its gradient on the GT-class logit ``= p_gt - 1``,
    magnitude → 1 (MAX) as ``p_gt → 0``. CE pushes HARDEST on the confidently-wrong
    flips.

So Lever-2 may be a DOWNGRADE from CE on the very pixels d_seg scores. This probe
runs a controlled, apples-to-apples A/B that holds EVERYTHING constant except the
seg-loss function and measures which arm reduces the EXACT d_seg (argmax-flip rate)
more, plus a flip fixed-vs-left decomposition split by decoded-frame confidence.

DISCIPLINE (CLAUDE.md):
  * $0 local torch-CPU advisory; REAL frozen SegNet only (NO MPS — never a score
    authority). Every number is ``[macOS-CPU advisory] NON-PROMOTABLE``.
  * Apples-to-apples: the seg-loss FUNCTION is the ONLY variable. Same forkpoint
    decoder/latent init (deepcopy per arm), same pairs, same LR, same seg_weight=100,
    NO pose term (seg isolated), same optimizer, same step count.
  * NO edit to driver.py lever code (read-only). This is a measurement; we re-drive
    the SAME math the driver runs (roundtrip + ``seg_pose_forward`` + the loss router
    ``_seg_loss_for_spec``) on a deepcopy so the live arm is never perturbed.
  * Contention-aware: small pair slice + few steps (siblings + the live arm share CPU).

The d_seg "flip" is defined EXACTLY as the contest metric: a pixel where the decoded
frame's SegNet argmax != the GT frame's SegNet argmax (= ``seg_targets_hard``, which
the scorer context precomputes as ``argmax(SegNet(GT_last_frame))``).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import _seg_loss_for_spec, import_vendored_bundle
from tac.torch_vehicle.scorer_context import RealScorerContext
from tac.torch_vehicle.vendored_imports import import_vendored

_EVAL_H, _EVAL_W = 384, 512
_SEG_NUM_CLASSES = 5

_FORKPOINT = Path(
    "experiments/results/forkpoints/basin_bc20_20260612T121523Z/"
    "torch_vehicle_checkpoint_state.pt"
)
_VIDEO = Path("upstream/videos/0.mkv")
_TARGETS_CACHE = Path(".omx/tmp/lever2_vs_ce_targets_cache")


# ---------------------------------------------------------------------------
# Roundtrip — a 1:1 port of the driver's per-step frame math (read-only; driver.py
# lines ~597-604). Bicubic↑ to camera, bilinear↓ to eval, uint8-STE round.
# ---------------------------------------------------------------------------
def _roundtrip(decoded_pair: torch.Tensor) -> torch.Tensor:
    """``decoded_pair`` (B, 2, 3, 384, 512) float in [0,255] -> (B, 2, 384, 512, 3)
    BHWC, eval-roundtripped + uint8-STE rounded (gradient preserved)."""
    B = decoded_pair.shape[0]
    flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    decoded_clamped = decoded_bhwc.clamp(0, 255)
    return decoded_clamped + (decoded_clamped.round() - decoded_clamped).detach()


@dataclass
class _FlipStats:
    """Exact d_seg + flip-fix decomposition for one decoder state on the slice."""

    d_seg: float  # mean argmax-flip rate (the contest metric on this slice)
    n_flips: int  # total flipped pixels (decoded argmax != gt argmax)
    # confidence of the DECODED frame's softmax at the flipped pixels' decoded-argmax
    # class — high = "confidently wrong" (the pixels CE attacks, soft-cosine stalls on).
    n_flips_confident_wrong: int  # flips where decoded p(decoded-argmax) >= conf_thresh
    n_flips_uncertain: int  # flips where decoded p(decoded-argmax) <  conf_thresh


@torch.no_grad()
def _measure_flips(
    decoder: torch.nn.Module,
    latents: torch.Tensor,
    idx: torch.Tensor,
    scorer: RealScorerContext,
    *,
    conf_thresh: float = 0.5,
) -> _FlipStats:
    """Exact d_seg (argmax-flip rate vs GT) + flip-confidence decomposition on the
    slice ``idx``. Uses the SAME roundtrip + SegNet forward the driver/eval run, and
    ``scorer.seg_targets_hard`` (= GT SegNet argmax) as the flip reference — so the
    flip count is the contest d_seg numerator EXACTLY (NO proxy)."""
    decoder.eval()
    decoded_pair = decoder(latents[idx])
    decoded_bhwc = _roundtrip(decoded_pair)
    # SegNet-ONLY forward (pose is isolated by pose_weight=0, so the PoseNet pass in
    # seg_pose_forward is pure waste). On the CPU-authority non-split context the
    # train net IS the authority net, so seg_forward_train returns the AUTHORITY
    # SegNet logits — bit-identical to seg_pose_forward's seg_out (both = net.segnet(
    # segnet_in)) at ~half the per-step cost.
    seg_out = scorer.seg_forward_train(decoded_bhwc)  # (B, 5, Hs, Ws)
    decoded_argmax = seg_out.argmax(dim=1)  # (B, Hs, Ws)
    gt_argmax = scorer.seg_targets_hard[idx]  # (B, Hs, Ws) — argmax(SegNet(GT))
    flips = decoded_argmax != gt_argmax  # (B, Hs, Ws) bool — the EXACT d_seg flip set
    d_seg = flips.float().mean().item()
    n_flips = int(flips.sum().item())
    # Decoded-frame confidence at each flipped pixel's (wrong) decoded-argmax class.
    decoded_probs = F.softmax(seg_out, dim=1)  # (B, 5, Hs, Ws)
    decoded_conf = decoded_probs.max(dim=1).values  # (B, Hs, Ws) = p(decoded-argmax)
    flip_conf = decoded_conf[flips]  # (n_flips,)
    n_conf_wrong = int((flip_conf >= conf_thresh).sum().item())
    return _FlipStats(
        d_seg=d_seg,
        n_flips=n_flips,
        n_flips_confident_wrong=n_conf_wrong,
        n_flips_uncertain=n_flips - n_conf_wrong,
    )


@dataclass
class _ArmResult:
    arm: str
    seg_surrogate: str | None
    temperature: float | None
    margin_weight_tau: float | None
    d_seg_before: float
    d_seg_after: float
    d_seg_reduction: float  # before - after (positive = improvement)
    flips_before: int
    flips_after: int
    flips_fixed: int  # flips_before - flips_after (net; positive = fixed)
    conf_wrong_before: int
    conf_wrong_after: int
    conf_wrong_fixed: int  # net confidently-wrong flips fixed (the discriminating stat)
    grad_norm_first_step: float
    loss_first: float
    loss_last: float
    seconds: float


def _make_spec(
    seg_surrogate: str | None,
    temperature: float,
    margin_weight_tau: float | None,
    *,
    seg_weight: float,
    adamw_lr: float,
    grad_clip: float,
    ce_seg_loss,
) -> StageSpec:
    """A minimal seg-only StageSpec: pose_weight=0 so seg is ISOLATED, seg_weight
    matched, the lever fields set per arm. ``seg_surrogate=None`` => vendored CE."""
    return StageSpec(
        name=f"probe_{seg_surrogate or 'ce'}",
        epochs=1,  # the probe drives steps manually; epochs is just metadata here
        seg_loss_fn=ce_seg_loss,  # the vendored CE — used iff seg_surrogate is None
        eval_every=1,
        batch_size=1,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=adamw_lr,
        muon_lr=0.0,
        muon_weight_decay=0.0,
        latent_lr_mult=1.0,
        grad_clip=grad_clip,
        grad_clip_muon=None,
        lr_floor_ratio=1.0,
        seg_weight=seg_weight,
        pose_weight=0.0,  # ISOLATE seg — no pose term (the apples-to-apples control)
        cat_lambda=0.0,
        cat_sigma=0.0,
        use_qat=False,
        init_latents_random=False,
        seg_surrogate=seg_surrogate,
        seg_temperature=temperature,
        seg_temperature_end=None,  # STATIC T per arm (we sweep T across arms)
        margin_weight_tau=margin_weight_tau,
    )


def _run_arm(
    *,
    arm_label: str,
    seg_surrogate: str | None,
    temperature: float,
    margin_weight_tau: float | None,
    base_decoder_state: dict,
    base_latents: torch.Tensor,
    scorer: RealScorerContext,
    decoder_factory,
    idx: torch.Tensor,
    n_steps: int,
    seg_weight: float,
    adamw_lr: float,
    grad_clip: float,
    ce_seg_loss,
    conf_thresh: float,
) -> _ArmResult:
    """Run ONE arm from the SHARED forkpoint init (deepcopy) for ``n_steps`` seg-only
    optimization steps on the SAME slice ``idx``. Returns the d_seg + flip decomposition
    before/after. EVERY arm gets a fresh deepcopy of the identical decoder/latent init,
    a fresh AdamW with the same LR — the seg-loss function is the ONLY variable."""
    t0 = time.time()
    # Bit-identical init per arm (deepcopy the shared forkpoint state).
    decoder = decoder_factory()
    decoder.load_state_dict(copy.deepcopy(base_decoder_state))
    decoder.train()
    latents = base_latents.clone().detach().requires_grad_(True)

    spec = _make_spec(
        seg_surrogate,
        temperature,
        margin_weight_tau,
        seg_weight=seg_weight,
        adamw_lr=adamw_lr,
        grad_clip=grad_clip,
        ce_seg_loss=ce_seg_loss,
    )

    before = _measure_flips(decoder, latents, idx, scorer, conf_thresh=conf_thresh)

    # Same optimizer family/LR for every arm (AdamW over decoder params + latents).
    opt = torch.optim.AdamW(
        [{"params": decoder.parameters()}, {"params": [latents]}], lr=adamw_lr
    )

    grad_norm_first = 0.0
    loss_first = 0.0
    loss_last = 0.0
    for step in range(n_steps):
        decoder.train()
        opt.zero_grad()
        decoded_pair = decoder(latents[idx])
        decoded_bhwc = _roundtrip(decoded_pair)
        seg_out = scorer.seg_forward_train(decoded_bhwc)  # SegNet-only (pose isolated)
        # Route through the EXACT driver loss router (read-only) — CE iff
        # seg_surrogate is None, else soft-cosine@T ±Lever-5. seg_weight matched.
        seg_l = _seg_loss_for_spec(
            spec, seg_out, scorer.seg_targets_hard[idx], temperature=temperature
        )
        loss = spec.seg_weight * seg_l  # NO pose term — seg isolated
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [*decoder.parameters(), latents], spec.grad_clip
        )
        if step == 0:
            grad_norm_first = float(gn)
            loss_first = float(loss.item())
        loss_last = float(loss.item())
        opt.step()

    after = _measure_flips(decoder, latents, idx, scorer, conf_thresh=conf_thresh)

    return _ArmResult(
        arm=arm_label,
        seg_surrogate=seg_surrogate,
        temperature=temperature,
        margin_weight_tau=margin_weight_tau,
        d_seg_before=before.d_seg,
        d_seg_after=after.d_seg,
        d_seg_reduction=before.d_seg - after.d_seg,
        flips_before=before.n_flips,
        flips_after=after.n_flips,
        flips_fixed=before.n_flips - after.n_flips,
        conf_wrong_before=before.n_flips_confident_wrong,
        conf_wrong_after=after.n_flips_confident_wrong,
        conf_wrong_fixed=before.n_flips_confident_wrong - after.n_flips_confident_wrong,
        grad_norm_first_step=grad_norm_first,
        loss_first=loss_first,
        loss_last=loss_last,
        seconds=time.time() - t0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=12, help="pair-slice size (small for contention)")
    ap.add_argument("--n-steps", type=int, default=20, help="seg-only optimization steps per arm")
    ap.add_argument("--seg-weight", type=float, default=100.0, help="matched seg Lagrangian weight")
    ap.add_argument("--adamw-lr", type=float, default=1e-3, help="matched LR for every arm")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--conf-thresh", type=float, default=0.5, help="decoded p(argmax)>=t = confidently wrong")
    ap.add_argument(
        "--slices",
        type=str,
        default="0,1",
        help="comma list of slice-START indices (each takes n-pairs from there) — robustness across >=2 slices",
    )
    ap.add_argument(
        "--temps",
        type=str,
        default="1.0,0.5,0.3",
        help=(
            "soft-cosine temperatures to sweep — the GRADIENT-ALIVE range only. "
            "Per A2 R12 (commit ab3969ebb) the combined Lever-5xLever-2 gradient at "
            "T=0.1 is ~10 orders below warm = DEAD (the cold-T floor was corrected "
            "0.1->0.3); testing T<=0.1 produces a no-op arm that loses to CE for the "
            "WRONG reason (a dead gradient, not a worse flip-fixer). So sweep only "
            "T in {1.0, 0.5, 0.3}: even in its alive range, does soft-cosine beat CE?"
        ),
    )
    ap.add_argument(
        "--init-latents",
        type=str,
        default="forkpoint,random",
        help=(
            "comma list of latent-init regimes. 'forkpoint' = the EMA-shadow latents "
            "(d_seg~0.003, the CONVERGED basin the from-0 run ends at — tests 'at the "
            "floor, which loss holds best / damages least'). 'random' = seeded-random "
            "latents on the EMA decoder (d_seg~0.043, ABUNDANT flips — tests genuine "
            "REPAIR, the dominant dynamic of the from-0 run driving d_seg DOWN; this is "
            "the more discriminating regime because there are real flips to fix)."
        ),
    )
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)  # SegNet has non-deterministic interp; eval is no_grad

    if not _FORKPOINT.exists():
        raise SystemExit(f"forkpoint not found: {_FORKPOINT}")
    if not _VIDEO.exists():
        raise SystemExit(f"video not found: {_VIDEO}")

    print(f"[probe] loading forkpoint {_FORKPOINT}", flush=True)
    ckpt = torch.load(_FORKPOINT, map_location="cpu", weights_only=False)
    base_channels = 20
    latent_dim = 28
    # The INFERENCE state is the EMA shadow (the EMA non-negotiable: inference/export
    # bytes are the shadow). The from-0 run scores the shadow — so the A/B trains FROM
    # the shadow (the actual state d_seg is measured on).
    base_decoder_state = ckpt["ema_decoder"]
    ema_latents = ckpt["ema_latents"].clone().detach()
    n_pairs_ckpt = ema_latents.shape[0]
    print(f"[probe] base_ch={base_channels} latent_dim={latent_dim} n_pairs_ckpt={n_pairs_ckpt}", flush=True)

    max_slice_end = max(int(s) for s in args.slices.split(",")) + args.n_pairs
    max_pairs_needed = min(n_pairs_ckpt, max_slice_end)

    print(f"[probe] building RealScorerContext (real frozen SegNet/PoseNet, CPU authority, max_pairs={max_pairs_needed})", flush=True)
    scorer = RealScorerContext(
        video_path=_VIDEO,
        device="cpu",  # CPU AUTHORITY — NO MPS
        max_pairs=max_pairs_needed,
        targets_cache=_TARGETS_CACHE,
    )
    print(f"[probe] scorer ready: n_pairs={scorer.n_pairs}", flush=True)

    vbundle = import_vendored_bundle()

    def decoder_factory():
        return vbundle.HNeRVDecoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            eval_size=(_EVAL_H, _EVAL_W),
        ).to("cpu")

    ce_seg_loss = import_vendored("stages.stage1_v328_ce").ce_seg_loss

    temps = [float(t) for t in args.temps.split(",")]
    slice_starts = [int(s) for s in args.slices.split(",")]

    # The arm matrix: CE (the strong flip-fixer) + soft_cosine@T ±Lever-5.
    # margin_weight_tau value mirrors a plausible Lever-5 setting; the point is the
    # COMPARISON ce vs soft-cosine, and soft-cosine ±margin.
    arm_specs: list[tuple[str, str | None, float, float | None]] = []
    arm_specs.append(("CE", None, 1.0, None))
    for t in temps:
        arm_specs.append((f"soft_cosine_T{t:g}", "soft_cosine", t, None))
        arm_specs.append((f"soft_cosine_T{t:g}_L5", "soft_cosine", t, 1.0))

    regimes = [r.strip() for r in args.init_latents.split(",") if r.strip()]

    def _base_latents_for(regime: str) -> torch.Tensor:
        if regime == "forkpoint":
            return ema_latents.clone().detach()
        if regime == "random":
            # Seeded-random latents on the EMA decoder => a high-flip starting state
            # (d_seg~0.043). Deterministic per --seed so the A/B init is bit-identical
            # across arms (every arm clones THIS tensor).
            g = torch.Generator(device="cpu").manual_seed(args.seed)
            return torch.randn(n_pairs_ckpt, latent_dim, generator=g)
        raise SystemExit(f"unknown --init-latents regime: {regime!r}")

    all_results: dict[str, list[dict]] = {}
    for regime in regimes:
        base_latents = _base_latents_for(regime)
        print(f"\n[probe] ##### REGIME init-latents={regime} #####", flush=True)
        for s_start in slice_starts:
            if s_start + args.n_pairs > scorer.n_pairs:
                print(f"[probe] SKIP slice {s_start} (would exceed n_pairs={scorer.n_pairs})", flush=True)
                continue
            idx = torch.arange(s_start, s_start + args.n_pairs, device="cpu")
            print(f"\n[probe] === REGIME {regime} SLICE start={s_start} n={args.n_pairs} ===", flush=True)
            slice_results: list[dict] = []
            for arm_label, surrogate, temp, mtau in arm_specs:
                res = _run_arm(
                    arm_label=arm_label,
                    seg_surrogate=surrogate,
                    temperature=temp,
                    margin_weight_tau=mtau,
                    base_decoder_state=base_decoder_state,
                    base_latents=base_latents,
                    scorer=scorer,
                    decoder_factory=decoder_factory,
                    idx=idx,
                    n_steps=args.n_steps,
                    seg_weight=args.seg_weight,
                    adamw_lr=args.adamw_lr,
                    grad_clip=args.grad_clip,
                    ce_seg_loss=ce_seg_loss,
                    conf_thresh=args.conf_thresh,
                )
                row = dict(res.__dict__)
                row["regime"] = regime
                slice_results.append(row)
                print(
                    f"  {arm_label:24s} d_seg {res.d_seg_before:.5f}->{res.d_seg_after:.5f} "
                    f"(Δ={res.d_seg_reduction:+.5f}) flips_fixed={res.flips_fixed:+d} "
                    f"conf_wrong_fixed={res.conf_wrong_fixed:+d} "
                    f"gn0={res.grad_norm_first_step:.3e} loss {res.loss_first:.4f}->{res.loss_last:.4f} "
                    f"[{res.seconds:.1f}s]",
                    flush=True,
                )
            all_results[f"{regime}__slice_{s_start}"] = slice_results

    # --- Verdict synthesis (PER REGIME) ------------------------------------
    # The two regimes are DIFFERENT operating points and must be judged separately:
    #   * random (d_seg~0.043): genuine REPAIR — a good seg loss DRIVES d_seg DOWN
    #     (Δd_seg > 0). This is the PRIMARY discriminator: it is closest to the
    #     from-0 run's dynamic (drive d_seg from ~0.5 toward the floor) and there
    #     are abundant flips to actually fix. Higher Δd_seg = better flip-fixer.
    #     ``conf_wrong_fixed`` (confidently-wrong flips removed) is the mechanism
    #     stat the hypothesis predicts CE should win on.
    #   * forkpoint (d_seg~0.003): the CONVERGED basin the from-0 run ENDS at. Near
    #     the floor every step tends to slightly INCREASE d_seg (Δd_seg < 0, the
    #     uint8-roundtrip noise floor); here the question is "which loss HOLDS the
    #     basin best / damages least" — still "higher Δd_seg = better", but a
    #     negative Δ is damage, not repair, and must be labelled as such.
    def _avg(regime: str, arm_label: str, fieldname: str) -> float:
        vals = [
            r[fieldname]
            for key, slr in all_results.items()
            for r in slr
            if r["arm"] == arm_label and r.get("regime") == regime
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    sc_arm_labels = [a[0] for a in arm_specs if a[0] != "CE"]
    per_regime_verdict: dict[str, dict] = {}
    for regime in regimes:
        ce_red = _avg(regime, "CE", "d_seg_reduction")
        ce_cwf = _avg(regime, "CE", "conf_wrong_fixed")
        is_repair = ce_red > 0  # CE drives d_seg DOWN => this regime actually repairs
        print(
            f"\n[probe] ===== VERDICT (regime={regime}; "
            f"{'REPAIR — Δ>0 fixes flips' if is_repair else 'BASIN/floor — Δ<0 = damage; higher=holds better'}) =====",
            flush=True,
        )
        print(f"  CE                       Δd_seg = {ce_red:+.6f}  conf_wrong_fixed={ce_cwf:+.1f}", flush=True)
        rows = []
        for arm_label in sc_arm_labels:
            sc_red = _avg(regime, arm_label, "d_seg_reduction")
            sc_cwf = _avg(regime, arm_label, "conf_wrong_fixed")
            beats_ce = sc_red >= ce_red  # higher d_seg reduction is always better
            rows.append(
                {
                    "arm": arm_label,
                    "d_seg_reduction": sc_red,
                    "conf_wrong_fixed": sc_cwf,
                    "beats_ce": beats_ce,
                }
            )
            print(
                f"  {arm_label:24s} Δd_seg = {sc_red:+.6f}  conf_wrong_fixed={sc_cwf:+.1f}  "
                f"{'>= CE (soft-cosine wins/ties)' if beats_ce else '< CE (CE wins — DOWNGRADE signal)'}",
                flush=True,
            )
        best = max(rows, key=lambda r: r["d_seg_reduction"]) if rows else None
        any_beats = any(r["beats_ce"] for r in rows)
        all_beat = all(r["beats_ce"] for r in rows) if rows else False
        if best is None:
            r_verdict = "INCONCLUSIVE_NO_ARMS"
        elif all_beat:
            r_verdict = "LEVER-2-NET-POSITIVE"
        elif not any_beats:
            r_verdict = "LEVER-2-DOWNGRADE"
        else:
            r_verdict = "MIXED"
        per_regime_verdict[regime] = {
            "is_repair": is_repair,
            "ce_d_seg_reduction": ce_red,
            "ce_conf_wrong_fixed": ce_cwf,
            "arms": rows,
            "best_soft_cosine_arm": best["arm"] if best else None,
            "best_soft_cosine_d_seg_reduction": best["d_seg_reduction"] if best else None,
            "verdict": r_verdict,
        }
        print(f"  [regime {regime}] VERDICT: {r_verdict}", flush=True)

    # --- Overall gate verdict ----------------------------------------------
    # The PRIMARY discriminator is the REPAIR regime (random init) — it is the
    # regime that actually has flips to fix and is closest to the from-0 dynamic.
    # If no regime repairs (all Δ<=0 even for CE), fall back to the regime whose CE
    # Δ is largest (least damage) as the discriminator. The basin regime is reported
    # but is a tie-breaker, not the lead.
    repair_regimes = [r for r in regimes if per_regime_verdict[r]["is_repair"]]
    primary_regime = (
        repair_regimes[0]
        if repair_regimes
        else max(regimes, key=lambda r: per_regime_verdict[r]["ce_d_seg_reduction"])
    )
    verdict = per_regime_verdict[primary_regime]["verdict"]
    print(
        f"\n[probe] PRIMARY discriminator regime = {primary_regime} "
        f"({'REPAIR' if per_regime_verdict[primary_regime]['is_repair'] else 'no-repair fallback'})",
        flush=True,
    )
    print(f"[probe] GATE VERDICT: {verdict}", flush=True)
    for regime in regimes:
        pv = per_regime_verdict[regime]
        print(
            f"  regime {regime:10s}: {pv['verdict']:22s} "
            f"CE Δd_seg={pv['ce_d_seg_reduction']:+.6f} "
            f"best_sc={pv['best_soft_cosine_arm']} "
            f"(Δ={pv['best_soft_cosine_d_seg_reduction']:+.6f})",
            flush=True,
        )

    out = {
        "probe": "lever2_softcosine_vs_ce_flipfix",
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE",
        "forkpoint": str(_FORKPOINT),
        "config": vars(args),
        "primary_regime": primary_regime,
        "gate_verdict": verdict,
        "per_regime_verdict": per_regime_verdict,
        "results": all_results,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"[probe] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
