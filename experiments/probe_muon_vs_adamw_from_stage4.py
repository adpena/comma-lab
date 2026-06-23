# SPDX-License-Identifier: MIT
"""MUON vs AdamW from the stage-4 fork — the decisive optimizer-axis d_seg test (2026-06-22).

THE QUESTION (the optimizer-axis disambiguator). The live decisive run
(``yousfi_r3_taper_marginhinge_e5_20260620``) shows d_seg PARKED at ~0.00207 across
stages 2-5 (all AdamW), after stage 1 (CE) did ~86% of the d_seg work. Two competing
explanations:

  * CONDITIONING thesis: the flatness is the PREDICTED AdamW behavior — a diagonal
    preconditioner cannot decorrelate the kappa~19 boundary Hessian — and the SPECTRAL
    Muon optimizer (stage 8) is the reserved d_seg finisher. If true, stages 5-7 are NOT
    necessary d_seg-prep and a jump-to-Muon-early is viable.
  * CAPACITY thesis: d_seg ~0.00207 is near the small decoder's CAPACITY floor and stages
    5-7 are over-long rate-tuning cargo-culted from PR95's bigger model. If true, Muon
    will ALSO stall flat.

DECISIVE LOCAL TEST. From the SAME fork point (the preserved ``stage4_v332_qat_end``
snapshot — QAT-on, Muon's hard prerequisite satisfied), run TWO short convergence arms
from a DEEP COPY of the same stage-4 weights and compare the d_seg(step) trajectories:

  * Arm A (Muon): ``MuonOptimizer`` on the trunk decoder params (the
    ``partition_params_for_muon`` Muon group) + AdamW on the excluded params (stem / rgb
    heads / biases) + AdamW on the latents — MIRRORING the driver's stage-8 construction
    (``driver.py`` ~1784: ``muon_params, adamw_params = partition_params_for_muon(decoder)``;
    FiLM is OFF for this run so there is NO FiLM-exclusion, matching the driver's no-FiLM
    branch exactly). This is what stage 8 does.
  * Arm B (AdamW control): AdamW on ALL decoder params + latents. This is what stages 5-7
    effectively do for d_seg.

KEY SIGNAL. Does Arm A (Muon) show a COHERENT downward d_seg trajectory that DIVERGES
BELOW Arm B (AdamW)? If Muon accumulates d_seg progress where AdamW stalls flat (matching
the live run's stall) -> conditioning confirmed -> stages 5-7 not needed for d_seg ->
jump-early viable. If both stay flat -> AMBIGUOUS (capacity OR stage4-not-yet-Muon-ready),
defer to the live run's real stage 8 (~2 days out). NEVER a kill.

APPLES-TO-APPLES (the optimizer is the ONLY variable):
  * SAME fork-point weights (deepcopy per arm), SAME 24 pairs, SAME seg loss
    (``specs[4].seg_loss_fn`` = the vendored l7_softplus tau=0.3 / l7_threshold=1.0 /
    l7_mult=4.0 — VERIFIED byte-identical to stage 8's seg_loss_fn), SAME LR per config,
    SAME grad-clip, SAME int8 fake-quant in the deployed/eval surface, SAME eval roundtrip.
  * The ONLY thing that differs is which optimizer drives the decoder weights.

DISCIPLINE (CLAUDE.md):
  * CPU ONLY (``device='cpu'``). MPS is FORBIDDEN — it is never a d_seg authority (2x
    SegNet drift) AND the live run owns the MPS device; we must not contend. CPU SegNet IS
    the d_seg authority. ``OMP_NUM_THREADS``/``MKL_NUM_THREADS`` capped at 2 by the launcher.
  * READ-ONLY on the live run: only the preserved ``stage4_v332_qat_end`` snapshot is read;
    weights are deep-copied into fresh decoders; all output goes to a separate dir. The
    live run (pid 79893) is never touched.
  * Every number is ``[contest-CPU advisory]`` NON-PROMOTABLE; pointer UNMOVED 0.19110; NO
    score claim; NO kill. ``research_only=true``.

REUSE (no scorer / decoder / quant / roundtrip / Muon / partition reimplemented):
  * ``RealScorerContext`` — frozen CPU SegNet + cached n-pair GT argmax targets.
  * ``ConfigurableTaperHNeRVDecoder`` — the run's decoder (bc20, taper [16,16,17,19,19,14,10],
    latent_dim 28).
  * ``tac.optimization.muon.MuonOptimizer`` + ``partition_params_for_muon`` — the EXACT
    stage-8 optimizer + partition.
  * ``build_curriculum()[4].seg_loss_fn`` — the vendored l7_softplus stage 8 uses.
  * ``tac.torch_vehicle.score_aware_qat._fake_quantize_n`` — the stage-4 int8 (127-level)
    per-tensor symmetric fake-quant.
  * the ``_roundtrip`` math (bicubic^874 -> bilinear v384 -> uint8-STE) is the 1:1 port from
    the sister probe (``probe_margin_hinge_vs_l7_softplus_grad_geometry``) so this probe is
    self-contained.

USAGE:
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
      experiments/probe_muon_vs_adamw_from_stage4.py \
      --n-pairs 24 --n-steps 300 --eval-every 50 \
      --out-json experiments/results/muon_vs_adamw_stage4_20260622/result.json
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

from tac.optimization.muon import MuonOptimizer, partition_params_for_muon
from tac.torch_vehicle.curriculum import build_curriculum
from tac.torch_vehicle.score_aware_qat import _fake_quantize_n
from tac.torch_vehicle.scorer_context import RealScorerContext

_EVAL_H, _EVAL_W = 384, 512
_RUN_DIR = Path("experiments/results/yousfi_r3_taper_marginhinge_e5_20260620")
_SNAP = _RUN_DIR / "stage_snapshots" / "stage4_v332_qat_end"
_VIDEO = Path("upstream/videos/0.mkv")
_TARGETS_CACHE = Path("experiments/results/capstone_gt_targets_cache")
# Run config (manifest): base_channels=20, latent_dim=28, taper as below.
_BASE_CH = 20
_LATENT_DIM = 28
_TAPER = [16, 16, 17, 19, 19, 14, 10]
_QAT_N_LEVELS = 127  # the vendored stage-4 int8 QAT level count
_DEFAULT_RESULTS_DIR = Path("experiments/results/muon_vs_adamw_stage4_20260622")

# The faithful stage-8 Muon hyperparameters (read off the curriculum; VERIFIED):
#   muon_lr=2e-4, adamw_lr=1e-5, muon_weight_decay=5e-4, momentum=0.95, nesterov, ns_steps=5.
_STAGE8_MOMENTUM = 0.95
_STAGE8_NESTEROV = True
_STAGE8_NS_STEPS = 5


def _roundtrip(decoded_pair: torch.Tensor) -> torch.Tensor:
    """1:1 port of the driver per-step frame math: bicubic^874 -> bilinear v384 -> uint8-STE.

    ``decoded_pair`` is (B, 2, 3, 384, 512) float [0,255]; returns (B, 2, 384, 512, 3)
    (BHWC, the scorer's ``preprocess_input`` expects the channel-last pair frames)."""
    B = decoded_pair.shape[0]
    flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    decoded_clamped = decoded_bhwc.clamp(0, 255)
    decoded_rounded = decoded_clamped.round()
    return decoded_clamped + (decoded_rounded - decoded_clamped).detach()


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
    which quantizes weights only). Mutates ``.data`` (caller works on a copy)."""
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


def _eval_both(decoder: torch.nn.Module, latents: torch.Tensor, scorer, n_levels: int):
    """Measure d_seg on the FP (live) weights AND on the int8 fake-quant (deployed/authority)
    surface — so quant-erosion is tracked along the whole descent, not just at one step."""
    d_fp = _measure_dseg(decoder, latents, scorer)
    dq = _build_decoder()
    dq.load_state_dict(decoder.state_dict())
    _quantize_decoder_inplace(dq, n_levels)
    d_q = _measure_dseg(dq, latents, scorer)
    return d_fp, d_q


def _seg_loss(seg_loss_fn, decoder, latents, scorer):
    """One forward + the vendored l7_softplus seg loss (the SAME callable for both arms)."""
    decoded_pair = decoder(latents)
    decoded_bhwc = _roundtrip(decoded_pair)
    seg_out = scorer.seg_forward_train(decoded_bhwc)  # (B,5,Hs,Ws), CPU frozen SegNet
    gt_hard = scorer.seg_targets_hard  # (B,Hs,Ws)
    return seg_loss_fn(seg_out, gt_hard)


def _run_arm(
    *,
    arm: str,
    seg_loss_fn,
    base_state: dict,
    base_latents: torch.Tensor,
    scorer,
    n_steps: int,
    eval_every: int,
    muon_lr: float,
    adamw_lr: float,
    muon_weight_decay: float,
    grad_clip: float,
    n_levels: int,
    max_seconds: float,
) -> dict:
    """Run ONE convergence arm (``muon`` or ``adamw``) from a DEEP COPY of the fork-point
    weights. Build the decoder/optimizer ONCE (no per-step rebuild — the probe-1 slowness
    fix); step in place each step.

    Arm A (``muon``): MuonOptimizer on the partition's Muon group (trunk conv weights) +
    AdamW on the excluded params (stem/rgb/biases) + AdamW on the latents — the EXACT
    stage-8 construction (driver.py ~1784; FiLM off so no FiLM-exclusion).
    Arm B (``adamw``): AdamW on ALL decoder params + latents.

    Returns the d_seg(step) curves (FP + int8-quant), the seg-loss + grad-norm traces, and
    endpoints. ``stopped_early`` is set if the per-arm wall-clock budget is exceeded."""
    decoder = _build_decoder()
    decoder.load_state_dict(copy.deepcopy(base_state))
    decoder.train()
    latents = base_latents.clone().detach().requires_grad_(True)

    if arm == "muon":
        muon_params, adamw_params = partition_params_for_muon(decoder)
        muon_opt: torch.optim.Optimizer | None = MuonOptimizer(
            muon_params,
            lr=muon_lr,
            momentum=_STAGE8_MOMENTUM,
            nesterov=_STAGE8_NESTEROV,
            ns_steps=_STAGE8_NS_STEPS,
            weight_decay=muon_weight_decay,
        )
        adamw_opt = torch.optim.AdamW(
            [
                {"params": adamw_params, "lr": adamw_lr},
                {"params": [latents], "lr": adamw_lr},
            ],
            weight_decay=0.0,
        )
        # The grad-clip set for Arm A = the AdamW-handled decoder params + latents (Muon
        # params are NS-normalized so the driver clips them via a SEPARATE grad_clip_muon;
        # here grad_clip_muon is None at stage 8 by default, so we clip only the AdamW set —
        # matching the driver's clip set ``adamw_clip_params`` + latents).
        clip_params = [*list(adamw_params), latents]
    elif arm == "adamw":
        muon_opt = None
        all_params = list(decoder.parameters())
        adamw_opt = torch.optim.AdamW(
            [
                {"params": all_params, "lr": adamw_lr},
                {"params": [latents], "lr": adamw_lr},
            ],
            weight_decay=0.0,
        )
        clip_params = [*all_params, latents]
    else:
        raise ValueError(arm)

    d0_fp, d0_q = _eval_both(decoder, latents, scorer, n_levels)
    curve_fp = [(0, d0_fp)]
    curve_q = [(0, d0_q)]
    loss_trace: list[tuple[int, float]] = []
    gradnorm_trace: list[tuple[int, float]] = []
    t_start = time.time()
    stopped_early = False
    for step in range(1, n_steps + 1):
        decoder.train()
        adamw_opt.zero_grad(set_to_none=True)
        if muon_opt is not None:
            muon_opt.zero_grad(set_to_none=True)
        seg_l = _seg_loss(seg_loss_fn, decoder, latents, scorer)
        seg_l.backward()
        # grad-norm BEFORE clip (the diagnostic), on the clip set.
        gnorm = float(
            torch.nn.utils.clip_grad_norm_(clip_params, grad_clip).item()
        )
        if muon_opt is not None:
            muon_opt.step()
        adamw_opt.step()
        if step == 1 or step % eval_every == 0 or step == n_steps:
            loss_trace.append((step, float(seg_l.item())))
            gradnorm_trace.append((step, gnorm))
        if step % eval_every == 0 or step == n_steps:
            d_fp, d_q = _eval_both(decoder, latents, scorer, n_levels)
            curve_fp.append((step, d_fp))
            curve_q.append((step, d_q))
            print(
                f"[probe] [{arm}] step {step:4d}  d_seg_fp={d_fp:.6f}  "
                f"d_seg_quant={d_q:.6f}  seg_loss={float(seg_l.item()):.6e}  "
                f"||g||={gnorm:.4e}  ({time.time() - t_start:.0f}s)",
                flush=True,
            )
        if time.time() - t_start > max_seconds:
            stopped_early = True
            print(
                f"[probe] [{arm}] per-arm budget {max_seconds:.0f}s exceeded at step "
                f"{step} -> stopping; reporting the PARTIAL curve (not a failure).",
                flush=True,
            )
            if not curve_fp or curve_fp[-1][0] != step:
                d_fp, d_q = _eval_both(decoder, latents, scorer, n_levels)
                curve_fp.append((step, d_fp))
                curve_q.append((step, d_q))
            break

    return {
        "arm": arm,
        "muon_lr": muon_lr if arm == "muon" else None,
        "adamw_lr": adamw_lr,
        "muon_weight_decay": muon_weight_decay if arm == "muon" else None,
        "d_seg_fp_curve": curve_fp,
        "d_seg_quant_curve": curve_q,
        "seg_loss_trace": loss_trace,
        "gradnorm_trace": gradnorm_trace,
        "d_seg_fp_start": curve_fp[0][1],
        "d_seg_fp_end": curve_fp[-1][1],
        "d_seg_quant_start": curve_q[0][1],
        "d_seg_quant_end": curve_q[-1][1],
        "delta_d_seg_fp_total": curve_fp[-1][1] - curve_fp[0][1],
        "delta_d_seg_quant_total": curve_q[-1][1] - curve_q[0][1],
        "steps_run": curve_fp[-1][0],
        "stopped_early": stopped_early,
        "seconds": time.time() - t_start,
    }


def _run_lr_config(
    *,
    label: str,
    muon_lr: float,
    adamw_lr: float,
    muon_weight_decay: float,
    seg_loss_fn,
    base_state: dict,
    base_latents: torch.Tensor,
    scorer,
    n_steps: int,
    eval_every: int,
    grad_clip: float,
    n_levels: int,
    max_seconds_per_arm: float,
) -> dict:
    """Run Arm A (muon) then Arm B (adamw) at ONE LR config and compute the contrast.

    The contrast metric is on the int8-quant (deployed/authority) d_seg curve:
        gap_quant = Arm_A.delta_d_seg_quant_total - Arm_B.delta_d_seg_quant_total
    gap_quant < 0  ==> Muon descended d_seg MORE (more negative) than AdamW post-quant
                       (conditioning signal). gap_quant > 0 ==> AdamW descended more.
    Also reports the FP gap for cross-check."""
    print(
        f"\n[probe] ===== LR config '{label}': Muon lr={muon_lr:.1e} "
        f"(wd={muon_weight_decay:.1e}) | AdamW lr={adamw_lr:.1e} =====",
        flush=True,
    )
    arm_a = _run_arm(
        arm="muon", seg_loss_fn=seg_loss_fn, base_state=base_state,
        base_latents=base_latents, scorer=scorer, n_steps=n_steps,
        eval_every=eval_every, muon_lr=muon_lr, adamw_lr=adamw_lr,
        muon_weight_decay=muon_weight_decay, grad_clip=grad_clip, n_levels=n_levels,
        max_seconds=max_seconds_per_arm,
    )
    arm_b = _run_arm(
        arm="adamw", seg_loss_fn=seg_loss_fn, base_state=base_state,
        base_latents=base_latents, scorer=scorer, n_steps=n_steps,
        eval_every=eval_every, muon_lr=muon_lr, adamw_lr=adamw_lr,
        muon_weight_decay=muon_weight_decay, grad_clip=grad_clip, n_levels=n_levels,
        max_seconds=max_seconds_per_arm,
    )
    gap_quant = arm_a["delta_d_seg_quant_total"] - arm_b["delta_d_seg_quant_total"]
    gap_fp = arm_a["delta_d_seg_fp_total"] - arm_b["delta_d_seg_fp_total"]
    print(
        f"[probe] LR config '{label}' CONTRAST: "
        f"Arm A (Muon) Δd_seg_quant={arm_a['delta_d_seg_quant_total']:+.6f} | "
        f"Arm B (AdamW) Δd_seg_quant={arm_b['delta_d_seg_quant_total']:+.6f} | "
        f"gap (A−B)={gap_quant:+.6f}  (<0 ⇒ Muon descends more)",
        flush=True,
    )
    return {
        "label": label,
        "muon_lr": muon_lr,
        "adamw_lr": adamw_lr,
        "muon_weight_decay": muon_weight_decay,
        "arm_a_muon": arm_a,
        "arm_b_adamw": arm_b,
        "gap_quant_a_minus_b": gap_quant,
        "gap_fp_a_minus_b": gap_fp,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument(
        "--n-steps", type=int, default=300,
        help="steps per arm (~300 ≈ 1.5-2h total on CPU at N=24).",
    )
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument(
        "--max-minutes-per-arm", type=float, default=90.0,
        help="wall-clock budget per arm; on overrun, stop and report the PARTIAL curve "
        "(NOT a failure).",
    )
    ap.add_argument(
        "--lr-configs", type=str, default="stage8,higher",
        help="comma list of LR configs to run. 'stage8' = the faithful stage-8 LRs "
        "(Muon 2e-4 / AdamW-aux 1e-5 vs AdamW-all 1e-5, the real stage8-vs-stage7 "
        "comparison). 'higher' = a matched higher LR (Muon 3e-3 / AdamW 3e-4) to surface "
        "the contrast faster.",
    )
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)

    if not _SNAP.exists():
        raise SystemExit(f"fork-point snapshot not found: {_SNAP}")
    dec_path = _SNAP / "best_ema_decoder.pt"
    lat_path = _SNAP / "best_ema_latents.pt"

    t0 = time.time()
    print(
        f"[probe] CPU Muon-vs-AdamW from-stage4 convergence probe; n={args.n_pairs} "
        f"n_steps={args.n_steps}",
        flush=True,
    )
    print(f"[probe] fork point: {_SNAP}", flush=True)
    print("[probe] device=cpu (MPS FORBIDDEN: never authority + live run owns it)", flush=True)

    scorer = RealScorerContext(
        _VIDEO,
        device="cpu",
        max_pairs=int(args.n_pairs),
        targets_cache=_TARGETS_CACHE,
    )

    # The vendored l7_softplus seg loss stage 8 uses (specs[4].seg_loss_fn == specs[7]'s,
    # VERIFIED byte-identical). The optimizer is the ONLY variable.
    specs = build_curriculum()
    seg_loss_fn = specs[4].seg_loss_fn

    base_state = torch.load(dec_path, map_location="cpu", weights_only=False)
    if isinstance(base_state, dict) and "state_dict" in base_state:
        base_state = base_state["state_dict"]
    all_latents = torch.load(lat_path, map_location="cpu", weights_only=False)
    base_latents = all_latents[: int(args.n_pairs)].detach().clone()

    # Sanity: a fresh decoder loads the fork state strictly (architecture match) + report
    # the partition sizes (the stage-8 Muon group) for the record.
    probe_dec = _build_decoder()
    probe_dec.load_state_dict(copy.deepcopy(base_state))  # raises on mismatch
    muon_params, adamw_params = partition_params_for_muon(probe_dec)
    print(
        f"[probe] partition (stage-8 mirror): Muon group {len(muon_params)} tensors "
        f"({sum(p.numel() for p in muon_params)} elems) | AdamW-handled "
        f"{len(adamw_params)} tensors ({sum(p.numel() for p in adamw_params)} elems)",
        flush=True,
    )

    # Baseline d_seg at the fork point (fp + int8-quant), for the record.
    base_dec_fp = _build_decoder()
    base_dec_fp.load_state_dict(copy.deepcopy(base_state))
    baseline_fp = _measure_dseg(base_dec_fp, base_latents, scorer)
    base_dec_q = _build_decoder()
    base_dec_q.load_state_dict(copy.deepcopy(base_state))
    _quantize_decoder_inplace(base_dec_q, _QAT_N_LEVELS)
    baseline_quant = _measure_dseg(base_dec_q, base_latents, scorer)
    print(
        f"[probe] baseline d_seg @ fork: fp={baseline_fp:.6f}  int8quant={baseline_quant:.6f}",
        flush=True,
    )

    # LR config table (label -> (muon_lr, adamw_lr, muon_wd)).
    _LR_TABLE = {
        "stage8": (2e-4, 1e-5, 5e-4),   # the faithful stage-8 LRs (the real comparison)
        "higher": (3e-3, 3e-4, 5e-4),   # matched higher LR to surface contrast faster
    }
    wanted = [x.strip() for x in args.lr_configs.split(",") if x.strip()]
    max_seconds_per_arm = float(args.max_minutes_per_arm) * 60.0

    configs: list[dict] = []
    for label in wanted:
        if label not in _LR_TABLE:
            raise SystemExit(f"unknown lr-config '{label}' (known: {list(_LR_TABLE)})")
        m_lr, a_lr, m_wd = _LR_TABLE[label]
        cfg = _run_lr_config(
            label=label, muon_lr=m_lr, adamw_lr=a_lr, muon_weight_decay=m_wd,
            seg_loss_fn=seg_loss_fn, base_state=base_state, base_latents=base_latents,
            scorer=scorer, n_steps=int(args.n_steps), eval_every=int(args.eval_every),
            grad_clip=float(args.grad_clip), n_levels=_QAT_N_LEVELS,
            max_seconds_per_arm=max_seconds_per_arm,
        )
        configs.append(cfg)
        # incremental write so a crash leaves the completed configs on disk.
        if args.out_json:
            _write_partial(args.out_json, args, baseline_fp, baseline_quant, configs, t0)

    # ---- HONEST verdict (NO kill / NO capacity-conclusion) ----
    # The discriminating signal is whether Arm A (Muon) descends d_seg (post-quant) below
    # Arm B (AdamW) by a non-noise margin in ANY config. ~10 pixels of d_seg over the n
    # pairs is the "real, not noise" band.
    noise_band = 1.0 / float(scorer.seg_targets_hard.numel())  # ~1 pixel of d_seg
    band = max(10.0 * noise_band, 5e-5)

    # Did EITHER arm move at all (post-quant) beyond noise? (capacity-vs-prep gate)
    def arm_moved(cfg: dict, arm_key: str) -> bool:
        return abs(cfg[arm_key]["delta_d_seg_quant_total"]) > band

    any_muon_bites = any(
        cfg["gap_quant_a_minus_b"] < -band for cfg in configs
    )
    any_arm_moved = any(
        arm_moved(cfg, "arm_a_muon") or arm_moved(cfg, "arm_b_adamw") for cfg in configs
    )
    best_gap = min((cfg["gap_quant_a_minus_b"] for cfg in configs), default=0.0)
    contrast_within_noise = all(
        abs(cfg["gap_quant_a_minus_b"]) <= band for cfg in configs
    )

    if any_muon_bites:
        verdict = "MUON_BITES_FROM_STAGE4"
        verdict_note = (
            "Arm A (Muon) descended d_seg (post-quant) clearly BELOW Arm B (AdamW) in at "
            "least one config -> conditioning confirmed; stages 5-7 are NOT necessary "
            "d_seg-prep; jump-to-Muon-early is viable. NOT a kill; the live stage 8 remains "
            "the decisive arbiter."
        )
    elif (not any_arm_moved) or contrast_within_noise:
        verdict = "MUON_FLAT_AMBIGUOUS"
        verdict_note = (
            "Both arms flat / contrast within noise -> AMBIGUOUS: capacity floor OR "
            "stage4-not-yet-Muon-ready. This does NOT prove capacity (a flat Muon arm from "
            "the stage-4 fork could be a fork confound, not Muon impotence). Defer to the "
            "live run's real stage 8 (~2 days out). NOT a capacity conclusion; NOT a kill."
        )
    else:
        verdict = "INCONCLUSIVE_NEEDS_MORE"
        verdict_note = (
            "An arm moved but the Muon-vs-AdamW contrast did not clearly favor Muon at "
            f"{args.n_steps} steps -> recommend more steps (CPU is the limiter). NOT a kill."
        )

    result = _result_dict(args, baseline_fp, baseline_quant, configs, t0)
    result["verdict"] = verdict
    result["verdict_note"] = verdict_note
    result["best_gap_quant_a_minus_b"] = best_gap
    result["noise_band_one_pixel"] = noise_band
    result["discrimination_band"] = band
    print(f"\n[probe] VERDICT: {verdict}", flush=True)
    print(f"[probe] {verdict_note}", flush=True)
    print(f"[probe] best gap (A−B) post-quant = {best_gap:+.6f}  (band ±{band:.6f})", flush=True)
    print(f"[probe] elapsed {result['seconds']:.1f}s", flush=True)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"[probe] wrote {out}", flush=True)


def _result_dict(args, baseline_fp, baseline_quant, configs, t0) -> dict:
    return {
        "schema": "muon_vs_adamw_from_stage4_v1",
        "utc": datetime.now(UTC).isoformat(),
        "authority": "contest-CPU advisory (NON-PROMOTABLE)",
        "research_only": True,
        "score_claim": False,
        "promotable": False,
        "device": "cpu",
        "fork_point": str(_SNAP),
        "n_pairs": int(args.n_pairs),
        "n_steps": int(args.n_steps),
        "eval_every": int(args.eval_every),
        "grad_clip": float(args.grad_clip),
        "qat_n_levels": _QAT_N_LEVELS,
        "seg_loss": "vendored l7_softplus (tau=0.3, l7_threshold=1.0, l7_mult=4.0); "
        "specs[4].seg_loss_fn == specs[7] (stage-8) seg_loss_fn (VERIFIED byte-identical)",
        "baseline_d_seg_fp": baseline_fp,
        "baseline_d_seg_quant": baseline_quant,
        "lr_configs": configs,
        "decisive_arbiter": "the live run's real stage-8 Muon d_seg slope (~2 days out); "
        "this probe only disambiguates the optimizer-axis candidate and NEVER kills.",
        "seconds": time.time() - t0,
    }


def _write_partial(out_json, args, baseline_fp, baseline_quant, configs, t0) -> None:
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = _result_dict(args, baseline_fp, baseline_quant, configs, t0)
    partial["verdict"] = "PARTIAL_IN_PROGRESS"
    out.write_text(json.dumps(partial, indent=2))


if __name__ == "__main__":
    main()
