#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lever-C: train the small per-pair-latent CONV frame1 decoder JOINTLY seg+pose (task #62 / #63).

THE ORIGINAL METHODOLOGY (no leaderboard entry uses this combination): a fresh-init small conv
decoder (Conv+PixelShuffle+bilinear-skip+sin, PR95 L18) generates frame1, trained DIRECTLY against
the FROZEN scorers with three original terms:
  1. null-space-primary recon: frame1 recon-MSE WEIGHTED by the SegNet margin free-budget (#52
     margin_polytope) — error is steered INTO the seg-null subspace (large-margin interior pixels are
     cheap to be wrong; small-margin boundary pixels are protected).
  2. Jacobian-aimed pose: frame0 (and frame1) recon-MSE WEIGHTED by the MEASURED PoseNet pixel-Jacobian
     (#61 posenet_jacobian_saliency) + the EXACT 6-dim PoseNet pose-MSE objective in the loop.
  3. seg term — ONE of THREE selectable d_seg losses (task #63 decisive test, ``--seg-loss``):
       * ``argmax_ce``      — boundary-weighted cross-entropy against the GT SegNet argmax labels (the
                              #62 baseline; the gradient touches the SegNet only via the GT-class
                              log-prob).
       * ``kl_distill_t2``  — PR95's actual trick: ``T^2 * KL(log_softmax(student/T) || softmax(
                              teacher/T))`` at T=2.0, teacher = frozen SegNet logits on GT frame1. The
                              gradient flows through the FULL soft 5-class distribution.
       * ``margin_hinge``   — the boundary-solver (#52/#55) gradient as a differentiable term:
                              ``max(0, gamma - (logit[GT_class] - max_{c!=GT} logit[c]))``, pushing the
                              student argmax margin past the flip point via the REAL SegNet input-
                              Jacobian (autograd through the frozen SegNet).
     The recon + pose terms are IDENTICAL across all three arms — only the seg term varies (the #63
     decisive test). The exact argmax-flip d_seg is RE-MEASURED on the frozen SegNet for every arm.

eval_roundtrip (uint8 STE) + differentiable rgb_to_yuv6 in the inner loop (CLAUDE.md non-negotiable).
EMA shadow is the inference checkpoint. The numpy-portable conv forward (``decoder_frame``) is verified
to reproduce the torch forward (parity gate); d_seg/d_pose RE-MEASURED on the exact frozen CPU scorer
with the numpy-decoded frame. GT via ``frame_utils.yuv420_to_rgb`` ONLY.

Authority ``[macOS-MLX research-signal]`` (decoder forward) + ``[local CPU-torch advisory]`` (exact
scorer). NO MPS. $0. Non-promotable. No paid dispatch from this smoke.

NO-FAKE (class 2 + class 8): the d_seg/d_pose are the EXACT frozen-scorer measurements on the decoded
frames (not a proxy); the byte cost is the brotli of the quantized weights+latents; a constant-frame
decoder would NOT reduce d_seg or d_pose (the smoke reports a constant-frame control row). Real
training (internal-consistency: elapsed >= epochs * MIN_SEC_PER_EPOCH).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = REPO_ROOT / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis"
for p in (REPO_ROOT, REPO_ROOT / "src", _HARNESS, REPO_ROOT / "upstream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tac.boundary_math.conv_pair_decoder import (  # noqa: E402
    ConvDecoderConfig,
    decoder_frame,
    decoder_param_count,
    measure_decoder_bytes,
    save_decoder_npz,
)
from tac.boundary_math.posenet_jacobian_saliency import (  # noqa: E402
    compute_posenet_pixel_saliency,
    saliency_to_weight_map,
)
# Canonical EMA WITH warmup (Catalog #388 / commit f771e6e00 ported to torch).
# The prior inline EMA used a CONSTANT decay with no warmup; on this 8-pair /
# ~120-epoch SHORT run the shadow FROZE near init, so exact d_seg read the
# near-init constant-frame floor 0.507 ("moved by zero") — the EMA-shadow-LAG
# artifact (negative_results_resurrection_ledger_20260611.md R1). The canonical
# EMA warms up min(decay,(1+t)/(10+t)) so the shadow tracks live from step 1.
from tac.training import EMA  # noqa: E402

CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512
_CONTEST_TOTAL_BYTES = 37_545_489
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
DEVICE = torch.device("cpu")  # NO MPS, NO cuda for the local advisory loop.
MIN_SEC_PER_EPOCH = 0.02  # internal-consistency floor (refuse a stub loop)


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# Torch conv decoder (mirror of the numpy ConvPairDecoder math — single arch).
# ---------------------------------------------------------------------------
class TorchConvPairDecoder(nn.Module):
    """Per-pair-latent conv decoder: latent -> seed -> N (Conv+PixelShuffle+bilinear-skip+sin) -> RGB."""

    def __init__(self, cfg: ConvDecoderConfig):
        super().__init__()
        self.cfg = cfg
        seed_out = cfg.seed_ch * cfg.seed_h * cfg.seed_w
        self.seed = nn.Linear(cfg.latent_dim, seed_out)
        self.stages = nn.ModuleList()
        self.skips = nn.ModuleList()
        in_ch = cfg.seed_ch
        for out_ch in cfg.stage_channels:
            self.stages.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(nn.Conv2d(in_ch, out_ch, 1))
            in_ch = out_ch
        self.out = nn.Conv2d(in_ch, cfg.n_channels, 3, padding=1)
        self.latents = nn.Parameter(torch.zeros(cfg.num_pairs, cfg.latent_dim))
        nn.init.normal_(self.latents, std=0.1)

    def forward(self, pair_idx: int) -> torch.Tensor:
        """Returns (n_channels, final_h, final_w) RGB in [0,255] at the block-stack resolution."""

        z = self.latents[pair_idx]
        h = self.seed(z).reshape(1, self.cfg.seed_ch, self.cfg.seed_h, self.cfg.seed_w)
        for stage, skip in zip(self.stages, self.skips, strict=True):
            conv = stage(h)
            up = F.pixel_shuffle(conv, 2)
            skip_in = F.interpolate(h, size=up.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.sin(up + skip(skip_in))
        rgb01 = torch.sigmoid(self.out(h))
        return (rgb01 * 255.0)[0]  # (n_channels, fh, fw)

    def numpy_params(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        weights: dict[str, np.ndarray] = {}
        weights["seed.weight"] = self.seed.weight.detach().cpu().numpy().astype(np.float32)
        weights["seed.bias"] = self.seed.bias.detach().cpu().numpy().astype(np.float32)
        for i, (stage, skip) in enumerate(zip(self.stages, self.skips, strict=True)):
            weights[f"stage{i}.weight"] = stage.weight.detach().cpu().numpy().astype(np.float32)
            weights[f"stage{i}.bias"] = stage.bias.detach().cpu().numpy().astype(np.float32)
            weights[f"stage{i}.skip"] = skip.weight.detach().cpu().numpy().astype(np.float32)
            weights[f"stage{i}.skip_bias"] = skip.bias.detach().cpu().numpy().astype(np.float32)
        weights["out.weight"] = self.out.weight.detach().cpu().numpy().astype(np.float32)
        weights["out.bias"] = self.out.bias.detach().cpu().numpy().astype(np.float32)
        latents = self.latents.detach().cpu().numpy().astype(np.float32)
        return weights, latents


# ---------------------------------------------------------------------------
# Scorer loading (exact frozen CPU PoseNet + SegNet).
# ---------------------------------------------------------------------------
def _load_posenet():
    from modules import PoseNet, posenet_sd_path
    from safetensors.torch import load_file

    net = PoseNet().eval().to(DEVICE)
    net.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _load_segnet():
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file

    net = SegNet().eval().to(DEVICE)
    net.load_state_dict(load_file(segnet_sd_path, device="cpu"))
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _gt_pose_targets(posenet, gt_pairs, pairs):
    targets = {}
    with torch.no_grad():
        for pi in pairs:
            g0 = gt_pairs[pi][0].float().to(DEVICE)
            g1 = gt_pairs[pi][1].float().to(DEVICE)
            x = torch.stack([g0, g1]).unsqueeze(0).permute(0, 1, 4, 2, 3)  # (1,2,3,H,W)
            out = posenet(posenet.preprocess_input(x))["pose"][..., :6]
            targets[pi] = out.detach().clone().requires_grad_(False)
    return targets


def _seg_in(frame1_chw: torch.Tensor) -> torch.Tensor:
    """SegNet input from a single (3,H,W) frame1 (camera res): resize to (384,512)."""

    return F.interpolate(frame1_chw.unsqueeze(0), size=(SEG_H, SEG_W), mode="bilinear", align_corners=False)


def _pose_from_frames(posenet, f0_chw, f1_chw):
    x = torch.stack([f0_chw, f1_chw]).unsqueeze(0)  # (1,2,3,H,W)
    return posenet(posenet.preprocess_input(x))["pose"][..., :6]


def _eval_roundtrip(frame_chw: torch.Tensor) -> torch.Tensor:
    """STE uint8 roundtrip at camera resolution (the contest eval quantizes to uint8)."""

    clamped = torch.clamp(frame_chw, 0, 255)
    return clamped + (torch.round(clamped) - clamped).detach()


def _to_camera(decoder_out_chw: torch.Tensor) -> torch.Tensor:
    """Bilinear-resize the block-stack output (3,fh,fw) to camera res (3,H,W)."""

    return F.interpolate(
        decoder_out_chw.unsqueeze(0), size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
    )[0]


# ---------------------------------------------------------------------------
# The THREE selectable d_seg loss terms (task #63 decisive test).
# Each takes student SegNet logits (1,5,384,512) [gradient flows] + the per-pair
# precomputed targets; returns a scalar. The ONLY thing that varies across arms.
# ---------------------------------------------------------------------------
SEG_LOSS_CHOICES = ("argmax_ce", "kl_distill_t2", "margin_hinge")
KL_TEMPERATURE = 2.0  # PR95 / Quantizr canon (Hinton-Vinyals-Dean 2014, T=2.0)
MARGIN_HINGE_GAMMA = 1.0  # slack past the flip point (logit-margin units)


def _seg_loss_argmax_ce(
    student_logits: torch.Tensor, seg_label: torch.Tensor, w_boundary: torch.Tensor
) -> torch.Tensor:
    """Arm 1 (#62 baseline): boundary-weighted CE against the GT SegNet argmax labels.

    Gradient touches the SegNet only through the GT-argmax-class log-prob.
    """

    ce = F.cross_entropy(student_logits, seg_label[None], reduction="none")[0]  # (384,512)
    return (ce * w_boundary).mean()


def _seg_loss_kl_distill_t2(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, temperature: float = KL_TEMPERATURE
) -> torch.Tensor:
    """Arm 2 (PR95): ``T^2 * KL(log_softmax(student/T) || softmax(teacher/T))`` at T=2.0.

    Mirrors ``tac.losses.u_die_kl.kl_distill_segnet_term`` (the canonical PR95 SegNet-logit
    distillation) for the single-frame1 SegNet path the trainer uses. The teacher is the FROZEN
    SegNet logits on GT frame1 (detached, no grad). The gradient flows through the FULL soft 5-class
    distribution — boundary-aware by construction (runner-up mass near a boundary).
    """

    T = float(temperature)
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1).detach()
    kl_per_pixel = F.kl_div(log_p, q, reduction="none").sum(dim=1)  # (1,384,512)
    return kl_per_pixel.mean() * (T * T)


def _seg_loss_margin_hinge(
    student_logits: torch.Tensor,
    seg_label: torch.Tensor,
    w_boundary: torch.Tensor,
    *,
    gamma: float = MARGIN_HINGE_GAMMA,
) -> torch.Tensor:
    """Arm 3 (#52/#55 boundary-solver gradient): differentiable argmax-margin hinge.

    For the GT-argmax source class ``s = A_GT(p)`` at each pixel, penalize
    ``max(0, gamma - (logit[s] - max_{c != s} logit[c]))``. This directly pushes the student argmax
    margin toward and past the flip point, using the REAL SegNet input-Jacobian
    ``g_p = J_{s,p} - J_{c2,p}`` via autograd through the frozen SegNet (the exact polytope coefficient
    the closed-form #55 solver is written over). Boundary-weighted by the same ``w_boundary``.
    """

    logits = student_logits[0]  # (5,384,512)
    n_cls = logits.shape[0]
    s = seg_label.long()  # (384,512), source = GT argmax class
    # logit of the source class per pixel
    src_logit = torch.gather(logits, 0, s[None])[0]  # (384,512)
    # max logit over the WRONG classes: mask the source class to -inf then max over classes.
    onehot = F.one_hot(s, num_classes=n_cls).permute(2, 0, 1).bool()  # (5,384,512)
    masked = logits.masked_fill(onehot, float("-inf"))
    max_wrong = masked.max(dim=0).values  # (384,512)
    margin = src_logit - max_wrong  # > 0 where student argmax == GT argmax
    hinge = F.relu(float(gamma) - margin)  # penalize margin below gamma (incl. flipped pixels)
    return (hinge * w_boundary).mean()


def _compute_seg_loss(
    seg_loss_mode: str,
    student_logits: torch.Tensor,
    *,
    seg_label: torch.Tensor,
    w_boundary: torch.Tensor,
    teacher_logits: torch.Tensor | None,
) -> torch.Tensor:
    if seg_loss_mode == "argmax_ce":
        return _seg_loss_argmax_ce(student_logits, seg_label, w_boundary)
    if seg_loss_mode == "kl_distill_t2":
        if teacher_logits is None:
            raise ValueError("kl_distill_t2 requires teacher_logits (GT-frame1 SegNet logits)")
        return _seg_loss_kl_distill_t2(student_logits, teacher_logits)
    if seg_loss_mode == "margin_hinge":
        return _seg_loss_margin_hinge(student_logits, seg_label, w_boundary)
    raise ValueError(f"unknown seg_loss_mode={seg_loss_mode!r}; choose from {SEG_LOSS_CHOICES}")


def train(
    targets_dir: Path,
    out_dir: Path,
    cfg: ConvDecoderConfig,
    *,
    n_pairs: int,
    epochs: int,
    lr: float,
    seed: int,
    eval_every: int,
    pose_carrier_mode: bool,
    seg_floor: float,
    pose_floor: float,
    seg_loss_mode: str = "argmax_ce",
    eval_weights: str = "ema",
) -> dict[str, Any]:
    import render_and_score_lib as L

    from tac.boundary_math.margin_polytope import free_budget_from_margin_jacobian
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally, unpatch_upstream_yuv6

    if seg_loss_mode not in SEG_LOSS_CHOICES:
        raise ValueError(f"seg_loss_mode={seg_loss_mode!r} not in {SEG_LOSS_CHOICES}")
    _refuse_tmp(out_dir, "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    meta = json.loads((targets_dir / "targets_meta.json").read_text())
    n_built = int(meta["num_pairs_built"])
    pairs = list(range(min(n_pairs, n_built)))
    cfg = ConvDecoderConfig(**{**cfg.to_dict(), "num_pairs": len(pairs),
                               "stage_channels": tuple(cfg.stage_channels)})

    t0 = time.time()
    posenet = _load_posenet()
    gt_pairs = L.decode_gt_pairs(pairs)
    gt_pose_targets = _gt_pose_targets(posenet, gt_pairs, pairs)

    # GT SegNet argmax labels (the soft d_seg target) + margin (the seg-null free budget) at 384x512.
    gt_argmax = np.memmap(targets_dir / "gt_segnet_argmax.u8", dtype=np.uint8, mode="r",
                          shape=(n_built, SEG_H, SEG_W))
    gt_margin = np.memmap(targets_dir / "gt_segnet_margin.f16", dtype=np.float16, mode="r",
                          shape=(n_built, SEG_H, SEG_W))
    seg_labels = {pi: torch.from_numpy(np.asarray(gt_argmax[pi]).astype(np.int64)) for pi in pairs}
    # margin free-budget weight (seg-null pixels = cheap to be wrong = LOW recon weight, but the seg-CE
    # is weighted HIGH at boundary). We build a per-pixel seg-CE weight = inverse free budget (boundary
    # protected). free_budget large => interior => low CE weight; small => boundary => high CE weight.
    seg_ce_weight = {}
    for pi in pairs:
        m = np.asarray(gt_margin[pi]).astype(np.float64)
        fb = free_budget_from_margin_jacobian(m, free_quantile=0.5)
        b = fb.budget
        # boundary protection: weight ~ 1/(floor + budget/max) so small-margin pixels weigh more.
        w = 1.0 / (seg_floor + b / (b.max() + 1e-8))
        w = w / w.mean()  # mean-1 normalize (redistribute, not rescale)
        seg_ce_weight[pi] = torch.from_numpy(w.astype(np.float32))

    # GT frames camera-res (3,H,W) — the recon anchor target.
    gt0 = {pi: gt_pairs[pi][0].float().permute(2, 0, 1).contiguous() for pi in pairs}
    gt1 = {pi: gt_pairs[pi][1].float().permute(2, 0, 1).contiguous() for pi in pairs}
    print(f"[setup] scorers+GT+targets {time.time()-t0:.1f}s n_pairs={len(pairs)}", flush=True)

    # ── the MEASURED PoseNet pixel-Jacobian saliency weight (Jacobian-aimed pose; #61) ──
    patch_token = patch_upstream_yuv6_globally()
    jac_weight = {}  # per-pair (H,W) camera-res pose-relevance weight for frame1 recon
    try:
        for pi in pairs:
            field = compute_posenet_pixel_saliency(
                posenet, gt0[pi], gt1[pi], frame_slot=1
            )  # frame1 Jacobian (frame1 carries pose too)
            wmap = saliency_to_weight_map(field, floor=pose_floor, gamma=1.0, normalize=True)
            jac_weight[pi] = torch.from_numpy(wmap.astype(np.float32))
    finally:
        unpatch_upstream_yuv6(patch_token)
    print(f"[setup] measured PoseNet pixel-Jacobian fields {time.time()-t0:.1f}s", flush=True)

    segnet = _load_segnet()

    # ── teacher SegNet logits on GT frame1 (arm 2 KL-T=2.0 target; frozen, detached) ──
    # The d_seg metric reads SegNet on the resize-to-(384,512) last frame; the teacher is the SAME
    # path on the GT frame1. Computed once per pair (cheap for the smoke pair-count).
    teacher_logits: dict[int, torch.Tensor] = {}
    if seg_loss_mode == "kl_distill_t2":
        with torch.no_grad():
            for pi in pairs:
                teacher_logits[pi] = segnet(_seg_in(gt1[pi])).detach()  # (1,5,384,512)
        print(f"[setup] teacher SegNet logits (GT frame1) for KL-T2 {time.time()-t0:.1f}s", flush=True)

    model = TorchConvPairDecoder(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    ema = EMA(model, decay=0.997)

    patch_token = patch_upstream_yuv6_globally()
    history: list[dict[str, Any]] = []
    try:
        for ep in range(1, epochs + 1):
            order = np.random.permutation(len(pairs))
            ep_pose, ep_seg, ep_recon = 0.0, 0.0, 0.0
            for j in order:
                pi = pairs[j]
                opt.zero_grad()
                blk = model(j)                  # (3, fh, fw) block-stack output
                f1 = _eval_roundtrip(_to_camera(blk))  # camera-res frame1, uint8-STE

                # (1) null-space-primary recon: margin-free-budget-weighted frame1 recon-MSE.
                # error in seg-null (large free budget) is cheap; near-boundary protected. We weight
                # the camera-res recon by the per-pixel seg-CE weight upsampled to camera res.
                seg_w_cam = F.interpolate(
                    seg_ce_weight[pi][None, None], size=(CAMERA_H, CAMERA_W),
                    mode="bilinear", align_corners=False)[0, 0]
                recon_err = (f1 - gt1[pi]).pow(2).mean(dim=0)  # (H,W)
                # combine null-space (seg) weight + Jacobian (pose) weight on the recon anchor.
                recon_w = 0.5 * seg_w_cam + 0.5 * jac_weight[pi]
                recon_mse = (recon_err * recon_w).mean()

                # (3) seg term — ONE of {argmax_ce, kl_distill_t2, margin_hinge} (#63 decisive test).
                # student logits flow gradient through the frozen SegNet; ONLY the seg loss varies.
                seg_logits = segnet(_seg_in(f1))  # (1,5,384,512)
                seg_loss = _compute_seg_loss(
                    seg_loss_mode, seg_logits,
                    seg_label=seg_labels[pi], w_boundary=seg_ce_weight[pi],
                    teacher_logits=teacher_logits.get(pi),
                )

                # (2) Jacobian-aimed pose: EXACT 6-dim PoseNet pose-MSE (the objective).
                # pose_carrier_mode reuses frame1 for frame0 (degenerate placeholder; off by default).
                f0 = f1 if pose_carrier_mode else gt0[pi]
                pose_pred = _pose_from_frames(posenet, f0, f1)
                pose_mse = F.mse_loss(pose_pred, gt_pose_targets[pi])

                # warm schedule: recon-heavy early (basin), seg+pose-heavy late (the objective).
                w_obj = min(1.0, ep / max(1, epochs // 3))
                loss = (
                    (1.0 - 0.5 * w_obj) * recon_mse
                    + (20.0 * w_obj) * seg_loss
                    + (50.0 * w_obj) * pose_mse * 1e4
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema.update(model)
                ep_pose += float(pose_mse)
                ep_seg += float(seg_loss)
                ep_recon += float(recon_mse)
            if ep % eval_every == 0 or ep == 1 or ep == epochs:
                row = {"epoch": ep, "pose_mse_train": ep_pose / len(pairs),
                       "seg_ce_train": ep_seg / len(pairs), "recon_train": ep_recon / len(pairs),
                       "wall_s": round(time.time() - t0, 1)}
                history.append(row)
                print(json.dumps({k: (round(v, 8) if isinstance(v, float) else v)
                                  for k, v in row.items()}), flush=True)
    finally:
        unpatch_upstream_yuv6(patch_token)

    elapsed = time.time() - t0
    if elapsed < epochs * MIN_SEC_PER_EPOCH:
        raise RuntimeError(
            f"internal-consistency: elapsed {elapsed:.2f}s < epochs {epochs} * {MIN_SEC_PER_EPOCH}s "
            f"= {epochs * MIN_SEC_PER_EPOCH:.2f}s — refusing a stub training loop (NO FAKE)."
        )

    # --- R1 disambiguator: measure BOTH the warmup-EMA shadow AND the LIVE
    #     weights. The original lever-C verdict used the CONSTANT-decay EMA
    #     shadow on an 8-pair/180ep short run, which FROZE near init → exact
    #     d_seg read the constant-frame floor 0.507 ("moved by zero"). With
    #     the canonical warmup EMA (Catalog #388) the shadow tracks live, and
    #     measuring LIVE directly removes any residual shadow-lag question.
    orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    scorer = L.ExactScorer()

    # (A) LIVE weights measurement.
    weights_live, latents_live = model.numpy_params()
    exact_live = _exact_measure(
        scorer, weights_live, latents_live, cfg, gt_pairs, pairs, pose_carrier_mode
    )

    # (B) warmup-EMA shadow measurement.
    model.load_state_dict(ema.state_dict())
    try:
        weights_ema, latents_ema = model.numpy_params()
        exact_ema = _exact_measure(
            scorer, weights_ema, latents_ema, cfg, gt_pairs, pairs, pose_carrier_mode
        )
    finally:
        model.load_state_dict(orig_state)

    # The inference checkpoint shipped is selected by eval_weights (default ema
    # = CLAUDE.md EMA-shadow-as-checkpoint discipline, now warmup-correct).
    if eval_weights == "live":
        weights, latents, exact = weights_live, latents_live, exact_live
    else:
        weights, latents, exact = weights_ema, latents_ema, exact_ema
    save_decoder_npz(out_dir / "decoder.npz", weights, latents, cfg)
    byte_acct = measure_decoder_bytes(weights, latents, cfg)
    # parity on the SELECTED weights: reload them so the torch model mirrors
    # the numpy export being byte-accounted.
    sel_state = ema.state_dict() if eval_weights != "live" else orig_state
    model.load_state_dict(sel_state)
    try:
        parity = _verify_parity(model, weights, latents, cfg, pairs[:4])
    finally:
        model.load_state_dict(orig_state)

    rate = 25.0 * byte_acct.total_bytes / _CONTEST_TOTAL_BYTES
    joint_hold = bool(exact["mean_d_seg"] < 0.01 and exact["mean_d_pose"] < 0.01
                      and byte_acct.total_bytes < 120_000)
    result = {
        "subagent": "task63_dseg_loss_decisive_test",
        "utc": _utc(),
        "evidence_grade": "[local CPU-torch advisory]",
        "promotion_eligible": False, "score_claim": False, "ready_for_exact_eval_dispatch": False,
        "seg_loss_mode": seg_loss_mode,
        "config": cfg.to_dict(),
        "param_count": decoder_param_count(cfg),
        "n_pairs": len(pairs),
        "mode": "joint_both_frames" if pose_carrier_mode else "frame1_only_pose_uses_gt0",
        "elapsed_s": round(elapsed, 1),
        "history": history,
        "byte_account": byte_acct.to_dict(),
        "rate_term_decoder_only": rate,
        "exact_mean_d_seg": exact["mean_d_seg"],
        "exact_mean_d_pose": exact["mean_d_pose"],
        "seg_term_contribution_100x": float(100.0 * exact["mean_d_seg"]),
        "pose_term_contribution_sqrt10": float(np.sqrt(10.0 * exact["mean_d_pose"])),
        "portability_parity": parity,
        "joint_hold_under_120kb": joint_hold,
        "constant_frame_control": exact.get("constant_control"),
        # R1 disambiguator: side-by-side LIVE vs warmup-EMA exact d_seg/d_pose.
        # If LIVE d_seg << EMA d_seg, the original "moved by zero" verdict was
        # the EMA-shadow-LAG artifact. If both agree near the floor, the
        # carrier is genuinely seg-blind (the verdict survives the fix).
        "eval_weights_selected": eval_weights,
        "r1_disambiguator": {
            "live_d_seg": exact_live["mean_d_seg"],
            "live_d_pose": exact_live["mean_d_pose"],
            "warmup_ema_d_seg": exact_ema["mean_d_seg"],
            "warmup_ema_d_pose": exact_ema["mean_d_pose"],
            "live_vs_ema_d_seg_gap": float(
                exact_ema["mean_d_seg"] - exact_live["mean_d_seg"]
            ),
            "constant_frame_floor_d_seg": (
                exact.get("constant_control", {}) or {}
            ).get("d_seg"),
            "live_descended_below_floor": bool(
                exact_live["mean_d_seg"]
                < (exact.get("constant_control", {}) or {}).get("d_seg", 1.0) - 0.01
            ),
        },
    }
    (out_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    return result


def _verify_parity(model, weights, latents, cfg, pairs):
    """torch RGB vs numpy RGB at camera res (the portability contract)."""

    agree = []
    for j in range(len(pairs)):
        with torch.inference_mode():
            blk = model(j)
            tr = _to_camera(blk)
            tr = torch.clamp(torch.round(tr.permute(1, 2, 0)), 0, 255).numpy().astype(np.uint8)
        npf = decoder_frame(weights, cfg, latents, j, CAMERA_H, CAMERA_W)
        agree.append(float(np.mean(np.abs(tr.astype(np.int32) - npf.astype(np.int32)) <= 1)))
    return {"rgb_within_1lsb_frac_min": float(min(agree)),
            "rgb_within_1lsb_frac_mean": float(np.mean(agree)),
            "pairs_checked": len(agree), "parity_pass": bool(min(agree) >= 0.99)}


def _exact_measure(scorer, weights, latents, cfg, gt_pairs, pairs, pose_carrier_mode):
    import render_and_score_lib as L

    d_seg_list, d_pose_list = [], []
    # constant-frame control (mean GT1) — proves the decoder is load-bearing.
    const1 = None
    for j, pi in enumerate(pairs):
        f1 = decoder_frame(weights, cfg, latents, j, CAMERA_H, CAMERA_W)  # (H,W,3) uint8 frame1
        f1_chw = torch.from_numpy(f1.transpose(2, 0, 1)).float()
        f0_chw = f1_chw if pose_carrier_mode else gt_pairs[pi][0].float().permute(2, 0, 1)
        comp = torch.stack([f0_chw, f1_chw])  # (2,3,H,W)
        gt_bthwc = torch.stack([gt_pairs[pi][0], gt_pairs[pi][1]]).float().unsqueeze(0)
        pose_d, seg_d = scorer.score_batch(gt_bthwc, L.comp_pair_to_bthwc(comp))
        d_pose_list.append(float(pose_d[0]))
        d_seg_list.append(float(seg_d[0]))
        if const1 is None:
            const1 = torch.from_numpy(
                np.broadcast_to(f1.reshape(-1, 3).mean(0).round(),
                                (CAMERA_H, CAMERA_W, 3)).transpose(2, 0, 1).copy()).float()
    # constant control on the first pair (cheap)
    pi0 = pairs[0]
    f0c = gt_pairs[pi0][0].float().permute(2, 0, 1)
    comp_c = torch.stack([f0c, const1])
    gt_c = torch.stack([gt_pairs[pi0][0], gt_pairs[pi0][1]]).float().unsqueeze(0)
    pose_c, seg_c = scorer.score_batch(gt_c, L.comp_pair_to_bthwc(comp_c))
    return {
        "mean_d_pose": float(np.mean(d_pose_list)),
        "mean_d_seg": float(np.mean(d_seg_list)),
        "per_pair_d_pose": d_pose_list,
        "per_pair_d_seg": d_seg_list,
        "constant_control": {"d_pose": float(pose_c[0]), "d_seg": float(seg_c[0])},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--targets-dir", type=Path, default=Path(base) / "targets_n600")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose-carrier-mode", action="store_true",
                    help="decoder also makes frame0 (joint); default frame0=GT0 for pure frame1 RD")
    ap.add_argument("--seg-floor", type=float, default=0.05)
    ap.add_argument("--pose-floor", type=float, default=0.05)
    ap.add_argument("--seg-loss", type=str, default="argmax_ce", choices=SEG_LOSS_CHOICES,
                    help="the #63 decisive-test d_seg loss arm (the ONLY thing that varies across arms)")
    ap.add_argument("--eval-weights", type=str, default="ema", choices=("ema", "live"),
                    help="inference checkpoint: warmup-EMA shadow (default) or LIVE weights. "
                         "Both d_seg/d_pose are ALWAYS reported in r1_disambiguator regardless.")
    # capacity knobs
    ap.add_argument("--latent-dim", type=int, default=24)
    ap.add_argument("--seed-ch", type=int, default=32)
    ap.add_argument("--stage-channels", type=str, default="32,24,16,12")
    ap.add_argument("--quant-bits", type=int, default=8)
    args = ap.parse_args(argv)

    stage_channels = tuple(int(x) for x in args.stage_channels.split(","))
    cfg = ConvDecoderConfig(
        num_pairs=args.n_pairs, latent_dim=args.latent_dim, seed_ch=args.seed_ch,
        stage_channels=stage_channels, quant_bits=args.quant_bits,
    )
    result = train(
        args.targets_dir, args.out_dir, cfg, n_pairs=args.n_pairs, epochs=args.epochs,
        lr=args.lr, seed=args.seed, eval_every=args.eval_every,
        pose_carrier_mode=args.pose_carrier_mode, seg_floor=args.seg_floor, pose_floor=args.pose_floor,
        seg_loss_mode=args.seg_loss, eval_weights=args.eval_weights,
    )
    print("\n=== LEVER-C TRAIN RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
