#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lever-C: train the small per-pair-latent CONV frame1 decoder JOINTLY seg+pose (task #62).

THE ORIGINAL METHODOLOGY (no leaderboard entry uses this combination): a fresh-init small conv
decoder (Conv+PixelShuffle+bilinear-skip+sin, PR95 L18) generates frame1, trained DIRECTLY against
the FROZEN scorers with three original terms:
  1. null-space-primary recon: frame1 recon-MSE WEIGHTED by the SegNet margin free-budget (#52
     margin_polytope) — error is steered INTO the seg-null subspace (large-margin interior pixels are
     cheap to be wrong; small-margin boundary pixels are protected).
  2. Jacobian-aimed pose: frame0 (and frame1) recon-MSE WEIGHTED by the MEASURED PoseNet pixel-Jacobian
     (#61 posenet_jacobian_saliency) + the EXACT 6-dim PoseNet pose-MSE objective in the loop.
  3. argmax-polytope-constrained seg: soft d_seg surrogate = boundary-weighted cross-entropy against
     the GT SegNet argmax (the precomputed targets) — the exact argmax-flip d_seg is RE-MEASURED on the
     frozen SegNet.

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


class EMA:
    """Quantizr-0.997 EMA shadow (CLAUDE.md EMA non-negotiable). Applied at eval, snapshot+restore."""

    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


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
) -> dict[str, Any]:
    import render_and_score_lib as L

    from tac.boundary_math.margin_polytope import free_budget_from_margin_jacobian
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally, unpatch_upstream_yuv6

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

                # (3) argmax-polytope-constrained seg: boundary-weighted CE against GT SegNet argmax.
                seg_logits = segnet(_seg_in(f1))  # (1,5,384,512)
                ce = F.cross_entropy(seg_logits, seg_labels[pi][None], reduction="none")[0]  # (384,512)
                seg_loss = (ce * seg_ce_weight[pi]).mean()

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

    # --- EMA shadow is the inference checkpoint (CLAUDE.md EMA non-negotiable) ---
    orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema.state_dict())
    try:
        weights, latents = model.numpy_params()
        save_decoder_npz(out_dir / "decoder.npz", weights, latents, cfg)
        byte_acct = measure_decoder_bytes(weights, latents, cfg)
        parity = _verify_parity(model, weights, latents, cfg, pairs[:4])
        scorer = L.ExactScorer()
        exact = _exact_measure(scorer, weights, latents, cfg, gt_pairs, pairs, pose_carrier_mode)
    finally:
        model.load_state_dict(orig_state)

    rate = 25.0 * byte_acct.total_bytes / _CONTEST_TOTAL_BYTES
    joint_hold = bool(exact["mean_d_seg"] < 0.01 and exact["mean_d_pose"] < 0.01
                      and byte_acct.total_bytes < 120_000)
    result = {
        "subagent": "task62_lever_c_viability_smoke",
        "utc": _utc(),
        "evidence_grade": "[local CPU-torch advisory]",
        "promotion_eligible": False, "score_claim": False, "ready_for_exact_eval_dispatch": False,
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
    )
    print("\n=== LEVER-C TRAIN RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
