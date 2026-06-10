#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Distill a SMALLER learned student from the frontier HNeRV teacher (task #74).

THE KEY INSIGHT (the wall-breaker): task #62 proved a small conv decoder CANNOT learn d_seg from
argmax-CE-against-GT (the RGB->frozen-SegNet path is ill-conditioned; the trained student pinned at
the constant-frame d_seg floor). BUT the frontier teacher (the 177KB HNeRV decoder) ALREADY decodes
1200 frames that ARE d_seg-correct (~5.4e-4) AND pose-in-tube (~2.3e-5). A SMALLER student trained to
MATCH THE TEACHER'S DECODED FRAMES learns from targets already on the scorer manifold -> it inherits
score-correctness via well-conditioned RGB recon instead of fighting the conditioning. **The teacher
IS the loss.**

THE KD LOSS (teacher is the target, NOT GT):
  1. teacher-frame recon (both frames; the well-conditioned objective)
  2. PR95 KL-T=2.0 SegNet-logit distill (student SegNet logits match the TEACHER frame1 logits)
  3. pose-MSE distill (student 6-dim pose matches the TEACHER's in-tube pose)

eval_roundtrip (uint8 STE) + differentiable rgb_to_yuv6 in the inner loop (CLAUDE.md non-negotiable).
EMA-0.997 shadow is the inference checkpoint. The numpy-portable student forward reproduces the torch
forward (parity gate). d_seg/d_pose RE-MEASURED on the exact frozen CPU scorer with the numpy-decoded
frames; GT via ``frame_utils.yuv420_to_rgb`` ONLY (the rgb24 path manufactures ~100x phantom pose).

Authority ``[macOS-MLX research-signal]`` (student forward) + ``[local CPU-torch advisory]`` (exact
scorer). NO MPS. $0 local. Non-promotable. No paid dispatch from this loop. Only the contest exact
``evaluate.py`` (CPU+CUDA) can move the frontier pointer.

NO-FAKE (class 2/6/8): real training (internal-consistency elapsed >= epochs*MIN_SEC); the d_seg/
d_pose are the EXACT frozen-scorer measurements (not a proxy); the bytes are the brotli of the ACTUAL
quantized weights+latents; a constant-frame student is the reported control and CANNOT reduce d_seg.
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

from tac.distillation.smaller_student import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    KL_TEMPERATURE,
    SEG_H,
    SEG_W,
    StudentByteAccount,
    StudentDecoderConfig,
    measure_student_bytes,
    save_student_npz,
    size_ladder,
    student_pair_frames,
    student_param_count,
)

_CONTEST_TOTAL_BYTES = 37_545_489  # evaluate.py:64 fixed denominator
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
DEVICE = torch.device("cpu")  # NO MPS, NO cuda for the local advisory loop.
MIN_SEC_PER_EPOCH = 0.02  # internal-consistency floor (refuse a stub loop).


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# Torch student decoder — mirror of smaller_student.numpy_reference_forward.
# Decodes BOTH frames of the pair (PoseNet reads both; SegNet reads frame1).
# ---------------------------------------------------------------------------
class TorchStudentDecoder(nn.Module):
    """Per-pair-latent conv decoder: latent -> seed -> N (Conv+PixelShuffle+bilinear-skip+sin) ->
    2*n_channels RGB (frame0 channels then frame1 channels)."""

    def __init__(self, cfg: StudentDecoderConfig):
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
        self.out = nn.Conv2d(in_ch, cfg.out_channels, 3, padding=1)
        self.latents = nn.Parameter(torch.zeros(cfg.num_pairs, cfg.latent_dim))
        nn.init.normal_(self.latents, std=0.1)

    def forward(self, pair_idx: int) -> torch.Tensor:
        """Returns (2, n_channels, final_h, final_w) RGB in [0,255] at the block-stack resolution."""

        z = self.latents[pair_idx]
        h = self.seed(z).reshape(1, self.cfg.seed_ch, self.cfg.seed_h, self.cfg.seed_w)
        for stage, skip in zip(self.stages, self.skips, strict=True):
            conv = stage(h)
            up = F.pixel_shuffle(conv, 2)
            skip_in = F.interpolate(h, size=up.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.sin(up + skip(skip_in))
        rgb01 = torch.sigmoid(self.out(h))[0]  # (2*n_channels, fh, fw)
        fh, fw = rgb01.shape[-2], rgb01.shape[-1]
        return (rgb01 * 255.0).reshape(2, self.cfg.n_channels, fh, fw)

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
# Scorer loading (exact frozen CPU PoseNet + SegNet) — frozen.
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


def _seg_in(frame1_chw: torch.Tensor) -> torch.Tensor:
    """SegNet input from a single (3,H,W) frame1 (camera res): resize to (384,512)."""

    return F.interpolate(
        frame1_chw.unsqueeze(0), size=(SEG_H, SEG_W), mode="bilinear", align_corners=False
    )


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
# Teacher targets — the frontier HNeRV decoder's decoded frames + scorer reads.
# ---------------------------------------------------------------------------
def _build_teacher_targets(pairs, posenet, segnet):
    """Decode the frontier teacher for ``pairs`` and precompute the distillation targets.

    Returns dict with:
      teacher_f0[pi], teacher_f1[pi]  : (3,H,W) float camera-res frames (the recon targets)
      teacher_seg_logits[pi]          : (1,5,384,512) frozen SegNet logits on teacher frame1 (KL-T2)
      teacher_pose[pi]                : (1,6) frozen PoseNet pose on the teacher pair (pose-MSE)
    """

    import render_and_score_lib as L

    rend = L.FrontierRenderer()
    comp = rend.render_baseline_pairs(pairs)  # dict pi -> (2,3,H,W) float rounded
    t = {"f0": {}, "f1": {}, "seg_logits": {}, "pose": {}}
    with torch.no_grad():
        for pi in pairs:
            # .clone() converts the inference-mode tensors from render_baseline_pairs into normal
            # tensors so they can be used as autograd targets in the training loop.
            f0 = comp[pi][0].float().contiguous().clone()  # (3,H,W) camera-res
            f1 = comp[pi][1].float().contiguous().clone()
            t["f0"][pi] = f0
            t["f1"][pi] = f1
            t["seg_logits"][pi] = segnet(_seg_in(f1)).detach().clone()  # (1,5,384,512)
            t["pose"][pi] = _pose_from_frames(posenet, f0, f1).detach().clone()  # (1,6)
    return t, rend


# ---------------------------------------------------------------------------
# KD loss terms (the teacher is the target).
# ---------------------------------------------------------------------------
def kd_seg_kl_t2(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, temperature: float = KL_TEMPERATURE
) -> torch.Tensor:
    """PR95 KL-T=2.0 SegNet-logit distill: ``T^2 * KL(log_softmax(student/T) || softmax(teacher/T))``.

    ``student_logits`` (1,5,384,512) gradient flows; ``teacher_logits`` is the FROZEN frontier-teacher
    SegNet logits on the teacher's frame1 (detached). Hinton T^2 normalization keeps the gradient
    scale temperature-invariant. The gradient flows through the FULL soft 5-class distribution =
    exactly the teacher's argmax partition (the dark-knowledge transfer).
    """

    T = float(temperature)
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1).detach()
    kl_per_pixel = F.kl_div(log_p, q, reduction="none").sum(dim=1)  # (1,384,512)
    return kl_per_pixel.mean() * (T * T)


def kd_pose_mse(student_pose: torch.Tensor, teacher_pose: torch.Tensor) -> torch.Tensor:
    """Pose-MSE distill: student 6-dim pose matches the TEACHER's in-tube 6-dim pose."""

    return F.mse_loss(student_pose, teacher_pose.detach())


def kd_frame_recon(
    student_f0: torch.Tensor, student_f1: torch.Tensor,
    teacher_f0: torch.Tensor, teacher_f1: torch.Tensor,
) -> torch.Tensor:
    """Teacher-frame recon (both frames): the well-conditioned objective the #62 finding endorses.

    Matching the teacher's frame (already on the scorer manifold) inherits score-correctness without
    fighting the RGB->SegNet conditioning that #62 proved blocks GT-argmax-CE.
    """

    return F.mse_loss(student_f0, teacher_f0.detach()) + F.mse_loss(student_f1, teacher_f1.detach())


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------
def train(
    out_dir: Path,
    cfg: StudentDecoderConfig,
    *,
    n_pairs: int,
    epochs: int,
    lr: float,
    seed: int,
    eval_every: int,
    w_seg: float,
    w_pose: float,
    w_recon: float,
) -> dict[str, Any]:
    import render_and_score_lib as L

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally, unpatch_upstream_yuv6

    _refuse_tmp(out_dir, "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    pairs = list(range(n_pairs))
    cfg = StudentDecoderConfig(**{**cfg.to_dict(), "num_pairs": len(pairs),
                                  "stage_channels": tuple(cfg.stage_channels)})

    t0 = time.time()
    posenet = _load_posenet()
    segnet = _load_segnet()
    teacher, _rend = _build_teacher_targets(pairs, posenet, segnet)
    gt_pairs = L.decode_gt_pairs(pairs)  # for the EXACT final measurement only (NOT a training target)
    print(f"[setup] scorers+teacher+GT {time.time()-t0:.1f}s n_pairs={len(pairs)} "
          f"size_label={cfg.size_label} params={student_param_count(cfg)}", flush=True)

    model = TorchStudentDecoder(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    ema = EMA(model, decay=0.997)

    patch_token = patch_upstream_yuv6_globally()
    history: list[dict[str, Any]] = []
    try:
        for ep in range(1, epochs + 1):
            order = np.random.permutation(len(pairs))
            ep_recon, ep_seg, ep_pose = 0.0, 0.0, 0.0
            for j in order:
                pi = pairs[j]
                opt.zero_grad()
                pair = model(j)  # (2,3,fh,fw) block-stack
                f0 = _eval_roundtrip(_to_camera(pair[0]))  # camera-res frame0, uint8-STE
                f1 = _eval_roundtrip(_to_camera(pair[1]))  # camera-res frame1, uint8-STE

                # (1) teacher-frame recon (the well-conditioned objective).
                recon = kd_frame_recon(f0, f1, teacher["f0"][pi], teacher["f1"][pi])
                # (2) PR95 KL-T2 SegNet distill: student logits match teacher frame1 logits.
                seg_logits = segnet(_seg_in(f1))  # (1,5,384,512) gradient flows
                seg = kd_seg_kl_t2(seg_logits, teacher["seg_logits"][pi])
                # (3) pose-MSE distill: student pose matches teacher in-tube pose.
                pose_pred = _pose_from_frames(posenet, f0, f1)
                pose = kd_pose_mse(pose_pred, teacher["pose"][pi])

                # warm schedule: recon-heavy early (basin), distill-heavy late (the objective).
                w_obj = min(1.0, ep / max(1, epochs // 3))
                loss = (
                    (w_recon * (1.0 - 0.3 * w_obj)) * recon
                    + (w_seg * w_obj) * seg
                    + (w_pose * w_obj) * pose * 1e4
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema.update(model)
                ep_recon += float(recon.detach())
                ep_seg += float(seg.detach())
                ep_pose += float(pose.detach())
            if ep % eval_every == 0 or ep == 1 or ep == epochs:
                row = {"epoch": ep, "recon_train": ep_recon / len(pairs),
                       "seg_kl_train": ep_seg / len(pairs), "pose_mse_train": ep_pose / len(pairs),
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
        save_student_npz(out_dir / "student.npz", weights, latents, cfg)
        byte_acct = measure_student_bytes(weights, latents, cfg)
        parity = _verify_parity(model, weights, latents, cfg, pairs[: min(4, len(pairs))])
        scorer = L.ExactScorer()
        exact = _exact_measure(scorer, weights, latents, cfg, gt_pairs, pairs)
    finally:
        model.load_state_dict(orig_state)

    rate = 25.0 * byte_acct.total_bytes / _CONTEST_TOTAL_BYTES
    score_after = (100.0 * exact["mean_d_seg"]
                   + float(np.sqrt(10.0 * exact["mean_d_pose"]))
                   + rate)
    result = {
        "subagent": "task74_distill_smaller_student",
        "utc": _utc(),
        "evidence_grade": "[local CPU-torch advisory]",
        "promotion_eligible": False, "score_claim": False, "ready_for_exact_eval_dispatch": False,
        "size_label": cfg.size_label,
        "config": cfg.to_dict(),
        "param_count": student_param_count(cfg),
        "n_pairs": len(pairs),
        "elapsed_s": round(elapsed, 1),
        "history": history,
        "byte_account": byte_acct.to_dict(),
        "rate_term_student_only": rate,
        "exact_mean_d_seg": exact["mean_d_seg"],
        "exact_mean_d_pose": exact["mean_d_pose"],
        "seg_term_contribution_100x": float(100.0 * exact["mean_d_seg"]),
        "pose_term_contribution_sqrt10": float(np.sqrt(10.0 * exact["mean_d_pose"])),
        "advisory_score_student_only": score_after,
        "portability_parity": parity,
        "teacher_d_seg_ref": exact.get("teacher_d_seg"),
        "teacher_d_pose_ref": exact.get("teacher_d_pose"),
        "constant_frame_control": exact.get("constant_control"),
    }
    (out_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    return result


def _verify_parity(model, weights, latents, cfg, pairs):
    """torch RGB vs numpy RGB at camera res (the portability contract), per frame."""

    agree = []
    for j in range(len(pairs)):
        with torch.inference_mode():
            pair = model(j)  # (2,3,fh,fw)
            f0 = _to_camera(pair[0]); f1 = _to_camera(pair[1])
            tr0 = torch.clamp(torch.round(f0.permute(1, 2, 0)), 0, 255).numpy().astype(np.uint8)
            tr1 = torch.clamp(torch.round(f1.permute(1, 2, 0)), 0, 255).numpy().astype(np.uint8)
        npf = student_pair_frames(weights, cfg, latents, j, CAMERA_H, CAMERA_W)  # (2,H,W,3)
        a0 = float(np.mean(np.abs(tr0.astype(np.int32) - npf[0].astype(np.int32)) <= 1))
        a1 = float(np.mean(np.abs(tr1.astype(np.int32) - npf[1].astype(np.int32)) <= 1))
        agree.append(min(a0, a1))
    return {"rgb_within_1lsb_frac_min": float(min(agree)),
            "rgb_within_1lsb_frac_mean": float(np.mean(agree)),
            "pairs_checked": len(agree), "parity_pass": bool(min(agree) >= 0.99)}


def _exact_measure(scorer, weights, latents, cfg, gt_pairs, pairs):
    """Exact frozen-scorer d_seg/d_pose on the numpy-decoded student frames (the authority surface
    for this advisory loop). Also reports the constant-frame control (proves the student is
    load-bearing — a constant frame CANNOT reduce d_seg)."""

    import render_and_score_lib as L

    d_seg_list, d_pose_list = [], []
    const1 = None
    for j, pi in enumerate(pairs):
        frames = student_pair_frames(weights, cfg, latents, j, CAMERA_H, CAMERA_W)  # (2,H,W,3) uint8
        f0_chw = torch.from_numpy(frames[0].transpose(2, 0, 1)).float()
        f1_chw = torch.from_numpy(frames[1].transpose(2, 0, 1)).float()
        comp = torch.stack([f0_chw, f1_chw])  # (2,3,H,W)
        gt_bthwc = torch.stack([gt_pairs[pi][0], gt_pairs[pi][1]]).float().unsqueeze(0)
        pose_d, seg_d = scorer.score_batch(gt_bthwc, L.comp_pair_to_bthwc(comp))
        d_pose_list.append(float(pose_d[0]))
        d_seg_list.append(float(seg_d[0]))
        if const1 is None:
            const1 = torch.from_numpy(
                np.broadcast_to(frames[1].reshape(-1, 3).mean(0).round(),
                                (CAMERA_H, CAMERA_W, 3)).transpose(2, 0, 1).copy()).float()
    # constant control on the first pair (cheap) — constant frame1 (its argmax is a blank partition).
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


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=list(size_ladder(1).keys()) + ["custom"], default="80kb")
    ap.add_argument("--latent-dim", type=int, default=None, help="override (custom size)")
    ap.add_argument("--seed-ch", type=int, default=None)
    ap.add_argument("--stage-channels", type=str, default=None, help="comma list, custom size")
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--w-seg", type=float, default=20.0)
    ap.add_argument("--w-pose", type=float, default=50.0)
    ap.add_argument("--w-recon", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.size == "custom":
        if args.latent_dim is None or args.seed_ch is None or args.stage_channels is None:
            raise SystemExit("--size custom requires --latent-dim --seed-ch --stage-channels")
        stages = tuple(int(x) for x in args.stage_channels.split(","))
        cfg = StudentDecoderConfig(num_pairs=args.n_pairs, latent_dim=args.latent_dim,
                                   seed_ch=args.seed_ch, stage_channels=stages, size_label="custom")
    else:
        cfg = size_ladder(args.n_pairs)[args.size]
    result = train(
        args.out_dir, cfg, n_pairs=args.n_pairs, epochs=args.epochs, lr=args.lr, seed=args.seed,
        eval_every=args.eval_every, w_seg=args.w_seg, w_pose=args.w_pose, w_recon=args.w_recon,
    )
    print(json.dumps({
        "size_label": result["size_label"],
        "total_bytes": result["byte_account"]["total_bytes"],
        "exact_mean_d_seg": result["exact_mean_d_seg"],
        "exact_mean_d_pose": result["exact_mean_d_pose"],
        "advisory_score_student_only": result["advisory_score_student_only"],
        "parity_pass": result["portability_parity"]["parity_pass"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
