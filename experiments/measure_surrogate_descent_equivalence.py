#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""LOCAL-THROUGHPUT-ATTACK Angle 2: distilled-surrogate SegNet scorer.

Question (MEASURED, not assumed): can a SMALL cheap student CNN, distilled from
the frozen 9.54M-param EfficientNet-B2 SegNet, supply a d_seg gradient that
DESCENDS to the SAME exact (canonical-scorer) d_seg basin as the full-scorer
gradient? If yes, the per-step training cost drops by the student/teacher cost
ratio on ANY hardware (the algorithmic unblock).

Protocol:
  1. Decode N real GT frames via frame_utils.yuv420_to_rgb (the authority decode).
  2. Distill a small student CNN to match the frozen SegNet's 5-class logits on
     those frames (a short supervised fit; teacher logits are the soft target).
  3. Measure per-step fwd+bwd cost: full SegNet vs student.
  4. Descent A/B: optimize a learnable RGB render to push its SegNet argmax AWAY
     from a fixed GT target (a stand-in score-aware descent that isolates the
     SegNet gradient). Arm A uses the FULL SegNet gradient; arm B uses the
     STUDENT gradient. BOTH arms are evaluated with the EXACT canonical SegNet
     d_seg (argmax-flip rate) every K steps. We report whether arm B reaches the
     same exact-d_seg trajectory/basin as arm A.

NO-FAKE: the only d_seg reported is the EXACT canonical SegNet argmax-flip rate
(upstream/modules.py SegNet.compute_distortion). The student is a THROUGHPUT
tool; it never produces a reported d_seg. NO MPS. torch-CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "upstream"))

from modules import SegNet  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

SEG_W, SEG_H = 512, 384


def load_gt_frames(n: int, device: str) -> torch.Tensor:
    """Decode n GT frames via the authority path, resize to SegNet input."""
    import av
    from frame_utils import yuv420_to_rgb

    c = av.open(str(REPO / "upstream/videos/0.mkv"))
    out = []
    for i, f in enumerate(c.decode(video=0)):
        if i >= n:
            break
        rgb = yuv420_to_rgb(f).permute(2, 0, 1).float()  # (3,H,W) 0..255
        out.append(rgb)
    c.close()
    x = torch.stack(out).to(device)  # (n,3,874,1164)
    x = F.interpolate(x, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False)
    return x  # (n,3,384,512), 0..255


def build_segnet(device: str) -> SegNet:
    seg = SegNet().to(device)
    seg.load_state_dict(load_file(str(REPO / "upstream/models/segnet.safetensors"), device=device))
    for p in seg.parameters():
        p.requires_grad_(False)
    seg.eval()
    return seg


class StudentSeg(nn.Module):
    """Small distilled student approximating SegNet's 5-class logits.

    A compact encoder-decoder: strided convs down then bilinear up to full res.
    Channel width `c` controls capacity/cost. Outputs (B,5,384,512) logits.
    """

    def __init__(self, c: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.GELU(),       # /2
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.GroupNorm(8, c * 2), nn.GELU(),  # /4
            nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.GroupNorm(8, c * 2), nn.GELU(),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.GroupNorm(8, c * 4), nn.GELU(),  # /8
            nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.GroupNorm(8, c * 4), nn.GELU(),
        )
        self.head = nn.Conv2d(c * 4, 5, 1)

    def forward(self, x):
        # x is 0..255; normalize to ~[0,1] like a typical CNN front-end
        h = self.enc(x / 255.0)
        h = self.head(h)
        return F.interpolate(h, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False)


def exact_d_seg(seg: SegNet, render: torch.Tensor, target_logits: torch.Tensor) -> float:
    """Canonical SegNet argmax-flip rate between render's seg and the target."""
    with torch.no_grad():
        out = seg(render)
        diff = (out.argmax(dim=1) != target_logits.argmax(dim=1)).float()
        return diff.mean().item()


def distill_student(seg: SegNet, frames: torch.Tensor, student: StudentSeg,
                    steps: int, lr: float) -> list:
    """Fit student logits to frozen SegNet logits on the GT frames."""
    with torch.no_grad():
        teacher = seg(frames)  # (n,5,384,512)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    student.train()
    log = []
    for s in range(steps):
        opt.zero_grad()
        pred = student(frames)
        # KD: soft logit MSE + argmax-agreement is the practical target
        loss = F.mse_loss(pred, teacher)
        loss.backward()
        opt.step()
        if s % max(1, steps // 8) == 0 or s == steps - 1:
            with torch.no_grad():
                agree = (pred.argmax(1) == teacher.argmax(1)).float().mean().item()
            log.append({"step": s, "kd_mse": loss.item(), "argmax_agree": agree})
    student.eval()
    return log


def time_step(fn, warmup=1, iters=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--student-channels", type=int, default=32)
    ap.add_argument("--distill-steps", type=int, default=400)
    ap.add_argument("--distill-lr", type=float, default=3e-4)
    ap.add_argument("--descent-steps", type=int, default=60)
    ap.add_argument("--descent-lr", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = "cpu"
    seg = build_segnet(device)
    frames = load_gt_frames(args.frames, device)
    print(f"=== Surrogate descent-equivalence: frames={args.frames} student_c={args.student_channels} ===")

    # --- param counts ---
    student = StudentSeg(c=args.student_channels).to(device)
    seg_params = sum(p.numel() for p in seg.parameters())
    stu_params = sum(p.numel() for p in student.parameters())
    print(f"SegNet params: {seg_params/1e6:.2f}M | Student params: {stu_params/1e6:.3f}M "
          f"({seg_params/stu_params:.0f}x smaller)")

    # --- distill ---
    print("Distilling student to SegNet logits...")
    distill_log = distill_student(seg, frames, student, args.distill_steps, args.distill_lr)
    final_agree = distill_log[-1]["argmax_agree"]
    print(f"Student final argmax-agreement with SegNet: {final_agree*100:.1f}%")

    # --- per-step cost (fwd+bwd on a learnable render leaf) ---
    B = args.frames
    seg_target = seg(frames).detach()

    def full_step():
        x = frames.clone().requires_grad_(True)
        out = seg(x)
        # score-aware-style loss: push argmax away from target -> minimize logit
        # match on target class (CE-like surrogate cost; we measure TIME here)
        loss = -F.cross_entropy(out, seg_target.argmax(1))
        loss.backward()

    def student_step():
        x = frames.clone().requires_grad_(True)
        out = student(x)
        loss = -F.cross_entropy(out, seg_target.argmax(1))
        loss.backward()

    full_dt = time_step(full_step)
    stu_dt = time_step(student_step)
    print(f"Full SegNet fwd+bwd: {full_dt:.3f} s/step | Student: {stu_dt:.4f} s/step "
          f"({full_dt/stu_dt:.1f}x faster)")

    # --- descent A/B: a REAL, non-degenerate score-aware descent ---
    # Target = the GT-frame SegNet class map (what a perfect renderer reproduces).
    # The render STARTS corrupted (heavy blur + noise) so initial exact d_seg is
    # HIGH, then both arms minimize CE-to-target to DESCEND d_seg toward 0 — the
    # renderer-approaches-GT direction of score-aware training. Arm A uses the
    # full-SegNet CE gradient; arm B uses the STUDENT CE gradient; BOTH evaluated
    # with the EXACT canonical SegNet d_seg (argmax-flip vs the GT target).
    target = seg_target.argmax(1)  # frozen GT class target (B,384,512)

    def corrupt(frames_):
        # large-kernel blur + additive noise -> a poor initial render with HIGH d_seg
        g = frames_.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)  # desaturate
        k = 31
        blur = F.avg_pool2d(g, kernel_size=k, stride=1, padding=k // 2)
        return (blur + 40.0 * torch.randn_like(blur)).clamp(0, 255)

    def run_descent(grad_model, label):
        torch.manual_seed(args.seed)  # identical init/noise for both arms
        render = corrupt(frames).clone().requires_grad_(True)
        opt = torch.optim.Adam([render], lr=args.descent_lr)
        traj = []
        for st in range(args.descent_steps + 1):
            if st % args.eval_every == 0:
                d = exact_d_seg(seg, render.detach(), seg_target)
                traj.append({"step": st, "exact_d_seg": d})
            if st == args.descent_steps:
                break
            opt.zero_grad()
            out = grad_model(render)
            # minimize CE-to-GT-target => DECREASE argmax flips => DESCEND d_seg
            loss = F.cross_entropy(out, target)
            loss.backward()
            opt.step()
            with torch.no_grad():
                render.clamp_(0, 255)
        return traj

    print("Descent A (full SegNet gradient)...")
    traj_a = run_descent(lambda r: seg(r), "full")
    print("Descent B (student surrogate gradient)...")
    traj_b = run_descent(lambda r: student(r), "student")

    print(f"\n{'step':>5} {'A exact_d_seg (full grad)':>26} {'B exact_d_seg (student grad)':>28}")
    for a, b in zip(traj_a, traj_b):
        print(f"{a['step']:>5} {a['exact_d_seg']:>26.5f} {b['exact_d_seg']:>28.5f}")

    final_a = traj_a[-1]["exact_d_seg"]
    final_b = traj_b[-1]["exact_d_seg"]
    gap = abs(final_a - final_b)
    print(f"\nFINAL exact d_seg: A(full)={final_a:.5f}  B(student)={final_b:.5f}  abs gap={gap:.5f}")
    verdict = ("DESCENT-EQUIVALENT (student grad reaches same exact-d_seg basin)"
               if gap < 0.02 else
               "NOT EQUIVALENT (student grad diverges from full-scorer basin)")
    print(f"VERDICT: {verdict}")

    result = {
        "frames": args.frames,
        "student_channels": args.student_channels,
        "seg_params_M": seg_params / 1e6,
        "student_params_M": stu_params / 1e6,
        "param_ratio": seg_params / stu_params,
        "distill_final_argmax_agree": final_agree,
        "full_step_s": full_dt,
        "student_step_s": stu_dt,
        "student_speedup": full_dt / stu_dt,
        "descent_traj_full": traj_a,
        "descent_traj_student": traj_b,
        "final_exact_d_seg_full": final_a,
        "final_exact_d_seg_student": final_b,
        "abs_gap": gap,
        "verdict": verdict,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
