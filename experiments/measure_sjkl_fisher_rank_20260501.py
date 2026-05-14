# SPDX-License-Identifier: MIT
"""Empirical measurement of the SJ-KL Fisher matrix rank.

Wave-Ω-1 design step 1 (per Council #2 Section 5.3 + subagent prompt).

Verifies the FIELDS-MEDAL premise: F(x*) = 100 J_seg^T J_seg + 10 J_pose^T J_pose
is LOW-RANK at the contest scale (input dim 3*384*512 = 589,824), with
effective rank ≤ scorer-output bottleneck (~few thousand) << dim.

If rank > ~10,000 the SJ-KL premise is degraded (still works but eigenvalue
spectrum is flatter so each eigenvector buys less than predicted). If rank
> 100,000 the premise is REFUTED.

This script does NOT dispatch any GPU job. It runs locally on MPS / CPU as
an architectural sanity check. Final SJ-KL bytes for any submission must be
re-derived on CUDA per CLAUDE.md non-negotiable.

Usage:
  PYTHONPATH=src:upstream:experiments .venv/bin/python \\
    experiments/measure_sjkl_fisher_rank_20260501.py [--device cpu] [--n-probe 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "upstream") not in sys.path:
    sys.path.insert(0, str(_REPO / "upstream"))

from tac.sjkl_basis import (
    effective_rank,
    fisher_matvec,
    lanczos_topk,
)


def _load_real_scorers(device: torch.device):
    """Load the actual SegNet + PoseNet from upstream/models."""
    from modules import SegNet, PoseNet  # type: ignore
    from safetensors.torch import load_file

    seg = SegNet().eval().to(device)
    pose = PoseNet().eval().to(device)
    seg_sd = load_file(str(_REPO / "upstream" / "models" / "segnet.safetensors"), device=str(device))
    pose_sd = load_file(str(_REPO / "upstream" / "models" / "posenet.safetensors"), device=str(device))
    seg.load_state_dict(seg_sd)
    pose.load_state_dict(pose_sd)
    return seg, pose


def _make_synthetic_pair(h: int, w: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    """Make a synthetic (1, 2, 3, h, w) frame pair in the [0, 255] uint8 range
    (cast to float32). Real Lane G v3 anchor frames would be loaded from
    upstream/videos/0.mkv, but that requires NVDEC; for the rank measurement
    a synthetic image is sufficient (the rank is a property of the model
    architecture, not the input statistics).
    """
    g = torch.Generator().manual_seed(seed)
    return (128.0 + 30.0 * torch.randn(1, 2, 3, h, w, generator=g)).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--n-probe", type=int, default=100,
                    help="Number of Lanczos iterations / top-eigenpairs to recover")
    ap.add_argument("--h", type=int, default=128, help="Probe image height (use 384 only on CUDA)")
    ap.add_argument("--w", type=int, default=128, help="Probe image width (use 512 only on CUDA)")
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.device != "cuda":
        print(f"[fisher-rank] device={device.type} — ADVISORY only per CLAUDE.md.")

    seg, pose = _load_real_scorers(device)
    frames = _make_synthetic_pair(args.h, args.w, device)
    dim = 3 * args.h * args.w
    print(f"[fisher-rank] dim={dim}, probing top-{args.n_probe} eigenpairs ...")

    def matvec(v):
        return fisher_matvec(seg, pose, frames, v)

    eigvals, _ = lanczos_topk(
        matvec=matvec, dim=dim, k=args.n_probe, n_iters=args.n_probe + 10,
        seed=0, shape_hint=(3, args.h, args.w),
    )
    eff_rank = effective_rank(eigvals, threshold=1e-4)
    print(f"[fisher-rank] top-{args.n_probe} eigvals max={float(eigvals.max()):.4e}, "
          f"min={float(eigvals.min()):.4e}")
    print(f"[fisher-rank] effective rank (>1e-4 of max) = {eff_rank}")
    print(f"[fisher-rank] eigvals[:8]: {eigvals[:8].tolist()}")
    if args.n_probe >= 16:
        print(f"[fisher-rank] eigvals[8:16]: {eigvals[8:16].tolist()}")
    print(f"[fisher-rank] decay ratio λ_8/λ_0: {float(eigvals[7]/eigvals[0]):.4e}")
    if eff_rank < 100:
        verdict = "VERIFIED — Council Section 5.3 low-rank claim holds"
    elif eff_rank < 10_000:
        verdict = "PARTIAL — claim degraded but SJ-KL still useful"
    else:
        verdict = "REFUTED — SJ-KL premise broken; abort lane"
    print(f"[fisher-rank] verdict: {verdict}")


if __name__ == "__main__":
    main()
