#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Localize the MPS PoseNet gradient drift: is it forward or backward, and WHICH layer?

The full-MPS scorer gradient FAILS descent-equivalence on the pose axis (d_pose gap
grows monotonically) while d_seg is bit-identical. Question (operator): can the
MPS-PoseNet drift be PATCHED (one fixable op -> recover full 104x) or is it
fundamental MPS numerics (-> split-by-head is the ceiling)?

This compares CPU vs MPS for PoseNet at three granularities:
  (1) FORWARD per-head output cosine + max-rel-error (is the forward already drifting?)
  (2) input-GRADIENT cosine (does backward drift beyond the forward?)
  (3) per-layer activation cosine via forward hooks (WHERE does divergence start?)

If the forward is clean but backward drifts -> a backward-kernel issue, maybe patchable.
If the forward already drifts at a specific layer -> localize that op for CPU-fallback/patch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, "upstream")

from tac.torch_mps_compat import patch_scorer_for_mps  # noqa: E402
from tac.score_aware_loop.targets import load_frozen_distortion_net  # noqa: E402
import frame_utils  # noqa: E402
import einops  # noqa: E402


def _cos(a, b):
    a = a.detach().float().cpu().flatten()
    b = b.detach().float().cpu().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _relmax(a, b):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    denom = a.abs().max().clamp_min(1e-8)
    return (a - b).abs().max().item() / denom.item()


def main():
    patch_scorer_for_mps()
    W, H = frame_utils.camera_size
    T = frame_utils.seq_len
    B = 4
    torch.manual_seed(0)
    x0 = torch.rand(B, T, H, W, 3)

    net_cpu = load_frozen_distortion_net(upstream_dir="upstream", device="cpu")
    net_mps = load_frozen_distortion_net(upstream_dir="upstream", device="mps")
    pose_cpu, pose_mps = net_cpu.posenet, net_mps.posenet

    def prep(x, dev):
        return pose_cpu.preprocess_input(
            einops.rearrange(x.to(dev), "b t h w c -> b t c h w").float()
        ) if dev == "cpu" else pose_mps.preprocess_input(
            einops.rearrange(x.to(dev), "b t h w c -> b t c h w").float()
        )

    # --- per-layer activation cosine (forward hooks) ---
    acts_cpu, acts_mps = {}, {}

    def mk_hook(store, name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                store[name] = t.detach().float().cpu()
        return hook

    hcpu = [m.register_forward_hook(mk_hook(acts_cpu, n)) for n, m in pose_cpu.named_modules() if n]
    hmps = [m.register_forward_hook(mk_hook(acts_mps, n)) for n, m in pose_mps.named_modules() if n]

    xc = einops.rearrange(x0, "b t h w c -> b t c h w").float()
    xm = xc.to("mps")
    out_cpu = pose_cpu(pose_cpu.preprocess_input(xc))
    out_mps = pose_mps(pose_mps.preprocess_input(xm))
    torch.mps.synchronize()
    for h in hcpu + hmps:
        h.remove()

    print("=== (1) FORWARD per-head output cosine / relmax (CPU vs MPS) ===")
    for k in out_cpu:
        print(f"  head {k:24s} cos={_cos(out_cpu[k], out_mps[k]):.6f}  relmax={_relmax(out_cpu[k], out_mps[k]):.4f}")

    print("\n=== (3) per-layer activation cosine — first divergences (lowest cosine) ===")
    rows = []
    for n in acts_cpu:
        if n in acts_mps and acts_cpu[n].shape == acts_mps[n].shape:
            rows.append((n, _cos(acts_cpu[n], acts_mps[n]), _relmax(acts_cpu[n], acts_mps[n])))
    rows.sort(key=lambda r: r[1])  # worst cosine first
    for n, c, r in rows[:18]:
        print(f"  {n:40s} cos={c:.6f}  relmax={r:.4f}")

    # --- input-gradient drift (backward) ---
    print("\n=== (2) input-GRADIENT cosine (backward) CPU vs MPS ===")
    xc2 = einops.rearrange(x0, "b t h w c -> b t c h w").float().requires_grad_(True)
    oc = pose_cpu(pose_cpu.preprocess_input(xc2))
    (sum(v.float().pow(2).mean() for v in oc.values())).backward()
    gc = xc2.grad.clone()

    xm2 = einops.rearrange(x0.to("mps"), "b t h w c -> b t c h w").float().requires_grad_(True)
    om = pose_mps(pose_mps.preprocess_input(xm2))
    (sum(v.float().pow(2).mean() for v in om.values())).backward()
    torch.mps.synchronize()
    gm = xm2.grad.clone()
    print(f"  input-grad cos={_cos(gc, gm):.6f}  relmax={_relmax(gc, gm):.4f}")
    print(f"  (forward output cos was ~{min(_cos(out_cpu[k], out_mps[k]) for k in out_cpu):.6f} worst-head)")


if __name__ == "__main__":
    main()
