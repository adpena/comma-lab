#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM-FC1 shared heavy input: cache the COPY-base decoder-side context, n600.

On the copy PREDICT base (warp family CLOSED per r2s), the decoder reconstructs frame_1 = frame_0.
Everything the correction stream may condition on is derivable from SegNet(f0) alone -- the decoder
has f0 and runs the (public) frozen SegNet. This caches, for all 600 pairs:

  - copy_argmax (600,384,512) uint8   : argmax(SegNet(f0))  == the copy-base class field
  - copy_margin (600,384,512) float16 : top1-top2 logit gap (the decoder-derivable confidence)

These are the STAGE-1 context inputs. lstars (the label to communicate) lives in the GT cache and is
NEVER read here (the encoder holds it; the decoder does not). Self-check downstream: flip mask =
(copy_argmax != lstars) must reproduce the oc1/r2s copy support EXACTLY (0.008642, 1,019,467 sites).

Chunked + resumable (skip existing chunk npz) so a background run survives the ~3-min foreground kill.
ONE SegNet job at a time on this host (~5 GB). `[macOS-CPU advisory]` -- no score claim.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "ddm_fc1_context_cache.v1"


def _load_segnet(repo_root: Path, weights: Path) -> Any:
    sys.path.insert(0, str(repo_root / "upstream"))
    import torch
    from modules import SegNet  # type: ignore[import-not-found]
    from safetensors.torch import load_file

    net = SegNet().eval()
    net.load_state_dict(load_file(str(weights)))
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 8)))
    return net


def _segnet_argmax_margin(net: Any, frames_bhwc_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (argmax uint8 (B,384,512), margin float16 (B,384,512)) for the last-frame contract.

    Feeds the exact upstream contract: (B, seq_len=1, C, H, W) float; SegNet uses the last frame,
    bilinear-resizes to (384,512), 5-class logits. margin = top1 - top2 logit (decoder-derivable).
    """
    import einops
    import torch

    x = torch.from_numpy(frames_bhwc_uint8).float()[:, None]  # (B,1,H,W,C)
    x = einops.rearrange(x, "b t h w c -> b t c h w")
    with torch.inference_mode():
        inp = net.preprocess_input(x)
        logits = net(inp)  # (B,5,384,512)
        top2 = torch.topk(logits, 2, dim=1).values  # (B,2,384,512)
        margin = (top2[:, 0] - top2[:, 1]).to(torch.float16).cpu().numpy()
        argmax = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
    return argmax, margin


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--segnet-weights", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=120, help="pairs per resumable chunk npz")
    ap.add_argument("--max-pairs", type=int, default=600)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(str(args.gt_cache))
    gt_f0 = cache["gt_f0"]  # (600,874,1164,3) uint8
    n = min(int(args.max_pairs), gt_f0.shape[0])

    net = None
    t0 = time.time()
    for c0 in range(0, n, args.chunk):
        c1 = min(c0 + args.chunk, n)
        out_path = args.out_dir / f"ctx_{c0:04d}_{c1:04d}.npz"
        if out_path.exists():
            print(f"[skip] {out_path.name} exists", flush=True)
            continue
        if net is None:
            net = _load_segnet(args.repo_root, args.segnet_weights)
        argmax_parts: list[np.ndarray] = []
        margin_parts: list[np.ndarray] = []
        for b0 in range(c0, c1, args.batch):
            b1 = min(b0 + args.batch, c1)
            am, mg = _segnet_argmax_margin(net, gt_f0[b0:b1])
            argmax_parts.append(am)
            margin_parts.append(mg)
            print(f"[chunk {c0}-{c1}] pairs {b0}-{b1} done ({time.time()-t0:.0f}s)", flush=True)
        argmax = np.concatenate(argmax_parts, axis=0)
        margin = np.concatenate(margin_parts, axis=0)
        tmp = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, copy_argmax=argmax, copy_margin=margin, start=c0, end=c1)
        tmp.replace(out_path)
        print(f"[write] {out_path.name} ({argmax.shape}) ({time.time()-t0:.0f}s)", flush=True)
    print(f"[done] all chunks up to {n} pairs cached in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
