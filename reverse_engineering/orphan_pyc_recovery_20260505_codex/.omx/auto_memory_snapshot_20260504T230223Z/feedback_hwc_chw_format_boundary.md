---
name: HWC/CHW format mismatch — verify tensor shapes at EVERY format boundary
description: F.interpolate expects (B,C,H,W) CHW but pair data uses (B,T,H,W,C) HWC. Reshape without permute produces garbage silently. Caused 50x scorer inflation in training.py postfilter roundtrip.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

The HWC/CHW format mismatch is one of the most insidious bugs in PyTorch vision code.
F.interpolate and most nn modules expect (B, C, H, W) — but pair-based data often uses
(B, T, H, W, C) for memory-contiguous frame storage. When you reshape without permuting,
dim 2 (H=384) gets treated as channels, and the bilinear interpolation runs on spatial
dimensions that are actually (W=512, C=3) — a 512x3 image instead of 384x512.
The interpolation "succeeds" (no error) but produces complete garbage.
Always verify tensor shapes at every format boundary.

**Why:** training.py Trainer's eval_roundtrip code had a comment saying
"filtered is (1, 2, 3, H, W)" but _apply_filter_to_pair actually returns
(1, 2, H, W, 3) in HWC format. The reshape without permute caused
simulate_eval_roundtrip to operate on garbled data. Baseline scorer jumped
from 1.70 (correct) to 86.2 (garbage). Took investigation to find because
PyTorch raised NO error — bilinear interpolation on (384, 512, 3) just
produced a garbled (874, 1164, 3) tensor without complaint.

**How to apply:**
- When writing ANY code that bridges HWC↔CHW, add an explicit shape assertion:
  `assert tensor.shape[1] == 3, f"Expected CHW, got {tensor.shape}"`
- At every call to F.interpolate, simulate_eval_roundtrip, or nn.Conv2d:
  verify the input is (B, C, H, W) with C in {1, 3, 6}
- When reshaping pair data (B, T, H, W, C) for per-frame ops:
  ALWAYS `.permute(0, 3, 1, 2)` to get CHW, NEVER just `.reshape()`
- After the op, `.permute(0, 2, 3, 1)` back to HWC
- Trust the code, not the comments — verify shapes empirically
