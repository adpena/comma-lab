# latest report

## current state - 2026-04-15

**CRITICAL BUG DISCOVERED AND FIXED**: TTO PoseNet gradients were ZERO.

### The bug

Upstream `frame_utils.py:rgb_to_yuv6()` has `@torch.no_grad()`. PoseNet's `preprocess_input()` calls this function. The training pipeline had a fix (patched scorer loading), but the TTO pipeline loaded scorers through a different code path without the fix. Every TTO experiment in the project ran with zero PoseNet gradients — the optimizer was blind to PoseNet.

### How it was found

Skunkworks council adversarial review of TTO v3 results. The Contrarian noted that 50 TTO steps made PoseNet *worse*, which is mathematically impossible if gradients are correct. Hotz traced the call chain to the `@torch.no_grad()` decorator. 13-0 council vote to fix.

### Impact

- All TTO results (v1-v4) are **invalidated** for PoseNet — they were SegNet-only optimizations
- Renderer training (auth=0.87) is **unaffected** — different code path had the fix
- Projected score with working TTO: **0.87 -> ~0.35** (PoseNet 0.031 -> ~0.003)

### Why it matters

A single decorator in upstream code silently broke an entire optimization pipeline for weeks. No errors, no warnings, no NaN — just suboptimal convergence masked by SegNet improvements. This is paper-worthy as a general failure mode for gradient-based optimization pipelines that compose functions from third-party code.

---

## renderer baseline (current best)

- Track: `robust_current`
- Variant: `asym_v5_lagrangian_fixed`
- Platform: `modal_t4`
- Auth score: **`0.87`** (seg=0.21, pose=0.56, rate=0.10)
- Checkpoint: `renderer_best.pt` at ep12600

## authoritative promoted floor (legacy)

- Track: `robust_current`
- Variant: `dilated_h64`
- Platform: `modal_a10g`
- Current-workflow score: **`1.33`** at `864,167` bytes
- Distortions: PoseNet `0.00218374`, SegNet `0.00609921`
- Rate: `0.02301653`
- Evidence: `reports/raw/2026-04-10-dilated-h64-authoritative/robust_current-dilated-h64-authoritative-cpu-report.txt`
