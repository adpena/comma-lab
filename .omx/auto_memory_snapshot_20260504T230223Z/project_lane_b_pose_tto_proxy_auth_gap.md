---
name: LANE-B pose TTO HURT score on dilated-h64 baseline
description: 6.5h pose TTO on the dilated-h64 baseline drove proxy PoseNet to 0.0007 but contest-CUDA measured 0.246 — a 350x proxy-auth gap despite noise_std=0.5 + eval_roundtrip=True. Final score 2.40 (vs 0.90 baseline).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
LANE-B (instance 35627136, 2026-04-26): the first pose TTO run after the noise_std=0.5 + eval_roundtrip=True fixes. Used `optimize_poses.py` against the dilated-h64 baseline `renderer.bin` (the one that hit 0.90 with the original poses).

**Result (contest-CUDA auth eval):**
- PoseNet dist: 0.246 → score 1.570 (vs baseline 0.0107 → 0.327)
- SegNet dist: 0.0037 → score 0.369
- Rate: 0.018 (685KB archive) → score 0.457
- **FINAL: 2.40** (worse than 0.90 baseline by 1.5 points)

Pose TTO did NOT improve the score; it made it dramatically worse on this arch.

**Why:** `optimize_poses.py` reported proxy PoseNet 0.0007 across all 75 pairs. Contest-CUDA measured 0.246. That's a **350x proxy-auth gap** — the same gap class as `feedback_proxy_auth_gap_835x` even with `noise_std=0.5` + `eval_roundtrip=True` already wired and threaded.

**How to apply:**
1. **Do NOT run pose TTO blind on a new renderer architecture.** Run a 50-pair smoke first, build archive, run auth eval — catch the gap BEFORE burning 6.5h.
2. **Hypotheses to test (task #122):** (a) renderer arch incompat — `motion.head` channel layout might not match what `optimize_poses` assumes; (b) chroma/YUV6 numerics on dilated-h64 (FastViT-T12 differs from EfficientNet); (c) ego_flow gating not threaded for dilated-h64; (d) `eval_roundtrip` resize math (384→874→uint8→384) misses something specific to dilated convs.
3. **Archive size also bad:** 685KB vs Quantizr 293KB. Masks.mkv at 421KB needs AV1 + half-frame to halve.
4. **The 0.90 baseline poses must be doing something right** that fresh TTO destroyed. Maybe the baseline used analytical poses (camera model + zoom from FoE) instead of learned. Investigate before re-running.
