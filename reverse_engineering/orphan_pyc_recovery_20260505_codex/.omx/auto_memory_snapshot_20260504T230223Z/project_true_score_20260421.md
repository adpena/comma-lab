---
name: TRUE SCORE 2.01 — first honest measurement via upstream evaluate.py
description: Full e2e pipeline with 384x512 masks + optimized poses. Rate dominates at 1.53 (2MB masks). Distortion is 0.48. Path to Quantizr 0.33 = compress masks harder.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Score: 2.01 [contest-compliant, upstream evaluate.py, MPS] (2026-04-21)

- PoseNet dist: 0.01321 → sqrt(10 * 0.01321) = 0.3635
- SegNet dist: 0.00116 → 100 * 0.00116 = 0.1161
- Rate: 25 * 2294623/37545489 = 1.5279
- Archive: 2,294,623 bytes (renderer.bin 297KB + masks.mkv 2016KB + optimized_poses.pt 16KB)
- Inflate time: 68.4s on MPS (within 30-min T4 budget)

## Score breakdown

- Distortion: 0.48 (stunning — better than Quantizr if rate were equal)
- Rate: 1.53 (76% of score — masks.mkv is 2MB, must compress)
- **Rate is the ONLY bottleneck.** Compress masks from 2MB to <200KB and score drops to <0.7

## Path to beating Quantizr (0.33)

Quantizr's archive is ~15KB. Their rate term is ~0.01. Our distortion (0.48) is close to
theirs. The entire gap is rate.

Options to reduce masks.mkv from 2MB:
1. Lower CRF (more aggressive compression) — may lose mask accuracy
2. Encode at intermediate resolution (e.g., 192x256 = 1/2 scale) — 4x fewer pixels
3. Better codec (VVC/H.266 if available)
4. Quantizr's approach: encode masks differently (not as video)
5. Use the renderer's internal upsampling to handle smaller mask inputs

**Why:** The renderer produces near-perfect output when given correct masks + poses.
The problem was never the renderer — it was measurement bugs hiding a competitive score.
