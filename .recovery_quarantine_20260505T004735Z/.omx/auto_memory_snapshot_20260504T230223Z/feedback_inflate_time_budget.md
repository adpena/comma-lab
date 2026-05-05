---
name: Inflate Time Budget — Only 5-12 Minutes on T4 (Not 30)
description: The 30-min eval timeout covers the ENTIRE job. Scoring takes 15-20 min. Setup 3-5 min. Only 5-12 min left for inflation.
type: feedback
originSessionId: 47bf3dd8-df75-4271-9ce1-428c19c2eb32
---
The 30-minute eval timeout covers the ENTIRE GitHub Actions job, not just inflation.

**Revised budget breakdown (from contest rules research 2026-04-15):**
- Checkout + git-lfs pull + uv sync + ffmpeg: ~3-5 min
- Inflation (our code): **~20-25 min available**
- Scoring (evaluate.py): ~3-5 min on T4 (faster than initially assumed)
- Total: 30 min timeout

**How to apply:**
- Renderer + TTO path works: renderer forward pass is ~2 seconds for all 1200 frames
- Pre-computed TTO frames (stored as tto_frames.pt in archive) are instant to load
- Constrained gen from scratch is a research contribution, NOT a practical submission
- Any inflate-time optimization must fit in 5-12 minutes
- On T4 at ~200ms/step: max ~3000-3600 gradient steps total across all pairs
- That's 5-6 steps per pair — insufficient for constrained gen quality
- The eval runner can be T4 OR CPU (ubuntu-latest) — CPU is even slower

**Rate optimization strategy given this constraint:**
- Keep renderer weights in archive (2-second forward pass is irreplaceable)
- Reduce archive size via weight quantization/pruning (150KB → 75-100KB)
- Pre-compute TTO frames at compress time (unlimited budget) and store in archive
- The renderer + pre-computed TTO path is the ONLY practical submission path on T4
