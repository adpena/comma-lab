---
name: Quantizr PR#55 — Score 0.33 (New #1, April 19 2026)
description: Quantizr improved from 0.60 to 0.33. 88K params, FiLM on pose vectors, depthwise separable, single mask per pair.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Quantizr PR#55 (2026-04-19)

**Score: 0.33** — PoseNet=0.00051, SegNet=0.00061, Rate=0.008 (300KB archive)

### Key Technical Changes from mask2mask (PR#53, 0.60):
1. **Dropped optical flow** — uses Feature-wise Linear Modulation (FiLM) on pose vectors instead
2. **Single mask per pair** — only encodes mask_t1, not both. Halves mask video size, higher CRF possible.
3. **Depthwise separable convs** — 88K params (was ~300K). Model is 64KB compressed.
4. **Train with eval-matched resize roundtrip** — interpolate up → clamp/round → interpolate down. Gradients compensate for resampling blur and uint8 rounding.
5. **Shuffled frame pairs** in training (custom dataloader).

### What We Should Adopt:
- FiLM on pose vectors ≈ our latent-conditioned renderer idea (they use actual pose, we planned learned latent)
- Eval-matched resize in training (we have simulate_resize in TTO but NOT in renderer training)
- Depthwise separable convs for parameter efficiency
- Single mask per pair (we use both masks currently)

### What We Have That They Don't:
- Hinge loss (25% better SegNet optimization — they use standard training)
- Two-phase TTO (they don't do TTO at all)
- Embedding loss (512D gradient directions)
- Per-pair difficulty map

### Competitive Position:
- Quantizr: 0.33 [contest-compliant]
- Us: 0.87 [contest-compliant], 0.275 proxy ~0.35 auth [unlimited-compute]
- Gap: we need to improve contest-compliant from 0.87 to <0.33
- They say sub-0.30 is possible with conv dim sweep. They're done working on it.
