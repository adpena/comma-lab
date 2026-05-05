# codex_metric_yshift_av1 deep audit — pixel-translation sidechannel (2026-05-04)

## Discovery

Audited `experiments/results/public_pr100_intake_20260504_codex/source/submissions/codex_metric_yshift_av1/inflate.py:579-596` (the sidechannel apply logic). Found a NOVEL pattern: per-frame (dy, dx) pixel translation as a correction sidechannel — **directly portable to PR106 with no architecture change required**.

## SC01 sidechannel actual modes

```python
SIDECHANNEL_HEADER = struct.Struct("<4sBBIf")  # magic + mode + n_channels + n_frames + step
SIDECHANNEL_MODE_Y_SAT   = 6  # per-frame: Y_offset + saturation_delta
SIDECHANNEL_MODE_Y_SHIFT = 7  # per-frame: Y_offset + dy_pixels + dx_pixels  ← NEW
SIDECHANNEL_SAT_RAW_STEP = 0.01

# Apply at inflate (line 579-596):
if mode_id == SIDECHANNEL_MODE_Y_SAT:
    raw = sidechannel["raw"][frame_idx]  # 2 channels
    x = x + raw[0] * step                # additive Y offset
    sat = 1.0 + raw[1] * 0.01
    y = luma_plane(x).unsqueeze(2)
    x = y + sat * (x - y)                # saturation around luma

elif mode_id == SIDECHANNEL_MODE_Y_SHIFT:
    raw = sidechannel["raw"][frame_idx]  # 3 channels
    x = x + raw[0] * step                # additive Y offset
    dy = int(round(float(raw[1])))       # pixel translation in Y direction
    dx = int(round(float(raw[2])))       # pixel translation in X direction
    x = shift_rgb(x, x, dy, dx).float()  # ← novel: integer pixel translation
```

## Why pixel translation matters

PR106 decoder pipeline: HNeRV decodes at (384, 512) → bicubic upsample to (874, 1164). Any systematic alignment error from the upsample would be punished by SegNet (which compares masks at 384x512 but eval pipeline uses upscaled output) and PoseNet (which is sensitive to pixel-level features).

A per-frame (dy, dx) integer translation is a tiny sidechannel:
- 1200 frames × 3 channels × 1 byte = 3,600 bytes raw
- Brotli compresses well (mostly 0 with sparse non-zero corrections)
- Estimated final size: ~1-2 KB

If the alignment error is systematic and scorer-detectable, this could be a measurable distortion improvement at sub-noise rate cost.

## How this fits the paradigm catalogue

Adds variant #5 to the score-aware sidechannel paradigm:

| # | Implementation | Granularity | Mechanism |
|---:|---|---|---|
| 1 | PR100 hnerv_lc_v2 | per-pair (1 dim) | latent additive correction |
| 2 | codex_metric_yshift Y_SAT | per-frame (2 ch) | Y offset + saturation |
| 3 | codex_metric_yshift Y_SHIFT | per-frame (3 ch) | Y offset + (dy, dx) pixel translation |
| 4 | Lane SJ-KL Fisher | per-frame (K coefs) | low-rank pixel residual |
| 5 | qpose14 seg_tile_actions | per-tile (codebook) | YUV codebook per tile |

The pixel-translation pattern (#3) is the **simplest sub-pixel-alignment correction**. It's also the most directly portable to PR106:
- Doesn't need architecture changes (works on any decoded RGB frame)
- Smaller sidechannel than per-pair latent correction (~1-2 KB vs ~1.2 KB but with ZERO architectural assumptions)
- Per-frame (not per-pair) — applies uniformly to fake1 + fake2 within a pair

## Predicted gain on PR106

Hard to estimate without running the search. Bounding cases:
- **Lower bound**: 0 (if PR106 has no systematic alignment error)
- **Upper bound**: similar to PR100 sidecar's empirical ~0.00218 score Δ on a comparable architecture (also HNeRV-family)
- **Most likely**: -0.0005 to -0.002 score Δ for ~1.5KB sidechannel cost (~+0.001 rate Δ)

Net: probably -0.0005 to -0.0015 score Δ. Smaller than the per-pair latent sidecar (~-0.00218) but ALSO stackable with it.

## Stacking analysis

Both PR100 sidecar (#1) and Y-shift (#3) could stack on PR106 if they correct different error modes:
- PR100 sidecar: latent-space correction (which dim of the 28 to perturb per pair)
- Y-shift: post-decode pixel-space correction (sub-pixel alignment)

These should be **roughly orthogonal** (different forward-pass stages). If true, stacked gain ≈ -0.003 score Δ at ~3KB total sidechannel cost.

## Decision

**SECONDARY PROPOSAL** — defer to council gate alongside the primary PR100 sidecar lane (`lane_pr106_latent_sidecar`). Don't implement until that lane lands an empirical result; if PR100 sidecar wins as predicted, Y-shift becomes the natural next stack-on lane.

Implementation cost when ready:
- Stage 1 (CPU + scorer, ~30 min @ Vast.ai 4090): brute-force per-frame (dy, dx, Y_offset) search minimizing distortion. Search space: dy ∈ [-3, 3], dx ∈ [-3, 3], Y_offset ∈ [-127, 127] = 49 × 255 = 12,495 candidates per frame × 1200 frames = 15M scorer evals. Tractable on A100.
- Stage 2 (CPU, ~5 min): wire SC01 sidechannel as 6th section in apogee_intN 0.bin layout.
- Stage 3 (CUDA, ~$0.30): contest auth eval.

Total cost: ~$0.60 wall-clock when council approves.

## Cross-refs

- codex_metric_yshift inflate source: `experiments/results/public_pr100_intake_20260504_codex/source/submissions/codex_metric_yshift_av1/inflate.py:579-596`
- shift_rgb implementation: in same file, search for `def shift_rgb`
- Sister memos: `docs/score_aware_sidechannel_paradigm_20260504.md` + `docs/qpose14_seg_tile_actions_paradigm_extension_20260504.md`
- Primary lead lane: `docs/pr100_latent_sidecar_porting_proposal_20260504.md` (`lane_pr106_latent_sidecar` at L1)
- LATENT_LUMA_MAGIC = "LRL1": secondary sidechannel for latent-space luma reconstruction; not yet audited (NOTE for future)
