# Revival plan: Lane Ω-W-V3: water-fill bit allocation on PR106 HNeRV decoder

**Gem**: `src/tac/water_filling_codec_v2.py`
**ID**: `01_water_filling_codec_v2_pr106_decoder`

## Current state

v2 codec is production-hardened (level 4). 40.98% byte savings empirical on Lane G v3 renderer.bin (88K-param). Tested via `experiments/results/lane_g_v3_owv3_*` lineage.

## Files touched

- experiments/profile_pr106_decoder.py (new): extract HNeRV decoder weights into a flat tensor
- src/tac/water_filling_codec_v2.py (no changes — reuse)
- submissions/apogee_v2/inflate.sh (new sibling of apogee inflate)
- submissions/apogee_v2/inflate.py (new — uses water-filling codec to decode)

## Integration sketch

1. Extract `decoder.state_dict()` from PR106 0.bin (170KB brotli inflate path).
2. Compute β-Fisher sensitivity map on the decoder over the contest video pairs.
3. Run `water_filling_v2.allocate(state_dict, sensitivity_map, target_bytes=145000)`.
4. Pack via codec layout. New 0.bin written ≤165KB total (incl. latents).
5. Inflate.py reverses: load codec → reconstruct decoder → run forward.

## Test plan

- Smoke: round-trip a single decoder layer through codec, assert max-Δ < 1e-4.
- Integration: full inflate via .venv/bin/python inflate.sh ; verify single 0.bin output.
- Empirical: contest_auth_eval.py on the new archive — must score ≤ 0.20946 to ship.

## Predicted score basis

0.20946 (PR106) - 0.005 to -0.015 (Δ band from audit) → score band [0.194, 0.204]. Lower bound (0.194) realistic if HNeRV decoder has high redundancy similar to our 88K-param renderer. Upper bound conservative if HNeRV is already entropy-near-optimal.

## What would change my mind

If the empirical T4 score is ≥0.20946, water-fill did not beat brotli on the HNeRV decoder layout — abandon and try block-FP instead.

## Blockers resolved in plan

- GPU $0 cap: sensitivity map can be computed on M5 Max MPS (`[advisory only]`) for design — but final score requires CUDA, deferred until next dispatch budget.

## Skunkworks council deliberation

Shannon LEAD + Dykstra CO-LEAD endorse; Selfcomp warns that overfit-to-our-renderer may not generalize; Hotz says 'just try it — 4 hours of work for potentially -0.01'.

**Verdict**: VOTE 8/10 GO; 1 concern (Selfcomp); 1 abstain (Quantizr — uninvolved with HNeRV)
