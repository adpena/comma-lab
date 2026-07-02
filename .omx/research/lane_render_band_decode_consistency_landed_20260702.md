# Lane render-band decode-consistency LANDED + measured rate wall (#224 Wave E, fork B)

**Axis:** `[macOS-CPU advisory] NON-PROMOTABLE`. Pointer UNMOVED 0.19110. Advisory/build-only.
**Date:** 2026-07-02. **Closes:** R5_BLOCK `lane_band_train_only_phantom_fork` (fork B).

## What landed (the phantom is CLOSED)

R5 found the analytic-lane render-band was TRAIN-ONLY: the training composite fits per-pair
`LaneLine`s from the GT class-1 argmax (NOT decode-available) and composites a band over the render,
but the shipped inflate had NO band code → a witness verdicted WITH the band would score WITHOUT it →
phantom. Fork B makes it decode-consistent (rule 118):

- **COUNTED (archive.zip, 5th LVLS1 block):** the per-pair lane MANIFOLD COORDS — `LaneLine`
  centerline (deg≤3) + halfwidth (deg 1) + dash (period,phase,duty) + forward_range, as bit-exact
  float64, brotli'd. Canonical `serialize_lane_band`/`deserialize_lane_band` in
  `src/tac/boundary_math/analytic_lane_render_band.py`.
- **FREE (inflate.py, 0 bytes):** the AA-SDF range-dependent coverage rasterizer
  (`rasterize_lane_coverage_range_dependent`) + the composite (`composite_band_on_render`) +
  the witness-margin uncertainty gate (`witness_uncertainty_mask`) — inlined op-for-op in the shipped
  `_INFLATE_PY` (`_lane_parse`/`_lane_coverage`/`_lane_composite`), and reproduced in the numpy-fp32
  oracle (`levelset_band_forward_numpy` superset returning lane_rgb + softmax margin).
- **NO GT mask, NO scorer weights, NO per-pixel table ship.** No smuggling into inflate "code".

## The DECODE-CONSISTENCY PROOF (bit-identical)

`bit_exact_roundtrip_gate` (band-ON) proves **shipped inflate.py band render == numpy-fp32 oracle band
render, max_abs_uint8_diff=0**, for BOTH coverage-only (c_range) AND witness-margin-uncertainty
(c_full_wit) forms. Covered by `experiments/tests/test_levelset_lane_band_decode_consistency.py`
(15 tests) + serialize↔deserialize bit-exact + coverage(train lines)==coverage(decoded lines)
bit-identical. numpy-fp32 is the CLAUDE.md deterministic authority; the torch measurement tool
(`tools/levelset_analytic_lane_band_dseg_n600.py`, fp32) is a ~0.9997-parity advisory of the same band.
`c_full_gt` (GT-SegNet-margin uncertainty) is deliberately UNSUPPORTED — it needs GT at decode → not
decode-consistent.

Default-off is BYTE-IDENTICAL to the pre-Wave-E 4-block grammar (no 5th block, no manifest key). All
22 pre-existing byte-close tests + 15 new + film/lane/yousfi lever tests green (91 total).

## THE MEASURED RATE WALL (honest, the reason fork B is decode-consistent but NOT yet net-positive)

Fitting the REAL `gt_n6` (6 real pairs, 5.0 lines/pair, band_recall 0.609):

| mode | raw bytes/6pair | COUNTED brotli | bytes/pair | EXTRAP n600 brotli | rate_term += |
|---|---|---|---|---|---|
| coverage-only | 3874 | 2203 | 367.2 | ~220,300 | **+0.14669** |
| witness-margin u_mask | 3919 | 2232 | 372.0 | ~223,200 | **+0.14862** |

The u_mask is a manifest scalar (negligible bytes). **~+0.147 rate at n600 is rate-dominating** (the
whole frontier is 0.19110; the current archive ~83 KB ≈ 0.055 rate). The naive per-pair float64
serialization means the band's d_seg win (FEED-dv, if net-positive after FP-killers) CANNOT beat the
rate cost as-serialized. So: **PHANTOM CLOSED (the shipped score is now honest — band d_seg AND band
rate both real), but the band is NOT yet rate-viable.**

## Rate levers to make it net-positive (follow-up, unmeasured)

1. **Temporal-delta across pairs** — lanes vary slowly frame-to-frame; store pair-0 lines + small
   per-pair deltas (est. 5–10× cut). Biggest lever.
2. **float32 / int-quantized coeffs** — the rasterizer casts to float64; storing float32 (or int16)
   coeffs, rasterized from the QUANTIZED values on BOTH oracle+inline, stays bit-exact and ~halves
   bytes (float64→float32). Cheap, low-risk.
3. **Fewer lines** — the fit emits ~5 lines/pair; cap to the 2–3 dominant lanes.
4. **Shared lane geometry** — one lane model + per-pair ego-pose offset (the screw ξ, dual-use).

## Files
- `src/tac/boundary_math/analytic_lane_render_band.py` — serialize/deserialize + `LaneBandRenderConfig`
  + `band_alpha`/`composite_band_on_render` + `build_lane_band_pairs_from_lstars`.
- `src/tac/boundary_math/lever_b_levelset_generator.py` — `levelset_band_forward_numpy`.
- `tools/levelset_byte_close_and_eval.py` — LVLS1 5th block + inline inflate band + numpy-oracle band +
  `--lane-render-band` (+ `--lane-band-{softness,dash-forward-max,weight,lane-cls,umask,tau,eps}`).
- `experiments/tests/test_levelset_lane_band_decode_consistency.py` — 15 decode-consistency tests.

## Commits (per-milestone)
- M1 `5429b6671` — canonical serialize/deserialize + band forward.
- M2 `bd38d80fe` — byte-close 5th block + inflate reproduction + numpy-oracle band + bit-exact gate.
- M4 `a41b4c299` — 15 decode-consistency tests.
