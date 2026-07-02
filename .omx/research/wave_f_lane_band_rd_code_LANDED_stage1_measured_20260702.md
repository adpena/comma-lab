# Wave-F — OPTIMAL lane-band RD code: Stage-1 LANDED + MEASURED (n600), Stage-2 L1 designed

**Status:** LANDED + MEASURED (2026-07-02). Advisory / build-only; **pointer 0.19110 UNMOVED** (moves only
via a byte-closed `upstream/evaluate.py` n600 exact row — Wave-F is the RATE ENABLER, not the d_seg win).
Design authority: `wave_f_optimal_lane_band_rd_code_design_20260702.md` (L1-L5). R1 synthesis:
`wave_f_lane_band_rd_research_synthesis_20260702.md`. R2 build design: `wave_f_lane_band_build_design_20260702.md`.

## What was built (Stage-1 CORE — complete, tested, decode-consistent)

Replaced the naive per-pair float64 lane serializer with the **OPTIMAL LBND2 rate-distortion codec**, INSIDE
the existing LVLS1 byte-close 5th-block grammar (R2's "build from scratch" was a grep false-negative on magic
`LBND`; the real byte-close magic is `LVLS1` and the Wave-E lane block already existed — I EXTENDED it):

- `src/tac/boundary_math/analytic_lane_render_band.py`: `serialize_lane_band_rd` / `deserialize_lane_band_rd`
  (magic `LBND2`), `derive_rd_base_steps` (L3 geometric-tolerance quantization), `_pack_pairs_to_matrix`
  (L4 lateral-sorted fixed slots + carry-forward hold), zigzag temporal-delta, `roundtrip_lines_through_rd`
  (measure-what-you-ship dequant), `serialize_lane_band_any`/`deserialize_lane_band_any` (LBND1/LBND2 magic
  dispatch), `lane_band_rd_rate_report` (measured per-lever bytes + Shannon floor + PTC1 + per-dim breakdown),
  `derive_task_rd_steps` (Stage-2 task-RD waterfill HOOK — consumes MEASURED sensitivities, never fabricates).
  `LBND1` kept intact for the default-off gate + the naive-vs-RD comparison.
- `tools/levelset_byte_close_and_eval.py`: inflate `_INFLATE_PY` gained `_lane_parse_rd` + `_lane_parse_any`
  (PURE numpy/struct/json — **ZERO new inflate dep**; brotli is the entropy backend, already present). Magic
  dispatch in `_setup` + both capped-inflate re-serialize paths (`run_inflate`, `bit_exact_roundtrip_gate`) are
  format-preserving. `build_lane_band_section(rd=True)` default; `--lane-band-naive` opt-out.
- `experiments/tests/test_levelset_lane_band_decode_consistency.py`: +11 tests (Group E). Full suite **26/26**.
- `tools/wave_f_lane_band_rd_rate_n600.py`: the measured-rate tool ($0, CPU, no scorer).

The codec: **L4** lateral-sort ragged lines into fixed slots (ego-lane + offsets), carry-forward hold for
absent slots → **L3** quantize each coeff to its OWN geometric-tolerance step (centerline per-power scaled by
`f_ref`, ~sub-px lateral) → **L2** temporal-delta across the 600 pairs → zigzag int32 → outer brotli (the
entropy backend). Bit-exact + decode-consistent by construction (both sides render the DEQUANTIZED lines).

## MEASURED rows (real `gt_n600.npz` lstars, `[macOS-CPU advisory]`, byte-closed serialization)

| axis | naive LBND1 | RD LBND2 (Stage-1) | ratio | Shannon floor | PTC1 range-coder |
|---|---:|---:|---:|---:|---:|
| **n600** bytes | 156,340 | **41,526** | 0.266 | 26,179 | 43,153 |
| **n600** rate_term | **0.1041** | **0.02765** | **3.76× less** | 0.0174 (bound) | 0.0287 |
| n96 bytes | 25,566 | 5,990 | 0.234 | 3,090 | 8,772 |

- **Stage-1 = 3.76× rate reduction** (n600 0.1041 → 0.02765). Naive was launch-unusable; RD is measurable.
- induced lateral RMS from quantization = **0.0212 m** (~2 cm, sub-pixel — well inside the argmax-band tol).
- **PTC1 / constriction range-coder is DOMINATED** at this data shape (43,153 B > brotli 41,526 B): the per-dim
  transmitted-PMF header cost (K*11 ≈ 66 dims × ~150 B/PMF ≈ 10 KB of headers) swamps the coded stream. **Brotli
  on zigzag-int32 is the correct backend** (KILLS the "add a range coder" gold-plating + keeps inflate dep-free).
  This is the honest "prefer solvable math, know the floor" answer, not a vibe.

## The CRUX (measured, sharpened) — Stage-1 misses the +0.005 floor; the residual is INFORMATION-bound

The +0.005 rate floor is 7,509 B; Stage-1 lands 41,526 B (0.02765). The **delta-stream Shannon floor is 26,179 B
(0.0174)** — so **even a PERFECT entropy coder on the camera-frame per-pair coeffs cannot cross +0.005.** The
residual is information-bound, not coding-bound. The per-dim-type breakdown (`delta_floor_bytes_per_dim_type`)
shows it is DISTRIBUTED across all coeff types: ~11.3 KB ego-swept centerline curvature (c3/c2/c1/c0) + ~5.7 KB
halfwidth + ~6 KB dash + ~3.2 KB forward_range. Two components:
1. **Ego-swept centerline curvature** (~11 KB): the lane geometry in the CAMERA frame changes every frame because
   the ego drives forward (the lane sweeps toward the car). This is exactly what **L1 SE(3) ego-factorization**
   removes (warp to the static world frame).
2. **Per-frame fit jitter** (~15 KB across hw/dash/fwd/c0): each pair is fit INDEPENDENTLY from the SegNet argmax,
   so the "same" world lane refits slightly differently each frame.

**The decisive insight: L1 is not merely a coding transform — it is a RE-PARAMETERIZATION of the source.** Doing
L1 right (transform all frames to the world frame, fit the world lane ONCE + a smooth ego trajectory, code the
tiny per-frame innovation) collapses BOTH the ego sweep AND the fit jitter → the research ~1-4 KB (~0.001) target.
Stage-1's per-frame-independent fit is why it plateaus at 0.0174 (floor) / 0.0276 (brotli). **This is the correct
next lever, and it is scorer-free (pure geometry).**

## rule-118 / NO-FAKE accounting (binding, honored)

- **COUNTED** (archive.zip): the quantized per-coeff temporal-delta stream + the presence bitmap. n600 = 41,526 B.
- **FREE** (inflate.py generic algorithm): quantize/dequantize, `rasterize_lane_coverage_range_dependent`, the
  composite, the zigzag/cumsum, brotli decode. NO GT mask, NO scorer weights, NO per-pixel table ship.
- Steps are **DERIVED** from a geometric tolerance (`derive_rd_base_steps`), never a fabricated number.
- The d_seg WIN is **OUT OF SCOPE** (that is the #205 trained-in run; a post-hoc band is break-even). Wave-F's
  deliverable is the rate-viable, decode-consistent band + the measured rate. **NOT a score claim.**

## Acceptance gates (all HONORED)

1. **Decode-consistency PRESERVED** — ✅ the shipped inflate LBND2 band render == the numpy-fp32 oracle render,
   **bit-for-bit** (`max_abs_uint8_diff == 0`), coverage-only AND witness-margin-umask, through the REAL inflate
   subprocess (`test_rd_bit_exact_gate_*`). The Wave-E gate re-proven with the new code.
2. **Default-off byte-identical 7/7** — ✅ untouched (RD is band-on-only; `wire_in_224_byte_identical_smoke.py`
   PASS; `test_rd_default_off_still_byte_identical` PASS).
3. **rule-118** — ✅ counted vs free split above; ego derived-vs-stored N/A at Stage-1 (no ξ yet).
4. **MEASURED rate @ n600** — ✅ real byte-closed serialization, reported per lever (naive/RD/floor/PTC1).
5. **Determinism + 30-min budget** — ✅ decode is O(1)/pixel numpy + the existing parallel inflate; deterministic.

## Stage-2 L1 — the SE(3) ego-factorization (DESIGNED; next build, scorer-free geometry)

Per operator/coordinator 2026-07-02 (unified-ξ + numpy-fp64-decode constraints):
- **ONE `tac.lie.se3_bspline` twist ξ_ego(t)** (~O(10-48) DOF for the whole drive), stored ONCE (counted, tiny),
  **triple-use**: (i) IS the pose (warp real keyframe luma → d_pose; reuse `warp_real_luma_frame0.py`'s twist
  storage, NOT the 6-vector), (ii) advects the argmax Morse-Smale complex to the static world frame (L1 lane
  rate), (iii) partition temporal-consistency. It **REPLACES** the `scorer_targets.py` 6-vector pose sidecar
  (that 6-vector is the fallback, not the plan). rule-118: ξ counted once, attributed across pose+lanes.
- **Re-parameterize the source**: warp all 600 frames' class-1 pixels to the world frame via ξ, fit the world
  lane geometry ONCE (near-static) + code the tiny per-frame innovation. Collapses the ~11 KB ego sweep AND the
  ~15 KB fit jitter (the Stage-1 residual).
- **Decode = numpy fp64 authority, ZERO mlx/metal in inflate**: the SE(3) ego-warp at decode uses
  `tac.lie._se3_numpy` + `warp_real_luma_frame0.homography_from_xi_numpy` (the fp64 oracle); the MLX/Metal path
  is training-only, matched by parity (~0.9997); the bit-exact gate is numpy-inflate == numpy-oracle
  (`max_abs_uint8_diff == 0`), exactly as Stage-1 did for the rasterizer.
- **Main build risk (R2)**: the ego warp/unwarp must be an EXACT algebraic inverse pair ON THE QUANTIZED GRID
  (`unwarp(warp(x)) == x` bit-exact), else fall back to STORING ξ (counted) so decode never re-derives — MEASURE
  which wins S. This is its own bit-exact-inverse determinism cycle (why it is a separate careful build, not
  rushed into Stage-1 — a rushed broken bit-exact gate would be a FAKE).
- **L3 task-RD** (`derive_task_rd_steps` hook, landed): refine the per-coeff steps by MEASURED ∂d_seg/∂coeff
  (finite-diff through R + frozen SegNet) via `frontier_exact_bitalloc.waterfill_bit_allocation` to the KKT
  operating point `∂d_seg/∂byte = 25/(100·37.5M)`. Needs the scorer → measured follow-up (never fabricated).

## Honest verdict

Stage-1 is REAL, complete, tested, decode-consistent, and takes the band from **launch-unusable (0.104)** to
**measurable (0.0276)** — a 3.76× rate enabler. It does NOT reach the +0.005 target; the measured Shannon floor
proves that is **impossible by coding alone** on the camera-frame per-frame fits. **L1 SE(3) ego-factorization
(a source re-parameterization, scorer-free) is the mathematically-identified lever to ~0.001**, designed per the
unified-ξ constraints, and is the next build. Whether the band NETS lower S is the #205 trained-in d_seg
measurement (a band recovering a meaningful fraction of the ~0.0012-max lane erasure clears 100·Δd_seg > 0.0276).

## Sisters
`analytic_lane_band_primary_authority_decomposition` · `pose-solved-screw-twist-dual-use-film-conditioned-sidecar`
· `project_dashgap_fp_deepdive_range_dependent_ego_phase_fp_as_signal` · `project_contest_is_indirect_rate_distortion_task_space_coding`
· `1b-capstone-build-state-4-components-landed-activation-resolved-wirein-pending` · the Wave-E lane-band closure.
