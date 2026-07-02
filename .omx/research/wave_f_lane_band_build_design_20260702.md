# Wave-F — concrete build design + in-repo primitive inventory (fork B: optimal RD lane-band code)

**Status:** BUILD DESIGN (2026-07-02). R2 of Wave-F. Advisory / design-only; NO serializer edits, NO GPU.
Pointer 0.19110 UNMOVED (moves only via a byte-closed n600 `upstream/evaluate.py` exact row). This memo is
the executable build plan a build agent runs directly. Design authority (the 5-lever deep math):
`.omx/research/wave_f_optimal_lane_band_rd_code_design_20260702.md`. Sisters:
`analytic_lane_band_primary_authority_decomposition`, `pose-solved-screw-twist-dual-use-film-conditioned-sidecar`,
`project_dashgap_fp_deepdive_range_dependent_ego_phase_fp_as_signal`, `project_contest_is_indirect_rate_distortion_task_space_coding`.

Composition, not from-scratch: Wave-F is ~90% wiring of VERIFIED in-repo primitives + one genuinely new
lever (L1 ego-factorization) + the byte-close/inflate 5th-block that (finding below) DOES NOT EXIST YET.

---

## 0. TL;DR — the two headline findings (read first)

1. **The Wave-E byte-close 5th-block + inflate `_lane_*` reproduction DO NOT EXIST in the tree** (verified in
   BOTH the worktree AND the main checkout `/Users/adpena/projects/pact`). `serialize_lane_band` /
   `deserialize_lane_band` / `composite_band_on_render` / `LaneBandRenderConfig` / `build_lane_band_pairs_from_lstars`
   EXIST as functions in `src/tac/boundary_math/analytic_lane_render_band.py` (landed commit `bfb71c384`), but:
   - `tools/levelset_byte_close_and_eval.py` `_io_pack(manifest, base, code, pose)` is still **4-block** — NO lane
     band 5th block, NO `LBND`/`LANE_BAND_MAGIC` reference (grep of both checkouts: 0 hits).
   - `src/tac/local_acceleration/torch_levelset_inflate.py` (`TORCH_INFLATE_PY`) and the tool's `_INFLATE_PY` have
     **no `_lane_*` / band functions** — grep 0 hits.
   - **No caller anywhere** invokes `serialize_lane_band` (grep of `src/ tools/ experiments/`: 0 hits).
   - **No test** exercises the lane-band serializer / decode-consistency (grep: 0 hits).
   The design-authority memo's claim *"Wave E CLOSED the lane-band phantom … parent-verified 15/15 decode-consistency
   + 7/7 default-off byte-identical"* is **NOT substantiated by wired code in this repo.** Consequence for Wave-F:
   the task framing "replace the naive serializer" is only half right — the build agent must **BUILD the byte-close
   5th block AND the inflate reproduction from scratch** (there is nothing to "replace" on the byte-close/inflate
   surface), while replacing the naive per-pair serializer in the module with the optimal RD code. Treat the "220 KB
   naive @ n600" as a DERIVED estimate (float64 payload sizing), not a measured byte-closed row.

2. **The L1 "reuse the stored ξ (already counted for d_pose → ~0 marginal)" claim is NOT valid as written.** The
   stored-pose sidecar (#206, `src/tac/scorer_targets.py`) stores the **PoseNet 6-vector OUTPUTS** per pair
   (`targets: (N_pairs, 6)`, the learned pose-head regression the scorer MSE-compares) — NOT a metric planar/se(3)
   ego displacement usable to warp lane geometry cam↔ground. They are different objects. So L1 must EITHER (a)
   DERIVE the per-pair planar ego displacement deterministically for FREE (from the lane observations / a monocular
   estimate — rule-118 clean), OR (b) STORE a small metric ego stream (COUNTED — the "~0 marginal" claim is then
   false and must be measured). This is a rule-118 honesty flag, not a blocker: L1 still works, the accounting just
   has to be honest and MEASURED.

Neither finding kills Wave-F. Both sharpen the build plan and the NO-FAKE accounting below.

---

## 1. The signal + the objective (from the design authority, restated exactly)

Per pair: up to 5 `LaneLine`s, each = `centerline_coeffs` (deg≤3, ≤4 floats) + `halfwidth_coeffs` (deg 1, 2 floats)
+ optional dash `(period, phase, duty)` (3 floats) + `forward_range` (2 floats) ≈ 7–13 floats/line. ~50 scalars/pair
× 600 pairs. Naive = float64 + brotli ≈ derived-220 KB → rate_term +0.147. Objective: minimize
`S(Q) = 100·d_seg(Q) + √(10·d_pose) + 25·bytes(Q)/37_545_489` over the lane-code quantizer/predictor `Q`, band
d_seg-win − band rate-cost > 0, decode-consistency PRESERVED, at the KKT interior operating point (§5).

**Crucial geometry fact (verified, `lane_sdf_component.py:114-133`):** the `LaneLine.centerline_coeffs` are ALREADY
in a **metric ground-relative frame** — `lateral(m) = polyval(centerline_coeffs, forward(m))`, `forward` = ground
forward distance (m), via flat-ground IPM (`_FX=400.3, _FY=399.5, _CAM_H=1.2, _V_HORIZON=174.0`). This is what makes
L1 clean: the per-pair centerline is a metric world-frame road curve, so ego-compensation is a metric shift, not a
pixel hack.

---

## 2. Verified in-repo primitive map (real APIs — grepped, not invented)

| Lever | Primitive (VERIFIED path + signature) | Status | Reuse verdict |
|---|---|---|---|
| **L2/L3 entropy + temporal-AR coder** | `tac.optimization.pose_trajectory_entropy` — `encode_pose_trajectory(trajectory:(n_pairs,n_dims), *, deltas) -> bytes` + `decode_pose_trajectory(payload) -> (n_pairs,n_dims) int64` + `pose_carrier_real_bytes(...) -> (real_bytes, quant)`. REAL reversible: per-dim uniform-quantize → **first-order temporal delta** → `constriction.stream.queue.RangeEncoder`/`RangeDecoder` over a **transmitted per-dim brotli'd uint16 PMF** + seed. `_POSE_MAGIC=b"PTC1"`, self-describing header. Roundtrip-asserted (NO-FAKE). | **EXISTS, exact fit** | **REUSE / thin-fork.** The lane coeffs ARE a `(600, ~50)` smooth trajectory — this codec already does quantize→temporal-delta→range-code→transmitted-PMF, which is L2+L3(quantize)+entropy in ONE verified primitive. |
| **Entropy coder (alt, pure-python, no dep)** | `tac.lossless.range_coder` — `RangeEncoder.encode(*, symbol, cumulative, total)` / `RangeDecoder` + `normalize_probabilities` + `cumulative_frequencies`. 387 LOC, deterministic, no `constriction` dep. | EXISTS | Fallback only. Prefer the constriction path (above) — it's the byte-close-proven one and matches PR103-silver's coder. |
| **L3 KKT reverse-waterfill bit allocation (#157)** | `tac.frontier_exact_bitalloc` — `waterfill_bit_allocation(sens: CombinedTensorSensitivity, lam, *, b_min=2, b_max=8) -> BitAllocation` (KKT `b_t*=log2(λ·ln2·c_t/n_t)`, `c_t=s_t·absmax_t·√n_t`) + `combine_sensitivities(...)` + `lam_for_target_mean_bits(sens, target, ...)` (bisection on λ). Takes **arbitrary per-key sensitivity/absmax/numel dicts** (numel=1 per lane coeff → per-coeff bit-width). | **EXISTS, takes arbitrary vectors** | **REUSE.** Feed a per-coeff `CombinedTensorSensitivity` (§4 L3). λ is the RD operating-point knob → §5 KKT solve. |
| **L3 task-RD sensitivity ∂d_seg/∂coeff (#141)** | `tac.margin_saliency_map` — `compute_margin_saliency_map(frame, segnet, ...) -> (H,W)` = `|∂(Σ margin)/∂input|`; `_topk_margin(logits)=top1−top2`; `MarginSaliency`, `DecoderTensorSaliency`, `compute_decoder_tensor_margin_saliency`. Produces the per-pixel/per-tensor margin gradient (the Fisher surrogate). | EXISTS (per-pixel/per-weight) | **COMPOSE** (needs a thin new wrapper — §4 L3): chain per-pixel margin-saliency → rasterizer → R → per-coeff via autodiff (MLX) OR finite-diff (n600-faithful, simplest). |
| **L1 se(3) engine (`tac.lie` #193)** | `tac.lie` — `exp_se3, log_se3, compose, inverse, adjoint_T, make_T, rotation_of, translation_of, left_jacobian_se3`; `so3.*`; `screw_blend` (DLB/ScLERP); `se3_bspline` (cumulative SE(3) B-spline). MLX + numpy-fp32 oracle, parity-gated. Twist `ξ=(ρ,ω)` translation-first. **Standalone — imports nothing from witness; the wire-in is OURS to build.** | EXISTS (engine only) | **COMPOSE.** For L1 we need only the **planar 3-DOF** sub-case (Δforward, Δlateral, Δyaw); full se(3) is available but overkill. The cam↔ground warp is a thin composition (§4 L1) — NOT a prebuilt function. |
| **L1 IPM ground geometry** | `tac.boundary_math.lane_sdf_component` — `LaneLine{centerline_coeffs, halfwidth_coeffs, dash_period_m, dash_phase_m, dash_duty, forward_range}` + `.lateral_of_forward(fwd)`, `.halfwidth_of_v(v)`, `.n_floats()`; `cluster_lane_lines`, `fit_lane_line`; consts `_FX,_FY,_CAM_H,_V_HORIZON`. | EXISTS | **REUSE.** Centerline already metric-ground-frame → ego-compensation is a metric shift of `centerline_coeffs`. |
| **The naive serializer to replace** | `tac.boundary_math.analytic_lane_render_band` — `serialize_lane_band(pairs_lines, cfg) -> bytes` (MAGIC `LBND1\x00`, u32 header_len, json header, **float64 payload**), `deserialize_lane_band`, `LaneBandRenderConfig`, `build_lane_band_pairs_from_lstars(lstars, cfg) -> (pairs_lines, stats)`, `composite_band_on_render`, `rasterize_lane_coverage_range_dependent`, `coverage_alpha_from_signed`(≡ the AA-SDF clip), `witness_uncertainty_mask`. | EXISTS (naive) | **REPLACE the serialize/deserialize; KEEP the raster + composite + fit** (those are the FREE inflate-time generic algorithm, rule-118). |
| **#206 stored-pose sidecar** | `tac.scorer_targets` — `extract_posenet_targets(...) -> {'targets':(n_pairs,6), ...}`, `save_posenet_targets`, `load_posenet_targets`, `extract_and_save`. Stores **PoseNet 6-vector outputs**, NOT metric ξ. | EXISTS | **DO NOT reuse as ego-warp source** (finding #2). Separate object. |
| **The byte-close 5th block + inflate `_lane_*`** | — | **DOES NOT EXIST** | **BUILD** (finding #1). |

---

## 3. The new serialize/deserialize architecture (encode pipeline)

Replace `serialize_lane_band`'s naive `float64 + brotli` with the RD pipeline. Compress-time (source video fully
available). Emit a NEW versioned block `LBND2` (keep `LBND1` for back-compat / the default-off byte-identical gate).

```
build_lane_band_pairs_from_lstars(lstars, cfg)              # REUSE — fit per-pair LaneLines (unchanged)
  -> pairs_lines : list[list[LaneLine]]  (600 pairs)

# --- L4: inter-line canonical order (ego-lane + lateral offsets) ---
canonicalize_lines(pairs_lines)                             # NEW (thin): sort lines per pair by signed lateral@fwd=near;
  -> per-pair ordered slots [left2,left1,ego,right1,right2] # ego-lane = min |lateral@near|; store lateral OFFSETS

# --- L1: ego-motion factorization (DOMINANT lever, the new logic) ---
estimate_planar_ego(pairs_lines)                            # NEW: per-pair (Δforward, Δlateral, Δyaw) DERIVED
  -> ego_disp : (n_pairs, 3)                                #   deterministically from the lane structure shift
                                                            #   (free monocular estimate) — rule-118 clean IF derived
warp_to_ground_frame(pairs_lines, ego_disp)                # NEW (uses tac.lie planar sub-case + IPM):
  -> world_coeffs : (n_pairs, K) near-STATIC road geometry  #   compensate ego → world-frame ~static

# --- L2: temporal-AR residual of the (near-static) world coeffs ---
# world_coeffs is now a slowly-varying (n_pairs, K) trajectory. Feed DIRECTLY to the verified coder:
# it already does per-dim quantize -> first-order temporal-delta -> range-code(transmitted PMF).

# --- L3: task-RD quantize (per-coeff Δ from KKT waterfill on d_seg sensitivity) ---
sens = per_coeff_dseg_sensitivity(world_coeffs, cfg)       # NEW wrapper (§4 L3): ∂d_seg/∂coeff via raster+R+SegNet
deltas = kkt_deltas_from_sensitivity(sens, lam*)           # frontier_exact_bitalloc.waterfill -> per-coeff Δ_k
                                                            #   Δ_k = quant step; coarse where d_seg-insensitive
payload_world = encode_pose_trajectory(world_coeffs, deltas=deltas)   # REUSE the verified reversible coder

# --- L5: dash phase = ego-forward-distance (3rd dual-use) ---
# dash phase is a deterministic function of cumulative ego-forward (from ego_disp) -> DERIVED, not stored.
# Only per-line dash (period, duty) ship (near-constant -> tiny, code with the coeff stream or a 2-scalar/line side).

# --- assemble LBND2 ---
LBND2 = MAGIC | u32 hdr_len | json_hdr(cfg + geom + n_pairs + per-pair line layout + ego-derivation mode)
        | ego_disp_block (DERIVED=empty, or COUNTED range-coded (n_pairs,3) if stored — MEASURED both ways)
        | payload_world (PTC-style)
        | dash_static_block (per-line period/duty, tiny)
```

Everything except the counted payload is **generic algorithm** regenerated FREE in inflate (rule-118).

## 3b. The decode-side inflate reproduction (must stay bit-exact train==decode — the Wave-E gate)

Build a **new inflate `_lane_*` section** (mirror of `_INFLATE_PY` / `torch_levelset_inflate.py` style, op-for-op):

```
_lane_read_lbnd2(blob) -> (hdr, ego_disp|None, world_quant, dash_static)   # parse LBND2
_lane_decode_world(world_quant, deltas) = decode_pose_trajectory(...)      # bit-exact inverse (verified)
_lane_ego(ego_disp | derive_from(world_quant, hdr))                        # DERIVED path must be byte-identical to compress-side
_lane_unwarp_to_camera(world_coeffs, ego_disp) -> per-pair LaneLines       # inverse of warp_to_ground_frame
_lane_raster = rasterize_lane_coverage_range_dependent(lines, ...)         # REUSE (unchanged, generic)
_lane_composite = composite_band_on_render(rgb, lane_rgb, coverage, u_mask, weight)  # REUSE (unchanged)
```

**Bit-exact gate (the whole point):** the decoded `world_coeffs` (float = `world_quant · deltas`) → `_lane_unwarp`
→ `LaneLine` → `rasterize` MUST reproduce the compress-side coverage BIT-FOR-BIT. Two determinism hazards to pin:
(i) the DERIVED ego path (if used) must be computed from the SAME quantized inputs both sides (no float drift);
(ii) `warp`/`unwarp` must be an exact algebraic inverse pair at the quantized coeff grid (test: `unwarp(warp(x))==x`
bit-exact on the quantized grid). If exact-inverse is fragile, fall back to **storing** `ego_disp` (COUNTED) so
decode never re-derives — trade rate for determinism, MEASURE which wins S.

---

## 4. The three new pieces of logic (build spec)

**L1 `estimate_planar_ego` + `warp_to_ground_frame`/`_unwarp` (the dominant + only genuinely-new lever).**
Planar 3-DOF `(Δs, Δy, Δψ)` per pair. Use `tac.lie` for the SE(2)⊂SE(3) compose/inverse (or a 3×3 homogeneous
planar transform directly — cheaper, still parity-gated vs `tac.lie` as oracle). `warp`: express pair-t centerline
in a fixed world frame anchored at pair-0 by composing cumulative ego. `estimate`: LEAST-SQUARES fit of `(Δs,Δy,Δψ)`
that best aligns pair-t's lane set onto pair-(t−1)'s (the lanes ARE the ego observation). **Rule-118 decision
(MEASURE both):** (a) DERIVED — recompute the same LS fit at decode from the decoded world coeffs (free, but must be
bit-exact); (b) COUNTED — range-code `ego_disp:(n_pairs,3)` via `encode_pose_trajectory` (small, deterministic).

**L3 `per_coeff_dseg_sensitivity` (the task-RD sensitivity wrapper).**
For each world-coeff `k`: perturb by ε, `_unwarp`→`raster`→`composite`→`R`→ frozen CPU-torch SegNet argmax, measure
Δd_seg (FINITE-DIFF, n600 — the allergic-to-toys authority path). Simplest + faithful. (Autodiff via the existing
MLX twins `rasterize_lane_coverage_mlx`/`composite_lane_band_mlx` + a differentiable-R is a later speedup, not v1.)
Feed `{coeff_k: |Δd_seg/Δcoeff|}` + `absmax` + `numel=1` into `combine_sensitivities`→`waterfill_bit_allocation`.

**L4 `canonicalize_lines` (inter-line correlation).** Order the ≤5 lines per pair into fixed slots by signed lateral
at near-range; code the ego-lane centerline absolutely + the other lanes as lateral OFFSETS (near-constant → tiny).
Pure reordering + subtraction; trivially invertible.

---

## 5. The KKT operating-point solve (math-first, no vibes band)

The S-optimal interior point is the KKT stationarity where the d_seg-term marginal balances the rate-term marginal:

    ∂d_seg/∂byte = 25 / (100 · 37_545_489)     (rate-slope condition; DERIVED from S, not a predicted band)

Operationalize on the verified primitives: `waterfill_bit_allocation` gives, for each λ, a per-coeff bit allocation
whose **measured** byte-cost (via `pose_carrier_real_bytes` on the quantized world coeffs) and **measured** d_seg
(finite-diff through R) define one RD point. Bisect λ (reuse `lam_for_target_mean_bits`'s bisection structure) until
the measured local slope Δd_seg/Δbyte crosses the rate-slope constant. This is a real convex-ish RD solve on MEASURED
points — first-principles anchored (the S gradient), so no predicted-ΔS-band assertion is made (the net-S is the
MEASURED byte-closed row, §6 gate 4). Dykstra-feasibility note: the lane block is a single separable-convex RD
allocation (no cross-constraint intersection), so alternating-projections feasibility is trivially satisfied.

---

## 6. Build task list + acceptance gates (step-by-step, executable)

Each task lands via `tools/subagent_commit_serializer.py` with post-edit working-tree sha (per CLAUDE.md).

- **T0 — Byte-close 5th block scaffold (BUILD; finding #1).** Extend `tools/levelset_byte_close_and_eval.py`
  `_io_pack` to a 5-tuple `(manifest, base, code, pose, lane)` (append a 5th length-prefixed chunk; `lane=b""` when
  band-off). Add the `_lane_*` inflate functions to `_INFLATE_PY` (and mirror in `torch_levelset_inflate.TORCH_INFLATE_PY`).
  **Gate T0:** default-off (`lane=b""`) produces a **byte-identical** archive vs the current 4-block path (the "7/7"
  gate — build it, don't cite it as pre-existing) + inflate output bit-identical.
- **T1 — LBND2 optimal serializer** in `analytic_lane_render_band.py` (new `serialize_lane_band_rd`/`deserialize_lane_band_rd`,
  `LBND2` magic; keep `LBND1` intact). Compose L1/L2/L3/L4/L5 per §3. **Gate T1:** `deserialize_rd(serialize_rd(x))`
  reproduces per-pair `LaneLine`s to the quantized grid bit-exact; `pose_carrier_real_bytes` roundtrip asserted.
- **T2 — L1 ego-factorization** (`estimate_planar_ego`, `warp_to_ground_frame`, `_unwarp`) with the DERIVED-vs-COUNTED
  switch. **Gate T2:** `unwarp(warp(x))==x` bit-exact on the quantized grid; DERIVED-ego reproduced bit-identically
  compress-side vs decode-side (or fall back to COUNTED).
- **T3 — L3 sensitivity + KKT λ-solve** (`per_coeff_dseg_sensitivity` finite-diff n600 through R + frozen CPU-torch
  SegNet; `kkt_deltas_from_sensitivity` via `waterfill_bit_allocation` + λ-bisection to the rate-slope). **Gate T3:**
  the λ-solve returns per-coeff Δ; measured Δd_seg/Δbyte brackets the rate-slope constant.
- **T4 — Decode-consistency proof (the Wave-E gate, PRESERVED).** The full LBND2 decode → raster → composite → R →
  SegNet argmax == the training-side composite, bit-exact (numpy-fp32 authority; NEVER MPS). Build the 15/15-style
  test. **Gate T4:** `max_abs_uint8_diff == 0` train-vs-decode over n600 sample.
- **T5 — MEASURED net-S @ n600.** Byte-close the LBND2 lane block INTO the LVLS1 archive on a real converged levelset
  ckpt; run `upstream/evaluate.py` (contest-CPU / contest-CUDA, NEVER MPS) with band-ON vs band-OFF. **Gate T5 (the
  verdict):** `S(band-on) < S(band-off)` — band d_seg-win − band rate-cost > 0 at the KKT point. If net-negative even
  at optimal coding → honest verdict: coding can't rescue the band → operator re-decides (fallback = launch witness-
  alone, band-OFF). Either outcome is a real measured row, not an interpretation.

**rule-118 / NO-FAKE boundary (binding).** COUNTED in archive.zip: the per-pair world-coeff residual stream + (if
COUNTED) the ego stream + per-line dash static. FREE generic algorithm in inflate: the IPM, `tac.lie` planar warp,
`rasterize_lane_coverage_range_dependent`, `composite_band_on_render`, the range-decoder, the DERIVED-ego LS re-fit.
NO GT mask, NO scorer weights, NO per-pixel table ship. The "reuse pose ξ" honesty flag (finding #2) is resolved by
MEASURING the DERIVED-vs-COUNTED ego accounting — never asserted as "free."

---

## What's MISSING / needs building (the deliverable checklist)

1. **Byte-close 5th block** (`_io_pack` 4→5) + **inflate `_lane_*` reproduction** — **DOES NOT EXIST** in either
   checkout. Highest-priority build; nothing to "replace."
2. **The Wave-E decode-consistency proof (15/15) + default-off byte-identical (7/7)** — **not reproducible in-tree;
   no test found.** Build them as gates T0/T4 (do not cite them as pre-existing).
3. **L1 `estimate_planar_ego` + `warp/_unwarp`** — new logic (composition of `tac.lie` planar sub-case + IPM). The
   dominant lever; the exact-inverse determinism is the main build risk.
4. **L3 `per_coeff_dseg_sensitivity`** wrapper (finite-diff through R) + **`kkt_deltas_from_sensitivity`** λ-solve
   wrapper — thin composition over verified `margin_saliency_map` + `frontier_exact_bitalloc`.
5. **L4 `canonicalize_lines`** (inter-line ordering + offsets) — thin new.
6. **Honest ego-accounting decision** (finding #2): DERIVED (free, bit-exact) vs COUNTED (small, stored) — MEASURE
   both; the "reuse stored pose ξ ~0 marginal" claim is INVALID (pose sidecar stores PoseNet 6-vector, not metric ξ).

**Primitives that EXIST and are directly reusable (no build):** `pose_trajectory_entropy.{encode,decode}_pose_trajectory`
(L2+quantize+entropy, verified reversible constriction coder), `frontier_exact_bitalloc.waterfill_bit_allocation`
(L3 KKT, arbitrary sensitivity vectors), `margin_saliency_map.compute_margin_saliency_map` (L3 margin gradient),
`tac.lie` (L1 se(3) engine), `lane_sdf_component` (IPM + `LaneLine` fit), `analytic_lane_render_band.{rasterize_lane_coverage_range_dependent,composite_band_on_render}` (FREE raster+composite).

## Canonical-vs-unique decision per layer
- Entropy coder: **ADOPT** `pose_trajectory_entropy` (verified reversible; matches PR103-silver constriction path).
- Bit allocation: **ADOPT** `frontier_exact_bitalloc` (exact KKT, arbitrary sensitivity).
- Ego-factorization L1: **FORK/NEW** (no canonical exists; planar sub-case of `tac.lie`).
- Raster/composite: **ADOPT** (already the FREE rule-118 generic algorithm).

## Observability surface
Per-lever byte cost (measured via `pose_carrier_real_bytes`), per-coeff KKT bit-width (`BitAllocation.nbits/continuous_bits`),
per-coeff Δd_seg (finite-diff), band-on vs band-off S decomposition (seg/pose/rate), DERIVED-vs-COUNTED ego byte delta,
decode-consistency `max_abs_uint8_diff`. All machine-readable JSON at byte-close time; cite-chain = ckpt sha + upstream sha.

---
*Advisory. Pointer 0.19110 UNMOVED — moves only via gate T5's byte-closed `upstream/evaluate.py` n600 row.*
