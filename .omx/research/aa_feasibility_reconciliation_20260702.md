# AA-feasibility RECONCILIATION — coverage-integrated AA-SDF vs brute supersample (train-mem + DECODE budget)

**Advisory. `[macOS-CPU / MLX advisory]` — `score_claim=false`, `promotable=false`. Pointer 0.19110
UNMOVED (moves ONLY via a byte-closed n600 `upstream/evaluate.py` exact row). No n600 launch fired.**

Decisive reconciliation of the supersample-AA disagreement (Wave C "~14GB safe" vs memory-design pass
"~166GB + decode blows the 30-min budget"). Method: source-trace + first-principles byte arithmetic +
a MEASURED decode-cost probe (single-thread faithful forward at the shipped config). Every number below
labeled **GROUNDED** (read/measured this pass) or **EXTRAPOLATION**.

---

## TL;DR VERDICT (decisive)

**The contest-feasible OPTIMAL AA is the ANALYTIC coverage-integrated LANE-BAND raster
(`analytic_lane_render_band.py`, `--lane-render-band`), NOT brute `--render-aa supersample`.**

Brute supersample is DISQUALIFIED on **two independent grounds**, either sufficient:

1. **It HURTS the witness −49%** (MEASURED n600, cited in `analytic_lane_render_band.py` docstring
   L26-27). The 0.00086 floor is a REAL-FRAME *ceiling* (SIGNAL A), NOT the witness's realized
   supersample d_seg (SIGNAL B). Supersampling an already-smooth softmax-of-SDF partition recovers
   nothing (no sub-pixel real texture to integrate) and forcing the INR to fit finer coords hurts.
2. **Decode budget:** ss²=4.03× the forward. On the 4-core CPU contest target the fp64 numpy inflate
   is **41.3 min > 30 min (BLOWS budget)**; fp32 torch = 17.7 min (marginal); and NEITHER shipped
   inflate even *applies* supersample today (both render base-grid point-sample) → a train/decode
   observation MISMATCH.

The analytic lane-band coverage raster is O(1)/pixel, renders at BASE grid (no ss fine grid),
`mx.compile`-friendly, decodes IN budget (≈ g384 point + a cheap composite), and is MEASURED to HELP
(witness-uncertainty-gated form kills the dash-gap FP). **Wave D wiring: set `--render-aa none` (drop
supersample), KEEP `--lane-render-band`.** This removes the 14 GB fine-dir-feat cache, the ss² decode
risk, the −49% witness harm, AND the train/decode mismatch simultaneously. If a full-partition AA is
still wanted, use `--render-aa ipe` (O(1), decode-safe) — never supersample.

**This CONTRADICTS + CORRECTS the current launch config**
(`capstone_witness_launch_config_deepmath_optimal_20260702.md` L42 ships `--render-aa supersample
--aa-supersample 2`). That config's supersample decision rests on the SIGNAL-A ceiling, not the
witness-realized (SIGNAL-B) number, which HURT.

---

## Q1 — What render produced the measured 0.00086 floor (#220)? **GROUNDED**

**BRUTE supersample→box on the REAL FRAME (SIGNAL A) — NOT a coverage-integrated analytic render.**

Traced in `tools/aa_sdf_observation_render_verify_n600.py`:
- imports `from tac.boundary_math.aa_sdf_observation_render import box_downsample_np` (L63).
- **SIGNAL A = REAL FRAME** = "confound-free ACHIEVABLE-THROUGH-R **upper bound**; a witness render is a
  strict subset of any RGB at the render grid, so this bounds what the witness can reach" (L14-15).
- **AA path** (L22, L97-99): "supersample to (ss·G) then `box_downsample_np`(ss) → G". `rg =
  box_downsample_np(fine[None].astype(np.float64), SS)[0]`. Nearest/point vs box-down comparison.
- Docstring: "oracle-R floor toward **0.00091**"; the launch memo/eq cites **0.00086 @ g384**.

**So 0.00086 is a REAL-FRAME CEILING measured with the BRUTE box integrator.** It is what a *perfect*
g384 render could achieve through R — an upper bound on the witness, not the witness's realized d_seg,
and not produced by an analytic-coverage render.

## Q2 — Is the AA-SDF observation render analytic-coverage (O(1)) or does it supersample? **GROUNDED**

`tac.boundary_math.aa_sdf_observation_render.py` has **TWO** modes; NEITHER is a full-partition
analytic-coverage render:

- **(1) supersample→box** (`render_aa_batch_through_R_mlx`, the trainer's `--render-aa supersample`):
  **BRUTE. Internally supersamples.** `witness.call_batch(coord_feats_fine, ...)` evaluates the witness
  at the fine grid `(ss·H, ss·W)` then `box_downsample_mlx`. Cost = **ss² × witness forward**
  (docstring L24: "Cost: ss² × the witness forward"). The docstring calls this "(1) the authority".
- **(2) IPE cone** (`ipe_curvelet_attenuation` / `apply_ipe_attenuation`, `--render-aa ipe`):
  **ANALYTIC, O(1)/pixel, base grid.** Multiplies each curvelet Fourier column by
  `exp(-2π²(Bx²σx²+By²σy²))`. Docstring L33-35: "integrates the INPUT features, not the (nonlinear) RGB
  output, so (1) is the authority and (2) is the **cheap decode-time proxy**." Launch memo: "ipe only
  SMOOTHS the basis" (weak benefit).

**The TRUE analytic O(1)/pixel coverage-integrated raster is a SEPARATE, LANE-ONLY module:**
`tac.boundary_math.analytic_lane_render_band.py` (`--lane-render-band`), `coverage_alpha_from_signed`
= `clip(s/soft+0.5, 0, 1)` (a 1-Lipschitz horizontal signed distance, analytic sub-pixel coverage of
the REAL lane geometry). It is `mx.compile`-friendly (no python pixel loops, L78) and the #212 Metal
candidate (L114-127). **The prompt's phrase "coverage-integrated AA-SDF (analytic O(1)) — the 0.00086
lever" CONFLATES this lane-band raster with the full-partition supersample.** They are different levers:
the 0.00086 came from brute supersample on the real frame (Q1); the analytic O(1) coverage raster is
the lane-band, and its own docstring (L26-27) says **supersample-AA HURTS the witness −49%** while
coverage-integration on real geometry is the correct AA.

## Q3 — Train memory: 14 GB or 166 GB? **RESOLVED (first-principles arithmetic, GROUNDED)**

Both are arithmetically correct — of DIFFERENT caching strategies. Reconciliation
(`ss=2`, 384×512 → fine grid 786,432 px, fp32, P=600; curvelet=80 cols, dir@ndf2=8 cols):

| what is cached | cols | bytes/pair | × 600 | = which claim |
|---|---|---|---|---|
| fine CURVELET feats — **ONE SHARED tensor** (pair-independent, trainer L1178-1179) | 80 | — | **0.23 GB total** | (shared) |
| fine DIR-feats per-pair, full mode @ **ndf2** | 8 | 25.2 MB | **14.06 GB** | **← Wave C "14GB"** |
| fine DIR-feats per-pair, full mode @ ndf6 | 24 | 75.5 MB | 42.2 GB | ← trainer's stale "~45GB" inline |
| NAIVE: full fine feats (curv80+dir8) per-pair @ ndf2 | 88 | 258 MB | **154.7 GB** | **← design-pass "166GB"** |
| NAIVE full fine feats per-pair @ ndf6 (104 cols) | 104 | 305 MB | 182.8 GB | ← trainer's stale "~164GB" inline |
| base `cf_mx_cache` (BASE grid 384×512, 88 cols, ×600) | 88 | 65 MB | **39 GB** | ← the "41GB anchor" |

**RESOLVED:**
- **166 GB** = NAIVELY caching the FULL fine feats (curvelet+dir) per-pair at ss² (option (a) the
  trainer comment L1117-1120 flags as OOM). Over-estimate — you do NOT need per-pair fine curvelet feats.
- **14 GB** = the SHIPPED design: fine curvelet feats are ONE SHARED tensor (0.23 GB); ONLY the fine
  DIR-feats are per-pair (25.2 MB/pair @ ndf2 × 600 = 14.06 GB, full mode). Peak ≈ **63 GB** (14 fine +
  39 base cf_mx_cache + ~8 fwd) on the 128 GB M5 Max → memory-SAFE (≤115 GB).
- The trainer's INLINE comments (`~164GB`, `~45GB` @ L1118/L1146) are **STALE**: ~45 GB = ndf6
  dir-feats (42.2 GB); ~164 GB = naive full-feats-per-pair. They predate the ndf2 + shared-curvelet
  design. **Wave D should update the trainer comment to the reconciled table above.**

**CAVEAT (honesty):** the 14 GB / 63 GB is a **SCALED EXTRAPOLATION** (24 MB/pair measured × 600),
NOT a real n600 allocation. Arithmetic is sound (my probe = 25.2 MB/pair matches the 24 MB claim), but
the launch memo's "probe-confirmed memory-safe" **overstates** — it is *extrapolation*-confirmed.
NOTE: this whole train-memory question is **MOOT** given the decode + witness-harm verdict — you would
not ship supersample regardless of whether it fits in memory.

## Q4 — DECODE cost per AA on the 4-core CPU contest target. **MEASURED (this pass, GROUNDED)**

Probe: faithful witness forward at the shipped config (hidden 96, n_hidden 4, mod-dim 19, in_feat 88,
chroma, hosc, so_iters=4 worst-case, no early-stop), single-thread (`OMP=1`, `torch.set_num_threads(1)`),
via the ACTUAL inflate helpers `torch_{in_proj_h0,outputs_from_h0,R}`. ss²/g384 forward ratio = **4.03×**
(exactly linear in pixels, as expected). Box-down is negligible. R (bicubic→camera) is CONSTANT
(box-down happens BEFORE R, so R always upsamples 384×512→874×1164).

| path | per-pair | n600 SERIAL | **n600 4-core CPU (÷4)** | 15-worker (÷~11) |
|---|---|---|---|---|
| **fp64 numpy** g384 point (current shipped) | 4.13 s | 41.3 min | **10.3 min ✓** | 3.8 min |
| **fp64 numpy** ss2 (if wired) | 16.5 s | 165 min | **41.3 min ✗ OVER** | 15.0 min |
| **fp32 torch** g384 point | 1.78 s | 17.8 min | **4.4 min ✓** | 1.6 min |
| **fp32 torch** ss2 (if wired) | 7.09 s | 70.9 min | **17.7 min ✓** | 6.4 min |

(fp64 serial 41 min ≈ the in-code FEED-eg estimate "~50-60 min g384 point n600 4-core" —
`levelset_byte_close_and_eval.py` L356; GROUNDED cross-check.)

**Findings:**
- **g384 point-sample (NO AA) is IN BUDGET both paths** (fp64 10.3 / fp32 4.4 min on 4-core).
- **Brute supersample ss² BLOWS the fp64 budget (41 min > 30)**, marginal on fp32 (17.7 min), fine on
  T4-GPU. The design-pass "~52-60 min @ ss2" concern is CORRECT for the fp64 reference path.
- **CRITICAL: neither shipped inflate applies supersample.** Both the numpy inflate (`_INFLATE_PY` in
  `levelset_byte_close_and_eval.py`, renders `_coords(render_h, render_w)`) AND the torch inflate
  (`decode_levelset_torch`, `coords_grid(rh, rw)`, rh=render_h=384) render BASE grid point-sample; the
  archive stores render_h/render_w = 384/512 (aa_ss multiplies only the *internal fine* grid, trainer
  L1166). So training with supersample AA but decoding point-sample = an **observation-model MISMATCH**:
  the witness is optimized for `box_down(ss render)` but decode emits the point-sample g384 render →
  the 0.00086/AA benefit is NOT realized at decode as currently wired. To realize it you must WIRE ss
  into inflate.py (currently absent) and pay the ss² decode cost above.
- **Analytic lane-band coverage raster decodes O(1) at g384** (base grid, per-pixel `clip(s/soft+0.5)`
  + alpha composite; `mx.compile`/Metal-friendly). Decode ≈ g384 point + a cheap elementwise raster →
  **IN budget both paths**, no ss.
- **IPE decodes O(1) at g384** (basis regenerated free in inflate; per-column multiplier) → in budget.

## Q5 — VERDICT + ranking (d_seg benefit s.t. train-mem ≤115 GB AND decode ≤30 min/4-core CPU)

| AA | witness d_seg benefit | train mem | decode (4-core CPU) | contest-feasible? |
|---|---|---|---|---|
| **coverage-AA lane-band** (`--lane-render-band`) | **HELPS** (FP-killer, uncertainty+range gated; naive band +25% but gated form is the primary lane lever) | base only (no fine cache) | **O(1) g384, IN budget** | **✓ OPTIMAL** |
| **ipe** (`--render-aa ipe`) | weak ("only SMOOTHS the basis") | base only | O(1) g384, IN budget | ✓ (weak secondary) |
| **brute supersample** (`--render-aa supersample 2`) | **HURTS −49%** on the witness (0.00086 is a real-frame CEILING, not witness-realized) | 63 GB (extrapolated) — fits but moot | fp64 41 min ✗ / fp32 17.7 min; NOT wired in inflate | **✗ DISQUALIFIED** |
| **none** | baseline (point-sample aliases thin lanes) | base only | in budget | ✓ fallback |

**Wave D wiring spec (the correction):**
1. In `witness_autoconfig._all_levers_base` + the launch memo argv: **`--render-aa none`** (remove
   `supersample`/`aa_supersample`/`aa_self_orient_fine_mode`). KEEP the already-present
   `--lane-render-band --lane-band-uncertainty-source witness --lane-band-tau 0.85 --lane-band-eps 0.35
   --lane-band-dash-forward-max-m 55.0 --lane-band-weight 1.0` — that IS the contest-feasible analytic
   coverage AA, and it composes with `--self-orient` (no fine dir-feat cache needed).
2. This eliminates the fine dir-feat cache (14 GB), the ss² decode (41 min fp64), the −49% witness harm,
   and the train/decode observation mismatch — one edit, four problems gone.
3. If a full-partition AA is still desired later: `--render-aa ipe` (O(1), decode-safe) — NEVER
   supersample. Re-open supersample ONLY if a witness-realized (SIGNAL-B, byte-closed through-R) n600
   measurement shows it HELPS AND the decode is moved to fp32-torch or T4-GPU inflate (< 18 min).
4. Update the trainer inline comments (L1117-1148: `~164GB`/`~45GB`) to the reconciled table in Q3.

**Provenance grades:** Q1/Q2 GROUNDED (source-traced). Q3 GROUNDED arithmetic (my probe 25.2 MB/pair =
the 24 MB claim); the 14 GB/63 GB total is a SCALED EXTRAPOLATION (labeled). Q4 MEASURED this pass
(faithful single-thread forward). The −49%-witness-harm + c1/c3 (0.00333/0.00415) numbers are cited in
`analytic_lane_render_band.py` as MEASURED n600 `[macOS-CPU advisory]` (the standalone measurement tool
was not located by name in this pass — flagged for the reviewer to re-confirm the −49% artifact before
the Wave D edit lands). Nothing here is an exact-eval score; pointer 0.19110 unmoved.
