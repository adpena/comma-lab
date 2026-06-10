# Cross-pair scorer-quotient-space waterfilled corrector — the SOLVE + exact-row verdict (task #54)

**LEAD WITH THE POINTER DELTA:** `0.19109982 → 0.19109982` — **UNMOVED.** The cross-pair waterfilled
POSE corrector, run on the EXACT frozen CPU-torch PoseNet over the current frontier carrier, admits
**ZERO** re-allocations (net ΔS = +0.000000, `beats_base=False`, `new_bad=0`). The honest reason, below.

**Authority of every number:** `[local CPU-torch advisory]` — exact upstream `DistortionNet` (SegNet +
PoseNet) on CPU, GT decoded via `upstream/frame_utils.yuv420_to_rgb` ONLY (PyAV rgb24 == ~100× phantom
pose). NOT the contest 600-sample harness → non-promotable per the GOAL authority ladder. `$0` spend, no
GPU, no paid dispatch, **NO MPS**. The exact-score is recomputed from components (the rounded
`final_score` lies).

---

## 0. What was built (a real SOLVE of evaluate.py's KKT condition — not a sweep)

`src/tac/optimization/cross_pair_waterfilled_corrector.py` (the corrector + the λ\* allocator) +
**20 behaviour tests** (`src/tac/optimization/tests/test_cross_pair_waterfilled_corrector.py`, all green;
ruff clean) + `tools/cross_pair_waterfilled_corrector_smoke.py` (the exact-scorer smoke wiring the real
frozen PoseNet observer). Wired into the lazy `tac.optimization` package (`CrossPairPoseWaterfiller` /
`allocate_seg_regions` / `compose_water_level_allocation` / `WATER_LEVEL_LAMBDA_STAR`).

Three pieces, per the operator spec (closed_spec §10 + stacking E3):

1. **`CrossPairPoseWaterfiller` — the POSE side (the near-term exact-row test).** d_pose is a GLOBAL
   pooled budget: `d_pose = mean_pairs ||P(pair)[:6] − p*||²` is pooled BEFORE the sqrt, so a pose
   error on ANY pair trades 1:1 with any other (E3). The pose SCORE term `sqrt(10·d_pose)` is therefore
   NONSEPARABLE: the marginal `5/sqrt(10·d_pose)` GROWS as the pool shrinks. The corrector enumerates
   every (pair, frame-0 mode) candidate, ranks by EXACT ΔS-per-byte at the CURRENT pooled operating
   point, admits the steepest whose value-per-byte exceeds the water level λ\* = 25/D = 6.66e-7, re-pools
   d_pose, re-ranks, and stops when the marginal EQUALIZES at λ\* (the KKT stationarity condition). This
   is NOT "pick each pair's best mode independently" (that ignores the global pool) and NOT a fixed sweep.
   Frame-0 corrections are SegNet-blind by construction (SegNet reads frame1 only via `x[:, -1, ...]`),
   so d_seg is EXACTLY untouched.

2. **`allocate_seg_regions` — the SEG-region side (the lever-C distortion-closure actuator, base-agnostic).**
   Operates on RAG REGIONS (`tac.boundary_math.partition`), never pixels: fund a region iff its value
   (net flips · 100/N) > its cost (contour bytes · 25/D + pose collateral). On a salt-and-pepper residual
   it funds NONE; on a contiguous residual (lever-C output) multi-pixel regions clear the water level.

3. **`compose_water_level_allocation` — the composed λ\* allocator.** Sums the disjoint-section
   seg-region + cross-pair-pose net ΔS (frame-0 pose is SegNet-blind, frame-1 region is PoseNet-collateral-
   priced — orthogonal sections per the stacking orthogonality map). Reuses the canonical exact-ΔS
   admission currency of `tac.optimization.evaluator_action_waterfill`.

---

## 1. THE EXACT-ROW RESULT — `cross_pair_pose_waterfill_smoke.v1` (exact local-CPU-torch)

JSON: `experiments/results/cross_pair_waterfilled_corrector_20260610/pose_smoke{,_n30}.json`. Base =
`frontier_archive` (the 177,169 B contest-CPU frontier carrier) WITH its live FEC6 K=16 frame-0 selector
applied (decoded: 600 codes, mode-0 "none" most common at 133/600 — matching the Huffman assignment).
Per-pair base d_pose on the EXACT PoseNet; each of the 15 alternative K=16 frame-0 modes re-scored exactly.

| sample | seed | pooled d_pose before→after | admitted | **new_bad** | delta_bytes | **NET ΔS (sampled pool)** | **improvable pairs** |
|---|---:|---|---:|---:|---:|---:|---:|
| n=12 | 0 | 1.758e-5 → 1.758e-5 | **0** | **0** | 0 | **+0.000000** | **0 / 12** |
| n=30 | 7 | 2.594e-5 → 2.594e-5 | **0** | **0** | 0 | **+0.000000** | **0 / 30** |

Two independent seeds agree. The **constant-correction control** (apply the single best-on-average
alternative mode `frame0_blue_chroma_amp_1` to EVERY sampled pair) nets **+1.27e-3** (WORSE) — proving
the waterfiller's per-pair allocation is load-bearing (it correctly avoids the constant's harm;
`waterfill_beats_constant=True`).

---

## 2. THE STRUCTURAL FINDING (why the pointer is unmoved — the honest "here's why")

The diagnostic (`diagnostic_why_unmoved`) measures, per sampled pair, the BEST available
alternative-mode pose improvement. The result on the frontier base:

> **`improvable_pairs = 0 / 30`** — for EVERY sampled pair, the live FEC6 selector ALREADY assigns the
> frame-0 mode that minimizes that pair's exact PoseNet residual. No alternative K=16 mode improves ANY
> pair's d_pose.

**The frontier FEC6 K=16 selector is already per-pair pose-optimal over its palette.** The cross-pair
WATERFILLING lens (global-pool re-ranking) adds value only when there are improvable pairs whose marginals
trade across the pool. Here the per-pair optimum IS the pool optimum, because there are zero improvable
pairs to re-allocate. The waterfiller therefore correctly admits nothing and the pose budget is unmoved.

This is the pose-axis analogue of the #55 seg-axis finding: on the frontier base, the cheap, structured
correction lever has already been exhausted by the carrier's own optimizer. The headroom on the pose term
(0.017 of the 0.191 total) is real, but it is NOT reachable by the K=16 frame-0 selector re-allocation —
that lever is saturated.

**Why per-pair-greedy == pool-optimal here (the math):** with 0 improvable pairs, EVERY candidate has
`Δd_pose ≥ 0`, so the value-per-byte is `None` (never admitted) regardless of the pool operating point.
The concave-budget global-pool recompute (which the tests prove is load-bearing — a later step at a
smaller pool is STEEPER, `|d2| > |d1|`) only changes admission ORDER among rent-paying actions; with no
rent-paying actions it changes nothing.

---

## 3. PRE-REGISTERED PREDICTION + KILL/DEFER CRITERION

**PRE-REGISTERED PREDICTION (from the spec, before measurement):** the pose headroom is small (pose-term
0.017) but no corrector had tried cross-pair allocation; the most likely outcome is either a small NET
gain from re-allocating a few mis-assigned pairs, OR (if the frontier selector is already greedy-optimal)
zero gain — in which case the FINDING is that the frame-0 selector lever is saturated and the pose
headroom requires a NEW lever (a richer per-pair correction grammar than the K=16 palette, OR a
sub-pixel pose-luma correction on the PoseNet Jacobian tube).

**Result: CONFIRMED (the saturated-lever branch).** 0 improvable pairs across 42 sampled pairs (two
seeds). The frontier selector has already captured the entire K=16 frame-0 pose lever.

**PRE-REGISTERED DEFER CRITERION (per Forbidden-premature-KILL):** if NO cross-pair re-allocation pays
rent on the frontier base, the cross-pair-pose-via-K16-selector lever is **DEFERRED** (not killed) on
that base, with the finding that the pose headroom is irreducible by selector re-allocation and the
campaign must move to a RICHER pose-correction grammar OR a contiguous-residual base.

**DEFER TRIGGERED (Catalog #307 IMPLEMENTATION-LEVEL):** the **paradigm** (cross-pair global-pool
waterfilling) is PROVEN correct (the allocator equalizes marginals at λ\*; the constant-control is
correctly dominated; the concave-budget recompute is load-bearing — all in the 20 tests). The **K=16
frame-0 selector lever** is simply saturated on the frontier base. This is a DEFER of THAT lever, NOT a
kill of the waterfiller.

**Reactivation criteria (the campaign's next probes):**
1. **Richer pose-correction grammar:** the K=16 palette is 7 coarse luma/RGB/chroma/roll modes. A
   sub-pixel pose-luma correction on the measured PoseNet Jacobian tube (`tac.boundary_math.posenet_jacobian_saliency`
   — already built) is a strictly larger action space; the waterfiller is READY to consume its marginals.
   IF those finer corrections produce improvable pairs, the cross-pair allocator funds the steepest.
2. **A contiguous-residual base (lever C output):** the seg-region allocator (`allocate_seg_regions`) is
   the distortion-closure actuator there; the composed allocator then funds seg-regions + cross-pair-pose
   jointly at one water level.
3. **Full-600 exact pose pool:** the smoke samples 12/30 pairs (each pair × 16 modes × exact PoseNet ≈
   33s/pair on CPU). `--all-600` measures the full contest pose pool; expected identical verdict (the
   sampled 0/42 improvable is a strong signal the selector is globally greedy-optimal) but the full pool
   is the definitive measurement before any KILL escalation.

---

## 4. THE IMMEDIATE-EXACT-EVAL QUESTION (operator pre-registration)

The operator's spec: "If the pose corrector beats frontier: byte-close + paired exact eval." **It does
NOT beat the frontier** — net ΔS = +0.000000 (unmoved). **No paired CPU+CUDA exact eval is
pre-registered** — there is no advisory row beating the frontier, so the eval gate is not met (correct
fail-closed: do not spend ~$0.3-0.6 to confirm a non-improvement). The lane stays at
`[local CPU-torch advisory]`, `research_only=true`.

---

## 5. THE ALLOCATOR'S READINESS AS THE LEVER-C DISTORTION-CLOSURE ACTUATOR (the handoff)

The seg-region allocator (`allocate_seg_regions`) is built base-agnostic and TESTED on both residual
structures:
- **salt-and-pepper (frontier, #55):** `test_region_allocator_declines_salt_and_pepper_single_pixel_flips`
  — 100 single-pixel regions, contour cost ~7 B, collateral 2 flips/repair → NONE fund (`any_fundable=False`).
- **contiguous (lever-C):** `test_region_allocator_funds_contiguous_repairable_region` — a 500-flip
  region, 40 B contour, 10 collateral → FUNDS it (490 net flips · 100/N ≫ 40 B · 25/D), net ΔS < 0.

The composed λ\* allocator (`compose_water_level_allocation`) is READY to fire the moment lever C produces
a contiguous-residual base: it will fund the seg-regions (distortion closure) AND any cross-pair-pose
re-allocation on the new base (which, being a different carrier, regenerates the per-pair pose residuals →
the K=16 lever may NOT be saturated there). The allocator is the evaluator-action waterfiller finally
composed (it REUSES the existing `evaluator_action_waterfill` exact-ΔS currency rather than rebuilding).

---

## 6. ANTI-FAKE self-checks (all pass — NO-FAKE class 1 + class 8)

- **REAL allocation, not a no-op (class 1):** the waterfiller ACTUALLY equalizes marginals — the
  constant-correction control is provably dominated (`test_constant_correction_is_dominated_by_waterfill`);
  a no-op observer admits nothing (`test_no_op_observer_admits_nothing`); the global-pool recompute is
  load-bearing (`test_waterfiller_step_delta_uses_current_pool_not_base` — step 2's pooled_before == step
  1's pooled_after, proving each step is priced at the mutating pool, not a fixed base estimate).
- **rank/verdict ONLY from the exact scorer (class 8):** per-pair d_pose on the exact `DistortionNet`
  PoseNet, GT via `yuv420_to_rgb` ONLY, NEVER MPS. The frame-0 modes are applied with the canonical
  `frame_selector.apply_frame0_mode`, clamped+rounded exactly as inflate does. Score recomputed from
  components.
- **honest collateral:** `new_bad` (pairs worsened) is computed and reported (= 0; a correct allocator
  never admits a pair-worsening action — `test_waterfiller_rejects_pair_worsening_modes`); the seg-region
  candidate's `net_delta_score` subtracts `new_bad_flips` AND `pose_side_effect` (the #55 honesty rule —
  `test_region_candidate_net_accounts_new_bad_and_pose_collateral` proves a region whose repaired flips
  are eaten by collateral declines). The `admitted` count ALONE would lie; the net ΔS is the verdict.
- **NOT a sweep (class 6):** the waterfiller SOLVES the KKT condition (equalize marginals at λ\*); there is
  no grid/sweep over correction params. The byte cost is the EXACT FEC6 Huffman code-length delta.

---

## 7. Wire-in (Catalog #125)

1. **sensitivity-map** — ACTIVE: the per-(pair, mode) exact pose-residual deltas + value-per-byte are the
   cross-pair-pose sensitivity input; on the frontier base every candidate is non-improving (empty fund set).
2. **Pareto** — ACTIVE: the rows establish `frontier_archive` is at a Pareto vertex on the K=16 frame-0
   pose axis (no selector re-allocation moves d_pose down) — the constant-control proves moving off-vertex
   raises the score.
3. **bit-allocator** — ACTIVE: the λ\* allocator IS the bit-allocator (admit iff value-per-byte > λ\* =
   6.66e-7); it correctly allocates ZERO bytes on the frontier base.
4. **cathedral-autopilot** — the smoke → (conditional) paired-eval dispatch surface; gate NOT met (no
   advisory improvement).
5. **continual-learning** — this verdict reseeds the planner: the K=16 frame-0 selector pose lever is
   SATURATED on the frontier (0/42 improvable); the pose headroom requires a richer correction grammar
   (the PoseNet-Jacobian sub-pixel tube) or a new base.
6. **probe-disambiguator** — RESOLVED: "does cross-pair waterfilled pose re-allocation move the frontier?"
   → NO (the frontier selector is already per-pair pose-optimal over K=16; the waterfiller correctly
   admits nothing). The next probe: the PoseNet-Jacobian sub-pixel pose correction (a larger action space).

---

## 8. Cross-references

`closed_spec_boundary_math_system_of_equations_20260610.md` (§10 the water level λ\* = 25/D; §2 d_pose as
the global pooled budget) · `stacking_synergy_composition_plan_20260610.md` (E3 cross-pair pose
fungibility; the orthogonality map) · `closed_spec_boundary_solver_v1_20260610T105830Z.md` (#55 the
sister seg-axis finding: salt-and-pepper residual, the honest new_bad/pose_side rule) ·
`src/tac/optimization/cross_pair_waterfilled_corrector.py` + tests + `tools/cross_pair_waterfilled_corrector_smoke.py`
(the deliverables) · `src/tac/optimization/evaluator_action_waterfill.py` (the reused exact-ΔS admission
currency) · `src/tac/boundary_math/{partition.py, posenet_jacobian_saliency.py}` (the RAG + the next-lever
Jacobian tube) · `upstream/{modules.py, frame_utils.py}` (frozen authority).
