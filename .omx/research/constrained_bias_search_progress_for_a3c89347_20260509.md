# Constrained bias coord search — progress note for sibling a3c89347

<!-- generated_at: 2026-05-09T11:14:00Z, from_state_hash: constrained_coord_search_pr101_bias_landed -->
<!-- HISTORICAL_PROVENANCE — append-only forensic record -->

## TL;DR for sibling a3c89347 (A1 bias correction sweep V0..V10)

Your sweep covered the **gross-magnitude axis**: 11 single-coordinate variants
(no_bias, full PR101, half, 1.5×, 2×, sign-flip, PR102 pattern, PR101+PR102
stack, frame-isolated). Result so far: V1 baseline = 0.19284 [contest-CPU GHA],
V2 half-magnitude = 0.19430 [contest-CPU GHA] (regression of +0.00145).

This subagent now lands a **constrained 3D/4D coordinate search** in the
neighborhood of the verified PR101 anchor (-1, -1, -1) per the convergent
recommendation in
`.omx/research/hnerv_lessons_docs_adversarial_review_20260509_codex.md` §4
("the correct next action is not 'try arbitrary channel constants' — it is a
constrained same-archive coordinate search around the verified PR101 bias and
sidecar scale, with exact CPU/CUDA custody and runtime-smoke closure").

## What landed (this subagent)

1. **`tools/constrained_coord_search_pr101_bias_sidecar.py`** (~470 LOC)
   - Coarse grid: 7³ = 343 candidates centered at -1.0
     `{-1.5, -1.25, -1.1, -1.0, -0.9, -0.75, -0.5}`
   - Coarse-coarse grid: 4³ = 64 candidates `{-1.0, -0.5, 0.0, +0.5}`
     (faster dev-loop)
   - Refined grid: 7-point centered offsets around `--center-coord`
     (default ±0.15 in 0.05 steps)
   - Optional 4D sidecar: 5-point ``{-0.5, -0.25, 0, +0.25, +0.5}`` on
     `up[:, 1, 0]` (frame 1 red — the first unperturbed cell after PR101's
     three perturbed cells; frame 2 in the operator brief is OUT OF RANGE
     since `up` has shape `(batch, 2, 3, H, W)`).

2. **Coordinate convention** that matches your V0..V10 sweep:
   - `coord = -1.0` → emits `sub_(1.000000)` (= PR101 baseline)
   - `coord = -0.5` → emits `sub_(0.500000)` (= your V2 regression anchor)
   - `coord = +1.0` → emits `sub_(-1.000000)` (= effective `add_(1.0)`)
   - `coord =  0.0` → line skipped
   The `v_n1_00_n1_00_n1_00` variant in this tool is bit-identical (modulo
   trailing zeros in float formatting) to your V1 baseline.

3. **`tests/test_constrained_coord_search_pr101_bias_sidecar.py`** (19 tests)
   - V1 baseline reproduction asserted
   - V2 half-magnitude regression anchor reproduction asserted
   - Anchor constants verified to match `tools/build_a1_inflate_time_bias_correction_sweep.py`

4. **Lane registry**: `lane_pr101_bias_constrained_coord_search` pre-registered
   at L0 via `tools/lane_maturity.py add-lane`.

## Coordination

- This tool's grid **includes** your `v_baseline` and `v2_half_magnitude`
  cases (as `v_n1_00_n1_00_n1_00` and `v_n0_50_n0_50_n0_50`). When you read
  back results, those two variant_ids are the regression-anchor sanity checks.
- This tool's grid **does NOT replicate** your V5-V10 (sign-flip /
  PR102-pattern / partial-stack / frame-isolated) cases — those are gross-
  magnitude one-off ablations and are properly your scope.
- This tool's grid **DOES extend** in the directions your sweep didn't cover:
  - Off-anchor neighbors (e.g., `(-1.1, -0.9, -1.0)`) which can isolate which
    coordinate has the most score-curvature.
  - Asymmetric inter-coordinate combinations (e.g., your V8 was symmetric
    "frame 0 only"; this grid covers `c0_0=-1, c0_2=-1, c1_1=0` as well as
    `c0_0=-0.9, c0_2=-1.1, c1_1=-1.0` etc).
  - Sidecar 4D: a fourth coordinate `up[:,1,0]` add term.

## What to do with this tool

1. **Generate variants** (cheap; ~1 second per variant):
   ```bash
   .venv/bin/python tools/constrained_coord_search_pr101_bias_sidecar.py \
       --coarse-coarse \
       --max-variants 64
   ```
   This emits 64 submission_dirs under
   `experiments/results/constrained_coord_search_pr101_bias_<utc>/`.

2. **Rank coarsely on M5 Max** (when sibling C's M5 Max sweep tool lands;
   $0 / ~25 min for the 64-grid):
   - Use sibling C's tool to evaluate every variant's submission_dir on M5 Max.
   - Tag results `[macOS-CPU calibrated]` per
     `feedback_macos_x86_64_epsilon_calibrated_tag_20260508.md` (ε ≈ 6×10⁻⁶).
   - Sort by score; identify top-5.

3. **Promote top-5 to GHA dispatch** ($0.40 each = $2 total):
   - Use `tools/dispatch_cpu_eval_via_github_actions.py` per variant.
   - Tag results `[contest-CPU GHA Linux x86_64]` (the leaderboard-ranking axis).

4. **Refine around the winner** (if any candidate beats A1's 0.19284):
   ```bash
   .venv/bin/python tools/constrained_coord_search_pr101_bias_sidecar.py \
       --refined --center-coord <winning_value> \
       --max-variants 343
   ```
   Then repeat M5 Max → GHA promotion.

## How this complements your work

| Your sweep covers | This sweep covers |
|---|---|
| Gross magnitude (× 0.5, 1, 1.5, 2, ±) | 7³ neighborhood at fine 0.1-0.25 step |
| Sign-flip (V5) | Asymmetric inter-coord combinations |
| PR102 pattern (V6) | Pure-bias 4D extension via sidecar |
| Frame-component isolation (V8/V9/V10) | Joint asymmetric perturbations |

The two sweeps are complementary: yours establishes the gross magnitude
landscape; this one drills into the local neighborhood where (per the V2
regression evidence) the optimum is highly localized.

## DEFERRED-pending-research, NOT killed

Per CLAUDE.md `forbidden_premature_kill_without_research_exhaustion`:

- Even after a 1715-candidate sweep, any "the bias correction is exhausted"
  verdict is **DEFERRED-pending-finer-grid** — the grid here can be refined
  to 0.025 step or below.
- A 4D-sidecar negative is a **sidecar measured-config retired**, not a
  KILL on the underlying bias-correction lane.

## Next-action ownership

- **Subagent THIS one**: tool + tests + this coordination memo (DONE).
- **Sibling C (M5 Max sweep)**: build the M5 Max parallel-sweep CLI; this
  tool's `submission_dir` outputs are ready for it.
- **Sibling a3c89347 (you)**: when V3..V10 GHA results return, please file
  them under `experiments/results/a1_bias_correction_sweep_v*_*/gha_dispatch/`
  in the same schema as V1/V2 so the M5 Max harness can ingest them as
  ground-truth pinning anchors.

## Cross-references

- Domain catalog atom #1 EIG/$: `feedback_domain_exploitation_catalog_landed_20260509.md`
- HNeRV lessons codex review: `.omx/research/hnerv_lessons_docs_adversarial_review_20260509_codex.md`
- Forensics dossier: `.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md`
- A1 sweep V1 baseline: `experiments/results/a1_bias_correction_sweep_v1_pr101_baseline_20260509T053000Z/`
- A1 sweep V2 regression: `experiments/results/a1_bias_correction_sweep_v2_half_magnitude_20260509T103000Z/`
- A1 sweep tool (read-only oracle): `tools/build_a1_inflate_time_bias_correction_sweep.py`
- Constrained coord search tool: `tools/constrained_coord_search_pr101_bias_sidecar.py`
- Constrained coord search tests: `tests/test_constrained_coord_search_pr101_bias_sidecar.py`
- Polymorphic codec port: `src/tac/codec/pr101_polymorphic.py`
- Polymorphic codec tests: `src/tac/tests/test_pr101_polymorphic_codec.py`
