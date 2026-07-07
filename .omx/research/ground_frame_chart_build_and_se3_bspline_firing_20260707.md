# GROUND-FRAME CHART (#194 / §17.1) build + se3_bspline FIRST FIRING — landing memo (2026-07-07)

**Outcome first:** the §17.1 ground-frame-chart machinery is BUILT, $0-VERIFIED (reproduces the
FEED-ll reach numbers EXACTLY through R), wired into the trainer behind a default-off flag, held by
the DSL as a `GroundFrameChart` Lever (never-fired, in the duty-to-measure queue), and the training
A/B is PREPARED as a validated loadable config — NOT launched (operator-GO-gated). Independently,
`tac.lie.se3_bspline` FIRED for the first time on the real n600 ξ table: direct spline replacement
of the ξ payload is MEASURED DEAD; spline-as-predictor residual coding has a MEASURED ~1 KB floor
headroom. **Pointer contest-CPU 0.19110 UNMOVED** — everything here is means; every number below is
`[macOS-CPU advisory]`.

## 1. What was built (BUILD wave C files)

- `src/tac/boundary_math/ground_frame_chart.py` — per-pair witness INPUT-coordinate pre-composition
  with the cumulative ξ-homography. **v0 = GROUND-plane single chart** (module implements all three
  FEED-ll regimes ground/rotonly/identity; the trainer wire-in uses ground only). Rationale: the
  chart acts on INPUT coords — per-class routing needs the class the field itself predicts;
  K-chart per-class blending is `screw_blend`'s designed future consumer. Ground covers
  Road/Lane/Movable (~82 % of measured flip mass, MEASURED FEED-dv). Math is REUSED from the
  measured reach tool (`tools/measure_screw_reach_through_R.py`) and **bit-parity-PINNED by test**
  (exact `np.array_equal`, not allclose). numpy fp32 reference + MLX twin (op-for-op; CPU-stream
  bit-exact per the MLX-GPU-not-bit-identical discipline). Intrinsics/camera height via
  `tac.clip_profile` (no clip hardcodes). `chart[ref] == identity` EXACTLY.
- `src/tac/boundary_math/tests/test_ground_frame_chart.py` — 25 tests, all green (parity pins,
  inverse consistency, normalized↔pixel round-trip, MLX CPU bit-parity, identity fast-path
  object-identity, z-guard, guards).
- Trainer wire-in (`experiments/train_levelset_witness_realized_through_R_mlx.py`):
  `--ground-frame-chart` (default OFF = byte-identical; the off-path is structurally unchanged) +
  `--gfc-ref-pair/--gfc-s-t/--gfc-s-r/--gfc-pitch` (defaults = the MEASURED FEED-ll reach
  calibration s_t=-0.003224707899359239, s_r=0, pitch=-0.01). Routed through the TWO canonical
  accessors (`_feats_np_for_pair` numpy verdict/deploy side; `_cf_mx` MLX training side — cache
  built ONCE, the chart is static). FAIL-CLOSED v0 combinations: `--self-orient` (frame-coords dir
  feats + ground-coords curvelet feats = two coordinate systems in one feature vector),
  `--render-aa != none` (ipe attenuates the shared feats; supersample uses an un-charted fine
  grid), `--structured-init` with ref≠0. **NOTE (absorption):** these trainer edits landed inside
  sibling commit `1d6704e5b` (telemetry) via the concurrent-staging absorption pattern (Catalog
  #340 class; failure-ledger `serializer_whole_file_staging_absorbs_sibling_hunks`); content
  verified intact in HEAD.
- DSL leg: `GroundFrameChart` Lever factory in `tac.witness_dsl.curriculum_dsl` (commit
  `049aa0d9f`, `[consumers-generic]`). `validate()` OK; compiles into real argv;
  `lever_registry.completeness()` shows all 5 flags mapped (unmapped=0).
- Equations leg: 3rd `EmpiricalAnchor` on the existing
  `store_nothing_pose_carrier_rate_collapse_vs_dpose_v1` (refine-don't-duplicate; commit
  `c6b83b1b7`) recording the se3_bspline firing curve. **The chart LAW itself is owed on the
  GO-gated A/B measurement, not before.**

## 2. $0 verification (correctness, NOT a training verdict)

- **Unit parity (MEASURED, exact):** module cumulative homography == reach-tool
  `cumulative_homography` bit-for-bit for ground/rotonly/identity, k∈{0,1,5,13}; incremental
  build == per-k from-scratch product bit-for-bit.
- **GT-side reach reproduction (MEASURED, through R + frozen CPU SegNet, n96, 16 forwards, 9.7 s):**
  warping GT argmax a→a+k with the CHART MODULE's homographies reproduces FEED-ll EXACTLY —
  k=0 strat_bulk **0.006036** (ref 0.006036, |Δ|=0.00e+00), k=47 strat_bulk **0.021857**
  (ref 0.021857, |Δ|=0.00e+00). NO-FAKE self-check SegNet(gt_f1)==lstars PASS. Artifact:
  `experiments/results/ground_frame_chart_20260707/chart_reach_n96_verification.json`
  (gitignored-durable; script `verify_chart_reach_n96.py` beside it, chunked-resumable foreground).
- **Trainer smokes (MEASURED, n1+n6 CPU, 2 epochs, ~min):** OFF path runs end-to-end (byte-identical
  path exercised); ON path runs end-to-end with NON-identity charts at n6 (chart build → per-pair
  cache → train step → verdict → deploy blob). These are wire-in EXECUTION checks only — n1/n6 are
  NOT evidence of any d_seg effect (allergic-to-toys; the A/B is the measurement).

## 3. se3_bspline FIRST FIRING (activation-ledger closure; independent rate item)

ξ source (MEASURED provenance, live #205 run.log `stage=pose_carrier`):
`ξ[p] = xi_from_pose_calibration(gt_poses[p], s_t=0.044, s_r=0, pitch=0)`, n600, q_levels=4096.
Machine-readable curve: `experiments/results/ground_frame_chart_20260707/se3_bspline_rate_error_curve.json`
(rows: knots, bytes_coded, rate_term, max/mean/p95 probe-point error px, max|Δξ|).

- Baseline (MEASURED): full-ξ payload raw 7232 B / delta_ar **3200 B** (rate 0.00213); the
  baseline's own quantization geometry error is **max 0.31 px / mean 0.13 px** (derive-H probe
  points, native grid).
- **Direct spline replacement: DEAD (MEASURED).** At EVERY knot count M∈{4..601} the
  sampled-control cumulative B-spline's decoded warp error is 53–703 px MEAN (300–1000× the
  quantization floor), non-monotone in M (sampling aliasing). The ξ sequence carries essential
  per-pair high-frequency content (ego jitter) that derive-H is very sensitive to.
  Classification (Catalog #307): falsifies the SAMPLED-CONTROL FIT implementation; the spline
  paradigm is intact (LSQ fit never fired) — but see the ceiling argument below.
- **Spline-as-predictor + lossless residual: VIABLE headroom (MEASURED floor).** M=16 knots
  (224 B) + order-0 entropy of the exact quantized residual (1958 B) = **2182 B floor vs 3200 B**
  shipped table — up to ~1 KB (~0.0007 rate) headroom, EXACT reconstruction on the quantized grid.
  DERIVED caveat: the floor excludes real-coder model overhead; the delta_ar baseline already
  includes its own. A real residual coder lands between 2182 B and 3200 B.
- Interpretation labels: probe-point H-displacement is the derive-H DECODE-PATH geometry error,
  NOT d_pose (INFERRED sensitivity; no PoseNet forwards spent). The twist-sequence→SE(3)-path
  lifting (prepended zero twist so ξ[0] is carried) is a compression transform; smoothness of the
  sequence is the ASSUMED structure the spline exploits (measured: only partially present).

## 4. Non-orphan enforcement evidence (coordinator directive)

- **Duty-to-measure queue (VERIFIED, not assumed):** after the DSL commit,
  `tac.witness_dsl.activation_ledger` reports —
  `ActivationStatus(lever='GroundFrameChart', ever_fired=False, ever_measured=False, retired=False,
  state='never-fired', last_event=None, last_ts=None, n_fired=0, n_measured=0)` — one of 35 rows in
  `duty_to_measure()`; `tools/costate_digest.py` surfaces the queue ("35 owed; *=never-fired…",
  alphabetical truncation hides G in the console line; the ledger row above is the full evidence).
- **§14 schedule semantics:** declared in the Lever docstring — STRUCTURAL, active from ep0 BY
  CONSTRUCTION when on; no λ(t) path to anneal; constancy is the DECISION.
- **Interaction constraint (treatment-arm design note for the council, NOT silently inherited):**
  the chart changes the input coordinate DISTRIBUTION; Fourier-frequency-derived constants
  (`--bank-*`, `--max-bank-freq` stem-Nyquist cap) were derived for the un-charted grid and may
  need re-derivation under the chart.
- **Drift-detector consumer leg:** `[consumers-generic]` in the DSL commit message; the lever
  renders via `describe()`/registry generically.

## 5. PREPARED (NOT launched) training A/B — the #194 completion criterion

Loadable artifact: `experiments/results/ground_frame_chart_20260707/treatment_arm_config.json`
(gitignored-durable; regenerate deterministically via `prepare_ab_config.py` beside it). Both arms
compiled + `validate()`d through the DSL from `sealed_205_program` + store-nothing pose source;
**single-variable delta = the 5 chart flags only** (positional argv diff = the 9 appended chart
tokens). Self-orient is OFF in BOTH arms (v0 fail-closed; keeping it in the control would
confound). Governed launch commands (OPERATOR-GO-GATED — never autonomous):

```
.venv/bin/python tools/launch_witness_run.py --config store_nothing_205 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 \
  --extra-trainer-flags "--no-self-orient"                     # control
.venv/bin/python tools/launch_witness_run.py --config store_nothing_205 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 \
  --extra-trainer-flags "--no-self-orient --ground-frame-chart"  # treatment
```

Memory note (DERIVED): the chart arm builds a per-pair MLX curvelet-feats cache — the self-orient
cf_mx_cache footprint class (~30–40 GiB at n600), built ONCE. The launcher memory preflight gates
it; do not co-schedule with another n600 run. Byte-close: the decode side must apply the same chart
(rule-118 free from the stored ξ) — owed WITH the A/B. Metrics owed: n600 d_seg + per-pair flicker
rate + d_pose through the byte-closed decode; the chart equation is registered THEN.

## 6. Commits

- `c3f4a50a2` module + 25 tests · `049aa0d9f` DSL lever · `c6b83b1b7` equations anchor ·
  trainer wire-in inside `1d6704e5b` (sibling absorption, content verified in HEAD).
