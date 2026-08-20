# Predictor upgrade: xi-advected prior plus per-class charts

`lane_id=lane_predictor_upgrade_xi_chart_578_20260721` · `task=578` ·
`research_only=true` · `[macOS-CPU advisory]` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED` · `MAIN_REVIEW_REQUIRED=true`

## Objective

Replace the measured five-site compatibility raster as the compression-side
generic predictor with the doctrine predictor: xi-advect the prior decoded
cell field, reconcile it with counted per-class static charts and the measured
region-adjacency graph, then emit only residual exceptions. Measure predictor
satisfaction and the resulting seed rate/distortion curve at n64 and n600.
This is a cell-description and byte-accounting experiment, not a camera-RGB
realization, `upstream/evaluate.py` score, or pointer-moving result.

The predecessor is decisive about the intervention. The five-site predictor
measured n600 satisfaction of Road 0.508503507, Lane 0.082758141,
Undrivable 0.616929787, Movable 0.997751186, and MyCar 0.865297321. Its
tightening marginals lost to the byte price. Do not add more exceptions to
that default and call it a predictor upgrade.

## Ownership and collision boundary

- Own one new compression-side predictor/measurement module, one CLI wrapper,
  focused tests, and dated Task #578 receipts/memos.
- `src/tac/optimization/predict_project_receiver.py` PROJECT realization is
  owned by sibling `realization_g2_lattice`. Do not edit its projection,
  inverse-R, realization, hard-oracle, or ladder stages.
- Coordinate through `src/tac/optimization/predict_project_schema.py`. Extend
  only the generic-predictor policy/custody surface required to name and parse
  this predictor. Persistence must remain additive and legacy-compatible:
  old v0 seeds without the new policy still parse and reserialize identically,
  while new seeds name the doctrine policy explicitly and fail closed on
  unknown policy fields or values.
- Reuse the existing serializer, receiver primitives, B2 composer artifacts,
  `tac.lie`, `PoseTargetEgoEstimator`, #234 Hungarian movable tracks, #595 lane
  chart custody, #139 hood clamp custody, registered LawRefs, and the frozen
  n600 cache. Do not fork any of them.
- Do not edit upstream, frontier pointers, main, other worktrees, live run
  directories, or sibling-owned files.

## Predictor contract

Build a deterministic, scorer-grid `uint8` predictor callable whose inputs
make receiver causality explicit:

1. For pair 0, use chart-derived initialization only. For pair `t>0`, start
   from the prior decoded cell field supplied by the caller. The measurement
   harness may use frozen `lstars[t-1]` only as an explicitly labeled oracle
   stand-in for that prior decoded field; it must not serialize those full
   fields or count them as free chart bytes.
2. Advect the prior field by the measured PoseNet-to-xi convention through
   canonical `tac.lie`/`PoseTargetEgoEstimator` custody. Use relative motion
   between adjacent pairs, not the absolute five-site shift. Record the
   consumed LawRefs and the G1 custody anchor: 0.279 px median bulk transport
   error with 8.4% event tails. Do not copy unregistered calibration literals
   into the schema.
3. Reconcile by class in canonical class-ID order, self-detected from the
   cache's scorer labels rather than luminance sorting: Road=0, Lane=1,
   Undrivable=2, Movable=3, MyCar=4.
   - Lane: reuse the #595 coherent arc-length/dash chart bytes. Do not claim
     that its mask-only F1 is d_seg or that the phase-conditioned correction
     already won; the measured phase R1 comparison remains part of D4.
   - MyCar: reuse the #139 static hood/clamp component and its IoU 0.994
     custody. Measure its actual satisfaction here.
   - Road and Undrivable: derive compact static ground slices from the n64
     development prefix, serialize the actual chart payload, parse it back,
     and report its finite bytes. Do not train those charts on held-out n600
     pairs without declaring the split.
   - Movable: keep the prior-advected bulk and reuse #234 Hungarian track
     updates; do not replace the already-99.8% predictor with a new box coder.
4. Resolve overlaps only along measured adjacency edges. Use a deterministic
   per-class priority/tie rule recorded in policy custody. No full n600 target
   raster, scorer weights, or GT table may enter the serialized predictor.
5. Double execution with identical inputs must produce byte-identical cell
   fields and identical exception streams.

## Chart and rate custody

Every chart/interpreter section reports raw bytes, deterministic zlib-9 bytes,
SHA-256, source/custody, and whether bytes are counted or generic/free. The
interpreter is generic/free; all video-derived chart parameters are counted.
Bind the #595 41,303-byte coherent dash chart as a measured external custody
row, not an invented replacement. If that payload cannot be parsed directly
by this module, record it as reused external counted custody and keep its
bytes out of a duplicate section.

The predictor policy receipt must state exactly which prior field is assumed
available at decode. Any row using `lstars[t-1]` is an oracle upper bound on a
future decoded-field chain, not receiver-closed evidence.

## D1 and D2 measurement

Use `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz`
with expected SHA-256
`cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
Record the `/Volumes/VertigoDataTier` storage preflight. Write resumable,
atomic, per-chunk checkpoints under
`/Volumes/VertigoDataTier/pact/evidence/predictor_upgrade_20260721/`, preserve
all stages, and refuse source/config drift on resume.

Run n64 first as `MEASURED_DEVELOPMENT_PREFIX`, then n600 as the decisive
`MEASURED [macOS-CPU advisory]` row. For each class and each available
stratum (`cell_interior`, `boundary_codim1`, `movable_track`,
`critical_event`) report correct, total, and satisfaction fraction. Also
report overall satisfaction. The target comparator is at least 0.99 for every
class, but report an honest negative if any class misses.

Every miss must have one exclusive primary cause:
`advection_residual`, `static_chart_miss`, `adjacency_or_tie_miss`,
`movable_track_miss`, or `critical_event_exception`. Give counts by class and
stratum. The 0.71 `jitter_bound` custody is a boundary-error diagnostic, not a
license to relabel all misses as jitter.

## D3 seed curve

Starting from the predecessor loose/knee/tight policy, re-emit three nested
seeds with the doctrine predictor named as the default and only predictor
violations serialized as exceptions. Do not silently apply every old
constraint. For every point report:

- exact PPCS bytes and SHA-256 after parse-back/re-serialization;
- deterministic compressed bytes and per-section byte accounting;
- exception count by class and stratum;
- cell-description `d_seg` before and after exceptions;
- advisory `100*d_seg + 25*seed_bytes/37_545_489` labeled as non-score;
- comparisons with the predecessor loose/knee/tight rows, 34.938972078984,
  and the 216,222-byte box.

The rate curve must distinguish doctrine predictor bytes, reused external
chart bytes, and exception bytes so double counting is visible. If a chart
plus exception point exceeds the box, say so.

## D4 causal jitter comparison

Emit the measured boundary-normal residual stream of the doctrine predictor.
Compare exception coding against the existing phase-conditioned R1 rung on
the same error inventory. Report raw/compressed bytes, corrected cells,
remaining errors, and whether the comparison is mask-only or through-R.
Never promote mask F1 into d_seg. If equal-fidelity through-R custody is
absent, the verdict is `FORMALIZATION_PENDING_THROUGH_R_REALIZED_SCORE_RECOVERY`.

## Triality and landing

- DSL/schema leg: generic-predictor policy and counted-section custody.
- DAG leg: dated `predictor_upgrade_xi_chart_DAG_FEED_20260721.md` showing
  prior decoded field -> xi advection -> class charts/adjacency -> exceptions
  -> PROJECT sibling boundary.
- Equation leg: dated canonical equation module or registered equation row for
  the predictor/reconciliation law, with empirical status no stronger than
  the measurements. Reuse LawRefs rather than copying constants.
- Durable outputs: measurement CLI/module, focused tests, machine-readable
  receipt JSON, findings memo, DAG FEED, canonical reuse manifest, and SSD
  evidence stages.

All result rows carry `MEASURED`, `DERIVED`, or `SPECULATIVE`; every negative
has narrow `verdict_scope`; every artifact says `score_claim=false`,
`promotion_eligible=false`, pointer unchanged, and MAIN review required.

## Verification and commit handoff

1. Focused pytest for schema legacy/new-policy round trips, deterministic
   advection/reconciliation, no hidden full-raster payload, exception-only
   emission, resume/source-drift refusal, and receipt classifications.
2. Run n64 then n600, JSON parse checks, Ruff on changed Python, py_compile,
   and `git diff --check`.
3. This executor does not commit. The parent reviewer will adversarially
   inspect all diffs, run final tests, obtain two clean `review_tracker`
   passes for every changed Python file, and commit through
   `tools/subagent_commit_serializer.py` with post-edit SHA-256 values.

## STORES CONSULTED

Delegated authority and both live inboxes; `CLAUDE.md`; `AGENTS.md`;
`PROGRAM.md`; craft handoff; vehicle OS; v7.5/v8 specs; predecessor B2 build
spec, receipt, findings, DAG, and reuse manifest; Task #597 schema/receiver;
#595 lane chart/custody; G1 worldsheet receipt; hood and movable-track
components; lane/task/subagent state; frozen n600 cache; SSD storage state.
