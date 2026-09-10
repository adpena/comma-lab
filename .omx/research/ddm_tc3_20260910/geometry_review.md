# TC3 receiver geometry: scoped implementation and reviews

Owned files: `experiments/ddm_tc3_geometry.py` and
`experiments/test_ddm_tc3_geometry.py`. Parent owns full-n600 pricing, public
receiver integration, source indexing, review-tracker marking and serialization.
No scorer, runtime/submission mutation or long job is performed by this subarm.

## RECALL EVIDENCE

Searched `.omx/research/` by content with `run.track|lane track|lane continuity|
coherent lane|slotwise`; searched the research-index globs and `sub015_DAG*`
with `run.track|lane track|coherent lane|lane_boundary_context_map_bound_v1`;
read the TC3 charter, common contract, TC2 memo and successor charter, governing
NO-FAKE/review rules, PROGRAM, operating manual and live board. Queried the
canonical equations registry via `tools/list_canonical_equations.py --json` for
`lane_boundary_context_map_bound_v1`. Parent handles full task-ledger recall.
The memory registry search `ddm_tc3|lane|run.trac|tc2` supplied only older general
lane context and was not used to transfer a numerical premise.

Beyond the seeds, `wave_f_lane_tracking_coherent_fit_measured_20260702.md`
corrects the older `lane_coeff_tracking_denoising_optimal_survey_20260702.md`:
lossless correspondence bought only 0.5% on that different LBND2 coefficient
codec; per-frame fit jitter dominated and persistent packing exploded columns.
The DAG's openpilot prior does not establish exact SegNet-label prediction.
This changed the design: track actual short runs/endpoints, keep explicit dash
gaps and births, do not revive a fitted global polynomial or promise a large
correspondence gain. No LBND2 numerical result transfers to TC3's context map.

## Derived form before measurement

Variant A ports TC2 LaneGeometry with the identical float64 moments, guards,
binning and tie rule. It has no experiment or SciPy import in the receiver.

Variant B carries two quarter-pixel endpoint positions and two velocities for
each run, with hit count and gap age. Each row first predicts the previous row's
tracks, then matches the new row's distinct runs one-to-one by the smallest sum
of endpoint errors. Ties use existing track order then left-to-right run order.
The fixed alpha=3/4, beta=1/4 filter updates the matched endpoints; slopes are
bounded to two pixels per row. Tracks can cross HPAC slot boundaries. A track
requires two observations and expires after four observed missing rows; new
runs are born explicitly. It does not pool two runs into a mean.

At group g, only labels already observed in groups below g replace the previous
plane's lane mask. The per-group row scan forecasts y before reading row y:
current labels obey both y'<y and g'<g. Previous-plane data are copied at
construction. A strict API refuses skipped, partial, duplicate or reordered
group calls. All B arithmetic is integer, including signed nearest-quarter
updates (ties away from zero) and unsigned distance rounding (half up). No
fit, coefficient, side information or video-specific constant is hidden in code.

## Visible review pass 1: causality and semantic mechanism

Reviewed the full source and tests after the initial Ruff corrections. Traced
each current-frame access from observe through the strict group check, mask
construction and the row loop. Forecasts are emitted before current-row
assimilation; above-row moments in A exclude y. The run matcher gives each
observation and existing track at most one match. Separate runs crossing a
64-column boundary remain separate. Births, four-row disappearance and current
row mutation have behavioral controls. No FAKE constant/metadata result is
substituted for geometry. Clean pass after corrections; no scientific claim.

Shared assumption: a causal boundary map supplies residual information beyond
the existing mixer. This is ASSUMED_AWAITING_VERIFICATION; full-n600 joint refit
and real twins can refute it. Breaking the context-map framing may favor the
generator, but this subarm neither establishes nor closes that path.

## Visible review pass 2: arithmetic, parity and integration boundary

Re-read both complete Python files independently of the initial construction.
Checked the left/right neighbor signs in the two-sided distance, finite sentinel
masking, empty maps, negative-symbol rejection, previous-array ownership,
endpoint update rounding, deterministic association ties and dash expiry.
Future-row nonzero extraction changes only later slices; row-local observations
are unchanged, and padding uses a masked sentinel, so it cannot change a current
bin. B has no transcendental or floating arithmetic. A deliberately retains
TC2's floating arithmetic and still needs cross-host parity testing. Public
receiver, n600 gain and contest runtime are parent-owned unmeasured boundaries.
Clean pass; review tracker must be marked by parent after source indexing and
any parent edits reset these passes.

Same shared assumption as pass 1, with the explicit counterexample that prior
plane fallback can be stale and reduce map usefulness; this is for the actual
calibration/byte comparison to resolve, not a claim of an optimal estimator.

Validation at this point: Ruff passes; eight algorithm controls pass. Synthetic
controls establish implementation behavior only, not empirical lane evidence.

## Real-input implementation controls after both reviews

Axis: `[macOS-CPU advisory / scorer-free implementation controls, frames 0-1]`.
The original move39 parsed token checkpoint was opened read-only with NumPy
memmap, shape `(600,384,512)`, uint8:
`/Volumes/VertigoDataTier/pact/ddm_rp1_rate_directed_predistortion/parseback/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8`.
Its existing retained receipt binds archive
`8877f75d87bf25b410264e08682959c7710cf677307bd5452039ce53835f6bf4`,
field `4aa519a25e4b02afb564498025b366cb9007ea663093c60bf8ce90d079dc8791`
and 117,964,800 bytes. This subarm did not independently hash the full field.

For frames 0 and 1, instantiated both the existing
`experiments.ddm_tc2_codec.LaneGeometry` and the new A with the same previous
plane; asserted exact bin equality before every one of 190 groups, then observed
the same group labels in both. All 380 group comparisons passed. Combined A
reference-plus-port wall times were 0.1646355 s and 0.1652432 s.

B traversed the same two complete frames, asserting each returned map's shape
and bin range, then exact final `.plane` equality. Wall times were **0.2424584 s**
and **0.2676386 s** per 190-group frame. This is a two-frame implementation
timing, not a full-n600 runtime verdict, gain, score or population result.
Eight algorithm tests then passed again in 0.32 s and Ruff passed. No payload
encode, scorer or public receiver execution occurred in this subarm.

Source hashes after these controls:
- `experiments/ddm_tc3_geometry.py`:
  `ede05dae071706e38d9022fba609caac1d254d2b2b9a62c469df9d0e153a0e57`.
- `experiments/test_ddm_tc3_geometry.py`:
  `6de394343ab1ed8a329109ed885ca648306e5d08a4c06e18602c1c82453c30e3`.

Public integration API: `LaneGeometry(previous)` for A or
`RunTrackingGeometry(previous)` for B, then for every `POSITIONS[g]` in order:
`bins = tracker.contexts(positions)`; decode the complete group;
`tracker.observe(positions, symbols)`. Positions must be sorted flattened HxW
indices for the whole next group; symbols are an integer vector in `[0,4]`.
The copied previous plane must be complete or None. `.plane` is the current
uint8 plane with unknown labels represented by 5. Module has only NumPy import.

Handoff is FOLDED into parent ddm_tc3's active integration and full-n600 pricing;
consumer store `/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/`;
fire trigger is the parent's current-field trace completion and reviews. Parent
also owns source indexing/review marking and serializer commit; this subarm
did not mutate the shared index or review state.

LIVE-HYPOTHESES: coherent endpoint continuity plus joint mixer refit may recover
useful information because B preserves run identity and exact short-run shape
across the slots A treats independently. This remains unmeasured on n600.

DEAD-ENDS: no new scientific path was closed. Do not revive the older claim that
correspondence alone must remove most lane-fit entropy: the Wave-F n600 result
found fit jitter dominant on its LBND2 coefficient object. That formulation
boundary does not close TC3's endpoint context map.

Live pointer read after controls: own-vehicle frontier remains
S 0.13766931482209038 @ 180,186 B `[contest-CUDA T4 n600]`, move39 archive
`8877f75d87bf25b410264e08682959c7710cf677307bd5452039ce53835f6bf4`.
No geometry subarm measurement moved it.
