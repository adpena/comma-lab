# Build specification — whole-teacher centered-logit distilled student

**Date:** 2026-07-13  
**Lane:** `lane_whole_teacher_distilled_student_20260713`  
**Mode:** `$0` local cached-data build/fit/measure; `research_only=true`  
**Authority:** NumPy-fp32 reference; MLX is advisory; no score/pointer authority  
**Purpose:** throughput **MEANS** only. Only a byte-closed exact evaluator row can move the pointer.

## Settled premises consumed without re-opening

1. The cheap fixed-feature localizer family is closed at `verdict_scope=FAMILY x
   FEATURE-SOURCE x FIXED-REPLAY`, with req-R across convex/nonlinear and
   single-/multi-source formulations. This build never reruns rounds 3–5,
   `pre_se_locus`, or `pre_se_reopen_a`.
2. This is a structurally different family: a differentiable RGB-to-decision
   student distills the full frozen teacher's centered-logit value and input
   VJP. It has no fixed feature-source or tileability admission premise.
3. The target is the four-dimensional centered-logit decision quotient. Value
   fidelity alone cannot admit the student; exact input-VJP fidelity is the
   decisive gate.
4. `K=20` cannot inclusively kill 95% of a teacher slice when student or update
   cost is positive. Economics uses every charged term and reports advisory
   forward-only and training-gradient tiers separately.
5. The exact-costate reuse sibling's event-sensitive admission cap is `K=2`.
   That cap applies only to its stale-exact-costate controller; it does **not**
   cap the student's distinct periodic exact-anchor cadence. Composition types
   both cadences separately and inherits no speed claim.

## Custody finding that constrains execution

The prompt snapshot said raw n600 rendered-state decision quotients and exact
costates were cached. Current artifact truth falsifies that premise:

- Round 3/4/5 and PRE-SE caches preserve compact support/mass, sufficient
  statistics, and exact tensor hashes; their receipts explicitly record that
  raw exact costates were not preserved.
- `p0_sparse_adjoint_costate_vjp_20260713` temporarily materialized 120 full
  costates, certified deterministic rebuildability, then success-cleaned them.
- The SSD's 600-pair, five-logit cache is source-video/GT-frame custody, not the
  fixed V9 rendered-state replay requested here.

Therefore the probe MUST fail closed before fitting unless a cache manifest
contains exactly 600 unique replay assignments and, for each assignment, a
rendered frame, teacher centered quotient, exact teacher input costate, labels,
and their SHA-256s. The top-level custody contract must also bind the actual R
operator, post-R `float32` RGB 0..255 differentiation surface, frozen teacher
source and weights, exact scalar CE objective/reduction, fixed Helmert quotient
basis, renderer/source/config, upstream/source manifest, generation axis, and
any charged teacher-timing receipt by path/bytes/SHA-256. It MUST NOT silently
substitute source-video logits, normalized/pre-R costates, a rotated quotient
basis, compact support targets, synthetic values, or regenerated teacher calls.
Deterministic cache reconstruction is a separately governed input-custody
action unless a later live-inbox directive authorizes it.

## Owned files

Only these new task files may be authored by this lane:

- `src/tac/scorer_surrogate/whole_teacher_distilled_student.py`
- `src/tac/witness_dsl/whole_teacher_distilled_student_policy.py`
- `src/tac/canonical_equations/whole_teacher_distilled_student_20260713.py`
- `src/tac/tests/test_whole_teacher_distilled_student.py`
- `src/tac/tests/test_whole_teacher_distilled_student_vjp.py`
- `src/tac/tests/test_probe_whole_teacher_distilled_student.py`
- `src/tac/witness_dsl/tests/test_whole_teacher_distilled_student_policy.py`
- `src/tac/canonical_equations/tests/test_whole_teacher_distilled_student_20260713.py`
- `tools/probe_whole_teacher_distilled_student.py`
- `.omx/research/whole_teacher_distilled_student_build_spec_20260713.md`
- `.omx/research/whole_teacher_distilled_student_DAG_FEED_20260713.md`
- `.omx/research/operator_authorizations/GO_PACKET_whole_teacher_distilled_student_20260713.md`
- `.omx/research/whole_teacher_distilled_student_20260713.md`
- `.omx/research/whole_teacher_distilled_student_storage_preflight_20260713.json`
- `.omx/research/whole_teacher_distilled_student_blocker_receipt_20260713.json`
- `experiments/results/whole_teacher_distilled_student_20260713/`

No trainer, `witness_control/*`, `curriculum_dsl.py`, resume registry, live-run
directory, v9/#432 file, or sibling-owned file may change.

## Implementation contract

### Student and references

- A small depthwise-separable convolutional RGB-to-four-margin student, with
  typed widths and deterministic initialization.
- A NumPy-fp32 forward reference whose parameters serialize without framework
  ambiguity.
- An MLX implementation and input-VJP surface with identical parameter layout.
- A Torch-CPU parity/debug surface is permitted only to test formulas when
  Metal is unavailable; it is not relabeled as an MLX measurement.
- All four output coordinates are converted to a five-class zero-sum
  representative before CE, making the quotient gauge explicit.

### Cache and split

- Import the sealed n600 replay assignment/custody helpers from the Round-5 and
  PRE-SE harnesses; do not copy their solver logic.
- Verify exactly 600 unique real assignments spanning the sealed
  `{ep150, ep251, ep275}` renderer states.
- Refuse semantic drift in R, post-R units, teacher source/weights, quotient
  basis, objective/reduction, renderer/source config, or generation custody
  even when every raw tensor hash is internally self-consistent.
- Reuse the sealed deterministic 480 train / 120 untouched heldout split.
- Fit only boundary-region cells, but compute the decisive exact input-VJP
  metrics against the full cached teacher vector. Report boundary-restricted
  metrics separately; never substitute them for the full-vector gate.

### Fit, checkpoints, and hygiene

- Seed every RNG from the typed policy.
- Preserve atomic, stage-named checkpoints for cache validation, fit epochs,
  best fit, heldout measurement, and completion. Resume verifies the policy,
  source bundle, cache manifest, and parameter-layout hashes.
- Student/update timing excludes first-use warmup symmetrically and records the
  active device. A missing Metal device is `UNMEASURED_BLOCKED`, not a CPU
  estimate.
- All transient files use a task output directory and success-only certified
  cleanup. No evidence path points to `/tmp`.

## Metrics and admission

For each heldout pair and aggregate n600 cohort, report:

- forward quotient relative-L2, cosine, absolute error, margin-sign agreement,
  and argmax disagreement; the headline forward QoI is the worst pair;
- full exact input-VJP cosine and relative-L2; the headline VJP QoI is the
  worst pair and this is the decisive gate;
- boundary-restricted VJP metrics as a diagnostic only;
- NumPy-fp32/framework parity and deterministic repeat hashes;
- student forward, student forward+input-VJP, anchor update, exact teacher
  forward, and exact teacher forward+input-VJP timings where actually measured.

No empirical gate threshold may be invented after seeing heldout data. The
typed policy preregisters thresholds and all negative results carry
`verdict_scope` plus `req-R` disposition. If raw n600 custody is absent, the
result is `BLOCKED-DATA-CUSTODY` at `INSTANCE x INPUT-CACHE`, not a student
family failure.

## Economics

For student size `s`, cadence `K`, and tier `t`:

`C_bar_t(s,K) = C_student,t(s) + (C_teacher,t + U(s))/K`.

The table spans all preregistered sizes and cadences, explicitly including
`K=2` composition and `K=20` comparison. A row "pays" only when its measured
charged cost is below the relevant exact-teacher baseline and its tier gate
passes. A forward-advisory win cannot authorize a training-gradient row.

## Triality and one-GO boundary

- DSL: default-off, `research_only=true`, target fixed to centered logits,
  typed student size/anchor cadence/gates, exact-anchor fallback, and a
  separate optional exact-costate-reuse `K_max=2` field.
- Equation: quotient map, VJP error decomposition, and charged cadence law.
- DAG: source custody -> cache gate -> fit -> value gate -> VJP gate ->
  economics -> operator GO -> governed in-loop matched-window launch -> exact
  byte-close evaluation.
- The GO packet may specify the exact integration seam and budget, but it must
  remain `NOT_FIRED` and cannot edit the live trainer.

## Acceptance tests

1. NumPy quotient gauge/CE/argmax invariance.
2. Student parameter layout and deterministic forward repeat.
3. Analytic input VJP versus central finite differences on a tiny real-shaped
   crop, with cosine and relative-L2 checks.
4. Exact metric aggregation catches a single bad worst pair.
5. Cache validator refuses 599 rows, duplicates, wrong replay states, missing
   raw costates, source-video cache substitution, or any SHA drift.
6. Economics proves K=20 inclusive-95 impossible with positive student/update
   cost and keeps the optional sibling `K_max=2` controller typed separately
   from the student's anchor cadence.
7. Policy is default-off, research-only, and emits no trainer argv.
8. Resume/load rejects source or policy drift and preserves stage checkpoints.
