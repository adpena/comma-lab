---
title: DDM M2 kinetic Laguerre at-tolerance probe build spec
date_utc: 2026-07-23
lane_id: lane_ddm_m2_kinetic_laguerre_probe_20260723
research_only: true
score_claim: false
main_landing_review_required: true
---

# Purpose

Build and fire the bounded M1 rank-1 test. The probe asks whether a counted,
deterministically decoded kinetic anisotropic Laguerre program reaches the
sealed n600 label error box (`136839 / 117964800`) inside the predictor-home
budget (`100099` bytes). A receiver row is allowed only after that Stage-A
gate.

# Stores consulted

- delegated authority SHA-256
  `cb5bc9d90cc9285407cd830dc9c4f310aabb706ce437cfa4effab3c5499248a8`
- `CLAUDE.md`, `AGENTS.md`, craft handoff, `PROGRAM.md`
- v7.5 §8 operating contract and v8 per-class SPEC
- M1 missed-optimum memo, config, and DAG FEED
- sealed v8 Laguerre/hybrid feasibility memo
- v19b receipt and immutable n600 archive
- `partition_collapse.py`, frozen v14 scorer loader, and v19/v19b receiver
  measurement helpers
- canonical lane/task/frontier and subagent-ownership surfaces

# Representation and fitted state

For every registered cell

`K in {64,128,256,512} x degree in {1,2,3} x metric x temporal`,

the fitted object is a class-labelled set of weighted sites

`P_i(x,t) = (x-q_i(t))^T M_{c(i)}(t)(x-q_i(t)) - w_i(t)`.

The three metric modes are:

1. `isotropic_power_control`: `M_c=I`;
2. `shared_chart_anisotropic_spd`: one measured diagonal SPD chart per class,
   encoded once per temporal segment;
3. `projective_depth_stratified`: class charts additionally scale the row
   coordinate by a deterministic depth stratum.

The independent-frame control encodes one quantized generator state per frame.
The kinetic arm encodes piecewise-polynomial site/weight trajectories. It
includes the counted quantized `xi(t)` stream and a fitted linear xi advection
term; it is not allowed to borrow uncounted target state. Temporal knot count is
chosen deterministically by reverse waterfilling under the `100099`-byte home,
never by inspecting the verdict after the fact.

Class-specific nearest-power distances are evaluated with a transformed
`scipy.spatial.cKDTree`; class weights remain in the final cross-class power
comparison. A NumPy-fp32 brute-force kernel is the test authority on bounded
arrays. Every encoded program must parse back byte-identically and reproduce
the same cell digest twice.

# Fit and evaluation plan

1. Preflight exact source hashes, free space, config schema/authority, and
   memory ceiling. Local output is small receipts/program bytes only; no bulk
   RGB tree is retained.
2. Fit per-frame deterministic class-quantile sites once for the maximal K.
   Smaller K values are nested prefixes. Kinetic coefficients are fitted from
   these states plus the counted xi stream.
3. Fire n64 as compute/integrity evidence only. Preserve every cell receipt and
   program.
4. Fire all registered cells at real n600, chunked by frame batches, with an
   atomic aggregate checkpoint per cell. No n64 ranking is promoted.
5. Race three real coders over the exact same canonical program:
   Brotli q11, XZ preset9 extreme, and a split-metadata/Rice-Golomb stream.
   The minimum complete parse-back-valid stream is the charged Stage-A byte
   count.
6. If and only if a program reaches both Stage-A limits, materialize its
   scorer-free RGB pullback in memory, run the real frozen scorer in preserved
   batches, and require exact double replay plus the Pose and 200KB gates.
7. The v19b correction composition is measured only through an actual common
   receiver. If no Stage-A winner exists, its row is explicitly
   `NOT_RUN_STAGE_A_GATE_CLOSED`; no additive/synergy credit is inferred.

# Fail-closed points

- The design-only config must be transitioned to execution by this exact
  delegated authority receipt; a bare `execution_allowed=false` config refuses.
- Missing input/hash/model/storage custody refuses before fit.
- Independent per-frame payloads are charged in full.
- No label map, scorer weights, source RGB, or target-derived decoder code may
  enter a program.
- n64 cannot decide fidelity, family status, or the matched-fidelity race.
- A Stage-A failure scopes only to
  `FORMULATION:KINETIC_ANISOTROPIC_LAGUERRE_REGISTERED_LADDER`.
- Local receiver evidence remains
  `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`,
  `promotion_eligible=false`, and `not_a_candidate=true`.

# Triality and downstream hooks

- Lane: `lane_ddm_m2_kinetic_laguerre_probe_20260723`.
- DAG: land one dated FEED with the measured Stage-A gate and receiver
  disposition.
- Equation: register a generator-rate law only if n600 yields a stable,
  non-vacuous matched-error relationship. Otherwise record equation leg N/A.
- Sensitivity/Pareto/bit allocator/autopilot/posterior: research-only measured
  receipt is the handoff; no dispatch hook is enabled. The probe itself is the
  preregistered disambiguator between generator and describe-line forms.

# Verification and landing

- focused deterministic kernel/config/coder/checkpoint tests;
- Ruff and Python compile;
- real n64 then n600 re-derivation;
- review tracker on the new Python files;
- serializer commits;
- MAIN re-derives hashes, all 72 n600 cell dispositions, winner selection,
  Stage-B gating, false-authority labels, and the final branch diff before
  merge.
