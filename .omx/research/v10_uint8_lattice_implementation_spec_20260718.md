# V10 factor 2 implementation spec — bounded uint8 lattice feasibility

- Date: 2026-07-18
- Lane: `lane_v10_uint8_lattice_20260718`
- Role: `SOLVE` (`training_bytes=0`)
- Authority: Task #532 / completeness factor 2; local advisory build and
  measurement only
- Pointer: `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**

## Objective

Land a reusable `src/tac` solver that constructs a legal camera-resolution
`uint8` preimage while the integer lattice is inside the optimization.  The
verdict chain is explicit:

```text
target scorer plane y
  -> bounded continuous preimage (box-constrained affine equality)
  -> bounded integer block solve (uint8 throughout)
  -> exact hard winner-cell oracle
  -> optional re-linearized integer repair with hard-oracle acceptance
  -> serialize -> parse back -> uint8 decode -> A -> frozen CPU SegNet
```

The implementation must not call `clip(round(P_A x))` a solver.  That baseline
remains a comparator and is expected to reproduce the measured realized-
exactness wall (`62.74` max scorer-plane discrepancy on the original probe).

## Form choice

Selected form: **exact rational block-lattice feasibility plus deterministic
margin repair**.  Randomized/adjacent rounding is retained only as an explicitly
heuristic fallback, never as feasibility evidence.

Reasoning:

1. The canonical resize has disjoint two-tap row supports on each axis.  Every
   scorer pixel therefore owns one disjoint `2 x 2` camera block.  Bounded affine
   preimage feasibility separates exactly by scorer pixel and RGB channel.
2. A full MILP of the frozen scorer is the wrong form: only the final rank-4
   head cone is affine; the camera-to-feature map contains the frozen nonlinear
   SegNet.  An affine-head-only MILP would omit the realization map it purports
   to certify.
3. Classical Dykstra guarantees do not transfer to the nonconvex frozen-CNN
   preimage or the discrete uint8 set.  Reusing the name would overclaim global
   convergence.  The honest form is local sequential linearization with exact
   hard-lattice acceptance and cycle/stall reporting.
4. The half-pixel bilinear weights are derived as integer numerator / common
   denominator values from the input/output geometry, not recovered from
   rounded floating-point coefficients.  Each `2 x 2` block is the bounded
   Diophantine equation `sum(c[k] * x[k]) = T`.  A deterministic gcd-pruned DFS
   orders coefficients, precomputes suffix sums/gcds, bounds each variable by
   the residual and remaining range, and enumerates only the congruence class
   compatible with the suffix gcd.  The last variable is one divisibility and
   bounds check.  A found point is therefore a certified exact uint8 preimage
   for that block.  Fully exhausting the finite tree may certify affine-block
   infeasibility.  A search stopped by a configured node budget is only
   `NOT_FOUND_BUDGET`, never `INFEASIBLE`.
   Exact certification also requires the target's integer numerator as explicit
   custody; a floating target may be checked against that numerator but is never
   silently rationalized into an exact claim.
5. Feasible variable values are visited deterministically around the continuous
   reference.  The first exact feasible point need not be the global nearest
   lattice projection, so no nearest-point or global-optimum claim is made.
   Adjacent-corner rounding may supply a low-residual candidate after an exact-
   search budget expires, but is labeled `HEURISTIC_CANDIDATE` and cannot close
   factor 2 by itself.
6. When the exact frozen scorer still reports a target-cell violation, use its
   target-vs-runner margin gradient only to PROPOSE `+/- 1` integer moves on the
   boundary annulus.  Accept a move only when a fresh hard uint8 scorer forward
   improves the lexicographic feasibility key.  The gradient is never verdict
   evidence.

No global optimum is claimed.  A negative result is scoped to the measured
instance, support mask, repair budget, and representation.

## Owned files

- `src/tac/optimization/uint8_lattice_feasibility.py`
- `src/tac/tests/test_uint8_lattice_feasibility.py`
- `tools/measure_uint8_lattice_feasibility.py`
- `.omx/research/v10_uint8_lattice_feasibility_receipt_20260718.{json,md}`
- `.omx/research/v10_uint8_lattice_DAG_FEED_20260718.md`
- `.omx/research/canonical_equation_candidates_uint8_lattice_20260718.jsonl`
- factor-2 row only in
  `.omx/research/inverse_solve_completeness_matrix_20260718.md`
- this lane's canonical registry/audit/checkpoint rows

## Required module contract

The module will expose typed, NumPy-portable surfaces:

- a certified sparse/disjoint representation of the canonical separable resize;
- exact `A` application and minimum-norm real preimage helpers;
- a bounded continuous preimage projection;
- exact rational, bounded Diophantine block feasibility with statuses
  `FEASIBLE_EXACT`, `INFEASIBLE_EXHAUSTIVE`, and `NOT_FOUND_BUDGET`;
- an explicit integer-numerator target surface; float-only targets remain
  uncertified even when numerically close to a lattice point;
- fail-closed exact-scalar custody: coefficients, denominators, target
  numerators, node budgets, and geometry are integer types excluding booleans;
  targets/tolerances are finite real scalars and tolerance is non-negative;
  no silent numeric/string truncation is permitted;
- factor-2 search arity is bounded to at most four taps before any adjacent-
  corner fallback is materialized; caller tolerance may only tighten the fixed
  machine-derived rational/float agreement bound and can never widen it;
- optional adjacent-corner low-residual fallback labeled
  `HEURISTIC_CANDIDATE`, producing valid `uint8` directly but no feasibility
  certificate;
- orthogonal proof-status and candidate-provenance fields plus a frame-level
  `certified_exact` aggregate; a fallback candidate must never inherit an exact
  proof label;
- no source/reference construction input: a compatibility `reference` value is
  assertion-only, must equal the internally re-derived minimum-norm preimage
  bit-for-bit, and is ignored when choosing lattice values;
- an optional hard-oracle integer repair loop with deterministic ordering,
  finite/integral repair caps, fixed obligation type/shape, projection-drift
  accounting, cycle detection, and scoped terminal status;
- certificate-bound frame/oracle arrays are owned behind immutable bytes and
  cannot be reopened for mutation; exact-`A` inputs are integral and inside the
  uint8 lattice, with int64-safe accumulation and finite aggregate diagnostics;
- serializer and default parser share one maximum decoded-frame byte contract;
  both refuse oversized/nonempty-geometry violations before allocation;
- typed diagnostics including max/mean scorer-plane discrepancy, out-of-gamut
  count before the bounded solve, changed lattice coordinates, hard cell counts,
  and iteration history.

The module must refuse overlapping resize-row supports rather than silently
using the canonical separation proof on another operator.

## Acceptance tests

Behavioral tests, not constant-only checks:

1. `A B = I`, disjoint support, and bounded continuous preimage equality on a
   small contest-shaped resize.
2. Candidate is shape-preserving `uint8`, never out of gamut, deterministic,
   and does not receive/copy the hidden source camera frame.
3. Exact bounded lattice solve recovers a uint8 preimage with zero rational
   projection residual for a target generated from a distant, non-adjacent
   lattice point; this regression prevents a `2^4`-corner search from being
   mistaken for completeness.
4. A deliberately off-lattice target is `INFEASIBLE_EXHAUSTIVE` only after the
   complete bounded search tree is examined; the same target under a smaller
   node budget is `NOT_FOUND_BUDGET`.
5. Bounded lattice solve sharply reduces max projection discrepancy versus
   `clip(round(minimum_norm_real_preimage))` on a constructed out-of-gamut case.
6. A decoded real-size uint8 canary passes canonical `A` parity and an actual
   frozen CPU Torch SegNet hard winner-cell forward; toy threshold oracles do
   not satisfy this control.
7. Known-infeasible/zero-gradient canary stays failed and returns a scoped
   `stalled` result; soft hinge movement cannot fake a hard-cell pass.
8. Hard-oracle acceptance is monotone in the declared lexicographic key and a
   two-state cycle terminates deterministically.
9. Payload serialize/parse-back is byte-exact and the decoded frame is the one
   scored.
10. Torch parity for `A` on a reduced geometry where Torch is available.
11. Malformed certificate inputs (`NaN`, infinite/negative tolerance,
    booleans, non-integral coefficients/denominators/numerators/budgets, and
    forged direct supports) fail closed; valid NumPy integer scalars remain
    accepted.
12. Complex/string/bool arrays, caller-widened tolerance including extreme
    float values, more than four taps, out-of-range exact-`A` values, non-finite
    preimages/oracle debt, changing oracle obligation shapes, mutable result
    arrays, and serializer/parser cap asymmetry all fail closed.
13. Resume reopens each preserved stage and re-derives its source/target,
    exact-`A`, frozen-SegNet, diagnostics, and payload-custody row before a prior
    checkpoint can contribute to a final receipt; executed scorer module paths
    must equal the hashed pinned upstream paths.

## Primary measurement contract

Preferred full n600 is governor-gated.  If the governor is not used for a full
n600 forward, run a deterministic temporal/fragility-stratified real subset and
label it `[macOS-CPU advisory subset]`, non-promotable.  The receipt must bind:

- real `gt_n600.npz` and frozen `segnet.safetensors` hashes;
- pinned upstream `modules.py` hash and exact class targets;
- source pair IDs and selection policy;
- the real-valued `P_A` baseline, `clip(round(P_A))` baseline, and lattice result;
- max/mean `A` discrepancy for each arm;
- hard frozen-SegNet total and per-class `d_seg` for each arm;
- hard target-cell holds specifically on pixels where clip/round failed;
- candidate payload bytes/hash, parse-back frame hash parity, exact command,
  hardware, runtime, and axis labels.

The candidate payload is an honestly named incremental lattice sidecar unless a
complete submission archive is actually built.  It must never be called a
contest archive or score.

## Named confound hunt and controls

Named confound: **soft margin improvement masquerading as a real argmax-cell
flip after uint8/resize/parse-back**.

- Positive control: an independently known feasible uint8 frame/target must be
  recognized as feasible after parse-back.
- Negative control: a deliberately always-false oracle must remain failed with
  a scoped unknown/stalled status; it must become neither feasible nor a global
  infeasibility claim.
- The primary verdict uses hard target-vs-predicted argmax after decoded uint8;
  hinge/gradient values are diagnostic only.
- A second confound is source-copy leakage. The solver derives its own
  minimum-norm preimage from the target. A compatibility `reference` is only an
  equality assertion against that internal value and cannot steer the solve;
  the hidden source frame is retained solely for independent target/custody
  verification and is deleted before the solver boundary.

## Triality and disposition

- DSL: typed solver config plus measurement CLI; no trainer argv and no launch
  authority.
- DAG: dedicated FEED artifact, with factor-2 producer/consumer and remaining
  debts.
- Equation: candidate `bounded_uint8_resize_preimage_cell_feasibility_v1` goes
  only to a temporary JSONL until receiver byte-close parity and MAIN review.
- Completeness factor 2 can move from `MISSING` to `HAVE / PARTIAL` only after
  code, behavioral tests, and a real decoded hard-scorer receipt.  Full n600,
  complete receiver/archive custody, both-scorer interaction, and adoption
  remain blockers.

## Explicit non-targets

- No training, scorer-weight mutation, or learned bytes.
- No paid/GPU dispatch and no heavy n600 scorer forward outside the governor.
- No mutation under
  `experiments/results/levelset_n600_witness_20260717T113932Z`.
- No pointer, canonical equation registry, shared DAG, live-run, or submission
  archive mutation.
- No Fourier proposal or controller.
- No family/paradigm kill from a local stall.

MAIN must independently review the algorithm, real scorer custody, confound
controls, receipt labels, and factor-2 disposition before landing.
