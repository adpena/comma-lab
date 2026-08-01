# G18 — closed G14 interaction-feedback harvest

Date: 2026-07-26  
Lane: `g18_taskspace_g8_a3_interaction_feedback`  
Status: frozen before implementation  
Axis: `[macOS-CPU advisory]`, real n2 only, noncomparable to every n600 or contest-authority row

## Purpose

G14 measures a coupled four-cell experiment: no-G8/PASS-A, no-G8/A, G8/PASS-A,
and G8/A.  G18 is the deterministic, dense-free adapter that prevents any of
that signal from becoming orphaned.  It consumes only a G14 final receipt that
the frozen G14 parser accepts; it does not reopen archives, run the receiver or
scorer, inspect a live run, append a ledger, dispatch, or change a pointer.

The adapter preserves the complete four-way `Z/T/H` partition, every exact row
occurrence (including repeated matched-control occurrences), every exact
component/whole-object transition, every finite byte ceiling, and every
nonlinear G-by-A interaction.  Family, order, palette, prefix, G8 nondomination,
and retained-for-A decisions remain first-class coordinates rather than prose.

## Public surface

New module:
`src/tac/witness_dsl/taskspace_g8_a3_interaction_feedback.py`

```python
build_taskspace_g8_a3_interaction_feedback(
    receipt: bytes | Mapping[str, Any],
) -> dict[str, Any]
```

Both inputs are reparsed through G14's frozen `parse_final_receipt`.  Mapping
inputs are first canonically serialized.  The output is canonical-JSON-safe and
deterministic: no clock, PID, host, filesystem read, or live pointer lookup.

## Closed record

The record schema is `tac.taskspace_g8_a3_interaction_feedback.v1` and contains:

1. exact source/canonical receipt SHA-256 custody;
2. copied G14 truth and noncomparable classification, never upgraded;
3. the exhaustive dense-free four-way `Z/T/H` partition with per-target-class
   counts and closure check;
4. a path-addressed row-occurrence inventory for G0, G8, G8+A, and matched G0+A,
   plus the diagnostic semantic-G control kept explicitly outside selection;
5. exact per-axis and nonlinear whole-object deltas, finite real and strict
   integer byte ceilings, and the G-by-A interaction
   `S(G8,A)-S(G8,PASS)-S(G0,A)+S(G0,PASS)`;
6. family/order/palette/prefix coordinates and exact G14 nondominated/retained
   membership;
7. bounded scorer/archive custody, including the n2 per-pair values and hashes;
8. narrow downstream payloads for pair ranking, Pareto history, measured
   marginal bit-allocation input, Tier-A autopilot observability, and deferred
   probe-outcome registration;
9. explicit non-consumption by the authoritative continual-learning posterior.

## Fail-closed invariants

G18 raises `TaskspaceG8A3InteractionFeedbackError` before returning a record if:

- the frozen G14 parser rejects the receipt;
- any target-relative field (`below_target`, `gap_to_target`, or
  `target_sublevel_admission`) occurs anywhere in the receipt;
- G14 truth or comparison labels imply any authority upgrade;
- the four `Z/T/H` masks are non-exhaustive, class counts do not close, summary
  counts disagree, or dense masks/frames are claimed serialized;
- G0 programs and rows, or G8 branches and rows, are not exact one-to-one
  aligned; the baseline is not the first G0/PASS row; branch/program hashes,
  sources, modes, or row counts disagree;
- any screened branch, retained ID, nondominated ID, or conditional treatment
  refers to a missing/duplicate object;
- a retained branch has no non-PASS treatment, a nonretained branch receives a
  treatment, or duplicate treatment identity appears;
- a stored transition, interaction, selected row, or G14 aggregate differs from
  exact recomputation;
- any production row occurrence is dropped by the feedback inventory or any
  interaction lacks sensitivity, Pareto, bit-allocation, autopilot, and probe
  foreign keys.

No fixed segment/pose/rate threshold is introduced.  `exact_score_delta < 0` is
the only improvement predicate, and the finite byte ceiling is derived from the
same exact nonlinear score equation.

## Existing-system boundaries

- `tac.multi_granularity_sensitivity.per_pair_axis_score_contribution` is used
  only for pair ranking; its pose value is explicitly non-additive.
- `tac.boosting.pareto_front.ParetoFrontTracker` receives every occurrence on
  one `[macOS-CPU advisory]` tracker.  G14's original 3-axis nondominated set is
  preserved separately.
- `tac.witness_sensitivity_bitalloc.score_delta` independently verifies every
  exact transition.  G18 emits measured atomic marginals; it does not invent a
  bit budget or pretend these rows are tensor precision ladders.  Each marginal
  is explicitly bound to its measured base archive.  Actions are not assumed
  additive or independent: mutually exclusive alternatives and composed macros
  require byte-closed bundle acquisition/re-measurement.  This adopts the useful
  dynamic-cost/interaction caution from HOPE (arXiv:2607.21366) without importing
  its static payoff assumptions, which do not hold for the active V9 HOSC/FiLM
  vehicle.
- `tac.cathedral.consumer_contract` is represented as Tier-A observability only:
  zero predicted adjustment, no dispatch, no promotion.
- `tac.probe_outcomes_ledger.register_probe_outcome` is not called.  G18 emits
  deterministic kwargs for the root owner to append after reviewing the real
  run receipt.
- `tac.continual_learning.posterior_update` is authority-only and therefore
  incompatible with n2/macOS advisory evidence.  G18 emits a typed blocked
  handoff rather than laundering the row into the posterior.
- `tac.sensitivity_map.SensitivityMap` is a model-gradient/CUDA surface and is
  not instantiated.  G18's pair-ranked finite differences remain explicitly
  distinct.

## Acceptance tests

Synthetic receipt-shaped tests only:

- deterministic byte/mapping round trip;
- exact row-count and four-way closure;
- exact nonlinear deltas, byte ceilings, and interaction;
- every retained family receives treatment and all six hooks have foreign-key
  coverage or an explicit typed incompatibility;
- rejection of dropped rows, aggregate drift, target-relative fields,
  authority upgrades, and orphaned interaction references;
- no live run, archive, scorer, eval, pointer mutation, or ledger append.

Pointer delta: none; G18 is an advisory signal-custody landing, not goal
progress.  Root ownership begins only after a real G14 final receipt exists.
