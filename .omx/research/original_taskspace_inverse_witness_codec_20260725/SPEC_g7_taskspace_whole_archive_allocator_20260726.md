# G7 whole-object nonlinear marginal allocator specification

## Status and authority

This landing implements the missing ordered admission surface for exact counted
`P + G + A [+ T]` section bundles.  It is an L0 research-only allocator.  It
does not invoke a scorer, dispatch an evaluation, materialize a contest
candidate, establish an exact score, claim originality, or move a frontier.

The executable truth labels are fixed to:

- `authority = derived_arithmetic_research_only`
- `research_only = true`
- `candidate_claim = false`
- `score_claim = false`
- `evaluation_claim = false`
- `originality_claim = false`
- `promotion_eligible = false`
- `ready_for_exact_eval_dispatch = false`
- `scorer_dispatch_performed_by_allocator = false`
- `pointer_moved = false`

Synthetic deterministic test measurements exercise custody and arithmetic only.
They are not empirical SegNet/PoseNet findings and are not score evidence.

## Problem closed

Raw section bytes are not the contest rate coordinate.  A change to one section
also changes the monolithic directory hashes and CRCs, and DEFLATE can reverse
the order suggested by raw byte deltas.  Pose's square-root term also makes an
axis-local or additive marginal threshold invalid.  The allocator therefore
admits an ordered proposal only after rebuilding and measuring the complete
current object:

`S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`.

For a current prefix `x` and transformed bundle `x'`, acceptance is exactly:

`score_transition_against_dynamic_frontier(x, x').exact_score_delta < 0`.

There is no independent Seg, Pose, section-byte, member-byte, or archive-byte
threshold.  A Seg-harming change can be accepted when its realized Pose/rate
trade is better in the exact total; an axis-local win is rejected when the
other terms make the total nonnegative.

## Public executable contract

The implementation is
`src/tac/witness_dsl/taskspace_whole_archive_allocator.py`.

### Counted input and proposals

`TaskspaceSectionBundleV1` retains exact immutable nonempty bytes for:

1. `predictor_packet` (`P`),
2. `generative_correction_packet` (`G`),
3. `coupled_preimage_packet` (`A`), and
4. optional `terminal_quotient_packet` (`T`).

`TaskspaceWholeArchiveProposalV1` owns a unique bounded ASCII proposal ID and a
callable that transforms the exact current bundle into another exact bundle.
Duplicate proposal IDs are refused before any archive or callback work.

### Whole-object reconstruction and pricing

For the baseline, every proposal trial, and every accepted prefix rebuild, the
allocator calls the existing canonical
`build_taskspace_monolithic_pga_archive(...)` API.  That API creates the full
inner member and both canonical outer encodings:

- `zip_stored`, and
- `zip_deflated` with the outer codec's recorded deterministic profile.

The allocator independently strict-parses both returned archives with their
expected archive and member SHA-256 values, strict-parses the monolithic member,
and requires exact `P -> G -> A -> optional T` payload equality.  It selects the
smaller exact valid archive using the existing outer-codec decision, including
its deterministic STORE tie-break.  It never sums section estimates and never
uses the frozen ep725 ZIP size as a base-rate constant.

Every proposal audit reports:

- per-role raw section before/after/delta bytes;
- total raw section-byte delta;
- full monolithic member-byte delta;
- exact STORE archive-byte delta;
- exact DEFLATE archive-byte delta; and
- selected whole-archive byte delta and the before/trial selected encodings.

### Receiver double-decode custody

The allocator has no built-in semantic receiver.  It passes a typed
`TaskspaceReceiverRequestV1` to an injected receiver callback for each exact
encoding, twice.  The callback receives the exact archive bytes plus expected
encoding, archive SHA/bytes, and member SHA/bytes.  It must return a typed
`TaskspaceReceiverReceiptV1` bound to those fields and containing the decoded
output SHA/bytes plus immutable receipt payload.

The allocator refuses:

- a receipt bound to a different archive or member;
- a non-identical double-decode receipt;
- different decoded-output identities between STORE and DEFLATE; or
- canonical pointer drift around either receiver call.

The callback is responsible for the real semantic receive/decode operation.
The allocator additionally performs its own strict outer/member parse-back, so
neither callback trust nor structural parsing alone is treated as sufficient.

### Realized measurement binding

Only the selected exact archive is presented to the injected typed measurement
callback.  `TaskspaceMeasurementRequestV1` includes the exact selected archive
bytes, encoding, archive SHA/bytes, member SHA/bytes, decoded-output SHA/bytes,
and selected receiver-receipt SHA.

The callback must return `TaskspaceRealizedMeasurementReceiptV1` with exact
matching foreign keys and finite nonnegative `d_seg`/`d_pose`; `d_seg` is also
bounded to `[0, 1]`.  The allocator does not import or invoke SegNet, PoseNet,
`upstream/evaluate.py`, an evaluation dispatcher, or a lane dispatcher.

A measurement is refused if it is swapped across archives, members, decoded
outputs, or receiver receipts, or if either distance is nonfinite.  The dynamic
frontier snapshot is reopened before and after the callback.

### Dynamic pointer and exact transition

The caller supplies an exact `DynamicFrontierTargetSnapshot`.  The allocator
reopens and verifies it:

- before baseline work;
- around every receiver callback;
- around every measurement callback;
- inside every delegated canonical transition audit; and
- before returning the final result.

Stale, replaced, edited, forged, or custody-drifted pointer objects fail closed.
The exact transition is delegated to
`score_transition_against_dynamic_frontier`; the allocator does not fork the
score equation.  All exact and diagnostic transition fields are checked for
finiteness before use.  The transition's `improves_score` flag must agree with
the strict `exact_score_delta < 0` predicate.

### Prefix semantics and rollback

Proposals are evaluated in caller-supplied order against the last accepted
prefix.  For each proposal:

1. transform the immutable current section bundle;
2. build/parse/price both full outer archives;
3. double-decode both exact encodings;
4. measure the selected exact archive/output;
5. compute the exact finite nonlinear transition from the current prefix;
6. reject without changing the prefix when total delta is nonnegative; or
7. if negative, rebuild, double-decode, and remeasure the accepted prefix from
   scratch before making it current.

The accepted-prefix repeat must preserve exact bundle, both archive builds,
both decoded-output identities, selected archive/member/output identity, and
realized component distances.  Any drift blocks the allocation.  This second
pass makes the accepted prefix a fresh measured state rather than an alias to a
proposal-trial object.

This is an ordered greedy allocator over atomic proposals.  It does not claim
global optimality and does not rescue a jointly profitable multi-proposal bundle
whose individual caller-supplied atoms are uphill.  Callers that need an atomic
multi-coordinate move must package that move as one proposal transform.

## Fail-closed blocker classes

`TaskspaceWholeArchiveAllocatorError` covers:

- invalid/duplicate proposal identity or wrong transform return type;
- exact archive/member/section parse-back mismatch;
- receiver archive/member binding mismatch;
- nondeterministic double decode;
- STORE/DEFLATE semantic-output disagreement;
- measurement archive/member/output/receiver binding mismatch;
- nonfinite or invalid component distances/transition fields;
- accepted-prefix rebuild/measurement drift; and
- attempted in-place mutation detectable through bundle identity.

Dynamic-pointer failures retain the canonical
`DynamicFrontierTargetError` type.  Existing monolithic/outer codec failures
retain their own strict error types.  No failure is converted into acceptance.

## Canonical versus unique decisions

| Layer | Decision | Reason |
|---|---|---|
| Monolithic member grammar | Reuse canonical P/G/A[/T] builder/parser | One directory and role-order authority |
| Outer archive bytes | Reuse canonical STORE/DEFLATE builder/parser | Exact measured rate coordinate and deterministic tie-break |
| Score geometry | Reuse dynamic-frontier transition wrapper | No equation or target fork |
| Semantic receiver | Inject typed callback | Vehicle-specific decoding remains with the real receiver owner |
| Realized measurement | Inject typed receipt callback | Allocator must not dispatch or counterfeit scorer authority |
| Proposal order/prefix state | New allocator-local logic | This was the missing whole-object nonlinear marginal surface |

The module imports the sibling-owned public monolithic functions
`build_taskspace_monolithic_pga_archive` and
`parse_taskspace_monolithic_pga_member`.  Integration therefore depends on
those function names and their current exact section-order contract remaining
public.  It does not import or edit the ep725 semantic adapter/receiver internals.

## Wire-in and operational profile

- Compiler/inflate hook: not wired; this is a research allocation API.
- Trainer/DSL hook: not wired; proposals are exact bundle transforms, not flags.
- Materializer hook: intentionally left to the parent lane.
- Scorer/evaluator hook: forbidden here; injected receipts only.
- Dispatch hook: absent; `ready_for_exact_eval_dispatch=false`.
- State/receipt hook: typed in-memory result; no candidate registry mutation.

No long job, training, materialization, scorer run, dispatch, or bulky artifact
is launched or created.  Storage waterfall, resume checkpoints, and SSD cleanup
are therefore not applicable to this source/test landing.  Any future real
measurement callback remains bound by the project-wide n600, realized-through-R,
resumability, storage, lane-claim, and authority rules.

## Synthetic behavioral verification

The focused test module is
`src/tac/witness_dsl/tests/test_taskspace_whole_archive_allocator.py`.
It exercises 19 deterministic cases without scorer calls, including:

1. raw section ranking reversed by full monolithic DEFLATE pricing;
2. Seg harm accepted when Pose/rate yield negative exact total delta;
3. stale and swapped dynamic-pointer refusal;
4. pointer mutation during measurement refusal;
5. measurement/archive/output binding refusal;
6. receiver/archive binding refusal;
7. repeat-build determinism and two receiver calls per encoding;
8. nondeterministic double-decode refusal;
9. STORE/DEFLATE decoded-identity disagreement refusal;
10. rejected-proposal rollback before the next transform;
11. accepted-prefix full rebuild and remeasurement;
12. accepted-prefix measurement-drift refusal;
13. axis-local Seg win rejected when total score worsens;
14. duplicate proposal IDs refused before callbacks;
15. nonfinite measurement refusal;
16. optional-T raw/member/whole-archive delta reporting;
17. zero-delta strict rejection and false-authority labels; and
18. wrong transform return-type refusal.

These are structural/arithmetic tests.  They do not show an empirical
distortion improvement, a candidate, a score row, or frontier movement.

## Pointer-delta honesty

This landing moves no canonical candidate or exact frontier pointer.  It closes
an infrastructure gap needed to allocate future exact same-object proposals.
The mission-level exact score remains unchanged by this work.
