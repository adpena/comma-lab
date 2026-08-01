# Selective topology acquisition from the canonical Z/T/H partition (2026-07-26)

Status: implementation contract for a bounded encoder-only research surface.
No scorer, evaluator, real-data run, candidate materialization, dispatch, score
claim, or frontier mutation is authorized by this specification.

Owned landing:

- `src/tac/witness_dsl/taskspace_selective_topology_acquisition.py`
- `src/tac/witness_dsl/tests/test_taskspace_selective_topology_acquisition.py`
- this specification

The frozen G10, G12, G13, and G14 implementations and specifications are
read-only dependencies.  Existing `TACG1C`, `TACG8S1`, `TACA3P1`, `TACA8P1`,
and G7 receipt names retain their exact historical meanings.  This landing
does not rename diagnostic evidence or synthesize a receiver receipt.

## 1. The coordinate

For current semantic labels `Z`, encoder target labels `T`, and labels `H`
measured by the frozen scorer on the current realized camera plane, the
canonical exhaustive partition is:

1. `Z=T, H=T`: closed;
2. `Z=T, H!=T`: same-class realization debt;
3. `Z!=T, H!=T`: topology debt;
4. `Z!=T, H=T`: fortunate semantic mismatch.

Exact semantic G over all `Z!=T` is a diagnostic upper-bound control.  It
destroys row 4 even though the current realized witness already satisfies the
scorer there.  PASS preserves row 4 but cannot acquire row 3.  The new
coordinate changes a selected prefix of row 3 to `T` and changes no other
semantic cell:

`Z' = where(selected_row3, T, Z)`.

For every emitted packet, strict TACG1C decode must prove:

- `changed(decoded_Z, Z) == selected_row3` exactly;
- every selected cell decodes to its exact target class;
- every row-4 semantic cell is bit-identical to `Z`;
- all unselected rows, including unselected row 3, are bit-identical to `Z`;
- the packet carries no dense label, scorer, RGB plane, or target table.

The generic predictor-preserving overlay consequently owns camera supports
only for the selected semantic cells.  That proves exact exclusion of row-4
cells from G ownership; it does **not** prove that the nonlocal frozen scorer's
post-topology label remains `T` at row 4.  Such a claim is forbidden until the
actual post-topology realization is scored as `H'`.

## 2. Existing TACG1C is sufficient for bounded exact selection

No new wire grammar is needed for a bounded prefix.  `TopologyEventV1` admits
one-pair, zero-transport, one-row boxes, including one-cell boxes.  TACG1C
begins from mutually exclusive predictor masks and composites roles in
`REALIZATION_PAINT_ORDER`.

For each selected source-class to target-class transition:

1. emit a target-role `birth` over the exact selected box;
2. when the source role has higher paint precedence than the target role,
   emit a source-role `death` over the same exact selected box;
3. use `lifetime=1` and zero transport gains.

This is the same precedence law already used by the exact bounded-target
control, restricted to the selected row-3 mask.  Strict apply-back, rather
than the construction argument alone, is authority for exact mutation support.

Typed blockers must distinguish at least:

- empty topology debt or an empty resolved prefix;
- predictor/evidence/teacher custody mismatch;
- TACG1C uint16 event-cardinality overflow;
- TACG1C compile/parse/apply refusal; and
- any decoded mutation outside the exact selected row-3 mask.

The last case is the trigger for a future versioned grammar extension.  It
must never be converted into a partial success.  Since bounded singleton and
row-run programs are representable today, this landing adds no speculative
wire extension.

## 3. Encoder custody and four-way telemetry

`EncoderOnlySelectiveTopologyEvidenceV1` owns immutable encoder-side `Z`, `T`,
and baseline `H` arrays plus exact hashes and foreign keys for:

- the contiguous source-pair window;
- predictor state and counted P section;
- current camera Y1;
- the frozen target scorer-RGB content hash needed by a later G8 acquisition;
- frozen scorer;
- current-realization forward receipt; and
- target forward receipt.

It has no serializer.  Its finite binding hashes array identities and custody
metadata; `serialized_dense_evidence_bytes=0` is invariant.  `Z` must equal the
decoder-owned predictor labels byte-for-byte.

The module reuses the public canonical
`SameClassRealizationCellPartitionTelemetryV1` type.  Every acquisition and
proposal receipt nests the complete four-way counts, per-target-class counts,
and mask hashes.  Row 3 may not be reported without rows 1, 2, and 4.

The caller supplies a real `EncoderOnlyTeacherEvidenceV1`.  Its target-label
bytes must equal `T`, and its teacher-event count must equal all baseline
semantic debt `count(Z!=T)`.  The selective compiler passes this same object to
TACG1C.  It does not invent PBR, obligation, oracle, or dense-Y hashes, and it
does not relabel `Z'` as the frozen target.  Therefore the generic compile
receipt truthfully reports residual target debt for a partial prefix.

## 4. Bounded proposal lattice

`SelectiveTopologyAcquisitionPlanV1` contains explicit, sorted, unique positive
singleton prefix sizes.  Resolution clamps each requested size to the exact
row-3 population, deduplicates the result, and applies the TACG1C ABI bound.
These are enumeration coordinates, not independent Seg/Pose/rate admission
thresholds.

Every resolved prefix is emitted under all closed orders:

- `CANONICAL_ADDRESS_V1`: pair, row, column;
- `FORTUNATE_CLEARANCE_DESCENDING_V1`: greatest deterministic Manhattan
  clearance from baseline row 4 first, then canonical address.  This is an
  ordering prior only; it applies no clearance cutoff and makes no scorer
  survival claim.
- `TRANSITION_MASS_DESCENDING_V1`: most populous exact `(source,target)` row-3
  transition first, then canonical address.  This is an MDL-oriented ordering
  prior only.

Every prefix/order is represented under all closed grammar families:

- `SINGLETON_BOX_CONTROL_V1`: exact one-cell boxes;
- `SOURCE_TARGET_ROW_RUN_BOX_V1`: exact horizontal runs grouped by source and
  target class;
- `TARGET_BIRTH_SOURCE_DEATH_ROW_RUN_BOX_V1`: target births coalesced across
  source classes, with precedence-required deaths grouped by exact source and
  target transition.

All three families must decode to the same selected semantic support.  They
are alternative byte grammars, not distinct evidence.  Proposals retain every
interpretation receipt, while `unique_program_proposals` deduplicates on exact
compiled TACG1C packet bytes/SHA-256 in deterministic first-representative
order.  Program-object hashes or estimated bytes are not a substitute for
wire-byte dedupe.

No component, class, Seg, Pose, or byte threshold admits a proposal.  Ordering
is advisory.  G7's exact whole-object nonlinear transition is the only
admission surface.

## 5. Proposal and receipt surface

Each `SelectiveTopologyProgramProposalV1` contains:

- exact `CompiledGenerativeCorrectionV1` and its counted TACG1C packet;
- family, order, requested/resolved prefix, selected-mask hash and counts;
- source-to-target transition census;
- exact birth/death/event counts and packet bytes/SHA-256;
- baseline four-way telemetry;
- exact post-topology semantic hash and residual semantic-debt count;
- row-3 selection and row-4 preservation proof flags; and
- false-authority labels for scorer, through-R, whole-object allocation,
  candidate, score, exact evaluation, and promotion.

Receipts are canonical finite ASCII JSON and strict parse/re-emit.  Dense
evidence, target tables, scorer planes, and RGB planes are never serialized.

## 6. Fresh post-topology cascade

Baseline G8 debt is invalid after any semantic topology mutation.  The lawful
causal chain is:

`compile selective TACG1C -> realize/overlay actual Y1' -> run frozen scorer on
actual Y1' -> obtain H' -> recompute all four rows for (Z',T,H') -> acquire G8
only on {Z'=T,H'!=T} -> compile A against the actual post-G8 Y1`.

`bind_selective_topology_post_forward(...)` is encoder-only.  It accepts
caller-produced actual post-topology `H'`, scorer RGB/Y1 custody, and a new
forward-receipt hash.  It must refuse the baseline forward receipt identity,
recompute complete four-way telemetry, and expose the fresh row-2 mask only as
immutable encoder evidence.  It records both baseline and fresh debt-mask
hashes and `baseline_g8_debt_reused=false`.

Fresh G8 acquisition also needs the frozen target scorer RGB from which its
repair values are selected; `H'` and `T` alone are insufficient.  The bind
therefore reopens/carries the immutable target scorer-RGB array against the
target hash and target-forward receipt already bound by the baseline evidence.
Neither candidate nor target RGB is serialized in a selective-G or receipt
payload.  This keeps the request sufficient for the later G12 extension
without inventing target colours from semantic classes.

The current frozen G12 acquisition evidence cannot be constructed honestly for
this base: its closed base enum names only `EXACT_SEMANTIC_G_CONTROL_V1` and
`PASS_PREDICTOR_SEMANTIC_TOPOLOGY_V1`, while selective TACG1C is neither.  This
module must therefore expose a typed `SelectiveTopologyFreshG8RequestV1` (or
equivalent) containing the actual post-forward arrays, hashes, fresh row-2
mask, and exact selective G foreign key, plus a typed seam status
`G12_SELECTIVE_TACG1C_BASE_INTERPRETATION_OWED`.  It must **not** manufacture a
G12 evidence object by renaming selective G as the exact-semantic diagnostic.
The request is sufficient input for a future narrow G12 base-enum extension or
for direct acquisition followed by the already-existing generic G8 compiler.
It may not copy a baseline G8 program or debt mask.

Existing receiver domains remain exact:

- no G8: plain `TACG1C` plus the existing G-conditioned `TACA3P1` path;
- with G8: `TACG8S1` containing the exact selective `TACG1C`, followed by the
  existing post-G8 `TACA8P1` path.

`TACG8S1 + TACA8P1` remains the generic composite-G8 diagnostic/control
identity already implemented.  This landing does not rename it as production
evidence.  The PASS production domain `TACPG81 + TACAPG1` is separate and is
not accepted as a selective semantic-G result.

This is also an explicit production blocker, not a documentation footnote.
Every acquisition/cascade receipt must carry typed status
`PRODUCTION_SELECTIVE_TOPOLOGY_ENVELOPE_OWED`: selective TACG1C currently lives
only in the generic/diagnostic G/A packet domain, so it is ineligible for a
candidate even after encoder and advisory cascade proof.  A later versioned
production selective-topology envelope, or an exact reviewed reclassification
guard that proves the existing domain lawful for production, must close this
status before candidate use.  The blocker must be queryable and must not be
silently relaxed by a successful G7 structural transform.

## 7. G7 whole-object hooks

`make_selective_topology_g7_proposals(...)` converts only exact-wire-unique
topology proposals into `TaskspaceWholeArchiveProposalV1` objects.  A caller
supplies a typed whole-bundle builder because G and A foreign keys must be
rebuilt together.  Merely replacing the G section while retaining a stale A
section is forbidden.

Each lazy transform must verify that the returned bundle:

- preserves exact counted P;
- contains either the proposal's exact plain TACG1C packet or a strict
  `TACG8S1` whose inner semantic packet equals it;
- uses the matching existing A packet domain (`TACA3P1` or `TACA8P1`);
- is a new exact `TaskspaceSectionBundleV1`; and
- does not introduce optional T through this hook.

The hook invokes no receiver, scorer, measurement callback, allocator, or
archive builder itself.  G7 later rebuilds the entire archive, double-decodes,
measures the same object, and admits only negative exact coupled score delta.

## 8. Synthetic verification

The focused tests use synthetic bounded arrays only.  They must cover:

1. exhaustive canonical four-way telemetry and per-class closure;
2. selection from row 3 only with row 4 and every unselected cell unchanged;
3. precedence-required death events in both paint directions;
4. all grammar families and all orders over multiple bounded prefixes;
5. exact TACG1C parse/apply determinism and no dense target serialization;
6. exact packet-byte dedupe across equivalent interpretations;
7. canonical receipt parse/re-emit and mutation rejection;
8. typed empty-debt, custody, cardinality, and unrelated-mutation blockers;
9. fortunate-clearance ordering without an admission cutoff;
10. fresh post-topology telemetry and refusal of a baseline forward receipt;
11. a fresh-G8 request deriving debt from `H'`, never baseline `H`, plus the
    typed current G12 selective-base seam blocker (and no diagnostic renaming);
12. G7 hooks accepting only exact plain/composite G plus matching A domains,
    while rejecting stale-A, PASS-domain, inner-G substitution, P mutation,
    and optional-T smuggling.

These tests establish structural and arithmetic behavior only.  They are not
n600 evidence and cannot choose a scientific branch.

## Observability surface

- Per layer: evidence binding, complete baseline four-way telemetry, resolved
  prefix/order/family, exact event plan, compiled packet identity, decoded
  mutation support, post-forward four-way telemetry, and lazy G7 bundle result.
- Per signal: counts close globally, by target class, and by source-to-target
  transition; birth/death/event and exact packet-byte costs remain separate.
- Run-to-run diff: canonical receipt bytes and exact TACG1C packet bytes are
  deterministic content identities; duplicate interpretations point to the
  same packet SHA-256 without deleting their receipts.
- Post-hoc query: acquisitions retain the complete typed proposal tuple plus
  the exact-wire-unique view and both pre/post telemetry hashes; dense arrays
  remain encoder-memory-only.
- Cite chain: P, predictor state, scorer, candidate/target forward receipts,
  target semantic/RGB custody, packet, and G7 proposal identities are explicit
  foreign keys.
- Counterfactual: each proposal is one exact prefix/order/family program, so G7
  can replace it atomically and measure the full same-object score delta.  No
  receipt field pretends such a measurement occurred here.

## 9. Pointer-delta honesty

This landing emits no real proposal, archive, decoded video, scorer row, or
exact evaluation.  The canonical pointer is unchanged.  The next evidence
operation is the already-active, cheapest reviewed and resumable **n2
mechanism/cascade smoke** through the actual receiver and actual post-topology
forward: recompute `H'`, derive fresh G8/A, and price the same whole object.  It
is not a scientific verdict.  Only surviving whole-object rows escalate to n24
integration, then n600 decision-quality/authority work.  Those measurements and
the versioned production-domain closure are outside this encoder-only landing.
