# G45 findings — ep725/V2 label-local G streaming receiver seam

Date: 2026-07-26  
Lane: `lane_g45_ep725_label_local_g_stream_20260726`  
Mode: implementation mechanics; `research_only=true`; no n600 run, scorer, candidate, dispatch,
archive mutation, or pointer mutation

## Outcome

The previously named ep725/V2 receiver gap is implemented as a path-backed, chronological,
pairwise streaming seam.  It executes the exact frozen runtime's actual frame-1
`phi.argmax`, constructs a true `TaskspacePredictorStateV2+NoTransportV2` n1 slice, parses and
admits one exact counted label-local G page, applies the unchanged V2 semantic donor, and
delegates camera realization to the unchanged predictor-preserving overlay donor.  It writes
only Y1; Y0 and all G-unowned Y1 camera values are re-opened and proven byte-identical.

This is a necessary receiver closure, not frontier progress.  The canonical pointer remains
unmoved; its current `effective_frontier.score` is the external official-leaderboard target
`0.172`, not a locally custodied archive score.  No G45 artifact has been scored and nothing in
this lane is promotion eligible.

## Landed surfaces

- `src/tac/witness_dsl/taskspace_ep725_label_local_g_stream.py`
  - SHA-256 `353a16a75b42a5a79fbcf5b12ed978931f4e5162865cb8f21908bb28a9f7b5d6`
  - public input contract `Ep725StreamSourceContractV1`
  - ordered counted-page reference `GPageRefV1`
  - entrypoint `execute_ep725_label_local_g_stream(...) -> Ep725StreamExecutionReceiptV1`
- `src/tac/witness_dsl/tests/test_taskspace_ep725_label_local_g_stream.py`
  - SHA-256 `5f1fe40b584ef7ca68fc29cf7fe5416d7dce6f8e2a8caeefaf82954ff5ee7a6c`
- `SPEC_g45_ep725_label_local_g_stream_20260726.md`
  - SHA-256 `b52ee73e9b51c2a60a5860d791f2ec8930456fab63cbcd424185af2f752f196d`

## Scale and durability properties

- The realization and counted G payload are O(one pair/one page) in Python.  Page references
  may cover all 600 pairs, but each payload is reopened, hash-verified, consumed, and released
  serially.
- The chronological raw is preallocated under the configured SSD waterfall.  Local storage is
  possible only through the explicit mechanics-test escape hatch and always remains
  research-only.
- Every realized pair has an immutable checkpoint binding the exact source archive/member,
  runtime, renderer, G page, V2 state, labels, base pair, delegated overlay receipt, output
  range, and preservation facts.
- Resume accepts only a gap-free checkpoint prefix and rehashes each committed raw range plus
  its overlay receipt before continuing.
- Final promotion fsyncs a content-addressed execution receipt before atomic raw rename.  If a
  process dies after rename but before the canonical receipt write, the next invocation
  verifies the final raw and exact inputs against that durable intent and completes the
  receipt promotion without rerendering or guessing.

## Composition with G17 and target authority

G43's `taskspace_g17_actuator_ir_v1.py` remains the bounded n1/n2 closed-IR and deterministic
double-replay anchor over the same V2 semantic and overlay donors.  It is not used as the
production loop because its contract intentionally retains a bounded chronology in memory and
consumes packed G17 member spans.  G45 supplies the missing path-backed n600 execution shape;
the selected-solution packer should map exact G17 operand spans into `GPageRefV1` without
changing receiver physics.

G46's batch-4 fresh-teacher materialization is **not an authoritative encoder target**.  The
post-landing audit found two label cells that differ from `gt_n600::lstars`; the authoritative
`upstream/evaluate.py` geometry defaults to batch 16, and an earlier batch-32/cache comparison
also drifted by three cells.  Keep the batch-4 output as mechanics evidence only.  Do not compile
counted G pages from it.  Root must first rebuild at the exact authoritative batch geometry and
prove byte-exact label parity against the scorer-owned target surface.  After that gate closes,
the validated target is the encoder-side upstream for one predictor-bound label-local page per
pair.  Target labels, scorer weights, and teacher evidence remain encoder-only and must not
enter those pages or the public runtime.

## Verification

Green focused plus donor-parity run:

```text
.venv/bin/python -m pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_ep725_label_local_g_stream.py \
  src/tac/witness_dsl/tests/test_taskspace_predictor_v2_consumer_seam.py \
  src/tac/witness_dsl/tests/test_predictor_preserving_taskspace_overlay.py \
  src/tac/witness_dsl/tests/test_taskspace_g17_actuator_ir_v1.py
26 passed in 27.06s
```

```text
.venv/bin/ruff check \
  src/tac/witness_dsl/taskspace_ep725_label_local_g_stream.py \
  src/tac/witness_dsl/tests/test_taskspace_ep725_label_local_g_stream.py
All checks passed!
```

The focused tests cover full camera/scorer geometry, real donor execution, exact frame-1 phi
capture cardinality, path-backed chronology, immutable prefix resume, committed-output drift,
page drift, transport-dependent lifetime refusal before semantic/camera mutation, post-rename
receipt recovery, and exact Y0/unowned-Y1 preservation.  They are mechanics fixtures, not n600
evidence and not a score finding.

## Exact remaining finish-line chain

1. Rebuild the fresh n600 teacher at the exact `upstream/evaluate.py` batch-16 geometry and prove
   byte-exact parity with `gt_n600::lstars`.  Quarantine the current batch-4 output from all
   compiler inputs.
2. Compile the authority-closed obligations into 600 exact V2-bound label-local G pages and pack those pages as
   counted G17 operands beside P.  Reject transport-dependent atoms rather than fabricating
   Pose6 or weakening `NoTransportV2`.
3. Execute the same exact pages through G45 on all 600 pairs under the governed SSD launcher,
   preserving all per-pair checkpoints and the final raw/receipt.
4. Measure the changed raw through the frozen n600 scorer and compare the joint score value of
   G against its exact added archive bytes.  This is the first useful actuator-economics row.
5. Only if the joint row is favorable, close the same selected P+G bytes through the public
   `inflate.py`/`inflate.sh` archive path and run authoritative contest-CPU/CUDA evaluation.
6. Move the canonical pointer only for a qualifying exact archive whose score is lower than the
   then-current effective frontier; continue toward the binding sub-0.15 goal.

The immediate blocker is no longer receiver mechanics.  It is exact target-geometry authority,
then fresh full-n600 G-page materialization and counted outer packing, followed by a governed
full execution and score.  A new abstraction layer before that row would delay the actual
frontier test.
