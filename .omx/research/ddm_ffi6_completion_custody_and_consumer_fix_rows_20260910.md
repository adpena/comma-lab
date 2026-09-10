# ddm_ffi6 — completion custody and typed consumer fixes

Implementation and host validation are complete; serializer disposition is recorded separately in
`ddm_ffi6_20260910/SERIALIZER_RESULT.json` after the single final attempt. No exact pointer move,
real completion, seal write, authorization, or fire was performed. This is a consumer repair,
not score progress.

## Requirement → code → test

| Requirement | Code | Test / evidence |
|---|---|---|
| Independently recompute and join content across context, request, exact argv, result, worker provenance | `candidate_seal.validate_first_measurement_runtime_custody`; called by `_pf_completion_facts`, therefore both completion and subsequent v3 validation | `test_completion_runtime_custody_refuses_each_join_drift` (15 variants); `RUN4_DRY_VALIDATION.json` |
| Project local dependencies onto the recorded retained root; validate actual worker tree, rows, count and file digest | Same helper uses the existing `modal_uploaded_submission_dir_runtime_manifest` with the provenance root; also verifies `inflate_script` belongs to that root | `test_run3_retained_provenance_projection_and_terminal_refusal`; root/tree/row/file-digest/count negatives; synthetic relocation positive |
| Identical custody in v3 and its direct leg; reread pinned sources | `complete_first_fire_intent`, `_pf_validate_completed_seal`; source references pin the containing file plus exact field paths, including JSON-encoded provenance inside the result | `test_completed_fixture_seal_uses_unchanged_direct_builder`; `test_completion_seal_rechecks_custody_and_refuses_unequal_objects` |
| Legacy direct-leg requirements unchanged | No edits to `decode_wall_clock.py`; completion adds one custody field after the existing builder validates | All existing `test_decode_wall_clock_t4_direct.py` and `test_decode_wall_clock.py` cases; fixture compares the remaining leg exactly with the unchanged builder |
| Typed prospective consumer batches, complete previous-row chain, immutable pr14 definition parent | `_pf_freeze_history`, `_pf_contract`, `build_prefire_intent`; exact schema allowlist forbids definition/receiver/threshold/evidence overrides; new intents carry both parent and latest-row digests | `test_consumer_fix_batch_keeps_definition_and_refuses_stale_latest`; `test_consumer_fix_typed_refusals_with_real_git` (18 variants) |
| Full manifest identity, causal/implementation ancestry and stale-row refusal | Every historical manifest is checked against its owning committed blobs; latest manifest also matches live files; base and sequential implementation ancestry plus causal ancestry are checked | Real isolated Git-object controls for manifest drift, live drift, orphan/unreachable commits, wrong parent/latest pins; existing amendment tests remain green |
| Add typed migration semantics without rewriting historical rows | `CONSUMER_FIX_ROWS_DRAFT.json` contains one allowed batch: five migrated consumer fixes from amendments[1..3], plus completion custody and typed-history enforcement | `INPUT_PINS.json` preserves all four existing canonical row digests; frozen receipt untouched; batch validator supports multiple fixes and multiple chained batches |

All code above is in `src/tac/candidate_seal.py`; regression tests are in
`src/tac/tests/test_candidate_prefire_intent.py`. The CLI already delegates to these functions,
so no parallel CLI implementation or new completion mode was added. Runtime helper invariants and
negative controls provide the self-protection within pr16's authorized consumer scope.

## Measured host verification

Axis: **host custody / scorer-free unit tests**, not contest or macOS scorer authority.

- Required four suites: **231 passed, 0 failed, 0 skipped**, 5.49 s on the final run.
- Ruff on both changed Python files: **passed**.
- Two visible review passes per Python file; tracker rescanned 89 production entities and 56 test
  entities before marking each pass. `REVIEW_PASSES.json` pins the reviewed final Python bytes.
- `VALIDATION.json` names the exact suite command. The baseline after initial implementation was
  195 tests; 36 additional cases cover the new refusal/positive paths and retained control.

These tests exercise real local file reads and real isolated Git objects. Broad pre-existing fixtures
still substitute the large raw-file transport and some Git/authorization plumbing, as their module
docstring states. The retained run3 test is additional evidence, not a substitute for that disclosure.
No scorer, decoder, timing sampler or provider was run by this arm.

## Run3 retained control

`RUN3_DRY_VALIDATION.json` pins the real provenance and result. Provenance SHA:
`4cd0463214a90b8919489eabe77062df8720c5d8405faf2e3d5ef10ddbd74405`.
All **48** runtime file rows agree. Independently derived content:
`e1e6d1252b56b66b7f7559fe0df751ffbf38559d99abf21f06f00134f685bcaf`;
files digest: `fb1f6295a3527fdc07b3682d7e8ed42354e14e54dbbb4317deaa9f4f27ba21d4`.

The real evaluator helper passes content-only validation and reproduces the original tree refusal:
normal prediction `e3d2371917920ce3dad002761203af48cfa72d98b07f82f077ef6dc0115b1891`
versus retained worker `baeb53afc8d1bb0c43bb2ce91fe1e09e845809b76fc3047bc8ed3397254155af`.
Projecting onto the recorded root reproduces the latter exactly. Requiring that *correct* projection
to refuse would contradict pr16. The test therefore proves the original normal-tree refusal, accepts
the correct retained projection, and refuses a wrong tree at that retained root while content agrees.
It exercises production custody validation with the exact retained provenance in an explicitly
synthetic local join envelope; that envelope is not represented as a real harvest.

The unmodified real run3 completion-facts path refuses with `FIRST_MEASUREMENT_RESULT_REFUSED`
(`authorization not harvested`). Its reserved nonce and terminal worker failure remain unchanged.
MAIN's immutable reconciliation receipt is read and pinned; it supplies no completion authority.
The explicit failed-result check also prevents a failed receipt from proceeding after a harvested
state. Run3 is non-replayable and not a score.

## Run4 dry control and the remaining contract boundary

`RUN4_DRY_VALIDATION.json` records **PASS** for the five-way content join, all 48 file rows,
files digest, retained-root projection, and the read-only `_pf_completion_facts` checks.
The result's embedded provenance is covered by the result file's SHA pin and its exact field path;
standalone `provenance.json` was absent and is not required by the actual retained artifact shape.
The retained root projects to
`8d31edd6f7c3291b37b87eb8404a5944713a80a1d0e4a66fb66bab7650599dbc`.
Receipt SHA: `6a7267f4efe7ab54edc6e556030dbfa174d6d027e3cb6b008ad159cce51c0136`.
Its existing T4 inflate time, **1124.5857065260002 s**, passes the unchanged 1260 s policy in the
read-only facts consumer. This is a read of MAIN's retained measurement, not a new timing sample.

No `complete_first_fire_intent` call was made on real data, no seal/direct-leg file was written,
and no full current-contract completion was claimed. **Appending a typed freeze row makes run4's
v4 intent stale.** pr16 explicitly preserves latest-row refusal and says a post-amendment scored
fire needs a newly emitted intent and fresh authorization. The charter also asks to accept run4
when harvested. The code meets the run4 custody/facts acceptance and retains the refusal rule;
it does not invent grandfathering. MAIN/pr17 must resolve the run4 full-completion disposition
within the existing lifecycle. This arm does not authorize a refire or nonce reuse.

## Manifest and draft handoff

`PREFIRE_IMPLEMENTATION_MANIFEST.json` hashes **17** post-edit live files: all 16 files in ffi5's
latest manifest and the expanded test module. Its content is independent of a future commit hash.
`CONSUMER_FIX_ROWS_DRAFT.json` is explicitly a draft envelope, with an unbound implementation commit.
MAIN must bind the successful serializer/landing commit only after all 17 rows match its committed
blobs, then append the contained typed row. A null draft commit is not a valid frozen row.
The existing four amendment objects are never rewritten. Historical implementation snapshots with
the same pr14 definition retain row[0] as the definition parent; the new row chains row[3] as its
immediate predecessor. This is the charter's explicit pr14-parent interpretation.

The serializer is invoked last, once, with post-edit hashes, no co-author trailer, and the required
`[no-triality] [p0-ledger-ok]` tags. An explicit arm-local fallback directory prevents the serializer
from writing to forbidden volume paths; only this small source/evidence commit is eligible. A Git
object denial or reserve refusal is reported as such; files remain available to MAIN. No shared
staged index is modified directly.

## RECALL EVIDENCE

Searched the external memory registry for `ffi6|completion.custody|consumer.fix`: no relevant hit;
no external memory-derived fact is used. In the repository, searched research memos and JSON receipts
by `content.only|runtime_content_tree_sha256|consumer.fix`, then the canonical equation JSON output,
research index, `sub015_DAG_*`, docs/design surfaces, and canonical task rows by
`first.measurement|prefire|runtime.content`. Also read recent operator directive files and the live
board/lane registry. Search results are bounded; no global absence claim is made.

Beyond the charter seeds, the May 23 `codex_findings_modal_runtime_content_fail_closed` memo separates
content identity from path/package identity and requires failed/missing evaluator output to remain
failed. `docs/experiment_scheduler_design.md` separately names archive, content/tree and axis identity.
The canonical task ledger's rlc2 decision rows confirm that MAIN owns lifecycle adjudication and that
an arm's evidence must not become dispatch permission. These reinforced fresh content/projection
checks and explicit terminal failure refusal; no additional contract definition was introduced.
The DAG prefire references concern earlier reviews, and the equation search exposed no competing
completion-custody equation in the queried scope.

## Boundaries and frontier

No Modal API/CLI, dispatch, real authorization/completion, timing window, n600 run, scorer, payload
mutation, upstream edit, PR/sealed-tree edit, frozen receipt edit, or write under `/Volumes/`.
No gdc2/sr4 directory touched. Unrelated dirty files preserved. Only small code, test and evidence
files were created/edited; no real payload was produced or discarded. Shared tracker/checkpoint
metadata is maintained by its canonical tools. No canonical pointer or consumer store owned by MAIN
was edited.

Canonical pointer values read live: **S 0.1372848557085275 [contest-CUDA T4 n600]**,
archive `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`.
Archive size **180,466 B** is the live board's move-43 custody value (the pointer JSON itself has no
byte-size field). No exact score was measured or lowered by this arm; sub-0.12 remains unachieved.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer store `ddm_ffi6_20260910/CONSUMER_FIX_ROWS_DRAFT.json` and the canonical frozen receipt; trigger verified code/test/evidence landing:** bind the exact implementation commit and manifest, ratify the typed batch, and append without rewriting rows[0..3].
- **FOLDED into MAIN's existing run4 lifecycle; owner MAIN/pr17; consumer store `ddm_ffi6_20260910/NEXT_FIRE_ORDERS.json` and the existing rlc5 completion handoff; trigger typed batch ratification:** adjudicate run4's full-completion disposition under exact latest-row refusal, using the passing dry custody receipt; no automatic refire or nonce reuse.

## LIVE-HYPOTHESES

- MAIN may be able to close the candidate lifecycle from the retained run4 evidence after an explicit
  contract disposition: all runtime joins and read-only completion facts pass, but freshness is separate.

## DEAD-ENDS

- Normal upload-root hash equals retained-root hash: closed for these receipts by the reproduced root-dependent mismatch.
- Discarding the path-tree digest: closed; it is independently verified custody at the recorded root.
- Replaying or completing run3: closed at INSTANCE scope by its retained failure and non-harvested nonce state.
- Treating run4 custody PASS as a completed v3 seal: closed; no real seal was written and latest-row freshness still applies.

<!-- # FORMALIZATION_PENDING: consumer implementation and draft freeze batch; no scorer row or new equation -->
