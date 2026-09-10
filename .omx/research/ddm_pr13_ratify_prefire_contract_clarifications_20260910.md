# ddm_pr13 — second-family ratification of the pre-fire contract clarifications

Date: 2026-09-10  
Axis: `[review; scorer-free; source and retained-byte inspection]`  
`research_only=true; score_claim=false; promotion_eligible=false`  
Tokens: `[no-triality] [p0-ledger-ok]`

## Verdicts

| Item | Verdict | Ruling |
|---|---|---|
| 1. File SHA/bytes plus canonical digest | **RATIFY** | The split is a strengthening reconciliation of pr12's two simultaneous identities, not a loosening. The fire path pins the exact file it reads: resolved path, raw-file byte count, raw-file SHA-256, canonical self-omitting digest, and a Git commit containing those exact file bytes must all agree. Two differently serialized intent files cannot substitute for one another even when they have the same canonical digest. |
| 2. New-reference scope and unchanged legacy leg | **RATIFY** | Every file reference introduced by the new lifecycle is byte-counted and SHA-pinned. The completed legacy `t4_direct` leg is itself referenced as a byte-counted file; its interior `candidate_t4_receipt` remains SHA-only under the unchanged legacy validator. The complete inventory below contains no unpinned content reference. |
| 3. Arm produces, MAIN commits, MAIN authorizes | **RATIFY** | pr12 requires a committed intent before authorization, not a producer-authored Git object. `arm produces -> MAIN commits -> MAIN authorizes` satisfies the ordering if MAIN commits the exact intent bytes first. The authorization must record a full commit SHA containing that exact intent path and bytes; the landed producer records current `HEAD`, after validating that `HEAD` contains the intent byte-identically. |
| 4. Real positive control | **RATIFY — NOT YET OPEN** | The contract's evidence requirement is correct, but rlc4 has not yet emitted a real intent. Fixtures close no pass path. The producer door becomes open only on the real evidence bundle listed below; end-to-end completion remains unproven until MAIN's one-shot real fire and harvest produce either a valid v3 seal or a retained real refusal. |

No AMEND or REFUSE verdict is issued, so there is no patch text for MAIN to land.

## 1. Dual intent identity

The controlling pr12 sentences are:

> “Paths are absolute custody paths. Every referenced JSON or manifest carries `{path, bytes, sha256}`; the validator re-reads it, requires exact byte count and SHA-256, parses it, and checks the semantic fields named below. A typed boolean never substitutes for re-derivation from the referenced bytes.” (pr12 lines 84–86)

> `"intent": {"path": "<candidate_prefire_intent.v1>", "bytes": "<positive int>", "sha256": "<intent digest>", "commit": "<40-hex commit containing that exact intent>"}` (pr12 line 308)

These clauses name two different inputs to SHA-256. The retained JSON file includes `intent_sha256` and formatting; the canonical digest omits that self field and canonicalizes the remaining object. Binding both preserves both pr12 requirements.

The emitter and fire side use the same content-only algorithm, `prefire_digest`, at
`src/tac/candidate_seal.py:1662-1666`. `build_first_measurement_authorization` reads and validates the
committed intent, then records `path`, `file_sha256`, `file_bytes`, `digest`, and `commit` at
`src/tac/candidate_seal.py:2291-2315`. The authorization validator recomputes the file reference from
the exact `intent_path` supplied by the consumer, requires resolved-path equality and all three identity
values, and compares both current-HEAD and recorded-commit blobs byte-for-byte at
`src/tac/candidate_seal.py:2318-2360`. The first-measurement fire calls that validator on the exact CLI
intent path before any reservation and again immediately before reservation at
`tools/fire_modal_auth_eval.py:646-652,688-691`; its dispatch context records all three identity values at
`tools/fire_modal_auth_eval.py:668-676`.

Therefore the proposed construction with one committed file and a differently formatted fired file
fails at least path equality and raw-file SHA equality. Even if two files have identical canonical
content, changing whitespace changes the raw-file binding; using a second byte-identical copy at a
different path still fails path equality. The fire path pins the file it read, not only its canonical
digest.

Executed controls: `test_canonical_digest_is_not_file_digest`,
`test_authorization_binds_all_three_intent_values` (independent `file_sha256`, `file_bytes`, and
`digest` mutations), `test_dry_run_consumes_nothing_and_explicit_argv`, and the Git loose/packed/live
blob controls in `test_git_custody_read.py`.

## 2. Complete reference-pin inventory

“Pinned by bytes” means the new object or a mandatory joined receipt records and rechecks exact byte
count plus SHA-256. “Pinned by SHA only” means content is still cryptographically identified but that
specific field has no byte-count member; these rows are inline semantic identities or the expressly
unchanged legacy interior, not new JSON/manifest file references. There are no unpinned content rows.

| Object and reference | Pin class | Consumer proof |
|---|---|---|
| Intent `contract.adjudication_memo` | pinned by bytes | `_pf_ref(..., parse=False)` plus fixed pr12 SHA and committed-blob equality (`candidate_seal.py:1784-1788`). |
| Intent `contract.implementation_manifest` | pinned by bytes | `_pf_ref`; manifest file byte/hash and declared manifest SHA agree (`:1761-1777`). |
| Implementation-manifest source rows | pinned by SHA only | pr12 defines these rows as source path plus committed SHA; live SHA and `git show <commit>:<path>` byte equality are both checked (`:1776-1783`). |
| Intent `candidate.archive` | pinned by bytes | `_pf_ref(..., parse=False)`, ZIP validation, and member re-read (`:1835-1872`). |
| Intent `candidate.runtime` directory | pinned by bytes through closure | Tree SHA, file count, total bytes, and the exact per-file `relative_path/bytes/sha256` candidate manifest are re-derived (`:1840-1848,1923-1935`). |
| Intent `candidate.normalized_receiver` | pinned by SHA only | Content-only receiver digest is recomputed from the same runtime (`:1849-1851`). |
| Intent `candidate.receiver_pins[*]` | pinned by bytes | Every receiver pin has exact relative path, bytes, and SHA and is joined to the runtime map (`:1852-1860`). |
| Intent `candidate.archive_member` when non-null | pinned by bytes | Exact member name, bytes, and SHA are re-read from the ZIP (`:1866-1870`). |
| Embedded public-smoke candidate/frontier runtime and archive identities | pinned by SHA only inside the embedded block | The existing smoke validator re-derives candidate endpoints and binds the frontier archive SHA to the live pointer (`:1890-1896,1912-1914`; `_public_smoke_problems`). Candidate byte counts also come from the top-level candidate references. |
| Intent evidence refs: candidate manifest, validation, twin encode, parse-back, raw identity, literal census, retention manifest, timing risk | pinned by bytes | All are built by `prefire_file_reference`; `_pf_ref` requires positive bytes, SHA, and exact live equality (`:1669-1708,1908-1911,2040-2044,2190-2201`). |
| Candidate-manifest and literal-census file rows | pinned by bytes | Exact runtime `relative_path/bytes/sha256` maps must equal the live tree (`:1923-1935,1986-1993`). |
| Manifest-validation `manifest` reference | pinned by bytes | It must equal the intent's byte-counted candidate-manifest reference (`:1932-1935`). |
| Twin `payloads[*]`, `executions[*]`, and each execution's `payload` | pinned by bytes | Every file is `_pf_ref`-checked; both executions and their payload identities are joined and compared (`:1940-1961`). |
| Parse-back `member` | pinned by bytes | Member bytes and SHA must equal both retained twin payloads (`:1962-1966`). |
| Raw-identity `candidate_public_stdout`, `candidate_raw`, `pointer_raw` | pinned by bytes | All file refs are `_pf_ref`-checked; both raw refs require the exact 3,662,409,600-byte count and equal SHA (`:1967-1985`). |
| Retention-manifest `payloads[*]` | pinned by bytes | Each retained file is byte/hash checked and required paths are joined by exact absolute path (`:1994-2004`). |
| Intent `retained_payload_paths[*]` | pinned by bytes through closure | Every bare path must appear in the retention manifest's byte-counted payload references (`:2002-2004`). |
| Authorization `intent` | pinned by bytes | Resolved path, `file_bytes`, `file_sha256`, canonical `digest`, and containing `commit` are checked (`:2302-2306,2331-2356`). |
| Authorization `cost_preflight` | pinned by bytes | Exact cost JSON file reference is re-read (`:2300-2301,2357-2360`). |
| Cost receipt `provider_price_source` | pinned by bytes | `_pf_ref` checks its path, bytes, and SHA before price-field re-derivation (`:2242-2277`). |
| Authorization file supplied to fire | pinned by exact Git bytes | It is not a self-reference field: the validator requires the exact CLI file to be byte-identical to its current-HEAD blob and requires its canonical authorization digest (`:2323-2330,2349-2355`). V3 later adds an explicit byte-counted reference. |
| Risk `source_t4_leg` | pinned by bytes | The new risk object references the legacy-leg file with `_pf_ref` and then invokes unchanged `validate_decode_wall_clock` (`:2040-2052`). |
| Risk `source_receiver` and `candidate_receiver` | pinned by SHA only | Both receiver digests are joined to the validated source leg and live candidate receiver (`:2053-2058`). |
| Risk `diagnostic_reference_receiver` directory | pinned by SHA only | Its normalized receiver digest is recomputed from the named retained runtime and must equal the live candidate receiver (`:2055-2058`). The delta manifest supplies the per-file byte counts. |
| Risk `receiver_delta_manifest` and its endpoint rows | pinned by bytes | The manifest file has bytes/SHA; every normalized path row carries source/candidate `[bytes, sha256]` or null and must equal recomputation (`:2059-2065`). |
| Risk `base_local_diagnostic` and `candidate_local_diagnostics[*]` | pinned by bytes | Each reference is `_pf_ref`-checked before semantic and arithmetic re-derivation (`:2066-2094`). |
| Legacy source leg's `candidate_t4_receipt` | pinned by SHA only — allowed legacy interior | The outer legacy-leg file is byte-counted by the new risk object. Inside it, unchanged `tac.decode_wall_clock._receipt` re-hashes the receipt at its `{path, sha256}` reference (`decode_wall_clock.py:65-77,326-343,389-408`). |
| Legacy source leg's runtime/archive paths and receiver/archive/T4-runtime digests | pinned by SHA only — allowed legacy interior | Unchanged direct validation recomputes each digest and checks the named paths (`decode_wall_clock.py:300-323`). |
| V3 `prefire_intent` | pinned by bytes | Completion emits `prefire_file_reference`; normal v3 validation `_pf_ref`-checks it and revalidates the intent (`candidate_seal.py:2558-2561,2571-2581`). |
| V3 `first_measurement_authorization` | pinned by bytes | Emitted as a byte-counted reference and revalidated through the authorization consumer (`:2559-2561,2592-2594`). |
| V3 `first_measurement_receipt` | pinned by bytes | Emitted as a byte-counted reference, checked by `_pf_ref`, and joined to consumption/call/result lineage (`:2559-2561,2574-2575,2594-2595`). |
| V3 `decode_wall_clock_reference` | pinned by bytes | The separately retained legacy leg file is byte-counted, parsed, and required equal to both the inline leg and a fresh unchanged direct build (`:2556-2559,2594-2598`). |
| V3 `prefire_intent_sha256` | pinned by SHA only | This is the canonical semantic identity secondary to the byte-counted intent file reference; it must equal the revalidated intent digest (`:2576-2581`). |
| V3 inline `decode_wall_clock.candidate_t4_receipt` | pinned by SHA only — allowed legacy interior | It remains the unchanged legacy reference; the byte-counted `decode_wall_clock_reference` and `first_measurement_receipt` close the new-object file custody around it. |

Authorization `output_dir` and `receipt_path` are deterministic future destinations, not references to
pre-existing evidence. `_pf_output` restricts the canonical destination to durable SSD custody and the
receipt path must be exactly `<output_dir>/MODAL_REMOTE_RESULT.json` (`candidate_seal.py:2281-2288,
2345-2346`). They therefore do not constitute unpinned content references.

The controlling pr12 sentences are the general reference rule at lines 84–86, completion step 2
requiring the existing `build_t4_direct_leg` at lines 456–460, and: “Legacy
`candidate_seal.v1/v2`, measured/inherited/direct legs, and normal `--seal` semantics remain unchanged”
(lines 471–473). Clarification 2 is the only reading that satisfies all three without rewriting the
pinned historical leg.

Executed controls: `test_evidence_byte_drift`,
`test_risk_refuses_forged_arithmetic_but_keeps_legacy_leg`,
`test_completed_fixture_seal_uses_unchanged_direct_builder`, the existing candidate archive/runtime
drift controls in `test_candidate_seal.py`, and the Git-custody tests.

## 3. Producer/commit/authorization ordering

The controlling pr12 sentences are:

> “MAIN creates this only after the committed intent validates.” (line 300)

> “The immediate producer check performs every semantic and live-byte validation except the necessarily later ‘exact intent blob is committed’ check; the pre-dispatch consumer requires that Git custody.” (lines 379–383)

> “Resume means materialize the real move-42 rebase, retain both full encodes, prove full-n600 public raw identity, regenerate and independently validate the manifest, repeat the PR9 literal census, capture candidate/frontier smoke, build the exact risk receipt, and produce a committed intent. It does not mean RLC2 may dispatch. MAIN alone may create the separate authorization and fire after revalidating that real intent.” (lines 505–509)

These sentences separate roles and states. They do not say the candidate producer must be the actor
that writes the Git object. The required state transition is that exact intent bytes exist in a commit
before MAIN derives authorization.

`build_prefire_intent` deliberately self-validates with `require_committed=False` and deletes a rejected
new object (`candidate_seal.py:2177-2225`). `build_first_measurement_authorization` then calls normal
`validate_prefire_intent` with committed custody required, resolves current `HEAD`, verifies that commit
is on `main`, and records it as `intent.commit` (`:2291-2306`). The authorization validator requires the
intent bytes in both current `HEAD` and the recorded commit, requires the implementation commit to be an
ancestor, and requires the authorization itself to be present byte-identically in current `HEAD` before
fire (`:2347-2356`). The dedicated CLI accepts no actor or commit override and tells MAIN to commit the
exact authorization before dispatch (`tools/authorize_candidate_first_measurement.py:19-33`).

Required custody field: `authorization.intent.commit = <full 40-hex commit containing the exact intent
path and byte-identical file>`. The current tool records current `HEAD` after committed intent validation;
that is a containing commit, even if MAIN rather than the arm authored it. The commit need not be the
first commit that introduced the file, and pr12 does not require such an introduction-commit field.

The chronology condition held so far. The implementation landed at
`a475431997d0e0c66563448524e44c2ca8ddb384`, and the freeze receipt landed at
`637d46c02223bf6eede9dc77b3dc9de8f4de8b6f`, before rlc4's real production launch; the
clarifications therefore existed before any rlc4 intent. At this
review snapshot rlc4 had emitted no intent and MAIN had emitted no authorization. This memo or its
serializer bundle must be landed before authorization; a serializer Git-object denial does not waive
that order.

Executed controls: `test_intent_producer_emits_and_self_validates`,
`test_authorization_binds_all_three_intent_values`, `test_nonce_single_use_and_forward_transitions`,
and `test_git_custody_read.py`'s loose, packed-delta, ancestry, blob, and timestamp controls.

## 4. Real positive-control status

The controlling pr12 sentences are:

> “Negative and synthetic fixture tests are useful but cannot close the pass path.” (lines 525–527)

> “RLC2's real candidate producer emits and self-validates `candidate_prefire_intent.v1`; the normal `validate_seal` and normal `--seal` path both refuse it as not a seal.” (lines 529–530)

> “If RLC2 does not reach the passing completion branch, `candidate_prefire_intent.v1` remains implemented but not positively proven end to end.” (lines 540–542)

At the read-only snapshot used for this review, `.omx/research/ddm_rlc4_20260910/` contained recall
artifacts only. The SSD work root held retained, resumable trace checkpoints and an active trace log, but
no real intent, normal-validator refusal receipt, authorization, or completion. Rlc4 therefore has not
opened even the real producer door yet. The 62 passing focused tests remain fixtures by their own
module declaration.

The rlc4 output required to declare the real producer door open is all of the following:

1. The real emitter's retained argv, stdout/stderr, and zero return code showing that
   `tools/make_candidate_seal.py --first-fire-intent ...` wrote and self-validated the named real
   `candidate_prefire_intent.v1`; the intent and every referenced receipt must be rlc4's retained real
   move-42 candidate evidence, not the synthetic fixture.
2. A read-only recomputation of the exact intent file's absolute path, byte count, raw-file SHA-256,
   and canonical `intent_sha256`, with the latter equal to `prefire_digest(intent,
   "intent_sha256")`. Rlc4's memo and serializer bundle must record all three identity values.
3. A retained normal `validate_seal` verdict with exact code `PREFIRE_INTENT_SCHEMA_REFUSED`, plus the
   normal `tools/fire_modal_auth_eval.py --seal ...` refusal before subprocess. Acceptance as a seal,
   another code caused by malformed evidence, an exception/traceback, or any subprocess reach is not
   the required positive control.
4. MAIN's subsequent commit of those exact intent bytes, followed by successful committed
   `validate_prefire_intent` and authorization emission whose `intent.path`, `file_bytes`,
   `file_sha256`, `digest`, and `commit` all match. This is the committed-consumer half of the producer
   control; it remains after rlc4 because the arm cannot write Git objects in this sandbox.

The producer control is falsified if the emitter refuses or deletes the object; any non-timing gate is
missing; the intent's canonical digest or raw file identity differs on re-read; the normal validator
accepts it as a seal, crashes, or reports a different cause; the normal fire reaches a subprocess; MAIN
commits different bytes/path; authorization succeeds before committed custody; or authorization can be
satisfied by another serialization. Such a result is a real retained contract finding, not permission
to patch the live contract in this review.

Even a passing producer control does not establish the end-to-end door. That requires the same real
authorization to reserve exactly once, a real retained call and harvest, unchanged `t4_direct`
validation at no more than 1,260 seconds, every frozen non-timing/pointer/bar recheck, and a normal-consumer-
valid `candidate_seal.v3`; otherwise the retained real refusal is the result and positive end-to-end
proof remains open.

Executed fixture controls defining, but not satisfying, this evidence are
`test_fixture_intent_validates_but_is_not_a_seal`, `test_intent_producer_emits_and_self_validates`,
`test_first_fire_cli_override_refuses_before_subprocess`,
`test_missing_authorization_refuses_before_subprocess`, and
`test_completed_fixture_seal_uses_unchanged_direct_builder`.

## Validation

- `.venv/bin/python -m pytest src/tac/tests/test_candidate_seal.py -q` — **53 passed**.
- `.venv/bin/python -m pytest src/tac/tests/test_candidate_prefire_intent.py -q` — **62 passed**.
- `.venv/bin/python -m pytest src/tac/tests/test_git_custody_read.py -q` — **5 passed**.
- The reviewed contract files are unchanged from implementation commit `a47543199`; it is an ancestor
  of current `HEAD`. Pr12 SHA-256 is
  `50d00e3956dc7ae5d3b15379d2ae6f8704119817b0b58f50aa30413b97eacadc`; frozen receipt SHA-256 is
  `59158b8fce89e12c06eb4ae3e1cb71f347daa78a2cc5e8a061e1899be1150c2a`; ffi3 charter SHA-256 is
  `dd0de8cd84b59c795750ac99ebe9c5e1f5a543951537da4c66fb72fa6951814f`; rlc4 charter SHA-256 is
  `096f0db8f4e91be09eb4b8e742233a2ce6c2da1d61257a7b2bea9057ea6698b6`.

## RECALL EVIDENCE

Searched the complete `.omx/research/` Markdown/JSON corpus by content for
`candidate_prefire|first[-_ ]measurement|intent_file_sha256|intent digest|digest collision|same
canonical|different whitespace|t4_direct|gate with no door`; inspected the pr8–pr12/ffi/rlc lineage,
the r9m digest precedent, pr9's stale-manifest finding, the frozen receipt, and current rlc4 custody.
Searched `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, docs/design/SPEC surfaces,
`canonical_task_status.jsonl`, the lane registry, and the operator ledger for the same lifecycle terms.
Generated the canonical equation registry with
`.venv/bin/python tools/list_canonical_equations.py --json` and filtered it for
`prefire|first.measurement|intent|authorization|t4_direct|digest`.

Beyond the charter seeds, r9m's environment-free digest precedent says both validators must compute the
same content identity, which reinforced tracing `prefire_digest` on both emitter and fire sides. Pr9's
stale-manifest finding reinforced requiring the exact per-file byte map rather than treating a tree SHA
or success boolean as complete custody. The live ledger confirms pr13 remains before authorization.
No canonical equation, older DAG row, design document, or task-ledger row supplied a competing
first-measurement lifecycle in the searched scope, so recall changed no verdict.

## Boundaries

- This is adversarial, read-only review of code and live/sealed trees. Only this memo and serializer/
  checkpoint custody are produced. No `src/tac/`, `tools/`, `upstream/`, PR tree, sealed tree, rlc4/sj1
  live directory, pointer, lane/spend ledger, or shared staged index was edited.
- No Modal/provider call, fire, authorization, timing window, encoder, decoder, scorer, evaluator,
  candidate materialization, completion, or payload cleanup ran. The only executions were scorer-free
  unit tests and read-only inspection.
- No score was measured and the exact pointer did not move. This ratification is apparatus work, not
  goal progress.
- The common contract's historical frontier paragraph is superseded by the charter and live hot state.
  MPS was not used; no GT decode occurred.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `ddm_rlc4`; consumer store: the retained SSD candidate/evidence tree, `.omx/research/ddm_rlc4_20260910/`, the real `candidate_prefire_intent.v1`, and its serializer bundle; fire trigger: the retained move-42 trace, twin encodes, full raw identity, external manifest validation, census, smokes, and timing-risk receipt all complete and the real intent emitter exits zero. Finish the real producer evidence and both normal refusal controls; do not authorize or dispatch.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: this ratification memo/bundle plus rlc4's exact committed intent and `candidate_first_measurement_authorization.v1`; fire trigger: MAIN lands this memo and rlc4's serializer bundle, rechecks the intent file byte count/SHA/canonical digest and normal refusals, and confirms all frozen gates still pass. Commit the exact intent first, then run the authorization producer and commit the exact authorization.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: the authorization's durable SSD `output_dir`, one-way nonce-consumption record, call-id ledger row, retained worker tree, and `MODAL_REMOTE_RESULT.json`; fire trigger: committed authorization validates, current provider cost is strictly below USD 5, the T4 lane is actively claimed by MAIN, global single-flight/cloud state is clear, and the pointer still matches. Fire once, retain everything, and harvest the same call; never retry an ambiguous reservation.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: the real `candidate_seal.v3`, exact-result adjudication, evaluation ledger, and canonical pointer packet; fire trigger: the harvested receipt passes unchanged cold n600 `t4_direct` at no more than 1,260 seconds and every identity, non-timing, pointer, and strict score-bar recheck. Complete and normal-validate v3, then move the pointer only if that exact row qualifies.

Frontier unchanged: `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)`.

## LIVE-HYPOTHESES

- Rlc4's real intent may pass the producer and committed-consumer controls because the landed validator
  already exercises the same file/canonical identities on synthetic fixtures, and rlc4 is building the
  required receipts through the real move-42 path. The real evidence joins remain untested.
- The cured receiver may complete cold T4 below 1,260 seconds because pr12's retained source-T4 plus
  receiver-delta calculation gives a 1,032.725-second spend-risk ceiling. That projection is diagnostic,
  not timing authority, and host transfer remains untested.
- The dual binding should remain stable across producer and fire environments because both sides call
  the same canonical `prefire_digest` and separately re-read exact file bytes. The real rlc4-to-MAIN
  handoff has not exercised this cross-actor path.

## DEAD-ENDS

- One digest standing for both serialized-file identity and canonical object identity is closed: the
  two digests have different inputs, and keeping only either one drops a pr12 binding.
- Substituting a differently formatted or differently located intent with the same canonical content is
  closed: path, byte count, raw SHA, canonical digest, and committed blob all have to agree.
- Recursively rewriting the historical `t4_direct` leg is closed: pr12 requires the unchanged builder
  and validator, while the new risk/v3 objects byte-pin the legacy leg file around its SHA-only interior.
- Requiring the arm rather than MAIN to author the intent Git object is closed: pr12 requires committed
  state before MAIN authorization, and the consumer proves byte-identical custody independently of actor.
- Treating fixture passes, the running trace, or an emitted intent alone as end-to-end positive proof is
  closed: the real one-shot fire, harvest, unchanged direct leg, and normal-consumer-valid v3 remain
  mandatory.
