# ddm_ffi2 — STOP on reference-custody scope inside unchanged T4 legs

Date: 2026-09-10  
Status: BLOCKED_CONTRACT_AMBIGUITY; implementation NOT landed; contract NOT frozen.  
Axis: `[source inspection; retained-file byte hashing; scorer-free]`  
`research_only=true; score_claim=false; promotion_eligible=false`  
Tokens: `[no-triality] [p0-ledger-ok]`

The ffi2 clarification resolves ffi1's digest question. A different scope ambiguity remains:
does pr12's requirement that **every referenced JSON** carry a recorded byte count extend into
the immutable historical `t4_direct` leg it requires us to consume? That leg's nested
`candidate_t4_receipt` contains only `path` and `sha256`. The unchanged completion builder emits
the same two-field reference. I stopped before implementation, as the charter requires.
This is an INSTANCE-scoped specification blocker, not a failed candidate or a failed lifecycle.

## Exact sentences and primary evidence

The inherited ffi1 charter says: **“Where the text is ambiguous, STOP and report the exact
sentence”**. The ffi2 charter preserves that rule: **“if another appears, STOP the same way with
the exact sentence.”**

Pr12, lines 84–86:

> Paths are absolute custody paths. Every referenced JSON or manifest carries `{path, bytes, sha256}`;
> the validator re-reads it, requires exact byte count and SHA-256, parses it, and checks the semantic
> fields named below. A typed boolean never substitutes for re-derivation from the referenced bytes.

Pr12, lines 265–266:

> Validation requires the source leg to pass the unchanged `validate_decode_wall_clock` as completed
> `t4_direct`; its receiver to equal `source_receiver.sha256`; and the diagnostic reference receiver to

Pr12, lines 278–282, mandates a particular historical source leg, including its exact bytes and
SHA-256. The retained observation in
`ddm_ffi2_20260910/CONTRACT_SCOPE_OBSERVATION.json` independently verifies:

| MEASURED item, scorer-free byte inspection | Result |
|---|---|
| Required source leg | `/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/SEAL_ddm_sj1_compose39_rp1_union_contest_cuda.json.decode_wall_clock.json` |
| Source leg bytes / SHA-256 | 1,553 / `ed929b24cc876bf8ffabc3004b856decbb5e73d0fee13c1d3c87d91d659f9521` |
| Its nested `candidate_t4_receipt` keys | Exactly `path`, `sha256`; no `bytes` |
| Referenced remote result bytes / SHA-256 | 159,313 / `cd6d5ef5243e26fa1868100a2bbb34efe0d6446b4a7fb9bd1eab7f309f5d3d00`; SHA matches the leg |
| Existing `receipt_reference` return keys, inspected via AST | Exactly `path`, `sha256` (`src/tac/decode_wall_clock.py:68`) |
| Completion builder | `build_t4_direct_leg` calls that helper for `candidate_t4_receipt` (`src/tac/decode_wall_clock.py:326`) |

Pr12 completion step 2 also requires **“call the existing `build_t4_direct_leg` on that exact
receipt and unchanged candidate bytes”**, and its last lifecycle paragraph says
**“Legacy `candidate_seal.v1/v2`, measured/inherited/direct legs, and normal `--seal` semantics
remain unchanged.”**

The remote result exists and its hash matches; this is not missing payload custody. The missing
item is a normative scope decision about where a recorded `bytes` field must appear.

## Why this needs clarification rather than a guessed implementation

Two readings produce different validators:

- Applying “every referenced JSON” recursively rejects the mandatory historical source leg's
  nested receipt reference. Adding `bytes` to that retained leg changes its pinned 1,553-byte
  content and SHA-256. I did not do that.
- Applying it only to references introduced by the new lifecycle objects preserves the legacy
  leg and its existing SHA-bound receipt validation. That may be the intended effect of
  “unchanged,” but the contract does not explicitly state this scope exception.

Adversarial counter-reading: the specific unchanged-legacy clauses could be intended to limit
the general custody sentence implicitly. That is plausible, and is why this is labeled an
ambiguity rather than a proven inconsistency. Under the explicit STOP instruction I have not
chosen that reading. Measuring the nested result's length now also does not make that length a
recorded, frozen binding in the immutable historical leg.

Required decision: state whether `{path, bytes, sha256}` is required only for new lifecycle
reference blocks, with nested legacy references validated by unchanged legacy consumers, or
also recursively inside referenced legacy JSON. If recursive binding is required, specify the
allowed supplemental binding without modifying the pinned source leg or its existing validator.

## Digest clarification accepted; not reopened

MAIN's split binds all three authorization fields: `intent.file_sha256` pins the exact
committed file bytes, `intent.file_bytes` pins their count, and `intent.digest` pins the
canonical self-omitting digest. Both hashes must pass. This resolves the original ambiguity;
no alternative single-digest implementation is proposed. The clarification still awaits pr12's
family ratification as the charter states. Its full source and hash are retained in the
observation JSON. No frozen-contract receipt was created with an absent implementation commit.

## Per-clause compliance status

No row is an implementation PASS. This is the blocked inventory, not the completed compliance
table owed after implementation.

| Pr12 clause | Code location inspected / required surface | Test and implementation status |
|---|---|---|
| 74–100, canonical/reference hashing and implementation custody | `decode_wall_clock.receipt_reference`; future intent validator | BLOCKED on nested-reference scope |
| 102–201, exact intent schema and false-authority exclusions | `candidate_seal.validate_seal`; future intent producer/consumer | NOT IMPLEMENTED |
| 203–230, nine non-timing comparisons | `candidate_seal._public_smoke_problems`, pin/runtime helpers; future evidence consumer | NOT IMPLEMENTED |
| 232–296, receiver-delta risk with completed source leg | `decode_wall_clock.validate_decode_wall_clock`; pinned historical leg | BLOCKED on reference scope; existing validator unchanged |
| 298–341, independent authorization and single-use nonce | Dedicated authorization producer and fire consumer | NOT IMPLEMENTED; split-field meaning resolved |
| 343–400, exact producer flags and self-validation | `tools/make_candidate_seal.py`; dedicated authorization CLI | NOT IMPLEMENTED |
| 402–439, first measurement, explicit limits and quarantine | `tools/fire_modal_auth_eval.py`, CUDA worker, harvest poller | NOT IMPLEMENTED; existing timeout-default and mirror surfaces identified |
| 441–473, completion and new v3 seal | `build_t4_direct_leg`, completion producer, normal seal consumer | NOT IMPLEMENTED; unchanged helper emits two-field nested reference |
| 475–501, fifteen typed refusals | Future intent/authorization/dispatch/completion consumers | NOT IMPLEMENTED; no tests executed |
| 523–543, prospective freeze and mandatory real control | Retained frozen receipt path; RLC2 producer; MAIN harvest | NOT FROZEN; real control NOT EXERCISED |

Synthetic fixtures cannot close the producer pass path. RLC2's real producer, MAIN's one-shot
fire, and the real harvest/completion remain required even after implementation tests pass.

## RECALL EVIDENCE

- Queried the Codex memory registry with `ffi2|prefire|intent contract`; no hits in that scope.
- Searched the `.omx/research/` Markdown/JSON corpus by content for
  `prefire|first.measurement|intent.contract`, then searched Markdown for
  `legacy.*(reference|receipt)|nested.*(bytes|reference|receipt)|every referenced JSON|source_t4_leg|candidate_t4_receipt.*bytes`.
  Broad searches returned unrelated historical training and custody material; the directly
  relevant contract matches were pr12, ffi1, ffi2, and RLC2's decision/fire-order packet.
  No explicit nested-legacy byte-count exception was found among those inspected matches.
- Searched design/docs, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, and
  `.omx/state/canonical_task_status.jsonl` for `prefire|first.measurement|intent.contract`.
  The task ledger preserves the MAIN/second-family decision gate; older DAG first-fire entries
  concern training work and do not resolve this reference scope.
- Generated the canonical equations registry using
  `.venv/bin/python tools/list_canonical_equations.py --json` and filtered the same terms.
  The matching GT-lineage offset equation uses “First measurement” in an unrelated empirical
  description; it provides no lifecycle custody exception.
- Read the live MAIN hot state and inspected existing seal, fire, CUDA worker, harvest-poller,
  and completed-direct-leg code. Beyond the charter seeds, the actual historical nested
  reference and the helper's emitted key set establish the new scope question. That changed
  this run from implementation to the charter-mandated STOP.
- PR9's manifest finding confirms why real dependency coverage cannot be replaced by a
  typed success boolean. It supplies no nested-reference exception.

This is bounded recall, not a claim of global absence or an exhaustive ambiguity audit.

## Provenance, boundaries, and delivery state

- Pr12 memo SHA-256: `50d00e3956dc7ae5d3b15379d2ae6f8704119817b0b58f50aa30413b97eacadc`.
- Ffi1 report SHA-256: `b61562bb9bce9450689648b43b395e19ce81fb7099d82006446b6631c190d78d`.
  Verified byte-identical to `HEAD`; not edited or re-landed.
- Ffi2 clarification charter SHA-256: `3c0f391541a5af87f0cccee30e5b0c639c2d576d5008d3c6ad8617cf27b9d535`.
- Completed-direct-leg provenance pin: `0524522f024cc30d0bfe0296815cc3d9a2804c3b`;
  pointer move-42 provenance pin: `d2803c2148b6153e6fc26d18428cd2d3cda3ea3e`.
- Implementation commit: absent. Frozen receipt and frozen receipt SHA: absent. The mandated
  `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json` path remains reserved for an
  actual freeze after implementation; an observation is not substituted for that receipt.
- Unit tests: 0 run. Ruff and the four requested regression suites: not run because no source
  or test changes were made. No Python review passes are claimed. No catalog number claimed.
- Only this report, the observation JSON, and the `ddm_ffi2` checkpoint are this arm's work.
  Report landing custody is reported separately from the absent implementation landing.
- No Modal call, fire, timing window, timing producer, encoder, decoder, scorer, evaluator,
  n600 pass, paid dispatch, candidate materialization, or harvest ran. The byte inspection
  did not execute the timing validator or remeasure decode time.
- No candidate payload was created, moved, deleted, deduplicated, or discarded. No bulky
  artifact was generated. Existing payloads and sibling work remain under their existing custody.
- No source/test, sealed tree, PR tree, `upstream/`, rlc2/sj1 live directory, protected common-
  contract file, pointer, timing requirement, or lane/resource/spend guard was edited. The
  shared staged index was inspected read-only; no manual staging or stash was used.
- All six solver-stack hooks are N/A for this research-only specification STOP: no sensitivity,
  Pareto constraint, allocator update, deployable actuator, or empirical posterior exists;
  the explicit ambiguity STOP prevents implementing competing interpretations as a probe.
- The common contract's old frontier paragraph is superseded by the charter and current hot
  state. The exact pointer did not move; the score goal was not achieved.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / pr12 adjudicator; consumer store: this memo,
  `ddm_ffi2_20260910/CONTRACT_SCOPE_OBSERVATION.json`, and the normative contract; fire trigger:
  harvest this STOP packet. Clarify byte-count custody scope for nested legacy references,
  covering both the pinned source leg and the unchanged completion builder, while retaining
  MAIN's accepted authorization digest split.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: ddm_ffi2; consumer store: charter-named source/tests,
  the completed compliance memo, and `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`;
  fire trigger: binding clarification resolves that scope. Resume the full implementation,
  required reviews/tests, single code-and-test serializer commit, and truthful freeze receipt.
  RLC2/MAIN's existing producer/fire/harvest orders are FOLDED into their existing gated chain;
  this packet releases none of them.

Frontier unchanged: `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)`.

## LIVE-HYPOTHESES

- The new-reference-only reading may be intended because pr12 explicitly preserves the old
  timing schema, helper, validator, and pinned source leg. It requires a scope clarification.
- Recursive byte-count custody could be preserved through a separately specified supplemental
  binding while leaving the historical leg untouched. That is plausible because the receipt
  exists and its SHA matches, but no such supplemental contract has been authorized here.
- The overall first-measurement lifecycle remains plausible because permission, measurement,
  and promotion are separate states. Its implementation and real end-to-end control remain unproven.

## DEAD-ENDS

- Reopening ffi1's file-byte/canonical-digest question: closed by MAIN's explicit split binding.
- Inserting a byte count into the required historical leg: closes against its exact immutable
  1,553-byte SHA pin and the charter's protected-tree/unchanged-leg boundaries.
- Silently applying or exempting recursive reference checks: closed for this implementation
  attempt by the explicit ambiguity-STOP rule; the alternatives are not adjudicated.
- Calling this observation a freeze, implementation, or fixture/real positive control: closed
  because none of those actions occurred. No score improvement was measured.
