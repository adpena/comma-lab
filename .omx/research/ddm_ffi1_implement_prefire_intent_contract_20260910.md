# ddm_ffi1 — STOP at an ambiguous intent-reference digest

Date: 2026-09-10  
Status: BLOCKED_CONTRACT_AMBIGUITY; implementation NOT landed; contract NOT frozen.  
Axis: source inspection only; research_only=true; score_claim=false; promotion_eligible=false.  
Tokens: [no-triality] [p0-ledger-ok]

## Exact blocker

The charter says: **“Where the text is ambiguous, STOP and report the exact sentence”**.
I stopped before editing implementation or tests. The pinned pr12 memo has SHA-256
`50d00e3956dc7ae5d3b15379d2ae6f8704119817b0b58f50aa30413b97eacadc`, verified from disk.

In `.omx/research/ddm_pr12_adjudicate_first_fire_intent_contract_20260910.md:84–86`:

> Paths are absolute custody paths. Every referenced JSON or manifest carries `{path, bytes, sha256}`;
> the validator re-reads it, requires exact byte count and SHA-256, parses it, and checks the semantic
> fields named below. A typed boolean never substitutes for re-derivation from the referenced bytes.

But the authorization shape at line 308 says:

```json
"intent": {"path": "<candidate_prefire_intent.v1>", "bytes": "<positive int>", "sha256": "<intent digest>", "commit": "<40-hex commit containing that exact intent>"}
```

Lines 76–81 define `intent_sha256` as SHA-256 of canonical JSON with the self field omitted.
The retained intent file includes that self field, and its serialization may also contain whitespace.
Consequently, file-byte SHA-256 and canonical self-omitting object SHA-256 are different calculations;
they cannot be treated as interchangeable identities. The contract does not state an exception for
this reference or explicitly say that “intent digest” here means the file-byte digest.

**Required clarification:** Does `authorization.intent.sha256` contain the SHA-256 of the exact
retained file bytes, with `intent.intent_sha256` separately checked for canonical identity, or does
this reference explicitly use the canonical object digest as an exception to lines 84–86?
I have adopted neither interpretation. This is a schema comparison ambiguity, not a claim that the
first-measurement lifecycle is infeasible. Verdict scope: INSTANCE of the pinned specification.

## Clause status table

No row below is an implementation PASS. Locations are pr12 line intervals; code/test entries are
absent because the charter requires STOP rather than implementation through this ambiguity.

| Pr12 clause | Code location | Test / status |
|---|---|---|
| 74–100: canonical and reference hashing, implementation custody | No change | BLOCKED: reference digest meaning above |
| 102–201: exact intent shape and false-authority exclusions | No change | NOT IMPLEMENTED |
| 203–230: non-timing comparisons from real retained bytes | No change | NOT IMPLEMENTED |
| 232–296: completed-T4 receiver-delta risk evidence | No change | NOT IMPLEMENTED |
| 298–341: independent committed authorization and one-shot consumption | No change | BLOCKED at line 308; remaining clauses NOT IMPLEMENTED |
| 343–400: exact intent and authorization producers | No change | NOT IMPLEMENTED |
| 402–439: first-measurement dispatch and quarantine | No change | NOT IMPLEMENTED |
| 441–473: harvest, unchanged direct leg, completed v3 seal | No change | NOT IMPLEMENTED |
| 475–501: typed refusals | No change | NOT IMPLEMENTED |
| 523–543: prospective freeze and mandatory real positive control | No freeze | NOT EXERCISED; synthetic fixtures cannot close the pass path |

There are no implemented deviations: there is no implementation. This table is a blocked-work
inventory, not the completed per-sentence compliance table owed by the charter.

## RECALL EVIDENCE

- Searched `.omx/research/` Markdown/JSON by content for `intent_sha256`,
  `implementation_manifest_sha256`, `candidate_first_measurement_authorization`, and
  `canonical object hashing`, excluding pr12 itself and generated index files. No additional
  clarification was found in that scope.
- Searched `docs/` (including design/SPEC documents), `CANONICAL_RESEARCH_INDEX*`,
  `sub015_DAG_*`, and `.omx/state/canonical_task_status.jsonl` for
  `prefire|first.measurement|intent.contract`. The task ledger's RLC2 fire order routes an
  approved implementation through MAIN and the second family before production. DAG matches
  concern earlier training/pre-fire reviews; they do not define this authorization reference.
- Generated the canonical equations registry with
  `.venv/bin/python tools/list_canonical_equations.py --json`: searched 483 returned rows for
  `prefire|first.measurement|intent.contract|canonical.object.hash`. The only match was the
  phrase “First measurement” in `cw1_gt_lineage_additive_pose_offset_v1`, unrelated to hashing.
- Inspected the current `src/tac/candidate_seal.py`, `tools/make_candidate_seal.py`, and
  `tools/fire_modal_auth_eval.py` entry surfaces and searched them for
  `prefire|first.measurement|intent_sha256`: no existing implementation resolving this field.
- Read the live MAIN hot state: ffi1 is the prerequisite to RLC2's resume. Its pointer line
  remains move 42. The common contract's older frontier paragraph is historical, superseded
  by the explicit charter frontier and live hot state.
- Queried the Codex memory registry for `prefire|intent.contract|serializer|governing`.
  Relevant process guidance was to repeat each post-edit hash option and verify actual landing;
  current serializer source confirms the process. It supplied no schema clarification.

Beyond the charter seeds, the ledger and current CLI inspection confirm that there is no already
implemented exception to reuse in the searched surfaces. This changed the disposition to the
charter's explicit ambiguity STOP. It did not justify a new field, renamed field, or weaker check.

## Boundaries and delivery state

- No source/test, upstream, PR, sealed candidate, rlc2, or sj1 tree was edited. No direct-leg
  requirements, pointer, lane/resource/spend guard, or shared staged index was changed manually.
- No Modal/fire, timing window, timing producer, encoder/decoder, scorer, evaluator, n600 pass,
  candidate materialization, or payload cleanup ran. No bulky payload was created or discarded.
- Unit tests: 0 run. Ruff and the four required suites: not run because implementation stopped.
  No Python file was changed, so Python review passes are not claimed.
- Implementation commit: absent. Frozen-contract receipt and receipt SHA: absent. In particular,
  no `PREFIRE_CONTRACT_FROZEN.json` was written with a false implementation or freeze claim.
- This memo and the ddm_ffi1 checkpoint are the retained report; their custody is separate from
  the still-owed single implementation commit. No additional catalog gate was claimed.
- Solver-stack hooks are N/A: this is a research-only specification blocker with no actuator,
  measured sensitivity, allocator update, deployment, or empirical posterior anchor.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / pr12 adjudicator; consumer store:
  `.omx/research/ddm_ffi1_implement_prefire_intent_contract_20260910.md` and the pr12 normative
  contract; fire trigger: harvest this STOP report. Clarify the exact meaning of
  `authorization.intent.sha256` and its relationship to the referenced-file hashing rule.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: ddm_ffi1; consumer store: charter-named source/tests,
  this compliance memo, and `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`;
  fire trigger: a binding clarification resolves that digest comparison. Implement and review
  the full contract, run the mandated suites, land through the serializer, and then retain the
  actual freeze receipt before RLC2 resumes. Existing RLC2/MAIN fire orders remain gated by
  that completed freeze; this report does not release them.

Frontier unchanged: `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)`.

## LIVE-HYPOTHESES

- The intended reference may be the file-byte SHA, with canonical intent identity checked inside
  the referenced object; this fits the general reference custody rule. It needs explicit resolution.
- The intended reference may instead be the canonical digest; the literal “intent digest” in the
  authorization shape makes that plausible. It needs an explicit exception or distinct custody rule.
- The overall lifecycle remains plausible because it separates measurement permission from timing
  and promotion authority. Neither implementation tests nor the real producer/harvest have proven it.

## DEAD-ENDS

- Silently equating canonical self-omitting SHA with full-file SHA: closed by their different inputs.
- Choosing a digest interpretation or adding a convenience field without clarification: closed by
  the charter's verbatim-implementation and ambiguity-STOP instructions.
- Calling this report an implementation landing, freeze, or real positive control: closed because
  no code was implemented, no freeze exists, and RLC2/MAIN have not exercised the lifecycle.
