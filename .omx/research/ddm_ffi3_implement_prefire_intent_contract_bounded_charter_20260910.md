# ddm_ffi3 — implement ddm_pr12's pre-fire intent contract with BOTH clarifications and a bounded STOP rule (charter, MAIN 2026-09-10)

Two arms (ffi1, ffi2) stopped on clause-level questions in pr12's text. Both are answered here, and the
STOP rule is bounded so that only a genuine contradiction between two NEW-object requirements stops
this arm. Everything in `.omx/research/ddm_ffi1_implement_prefire_intent_contract_charter_20260910.md`
(surfaces, deliverables, boundaries, OPTIMAL FORM) binds unchanged; read it and ffi1's/ffi2's clause
tables (`.omx/research/ddm_ffi1_implement_prefire_intent_contract_20260910.md`,
`.omx/research/ddm_ffi2_implement_prefire_intent_contract_20260910.md`) before starting.

## Clarification 1 (ffi1; MAIN, pending pr12-family ratification) — the authorization's intent binding
Where pr12 writes `authorization.intent.sha256 = "<intent digest>"` (line ~308) while line ~85 requires
exact byte count + SHA-256 for referenced JSON: bind BOTH and refuse on either mismatch —
`authorization.intent.file_sha256` + `authorization.intent.file_bytes` (exact file bytes of the committed
intent JSON) AND `authorization.intent.digest` (the canonical self-omitting intent digest exactly as pr12
defines it). Dispatch custody records all three.

## Clarification 2 (ffi2; MAIN, pending ratification) — scope of `{path, bytes, sha256}`
pr12 line ~84 ("Every referenced JSON or manifest carries `{path, bytes, sha256}`") governs every reference
INTRODUCED by the new objects (`candidate_prefire_intent.v1`, the authorization, the completion producer's
`candidate_seal.v3` fields, the risk receipt, the retention manifest). It does NOT rewrite the embedded
LEGACY objects that pr12 explicitly preserves unchanged — the completed `t4_direct` leg and its
`{path, sha256}` receipt references validated by the unchanged `tac.decode_wall_clock` (pr12: "retains the
completed t4_direct requirements"). Where a new object embeds a legacy leg, it references the legacy leg's
FILE with `{path, bytes, sha256}` and leaves the leg's own interior to the legacy validator.

## Bounded STOP rule (replaces the ffi1 charter's ambiguity clause for this arm)
- A question of the form "does a NEW-object requirement also rewrite an unchanged LEGACY object?" is
  answered by Clarification 2: new objects only. Apply it and record the choice in the compliance table
  (`clause → decision → code location → test`). Do not stop.
- A question of the form "which of two representations does the authorization bind?" is answered by
  Clarification 1: both. Do not stop.
- STOP only if two requirements on the SAME new object contradict each other so that no implementation can
  satisfy both (quote both sentences). Anything else: implement the stricter reading, record it, continue.

## Deliverable (as ffi1's charter, restated)
1. Code + tests in ONE serializer commit: `src/tac/candidate_seal.py` (intent schema + typed refusal by the
   normal validator + completion producer → unchanged `build_t4_direct_leg` → `candidate_seal.v3` naming
   the intent digest and receipt), `tools/make_candidate_seal.py` (intent emitter, exact flags from pr12),
   `tools/fire_modal_auth_eval.py` (first-measurement mode bound to intent file_sha256+bytes+digest, exact
   candidate identity, CUDA axis, cold n600 public entrypoint, lane, retained receipt destination,
   budget/timeout, one-time nonce; dry-run consumes nothing; identity/pointer drift, unknown state, missing
   authorization refuse before subprocess; every existing lane/resource/spend guard kept), tests for every
   typed refusal pr12 lists and the completion path on a synthetic receipt; `ruff` clean; two visible review
   passes per changed .py; no co-author trailer; tags `[no-triality] [p0-ledger-ok]`; the existing suites
   (`test_candidate_seal.py`, `test_decode_wall_clock.py`, `test_decode_wall_clock_t4_direct.py`,
   `test_decode_timing_concurrency.py`) still green.
2. `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`: pr12 memo sha, both clarifications
   verbatim + this charter's sha, the landed commit sha, schema ids/versions, flag names, utc — committed.
3. Memo `.omx/research/ddm_ffi3_implement_prefire_intent_contract_20260910.md` with the per-clause table incl.
   both clarification rows, and the statement that fixtures do not close the pass path (pr12 §freeze: the real
   control is rlc2's producer + MAIN's real harvest).

## Boundaries (unchanged)
No Modal, no fires, no timing windows, no scorer/n600 runs; never edit `upstream/`, the PR tree, sealed
trees, or `src/tac/decode_wall_clock.py`'s completed `t4_direct` requirements; do not touch sj1's live
directories. Host is shared with sj1 pass 6 (Opus).

## OPTIMAL FORM
- Reference form: pr12's exact contract (sha 50d00e3956dc7ae5…) + Clarifications 1–2; the landed
  `t4_direct` (0524522f0) and v3-sampler (6a857a1ec) landings as the "verbatim + frozen" pattern.
- Provenance pins: pr12 memo sha; ffi1/ffi2 report shas (record); pointer move 42 d2803c214.

## Prior negatives accounted (operator 2026-08-15)
- ffi1/ffi2/rlc2/mv2/mv3 stopped on exact clauses — this charter answers the two found and bounds the rule.
- dwc1's gate with no door — fixtures cannot prove the door; say so.
- pr10 — freeze BEFORE rlc2 resumes; the frozen receipt is part of the landing.

Checkpoint as `ddm_ffi3`. Final message: commit sha, frozen receipt sha, test counts, the compliance table
summary incl. the two clarification rows, every boundary, and the frontier line
`composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)` unchanged.
