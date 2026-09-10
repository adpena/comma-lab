# ddm_ffi1 — implement ddm_pr12's pre-fire intent contract VERBATIM (charter, MAIN 2026-09-10)

## What
ddm_pr12 (second family, sol xhigh) ruled APPROVE-WITH-CHANGES on rlc2's first-fire proposal and wrote
the exact contract: `.omx/research/ddm_pr12_adjudicate_first_fire_intent_contract_20260910.md`
(sha256 prefix 50d00e3956dc7ae5; sections "Exact normative contract" (lines ~69–342), "Exact CLI and
lifecycle" (~343–474), "Required typed refusals" (~475–501), "Prospective freeze and real positive
control" (~523–543)). Implement THAT text. Do not redesign, rename, loosen, or add convenience paths.
Where the text is ambiguous, STOP and report the exact sentence; "implement the schema as written" is
not ambiguity.

## Surfaces (pr12 names them; read each first)
- `src/tac/candidate_seal.py`: `candidate_prefire_intent.v1` (typed, versioned; all non-timing gates
  frozen; NO seconds; `score_claim=false`); the normal `validate_seal` and every normal consumer REFUSE
  it as not a seal (typed refusal); the completion producer that, after harvest, builds the unchanged
  `t4_direct` leg (`tac.decode_wall_clock.build_t4_direct_leg`) and issues the NEW completed seal
  (`candidate_seal.v3` per pr12) naming the immutable intent digest and the receipt; never mutate the
  intent or a historical seal.
- `tools/make_candidate_seal.py`: the intent emitter (exact flag names from pr12); refuses when any
  non-timing gate fails; the risk receipt fields pr12 requires (lineage to a receiver with a completed
  T4 leg, measured local cost fraction of the delta, risk ceiling ≤ 1,260 s, etc. — exactly as written).
- `tools/fire_modal_auth_eval.py`: the separate first-measurement mode (exact flag names), bound to the
  intent digest, exact candidate identity, contest-CUDA T4 axis, cold n600 public entrypoint, lane,
  retained receipt destination, budget/timeout, and a one-time nonce (a second invocation with the same
  authorization refuses before subprocess); dry-run consumes nothing; unknown state, missing
  authorization, identity/pointer drift refuse before dispatch. Keep every existing lane/resource/spend
  guard. Record the intent digest in dispatch custody.
- Tests: every typed refusal pr12 lists (negative), the normal-seal refusal of an intent, nonce
  single-use, drift refusals, and the completion path on a synthetic receipt — while stating in the
  memo (pr12 §"Prospective freeze") that fixtures cannot close the pass path: the real positive
  control is rlc2's producer + MAIN's real harvest.
- `docs/meta_bug_class_catalog.md`: if pr12 requires a gate number, claim it via
  `tools/claim_catalog_number.py` (never hand-edit the counter).

## Deliverable
1. Code + tests landed in ONE serializer commit (`--expected-content-sha256` per file, zsh arrays, two
   visible review passes per changed .py, no co-author trailer, tags `[no-triality] [p0-ledger-ok]`);
   ruff clean; the existing suites `test_candidate_seal.py`, `test_decode_wall_clock.py`,
   `test_decode_wall_clock_t4_direct.py`, `test_decode_timing_concurrency.py` still green.
2. A frozen contract receipt `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`: pr12 memo
   sha, the landed commit sha, the schema ids and version strings, the flag names, utc — committed. The
   implementation landing must precede every rlc2 producer timestamp (pr12 §freeze).
3. Memo `.omx/research/ddm_ffi1_implement_prefire_intent_contract_20260910.md` with a per-clause
   compliance table (pr12 sentence → code location → test) and every deviation (there should be none).

## Boundaries
- No Modal, no fires, no timing windows, no scorer runs, no n600 passes. Unit tests only.
- Never edit `upstream/`, the PR tree, sealed candidate trees, or `src/tac/decode_wall_clock.py`'s
  completed `t4_direct` requirements (pr12: unchanged). Do not touch rlc2's or sj1's live directories.
- Host: sj1 pass 6 (Opus) and vr8 (custody audit) share the host; no quiet windows are planned.

## OPTIMAL FORM
- Reference form: pr12's exact contract; the landed `t4_direct` implementation (0524522f0) and the v2/v3
  timing sampler landings (52c4962cc, 6a857a1ec) as the pattern for "second-family text implemented
  verbatim and hash-frozen".
- Provenance pins: pr12 memo sha 50d00e3956dc7ae5…, rlc2 proposal sha 021b44b6…, pointer move 42
  d2803c214.

## Prior negatives accounted (operator 2026-08-15)
- dwc1: a validator that only fixtures satisfy is a gate with no door — pr12's real-control clause is
  binding; say so in the memo, do not claim the door is proven.
- mv2/mv3/rlc2 stopped on exact clauses; here the text is exact — implement, report ambiguity only.
- pr10: rules fixed after data — freeze BEFORE rlc2 resumes.

Checkpoint as `ddm_ffi1` every ~10 tool uses. Final message: commit sha, frozen-contract receipt sha,
test counts, the per-clause table summary, every boundary, and the frontier line
`composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)` unchanged.
