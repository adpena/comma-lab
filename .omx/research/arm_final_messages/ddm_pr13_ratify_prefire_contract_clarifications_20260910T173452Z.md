Review complete. The full adjudication is in [ddm_pr13_ratify_prefire_contract_clarifications_20260910.md](/Users/adpena/Projects/pact/.omx/research/ddm_pr13_ratify_prefire_contract_clarifications_20260910.md).

Clarification 1 — **RATIFY:** raw file path/bytes/SHA plus canonical digest are distinct, jointly enforced bindings.

Clarification 2 — **RATIFY:** every new content reference is byte-pinned; the legacy `t4_direct` interior remains unchanged and is pinned by its containing file.

Clarification 3 — **RATIFY:** the valid order is arm produces → MAIN commits exact intent → MAIN authorizes; custody is recorded in `authorization.intent.commit`.

Clarification 4 — **RATIFY, NOT YET OPEN:** the real rlc4 intent and both normal-consumer refusal receipts had not been emitted at the review snapshot.

AMEND text: none.

Validation passed:

- `test_candidate_seal.py`: 53 passed
- `test_candidate_prefire_intent.py`: 62 passed
- `test_git_custody_read.py`: 5 passed

No scorer, exact evaluator, encoder, decoder, provider, Modal, or candidate fire ran. This review measured no score.

The single serializer attempt returned **rc 17** because sandbox Git writes were denied. MAIN must land the verified [fallback bundle](/Volumes/VertigoDataTier/pact/ddm_pr13_20260910/serializer/20260910T173038.453464Z-3819/intended-commit.bundle), fallback commit `0fc444e6943a27fc19320538fac3e09a040264c6`. Memo SHA-256: `3156992449ea4f71966dfe96adea1471aa3b735dae3bac4a9f49c4a16ce4eba3`. The memo remains untracked in the main worktree; the index is empty.

Frontier unchanged: `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `ddm_rlc4`; consumer store: retained SSD evidence tree, `.omx/research/ddm_rlc4_20260910/`, real intent, and serializer bundle; fire trigger: all real move-42 evidence completes and the intent emitter exits zero. Produce the intent and both refusal controls without dispatching.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: the pr13 bundle, committed rlc4 intent, and authorization object; fire trigger: MAIN lands the bundles and verifies exact path, bytes, raw SHA, canonical digest, refusals, and frozen gates. Commit the intent before producing authorization.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: authorization output directory, nonce record, call ledger, retained worker tree, and `MODAL_REMOTE_RESULT.json`; fire trigger: authorization validates, cost is below USD 5, MAIN holds the T4 lane, global single-flight is clear, and the pointer still matches. Fire exactly once.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: real `candidate_seal.v3`, adjudication, evaluation ledger, and pointer packet; fire trigger: the harvested receipt passes unchanged cold n600 `t4_direct` within 1,260 seconds plus every identity and score-bar check. Move the pointer only if the exact row qualifies.

## LIVE-HYPOTHESES

- Rlc4’s real intent may pass because the landed validator already passes equivalent synthetic identity checks; the real evidence joins remain untested.
- The cured receiver may finish cold T4 within 1,260 seconds because the retained calculation projects a 1,032.725-second ceiling; cross-host timing remains untested.
- Dual binding may survive the producer-to-MAIN handoff because both sides use the same canonical digest while independently checking exact bytes; that real cross-actor path remains untested.

## DEAD-ENDS

- Using one digest for both serialized-file and canonical-object identity is closed because they bind different inputs.
- Replacing the intent with equivalent content at another path or formatting is closed because path, bytes, SHA, digest, and committed blob must all agree.
- Rewriting the historical `t4_direct` interior is closed; the new objects pin its containing file without changing legacy semantics.
- Requiring the producing arm to create the Git object is closed; MAIN may commit the exact bytes before authorization.
- Treating fixture tests, a running trace, or intent emission alone as end-to-end proof is closed; real fire, harvest, direct timing, and consumer-valid v3 remain mandatory.