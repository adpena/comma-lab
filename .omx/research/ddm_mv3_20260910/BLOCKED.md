# ddm_mv3 — signature-change STOP receipt

No implementation landed. The exact mv3 STOP definition applies to a call in mv1's new test.
This is a charter boundary, not a claim that the optional keyword makes the call incompatible.

## Exact STOP evidence

mv3 says: “Semantic conflicts (the intervening change edits the SAME lines mv1 edits, or mv1's hunk calls a function whose signature changed) → STOP and report the exact lines; do not improvise.”

Commit `86961e487` added `require_decode_wall_clock: bool = False` to
`src/tac/candidate_seal.py:1154` (`validate_seal`, definition begins at 1148).
The base definition begins at 1131 and does not include that parameter.
The mv1 patch adds three calls to `seal.validate_seal(path, pointer_path=pointer)` in
`src/tac/tests/test_artifact_moved.py:237`, `:239`, and `:242` (physical patch lines 412, 414, 417).
AST extraction verified both signatures, all three calls, and attribution to `86961e487`.
The new parameter is optional. Compatibility remains untested; this receipt does not label it broken.
The exact comparison is retained in `BLOCKED.json`.

## Custody and scope

- Patch SHA verified: `ea104296ba676338472bb884b0356bb88c44e1da122d2eb68e4d6e645d2e6c92`. Exactly 21 patch files; ten pre-existing evidence/memo files match their declared hashes. No ExFAT stub included.
- First observed HEAD: `8a53de8b0ae58f3af283f645a687413b079366c9`; saved pre-work snapshot: `9f54ffdb779f5ce0f379047fb75d35f755833a9b`; signature audit HEAD: `7e770638e5609fed2a3bcfe231508ab11bd0855b`; receipt HEAD: `7e770638e5609fed2a3bcfe231508ab11bd0855b`. HEAD moved through other workers; this arm made no commit.
- rp1 GO absent at final check. No rp1/rlc1 payload or runtime edited. Shared staged diff unchanged.
- No patch applied; no catalog claimed; no review marks; no serializer attempted; no queue landed marking. Receipt files and a canonical task blocker event are this arm's only changes.
- Tests not run: 0 passed / 0 failed. Ruff, import verification and strict census not run. The original 171-pass receipt belongs to mv1 and is not current validation.
- No rebased hunks. Classified context-only insertions: dwc1 `86961e487` beside retained-payload validation; vr7 `03efa6682` beside imports. `bf467026f` is the timing-instrument commit and does not touch either named source file in the prescribed path log.
- The named `final_serializer.stderr` input does not exist. Read equivalent `ddm_mv1_20260910/serializer.log`: Git object creation was denied, rc=17 with fallback bundle. No fresh Git-write denial is claimed.

## RECALL EVIDENCE

Content search `MOVED.json|artifact_moved|moved.manifest|move.*certificate` across `.omx/research/` memos and arm receipts; then `MOVED.json|artifact_moved|ddm_mv1` across `docs/`, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, canonical task status and lane registry. Raw graph/task matches retained in `corpus_recall.txt`. Ran `tools/list_canonical_equations.py --json`: 483 entries; zero direct `MOVED.json|artifact_moved|moved.manifest` matches. Full export retained in `equations_recall.json`. This is a scoped absence.

Beyond charter seeds, `ddm_tc3_20260910/recovery_review.md` isolates the original MOVED incident to receipt construction after a successful child run; it does not prove the new combined tree. `ddm_vr7_certified_deletion_20260910.md` confirms retained reproducer and runtime deletion checks remain binding. `ddm_rp1_round2_respawn_charter_20260910.md` confirms source binding and stage-boundary constraints. Task rows already own installation and later census migration: reused `ddm_mv1_install_20260910` instead of inventing another task. No matching design/graph item supplied an exception to the STOP rule.

Memory quick pass used the custody guidance in MEMORY.md:118 to keep unlanded evidence distinct from commits. Current patch/refusal facts were re-read locally, not asserted from memory. Plan change: preserve the live tree and return the signature exception to MAIN; no mechanism rewrite or broad migration.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store: `.omx/research/ddm_mv3_20260910/BLOCKED.json`; fire trigger: harvest this blocker. Amend the STOP exception for this optional keyword or supply a compliant refreshed patch, then recheck rp1 GO absence. Canonical task: `ddm_mv1_install_20260910`; event receipt: `task_event.json`.

composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40) unchanged. No score measured by this arm.

**LIVE-HYPOTHESES:** The combined tree may preserve mv1's behavior because the added keyword defaults to False and the new calls omit it. This needs authorization under the charter and the full required regression suites.

**DEAD-ENDS:** Treating all changes as context-only is disallowed by this charter's explicit signature clause (INSTANCE scope). Counting mv1's historical tests as current combined-tree validation is unsupported. Neither conclusion kills the MOVED resolver mechanism.
