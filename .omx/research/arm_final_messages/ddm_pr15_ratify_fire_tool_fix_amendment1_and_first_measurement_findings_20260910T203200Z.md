The review is complete in [ddm_pr15 memo](/Users/adpena/Projects/pact/.omx/research/ddm_pr15_ratify_fire_tool_fix_amendment1_and_first_measurement_findings_20260910.md), SHA-256 `193a69241e748d08b96fc407f4d92e3ab918c3cf0b1803e286b24fd04d929b44`.

1. **Dispatch-path ordering — RATIFY.** The narrow exclusion is sound; `21c9b17b` is the causal fix and `808da43c` is the valid pinned snapshot.
2. **Replay guard — RATIFY AS IS.** Direct refusal receipts are already ignored; the empty launcher directory correctly caused refusal.
3. **Authorize import path — AMEND.** Add `sys.path.insert(1, str(REPO))`, add the foreign-cwd regression, regenerate the implementation manifest, and append amendment `[3]`.

Superseded authorizations must remain immutable. No in-object `SUPERSEDED` marker is needed: latest-freeze validation, nonce consumption, and job/output reuse already fail closed.

Validation: 136 tests passed, another 15 tests passed, and relevant Ruff checks passed.

Serializer: **rc 19**. Git-object writes were denied, then fallback custody refused because Vertigo had 36 GiB and APDataStore 32 GiB available—both below the mandatory 40 GiB reserve. The memo remains untracked and unstaged; no shared index contamination occurred.

Frontier: **move 43**, `S=0.1372848557085275`, 180,466 B, `[contest-CUDA T4 n600]`, archive `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`. Run3 still has no retained `MODAL_REMOTE_RESULT.json`; move 44 is unproven.

## NEXT_IF_RESUMED

- **BLOCKED-LANDING / owner operator or MAIN storage steward / consumer store:** the memo commit or governed fallback receipt; **fire trigger:** restore the SSD tier above the 40 GiB reserve, then rerun the serializer against unchanged memo hash `193a6924…`.
- **MEASURE / owner MAIN / consumer store:** run3 `MODAL_REMOTE_RESULT.json`, completed `candidate_seal.v3`, and pointer packet; **fire trigger:** none—v6 was already fired; harvest only when the retained result appears.
- **AMEND / owner MAIN / consumer store:** authorization tool, foreign-cwd test, implementation manifest, and freeze amendment `[3]`; **fire trigger:** run3 must first become terminal and its completion/pointer adjudication must be committed.

## LIVE-HYPOTHESES

- Run3 may turn the 60-byte rider into move 44, but only an accepted retained exact result can establish that.
- The one-line repository-root insertion should close the authorization consumer’s import graph; the regression and re-pin remain required proof.

## DEAD-ENDS

- Calling `808da43c` the causal fix is closed; it is only the valid implementation snapshot.
- Weakening replay protection for empty launcher directories is closed.
- Mutating old authorization objects to mark supersession is closed.
- Using amendment `[2]` for the import fix is closed; `[2]` is already occupied.
- Treating conditional arithmetic or a fire manifest as an exact score is closed.