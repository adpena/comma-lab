No data moved or deleted. AP had only 5.726807 GiB available—already below the mandatory 10 GiB reserve—so the governed certifier blocked both candidates:

- `pk4`: 56.742130 GiB allocated; projected AP availability −51.015 GiB.
- `wwc1`: 1.954491 GiB allocated; projected AP availability 3.772 GiB.
- Reclaimed: 0 B; 50 GiB target not met.
- Classified: 142 items—46 blocked, 3 live-custody skips, 93 protected skips.

Evidence is in the [memo](/Users/adpena/Projects/pact/.omx/research/ddm_vr2_vertigo_reclaim_round2_20260831.md:7), [item ledger](/Users/adpena/Projects/pact/.omx/research/ddm_vr2_vertigo_reclaim_round2_20260831.jsonl:1), and [phase ledger](/Users/adpena/Projects/pact/.omx/research/ddm_vr2_move_phase_ledger.jsonl:1).

The required serializer landing was attempted but remains uncommitted: Git object writes failed with `Operation not permitted`, and serializer fallback refused both SSDs because neither retained its required 40 GiB reserve. The [refusal receipt](/Users/adpena/Projects/pact/.omx/state/commit_serializer_fallback_refusals/20260831T175943.544611Z-76758/receipts.jsonl:1) preserves the blocker. The index remains untouched.

No scorer or exact evaluation ran. The frontier remains AFR1: `S=0.14797617125559104`, 180,002 B, `[contest-CUDA T4 n600]`.

## NEXT_IF_RESUMED

- **Disposition:** QUEUED; **owner:** MAIN storage; **consumer store:** `/Volumes/APDataStore/pact/ddm_pk4_20260813`; **fire trigger:** AP availability reaches 66.742130 GiB and terminal-consumer, `lsof`, and manifest checks remain clean.
- **Disposition:** QUEUED; **owner:** MAIN storage; **consumer store:** `/Volumes/APDataStore/pact/ddm_wwc1_20260831`; **fire trigger:** pk4 reaches a terminal disposition and AP availability reaches 11.954491 GiB; preserve committed references with an old-path symlink.
- **Disposition:** CONDITIONAL; **owner:** next reclaim arm; **consumer store:** AP cold storage; **fire trigger:** a certified batch leaves at least 10 GiB on AP and Vertigo still lacks 50 GiB free; re-scan before selection.
- **Disposition:** OWED-HARDENING; **owner:** storage apparatus; **consumer store:** `tools/vertigo_certify_move.py`; **fire trigger:** the next retirement or any candidate containing a symlink; add target-manifest equality and durable partial-retirement failure evidence.
- **Disposition:** BLOCKED-LANDING; **owner:** MAIN Git; **consumer store:** current Pact worktree and serializer receipt; **fire trigger:** Git object writes become permitted or an authorized external fallback tier has at least 40 GiB reserve; rerun the serializer unchanged.

## LIVE-HYPOTHESES

- Restoring AP capacity should reopen both moves because destination headroom was the only certifier gate reached.
- The 46 headroom-blocked rows contain 288.673870 GiB, so a later certified subset could clear the Vertigo target once AP capacity exists.
- Some conservatively protected rows may contain movable subtrees; precise consumer carve-outs could expose them without breaking custody.

## DEAD-ENDS

- Copying either candidate now is closed: both violate the mandatory AP reserve.
- Moving the entire pk4 store is closed: prior evidence authorizes only its `jacobian_bank`.
- Hand-written `mv`, deletion, local fallback, or citation rewriting is closed: the charter requires certified AP moves and preserved paths.
- Inventing hashes for blocked or skipped items is closed: no bytes were copied or retired, so their ledger hashes correctly remain null.
- Moving sealed referenced trees without a proven carve-out is closed: conservative protection remains binding.
- Bypassing the serializer with a direct commit is closed: Git object writes are denied and the contract forbids alternate landing paths.