Completed the charter. The full handoff is in [ddm_cd1_working_tree_debt_landing_20260903.md](/Users/adpena/Projects/pact/.omx/research/ddm_cd1_working_tree_debt_landing_20260903.md:1).

Key results:

- Every observed path has an explicit `LAND`, `REVERT`, or `HOLD(live-arm)` disposition.
- 27/39 original non-live paths matched prior fallback bundles exactly: 69.23%, below the predicted 80%.
- Nine new serializer bundles were verified: eight logical landing units plus the final memo.
- Git index writes were denied, so those commits remain bundle-ready—not merged into `HEAD`.
- Two stray PQ1 markers were reverted after retaining their exact 266 bytes and hashes in [custody_receipt.json](/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/reverted_pq1_nested_markers/custody_receipt.json).
- FPC3 and GF2 landed concurrently on `main` as `7aada29099` and `ab93088002`.
- QBR1’s active claim ledger and GC1’s live source/test remain untouched.
- Registry lost-update and stale pointer/JG4 fixtures were repaired.
- Tests: 501 passed with one diagnosed stale fixture; the repaired JG2/JG4 suite then passed 46/46. Canonical-equation tests passed 22/22.
- All nine fallback bundles verify. Fast preflight passed, explicitly with zero full-codebase gates.
- No scorer, Modal, Metal, archive mutation, or exact evaluation was run. Frontier remains AFR1: `S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `main` and the CD1 serializer receipt directories; fire trigger: a Git-index-writable session. Apply the eight payload units in the memo’s stated order, verifying receipt hashes before each serializer landing.
- **HOLD(live-arm)** — owner: QBR1/MAIN; consumer store: `.omx/state/active_lane_dispatch_claims.md`; fire trigger: terminal six-cell burn receipt. Reconcile and serialize the stable claim delta.
- **HOLD(live-arm)** — owner: GC1; consumer store: GC1 verdict and `main`; fire trigger: terminal arm message. Run its named tests and review gates before landing.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: retired EP725 maintainer; consumer store: retained EP725 decode receipts; fire trigger: current-renderer bit-exact output receipt or explicit quarantine decision.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: retired V15 maintainer; consumer store: retained V15 compile custody; fire trigger: fresh current-source compile receipt proving deterministic archive behavior.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: retired PBR2/V9 maintainer; consumer store: the retained teacher-census receipt; fire trigger: rematerialization closing packet, pair-window, and renderer-manifest identity.

## LIVE-HYPOTHESES

- EP725 may remain output-identical because its drift came from safe-extraction hardening outside decoder arithmetic.
- V15 may reproduce its semantic archive because the last source change appears validation-oriented.
- PBR2/V9 may preserve its n64 result despite source-manifest drift if intervening changes are inactive for that fixture.
- The eight fallback units should apply without GF2 conflicts because their declared paths are disjoint.

## DEAD-ENDS

- Blind source-hash refreshes are closed; receipts or explicit quarantine are required.
- Treating serializer `rc=17` as a merged commit is closed.
- Auto-sorting inherited Ruff findings is closed for this landing because it would break verified owner-bundle identity.
- Landing QBR1 or GC1 state before their terminal triggers is closed.
- Unretained deletion of the PQ1 markers is closed.
- The ≥80% fallback-bundle prediction is closed at 69.23%.

