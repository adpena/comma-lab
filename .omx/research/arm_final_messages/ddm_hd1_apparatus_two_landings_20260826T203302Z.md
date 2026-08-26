Both cures are complete and verified, but not landed on `main`: this sandbox denied Git-object writes, so the charter-authorized fallback retained two verified, cherry-pickable bundles.

- Landing 1, serializer fallback: intended commit `9d89f98d…`; [bundle](/Volumes/VertigoDataTier/pact/ddm_hd1_landing1/receipts/commit_serializer_fallbacks/20260826T202050.386739Z-34035/intended-commit.bundle), SHA-256 `ae44b6b4…`. Synthetic denial returned rc=17; normal commit returned rc=0. The live denial also returned rc=17 with full custody.
- Landing 2, pin preflights: intended commit `bdac5e0e…`; [bundle](/Volumes/VertigoDataTier/pact/ddm_hd1_landing2/receipts/commit_serializer_fallbacks/20260826T203053.720859Z-45854/intended-commit.bundle), SHA-256 `3ec66dcd…`.
- Real jf2 control: broken original copy refused before advisory work at rc=2 and before seal creation at rc=3. The fixed runtime passed dry-run rc=0 and produced a `SEAL_VALID` seal.
- SSD census: 23 qualifying runtime directories; 11 mismatched and 12 consistent. Four po1 mismatches confirmed the prediction of additional cases beyond dg2/jf2.
- Validation: serializer controls 3 passed; serializer regressions 81 passed; consumer regressions 82 passed; Ruff, compilation, diff checks, and two review passes all passed.
- Durable handoff: [ddm_hd1_apparatus_two_landings_20260826.md](/Users/adpena/Projects/pact/.omx/research/ddm_hd1_apparatus_two_landings_20260826.md) and [machine-readable control receipt](/Volumes/VertigoDataTier/pact/ddm_hd1_apparatus_two_landings/receipts/pin_consistency_controls_and_sweep.json).

No scorer, Modal, or authoritative evaluation ran. `upstream/` was untouched. Own-vehicle frontier remains **gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]**.

Prior memory informed only the controls-first and review discipline; all operative facts were verified against the current checkout and retained artifacts.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store: landing-1 `receipts.jsonl`; fire trigger: verify/fetch its bundle, cherry-pick `9d89f98d…`, then rerun the serializer suites.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store: landing-2 `receipts.jsonl`; fire trigger: after landing 1, verify/fetch its bundle, cherry-pick `bdac5e0e…`, then rerun the 82-test consumer suite.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: jf2/po1 runtime custodians; consumer store: `pin_consistency_controls_and_sweep.json`; fire trigger: before consuming any of the 11 mismatched runtimes, re-pin and recheck it or explicitly retire it.

## LIVE-HYPOTHESES

- Bespoke materializers outside the canonical runtime assembler likely create the remaining half-updated pins, because all 11 mismatches cluster in jf2 and po1 while 12 other runtimes pass.
- The Git denial likely comes from per-process sandbox policy, because `.git/objects` reports mode 0755 while workspace and SSD writes succeed.

## DEAD-ENDS

- Creating another pin-checking framework: the canonical checker, repair helper, Modal wiring, and assembler prevention already existed.
- Prevention-only controls: both real already-broken cases were exercised.
- Treating #1237 as historically drained: 11/23 current in-scope runtimes mismatch.
- Retrying direct commits from this process: repeated attempts reproduced the same object-store denial; MAIN must consume the retained bundles.

