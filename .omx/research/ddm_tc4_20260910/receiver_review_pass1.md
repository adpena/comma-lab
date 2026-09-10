# TC4 receiver independent review 1 — findings

Reviewer: `/root/tc4_review`; 2026-09-10. Read-only source review; no receiver, scorer, or measurement launched.

Reviewed `experiments/ddm_tc4_receiver.py` SHA-256 `afa49ca3252a1c6ba9b13c69d411504192007dd1adf343b186385d594e11dcaa`, including the source residual reader, native build helper, receiver checkpoint implementation, and TC4 maps/fast algorithm export.

1. **Medium, fresh-run request aliases completed first run.** `public(600, "public_fresh2")` returns global `PUBLIC_IDENTITY.json` before considering the requested work directory. A fresh-run request after the first completed identity therefore performs no new decode and returns the first run's receipt. Scope completion receipts to `name`; reserve the canonical receipt for the designated primary run.
2. **Medium, staged-candidate recovery is not tied to current selection.** `stage()` checks the freshly selected result, but the existing-destination branch checks only runtime identity and source binding. If `SELECTION.json` is changed to another valid retained result, the branch returns the previously staged candidate. Require recovered mask and archive hash/size to equal the current selected result.

The algorithm-only fast export preserves relative imports and excludes the research CLI. TC4 unpacking and dispatch retain the TC3 observe hook. Full-frame checkpoints delegate to the nested TC4 snapshot/restore, and bind runtime sources, stream, weights and native libraries. The native helper reuses verified libraries rather than rebuilding them on every resume. Public identity requires exact raw bytes, all 600 decoded token planes and matching literal report; it makes no score claim. These findings concern recovery custody, not observed decoding or pricing correctness. Two clean review passes remain required after fixes.
