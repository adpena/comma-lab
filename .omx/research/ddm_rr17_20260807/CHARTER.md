# ddm_rr17 — round-17 recursive adversarial review (counter 0/3 after rr16 F1)

**Corpus (the complete delta since rr16 read the tree):**
1. MAIN's cure for rr16 F1: `.omx/research/ddm_mx1t_20260807/mx1t_provenance_addendum.json` —
   verify it HONESTLY releases the HOLD: recompute the gt_seg_cache.pt sha256 independently and
   compare; confirm the binding claim matches the live ARM-CAP fire argv (launch ticket
   .omx/research/ddm_mx1g_20260807/launch_ticket_mx1g_from_regen2.json argv_n32_arm_cap) and
   rr16's own reproduction numbers. If the addendum satisfies F1's release condition, SAY SO
   explicitly — the tail-average selection policy (K=8, n32 advisory) then stands for the
   arm-selection boundary.
2. MAIN's repair commits: "ddm_aa1+rr16 arm artifact commit repair..." + the addendum commit —
   verify NO absorption of unrelated working-tree rows (#911 genus): each commit contains ONLY
   the named files.
3. Light-touch data-doc audit of the three harvest tables (committed b61cacae01 / 81bb5edcb5 /
   the aa1 repair): JSONL parses, no fabricated receipts (spot-check 3 rows per table against
   the artifacts they cite), honesty labels present (score_claim=false where applicable).
   These are DATA docs — findings only where a row asserts a number its cited receipt does not
   contain.
4. Per-round assumption-challenge axis: answer explicitly.

**Verdict:** ROUND17_FINDINGS.md with CLEAN (→1/3) or NOT-CLEAN (fix inline where small, →0/3).
State loudly whether the selection-policy HOLD is RELEASED. NO Metal; CPU-only; no scorer
passes needed. Do NOT touch the live run dir. If the serializer fails with git-perms
"Operation not permitted", write findings to disk and say so in the final message — MAIN
repairs the commit.

**Discipline:** serializer + POST-EDIT `--expected-content-sha256` per file; tags
`[no-triality] [p0-ledger-ok]`; review_tracker ×2 per .py; NO Claude/AI attribution or
Co-Authored-By trailer — commits are the operator's alone.
