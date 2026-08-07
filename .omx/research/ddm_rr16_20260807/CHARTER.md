# ddm_rr16 — round-16 recursive adversarial review (counter 1/3; the load-bearing round)

**Critical-path clause:** the cycle is ONGOING at 1/3 (rr13 NOT-CLEAN → rr14 NOT-CLEAN → rr15
CLEAN). New unreviewed code landed AFTER rr15 read the tree: mx1t's `torch-facets` analyzer +
tail-average A/B (committed by MAIN as "ddm_mx1t torch-facets analyzer + tail-average A/B (arm
commit repair)"). This code is LOAD-BEARING: its tail-average verdict (K=8 beats final,
0.0010673205057779949 vs 0.0010732014973958333) is now the recorded SELECTION POLICY for the
ARM-CAP vs ARM-VEH race and the n120 config. The endpoint (~21:09 timeout → resume → verdict →
selection) consumes it within hours. A clean round advances 1/3 → 2/3; a defect found NOW is
worth far more than one found after selection.

**Review corpus (the complete delta since rr15):**
- The mx1t commit (git log --grep "ddm_mx1t torch-facets" for the sha; full diff): the
  `torch-facets` mode in experiments/ddm_mx1_pr130_semantic_renderer.py (~line 1166+) + the
  tests appended to experiments/tests/test_ddm_mx1_memory_probe.py (~line 953+)
- MX1T_FINDINGS.md + mx1t_facets_receipts.jsonl + mx1t_checkpoint_copy_receipts.jsonl
  (.omx/research/ddm_mx1t_20260807/)
- The commit-repair event itself: the arm's serializer failed rc=128 in its sandbox; MAIN
  committed the arm-authored working tree. Verify the committed bytes match what the arm's
  receipts describe (no drift between the arm's measured artifacts and the committed code).

**Review axes (findings CRITICAL/Medium/Low; fix-or-route inline where small):**
1. **Tail-average correctness (THE axis):** is the K-checkpoint average a true parameter-space
   mean over the SAME tensor set the model loads (no missed/extra tensors, no dtype-degrading
   accumulation, no averaging of non-parameter state like optimizer moments)? Does the averaged
   model go through the IDENTICAL eval path as the final checkpoint (same batching, same
   roundtrip, same pair_ids)? A subtle asymmetry here silently biases the selection policy.
2. **Statistical honesty of the K=8 win:** delta is −5.88e-6 on n32. Is that above the
   instrument's own repeatability floor? Cheap check: re-evaluate one checkpoint twice (bit-
   identical expected on CPU — if not bit-identical, the noise floor must be stated) and/or
   compare the K∈{2,4,8} ordering for monotonic coherence. If the win is sub-noise, the
   selection policy should say "tail-average NOT WORSE, adopted for doctrine-compliance" —
   an honesty relabel, not a reversal.
3. **Facet-metric definitions:** per-class d_seg directionality (GT-side vs prediction-side),
   margin-histogram bin edges fixed across steps (a moving binning would fake trends), churn
   metric denominator. Verify the near-flip-vs-stuck VERDICT actually follows from the
   receipts' numbers (re-derive the trend from mx1t_facets_receipts.jsonl, don't trust prose).
4. **Checkpoint-copy custody:** copies + SHA receipts exist under checkpoint_copies/ but are
   deliberately unstaged (large binaries). Verify the receipts JSONL (committed) pins each
   copy's sha so the analysis is reproducible; flag if any receipt lacks its sha.
5. **Repair integrity:** confirm the MAIN-committed diff contains ONLY mx1t-authored changes
   (no absorption of unrelated dirty working-tree rows — the #911 genus; the active-lane
   ledger was deliberately left out, verify it stayed out).
6. Per-round assumption-challenge axis: answer explicitly.

**Verdict:** ROUND16_FINDINGS.md with CLEAN (→ 2/3) or NOT-CLEAN (fix inline, → 0/3). If a
tail-average defect is found, say loudly whether the SELECTION POLICY must be held before the
endpoint consumes it. Do NOT touch the live run dir; NO Metal; CPU re-evaluations of copied
checkpoints are allowed (that is the scorer slot — mark it; modest RAM, chunked).

**Discipline:** serializer + POST-EDIT `--expected-content-sha256` per file; tags
`[no-triality] [p0-ledger-ok]`; review_tracker ×2 per .py; NO Claude/AI attribution or
Co-Authored-By trailer — commits are the operator's alone.
