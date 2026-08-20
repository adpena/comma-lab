Round 3 completed as a **FINDING round**: nine defects were confirmed—five claim/object-scope defects and four apparatus/test defects. The recursive clean-pass counter resets to **0**.

Key conclusions:

- rr8 proves one instrumented T4 instance passed at 464.5586 s. It does not prove repeat stability or clean-tree shipping.
- The 1.2855 host ratio is a one-sample point estimate, not a variance bound.
- rc1 is merge- and parse-proven, not full composed-receiver decode-proven.
- rr5’s correct identity denominator is 9/10 parsed fields; the tenth is the intentional raw-container delta.
- The task-ID lint, GT-lineage scanner, certification/cache ordering, and recall-lint test isolation were repaired.
- F19 controls prove different-session and stale checkpoints cannot be inferred as self; a unique fresh same-session checkpoint can.

The live Python denominator was **29**, not the charter’s stale 25. All 29 received two fresh review passes. Final verification: **342 tests passed**, compilation passed, fatal Ruff rules passed, live GT-lineage census returned zero findings, and `git diff --check` passed.

The complete receipt is [ddm_rv16_round3_finding_wave_20260820.md](/Users/adpena/Projects/pact/.omx/research/ddm_rv16_round3_finding_wave_20260820.md).

The required governed commit was attempted but refused before staging because this sandbox cannot create temporary Git object files. HEAD remains `adee286defed`; the shared index is empty. No landing is claimed.

Exact recovery custody:

- [Bundle manifest](/Volumes/VertigoDataTier/pact/ddm_rv16/retained/DDM_RV16_REVIEWED_FILES_MANIFEST.json)
- [Reviewed files bundle](/Volumes/VertigoDataTier/pact/ddm_rv16/retained/DDM_RV16_REVIEWED_FILES_v3.tar.gz), SHA-256 `acbb0060c851e8ba0d49225181eb68068ef3a128b3a2bdb1a41d301b3d5c9876`
- [Retained final memo](/Volumes/VertigoDataTier/pact/ddm_rv16/retained/DDM_RV16_MEMO.md), SHA-256 `0e60443e2bc2fe3fbea8b7efc4aa432003de69def2fde237a9b6a0d6edcee32a`

## NEXT_IF_RESUMED

- **QUEUED-WITH-OWNER** — disposition: land the exact reviewed rv16/rvf1/gt2 batch; owner: MAIN/operator in a Git-writable checkout; consumer store: Git `main`; fire trigger: compare against the retained manifest, rerun 342 tests, then invoke the SHA-pinned serializer once.
- **QUEUED-WITH-A-FIRE-ORDER** — disposition: run clean-review round 1; owner: MAIN review coordinator; consumer store: the next adversarial-review memo; fire trigger: the rv16 cure batch lands unchanged.
- **QUEUED-WITH-A-FIRE-ORDER** — disposition: execute and seal the clean port × rider receiver, then request one T4 row; owner: MAIN/rc1; consumer store: rc1 custody and the frontier pointer if admitted; fire trigger: real composed `inflate.sh` succeeds with semantic identity and a valid seal.
- **DEFERRED-SECOND-LANDING** — disposition: consider making the GT-lineage host gate strict; owner: next custody-gate landing; consumer store: `src/tac/preflight.py`; fire trigger: this batch lands and a fresh census remains zero.
- **QUEUED-WITH-OWNER** — disposition: classify rvf1’s five SSD authored-signal rows; owner: MAIN/rr8, MAIN/rr5, and gt2 custody; consumer store: the certification ledger or tracked source homes; fire trigger: byte comparison establishes generated identity or a unique authored delta.

## LIVE-HYPOTHESES

- The clean composed port × rider receiver will preserve semantics and retain most of the native speedup. The mechanisms are stage-disjoint and rider parsing restores the carrier, but the exact composed receiver has not run.
- The observed margin probably survives ordinary host variation, but one unaffected stage is not a variance population.
- A content-resolvable task-to-memo join can replace the remaining lexical association heuristic because the harness bridge, task store, and research index carry complementary keys.
- Certification and cache writing may need a shared lock if they can execute concurrently; only single-call failure ordering is currently proved.

## DEAD-ENDS

- Reading one instrumented T4 row as repeat-stable clean-shipping proof is closed.
- Treating the 1.2855 stage ratio as a variance bound is closed.
- Treating `free_corrector` alone as every-pair proof is closed; selection plus loop wiring is required.
- Treating a conflict-free merge or composed parse as full receiver execution is closed.
- Repeating rr5’s “10/10 identical” claim is closed.
- Letting an unrelated same-line memo launder a task ID is closed.
- Skipping an entire executable line because it contains a docstring is closed.
- Appending certification before cache invalidation is closed.

**Own-vehicle frontier: S 0.14839100138338618 @ 180,625 B `[contest-CUDA T4, n600]`, unmoved by ddm_rv16.**

