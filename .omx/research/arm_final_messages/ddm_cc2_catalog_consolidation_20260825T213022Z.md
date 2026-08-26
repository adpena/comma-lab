Pointer unchanged. The full review is committed as [ddm_cc2_catalog_consolidation_20260825.md](/Users/adpena/Projects/pact/.omx/research/ddm_cc2_catalog_consolidation_20260825.md) in commit `77167f32f5`.

Measured:

- 285 numeric rows, 285 unique IDs, 284 unique names.
- Six rows above #400: #401, #402, #404, #406, #407, #408.
- All six have a valid `CLAUDE.md:2` quota waiver.
- Catalog #176: 268 strict callsites; 184 cataloged, 76 legacy-allowlisted, eight waived, zero uncovered.
- No retirement candidates. #402 and #404 currently report findings; #406 and #408 have documented post-landing catches.
- Recommendation: consolidate five overage rows into three existing families; KEEP #406 until its missing #332/#351 host identity is resolved.
- Found one duplicate function identity (#203/#224) and one wrong cross-reference: #407 cites #168, but the contest-score gate is #391.

Cumulative fire histories without durable evidence remain `NOT-MEASURED`. No scorer, evaluator, Modal job, payload, catalog edit, preflight edit, or upstream mutation occurred. The worktree is clean. Own-vehicle frontier remains **gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]**.

## NEXT_IF_RESUMED

- **QUEUED FOR OPERATOR ADJUDICATION** — owner: MAIN + operator; consumer store: `docs/meta_bug_class_catalog.md` and `src/tac/preflight.py`; fire trigger: approval of the #397 confound-registry replacement.
- **QUEUED FOR OPERATOR ADJUDICATION** — owner: MAIN + operator; consumer store: `docs/meta_bug_class_catalog.md` and `src/tac/preflight.py`; fire trigger: approval of the #389 process-lifecycle replacement.
- **QUEUED FOR OPERATOR ADJUDICATION** — owner: MAIN + operator; consumer store: `docs/meta_bug_class_catalog.md`, `src/tac/contest_score.py`, and `src/tac/preflight.py`; fire trigger: approval of the #391 score-authority extension.
- **QUEUED FOR IDENTITY REPAIR** — owner: MAIN + catalog custodian; consumer store: `docs/meta_bug_class_catalog.md`; fire trigger: operator selection of a valid host for #406 or confirmation that it remains separately waived.

## LIVE-HYPOTHESES

- A severity-aware #397 registry can absorb #401/#402/#404 safely because they already execute through `CONFOUND_GATES`.
- One #389 umbrella can cover detached and timed-wrapper process failures because both enforce the same process-group lifecycle invariant.
- The absent #332/#351 rows may be extraction drift because live comments and the catalog header still reference them as established identities.

## DEAD-ENDS

- Retirement is closed: none satisfies both zero fires and structural extinction.
- Treating #513/#417 or task #812/#842 as claimed gates is closed: they are references, not numeric authority rows.
- Consolidating #406 into #332/#351 now is closed: neither is currently a numbered catalog host.
- Using #168 for #407 is closed: #168 is the Assign/AnnAssign AST gate; #391 is the contest-score authority.