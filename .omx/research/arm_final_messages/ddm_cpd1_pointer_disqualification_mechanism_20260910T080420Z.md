Implemented and verified: **157 tests pass**, Ruff clean, two clean review passes. Offline replay selects move 40.

**Not landed:** sandbox Git writes were denied. Both serializer bundles verify; patches, refs, hashes, and commands are in the [handoff](/Users/adpena/Projects/pact/.omx/research/ddm_cpd1_20260910/LANDING_HANDOFF.json). Full findings: [memo](/Users/adpena/Projects/pact/.omx/research/ddm_cpd1_pointer_disqualification_20260910.md).

No scorer ran. Live eligibility remains unchanged. Submittable frontier: **move 40 — S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600]**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner MAIN; consumer `LANDING_HANDOFF.json`; trigger writable Git storage: land both bundles and reconcile cpd1 metadata.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner MAIN; consumers disqualification journal and canonical pointer; trigger landing harvest: execute the memo’s disqualification command and refresh.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner MAIN; consumers `preflight.py` and catalog; trigger zero live violations after that refresh: strict-flip Catalog 418 in the same batch.

**LIVE-HYPOTHESES:** MAIN’s live transition should reproduce the rehearsal: the same retained anchors and eligibility predicate selected move 40. Live execution remains untested.

**DEAD-ENDS:** Filtering after top-five selection hides valid successors; SHA-only bans conflate receivers; lane overrides cannot establish custody; one initial packet check permits concurrent invalidation. Regression tests now cover these failures.