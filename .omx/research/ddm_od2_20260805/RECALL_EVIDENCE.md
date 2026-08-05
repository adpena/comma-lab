# OD2 recall evidence - 2026-08-05

Status: `RECALL_COMPLETE_FOR_OD2_STAGE12`.

Axis: `[macOS-CPU advisory / document-and-ledger recall]`.
`score_claim=false`, `promotion_eligible=false`, `full_n600_scorer_job=false`.

## Governing Reads

| source | use |
|---|---|
| `.omx/tmp/codex_runs/od2_prompt.md` | OD2 charter and deliverables |
| `.omx/tmp/codex_runs/_common_contract.md` | serializer, scorer-slot, recall, protected-file, and denominator contract |
| `PROGRAM.md` | mutation boundaries |
| `CLAUDE.md` and `AGENTS.md` | no-fake, exact-score, storage, and checkpoint discipline |
| `docs/operating_manual_craft_handoff.md` | artifact-derived answer-first handoff discipline |
| `.omx/state/main_hot_state.md` | current own-vehicle frontier, live scorer ownership, OD1/pe2 context |
| `.omx/state/canonical_frontier_pointer.json` | borrowed contest pointer and PR130 display bar |
| `upstream/evaluate.py` | score formula and byte denominator |
| `.omx/research/operator_directive_per_edge_optimality_criteria_20260805.md` | Addendum 4 ordering law: seg first, joint-descent pose recovery after |
| `.omx/research/ddm_od1_20260805/*` | OD1 Stage 1+2 contract, blockers, and fire order |

## Searches

Queries and scopes used:

- `rg --files .omx/research/ddm_od1_20260805`
- `rg --files experiments tools src/tac .omx/research/ddm_sq2_20260804 .omx/research/ddm_tj1_20260805 | rg '(sq2|tj1|js1|od1|phase|c_prime|cprime|carriage|frame0|solved|continuation|subset_selection|stratified)'`
- `rg --files .omx/research | rg 'operator_directive|addendum|per_edge|optimality|ordering|directive'`
- Memory quick pass over `/Users/adpena/.codex/memories/MEMORY.md` for current Pact pointer/hot-state and queue discipline.
- Target directory check for existing OD2 artifacts before writing new receipts.

## Found Beyond The Charter Seeds

1. `CHARTER_ADDENDUM_ST1_TARGETER.md` was already tracked in the OD2 directory. It records st1 and st2 targeters, with st2 superior at 3,602 counted B and 99.84% recall. Plan change: record as a next-resume solve-budget prior only. This OD2 runner did not allocate per-cell budget, and the addendum explicitly forbids using the targeter as a paint mask.

2. The old SQ1 n32 selection was stratified systematic, not stratified-random. Plan change: generate a fresh OD2 n32 stratified-random set with seed `20260805`, 10 temporal blocks, and governing ratio receipts for seg and pose.

3. `experiments/ddm_js1_staging_discriminator.py` was dropping the SQ1 solver diagnostics by unpacking only three return values from a solver that now returns diagnostics as a fourth value. Plan change: patch JS1 to preserve Stage-1 stop reason, trajectory stop payload, steps run, and best step in every row.

4. Current `main_hot_state.md` reports the own-vehicle frontier as `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`, not JS1's older `0.7910689` constants. Plan change: OD2 aggregate recomputes the projection against the current own-vehicle line and keeps the legacy JS1 aggregate as a secondary compatibility artifact only.

5. The OD1 blocker ledger requires closing `OD1_BLOCKER_SEG_BASE_CAP_BOUND` and `OD1_BLOCKER_FRAME0_CARRIAGE_POPULATION`. Result change: OD2 closes the frame0 carriage population gate on n32, but does not close the seg-base cap-bound blocker because 29/32 Stage-1 rows remain `iteration_cap_best_at_cap`.

## Scoped Negatives

- Did not find, in the searched OD1/OD2 target package and current hot state, a prior OD2 n32 staged-composition receipt on a seeded stratified-random pair set.
- Did not find, in the completed OD2 receipt, Stage-1 terminality sufficient to close the cap-bound blocker.
- Did not run full n600, contest-CPU, contest-CUDA, or any receiver-closed archive build in this arm.

Own-vehicle frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
