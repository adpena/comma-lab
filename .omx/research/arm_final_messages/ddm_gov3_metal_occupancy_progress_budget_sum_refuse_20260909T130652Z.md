Implemented and landed all chartered governor items. Full evidence is in [ddm_gov3 memo](/Users/adpena/Projects/pact/.omx/research/ddm_gov3_metal_occupancy_progress_budget_sum_refuse_20260908.md).

- Metal jobs now carry real occupancy/footprint evidence and obey the one-occupant rule until 3 distinct dual-Metal FIT windows exist.
- Timeouts now follow `completed_steps` progress, retaining the legacy timeout when no heartbeat exists and a 3× far ceiling.
- Projections over the adaptive ceiling by more than 5 GiB hard-refuse unless given a substantive explicit rationale.
- Historical replays: 16:12 REFUSE; 22:30 ADMIT; 21:05 REFUSE.
- Seed denominator: 3 profile rows, 0 qualifying dual-Metal FIT windows.
- Validation: 138 + 31 = 169 tests passed; targeted Ruff and canonical-equation preflight passed.
- ITEMs 3, 5, and 6 serve as `completed`, `test_status=green`. Unrelated historical ledger warnings remain pre-existing.
- No scorer, Modal dispatch, governed process, payload, `upstream/`, sj1, or submission surface was touched. No score movement is claimed.

Landed commits: `29f76e36d`, `9a943ac99`, `0ff04fcd2`, `45d527141`, and `60deb570d`. All used the serializer and contain no attribution trailer.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER — owner: MAIN; consumer store: `.omx/state/metal_admission_table.jsonl`; fire trigger: an operator-authorized dual-Metal window whose RAM projection fits, leaves two logical cores per Metal cell, and has no scorer/single-flight collision.
- QUEUED-WITH-A-FIRE-ORDER — owner: ddm_gov3 successor; consumer store: `src/tac/canonical_equations/` and its registry; fire trigger: at least three distinct qualifying dual-Metal FIT windows exist.

## LIVE-HYPOTHESES

- Smaller-footprint Metal families may safely coexist because the cl3 and md3 profiles leave materially different headroom; this remains untested at 0/3 qualifying windows.
- Progress-derived stall budgets may preserve healthy CPU-starved Metal runs because `completed_steps` distinguishes slow progress from a hang; mechanism tests pass, but no live long-run distribution was measured.

## DEAD-ENDS

- SIGSTOP remains retired: four measured pauses helped zero times and made the stopped process a swap victim.
- Renicing CPU arms remains retired as the timeout cure: +10 niceness did not improve measured Metal-host throughput.
- Compressor delta remains retired for Metal attribution: the 45-second measurement used the wrong instrument.
- Existing N=2 concurrency observations cannot admit two Metal occupants: the ratios straddle baseline and their spread exceeds the effect.

sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]