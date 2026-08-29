Executed and landed as commit `ac5f43b341`.

Measured:

- Direct DALI-GT fit produced a deterministic, receiver-parseable HG1 container of **100,862 B**, **37,124 B below** the fixed-distortion cap.
- Exact archive repeat and parse-back passed.
- Native mismatch: **1,324,687 / 117,964,800 = 1.1229511%**. This is not `d_seg`.
- BO2’s refusal reproduced exactly: **209.06668593299145×**. Its corrected rank is second-best of five, not fourth-worst.
- Pair holdout degraded from 1.1261% to 1.6231%; spatial holdout was 1.1626% versus 1.1523% observed.

Not measured:

- No render/R/uint8 frozen-scorer realization was run because BZ2 did not own the scorer lane.
- Consequently, no capacity ceiling, cross-object law, exact score, or frontier movement is claimed.

Artifacts: [memo](/Users/adpena/Projects/pact/.omx/research/ddm_bz2_bornsmall_capacity_ceiling_20260829.md), [executable](/Users/adpena/Projects/pact/experiments/ddm_bz2_bornsmall_capacity_ceiling.py), [RESULT.json](/Volumes/APDataStore/pact/ddm_bz2_bornsmall_capacity_ceiling/RESULT.json), [FIRE_ORDER.json](/Volumes/APDataStore/pact/ddm_bz2_bornsmall_capacity_ceiling/FIRE_ORDER.json).

`[contest-CUDA T4 n600] own-vehicle frontier: GB1 — S=0.14811799921260607, archive=180,215 B; BZ2 did not move the pointer.`

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN scorer-lane router; consumer store: `/Volumes/APDataStore/pact/ddm_bz2_bornsmall_capacity_ceiling/realized_capacity_terminal/`; fire trigger: FCD3 explicitly releases the scorer lane, MAIN records a BZ2 claim, and all hashes revalidate. Run the resumable render/R/uint8 frozen-scorer terminal.
- **QUEUED-AFTER-TWO-TERMINALS** — owner: MAIN-designated BZ2/QBZ1 synthesis successor; consumer store: `/Volumes/APDataStore/pact/ddm_bz2_bornsmall_capacity_ceiling/cross_object_law/`; fire trigger: both objects publish scorer-realized ceilings. Fit the preregistered n≤2 law or publish `NO_PORTABLE_LAW`.

## LIVE-HYPOTHESES

- Direct DALI fitting may reduce realized damage versus BO2: it changed 74,650 categorical sites and removed 8,500 native GT mismatches, but scorer survival is untested.
- HG1 may have adequate within-frame capacity but weak pair-address generalization: spatial holdout barely degraded, whereas unseen-pair error increased materially.
- Address freedom or decoder degrees of freedom may predict capacity across BZ2 and QBZ1; this remains plausible but unidentifiable until both realized ceilings exist.

## DEAD-ENDS

- Treating BO2’s 209× result as BZ2’s capacity ceiling is closed: BO2 measured different fitted bytes and an advisory instrument.
- Calling the 1.1229511% categorical mismatch `d_seg` is closed: it precedes the renderer, R, uint8, and SegNet.
- Fitting a cross-object law from native BZ2/QBZ1 proxies is closed: neither is a scorer-realized ceiling.
- Reopening the BS4Y carrier solve as the capacity test is closed at its measured instance/formulation scope; it did not fit the full representation directly to DALI GT.