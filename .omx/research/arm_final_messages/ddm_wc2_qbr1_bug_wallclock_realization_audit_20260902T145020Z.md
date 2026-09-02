Execution is partially complete but could not be committed: the local command host disappeared mid-run with `codex-code-mode-host: No such file or directory`.

Concrete results retained:

- Issued a `BURN-INVALIDATING` [fire alarm](/Volumes/APDataStore/pact/ddm_wc2_qbr1_bug_wallclock_realization_audit/WC2_FIRE_ALARM.json): the sealed EMA law targets 1% initial-shadow retention after 5,000 updates, while `warmup=True` executes a different law retaining approximately `1.838e-27`.
- Verified all 20 source pins by exact byte count and SHA-256.
- Found a second, boundary-fixable estimator mismatch: the dual controller ignores HT weights. On retained step-0 data, this changes Lane error from `0.11345` to `0.10459` and Movable from `0.00948` to `0.00792`; Movable therefore crosses from above to below its `0.009` bound.
- Staged an apply-checked [pinned-source cure](/Volumes/APDataStore/pact/ddm_wc2_qbr1_bug_wallclock_realization_audit/STAGED_QBR1_PINNED_CURES.patch), including versioned custody paths, constant EMA, HT-weighted constraints, stricter config validation, and history identity.
- Built the default-off [timing harness](/Users/adpena/Projects/pact/experiments/ddm_wc2_qbr1_stage_timing.py) with [tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_wc2_qbr1_stage_timing.py). Six focused tests passed.
- The retained CPU resume tree remained byte-identical: 168 files, 257,214,191 bytes, SHA-256 `2ee47df2…`. [Receipt](/Volumes/APDataStore/pact/ddm_wc2_qbr1_bug_wallclock_realization_audit/DEFAULT_OFF_IDENTITY.json).
- Artifact timing measured `1.99125 s/update`; milestone scoring blocks synchronously and adds approximately `16.09 s/milestone`. Projected six-cell duration is about `16.75 h`.
- The [memo](/Users/adpena/Projects/pact/.omx/research/ddm_wc2_qbr1_bug_wallclock_realization_audit_20260902.md) exists but remains incomplete and uncommitted.
- The live burn and all pinned sources were left untouched. No Metal, scorer-lane, Modal, or contest evaluation was run.

Own-vehicle frontier: `S=0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`; unchanged.

## NEXT_IF_RESUMED

- `HOLD_NEXT_CELLS` — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_wc2_qbr1_bug_wallclock_realization_audit/`; fire trigger: before any QBR1 cell launches after the current cell’s durable boundary.
- `BLOCKED-LOCAL-TOOLING` — owner: WC2 successor; consumer store: `.omx/research/ddm_wc2_qbr1_bug_wallclock_realization_audit_20260902.md`; fire trigger: command host restored; finish the realization rows and receipts, perform two review passes, and commit through the serializer.
- `QUEUED-WITH-FIRE-ORDER` — owner: MAIN; consumer store: the WC2 arm store and WX1 Route-1 inventory; fire trigger: staged cure applied and reviewed, versioned configs regenerated, corrected resume identity passes, and live Metal/scorer claims are bound; run one complete corrected cell with stage profiling.

## LIVE-HYPOTHESES

- Frozen scorers plus the fused realizer consume at least half of step time; they contain the largest full-resolution operations, but the queued synchronized profile must decide this.
- Caching the repeatedly reopened GT-cache members may save time because two ZIP directories are parsed every update; the CPU-proxy upper bound is about 2.25 minutes across six cells.
- Batching history durability at checkpoint cadence may save up to 4.34 minutes across six cells; the present path performs an `fsync` every update.
- Route 1 should combine native hard separation with through-R realized loss and explicit repair preservation; MST1 showed most manufactured error originates natively while R and uint8 are net repairers.

## DEAD-ENDS

- HT denominator bug: closed—the milestone estimator uses weights summing to 600 and divides by the population denominator 600.
- History-field conflation: closed—the native expected flip, realized expected flip, and exact within-class error have distinct producers; adjudication reads milestone `S_hat`, not history proxies.
- Tau-schedule drift: closed—the executable traverses exactly `0.15` at update 0 to `0.05` at update 4,999.
- Pair-stratification regression: closed—the two chunks each carry HT mass 300 and preserve the non-prefix NO2 design.
- Nonblocking milestone assumption: closed—the milestone call is synchronous and its artifact-clock excess is measurable.
- Tiny-window timing extrapolation: rejected; the prepared fire order requires a complete corrected cell.