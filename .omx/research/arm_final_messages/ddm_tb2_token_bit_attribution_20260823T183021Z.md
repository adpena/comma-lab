TB2 completed and landed in commit `cb8b64fcc6`.

Measured on all 117,964,800 symbols:

- Gini: **0.9951593787**
- Top 1% bit mass: **96.323842%**
- Top 10% bit mass: **99.900879%**
- Physical stream: **910,216 bits**
- Attributed cost: **910,209.280609 bits**
- Explicit arithmetic termination/padding residual: **6.719391 bits**
- Manufactured-error overlap in top 1%: **90.96× independence by count, 257.48× by bit mass**

The full cost and aligned RR4, FX2, and MA1 context fields are retained on APDataStore. All **322 artifacts / 3.725 GB** rehashed successfully.

Artifacts:

- [Research memo](/Users/adpena/Projects/pact/.omx/research/ddm_tb2_token_bit_attribution_20260823.md)
- [Instrument](/Users/adpena/Projects/pact/experiments/ddm_tb2_token_bit_attribution.py)
- [RESULT.json](/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/RESULT.json)
- [CB2 allocation handoff](/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/CB2_ALLOCATION_HANDOFF.json)

No scorer, candidate archive, d_seg, d_pose, or score was measured. The frontier did not move.

## NEXT_IF_RESUMED

- `FIRED` — disposition: `CB2_ALLOCATION_HANDOFF_FIRED`; owner: `MAIN-designated CB2 task-weighted K2048 dictionary successor`; consumer store: `/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/reactivated_task_weighted_refit/`; fire trigger: TB2 verification and all handoff hashes match, which is now **MET**. Fold before scoring unless it produces a receiver-closed archive no larger than 137,986 B.

## LIVE-HYPOTHESES

- A task/transition-weighted K2048 refit may avoid RC1’s task collapse because it can focus on the 1% of positions carrying 96.32% of cost.
- Causal-cell weighting may outperform class-only weighting without stored addresses: the top 100 MA1 and FX2 cells carry 98.34% and 82.10% of cost.
- Middle-band allocation may help because rows 144–287 contain 96.83% of cost and 96.64% of manufactured bit mass.

## DEAD-ENDS

- The claim that nobody had measured per-symbol token cost is closed: BL1 already retained the identical field.
- The manufactured-error join is not new: WJ1 already measured it; TB2 reproduced it exactly.
- MST1’s `+22,321` net delta cannot be used as a position set; only its 28,602-position gross mask is joinable.
- High modeled cost is not removable-byte credit. No top-set, class, band, or context mass is promoted into projected savings.
- Reordering, generic coder substitution, fixed-model Lane coarsening, and existing context-summary tables remain closed in their measured scopes.

**Own-vehicle frontier: UNMOVED at dx2 — S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`.**