Executed and landed as commit `873947c665`.

Key findings:

- Top 1% of positions carry **96.3238%** of modeled bits; top 0.1% carries **52.9507%**. Gini: **0.995159**.
- Lane is 0.5856% of positions but carries **33.5598%** of bits at **57.31×** mean bpp.
- The last 100 frames rise to **0.008902 bpp**, versus the 300–399 minimum of 0.006816.
- 90.70% of MS9 Seg-error positions fall within the top 1%, but all Seg errors account for only **5.2655%** of rate mass.
- The primary sum is 910,209.280609 bits, reconciling to the 910,216-bit stream with **6.719391 bits** of bounded coder overhead.
- The instrumented decoded field exactly reproduces TO2’s SHA.

Artifacts:

- [Findings memo](/Users/adpena/Projects/pact/.omx/research/ddm_bl1_per_position_bit_allocation_20260822.md)
- [Instrument](/Users/adpena/Projects/pact/experiments/ddm_bl1_per_position_bit_allocation.py)
- [RESULT.json](/Volumes/VertigoDataTier/pact/ddm_bl1_per_position_bit_allocation/measurement_v1/RESULT.json)
- [MANIFEST.json](/Volumes/VertigoDataTier/pact/ddm_bl1_per_position_bit_allocation/measurement_v1/MANIFEST.json)
- Primary float64 field: 943,718,400 B, SHA `99d7833d…e2c86`.

All 191 retained artifacts were rehashed. Thirty resumable stages and a resume-only deterministic repeat passed. No scorer, Modal, Metal, receiver mutation, or score claim occurred. DX2 remains `S=0.14821987563243377 @ 180,368 B`.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — Owner: MAIN. Consumer store: `/Volumes/VertigoDataTier/pact/<new-claimed-rate-lane>/RESULT.json`. Fire trigger: a non-duplicate charter consumes the retained field and exact top-1% mask, counts the complete receiver-closed payload, includes LQ1 collateral and MS9 overlap columns, and beats 113,777 B locally before requesting a scorer.

## LIVE-HYPOTHESES

- The retained top-1% set can support targeted rate work because it contains 109,593.569 B of gross incumbent cost.
- Lane remains a high-value but collateral-sensitive target because its bpp is 57.31× mean while previous Lane interventions caused broad spill.
- The MS9 overlap may offer joint rate/distortion leverage because 90.70% of Seg errors lie in the expensive tail, although no causal intervention has tested this.
- The late-clip rise may reflect a distinct predictor regime because it appears in the shipped model as well as EF1’s generic measurements.

## DEAD-ENDS

- Diffuse-mass interpretation: closed; top 1% carries 96.3238%, far above the 25% falsifier.
- “Current Seg errors explain the rate tail”: closed; they carry only 5.2655% of bits.
- “Lane alone is the complete target”: closed; Lane carries 33.5598%, not the majority, and collateral remains binding.
- Reusing DC1S as the requested full field: closed; it retained aggregates and selected non-MAP rows, not all 117,964,800 costs.
- Re-running flat orderings, generic estimators, named summaries, or isolated coder races: closed by TO2, EF1, CX3, and RB1 on this exact object.

