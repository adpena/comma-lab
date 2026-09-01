DDS1 is complete and committed as `22498c03b3`.

Typed verdict: `PARTIAL` (`SCREEN`, seeded random n=120).

- Best zero-payload surrogate: HPAC’s current argmax plus strictly causal predicted-boundary state.
- It captures 10.8575% of GF1’s wrong-half predictive gain.
- Exact GF1 tuple agreement is only 6.2935% on wrong sites, so original candidates 1+2 are not derived.
- Original packet formulation still must replace 89,620 B; the weakened M-only variant removes the packet but must still replace 42,017 B.
- These are pool requirements, not measured savings. No real coder or scorer ran.

Artifacts:

- [Verdict memo](/Users/adpena/Projects/pact/.omx/research/ddm_dds1_decoder_derivable_verdict_20260901.md)
- [Retained runner](/Users/adpena/Projects/pact/experiments/ddm_dds1_decoder_derivable_born_stats.py)
- Retained v2 evidence: `/Volumes/VertigoDataTier/pact/ddm_dds1/measurement_v2/`
- `RESULT.json`: `057b073e…aa54d8`
- Manifest: 27 files, 424,945,477 B

Verification passed compilation, Ruff, payload-retention preflight, two review-tracker passes, causal future-peeking controls, receiver boundary parity, all retained hashes, serializer checks, and post-commit HEAD verification.

Frontier remains AFR1 `S=0.14797617125559104 @ 180,002 B` `[contest-CUDA T4 n600]`, archive `cbb8d928…405bf25`.

## NEXT_IF_RESUMED

- Disposition: `FOLDED-INTO-ACTIVE-OWNER`; owner: task #1374 SCMDL `X,G,M`; consumer store: `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`; fire trigger: after `GATE-1-PASSED`, deduplicate the M-only variant against shipped mixer state and admit it to one joint price only if distinct.
- Disposition: `DEFERRED-FULL-POPULATION-CONFIRMATION`; owner: MAIN-selected #1374 successor; consumer store: `/Volumes/VertigoDataTier/pact/ddm_dds1/measurement_v2/`; fire trigger: run frozen n600 confirmation only if the M-only variant survives deduplication and its near-threshold screen affects a pricing decision.

## LIVE-HYPOTHESES

- The M-argmax/predicted-boundary route may retain a small independent marginal on AFR1 because it alone crossed 10%; OC2’s prior negative makes a substantial win unlikely.
- Full n600 measurement could move the 10.8575% estimate across the 10% boundary.
- XOV1 candidate 3 and SFP1’s changed-field proposals remain live because they do not require GF1’s tuple.

## DEAD-ENDS

- Exact temporal-prefix HG1 refitting closed at 0% overlap with negative wrong-half gain.
- Latest-neighbour boundary, previous-frame, and temporal-mode routes closed at 2.5529%, 0.8823%, and 4.1328%.
- Exact GF1 tuple derivation was not found across 5/5 formulations and 46,000 wrong sites.
- The initial 50-cell and 2,500-cell analyses are superseded due to an omitted model-class dimension and complexity-induced overlap above 100%.
- Converting screen bits into bytes, scaling the packet by overlap, or adding banked coder credits is closed without a real joint encode.