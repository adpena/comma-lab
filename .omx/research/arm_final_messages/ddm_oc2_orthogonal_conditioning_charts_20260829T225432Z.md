The charter is complete and committed as `ed5f20b6de`.

Verdict: `FAMILY_DRAINED_ON_LB1__MISS_RANK8_BANKED_2B__NO_FIRE`.

| Chart | Predicted overlap | Measured overlap | Real marginal vs lb1 |
|---|---:|---:|---:|
| Temporal transition | 20% | 991.624% | Not run; nonorthogonal screen closure |
| Causal class edge | 35% | Undefined; negative solo gain | Not run; screen closure |
| Miss-rank8 | 15% | 0% | **2 B saved** |

Miss-rank8 produced a retained 180,081 B archive, SHA `67fb0b8e…80cab`, but missed the preregistered 30 B bar by 28 B. It is banked, not admitted. Therefore no native decode, seal, scorer load, lane claim, Modal dispatch, or authority evaluation occurred.

Re-derived remaining conditioning proxy: **1,877.126399 B over 117,964,800 positions**, after 285 physical bytes collected by gb1, jt21, and lb1. jt22’s off-branch 1 B was not added.

Artifacts:

- [Evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_oc2_orthogonal_conditioning_charts_20260829.md)
- [Experiment implementation](/Users/adpena/Projects/pact/experiments/ddm_oc2_orthogonal_conditioning_charts.py)
- Retained store: `/Volumes/APDataStore/pact/ddm_oc2_orthogonal_conditioning_charts/`
- 346 retained files totaling 25,130,549 B
- Two genuine Python review passes completed; Ruff, compilation, refusal gates, artifact integrity, and claim-boundary checks passed.
- Unrelated worktree changes and the shared index were preserved.

Own-vehicle frontier remains **S=0.14811799921260607 @ 180,215 B [contest-CUDA n600, gb1]**. The pointer did not move.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER / BANKED:** owner `MAIN lossless-bank composer`; consumer store `/Volumes/APDataStore/pact/ddm_oc2_orthogonal_conditioning_charts/`; fire trigger: when MAIN assembles the next independently admitted, lb1-compatible lossless candidate, include miss-rank8 in its single real n600 joint re-encode and proceed to generation-22 native identity, sealing, lane claim, and authority fire only if the combined marginal clears the then-live bar.

## LIVE-HYPOTHESES

- Miss-rank8 may retain its 2 B inside a future independently admitted lossless composition because it saved exactly 2 B both before and after patch192, implying measured 0% overlap. It is plausible only as a rider.
- A genuinely new representation may reopen conditioning on a future body because mi1 still exposes substantial fitted-model headroom while free adaptive marginals on this body have collapsed.

## DEAD-ENDS

- Temporal-transition conditioning is closed on this body: 991.624% overlap and −105.099 B conditional held-out marginal.
- Causal-edge conditioning is closed: negative solo gain and only +21.978 B conditional ledger signal.
- Miss-rank8 is closed as a solo-fire path: only 2 B from the real n600 encode. Do not retune its bins.
- A new scan-phase relabel is duplicate structure because gb1 already ships `groupbin8_surprise`.
- Additive bank arithmetic, coder/ZIP reraces, transmitted patch tables, paid conditioning, the incorrect patch index, and generation-20 native reuse remain closed.