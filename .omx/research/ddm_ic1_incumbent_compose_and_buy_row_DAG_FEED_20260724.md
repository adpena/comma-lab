---
schema: ddm_ic1_incumbent_compose_and_buy_row_dag_feed.v1
date_utc: 2026-07-24
lane_id: lane_ddm_ic1_incumbent_compose_and_buy_row_20260724
research_only: true
execution_allowed: false
score_claim: false
main_review_required: true
---

# IC1 incumbent compose-and-buy DAG feed

## Executed path

`V19C nested state + W_joint warm payload`
→ strict E5 parse-back identity
→ explicit `DDMIC1RuntimeExporterConfigV1`
→ scorer-derived PA1 transform on Seg-free frame 0 only
→ preserved base and composed receiver checkpoints
→ Brotli-Q11 packet `aba831de…9d9` at `131,582 B`
→ locked upstream `evaluate.sh` PASS
→ independent frozen-scorer batch32 PASS
→ append-only incumbent scoreboard row
→ contest-CPU Modal bundle `PREPARED_NOT_DISPATCHED`.

No historical score delta was added. The raw composed surface was freshly
decoded and both scorers were rerun.

## Excluded conflict edges

- `MC1 → IC1`: excluded because its best measured same-pool static arm worsens
  joint advisory objective by `+4.850055382139988`.
- `E2 → W_joint`: excluded because E2 is a separate receiver-closed endpoint,
  not an additive pose pool, and has no admitted typed W_joint composition.
- `DM2 → W_joint`: excluded because its freshly composed aggregate costs
  `+2.350835831188035` score units and its one favorable independent row is not
  additive authority.
- `V19B → IC1`: superseded by V19C inside the chosen state.

## Construction-provenance edge

The exact W_joint point is retained as a `[naive-menu upper bound]`; it does not
close paint, placement, exception, or correction families. PA1 is the only new
composed operation and is scorer-derived from the frozen PoseNet first stem and
BN moments, then restricted by the exact evaluator factorization to Seg-free
frame 0. The replacement edge for W_joint is corrected-inner-Jacobian proposal
generation with exact resize-footprint and stride-2 stem-lattice support,
Fisher-margin ranking, shearlet support, and explicit Pose-null/price custody.

## A2 price edge

The measured graph has two same-parent nodes:

1. W_joint at `131,294 B`, `S_advisory=26.27522494187002`;
2. W_joint→PA1 at `131,582 B`, `S_advisory=23.66179213623354`.

This establishes one local secant. It does not establish a global lower convex
envelope or KKT dual over omitted scorer-recursive streams. A2-06, A2-15, and
A2-17 therefore remain suspended and all per-stream prices remain `NULL`.

## Dispatch edge

MAIN may, after review, execute the one command stored in
`ddm_ic1_incumbent_modal_contest_cpu_bundle_20260724.json`. The wrapper owns the
atomic lane claim and Modal call-id ledger. This delegated lane did not claim,
dispatch, or infer a contest-axis score.

## Pointer delta

`0.1910828242 [contest-CPU] → 0.1910828242 [contest-CPU]` (`UNMOVED`).
