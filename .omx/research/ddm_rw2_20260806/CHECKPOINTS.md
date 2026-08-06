# CHECKPOINTS - RW2

RW2 launched no long-running training, no GPU job, no exact replay, and no resumable staged run. Therefore there are no ML checkpoints to preserve or cold-store for this arm.

Durable RW2 evidence written under `.omx/research/ddm_rw2_20260806/`:

| artifact | role |
|---|---|
| `q3x_naive_round_n1.json` | Q3 naive-round bounded control |
| `q3x_dk1_cvp_n1_capped64.json` | Q3 DK1-CVP capped `max_blocks=64` control |
| `q3x_dk1_cvp_n1_full_coverage.json` | Q3 DK1-CVP full requested block-mask receipt |
| `dk1_stratified_ladder_n32_scorer_free.json` | DK1 stratified n32 scorer-free local ladder |
| `vo2_registry_rebuild/INSTRUMENT_REGISTRY.jsonl` | rebuilt VO2 instrument registry |
| `vo2_registry_rebuild/ROUND_SUMMARY.json` | registry denominator and ordering summary |
| `vo2_registry_rebuild/MANIFEST.sha256.json` | registry rebuild manifest |
| `registry_rebuild_stability.json` | unaffected-row hash stability check |
| `TRAINER_STOP_RECEIPT_INVENTORY.json` | trainer stop/checkpoint emit-surface inventory |
| `RW2_ELEMENT_REGRADES.jsonl` | RW2 element regrade delta ledger |
| `NAIVE_COUNT.json` | R2 23-row NAIVE count update |

All listed artifacts are durable repo-path evidence, not `/tmp` evidence. None is a score row or promotion artifact.
