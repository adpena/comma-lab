# DAG feed: costate organ v3 rank sharpening

UTC: 2026-07-21T02:50:10Z  
`research_only=true` · `actuation=NONE` · `promotion_eligible=false`

## Dependency and decision DAG

```text
v2 fixed 24-row receipt (SHA guarded)
  + r1b7 498-site stage autopsy (SHA guarded)
        -> Jeffreys-smoothed graded receiver survival
  + witness_measured_reverse_waterfill_v1
        -> shared-opportunity-pool KKT marginal
  + ema_decay_run_geometry_v1 executable DSL LawRef
        -> target-only EMA de-lag + precision weights
        -> staged metrics + paired 10k bootstrap
        -> NO ADMISSION / NO PROMOTION
        -> typed identity corpus snapshot (24 zero-byte rows)

future M1 byte-close receipt
  -> source SHA + realized block + nonzero byte delta + byte-close checks
  -> first byte-paying corpus row, or fail-closed NOT_EMITTED
```

## Triality legs

- DSL: consumes the existing typed `EmaDecayCalibrated` LawRef; no new trainer
  flag or live configuration was introduced.
- DAG: the v3 organ is a read-only observer downstream of custodied receipts;
  only a fully realized M1 proof may extend the append-only corpus.
- Equations: `costate_v3_rank_sharpen_composition_v1`, graded survival,
  reverse-waterfill pool marginal, EMA inverse response, weighted Spearman, and
  NDCG@8 are stated in the companion equations note.

## Unified-stack wire-in

1. Sensitivity map: no mutation; consumes already-measured receipt fields.
2. Pareto constraint: decision NDCG@8 is the explicit benefit-order constraint;
   its adverse direction blocks admission.
3. Bit allocator: pool marginal reuses
   `witness_measured_reverse_waterfill_v1`; no allocation actuation.
4. Cathedral/autopilot: no dispatch hook because this is advisory-only and not
   promotion eligible; the machine receipt is the reusable consumer surface.
5. Continual learning: the typed JSONL corpus is the append-only empirical
   anchor; no posterior is updated from identity/zero-byte seed rows.
6. Probe disambiguator: ordinary Spearman, weighted Spearman, top8 precision,
   and NDCG@8 are all emitted so disagreement blocks silent arbitration.

## Pointer delta

Pointer unchanged. No contest-axis score, archive, provider call, or live run.
MAIN landing review is required.
