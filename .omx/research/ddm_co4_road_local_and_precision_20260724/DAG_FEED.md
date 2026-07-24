# DAG FEED — DDM CO4 Road-local ranker and precision closure

UTC: 2026-07-24T23:52:21Z  
Lane: `lane_ddm_co4_road_local_and_precision_20260724`  
Status: `FEED+ COMPLETE_LOCAL_ADVISORY`; MAIN landing review required.

## Executable evidence DAG

```text
sealed CO3 OOF receipt -------------------------------+
G3 hard-rank atlas ----------+                        |
EV1 N600 realized closure ---+--> CO4 held-out race --+--> selected OOF vector
G4 stationarity -------------+          |             |       |
PF2 exact bucket assignment -+          | Road gate   |       +--> innovations
MS4D 1,200 Fisher Grams ------+          +-- fail ---->+       +--> pair intervals
SN1 boundary/cell partition --+                                |
                                                               v
CO4 receipt --> campaign source registry --> one campaign state digest
                                      +--> digest consumer
                                      +--> dashboard consumer
                                      +--> duty-queue consumer
                                      +--> activation-nag consumer
```

The Road evaluation stratum is outcome-derived and stays on the evaluation
edge only. The expert-routing edge consumes the target-free G3 dominant
pre-outcome class-flip stratum.

## Typed outcomes

- Road-local race: `FORMULATION_SCOPED_GATE_FAILURE`.
  `g3_stratum_experts` measured Road NDCG@4 `0.1796465097835245`
  over 288 held-out Road pairs and global NDCG@4
  `0.8133546756293046`; the `0.60` Road bar failed. The exact sealed CO3 OOF
  predictions are retained.
- Precision: `COMPLETE`, with `15 DIRECT`, `585 PROPAGATED`, `0 UNRANKED`.
  Every propagated interval is wider than its nominal interval and carries
  the frozen propagation assumptions.
- #611 scorer-recursive construction:
  `BLOCKED_TYPED_COUNTED_SCORER_RECURSIVE_APPLICATION_OPERATOR_OWED`.
- MS2R immutable mismatch:
  `DIAGNOSED_ORACLE_INPUT_SUPERSESSION_NEW_STAGE_NAMESPACE_REQUIRED`.
  The old immutable stage is preserved.
- Pontryagin/Bellman: `AWAITING_J8F`.
- M34 per-state dual consistency:
  `AWAITING_J8F_M34_PER_STATE_DUALS`.

## Triality

- Typed/DSL leg: no flag or launch path was added. The additive v1 receipt
  schema, precision classes, DECIDE rows, and authority firewalls are the
  typed configuration surface.
- DAG leg: the CO4 receipt replaces the campaign source pointer and all four
  consumers inherit the same state digest.
- Equation leg: [EQUATIONS.md](EQUATIONS.md) owns the Road information block,
  propagation rule, design-effect penalty, and admission inequalities.

No provider, GPU, scorer replay, archive mutation, run mutation, task closure,
or frontier-pointer action occurred.

