# Codex findings — live DDM costate organ elevation 2

Date: 2026-07-23  
Lane: `lane_ddm_costate_organ_elevation2_20260723`  
Verdict: `LIVE_DDM_ADVISORY_REPOINT_COMPLETE_MAIN_REVIEW_REQUIRED`  
Axis: `[macOS-CPU frozen-scorer advisory]`  
Authority: `research_only=true`; `execution_allowed=false`; score claim false; pointer unmoved

## Outcome

The digest no longer consults the witness run when the required live DDM fleet is present. It reads
the latest schema-checked dv1, g3, g4, v19, and v19b receipts, verifies seven input hashes including
the dv1 summary and full g3 SSD atlas, and leaves e1/dv2 as explicit pending producers.

### Audit table

| Surface | Before | After | Disposition |
|---|---|---|---|
| SENSE source | witness run/trajectory | dv1+g3+g4+v19-family latest receipts | repaired |
| pair/site lambda | owed | 8 pair + 40 pair×stratum rows | repaired |
| primitive marginal chain | absent | measured bytes → described fraction → realized proxy | repaired; dv1/g4 realization still owed |
| D2 tolerance | absent/unscoped | pair closure fraction; v19b family-local joint survival | repaired |
| scheduler | witness lever queue | topology → free → coarse → Gauss-Southwell | repaired |
| staleness | run age/cache age | consumer-verified input hashes + horizons + queue | repaired |
| legacy 115 duties | rank-eligible | retained; current registry has 116; all dominated/stale | repaired without deletion |
| NCDE (`r2=.060`) | surfaced from witness epochs | invalid for discrete DDM recursion | retired honestly |
| factorized adjoint | witness snapshot | g3×v19 exact-factor backtest | re-backtested |
| pose sigma-min gate | witness finisher control | not applicable to DDM; degenerate fail-to-banked-R1 guard preserved | retired |
| maturity | `_dev` implicit | explicit six-condition `_prod` gate | repaired |

## Re-pointed digest sample

```text
DDM-LIVE reach=70.535% Road[semantic-cell] box=IN fleet=5/7 v19b=10 moves deltaS=-0.085020 [macOS-CPU frozen-scorer advisory]
DDM-lambda: pair/site=8/40 factorized-backtest rho=0.903 NDCG@4=0.927; shared candidate bytes=OWED_NOT_INVENTED
DDM-next: j_paint_dv1_persistent_ground rank=1 mode=UPPER_BOUND_ONLY_UINT8_REALIZABILITY_OWED GS=0
DDM-duty: J_paint > R6_rehearsal > DDM_iteration_curves; legacy=116 retained DOMINATED
DDM-staleness: hashes=7 rederive-queue=2 maturity=_dev actuation=NONE MAIN-review=REQUIRED
```

The 70.535% value is semantic-cell Road reach, not RGB realization or a score. The scheduler's
`GS=0` is intentional: J_paint is recommended as the measurement that identifies the missing
uint8-realizability factor, not as an admitted actuator.

## Lambda backtest

Scoped instance: v19 eight-pair exact receiver replay joined to the SHA-verified g3 full atlas.

- predictor: `g3_gap × g3_usable_support × v19_helpful/changed × 1/g3_allocated_bytes`;
- target: positive exact v19 Seg/Pose distortion closure;
- shared global candidate rate bytes: excluded, because per-pair allocation is null;
- pairs: 8; positive realized pairs: 5;
- Spearman rho: `0.9027075674773932`;
- NDCG@4: `0.9268617843989323`;
- verdict: `FACTORIZED_ADJOINT_VALID_ON_THIS_EIGHT_PAIR_BACKTEST`.

This is an exact-factor retrospective backtest. It does not claim prospective generalization.

## Scheduler demo

The regression fixture proves:

1. a dependency-blocked row never enters the frontier;
2. a byte-freeing row ranks before a higher-lambda spending row;
3. coarse spending ranks before fine spending;
4. at the same scale, `lambda=.4,radius=.5` ranks before
   `lambda=.2,radius=.5` (`G=.2` vs `.1`).

The live cycle then ranks J_paint first, R6 downstream, and the DDM iteration instrument as an
independent later block.

## Duty rerank

1. `J_paint` — close receiver/uint8 realization for the 283-byte persistent primitive.
2. `R6_rehearsal` — official receiver-closed exact rehearsal after J_paint.
3. `DDM_iteration_curves` — replace the invalid witness-era NCDE input domain.

The authority snapshot's 115 old owed rows are retained. Canonical state now contains 116, so the
receipt records `AT_LEAST_115_RETAINED`; all are marked `DOMINATED_STALE`, not deleted.

## Staleness and resume

Each cycle stores receipt/summary/atlas SHA-256 values, completed block IDs, and cycle index.
`--resume-from` refuses changed hashes. The current queue contains missing e1 and dv2 receipts.

Re-derive:

```text
python3 tools/ddm_costate_organ.py --write-receipt .omx/research/ddm_costate_organ_elevation2_<new-utc>/ddm_costate_organ_elevation2_receipt.json
```

## Stores consulted

- `.omx/research/ddm_dv1_description_vocabulary_n600_20260723T141407Z/`
- `.omx/research/ddm_g3_score_atlas_n600_20260722T204000Z/` plus its SHA-pinned SSD JSONL
- `.omx/research/ddm_g4_spatial_stationarity_n600_20260722T212138Z/`
- `.omx/research/ddm_v19_pure_priced_objective_20260723T041500Z/`
- `.omx/research/ddm_v19b_joint_remeasure_stack_20260723T051914Z/`
- FEED-603 recursion, gap-sweep, staleness, order, and train-least laws
- live lane/task/review state required by the operating contract

The quarantined 20260717 witness run was not consulted by the implementation or live cycle.

## Durable artifacts

- `src/tac/ddm_costate_law.py`
- `src/tac/ddm_costate_organ.py` (live implementation)
- `src/tac/witness_control/ddm_costate_organ.py` (compatibility re-export only)
- `tools/ddm_costate_organ.py`
- `tools/costate_digest.py`
- `src/tac/tests/test_ddm_costate_organ.py`
- `.omx/research/ddm_costate_organ_elevation2_20260723T154610Z/ddm_costate_organ_elevation2_receipt.json`
- this findings memo plus the DSL/equations/DAG triality

MAIN landing review is required. No merge, score promotion, launch, or provider dispatch is
authorized by this arm.
