# DDM AT1 scorer analytic atlas — DAG / FEED handoff

**Date:** 2026-07-23  
**Authority:** research-only; no execution, score, actuation, promotion, or
pointer movement. MAIN landing review required.

## Executable dependency DAG

```text
immutable upstream evaluate.py/modules.py/frame_utils.py
  + frozen Pose/Seg checkpoints + public names + uv.lock
                              |
                              v
       scorer_module_inventory_receipt
       body f02031d6025f869... BLOCKED on 7-version drift
              | exact immutable/checkpoint facts
              | observed library source only (not locked authority)
              v
      typed factor builders + source-hash freshness
          |              |                 |
          |              |                 +--> #580 spatial certificate
          |              |                      refuses fake DFT-null bands
          |              |
          |              +--> n600 gaze + per-layer Jacobian shards
          |                   BLOCKED until exact locked source + execution
          |
          +--> closed-form factor shards
               BLOCKED until exact locked source

SDWL1 receipt --------------------+
                                  +--> loss-accounted coordinate bridge
E2 runtime receipt ---------------+    (not invertible; no cross-price)

g3 n600 atlas + exact v19 rows
              |
              v
  build_ddm_lambda_bundle  [SINGLE producer]
  content 90ebdbb9af557... 8 pair / 40 site / 592 inert
              |
              v
  ddm_costate_organ [consumer/controller]
  LIVE_DDM_ADVISORY / actuation NONE

all Phase-0 rows + divergence table
              |
              v
  atlas_receipt body a446465df33af...
              |
       MAIN adversarial landing review
```

## Triality

| Leg | Landed object | Status |
|---|---|---|
| DSL / typed state | `scorer_analytic_atlas.py`, `scorer_module_inventory.py`; factor/checkpoint/bridge/λ schemas | LANDED |
| DAG | closure → inventory gate → factors → gaze/J composition → axes/pools → λ producer → organ | LANDED here and in atlas receipt |
| equations | BN affine; SE gate; kernel DFT; `λ_k=∂S/∂z_k`; `dS/dx=Jᵀ...λ`; canonical contest score | LANDED as typed operations / receipt laws |

## FEED rows

| Feed ID | Producer | Consumer | Gate / payload |
|---|---|---|---|
| `FEED-AT1-LOCKED-SOURCE` | exact `uv.lock` environment materializer | network closed-form materializer | require zero version drift and hash-stamped source-law rows |
| `FEED-AT1-GAZE-N600` | Pose 6-VJP + Seg rank-4 materializer | atlas manifest / relay solver | exact pairs 0..599, preserved stage shards, both networks complete |
| `FEED-AT1-JACOBIAN` | per-layer GT-trajectory factorizer | relay-depth selector / axis composition | exact source/tensor hashes, composition shape closure |
| `FEED-AT1-SN1-RESIDUAL` | sn1 measured factor receipt | validation harness | exact factor ID/pair/tensor/source join; residual is a finding |
| `FEED-AT1-DR2B` | SDWL1/E2 bridge successor | U1/mode/ξ consumers | price transfer only on an exact invertible row; current rows forbid transfer |
| `FEED-AT1-LAMBDA` | `build_ddm_lambda_bundle` | costate organ | current 8 pair/40 site exact-v19 rows; 592 missing rows stay inert |
| `FEED-AT1-RNULL` | exact global resize diagonalization successor | DR2b frequency admission | current #580 spatial result is insufficient; no exact dead band admitted |

## Exact blocker routing

`BLOCKED_LOCKED_LIBRARY_SOURCE_NOT_MATERIALIZED` is the first open gate. Its
scope is closed-form binding to third-party evaluator source. It does not
invalidate:

- immutable evaluator composition laws;
- exact checkpoint tensor names/shapes/dtypes;
- the typed atlas/freshness/resume contracts;
- the SDWL1↔E2 loss ledger;
- the current eight-pair atlas-produced λ backtest;
- the formulation families waiting behind the gate.

After exact locked-source materialization, the next gate is n600 gaze/Jacobian
materialization. No consumer may skip directly from this Phase-0 branch to a
promotion or scorer claim.

