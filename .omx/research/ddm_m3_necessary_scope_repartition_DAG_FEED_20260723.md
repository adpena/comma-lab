---
schema: ddm_m3_necessary_scope_repartition_dag_feed.v1
date_utc: 2026-07-23
lane_id: lane_ddm_m3_necessary_scope_repartition_20260723
research_only: true
execution_allowed: false
score_claim: false
verdict: NUMERIC_TRUE_SCOPE_NOT_CERTIFIABLE_CURRENT_CUSTODY
verdict_scope: "INSTANCE:c1/v14/v19b/G2/G2G2 evidence; no launch, family closure, score, or promotion verdict"
pointer: "0.1910828242 [contest-CPU Linux x86_64]"
pointer_moved: false
main_landing_review_required: true
---

# FEED-603-M3 — necessary-scope repartition

## Pointer delta

`0.1910828242 [contest-CPU Linux x86_64] -> unchanged`.

## DAG delta

```text
c1 pre-synergy assignment
  N_c1 = 2,377,273 Road/Undrivable/MyCar errors
  |
  +-> G2G2 multicoefficient solve
  |     six selected pairs / 20 Lane centerline coordinates / 0 of 6 admitted
  |     n600 REFUSED
  |       -> FULL-STRATUM SOLVE REACH = RED / UNKNOWN for all five classes
  |
  +-> G2 n600 operator atlas
  |     all 600 pairs
  |     INNER_ENCODER_JACOBIAN absent
  |     receiver delta d_seg absent
  |       -> byte-ranked operators are NOT solve-coverage rows
  |
  +-> v19b exact common master
  |     +3,884 shared bytes
  |     +73,945 Road/Undrivable/MyCar net flips
  |     +29,377 Lane/Movable net flips
  |     +0.080496721217 amplified gain
  |       -> PRE-SYNERGY PARTITION STALE (MEASURED)
  |       -> solve x correction conditional gain still RED / UNKNOWN
  |
  +-> v19c correction saturation
  |       -> RECEIPT ABSENT / PENDING
  |
  +-> frame incidence
        frame_1 = all Seg obligation
        frame_0 = exact Seg-null, Pose-only
        conditional frame_0 Pose preimage = OPEN / unmeasured on current vehicle
          R1 0.001610 at 7,195 B = comparator only
          |
          +-> TRUE N_366 = UNKNOWN in [0, 2,377,273]
          +-> numeric X percent = NOT CERTIFIABLE
          +-> #366 remains prepared but minimal-scope claim BLOCKED
```

## Consumer routing

- **#366 / J5-J6a:** retain apparatus and no config mutation; consume this feed as a
  minimal-scope blocker before operator fire.
- **c1 / R2:** replace independent pool arithmetic with one sequential exact chain and preserve
  per-class signed collateral.
- **G2/G2G2 successor:** first required output is a receiver-closed full-stratum
  multicoefficient solve row, not another scorer-free byte proxy.
- **v19c:** append saturation asymptote only from its exact receipt; do not infer it from v19b.
- **frame-0 M1 race:** fix frame 1 byte-identically and measure conditional Pose rate; R1 remains
  nontransferable.
- **continual learning:** treat `UNKNOWN` as a build/measurement gap, not zero reach or
  infeasibility.

## Triality

- DSL/data:
  `.omx/research/ddm_m3_necessary_scope_repartition_receipt_20260723.json`.
- DAG: this feed.
- Equations:
  `.omx/research/ddm_m3_necessary_scope_repartition_canonical_equations_20260723.md`.
- Re-deriver:
  `tools/audit_ddm_m3_necessary_scope_repartition.py`.
- Finding:
  `.omx/research/codex_premise_falsification_ddm_m3_necessary_scope_repartition_20260723_codex.md`.

MAIN must re-run the SHA-pinned helper, verify the historical G2G2 custody, confirm v19c remains
absent or replace the pending row with its exact receipt, and review the no-fake distinction
between 3.110496775% stale-partition evidence and an unavailable certified `X%`.

