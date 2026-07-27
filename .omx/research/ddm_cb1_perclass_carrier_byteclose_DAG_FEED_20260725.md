---
title: DDM CB1 per-class carrier byte-close DAG feed
date_utc: 2026-07-25T22:13:59Z
lane_id: ddm_cb1_perclass_carrier_byteclose
research_only: true
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
exact_eval: false
pointer_moved: false
main_landing_review_required: true
verdict: CB1_HAS_STRICT_NEGATIVE_JOINT_ROW
verdict_scope: "INSTANCE per exact carrier payload on merged RG4 source-local PC1 base"
---

# Executable feed

```text
merged RG4 source-local PC1 archive
  B=139685
  SHA=d86710793f776cb28144d9c7d817a3c9965e8160a1fc001c87deee206f9ccbf6
  |
  +-> fresh control CB1/E4 wrap
  |     -> deterministic compile x2
  |     -> strict manifest + state/rg4.ddr4 closure
  |     -> emitted runtime
  |     -> RG4 + parent parse/reemit identity
  |     -> 19 resumable render stages
  |     -> raw SHA 9b41b650...
  |     -> n600 frozen scorer baseline
  |           d_seg=.061912604437934025
  |           d_pose=31.281041046492344
  |           B=131301
  |
  +-> MyCar static carrier
  |     -> consume MC1 static-majority support (139 B)
  |     -> self-detect unique spatial/static class
  |     -> detected canonical class 4, never fixed by implementation
  |     -> target-bearing static rule (146 B)
  |     -> same CB1/E4 + runtime + n600 chain
  |     -> delta B=+319
  |     -> delta d_seg=-.00001052008734808707
  |     -> delta d_pose=-.17945745448680483
  |     -> delta joint S=-.0516456148508837
  |     -> ADMIT_STRICT_NEGATIVE_JOINT_DELTA_S_TO_C1_WATERFILL
  |
  +-> polished v13 Lane carrier
        -> consume six periodic programs (90 B)
        -> consume 130 drift knots (1962 B)
        -> same CB1/E4 + runtime + n600 chain
        -> delta B=+1530
        -> delta d_seg=+.03659233940972222
        -> delta d_pose=+22.718325524601777
        -> delta joint S=+9.21156940553832
        -> REJECT_UPHILL_FROM_C1_WATERFILL
        -> typed cause:
             pose_survival
             + receiver-paint collision
             + all-class collateral
             + no E4 quantization loss
```

# c1 bucket-attribution edge

Producer:
`.omx/research/ddm_cb1_perclass_carrier_byteclose_20260725/ddm_cb1_perclass_carrier_byteclose_receipt.json`
at JSONPath `$.c1_bucket_attribution_rows`.

Consumer contract: `ddm_c1_bucket_attribution_row.v1`.

- Consume `mycar_static_mask` with `waterfill_eligible=true`.
- Preserve `lane_band_v13_polished` with `waterfill_eligible=false`.
- Do not add rows: these carrier effects were jointly remeasured and are not
  additive estimates.
- Re-measure any successor composition jointly through the same real chain.

# Verdict scope and reopener

Both rows are `INSTANCE`: exact payload, exact application order, exact merged
RG4 base. The MyCar admission does not imply that a later composition retains
its gain. The Lane rejection does not kill the Lane family. Its reopener is a
source-local, scorer-recursive Lane realization derived for the RG4 base that
passes the same joint Pose-survival and all-class collateral gate.

# Triality and custody

- DSL/data: typed JSON config and c1 bucket-attribution rows.
- DAG: this feed.
- Equations: exact joint advisory objective plus per-class error conservation.
- Receiver proof: deterministic CB1/E4 compile x2, strict two-member closure,
  RG4 and parent parse/reemit identity, emitted-runtime output identity, 19
  preserved stages, three raw SHA-256s.
- Measurement proof: n600, real uint8 outputs, frozen scorers, 38 immutable
  JSON+NPZ batches per candidate, independent aggregate replay.

No paid dispatch, training, exact contest evaluation, archive promotion,
campaign fire, or frontier mutation occurred. The official competitive target
is PR130 displayed `0.172`. The local `0.1910828242` row remains a separate
custody baseline only.

# STORES CONSULTED

Delegated authority; `PROGRAM.md`; `CLAUDE.md`; `AGENTS.md`; craft handoff;
v7.5 operating contract; v8 per-class spec; A4 eureka memo; per-class carrier
recall memory; MC1 receipt and static support; v13 worldsheet/Lane receipts;
c1 ledger; RG4 receipt/current archive; E4 exporter and runtime receiver;
exact n600 target cache; frozen scorer modules/weights; watched inboxes.

# MAIN landing review

MAIN must review the entire branch diff and independently check the scorer
batch aggregates, class self-detection, carrier payload identities, CB1/E4
member and runtime custody, raw hashes, objective arithmetic, false-authority
labels, Lane instance scope, and single-row c1 admission before merge.
