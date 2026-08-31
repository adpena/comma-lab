# ddm_lfb1 — Lane fold carriage costs 200,975.25 B in gf1 currency, 9.262x the 21,699 B bar

Date: 2026-08-31 · Owner: `ddm_lfb1_lane_fold_carriage_bar`  
Axis: `[macOS-CPU scorer-free exact count]` · `score_claim=false` · `promotable=false`  
Verdict scope: **INSTANCE** — class-1 Lane folded into class-0 Road on the SHA-pinned
`9ba2e52b...` n600 token field, with the exact count priced at gf1's measured clustered-residual
rate. This is not a new coder measurement and closes no Lane-carrier family.

## 1. CONTROL FIRST — gf1 reproduces 1,325,033 exactly

I ran the charter's unchanged positive control before the fold:

```text
.venv/bin/python experiments/ddm_gf1_generator_form_on_lb1_field.py \
  --field /Volumes/APDataStore/pact/ddm_dc1_20260816/retained/redecoded_tokens_n600.u8 \
  --out /Volumes/VertigoDataTier/pact/ddm_lfb1_lane_fold_carriage_bar/control_gf1_unfolded
```

| control fact | expected | reproduced |
|---|---:|---:|
| input bytes | 117,964,800 | 117,964,800 |
| input SHA-256 | `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52` | exact |
| gf1 capacity-gap mismatches | **1,325,033** | **1,325,033** |
| gf1 replacement bytes | 433,051 | 433,051 |

Control result: `/Volumes/VertigoDataTier/pact/ddm_lfb1_lane_fold_carriage_bar/control_gf1_unfolded/RESULT.json`,
SHA-256 `9a227cc82439d7fdb642ea618d91949ed06256e830d195a39f611d3e3da029d3`.
The control retained its generated field, three full residual orders, four raw generator streams,
their coder outputs, packet, logs, and result. No scalar-only run occurred.

## 2. THE MEASUREMENT — 690,874 exact Lane-to-Road corrections

I copied the full `(600, 384, 512)` uint8 field, replaced every class-1 Lane token with class-0
Road, left every other byte unchanged, and applied gf1's exact expression:

```python
mismatches = int(np.count_nonzero(folded != target))
```

Full-population result, with no subset and no projection:

> **MEASURED: 690,874 / 117,964,800 positions = 0.5856611464% require restoration.**

Two independent same-field controls close the count:

1. `np.bincount(target.reshape(-1))[1] = 690,874`.
2. gf1's prior retained `GF1_CLASS_DECOMPOSITION.json` records exactly **690,874** true Lane
   positions on this same field; the fold count matches.

The retained fold and an independently written determinism repeat are byte-identical:

| retained payload | bytes | SHA-256 |
|---|---:|---|
| `retained/lane1_folded_to_road0_n600.u8` | 117,964,800 | `dd5c1b1e0bd4a6c5b5aa784ae1c2e9874d1379d253d6ecc06af18d8dbf720499` |
| `retained/lane1_folded_to_road0_n600.repeat.u8` | 117,964,800 | `dd5c1b1e0bd4a6c5b5aa784ae1c2e9874d1379d253d6ecc06af18d8dbf720499` |

## 3. PRICE AGAINST THE EXACT BAR — D3 stays closed

The mismatch count is **MEASURED exact**. The byte figure below is **DERIVED** with the charter's
required conversion, gf1's own **MEASURED 0.2909 coded B/correction**:

```text
690,874 corrections * 0.2909 coded B/correction = 200,975.2466 B
200,975.2466 / 21,699                           = 9.26195892x
200,975.2466 - 21,699                           = 179,276.2466 B over
```

> **Answer: restoring Lane from the folded field costs 200,975.25 B in gf1 currency, MORE than
> the 21,699 B bar by 179,276.25 B, or 9.262x. The D3 route is CLOSED at INSTANCE scope.**

Currency boundary: **200,975.2466 B is not a newly encoded Lane payload.** It is the exact n600
count multiplied by gf1's measured generic-LZ clustered-residual price, as the charter ordered.
An actual carrier can only claim its own coded byte count after retaining and receiver-checking its
bytes.

## 4. THE THREE-ROW CARRIAGE TABLE

| Lane carrier / price | carriage B | vs 21,699 B bar | evidence and disposition |
|---|---:|---:|---|
| `gf1` lane stream | **36,044** | **1.661x** | MEASURED coded; HG1 formulation REFUSED at 5.09x and leaves 318,406 Lane misses |
| D3 `block_s3_t3` + D3 framing | **52,539** | **2.421x** | 52,531 B measured carrier payload + 8 B measured tag/length framing; n600 scorer-refused INSTANCE |
| **Lane-to-Road fold restored at gf1's unit price** | **200,975.2466** | **9.262x** | **MEASURED count x MEASURED unit price; DERIVED B; BLOCKS D3** |

The prior-law prediction got only the sign right. It predicted **above 21,699 B**, which holds, but
also predicted a result between gf1's 36,044 B and D3's 52,539 B carriage footprint. The
measured-count price is **5.576x** gf1's stream and **3.825x** D3's carrier-plus-framing. That
range prediction is falsified.
The charter's re-open falsifier, `<21,699 B`, did not fire.

## 5. PER-CLASS ATTRIBUTION — concentrated entirely in Lane-to-Road

| class | before fold | after fold | changed positions |
|---|---:|---:|---:|
| Road (0) | 27,407,378 | 28,098,252 | +690,874 |
| Lane (1) | 690,874 | 0 | -690,874 |
| Undrivable (2) | 58,413,399 | 58,413,399 | 0 |
| Movable (3) | 1,460,319 | 1,460,319 | 0 |
| MyCar (4) | 29,992,830 | 29,992,830 | 0 |

All **690,874 / 690,874 = 100%** non-identity transitions are `Lane 1 -> Road 0`. Restoration is
therefore concentrated, not spread across classes. The fold count is **52.1401%** of gf1's
1,325,033-control mismatch count, versus gf1's own Lane stream contributing 318,406 misses
(24.03%) while consuming 75.7% of its packet. The fold does not inherit a favorable version of
that asymmetry: deleting Lane creates **2.169x** as many Lane corrections as gf1's fitted Lane
stream misses, and pricing them overwhelms the bar.

## 6. TYPED DISPOSITION

```text
type: BLOCKED_TYPED_D3_LANE_CARRIAGE_BAR
disposition: FOLDED_CLOSED_INSTANCE
owner: ddm_lfb1_lane_fold_carriage_bar
consumer_store: .omx/state/main_hot_state.md — D3/gf1 Lane-carriage route table
blocked_object: D3 rate-only archive 116,287 B plus Lane restoration on the 9ba2e52b field
measured_count: 690,874 corrections
derived_price: 200,975.2466 B at gf1 measured 0.2909 B/correction
bar: 21,699 B
fire_trigger: a retained receiver-closed Lane carrier with actual coded bytes <=21,699, or a new exact distortion credit that explicitly enlarges this bar
```

This is a terminal **FOLDED** disposition, not an unfired follow-on. No scorer, Modal job, archive
mutation, or dispatch is queued.

## RECALL EVIDENCE

Sources searched before adjudication:

- Full memo/receipt content queries over `.omx/research/` for `lane fold`, `fold Lane`, `dominant
  neighbor`, `Lane carriage`, `21,699`, `0.2909`, `318,406`, and `1,325,033`.
- Charter seeds read at source: `ddm_rt3`, `ddm_d3rec`, `ddm_gf1`, `ddm_d3`, `ddm_af1`, `ddm_ld1`,
  `ddm_d3a`, `ddm_ffrec`, `ddm_msr1`, and `ddm_rr9`.
- `.venv/bin/python tools/list_canonical_equations.py --json`,
  `CANONICAL_RESEARCH_INDEX_20260629.md`, `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`,
  `.omx/state/main_hot_state.md`, `.omx/state/canonical_task_status.jsonl`,
  `harness_tasklist_bridge_20260803.jsonl`, and `active_lane_dispatch_claims.md`.

Beyond the charter seeds, `ddm_ma2_merged_alphabet_lane_fold_20260824.md` confirmed that this
vehicle's merge semantics are Lane class 1 into Road class 0, but its retained token field is
**`cc10a7b0...`**, not this charter's **`9ba2e52b...`**. Its decoded-Lane count is 691,095, differing
by 221 sites. I initially treated it as a same-field positive control; the digest/count guard
refused, both already-written folded payloads were retained, and I corrected the join instead of
laundering the adjacent-object count. The plan changed to use gf1's own same-field retained
decomposition, which independently records 690,874 Lane sites. No canonical-equation, DAG, task,
or dispatch row already measured this exact 21,699 B carriage cell.

The source read also reconciled the apparent **8 B** D3 discrepancy. `block_s3_t3` is a 52,531 B
counted carrier payload (52,529 B Brotli body plus two carrier bytes), but the actual D3 archive adds
an 8 B tag/length frame. Because 21,699 B is a whole-archive carriage bar, the comparable D3 row is
**52,539 B**, exactly as the charter and `ddm_d3rec` carry. The payload-only 52,531 B must not be
substituted into this table.

## 7. CUSTODY AND BOUNDARIES

All new materialized bytes live under
`/Volumes/VertigoDataTier/pact/ddm_lfb1_lane_fold_carriage_bar/` as required. The store is 354 MB.
The machine-readable result is `LANE_FOLD_CARRIAGE_RESULT.json`, SHA-256
`e8f36a01eadfc0c9fce91f334b03cbf69ba40f54313d8a83436a79c1c4aee4d3`. The 58-artifact custody
inventory, commands, paths, byte counts, and SHA-256 values are in `CUSTODY_MANIFEST.json`, SHA-256
`a3412416218a8c271486d02ba905e0742a5b7276cb1e166a56ad99ce4e2440c8`.

The first refused join is preserved in `fold_measurement.log`; the corrected run is preserved in
`fold_measurement_v2.log`. Nothing was deleted or routed to local disk.

What I did **not** measure: no new residual coder, no actual ≤21,699 B carrier, no SegNet/PoseNet,
no receiver rendering of a candidate, no archive size, and no exact contest score. Therefore this
memo cannot close every possible Lane carrier and cannot move the frontier. It closes the D3 route
under the exact fold/count conversion the charter specified.

Denominator — candidates enumerated / measured / closed-by-recall / ABSENT:
**1 / 1 / 0 / 0** (the full-n600 Lane-to-Road fold on the pinned field). The comparison table also
carries two prior measured carriers, both traced to source; neither was re-measured here.

The exact pointer did not move. This unit did **not** achieve the sub-0.12 goal; it replaced an
open inferred carriage cell with a full-n600 scorer-free blocker.

`[contest-CUDA T4 n600] own-vehicle frontier: LB1 — S=0.14803010583079396, archive=180,083 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9.`
