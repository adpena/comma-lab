# Codex findings — DDM DM1 25-row solved-value pricing

`lane_id=lane_ddm_dm1_25_row_solved_value_pricing_20260724`  
`research_only=true` · `score_claim=false` · `promotion_eligible=false` ·
`main_review_required=true`  
`evidence_axis=[macOS-CPU frozen-scorer advisory]`  
`0.1910828242 [contest-CPU] UNMOVED`

## One-line verdict

**MEASURED:** the 25 SHA-bound semantic records cost 4,124 bytes when each row
chooses its own exact coder, or 1,569 bytes as one exact LZMA-9 shared-context
record; boundary rows re-home to `SKELETON/L4_scorer_feature` with an owed L3
realization, cell rows confirm `FIBER/L4_scorer_feature`, and the cheapest
receiver-closed archive price remains `NULL`.

Receipt:
`.omx/research/ddm_dm1_25_row_solved_value_pricing_20260724T123443Z/ddm_dm1_25_row_solved_value_pricing_receipt.json`
(`sha256=4c2fe77927e300e341d5ce9ce00ae8a37c58dbebbde8e5860fe514958990de28`).

## What was materialized

The implementation used the same production input context and bounded
solved-plane loader as IS1. It streamed the 15 requested pairs from ten
SHA-verified solved-plane chunks, constructed the exact camera preimage, and
ran only the frozen SegNet last-frame path. No n600 RGB plane or logits tensor
was persisted.

For every PF2 `{pair_id,bucket_id}` row:

- the exact PF2 event-index coordinates were selected and SHA-bound;
- the solved SegNet argmax symbol and the left-minus-right pairwise-margin
  relation were materialized;
- a canonical typed semantic record was encoded;
- zlib level 9, LZMA preset 9, and adaptive order-1 byte-context arithmetic
  were each decoded and canonically re-encoded;
- the row receipt names the exact next receiver-closure measurement.

Boundary records count their delta-coded placement coordinates plus solved
choice. Cell records reference the already-existing PF2 support by SHA and
count only the solved within-support choice. If that external support is absent
or differs by one coordinate, cell parseback refuses.

## Exact row ledger

All byte counts below include the self-identifying container header, raw length,
payload length, and raw-record SHA. `n` is the exact PF2 support count.

| row | pair | bucket | n | adjudicated home | best coder | bytes |
|---:|---:|---|---:|---|---|---:|
| 0 | 523 | `lane_movable__cell__static_in_image` | 34 | FIBER/L4 | zlib9 | 157 |
| 1 | 523 | `lane_undrivable__boundary__static_in_image` | 291 | SKELETON/L4 | zlib9 | 258 |
| 2 | 523 | `lane_undrivable__cell__static_in_image` | 36 | FIBER/L4 | zlib9 | 161 |
| 3 | 54 | `road_mycar__cell__static_in_image` | 23 | FIBER/L4 | zlib9 | 155 |
| 4 | 90 | `lane_mycar__boundary__static_in_image` | 6 | SKELETON/L4 | context arithmetic | 162 |
| 5 | 90 | `undrivable_movable__boundary__transient` | 2 | SKELETON/L4 | context arithmetic | 156 |
| 6 | 446 | `lane_undrivable__boundary__static_in_image` | 190 | SKELETON/L4 | zlib9 | 223 |
| 7 | 446 | `lane_undrivable__cell__static_in_image` | 51 | FIBER/L4 | zlib9 | 164 |
| 8 | 0 | `road_undrivable__boundary__transient` | 4 | SKELETON/L4 | context arithmetic | 158 |
| 9 | 14 | `lane_undrivable__cell__static_in_image` | 1 | FIBER/L4 | context arithmetic | 150 |
| 10 | 327 | `road_mycar__cell__static_in_image` | 3 | FIBER/L4 | context arithmetic | 148 |
| 11 | 60 | `lane_movable__boundary__static_in_image` | 2 | SKELETON/L4 | context arithmetic | 158 |
| 12 | 60 | `road_mycar__cell__static_in_image` | 3 | FIBER/L4 | context arithmetic | 148 |
| 13 | 323 | `lane_mycar__boundary__static_in_image` | 5 | SKELETON/L4 | context arithmetic | 161 |
| 14 | 323 | `undrivable_movable__boundary__transient` | 17 | SKELETON/L4 | zlib9 | 167 |
| 15 | 38 | `lane_mycar__boundary__static_in_image` | 8 | SKELETON/L4 | zlib9 | 165 |
| 16 | 42 | `lane_movable__boundary__static_in_image` | 2 | SKELETON/L4 | context arithmetic | 158 |
| 17 | 4 | `lane_undrivable__boundary__static_in_image` | 12 | SKELETON/L4 | zlib9 | 178 |
| 18 | 55 | `lane_undrivable__boundary__static_in_image` | 2 | SKELETON/L4 | context arithmetic | 159 |
| 19 | 55 | `road_undrivable__boundary__transient` | 2 | SKELETON/L4 | context arithmetic | 153 |
| 20 | 56 | `lane_movable__boundary__static_in_image` | 3 | SKELETON/L4 | context arithmetic | 161 |
| 21 | 56 | `lane_movable__cell__static_in_image` | 18 | FIBER/L4 | zlib9 | 158 |
| 22 | 56 | `lane_mycar__boundary__static_in_image` | 4 | SKELETON/L4 | context arithmetic | 159 |
| 23 | 16 | `road_mycar__cell__static_in_image` | 6 | FIBER/L4 | context arithmetic | 151 |
| 24 | 16 | `undrivable_movable__boundary__transient` | 2 | SKELETON/L4 | context arithmetic | 156 |

## Byte accounting

**MEASURED:**

| pricing mode | zlib9 | LZMA-9 | context arithmetic | selected |
|---|---:|---:|---:|---:|
| sum of 25 independent containers | 4,216 | 5,629 | 4,458 | 4,124 per-row minimum |
| one joint 25-row container | 1,593 | 1,569 | 2,387 | 1,569 LZMA-9 |

**DERIVED:** shared coding saves `4,124 - 1,569 = 2,555` bytes relative to
independent per-row selection. This is description redundancy in the 25
semantic records. It is not a nonadditive score pool and is not permission to
subtract 1,569 bytes from any archive or #613 ledger.

## #669c re-homing adjudication

### Boundary rows

`SKELETON` is confirmed: the counted value includes the video-specific
placement of a class interface. The candidate `L3_raster` home is corrected to
`L4_scorer_feature` under this task's explicit **deepest surviving L1→L4**
rule. The exact materialized object is the interface-support coordinate set
plus its frozen-SegNet choice, and that object exists at L4 immediately before
the L5 argmax verdict.

This correction does **not** make L3 free or solved. The contest receiver can
emit RGB, not inject SegNet features; a legal realization still owes an L3 RGB
preimage that reproduces this exact L4 record without Pose harm. Therefore
1,569 bytes is a semantic-record price, while the receiver-closed archive price
is `NULL`.

Wrong candidates:

- `SKELETON/L3` as the information home is too shallow under the mandated
  deepest-surviving rule; L3 is the realization surface.
- `CONNECTION` is unmeasured: the ledger has no like-for-like, same-bucket
  consecutive-pair comparator.
- `GAUGE` is false in the Seg verdict scope: changing the recorded choice
  changes the L5 argmax object.
- `RESIDUAL` is premature because SKELETON is an exact admissible semantic
  home.

### Cell rows

`FIBER/L4_scorer_feature` is confirmed. Placement already exists in the
SHA-bound PF2 support context; the added video-specific information is the
within-support class/pairwise-margin choice. Counting the coordinates again
would erase the measured SKELETON/FIBER asymmetry.

Wrong candidates:

- `SKELETON/L3` double-counts existing support placement.
- `CONNECTION` is unmeasured for the same missing like-for-like adjacency.
- `GAUGE` is false for the Seg verdict choice.
- `RESIDUAL` is premature because FIBER is an exact admissible semantic home.

## ξ-adjacent predictability

The preregistered exact comparison is same `bucket_id` at consecutive pair IDs
(`Δξ` proxy gap 1). The 25-row ledger contains zero such pairs.

**MEASURED NEGATIVE / NULL:** `CONNECTION` has no eligible comparator, so its
relation payload and byte price remain `NULL`. This is not evidence that the
CONNECTION family is absent. Next measurement: register at least one
same-bucket consecutive-pair solved support, then price the deterministic
relation with exact parseback.

## Context only: #613 and the tangent

The total pointer remains `0.1910828242 [contest-CPU]`. The #613 slack and the
154,522-byte fixed-distortion tangent are cited only as inherited context.
No new box arithmetic was performed and no nonadditive byte pool was created.
Because the measured record has no receiver-closed RGB/Pose proof, neither
4,124 nor 1,569 is an admissible archive byte delta.

## Directive disposition table

| Binding directive | Disposition |
|---|---|
| Read solved object, not problem reachability | PASS — records contain solved L4 choice and exact support/placement only |
| No RG4 or vocabulary sweep | PASS — only the registered 25 rows were opened |
| Same production loaders as IS1 | PASS — `_open_production_inputs` and `_load_production_inputs`, one pair at a time |
| Revalidate every input SHA | PASS — demand, RG3 summary, source config, PF2 receipt/index, scorer weights/modules, solved receipt/chunks |
| Sealed five-type/five-layer contract only | PASS — `StreamType`, `LayerHome`, and `TypedStreamTag` reused unchanged |
| Boundary/cell asymmetry | PASS — boundary counts placement; cell references existing support by SHA |
| Deterministic zlib-9/LZMA/context arithmetic | PASS — all three exact and canonical; independent and joint prices recorded |
| Exact parseback | PASS — semantic, coder, and 25-row joint parseback verified |
| ξ-adjacent candidate measurement | NULL — zero eligible same-bucket `Δpair=1` comparisons; next measurement named |
| #613 / 154,522B context only | PASS — no score or box arithmetic |
| No training, paid dispatch, exact eval, archive, or pointer mutation | PASS |
| Every row names next measurement | PASS — machine-readable in receipt |
| Triality + durable artifacts | PASS — code/config, DAG FEED, registered callable equation, receipt/findings |
| MAIN landing review | REQUIRED — see review surface below |

## STORES CONSULTED

- delegated authority prompt, SHA
  `2486fab73378861a3e53e499f69ea8995fb1b09f950af2bdcdeb8362e7baa3d2`
- `CLAUDE.md`, `AGENTS.md`, and the DDM operating manual
- top ten Claude memory entries and current Codex/Claude sister memos
- `reports/latest.md`, lane registry, subagent progress, canonical equation
  registry, probe ledgers, and both directive inboxes
- IS1 findings, DAG FEED, canonical-equation note, source config, and exact
  residual receipt
- the 25-row solution-demand ledger and RG3 summary/assignment/PF2 tables
- SHA-bound solved-plane receipt/chunks and GT cache through the production
  loader
- PF2 event-index receipt and NPZ on the SSD tier
- frozen `segnet.safetensors` and `upstream/modules.py`

No web source, old vehicle lineage, donor payload, paid provider, or live
frontier artifact was used.

## Triality and MAIN review surface

- **DSL/code:** `ddm_dm1_solved_value_pricing.py`, its CLI, typed contract
  reuse, config, and tests.
- **DAG:** `ddm_dm1_25_row_solved_value_pricing_DAG_FEED_20260724.md`.
- **Equations:** callable
  `ddm_dm1_semantic_record_price_and_rehome_v1`, registered through
  `populate_ddm_dm1_semantic_record_price_and_rehome_v1`, plus its note.
- **Evidence:** SHA-bound receipt and manifest above.

MAIN must review the semantic choice that boundary placement is homed at L4
under the task's deepest-surviving rule while L3 remains mandatory realization
debt; the external-support accounting for FIBER; the self-contained container
overhead; and the explicit firewall preventing 1,569 bytes from being treated
as a receiver-closed archive or score delta.

## Verdict scope

`MEASURED` means exact bytes and parseback for these 25 semantic records on the
local frozen-SegNet axis. It excludes a receiver-closed RGB compiler, Pose
survival, full-video replay, `upstream/evaluate.py`, contest-CPU/CUDA, archive
bytes, submission score, promotion, and frontier movement.
