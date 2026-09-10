# ddm_bnd3 — the address test does not close the boundary family

`[no-triality] [p0-ledger-ok]` · owner `ddm_bnd3` · `research_only=true` · `score_claim=false`.

**MEASURED: the zero-cost supplied-miss oracle displaces 83,259 B, not ≤20 KB. The best
new joint-assignment recode is 121,689 B, losing 1,905 B to the 119,784 B move-37 envelope.
All 48 new full-n600 variants lose.** bnd2's 121,200 B best remains smaller than these new variants.

**verdict_scope: FORMULATION.** The measured greedy and joint-path formulations lose at these
settings. A FAMILY closure is not established. The charter's address-below-half-share falsifier
fires, but this does not establish an economical full-population boundary code or pass the draw gate.

## Field and authority boundary

Every decomposition, oracle and variant number in this memo is on **move 37**:
archive `670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc`,
field `361cc6c9749fdec1381936836c9b45f4e04702f02eed9f8ea5343b1afa957b94`.
Population: all **600 frames / 117,964,800 cells**, including all **235,044** integer-frequency
coder misses. A support count is the number supplied by the side channel; every remaining token
is still encoded. No subset price is extrapolated.

Axis throughout the tables: **[exact serialized bytes; scorer-free macOS-CPU n600]**.
The live pointer advanced to move 38 during the run. Both workers stopped at their next stage
boundary. `FIELD_SELECTION.json` then explicitly selected the charter's historical move-37 field
for completion. Every subsequent stage rereads the real live pointer and records both identities
in `FROZEN_FIELD_RECHECK.json`; another promotion refuses the launch. Move 39 was observed at
final harvest, after all pricing stages finished. No live pointer or live
sj1/rp1/cmp2 file was edited. These measurements do not adjudicate the move-38 field.

Bulk and all losing candidates:
`/Volumes/VertigoDataTier/pact/ddm_bnd3_address_term/`.
Machine-readable results and all rows: `.omx/research/ddm_bnd3_20260910/RESULT.json` and
`price_table.json`. Final landing custody is separate in `FINAL_HANDOFF.json`.

## Decomposing bnd2's 121,200 B

The actual bnd2 packet has one shared Brotli stream containing addresses, class pairs and normals.
It has **no unique physical compressed address/content allocation**. Assigning a fraction of that
shared stream to offsets would invent a number. The physical decomposition, reproduced with twins,
is instead:

| Physical component | Bytes |
|---|---:|
| TC1 remainder envelope | 119,187 |
| Shared Brotli segment block: counts, starts, lengths, turns, class pairs, normals | 1,997 |
| Segment magic, raw length, CRC, and outer segment length | 16 |
| **Original total** | **121,200** |

To measure the address/content distinction, independently encode the logical streams:

| Actual separate-stream encode | Address B | Class/offset B | Boundary |
|---|---:|---:|---|
| Original bnd2 planar transforms, Brotli q11 | **1,717** | **189** | Raw stream lengths are separately retained; these are substream prices, not a new framed envelope |
| Fixed-width split packet, Brotli q11 | **2,312** | **189** | Adds 22 B total framing and the 119,187 B remainder: **121,710 B** |

Both rows supply **2,125 misses**, not all 235,044. The original side channel costs 2,013 B
including framing and displaces only **597 B on that same support**. Thus its +1,416 B loss is
exactly `2,013 − 597`. The 119,187 B remainder is not an address charge. Class-pair identity is
counted with offset/content; no source-derived graph is free to the decoder.

## Signalled oracle and the displaceable share

The oracle supplies **both the miss location and its true symbol at zero charged cost**. Every
true symbol still participates in the original causal trajectory. The encoder skips the supplied
symbols and encodes the remainder through the original TC1 rows in the original group order.
This removes surprise without changing the field, unlike bnd2's masked-symbol intervention.
The free oracle support and values are physically retained, but expressly excluded from the
oracle's hypothetical charge; this is not a legal submission.

| Full-population quantity | Bytes |
|---|---:|
| Original TC1 envelope | 119,784 |
| Oracle remainder envelope, primary = repeat | **36,525** |
| **Displaceable share: original minus oracle** | **83,259** |

The oracle resumed the complete arithmetic state saved after frame 32, then completed frame 600.
Its full-field cached-row decode reproduces the pinned field. The bound is an ideal saving for
this **unchanged TC1 remainder**, with all miss support and values free. It is not a universal
bound over other probability models, use of negative flag information, representations, or fields.
A flag giving only miss/not-miss would not itself identify which non-modal class to output; the
stronger free-symbol oracle is what was actually measured.

## Joint assignment and gap bridging

The joint method exposes every allowed class-pair unit crack and signed normal −1/0/+1 for
each eligible miss. Binary variables jointly choose one assignment per miss and predecessor/
successor links. Cycle cuts ensure a representable path cover. The MILP minimizes the **actual
raw address wire**, `4 B/frame + 8 B/segment + 1 B/visited edge`; the emitted raw length is checked
against that objective. HiGHS reports zero integer gap, and independent repeated solves select
identical assignments. The complete frame optima sum because that raw wire is frame-separable.
This is a joint solve, not greedy-only assignment. It does **not** claim an optimum of Brotli size.

For k = 1, 2, 4, links may cross at most k correctly predicted cells on the same actual crack.
Every skipped cell is charged a direction/skip byte and emits no token. All orientations and
minimum supplied-run lengths 1, 2, 4 are priced. Two layouts are retained for each: the fixed-width
wire, and a sorted columnar varint transform matching bnd2's compression design. The latter
prevents a worse address layout from being mistaken for evidence against joint assignment.

Each row below is the smallest measured envelope for that orientation/gap across both layouts
and all three run thresholds. All selected rows use the columnar layout and minimum run 4.
**22 B framing** is included in every total. Complete 48-row accounting is in `price_table.json`.

| Orientation | Bridged correct cells k | Supplied misses | Address B | Content B | Remainder B | Total B | Loss vs baseline B |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0, joint only | 3,128 | 2,559 | 269 | 118,839 | **121,689** | +1,905 |
| 0 | 1 | 4,796 | 4,266 | 388 | 118,300 | 122,976 | +3,192 |
| 0 | 2 | 5,642 | 5,279 | 448 | 118,018 | 123,767 | +3,983 |
| 0 | 4 | 7,265 | 7,452 | 565 | 117,500 | 125,539 | +5,755 |
| 1 | 0, joint only | 3,168 | 2,614 | 293 | 118,828 | 121,757 | +1,973 |
| 1 | 1 | 5,117 | 4,667 | 446 | 118,183 | 123,318 | +3,534 |
| 1 | 2 | 6,326 | 6,176 | 544 | 117,777 | 124,519 | +4,735 |
| 1 | 4 | 9,357 | 10,235 | 762 | 116,753 | 127,772 | +7,988 |

The best new row displaces **945 B** on its own 3,128-cell support but spends **2,850 B** on
address, content and framing. Its actual ZIP is **182,293 B**. No total score is inferred.
For minimum-run-1 variants carrying all edge-eligible misses, the smallest achieved address
stream is **338,805 B**, already above 83,259 B. That closes the measured all-eligible variants,
not all possible address codes: an achieved code length is an upper bound on attainable rate.

## Prior law and closure level

- **Address ≥60% of 121,200 B: rejected.** The original entire segment/framing region is only
  2,013 B; 119,187 B is the residual. Separately encoded address cost is 1,717 B.
- **Oracle displaceable share ≤20 KB: rejected.** The actual difference is 83,259 B, more than
  either decimal or binary interpretation of 20 KB.
- **No tested variant crosses 119,784 B: confirmed for these 48 variants.** Neither the
  114,784 B byte gate nor a lossless archive win occurs. Joint/gap predictions about modest
  improvement do not imply a global compressed optimum; exact raw-wire statistics are retained
  separately in `geometry_table.json`.
- **Address <50% of whole-population displacement: the charter's falsifier fires.** The best
  new address stream is 2,559 B versus half-share 41,629.5 B. However, its support is 3,128 cells;
  the whole oracle support is 235,044. This comparison prevents the proposed family closure;
  it does not show that all misses can be addressed that cheaply.

**verdict_scope: FORMULATION.** These explicit joint-path/gap grammars and bnd2's greedy
grammars are losing formulations on move37. No FAMILY or PARADIGM closure follows. In particular,
the minimum achieved address length among tested encodes is not a universal address lower bound.
GS3 Addendum 19's broader all-representation wording must retain this scope restriction.

## RECALL EVIDENCE

The full query receipt and beyond-seed findings are in `ddm_bnd3_20260910/RECALL.md`,
`recall_queries.json`, and `equations_recall.json`. Research memos and arm receipts were searched
by content for address cost/bytes/packets/terms, joint assignment, gap bridging and boundary
segments. The equations CLI, dated canonical research index, DAG, SPEC/design docs and task ledger
were independently searched. The initially missing unsuffixed index path was corrected and searched.

**Beyond charter seeds:** AD2's QEVENT accounting explicitly forbids a fictitious address/value
split inside a shared compressed stream. Its successful order transform also motivated the
matched columnar control here. The TC1 context and reorder laws required same-field causal rows;
their older prices were not transferred. OR1 and OF1 remain scope-specific priors. No additional
matching construction was found in the exact index/DAG/docs query scope. No external candidate
payload or weights were imported. The canonical Codex memory registry query found no relevant hit.

## Verification, integration and landing

All 48 payload/archive twin pairs match, and all 48 retained full-field cached-row parsebacks
match the move-37 SHA; the final audit rehashed all 48 retained fields and rechecked every packet/
archive twin. Every complete geometry solve has a repeated assignment and an actual
raw-wire length check. Random n32 real-frame regression tests verify the joint-vs-greedy raw
objective, true crack sides, correct-cell skip behavior, both packet decoders and malformed-packet
refusals. Three tests passed; all five Python files have two visible review passes and clean Ruff.

The oracle and initial geometry were resumed from disk after frame 32. Encoders checkpoint complete
native arithmetic state every 20 frames and at a requested stage end; decoders checkpoint native
state every 20 frames. Geometry checkpoints every completed frame. All stage and intermediate
payloads remain under the owned SSD root. No bulk was deleted or moved; all bytes are active
reproducibility inputs, and no disposable scratch or scorer cache was produced. Launch storage
checks and the 40 GiB reserve fail closed. At most two numerical workers ran; every launch used
`--nice 10 --nice-best-effort`. Short successful probes that exited inside the launcher's initial
3-second liveness window have child rc=0 receipts; later launches use a 0.1-second window.

**Not measured:** new Seg/Pose outputs, fresh causal-model recomputation by a bnd3 decoder,
public inflate/video identity, contest-CPU/CUDA replay, compressed-size-optimal joint assignment,
or move-38 representation prices. The inherited source trace is causal and identity-verified;
the new parsebacks explicitly use its cached coding rows. No scorer, draw, remote dispatch or
public runtime build was performed.

Underlying accounting is anchored on registered **`boundary_segment_recode_price_v1`**:
`total = address + content + framing + unchanged-causal-remainder`. The bnd3 empirical anchor and
scope clarification are retained with the narrow canonical API events. No fictitious universal
`boundary_address_term_floor_v1` is registered.

Six solver hooks: sensitivity, Pareto deployment constraint, per-tensor allocation and autopilot
deployment are N/A because this is a losing research-only representation with no new scorer or
production actuator. Canonical equation evidence and typed task dispositions retain the result;
no numerical scorer posterior update is claimed. The probe-disambiguation hook is the measured
oracle and paired same-support displacement. The draw successor remains blocked by its concrete
byte gate and receiver/field/authorization conditions, not a broad family verdict.

The charter's falsifier requeues the old bnd1 ITEM_2 through a linked successor because the original
cancelled ledger row is terminal. That successor is **QUEUED-WITH-A-FIRE-ORDER**, owner MAIN;
the old row is annotated, not rewritten. The archive gate still fails, so no draw is fired.
The exact serializer outcome and any main-write denial, patch, bundle and commit identity are in
`ddm_bnd3_20260910/FINAL_HANDOFF.json`. A fallback commit is not a main landing.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer stores: the owned source/memo/receipts and
  `.omx/state/canonical_task_status.jsonl`, `.omx/state/canonical_equations_registry.jsonl`,
  `.omx/state/lane_registry.json`; fire trigger: verified final serializer artifacts are harvested
  in a Git-writable context. Land only the exact unit in `FINAL_HANDOFF.json`, replay absent narrow
  API events, and record the verified main commit and checks. A verified main landing folds this action.
- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer store `.omx/state/canonical_task_status.jsonl`,
  task `ddm_bnd3_address_term_decomposition_20260910::ITEM_2_REOPENED`; fire trigger: a new lossless
  receiver-verified twin candidate meets the same-field 5,000 B saving gate (≤114,784 B for move37),
  is explicitly bound to the field selected for the draw, and receives a scorer allocation. Then
  execute the linked bnd1 ITEM_2 draw. If the field changes, rederive and satisfy the gate first.

## LIVE-HYPOTHESES

- A compression-aware joint choice of support and edge assignment remains untested. It is plausible
  because the raw-byte minimum and the smallest compressed packet need not select the same paths.
  **FOLDED** into the reopened ITEM_2 admission gate; no unpriced draw or new dispatch is implied.
- A shared region or temporal description may amortize support cost differently from per-token paths.
  These tests supply no universal lower bound against it. **FOLDED** into that same admission gate.

## DEAD-ENDS

- **FORMULATION:** these 48 joint/gap recodes lose; repeating their two orientations, k=0/1/2/4,
  run thresholds 1/2/4 and two layouts on the same field does not open the draw.
- **Accounting premise:** treating the 121,200 B envelope as predominantly address bytes is false;
  119,187 B is the TC1 remainder. A shared Brotli block has no unique physical address/content split.
- **Intervention premise:** replacing true misses by predictions changes the causal trajectory;
  bnd2's 129,316 B masked control is not the displaceable-share oracle.
- **Proof premise:** an achieved address length is not a universal lower bound, and a small selected
  support cannot inherit the whole-population oracle saving. Neither premise supports FAMILY closure.

Own-vehicle live frontier, reread from the pointer and recomputed from its recorded components:
**S 0.13766931482209038 @ 180,186 B [contest-CUDA T4 n600]** (move 39, advanced by rp1).
This arm did not move the exact score pointer and did not achieve the sub-0.12 goal.
