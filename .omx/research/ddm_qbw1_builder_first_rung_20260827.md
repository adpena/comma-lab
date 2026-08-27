# ddm_qbw1 builder first rung — scorer-free verdict

Date: 2026-08-27 · arm: `ddm_qbw1_builder_first_rung` · disposition:
**CLOSED at FIRST-RUNG FORMULATION scope on the rate leg**.

The exact frontier did not move.  The scorer-free QBW1 v1 receiver contract passed, but the
pre-registered HT estimate is **336,286 B for the quotient** and **389,362 B for the complete
fixed-envelope portrait**, versus the **137,986 B** cap.  It misses by **251,376 B**.  The sealed
stage-03+ request therefore has disposition `FOLDED_BY_SCORER_FREE_GATE`; no training, Metal,
PoseNet, SegNet, Modal, n600 build, or contest evaluation was launched.

Axis: `[macOS-CPU scorer-free advisory, seeded-stratified random n32]` · `score_claim=false` ·
`promotion_eligible=false` · `pointer_moved=false`.

## Measured answer

| quantity | measured value | gate / interpretation |
|---|---:|---|
| selected exact reset records + winning shared model, n32 | 17,595 B | real serialized payloads |
| `B_shared` | 16 B | dictionary-free `QBM1` winner |
| HT `B_var_hat` | 336,270 B | `sum_h (N_h/n_h) sum_i b_i` |
| HT `B_hat_quotient` | **336,286 B** | allowance 84,910 B — FAIL |
| fixed renderer/pose/framing envelope | 53,076 B | projection only, not materialized here |
| complete archive portrait | **389,362 B** | cap 137,986 B — FAIL by 251,376 B |
| projected source interface length | 1,629,645 | all four-neighbour class interfaces |
| Road↔Lane / other projected interfaces | 822,585 / 807,060 | decomposition only, no score credit |
| serialized quotient bytes per interface | **0.20635537187546982** | cap-equivalent 0.05210337220683032 |
| reduction still required | **3.9605x** / **74.7507%** | before any renderer/carrier growth |

This is the portrait's first real number.  It is a serialized reset-record measurement, not entropy,
loss, mask IoU, a temporal diagnostic, or an n600 score.

### Projected byte decomposition

The independent post-run parser re-read every counted envelope and exactly reproduced
`B_var_hat=336,270 B`:

| component | HT projected coded/envelope bytes | HT projected raw bytes |
|---|---:|---:|
| reset-record framing | 54,000 | — |
| oriented base crack chains | 173,370 | 833,610 |
| region seed labels | 3,255 | 2,055 |
| Lane Road-graph dash events | 105,645 | 108,900 |
| **variable total** | **336,270** | — |

Even deleting every v1 framing byte would leave 282,286 quotient bytes including the shared model,
still 197,376 B over the allowance.  The negative therefore does not depend on the deliberately
self-describing reset envelope, although a successor must not inherit that envelope without a fresh
price.

### Frozen grammar race

Every model and every record below was actually serialized and retained.  Candidate dictionaries
are exact suffixes of the retained 49,657-B raw dictionary-fit source.

| dictionary capacity | model B | n32 records B | exact model + records B | disposition |
|---:|---:|---:|---:|---|
| 0 | 16 | 17,579 | **17,595** | selected |
| 4,096 | 4,112 | 15,126 | 19,238 | retained loser |
| 16,384 | 16,400 | 11,619 | 28,019 | retained loser |
| 32,768 | 32,784 | 8,197 | 40,981 | retained loser |

The larger dictionaries compress records better but cost more counted shared bytes than they save.

## Schema freeze receipt

The schema and parser landed **before the first QBW1 payload existed**:

- `.omx/research/SPEC_ddm_qbw1_packet_schema_v1_20260827.md` — 119 lines, SHA-256
  `762d2845e93c6d6546a07dbe7ae9cd7f399b53b3ea716cd20b6f7421b8a9a42e`;
- `experiments/ddm_qbw1_packet.py` — 674 lines, SHA-256
  `d1bc4851e6381145d5625c44b3ad32300340294b7866df71ba4c2cf56f95fcec`;
- serializer commit `c4db9a5e69`.

The wire object is a four-label Lane→Road base partition represented by oriented maximal crack
chains and canonical region seeds.  Lane is represented only by signed tangent/normal dash events
anchored to the decoded Road-boundary graph.  No event carries `(pair,x,y)`.  The categorical Lane
raster is observability output, not a fixed RGB paint path.  The parser verifies locations, oriented
side labels, integrated cells, closed-chain equality, section CRCs, model SHA, minimal varints, and
trailing-byte absence.

## Builder inventory and review

| file | role | lines | content SHA-256 |
|---|---|---:|---|
| `experiments/ddm_qbw1_builder.py` | storage preflight, selection, object extraction, stages 00–02, sealed fire order, custody/replay validation | 1,127 | `6be96c6385383003e183eab145850669f3d220403676af86ab8b7dd1ec6f9f8e` |
| `experiments/tests/test_ddm_qbw1_builder.py` | real-field receiver/mutation and exact seeded-selection tests | 81 | `dc0d305f6c4cf2425ecec114618794c08859ef00d7619331f240e471f5ff0790` |

Builder commits: `b1031932e2` (stages), `9758e96ff7` (immutable fire-order resume),
`e9163f927d` (immutable custody-manifest rehash on resume).  Every Python edit received two visible
review-tracker passes; no `REVIEW_GATE_OVERRIDE` was used.  The final checks were:

- real-field pytest: **2 passed**;
- Ruff: clean;
- `py_compile`: clean;
- payload-retention gate on the builder: **0 findings**;
- `git diff --check`: clean;
- final full `run-00-02` replay: rc=0 in 0.59 s, reusing and validating immutable checkpoints.

The replay found two real resumability defects before the final clean pass: live available-memory
made a regenerated sealed request differ, then the historical manifest's run-time git head made a
regenerated manifest differ.  The fixes make both artifacts immutable and revalidate them byte by
byte on resume.  These were implementation defects, not measurement changes; the original payload
hashes and byte result stayed fixed.

## Stage receipts

### Stage 00 — fresh preregistered selection

Storage preflight selected `/Volumes/APDataStore/pact/ddm_qbw1_boundary_event_quotient/` with
76 GiB free at fire time; Vertigo had 8.3 GiB.  The builder required 12 GiB including the 8-GiB
reserve and failed closed if AP did not pass.  Source field size/SHA were rechecked as
117,964,800 B / `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.

NumPy `PCG64`, seed `20260827`, selected these 32 pair IDs from the ten temporal blocks crossed with
low/high Road↔Lane crack count:

`4, 31, 49, 52, 62, 90, 100, 113, 128, 148, 173, 179, 186, 187, 214, 236, 256, 260, 268, 278,
326, 328, 341, 352, 368, 382, 444, 456, 483, 508, 563, 573`.

Each stratum has `N_h=30`; blocks 0–5 use `n_h=2`, blocks 6–9 use `n_h=1`.  IDs, full stratum
membership, inclusion probabilities, and selected crack counts are in the checkpoint.  BD1's
diagnostic IDs were not force-included.

### Stage 01 — real grammar fit

All 32 source fields were converted to the declared quotient object.  Encoder bases, integrated
cells, Lane-event rasters, raw typed sections, the complete dictionary-fit source, four shared
models, and all 128 candidate records are retained.  The dictionary-free candidate won the exact
counted n32 race.

### Stage 02 — encode/decode and refusal

- 32/32 primary records parsed to their exact declared raw objects;
- 32/32 integrated base fields equal the encoder's Lane→Road base field;
- 32/32 repeat records are byte-identical to primary and to the winning stage-01 record;
- 129/129 one-bit mutations were refused: one shared model plus record framing and all three counted
  sections for each selected pair;
- every encoder/receiver cell array, source field, Lane raster, categorical observability field,
  base one-hot, boundary distance, and boundary tangent input is retained per pair;
- per-pair, per-stratum, and mutation JSONL query stores point back to their producing payloads.

Checkpoint SHA-256 values:

- stage 00: `86b9098c16ef5cde8c9f3ff4f92b258891d11ee0f71ad53c48d4a93f4abbb352`;
- stage 01: `686ab7f18b1f072d4ce6268ae7d9ba9362eb6fe393a96ce4a2fb4b7b42e810b7`;
- stage 02: `f7aa52feeb21cae946fd9da176094b03813124b454790cb5276ff7a383c4bcc4`.

## Independent audit and payload custody

The primary custody manifest contains **535 files / 8,075,206 B**, all rehashed successfully:

- `CUSTODY_MANIFEST.json`: SHA-256
  `ec497149314cdf168409117728e2f48e8c0f94247c073100dd1951ab244650fc`;
- `POSTRUN_AUDIT.json`: 11,001 B, SHA-256
  `a3e2c280106d246fb5bff2c9b837b4ddcf7e68e858ce1b9c37f12cc20334891b`;
- `SUPPLEMENTAL_CUSTODY_MANIFEST.json`: 495 B, SHA-256
  `c1499d92ab3e9ec54daab75e81eda7415de448484bc4310cb46bd36a862b7495`.

The independent audit re-decoded 32 packets, compared 32 repeats, re-executed all 129 mutation
refusals, checked all primary-manifest sizes and SHA-256 values, and recomputed the HT and component
arithmetic from file sizes.  It reproduced `336,270`, `336,286`, and `389,362` exactly.  No material
payload was deleted.  Only success-only atomic `.part` scratch and the storage write probe are
removed automatically.

## Sealed stage-03+ fire order

The sealed config is 2,075 B, SHA-256
`2f0012e843d22111c01c41ebaeded9f02fbd0548ba15981f1242e51f5a6d4811`; the fire order is 1,578 B,
SHA-256 `5fd3e62ec36d3c3efc0d590d788410f765a71e7709a992ceb95ddb85589ec51a`.

It binds real n32 geometry, chunk partition `30+2` from birth, 65 epochs, five-epoch periodic saves,
distinct stage boundaries, EMA retention, canonical `frame_utils.yuv420_to_rgb` GT decode, retained
frames before scoring, per-pair logits/argmax/Pose6, the exact GB1 archive control, and the complete
no2 §5 gate.  The WD3-derived conservative peak projection is 85.76 GiB under the 116-GiB ceiling;
the schedule projection is 3,931.73 s.  Both require live recheck by MAIN.

**Disposition: `FOLDED_BY_SCORER_FREE_GATE`.**  Receiver and memory prerequisites pass; the rate
prerequisite does not.  Owner `MAIN quotient-body joint-realizer operator` must not claim Metal or
scorer lanes and must not launch stages 03–05 for this packet.  This is the exact fire order's
refusal outcome, not an orphaned pending request.

## RECALL EVIDENCE

Sources searched before design included the full `.omx/research/` memo/receipt corpus, canonical
research indexes, `sub015_DAG_*` FEED blocks, design/SPEC files, task/live-state stores, the lane
registry, and the canonical equations registry from
`.venv/bin/python tools/list_canonical_equations.py --json`.  Content queries included:

- `quotient|boundary event|worldsheet|oriented crack|causal innovation|reset-record`;
- `Lane.*Road|Road.*Lane|lane carrier|lane band|LBND2|interface length`;
- `task-cell|score quotient|receiver-close|explicit address|implicit address`;
- `GB1|groupbin8|ba1f3830...|decoded_tokens_instrumented`;
- `QBW1|no2|NR1|WS0|WS1|D3|D3A|D3B|V14|BD1`.

Beyond the charter's named seeds, the search changed the build in four ways:

1. D3/D3A/D3B showed that Lane's entropy saving is real only when Lane is native; every separately
   carried raster/analytic/lossless Lane object either destroys Seg/Pose or spends the saving.  This
   forced Road-graph-anchored Lane events rather than a bitmap sidecar.
2. NR1/WS0/WS1 priced dense/task-cell/full-worldsheet forms far above the box.  This forced maximal
   crack-chain traversal plus integrated region seeds rather than direct crack or cell storage.
3. The canonical DM2 lane/IPM law and the existing AA-SDF/LBND receivers were found, but D3A closes
   importing their measured carrier bytes as if they transfer.  They informed local coordinates;
   no ancestor byte credit was booked.
4. No receiver-closed, independently reset QBW boundary packet matching this object was found in
   the searched scopes.  Therefore the schema was genuinely frozen and measured instead of citing
   an older worldsheet projection.  This is bounded absence, not global nonexistence.

## Boundaries and verdict scope

- **MEASURED:** real source-field n32 selection; all typed QBW1 v1 objects; real raw-DEFLATE reset
  packets; exact model race; HT bytes/interfaces; receiver equality; repeat identity; corruption
  refusal; storage, payload, and resume custody.
- **NOT MEASURED:** RGB realization, `R`, SegNet, PoseNet, `d_seg`, `d_pose`, `S_hat`, GB1 matched n32
  action, n600 bytes/distortion, runtime inflate, or contest CPU/CUDA score.
- **CLOSED(FIRST-RUNG FORMULATION):** QBW1 v1 reset snapshots with maximal base crack chains,
  canonical region seeds, ellipse/PCA Lane dash events, and the frozen raw-DEFLATE coder.  The rate
  miss is 251,376 B before distortion.
- **NOT A FAMILY KILL:** a new quotient grammar may reopen only by changing the rate-bearing object or
  proving a credible route to at least a 3.9605x complete quotient reduction under the same reset
  estimator.  A better compressor, smaller headers, or temporal diagnostic alone is not this proof.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — new quotient grammar derivation, not stage 03.** Owner: `MAIN
  object-family router`. Consumer store: a newly chartered versioned sibling of
  `/Volumes/APDataStore/pact/ddm_qbw1_boundary_event_quotient/`. Fire trigger: MAIN consumes this
  v1 rate closure, confirms no duplicate active lane, and preregisters a different rate-bearing
  object/schema with scorer-free arithmetic capable of removing at least 251,376 projected bytes;
  otherwise do not build or score it.

## LIVE-HYPOTHESES

- A shared topology grammar could replace the one-step-per-crack alphabet.  This remains plausible
  because the base-chain section is the largest semantic component at 173,370 projected coded bytes,
  but it must demonstrate nearly 4x complete quotient reduction rather than a local entropy gain.
- Lane may need curve/dash primitives shared across components instead of one PCA ellipse per
  connected component.  This remains plausible because Lane events cost 105,645 projected bytes and
  the canonical lane/IPM law gives reusable structure; D3A forbids treating an ancestor carrier's
  bytes or paint behavior as transferable.
- Joint realization may still solve the distortion wall for a future in-box packet.  This remains
  plausible because v14 closes fixed paint, while QBW's boundary/interior split is a different
  mechanism; the present packet never reached that experiment because rate already refused it.

## DEAD-ENDS

- Firing stages 03–05 on QBW1 v1: closed by the scorer-free cap, 389,362 B versus 137,986 B.  Training
  cannot repair a 251,376-B rate miss.
- Treating parseback success as admission: closed; all receiver checks pass but rate fails.
- Buying the result with a larger shared DEFLATE dictionary: closed within v1; every retained
  dictionary candidate is larger in complete n32 bytes than the 16-B no-dictionary model.
- Header-only optimization as the escape: closed by arithmetic; zeroing all 54,000 projected framing
  bytes still leaves the quotient 197,376 B over allowance.
- A temporal cross-record coder as a replacement for this verdict: closed by no2 §5; it may be a
  secondary diagnostic but cannot replace the independently reset primary estimator.
- Reintroducing explicit `(pair,x,y)` residuals or a separate Lane mask/carrier: closed by the rung-1
  schema and the inherited DF1/D3/D3A/D3B evidence.

Own-vehicle frontier: **gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]**.  UNMOVED.
