# ddm_jrx1 — fixed K24 control is incomplete; joint exchange remains unmeasured

**Verdict: INSTRUMENT_INCOMPLETE.** All 24 seeded reference searches finished. Twenty-two pairs produced resolved proposals; pairs 502 and 547 produced no admissible single anchor under pd4's unchanged screen, so their two-token search could not start. The fixed-K24 changed-token exchange is undefined. The preparation gate refused before any new price encode. This is neither a numerical tolerance failure nor a closure of the joint method. The three-row scientific probe remains incomplete.

Lane: `ddm_jrx1_joint_exchange_probe_with_scorer_slot_20260916`. Axis: **[macOS-CPU advisory, frozen CPU-torch SegNet/PoseNet, DALI lineage]**. `research_only=true`, `score_claim=false`, lane L0. Sole scorer slot released after every child was terminal. No frontier progress.

## Measured reference search

| Quantity | Final result |
|---|---:|
| Fixed seeded sample | 24 = 12 residual-pool + 12 complement; seed 20260916 |
| Single-token proposals screened | 435 |
| Single-token resolved rows | 215 |
| Two-token proposals screened | 161 |
| Two-token resolved rows | 121 |
| Pairs with resolved proposals | 22/24 |
| Pairs with no admissible single anchor | 2/24 |
| New real archive encodes | 0 |

Source: `FIELD_CONTROL.json`, with every pair's hashed DONE and reference counters. All eight final search-shard receipts are rc0. The earlier v1 generation failed a concurrent atomic-rename/census race; its bytes and source are retained separately. The race was repaired and regression-tested before v2, without altering the scientific helpers.

| Complement pair | Legal single proposals | Added SegNet errors, min / median / max | Allowed maximum | Pose refines | Two-token search |
|---|---:|---:|---:|---:|---|
| 502 | 32 | 6 / 10 / 14 | 2 | 0 | Unmeasured: no anchor |
| 547 | 30 | 5 / 9 / 14 | 2 | 0 | Unmeasured: no anchor |

These are exact observations for the declared 16 saliency-ranked cells and legal ±1 symbols. They do not establish that every single-token move fails, that every two-token move fails, or that the joint object is finished. We did not weaken the screen, substitute sample pairs, discard nulls, or give no-op rows a fabricated changed-token price. The reference cluster mechanism requires a surviving single anchor; a compensating second token behind a rejected anchor remains untested.

## The requested exchange rows and gate

| Row | Median S/B / IQR over K24 | Median bits/token / IQR | Status |
|---|---|---|---|
| 1. Field-only | undefined / undefined | undefined / undefined | Reference search complete; fixed-sample price input incomplete |
| 2. Renderer-only | unmeasured / unmeasured | unmeasured / unmeasured | Blocked by control |
| 3. Joint field + renderer | unmeasured / unmeasured | unmeasured / unmeasured | Blocked by control |

Joint/best-single ratio: **null**. Other-599 and all-600 collateral: **unmeasured**, not zero. Neither the ≥2x gate nor the <1.5x falsifier was tested. The charter's ±35B joint band was not exercised; real residual bytes would still be charged at 6.658589531221714e-7 S/B. Predecessor jrd1's int4 byte-only −4…+29B controls are not a newly measured renderer row.

`prices/MISSING_CONTROL_PAIRS.json` retains both nulls in the full sample. The source-reviewed preparation gate returned the expected rc1 refusal before materializing a price sheet. Encoder and winner-selection stages of `experiments/ddm_jrx1_price_control.py` remain unexecuted. Its real encoder reuses the canonical twin RLC1 implementation; it never charges ranking ledgers. The diagnostic tail archives would hold the shipped carrier, so resolved-pose diagnostics alone cannot be called a byte-closed candidate.

The second control defect is an estimand mismatch. Pd4's **12.0 bits/token** is a selected pooled 18-pair/24-token archive price. The charter asks for a random-K24 per-pair median. Re-reading the 12 overlapping pairs in pd4's retained frame-local winning rows gives a **13.731024070989292 bits/token median**, and a median resolved-pose-plus-seg credit of **2.987873520403345e-8 S/bit**. These are DERIVED predecessor statistics, not new jrx1 prices and not a valid numerical gate failure. Eleven of those twelve rows have zero seg-only benefit, so dropping pose from the numerator would also change the instrument. `PD4_REFERENCE_AUDIT.json` preserves the exact rows, edits and source hash.

## Form, provenance and boundaries

Reference form remained pd1 single search + pd4 cluster search, 16 cells, ±1 symbols, interior radius 3, ≤2 added seg cells, 12 refinements, radius 1 cluster neighborhood, and the actual `ddm_jg5_pose_resolve_on_edited_renders.refine_pair` solver at 40 outer rounds / 400 GN iterations. The only declared selection extension keeps nonpositive-benefit anchors in the fixed sample. No MECHANISM reduction or proxy scorer was introduced. The finite proposal search is not a claim of a global optimum.

`PROVENANCE_START.json` and `PROVENANCE_FINISH.json` pin and reverify all 14 charter/source anchors. `search_v2/BINDING.json` additionally pins the actual runtime, helper modules, scorer weights, GT caches, sample and base pose. The terminal verifier checked 66 top-level source facts plus nested raw/field/argmax identities. Runtime, upstream, receiver, prior, basis, shipped weights and predecessor stores remained read-only. Vertigo was not written. No training, burn, Modal, fire, packet, or authority evaluation ran.

The sample SHA is `b905cbabf06ca4a9dc78c1f9ed49a929e0c15d5f2b580a55fa769c23a5256613`. Base archive SHA is `ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`, 179,332 B. The pointer still recomputes to 0.13620226906030858 from d_seg 0.00010304, d_pose 4.21e-6 and 25B/37,545,489. At held distortion the strict sub-0.12 byte ceiling is 154,999 B, a 24,333 B cut. No candidate changed those quantities.

## Retention, resume and verification

Owned SSD store: `/Volumes/APDataStore/pact/ddm_jrx1`. Terminal census: **6,643 files, 616,401,965 logical bytes (0.574069 GiB), below 3 GiB**; AP free space 17,951,621,120B. Every path, byte count and SHA is in `RETENTION_MANIFEST.json`, including failed-v1 and regression payloads and filesystem metadata. Nothing was deleted. Local receipt/bundle metadata is separate from that SSD census.

The terminal verifier passed: 737 losslessly retained frame payloads, 2,249,329,896 reconstructed raw frame bytes, 391 solver-result checksums across v1+v2, 174 completed-pair artifacts, exact K24 v2 DONE coverage, and v2 solver-result/checksum coverage. The 26 total pair DONE records include 2 old-generation records; they do not expand the fixed 24 denominator. There were no orphan frame payloads and no unchecked v1 solver results.

Complete proposal solves are immutable disk checkpoints keyed by source binding, frame bytes, starting codes and solver configuration. An interrupted individual solve restarts from its pinned proposal inputs; completed solves are reused. An interrupted JSONL is retained in a distinct attempt directory, never mistaken for a completed pair. Heavy search/verification launches used the detached helper, named receipts and requested niceness 0; the sandbox could not set niceness, so best-effort launch status is recorded and child code asserted actual niceness 0 before work. `ALL_LAUNCHES.json` and `DONE_RECEIPTS.json` hold the commands, storage reports and terminal receipts.

The observed retention race is covered by the real-archive 16-task/8-process regression: 136 exact payloads passed, including forced reimports and same-destination contention. Ruff passed all three source files. Two visible review-tracker passes cover each Python file; the final verifier also received independent source review. **Correction:** early progress-only `*.sha.json` file counts included ExFAT AppleDouble sidecars. They were never used in a gate or score and are superseded by the parsed 336 resolved rows and the final checksum verification. See `PROGRESS_COUNT_CORRECTION.json`.

## Follow-on dispositions

- **FIRED:** jrd1's scorer phase was consumed by this charter, through the completed fixed-sample reference search. Its original task remains blocked because the scientific joint probe is incomplete.
- **QUEUED-WITH-A-FIRE-ORDER:** MAIN owns `ddm_jrx1_matched_control_repair_20260916`, consumed through `CONTROL_REPAIR_PROPOSAL.md` and this SSD store. Fire on harvest of this receipt: settle matched calibration and null-pair handling before any replacement prices are selected.
- **FOLDED:** the renderer/joint continuation remains in `ddm_jrd1_joint_exchange_scorer_phase_20260916`. Fire only after the repaired Row 1 gate passes, the current pointer validates and the sole scorer slot is assigned again. No burn is authorized.
- **FOLDED:** attempted scheduling split was abandoned when process-identity inspection was denied; no signal was sent and no duplicate pair was launched.

No formulation/family negative is banked from K24. The specific 62 rejected single proposals are retained instance observations. The compensating neighborhoods in the repair proposal, and the field/int4 joint direction with all 600 collateral, remain live hypotheses because neither was measured.

## RECALL EVIDENCE

The original corpus search preceded scientific work. Exact queries, result hashes and bounded search counts are in `RECALL_SEARCHES.json`; source-by-source findings and changes to the plan are in `RECALL_EVIDENCE.md` (included in this delivery). Surfaces searched were the full research memo/receipt corpus by content; all 490 canonical equation rows; research index/DAG FEED surfaces; design/SPEC files; and the task ledger. Beyond the charter seeds, JF1/RJ2 prevent a false claim that no joint mechanism was previously tried; XR1 separates same-input encoder determinism from across-object container variation; terminal pd5 changes the live queue and price context without moving the pointer; PC2's older S1 closure does not transfer to this object. The compensated semantic exchange and union-not-sum laws forbid proxy charges or additive composition claims. The pd4 source/row audit exposed the selected-pool versus random-median mismatch and the need to keep pose in the numerator.

## Landing and operational custody

Commit is attempted LAST through the required serializer, with post-edit SHA per owned file, two Python review passes, `[no-triality] [p0-ledger-ok]`, and no attribution trailer. `SERIALIZER_RESULT.json` and its retained output are the outcome authority. If rc17 occurs, the verified bundle and format-patch under `serializer/` are for MAIN to land; rc17 is not treated as a scientific stop. Shared initially dirty state files are not whole-file staged; `OPERATIONAL_ROWS.json` preserves this arm's live rows for review. The shared index is checked against its start hash.

composition S 0.13620226906030858 @ 179,332 B [contest-CUDA T4 n600] (move 52), unchanged.
