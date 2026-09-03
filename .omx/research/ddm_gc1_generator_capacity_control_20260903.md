# GC1: explicit generator capacity is FORMULATION-CLOSED

Date: 2026-09-03  
Axis: `[macOS-CPU scorer-free exact byte measurement]`  
Verdict scope: **FORMULATION** — GF1's four-stream analytic generator followed by the GC1 all-class, non-overlapping recursive dyadic block-paint overlay at maximum depth 7.

## Result first

**CAPACITY-CLOSED.** The clean full-n600 run measured all **4/4** preregistered physical capacities, with no subset and no missing point. No point passed either gate:

- `packet <= 71,404.5 B AND mismatches <= 46,804`; or
- `packet + actual domain-matched residual <= 85,020 B`.

The best replacement accounting was penalty 32: **53,277 B packet + 348,260 B residual = 401,537 B**, which is **316,517 B over** the 85,020 B cap (4.723x the cap). The largest admissible measured packet was **76,113 B = 1.5989118x** the 47,603 B charter reference and still had **725,965 mismatches**. Therefore `scorer_fire_order` is `null`: no scorer, archive build, Modal call, or pointer move is owed from this result.

Authority artifacts:

- strict result: `/Volumes/VertigoDataTier/pact/ddm_gc1_generator_capacity_control/final/RESULT.json` — 140,581 B, SHA-256 `57e5a0a86328dfadb826c6aa3fc91b47dba6cb3d87f9dbdcf387aea35a2abd12`
- strict manifest: `/Volumes/VertigoDataTier/pact/ddm_gc1_generator_capacity_control/final/MANIFEST.json` — SHA-256 `d150232e154b3fc0ecd0e2c840a8add18188e9400f4a2b0535ecff854be82605`
- manifest census: **365 files / 2,584,642,532 B**, inventory SHA-256 `4ea29f4256f1792c2269ae6808cc801599064162aa76fa01d921c2653e9d05b5`
- exact target for every closed receiver: 117,964,800 B, SHA-256 `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`

An independent manifest pass rehashed all 365 listed files, recomputed the inventory hash, and found no missing or extra file. The result-first structure and typed negative follow the evidence/handoff rules in [The Operating Manual — a craft handoff](../../docs/operating_manual_craft_handoff.md).

## Capacity control and receiver

The implementation is [experiments/ddm_gc1_generator_capacity_control.py](../../experiments/ddm_gc1_generator_capacity_control.py), SHA-256 `afcd0c95e070d39dae324e6bc93e84412597db77dcb175a1153976ef961cbb4a`.

GC1 preserves GF1/HG1's four real fitted streams (`road_undrivable`, `lane`, `movable`, `mycar`) and adds one counted receiver-consumed basis:

1. A retained atom paints one canonical class over one node of a recursive 2x2 dyadic partition.
2. Depths 0 through 7 are available, so the finest atom is 3x4 sites on each 384x512 pair field.
3. The scalar capacity control is an integer **mismatch-site penalty per retained atom**.
4. Each pair is solved by an exact bottom-up dynamic program over `leave GF1`, `paint one class`, or `recurse to four children`; ties minimize mismatches and then atom count.
5. Canonical non-overlapping atoms are Morton-addressed, delta-ULEB encoded with their class label, and raced through Brotli q11, zlib-9, and LZMA2-extreme.
6. The receiver parses the physical integer packet and applies those exact atoms after decoding GF1. The compact v2 generator envelope is 20 B; the redundant 96 in-packet hash bytes from the first prototype were removed because every retained section already has an external SHA receipt.

This is not a float parameter-count proxy. Every table byte below is the size of a retained physical packet. The fit budget was the same deterministic full-n600 categorical fit at every point. The only declared scope reduction is scorer-free target-field Hamming fit: that is the exact pre-scorer quantity named by the charter and avoids taking QBR1's device/scorer lane. There was no mechanism reduction.

## Capacity-to-bytes and mismatch-to-bytes curve

All rows are `[macOS-CPU scorer-free exact byte measurement]`, full n600, denominator 117,964,800 categorical sites. `Packet/reference` uses the charter's 47,603 B GF1 reference. The measured baseline is 47,971 B because it is freshly fitted and framed against the exact JBP1 null field rather than inheriting GF1's earlier lb1 packet.

| Penalty | Atoms | Overlay raw B | Overlay coded B | Packet B | Packet/reference | Total mismatches | Road | Lane | Undriv | Movable | MyCar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0 | 0 | 47,971 | 1.007731x | 1,334,939 | 639,336 | 319,147 | 262,741 | 1,331 | 112,384 |
| 32 | 3,931 | 10,431 | 5,286 | 53,277 | 1.119194x | 985,100 | 373,770 | 327,885 | 134,788 | 38,503 | 110,154 |
| 12 | 13,365 | 32,202 | 17,102 | 65,093 | 1.367414x | 822,610 | 281,062 | 296,762 | 99,500 | 44,017 | 101,269 |
| 8 | 24,486 | 57,061 | 28,122 | 76,113 | 1.598912x | 725,965 | 246,382 | 275,034 | 86,579 | 40,542 | 77,428 |

The endpoint buys only a **1.83885x** total-mismatch reduction, not the required 28.31x. Road and Undriv improve by 2.59x and 3.03x, but Lane improves only 1.16x. More seriously, the class-blind paint basis turns Movable from 1,331 errors into 40,542 errors at the endpoint, a 30.46x regression. This identifies why added block capacity is not translating into a useful whole-field correction budget.

The clean calibration and verdict roster contains exactly baseline plus penalties 32, 12, and 8. Its maximum is 1.598912x; no 1.8x-or-larger fit or packet exists under `final/`.

## Domain-matched residual price

Each point retained and raced the same eight HG1 residual orders (`frame_raster`, `class_frame_raster`, `tile8_time`, `tile16_time`, `tile32_time`, `tile64_time`, `class_tile16_time`, `pair_tile16`) through all three real coders, including deterministic coded repeats. `tile64_time + lzma2_extreme` won every row. Every winning residual was wrapped with its generator packet, decoded, applied, and proved byte-identical to the JBP1 target.

| Penalty | Mismatches | Generic 0.2909 B/site projection | Actual residual B | Actual B/mismatch | Actual minus generic B | Packet + actual residual B |
|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1,334,939 | 388,333.76 | 359,280 | 0.269136 | -29,053.76 | 407,251 |
| 32 | 985,100 | 286,565.59 | 348,260 | 0.353528 | +61,694.41 | **401,537** |
| 12 | 822,610 | 239,297.25 | 343,128 | 0.417121 | +103,830.75 | 408,221 |
| 8 | 725,965 | 211,183.22 | 340,552 | 0.469103 | +129,368.78 | 416,665 |

The generic price changes sign from optimistic to severely optimistic as capacity rises. Although the endpoint removes 608,974 mismatches, the actual residual shrinks only 18,728 B because the remaining sites cost more per mismatch. Thus the real hybrid curve bottoms at penalty 32 and then worsens. The diagnostic `GC1C` receiver-closed container adds 116 B of framing beyond the charter's `packet + coded residual` line; excluding that framing is favorable to the candidate and does not affect the negative.

## Recursive-fractal power law

The registered fit over the four physical packets is:

`log(mismatches) = 27.2633600712 - 1.2284382471 * log(packet_bytes)`

- log-space `R^2 = 0.9155651`
- extrapolated packet at 46,804 mismatches: **686,617.61 B**
- extrapolated crossing / 71,404.5 B direct cap: **9.61589x**
- extrapolated crossing / 47,603 B GF1 reference: **14.42383x**

The crossing is an extrapolation, not a measured packet or family theorem. The endpoint's less-than-2x mismatch reduction agrees with the charter's operational prior. The literal local exponent is steeper than `-1`, however, so GC1 does **not** claim that the prior's separate “slower than inverse capacity” exponent statement was validated. Either way, the fitted crossing is far beyond both byte gates.

## Typed decision and boundaries

- **Verdict:** `CAPACITY-CLOSED`.
- **Scope:** this GF1 + class-blind depth-7 recursive dyadic block-overlay formulation only. This is not a family-wide death for anisotropic, class-protected, learned, or curve-domain recursive generators.
- **Candidate denominator:** 0/4 points; 4/4 measured; 0 missing.
- **Scorer fire:** none; `scorer_fire_order=null`.
- **Measured:** physical packets, receiver-decoded categorical fields, per-class/total mismatches, eight-order/three-coder actual residual prices, deterministic repeats, exact residual closure, and the four-point log fit.
- **Not measured:** `d_seg`, `d_pose`, archive bytes, contest score, CPU/CUDA parity, or token-to-argmax amplification on this vehicle.
- **Calls:** 0 scorer, 0 Modal, 0 Metal/MPS, 0 contest evaluations.
- **Mutation boundaries:** `upstream/` unchanged; `submissions/semantic_joint_ctxmix/` unchanged; no staged-index operation; no payload deletion.

The first parent-root calibration attempt used an overly broad exploratory penalty grid and materialized calibration-only packets above the charter's 1.8x prohibition. Those bytes were retained and independently manifested rather than hidden or deleted, but that attempt is **non-authority**. The clean authority rerun is only `/final`: it used exactly the safe roster and independently reproduced the same four-row result. Transport interruptions were recovered only through atomic, distinct stage outputs and source-recovery receipts. A detached-wrapper attempt recorded `PermissionError: Operation not permitted`; its launch manifest/PID/empty log are retained inside the clean manifest, and no detached completion was claimed.

Final verification after the two genuine source-review passes:

- `ruff format --check`: pass
- `ruff check`: pass
- `py_compile`: pass
- focused pytest: **7 passed**
- strict `check_no_measure_and_discard_payload`: **0 findings**
- independent 365-file hash/inventory audit: pass

## RECALL EVIDENCE

The charter seeds were treated as a floor. Before choosing the form, the search covered `.omx/research/` bodies and receipts, the canonical equation registry, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design/SPEC files, task-ledger/hot-state surfaces, and the live GF1/HG1/JBP1 sources.

Content queries included:

- `generator form|capacity control|target-independent|mismatch.*bytes|residual`
- `recursive|fractal|quadtree|hierarchical|static.*dynamic|born.*generator`
- `GF1|HG1|HZS1|generator_form_fit_error_entanglement`
- `ddm_gc1|ddm_wc3|QBR1|ddm_fpc3` for duplicate/live-owner checks

The canonical-equation JSON was enumerated and narrowed to `generator_form_fit_error_entanglement_v1`, `v8_geometric_rate_decomposition_v1`, `partition_temporal_transport_amortization_jitter_bound_v1`, `procedural_predictor_plus_residual_correction_savings_v1`, and `generic_shared_helper_vs_individually_fractal_negative_amplification_v1`.

Beyond the charter's named seeds, recall found:

1. [GF1 capacity-gap decomposition](ddm_gf1_capacity_gap_decomposition_20260831.md) had already shown the existing residual order optimal and isolated all-class downstream support. That changed GC1 from another stream-order sweep into an all-class receiver-consumed overlay while still reracing every actual residual order at each new point.
2. [HZS1 horizon shape](ddm_hzs1_horizon_shape_20260831.md) had already swept 17 through 512 horizon knots and stayed near the same floor. That ruled out a duplicate horizon-only capacity knob and caused GC1 to own downstream support jointly.
3. [Gestalt generate-vs-serialize pincer](ddm_gestalt_generate_vs_serialize_pincer_20260831.md) supplied the relevant generate/serialize distinction and prevented a packet-only win from being confused with a field-fit win.
4. [NSCS06 v8 class-stream optimal-technique supersession](nscs06_v8_cls_stream_optimal_technique_pv_supersession_landed_20260526.md) measured a 95,754 B Shannon lower bound for an exact quadtree on a different 48x64 class-stream object. That did not close GC1's lossy post-GF1 problem, but it warned against an exact dense quadtree and changed the design to penalized lossy atoms plus a separately priced residual.
5. [RN1's n600 reopen sweep](ddm_rn1_n600_reopen_sweep_20260903.md) and the live board showed QBR1/WC3 and FPC3 already owned training-side and neighboring work. GC1 therefore stayed CPU/numpy, scorer-free, and did not dispatch or duplicate them.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_gc1_generator_capacity_control/future_changed_form/`; fire trigger:** a source-level, receiver-consumed anisotropic/curve-domain recursive generator is named that protects sparse existing classes and has closed-form packet arithmetic placing at least one real coded point at or below 71,404.5 B. On that trigger, claim a new lane, rerun this same full-n600 packet/mismatch/residual gate, and do not request a scorer until the gate passes.

## LIVE-HYPOTHESES

- A class-protected anisotropic or curve-domain recursive basis may shift the curve enough to merit this pre-scorer gate. It is plausible because GC1's square atoms reduce Road/Undriv strongly but barely reduce thin Lane error and create roughly 39,000 new Movable errors; this is the signature of the wrong atom geometry and class-blind overpaint, not proof that every recursive generator is bad.
- Joint generator/residual context may be more valuable than more class-blind atoms. It is plausible because the same `tile64_time` residual order wins all four points while actual residual cost falls only 18,728 B despite 608,974 fewer errors; the remaining errors become systematically harder to describe. This must first be duplicate-checked against FPC3 before anyone builds it.
- The local capacity response is real but attached to a fatal intercept. It is plausible that a changed form can shift that intercept because the fitted exponent is `-1.228`, yet this form's extrapolated crossing remains 9.62x over the direct cap; more capacity in the same form is not the experiment.

## DEAD-ENDS

- More capacity in this exact class-blind depth-7 dyadic overlay is closed: the largest legal point is already 1.5989x, misses the mismatch bar by 679,161 sites, and the extrapolated crossing is 686,618 B.
- The generic `0.2909 B/site` correction price is closed as the sole decision price: it underprices the endpoint residual by 129,369 B.
- Reordering the current HG1 residual is closed for these four fields: all eight named orders and all three coders were raced, and `tile64_time + lzma2_extreme` won every point while remaining hundreds of kilobytes over the cap.
- A scorer or contest archive for GC1 is closed: 0/4 points passed the pre-registered byte gate, so firing a scorer would turn a failed means-gate into fake progress.
- The parent-root broad calibration is closed as evidence authority because it crossed the prohibited range. Only the independently reproduced `/final` result and manifest are consumable.

`[contest-CUDA T4 n600]` own-vehicle frontier **UNMOVED**: AFR1 — `S=0.14797617125559104`, archive `180,002 B`, `d_seg=0.00020139`, `d_pose=6.37e-6`, SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`; GC1 measured no score.
