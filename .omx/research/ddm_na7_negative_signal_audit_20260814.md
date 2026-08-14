# DDM NA7 — MC36 negative-signal audit

Tags: `[no-triality] [p0-ledger-ok]`  
Date: 2026-08-14  
Authority: read-only receipt audit plus one typed consumer-mapping correction; no scorer, evaluator, trainer, paid dispatch, or payload materializer was run

## Result first

The negative corpus does not contain a valid new super-band add-on to MC36. The exact effective frontier remains MC36 Variant C at **MEASURED `S=0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]`**, archive SHA-256 `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`. The remaining gap to `0.15` is **DERIVED `0.0119344578804448 S`**. This audit moved no pointer and is not goal progress.

The strongest reusable result is a routing correction. JS8's “train with the Road-hub gate active” hypothesis is **not** encoded by the current RX2 run. RX2 learns a probability model for the fixed MC36 token stream and requires token/raw identity; it can improve bytes, but cannot change the rendered Seg field. I wrote the required typed correction at:

`/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/NA7_JS8_GATE_AWARE_CONSUMER_ADJUDICATION.json`

The JSON is valid, 2,018 bytes, SHA-256 `d83e14f65ea70c6afe9333da535bb1b0bca0b6ecaa0242a87c0dc130e46ec9c9`. It folds the false RX2 consumer mapping and queues the hypothesis under a separately registered, resumable JS1/#982 joint receiver treatment. The MT1/#978 multi-token hypothesis was already encoded correctly: its sealed T4 sign gate remains queued, and no train stage is authorized unless `positive_t4_sign=true`.

## Authority and method boundary

The governing charter SHA-256 is `ff0ef0f6bf3d2640d1a5bccdde82ecb024324618eabf5c989c137c78fc8f5cee`; the common contract SHA-256 is `eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771`. The final read of `main_hot_state.md` used SHA-256 `ad5f33c66b738e95fea6650b62f56cf6aa4ea329eca68f70aaf179e44129a79f`.

Labels below mean:

- **MEASURED**: an existing retained receipt on the stated instrument and axis.
- **DERIVED**: exact arithmetic from measured components.
- **TOY-BRACKET**: arithmetic mixes objects or instruments and therefore only supplies a bound, never an admission row.
- **INSTANCE / FORMULATION / FAMILY**: the m69 verdict-scope ladder. No row below earns FAMILY death.
- **Route A**: JS1/#982 joint distortion training. **Route B**: CPU/decode axis. **Route C**: RX2 rate training. **DEAD**: no current consumer.

The m94 test is explicit in each row's instrument column. Subset instruments state their sampling defect under m88/m96. A prefix result is never promoted to a population verdict. No new score was measured; all candidate payloads referenced here pre-existed and remain under their original custody.

### Source register

Every regrade below was checked against its source receipt or the later memo that consumed it. The compact mapping is:

| Row IDs | Primary source receipts |
|---|---|
| P01-P03 | `.omx/research/ddm_pk3_frame0_pose_representation_20260813.md`; `.omx/research/ddm_pk4_optimal_form_frame0_pose_verdict_20260813.md`; `.omx/research/ddm_ps135_pass4_exact_row_harvest_20260812.md` |
| P04-P06, S07-S08 | `.omx/research/ddm_qs4_collateral_suppression_20260813.md`; `.omx/research/ddm_qs5_verdict_and_no_toy_enforcement_20260813.md`; `.omx/research/ddm_re1_round1_dual_axis_verdict_20260814.md`; `.omx/research/ddm_qs3_saturation_compose_20260813.md` |
| P07, S03, S06 | `.omx/research/ddm_js8_implicit_edge_conditioning_20260814.md`; `.omx/research/ddm_js8_seg_stack_compensated_rerun_20260813.md`; current JS8 retained `ADJUDICATION.json` under `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/js8/` |
| S01-S05 | `.omx/research/ddm_js1c_cuda_custody_stage0_verdict_20260814.md`; `.omx/research/ddm_js6b_pose_screened_compile_20260813.md`; `.omx/research/ddm_ec2_oriented_adapter_trainer_20260814.md`; `.omx/research/ddm_bg1_bilinear_gate_pricing_20260814.md`; `.omx/research/ddm_bg2_postmortem_execute_20260814.md` |
| R01-R04 | `.omx/research/ddm_rx1_rate_representation_attack_20260814.md`; `.omx/research/ddm_rx2_mc36_label_hpac_20260814.md`; `.omx/research/ddm_lp135_lossless_pack_20260810.md`; current RX2 retained training/checkpoint root |
| C01-C04 | `.omx/research/ddm_eu4_fresh_eyes_fractal_composition_20260813.md`; `.omx/research/ddm_rfo1_fresh_hybrid_compose_20260814.md`; `.omx/research/ddm_mc35_micro35_union_build_20260814.md`; `.omx/research/ddm_mc36_micro35_variants_20260814.md`; `.omx/research/ddm_mc36_dual_axis_t4_verdict_20260814.md` |
| D01-D03 | `.omx/research/ddm_f26p_runtime_cpu_lift_20260814.md`; `.omx/research/ddm_f26q_rc64_native_lowering_20260814.md`; `.omx/research/ddm_f26r_hpac_hot_stage_final_rung_20260814.md`; `/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/receipts/result.json` |
| I01-I06 | `.omx/research/ddm_mt1_978_multitoken_screen_20260814.md`; `.omx/research/ddm_fs1_fire_seal_adapters_20260814.md`; `.omx/research/ddm_dt1_repeated_lesson_determinizer_20260814.md`; `.omx/research/ddm_ac1_automatic_endpoint_closure_20260814.md`; `.omx/research/ddm_mc36_promotion_complete_s_verdict_20260814.md` |
| B01-B10 | `.omx/research/ddm_hv1_fresh_eyes_hybrid_review_20260813.md`; `.omx/research/ddm_po1_t4_error_feedback_pose_compensation_20260813.md`; `.omx/research/ddm_pz4r_full_n600_eval_20260813.md`; `.omx/research/ddm_js7_exact_row_verdict_20260812.md`; `.omx/research/ddm_cp5v_compose_five_validated_events_20260812.md`; `.omx/research/ddm_jo1_t4_row_adjudication_20260813.md`; `.omx/research/ddm_sa1_shipping_axis_seg_actuator_20260813.md`; `.omx/research/ddm_hc1_hy1_container_push_20260812.md`; `.omx/research/ddm_ec3_t4_targeted_events_20260813.md`; `.omx/research/ddm_tf1_theoretical_floor_and_beyond_20260812.md`; `.omx/research/ddm_pz4a_precision_preproof_20260811.md` |
| B11-B15 | `.omx/research/l28_engineered_corrections_witness_remeasure_negative_20260701.md`; `.omx/research/ddm_lv2_terminal_campaign_completeness_20260811.md`; `.omx/research/ddm_gs1_gestalt_signal_census_20260814.md`; retained PR136 source snapshot named there |

## Seed-row regrade

The 32 rows in this table are the charter's deduplicated seed surface plus the exact MC36 and apparatus descendants needed to decide whether the signal is still live.

| ID | Negative or mixed receipt | Narrow verdict and actual instrument | Portable cure / reactivation precondition | Consumer and disposition |
|---|---|---|---|---|
| P01 | PK3 linear frame-0 overlay | **FORMULATION** for linear overlays fitted on a biased toy `n9`; `23/23` in-sample winners became `0/23` LOPO winners. The instrument cannot support a population claim. | Only a nonlinear jointly trained representation on a seeded random population changes the precondition. | Route A, **FOLDED** into the joint-representation class; no PK3 retry. |
| P02 | PK4 optimal-form overlay | **FORMULATION** for linear frame-0 overlays on a seeded random `n64` split (`48` train/`16` heldout): LOPO-positive but heldout-negative at the useful 42 B and 1,000 B rungs; 250 B was heldout-neutral. | Learned nonlinear frame-0 conditioning with matched full-population validation. | Route A, **DEAD as linear overlay**. Memo commit `31074f0ad6`. |
| P03 | PS135B carrier | **INSTANCE** on its exact retained `187,223 B` archive and `[contest-CUDA T4, n600]` row: `d_pose=0.00014674`, about `21.3x` CP135 pose. The `-0.0083 S` pose ceiling is unclaimed headroom, not a candidate. | Change the learned pose representation and train through the shipping receiver. | Route A, **FOLDED**; do not recode this carrier. |
| P04 | QS4 stale Schur result | **INSTANCE / stale instrument**: post-export compensation used stale derivatives and left roughly `+2.4e-4` pose damage. | QS5 changed the precondition by solving compensation inside the actual exported compile. | Route A, **CURED-AS-MECHANISM** by QS5; QS4 bytes are dead. |
| P05 | QS5 near miss | **INSTANCE** on a same-worker n600 component instrument: archive `+26 B`, `17` fewer Seg flips, `d_pose=6.88500995e-6 < base`, but `Delta S=+2.519822e-6`. | Reuse only in-compile compensation on a different object whose Seg/rate economics already pass. | Route A, **FOLDED-AS-CURE**, not banked as a score win. |
| P06 | RE1 pose leakage | **INSTANCE** on its exact n600 T4 component row: `0 B`, `2` fewer Seg flips, `Delta d_pose=+8.108145266e-10`, `Delta S=-1.206873849e-6`. | It was consumed into the MICRO35/MC36 union; only a fresh whole-object rebuild can reuse the technique. | **FOLDED / CONSUMED** by MC36; no additive bank remains. |
| P07 | JS8 pose damage | **INSTANCE** on current JS8's local CPU n600 object: `Delta d_pose≈+1.64e-4`. Pose compensation was not applied, but Seg plus rate is already positive. | Only train the gate inside a joint Seg/Pose receiver; post-hoc compensation cannot repair the failed Seg/rate scale. | Route A, **REFUSED current object**. |
| S01 | JS1C/T1R1 fixed linear per-edge controls | **FORMULATION** on `[contest-CUDA T4 component, n600]`; every tested fixed edge control has `rho < 0.827795`. This measures realized correction per requested correction, in the claim's units. | Relinearize after accepted states or learn a receiver-native conditional interaction. | Route A, **DEAD as fixed linear per-edge**; joint nonlinear treatment remains open. |
| S02 | JS6B pose-screened event compiler | **FORMULATION** for its enumerated `200`-proposal alphabet: `0/200` proposals survived the registered joint gate. The alphabet census is complete; it is not a family census. | Coupled/multi-token support or learned amplitude changes the proposal class. | Route A/#978, **FOLDED** into MT1; no singleton rerun. |
| S03 | older JS8 200-event stack (2026-08-13) | **INSTANCE / FORMULATION** on an exact n600 T4 census of the full 200-event alphabet: `26` beneficial, `38` positive flips total, `38` neutral, `136` harmful, signed total `-166`. Even perfect compensation cannot cure the `962`-flip shortfall to its 1,000-flip gate. | Learn implicit joint distortion rather than selecting the same events post hoc. | Route A, **FOLDED**; do not reuse this alphabet. |
| S04 | EC2 uniform-trained conditioner | **FORMULATION** on the terminal n600 field: `40,779` fewer original errors were outweighed by later JS8's mechanism census of `12,075` fixed and `52,854` introduced, including `42,184` GT-Road collateral. The tested receiver was never trained with the gate. | Gate-aware joint training is the cure; the missing interaction must be consumed during training. | Route A, **REACTIVATED as a new training formulation**, not an EC2 refire. |
| S05 | BG1/BG2 4x8 bilinear gate | **FORMULATION**. BG1 priced a real small module, then BG2 used full n600 labels: exposure-only OOF `R2=0.427685`; adding existing 8-D context reduced it to `0.409237` (`-0.018448`, `p=0.897810`). | A different interaction, support, or jointly learned representative is required. | #978, **DEAD as this 4x8 gate**. |
| S06 | current JS8 post-hoc Road-hub gate | **INSTANCE** on archive `188,018 B` (`+1,749 B`) and local CPU n600: `50,388→50,381`, only `7` flips, Seg `-5.93397e-6 S`, rate `+0.001164587 S`, before pose. | Consume the gate during joint training, not after uniform training. | Route A, **REFUSED_LOCAL_ADMISSION**; commit `c6cd5d6ecf4d6a0033e2851061af8fd3c1e0f457`. |
| S07 | QS3 collateral model | **FORMULATION / support-scoped** on QS2's `189` changed pixels: `97.4%` realization, `57.1%` beneficial, failure dominated by collateral. It is not a JS8 or population B-rate. | Select gates with an explicit collateral cost on the new object's own support. | Route A/#978, **LESSON CONSUMED** by BG2/MT1; no transferred percentage. |
| S08 | QS4/QS5 17-flip concordance | **FORMULATION** for the exact three-pair support: both implementations top out at `17` net flips on different supports. | Expand or learn support; changing the compensation solver alone does not change the ceiling. | Route A, **DEAD for the three-pair support**. |
| R01 | frozen TQ1C/IHS1 transfer | **INSTANCE / FROZEN_TRANSFER**. Exact parse/identity passed, but the cross-label model produced `191,746 B`, `+5,477 B` versus MC36; token/model/container deltas were `+5,462/+15 B`. | Retrain the probability state on exact MC36 labels. | Route C, **CURED-IN-PROGRESS** by RX2; no frozen rerace. |
| R02 | RX2 weak-count contexts | **FORMULATION / WEAK_COUNT_CONTEXT**: best tested weak count was `305,035 B`. By m94 this instrument counts weak contexts; it does not bound a learned IHS1 model. | Train the actual IHS1 probability object and serialize the complete archive. | Route C, **NON-KILL**. |
| R03 | RX2 epoch-1 signal | **MIXED / training-surrogate only**: `144,906 B = 27,026 model + 117,880 token` is below MC36's payload scale, but is not a serialized whole archive or authority row. | Terminal EMA, exact token/raw identity, RC64 encode, deterministic repeat, and whole-container count. | Route C, **FIRED_EXISTING_OWNER_NO_DUPLICATE**. |
| R04 | task #996 coder reraces | **FORMULATION** for unchanged symbol/probability state: same-state coder work is closed. | Reopen only after RX2 produces a new probability state. | Route C, **FOLDED** into the RX2 terminal identity race. |
| C01 | EU4 additive bank | **FORMULATION / bank-only projection**: QS2 plus RE1 was about `-5.6e-6`, below the `1e-5` naming bar and not a same-object union. | Build, parse, and remeasure one union. | **CONSUMED** by RFO1→MC35→MC36; it is not additive residue on MC36. |
| C02 | RFO1 MICRO35 specification | **DESIGN-ONLY**: the projected `<=-1e-5` depended on `>=35` distinct flips, `<=+29 B`, and a pose cap. | MC35 built it; MC36 repaired the failed union into Variant C. | **CONSUMED**; no second MICRO35 build. |
| C03 | MC36 Variant A | **INSTANCE** on the receiver-closed local n600 component instrument: `35` flips and `+61 B`; pose passed, rate failed. | Reduce container bytes without losing distinct support. | **DEAD exact bytes**; its support lesson was consumed by Variant C. |
| C04 | MC36 Variant B | **INSTANCE** on the same local instrument: `37` flips and `+40 B`; pose/rate gate failed. | The five-byte repack and support repair changed the precondition. | **DEAD exact bytes**; superseded by Variant C. |
| D01 | F26P Python decode | **INSTANCE / decode implementation**. Full Modal decode took `2,933.2 s`, exceeding the budget while preserving exact tokens. | Native lowering; do not kill the model. | Route B, **CURED-BY-ENGINEERING**. |
| D02 | F26Q native RC64 | **INSTANCE**. Full M5 decode took `203.843 s`; measured cross-axis projection was `1,709.2 s`, still above the `1,600 s` fire gate. | Lower the hot context/class stage. | Route B, **CURED by F26R**; no F26Q exact fire. |
| D03 | F26R hot-stage lowering | **MIXED / ready**. M5 `147.005 s`, projected Modal `1,321.647 s`, all identity and scalar twins pass. Its own receipt says the arm did not fire the exact row; current hot state records the contest-CPU closer as in flight. | Harvest the already-fired row; never duplicate it. | Route B, **FIRED_EXISTING_OWNER_NO_DUPLICATE**. |
| I01 | MT1 queue starvation twice | **APPARATUS INSTANCE**, not a scientific negative: `$0`, no scorer result, sealed commit `c9d6d62c` remains valid. | Fire the sealed T4 sign gate when the higher-priority lane is free. | #978, **QUEUED_WITH_A_FIRE_ORDER**. |
| I02 | nonconformant MC36 fire seal | **APPARATUS INSTANCE**. FS1 found and corrected request/schema drift before launch. | Reuse the real dispatcher loader and sealed content hashes. | **CURED / CONSUMED** by MC36 exact fire. |
| I03 | MT1 Modal image phase order | **APPARATUS INSTANCE**. FS1 repaired the image setup order without changing the scientific payload. | Keep the repaired sealed request. | **CURED**; no reseal absent source drift. |
| I04 | MT1 spawn-metadata signature / call registration | **APPARATUS FORMULATION** for the current dispatcher adapter: the worker could start while metadata registration failed. AC1 preserved endpoint custody, but the signature mismatch itself remains a recurrence risk. | Add a focused dispatcher compatibility test and require post-spawn call-ID registration. | **OPTIMIZE**, apparatus owner; no scorer needed. |
| I05 | claim-override and dual-ledger drift | **APPARATUS FORMULATION**. MC36 needed late reconciliation because the paired launcher and axis claims did not close atomically. | DT1's strict claim-before-fire, dependency, and dual-ledger gates. | **CURED / FOLDED**; audit the next paired fire, do not reopen science. |
| I06 | terminal endpoint orphaning | **APPARATUS INSTANCE**. The closer had to recover a terminal result after the initiating arm lost custody. | AC1's idempotent endpoint closure and retained-result recovery. | **CURED**; only act on a new orphan receipt. |

## Beyond-seed dry-loop rows

The loop surfaced 15 additional findings whose old status could have created a phantom branch. Each was resolved against its later consumer rather than re-litigated.

| ID | Finding | Regrade and signal extraction | Current disposition |
|---|---|---|---|
| B01 | PO1 Round-2b local-Jacobian/int16 correction | **FORMULATION** on an exact T4 component instrument: `d_pose` worsened `8.257448x`, same-object `Delta S=+0.015602170862732995`; no third T4 fire under its registered F2 gate. | **DEAD** as this one-shot local-J formulation. Exact-feedback-per-step or learned pose is a different family. |
| B02 | PZ4R direct-v6 | **INSTANCE** on matched macOS-CPU n600: `-3,115 B` vs CP135 but `d_pose=0.6310142278671265`, total worsened `+2.471539547510437 S`; T4 literals were not mixed into that delta. | **DEAD exact archive**. A new jointly learned pose payload may reuse apparatus, never weights/numbers. |
| B03 | JS7 44-event stack | **INSTANCE** on exact T4 n600: `186,575 B`, `S=0.16342603740620176`, `+0.00147089912796 S`; projected Seg sign did not survive stack scale. | **DEAD exact events**; lesson is complete-object joint selection. |
| B04 | CP5V five-event compose | **INSTANCE** on exact T4 n600: equal bytes, about `-4.0e-6 S` Seg and `+6.03e-6 S` Pose, net `+2.03e-6 S`. | **DEAD exact events**; closes independent per-axis admission. |
| B05 | JO1 six-event compose | **INSTANCE** on exact T4 n600: `+1 B`, `S=0.1621711682636563`, about `+0.00021603 S`. | **DEAD exact events**; B/robust-flip ranking survives only as a proposal prior. |
| B06 | SA1 Stage-1 EMA | **INSTANCE / train-local-gate-T4 formulation**: exact T4 Seg `34,970→34,970`, `+926 B`; changed field but zero net distortion. | **DEAD exact candidate**; true CUDA-in-loop training is outside scope. |
| B07 | HY1/HC1 literal C1 head | **FORMULATION** on exact T4 n600: all tokens decoded, but `S=0.4044688071472634`; failure is rendered realization, not addressability. | **DEAD direct representative**; C1 may be a teacher for Route A. |
| B08 | EC3 minimal one-token/one-pair support | **FORMULATION**: `2/200` locally eligible, optimistic `3.97595776e-6 S`, `54.3265x` below fire bar. | **DEAD minimal support**; coupled support is MT1. |
| B09 | TF1 full-resolution global-xi raster transport | **FORMULATION**: `453,449 B` versus `356,636 B` intra (`1.27146x`), every class losing. | **DEAD dense transport**; sparse learned support remains distinct. |
| B10 | PZ4A adaptive absolute precision | **FORMULATION**: at most `500 B` coefficient saving with a `2,732 B` allocation map. | **DEAD absolute per-cell map**; grouped/map-free rate-aware QAT is held behind a `>=2,000 B` parser-equal preproof. |
| B11 | isolated L28 free receiver transform | Old witness row is only **INSTANCE** scope on a different receiver: Seg slightly worse, pose-blind metric better; isolated current MC36 terminal remains unmeasured. | **HELD**, Route A/B boundary. A repeat-identical zero-counted-byte receiver preflight is required before one governed A/B. |
| B12 | WeightEntropyPenaltyMLX | Historical Torch `lambda=50` showed `-16,007 B` live entropy but roughly `+0.038 d_seg`; neither transfers to MC36. | **HELD**, Route C/A. Reopen only with a matched resumable current-object A/B and complete archives. |
| B13 | original MC35 union | Receiver-closed but `+44 B` and pose-failing; its support and repack lessons produced MC36 C. | **DEAD exact bytes / CONSUMED**. |
| B14 | PR136 adaptive order-0 coder | **FORMULATION**: retained code is a per-tensor adaptive order-0 histogram, not a context model; current sections already close same-state order-0/coder work. | **DEAD as a new MC36 mechanism**; exact binary retrieval is custody-only. |
| B15 | HV1/RFO1 broad branch list | GS1's route-stratified top-30 read graded HV1, RFO1, CN4, BG2, and PZ4 findings as consumed; the only live scientific routes were already EC2/JS1, MT1, and MC36. MC36 is now terminal. | **FOLDED**. No duplicate branch was created; L28 and matched rate-in-loss remain explicitly held above. |

## Composition check on the MC36 base

The exact score atoms at the current denominator are:

```text
one Seg flip = 100 / (600*512*384)
             = 8.477105034722222222e-7 S

one archive byte = 25 / 37,545,489
                 = 6.658589531221713479e-7 S
```

The historical QS2+RE1 arithmetic is **DERIVED `-5.5817878142738076e-6 S`**, missing the `1e-5` naming bar by `4.4182121857261924e-6 S`. It is also no longer a bank: MC35/MC36 consumed the supports, repack, and compensation route into the current `f0ba...` object. Adding it again would double-count ancestry.

MC36 C itself supplied a real super-band move against CP135: **MEASURED exact complete-score `Delta S=-2.068039779696e-5`**. That win is already the pointer. It is an anchor, not an add-on found by NA7.

### QS5 compensation on JS8

Current JS8 has an exact local Seg-plus-rate lower bound before pose:

```text
Delta S_seg+rate
  = -7*(100/117,964,800) + 1,749*(25/37,545,489)
  = +0.001158653335486372... S
```

Even an oracle that removes all JS8 pose damage at zero bytes leaves the object about `+0.00115865 S` worse. QS5 cannot change that conclusion: its proven mechanism lives on a different three-pair object and that tested object is itself `+2.519822e-6 S` worse. A naive “add only QS5's 26 bytes” calculation gives **TOY-BRACKET `+0.001175965668267548... S`** and is deliberately not called a candidate. The current JS8 object is closed without a scorer run.

### QS3 collateral selection on JS8/MT1

QS3's `57.1%` beneficial rate is instrument-scoped to QS2's `189` changed pixels. Transferring it to JS8 would violate m94 and the subset law. Later work did test the portable lesson—condition selection on collateral—on fuller instruments:

- BG2 found the 4x8 bilinear gate had no held-out predictive gain.
- MT1 changed heldout Road→Lane errors `306→297` but total heldout Seg stayed `1,529→1,529`, while local pose worsened by `+0.000102902`.
- Current JS8's full n600 post-hoc gate found only `7` net flips for `+1,749 B`.

Therefore the collateral lesson is consumed, but no percentage composes into MC36. The only verdict-changing reactivation is a newly trained gate-aware receiver (Route A), not re-selection of the retained JS8 tables.

### Newly surfaced rate and decode rows

RX2 epoch 1 could be super-band by rate if the terminal object preserves its surrogate compression after model/container serialization and identity closure. It cannot be composed today because `144,906 B` is a training surrogate, not an archive. F26R changes runtime only; it may make an existing archive admissible on CPU but contributes no unmeasured score atom. L28 and WeightEntropyPenaltyMLX likewise have no same-object MC36 component receipt. They remain held rather than projected.

**Composition verdict:** zero unconsumed same-object candidates at `|Delta S| >= 1e-5`. All apparently qualifying sums either double-count MC36 ancestry, mix instruments, or depend on an unmaterialized terminal object.

## JS8 consumer encoding verification

### Gate-aware joint training

JS8 named RX2 as the live consumer. Source inspection refuted that mapping. The current RX2 trainer consumes fixed MC36 token labels, optimizes a probability model, and requires token/raw identity. It cannot consume a gate that changes the rendered Seg distribution. The typed adjudication receipt records:

- `disposition=FOLDED_AS_RX2_CONSUMER_MISMATCH`;
- current RX2 may claim only rate evidence;
- current RX2 must not claim a test or cure of JS8 gate-aware distortion training;
- reactivation is `QUEUED_WITH_A_FIRE_ORDER` under the JS1 joint trained-receiver line/task `#982`;
- first action is scorer-free whole-container price, identity, resume, and payload-retention preflight.

This is a consumer correction only. It moves no score and authorizes no training from this audit.

### Joint multi-token conditioning

MT1 correctly encodes the second hypothesis. Its exact `1,270 B` module is retained; the seeded random-stratified local split is `n32`, not a population verdict; the heldout total Seg tie and pose loss prevent local admission. The sealed T4 sign gate at commit `c9d6d62c` is still valid after two queue-starvation events. The train-stage design is conditional: it must remain suppressed unless the T4 receipt explicitly writes `positive_t4_sign=true`. No new store row was needed.

## Ranked ITERATE / OPTIMIZE order

| Rank | Action | Why it survives | Owner, consumer, fire order | Cheapest `$0` verdict-changing probe |
|---:|---|---|---|---|
| 1 | Harvest F26R CPU exact row | Runtime cure is sealed and the current state records the closer in flight. | **FIRED_EXISTING_OWNER_NO_DUPLICATE**; MAIN CPU closer; F26R returned-artifact store then hot state; harvest before any duplicate. | Read terminal provider/retained result, verify archive/runtime hashes and wall clock, close both ledgers idempotently. |
| 2 | Harvest RX2 terminal IHS1 | It is the only live row with enough rate scale to matter directly; epoch 1 is encouraging but non-authority. | **FIRED_EXISTING_OWNER_NO_DUPLICATE**; RX2 harvester; `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac`; terminal EMA and identity race first. | Serialize the terminal EMA, re-encode exact fixed tokens, prove raw/token equality, deterministic repeat, and whole-container bytes before any scorer. |
| 3 | Price and seal a real gate-aware JS1 treatment | JS8 refuted post-hoc gating, not joint training; this is the precise changed precondition. | **QUEUED_WITH_A_FIRE_ORDER**; JS1/#982 owner; `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge`; after RX2 terminal harvest and a distinct lane/registry entry. | Build a no-scorer identity/whole-container/resume preflight that proves the gate is consumed inside Seg/Pose training and that every candidate/stage is retained. Falsifier: gate is bypassed or the incremental counted object is unaffordable before training. |
| 4 | Run MT1 T4 sign gate | Local n32 is too small and axis-limited; the already-sealed T4 gate is the narrow valid discriminator. | **QUEUED_WITH_A_FIRE_ORDER**; MAIN #978 router; MT1 retained `t4_sign_gate_r1`; after higher-priority exact rows are terminal and hashes match. | Reverify sealed request/source/archive hashes and lane freedom. Falsifier: `positive_t4_sign != true`; then suppress training. |
| 5 | Repair dispatcher spawn-metadata compatibility | The scientific payload can run while endpoint registration fails, risking lost custody. | **OPTIMIZE**; dispatcher/AC1 maintainer; dispatcher tests plus call-ID/claim ledgers; before the next paired fire. | Replay a fake returned call through the real loader and assert one call-ID row, one claim, and idempotent terminal closure. |
| 6 | Screen isolated L28 on MC36 | It is zero-counted-byte generic receiver code and was never isolated on this terminal. | **HELD**; current-terminal receiver owner; `.../free_receiver_treatments/l28/`; only after active exact rows finish. | Build repeat-identical incumbent/L28 children, prove unchanged archive bytes and actual receiver consumption. Falsifier: any identity/runtime failure; otherwise queue one matched A/B. |
| 7 | Reopen rate-in-loss only on a matched MC36 pair | Historical entropy movement is large, but historical distortion damage does not transfer. | **HELD**; current-vehicle rate-in-loss owner; `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/rate_in_loss/`; after a terminal matched resumable pair exists. | Search for an existing equal-parameter checkpoint pair and serialize both complete archives. If none exists, there is no `$0` verdict-changing probe. |
| 8 | Map-free grouped QAT | PZ4A killed the explicit allocation map, not all rate-aware quantization. | **HELD**; PZ4-QAT successor; #984 rate branch; only after a parser-equal retained object saves at least `2,000 B`. | Byte-only parser-equal archive build with no per-cell map. Falsifier: `<2,000 B` whole-container saving. |

## Clean-round seal and recall evidence

Recall was content-first. It covered all seven indexed stores—research, equations, memory, DAG, council, tasks, and docs—plus the canonical task-status/bridge, hot state, source memos, and retained SSD/APDataStore receipts. Initial queries targeted the named pose, Seg, rate, decode, MC36, and apparatus rows. Later rounds deliberately searched `negative/refused/mixed/near-miss`, JS8/RX2/MT1 consumer joins, RFO1/MICRO35, and older unconsumed rows whose base/receiver/budget preconditions changed.

Findings beyond the charter seeds changed the answer in four ways:

1. Current JS8 and older JS8 are different receipts; both are negative, for different reasons, and neither may be summarized as a family kill.
2. RFO1's MICRO35 and EU4's bank were already consumed by MC35→MC36; they are not live addends on MC36.
3. HV1's initially open PZ4R row had a later full-n600 matched-axis receipt that closes the exact archive through pose collapse. Its other broad routes were either consumed by RFO1/MT1/CN4 or remain explicitly held as L28/rate-in-loss.
4. GS1's route-stratified top-30 census found no fifth live route beyond the then-live EC2/JS1, MT1, and MC36 set. MC36 has since become the exact pointer; EC2/JS8 are terminal negatives at their stated scopes.

The penultimate dry query found RFO1 and HV1. After resolving their later descendants, the final full query returned only already-consumed RFO1/EU4/errata rows, standing historical regrade surfaces, and the same known held hypotheses. It produced **zero new eligible rows**. This satisfies the charter's loop-until-dry seal in the bounded seven-store plus direct-receipt scope. The claim is not global nonexistence.

No candidate or scorer payload was materialized by NA7. The only new durable object is the small typed consumer-adjudication JSON; it contains no score/payload and lives on the APDataStore tier. The shared worktree and pre-existing index were not mutated except for this memo through the required serializer.

Exact effective frontier: **UNMOVED** at MC36 Variant C, `S=0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]`.  
Own-vehicle frontier: **UNMOVED** at LC2, `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **Disposition: FIRED_EXISTING_OWNER_NO_DUPLICATE; owner: MAIN F26R CPU closer; consumer store: `/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/` plus `.omx/state/main_hot_state.md`; fire trigger: the existing contest-CPU call becomes terminal with retained result, archive/runtime hashes, and wall clock.** Harvest and close it idempotently; do not fire another row.
- **Disposition: FIRED_EXISTING_OWNER_NO_DUPLICATE; owner: RX2 harvester/finisher; consumer store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac`; fire trigger: the governed training writes its terminal EMA/stage receipt.** Run the exact fixed-token identity and whole-container RC64 race, retain deterministic repeats, and report rate only.
- **Disposition: QUEUED_WITH_A_FIRE_ORDER; owner: JS1/#982 joint trained-receiver owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge`; fire trigger: RX2 terminal rate is harvested and a separate registered resumable treatment proves the Road-hub gate is consumed during joint Seg/Pose training with per-stage and per-candidate retention.** Run the scorer-free identity, price, and resume preflight first; do not treat current RX2 as this test.
- **Disposition: QUEUED_WITH_A_FIRE_ORDER; owner: MAIN #978 scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/`; fire trigger: higher-priority exact rows are terminal, the lane is free, and sealed hashes match.** Run only the T4 sign gate; train only if `positive_t4_sign=true`.
- **Disposition: OPTIMIZE; owner: dispatcher/AC1 maintainer; consumer store: the dispatcher compatibility tests plus `.omx/state/modal_call_id_ledger.jsonl` and `.omx/state/active_lane_dispatch_claims.md`; fire trigger: before the next paired/remote fire.** Fix and test the spawn-metadata signature so one launch produces exactly one registered call, claim, and idempotent terminal closure.
- **Disposition: HELD; owner: current-terminal receiver-treatment owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/stage0_v14/free_receiver_treatments/l28/`; fire trigger: active exact rows are terminal and repeat-identical MC36 incumbent/L28 children prove unchanged archive bytes, runtime closure, and actual L28 consumption.** Queue one governed same-terminal A/B only after that preflight passes.
- **Disposition: HELD; owner: current-vehicle rate-in-loss owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/rate_in_loss/`; fire trigger: an equal-parameter, resumable MC36 checkpoint pair exists and both complete archives can be retained and recounted.** Admit only a complete-score win; historical entropy numbers do not transfer.
- **Disposition: HELD; owner: PZ4-QAT successor; consumer store: #984 rate branch; fire trigger: a retained parser-equal byte-only archive saves at least `2,000 B` without a per-cell allocation map.** Only then consider resumable rate-aware in-loop QAT plus exported-object compensation.

## LIVE-HYPOTHESES

- A jointly trained Road-hub interaction may work where current JS8 failed because JS8 applied a scalar gate after uniform training; changing the training distribution is the exact untested precondition, and the current RX2 rate learner does not test it.
- RX2's terminal IHS1 may deliver a material rate win because epoch 1 put model plus token surrogate bytes below MC36's payload scale; it remains plausible only until whole-container serialization, exact token/raw identity, and deterministic RC64 repeats are measured.
- MT1 may reverse its local sign on T4 because its random-stratified n32 screen is too small for population authority and prior rows show CPU/CUDA component drift; the sealed sign gate is designed to decide only that narrow question.
- Isolated L28 may supply a small zero-counted-byte terminal correction because the old negative used a different pose-blind witness and co-applied another transform; plausibility ends if the MC36 receiver cannot consume it byte-identically within runtime.
- A matched current-object rate-in-loss treatment may trade bytes for distortion more favorably than the historical Torch row because it changes learned model/probability state rather than the already-closed coder; no historical effect size transfers.
- Map-free grouped QAT may survive PZ4A's metadata failure because an existing deterministic structural partition could supply precision classes without a 2,732-byte allocation wire; the `2,000 B` parser-equal preproof is the plausibility gate.

## DEAD-ENDS

- Re-adding QS2, RE1, HP4, or the RFO1 MICRO35 projection to MC36 is closed because MC36 already consumes that ancestry; doing so would double-count signal.
- Applying QS5 compensation to current JS8 is closed as a rescue: even perfect zero-byte pose repair leaves JS8 about `+0.00115865 S` worse from Seg plus rate alone.
- Transferring QS3's `57.1%` beneficial rate to JS8 is closed because it is scoped to 189 QS2 pixels; BG2, MT1, and current JS8 already consumed the collateral-aware lesson on their own instruments.
- PK3/PK4 linear frame-0 overlays, PS135B's exact carrier, QS4/QS5's tested three-pair objects, JS1C fixed linear controls, JS6B/EC3 singleton proposals, BG1's 4x8 gate, older JS8's 200-event alphabet, and current JS8's post-hoc gate are closed at their stated instance/formulation scopes.
- PO1 Round-2b, PZ4R direct-v6, JS7, CP5V, JO1, SA1, HY1/HC1 direct C1, TF1 dense transport, PZ4A's per-cell precision map, MC35, and MC36 A/B are closed at their stated scopes by retained receipts; none is a family kill.
- Frozen TQ1C transfer, weak-count RX2 bounds, unchanged-state coder reraces, same-state ANS/RC64 hunting, PR136 adaptive order-0 transfer, and bank-only EU4 arithmetic are closed as claims about the current MC36 rate object. Only a new trained probability state reopens rate.
- Treating F26P's 2,933-second miss as a model kill is closed: F26Q/F26R demonstrate that decode implementation was the failed precondition.
- Treating MT1's two queue-starvation events as scientific evidence is closed: they cost `$0`, measured no score, and did not invalidate the sealed request.
- Claiming that RX2 tests JS8's gate-aware distortion hypothesis is closed by the typed consumer adjudication: RX2 reproduces fixed tokens and can only test rate.
- Claiming NA7 as goal progress is closed: it fired no exact row and left both the effective and own-vehicle frontiers unmoved.
