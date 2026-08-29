# ddm_qbz1 descent-rate configuration adjudication — supervised field fit complete; realized ceiling queued

**Task:** #1324  
**Date:** 2026-08-29  
**Status:** `PARTIAL — SCORER-FREE FIT COMPLETE; REALIZED CAPACITY CEILING QUEUED`  
**Axes:** source rows `[macOS-MPS governed n32 research row; not contest authority]`; new fit `[macOS-CPU scorer-free native-field advisory]`; queued terminal `[macOS-CPU frozen-scorer advisory]`  
**Score claim / promotable:** `false / false`

## Conclusion

The full landed QBF1/qbt2b schema was fitted at real n600 from the r10 EMA state, directly against the registered DALI GT partition, with the inherited counted FP1 palette as the renderer target. The exact quantized packet has **native partition error 0.0141554381 on the 23,591,640 held-out pixels and 0.0141491818 on the 94,373,160 train pixels**. The heldout-minus-train gap is only **0.00000625629**, so this single-seed fit did not spatially overfit its direct target. A separate 120-pair holdout control, measured before those pair latents were admitted, was 0.0232724508.

The fitted object projects to **122,062 B**, only 134 B above r10's 121,928 B. Its rate term is 0.0812760755, leaving 0.0668419237 distortion to tie gb1 or 0.0387239245 to cross 0.12.

That is not yet the charter's capacity ceiling. No frozen scorer ran in this arm. Therefore no capacity/optimization fork is claimed: native logits and palette error are not a substitute for the required real render → camera R → uint8 → frozen SegNet/PoseNet realization. The scorer terminal is sealed as `QUEUED_WITH_FIRE_ORDER`; the latest fcd3 row is terminal, but this charter forbids launching or claiming the scorer lane from the arm.

## R7–R10 source re-derivation

Every `S_hat` reproduced exactly from `100*d_seg_hat + sqrt(10*d_pose_hat) + 25*B_hat/37,545,489`; no figure disagreed with its retained GATE.

| run | B_hat (B) | d_seg_hat | d_pose_hat | distortion recomputed | rate recomputed | S recomputed | abs error vs stored |
|---|---:|---:|---:|---:|---:|---:|---:|
| r7 | 122,574 | 0.0131352742513 | 0.00182862221072 | 1.44875398357 | 0.0816169953200 | 1.53037097889 | 0 |
| r8 | 122,325 | 0.00459289550781 | 0.000996510556168 | 0.559114926121 | 0.0814511964407 | 0.640566122561 | 0 |
| r9 | 122,171 | 0.00305875142415 | 0.000957764018092 | 0.403740561134 | 0.0813486541619 | 0.485089215296 | 0 |
| r10 | 121,928 | 0.00251833597819 | 0.000575745612061 | 0.327711500535 | 0.0811868504363 | 0.408898350971 | 0 |

All four rows retain `estimator_status=NO2_SECTION5_HT_COMPLETE`, `selection_count=32`, and `control_status=REFUSED_MISSING_REAL_SAME_BUDGET_QBW1_CONTROL`. Their three unmet gates are `d_pose_hat`, `s_hat`, and `same_budget_qbw1_control`. Their levels are not comparable to gb1; only the retained trajectory is consumed here.

Source receipt: `/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration/SOURCE_REDERIVATION.json` (8,001 B).

## Supervised full-schema capacity fit

### Exact object and split

- Warm start: r10 EMA checkpoint, SHA-256 `bf0a3a64b3f9ff59e1662d1c9676aa8c249f1d32738a8ed9cf967625e08c2f75`.
- DALI partition: `/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy`, SHA-256 `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248`; `assert_gt_lineage(..., required=AUTHORITY_LINEAGE)` passed before reading.
- Renderer target: inherited video-derived FP1 palette, SHA-256 `19e6524b75724f0b19f0e2e49a827d9f28b40d087b1e5504c3a85577a9e76f0b`.
- Frozen ABI: all 44 r10 state tensors, all 600 pair-latent records, and the existing four-section QBF1 packet. No shape, tensor, section, codec, or receiver rule changed.
- Seed: 20260829. Optimizer: AdamW, LR 2e-4, weight decay 0, deterministic CPU algorithms.
- Phase 1: two epochs over 480 deterministic train pairs; 120 disjoint pairs withheld entirely. This tests shared-field generalization without fitting the heldout pair latents.
- Phase 2: ten epochs over all 600 pairs; a deterministic pair-specific 1/5 pixel mask withheld from both native CE and palette MSE. This is the primary real-n capacity split because every per-pair latent is admitted while 23,591,640 target sites remain unseen by the loss.
- Loss: unweighted native-class CE to DALI GT plus 8× last-frame RGB MSE to the FP1 palette indexed by that same GT. No SegNet, PoseNet, MPS, Modal, or contest evaluator was loaded or invoked.
- Resumability: 12 distinct epoch-end checkpoints plus periodic 300-update checkpoints, with model, optimizer, RNG, cursor, config, and history. Total fit time including encoding/evaluation was 2,398.779 s; this is the measured runtime, not a smoke projection.

### Measured scorer-free rows

| row | scope | native d_seg | train native d_seg | heldout sites | packet B_hat |
|---|---|---:|---:|---:|---:|
| pair generalization control | 120 never-trained pairs after 2×480 updates | 0.0232724507650 | N/A | 23,592,960 pixels | 121,936 B |
| primary spatial holdout | n600 quantized packet after 10 full-population epochs | **0.0141554381128** | **0.0141491818225** | **23,591,640 / 94,373,160** | **122,062 B** |

The spatial gap is negligible, but the 1.415% native partition level is not itself an evaluator distortion and cannot decide the fork. The direct target fit is single-seed; no family claim transfers until the realized terminal lands, and any fork-driving single-seed result must retain that boundary.

## Typed exit

| required field | disposition |
|---|---|
| schema capacity ceiling, realized through real path | **NOT MEASURED** — queued; scorer-free native field is retained but is not the ceiling |
| fork verdict | **NOT ADJUDICATED** — no rounding into either branch |
| fork scope | single-seed full-schema qbt2b probe; family closure forbidden until n600 frozen-scorer realization |
| alignment decomposition | **MEASURED from retained r7–r10 telemetry**, scoped to integrated objective values, not gradients or LR vectors |
| named next measurement | MAIN-owned n600 realization of the exact retained packet, 30-pair chunks, all frames/scorer outputs retained |

## Objective-alignment cross-check

The sister trainer's 81.19% phase-budget claim does **not** transfer literally. Qbt2b used only 10 CE-birth updates followed by 5,000/15,000/10,000/10,000 margin updates: CE shares were 0.1996%, 0.06662%, 0.09990%, and 0.09990% for r7–r10. Its configuration issue is inside the margin objective, not a long CE phase.

The table below integrates the scalar components stored at every margin update. These are **objective-value shares**, not gradient norms, cosine alignment, or LR-budget shares.

| run | realized Seg share | native-interface Seg share | constraint share | pose share |
|---|---:|---:|---:|---:|
| r7 | 35.30% | 49.20% | 13.45% | 2.06% |
| r8 | 38.35% | 42.72% | 14.59% | 4.34% |
| r9 | 33.41% | 50.11% | 11.32% | 5.16% |
| r10 | **30.97%** | **54.67%** | **9.10%** | **5.27%** |

The specific configuration change is: **use the retained supervised packet as the warm start, then A/B the scorer finish with the native-interface coefficient changed from 100 to 0 while realized Seg remains 100, keeping pose and constraints unchanged**. Nominally this changes realized Seg from 50% to 100% of the two Seg coefficients and removes the component that consumed 54.67% of r10's integrated objective value. It does not justify importing CW1's 13.6069× allocation factor, the CE ancestor's 81.19%, or the architecture-specific 92.7% surcharge. The predicted effect is narrower: if the native surrogate is competing with realized Seg, the zero-native finish should reach any realized-distortion milestone in fewer equal-cost steps. That prediction remains unmeasured until the capacity terminal chooses the optimization branch, after which the exact measurement is a same-start, same-seed `100/100` versus `100/0` steps-to-target A/B.

## Custody and reproducibility

- Fit result: `/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration/FIT_RESULT.json`, 433,162 B, SHA-256 `69b33e5d393deff7f1fcd76844cf524d7c19691f431aa399a876b2ad1ce227bf`.
- Final checkpoint: `/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration/checkpoints/final_end.pt`, 1,218,485 B, SHA-256 `2fd239d1883f2395ab3f0dc085638c7003a09347beb346c48d397d0931362959`.
- Exact retained coder container: `/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration/final_reencode/reencode_payloads.tar`, 2,723,840 B, SHA-256 `4c16e6c045768b2dee62f59ac9a2a27b7386280dfccff3dd5331a8d9509d95f7`; 489 members / 2,375,652 logical member bytes.
- Inner complete archive: `archive.zip`, 106,832 B, SHA-256 `0e2ffdfaa5fe481d481dd70a9672a67f80b9aad7648f0c775fe2956dd3a4841d`; parse-back exact. The HT reset projection is 122,062 B.
- Final native field: 20 retained NPZ chunks, 1,132,509,166 B total, carrying fp16 logits, uint8 argmax, DALI targets, and uint8 two-frame RGB for all 600 pairs.
- Pair-holdout field: four retained NPZ chunks, 261,015,341 B total, carrying the same payload classes for all 120 heldout pairs.
- Total qbz1 custody at completion: 141 files / 1,443,898,170 logical bytes. Nothing was deleted; AP had 16,665,216 KiB free after the run.
- Realization fire order: `/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration/REALIZATION_FIRE_ORDER.json`, 1,654 B, SHA-256 `262cab0c4535255d470e05b2f0f75088575f7c3233a5e4d31593e96018a4f55f`.

## RECALL EVIDENCE

I searched beyond the charter seeds before adjudication:

- Canonical equations: `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for `qbt|capacity|configuration|allocation|descent|alignment`. The CE softmax/mirror-descent and weak-KAM power-law entries were relevant context; no registered qbt-specific capacity equation was found.
- Full memo/index/DAG/ledger content searches: `qbt2b|qbflow|capacity ceiling|configuration|allocation|alignment|expected_flip|81.19|92.7|task #1250` across `.omx/research/`, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, and task-ledger/status surfaces.
- Beyond-seed findings consumed: `ddm_cw1_corrected_window_20260817.md` makes the 92.7% figure an architecture-specific surcharge rather than a total wall and records a 13.6069× allocation result with borrowed-LR/tau confounds; `src/tac/gt_lineage_registry.json` supplied the registered DALI n600 partition and pose-target artifacts; the live lane ledger showed fcd3 owned the scorer slot at launch and later closed terminal as `refused_seg_band`.
- Negative-control sources re-read: `ddm_na10_negative_audit_fresh_laws_20260819.md`, `ddm_pk4_optimal_form_frame0_pose_20260813.md`, `ddm_ny1_live_lineage_toy_and_reactivation_audit_20260823.md`, and `ddm_w96a_aligned_config_renderer_window_20260826.md`. They changed the design to GT-not-gb1 supervision, real n600, explicit pair and pixel holdouts, and a single-seed boundary.
- Task #1250's owning memo was not found in the bounded research/index/DAG/ledger scopes searched. I treated its conditional as unowned and did not use it as evidence.

## Boundaries

- The new 122,062 B is an HT packet projection for the fitted object, not a bankable independent rate win.
- No realized `d_seg`, `d_pose`, distortion, or `S` exists for this fitted object yet.
- No exact contest CPU/CUDA evaluation ran; no pointer moved.
- The scorer-free native field is useful evidence about representation and heldout fit, but using it as the family ceiling would be a fake completion.
- AP is at 100% displayed capacity despite 16.7 GiB free; the queued terminal retains several GiB and must repeat the 5,000,000,000-byte preflight.

## NEXT_IF_RESUMED

- **QUEUED_WITH_FIRE_ORDER** — owner: `MAIN local scorer scheduler`; consumer store: `/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration/REALIZED_RESULT.json` plus `/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration/realized_n600/`; fire trigger: verify fcd3's newest row remains terminal, append a fresh active `ddm_qbz1_scorer_20260829` `local_macos_cpu` claim with no other newest active scorer claim in the prior 24 hours, and pass the AP ≥5,000,000,000-byte preflight; then execute the exact command in `REALIZATION_FIRE_ORDER.json`.

## LIVE-HYPOTHESES

- **Optimization remains plausible.** R7–r10 descended monotonically across all retained score components, and the supervised field has essentially no spatial generalization gap. A real realization below 0.30 would falsify family-level capacity closure.
- **The native surrogate may be consuming descent budget that should go to realized Seg.** It accounts for 42.7–54.7% of integrated margin-objective value across r7–r10 while realized Seg falls to 31.0% at r10; a `100/0` scorer finish is a clean same-object test after realization.
- **The FP1 palette may transfer the learned DALI partition through SegNet well enough for the tie branch.** It is video-derived and was already the landed qbt2b renderer initialization, but only the queued render/R/uint8/SegNet terminal can establish this.

## DEAD-ENDS

- **Treating the r10 level as pointer-comparable is closed.** The n32 HT row has three unmet gates and the same-budget control is explicitly refused.
- **Using gb1's realized field as teacher is closed at the consumed formulation scope.** It is a noisier copy of GT; the direct DALI target was available and used.
- **Calling the native 0.0141554 row the realized capacity ceiling is closed.** It has not passed R/uint8/SegNet or PoseNet.
- **Transferring the CE sister's 81.19% or CW1's 13.6069×/92.7% figures numerically is closed.** Qbt2b's phase shares and confounds differ; this memo uses only its own stored component telemetry.
- **A reduced-n or in-sample capacity verdict is closed.** The probe used n600 plus independent 120-pair and 1/5-pixel holdouts; neither native row is promoted beyond its declared surface.

**Own-vehicle frontier remains:** S = 0.7539807296911207 @ 357,836 B `[macOS-CPU advisory]` n600 (qo1 `sub_auto_pairbit`); qbz1 did not move it.
