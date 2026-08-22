# DDM NL1 — the parked-lever population was mostly a vehicle-custody defect

**Date:** 2026-08-22  
**Arm:** `ddm_nl1_never_fired_levers`  
**Authority:** scorer-free retained-receipt and append-only ledger audit; one new CPU-only checkpoint-support measurement  
**Disposition:** `DRAINED__31_WRONG_VEHICLE_RETIRED__10_LIVE_REGISTRY_ROWS_MEASURED__4_RETIRED__4_QUEUED`  
**Pointer:** **S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**, DX2 archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; NL1 did not move it.

## Answer first

The charter's two population labels are real, but its raw event count had drifted. At the pre-NL1
ledger snapshot there were **287 rows: 279 `fired`, 7 `measured`, and 1 `built`**, not 275 firings.
The stable scientific denominator is the distinct-name join: **37 of 41 fired lever names had no
measurement anywhere (90.24%)**. At event granularity, **276 of 279 firing rows** and **188 unique
`(lever, run_ref)` firings** lacked a same-run measurement. The charter's **37** is therefore
confirmed; only its raw firing count was stale.

The source-derived replacement-trainer denominator was **18 flags: 6 fired and 12 never fired** at
the 2026-08-17 snapshot. Full-corpus recall found three later retained treatments that the ledger
had not ingested (`FRD077`, `Q3Q4OFF`, `F1SIG1`) and six retained verdicts for the already-fired
flags. After append-only repair, those 18 rows are **10 measured, 4 retired with evidence, and 4
queued under two exact fire orders**. There are **zero non-retired fired-without-measurement names**.

The prior-law prediction is **confirmed**. **31/37 (83.78%)** fired-unmeasured names belonged only
to the retired levelset/v7.5/v9 lineage or its apparatus. The remaining 6/37 and all 12/12 source-
derived flags belong to the PR130 semantic replacement trainer, not to DX2's shipped HPAC
architecture. Thus **49/49 inventory rows are not direct knobs on the exact incumbent**. Four
replacement-trainer mechanisms nevertheless had real retained measurements; that does not falsify
the prediction because their verdict surface is explicitly a replacement vehicle. The campaign
parked mostly for vehicle-relevance reasons, not because it had measured a large live-vehicle
failure population.

The sub-0.12 consequence is severe. RB1 measures **0 B** admissible supply in every incumbent DX2
stream, a fixed-distortion archive cap of **137,986 B**, and a **42,382 B** missing new-
representation residual. Even impossible zero distortion leaves a 150 B representation cut. No
legacy training flag below receives a byte credit merely for moving a proxy or shrinking an
unclosed field.

## Denominators and ledger receipt

| surface | before NL1 | after NL1 | interpretation |
|---|---:|---:|---|
| ledger rows | 287 | 340 | 53 append-only custody rows added; no rewrite |
| event rows | 279 fired / 7 measured / 1 built | 283 fired / 17 measured / 35 retired / 4 queued / 1 built | four conditionally active fields were missing real `fired` rows |
| distinct fired names | 41 | 45 | `FRD077` plus the three behaviorally consumed F1 fields repaired |
| distinct measured names | 6 | 16 | retained receipts harvested, not re-run |
| fired names without any measurement | 37 | 31 | all remaining 31 are retired at vehicle scope |
| fired names without measurement and not retired | 37 | **0** | fired-unmeasured orphan queue drained |
| source-derived semantic replacement flags | 18 | 18 | AST-derived denominator unchanged |
| live-registry duty queue | 12 never fired | **4 never fired, all queued** | two coupled treatments, no unknown row |

Post-drain ledger SHA-256: `e36fe4c070e1dfd886c58e02ee377d413648db4613a4364778507cc90547fb91`.

## Population A — 12 source-derived never-fired flags

“Current relevance” distinguishes the source module's historical word *live* from exact archive
custody. These flags train a 66,339-parameter FiLM semantic renderer. DX2's shipped HPAC has 39,375
parameters and no FiLM tensors; the semantic trainer is a named **replacement-vehicle candidate**,
not a modifier of the incumbent architecture.

| flag / ledger name | current relevance | cost and what a fire measures | exit |
|---|---|---|---|
| `--bits` / `bits` | replacement trainer only; a global lossy representation choice | A real test needs training, full Seg/Pose closure, and complete archive bytes. Same-object re-coding cannot supply RB1's 42,382 B residual. | **RETIRED, current-representation scope.** A born new representation chooses its own quantization; this legacy flag is not an independent arm. |
| `--carrier-rank-penalty` / `carrier_rank_penalty` | replacement trainer only; from-scratch training is outside RB1's post-hoc rank/refit closure | Prior estimate about 25 Metal minutes for one A/B, then actual coder plus scorer closure. It would measure whether spectral concentration reduces *coded* carrier bytes without losing joint score. | **QUEUED-WITH-A-FIRE-ORDER**, treatment 2 below. |
| `--carrier-tensors` / `carrier_tensors` | selection half of the same rank treatment | No independent fire. It must measure which tensors lower real coded bytes; stable rank is forbidden as a proxy because EF3000 needs rank 72–74 at 99% energy, costing 150–154% of dense. | **QUEUED-WITH-A-FIRE-ORDER**, treatment 2 below. |
| `--distill-max-seg` / `distill_max_seg` | replacement trainer only; control field for a new student objective | No independent fire. A valid row derives the switch from a frozen endpoint and measures full-population joint distortion plus complete student bytes. | **QUEUED-WITH-A-FIRE-ORDER**, treatment 1 below. |
| `--distill-weight` / `distill_weight` | replacement trainer only; potentially reusable only inside NR1's born-small student | Future training/scorer cost is unmeasured. It would measure endpoint/C1 teacher supervision from birth, not repeat WD3 fresh-init or finishing-only KD. | **QUEUED-WITH-A-FIRE-ORDER**, treatment 1 below. |
| `--ema-target-seed-fraction` / `ema_target_seed_fraction` | replacement trainer only; current default is structurally decorative | **$0 source/LawRef derivation.** At `f=0.01`, crossover is about `1.954*N`, so target decay never governs within any N-step run; the warmup ramp is the policy. | **RETIRED, formulation scope.** Reactivate only below `exp(-9)` or after a warmup-policy change. |
| `--film-critical-multiplier` / `film_critical_multiplier` | replacement trainer only; behaviorally active when F1 sigma is nonzero | **$0 retained receipt.** `F1SIG1` consumed the default `9.679876...` multiplier. It measures the quantization-shaped, Film-amplified sigma-1 treatment, not an independent multiplier curve. | **MEASURED, instance scope.** The coupled treatment blocked descent completely. |
| `--film-row-dropout` / `film_row_dropout` | replacement trainer only; FiLM is structurally absent from incumbent HPAC | **$0 retained receipt.** `FRD077` measures `p=0.077` against EF3000 at n600 and equal packed bytes. | **MEASURED, formulation scope.** +158 flips (0.18 sigma worse), 40,252 B both; receiver-closed family price is +0.062227 S. |
| `--film-row-dropout-protect-top` / `film_row_dropout_protect_top` | inert companion unless dropout fires; replacement trainer only | A separate count sweep would measure whether protecting sensitivity-ranked rows changes droppability. It cannot repair the already measured pose-negative family without a new pose-safe representation. | **RETIRED with the Film-row family.** Reactivate only on a new pose-safe representation. |
| `--fixed-zero-mask` / `fixed_zero_mask` | replacement trainer only; its sole consumer pins exact-zero initialization weights | **MEASURED scorer-free on retained init, $0 CPU.** It measures eligible mask support, not ablation. Checkpoint is 282,352 B, SHA-256 `3948ccfcd44778dc42affee18a10c3f3baa434d1a2eb2345a013146c1dbfb647`. | **RETIRED, instance scope.** 0/66,339 exact-zero parameters overall and **0/63,936** across all 16 matrix tensors; the mask is empty. |
| `--weight-perturb-robustness` / `weight_perturb_robustness` | replacement trainer only | **$0 retained receipt.** `F1SIG1` measures sigma 1.0 at n600 and 40,252 packed bytes. | **MEASURED, instance scope.** Best step 0; selected endpoint equals init to 17 digits; descent blocked. |
| `--weight-perturb-shape` / `weight_perturb_shape` | conditionally active F1 companion; replacement trainer only | **$0 retained receipt.** The actual treatment used `quantization`; no Gaussian comparison exists. | **MEASURED for quantization shape only.** It shares the F1 block; Gaussian remains outside the finding and has no independent entitlement. |

The scorer-free support check is the only new computation in NL1. It loaded the retained init on
CPU, counted exact equality to zero, and materialized no payload. All trained checkpoints and packed
payloads were already retained under `/Volumes/APDataStore/pact/ddm_ce1/`; NL1 discarded nothing.

## Population B — 37 fired names without ledger measurements

### Six replacement-trainer names with retained measurements

| lever | firings / unique runs | current relevance | retained measurement and exit |
|---|---:|---|---|
| `band_objective_weight` | 1 / 1 | semantic replacement only | `[macOS-MPS training-signal]`, n600: 600 activations rotated the step by 66.3 degrees, but best exact Seg stayed at init; residual judge underpowered; 40,252 B. **MEASURED, formulation scope.** |
| `ce_fraction` | 4 / 4 | semantic replacement only | `[macOS-MPS training-signal]`, n600 matched allocation ladder. It helps separate the CE allocation effect but makes no exact-score claim. **MEASURED.** |
| `float_warmup_steps` | 1 / 1 | semantic replacement only | `[macOS-MPS training-signal]`, n600: 100-step float warmup broke the QAT-from-step-0 displacement law by about 3x and ended worse; 40,252 B. **MEASURED, formulation scope.** |
| `lr` | 3 / 3 | semantic replacement only | `[macOS-MPS training-signal]`, n600 family: UP was weakened-directional worse, DOWN null, `2e-5` the observed plateau for this window. **MEASURED, formulation scope.** |
| `softplus_fraction` | 3 / 3 | semantic replacement only | `[macOS-MPS training-signal]`, n600: control-to-EF0 removed 8,018 endpoint flips, 92.651% of the control excess; single-seed signal, no score. **MEASURED.** |
| `weight_qat_q3q4` | 24 / 12 | semantic replacement only | `[macOS-MPS training-signal]`, n600: ON contributed -563 flips, 0.66 sigma and within noise; ON/OFF both 40,252 B. **MEASURED, instance scope.** |

Retained result receipts used directly: EF3000 31,658 B / SHA `919e4c66...b49cbf0`;
FRD077 31,638 B / `ce463c83...57024e8`; Q3Q4OFF 31,657 B /
`d82a0072...cb29c`; F1SIG1 31,622 B / `57103043...ed0f4a`; A1/A3/C0/W1 and
band-a1 remain under their original APDataStore directories. These are training-signal and packed-
parameter receipts, not contest authority.

### Thirty-one wrong-vehicle names

For every row below the allowed NL1 cost was **$0 to classify and retire**. A legitimate re-fire
would first require a port into a current receiver/trainer, then retained training, actual coding,
and n600 scorer closure; that cost is unpriced and the port would define a new lever rather than
measure the recorded one. The “what it measures” column states the old surface, so the retirement
does not masquerade as a measured failure.

| lever | firings / runs | what the recorded fire would measure | current relevance and exit |
|---|---:|---|---|
| `DsegAwareTaper` | 1 / 1 | a rate-instrument memo declaration, not a receiver-closed run | Absent from DX2 and the semantic replacement trainer. **RETIRED, vehicle scope.** |
| `FEED_08a_length_sigma` | 11 / 7 | old levelset length-scale schedule | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `R7_beta2_window_rewarmup` | 11 / 7 | old optimizer beta2 rewarmup | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `R7_polyak_finisher` | 13 / 9 | old Polyak finishing schedule | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `adaptive_grad_clip_autoclip` | 3 / 1 | n24 gradient clipping behavior | Retired levelset arm only. **RETIRED, vehicle scope.** |
| `c2_component_wallclock_telemetry` | 1 / 1 | telemetry completeness, not score | Apparatus row for an old n600 run. **RETIRED, vehicle scope.** |
| `c2_speed_stack` | 1 / 1 | old trainer throughput stack | No current representation effect. **RETIRED, vehicle scope.** |
| `c2_warm_start_weights_only` | 1 / 1 | old weights-only resume behavior | No current receiver binding. **RETIRED, vehicle scope.** |
| `compute_dtype_bf16_qc_gate` | 1 / 1 | n24 bf16 quality-control gate | Wrong substrate and retired trainer. **RETIRED, vehicle scope.** |
| `dseg_aware_taper` | 10 / 6 | old levelset d_seg taper | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `g111_physical_batch16_target_custody` | 1 / 1 | v9 batch-16 custody condition | Custody marker, not a DX2 lever. **RETIRED, vehicle scope.** |
| `grad_normalize_none` | 5 / 3 | old n24 gradient normalization ablation | Retired levelset arms only. **RETIRED, vehicle scope.** |
| `head_offset_solver` | 1 / 1 | old levelset head-offset solve | Structurally absent from current representations. **RETIRED, vehicle scope.** |
| `margin_band_satisficing` | 12 / 8 | old margin-band objective | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `margin_saliency` | 11 / 7 | old margin-saliency weighting | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `n287_dash_comb` | 13 / 9 | dash-comb regularizer on old geometry | Structurally absent from current representations. **RETIRED, vehicle scope.** |
| `n292_closed_loop_eikonal_control` | 11 / 7 | old eikonal control loop | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `n323_ladder_island_homotopy` | 13 / 9 | old island-homotopy schedule | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `perclass_sensitivity_bitalloc_336_sparc` | 1 / 1 | n48 per-class bit-allocation proxy | Bounded old-vehicle receipt, no current receiver binding. **RETIRED, vehicle scope.** |
| `phase_advection_consistency` | 13 / 9 | old phase-advection regularizer | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `pose_blind_compute_gate` | 1 / 1 | old compute gating around a pose-blind phase | Structurally absent from current trainers. **RETIRED, vehicle scope.** |
| `pose_finish_conditioning_gate` | 14 / 10 | old terminal pose-conditioning schedule | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `seg_form_unify_tau` | 13 / 9 | old tau unification schedule | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `tail_k_warm_restart` | 13 / 9 | old tail-k restart | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `temporal_screw_consistency` | 13 / 9 | old temporal screw regularizer | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `tie_locus_displacement` | 12 / 8 | old tie-locus displacement term | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `unified_tau_eikonal_hold` | 11 / 7 | old tau/eikonal hold | Retired levelset/v9 lineage only. **RETIRED, vehicle scope.** |
| `v75_area_constraint_birth` | 13 / 9 | v7.5 area-constrained birth | Retired v7.5/v9 lineage only. **RETIRED, vehicle scope.** |
| `v75_birth_completion_event` | 13 / 9 | v7.5 lifecycle event | Telemetry/lifecycle row, not a DX2 lever. **RETIRED, vehicle scope.** |
| `v9_flag_custody_rollup` | 1 / 1 | v9 flag-custody completeness | Custody marker for a retired vehicle. **RETIRED, vehicle scope.** |
| `verdict_live_gap` | 1 / 1 | old apparatus gap field | Telemetry, not an actuator. **RETIRED, vehicle scope.** |

The 31 rows remain historically “fired without a measurement”; retirement does not invent missing
numbers. Their current duty is terminal because they cannot measure the exact object without first
becoming different levers on a different receiver.

## Ranked fire-order queue

The four surviving flags are two treatments, not four independent votes. No NL1 scorer, Metal job,
Modal call, new candidate, or coder mutation was launched. Current JO/r9 custody and every path under
`experiments/.scratch/ddm_jo2_joint_objective_solve/**` were untouched.

1. **Distillation treatment — `distill_weight` + `distill_max_seg`.** Owner: MAIN/NR1. Consumer:
   `/Volumes/VertigoDataTier/pact/ddm_nr1_taskcell_body_rebase/retained/teacher_ablations/`. Fire only
   after the JO endpoint is frozen and hashed, a deterministic born-small receiver with actual
   complete-container coder exists, the scorer lane is free, and TL1 endpoint/C1 supervision is an
   identical-seed from-birth A/B. `distill_max_seg` is derived from that endpoint and its exact byte
   bar. An n32 gate may stop engineering; only receiver-closed n600 is evidence.
2. **Rank-coded-size treatment — `carrier_rank_penalty` + `carrier_tensors`.** Owner: MAIN/NR1.
   Consumer: `/Volumes/VertigoDataTier/pact/ddm_nr1_taskcell_body_rebase/retained/rank_ab/`. Fire only
   if NR1 first supplies at least **20,372 B** outside the physical 22,010 B carrier, so even a perfect
   carrier deletion could compose to the 42,382 B bar; the Metal/scorer lane is free; and tensor
   selection comes from measured coded-byte sensitivity. The result must be a retained control and
   penalty checkpoint, real coded payloads, receiver closure, and joint score. Stable rank or gross
   factorization bytes cannot pass the gate.

## Measurement boundary

**Measured here:** the pre/post ledger denominators; the exact-zero support of the retained semantic
initialization (`0/66,339`, matrix support `0/63,936`); file bytes and hashes; source-level structural
vehicle mismatch; and the append-only terminal state of every inventory row.

**Read as retained measurements:** the CE/softplus/lr/window, band-objective, q3/q4, FRD077, and
F1SIG1 results at their explicitly advisory/training-signal axes. NL1 did not rerun them and does not
promote them to contest authority.

**Not measured here:** a new rank-penalized checkpoint, a distilled NR1 student, Gaussian F1 shape,
an independent Film multiplier curve, a complete new representation, any new Seg/Pose component,
any exact score, or any movement of the 42,382 B residual.

## RECALL EVIDENCE

### Sources and queries

- Governing surfaces: `PROGRAM.md`, identical `CLAUDE.md`/`AGENTS.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, this charter, and
  `_common_contract.md`.
- Ledger joins: event/type counts; distinct fired minus distinct measured; same-`run_ref` joins;
  `live_levers()`/`live_never_fired()`; per-name firing/run denominators.
- Full-corpus queries included `carrier_rank_penalty|carrier_tensors|distill_weight|distill_max_seg`;
  `film_row_dropout|film_critical_multiplier|weight_perturb`; `fixed_zero_mask|already-zero`;
  `ema_target_seed_fraction|target.*bind`; and all 37 fired-without-measurement names.
- Primary beyond-seed memos: `ddm_drain_vehicle_split_and_lever2_payoff_20260817.md`,
  `ddm_q3q4_owed_control_verdict_20260817.md`,
  `ddm_frd077_lever_verdict_and_zero_row_nan_20260817.md`,
  `ddm_na9_gestalt_negative_audit_20260818.md`,
  `ddm_ce1_allocation_ladder_verdict_20260817.md`,
  `ddm_cw1_corrected_window_20260817.md`,
  `ddm_jr1_band_objective_judge_repair_20260817.md`, RB1, and TL1.
- Canonical equation registry: `.venv/bin/python tools/list_canonical_equations.py --json`, with
  rate, representation, rank, distillation, quantization, mask, perturbation, carrier, and EMA
  terms. `ema_decay_substrate_stage_aware_v1` was relevant; no equation licensed a byte credit for
  any queued treatment.
- Graph/queue surfaces: `CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED blocks,
  `canonical_task_status.jsonl`, the active lane ledger, and the bounded harness-bridge surfaces.
  The exact flag-name search found no newer canonical-index row beyond the DAG/memos above.
- Evaluator source was read only. Its authority formula is
  `100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/uncompressed_bytes`; no printed two-decimal score
  was used.

Recall-neighborhood adjudication, including every top-12 automated hit:

- `.omx/research/ddm_rc2_regime_charter_and_lr_probe_20260816.md` supplied the original F1–F4
  treatment coupling, the derived Film multiplier, and the rule that rank remains off until a rank
  edit enters the deployment distribution. Later FRD/F1/RB1 receipts supersede its “never fired”
  state, not its mechanism definitions.
- `.omx/research/ddm_todo_p0_live_lever_queue_20260817.md` is the 18/6/12 source denominator and
  ingester contract. NL1 updates its snapshot with later receipts; it does not replace the AST
  denominator.
- `.omx/research/ddm_deferral_queue_ledger_20260729.md` is a pre-DX2 standing queue on the t3/v7.5
  lineage. Its similarly named levers are wrong-vehicle rows here; none supplies a DX2 or PR130
  semantic replacement measurement.
- `.omx/research/t5_crucible/CONTEXT_COMPENDIUM_20260707.md` is likewise the v7.5 operating context.
  It supports the tracked-off discipline but has no current archive or retained treatment for the
  12 semantic flags.
- `.omx/research/ddm_b2e_landing_and_charter_repin_20260816.md` proves F1–F4 are real behavior, not
  argparse stubs, and exposes conditionally active companion fields. That changed the ledger repair
  for F1 shape/multiplier; it contains no later A/B verdict.
- `.omx/research/ddm_na9_gestalt_negative_audit_20260818.md` supplies the retained F1SIG1 drain
  verdict and the required instance scope.
- `.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md` is a May design/prediction
  document for Pact-NeRV/scorer distillation. It predates the current representation and carries no
  retained DX2/semantic-treatment measurement, so it does not alter the NR1-only distillation gate.
- `.omx/research/nerv_fleet_reactivation_and_arch_selection_20260610T192434Z.md` is a June
  reconstruction-NeRV fleet audit. Its shared-loop cure and smaller-basis predictions are not the
  PR130 semantic trainer flags and provide no current coded payload; ruled out at vehicle scope.
- `.omx/research/default_off_comprehensive_sweep_20260710.md` inventories v7.5 defaults and is the
  clearest historical instance of the vehicle-stale queue. It supports retiring the 31 old names;
  its unmeasured v7.5 rows are not transferred into the 18-flag denominator.
- `.omx/research/ddm_pr130_reproduce_20260809/RR2_SEMANTIC_LEG_AUDIT.md` confirms the exact retained
  checkpoint SHA, 66,339-parameter architecture, pack/receiver coherence, 0/18,000 historical
  fixed-mask steps, and zero historical distillation steps. It strengthened the fixed-mask custody
  check but does not itself count exact-zero support; NL1 measured that support directly.
- `.omx/research/ddm_rb1_rate_bound_decomposition_20260822.md` is the pinned 42,382 B/zero-headroom
  authority and supplies the NR1 and carrier/distillation disposition boundaries.
- `.omx/research/ddm_r012_rate_representation_20260821.md` is the prior rc2-line rate theorem. DX2/RB1
  supersede its 238 B zero-distortion deficit with 150 B; its new-representation conclusion survives.

### Beyond-seed findings that changed the plan

1. The 275-firing seed was stale; the actual pre-drain event denominator was 279. The distinct 37
   remained correct, so the row inventory did not change.
2. `FRD077`, `Q3Q4OFF`, and `F1SIG1` were complete retained runs but absent from the activation
   ledger. This converted four of the 12 rows into measurements and supplied six missing verdicts
   for already-fired fields without consuming the scorer.
3. The source-derived “live trainer” is a semantic replacement candidate with a different
   architecture from DX2 HPAC. This changed the main classification from “incumbent levers” to
   “replacement-vehicle levers.”
4. The F1 treatment behaviorally consumed default-valued `weight_perturb_shape` and
   `film_critical_multiplier`; manifest nondefault comparison had missed both. They were repaired
   as real fired/measured companion fields, not counted as independent experiments.
5. The EMA target is structurally inert at `f=0.01` for every run length, stronger than a short-run
   negative. That retired the row without another training run.
6. The retained initialization has no exact zeros. That made `fixed_zero_mask` an empty operator on
   the actual object and closed the only new scorer-free mechanism probe.
7. RB1 and TL1 already assign the new-representation and teacher/student work to NR1. Distillation
   and rank were therefore queued as conditional NR1 treatments, not spawned as parallel legacy
   vehicles.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — action: run the from-birth endpoint/C1 distillation A/B for `distill_weight` plus derived `distill_max_seg`; owner: MAIN/NR1; consumer store: `/Volumes/VertigoDataTier/pact/ddm_nr1_taskcell_body_rebase/retained/teacher_ablations/`; fire trigger: frozen hashed JO endpoint, deterministic born-small receiver and actual coder, complete byte bar, retained stage checkpoints, and free scorer lane.
- **QUEUED-WITH-A-FIRE-ORDER** — action: run the actual-coded rank treatment for `carrier_rank_penalty` plus sensitivity-selected `carrier_tensors`; owner: MAIN/NR1; consumer store: `/Volumes/VertigoDataTier/pact/ddm_nr1_taskcell_body_rebase/retained/rank_ab/`; fire trigger: another NR1 section has already supplied at least 20,372 measured composable bytes, the Metal/scorer lane is free, and matched retained control/penalty checkpoints can be receiver-closed.

## LIVE-HYPOTHESES

- A born-small, current-endpoint distillation treatment can preserve scorer-visible behavior at far fewer complete bytes because TL1's useful teacher object is logits/margins/cells, not the 113,777 B exact token stream. It is plausible only inside NR1's new representation, not as a legacy semantic-trainer tweak.
- Training-time spectral concentration may reduce real coded bytes even though post-hoc factorization loses, because entropy cost depends on the quantized symbol distribution rather than the number of singular vectors. It is plausible only after another supplier makes the carrier's 22,010 B physical ceiling composable with the 42,382 B demand.
- A Gaussian F1 perturbation could differ from the blocked quantization-shaped sigma-1 instance. It remains scientifically untested, but has no fire entitlement until a new representation exposes a concrete perturbation payoff and complete-byte bar.

## DEAD-ENDS

- Re-firing any of the 31 levelset/v7.5/v9 names is closed at vehicle scope: none is a knob on DX2 or the semantic replacement trainer, and several are telemetry rather than actuators.
- `fixed_zero_mask` is closed on the retained semantic initialization: its support is exactly empty, 0/63,936 matrix parameters.
- EMA target tuning at the canonical `f=0.01` is closed under the current warmup law: the target never governs inside the run.
- Film row dropout/protection is closed on the measured family scope: FRD077 is seg/byte neutral and the receiver-closed payoff is pose-negative by +0.062227 S.
- Quantization-shaped F1 at sigma 1.0 with the measured Film multiplier is closed at instance scope: it blocks descent and supplies zero bytes.
- Q3/Q4 training is not a hidden byte win on this replacement vehicle: its isolated effect is within noise and byte-neutral at 40,252 B.
- Post-hoc carrier rank/refit and EF3000 factorization are closed on their measured scopes: the former misses break-even by at least 35.5x, while the latter costs 150–154% of dense at 99% energy. Only a new from-scratch coded-size mechanism remains outside those closures.
- Another legacy width/fresh-init or finishing-only distillation run is closed by WD3/WD4/DW1. The queued hypothesis is a born NR1 quotient trained on the frozen current endpoint, not the same defect under a new width.

**Own-vehicle frontier:** **S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**, DX2 archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; NL1 did not move it.
