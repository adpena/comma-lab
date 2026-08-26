# ddm_hv2 — harvest-consumption sweep: 82 clean finishes routed to owned exits

Date: 2026-08-26  
Actor: `ddm_hv2`  
Authority: source/receipt and queue-state audit; no scorer, archive mutation, Modal, Metal, or evaluator run

## Outcome

The live keeper truth was **82 clean finishes awaiting harvest**, not the `79` in the stale hot-state sentence. All 82 keeper receipts had one indexed retained final; all 82 retained files matched their indexed SHA-256. The extraction found **79 `NEXT_IF_RESUMED` blocks, 82 `LIVE-HYPOTHESES` blocks, and 82 `DEAD-ENDS` blocks**.

Every source row now has exactly one disposition in the machine ledger:

| Disposition | Rows | Meaning |
|---|---:|---|
| `ALREADY-CONSUMED` | 50 | A later artifact, commit, dispatch, or exact row already consumed the exit. |
| `STALE-SUPERSEDED` | 25 | A later object/family/packet receipt makes the historical fire order invalid. |
| `QUEUED-with-order` | 2 | The action remains valid but its named gate is not met. |
| `FIRE-NOW` | 5 | The source trigger is met and the action has no later consumer. |
| **Total** | **82** | Complete clean-finish denominator. |

Machine-readable ledger: `.omx/research/ddm_hv2_harvest_consumption_ledger_20260826.jsonl`  
Ledger rows: **82 / 82**, unique `(source_path, source_sha256)`: **82 / 82**  
Ledger SHA-256 before serializer landing: `a941c5b5fb7d427a5de3d2db8bb80db6cbc6031382d160fcbf12dcc8ab059589`

The two final-message files absent from the index are bounded duplicate captures, not additional finishes: the `20260822T220000Z` AE1 and XS1 files are byte-identical to their indexed captures. Therefore they add **0** to the true-set denominator.

## True-set construction

The cohort is the intersection of:

1. the latest keeper row was `live` but no process existed;
2. a clean `.done` receipt existed (`rc=0`);
3. the retained final-message index named a file and SHA;
4. the retained file existed and matched that SHA.

Measured source denominators before routing:

| Surface | Count |
|---|---:|
| Latest tracked arm names | 476 |
| Processless latest-`live` rows | 85 |
| Clean `.done` finishes | 82 |
| Processless rows with no `.done` receipt | 3 (`ddm_d3b`, `ddm_pc2`, `ddm_hv2`) |
| Retained final-message files | 429 |
| Indexed final-message rows | 427 |
| Indexed clean-finish files verified | 82 / 82 |

The three receiptless rows are resumable deaths, not finished messages, and are outside this harvest denominator. A `.done` file alone was not treated as consumption. Consumption required a later commit/artifact/measurement/dispatch reference or a newly owned route in the ledger.

## Exhaustive routed table

The ledger preserves every source filename, source SHA, `.done` receipt, extraction line span and block hash, final disposition, owner, consumer store, fire trigger, and the retained LIVE/DEAD block hashes. The full source routing is summarized here without dropping any row:

| Disposition | Source rows |
|---|---|
| `FIRE-NOW` | `ddm_jf1_joint_field_model_refit`; `ddm_sw1_portable_paths_secrets_scrub`; `ddm_sy2_composition_synergy_deep_pass`; `ddm_tb2_token_bit_attribution`; `ddm_wj1_cost_error_position_join` |
| `QUEUED-with-order` | `ddm_cc2_catalog_consolidation`; `ddm_cm1_coder_matched_surrogate` |
| `ALREADY-CONSUMED` | `ddm_ad2_addressing_cost_decomposition`; `ddm_ap1_residue_purchase_scorer`; `ddm_bl1_per_position_bit_allocation`; `ddm_cb2_class_balanced_dictionary`; `ddm_d3_alphabet_merge`; `ddm_db1_decode_boundary_families`; `ddm_dc1_decode_time_compute`; `ddm_dc1s_sparse_grid_sweep`; `ddm_dj1_dual_lineage_carrier`; `ddm_dx2_cabac_receiver_fold`; `ddm_gs3_unbridled_gestalt`; `ddm_gt2`; `ddm_gv1_gestalt_validation`; `ddm_hg1_heterogeneous_analytic_generator_gate`; `ddm_ht1_hard_tail_student`; `ddm_ig1_implicit_carriage_gestalt`; `ddm_jo1_joint_objective_design`; `ddm_jo1u2_materializer_cure`; `ddm_jo1u_payload_unblock`; `ddm_jo2_solve_reseal`; `ddm_jo3_entrypoint_and_final_reseal`; `ddm_jo4_certified_retention_reseal`; `ddm_jo5_determinism_cure_reseal`; `ddm_jx1_joint_exchange_envelope`; `ddm_lq1_lane_quotient_representability`; `ddm_ni1_nr1_k32_receiver_distortion`; `ddm_nr1_taskcell_quotient_prebuild`; `ddm_nt1_naive_toy_generic_audit`; `ddm_ny1_live_lineage_toy_and_reactivation_audit`; `ddm_oe1_online_escape_member`; `ddm_os1_orphan_signal_reconciliation`; `ddm_r012_rate_representation`; `ddm_rb1_rate_bound_decomposition`; `ddm_rc1_rate_crush`; `ddm_rc2_composed_clean_decode_and_seal`; `ddm_ri1_rc1_full_rgb_receiver`; `ddm_rj2_joint_renderer_object_change`; `ddm_rp2_reaper_shim_cure`; `ddm_rv16_round3_finding_wave`; `ddm_rvf1`; `ddm_s1_trained_renderer_diagonal`; `ddm_s1a_stage_a_adapter`; `ddm_s1e_off_floor_adjudicator`; `ddm_tl1_teacher_ledger`; `ddm_to2_token_ordering_race`; `ddm_wc2_jo1_wallclock`; `ddm_wh1_wrong_half_decomposition`; `ddm_ws0_worldsheet_grammar_price`; `ddm_ws1_optimal_worldsheet_grammar`; `ddm_xt1_exact_solve_teacher_student` |
| `STALE-SUPERSEDED` | `ddm_ae1_anti_predicted_excess`; `ddm_ar1b_archive_residue_purchase`; `ddm_cx3_context_axis_ceiling`; `ddm_d3a_analytic_lane_carrier`; `ddm_ec2_collateral_suppressed_conditioner`; `ddm_ef1_token_entropy_floor`; `ddm_es1_end_state_characterization`; `ddm_et1_edge_topology_container_gate`; `ddm_hr3_residual_implicit_carrier`; `ddm_jo6_receiver_container_compat`; `ddm_ld1_lane_lossy_drop_exchange`; `ddm_lx2_lane_bit_budget_exchange`; `ddm_mf1_manufactured_seg_repair`; `ddm_mp3_hpac_member_prune`; `ddm_ms9_dx2_seg_manufactured_fraction`; `ddm_mst1_manufactured_stage_split`; `ddm_na12_post_sy2_negative_regrade`; `ddm_nl1_never_fired_levers`; `ddm_pq10_codex_packet_review_round`; `ddm_pq9_pr_final_polish`; `ddm_rj1_renderer_joint_move`; `ddm_rx3_receiver_precompensation`; `ddm_vf1_evaluator_visible_floor`; `ddm_wd4_warm_lineage_width`; `ddm_xs1_cross_section_conditioning` |

## FIRE-NOW head — do not auto-fire score work

| Order | Source | Owner | Consumer store | Fire trigger now satisfied | Action |
|---:|---|---|---|---|---|
| 1 | SW1 | credential owner/operator | `.omx/state/operator_p0_ledger.jsonl` | `immediately` | Rotate or revoke the historical GCP/JWT credentials and record revocation evidence without recording secret values. This is an operator action. |
| 2 | JF1 | MAIN/JF1 byte custodian | `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/rows/` | all seven `epoch_0060.pt` checkpoints exist; all seven launch PIDs are terminal and status receipts say `ok` | Physically encode and receiver-close the seven terminal model+stream rows and write `BYTE_DIAGONAL.json`. No scorer is authorized by this harvest. |
| 3 | SY2 | MAIN/JF1 harvester | `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/` | the seven JF1 epoch-60 fits are terminal | Compare terminal absolute model+stream bytes with the pre-registered 127,292 B threshold and record the harvest. |
| 4 | WJ1 | JF1 byte/model successor | `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/wj1_target_consumer/` | WJ1 is `COMPLETE`, the position-list SHA is pinned, and terminal JF1 controls now exist | Run the retained field-coarsen plus real-model-refit rung against the terminal null-refit control. No score claim without receiver closure and scorer authority. |
| 5 | TB2 | MAIN-designated CB2 task-weighted K2048 successor | `/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/reactivated_task_weighted_refit/` | TB2 says `CB2_ALLOCATION_HANDOFF_FIRED` and its verification/hash trigger is `MET` | Consume the cost/task/context coordinates as fitting weights; stop scorer-free unless the complete receiver archive is at most 137,986 B. |

This is **5**, not the charter prediction of at least 10 source-verified unconsumed rows with met conditions. The prediction is falsified on the bounded 82-row cohort. It is not an apparatus-only residue: JF1, SY2, WJ1, and TB2 are model/token/rate rows, while SW1 is P0 security custody.

## Valid queued exits

## ITEM 1 — JF1 terminal byte harvest

- Disposition: `FIRE-NOW`.
- Owner: `MAIN/JF1 byte custodian`.
- Consumer store: `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/rows/`.
- Fire trigger: seven terminal epoch-60 checkpoints and terminal `ok` launcher receipts, already observed.
- Completion: exact physical model+stream rows, receiver closure, `BYTE_DIAGONAL.json`, retained hashes; no scorer unless separately authorized.

## ITEM 2 — SW1 historical credential revocation

- Disposition: `FIRE-NOW / OPERATOR`.
- Owner: `credential owner/operator`.
- Consumer store: `.omx/state/operator_p0_ledger.jsonl`.
- Fire trigger: `immediately`.
- Completion: rotate/revoke and record non-secret revocation evidence.

## ITEM 3 — SY2 terminal JF1 threshold harvest

- Disposition: `FIRE-NOW`.
- Owner: `MAIN/JF1 harvester`.
- Consumer store: `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/`.
- Fire trigger: all seven epoch-60 fits terminal, already observed.
- Completion: recorded terminal-byte comparison against 127,292 B.

## ITEM 4 — WJ1 target-consumer refit

- Disposition: `FIRE-NOW`.
- Owner: `JF1 byte/model successor`.
- Consumer store: `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/wj1_target_consumer/`.
- Fire trigger: WJ1 `COMPLETE` plus pinned position-list SHA plus terminal JF1 controls, already observed.
- Completion: retained coarsened-field model refit and exact byte comparison.

## ITEM 5 — TB2 task-weighted K2048 handoff

- Disposition: `FIRE-NOW`.
- Owner: `MAIN-designated CB2 task-weighted K2048 successor`.
- Consumer store: `/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/reactivated_task_weighted_refit/`.
- Fire trigger: TB2 verification and all handoff hashes match; source trigger is `MET`.
- Completion: complete receiver count first; scorer only at or below 137,986 B with a separately granted lane.

## ITEM 6 — CC2 catalog consolidation adjudication

- Disposition: `QUEUED-with-order`.
- Owner: `MAIN plus operator`.
- Consumer store: `docs/meta_bug_class_catalog.md` plus `src/tac/preflight.py`.
- Fire trigger: operator approval of the named replacement/consolidation set.
- Completion: approved consolidation landing with protection retained.

## ITEM 7 — CM1 coder-matched objective validation

- Disposition: `QUEUED-with-order`.
- Owner: `MAIN`.
- Consumer store: `.omx/state/canonical_task_status.jsonl::ddm_no1_row1_three_term_objective`.
- Fire trigger: either a restartable F26/HPAC exact-increment cache validates Pearson and Spearman at least 0.9 on stratified-random `n>=32`, or Metal plus the outstanding wd3 seed control becomes available.
- Completion: a validated trainable objective input, not a proxy-based score claim.

## Negative and hypothesis retention

The 82 `DEAD-ENDS` blocks were not copied into a second prose registry. Their source line spans and SHA-256 hashes are registered in each ledger row under `dead_end_registry.status=REGISTERED_IN_HV2_LEDGER`. The 82 `LIVE-HYPOTHESES` blocks are preserved the same way. The source final and owning memo remain immutable evidence. This prevents both retry loss and duplicate, decontextualized negative claims.

## GESTALT-DELTA

The queue looked like a large reservoir of missed score work because the hot headline counted clean finishes, not unconsumed exits. Joining retained finals to later object-level receipts changes that picture: **75 / 82 rows are already consumed or invalidated**, and only **7 / 82** retain an owned future action. The five met actions form two coupled clusters, not five independent frontier bets: one JF1 byte-harvest cluster (JF1 + SY2 + WJ1), one TB2 rate-refit, and one operator security cure. The remaining two are gated governance/objective work. The highest-value correction is therefore consumption bookkeeping plus one bounded terminal harvest, not respawning the historical 82-arm wave.

The source crosswalk also prevents a specific false rate signal: AD2's 17,957 B QPAIR win belongs to the now-authority-dead NI1/NR1 K32 object, while the current GB1 object has no such measured win. It is retained as evidence, not promoted as current-body headroom.

## RECALL EVIDENCE

Recall was not seed-only. It covered:

- governing law: `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, `docs/vehicle_operating_system.md`, the charter, common contract, `main_hot_state.md`, and the canonical frontier pointer;
- queue/custody truth: `tools/codex_arm_queue.py`, 476 latest queue names, `.done` receipts, 429 retained final files, 427 indexed final rows, the np1 extraction implementation at commit `499ffd68a1`, prior HV2R 97-file harvest, and OQ1's older drain;
- all 82 source-verified retained finals, including every retained reasoning block;
- consumption evidence: canonical task status, current hot state, later exact/receiver receipts, `ddm_s1e_stage_a_off_floor_verdict_20260825.md`, `ddm_no1_new_object_derivation_20260826.md`, D3/D3A/D3B lineage, and current Git custody;
- research graph: `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`, design/SPEC/task-ledger searches, and `tools/list_canonical_equations.py --json` queries for harvest, consumption, token/rate, quotient, HPAC, and follow-on terms.

The beyond-seed facts that changed routing were: NI1/NR1 K32 later received an authority row and is dead at any archive; S1 Stage A later closed its family after both OFF seeds; JO1 never produced the promised joint row and NO1 superseded its RC2-bound formulation; all seven JF1 epoch-60 checkpoints now exist even though no terminal byte harvest was written; and the hot `79` figure was stale against the keeper's measured 82.

## Verification boundary

- Measured here: source counts, receipt joins, file/index SHA equality, extraction counts, later-consumer references, JF1 terminal checkpoint/PID/status facts, and ledger validation.
- Not measured here: any candidate distortion, scorer output, new archive size, exact contest score, or frontier delta.
- No score job was fired. No retained payload was deleted or moved. No protected file or `upstream/` file was edited.
- The pre-existing dirty canonical task-status file was preserved. Seven `ddm_hv2` registration rows were appended through the canonical CLI, but that shared file is excluded from the landing because it also contains unrelated pre-existing and concurrent rows. The untracked D3B experiment was not touched.
- Serializer landing: **BLOCKED in the primary checkout**. The required serializer was invoked with post-edit SHA guards and no co-author; `git add` failed before staging with `unable to create temporary file: Operation not permitted` / `failed to insert into database`. The staged index remains empty. This memo and ledger therefore remain uncommitted in the primary checkout; no commit SHA is claimed.

## LIVE-HYPOTHESES

- JF1's terminal full-budget fits may reverse its epoch-2 byte-negative result because all seven fits reached epoch 60 and the physical terminal model+stream packs have not been compared.
- TB2's task/context weighting may outperform CB2's refuted area-tracking allocation because TB2 measured concentrated scorer debt and coder cost on different coordinates; it remains plausible only if the complete receiver closes at or below 137,986 B.
- A coder-matched three-term objective remains plausible because the current training loss omitted rate and pose, but CM1 showed that a cheap trustworthy surrogate still needs an exact-increment cache or equivalent validation.

## DEAD-ENDS

- Treating every clean `.done` receipt as unconsumed work is closed: 75 / 82 have named later consumers or superseding receipts.
- Reusing AD2's 17,957 B as current-GB1 headroom is closed: it was measured on the later authority-dead NI1/NR1 K32 object.
- Respawning the S1 ON/B/C chain is closed: the two-seed Stage-A floor entered and refused, and the family verdict did not authorize ON.
- Respawning the RC2-bound JO1 chain as the current joint-objective route is closed: it never ran, retained five blockers, and NO1 defined the materially different current-object objective.
- Using the stale hot-state number `79` as the finished denominator is closed: keeper state plus clean receipts gives 82.

Own-vehicle frontier: **GB1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]; unchanged by this read-only harvest.**
