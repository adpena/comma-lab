# ddm_wk4 orphan-token sweep

Date: 2026-08-04
Agent: ddm_wk4 respawn #3
Source data: `.omx/tmp/wk4_scan_result.json`
Checkpoint: `.omx/tmp/wk4_ckpt.json`

## Verdict

The persisted scan is complete and this memo does not rescan the corpus.
The scan classified score-relevant numeric-token occurrences by exact
normalized-token presence in the bounded consumer index:

- CONSUMED: at least one normalized numeric form appears in the bounded consumer
  surfaces.
- ORPHAN: no normalized numeric form appears in the bounded consumer surfaces.

Consumer surfaces are exactly the original charter surfaces recorded in the
scan result: `src/`, `tools/`, `experiments/`, `configs/`, plus
`.omx/state/canonical_equations_registry.jsonl` and
`.omx/state/canonical_task_status.jsonl`, with the recorded 1 MB per-file cap.

This is a token-level bounded absence result. "Orphan" means "not found in the
bounded consumer index used by this scan"; it does not mean the surrounding
research claim globally lacks an owner.

## Full Denominators

| Denominator | Count |
|---|---:|
| docs scanned | 7,050 |
| docs with score tokens | 4,559 |
| docs with orphans | 2,049 |
| score-relevant token occurrences found | 117,336 |
| consumed occurrences | 107,562 |
| orphan occurrences | 9,774 |
| orphan occurrences with est_S >= 0.001 | 9,032 |
| consumer files indexed | 10,585 |
| consumer numeric occurrences indexed | 778,913 |
| consumer numeric keys | 26,339 |

Consumer key file: `.omx/tmp/wk4_consumer_numeric_keys.txt`
Consumer key sha256: `da4ff7cecc898a52fd8926f640d4339ad955bb6c783bfbcd2bf939646e2afb2f`

## Metric Split

| metric | total occurrences | consumed | orphan |
|---|---:|---:|---:|
| bytes | 32,571 | 29,271 | 3,300 |
| d_pose | 21,453 | 19,551 | 1,902 |
| d_seg | 15,080 | 13,170 | 1,910 |
| percent_of_gap | 1,011 | 859 | 152 |
| rate_ratio | 3,152 | 2,822 | 330 |
| rate_term | 1,446 | 1,280 | 166 |
| score_delta | 39,496 | 37,730 | 1,766 |
| score_value | 3,127 | 2,879 | 248 |

## Exclusions And Boundaries

- No scorer was run.
- No training was run.
- No receiver or inflate path was executed.
- No receiver edits, scorer edits, or code edits were made.
- No score was measured by this memo.
- The scan was static and numeric-token based; it can rank parser artifacts
  above real score-bearing research findings.
- Consumer index rule: source-like files under `src/`, `tools/`,
  `experiments/`, and `configs/` up to 1 MB, plus the two explicit canonical
  state ledgers. All other `.omx/state` files were excluded from the consumer
  index.
- Target exclusion rule: target Markdown docs over 1 MB were excluded from the
  target scan. Per the m50 vacuity law, this memo makes no orphan or consumed
  claim about excluded target contents.
- Excluded large target doc: `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- Excluded large consumer source files:
  `src/tac/preflight.py`,
  `tools/run_compact_renderer_mlx_spine_runner.py`,
  `experiments/train_levelset_witness_realized_through_R_mlx.py`.
- Excluded large non-source files under code roots: 5.
- Excluded other `.omx/state` files: 2,100, including 208 over 1 MB.
- `gk2` decode-read surfaces were excluded from targets.

## Ranked Orphan Tokens By est_S

N = 25, taken directly from `ranked_orphans` in the persisted result. The
"actionability / route" column distinguishes true score-consumer debt from
static-token vacuity and operational-byte artifacts.

| rank | token | metric | est_S | source | actionability / route |
|---:|---|---|---:|---|---|
| 1 | `98,109,112,134,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588` | bytes | 6.53268e+76 | `.omx/research/codex_findings_dqs1_rank019_queue_worker_drift_hardening_20260522T201951Z_codex.md:90` | Pairset ID list parsed as archive bytes. Route to `canonical_task_status.jsonl` only if DQS1 dynamic-sweep candidate membership still matters; otherwise scanner-exclusion debt. |
| 2 | `98,109,112,134,151,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588` | bytes | 6.53268e+76 | `.omx/research/codex_findings_dqs1_pairset_observation_feedback_20260522T164706Z_codex.md:566` | Pairset ID list parsed as archive bytes. Same DQS1 candidate-membership route as rank 1; otherwise scanner-exclusion debt. |
| 3 | `4e40` | bytes | 2.66344e+34 | `.omx/research/MEMORY_archive_2026Q2.md:196` | Memory/prose numeric artifact, not a score row. Route to scanner hardening ledger; no contest-score consumer. |
| 4 | `1.8e19` | bytes | 1.19855e+13 | `.omx/research/wyner_ziv_q4_tier_2_comma2k19_smoke_packet_design_20260518.md:401` | Seed entropy bound. Route to canonical equation `seed_entropy_sufficiency_v1` or append to existing procedural-codebook equation if the entropy bound is still used. |
| 5 | `90,175,277,381,424,573` | bytes | 6.0044e+10 | `.omx/research/v10_capstone_state_review_20260719_codex.md:14` | n6 sample-membership list parsed as bytes. Route the actual feasibility result to a `v10_shared_scorer_plane_feasibility_n6` task row if still live. |
| 6 | `1,019,568,099,328` | bytes | 678889 | `.omx/research/hprc_hierarchical_predictive_receiver_codec_design_20260531T223400Z_codex.md:215` | Free disk bytes, not archive bytes. Route to storage waterfall / operator P0 ledger only if the old admission fact still matters. |
| 7 | `827,380,576,256` | bytes | 550919 | `.omx/research/codex_findings_perclass_convergence_ab_20260714_codex.md:43` | Free disk bytes / sandbox permission blocker. Route to storage waterfall ledger; no score consumer. |
| 8 | `827,380,576,256` | bytes | 550919 | `.omx/research/launch_prego_worklist_20260713.md:15` | Duplicate storage-admission fact. Route with rank 7. |
| 9 | `827,380,576,256` | bytes | 550919 | `.omx/research/launch_prego_worklist_20260713.md:95` | Duplicate storage-admission fact. Route with rank 7. |
| 10 | `824,820,822,016` | bytes | 549214 | `.omx/research/v10_uint8_lattice_feasibility_receipt_20260718.md:92` | Free disk bytes. Route to storage waterfall ledger; no score consumer. |
| 11 | `816,407,822,336` | bytes | 543612 | `.omx/research/c1_two_plane_receiver_timing_20260719_codex.md:59` | Free disk bytes. Route to storage waterfall ledger; no score consumer. |
| 12 | `779,043,606,528` | bytes | 518733 | `.omx/research/codex_findings_r1b_boundary_generator_solve_20260720_codex.md:29` | Operational/free-space byte reading near raw-output custody text. Route to storage/custody ledger, not score. |
| 13 | `773,458,149,376` | bytes | 515014 | `.omx/research/c1_two_plane_receiver_timing_20260719_codex.md:60` | Usable disk bytes. Route to storage waterfall ledger; no score consumer. |
| 14 | `690,998,198,272` | bytes | 460107 | `.omx/research/r1b7_uint8_survival_carrier_DAG_FEED_20260720T224624Z.md:17` | SSD free bytes. Route to storage waterfall ledger; no score consumer. |
| 15 | `519,527,540,541` | bytes | 345932 | `.omx/research/snerv_fullstack_extreme_scrutiny_vs_evaluate_py_20260609.md:21` | Parser artifact from prose/context, not a validated archive-size finding. Route to scanner hardening ledger unless source reread proves live SNeRV consumer debt. |
| 16 | `424,198,135,808` | bytes | 282456 | `.omx/research/codex_findings_ddm_p1_frame0_pose_quotient_carrier_20260725T143303Z_codex.md:264` | Storage admission bytes. Route to storage waterfall ledger; no score consumer. |
| 17 | `91,979,186,176` | bytes | 61245.2 | `.omx/research/codex_findings_ddm_p1_frame0_pose_quotient_carrier_20260725T143303Z_codex.md:262` | RAM admission bytes. Route to storage/admission ledger; no score consumer. |
| 18 | `90,005,356,544` | bytes | 59930.9 | `.omx/research/codex_findings_ddm_e5a_midcampaign_e5_adapter_20260725T132247Z_codex.md:51` | psutil availability bytes. Route to storage/admission ledger; no score consumer. |
| 19 | `44,106,148,154` | bytes | 29368.5 | `.omx/research/codex_findings_no_fourier_basis_20260715_codex.md:106` | Source line-number list parsed as bytes. Route to scanner hardening ledger; no score consumer. |
| 20 | `25,769,803,776` | bytes | 17159.1 | `.omx/research/c1_two_plane_receiver_timing_20260719_codex.md:58` | Storage waterfall requirement. Route to storage waterfall ledger; no score consumer. |
| 21 | `25,637,045,658` | bytes | 17070.7 | `.omx/research/codex_findings_dqs1_local_harvest_queue_hardening_20260523T124000Z_codex.md:19` | Certified scratch cleanup bytes. Route to storage hygiene ledger; no score consumer. |
| 22 | `2.28e10` | bytes | 15181.6 | `.omx/research/generator_dseg_powerlaw_to_frontier_20260621.md:78` | Real curve-fit/power-law numeric, but the token is not an archive byte. Route to canonical equation for generator d_seg power-law only if still used for launch math. |
| 23 | `21,474,836,480` | bytes | 14299.2 | `.omx/research/codex_findings_ddm_p1_frame0_pose_quotient_carrier_20260725T143303Z_codex.md:262` | RAM gate threshold. Route to storage/admission ledger; no score consumer. |
| 24 | `21,474,836,480` | bytes | 14299.2 | `.omx/research/codex_findings_ddm_p1_frame0_pose_quotient_carrier_20260725T143303Z_codex.md:264` | Storage gate threshold. Route with rank 23. |
| 25 | `14,649,816,858` | bytes | 9754.71 | `.omx/research/codex_findings_dqs1_autopilot_retention_executor_20260523T112909Z_codex.md:66` | Certified artifact move bytes. Route to storage hygiene ledger; no score consumer. |

## Top Routing Decisions

1. DQS1 pairset lists at ranks 1-2: if DQS1 is still referenced by any live
   launch or historical receipt, create or update a canonical task-status row
   for `dqs1_dynamic_sweep_pairset_membership`; otherwise record them as
   scanner-vacuity exclusions. They are not archive bytes.
2. `1.8e19` seed entropy at rank 4: route to a canonical equation
   `seed_entropy_sufficiency_v1`, or attach it to the existing procedural
   codebook equation lineage. This is the highest ranked non-operational concept
   token in the top 10.
3. v10 n6 sample list at rank 5: route the real finding, not the list token, to
   a `v10_shared_scorer_plane_feasibility_n6` task row if the feasibility result
   still informs any receiver/witness design.
4. Storage/free-space/admission byte tokens at ranks 6-14 and 16-25: absorb
   only in a storage waterfall or operator P0 hygiene ledger. They must not feed
   contest rate accounting.
5. Parser artifacts at ranks 3, 15, and 19: route to scanner hardening debt if
   this scan will be repeated. They have no score consumer.
6. Generator d_seg power-law token at rank 22: route to a canonical equation only
   if the law remains an active launch prior; otherwise leave it as historical
   orphaned research context.

## Boundary Statement

This memo is a completion artifact for the persisted `ddm_wk4` scan, not a new
measurement. It does not prove any score movement, and it does not promote any
advisory, proxy, partial-sample, static-token, storage-admission, or parser
artifact into score authority.

own-vehicle frontier S = 0.7910689 @ 353,805 B [macOS-CPU advisory] -- UNMOVED.
