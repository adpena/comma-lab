# DDM DD1 drift-debt ledger verdict — 2026-09-01

## Verdict

**GENERATION 2 COMPLETE; RECURSIVE K=2 SEAL NOT EARNED.** The live harness task store, not the repository's historical task-status JSONL, currently contains **1,353 tasks: 1,163 completed, 180 pending, and 10 in progress**. Generation 1 observed 1,108/181/64; MAIN then applied 54 accepted in-progress dispositions between 2026-09-01T20:34:04Z and 20:36:00Z, while the independently running #1378 moved pending→completed. Generation 2 therefore re-read the post-application store instead of repeating the stale snapshot. This arm made no TaskList, operator-P0, hot-state, detector, equation, lane, scorer, candidate, or pointer mutation. Everything below remains an operator-facing disposition proposal.

The adoption-decay prediction is supported in the generation-1 in-progress population: **55/64 (85.94%)** rows are one-shot findings or superseded vehicle/run rows, not current work. This is a typed census result, not a claim that those 55 rows may be silently closed.

Authority boundaries:

- Task census: `/Users/adpena/.claude/tasks/89ff112f-013d-43b5-b949-2a6d43b650c3/*.json`, read at generation 1.
- `.omx/state/canonical_task_status.jsonl` is historical evidence only: 722 readable events and 2 unreadable rows in the bounded parse, not the live status authority.
- P0 census: latest-row digest from `tools/operator_p0_digest.py --json`, **92 unique P0 IDs = 59 complete + 1 completed + 29 in progress + 3 open**; active denominator **32/92**.
- Score, archive, scorer, training, and hardware were not measured. The current exact pointer is repeated only from the canonical hot state.

## RECALL EVIDENCE

| Generation | Query / source | What changed |
|---|---|---|
| 0 | charter, common contract, `PROGRAM.md`, hot state, craft handoff, live TaskList, P0 digest | Replaced the stale 54/~120 premise with the live 64/181 census and separated live status from the historical JSONL. |
| 1 | graph-memory reconstruct: `task status drift operator P0 premise warn-only purgatory` | Recovered the canonical-task-ledger supersession record, the operator-P0 recovery precedent, and the deferral ledger. This changed the plan from editing repo status rows to a proposal-only live-harness crosswalk. |
| 1 | corpus query over research/tasks/DAG/equations/docs with the same terms | Found the existing AU1, premise-lint, P0-digest, corrections-index, charter-lint, triality, and consolidation instruments. This killed the idea of adding parallel detectors. |
| 1 | AU1 corrections index + canonical-equation recall | Confirmed that task-status-versus-content is not an existing typed output and that equation/organ follow-up belongs to `ddm_lv3`, not DD1. |
| 2 | graph-memory reconstruct: `closed task stale child orphaned follow-on reparented P0 current campaign` (39,593 nodes / 162,114 edges) | Found the quadrality task-status law, the prior P0 burn-down form, #540's superseded v10 lineage, and DD1's own generation-1 node. This forced a post-application body/premise check, not merely another status count. |
| 2 | corpus query: `closed task`, `stale child`, `orphaned follow-on`, `reparent`, `P0 premise`, `status drift`, `accepted disposition` across research/state/docs/tools | Found current #1378/#1379 receipts, the IV1/OQ1 prior backlog dispositions, and the current HT1 gate4 correction. QXO1 was live during the sweep and then landed concurrently at `b34573e5f0`; its same-object n600 realization is now the successor. This changed #1363 from the generation-1 supersession proposal to OPERATOR-DECISION and exposed stale bodies in #1111/#1182/#1374. |
| 2 | AU1 corrections index SHA `af60161c...6857` + canonical-equation list (457 current equations) | Corrections were carried into the pending-tail closures; equation/organ work remains routed to `ddm_lv3`. No DD1-owned equation registration was opened. |

Seal statement: **NOT SEALED.** Generation 1 was consumed in part, but generation 2 produced new disposition rows: 110 previously deferred pending-tail rows plus body/premise repairs for #1111, #1182, #1363, and #1374. MAIN must apply or reject those rows before the next generation. There are zero consecutive dry generations, so K=2 is not claimed.

## Generation-2 post-application delta

The current 10-row in-progress denominator was inspected 10/10 and classified 10/10. No row is silently dropped.

| Disposition | Rows | Owner / consumer / fire trigger |
|---|---|---|
| **LIVE, body current** | #328, #332, #343, #381, #955, #1001 | Existing owners and consumers from generation 1; fire only on their recorded conditions. |
| **OPERATOR-DECISION** | #1111, #1363 | Operator → submission/policy decision store; fire only on an explicit policy and publish decision. #1363 is not safely superseded by #1111 because its current body contains the unresolved policy choice that blocks #1111. |
| **LIVE, premise/body repair due** | #1182 | MAIN → sub-0.12 mission umbrella; fire immediately as a ledger-only body correction: replace the rc2 claim that rate is mandatory with afr1's two-open-corners arithmetic from hot state. Keep the goal row live. |
| **LIVE-REPARENT** | #1374 | MAIN → SCMDL task body and QXO1 same-object realization consumer; fire immediately because #1378 is completed (task receipt SHA `ec864802...a609`) and QXO1 landed concurrently at `b34573e5f0` with a 129,309 B scorer-free representation row. Keep the campaign umbrella, stop naming ccs1/QXO1 as live, and route the next gate to the retained same-object n600 Seg/Pose realization. |

The generation-1 draft proposed #1363 → #1111 supersession. MAIN did not accept that proposal, and the post-application source read shows why: #1363 is an operator-policy decision package, not a completed duplicate. Generation 2 corrects that disposition rather than defending the earlier guess.

## Generation-1 task disposition ledger — all 64 then-live `in_progress` rows

Every ID appears exactly once below. “Close” always means “propose to MAIN/operator,” never an arm-side mutation.

Supersession receipt bundle, read at source: `.omx/state/main_hot_state.md` SHA-256 `8c99d16636c35cc4ce819d73c8ef501a755479497693daeb62809c06e185d916`; live harness task `/Users/adpena/.claude/tasks/89ff112f-013d-43b5-b949-2a6d43b650c3/1374.json`, whose current body names the landed dcc1 conditioning-transport law and live ccs1/#1378 rank-1 arm; and `.omx/research/ddm_wl1_20260805/TRANSFER_TABLE.md` SHA-256 `387d6101c3a3db7438f2a23006bc072dc34089a21a021231b65408f75d8820af`. The three non-SCMDL reparentings cite their live task sources directly: #540 → `ddm_lv3_recursive_leverage_20260901`, #847 → harness task #955, and #1363 → harness task #1111. These receipts support routing; they do not transfer historical scores or vehicle-specific numbers.

| Disposition | Task IDs | Owner / consumer / fire trigger |
|---|---|---|
| **LIVE** | #328, #332, #955, #1001, #1182, #1374 | #328 clip-profile maintainer → clip-profile consumers, fire when another consumer needs per-clip constants; #332 DSL/provenance maintainer → DSL completeness gates, fire before the next completeness/STRICT claim; #955 censorship-program owner → its tail ledger, fire on a named remaining leg; #1001 retention maintainer → retention census, fire before any new payload materializer; #1182 MAIN → canonical pointer, keep only as the sub-0.12 mission umbrella; #1374 MAIN → SCMDL receipts/task #1374, fire its recorded gate chain. |
| **OPERATOR-DECISION** | #343, #381, #1111 | #343 requires the explicit tunnel/deployment choice; #381 requires any new spend/cap authority; #1111 requires the already-recorded one-line publish/hosting confirmation. Consumer is the named task; no automatic fire. |
| **SUPERSEDED-BY #1374 / current afr1 campaign** | #171, #221, #223, #248, #336, #337, #366, #380, #386, #394, #396, #400, #406, #497, #539, #541, #563, #572, #578, #597, #603, #604, #609, #610, #613, #982, #995, #1074, #1185, #1186 | MAIN → task #1374/SCMDL consumer stores. Preserve receipts and close the old vehicle/run rows; transfer only an explicitly named mechanism at a current SCMDL gate, never the old score or premise. |
| **SUPERSEDED-BY another named owner** | #540 → `ddm_lv3`; #847 → #955; #1363 → #1111 | Equation/organ work belongs to `ddm_lv3`; guard migration belongs to the live censorship tail; the LLM-policy decision is already resolved in #1363's body and any publish effect belongs at #1111. |
| **FINDING-RECORDED** | #297, #349, #377, #425, #434, #445, #449, #494, #509, #535, #536, #571, #807, #823, #824, #856, #888, #891, #897, #923, #999, #1063 | MAIN → TaskList status store. Each body describes a completed measurement/build/review or an expired run/ration state. Close after preserving its cited receipt; reopen only on the row's own falsifier or a current consumer request. #1063's ration constraint remains in the standing contract even if the event row closes. |

Concentration test: `FINDING-RECORDED` 22 + `SUPERSEDED-BY` 33 = **55/64**, above the preregistered 60% drift threshold by 25.94 percentage points.

## Pending-head ledger — #1190–#1266

Selection mode was the charter's entire bounded head, not a sample: **77 numeric IDs inspected; 7 were already completed** (#1218, #1226, #1227, #1232, #1237, #1238, #1239), leaving **70 pending**. The external receipt anchor below was checked for existence; commit tokens were additionally checked as Git commits. All are **FINDING-RECORDED / proposed closure** except #1256, which remains **LIVE receipt-link repair**. This is not a bulk content endorsement: later corrections, especially the #1220 re-derivation, travel with the closure.

| IDs and verified receipt anchors |
|---|
| #1190 `955b7e4266`; #1191 `64457a2e11`; #1192 `37d9474c1a`; #1193 `37d9474c1a`; #1194 `ddm_tk1_20260806/RECEIPT.md`; #1195 `787c93e846`; #1196 `b1a0825b8a` |
| #1197 `35710b32bd`; #1198 `384ace13ad`; #1199 `ddm_ri1_rc1_full_rgb_receiver_20260822.md`; #1200 `4a49821f8f`; #1201 `ddm_cx3_context_axis_ceiling_20260822.md`; #1202 `ddm_ef1_token_entropy_floor_20260822.md`; #1203 `ddm_xs1_cross_section_conditioning_20260822.md` |
| #1204 `ddm_bl1_per_position_bit_allocation_20260822.md`; #1205 `ddm_mst1_manufactured_stage_split_20260822.md`; #1206 `873947c665`; #1207 `a2a25012d8`; #1208 `ddm_ae1_anti_predicted_excess_20260822.md`; #1209 `5c073e915`; #1210 `f6b8ab7f83` |
| #1211 `1c33f278920b91bf922e9620deb9ce20615135e8`; #1212 `5e8d6011ba`; #1213 `ddm_ar1b_archive_residue_purchase_20260822.md`; #1214 `ddm_oe1_online_escape_member_20260822.md`; #1215 `9f93ef30c3`; #1216 `362463ca80`; #1217 `72975fcaa1` |
| #1219 `d50590cf7e`; #1220 `ddm_rt3_route_rederivation_20260831.md` (**corrects 148x to 166.8x**); #1221 `ddm_jf1_joint_field_model_refit_20260823.md`; #1222 `ddm_dg2_diagonal_distortion_verdict_20260824.md`; #1223 `df757a3d24c2`; #1224 `ddm_rj1_renderer_joint_move_20260823.md`; #1225 `ddm_wa1_week_audit_gestalt_toy_orphan_synergy_20260825.md` |
| #1228 `8a571e3123`; #1229 `9c137a91ed`; #1230 `637af0c8c1`; #1231 `9c137a91ed`; #1233 `562b35b0f0`; #1234 `85f6741ff6`; #1235 `7624816b02` |
| #1236 `src/tac/pr86_hpac_codec.py`; #1240 `ddm_rf1_renderer_film_rung_20260824.md`; #1241 `1cc670031c`; #1242 `src/tac/pr130_lift/train_semantic_quantized_resumable.py` plus its append-only correction body; #1243 `bfcff07016`; #1244 `ddm_rr9_reorder_refit_20260824.md`; #1245 `a4acb886f4` |
| #1246 `1b9f88b887`; #1247 `4263fbb69b`; #1248 task-body correction plus its cited `ddm_ds1_cheap_to_shrink_objective_20260824.md`; #1249 `1b9f88b887`; #1250 `c04cfcb84c`; #1251 `18f7f44a5a`; #1252 `33095ea5fb` |
| #1253 `ddm_na11_negative_regrade_20260829.md`; #1254 `b6186cc7f7`; #1255 `0b594863f6`; #1256 **NO EXTERNAL RECEIPT LINK FOUND — LIVE**; #1257 `1ff606fc47`; #1258 `ddm_hc1_hy1_container_push_20260812.md`; #1259 `3ec36a5ddf` |
| #1260 `ddm_hc1_hy1_container_push_20260812.md`; #1261 `src/tac/boundary_math/region_merge.py`; #1262 `6f84ff79e7`; #1263 `05a549fa66`; #1264 `a4acb886f4`; #1265 `4db3f7fe2f`; #1266 `caf5c0d36a` |

#1256 disposition details: **LIVE**, owner MAIN memory-index custodian, consumer TaskList #1256 and the memory resolver, fire trigger = attach a durable external receipt that proves or corrects the claimed 18 unresolved L-key targets; only then propose closure.

## Generation-2 pending-tail ledger — all 110 pending rows outside #1190–#1266

Selection mode is a census, not a sample: **110/110 examined and 110/110 classified**. Together with generation 1's 70 pending rows in #1190–#1266, this covers the current pending denominator **180/180**. The source bundle for the older backlog is IV1 SHA `db8250c8...7912`, OQ1 memo SHA `3418f8a7...cc92`, and OQ1 JSON SHA `ea120d3b...3e75`; recent rows were read from their live task bodies and their named memo/commit receipts. Supersession uses the same hot-state/#1374/WL1 source bundle pinned above. No task status was edited.

| Disposition | Task IDs | Owner / consumer / fire trigger |
|---|---|---|
| **FINDING-RECORDED** | #236, #450, #834, #835, #848, #849, #857, #862, #882, #894, #896, #901, #906, #911, #912, #915, #916, #917, #918, #919, #924, #933, #934, #939, #954, #971, #985, #986, #992, #994, #1000, #1087, #1088, #1089, #1090, #1091, #1095, #1169, #1305, #1328, #1337, #1338, #1339, #1340, #1341, #1342, #1343, #1344, #1345, #1346, #1347, #1348, #1349, #1350, #1351, #1352, #1355, #1357, #1375 | MAIN → live TaskList status store; close after retaining the task body and its cited receipt/correction. #236/#450 are source-verified landed in IV1; #939 closes only with #1357's correction attached; #1305 closes against HT1 SHA `05c4887f...72c5`, not by raising the stale bound; #1328/#1375 are landed cures with separately routed conditional follow-ons. |
| **SUPERSEDED-BY #1374 / current afr1 campaign** | #51, #137, #183, #211, #213, #222, #226, #227, #319, #408, #556, #573, #611, #659, #669, #706, #729, #815, #920, #926, #940, #949, #990, #1038, #1092, #1156 | MAIN → SCMDL/QXO1/current campaign stores; preserve the historical task body, transfer only a mechanism named by the current owner, and close the retired vehicle/run row. No historical score or old training premise transfers. |
| **SUPERSEDED-BY named ledger owner** | #860 → #670; #875 → #955; #984 → #1182 | MAIN → warn-only debt, censorship tail, and sub-0.12 mission stores respectively; close the duplicate event row only after the consumer body names the transferred residue. |
| **OPERATOR-DECISION** | #199, #1168 | Operator → Git/worktree custody and GCP-key verification stores. Fire only with explicit authority for the proposed branch/worktree cleanup or the bounded console check; DD1 performs neither. |

The 20 rows that remain live need explicit ownership and triggers rather than a generic “pending” label:

| Row(s) | Live action | Owner → consumer store | Fire trigger |
|---|---|---|---|
| #198 | Factor the private fleet loader, migrate `bat00.py`, add the no-hardcoded-IP guard. | fleet-loader owner → shared fleet-config consumer | Next apparatus slot touching fleet configuration. |
| #252 | Continue the standing MLX/Metal optimization program without treating it as score authority. | compute maintainer → MLX/Metal benchmark and parity stores | A named compute increment with NumPy-fp32 parity and benchmark custody. |
| #670 | Drain LandingDiffManifest and RED-dev remediation; lane and raw-duplicate sublegs are already dispositioned. | landing/apparatus maintainer → landing ledger | Valid future BASE..HEAD receipts plus typed legacy boundary; route #860's genuine residue here. |
| #716 | Finish or retire the S-primacy/objective-contradiction structural cure without duplicating equation work. | `ddm_lv3`/apparatus owner → existing equation and window-gate stores | Source check shows an unconsumed law or guard leg on the current path. |
| #833, #840, #844 | Resolve the degenerate-control gap, the bounded corpus-sweep denominator, and triality's path-prefix scope using existing instruments. | current detector owners → preflight/triality/AU1 stores | A named positive control or reproducible live false-positive; never a new twin detector. |
| #974, #977, #979 | Re-parent the true-domain, fractal-instrument, and no-toy/hybrid standing laws to current charter review. | charter-lint/instrument owners → current campaign charter gate | Next mechanism charter or detector incident. |
| #1162 | Re-evaluate CUDA decode overhead only as part of the selected submission packet. | #1111 runtime owner → submission packet | Operator clears #1363 policy and #1111 publish gates. |
| #1164 | Fix the three named SSD-review defects and adjudicate testpaths/cadence. | SSD review owner → code-review debt store | Quiet apparatus boundary; no score claim. |
| #1171 | Repair the 13 off-lineage asymmetric quantize/dequantize sites with a class regression. | codec hygiene owner → future-vehicle guard | Next touch of those sites or a scheduled hygiene window. |
| #1181 | Re-run the seeded random carrier-price check only if the Road↔Undrivable price is consumed. | current carrier-price consumer → pricing receipt | A downstream decision cites the prefix-derived factor. |
| #1273 | Cure resume-within-stage schedule extension and sweep sister trainers. | trainer resumability owner → checkpoint/resume tests | Next WD3/sister-trainer touch. |
| #1280 | Backfill/waive 14 council memos and strict-flip at zero with the positive control retained. | Council gate owner → Catalog #363 preflight | Free apparatus slot after strict REDs are drained. |
| #1282 | Adjudicate 13 pre-existing RED tests as stale fixtures versus real regressions. | affected gate/test owners → test and defect ledgers | Before any bound change; each row gets source-level verdict. |
| #1295 | Re-parent field-for-coder co-optimization to the current changed-coder object; do not infer viability from QXO1's byte-only landing. | #1374/QXO1 realization owner → current SCMDL consumer | The retained QXO1 object receives same-object n600 distortion and survives its gate, or another current conditional coder lands with an exact surprise decomposition. |
| #1306 | Replace ps-RSS with physical-footprint authority for Metal workloads and execute a positive control. | resource-safety owner → `safe_run`/liveness stores | Next Metal resource-enforcer landing. |
| #1380 | Keep the four-arm wave row open only until DD1's serializer landing is consumed; the other three named arms are terminal. | MAIN → TaskList #1380 | DD1 commit lands; then mark this wave FINDING-RECORDED. |

Generation-2 pending outcome: **59 FINDING-RECORDED + 29 SUPERSEDED-BY + 2 OPERATOR-DECISION + 20 LIVE = 110/110**. Operator-queued rows in this tail are 2/110; MAIN disposition proposals are 88/110; live owned rows are 20/110.

## Active operator-P0 digest — all 32 rows

These are recommendations for operator disposition. The P0 ledger was not edited.

| P0 ID | Disposition | Owner, consumer, and fire trigger |
|---|---|---|
| `p0_1111_completion_and_frontier_lowering_20260817` | OPERATOR-DECISION | Operator → #1111; fire only on explicit one-line publish/hosting confirmation. |
| `p0_328_clip_profile_rewire` | LIVE | Clip-profile maintainer → `tac.clip_profile` consumers; fire on a new consumer or Phase-3 consolidation window. |
| `p0_332_provenance_bijection_backfill_20260717` | LIVE | DSL maintainer → provenance/DSL gates; reverify current branch ancestry and fire the owed strict flip only at zero. |
| `p0_343_dashboard_engineering` | OPERATOR-DECISION | Operator → dashboard supervisor; fire only if the tunnel/deployment is wanted. |
| `p0_366_joint_pose_finishing` | SUPERSEDED-BY #1374 | MAIN → SCMDL; preserve the terminal-solve lesson, retire the old vehicle premise. |
| `p0_408_telemetry_resume_boundary` | FINDING-RECORDED | MAIN → P0 ledger; merged/consumed premise is historical, so close after ancestry receipt check. |
| `p0_497_basis_cure_decisive_ab` | SUPERSEDED-BY #1374 | MAIN → SCMDL; no transfer of the old proxy claim without current matched evidence. |
| `p0_SUPREME_duty_queue_old_findings_first_20260717` | LIVE-REPARENT | MAIN → DD1 dispositions/current task store; fire generation 2 after accepted closes/reparents land. |
| `p0_UNIFICATION_projection_preimage_SUPREME_20260715` | SUPERSEDED-BY #1374 | MAIN → SCMDL current-object gates. |
| `p0_all_lenses_facets_of_unification_20260715` | SUPERSEDED-BY #1374 | MAIN → SCMDL; old c2 blocker is expired. |
| `p0_always_keep_the_payload_20260809` | LIVE | Retention maintainer → retention census/preflight; fire before new payload-producing launches and at the named A1/A3 decision. |
| `p0_autonomous_frontier_loop_20260812` | SUPERSEDED-BY #1374 | MAIN → current SCMDL loop; retire the stale sa1 live-run premise. |
| `p0_bug_class_sweep_20260717` | LIVE-REPARENT | Apparatus owner → existing bug-class/tail ledgers; fire only on a named remaining class, not a broad new sweep. |
| `p0_codex_findings_consumption_audit_20260717` | LIVE-REPARENT | AU1/MAIN → named consumer ledger; fire on an unowned landed finding or the weekly cadence. |
| `p0_costate_organ_factorization_grounded_ABC_20260718` | LIVE-REPARENT TO `ddm_lv3` | `ddm_lv3` owner → organ/equation stores; DD1 does not duplicate the factorization work. |
| `p0_crux_knee_recursive_fractal_20260721` | SUPERSEDED-BY #1374 | MAIN → SCMDL current-object schedule. |
| `p0_ema_calibration_20260717` | SUPERSEDED-BY #1374 | MAIN → SCMDL; transfer only if the current restartable coder has the same EMA surface. |
| `p0_fisher_full_leverage_20260717` | SUPERSEDED-BY #1374 | MAIN → SCMDL; old witness natural-gradient premise is not current authority. |
| `p0_instrument_fractal_audit_20260806` | LIVE-REPARENT | Apparatus owner → existing detector map; fire on a detector incident or cadence, not an unbounded audit. |
| `p0_inverse_solve_gap_register_20260721` | SUPERSEDED-BY #1374 | MAIN → SCMDL current gate ledger. |
| `p0_jg5_custody_version_shipping_decoder_20260820` | OPERATOR-DECISION | Operator → shipping/custody store; decide the remaining absolute-path/compress-staging choices explicitly. |
| `p0_lane_three_cruxes_20260717` | SUPERSEDED-BY #1374 | MAIN → SCMDL; current Lane demand is governed by current receiver/object evidence. |
| `p0_naive_toy_allergy_permanent_fix_20260806` | LIVE-REPARENT | Charter-lint owner → existing charter gates; fire when a new mechanism-reduced charter is proposed. |
| `p0_null_subspace_gauge_kerA_20260717` | SUPERSEDED-BY #1374 | MAIN → SCMDL current-object law. |
| `p0_realization_limited_not_gradient_20260715` | SUPERSEDED-BY #1374 | MAIN → SCMDL current receiver gate. |
| `p0_segnet_fractal_cluster_20260715` | SUPERSEDED-BY #1374 | MAIN → SCMDL; no old c2 number transfers. |
| `p0_session_standing_turn_contract_20260816` | FINDING-RECORDED | MAIN → common contract; close the event row after verifying the standing contract carries the rule. |
| `p0_swap_procedure_no_push_without_confirm_20260817` | OPERATOR-DECISION | Operator → #1111; remains a binding no-publish-without-confirm constraint. |
| `p0_todo_class_is_p0_20260817` | FINDING-RECORDED | MAIN → TaskList/AU1 cadence; keep the process law, retire the one-shot P0 event after detector consumption. |
| `p0_triggers_forces_review_all_findings_20260717` | SUPERSEDED-BY #1374 | MAIN → current SCMDL trigger ledger. |
| `p0_true_domain_optimization_triple_20260806` | LIVE-REPARENT | Charter-lint/current-campaign owner → SCMDL charter review; fire on the next mechanism charter. |
| `p0_v10_capstone_cold_start_seeded_20260717` | SUPERSEDED-BY #1374 | MAIN → SCMDL; old v10 cold-start vehicle is not the current candidate chain. |

P0 concentration: **15 superseded + 3 finding-recorded = 18/32 (56.25%)** premise/status debt; **10 live/reparented** and **4 operator decisions** remain. No active row was silently dropped.

## Warn-only purgatory and cost to clear

All costs are **PROJECTED engineering time**, not measurements of elapsed work.

| Cluster | Current evidence | Disposition and projected cost | Strict-flip condition |
|---|---|---|---|
| #670 LandingDiffManifest | 548 readable landing-ledger rows; 481 terminal; 261 rows carry a typed manifest object and **261/261 explicitly report `landing_diff_manifest_missing`**. | EXTEND existing delegation/landing contract; do not add a gate. Future-row cure 1–2 days; historical 481-row adjudication 2–4 days if demanded. Prefer a typed legacy boundary over invented backfill data. | New terminal rows emit valid BASE..HEAD receipts; historical rows are either honestly typed legacy or backfilled from custody. |
| #670 lane registry | `tools/lane_maturity.py validate` is now **0 violations / 2,295 lanes**. | FINDING-RECORDED; retire the stale “110” sub-debt, $0 further work unless it regresses. | Already clean on this read. |
| #670 exact duplicate | The named `witness_realization_lsb_regime_v1` has two append-only events; latest-row-wins makes it non-corrupting. Raw repeated equation IDs are not themselves defects in an append-only event registry. | Route the named exact-dup semantics to `ddm_lv3`; under 1 hour to add a supersession/tombstone interpretation if one is still absent. | Validator distinguishes legitimate event history from simultaneous live definitions. |
| #1280 / Catalog #363 | Direct function call reproduced **14 violations / 14 named council memos**. | Backfill the required verification-status token or reasoned waiver in each memo, 2–4 hours, then rerun. | Exact count zero, positive-control test still catches a missing-status memo, same landing flips WARN→STRICT. |
| #860 RED-dev population | Latest task receipt says **163 remaining = about 6 genuine code debts + 124 outside-memory structurally unclearable + 25 policy + 7 false positives + 1 policy phrase**. This arm did not rerun its mutable aggregate counter. | Fix the 6 code sites with their owners, about 1–2 days; separately adjudicate/re-scope the 157 non-code rows, about 0.5–1 day. Do not edit 51 compliant substrate files or register fake lanes. | Re-derived zero genuine defects plus explicit policy/out-of-scope typing; stable on two consecutive reads before hook flip. |

## Hot-state and consolidation drift

- `main_hot_state.md` self-reports stale `live_processes` and `monitor_tasks` (1.14 days > 1.0 day). Its process text still names wwc1, while #1360 is completed. Its watches still name lc3 and vr2, while #1361 and #1165 are completed; #1365's Lane-generator floor is also completed. Proposal: MAIN re-render those two sections and retire the lc3/vr2 watches. DD1 did not edit the hot state.
- Fresh `tools/consolidation_debt.py --json`: **CONSOLIDATE-NOW, severity 2; 45 pile files; 1,438 pile lines; 1 system-intelligence landing; 20 stale commits; 89.0 memo/landing signal ratio; 118 SSD-only code files**. The SSD cache was fresh (`measured_at_utc=2026-09-01T19:39:33Z`, 140,470 paths scanned); the charter's stale 96 and the generation-1 draft's transient 13/80.0 values must not be quoted as current.
- This is a custody/consolidation signal, not permission to delete or move any file. Certify-or-block remains binding.

## Detector coverage map and cadence proposals

| Surface | Existing instrument | Gap / smallest extension |
|---|---|---|
| Headline/body corrections | AU1 corrections index and `au1_measurement_integrity_audit.py` | Covered for correction linkage, not TaskList state. |
| Correction freshness | `corrections_index_freshness.py` + charter lint | Covered. |
| Charter/premise discipline | `codex_arm_queue.py`, `premise_lint.py`, premise registry | Covered at charter time. |
| Triality | `triality_drift_detector.py` | Covered. |
| Consolidation/SSD | `consolidation_debt.py` | Covered; refresh before quoting. |
| Task status vs terminal content | **Uncovered as a typed output** | Extend AU1 to emit `status_content_mismatch` when a pending/in-progress task cites a terminal receipt/commit. Report-only; never auto-close. |
| P0 premise drift | P0 digest lists status, evidence, watch paths, and verification time but does not join them | Extend `operator_p0_digest.py` with `premise_reverify_due` and a reparent suggestion using task status, watch-path existence, correction index, and premise-lint state. Never auto-dispose a P0. |

At most five recurring proposals:

1. **Weekly and SessionStop:** AU1 task-status/content join; owner AU1 maintainer; consumer MAIN TaskList review; trigger any typed mismatch.
2. **Weekly:** P0 premise-TTL/reparent join in the existing digest; owner P0 digest maintainer; consumer operator P0 review; trigger expired verification or terminal watched task.
3. **Each terminal landing:** existing LandingDiffManifest emission, with a weekly burn-down of legacy rows; owner delegation/landing maintainer; consumer landing ledger; trigger missing/invalid receipt.
4. **On watched-task terminal transition and daily TTL:** hot-state renderer reconciliation; owner MAIN; consumer `live_processes`/`monitor_tasks`; trigger completed watched task or stale section.
5. **Weekly or before consolidation claims:** corrections-index + consolidation refresh; owners existing maintainers; consumers AU1/SSD debt stores; trigger stale cache or a changed SSD code count.

## Live hypotheses

- The 85.94% one-shot/superseded share will recur after MAIN applies generation 1 because most status debt is created by event-shaped tasks without a terminal status-consumption cadence. This is plausible from the measured concentration, but generation 2 is required to test it.
- Joining P0 verification age to watched-task terminal status will explain a large fraction of the 18/32 active P0 premise/status debts. It is plausible because several active rows explicitly watch already-merged or completed work.
- A future-only LandingDiffManifest strict boundary plus honestly typed legacy rows will stop new purgatory without fabricating historical BASE..HEAD receipts. It is plausible because every row that currently carries the manifest field reports the same missing-receipt blocker.
- #1256's 18 unresolved L-key claim may be correct but is not closure-grade until a durable external receipt is linked; the detailed task body makes it plausible, while the missing anchor keeps it unverified here.
- QXO1's 129,309 B target-overwrite representation may survive same-object realization despite BR2's adverse prior because it changes 8,749 semantic sites and sits 8,676 B under the byte gate. This is plausible enough for the retained n600 realization, but no distortion or score transfers from another object.

## Dead ends

- Editing `.omx/state/canonical_task_status.jsonl` as the live task store: closed because its own supersession record and the live harness disagree; it is historical evidence only.
- Bulk-closing #1190–#1266 from prose shape alone: closed because receipt linkage matters; every proposed closure has an external anchor, and #1256 remains live.
- Adding new drift-detector tools: closed because AU1 and the P0 digest are the narrower existing extension points.
- Treating all raw duplicate equation IDs as defects: closed because the registry is append-only; only simultaneous live-definition ambiguity is actionable, and that work belongs to `ddm_lv3`.
- Re-citing #670's 110 lane failures, #860's older 316/231/210 counts, or the SSD count 96: closed by current 0/2,295 lane validation, #860's typed 163 decomposition, and the refreshed SSD count 118.
- Claiming recursive K=2 completion now: closed because generation 2 produced new disposition rows, MAIN has not yet applied or rejected them, and the dry-generation count is 0/2.
- Treating QXO1's under-gate byte row as a score win: closed because it ran no scorer and BR2's distortion belongs to a different object; only same-object retained realization can answer the score question.

S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600], afr1 sha cbb8d928…d405bf25 — UNMOVED
