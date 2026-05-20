# Post-PR110 Frontier Execution Queue And Staircase

**UTC:** 2026-05-20T14:12:54Z
**Author:** Codex
**Scope:** queue artifact only; no code/state/report edits
**Score claim:** false
**Promotion eligible:** false
**Dispatch attempted:** false

## Current Authority Snapshot

Canonical frontier pointer refreshed at `2026-05-20T13:47:03Z`:

| Axis | Current best | Archive SHA-256 | Evidence source |
|---|---:|---|---|
| `[contest-CPU]` | `0.1920513168811056` | `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` | `.omx/state/canonical_frontier_pointer.json` |
| `[contest-CUDA]` | `0.20533002902019143` | `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4` | `.omx/state/canonical_frontier_pointer.json` |

PR #110 post-review state:

- PR #110 source-map review is clean after wording/citation fixes.
- Release archive verified at `178517` bytes with single stored member `x`.
- SHA matches current `[contest-CPU]` frontier: `6bae0201...`.
- PR/docs posture: no new public positioning artifact until maintainer feedback.

Operational constraints from current canonical surfaces:

- Same-runtime FEC6 byte-only polish is below the useful threshold. The selector/operator audit found only about `8` selector-payload entropy bytes against about `78` charged bytes needed to cross strict `<0.192` with unchanged components.
- PR101/FEC6 PacketIR runtime-consumption proof is green, but non-identity candidates are still unmaterialized and non-promotional.
- VQ K=2 is terminalized as diagnostic-only; do not fan out more K sweeps without a K-dependent archive grammar.
- T3/T4 council cadence is over budget; this queue favors artifact production and existing gates over new high-tier deliberation.

## Recommended Immediate Frontier-Moving Artifact Path

**Build the first materialized Rule #6 / FEC6 PacketIR component-moving candidate archive** under:

```text
experiments/results/pr101_fec6_rule6_component_candidate_20260520_codex/
```

Required first artifact set:

1. `archive.zip` whose member bytes differ from the PR #110/FEC6 archive.
2. `manifest.json` with archive bytes, archive SHA-256, member SHA-256, changed PacketIR section(s), selected operator, and rationale.
3. Runtime-consumption/no-op proof showing the changed bytes are consumed by the PR101/FEC6 inflate path.
4. Local inflate smoke or exact failure classification.
5. Status fields fixed to `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false` until paired exact CPU/CUDA evidence exists.

If no queue row can be materialized into a consumed component-moving candidate, terminalize the result as `packetir_queue_has_no_materializable_component_candidate` and immediately move to TT5L doctor/source-manifest execution prep.

## Execution Queue

| Rank | Lane | Evidence axis | First command / gate | Compute class | Expected artifact | Stop / continue threshold | Blocker / claim requirement | Frontier rationale |
|---:|---|---|---|---|---|---|---|---|
| 1 | `lane_pr101_fec6_packetir_compiler_identity_queue_20260519` | Local byte/runtime proof, then paired `[contest-CPU]` + `[contest-CUDA]` only after proof | Start from `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/packetir_candidate_queue.json`; materialize one grammar-aware selector/procedural-residual candidate; rerun runtime-consumption/no-op proof | Local CPU/editor first; no provider | `experiments/results/pr101_fec6_rule6_component_candidate_20260520_codex/archive.zip` + manifest + no-op proof | Continue only if bytes differ, inflate succeeds, changed bytes are consumed, and candidate plausibly changes Seg/Pose or a consumed runtime state. Stop if every row is proxy-only or byte-polish-only below the 78-byte threshold | No dispatch claim for local build. Before exact eval: `tools/claim_lane_dispatch.py claim ...` plus paired-axis plan and terminal-claim plan | Closest artifact path: PacketIR identity and runtime-consumption surfaces already exist; this can produce a byte-closed candidate instead of more review |
| 2 | `lane_l5_v2_tt5l_side_info_effect_curve` | Paired `[contest-CPU]` + `[contest-CUDA]`; rate-only bound tracked separately | Regenerate route/doctor packet; run required Lightning doctor from `.omx/research/l5_v2_tt5l_lightning_required_doctor_plan_20260517_codex.md`; stage per-cell source manifests | Lightning or Modal exact-eval cells after doctor | 10-cell paired CPU/CUDA side-info effect curve with source manifests, archive manifests, exact eval JSONs, and terminal claims | Continue if doctor/source-manifest custody green and rate-only shrink approaches the corrected `~0.0019-0.0032` score-savings band or shows component movement. Stop if doctor fails, source manifests stale, or side-info is non-causal/no-op | Claim each cell before provider execution; no architecture lock before harvested paired cells | This is the best longer staircase path to cross `<0.192` by real rate or component movement, not selector-byte polish |
| 3 | `lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518` | Paid smoke is diagnostic until paired exact evidence; no promotion | Fresh post-fix Z6 4c smoke after verifying driver mode/env handling and no active claim conflict | Modal A10G/T4 smoke, about `$3` envelope | Harvested smoke with archive/runtime custody, full-vs-identity disambiguator rows, and Catalog #313 outcome | Continue only if the corrected 4c path actually runs the intended architecture and beats identity on the measured component target. Stop if it re-enters smoke-hardcoded/synthetic mode or misses band by implementation-class failure | Must claim lane before dispatch; stale duplicate task rows should be closed or superseded | Highest unblock leverage from tactical triage; tests whether a class-shift ego/scorer-logit path can move components |
| 4 | `lane_dreamer_v3_rssm_3_free_probes_20260520` / proposed `lane_dreamer_v3_rssm_b2_optimal_form_20260520` | Free probes only; future smoke diagnostic until exact eval | Consume 3-probe result: MPS prescreen, Dykstra feasibility, canonical equation lookup; then register/consume canonical equation and iterate design to PROCEED-unconditional | $0 design/probe; later Modal T4 `$5-15` smoke | Optimal-form design memo + equation/probe closure + optional 50-100ep smoke archive | Continue to smoke only after the blocking canonical-equation gap is closed and Catalog #315 optimal-form iteration is satisfied. Stop paid dispatch if still PROCEED_WITH_REVISIONS or `research_only=true` | Paid smoke requires lane claim and operator authorization; no dispatch while Catalog #313 blocking DEFER remains active | Best mid-term class-shift candidate, but only if it exits lifted-trainer form; otherwise it repeats the 4-of-5 failed-dispatch pattern |
| 5 | `lane_nscs06_v8_hybrid_class_shift_path_c_20260520` | Design/K-coverage first; future smoke diagnostic | Write Variant C K-coverage methodology validation and hybrid path-C design; iterate cargo-cult unwind to PROCEED-unconditional | $0 design first; later `$15-50` smoke | Design memo, K-coverage validation manifest, smoke recipe only if validation passes | Continue only if K-coverage is predicted-reachable and the runtime effect is operational. Stop if validation falsifies K-coverage or the lane is scaffold-only without `research_only=true` | Paid smoke requires claim and explicit post-validation approval | NSCS06 v6->v7 was the strongest evidence that optimal-form iteration can move score; path C is the candidate that tries to preserve that lesson without retreading failed Path B |
| 6 | `lane_master_gradient_operator_response_fec6_20260520` | Local operator rows; no score authority until exact eval | Extend grammar-aware operator rows beyond current FEC6 selector table; avoid raw archive-byte gradients | Local CPU/editor | `CandidateModificationSpec` manifest with `raw_archive_byte_rows_emitted=0`, ZIP/CRC rebuild proof, inflate proof | Continue if rows rebuild valid packets and point to component-moving candidates. Stop if rows are raw byte flips, proxy-only, or cannot prove inflate success | No provider claim until a packet-valid candidate exists | Converts sensitivity/master-gradient infrastructure into packet-valid modifications that can feed rankers without false authority |
| 7 | `lane_stc_v2_ratify_or_defer_20260520` | Diagnostic smoke; Catalog #313 outcome | Run post-fix STC v2 smoke only if no conflicting active claim | Modal T4, about `$0.20` | Harvested smoke and RATIFY-or-DEFER memo | Continue only on green smoke with runtime custody; otherwise register DEFER with reactivation criteria | Claim before dispatch | Cheap information gain; lower immediate frontier odds than FEC6/TT5L/Z6 but low cost and backlog-unblocking |

## Short / Mid / Long-Term Staircase

### Short Term: 0-72h

1. Keep PR #110 public surface quiet; monitor comments/eval only.
2. Build the FEC6 PacketIR component-moving local artifact above.
3. If materialization fails, write the exact failure classification and pivot to TT5L doctor/source-manifest prep.
4. Do not add new T3/T4 strategy deliberations unless a hard elevation trigger fires.

### Mid Term: 3-14d

1. Execute TT5L paired side-info effect cells only after doctor/source-manifest custody is green.
2. Re-fire Z6 4c only after the driver-mode fix is verified against the intended architecture.
3. Close DreamerV3 RSSM blocking free-probe/equation gap, then decide whether it can enter paid smoke.
4. Advance NSCS06 v8 path C through K-coverage validation before any spend.

### Long Term: 1-3mo

1. Mature the unified `S_total(theta, archive_bytes, hardware)` meta-Lagrangian solver so candidate ranking consumes per-axis deltas rather than prose.
2. Convert per-pair/per-byte/per-class priors into per-element routing through cathedral consumers and bit allocator surfaces.
3. Use the solver to pick the next 2 paid class-shift candidates per 7-day window, not the next council memo.

## Dispatch And Claim Discipline

- No provider dispatch from this memo.
- Every future remote/GPU/paid exact-eval job must first claim its lane with `tools/claim_lane_dispatch.py claim ...`.
- Every terminal job must append a terminal claim row.
- Every score must carry an evidence axis label.
- Local/macOS/proxy/diagnostic results may route candidates but cannot promote, rank/kill, or claim frontier movement.

## Source Surfaces Consulted

- `reports/latest.md`
- `.omx/state/canonical_frontier_pointer.json`
- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/state/active_lane_dispatch_claims.md`
- `.omx/state/canonical_task_status.jsonl`
- `.omx/state/subagent_progress.jsonl`
- `.omx/research/codex_findings_pr110_source_map_recursive_review_20260520T140229Z_codex.md`
- `.omx/research/codex_chosen_frontier_path_rule6_fec6_component_moving_20260520T121606Z_codex.md`
- `.omx/research/council_t3_grand_strategy_review_20260520T120000Z.md`
- `.omx/research/strategy_staircase_synthesis_20260520T120000Z.md`
- `.omx/research/comprehensive_plan_short_mid_long_term_20260520T120000Z.md`
- `.omx/research/task_triage_inventory_20260520T120607Z.md`
- `.omx/research/codex_session_summary_20260520T065500Z_codex.md`
- `.omx/research/codex_findings_vq_k2_diagnostic_terminalized_20260520T121133Z_codex.md`
- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/packetir_candidate_queue.json`
- `experiments/results/fec6_selector_operator_space_20260517_codex/operator_space_manifest.json`
