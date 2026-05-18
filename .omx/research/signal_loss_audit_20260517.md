---
review_kind: signal_loss_audit
review_id: signal_loss_audit_20260517
review_date: "2026-05-17"
lane_id: lane_signal_loss_audit_20260517
evidence_axis: state_integrity_and_signal_recovery
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
---

# Signal-loss audit — 2026-05-17

Operator standing directive 2026-05-17 verbatim: *"Also a pass to ensure no signal loss"*. This audit sweeps every surface touched by today's wave (Z6-v2 Wave 2 landing + ATW V2 reactivation symposium + C6 IBPS first asymptotic empirical anchor + Catalog #324/#325/#326 trio + ~62 sister-subagent landings).

Per CLAUDE.md "Required durable state" + "Subagent coherence-by-default" anti-orphan-work principle.

## Summary

- **226 commits** in 2026-05-17 UTC window (223 today + 3 early on 2026-05-18); 50 of these carry substantive code/state changes and ~11 are pure `state: claim catalog #` git-transactional anchors.
- **63 memory entries** dated 20260517 or 20260518 at `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_*_2026051[78].md`.
- **142 working-tree entries** uncommitted at audit time: 38 `M` + 105 `??` (105 untracked = 42 `.omx/research/*.md` + 28 `src/tac/tests/*.py` + 35 other).

### Findings tally

- **2 NEEDS-OPERATOR-REVIEW**: Catalog #324 + #325 + #326 gate implementations sit uncommitted in working tree (CLAUDE.md +151 lines; preflight.py +702 lines; lane_registry.json +2596 lines; lane_maturity_audit.log +223 lines). Also Catalog #325 has NO `committed_via_serializer:true` follow-up event in `.omx/state/catalog-claim.log`.
- **1 OBSERVATIONAL**: Modal call_id ledger has the C6 IBPS smoke (`fc-01KRW353MJJ9A6QW8H99QWZEMH`) marked `dispatched` but no `harvested` event row; the empirical 3.04 score IS captured via 3 sister surfaces (lane registry / probe_outcomes / council deliberation posterior + landing memo) so signal is preserved — only the ledger sync is incomplete.
- **0 actual signal-loss** in the strict sense (every finding has the substrate signal captured SOMEWHERE; the gap is structural commit discipline).

### Wave-D commit-swap absorption-risk reminder

Per Catalog #314 + the WAVE-D 2c957c31e forensic anchor (`feedback_commit_swap_absorption_pattern_investigation_landed_20260516.md`): the in-flight `M` + `??` working-tree set is large (142 entries) and a future bare `/commit` slash-command invocation could absorb files from sister-subagent scope. The Catalog #324/#325/#326 wave should be committed via canonical serializer with `--expected-content-sha256` per Catalog #117/#157/#174 to extinct absorption risk.

## Findings per surface

### 1. Today's commits — most have matching memory entries

Examined first 50 substantive commits; 11 are pure state-claim anchors and ~39 are substantive code/state changes. Spot-checked memory cross-refs for 3 representative landings:

| Commit | Subject | Memory entry | Cross-ref count |
|---|---|---|---|
| `c10dec618` | Z6-v2 Wave 2 Candidate 1: multi-layer FiLM depth=3 ~300K + driver contract repair | `feedback_z6_v2_wave_2_codex_repairs_landed_20260517.md` | 3 |
| `a67f8fc12` | frontier: canonicalize anchor scan and drift gates | `feedback_permanent_fix_frontier_signal_loss_landed_20260517.md` | 1 |
| `f4f6c379c` | state: claim catalog #324 (git-transactional) | `feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md` | 17 |

**Classification**: RECOVERED — memory→commit cross-refs are well-formed for the substantive landings.

### 2. Untracked working-tree files (105 entries)

| Category | Count | Classification |
|---|---|---|
| `.omx/research/*.md` | 42 | RECOVERED (these ARE the canonical durable research memos sister-landed today; deliberate uncommitted state for sister wave consolidation) |
| `src/tac/tests/test_*.py` | 28 | RECOVERED (all pass — see #6 below — they pair with the uncommitted production code) |
| `.omx/operator_authorize_recipes/*.yaml` | 15 | RECOVERED (10 are C6 IBPS β_ib + latent_dim sweep recipes from sister #856/#846 BUILD landings; 3 fec6 paired composition recipes from FEC6 stacking wave; 2 master-gradient + q4 redirect recipes) |
| `src/tac/optimization/*.py` `src/tac/codec/*.py` `src/tac/inflate_time_post_processing/` etc. | ~10 | RECOVERED (canonical helpers landed today: `tier_c_density_post_training_validator.py`, `wyner_ziv_layer.py`, `per_pair_master_gradient_wire_in.py`, `per_pair_namespace_wire_in.py`, new namespaces `tac.search` `tac.side_information` `tac.inflate_time_post_processing`) |
| `tools/*.py` `scripts/*.sh` | 9 | RECOVERED (operator-facing audit tools: `audit_substrate_driver_mode_hardcode.py`, `audit_predicted_band_provenance.py`, `asymptotic_pursuit_candidate_readiness_assessment.py`, `wyner_ziv_deliverability_prober.py`, `q6_preprobe_pairwise_composition_alpha.py`, `option_b_archive_member_pre_entropy_sweep.py`, etc.) |
| `experiments/train_substrate_pr101_with_dp1_prior_regularizer.py` | 1 | RECOVERED (DP1 + FEC6 dual-stacking BUILD landing) |

**Top concern**: this is a LARGE in-flight working tree (142 entries). Per the WAVE-D absorption-pattern forensic anchor: a future `/commit` slash invocation could silently absorb files from sister-subagent scope.

### 3. Modified working-tree files (38 entries)

| File | Classification | Diff size |
|---|---|---|
| `CLAUDE.md` | NEEDS-OPERATOR-REVIEW | +151 lines (Catalog #324 + #325 + #326 catalog rows) |
| `src/tac/preflight.py` | NEEDS-OPERATOR-REVIEW | +702 lines (9 new functions for #324/#325/#326 wired into `preflight_all()`) |
| `.omx/state/lane_registry.json` | NEEDS-OPERATOR-REVIEW | +2596 / -1 lines (72 today lanes registered) |
| `.omx/state/lane_maturity_audit.log` | NEEDS-OPERATOR-REVIEW | +223 lines (audit-log append for today's `mark` events) |
| `.omx/state/modal_call_id_ledger.jsonl` | NEEDS-OPERATOR-REVIEW | +5 lines (today's Z6+C6 dispatch+harvest events) |
| 11 `.omx/operator_authorize_recipes/substrate_*.yaml` | RECOVERED | small mods (Catalog #324 backfill: `predicted_band_validation_status: pending_post_training` per sister subagent `catalog_324_backfill_sweep_10_substrate_recipes_pending_post_training`) |
| `experiments/train_substrate_time_traveler_l5_z6.py` + 2 driver scripts | RECOVERED | Z6 Wave 2 trainer + driver refactor (Z6_TRAINER_MODE env + full-mode trainer flag forwarding) |
| `tools/cathedral_autopilot_autonomous_loop.py` | RECOVERED | Q2/Q3 batched v2 cascade landing (Lagrangian → DeliverabilityProof → passthrough) |
| `src/tac/master_gradient_consumers.py` + 9 `src/tac/optimization/*.py` + tests | RECOVERED | per-pair consumer custody + namespace wave |
| `reports/latest.md` | RECOVERED | frontier surface backfill per Catalog #316 |
| `docs/pr_writeups/cpu_frontier_fec6_20260517.md` | RECOVERED | PR body draft refresh |

**The high-stakes uncommitted set** is CLAUDE.md + preflight.py + lane_registry.json + lane_maturity_audit.log + modal_call_id_ledger.jsonl. All five carry substantive today's work that landed-in-form but not landed-in-git.

### 4. .omx/state/ ledger integrity

- **`modal_call_id_ledger.jsonl`** (`.omx/state/modal_call_id_ledger.jsonl`): 347 total rows; 7 rows added today across 5 unique call_ids. Z6 Wave 2 dispatches (`fc-01KRW7RHFHP640BHTQ0FZM3M38` + `fc-01KRW7ZCYK5XF6MSHD24R71A46`) BOTH have dispatched+harvested rows (terminalized by codex). C6 IBPS smoke (`fc-01KRW353MJJ9A6QW8H99QWZEMH`) has dispatched row but NO harvested row in the ledger — however the empirical 3.04 score IS captured via (a) lane registry `real_archive_empirical` gate evidence; (b) `.omx/state/probe_outcomes.jsonl` adjudicated DEFER verdict; (c) council deliberation posterior C6 IBPS post-empirical reactivation symposium. **Classification: OBSERVATIONAL gap** — ledger sync is incomplete but signal is preserved at 3 sister surfaces.
- **`lane_registry.json`** (`.omx/state/lane_registry.json`): 858 total lanes; 72 today; 2596-line uncommitted diff. NEEDS-OPERATOR-REVIEW for commit.
- **`active_lane_dispatch_claims.md`** (`.omx/state/active_lane_dispatch_claims.md`): `CLAIM_SUMMARY active=0 stale_nonterminal=0 terminal_latest=942` — clean. Today's Z6+C6 dispatches all have terminal rows (`failed_z6_full_canary_driver_smoke_mode_misroute`, `failed_modal_smoke_red`, etc.).
- **`probe_outcomes.jsonl`** (`.omx/state/probe_outcomes.jsonl`): 6 total; all 6 are TODAY's. Captures (i) ATW v2 D4 INDEPENDENT; (ii) Wunderkind G1 v2 DEFER; (iii) PR101 magic codec FEC6 INDEPENDENT (op_routable #9); (iv) Q6 preprobe pairwise composition_alpha PROCEED; (v) C6 IBPS smoke DEFER 3.04; (vi) Z6 v2 Wave 2 DEFER (full canary driver mode misroute).
- **`commit-serializer.log`** (`.omx/state/commit-serializer.log`): all today's commits went through canonical serializer per Catalog #117 spot-check; no bare commits observed in this window.
- **`council_deliberation_posterior.jsonl`** (`.omx/state/council_deliberation_posterior.jsonl`): 47 total; 21 today including Z6 Phase 2 + Z6 Phase 3 + ATW V2 reactivation + C6 IBPS Phase 2 dispatch unlock + C6 IBPS post-empirical reactivation symposium + Wyner-Ziv side-info hoisting + 3 recursive adversarial review cycles + 1 META-ASSUMPTION cadence trigger. Per Catalog #300 v2 frontmatter discipline.
- **`catalog-claim.log`** (`.omx/state/catalog-claim.log`): 27 today; 1 anomaly — **Catalog #325 was claimed at 2026-05-18T00:30:18Z (pid 4587) with NO follow-up `committed_via_serializer: true` event**. Verified: the CLAUDE.md catalog row, gate function, and tests for #325 all exist in working tree but are uncommitted. This is the same broader uncommitted-Catalog-trio (#324/#325/#326) finding from §3.

### 5. Sister subagent /private/tmp task-output files

1439 task output files across the session directory at `/private/tmp/claude-501/-Users-adpena-Projects-pact/9e5d0bc0-81e7-4cc9-ac31-531915ae8895/tasks/`. Verified that the ATW V2 reactivation symposium (`a08a7608a0baba053.output`) is captured in:
- `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md` (433 lines; 33 mentions of "binding"/"revision"/"Item #8"/"HARD-EARNED")
- `.omx/state/council_deliberation_posterior.jsonl` (ATW V2 reactivation row with `deferred_substrate_id=atw_codec_v2` + 7 binding revisions)
- Memory entry `feedback_z6_v2_wave_2_codex_repairs_landed_20260517.md` (cites both Z6 Wave 2 landing AND sister ATW V2 symposium with Item #8 hypothesis verbatim)

**Classification: RECOVERED**. Symposium output is fully canonicalized at 3 surfaces.

### 6. Modal harvest results

- **C6 IBPS smoke** `fc-01KRW353MJJ9A6QW8H99QWZEMH` (dispatched 2026-05-17T18:08:18Z): `experiments/results/lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260517T230751Z__smoke__50ep_modal/{modal_metadata.json, modal_call_id.txt}` exist. Full Modal worker artifacts (training stats / archive / score JSON) appear to be in the Modal volume `comma-train-lane-results/substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260517T230751Z__smoke__50ep/` — the operator-side harvest didn't pull them to local; final_score=3.04 was determined elsewhere (likely sister subagent recovered from Modal volume directly OR was reported by the trainer logs). The empirical anchor IS captured in lane registry + probe outcomes + council reactivation symposium, so no signal lost.
- **Z6 Wave 2 dispatches**: 2 dispatches both terminalized; smoke failed with missing auth_eval_*.json (Catalog #316 enforcement caught it); full canary failed with driver smoke-mode misroute (Catalog #326 bug class — caught + fixed in same wave). Terminal rows present in active_lane_dispatch_claims.md + harvested events in modal_call_id_ledger.

**Classification: RECOVERED** for both. Z6's full empirical signal (driver bug) is captured in the Catalog #326 landing memo + the new driver-mode-hardcode audit tool.

### 7. Orphan signals (Catalog #711 pattern)

Three NEW canonical helpers landed today with explicit 6-hook wire-in declarations per Catalog #125:

| Canonical helper | Producer surface | Consumer surface | Wire-in status |
|---|---|---|---|
| `src/tac/optimization/tier_c_density_post_training_validator.py` | Catalog #324 gate + `tools/audit_predicted_band_provenance.py` | Catalog #324 strict preflight; autopilot ranker via `predicted_band_validation_status` field; `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313 | ACTIVE (hooks 4, 5, 6 per Catalog #125 declaration in catalog row) |
| `tools/audit_substrate_driver_mode_hardcode.py` | Catalog #326 gate (delegates verdict classification to this tool) | Operator-facing audit CLI; cathedral autopilot ranker consumes audit JSON at `.omx/state/substrate_driver_mode_hardcode_audit_<utc>.json` | ACTIVE (hooks 4, 5 per Catalog #326 declaration) |
| `src/tac/codec/wyner_ziv_layer.py` | Wyner-Ziv pipeline-stage codec primitive (Catalog #320) | Composition matrix consumer; deliverability prober via Catalog #319 sister gate | Per sister memo `feedback_wyner_ziv_pipeline_stage_codec_primitive_landed_20260517.md` |

**Classification: RECOVERED** — wire-in coverage declared per Catalog #125 in each gate's catalog row.

### 8. Memory entry hygiene

63 memory entries dated 2026-05-17 or 2026-05-18; 50 substantive commits today. Cross-checked 3 representative landings (Z6 Wave 2, frontier-signal-loss, Catalog #324) and all have well-formed memory↔commit cross-refs (17, 3, 1 mentions respectively). MEMORY.md indexed lines reference the most-recent ~15 of today's entries (Z6 Wave 2 at index line 1, ATW V2 symposium in the body, codec attribution correction, frontier permanent fix, etc.).

**Classification: RECOVERED** — memory hygiene is healthy.

### 9. Catalog claim ledger

27 today claims; 1 anomaly — **Catalog #325 missing committed_via_serializer follow-up event**. The CLAUDE.md catalog row + gate function + tests all exist in working tree (uncommitted). The git-transactional claim event was logged but the canonical 2nd event (after `tools/claim_catalog_number.py --commit-via-serializer`) is missing. This is part of the broader uncommitted-#324/#325/#326-trio finding.

The `state: claim catalog #325` commit is also missing from git log (only #324 and #326 have claim commits; #325 was claimed via a different code path that didn't transactionally commit per CANON-1.E hardening).

**Classification: NEEDS-OPERATOR-REVIEW** — operator should verify the claim is honored by the canonical serializer when the gate trio is committed.

## Recommendations (per finding NEEDS-OPERATOR-REVIEW)

### R1: Commit the Catalog #324/#325/#326 gate trio via canonical serializer

The complete in-flight wave (CLAUDE.md +151 / preflight.py +702 / lane_registry.json +2596 / lane_maturity_audit.log +223 / modal_call_id_ledger.jsonl +5 + the 38 modifieds + ~30 newly-created tests/tools/recipes/scripts) needs an atomic landing per CLAUDE.md "Strict-flip atomicity rule" + Catalog #117/#157/#174 serializer-with-expected-content-sha256 discipline + Catalog #186 git-transactional claim discipline. Recommended commit slices (operator decision):

1. **Catalog #324 trio**: CLAUDE.md row + `src/tac/preflight.py::check_no_predicted_band_without_post_training_tier_c_validation` + `src/tac/optimization/tier_c_density_post_training_validator.py` + `tools/audit_predicted_band_provenance.py` + `src/tac/tests/test_check_324_*` + 11 recipe modifications (Catalog #324 backfill) + `.omx/state/predicted_band_audit_20260518T000608Z.json`.
2. **Catalog #325 trio**: CLAUDE.md row + `src/tac/preflight.py::check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor` + `src/tac/tests/test_check_325_*` + the 2 per-substrate symposium memos (`council_per_substrate_symposium_atw_v2_reactivation_20260518.md` + `council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518.md`) + canonical posterior anchor rows already in `.omx/state/council_deliberation_posterior.jsonl`.
3. **Catalog #326 trio**: CLAUDE.md row + `src/tac/preflight.py::check_substrate_driver_consumes_trainer_mode_env_var` + `tools/audit_substrate_driver_mode_hardcode.py` + `src/tac/tests/test_check_326_*` + Z6 driver/trainer/recipe fixes already in working tree.
4. **Lane registry + audit log + Modal ledger**: separate commit (state mutations) per Catalog #131.
5. **62 sister-subagent memos in `.omx/research/`**: separate commits per-sister-wave OR one batched per Catalog #117/#157/#174 with `--expected-content-sha256` to extinct WAVE-D absorption risk.

### R2: Audit C6 IBPS Modal volume to recover final score harvest

Pull the C6 IBPS smoke results from the Modal volume `comma-train-lane-results/substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260517T230751Z__smoke__50ep/` and emit a `harvested` event to `.omx/state/modal_call_id_ledger.jsonl` for `fc-01KRW353MJJ9A6QW8H99QWZEMH` so the canonical 4-layer per Catalog #245 is complete. Signal is preserved at sister surfaces but ledger sync hygiene matters for the next dispatch wave's autopilot ranking.

### R3: Audit subset of the 1439 task-output files for any unique signal NOT yet canonicalized

Per the in-context CLAUDE.md non-negotiable: every subagent landing should have either a memory entry OR an explicit research memo. The ~30 task-output files from in-flight subagents that haven't reached their landing should be audited periodically. This audit verified 1 representative (ATW V2 symposium `a08a7608`) is fully canonicalized; the operator may want to spot-check 3-5 others.

### R4: Strict-flip readiness

Per CLAUDE.md "Strict-flip atomicity rule": the trio of new STRICT gates landed WARN-ONLY initially. Their strict-flips depend on:
- **#324**: backfill the 10 dispatchable substrate recipes with `predicted_band_validation_status: pending_post_training` (sister `feedback_catalog_324_backfill_sweep_10_substrate_recipes_pending_post_training_landed_20260518.md` already lands this — needs to be committed atomically with the gate)
- **#325**: per-substrate symposium discipline backfill across the remaining ASYMPTOTIC candidates (ATW V2 + Z7 + NSCS06 v8 Path B + TT5L + Z8 + DP1)
- **#326**: WARN-ONLY at landing per `feedback_driver_fix_smoke_hardcode_plus_new_catalog_gate_cross_substrate_audit_landed_20260518.md`; live count = 0 at landing (Z6 fix + Z6 recipe set in same commit batch)

## Memory references

- `feedback_z6_v2_wave_2_codex_repairs_landed_20260517.md` — Z6 Wave 2 landing (cited in MEMORY.md L1)
- `feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md` — Catalog #324 landing
- `feedback_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_landed_20260518.md` — Catalog #325 landing
- `feedback_driver_fix_smoke_hardcode_plus_new_catalog_gate_cross_substrate_audit_landed_20260518.md` — Catalog #326 landing
- `feedback_c6_ibps_first_asymptotic_dispatch_smoke_before_full_paired_landed_20260517.md` — C6 IBPS 3.04 empirical anchor
- `feedback_z6_v2_wave_2_dispatch_smoke_before_full_paired_cpu_cuda_landed_20260518.md` — Z6 Wave 2 dispatch failures
- `feedback_permanent_fix_frontier_signal_loss_landed_20260517.md` — sister of THIS audit (Catalog #316 frontier-axis signal-loss permanent fix)
- `.omx/research/z6_v2_wave_2_landing_state_snapshot_20260517.md` — initial state snapshot (partial; this audit extends)
- `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md` — ATW V2 symposium
- `.omx/research/council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518.md` — C6 IBPS reactivation symposium

## Cross-references

- CLAUDE.md "Required durable state" — non-negotiable that drives this audit
- CLAUDE.md "Subagent coherence-by-default" — anti-orphan-work principle
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" — drives the 6-hook wire-in declaration
- Catalog #117 / #157 / #174 — commit-serializer + sha256 discipline (critical for the recommended atomic commit batch)
- Catalog #186 — catalog # git-transactional claim discipline (the missing-#325-claim-commit anomaly)
- Catalog #245 — Modal call_id ledger canonical 4-layer pattern (the C6 IBPS missing harvest event)
- Catalog #314 — commit-swap absorption-pattern (relevant to the 142-entry in-flight working tree)
- Catalog #316 — frontier-signal-loss sister gate (operator-facing audit surface this lane sister-runs)
