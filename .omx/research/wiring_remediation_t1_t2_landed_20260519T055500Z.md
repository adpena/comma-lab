---
schema: wiring_remediation_landing
lane_id: lane_wiring_remediation_t1_plus_t2_20260519
landing_utc: "2026-05-19T05:55:00Z"
parent_audit_commit: 3821cfb6b
parent_audit_memo: .omx/research/wiring_and_integration_audit_pass_20260519T052433Z.md
battle_plan_commit: 6a1e94a63
battle_plan_memo: .omx/research/integrated_battle_plan_priority_queue_dag_cables_gates_20260519T052801Z.md
sister_subagents_at_landing:
  - mps_phase_b_fire_and_harvest (in flight; disjoint scope)
  - codex (autonomous on Cluster B+C+E.1+F)
  - cable_b1_e7_e8_dispatch (concurrent recipe + dispatch sister)
  - cable_c_substrate_symposium_drafts (concurrent symposium DRAFT sister)
  - cable_d_master_gradient_extension (concurrent master-gradient sister)
---

# Wiring remediation T1 + T2 landed — 2026-05-19T05:55Z

## T1 — quick mechanical fixes

### T1.1 Z7-LSTM recipe + trainer backfill

- `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_lstm_predictive_coding_modal_t4_dispatch.yaml`:
  - added `video_input_strategy: per_dispatch_local_copy` per Catalog #171
  - added `canary_status: post_canary_dependent` + `canary_dependency: time_traveler_l5_z7_mamba2` per Catalog #173
  - added `pyav_decode_strategy: cpu_thread_async_upload` per Catalog #181
- `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml`:
  - added `video_input_strategy: per_dispatch_local_copy` per Catalog #171
  - (sister subagent owns rest of file; coordinated via concurrent edit + my field preserved)
- `experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py`:
  - added file-level `# AUTOCAST_FP16_WAIVED:research-only-prebuild-scaffold-not-on-paid-dispatch-path-recipe-research_only-true-dispatch_enabled-false-canonical-autocast-backport-pending-Z6-4c-paired-exact-eval-and-Wave-N-plus-1-council` per Catalog #172

### T1.2 council memos v2 frontmatter backfill (Catalog #300)

- `.omx/research/codex_findings_t3_grand_council_synthesis_20260518T145900Z_codex.md`: added complete v2 frontmatter (council_tier T3, 7 attendees, PROCEED_WITH_REVISIONS verdict, Cicero dissent, Assumption-Adversary verdict, mission_contribution frontier_protecting, override_invoked false, decisions_recorded)
- `.omx/research/council_z8_hierarchical_predictive_coding_per_substrate_symposium_20260518.md`: added `council_predicted_mission_contribution: frontier_breaking` + `council_override_invoked: false`
- `.omx/research/grand_council_findings_deliberation_wave_aggregate_dispatch_plan_20260518.md`: added complete v2 frontmatter (council_tier T3, 15 attendees, PROCEED_WITH_REVISIONS verdict, Contrarian dissent on T3 OVER_CADENCE, Assumption-Adversary HARD-EARNED verdict, mission_contribution rigor_overhead, decisions_recorded summarizing T1/T2/T3 distribution)
- `.omx/research/grand_council_symposium_inflate_py_extreme_compression_20260518.md`: added `council_predicted_mission_contribution: apparatus_maintenance` + `council_override_invoked: false`

### T1.3 substrate symposium (deferred to sister per audit)

Per audit memo TIER-1 #3 explicit note: *"operator-routed wave + sister symposium wave already in flight per task queue"*. Sister subagent #4 (CABLE C substrate symposium DRAFTs) owns this surface. Did NOT touch `rudin_floor_interpretable_ml` or `z6_v2_candidate_1_multi_layer_film` recipes — they are operator-set to `dispatch_enabled: true` and modifying without per-substrate symposium evidence is sister-owned per Catalog #325.

**Catalog #325 live count snapshot at landing**: 4 (rudin_floor + stack_of_stacks + vq_vae_k_sweep + z6_v2_candidate_1). The audit memo named 2; live audit shows 4. Sister subagent will address.

## T2 — cathedral autopilot consumes 12 new tac.* packages

### Paradigm: convention-over-configuration auto-discovery

Per the prior CATHEDRAL-AUTO-INGEST-PARADIGM-SHIFT landing 2026-05-19 (commit a334f0e60 / earlier) + Catalog #335: `src/tac/cathedral_consumers/` is the canonical auto-discovery directory. The `tools/cathedral_autopilot_autonomous_loop.discover_and_register_consumers` function iterates every subdirectory at loop start, validates each via `tac.cathedral.consumer_contract.CathedralConsumerContract`, and registers compliant consumers WITHOUT manual ranker-cascade edits.

**The fix is structural**: I added 12 NEW consumer packages, one per orphan namespace. The auto-discovery loop will now ingest all 12 at next iteration. No ranker-cascade edits required.

### 12 new consumer packages landed

| Consumer | Source namespace | Hook contribution | Axis tag |
|---|---|---|---|
| `atom_consumer` | `tac.atom` | DISPATCH + CONTINUAL_LEARNING | `[predicted]` |
| `formula_extinctions_consumer` | `tac.formula_extinctions` | DISPATCH | `[predicted]` |
| `experimental_extinctions_consumer` | `tac.experimental_extinctions` | DISPATCH + CONTINUAL_LEARNING | `[predicted]` |
| `contest_oracle_consumer` | `tac.contest_oracle` | DISPATCH | `[predicted]` |
| `utility_curves_consumer` | `tac.utility_curves` | DISPATCH + SENSITIVITY_MAP | `[predicted]` |
| `solvers_consumer` | `tac.solvers` | DISPATCH | `[predicted]` |
| `unified_action_consumer` | `tac.unified_action` | DISPATCH + PARETO_CONSTRAINT | `[predicted]` |
| `procedural_codebook_generator_consumer` | `tac.procedural_codebook_generator` | DISPATCH + PROBE_DISAMBIGUATOR | `[predicted]` |
| `mps_diagnostic_consumer` | `tac.mps_diagnostic` | DISPATCH | `[MPS-PROXY]` |
| `mps_gap_experiment_consumer` | `tac.mps_gap_experiment` | DISPATCH | `[MPS-PROXY]` |
| `contest_exploits_consumer` | `tac.contest_exploits` | DISPATCH + BIT_ALLOCATOR | `[predicted]` |
| `analytical_solve_extinctions_consumer` | `tac.analytical_solve_extinctions` | DISPATCH | `[predicted]` |

All 12 consumers are **observability-only**: zero `predicted_delta_adjustment`, `promotable=False`, with `axis_tag` per CLAUDE.md "Apples-to-apples evidence discipline" (NEVER promotable to contest axis without paired-axis verification per Catalog #127 + #323). Per-candidate score adjustment requires explicit per-helper integration (e.g. the existing `master_gradient_consumers` sister-#817 sidecar cascade in `apply_z1_empirical_revision_to_candidate_delta`).

### Why observability-only and not delta-adjustment?

Per the wiring audit's TIER 2 #4 design + per CLAUDE.md "Apples-to-apples evidence discipline": these 12 namespaces emit CANONICAL HELPER AVAILABILITY signals, not per-candidate score predictions. The right semantic is "this canonical helper is in the toolbox" — promotion to a delta adjustment requires (a) per-archive empirical sidecar (e.g. master-gradient ledger row) OR (b) per-candidate explicit integration with a feasibility constraint (e.g. unified_action.Action construction with full DualVariables). Both paths are sister-owned by the existing ranker cascades; #322/#323 phantom-provenance + Catalog #287 docstring-overstatement gates would refuse any false-authority delta from these consumers.

### Auto-discovery verified

```
13 consumer packages discovered (12 new + _example_consumer reference)
ALL 13 contract-COMPLIANT
0 waivers required
```

### Tests pinned

`src/tac/tests/test_cathedral_autopilot_consumes_new_tac_namespaces.py` — 76 tests:
- 12 × per-consumer importability
- 12 × per-consumer contract compliance
- 12 × per-consumer canonical row contract (predicted_delta + rationale + axis_tag)
- 12 × per-consumer zero-adjustment regression guard
- 12 × per-consumer non-promotable regression guard
- 12 × per-consumer update_from_anchor callable
- 1 × all 12 auto-discovered
- 1 × all 12 contract-compliant in discovery loop
- 1 × audit-named-namespaces-match regression guard
- 1 × cathedral autopilot module imports with new consumers present

All 76 PASS. Sister Catalog #335 (18 tests) STILL pass.

## Catalog drift verification post-landing

| Gate | Pre-landing | Post-landing |
|---|---|---|
| Catalog #185 LIVE_COUNT drift | 5 violations | **0** |
| Catalog #300 council v2 frontmatter | 4 violations | **0** |
| Catalog #171 video_input_strategy | 2 violations | **0** |
| Catalog #172 autocast_fp16 | 1 violation | **0** |
| Catalog #173 canary_status | 1 violation | **0** |
| Catalog #181 pyav_decode_strategy | 1 violation | **0** |
| Catalog #335 cathedral consumer directory | 0 violations | **0** (preserved) |
| Catalog #325 substrate symposium | 4 violations | **4** (sister-owned; not in scope) |

## Audit's 7-item TIER queue updated status

| Item | Audit tier | Status |
|---|---|---|
| 1. Z7-LSTM recipe + trainer backfill | TIER 1 | **RESOLVED** (this subagent) |
| 2. 4 council memos v2 frontmatter | TIER 1 | **RESOLVED** (this subagent) |
| 3. 2 substrate symposium backfill | TIER 1 | sister-owned (subagent #4 / operator wave) |
| 4. Cathedral autopilot consumes new tac.* packages | TIER 2 | **RESOLVED** (this subagent — auto-discovery paradigm) |
| 5. Catalog #125 19/26 continual_learning hook gap | TIER 2 | UNCHANGED (warn-only baseline acceptable) |
| 6. Catalog #287 phantom-API/empirical-claim (224) | TIER 3 | UNCHANGED (Wave 2 follow-on bounded) |
| 7. Catalog #272 distinguishing-feature (1) | TIER 3 | UNCHANGED (pre-existing prober lane) |

**3 of 7 items RESOLVED this subagent; 1 deferred to sister; 3 unchanged per audit guidance (acceptable baselines).**

## 6-hook wire-in declaration (this landing memo IS the canonical declaration)

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable mandatory wire-in:

1. **Sensitivity-map**: ACTIVE via `utility_curves_consumer` declaring HookNumber.SENSITIVITY_MAP
2. **Pareto constraint**: ACTIVE via `unified_action_consumer` declaring HookNumber.PARETO_CONSTRAINT
3. **Bit-allocator**: ACTIVE via `contest_exploits_consumer` declaring HookNumber.BIT_ALLOCATOR
4. **Cathedral autopilot dispatch**: ACTIVE via ALL 12 consumers declaring HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH (this IS the primary contribution)
5. **Continual-learning posterior**: ACTIVE via `atom_consumer` + `experimental_extinctions_consumer` declaring HookNumber.CONTINUAL_LEARNING_POSTERIOR; memory entry registered per Catalog #131 fcntl-locked discipline
6. **Probe-disambiguator**: ACTIVE via `procedural_codebook_generator_consumer` declaring HookNumber.PROBE_DISAMBIGUATOR

## Mission alignment

Per CLAUDE.md "Mission alignment — non-negotiable": this landing is **frontier_protecting** (extincts orphan-signal accumulation at the cathedral autopilot consumer surface — the wiring audit's TOP-priority finding). Operator-attention budget consumed: 1 subagent slot, ~$0 GPU, no race-mode rigor inversion. Aligns with the standing directive that discipline serves the mission AND apparatus exposes its own behavior maximally per Catalog #305 observability surface.

## Cross-references

- Parent audit: `.omx/research/wiring_and_integration_audit_pass_20260519T052433Z.md` (commit 3821cfb6b)
- Battle plan: `.omx/research/integrated_battle_plan_priority_queue_dag_cables_gates_20260519T052801Z.md` (commit 6a1e94a63)
- Cathedral consumer paradigm: `src/tac/cathedral_consumers/README.md`
- Canonical contract: `src/tac/cathedral/consumer_contract.py`
- Auto-discovery loop: `tools/cathedral_autopilot_autonomous_loop.discover_and_register_consumers` (line 5937)
- Catalog #335 STRICT preflight gate (cathedral consumer directory contract)
- Catalog #125 6-hook wire-in non-negotiable
- Catalog #287 phantom-API / empirical-claim-without-tag discipline
- Catalog #323 canonical Provenance discipline (umbrella over the [predicted]/[MPS-PROXY] axis-tag chain)

## Lane

`lane_wiring_remediation_t1_plus_t2_20260519` L1 (impl_complete + memory_entry).
