---
audit_kind: wiring_and_integration_audit_pass
audit_window_utc_start: "2026-05-18T00:00:00Z"
audit_window_utc_end: "2026-05-19T05:22:00Z"
prior_wiring_pass: aa16f7f8c (task #913)
substantive_landings_audited: ~24
lane_id: lane_wiring_and_integration_audit_pass_20260519
sister_subagents_at_landing:
  - mps_phase_b_fire_and_harvest_20260519 (in flight, disjoint scope)
  - hardening_license_plus_memory_hygiene_20260519 (completed 05:23Z)
authority: operator NON-NEGOTIABLE "Perhaps one should be used for a wiring and integration pass"
---

# Wiring + Integration Audit Pass — 2026-05-19T05:24Z

## Scope

50 commits in `git log --since="2026-05-18 00:00"` window (~24 substantive landings after filtering codex docs/typo-only/file-state commits). 47 landing memos in audit window. 5 META-meta drift violations + 26 in-window Catalog #125 6-hook wire-in gaps + 224 total Catalog #287 phantom-API/empirical-claim-without-tag violations + 2 Catalog #325 per-substrate symposium gaps. Per CLAUDE.md "Subagent coherence-by-default" + "Max observability".

## Phase 1 — Per-landing 6-hook compliance

Sample of 10 most-recent in-window landing memos shows EVERY one declares at least 2 hooks; landings with high orphan-signal risk (deep_research_wave, asymptotic_stacking_audit, cross_stack_synthesis, systematic_reclaimability_re_examination) declare ZERO hooks. The Catalog #125 strict-gate run returns **26 in-window violations** (out of 57 total post-2026-05-09). The most common gaps:

| Hook | Missing-in count (in-window 26) | Comment |
|---|---|---|
| continual_learning | 19 / 26 | Single most-skipped hook. Most landings emit a posterior anchor implicitly via canonical helpers (e.g. atom emission auto-records to `tac.atom`'s ledger) but never DECLARE this hook explicitly in body. |
| probe_disambiguator | 11 / 26 | Audit + sweep landings rarely include — but they SHOULD (the sweep IS the disambiguator). |
| bit_allocator | 7 / 26 | Genuinely N/A for most analysis-only landings (correct to omit when not changing per-tensor importance). |
| sensitivity_map | 5 / 26 | Mostly N/A correctly. |
| pareto | 5 / 26 | Mostly N/A correctly. |
| cathedral_autopilot | 4 / 26 | Mostly N/A correctly. |

**Empirical signal**: the canonical "5+ hooks declared" landings (cathedral_autopilot_realistic_ev_update, gate_empirical_anchor_audit, mps_local_compute_frontier_diagnostic, z7_mamba_2_stability) are exactly the landings that move signal into solver consumers. Landings without explicit hook declaration AND without atom emission silently orphan their signal.

## Phase 2 — Orphan-signal + missing-wire-in inventory

### Orphan producers (signal emitted but no canonical consumer)

12 new `src/tac/*/__init__.py` packages landed this window. **Canonical consumers (cathedral_autopilot, continual_learning, sensitivity_map, probe_outcomes_ledger) do NOT import any of them.** Sampled:

| New package | Consumer imports |
|---|---|
| `tac.atom` | 10 in-tree analytical_solve_extinctions + 1 experimental; cathedral_autopilot=0; continual_learning=0 |
| `tac.formula_extinctions` | 1 tools/ empirical validator only |
| `tac.experimental_extinctions` | 1 tools/ empirical validator only |
| `tac.contest_oracle` | 0 (self-loop in score_predictor) |
| `tac.utility_curves` | 1 test only |
| `tac.solvers` | 1 tool + 2 tests |
| `tac.unified_action` | 1 test + 1 production (contest_oracle.score_predictor) |
| `tac.procedural_codebook_generator` | unknown (not sampled) |
| `tac.mps_diagnostic` | unknown |
| `tac.mps_gap_experiment` | sister-owned, in flight |
| `tac.contest_exploits` | unknown |
| `tac.analytical_solve_extinctions` | self-loop via tac.atom emissions |

**Verdict**: most new packages emit canonical signals (atoms, posterior rows) but the canonical solver consumers (`tools/cathedral_autopilot_autonomous_loop.py`) have NOT been extended to load and consume them. This is the classic "lateral-leap landing produces research artifacts the planner cannot see" failure mode per CLAUDE.md "Subagent coherence-by-default" anti-fragmentation primitive.

### Missing consumer wire-ins

- Catalog #319 expects `DeliverabilityProof` for HIGH_PAIR_INVARIANT — coverage not measured this audit (sister Q1+Q2+Q3 landed).
- Catalog #324 predicted_band_validation_status — **0 violations** (Catalog #324 strict-clean).
- Catalog #325 per-substrate symposium — **2 violations**: `rudin_floor_interpretable_ml` + `z6_v2_candidate_1_multi_layer_film` recipes dispatch_enabled=true with NO 14-day symposium anchor.
- Catalog #240 recipe-vs-trainer-state — **0 violations**.
- Catalog #298 stale L1 substrates — **0 violations**.

## Phase 3 — META-meta gate drift sweep

| Gate | Status | Drift cases |
|---|---|---|
| #185 LIVE_COUNT drift | **5 violations** | #171/#172/#173/#181 all Z7-LSTM substrate-recipe missing canonical fields; #300 = 4 council memos missing v2 frontmatter |
| #176 callsite-row parity | 0 | clean |
| #159 text-strict parity | 0 | clean |
| #118 no duplicate numbers | 0 | clean |

### #185 drift attribution

All 4 of #171/#172/#173/#181 fire on the new Z7-LSTM substrate (`experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py` + recipe). Z7-Mamba2 is clean (sister memo + recipe + trainer; commit `c85b4f2cc` + `e16c4ac2d` + `7fbfb20a0` landed all surfaces). Z7-LSTM is the paired-MPS-proxy comparison variant from commit `c88ac969a` — recipe shipped without `video_input_strategy` / `canary_status` / `pyav_decode_strategy` fields and trainer without `--enable-autocast-fp16` flag.

### #300 drift attribution

4 council memos dated 20260518 missing v2 frontmatter fields (`council_attendees` / `council_dissent` / `council_override_invoked`):
- `codex_findings_t3_grand_council_synthesis_20260518T145900Z_codex.md`
- `council_z8_hierarchical_predictive_coding_per_substrate_symposium_20260518.md`
- `grand_council_findings_deliberation_wave_aggregate_dispatch_plan_20260518.md`
- `grand_council_symposium_inflate_py_extreme_compression_20260518.md`

## Phase 4 — Synthesis: top remediation candidates

Per CLAUDE.md "Forbidden premature KILL": no landing rolled back. All remediations ADDITIVE.

### TIER 1 — HIGH EV, BOUNDED SCOPE (operator-routable, future subagent slot)

1. **Z7-LSTM recipe + trainer backfill** (extincts 4 of 5 #185 drift cases). Add `video_input_strategy: per_dispatch_local_copy`, `canary_status: post_canary_dependent`, `canary_dependency: time_traveler_l5_z7_mamba2`, `pyav_decode_strategy: cpu_thread_async_upload` to the Z7-LSTM recipe; add `--enable-autocast-fp16` flag to Z7-LSTM trainer argparse (or file-level `# AUTOCAST_FP16_WAIVED:<rationale>`). Cost: ~10 min editor. Bug class: a future Z7-LSTM Modal dispatch silently uses default T4 routing without canary-first ordering per Catalog #173.

2. **4 council memo v2 frontmatter backfill** (extincts #300 4 cases). Add `council_attendees` + `council_dissent` + `council_override_invoked` fields to the 4 cited memos per Catalog #110/#113 APPEND-ONLY discipline (HISTORICAL_PROVENANCE — body unchanged; frontmatter extended only). Cost: ~15 min editor.

3. **2 substrate symposium backfill** (extincts #325 2 cases). Either (a) flip `rudin_floor_interpretable_ml` + `z6_v2_candidate_1_multi_layer_film` recipes to `dispatch_enabled: false` until per-substrate symposium memos land, OR (b) the operator-routed symposium wave dispatches the 2 symposium subagents per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM" non-negotiable. The two recipes pre-existed this audit window; sister symposium wave already in flight per task queue.

### TIER 2 — MEDIUM EV (track but don't fix this turn)

4. **Cathedral autopilot extension to consume new packages.** The 12 new `tac.*/__init__.py` packages emit canonical signals (atoms / posterior rows / probe outcomes / advisory rows) but `tools/cathedral_autopilot_autonomous_loop.py` does NOT import any of them. The canonical fix is to add `load_atoms` / `load_advisory_rows` / `load_extinction_rows` consumers and incorporate them into rank-time signal mixing. Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE": *"every stackable or substitutive idea should move toward a typed row consumed by the planner; if a research artifact can affect score but is not visible to the selector, it is orphaned work."* This is the canonical orphan-signal anti-pattern at scale (12 packages this window alone). Cost: ~3-5h for canonical consumer wire-ins. EV: prevents the orphan-signal accumulation that drives the 0.196-0.199 plateau per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode retrospective.

5. **Catalog #125 in-window 19/26 `continual_learning` hook gap.** Most landings emit posterior signals via canonical helpers but skip the explicit hook declaration. Two paths: (a) backfill the 26 violating memos with explicit `Hook 5: continual-learning posterior = ACTIVE via <canonical helper> (or N/A — <rationale>)` block per CLAUDE.md "Subagent coherence-by-default" Mandatory wire-in. (b) Treat as warn-only and let it drift. Path (a) is the discipline path; path (b) is the de-facto current state. Cost: ~1-2h editor for backfill OR ~$0 if accepting drift.

### TIER 3 — LOW EV (declared / known)

6. **Catalog #287 phantom-API / empirical-claim-without-tag (224 violations)**. Sister `phantom_api_backfill_wave_1` landing 2026-05-18 (commit `dc9ecfdaa`) already drove sub-scope B from 418 to 194 (53.6% reduction). The 224 current total is bounded by Wave 2 follow-on (`Wave 2 candidates pre-computed in manifest`). Acceptable warn-only baseline pending Wave 2 dispatch.

7. **Catalog #272 distinguishing-feature contract (1 violation)**. Lane `lane_pre_entropy_substrate_pivot_prober_20260517` at L2 missing 4/4 contract fields. Pre-existing this audit window. Backfill per opt-out cascade (likely `research_only=true` since the lane is a prober not a substrate).

## Phase 5 — Bounded fixes applied this subagent

**None** beyond this synthesis memo + memory entry. The 3 TIER-1 fixes touch sister-owned surfaces (Z7-LSTM = z7_mamba_2_path_forward subagent territory; council memos = grand_council_findings_deliberation_wave subagent territory; substrate symposium = operator-routed wave). Per Catalog #314 absorption-pattern avoidance, queueing them as operator-routable rather than applying inline.

## 6-hook wire-in (this audit memo IS the canonical declaration)

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable mandatory wire-in:

1. **Sensitivity-map contribution**: N/A — audit is meta-discipline, no per-tensor score signal contributed.
2. **Pareto constraint**: N/A — audit does not add a feasibility constraint.
3. **Bit-allocator hook**: N/A — no per-tensor importance change.
4. **Cathedral autopilot dispatch hook**: ACTIVE via documenting orphan-signal anti-pattern (TIER 2 remediation #4) so the next autopilot extension wave has the canonical consumer-wire-in list machine-readable in this memo.
5. **Continual-learning posterior**: ACTIVE via memory entry registration per Catalog #131 fcntl-locked discipline; this audit's verdicts feed future per-discipline audit cadence.
6. **Probe-disambiguator**: N/A — audit findings are deterministic (gate function return values are the canonical disambiguator).

## Mission alignment

Per CLAUDE.md "Mission alignment — non-negotiable": this audit is **frontier_protecting** (prevents apparatus-discipline drift that would erode signal quality over time). Operator-attention budget consumed: 1 subagent slot, ~$0 GPU, no race-mode rigor inversion (no active leaderboard window). Aligns with the standing directive that discipline serves the mission — the 5 META-meta drift + 26 6-hook gaps + 2 symposium gaps are all *bounded* and *small-fix* class; documenting them in this memo enables targeted future subagent dispatches rather than letting them accumulate undetected.

## Cross-references

- CLAUDE.md "Subagent coherence-by-default" non-negotiable (6-hook wire-in mandate)
- CLAUDE.md "Max observability — non-negotiable" (observability surface declaration)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" (no landings rolled back)
- Catalog #125 / #126 / #176 / #185 / #287 / #300 / #325 (gates exercised)
- Prior wiring pass: commit aa16f7f8c (task #913)
- Sister landing: `feedback_hardening_license_plus_memory_hygiene_landed_20260519.md`
- Sister in-flight: lane `lane_mps_phase_b_fire_and_harvest_20260519`
