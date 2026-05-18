# ORPHAN-SIGNAL-AUDIT 2026-05-17 — postmortem + remaining 10/15 orphan inventory

Date: 2026-05-17
Lane: `lane_producer_to_cathedral_autopilot_wire_in_20260517`
Subagent: producer_wire_in_20260517 (orphan-signal-audit closure wave)
Operator directive (verbatim): *"Ensure all producers wired up and integrated into consumers as appropriate with the cathedral autopilot the ultimate consumer."*

---

## Context

ORPHAN-SIGNAL-AUDIT (task #711) identified 15 ORPHANS — canonical-helper producers whose outputs were NOT consumed by `tools/cathedral_autopilot_autonomous_loop.py` (the ultimate consumer). The cathedral autopilot ALREADY consumes 8 producer surfaces:

1. `tac.frontier_scan` → `_resolve_canonical_frontier_threshold_cpu` (per Catalog #316)
2. `tac.master_gradient` → `rerank_candidates_via_master_gradient` (per symposium §3.6)
3. `tac.optimization.substrate_composition_matrix` → `load_candidates_from_substrate_composition_ranking` (T1-F composition matrix)
4. `tac.optimization.macos_cpu_advisory_signal` → `load_candidates_from_macos_cpu_advisory_manifest` (Catalog #192)
5. `exact_ready_queue` → `load_candidates_from_exact_ready_queue` (operator-authorize ≤$5 mode)
6. `tac.autopilot_rudin_daubechies` (SLIM + Rashomon + Wavelet + Compressive + GOSDT, Catalog #250-#255) → `rerank_candidates_via_rudin_daubechies` + `rerank_candidates_via_compressive_sensing_lattice`
7. `tac.continual_learning` → `load_continual_learning_posterior` + `posterior_query_track_correction` (W/I/A I-1 wire-in 2026-05-12)
8. `tac.cost_band_calibration` → `predict` (W/I/A I-1 wire-in 2026-05-12)

This wave closes **5 of the 15** high-EV orphans. The remaining 10 are listed in §"Remaining 10/15 Orphans" below for next-session triage.

---

## Closed in this wave (5/15)

Each closure is per CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in declaration:

| # | Producer | New consumer function | Hook# |
|---|----------|----------------------|-------|
| 1 | `tac.council_continual_learning` (Catalog #300) | `rerank_candidates_via_council_continual_learning` | 5 (continual-learning) |
| 2 | `tac.probe_outcomes_ledger` (Catalog #313) | `refuse_candidates_via_probe_outcomes` | 3 (bit-allocator) + 6 (probe-disambiguator) |
| 3 | `tac.deploy.modal.call_id_ledger` (Catalog #245) | `update_cost_band_from_modal_call_id_ledger` | 4 (cathedral autopilot dispatch hook) |
| 4 | `tac.substrates.pretrained_driving_prior.composition` | `load_candidates_from_dp1_composition_primitives` | 3 (bit-allocator; new primitive enumeration) |
| 5 | `tac.recursive_adversarial_review` | `refuse_candidates_via_recursive_review_unsealed` | 4 (cathedral autopilot dispatch hook) |

All 5 wire-ins are **fail-CLOSED** per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
- Missing canonical helper (ImportError) → re-raise (NOT silent skip)
- Missing ledger file → return empty result with explicit status / pass-through (NOT silent default-to-approve-all)
- Refused candidates surfaced explicitly via `(kept, refused)` tuple OR blockers/notes append (NOT silently dropped)

Empirical receipts:
- 45 dedicated tests in `src/tac/tests/test_cathedral_autopilot_orphan_signal_wire_in.py`, all green
- 149 existing `test_cathedral_autopilot_autonomous_loop.py` tests still green (zero regression)
- 404 / 405 of `test_cathedral_autopilot*.py` tests green; the 1 failure (`test_autopilot_requires_exact_cuda_for_promotable_evidence` in `test_cathedral_autopilot_proxy_guards.py`) is **pre-existing** (reproduced via `git stash` test, independent of this wave's edits)
- Live-repo smoke (against actual `.omx/state/` ledgers): all 5 wire-ins run cleanly with no exceptions
- Live `tac.deploy.modal.call_id_ledger` integration scans 342 rows; 129 dispatched-only (skipped), 213 missing required cost-band fields (skipped because the Modal call-id ledger as currently populated frequently omits `recipe` / `gpu` / `cost_actual_usd` for backfilled historical rows — future ledger writes will carry richer metadata so the anchor count will grow over time)

---

## Remaining 10/15 Orphans (next-session triage)

These producers exist in `tac.*` / `tools/*` but are NOT yet consumed by `tools/cathedral_autopilot_autonomous_loop.py`. Routing each via a sister wire-in is queued for the next session — operator may sequence as needed.

### Orphan #6: `tac.sensitivity_map.*`
Status: PARTIAL — used by `discover_sensitivity_map_artifacts` (line 1292) but the per-axis weight rows do NOT yet feed the SLIM risk scorer's `predicted_dispatch_risk` field. Closure: wire each sensitivity-map axis weight into a CandidateRow field `sensitivity_map_weight: dict[str, float]` consumed by `rerank_candidates_via_rudin_daubechies` as a side feature.

### Orphan #7: `src/tac/xray/wire_in.py`
Status: ORPHAN — XRay lenses register but NONE are auto-consumed by the autopilot per the file's `register_lens_with_cathedral_autopilot_ranker` stub. Closure per the design memo `feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516`: per-lens activation produces an axis-tagged candidate-row decorator function.

### Orphan #8: `tac.deploy.lightning.active_jobs_state`
Status: ORPHAN — Lightning dispatch outcomes are persisted but NOT consumed by `update_cost_band_from_modal_call_id_ledger`'s sister. Closure: lift the join-by-call-id pattern from this wave into a generic `update_cost_band_from_dispatch_ledger(*, ledger=...)` and call it once per provider.

### Orphan #9: `tac.deploy.azure.active_vms_state`
Status: ORPHAN — sister of #8 for Azure provider.

### Orphan #10: `tac.vastai_tracker`
Status: ORPHAN — sister of #8 for Vast.ai provider.

### Orphan #11: `tac.preflight_rudin_daubechies.compressive_coverage_estimator`
Status: PARTIAL — used by `rerank_candidates_via_compressive_sensing_lattice` (line 3626) BUT the K=8 anchor coverage is NOT auto-loaded from the live posterior at autopilot start. Closure: wire `compressive_coverage_estimator.load_anchors()` into `load_planner_posterior_for_loop`.

### Orphan #12: `tac.optimization.literature_source_scope`
Status: PARTIAL — used by `literature_source_scope_blockers` but the `source_supports` / `paper_claim_scope` / `pact_must_prove` / `decode_complexity_evidence` fields are surfaced as candidate blockers but NOT contributed to the rank via a fidelity-discount factor. Closure: add `adjust_predicted_delta_for_literature_source_scope` in the existing rank_candidates pipeline (sister of `adjust_predicted_delta_for_class_shift`).

### Orphan #13: `tac.optimizer.exact_dispatch_authority`
Status: USED — already consumed by `CandidateRow.dispatch_authority_blockers`. Not an orphan; misclassified in initial audit.

### Orphan #14: `tac.lane_maturity` / `.omx/state/lane_registry.json`
Status: ORPHAN — lane-maturity Level is NOT used in autopilot ranking. Closure: emit candidate-row `lane_maturity_level: int` field and add `adjust_predicted_delta_for_lane_maturity` (Level 3 lanes get a small boost; Level 0/1 sketch lanes get a "pre-research" tag).

### Orphan #15: `tac.substrate_registry.contract` (Catalog #241/#242)
Status: ORPHAN — META layer substrate contracts are NOT consumed by autopilot. Closure: validate each candidate's `family` against `tac.substrate_registry._REGISTERED_SUBSTRATES` and surface unregistered families as a soft-warn block (per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY").

---

## Postmortem: why this wave's pattern works

Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode, each of the 5 wire-ins follows the **same shape** structurally (canonical fail-closed loader + per-candidate adjustment + explicit refused-rows surface) but **adapts the canonical helper signature to the per-producer optimal**. The "share the shape, fork the implementation" pattern is the META-level expression of HNeRV parity L7 (bolt-ons share, substrate engineering unique-ifies).

Specifically:
- DP1 composition → emits NEW candidate rows (LOADER pattern)
- Council ledger → adjusts existing predicted_score_delta (RERANK pattern)
- Probe outcomes → filters out candidates pre-rank (REFUSE pattern)
- Modal call-id ledger → reverse direction: autopilot ingestion → cost-band POSTERIOR write (UPDATE pattern)
- Recursive review SEAL → filters out unsealed bundles pre-rank (REFUSE pattern)

The 5 patterns map to the 4 hook types the autopilot already supports (LOAD, RERANK, REFUSE, UPDATE-DOWNSTREAM). This wave's choice to NOT extend the CandidateRow schema (instead using `blockers` + `notes` + existing `predicted_score_delta` adjustment) keeps the wire-in additive and trivially reversible — per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + Catalog #299 gate-quota discipline (no new schema invariants without retiring something).

---

## Per-CLAUDE.md §"Mission alignment — non-negotiable" classification

`predicted_mission_contribution`: **apparatus_maintenance** (this wave is observability + producer-consumer integration; no direct score-lowering claim, but enables downstream score-lowering via the operator seeing more producer signal at rank time).

`override_invoked`: false (no operator override required; this wave operationalizes a standing directive)

Per the operator standing directive 2026-05-17 mission-alignment binding context: this work is operator-routed apparatus maintenance that closes the producer-orphan gap. It is NOT frontier-breaking by itself, but the closure unblocks the autopilot's ability to consume signal that COULD steer toward frontier-breaking decisions (e.g. when a council deliberation produces PROCEED-unconditional verdict on a class-shift substrate, that signal now reaches the autopilot ranker for the FIRST time).

---

## Cross-references

- Lane registry: `.omx/state/lane_registry.json` row `lane_producer_to_cathedral_autopilot_wire_in_20260517`
- Premise verifier: `.omx/tmp/producer_wire_in_premise_verifier.txt`
- Landing memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_cathedral_autopilot_orphan_signal_wire_in_landed_20260517.md`
- ORPHAN-SIGNAL-AUDIT origin: task #711
- 8 existing producer wire-ins: `tools/cathedral_autopilot_autonomous_loop.py:2515` (`load_candidates_from_exact_ready_queue`), `:2594` (`load_candidates_from_substrate_composition_ranking`), `:3154` (`load_candidates_from_macos_cpu_advisory_manifest`), `:3408` (`rerank_candidates_via_rudin_daubechies`), `:3626` (`rerank_candidates_via_compressive_sensing_lattice`), `:3706` (`rerank_candidates_via_master_gradient`)
- 5 new wire-ins: `tools/cathedral_autopilot_autonomous_loop.py:3831+` (post-`diagnose_compressive_sensing_lattice_undersampling`, pre-`main`)
- 45 new tests: `src/tac/tests/test_cathedral_autopilot_orphan_signal_wire_in.py`
