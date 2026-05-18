# Comprehensive State Tracker — 2026-05-17

**Purpose**: Canonical no-signal-loss artifact per operator standing directive 2026-05-17 verbatim *"Use tasks and .omx design memos as tracking docs for state; we need to ensure no signal loss and all are fully production hardened implemented and wired and integrated and adversarially reviewed recursively"*.

**Refresh cadence**: After every wave completion. Tasks (#790–#805) are the live machine-readable mirror; this memo is the human-readable consolidation per CLAUDE.md "Beauty, simplicity, and developer experience".

**horizon_class**: `frontier_pursuit`  <!-- Catalog #309 canonical format; FEC6 frontier 0.19205 [contest-CPU] → predicted post-Wyner-Ziv [0.147, 0.167] -->

---

## In-flight subagents (3 / 3 slot cap)

| Subagent | Task | Lane | ETA | Output anchor |
|---|---|---|---|---|
| `a2ab518a45026e7a7` | Lagrangian-dual optimal planner (operator spec) | `lane_per_pair_optimal_treatment_plan_via_lagrangian_dual_20260517` | ~3-5h | `src/tac/master_gradient_consumers.py` + tests + landing memo |
| `a7cf98003dfd05f4f` | Wyner-Ziv deliverability prober | `lane_wyner_ziv_deliverability_prober_20260517` | ~45 min | `tools/wyner_ziv_deliverability_prober.py` + `.omx/state/wyner_ziv_deliverability/probe_f174192aeadf_*.json` |
| Q1 (dispatching) | deliverability_proof_builder canonical helper | `lane_deliverability_proof_builder_canonical_20260517` | ~2h | `src/tac/wyner_ziv_deliverability/proof_builder.py` |

Background bg-bash (does NOT count against subagent cap):
- 600-pair fp64 master gradient extraction (nohup-detached); ETA ~30 min remaining

## Completed this session

| ID | Item | Anchor |
|---|---|---|
| `a24279920878bf939` | Grand Council Symposium T3 verdict PROCEED_WITH_REVISIONS | `.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md` + `.omx/research/wyner_ziv_optimal_implementation_queue_20260517.md` |
| `aeba5223db3ae6169` | Wire-in #2 Wyner-Ziv → tac.sensitivity_map | `src/tac/sensitivity_map/wyner_ziv_reweight.py` (~316 LOC, 29 tests) |
| Post-landing audit HIGH-1 | Lane `lane_master_gradient_consumers_module_20260517` registered at L1 | `.omx/state/lane_registry.json` |
| Post-landing audit MEDIUM-1 | horizon_class format fixed | `.omx/research/tac_side_information_namespace_design_20260517.md` |
| Post-landing audit MEDIUM-3 | Test file for new CLI flags | `src/tac/tests/test_session_20260517_cli_flag_additions.py` (14 tests) |

## Queued (pending slot or operator decision)

| Task # | Item | Dependency | Operator decision needed? |
|---|---|---|---|
| #793 | Q2: Catalog #318 STRICT gate | Q1 | No (council pre-approved) |
| #794 | Q3: Autopilot reweight v2 tier-aware | Q1 | Per-tier factors (1.20/1.10/1.05) — operator confirm |
| #795 | Q4: FEC6 Tier-2 Comma2k19 smoke | Q1+Q3 + operator | **YES — $0.70 GPU spend** |
| #796 | Q5: PR101 FEC6 lane registry integration | Q1+Q4 anchor | No |
| #797 | VIZ: master_gradient_xray.py | 600-pair tensor | No (in-context) |
| #798 | 600-pair Wyner-Ziv re-classifier | 600-pair tensor | $0.30 paired Modal CUDA |
| #799 | Consumers 7-14 builder wave | Q1 | No |
| #800 | Wire-in #3: bit_allocator | Lagrangian planner landing | No |
| #801 | Wire-in #4: autopilot risk reweight | Wire-in #2 + Lagrangian planner | No |
| #802 | Wire-in #5: per-pair difficulty continual-learning | Consumer 5 landed | No |
| #803 | RECURSIVE-REVIEW 3-clean-pass | All Q1-Q5 + viz + wire-ins | No |
| #804 | This state tracker memo (in-progress) | — | No |
| #805 | PENDING OPERATOR DECISIONS | — | **YES — 4 decisions** |

## Operator decisions blocking forward motion

1. **Per-tier reward calibration** (Tier 1: 1.20× / Tier 2: 1.10× / Tier 3: 1.05× / no-proof: 1.0×) — symposium proposal; default OK?
2. **Tier 3 operator-review channel** — Codex pre-dispatch (Catalog #271) vs standalone manual vs autopilot pre-dispatch checkpoint (Catalog #243)
3. **Q4 $0.70 GPU spend approval** — Modal CPU $0.30 + paired CUDA T4 $0.40 for FEC6 Tier-2 FIRST empirical anchor
4. **HORIZON-CLASS scope** — FRONTIER_PURSUIT [0.147, 0.167] stops at Q1-Q5; ASYMPTOTIC_PURSUIT [0.050, 0.120] requires additional Q6+ stacking

## Empirical anchors (no-signal-loss)

| Anchor | Score | Axis | Hardware | Archive | Status |
|---|---|---|---|---|---|
| fec6 (current CPU frontier) | 0.19205 | [contest-CPU] | GHA Linux x86_64 | sha `6bae0201...` | Confirmed |
| pr106 format0d (CUDA frontier) | 0.20533 | [contest-CUDA] | T4 | sha `9cb989cef519...` | Confirmed |
| 8-pair fp64 master gradient | n/a | n/a | M5 Max CPU | `.omx/tmp/master_gradient_per_pair_8pair_fp64_validate.npy` | Validated; max\|G_avg − G_pp.mean\| = 4.29e-6 |
| 600-pair fp64 master gradient | n/a | n/a | M5 Max CPU | `.omx/tmp/master_gradient_per_pair_600pair_fp64_<utc>.npy` | EXTRACTING (~30 min) |
| FEC6 Venn classification (8-pair) | 90.7% PAIR_INVARIANT | — | — | `.omx/state/master_gradient_consumers/venn_classification_f174192aeadf_20260517T201006.json` | Validated; will re-classify on 600-pair |

## Wire-in coverage matrix (Catalog #125 hooks)

| Hook | Status | Lane |
|---|---|---|
| #1 Sensitivity-map | ✅ Wire-in #2 landed | `lane_wire_in_2_wyner_ziv_covariance_to_sensitivity_map_20260517` |
| #2 Pareto constraint | 🔄 Lagrangian planner in flight | `lane_per_pair_optimal_treatment_plan_via_lagrangian_dual_20260517` |
| #3 Bit-allocator | ⏳ queued #800 | `lane_wire_in_3_bit_allocator_per_pair_sensitivity_20260517` |
| #4 Cathedral autopilot dispatch hook | ⏳ queued #801 (partial via Venn already) | `lane_wire_in_4_cathedral_autopilot_per_byte_sensitivity_20260517` |
| #5 Continual-learning posterior | ⏳ queued #802 | `lane_wire_in_5_per_pair_difficulty_continual_learning_20260517` |
| #6 Probe-disambiguator | ⏳ Lagrangian planner will stub | — |

## Recursive adversarial review status

| Round | Scope | Status |
|---|---|---|
| Symposium T3 | Wyner-Ziv compliance + optimal design | ✅ Complete (8 op-routables) |
| Post-Q1-Q5 R1 | Yousfi+Fridrich+Wyner | ⏳ queued #803 (council rotation A) |
| Post-Q1-Q5 R2 | Boyd+Atick+Tishby | ⏳ queued #803 (rotation B) |
| Post-Q1-Q5 R3 | Carmack+Contrarian+Assumption-Adversary | ⏳ queued #803 (rotation C) |
| SEAL | 3 consecutive CLEAN | Pending |

Every round MUST include CLAUDE.md "Recursive adversarial review protocol" item #8 assumption-challenge axis per Catalog #291.

## Observability surface

- Continual-learning posterior: `.omx/state/continual_learning_posterior.jsonl`
- Council deliberations: `.omx/state/council_deliberation_posterior.jsonl`
- Modal call_id ledger: `.omx/state/modal_call_id_ledger.jsonl`
- Subagent progress: `.omx/state/subagent_progress.jsonl`
- Probe outcomes ledger: `.omx/state/probe_outcomes.jsonl`
- Wyner-Ziv deliverability probes (NEW): `.omx/state/wyner_ziv_deliverability/probe_*.json`
- Master gradient consumers sidecars: `.omx/state/master_gradient_consumers/*.json`
- This tracker: `.omx/research/comprehensive_state_tracker_20260517.md`
- Tasks: TaskList (#790-#805 this session)

## Mission alignment per CLAUDE.md "Mission alignment — non-negotiable"

`council_predicted_mission_contribution: frontier_breaking` — the Wyner-Ziv optimal stack predicted to land in FRONTIER_PURSUIT band [0.147, 0.167] beats the current 0.19205 CPU frontier by 0.025-0.045 points. The Lagrangian-dual planner addresses the 3 weaknesses (interaction modeling, sqrt(10·pose) non-linearity, global budget) that the heuristic cannot.

## Cross-references

- `.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md` — T3 verdict
- `.omx/research/wyner_ziv_optimal_implementation_queue_20260517.md` — 5-subagent dispatch plan
- `.omx/research/tac_side_information_namespace_design_20260517.md` — side-info channel architecture
- `feedback_per_pair_optimal_treatment_plan_via_lagrangian_dual_landed_20260517.md` (pending) — planner landing
- `feedback_grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_landed_20260517.md` — symposium landing

---

## ADDENDUM 2026-05-17 22:30 UTC — REDO+PIVOT comprehensive fix

**Lane**: `lane_redo_pivot_fix_all_phantom_score_substrate_class_shift_q4_budget_redirect_20260517`

**Operator directive**: *"We need to fix all and redo"* + *"Need to do an adversarial and bug hunter and rigor recursive pass this is concerning"*.

### Empirical state shift

- **Option B sweep landed 2026-05-17 22:10 UTC** at `.omx/state/wyner_ziv_deliverability/option_b_archive_member_sweep_20260517T221034.json`: all 8 VALIDATED contest archives AT entropy floor; aggregate ΔS = 0.000421 ≈ 0 at 0.001 leaderboard precision floor. **Recommended Q4 target = None. Q4 verdict = DEFER_Q4.**
- **Catalog #321 landed 2026-05-17 21:55 UTC** (Option C) extincted the phantom-score-from-research-sidecar class. 3 candidates (pr101_state_dict / pr106_state_dict / posenet_class_sensitivity) REJECTED_RESEARCH_SIDECAR.
- **Sister #823 SUPER_ADDITIVE integration** landed v2 cascade in `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2` for FUTURE genuine SUPER_ADDITIVE discoveries but self-flagged α=4.74 as FALSE_SIGNAL ARTIFACT (byte-identical renderer.bin SIREN timeout).

### Q4 / Stage 2 / SUPER_ADDITIVE → DEFERRED-pending-pivot

Per CLAUDE.md "Forbidden premature KILL" (DEFERRED not KILLED; reactivation criteria pinned):

| Lane | Verdict | Reactivation criteria |
|------|---------|------------------------|
| `lane_super_additive_lane_g_v3_siren_topology_integration_20260517` | DEFERRED (research_only=true) | Re-run pairwise_alpha probe against VALIDATED_CONTEST_MEMBER substrates only |
| `lane_q6_preprobe_pairwise_composition_alpha_20260517` | DEFERRED (research_only=true) | VALIDATED_CONTEST_MEMBER inputs only |
| `lane_batched_815_consumer_15_amendment_plus_816_meta_meta_cleanup_plus_q6_op3_extended_20260517` | DEFERRED (research_only=true) | Catalog #321 revalidation |

HORIZON-CLASS Stage 2 reactivation gate: clause #1 (Q4 anchor within ±10% of L5 codex band) is now STRUCTURALLY UNSATISFIABLE since Q4 = DEFER_Q4. **1/4 clauses compromised.**

### Substrate-class-shift pivot — top-1 redirect target

**Z6 ego-motion-conditioned predictive coding** (Rao-Ballard + Atick-Redlich + FiLM) is the recommended Q4 redirect target:

- Lane: `lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516` (L1; Phase 1B lift landed 2026-05-16)
- Recipe: `substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml` (research_only=true; pending Phase 2 council + Catalog #167 smoke-before-full)
- Predicted band [0.13, 0.16] [Dykstra-feasibility-validated per Z6 design memo §18]
- Smoke cost $1 Modal T4 (Q4's $0.70 rounds to canonical cost-band)

Alternative ranked candidates (#2 C6 IBPS, #3 ATW V2, #4 TT5L, #5 Z7, #6 Z8) enumerated in `.omx/operator_authorize_recipes/q4_substrate_class_shift_redirect_z6_predictive_coding_20260517T222800Z.yaml`.

### Catalog #322 STRICT preflight gate landed

`check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha` — sister of #321 at downstream autopilot-consumer surface. Refuses substrate_composition_matrix.json entries + pairwise_alpha_*.json artifacts that reference phantom-provenance candidates. **STRICT-from-byte-one; live count 0** (2 phantom pairwise_alpha files quarantined to `quarantine_phantom_pre_catalog_322/`). 25 dedicated tests pass.

### Cathedral autopilot adjustment chain audit

Helper `tac.optimization.substrate_composition_matrix.revert_phantom_source_rows()` audits both `canonical_substrate_inventory()` and `.omx/state/substrate_composition_matrix.json`. **Audit result**: 0 phantom-provenance rows in canonical_inventory (56 rows total clean); 0 phantom pair keys in matrix file (entries dict empty). v2 cascade `adjust_predicted_delta_for_composition_alpha_v2` no longer at risk of phantom-α absorption (the quarantined pairwise_alpha files could have promoted into the matrix posterior; gate #322 now refuses).

### Mission alignment per CLAUDE.md

This REDO is `frontier_protecting` (preventing phantom anchor pollution + apparatus drift) + queues `frontier_breaking` substrate-class-shift work via Q4 budget redirect to Z6 predictive coding. **Council predicted mission contribution: frontier_protecting + downstream frontier_breaking enablement.**

### Op-routables for R1 recursive adversarial review (#829)

1. **Yousfi+Fridrich+Wyner**: audit revert_phantom_source_rows helper for completeness (does it catch all phantom-provenance vectors? does it correctly handle the substrate_alias resolution path?).
2. **Boyd+Atick+Tishby**: assess Z6 selection rationale (is the Dykstra-feasibility-validated band [0.13, 0.16] still valid post-Option-B? does Atick-Redlich apply at the contest video's ego-motion regime?).
3. **Carmack+Contrarian+Assumption-Adversary**: META-ASSUMPTION ADVERSARIAL REVIEW per Catalog #291 — what shared assumption did the REDO itself operate within? Is "substrate-class-shift dominates within-class refinement" hard-earned or cargo-culted?
4. **Symposium T4**: kill-and-replace assessment — should Q4 be PERMANENTLY KILLED or merely DEFERRED? (Per CLAUDE.md: default DEFERRED unless research-path exhausted.)

### Sister-subagent ownership map honored

This REDO declared scope to lane registry mutations + canonical_inventory revert + reports/latest.md + state tracker memo + Catalog #322 + tests. No overlap with sister #824 (sidecar emission on src/tac/optimization/bit_allocator_end_to_end / jacobian_fisher_importance_allocator / field_equation_planner) per Catalog #230. Coordinated with sister #823 (SUPER_ADDITIVE integration landed earlier today) via DEFERRED-not-revert path on the v2 cascade (cascade preserved structurally; only the FALSE_SIGNAL input revoked).
