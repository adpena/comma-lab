# Retroactive sweep for META Finding A canonical 2-landing pattern — 2026-05-30

Companion sweep per Catalog #348 (`check_new_gate_landing_includes_retroactive_sweep_evidence`).

## 4-field contract

### 1. Bug-class symptom signature

The META Finding A bug class: `register_probe_outcome` callers populate the `next_action` field with substantive content but leave `reactivation_criteria` EMPTY. Downstream cathedral feeder consumers (per Catalog #335 auto-discovery + Catalog #344 canonical equation lookup) query `reactivation_criteria` field on probe outcomes to surface reactivation-ready deferred items. An EMPTY field structurally prevents the feeder from auto-picking up the probe.

**Symptom**: `reactivation_criteria` is None / empty list / placeholder string while `next_action` is a substantive (>=4 chars; non-placeholder) string. The feeder cannot route the probe for reactivation despite the operator having recorded the canonical action in `next_action`.

### 2. Pre-fix window

- Window start: 2026-05-16 (Catalog #313 canonical probe-outcomes ledger landing per `feedback_probe_outcomes_canonical_ledger_landed_20260516.md`)
- Window end: 2026-05-30 (THIS landing — Landing 1 forward fix + Landing 2 backward fix)
- Duration: 14 days
- Surface: every `register_probe_outcome` callsite in:
  - `src/tac/optimization/dp1_procedural_paired_adjudication.py:419`
  - `src/tac/master_gradient_wire_in.py:394`
  - `src/tac/provenance/builders.py:690`
  - `tools/probe_b1_patch_distribution_density.py:99`
  - All operator-spawned subagent direct calls (numerous)

### 3. Historical-KILL/DEFER/FALSIFY search results

**Search scope**: `.omx/state/probe_outcomes.jsonl` (canonical ledger) + `.omx/research/*.md` (operator-facing findings) for verdicts with EMPTY `reactivation_criteria`.

**Result**: 259 of 280 unique probe_ids in the canonical ledger have LATEST rows with EMPTY `reactivation_criteria`. Verdict breakdown (latest-row-wins per probe_id):

| Verdict | Count (latest row empty criteria) |
|---|---|
| DEFER | 104 |
| PARTIAL | 52 |
| PROCEED | 47 |
| INDEPENDENT | 24 |
| PROMOTE | 23 |
| OPERATOR_REVIEW_REQUIRED | 8 |
| KILL | 1 |

The DEFER verdicts are the highest-priority for backfill because they are BLOCKING per Catalog #313 — the feeder cannot route them for reactivation until criteria are populated.

**Honest classification per CLAUDE.md "NO FAKE IMPLEMENTATIONS"**: my read-only scan does NOT classify any historical KILL/DEFER/FALSIFY verdicts as "invalidated by this landing" because the canonical helper extension is forward-compatible — historical verdicts remain valid; the auto-derived `reactivation_criteria` adds queryable structure without overriding the operator's recorded `next_action`.

### 4. Per-finding RE-EVAL-priority assignment

Per the deferred-items feeder audit memo (commit `a9d45b171`) op-routable item enumeration, the 9 highest-EV operator-routable items blocked by EMPTY `reactivation_criteria` are:

| Priority | Probe ID | Classification | Backfill enables |
|---|---|---|---|
| HIGH | `probe_slot_ggg_x_cascade_a_fec10_paired_cuda_respawn_stand_down_20260530` | CRITERION_SISTER_SUBAGENT_REQUIRED | Auto-discovery of sister subagent spawn |
| HIGH | `slot_ccc_hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010` | CRITERION_PAID_DISPATCH_REQUIRED | Auto-discovery of paired-CUDA dispatch readiness |
| HIGH | `slot_ddd_paired_cuda_ratification_dispatch_wave_stand_down_20260529` | CRITERION_PAID_DISPATCH_REQUIRED | Auto-discovery of L28-patched archive build readiness |
| HIGH | `slot_yy_hill_canonical_l0_scaffold_landing_20260529` | CRITERION_PAID_DISPATCH_REQUIRED | Auto-discovery of paired empirical anchor readiness |
| HIGH | `slot_tt_pr110_opt_5_boundary_region_waterfill_l0_scaffold_landed_20260529` | CRITERION_PAID_DISPATCH_REQUIRED | Auto-discovery of paired-CUDA anchor readiness |
| HIGH | `slot_ff_pr110_opt_7_uniward_inverse_scorer_basis_expansion` | CRITERION_PAID_DISPATCH_REQUIRED | Auto-discovery of paired CPU+CUDA dispatch readiness |
| MED | `z5_rao_ballard_identity_predictor_disambiguator_wave_n_plus_43_anchor_3_of_3` | CRITERION_PATHS_ENUMERATED_IN_MEMO | Auto-discovery of 4 enumerated reactivation paths |
| MED | `slot_w_wave_n_plus_40_z6_v2_identity_predictor_disambiguator_mlx_infrastructure_gap` | CRITERION_WIRE_IN_REQUIRED | Auto-discovery of identity_predictor wire-in landing |
| MED | `slot_w_wave_n_plus_40_z6_identity_predictor_disambiguator_mlx_infrastructure_gap` | CRITERION_WIRE_IN_REQUIRED | Auto-discovery of Z7-Mamba-2 canonical pattern replication |

After backfill `--apply`, all 9 items become structurally queryable by the cathedral feeder consumer.

**Honest non-finding**: No historical KILL/DEFER/FALSIFY verdicts were INVALIDATED by this landing — the canonical helper extension is forward-compatible and the backfill tool is APPEND-ONLY per Catalog #110/#113 HISTORICAL_PROVENANCE discipline. The 259 historical EMPTY rows simply gain auto-derived queryable structure without overriding the operator's recorded `next_action`.

## Catalog #348 acceptance fields

- **gate_function_name**: `check_new_gate_landing_includes_retroactive_sweep_evidence`
- **landing_commit_hash**: pending commit (this memo + canonical helper + tool + tests)
- **sweep_completion_date**: 2026-05-30
- **historical_findings_audited**: 259 EMPTY-criteria rows (across 7 verdict classes)
- **findings_re_evaluated**: 9 highest-EV operator-routable items per deferred-items feeder audit
- **findings_status_changes**: 0 (none invalidated; APPEND-ONLY semantics)

## Mission contribution per Catalog #300

`apparatus_maintenance` — closes the META Finding A canonical bug class structurally at TWO surfaces (forward Landing 1 + backward Landing 2); unblocks downstream cathedral feeder consumers from silently dropping reactivation-ready deferred items. No direct score-lowering claim; the apparatus mutation enables downstream feeder-consumer-driven candidate dispatch which routes future score improvements.
