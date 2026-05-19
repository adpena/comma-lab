---
council_tier: T2
council_attendees: [Shannon, Dykstra, Contrarian, Assumption-Adversary, Hassabis, Hafner, vanDenOord]
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_dissent:
  - member: Contrarian
    verbatim: "Each of the 3 E.7/E.8/E.10 dispatches has at least one structural blocker (Catalog #313 probe outcome / Catalog #240 recipe-vs-trainer-state / Catalog #325 per-substrate symposium). Bypassing them via operator-frontier-override per Catalog #300 is documented but should require explicit per-experiment operator-verbatim quote. The plain 'Approved proceed with all' standing approval is NOT per-experiment-frontier-override; the standing approval authorizes the experiments to be SET UP within discipline, not to BYPASS the per-experiment guards."
council_assumption_adversary_verdict:
  - assumption: "Operator's standing 'Approved proceed with all' authorizes bypassing per-experiment Catalog #313/#240/#325 guards"
    classification: CARGO-CULTED
    rationale: "Standing operator approval covers ENVELOPE ($3.50-9.50) but per-experiment guards are STRUCTURAL discipline that document why dispatches were deferred. Operator-frontier-override per Catalog #300 requires per-deliberation verbatim quote, not a session-level standing approval."
  - assumption: "3-of-3 dispatches can fire without trainer surgery / recipe flips / symposium evidence"
    classification: CARGO-CULTED-EMPIRICALLY-CONFIRMED-BLOCKED
    rationale: "Phase-0 PV confirms 3-of-3 have at least one blocker (E.7 no --codebook-size CLI flag, E.8 t_final default 1e-4 != council 1.0 cap, E.10 trainer + recipe + probe outcomes ALL blocking)"
council_decisions_recolined: []
council_decisions_recorded:
  - "STOP-AND-REPORT: defer the 3 dispatches pending per-experiment preparation"
  - "Emit per-experiment blocker enumeration + canonical preparation work items"
  - "Surface to operator for per-experiment frontier-override decisions"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: multi
deferred_substrate_retrospective_due_utc: "2026-06-18T04:23:01Z"
predicted_mission_contribution: rigor_overhead
finding_action_class: defer
finding_followup_dispatch_envelope_usd: 0.00
finding_canonical_path: pre_dispatch_blocker_audit
---

# 3-smoke paid-GPU reactivation wave: STOP-AND-REPORT blocker audit

**Lane**: `lane_3_smoke_paid_gpu_reactivation_wave_20260518` L0 → L1 (impl_complete via this audit memo + memory entry)
**Subagent**: `3_smoke_paid_gpu_reactivation_wave_20260518`
**Phase**: 0 (premise verification) — BLOCKERS DISCOVERED across all 3 experiments
**GPU spend**: $0 (no dispatches fired)

## Summary

Phase-0 premise verification per Catalog #229 (before edit) discovered structural blockers for ALL 3 of the operator-approved E.7/E.8/E.10 dispatches. Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + "Forbidden premature KILL" + Catalog #313 predecessor probe outcome consultation + Catalog #325 per-substrate symposium discipline: none of the 3 dispatches can fire as-is without either (a) preparation work to remove the blockers, or (b) explicit operator-frontier-override per Catalog #300 with per-experiment verbatim quote.

## Per-experiment blocker enumeration

### E.7 — VQ K-sweep paired-comparison smoke

| Blocker | Source | Status | Remediation |
|---|---|---|---|
| **B1** — VQ-VAE trainer hardcodes `codebook_size=16` at `experiments/train_substrate_vq_vae.py:730`; no `--codebook-size` CLI flag | grep `add_argument.*codebook` returns 0 matches | BLOCKING | Add `--codebook-size` CLI flag + plumb through `cfg.codebook_size` (~30 LOC; ~30 min editor) |
| **B2** — No per-substrate symposium for VQ K-sweep variant within 14 days per Catalog #325 | No matching memo at `.omx/research/council_*_vq_vae_k_sweep_*_<YYYYMMDD>.md` | BLOCKING | Convene per-substrate symposium (Wave 2A row #2+#3 + Catalog #325 6-step contract); ~1h |
| **B3** — VQ-VAE recipe lacks `predicted_band_validation_status` per Catalog #324 | grep recipe shows no `predicted_band_validation_status` line | BLOCKING | Add `predicted_band_validation_status: pending_post_training` + reactivation criteria (~5 min) |
| **B4** — Recipe currently targets full 2000-epoch dispatch ($6 p50); need K-sweep variant recipe ($0.30 × 8 K-values = $2.40) | Recipe is `substrate_vq_vae_modal_a100_dispatch.yaml`, 2000 epochs | BLOCKING | Create variant recipe `substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml` per Catalog #167 (~30 min) |

**Verdict**: PREP-2H-EDITOR-BEFORE-DISPATCH. Total prep work: ~2h editor; recipe variant + trainer CLI flag + per-substrate symposium memo + Catalog #324 field. THEN dispatch via canonical operator-authorize per the original plan.

### E.8 — SGLD convergence-diagnostic smoke

| Blocker | Source | Status | Remediation |
|---|---|---|---|
| **B1** — Recipe `substrate_stack_of_stacks_modal_a100_dispatch.yaml` is `dispatch_enabled: false` + `research_only: false` (mixed state) | grep yaml shows `dispatch_enabled: false` | BLOCKING | Flip `dispatch_enabled: true` after per-substrate symposium per Catalog #240 (~10 min) |
| **B2** — Trainer `--langevin-t-final` default 1e-4 differs from council memo references (cap=1.0 vs formula=17.4); the cap=1.0 referenced in council memo is from Wave 2A row #8 not from the trainer default | grep trainer shows `--langevin-t-final default 1e-4` | INVESTIGATIVE | Clarify: council's "cap=1.0" is in Wave 2A audit (analytical row), NOT in trainer code; trainer default is operator-set 1e-4; council deliberation may have been on stale assumptions. Verify with Wave 2A row #8 source before sweep. (~30 min) |
| **B3** — No per-substrate symposium for SGLD convergence variant within 14 days per Catalog #325 | No matching memo | BLOCKING | Convene per-substrate symposium (~1h) |
| **B4** — Predecessor probe outcome `sgld_t_final_convergence_diagnostic_pending_20260518` blocking status + DEFER verdict in `.omx/state/probe_outcomes.jsonl` | grep ledger shows blocking | BLOCKING | The probe outcome's `next_action: "dispatch convergence-diagnostic smoke"` IS the unlock criterion; operator-frontier-override per Catalog #300 OR fresh probe outcome registration ratifying the dispatch (~5 min) |

**Verdict**: PREP-1.5H-EDITOR-BEFORE-DISPATCH. Total prep work: ~1.5h editor; recipe variant + Catalog #325 symposium + Catalog #313 probe outcome ratification.

### E.10 — Z7-Mamba-2 stability-fix smoke

| Blocker | Source | Status | Remediation |
|---|---|---|---|
| **B1** — Recipe `substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml` is `dispatch_enabled: false` + `research_only: true` with 8 explicit `dispatch_blockers` | YAML | BLOCKING (HARD) | Wave N+1 council required to flip; 8 blockers include "z7_mamba2_dispatch_requires_z7_gru_wave_2_disambiguator_outcome" + "z7_mamba2_dispatch_requires_wave_n_plus_1_council_after_z7_gru_outcome" + "z7_mamba2_beta_ib_parameter_requires_c6_ibps_phase_2_empirical_beta_anchor" + "z7_mamba2_mamba_ssm_pypi_install_must_succeed_in_modal_a100_image_pre_dispatch" |
| **B2** — TWO blocking probe outcomes in `.omx/state/probe_outcomes.jsonl`: `z7_mamba2_vs_z7_lstm_paired_mps_proxy_20260518` (advisory DEFER) + `z7_mamba2_canonical_scale_stability_20260518` (blocking DEFER) | grep ledger | BLOCKING | Catalog #313 dispatch refused; operator-frontier-override OR probe outcome ratification required |
| **B3** — Recipe explicitly references "Wave N+1 council" + "Z7-GRU Wave 2 disambiguator" as prerequisites; sister C6 IBPS Phase 2 β-anchor blocker is on critical path | YAML dispatch_blockers | BLOCKING (HARD) | Multi-week dependency chain; cannot ship in this 3-smoke wave |
| **B4** — Even if recipe flipped, `mamba_ssm` PyPI install may fail in Modal A100 image; not validated | recipe says "mamba_ssm_pypi_install_must_succeed_in_modal_a100_image_pre_dispatch" | BLOCKING | Pre-flight smoke testing required ($0.10 sanity-check dispatch) |

**Verdict**: PREP-MULTI-WEEK-BEFORE-DISPATCH. This experiment cannot ship in the 3-smoke wave. The trainer `_full_main` IS implemented (despite stale docstring claiming NotImplementedError) but recipe blockers are HARD and depend on prerequisite outcomes (Z7-GRU Wave 2, C6 IBPS Phase 2 β-anchor). Operator-frontier-override per Catalog #300 with explicit acknowledgment of the 8 dispatch_blockers would be required.

## Aggregate verdict

| Experiment | Dispatch envelope | Prep envelope | Verdict |
|---|---|---|---|
| E.7 VQ K-sweep | $1-3 (Modal T4, K-sweep × 8) | ~2h editor | **PREP-THEN-DISPATCH** (defer to follow-on subagent) |
| E.8 SGLD convergence | $0.50-1.50 (Modal T4) | ~1.5h editor | **PREP-THEN-DISPATCH** (defer to follow-on subagent) |
| E.10 Z7-Mamba-2 stability | $2-5 (Modal T4) | MULTI-WEEK prerequisites | **DEFER-MULTI-WEEK** (cannot ship in this wave) |

**Total spend this wave**: $0 (no dispatches fired).

**Recommended next move**: spawn 2 follow-on subagents to handle E.7 + E.8 prep (the 3.5h editor work is straightforward + composable in parallel), then fire the 2 prepared dispatches in a subsequent wave. E.10 routes to the multi-week Z7-Mamba-2 pursuit per the existing dispatch_blockers (Z7-GRU Wave 2 outcome + C6 IBPS Phase 2 β-anchor + Wave N+1 council).

## Sister coordination per Catalog #230 ownership map

- Sister 1 (MPS Conv2d wrap fix): owns `src/tac/mps_diagnostic/targeted_fix.py` — already COMPLETE per checkpoint ledger (commit `24278cf06`)
- Sister 2 (META-PHANTOM-API): owns `src/tac/preflight.py` Catalog #287 scope-extend — IN-FLIGHT (`catalog_287_scope_extend_phantom_api_20260518`)
- This audit: only touches NEW namespace paths (`.omx/research/3_smoke_paid_gpu_reactivation_wave_blocker_audit_*.md` + lane registry for this lane + memory entry); disjoint from both sisters

## Catalog #229 premise verification log

- PV-0: canonical helpers exist (`tac.probe_outcomes_ledger.register_probe_outcome`, `tac.deploy.modal.call_id_ledger.register_dispatched_call_id`, `tac.council_continual_learning.append_council_anchor`, `tools/operator_authorize.py`) — CONFIRMED via direct path checks
- PV-1: 3 council memos exist at `.omx/research/council_t3_finding_1_*_20260518.md` + `council_t2_finding_2_*_20260518.md` + `council_t3_finding_4_*_20260518.md` — CONFIRMED
- PV-2: aggregate dispatch plan at `.omx/research/grand_council_findings_deliberation_wave_aggregate_dispatch_plan_20260518.md` — CONFIRMED
- PV-3: sister subagents in flight understood via `.omx/state/subagent_progress.jsonl` — CONFIRMED (1 active: catalog_287_scope_extend_phantom_api_20260518)
- PV-4: VQ-VAE trainer CLI flags inspected (`grep add_argument.*codebook` returns 0 matches) — CONFIRMED no `--codebook-size`
- PV-5: stack_of_stacks trainer CLI flags inspected (`grep --langevin`) — CONFIRMED `--langevin-t-final` exists, default 1e-4
- PV-6: Z7-Mamba-2 recipe inspected — CONFIRMED `dispatch_enabled: false`, `research_only: true`, 8 dispatch_blockers
- PV-7: probe outcomes ledger inspected — CONFIRMED 2 blocking outcomes for Z7-Mamba-2 + 1 blocking for SGLD
- PV-8: active lane dispatch claims inspected — CONFIRMED no conflicting claims on E.7/E.8/E.10 lane_ids

## 6-hook wire-in declaration per Catalog #125

1. Sensitivity-map contribution: N/A — blocker audit only; no signal weight emission
2. Pareto constraint: N/A — defer-only audit, no constraint contribution
3. Bit-allocator hook: N/A — no per-tensor importance change
4. Cathedral autopilot dispatch hook: ACTIVE — the blocker audit feeds the autopilot dispatch decision (3 deferred candidates with prep envelopes)
5. Continual-learning posterior: ACTIVE — append_council_anchor for THIS T2 deliberation
6. Probe-disambiguator: N/A — this IS the disambiguator (which experiments are dispatchable now vs need prep vs need multi-week unlock)

## Atom emission per Catalog #245/#323

Atom: `build_council_deliberation_atom(atom_id="audit_3_smoke_paid_gpu_reactivation_wave_blocker_20260519T042301Z", deliberation_id="3_smoke_paid_gpu_reactivation_wave_blocker_audit", council_tier="T2", council_verdict="DEFER_PENDING_EVIDENCE", predicted_impact_lower=0.0, predicted_impact_upper=0.0, cost_envelope_usd=0.00, memory_path=".omx/research/3_smoke_paid_gpu_reactivation_wave_blocker_audit_20260519T042301Z.md")`

## Cross-references

- Standing directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_findings_review_grand_council_deliberation_standing_directive_20260518.md`
- Operator approval: "Approved proceed with all" 2026-05-18 (covers $3.50-9.50 envelope but not per-experiment blockers)
- E.7 council memo: `.omx/research/council_t3_finding_1_vq_codebook_anti_pareto_20260518.md`
- E.8 council memo: `.omx/research/council_t2_finding_2_sgld_t_final_ceiling_cap_20260518.md`
- E.10 council memo: `.omx/research/council_t3_finding_4_z7_mamba2_indeterminate_with_nuance_20260518.md`
- Aggregate plan: `.omx/research/grand_council_findings_deliberation_wave_aggregate_dispatch_plan_20260518.md`
- Z7-Mamba-2 design memo: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
- Z7-Mamba-2 recipe: `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml`
- VQ-VAE recipe: `.omx/operator_authorize_recipes/substrate_vq_vae_modal_a100_dispatch.yaml`
- stack_of_stacks recipe: `.omx/operator_authorize_recipes/substrate_stack_of_stacks_modal_a100_dispatch.yaml`
- Probe outcomes (blocking): `.omx/state/probe_outcomes.jsonl` rows for `z7_mamba2_vs_z7_lstm_paired_mps_proxy_20260518`, `z7_mamba2_canonical_scale_stability_20260518`, `sgld_t_final_convergence_diagnostic_pending_20260518`
- CLAUDE.md non-negotiables: "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" (Catalog #315), "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (Catalog #325), "Forbidden premature KILL", "Mission alignment", "Operator gates must be wired and used"

## Reactivation criteria (per Catalog #298)

- **E.7 reactivation**: VQ-VAE trainer gets `--codebook-size` flag + per-substrate symposium memo + Catalog #324 field + variant recipe lands → dispatch via canonical operator-authorize at $1-3 envelope
- **E.8 reactivation**: stack_of_stacks recipe flipped + per-substrate symposium memo + probe outcome ratified → dispatch via canonical operator-authorize at $0.50-1.50 envelope
- **E.10 reactivation**: Z7-GRU Wave 2 disambiguator outcome lands + C6 IBPS Phase 2 β-anchor lands + Wave N+1 council PROCEED-unconditional + recipe `dispatch_enabled: true` flip + `mamba_ssm` pre-flight smoke passes → dispatch at $2-5 envelope (multi-week timeline)


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
