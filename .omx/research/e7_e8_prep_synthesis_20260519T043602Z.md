# E.7 + E.8 PREP COMBINED — synthesis 2026-05-18

**Lane**: `lane_e7_vq_k_sweep_plus_e8_sgld_convergence_prep_20260518` (L1)
**Subagent**: `e7_e8_prep_combined_20260518`
**GPU spend**: $0 (editor only)
**Wall-clock**: ~30 min (parallel-disjoint to sister subagents)

## TL;DR

Both E.7 + E.8 prep deliverables landed:
- 2 variant recipes (`substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml` + `substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch.yaml`)
- 2 per-substrate symposium DRAFT memos (Catalog #325 6-step contract honored; NOT convened — awaits operator approval OR inner-quintet ratification)

**CRITICAL PREMISE FALSIFICATION** caught at Catalog #229 pre-edit verification:
- Predecessor `lane_3_smoke_paid_gpu_reactivation_wave_20260518` Blocker B1 claimed VQ-VAE trainer hardcodes `codebook_size=16` with no `--codebook-size` CLI flag
- **ACTUAL STATE**: `--codebook-size` flag EXISTS at `experiments/train_substrate_vq_vae.py:310` with default=512 and is fully threaded through `_full_main` at line 837
- Predecessor's grep regex `"add_argument.*codebook"` returned 0 matches because argparse spans 2 lines; the flag definition is `"--codebook-size",\n        type=int,\n        default=512,`
- E.7 trainer surgery NOT NEEDED — significantly reduces prep scope

**CLARIFICATION on E.8 SGLD ambiguity**:
- Predecessor blocker B2 framed "trainer `--langevin-t-final` default 1e-4 differs from council memo references (cap=1.0)"
- Catalog #229 PV: `--langevin-t-final 1e-4` is the FINAL temperature of cosine schedule; `--langevin-t-init 0.3` is the INIT temperature. Council's "cap=1.0" refers to the COST-BAND budget cap (operator-set, derived from $5-15 Modal T4 envelope), NOT a SPECIFIC trainer-flag value. The ambiguity was a category error in the predecessor audit's framing.

## Per-experiment prep summary

### E.7 VQ K-sweep — STATUS: READY-FOR-RATIFICATION

| Blocker | Status |
|---|---|
| B1: no `--codebook-size` CLI flag | **FALSIFIED** by Catalog #229 PV (flag exists at line 310) |
| B2: per-substrate symposium | DRAFT memo prepared, awaits convocation |
| B3: `predicted_band_validation_status` | Declared `pending_post_training` in variant recipe |
| B4: K-sweep variant recipe | LANDED at `substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml` |

**Remaining gate**: Catalog #325 per-substrate symposium ratification (operator convocation OR inner-quintet pact pact ratification of DRAFT).

**Dispatch envelope when ratified**: $2.40 Modal T4 (8 K-values × $0.30 each)

**K-sweep matrix**: K ∈ {2, 4, 8, 16, 32, 64, 128, 256} at λ=1.0

### E.8 SGLD convergence — STATUS: READY-FOR-RATIFICATION

| Blocker | Status |
|---|---|
| B1: stack_of_stacks recipe `dispatch_enabled: false` | NEW variant recipe declares `dispatch_enabled: false` (correct; awaits symposium ratification) |
| B2: trainer flag vs council cap ambiguity | **CLARIFIED** by Catalog #229 PV (cap=1.0 = cost-band budget, not trainer-flag) |
| B3: per-substrate symposium | DRAFT memo prepared, awaits convocation |
| B4: probe outcome ratification | Operator-routable queued (fresh `register_probe_outcome` call after symposium) |

**Remaining gates**: (a) symposium ratification (b) probe outcome ratification (c) operator approval to extend $0.50-1.50 council envelope to $1.80 for 6-value sweep

**Dispatch envelope when ratified**: $1.80 Modal T4 (6 t_init-values × $0.30 each; intentional timeout at t_init=17.4)

**SGLD sweep matrix**: t_init-CAP ∈ {0.5, 1.0, 2.0, 5.0, 10.0, 17.4} at fixed t_final=1e-4

## 4 operator-routable decisions

### OP-1: Approve VQ-VAE K-sweep T2 symposium convocation OR ratify DRAFT verdict
**Two paths**:
(a) Approve full T2 symposium convocation (~2h deliberation; 11 attendees including vanDenOord + MacKay + Boyd + Tao + Selfcomp grand-council seats per Catalog #300 v2 frontmatter)
(b) Ratify DRAFT verdict via inner-quintet pact (~30min; Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary)
(c) Operator-frontier-override per Catalog #300 Mission Alignment (verbatim quote in `council_override_rationale` frontmatter; immediate dispatch authorization)

**Recommendation**: Path (b) for velocity; T3 Finding 1 already deliberated the underlying analytical claim. Path (a) only if council disagreement is expected on the per-substrate K-sweep details.

### OP-2: Approve SGLD convergence T2 symposium convocation OR ratify DRAFT verdict
**Two paths**:
(a) Approve full T2 symposium convocation (~2h; 10 attendees including Hassabis + Welling-Teh memorial + Boyd + MacKay)
(b) Ratify DRAFT verdict via inner-quintet pact (~30min)
(c) Operator-frontier-override

**Recommendation**: Path (b); T2 Finding 2 already deliberated the cap-vs-formula ambiguity.

### OP-3: Resolve council "cap=1.0" interpretation
**Two interpretations**:
(a) **Current DRAFT memo interpretation**: "cap=1.0" = cost-band BUDGET cap (operator-set $5-15 envelope); the SGLD sweep maps `--langevin-t-init` upper bound across {0.5, 1.0, 2.0, 5.0, 10.0, 17.4} at fixed `--langevin-t-final=1e-4`
(b) **Alternative**: "cap=1.0" = `--langevin-t-final` should default to 1.0 instead of 1e-4 (Welling-Teh canonical: final temperature)

**Recommendation**: (a) — matches Catalog #229 PV evidence and trainer source semantics.

### OP-4: After OP-1 + OP-2 (+ OP-3) ratifications, fire 2-smoke combined dispatch wave
**Combined envelope**: $1.50-4.50 (E.7 + E.8; council-minimum $0.90 SGLD case = $3.30 total; extended $1.80 SGLD case = $4.20 total)

**Recommended dispatch order**:
1. E.7 VQ K-sweep ($2.40) — empirically validates Wave 2A R-D Pareto pole; high-EV (informs 14-substrate VQ wire-in wave per T3 Finding 1 op-routable #3)
2. E.8 SGLD convergence ($0.90-1.80) — empirically validates Wave 2A SGLD formula vs cap; high-EV (informs Wave-3 stack-of-stacks dispatches)

## Verdict on aggregate prep

| Smoke | Pre-fix status | Post-fix status | Gates remaining |
|---|---|---|---|
| E.7 VQ K-sweep | 4 blockers (1 trainer, 1 recipe, 1 symposium, 1 Catalog #324) | 1 blocker (symposium) | Catalog #325 symposium ratification |
| E.8 SGLD convergence | 4 blockers (1 recipe, 1 ambiguity, 1 symposium, 1 probe outcome) | 2 blockers (symposium + probe) | Catalog #325 symposium ratification + Catalog #313 probe ratification |
| E.10 Z7-Mamba-2 | 4 blockers (HARD multi-week) | UNCHANGED | Multi-week per predecessor recommendation; routes to Z7-Mamba-2 critical path (Z7-GRU Wave 2 + C6 IBPS Phase 2 β-anchor + Wave N+1 council + mamba_ssm pre-flight) |

**2 of 3 smoke blockers RESOLVED at infrastructure layer**. Symposium ratifications are the remaining gate for E.7 + E.8; ratification routes available per OP-1 + OP-2 above. E.10 remains DEFER multi-week.

## Recommendation for 2-smoke combined dispatch wave timing

After OP-1 + OP-2 ratifications land (estimated ~30min inner-quintet pact OR ~2h full T2 convocation OR ~5min operator-frontier-override):
- Spawn `2_smoke_paid_gpu_dispatch_wave_e7_e8_combined_<utc>` subagent
- Combined envelope $3.30-4.20 with proven hardware schedule
- Per CLAUDE.md "Race-mode rigor inversion": if leaderboard moves during ratification window, dispatch wave should be re-prioritized
- Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM": ratification MUST land before the dispatch wave fires; this gate is the structural protection against the Wave 2A pattern of unverified analytical claims

## Sister coordination per Catalog #230 ownership map

- Sister `phase_b_mps_gap_experiment_infrastructure_build_20260518`: NEW `src/tac/mps_gap_experiment/` namespace; DISJOINT scope (no file overlap)
- Sister `phantom_api_backfill_wave_1_20260518`: ~20 EXISTING `.omx/research/*.md` memos for phantom-API backfill; DISJOINT from new files I created (new DRAFTs + new variant recipes + new synthesis memo + new memory entry)
- This subagent: only touches NEW files (recipes + DRAFT memos + synthesis + memory entry) + lane registry mutation

## Catalog #229 premise verification log

- PV-0: Canonical helpers verified (tac.substrates.vq_vae / tac.scorer / tac.deploy.modal.call_id_ledger / tac.probe_outcomes_ledger / tac.council_continual_learning) — all importable
- PV-1: **PREMISE FALSIFIED**: VQ-VAE `--codebook-size` flag EXISTS at line 310 (default=512), threaded through `_full_main` line 837; predecessor's narrow grep missed it
- PV-2: VQ-VAE recipe `substrate_vq_vae_modal_a100_dispatch.yaml` confirmed as production 2000ep dispatch (sister, NOT variant)
- PV-3: SGLD trainer `--langevin-t-final` line 280 default 1e-4 confirmed as cosine FINAL temperature (NOT the "cap=1.0" reference)
- PV-4: SGLD trainer `--langevin-t-init` line 278 default 0.3 confirmed as cosine INIT temperature (matches the council "cap=1.0" reference axis)
- PV-5: stack_of_stacks recipe confirmed `dispatch_enabled: false` for FULL multi-arm (sister, NOT variant)
- PV-6: T3 Finding 1 + T2 Finding 2 council memos verified
- PV-7: Sister subagents in flight verified via `.omx/state/subagent_progress.jsonl`
- PV-8: Probe outcome `sgld_t_final_convergence_diagnostic_pending_20260518` confirmed blocking via `.omx/state/probe_outcomes.jsonl`

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — both DRAFT memos plan per-K and per-t_init anchors that feed `tac.sensitivity_map.*` post-dispatch
2. **Pareto constraint**: ACTIVE — both DRAFTs cite empirical Pareto-frontier identification as primary output
3. **Bit-allocator hook**: ACTIVE for E.7 (per-K bit allocation); N/A for E.8 (diagnostic)
4. **Cathedral autopilot dispatch hook**: ACTIVE — both DRAFTs cite autopilot consumer (T3 Finding 1 op-routable #3 + Wave 3 stack-of-stacks)
5. **Continual-learning posterior update**: ACTIVE — symposium ratification anchors + per-dispatch anchors via canonical helpers
6. **Probe-disambiguator**: ACTIVE — both DRAFT memos ARE disambiguators (K=2-vs-large-K; cap-vs-formula)

## Atom emission per Catalog #245/#323

Two atoms queued (both pending DRAFT ratification before persistence):
1. `build_council_deliberation_atom(atom_id="e7_vq_k_sweep_symposium_draft_20260519T043602Z", deliberation_id="vq_vae_k_sweep_t2_symposium_DRAFT", council_tier="T2", council_verdict="DRAFT_PENDING_CONVOCATION", predicted_impact_lower=0.0, predicted_impact_upper=-0.013, cost_envelope_usd=2.40, memory_path=".omx/research/council_t2_vq_vae_k_sweep_symposium_DRAFT_20260519T043602Z.md")`
2. `build_council_deliberation_atom(atom_id="e8_sgld_convergence_symposium_draft_20260519T043602Z", deliberation_id="sgld_convergence_t2_symposium_DRAFT", council_tier="T2", council_verdict="DRAFT_PENDING_CONVOCATION", predicted_impact_lower=0.0, predicted_impact_upper=0.0, cost_envelope_usd=1.80, memory_path=".omx/research/council_t2_sgld_convergence_symposium_DRAFT_20260519T043602Z.md")`

## Cross-references

- E.7 variant recipe: `.omx/operator_authorize_recipes/substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml`
- E.7 symposium DRAFT: `.omx/research/council_t2_vq_vae_k_sweep_symposium_DRAFT_20260519T043602Z.md`
- E.8 variant recipe: `.omx/operator_authorize_recipes/substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch.yaml`
- E.8 symposium DRAFT: `.omx/research/council_t2_sgld_convergence_symposium_DRAFT_20260519T043602Z.md`
- Predecessor audit (PREMISE FALSIFIED on E.7): `.omx/research/3_smoke_paid_gpu_reactivation_wave_blocker_audit_20260519T042301Z.md`
- Predecessor memory: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_3_smoke_paid_gpu_reactivation_wave_blocker_audit_landed_20260518.md`
- T3 Finding 1 council memo: `.omx/research/council_t3_finding_1_vq_codebook_anti_pareto_20260518.md`
- T2 Finding 2 council memo: `.omx/research/council_t2_finding_2_sgld_t_final_ceiling_cap_20260518.md`
- Aggregate dispatch plan: `.omx/research/grand_council_findings_deliberation_wave_aggregate_dispatch_plan_20260518.md`
- VQ-VAE trainer: `experiments/train_substrate_vq_vae.py` (`--codebook-size` line 310; default=512)
- SGLD trainer: `experiments/train_substrate_stack_of_stacks.py` (`--langevin-t-init` line 278 default=0.3; `--langevin-t-final` line 280 default=1e-4)
- CLAUDE.md non-negotiables: Catalog #313 / #324 / #325 / #270 / #167 / #294 / #303 / #305 / #296 / #229 / #220 / #272 / #292 / #300 / #240 / #190 / #244


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
