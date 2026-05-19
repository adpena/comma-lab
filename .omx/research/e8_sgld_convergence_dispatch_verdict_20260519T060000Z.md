---
council_tier: T1
council_attendees: [Claude-subagent-cable_b1_e7_e8_dispatch]
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "stack_of_stacks trainer entry reachable via this recipe runs SGLD polish epochs"
    classification: CARGO-CULTED
    rationale: "Empirical dispatch revealed trainer enters single_arm_a1_passthrough mode in 1 second; SGLD polish loop is NOT reached via the recipe's STACK_OF_STACKS_OUTER_K=1 + middle_arm_substrate_ids=a1 single-arm canary path"
  - assumption: "auth_eval supports /tmp/pact temp_work_dir on Modal worker"
    classification: CARGO-CULTED
    rationale: "experiments/contest_auth_eval.py refuses temp storage per Catalog #204 durable provider work_dir discipline; recipe needs --allow-temp-work-dir for diagnostic-only OR durable redirect"
council_decisions_recorded:
  - "op-routable #1: re-scope SGLD convergence-diagnostic to a dedicated SGLD-only trainer entrypoint that actually runs Welling-Teh SGLD polish epochs at the recipe-reachable entry"
  - "op-routable #2: recipe redirect OUTPUT_DIR to durable provider work_dir OR add --allow-temp-work-dir for diagnostic-only auth_eval scope"
  - "op-routable #3: STACK_OF_STACKS_LANGEVIN_T_INIT_CAP env var currently INERT in trainer's single_arm_a1_passthrough mode — sweep is not actuated until op-routable #1 lands"
  - "op-routable #4: re-register fresh probe outcome PROCEED after op-routable #1-3 land per Catalog #313 sister-probe-alternative-reducer (Catalog #308)"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: stack_of_stacks
deferred_substrate_retrospective_due_utc: 2026-06-18T06:00:00Z
predicted_mission_contribution: rigor_overhead
finding_action_class: research
finding_followup_dispatch_envelope_usd: 0.0
finding_canonical_path: research
---

# E.8 SGLD convergence-diagnostic dispatch verdict

**Status:** DEFER_PENDING_EVIDENCE per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #308 alternative-reducer discipline

**Cable B1 lane:** `lane_cable_b1_e7_e8_combined_dispatch_20260519` (operator-frontier-override ratified per Catalog #300)

## Empirical anchor

Dispatch `fc-01KRZCSQ7FPVMSAXZQDSZJCTN4` 2026-05-19T05:54:09Z (Modal T4):

- rc=1 at 19.5s elapsed
- Stage progression: device_resolved → arm_spec_resolved → base_archive_loaded → arm_0_residual_train_begin → arm_0_residual_train_done → byte_budget_validated → composed → archive_built (all in same second 2026-05-19T05:56:14Z)
- Archive built: `archive.zip` sha=110cfaa3f2ebbd02b91542633445e54a837ea663f98a7807a914f95651fdff9f (179008 bytes)
- Auth-eval refused: "evidence path is under temp storage: /tmp/pact/lane_stack_of_stacks_results/auth_eval/eval_work" per Catalog #204

## Root cause analysis

### Finding 1: Trainer scope mismatch

The recipe's `STACK_OF_STACKS_OUTER_K=1` + `STACK_OF_STACKS_MIDDLE_ARM_SUBSTRATE_IDS=a1` configuration triggers the trainer's `single_arm_a1_passthrough` mode (in `experiments/train_substrate_stack_of_stacks.py::main`). This mode:

1. Builds a deterministic residual payload (no learning)
2. Composes A1 archive + residual into stack-of-stacks archive grammar
3. Returns immediately (no SGLD polish loop)

The `STACK_OF_STACKS_LANGEVIN_T_INIT_CAP` env var declared in the recipe IS INERT in this mode — the trainer never reaches the SGLD polish epochs path.

### Finding 2: Auth-eval temp_work_dir refusal

Per Catalog #204 `check_pr95plus_modal_smoke_uses_durable_provider_output`, `experiments/contest_auth_eval.py` refuses temp_work_dir for score-grade JSON. The driver writes auth_eval JSON to `/tmp/pact/...` which is correctly refused. Recipe needs either `--allow-temp-work-dir` (diagnostic-only) or durable provider output redirect.

## Per CLAUDE.md "Forbidden premature KILL" verdict

This is a DEFER not a KILL. The Welling-Teh 2011 SGLD paradigm is canonical; what is falsified here is the SPECIFIC IMPLEMENTATION that the operator's PREP subagent set up (single-arm A1 passthrough mode is wired but SGLD polish loop is not exposed via this recipe's entry).

Per Catalog #307 paradigm-vs-implementation classification: this is an IMPLEMENTATION-LEVEL falsification (specific trainer scope incomplete), NOT a PARADIGM-LEVEL falsification (Welling-Teh SGLD intact).

## Reactivation criteria

1. **Trainer scope re-architecture** (op-routable #1): expose a dedicated `--sgld-convergence-diagnostic` mode in `experiments/train_substrate_stack_of_stacks.py` that actually runs SGLD polish epochs at the recipe-reachable entry. Estimated 50-100 LOC trainer edit; should be in a sister substrate or a new dedicated SGLD-only trainer per UNIQUE-AND-COMPLETE-PER-METHOD operating mode.
2. **Output dir durable redirect** (op-routable #2): recipe sets `STACK_OF_STACKS_OUTPUT_DIR` to a durable provider work_dir under `/modal_results/` per Catalog #204, OR recipe adds `--allow-temp-work-dir` flag for the diagnostic-only branch.
3. **STACK_OF_STACKS_LANGEVIN_T_INIT_CAP sweep activation** (op-routable #3): after op-routable #1 lands, the env var becomes active and the t_init-CAP ∈ {0.5, 1.0, 2.0, 5.0, 10.0, 17.4} sweep can fire.
4. **Probe outcome supersession** (op-routable #4): after op-routable #1-3 land, register a fresh PROCEED probe outcome row per Catalog #313 sister-probe-alternative-reducer (Catalog #308) ratifying the next dispatch attempt.

## Per-substrate symposium continuity

The Cable B1 operator-frontier-override per Catalog #300 ratified the symposium DRAFT verdict to PROCEED. This DEFER outcome is a per-dispatch infrastructure verdict, NOT a symposium-level deferral. The symposium ratification remains valid; per Catalog #325 14-day window the next dispatch attempt after the op-routables land is still admissible without re-convocation.

## Continual-learning anchor

Per CLAUDE.md "Council hierarchy: 4-tier protocol" + Catalog #300 v2 frontmatter, this T1 working-group anchor will be persisted via `tac.council_continual_learning.append_council_anchor` so the downstream autopilot ranker / Rashomon ensemble / next-iteration council see this verdict.

## Cross-references

- Operator-frontier-override memo: `.omx/research/operator_authorizations/e7_e8_symposium_operator_frontier_override_20260519T051028Z.md`
- E.7 sister verdict: `.omx/research/e7_vq_k_sweep_dispatch_verdict_20260519T060000Z.md` (sister)
- DEFER probe outcome row: `sgld_convergence_dispatch_trainer_only_single_arm_passthrough_not_real_sgld_DEFER_20260519` in `.omx/state/probe_outcomes.jsonl`
- Pending probe outcome supersession: `sgld_t_final_convergence_diagnostic_ratified_20260519` (still standing PROCEED but conditioned on trainer scope-fix per op-routable #1)
- Catalog #204 sister anchor: PR95++ Modal smoke durable output (~`feedback_pr95plus_modal_smoke_durable_output_landed_*.md`)
- Catalog #218 sister anchor: D4 OOM fix mini-batch (the bug class this dispatch encountered at the sister memory-pressure surface)
- HNeRV parity L1 lesson 2: export-first design (the trainer must EXPORT SGLD convergence curves at the recipe-reachable entry)

## Mission-alignment classification

**`predicted_mission_contribution: rigor_overhead`** — this DEFER does NOT directly lower contest score; its mission contribution is the structural validation that the SGLD trainer's recipe-reachable entry is currently a single-arm A1 passthrough, NOT a multi-K SGLD convergence diagnostic. The DEFER + op-routables route the operator to the next-most-actionable work (trainer scope-fix) without burning paid GPU on the un-instrumented current code path.

— Cable B1 E.7+E.8 combined dispatch subagent
2026-05-19T06:00:00Z


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
