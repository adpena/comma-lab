# Operator decisions executed (2026-05-09)

<!-- generated_at: 2026-05-09T06:05:00Z, from_state_hash: operator_approved_all -->

## All 11 operator decisions APPROVED 2026-05-09

Operator response: "all operator decisions required are approved as to your recommendations, proceed with all of the next steps and roadmap and outstanding tasks and remaining work and everything"

## Status of each decision

| # | Decision | Status |
|---|---|---|
| 1 | DEFER T9 OR re-scope to single-axis A1 branching | **APPROVED → DEFER T9; routing directive to a0be36e** |
| 2 | 2-3 day pre-design pass per track for T15/T17/T18 BEFORE GPU spend | **APPROVED → no GPU dispatch on T15/T17/T18 until pre-design lands** |
| 3 | $20 CPU + $40 CUDA budget to re-scope Lane 12 NeRV → "Lane 12-v2 NeRV-as-renderer" | **APPROVED → spawning re-scoping subagent** |
| 4 | STRICT preflight check #124 warn-only initially | **APPROVED → spawning impl subagent** |
| 5 | Top-tier insertion of T11+T13+T19 alongside T7 | **APPROVED → already at scaffold; integrating into Phase 1 trainer** |
| 6 | Paste CLAUDE.md addition (HNeRV parity discipline) | **DONE 2026-05-09T06:05Z, inserted at line 63 between Race-mode and Main-branch sections** |
| 7 | $0 probe-sweep T7/T8/T11 sub-additivity disambiguator | **APPROVED → spawning subagent** |
| 8 | T13 √n latent shrink as Phase 1 component | **APPROVED → integrating into trainer (subagent)** |
| 9 | T19 adaptive-ρ ADMM lands BEFORE Phase 1 dispatch | **APPROVED → integrating into trainer (subagent)** |
| 10 | T20 + T22 added NOW ($0 each) | **APPROVED → spawning impl subagent** |
| 11 | Hotz race-mode dissent: check leaderboard | **DONE 2026-05-09T06:00Z — last leaderboard activity May 4 (5 days ago); NO race window active** |

## Concrete in-flight directives (this turn)

### DEFER T9 → a0be36e
The in-flight T8+T9 implementation subagent (a0be36e) MUST halt T9 work or reframe it to single-axis A1 branching. Operator approved DEFER per kitchen_sink anti-pattern. Continue T8 W₂ Sinkhorn work but apply the codex HIGH 3 fix already in `src/tac/losses.py`.

### Spawning 5 parallel subagents
- **A**: $0 probe-sweep T7/T8/T11 sub-additivity disambiguator
- **B**: T20 (KL pose-axis loss) + T22 (temporal-consistency regularizer) implementations
- **C**: Lane 12-v2 NeRV-as-renderer re-scoping ($20 CPU + $40 CUDA budget)
- **D**: T13 + T19 integration into Phase 1 trainer
- **E**: STRICT preflight check #124 implementation (warn-only)

## Cross-references

- HNeRV retrospective memo: `~/.claude/projects/.../feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`
- Coherence council memo: `~/.claude/projects/.../feedback_grand_council_portfolio_coherence_journal_grade_20260509.md`
- Phase 2 launch memo: `~/.claude/projects/.../feedback_paradigm_dezeta_phase2_architectural_launch_20260509.md`
- T11+T13+T19 landing: `~/.claude/projects/.../feedback_t11_t13_t19_free_lateral_leaps_landed_20260509.md`
- Codex review fixes: `.omx/research/codex_adversarial_review_findings_for_inflight_subagents_20260509.md`
- HNeRV forensics directive: `.omx/research/hnerv_forensics_author_repo_search_directive_20260509.md`
