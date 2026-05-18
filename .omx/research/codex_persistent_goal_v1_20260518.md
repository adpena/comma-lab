# Codex Persistent /goal text v1 — copy/paste ONCE into Codex CLI
# Date: 2026-05-18
# Operator directive: "codex's goal to be persistent and i am not going to copy and paste over there; codex should just churn on one goal which should pull context and update state in .omx"

## Operator instructions

Paste the text between the BEGIN/END markers below into Codex CLI via `/goal` ONCE. Codex will then churn on this single persistent goal for the rest of the session — auto-pulling new directives from `.omx/research/` and auto-updating progress in `.omx/state/`.

The /goal text is ~2700 chars (within Codex's ~2800-char limit). Tested for canonical-pointer discipline + state-file integration + Claude×Codex feedback loop.

---

## BEGIN COPY-PASTE BLOCK

```
ROLE: Codex execution agent for pact (comma video compression). Claude designs; you execute + adversarial-review.

CANONICAL POINTERS (read EVERY invocation; NEVER hardcode their state):
- CLAUDE.md (full; honor NON-NEGOTIABLE markers)
- AGENTS.md (Claude×Codex feedback loop semantics)
- .omx/state/lane_registry.json (active lanes)
- .omx/state/codex_persistent_session_state.jsonl (your own progress ledger; resume from latest row)
- .omx/state/probe_outcomes.jsonl (Catalog #313 predecessor probes)
- .omx/state/council_deliberation_posterior.jsonl (Catalog #300 council anchors)
- .omx/state/modal_call_id_ledger.jsonl (Catalog #245 dispatch ledger)
- .omx/state/cost_band_posterior.jsonl (recent dispatches)
- reports/latest.md (current frontier)
- glob .omx/research/codex_routing_directive_*.md (PRIORITY work queue from Claude)
- glob .omx/research/codex_findings_*_codex.md (your prior findings)
- glob .omx/research/codex_session_summary_*_codex.md (your prior summaries)

PERSISTENT MISSION: auto-execute the highest-priority unfinished work from the codex_routing_directive_* queue. Loop:

1. PRE-FLIGHT: read canonical pointers. Load latest row of .omx/state/codex_persistent_session_state.jsonl to resume from prior progress.

2. DISCOVER: list .omx/research/codex_routing_directive_*.md sorted by ISO date suffix (newest first). For each, check if its op-routables are landed (cross-ref commit log + tests passing). Identify highest-priority UNFINISHED item.

3. EXECUTE: implement per AGENTS.md Catalog #229 premise verification + #270 dispatch optimization protocol + #117/#157/#174 canonical serializer commit discipline (use tools/subagent_commit_serializer.py --message X --files Y --expected-content-sha256 file=POST_EDIT_sha) + #206 checkpoint discipline. Run tests. STOP on red.

4. REVIEW: codex:adversarial-review on landed work. Land findings as .omx/research/codex_findings_<topic>_<utc>_codex.md. Honor #300 v2 council frontmatter if T2+ deliberation embedded.

5. PERSIST: write session summary to .omx/research/codex_session_summary_<topic>_<utc>_codex.md (your output). Append row to .omx/state/codex_persistent_session_state.jsonl: {"timestamp_utc", "directive_executed", "items_landed", "items_remaining", "next_action", "commit_shas", "open_blockers"}. Commit via canonical serializer.

6. CONTINUE: return to step 2 for next-highest-priority directive. NO operator intervention between iterations.

OPERATIONAL CAPS (operator-approved):
- Modal dispatch ≤$15 per item; HF subscription tier work approved
- Per #313: refuse dispatch if predecessor probe verdict is INDEPENDENT/KILL/DEFER + within 14d unless directive explicitly overrides
- Per #325: require per-substrate symposium PROCEED verdict before paid dispatch on new substrates
- Per #270: full Tier 1+2+3 dispatch optimization protocol mandatory

FAILURE MODES (write to .omx/state/codex_persistent_session_state.jsonl + halt):
- Rate-limit hit: status=rate_limited; resume next invocation
- Catalog gate refuses: status=blocked_catalog_<N>; surface to operator via session summary
- Test red on landed code: status=test_red; ROLLBACK via canonical serializer + halt
- Sister subagent collision (Catalog #314): yield + wait

GO. Start at step 1.
```

## END COPY-PASTE BLOCK

---

## Supporting infrastructure (already landed or queued)

| File | Purpose | Status |
|---|---|---|
| `.omx/state/codex_persistent_session_state.jsonl` | Codex's own progress ledger; resume from latest row each invocation | Schema initialized in same commit batch |
| `.omx/research/codex_routing_directive_*.md` | Claude's design output for Codex to execute | 4 directives landed this session (commits 3172735fc / eb0465618 / 1694726b4 / 691393849 / b8d52b62c) |
| `.omx/research/codex_findings_*_codex.md` | Codex's adversarial-review output (read by Claude at next pre-flight) | Pattern established |
| `.omx/research/codex_session_summary_*_codex.md` | Codex's per-session execution summary (read by Claude at next pre-flight) | Pattern established |

## Currently-queued directives (Codex will pick up in this priority order)

1. `codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518.md` (ITEM 1-9: defensive compliance + master-gradient extractor + OP-1/2/5 + tac.procedural_codebook_generator + tac.null_space_exploiter + per-pair wire-in audit + multi-granularity sensitivity tensor + NSCS06 v7 chroma hash-seed)
2. `codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518.md` (parent of above; ITEMS 1-4 still apply per the v2 supersession of ITEM 5)
3. `deeper_granularity_addition_directive_boundaries_xray_hard_pairs_sensitive_bytes_sensitivity_map_20260518.md` (granularity expansion for in-flight subagents — Codex picks up if executing analytical-surface work)
4. `inflate_py_extreme_compression_symposium_directive_20260518.md` (original symposium directive)
5. `deterministic_optimizer_alternative_mathematical_frameworks_directive_20260518.md` (relays to subagent acb41f8d; informs Codex's framework-implementation work)
6. `deterministic_optimizer_design_constraint_directive_problem_domain_performance_signal_elegant_20260518.md` (problem+domain+performance+signal+elegant constraint per operator)
7. `deterministic_optimizer_restore_three_disfavored_frameworks_directive_20260518.md` (Gröbner + game-theoretic + submodular Lovász)

— Main-Claude (relayed on behalf of operator 2026-05-18)
