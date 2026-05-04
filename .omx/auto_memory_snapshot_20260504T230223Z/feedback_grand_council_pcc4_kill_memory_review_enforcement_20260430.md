---
name: Grand Council deliberation — PCC4 KILL/FALSIFIED memory file enforcement (CLAUDE.md non-negotiable)
description: 2026-04-30 ~23:30 UTC. Inner-10 council deliberation on the PCC4 STRICT preflight check that enforces Grand Council adversarial review on every KILL / FALSIFIED memory file. Triggered by the Lane 17 IMP premature-KILL incident where the agent recorded a KILL verdict at ~22:50 UTC based on a 3.47-second "200-epoch" stub-loop measurement bug. The user's adversarial challenge ("was the IMP results reliable and is that verdict actually hold up acording to etreme adversarail grand councill") caught the premature kill before it became durable folklore. CLAUDE.md non-negotiable now mandates that every KILL memory file contain a council adversarial review section; PCC4 enforces it.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Context

User mandate 2026-04-30 ~22:55 UTC:
> "permanently fix all bugs and bug classes and metabugs and everything and have all design decisions and ultimate experiment subject to extreme paranoia and adversarial grand council reviews"

The Lane 17 IMP incident exposed a class of failure: an agent records a KILL verdict, the verdict gets indexed in MEMORY.md, and downstream sessions inherit it as fact. There is no enforcement that the KILL was reviewed against the adversarial inner council. PCC4 closes that gap.

## What PCC4 enforces

`check_kill_memory_files_have_council_review()` scans `~/.claude/projects/-Users-adpena-Projects-pact/memory/` for `.md` files matching:

- filename glob `project_*killed*.md` / `project_*falsified*.md` / `project_*retired*.md` (case-insensitive)
- body literal `VERDICT: KILL`, `FALSIFIED`, or `RETIRED` (case-sensitive)

For each matched file, the check requires 3 sections:

1. **Grand Council adversarial review** — header `## Grand Council` (or one of the accepted aliases) + at least 5 named members from the inner-10 list with rationale on a non-empty line.
2. **Internal-consistency check** — header `## Internal-consistency` (or alias) + at least 1 enumerated check (bullet or numbered item).
3. **Reactivation criteria** — header `## What would change my mind` / `## Reactivation criteria` / `## Conditions for retracting` + at least 1 enumerated condition.

Auto-pass conditions (intentional):
- `COUNCIL_REVIEW_SKIPPED_USER_OVERRIDE: <reason>` on its own line.
- Title contains `WITHDRAWN` (a kill REVERSED under adversarial scrutiny IS the success outcome of this check).
- Filename timestamp suffix < `20260430` (legacy grandfather; the protocol was established 2026-04-30).

## Council deliberation outcomes

### DD1: Should the check scan the GLOBAL memory dir or only repo-local memory?

| Member | Vote | Rationale |
|---|---|---|
| Shannon (LEAD) | GLOBAL | "The kill record IS information; entropy of the decision lives in the global durable store, not repo-local. Scanning only repo-local would let kills re-enter via cross-session memory inheritance." |
| Dykstra (CO-LEAD) | GLOBAL | "The convex feasibility set of 'kill records reviewable at preflight time' must include the global dir; otherwise the check has a hole." |
| Yousfi | GLOBAL | "Contest-design instinct: the constraint must close the loophole. Repo-only scan is the loophole." |
| Fridrich | GLOBAL | "Steganalysis lesson: any unreviewed channel becomes a covert channel. The agent's memory IS a channel." |
| Contrarian | GLOBAL | "I challenge the bold proposal but cannot defeat the GLOBAL argument. Repo-local would make the rule trivially bypassable." |
| Quantizr | GLOBAL | "Competitive intelligence is in the global memory; we can't have an unreviewed kill there. Pragma: the test fixture lives there too." |
| Hotz | GLOBAL | "30-second analysis: global is right. Anything else is theater." |
| Selfcomp | GLOBAL | "Empirical: my own kill records (PR #56 deferred decisions) live in the global dir and need the same gate." |
| MacKay (memorial) | GLOBAL | "MDL framing: the global memory IS the operator's compressed model of which lanes are dead. An unreviewed compression is invalid." |
| Ballé | GLOBAL | "Hyperprior framing: the side-information (what we 'know' is dead) must be encoded with the same fidelity guarantees as the primary stream." |

**Vote: 10/10 GLOBAL.** Implementation: default `memory_dir=None` resolves to `Path.home() / ".claude" / "projects" / "-Users-adpena-Projects-pact" / "memory"`. Tests override via `memory_dir=` parameter.

### DD2: What is the canonical name of the "what would change my mind" subsection?

| Member | Vote | Preferred name |
|---|---|---|
| Shannon | accept-multiple | "## Reactivation criteria" — terse, info-theoretic. |
| Dykstra | accept-multiple | "## Conditions for retracting" — projection language. |
| Yousfi | accept-multiple | "## What would change my mind" — Tetlock superforecaster framing; most cognitively useful. |
| Fridrich | accept-multiple | Indifferent; any of the three. |
| Contrarian | accept-multiple | The variants are functionally equivalent. Forcing one canonical name would be theater. |
| Quantizr | accept-multiple | "Reactivation criteria" wins on brevity. |
| Hotz | accept-multiple | "What would change my mind" — humanly readable, no jargon. |
| Selfcomp | accept-multiple | All three; let authors choose. |
| MacKay | accept-multiple | "Conditions for retracting" — Bayesian evidence framing. |
| Ballé | accept-multiple | Indifferent. |

**Vote: 10/10 ACCEPT MULTIPLE.** All three canonical headers (plus several aliases) accepted. The check enumerates an accept-list (`_PCC4_REACTIVATION_HEADERS`) and any one match satisfies the gate. Each variant must be followed by an enumerated list (bullet or numbered).

### DD3: Should `FALSIFIED` alone trigger the check, or only when paired with KILL semantics?

| Member | Vote | Rationale |
|---|---|---|
| Shannon | TRIGGER | "FALSIFIED is an explicit verdict. The information content is identical to KILL." |
| Dykstra | TRIGGER | "Convex feasibility: the failure region of 'unreviewed kills' includes FALSIFIED. Excluding it leaves a leak." |
| Yousfi | TRIGGER | "Steganalysis FALSIFIED is the strongest possible verdict. Must review." |
| Fridrich | TRIGGER | "Same as Yousfi." |
| Contrarian | CHALLENGE-FAILED | "I argued for KILL+FALSIFIED-paired-only, but the precedent (Lane STC clean-source FALSIFIED, Lane MM v2 falsified) shows FALSIFIED alone is used as a final verdict. My challenge fails." |
| Quantizr | TRIGGER | "Leaderboard interpretation: FALSIFIED on a competitor's claim is a strong narrative move; we must back it with council review." |
| Hotz | TRIGGER | "If you write FALSIFIED, you better mean it. Mean it = council reviewed it." |
| Selfcomp | TRIGGER | "FALSIFIED in my own work (PR #56 several deferred sub-claims) needs the same gate." |
| MacKay | TRIGGER | "MDL: FALSIFIED is a final code-length-zero assignment. Cannot be unreviewed." |
| Ballé | TRIGGER | "Side-information must be reviewed at the same rigor as primary." |

**Vote: 10/10 TRIGGER.** `FALSIFIED` is in `_PCC4_KILL_BODY_LITERALS`. Same applies to `RETIRED`. **`DEAD` is INTENTIONALLY EXCLUDED** because it matches incidental usage (`dead-flag bug`, `dead resolver`, `dead code`) — too noisy. Files using `DEAD` as their verdict literal must use the filename glob convention (`project_*dead*.md` not currently supported; if needed, authors should use `project_*killed*.md`).

### DD4: Should we land STRICT or WARN-ONLY initially?

Live audit found 4 violations on 2026-04-30 kill records:
- `project_all_scores_forensic_audit_20260430.md` (forensic audit, references kills but isn't itself a kill — debatable scope)
- `project_lane_7_psd_killed_or_deferred_20260430.md` (real kill, has council vote 10/10 in description but no `## Grand Council` header)
- `project_lane_gp_class_forensic_audit_20260430.md` (forensic audit; same scope question)
- `project_lane_gp_v4_killed_basis_fit_infeasible_20260430.md` (real kill, has 4-round adversarial review per description but no canonical headers)

| Member | Vote | Rationale |
|---|---|---|
| Shannon | WARN | "Lane A pattern: warn-only first, backfill, then STRICT." |
| Dykstra | WARN | "Convex feasibility says: don't introduce a hard constraint that violates 4 existing valid records before backfill." |
| Yousfi | WARN | "Standard contest-rule rollout: announce, give a backfill window, then enforce." |
| Fridrich | WARN | Same. |
| Contrarian | STRICT | "I challenge — the 4 violators DO need the canonical sections. Forcing STRICT now drives the backfill." |
| Quantizr | WARN | "Pragmatic: wire-in shouldn't break the next 5 commits while backfill happens in parallel." |
| Hotz | WARN-WITH-DEADLINE | "WARN now, STRICT in 24h. Don't let backfill drift." |
| Selfcomp | WARN | Same. |
| MacKay | WARN | "MDL: don't impose code-length-infinity on partially-compliant records." |
| Ballé | WARN | Same. |

**Vote: 9 WARN / 1 STRICT.** Lands WARN-ONLY initially. **Backfill the 4 violators in a follow-up commit, then promote to STRICT**, per the established Lane A pattern (commits 1-11 in `preflight_all()`).

## Implementation

- New function `check_kill_memory_files_have_council_review()` in `src/tac/preflight.py` (~330 LOC including helpers + module constants).
- Wired into `preflight_all()` STRICT=False initially with detailed inline rationale referencing this memory file + the Lane 17 IMP fixture.
- 22 regression tests in `src/tac/tests/test_check_pcc4_kill_memory_council_review.py` covering: filename-glob detection, body-literal detection, all 3 missing-section paths, too-few-members detection, the 3 auto-pass conditions (override / WITHDRAWN / grandfather), strict-mode raising, edge cases (no dir, empty dir, MEMORY.md skip, non-md skip), the real Lane 17 IMP fixture, and 3 council-decision invariants.

## Success criteria

- Lane 17 IMP fixture (`project_lane_17_imp_killed_cycle_0_198_regression_20260430.md`) PASSES via WITHDRAWN-in-title rule. ✓ verified
- All 22 unit tests pass. ✓ verified
- Live-codebase audit returns 4 violations (the 2026-04-30 kill records that pre-date this protocol). Will go to 0 after backfill.
- Promotion to STRICT happens in a follow-up commit after backfill.

## Cross-refs

- `project_lane_17_imp_killed_cycle_0_198_regression_20260430.md` — the originating fixture.
- `feedback_grand_council_imp_permanent_fix_review_20260430.md` — DD3 protocol established.
- CLAUDE.md "Council conduct" non-negotiable.
- `src/tac/preflight.py` — `check_kill_memory_files_have_council_review`.
- `src/tac/tests/test_check_pcc4_kill_memory_council_review.py` — 22 tests.
