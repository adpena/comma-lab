# Harvest: Fable 5 prompting guide + Loops blog (+ linked pages) — 2026-07-07

Operator-directed harvest, companion to `docs/operating_manual_craft_handoff.md`. Sources:
platform.claude.com prompting-claude-fable-5 · introducing-claude-fable-5-and-mythos-5 ·
claude.com/blog/getting-started-with-loops · claude-prompting-best-practices (55KB full text
persisted at `~/.claude/projects/-Users-adpena-Projects-pact/89ff112f-*/tool-results/toolu_01GechvTSq7VkZX42AXU2NCv.txt`
— re-fetchable from the URL; only the Fable-5 section was context-loaded).

## ⚑ FINDING: the intro page states "Access to Claude Fable 5 and Claude Mythos 5 has been
RESTORED" (links anthropic.com/news/redeploying-fable-5-mythos-5). CHECK against the account —
may supersede "access narrows tomorrow." Mythos 5 = same capabilities WITHOUT safety
classifiers, limited release via Project Glasswing. Fable 5 API: `claude-fable-5`, 1M context,
128k output, $10/$50 per Mtok, adaptive-thinking-only (no disable, effort controls depth), raw
CoT never returned (`thinking.display: summarized|omitted`), `stop_reason:"refusal"` +
server/client fallback to Opus 4.8 + fallback credit (no double cache cost; refused-before-
output = unbilled). Supports: effort, task budgets (beta), memory tool, code exec, programmatic
tool calling, context editing, compaction. 30-day retention, no ZDR.

## Fable-5 prompting patterns (verbatim-grade, for OUR subagent prompts + future Fable access)

1. **Assign the hardest problems** — testing on simple workloads undersells it; it scopes, asks,
   executes end-to-end (hours-to-weeks tasks). Start at the top of the difficulty range.
2. **Longer turns by default** — restructure harnesses to check runs ASYNC (scheduled jobs, not
   blocking). Anti-overplanning prompt: "When you have enough information to act, act. Do not
   re-derive facts already established… give a recommendation, not an exhaustive survey."
3. **Effort is THE control** — high default; xhigh for capability-critical; low/medium beats
   prior models' xhigh on routine work. Anti-goldplating prompt: "Don't add features, refactor,
   or introduce abstractions beyond what the task requires… simplest thing that works well…
   only validate at system boundaries."
4. **Brief steering beats enumeration** — one brevity instruction replaces listing behaviors:
   "Lead with the outcome… drop details that don't change what the reader would do next, not
   compress into fragments/arrow-chains." Checkpoints: "Pause only when work genuinely requires
   the user: destructive/irreversible action, real scope change, or input only they can provide."
5. **Ground progress claims (kills fabricated status)**: "Before reporting progress, audit each
   claim against a tool result from this session. Only report work you can point to evidence
   for; if not yet verified, say so explicitly." (= manual §4/§5; Anthropic: "nearly eliminated
   fabricated status reports even on tasks designed to elicit them.")
6. **State boundaries** — "When the user is describing a problem… the deliverable is your
   assessment. Report findings and stop. Don't apply a fix until they ask." Guards unrequested
   actions (defensive branches, unasked emails).
7. **Parallel subagents** — dispatches more readily; prefer async orchestrator↔subagent comms;
   LONG-LIVED subagents that keep context across subtasks save cost via cache + avoid
   slowest-subagent bottleneck. "Delegate independent subtasks and keep working while they run."
8. **Memory system** — "one lesson per file, one-line summary at top; record corrections AND
   confirmed approaches with why; don't save what repo/history records; update-don't-duplicate;
   delete wrong notes." Bootstrap: reflect over past sessions with subagents. (= our MEMORY.md
   discipline, independently converged.)
9. **Early-stop mitigation** — deep in long sessions may state intent without the tool call;
   autonomous-pipeline reminder: "Before ending your turn, check your last paragraph. If it is
   a plan/question/promise about undone work, do that work now with tool calls."
10. **Context-budget anxiety** — avoid showing remaining-token countdowns; if shown: "You have
    ample context remaining. Do not stop, summarize, or suggest a new session."
11. **Give the reason, not only the request** — "I'm working on [larger task] for [who]. They
    need [what it enables]. With that in mind: [request]." (= manual §1 from the asker's side.)
12. **Final-message readability** — working shorthand is fine between tool calls; the final
    summary is a RE-GROUNDING for a reader who saw none of it: outcome first, plain language,
    no arrow chains/invented labels, each identifier its own clause.
13. **send_to_user tool** — client tool to surface verbatim content mid-turn (inputs never
    summarized); MUST pair with elicitation prompt or it goes uncalled; never for narration.
14. **Scaffolding**: fresh-context VERIFIER SUBAGENTS beat self-critique ("Establish a method
    for checking your own work at interval [X]… verify with subagents against the spec");
    old prescriptive skills DEGRADE Fable-5 output — prune them; NEVER instruct reasoning
    echo/transcription (triggers reasoning_extraction refusals → fallback storms) — read
    thinking blocks instead; safety classifiers cover offensive-cyber + bio + reasoning
    extraction (benign work can trip them → configure fallback).

## Loops (claude.com blog) — taxonomy + practices

- **Turn-based** (prompt → Claude judges done): short non-recurring work.
- **Goal-based `/goal`**: "…stop after N tries" — verifiable exit criteria + explicit turn caps
  (e.g. "/goal get Lighthouse ≥90, stop after 5 tries").
- **Time-based `/loop [interval]` (local) + `/schedule` (cloud)**: recurring/monitoring (e.g.
  "/loop 5m check my PR, address review comments, fix failing CI"). Prefer longer intervals;
  react to EVENTS not time.
- **Proactive (research preview)**: /schedule × /goal × skills × dynamic workflows × auto mode —
  e.g. hourly bug-triage that doesn't stop until every found report is triaged/actioned.
  Route routine to smaller models; capable models for judgment.
- **Verification-as-SKILL.md**: encode manual checks as reusable skills (dev server → interact →
  screenshot before/after → console → perf trace → rerun on failure, "don't hand back partial
  work"). = our gates philosophy at the skill layer.
- **Cost**: deterministic stop criteria >> "good enough"; scripts for deterministic work instead
  of reasoning; pilot workflows on small slices; /usage /goal /workflows for tracking.
- **Caution**: "start with the simplest solution; use these patterns selectively."

## Mapping onto OUR apparatus (what to adopt)

1. **Subagent prompt template upgrade**: fold patterns 5 (evidence-audited progress), 9 (no
   ending on promises), 12 (final-message re-grounding), 14 (fresh-context verifiers) into the
   standard subagent contract (#337 build-wave already cites the manual; add these verbatim
   blocks). The grounding prompt (#5) is the single highest-value adoption — it is the
   prompt-level enforcement of manual §4/§5.
2. **#205-style check-ins → the loops taxonomy**: our autonomous check-in protocol is a
   time-based loop; the blog's "react to events rather than time" endorses moving to
   event-triggered wakeups (verdict-row appearance) over fixed intervals where possible.
3. **Skill-encoded verification**: our verify steps (ruff F821 + review-gate + suite + parse
   test) could be one SKILL.md the successor invokes — candidate for the build-wave.
4. **If Fable 5 access is restored**: prune over-prescriptive instructions when running it
   (pattern 14 — old skills degrade it); keep CLAUDE.md non-negotiables (boundaries/NO-FAKE are
   exactly the "state the boundaries" class the guide endorses).
5. **Never** add reasoning-echo instructions to any prompt/skill (refusal-storm risk on Fable).

Owed (not fetched, links recorded): migration-guide (Opus 4.8→Fable 5), effort docs, adaptive-
thinking, refusals-and-fallback, fallback-credit, task-budgets, the fallback/billing cookbook,
anthropic.com/news/redeploying-fable-5-mythos-5, + the persisted 55KB best-practices full text.

## Addendum (2026-07-07): "A field guide to Claude Fable 5: Finding your unknowns" (claude.com blog)

Framework: map-vs-territory (prompts/skills = map; codebase = territory) + the four unknowns
quadrants. Patterns: blind-spot pass · brainstorms/prototypes · interviews · references ·
implementation plans · implementation notes · pitches/explainers · quizzes. Principle: over-
specification sends Claude down unsuitable paths; under-specification breeds misaligned
assumptions — accounting for unknowns navigates both.

CONVERGED (our existing surface): references-over-descriptions = vendor-the-real-source /
read-the-real-argparse (never-invent-flags); implementation-plans-reviewed = council design
authority + symposium gates; implementation notes = the DAG FEED discipline (deviations +
reasoning, exactly); interviews = AskUserQuestion GO flows + the council draft's 13 open
questions; quizzes ≈ fresh-context verifier subagents; map-vs-territory = EXECUTE_DONT_READ.

ADOPTED INSIGHT — **unknown-knowns are our incident class**: "information so implicit it goes
undocumented, yet recognizable when encountered" is precisely what produced the log-lives-in-
.omx/tmp split (launcher knew; nothing documented it; froze telemetry 12h + forced the
dashboard's hardcoded constants), the comma10k class-order luma-sort trap, and the
--logit-adjust-tau flag collision. The failure ledger captures these POST-hoc; the field guide's
delta is PRE-hoc surfacing. PRACTICE (dispatcher-level, no new gate): when dispatching into an
UNFAMILIAR surface, include a blind-spot ask in the prompt — "before building, list the
unknown-knowns of this surface: implicit conventions, path/layout facts, naming traps a
newcomer would trip on; check them against the failure ledger first." Cheap, targets our
measured top incident class. Not added as a contract constant (gate-locked module; marginal
value below the review cost) — documented here as standing dispatcher practice.
