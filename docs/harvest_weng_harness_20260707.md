# Harvest: Lilian Weng — "Harness Engineering for Self-Improvement" (2026-07-04) — 2026-07-07

Operator-directed harvest, format sibling of `docs/harvest_fable5_prompting_and_loops_20260707.md`.
Source: https://lilianweng.github.io/posts/2026-07-04-harness/ (fetched 2026-07-07). Definition
(verbatim-grade): a **harness** is "the system surrounding a base model that orchestrates execution
and decides how the model thinks and plans, calls tools and acts, perceives and manages context,
stores artifacts, and evaluates results." Post structure: Harness Design Patterns (workflow
automation · file-system-as-persistent-memory · sub-agents/backend jobs · coding-agent case study ·
harness-vs-core-intelligence) → Harness Optimization (context engineering/ACE · workflow design/
AFlow · Self-Harness · evolutionary search/AlphaEvolve · joint optimization/SIA · STOP) → Future
Challenges (7 bottlenecks) → benchmark appendix.

**Verification note:** the parent agent's mapping was checked against the REAL codebase surfaces
below (each named surface was opened/grepped, not recalled). Where our apparatus already had the
pattern, the convergence is independent (our surfaces predate the post).

## (a) CONVERGED — we already have it (post pattern ↔ OUR surface)

1. **File system as persistent memory** ("keep durable state in files"; the agent "can recover
   after interruptions and reason over its own execution history") ↔ the `.omx/` spine: the DAG
   (`.omx/research/sub015_DAG_*.md` FEED blocks = inspectable execution history), `.omx/state/*`
   fcntl-locked JSONL ledgers, `MEMORY.md` + memory files, and the Ralph execution model
   ("Treat files and git as memory… each iteration resumable from disk").
2. **Explicit, inspectable parallelism** ("If subagent outputs only live in a transient chat
   context, they quickly become obsolete and hidden") ↔ wave dispatches with durable logs,
   `subagent_checkpoint.py` / `.omx/state/subagent_progress.jsonl` crash-resume protocol, the
   commit serializer's forensic JSONL, and the marker-on-exit / harvest-or-lose disciplines.
3. **Self-Harness weakness mining** (cluster failures into "verifier-grounded failure patterns";
   "rich failure records" separating terminal cause from causal mechanism) ↔ the confound-hunt
   cadence (`confound_hunt_synthesis_20260705.md`, 18 confounds, 3-layer immune system) + the
   recursive adversarial review protocol + the meta-bug class catalog (each catalog row IS a mined
   weakness with a landed narrow fix). NEW as of this harvest: the failure LEDGER (§b) makes the
   records queryable/countable rather than prose-only.
4. **ACE Generator/Reflector/Curator** (incremental itemized bullets "merged into a structured
   context logbook with deterministic logic" to "prevent context collapse and brevity bias") ↔
   our run → review → memory-curation loop: one-line hooks in MEMORY.md with detail in topic
   files, the **<17KB index budget** (our anti-context-collapse mechanism — a fat index
   partial-loads, which is exactly ACE's collapse failure), append-only supersession instead of
   full rewrites (Catalog #110/#113).
5. **Reward-hacking defense** ("The evaluator and permission control should likely sit outside
   the loop that evolves harness, with held-out tests, trace audits, and human review") ↔ frozen
   `upstream/evaluate.py` as the ONLY score authority (pinned-upstream non-negotiable: never
   edit/monkeypatch), operator-GO gates on actuation/paid dispatch, and the NO-FAKE supreme rule
   (class #8: surrogate-optimized-but-not-exact-authority-verified IS reward hacking named).
6. **Negative-results preservation** ("make failed attempts easy to preserve, as learning from
   failure is the best way to trim down the task search space") ↔ NO-GO rows WITH mechanisms
   (e.g. FEED-07k: contour coder NO-GO 0.820>0.65 B/flip + the measured WHY), the papers-checked
   ledger (`reference_papers_checked_not_relevant_or_watch_item_ledger`), KILL-verdict structural
   requirements (reactivation criteria mandatory), and the canonical anti-patterns registry.
7. **Diversity-collapse defense** ("Evolutionary and RL loops tend to exploit known high-reward
   patterns"; the best path "may initially look worse under the current evaluator") ↔ the
   activation ledger's duty-to-measure: "off" is a tracked queue; never-fired levers are ranked
   into the DECIDE queue (`rank_duty_to_measure`, #247 EIG-bridge) instead of starving under
   exploit-only ranking. §c invariant (ii) makes this link binding.
8. **STOP's "base model must be capable enough"** ("Recursive structure alone is not enough…
   harness improvement enables better deployment of the model but intelligence is still the
   core") ↔ the operating manual's purpose: `docs/operating_manual_craft_handoff.md` exists
   precisely to hand craft to successor sessions because the harness cannot substitute for the
   operator-model partnership (memory: `project_how_the_breakthrough_happened_partnership_over_
   autonomy_20260703` — set-and-forget autonomy FAILED; leaning into the relationship worked).

## (b) ADOPT-NOW (landed in this harvest)

1. **The FAILURE LEDGER** — the post's "rich failure records" formalized as a canonical surface:
   `.omx/state/harness_failure_ledger.jsonl` (append-only, fcntl-locked) +
   `src/tac/harness_failure_ledger.py` (writer/query, 16 tests). Schema per record:
   `{failure_id, ts, surface(daemon|subagent|gate|tool|trainer-launch), terminal_cause,
   causal_status(hypothesized|measured|falsified — full HISTORY preserved, wrong diagnoses stay
   recorded), mechanism_exposed, recurrence_count, related_ref, resolution(open|worked-around|
   class-fixed|gate-landed)}`. Seeded with 4 real incident classes (daemon 5-min kill with its
   4-falsified-diagnosis history · SIGURG-144 with the recorded 2026-05-14 misattribution ·
   serializer absorbed-hunks · dashboard false-FAIL-at-init). Wired as a **costate-controller
   SENSE input**: `tac.witness_control.producer_bridge._harness_failure_signal` reads
   `sense_rows()` (ranking = unresolved first, recurrence descending — the post's "preference for
   recurrent, addressable patterns" verbatim).

   **Boundary vs existing surfaces (checked, not assumed):** `tools/memory_blackbox.py` records
   the system MEMORY trajectory (samples, not failures); `subagent_progress.jsonl` records live
   resume checkpoints; `council_deliberation_posterior.jsonl` records deliberation VERDICTS;
   `probe_outcomes.jsonl` records lever-measurement outcomes; memory files hold the LESSON prose.
   None is a countable, rankable, causal-status-tracked failure-class store — the ledger fills
   that hole and cross-references the prose via `related_ref` instead of duplicating it.

2. **Design invariants into the costate controller design doc** — see the dated appendix in
   `.omx/research/costate_controller_design_20260705.md`: (i) AUTHORITY-OUTSIDE-THE-LOOP,
   (ii) DIVERSITY-FLOOR, (iii) the Self-Harness D_in/D_out acceptance template for harness
   self-edits. Detail in §c below (the appendix is canonical; this is the pointer).

## (c) The three invariants (summary; canonical text lives in the design doc appendix)

- **(i) AUTHORITY-OUTSIDE-THE-LOOP.** The controller may propose lever/schedule/config actions
  but may NEVER propose edits to authority surfaces: `upstream/evaluate.py` + the pinned
  snapshot, byte-close scoring paths, permission/GO gates, the NO-FAKE gates themselves. Post:
  "A self-improvement loop optimizes whatever signal it is given" — the evaluator/permissions
  sit OUTSIDE the loop or the loop Goodharts them.
- **(ii) DIVERSITY-FLOOR.** When ranking levers by measured ΔS, a fixed exploration share of
  each measurement budget goes to never-fired activation-ledger entries. The activation ledger
  IS our anti-diversity-collapse mechanism; this makes the exploit/explore split explicit
  instead of emergent.
- **(iii) D_in/D_out regression discipline** as the acceptance template for future harness
  self-edits: held-in (does the edit resolve the mined weakness?) + held-out (no new failures)
  + "rejected candidates are logged without changing the active harness" — mapped onto our
  review-gate + 3-clean-pass counter + the failure ledger (rejected proposals = ledger rows with
  `resolution=open` + a note, never silent discards).

## (d) ADOPT-LATER — named triggers (tracked, not forgotten; per the default-off discipline)

| Post idea | What it is | Trigger to adopt |
|---|---|---|
| **MCE bi-level skill optimization** | inner loop finds best context per skill, outer loop selects/crossbreeds skills (`c_s=(ρ_s,F_s)`) | when `.claude/skills` count grows past ~a dozen active skills AND a measurable per-skill objective exists (today: few skills, no per-skill metric — bi-level machinery would be goldplating) |
| **AFlow MCTS workflow search** | tree search over workflow graphs, "soft mixture of score and uniform exploration" | when the campaign layer has a FAST objective (cheap n600-advisory or better) so a search can afford many workflow evaluations; today one campaign evaluation is hours — MCTS starves |
| **AlphaEvolve EVOLVE-BLOCK markers** | `# EVOLVE-BLOCK-START/END` restricts auto-editable code regions | only if we ever let an agent AUTO-EDIT trainer code in a loop; today all trainer edits go through review-gate + serializer + human GO — if that changes, the markers are the editable-surface control the post demands |
| **SIA joint harness+weights** | Feedback-Agent chooses harness-edit vs weight-update | the post itself calls the evidence "provisional" (weak baselines, confounded design); our weights (witness) and harness already co-evolve through the triality cycle — revisit only if a measured case shows the split decision is load-bearing |

## What was deliberately NOT adopted

- **No auto-editing harness loop.** The post's Self-Harness closes the propose→validate→deploy
  loop automatically; ours stays human-gated (review-gate ×2, serializer, operator GO). Rationale:
  invariant (i) + the post's own bottleneck list (reward hacking, long-term-success blindness of
  sandbox metrics) + our measured experience that the closed-loop controller once certified a
  frozen run "converging" (confound hunt meta-confound).
- **No new orchestration layer.** Per CLAUDE.md "Background-execution clarification": the
  non-negotiables ARE the orchestration layer; the ledger extends SENSE, it does not add a loop.

Pointer 0.19110 UNMOVED — this is apparatus/means; the exact score moves only through a
byte-closed `upstream/evaluate.py` row.
