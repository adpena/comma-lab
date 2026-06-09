# Video analysis — @claudedevs dev-velocity post → how Claude (+ Codex CLI + ChatGPT 5.5 Pro) can do better, faster work

**Date:** 2026-06-09
**Subagent task:** transcript + actionable analysis of an X/Twitter video by @claudedevs (Anthropic's Claude Code dev account); write durable memo. No commits, no code edits.
**Target:** `https://x.com/claudedevs/status/2061900434722496604`

---

## ⚠️ TRANSCRIPT UNAVAILABLE — X video is not machine-fetchable

The specific tweet/video could **not** be retrieved. X/Twitter is JS-rendered and auth-walled:

- `WebFetch https://x.com/claudedevs/status/2061900434722496604` → **HTTP 402 Payment Required** (X API paywall).
- `WebFetch https://nitter.net/claudedevs/status/2061900434722496604` → **empty body** (nitter mirror dead/rate-limited).
- `WebSearch` on the raw tweet ID `2061900434722496604` → no result indexed the post body; X results explicitly note "JavaScript is not available."
- I **cannot watch video** in any case.

**Therefore: there is NO verified transcript or direct quote from this specific video below.** The analysis is built from (a) Anthropic's official Claude Code "What's new" docs (VERIFIED, fetched), (b) the Code w/ Claude 2026 conference coverage (VERIFIED, fetched), and (c) my own knowledge of Claude Code + multi-agent dev practice (INFERRED). Every line is tagged. I have **not** fabricated a transcript or quotes per the NO-FAKE non-negotiable.

---

## (a) Best-available content summary (what @claudedevs almost certainly covers)

The @claudedevs account (announced 2026-W?? as Anthropic's dedicated Claude Code dev channel — VERIFIED the account exists: `x.com/claudeai/status/2044779666477646187` "@ClaudeDevs is now on X.") posts feature drops and dev-velocity stories. A dev-velocity video from this account in the relevant window maps to the **Code w/ Claude 2026** conference + the weekly "What's new" digests. Verified content from those primary sources:

**Code w/ Claude 2026 (May 6 SF / May 19 London / Jun 10 Tokyo)** — VERIFIED via `simonwillison.net/2026/May/6/code-w-claude-2026/` + `claude.com/code-with-claude`:
- **Routines** — async automations / "higher-order prompts": "set up async automations and wake up to PRs that are ready to merge." (VERIFIED)
- **Code Review** (`/code-review`) — "used by every team at Anthropic." (VERIFIED)
- **CI auto-fix** — automatic fixes filed against PRs (`/autofix-pr`). (VERIFIED)
- **Remote agents** — drive your laptop agent from your phone; mobile push when a long task finishes or Claude needs you. (VERIFIED)
- Dev-velocity claims: Mercado Libre (23,000 engineers) targeting **"90% autonomous coding by Q3"**; Managed Agents pitch of shipping **"10× faster."** (VERIFIED — these are the kind of soundbite a dev-velocity video features.)

**Claude Code "What's new" 2026 features** — VERIFIED via `code.claude.com/docs/en/whats-new` + `.../2026-w22`:
- **Opus 4.8** (Week 22, May 25–29) — new default; **high effort by default**, `/effort xhigh` for hardest tasks. (This subagent is running on `claude-opus-4-8[1m]`.) (VERIFIED)
- **Dynamic workflows** (Week 22, research preview) — "an orchestration script Claude writes for your task and runs across **dozens to hundreds of subagents** in the background … for a task too large for one conversation to coordinate: a codebase-wide audit, a large migration, a research question that needs cross-checking." Managed with `/workflows`. (VERIFIED)
- **Security-guidance plugin** (Week 22) — fast pattern check per edit + model review per turn + deep agentic review on commit/push; rules in `.claude/claude-security-guidance.md`. (VERIFIED)
- **`/usage`** (Week 21) — breaks down plan limits **by skill, subagent, plugin, and MCP server**. (VERIFIED)
- **Auto mode** (Week 13 → Pro in Week 21) — classifier handles permission prompts; safe actions run, risky ones blocked; "hard deny rules block unconditionally." (VERIFIED)
- **`/goal`** (Week 20) — keeps Claude working across turns until a completion condition holds. (VERIFIED)
- **`claude agents` / Agent view** (Week 20) — one screen for every session: running / blocked-on-you / done; `←←` to open. (VERIFIED)
- **Background sessions** (Week 21) — appear in `/resume`, stay alive when pinned; `claude --bg --exec 'pytest -x'` and `!`-prefixed background jobs in `claude agents`. (VERIFIED)
- **Rewind / "Summarize up to here"** (Week 20) — compress earlier context in place. (VERIFIED)
- **Skills auto-load** (Week 22) — `.claude/skills` dirs load automatically (no marketplace); `/reload-skills`; `SessionStart` hooks can return `reloadSkills: true`; skills/commands can set `disallowed-tools` in frontmatter. (VERIFIED)
- **`/ultrareview` / `claude ultrareview`** (Weeks 17–18) — a **fleet of bug-hunting agents in the cloud**; findings land back in CLI/Desktop. (VERIFIED)
- **Monitor tool** (Week 15) — streams background events into the conversation so Claude can tail logs and react live. (VERIFIED — this lab's deferred toolset includes `Monitor`.)

**Extensibility stack consensus** (3rd-party guides, lower trust but consistent) — INFERRED/SECONDARY: hooks = deterministic lifecycle enforcement ("cannot hallucinate"); subagents = parallel delegation + isolated context windows; skills = invocable playbooks that "cost almost nothing until invoked." Start with skills → add hooks for enforcement → use subagents for parallel/context-isolation work.

---

## (b) How Claude (Claude Code) can do better work in THIS lab

Concrete feature → lab application. (Lab = `/Users/adpena/Projects/pact`, the comma video-compression challenge.)

1. **Parallel subagents as a FIRST-CLASS deliverable, not an afterthought.**
   - Feature: subagents (isolated context) + **dynamic workflows** (Claude writes an orchestration script over dozens–hundreds of subagents).
   - Lab fit: CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" already mandates `tools/parallel_dispatch_top_k.py` as the FIRST file built. The W22 dynamic-workflows feature is the native version of exactly this: when the operator says "parallel/search/sweep," ask Claude for a **workflow** that fans out K candidate dispatches + a harvest leg, rather than hand-rolling sequential validation. A codebase-wide audit (e.g. the Wave N+48 substrate × PR-parity re-audit, or a stale-L1-substrate sweep) is the canonical "too large for one conversation" workflow.

2. **`/goal` + background sessions for long MLX/training jobs.**
   - Feature: `/goal` runs until a completion condition; `claude --bg --exec`, pinned background sessions in `/resume`.
   - Lab fit: directly supersedes the fragile "tool-bg sleep-loop dies at SIGURG-144" pattern flagged in MEMORY (`feedback_durable_detached_daemons_not_session_watchers_20260609.md`). For 12-hour training / full-video VJP / exact replay, use `/goal` with a completion predicate (e.g. "until `contest_auth_eval_*.json` lands AND archive byte-closes") + a **durable SSD checkpoint** on `/Volumes/VertigoDataTier/pact` per the local-disk non-negotiable. The Claude session becomes a *reader* of durable state, not the orchestrator.

3. **Monitor tool to wait on conditions instead of foreground `sleep`.**
   - Feature: `Monitor` streams background events; foreground `sleep` is blocked in this harness.
   - Lab fit: tail dispatch logs / harvest JSONL live and react when a dispatch completes, instead of polling. Pairs with the harvest-or-lose Modal `.spawn()` discipline.

4. **`/code-review` + `/ultrareview` + security-guidance as the adversarial-review apparatus.**
   - Feature: `/code-review` (correctness bugs), `claude ultrareview` (cloud bug-hunting fleet), security-guidance plugin (per-edit + per-turn + on-commit reviews).
   - Lab fit: this lab already runs a "Recursive adversarial review protocol — 3 consecutive clean passes" + a Codex reviewer. Wire `/ultrareview`'s cloud fleet as one more independent reviewer in that loop, and let security-guidance enforce no-secret-leak on the Public Disclosure Hygiene surface (`.claude/claude-security-guidance.md` can encode "no Cloudflare/Lightning URLs, no `/tmp` evidence paths, no provider tokens").

5. **`/usage` to attribute plan burn by skill/subagent/plugin/MCP — feed it back as system intelligence.**
   - Feature: `/usage` decomposes limits per skill/subagent/plugin/MCP server.
   - Lab fit: CLAUDE.md "Results must become system intelligence" + "Max observability." `/usage` is a free observability signal: which subagent classes burn the most budget? Canonicalize into a ledger row so the cathedral autopilot ranker can weight cheap-but-high-EV subagent patterns.

6. **Skills + hooks to make the CLAUDE.md non-negotiables *executable*, not just prose.**
   - Feature: auto-loaded `.claude/skills`, `disallowed-tools` frontmatter, deterministic hooks (`PreToolUse`/`Stop`/`MessageDisplay`), `SessionStart → reloadSkills:true`.
   - Lab fit: the lab's coherence-by-default thesis is "rules not orchestrators." Hooks are the deterministic teeth: e.g. a `PreToolUse` hook that blocks `git commit` outside `tools/subagent_commit_serializer.py` (already half-done via the recommended pre-commit hook in CLAUDE.md), a hook that refuses `--device mps` in auth-eval invocations, a hook that rejects `/tmp` paths in persisted artifacts. These convert "FORBIDDEN PATTERNS" prose into system-level enforcement that "cannot hallucinate."

7. **Opus 4.8 high-effort + `/effort xhigh` for the genuinely hard sub-problems.**
   - Feature: high effort by default; `xhigh` for hardest tasks.
   - Lab fit: reserve `xhigh` for the score-domain Lagrangian / Dykstra-feasibility / entropy-coder math where a wrong derivation costs paid GPU; keep default effort for plumbing. This is the MVP-first phasing rigor cadence applied to *thinking budget*.

8. **Plan mode / Ultraplan before paid dispatch.**
   - Feature: Ultraplan (draft plan in cloud, review in web editor, run remote or pull local).
   - Lab fit: every paid GPU dispatch >$0.30 already requires an MVP-first 5-step recipe + per-substrate symposium. Drafting that as a reviewable plan (operator + Codex can comment) before the meter starts is the native fit.

---

## (c) Claude + Codex CLI (gpt-5.5) + ChatGPT 5.5 Pro — interface/handoff protocols for max-velocity, max-quality code

**The three surfaces in THIS lab:**
- **Claude Code** = primary autonomous agent; edits, commits (via serializer), runs dispatches, owns `main`.
- **Codex CLI (gpt-5.5, `codex exec`)** = sister reviewer/implementer, launched detached (nohup Pattern A) or via the `codex:rescue` skill. Reads `AGENTS.md` natively.
- **ChatGPT 5.5 Pro** = strategic oracle; operator relays verdicts into chat (human in the loop, high latency, no file access).

VERIFIED interop facts (`firecrawl.dev/blog/claude-code-vs-codex`): "the harness matters as much as the base model"; most heavy users run both simultaneously; both speak **MCP** so tools/artifacts are shared; **Codex reads AGENTS.md, Claude reads CLAUDE.md, and both can read each other's format.** The rest of this section is INFERRED protocol design grounded in this lab's existing primitives.

### C1. Typed artifacts both agents read (the shared contract layer)
The lab already has the right substrate — make every handoff a **typed row on disk**, never chat-only:
- **Candidate-action rows** → the meta-Lagrangian/Pareto solver's typed atom (candidate id, family, charged bytes, predicted Seg/Pose/rate deltas, uncertainty, evidence grade, archive/runtime custody, blockers, next proof). Both Claude and Codex emit and consume these. JSON Schema lives next to the consumer.
- **Council/probe verdicts** → `.omx/state/council_deliberation_posterior.jsonl` (fcntl-locked, v2 frontmatter). Codex appends its adversarial verdict as a row; Claude reads it before landing.
- **Subagent checkpoints** → `.omx/state/subagent_progress.jsonl` (`tools/subagent_checkpoint.py`). A Codex implementer and a Claude implementer resuming the same lane read the same `next_action`.
- **Empirical anchors** → `.omx/calibration/anchors_*.json` + `.omx/state/canonical_equations_registry.jsonl`. ChatGPT 5.5 Pro's strategic verdicts get distilled by the operator into a dated `.omx/research/*_directive_*.md` (which CLAUDE.md already mandates every subagent read within 24h) — that file is the ChatGPT→agents bridge.

**Rule:** if a cross-agent decision exists only in chat, it is orphaned. Every handoff MUST land as a typed row or a dated `.omx/research/` directive.

### C2. Division of labor (who does what)
- **Claude Code (owns `main`, owns the meter):** edits code, runs dispatches, lands commits via the serializer, maintains lane registry + canonical equations, drives parallel-dispatch workflows. Default implementer.
- **Codex CLI (reviewer-first, implementer-second):** independent adversarial review of Claude's diffs (the lab's canonical "Codex reviews Claude's diffs" pattern); second-source implementer on DISJOINT files when Claude is busy on a long run (the COMPLEMENTARY convergence pattern — Codex lands the operational module, Claude lands the design spec + ratification). Codex is also the byte-mutation-smoke falsifier (the CODEX-EMPIRICAL-FALSIFICATION-OF-CLAUDE-DESIGN pattern in CLAUDE.md).
- **ChatGPT 5.5 Pro (strategic, no file access):** floor-discovery strategy, paradigm-level kill/keep calls, cross-paradigm composition theory. Operator relays as a directive memo. Never touches files; its output is a *prior*, not an artifact, until a typed row encodes it.

### C3. Adversarial-review handoff (the canonical loop)
1. Claude lands a diff via `tools/subagent_commit_serializer.py` (with `--expected-content-sha256` post-edit working-tree shas).
2. Claude (or operator) dispatches Codex via **Pattern A detached nohup** (`codex exec --skip-git-repo-check --sandbox read-only -m gpt-5.5 -c model_reasoning_effort=xhigh -o .omx/tmp/codex_runs/<label>.last.txt "<review prompt + CLAUDE.md non-negotiable refs>"`). The lab's `codex:rescue` skill wraps this.
3. Codex returns HIGH/MEDIUM findings → each becomes a **fix + a STRICT preflight gate** per the "Bugs must be permanently fixed AND self-protected against" two-landing rule.
4. The clean-pass counter advances only on a round that also answers the **assumption-challenge axis** (META-ASSUMPTION review). Codex and `/ultrareview`'s cloud fleet are independent reviewers; agreement across surfaces is the SEAL signal.
- **Reviewer-vs-author separation** (CLAUDE.md recusal trigger #2): the agent that wrote the artifact must not be its sole approver. Codex reviewing Claude's diff is the structural fix.

### C4. Race avoidance (the serializer + ownership map)
- **Commit races:** every agent commits through the fcntl-locked serializer; bare `git add`/`git commit` is FORBIDDEN. The `--expected-content-sha256` discipline makes the *losing* agent rc=4 and rebase on the winner's landed work instead of swallowing it.
- **File ownership:** Codex and Claude work on **DISJOINT file sets** in the same session (the STAND_DOWN / COMPLEMENTARY patterns). Declare ownership in `.omx/state/active_lane_dispatch_claims.md` + checkpoint `files_touched`. A `PreToolUse` hook (`tools/check_sister_checkpoint_before_git_add.py`) blocks staging a file a sister has uncommitted edits to.
- **Dispatch races:** claim the lane with `tools/claim_lane_dispatch.py claim` before any paid GPU; refuse same-`lane_id` conflicts inside the 24h TTL. This is the $5-10-duplicate-spend fix from 2026-05-01.

### C5. Minimizing round-trip latency
- **Async-by-default:** launch Codex detached (Pattern A) and keep working; poll `.omx/tmp/codex_runs/<label>.last.txt`. Use the **Monitor** tool to react when the file lands instead of blocking. Don't serialize on the reviewer.
- **Batch the ChatGPT round-trip:** ChatGPT 5.5 Pro has the highest latency (human relay). Batch a *set* of strategic questions into one directive ask rather than one-at-a-time; encode its answers once into a directive memo + canonical-equation rows so agents don't re-ask.
- **Shared MCP tools** (Firecrawl/Linear/Postgres style): one MCP setup serves both Claude and Codex — no per-agent re-plumbing. Keep tool surfaces identical so a candidate row produced under Codex runs unchanged under Claude.
- **AGENTS.md ↔ CLAUDE.md parity:** keep the two instruction files in sync (this lab has both). Drift = Codex and Claude operate under different non-negotiables = silent divergence. A periodic parity check (skill or hook) is high-EV.

---

## TOP 5 ACTIONABLE CHANGES

1. **Adopt native dynamic workflows / `/goal` + background sessions for long jobs** — replace fragile session-watcher sleep-loops (SIGURG-144 death) with `/goal` + durable SSD checkpoints; Claude reads state, doesn't babysit. Use a Claude-authored *workflow* whenever the operator says "parallel/sweep/search."
2. **Make handoffs typed-on-disk, never chat-only** — every Claude↔Codex↔ChatGPT decision lands as a candidate-action row, a council posterior JSONL row, or a dated `.omx/research/*_directive_*.md`; chat-only = orphaned.
3. **Wire Codex (detached Pattern A) + `/ultrareview` as independent adversarial reviewers** of every Claude diff; clean-pass counter advances only with the assumption-challenge axis answered; each finding → fix + STRICT preflight gate.
4. **Convert FORBIDDEN-PATTERN prose into deterministic hooks** — `PreToolUse` blocks bare `git commit`, `--device mps` auth-eval, and `/tmp` evidence paths; security-guidance plugin enforces public-disclosure hygiene. Rules with teeth, not just text.
5. **Treat `/usage` + per-subagent checkpoints as observability fed back into the ranker** — attribute budget burn by subagent/skill, canonicalize into a ledger row so the autopilot weights cheap-high-EV patterns (Results-must-become-system-intelligence).

---

### Sources used
- X tweet (target) — **HTTP 402, not retrievable**: `https://x.com/claudedevs/status/2061900434722496604`
- `https://x.com/claudeai/status/2044779666477646187` ("@ClaudeDevs is now on X")
- `https://code.claude.com/docs/en/whats-new` (VERIFIED — weekly feature digests Weeks 13–22, 2026)
- `https://code.claude.com/docs/en/whats-new/2026-w22` (VERIFIED — Opus 4.8, dynamic workflows, security-guidance, fast mode)
- `https://simonwillison.net/2026/May/6/code-w-claude-2026/` (VERIFIED — Code w/ Claude 2026 coverage)
- `https://claude.com/code-with-claude` (VERIFIED — conference page)
- `https://www.firecrawl.dev/blog/claude-code-vs-codex` (VERIFIED — Claude Code + Codex interop, MCP, AGENTS.md↔CLAUDE.md)
- 3rd-party extensibility guides (SECONDARY/lower-trust, consistent): ofox.ai, dev.to/owen_fox, alexop.dev, boringbot.substack.com
