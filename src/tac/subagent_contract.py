"""Canonical subagent prompt contract — the harvested Fable-5 prompting patterns as code.

Sources (verbatim-grade): ``docs/harvest_fable5_prompting_and_loops_20260707.md`` (the
operator-directed harvest of the Fable-5 prompting guide + Loops blog) and
``docs/operating_manual_craft_handoff.md`` (the binding craft handoff). Operator directive
2026-07-07: *"Save and engineer all those patterns as standard behaviors and gates"* — this
module is the STRUCTURAL form of that directive: dispatchers COMPOSE subagent prompts from
these constants instead of re-typing the patterns per prompt (re-typed prompts drift, and a
drifted prompt silently loses the grounded-progress / no-ending-on-promises protections that
Anthropic reports "nearly eliminated fabricated status reports").

Usage (dispatcher side)::

    from tac.subagent_contract import standard_contract

    prompt = f"{task_body}\\n\\n{standard_contract()}"

Anti-rot: ``tac.preflight.check_subagent_contract_module_integrity`` verifies every named
constant + the composer output on each preflight run; the sister gate
``tac.preflight.check_no_reasoning_echo_instructions`` refuses reasoning-echo instructions
in any prompt surface (refusal-storm prevention per harvest pattern 14).
"""

from __future__ import annotations

__all__ = [
    "ANTI_GOLDPLATING",
    "AUTONOMOUS_REFORMULATION",
    "CITATION_CLAUSE",
    "CHECKPOINT_FINDINGS",
    "COMMIT_DISCIPLINE",
    "CONFIRMED_VS_PLAUSIBLE",
    "CONTRACT_CONSTANT_NAMES",
    "CONTROL_LAW_CLAUSE",
    "DECOMPOSE_HEADLINE",
    "EIGHTFOLD_CLAUSE",
    "EXECUTE_DONT_READ",
    "FINAL_MESSAGE_REGROUNDING",
    "FIXES_ARE_UNREVIEWED",
    "FRESH_CONTEXT_VERIFIER",
    "GROUNDED_PROGRESS",
    "KEY_PHRASES",
    "MANUAL_CITATION",
    "MASTER_THESIS_FRAMING",
    "NEVER_REASONING_ECHO",
    "NO_ENDING_ON_PROMISES",
    "NO_MANUFACTURED_FINDINGS",
    "OWN_ROUND1_REVIEW",
    "PAPER_WARM_START_FROM_DIVERGENCE",
    "RECURSION_CLAUSE",
    "CONTENT_LINEAGE_CRUX",
    "CORRECT_OVER_EASY",
    "RESEARCH_AUTHORITY",
    "RESEARCH_ORIGINAL_DESIGN_AUTHORITY",
    "RETAINED_REASONING",
    "RETRIEVAL_FIRST_CLAUSE",
    "PRIMARY_SOURCE_RE_DERIVATION",
    "OPTIMAL_FORM_NO_GENERIC_BASIS",
    "REVIEW_ONLY_CONSTANT_NAMES",
    "REVIEW_STATUS_CLAUSE",
    "RISK_RANKING",
    "SECTION8_CHECKLIST",
    "STATE_THE_BOUNDARIES",
    "TIEBREAK_LEAST_COMPLEXITY",
    "TRIALITY_WIRING",
    "VERDICT_SCOPE_LADDER",
    "review_contract",
    "standard_contract",
]

# --- Harvested behavior blocks (harvest patterns 5 / 9 / 12 / 6 / 3 / 14) -------------------

#: Harvest pattern 5 — the single highest-value adoption (kills fabricated status reports).
GROUNDED_PROGRESS = (
    "GROUNDED PROGRESS: Before reporting progress, audit each claim against a tool result "
    "from this session. Only report work you can point to evidence for; if a claim is not "
    "yet verified, say so explicitly instead of promoting it."
)

#: Harvest pattern 9 — early-stop mitigation for long autonomous sessions.
NO_ENDING_ON_PROMISES = (
    "NO ENDING ON PROMISES: Before ending your turn, check your last paragraph. If it is a "
    "plan, question, or promise about undone work, do that work now with tool calls instead "
    "of ending the turn."
)

#: Harvest pattern 12 — the final summary is written for a reader who saw none of the work.
FINAL_MESSAGE_REGROUNDING = (
    "FINAL-MESSAGE RE-GROUNDING: Working shorthand is fine between tool calls, but the "
    "final summary is a RE-GROUNDING for a reader who saw none of it: outcome first, plain "
    "language, no arrow chains or invented labels, each identifier in its own clause."
)

#: Harvest pattern 6 — guards unrequested actions (fixes, defensive branches, unasked sends).
STATE_THE_BOUNDARIES = (
    "STATE THE BOUNDARIES: When the user is describing a problem, the deliverable is your "
    "assessment. Report findings and stop. Don't apply a fix until they ask."
)

#: Harvest pattern 3 — anti-goldplating (effort goes to the ask, not to unrequested scope).
ANTI_GOLDPLATING = (
    "ANTI-GOLDPLATING: Don't add features, refactor, or introduce abstractions beyond what "
    "the task requires. Build the simplest thing that works well; only validate at system "
    "boundaries."
)

#: Harvest pattern 14 — fresh-context verifier subagents beat self-critique.
FRESH_CONTEXT_VERIFIER = (
    "FRESH-CONTEXT VERIFICATION: Establish a method for checking your own work at interval "
    "[X]: fresh-context verifier subagents beat self-critique — verify with subagents "
    "against the specification, not against your own recollection of it."
)

#: Harvest pattern 14 (WARNING half) — dispatcher-side only; NEVER pasted as a model task.
#: Every trigger phrase below deliberately shares a source line with a negation token so the
#: sister gate ``check_no_reasoning_echo_instructions`` reads this constant as negated.
NEVER_REASONING_ECHO = (
    "WARNING (dispatcher-side, never a model instruction): NEVER tell a model to 'show your "
    "thinking', and don't ask it to 'transcribe your reasoning'; never demand it 'reproduce "
    "your chain of thought' and never require it to 'echo your internal reasoning'. "
    "Reasoning-echo instructions trip reasoning-extraction classifiers on Fable-class "
    "models and cause refusal storms (fallback storms). Read thinking blocks instead."
)

#: #1121 waiter discipline — a waiter that dies is a waiter that LIES about being done.
#:
#: Measured 2026-08-18: ddm_iv1 finished its real work, then re-invoked MAIN FOUR separate
#: times as its backgrounded `sleep`-style waiters expired one by one. Every one of those
#: notifications carried zero information — the arm's own words across them were "stale
#: waiter", "drained waiter", "another drained solve waiter". Each cost a full orchestrator
#: turn to read and dismiss. The waiters outlived the thing they were waiting for, and a
#: dead waiter is indistinguishable at the notification boundary from a finished job.
#:
#: The cure has two halves. The NOISE half: bind the waiter to the completion artifact so it
#: fires once, on a real event, and is reaped with its subject. The DANGEROUS half, found the
#: same day by the same arm: a waiter shaped `until ! pgrep <predecessor>; do sleep; done;
#: <launch successor>` is not a waiter at all — it is a LATENT ACTUATOR. It fired ~30 minutes
#: after its successor step had already been run, adjudicated and written up, launching a
#: duplicate that was on course to overwrite an adjudicated receipt mid-read. The arm caught
#: it only because one notification said "completed" instead of "killed" — a thin thread.
#:
#: The mechanism, stated by that arm and worth keeping in its words: a wait CONDITION and a
#: launch DECISION have different lifetimes. The condition can come true long after the
#: decision stopped being correct.
WAITER_DISCIPLINE = (
    "WAITER DISCIPLINE (#1121), two rules. (1) WAITERS OBSERVE, THEY DO NOT ACTUATE. Never "
    "write `until ! pgrep <predecessor>; do sleep; done; <launch successor>`. That is not a "
    "wait, it is a latent actuator: a wait CONDITION and a launch DECISION have different "
    "lifetimes, and the condition can come true long after the decision stopped being "
    "correct. Measured 2026-08-18: one such waiter fired ~30 minutes after its step had "
    "already run and been adjudicated, launching a duplicate on course to overwrite an "
    "adjudicated receipt mid-read; it was caught only by a lucky difference in one "
    "notification's wording. Re-decide at fire time, with fresh state, or don't fire. "
    "(2) Never background a bare `sleep`/poll loop to wait for your own child work — those "
    "waiters outlive their subject, expire independently, and each death re-invokes the "
    "orchestrator with NO information (four consecutive zero-signal notifications the same "
    "day, each costing a full turn, still arriving while this clause was being written). "
    "Bind the wait to the completion ARTIFACT instead: launch through "
    "tools/launch_detached_process.py with a `--done` marker, or poll a file/receipt the "
    "work itself writes, so the waiter fires exactly once on a real event and is reaped "
    "with its subject. Wait on an artifact's existence, never on a clock. A waiter that can "
    "fire when nothing happened is not an instrument."
)

# --- #346 retrieval-first layer clauses (behavioral defaults as structure) -------------------
#
# Source: memory `apparatus_writes_better_than_it_reads_retrieval_first_nexus_20260707` —
# every 2026-07-07 operator catch already existed in a durable store but wasn't loaded at
# decision time. These four clauses make the behavioral defaults (retrieval-first,
# recursion-default, positive-design control laws, review-status provenance) part of EVERY
# composed subagent prompt instead of volitional habits.

#: #346 recursion-default — a conclusion is the first move of a chain, not a resting state.
RECURSION_CLAUSE = (
    "RECURSION-DEFAULT: a conclusion is the start of a chain, not the end of one. Pursue "
    "the follow-ups you can run yourself to a wall or a measured terminus; proposing a "
    "follow-up you could have run yourself is a violation."
)

#: #346 positive-design — every recommended knob ships as a CONTROL LAW, never a TBD.
CONTROL_LAW_CLAUSE = (
    "CONTROL LAWS: every recommended knob is a control law — one of: constant | "
    "ramp/anneal with a completion guarantee | self-deriving formula | event-conditioned "
    "tested predicate | fractional/partial gate — each with its derivation/anchor, or a "
    "default plus a NAMED recess measurement that sets it. 'TBD' is forbidden."
)

#: #346 retrieval-first — consult the durable stores BEFORE concluding; say which ones.
RETRIEVAL_FIRST_CLAUSE = (
    "RETRIEVAL-FIRST: before any verdict/design/charter, consult the durable stores "
    "(tools/corpus_query.py '<topic>' — one query over research/equations/memory/DAG/"
    "council/tasks/docs) and state a 'STORES CONSULTED:' line naming what you loaded and "
    "what you deliberately did not. The Stop hook enforces the line on decision-class docs."
)

#: #346 review-status provenance — verdicts carry how reviewed they are, always.
REVIEW_STATUS_CLAUSE = (
    "REVIEW STATUS: tag every load-bearing verdict you rely on or emit with its review "
    "provenance — pre-registered-only / recovery-written-UNREVIEWED / "
    "fresh-eyes-reviewed(N). An untagged verdict reads as more reviewed than it is."
)

# --- Requirement-S citation provenance clause (T5 crucible, 2026-07-08) ----------------------
#
# Source: ORCHESTRATION_LEDGER requirement S (operator 2026-07-08: "make sure we are recording
# arxiv paper citations and stuff too for scientific rigor and provenance"). Measured gap at
# binding: CT-1 = 0 resolvable citations, CT-2 = 11, v5 = 1. Backfill:
# `.omx/research/t5_crucible/BIBLIOGRAPHY_20260708.md`.

#: Requirement S — citations are provenance for claims exactly as anchors are for measurements.
CITATION_CLAUSE = (
    "CITATIONS (requirement S): record a resolvable citation — authors · year · exact title "
    "· arXiv ID or DOI — for every imported result/theorem/method AT THE POINT OF DERIVATION; "
    "verify each ID actually resolves to the named paper (fetch the abstract page) before "
    "recording it; where no supporting paper exists, say so explicitly — an uncited imported "
    "theorem is the literature-side analog of an unanchored verdict."
)

# --- #337 contract blocks + manual citation --------------------------------------------------

#: #337 contract: the builder owns round 1 of the adversarial review of its OWN output.
OWN_ROUND1_REVIEW = (
    "OWN ROUND-1 ADVERSARIAL REVIEW (#337 contract): after you build, switch roles and run "
    "round 1 of the adversarial review on your OWN output before handing it over — trace "
    "every assumed dict key / flag / unit, check whether each fix repairs the CLASS or just "
    "the instance, and ask whether your tests would still pass if the code were broken. "
    "Your own fixes are unreviewed new code: a fix round resets the clean-pass counter."
)

#: #337 contract: keep the three triality legs consistent in the SAME commit batch.
TRIALITY_WIRING = (
    "TRIALITY WIRING (#337 contract): a lever/wire-in/curriculum change must land in the "
    "DSL (tac.witness_dsl) and a measured finding must land in tac.canonical_equations, "
    "with the DAG FEED appended — in the SAME commit batch, proactively (the triality "
    "drift detector flags misses; a firing hook is a miss, not a reminder)."
)

#: Catalog #405 (2026-07-08) — commit-through-the-serializer discipline, hardened.
#: Post-commit HEAD verification (rc=7) is automatic when the caller declares
#: --expected-content-sha256; shared hot files must be staged via --patch-file
#: (intent-manifest) so a co-mingled/clobbered working tree cannot leak a
#: sibling's hunks into the commit body.
COMMIT_DISCIPLINE = (
    "COMMIT DISCIPLINE (#405): land every change via "
    "tools/subagent_commit_serializer.py with post-edit --expected-content-sha256 "
    "— post-commit HEAD verification is automatic (rc=7 if the committed content "
    "is not what you declared, e.g. a sibling clobbered the file before your "
    "snapshot); when you edit a shared hot file (the trainer, curriculum_dsl.py, "
    "preflight.py, the DAG, CLAUDE.md), shared-file edits use --patch-file "
    "(supply exactly your hunks — the serializer applies them to a clean index "
    "and ignores the working tree, so no sibling hunk is absorbed). "
    "IF THE SERIALIZER CANNOT RUN AT ALL (it fails before staging — e.g. the "
    "sandbox refuses .git writes), do NOT bypass it and do NOT stop silently: "
    "emit an UNCOMMITTED-WORK MANIFEST in your final message and in a durable "
    "receipt — every path you changed, each with the sha256 of its POST-EDIT "
    "working-tree content — so MAIN can verify byte-identity and land it "
    "unchanged. Measured work you cannot commit is signal loss unless the "
    "manifest exists (ddm_ai1 2026-08-09: −2,416 B receiver-closed, stranded by "
    "a git-write refusal, recovered in one turn because its handoff listed all "
    "four files with post-edit hashes that verified ALL-MATCH)."
)

#: `ddm_rs2` 2026-08-03 — the checkpoint store answers "where do I resume" and never
#: "what did we learn", so a killed arm's hard-won insight dies with its context.
CHECKPOINT_FINDINGS = (
    "CHECKPOINT A FINDING EVERY TIME: every tools/subagent_checkpoint.py write carries at "
    "least one --finding — a one-line thing you LEARNED at that step (a measured number, a "
    "refuted assumption, a structural blocker), not what you are about to do next. "
    "next_action tells a successor where to resume; only findings tell them what you "
    "already know, and an arm killed mid-flight takes everything else with it."
)

#: The eight design philosophies (operator 2026-07-09 "Encode all"; memory
#: design_philosophies_eightfold_20260709). One compact paragraph so every composed dispatch
#: carries them — the honesty disciplines that fall out of the geometry-first stance + clauses A/B.
EIGHTFOLD_CLAUSE = (
    "DESIGN PHILOSOPHIES (design_philosophies_eightfold_20260709): honor the eight — "
    "P1 one fact, one store, one key (no parallel key-spaces); "
    "P2 every comparison carries its noise floor (a Δ below the composed floor is INSTANCE-level, "
    "not a verdict — and our single-seed spine leaves across-seed variance UNKNOWN); "
    "P3 tolerance budgets (the dual waterfill — each pipeline stage proves it stays inside its "
    "distortion allocation); P4 no meter without a canary (a new measurement surface ships with a "
    "positive + negative control before its readings count); P5 no arm without its in-run control "
    "(borrowed baselines forbidden; both A/B arms under identical conditions); P6 the sequence is the "
    "object (temporal first-class, not per-frame fixes); P7 falsifier before build (name the kill "
    "criterion + pre-registered threshold before starting); P8 floor-first (derive/measure a term's "
    "floor, optimize only the gap-to-floor, a surface at floor is CLOSED to polish). Plus the two "
    "clauses: A no duplicate data — every byte names ONE geometric home; B waterfill bits by marginal "
    "distortion. Also: fmtools (our on-device FM classification primitive, #259) is available for "
    "hard/fuzzy classification surfaces — use it as an ADVISORY classifier where regex/name heuristics "
    "are uncertain (auto-push hook #375 is the usage exemplar), NEVER as sole authority on "
    "score-relevant decisions."
)

#: The craft handoff binds every subagent; cite it so the subagent actually loads it.
MANUAL_CITATION = (
    "OPERATING MANUAL: docs/operating_manual_craft_handoff.md is binding for this task — "
    "read the real ask, verify by re-deriving from primary artifacts, label every claim "
    "MEASURED / DERIVED / INFERRED / ASSUMED / UNKNOWN, and answer first, reasoning small, "
    "risk always."
)

# --- Review-dispatch blocks (operating manual §3 / §5 / §6 / §8 as verbatim-grade code) ------
#
# These make the manual's REVIEW method structural: a review dispatcher composes them via
# ``review_contract()`` instead of re-typing (re-typed review prompts drift and silently lose
# the risk-ranking / counter-reset protections). They are NOT part of ``standard_contract()``
# — see ``REVIEW_ONLY_CONSTANT_NAMES``.

#: Manual §3 — effort follows probability × blast-radius × SILENCE, not line count.
RISK_RANKING = (
    "RISK RANKING (manual §3): rank findings by probability × blast-radius × SILENCE; a "
    "quiet byte-identity break outranks a loud viz nit. Blast radius is not line count — a "
    "3-line change to the score-authority path outranks a 500-line presentation change. "
    "Spend reviewer depth on (a) whatever produces the authoritative number, (b) whatever "
    "claims safety, (c) any default that changed, (d) any 'identical/equivalent' claim."
)

#: Manual §5 — findings are labeled by how they were obtained; the labels never blur.
CONFIRMED_VS_PLAUSIBLE = (
    "CONFIRMED vs PLAUSIBLE (manual §5): label every finding CONFIRMED (you reproduced/"
    "verified it by execution) or PLAUSIBLE (needs author confirmation); never blur them. "
    "A caveat travels WITH the finding every time it is repeated."
)

#: Manual §6.3 / §8.8 — the clean-pass counter semantics, verbatim-grade.
FIXES_ARE_UNREVIEWED = (
    "FIXES ARE UNREVIEWED NEW CODE (manual §6): fixes made in response to review are "
    "unreviewed new code; the clean-pass counter resets on ANY finding; round-finished ≠ "
    "clean-pass. Review fix rounds with the same hostility as the original code."
)

#: Manual §7.4 — honest negatives are deliverables; manufactured findings are fakes.
NO_MANUFACTURED_FINDINGS = (
    "NO MANUFACTURED FINDINGS: if nothing severe survives verification, say so plainly; "
    "honest limits ≠ defects. Do not pad a clean pass into fake findings, and do not "
    "promote a nit to justify the dispatch."
)

#: Manual §8 — the ten competence-lookalike mistakes as a per-commit review checklist.
SECTION8_CHECKLIST = (
    "SECTION-8 CHECKLIST (manual §8, per reviewed commit): check each of the ten "
    "competence-lookalike mistakes — (1) means narrated as ends (pointer unmoved but "
    "'progress' claimed); (2) capacity-sweep reflex displacing an already-measured lever; "
    "(3) point-fix with the CLASS still live; (4) plausible summary written from memory "
    "instead of the live artifact; (5) borrowed number cited without its vehicle+surface; "
    "(6) agreeing with the test instead of stating the epistemic state; (7) fan-out as "
    "theater where one measurement decides; (8) round-finished treated as clean-pass; "
    "(9) silent fail-open guard; (10) polish-hoarding over the highest-risk open item."
)

#: Manual §4.3 — testimony is not evidence; execute the claim.
EXECUTE_DONT_READ = (
    "EXECUTE, DON'T READ: verify claims by running tests/code, not by reading commit "
    "messages or docstrings. A docstring is testimony, not evidence — trace the call site "
    "or execute the two paths and diff the outputs."
)

#: Operator 2026-07-14 — a naive NO-GO is the start of the reformulation ladder, not a stop.
AUTONOMOUS_REFORMULATION = (
    "AUTONOMOUS REFORMULATION: a naive/first-cut NO-GO is the start of the reformulation "
    "ladder, not a stopping point. NEVER hand back a naive negative. If your first "
    "implementation hits NO-GO/BLOCKED, build the OPTIMAL FORM (all canonical fixes "
    "applied) and pursue the reformulation queue + follow-ons AUTONOMOUSLY within this run "
    "— iterate until you have an optimal-form verdict or the family is genuinely exhausted. "
    "A negative from a naive form is INSTANCE-scoped only, never a family/paradigm kill, and "
    "must carry the untested-reformulation queue. Only a hard operator-GO gate (heavy/paid "
    "launch) or a truly exhausted optimal-form family ends the pursuit — not a first cut."
)

#: Operator 2026-07-14 — external research rarely hands a precisely-on-point implementation,
#: but carries rich signal; trace assumption-divergence, warm-start with OUR premises, carry to code.
PAPER_WARM_START_FROM_DIVERGENCE = (
    "PAPER WARM-START FROM DIVERGENCE: an external paper rarely hands you a precisely-on-point "
    "implementation — but it almost always carries rich signal. Do NOT collapse intake to "
    "'directly-applicable -> route / else cross-ref-and-dismiss' (that binary is the failure "
    "mode). When a paper is NOT directly applicable: TRACE UPSTREAM through its derivation ONLY "
    "AS FAR AS its assumptions diverge from OURS, and stop at that fork. WARM-START at the "
    "paper's research from that fork — but substitute OUR assumptions, OUR problem space "
    "(v9·CGauge witness + level-set flow), and the FROZEN CONTEST INFORMATION SPACE (SegNet "
    "argmax / PoseNet YUV6 / exact archive bytes) for theirs. Then carry the re-derivation ALL "
    "THE WAY THROUGH — to a concrete DESIGN and an IMPLEMENTATION against our codebase — not a "
    "'conceptual cross-ref'. This is a DEEP undertaking, NOT a one-shot map or a first-cut: "
    "DEEP-READ the paper's FULL METHODOLOGY (derivations, proofs, algorithm, appendices — never "
    "just the abstract), re-derive the DEEP MATH with our assumptions substituted, and EXPERIMENT "
    "+ ITERATE on OUR data + surfaces (through R at n600) until the formulation converges (a shallow "
    "first-cut is itself an INSTANCE-only result, per the reformulation ladder). The deliverable is "
    "a v9·CGauge design brief (and, where "
    "$0-measurable, a queued/dispatched lever under the per-task cap; heavy/paid stays "
    "operator-GO), NEVER a dismissal. 'Not directly applicable' is the START of the warm-start, "
    "not a verdict."
)

# --- Workflow-v2 velocity/rigor/autonomy doctrine (operator 2026-07-20) ----------------------

#: Information authority is independent of dispatch-spend authority.
RESEARCH_AUTHORITY = (
    "RESEARCH AUTHORITY: full online research (web, arXiv, documentation, and OSS repositories) "
    "and use of all open-source software are IN-AUTHORITY. '$0 local' bounds spend only, never "
    "information: it means no paid dispatch, not no research. On explored territory, search the "
    "literature and existing implementations first; deep-read relevant methods, preserve exact "
    "citations and source provenance, then adapt or build instead of rediscovering known work."
)

#: Composite headlines conceal the interaction terms that decide the next action.
DECOMPOSE_HEADLINE = (
    "DECOMPOSE EVERY HEADLINE: a bare composite number is UNMEASURED. Ship the authority-relevant "
    "per-class, per-stratum, per-byte-section, and per-objective-term decomposition, including "
    "interaction terms and the surface/axis that produced it. Reuse a canonical splitter when one "
    "exists; otherwise build the splitter before treating the headline as decision evidence."
)

#: Score-neutral choices have a deterministic simplicity tiebreak.
TIEBREAK_LEAST_COMPLEXITY = (
    "TIEBREAK — LEAST COMPLEXITY: without touching score, ALWAYS choose the least complex "
    "solution or design. Wall clock and complexity are heuristics for each other: prefer the form "
    "with fewer stages, states, special cases, and operational dependencies when score authority "
    "is neutral; score movement remains the primary objective."
)

#: The pursuit searches the full frozen evaluator information space, not one implementation basin.
MASTER_THESIS_FRAMING = (
    "MASTER THESIS FRAMING: treat the frozen contest information space as an inversion over "
    "formulation × realization × completeness, not ordinary codec fidelity or a single model "
    "family. Search hybrids across all three axes against the joint (score, wall-clock) objective; "
    "the winning witness is the shortest complete evaluator-equivalent realization, regardless of "
    "which familiar representation supplied its parts."
)

#: A negative must say exactly how much of the search space it actually closes.
VERDICT_SCOPE_LADDER = (
    "VERDICT SCOPE LADDER: every negative carries one explicit scope token from "
    "INSTANCE < FORMULATION < FAMILY < PARADIGM. Default to the narrowest supported scope; one "
    "failed formulation never kills a family. A family verdict requires a proved bound or at least "
    "two materially distinct optimal-form formulations, while a paradigm verdict requires explicit "
    "operator/council authority. Record the still-open reformulations with every narrower negative. "
    "RECORD IT THROUGH THE TYPED PRODUCER, not as prose: call "
    "tac.verdicts.emit_verdict(..., scope=VerdictScope(level=ScopeLevel.<LEVEL>, scoped_to=...), "
    "is_negative=True, reformulation_queue=[...]) — it REFUSES a family/paradigm scope that carries "
    "no family_evidence and a negative that carries no reformulation queue, so the ladder binds at "
    "write time instead of being audited after the fact. A scope token that exists only as prose in "
    "a memo is unqueryable: no consumer can count it, so it neither protects the killed signal nor "
    "survives the next agent's recall."
)

#: The archive-gravity guard: content lineage + crux alignment declared up front (operator 2026-07-20).
CONTENT_LINEAGE_CRUX = (
    "CONTENT LINEAGE + CRUX ALIGNMENT: state, up front and in the memo, the lineage of every "
    "consumed content artifact (from-scratch / our-solve / inherited-JUSTIFIED) and which stage "
    "of describe->realize your work attacks. Inherited content is harvest-only signal, never a "
    "starting point or a candidate component; the submittable line is 100% ours. The crux is "
    "REALIZATION - if your work does not touch it, say which solved end it serves instead."
)

# --- Primary-source re-derivation (operator 2026-08-01) -------------------------------------
#
# Source: operator binding 2026-08-01 verbatim — "Make sure that it never recalls anything only
# from the working memory and that all implementations are optimal and informed. never naive or
# toy or generic basis." Sister of the 2026-07-31 directive that produced ddm_us1 ("you're just
# recalling from working memory").
#
# THE GAP THIS FILLS, and why RETRIEVAL_FIRST_CLAUSE did not already cover it: retrieval-first
# says CONSULT THE STORES. It says nothing about the numbers in the arm's OWN PROMPT. A dispatch
# seed is a recollection written by the dispatcher — it carries the dispatcher's errors, its
# staleness, and its framing, and an arm that builds on it inherits all three without ever
# touching a source. MEASURED the same day: MAIN's own routing note to ddm_rh1 quoted
# `ddm_tr1_runtime.py:319` correctly and read it as SAFE; the line admits an ABSENT token_codec,
# and rh1 found the hole MAIN had on screen. Reading is not re-deriving.

#: Every number handed to you is a POINTER, not evidence — including the ones in this prompt.
PRIMARY_SOURCE_RE_DERIVATION = (
    "NEVER RECALL ONLY FROM WORKING MEMORY: every number, mechanism, and verdict in your "
    "dispatch prompt is a POINTER to a primary artifact, not evidence. Before any of it becomes "
    "a premise, re-derive it at the source - file:line for a mechanism, commit sha or receipt "
    "path for a measurement, the canonical pointer file for a score. The seed was written by "
    "someone recalling, so it carries their errors, staleness, and framing; building on it "
    "without checking inherits all three silently. State which seed facts you RE-DERIVED and "
    "which you could not - an unchecked premise must be labelled, never assumed."
)

#: Optimal-form + no-generic, with the measured sharpening that DERIVED is not CORRECT.
OPTIMAL_FORM_NO_GENERIC_BASIS = (
    "OPTIMAL AND INFORMED, NEVER NAIVE/TOY/GENERIC: no generic basis, metric, default, ladder, "
    "or threshold anywhere - in what you propose, what you build, or what you EVALUATE with. "
    "Cosine / Fourier / Euclidean / uniform / round-number tolerances are CONTROLS, never "
    "answers; every constant sits on the value-provenance ladder and every knob is derived or "
    "raced. Propose mechanisms at OPTIMAL FORM (per-lever hyperparameters at their own optimum, "
    "not a shared default) and name the real measurement that would score them. A verdict drawn "
    "on a toy, a subset, a proxy, or a naive first implementation falsifies the IMPLEMENTATION, "
    "not the paradigm - scope it that way or do not draw it. MEASURED SHARPENING (ddm_rh1, "
    "2026-08-01, 8f8af336b7): racing three candidates against the REAL encoder gave "
    "hand-DERIVED -1,140 B, GENERIC control -2,117 B, RACED family -2,781 B - the derived "
    "surrogate LOST to the generic control by 1.9x because it was context-blind. 'Derived' is "
    "not a synonym for 'correct': a derivation buys a better CANDIDATE, never a skipped RACE."
)

#: The root selection rule (operator 2026-07-20): never easy; correct, true, understanding.
CORRECT_OVER_EASY = (
    "CORRECT OVER EASY: never go for easy - go for correct and true and understanding. Easy is "
    "the generator of every drift class: archives over from-scratch, proxies over exact, "
    "constants over derivations, composites over decompositions. Encode-side compute is "
    "UNBOUNDED: never scope a solve down for cost; a wall met mid-solve is a missing equation "
    "to register, not a stop. DECODE TIME IS NOT A DISQUALIFIER (operator binding 2026-08-09, "
    "'Don't disqualify anything decoder due to time unless I give authorization'): never "
    "reject, down-rank, cap, or scope down a decoder/receiver/inflate candidate on wall-clock "
    "alone without explicit operator authorization. The score has NO time term "
    "(upstream/evaluate.py:92) - it charges archive.zip bytes only - and `timeout-minutes: 30` "
    "bounds the WHOLE CI job, of which decode is the term WE MOVE with measured headroom. "
    "Report a measured decode time as a FACT alongside its bytes; never convert it into a "
    "verdict, a gate, or a reason a candidate was not raced."
)

# --- Harness retained-reasoning + original-design authority (ddm_hw1, task #785; 2026-07-30) --
#
# Source: operator directive 2026-07-29 ("make you smarter and more capable and less forgetful
# and more coherent") + the ARC-AGI-3 harness crosswalk (Bigio & Sanders 2026-07-29: two
# settings — retained reasoning + context compaction — took GPT-5.6 Sol 13.3%->38.3% at 6x
# fewer output tokens). An arm's reasoning dies with the arm today; the retained-reasoning
# clause is OUR agent-side equivalent: the final message carries the live state a successor
# would otherwise re-walk. #767 (the standing original-design authority) rides the same touch.

#: Retained-reasoning handoff — the arm's live state survives the arm (ARC-AGI-3 crosswalk).
#: Asks for CONCLUSIONS / hypotheses / next actions (NOT a reasoning transcript), so the sister
#: gate check_no_reasoning_echo_instructions stays green (no echo-your-thinking phrasing).
RETAINED_REASONING = (
    "RETAINED REASONING (harness crosswalk, ARC-AGI-3): end your final message with "
    "LIVE-HYPOTHESES (untested leads still worth pursuing, each with why it is plausible) and "
    "DEAD-ENDS (paths you closed, each with the reason, so no successor re-tries them). If any "
    "future action remains, add the exact Markdown heading `## NEXT_IF_RESUMED` and put one "
    "action per bullet under it, naming its disposition, owner, consumer store, and fire trigger. "
    "Omit that heading entirely when no future action remains, so the extractor cannot create a "
    "phantom plan row. State conclusions, hypotheses, and next actions in plain language — this "
    "is a handoff of WHAT you found and WHERE to go next, not a replay of how you got there."
)

#: #767 standing original-design authority (operator 2026-07-29 verbatim). An arm asking
#: permission to design an original is a signal the charter under-scoped it — so it ships in the
#: contract. Memory: all_arms_online_research_and_oss_authority_standing_20260720.
RESEARCH_ORIGINAL_DESIGN_AUTHORITY = (
    "ORIGINAL-DESIGN AUTHORITY (#767, operator 2026-07-29 verbatim: \"Remember all always have "
    "full research and OSS authority and authority to derive and engineer and iterate and "
    "optimize our own variants or original\"): deriving/engineering/iterating/optimizing OUR "
    "OWN variant or original (coder, optimizer, carrier, solver, schedule, kernel — any "
    "surface) is in-authority BY DEFAULT whenever the found/published form is not "
    "measured-optimal for our frozen-scorer task — never a per-arm grant to re-request. "
    "Guardrail: an original is RACED against the incumbent on real payloads (never presumed), "
    "priced with a real coder, honesty-labeled, and borrowed-substrate-accounted."
)

#: Operator 2026-08-09 verbatim: "All sol ultra agents from here have full online research
#: authority and full authority to leverage anything that we have anywhere in our code base or
#: research documentation or anywhere. To use off the shelf or to adapt or refactor or extend or
#: enhance or otherwise drive our own variant or original." RESEARCH_AUTHORITY already grants the
#: EXTERNAL half and RESEARCH_ORIGINAL_DESIGN_AUTHORITY the ORIGINAL half; this is the INTERNAL
#: half, and it is the one our measured pathologies live in — built-elsewhere-unwired (#864/#868),
#: adoption decay (#936: a surface with 0 production callers), and arms re-deriving what the corpus
#: already holds. Filing a debt row about our own unwired code is now the WRONG move when wiring it
#: is in-authority. Memory: all_sol_ultra_full_internal_leverage_authority_standing_20260809.
INTERNAL_LEVERAGE_AUTHORITY = (
    "INTERNAL-LEVERAGE AUTHORITY (operator 2026-08-09 verbatim: \"full authority to leverage "
    "anything that we have anywhere in our code base or research documentation or anywhere. To "
    "use off the shelf or to adapt or refactor or extend or enhance or otherwise drive our own "
    "variant or original\"): ANY surface we already own — shipped code, dead code, a built-but-"
    "unwired module, a landed-but-unconsumed receipt, a worktree residue, a research memo, a "
    "registry, an equation, a harness, a kernel — is yours to USE OFF THE SHELF, ADAPT, REFACTOR, "
    "EXTEND, ENHANCE, or fork into our own variant. You do NOT need a per-arm grant, and you do "
    "NOT file a debt row about our own unwired code when wiring it is the cheaper correct move. "
    "Consult the corpus FIRST (tools/corpus_query.py) — re-deriving what we already measured is "
    "the cardinal signal-loss sin. Guardrails, all unchanged: reuse is PROVENANCE-PINNED (cite "
    "path + commit/sha of what you took); a reused number is re-measured on THIS object, never "
    "inherited across vehicles or regimes; a refactor that touches a score-authority path proves "
    "byte-identity or measures the delta; and the READ-ONLY surfaces stay read-only (pinned "
    "upstream snapshot, public-PR intake clones, sister arms' landed artifacts are append-only)."
)

# --- Registry (consumed by tests + the preflight integrity gate) -----------------------------

#: Every named contract constant this module guarantees. The preflight integrity gate
#: hardcodes the same names so emptying this registry cannot self-waive the gate.
CONTRACT_CONSTANT_NAMES: tuple[str, ...] = (
    "GROUNDED_PROGRESS",
    "NO_ENDING_ON_PROMISES",
    "FINAL_MESSAGE_REGROUNDING",
    "STATE_THE_BOUNDARIES",
    "ANTI_GOLDPLATING",
    "FRESH_CONTEXT_VERIFIER",
    "RECURSION_CLAUSE",
    "CONTROL_LAW_CLAUSE",
    "RETRIEVAL_FIRST_CLAUSE",
    "PRIMARY_SOURCE_RE_DERIVATION",
    "OPTIMAL_FORM_NO_GENERIC_BASIS",
    "REVIEW_STATUS_CLAUSE",
    "CITATION_CLAUSE",
    "EIGHTFOLD_CLAUSE",
    "NEVER_REASONING_ECHO",
    "OWN_ROUND1_REVIEW",
    "TRIALITY_WIRING",
    "COMMIT_DISCIPLINE",
    "MANUAL_CITATION",
    "RISK_RANKING",
    "CONFIRMED_VS_PLAUSIBLE",
    "FIXES_ARE_UNREVIEWED",
    "NO_MANUFACTURED_FINDINGS",
    "SECTION8_CHECKLIST",
    "EXECUTE_DONT_READ",
    "AUTONOMOUS_REFORMULATION",
    "PAPER_WARM_START_FROM_DIVERGENCE",
    "RESEARCH_AUTHORITY",
    "DECOMPOSE_HEADLINE",
    "TIEBREAK_LEAST_COMPLEXITY",
    "MASTER_THESIS_FRAMING",
    "VERDICT_SCOPE_LADDER",
    "CONTENT_LINEAGE_CRUX",
    "CORRECT_OVER_EASY",
    "RETAINED_REASONING",
    "RESEARCH_ORIGINAL_DESIGN_AUTHORITY",
    "INTERNAL_LEVERAGE_AUTHORITY",
    "CHECKPOINT_FINDINGS",
)

#: Review-dispatch-only constants: composed by ``review_contract()``, deliberately NOT
#: composed by ``standard_contract()`` (a build dispatch is not a review dispatch).
REVIEW_ONLY_CONSTANT_NAMES: tuple[str, ...] = (
    "RISK_RANKING",
    "CONFIRMED_VS_PLAUSIBLE",
    "FIXES_ARE_UNREVIEWED",
    "NO_MANUFACTURED_FINDINGS",
    "SECTION8_CHECKLIST",
    "EXECUTE_DONT_READ",
)

#: Per-constant key phrase — the load-bearing fragment each constant must keep verbatim.
KEY_PHRASES: dict[str, str] = {
    "GROUNDED_PROGRESS": "Before reporting progress, audit each claim against a tool result",
    "NO_ENDING_ON_PROMISES": "Before ending your turn, check your last paragraph",
    "FINAL_MESSAGE_REGROUNDING": "RE-GROUNDING for a reader who saw none of it",
    "STATE_THE_BOUNDARIES": "Report findings and stop",
    "ANTI_GOLDPLATING": "beyond what the task requires",
    "FRESH_CONTEXT_VERIFIER": "verify with subagents against the specification",
    "RECURSION_CLAUSE": "a conclusion is the start of a chain",
    "CONTROL_LAW_CLAUSE": "every recommended knob is a control law",
    "RETRIEVAL_FIRST_CLAUSE": "STORES CONSULTED",
    "PRIMARY_SOURCE_RE_DERIVATION": "NEVER RECALL ONLY FROM WORKING MEMORY",
    "OPTIMAL_FORM_NO_GENERIC_BASIS": "NEVER NAIVE/TOY/GENERIC",
    "REVIEW_STATUS_CLAUSE": "fresh-eyes-reviewed(N)",
    "CITATION_CLAUSE": "authors · year · exact title · arXiv ID or DOI",
    "EIGHTFOLD_CLAUSE": "one fact, one store, one key",
    "NEVER_REASONING_ECHO": "refusal storms",
    "OWN_ROUND1_REVIEW": "resets the clean-pass counter",
    "TRIALITY_WIRING": "SAME commit batch",
    "COMMIT_DISCIPLINE": "shared-file edits use --patch-file",
    "MANUAL_CITATION": "docs/operating_manual_craft_handoff.md",
    "RISK_RANKING": "probability × blast-radius × SILENCE",
    "CONFIRMED_VS_PLAUSIBLE": "never blur them",
    "FIXES_ARE_UNREVIEWED": "round-finished ≠ clean-pass",
    "NO_MANUFACTURED_FINDINGS": "honest limits ≠ defects",
    "SECTION8_CHECKLIST": "competence-lookalike mistakes",
    "EXECUTE_DONT_READ": "not by reading commit messages",
    "AUTONOMOUS_REFORMULATION": "start of the reformulation ladder",
    "PAPER_WARM_START_FROM_DIVERGENCE": "the warm-start, not a verdict",
    "RESEARCH_AUTHORITY": "'$0 local' bounds spend only, never information",
    "DECOMPOSE_HEADLINE": "a bare composite number is UNMEASURED",
    "TIEBREAK_LEAST_COMPLEXITY": "without touching score, ALWAYS choose the least complex",
    "MASTER_THESIS_FRAMING": "formulation × realization × completeness",
    "VERDICT_SCOPE_LADDER": "INSTANCE < FORMULATION < FAMILY < PARADIGM",
    "CONTENT_LINEAGE_CRUX": "harvest-only signal, never a starting point",
    "CORRECT_OVER_EASY": "never go for easy - go for correct and true and understanding",
    "RETAINED_REASONING": "LIVE-HYPOTHESES",
    "RESEARCH_ORIGINAL_DESIGN_AUTHORITY": (
        "derive and engineer and iterate and optimize our own variants or original"
    ),
    "INTERNAL_LEVERAGE_AUTHORITY": (
        "use off the shelf or to adapt or refactor or extend or enhance"
    ),
    "CHECKPOINT_FINDINGS": "every tools/subagent_checkpoint.py write carries at least one --finding",
}


def standard_contract(*, review: bool = True, triality: bool = True) -> str:
    """Compose the standard subagent contract text for a dispatcher prompt.

    Always includes the six harvested behavior blocks (grounded progress, no ending on
    promises, final-message re-grounding, state-the-boundaries, anti-goldplating,
    fresh-context verification), the four #346 retrieval-first clauses (recursion-default,
    control laws, retrieval-first, review-status provenance), the requirement-S citation
    clause, the eight design philosophies clause (P1-P8 + clauses A/B + fmtools availability),
    the five workflow-v2 doctrine blocks, the #767 original-design-authority block, the
    retained-reasoning handoff clause (final-message live-state carry, ARC-AGI-3 crosswalk),
    the #405 commit-through-serializer discipline block, plus the operating-manual citation.

    Args:
        review: include the #337 own-round-1 adversarial-review block (default True;
            set False only for pure read-only reporting subagents that build nothing).
        triality: include the #337 triality-wiring block (default True; set False for
            subagents whose surface cannot touch levers/findings, e.g. docs-only).

    ``NEVER_REASONING_ECHO`` is deliberately NOT composed into the output: it is a
    dispatcher-side warning about how to write prompts, not an instruction for the model.
    """
    blocks = [
        GROUNDED_PROGRESS,
        NO_ENDING_ON_PROMISES,
        FINAL_MESSAGE_REGROUNDING,
        STATE_THE_BOUNDARIES,
        ANTI_GOLDPLATING,
        AUTONOMOUS_REFORMULATION,
        PAPER_WARM_START_FROM_DIVERGENCE,
        RESEARCH_AUTHORITY,
        DECOMPOSE_HEADLINE,
        TIEBREAK_LEAST_COMPLEXITY,
        MASTER_THESIS_FRAMING,
        VERDICT_SCOPE_LADDER,
        CONTENT_LINEAGE_CRUX,
        CORRECT_OVER_EASY,
        RESEARCH_ORIGINAL_DESIGN_AUTHORITY,
        INTERNAL_LEVERAGE_AUTHORITY,
        RETAINED_REASONING,
        WAITER_DISCIPLINE,
        CHECKPOINT_FINDINGS,
        FRESH_CONTEXT_VERIFIER,
        RECURSION_CLAUSE,
        CONTROL_LAW_CLAUSE,
        PRIMARY_SOURCE_RE_DERIVATION,
        OPTIMAL_FORM_NO_GENERIC_BASIS,
        RETRIEVAL_FIRST_CLAUSE,
        REVIEW_STATUS_CLAUSE,
        CITATION_CLAUSE,
        EIGHTFOLD_CLAUSE,
    ]
    if review:
        blocks.append(OWN_ROUND1_REVIEW)
    if triality:
        blocks.append(TRIALITY_WIRING)
    blocks.append(COMMIT_DISCIPLINE)
    blocks.append(MANUAL_CITATION)
    return "\n\n".join(blocks)


def review_contract(*, counter_context: str = "") -> str:
    """Compose the review-dispatch addendum (the manual's review method as structure).

    Composes the six review-only blocks (risk ranking, CONFIRMED-vs-PLAUSIBLE labeling,
    execute-don't-read, fixes-are-unreviewed counter semantics, no manufactured findings,
    the §8 ten-mistake checklist) plus grounded-progress, the final-message re-grounding,
    and the operating-manual citation. Use for REVIEW dispatches; build dispatches use
    :func:`standard_contract`.

    Args:
        counter_context: optional current clean-pass counter state for the surface under
            review (e.g. the output of ``tac.review_counter.current_state(...).describe()``)
            — prepended so the reviewer knows whether it is reviewing original code or an
            unreviewed fix round.
    """
    blocks: list[str] = []
    if counter_context.strip():
        blocks.append(f"REVIEW-COUNTER CONTEXT: {counter_context.strip()}")
    blocks.extend(
        [
            RISK_RANKING,
            CONFIRMED_VS_PLAUSIBLE,
            EXECUTE_DONT_READ,
            FIXES_ARE_UNREVIEWED,
            NO_MANUFACTURED_FINDINGS,
            SECTION8_CHECKLIST,
            GROUNDED_PROGRESS,
            FINAL_MESSAGE_REGROUNDING,
            MANUAL_CITATION,
        ]
    )
    return "\n\n".join(blocks)
