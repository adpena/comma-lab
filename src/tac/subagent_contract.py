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
    "CONFIRMED_VS_PLAUSIBLE",
    "CONTRACT_CONSTANT_NAMES",
    "EXECUTE_DONT_READ",
    "FINAL_MESSAGE_REGROUNDING",
    "FIXES_ARE_UNREVIEWED",
    "FRESH_CONTEXT_VERIFIER",
    "GROUNDED_PROGRESS",
    "KEY_PHRASES",
    "MANUAL_CITATION",
    "NEVER_REASONING_ECHO",
    "NO_ENDING_ON_PROMISES",
    "NO_MANUFACTURED_FINDINGS",
    "OWN_ROUND1_REVIEW",
    "REVIEW_ONLY_CONSTANT_NAMES",
    "RISK_RANKING",
    "SECTION8_CHECKLIST",
    "STATE_THE_BOUNDARIES",
    "TRIALITY_WIRING",
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
    "NEVER_REASONING_ECHO",
    "OWN_ROUND1_REVIEW",
    "TRIALITY_WIRING",
    "MANUAL_CITATION",
    "RISK_RANKING",
    "CONFIRMED_VS_PLAUSIBLE",
    "FIXES_ARE_UNREVIEWED",
    "NO_MANUFACTURED_FINDINGS",
    "SECTION8_CHECKLIST",
    "EXECUTE_DONT_READ",
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
    "NEVER_REASONING_ECHO": "refusal storms",
    "OWN_ROUND1_REVIEW": "resets the clean-pass counter",
    "TRIALITY_WIRING": "SAME commit batch",
    "MANUAL_CITATION": "docs/operating_manual_craft_handoff.md",
    "RISK_RANKING": "probability × blast-radius × SILENCE",
    "CONFIRMED_VS_PLAUSIBLE": "never blur them",
    "FIXES_ARE_UNREVIEWED": "round-finished ≠ clean-pass",
    "NO_MANUFACTURED_FINDINGS": "honest limits ≠ defects",
    "SECTION8_CHECKLIST": "competence-lookalike mistakes",
    "EXECUTE_DONT_READ": "not by reading commit messages",
}


def standard_contract(*, review: bool = True, triality: bool = True) -> str:
    """Compose the standard subagent contract text for a dispatcher prompt.

    Always includes the six harvested behavior blocks (grounded progress, no ending on
    promises, final-message re-grounding, state-the-boundaries, anti-goldplating,
    fresh-context verification) plus the operating-manual citation.

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
        FRESH_CONTEXT_VERIFIER,
    ]
    if review:
        blocks.append(OWN_ROUND1_REVIEW)
    if triality:
        blocks.append(TRIALITY_WIRING)
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
