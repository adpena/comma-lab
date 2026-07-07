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
    "CONTRACT_CONSTANT_NAMES",
    "FINAL_MESSAGE_REGROUNDING",
    "FRESH_CONTEXT_VERIFIER",
    "GROUNDED_PROGRESS",
    "KEY_PHRASES",
    "MANUAL_CITATION",
    "NEVER_REASONING_ECHO",
    "NO_ENDING_ON_PROMISES",
    "OWN_ROUND1_REVIEW",
    "STATE_THE_BOUNDARIES",
    "TRIALITY_WIRING",
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
