"""The subagent contract's own ``## DEAD-ENDS`` header is a label, not a verdict.

MEASURED 2026-08-22: 198 of 372 arm finals carry the header that
src/tac/subagent_contract.py:528 instructs every arm to emit, and 194 of those were
flagged by the verdict-scope leg on the header ALONE — one template counted 194 times
(#821's one-fact-counted-N-times genus). The case-sensitivity heuristic guarding
"dead code" prose cannot help: the header is legitimately uppercase.

The cure is deliberately ANCHORED to the bare header line rather than exempting the
token or the arm_final_messages directory — a blanket exemption would be the
vacuous-gate trap (silencing a surface to quiet a gate).
"""

from __future__ import annotations

import sys

sys.path.insert(0, "tools")

from triality_drift_detector import _verdict_line_exempt, negative_verdict_tokens


def test_bare_template_header_is_exempt() -> None:
    assert _verdict_line_exempt("## DEAD-ENDS")
    assert negative_verdict_tokens(["## DEAD-ENDS"])[0] == []


def test_underscore_and_depth_variants_of_the_bare_header() -> None:
    for header in ("# DEAD-ENDS", "### DEAD_ENDS", "##   DEAD-ENDS   "):
        assert _verdict_line_exempt(header), header


def test_a_heading_that_IS_a_claim_still_fires() -> None:
    # NEGATIVE DIRECTION: the cure must not become a blanket DEAD suppression.
    toks, _ = negative_verdict_tokens(["## DEAD-END: the phase-field family"])
    assert toks, "a heading carrying a claim must still be scanned"


def test_header_with_extra_words_is_not_the_template() -> None:
    toks, _ = negative_verdict_tokens(["## DEAD-ENDS AND OPEN ROWS"])
    assert toks, "only the bare template header is exempt"


def test_prose_claims_still_fire() -> None:
    for claim in (
        "- The chart-symbol family is DEAD on the measured reach.",
        "The lattice family is DEAD.",
        "Verdict: NO-GO for the band family.",
    ):
        toks, _ = negative_verdict_tokens([claim])
        assert toks, f"must still fire: {claim!r}"


def test_kill_class_detection_survives_the_exemption() -> None:
    # A KILL-class token on a real line still sets the kill flag (req R rule (b)).
    _, kill = negative_verdict_tokens(["## DEAD-ENDS", "The X family is KILLED."])
    assert kill, "kill-class detection must survive an exempt header in the same doc"
