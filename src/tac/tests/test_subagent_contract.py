"""Tests for tac.subagent_contract — the canonical dispatcher prompt contract.

Sources: docs/harvest_fable5_prompting_and_loops_20260707.md +
docs/operating_manual_craft_handoff.md. Each constant must keep its verbatim-grade
key phrase; the composer must include/exclude the #337 blocks correctly.
"""

from __future__ import annotations

import pytest

from tac import subagent_contract as sc


@pytest.mark.parametrize("name", sc.CONTRACT_CONSTANT_NAMES)
def test_constant_non_empty_and_contains_key_phrase(name: str) -> None:
    value = getattr(sc, name)
    assert isinstance(value, str) and value.strip(), f"{name} must be a non-empty string"
    phrase = sc.KEY_PHRASES[name]
    assert phrase in value, f"{name} lost its key phrase {phrase!r}"


def test_key_phrases_registry_covers_exactly_the_constants() -> None:
    assert set(sc.KEY_PHRASES) == set(sc.CONTRACT_CONSTANT_NAMES)


def test_standard_contract_default_includes_all_blocks() -> None:
    composed = sc.standard_contract()
    for name in sc.CONTRACT_CONSTANT_NAMES:
        if name == "NEVER_REASONING_ECHO":
            continue  # dispatcher-side warning, never composed into model instructions
        assert getattr(sc, name) in composed, f"default contract must include {name}"


def test_standard_contract_never_composes_reasoning_echo_warning() -> None:
    composed = sc.standard_contract()
    assert sc.NEVER_REASONING_ECHO not in composed
    assert "refusal storms" not in composed


def test_standard_contract_review_false_excludes_own_round1() -> None:
    composed = sc.standard_contract(review=False)
    assert sc.OWN_ROUND1_REVIEW not in composed
    assert sc.TRIALITY_WIRING in composed
    assert sc.GROUNDED_PROGRESS in composed


def test_standard_contract_triality_false_excludes_triality_wiring() -> None:
    composed = sc.standard_contract(triality=False)
    assert sc.TRIALITY_WIRING not in composed
    assert sc.OWN_ROUND1_REVIEW in composed


def test_standard_contract_both_false_keeps_core_blocks() -> None:
    composed = sc.standard_contract(review=False, triality=False)
    assert sc.OWN_ROUND1_REVIEW not in composed
    assert sc.TRIALITY_WIRING not in composed
    for name in (
        "GROUNDED_PROGRESS",
        "NO_ENDING_ON_PROMISES",
        "FINAL_MESSAGE_REGROUNDING",
        "STATE_THE_BOUNDARIES",
        "ANTI_GOLDPLATING",
        "FRESH_CONTEXT_VERIFIER",
        "MANUAL_CITATION",
    ):
        assert getattr(sc, name) in composed, f"core block {name} must always compose"


def test_standard_contract_grounding_phrase_always_present() -> None:
    for kwargs in ({}, {"review": False}, {"triality": False},
                   {"review": False, "triality": False}):
        composed = sc.standard_contract(**kwargs)
        assert "Before reporting progress, audit each claim against a tool result" in composed


def test_standard_contract_blocks_separated_by_blank_lines() -> None:
    composed = sc.standard_contract()
    # 9 blocks by default -> 8 separators.
    assert composed.count("\n\n") == 8


def test_never_reasoning_echo_phrases_only_in_negated_lines() -> None:
    """The warning constant must keep every echo phrase on a negated source line
    (never/don't) so the sister preflight gate reads it as guidance, not violation."""
    negations = ("never", "don't", "do not", "avoid", "forbidden")
    phrases = (
        "show your thinking",
        "transcribe your reasoning",
        "reproduce your chain of thought",
        "echo your internal reasoning",
    )
    for line in sc.NEVER_REASONING_ECHO.splitlines():
        lowered = line.lower()
        if any(p in lowered for p in phrases):
            assert any(tok in lowered for tok in negations), (
                f"echo phrase on non-negated line: {line!r}"
            )
