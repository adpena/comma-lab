"""Tests for the harvest-engineered prompting gates (2026-07-07):

- check_no_reasoning_echo_instructions (refusal-storm prevention)
- check_subagent_contract_module_integrity (contract-module anti-rot)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.preflight import (
    PreflightError,
    check_no_reasoning_echo_instructions,
    check_subagent_contract_module_integrity,
)


def _make_repo(tmp_path: Path, doc_text: str) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "sample.md").write_text(doc_text, encoding="utf-8")
    return tmp_path


# --- check_no_reasoning_echo_instructions -----------------------------------------------------


def test_reasoning_echo_positive_catches_instruction(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, "Please show your thinking step by step.\n")
    violations = check_no_reasoning_echo_instructions(repo_root=root)
    assert len(violations) == 1
    assert "show your thinking" in violations[0]
    assert "docs/sample.md:1" in violations[0]


def test_reasoning_echo_negated_context_exempt(tmp_path: Path) -> None:
    root = _make_repo(
        tmp_path,
        "Never show your thinking in prompts.\n"
        "Don't ask a model to transcribe your reasoning.\n"
        "Avoid telling it to reproduce your chain of thought.\n",
    )
    assert check_no_reasoning_echo_instructions(repo_root=root) == []


def test_reasoning_echo_waiver_respected(tmp_path: Path) -> None:
    root = _make_repo(
        tmp_path,
        "show your thinking  # REASONING_ECHO_OK:documented-anti-pattern-example\n",
    )
    assert check_no_reasoning_echo_instructions(repo_root=root) == []


def test_reasoning_echo_placeholder_waiver_rejected(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, "show your thinking  # REASONING_ECHO_OK:<rationale>\n")
    violations = check_no_reasoning_echo_instructions(repo_root=root)
    assert len(violations) == 1


def test_reasoning_echo_scans_prompts_and_skills_dirs(tmp_path: Path) -> None:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "agent.md").write_text(
        "Then echo your internal reasoning back.\n", encoding="utf-8")
    skills = tmp_path / ".claude" / "skills" / "demo"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "transcribe your reasoning into the answer\n", encoding="utf-8")
    violations = check_no_reasoning_echo_instructions(repo_root=tmp_path)
    assert len(violations) == 2


def test_reasoning_echo_placeholder_reason_waiver_rejected(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, "show your thinking  # REASONING_ECHO_OK:<reason>\n")
    assert len(check_no_reasoning_echo_instructions(repo_root=root)) == 1


def test_reasoning_echo_contract_module_is_in_scope(tmp_path: Path) -> None:
    """A scope regression dropping src/tac/subagent_contract.py must not pass silently."""
    contract = tmp_path / "src" / "tac"
    contract.mkdir(parents=True)
    (contract / "subagent_contract.py").write_text(
        'X = "please echo your internal reasoning"\n', encoding="utf-8")
    violations = check_no_reasoning_echo_instructions(repo_root=tmp_path)
    assert len(violations) == 1
    assert "src/tac/subagent_contract.py:1" in violations[0]


def test_reasoning_echo_strict_raises(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, "reproduce your chain of thought verbatim\n")
    with pytest.raises(PreflightError):
        check_no_reasoning_echo_instructions(repo_root=root, strict=True)


def test_reasoning_echo_live_repo_clean() -> None:
    """Live count must be 0 on the real repo (strict-flip precondition)."""
    assert check_no_reasoning_echo_instructions() == []


# --- check_subagent_contract_module_integrity -------------------------------------------------


def test_contract_integrity_live_repo_clean() -> None:
    assert check_subagent_contract_module_integrity() == []


def test_contract_integrity_missing_file_flagged(tmp_path: Path) -> None:
    violations = check_subagent_contract_module_integrity(repo_root=tmp_path)
    assert len(violations) == 1
    assert "MISSING" in violations[0]


def test_contract_integrity_broken_module_flagged_and_strict_raises() -> None:
    broken = SimpleNamespace(
        KEY_PHRASES={},
        GROUNDED_PROGRESS="",  # empty -> violation
        # every other required constant missing -> violations
        standard_contract=lambda **kw: "no grounding here",
    )
    violations = check_subagent_contract_module_integrity(_module=broken)
    assert any("GROUNDED_PROGRESS" in v for v in violations)
    assert any("grounded-progress phrase" in v for v in violations)
    with pytest.raises(PreflightError):
        check_subagent_contract_module_integrity(_module=broken, strict=True)


def test_contract_integrity_requires_review_constants_and_composer() -> None:
    # A module missing the review_contract composer + review constants is flagged
    # (2026-07-07 extension: the review method must stay structural).
    broken = SimpleNamespace(
        KEY_PHRASES={},
        standard_contract=lambda **kw: "x",
    )
    violations = check_subagent_contract_module_integrity(_module=broken)
    assert any("RISK_RANKING" in v for v in violations)
    assert any("SECTION8_CHECKLIST" in v for v in violations)
    assert any("review_contract composer missing" in v for v in violations)


def test_contract_integrity_review_composer_lost_risk_phrase_flagged() -> None:
    import tac.subagent_contract as real

    drifted = SimpleNamespace(
        KEY_PHRASES=real.KEY_PHRASES,
        standard_contract=real.standard_contract,
        review_contract=lambda **kw: "a review addendum without the ranking rule",
        **{name: getattr(real, name) for name in real.CONTRACT_CONSTANT_NAMES},
    )
    violations = check_subagent_contract_module_integrity(_module=drifted)
    assert any("risk-ranking phrase" in v for v in violations)
