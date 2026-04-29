"""Regression tests for Round 3 council-prescribed preflight checks 82+83.

Check 82 — check_callsite_contracts_satisfied: catches the Lane GP class of
bug where a kwarg lands in a helper but the actual call site never passes it.

Check 83 — check_no_proxy_metric_drives_decision: catches the STC class of
bug where a kill/promote decision is made from MPS/CPU evidence without a
contest-CUDA artifact in the same paragraph.

Memory: feedback_three_active_bug_classes_needing_strict_checks_20260429.md
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    CALLSITE_CONTRACTS,
    MetaBugViolation,
    _check_mps_decision_in_text,
    _scan_python_for_callsite_contract_violations,
    check_callsite_contracts_satisfied,
    check_no_proxy_metric_drives_decision,
)


# ── Check 82 (callsite contracts) ────────────────────────────────────────────


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content))
    return p


def test_callsite_contract_satisfied_when_kwarg_present(tmp_path: Path) -> None:
    src = tmp_path / "experiments"
    src.mkdir()
    p = _write(
        tmp_path, "experiments/good.py",
        """
        from tac.pose_gaussian_process import reconstruct_poses
        x = reconstruct_poses(model, 600, baseline_poses=baseline)
        """,
    )
    violations = _scan_python_for_callsite_contract_violations(p, tmp_path)
    assert violations == []


def test_callsite_contract_violated_when_kwarg_missing(tmp_path: Path) -> None:
    p = _write(
        tmp_path, "experiments/bad.py",
        """
        from tac.pose_gaussian_process import reconstruct_poses
        x = reconstruct_poses(model, 600)
        """,
    )
    violations = _scan_python_for_callsite_contract_violations(p, tmp_path)
    assert len(violations) == 1
    assert "baseline_poses" in violations[0]
    assert "experiments/bad.py" in violations[0]


def test_callsite_contract_kwargs_splat_treated_as_satisfied(tmp_path: Path) -> None:
    """A **kwargs splat is opaque — assume the contract may be satisfied
    rather than fire a false positive."""
    p = _write(
        tmp_path, "experiments/splat.py",
        """
        from tac.pose_gaussian_process import reconstruct_poses
        x = reconstruct_poses(model, 600, **kwargs)
        """,
    )
    violations = _scan_python_for_callsite_contract_violations(p, tmp_path)
    assert violations == []


def test_callsite_contract_attribute_access_form(tmp_path: Path) -> None:
    """`mod.reconstruct_poses(...)` should be detected just like the
    bare-name form."""
    p = _write(
        tmp_path, "experiments/attr.py",
        """
        import tac.pose_gaussian_process as gp
        x = gp.reconstruct_poses(model, 600)
        """,
    )
    violations = _scan_python_for_callsite_contract_violations(p, tmp_path)
    assert len(violations) == 1
    assert "reconstruct_poses" in violations[0]


def test_callsite_contract_repo_clean() -> None:
    """The actual repo must have 0 violations; this test fails if any
    callsite stops passing the registered kwargs."""
    violations = check_callsite_contracts_satisfied(strict=False)
    assert violations == [], (
        f"Repo has {len(violations)} callsite-contract violation(s):\n"
        + "\n".join(f"  • {v}" for v in violations)
    )


def test_callsite_contract_strict_mode_raises(tmp_path: Path) -> None:
    """In strict mode, a violation must raise MetaBugViolation."""
    _write(
        tmp_path, "experiments/bad.py",
        """
        from tac.pose_gaussian_process import reconstruct_poses
        x = reconstruct_poses(model, 600)
        """,
    )
    with pytest.raises(MetaBugViolation, match="CALLSITE-CONTRACT"):
        check_callsite_contracts_satisfied(
            strict=True, verbose=False, repo_root=tmp_path,
        )


def test_callsite_contracts_registry_has_known_entries() -> None:
    """Sanity: registry must include the Lane GP entry that motivated
    the check. If someone removes it without leaving an audit comment,
    this fails loud."""
    assert "reconstruct_poses" in CALLSITE_CONTRACTS
    assert "baseline_poses" in CALLSITE_CONTRACTS["reconstruct_poses"]


# ── Check 83 (no MPS-driven strategic decisions) ─────────────────────────────


def test_mps_decision_with_contest_cuda_tag_is_clean() -> None:
    text = """
    Lane STC dispatched on Modal T4 [contest-CUDA]
    promoted to GREEN after MPS smoke + CUDA auth eval.
    """
    violations = _check_mps_decision_in_text(text, "test.md")
    assert violations == []


def test_mps_decision_without_contest_cuda_tag_is_violation() -> None:
    text = """
    Lane STC FALSIFIED based on local MPS encoder argmax.
    The result was 21MB vs Lane A 421KB — kill the lane.
    """
    violations = _check_mps_decision_in_text(text, "test.md")
    assert len(violations) >= 1
    assert any("FALSIFIED" in v or "MPS" in v for v in violations)


def test_post_mortem_tag_exempts_otherwise_violating_paragraph() -> None:
    """A WITHDRAWN / POST-MORTEM tag in the same paragraph documents
    the rule (rather than violating it)."""
    text = """
    Lane STC FALSIFICATION WITHDRAWN — original FALSIFIED verdict was
    based on MPS encoder, which is unfit for strategic decisions.
    """
    violations = _check_mps_decision_in_text(text, "test.md")
    assert violations == []


def test_pure_mps_mention_without_decision_verb_is_clean() -> None:
    """Just mentioning MPS without a kill/promote/falsify verb is fine."""
    text = """
    Local smoke test runs on MPS in 3 minutes.
    The CUDA auth eval is queued.
    """
    violations = _check_mps_decision_in_text(text, "test.md")
    assert violations == []


def test_repo_no_proxy_decision_violations_clean() -> None:
    """The actual repo must have 0 violations after the docs/hardware_layout
    cleanup. If a future write reintroduces an MPS-derived decision, this
    test fails."""
    violations = check_no_proxy_metric_drives_decision(strict=False)
    assert violations == [], (
        f"Repo has {len(violations)} no-MPS-decision violation(s):\n"
        + "\n".join(f"  • {v}" for v in violations[:10])
    )


def test_no_mps_decision_strict_mode_raises(tmp_path: Path) -> None:
    """A planted bad doc should raise in strict mode."""
    docs = tmp_path / "docs"
    docs.mkdir()
    _write(
        tmp_path, "docs/bad.md",
        """
        ## Verdict
        Lane STC FALSIFIED based on local MPS scorer eval.
        Recommend KILL.
        """,
    )
    with pytest.raises(MetaBugViolation, match="MPS-DERIVED"):
        check_no_proxy_metric_drives_decision(
            strict=True, verbose=False, repo_root=tmp_path,
        )
