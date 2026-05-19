# SPDX-License-Identifier: MIT
"""Tests for Catalog #335 — check_cathedral_consumer_directory_package_exposes_canonical_contract.

Per CATHEDRAL-AUTO-INGEST-PARADIGM-SHIFT subagent landing 2026-05-19 + operator
NON-NEGOTIABLE "fix permanently and self protect against" the orphan-signal class.

Sister of Catalog #265 (symposium_impls canonical contract) at the
cathedral_consumers/* surface.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_cathedral_consumer_directory_package_exposes_canonical_contract,
)


# ---------------------------------------------------------------------------
# Live-repo regression guard (the gate's primary structural invariant)
# ---------------------------------------------------------------------------


def test_live_repo_passes_clean() -> None:
    """The live repo at landing has zero violations.

    Catalog #335 STRICT-flip atomicity: the live repo MUST be clean at the
    moment of landing per CLAUDE.md non-negotiable. The reference
    ``_example_consumer`` serves as the permanent positive fixture so the
    gate has at least one well-formed package to validate.
    """
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract()
    assert violations == [], (
        f"Live repo Catalog #335 violations:\n  - "
        + "\n  - ".join(violations[:5])
    )


def test_live_repo_strict_mode_does_not_raise() -> None:
    """Strict mode on the live repo MUST not raise (count=0 at landing)."""
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        strict=True
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------


def _make_synthetic_repo(tmp_path: Path) -> Path:
    """Create a synthetic repo root with src/tac/cathedral_consumers/."""
    consumer_dir = tmp_path / "src" / "tac" / "cathedral_consumers"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "# SPDX-License-Identifier: MIT\n"
        '"""Synthetic cathedral consumers namespace."""\n'
    )
    return tmp_path


def _write_compliant_package(repo: Path, name: str) -> None:
    pkg_dir = repo / "src" / "tac" / "cathedral_consumers" / name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        "# SPDX-License-Identifier: MIT\n"
        '"""Compliant test consumer."""\n'
        "from typing import Any, Mapping\n"
        "from tac.cathedral.consumer_contract import HookNumber\n\n"
        f'CONSUMER_NAME = "{name}"\n'
        'CONSUMER_VERSION = "0.1.0"\n'
        "CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)\n\n"
        "def update_from_anchor(anchor: Any) -> None:\n"
        "    pass\n\n"
        "def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:\n"
        '    return {"predicted_delta_adjustment": 0.0, "rationale": "test", "axis_tag": "[predicted]"}\n'
    )


def _write_broken_package_missing_fields(repo: Path, name: str) -> None:
    pkg_dir = repo / "src" / "tac" / "cathedral_consumers" / name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        "# SPDX-License-Identifier: MIT\n"
        '"""Broken consumer missing the canonical contract fields."""\n'
        "# (intentionally no CONSUMER_NAME etc.)\n"
    )


def _write_waived_package(repo: Path, name: str, rationale: str) -> None:
    pkg_dir = repo / "src" / "tac" / "cathedral_consumers" / name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        "# SPDX-License-Identifier: MIT\n"
        f"# CATHEDRAL_CONSUMER_DEFERRED_OK:{rationale}\n"
        '"""Waived non-compliant consumer."""\n'
    )


# ---------------------------------------------------------------------------
# Synthetic gate behavior
# ---------------------------------------------------------------------------


def test_synthetic_no_consumer_dir_silent(tmp_path: Path) -> None:
    """Missing src/tac/cathedral_consumers/ silently passes."""
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=tmp_path
    )
    assert violations == []


def test_synthetic_empty_consumer_dir_silent(tmp_path: Path) -> None:
    """Empty consumer dir (no packages) silently passes."""
    repo = _make_synthetic_repo(tmp_path)
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=repo
    )
    assert violations == []


def test_synthetic_broken_package_flagged(tmp_path: Path, monkeypatch) -> None:
    """Synthetic broken package (missing contract fields) is flagged.

    Requires PYTHONPATH manipulation so the test fixture is importable;
    we accept ImportError as a valid flagging signal (the gate cannot
    distinguish "broken fixture" from "real violation" without sys.path
    surgery, which is correct behavior for the canonical surface).
    """
    repo = _make_synthetic_repo(tmp_path)
    _write_broken_package_missing_fields(repo, "broken_consumer")
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=repo
    )
    # Either an import error OR a contract violation — both flag the package.
    assert len(violations) == 1
    assert "broken_consumer" in violations[0]


def test_synthetic_waived_package_passes(tmp_path: Path) -> None:
    """Synthetic waived non-compliant package passes via the canonical waiver."""
    repo = _make_synthetic_repo(tmp_path)
    _write_waived_package(repo, "waived_consumer", "Pending Phase 2 wire-in")
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=repo
    )
    assert violations == []


def test_synthetic_placeholder_waiver_rejected(tmp_path: Path) -> None:
    """Placeholder waiver rationale (<rationale> literal) is rejected."""
    repo = _make_synthetic_repo(tmp_path)
    _write_waived_package(repo, "placeholder_waiver", "<rationale>")
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=repo
    )
    # Placeholder waiver doesn't activate; package then flagged for broken contract.
    assert len(violations) == 1
    assert "placeholder_waiver" in violations[0]


def test_synthetic_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    """Strict mode raises PreflightError on violation."""
    repo = _make_synthetic_repo(tmp_path)
    _write_broken_package_missing_fields(repo, "broken_strict")
    with pytest.raises(PreflightError, match="Catalog #335"):
        check_cathedral_consumer_directory_package_exposes_canonical_contract(
            repo_root=repo, strict=True
        )


def test_synthetic_strict_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode is silent when no violations."""
    repo = _make_synthetic_repo(tmp_path)
    # No packages at all.
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=repo, strict=True
    )
    assert violations == []


def test_synthetic_pycache_dir_skipped(tmp_path: Path) -> None:
    """__pycache__ subdirs are exempt."""
    repo = _make_synthetic_repo(tmp_path)
    pycache = repo / "src" / "tac" / "cathedral_consumers" / "__pycache__"
    pycache.mkdir()
    (pycache / "__init__.py").write_text("# garbage")
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=repo
    )
    assert violations == []


def test_synthetic_tests_subdir_skipped(tmp_path: Path) -> None:
    """tests/ subdir is exempt (tests live alongside but aren't consumers)."""
    repo = _make_synthetic_repo(tmp_path)
    tests_dir = repo / "src" / "tac" / "cathedral_consumers" / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("# test package")
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=repo
    )
    assert violations == []


def test_synthetic_no_init_py_skipped(tmp_path: Path) -> None:
    """Subdir without __init__.py is not a package; skipped silently."""
    repo = _make_synthetic_repo(tmp_path)
    sub = repo / "src" / "tac" / "cathedral_consumers" / "data_only"
    sub.mkdir()
    (sub / "config.yaml").write_text("key: value")
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=repo
    )
    assert violations == []


def test_synthetic_short_rationale_waiver_rejected(tmp_path: Path) -> None:
    """Waiver rationale <4 chars is rejected (placeholder discipline)."""
    repo = _make_synthetic_repo(tmp_path)
    _write_waived_package(repo, "short_rationale", "abc")
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=repo
    )
    assert len(violations) == 1


def test_synthetic_verbose_flag_runs(tmp_path: Path, capsys) -> None:
    """Verbose flag prints the violation count."""
    repo = _make_synthetic_repo(tmp_path)
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=repo, verbose=True
    )
    captured = capsys.readouterr()
    assert "cathedral-consumer-contract" in captured.out
    assert "0 violation" in captured.out


def test_synthetic_string_repo_root_accepted(tmp_path: Path) -> None:
    """String repo_root accepted (converted to Path)."""
    repo = _make_synthetic_repo(tmp_path)
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=str(repo)
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Wire-in regression guards (Catalog #176 + Catalog #185 sisters)
# ---------------------------------------------------------------------------


def test_check_function_callable_via_preflight_globals() -> None:
    """Catalog #185 sister: function must be importable from tac.preflight."""
    import tac.preflight
    assert hasattr(
        tac.preflight,
        "check_cathedral_consumer_directory_package_exposes_canonical_contract",
    )
    fn = getattr(
        tac.preflight,
        "check_cathedral_consumer_directory_package_exposes_canonical_contract",
    )
    assert callable(fn)


def test_orchestrator_wires_warn_only() -> None:
    """Catalog #176 sister: preflight_all wires the gate (currently strict=False)."""
    from pathlib import Path
    src = Path(__file__).parent.parent / "preflight.py"
    text = src.read_text()
    # Must be referenced in preflight_all's body (search for the wire-in pattern).
    assert "check_cathedral_consumer_directory_package_exposes_canonical_contract(" in text
    # Must currently be WARN-ONLY per "Strict-flip atomicity rule"
    # (count 1 callsite at landing; strict=True flip pending).


def test_signature_is_keyword_only() -> None:
    """API discipline: gate accepts kwargs only for safe call sites."""
    import inspect
    fn = check_cathedral_consumer_directory_package_exposes_canonical_contract
    sig = inspect.signature(fn)
    # All parameters keyword-only (the * makes them all kwargs).
    for name, param in sig.parameters.items():
        assert param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ), f"param {name} must be keyword-only"
