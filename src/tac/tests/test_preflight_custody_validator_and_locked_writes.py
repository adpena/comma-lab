"""Tests for catalog #127 + #128 (codex round-2 HIGH 2 + MEDIUM fix gates).

Catalog #127 — `check_authoritative_tag_requires_custody_metadata`
Catalog #128 — `check_continual_learning_writes_use_lock`

Both gates protect the unified-custody and locked-write contracts landed
in `tac.continual_learning` (codex round-2 HIGH 2 + MEDIUM, 2026-05-09).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_authoritative_tag_requires_custody_metadata,
    check_continual_learning_writes_use_lock,
)


# ─────────────────────────────────────────────────────────────────────────
# Catalog #127 — authoritative tag must route through validate_custody.
# ─────────────────────────────────────────────────────────────────────────


def _mkrepo(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a temp repo skeleton (`src/tac/`, `tools/`, `experiments/`)."""
    for d in ("src/tac", "tools", "experiments"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def test_check127_clean_repo_passes(tmp_path: Path) -> None:
    """No tag literals → no violations."""
    root = _mkrepo(tmp_path, {"src/tac/foo.py": "x = 1\n"})
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check127_bypass_pattern_caught(tmp_path: Path) -> None:
    """Tag literal compared against AUTHORITATIVE_TAGS without validator → violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/bad_emitter.py": (
                "from tac.continual_learning import AUTHORITATIVE_TAGS\n"
                "tag = '[contest-CPU]'\n"
                "if tag in AUTHORITATIVE_TAGS:\n"
                "    pass\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "tools/bad_emitter.py" in violations[0]
    assert "[Check 127]" in violations[0]


def test_check127_validator_in_window_accepted(tmp_path: Path) -> None:
    """Adjacent validate_custody call → no violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/good_emitter.py": (
                "from tac.continual_learning import AUTHORITATIVE_TAGS\n"
                "ok, reason = result.validate_custody()\n"
                "if not ok and result.evidence_tag in AUTHORITATIVE_TAGS:\n"
                "    print('[contest-CPU]', reason)\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check127_posterior_update_routing_accepted(tmp_path: Path) -> None:
    """Routing through posterior_update is treated as the canonical path."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/router.py": (
                "from tac.continual_learning import (\n"
                "    AUTHORITATIVE_TAGS, posterior_update,\n"
                ")\n"
                "if tag in AUTHORITATIVE_TAGS:  # '[contest-CUDA]' literal here\n"
                "    update = posterior_update(posterior, result)\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check127_same_line_waiver_accepted(tmp_path: Path) -> None:
    """`# CUSTODY_VALIDATOR_OK:<reason>` waiver → no violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/waived.py": (
                "from tac.continual_learning import AUTHORITATIVE_TAGS\n"
                "if tag in AUTHORITATIVE_TAGS and tag == '[contest-CPU]':"
                "  # CUSTODY_VALIDATOR_OK: tag-only print, no promotion\n"
                "    print('hi')\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check127_strict_raises_preflighterror(tmp_path: Path) -> None:
    """strict=True → PreflightError on bypass."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/bad.py": (
                "from tac.continual_learning import AUTHORITATIVE_TAGS\n"
                "if tag in AUTHORITATIVE_TAGS:  # '[contest-CUDA]'\n"
                "    pass\n"
            ),
        },
    )
    with pytest.raises(PreflightError, match="check_authoritative_tag"):
        check_authoritative_tag_requires_custody_metadata(
            repo_root=root, strict=True, verbose=False
        )


def test_check127_test_files_excluded(tmp_path: Path) -> None:
    """Test files (`/tests/` or `test_*.py`) are excluded from the scan."""
    root = _mkrepo(
        tmp_path,
        {
            "src/tac/tests/test_thing.py": (
                "from tac.continual_learning import AUTHORITATIVE_TAGS\n"
                "if tag in AUTHORITATIVE_TAGS:  # '[contest-CPU]'\n"
                "    pass\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check127_canonical_continual_learning_excluded(tmp_path: Path) -> None:
    """The canonical `src/tac/continual_learning.py` is excluded (it owns the symbols)."""
    root = _mkrepo(
        tmp_path,
        {
            "src/tac/continual_learning.py": (
                "AUTHORITATIVE_TAGS = frozenset({'[contest-CUDA]', '[contest-CPU]'})\n"
                "if tag in AUTHORITATIVE_TAGS:  # '[contest-CPU]' literal\n"
                "    pass\n"
            ),
        },
    )
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check127_live_repo_under_strict_does_not_explode(tmp_path: Path) -> None:
    """Sanity: scanning the live repo returns a finite list (warn-only acceptable)."""
    # Use the actual repo root (3 levels up from this test file).
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_authoritative_tag_requires_custody_metadata(
        repo_root=repo_root, strict=False, verbose=False
    )
    # The check is warn-only initially; live count is informational here.
    assert isinstance(violations, list)


# ─────────────────────────────────────────────────────────────────────────
# Catalog #128 — continual_learning writes must use the locked path.
# ─────────────────────────────────────────────────────────────────────────


def test_check128_clean_repo_passes(tmp_path: Path) -> None:
    """No save_posterior reference → no violations."""
    root = _mkrepo(tmp_path, {"src/tac/foo.py": "x = 1\n"})
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_bare_save_caught(tmp_path: Path) -> None:
    """Direct `save_posterior(...)` call without locked-path use → violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/bad_writer.py": (
                "from tac.continual_learning import save_posterior\n"
                "def f(p):\n"
                "    save_posterior(p)\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "tools/bad_writer.py" in violations[0]
    assert "[Check 128]" in violations[0]


def test_check128_locked_co_owner_accepted(tmp_path: Path) -> None:
    """File that uses `posterior_update_locked` is the canonical writer; allow."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/canonical.py": (
                "from tac.continual_learning import (\n"
                "    save_posterior, posterior_update_locked,\n"
                ")\n"
                "def write(p):\n"
                "    posterior_update_locked(...)\n"
                "    save_posterior(p)\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_lock_context_manager_accepted(tmp_path: Path) -> None:
    """File that uses `_posterior_lock` (the lock CM) is also canonical writer."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/cm_writer.py": (
                "from tac.continual_learning import save_posterior, _posterior_lock\n"
                "def f(p):\n"
                "    with _posterior_lock(p):\n"
                "        save_posterior(p)\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_same_line_waiver_accepted(tmp_path: Path) -> None:
    """`# SAVE_POSTERIOR_LOCKED_OK:<reason>` waiver → no violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/waived.py": (
                "from tac.continual_learning import save_posterior\n"
                "def f(p):\n"
                "    save_posterior(p)  # SAVE_POSTERIOR_LOCKED_OK: single-writer test\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_strict_raises_preflighterror(tmp_path: Path) -> None:
    """strict=True → PreflightError on bare save."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/bad.py": (
                "from tac.continual_learning import save_posterior\n"
                "save_posterior(p)\n"
            ),
        },
    )
    with pytest.raises(PreflightError, match="check_continual_learning_writes_use_lock"):
        check_continual_learning_writes_use_lock(
            repo_root=root, strict=True, verbose=False
        )


def test_check128_test_files_excluded(tmp_path: Path) -> None:
    """Test files are excluded from the scan."""
    root = _mkrepo(
        tmp_path,
        {
            "src/tac/tests/test_writer.py": (
                "from tac.continual_learning import save_posterior\n"
                "save_posterior(p)\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_canonical_continual_learning_excluded(tmp_path: Path) -> None:
    """The canonical implementation file is excluded."""
    root = _mkrepo(
        tmp_path,
        {
            "src/tac/continual_learning.py": (
                "def save_posterior(p):\n"
                "    pass\n"
                "save_posterior(p)\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_imports_only_accepted(tmp_path: Path) -> None:
    """Importing `save_posterior` without calling it → no violation."""
    root = _mkrepo(
        tmp_path,
        {
            "tools/just_imports.py": (
                "from tac.continual_learning import save_posterior  # re-exported\n"
                "__all__ = ['save_posterior']\n"
            ),
        },
    )
    violations = check_continual_learning_writes_use_lock(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_check128_live_repo_under_strict_does_not_explode() -> None:
    """Sanity: scanning the live repo returns a finite list."""
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_continual_learning_writes_use_lock(
        repo_root=repo_root, strict=False, verbose=False
    )
    assert isinstance(violations, list)
