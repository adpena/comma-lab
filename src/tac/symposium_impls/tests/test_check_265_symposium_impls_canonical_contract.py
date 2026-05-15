# SPDX-License-Identifier: MIT
"""Tests for Catalog #265 META gate ``check_symposium_impls_canonical_contract``.

Per the Grand Reunion symposium 2026-05-15 + operator mandate to implement
fully + correctly + rigorously to spec.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_symposium_impls_canonical_contract,
)


_LIVE_REPO_ROOT = Path(__file__).resolve().parents[4]


# ----- live repo regression guard --------------------------------------------------------------


def test_live_repo_has_zero_violations() -> None:
    """STRICT-from-byte-one regression guard: package contract holds."""
    violations = check_symposium_impls_canonical_contract(
        repo_root=_LIVE_REPO_ROOT, strict=False
    )
    assert violations == [], f"unexpected violations:\n  " + "\n  ".join(
        v[:200] for v in violations
    )


def test_live_repo_strict_does_not_raise() -> None:
    check_symposium_impls_canonical_contract(repo_root=_LIVE_REPO_ROOT, strict=True)


# ----- canonical contract enforcement ---------------------------------------------------------


def _build_minimal_compliant_module(tmp_path: Path, filename: str) -> Path:
    """Write a minimal module that satisfies all 5 canonical-contract tokens."""
    package_dir = tmp_path / "src" / "tac" / "symposium_impls"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("# __init__ exempted from contract\n")
    body = (
        "# SPDX-License-Identifier: MIT\n"
        '"""[verified-against: cited canonical reference] Catalog #999 minimal compliant module."""\n'
        "from __future__ import annotations\n"
        "\n"
        "__all__ = ('update_from_anchor',)\n"
        "\n"
        "def update_from_anchor(anchor):\n"
        "    return None\n"
    )
    target = package_dir / filename
    target.write_text(body)
    return target


def test_compliant_module_passes(tmp_path: Path) -> None:
    _build_minimal_compliant_module(tmp_path, "compliant.py")
    violations = check_symposium_impls_canonical_contract(repo_root=tmp_path, strict=False)
    assert violations == []


def test_init_py_is_self_exempt(tmp_path: Path) -> None:
    """__init__.py is exempt: package-level metadata-only file."""
    package_dir = tmp_path / "src" / "tac" / "symposium_impls"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("# stripped of contract tokens\n")
    violations = check_symposium_impls_canonical_contract(repo_root=tmp_path, strict=False)
    assert violations == []


def test_missing_spdx_header_flagged(tmp_path: Path) -> None:
    package_dir = tmp_path / "src" / "tac" / "symposium_impls"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("\n")
    body = (
        '"""[verified-against: ref] Catalog #999 missing SPDX."""\n'
        "from __future__ import annotations\n"
        "__all__ = ('update_from_anchor',)\n"
        "def update_from_anchor(anchor): return None\n"
    )
    (package_dir / "no_spdx.py").write_text(body)
    violations = check_symposium_impls_canonical_contract(repo_root=tmp_path, strict=False)
    assert len(violations) == 1
    assert "no_spdx.py" in violations[0]
    assert "SPDX-License-Identifier" in violations[0]


def test_missing_all_export_flagged(tmp_path: Path) -> None:
    package_dir = tmp_path / "src" / "tac" / "symposium_impls"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("\n")
    # Note: docstring deliberately avoids the literal `__all__` token
    # so the substring-match contract check correctly flags this fixture.
    body = (
        "# SPDX-License-Identifier: MIT\n"
        '"""[verified-against: ref] Catalog #999 missing public-api export."""\n'
        "def update_from_anchor(anchor): return None\n"
    )
    (package_dir / "no_all.py").write_text(body)
    violations = check_symposium_impls_canonical_contract(repo_root=tmp_path, strict=False)
    assert len(violations) == 1
    assert "__all__" in violations[0]


def test_missing_update_from_anchor_flagged(tmp_path: Path) -> None:
    package_dir = tmp_path / "src" / "tac" / "symposium_impls"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("\n")
    body = (
        "# SPDX-License-Identifier: MIT\n"
        '"""[verified-against: ref] Catalog #999 missing continual hook."""\n'
        "__all__ = ('foo',)\n"
        "def foo(): return None\n"
    )
    (package_dir / "no_hook.py").write_text(body)
    violations = check_symposium_impls_canonical_contract(repo_root=tmp_path, strict=False)
    assert len(violations) == 1
    assert "update_from_anchor" in violations[0]


def test_missing_verified_against_citation_flagged(tmp_path: Path) -> None:
    package_dir = tmp_path / "src" / "tac" / "symposium_impls"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("\n")
    body = (
        "# SPDX-License-Identifier: MIT\n"
        '"""Catalog #999 missing math citation."""\n'
        "__all__ = ('update_from_anchor',)\n"
        "def update_from_anchor(anchor): return None\n"
    )
    (package_dir / "no_cite.py").write_text(body)
    violations = check_symposium_impls_canonical_contract(repo_root=tmp_path, strict=False)
    assert len(violations) == 1
    assert "verified-against" in violations[0]


def test_missing_catalog_citation_flagged(tmp_path: Path) -> None:
    package_dir = tmp_path / "src" / "tac" / "symposium_impls"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("\n")
    body = (
        "# SPDX-License-Identifier: MIT\n"
        '"""[verified-against: ref] Module missing catalog #."""\n'
        "__all__ = ('update_from_anchor',)\n"
        "def update_from_anchor(anchor): return None\n"
    )
    (package_dir / "no_catalog.py").write_text(body)
    violations = check_symposium_impls_canonical_contract(repo_root=tmp_path, strict=False)
    assert len(violations) == 1
    assert "Catalog #" in violations[0]


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    package_dir = tmp_path / "src" / "tac" / "symposium_impls"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("\n")
    (package_dir / "broken.py").write_text("# nothing\n")
    with pytest.raises(PreflightError) as exc:
        check_symposium_impls_canonical_contract(repo_root=tmp_path, strict=True)
    assert "Catalog #265" in str(exc.value)


def test_no_package_dir_silent_skip(tmp_path: Path) -> None:
    violations = check_symposium_impls_canonical_contract(repo_root=tmp_path, strict=True)
    assert violations == []


def test_string_repo_root_accepted(tmp_path: Path) -> None:
    violations = check_symposium_impls_canonical_contract(repo_root=str(tmp_path), strict=False)
    assert violations == []


def test_multiple_violations_aggregate(tmp_path: Path) -> None:
    package_dir = tmp_path / "src" / "tac" / "symposium_impls"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("\n")
    (package_dir / "broken_a.py").write_text("# nothing\n")
    (package_dir / "broken_b.py").write_text("# nothing\n")
    violations = check_symposium_impls_canonical_contract(repo_root=tmp_path, strict=False)
    assert len(violations) == 2


def test_compliant_and_broken_mix(tmp_path: Path) -> None:
    _build_minimal_compliant_module(tmp_path, "compliant.py")
    package_dir = tmp_path / "src" / "tac" / "symposium_impls"
    (package_dir / "broken.py").write_text("# missing tokens\n")
    violations = check_symposium_impls_canonical_contract(repo_root=tmp_path, strict=False)
    assert len(violations) == 1
    assert "broken.py" in violations[0]


def test_verbose_mode_does_not_raise(tmp_path: Path) -> None:
    _build_minimal_compliant_module(tmp_path, "compliant.py")
    violations = check_symposium_impls_canonical_contract(
        repo_root=tmp_path, strict=True, verbose=True
    )
    assert violations == []
