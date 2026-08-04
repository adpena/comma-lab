# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from pathlib import Path

import pytest

import tac.canonical_equations.einstein_kolmogorov_crux_20260719 as crux_equation


def _canonical_file(repo_root: Path) -> tuple[str, Path]:
    relpath = "evidence/measurement.json"
    path = repo_root / relpath
    path.parent.mkdir(parents=True)
    path.write_text("canonical", encoding="utf-8")
    return relpath, path


def test_canonical_producer_reference_accepts_exact_regular_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    relpath, canonical = _canonical_file(repo_root)
    monkeypatch.setattr(crux_equation, "REPO_ROOT", repo_root)

    result_path, result_label = crux_equation._canonical_producer_reference(relpath, canonical_path=relpath)

    assert result_path == canonical.resolve()
    assert result_label == relpath


def test_canonical_producer_reference_rejects_caller_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    relpath, canonical = _canonical_file(repo_root)
    alias = tmp_path / "measurement-copy.json"
    alias.write_bytes(canonical.read_bytes())
    monkeypatch.setattr(crux_equation, "REPO_ROOT", repo_root)

    with pytest.raises(ValueError, match="must resolve"):
        crux_equation._canonical_producer_reference(alias, canonical_path=relpath)


def test_canonical_producer_reference_rejects_parent_traversal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    relpath, _canonical = _canonical_file(repo_root)
    monkeypatch.setattr(crux_equation, "REPO_ROOT", repo_root)

    with pytest.raises(ValueError, match="parent traversal"):
        crux_equation._canonical_producer_reference("../measurement.json", canonical_path=relpath)

    with pytest.raises(ValueError, match="parent traversal"):
        crux_equation._canonical_producer_reference(relpath, canonical_path="../measurement.json")


def test_canonical_producer_reference_rejects_symlinked_canonical_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    relpath = "evidence/measurement.json"
    canonical = repo_root / relpath
    canonical.parent.mkdir(parents=True)
    target = tmp_path / "measurement-target.json"
    target.write_text("canonical", encoding="utf-8")
    try:
        canonical.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(crux_equation, "REPO_ROOT", repo_root)

    with pytest.raises(ValueError, match="symlinked component"):
        crux_equation._canonical_producer_reference(relpath, canonical_path=relpath)


def test_canonical_producer_reference_rejects_hardlinked_canonical_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    relpath = "evidence/measurement.json"
    canonical = repo_root / relpath
    canonical.parent.mkdir(parents=True)
    target = tmp_path / "measurement-target.json"
    target.write_text("canonical", encoding="utf-8")
    try:
        os.link(target, canonical)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")
    monkeypatch.setattr(crux_equation, "REPO_ROOT", repo_root)

    with pytest.raises(ValueError, match="exactly one hard link"):
        crux_equation._canonical_producer_reference(relpath, canonical_path=relpath)

