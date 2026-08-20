# SPDX-License-Identifier: MIT
"""Fail-closed controls for the Vertigo certify-or-block mover."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load():
    path = REPO / "tools" / "vertigo_certify_move.py"
    spec = importlib.util.spec_from_file_location("_rvf1_vertigo_certify_move", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mover():
    return _load()


def test_census_refuses_unreadable_file_instead_of_omitting_it(mover, tmp_path, monkeypatch):
    root = tmp_path / "tree"
    root.mkdir()
    victim = root / "payload.bin"
    victim.write_bytes(b"retained payload")
    real_stat = Path.stat

    def fail_victim(path, *args, **kwargs):
        if path == victim:
            raise PermissionError("injected unreadable file")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_victim)
    with pytest.raises(mover.CensusError, match=r"payload\.bin"):
        mover.take_census(root)


def test_census_refuses_unreadable_directory_from_os_walk(mover, tmp_path, monkeypatch):
    root = tmp_path / "tree"
    root.mkdir()

    def walk_with_error(_root, *, followlinks, onerror):
        assert followlinks is False
        onerror(PermissionError("injected scandir refusal"))
        yield from ()

    monkeypatch.setattr(mover.os, "walk", walk_with_error)
    with pytest.raises(mover.CensusError, match="unreadable directory"):
        mover.take_census(root)


def test_census_counts_directory_symlinks_without_following_them(mover, tmp_path):
    root = tmp_path / "tree"
    target = tmp_path / "target"
    root.mkdir()
    target.mkdir()
    (target / "outside.bin").write_bytes(b"outside")
    (root / "linkdir").symlink_to(target, target_is_directory=True)

    census = mover.take_census(root)

    assert census.n_symlinks == 1
    assert census.files == []


def test_apply_refuses_unaccounted_reference_surface(mover, monkeypatch, capsys):
    """An empty referenced_by field cannot silently enter a destructive plan."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vertigo_certify_move.py",
            "--source", "/Volumes/VertigoDataTier/pact/example",
            "--dest-root", "/Volumes/APDataStore/pact/vertigo_coldstore",
            "--ledger", "/tmp/rvf1-ledger.jsonl",
            "--category", "rebuildable",
            "--reason", "deterministically rebuildable fixture",
            "--apply",
        ],
    )
    assert mover.main() == 2
    assert "requires --referenced-by" in capsys.readouterr().err
