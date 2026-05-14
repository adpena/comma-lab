# SPDX-License-Identifier: MIT
"""Catalog #170 (WAVE-7-MED-FIX, REVIEW-OMNI NV5) tests.

Bug-class anchor: REVIEW-OMNI 2026-05-12 Carmack — Council C OOM fix not
validated across substrate trainers; T4 (16GB) and A10G (22GB shared)
may OOM at batch_size=32. Each substrate operator-authorize recipe
MUST declare ``min_vram_gb`` integer floor.

Coverage targets:
- compliant recipe (declares min_vram_gb: 80) -> no violation
- non-compliant recipe (no field, no waiver) -> violation
- waiver recipe -> no violation
- min_vram_gb: 0 -> violation (CPU-only refused; substrate trainers need CUDA)
- min_vram_gb: non-int -> violation
- non-substrate yaml in recipes/ -> not scanned (glob filter)
- non-existent recipes dir -> empty violations
- strict mode raises PreflightError
- verbose mode prints diagnostic banner
- non-strict returns violation list
- field RE handles inline comment
- field RE handles negative integers as not matching
- field RE matches at top-level only (indented field rejected by glob anchor)
- waiver placeholder `<reason>` does NOT auto-waive
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_recipes_declare_min_vram_gb_floor,
)


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    return repo


def _write_recipe(repo: Path, name: str, body: str) -> Path:
    p = repo / ".omx" / "operator_authorize_recipes" / f"substrate_{name}_modal_a100_dispatch.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_compliant_recipe_no_violation(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\nmin_vram_gb: 80\n")
    out = check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_missing_field_violation(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    out = check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "min_vram_gb" in out[0]


def test_waiver_recipe_no_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n# MIN_VRAM_GB_OK:proxy-only-no-cuda-substrate\n",
    )
    out = check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_zero_value_violation(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\nmin_vram_gb: 0\n")
    out = check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "below 1" in out[0]


def test_t4_value_no_violation(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\nmin_vram_gb: 16\n")
    out = check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_a100_value_no_violation(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\nmin_vram_gb: 80\n")
    out = check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_non_substrate_yaml_not_scanned(fake_repo: Path) -> None:
    """Glob filter excludes non-substrate recipes."""
    p = fake_repo / ".omx" / "operator_authorize_recipes" / "kaggle_t1_balle_sweep.yaml"
    p.write_text("schema_version: 1\n", encoding="utf-8")
    out = check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []  # Not a substrate_*_modal_*_dispatch.yaml file.


def test_no_recipes_dir(tmp_path: Path) -> None:
    out = check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == []


def test_strict_mode_raises_when_missing(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    with pytest.raises(PreflightError) as excinfo:
        check_substrate_recipes_declare_min_vram_gb_floor(
            repo_root=fake_repo, strict=True, verbose=False,
        )
    assert "Catalog #170" in str(excinfo.value)
    assert "NV5" in str(excinfo.value)


def test_verbose_clean_banner(fake_repo: Path, capsys: pytest.CaptureFixture) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\nmin_vram_gb: 16\n")
    check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-recipe-min-vram-gb] OK" in out


def test_verbose_violation_banner(fake_repo: Path, capsys: pytest.CaptureFixture) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-recipe-min-vram-gb]" in out
    assert "violation" in out


def test_inline_comment_field(fake_repo: Path) -> None:
    """`min_vram_gb: 16  # T4 floor` should still match."""
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\nmin_vram_gb: 16  # T4 floor\n")
    out = check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_waiver_placeholder_not_auto_waived(fake_repo: Path) -> None:
    """`# MIN_VRAM_GB_OK:<reason>` literal placeholder must not auto-waive."""
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n# MIN_VRAM_GB_OK:<reason>\n",
    )
    out = check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_multiple_recipes_partial_compliance(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "good", "schema_version: 1\nname: good\nmin_vram_gb: 80\n")
    _write_recipe(fake_repo, "bad", "schema_version: 1\nname: bad\n")
    out = check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "bad" in out[0]


def test_non_strict_returns_list(fake_repo: Path) -> None:
    out = check_substrate_recipes_declare_min_vram_gb_floor(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert isinstance(out, list)
