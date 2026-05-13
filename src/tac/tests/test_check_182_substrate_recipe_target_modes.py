"""Catalog #182 (WAVE-7-LOW-FIX, REVIEW-OMNI NV10) tests.

Bug-class anchor: REVIEW-OMNI 2026-05-12 — per-substrate ``target_modes``
not declared. Production-graduation discipline gap. Each substrate
operator-authorize recipe MUST declare ``target_modes`` as a YAML list
containing at least one of: contest_one_video_replay, contest_generalized,
production_generalized, production_edge_adaptive, research_substrate.
Same-line waiver: ``# TARGET_MODES_OK:<reason>``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_recipes_declare_target_modes,
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


def test_compliant_inline_list_no_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n"
        "target_modes: [research_substrate]\n",
    )
    out = check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_compliant_block_list_no_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n"
        "target_modes:\n  - research_substrate\n  - contest_generalized\n",
    )
    out = check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_compliant_production_targets_no_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n"
        "target_modes: [production_generalized, production_edge_adaptive]\n",
    )
    out = check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_missing_field_violation(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    out = check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "target_modes" in out[0]


def test_empty_list_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\ntarget_modes: []\n",
    )
    out = check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "no valid value" in out[0] or "target_modes" in out[0]


def test_invalid_value_only_violation(fake_repo: Path) -> None:
    """A target_modes list with ONLY invalid values is treated as
    empty (no valid value)."""
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\ntarget_modes: [bogus]\n",
    )
    out = check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_waiver_recipe_no_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n"
        "# TARGET_MODES_OK:design-time-deferred-pending-graduation\n",
    )
    out = check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_waiver_placeholder_not_auto_waived(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\n# TARGET_MODES_OK:<reason>\n",
    )
    out = check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_non_substrate_yaml_not_scanned(fake_repo: Path) -> None:
    p = fake_repo / ".omx" / "operator_authorize_recipes" / "kaggle_t1_balle_sweep.yaml"
    p.write_text("schema_version: 1\n", encoding="utf-8")
    out = check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_no_recipes_dir(tmp_path: Path) -> None:
    out = check_substrate_recipes_declare_target_modes(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == []


def test_strict_mode_raises_when_missing(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    with pytest.raises(PreflightError) as excinfo:
        check_substrate_recipes_declare_target_modes(
            repo_root=fake_repo, strict=True, verbose=False,
        )
    assert "Catalog #182" in str(excinfo.value)
    assert "NV10" in str(excinfo.value)


def test_verbose_clean_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\ntarget_modes: [research_substrate]\n",
    )
    check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-recipe-target-modes] OK" in out


def test_verbose_violation_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\n")
    check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-recipe-target-modes]" in out
    assert "violation" in out


def test_multiple_recipes_partial_compliance(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "good",
        "schema_version: 1\ntarget_modes: [research_substrate]\n",
    )
    _write_recipe(fake_repo, "bad", "schema_version: 1\n")
    out = check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "bad" in out[0]


def test_mixed_valid_invalid_values_passes(fake_repo: Path) -> None:
    """As long as at least one valid value is present, the recipe passes."""
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\ntarget_modes: [bogus, research_substrate]\n",
    )
    out = check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_non_strict_returns_list(fake_repo: Path) -> None:
    out = check_substrate_recipes_declare_target_modes(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert isinstance(out, list)
