"""Catalog #173 (WAVE-7-MED-FIX, REVIEW-OMNI C4) tests.

Bug-class anchor: REVIEW-OMNI 2026-05-12 Quantizr/Fridrich — sane_hnerv
had 5 failed first-anchor attempts; treating sane_hnerv as just-one-of-N
in parallel fan-out is a Race-mode rigor inversion at the wrong moment.
Canary-first ordering is correct. Each substrate recipe MUST declare
``canary_status`` (canary / post_canary_dependent / independent_substrate).
A ``post_canary_dependent`` recipe MUST also declare ``canary_dependency:
<substrate_id>``.

Coverage targets:
- canary -> no violation
- post_canary_dependent + canary_dependency -> no violation
- post_canary_dependent without canary_dependency -> violation
- independent_substrate -> no violation
- missing canary_status -> violation
- invalid canary_status value -> violation
- waiver -> no violation
- waiver placeholder -> NOT auto-waived
- non-substrate yaml -> not scanned
- no recipes dir -> empty
- strict raises
- verbose banners
- multiple recipes mixed compliance
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_dispatch_honors_canary_first_ordering,
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


def test_canary_no_violation(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo",
                  "schema_version: 1\nname: foo\ncanary_status: canary\n")
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_post_canary_with_dep_no_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n"
        "canary_status: post_canary_dependent\n"
        "canary_dependency: sane_hnerv\n",
    )
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_post_canary_without_dep_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\ncanary_status: post_canary_dependent\n",
    )
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "canary_dependency" in out[0]


def test_independent_substrate_no_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\ncanary_status: independent_substrate\n",
    )
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_missing_field_violation(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "canary_status" in out[0]


def test_invalid_value_violation(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo",
                  "schema_version: 1\nname: foo\ncanary_status: nonsense\n")
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "not a recognized" in out[0]


def test_waiver_no_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n# CANARY_FIRST_OK:proxy-only-recipe\n",
    )
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_waiver_placeholder_not_auto_waived(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n# CANARY_FIRST_OK:<reason>\n",
    )
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_non_substrate_yaml_not_scanned(fake_repo: Path) -> None:
    p = fake_repo / ".omx" / "operator_authorize_recipes" / "kaggle_t1_balle_sweep.yaml"
    p.write_text("schema_version: 1\n", encoding="utf-8")
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_no_recipes_dir(tmp_path: Path) -> None:
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == []


def test_strict_mode_raises(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    with pytest.raises(PreflightError) as excinfo:
        check_substrate_dispatch_honors_canary_first_ordering(
            repo_root=fake_repo, strict=True, verbose=False,
        )
    assert "Catalog #173" in str(excinfo.value)
    assert "C4" in str(excinfo.value)


def test_verbose_clean_banner(fake_repo: Path, capsys: pytest.CaptureFixture) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\ncanary_status: canary\n",
    )
    check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-recipe-canary-first] OK" in out


def test_verbose_violation_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-recipe-canary-first]" in out
    assert "violation" in out


def test_quoted_value_accepted(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\ncanary_status: \"canary\"\n",
    )
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_inline_comment_value(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\ncanary_status: canary  # the One True canary\n",
    )
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_multiple_recipes_mixed(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "good",
                  "schema_version: 1\nname: good\ncanary_status: canary\n")
    _write_recipe(fake_repo, "bad", "schema_version: 1\nname: bad\n")
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "bad" in out[0]


def test_non_strict_returns_list(fake_repo: Path) -> None:
    out = check_substrate_dispatch_honors_canary_first_ordering(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert isinstance(out, list)
