# SPDX-License-Identifier: MIT
"""Catalog #171 (WAVE-7-MED-FIX, REVIEW-OMNI NV9) tests.

Bug-class anchor: REVIEW-OMNI 2026-05-12 Carmack — `upstream/videos/0.mkv`
concurrent-dispatch FS read contention. Each substrate recipe MUST declare
``video_input_strategy``.

Coverage targets:
- valid `per_dispatch_local_copy` -> no violation
- valid `readonly_mmap` -> no violation
- valid `shared_volume_no_contention_expected` -> no violation
- missing field -> violation
- invalid value -> violation
- waiver -> no violation
- waiver placeholder `<reason>` -> NOT auto-waived
- non-substrate yaml -> not scanned
- no recipes dir -> empty
- strict raises
- verbose banners
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_recipes_declare_video_input_strategy,
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


def test_per_dispatch_local_copy_ok(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo",
                  "schema_version: 1\nname: foo\nvideo_input_strategy: per_dispatch_local_copy\n")
    out = check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_readonly_mmap_ok(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo",
                  "schema_version: 1\nname: foo\nvideo_input_strategy: readonly_mmap\n")
    out = check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_shared_volume_no_contention_expected_ok(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n"
        "video_input_strategy: shared_volume_no_contention_expected\n",
    )
    out = check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_missing_field_violation(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    out = check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_invalid_value_violation(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo",
                  "schema_version: 1\nname: foo\nvideo_input_strategy: nonsense\n")
    out = check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "not a recognized" in out[0]


def test_waiver_no_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n# VIDEO_INPUT_STRATEGY_OK:proxy-no-video\n",
    )
    out = check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_waiver_placeholder_not_auto_waived(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n# VIDEO_INPUT_STRATEGY_OK:<reason>\n",
    )
    out = check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_non_substrate_yaml_not_scanned(fake_repo: Path) -> None:
    p = fake_repo / ".omx" / "operator_authorize_recipes" / "kaggle_t1_balle_sweep.yaml"
    p.write_text("schema_version: 1\n", encoding="utf-8")
    out = check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_no_recipes_dir(tmp_path: Path) -> None:
    out = check_substrate_recipes_declare_video_input_strategy(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == []


def test_strict_mode_raises_when_missing(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    with pytest.raises(PreflightError) as excinfo:
        check_substrate_recipes_declare_video_input_strategy(
            repo_root=fake_repo, strict=True, verbose=False,
        )
    assert "Catalog #171" in str(excinfo.value)
    assert "NV9" in str(excinfo.value)


def test_verbose_clean_banner(fake_repo: Path, capsys: pytest.CaptureFixture) -> None:
    _write_recipe(fake_repo, "foo",
                  "schema_version: 1\nname: foo\nvideo_input_strategy: readonly_mmap\n")
    check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-recipe-video-input-strategy] OK" in out


def test_verbose_violation_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-recipe-video-input-strategy]" in out
    assert "violation" in out


def test_quoted_value_accepted(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo",
                  "schema_version: 1\nname: foo\n"
                  "video_input_strategy: \"per_dispatch_local_copy\"\n")
    out = check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_inline_comment_value(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo",
                  "schema_version: 1\nname: foo\n"
                  "video_input_strategy: readonly_mmap  # mmap O_RDONLY\n")
    out = check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_non_strict_returns_list(fake_repo: Path) -> None:
    out = check_substrate_recipes_declare_video_input_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert isinstance(out, list)
