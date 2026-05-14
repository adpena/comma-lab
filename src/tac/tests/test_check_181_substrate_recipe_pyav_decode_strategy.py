# SPDX-License-Identifier: MIT
"""Catalog #181 (WAVE-7-LOW-FIX, REVIEW-OMNI NV8) tests.

Bug-class anchor: REVIEW-OMNI 2026-05-12 — pyav decode -> CUDA upload
synchronicity not profiled. Each substrate operator-authorize recipe
MUST declare ``pyav_decode_strategy`` (cpu_thread_async_upload /
cuda_nvdec / cpu_blocking_upload / not_applicable). Same-line waiver:
``# PYAV_DECODE_STRATEGY_OK:<reason>``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_recipes_declare_pyav_decode_strategy,
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
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n"
        "pyav_decode_strategy: cpu_thread_async_upload\n",
    )
    out = check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_compliant_cuda_nvdec_no_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n"
        "pyav_decode_strategy: cuda_nvdec\n",
    )
    out = check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_compliant_not_applicable_no_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n"
        "pyav_decode_strategy: not_applicable\n",
    )
    out = check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_missing_field_violation(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    out = check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "pyav_decode_strategy" in out[0]


def test_invalid_value_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n"
        "pyav_decode_strategy: bogus_strategy\n",
    )
    out = check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_waiver_recipe_no_violation(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\nname: foo\n"
        "# PYAV_DECODE_STRATEGY_OK:cached-latent-no-video-decode\n",
    )
    out = check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_waiver_placeholder_not_auto_waived(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\n# PYAV_DECODE_STRATEGY_OK:<reason>\n",
    )
    out = check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_non_substrate_yaml_not_scanned(fake_repo: Path) -> None:
    p = fake_repo / ".omx" / "operator_authorize_recipes" / "kaggle_t1_balle_sweep.yaml"
    p.write_text("schema_version: 1\n", encoding="utf-8")
    out = check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_no_recipes_dir(tmp_path: Path) -> None:
    out = check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == []


def test_strict_mode_raises_when_missing(fake_repo: Path) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\nname: foo\n")
    with pytest.raises(PreflightError) as excinfo:
        check_substrate_recipes_declare_pyav_decode_strategy(
            repo_root=fake_repo, strict=True, verbose=False,
        )
    assert "Catalog #181" in str(excinfo.value)
    assert "NV8" in str(excinfo.value)


def test_verbose_clean_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\npyav_decode_strategy: cpu_thread_async_upload\n",
    )
    check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-recipe-pyav-decode-strategy] OK" in out


def test_verbose_violation_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_recipe(fake_repo, "foo", "schema_version: 1\n")
    check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-recipe-pyav-decode-strategy]" in out
    assert "violation" in out


def test_inline_comment_field(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "foo",
        "schema_version: 1\n"
        "pyav_decode_strategy: cuda_nvdec  # GPU NVDEC pipeline\n",
    )
    out = check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_multiple_recipes_partial_compliance(fake_repo: Path) -> None:
    _write_recipe(
        fake_repo, "good",
        "schema_version: 1\npyav_decode_strategy: cuda_nvdec\n",
    )
    _write_recipe(fake_repo, "bad", "schema_version: 1\n")
    out = check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "bad" in out[0]


def test_non_strict_returns_list(fake_repo: Path) -> None:
    out = check_substrate_recipes_declare_pyav_decode_strategy(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert isinstance(out, list)
