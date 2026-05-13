"""Catalog #179 (WAVE-7-LOW-FIX, REVIEW-OMNI NV3) tests.

Bug-class anchor: REVIEW-OMNI 2026-05-12 — Substrate trainers don't use
``torch.compile`` / Inductor. NOT a regression; just unrealized speedup.
This META gate refuses any substrate trainer that neither declares
``--enable-torch-compile``, invokes ``torch.compile(...)``, nor carries
a file-level ``# TORCH_COMPILE_WAIVED:<reason>`` waiver.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_trainers_declare_torch_compile_support,
)


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    return repo


def _write_trainer(repo: Path, name: str, body: str) -> Path:
    p = repo / "experiments" / f"train_substrate_{name}.py"
    p.write_text(body, encoding="utf-8")
    return p


def test_compliant_flag_no_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "import argparse\np = argparse.ArgumentParser()\n"
        "p.add_argument('--enable-torch-compile', action='store_true')\n",
    )
    out = check_substrate_trainers_declare_torch_compile_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_compliant_api_invocation_no_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "import torch\nmodel = torch.compile(model)\n",
    )
    out = check_substrate_trainers_declare_torch_compile_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_missing_violation(fake_repo: Path) -> None:
    _write_trainer(fake_repo, "foo", "import torch\n")
    out = check_substrate_trainers_declare_torch_compile_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_waiver_no_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "# TORCH_COMPILE_WAIVED:research-substrate-known-engineering-gap\n"
        "import torch\n",
    )
    out = check_substrate_trainers_declare_torch_compile_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_waiver_placeholder_not_auto_waived(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "# TORCH_COMPILE_WAIVED:<reason>\nimport torch\n",
    )
    out = check_substrate_trainers_declare_torch_compile_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_non_substrate_trainer_not_scanned(fake_repo: Path) -> None:
    p = fake_repo / "experiments" / "train_paradigm_delta_epsilon_zeta.py"
    p.write_text("import torch\n", encoding="utf-8")
    out = check_substrate_trainers_declare_torch_compile_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_no_experiments_dir(tmp_path: Path) -> None:
    out = check_substrate_trainers_declare_torch_compile_support(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == []


def test_strict_mode_raises(fake_repo: Path) -> None:
    _write_trainer(fake_repo, "foo", "import torch\n")
    with pytest.raises(PreflightError) as excinfo:
        check_substrate_trainers_declare_torch_compile_support(
            repo_root=fake_repo, strict=True, verbose=False,
        )
    assert "Catalog #179" in str(excinfo.value)
    assert "NV3" in str(excinfo.value)


def test_verbose_clean_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_trainer(
        fake_repo, "foo",
        "torch.compile(model)\n",
    )
    check_substrate_trainers_declare_torch_compile_support(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-trainer-torch-compile] OK" in out


def test_verbose_violation_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_trainer(fake_repo, "foo", "import torch\n")
    check_substrate_trainers_declare_torch_compile_support(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-trainer-torch-compile]" in out
    assert "violation" in out


def test_multiple_trainers_mixed(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "good",
        "torch.compile(model)\n",
    )
    _write_trainer(fake_repo, "bad", "import torch\n")
    out = check_substrate_trainers_declare_torch_compile_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "bad" in out[0]


def test_non_strict_returns_list(fake_repo: Path) -> None:
    out = check_substrate_trainers_declare_torch_compile_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert isinstance(out, list)


def test_flag_or_api_both_satisfy(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "a",
        "p.add_argument('--enable-torch-compile')\n",
    )
    _write_trainer(
        fake_repo, "b",
        "model = torch.compile(model)\n",
    )
    out = check_substrate_trainers_declare_torch_compile_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []
