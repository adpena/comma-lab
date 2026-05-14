# SPDX-License-Identifier: MIT
"""Catalog #178 (WAVE-7-LOW-FIX, REVIEW-OMNI NV1) tests.

Bug-class anchor: REVIEW-OMNI 2026-05-12 Carmack — Substrate trainers
don't enable TF32 (``torch.backends.cuda.matmul.allow_tf32 = True``).
On Ampere/Hopper (A100, 4090, H100) TF32 gives ~1.5-2x speedup on
matmul-bound workloads with no accuracy regression. This META gate
refuses any substrate trainer that neither enables TF32 NOR carries a
file-level ``# TF32_WAIVED:<reason>`` waiver.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_trainers_declare_tf32_support,
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


def test_compliant_matmul_tf32_no_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "import torch\n"
        "torch.backends.cuda.matmul.allow_tf32 = True\n",
    )
    out = check_substrate_trainers_declare_tf32_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_compliant_cudnn_tf32_no_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "import torch\n"
        "torch.backends.cudnn.allow_tf32 = True\n",
    )
    out = check_substrate_trainers_declare_tf32_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_missing_violation(fake_repo: Path) -> None:
    _write_trainer(fake_repo, "foo", "import torch\n")
    out = check_substrate_trainers_declare_tf32_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "TF32" in out[0] or "tf32" in out[0]


def test_waiver_no_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "# TF32_WAIVED:research-substrate-known-engineering-gap\n"
        "import torch\n",
    )
    out = check_substrate_trainers_declare_tf32_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_waiver_placeholder_not_auto_waived(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "# TF32_WAIVED:<reason>\nimport torch\n",
    )
    out = check_substrate_trainers_declare_tf32_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_non_substrate_trainer_not_scanned(fake_repo: Path) -> None:
    p = fake_repo / "experiments" / "train_paradigm_delta_epsilon_zeta.py"
    p.write_text("import torch\n", encoding="utf-8")
    out = check_substrate_trainers_declare_tf32_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_no_experiments_dir(tmp_path: Path) -> None:
    out = check_substrate_trainers_declare_tf32_support(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == []


def test_strict_mode_raises(fake_repo: Path) -> None:
    _write_trainer(fake_repo, "foo", "import torch\n")
    with pytest.raises(PreflightError) as excinfo:
        check_substrate_trainers_declare_tf32_support(
            repo_root=fake_repo, strict=True, verbose=False,
        )
    assert "Catalog #178" in str(excinfo.value)
    assert "NV1" in str(excinfo.value)


def test_verbose_clean_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_trainer(
        fake_repo, "foo",
        "torch.backends.cuda.matmul.allow_tf32 = True\n",
    )
    check_substrate_trainers_declare_tf32_support(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-trainer-tf32] OK" in out


def test_verbose_violation_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_trainer(fake_repo, "foo", "import torch\n")
    check_substrate_trainers_declare_tf32_support(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-trainer-tf32]" in out
    assert "violation" in out


def test_multiple_trainers_mixed(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "good",
        "torch.backends.cuda.matmul.allow_tf32 = True\n",
    )
    _write_trainer(fake_repo, "bad", "import torch\n")
    out = check_substrate_trainers_declare_tf32_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "bad" in out[0]


def test_non_strict_returns_list(fake_repo: Path) -> None:
    out = check_substrate_trainers_declare_tf32_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert isinstance(out, list)


def test_real_substrate_trainers_count(fake_repo: Path) -> None:
    for name in ("a", "b", "c"):
        _write_trainer(fake_repo, name, "import torch\n")
    out = check_substrate_trainers_declare_tf32_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 3
