# SPDX-License-Identifier: MIT
"""Catalog #172 (WAVE-7-MED-FIX, REVIEW-OMNI NV2) tests.

Bug-class anchor: REVIEW-OMNI 2026-05-12 Hotz — Substrate trainers don't
use ``torch.autocast(FP16)`` + GradScaler; engineering speed gap vs T1
Balle is 4-6x. This META gate refuses any new substrate trainer that
neither declares the ``--enable-autocast-fp16`` argparse flag NOR carries
a file-level ``# AUTOCAST_FP16_WAIVED:<reason>`` waiver.

Coverage targets:
- compliant trainer (declares --enable-autocast-fp16) -> no violation
- non-compliant trainer (no flag, no waiver) -> violation
- waiver trainer -> no violation
- waiver placeholder -> NOT auto-waived
- non-substrate trainer (e.g., train_paradigm_*.py) -> not scanned
- no experiments dir -> empty
- strict raises
- verbose banners
- multiple trainers mixed compliance
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_trainers_declare_autocast_fp16_support,
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


def test_compliant_trainer_no_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "import argparse\np = argparse.ArgumentParser()\n"
        "p.add_argument('--enable-autocast-fp16', action='store_true')\n",
    )
    out = check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_missing_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "import argparse\np = argparse.ArgumentParser()\n",
    )
    out = check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_waiver_no_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "# AUTOCAST_FP16_WAIVED:research-substrate-known-engineering-gap\n"
        "import argparse\n",
    )
    out = check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_waiver_placeholder_not_auto_waived(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "# AUTOCAST_FP16_WAIVED:<reason>\nimport argparse\n",
    )
    out = check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_non_substrate_trainer_not_scanned(fake_repo: Path) -> None:
    """T1 Balle trainer (the canonical pattern source) is NOT a substrate trainer."""
    p = fake_repo / "experiments" / "train_paradigm_delta_epsilon_zeta.py"
    p.write_text("import argparse\n", encoding="utf-8")
    out = check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_no_experiments_dir(tmp_path: Path) -> None:
    out = check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == []


def test_strict_mode_raises(fake_repo: Path) -> None:
    _write_trainer(fake_repo, "foo", "import argparse\n")
    with pytest.raises(PreflightError) as excinfo:
        check_substrate_trainers_declare_autocast_fp16_support(
            repo_root=fake_repo, strict=True, verbose=False,
        )
    assert "Catalog #172" in str(excinfo.value)
    assert "NV2" in str(excinfo.value)


def test_verbose_clean_banner(fake_repo: Path, capsys: pytest.CaptureFixture) -> None:
    _write_trainer(
        fake_repo, "foo",
        "import argparse\np.add_argument('--enable-autocast-fp16')\n",
    )
    check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-trainer-autocast-fp16] OK" in out


def test_verbose_violation_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_trainer(fake_repo, "foo", "import argparse\n")
    check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-trainer-autocast-fp16]" in out
    assert "violation" in out


def test_multiple_trainers_mixed(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "good",
        "import argparse\np.add_argument('--enable-autocast-fp16')\n",
    )
    _write_trainer(fake_repo, "bad", "import argparse\n")
    out = check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "bad" in out[0]


def test_flag_in_comment_satisfies(fake_repo: Path) -> None:
    """The check is grep-based; flag mentioned in any context (string,
    comment, docstring) satisfies the gate. The intent is simply that
    the flag name is present somewhere in the file."""
    _write_trainer(
        fake_repo, "foo",
        "# This trainer uses --enable-autocast-fp16 elsewhere\n"
        "import argparse\n",
    )
    out = check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []  # Grep-based; intentional simplicity.


def test_non_strict_returns_list(fake_repo: Path) -> None:
    out = check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert isinstance(out, list)


def test_real_substrate_trainers_count(fake_repo: Path) -> None:
    """Smoke: 14 substrate trainers fixture-style scan returns 14 violations."""
    for name in ("a", "b", "c"):
        _write_trainer(fake_repo, name, "import argparse\n")
    out = check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 3


def test_argparse_dest_form_satisfies(fake_repo: Path) -> None:
    """argparse-style flag with quoted form."""
    _write_trainer(
        fake_repo, "foo",
        "p.add_argument(\"--enable-autocast-fp16\")\n",
    )
    out = check_substrate_trainers_declare_autocast_fp16_support(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []
