"""Catalog #180 (WAVE-7-LOW-FIX, REVIEW-OMNI NV4) tests.

Bug-class anchor: REVIEW-OMNI 2026-05-12 Carmack — Eval-time scorer
forwards may not be ``with torch.no_grad():`` wrapped. Activation memory
pressure during eval. This META gate refuses any substrate trainer that
contains neither a ``torch.no_grad`` context nor a file-level
``# NO_GRAD_WAIVED:<reason>`` waiver. Strict-from-byte-one (live count
at landing: 0).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_trainers_use_no_grad_at_eval,
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


def test_compliant_context_no_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "import torch\n"
        "with torch.no_grad():\n    score = model(x)\n",
    )
    out = check_substrate_trainers_use_no_grad_at_eval(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_compliant_decorator_no_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "import torch\n"
        "@torch.no_grad()\ndef eval_fn(): pass\n",
    )
    out = check_substrate_trainers_use_no_grad_at_eval(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_compliant_inference_mode_no_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "import torch\n"
        "with torch.inference_mode():\n    pass\n",
    )
    out = check_substrate_trainers_use_no_grad_at_eval(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_missing_violation(fake_repo: Path) -> None:
    _write_trainer(fake_repo, "foo", "import torch\n")
    out = check_substrate_trainers_use_no_grad_at_eval(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "no_grad" in out[0] or "NO_GRAD" in out[0]


def test_waiver_no_violation(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "# NO_GRAD_WAIVED:training-only-no-eval-path\nimport torch\n",
    )
    out = check_substrate_trainers_use_no_grad_at_eval(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_waiver_placeholder_not_auto_waived(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "foo",
        "# NO_GRAD_WAIVED:<reason>\nimport torch\n",
    )
    out = check_substrate_trainers_use_no_grad_at_eval(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1


def test_non_substrate_trainer_not_scanned(fake_repo: Path) -> None:
    p = fake_repo / "experiments" / "train_paradigm_delta_epsilon_zeta.py"
    p.write_text("import torch\n", encoding="utf-8")
    out = check_substrate_trainers_use_no_grad_at_eval(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert out == []


def test_no_experiments_dir(tmp_path: Path) -> None:
    out = check_substrate_trainers_use_no_grad_at_eval(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == []


def test_strict_mode_raises(fake_repo: Path) -> None:
    _write_trainer(fake_repo, "foo", "import torch\n")
    with pytest.raises(PreflightError) as excinfo:
        check_substrate_trainers_use_no_grad_at_eval(
            repo_root=fake_repo, strict=True, verbose=False,
        )
    assert "Catalog #180" in str(excinfo.value)
    assert "NV4" in str(excinfo.value)


def test_verbose_clean_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_trainer(
        fake_repo, "foo",
        "with torch.no_grad():\n    pass\n",
    )
    check_substrate_trainers_use_no_grad_at_eval(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-trainer-no-grad] OK" in out


def test_verbose_violation_banner(
    fake_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_trainer(fake_repo, "foo", "import torch\n")
    check_substrate_trainers_use_no_grad_at_eval(
        repo_root=fake_repo, strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[substrate-trainer-no-grad]" in out
    assert "violation" in out


def test_multiple_trainers_mixed(fake_repo: Path) -> None:
    _write_trainer(
        fake_repo, "good",
        "with torch.no_grad():\n    pass\n",
    )
    _write_trainer(fake_repo, "bad", "import torch\n")
    out = check_substrate_trainers_use_no_grad_at_eval(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert len(out) == 1
    assert "bad" in out[0]


def test_non_strict_returns_list(fake_repo: Path) -> None:
    out = check_substrate_trainers_use_no_grad_at_eval(
        repo_root=fake_repo, strict=False, verbose=False,
    )
    assert isinstance(out, list)


def test_real_substrate_trainers_count_zero_in_live_repo() -> None:
    """Live-repo regression: at landing, ALL 14 substrate trainers
    already use torch.no_grad — live count is 0.
    """
    repo = Path.cwd()
    out = check_substrate_trainers_use_no_grad_at_eval(
        repo_root=repo, strict=False, verbose=False,
    )
    assert isinstance(out, list)
    # Strict-from-byte-one — live count at landing MUST be 0.
    assert len(out) == 0, f"Live regression: {len(out)} violations: {out[:3]}"
