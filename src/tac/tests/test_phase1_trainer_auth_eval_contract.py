from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _load_trainer_module():
    path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    spec = importlib.util.spec_from_file_location("phase1_t1_trainer_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase1_trainer_auth_eval_refuses_scaffold_before_training() -> None:
    module = _load_trainer_module()

    assert module.PHASE1_SCAFFOLD_ONLY is True
    blockers = module.phase1_scaffold_blockers()
    assert "auth_eval_custody_not_wired" in blockers
    assert "exact_cuda_score_not_run" in blockers


def test_phase1_trainer_auth_eval_cli_exits_nonzero_for_smoke_scaffold(
    tmp_path: Path,
) -> None:
    trainer_path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    result = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--smoke",
            "--allow-missing-canonical-a1",
            "--auth-eval",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--auth-eval refused" in combined
    assert "scaffold-only" in combined
    assert not (tmp_path / "out" / "contest_auth_eval.json").exists()


def test_phase1_trainer_auth_eval_refuses_even_with_scorer_domain_loss(
    tmp_path: Path,
) -> None:
    trainer_path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    result = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--smoke",
            "--allow-missing-canonical-a1",
            "--enable-scorer-domain-loss",
            "--auth-eval",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--auth-eval refused" in combined
    assert "scaffold-only" in combined
    assert not (tmp_path / "out" / "contest_auth_eval.json").exists()


def test_phase1_trainer_scorer_domain_refuses_disabled_yuv6(
    tmp_path: Path,
) -> None:
    trainer_path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    result = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--smoke",
            "--allow-missing-canonical-a1",
            "--enable-scorer-domain-loss",
            "--disable-differentiable-yuv6",
            "--no-auth-eval",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--enable-scorer-domain-loss requires --enable-differentiable-yuv6" in combined
    assert not (tmp_path / "out").exists()


def test_phase1_trainer_scorer_domain_refuses_zero_non_smoke_epochs(
    tmp_path: Path,
) -> None:
    trainer_path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    result = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--enable-scorer-domain-loss",
            "--epochs",
            "0",
            "--no-auth-eval",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--enable-scorer-domain-loss requires --epochs > 0" in combined
    assert not (tmp_path / "out").exists()


def test_phase1_trainer_scorer_domain_refuses_zero_smoke_epochs(
    tmp_path: Path,
) -> None:
    trainer_path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    result = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--smoke",
            "--allow-missing-canonical-a1",
            "--enable-scorer-domain-loss",
            "--epochs",
            "0",
            "--no-auth-eval",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--enable-scorer-domain-loss requires --epochs > 0" in combined
    assert not (tmp_path / "out").exists()


def test_phase1_trainer_non_smoke_cli_exits_nonzero_before_training(
    tmp_path: Path,
) -> None:
    trainer_path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    result = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--no-auth-eval",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "non-smoke training refused" in combined
    assert "auth_eval_custody_not_wired" in combined
    assert not (tmp_path / "out" / "archive.zip").exists()


def test_phase1_trainer_maybe_run_auth_eval_direct_refuses_scaffold(
    tmp_path: Path,
) -> None:
    module = _load_trainer_module()
    with pytest.raises(SystemExit, match="--auth-eval refused"):
        module.maybe_run_auth_eval(
            archive_path=tmp_path / "archive.zip",
            submission_dir=tmp_path / "submission",
            output_dir=tmp_path / "out",
            enabled=True,
            dispatch_lane_id="lane_t1_phase1",  # FAKE_LANE_OK: auth-eval fixture
            dispatch_claims_path=tmp_path / "claims.md",
        )


def test_phase1_trainer_maybe_run_auth_eval_disabled_still_noops(tmp_path: Path) -> None:
    module = _load_trainer_module()
    assert (
        module.maybe_run_auth_eval(
            archive_path=tmp_path / "archive.zip",
            submission_dir=tmp_path / "submission",
            output_dir=tmp_path / "out",
            enabled=False,
            dispatch_lane_id=None,
            dispatch_claims_path=tmp_path / "claims.md",
        )
        is None
    )
