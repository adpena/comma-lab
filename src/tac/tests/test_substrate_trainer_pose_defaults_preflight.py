from __future__ import annotations

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_trainer_pose_defaults_match_contest_formula,
)


def _write_trainer(tmp_path, body: str) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "train_substrate_example.py").write_text(body)


def test_substrate_trainer_pose_defaults_accept_contest_formula(tmp_path) -> None:
    _write_trainer(
        tmp_path,
        """
import argparse
import math

def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0)
    return p
""",
    )

    assert (
        check_substrate_trainer_pose_defaults_match_contest_formula(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )
        == []
    )


def test_substrate_trainer_pose_defaults_accept_numeric_sqrt10(tmp_path) -> None:
    _write_trainer(
        tmp_path,
        """
import argparse

def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma-pose", type=float, default=3.1622776601683795)
    p.add_argument("--pose-weight-scale", type=float, default=1)
    return p
""",
    )

    assert (
        check_substrate_trainer_pose_defaults_match_contest_formula(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )
        == []
    )


def test_substrate_trainer_pose_defaults_reject_pr106_operating_point_default(
    tmp_path,
) -> None:
    _write_trainer(
        tmp_path,
        """
import argparse

def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma-pose", type=float, default=1.0)
    p.add_argument("--pose-weight-scale", type=float, default=2.71)
    return p
""",
    )

    with pytest.raises(PreflightError, match="--gamma-pose default"):
        check_substrate_trainer_pose_defaults_match_contest_formula(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )

