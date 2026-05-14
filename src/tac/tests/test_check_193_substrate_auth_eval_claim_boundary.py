# SPDX-License-Identifier: MIT
"""Catalog #193 tests for substrate auth-eval score-claim boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_trainers_do_not_use_finite_auth_eval_parser_for_claims,
)


def _write_trainer(root: Path, name: str, text: str) -> Path:
    path = root / "experiments" / f"train_substrate_{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_check_193_rejects_finite_auth_eval_parser_in_substrate_trainer(tmp_path: Path) -> None:
    _write_trainer(
        tmp_path,
        "bad",
        "from tac.auth_eval_result import parse_finite_auth_eval_score\n"
        "parse_finite_auth_eval_score({})\n",
    )

    out = check_substrate_trainers_do_not_use_finite_auth_eval_parser_for_claims(
        repo_root=tmp_path,
    )

    assert len(out) == 1
    assert "parse_finite_auth_eval_score" in out[0]


def test_check_193_accepts_canonical_claim_helper(tmp_path: Path) -> None:
    _write_trainer(
        tmp_path,
        "good",
        "from tac.substrates._shared.trainer_skeleton import "
        "require_contest_cuda_auth_eval_claim\n"
        "require_contest_cuda_auth_eval_claim(path, archive_sha256='x', "
        "substrate_tag='good')\n",
    )

    out = check_substrate_trainers_do_not_use_finite_auth_eval_parser_for_claims(
        repo_root=tmp_path,
    )

    assert out == []


def test_check_193_allows_explicit_waiver(tmp_path: Path) -> None:
    _write_trainer(
        tmp_path,
        "waived",
        "# AUTH_EVAL_FINITE_PARSER_OK:diagnostic-only-test-fixture\n"
        "from tac.auth_eval_result import parse_finite_auth_eval_score\n",
    )

    out = check_substrate_trainers_do_not_use_finite_auth_eval_parser_for_claims(
        repo_root=tmp_path,
    )

    assert out == []


def test_check_193_strict_raises(tmp_path: Path) -> None:
    _write_trainer(
        tmp_path,
        "bad",
        "from tac.auth_eval_result import parse_finite_auth_eval_score\n",
    )

    with pytest.raises(PreflightError, match="Catalog #193"):
        check_substrate_trainers_do_not_use_finite_auth_eval_parser_for_claims(
            repo_root=tmp_path,
            strict=True,
        )
