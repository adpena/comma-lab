# SPDX-License-Identifier: MIT
"""Catalog #223 tests for substrate auth-eval CLI freshness."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_auth_eval_invocations_use_current_cli,
    preflight_all,
)


def _repo(tmp_path: Path, rel: str, text: str) -> Path:
    (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return tmp_path


def test_check223_rejects_stale_archive_zip_and_output_json_flags(
    tmp_path: Path,
) -> None:
    root = _repo(
        tmp_path,
        "experiments/train_substrate_bad.py",
        "CONTEST_AUTH_EVAL_SCRIPT = 'experiments/contest_auth_eval.py'\n"
        "cmd = [CONTEST_AUTH_EVAL_SCRIPT, '--archive-zip', 'a.zip', "
        "'--output-json', 'out.json']\n",
    )

    violations = check_substrate_auth_eval_invocations_use_current_cli(
        repo_root=root,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 3
    assert any("--archive-zip" in v for v in violations)
    assert any("--output-json" in v for v in violations)
    assert any("--inflate-sh" in v for v in violations)


def test_check223_rejects_direct_invocation_without_inflate_sh(
    tmp_path: Path,
) -> None:
    root = _repo(
        tmp_path,
        "experiments/train_substrate_bad.py",
        "CONTEST_AUTH_EVAL_SCRIPT = 'experiments/contest_auth_eval.py'\n"
        "cmd = [CONTEST_AUTH_EVAL_SCRIPT, '--archive', 'a.zip', "
        "'--json-out', 'out.json']\n",
    )

    violations = check_substrate_auth_eval_invocations_use_current_cli(
        repo_root=root,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "--inflate-sh" in violations[0]


def test_check223_accepts_direct_current_cli(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        "experiments/train_substrate_ok.py",
        "CONTEST_AUTH_EVAL_SCRIPT = 'experiments/contest_auth_eval.py'\n"
        "cmd = [CONTEST_AUTH_EVAL_SCRIPT, '--archive', 'a.zip', "
        "'--inflate-sh', 'inflate.sh', '--json-out', 'out.json']\n",
    )

    assert check_substrate_auth_eval_invocations_use_current_cli(
        repo_root=root,
        strict=False,
        verbose=False,
    ) == []


def test_check223_accepts_canonical_gate(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        "experiments/train_substrate_ok.py",
        "from tac.substrates._shared.smoke_auth_eval_gate import gate_auth_eval_call\n"
        "CONTEST_AUTH_EVAL_SCRIPT = 'experiments/contest_auth_eval.py'\n"
        "gate_auth_eval_call(contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT)\n",
    )

    assert check_substrate_auth_eval_invocations_use_current_cli(
        repo_root=root,
        strict=False,
        verbose=False,
    ) == []


def test_check223_strict_raises(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        "experiments/train_substrate_bad.py",
        "CONTEST_AUTH_EVAL_SCRIPT = 'experiments/contest_auth_eval.py'\n"
        "cmd = [CONTEST_AUTH_EVAL_SCRIPT, '--archive-zip', 'a.zip']\n",
    )

    with pytest.raises(
        PreflightError,
        match="check_substrate_auth_eval_invocations_use_current_cli",
    ):
        check_substrate_auth_eval_invocations_use_current_cli(
            repo_root=root,
            strict=True,
            verbose=False,
        )


def test_check223_live_repo_clean() -> None:
    assert check_substrate_auth_eval_invocations_use_current_cli(
        strict=False,
        verbose=False,
    ) == []


def test_check223_wired_into_preflight_all_strict() -> None:
    source = inspect.getsource(preflight_all)
    assert "check_substrate_auth_eval_invocations_use_current_cli" in source
    callsite = source.split(
        "check_substrate_auth_eval_invocations_use_current_cli",
        1,
    )[1][:140]
    assert "strict=True" in callsite


def test_c6_remote_driver_requires_valid_auth_eval_before_done() -> None:
    script = Path("scripts/remote_lane_substrate_c6_e4_mdl_ibps.sh").read_text(
        encoding="utf-8",
    )
    validation_pos = script.index("auth_eval_score_claim_valid")
    done_pos = script.index("LANE_C6_MDL_IBPS_DONE [contest-CUDA]")
    assert validation_pos < done_pos
    assert 'auth_eval_score_axis") != "contest_cuda"' in script
    assert 'auth_eval_exact_cuda_complete") is not True' in script


def test_c6_remote_driver_routes_modal_outputs_to_modal_results() -> None:
    script = Path("scripts/remote_lane_substrate_c6_e4_mdl_ibps.sh").read_text(
        encoding="utf-8",
    )
    assert 'MODAL_RUNTIME:-0}" = "1"' in script
    assert 'LOG_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}"' in script
    assert 'C6_E4_MDL_IBPS_OUTPUT_DIR="$OUTPUT_DIR"' in script


def test_c6_remote_driver_threads_autocast_env_to_trainer() -> None:
    script = Path("scripts/remote_lane_substrate_c6_e4_mdl_ibps.sh").read_text(
        encoding="utf-8",
    )
    assert "C6_E4_MDL_IBPS_ENABLE_AUTOCAST_FP16" in script
    assert "C6_TRAINER_ARGS+=(--enable-autocast-fp16)" in script
    assert '${C6_TRAINER_ARGS[@]+"${C6_TRAINER_ARGS[@]}"}' in script
