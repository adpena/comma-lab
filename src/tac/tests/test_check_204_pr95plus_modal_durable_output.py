from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_pr95plus_modal_smoke_uses_durable_provider_output,
)


REMOTE = Path("scripts/remote_lane_substrate_pr101_lc_v2_clone_enhanced_curriculum.sh")
TRAINER = Path("experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py")
MODAL = Path("experiments/modal_train_lane.py")


def _write_minimal_contract(root: Path) -> None:
    (root / REMOTE).parent.mkdir(parents=True, exist_ok=True)
    (root / TRAINER).parent.mkdir(parents=True, exist_ok=True)
    (root / MODAL).parent.mkdir(parents=True, exist_ok=True)
    (root / REMOTE).write_text(
        "\n".join(
            [
                'elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then',
                '    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"',
                'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"',
            ]
        ),
        encoding="utf-8",
    )
    (root / TRAINER).write_text("cmd = ['--json-out', str(output_json)]\n", encoding="utf-8")
    (root / MODAL).write_text("scan_roots = [workspace / 'results', volume_dir,]\n", encoding="utf-8")


def test_check_204_live_repo_clean() -> None:
    assert check_pr95plus_modal_smoke_uses_durable_provider_output(
        repo_root=Path.cwd(), strict=False
    ) == []


def test_check_204_blocks_tmp_workspace_output_default(tmp_path: Path) -> None:
    _write_minimal_contract(tmp_path)
    (tmp_path / REMOTE).write_text(
        'OUTPUT_DIR="${PR95PLUS_OUTPUT_DIR:-$LOG_DIR/output}"\n',
        encoding="utf-8",
    )

    violations = check_pr95plus_modal_smoke_uses_durable_provider_output(
        repo_root=tmp_path, strict=False
    )

    assert any("$LOG_DIR/output" in v for v in violations)
    assert any("/modal_results" in v for v in violations)


def test_check_204_blocks_temp_auth_eval_bypass(tmp_path: Path) -> None:
    _write_minimal_contract(tmp_path)
    (tmp_path / TRAINER).write_text(
        "cmd.append('--allow-temp-work-dir')\n",
        encoding="utf-8",
    )

    violations = check_pr95plus_modal_smoke_uses_durable_provider_output(
        repo_root=tmp_path, strict=False
    )

    assert any("--allow-temp-work-dir" in v for v in violations)


def test_check_204_blocks_missing_modal_volume_harvest(tmp_path: Path) -> None:
    _write_minimal_contract(tmp_path)
    (tmp_path / MODAL).write_text(
        "scan_roots = [workspace / 'results']\n",
        encoding="utf-8",
    )

    violations = check_pr95plus_modal_smoke_uses_durable_provider_output(
        repo_root=tmp_path, strict=False
    )

    assert any("volume_dir" in v for v in violations)


def test_check_204_strict_raises(tmp_path: Path) -> None:
    _write_minimal_contract(tmp_path)
    (tmp_path / TRAINER).write_text(
        "cmd.append('--allow-temp-work-dir')\n",
        encoding="utf-8",
    )

    with pytest.raises(PreflightError):
        check_pr95plus_modal_smoke_uses_durable_provider_output(
            repo_root=tmp_path, strict=True
        )
