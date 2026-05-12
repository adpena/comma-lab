"""Static hardening checks for experiments/modal_train_lane.py."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SOURCE = REPO_ROOT / "experiments" / "modal_train_lane.py"


def test_modal_train_lane_disables_lane_local_exact_eval() -> None:
    text = SOURCE.read_text()
    assert '"T1_RUN_CONTEST_CUDA_AUTH_EVAL": "0"' in text
    assert '"SCPP_RUN_CONTEST_CUDA_AUTH_EVAL": "0"' in text
    assert '"RUN_CONTEST_EVAL": "0"' in text
    assert "refusing exact CUDA auth-eval from modal_train_lane.py" in text
    assert "Use the canonical claimed exact-eval dispatcher instead" in text
    assert "MODAL_ALLOW_EXACT_CUDA_AUTH_EVAL" not in text
    assert "env.update(env_overrides)" in text
    assert 'if env.get("MODAL_ALLOW_EXACT_CUDA_AUTH_EVAL"' not in text


def test_modal_env_sh_also_fails_closed_for_sourced_lane_scripts() -> None:
    text = SOURCE.read_text()
    assert "export T1_RUN_CONTEST_CUDA_AUTH_EVAL=0" in text
    assert "export SCPP_RUN_CONTEST_CUDA_AUTH_EVAL=0" in text
    assert "export RUN_CONTEST_EVAL=0" in text
