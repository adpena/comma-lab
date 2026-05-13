"""Static hardening checks for experiments/modal_train_lane.py."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SOURCE = REPO_ROOT / "experiments" / "modal_train_lane.py"


def test_modal_train_lane_keeps_wrapper_non_promotional_but_allows_inline_custody_eval() -> None:
    text = SOURCE.read_text()
    assert '"T1_RUN_CONTEST_CUDA_AUTH_EVAL": "0"' in text
    assert '"SCPP_RUN_CONTEST_CUDA_AUTH_EVAL": "0"' in text
    assert '"RUN_CONTEST_EVAL": "0"' in text
    assert "refusing exact CUDA auth-eval from modal_train_lane.py" in text
    assert "Lane-local auth-eval subprocesses are allowed only when" in text
    assert "wrapper_score_claim" in text
    assert "inline_auth_eval_contract_required" in text
    assert "MODAL_ALLOW_EXACT_CUDA_AUTH_EVAL" not in text
    assert "env.update({str(k): _modal_workspace_env(v)" in text
    assert 'readonly = "/workspace/pact"' in text
    assert 'if env.get("MODAL_ALLOW_EXACT_CUDA_AUTH_EVAL"' not in text


def test_modal_env_sh_also_fails_closed_for_sourced_lane_scripts() -> None:
    text = SOURCE.read_text()
    assert "export T1_RUN_CONTEST_CUDA_AUTH_EVAL=0" in text
    assert "export SCPP_RUN_CONTEST_CUDA_AUTH_EVAL=0" in text
    assert "export RUN_CONTEST_EVAL=0" in text


def test_modal_train_lane_copies_dispatch_claim_ledger_to_remote_workspace() -> None:
    text = SOURCE.read_text()
    assert "claim_ledger_bytes: bytes" in text
    assert 'workspace / ".omx/state/active_lane_dispatch_claims.md"' in text
    assert "claim_path.parent.mkdir(parents=True, exist_ok=True)" in text
    assert "claim_path.write_bytes(claim_ledger_bytes)" in text
    assert '"T1_DISPATCH_CLAIMS_PATH": str(claim_path)' in text
    assert '"SCPP_DISPATCH_CLAIMS_PATH": str(claim_path)' in text
    assert "claims_path = repo_root / \".omx/state/active_lane_dispatch_claims.md\"" in text
    assert "claim_ledger_bytes = claims_path.read_bytes()" in text
    assert "fn.spawn(" in text


def test_modal_train_lane_claims_before_spawn_and_records_lane_id() -> None:
    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]

    assert '"scripts/remote_lane_t1_balle_endtoend.sh": "t1_balle_128k_endtoend"' in text
    assert '"scripts/remote_lane_scpp_stage1.sh": "lane_scpp_stage1_smoke_anchor"' in text
    assert "def _ensure_dispatch_claim(" in text
    assert "tools/claim_lane_dispatch.py" in text
    assert "--status" in text
    assert "active_dispatching" in text
    assert "aborting before Modal GPU spawn" in text
    assert main_src.index("_ensure_dispatch_claim(") < main_src.index("fn.spawn(")
    assert '"lane_id": resolved_lane_id' in text
    assert "from tac.deploy.modal.auth_eval import function_call_id" in text
    assert "call_id = function_call_id(fn_call)" in text
    assert "fn_call.object_id" not in text


def test_modal_train_lane_passes_mounted_git_custody_to_remote_scripts() -> None:
    text = SOURCE.read_text()

    assert "mounted_code_git_head: str" in text
    assert "mounted_code_git_branch: str" in text
    assert '"T1_MOUNTED_CODE_GIT_HEAD": mounted_code_git_head' in text
    assert '"T1_MOUNTED_CODE_GIT_BRANCH": mounted_code_git_branch' in text
    assert '"SCPP_MOUNTED_CODE_GIT_HEAD": mounted_code_git_head' in text
    assert '"SCPP_MOUNTED_CODE_GIT_BRANCH": mounted_code_git_branch' in text
    assert 'mounted_code_git_head = _git_value(repo_root, "rev-parse", "HEAD")' in text
    assert (
        'mounted_code_git_branch = _git_value(repo_root, "branch", "--show-current")'
        in text
    )
    assert "unable to resolve mounted git custody for Modal training" in text


def test_modal_train_lane_records_cost_band_metadata_without_score_authority() -> None:
    text = SOURCE.read_text()

    assert "cost_band_trainer: str = \"\"" in text
    assert "cost_band_epochs: int = 0" in text
    assert "cost_band_batch_size: int = 0" in text
    assert "FATAL: --cost-band-trainer is required" in text
    assert "FATAL: --cost-band-epochs must be positive" in text
    assert "FATAL: --cost-band-batch-size must be positive" in text
    assert '"schema": "modal_training_cost_anchor_metadata_v1"' in text
    assert '"score_claim": False' in text
    assert '"promotion_eligible": False' in text
    assert 'metadata["cost_band_anchor"] = cost_band_anchor' in text


def test_modal_train_lane_returns_experiments_results_artifacts() -> None:
    text = SOURCE.read_text()

    assert 'workspace / "experiments" / "results"' in text
