import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "experiments" / "modal_phase_a1_score_gradient_pr101.py"


def _source() -> str:
    return TOOL.read_text()


def _executable_source() -> str:
    text = _source()
    module = ast.parse(text)
    if not module.body:
        return text
    first = module.body[0]
    if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant):
        return text
    if not isinstance(first.value.value, str):
        return text
    lines = text.splitlines()
    start = first.lineno - 1
    end = first.end_lineno or first.lineno
    return "\n".join(lines[:start] + lines[end:])


def test_modal_phase_a1_dispatcher_is_cuda_auth_eval_only() -> None:
    text = _source()
    executable = _executable_source()

    assert "contest_auth_eval.py" in text
    assert '"--device", "cuda"' in text
    assert "AUTH_EVAL_DEVICE=cpu" not in executable
    assert "MODAL_AUTH_EVAL_ADVISORY_ONLY" not in executable
    assert '"auth_eval_advisory_only": False' in text


def test_modal_phase_a1_claims_lane_before_spawn() -> None:
    text = _source()

    claim_idx = text.index("claim_rc = _claim_lane(")
    spawn_idx = text.index("call = run_phase_a1_t4.spawn(")
    assert claim_idx < spawn_idx
    assert "aborting before GPU spend" in text


def test_modal_phase_a1_requires_t4_dali_and_nvdec_preflight() -> None:
    text = _source()

    assert "nvidia.dali" in text
    assert "nvdec_probe_passed" in text
    assert "gpu_t4_match" in text
    assert "refusing CPU fallback" in text


def test_modal_phase_a1_recover_preserves_harvest_command_and_ttl() -> None:
    text = _source()

    assert "Modal result-cache TTL" in text
    assert "tools/harvest_modal_calls.py" in text
    assert "recover --label <instance_job_id>" in text
