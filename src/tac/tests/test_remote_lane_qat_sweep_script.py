"""Structure tests for scripts/remote_lane_qat_sweep.sh.

Same hardness as test_remote_lane_a_sweep_script.py, adapted for QAT-specific
placeholders + qat_finetune.py argparse verification.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_qat_sweep.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


def test_script_exists():
    assert SCRIPT.exists(), f"missing canonical sweep script: {SCRIPT}"


def test_script_executable():
    import os
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} must be chmod +x"


# ---------- shell discipline --------------------------------------------


def test_set_e_present(script_text: str):
    has_set_e = bool(re.search(r"^set -[eu]*e[uo]*\s", script_text, re.MULTILINE))
    assert has_set_e


def test_pipefail_present(script_text: str):
    assert "pipefail" in script_text


def test_no_shell_zip_binary(script_text: str):
    code_lines = []
    for line in script_text.splitlines():
        if line.strip().startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    bad_match = re.search(r"(^|[\s;&|`(])zip\s+(?!file)", code)
    assert not bad_match


def test_uses_python_zipfile(script_text: str):
    assert "zipfile.ZipFile" in script_text


# ---------- non-negotiables --------------------------------------------


def test_no_mps_or_cpu_device(script_text: str):
    assert not re.search(r"--device\s+mps\b", script_text)
    assert not re.search(r"--device\s+cpu\b", script_text)


def test_nvdec_probe_called_before_gpu_work(script_text: str):
    probe_idx = script_text.find("probe_nvdec.sh")
    assert probe_idx > 0
    # Match the actual python invocation, not stray mentions in comments.
    qat_invoke = re.search(r'\$PYBIN[^\n]*qat_finetune\.py', script_text)
    if qat_invoke:
        assert probe_idx < qat_invoke.start(), \
            "NVDEC probe must precede qat_finetune.py invocation"


def test_uv_pip_install_optuna(script_text: str):
    assert re.search(r"uv pip install.*optuna", script_text)
    bad = re.search(r"^\s*pip\s+install\s+optuna\b", script_text, re.MULTILINE)
    assert not bad


# ---------- placeholder discipline -------------------------------------


def test_all_search_space_placeholders_present(script_text: str):
    required_placeholders = [
        "__PARAM_INT8_WARMUP_EPOCHS__",
        "__PARAM_FP4_EPOCHS__",
        "__PARAM_LR__",
        "__PARAM_LR_SCHEDULE__",
        "__PARAM_DEVICE__",
    ]
    for tok in required_placeholders:
        assert tok in script_text, f"missing template placeholder {tok}"


def test_sweep_provenance_tags_present(script_text: str):
    for tag in ("__SWEEP_NAME__", "__SWEEP_TRIAL_NUMBER__", "__SWEEP_SEARCH_SPACE_HASH__"):
        assert tag in script_text


def test_provenance_marks_is_sweep(script_text: str):
    assert "'is_sweep': True" in script_text or "\"is_sweep\": True" in script_text


def test_provenance_records_n_trials(script_text: str):
    assert "n_trials" in script_text


# ---------- safety gates ------------------------------------------------


def test_archive_size_validated_before_auth_eval(script_text: str):
    auth_idx = script_text.find("contest_auth_eval.py")
    assert auth_idx > 0
    pre = script_text[:auth_idx]
    has_empty_check = bool(re.search(r'-z\s+"\$\{?ARCHIVE_BYTES', pre))
    has_zero_check = bool(re.search(r'\$ARCHIVE_BYTES"?\s*-le\s*0', pre))
    assert has_empty_check or has_zero_check


def test_auth_eval_log_gated_on_result_json(script_text: str):
    assert re.search(r"grep\s+-q\s+'?\^?RESULT_JSON", script_text)


def test_writes_sidecar_result_json(script_text: str):
    assert ".result.json" in script_text


# ---------- argparse-flag verification (NEVER invent flags) -----------


def test_qat_finetune_flags_exist_in_target(script_text: str):
    """Every flag passed to qat_finetune.py must exist in its argparse.

    This is the dead-flag wiring trap (CLAUDE.md non-negotiable). The Lane A
    chain shipped 2 rounds with --auth-eval-masks that didn't exist; we
    catch that pattern here at test time.
    """
    target = REPO / "experiments" / "qat_finetune.py"
    target_text = target.read_text()

    # Anchor on actual python invocation (matches `"$PYBIN" -u ... qat_finetune.py ...`),
    # not stray mentions in comment headers. The block ends at the next
    # `$PYBIN` invocation OR the next pipe-into-tee (whichever comes first),
    # so we capture only the flag list passed to qat_finetune.
    qat_block_match = re.search(
        r'\$PYBIN[^\n]*qat_finetune\.py(.*?)(?=\$PYBIN|\|\s*tee\b|\Z)',
        script_text,
        re.S,
    )
    assert qat_block_match, "could not locate qat_finetune.py python invocation"
    block = qat_block_match.group(1)
    flags_passed = set(re.findall(r"--([a-z][a-z0-9-]+)", block))
    for flag in flags_passed:
        pattern = rf'add_argument\(\s*"--{re.escape(flag)}"'
        assert re.search(pattern, target_text), (
            f"--{flag} passed to qat_finetune.py but not in its argparse "
            f"(dead-flag wiring trap)"
        )


def test_contest_auth_eval_flags_exist_in_target(script_text: str):
    target = REPO / "experiments" / "contest_auth_eval.py"
    target_text = target.read_text()
    eval_block_match = re.search(
        r'\$PYBIN[^\n]*contest_auth_eval\.py(.*?)(?=\$PYBIN|\|\s*tee\b|\Z)',
        script_text,
        re.S,
    )
    assert eval_block_match
    block = eval_block_match.group(1)
    flags_passed = set(re.findall(r"--([a-z][a-z0-9-]+)", block))
    for flag in flags_passed:
        pattern = rf'add_argument\(\s*"--{re.escape(flag)}"'
        assert re.search(pattern, target_text), (
            f"--{flag} passed to contest_auth_eval.py but not in its argparse"
        )


# ---------- archive integrity ------------------------------------------


def test_archive_input_files_asserted(script_text: str):
    py_idx = script_text.find("zipfile.ZipFile")
    assert py_idx > 0
    block_end = script_text.find('print(', py_idx)
    block = script_text[py_idx:block_end]
    assert "assert os.path.isfile" in block


# ---------- QAT sweep ALSO needs a sweep_lane_qat.py companion test --


def test_sweep_companion_exists():
    companion = REPO / "experiments" / "sweep_lane_qat.py"
    assert companion.exists(), f"missing sweep driver: {companion}"
