"""Structure tests for scripts/remote_lane_a_sweep.sh.

This template-cum-orchestrator script is the canonical Lane A-Sweep entry.
The same lessons LANE-B taught us (set -e, no shell zip, archive size
validation, RESULT_JSON gate) apply here AND additional sweep-specific
discipline:

  * Every search-space placeholder must appear in the trial-mode block.
  * Provenance must include `is_sweep: true` + `n_trials` + `sweep_name`.
  * Optuna must be installed via `uv pip` (CLAUDE.md uv-mandatory).
  * NVDEC probe BEFORE GPU spend (memory: feedback_vastai_nvdec_host_variation).
  * `--device cuda` literal (CLAUDE.md non-negotiable; no MPS fallback).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_a_sweep.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


def test_script_exists():
    assert SCRIPT.exists(), f"missing canonical sweep script: {SCRIPT}"


def test_script_executable():
    import os
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} must be chmod +x"


# ---------- shell discipline (CLAUDE.md non-negotiables) ----------------


def test_set_e_present(script_text: str):
    """LANE-B taught us: set -e mandatory to abort on first failure."""
    has_set_e = bool(re.search(r"^set -[eu]*e[uo]*\s", script_text, re.MULTILINE))
    assert has_set_e, "script must use `set -euo pipefail` (e flag mandatory)"


def test_pipefail_present(script_text: str):
    assert "pipefail" in script_text, "script must use pipefail"


def test_no_shell_zip_binary(script_text: str):
    """PyTorch container has no `zip` — use Python zipfile."""
    code_lines = []
    for line in script_text.splitlines():
        if line.strip().startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    bad_match = re.search(r"(^|[\s;&|`(])zip\s+(?!file)", code)
    assert not bad_match, (
        f"shell `zip` is missing on PyTorch container. "
        f"Match: {bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    assert "zipfile.ZipFile" in script_text


# ---------- non-negotiables from CLAUDE.md ------------------------------


def test_no_mps_or_cpu_device_default(script_text: str):
    """Device must be cuda (or __PARAM_DEVICE__ which is fixed to cuda).

    NEVER `--device mps` or `--device cpu` literally appearing in the script.
    """
    assert not re.search(r"--device\s+mps\b", script_text), \
        "remote sweep must not reference --device mps"
    assert not re.search(r"--device\s+cpu\b", script_text), \
        "remote sweep must not reference --device cpu"


def test_nvdec_probe_called_before_gpu_work(script_text: str):
    """Stage 0 must be NVDEC probe (catches bad host in 5s, saves $0.20)."""
    probe_idx = script_text.find("probe_nvdec.sh")
    assert probe_idx > 0, "must call probe_nvdec.sh"
    # Probe must come BEFORE the actual optimize_poses python invocation.
    pose_invoke = re.search(r'\$PYBIN[^\n]*optimize_poses\.py', script_text)
    if pose_invoke:
        assert probe_idx < pose_invoke.start(), "NVDEC probe must precede pose TTO"


def test_uv_pip_install_optuna(script_text: str):
    """CLAUDE.md uv-mandatory: install optuna via uv, not raw pip."""
    assert re.search(r"uv pip install.*optuna", script_text), \
        "must install optuna via `uv pip install`, not pip"
    # Negative: never raw pip (besides `uv pip`).
    bad = re.search(r"^\s*pip\s+install\s+optuna\b", script_text, re.MULTILINE)
    assert not bad, "must NOT use raw pip (use uv pip)"


# ---------- placeholder discipline (template integrity) ----------------


def test_all_search_space_placeholders_present(script_text: str):
    """The template MUST have a placeholder for every Lane A search-space key."""
    required_placeholders = [
        "__PARAM_TTO_STEPS__",
        "__PARAM_BATCH_PAIRS__",
        "__PARAM_TTO_LR__",
        "__PARAM_POSETTO_NOISE_STD__",
        "__PARAM_EVAL_ROUNDTRIP__",
        "__PARAM_DEVICE__",
    ]
    for tok in required_placeholders:
        assert tok in script_text, f"missing template placeholder {tok}"


def test_sweep_provenance_tags_present(script_text: str):
    for tag in ("__SWEEP_NAME__", "__SWEEP_TRIAL_NUMBER__", "__SWEEP_SEARCH_SPACE_HASH__"):
        assert tag in script_text, f"missing sweep provenance tag {tag}"


def test_provenance_marks_is_sweep(script_text: str):
    """provenance.json must record `is_sweep: true` so log scrapers can split."""
    assert "'is_sweep': True" in script_text or "\"is_sweep\": True" in script_text or \
           "'is_sweep': true" in script_text.lower(), \
        "provenance.json must include is_sweep: true"


def test_provenance_records_n_trials(script_text: str):
    assert "n_trials" in script_text, "provenance must record n_trials"


def test_provenance_records_sweep_name(script_text: str):
    assert "'sweep_name'" in script_text or '"sweep_name"' in script_text


# ---------- safety gates --------------------------------------------------


def test_archive_size_validated_before_auth_eval(script_text: str):
    """ARCHIVE_BYTES must be checked for empty/zero BEFORE auth_eval (LANE-B)."""
    auth_idx = script_text.find("contest_auth_eval.py")
    assert auth_idx > 0
    pre = script_text[:auth_idx]
    has_empty_check = bool(re.search(r'-z\s+"\$\{?ARCHIVE_BYTES', pre))
    has_zero_check = bool(re.search(r'\$ARCHIVE_BYTES"?\s*-le\s*0', pre))
    assert has_empty_check or has_zero_check, \
        "ARCHIVE_BYTES must be validated before auth_eval invocation"


def test_auth_eval_log_gated_on_result_json(script_text: str):
    """The script must verify RESULT_JSON appears before declaring success."""
    assert re.search(r"grep\s+-q\s+'?\^?RESULT_JSON", script_text), \
        "must validate auth_eval log contains RESULT_JSON"


def test_writes_sidecar_result_json(script_text: str):
    """Trial mode must write a sidecar `.result.json` for the sweep parser."""
    assert ".result.json" in script_text, \
        "trial mode must write sidecar <script>.result.json"


# ---------- argparse-flag verification (NEVER invent flags) -------------


def test_optimize_poses_flags_exist_in_target(script_text: str):
    """Every flag this script passes to optimize_poses.py must exist."""
    target = REPO / "experiments" / "optimize_poses.py"
    target_text = target.read_text()
    # Find every --flag this script passes to optimize_poses.py.
    pose_block_match = re.search(
        r'\$PYBIN[^\n]*optimize_poses\.py(.*?)(?=\$PYBIN|\|\s*tee\b|\Z)',
        script_text,
        re.S,
    )
    assert pose_block_match, "could not locate optimize_poses.py python invocation"
    block = pose_block_match.group(1)
    flags_passed = set(re.findall(r"--([a-z][a-z0-9-]+)", block))
    for flag in flags_passed:
        # Search target for `--flag` exactly.
        pattern = rf'add_argument\(\s*"--{re.escape(flag)}"'
        assert re.search(pattern, target_text), (
            f"--{flag} passed to optimize_poses.py but not in its argparse"
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


# ---------- archive integrity (Python zipfile asserts each input) ------


def test_archive_input_files_asserted(script_text: str):
    py_idx = script_text.find("zipfile.ZipFile")
    assert py_idx > 0
    block_end = script_text.find('print(', py_idx)
    block = script_text[py_idx:block_end]
    assert "assert os.path.isfile" in block, (
        "zipfile builder must assert each input file exists "
        "(prevents tiny-archive-from-missing-renderer trap)"
    )
