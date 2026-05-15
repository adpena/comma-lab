# SPDX-FileCopyrightText: 2026 comma-lab contributors
# SPDX-License-Identifier: MIT
"""Tests for Z4 + Z5 remote_lane scripts and operator-authorize wrappers.

Lane: lane_z4_z5_remote_lane_script_build_20260515
Source: GRAND-COUNCIL-AD-HOC-DISPATCH-UNBLOCK landing op-routable #2 (HIGH).

These tests validate the Z4 (cooperative-receiver loss) and Z5 (predictive-coding
world-model) remote_lane shell drivers + operator-authorize wrappers built in
parallel with the Z3 v2 smoke per the Time-Traveler L5 staircase.

Per CLAUDE.md "FORBIDDEN PATTERNS" + Catalog #163/#189/#167/#162 these scripts
must:
  - exist + be executable
  - pass `bash -n` syntax check
  - declare ``set -euo pipefail``
  - source the canonical bootstrap helper with ``REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1``
    sentinel (Catalog #163)
  - guard every optional-array expansion with ``${ARR[@]+"${ARR[@]}"}`` (Catalog #189)
  - dry-run cleanly under ``set -u`` macOS bash 3.2 (Catalog #189 runtime)
  - reference the canonical Catalog #167 smoke-before-full helper (operator-authorize)
  - reference the canonical recipe per Catalog #162
  - Catalog #146 trainer 3-arg contract is enforced at the trainer-side; here
    we verify the recipe-driven trainer flag-threading is consistent with the
    Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_RECIPES_DIR = _REPO_ROOT / ".omx" / "operator_authorize_recipes"


REMOTE_LANE_SCRIPTS = (
    "remote_lane_substrate_z4_cooperative_receiver_loss.sh",
    "remote_lane_substrate_z5_predictive_coding_world_model.sh",
)

OPERATOR_AUTHORIZE_WRAPPERS = (
    "operator_authorize_substrate_z4_cooperative_receiver_loss_modal_t4_dispatch.sh",
    "operator_authorize_substrate_z5_predictive_coding_world_model_modal_t4_dispatch.sh",
)

ALL_SCRIPTS = REMOTE_LANE_SCRIPTS + OPERATOR_AUTHORIZE_WRAPPERS


def _script_body(name: str) -> str:
    return (_SCRIPTS_DIR / name).read_text()


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_script_exists(name: str) -> None:
    p = _SCRIPTS_DIR / name
    assert p.is_file(), f"missing script: {p}"


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_script_is_executable(name: str) -> None:
    p = _SCRIPTS_DIR / name
    assert p.stat().st_mode & 0o111, f"not executable: {p}"


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_script_bash_syntax_clean(name: str) -> None:
    p = _SCRIPTS_DIR / name
    result = subprocess.run(
        ["bash", "-n", str(p)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"bash -n failed for {name}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_script_has_set_euo_pipefail(name: str) -> None:
    """Catalog #2 + Catalog #163 require strict shell discipline."""
    body = _script_body(name)
    assert "set -euo pipefail" in body, (
        f"{name} missing 'set -euo pipefail' (Catalog #2 + #163)"
    )


@pytest.mark.parametrize("name", REMOTE_LANE_SCRIPTS)
def test_remote_lane_uses_canonical_bootstrap_sentinel(name: str) -> None:
    """Catalog #163: remote_lane scripts MUST prepend REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1."""
    body = _script_body(name)
    assert "REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source" in body, (
        f"{name} missing Catalog #163 sentinel before sourcing canonical bootstrap"
    )
    assert "bootstrap_runtime_deps" in body, (
        f"{name} missing call to canonical bootstrap_runtime_deps()"
    )


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_script_guards_empty_arrays(name: str) -> None:
    """Catalog #189: every ``${ARR[@]}`` expansion must be guarded.

    Forms accepted: ``${ARR[@]+"${ARR[@]}"}`` (canonical) or no array
    expansion at all. Bare ``"${ARR[@]}"`` is rejected.
    """
    body = _script_body(name)
    # Look for any bare "${SOMETHING[@]}" that's NOT inside a +"..." guard
    import re
    bare = re.findall(r'(?<!\+)"(\$\{\w+\[@\]\})"', body)
    if bare:
        # Filter out occurrences that ARE inside the guard form: ${ARR[@]+"${ARR[@]}"}
        # The guard form is: ${ARR[@]+"${ARR[@]}"}
        guarded_form = re.findall(r'\$\{\w+\[@\]\+"(\$\{\w+\[@\]\})"\}', body)
        truly_bare = set(bare) - set(guarded_form)
        assert not truly_bare, (
            f"{name} has unguarded bare empty-array expansion (Catalog #189): {truly_bare}"
        )


@pytest.mark.parametrize("name", OPERATOR_AUTHORIZE_WRAPPERS)
def test_operator_authorize_wrapper_dry_run_clean(name: str) -> None:
    """Catalog #189 runtime: --dry-run must succeed under set -u + macOS bash 3.2.

    Empirical anchor: SIREN smoke timeout 2026-05-13 was caused by
    ``SMOKE_ARGS[@]: unbound variable`` under macOS bash 3.2.
    """
    p = _SCRIPTS_DIR / name
    result = subprocess.run(
        ["bash", str(p), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{name} --dry-run failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "--dry-run; no Modal dispatch" in result.stdout
    assert "would dispatch" in result.stdout


@pytest.mark.parametrize("name", OPERATOR_AUTHORIZE_WRAPPERS)
def test_operator_authorize_wrapper_uses_canonical_smoke_before_full(name: str) -> None:
    """Catalog #167: substrate dispatch must route through smoke-before-full helper."""
    body = _script_body(name)
    assert "tools/run_modal_smoke_before_full.py" in body, (
        f"{name} missing canonical Catalog #167 helper invocation"
    )
    assert "--recipe substrate_z" in body, (
        f"{name} missing --recipe argument to canonical helper"
    )


@pytest.mark.parametrize("name", OPERATOR_AUTHORIZE_WRAPPERS)
def test_operator_authorize_wrapper_recipe_exists(name: str) -> None:
    """Each wrapper's --recipe argument must point to an existing YAML."""
    body = _script_body(name)
    import re
    m = re.search(r'--recipe\s+(\S+)', body)
    assert m, f"{name} missing --recipe argument"
    recipe_name = m.group(1)
    recipe_path = _RECIPES_DIR / f"{recipe_name}.yaml"
    assert recipe_path.is_file(), (
        f"{name} references missing recipe: {recipe_path}"
    )


@pytest.mark.parametrize("name", REMOTE_LANE_SCRIPTS)
def test_remote_lane_writes_to_modal_results_under_modal_runtime(name: str) -> None:
    """Catalog #204: MODAL_RUNTIME=1 + DISPATCH_INSTANCE_JOB_ID -> /modal_results/<id>/output."""
    body = _script_body(name)
    assert "/modal_results/" in body, (
        f"{name} missing Catalog #204 durable provider output path"
    )
    assert 'MODAL_RUNTIME:-0' in body, (
        f"{name} missing MODAL_RUNTIME env-var check"
    )


@pytest.mark.parametrize("name", REMOTE_LANE_SCRIPTS)
def test_remote_lane_records_terminal_dispatch_claim(name: str) -> None:
    """CLAUDE.md CROSS-AGENT DISPATCH COORDINATION: terminal status must be appended."""
    body = _script_body(name)
    assert "claim_lane_dispatch.py" in body, (
        f"{name} missing terminal dispatch-claim record"
    )
    assert "append_terminal_claim" in body, (
        f"{name} missing canonical append_terminal_claim helper"
    )


@pytest.mark.parametrize("name", REMOTE_LANE_SCRIPTS)
def test_remote_lane_does_not_emit_contest_cuda_marker_unconditionally(name: str) -> None:
    """CLAUDE.md FORBIDDEN_PATTERNS: [contest-CUDA] only on validated score claim.

    The completion marker MUST gate [contest-CUDA] on
    auth_eval_score_claim_valid=true AND auth_eval_score_axis=contest_cuda.
    Default marker should be [training-artifact-no-score-claim] for smoke
    paths or non-validated runs.
    """
    body = _script_body(name)
    assert "training-artifact-no-score-claim" in body, (
        f"{name} missing default no-score-claim marker"
    )
    assert "auth_eval_score_claim_valid" in body, (
        f"{name} missing auth_eval_score_claim_valid gate before [contest-CUDA]"
    )
    assert 'auth_eval_score_axis") == "contest_cuda"' in body, (
        f"{name} missing axis check before [contest-CUDA]"
    )


@pytest.mark.parametrize("name", REMOTE_LANE_SCRIPTS)
def test_remote_lane_emits_heartbeat_loop(name: str) -> None:
    """CLAUDE.md "Remote code parity" requires heartbeat every N min."""
    body = _script_body(name)
    assert "HEARTBEAT_PID" in body, (
        f"{name} missing heartbeat PID tracking"
    )
    assert "heartbeat.log" in body, (
        f"{name} missing heartbeat log emission"
    )


@pytest.mark.parametrize("name", REMOTE_LANE_SCRIPTS)
def test_remote_lane_threads_required_input_video_path(name: str) -> None:
    """Catalog #151 + #152: --video-path MUST be threaded from env var ladder."""
    body = _script_body(name)
    if "z4" in name:
        assert "Z4_VIDEO_PATH" in body
        assert "--video-path" in body
    if "z5" in name:
        assert "Z5_VIDEO_PATH" in body
        assert "--video-path" in body


@pytest.mark.parametrize("name", REMOTE_LANE_SCRIPTS)
def test_remote_lane_exports_cublas_workspace_for_deterministic_inflate(name: str) -> None:
    """Catalog #224: CUBLAS_WORKSPACE_CONFIG + DALI_DISABLE_NVML before inflate."""
    body = _script_body(name)
    assert "CUBLAS_WORKSPACE_CONFIG" in body
    assert "DALI_DISABLE_NVML" in body
    assert "PYTORCH_CUDA_ALLOC_CONF" in body


def test_z4_remote_lane_threads_lambda_pixel() -> None:
    """Z4 cooperative-receiver substrate-specific flag (probe-disambiguator)."""
    body = _script_body("remote_lane_substrate_z4_cooperative_receiver_loss.sh")
    assert "Z4_LAMBDA_PIXEL" in body
    assert "--lambda-pixel" in body


def test_z4_remote_lane_threads_tier1_engineering_flags() -> None:
    """Z4 recipe Tier-1 env vars must become trainer CLI flags."""
    body = _script_body("remote_lane_substrate_z4_cooperative_receiver_loss.sh")
    expected = {
        "Z4_ENABLE_TF32": "--enable-tf32",
        "Z4_ENABLE_TORCH_COMPILE": "--enable-torch-compile",
        "Z4_ENABLE_GT_SCORER_CACHE": "--enable-gt-scorer-cache",
        "Z4_RELAX_DETERMINISM": "--relax-determinism-for-backward",
    }
    for env_name, flag in expected.items():
        assert env_name in body
        assert flag in body

    for args_name in (
        "TF32_FLAG_ARGS",
        "TORCH_COMPILE_FLAG_ARGS",
        "GT_SCORER_CACHE_FLAG_ARGS",
        "RELAX_DETERMINISM_FLAG_ARGS",
    ):
        assert f'${{{args_name}[@]+"${{{args_name}[@]}}"}}' in body


def test_z4_remote_lane_invokes_trainer_with_tier1_recipe_flags(tmp_path: Path) -> None:
    """Runtime proof: true recipe env values reach the trainer argv."""
    workspace = tmp_path / "workspace"
    trainer_dir = workspace / "experiments"
    trainer_dir.mkdir(parents=True)
    trainer_path = trainer_dir / "train_substrate_z4_cooperative_receiver_loss.py"
    trainer_path.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

argv = sys.argv[1:]
out = Path(argv[argv.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "argv.json").write_text(json.dumps(argv))
(out / "stats.json").write_text(json.dumps({"evidence_grade": "training-artifact-no-score-claim"}))
""",
    )
    trainer_path.chmod(0o755)

    output_dir = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "WORKSPACE": str(workspace),
            "PYBIN": sys.executable,
            "LOG_DIR": str(tmp_path / "logs"),
            "OUTPUT_DIR": str(output_dir),
            "Z4_OUTPUT_DIR": str(output_dir),
            "Z4_DISPATCH_INSTANCE_JOB_ID": "z4-test-job",
            "SMOKE_ONLY": "0",
            "Z4_ENABLE_AUTOCAST_FP16": "true",
            "Z4_ENABLE_TF32": "true",
            "Z4_ENABLE_TORCH_COMPILE": "true",
            "Z4_ENABLE_GT_SCORER_CACHE": "true",
            "Z4_RELAX_DETERMINISM": "true",
        }
    )

    stdout_path = tmp_path / "remote.stdout"
    stderr_path = tmp_path / "remote.stderr"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(
            ["bash", str(_SCRIPTS_DIR / "remote_lane_substrate_z4_cooperative_receiver_loss.sh")],
            stdout=stdout,
            stderr=stderr,
            text=True,
            cwd=_REPO_ROOT,
            env=env,
            timeout=30,
        )
    assert result.returncode == 0, (
        "Z4 remote lane script failed\n"
        f"stdout:\n{stdout_path.read_text()}\n"
        f"stderr:\n{stderr_path.read_text()}"
    )

    argv = json.loads((output_dir / "argv.json").read_text())
    for flag in (
        "--enable-autocast-fp16",
        "--enable-tf32",
        "--enable-torch-compile",
        "--enable-gt-scorer-cache",
        "--relax-determinism-for-backward",
    ):
        assert flag in argv


def test_z4_full_trainer_selects_checkpoint_by_score_aware_validation_loss() -> None:
    """Z4 best checkpoint must not regress to pixel-MSE proxy selection."""
    body = (_REPO_ROOT / "experiments" / "train_substrate_z4_cooperative_receiver_loss.py").read_text()
    assert "val_score_loss, val_parts = score_loss(" in body
    assert "noise_std=0.0" in body
    assert '"val_loss_metric": "score_aware_cooperative_receiver_loss"' in body
    assert '"pixel_val_loss": pixel_val_loss_float' in body
    assert "pixel_val_loss={pixel_val_loss_float:.5f}" in body
    assert "Use pixel-MSE proxy as val signal" not in body


def test_z5_remote_lane_threads_predictor_layers_and_identity_predictor() -> None:
    """Z5 predictive-coding substrate-specific flags (probe-disambiguator)."""
    body = _script_body("remote_lane_substrate_z5_predictive_coding_world_model.sh")
    assert "Z5_PREDICTOR_NUM_LAYERS" in body
    assert "--predictor-num-layers" in body
    assert "Z5_IDENTITY_PREDICTOR" in body
    assert "--identity-predictor" in body
    assert "Z5_LAMBDA_RESIDUAL_ENTROPY" in body
    assert "--lambda-residual-entropy" in body
