"""Dedicated tests for Catalog #152
(`check_operator_wrapper_validates_required_input_files_pre_dispatch`)
and the canonical wrapper-side validator tool
(`tools/validate_dispatch_required_inputs.py`).

Bug-class anchor (the reason these tests exist):
    2026-05-12T17:12 Modal A100 dispatch (call_id fc-01KREJST89QHFRWJXHAKXD850C)
    burned $0.016 in 15s before crashing with rc=1 because
    `--pr95-parity-profile` pointed to a non-existent file on the Modal worker.
    A local 100ms validation would have caught it before the meter started.

Acceptance rules tested:
  (a) literal `[ -f "$ENV" ]` / `test -f "$ENV"` shell test referencing the
      env-var
  (b) `tools/validate_dispatch_required_inputs.py` invocation in the body
  (c) same-line `# REQUIRED_INPUT_VALIDATION_OK:<reason>` waiver

Plus:
  - Dispatch-only scope (utility scripts without dispatch tokens are skipped)
  - Indirect trainer detection via `--lane-script scripts/X.sh`
  - Vendored intake exclusion (mirrors Catalog #151 R6)
  - Strict-mode raises PreflightError
  - Live-repo invariant: 0 violations on landing
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_152_collect_required_input_flags,
    _check_152_collect_waivers,
    _check_152_extract_indirect_trainer_paths,
    _check_152_wrapper_has_dispatch,
    _check_152_wrapper_validates_required_input,
    check_operator_wrapper_validates_required_input_files_pre_dispatch,
)


# -- Fixture helpers ---------------------------------------------------------

def _write_trainer(tmp: Path, name: str, manifest_src: str = "") -> Path:
    path = tmp / "experiments" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest_src or "# trainer with no manifest\n")
    return path


def _write_wrapper(tmp: Path, name: str, body: str) -> Path:
    path = tmp / "scripts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


_TRAINER_WITH_REQ_INPUT = '''
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--config-file": {
        "env": "MY_CONFIG",
        "rationale": "required config JSON",
        "default": ".omx/research/my_config.json",
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": "python tools/build_my_config.py",
    },
    "--enable-feature-x": {
        "env": "FEATURE_X",
        "rationale": "non-input semantic gate (not a file)",
        "default": None,
        "satisfied_by_profile": (),
        "requires": (),
    },
}
'''


_TRAINER_NO_REQ_INPUT = '''
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--enable-feature-y": {
        "env": "FEATURE_Y",
        "rationale": "semantic gate, no file",
        "default": None,
        "satisfied_by_profile": (),
        "requires": (),
    },
}
'''


# -- Trainer-manifest extraction --------------------------------------------

def test_collect_required_input_flags_filters_to_required_input_true(
    tmp_path: Path,
) -> None:
    p = _write_trainer(tmp_path, "train_foo.py", _TRAINER_WITH_REQ_INPUT)
    out = _check_152_collect_required_input_flags(p)
    assert set(out.keys()) == {"--config-file"}, (
        f"expected only --config-file (required_input_file=True), got {out}"
    )


def test_collect_required_input_flags_empty_when_no_required_input(
    tmp_path: Path,
) -> None:
    p = _write_trainer(tmp_path, "train_bar.py", _TRAINER_NO_REQ_INPUT)
    assert _check_152_collect_required_input_flags(p) == {}


def test_collect_required_input_flags_no_manifest_returns_empty(
    tmp_path: Path,
) -> None:
    p = _write_trainer(tmp_path, "train_bare.py", "def main(): pass\n")
    assert _check_152_collect_required_input_flags(p) == {}


# -- Dispatch-token scope gate ----------------------------------------------

def test_dispatch_token_modal_run() -> None:
    text = '.venv/bin/modal run --detach experiments/modal_train_lane.py\n'
    assert _check_152_wrapper_has_dispatch(text)


def test_dispatch_token_launch_lane_vastai() -> None:
    text = 'python scripts/launch_lane_on_vastai.py --gpu RTX_4090\n'
    assert _check_152_wrapper_has_dispatch(text)


def test_dispatch_token_lightning_run() -> None:
    text = 'lightning run --gpu T4 something\n'
    assert _check_152_wrapper_has_dispatch(text)


def test_no_dispatch_token_skips_wrapper() -> None:
    text = 'echo "hello"\npython tools/analyze.py --in foo.json\n'
    assert not _check_152_wrapper_has_dispatch(text)


# -- Acceptance: shell `[ -f "$ENV" ]` test ---------------------------------

def test_acceptance_shell_existence_test_bracket_form() -> None:
    text = '[ -f "$MY_CONFIG" ] || exit 1\n'
    assert _check_152_wrapper_validates_required_input(text, "MY_CONFIG")


def test_acceptance_shell_existence_test_braced_form() -> None:
    text = '[ -f "${MY_CONFIG:-default}" ] || exit 1\n'
    assert _check_152_wrapper_validates_required_input(text, "MY_CONFIG")


def test_acceptance_shell_negated_existence_test() -> None:
    text = 'if [ ! -f "$MY_CONFIG" ]; then echo FATAL; exit 1; fi\n'
    assert _check_152_wrapper_validates_required_input(text, "MY_CONFIG")


def test_acceptance_posix_test_command() -> None:
    text = 'test -f "$MY_CONFIG" || exit 7\n'
    assert _check_152_wrapper_validates_required_input(text, "MY_CONFIG")


def test_rejection_existence_test_wrong_env_var() -> None:
    text = '[ -f "$OTHER_CONFIG" ] || exit 1\n'
    assert not _check_152_wrapper_validates_required_input(text, "MY_CONFIG")


def test_rejection_bare_env_var_reference_no_existence_test() -> None:
    # Just echoing the env-var is not validation.
    text = 'echo "Using $MY_CONFIG"\n'
    assert not _check_152_wrapper_validates_required_input(text, "MY_CONFIG")


# -- Acceptance: canonical validator-tool invocation ------------------------

def test_acceptance_validator_tool_invocation() -> None:
    text = (
        'python tools/validate_dispatch_required_inputs.py '
        '--trainer experiments/train_x.py || exit 7\n'
    )
    assert _check_152_wrapper_validates_required_input(text, "MY_CONFIG")


def test_acceptance_validator_tool_invocation_satisfies_any_env() -> None:
    """The canonical tool covers ALL required-input flags by manifest read,
    so any env-var that the tool covers is implicitly satisfied."""
    text = (
        '.venv/bin/python tools/validate_dispatch_required_inputs.py '
        '--trainer experiments/train_x.py\n'
    )
    assert _check_152_wrapper_validates_required_input(text, "ANY_ENV")
    assert _check_152_wrapper_validates_required_input(text, "MY_CONFIG")


# -- Same-line waiver -------------------------------------------------------

def test_waiver_collection_single() -> None:
    text = 'modal run x.py  # REQUIRED_INPUT_VALIDATION_OK:--config-file:smoke-only\n'
    waivers = _check_152_collect_waivers(text)
    # The waiver captures the full reason text — semantics handle the
    # `token:reason` split at validation time.
    assert any("--config-file" in w for w in waivers)


def test_waiver_collection_with_env_var() -> None:
    text = (
        'modal run x.py  '
        '# REQUIRED_INPUT_VALIDATION_OK:T1_PR95_PARITY_PROFILE:smoke-only\n'
    )
    waivers = _check_152_collect_waivers(text)
    assert any(w.startswith("T1_PR95_PARITY_PROFILE") for w in waivers)


def test_waiver_blanket_all() -> None:
    text = 'modal run x.py  # REQUIRED_INPUT_VALIDATION_OK:ALL:smoke-only\n'
    waivers = _check_152_collect_waivers(text)
    assert any(w.startswith("ALL") for w in waivers)


# -- Indirect lane-script detection -----------------------------------------

def test_extract_indirect_lane_script_paths() -> None:
    text = (
        'modal run experiments/modal_train_lane.py '
        '--lane-script scripts/remote_lane_t1.sh --gpu T4\n'
    )
    out = _check_152_extract_indirect_trainer_paths(text)
    assert out == ["scripts/remote_lane_t1.sh"]


def test_extract_indirect_lane_script_equals_form() -> None:
    text = 'modal run x.py --lane-script=scripts/remote_lane_y.sh\n'
    out = _check_152_extract_indirect_trainer_paths(text)
    assert out == ["scripts/remote_lane_y.sh"]


def test_extract_no_lane_script() -> None:
    text = 'modal run x.py --gpu T4 --timeout-hours 2\n'
    assert _check_152_extract_indirect_trainer_paths(text) == []


# -- End-to-end (whole-check) tests -----------------------------------------

def test_dispatch_wrapper_missing_validation_is_violation(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _TRAINER_WITH_REQ_INPUT)
    inner = tmp_path / "scripts" / "remote_lane_inner.sh"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text(
        '#!/bin/bash\n"$PYBIN" -u experiments/train_foo.py --config-file "$MY_CONFIG"\n'
    )
    _write_wrapper(
        tmp_path,
        "operator_authorize_x.sh",
        '#!/bin/bash\n'
        'modal run experiments/modal_train_lane.py '
        '--lane-script scripts/remote_lane_inner.sh --gpu T4\n',
    )
    out = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(out) == 1, f"expected 1 violation, got {out}"
    assert "--config-file" in out[0]
    assert "MY_CONFIG" in out[0]


def test_dispatch_wrapper_with_validator_tool_invocation_passes(
    tmp_path: Path,
) -> None:
    _write_trainer(tmp_path, "train_foo.py", _TRAINER_WITH_REQ_INPUT)
    inner = tmp_path / "scripts" / "remote_lane_inner.sh"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text(
        '#!/bin/bash\n"$PYBIN" -u experiments/train_foo.py --config-file "$MY_CONFIG"\n'
    )
    _write_wrapper(
        tmp_path,
        "operator_authorize_x.sh",
        '#!/bin/bash\n'
        'python tools/validate_dispatch_required_inputs.py '
        '--trainer experiments/train_foo.py || exit 7\n'
        'modal run experiments/modal_train_lane.py '
        '--lane-script scripts/remote_lane_inner.sh --gpu T4\n',
    )
    out = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], (
        f"validator tool invocation should satisfy; got {out}"
    )


def test_dispatch_wrapper_with_shell_existence_test_passes(
    tmp_path: Path,
) -> None:
    _write_trainer(tmp_path, "train_foo.py", _TRAINER_WITH_REQ_INPUT)
    inner = tmp_path / "scripts" / "remote_lane_inner.sh"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text(
        '#!/bin/bash\n"$PYBIN" -u experiments/train_foo.py --config-file "$MY_CONFIG"\n'
    )
    _write_wrapper(
        tmp_path,
        "operator_authorize_x.sh",
        '#!/bin/bash\n'
        '[ -f "$MY_CONFIG" ] || { echo FATAL; exit 1; }\n'
        'modal run experiments/modal_train_lane.py '
        '--lane-script scripts/remote_lane_inner.sh --gpu T4\n',
    )
    out = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], f"shell existence test should satisfy; got {out}"


def test_dispatch_wrapper_with_explicit_flag_waiver_passes(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _TRAINER_WITH_REQ_INPUT)
    inner = tmp_path / "scripts" / "remote_lane_inner.sh"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text(
        '#!/bin/bash\n"$PYBIN" -u experiments/train_foo.py --config-file "$MY_CONFIG"\n'
    )
    # Per-flag waiver explicitly names `--config-file`.
    _write_wrapper(
        tmp_path,
        "operator_authorize_x.sh",
        '#!/bin/bash\n'
        'modal run experiments/modal_train_lane.py '
        '--lane-script scripts/remote_lane_inner.sh --gpu T4  '
        '# REQUIRED_INPUT_VALIDATION_OK:--config-file:smoke-only-input-not-needed\n',
    )
    out = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], f"per-flag waiver should satisfy; got {out}"


def test_dispatch_wrapper_with_blanket_all_waiver_passes(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _TRAINER_WITH_REQ_INPUT)
    inner = tmp_path / "scripts" / "remote_lane_inner.sh"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text(
        '#!/bin/bash\n"$PYBIN" -u experiments/train_foo.py --config-file "$MY_CONFIG"\n'
    )
    _write_wrapper(
        tmp_path,
        "operator_authorize_x.sh",
        '#!/bin/bash\n'
        'modal run experiments/modal_train_lane.py '
        '--lane-script scripts/remote_lane_inner.sh --gpu T4  '
        '# REQUIRED_INPUT_VALIDATION_OK:ALL:smoke-only\n',
    )
    out = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], f"ALL waiver should satisfy; got {out}"


def test_unrelated_reason_only_waiver_does_NOT_blanket(tmp_path: Path) -> None:
    """A waiver token that names a reason but NOT the flag/env/ALL sentinel
    must NOT silently waive a real flag — per CLAUDE.md "Comment-only contracts
    are FORBIDDEN" the operator must explicitly name what they're waiving."""
    _write_trainer(tmp_path, "train_foo.py", _TRAINER_WITH_REQ_INPUT)
    inner = tmp_path / "scripts" / "remote_lane_inner.sh"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text(
        '#!/bin/bash\n"$PYBIN" -u experiments/train_foo.py --config-file "$MY_CONFIG"\n'
    )
    _write_wrapper(
        tmp_path,
        "operator_authorize_x.sh",
        '#!/bin/bash\n'
        'modal run experiments/modal_train_lane.py '
        '--lane-script scripts/remote_lane_inner.sh --gpu T4  '
        '# REQUIRED_INPUT_VALIDATION_OK:smoke-only-just-a-reason\n',
    )
    out = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(out) == 1, (
        f"reason-only waiver without flag/env/ALL token must NOT silently "
        f"waive; got {out}"
    )


def test_no_dispatch_token_skips_wrapper_entirely(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _TRAINER_WITH_REQ_INPUT)
    # Wrapper has no dispatch token → out of scope.
    _write_wrapper(
        tmp_path,
        "utility_x.sh",
        '#!/bin/bash\n'
        'echo "Analyzing experiments/train_foo.py"\n'
        'python tools/analyze.py --trainer experiments/train_foo.py\n',
    )
    out = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], f"utility script without dispatch should be skipped; got {out}"


def test_trainer_with_no_required_input_flags_is_fail_open(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_no_inputs.py", _TRAINER_NO_REQ_INPUT)
    inner = tmp_path / "scripts" / "remote_lane_inner.sh"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text(
        '#!/bin/bash\n"$PYBIN" -u experiments/train_no_inputs.py\n'
    )
    _write_wrapper(
        tmp_path,
        "operator_authorize_x.sh",
        '#!/bin/bash\n'
        'modal run experiments/modal_train_lane.py '
        '--lane-script scripts/remote_lane_inner.sh --gpu T4\n',
    )
    out = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], (
        f"trainer with no required_input_file flags should not violate; got {out}"
    )


def test_direct_invocation_no_lane_script_indirection(tmp_path: Path) -> None:
    """A wrapper that invokes the trainer DIRECTLY (not via --lane-script)
    still triggers the check when it has a dispatch token."""
    _write_trainer(tmp_path, "train_foo.py", _TRAINER_WITH_REQ_INPUT)
    _write_wrapper(
        tmp_path,
        "remote_lane_direct.sh",
        '#!/bin/bash\n'
        'modal run experiments/train_foo.py --config-file "$MY_CONFIG"\n',
    )
    out = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(out) == 1, f"direct invocation should be in scope, got {out}"


# -- R6 vendored intake exclusion -------------------------------------------

def test_R6_excludes_public_pr_intake(tmp_path: Path) -> None:
    intake_dir = tmp_path / "experiments" / "results" / "public_pr107_intake_codex"
    intake_dir.mkdir(parents=True)
    _write_trainer(tmp_path, "train_foo.py", _TRAINER_WITH_REQ_INPUT)
    (intake_dir / "wrapper.sh").write_text(
        '#!/bin/bash\nmodal run experiments/train_foo.py --gpu T4\n'
    )
    out = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert out == [], "vendored intake clones must be excluded"


# -- Strict-flip behavior ---------------------------------------------------

def test_strict_raises_on_violation(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _TRAINER_WITH_REQ_INPUT)
    inner = tmp_path / "scripts" / "remote_lane_inner.sh"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text(
        '#!/bin/bash\n"$PYBIN" -u experiments/train_foo.py --config-file "$MY_CONFIG"\n'
    )
    _write_wrapper(
        tmp_path,
        "operator_authorize_x.sh",
        '#!/bin/bash\n'
        'modal run experiments/modal_train_lane.py '
        '--lane-script scripts/remote_lane_inner.sh --gpu T4\n',
    )
    with pytest.raises(
        PreflightError,
        match="check_operator_wrapper_validates_required_input_files_pre_dispatch",
    ):
        check_operator_wrapper_validates_required_input_files_pre_dispatch(
            repo_root=tmp_path, strict=True, verbose=False,
        )


def test_strict_passes_on_clean(tmp_path: Path) -> None:
    _write_trainer(tmp_path, "train_foo.py", _TRAINER_WITH_REQ_INPUT)
    inner = tmp_path / "scripts" / "remote_lane_inner.sh"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text(
        '#!/bin/bash\n"$PYBIN" -u experiments/train_foo.py --config-file "$MY_CONFIG"\n'
    )
    _write_wrapper(
        tmp_path,
        "operator_authorize_x.sh",
        '#!/bin/bash\n'
        'python tools/validate_dispatch_required_inputs.py '
        '--trainer experiments/train_foo.py || exit 7\n'
        'modal run experiments/modal_train_lane.py '
        '--lane-script scripts/remote_lane_inner.sh --gpu T4\n',
    )
    out = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=tmp_path, strict=True, verbose=False,
    )
    assert out == []


# -- Live-repo invariant: STRICT @ 0 ---------------------------------------

def test_live_repo_strict_at_zero() -> None:
    """The canonical assertion: Catalog #152 lands strict-from-byte-one
    because the T1 Ballé operator-authorize wrapper invokes the canonical
    validator tool in the same commit batch. If this test fails, either:
      (a) someone added a `required_input_file=True` flag to a trainer
          without updating the dispatch wrapper to validate it, or
      (b) someone removed the validator-tool invocation from the T1 Ballé
          wrapper (regression of the structural fix)."""
    out = check_operator_wrapper_validates_required_input_files_pre_dispatch(
        repo_root=Path.cwd(), strict=False, verbose=False,
    )
    assert out == [], (
        f"Catalog #152 live violations: {len(out)}\n  "
        + "\n  ".join(out[:5])
    )


# -- tools/validate_dispatch_required_inputs.py CLI tests -------------------

_VALIDATOR_TOOL = Path("tools/validate_dispatch_required_inputs.py").resolve()


def _run_validator(
    trainer_rel: str,
    repo_root: Path,
    env: dict | None = None,
) -> tuple[int, str, str]:
    """Run the validator tool as a subprocess; return (rc, stdout, stderr)."""
    cmd = [
        sys.executable,
        str(_VALIDATOR_TOOL),
        "--trainer",
        trainer_rel,
        "--repo-root",
        str(repo_root),
        "--quiet",
    ]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    # Clear any pre-set env-vars from the parent test session that would
    # leak into the subprocess and accidentally satisfy validation.
    for key in list(merged_env):
        if key.startswith("MY_CONFIG") or key.startswith("UNRESOLVED_"):
            del merged_env[key]
    if env:
        merged_env.update(env)
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=merged_env
    )
    return result.returncode, result.stdout, result.stderr


def test_validator_rc0_when_default_file_exists(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    trainer_src = f'''
TIER_1_OPERATOR_REQUIRED_FLAGS = {{
    "--config-file": {{
        "env": "MY_CONFIG_T0",
        "rationale": "r",
        "default": "{config_path.name}",
        "required_input_file": True,
    }},
}}
'''
    _write_trainer(tmp_path, "train_x.py", trainer_src)
    rc, _, stderr = _run_validator("experiments/train_x.py", tmp_path)
    assert rc == 0, f"expected rc=0 when default file exists; stderr={stderr}"


def test_validator_rc1_when_file_missing(tmp_path: Path) -> None:
    trainer_src = '''
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--config-file": {
        "env": "MY_CONFIG_T1",
        "rationale": "needs a config",
        "default": "does/not/exist.json",
        "required_input_file": True,
        "generator_command": "python tools/build_config.py",
    },
}
'''
    _write_trainer(tmp_path, "train_x.py", trainer_src)
    rc, _, stderr = _run_validator("experiments/train_x.py", tmp_path)
    assert rc == 1, f"expected rc=1 when file missing; stderr={stderr}"
    assert "FATAL" in stderr
    assert "--config-file" in stderr
    # Generator command should be surfaced in the actionable error.
    assert "build_config.py" in stderr


def test_validator_env_var_overrides_default(tmp_path: Path) -> None:
    config_path = tmp_path / "override.json"
    config_path.write_text("{}")
    trainer_src = '''
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--config-file": {
        "env": "MY_CONFIG_T2",
        "rationale": "r",
        "default": "does/not/exist.json",
        "required_input_file": True,
    },
}
'''
    _write_trainer(tmp_path, "train_x.py", trainer_src)
    rc, _, stderr = _run_validator(
        "experiments/train_x.py", tmp_path, env={"MY_CONFIG_T2": str(config_path)},
    )
    assert rc == 0, f"env-var override should resolve to existing file; stderr={stderr}"


def test_validator_rc1_unresolvable_no_default_no_env(tmp_path: Path) -> None:
    trainer_src = '''
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--config-file": {
        "env": "UNRESOLVED_ENV_VAR",
        "rationale": "r",
        "default": None,
        "required_input_file": True,
    },
}
'''
    _write_trainer(tmp_path, "train_x.py", trainer_src)
    rc, _, stderr = _run_validator("experiments/train_x.py", tmp_path)
    assert rc == 1, f"expected rc=1 when unresolvable; stderr={stderr}"
    assert "unresolved" in stderr.lower() or "FATAL" in stderr


def test_validator_rc2_when_trainer_missing(tmp_path: Path) -> None:
    rc, _, stderr = _run_validator(
        "experiments/does_not_exist.py", tmp_path,
    )
    assert rc == 2, f"expected rc=2 when trainer missing; stderr={stderr}"
    assert "trainer file not found" in stderr.lower()


def test_validator_rc0_when_no_required_input_flags(tmp_path: Path) -> None:
    """Trainer declares manifest but no `required_input_file=True` entries."""
    trainer_src = _TRAINER_NO_REQ_INPUT
    _write_trainer(tmp_path, "train_x.py", trainer_src)
    rc, _, stderr = _run_validator("experiments/train_x.py", tmp_path)
    assert rc == 0, f"no required-input flags should pass; stderr={stderr}"


def test_validator_rc0_when_no_manifest(tmp_path: Path) -> None:
    """Trainer with no TIER_N manifest at all → fail-open (opt-in contract)."""
    _write_trainer(tmp_path, "train_bare.py", "def main(): pass\n")
    rc, _, stderr = _run_validator("experiments/train_bare.py", tmp_path)
    assert rc == 0, f"no manifest should pass; stderr={stderr}"
