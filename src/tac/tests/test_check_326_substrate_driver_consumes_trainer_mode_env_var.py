"""Tests for Catalog #326 — substrate driver consumes trainer mode env var.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable + Z6-v2 Wave 2 DEFER 2026-05-18 anchor.

Coverage:
- Helper unit tests (audit row classifier; rationale placeholder rejection;
  file-level + line-level waiver detection).
- End-to-end gate behavior (synthetic Z6-style bug class; recipe opt-out;
  waiver; aggregation; strict raises).
- Live-repo regression guard.
- Orchestrator-callsite warn-only wire-in regression guard.
- Catalog #185 sister regression (function callable via globals).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import tac.preflight as preflight_mod
from tac.preflight import (
    PreflightError,
    _check_326_driver_has_file_waiver,
    _check_326_driver_has_line_waiver,
    _check_326_rationale_is_placeholder,
    check_substrate_driver_consumes_trainer_mode_env_var,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_HELPER = REPO_ROOT / "tools" / "audit_substrate_driver_mode_hardcode.py"


# --- Helper unit tests --------------------------------------------------------


def test_rationale_is_placeholder_rejects_empty() -> None:
    assert _check_326_rationale_is_placeholder("") is True
    assert _check_326_rationale_is_placeholder("   ") is True


def test_rationale_is_placeholder_rejects_canonical_placeholders() -> None:
    for placeholder in ("<rationale>", "<reason>", "  <rationale>  ", "<REASON>"):
        assert _check_326_rationale_is_placeholder(placeholder) is True, placeholder


def test_rationale_is_placeholder_accepts_real_text() -> None:
    assert _check_326_rationale_is_placeholder("research-only smoke fixture") is False
    assert _check_326_rationale_is_placeholder("intentionally smoke-only per design memo") is False


def test_file_waiver_accepts_real_rationale(tmp_path: Path) -> None:
    driver = tmp_path / "remote_lane_substrate_fake.sh"
    driver.write_text(
        "#!/bin/bash\n"
        "# DRIVER_MODE_HARDCODE_OK_FILE: intentional smoke-only per research-only recipe\n"
        "set -euo pipefail\n"
        "--smoke\n"
    )
    assert _check_326_driver_has_file_waiver(driver) is True


def test_file_waiver_rejects_placeholder(tmp_path: Path) -> None:
    driver = tmp_path / "remote_lane_substrate_fake.sh"
    driver.write_text(
        "#!/bin/bash\n"
        "# DRIVER_MODE_HARDCODE_OK_FILE: <rationale>\n"
        "set -euo pipefail\n"
        "--smoke\n"
    )
    assert _check_326_driver_has_file_waiver(driver) is False


def test_file_waiver_must_be_in_first_30_lines(tmp_path: Path) -> None:
    body_lines = ["#!/bin/bash", "set -euo pipefail"] + ["echo line"] * 35
    body_lines.append("# DRIVER_MODE_HARDCODE_OK_FILE: late waiver should NOT count")
    body_lines.append("--smoke")
    driver = tmp_path / "remote_lane_substrate_fake.sh"
    driver.write_text("\n".join(body_lines))
    assert _check_326_driver_has_file_waiver(driver) is False


def test_line_waiver_accepts_real_rationale(tmp_path: Path) -> None:
    driver = tmp_path / "remote_lane_substrate_fake.sh"
    driver.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "trainer.py \\\n"
        "    --smoke # DRIVER_MODE_HARDCODE_OK: research-only L1 scaffold per design memo\n"
    )
    assert _check_326_driver_has_line_waiver(driver) is True


def test_line_waiver_rejects_placeholder(tmp_path: Path) -> None:
    driver = tmp_path / "remote_lane_substrate_fake.sh"
    driver.write_text(
        "#!/bin/bash\n"
        "    --smoke # DRIVER_MODE_HARDCODE_OK: <rationale>\n"
    )
    assert _check_326_driver_has_line_waiver(driver) is False


def test_line_waiver_must_be_same_line(tmp_path: Path) -> None:
    driver = tmp_path / "remote_lane_substrate_fake.sh"
    driver.write_text(
        "#!/bin/bash\n"
        "# DRIVER_MODE_HARDCODE_OK: above-line does not count\n"
        "    --smoke\n"
    )
    assert _check_326_driver_has_line_waiver(driver) is False


def test_line_waiver_only_fires_on_smoke_line(tmp_path: Path) -> None:
    driver = tmp_path / "remote_lane_substrate_fake.sh"
    driver.write_text(
        "#!/bin/bash\n"
        "echo unrelated # DRIVER_MODE_HARDCODE_OK: real rationale but NOT on smoke line\n"
        "    --smoke\n"
    )
    assert _check_326_driver_has_line_waiver(driver) is False


# --- End-to-end gate tests ----------------------------------------------------


def _isolated_repo(tmp_path: Path) -> Path:
    """Create a fresh repo skeleton with scripts/ + tools/ + .omx/operator_authorize_recipes/.

    Copies the canonical audit helper so the gate's importlib-based loader
    can resolve it from the isolated root.
    """
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    # Copy canonical helper for delegation
    shutil.copy(AUDIT_HELPER, tmp_path / "tools" / "audit_substrate_driver_mode_hardcode.py")
    return tmp_path


def test_gate_passes_when_no_drivers(tmp_path: Path) -> None:
    repo = _isolated_repo(tmp_path)
    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert result == []


def test_gate_passes_when_helper_missing(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    # No tools/ dir at all
    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert result == []


def test_gate_flags_z6_style_bug_class(tmp_path: Path) -> None:
    """Driver consumes SMOKE_ONLY (default 1) but matching recipe does not override."""
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_synthbug.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        SMOKE_ONLY="${SMOKE_ONLY:-1}"
        SMOKE_FLAG_ARGS=()
        if [ "$SMOKE_ONLY" = "1" ]; then
            SMOKE_FLAG_ARGS+=(--smoke)
        fi
        python trainer.py ${SMOKE_FLAG_ARGS[@]+"${SMOKE_FLAG_ARGS[@]}"}
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_synthbug_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_synthbug_modal_t4_dispatch
        research_only: false
        dispatch_enabled: true
        env_overrides:
          SOMETHING_ELSE: "value"
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(result) == 1
    assert "synthbug" in result[0]
    assert "CONSUMES_ENV_DEFAULTS_SMOKE_BUG_CLASS" in result[0]


def test_gate_accepts_z6_style_when_recipe_overrides_env(tmp_path: Path) -> None:
    """Driver consumes SMOKE_ONLY (default 1); matching recipe sets SMOKE_ONLY=0."""
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_synthok.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        SMOKE_ONLY="${SMOKE_ONLY:-1}"
        SMOKE_FLAG_ARGS=()
        if [ "$SMOKE_ONLY" = "1" ]; then
            SMOKE_FLAG_ARGS+=(--smoke)
        fi
        python trainer.py ${SMOKE_FLAG_ARGS[@]+"${SMOKE_FLAG_ARGS[@]}"}
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_synthok_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_synthok_modal_t4_dispatch
        research_only: false
        dispatch_enabled: true
        env_overrides:
          SMOKE_ONLY: "0"
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert result == []


def test_gate_accepts_z6_style_when_recipe_opted_out(tmp_path: Path) -> None:
    """Driver consumes SMOKE_ONLY (default 1); matching recipe declares research_only: true."""
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_synthopt.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        SMOKE_ONLY="${SMOKE_ONLY:-1}"
        SMOKE_FLAG_ARGS=()
        if [ "$SMOKE_ONLY" = "1" ]; then
            SMOKE_FLAG_ARGS+=(--smoke)
        fi
        python trainer.py ${SMOKE_FLAG_ARGS[@]+"${SMOKE_FLAG_ARGS[@]}"}
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_synthopt_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_synthopt_modal_t4_dispatch
        research_only: true
        dispatch_enabled: false
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert result == []


def test_gate_flags_hardcoded_smoke_without_recipe_opt_out(tmp_path: Path) -> None:
    """Driver passes --smoke unconditionally; recipe is dispatchable (full intent)."""
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_hardcoded.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        python trainer.py \\
            --output-dir /tmp/out \\
            --smoke
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_hardcoded_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_hardcoded_modal_t4_dispatch
        research_only: false
        dispatch_enabled: true
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(result) == 1
    assert "hardcoded" in result[0]
    assert "HARDCODES_SMOKE_NO_RECIPE_OPT_OUT" in result[0]


def test_gate_accepts_hardcoded_smoke_with_recipe_opt_out(tmp_path: Path) -> None:
    """Driver passes --smoke unconditionally; recipe is opted-out (research_only)."""
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_hardopt.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        python trainer.py \\
            --output-dir /tmp/out \\
            --smoke
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_hardopt_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_hardopt_modal_t4_dispatch
        smoke_only: true
        research_only: true
        dispatch_enabled: false
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert result == []


def test_gate_flags_hardcoded_smoke_when_only_one_matching_recipe_opted_out(
    tmp_path: Path,
) -> None:
    """One opted-out sibling must not hide a dispatchable sibling recipe."""
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_hardmixed.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        python trainer.py --smoke
    """))
    opted = repo / ".omx" / "operator_authorize_recipes" / "substrate_hardmixed_smoke_only.yaml"
    opted.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_hardmixed_smoke_only
        research_only: true
        dispatch_enabled: false
        remote_driver: scripts/remote_lane_substrate_hardmixed.sh
    """))
    dispatchable = (
        repo
        / ".omx"
        / "operator_authorize_recipes"
        / "substrate_hardmixed_modal_t4_dispatch.yaml"
    )
    dispatchable.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_hardmixed_modal_t4_dispatch
        research_only: false
        dispatch_enabled: true
        remote_driver: scripts/remote_lane_substrate_hardmixed.sh
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(result) == 1
    assert "hardmixed" in result[0]
    assert "HARDCODES_SMOKE_NO_RECIPE_OPT_OUT" in result[0]
    assert "substrate_hardmixed_modal_t4_dispatch.yaml" in result[0]


def test_gate_accepts_hardcoded_smoke_with_line_waiver(tmp_path: Path) -> None:
    """Driver passes --smoke unconditionally; carries same-line waiver."""
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_hardwaiver.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        python trainer.py \\
            --output-dir /tmp/out \\
            --smoke # DRIVER_MODE_HARDCODE_OK: intentional smoke-only per research-only spec
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_hardwaiver_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_hardwaiver_modal_t4_dispatch
        research_only: false
        dispatch_enabled: true
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert result == []


def test_gate_accepts_hardcoded_smoke_with_file_waiver(tmp_path: Path) -> None:
    """Driver passes --smoke unconditionally; carries file-level waiver."""
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_hardfile.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        # DRIVER_MODE_HARDCODE_OK_FILE: intentional smoke-only L1 scaffold per design memo §3
        set -euo pipefail
        python trainer.py --smoke
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_hardfile_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_hardfile_modal_t4_dispatch
        research_only: false
        dispatch_enabled: true
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert result == []


def test_gate_rejects_smoke_default_when_env_override_is_only_in_notes(
    tmp_path: Path,
) -> None:
    """Only env_overrides may satisfy a smoke/full mode override."""
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_noteonly.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        SMOKE_ONLY="${SMOKE_ONLY:-1}"
        SMOKE_FLAG_ARGS=()
        if [ "$SMOKE_ONLY" = "1" ]; then
            SMOKE_FLAG_ARGS+=(--smoke)
        fi
        python trainer.py ${SMOKE_FLAG_ARGS[@]+"${SMOKE_FLAG_ARGS[@]}"}
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_noteonly_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_noteonly_modal_t4_dispatch
        research_only: false
        dispatch_enabled: true
        notes: |
          Example operator command:
            SMOKE_ONLY: "0"
        remote_driver: scripts/remote_lane_substrate_noteonly.sh
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(result) == 1
    assert "noteonly" in result[0]
    assert "CONSUMES_ENV_DEFAULTS_SMOKE_BUG_CLASS" in result[0]


def test_gate_accepts_smoke_default_when_env_override_is_in_env_overrides(
    tmp_path: Path,
) -> None:
    """A real env_overrides entry still clears the smoke-default bug class."""
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_envok.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        SMOKE_ONLY="${SMOKE_ONLY:-1}"
        SMOKE_FLAG_ARGS=()
        if [ "$SMOKE_ONLY" = "1" ]; then
            SMOKE_FLAG_ARGS+=(--smoke)
        fi
        python trainer.py ${SMOKE_FLAG_ARGS[@]+"${SMOKE_FLAG_ARGS[@]}"}
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_envok_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_envok_modal_t4_dispatch
        research_only: false
        dispatch_enabled: true
        env_overrides:
          SMOKE_ONLY: "0"
        notes: |
          This recipe is full mode.
        remote_driver: scripts/remote_lane_substrate_envok.sh
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert result == []


def test_gate_flags_multi_key_smoke_default_when_recipe_declares_no_mode(
    tmp_path: Path,
) -> None:
    """Z6-style TRAINER_MODE > SMOKE_ONLY drivers still need a recipe override."""
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_z6ish.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        Z6_TRAINER_MODE="${Z6_TRAINER_MODE:-}"
        SMOKE_ONLY="${SMOKE_ONLY:-}"
        if [ -n "$Z6_TRAINER_MODE" ]; then
            case "$Z6_TRAINER_MODE" in
                smoke) SMOKE_ONLY="1" ;;
                full) SMOKE_ONLY="0" ;;
            esac
        elif [ -z "$SMOKE_ONLY" ]; then
            SMOKE_ONLY="1"
        fi
        SMOKE_FLAG_ARGS=()
        if [ "$SMOKE_ONLY" = "1" ]; then
            SMOKE_FLAG_ARGS+=(--smoke)
        fi
        python trainer.py ${SMOKE_FLAG_ARGS[@]+"${SMOKE_FLAG_ARGS[@]}"}
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_z6ish_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_z6ish_modal_t4_dispatch
        research_only: false
        dispatch_enabled: true
        remote_driver: scripts/remote_lane_substrate_z6ish.sh
        env_overrides:
          OTHER_ENV: "value"
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )

    assert len(result) == 1
    assert "z6ish" in result[0]
    assert "CONSUMES_ENV_DEFAULTS_SMOKE_BUG_CLASS" in result[0]


def test_gate_accepts_multi_key_driver_when_recipe_sets_trainer_mode_full(
    tmp_path: Path,
) -> None:
    """A canonical TRAINER_MODE=full declaration is enough; SMOKE_ONLY is legacy."""
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_z6modeok.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        Z6_TRAINER_MODE="${Z6_TRAINER_MODE:-}"
        SMOKE_ONLY="${SMOKE_ONLY:-}"
        if [ -n "$Z6_TRAINER_MODE" ]; then
            case "$Z6_TRAINER_MODE" in
                smoke) SMOKE_ONLY="1" ;;
                full) SMOKE_ONLY="0" ;;
            esac
        elif [ -z "$SMOKE_ONLY" ]; then
            SMOKE_ONLY="1"
        fi
        SMOKE_FLAG_ARGS=()
        if [ "$SMOKE_ONLY" = "1" ]; then
            SMOKE_FLAG_ARGS+=(--smoke)
        fi
        python trainer.py ${SMOKE_FLAG_ARGS[@]+"${SMOKE_FLAG_ARGS[@]}"}
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_z6modeok_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_z6modeok_modal_t4_dispatch
        research_only: false
        dispatch_enabled: true
        remote_driver: scripts/remote_lane_substrate_z6modeok.sh
        env_overrides:
          Z6_TRAINER_MODE: "full"
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )

    assert result == []


def test_gate_rejects_placeholder_line_waiver(tmp_path: Path) -> None:
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_phold.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        python trainer.py --smoke # DRIVER_MODE_HARDCODE_OK: <rationale>
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_phold_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_phold_modal_t4_dispatch
        research_only: false
        dispatch_enabled: true
    """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(result) == 1


def test_gate_strict_raises_on_violation(tmp_path: Path) -> None:
    repo = _isolated_repo(tmp_path)
    driver = repo / "scripts" / "remote_lane_substrate_strict.sh"
    driver.write_text(textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        python trainer.py --smoke
    """))
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_strict_modal_t4_dispatch.yaml"
    recipe.write_text(textwrap.dedent("""\
        schema_version: 1
        name: substrate_strict_modal_t4_dispatch
        dispatch_enabled: true
    """))

    with pytest.raises(PreflightError, match=r"Catalog #326"):
        check_substrate_driver_consumes_trainer_mode_env_var(
            repo_root=repo, strict=True, verbose=False
        )


def test_gate_strict_silent_on_clean_repo(tmp_path: Path) -> None:
    repo = _isolated_repo(tmp_path)
    # Empty drivers dir → no violations
    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=True, verbose=False
    )
    assert result == []


def test_gate_aggregates_multiple_violations(tmp_path: Path) -> None:
    repo = _isolated_repo(tmp_path)
    for name in ("multi_a", "multi_b", "multi_c"):
        driver = repo / "scripts" / f"remote_lane_substrate_{name}.sh"
        driver.write_text(textwrap.dedent("""\
            #!/bin/bash
            set -euo pipefail
            python trainer.py --smoke
        """))
        recipe = repo / ".omx" / "operator_authorize_recipes" / f"substrate_{name}_modal_t4_dispatch.yaml"
        recipe.write_text(textwrap.dedent(f"""\
            schema_version: 1
            name: substrate_{name}_modal_t4_dispatch
            dispatch_enabled: true
        """))

    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(result) == 3


def test_gate_passes_when_helper_raises_import_error_warn_only(tmp_path: Path) -> None:
    """If canonical helper raises during import, warn-only mode returns []."""
    repo = _isolated_repo(tmp_path)
    # Write a malformed helper
    (repo / "tools" / "audit_substrate_driver_mode_hardcode.py").write_text(
        "raise ImportError('malformed test fixture')\n"
    )
    # Add a driver so the audit would have something to scan
    driver = repo / "scripts" / "remote_lane_substrate_x.sh"
    driver.write_text("#!/bin/bash\n--smoke\n")
    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=repo, strict=False, verbose=False
    )
    # Warn-only swallows the import error
    assert result == []


# --- Live-repo regression guard -----------------------------------------------


def test_live_repo_regression_guard() -> None:
    """The live repo MUST have 0 violations at landing per CLAUDE.md
    'Strict-flip atomicity rule'."""
    result = check_substrate_driver_consumes_trainer_mode_env_var(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    # Bound at 0 at landing; strict-flip planned after one full session
    # without regression. Allow up to 2 to provide a small buffer for
    # in-flight sister-subagent driver edits that may temporarily flag.
    assert len(result) <= 2, (
        f"Catalog #326 live count expected 0 (or ≤2 buffer) at landing; got "
        f"{len(result)}: {result[:3]}"
    )


# --- Orchestrator wire-in regression guard ------------------------------------


def test_orchestrator_callsite_wired_warn_only() -> None:
    """Catalog #326 MUST be wired into preflight_all() with strict=False at
    landing per CLAUDE.md "Strict-flip atomicity rule"."""
    import inspect
    src = inspect.getsource(preflight_mod.preflight_all)
    assert "check_substrate_driver_consumes_trainer_mode_env_var" in src, (
        "Catalog #326 wire-in missing from preflight_all"
    )
    # Find the callsite and assert it carries strict=False
    callsite_idx = src.index("check_substrate_driver_consumes_trainer_mode_env_var")
    # Look ahead within 200 chars for the strict= argument
    next_500 = src[callsite_idx : callsite_idx + 500]
    assert "strict=False" in next_500, (
        "Catalog #326 wire-in must carry strict=False at landing per Strict-flip atomicity rule"
    )


def test_catalog_185_sister_function_callable_via_globals() -> None:
    """Catalog #185 META-meta-meta sister regression: check_substrate_driver_consumes_trainer_mode_env_var
    MUST be callable via tac.preflight module globals so Catalog #185 can
    look it up and verify live-count=0 claims in CLAUDE.md."""
    fn = getattr(preflight_mod, "check_substrate_driver_consumes_trainer_mode_env_var", None)
    assert callable(fn), (
        "Catalog #326 gate function must be callable via tac.preflight module globals"
    )


# --- Catalog # claim sanity ---------------------------------------------------


def test_catalog_326_claimed_in_state_log() -> None:
    """Catalog #326 must appear in `.omx/state/catalog-claim.log` per Catalog #186
    (canonical-serializer transactional claim)."""
    claim_log = REPO_ROOT / ".omx" / "state" / "catalog-claim.log"
    if not claim_log.is_file():
        pytest.skip("catalog-claim.log not present in this checkout")
    text = claim_log.read_text(encoding="utf-8")
    assert "326" in text, "Catalog #326 must be in catalog-claim.log per Catalog #186"


# --- CLI smoke test on canonical helper ---------------------------------------


def test_canonical_helper_cli_smoke() -> None:
    """The canonical operator-facing audit tool MUST be runnable as a CLI."""
    if not AUDIT_HELPER.is_file():
        pytest.skip("Canonical helper missing")
    # Run in summary mode (default); should exit 0 since live count is 0
    result = subprocess.run(
        [sys.executable, str(AUDIT_HELPER), "--format", "summary"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, f"helper failed: {result.stderr[:500]}"
    assert "Catalog #326" in result.stdout
    assert "Bug class count" in result.stdout


def test_canonical_helper_cli_json_output() -> None:
    if not AUDIT_HELPER.is_file():
        pytest.skip("Canonical helper missing")
    result = subprocess.run(
        [sys.executable, str(AUDIT_HELPER), "--format", "json"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, f"helper failed: {result.stderr[:500]}"
    # Should parse as JSON
    data = json.loads(result.stdout)
    assert data.get("schema_version") == 1
    assert "verdict_counts" in data
    assert "rows" in data
    assert data.get("empirical_anchor_z6_v2_wave_2_defer", {}).get("date_utc") == "2026-05-18"


# --- OVERNIGHT-RR META-class extension regression tests -----------------------
# Per OVERNIGHT-RR 2026-05-21 NSCS06 v8 chroma-LUT QQ dispatch rc=22 root-cause
# diagnosis. The driver READS NSCS06_V8_TRAINER_MODE env var (so passes the
# existing CONSUMES_ENV_MULTI_KEY_DEFAULT_RECIPE_OK verdict path) BUT uses it
# to REFUSE non-smoke at startup (FATAL + exit 22) rather than to BRANCH on it.
# Verdict: REFUSES_NON_SMOKE_RECIPE_FORCES_FULL_BUG_CLASS.


def test_check_326_overnight_rr_meta_extension_flags_refuses_non_smoke_pattern(
    tmp_path: Path,
) -> None:
    """Synthetic pre-fix v8 driver pattern MUST be flagged with the new
    OVERNIGHT-RR META-extension verdict.
    """
    from tools.audit_substrate_driver_mode_hardcode import audit_all_drivers

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)

    (scripts_dir / "remote_lane_substrate_synth_overnight_rr.sh").write_text(
        textwrap.dedent(
            """\
            #!/bin/bash
            set -euo pipefail
            SYNTH_OVERNIGHT_RR_TRAINER_MODE="${SYNTH_OVERNIGHT_RR_TRAINER_MODE:-smoke}"
            if [ "$SYNTH_OVERNIGHT_RR_TRAINER_MODE" != "smoke" ]; then
                echo "FATAL: only smoke supported"
                exit 22
            fi
            """
        )
    )
    (recipes_dir / "substrate_synth_overnight_rr_modal_t4_dispatch.yaml").write_text(
        textwrap.dedent(
            """\
            name: substrate_synth_overnight_rr_modal_t4_dispatch
            dispatch_enabled: true
            research_only: false
            env_overrides:
              SYNTH_OVERNIGHT_RR_TRAINER_MODE: "full"
            """
        )
    )

    result = audit_all_drivers(repo_root=tmp_path)
    assert result["bug_class_count"] == 1
    assert result["bug_class_drivers"] == ["synth_overnight_rr"]
    row = result["rows"][0]
    assert row["verdict"] == "REFUSES_NON_SMOKE_RECIPE_FORCES_FULL_BUG_CLASS"
    assert "FATAL+exit pattern at startup" in row["explanation"]
    assert "OVERNIGHT-RR" in row["explanation"]
    assert row["refuse_per_var"] == {"SYNTH_OVERNIGHT_RR_TRAINER_MODE": True}
    assert row["unsafe_recipes"] == [
        ".omx/operator_authorize_recipes/substrate_synth_overnight_rr_modal_t4_dispatch.yaml"
    ]


def test_check_326_overnight_rr_meta_extension_exempts_multi_value_validator(
    tmp_path: Path,
) -> None:
    """The canonical post-fix pattern that validates {smoke, full} with a
    compound test (``[ != "smoke" ] && [ != "full" ]``) MUST NOT be flagged.
    """
    from tools.audit_substrate_driver_mode_hardcode import audit_all_drivers

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)

    (scripts_dir / "remote_lane_substrate_synth_canonical.sh").write_text(
        textwrap.dedent(
            """\
            #!/bin/bash
            set -euo pipefail
            SYNTH_CANONICAL_TRAINER_MODE="${SYNTH_CANONICAL_TRAINER_MODE:-smoke}"
            if [ "$SYNTH_CANONICAL_TRAINER_MODE" != "smoke" ] && [ "$SYNTH_CANONICAL_TRAINER_MODE" != "full" ]; then
                echo "FATAL: only smoke or full accepted"
                exit 22
            fi
            SMOKE_FLAG=""
            if [ "$SYNTH_CANONICAL_TRAINER_MODE" = "smoke" ]; then
                SMOKE_FLAG="--smoke"
            fi
            """
        )
    )
    (recipes_dir / "substrate_synth_canonical_modal_t4_dispatch.yaml").write_text(
        textwrap.dedent(
            """\
            name: substrate_synth_canonical_modal_t4_dispatch
            dispatch_enabled: true
            research_only: false
            env_overrides:
              SYNTH_CANONICAL_TRAINER_MODE: "full"
            """
        )
    )

    result = audit_all_drivers(repo_root=tmp_path)
    assert result["bug_class_count"] == 0, (
        f"canonical multi-value-validator pattern incorrectly flagged: "
        f"{result['rows']}"
    )


def test_check_326_flags_full_mode_driver_default_cpu_device(
    tmp_path: Path,
) -> None:
    """Full trainer mode plus driver-default ``--device cpu`` is a pre-dispatch bug."""
    from tools.audit_substrate_driver_mode_hardcode import audit_all_drivers

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)

    (scripts_dir / "remote_lane_substrate_synth_device_cpu.sh").write_text(
        textwrap.dedent(
            """\
            #!/bin/bash
            set -euo pipefail
            SYNTH_DEVICE_TRAINER_MODE="${SYNTH_DEVICE_TRAINER_MODE:-smoke}"
            SYNTH_DEVICE_DEVICE="${SYNTH_DEVICE_DEVICE:-cpu}"
            SMOKE_FLAG=""
            if [ "$SYNTH_DEVICE_TRAINER_MODE" = "smoke" ]; then
                SMOKE_FLAG="--smoke"
            fi
            python experiments/train_substrate_synth.py --device "$SYNTH_DEVICE_DEVICE" $SMOKE_FLAG
            """
        )
    )
    (recipes_dir / "substrate_synth_device_cpu_modal_t4_dispatch.yaml").write_text(
        textwrap.dedent(
            """\
            name: substrate_synth_device_cpu_modal_t4_dispatch
            dispatch_enabled: true
            research_only: false
            env_overrides:
              SYNTH_DEVICE_TRAINER_MODE: "full"
            """
        )
    )

    result = audit_all_drivers(repo_root=tmp_path)
    assert result["bug_class_count"] == 1
    row = result["rows"][0]
    assert row["verdict"] == "FULL_MODE_DEVICE_CPU_BUG_CLASS"
    assert row["unsafe_recipes"][0]["device_env_var"] == "SYNTH_DEVICE_DEVICE"
    assert row["unsafe_recipes"][0]["effective_device"] == "cpu"


def test_check_326_flags_full_mode_recipe_cpu_device_override(
    tmp_path: Path,
) -> None:
    """A recipe-side CPU override is also unsafe even if driver defaults to CUDA."""
    from tools.audit_substrate_driver_mode_hardcode import audit_all_drivers

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)

    (scripts_dir / "remote_lane_substrate_synth_recipe_cpu.sh").write_text(
        textwrap.dedent(
            """\
            #!/bin/bash
            set -euo pipefail
            SYNTH_RECIPE_TRAINER_MODE="${SYNTH_RECIPE_TRAINER_MODE:-smoke}"
            SYNTH_RECIPE_DEVICE="${SYNTH_RECIPE_DEVICE:-cuda}"
            python experiments/train_substrate_synth.py --device "$SYNTH_RECIPE_DEVICE"
            """
        )
    )
    (recipes_dir / "substrate_synth_recipe_cpu_modal_t4_dispatch.yaml").write_text(
        textwrap.dedent(
            """\
            name: substrate_synth_recipe_cpu_modal_t4_dispatch
            dispatch_enabled: true
            research_only: false
            env_overrides:
              SYNTH_RECIPE_TRAINER_MODE: "full"
              SYNTH_RECIPE_DEVICE: "cpu"
            """
        )
    )

    result = audit_all_drivers(repo_root=tmp_path)
    assert result["bug_class_count"] == 1
    assert result["rows"][0]["verdict"] == "FULL_MODE_DEVICE_CPU_BUG_CLASS"
