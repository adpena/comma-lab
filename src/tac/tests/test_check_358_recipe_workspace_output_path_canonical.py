"""Tests for Catalog #358 — check_recipe_workspace_output_path_canonical_or_modal_aware.

WAVE-3-HARDEN-1-MASTER-GRADIENT-TMP-PATH-BUG-CLASS-EXTINCTION self-protection
landed 2026-05-20 per operator NON-NEGOTIABLE "must fix and harden all".

Sister of Catalog #204 (per-driver Modal-aware OUTPUT path fix) at the
recipe-level surface. Refuses tool-dispatch recipes whose env_overrides
emit /workspace/pact/<OUTPUT> paths without a canonical Catalog #204
3-branch Modal-aware driver override.

Bug class anchor: WAVE-3-OP3 dispatch fc-01KS2Z2WJQW532A9226JAVQM8Y
(2026-05-20T15:11:22Z) failed rc=1 at 9.74s because the recipe's
MASTER_GRADIENT_OUTPUT_NPY=/workspace/pact/.omx/state/... resolved to
/tmp/pact/.omx/state/... on the Modal worker and triggered the extractor's
/tmp refusal per CLAUDE.md "Forbidden /tmp paths" (Catalog #220).
"""
# SPDX-License-Identifier: MIT

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_358_driver_implements_canonical_override,
    _check_358_extract_lane_script,
    _check_358_extract_output_env_vars,
    _check_358_line_has_waiver,
    _check_358_recipe_is_in_scope,
    _check_358_recipe_is_opted_out,
    check_recipe_workspace_output_path_canonical_or_modal_aware,
    preflight_all,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


CANONICAL_DRIVER_WITH_OVERRIDE = textwrap.dedent("""\
    #!/bin/bash
    set -euo pipefail

    WORKSPACE="${WORKSPACE:-/workspace/pact}"
    MASTER_GRADIENT_OUTPUT_NPY="${MASTER_GRADIENT_OUTPUT_NPY:-$WORKSPACE/.omx/state/foo.npy}"
    DISPATCH_INSTANCE_JOB_ID="${DISPATCH_INSTANCE_JOB_ID:-}"

    # Canonical Catalog #204 override pattern.
    if [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ] && [ -n "${DISPATCH_INSTANCE_JOB_ID:-}" ]; then
        MASTER_GRADIENT_OUTPUT_NPY="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output/foo.npy"
    fi

    echo "$MASTER_GRADIENT_OUTPUT_NPY"
""")


BROKEN_DRIVER_NO_OVERRIDE = textwrap.dedent("""\
    #!/bin/bash
    set -euo pipefail

    WORKSPACE="${WORKSPACE:-/workspace/pact}"
    MASTER_GRADIENT_OUTPUT_NPY="${MASTER_GRADIENT_OUTPUT_NPY:-$WORKSPACE/.omx/state/foo.npy}"

    echo "$MASTER_GRADIENT_OUTPUT_NPY"
""")


CANONICAL_TOOL_RECIPE_WITH_OVERRIDE_DRIVER = textwrap.dedent("""\
    schema_version: 1
    name: synthetic_tool_recipe_with_override
    lane_id: lane_synthetic_test
    dispatch_kind: tool
    platform: modal
    gpu: "CPU"

    modal:
      lane_script: scripts/synthetic_canonical_driver.sh

    env_overrides:
      MASTER_GRADIENT_OUTPUT_NPY: /workspace/pact/.omx/state/foo.npy
""")


BUG_CLASS_RECIPE_NO_DRIVER_OVERRIDE = textwrap.dedent("""\
    schema_version: 1
    name: synthetic_op3_bug_class
    lane_id: lane_synthetic_test
    dispatch_kind: tool
    platform: modal
    gpu: "T4"

    modal:
      lane_script: scripts/synthetic_broken_driver.sh

    env_overrides:
      MASTER_GRADIENT_OUTPUT_NPY: /workspace/pact/.omx/state/foo.npy
""")


RESEARCH_ONLY_RECIPE = textwrap.dedent("""\
    schema_version: 1
    name: synthetic_research_only
    lane_id: lane_synthetic_research
    dispatch_kind: tool
    research_only: true
    platform: modal
    gpu: "CPU"

    env_overrides:
      MASTER_GRADIENT_OUTPUT_NPY: /workspace/pact/.omx/state/foo.npy
""")


DISPATCH_DISABLED_RECIPE = textwrap.dedent("""\
    schema_version: 1
    name: synthetic_dispatch_disabled
    lane_id: lane_synthetic_disabled
    dispatch_kind: tool
    dispatch_enabled: false
    platform: modal
    gpu: "CPU"

    env_overrides:
      MASTER_GRADIENT_OUTPUT_NPY: /workspace/pact/.omx/state/foo.npy
""")


WAIVER_RECIPE = textwrap.dedent("""\
    schema_version: 1
    name: synthetic_waiver
    lane_id: lane_synthetic_waiver
    dispatch_kind: tool
    platform: modal
    gpu: "CPU"

    modal:
      lane_script: scripts/synthetic_missing_driver.sh

    env_overrides:
      MASTER_GRADIENT_OUTPUT_NPY: /workspace/pact/.omx/state/foo.npy  # WORKSPACE_OUTPUT_PATH_OK:operator_reviewed_driver_scaffold_pending
""")


SUBSTRATE_RECIPE_OUT_OF_SCOPE = textwrap.dedent("""\
    schema_version: 1
    name: synthetic_substrate_recipe
    lane_id: lane_synthetic_substrate
    # No dispatch_kind:tool declared → not in Catalog #358 scope.
    platform: modal
    gpu: "T4"

    modal:
      lane_script: scripts/synthetic_substrate_driver.sh

    env_overrides:
      SUBSTRATE_OUTPUT_DIR: /workspace/pact/experiments/results/synthetic
""")


PLACEHOLDER_WAIVER_RECIPE = textwrap.dedent("""\
    schema_version: 1
    name: synthetic_placeholder
    lane_id: lane_synthetic_placeholder
    dispatch_kind: tool
    platform: modal
    gpu: "CPU"

    modal:
      lane_script: scripts/synthetic_missing.sh

    env_overrides:
      MASTER_GRADIENT_OUTPUT_NPY: /workspace/pact/.omx/state/foo.npy  # WORKSPACE_OUTPUT_PATH_OK:<rationale>
""")


SHORT_WAIVER_RECIPE = textwrap.dedent("""\
    schema_version: 1
    name: synthetic_short_waiver
    lane_id: lane_synthetic_short_waiver
    dispatch_kind: tool
    platform: modal
    gpu: "CPU"

    modal:
      lane_script: scripts/synthetic_missing.sh

    env_overrides:
      MASTER_GRADIENT_OUTPUT_NPY: /workspace/pact/.omx/state/foo.npy  # WORKSPACE_OUTPUT_PATH_OK:ab
""")


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_recipe_in_scope_tool() -> None:
    assert _check_358_recipe_is_in_scope("dispatch_kind: tool\n") is True


def test_recipe_in_scope_substrate() -> None:
    assert _check_358_recipe_is_in_scope("platform: modal\n") is False


def test_recipe_in_scope_explicit_substrate() -> None:
    assert _check_358_recipe_is_in_scope("dispatch_kind: substrate\n") is False


def test_recipe_in_scope_quoted_tool() -> None:
    assert _check_358_recipe_is_in_scope('dispatch_kind: "tool"\n') is True


def test_recipe_opted_out_research_only() -> None:
    assert _check_358_recipe_is_opted_out("research_only: true\n") is True


def test_recipe_opted_out_dispatch_disabled() -> None:
    assert _check_358_recipe_is_opted_out("dispatch_enabled: false\n") is True


def test_recipe_not_opted_out_default() -> None:
    assert _check_358_recipe_is_opted_out("platform: modal\n") is False


def test_recipe_not_opted_out_research_only_false() -> None:
    assert _check_358_recipe_is_opted_out("research_only: false\n") is False


def test_recipe_not_opted_out_indented_research_only() -> None:
    # Only top-level fields count.
    assert _check_358_recipe_is_opted_out("  research_only: true\n") is False


def test_extract_lane_script_modal_block() -> None:
    text = "modal:\n  lane_script: scripts/foo.sh\n"
    assert _check_358_extract_lane_script(text) == "scripts/foo.sh"


def test_extract_lane_script_remote_driver_fallback() -> None:
    text = "remote_driver: scripts/foo.sh\n"
    assert _check_358_extract_lane_script(text) == "scripts/foo.sh"


def test_extract_lane_script_missing() -> None:
    text = "schema_version: 1\nplatform: modal\n"
    assert _check_358_extract_lane_script(text) is None


def test_extract_output_env_vars_matches_npy() -> None:
    text = textwrap.dedent("""\
        env_overrides:
          MASTER_GRADIENT_OUTPUT_NPY: /workspace/pact/.omx/state/foo.npy
    """)
    results = _check_358_extract_output_env_vars(text)
    assert len(results) == 1
    assert results[0][0] == "MASTER_GRADIENT_OUTPUT_NPY"
    assert results[0][1] == "/workspace/pact/.omx/state/foo.npy"


def test_extract_output_env_vars_skips_non_workspace_paths() -> None:
    text = textwrap.dedent("""\
        env_overrides:
          MASTER_GRADIENT_OUTPUT_NPY: /modal_results/foo.npy
    """)
    results = _check_358_extract_output_env_vars(text)
    assert results == []


def test_extract_output_env_vars_skips_non_output_suffixes() -> None:
    text = textwrap.dedent("""\
        env_overrides:
          MASTER_GRADIENT_INPUT_PATH: /workspace/pact/input.bin
    """)
    results = _check_358_extract_output_env_vars(text)
    assert results == []


def test_extract_output_env_vars_matches_output_dir() -> None:
    text = textwrap.dedent("""\
        env_overrides:
          FOO_OUTPUT_DIR: /workspace/pact/results
    """)
    results = _check_358_extract_output_env_vars(text)
    assert len(results) == 1
    assert results[0][0] == "FOO_OUTPUT_DIR"


def test_driver_implements_canonical_override_positive() -> None:
    assert _check_358_driver_implements_canonical_override(
        CANONICAL_DRIVER_WITH_OVERRIDE, "MASTER_GRADIENT_OUTPUT_NPY"
    ) is True


def test_driver_implements_canonical_override_negative() -> None:
    assert _check_358_driver_implements_canonical_override(
        BROKEN_DRIVER_NO_OVERRIDE, "MASTER_GRADIENT_OUTPUT_NPY"
    ) is False


def test_driver_implements_canonical_override_wrong_env_var() -> None:
    # Canonical pattern exists but for a different env var.
    assert _check_358_driver_implements_canonical_override(
        CANONICAL_DRIVER_WITH_OVERRIDE, "DIFFERENT_OUTPUT_NPY"
    ) is False


def test_line_has_waiver_rationale_accepted() -> None:
    line = "  FOO: bar  # WORKSPACE_OUTPUT_PATH_OK:driver_scaffold_pending_for_real_reason"
    assert _check_358_line_has_waiver(line) is True


def test_line_has_waiver_placeholder_rejected() -> None:
    line = "  FOO: bar  # WORKSPACE_OUTPUT_PATH_OK:<rationale>"
    assert _check_358_line_has_waiver(line) is False


def test_line_has_waiver_reason_placeholder_rejected() -> None:
    line = "  FOO: bar  # WORKSPACE_OUTPUT_PATH_OK:<reason>"
    assert _check_358_line_has_waiver(line) is False


def test_line_has_waiver_short_rationale_rejected() -> None:
    line = "  FOO: bar  # WORKSPACE_OUTPUT_PATH_OK:ab"
    assert _check_358_line_has_waiver(line) is False


def test_line_has_waiver_no_marker() -> None:
    line = "  FOO: bar"
    assert _check_358_line_has_waiver(line) is False


# ---------------------------------------------------------------------------
# End-to-end gate behavior
# ---------------------------------------------------------------------------


def test_no_recipes_dir_silent(tmp_path: Path) -> None:
    violations = check_recipe_workspace_output_path_canonical_or_modal_aware(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_canonical_recipe_passes(tmp_path: Path) -> None:
    """Recipe with /workspace/pact/<OUTPUT> + canonical driver override accepted."""
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (recipes_dir / "synthetic.yaml").write_text(
        CANONICAL_TOOL_RECIPE_WITH_OVERRIDE_DRIVER, encoding="utf-8"
    )
    (scripts_dir / "synthetic_canonical_driver.sh").write_text(
        CANONICAL_DRIVER_WITH_OVERRIDE, encoding="utf-8"
    )
    violations = check_recipe_workspace_output_path_canonical_or_modal_aware(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_bug_class_recipe_flagged(tmp_path: Path) -> None:
    """The OP3 bug-class anchor pattern is flagged."""
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (recipes_dir / "bug.yaml").write_text(
        BUG_CLASS_RECIPE_NO_DRIVER_OVERRIDE, encoding="utf-8"
    )
    (scripts_dir / "synthetic_broken_driver.sh").write_text(
        BROKEN_DRIVER_NO_OVERRIDE, encoding="utf-8"
    )
    violations = check_recipe_workspace_output_path_canonical_or_modal_aware(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "MASTER_GRADIENT_OUTPUT_NPY" in violations[0]
    assert "Catalog #204" in violations[0]


def test_research_only_recipe_passes(tmp_path: Path) -> None:
    """research_only: true opts out of the gate per CLAUDE.md."""
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "research.yaml").write_text(
        RESEARCH_ONLY_RECIPE, encoding="utf-8"
    )
    violations = check_recipe_workspace_output_path_canonical_or_modal_aware(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_dispatch_disabled_recipe_passes(tmp_path: Path) -> None:
    """dispatch_enabled: false opts out of the gate."""
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "disabled.yaml").write_text(
        DISPATCH_DISABLED_RECIPE, encoding="utf-8"
    )
    violations = check_recipe_workspace_output_path_canonical_or_modal_aware(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_waiver_recipe_passes(tmp_path: Path) -> None:
    """Same-line waiver with substantive rationale accepted."""
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "waiver.yaml").write_text(WAIVER_RECIPE, encoding="utf-8")
    violations = check_recipe_workspace_output_path_canonical_or_modal_aware(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_substrate_recipe_out_of_scope(tmp_path: Path) -> None:
    """Recipes without `dispatch_kind: tool` are out of Catalog #358 scope."""
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "substrate.yaml").write_text(
        SUBSTRATE_RECIPE_OUT_OF_SCOPE, encoding="utf-8"
    )
    violations = check_recipe_workspace_output_path_canonical_or_modal_aware(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_placeholder_waiver_rejected(tmp_path: Path) -> None:
    """Placeholder `<rationale>` literal cannot self-waive."""
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "placeholder.yaml").write_text(
        PLACEHOLDER_WAIVER_RECIPE, encoding="utf-8"
    )
    violations = check_recipe_workspace_output_path_canonical_or_modal_aware(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_short_rationale_rejected(tmp_path: Path) -> None:
    """Rationale <4 chars rejected per Catalog #287 sister discipline."""
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "short.yaml").write_text(
        SHORT_WAIVER_RECIPE, encoding="utf-8"
    )
    violations = check_recipe_workspace_output_path_canonical_or_modal_aware(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_strict_mode_raises(tmp_path: Path) -> None:
    """Strict mode raises PreflightError with Catalog #358 message."""
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (recipes_dir / "bug.yaml").write_text(
        BUG_CLASS_RECIPE_NO_DRIVER_OVERRIDE, encoding="utf-8"
    )
    (scripts_dir / "synthetic_broken_driver.sh").write_text(
        BROKEN_DRIVER_NO_OVERRIDE, encoding="utf-8"
    )
    with pytest.raises(PreflightError) as exc_info:
        check_recipe_workspace_output_path_canonical_or_modal_aware(
            repo_root=tmp_path, strict=True, verbose=False
        )
    assert "Catalog #358" in str(exc_info.value)
    assert "WAVE-3-HARDEN-1" in str(exc_info.value)


def test_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode is silent (no raise) when no violations."""
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "research.yaml").write_text(RESEARCH_ONLY_RECIPE, encoding="utf-8")
    # No exception.
    check_recipe_workspace_output_path_canonical_or_modal_aware(
        repo_root=tmp_path, strict=True, verbose=False
    )


def test_multi_violation_aggregation(tmp_path: Path) -> None:
    """Multiple violations across recipes are aggregated."""
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    for n in range(3):
        text = BUG_CLASS_RECIPE_NO_DRIVER_OVERRIDE.replace(
            "synthetic_op3_bug_class", f"synthetic_op3_bug_class_{n}"
        ).replace(
            "scripts/synthetic_broken_driver.sh",
            f"scripts/synthetic_broken_driver_{n}.sh",
        )
        (recipes_dir / f"bug_{n}.yaml").write_text(text, encoding="utf-8")
        (scripts_dir / f"synthetic_broken_driver_{n}.sh").write_text(
            BROKEN_DRIVER_NO_OVERRIDE, encoding="utf-8"
        )
    violations = check_recipe_workspace_output_path_canonical_or_modal_aware(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 3


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_live_repo_regression_guard() -> None:
    """Live repo must be at live-count = 0 per WAVE-3-HARDEN-1 landing.

    STRICT-from-byte-one per CLAUDE.md "Strict-flip atomicity rule" — the
    L2 sister CPU driver fix + mps_gap driver fix + 2 orphan recipe waivers
    land in the same commit batch as this gate, driving live count to 0.
    """
    violations = check_recipe_workspace_output_path_canonical_or_modal_aware(
        strict=False, verbose=False
    )
    assert len(violations) == 0, (
        f"Catalog #358 live count regressed to {len(violations)} "
        f"(expected 0 per WAVE-3-HARDEN-1 landing): {violations[:3]}"
    )


# ---------------------------------------------------------------------------
# Catalog #185 sister regression: gate callable via globals
# ---------------------------------------------------------------------------


def test_catalog_185_sister_gate_callable() -> None:
    """Per Catalog #185 META-meta-meta regression guard: every catalog
    function declared in CLAUDE.md MUST be importable + callable."""
    from tac import preflight as preflight_module

    fn = getattr(
        preflight_module,
        "check_recipe_workspace_output_path_canonical_or_modal_aware",
        None,
    )
    assert fn is not None
    assert callable(fn)


# ---------------------------------------------------------------------------
# Orchestrator wire-in regression guard
# ---------------------------------------------------------------------------


def test_orchestrator_strict_true_wire_in_regression() -> None:
    """preflight_all() must wire Catalog #358 with strict=True per
    STRICT-from-byte-one declaration in CLAUDE.md."""
    import inspect

    source = inspect.getsource(preflight_all)
    assert (
        "check_recipe_workspace_output_path_canonical_or_modal_aware"
        in source
    )
    # The wire-in must use strict=True per STRICT-from-byte-one.
    # Find the line; verify it's wired with strict=True.
    lines = source.splitlines()
    found_strict_true = False
    for i, line in enumerate(lines):
        if "check_recipe_workspace_output_path_canonical_or_modal_aware" in line:
            # Check this and next 3 lines for strict=True.
            window = "\n".join(lines[i : i + 4])
            if "strict=True" in window:
                found_strict_true = True
                break
    assert found_strict_true, (
        "Catalog #358 must be wired with strict=True per STRICT-from-byte-one "
        "(no warn-only intermediate state per Strict-flip atomicity rule)"
    )
