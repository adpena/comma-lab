# SPDX-License-Identifier: MIT
"""Slot F 2026-05-29 sister-tool migrations to canonical helper.

Validates that the 2 sister research-pipeline tools migrated in Slot F
(``tools/build_pr95_mlx_optimizer_matrix_queue.py`` +
``tools/build_monolithic_runtime_consumption_proof.py``) route through
``tac.research_pipeline_output_dir_safety.enforce_research_pipeline_output_dir``
per the Slot D canonical 2-landing pattern and that Catalog #381 STRICT
preflight gate live-count remains 0 across the extended 5-producer set.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable + canonical 2-landing pattern (canonical helper at
``tac.research_pipeline_output_dir_safety`` + STRICT preflight gate
Catalog #381). Sister of ``test_check_381_research_pipeline_canonical_output_dir_safety.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PR95_MLX_TOOL = REPO_ROOT / "tools" / "build_pr95_mlx_optimizer_matrix_queue.py"
MONOLITHIC_TOOL = REPO_ROOT / "tools" / "build_monolithic_runtime_consumption_proof.py"


# ---------------------------------------------------------------------------
# Source-level invariants — tool body references the canonical helper.
# ---------------------------------------------------------------------------


def _read_tool_body(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_pr95_mlx_tool_imports_canonical_helper() -> None:
    body = _read_tool_body(PR95_MLX_TOOL)
    assert "tac.research_pipeline_output_dir_safety" in body
    assert "enforce_research_pipeline_output_dir" in body
    assert "OutputDirSafetyError" in body


def test_pr95_mlx_tool_declares_overwrite_flags() -> None:
    body = _read_tool_body(PR95_MLX_TOOL)
    assert "--allow-overwrite-existing-historical-provenance" in body
    assert "--overwrite-rationale" in body


def test_pr95_mlx_tool_invokes_helper_against_output_root() -> None:
    body = _read_tool_body(PR95_MLX_TOOL)
    # helper invocation must reference args.output_root (the canonical write target)
    assert "enforce_research_pipeline_output_dir(" in body
    assert "args.output_root" in body


def test_monolithic_tool_imports_canonical_helper() -> None:
    body = _read_tool_body(MONOLITHIC_TOOL)
    assert "tac.research_pipeline_output_dir_safety" in body
    assert "enforce_research_pipeline_output_dir" in body
    assert "OutputDirSafetyError" in body


def test_monolithic_tool_declares_overwrite_flags() -> None:
    body = _read_tool_body(MONOLITHIC_TOOL)
    assert "--allow-overwrite-existing-historical-provenance" in body
    assert "--overwrite-rationale" in body


def test_monolithic_tool_invokes_helper_against_json_out_parent() -> None:
    body = _read_tool_body(MONOLITHIC_TOOL)
    # canonical helper validates parent dir of --json-out (file output)
    assert "enforce_research_pipeline_output_dir(" in body
    assert "args.json_out.parent" in body


def test_monolithic_tool_skips_helper_when_no_json_out() -> None:
    """Tool supports stdout-only mode (--json-out optional)."""
    body = _read_tool_body(MONOLITHIC_TOOL)
    # The canonical helper invocation is gated on args.json_out is not None
    assert "if args.json_out is not None:" in body


# ---------------------------------------------------------------------------
# Catalog #381 STRICT gate extension — live count remains 0 across 5 producers.
# ---------------------------------------------------------------------------


def test_catalog_381_gate_extended_to_5_producers() -> None:
    from tac.preflight import _CHECK_381_RESEARCH_PIPELINE_TOOLS

    expected = {
        "tools/run_pr95_local_training_probe.py",
        "tools/run_repair_autonomous_multi_archive_runner.py",
        "tools/build_frontier_final_rate_attack_queue.py",
        "tools/build_pr95_mlx_optimizer_matrix_queue.py",
        "tools/build_monolithic_runtime_consumption_proof.py",
    }
    assert set(_CHECK_381_RESEARCH_PIPELINE_TOOLS) == expected, (
        "Catalog #381 producer set must include the 5 canonical producers per "
        "the canonical anti-pattern producers field "
        "(research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1)"
    )


def test_catalog_381_strict_live_count_zero_with_extended_producer_set() -> None:
    from tac.preflight import (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper,
    )

    violations = (
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=False, verbose=False
        )
    )
    assert violations == [], (
        f"Catalog #381 STRICT gate must be at live count 0 across the 5-producer "
        f"set; got {len(violations)} violation(s): {violations}"
    )


def test_catalog_381_strict_mode_raises_when_synthetic_violation(tmp_path: Path) -> None:
    """Synthetic test: a sister tool stub with no canonical helper triggers strict-raise."""

    from tac.preflight import (
        PreflightError,
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper,
        _CHECK_381_RESEARCH_PIPELINE_TOOLS,
    )

    # synthesize a tool with no canonical helper invocation
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    bad_tool = tools_dir / "build_pr95_mlx_optimizer_matrix_queue.py"
    bad_tool.write_text(
        "#!/usr/bin/env python3\n"
        "# missing canonical helper invocation\n"
        "print('hello')\n"
    )

    with pytest.raises(PreflightError, match="Catalog #381|RESEARCH_PIPELINE_OUTPUT_DIR_SAFETY"):
        check_research_pipeline_tools_route_through_canonical_output_dir_safety_helper(
            strict=True, verbose=False, repo_root=tmp_path
        )


# ---------------------------------------------------------------------------
# End-to-end smoke — canonical helper actually fires in subprocess invocation.
# ---------------------------------------------------------------------------


def _build_monolithic_smoke_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Build minimal candidate manifest + runtime log for E2E smoke."""

    manifest = {
        "schema": "tac_monolithic_packet_candidate_v1",
        "candidate_id": "slot_f_smoke",
        "score_claim": False,
        "candidate_archive": {
            "sha256": "a" * 64,
            "bytes": 1024,
            "path": str(tmp_path / "test.zip"),
        },
        "monolithic_layout": {"new_member_sha256": "b" * 64},
        "replacements": [{"section_name": "seg", "new_sha256": "c" * 64}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text(
        "candidate_archive_sha256=" + "a" * 64 + "\n"
        "new_member_sha256=" + "b" * 64 + "\n"
        "section_sha=" + "c" * 64
    )
    return manifest_path, runtime_log


def test_monolithic_tool_e2e_cascade_a_out_of_scope(tmp_path: Path) -> None:
    """Cascade A: --json-out NOT under .omx/research/ → PROCEED."""

    manifest_path, runtime_log = _build_monolithic_smoke_inputs(tmp_path)
    out = tmp_path / "fresh.json"
    rc = subprocess.call(
        [
            sys.executable,
            str(MONOLITHIC_TOOL),
            "--candidate-manifest",
            str(manifest_path),
            "--command-text",
            "foo",
            "--runtime-log",
            str(runtime_log),
            "--json-out",
            str(out),
        ],
        cwd=str(REPO_ROOT),
    )
    assert rc == 0, f"Cascade A expected rc=0, got rc={rc}"
    assert out.exists()


def test_monolithic_tool_e2e_cascade_d_refuses_existing_historical_provenance(
    tmp_path: Path,
) -> None:
    """Cascade D: --json-out parent already contains canonical HISTORICAL_PROVENANCE → REFUSE rc=3."""

    manifest_path, runtime_log = _build_monolithic_smoke_inputs(tmp_path)

    # Synthesize a per-invocation .omx/research/ dir with canonical HISTORICAL_PROVENANCE JSON.
    # Use tmp_path-based isolated repo subtree to avoid touching real .omx/research/.
    research_root = tmp_path / ".omx" / "research" / "slot_f_test_d_refuse"
    research_root.mkdir(parents=True)
    (research_root / "manifest.json").write_text('{"existing": true}')

    out = research_root / "proof.json"
    # We must invoke from the synthesized repo root so the canonical helper
    # sees this tmp_path as repo_root. The tool resolves repo_root via
    # repo_root_from_tool(__file__), which uses the real repo. So we use the
    # REAL .omx/research/ for this smoke (with cleanup), via short-lived dir.
    real_research = REPO_ROOT / ".omx" / "research" / "slot_f_test_d_refuse_unique_20260529T051500Z"
    real_research.mkdir(parents=True, exist_ok=True)
    (real_research / "manifest.json").write_text('{"existing": true}')
    try:
        out = real_research / "proof.json"
        rc = subprocess.call(
            [
                sys.executable,
                str(MONOLITHIC_TOOL),
                "--candidate-manifest",
                str(manifest_path),
                "--command-text",
                "foo",
                "--runtime-log",
                str(runtime_log),
                "--json-out",
                str(out),
            ],
            cwd=str(REPO_ROOT),
        )
        assert rc == 3, f"Cascade D expected rc=3, got rc={rc}"
        assert not out.exists(), "Cascade D must refuse BEFORE writing"
    finally:
        # clean up
        import shutil

        shutil.rmtree(real_research, ignore_errors=True)


def test_monolithic_tool_e2e_cascade_c_explicit_opt_in(tmp_path: Path) -> None:
    """Cascade C: --allow-overwrite-existing-historical-provenance + --overwrite-rationale → PROCEED."""

    manifest_path, runtime_log = _build_monolithic_smoke_inputs(tmp_path)
    real_research = REPO_ROOT / ".omx" / "research" / "slot_f_test_c_optin_unique_20260529T051501Z"
    real_research.mkdir(parents=True, exist_ok=True)
    (real_research / "manifest.json").write_text('{"existing": true}')
    try:
        out = real_research / "proof.json"
        rc = subprocess.call(
            [
                sys.executable,
                str(MONOLITHIC_TOOL),
                "--candidate-manifest",
                str(manifest_path),
                "--command-text",
                "foo",
                "--runtime-log",
                str(runtime_log),
                "--json-out",
                str(out),
                "--allow-overwrite-existing-historical-provenance",
                "--overwrite-rationale",
                "slot F smoke test of canonical helper Cascade C explicit opt-in path",
            ],
            cwd=str(REPO_ROOT),
        )
        assert rc == 0, f"Cascade C expected rc=0, got rc={rc}"
        assert out.exists()
    finally:
        import shutil

        shutil.rmtree(real_research, ignore_errors=True)


# ---------------------------------------------------------------------------
# Catalog #287 placeholder-rationale rejection — exercised through canonical helper
# ---------------------------------------------------------------------------


def test_monolithic_tool_e2e_cascade_c_rejects_placeholder_rationale(
    tmp_path: Path,
) -> None:
    """Placeholder rationale `<rationale>` rejected per Catalog #287."""

    manifest_path, runtime_log = _build_monolithic_smoke_inputs(tmp_path)
    real_research = REPO_ROOT / ".omx" / "research" / "slot_f_test_c_placeholder_unique_20260529T051502Z"
    real_research.mkdir(parents=True, exist_ok=True)
    (real_research / "manifest.json").write_text('{"existing": true}')
    try:
        out = real_research / "proof.json"
        rc = subprocess.call(
            [
                sys.executable,
                str(MONOLITHIC_TOOL),
                "--candidate-manifest",
                str(manifest_path),
                "--command-text",
                "foo",
                "--runtime-log",
                str(runtime_log),
                "--json-out",
                str(out),
                "--allow-overwrite-existing-historical-provenance",
                "--overwrite-rationale",
                "<rationale>",
            ],
            cwd=str(REPO_ROOT),
        )
        assert rc == 3, f"Placeholder rationale must be rejected; expected rc=3, got rc={rc}"
    finally:
        import shutil

        shutil.rmtree(real_research, ignore_errors=True)
