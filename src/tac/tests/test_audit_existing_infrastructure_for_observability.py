# SPDX-License-Identifier: MIT
"""Tests for tools/audit_existing_infrastructure_for_observability.py.

Per operator standing directive 2026-05-16 (MAX-OBSERVABILITY-INTO-BEHAVIOR);
sister of Catalog #305 STRICT preflight gate.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
AUDIT_TOOL = REPO_ROOT / "tools" / "audit_existing_infrastructure_for_observability.py"


def _load_audit_module():
    """Load the audit tool as a module for direct API access.

    Python 3.12 dataclass introspection requires the module be registered in
    sys.modules BEFORE exec_module is called (it walks cls.__module__ → sys.modules
    to resolve forward references and types).
    """
    module_name = "_audit_observability_test_loader"
    spec = importlib.util.spec_from_file_location(module_name, AUDIT_TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def test_audit_tool_exists() -> None:
    assert AUDIT_TOOL.exists(), f"Missing canonical audit tool at {AUDIT_TOOL}"


def test_audit_tool_cli_json_emit_succeeds() -> None:
    """The audit CLI should emit valid JSON and exit rc=0."""
    result = subprocess.run(
        [sys.executable, str(AUDIT_TOOL)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )
    assert result.returncode == 0, f"audit tool failed: {result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "observability_audit_v1"
    assert payload["tools_audited"] >= 8
    assert "reports" in payload
    assert len(payload["reports"]) == payload["tools_audited"]


def test_audit_tool_cli_summary_succeeds() -> None:
    """The --summary mode should emit human-readable output."""
    result = subprocess.run(
        [sys.executable, str(AUDIT_TOOL), "--summary"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )
    assert result.returncode == 0
    assert "Observability audit:" in result.stdout
    assert "tac.cost_band_calibration" in result.stdout
    assert "Highest ROI extensions:" in result.stdout


def test_audit_reports_canonical_tools_covered() -> None:
    """All 8 canonical tools required by the memory file consequence 3 are
    covered by the audit."""
    mod = _load_audit_module()
    audit = mod.build_full_audit(REPO_ROOT)
    tool_names = {r["tool_name"] for r in audit["reports"]}
    required = {
        "tac.sensitivity_map",
        "tac.cost_band_calibration",
        "tools/cathedral_autopilot_autonomous_loop.py",
        "src/tac/xray/*",
        "tools/audit_*.py family",
        "experiments/contest_auth_eval.py",
        "tac.continual_learning.posterior_update_locked",
        "tac.council_continual_learning (Catalog #300)",
    }
    assert tool_names == required, (
        f"Audit tool coverage mismatch: missing {required - tool_names}; "
        f"extra {tool_names - required}"
    )


def test_audit_each_tool_has_6_facets() -> None:
    """Per the directive, each tool MUST be scored across all 6 observability
    facets."""
    mod = _load_audit_module()
    audit = mod.build_full_audit(REPO_ROOT)
    expected_facets = {
        "per_layer_inspection",
        "per_signal_decomposition",
        "run_to_run_diff",
        "post_hoc_query",
        "cite_chain",
        "counterfactual_hooks",
    }
    for r in audit["reports"]:
        facet_names = {f["facet_name"] for f in r["facets"]}
        assert facet_names == expected_facets, (
            f"Tool {r['tool_name']} missing facets: "
            f"{expected_facets - facet_names}"
        )


def test_audit_score_in_valid_range() -> None:
    """All facet scores MUST be in [0, 2] range; total MUST be in [0, 12]."""
    mod = _load_audit_module()
    audit = mod.build_full_audit(REPO_ROOT)
    for r in audit["reports"]:
        assert 0 <= r["total_score"] <= 12, (
            f"Tool {r['tool_name']} total_score {r['total_score']} out of range."
        )
        for f in r["facets"]:
            assert 0 <= f["score"] <= 2, (
                f"Tool {r['tool_name']} facet {f['facet_name']} score "
                f"{f['score']} out of range."
            )


def test_audit_highest_roi_extensions_present() -> None:
    """The audit surfaces the top extension targets for follow-on subagents."""
    mod = _load_audit_module()
    audit = mod.build_full_audit(REPO_ROOT)
    assert "highest_roi_extensions" in audit
    assert len(audit["highest_roi_extensions"]) >= 1
    # The audit_*.py family is the lowest-scoring (3/12) so it should appear.
    extensions_blob = " ".join(audit["highest_roi_extensions"])
    assert "audit_" in extensions_blob.lower() or "AuditReport" in extensions_blob


def test_audit_cites_directive_verbatim() -> None:
    """The audit report cites the standing directive verbatim per
    apples-to-apples evidence discipline."""
    mod = _load_audit_module()
    audit = mod.build_full_audit(REPO_ROOT)
    assert "directive_verbatim" in audit
    assert "absolute" in audit["directive_verbatim"]
    assert "observability" in audit["directive_verbatim"]


def test_audit_anchors_to_memory_file() -> None:
    """The audit report cites the canonical memory file anchor."""
    mod = _load_audit_module()
    audit = mod.build_full_audit(REPO_ROOT)
    assert "anchor_memo" in audit
    assert "max_observability_into_behavior" in audit["anchor_memo"]
