"""Smoke tests for the silent-defaults audit tool.

DX #1 (2026-04-26): the audit tool is the first line of defence against
the KL-distill bug class. These tests verify it (a) imports, (b) produces
a non-empty report, and (c) correctly classifies a known-critical entry.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools" / "audit_silent_defaults.py"


def _load_audit_module():
    """Side-load the audit script as a module so tests can call its API
    directly. The tool is intentionally a script (no setup.py entry point)
    so importlib.util is the cleanest path."""
    spec = importlib.util.spec_from_file_location("audit_silent_defaults", TOOL_PATH)
    assert spec and spec.loader, f"could not load {TOOL_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_silent_defaults"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_audit_tool_loads():
    """Sanity: the script imports without side-effects on import path."""
    mod = _load_audit_module()
    assert hasattr(mod, "collect_records")
    assert hasattr(mod, "write_report")
    assert hasattr(mod, "_import_profile_keys")


def test_collect_records_finds_known_args():
    """The audit must find at least the well-known --profile and
    --kl-distill-weight flags in train_renderer.py. If this regresses,
    SCAN_DIRS or the AST visitor is broken."""
    mod = _load_audit_module()
    records = mod.collect_records()
    assert records, "expected non-empty records list"
    arg_names = {r["arg_name"] for r in records}
    # train_renderer.py is canonical; both flags MUST be present.
    assert "--profile" in arg_names, f"missing --profile (found {len(arg_names)} args)"
    assert "--kl-distill-weight" in arg_names


def test_argname_normalisation():
    """Verify the argparse-name → profile-key conversion is idempotent
    and matches Python's add_argument-to-Namespace transformation."""
    mod = _load_audit_module()
    assert mod._argname_to_key("--kl-distill-weight") == "kl_distill_weight"
    assert mod._argname_to_key("--use-zoom-flow") == "use_zoom_flow"
    assert mod._argname_to_key("--profile") == "profile"


def test_is_risky_default_classifies_kl_pattern():
    """The KL-distill bug pattern (default=0.0 with matching profile key)
    is exactly what the auditor must flag. Verify the classifier agrees."""
    mod = _load_audit_module()
    # Risky: a numeric default
    assert mod._is_risky_default({
        "default_repr": "0.0", "action": None, "key": "kl_distill_weight",
    })
    # Safe: default=None
    assert not mod._is_risky_default({
        "default_repr": "None", "action": None, "key": "kl_distill_weight",
    })
    # Safe: store_true with default=None (the new pattern)
    assert not mod._is_risky_default({
        "default_repr": "None", "action": "store_true", "key": "use_zoom_flow",
    })


def test_report_writes_markdown(tmp_path, monkeypatch):
    """End-to-end: the tool produces a valid markdown report with the
    expected sections and counts."""
    mod = _load_audit_module()
    report_path = tmp_path / "report.md"
    monkeypatch.setattr(mod, "REPORT", report_path)
    profile_keys = mod._import_profile_keys()
    records = mod.collect_records()
    mod.write_report(records, profile_keys)
    text = report_path.read_text()
    assert "# Silent Argparse Defaults Audit" in text
    assert "## CRITICAL" in text
    assert "## SUSPICIOUS" in text
    # The report must mention at least one of the canonical training scripts.
    assert "train_renderer.py" in text or "pipeline.py" in text
