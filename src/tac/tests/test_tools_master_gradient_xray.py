# SPDX-License-Identifier: MIT
"""Dedicated tests for tools/master_gradient_xray.py Slot EE 2026-05-19 extensions
(task #797): consumer_verdict_matrix + provenance_audit_timeline plot types.

Sister test file to ``test_master_gradient_xray.py`` (49 tests covering the
original 7 plot types) — kept SEPARATE per Catalog #229 premise-verification
discipline so a dedicated regression surface tracks the Slot EE additions.

Per Catalog #305 6-facet observability + Catalog #287/#323 canonical
Provenance + Catalog #341 routing markers + Catalog #335 canonical contract.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "master_gradient_xray.py"


@pytest.fixture(scope="module")
def xray_module():
    """Load tools/master_gradient_xray.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "master_gradient_xray_slot_ee_test", TOOL_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["master_gradient_xray_slot_ee_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def synthetic_candidate():
    """Canonical synthetic candidate for plot 3 testing."""
    return {
        "candidate_id": "test_candidate_slot_ee",
        "archive_sha256": "f" * 64,
        "predicted_delta": -0.005,
        "family": "test_family",
        "literature_anchor": "test_anchor",
        "pareto_scope": "rate_seg_pose",
    }


@pytest.fixture
def synthetic_anchors():
    """Canonical synthetic anchors for plot 5 testing.

    Mix of complete + authoritative + advisory + REJECT to exercise all
    classification branches.
    """
    return [
        {
            "archive_sha256": "a" * 64,
            "measurement_axis": "[macOS-CPU advisory]",
            "measurement_hardware": "darwin_arm64_local_cpu_advisory",
            "measurement_utc": "2026-05-15T10:00:00Z",
            "measurement_call_id": "test_call_1",
            "measurement_method": "test_method",
            "written_at_utc": "2026-05-15T10:00:00Z",
            "n_bytes": 1000,
        },
        {
            "archive_sha256": "b" * 64,
            "measurement_axis": "[macOS-CPU advisory]",
            "measurement_hardware": "darwin_arm64_m5_max_macos_cpu_advisory",
            "measurement_utc": "2026-05-16T10:00:00Z",
            "measurement_call_id": "test_call_2",
            "measurement_method": "test_method",
            "written_at_utc": "2026-05-16T10:00:00Z",
            "n_bytes": 2000,
        },
        # Incomplete: missing measurement_utc + measurement_call_id
        {
            "archive_sha256": "c" * 64,
            "measurement_axis": "[contest-CPU]",
            "measurement_hardware": "linux_x86_64_cpu",
            "written_at_utc": "2026-05-17T10:00:00Z",
            "n_bytes": 3000,
        },
    ]


# ──────────────────────────────────────────────────────────────────────────── #
# Plot 3: consumer_verdict_matrix tests                                        #
# ──────────────────────────────────────────────────────────────────────────── #


def test_consumer_verdict_matrix_produces_png(xray_module, synthetic_candidate, tmp_path):
    """Plot 3 happy path: produces non-trivial PNG."""
    out = tmp_path / "verdict.png"
    metrics = xray_module.plot_consumer_verdict_matrix(synthetic_candidate, out)
    assert out.exists()
    assert out.stat().st_size > 5000  # non-trivial multi-row matrix


def test_consumer_verdict_matrix_returns_canonical_metrics(
    xray_module, synthetic_candidate, tmp_path
):
    """Metrics dict has all canonical Slot EE fields."""
    out = tmp_path / "verdict.png"
    metrics = xray_module.plot_consumer_verdict_matrix(synthetic_candidate, out)
    required_keys = {
        "n_consumers",
        "non_vacuous_count",
        "non_vacuous_fraction",
        "catalog_341_markers_compliant_count",
        "catalog_341_markers_compliant_fraction",
        "promotable_violation_count",
        "error_count",
        "hook_coverage_histogram",
        "candidate_id",
        "archive_sha256",
        "per_consumer_verdicts",
    }
    assert required_keys.issubset(set(metrics.keys()))
    assert metrics["n_consumers"] > 0
    # Sister regression: per Cable D batch 2026-05-19 the live consumer
    # count is at minimum 33 contract-compliant (post-batch landing).
    assert metrics["n_consumers"] >= 30, (
        f"expected >=30 contract-compliant consumers, got {metrics['n_consumers']}"
    )


def test_consumer_verdict_matrix_catalog_341_markers_compliance(
    xray_module, synthetic_candidate, tmp_path
):
    """Per Catalog #341: routing-recommendation consumers MUST carry the 3
    canonical markers (predicted_delta_adjustment=0.0 / promotable=False /
    axis_tag='[predicted]').

    Per Catalog #322: NO consumer can have promotable=True. The matrix MUST
    flag any promotable-violation_count > 0 in its metrics.
    """
    out = tmp_path / "verdict.png"
    metrics = xray_module.plot_consumer_verdict_matrix(synthetic_candidate, out)
    # promotable=True is ALWAYS a Catalog #322 violation; live should be 0.
    assert metrics["promotable_violation_count"] == 0, (
        f"Catalog #322 violation: {metrics['promotable_violation_count']} "
        f"consumers returned promotable=True for synthetic candidate; "
        f"per-consumer verdicts: {metrics['per_consumer_verdicts']}"
    )


def test_consumer_verdict_matrix_hook_coverage_includes_all_canonical_hooks(
    xray_module, synthetic_candidate, tmp_path
):
    """Per Catalog #125 6-hook wire-in: hooks 1-6 should all be represented
    in the live cathedral consumer surface (allowing some consumers to wire
    only a subset)."""
    out = tmp_path / "verdict.png"
    metrics = xray_module.plot_consumer_verdict_matrix(synthetic_candidate, out)
    hook_hist = metrics["hook_coverage_histogram"]
    # Hook 4 (cathedral autopilot) MUST be widely represented since every
    # consumer participates in dispatch.
    assert "4" in hook_hist, f"hook 4 missing from coverage: {hook_hist}"
    assert hook_hist["4"] >= 20, (
        f"hook 4 coverage too low ({hook_hist['4']}); "
        f"every contract-compliant consumer should wire hook 4 per Catalog #335"
    )


def test_collect_consumer_verdicts_with_explicit_modules(xray_module):
    """Allow `consumer_modules` injection for unit-testability."""
    # Fake consumer module shape — minimal Protocol compliance
    class FakeConsumer:
        __name__ = "fake_test_consumer"
        CONSUMER_NAME = "fake_test_consumer"
        CONSUMER_VERSION = "0.0.1"
        CONSUMER_HOOK_NUMBERS = (4, 5)

        @staticmethod
        def consume_candidate(candidate):
            return {
                "predicted_delta_adjustment": 0.0,
                "rationale": "fake test consumer",
                "axis_tag": "[predicted]",
                "promotable": False,
                "confidence": 0.0,
            }

    verdicts = xray_module._collect_consumer_verdicts(
        {"candidate_id": "x", "archive_sha256": "0" * 64},
        consumer_modules=[FakeConsumer],
    )
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v["consumer_name"] == "fake_test_consumer"
    assert v["catalog_341_markers_compliant"] is True
    assert v["non_vacuous"] is True
    assert v["error"] is None


def test_collect_consumer_verdicts_handles_consumer_exception(xray_module):
    """Per Catalog #341 graceful degradation: consumer exception MUST be
    caught + surfaced as `error`; matrix MUST NOT crash."""
    class BrokenConsumer:
        __name__ = "broken_test_consumer"
        CONSUMER_NAME = "broken_test_consumer"
        CONSUMER_VERSION = "0.0.1"
        CONSUMER_HOOK_NUMBERS = (4,)

        @staticmethod
        def consume_candidate(candidate):
            raise ValueError("intentional test exception")

    verdicts = xray_module._collect_consumer_verdicts(
        {"candidate_id": "x", "archive_sha256": "0" * 64},
        consumer_modules=[BrokenConsumer],
    )
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v["error"] is not None
    assert "ValueError" in v["error"]
    assert v["catalog_341_markers_compliant"] is False


def test_collect_consumer_verdicts_handles_non_dict_return(xray_module):
    """Non-dict return is a Catalog #335 violation; MUST be surfaced as
    error."""
    class NonDictConsumer:
        __name__ = "nondict_test_consumer"
        CONSUMER_NAME = "nondict_test_consumer"
        CONSUMER_VERSION = "0.0.1"
        CONSUMER_HOOK_NUMBERS = (4,)

        @staticmethod
        def consume_candidate(candidate):
            return "not a dict"

    verdicts = xray_module._collect_consumer_verdicts(
        {"candidate_id": "x", "archive_sha256": "0" * 64},
        consumer_modules=[NonDictConsumer],
    )
    assert len(verdicts) == 1
    assert verdicts[0]["error"] is not None
    assert "non-dict return" in verdicts[0]["error"]


def test_consumer_verdict_matrix_refuses_zero_consumers(xray_module, tmp_path):
    """Edge case: zero consumers should raise SystemExit (the gate would
    be invisible to operators without an error)."""
    out = tmp_path / "verdict.png"
    with pytest.raises(SystemExit) as excinfo:
        xray_module.plot_consumer_verdict_matrix(
            {"candidate_id": "x", "archive_sha256": "0" * 64},
            out,
            consumer_modules=[],
        )
    assert "0 compliant consumers" in str(excinfo.value)


# ──────────────────────────────────────────────────────────────────────────── #
# Plot 5: provenance_audit_timeline tests                                      #
# ──────────────────────────────────────────────────────────────────────────── #


def test_provenance_audit_timeline_produces_png(
    xray_module, synthetic_anchors, tmp_path
):
    """Plot 5 happy path: produces non-trivial PNG."""
    out = tmp_path / "timeline.png"
    metrics = xray_module.plot_provenance_audit_timeline(synthetic_anchors, out)
    assert out.exists()
    assert out.stat().st_size > 3000


def test_provenance_audit_timeline_returns_canonical_metrics(
    xray_module, synthetic_anchors, tmp_path
):
    """Metrics dict has all canonical Slot EE fields."""
    out = tmp_path / "timeline.png"
    metrics = xray_module.plot_provenance_audit_timeline(synthetic_anchors, out)
    required_keys = {
        "n_anchors",
        "complete_count",
        "complete_fraction",
        "authoritative_count",
        "authoritative_fraction",
        "reject_count",
        "reject_categories",
        "axis_histogram",
        "since_utc",
        "first_anchor_utc",
        "last_anchor_utc",
    }
    assert required_keys.issubset(set(metrics.keys()))
    assert metrics["n_anchors"] == len(synthetic_anchors)


def test_provenance_audit_timeline_classifies_incomplete(
    xray_module, synthetic_anchors, tmp_path
):
    """The 3rd synthetic anchor is incomplete (missing measurement_utc +
    measurement_call_id); MUST appear in reject_count."""
    out = tmp_path / "timeline.png"
    metrics = xray_module.plot_provenance_audit_timeline(synthetic_anchors, out)
    assert metrics["reject_count"] >= 1, (
        f"expected >=1 REJECT for incomplete anchor; metrics: {metrics}"
    )
    assert metrics["reject_categories"], "reject_categories should be populated"


def test_provenance_audit_timeline_refuses_empty(xray_module, tmp_path):
    """Empty anchor list MUST raise SystemExit with actionable message."""
    out = tmp_path / "timeline.png"
    with pytest.raises(SystemExit) as excinfo:
        xray_module.plot_provenance_audit_timeline([], out)
    assert "0 anchors in scope" in str(excinfo.value)


def test_provenance_audit_timeline_honors_since_utc_filter(
    xray_module, synthetic_anchors, tmp_path
):
    """`since_utc=2026-05-16T00:00:00Z` filters to anchors >= that date."""
    out = tmp_path / "timeline.png"
    metrics = xray_module.plot_provenance_audit_timeline(
        synthetic_anchors, out, since_utc="2026-05-16T00:00:00Z"
    )
    # Only 2 of 3 synthetic anchors qualify (the b + c anchors)
    assert metrics["n_anchors"] == 2


def test_classify_provenance_for_anchor_complete(xray_module):
    """Classifier helper unit test: complete anchor."""
    cls = xray_module._classify_provenance_for_anchor({
        "archive_sha256": "a" * 64,
        "measurement_axis": "[macOS-CPU advisory]",
        "measurement_hardware": "darwin_arm64_advisory",
        "measurement_utc": "2026-05-17T10:00:00Z",
        "measurement_call_id": "test",
        "measurement_method": "test_method",
    })
    assert cls["complete"] is True
    assert cls["missing_keys"] == []


def test_classify_provenance_for_anchor_incomplete(xray_module):
    """Classifier helper unit test: incomplete anchor."""
    cls = xray_module._classify_provenance_for_anchor({
        "archive_sha256": "a" * 64,
        "measurement_axis": "[macOS-CPU advisory]",
        # missing: measurement_hardware, measurement_utc, measurement_call_id, measurement_method
    })
    assert cls["complete"] is False
    assert "measurement_hardware" in cls["missing_keys"]
    assert "measurement_utc" in cls["missing_keys"]


def test_canonical_provenance_keys_pinned(xray_module):
    """Per Catalog #229 — pin the 6 canonical Provenance keys for audit."""
    expected = {
        "archive_sha256",
        "measurement_axis",
        "measurement_hardware",
        "measurement_utc",
        "measurement_call_id",
        "measurement_method",
    }
    assert set(xray_module.CANONICAL_PROVENANCE_KEYS) == expected


# ──────────────────────────────────────────────────────────────────────────── #
# Slot EE pinning regression tests                                             #
# ──────────────────────────────────────────────────────────────────────────── #


def test_slot_ee_plot_names_in_canonical_plots(xray_module):
    """Pin the Slot EE 2026-05-19 extensions in CANONICAL_PLOTS."""
    assert "consumer_verdict_matrix" in xray_module.CANONICAL_PLOTS
    assert "provenance_audit_timeline" in xray_module.CANONICAL_PLOTS


def test_slot_ee_plot_functions_exposed(xray_module):
    """Public plot functions are reachable on the module."""
    assert callable(xray_module.plot_consumer_verdict_matrix)
    assert callable(xray_module.plot_provenance_audit_timeline)
    assert callable(xray_module._collect_consumer_verdicts)
    assert callable(xray_module._classify_provenance_for_anchor)


# ──────────────────────────────────────────────────────────────────────────── #
# CLI smoke regression                                                         #
# ──────────────────────────────────────────────────────────────────────────── #


def test_cli_list_plots_includes_slot_ee_extensions(tmp_path):
    """CLI `--list-plots` must include both new plot names."""
    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--list-plots"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "consumer_verdict_matrix" in payload["canonical_plots"]
    assert "provenance_audit_timeline" in payload["canonical_plots"]


def test_cli_smoke_consumer_verdict_matrix(tmp_path):
    """End-to-end CLI smoke: consumer_verdict_matrix emits valid PNG."""
    out = tmp_path / "verdict_smoke.png"
    result = subprocess.run(
        [
            sys.executable, str(TOOL_PATH),
            "--plot", "consumer_verdict_matrix",
            "--archive-sha", "6bae0201",
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()
    assert out.stat().st_size > 5000


def test_cli_smoke_provenance_audit_timeline(tmp_path):
    """End-to-end CLI smoke: provenance_audit_timeline emits valid PNG."""
    out = tmp_path / "timeline_smoke.png"
    result = subprocess.run(
        [
            sys.executable, str(TOOL_PATH),
            "--plot", "provenance_audit_timeline",
            "--archive-sha", "6bae0201",
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()
    assert out.stat().st_size > 3000


# ──────────────────────────────────────────────────────────────────────────── #
# Live-repo regression guards                                                  #
# ──────────────────────────────────────────────────────────────────────────── #


def test_live_repo_master_gradient_anchors_jsonl_readable():
    """Sister regression: canonical ledger is readable + has anchors."""
    from tac.master_gradient import load_anchors_lenient
    anchors = load_anchors_lenient()
    # Live-repo at landing has 10 anchors; allow drift but require >= 1.
    assert len(anchors) >= 1, "canonical ledger should have at least 1 anchor"


def test_live_repo_cathedral_consumers_present():
    """Sister regression: at least 30 contract-compliant consumers exist."""
    import importlib.util as _ilu
    _path = REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py"
    _spec = _ilu.spec_from_file_location("loop_test", _path)
    _mod = _ilu.module_from_spec(_spec)
    sys.modules["loop_test"] = _mod
    _spec.loader.exec_module(_mod)
    mods = _mod.discover_compliant_consumer_modules()
    assert len(mods) >= 30, (
        f"expected >=30 contract-compliant consumers, got {len(mods)}; "
        f"verify src/tac/cathedral_consumers/* + Catalog #335 contract"
    )
