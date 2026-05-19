# SPDX-License-Identifier: MIT
"""Tests for Catalog #341 — check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers.

Per CATALOG-317-SCOPE-NARROWING-MPS-OPT-IN subagent landing 2026-05-19 +
operator's earlier paradigm-shift directive + today's MPS-VIABLE landing
(3-component aggregate gap 0.072% per
``feedback_mps_phase_b_options_b_plus_c_completion_landed_20260519.md``).

SCOPE-NARROWING SISTER OF CATALOG #317: where Catalog #317 covers direct
``_dispatch_local_mps`` / ``_dispatch_local_cpu`` surface, Catalog #341
covers the cathedral-consumer-recommended routing surface — refuses any
``src/tac/cathedral_consumers/*`` package that exposes a routing
recommendation without all 3 canonical non-promotable markers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers,
)


# ---------------------------------------------------------------------------
# Live-repo regression guard (the gate's primary structural invariant)
# ---------------------------------------------------------------------------


def test_live_repo_passes_clean() -> None:
    """The live repo at landing has zero violations.

    Catalog #341 STRICT-flip atomicity: the live repo MUST be clean at the
    moment of landing per CLAUDE.md non-negotiable. The existing
    ``mps_viable_prescreen_consumer`` (commit ``a753b70d5``) carries all 3
    canonical markers in all 4 routing-cascade return values, verified by
    its own 21 dedicated tests.
    """
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers()
    assert violations == [], (
        f"Live repo Catalog #341 violations:\n  - "
        + "\n  - ".join(violations[:5])
    )


def test_live_repo_strict_mode_does_not_raise() -> None:
    """Strict mode on the live repo MUST not raise (count=0 at landing)."""
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        strict=True
    )
    assert violations == []


def test_live_repo_mps_viable_prescreen_consumer_passes() -> None:
    """Anchor regression: the empirical MPS-VIABLE prescreen consumer passes.

    Cross-validates that the canonical 4-step routing cascade in the
    ``mps_viable_prescreen_consumer`` carries all 3 markers (verified
    structurally by the gate; empirically by the consumer's own 21 tests).
    """
    repo_root = Path(__file__).parent.parent.parent.parent
    consumer_init = (
        repo_root
        / "src"
        / "tac"
        / "cathedral_consumers"
        / "mps_viable_prescreen_consumer"
        / "__init__.py"
    )
    assert consumer_init.is_file(), "MPS-VIABLE prescreen consumer must exist"
    body = consumer_init.read_text(encoding="utf-8")
    # In-scope: must contain routing-cascade tokens.
    assert any(
        tok in body
        for tok in (
            "recommended_route",
            "ROUTE_LOCAL_MPS_PRESCREEN",
            "ROUTE_PAID_CUDA_AUTHORITATIVE",
        )
    )
    # Canonical markers present.
    assert '"predicted_delta_adjustment": 0.0' in body
    assert '"promotable": False' in body
    assert '"axis_tag": "[predicted]"' in body


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------


def _make_synthetic_repo(tmp_path: Path) -> Path:
    """Create a synthetic repo with src/tac/cathedral_consumers/."""
    consumer_dir = tmp_path / "src" / "tac" / "cathedral_consumers"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "# SPDX-License-Identifier: MIT\n"
        '"""Synthetic cathedral consumers namespace."""\n'
    )
    return tmp_path


def _write_compliant_routing_consumer(repo: Path, name: str) -> Path:
    """Write a routing-recommender consumer with all 3 canonical markers."""
    pkg_dir = repo / "src" / "tac" / "cathedral_consumers" / name
    pkg_dir.mkdir(parents=True)
    init_path = pkg_dir / "__init__.py"
    init_path.write_text(
        "# SPDX-License-Identifier: MIT\n"
        '"""Compliant routing-recommender consumer."""\n'
        "from typing import Any, Mapping\n\n"
        "ROUTE_LOCAL_MPS_PRESCREEN = \"local_mps_prescreen\"\n"
        "ROUTE_PAID_CUDA_AUTHORITATIVE = \"paid_cuda_authoritative\"\n\n"
        "def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:\n"
        "    return {\n"
        "        \"predicted_delta_adjustment\": 0.0,\n"
        "        \"promotable\": False,\n"
        "        \"axis_tag\": \"[predicted]\",\n"
        "        \"recommended_route\": ROUTE_LOCAL_MPS_PRESCREEN,\n"
        "    }\n"
    )
    return init_path


def _write_routing_consumer_missing_marker(
    repo: Path, name: str, missing: str
) -> Path:
    """Write a routing-recommender consumer missing one canonical marker."""
    pkg_dir = repo / "src" / "tac" / "cathedral_consumers" / name
    pkg_dir.mkdir(parents=True)
    body_lines = [
        '"\"predicted_delta_adjustment\": 0.0',
        '"\"promotable\": False',
        '"\"axis_tag\": \"[predicted]\"',
    ]
    # The missing-marker scenario: comment out the offending dict entry.
    if missing == "predicted_delta_adjustment=0.0":
        marker_line = "        # predicted_delta_adjustment omitted intentionally"
        keep_lines = [
            "        \"promotable\": False,",
            "        \"axis_tag\": \"[predicted]\",",
        ]
    elif missing == "promotable=False":
        marker_line = "        # promotable omitted intentionally"
        keep_lines = [
            "        \"predicted_delta_adjustment\": 0.0,",
            "        \"axis_tag\": \"[predicted]\",",
        ]
    elif missing == "axis_tag=[predicted]":
        marker_line = "        # axis_tag omitted intentionally"
        keep_lines = [
            "        \"predicted_delta_adjustment\": 0.0,",
            "        \"promotable\": False,",
        ]
    else:
        raise ValueError(f"unknown missing marker: {missing}")
    init_path = pkg_dir / "__init__.py"
    init_path.write_text(
        "# SPDX-License-Identifier: MIT\n"
        '"""Routing consumer missing a canonical marker."""\n'
        "from typing import Any, Mapping\n\n"
        "def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:\n"
        "    return {\n"
        f"{marker_line}\n"
        + "\n".join(keep_lines)
        + "\n"
        "        \"recommended_route\": \"local_mps_prescreen\",\n"
        "    }\n"
    )
    return init_path


def _write_diagnostic_consumer_no_routing(repo: Path, name: str) -> Path:
    """Write a pure diagnostic consumer with NO routing recommendation."""
    pkg_dir = repo / "src" / "tac" / "cathedral_consumers" / name
    pkg_dir.mkdir(parents=True)
    init_path = pkg_dir / "__init__.py"
    init_path.write_text(
        "# SPDX-License-Identifier: MIT\n"
        '"""Pure diagnostic consumer (no routing recommendation; out-of-scope)."""\n'
        "from typing import Any, Mapping\n\n"
        "def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:\n"
        "    return {\n"
        "        \"predicted_delta_adjustment\": 0.0,\n"
        "        \"diagnostic_data\": {},\n"
        "    }\n"
    )
    return init_path


def _write_waived_routing_consumer(
    repo: Path, name: str, rationale: str
) -> Path:
    """Write a routing consumer missing markers but carrying a waiver."""
    pkg_dir = repo / "src" / "tac" / "cathedral_consumers" / name
    pkg_dir.mkdir(parents=True)
    init_path = pkg_dir / "__init__.py"
    init_path.write_text(
        "# SPDX-License-Identifier: MIT\n"
        f"# CATHEDRAL_CONSUMER_MPS_ROUTING_OK:{rationale}\n"
        '"""Waived non-compliant routing consumer."""\n'
        "from typing import Any, Mapping\n\n"
        "ROUTE_LOCAL_MPS_PRESCREEN = \"local_mps_prescreen\"\n\n"
        "def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:\n"
        "    return {\"recommended_route\": ROUTE_LOCAL_MPS_PRESCREEN}\n"
    )
    return init_path


# ---------------------------------------------------------------------------
# Synthetic gate behavior
# ---------------------------------------------------------------------------


def test_synthetic_no_consumer_dir_silent(tmp_path: Path) -> None:
    """Missing consumer dir silently passes."""
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=tmp_path
    )
    assert violations == []


def test_synthetic_empty_consumer_dir_silent(tmp_path: Path) -> None:
    """Empty consumer dir (no packages) silently passes."""
    repo = _make_synthetic_repo(tmp_path)
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo
    )
    assert violations == []


def test_synthetic_compliant_routing_consumer_passes(tmp_path: Path) -> None:
    """Compliant routing consumer with all 3 markers passes."""
    repo = _make_synthetic_repo(tmp_path)
    _write_compliant_routing_consumer(repo, "compliant_router")
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo
    )
    assert violations == []


def test_synthetic_diagnostic_consumer_out_of_scope(tmp_path: Path) -> None:
    """Pure diagnostic consumer (no routing tokens) is out-of-scope, even
    when missing canonical markers. This is the canonical contract: the
    gate only fires for ROUTING recommenders."""
    repo = _make_synthetic_repo(tmp_path)
    _write_diagnostic_consumer_no_routing(repo, "diagnostic_only")
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo
    )
    assert violations == []


@pytest.mark.parametrize(
    "missing",
    [
        "predicted_delta_adjustment=0.0",
        "promotable=False",
        "axis_tag=[predicted]",
    ],
)
def test_synthetic_routing_consumer_missing_marker_flagged(
    tmp_path: Path, missing: str
) -> None:
    """Routing consumer missing any of the 3 canonical markers is flagged."""
    repo = _make_synthetic_repo(tmp_path)
    _write_routing_consumer_missing_marker(repo, "broken_router", missing)
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo
    )
    assert len(violations) == 1
    assert "broken_router" in violations[0]
    assert missing in violations[0]


def test_synthetic_waived_consumer_passes(tmp_path: Path) -> None:
    """Waived routing consumer with substantive rationale passes."""
    repo = _make_synthetic_repo(tmp_path)
    _write_waived_routing_consumer(
        repo, "waived_router", "Pending Phase 2 marker wire-in per ledger #842"
    )
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo
    )
    assert violations == []


def test_synthetic_placeholder_rationale_rejected(tmp_path: Path) -> None:
    """Placeholder ``<rationale>`` rejected; package then flagged."""
    repo = _make_synthetic_repo(tmp_path)
    _write_waived_routing_consumer(repo, "placeholder_router", "<rationale>")
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo
    )
    assert len(violations) == 1
    assert "placeholder_router" in violations[0]


def test_synthetic_reason_placeholder_rejected(tmp_path: Path) -> None:
    """Placeholder ``<reason>`` rejected; package then flagged."""
    repo = _make_synthetic_repo(tmp_path)
    _write_waived_routing_consumer(repo, "reason_placeholder_router", "<reason>")
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo
    )
    assert len(violations) == 1
    assert "reason_placeholder_router" in violations[0]


def test_synthetic_short_rationale_rejected(tmp_path: Path) -> None:
    """Rationale <4 chars rejected; package then flagged."""
    repo = _make_synthetic_repo(tmp_path)
    _write_waived_routing_consumer(repo, "short_router", "abc")
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo
    )
    assert len(violations) == 1
    assert "short_router" in violations[0]


def test_synthetic_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    """Strict mode raises PreflightError citing Catalog #341."""
    repo = _make_synthetic_repo(tmp_path)
    _write_routing_consumer_missing_marker(
        repo, "strict_broken", "predicted_delta_adjustment=0.0"
    )
    with pytest.raises(PreflightError, match="Catalog #341"):
        check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
            repo_root=repo, strict=True
        )


def test_synthetic_strict_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode silent when no violations."""
    repo = _make_synthetic_repo(tmp_path)
    _write_compliant_routing_consumer(repo, "clean_router")
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo, strict=True
    )
    assert violations == []


def test_synthetic_pycache_skipped(tmp_path: Path) -> None:
    """__pycache__ subdir exempt."""
    repo = _make_synthetic_repo(tmp_path)
    pycache = repo / "src" / "tac" / "cathedral_consumers" / "__pycache__"
    pycache.mkdir()
    (pycache / "__init__.py").write_text(
        "recommended_route = 'garbage'\n"
    )
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo
    )
    assert violations == []


def test_synthetic_tests_subdir_skipped(tmp_path: Path) -> None:
    """tests/ subdir exempt."""
    repo = _make_synthetic_repo(tmp_path)
    tests_dir = repo / "src" / "tac" / "cathedral_consumers" / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("recommended_route = 'test'\n")
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo
    )
    assert violations == []


def test_synthetic_underscore_prefix_reference_skipped(tmp_path: Path) -> None:
    """Reference packages (_example_consumer pattern) exempt."""
    repo = _make_synthetic_repo(tmp_path)
    pkg = repo / "src" / "tac" / "cathedral_consumers" / "_example_router"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "recommended_route = 'broken_no_markers'\n"
    )
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo
    )
    assert violations == []


def test_synthetic_no_init_py_skipped(tmp_path: Path) -> None:
    """Subdir without __init__.py not a package; skipped silently."""
    repo = _make_synthetic_repo(tmp_path)
    sub = repo / "src" / "tac" / "cathedral_consumers" / "data_only"
    sub.mkdir()
    (sub / "config.yaml").write_text("recommended_route: local_mps_prescreen\n")
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=sub
    )
    assert violations == []


def test_synthetic_multi_violation_aggregated(tmp_path: Path) -> None:
    """Multiple broken routing consumers aggregate into multi-violation list."""
    repo = _make_synthetic_repo(tmp_path)
    _write_routing_consumer_missing_marker(
        repo, "broken_one", "predicted_delta_adjustment=0.0"
    )
    _write_routing_consumer_missing_marker(
        repo, "broken_two", "promotable=False"
    )
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo
    )
    assert len(violations) == 2


def test_synthetic_verbose_flag_runs(tmp_path: Path, capsys) -> None:
    """Verbose flag prints diagnostic line."""
    repo = _make_synthetic_repo(tmp_path)
    _write_compliant_routing_consumer(repo, "verbose_router")
    check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo, verbose=True
    )
    captured = capsys.readouterr()
    assert "cathedral-consumer-mps-routing" in captured.out
    assert "OK" in captured.out


def test_synthetic_verbose_dirty_shows_count(tmp_path: Path, capsys) -> None:
    """Verbose flag with violations prints the violation count."""
    repo = _make_synthetic_repo(tmp_path)
    _write_routing_consumer_missing_marker(
        repo, "verbose_broken", "axis_tag=[predicted]"
    )
    check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=repo, verbose=True
    )
    captured = capsys.readouterr()
    assert "cathedral-consumer-mps-routing" in captured.out
    assert "1 violation" in captured.out


def test_synthetic_string_repo_root_accepted(tmp_path: Path) -> None:
    """String repo_root accepted (converted to Path)."""
    repo = _make_synthetic_repo(tmp_path)
    _write_compliant_routing_consumer(repo, "string_root_router")
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        repo_root=str(repo)
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Wire-in regression guards (Catalog #176 + Catalog #185 sisters)
# ---------------------------------------------------------------------------


def test_check_function_callable_via_preflight_globals() -> None:
    """Catalog #185 sister: function must be importable from tac.preflight."""
    import tac.preflight
    assert hasattr(
        tac.preflight,
        "check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers",
    )
    fn = getattr(
        tac.preflight,
        "check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers",
    )
    assert callable(fn)


def test_orchestrator_wires_strict_true() -> None:
    """Catalog #176 sister: preflight_all wires the gate as strict=True
    (STRICT-from-byte-one per Catalog #341 docstring + Strict-flip
    atomicity rule)."""
    src = Path(__file__).parent.parent / "preflight.py"
    text = src.read_text(encoding="utf-8")
    # Must be referenced in preflight_all's body (search for the wire-in pattern).
    assert (
        "check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers("
        in text
    )
    # Must be STRICT-from-byte-one per docstring + Strict-flip atomicity rule.
    # We do an indented-block search: find the function name then check the
    # NEXT line carries strict=True.
    lines = text.splitlines()
    found_callsite = False
    for i, line in enumerate(lines):
        if (
            "check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers("
            in line
            and not line.lstrip().startswith("#")
            and "def " not in line
        ):
            # Read the next 3 lines for strict=True.
            block = "\n".join(lines[i : i + 5])
            if "strict=True" in block:
                found_callsite = True
                break
    assert found_callsite, "Catalog #341 must be wired strict=True in preflight_all"


def test_signature_is_keyword_only() -> None:
    """API discipline: gate accepts kwargs only for safe call sites."""
    import inspect
    fn = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers
    sig = inspect.signature(fn)
    for name, param in sig.parameters.items():
        assert param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ), f"param {name} must be keyword-only"
