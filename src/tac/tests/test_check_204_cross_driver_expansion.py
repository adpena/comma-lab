"""Catalog #204 cross-driver expansion tests.

Sister of `test_check_204_pr95plus_modal_durable_output.py`. This module covers
the cross-driver scope-extension landed 2026-05-19 per
`lane_catalog_204_cross_driver_expansion_20260519`.

The original gate scoped only to
`scripts/remote_lane_substrate_pr101_lc_v2_clone_enhanced_curriculum.sh`.
The cross-driver scan now refuses any `scripts/remote_lane_substrate_*.sh`
that lacks the canonical 3-branch Modal-aware OUTPUT_DIR resolution.
Sister anchor: `stack_of_stacks` driver fix (commit `956ad2e76`) +
`stc_v2` driver fix 2026-05-14 + the original PR95++ anchor.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _CHECK_204_CROSS_DRIVER_THREE_BRANCH_RE,
    _CHECK_204_CROSS_DRIVER_WAIVER_RE,
    _CHECK_204_REMOTE_LANE_PATH,
    _CHECK_204_TRAINER_PATH,
    _CHECK_204_MODAL_DISPATCHER_PATH,
    check_pr95plus_modal_smoke_uses_durable_provider_output,
)

# ---------------------------------------------------------------------------
# Canonical fixture: synthesize a tiny scaffold repo with the gate's required
# anchors (PR95++ driver/trainer + Modal dispatcher) plus a sweep of substrate
# drivers under `scripts/` so the cross-driver scan has live targets.
# ---------------------------------------------------------------------------

CANONICAL_THREE_BRANCH = """\
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_results}"
# Catalog #204 cross-driver expansion:
if [ -n "${MY_LANE_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$MY_LANE_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
"""

NON_COMPLIANT_DEFAULT = 'OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"\n'

PR95_DRIVER_CONTENT = """\
#!/bin/bash
set -euo pipefail
LOG_DIR="${LOG_DIR:-$WORKSPACE/results}"
if [ -n "${PR95PLUS_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$PR95PLUS_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
"""

PR95_TRAINER_CONTENT = """# trainer\nimport os\n"""

MODAL_DISPATCHER_CONTENT = """\
# experiments/modal_train_lane.py
def scan_roots(volume_dir, **kw):
    pass
"""


def _make_minimal_repo(root: Path) -> None:
    """Lay down the minimum file set so the gate's anchor checks pass."""
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "experiments").mkdir(parents=True, exist_ok=True)
    (root / _CHECK_204_REMOTE_LANE_PATH).write_text(PR95_DRIVER_CONTENT)
    (root / _CHECK_204_TRAINER_PATH).write_text(PR95_TRAINER_CONTENT)
    (root / _CHECK_204_MODAL_DISPATCHER_PATH).write_text(MODAL_DISPATCHER_CONTENT)


# ---------------------------------------------------------------------------
# Helper regex unit tests
# ---------------------------------------------------------------------------


def test_three_branch_re_matches_canonical_shape() -> None:
    assert _CHECK_204_CROSS_DRIVER_THREE_BRANCH_RE.search(CANONICAL_THREE_BRANCH)


def test_three_branch_re_rejects_non_compliant_default() -> None:
    text = "LOG_DIR=foo\n" + NON_COMPLIANT_DEFAULT
    assert not _CHECK_204_CROSS_DRIVER_THREE_BRANCH_RE.search(text)


def test_three_branch_re_accepts_modal_results_d_check_optional() -> None:
    # Either with or without the `&& [ -d "/modal_results" ]` clause.
    no_dcheck = """\
elif [ "${MODAL_RUNTIME:-0}" = "1" ] ; then
    OUTPUT_DIR="/modal_results/${JOB}/output"
"""
    assert _CHECK_204_CROSS_DRIVER_THREE_BRANCH_RE.search(no_dcheck)


def test_waiver_re_extracts_rationale() -> None:
    m = _CHECK_204_CROSS_DRIVER_WAIVER_RE.search(
        '# CATALOG_204_CROSS_DRIVER_WAIVED: scaffold-only no-modal-dispatch'
    )
    assert m is not None
    assert "scaffold-only" in m.group(1)


def test_waiver_re_rejects_bare_marker() -> None:
    # Empty rationale should be rejected by the gate's runtime check.
    assert not _CHECK_204_CROSS_DRIVER_WAIVER_RE.search('# CATALOG_204_CROSS_DRIVER_WAIVED:')


# ---------------------------------------------------------------------------
# End-to-end gate behaviour
# ---------------------------------------------------------------------------


def test_empty_substrate_canvas_is_clean(tmp_path: Path) -> None:
    _make_minimal_repo(tmp_path)
    v = check_pr95plus_modal_smoke_uses_durable_provider_output(repo_root=tmp_path)
    assert v == []


def test_compliant_substrate_driver_passes(tmp_path: Path) -> None:
    _make_minimal_repo(tmp_path)
    drv = tmp_path / "scripts" / "remote_lane_substrate_demo.sh"
    drv.write_text("#!/bin/bash\n" + CANONICAL_THREE_BRANCH)
    v = check_pr95plus_modal_smoke_uses_durable_provider_output(repo_root=tmp_path)
    assert v == []


def test_non_compliant_substrate_driver_flagged(tmp_path: Path) -> None:
    _make_minimal_repo(tmp_path)
    drv = tmp_path / "scripts" / "remote_lane_substrate_demo.sh"
    drv.write_text("#!/bin/bash\n" + NON_COMPLIANT_DEFAULT)
    v = check_pr95plus_modal_smoke_uses_durable_provider_output(repo_root=tmp_path)
    assert any("scripts/remote_lane_substrate_demo.sh" in x for x in v)
    assert any("canonical 3-branch" in x for x in v)


def test_substrate_driver_without_output_dir_is_skipped(tmp_path: Path) -> None:
    _make_minimal_repo(tmp_path)
    drv = tmp_path / "scripts" / "remote_lane_substrate_scaffold.sh"
    drv.write_text("#!/bin/bash\necho 'scaffold only'\n")
    v = check_pr95plus_modal_smoke_uses_durable_provider_output(repo_root=tmp_path)
    assert v == []


def test_substrate_driver_same_line_waiver_accepted(tmp_path: Path) -> None:
    _make_minimal_repo(tmp_path)
    drv = tmp_path / "scripts" / "remote_lane_substrate_local_only.sh"
    drv.write_text(
        '#!/bin/bash\nOUTPUT_DIR="$LOG_DIR/output"  # CATALOG_204_CROSS_DRIVER_WAIVED: local-only diagnostic\n'
    )
    v = check_pr95plus_modal_smoke_uses_durable_provider_output(repo_root=tmp_path)
    assert v == []


def test_substrate_driver_placeholder_waiver_rejected(tmp_path: Path) -> None:
    _make_minimal_repo(tmp_path)
    drv = tmp_path / "scripts" / "remote_lane_substrate_bad_waiver.sh"
    drv.write_text(
        '#!/bin/bash\nOUTPUT_DIR="$LOG_DIR/output"  # CATALOG_204_CROSS_DRIVER_WAIVED: <rationale>\n'
    )
    v = check_pr95plus_modal_smoke_uses_durable_provider_output(repo_root=tmp_path)
    assert any("remote_lane_substrate_bad_waiver.sh" in x for x in v)


def test_substrate_driver_empty_waiver_rejected(tmp_path: Path) -> None:
    _make_minimal_repo(tmp_path)
    drv = tmp_path / "scripts" / "remote_lane_substrate_empty_waiver.sh"
    drv.write_text(
        '#!/bin/bash\nOUTPUT_DIR="$LOG_DIR/output"  # CATALOG_204_CROSS_DRIVER_WAIVED:\n'
    )
    v = check_pr95plus_modal_smoke_uses_durable_provider_output(repo_root=tmp_path)
    assert any("remote_lane_substrate_empty_waiver.sh" in x for x in v)


def test_multi_violation_aggregation(tmp_path: Path) -> None:
    _make_minimal_repo(tmp_path)
    for name in ("alpha", "beta", "gamma"):
        drv = tmp_path / "scripts" / f"remote_lane_substrate_{name}.sh"
        drv.write_text("#!/bin/bash\n" + NON_COMPLIANT_DEFAULT)
    v = check_pr95plus_modal_smoke_uses_durable_provider_output(repo_root=tmp_path)
    assert sum(1 for x in v if "alpha" in x or "beta" in x or "gamma" in x) == 3


def test_strict_mode_raises_with_catalog_204_message(tmp_path: Path) -> None:
    _make_minimal_repo(tmp_path)
    drv = tmp_path / "scripts" / "remote_lane_substrate_demo.sh"
    drv.write_text("#!/bin/bash\n" + NON_COMPLIANT_DEFAULT)
    with pytest.raises(PreflightError) as exc_info:
        check_pr95plus_modal_smoke_uses_durable_provider_output(
            repo_root=tmp_path, strict=True
        )
    msg = str(exc_info.value)
    assert "check_pr95plus_modal_smoke_uses_durable_provider_output" in msg


def test_strict_silent_on_clean(tmp_path: Path) -> None:
    _make_minimal_repo(tmp_path)
    # No exception.
    v = check_pr95plus_modal_smoke_uses_durable_provider_output(
        repo_root=tmp_path, strict=True
    )
    assert v == []


def test_compliant_and_non_compliant_mixed(tmp_path: Path) -> None:
    _make_minimal_repo(tmp_path)
    good = tmp_path / "scripts" / "remote_lane_substrate_good.sh"
    bad = tmp_path / "scripts" / "remote_lane_substrate_bad.sh"
    good.write_text("#!/bin/bash\n" + CANONICAL_THREE_BRANCH)
    bad.write_text("#!/bin/bash\n" + NON_COMPLIANT_DEFAULT)
    v = check_pr95plus_modal_smoke_uses_durable_provider_output(repo_root=tmp_path)
    assert any("remote_lane_substrate_bad.sh" in x for x in v)
    assert not any("remote_lane_substrate_good.sh" in x for x in v)


def test_substrate_driver_with_default_output_dir_variant_rejected(tmp_path: Path) -> None:
    """DEFAULT_OUTPUT_DIR variant (pre-fix Z6/Z7/Z3/Z4/Z5 shape) does NOT match the
    canonical 3-branch shape and should be flagged for refactor."""
    _make_minimal_repo(tmp_path)
    drv = tmp_path / "scripts" / "remote_lane_substrate_default_output_dir_variant.sh"
    drv.write_text(
        """#!/bin/bash
if [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -n "${DISPATCH_INSTANCE_JOB_ID:-}" ]; then
    DEFAULT_OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    DEFAULT_OUTPUT_DIR="$WORKSPACE/local_results/output"
fi
LOG_DIR="${LOG_DIR:-$WORKSPACE/local_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
"""
    )
    v = check_pr95plus_modal_smoke_uses_durable_provider_output(repo_root=tmp_path)
    assert any("default_output_dir_variant" in x for x in v)


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_live_repo_cross_driver_clean() -> None:
    """Post-Phase-2 commit `b43c8f2fd` all 50 substrate drivers must pass
    the cross-driver scan. This guard protects against regressions where a
    sister subagent introduces a new driver without the canonical pattern."""
    v = check_pr95plus_modal_smoke_uses_durable_provider_output()
    cross_driver_violations = [
        x for x in v if "scripts/remote_lane_substrate_" in x
    ]
    assert cross_driver_violations == [], (
        f"Live repo regressed: {len(cross_driver_violations)} cross-driver "
        "violations:\n  " + "\n  ".join(cross_driver_violations[:5])
    )


def test_live_repo_total_zero() -> None:
    """The entire Catalog #204 contract (PR95++ anchors + cross-driver scan)
    must be clean at landing per the Strict-flip atomicity rule."""
    v = check_pr95plus_modal_smoke_uses_durable_provider_output()
    assert v == [], (
        f"Live repo has {len(v)} Catalog #204 violations:\n  "
        + "\n  ".join(v[:5])
    )


def test_orchestrator_callsite_remains_strict_true() -> None:
    """preflight_all() must keep the Catalog #204 wire-in strict=True after
    the cross-driver expansion (no regression to warn-only)."""
    src = Path("src/tac/preflight.py").read_text(encoding="utf-8")
    # Find the orchestrator callsite and confirm strict=True remains.
    callsite_marker = "check_pr95plus_modal_smoke_uses_durable_provider_output("
    idx = src.find(callsite_marker, src.find("def preflight_all"))
    assert idx != -1, "orchestrator callsite missing"
    nearby = src[idx:idx + 200]
    assert "strict=True" in nearby
