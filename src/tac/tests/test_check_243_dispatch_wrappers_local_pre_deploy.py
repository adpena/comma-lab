# SPDX-License-Identifier: MIT
"""Tests for Catalog #243 — dispatch wrappers must invoke local pre-deploy check.

Sister of Catalog #152 (required-input-files validation) + Catalog #167
(smoke-before-full pattern). Lands as part of WIRE-AND-INTEGRATE-ALL
2026-05-15 (lane_wire_and_integrate_all_cross_stack_20260515).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_dispatch_wrappers_invoke_local_pre_deploy_check_first,
)


def _make_repo(tmp_path: Path) -> Path:
    """Create a minimal repo skeleton with the canonical scan dirs."""
    for sub in ("tools", "scripts", "experiments", "src/tac"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_check_243_clean_repo_no_violations(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    violations = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_check_243_dispatch_token_without_canonical_routing_flagged(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    wrapper = repo / "scripts" / "launch_evil_dispatch.sh"
    wrapper.write_text(
        "#!/bin/bash\nset -euo pipefail\nmodal run experiments/foo.py\n"
    )
    violations = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(violations) == 1, violations
    assert "launch_evil_dispatch.sh" in violations[0]
    assert "Catalog #243" in violations[0] or "[Check 243]" in violations[0]


def test_check_243_routes_through_operator_authorize_accepted(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    wrapper = repo / "scripts" / "launch_good_dispatch.sh"
    wrapper.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "tools/operator_authorize.py --recipe foo\n"
        "modal run experiments/bar.py\n"
    )
    violations = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_check_243_calls_local_pre_deploy_directly_accepted(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    wrapper = repo / "tools" / "manual_dispatch.py"
    wrapper.write_text(
        "import subprocess\n"
        "subprocess.run(['python', 'tools/local_pre_deploy_check.py', "
        "'--trainer', 'experiments/foo.py', '--strict'], check=True)\n"
        "subprocess.run(['modal', 'run', 'experiments/foo.py'])\n"
    )
    violations = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_check_243_same_line_waiver_accepted(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    wrapper = repo / "scripts" / "intentional_bypass.sh"
    wrapper.write_text(
        "#!/bin/bash\n"
        "modal run experiments/foo.py  # LOCAL_PRE_DEPLOY_CHECK_BYPASS_OK:emergency-recovery\n"
    )
    violations = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_check_243_placeholder_waiver_rejected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    wrapper = repo / "scripts" / "lazy_waiver.sh"
    wrapper.write_text(
        "#!/bin/bash\n"
        "modal run experiments/foo.py  # LOCAL_PRE_DEPLOY_CHECK_BYPASS_OK:<reason>\n"
    )
    violations = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "lazy_waiver.sh" in violations[0]


def test_check_243_self_exempt_canonical_helper(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    canonical = repo / "tools" / "local_pre_deploy_check.py"
    # The canonical helper itself contains dispatch-token mentions in
    # docstrings/error messages but should be self-exempt.
    canonical.write_text(
        "# canonical local pre-deploy harness\n"
        "# example: modal run experiments/foo.py\n"
    )
    violations = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_check_243_self_exempt_operator_authorize(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    canonical = repo / "tools" / "operator_authorize.py"
    # operator_authorize.py contains dispatch tokens in error messages /
    # docstrings AND in dispatch dispatch logic; it's self-exempt because
    # it IS the canonical wrapper.
    canonical.write_text("# operator_authorize\n# modal run example\n")
    violations = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_check_243_test_files_excluded(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    test_file = repo / "src/tac/tests/test_dispatch.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# fixture mentions modal run for synthetic test\n")
    violations = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_check_243_intake_clones_excluded(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    intake = repo / "experiments/results/public_pr_intake/script.sh"
    intake.parent.mkdir(parents=True, exist_ok=True)
    intake.write_text("#!/bin/bash\nmodal run vendored/foo.py\n")
    violations = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_check_243_strict_mode_raises(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    wrapper = repo / "scripts" / "bad_dispatch.sh"
    wrapper.write_text("#!/bin/bash\nmodal run experiments/foo.py\n")
    with pytest.raises(PreflightError) as ei:
        check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
            repo_root=repo, strict=True, verbose=False
        )
    assert "Catalog #243" in str(ei.value)
    assert "WIRE-AND-INTEGRATE-ALL" in str(ei.value)


def test_check_243_strict_silent_on_clean_repo(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    # No violations - strict=True should NOT raise
    result = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        repo_root=repo, strict=True, verbose=False
    )
    assert result == []


def test_check_243_unrelated_files_not_flagged(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    f = repo / "tools" / "compute_metrics.py"
    f.write_text("def compute():\n    return 42\n")
    violations = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_check_243_live_repo_regression_guard() -> None:
    """Live-repo regression guard: violations are bounded.

    The actual count varies as wrappers migrate; this test pins an upper
    bound to detect regression spikes (e.g., a sister subagent landing 100
    new bypass wrappers).
    """
    violations = check_dispatch_wrappers_invoke_local_pre_deploy_check_first(
        strict=False, verbose=False
    )
    # As of WIRE-AND-INTEGRATE-ALL landing 2026-05-15, ~56 historical wrappers
    # exist. Bound at 200 to detect 3x regression.
    assert len(violations) < 200, (
        f"Catalog #243 violation regression: {len(violations)} wrappers bypass "
        f"the local pre-deploy harness (expected <200). Sample:\n"
        + "\n".join(v[:200] for v in violations[:5])
    )


def test_check_243_orchestrator_wires_warn_only() -> None:
    """Regression guard: preflight_all() wires #243 strict=False (initial)."""
    import inspect

    from tac.preflight import preflight_all

    src = inspect.getsource(preflight_all)
    assert "check_dispatch_wrappers_invoke_local_pre_deploy_check_first" in src, (
        "Catalog #243 must be wired into preflight_all()"
    )
    # Find the callsite and verify strict=False (initial)
    idx = src.index("check_dispatch_wrappers_invoke_local_pre_deploy_check_first")
    snippet = src[idx : idx + 200]
    assert "strict=False" in snippet, (
        "Catalog #243 should be warn-only initially per "
        "Strict-flip atomicity rule. Snippet:\n" + snippet
    )
