from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_cross_module_declared_never_read_paths,
)


def test_cross_module_declared_never_read_warn_only_runs_live_controls() -> None:
    violations = check_cross_module_declared_never_read_paths(
        repo_root=Path(__file__).resolve().parents[3],
        strict=False,
        verbose=False,
    )

    joined = "\n".join(violations)
    assert "DirectDescriptionJointDescentMLXModule.margin_targets" in joined
    assert "DirectDescriptionJointDescentMLXModule.seg_targets" not in joined
    assert "DirectDescriptionJointDescentMLXModule.pose_targets" not in joined
    assert "DirectDescriptionJointDescentMLXModule.margin_hinge_weight" not in joined


def test_cross_module_declared_never_read_strict_raises_on_live_hits() -> None:
    with pytest.raises(PreflightError, match="declared_never_read"):
        check_cross_module_declared_never_read_paths(
            repo_root=Path(__file__).resolve().parents[3],
            strict=True,
            verbose=False,
        )
