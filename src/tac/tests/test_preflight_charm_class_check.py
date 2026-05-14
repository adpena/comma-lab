# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight_charm_class_check import (
    CharmClassCheckError,
    check_charm_class_actually_implements_channel_conditional,
)


def test_charm_context_gate_flags_declared_but_unused_context(tmp_path: Path) -> None:
    bad = tmp_path / "bad_charm.py"
    bad.write_text(
        """
class ChARM2020Coder:
    def __init__(self):
        self.context = object()

    def forward(self, z):
        return z
""",
        encoding="utf-8",
    )

    with pytest.raises(CharmClassCheckError):
        check_charm_class_actually_implements_channel_conditional(
            repo_root=tmp_path,
            scan_paths=[bad],
            strict=True,
        )


def test_charm_context_gate_accepts_actual_context_call(tmp_path: Path) -> None:
    good = tmp_path / "good_charm.py"
    good.write_text(
        """
class ChARM2020Coder:
    def __init__(self):
        self.context = object()

    def forward(self, z):
        return self.context(z)
""",
        encoding="utf-8",
    )

    result = check_charm_class_actually_implements_channel_conditional(
        repo_root=tmp_path,
        scan_paths=[good],
        strict=True,
    )

    assert result["passed"] is True
    assert result["files_scanned"] == 1


def test_charm_context_gate_is_wired_into_normal_preflight() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    preflight_source = (repo_root / "src" / "tac" / "preflight.py").read_text(
        encoding="utf-8"
    )

    assert "check_charm_class_actually_implements_channel_conditional" in preflight_source
    assert "[charm-context-conditioning]" in preflight_source
