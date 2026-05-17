# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

import tools.operator_authorize as op


def _local_recipe(tmp_path: Path, *, platform: str) -> op.Recipe:
    driver = tmp_path / f"{platform}_driver.sh"
    driver.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    return op.Recipe(
        name=f"unit_{platform}",
        path=tmp_path / f"unit_{platform}.yaml",
        raw={
            "lane_id": f"lane_unit_{platform}_20260517",
            "platform": platform,
            "remote_driver": str(driver),
        },
    )


def test_local_cpu_manifest_append_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local-CPU subprocess success is not success unless the manifest row lands."""

    import tac.optimization.macos_cpu_advisory_signal as advisory

    recipe = _local_recipe(tmp_path, platform="local_cpu")
    monkeypatch.setattr(op.subprocess, "call", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(advisory, "is_running_on_macos_arm64", lambda: True)

    def fail_append(*_args: object, **_kwargs: object) -> None:
        raise OSError("manifest disk unavailable")

    monkeypatch.setattr(advisory, "append_manifest_row_to_jsonl", fail_append)

    with pytest.raises(SystemExit) as exc:
        op._dispatch_local_cpu(recipe, "unit_job", "")

    assert "failed to write macOS-CPU advisory manifest row" in str(exc.value)
    assert "refusing success to avoid signal loss" in str(exc.value)


def test_local_mps_manifest_append_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local-MPS subprocess success is not success unless the manifest row lands."""

    import tac.optimization.mps_research_signal as mps_signal

    torch = pytest.importorskip("torch")
    recipe = _local_recipe(tmp_path, platform="local_mps")
    monkeypatch.setattr(op.subprocess, "call", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    def fail_append(*_args: object, **_kwargs: object) -> None:
        raise OSError("manifest disk unavailable")

    monkeypatch.setattr(mps_signal, "append_manifest_row_to_jsonl", fail_append)

    with pytest.raises(SystemExit) as exc:
        op._dispatch_local_mps(recipe, "unit_job", "")

    assert "failed to write MPS research-signal manifest row" in str(exc.value)
    assert "refusing success to avoid signal loss" in str(exc.value)


def test_source_does_not_downgrade_local_signal_manifest_failure_to_warning() -> None:
    """Regression guard: local proxy/advisory manifest append is mandatory."""

    text = (op.REPO_ROOT / "tools/operator_authorize.py").read_text(
        encoding="utf-8"
    )
    local_block = text[
        text.index("def _dispatch_local_mps(") : text.index(
            "def _dispatch_noop("
        )
    ]

    assert "refusing success to avoid" in local_block
    assert "signal loss" in local_block
    assert "WARN: failed to write MPS research-signal" not in local_block
    assert "WARN: failed to write macOS-CPU advisory" not in local_block
