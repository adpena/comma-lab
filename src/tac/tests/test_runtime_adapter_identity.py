# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tac.optimization.runtime_adapter_identity import runtime_adapter_identity_blockers
from tac.repo_io import tree_sha256


def _runtime_dir(path: Path) -> Path:
    path.mkdir()
    inflate = path / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    inflate.chmod(0o755)
    return path


def test_runtime_adapter_identity_strict_mode_requires_ready_claim(
    tmp_path: Path,
) -> None:
    runtime = _runtime_dir(tmp_path / "runtime")
    actual_tree_sha = tree_sha256(runtime)
    payload = {
        "runtime_dir": runtime.as_posix(),
        "runtime_tree_sha256": actual_tree_sha,
    }

    assert (
        runtime_adapter_identity_blockers(
            payload,
            repo_root=tmp_path,
            context="unit",
        )
        == []
    )
    assert runtime_adapter_identity_blockers(
        payload,
        repo_root=tmp_path,
        context="unit",
        require_claimed=True,
    ) == ["unit_runtime_adapter_identity_claim_missing"]


def test_runtime_adapter_identity_blocks_expected_runtime_tree_mismatch(
    tmp_path: Path,
) -> None:
    runtime = _runtime_dir(tmp_path / "runtime")
    actual_tree_sha = tree_sha256(runtime)

    blockers = runtime_adapter_identity_blockers(
        {
            "runtime_adapter_ready": True,
            "runtime_dir": runtime.as_posix(),
            "runtime_tree_sha256": actual_tree_sha,
            "expected_runtime_tree_sha256": "b" * 64,
        },
        repo_root=tmp_path,
        context="unit",
    )

    assert "unit_expected_runtime_tree_sha256_mismatch" in blockers
    assert "unit_runtime_tree_sha256_mismatch" not in blockers


def test_runtime_adapter_identity_ready_tree_requires_expected_runtime_identity(
    tmp_path: Path,
) -> None:
    runtime = _runtime_dir(tmp_path / "runtime")
    actual_tree_sha = tree_sha256(runtime)

    blockers = runtime_adapter_identity_blockers(
        {
            "runtime_adapter_ready": True,
            "runtime_dir": runtime.as_posix(),
            "runtime_tree_sha256": actual_tree_sha,
        },
        repo_root=tmp_path,
        context="unit",
    )

    assert blockers == ["unit_expected_runtime_tree_sha256_missing"]


def test_runtime_adapter_identity_accepts_expected_runtime_tree_as_live_identity(
    tmp_path: Path,
) -> None:
    runtime = _runtime_dir(tmp_path / "runtime")
    actual_tree_sha = tree_sha256(runtime)

    blockers = runtime_adapter_identity_blockers(
        {
            "runtime_adapter_ready": True,
            "runtime_dir": runtime.as_posix(),
            "expected_runtime_tree_sha256": actual_tree_sha,
        },
        repo_root=tmp_path,
        context="unit",
    )

    assert blockers == []


def test_runtime_adapter_identity_blocks_nested_expected_runtime_tree_mismatch(
    tmp_path: Path,
) -> None:
    runtime = _runtime_dir(tmp_path / "runtime")
    actual_tree_sha = tree_sha256(runtime)

    blockers = runtime_adapter_identity_blockers(
        {
            "receiver_verification": {
                "runtime_adapter_ready": True,
                "runtime_dir": runtime.as_posix(),
                "runtime_tree_sha256": actual_tree_sha,
                "expected_inflate_runtime_tree_sha256": "c" * 64,
            },
        },
        repo_root=tmp_path,
        context="unit",
    )

    assert "unit_expected_runtime_tree_sha256_mismatch" in blockers
