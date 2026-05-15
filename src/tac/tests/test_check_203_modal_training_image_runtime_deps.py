# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_modal_training_image_includes_hard_runtime_deps,
)


def _write_repo(tmp_path: Path, modal_body: str, pyproject_deps: str | None = None) -> Path:
    repo = tmp_path
    (repo / "experiments").mkdir(parents=True)
    (repo / "experiments" / "modal_train_lane.py").write_text(modal_body)
    deps = pyproject_deps or """
[project]
dependencies = [
  "brotli>=1.0",
  "constriction>=0.4,<0.5",
  "pyppmd>=1.3,<2.0",
]
"""
    (repo / "pyproject.toml").write_text(deps)
    return repo


def _modal_with(*deps: str) -> str:
    dep_args = ",\n        ".join(repr(dep) for dep in deps)
    return f"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        {dep_args},
    )
    .env({{
        "DALI_DISABLE_NVML": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }})
)
"""


def test_live_repo_modal_training_image_has_hard_runtime_deps() -> None:
    assert check_modal_training_image_includes_hard_runtime_deps(verbose=False) == []


def test_missing_constriction_is_violation(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path, _modal_with("brotli", "pyppmd>=1.3,<2.0"))

    violations = check_modal_training_image_includes_hard_runtime_deps(
        repo_root=repo,
        verbose=False,
    )

    assert len(violations) == 1
    assert "constriction" in violations[0]


def test_missing_pyppmd_is_violation(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path, _modal_with("brotli", "constriction>=0.4,<0.5"))

    violations = check_modal_training_image_includes_hard_runtime_deps(
        repo_root=repo,
        verbose=False,
    )

    assert len(violations) == 1
    assert "pyppmd" in violations[0]


def test_comment_only_dependency_reference_does_not_satisfy_gate(tmp_path: Path) -> None:
    repo = _write_repo(
        tmp_path,
        """
import modal

# constriction>=0.4,<0.5 is mentioned here but not installed.
image = modal.Image.debian_slim().pip_install(
    "brotli",
    "pyppmd>=1.3,<2.0",
).env({
    "DALI_DISABLE_NVML": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})
""",
    )

    violations = check_modal_training_image_includes_hard_runtime_deps(
        repo_root=repo,
        verbose=False,
    )

    assert len(violations) == 1
    assert "constriction" in violations[0]


def test_missing_modal_dali_nvml_env_is_violation(tmp_path: Path) -> None:
    repo = _write_repo(
        tmp_path,
        """
import modal

image = modal.Image.debian_slim().pip_install(
    "brotli",
    "constriction>=0.4,<0.5",
    "pyppmd>=1.3,<2.0",
)
""",
    )

    violations = check_modal_training_image_includes_hard_runtime_deps(
        repo_root=repo,
        verbose=False,
    )

    assert any("DALI_DISABLE_NVML" in violation for violation in violations)
    assert any("PYTORCH_CUDA_ALLOC_CONF" in violation for violation in violations)


def test_pyproject_without_entropy_dep_does_not_force_modal_literal(tmp_path: Path) -> None:
    repo = _write_repo(
        tmp_path,
        _modal_with("brotli"),
        pyproject_deps="""
[project]
dependencies = [
  "brotli>=1.0",
]
""",
    )

    assert check_modal_training_image_includes_hard_runtime_deps(
        repo_root=repo,
        verbose=False,
    ) == []


def test_strict_raises_on_missing_runtime_dep(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path, _modal_with("brotli"))

    with pytest.raises(PreflightError, match="constriction"):
        check_modal_training_image_includes_hard_runtime_deps(
            repo_root=repo,
            strict=True,
            verbose=False,
        )
