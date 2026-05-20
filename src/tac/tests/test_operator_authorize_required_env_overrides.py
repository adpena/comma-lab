# SPDX-License-Identifier: MIT
"""Regression coverage for required operator-authorize env overrides."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import operator_authorize as oa  # noqa: E402


def _recipe(raw_env: dict[str, str]) -> oa.Recipe:
    return oa.Recipe(name="test", path=Path("test.yaml"), raw={"env_overrides": raw_env})


def test_required_env_override_missing_refuses_before_provider_dispatch(monkeypatch) -> None:
    monkeypatch.delenv("VQ_VAE_CODEBOOK_SIZE", raising=False)

    recipe = _recipe(
        {
            "VQ_VAE_DISPATCH_INSTANCE_JOB_ID": "${INSTANCE_JOB_ID}",
            "VQ_VAE_CODEBOOK_SIZE": "${VQ_VAE_CODEBOOK_SIZE}",
        }
    )

    with pytest.raises(SystemExit, match="requires explicit environment variable"):
        oa._build_env_overrides(recipe, "job-123")


def test_required_env_override_uses_explicit_value(monkeypatch) -> None:
    monkeypatch.setenv("VQ_VAE_CODEBOOK_SIZE", "2")

    recipe = _recipe(
        {
            "VQ_VAE_DISPATCH_INSTANCE_JOB_ID": "${INSTANCE_JOB_ID}",
            "VQ_VAE_CODEBOOK_SIZE": "${VQ_VAE_CODEBOOK_SIZE}",
        }
    )

    out = oa._build_env_overrides(recipe, "job-123")
    assert "VQ_VAE_DISPATCH_INSTANCE_JOB_ID=job-123" in out
    assert "VQ_VAE_CODEBOOK_SIZE=2" in out
