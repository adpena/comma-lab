"""Tests for the Track 4 weight-domain saliency preflight guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_weight_domain_saliency_on_score_gradient_substrate,
)


def _write_tool(root: Path, name: str, text: str) -> Path:
    tools = root / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    path = tools / name
    path.write_text(text, encoding="utf-8")
    return path


def test_weight_domain_saliency_on_score_gradient_substrate_blocks(tmp_path: Path) -> None:
    _write_tool(
        tmp_path,
        "build_bad.py",
        """
def compute_fisher_proxy(state_dict):
    substrate = "track1_phase_a1_score_gradient"
    return {k: v.pow(2).mean().item() for k, v in state_dict.items()}
""",
    )

    violations = check_no_weight_domain_saliency_on_score_gradient_substrate(
        repo_root=tmp_path, strict=False, verbose=False
    )

    assert len(violations) == 1
    assert "build_bad.py" in violations[0]
    assert "score_gradient" in violations[0]
    with pytest.raises(PreflightError):
        check_no_weight_domain_saliency_on_score_gradient_substrate(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_weight_domain_saliency_passes_with_score_gradient_opt_in(tmp_path: Path) -> None:
    _write_tool(
        tmp_path,
        "build_good.py",
        """
def compute_fisher_proxy(state_dict):
    substrate = "track1_phase_a1_score_gradient"
    parser_help = "--saliency-source"
    return {k: v.pow(2).mean().item() for k, v in state_dict.items()}
""",
    )

    violations = check_no_weight_domain_saliency_on_score_gradient_substrate(
        repo_root=tmp_path, strict=True, verbose=False
    )

    assert violations == []


def test_weight_domain_saliency_same_line_waiver_passes(tmp_path: Path) -> None:
    _write_tool(
        tmp_path,
        "build_waived.py",
        """
def compute_fisher_proxy(state_dict):
    substrate = "track1_phase_a1_score_gradient"
    return {k: v.pow(2).mean().item() for k, v in state_dict.items()}  # WEIGHT_SALIENCY_OK_ON_SCORE_AWARE:test fixture
""",
    )

    violations = check_no_weight_domain_saliency_on_score_gradient_substrate(
        repo_root=tmp_path, strict=True, verbose=False
    )

    assert violations == []


def test_forward_loss_pow2_mean_is_not_saliency_context(tmp_path: Path) -> None:
    _write_tool(
        tmp_path,
        "build_loss.py",
        """
def loss_fn(x):
    substrate = "track1_phase_a1_score_gradient"
    return x.pow(2).mean()
""",
    )

    violations = check_no_weight_domain_saliency_on_score_gradient_substrate(
        repo_root=tmp_path, strict=True, verbose=False
    )

    assert violations == []
