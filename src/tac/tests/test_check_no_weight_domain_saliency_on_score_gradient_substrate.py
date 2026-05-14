# SPDX-License-Identifier: MIT
"""Tests for preflight Catalog #123:
``check_no_weight_domain_saliency_on_score_gradient_substrate``.

The check guards against the Track 4 v1 bug class: applying
``mean(theta^2)`` / ``var(theta)`` / ``norm(theta)`` etc. as a Fisher
saliency proxy on score-gradient-trained substrates (A1, etc.). On those
substrates the proxy is ANTI-correlated with true score saliency and the
pipeline catastrophically regresses.

This test set verifies:
  1. Positive detection: builder using mean(theta^2) on A1 substrate
     without an opt-in to score-gradient saliency raises a violation.
  2. Negative detection (clean): builder that does not touch
     score-aware substrates is silent.
  3. Same-line `# WEIGHT_SALIENCY_OK_ON_SCORE_AWARE:<reason>` waiver
     suppresses the violation.
  4. `--saliency-source` opt-in suppresses the violation (file-level).
  5. Multi-substrate file: multiple substrate tokens still produce
     ONE violation per offending line.
  6. Function-context filter: the SAME `.pow(2).mean()` regex inside a
     forward-pass loss function is NOT a violation (Track 4-class regression
     specifically targets weight-domain saliency, not score-axis MSE).
  7. Live codebase invariant: the check is at zero violations on the
     real tree.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_weight_domain_saliency_on_score_gradient_substrate,
)


def _write_under_repo(repo: Path, rel: str, body: str) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip())
    return p


def test_positive_detection_a1_substrate_with_mean_theta_squared(tmp_path: Path) -> None:
    repo = tmp_path / "fake_repo"
    (repo / "tools").mkdir(parents=True)
    _write_under_repo(repo, "tools/build_offending_a1_v2.py", """
        # Hypothetical Track-4-clone that doesn't offer score-gradient
        from pathlib import Path
        A1_ARCHIVE = "experiments/results/track1_phase_a1_score_gradient_..."

        def compute_fisher_proxy(state_dict):
            proxy = {}
            for name, t in state_dict.items():
                proxy[name] = float((t * t).mean().item())
            return proxy
    """)
    violations = check_no_weight_domain_saliency_on_score_gradient_substrate(
        repo_root=repo, strict=False, verbose=False,
    )
    assert len(violations) >= 1
    assert "build_offending_a1_v2.py" in violations[0]
    assert "compute_fisher_proxy" in violations[0]


def test_negative_detection_clean_builder_no_substrate_token(tmp_path: Path) -> None:
    """A builder that never references score-aware substrates is silent."""
    repo = tmp_path / "fake_repo"
    (repo / "tools").mkdir(parents=True)
    _write_under_repo(repo, "tools/build_unrelated_codec.py", """
        # Generic codec for unrelated payload.
        def compute_fisher_proxy(state_dict):
            return {n: float((t * t).mean().item()) for n, t in state_dict.items()}

        def main():
            print("hello")
    """)
    violations = check_no_weight_domain_saliency_on_score_gradient_substrate(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_same_line_waiver_suppresses_violation(tmp_path: Path) -> None:
    repo = tmp_path / "fake_repo"
    (repo / "tools").mkdir(parents=True)
    _write_under_repo(repo, "tools/build_waived_a1.py", """
        A1_ARCHIVE = "..."

        def compute_fisher_proxy(state_dict):
            return {n: float((t * t).mean().item()) for n, t in state_dict.items()}  # WEIGHT_SALIENCY_OK_ON_SCORE_AWARE: legacy diagnostic-only path
    """)
    violations = check_no_weight_domain_saliency_on_score_gradient_substrate(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_score_gradient_opt_in_flag_suppresses_violation(tmp_path: Path) -> None:
    """File offering --saliency-source score_gradient is allowed to keep the
    weight-domain proxy as the backward-compat default."""
    repo = tmp_path / "fake_repo"
    (repo / "tools").mkdir(parents=True)
    _write_under_repo(repo, "tools/build_optin_a1.py", """
        # File offers --saliency-source score_gradient AND uses score_gradient_param_saliency
        A1_ARCHIVE = "..."
        from tac.score_gradient_param_saliency import compute_score_gradient_param_saliency

        def compute_fisher_proxy(state_dict):
            # Backward-compat default; operator can choose score_gradient via --saliency-source
            return {n: float((t * t).mean().item()) for n, t in state_dict.items()}

        def main():
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("--saliency-source", choices=("mean_theta_squared", "score_gradient"), default="mean_theta_squared")
    """)
    violations = check_no_weight_domain_saliency_on_score_gradient_substrate(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_multi_substrate_file_multiple_lines_each_count_once(tmp_path: Path) -> None:
    repo = tmp_path / "fake_repo"
    (repo / "tools").mkdir(parents=True)
    _write_under_repo(repo, "tools/build_multi_substrate.py", """
        track1_phase_a1_score_gradient = "yes"
        a1_latent_aligned = "also yes"

        def compute_fisher_proxy(sd):
            return {n: float((t * t).mean().item()) for n, t in sd.items()}

        def hessian_diag(sd):
            return {n: float(t.pow(2).mean().item()) for n, t in sd.items()}
    """)
    violations = check_no_weight_domain_saliency_on_score_gradient_substrate(
        repo_root=repo, strict=False, verbose=False,
    )
    # Expect 2 violations (one per offending function); each function fires
    # at most once per line.
    assert len(violations) == 2
    assert any("compute_fisher_proxy" in v for v in violations)
    assert any("hessian_diag" in v for v in violations)


def test_function_context_filter_forward_pass_mse_not_flagged(tmp_path: Path) -> None:
    """The SAME `.pow(2).mean()` regex inside a forward-pass loss function
    (NOT a saliency function) is NOT a violation. This is the false-positive
    surfaced by the v1 of the check that flagged
    `experiments/build_sjkl_residual.py:307`.
    """
    repo = tmp_path / "fake_repo"
    (repo / "experiments").mkdir(parents=True)
    _write_under_repo(repo, "experiments/build_score_aware_loss.py", """
        a1_archive = "fake"  # substrate token present

        def score_fn(pair, target_pose, posenet):
            # Forward-pass MSE on PoseNet output; NOT a weight saliency proxy.
            pose_out = posenet(posenet.preprocess_input(pair))["pose"][..., :6]
            return (pose_out - target_pose).pow(2).mean()
    """)
    violations = check_no_weight_domain_saliency_on_score_gradient_substrate(
        repo_root=repo, strict=False, verbose=False,
    )
    # No violation: `score_fn` does not match any saliency function-name token.
    assert violations == []


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    repo = tmp_path / "fake_repo"
    (repo / "tools").mkdir(parents=True)
    _write_under_repo(repo, "tools/build_offender.py", """
        A1_ARCHIVE = "..."
        def compute_fisher_proxy(sd):
            return {n: float((t * t).mean().item()) for n, t in sd.items()}
    """)
    with pytest.raises(PreflightError) as exc_info:
        check_no_weight_domain_saliency_on_score_gradient_substrate(
            repo_root=repo, strict=True, verbose=False,
        )
    assert "weight-domain saliency" in str(exc_info.value).lower()


def test_check_is_zero_on_real_codebase() -> None:
    """STRICT-flip baseline: live codebase MUST have 0 violations.

    This is the gate that lets us flip the check to STRICT in
    ``preflight_all()``. If this test fails, the codebase has a Track 4 v1
    class regression that needs to be fixed before STRICT promotion.
    """
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_no_weight_domain_saliency_on_score_gradient_substrate(
        repo_root=repo_root, strict=False, verbose=False,
    )
    assert violations == [], (
        f"Live codebase has {len(violations)} weight-saliency-on-score-aware "
        f"violation(s); fix before STRICT promotion. First 3:\n  "
        + "\n  ".join(violations[:3])
    )
