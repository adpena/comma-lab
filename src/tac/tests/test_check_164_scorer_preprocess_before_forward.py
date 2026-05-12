"""Tests for Catalog #164 — substrate score-aware loss must call
``<scorer>.preprocess_input(...)`` before ``<scorer>(...)`` forward.

Bug class: WWW4 dispatch (Modal A100 fc-01KREXK209TRX7ED5ZRVXHY1VT,
2026-05-12) crashed at SegNet because ``score_aware_loss.py`` passed
5D directly to ``self.seg_scorer(...)`` (smp.Unet expects 4D).
Sibling latent bug fed 4D 6-channel to PoseNet's forward (which expects
4D 12-channel after yuv6).

These tests pin: positive (catches bare forward), negative (allows
correct sequence), waiver-respect, multi-scorer body, edge cases.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_score_aware_loss_calls_preprocess_input_before_scorer,
)


def _make_repo_with_substrate(
    substrate_name: str,
    file_name: str,
    content: str,
    tmp_path: Path,
) -> Path:
    """Create a fake repo with src/tac/substrates/<name>/<file>."""
    substrate_dir = tmp_path / "src" / "tac" / "substrates" / substrate_name
    substrate_dir.mkdir(parents=True, exist_ok=True)
    (substrate_dir / file_name).write_text(dedent(content))
    return tmp_path


def test_bare_scorer_forward_is_caught(tmp_path):
    """Bare ``self.seg_scorer(x)`` without preprocess_input is a violation."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "score_aware_loss.py",
        """
        import torch

        class Loss(torch.nn.Module):
            def __init__(self, seg_scorer):
                super().__init__()
                self.seg_scorer = seg_scorer

            def forward(self, x):
                return self.seg_scorer(x)
        """,
        tmp_path,
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=False, verbose=False
        )
    )
    assert len(vlist) == 1
    assert "seg_scorer" in vlist[0]
    assert "preprocess_input" in vlist[0]


def test_preprocess_then_forward_is_accepted(tmp_path):
    """``self.seg_scorer.preprocess_input(...)`` then ``self.seg_scorer(...)`` is OK."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "score_aware_loss.py",
        """
        import torch

        class Loss(torch.nn.Module):
            def __init__(self, seg_scorer):
                super().__init__()
                self.seg_scorer = seg_scorer

            def forward(self, x):
                y = self.seg_scorer.preprocess_input(x)
                return self.seg_scorer(y)
        """,
        tmp_path,
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=False, verbose=False
        )
    )
    assert vlist == []


def test_pose_scorer_path_also_checked(tmp_path):
    """The check covers ``pose_scorer`` in addition to ``seg_scorer``."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "score_aware_loss.py",
        """
        import torch

        class Loss(torch.nn.Module):
            def __init__(self, pose_scorer):
                super().__init__()
                self.pose_scorer = pose_scorer

            def forward(self, x):
                return self.pose_scorer(x)
        """,
        tmp_path,
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=False, verbose=False
        )
    )
    assert len(vlist) == 1
    assert "pose_scorer" in vlist[0]


def test_multiple_scorers_in_body_each_checked(tmp_path):
    """Two scorers in same body — each needs its own preprocess_input."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "score_aware_loss.py",
        """
        import torch

        class Loss(torch.nn.Module):
            def __init__(self, seg_scorer, pose_scorer):
                super().__init__()
                self.seg_scorer = seg_scorer
                self.pose_scorer = pose_scorer

            def forward(self, x):
                # Only pose has preprocess; seg is missing
                p = self.pose_scorer.preprocess_input(x)
                return self.seg_scorer(x) + self.pose_scorer(p)
        """,
        tmp_path,
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=False, verbose=False
        )
    )
    assert len(vlist) == 1
    assert "seg_scorer" in vlist[0]


def test_both_scorers_with_preprocess_is_accepted(tmp_path):
    """Both scorers preprocessed → no violation."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "score_aware_loss.py",
        """
        import torch

        class Loss(torch.nn.Module):
            def __init__(self, seg_scorer, pose_scorer):
                super().__init__()
                self.seg_scorer = seg_scorer
                self.pose_scorer = pose_scorer

            def forward(self, x):
                s = self.seg_scorer.preprocess_input(x)
                p = self.pose_scorer.preprocess_input(x)
                return self.seg_scorer(s) + self.pose_scorer(p)
        """,
        tmp_path,
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=False, verbose=False
        )
    )
    assert vlist == []


def test_same_line_waiver_is_respected(tmp_path):
    """``# SCORER_PREPROCESS_HANDLED_OK:<reason>`` waiver bypasses the check."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "score_aware_loss.py",
        """
        import torch

        class Loss(torch.nn.Module):
            def __init__(self, seg_scorer):
                super().__init__()
                self.seg_scorer = seg_scorer

            def forward(self, x):
                # Already 4D from caller — direct forward is intentional
                return self.seg_scorer(x)  # SCORER_PREPROCESS_HANDLED_OK:caller handles preprocess
        """,
        tmp_path,
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=False, verbose=False
        )
    )
    assert vlist == []


def test_strict_mode_raises_on_violation(tmp_path):
    """``strict=True`` must raise PreflightError on any violation."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "score_aware_loss.py",
        """
        import torch

        class Loss(torch.nn.Module):
            def __init__(self, seg_scorer):
                super().__init__()
                self.seg_scorer = seg_scorer

            def forward(self, x):
                return self.seg_scorer(x)
        """,
        tmp_path,
    )
    with pytest.raises(PreflightError):
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=True, verbose=False
        )


def test_multiple_function_bodies_each_checked(tmp_path):
    """A class with multiple methods — each method's body is its own scope."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "score_aware_loss.py",
        """
        import torch

        class Loss(torch.nn.Module):
            def __init__(self, seg_scorer):
                super().__init__()
                self.seg_scorer = seg_scorer

            def correct_method(self, x):
                y = self.seg_scorer.preprocess_input(x)
                return self.seg_scorer(y)

            def buggy_method(self, x):
                return self.seg_scorer(x)
        """,
        tmp_path,
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=False, verbose=False
        )
    )
    assert len(vlist) == 1
    assert "buggy_method" in vlist[0]


def test_multiple_forward_calls_after_preprocess_all_accepted(tmp_path):
    """Once preprocess is called once, multiple forward calls all OK."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "score_aware_loss.py",
        """
        import torch

        class Loss(torch.nn.Module):
            def __init__(self, seg_scorer):
                super().__init__()
                self.seg_scorer = seg_scorer

            def forward(self, x_pred, x_gt):
                # Preprocess both inputs through the same scorer
                s_pred = self.seg_scorer.preprocess_input(x_pred)
                s_gt = self.seg_scorer.preprocess_input(x_gt)
                # Now we can forward repeatedly
                return self.seg_scorer(s_pred) - self.seg_scorer(s_gt)
        """,
        tmp_path,
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=False, verbose=False
        )
    )
    assert vlist == []


def test_no_substrate_dir_returns_empty(tmp_path):
    """Empty repo returns 0 violations, no exception."""
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=tmp_path, strict=True, verbose=False
        )
    )
    assert vlist == []


def test_unrelated_file_not_scanned(tmp_path):
    """A non-``score_aware*.py`` file in the substrate dir is not scanned."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "architecture.py",  # not score_aware_*
        """
        import torch

        class Renderer(torch.nn.Module):
            def __init__(self, seg_scorer):
                super().__init__()
                self.seg_scorer = seg_scorer

            def forward(self, x):
                return self.seg_scorer(x)
        """,
        tmp_path,
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=False, verbose=False
        )
    )
    assert vlist == []


def test_score_aware_loss_naming_variants(tmp_path):
    """``score_aware_loss.py`` AND ``score_aware_distillation.py`` covered."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "score_aware_distillation.py",
        """
        import torch

        class Distill(torch.nn.Module):
            def __init__(self, seg_scorer):
                super().__init__()
                self.seg_scorer = seg_scorer

            def forward(self, x):
                return self.seg_scorer(x)
        """,
        tmp_path,
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=False, verbose=False
        )
    )
    assert len(vlist) == 1


def test_attr_method_call_not_treated_as_forward(tmp_path):
    """``self.seg_scorer.some_method(x)`` is NOT a bare forward call."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "score_aware_loss.py",
        """
        import torch

        class Loss(torch.nn.Module):
            def __init__(self, seg_scorer):
                super().__init__()
                self.seg_scorer = seg_scorer

            def forward(self, x):
                # Calling a non-forward method — not a forward pass
                return self.seg_scorer.compute_distortion(x, x)
        """,
        tmp_path,
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=False, verbose=False
        )
    )
    # compute_distortion is not preprocess_input AND not a bare forward; no violation
    assert vlist == []


def test_sane_hnerv_substrate_is_clean():
    """Sane_hnerv (FIX-H Part 1 anchor) must be at 0 violations.

    The META gate runs WARN-ONLY in preflight_all because sister substrates
    (balle_renderer, siren, tc_nerv, vq_vae, wavelet, etc.) share the same
    WWW4-class bug; their cleanup is an operator-routable follow-up lane.

    Sane_hnerv itself MUST stay clean — the FIX-H Part 1 fix is the
    self-protection anchor for this lane.
    """
    repo_root = Path(__file__).resolve().parents[3]
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo_root, strict=False, verbose=False
        )
    )
    sane_hnerv_violations = [v for v in vlist if "sane_hnerv" in v]
    assert sane_hnerv_violations == [], (
        "sane_hnerv substrate has reintroduced the WWW4 bug class:\n  "
        + "\n  ".join(sane_hnerv_violations)
    )


def test_multiple_violations_across_substrates(tmp_path):
    """Two different substrates both violating produce 2 violations."""
    substrate_dir_a = tmp_path / "src" / "tac" / "substrates" / "a"
    substrate_dir_a.mkdir(parents=True, exist_ok=True)
    (substrate_dir_a / "score_aware_loss.py").write_text(
        dedent(
            """
            import torch

            class L(torch.nn.Module):
                def __init__(self, seg_scorer):
                    super().__init__()
                    self.seg_scorer = seg_scorer

                def forward(self, x):
                    return self.seg_scorer(x)
            """
        )
    )
    substrate_dir_b = tmp_path / "src" / "tac" / "substrates" / "b"
    substrate_dir_b.mkdir(parents=True, exist_ok=True)
    (substrate_dir_b / "score_aware_loss.py").write_text(
        dedent(
            """
            import torch

            class L(torch.nn.Module):
                def __init__(self, pose_scorer):
                    super().__init__()
                    self.pose_scorer = pose_scorer

                def forward(self, x):
                    return self.pose_scorer(x)
            """
        )
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert len(vlist) == 2


def test_segnet_attr_alias_also_caught(tmp_path):
    """``self.segnet`` (alternate attr name) is also covered."""
    repo = _make_repo_with_substrate(
        "test_substrate",
        "score_aware_loss.py",
        """
        import torch

        class Loss(torch.nn.Module):
            def __init__(self, segnet):
                super().__init__()
                self.segnet = segnet

            def forward(self, x):
                return self.segnet(x)
        """,
        tmp_path,
    )
    vlist = (
        check_substrate_score_aware_loss_calls_preprocess_input_before_scorer(
            repo_root=repo, strict=False, verbose=False
        )
    )
    assert len(vlist) == 1
    assert "segnet" in vlist[0]
