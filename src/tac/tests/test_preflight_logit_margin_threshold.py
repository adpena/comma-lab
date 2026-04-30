"""Tests for Check 93: Lane 19 (logit-margin) callers pass explicit threshold=.

Per CLAUDE.md FORBIDDEN PATTERNS / silent-default bug class:
- The Lane 19 module (src/tac/losses_logit_margin.py) raises ValueError when
  threshold is None. STRICT preflight ensures NO caller relies on positional
  defaults — every invocation passes threshold from a profile resolver.

Memory: .omx/research/council_lane_19_logit_margin_design_20260430.md
Memory: feedback_silent_default_bug_class_findings_20260429.md
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.preflight import (  # noqa: E402
    MetaBugViolation,
    check_logit_margin_loss_uses_boundary_mask,
    _scan_lane19_threshold_calls,
)


def _write_synthetic_repo(tmp: Path, py_files: dict[str, str]) -> Path:
    """Build a fake repo skeleton at tmp with the listed python files.

    Files are written into <tmp>/src/tac/ matching the real layout so the
    scanner's _META_PY_SCAN_DIRS = ["src/tac", ...] picks them up.
    """
    tac_dir = tmp / "src" / "tac"
    tac_dir.mkdir(parents=True, exist_ok=True)
    for name, body in py_files.items():
        (tac_dir / name).write_text(textwrap.dedent(body).strip() + "\n")
    return tmp


def test_real_codebase_passes_strict() -> None:
    """[regression] Real repo has 0 Lane 19 callers missing threshold=."""
    v = check_logit_margin_loss_uses_boundary_mask(strict=False, verbose=False)
    assert v == [], f"Real codebase should be clean; got {len(v)} violations: {v}"


def test_strict_real_codebase_does_not_raise() -> None:
    """[regression] strict=True on real codebase passes without raising."""
    # Should not raise.
    check_logit_margin_loss_uses_boundary_mask(strict=True, verbose=False)


def test_violation_caught_when_threshold_missing(tmp_path: Path) -> None:
    """Synthetic file calling logit_margin_loss without threshold= → flagged."""
    repo = _write_synthetic_repo(tmp_path, {
        "_offending.py": """
            from tac.losses_logit_margin import logit_margin_loss
            import torch

            def bad():
                logits = torch.randn(1, 5)
                gt = torch.zeros(1, dtype=torch.long)
                # MISSING threshold= — should be flagged
                return logit_margin_loss(logits, gt)
        """,
    })
    v = check_logit_margin_loss_uses_boundary_mask(
        repo_root=repo, strict=False, verbose=False,
    )
    assert any("logit_margin_loss" in line and "threshold=" in line for line in v), (
        f"Expected violation for missing threshold=; got {v}"
    )


def test_explicit_threshold_kwarg_passes(tmp_path: Path) -> None:
    """Synthetic caller with explicit threshold= → no violation."""
    repo = _write_synthetic_repo(tmp_path, {
        "_clean.py": """
            from tac.losses_logit_margin import logit_margin_loss
            import torch

            def clean(thr: float):
                logits = torch.randn(1, 5)
                gt = torch.zeros(1, dtype=torch.long)
                return logit_margin_loss(
                    logits=logits, gt_argmax=gt,
                    threshold=thr, reduction="mean",
                )
        """,
    })
    v = check_logit_margin_loss_uses_boundary_mask(
        repo_root=repo, strict=False, verbose=False,
    )
    assert v == [], f"Clean caller should not be flagged; got {v}"


def test_fragility_weights_caught_too(tmp_path: Path) -> None:
    """fragility_weights() callers are also covered."""
    repo = _write_synthetic_repo(tmp_path, {
        "_frag.py": """
            from tac.losses_logit_margin import fragility_weights
            import torch

            def x():
                logits = torch.randn(1, 5)
                # MISSING threshold=
                return fragility_weights(logits)
        """,
    })
    v = check_logit_margin_loss_uses_boundary_mask(
        repo_root=repo, strict=False, verbose=False,
    )
    assert any("fragility_weights" in line for line in v), (
        f"Expected violation for fragility_weights; got {v}"
    )


def test_compute_segnet_logit_margin_aux_caught(tmp_path: Path) -> None:
    """compute_segnet_logit_margin_aux() callers are also covered."""
    repo = _write_synthetic_repo(tmp_path, {
        "_aux.py": """
            from tac.losses_logit_margin import compute_segnet_logit_margin_aux

            def y(rendered, gt, segnet):
                # MISSING threshold=
                return compute_segnet_logit_margin_aux(
                    rendered_pair=rendered, gt_pair=gt, segnet=segnet,
                )
        """,
    })
    v = check_logit_margin_loss_uses_boundary_mask(
        repo_root=repo, strict=False, verbose=False,
    )
    assert any("compute_segnet_logit_margin_aux" in line for line in v), (
        f"Expected violation; got {v}"
    )


def test_strict_raises_metabugviolation(tmp_path: Path) -> None:
    """strict=True + violation → MetaBugViolation raised."""
    repo = _write_synthetic_repo(tmp_path, {
        "_bad.py": """
            from tac.losses_logit_margin import logit_margin_loss
            import torch
            def b():
                return logit_margin_loss(torch.randn(1, 5), torch.zeros(1, dtype=torch.long))
        """,
    })
    with pytest.raises(MetaBugViolation, match="Lane 19"):
        check_logit_margin_loss_uses_boundary_mask(
            repo_root=repo, strict=True, verbose=False,
        )


def test_attribute_call_form_recognised(tmp_path: Path) -> None:
    """Calls like `mod.logit_margin_loss(...)` (attribute form) are also scanned."""
    repo = _write_synthetic_repo(tmp_path, {
        "_attr.py": """
            from tac import losses_logit_margin as lm
            import torch
            def z():
                return lm.logit_margin_loss(
                    torch.randn(1, 5),
                    torch.zeros(1, dtype=torch.long),
                )
        """,
    })
    v = check_logit_margin_loss_uses_boundary_mask(
        repo_root=repo, strict=False, verbose=False,
    )
    assert any("logit_margin_loss" in line for line in v), (
        f"Attribute-form call should be flagged; got {v}"
    )


def test_unrelated_function_not_flagged(tmp_path: Path) -> None:
    """A function with the same arg name but different name is not flagged."""
    repo = _write_synthetic_repo(tmp_path, {
        "_unrelated.py": """
            def some_other_function(logits, gt):
                # 'threshold' not present, but this is NOT a Lane 19 caller
                return logits.sum() + gt.sum()
            def caller():
                return some_other_function(None, None)
        """,
    })
    v = check_logit_margin_loss_uses_boundary_mask(
        repo_root=repo, strict=False, verbose=False,
    )
    assert v == [], f"Unrelated function must not be flagged; got {v}"


def test_scan_helper_handles_syntax_error_silently(tmp_path: Path) -> None:
    """Files with syntax errors are skipped (other checks catch those)."""
    bad_file = tmp_path / "src" / "tac" / "_broken.py"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("def x(:::")  # syntax error
    # Should not raise.
    v = _scan_lane19_threshold_calls(bad_file, tmp_path)
    assert v == []


def test_no_violation_when_no_lane19_calls(tmp_path: Path) -> None:
    """Files that never reference Lane 19 names short-circuit cleanly."""
    repo = _write_synthetic_repo(tmp_path, {
        "_irrelevant.py": """
            import torch

            def hello():
                return torch.zeros(1)
        """,
    })
    v = check_logit_margin_loss_uses_boundary_mask(
        repo_root=repo, strict=False, verbose=False,
    )
    assert v == []
