# SPDX-License-Identifier: MIT
"""Tests for Check 95: Lane 12 NeRV mask codec training-path discipline.

Per CLAUDE.md non-negotiables (EMA, no-MPS, no-bare-round, auth-eval-everywhere)
applied to the Lane 12 NeRV codec lane. Sibling of:
- Check 86 (no bare ``.round()`` in eval-roundtrip chains)
- Check 88 (training paths use canonical ``tac.training.EMA``)
- Check 22 (training scripts have auth eval)

Memory: .omx/research/council_lane_12_nerv_design_20260430.md
Memory: feedback_three_active_bug_classes_needing_strict_checks_20260429.md
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
    check_nerv_codec_uses_ema_and_no_mps_and_auth_eval,
)


def test_real_codebase_passes_strict() -> None:
    """[regression] Real codebase has 0 Lane 12 discipline violations."""
    v = check_nerv_codec_uses_ema_and_no_mps_and_auth_eval(strict=False, verbose=False)
    assert v == [], f"Real codebase should be clean; got {len(v)} violations: {v}"


def test_strict_real_codebase_does_not_raise() -> None:
    """[regression] strict=True on real codebase passes without raising."""
    check_nerv_codec_uses_ema_and_no_mps_and_auth_eval(strict=True, verbose=False)


def _write_synth_codec_repo(tmp: Path, codec_body: str, trainer_body: str = "") -> Path:
    """Skeleton repo with src/tac/nerv_mask_codec.py + experiments/train_nerv_mask.py."""
    (tmp / "src" / "tac").mkdir(parents=True, exist_ok=True)
    (tmp / "experiments").mkdir(parents=True, exist_ok=True)
    (tmp / "src" / "tac" / "nerv_mask_codec.py").write_text(textwrap.dedent(codec_body).strip() + "\n")
    if trainer_body:
        (tmp / "experiments" / "train_nerv_mask.py").write_text(
            textwrap.dedent(trainer_body).strip() + "\n"
        )
    return tmp


def test_missing_canonical_ema_import_caught(tmp_path: Path) -> None:
    """Codec without `from tac.training import EMA` → flagged."""
    repo = _write_synth_codec_repo(
        tmp_path,
        codec_body="""
            class NeRVMaskTrainer:
                def __init__(self):
                    if 'mps' in 'cpu':
                        raise ValueError("refuses device='mps'")
        """,
    )
    v = check_nerv_codec_uses_ema_and_no_mps_and_auth_eval(
        strict=False, verbose=False, repo_root=repo,
    )
    assert any("missing canonical EMA import" in line for line in v), (
        f"Expected EMA-import violation; got {v}"
    )


def test_missing_mps_refusal_caught(tmp_path: Path) -> None:
    """Codec without MPS-refusal raise → flagged."""
    repo = _write_synth_codec_repo(
        tmp_path,
        codec_body="""
            from tac.training import EMA

            class NeRVMaskTrainer:
                def __init__(self, device='cuda'):
                    self.device = device
                    self.ema = EMA(self, decay=0.997)
        """,
    )
    v = check_nerv_codec_uses_ema_and_no_mps_and_auth_eval(
        strict=False, verbose=False, repo_root=repo,
    )
    assert any("must refuse device='mps'" in line for line in v), (
        f"Expected MPS-refusal violation; got {v}"
    )


def test_bare_round_in_step_caught(tmp_path: Path) -> None:
    """Codec with bare ``.round()`` inside a ``step`` method → flagged."""
    repo = _write_synth_codec_repo(
        tmp_path,
        codec_body="""
            from tac.training import EMA
            import torch

            class NeRVMaskTrainer:
                def __init__(self):
                    if False:
                        raise ValueError("refuses device='mps'")
                    self.ema = EMA(self, decay=0.997)

                def step(self, masks, batch_size=4096):
                    x = torch.randn(batch_size, 3)
                    # FORBIDDEN: bare .round() in autograd-active forward
                    return x.round()
        """,
    )
    v = check_nerv_codec_uses_ema_and_no_mps_and_auth_eval(
        strict=False, verbose=False, repo_root=repo,
    )
    assert any("bare `.round()`" in line and "step" in line for line in v), (
        f"Expected bare-.round() violation in step; got {v}"
    )


def test_bare_round_in_forward_caught(tmp_path: Path) -> None:
    """Codec with bare ``.round()`` inside ``forward`` → flagged."""
    repo = _write_synth_codec_repo(
        tmp_path,
        codec_body="""
            from tac.training import EMA
            import torch

            class NeRVMaskTrainer:
                def __init__(self):
                    if False:
                        raise ValueError("refuses device='mps'")

            class FakeCodec:
                def forward(self, x):
                    return x.round()
        """,
    )
    v = check_nerv_codec_uses_ema_and_no_mps_and_auth_eval(
        strict=False, verbose=False, repo_root=repo,
    )
    assert any("bare `.round()`" in line and "forward" in line for line in v), (
        f"Expected bare-.round() violation in forward; got {v}"
    )


def test_round_in_encode_NOT_caught(tmp_path: Path) -> None:
    """``.round()`` in encode-side numpy quantization is allowed
    (CPU-side, not autograd-active forward). The check whitelists it."""
    repo = _write_synth_codec_repo(
        tmp_path,
        codec_body="""
            from tac.training import EMA
            import torch

            class NeRVMaskTrainer:
                def __init__(self):
                    if False:
                        raise ValueError("refuses device='mps'")

            def encode_nerv_codec(codec, weight_dtype='int8'):
                # Encode-side quantization may use .round() — CPU/numpy path,
                # not autograd-active.
                t = torch.zeros(10)
                return (t / 1.0).round()
        """,
    )
    v = check_nerv_codec_uses_ema_and_no_mps_and_auth_eval(
        strict=False, verbose=False, repo_root=repo,
    )
    # Should NOT include a `.round()` violation for encode_nerv_codec (it's
    # not in forbidden_contexts = forward / step / _sample_batch / evaluate*)
    round_violations = [v_ for v_ in v if "bare `.round()`" in v_]
    assert round_violations == [], (
        f"encode-side .round() should be allowed; got round violations: {round_violations}"
    )


def test_trainer_script_missing_auth_eval_caught(tmp_path: Path) -> None:
    """``experiments/train_nerv_mask.py`` without auth-eval invocation
    OR delegation to dispatch script → flagged."""
    repo = _write_synth_codec_repo(
        tmp_path,
        codec_body="""
            from tac.training import EMA

            class NeRVMaskTrainer:
                def __init__(self, device='cuda'):
                    if device == 'mps':
                        raise ValueError("NeRVMaskTrainer refuses device='mps'")
        """,
        trainer_body="""
            def main():
                # No contest_auth_eval call AND no delegation note
                pass
        """,
    )
    v = check_nerv_codec_uses_ema_and_no_mps_and_auth_eval(
        strict=False, verbose=False, repo_root=repo,
    )
    assert any(
        "must either invoke `contest_auth_eval.py`" in line for line in v
    ), f"Expected auth-eval violation; got {v}"


def test_trainer_script_with_dispatch_delegation_OK(tmp_path: Path) -> None:
    """``experiments/train_nerv_mask.py`` that documents delegation to
    ``scripts/remote_lane_nerv.sh`` Stage 3 is acceptable."""
    repo = _write_synth_codec_repo(
        tmp_path,
        codec_body="""
            from tac.training import EMA

            class NeRVMaskTrainer:
                def __init__(self, device='cuda'):
                    if device == 'mps':
                        raise ValueError("NeRVMaskTrainer refuses device='mps'")
        """,
        trainer_body="""
            # Auth eval delegated to scripts/remote_lane_nerv.sh Stage 3
            # (the dispatch script invokes contest_auth_eval after this trainer
            # writes its outputs).
            def main():
                pass
        """,
    )
    v = check_nerv_codec_uses_ema_and_no_mps_and_auth_eval(
        repo_root=repo, strict=False, verbose=False,
    )
    auth_violations = [
        line for line in v if "contest_auth_eval.py" in line
    ]
    assert auth_violations == [], (
        f"Delegation note should satisfy auth-eval check; got {auth_violations}"
    )


def test_strict_raises_on_synthetic_violation(tmp_path: Path) -> None:
    """strict=True raises MetaBugViolation when violations present."""
    repo = _write_synth_codec_repo(
        tmp_path,
        codec_body="""
            # No EMA import, no MPS refusal — multiple violations
            class NeRVMaskTrainer:
                def __init__(self):
                    pass
        """,
    )
    with pytest.raises(MetaBugViolation, match="LANE 12 NERV CODEC DISCIPLINE"):
        check_nerv_codec_uses_ema_and_no_mps_and_auth_eval(
            repo_root=repo, strict=True, verbose=False,
        )
