"""Tests for Check 103 (PCC1): IMP dispatch must invoke train_distill.

Bug class this guards: stub-pretending-to-be-real (the IMP cycle 0 = 1.98
[contest-CUDA] metabug, 2026-04-30). ``experiments/train_imp_cycle.py``'s
``_finetune`` is a documented in-script STUB loop (synthetic tensors, toy
L2 loss, ~0.017s/epoch on L40S). Without a Stage 1.X swap to a real trainer
(``train_distill.py`` / ``train_renderer.py`` / ``train_renderer_fridrich.py``),
a 10-cycle IMP dispatch reproduces the metabug cycle after cycle, burning
GPU hours on stub-fine-tuned weights and silently shipping a
non-trained 89%-sparse renderer into the contest archive pipeline.

The check is the dispatch-time companion to PCC3 (the wall-clock assertion
in ``train_imp_cycle.py`` main, ~L362-374) which catches the bug at runtime
if the stub somehow ships despite this check.

Memory:
- feedback_grand_council_imp_permanent_fix_review_20260430.md (parent
  council 6/3/1 vote for Option B+assertion)
- feedback_grand_council_imp_train_distill_swap_design_20260430.md
  (sub-question deliberation: 9/10 vote each on epochs/masks/auth-smoke)
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
    _scan_imp_dispatcher_for_train_distill_swap,
    check_imp_dispatch_calls_train_distill,
)


# ─────────────────────────────────────────────────────────────────────────────
# Real-codebase regression — STRICT @ 0 violations
# ─────────────────────────────────────────────────────────────────────────────


def test_real_codebase_passes_warn() -> None:
    """[regression] Real codebase has 0 PCC1 violations.

    The canonical IMP dispatcher
    ``scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh`` invokes
    BOTH train_imp_cycle.py AND train_distill.py (Stage 1.X swap landed
    2026-04-30 PM).
    """
    v = check_imp_dispatch_calls_train_distill(strict=False, verbose=False)
    assert v == [], (
        f"Real codebase should be clean; got {len(v)} violation(s):\n"
        + "\n".join(f"  • {x}" for x in v)
    )


def test_real_codebase_passes_strict() -> None:
    """[regression] strict=True on real codebase passes without raising."""
    check_imp_dispatch_calls_train_distill(strict=True, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic counter-examples — verify the bug class is caught
# ─────────────────────────────────────────────────────────────────────────────


def _write_synth_dispatcher(tmp: Path, name: str, body: str) -> Path:
    """Skeleton repo with scripts/<name> dispatcher."""
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    p = scripts_dir / name
    p.write_text(textwrap.dedent(body).strip() + "\n")
    return p


def test_dispatcher_with_only_stub_invocation_is_flagged(tmp_path: Path) -> None:
    """Dispatcher invokes train_imp_cycle.py but NOT a real trainer → flagged.

    This is the cycle 0 = 1.98 [contest-CUDA] metabug shape: dispatcher runs
    the stub fine-tune cycle after cycle and silently ships a non-trained
    renderer. The PCC1 check fails LOUD at preflight time.
    """
    p = _write_synth_dispatcher(
        tmp_path,
        "remote_lane_j_imp_buggy.sh",
        '''
        #!/bin/bash
        set -euo pipefail
        WORKSPACE=/workspace/pact
        PYBIN=/opt/conda/bin/python
        for i in 0 1 2; do
            "$PYBIN" -u experiments/train_imp_cycle.py \\
                --cycle "$i" \\
                --checkpoint "$WORKSPACE/anchor.bin" \\
                --output-dir "$WORKSPACE/cycle_${i}" \\
                --epochs 200
        done
        ''',
    )
    v = _scan_imp_dispatcher_for_train_distill_swap(p, tmp_path)
    assert len(v) == 1
    assert "without a subsequent real-trainer invocation" in v[0]
    assert "cycle 0 = 1.98" in v[0]


def test_dispatcher_with_train_distill_swap_passes(tmp_path: Path) -> None:
    """Dispatcher invokes train_imp_cycle.py AND train_distill.py → clean."""
    p = _write_synth_dispatcher(
        tmp_path,
        "remote_lane_j_imp_clean.sh",
        '''
        #!/bin/bash
        set -euo pipefail
        WORKSPACE=/workspace/pact
        PYBIN=/opt/conda/bin/python
        for i in 0 1 2; do
            "$PYBIN" -u experiments/train_imp_cycle.py \\
                --cycle "$i" \\
                --checkpoint "$WORKSPACE/anchor.bin" \\
                --output-dir "$WORKSPACE/cycle_${i}" \\
                --epochs 200
            "$PYBIN" -u experiments/train_distill.py \\
                --resume "$WORKSPACE/cycle_${i}/renderer.pt" \\
                --output-dir "$WORKSPACE/cycle_${i}/distill" \\
                --only-phase1 --phase1-epochs 500
        done
        ''',
    )
    v = _scan_imp_dispatcher_for_train_distill_swap(p, tmp_path)
    assert v == []


def test_dispatcher_with_train_renderer_swap_passes(tmp_path: Path) -> None:
    """Dispatcher invokes train_imp_cycle.py AND train_renderer.py → clean.

    train_renderer.py is one of the accepted real-trainer alternatives in
    `_IMP_REAL_TRAINER_INVOCATIONS`.
    """
    p = _write_synth_dispatcher(
        tmp_path,
        "remote_lane_j_imp_renderer_swap.sh",
        '''
        #!/bin/bash
        "$PYBIN" -u experiments/train_imp_cycle.py --cycle 0 --output-dir foo
        "$PYBIN" -u experiments/train_renderer.py --resume foo/renderer.pt --epochs 500
        ''',
    )
    v = _scan_imp_dispatcher_for_train_distill_swap(p, tmp_path)
    assert v == []


def test_dispatcher_with_train_renderer_fridrich_swap_passes(
    tmp_path: Path,
) -> None:
    """train_renderer_fridrich.py is also an accepted real-trainer."""
    p = _write_synth_dispatcher(
        tmp_path,
        "remote_lane_j_imp_fridrich_swap.sh",
        '''
        #!/bin/bash
        "$PYBIN" -u experiments/train_imp_cycle.py --cycle 0
        "$PYBIN" -u experiments/train_renderer_fridrich.py --resume foo
        ''',
    )
    v = _scan_imp_dispatcher_for_train_distill_swap(p, tmp_path)
    assert v == []


def test_heredoc_reference_to_train_distill_does_not_count(tmp_path: Path) -> None:
    """A bare `open('experiments/train_distill.py')` inside a python heredoc
    is NOT a real invocation — the check should still flag the dispatcher
    if it lacks `python -u experiments/train_distill.py`.

    This is the false-positive class to avoid: a pre-flight argparse-scan
    heredoc references the target path as a STRING but never INVOKES it.
    """
    p = _write_synth_dispatcher(
        tmp_path,
        "remote_lane_j_imp_heredoc_only.sh",
        '''
        #!/bin/bash
        "$PYBIN" -u experiments/train_imp_cycle.py --cycle 0 --output-dir foo
        # Pre-flight scan that READS train_distill.py argparse but never INVOKES it:
        "$PYBIN" -c "
        import re
        src = open('experiments/train_distill.py').read()
        flags = re.findall(r'add_argument\\(--([a-z]+)', src)
        print(flags)
        "
        ''',
    )
    v = _scan_imp_dispatcher_for_train_distill_swap(p, tmp_path)
    # Heredoc reference is NOT a real invocation; dispatcher is buggy.
    assert len(v) == 1
    assert "without a subsequent real-trainer invocation" in v[0]


def test_dispatcher_with_no_imp_invocation_is_skipped(tmp_path: Path) -> None:
    """A script that doesn't invoke train_imp_cycle.py at all is NOT an IMP
    dispatcher and should be skipped (no false positive)."""
    p = _write_synth_dispatcher(
        tmp_path,
        "remote_lane_j_imp_noop.sh",
        '''
        #!/bin/bash
        echo "this script does no IMP work"
        "$PYBIN" -u experiments/something_else.py --foo bar
        ''',
    )
    v = _scan_imp_dispatcher_for_train_distill_swap(p, tmp_path)
    assert v == []


def test_python3_runner_token_is_recognized(tmp_path: Path) -> None:
    """``python3 -u`` is an accepted runner pattern (not just ``$PYBIN -u``)."""
    p = _write_synth_dispatcher(
        tmp_path,
        "remote_lane_j_imp_python3.sh",
        '''
        #!/bin/bash
        python3 -u experiments/train_imp_cycle.py --cycle 0 --output-dir foo
        python3 -u experiments/train_distill.py --resume foo/renderer.pt
        ''',
    )
    v = _scan_imp_dispatcher_for_train_distill_swap(p, tmp_path)
    assert v == []


# ─────────────────────────────────────────────────────────────────────────────
# Strict-mode raise behavior
# ─────────────────────────────────────────────────────────────────────────────


def test_strict_raises_on_synthetic_violation(tmp_path: Path, monkeypatch) -> None:
    """strict=True must raise MetaBugViolation when a synthetic violation exists.

    Uses repo_root override to point the check at a tmp_path that holds
    the buggy dispatcher.
    """
    _write_synth_dispatcher(
        tmp_path,
        "remote_lane_j_imp_strict_raises.sh",
        '''
        #!/bin/bash
        "$PYBIN" -u experiments/train_imp_cycle.py --cycle 0 --output-dir foo
        ''',
    )
    with pytest.raises(MetaBugViolation) as exc:
        check_imp_dispatch_calls_train_distill(
            strict=True, verbose=False, repo_root=tmp_path,
        )
    assert "PCC1" in str(exc.value)
    assert "train_distill" in str(exc.value).lower()


def test_strict_does_not_raise_on_clean_synthetic(tmp_path: Path) -> None:
    """strict=True must NOT raise when a synthetic dispatcher has the swap."""
    _write_synth_dispatcher(
        tmp_path,
        "remote_lane_j_imp_strict_clean.sh",
        '''
        #!/bin/bash
        "$PYBIN" -u experiments/train_imp_cycle.py --cycle 0
        "$PYBIN" -u experiments/train_distill.py --resume foo
        ''',
    )
    # Should not raise.
    v = check_imp_dispatch_calls_train_distill(
        strict=True, verbose=False, repo_root=tmp_path,
    )
    assert v == []
