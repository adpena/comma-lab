"""Tests for ``--full-cpu`` mode in time-traveler L5 autonomy trainer.

Operator approved 2026-05-13. Per CLAUDE.md "Submission auth eval — BOTH CPU
AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" + Catalog #127 + #192 + #197:
``--full-cpu`` opens a non-smoke CPU path that produces ``[macOS-CPU advisory
only]`` non-promotable scores. The mode requires explicit operator
acknowledgement via ``--advisory-cpu-explicitly-waived``.

Scope:

- (R1) flag parsing + defaults
- (R1) waiver requirement (the device-or-die-gate-bypass-via-uncoupled-flags
  bug class is structurally extincted here)
- (R1) mutual exclusion with ``--smoke``
- (R1) ``--full-cpu`` requires ``--device cpu``
- (R1) dangling waiver flag refused
- (R2) ``--max-wall-clock-hours`` default + custom value
- (R2) banner written to stderr
- (R3) CLI help mentions advisory-only nature

The trainer's full _full_main path is exercised by remote dispatches; here we
test the contract surface (CLI + helper functions) that this landing adds.
"""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_PATH = REPO_ROOT / "experiments" / "train_substrate_time_traveler_l5_autonomy.py"


@pytest.fixture(scope="module")
def trainer_module():
    """Load the trainer module by path (it's not a package member)."""
    spec = importlib.util.spec_from_file_location(
        "tt5l_trainer_under_test", str(TRAINER_PATH)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# R1 — flag parsing + defaults
# ---------------------------------------------------------------------------


def test_full_cpu_flag_defaults_false(trainer_module):
    """``--full-cpu`` and ``--advisory-cpu-explicitly-waived`` default to False."""
    parser = trainer_module._build_parser()
    args = parser.parse_args(["--output-dir", "/x", "--epochs", "1"])
    assert args.full_cpu is False
    assert args.advisory_cpu_explicitly_waived is False


def test_max_wall_clock_hours_default(trainer_module):
    """``--max-wall-clock-hours`` defaults to 12.0 (Carmack-pessimistic bound)."""
    parser = trainer_module._build_parser()
    args = parser.parse_args(["--output-dir", "/x", "--epochs", "1"])
    assert args.max_wall_clock_hours == 12.0


def test_full_cpu_flag_set(trainer_module):
    """``--full-cpu`` flag parses to True when passed."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--full-cpu", "--advisory-cpu-explicitly-waived",
    ])
    assert args.full_cpu is True
    assert args.advisory_cpu_explicitly_waived is True
    assert args.device == "cpu"


def test_max_wall_clock_hours_custom(trainer_module):
    """``--max-wall-clock-hours`` accepts a custom value (e.g. 6.0)."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--max-wall-clock-hours", "6.0",
    ])
    assert args.max_wall_clock_hours == 6.0


# ---------------------------------------------------------------------------
# R1 — waiver requirement (the canonical bug class this landing extincts)
# ---------------------------------------------------------------------------


def test_full_cpu_without_waiver_raises(trainer_module):
    """``--full-cpu`` without ``--advisory-cpu-explicitly-waived`` is refused."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--full-cpu",
    ])
    with pytest.raises(SystemExit) as exc_info:
        trainer_module._validate_full_cpu_flags(args)
    assert "advisory-cpu-explicitly-waived" in str(exc_info.value)


def test_full_cpu_with_waiver_passes(trainer_module):
    """``--full-cpu`` with the waiver passes validation."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--full-cpu", "--advisory-cpu-explicitly-waived",
    ])
    # No exception.
    trainer_module._validate_full_cpu_flags(args)


def test_dangling_waiver_without_full_cpu_raises(trainer_module):
    """``--advisory-cpu-explicitly-waived`` without ``--full-cpu`` is refused.

    Per CLAUDE.md Catalog #133 + #136 pattern (no broad accept tokens): a
    dangling waiver flag could silently waive a later, unrelated check.
    """
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--advisory-cpu-explicitly-waived",
    ])
    with pytest.raises(SystemExit) as exc_info:
        trainer_module._validate_full_cpu_flags(args)
    assert "Dangling waiver flag" in str(exc_info.value)


def test_full_cpu_and_smoke_mutually_exclusive(trainer_module):
    """``--full-cpu`` + ``--smoke`` is refused (distinct modes)."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--full-cpu", "--advisory-cpu-explicitly-waived",
        "--smoke",
    ])
    with pytest.raises(SystemExit) as exc_info:
        trainer_module._validate_full_cpu_flags(args)
    assert "mutually exclusive" in str(exc_info.value)


def test_full_cpu_with_device_cuda_raises(trainer_module):
    """``--full-cpu`` with ``--device cuda`` is refused (contradiction)."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cuda", "--full-cpu", "--advisory-cpu-explicitly-waived",
    ])
    with pytest.raises(SystemExit) as exc_info:
        trainer_module._validate_full_cpu_flags(args)
    assert "--device cpu" in str(exc_info.value)


# ---------------------------------------------------------------------------
# R2 — banner emission
# ---------------------------------------------------------------------------


def test_full_cpu_banner_emitted_to_stderr(trainer_module, capsys):
    """``_full_cpu_banner`` writes the loud advisory banner to stderr."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--full-cpu", "--advisory-cpu-explicitly-waived",
    ])
    trainer_module._full_cpu_banner(args)
    captured = capsys.readouterr()
    assert "[macOS-CPU advisory only]" in captured.err
    assert "NON-promotable" in captured.err
    assert 'evidence_grade = "macOS-CPU-advisory"' in captured.err
    assert "score_claim" in captured.err
    assert "promotion_eligible" in captured.err
    assert "ready_for_exact_eval_dispatch" in captured.err


def test_banner_silent_without_full_cpu(trainer_module, capsys):
    """Banner is silent in non-``--full-cpu`` runs."""
    parser = trainer_module._build_parser()
    args = parser.parse_args(["--output-dir", "/x", "--epochs", "1"])
    trainer_module._full_cpu_banner(args)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_banner_includes_max_wall_clock_hours(trainer_module, capsys):
    """Banner surfaces ``--max-wall-clock-hours`` so the operator sees the budget."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--full-cpu", "--advisory-cpu-explicitly-waived",
        "--max-wall-clock-hours", "6.5",
    ])
    trainer_module._full_cpu_banner(args)
    captured = capsys.readouterr()
    assert "6.5" in captured.err


# ---------------------------------------------------------------------------
# R3 — CLI help mentions advisory-only nature
# ---------------------------------------------------------------------------


def test_full_cpu_help_mentions_advisory(trainer_module):
    """``--full-cpu`` help text mentions advisory-only + non-promotable nature."""
    parser = trainer_module._build_parser()
    buf = io.StringIO()
    parser.print_help(buf)
    help_text = buf.getvalue()
    assert "--full-cpu" in help_text
    assert "advisory" in help_text.lower()
    assert "non-promotable" in help_text.lower()
    assert "advisory-cpu-explicitly-waived" in help_text.lower()


def test_full_cpu_help_mentions_wall_clock_range(trainer_module):
    """``--full-cpu`` help mentions the 2-12h wall-clock expectation."""
    parser = trainer_module._build_parser()
    buf = io.StringIO()
    parser.print_help(buf)
    help_text = buf.getvalue()
    assert "2-12h" in help_text or ("2-6" in help_text and "12" in help_text)


# ---------------------------------------------------------------------------
# Defense in depth — smoke + canonical CUDA path are unaffected
# ---------------------------------------------------------------------------


def test_smoke_path_unaffected_by_new_flags(trainer_module):
    """``--smoke`` without ``--full-cpu`` still works (no regression)."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1",
        "--device", "cpu", "--smoke",
    ])
    # No exception.
    trainer_module._validate_full_cpu_flags(args)
    assert args.smoke is True
    assert args.full_cpu is False


def test_cuda_path_unaffected_by_new_flags(trainer_module):
    """``--device cuda`` without ``--full-cpu`` still works (no regression)."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--output-dir", "/x", "--epochs", "1", "--device", "cuda",
    ])
    # No exception.
    trainer_module._validate_full_cpu_flags(args)
    assert args.device == "cuda"
    assert args.full_cpu is False
