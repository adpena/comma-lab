# SPDX-License-Identifier: MIT
"""Catalog #161 — `check_quantize_degenerate_range_clamped_correctly` tests.

(Originally landed as Catalog #160 then renumbered to #161 by FIX-A
2026-05-12 after ZZZZZ audit found a #158 collision with DDDD's
`check_deterministic_compiler_canonical_use`. File path retained for
git-blame continuity; the canonical catalog number is now #161.)

The check refuses substrate archive `_quantize_intN` functions whose
degenerate (``hi <= lo``) branch fills `q` with zeros instead of
`-(MAX_LEVELS // 2)`. Per CLAUDE.md "Bugs must be permanently fixed AND
self-protected against" non-negotiable.

Sister bug class (NNN, FFFF Bug 1, 2026-05-12): two substrate archives
already had the bug fixed (block_nerv via NNN, sane_hnerv via FFFF).
The META gate refuses re-introduction at any of the 12 substrate
archives.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_quantize_degenerate_range_clamped_correctly,
)


def _write_archive(tmp_path: Path, body: str, dir_name: str = "test_sub") -> Path:
    """Write a minimal substrate archive.py to tmp_path/src/tac/substrates/<dir_name>/archive.py."""
    archive_dir = tmp_path / "src" / "tac" / "substrates" / dir_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_py = archive_dir / "archive.py"
    archive_py.write_text(body)
    return archive_py


_CORRECT_BODY_INT16 = '''
import torch

def _quantize_int16(t):
    """Quantize a float tensor to int16."""
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError("must be float")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (
            torch.full_like(f, -32767, dtype=torch.int16),
            1.0,
            lo,
        )
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)
'''

_BUGGY_BODY_ZEROS_INT16 = '''
import torch

def _quantize_int16(t):
    """Quantize a float tensor to int16 — BUGGY zero-fill."""
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError("must be float")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (torch.zeros_like(f, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)
'''


_BUGGY_BODY_ZEROS_INT8 = '''
import torch

def _quantize_int8(t):
    """Quantize a float tensor to int8 — BUGGY zero-fill."""
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (torch.zeros_like(f, dtype=torch.int8), 1.0, lo)
    scale = (hi - lo) / 254.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 254.0)
    q = (q_unsigned - 127.0).to(torch.int8)
    return (q, scale, lo)
'''


_CORRECT_BODY_INT8 = '''
import torch

def _quantize_int8(t):
    """Quantize a float tensor to int8."""
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (
            torch.full_like(f, -127, dtype=torch.int8),
            1.0,
            lo,
        )
    scale = (hi - lo) / 254.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 254.0)
    q = (q_unsigned - 127.0).to(torch.int8)
    return (q, scale, lo)
'''


def test_correct_int16_pattern_passes(tmp_path):
    _write_archive(tmp_path, _CORRECT_BODY_INT16, "good_sub")
    v = check_quantize_degenerate_range_clamped_correctly(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_correct_int8_pattern_passes(tmp_path):
    _write_archive(tmp_path, _CORRECT_BODY_INT8, "good_int8")
    v = check_quantize_degenerate_range_clamped_correctly(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_buggy_int16_zero_fill_violates(tmp_path):
    _write_archive(tmp_path, _BUGGY_BODY_ZEROS_INT16, "buggy_sub")
    v = check_quantize_degenerate_range_clamped_correctly(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1, f"expected 1 violation; got {v}"
    assert "_quantize_int16" in v[0]
    assert "-32767" in v[0]


def test_buggy_int8_zero_fill_violates(tmp_path):
    _write_archive(tmp_path, _BUGGY_BODY_ZEROS_INT8, "buggy_int8_sub")
    v = check_quantize_degenerate_range_clamped_correctly(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "_quantize_int8" in v[0]
    assert "-127" in v[0]


def test_strict_mode_raises(tmp_path):
    _write_archive(tmp_path, _BUGGY_BODY_ZEROS_INT16, "strict_bad")
    with pytest.raises(PreflightError):
        check_quantize_degenerate_range_clamped_correctly(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_same_line_waiver_on_return_accepted(tmp_path):
    body = '''
import torch

def _quantize_int16(t):
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (torch.zeros_like(f, dtype=torch.int16), 1.0, lo)  # QUANTIZE_DEGENERATE_OK: legacy migration in progress
    return (torch.zeros_like(f, dtype=torch.int16), 1.0, lo)
'''
    _write_archive(tmp_path, body, "waived_sub")
    v = check_quantize_degenerate_range_clamped_correctly(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # Only the unwaived second return outside an if block does not match
    # the degenerate branch pattern, so 0 violations expected.
    assert v == []


def test_same_line_waiver_on_if_line_accepted(tmp_path):
    body = '''
import torch

def _quantize_int16(t):
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:  # QUANTIZE_DEGENERATE_OK: legacy code under migration
        return (torch.zeros_like(f, dtype=torch.int16), 1.0, lo)
    return (torch.zeros_like(f, dtype=torch.int16), 1.0, lo)
'''
    _write_archive(tmp_path, body, "waived_if_sub")
    v = check_quantize_degenerate_range_clamped_correctly(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_non_substrate_directory_not_scanned(tmp_path):
    # Write to wrong location — should not be scanned
    other_dir = tmp_path / "src" / "tac" / "other"
    other_dir.mkdir(parents=True)
    (other_dir / "archive.py").write_text(_BUGGY_BODY_ZEROS_INT16)
    v = check_quantize_degenerate_range_clamped_correctly(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_quantize_latents_to_int16_naming_accepted(tmp_path):
    """The alternate naming pattern `_quantize_latents_to_int16` matches."""
    body = '''
import torch

def _quantize_latents_to_int16(latents):
    f = latents.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (torch.zeros_like(f, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    return (((f - lo) / scale).round().to(torch.int16), scale, lo)
'''
    _write_archive(tmp_path, body, "latents_named_sub")
    v = check_quantize_degenerate_range_clamped_correctly(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "_quantize_latents_to_int16" in v[0]


def test_no_degenerate_branch_skips_check(tmp_path):
    """A quantize function without `if hi <= lo` is not checked."""
    body = '''
import torch

def _quantize_int16(t):
    """No degenerate-range guard at all."""
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    scale = (hi - lo) / 65534.0
    return (((f - lo) / scale).round().to(torch.int16), scale, lo)
'''
    _write_archive(tmp_path, body, "no_guard_sub")
    v = check_quantize_degenerate_range_clamped_correctly(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_live_repo_clean(tmp_path):
    """The live repo state must be clean — extracts the real repo root."""
    # Use the real repo root for this integration check.
    repo_root = Path(__file__).resolve().parents[3]
    v = check_quantize_degenerate_range_clamped_correctly(
        repo_root=repo_root, strict=False, verbose=False
    )
    assert v == [], f"live repo has degenerate-range violations: {v}"


def test_strict_mode_passes_on_clean_state(tmp_path):
    """Strict mode raises only when violations exist; clean state returns []."""
    _write_archive(tmp_path, _CORRECT_BODY_INT16, "clean_strict")
    result = check_quantize_degenerate_range_clamped_correctly(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert result == []


def test_no_substrates_dir_returns_empty(tmp_path):
    """If `src/tac/substrates/` doesn't exist, the check returns [] gracefully."""
    v = check_quantize_degenerate_range_clamped_correctly(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_multiple_buggy_substrates_all_flagged(tmp_path):
    _write_archive(tmp_path, _BUGGY_BODY_ZEROS_INT16, "buggy_a")
    _write_archive(tmp_path, _BUGGY_BODY_ZEROS_INT16, "buggy_b")
    _write_archive(tmp_path, _BUGGY_BODY_ZEROS_INT16, "buggy_c")
    v = check_quantize_degenerate_range_clamped_correctly(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 3
    assert any("buggy_a" in x for x in v)
    assert any("buggy_b" in x for x in v)
    assert any("buggy_c" in x for x in v)
