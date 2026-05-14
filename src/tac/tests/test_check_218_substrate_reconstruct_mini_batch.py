# SPDX-License-Identifier: MIT
"""Tests for Catalog #218 — substrate reconstruct methods must support a
mini-batch / no-grad path.

Empirical anchor 2026-05-14: fc-01KRK9RKD3QV4C276Y5KXFMF65 (D4 Modal T4
smoke at 13:10:25Z) returned rc=1 elapsed 121s with CUDA OOM at
``synthesize_frame_0`` F.interpolate residual upsample with 600-pair
batch. The 384x512 residual upsample requires ~13 GB activation memory,
which exceeds T4's 14.56 GB capacity.

Companion fix landed ``pair_indices`` kwarg on
``WynerZivFrame0Substrate.reconstruct_pair`` in the same commit batch.
This gate keeps the bug class extinct across all future substrates that
ship a per-pair reconstruct surface.

Sister of Catalog #154 + #207 — canonical-path-enforcement gates.

Memory: feedback_d4_oom_fix_minibatch_landed_20260514.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_218_collect_violations_in_file,
    _check_218_function_name_matches,
    _check_218_iter_substrate_architecture_files,
    _check_218_signature_has_minibatch_kwarg,
    check_substrate_reconstruct_methods_support_mini_batch,
)


@pytest.fixture
def fake_substrates_root(tmp_path: Path) -> Path:
    """Build a fake src/tac/substrates/ layout with one substrate dir."""
    root = tmp_path
    substrates = root / "src" / "tac" / "substrates" / "fake_sub"
    substrates.mkdir(parents=True)
    return root


def _write_arch(root: Path, name: str, body: str) -> Path:
    """Write a fake substrate architecture.py under
    src/tac/substrates/<name>/architecture.py and return the path."""
    sub_dir = root / "src" / "tac" / "substrates" / name
    sub_dir.mkdir(parents=True, exist_ok=True)
    p = sub_dir / "architecture.py"
    p.write_text(body)
    return p


# ----------------------------------------------------------------------
# Positive cases — violations must be flagged
# ----------------------------------------------------------------------


def test_reconstruct_pair_without_kwarg_flagged(fake_substrates_root):
    body = """
import torch
class Sub:
    def reconstruct_pair(self, frame_1):
        return frame_1, frame_1
"""
    _write_arch(fake_substrates_root, "sub_a", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert len(violations) == 1
    assert "reconstruct_pair" in violations[0]


def test_reconstruct_pairs_plural_flagged(fake_substrates_root):
    body = """
import torch
class Sub:
    def reconstruct_pairs(self, frames):
        return frames
"""
    _write_arch(fake_substrates_root, "sub_b", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert len(violations) == 1


def test_archive_build_pass_flagged(fake_substrates_root):
    body = """
import torch
class Sub:
    def archive_build_pass(self, frames):
        return frames
"""
    _write_arch(fake_substrates_root, "sub_c", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert len(violations) == 1


def test_frame_synthesis_helper_flagged(fake_substrates_root):
    """frame*_synthesis.py also scanned (sister surface to architecture.py)."""
    sub_dir = fake_substrates_root / "src" / "tac" / "substrates" / "sub_d"
    sub_dir.mkdir(parents=True)
    (sub_dir / "frame0_synthesis.py").write_text("""
import torch
def reconstruct_pair(frame_1):
    return frame_1
""")
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert len(violations) == 1


def test_multiple_violations_in_one_file_all_flagged(fake_substrates_root):
    body = """
import torch
class Sub:
    def reconstruct_pair(self, frame_1):
        return frame_1, frame_1
    def reconstruct_pairs(self, frames):
        return frames
"""
    _write_arch(fake_substrates_root, "sub_e", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert len(violations) == 2


# ----------------------------------------------------------------------
# Negative cases — must NOT be flagged
# ----------------------------------------------------------------------


def test_pair_indices_kwarg_satisfies_gate(fake_substrates_root):
    body = """
import torch
class Sub:
    def reconstruct_pair(self, frame_1, pair_indices=None):
        return frame_1, frame_1
"""
    _write_arch(fake_substrates_root, "sub_f", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert violations == []


def test_chunk_size_kwarg_satisfies_gate(fake_substrates_root):
    body = """
import torch
class Sub:
    def reconstruct_pair(self, frame_1, chunk_size=32):
        return frame_1, frame_1
"""
    _write_arch(fake_substrates_root, "sub_g", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert violations == []


def test_indices_kwarg_satisfies_gate(fake_substrates_root):
    body = """
import torch
class Sub:
    def reconstruct_pair(self, frame_1, indices=None):
        return frame_1, frame_1
"""
    _write_arch(fake_substrates_root, "sub_h", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert violations == []


def test_no_grad_body_context_satisfies_gate(fake_substrates_root):
    body = """
import torch
class Sub:
    def reconstruct_pair(self, frame_1):
        with torch.no_grad():
            return frame_1, frame_1
"""
    _write_arch(fake_substrates_root, "sub_i", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert violations == []


def test_inference_mode_body_context_satisfies_gate(fake_substrates_root):
    body = """
import torch
class Sub:
    def reconstruct_pair(self, frame_1):
        with torch.inference_mode():
            return frame_1, frame_1
"""
    _write_arch(fake_substrates_root, "sub_j", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert violations == []


def test_no_grad_decorator_satisfies_gate(fake_substrates_root):
    body = """
import torch
class Sub:
    @torch.no_grad()
    def reconstruct_pair(self, frame_1):
        return frame_1, frame_1
"""
    _write_arch(fake_substrates_root, "sub_k", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert violations == []


def test_unrelated_method_not_flagged(fake_substrates_root):
    """Methods with names not matching the reconstruct pattern are
    out-of-scope (e.g. ``decode``, ``encode``)."""
    body = """
class Sub:
    def decode(self, frame_1):
        return frame_1
    def encode(self, frame_1):
        return frame_1
"""
    _write_arch(fake_substrates_root, "sub_l", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert violations == []


# ----------------------------------------------------------------------
# Waiver semantics
# ----------------------------------------------------------------------


def test_same_line_waiver_with_reason_accepted(fake_substrates_root):
    body = """
import torch
class Sub:
    def reconstruct_pair(self, frame_1):  # RECONSTRUCT_BATCH_OOM_OK: smoke-only fixture, batch<=4
        return frame_1, frame_1
"""
    _write_arch(fake_substrates_root, "sub_m", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert violations == []


def test_placeholder_reason_rejected(fake_substrates_root):
    body = """
import torch
class Sub:
    def reconstruct_pair(self, frame_1):  # RECONSTRUCT_BATCH_OOM_OK: <reason>
        return frame_1, frame_1
"""
    _write_arch(fake_substrates_root, "sub_n", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert len(violations) == 1


def test_waiver_without_colon_not_recognized(fake_substrates_root):
    body = """
import torch
class Sub:
    def reconstruct_pair(self, frame_1):  # RECONSTRUCT_BATCH_OOM_OK
        return frame_1, frame_1
"""
    _write_arch(fake_substrates_root, "sub_o", body)
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False
    )
    assert len(violations) == 1


# ----------------------------------------------------------------------
# Strict-mode behavior
# ----------------------------------------------------------------------


def test_strict_mode_raises_preflight_error(fake_substrates_root):
    body = """
class Sub:
    def reconstruct_pair(self, frame_1):
        return frame_1, frame_1
"""
    _write_arch(fake_substrates_root, "sub_p", body)
    with pytest.raises(PreflightError, match="Catalog #218"):
        check_substrate_reconstruct_methods_support_mini_batch(
            repo_root=fake_substrates_root, strict=True
        )


def test_strict_mode_passes_when_clean(fake_substrates_root):
    body = """
class Sub:
    def reconstruct_pair(self, frame_1, pair_indices=None):
        return frame_1, frame_1
"""
    _write_arch(fake_substrates_root, "sub_q", body)
    # Should not raise
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=True
    )
    assert violations == []


# ----------------------------------------------------------------------
# Helper unit tests
# ----------------------------------------------------------------------


def test_function_name_matches_helper():
    import ast
    src = "def reconstruct_pair(): pass\ndef decode(): pass"
    tree = ast.parse(src)
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert _check_218_function_name_matches(funcs[0])
    assert not _check_218_function_name_matches(funcs[1])


def test_signature_has_minibatch_kwarg_helper():
    import ast
    src = (
        "def f1(x, pair_indices=None): pass\n"
        "def f2(x, y): pass\n"
        "def f3(x, *, chunk_size=32): pass\n"
        "def f4(x, **kwargs): pass\n"
    )
    tree = ast.parse(src)
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert _check_218_signature_has_minibatch_kwarg(funcs[0])
    assert not _check_218_signature_has_minibatch_kwarg(funcs[1])
    assert _check_218_signature_has_minibatch_kwarg(funcs[2])
    assert not _check_218_signature_has_minibatch_kwarg(funcs[3])


def test_iter_substrate_files_returns_empty_for_missing_dir(tmp_path):
    files = _check_218_iter_substrate_architecture_files(tmp_path)
    assert files == []


def test_collect_violations_handles_syntax_error(fake_substrates_root):
    """SyntaxError in a substrate file must not raise; return empty."""
    sub_dir = fake_substrates_root / "src" / "tac" / "substrates" / "sub_syn"
    sub_dir.mkdir(parents=True)
    arch = sub_dir / "architecture.py"
    arch.write_text("def reconstruct_pair(broken syntax")
    violations = _check_218_collect_violations_in_file(arch, fake_substrates_root)
    assert violations == []


def test_verbose_output_on_clean(fake_substrates_root, capsys):
    check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False, verbose=True
    )
    out = capsys.readouterr().out
    assert "OK" in out and "scanned" in out


def test_verbose_output_on_dirty(fake_substrates_root, capsys):
    body = """
class Sub:
    def reconstruct_pair(self, frame_1):
        return frame_1
"""
    _write_arch(fake_substrates_root, "sub_v", body)
    check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=fake_substrates_root, strict=False, verbose=True
    )
    out = capsys.readouterr().out
    assert "WARN" in out


def test_string_repo_root_accepted(fake_substrates_root):
    """The function accepts a string repo_root and converts to Path."""
    violations = check_substrate_reconstruct_methods_support_mini_batch(
        repo_root=str(fake_substrates_root), strict=False
    )
    assert violations == []


# ----------------------------------------------------------------------
# Live-repo regression guard
# ----------------------------------------------------------------------


def test_live_repo_count_is_zero():
    """The live repo must have ZERO violations after the D4 fix landed.

    Per CLAUDE.md "Strict-flip atomicity rule" — the gate is STRICT @ 0
    from byte one because the D4 substrate's reconstruct_pair gained the
    ``pair_indices`` kwarg in the same commit batch as this gate.
    """
    violations = check_substrate_reconstruct_methods_support_mini_batch(strict=False)
    assert violations == [], (
        f"Live repo has {len(violations)} Catalog #218 violations; "
        f"first 3:\n  " + "\n  ".join(violations[:3])
    )


def test_orchestrator_callsite_uses_strict_true():
    """preflight_all() must call this gate with strict=True per the
    Strict-flip atomicity rule."""
    from pathlib import Path as _P
    src = _P(__file__).parent.parent / "preflight.py"
    text = src.read_text()
    # The wire-in must use strict=True
    needle = (
        "check_substrate_reconstruct_methods_support_mini_batch(\n"
        "            strict=True"
    )
    assert needle in text, (
        "preflight_all() wire-in for Catalog #218 must use strict=True; "
        "Strict-flip atomicity rule per CLAUDE.md"
    )
