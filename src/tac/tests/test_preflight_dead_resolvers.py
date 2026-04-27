"""Tests for preflight_dead_resolvers — the 2026-04-27 R5 scanner that
catches the pose_dim / segnet_uncertainty_weighted_loss / uncertainty_loss_floor
bug class (silent dead resolvers + stale .pyc-masked dead imports).

These tests pin down the scanner mechanics so the bug class cannot regress.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    DeadResolverViolation,
    _import_inside_try_handler,
    _is_resolvable_submodule,
    _module_top_level_names,
    _scan_python_for_dead_imports,
    _scan_python_for_dead_resolvers,
    preflight_dead_resolvers,
)


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip("\n"))


def _stub_repo(tmp_path: Path) -> Path:
    """Build a minimal fake repo with src/tac/ for in-repo imports."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    _write(tmp_path / "src" / "tac" / "__init__.py", "")
    return tmp_path


# ── Dead resolvers ────────────────────────────────────────────────────────────


def test_dead_resolver_pose_dim_class_is_caught(tmp_path: Path) -> None:
    """The exact pose_dim bug pattern: getattr(args, 'X', DEFAULT) with no
    --X flag and no `args.X = ...` resolver."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "broken.py"
    _write(script, """
        import argparse
        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--profile", default="default")
            return p.parse_args()
        def build():
            args = parse_args()
            x = getattr(args, "pose_dim", 0)
            return x
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert len(v) == 1, v
    assert "pose_dim" in v[0]
    assert "DEAD RESOLVER" in v[0]


def test_dead_resolver_satisfied_by_argparse_flag(tmp_path: Path) -> None:
    """getattr is fine if there's a corresponding --pose-dim argparse flag."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "good_argparse.py"
    _write(script, """
        import argparse
        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--pose-dim", type=int, default=0)
            return p.parse_args()
        def build():
            args = parse_args()
            x = getattr(args, "pose_dim", 0)
            return x
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert v == [], v


def test_dead_resolver_satisfied_by_explicit_assignment(tmp_path: Path) -> None:
    """getattr is fine if there's an explicit `args.pose_dim = ...` somewhere."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "good_resolver.py"
    _write(script, """
        import argparse
        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--profile")
            args = p.parse_args()
            args.pose_dim = 6  # resolver
            return args
        def build():
            args = parse_args()
            x = getattr(args, "pose_dim", 0)
            return x
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert v == [], v


def test_dead_resolver_skips_private_underscore_names(tmp_path: Path) -> None:
    """Private-by-convention attrs (start with _) are skipped — usually
    internal state, not profile knobs."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "private.py"
    _write(script, """
        import argparse
        def main():
            p = argparse.ArgumentParser()
            args = p.parse_args()
            x = getattr(args, "_internal_state", None)
            return x
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert v == [], v


def test_dead_resolver_handles_aug_assign(tmp_path: Path) -> None:
    """args.X += ... should also count as a resolver (not just plain =)."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "augassign.py"
    _write(script, """
        import argparse
        def main():
            p = argparse.ArgumentParser()
            args = p.parse_args()
            args.weight = 0
            args.weight += 1
            x = getattr(args, "weight", None)
            return x
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert v == [], v


def test_dead_resolver_multiple_violations_same_file(tmp_path: Path) -> None:
    """Multiple dead getattrs in the same file all surface."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "many_bugs.py"
    _write(script, """
        import argparse
        def main():
            p = argparse.ArgumentParser()
            args = p.parse_args()
            a = getattr(args, "blend_mode", "scalar")
            b = getattr(args, "noise_mode", "deterministic")
            c = getattr(args, "motion_type", "learned_cnn")
            return a, b, c
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert len(v) == 3, v
    assert any("blend_mode" in x for x in v)
    assert any("noise_mode" in x for x in v)
    assert any("motion_type" in x for x in v)


# ── Dead imports ──────────────────────────────────────────────────────────────


def test_dead_import_caught(tmp_path: Path) -> None:
    """from tac.X import Y where Y is not in tac.X — the
    segnet_uncertainty_weighted_loss class."""
    root = _stub_repo(tmp_path)
    _write(root / "src" / "tac" / "losses.py", """
        def real_function():
            return 1
    """)
    script = root / "experiments" / "broken_import.py"
    _write(script, """
        from tac.losses import nonexistent_function
    """)
    v = _scan_python_for_dead_imports(script, root)
    assert len(v) == 1
    assert "nonexistent_function" in v[0]
    assert "DEAD IMPORT" in v[0]


def test_dead_import_satisfied_by_real_definition(tmp_path: Path) -> None:
    """Import resolves cleanly when the name is actually defined."""
    root = _stub_repo(tmp_path)
    _write(root / "src" / "tac" / "losses.py", """
        def real_function():
            return 1
    """)
    script = root / "experiments" / "good_import.py"
    _write(script, """
        from tac.losses import real_function
    """)
    v = _scan_python_for_dead_imports(script, root)
    assert v == [], v


def test_dead_import_skips_try_except_importerror(tmp_path: Path) -> None:
    """Imports inside try/except ImportError are intentional graceful
    fallback patterns — should not be flagged."""
    root = _stub_repo(tmp_path)
    _write(root / "src" / "tac" / "losses.py", "def x(): pass")
    script = root / "experiments" / "graceful.py"
    _write(script, """
        try:
            from tac.losses import maybe_missing
        except ImportError:
            maybe_missing = None
    """)
    v = _scan_python_for_dead_imports(script, root)
    assert v == [], v


def test_dead_import_skips_try_bare_except(tmp_path: Path) -> None:
    """Bare except: also counts as graceful fallback."""
    root = _stub_repo(tmp_path)
    _write(root / "src" / "tac" / "losses.py", "def x(): pass")
    script = root / "experiments" / "bare_except.py"
    _write(script, """
        try:
            from tac.losses import maybe_missing
        except:
            maybe_missing = None
    """)
    v = _scan_python_for_dead_imports(script, root)
    assert v == [], v


def test_dead_import_resolves_submodule(tmp_path: Path) -> None:
    """`from tac.lossless import next_frame_coder` where next_frame_coder is
    a submodule (not a top-level name in __init__.py) should pass."""
    root = _stub_repo(tmp_path)
    (root / "src" / "tac" / "lossless").mkdir()
    _write(root / "src" / "tac" / "lossless" / "__init__.py", "")
    _write(root / "src" / "tac" / "lossless" / "next_frame_coder.py", "x = 1")
    script = root / "experiments" / "submod.py"
    _write(script, """
        from tac.lossless import next_frame_coder
    """)
    v = _scan_python_for_dead_imports(script, root)
    assert v == [], v


def test_dead_import_skips_external_modules(tmp_path: Path) -> None:
    """Only tac.X imports are checked; numpy, torch, etc. are out of scope."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "external.py"
    _write(script, """
        from numpy import does_not_exist_in_numpy
        from torch import maybe_not_a_real_thing
    """)
    v = _scan_python_for_dead_imports(script, root)
    assert v == [], v


# ── Helper unit tests ─────────────────────────────────────────────────────────


def test_module_top_level_names_handles_def_class_assign_reexport(tmp_path: Path) -> None:
    """The name extractor handles all four definition forms."""
    root = _stub_repo(tmp_path)
    mod = root / "src" / "tac" / "sample.py"
    _write(mod, """
        from tac.other import re_exported_thing
        TOP_CONST = 42
        ANN_CONST: int = 99
        def some_func(): pass
        class SomeClass: pass
    """)
    names = _module_top_level_names(mod)
    assert "re_exported_thing" in names
    assert "TOP_CONST" in names
    assert "ANN_CONST" in names
    assert "some_func" in names
    assert "SomeClass" in names


def test_is_resolvable_submodule_detects_pyfile(tmp_path: Path) -> None:
    root = _stub_repo(tmp_path)
    (root / "src" / "tac" / "pkg").mkdir()
    _write(root / "src" / "tac" / "pkg" / "__init__.py", "")
    _write(root / "src" / "tac" / "pkg" / "child.py", "")
    assert _is_resolvable_submodule("tac.pkg", "child", root)
    assert not _is_resolvable_submodule("tac.pkg", "nonchild", root)


# ── End-to-end ────────────────────────────────────────────────────────────────


def test_preflight_dead_resolvers_strict_raises_on_violation(tmp_path: Path) -> None:
    """Strict mode raises DeadResolverViolation, listing every violation."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "broken.py"
    _write(script, """
        import argparse
        def main():
            p = argparse.ArgumentParser()
            args = p.parse_args()
            x = getattr(args, "missing_resolver", 0)
            return x
    """)
    with pytest.raises(DeadResolverViolation) as ei:
        preflight_dead_resolvers(
            repo_root=root,
            target_dirs=["experiments"],
            strict=True,
            verbose=False,
        )
    assert "missing_resolver" in str(ei.value)
    assert "DEAD RESOLVER" in str(ei.value)


def test_preflight_dead_resolvers_warn_only_returns_violations(tmp_path: Path) -> None:
    """Non-strict mode returns the violation list without raising."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "broken.py"
    _write(script, """
        import argparse
        def main():
            p = argparse.ArgumentParser()
            args = p.parse_args()
            x = getattr(args, "missing_resolver", 0)
            return x
    """)
    v = preflight_dead_resolvers(
        repo_root=root,
        target_dirs=["experiments"],
        strict=False,
        verbose=False,
    )
    assert len(v) == 1
    assert "missing_resolver" in v[0]


def test_preflight_dead_resolvers_clean_repo_passes(tmp_path: Path) -> None:
    """A repo with no violations returns an empty list and does not raise."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "clean.py"
    _write(script, """
        import argparse
        def main():
            p = argparse.ArgumentParser()
            p.add_argument("--profile")
            args = p.parse_args()
            return args.profile
    """)
    v = preflight_dead_resolvers(
        repo_root=root,
        target_dirs=["experiments"],
        strict=True,
        verbose=False,
    )
    assert v == []


# ── Real-world regression ─────────────────────────────────────────────────────


def test_pose_dim_dead_resolver_no_longer_in_real_train_renderer() -> None:
    """The 2026-04-27 R5 incidental fix added a pose_dim resolver to
    train_renderer.py:parse_args. Pin that the scanner no longer flags
    pose_dim — would have flagged it pre-fix."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    script = repo_root / "src" / "tac" / "experiments" / "train_renderer.py"
    if not script.exists():
        pytest.skip("train_renderer.py not present in this checkout")
    v = _scan_python_for_dead_resolvers(script, repo_root)
    # Filter on the literal getattr call signature, not the bug-class
    # reference text in the suffix (which contains "pose_dim").
    pose_dim_violations = [x for x in v if "getattr(args, 'pose_dim'" in x]
    assert pose_dim_violations == [], (
        f"pose_dim should be resolved post-R5 but scanner flagged: "
        f"{pose_dim_violations}"
    )


def test_segnet_uncertainty_weighted_loss_no_longer_dead_in_train_renderer() -> None:
    """The 2026-04-27 R5 incidental fix added segnet_uncertainty_weighted_loss
    to tac.losses. Pin that the scanner no longer flags it as a dead import."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    script = repo_root / "src" / "tac" / "experiments" / "train_renderer.py"
    if not script.exists():
        pytest.skip("train_renderer.py not present in this checkout")
    v = _scan_python_for_dead_imports(script, repo_root)
    # Filter on the literal `imports 'NAME' from ...` form, not the
    # bug-class reference text in the suffix.
    target = [x for x in v if "imports 'segnet_uncertainty_weighted_loss'" in x]
    assert target == [], (
        f"segnet_uncertainty_weighted_loss should be defined in tac.losses "
        f"post-R5 but scanner flagged: {target}"
    )
