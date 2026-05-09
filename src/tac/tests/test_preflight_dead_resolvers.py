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
    _is_resolvable_submodule,
    _module_top_level_names,
    _scan_python_for_dead_imports,
    _scan_python_for_dead_objective_feature,
    _scan_python_for_dead_resolvers,
    check_feature_flags_have_live_objective_effect,
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


def test_dead_resolver_satisfied_by_argparse_subclass_flag(tmp_path: Path) -> None:
    """Strict parser subclasses are still argparse parser instances."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "good_argparse_subclass.py"
    _write(script, """
        import argparse
        class StrictArgumentParser(argparse.ArgumentParser):
            pass
        def parse_args():
            p = StrictArgumentParser()
            p.add_argument("--pose-dim", type=int, default=0)
            return p.parse_args()
        def build():
            args = parse_args()
            x = getattr(args, "pose_dim", 0)
            return x
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert v == [], v


def test_dead_resolver_satisfied_by_argparse_dest_override(tmp_path: Path) -> None:
    """Explicit argparse dest= creates the Namespace attr even when the flag
    spelling differs from the getattr name."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "good_dest.py"
    _write(script, """
        import argparse
        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--lane", dest="lane_id")
            return p.parse_args()
        def build():
            args = parse_args()
            return getattr(args, "lane_id", None)
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert v == [], v


def test_dead_resolver_flags_wrong_attr_when_argparse_dest_renames_it(tmp_path: Path) -> None:
    """A flag with dest= does not also create the flag-spelled attr."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "bad_dest_flag_attr.py"
    _write(script, """
        import argparse
        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--lane", dest="lane_id")
            return p.parse_args()
        def build():
            args = parse_args()
            return getattr(args, "lane", None)
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert len(v) == 1, v
    assert "lane" in v[0]
    assert "DEAD RESOLVER" in v[0]


def test_dead_resolver_satisfied_by_subparser_dest(tmp_path: Path) -> None:
    """argparse add_subparsers(dest='cmd') is a real Namespace attr; the
    scanner must not force launchers to assign args.cmd by hand."""
    root = _stub_repo(tmp_path)
    script = root / "scripts" / "good_subparser_dest.py"
    _write(script, """
        import argparse
        def parse_args():
            p = argparse.ArgumentParser()
            sub = p.add_subparsers(dest="cmd", required=True)
            sub.add_parser("exact-eval")
            return p.parse_args()
        def build():
            args = parse_args()
            return getattr(args, "cmd", None)
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert v == [], v


def test_dead_resolver_satisfied_by_argparse_set_defaults(tmp_path: Path) -> None:
    """set_defaults(func=...) creates the attr selected by subcommands."""
    root = _stub_repo(tmp_path)
    script = root / "scripts" / "good_set_defaults.py"
    _write(script, """
        import argparse
        def parse_args():
            p = argparse.ArgumentParser()
            sub = p.add_subparsers(dest="cmd", required=True)
            exact = sub.add_parser("exact-eval")
            exact.set_defaults(func=lambda args: 0)
            return p.parse_args()
        def build():
            args = parse_args()
            return getattr(args, "func", None)
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert v == [], v


def test_dead_resolver_ignores_non_argparse_set_defaults_lookalike(
    tmp_path: Path,
) -> None:
    """A random object's set_defaults() cannot prove argparse creates args.X."""
    root = _stub_repo(tmp_path)
    script = root / "scripts" / "bad_set_defaults_lookalike.py"
    _write(script, """
        import argparse
        class Config:
            def set_defaults(self, **kwargs):
                pass
        def parse_args():
            p = argparse.ArgumentParser()
            config = Config()
            config.set_defaults(pose_dim=6)
            return p.parse_args()
        def build():
            args = parse_args()
            return getattr(args, "pose_dim", None)
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert len(v) == 1, v
    assert "pose_dim" in v[0]


def test_dead_resolver_ignores_non_argparse_add_argument_lookalike(
    tmp_path: Path,
) -> None:
    """A builder.add_argument() helper cannot prove argparse creates args.X."""
    root = _stub_repo(tmp_path)
    script = root / "scripts" / "bad_add_argument_lookalike.py"
    _write(script, """
        import argparse
        class Builder:
            def add_argument(self, *args, **kwargs):
                pass
        def parse_args():
            p = argparse.ArgumentParser()
            builder = Builder()
            builder.add_argument("--pose-dim")
            return p.parse_args()
        def build():
            args = parse_args()
            return getattr(args, "pose_dim", None)
    """)
    v = _scan_python_for_dead_resolvers(script, root)
    assert len(v) == 1, v
    assert "pose_dim" in v[0]


def test_dead_resolver_satisfied_by_positional_argparse_attr(tmp_path: Path) -> None:
    """Positional argparse arguments also create Namespace attrs."""
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "good_positional.py"
    _write(script, """
        import argparse
        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("archive_path")
            return p.parse_args()
        def build():
            args = parse_args()
            return getattr(args, "archive_path", None)
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


def test_dead_import_fast_path_handles_alias_and_multiple_names(tmp_path: Path) -> None:
    root = _stub_repo(tmp_path)
    _write(root / "src" / "tac" / "losses.py", """
        def real_function():
            return 1
    """)
    script = root / "experiments" / "alias_import.py"
    _write(script, """
        from tac.losses import real_function as renamed, missing_function
    """)

    v = _scan_python_for_dead_imports(script, root)

    assert len(v) == 1, v
    assert "missing_function" in v[0]
    assert "real_function" not in v[0]


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


def test_preflight_dead_resolvers_cache_invalidates_changed_target(
    tmp_path: Path,
) -> None:
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "cached.py"
    _write(script, """
        import argparse
        def main():
            p = argparse.ArgumentParser()
            p.add_argument("--profile")
            args = p.parse_args()
            return args.profile
    """)
    assert preflight_dead_resolvers(
        repo_root=root,
        target_dirs=["experiments"],
        strict=True,
        verbose=False,
    ) == []
    assert (root / ".omx" / "cache" / "dead_resolvers_clean.json").exists()

    _write(script, """
        import argparse
        def main():
            p = argparse.ArgumentParser()
            args = p.parse_args()
            return getattr(args, "missing_resolver", 0)
    """)
    violations = preflight_dead_resolvers(
        repo_root=root,
        target_dirs=["experiments"],
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "missing_resolver" in violations[0]


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


def test_variance_noise_flag_has_live_train_renderer_callsite() -> None:
    """Guard the second-order dead-feature class: resolver exists, but the
    feature is never applied in the training objective."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    script = repo_root / "src" / "tac" / "experiments" / "train_renderer.py"
    src = script.read_text()
    assert "uniward_quant_noise_loss" in src
    assert "args.use_variance_noise and args.variance_noise_weight > 0" in src
    assert "loss = loss + args.variance_noise_weight * variance_loss" in src


def test_variance_noise_flag_has_live_train_distill_callsite() -> None:
    """The distillation entrypoint had the same live flag but missing helper
    problem. Keep both training surfaces protected."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    script = repo_root / "experiments" / "train_distill.py"
    src = script.read_text()
    assert "uniward_quant_noise_loss" in src
    assert "cfg.use_variance_noise and cfg.variance_noise_weight > 0" in src
    assert "cfg.variance_noise_weight * variance_loss" in src


def test_dead_objective_feature_scanner_catches_guarded_notimplemented(tmp_path: Path) -> None:
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "broken_feature.py"
    _write(script, """
        def train(cfg):
            total = 0
            if cfg.use_variance_noise and cfg.variance_noise_weight > 0:
                raise NotImplementedError("not wired")
            return total
    """)
    violations = _scan_python_for_dead_objective_feature(
        script,
        root,
        feature_attr="use_variance_noise",
        weight_attr="variance_noise_weight",
        function_name="uniward_quant_noise_loss",
    )
    assert any("NotImplementedError" in v for v in violations)
    assert any("does not call" in v for v in violations)
    assert any("does not add" in v for v in violations)


def test_dead_objective_feature_scanner_rejects_helper_noop(tmp_path: Path) -> None:
    root = _stub_repo(tmp_path)
    script = root / "experiments" / "noop_feature.py"
    _write(script, """
        def uniward_quant_noise_loss(a, b):
            return 1
        def train(args):
            loss = 0
            if args.use_variance_noise and args.variance_noise_weight > 0:
                variance_loss = uniward_quant_noise_loss(1, 2)
                print(variance_loss)
            return loss
    """)
    violations = _scan_python_for_dead_objective_feature(
        script,
        root,
        feature_attr="use_variance_noise",
        weight_attr="variance_noise_weight",
        function_name="uniward_quant_noise_loss",
    )
    assert any("does not add" in v for v in violations)


def test_feature_flags_have_live_objective_effect_real_codebase() -> None:
    violations = check_feature_flags_have_live_objective_effect(
        strict=False,
        verbose=False,
    )
    assert violations == []


# ── Codex R5-2 Finding #2 strict-mode flip regressions ───────────────────────
#
# The 2026-04-27 codex R5-2 review surfaced the scanner running with
# strict=False inside preflight_all() despite 19 known violations, defeating
# the entire point of the scanner. After fixing every violation, preflight_all
# was flipped to strict=True. The tests below pin both halves of that flip:
# (1) the live codebase has zero violations and (2) preflight_all() actually
# calls the scanner in strict mode.


def test_preflight_dead_resolvers_strict_passes_on_real_codebase() -> None:
    """Real-world contract: the live codebase MUST scan clean under
    `strict=True`. A failure here means a new resolver dropped a profile knob
    silently OR a refactor renamed a tac.X export — both are the same bug
    class as pose_dim / luma_local_variance / uniward_quant_noise_loss.

    Codex R5-2 Finding #2 (2026-04-27): all 19 known violations were fixed
    before flipping preflight_all() to strict=True; this test guards against
    regression."""
    violations = preflight_dead_resolvers(strict=False, verbose=False)
    assert violations == [], (
        f"Codex R5-2 Finding #2: {len(violations)} dead-resolver/dead-import "
        f"violations in the live codebase. preflight_all() runs the scanner in "
        f"strict mode now, so any violation BLOCKS launch:\n\n"
        + "\n".join(f"  • {v}" for v in violations)
    )


def test_preflight_all_invokes_dead_resolvers_strict() -> None:
    """Source-level pin: preflight_all MUST pass strict=True to
    preflight_dead_resolvers. If someone reverts to strict=False (or removes
    the call), the silent-default bug class regresses to warn-only and the
    scanner's value collapses."""
    import inspect

    from tac.preflight import preflight_all
    src = inspect.getsource(preflight_all)
    # The literal call must appear with strict=True. We don't trust comment
    # text alone — a regex match against the actual call.
    import re
    matches = re.findall(
        r"preflight_dead_resolvers\s*\(\s*strict\s*=\s*(True|False)",
        src,
    )
    assert matches, (
        "preflight_all does not call preflight_dead_resolvers — codex R5-2 "
        "Finding #2 regressed (or the call was deleted)."
    )
    assert all(m == "True" for m in matches), (
        f"preflight_all calls preflight_dead_resolvers with strict={matches!r}; "
        f"codex R5-2 Finding #2 requires strict=True (warn-only defeats the "
        f"silent-default bug class the scanner exists to catch)."
    )
