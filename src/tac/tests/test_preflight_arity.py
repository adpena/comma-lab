"""Tests for preflight_arity, preflight_profiles, and codebase drift checks.

These tests pin down the exact bug class (SHIRAZ A100 disaster) so it cannot
silently regress: a launcher passes flags that don't exist on the target, or
forgets a required architectural flag, or a profile is missing a key.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    ARCH_FLAGS_BOOLEAN,
    ARCH_FLAGS_REQUIRED,
    ArityViolation,
    PreflightError,
    _build_target_signatures,
    _parse_argparse_signature,
    _scan_launcher_invocations,
    preflight_arity,
    preflight_profiles,
)


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip())


def _stub_repo(tmp_path: Path) -> Path:
    """Create a tiny repo with experiments/ and scripts/ directories."""
    (tmp_path / "experiments").mkdir()
    (tmp_path / "scripts").mkdir()
    return tmp_path


# ── _parse_argparse_signature ─────────────────────────────────────────────────

def test_parse_argparse_finds_all_flags(tmp_path: Path) -> None:
    target = tmp_path / "qat.py"
    _write(target, """
        import argparse
        def main():
            p = argparse.ArgumentParser()
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--motion-hidden", type=int, default=32)
            p.add_argument("--use-dsconv", action="store_true")
            args = p.parse_args()
    """)
    sig = _parse_argparse_signature(target)
    assert sig is not None
    assert "--checkpoint" in sig and sig["--checkpoint"]["required"] is True
    assert "--motion-hidden" in sig and sig["--motion-hidden"]["required"] is False
    assert "--use-dsconv" in sig and sig["--use-dsconv"]["action"] == "store_true"


def test_parse_argparse_returns_none_for_no_argparse(tmp_path: Path) -> None:
    target = tmp_path / "no_argparse.py"
    target.write_text("def f(): return 42\n")
    assert _parse_argparse_signature(target) is None


# ── _scan_launcher_invocations ────────────────────────────────────────────────

def test_scanner_handles_cmd_extend_pattern(tmp_path: Path) -> None:
    """Launchers commonly do cmd = [...]; if x: cmd.extend([...]). Scanner must follow."""
    launcher = tmp_path / "launcher.py"
    _write(launcher, """
        import subprocess
        def run():
            cmd = [
                "python", "-u", "experiments/qat.py",
                "--checkpoint", "x",
            ]
            cmd.extend(["--motion-hidden", "24"])
            cmd.append("--use-dsconv")
            subprocess.run(cmd)
    """)
    invocations, _ = _scan_launcher_invocations(launcher)
    assert len(invocations) == 1
    _, target, flags = invocations[0]
    assert target == "experiments/qat.py"
    assert "--checkpoint" in flags
    assert "--motion-hidden" in flags
    assert "--use-dsconv" in flags


def test_scanner_skips_calls_without_target_script(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher.py"
    _write(launcher, """
        import subprocess
        def run():
            subprocess.run(["ls", "-la"])
            subprocess.run(["git", "status"])
    """)
    invocations, _ = _scan_launcher_invocations(launcher)
    assert invocations == []


def test_scanner_isolates_per_function_scope(tmp_path: Path) -> None:
    """Round 23 CRITICAL: cross-function `cmd` variables must not pollute each other.

    Function A binds cmd to a list invoking target_a.py.
    Function B binds cmd (same name!) and extends it with an arch flag.
    The scanner must NOT report function A's invocation as carrying B's flag.
    """
    launcher = tmp_path / "launcher.py"
    _write(launcher, """
        import subprocess
        def launch_a():
            cmd = ["python", "experiments/target_a.py", "--checkpoint", "a"]
            subprocess.run(cmd)
        def launch_b():
            cmd = ["python", "experiments/target_b.py"]
            cmd.extend(["--motion-hidden", "24"])
            subprocess.run(cmd)
    """)
    invocations, _ = _scan_launcher_invocations(launcher)
    assert len(invocations) == 2
    by_target = {t: f for _, t, f in invocations}
    assert "--motion-hidden" not in by_target["experiments/target_a.py"]
    assert "--motion-hidden" in by_target["experiments/target_b.py"]


def test_scanner_walks_top_level_subprocess_calls(tmp_path: Path) -> None:
    """Top-level subprocess calls (not inside functions) must still be detected."""
    launcher = tmp_path / "launcher.py"
    _write(launcher, """
        import subprocess
        subprocess.run(["python", "experiments/foo.py", "--checkpoint", "x"])
    """)
    invocations, _ = _scan_launcher_invocations(launcher)
    assert len(invocations) == 1
    assert invocations[0][1] == "experiments/foo.py"


def test_scanner_detects_bash_c_wrapped_invocation(tmp_path: Path) -> None:
    """bash -c "python experiments/x.py --foo" must not be a blind spot."""
    launcher = tmp_path / "launcher.py"
    _write(launcher, """
        import subprocess
        subprocess.run(["bash", "-c", "python experiments/foo.py --checkpoint x --bar y"])
    """)
    invocations, _ = _scan_launcher_invocations(launcher)
    assert len(invocations) == 1
    _, target, flags = invocations[0]
    assert target == "experiments/foo.py"
    assert "--checkpoint" in flags
    assert "--bar" in flags


def test_argparse_short_form_alias_indexed(tmp_path: Path) -> None:
    """add_argument('-m', '--motion-hidden') must register --motion-hidden."""
    target = tmp_path / "qat.py"
    _write(target, """
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("-c", "--checkpoint", required=True)
        p.add_argument("-m", "--motion-hidden", type=int, default=32)
        args = p.parse_args()
    """)
    sig = _parse_argparse_signature(target)
    assert sig is not None
    assert "--checkpoint" in sig
    assert "--motion-hidden" in sig
    # Short forms are not indexed (intentional — launchers always use long form)
    assert "-c" not in sig
    assert "-m" not in sig


# ── preflight_arity ───────────────────────────────────────────────────────────

def test_arity_passes_when_all_flags_match(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/target.py", """
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--device", default="cuda")
        args = p.parse_args()
    """)
    _write(repo / "experiments/launcher.py", """
        import subprocess
        def go():
            subprocess.run([
                "python", "experiments/target.py",
                "--checkpoint", "x", "--device", "cuda",
            ])
    """)
    violations = preflight_arity(
        repo_root=repo,
        launcher_files=["experiments/launcher.py"],
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_arity_catches_unknown_flag(tmp_path: Path) -> None:
    """Rule A: launcher passes a flag the target doesn't accept."""
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/target.py", """
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint", required=True)
        args = p.parse_args()
    """)
    _write(repo / "experiments/launcher.py", """
        import subprocess
        subprocess.run([
            "python", "experiments/target.py",
            "--checkpoint", "x", "--nonexistent-flag", "y",
        ])
    """)
    with pytest.raises(ArityViolation, match="--nonexistent-flag"):
        preflight_arity(
            repo_root=repo,
            launcher_files=["experiments/launcher.py"],
            strict=True,
            verbose=False,
        )


def test_arity_catches_missing_required(tmp_path: Path) -> None:
    """Rule B: launcher omits a required arg."""
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/target.py", """
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--device", default="cuda")
        args = p.parse_args()
    """)
    _write(repo / "experiments/launcher.py", """
        import subprocess
        subprocess.run(["python", "experiments/target.py", "--device", "cuda"])
    """)
    with pytest.raises(ArityViolation, match="--checkpoint"):
        preflight_arity(
            repo_root=repo,
            launcher_files=["experiments/launcher.py"],
            strict=True,
            verbose=False,
        )


def test_arity_catches_silent_arch_default_the_shiraz_disaster(tmp_path: Path) -> None:
    """Rule C: target accepts --motion-hidden but launcher doesn't pass it.

    This is the SHIRAZ A100 failure: profile said motion_hidden=24, qat_finetune.py
    silently used default=32, QAT trained the wrong architecture. We must catch
    this AT PREFLIGHT, not after wasting GPU dollars.
    """
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/qat.py", """
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--motion-hidden", type=int, default=32)
        p.add_argument("--depth", type=int, default=1)
        args = p.parse_args()
    """)
    _write(repo / "experiments/launcher.py", """
        import subprocess
        subprocess.run([
            "python", "experiments/qat.py",
            "--checkpoint", "x",
            # Note: --motion-hidden and --depth NOT passed → silent defaults
        ])
    """)
    with pytest.raises(ArityViolation, match="--motion-hidden"):
        preflight_arity(
            repo_root=repo,
            launcher_files=["experiments/launcher.py"],
            strict=True,
            verbose=False,
        )


def test_arity_skips_target_with_no_argparse(tmp_path: Path) -> None:
    """If a target script has no argparse usage, we can't validate against it."""
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/util.py", """
        def main(): pass
    """)
    _write(repo / "experiments/launcher.py", """
        import subprocess
        subprocess.run(["python", "experiments/util.py", "--anything"])
    """)
    # Should not raise — no argparse means we can't tell what's valid
    violations = preflight_arity(
        repo_root=repo,
        launcher_files=["experiments/launcher.py"],
        strict=True,
        verbose=False,
    )
    assert violations == []


# ── _build_target_signatures ──────────────────────────────────────────────────

def test_arity_rule_d_catches_boolean_flag_never_mentioned(tmp_path: Path) -> None:
    """Round 23: target accepts --use-dsconv, launcher source NEVER mentions it.

    The launcher cannot conditionally pass a flag whose name doesn't appear
    anywhere in its source — that's a guaranteed silent-default risk.
    """
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/qat.py", """
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--base-ch", type=int, default=36)
        p.add_argument("--mid-ch", type=int, default=60)
        p.add_argument("--motion-hidden", type=int, default=32)
        p.add_argument("--depth", type=int, default=1)
        p.add_argument("--embed-dim", type=int, default=6)
        p.add_argument("--pose-dim", type=int, default=6)
        p.add_argument("--padding-mode", type=str, default="zeros")
        p.add_argument("--use-dsconv", action="store_true")
        args = p.parse_args()
    """)
    _write(repo / "experiments/launcher.py", """
        import subprocess
        # Launcher passes every required arch flag conditionally, but the
        # word "use-dsconv" appears NOWHERE in this source — silent-default.
        subprocess.run([
            "python", "experiments/qat.py",
            "--checkpoint", "x",
            "--base-ch", "32", "--mid-ch", "48",
            "--motion-hidden", "24", "--depth", "1", "--embed-dim", "6",
            "--pose-dim", "6", "--padding-mode", "replicate",
        ])
    """)
    with pytest.raises(ArityViolation, match="--use-dsconv"):
        preflight_arity(
            repo_root=repo,
            launcher_files=["experiments/launcher.py"],
            strict=True,
            verbose=False,
        )


def test_arity_rule_d_passes_when_boolean_flag_mentioned_conditionally(tmp_path: Path) -> None:
    """Conditional pass of boolean flag (if cfg.use_dsconv: append) is OK."""
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/qat.py", """
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--base-ch", type=int, default=36)
        p.add_argument("--mid-ch", type=int, default=60)
        p.add_argument("--motion-hidden", type=int, default=32)
        p.add_argument("--depth", type=int, default=1)
        p.add_argument("--embed-dim", type=int, default=6)
        p.add_argument("--pose-dim", type=int, default=6)
        p.add_argument("--padding-mode", type=str, default="zeros")
        p.add_argument("--use-dsconv", action="store_true")
        args = p.parse_args()
    """)
    _write(repo / "experiments/launcher.py", """
        import subprocess
        cmd = [
            "python", "experiments/qat.py",
            "--checkpoint", "x",
            "--base-ch", "32", "--mid-ch", "48",
            "--motion-hidden", "24", "--depth", "1", "--embed-dim", "6",
            "--pose-dim", "6", "--padding-mode", "replicate",
        ]
        if True:
            cmd.append("--use-dsconv")  # source mentions the flag → Rule D OK
        subprocess.run(cmd)
    """)
    violations = preflight_arity(
        repo_root=repo,
        launcher_files=["experiments/launcher.py"],
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_profiles_validator_fails_on_missing_experiment_type(monkeypatch) -> None:
    """Round 23: profile without experiment_type silently skipped → must fail."""
    import tac.profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "PROFILES", {
        "broken": {"base_ch": 32, "mid_ch": 48, "depth": 1, "pose_dim": 6,
                   "padding_mode": "zeros", "eval_roundtrip": True},
    })
    with pytest.raises(PreflightError, match="experiment_type"):
        preflight_profiles(strict=True, verbose=False)


def test_profiles_validator_fails_on_unknown_experiment_type(monkeypatch) -> None:
    import tac.profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "PROFILES", {
        "weird": {"experiment_type": "tofu_distillation"},
    })
    with pytest.raises(PreflightError, match="unknown experiment_type"):
        preflight_profiles(strict=True, verbose=False)


def test_apply_profile_loads_overlapping_keys(monkeypatch) -> None:
    """_apply_profile must populate args from profile for overlapping keys."""
    import argparse
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(
        "_pipeline_under_test",
        Path(__file__).resolve().parent.parent.parent.parent / "experiments" / "pipeline.py",
    )
    pipeline = importlib.util.module_from_spec(spec)
    _sys.modules["_pipeline_under_test"] = pipeline
    spec.loader.exec_module(pipeline)

    ns = argparse.Namespace(base_ch=36, mid_ch=60, use_dsconv=False, padding_mode="zeros")
    pipeline._apply_profile(ns, "shiraz", user_provided_flags=set())
    assert ns.base_ch == 32
    assert ns.mid_ch == 48
    assert ns.use_dsconv is True
    assert ns.padding_mode == "replicate"


def test_apply_profile_user_flags_win_over_profile(monkeypatch) -> None:
    """When user passes --base-ch 999, the profile's base_ch=32 must NOT win.

    R26 finding: prior version overwrote CLI flags with profile values
    despite the comment claiming the opposite.
    """
    import argparse
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(
        "_pipeline_under_test2",
        Path(__file__).resolve().parent.parent.parent.parent / "experiments" / "pipeline.py",
    )
    pipeline = importlib.util.module_from_spec(spec)
    _sys.modules["_pipeline_under_test2"] = pipeline
    spec.loader.exec_module(pipeline)

    ns = argparse.Namespace(base_ch=999, mid_ch=60, use_dsconv=False, padding_mode="zeros")
    pipeline._apply_profile(ns, "shiraz", user_provided_flags={"base_ch"})
    # User override survives; non-overridden fields take profile values
    assert ns.base_ch == 999
    assert ns.mid_ch == 48
    assert ns.padding_mode == "replicate"


def test_apply_profile_unknown_key_fails_loudly(monkeypatch) -> None:
    """Typo'd profile key must SystemExit, not silently drop (the SHIRAZ class)."""
    import argparse
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(
        "_pipeline_under_test3",
        Path(__file__).resolve().parent.parent.parent.parent / "experiments" / "pipeline.py",
    )
    pipeline = importlib.util.module_from_spec(spec)
    _sys.modules["_pipeline_under_test3"] = pipeline
    spec.loader.exec_module(pipeline)

    import tac.profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "PROFILES", {
        "typo_profile": {
            "experiment_type": "renderer_training",
            "base_ch": 32,
            "motino_hidden": 24,  # TYPO: should be motion_hidden
        },
    })
    ns = argparse.Namespace(base_ch=36)
    with pytest.raises(SystemExit, match="motino_hidden"):
        pipeline._apply_profile(ns, "typo_profile", user_provided_flags=set())


def test_apply_profile_unknown_profile_name_fails_loudly() -> None:
    """Profile name not in PROFILES must SystemExit."""
    import argparse
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(
        "_pipeline_under_test4",
        Path(__file__).resolve().parent.parent.parent.parent / "experiments" / "pipeline.py",
    )
    pipeline = importlib.util.module_from_spec(spec)
    _sys.modules["_pipeline_under_test4"] = pipeline
    spec.loader.exec_module(pipeline)

    ns = argparse.Namespace()
    with pytest.raises(SystemExit, match="unknown --profile"):
        pipeline._apply_profile(ns, "does_not_exist", user_provided_flags=set())


def _load_pipeline_module(name: str = "_pipeline_under_test_shared"):
    """Helper: import experiments/pipeline.py as a module via spec_from_file_location."""
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(
        name,
        Path(__file__).resolve().parent.parent.parent.parent / "experiments" / "pipeline.py",
    )
    pipeline = importlib.util.module_from_spec(spec)
    _sys.modules[name] = pipeline
    spec.loader.exec_module(pipeline)
    return pipeline


def test_user_provided_flags_detects_explicit_cli_arg() -> None:
    """The function that powers '--profile X --base-ch 999 wins' must work.

    R27 finding: this function had ZERO test coverage; if it silently returned
    an empty set, every CLI override would be clobbered by the profile.
    """
    import argparse
    pipeline = _load_pipeline_module("_pipeline_uvf_1")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    comp = sub.add_parser("compress")
    comp.add_argument("--base-ch", type=int, default=36)
    comp.add_argument("--profile", default=None)
    flags = pipeline._user_provided_flags(
        parser, ["compress", "--base-ch", "999", "--profile", "shiraz"],
        active_subcommand="compress",
    )
    assert "base_ch" in flags
    assert "profile" in flags


def test_user_provided_flags_handles_equals_form() -> None:
    """`--base-ch=999` must be detected just like `--base-ch 999`."""
    import argparse
    pipeline = _load_pipeline_module("_pipeline_uvf_2")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    comp = sub.add_parser("compress")
    comp.add_argument("--base-ch", type=int, default=36)
    flags = pipeline._user_provided_flags(parser, ["compress", "--base-ch=999"],
                                           active_subcommand="compress")
    assert "base_ch" in flags


def test_user_provided_flags_does_not_walk_inactive_subparsers() -> None:
    """If two subparsers share an option string with different dest, walking
    BOTH would mark the wrong dest. Scope to active subcommand only.
    """
    import argparse
    pipeline = _load_pipeline_module("_pipeline_uvf_3")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    comp = sub.add_parser("compress")
    comp.add_argument("--out", dest="output_dir")
    ev = sub.add_parser("eval")
    ev.add_argument("--out", dest="out_dir")  # different dest!
    flags = pipeline._user_provided_flags(parser, ["compress", "--out", "X"],
                                           active_subcommand="compress")
    assert "output_dir" in flags
    assert "out_dir" not in flags


def test_capture_provenance_returns_required_fields() -> None:
    """Provenance JSON must include config + timestamp at minimum, and
    torch_version when torch is importable."""
    pipeline = _load_pipeline_module("_pipeline_prov_1")
    cfg = pipeline.PipelineConfig()
    prov = pipeline._capture_provenance(cfg)
    assert "config" in prov
    assert "timestamp_utc" in prov
    assert prov["config"]["device"] == "cuda"  # default
    # torch is a hard dep of this repo; should always be present
    assert "torch_version" in prov


def test_capture_provenance_handles_non_git_dir(tmp_path, monkeypatch) -> None:
    """If git rev-parse fails, the function must not raise — just record the error."""
    pipeline = _load_pipeline_module("_pipeline_prov_2")
    # Force git to fail by giving it a bogus cwd via subprocess.check_output
    # monkey: we patch subprocess.check_output to raise for git calls
    import subprocess as _sp
    real_check = _sp.check_output
    def fake_check(cmd, **kw):
        if cmd and cmd[0] == "git":
            raise FileNotFoundError("git not found (test)")
        return real_check(cmd, **kw)
    monkeypatch.setattr(pipeline.subprocess, "check_output", fake_check)
    prov = pipeline._capture_provenance(pipeline.PipelineConfig())
    assert "git_hash_error" in prov
    # Still has timestamp + config — not crashed
    assert "config" in prov
    assert "timestamp_utc" in prov


def test_extract_eval_score_ignores_timing_lines() -> None:
    """`elapsed 45.2s` must NOT be parsed as a score. Only score-line patterns count."""
    pipeline = _load_pipeline_module("_pipeline_eval_1")
    stdout = "\n".join([
        "Loading scorer models...",
        "Pair 5/600 elapsed 45.2s",
        "Pair 100/600 elapsed 12.3s",
        "TOTAL 2.014",
        "All done in 67.8 seconds",
    ])
    assert pipeline._extract_eval_score(stdout) == 2.014


def test_extract_eval_score_returns_none_when_no_score_line() -> None:
    pipeline = _load_pipeline_module("_pipeline_eval_2")
    stdout = "\n".join(["Loading models...", "Pair 5/600 elapsed 45.2s"])
    assert pipeline._extract_eval_score(stdout) is None


def test_all_renderer_profiles_load_without_typo_errors() -> None:
    """Every PROFILES entry must pass _apply_profile without SystemExit.

    R27 finding: the prior typo-allowlist was incomplete; ~30 profiles
    would crash _apply_profile with 'unknown keys'. The new Levenshtein
    close-match approach should accept all of them.
    """
    import argparse
    pipeline = _load_pipeline_module("_pipeline_profile_matrix")
    from tac.profiles import PROFILES
    failures = []
    for name in PROFILES:
        ns = argparse.Namespace()
        try:
            pipeline._apply_profile(ns, name, user_provided_flags=set())
        except SystemExit as e:
            failures.append(f"{name}: {e}")
    assert not failures, "Profiles failed _apply_profile:\n" + "\n".join(failures)


def test_apply_profile_catches_levenshtein_typos() -> None:
    """Typo close to a real PipelineConfig field still fails loudly."""
    import argparse
    pipeline = _load_pipeline_module("_pipeline_profile_typo")
    import tac.profiles as profiles_mod
    saved = profiles_mod.PROFILES.copy()
    try:
        profiles_mod.PROFILES["typo_close"] = {
            "experiment_type": "renderer_training",
            "motino_hidden": 24,  # close to "motion_hidden"
        }
        with pytest.raises(SystemExit, match="motion_hidden"):
            pipeline._apply_profile(argparse.Namespace(), "typo_close",
                                    user_provided_flags=set())
    finally:
        profiles_mod.PROFILES.clear()
        profiles_mod.PROFILES.update(saved)


def test_arch_flag_constants_are_disjoint() -> None:
    """ARCH_FLAGS_REQUIRED and ARCH_FLAGS_BOOLEAN must not overlap.

    A flag must be classified as either value-bearing (Rule C) or boolean
    (Rule D) — overlap would double-fire.
    """
    overlap = ARCH_FLAGS_REQUIRED & ARCH_FLAGS_BOOLEAN
    assert overlap == set(), f"flags appear in both sets: {overlap}"


def test_build_target_signatures_indexes_all_argparse_scripts(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/a.py", """
        import argparse
        argparse.ArgumentParser().add_argument("--foo")
    """)
    _write(repo / "experiments/b.py", """
        import argparse
        argparse.ArgumentParser().add_argument("--bar")
    """)
    _write(repo / "experiments/no_args.py", "x = 1\n")
    sigs = _build_target_signatures(repo)
    assert "experiments/a.py" in sigs
    assert "experiments/b.py" in sigs
    assert "experiments/no_args.py" not in sigs
