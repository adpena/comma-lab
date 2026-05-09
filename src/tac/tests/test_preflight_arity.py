"""Tests for preflight_arity, preflight_profiles, and codebase drift checks.

These tests pin down the exact bug class (SHIRAZ A100 disaster) so it cannot
silently regress: a launcher passes flags that don't exist on the target, or
forgets a required architectural flag, or a profile is missing a key.
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    ARCH_FLAGS_BOOLEAN,
    ARCH_FLAGS_REQUIRED,
    ArityViolation,
    FilenameContractError,
    PreflightError,
    _build_target_signatures,
    _extract_artifact_filenames,
    _parse_argparse_signature,
    _scan_launcher_invocations,
    preflight_arity,
    preflight_filename_contract,
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


def test_extract_artifact_filenames_skips_ast_when_text_has_no_artifact_suffix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "producer.py"
    path.write_text("VALUE = 'ordinary source with no artifact literals'\n")

    class Index:
        def read_text(self, target: Path) -> str:
            assert target == path
            return path.read_text()

        def python_ast(self, target: Path) -> ast.AST:
            raise AssertionError("artifact-free source should not be AST parsed")

    assert _extract_artifact_filenames(path, source_index=Index()) == set()


def test_extract_artifact_filenames_still_parses_when_suffix_is_present(
    tmp_path: Path,
) -> None:
    path = tmp_path / "producer.py"
    path.write_text("OUTPUT = 'custom_renderer_blob.bin'\n")

    class Index:
        def read_text(self, target: Path) -> str:
            assert target == path
            return path.read_text()

        def python_ast(self, target: Path) -> ast.AST:
            assert target == path
            return ast.parse(path.read_text(), filename=str(path))

    assert _extract_artifact_filenames(path, source_index=Index()) == {
        "custom_renderer_blob.bin"
    }


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


def test_scanner_resolves_binop_list_concatenation(tmp_path: Path) -> None:
    """R38 fix: launchers commonly do `cmd = base + extras` or
    `["python", target] + flag_list`. Prior scanner returned None for
    BinOp, silently skipping the invocation — Rule C/D could not fire.
    """
    launcher = tmp_path / "launcher.py"
    _write(launcher, """
        import subprocess
        def run():
            base = ["python", "experiments/qat.py", "--checkpoint", "x"]
            extras = ["--motion-hidden", "24"]
            cmd = base + extras
            subprocess.run(cmd)
    """)
    invocations, _ = _scan_launcher_invocations(launcher)
    assert len(invocations) == 1
    _, target, flags = invocations[0]
    assert target == "experiments/qat.py"
    assert "--checkpoint" in flags
    assert "--motion-hidden" in flags


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
    """If git rev-parse fails, the function must not raise — just record the error.

    R28: track that ALL git calls (rev-parse + status --porcelain) are made,
    not just the first; a regression in either branch would slip past a
    fail-on-first test.
    """
    pipeline = _load_pipeline_module("_pipeline_prov_2")
    git_call_count = 0
    def fake_check(cmd, **kw):
        nonlocal git_call_count
        if cmd and cmd[0] == "git":
            git_call_count += 1
            raise FileNotFoundError("git not found (test)")
        import subprocess as _sp
        return _sp.check_output(cmd, **kw)
    monkeypatch.setattr(pipeline.subprocess, "check_output", fake_check)
    prov = pipeline._capture_provenance(pipeline.PipelineConfig())
    assert "git_hash_error" in prov
    assert "config" in prov
    assert "timestamp_utc" in prov
    # The whole try-block bails on the first exception, so only one git call
    # is expected. Document this behavior so a refactor that splits git into
    # two try-except branches won't silently regress.
    assert git_call_count == 1, \
        f"expected one git invocation in single try-block, got {git_call_count}"


def test_user_provided_flags_requires_active_subcommand() -> None:
    """R28: the prior None default was a footgun. Now positional kw-only required."""
    import argparse
    pipeline = _load_pipeline_module("_pipeline_uvf_required")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    comp = sub.add_parser("compress")
    comp.add_argument("--base-ch", type=int)
    with pytest.raises(ValueError, match="active_subcommand"):
        pipeline._user_provided_flags(parser, ["compress", "--base-ch", "32"],
                                       active_subcommand="")
    with pytest.raises(TypeError):
        # Caller forgot the kw-only arg entirely
        pipeline._user_provided_flags(parser, ["compress", "--base-ch", "32"])


def test_extract_eval_score_human_lines_ignored_in_favor_of_json() -> None:
    """R29: human-readable score lines must be ignored — only RESULT_JSON counts.

    Replaces the R28 regex-era test. The new parser fundamentally cannot be
    fooled by component-value lines because it doesn't read them at all.
    """
    pipeline = _load_pipeline_module("_pipeline_eval_distortion")
    # Lots of distractor numbers; only the JSON sentinel should be parsed.
    stdout = (
        "Distortion components: seg=0.116 pose=0.476 rate=1.528 TOTAL 2.014\n"
        'RESULT_JSON: {"schema_version":1,"final_score":99.9,'
        '"score_seg":0,"score_pose":0,"score_rate":0,'
        '"avg_segnet_dist":0,"avg_posenet_dist":0,"rate":0,'
        '"archive_size_bytes":0,"gt_size_bytes":1}'
    )
    # Returns 99.9 (the JSON value), not 2.014 (the human line).
    assert pipeline._extract_eval_score(stdout) == 99.9


def test_extract_eval_score_validates_schema_strictly() -> None:
    """R29: malformed RESULT_JSON must raise ValidationError, not return None.

    A garbage payload means the eval ran but emitted bad data — that's
    worse than no data and must fail loud.
    """
    pipeline = _load_pipeline_module("_pipeline_eval_strict")
    from pydantic import ValidationError
    stdout = 'RESULT_JSON: {"schema_version":1,"final_score":"not_a_number"}'
    with pytest.raises(ValidationError):
        pipeline._extract_eval_score(stdout)


def test_expected_auth_eval_schema_matches_emitter_source() -> None:
    """R32: contract regression test — the schema_version in
    auth_eval_renderer.py's emit code must match EXPECTED_AUTH_EVAL_SCHEMA
    in pipeline.py. Without this test, a bump in one without the other
    silently breaks every pipeline eval (parser raises RuntimeError, no
    score, .done_eval gets None, pipeline loops without convergence).
    """
    pipeline = _load_pipeline_module("_pipeline_schema_match")
    auth_eval_path = (Path(__file__).resolve().parent.parent.parent.parent
                      / "experiments" / "auth_eval_renderer.py")
    text = auth_eval_path.read_text()
    # Find the literal "schema_version": <int> in the emit payload
    import re as _re
    m = _re.search(r'"schema_version":\s*(\d+)', text)
    assert m, "auth_eval_renderer.py must emit a schema_version int literal"
    emitter_version = int(m.group(1))
    assert emitter_version == pipeline.EXPECTED_AUTH_EVAL_SCHEMA, (
        f"schema drift: auth_eval_renderer.py emits schema_version="
        f"{emitter_version} but pipeline.py expects "
        f"{pipeline.EXPECTED_AUTH_EVAL_SCHEMA}. Bump both in lockstep."
    )


def test_extract_eval_score_uses_last_result_json_when_multiple() -> None:
    """If the eval emits multiple RESULT_JSON lines (e.g., per-iteration),
    the LAST one wins."""
    pipeline = _load_pipeline_module("_pipeline_eval_multi")
    payload1 = ('RESULT_JSON: {"schema_version":1,"final_score":1.5,'
                '"score_seg":0,"score_pose":0,"score_rate":0,'
                '"avg_segnet_dist":0,"avg_posenet_dist":0,"rate":0,'
                '"archive_size_bytes":0,"gt_size_bytes":1}')
    payload2 = ('RESULT_JSON: {"schema_version":1,"final_score":2.5,'
                '"score_seg":0,"score_pose":0,"score_rate":0,'
                '"avg_segnet_dist":0,"avg_posenet_dist":0,"rate":0,'
                '"archive_size_bytes":0,"gt_size_bytes":1}')
    assert pipeline._extract_eval_score(f"{payload1}\n{payload2}") == 2.5


def test_extract_eval_score_parses_result_json_sentinel() -> None:
    """R29: parser is now schema-first JSON, not regex."""
    pipeline = _load_pipeline_module("_pipeline_eval_1")
    stdout = "\n".join([
        "Loading scorer models...",
        "Pair 5/600 elapsed 45.2s",
        "TOTAL 2.014",  # human line — must be IGNORED by parser
        'RESULT_JSON: {"schema_version":1,"final_score":2.014,'
        '"score_seg":11.6,"score_pose":2.18,"score_rate":9.4,'
        '"avg_segnet_dist":0.116,"avg_posenet_dist":0.476,'
        '"rate":0.376,"archive_size_bytes":338000,"gt_size_bytes":37545489}',
        "All done in 67.8 seconds",
    ])
    assert pipeline._extract_eval_score(stdout) == 2.014


def test_extract_eval_score_returns_none_when_no_sentinel_line() -> None:
    """No RESULT_JSON line → None (caller decides what to do)."""
    pipeline = _load_pipeline_module("_pipeline_eval_2")
    stdout = "\n".join(["Loading models...", "TOTAL 2.014", "Pair 5/600 elapsed 45.2s"])
    assert pipeline._extract_eval_score(stdout) is None


def test_all_renderer_profiles_load_without_typo_errors() -> None:
    """Every PROFILES entry must pass _apply_profile without SystemExit AND
    actually construct a PipelineConfig without TypeError.

    R28 finding: the prior version of this test only setattr'd onto an empty
    Namespace and never built PipelineConfig — string-vs-int type mismatches
    in profile values would silently pass. Now we round-trip to a real
    PipelineConfig.
    """
    import argparse
    from dataclasses import fields
    pipeline = _load_pipeline_module("_pipeline_profile_matrix")
    from tac.profiles import PROFILES
    failures = []
    pcfg_field_names = {f.name for f in fields(pipeline.PipelineConfig)}
    for name in PROFILES:
        ns = argparse.Namespace()
        try:
            pipeline._apply_profile(ns, name, user_provided_flags=set())
        except SystemExit as e:
            failures.append(f"{name}: SystemExit {e}")
            continue
        # Construct a real PipelineConfig from the applied namespace.
        # Type mismatches (e.g., profile value "32" str instead of 32 int)
        # would surface here as TypeError or unexpected behavior.
        try:
            kwargs = {k: v for k, v in vars(ns).items() if k in pcfg_field_names}
            cfg = pipeline.PipelineConfig(**kwargs)
            # Spot-check a couple of known-int and known-str fields if the
            # profile sets them.
            if "base_ch" in kwargs:
                assert isinstance(cfg.base_ch, int), \
                    f"profile {name!r}: base_ch type {type(cfg.base_ch).__name__}"
            if "padding_mode" in kwargs:
                assert isinstance(cfg.padding_mode, str), \
                    f"profile {name!r}: padding_mode type {type(cfg.padding_mode).__name__}"
        except (TypeError, AssertionError) as e:
            failures.append(f"{name}: {type(e).__name__} {e}")
    assert not failures, "Profiles failed _apply_profile or PipelineConfig:\n" + "\n".join(failures)


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


def test_filename_contract_extracts_artifact_literals(tmp_path: Path) -> None:
    """_extract_artifact_filenames pulls .bin/.pt/.mkv/.zip/.tar.gz literals."""
    src = tmp_path / "consumer.py"
    _write(src, '''
        from pathlib import Path
        x = Path("foo/renderer_qat.bin")
        y = Path("bar/qat_best_float.pt")
        z = Path("masks_full.mkv")
        bundle = Path("output/bundle.tar.gz")
        ignore = "logs/run.txt"  # not artifact
        ignore2 = "*.bin"  # glob pattern
        ignore3 = ".bin"  # extension fragment
        ignore4 = "_int4lzma2.bin"  # suffix fragment, no real stem
    ''')
    found = _extract_artifact_filenames(src)
    assert "renderer_qat.bin" in found
    assert "qat_best_float.pt" in found
    assert "masks_full.mkv" in found
    assert "bundle.tar.gz" in found
    assert "run.txt" not in found
    assert "*.bin" not in found
    assert ".bin" not in found
    assert "_int4lzma2.bin" not in found  # suffix fragment filtered


def test_magic_bytes_bidirectional_consistency() -> None:
    """R35/R36: magic-byte drift between producer and inflate consumer
    silently fails at contest submission. Test BOTH directions:

    1. Backward: every magic the consumer recognizes still appears in some
       producer (catches consumer that drops support for an old format).
    2. Forward: every magic any producer emits is also recognized by the
       consumer (catches producer that adds a new format the consumer
       doesn't understand yet — the high-impact direction).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    inflate_path = repo_root / "submissions" / "robust_current" / "inflate_renderer.py"
    if not inflate_path.exists():
        pytest.skip("inflate_renderer.py not present in this checkout")
    inflate_src = inflate_path.read_text()

    producer_files = [
        repo_root / "src" / "tac" / "renderer_export.py",
        repo_root / "src" / "tac" / "mixed_precision_export.py",
    ]
    # Match `b"XXXX"` and `b'XXXX'` — 3-5 char ASCII uppercase magic literals
    magic_re = re.compile(rb"b['\"]([A-Z0-9]{3,5})['\"]")
    producer_magics: set[bytes] = set()
    for pf in producer_files:
        if not pf.exists():
            continue
        for m in magic_re.finditer(pf.read_bytes()):
            producer_magics.add(m.group(1))

    # Magic bytes that are intentionally INTERNAL — produced by experimental
    # formats but never shipped in a contest archive. Any new entry here
    # MUST have a guard in pipeline.py that prevents the format from
    # reaching the archive (R24-style arch_header guard for MXLZ).
    INTERNAL_ONLY_MAGICS = {
        b"MXLZ",  # mixed-precision LZMA2; gated out by step_compress_weights
                  # arch_header check in pipeline.py (R24 finding).
    }

    # Forward: every producer magic must appear in inflate source OR be in
    # the internal-only allowlist (with explicit pipeline.py guard).
    inflate_bytes = inflate_src.encode("utf-8", errors="ignore")
    for magic in sorted(producer_magics):
        if magic in INTERNAL_ONLY_MAGICS:
            continue
        assert magic in inflate_bytes, (
            f"Producer emits magic {magic!r} but inflate_renderer.py "
            f"doesn't recognize it. Contest submission would fail at inflate. "
            f"Either add the magic to _load_renderer's dispatch OR add it to "
            f"INTERNAL_ONLY_MAGICS with a pipeline.py guard."
        )

    # Backward: known consumer magics must still appear in some producer
    consumer_magics = {b"FP4A", b"ASYM", b"I4LZ"}
    for magic in consumer_magics:
        assert magic in inflate_bytes, (
            f"Consumer no longer recognizes {magic!r} — bytecode regression"
        )



def test_filename_contract_write_prefix_function_return_is_produced(tmp_path: Path) -> None:
    """R36 scoped Return-tracking: a Return inside a function whose name
    starts with a write prefix (export_, save_, build_, etc.) counts as
    produced. Returns inside non-write-named functions do NOT count
    (would falsely mark input/read paths as produced).
    """
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/consumer.py", '''
        from pathlib import Path
        def save_artifact_path(iter_dir):
            return iter_dir / "produced_via_return.bin"
        x = Path("output/produced_via_return.bin")
    ''')
    # Function name starts with `save_` so the Return literal is treated
    # as produced. Validator should pass.
    violations = preflight_filename_contract(
        repo_root=repo,
        consumer_files=["experiments/consumer.py"],
        producer_dirs=["experiments"],
        strict=False, verbose=False,
    )
    assert violations == []


def test_preflight_training_inputs_catches_gt_video_range(tmp_path) -> None:
    """R38 regression: TTO frames at GT-video range [0,255] (max ~255)
    must FAIL — this is the WILDE catastrophe. Frames at TTO-optimized
    range cluster around max ~184."""
    import torch
    from tac.preflight import preflight_training_inputs, PreflightError
    # Synthetic GT-video-range frames (max 255, NOT TTO-optimized)
    bad = torch.randint(0, 256, (1200, 384, 512, 3), dtype=torch.uint8).float()
    bad_path = tmp_path / "tto_frames.pt"
    torch.save(bad, bad_path)
    poses_path = tmp_path / "gt_poses.pt"
    torch.save(torch.randn(600, 6), poses_path)
    # Need a fake masks file too (ffprobe will fail but TTO check fires first)
    masks_path = tmp_path / "masks.mkv"
    masks_path.write_bytes(b"fake")
    with pytest.raises(PreflightError, match="GT-video range"):
        preflight_training_inputs(
            tto_frames_path=bad_path, gt_poses_path=poses_path,
            masks_path=masks_path, profile_name="test",
            profile_arch={"base_ch": 32, "mid_ch": 48, "depth": 1,
                          "pose_dim": 6, "padding_mode": "zeros",
                          "eval_roundtrip": True},
            verbose=False,
        )


def test_preflight_training_inputs_accepts_tto_optimized_range(tmp_path) -> None:
    """Counterpart: frames at TTO-optimized range (max ~184) must PASS the
    range check (mask ffprobe will still fail since masks file is fake;
    we only verify the range check itself doesn't raise prematurely).
    """
    import torch
    from tac.preflight import preflight_training_inputs, PreflightError
    good = torch.rand(1200, 384, 512, 3) * 184  # TTO-optimized range
    good_path = tmp_path / "tto_frames.pt"
    torch.save(good, good_path)
    poses_path = tmp_path / "gt_poses.pt"
    torch.save(torch.randn(600, 6), poses_path)
    masks_path = tmp_path / "masks.mkv"
    masks_path.write_bytes(b"fake")
    # Range check passes; ffprobe on fake masks raises later.
    # We catch that and assert it's NOT the TTO-range error.
    try:
        preflight_training_inputs(
            tto_frames_path=good_path, gt_poses_path=poses_path,
            masks_path=masks_path, profile_name="test",
            profile_arch={"base_ch": 32, "mid_ch": 48, "depth": 1,
                          "pose_dim": 6, "padding_mode": "zeros",
                          "eval_roundtrip": True},
            verbose=False,
        )
    except PreflightError as e:
        assert "GT-video range" not in str(e), (
            f"TTO-optimized range was rejected as GT range: {e}"
        )


def test_preflight_check_rejects_wrong_pair_count_in_poses(tmp_path) -> None:
    """R38 regression: the 1199-overlapping-pairs catastrophe. preflight_check
    must reject pose tensors with the wrong N (not 600 for 600 pairs)."""
    import torch
    from tac.preflight import preflight_check, PreflightError
    pose_path = tmp_path / "poses.pt"
    torch.save(torch.randn(1199, 6), pose_path)
    with pytest.raises(PreflightError):
        preflight_check(poses_path=pose_path, verbose=False)


def test_filename_contract_return_with_name_indirection_resolves(tmp_path: Path) -> None:
    """R37: write-prefix function that returns a Name (not a literal) where
    the Name was assigned a Path-with-literal earlier — should resolve.
    """
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/consumer.py", '''
        from pathlib import Path
        def save_via_indirection(iter_dir):
            path = iter_dir / "indirect_produced.bin"
            return path
        x = Path("output/indirect_produced.bin")
    ''')
    violations = preflight_filename_contract(
        repo_root=repo,
        consumer_files=["experiments/consumer.py"],
        producer_dirs=["experiments"],
        strict=False, verbose=False,
    )
    assert violations == []


def test_filename_contract_non_write_function_return_does_not_silence_phantom(tmp_path: Path) -> None:
    """Inverse: a Return inside a function with a non-write name (e.g.,
    `get_path` or `find_input`) must NOT count as produced — otherwise an
    input-path helper would mask phantom reads of the same name.
    """
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/consumer.py", '''
        from pathlib import Path
        def get_input_path():
            return Path("input/never_produced.bin")
        x = Path("input/never_produced.bin")  # phantom: no producer
    ''')
    with pytest.raises(FilenameContractError, match="never_produced.bin"):
        preflight_filename_contract(
            repo_root=repo,
            consumer_files=["experiments/consumer.py"],
            producer_dirs=["experiments"],
            strict=True, verbose=False,
        )


def test_filename_contract_catches_phantom_path(tmp_path: Path) -> None:
    """R33/R34 bug class: consumer reads a name no producer writes → fail."""
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/consumer.py", '''
        from pathlib import Path
        x = Path("output/renderer_typo.bin")  # NO producer writes this
        subprocess_path = Path("output/renderer_qat.bin")  # producer DOES write
    ''')
    _write(repo / "experiments/producer.py", '''
        import torch
        torch.save({}, "output/renderer_qat.bin")  # writes the canonical name
    ''')
    with pytest.raises(FilenameContractError, match="renderer_typo.bin"):
        preflight_filename_contract(
            repo_root=repo,
            consumer_files=["experiments/consumer.py"],
            producer_dirs=["experiments"],
            strict=True, verbose=False,
        )


def test_filename_contract_passes_when_all_consumed_names_are_produced(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/consumer.py", '''
        from pathlib import Path
        x = Path("output/renderer_qat.bin")
    ''')
    _write(repo / "experiments/producer.py", '''
        import torch
        torch.save({}, "output/renderer_qat.bin")
    ''')
    violations = preflight_filename_contract(
        repo_root=repo,
        consumer_files=["experiments/consumer.py"],
        producer_dirs=["experiments"],
        strict=False, verbose=False,
    )
    assert violations == []


def test_filename_contract_cache_invalidates_changed_consumer(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    consumer = repo / "experiments" / "consumer.py"
    _write(consumer, '''
        from pathlib import Path
        x = Path("output/renderer_qat.bin")
    ''')
    _write(repo / "experiments" / "producer.py", '''
        import torch
        torch.save({}, "output/renderer_qat.bin")
    ''')
    assert preflight_filename_contract(
        repo_root=repo,
        consumer_files=["experiments/consumer.py"],
        producer_dirs=["experiments"],
        strict=True,
        verbose=False,
    ) == []
    assert (
        repo / ".omx" / "cache" / "filename_contract_source_clean.json"
    ).exists()

    _write(consumer, '''
        from pathlib import Path
        x = Path("output/renderer_typo.bin")
    ''')
    with pytest.raises(FilenameContractError, match="renderer_typo.bin"):
        preflight_filename_contract(
            repo_root=repo,
            consumer_files=["experiments/consumer.py"],
            producer_dirs=["experiments"],
            strict=True,
            verbose=False,
        )


def test_filename_contract_skips_external_filenames(tmp_path: Path) -> None:
    """0.mkv, masks.mkv, archive.zip etc. are contest-external; not validated."""
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/consumer.py", '''
        from pathlib import Path
        x = Path("upstream/videos/0.mkv")  # external
        y = Path("submission.zip")  # external
    ''')
    # No producer writes either. Should still pass — they're whitelisted.
    violations = preflight_filename_contract(
        repo_root=repo,
        consumer_files=["experiments/consumer.py"],
        producer_dirs=["experiments"],
        strict=False, verbose=False,
    )
    assert violations == []


def test_filename_contract_handles_shell_script_producers(tmp_path: Path) -> None:
    """Shell scripts (compress.sh, inflate.sh) often write archive artifacts."""
    repo = _stub_repo(tmp_path)
    _write(repo / "experiments/consumer.py", '''
        from pathlib import Path
        x = Path("output/result.tar.gz")
    ''')
    sh_path = repo / "experiments/build.sh"
    sh_path.parent.mkdir(parents=True, exist_ok=True)
    sh_path.write_text("#!/bin/bash\ntar czf output/result.tar.gz output/\n")
    violations = preflight_filename_contract(
        repo_root=repo,
        consumer_files=["experiments/consumer.py"],
        producer_dirs=["experiments"],
        strict=False, verbose=False,
    )
    assert violations == []


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
