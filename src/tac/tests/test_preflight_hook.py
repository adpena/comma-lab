# SPDX-License-Identifier: MIT
from __future__ import annotations

import re
import subprocess

import tools.preflight_hook as preflight_hook


def test_preflight_hook_defaults_to_no_codebase(monkeypatch) -> None:
    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)

    # `--acknowledge-empty-scope` joined the contract 2026-08-01 (task #842):
    # `--no-codebase` examines 0 gates, and the CLI now refuses rc=3 on an
    # empty scope instead of printing a bare "PREFLIGHT PASSED". This hook is
    # the one caller with a designed reason to accept vacuity, so it must NAME
    # it. The verdict still prints VACUOUS.
    assert preflight_hook._preflight_command() == [
        ".venv/bin/python",
        "-m",
        "tac.preflight",
        "--no-codebase",
        "--acknowledge-empty-scope",
    ]
    assert preflight_hook._preflight_timeout_seconds() == 30


def test_preflight_hook_full_mode_is_explicit(monkeypatch) -> None:
    monkeypatch.setenv("PREFLIGHT_FULL", "1")
    monkeypatch.delenv("PREFLIGHT_ALLOW_SLOW", raising=False)

    assert preflight_hook._preflight_command() == [
        ".venv/bin/python",
        "-m",
        "tac.preflight",
        "--scope",
        "all",
    ]
    assert preflight_hook._preflight_timeout_seconds() == 30


def test_preflight_hook_slow_release_mode_requires_separate_env(monkeypatch) -> None:
    monkeypatch.setenv("PREFLIGHT_FULL", "1")
    monkeypatch.setenv("PREFLIGHT_ALLOW_SLOW", "1")

    assert preflight_hook._preflight_command() == [
        ".venv/bin/python",
        "-m",
        "tac.preflight",
        "--scope",
        "all",
        "--allow-slow-preflight",
    ]
    assert preflight_hook._preflight_timeout_seconds() == 600


def test_preflight_hook_timeout_env_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("PREFLIGHT_TIMEOUT_SECONDS", "12")
    assert preflight_hook._preflight_timeout_seconds() == 12

    monkeypatch.setenv("PREFLIGHT_TIMEOUT_SECONDS", "not-an-int")
    assert preflight_hook._preflight_timeout_seconds() == 30


def test_preflight_hook_detects_pre_push_invocation(monkeypatch) -> None:
    monkeypatch.setattr(preflight_hook.sys, "argv", [".git/hooks/pre-push"])
    assert preflight_hook._is_pre_push_invocation() is True

    monkeypatch.setattr(preflight_hook.sys, "argv", [".git/hooks/pre-commit"])
    assert preflight_hook._is_pre_push_invocation() is False


def test_run_preflight_reports_timeout(monkeypatch, capsys) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("args") or args[0],
            timeout=kwargs["timeout"],
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)
    monkeypatch.setattr(preflight_hook.subprocess, "run", fake_run)

    assert preflight_hook.run_preflight() == 1
    captured = capsys.readouterr()
    assert "preflight timed out" in captured.err
    assert "tac.preflight --no-codebase" in captured.err


# ---------------------------------------------------------------------------
# CI-blind (MLX-gated) test step. `.github/workflows/ci.yml` runs pytest on
# ubuntu-24.04 with `.[dev,runtime]`; `mlx` is a separate extra with no Linux
# wheel, so `pytest.importorskip("mlx...")` modules are SKIPPED there and pytest
# reports the skip as GREEN. Before this step no automated surface ran them at
# all — test_ddm_tb1_tr1_renderer.py sat red on main for ~2 days that way.
# ---------------------------------------------------------------------------
def test_ci_blind_modules_all_carry_the_mlx_import_guard() -> None:
    # Independent (regex) cross-check of the AST detector: every module it returns must
    # really call importorskip on an "mlx*" target, or CI is not in fact blind to it.
    guard = re.compile(r"""importorskip\(\s*["']mlx""")
    blind = preflight_hook._ci_blind_test_modules()
    assert blind, "expected MLX-gated test modules under src/tac/tests"
    for path in blind:
        assert guard.search(path.read_text(encoding="utf-8")), path


def test_ci_blind_modules_include_the_tr1_ledger_test() -> None:
    names = {p.name for p in preflight_hook._ci_blind_test_modules()}
    # the module whose counted-ledger contract test sat red unseen (task #845)
    assert "test_ddm_tb1_tr1_renderer.py" in names


def test_module_reference_tokens_strip_src_and_yield_every_suffix() -> None:
    assert preflight_hook._module_reference_tokens(
        "src/tac/witness_dsl/qa84_rowband_grammar_20260731.py") == {
        "tac.witness_dsl.qa84_rowband_grammar_20260731",
        "witness_dsl.qa84_rowband_grammar_20260731",
        "qa84_rowband_grammar_20260731"}


def test_module_reference_tokens_for_top_level_package() -> None:
    assert preflight_hook._module_reference_tokens(
        "experiments/train_tr1_partition_renderer_mlx.py") == {
        "experiments.train_tr1_partition_renderer_mlx",
        "train_tr1_partition_renderer_mlx"}


def test_module_reference_tokens_empty_path_is_empty() -> None:
    assert preflight_hook._module_reference_tokens("") == set()


def test_select_ci_blind_tests_empty_staged_selects_nothing() -> None:
    assert preflight_hook._select_ci_blind_tests([]) == []


def test_select_ci_blind_tests_picks_up_reverse_dependency() -> None:
    sel = preflight_hook._select_ci_blind_tests(
        ["experiments/train_tr1_partition_renderer_mlx.py"])
    assert "src/tac/tests/test_ddm_tb1_tr1_renderer.py" in sel


def test_select_ci_blind_tests_includes_a_staged_blind_module_itself() -> None:
    # a staged blind module is selected even when nothing else references it
    rel = "src/tac/tests/test_ddm_tb1_tr1_renderer.py"
    assert rel in preflight_hook._select_ci_blind_tests([rel])


def test_select_ci_blind_tests_matches_whole_words_not_substrings() -> None:
    # MEASURED: token "ml" is a substring of "mlx" (and of "html", "yaml", ...), so a
    # naive substring matcher selects ALL 57 blind modules for this staged file. Word
    # boundaries are what keep the step from running the whole MLX suite every commit.
    assert preflight_hook._select_ci_blind_tests(["docs/ml.py"]) == []


def test_select_ci_blind_tests_skips_modules_ci_can_already_run() -> None:
    # test_ddm_b2b_burn2_composition.py imports the same trainer but is NOT
    # MLX-gated, so CI runs it — this step must not duplicate CI's coverage.
    sel = preflight_hook._select_ci_blind_tests(
        ["experiments/train_tr1_partition_renderer_mlx.py"])
    assert "src/tac/tests/test_ddm_b2b_burn2_composition.py" not in sel


def test_ci_blind_timeout_default_and_override(monkeypatch) -> None:
    monkeypatch.delenv("PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS", raising=False)
    assert preflight_hook._ci_blind_timeout_seconds() == 180
    monkeypatch.setenv("PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS", "42")
    assert preflight_hook._ci_blind_timeout_seconds() == 42


def test_ci_blind_timeout_rejects_garbage(monkeypatch) -> None:
    monkeypatch.setenv("PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS", "not-an-int")
    assert preflight_hook._ci_blind_timeout_seconds() == 180
    monkeypatch.setenv("PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS", "-5")
    assert preflight_hook._ci_blind_timeout_seconds() == 180


def test_run_ci_blind_tests_env_skip_is_explicit(monkeypatch) -> None:
    monkeypatch.setenv("PREFLIGHT_SKIP_CI_BLIND_TESTS", "1")

    def boom(*a, **k):  # must not even select, let alone run
        raise AssertionError("skip env must short-circuit before selection")

    monkeypatch.setattr(preflight_hook, "_select_ci_blind_tests", boom)
    assert preflight_hook.run_ci_blind_tests(["x.py"]) == 0


def test_run_ci_blind_tests_no_selection_is_a_noop(monkeypatch) -> None:
    monkeypatch.delenv("PREFLIGHT_SKIP_CI_BLIND_TESTS", raising=False)
    monkeypatch.setattr(preflight_hook, "_select_ci_blind_tests", lambda staged: [])
    assert preflight_hook.run_ci_blind_tests(["x.py"]) == 0


def test_run_ci_blind_tests_passes_a_real_green_module(monkeypatch, tmp_path) -> None:
    green = tmp_path / "test_green_ci_blind.py"
    green.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    monkeypatch.delenv("PREFLIGHT_SKIP_CI_BLIND_TESTS", raising=False)
    monkeypatch.setattr(preflight_hook, "_select_ci_blind_tests",
                        lambda staged: [str(green)])
    assert preflight_hook.run_ci_blind_tests(["x.py"]) == 0


def test_run_ci_blind_tests_blocks_on_a_real_red_module(monkeypatch, tmp_path) -> None:
    red = tmp_path / "test_red_ci_blind.py"
    red.write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    monkeypatch.delenv("PREFLIGHT_SKIP_CI_BLIND_TESTS", raising=False)
    monkeypatch.setattr(preflight_hook, "_select_ci_blind_tests",
                        lambda staged: [str(red)])
    assert preflight_hook.run_ci_blind_tests(["x.py"]) == 1


def test_run_ci_blind_tests_blocks_on_timeout_never_soft_passes(monkeypatch, capsys) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.delenv("PREFLIGHT_SKIP_CI_BLIND_TESTS", raising=False)
    monkeypatch.setattr(preflight_hook, "_select_ci_blind_tests",
                        lambda staged: ["src/tac/tests/test_ddm_tb1_tr1_renderer.py"])
    monkeypatch.setattr(preflight_hook.subprocess, "run", fake_run)
    # a soft-pass here would re-create the exact silence the step exists to remove
    assert preflight_hook.run_ci_blind_tests(["x.py"]) == 1
    assert "CI-blind tests timed out" in capsys.readouterr().err


def test_ci_blind_step_is_wired_into_main_not_orphaned() -> None:
    src = (preflight_hook.REPO_ROOT / "tools" / "preflight_hook.py").read_text(
        encoding="utf-8")
    main_body = src.split("def main()", 1)[1]
    assert "run_ci_blind_tests(staged)" in main_body
