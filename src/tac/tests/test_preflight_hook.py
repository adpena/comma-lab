# SPDX-License-Identifier: MIT
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import tools.preflight_hook as preflight_hook


def test_preflight_hook_defaults_to_no_codebase(monkeypatch) -> None:
    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)
    # PREFLIGHT_SCOPE joined the selector set with the #905 dev branch; clear it
    # so this test asserts the DEFAULT rather than whatever the ambient shell has.
    monkeypatch.delenv("PREFLIGHT_SCOPE", raising=False)

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


# --- task #905: the missing third hook mode (`--scope dev`) -------------------
# The hook advertised a bounded developer stack in its own docstring but could
# only emit 0 gates or the exhaustive release sweep. These pin the third branch,
# its timeout, and — most importantly — that adding it did not weaken the other
# two or leak the 0-gate vacuity waiver into it.


def test_preflight_hook_dev_scope_branch_exists(monkeypatch) -> None:
    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)
    monkeypatch.delenv("PREFLIGHT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("PREFLIGHT_SCOPE", "dev")

    assert preflight_hook._preflight_scope() == "dev"
    assert preflight_hook._preflight_command() == [
        ".venv/bin/python",
        "-m",
        "tac.preflight",
        "--scope",
        "dev",
    ]


def test_preflight_hook_dev_scope_never_acknowledges_empty_scope(monkeypatch) -> None:
    """The vacuity waiver is earned by the 0-gate mode ONLY.

    If it leaked into dev mode, a dev scope that examined 0 gates would print
    PASSED instead of refusing rc=3 — re-importing the exact
    vacuity-indistinguishable-from-PASS bug this hook reports on.
    """
    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)
    monkeypatch.setenv("PREFLIGHT_SCOPE", "dev")

    assert "--acknowledge-empty-scope" not in preflight_hook._preflight_command()
    assert "--no-codebase" not in preflight_hook._preflight_command()


def test_preflight_hook_dev_scope_has_headroom_over_measured_cost(monkeypatch) -> None:
    """MEASURED 22.7s warm / 24.3s cold (ddm_rg2). 30s left only 19-24% headroom.

    A clock failure is indistinguishable from a finding to the committer, so the
    dev branch must not inherit the 0-gate mode's 30s bound.
    """
    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)
    monkeypatch.delenv("PREFLIGHT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("PREFLIGHT_SCOPE", "dev")

    timeout = preflight_hook._preflight_timeout_seconds()
    assert timeout == 60
    assert timeout >= 2 * 24.3, "dev bound must clear 2x the COLD measurement"


def test_preflight_hook_dev_scope_widens_the_serializer_lock_patience(monkeypatch) -> None:
    """The serializer derives its lock patience from this bound; it must track."""
    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)
    monkeypatch.delenv("PREFLIGHT_TIMEOUT_SECONDS", raising=False)

    monkeypatch.delenv("PREFLIGHT_SCOPE", raising=False)
    default_bound = preflight_hook.effective_hook_wall_clock_bound_seconds()
    monkeypatch.setenv("PREFLIGHT_SCOPE", "dev")
    dev_bound = preflight_hook.effective_hook_wall_clock_bound_seconds()

    assert dev_bound == default_bound + 30


def test_preflight_full_still_takes_precedence_over_scope(monkeypatch) -> None:
    """Back-compat: every existing runbook/caller that sets PREFLIGHT_FULL is unchanged."""
    monkeypatch.setenv("PREFLIGHT_FULL", "1")
    monkeypatch.setenv("PREFLIGHT_SCOPE", "dev")
    monkeypatch.delenv("PREFLIGHT_ALLOW_SLOW", raising=False)

    assert preflight_hook._preflight_scope() == "all"
    assert preflight_hook._preflight_command()[-2:] == ["--scope", "all"]


def test_preflight_hook_unknown_scope_falls_back_to_the_default_mode(monkeypatch) -> None:
    """An unrecognized value must not error and must not silently widen scope."""
    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)
    monkeypatch.delenv("PREFLIGHT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("PREFLIGHT_SCOPE", "definitely-not-a-scope")

    assert preflight_hook._preflight_scope() == "none"
    assert preflight_hook._preflight_command() == [
        ".venv/bin/python",
        "-m",
        "tac.preflight",
        "--no-codebase",
        "--acknowledge-empty-scope",
    ]
    assert preflight_hook._preflight_timeout_seconds() == 30


def test_preflight_hook_scope_all_matches_preflight_full(monkeypatch) -> None:
    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)
    monkeypatch.setenv("PREFLIGHT_SCOPE", "all")
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


def test_preflight_hook_explicit_timeout_override_still_wins_in_dev(monkeypatch) -> None:
    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)
    monkeypatch.setenv("PREFLIGHT_SCOPE", "dev")
    monkeypatch.setenv("PREFLIGHT_TIMEOUT_SECONDS", "15")

    assert preflight_hook._preflight_timeout_seconds() == 15


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


def test_module_reference_tokens_package_init_resolves_to_the_package(tmp_path) -> None:
    """#936 (`ddm_vw1`): a package `__init__.py` must yield PACKAGE names, never a bare
    `__init__` stem — the old behavior was wrong in BOTH directions.

    OVER-match: `\\b__init__\\b` is a whole-word hit in essentially every Python file, so
    staging one 2-line `__init__.py` selected 14 heavy MLX GPU modules, 14/14 matched
    solely by that token and 0/14 referencing the staged package.
    UNDER-match (the failure this step exists to prevent): a test doing
    `from tac.verdicts import X` was never matched by the token `tac.verdicts.__init__`.
    """
    assert preflight_hook._module_reference_tokens(
        "src/tac/verdicts/__init__.py") == {"tac.verdicts", "verdicts"}
    # The bare stem must be gone — it is the universal false positive.
    assert "__init__" not in preflight_hook._module_reference_tokens(
        "src/tac/verdicts/__init__.py")
    # A degenerate top-level "__init__.py" keeps its only name (no empty token).
    assert preflight_hook._module_reference_tokens("__init__.py") == {"__init__"}
    # Non-package modules are untouched.
    assert preflight_hook._module_reference_tokens("src/tac/verdicts/emit.py") == {
        "tac.verdicts.emit", "verdicts.emit", "emit"}


def test_nested_package_init_drops_generic_leaf_token() -> None:
    """Nested package leaves can be broad domain words; keep package suffixes, not the bare leaf."""
    assert preflight_hook._module_reference_tokens(
        "src/tac/pr130_lift/pose/__init__.py") == {
        "tac.pr130_lift.pose", "pr130_lift.pose"}
    assert "pose" not in preflight_hook._module_reference_tokens(
        "src/tac/pr130_lift/pose/__init__.py")


def test_package_init_does_not_select_unrelated_mlx_modules() -> None:
    """The regression itself: staging tac/verdicts/__init__.py must not drag in MLX GPU
    modules that never reference it (they crashed under concurrent-MLX GPU contention)."""
    selected = preflight_hook._select_ci_blind_tests(["src/tac/verdicts/__init__.py"])
    modules = {s.split("::")[0] for s in selected}
    offenders = [m for m in modules if "tac.verdicts" not in Path(m).read_text()]
    assert offenders == [], f"selected modules that never reference tac.verdicts: {offenders}"


def test_nested_pose_package_init_does_not_select_pose_word_mlx_modules() -> None:
    """Regression for #983: bare token `pose` selected 29 unrelated MLX-gated targets."""
    selected = preflight_hook._select_ci_blind_tests(
        ["src/tac/pr130_lift/pose/__init__.py"])
    assert selected == []


def test_cb2_repro_pair_targets_are_ordered_subset_of_legacy_selection() -> None:
    """#983 residual: the repro tool's smaller pair must stay in legacy selection order."""
    from tools import repro_cb2_pr130_lift_pose_ci_blind_order as repro

    legacy = repro.legacy_pose_token_targets()
    positions = [legacy.index(target) for target in repro.PAIR_TARGETS]
    assert positions == sorted(positions)


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


# These two pinned the literal 180 and correctly went red when it changed. They now pin
# the BEHAVIOUR (env override wins; garbage falls back to the default) against the
# derived constant; the constant itself is pinned to its measurement further down, in
# `test_ci_blind_timeout_default_covers_the_measured_worst_case`.
def test_ci_blind_timeout_default_and_override(monkeypatch) -> None:
    monkeypatch.delenv("PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS", raising=False)
    assert preflight_hook._ci_blind_timeout_seconds() == \
        preflight_hook._CI_BLIND_TIMEOUT_DEFAULT_SECONDS
    monkeypatch.setenv("PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS", "42")
    assert preflight_hook._ci_blind_timeout_seconds() == 42


def test_ci_blind_timeout_rejects_garbage(monkeypatch) -> None:
    default = preflight_hook._CI_BLIND_TIMEOUT_DEFAULT_SECONDS
    monkeypatch.setenv("PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS", "not-an-int")
    assert preflight_hook._ci_blind_timeout_seconds() == default
    monkeypatch.setenv("PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS", "-5")
    assert preflight_hook._ci_blind_timeout_seconds() == default


def test_governed_metal_burn_pids_excludes_observer_flag_values(monkeypatch) -> None:
    ps_output = (
        "101 python tools/watch.py --training-sig train_tr1_partition_renderer_mlx.py\n"
        "202 python experiments/train_tr1_partition_renderer_mlx.py --epochs 10\n"
    )

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=ps_output)

    monkeypatch.setattr(preflight_hook.subprocess, "run", fake_run)
    assert preflight_hook._governed_metal_burn_pids() == ["202"]


def test_governed_metal_burn_pids_ignores_process_table_readers(monkeypatch) -> None:
    ps_output = (
        "303 rg train_levelset_witness_realized_through_R_mlx.py tools\n"
        "404 ps -axo pid=,command=\n"
    )

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=ps_output)

    monkeypatch.setattr(preflight_hook.subprocess, "run", fake_run)
    assert preflight_hook._governed_metal_burn_pids() == []


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
    # Per-module isolation (#1154) names the module that timed out; the rc=1
    # never-soft-pass contract is unchanged.
    err = capsys.readouterr().err
    assert "CI-blind module timed out" in err
    assert "test_ddm_tb1_tr1_renderer.py" in err


def test_ci_blind_step_is_wired_into_main_not_orphaned() -> None:
    src = (preflight_hook.REPO_ROOT / "tools" / "preflight_hook.py").read_text(
        encoding="utf-8")
    main_body = src.split("def main()", 1)[1]
    assert "run_ci_blind_tests(staged)" in main_body


def test_followon_regrow_scan_warns_with_live_count(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        preflight_hook,
        "_staged_added_lines",
        lambda staged_docs: [
            (
                ".omx/research/new.md",
                "NEXT-IF-RESUMED: $0 follow-on should be run with owner sq2.",
            ),
            (".omx/research/new.md", "ordinary line"),
        ],
    )

    assert preflight_hook.run_followon_regrow_scan([".omx/research/new.md"]) == 0
    err = capsys.readouterr().err
    assert "1 staged .md" in err
    assert "2 added lines" in err
    assert "1 new cheap follow-on lines" in err


def test_followon_regrow_step_is_wired_before_preflight() -> None:
    src = (preflight_hook.REPO_ROOT / "tools" / "preflight_hook.py").read_text(
        encoding="utf-8")
    main_body = src.split("def main()", 1)[1]
    assert "run_followon_regrow_scan(staged_docs)" in main_body
    assert main_body.index("run_followon_regrow_scan(staged_docs)") < main_body.index(
        "rc = run_preflight()")


def test_shell_driver_rc_scan_warns_on_implicit_success_after_rc_echo(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(preflight_hook, "REPO_ROOT", tmp_path)
    script = tmp_path / "driver.sh"
    script.write_text(
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        "python final.py\n"
        "echo \"final rc: $?\"\n",
        encoding="utf-8",
    )

    warnings = preflight_hook._shell_driver_rc_receipt_warnings(["driver.sh"])
    assert len(warnings) == 1
    assert "no variable exit/return propagates" in warnings[0]


def test_shell_driver_rc_scan_warns_on_unconditional_exit_zero(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(preflight_hook, "REPO_ROOT", tmp_path)
    script = tmp_path / "driver.sh"
    script.write_text(
        "#!/bin/bash\n"
        "rc=0\n"
        "python final.py || rc=$?\n"
        "echo \"final rc=$rc\"\n"
        "exit 0\n",
        encoding="utf-8",
    )

    warnings = preflight_hook._shell_driver_rc_receipt_warnings(["driver.sh"])
    assert any("unconditional exit 0" in warning for warning in warnings)


def test_shell_driver_rc_scan_honors_same_line_waiver(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(preflight_hook, "REPO_ROOT", tmp_path)
    script = tmp_path / "driver.sh"
    script.write_text(
        "#!/bin/bash\n"
        "echo \"status rc=manual-sentinel\" # DRIVER_RC_EXIT0_OK: report-only sentinel\n"
        "exit 0 # DRIVER_RC_EXIT0_OK: report-only sentinel\n",
        encoding="utf-8",
    )

    assert preflight_hook._shell_driver_rc_receipt_warnings(["driver.sh"]) == []


def test_shell_driver_rc_scan_accepts_variable_exit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(preflight_hook, "REPO_ROOT", tmp_path)
    script = tmp_path / "driver.sh"
    script.write_text(
        "#!/bin/bash\n"
        "rc=0\n"
        "python final.py || rc=$?\n"
        "echo \"final rc=$rc\"\n"
        "exit \"$rc\"\n",
        encoding="utf-8",
    )

    assert preflight_hook._shell_driver_rc_receipt_warnings(["driver.sh"]) == []


def test_shell_driver_rc_scan_is_wired_before_preflight() -> None:
    src = (preflight_hook.REPO_ROOT / "tools" / "preflight_hook.py").read_text(
        encoding="utf-8")
    main_body = src.split("def main()", 1)[1]
    assert "run_shell_driver_rc_receipt_scan(staged_shell)" in main_body
    assert main_body.index("run_shell_driver_rc_receipt_scan(staged_shell)") < \
        main_body.index("rc = run_preflight()")


# ---------------------------------------------------------------------------
# CI-blind GATE SCOPE (task #854).
#
# The step exists to cover what GitHub Actions structurally cannot run. Whether CI can
# run a module depends on WHERE its MLX gate sits: a module-scope `importorskip` stops
# collection dead, an in-test one does not. Conflating the two made this step re-run
# work CI already does.
#
# MEASURED 2026-08-01 (denominator: the 57 modules `_ci_blind_test_modules()` returns):
#   * 32 module-scope, 25 test-scope.
#   * `--collect-only` with every `mlx*` import blocked collected 0 tests from all 32
#     module-scope modules and 769 from the 25 test-scope ones.
#   * test_compact_renderer_mlx_spine_runner.py: 309 passed in 664.0s standalone; the
#     8 gate-selected node ids account for 24.2s of that (3.7%). Selecting on the gate
#     instead of the whole file took the same staged-file commit from 664s to 30.8s.
# ---------------------------------------------------------------------------
import ast  # noqa: E402
import textwrap  # noqa: E402


def _scope(source: str) -> str:
    return preflight_hook._mlx_gate_scope(ast.parse(textwrap.dedent(source)))


def test_module_scope_importorskip_is_module_gated() -> None:
    assert _scope('''
        import pytest
        mx = pytest.importorskip("mlx.core")

        def test_a():
            assert mx
    ''') == "module"


def test_importorskip_inside_a_module_level_try_is_still_module_gated() -> None:
    # ast.walk would find this either way; the point is that PRUNING at defs must not
    # also prune compound statements at module scope.
    assert _scope('''
        import pytest
        try:
            mx = pytest.importorskip("mlx.core")
        except Exception:
            mx = None
    ''') == "module"


def test_in_test_importorskip_is_test_gated_not_module_gated() -> None:
    assert _scope('''
        import pytest

        def test_a():
            mx = pytest.importorskip("mlx.core")
            assert mx

        def test_b():
            assert True
    ''') == "test"


def test_module_level_pytestmark_conditioned_on_mlx_is_module_gated() -> None:
    assert _scope('''
        import pytest
        try:
            import mlx.core  # noqa: F401
            _MLX_AVAILABLE = True
        except ImportError:
            _MLX_AVAILABLE = False
        pytestmark = pytest.mark.skipif(not _MLX_AVAILABLE, reason="needs mlx")

        def test_a():
            mx = pytest.importorskip("mlx.core")
    ''') == "module"


def test_module_level_allow_module_level_skip_is_module_gated() -> None:
    # `allow_module_level` stops collection dead on CI, so it outranks any in-test gate.
    assert _scope('''
        import pytest
        pytest.skip("no mlx", allow_module_level=True)

        def test_a():
            pytest.importorskip("mlx.core")
    ''') == "module"
    assert _scope('''
        import pytest
        if not _HAVE:
            pytest.skip("no mlx", allow_module_level=True)

        def test_a():
            pytest.importorskip("mlx.core")
    ''') == "module"


def test_gated_test_names_follow_the_gate_not_the_word_mlx() -> None:
    names = preflight_hook._mlx_gated_test_names(ast.parse(textwrap.dedent('''
        import pytest

        def test_gated():
            pytest.importorskip("mlx.core")

        def test_merely_mentions_mlx():
            # runs fine on Linux CI: the mlx-named symbol is monkeypatched away
            assert "run_mlx_thing" == "run_mlx_thing"
    ''')))
    assert names == ["test_gated"]


def test_gated_test_names_are_transitive_through_helpers_and_fixtures() -> None:
    names = set(preflight_hook._mlx_gated_test_names(ast.parse(textwrap.dedent('''
        import pytest

        def _inner():
            pytest.importorskip("mlx.core")

        def _outer():
            _inner()

        @pytest.fixture
        def gated_fixture():
            import mlx.core
            return mlx.core

        def test_via_helper():
            _outer()

        def test_via_fixture(gated_fixture):
            assert gated_fixture

        def test_clean():
            assert True
    '''))))
    assert names == {"test_via_helper", "test_via_fixture"}


def test_gated_test_names_include_skipif_decorated_class_methods() -> None:
    names = preflight_hook._mlx_gated_test_names(ast.parse(textwrap.dedent('''
        import pytest
        _MLX_AVAILABLE = False

        class TestThing:
            @pytest.mark.skipif(not _MLX_AVAILABLE, reason="needs mlx")
            def test_gated(self):
                assert True

            def test_clean(self):
                assert True
    ''')))
    assert names == ["TestThing::test_gated"]


def test_targets_for_module_scope_module_is_the_whole_file() -> None:
    # Live anchor: 64 tests, module-scope `importorskip` at line 26, MEASURED 214.4s.
    # CI collects nothing from it, so this hook owns every one of those tests.
    path = preflight_hook.REPO_ROOT / "src/tac/tests/test_micro_batch_bit_identity_probe.py"
    assert preflight_hook._ci_blind_targets_for(path, path.read_text(encoding="utf-8")) == [
        "src/tac/tests/test_micro_batch_bit_identity_probe.py"
    ]


def test_targets_for_test_scope_module_are_node_ids_covering_the_real_gates() -> None:
    rel = "src/tac/tests/test_compact_renderer_mlx_spine_runner.py"
    path = preflight_hook.REPO_ROOT / rel
    targets = preflight_hook._ci_blind_targets_for(path, path.read_text(encoding="utf-8"))
    assert all("::" in t for t in targets), targets
    assert 0 < len(targets) < 50, f"{len(targets)} of 302 tests — see the header note"
    names = {t.split("::", 1)[1] for t in targets}
    # the two in-test `pytest.importorskip("mlx.core")` tests ...
    assert "test_hinerv_live_birth_hysteresis_probe_restores_model_state" in names
    assert (
        "test_hinerv_live_birth_survival_writes_four_arm_rows_when_birth_not_accepted"
        in names
    )
    # ... and the `skipif(not _MLX_AVAILABLE or not _AV_AVAILABLE)` one
    assert "test_hinerv_execute_runs_training_archive_and_receiver_proof" in names


def test_targets_fall_back_to_the_whole_file_when_no_gate_is_visible(tmp_path) -> None:
    # A test-scope module whose gate the static scan cannot see must run WHOLE, never
    # partially: running less is the silence this step exists to remove.
    path = preflight_hook.REPO_ROOT / "src/tac/tests/_ct2_fixture.py"
    text = 'import pytest\n\ndef helper():\n    pytest.importorskip("mlx.core")\n'
    assert preflight_hook._ci_blind_targets_for(path, text) == [
        "src/tac/tests/_ct2_fixture.py"
    ]


def test_targets_fall_back_to_the_whole_file_on_unparseable_source() -> None:
    path = preflight_hook.REPO_ROOT / "src/tac/tests/_ct2_fixture.py"
    assert preflight_hook._ci_blind_targets_for(path, "def broken(:\n") == [
        "src/tac/tests/_ct2_fixture.py"
    ]


def test_every_ci_blind_module_yields_at_least_one_target() -> None:
    # A module in the blind set that selects NOTHING would be a silent coverage hole.
    for path in preflight_hook._ci_blind_test_modules():
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert preflight_hook._ci_blind_targets_for(path, text), path


# ---------------------------------------------------------------------------
# The two CONTROL knobs (task #854): both DERIVED from measurement, both pinned here so
# a future edit that lowers them below the measurement fails a test instead of silently
# re-firing on green modules.
# ---------------------------------------------------------------------------
def test_ci_blind_timeout_default_covers_the_measured_worst_case() -> None:
    derived = (preflight_hook._CI_BLIND_PRE_SPLIT_WORST_MEASURED_SECONDS
               * preflight_hook._CI_BLIND_LOAD_SPREAD)
    assert preflight_hook._CI_BLIND_TIMEOUT_DEFAULT_SECONDS >= derived, (
        "the ceiling must cover 975s (both slow modules in one invocation, ddm_tr6 "
        "2026-08-01) times the measured 1.20x run-to-run load spread"
    )
    # 180s — the value this replaced — did NOT cover it. That is the bug being fixed.
    assert preflight_hook._CI_BLIND_TIMEOUT_DEFAULT_SECONDS > 180


def test_effective_hook_bound_composes_both_timeouts_and_follows_the_env(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PREFLIGHT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)
    base = preflight_hook.effective_hook_wall_clock_bound_seconds()
    assert base == (preflight_hook._preflight_timeout_seconds()
                    + preflight_hook._ci_blind_timeout_seconds()
                    + preflight_hook._HOOK_FIXED_OVERHEAD_SECONDS)
    # The serializer derives its lock patience from this, so an override must propagate.
    monkeypatch.setenv("PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS", "1500")
    assert preflight_hook.effective_hook_wall_clock_bound_seconds() == base - \
        preflight_hook._CI_BLIND_TIMEOUT_DEFAULT_SECONDS + 1500


def test_ci_blind_runs_each_module_in_its_own_subprocess(monkeypatch, tmp_path) -> None:
    """#1154: co-loading MLX modules in ONE pytest process is its own failure class.

    ddm_cu1 measured `Fatal Python error: Bus error` across 36 co-selected
    targets where the 11th passes standalone in 0.47s — no individual test was
    at fault, and the co-load red forced a documented skip. Isolation extincts
    the class, so the invocation shape is the contract: one subprocess per
    MODULE, never one process carrying every module's imports.
    """
    invocations: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        invocations.append(list(cmd))
        return _Result()

    targets = [
        "src/tac/tests/test_alpha.py",
        "src/tac/tests/test_beta.py::test_one",
        "src/tac/tests/test_beta.py::test_two",
    ]
    monkeypatch.delenv("PREFLIGHT_SKIP_CI_BLIND_TESTS", raising=False)
    monkeypatch.setenv("PREFLIGHT_CI_BLIND_FORCE", "1")
    monkeypatch.setattr(preflight_hook, "_select_ci_blind_tests", lambda staged: targets)
    monkeypatch.setattr(preflight_hook.subprocess, "run", fake_run)

    assert preflight_hook.run_ci_blind_tests(["x.py"]) == 0
    # Two distinct MODULES -> two subprocesses (not one, not three).
    assert len(invocations) == 2, invocations
    modules_per_call = [
        {arg.split("::", 1)[0] for arg in cmd if arg.startswith("src/")}
        for cmd in invocations
    ]
    assert all(len(mods) == 1 for mods in modules_per_call), modules_per_call
    assert {next(iter(mods)) for mods in modules_per_call} == {
        "src/tac/tests/test_alpha.py",
        "src/tac/tests/test_beta.py",
    }
    # The two node ids of one module still share that module's single process.
    beta = next(cmd for cmd in invocations if any("test_beta" in a for a in cmd))
    assert sum(1 for a in beta if a.startswith("src/")) == 2


def test_ci_blind_aggregate_budget_is_shared_across_isolated_modules(
    monkeypatch,
) -> None:
    """Isolation must not multiply the wall-clock bound the serializer relies on."""
    seen_timeouts: list[float] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        seen_timeouts.append(kwargs["timeout"])
        return _Result()

    monkeypatch.delenv("PREFLIGHT_SKIP_CI_BLIND_TESTS", raising=False)
    monkeypatch.setenv("PREFLIGHT_CI_BLIND_FORCE", "1")
    monkeypatch.setattr(
        preflight_hook, "_select_ci_blind_tests",
        lambda staged: ["src/tac/tests/test_a.py", "src/tac/tests/test_b.py"],
    )
    monkeypatch.setattr(preflight_hook.subprocess, "run", fake_run)
    assert preflight_hook.run_ci_blind_tests(["x.py"]) == 0

    budget = preflight_hook._ci_blind_timeout_seconds()
    assert len(seen_timeouts) == 2
    # Each module gets the REMAINING budget, never a fresh full one.
    assert seen_timeouts[0] <= budget
    assert seen_timeouts[1] <= seen_timeouts[0]


def test_ci_blind_reports_nothing_verified_when_every_module_collects_nothing(
    monkeypatch,
) -> None:
    """rc=5 from every module is not a green tick — vacuity must be said out loud."""

    class _Result:
        returncode = 5
        stdout = ""
        stderr = ""

    monkeypatch.delenv("PREFLIGHT_SKIP_CI_BLIND_TESTS", raising=False)
    monkeypatch.setenv("PREFLIGHT_CI_BLIND_FORCE", "1")
    monkeypatch.setattr(
        preflight_hook, "_select_ci_blind_tests",
        lambda staged: ["src/tac/tests/test_a.py"],
    )
    monkeypatch.setattr(preflight_hook.subprocess, "run", lambda cmd, **kw: _Result())
    assert preflight_hook.run_ci_blind_tests(["x.py"]) == 0


# --------------------------------------------------------------------------
# ddm_eq1 (2026-09-04): Catalog #344 on the COMMIT PATH.
#
# The gate is registered strict=True in preflight_all() at src/tac/preflight.py:7510,
# inside `if check_codebase:`. This hook's default mode is `--no-codebase`, which
# examines 0 of 27 codebase gates -- so a STRICT gate ran at release time and never
# once at commit time, and 29 memos accumulated in a week. These tests pin the cure.
# --------------------------------------------------------------------------
def _write_memo(tmp_path: Path, monkeypatch, name: str, body: str) -> str:
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True, exist_ok=True)
    (research / name).write_text(body, encoding="utf-8")
    monkeypatch.setattr(preflight_hook, "REPO_ROOT", tmp_path)
    return f".omx/research/{name}"


def test_catalog_344_scan_blocks_a_post_cutoff_memo_with_no_equation(
    tmp_path, monkeypatch
) -> None:
    rel = _write_memo(
        tmp_path,
        monkeypatch,
        "ddm_zz1_thing_20260901.md",
        "# zz1\n\nThe prior law is FALSIFIED at n600: the ratio is 3.14.\n",
    )
    assert preflight_hook.run_canonical_equation_reference_scan([rel]) == 1


def test_catalog_344_scan_accepts_an_equation_citation(tmp_path, monkeypatch) -> None:
    rel = _write_memo(
        tmp_path,
        monkeypatch,
        "ddm_zz2_thing_20260901.md",
        "# zz2\n\nFALSIFIED at n600. Anchors `some_law_v1` "
        "(`tac.canonical_equations`).\n",
    )
    assert preflight_hook.run_canonical_equation_reference_scan([rel]) == 0


def test_catalog_344_scan_accepts_a_substantive_waiver(tmp_path, monkeypatch) -> None:
    rel = _write_memo(
        tmp_path,
        monkeypatch,
        "ddm_zz3_review_20260901.md",
        "# zz3\n\nThe claim is FALSIFIED.\n"
        "<!-- # FORMALIZATION_PENDING: process review, no measured row of its own -->\n",
    )
    assert preflight_hook.run_canonical_equation_reference_scan([rel]) == 0


def test_catalog_344_scan_rejects_a_bare_placeholder_waiver(tmp_path, monkeypatch) -> None:
    rel = _write_memo(
        tmp_path,
        monkeypatch,
        "ddm_zz4_thing_20260901.md",
        "# zz4\n\nFALSIFIED.\n# FORMALIZATION_PENDING: <rationale>\n",
    )
    assert preflight_hook.run_canonical_equation_reference_scan([rel]) == 1


def test_catalog_344_placeholder_rejection_is_exact_match_only_MEASURED_GAP(
    tmp_path, monkeypatch
) -> None:
    """MEASURED 2026-09-04 (ddm_eq1): the placeholder check is EXACT-match, so the
    canonical HTML-comment form smuggles a placeholder past it.

    ``_CHECK_344_PLACEHOLDER_RATIONALES`` is compared with ``rationale in (...)`` after a
    ``strip()``, and the waiver regex captures ``[^\\n]+`` -- so inside the corpus's own
    ``<!-- # FORMALIZATION_PENDING:... -->`` form the captured rationale is
    ``"<rationale> -->"``, which is neither an exact placeholder nor shorter than the
    4-char floor. It is ACCEPTED.

    This test does not assert that the behaviour is right; it PINS it, so the gap is a
    recorded finding rather than an assumption. Widening the check would change the
    STRICT gate's semantics for the whole corpus and belongs to a charter that owns it.
    """
    rel = _write_memo(
        tmp_path,
        monkeypatch,
        "ddm_zz4b_thing_20260901.md",
        "# zz4b\n\nFALSIFIED.\n<!-- # FORMALIZATION_PENDING: <rationale> -->\n",
    )
    assert preflight_hook.run_canonical_equation_reference_scan([rel]) == 0


def test_catalog_344_scan_honours_the_gates_own_date_cutoff(tmp_path, monkeypatch) -> None:
    """Pre-cutoff memos are grandfathered by the gate; the hook must not re-open them."""
    rel = _write_memo(
        tmp_path, monkeypatch, "ddm_zz5_thing_20260101.md", "# zz5\n\nFALSIFIED.\n"
    )
    assert preflight_hook.run_canonical_equation_reference_scan([rel]) == 0


def test_catalog_344_scan_ignores_docs_outside_omx_research(tmp_path, monkeypatch) -> None:
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "ddm_zz6_thing_20260901.md").write_text(
        "# zz6\n\nFALSIFIED.\n", encoding="utf-8"
    )
    monkeypatch.setattr(preflight_hook, "REPO_ROOT", tmp_path)
    assert (
        preflight_hook.run_canonical_equation_reference_scan(
            ["docs/ddm_zz6_thing_20260901.md"]
        )
        == 0
    )


def test_catalog_344_scan_is_silent_and_passing_on_an_empty_stage() -> None:
    assert preflight_hook.run_canonical_equation_reference_scan([]) == 0


def test_catalog_344_scan_fails_open_when_a_memo_cannot_be_read(
    tmp_path, monkeypatch
) -> None:
    """A broken guard must not block every commit -- fail OPEN and loud, like its siblings."""
    monkeypatch.setattr(preflight_hook, "REPO_ROOT", tmp_path)
    assert (
        preflight_hook.run_canonical_equation_reference_scan(
            [".omx/research/ddm_zz7_missing_20260901.md"]
        )
        == 0
    )


def test_catalog_344_scan_env_disable_is_named_not_silent(tmp_path, monkeypatch, capsys) -> None:
    rel = _write_memo(
        tmp_path, monkeypatch, "ddm_zz8_thing_20260901.md", "# zz8\n\nFALSIFIED.\n"
    )
    monkeypatch.setenv("CANONICAL_EQUATION_REFERENCE_SCAN_ENABLED", "0")
    assert preflight_hook.run_canonical_equation_reference_scan([rel]) == 0
    assert "DISABLED by env" in capsys.readouterr().err


def test_catalog_344_scan_reuses_the_gates_own_predicates_not_a_second_copy() -> None:
    """One law, two statements drift. The hook step must import, never restate."""
    source = Path(preflight_hook.__file__).read_text(encoding="utf-8")
    body = source[source.index("def run_canonical_equation_reference_scan") :]
    body = body[: body.index("\ndef _staged_added_lines")]
    assert "from tac.preflight import (" in body
    assert "_check_344_text_has_empirical_finding" in body
    assert "_check_344_text_has_canonical_equation_reference" in body
    assert "_check_344_text_has_valid_waiver" in body
    # No re-declared trigger/acceptance vocabulary in the hook.
    assert "empirical anchor" not in body.lower().replace("_check_344", "")


def test_catalog_344_scan_is_wired_into_main_before_run_preflight() -> None:
    """run_preflight() early-returns on failure, so the step must precede it."""
    source = Path(preflight_hook.__file__).read_text(encoding="utf-8")
    main_body = source[source.index("def main() -> int:") :]
    step = main_body.index("run_canonical_equation_reference_scan(staged_docs)")
    preflight_call = main_body.index("rc = run_preflight()")
    assert step < preflight_call
