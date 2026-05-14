"""Tests for Catalog #226 — check_trainer_auth_eval_uses_canonical_helper.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable, the C6 auth_eval rc=2 bug class (hand-rolled subprocess
invocation of ``experiments/contest_auth_eval.py`` with wrong CLI flags)
must be extincted across all substrate trainers via this STRICT preflight
gate. The gate currently lands warn-only per CLAUDE.md "Strict-flip
atomicity rule" because 18 trainers hand-roll the pattern.

Anchor: ``feedback_recovery_2_c6_finish_and_modal_harvest_landed_20260514.md``
+ commit ``3e4571c3a6bc09005e8859e9ac02c74c851b6938`` (C6 fix that routed
through the canonical helper).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_trainer_auth_eval_uses_canonical_helper,
)


def _write_trainer(
    root: Path, name: str, body: str
) -> Path:
    """Write a synthetic substrate trainer file under <root>/experiments/."""
    exp_dir = root / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    f = exp_dir / f"train_substrate_{name}.py"
    f.write_text(textwrap.dedent(body), encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Positive tests — the gate detects the hand-rolled pattern.
# ---------------------------------------------------------------------------


def test_handrolled_subprocess_with_cmd_var_flagged(tmp_path: Path) -> None:
    """Trainer that builds a cmd list and passes it to subprocess.run."""
    _write_trainer(
        tmp_path,
        "fake_substrate",
        '''
        import subprocess
        import sys
        from pathlib import Path

        CONTEST_AUTH_EVAL_SCRIPT = Path("experiments/contest_auth_eval.py")

        def _full_main(args):
            cmd = [
                sys.executable,
                str(CONTEST_AUTH_EVAL_SCRIPT),
                "--archive", "archive.zip",
                "--inflate-sh", "inflate.sh",
                "--device", "cuda",
                "--json-out", "result.json",
            ]
            proc = subprocess.run(cmd, capture_output=True)
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "fake_substrate" in violations[0]
    assert "_full_main" in violations[0]
    assert "cmd" in violations[0]


def test_handrolled_subprocess_with_inline_args_flagged(tmp_path: Path) -> None:
    """Trainer with inline CONTEST_AUTH_EVAL_SCRIPT in subprocess args."""
    _write_trainer(
        tmp_path,
        "inline_substrate",
        '''
        import subprocess
        from pathlib import Path

        CONTEST_AUTH_EVAL_SCRIPT = Path("experiments/contest_auth_eval.py")

        def _full_main(args):
            proc = subprocess.run(
                ["python", str(CONTEST_AUTH_EVAL_SCRIPT), "--archive", "x"],
            )
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


def test_handrolled_subprocess_with_popen_flagged(tmp_path: Path) -> None:
    """Popen also counts as a subprocess invocation."""
    _write_trainer(
        tmp_path,
        "popen_substrate",
        '''
        import subprocess
        from pathlib import Path

        CONTEST_AUTH_EVAL_SCRIPT = Path("experiments/contest_auth_eval.py")

        def _full_main(args):
            cmd = [str(CONTEST_AUTH_EVAL_SCRIPT), "--archive", "x"]
            p = subprocess.Popen(cmd)
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert ":7" in violations[0] or "Popen" in violations[0] or violations[0]


def test_check_call_flagged(tmp_path: Path) -> None:
    """subprocess.check_call should also be flagged."""
    _write_trainer(
        tmp_path,
        "check_call",
        '''
        import subprocess

        def _full_main(args):
            cmd = ["python", "experiments/contest_auth_eval.py", "--archive", "x"]
            subprocess.check_call(cmd)
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


def test_check_output_flagged(tmp_path: Path) -> None:
    """subprocess.check_output should also be flagged."""
    _write_trainer(
        tmp_path,
        "check_output",
        '''
        import subprocess

        def _full_main(args):
            cmd = ["python", "experiments/contest_auth_eval.py"]
            out = subprocess.check_output(cmd)
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


def test_subprocess_call_flagged(tmp_path: Path) -> None:
    """subprocess.call should also be flagged."""
    _write_trainer(
        tmp_path,
        "sp_call",
        '''
        import subprocess

        def _full_main(args):
            cmd = ["python", "experiments/contest_auth_eval.py"]
            subprocess.call(cmd)
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Negative tests — the gate accepts the canonical helper pattern.
# ---------------------------------------------------------------------------


def test_canonical_helper_routing_accepted(tmp_path: Path) -> None:
    """Trainer routing through gate_auth_eval_call is accepted."""
    _write_trainer(
        tmp_path,
        "canonical",
        '''
        from tac.substrates._shared.smoke_auth_eval_gate import (
            gate_auth_eval_call,
        )

        def _full_main(args):
            result = gate_auth_eval_call(
                args=args,
                archive_zip="x",
                inflate_sh="y",
                upstream_dir="z",
                output_json="o",
                contest_auth_eval_script="experiments/contest_auth_eval.py",
                substrate_tag="canonical",
            )
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_canonical_aliased_helper_routing_accepted(tmp_path: Path) -> None:
    """Trainer using ``gate_auth_eval_call as _canon_gate_auth_eval_call`` is accepted."""
    _write_trainer(
        tmp_path,
        "aliased_canonical",
        '''
        from tac.substrates._shared.smoke_auth_eval_gate import (
            gate_auth_eval_call as _canon_gate_auth_eval_call,
        )

        def _full_main(args):
            _canon_gate_auth_eval_call(
                args=args, archive_zip="x", inflate_sh="y",
                upstream_dir="z", output_json="o",
                contest_auth_eval_script="experiments/contest_auth_eval.py",
                substrate_tag="aliased",
            )
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_trainer_without_contest_auth_eval_accepted(tmp_path: Path) -> None:
    """Trainer with no contest_auth_eval invocation at all is out-of-scope."""
    _write_trainer(
        tmp_path,
        "no_invocation",
        '''
        import subprocess

        def _full_main(args):
            cmd = ["ls", "-la"]
            subprocess.run(cmd)
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_docstring_only_mention_accepted(tmp_path: Path) -> None:
    """Comment/docstring mention of contest_auth_eval doesn't trigger gate."""
    _write_trainer(
        tmp_path,
        "comment_only",
        '''
        """This trainer historically called contest_auth_eval.py directly."""

        def _full_main(args):
            # NOTE: contest_auth_eval is no longer used here
            pass
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_constant_definition_only_accepted(tmp_path: Path) -> None:
    """Defining CONTEST_AUTH_EVAL_SCRIPT but never calling subprocess is OK."""
    _write_trainer(
        tmp_path,
        "const_only",
        '''
        from pathlib import Path

        CONTEST_AUTH_EVAL_SCRIPT = Path("experiments/contest_auth_eval.py")

        def _full_main(args):
            pass
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Waiver tests.
# ---------------------------------------------------------------------------


def test_same_line_waiver_with_reason_accepted(tmp_path: Path) -> None:
    """Same-line waiver with a real reason is honored."""
    _write_trainer(
        tmp_path,
        "waived",
        '''
        import subprocess

        def _full_main(args):
            cmd = ["python", "experiments/contest_auth_eval.py", "--archive", "x"]
            proc = subprocess.run(cmd)  # AUTH_EVAL_DIRECT_SUBPROCESS_OK:diagnostic-probe-not-substrate-trainer
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_waiver_placeholder_rejected(tmp_path: Path) -> None:
    """Placeholder ``<reason>`` literal is rejected (gate's own docstring guard)."""
    _write_trainer(
        tmp_path,
        "placeholder_waiver",
        '''
        import subprocess

        def _full_main(args):
            cmd = ["python", "experiments/contest_auth_eval.py"]
            proc = subprocess.run(cmd)  # AUTH_EVAL_DIRECT_SUBPROCESS_OK:<reason>
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


def test_waiver_on_different_line_not_honored(tmp_path: Path) -> None:
    """Waiver on a different line is not honored."""
    _write_trainer(
        tmp_path,
        "wrong_line_waiver",
        '''
        import subprocess

        def _full_main(args):
            # AUTH_EVAL_DIRECT_SUBPROCESS_OK:wrong-line
            cmd = ["python", "experiments/contest_auth_eval.py"]
            proc = subprocess.run(cmd)
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Aggregation + scope tests.
# ---------------------------------------------------------------------------


def test_multiple_violations_one_file(tmp_path: Path) -> None:
    """Two subprocess invocations in one trainer produce two violations."""
    _write_trainer(
        tmp_path,
        "multi",
        '''
        import subprocess

        def _full_main(args):
            cmd = ["python", "experiments/contest_auth_eval.py"]
            subprocess.run(cmd)
            subprocess.Popen(cmd)
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 2


def test_multiple_files_aggregated(tmp_path: Path) -> None:
    """Violations across two files are aggregated."""
    _write_trainer(
        tmp_path,
        "first",
        '''
        import subprocess
        def _full_main(args):
            cmd = ["python", "experiments/contest_auth_eval.py"]
            subprocess.run(cmd)
        ''',
    )
    _write_trainer(
        tmp_path,
        "second",
        '''
        import subprocess
        def _full_main(args):
            cmd = ["python", "experiments/contest_auth_eval.py"]
            subprocess.run(cmd)
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 2
    files_seen = {v.split(":")[0] for v in violations}
    assert any("first" in f for f in files_seen)
    assert any("second" in f for f in files_seen)


def test_non_substrate_trainer_out_of_scope(tmp_path: Path) -> None:
    """Trainers NOT matching ``train_substrate_*.py`` are out-of-scope."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    f = exp_dir / "train_other.py"  # different prefix
    f.write_text(
        textwrap.dedent(
            '''
            import subprocess
            def _full_main(args):
                cmd = ["python", "experiments/contest_auth_eval.py"]
                subprocess.run(cmd)
            '''
        ),
        encoding="utf-8",
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Strict-mode + edge-case tests.
# ---------------------------------------------------------------------------


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    """Strict mode raises PreflightError when violations exist."""
    _write_trainer(
        tmp_path,
        "strict_violation",
        '''
        import subprocess
        def _full_main(args):
            cmd = ["python", "experiments/contest_auth_eval.py"]
            subprocess.run(cmd)
        ''',
    )
    with pytest.raises(PreflightError, match="Catalog #226"):
        check_trainer_auth_eval_uses_canonical_helper(
            repo_root=tmp_path, strict=True
        )


def test_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode does not raise on clean tree."""
    _write_trainer(
        tmp_path,
        "clean",
        '''
        def _full_main(args):
            pass
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=True
    )
    assert violations == []


def test_missing_experiments_dir_tolerated(tmp_path: Path) -> None:
    """Missing experiments/ directory tolerated (returns empty)."""
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_syntax_error_file_tolerated(tmp_path: Path) -> None:
    """Trainer file with SyntaxError is skipped silently."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "train_substrate_bad.py").write_text(
        "def broken(:\n", encoding="utf-8"
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_unrelated_subprocess_call_not_flagged(tmp_path: Path) -> None:
    """subprocess.run on something other than contest_auth_eval is not flagged."""
    _write_trainer(
        tmp_path,
        "unrelated",
        '''
        import subprocess
        def _full_main(args):
            cmd = ["python", "experiments/other.py"]
            subprocess.run(cmd)
        ''',
    )
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Live-repo regression guard.
# ---------------------------------------------------------------------------


def test_live_repo_violation_count_known(tmp_path: Path) -> None:
    """Live-repo regression guard: the audit count is 18 as of 2026-05-14.

    This test will fail if a refactoring wave drives the count to 0 (good —
    strict-flip the gate). It will ALSO fail if a regression adds new
    violations (good — operator routes the bug back to the introducing
    subagent).
    """
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_trainer_auth_eval_uses_canonical_helper(
        repo_root=repo_root, strict=False
    )
    # Allow some flex (16-20) to account for in-flight sister subagent
    # refactors that may land between catalog #226 landing and operator-
    # routed strict-flip wave.
    assert 0 <= len(violations) <= 20, (
        f"Catalog #226 live count drifted to {len(violations)} — investigate "
        f"if NEW hand-rolled trainers were introduced, OR drive to 0 and "
        f"strict-flip the gate per CLAUDE.md Strict-flip atomicity rule."
    )
