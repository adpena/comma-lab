# SPDX-License-Identifier: MIT
"""Catalog #408 — controls for `check_no_timed_shell_wrapper_without_group_kill`.

The bug class: ``subprocess.run(["bash", ...], timeout=N)`` kills the DIRECT
CHILD only, so the grandchild worker survives the timeout. MEASURED anchor,
ddm_cpu1 2026-08-20: the harness raised ``TimeoutExpired`` at 1799.99997045 s
and the decoder underneath ran to 4,369.600210089 s.

Every control here runs the REAL check against a REAL on-disk tree. The
headline is `test_positive_control_synthetic_violation_fires`: a guard that has
never been observed to fire is the #1086 bug class, so the un-cured shape is
asserted to FIRE before any "the cure passes" assertion is trusted.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_408_rationale_is_placeholder,
    _check_408_scan_source,
    check_no_timed_shell_wrapper_without_group_kill,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _tree(tmp_path: Path, relpath: str, body: str) -> Path:
    """Write `body` at `relpath` under a synthetic repo root and return the root."""
    target = tmp_path / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(body), encoding="utf-8")
    return tmp_path


def _run(root: Path) -> list[str]:
    return check_no_timed_shell_wrapper_without_group_kill(repo_root=root)


# ---------------------------------------------------------------------------
# POSITIVE CONTROL — the un-cured shape must FIRE.
# ---------------------------------------------------------------------------


def test_positive_control_synthetic_violation_fires(tmp_path: Path) -> None:
    """The exact ddm_cpu1 shape: `bash inflate.sh` under a timeout."""
    root = _tree(
        tmp_path,
        "tools/orphaner.py",
        """
        import subprocess

        def run_inflate(submission_dir):
            return subprocess.run(
                ["bash", "inflate.sh", str(submission_dir)],
                capture_output=True,
                text=True,
                timeout=1800,
            )
        """,
    )
    violations = _run(root)
    assert len(violations) == 1, violations
    assert "tools/orphaner.py:5" in violations[0]
    assert "run_in_process_group" in violations[0]
    assert "4,369.600210089" in violations[0]


@pytest.mark.parametrize(
    "argv0",
    ["'bash'", "'sh'", "'/bin/bash'", "'/usr/bin/env'", "'nohup'", "'timeout'"],
)
def test_every_shell_head_fires(tmp_path: Path, argv0: str) -> None:
    root = _tree(
        tmp_path,
        "tools/x.py",
        f"""
        import subprocess
        subprocess.run([{argv0}, "job.sh"], timeout=30)
        """,
    )
    assert len(_run(root)) == 1, f"{argv0} did not fire"


def test_dot_sh_script_without_shell_head_fires(tmp_path: Path) -> None:
    """`./inflate.sh` executed directly still forks its own children."""
    root = _tree(
        tmp_path,
        "tools/x.py",
        """
        import subprocess
        subprocess.run([str(runtime / "inflate.sh"), "a", "b"], timeout=60)
        """,
    )
    assert len(_run(root)) == 1


def test_variable_cmd_resolved_through_local_assignment_fires(tmp_path: Path) -> None:
    """The shape a module-wide name pool loses (measured on two real files)."""
    root = _tree(
        tmp_path,
        "tools/x.py",
        """
        import subprocess

        def other():
            command = ["git", "rev-parse", "HEAD"]
            return subprocess.run(command, timeout=10)

        def real():
            command = ["bash", "evaluate.sh", "--device", "cpu"]
            return subprocess.run(command, timeout=600)
        """,
    )
    violations = _run(root)
    assert len(violations) == 1, violations
    assert ":10" in violations[0], violations


def test_path_expression_head_fires(tmp_path: Path) -> None:
    """`inflate_sh = pkt / 'inflate.sh'` must be resolved transitively."""
    root = _tree(
        tmp_path,
        "src/thing.py",
        """
        import subprocess

        def go(packet_dir):
            inflate_sh = packet_dir / "inflate.sh"
            return subprocess.run([str(inflate_sh)], timeout=120)
        """,
    )
    assert len(_run(root)) == 1


def test_check_output_and_call_apis_fire(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        "tools/x.py",
        """
        import subprocess
        subprocess.check_output(["bash", "a.sh"], timeout=5)
        subprocess.call(["bash", "b.sh"], timeout=5)
        subprocess.check_call(["bash", "c.sh"], timeout=5)
        """,
    )
    assert len(_run(root)) == 3


# ---------------------------------------------------------------------------
# INVERSE CONTROLS — the cure and the true negatives must PASS.
# ---------------------------------------------------------------------------


def test_migrated_site_passes(tmp_path: Path) -> None:
    """The cure: routed through the canonical helper."""
    root = _tree(
        tmp_path,
        "tools/cured.py",
        """
        from tac.process_group_kill import run_in_process_group

        def run_inflate(submission_dir):
            return run_in_process_group(
                ["bash", "inflate.sh", str(submission_dir)],
                capture_output=True,
                text=True,
                timeout=1800,
            )
        """,
    )
    assert _run(root) == []


def test_leaf_binary_passes(tmp_path: Path) -> None:
    """`git`/`unzip`/`ffprobe` spawn nothing — no grandchild to reach."""
    root = _tree(
        tmp_path,
        "tools/leaf.py",
        """
        import subprocess
        subprocess.run(["git", "rev-parse", "HEAD"], timeout=10)
        subprocess.run(["unzip", "-o", "a.zip"], timeout=60)
        subprocess.run(["ffprobe", "-i", "v.mkv"], timeout=60)
        """,
    )
    assert _run(root) == []


def test_untimed_shell_call_passes(tmp_path: Path) -> None:
    """No timeout means no kill, so the class defect cannot fire."""
    root = _tree(
        tmp_path,
        "tools/untimed.py",
        """
        import subprocess
        subprocess.call(["bash", "lane.sh"])
        subprocess.run(["bash", "lane.sh"], check=False)
        """,
    )
    assert _run(root) == []


def test_waiver_respected(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        "tools/waived.py",
        """
        import subprocess
        subprocess.run(
            # GROUP_KILL_OK: `bash -n` only parses, it never executes a command.
            ["bash", "-n", "x.sh"],
            timeout=10,
        )
        """,
    )
    assert _run(root) == []


def test_placeholder_waiver_rejected(tmp_path: Path) -> None:
    """Catalog #287 sister discipline — a placeholder rationale is not a waiver."""
    root = _tree(
        tmp_path,
        "tools/fake_waiver.py",
        """
        import subprocess
        subprocess.run(
            # GROUP_KILL_OK: <rationale>
            ["bash", "x.sh"],
            timeout=10,
        )
        """,
    )
    assert len(_run(root)) == 1


@pytest.mark.parametrize("bad", ["", "<rationale>", "<reason>", "  TODO ", "tbd"])
def test_placeholder_rationales_enumerated(bad: str) -> None:
    assert _check_408_rationale_is_placeholder(bad)


def test_real_rationale_is_not_placeholder() -> None:
    assert not _check_408_rationale_is_placeholder(
        " `unzip` is a single leaf binary — it spawns nothing."
    )


def test_tests_and_vendored_paths_skipped(tmp_path: Path) -> None:
    """Tests drive both shapes as controls; intake clones stay pristine."""
    body = """
        import subprocess
        subprocess.run(["bash", "x.sh"], timeout=10)
        """
    root = tmp_path
    for rel in (
        "src/tac/tests/test_something.py",
        "tools/test_helper.py",
        "experiments/results/public_pr95_intake_20260504/source/run.py",
        "tools/vendored/dep.py",
    ):
        _tree(root, rel, body)
    assert _run(root) == []


def test_strict_raises_and_nonstrict_returns(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        "tools/x.py",
        """
        import subprocess
        subprocess.run(["bash", "x.sh"], timeout=10)
        """,
    )
    assert len(check_no_timed_shell_wrapper_without_group_kill(repo_root=root)) == 1
    with pytest.raises(PreflightError, match="Catalog #408"):
        check_no_timed_shell_wrapper_without_group_kill(repo_root=root, strict=True)


# ---------------------------------------------------------------------------
# DENOMINATOR CONTROL — the scan must not pass by matching nothing.
# ---------------------------------------------------------------------------


def test_scanner_denominator_is_not_vacuous() -> None:
    """A silent instrument reads as a pass. Prove the scanner still sees shapes.

    ad1 Item 2's guard used a hand-typed denominator of 2 and was blind to a
    third emitter. This asserts the SCANNER itself still fires on the canonical
    shape, so a future refactor that breaks detection fails here instead of
    reporting a clean repo.
    """
    fires = _check_408_scan_source(
        'import subprocess\nsubprocess.run(["bash", "inflate.sh"], timeout=1800)\n'
    )
    assert len(fires) == 1
    assert fires[0][0] == 2

    silent = _check_408_scan_source(
        'import subprocess\nsubprocess.run(["git", "status"], timeout=10)\n'
    )
    assert silent == []


def test_live_repo_count_is_zero() -> None:
    """Live count over the real tree.

    ddm_kg1 re-derived the population at 48 un-migrated shell-shaped sites
    (the ddm_ad1 memo had named 26) and drove it to 0 by migration + waiver.
    A regression here means a new timed shell wrapper landed un-cured.
    """
    violations = check_no_timed_shell_wrapper_without_group_kill(repo_root=REPO_ROOT)
    assert violations == [], (
        f"{len(violations)} un-cured timed shell-wrapper site(s):\n  "
        + "\n  ".join(v[:300] for v in violations[:12])
    )
