"""Executed positive controls for the emitted ``inflate.sh`` (ddm_si1, task #929).

Bug class: *failure and success emit the same symbol*.  ``ddm_ob1`` (2026-08-03)
recorded an inflate of the shipped ``pu2`` archive that reported **exit code 0
while having produced nothing** (``python: command not found``), caught only by
reading a zero-byte log.

This module pins the measured decomposition of that incident, because the
one-line report is narrower than it reads:

* the emitted script was **never** the liar -- with ``set -euo pipefail`` a bare
  ``python`` that is absent fails loudly with 127 in the foreground
  (:func:`test_negative_control_old_script_failed_loudly_in_foreground`);
* the **trigger** was the bare ``python``, which is absent on any host that
  ships only ``python3``.  That is now resolved fail-closed;
* the **amplifier** was a backgrounding launcher, whose exit status is the
  launcher's and not the job's
  (:func:`test_amplifier_backgrounded_launcher_reports_zero_for_a_failed_job`).
  The amplifier is NOT fixed by this module and is recorded here so the
  remaining exposure is visible rather than implied.

A guard without a demonstrated failing case is not landed, so every control
below actually executes the script.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Both tools that emit a submission-side ``inflate.sh``.
EMITTERS = (
    ("pb1_p5", REPO_ROOT / "tools" / "pb1_p5_byte_close_and_eval.py", r'INFLATE_SH = """(.*?)"""'),
    ("rehearse_tr1", REPO_ROOT / "tools" / "rehearse_ddm_tr1_runtime.py", r'INFLATE_SH = b"""\\\n(.*?)"""'),
)

# A runner whose exit code the control chooses, so rc propagation is observable.
RUNNER = """import os, pathlib, sys
code = int(os.environ.get("EXITWITH", "0"))
if code == 0:
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    (out / "0.raw").write_bytes(b"decoded")
sys.exit(code)
"""

OLD_INFLATE_SH = (
    '#!/usr/bin/env bash\n'
    'set -euo pipefail\n'
    'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
    'python "$HERE/inflate_runner.py" "$1" "$2" "$3"\n'
)


def _emitted_script(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(), re.S)
    assert match is not None, f"could not extract INFLATE_SH from {path}"
    return match.group(1)


def _stage(tmp_path: Path, script_text: str) -> Path:
    (tmp_path / "inflate.sh").write_text(script_text)
    (tmp_path / "inflate_runner.py").write_text(RUNNER)
    return tmp_path / "inflate.sh"


def _no_interpreter_path(tmp_path: Path) -> str:
    """A PATH containing the shell utilities but no ``python``/``python3``."""
    bin_dir = tmp_path / "nopython"
    bin_dir.mkdir(exist_ok=True)
    for utility in ("bash", "env", "uname", "dirname", "pwd"):
        resolved = shutil.which(utility)
        if resolved:
            link = bin_dir / utility
            if not link.exists():
                link.symlink_to(resolved)
    assert shutil.which("python3", path=str(bin_dir)) is None
    assert shutil.which("python", path=str(bin_dir)) is None
    return str(bin_dir)


def _run(script: Path, out: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), "archive", str(out), "names.txt"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


@pytest.mark.parametrize(("label", "path", "pattern"), EMITTERS, ids=[e[0] for e in EMITTERS])
def test_emitted_script_carries_no_bare_python(label: str, path: Path, pattern: str) -> None:
    """The trigger itself: no emitter may ship an unqualified interpreter."""
    script = _emitted_script(path, pattern)
    assert not re.search(r"^\s*python\s+\"\$HERE", script, re.M), (
        f"{label} emits a bare `python` invocation; it must resolve the "
        "interpreter explicitly and fail closed"
    )
    assert "exit 127" in script, f"{label} must refuse explicitly when no interpreter exists"
    assert "exec" in script, f"{label} must exec so the runner's rc is not swallowed"


@pytest.mark.parametrize(("label", "path", "pattern"), EMITTERS, ids=[e[0] for e in EMITTERS])
def test_control_a_succeeds_and_produces_output(
    label: str, path: Path, pattern: str, tmp_path: Path
) -> None:
    """POSITIVE control: with an interpreter present the script must work."""
    script = _stage(tmp_path, _emitted_script(path, pattern))
    out = tmp_path / "outA"
    env = dict(os.environ, PYTHON=sys.executable, EXITWITH="0")
    result = _run(script, out, env)
    assert result.returncode == 0, result.stderr
    assert (out / "0.raw").read_bytes() == b"decoded"


@pytest.mark.parametrize(("label", "path", "pattern"), EMITTERS, ids=[e[0] for e in EMITTERS])
def test_control_b_refuses_when_no_interpreter_exists(
    label: str, path: Path, pattern: str, tmp_path: Path
) -> None:
    """The ob1 case, executed: a missing interpreter must REFUSE, not pass."""
    script = _stage(tmp_path, _emitted_script(path, pattern))
    out = tmp_path / "outB"
    result = _run(script, out, {"PATH": _no_interpreter_path(tmp_path)})
    assert result.returncode == 127, f"expected refusal, got rc={result.returncode}"
    assert "no Python interpreter" in result.stderr, "refusal must be diagnosed, not silent"
    assert not (out / "0.raw").exists(), "a refused inflate must leave no output"


@pytest.mark.parametrize(("label", "path", "pattern"), EMITTERS, ids=[e[0] for e in EMITTERS])
def test_control_a2_discovers_python3_on_a_python3_only_host(
    label: str, path: Path, pattern: str, tmp_path: Path
) -> None:
    """The real contest scenario: a host exposing ``python3`` but no ``python``.

    This is the case the pre-fix script could not survive, and it is distinct
    from control A, which exercises the ``PYTHON`` env-var branch rather than
    the discovery loop.
    """
    script = _stage(tmp_path, _emitted_script(path, pattern))
    bin_dir = Path(_no_interpreter_path(tmp_path))
    (bin_dir / "python3").symlink_to(sys.executable)
    assert shutil.which("python", path=str(bin_dir)) is None, "control requires no bare python"
    out = tmp_path / "outA2"
    result = _run(script, out, {"PATH": str(bin_dir), "EXITWITH": "0"})
    assert result.returncode == 0, f"python3-only host failed: {result.stderr}"
    assert (out / "0.raw").read_bytes() == b"decoded"


@pytest.mark.parametrize(("label", "path", "pattern"), EMITTERS, ids=[e[0] for e in EMITTERS])
def test_control_c_propagates_a_nonzero_runner_exit(
    label: str, path: Path, pattern: str, tmp_path: Path
) -> None:
    """A failing runner must surface its own rc, not a laundered zero."""
    script = _stage(tmp_path, _emitted_script(path, pattern))
    env = dict(os.environ, PYTHON=sys.executable, EXITWITH="3")
    result = _run(script, tmp_path / "outC", env)
    assert result.returncode == 3, f"rc swallowed: got {result.returncode}"


def test_negative_control_old_script_failed_loudly_in_foreground(tmp_path: Path) -> None:
    """REFUTATION, executed: the pre-fix script was not the component that lied.

    Run in the foreground it returns 127 exactly as it should.  This is why the
    fix above removes a portability trigger and not a dishonest exit code.
    """
    script = _stage(tmp_path, OLD_INFLATE_SH)
    result = _run(script, tmp_path / "outD", {"PATH": _no_interpreter_path(tmp_path)})
    assert result.returncode == 127


def test_amplifier_backgrounded_launcher_reports_zero_for_a_failed_job(tmp_path: Path) -> None:
    """The component that actually lied, pinned as a live reproduction.

    Backgrounding detaches the job's exit status from the launcher's.  The
    launcher exits 0 while the job fails and writes nothing.  This is UNFIXED by
    the inflate.sh change: any backgrounded invocation of *any* script has this
    property.  The cure is a completion marker carrying the job's real rc, which
    no canonical helper currently provides -- so this control exists to keep the
    exposure measured instead of forgotten.
    """
    script = _stage(tmp_path, OLD_INFLATE_SH)
    out = tmp_path / "outE"
    log = tmp_path / "inflate.log"
    launcher = subprocess.run(
        ["bash", "-c", f"bash '{script}' archive '{out}' names.txt > '{log}' 2>&1 &"],
        capture_output=True,
        text=True,
        env={"PATH": _no_interpreter_path(tmp_path)},
        timeout=120,
    )
    assert launcher.returncode == 0, "launcher unexpectedly surfaced the failure"
    assert not (out / "0.raw").exists(), "the job did not actually fail; control is void"
