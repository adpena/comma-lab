"""Regression tests for scripts/remote_pose_tto_only_bootstrap.sh.

LANE-B (2026-04-26) burned 6.5h + $2 because the bootstrap had three
silent-failure cascades:

  1. PyTorch container does not ship `zip`; the `zip ... archive.zip`
     command failed with 'command not found'.
  2. `set -uo pipefail` did NOT include `-e`, so the failed `zip` command
     did not abort the script.
  3. `ARCHIVE_BYTES=$(stat -c '%s' archive.zip)` returned an empty string
     (no file). The empty value was then passed to auth_eval as
     `--archive-size-bytes ""` which crashed argparse — but only AFTER
     the heartbeat said "STAGE 4 done".

These tests assert the fixes:
  - script uses `set -euo pipefail` (-e flag present)
  - archive is built via Python `zipfile`, not the missing `zip` binary
  - empty/zero ARCHIVE_BYTES hard-fails BEFORE auth eval is called
  - auth eval log is validated for RESULT_JSON before the script exits 0
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_pose_tto_only_bootstrap.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


def test_script_exists():
    assert SCRIPT.exists(), f"missing canonical bootstrap: {SCRIPT}"


def test_set_e_present(script_text: str):
    """The fix-everything memory says: silent-failure cascades are how
    we lost LANE-B. set -e must be on so a failed command aborts."""
    # Look for `set -e` in any combined form (set -e, set -eu, set -euo, etc.)
    has_set_e = bool(re.search(r"^set -[eu]*e[uo]*\s", script_text, re.MULTILINE))
    assert has_set_e, "script must use `set -e` (or -euo / -eu) to abort on first failure"


def test_set_pipefail_present(script_text: str):
    """pipefail catches failures inside `cmd | tee log` — without it
    a failed cmd whose output goes through tee will look successful."""
    assert "pipefail" in script_text, "script must use pipefail"


def test_no_shell_zip_binary(script_text: str):
    """The PyTorch container has no `zip`. Use Python zipfile instead.

    Specifically: there should be no line that invokes `zip` as a shell
    command. (The word may appear in comments or 'zipfile' module names.)"""
    # Strip comments and string literals to avoid false positives
    code_lines = []
    for line in script_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    # Look for `zip ` at the start of a command (after whitespace), but not
    # `zipfile`, not `unzip`, not `gzip`, not `--zip`, etc.
    bad_match = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad_match, (
        f"script should not invoke shell `zip` binary "
        f"(missing on PyTorch container); use Python zipfile instead. "
        f"Match: {bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    """Affirmative test: the archive must be built via Python's zipfile
    module so we don't rely on the apt `zip` binary."""
    assert "zipfile.ZipFile" in script_text, (
        "archive build must use python zipfile.ZipFile (no apt dep)"
    )


def test_archive_size_validated_before_auth_eval(script_text: str):
    """ARCHIVE_BYTES must be checked for empty/zero BEFORE being passed
    to auth_eval. This was the LANE-B failure mode — empty value
    crashed argparse with 'expected one argument'."""
    # Find the auth eval invocation — anchor on the actual python call,
    # not the first `--archive-size-bytes` mention (which is in a comment).
    auth_idx = script_text.find("auth_eval_renderer.py")
    assert auth_idx > 0, "auth eval must invoke auth_eval_renderer.py"

    # Walk back to find the validation block — must be between archive
    # build and auth eval, must check ARCHIVE_BYTES.
    pre = script_text[:auth_idx]
    # Look for either `[ -z "${ARCHIVE_BYTES` or `if [ "$ARCHIVE_BYTES" -le 0`
    # — both are valid hard-fail patterns.
    has_empty_check = bool(re.search(r'-z\s+"\$\{?ARCHIVE_BYTES', pre))
    has_zero_check = bool(re.search(r'\$ARCHIVE_BYTES"?\s*-le\s*0', pre))
    assert has_empty_check or has_zero_check, (
        "ARCHIVE_BYTES must be validated for empty/zero BEFORE auth_eval — "
        "this was the LANE-B silent-failure cascade root."
    )


def test_auth_eval_log_validated(script_text: str):
    """The script must check that auth_eval actually produced a
    RESULT_JSON line — otherwise the run is just a $5 receipt for
    nothing (no-wasted-resources rule)."""
    assert re.search(r"grep\s+-q\s+'?\^?RESULT_JSON", script_text), (
        "must validate auth_eval.log contains RESULT_JSON before exit 0"
    )


def test_archive_input_files_asserted(script_text: str):
    """The Python zipfile builder must assert each input file exists
    BEFORE writing — otherwise a missing input silently produces a tiny
    archive (e.g. only optimized_poses.bin if renderer.bin is missing)."""
    # Find the python -c block
    py_idx = script_text.find("zipfile.ZipFile")
    assert py_idx > 0
    block_end = script_text.find('print(', py_idx)
    block = script_text[py_idx:block_end]
    assert "assert os.path.isfile" in block, (
        "zipfile builder must assert each input file exists"
    )
