# SPDX-License-Identifier: MIT
"""Direct-parent custody for governed profiling and extraction entrypoints.

The ordinary admission marker says only that *some* launcher stamped the child
environment.  Large artifact producers need stronger evidence: while the child
is running, its direct parent must be this repository's ``tools/safe_run.py``;
safe_run's command must be the child's exact argv; and the outer RSS and timeout
caps must equal the caps requested by the child.  This module performs that
attestation without treating it as a completed safe-run status receipt.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import stat
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from tac import admission_guard

ATTESTATION_SCHEMA: Final = "governed_safe_run_parent_attestation.v1"
_TRUTHY: Final = {"1", "true", "yes", "on"}
_FORBIDDEN_OUTER_FLAGS: Final = {
    "--skip-admission-gate",
    "--admission-override-rationale",
}


class GovernedAdmissionError(RuntimeError):
    """Fail-closed direct-parent, cap, argv, or source-custody error."""


def _source_file_row(path: Path, *, role: str) -> dict[str, Any]:
    """Hash one non-linked regular source file without following a link."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise GovernedAdmissionError(f"{role} source custody is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise GovernedAdmissionError(f"{role} source must be a non-symlink regular file")
    if metadata.st_nlink != 1:
        raise GovernedAdmissionError(f"{role} source must have exactly one hard link")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GovernedAdmissionError(f"{role} source could not be hashed") from exc
    after = path.lstat()
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
    ) != (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_nlink,
    ):
        raise GovernedAdmissionError(f"{role} source changed while being hashed")
    return {
        "path": str(path),
        "bytes": metadata.st_size,
        "sha256": digest.hexdigest(),
    }


def _linux_process_argv(pid: int) -> list[str]:
    payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    if not payload or not payload.endswith(b"\0"):
        raise GovernedAdmissionError("direct parent /proc argv is unavailable or truncated")
    try:
        return [token.decode("utf-8") for token in payload[:-1].split(b"\0")]
    except UnicodeDecodeError as exc:
        raise GovernedAdmissionError("direct parent argv is not UTF-8") from exc


def _darwin_process_argv(pid: int) -> list[str]:
    """Read KERN_PROCARGS2, preserving OS argv token boundaries."""

    libc = ctypes.CDLL("libc.dylib", use_errno=True)
    mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2, pid
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 or size.value <= 4:
        raise GovernedAdmissionError("direct parent KERN_PROCARGS2 size lookup failed")
    buffer = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
        raise GovernedAdmissionError("direct parent KERN_PROCARGS2 read failed")
    payload = buffer.raw[: size.value]
    argc = struct.unpack_from("=i", payload)[0]
    if argc <= 0:
        raise GovernedAdmissionError("direct parent KERN_PROCARGS2 argc is malformed")
    cursor = struct.calcsize("=i")
    executable_end = payload.find(b"\0", cursor)
    if executable_end < 0:
        raise GovernedAdmissionError("direct parent KERN_PROCARGS2 executable is malformed")
    cursor = executable_end
    while cursor < len(payload) and payload[cursor] == 0:
        cursor += 1
    values: list[str] = []
    for _ in range(argc):
        end = payload.find(b"\0", cursor)
        if end < 0:
            raise GovernedAdmissionError("direct parent KERN_PROCARGS2 argv is truncated")
        try:
            values.append(payload[cursor:end].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise GovernedAdmissionError("direct parent argv is not UTF-8") from exc
        cursor = end + 1
    return values


def _read_process_argv(pid: int) -> list[str]:
    if sys.platform == "linux":
        return _linux_process_argv(pid)
    if sys.platform == "darwin":
        return _darwin_process_argv(pid)
    raise GovernedAdmissionError(f"direct parent argv custody is unsupported on {sys.platform}")


def _option_value(tokens: Sequence[str], option: str) -> str:
    values: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == option:
            if index + 1 >= len(tokens):
                raise GovernedAdmissionError(f"safe_run {option} is missing its value")
            values.append(tokens[index + 1])
            index += 2
            continue
        prefix = f"{option}="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
        index += 1
    if len(values) != 1:
        raise GovernedAdmissionError(f"safe_run must carry exactly one {option}")
    return values[0]


def _normalized_child_argv(value: Sequence[str]) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise GovernedAdmissionError("exact child argv must be a nonempty list or tuple")
    argv = list(value)
    if any(not isinstance(token, str) or not token for token in argv):
        raise GovernedAdmissionError("exact child argv contains an invalid token")
    return argv


def attest_safe_run_parent(
    *,
    exact_child_argv: Sequence[str],
    rss_cap_mb: int,
    timeout_seconds: float,
    repo_root: str | Path,
    env: Mapping[str, str] | None = None,
    parent_pid: int | None = None,
) -> dict[str, Any]:
    """Attest the running child's direct safe-run parent and exact outer caps.

    The return value is command/source custody captured *inside* the running
    child.  A completed safe-run status is necessarily emitted later by the
    parent and must be stored as a separate receipt.
    """

    child_argv = _normalized_child_argv(exact_child_argv)
    if type(rss_cap_mb) is not int or rss_cap_mb <= 0:
        raise GovernedAdmissionError("rss_cap_mb must be a positive integer")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise GovernedAdmissionError("timeout_seconds must be a positive finite number")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise GovernedAdmissionError("timeout_seconds must be a positive finite number")

    environment = os.environ if env is None else env
    marker = (environment.get(admission_guard.GOVERNED_MARKER_ENV, "") or "").strip().lower()
    if marker not in _TRUTHY:
        raise GovernedAdmissionError("governed admission marker is absent")
    if (environment.get(admission_guard.BYPASS_OVERRIDE_ENV, "") or "").strip():
        raise GovernedAdmissionError("admission bypass environment is forbidden")

    direct_parent_pid = os.getppid()
    if parent_pid is not None:
        if type(parent_pid) is not int or parent_pid <= 0 or parent_pid != direct_parent_pid:
            raise GovernedAdmissionError("attested process is not the direct parent")
    else:
        parent_pid = direct_parent_pid
    try:
        parent_argv = _read_process_argv(parent_pid)
    except OSError as exc:
        raise GovernedAdmissionError("direct parent argv custody is unavailable") from exc
    if not parent_argv or any(not token for token in parent_argv):
        raise GovernedAdmissionError("direct parent argv is empty or malformed")

    root = Path(repo_root).expanduser().resolve(strict=True)
    safe_run_path = root / "tools/safe_run.py"
    helper_path = Path(__file__).resolve(strict=True)
    admission_path = Path(admission_guard.__file__).resolve(strict=True)
    expected_safe_run = safe_run_path.resolve(strict=True)
    if len(parent_argv) >= 2:
        try:
            parent_interpreter = Path(parent_argv[0]).expanduser().resolve(strict=True)
            child_interpreter = Path(child_argv[0]).expanduser().resolve(strict=True)
            parent_tool = Path(parent_argv[1]).expanduser().resolve(strict=True)
        except OSError as exc:
            raise GovernedAdmissionError("safe_run parent source custody is unavailable") from exc
        if parent_interpreter != child_interpreter:
            raise GovernedAdmissionError("safe_run parent and child must use the same exact Python runtime")
        if parent_tool != expected_safe_run:
            raise GovernedAdmissionError("direct parent is not the exact repo safe_run Python command")
        outer_and_child = parent_argv[2:]
    else:
        raise GovernedAdmissionError("direct parent is not a safe_run Python command")
    if outer_and_child.count("--") != 1:
        raise GovernedAdmissionError("safe_run parent must use exactly one explicit command separator")
    separator = outer_and_child.index("--")
    outer = outer_and_child[:separator]
    parent_child = outer_and_child[separator + 1 :]
    for token in outer:
        flag = token.split("=", 1)[0]
        if flag in _FORBIDDEN_OUTER_FLAGS or "override" in flag.lower():
            raise GovernedAdmissionError(f"safe_run admission escape is forbidden: {flag}")
    if parent_child != child_argv:
        raise GovernedAdmissionError("safe_run child argv does not exactly match the running child argv")

    try:
        outer_rss = int(_option_value(outer, "--rss-mb"))
        outer_timeout = float(_option_value(outer, "--timeout"))
    except ValueError as exc:
        raise GovernedAdmissionError("safe_run resource cap value is malformed") from exc
    if outer_rss != rss_cap_mb:
        raise GovernedAdmissionError("safe_run outer RSS cap differs from the child request")
    if not math.isfinite(outer_timeout) or outer_timeout != timeout:
        raise GovernedAdmissionError("safe_run outer timeout differs from the child request")

    return {
        "schema": ATTESTATION_SCHEMA,
        "attestation_scope": "DIRECT_PARENT_COMMAND_AT_CHILD_START_NOT_COMPLETED_STATUS",
        "parent_pid": parent_pid,
        "parent_python_executable": str(parent_interpreter),
        "parent_exact_argv": parent_argv,
        "child_exact_argv": child_argv,
        "outer_resource_caps": {
            "rss_cap_mb": outer_rss,
            "timeout_seconds": outer_timeout,
        },
        "governed_marker_present": True,
        "admission_bypass_present": False,
        "completed_safe_run_status_receipt": None,
        "source_custody": {
            "governed_profile_admission": _source_file_row(helper_path, role="admission helper"),
            "safe_run": _source_file_row(expected_safe_run, role="safe_run"),
            "admission_guard": _source_file_row(admission_path, role="admission guard"),
        },
    }


__all__ = [
    "ATTESTATION_SCHEMA",
    "GovernedAdmissionError",
    "attest_safe_run_parent",
]
