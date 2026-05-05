#!/usr/bin/env python3
"""Prepare an auditable, reproducible Lightning Studio workspace.

This is the OSS/reproducible replacement for one-off ``rsync`` snippets. It
syncs a declared source set plus explicit artifacts to a Lightning Studio,
records SHA-256/byte manifests locally and remotely, verifies the remote bytes,
and can install the project from ``uv.lock`` with the runtime extra.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parent.parent / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.deploy.lightning.defaults import (  # noqa: E402
    DEFAULT_LIGHTNING_REMOTE_PACT,
    default_ssh_target,
)
from tac.repo_io import json_text, sha256_file, write_json  # noqa: E402

DEFAULT_STATE_DIR = REPO_ROOT / ".omx/state"
DEFAULT_REMOTE_PACT = DEFAULT_LIGHTNING_REMOTE_PACT
DEFAULT_SOURCE_PATHS = (
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "src",
    "experiments",
    "scripts",
    "submissions/robust_current/README.md",
    "submissions/robust_current/analyze_roi.py",
    "submissions/robust_current/config.env",
    "submissions/robust_current/config_env.py",
    "submissions/robust_current/inflate.sh",
    "submissions/robust_current/inflate_postfilter.py",
    "submissions/robust_current/inflate_renderer.py",
    "submissions/robust_current/inflate_renderer_grayscale.py",
    "submissions/robust_current/inflate_segmap.py",
    "submissions/robust_current/inflate_segmap_arithmetic.py",
    "submissions/robust_current/inflate_segmap_film_canvas.py",
    "submissions/robust_current/unpack_renderer_payload.py",
    "submissions/robust_current/runner.py",
    "submissions/robust_current/sky_degrade.py",
    "docs",
)
EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}
EXCLUDED_PREFIXES = (
    ".recovery_quarantine_",
    ".omx/state/",
    ".omx/tmp/",
    ".omx/cache/",
    "experiments/precomputed_local/",
    "experiments/results/",
    "experiments/raft_flow.pt",
    "experiments/postfilter_weights/",
    "submissions/robust_current/eval_runs/",
)
EXCLUDED_SOURCE_SUFFIXES = (
    ".bin",
    ".mkv",
    ".pt",
    ".raw",
    ".zip",
)
EXCLUDED_HIDDEN_SOURCE_NAMES = {
    ".DS_Store",
}
REQUIREMENTS_MODES = ("uv-sync", "verify-only", "no-install")
BANNED_LIGHTNING_VERSIONS = {"2.6.2", "2.6.3"}
MINI_SHAI_HULUD_REPO_INDICATORS = (
    ".claude/router_runtime.js",
    ".claude/setup.mjs",
    ".vscode/setup.mjs",
    ".github/workflows/format-check.yml",
)
SSH_AUTH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "PasswordAuthentication=no",
    "-o",
    "KbdInteractiveAuthentication=no",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
    "-o",
    "TCPKeepAlive=yes",
    "-o",
    "ConnectionAttempts=3",
)

RunFn = Callable[..., subprocess.CompletedProcess[str]]


class LightningSshPreflightError(RuntimeError):
    """Raised when Lightning SSH auth is not reproducibly configured."""

    def __init__(self, message: str, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


class LightningRuntimePreflightError(RuntimeError):
    """Raised when a reachable Lightning Studio runtime is not fit for use."""

    def __init__(self, message: str, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


class LightningLocalUvLockError(RuntimeError):
    """Raised when local uv.lock cannot support remote `uv sync --locked`."""

    def __init__(self, message: str, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    dry_run: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, cwd=cwd, text=True, check=check)  # subprocess-no-check-OK: check is parametrized by caller (defaults to True at line 104)


def _capture(cmd: list[str], *, cwd: Path = REPO_ROOT) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None
    return proc.stdout.strip()


def ensure_local_uv_lock_current(
    *,
    cwd: Path = REPO_ROOT,
    runner: RunFn = subprocess.run,
) -> dict[str, Any]:
    """Fail before SSH if remote `uv sync --locked` would reject uv.lock."""

    cmd = ["uv", "lock", "--check"]
    diagnostic: dict[str, Any] = {
        "schema_version": 1,
        "tool": "lightning_repro_workspace_local_uv_lock_preflight",
        "recorded_at_utc": _utc_now(),
        "cwd": str(cwd),
        "command": cmd,
    }
    try:
        proc = runner(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        diagnostic.update({"status": "fail", "error": repr(exc)})
        raise LightningLocalUvLockError(
            "FATAL: --requirements-mode uv-sync requires local `uv lock --check`, but uv is not on PATH",
            diagnostic,
        ) from exc

    diagnostic.update(
        {
            "status": "ok" if proc.returncode == 0 else "fail",
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    )
    if proc.returncode != 0:
        raise LightningLocalUvLockError(
            "FATAL: local uv.lock is stale for --requirements-mode uv-sync; "
            "run `uv lock` or relaunch with `--requirements-mode no-install` "
            "when using a previously verified remote exact-eval environment.",
            diagnostic,
        )
    return diagnostic


def _ssh_run(
    cmd: list[str],
    *,
    runner: RunFn = subprocess.run,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return runner(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def _ssh_options(connect_timeout: int | None) -> list[str]:
    options = list(SSH_AUTH_OPTIONS)
    if connect_timeout is not None:
        if connect_timeout <= 0:
            raise ValueError("ssh_connect_timeout must be positive")
        options.extend(["-o", f"ConnectTimeout={int(connect_timeout)}"])
    return options


def _ssh_command(ssh_bin: str, target: str, remote_command: str, *, connect_timeout: int | None) -> list[str]:
    return [ssh_bin, *_ssh_options(connect_timeout), target, remote_command]


def _scp_command(scp_bin: str, source: str | Path, dest: str, *, connect_timeout: int | None) -> list[str]:
    return [scp_bin, *_ssh_options(connect_timeout), str(source), dest]


def _rsync_command(
    source_file_list: Path,
    remote: str,
    remote_pact: str,
    *,
    connect_timeout: int | None,
) -> list[str]:
    ssh_transport = " ".join(shlex.quote(part) for part in ["ssh", *_ssh_options(connect_timeout)])
    return [
        "rsync",
        "-a",
        "-e",
        ssh_transport,
        "--files-from",
        str(source_file_list),
        "./",
        f"{remote}:{remote_pact}/",
    ]


def _expand_ssh_path(value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    return str(Path(expanded).resolve())


def _public_key_for_identity(identity_file: str) -> str:
    return _expand_ssh_path(identity_file) + ".pub"


def _fingerprint_public_key(
    public_key_file: str,
    *,
    runner: RunFn = subprocess.run,
) -> str | None:
    path = Path(public_key_file)
    if not path.is_file():
        return None
    proc = _ssh_run(["ssh-keygen", "-lf", str(path)], runner=runner)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _parse_ssh_g(stdout: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key, sep, value = line.partition(" ")
        if not sep:
            continue
        values.setdefault(key.lower(), []).append(value.strip())
    return values


def _last_ssh_config_value(values: dict[str, list[str]], key: str) -> str | None:
    entries = values.get(key)
    return entries[-1] if entries else None


def lightning_ssh_diagnostic(
    remote: str,
    *,
    ssh_bin: str = "ssh",
    connect: bool,
    connect_timeout: int = 15,
    runner: RunFn = subprocess.run,
) -> dict[str, Any]:
    """Return sanitized SSH config/auth diagnostics for a Lightning target."""

    target = str(remote).strip()
    if not target or any(ch in target for ch in "\r\n\0"):
        raise ValueError("Lightning SSH target must be non-empty and must not contain control characters")
    if target == "ssh.lightning.ai":
        raise ValueError(
            "Lightning SSH target must be a ~/.ssh/config alias or user-qualified target, not bare ssh.lightning.ai"
        )
    diagnostic: dict[str, Any] = {
        "schema_version": 1,
        "target": target,
        "ssh_bin": ssh_bin,
        "auth_probe_requested": bool(connect),
    }
    config_proc = _ssh_run([ssh_bin, "-G", target], runner=runner)
    diagnostic["ssh_g_returncode"] = config_proc.returncode
    if config_proc.returncode == 0:
        values = _parse_ssh_g(config_proc.stdout)
        diagnostic["resolved"] = {
            "host": _last_ssh_config_value(values, "host"),
            "hostname": _last_ssh_config_value(values, "hostname"),
            "user": _last_ssh_config_value(values, "user"),
            "port": _last_ssh_config_value(values, "port"),
            "identitiesonly": _last_ssh_config_value(values, "identitiesonly"),
            "stricthostkeychecking": _last_ssh_config_value(values, "stricthostkeychecking"),
            "userknownhostsfile": _last_ssh_config_value(values, "userknownhostsfile"),
        }
        identity_files = values.get("identityfile", [])
        identity_records = []
        seen_expanded: set[str] = set()
        duplicate_expanded: list[str] = []
        for identity in identity_files:
            expanded = _expand_ssh_path(identity)
            if expanded in seen_expanded:
                duplicate_expanded.append(expanded)
            seen_expanded.add(expanded)
            public_key = _public_key_for_identity(identity)
            identity_records.append(
                {
                    "identity_file": identity,
                    "expanded_identity_file": expanded,
                    "identity_file_exists": Path(expanded).is_file(),
                    "public_key_file": public_key,
                    "public_key_exists": Path(public_key).is_file(),
                    "public_key_fingerprint": _fingerprint_public_key(public_key, runner=runner),
                }
            )
        diagnostic["identity_files"] = identity_records
        diagnostic["duplicate_expanded_identity_files"] = sorted(set(duplicate_expanded))
    else:
        diagnostic["ssh_g_stderr"] = config_proc.stderr.strip()

    if connect:
        probe_cmd = _ssh_command(ssh_bin, target, "true", connect_timeout=connect_timeout)
        probe_proc = _ssh_run(probe_cmd, runner=runner)
        diagnostic["auth_probe"] = {
            "returncode": probe_proc.returncode,
            "stdout": probe_proc.stdout.strip(),
            "stderr": probe_proc.stderr.strip(),
        }
        diagnostic["status"] = "ok" if probe_proc.returncode == 0 else "fail"
    else:
        diagnostic["status"] = "not_probed"
    return diagnostic


def _format_lightning_ssh_failure(diagnostic: dict[str, Any]) -> str:
    resolved = diagnostic.get("resolved") if isinstance(diagnostic.get("resolved"), dict) else {}
    identities = diagnostic.get("identity_files") if isinstance(diagnostic.get("identity_files"), list) else []
    probe = diagnostic.get("auth_probe") if isinstance(diagnostic.get("auth_probe"), dict) else {}
    lines = [
        f"Lightning SSH preflight failed for target {diagnostic.get('target')!r}.",
        "This blocks reproducible staging/harvest before any rsync, scp, or Batch Job submission.",
    ]
    if resolved:
        lines.append(
            "Resolved SSH endpoint: "
            f"user={resolved.get('user')!r} host={resolved.get('hostname')!r} "
            f"identitiesonly={resolved.get('identitiesonly')!r} "
            f"strict_host_key_checking={resolved.get('stricthostkeychecking')!r}."
        )
    if identities:
        lines.append("Resolved identity/public-key candidates:")
        for item in identities:
            lines.append(
                "  - "
                f"identity={item.get('expanded_identity_file')} "
                f"exists={item.get('identity_file_exists')} "
                f"public_key={item.get('public_key_file')} "
                f"public_key_exists={item.get('public_key_exists')} "
                f"fingerprint={item.get('public_key_fingerprint')}"
            )
    duplicates = diagnostic.get("duplicate_expanded_identity_files")
    if duplicates:
        lines.append(f"Duplicate resolved IdentityFile entries: {duplicates!r}.")
    policy_violations = diagnostic.get("policy_violations")
    if policy_violations:
        lines.append(f"SSH policy violations: {policy_violations!r}.")
    stderr = str(probe.get("stderr") or "").strip()
    if stderr:
        lines.append(f"SSH stderr: {stderr}")
    lines.extend(
        [
            "Fix guidance:",
            "  1. Add the selected *.pub key to the Lightning Studio/account SSH keys.",
            "  2. Keep ~/.ssh/config on an alias with HostName ssh.lightning.ai, the Studio SSH User, IdentityFile, IdentitiesOnly yes, and BatchMode yes.",
            "  3. Verify with: ssh -o BatchMode=yes <alias> true",
            "  4. Do not run the bare lightning CLI for discovery; use lightning-sdk wrappers after SSH is healthy.",
        ]
    )
    return "\n".join(lines)


def _ssh_policy_violations(diagnostic: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    resolved = diagnostic.get("resolved") if isinstance(diagnostic.get("resolved"), dict) else {}
    strict = str(resolved.get("stricthostkeychecking") or "").lower()
    if strict in {"false", "no", "off"}:
        violations.append(
            "StrictHostKeyChecking is disabled; use accept-new or yes for Lightning custody"
        )
    identities = diagnostic.get("identity_files") if isinstance(diagnostic.get("identity_files"), list) else []
    if identities and not any(bool(item.get("identity_file_exists")) for item in identities if isinstance(item, dict)):
        violations.append("no resolved IdentityFile exists on disk")
    if identities and not any(bool(item.get("public_key_exists")) for item in identities if isinstance(item, dict)):
        violations.append("no resolved IdentityFile has a sibling .pub key for Lightning registration")
    return violations


def ensure_lightning_ssh_ready(
    remote: str,
    *,
    ssh_bin: str = "ssh",
    connect_timeout: int = 15,
    runner: RunFn = subprocess.run,
) -> dict[str, Any]:
    diagnostic = lightning_ssh_diagnostic(
        remote,
        ssh_bin=ssh_bin,
        connect=True,
        connect_timeout=connect_timeout,
        runner=runner,
    )
    if diagnostic.get("status") != "ok":
        raise LightningSshPreflightError(_format_lightning_ssh_failure(diagnostic), diagnostic)
    policy_violations = _ssh_policy_violations(diagnostic)
    if policy_violations:
        diagnostic["status"] = "fail"
        diagnostic["policy_violations"] = policy_violations
        raise LightningSshPreflightError(_format_lightning_ssh_failure(diagnostic), diagnostic)
    return diagnostic


RUNTIME_DIAGNOSTIC_MARKER = "LIGHTNING_RUNTIME_DIAGNOSTIC_JSON="


def _remote_runtime_probe_command(
    remote_pact: str,
    *,
    python_bin: str | None,
    require_cuda: bool,
) -> str:
    lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(remote_pact)}",
    ]
    if python_bin:
        lines.extend(
            [
                f"PYBIN={shlex.quote(python_bin)}",
                'test -x "$PYBIN"',
            ]
        )
    else:
        lines.extend(
            [
                "if [ -x .venv/bin/python ]; then",
                "  PYBIN=.venv/bin/python",
                "else",
                "  PYBIN=$(command -v python3 || command -v python)",
                "fi",
            ]
        )
    lines.extend(
        [
            '"$PYBIN" - <<\'PY\'',
            "import json, pathlib, shutil, subprocess, sys, time",
            "payload = {",
            "  'schema_version': 1,",
            "  'tool': 'lightning_remote_runtime_preflight',",
            "  'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),",
            "  'cwd': str(pathlib.Path.cwd()),",
            "  'python': sys.executable,",
            "  'python_version': sys.version,",
            f"  'python_bin_requested': {python_bin!r},",
            f"  'require_cuda': {require_cuda!r},",
            "}",
            "try:",
            "    import torch",
            "    payload['torch_version'] = getattr(torch, '__version__', None)",
            "    payload['torch_cuda_version'] = getattr(torch.version, 'cuda', None)",
            "    payload['torch_cuda_available'] = bool(torch.cuda.is_available())",
            "    payload['torch_cuda_device_count'] = int(torch.cuda.device_count())",
            "    payload['torch_cuda_device_names'] = [",
            "        torch.cuda.get_device_name(i) for i in range(int(torch.cuda.device_count()))",
            "    ] if torch.cuda.is_available() else []",
            "except Exception as exc:",
            "    payload['torch_import_error'] = repr(exc)",
            "nvidia_smi = shutil.which('nvidia-smi')",
            "payload['nvidia_smi'] = nvidia_smi",
            "if nvidia_smi:",
            "    probe = subprocess.run(",
            "        [nvidia_smi, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],",
            "        check=False,",
            "        capture_output=True,",
            "        text=True,",
            "        timeout=20,",
            "    )",
            "    payload['nvidia_smi_returncode'] = probe.returncode",
            "    payload['nvidia_smi_output'] = (probe.stdout or probe.stderr).strip().splitlines()[:8]",
            f"print({RUNTIME_DIAGNOSTIC_MARKER!r} + json.dumps(payload, sort_keys=True))",
            f"if {require_cuda!r} and payload.get('torch_cuda_available') is not True:",
            "    raise SystemExit('FATAL: --require-cuda requested but torch.cuda.is_available() is not true')",
            "PY",
        ]
    )
    return "\n".join(lines)


def lightning_remote_runtime_diagnostic(
    remote: str,
    *,
    remote_pact: str,
    python_bin: str | None,
    require_cuda: bool,
    ssh_bin: str = "ssh",
    connect_timeout: int = 15,
    runner: RunFn = subprocess.run,
) -> dict[str, Any]:
    """Probe the reachable Studio runtime, optionally requiring CUDA."""

    command = _remote_runtime_probe_command(
        remote_pact,
        python_bin=python_bin,
        require_cuda=require_cuda,
    )
    proc = _ssh_run(
        _ssh_command(ssh_bin, remote, command, connect_timeout=connect_timeout),
        runner=runner,
    )
    diagnostic: dict[str, Any] = {
        "schema_version": 1,
        "target": remote,
        "remote_pact": remote_pact,
        "python_bin_requested": python_bin,
        "require_cuda": bool(require_cuda),
        "ssh_bin": ssh_bin,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    for raw_line in proc.stdout.splitlines():
        if raw_line.startswith(RUNTIME_DIAGNOSTIC_MARKER):
            payload = raw_line[len(RUNTIME_DIAGNOSTIC_MARKER) :]
            try:
                diagnostic["remote_runtime"] = json.loads(payload)
            except json.JSONDecodeError:
                diagnostic["remote_runtime_parse_error"] = payload
    remote_runtime = diagnostic.get("remote_runtime")
    cuda_ok = (
        isinstance(remote_runtime, dict)
        and remote_runtime.get("torch_cuda_available") is True
    )
    diagnostic["status"] = "ok" if proc.returncode == 0 else "fail"
    if require_cuda and not cuda_ok:
        diagnostic["status"] = "fail"
        diagnostic["runtime_policy_violations"] = [
            "torch.cuda.is_available() is not true on the reachable Lightning Studio runtime"
        ]
    return diagnostic


def _format_lightning_runtime_failure(diagnostic: dict[str, Any]) -> str:
    remote_runtime = (
        diagnostic.get("remote_runtime")
        if isinstance(diagnostic.get("remote_runtime"), dict)
        else {}
    )
    lines = [
        f"Lightning remote runtime preflight failed for target {diagnostic.get('target')!r}.",
        (
            "The Studio SSH host is reachable, but this runtime is not valid for "
            "interactive CUDA score work."
        ),
    ]
    if remote_runtime:
        lines.append(
            "Remote runtime: "
            f"python={remote_runtime.get('python')!r} "
            f"torch={remote_runtime.get('torch_version')!r} "
            f"torch_cuda={remote_runtime.get('torch_cuda_version')!r} "
            f"cuda_available={remote_runtime.get('torch_cuda_available')!r} "
            f"device_count={remote_runtime.get('torch_cuda_device_count')!r} "
            f"device_names={remote_runtime.get('torch_cuda_device_names')!r} "
            f"nvidia_smi={remote_runtime.get('nvidia_smi')!r}."
        )
    policy_violations = diagnostic.get("runtime_policy_violations")
    if policy_violations:
        lines.append(f"Runtime policy violations: {policy_violations!r}.")
    stderr = str(diagnostic.get("stderr_tail") or "").strip()
    if stderr:
        lines.append(f"SSH/runtime stderr tail: {stderr}")
    lines.extend(
        [
            "Fix guidance:",
            "  1. Switch the Lightning Studio back to a GPU machine before interactive CUDA work.",
            "  2. For Batch Jobs, rely on the per-job CUDA preflight artifact instead of the Studio shell.",
            "  3. Never promote CPU/MPS results per CLAUDE.md non-negotiable; exact score evidence must come from CUDA auth eval artifacts.",
        ]
    )
    return "\n".join(lines)


def ensure_lightning_remote_runtime_ready(
    remote: str,
    *,
    remote_pact: str,
    python_bin: str | None,
    require_cuda: bool,
    ssh_bin: str = "ssh",
    connect_timeout: int = 15,
    runner: RunFn = subprocess.run,
) -> dict[str, Any]:
    diagnostic = lightning_remote_runtime_diagnostic(
        remote,
        remote_pact=remote_pact,
        python_bin=python_bin,
        require_cuda=require_cuda,
        ssh_bin=ssh_bin,
        connect_timeout=connect_timeout,
        runner=runner,
    )
    if diagnostic.get("status") != "ok":
        raise LightningRuntimePreflightError(
            _format_lightning_runtime_failure(diagnostic),
            diagnostic,
        )
    return diagnostic


def _repo_rel(path: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    raw = Path(path)
    resolved = (repo_root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        return resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path must stay inside repo: {path}") from exc


def _is_excluded(rel: Path) -> bool:
    parts = set(rel.parts)
    if parts & EXCLUDED_DIR_NAMES:
        return True
    for part in rel.parts:
        if part == "__MACOSX" or part.startswith("._") or part in EXCLUDED_HIDDEN_SOURCE_NAMES:
            return True
        if part.startswith("."):
            return True
    rel_posix = rel.as_posix()
    if rel_posix.endswith(EXCLUDED_SOURCE_SUFFIXES):
        return True
    for prefix in EXCLUDED_PREFIXES:
        exact = prefix.rstrip("/")
        if rel_posix == exact:
            return True
        if prefix.endswith("/") and rel_posix.startswith(prefix):
            return True
    return False


def _git_status_path(status_line: str) -> Path | None:
    body = status_line[3:] if len(status_line) > 3 else ""
    if " -> " in body:
        body = body.rsplit(" -> ", 1)[-1]
    body = body.strip()
    return Path(body) if body else None


def _filter_git_status(status_lines: list[str]) -> tuple[list[str], list[str]]:
    public_lines: list[str] = []
    excluded_lines: list[str] = []
    for line in status_lines:
        rel = _git_status_path(line)
        if rel is not None and _is_excluded(rel):
            excluded_lines.append(line)
        else:
            public_lines.append(line)
    return public_lines, excluded_lines


def _iter_files(root: Path, *, repo_root: Path, include_excluded: bool) -> Iterable[Path]:
    if root.is_file():
        rel = root.relative_to(repo_root)
        if include_excluded or not _is_excluded(rel):
            yield root
        return
    if not root.is_dir():
        raise FileNotFoundError(root)
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        kept_dirs = []
        for dirname in sorted(dirnames):
            rel_dir = (current / dirname).relative_to(repo_root)
            if include_excluded or not _is_excluded(rel_dir):
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        for filename in sorted(filenames):
            path = current / filename
            rel = path.relative_to(repo_root)
            if include_excluded or not _is_excluded(rel):
                yield path


def collect_files(
    source_paths: Iterable[str | Path],
    artifact_paths: Iterable[str | Path],
    *,
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, object]]:
    """Collect source and explicit artifact files with deterministic hashes."""

    role_by_rel: dict[str, str] = {}
    for role, paths, include_excluded in (
        ("source", source_paths, False),
        ("artifact", artifact_paths, True),
    ):
        for item in paths:
            rel = _repo_rel(item, repo_root=repo_root)
            root = repo_root / rel
            for file_path in _iter_files(root, repo_root=repo_root, include_excluded=include_excluded):
                file_rel = file_path.relative_to(repo_root).as_posix()
                if role_by_rel.get(file_rel) != "artifact":
                    role_by_rel[file_rel] = role

    files: list[dict[str, object]] = []
    for rel in sorted(role_by_rel):
        path = repo_root / rel
        files.append(
            {
                "path": rel,
                "role": role_by_rel[rel],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError("no files selected for Lightning workspace sync")
    return files


def build_manifest(args: argparse.Namespace, files: list[dict[str, object]]) -> dict[str, object]:
    git_status = _capture(["git", "status", "--short", "--untracked-files=all"])
    git_status_lines = git_status.splitlines() if git_status else []
    public_status, excluded_status = _filter_git_status(git_status_lines)
    payload = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "tool": "scripts/lightning_repro_workspace.py",
        "run_id": args.run_id,
        "repo_root": str(REPO_ROOT),
        "remote": args.remote,
        "remote_pact": args.remote_pact,
        "source_paths": list(args.source),
        "artifact_paths": list(args.artifact),
        "install_runtime": args.requirements_mode == "uv-sync",
        "requirements_mode": args.requirements_mode,
        "python_bin": args.python_bin,
        "require_cuda": bool(args.require_cuda),
        "runtime_security": {
            "banned_lightning_versions": sorted(BANNED_LIGHTNING_VERSIONS),
            "repo_indicators": list(MINI_SHAI_HULUD_REPO_INDICATORS),
            "package_indicator": "lightning/_runtime",
        },
        "uv_locked": True,
        "git": {
            "head": _capture(["git", "rev-parse", "HEAD"]),
            "branch": _capture(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "status_short": public_status,
            "status_short_excluded_private": excluded_status,
        },
        "exclusions": {
            "dir_names": sorted(EXCLUDED_DIR_NAMES),
            "prefixes": list(EXCLUDED_PREFIXES),
            "source_suffixes": list(EXCLUDED_SOURCE_SUFFIXES),
        },
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }
    manifest_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    write_json(path, payload)


def _remote_python_command(
    remote_pact: str,
    remote_manifest: str,
    remote_env: str,
    *,
    requirements_mode: str,
    python_bin: str | None,
    require_cuda: bool,
) -> str:
    if requirements_mode not in REQUIREMENTS_MODES:
        raise ValueError(f"unsupported requirements_mode: {requirements_mode}")

    requested_py = shlex.quote(python_bin) if python_bin else None
    if requirements_mode == "uv-sync":
        install_block = "\n".join(
            [
                "if [ ! -f scripts/ensure_remote_uv.sh ]; then",
                "  echo 'FATAL: scripts/ensure_remote_uv.sh missing; remote source bundle incomplete' >&2",
                "  exit 31",
                "fi",
                "UV_BIN=$(bash scripts/ensure_remote_uv.sh --symlink-system)",
                "export UV_BIN",
                'export PATH="$(dirname "$UV_BIN"):$PATH"',
                'test -x "$UV_BIN"',
                'UV_LINK_MODE=${UV_LINK_MODE:-copy} "$UV_BIN" sync --locked --extra runtime',
                f"PYBIN={requested_py}" if requested_py else "PYBIN=.venv/bin/python",
                'test -x "$PYBIN"',
            ]
        )
    elif requirements_mode == "verify-only":
        install_block = "\n".join(
            [
                f"PYBIN={requested_py}"
                if requested_py
                else 'PYBIN="${PYBIN:?set PYBIN or pass --python-bin with --requirements-mode verify-only}"',
                'test -x "$PYBIN"',
            ]
        )
    else:
        if requested_py:
            install_block = "\n".join([f"PYBIN={requested_py}", 'test -x "$PYBIN"'])
        else:
            install_block = "\n".join(
                [
                    "PYBIN=${PYBIN:-}",
                    "if [ -z \"$PYBIN\" ]; then",
                    "  if [ -x .venv/bin/python ]; then",
                    "    PYBIN=.venv/bin/python",
                    "  else",
                    "    PYBIN=$(command -v python3 || command -v python)",
                    "  fi",
                    "fi",
                ]
            )
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_pact)}",
            "UV_BIN=$(command -v uv || true)",
            install_block,
            "$PYBIN - <<'PY'",
            "import importlib.metadata as md, json, os, pathlib, platform, subprocess, sys, time",
            f"BANNED_LIGHTNING_VERSIONS = {json.dumps(sorted(BANNED_LIGHTNING_VERSIONS))}",
            f"REPO_INDICATORS = {json.dumps(list(MINI_SHAI_HULUD_REPO_INDICATORS))}",
            "payload = {",
            "  'schema_version': 1,",
            "  'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),",
            "  'python': sys.executable,",
            "  'python_version': sys.version,",
            "  'platform': platform.platform(),",
            f"  'requirements_mode': {json.dumps(requirements_mode)},",
            f"  'python_bin_requested': {python_bin!r},",
            f"  'require_cuda': {require_cuda!r},",
            "  'packages': {},",
            "}",
            "for name in ['tac', 'torch', 'torchvision', 'av', 'safetensors', 'numpy', 'segmentation-models-pytorch', 'timm', 'lightning', 'lightning-sdk']:",
            "    try:",
            "        payload['packages'][name] = md.version(name)",
            "    except md.PackageNotFoundError:",
            "        payload['packages'][name] = None",
            "security_findings = []",
            "repo_root = pathlib.Path.cwd()",
            "for rel in REPO_INDICATORS:",
            "    candidate = repo_root / rel",
            "    if candidate.exists():",
            "        security_findings.append({'type': 'repo_indicator', 'path': str(candidate)})",
            "lightning_version = payload['packages'].get('lightning')",
            "if lightning_version in BANNED_LIGHTNING_VERSIONS:",
            "    security_findings.append({'type': 'banned_lightning_version', 'version': lightning_version})",
            "try:",
            "    dist = md.distribution('lightning')",
            "except md.PackageNotFoundError:",
            "    dist = None",
            "if dist is not None:",
            "    runtime_path = pathlib.Path(dist.locate_file('lightning/_runtime'))",
            "    if runtime_path.exists():",
            "        security_findings.append({'type': 'package_indicator', 'path': str(runtime_path)})",
            "for base in sys.path:",
            "    if not base:",
            "        continue",
            "    runtime_path = pathlib.Path(base) / 'lightning' / '_runtime'",
            "    if runtime_path.exists():",
            "        finding = {'type': 'sys_path_package_indicator', 'path': str(runtime_path)}",
            "        if finding not in security_findings:",
            "            security_findings.append(finding)",
            "payload['runtime_security'] = {",
            "    'status': 'fail' if security_findings else 'ok',",
            "    'findings': security_findings,",
            "    'banned_lightning_versions': BANNED_LIGHTNING_VERSIONS,",
            "}",
            "try:",
            "    import torch",
            "    payload['torch_cuda_available'] = bool(torch.cuda.is_available())",
            "    payload['torch_cuda_device_count'] = int(torch.cuda.device_count())",
            "    payload['torch_cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None",
            "    payload['torch_cuda_version'] = getattr(torch.version, 'cuda', None)",
            "except Exception as exc:",
            "    payload['torch_import_error'] = repr(exc)",
            "try:",
            "    payload['uv_version'] = subprocess.check_output(['uv', '--version'], text=True).strip()",
            "except Exception as exc:",
            "    payload['uv_version_error'] = repr(exc)",
            f"open({remote_env!r}, 'w').write(json.dumps(payload, indent=2, sort_keys=True) + '\\n')",
            "print(json.dumps(payload, sort_keys=True))",
            "if security_findings:",
            "    raise SystemExit('FATAL_RUNTIME_SECURITY_JSON=' + json.dumps(payload['runtime_security'], sort_keys=True))",
            f"if {require_cuda!r} and payload.get('torch_cuda_available') is not True:",
            "    raise SystemExit('FATAL: --require-cuda requested but torch.cuda.is_available() is not true')",
            "PY",
            f"test -f {shlex.quote(remote_manifest)}",
            f"test -f {shlex.quote(remote_env)}",
        ]
    )


def _remote_verify_command(remote_pact: str, remote_manifest: str) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_pact)}",
            "PYVERIFY=$(command -v python3 || command -v python || echo /system/conda/miniconda3/bin/python)",
            f"$PYVERIFY - {shlex.quote(remote_manifest)} <<'PY'",
            "import hashlib, json, pathlib, sys",
            "manifest = json.load(open(sys.argv[1]))",
            "bad = []",
            "for item in manifest['files']:",
            "    path = pathlib.Path(item['path'])",
            "    if not path.is_file():",
            "        bad.append({'path': item['path'], 'error': 'missing'})",
            "        continue",
            "    h = hashlib.sha256()",
            "    with path.open('rb') as f:",
            "        for chunk in iter(lambda: f.read(1024 * 1024), b''):",
            "            h.update(chunk)",
            "    actual = {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}",
            "    if actual['bytes'] != item['bytes'] or actual['sha256'] != item['sha256']:",
            "        bad.append({'path': item['path'], 'expected': item, 'actual': actual})",
            "if bad:",
            "    raise SystemExit(json.dumps({'REMOTE_MANIFEST_VERIFY': 'FAIL', 'bad': bad[:20]}, indent=2))",
            "print(json.dumps({'REMOTE_MANIFEST_VERIFY': 'OK', 'file_count': len(manifest['files']), 'total_bytes': manifest['total_bytes']}, sort_keys=True))",
            "PY",
        ]
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=_default_remote())
    parser.add_argument("--remote-pact", default=DEFAULT_REMOTE_PACT)
    parser.add_argument("--run-id", default=f"lightning_repro_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument(
        "--requirements-mode",
        choices=REQUIREMENTS_MODES,
        default=None,
        help=(
            "Remote runtime mode: uv-sync runs `uv sync --locked --extra runtime`; "
            "verify-only requires an existing Python; no-install keeps the legacy fallback."
        ),
    )
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Remote Python executable to record/use for runtime verification.",
    )
    parser.add_argument("--require-cuda", action="store_true", help="Fail remote runtime verification unless torch CUDA is available.")
    parser.add_argument("--no-install", action="store_true", help="Deprecated alias for --requirements-mode no-install.")
    parser.add_argument("--no-verify", action="store_true", help="Skip remote SHA-256 verification.")
    parser.add_argument(
        "--ssh-check-only",
        action="store_true",
        help="Run the Lightning SSH config/auth preflight and exit without staging.",
    )
    parser.add_argument(
        "--ssh-diagnostics-out",
        default=None,
        help="Optional local JSON path for sanitized SSH config/auth diagnostics.",
    )
    parser.add_argument(
        "--ssh-connect-timeout",
        type=int,
        default=15,
        help="BatchMode SSH auth preflight timeout in seconds.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.no_install:
        if args.requirements_mode not in (None, "no-install"):
            parser.error("--no-install conflicts with --requirements-mode values other than no-install")
        args.requirements_mode = "no-install"
    elif args.requirements_mode is None:
        args.requirements_mode = "uv-sync"
    if args.source is None:
        args.source = list(DEFAULT_SOURCE_PATHS)
    return args


def _default_remote() -> str:
    return default_ssh_target()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.remote:
        raise SystemExit("set --remote or LIGHTNING_SSH_TARGET before staging to Lightning")
    if args.remote == "ssh.lightning.ai":
        raise SystemExit("set --remote or LIGHTNING_USER to include the Studio SSH user")
    if args.requirements_mode == "uv-sync" and not args.dry_run and not args.ssh_check_only:
        try:
            ensure_local_uv_lock_current()
        except LightningLocalUvLockError as exc:
            raise SystemExit(str(exc)) from exc

    if args.ssh_check_only:
        diagnostic: dict[str, Any]
        try:
            diagnostic = ensure_lightning_ssh_ready(
                args.remote,
                connect_timeout=args.ssh_connect_timeout,
            )
        except LightningSshPreflightError as exc:
            if args.ssh_diagnostics_out:
                _write_json(Path(args.ssh_diagnostics_out), exc.diagnostic)
            raise SystemExit(str(exc)) from exc
        if args.require_cuda:
            try:
                runtime_diagnostic = ensure_lightning_remote_runtime_ready(
                    args.remote,
                    remote_pact=args.remote_pact,
                    python_bin=args.python_bin,
                    require_cuda=True,
                    connect_timeout=args.ssh_connect_timeout,
                )
            except LightningRuntimePreflightError as exc:
                diagnostic["runtime_probe"] = exc.diagnostic
                if args.ssh_diagnostics_out:
                    _write_json(Path(args.ssh_diagnostics_out), diagnostic)
                raise SystemExit(str(exc)) from exc
            diagnostic["runtime_probe"] = runtime_diagnostic
        if args.ssh_diagnostics_out:
            _write_json(Path(args.ssh_diagnostics_out), diagnostic)
        print(json_text(diagnostic), end="")
        return 0

    files = collect_files(args.source, args.artifact)
    manifest = build_manifest(args, files)
    manifest_out = Path(args.manifest_out) if args.manifest_out else DEFAULT_STATE_DIR / f"{args.run_id}_manifest.json"
    _write_json(manifest_out, manifest)

    ssh_diagnostic: dict[str, Any] | None = None
    try:
        if args.dry_run:
            ssh_diagnostic = lightning_ssh_diagnostic(
                args.remote,
                connect=False,
                connect_timeout=args.ssh_connect_timeout,
            )
        else:
            ssh_diagnostic = ensure_lightning_ssh_ready(
                args.remote,
                connect_timeout=args.ssh_connect_timeout,
            )
    except LightningSshPreflightError as exc:
        ssh_diagnostic = exc.diagnostic
        if args.ssh_diagnostics_out:
            _write_json(Path(args.ssh_diagnostics_out), ssh_diagnostic)
        raise SystemExit(str(exc)) from exc
    if args.ssh_diagnostics_out and ssh_diagnostic is not None:
        _write_json(Path(args.ssh_diagnostics_out), ssh_diagnostic)

    with tempfile.TemporaryDirectory(prefix="lightning-repro-") as tmp:
        tmpdir = Path(tmp)
        file_list = tmpdir / "files.txt"
        file_list.write_text("\n".join(str(item["path"]) for item in files) + "\n")

        remote_state = f"{args.remote_pact}/.omx/state"
        remote_manifest = f"{remote_state}/{manifest_out.name}"
        remote_env = f"{remote_state}/{args.run_id}_environment.json"

        _run(
            _ssh_command(
                "ssh",
                args.remote,
                "mkdir -p "
                + shlex.quote(args.remote_pact)
                + " "
                + shlex.quote(remote_state),
                connect_timeout=args.ssh_connect_timeout,
            ),
            dry_run=args.dry_run,
        )
        _run(
            _rsync_command(
                file_list,
                args.remote,
                args.remote_pact,
                connect_timeout=args.ssh_connect_timeout,
            ),
            dry_run=args.dry_run,
        )
        _run(
            _scp_command(
                "scp",
                manifest_out,
                f"{args.remote}:{remote_manifest}",
                connect_timeout=args.ssh_connect_timeout,
            ),
            dry_run=args.dry_run,
        )

        if not args.no_verify:
            _run(
                _ssh_command(
                    "ssh",
                    args.remote,
                    _remote_verify_command(args.remote_pact, remote_manifest),
                    connect_timeout=args.ssh_connect_timeout,
                ),
                dry_run=args.dry_run,
            )
        _run(
            _ssh_command(
                "ssh",
                args.remote,
                _remote_python_command(
                    args.remote_pact,
                    remote_manifest,
                    remote_env,
                    requirements_mode=args.requirements_mode,
                    python_bin=args.python_bin,
                    require_cuda=args.require_cuda,
                ),
                connect_timeout=args.ssh_connect_timeout,
            ),
            dry_run=args.dry_run,
        )

    print(
        json_text(
            {
                "status": "DRY_RUN" if args.dry_run else "OK",
                "manifest": str(manifest_out),
                "remote_manifest": f"{args.remote}:{remote_manifest}",
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "manifest_sha256": manifest["manifest_sha256"],
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
