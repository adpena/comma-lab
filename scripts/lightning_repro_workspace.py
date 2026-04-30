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
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_DIR = REPO_ROOT / ".omx/state"
DEFAULT_REMOTE_PACT = "/teamspace/studios/this_studio/pact"
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
REQUIREMENTS_MODES = ("uv-sync", "verify-only", "no-install")
BANNED_LIGHTNING_VERSIONS = {"2.6.2", "2.6.3"}
MINI_SHAI_HULUD_REPO_INDICATORS = (
    ".claude/router_runtime.js",
    ".claude/setup.mjs",
    ".vscode/setup.mjs",
    ".github/workflows/format-check.yml",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise ValueError("no files selected for Lightning workspace sync")
    return files


def build_manifest(args: argparse.Namespace, files: list[dict[str, object]]) -> dict[str, object]:
    git_status = _capture(["git", "status", "--short", "--untracked-files=all"])
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
            "status_short": git_status.splitlines() if git_status else [],
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
                "uv sync --locked --extra runtime",
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
            'if [ -z "$UV_BIN" ] && [ ' + shlex.quote(requirements_mode) + ' = uv-sync ]; then',
            "  echo 'FATAL: uv is required for --requirements-mode uv-sync' >&2",
            "  exit 31",
            "fi",
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
            f"  'require_cuda': {repr(require_cuda)},",
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
            f"if {repr(require_cuda)} and payload.get('torch_cuda_available') is not True:",
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
    parser.add_argument("--remote", default=os.environ.get("REMOTE") or _default_remote())
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
    user = os.environ.get("LIGHTNING_USER")
    if user:
        return f"{user}@ssh.lightning.ai"
    return "ssh.lightning.ai"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.remote == "ssh.lightning.ai":
        raise SystemExit("set --remote or LIGHTNING_USER to include the Studio SSH user")

    files = collect_files(args.source, args.artifact)
    manifest = build_manifest(args, files)
    manifest_out = Path(args.manifest_out) if args.manifest_out else DEFAULT_STATE_DIR / f"{args.run_id}_manifest.json"
    _write_json(manifest_out, manifest)

    with tempfile.TemporaryDirectory(prefix="lightning-repro-") as tmp:
        tmpdir = Path(tmp)
        file_list = tmpdir / "files.txt"
        file_list.write_text("\n".join(str(item["path"]) for item in files) + "\n")

        remote_state = f"{args.remote_pact}/.omx/state"
        remote_manifest = f"{remote_state}/{manifest_out.name}"
        remote_env = f"{remote_state}/{args.run_id}_environment.json"

        _run(["ssh", args.remote, "mkdir", "-p", args.remote_pact, remote_state], dry_run=args.dry_run)
        _run(
            ["rsync", "-a", "--files-from", str(file_list), "./", f"{args.remote}:{args.remote_pact}/"],
            dry_run=args.dry_run,
        )
        _run(["scp", str(manifest_out), f"{args.remote}:{remote_manifest}"], dry_run=args.dry_run)

        if not args.no_verify:
            _run(["ssh", args.remote, _remote_verify_command(args.remote_pact, remote_manifest)], dry_run=args.dry_run)
        _run(
            [
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
            ],
            dry_run=args.dry_run,
        )

    print(
        json.dumps(
            {
                "status": "DRY_RUN" if args.dry_run else "OK",
                "manifest": str(manifest_out),
                "remote_manifest": f"{args.remote}:{remote_manifest}",
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
