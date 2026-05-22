#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generic contest-compliant auth evaluation for ANY submission archive.

This is the CANONICAL tool for verifying any submission against the contest
scorer. Unlike auth_eval_renderer.py (which loads a renderer checkpoint and
renders frames in-process — a development shortcut), this tool runs the
EXACT contest pipeline:

    archive.zip → submission's inflate.sh → upstream/evaluate.py → score

Works for ANY contest-compliant submission, not just renderer-shaped ones.
The inflate.sh path defaults to submissions/robust_current/inflate.sh but
can be overridden for non-renderer lanes.

This tool is what the contest scorer effectively does internally. If a
score from this tool differs from auth_eval_renderer.py, the difference
reveals an inflate-path bug or an in-process-vs-on-disk numerical drift.

Council R3 (2026-04-26) flagged that auth_eval_renderer.py is renderer-
specific; the user's binding rule is that auth eval should work with any
contest-compliant submission. This tool is the answer.

Usage:
    .venv/bin/python experiments/contest_auth_eval.py \\
        --archive submissions/baseline_dilated_h64_0_90/archive_baseline_0_9001.zip \\
        --upstream-dir upstream \\
        --device cuda

    # Override inflate.sh for a non-renderer submission:
    .venv/bin/python experiments/contest_auth_eval.py \\
        --archive my_submission.zip \\
        --inflate-sh submissions/exact_current/inflate.sh \\
        --upstream-dir upstream

    # Specify GT video names file (default: upstream/public_test_video_names.txt):
    .venv/bin/python experiments/contest_auth_eval.py \\
        --archive baseline.zip \\
        --upstream-dir upstream \\
        --video-names-file upstream/public_test_video_names.txt
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

from tac.device_axis_eval import is_contest_cuda_equivalent_gpu

# Line-buffer stdout so progress flushes to log files immediately.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

# Schema version for the JSON we emit. Bump when adding fields so downstream
# tooling (BATTLE_PLAN parsers, leaderboard, etc.) can detect compatibility.
SCHEMA_VERSION = 1
_RUNTIME_DEPENDENCY_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".env",
    ".h",
    ".hpp",
    ".json",
    ".py",
    ".sh",
    ".toml",
    ".txt",
}
_RUNTIME_DEPENDENCY_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
_RUNTIME_DEPENDENCY_ROOT_DIRECTIVE_RE = re.compile(
    r"^\s*#\s*PACT_RUNTIME_DEPENDENCY_ROOT\s*=\s*(?P<path>.+?)\s*$",
    re.MULTILINE,
)
_INFLATE_ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ALLOWED_INFLATE_ENV_PREFIXES = ("PACT_", "INFLATE_")
_ALLOWED_INFLATE_ENV_KEYS = {"CUDA_VISIBLE_DEVICES"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path, *, prefix: int = 16) -> str:
    """Hash a file's contents (full SHA256, return prefix chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    digest = h.hexdigest()
    return digest[:prefix] if prefix else digest


def _is_tac_module(module_name: str) -> bool:
    return module_name == "tac" or module_name.startswith("tac.")


def _runtime_python_files(runtime_root: Path) -> list[Path]:
    if not runtime_root.exists():
        return []
    paths: list[Path] = []
    for path in runtime_root.rglob("*.py"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(runtime_root).parts
        if any(part in _RUNTIME_DEPENDENCY_SKIP_DIRS for part in rel_parts):
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: p.relative_to(runtime_root).as_posix())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _runtime_dependency_extra_roots(inflate_sh: Path, repo_root: Path) -> list[Path]:
    """Return explicit external runtime roots declared by ``inflate.sh``.

    Public replay adapters sometimes live as a tiny shell shim while importing
    a checked-out public PR runtime from another repo-local directory. The
    directive makes that dependency part of the exact-eval custody hash instead
    of letting the adapter appear to be a one-file runtime.
    """

    try:
        text = inflate_sh.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    roots: list[Path] = []
    for match in _RUNTIME_DEPENDENCY_ROOT_DIRECTIVE_RE.finditer(text):
        raw = match.group("path").strip().strip("\"'")
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        resolved = path.resolve()
        if resolved != inflate_sh.parent.resolve() and resolved not in roots:
            roots.append(resolved)
    return roots


def _runtime_root_file_manifest(root: Path, repo_root: Path) -> list[dict]:
    files: list[dict] = []
    if not root.exists():
        return files
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in _RUNTIME_DEPENDENCY_SKIP_DIRS for part in rel_parts):
            continue
        if path.name.startswith("._") or path.name in {".DS_Store", "Thumbs.db"}:
            continue
        if path.suffix.lower() not in _RUNTIME_DEPENDENCY_SUFFIXES:
            continue
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "repo_relative_path": _repo_rel(path, repo_root),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path, prefix=0),
            }
        )
    return files


def _module_exists(module_name: str, repo_root: Path) -> bool:
    if not _is_tac_module(module_name):
        return False
    rel_parts = module_name.split(".")[1:]
    tac_root = repo_root / "src" / "tac"
    if not rel_parts:
        return (tac_root / "__init__.py").exists()
    return (
        (tac_root.joinpath(*rel_parts).with_suffix(".py")).exists()
        or (tac_root.joinpath(*rel_parts) / "__init__.py").exists()
    )


def _relative_import_base(module_name: str, level: int) -> str:
    parts = module_name.split(".")[:-1]
    if level > 1:
        parts = parts[: -(level - 1)]
    return ".".join(parts)


def _extract_tac_imports_from_source(
    source_path: Path,
    *,
    module_name: str | None,
    repo_root: Path,
) -> tuple[set[str], str | None]:
    try:
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
    except SyntaxError as exc:
        return set(), f"{exc.__class__.__name__}: {exc}"

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_tac_module(alias.name):
                    modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if module_name is None:
                    continue
                base = _relative_import_base(module_name, node.level)
                imported = f"{base}.{node.module}" if node.module else base
            else:
                imported = node.module or ""
            if _is_tac_module(imported):
                modules.add(imported)
                for alias in node.names:
                    candidate = f"{imported}.{alias.name}"
                    if _module_exists(candidate, repo_root):
                        modules.add(candidate)
        elif isinstance(node, ast.Call):
            func = node.func
            is_importlib_call = (
                isinstance(func, ast.Attribute)
                and func.attr == "import_module"
                and isinstance(func.value, ast.Name)
                and func.value.id == "importlib"
            )
            if is_importlib_call and node.args and isinstance(node.args[0], ast.Constant):
                value = node.args[0].value
                if isinstance(value, str) and _is_tac_module(value):
                    modules.add(value)

    return modules, None


def _module_paths(module_name: str, repo_root: Path) -> list[Path]:
    if not _is_tac_module(module_name):
        return []
    rel_parts = module_name.split(".")[1:]
    tac_root = repo_root / "src" / "tac"
    paths: list[Path] = []
    for i in range(len(rel_parts) + 1):
        init_path = tac_root.joinpath(*rel_parts[:i]) / "__init__.py"
        if init_path.exists():
            paths.append(init_path)
    if rel_parts:
        module_path = tac_root.joinpath(*rel_parts).with_suffix(".py")
        if module_path.exists():
            paths.append(module_path)
    return paths


def _module_name_for_tac_path(path: Path, repo_root: Path) -> str:
    rel = path.relative_to(repo_root / "src").with_suffix("")
    parts = rel.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _repo_local_tac_import_manifest(runtime_root: Path, repo_root: Path) -> dict:
    """Hash repo-local ``src/tac`` source reachable from robust runtime imports.

    This is intentionally static: it parses Python import surfaces without
    importing torch/av/scorer code or executing runtime branches. The closure is
    an allowlist-equivalent custody surface for repo-local tac helpers.
    """

    root_imports: set[str] = set()
    parse_errors: list[dict[str, str]] = []
    for path in _runtime_python_files(runtime_root):
        imports, error = _extract_tac_imports_from_source(
            path,
            module_name=None,
            repo_root=repo_root,
        )
        root_imports.update(imports)
        if error:
            parse_errors.append(
                {
                    "path": path.relative_to(runtime_root).as_posix(),
                    "error": error,
                }
            )

    queue = sorted(root_imports)
    seen_modules: set[str] = set()
    seen_files: dict[Path, str] = {}
    unresolved: set[str] = set()
    while queue:
        module = queue.pop(0)
        if module in seen_modules:
            continue
        seen_modules.add(module)
        paths = _module_paths(module, repo_root)
        if not paths:
            unresolved.add(module)
            continue
        for path in paths:
            path = path.resolve()
            file_module = _module_name_for_tac_path(path, repo_root)
            seen_files.setdefault(path, file_module)
            imports, error = _extract_tac_imports_from_source(
                path,
                module_name=file_module,
                repo_root=repo_root,
            )
            if error:
                parse_errors.append(
                    {
                        "path": path.relative_to(repo_root).as_posix(),
                        "error": error,
                    }
                )
            for imported in sorted(imports):
                if imported not in seen_modules:
                    queue.append(imported)

    files = []
    for path, module in sorted(
        seen_files.items(),
        key=lambda item: item[0].relative_to(repo_root).as_posix(),
    ):
        files.append(
            {
                "module": module,
                "relative_path": path.relative_to(repo_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path, prefix=0),
            }
        )

    return {
        "schema": "contest_auth_eval_repo_local_tac_import_manifest_v1",
        "discovery": "static_ast_recursive_import_closure",
        "runtime_root_name": runtime_root.name,
        "tac_root_relative_path": "src/tac",
        "root_import_modules": sorted(root_imports),
        "unresolved_modules": sorted(unresolved),
        "parse_errors": parse_errors,
        "module_count": len(seen_modules),
        "file_count": len(files),
        "files": files,
    }


def _runtime_dependency_manifest(
    inflate_sh: Path,
    upstream_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict:
    """Hash fixed runtime files that can affect exact archive behavior.

    The archive SHA is necessary but insufficient for custody whenever
    ``inflate.sh`` dispatches into repo-local Python. Two runs with identical
    archive bytes but different runtime helpers can produce different frames.
    Recording this tree hash makes those comparisons auditable.
    """

    root = inflate_sh.parent.resolve()
    repo_root = (repo_root or _repo_root()).resolve()
    files = _runtime_root_file_manifest(root, repo_root)
    extra_roots = _runtime_dependency_extra_roots(inflate_sh, repo_root)
    external_dependency_roots = []
    for extra_root in extra_roots:
        external_dependency_roots.append(
            {
                "root": str(extra_root),
                "repo_relative_root": _repo_rel(extra_root, repo_root),
                "exists": extra_root.exists(),
                "files": _runtime_root_file_manifest(extra_root, repo_root),
            }
        )

    repo_local_tac = _repo_local_tac_import_manifest(root, repo_root)
    evaluate_py = (upstream_dir / "evaluate.py").resolve()
    upstream_eval = None
    if evaluate_py.exists():
        upstream_eval = {
            "relative_path": "evaluate.py",
            "bytes": evaluate_py.stat().st_size,
            "sha256": _sha256(evaluate_py, prefix=0),
        }

    tree_payload = {
        "runtime_root_name": root.name,
        "files": files,
        "external_dependency_roots": external_dependency_roots,
        "repo_local_tac_import_manifest": repo_local_tac,
        "upstream_evaluate_py": upstream_eval,
    }
    tree_sha = hashlib.sha256(
        json.dumps(tree_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    content_payload = {
        "files": [
            {
                "relative_path": f["relative_path"],
                "bytes": f["bytes"],
                "sha256": f["sha256"],
            }
            for f in files
        ],
        "external_dependency_roots": [
            {
                "repo_relative_root": root.get("repo_relative_root"),
                "exists": root.get("exists"),
                "files": [
                    {
                        "relative_path": f["relative_path"],
                        "bytes": f["bytes"],
                        "sha256": f["sha256"],
                    }
                    for f in root.get("files", [])
                ],
            }
            for root in external_dependency_roots
        ],
        "repo_local_tac_import_manifest": {
            key: value
            for key, value in repo_local_tac.items()
            if key != "runtime_root_name"
        },
        "upstream_evaluate_py": upstream_eval,
    }
    content_tree_sha = hashlib.sha256(
        json.dumps(content_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "contest_auth_eval_runtime_dependency_manifest_v1",
        "runtime_root": str(root),
        "runtime_file_count": len(files),
        "runtime_tree_sha256": tree_sha,
        "runtime_content_tree_sha256": content_tree_sha,
        "files": files,
        "external_dependency_roots": external_dependency_roots,
        "repo_local_tac_import_manifest": repo_local_tac,
        "upstream_evaluate_py": upstream_eval,
    }


def _ensure_uv_available() -> None:
    """The robust_current inflate.sh shells out to `uv run python ...`.
    Verify uv is on PATH so we fail loud here, not 200 lines deep."""
    if shutil.which("uv") is not None:
        return

    candidate_dirs = [
        Path.home() / ".local" / "bin",
        Path("/root/.local/bin"),
    ]
    for candidate_dir in candidate_dirs:
        candidate = candidate_dir / "uv"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            os.environ["PATH"] = f"{candidate_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            if shutil.which("uv") is not None:
                return

    raise RuntimeError(
        "FATAL: `uv` is not on PATH. submissions/robust_current/inflate.sh "
        "uses `uv run python ...`. Install with `curl -LsSf "
        "https://astral.sh/uv/install.sh | sh` then re-run."
    )


def _inflate_sh_requires_config_env_guard(inflate_sh: Path) -> bool:
    """Return whether this inflate launcher declares the robust config contract.

    The F5 guard is mandatory for robust_current-style dispatchers that source
    config.env and route through PYTHON_INFLATE. Public contest submissions can
    be plain launchers that directly call their own inflate.py; requiring a
    sibling config.env for those would reject valid external traces.
    """

    try:
        text = inflate_sh.read_text(errors="replace")
    except OSError:
        return False
    return "PYTHON_INFLATE" in text or "CONFIG_ENV_PATH" in text


def _validate_config_env_for_renderer_dispatch(inflate_sh: Path) -> None:
    if not _inflate_sh_requires_config_env_guard(inflate_sh):
        return
    inflate_dir = inflate_sh.parent
    config_env = inflate_dir / "config.env"
    if not config_env.exists():
        raise SystemExit(
            f"FATAL: {config_env} missing -- inflate.sh would fall into the\n"
            f"       ffmpeg path and crash on extracted/0.mkv. Re-deploy with\n"
            f"       the fixed launcher (Codex F5 2026-04-28) which includes\n"
            f"       .env files via the .env suffix in _enumerate_python_and_shell."
        )
    config_text = config_env.read_text()
    if "PYTHON_INFLATE=renderer" not in config_text:
        raise SystemExit(
            f"FATAL: {config_env} exists but does not set PYTHON_INFLATE=renderer.\n"
            f"       inflate.sh would call its ffmpeg path which crashes on\n"
            f"       renderer archives (no extracted/0.mkv). Update config.env."
        )


def _record_provenance(work_dir: Path, archive: Path, inflate_sh: Path,
                       upstream_dir: Path, args: argparse.Namespace) -> dict:
    """Snapshot the env so a re-run on different hardware is detectable.
    Records gpu_model, driver, torch+cuda versions, ffmpeg+svtav1 versions,
    git commits, and SHA of every input file. Mandatory per CLAUDE.md
    'deterministic reproducibility' non-negotiable."""
    def _shell(cmd, *, timeout: int = 10) -> str | None:
        try:
            return subprocess.check_output(
                cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout,
            ).strip()
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            return f"<error:{exc!r}>"

    prov: dict = {
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "experiments/contest_auth_eval.py",
        "archive_path": str(archive),
        "archive_sha256": _sha256(archive, prefix=0),
        "archive_size_bytes": archive.stat().st_size,
        "inflate_script": str(inflate_sh),
        "inflate_script_sha256": _sha256(inflate_sh, prefix=0) if inflate_sh.exists() else None,
        "inflate_runtime_manifest": _runtime_dependency_manifest(inflate_sh, upstream_dir),
        "upstream_dir": str(upstream_dir),
        "device": args.device,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "inflate_timeout_seconds": int(args.inflate_timeout),
        "evaluate_timeout_seconds": int(args.evaluate_timeout),
        "video_names_file": str(args.video_names_file),
        "sys_argv": sys.argv,
        "effective_inflate_python": os.environ.get("PYTHON") or sys.executable,
        "modal_auth_eval_advisory_only": (
            os.environ.get("MODAL_AUTH_EVAL_ADVISORY_ONLY", "").strip() == "1"
        ),
        "env_vars": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
            "PYTHONHASHSEED", "PYTORCH_CUDA_ALLOC_CONF", "LD_LIBRARY_PATH",
            "CONFIG_ENV_PATH", "PYTHON_INFLATE", "LANE_MM_SIGMA",
            "INFLATE_BROTLI_SPEC", "INFLATE_AV_SPEC", "INFLATE_TORCH_SPEC",
            "INFLATE_TORCHVISION_SPEC", "INFLATE_NUMPY_SPEC",
            "MODAL_AUTH_EVAL_ADVISORY_ONLY", "UV_BIN",
            "UV_PROJECT_ENVIRONMENT", "PYTHON",
        )},
    }
    # GPU + driver — recorded in provenance for downstream comparison.
    # Contest scorer runs on Tesla T4; gpu_t4_match flag lets the operator
    # filter scores by hardware. No banner/warning printed (no editorializing
    # — the score IS what upstream/evaluate.py computed, period).
    prov["gpu_model"] = _shell(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    prov["gpu_driver"] = _shell(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    gm = (prov["gpu_model"] or "").strip()
    prov["gpu_t4_match"] = bool(gm) and "T4" in gm
    # torch + cuda
    try:
        import torch
        prov["torch_version"] = torch.__version__
        prov["cuda_version"] = torch.version.cuda
        prov["cuda_available"] = torch.cuda.is_available()
        prov["mps_available"] = bool(
            getattr(getattr(torch, "backends", None), "mps", None)
            and torch.backends.mps.is_available()
        )
        if torch.cuda.is_available():
            prov["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        prov["torch_import_error"] = True
    # ffmpeg + svtav1
    ffv = _shell(["ffmpeg", "-version"])
    prov["ffmpeg_version"] = (ffv.splitlines()[0] if ffv and not ffv.startswith("<error") else ffv)
    encs = _shell(["ffmpeg", "-encoders"])
    if encs and not encs.startswith("<error"):
        svt = [ln.strip() for ln in encs.splitlines()
               if "svtav1" in ln.lower() or "svt-av1" in ln.lower()]
        prov["libsvtav1_version"] = svt[0] if svt else None
    # git commits — pact + upstream
    source_commit = os.environ.get("PACT_SOURCE_COMMIT", "").strip()
    prov["pact_commit"] = source_commit or _shell(["git", "rev-parse", "HEAD"])
    prov["pact_commit_source"] = "PACT_SOURCE_COMMIT" if source_commit else "git_rev_parse"
    if (upstream_dir / ".git").exists() or (upstream_dir.parent / ".git").exists():
        prov["upstream_commit"] = _shell(
            ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"]
        )

    out = work_dir / "provenance.json"
    with open(out, "w") as f:
        json.dump(prov, f, indent=2)
    return prov


def _record_inflate_runtime_artifacts(prov: dict, work_dir: Path, extracted_dir: Path) -> None:
    """Attach inflate-produced custody summaries to provenance after inflate."""

    summaries: dict[str, dict] = {}
    renderer_summary = extracted_dir / "renderer_payload_unpack_summary.json"
    if renderer_summary.exists():
        try:
            payload = json.loads(renderer_summary.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"renderer payload unpack summary is not valid JSON: {renderer_summary}"
            ) from exc
        summaries["renderer_payload_unpack_summary"] = {
            "path": str(renderer_summary),
            "sha256": _sha256(renderer_summary, prefix=0),
            "payload": payload,
        }
    if not summaries:
        return
    prov["inflate_runtime_artifacts"] = summaries
    with open(work_dir / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)


def _record_inflated_output_artifacts(
    prov: dict,
    work_dir: Path,
    inflated_dir: Path,
    video_names_file: Path,
) -> dict:
    """Hash exact raw files produced by inflate.sh.

    Same archive bytes and same runtime source can still produce different
    scored frames when the runtime branches on CPU vs CUDA. The raw output
    hashes are the byte-level bridge between inflate and upstream/evaluate.py.
    """

    files: list[dict] = []
    for name in [n.strip() for n in video_names_file.read_text().splitlines() if n.strip()]:
        rel_raw = Path(name).with_suffix(".raw")
        raw_path = inflated_dir / rel_raw
        files.append(
            {
                "video_name": name,
                "relative_path": rel_raw.as_posix(),
                "exists": raw_path.exists(),
                "bytes": raw_path.stat().st_size if raw_path.exists() else None,
                "sha256": _sha256(raw_path, prefix=0) if raw_path.exists() else None,
            }
        )
    aggregate_payload = {
        "schema": "contest_auth_eval_inflated_output_manifest_v1",
        "inflated_dir": str(inflated_dir),
        "video_names_file": str(video_names_file),
        "raw_file_count": len(files),
        "total_bytes": sum(int(f["bytes"] or 0) for f in files),
        "files": files,
    }
    aggregate_payload["aggregate_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "files": [
                    {
                        "relative_path": f["relative_path"],
                        "bytes": f["bytes"],
                        "sha256": f["sha256"],
                    }
                    for f in files
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = work_dir / "inflated_outputs_manifest.json"
    manifest_path.write_text(json.dumps(aggregate_payload, indent=2) + "\n")
    prov["inflated_output_manifest"] = {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path, prefix=0),
        "payload": aggregate_payload,
    }
    with open(work_dir / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    return aggregate_payload


def _record_scorer_input_cache_hash_artifact(
    prov: dict,
    work_dir: Path,
    inflated_dir: Path,
    video_names_file: Path,
    inflated_manifest: dict,
    output_path: Path,
    *,
    batch_pairs: int,
) -> dict:
    """Write compact scorer-input array hashes for the inflated raw surface."""

    raw_paths: list[tuple[str, Path]] = []
    for name in [n.strip() for n in video_names_file.read_text().splitlines() if n.strip()]:
        raw_paths.append((name, inflated_dir / Path(name).with_suffix(".raw")))
    if len(raw_paths) != 1:
        raise RuntimeError(
            "scorer-input hash artifact currently expects exactly one raw file; "
            f"got {len(raw_paths)} from {video_names_file}"
        )
    video_name, raw_path = raw_paths[0]
    if not raw_path.is_file():
        raise RuntimeError(f"scorer-input hash raw file missing: {raw_path}")

    from tac.local_acceleration.mlx_preprocess import (
        write_scorer_input_cache_hash_manifest_from_raw_file,
    )

    target = _resolve_auth_artifact_output_under_work_dir(
        work_dir,
        output_path,
        label="scorer-input hash artifact",
    )
    manifest = write_scorer_input_cache_hash_manifest_from_raw_file(
        raw_path,
        target,
        archive_sha256=str(prov.get("archive_sha256") or ""),
        inflated_outputs_aggregate_sha256=str(
            inflated_manifest.get("aggregate_sha256") or ""
        ),
        batch_pairs=int(batch_pairs),
    )
    manifest["video_name"] = video_name
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    prov["scorer_input_cache_hash_manifest"] = {
        "path": str(target),
        "sha256": _sha256(target, prefix=0),
        "payload": manifest,
    }
    with open(work_dir / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    return manifest


def _record_scorer_input_cache_tensor_artifact(
    prov: dict,
    work_dir: Path,
    inflated_dir: Path,
    video_names_file: Path,
    inflated_manifest: dict,
    output_dir: Path,
    *,
    batch_pairs: int,
    allow_large_tensor_export: bool,
    large_pair_threshold: int,
) -> dict:
    """Write full scorer-input tensor cache for explicit local/volume export."""

    if batch_pairs < 1:
        raise RuntimeError("scorer-input tensor batch_pairs must be >= 1")
    if large_pair_threshold < 1:
        raise RuntimeError("scorer-input tensor large_pair_threshold must be >= 1")

    raw_paths: list[tuple[str, Path]] = []
    for name in [n.strip() for n in video_names_file.read_text().splitlines() if n.strip()]:
        raw_paths.append((name, inflated_dir / Path(name).with_suffix(".raw")))
    if len(raw_paths) != 1:
        raise RuntimeError(
            "scorer-input tensor artifact currently expects exactly one raw file; "
            f"got {len(raw_paths)} from {video_names_file}"
        )
    video_name, raw_path = raw_paths[0]
    if not raw_path.is_file():
        raise RuntimeError(f"scorer-input tensor raw file missing: {raw_path}")

    from tac.local_acceleration.mlx_preprocess import (
        load_raw_video_memmap,
        non_overlapping_pair_indices,
        write_scorer_input_cache_from_raw_file,
    )

    raw = load_raw_video_memmap(raw_path)
    pair_count = len(non_overlapping_pair_indices(raw.shape[0]))
    if pair_count > large_pair_threshold and not allow_large_tensor_export:
        raise RuntimeError(
            "refusing full scorer-input tensor cache export for "
            f"{pair_count} pairs (> threshold {large_pair_threshold}); pass "
            "--allow-large-scorer-input-cache-tensor-export or use "
            "--scorer-input-cache-hashes-out for compact identity only"
        )

    target = _resolve_auth_artifact_output_under_work_dir(
        work_dir,
        output_dir,
        label="scorer-input tensor cache directory",
    )
    manifest = write_scorer_input_cache_from_raw_file(
        raw_path,
        target,
        archive_sha256=str(prov.get("archive_sha256") or ""),
        inflated_outputs_aggregate_sha256=str(
            inflated_manifest.get("aggregate_sha256") or ""
        ),
        batch_pairs=batch_pairs,
    )
    manifest["video_name"] = video_name
    manifest["large_tensor_export_acknowledged"] = bool(allow_large_tensor_export)
    manifest["returned_via_modal_artifacts"] = False
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    prov["scorer_input_cache_tensor_manifest"] = {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path, prefix=0),
        "tensor_cache_dir": str(target),
        "tensor_payload_returned_via_modal_artifacts": False,
        "payload": manifest,
    }
    with open(work_dir / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    return manifest


def _resolve_auth_artifact_output_under_work_dir(
    work_dir: Path,
    output_path: Path,
    *,
    label: str,
) -> Path:
    """Resolve auth-side MLX artifact paths without escaping eval custody."""

    work_root = work_dir.resolve()
    target = output_path if output_path.is_absolute() else work_root / output_path
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(work_root)
    except ValueError as exc:
        raise RuntimeError(
            f"{label} must be inside contest_auth_eval work_dir: "
            f"work_dir={work_root} output={resolved}"
        ) from exc
    return resolved


def _validate_expected_runtime_tree(prov: dict, expected_runtime_tree_sha256: str | None) -> None:
    if not expected_runtime_tree_sha256:
        return
    manifest = prov.get("inflate_runtime_manifest")
    actual = manifest.get("runtime_tree_sha256") if isinstance(manifest, dict) else None
    if actual != expected_runtime_tree_sha256:
        raise RuntimeError(
            "inflate runtime tree hash mismatch: "
            f"expected={expected_runtime_tree_sha256} actual={actual}"
        )


def _extract_archive(archive: Path, dest: Path) -> list[str]:
    """Extract archive.zip into dest/. Returns list of member names.
    Refuses to write outside dest (zip-slip protection)."""
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    members: list[str] = []
    with zipfile.ZipFile(archive, "r") as z:
        _validate_zip_container_integrity(archive, z.infolist())
        for info in z.infolist():
            _validate_zip_member_name(info.filename)
            target = (dest / info.filename).resolve()
            try:
                target.relative_to(dest_resolved)
            except ValueError as exc:
                raise RuntimeError(f"Refusing zip-slip path: {info.filename}") from exc
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            members.append(info.filename)
    return members


def _decode_zip_name(raw: bytes, *, utf8: bool) -> str:
    encoding = "utf-8" if utf8 else "cp437"
    return raw.decode(encoding)


def _validate_zip_member_name(name: str) -> None:
    """Reject path names that can make ZIP readers disagree about custody."""
    if not name:
        raise RuntimeError("[archive-validate] EMPTY zip member filename")
    if "\\" in name:
        raise RuntimeError(f"[archive-validate] BACKSLASH in zip member name: {name!r}")
    if any(ord(ch) < 32 for ch in name):
        raise RuntimeError(f"[archive-validate] CONTROL character in zip member name: {name!r}")
    member = PurePosixPath(name)
    if member.is_absolute():
        raise RuntimeError(f"[archive-validate] ABSOLUTE zip member path: {name!r}")
    parts = member.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise RuntimeError(f"[archive-validate] NONCANONICAL zip member path: {name!r}")
    if ":" in parts[0]:
        raise RuntimeError(f"[archive-validate] DRIVE-like zip member path: {name!r}")


def _validate_zip_container_integrity(
    archive: Path,
    infos: list[zipfile.ZipInfo],
) -> None:
    """Fail closed on ZIP parser-divergence tricks before extraction.

    The official workflow currently shells out to `unzip`, while Python
    `zipfile` reads the central directory and then verifies local headers.
    A malformed archive can make those readers disagree about member names.
    Contest-custody archives must not rely on that ambiguity.
    """
    seen: set[str] = set()
    with archive.open("rb") as fh:
        for info in infos:
            _validate_zip_member_name(info.filename)
            if info.filename in seen:
                raise RuntimeError(
                    f"[archive-validate] DUPLICATE zip member name: {info.filename!r}"
                )
            seen.add(info.filename)
            fh.seek(info.header_offset)
            header = fh.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise RuntimeError(
                    "[archive-validate] MALFORMED zip local header for "
                    f"{info.filename!r}"
                )
            name_len = int.from_bytes(header[26:28], "little")
            extra_len = int.from_bytes(header[28:30], "little")
            local_name_raw = fh.read(name_len)
            if len(local_name_raw) != name_len:
                raise RuntimeError(
                    "[archive-validate] TRUNCATED zip local filename for "
                    f"{info.filename!r}"
                )
            # Advance over the extra field so a short/truncated local header is
            # caught before downstream extraction.
            if len(fh.read(extra_len)) != extra_len:
                raise RuntimeError(
                    "[archive-validate] TRUNCATED zip local extra field for "
                    f"{info.filename!r}"
                )
            if not local_name_raw:
                raise RuntimeError(
                    "[archive-validate] EMPTY zip local filename for central "
                    f"member {info.filename!r}"
                )
            try:
                local_name = _decode_zip_name(
                    local_name_raw,
                    utf8=bool(info.flag_bits & 0x800),
                )
            except UnicodeDecodeError as exc:
                raise RuntimeError(
                    "[archive-validate] UNDECODABLE zip local filename for "
                    f"{info.filename!r}: {exc}"
                ) from exc
            if local_name != info.filename:
                raise RuntimeError(
                    "[archive-validate] ZIP central/local filename mismatch: "
                    f"central={info.filename!r} local={local_name!r}"
                )
            _validate_zip_member_name(local_name)


# 2026-04-28 deep hardening pass 3 dimension 3: Whitelist-based archive
# validation. Catches the bug class where unexpected files in the archive
# (stale TTO frames, debug snapshots, .DS_Store from macOS) silently inflate
# the rate term. Memory: feedback_catastrophic_failures_20260421 (Auto-bundle
# by file existence — compress.sh auto-included any .pt/.bin file sitting
# next to the submission).
_KNOWN_ARCHIVE_SUFFIXES = (
    ".bin", ".bin.br",          # renderer (raw or brotli'd)
    ".mkv", ".mp4",             # mask video (svtav1 / h264 / etc.)
    ".tar.xz",                  # SegMap/Selfcomp-style packed weights
    ".fp16",                    # tiny charged fp16 payloads (e.g. LCT)
    ".nrv",                     # NeRV mask codec payload
    ".amrc",                    # lossless argmax-RLE mask codec payload
    ".cmg1",                    # charged mask grammar payload
    ".cmg2",                    # predictive charged mask grammar payload
    ".cmg3",                    # row-span charged mask grammar payload
    ".qma9",                    # PR85-style adaptive range-coded mask payload
    ".cdo1", ".cdo1.xz", ".cdo1.zlib", ".cdo1.br",  # decoded-mask overlay payload
    ".amr1",                    # Alpha sparse residual repair payload
    ".amr1.xz", ".amr1.zlib", ".amr1.br",
    ".pt",                       # poses, optionally other state dicts
    ".pt.gz",                    # PR86-style charged compressed model/state dict
    ".pt.ppmd",                  # PR86-style charged PPMd-compressed HPAC state
    ".json", ".txt",             # manifests / pose metadata
    ".bin.zst", ".bin.lzma",    # alternative compressors
    ".npy", ".npz",             # numpy state if used
)
_KNOWN_BROTLI_LOGICAL_SUFFIXES = (
    ".bin",
    ".mkv",
    ".mp4",
    ".pt",
    ".nrv",
    ".amrc",
    ".cmg1",
    ".cmg2",
    ".cmg3",
    ".qma9",
    ".tar.xz",
    ".fp16",
)
_KNOWN_ARCHIVE_BASENAMES = (
    "p",                        # top-submission-style packed payload member
    "x",                        # PR65/henosis-style packed payload member
    "fb",                       # PR89-style charged final-bias atom
)
_FORBIDDEN_ARCHIVE_NAMES = (
    ".DS_Store", "__MACOSX", "._",  # macOS resource forks
    "Thumbs.db",                     # Windows
)


def _validate_archive_members(members: list[str]) -> None:
    """Whitelist-based archive content validator.

    Raises RuntimeError if the archive contains files outside the known
    submission contract OR forbidden housekeeping files (macOS resource forks
    inflate rate by ~5-10KB silently). Called BEFORE eval so a corrupt
    archive fails fast instead of producing wrong scores.
    """
    if not members:
        raise RuntimeError(
            "[archive-validate] EMPTY archive — no members extracted. "
            "Likely corruption or wrong path."
        )
    forbidden_found: list[str] = []
    unknown_found: list[str] = []
    for member in members:
        # Forbidden housekeeping markers anywhere in the path
        for forbidden in _FORBIDDEN_ARCHIVE_NAMES:
            if forbidden in member:
                forbidden_found.append(member)
                break
        else:
            # Whitelist by exact basename or suffix. The exact basename path is
            # deliberately tiny: it admits top-submission-style member "p"
            # without allowing arbitrary extensionless debug payloads.
            lower = member.lower()
            basename = Path(member).name.lower()
            logical_lower = lower[:-3] if lower.endswith(".br") else lower
            if (
                basename not in _KNOWN_ARCHIVE_BASENAMES
                and not any(lower.endswith(s) for s in _KNOWN_ARCHIVE_SUFFIXES)
                and not (
                    lower.endswith(".br")
                    and any(logical_lower.endswith(s) for s in _KNOWN_BROTLI_LOGICAL_SUFFIXES)
                )
            ):
                unknown_found.append(member)
    if forbidden_found:
        raise RuntimeError(
            f"[archive-validate] FORBIDDEN files in archive: {forbidden_found}. "
            f"macOS resource forks / Windows housekeeping silently inflate the "
            f"rate term. Re-build the archive with the canonical zip helper "
            f"(see scripts/zip_archive.py) which strips these."
        )
    if unknown_found:
        raise RuntimeError(
            f"[archive-validate] UNKNOWN file types in archive: {unknown_found}. "
            f"Allowed suffixes: {_KNOWN_ARCHIVE_SUFFIXES}; allowed basenames: "
            f"{_KNOWN_ARCHIVE_BASENAMES}; allowed .br logical suffixes: "
            f"{_KNOWN_BROTLI_LOGICAL_SUFFIXES}. If a new artifact type was added "
            f"intentionally, append its suffix or exact basename to the "
            f"archive whitelist in experiments/contest_auth_eval.py."
        )


def _parse_inflate_env_overrides(items: list[str] | None) -> dict[str, str]:
    """Parse diagnostic-only environment overrides for the inflate subprocess."""

    overrides: dict[str, str] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"inflate env override must be KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        if not _INFLATE_ENV_KEY_RE.fullmatch(key):
            raise ValueError(f"invalid inflate env key {key!r}")
        if "\x00" in value:
            raise ValueError(f"inflate env override for {key!r} contains NUL")
        allowed = key in _ALLOWED_INFLATE_ENV_KEYS or key.startswith(_ALLOWED_INFLATE_ENV_PREFIXES)
        if not allowed:
            allowed_keys = sorted(_ALLOWED_INFLATE_ENV_KEYS)
            raise ValueError(
                f"inflate env key {key!r} is not allowed; use PACT_*/INFLATE_* "
                f"or one of {allowed_keys}"
            )
        overrides[key] = value
    return dict(sorted(overrides.items()))


def _inflate_env_for_device_policy(
    policy: str,
    overrides: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Return inflate env plus diagnostic blockers for an inflate-device policy."""

    normalized = str(policy or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"invalid inflate device policy {policy!r}")
    env = dict(overrides)
    blockers: list[str] = []
    if normalized != "auto":
        if "PACT_INFLATE_DEVICE" in env and env["PACT_INFLATE_DEVICE"] != normalized:
            raise ValueError(
                "conflicting PACT_INFLATE_DEVICE override: "
                f"{env['PACT_INFLATE_DEVICE']!r} vs --inflate-device {normalized!r}"
            )
        env["PACT_INFLATE_DEVICE"] = normalized
        blockers.append(f"inflate_device_policy_{normalized}")
        if normalized == "cpu":
            env.setdefault("CUDA_VISIBLE_DEVICES", "")
    if overrides:
        blockers.append("inflate_env_overrides_present")
    return dict(sorted(env.items())), sorted(set(blockers))


def _run_inflate(inflate_sh: Path, archive_dir: Path, inflated_dir: Path,
                 video_names_file: Path, *, timeout: int = 1800,
                 extra_env: dict[str, str] | None = None) -> float:
    """Invoke the submission's inflate.sh. Contest budget: 30 min on T4.
    Default timeout here is 30 min (1800s); pass --inflate-timeout for
    longer development runs.

    Council R3 #3 (CRITICAL): validate per-file byte counts so a partial
    inflate (silent drop of 1 of N videos) is caught here, not 200 lines
    later when upstream's `zip(dl_gt, dl_comp)` truncates to min().
    """
    inflated_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash", str(inflate_sh),
        str(archive_dir), str(inflated_dir), str(video_names_file),
    ]
    print(f"[inflate] cmd: {' '.join(cmd)}")
    print(f"[inflate] timeout: {timeout}s ({timeout / 60:.1f} min)")
    t0 = time.monotonic()
    env = {**os.environ}
    env.setdefault(
        "UV_PROJECT_ENVIRONMENT",
        str(inflated_dir.parent / "uv_project_env"),
    )
    # Public HNeRV/A1 runtimes commonly invoke either `${PYTHON:-python3}` or a
    # lane-local `${PYTHON_BIN:-python}`. Use the evaluator's interpreter by
    # default so exact-eval screens run in the repo venv that loaded this tool;
    # callers may still override these env vars for a contest container or
    # public replay environment.
    env.setdefault("PYTHON", sys.executable)
    env.setdefault("PYTHON_BIN", sys.executable)
    if extra_env:
        env.update(extra_env)
        print(f"[inflate] diagnostic env override keys: {sorted(extra_env)}")
    try:
        result = subprocess.run(cmd, timeout=timeout, check=False, env=env)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"[inflate] TIMED OUT after {timeout}s. Contest budget is "
            f"30 min on T4. If this is a development run, pass "
            f"--inflate-timeout 7200 (or higher) to bypass."
        ) from exc
    elapsed = time.monotonic() - t0
    print(f"[inflate] returncode={result.returncode} elapsed={elapsed:.1f}s")
    if result.returncode != 0:
        raise RuntimeError(f"[inflate] FAILED with returncode={result.returncode}")

    # Council R3 #3 + R4 #1 fix: STRICT per-video byte-count validation.
    # Each .raw is uint8 RGB at upstream/frame_utils.py's camera_size
    # (1164w x 874h) x NUM_FRAMES (1200) x 3 channels = 3,663,237,120 B.
    # R4 #1 (CRITICAL): use Path.with_suffix('.raw') NOT .stem so subdir
    # paths like 'subdir/0.mkv' resolve to 'inflated_dir/subdir/0.raw'
    # (matching submissions/robust_current/inflate.sh layout). The .stem
    # version stripped the parent dir and missed nested .raw files.
    test_videos = [n.strip() for n in video_names_file.read_text().splitlines()
                   if n.strip()]
    OUT_W, OUT_H, NUM_FRAMES = 1164, 874, 1200
    EXPECTED_RAW_BYTES = OUT_W * OUT_H * NUM_FRAMES * 3  # 3,663,237,120
    missing: list[str] = []
    wrong_size: list[tuple[str, int, int]] = []
    for vname in test_videos:
        # Preserve subdirs: 'a/b/0.mkv' → 'inflated_dir/a/b/0.raw'.
        rel_raw = Path(vname).with_suffix(".raw")
        raw_path = inflated_dir / rel_raw
        if not raw_path.exists():
            missing.append(str(rel_raw))
            continue
        actual = raw_path.stat().st_size
        if actual != EXPECTED_RAW_BYTES:
            wrong_size.append((str(rel_raw), actual, EXPECTED_RAW_BYTES))
    if missing:
        raise RuntimeError(
            f"[inflate] PARTIAL inflate — missing .raw for {len(missing)}/"
            f"{len(test_videos)} videos: {missing[:5]}{'…' if len(missing)>5 else ''}. "
            f"Upstream zip(dl_gt,dl_comp) would silently truncate to min(); "
            f"refusing to score."
        )
    if wrong_size:
        details = ", ".join(f"{n}={a}B (expected {e}B)" for n, a, e in wrong_size[:3])
        raise RuntimeError(
            f"[inflate] WRONG-SIZE .raw file(s): {details}. Each must be "
            f"{EXPECTED_RAW_BYTES:,} bytes (1164x874x1200x3). Likely "
            f"truncated mid-decode."
        )
    print(f"[inflate] produced {len(test_videos)} .raw file(s), each "
          f"{EXPECTED_RAW_BYTES:,} bytes — STRICT validation passed.")
    return elapsed


def _validate_uncompressed_dir(uncompressed_dir: Path,
                               video_names_file: Path) -> None:
    """Council R3 #2 (CRITICAL) + R4 #3 + R4 #4: upstream/evaluate.py
    computes the rate denominator as `sum(file.size for file in
    uncompressed_dir.rglob('*'))` — every file under the dir tree.
    ANY extra file (kaggle ingest leftovers, stray .raw caches, etc.)
    silently inflates the denominator and shifts the score.

    R4 #4 fix: hidden files (.DS_Store, .gitkeep) ARE counted by upstream
    (rglob doesn't filter), so they don't cause score drift — refusing
    on macOS-touched dirs is a false-positive. Only flag NON-hidden
    extras + missing.

    R4 #3 fix: also verify (uncompressed_dir / name).exists() for each
    expected video — upstream's frame_utils.py:107 does
    `assert (data_dir / fn).exists()` and would crash on a misplaced
    nested layout (videos/0.mkv vs 0.mkv at root)."""
    expected = {n.strip() for n in video_names_file.read_text().splitlines()
                if n.strip()}
    # R4-3: per-name existence check (catches nested-layout mismatch)
    not_found = [n for n in expected
                 if not (uncompressed_dir / n).exists()]
    if not_found:
        raise RuntimeError(
            f"[evaluate] expected video(s) not at "
            f"--uncompressed-dir/<name>: {not_found[:5]}. "
            f"Upstream's frame_utils asserts (data_dir / fn).exists() and "
            f"would crash. Check that {uncompressed_dir} contains the "
            f"videos listed in {video_names_file}."
        )
    # R5-1 fix: walk the dir and refuse on ANY extra file (including
    # hidden). Upstream's `rglob('*') if file.is_file()` counts every
    # file including .DS_Store, so a local .DS_Store WOULD shift the
    # rate vs a contest dir without it. Refuse so the operator cleans
    # the dir and gets 100% contest compliance. Also use FULL relative
    # path (not just .name) so duplicate-named files in subdirs are
    # caught (not aliased to the expected set).
    expected_paths = {str(Path(n)) for n in expected}
    extras: list[Path] = []
    for p in uncompressed_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(uncompressed_dir)
        if str(rel) not in expected_paths:
            extras.append(rel)
    if extras:
        raise RuntimeError(
            f"[evaluate] uncompressed-dir contamination — score would drift "
            f"vs official scorer: {len(extras)} EXTRA file(s) including "
            f"hidden (.DS_Store etc) and any duplicate-named subdir "
            f"entries: {[str(p) for p in extras[:8]]}"
            f"{'…' if len(extras)>8 else ''}. Move extras out of "
            f"{uncompressed_dir} for 100% contest compliance."
        )


def _run_upstream_evaluate(upstream_dir: Path, submission_dir: Path,
                           uncompressed_dir: Path, video_names_file: Path,
                           device: str, *, timeout: int = 1800) -> dict:
    """Invoke upstream/evaluate.py — the contest scorer. Returns the
    parsed score dict from the report.txt the script writes.

    R5-2 fix: do NOT pin --batch-size / --num-threads / --prefetch-queue-depth
    / --seed — let upstream/evaluate.py use its own defaults. Pinning them
    to specific values would itself be editorializing if the contest
    scorer ever uses different values. Per "100% contest compliance":
    pass exactly the args the contest scorer would, no more no less.
    Determinism env (CUBLAS_WORKSPACE_CONFIG) is set in subprocess env.
    Council R3 #2: pre-validate uncompressed-dir for contamination."""
    _validate_uncompressed_dir(uncompressed_dir, video_names_file)

    report_path = submission_dir / "report.txt"
    cmd = [
        sys.executable, str(upstream_dir / "evaluate.py"),
        "--submission-dir", str(submission_dir),
        "--uncompressed-dir", str(uncompressed_dir),
        "--video-names-file", str(video_names_file),
        "--device", device,
        "--report", str(report_path),
    ]
    print(f"[evaluate] cmd: {' '.join(cmd)}")
    t0 = time.monotonic()
    env = {**os.environ}
    # upstream/evaluate.py imports modules from upstream/ at top level
    pp = env.get("PYTHONPATH", "")
    if str(upstream_dir) not in pp:
        env["PYTHONPATH"] = f"{upstream_dir}:{pp}" if pp else str(upstream_dir)
    # Determinism env (Council R3 #4) — required per CLAUDE.md
    # "deterministic reproducibility" non-negotiable. CUBLAS_WORKSPACE_CONFIG
    # is required for torch.use_deterministic_algorithms.
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Modal/T4 DALI can hit NVML before scorer code has a chance to recover.
    # Keep this guard at the canonical evaluator boundary so lane scripts do
    # not have to rediscover it one-by-one.
    env.setdefault("DALI_DISABLE_NVML", "1")

    result = subprocess.run(cmd, timeout=timeout, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    print(f"[evaluate] returncode={result.returncode} elapsed={elapsed:.1f}s")
    print(f"[evaluate] stdout (last 4KB):\n{result.stdout[-4096:]}")
    if result.returncode != 0:
        print(f"[evaluate] stderr:\n{result.stderr[-2048:]}", file=sys.stderr)
        raise RuntimeError(f"[evaluate] FAILED with returncode={result.returncode}")

    # Parse report.txt AND captured stdout — they should be byte-identical
    # for the 6-line block (upstream prints + writes the same printed_results
    # list at upstream/evaluate.py:93-104). Cross-check catches:
    #   - report.txt write failure (script crashed between print and write)
    #   - stdout buffering / capture corruption
    #   - format drift introduced by an upstream patch
    archive_bytes_actual = (submission_dir / "archive.zip").stat().st_size

    if not report_path.exists():
        raise RuntimeError(
            f"[evaluate] no report.txt at {report_path}. Score-grade custody "
            "requires the evaluator report artifact; stdout is diagnostic only."
        )

    parsed_file = _parse_report(report_path, archive_size=archive_bytes_actual,
                                source="report.txt")
    if "Final score:" in result.stdout:
        try:
            parsed_stdout = _parse_report(result.stdout, archive_size=archive_bytes_actual,
                                          source="stdout")
        except RuntimeError as exc:
            print(f"[evaluate] stdout cross-check parse failed ({exc!r}) — "
                  f"trusting report.txt only", file=sys.stderr)
        else:
            # Cross-check: every numeric field must match within tiny tolerance.
            for k in ("avg_posenet_dist", "avg_segnet_dist", "rate_unscaled",
                      "final_score"):
                a, b = parsed_file[k], parsed_stdout[k]
                if abs(a - b) > 1e-6:
                    raise RuntimeError(
                        f"[evaluate] DIVERGENCE between report.txt and "
                        f"stdout for {k!r}: report={a} stdout={b}. One of "
                        f"the two surfaces was corrupted; refusing to ship."
                    )
    parsed_file["evaluate_elapsed_seconds"] = elapsed
    return parsed_file


def _parse_report(report_path: Path | str, *, archive_size: int,
                  source: str = "report.txt") -> dict:
    """Parse upstream/evaluate.py's report block into a structured dict.

    Accepts either a file path OR a raw string (per user directive
    "I thought we were getting away from fragile regex parsing" — we now
    parse both report.txt AND captured stdout, then cross-check to detect
    any divergence between the two source-of-truth surfaces).

    The contest report format (printed lines 96-100 of upstream/evaluate.py
    AND written to report.txt with identical content):
        === Evaluation results over 600 samples ===
          Average PoseNet Distortion: 0.01070000
          Average SegNet Distortion: 0.00240000
          Submission file size: 337748 bytes
          Original uncompressed size: 37545489 bytes
          Compression Rate: 0.00899
          Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.90
    """
    if isinstance(report_path, Path) and report_path.exists():
        text = Path(report_path).read_text()
    elif isinstance(report_path, str):
        # Treat strings containing report newlines as raw text, not paths.
        # Path.exists() on raw stdout can raise OSError for "file name too
        # long"; stdout cross-checks must be robust because they are part of
        # the exact-eval custody chain.
        if "\n" not in report_path and len(report_path) < 4096:
            try:
                candidate = Path(report_path)
                text = candidate.read_text() if candidate.exists() else report_path
            except OSError:
                text = report_path
        else:
            text = report_path
    else:
        raise RuntimeError(f"[{source}] not a path or readable string")

    def _grab(pattern: str, default: float | None = None) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    pose = _grab(r"Average PoseNet Distortion:\s*([0-9.eE+-]+)")
    seg = _grab(r"Average SegNet Distortion:\s*([0-9.eE+-]+)")
    rate_unscaled_reported = _grab(r"Compression Rate:\s*([0-9.eE+-]+)")
    final = _grab(r"Final score[^=]*=\s*([0-9.eE+-]+)")
    n_samples = re.search(r"results over (\d+) samples", text)
    submission_size = re.search(r"Submission file size:\s*([0-9,]+)\s*bytes", text)
    original_size = re.search(r"Original uncompressed size:\s*([0-9,]+)\s*bytes", text)

    if pose is None or seg is None or rate_unscaled_reported is None or final is None:
        raise RuntimeError(
            f"[evaluate] could not parse report.txt:\n{text[:1024]}"
        )

    submission_size_bytes = (
        int(submission_size.group(1).replace(",", ""))
        if submission_size is not None
        else archive_size
    )
    original_size_bytes = (
        int(original_size.group(1).replace(",", ""))
        if original_size is not None
        else None
    )
    if submission_size_bytes != archive_size:
        raise RuntimeError(
            f"[evaluate] report submission size {submission_size_bytes} != "
            f"observed archive size {archive_size}; refusing rounded-rate custody drift."
        )
    if original_size_bytes is None or original_size_bytes <= 0:
        raise RuntimeError(
            "[evaluate] could not parse positive Original uncompressed size from report"
        )
    rate_unscaled = archive_size / original_size_bytes

    # Council R3 #5 (Medium): reject NaN/inf — float() parses both silently.
    # A divide-by-zero in upstream's distortion sum would slip through as
    # final_score=NaN that "looks like" a number. Refuse loud.
    import math as _math
    for label, val in (("posenet_dist", pose), ("segnet_dist", seg),
                       ("rate_unscaled", rate_unscaled),
                       ("rate_unscaled_reported", rate_unscaled_reported),
                       ("final_score", final)):
        if not _math.isfinite(val):
            raise RuntimeError(
                f"[evaluate] non-finite {label}={val} in report.txt — refuse "
                f"to ship a NaN/inf score. Investigate upstream evaluate run."
            )
    if pose < 0 or seg < 0 or rate_unscaled < 0 or rate_unscaled_reported < 0 or final < 0:
        raise RuntimeError(
            f"[evaluate] negative metric in report (pose={pose}, seg={seg}, "
            f"rate={rate_unscaled_reported}, final={final}) — distortions must be ≥0."
        )
    expected_n = 600  # contest pair count (1200 frames / seq_len=2)
    actual_n = int(n_samples.group(1)) if n_samples else None
    if actual_n != expected_n:
        raise RuntimeError(
            f"[evaluate] expected {expected_n} samples but report says "
            f"{actual_n}. Likely partial inflate (Council R3 #3) slipped "
            f"past the .raw byte-count check."
        )

    score_pose = (10.0 * pose) ** 0.5
    score_seg = 100.0 * seg
    score_rate = 25.0 * rate_unscaled
    score_recomputed = score_seg + score_pose + score_rate
    score_rounding_abs_delta = abs(score_recomputed - final)

    # Council R3 #6 (Medium): assert recomputed score matches reported
    # within upstream's print precision (.2f → ±0.005, generous bound 0.01).
    # A formula divergence (upstream changes the 100/√10/25 weights) would
    # otherwise slip through without notice.
    if abs(score_recomputed - final) > 0.01:
        raise RuntimeError(
            f"[evaluate] score formula divergence: reported final={final:.4f} "
            f"but recomputed (100*seg + sqrt(10*pose) + 25*rate) = "
            f"{score_recomputed:.4f}. Diff={abs(score_recomputed - final):.4f} "
            f"exceeds 0.01 tolerance. Upstream may have changed weights."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "final_score": final,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "rate_unscaled": rate_unscaled,
        "rate_unscaled_reported_rounded": rate_unscaled_reported,
        "original_uncompressed_size_bytes": original_size_bytes,
        "score_pose_contribution": score_pose,
        "score_seg_contribution": score_seg,
        "score_rate_contribution": score_rate,
        "score_recomputed_from_components": score_recomputed,
        "canonical_score": score_recomputed,
        "canonical_score_source": "score_recomputed_from_components",
        "reported_final_score_display_rounded": final,
        "score_rounding_abs_delta": score_rounding_abs_delta,
        "score_reported_rounded_differs_from_canonical": score_rounding_abs_delta > 1e-12,
        "archive_size_bytes": archive_size,
        "n_samples": actual_n,
        "report_path": str(report_path),
    }


def _auth_eval_evidence_contract(
    device: str,
    n_samples: int,
    provenance: dict,
    *,
    diagnostic_blockers: list[str] | None = None,
) -> dict:
    """Return explicit evidence semantics for the selected eval device."""

    if provenance.get("modal_auth_eval_advisory_only") is True:
        diagnostic_blockers = [
            *(diagnostic_blockers or []),
            "modal_training_wrapper_auth_eval_advisory_only",
        ]

    if diagnostic_blockers:
        return {
            "evidence_grade": "B",
            "lane_tag": "[diagnostic-auth-eval]",
            "score_axis": f"diagnostic_{device}",
            "evidence_semantics": "diagnostic_auth_eval_non_promotable",
            "exact_cuda_eval_complete": False,
            "score_claim": False,
            "promotion_eligible": False,
            "score_claim_valid": False,
            "rank_or_kill_eligible": False,
            "cpu_leaderboard_reproduction_eligible": False,
            "diagnostic_blockers": sorted(set(diagnostic_blockers)),
            "allowed_uses": [
                "diagnostic_debugging",
                "mechanism_localization",
            ],
        }

    is_linux_x86_64 = (
        provenance.get("platform_system") == "Linux"
        and str(provenance.get("platform_machine") or "").lower() in {"x86_64", "amd64"}
    )
    # Per CLAUDE.md SIREN audit 2026-05-13 DEFECT #9 + "Submission auth eval —
    # BOTH CPU AND CUDA" section: NVIDIA T4 / A100 / 4090 / H100 / A10G / L40S
    # are all 1:1 contest-compliant for the CUDA axis (Linux x86_64 + CUDA
    # runtime). The contest's GitHub Actions bot scores CUDA on T4, but exact
    # contest-faithful CUDA replays on A100/4090/H100/A10G/L40S also qualify
    # as evidence_grade="contest-CUDA". Previously this gate accepted T4 only,
    # silently downgrading every A100 / 4090 / H100 result to "B" (diagnostic).
    _gpu_contest_faithful_cuda = is_contest_cuda_equivalent_gpu(
        gpu_model=str(provenance.get("gpu_model") or ""),
        gpu_t4_match=provenance.get("gpu_t4_match") is True,
    )
    is_cuda_contest_full = (
        device == "cuda"
        and n_samples == 600
        and is_linux_x86_64
        and _gpu_contest_faithful_cuda
    )
    is_cpu_full = device == "cpu" and n_samples == 600 and is_linux_x86_64
    if is_cuda_contest_full:
        return {
            "evidence_grade": "contest-CUDA",
            "lane_tag": "[contest-CUDA]",
            "score_axis": "contest_cuda",
            "evidence_semantics": "contest_cuda_exact_auth_eval",
            "exact_cuda_eval_complete": True,
            "score_claim": True,
            "promotion_eligible": False,
            "score_claim_valid": True,
            "rank_or_kill_eligible": False,
            "cpu_leaderboard_reproduction_eligible": False,
            "promotion_blockers": [
                "raw_auth_eval_does_not_verify_submission_policy_gates",
                "cpu_leaderboard_reproduction_not_adjudicated",
                "pre_submission_compliance_check_not_recorded",
            ],
            "rank_or_kill_blockers": [
                "raw_auth_eval_not_rank_or_kill_authority",
                "requires_adjudicated_cuda_cpu_policy_review",
            ],
            "allowed_uses": [
                "internal_cuda_frontier_ranking",
                "promotion_review_input",
                "paper_empirical_score_when_custody_complete",
            ],
        }
    if is_cpu_full:
        return {
            "evidence_grade": "contest-CPU",
            "lane_tag": "[contest-CPU]",
            "score_axis": "contest_cpu",
            "evidence_semantics": "public_leaderboard_cpu_reproduction",
            "exact_cuda_eval_complete": False,
            "score_claim": True,
            "promotion_eligible": False,
            "score_claim_valid": True,
            "rank_or_kill_eligible": False,
            "cpu_leaderboard_reproduction_eligible": True,
            "promotion_blockers": [
                "raw_auth_eval_does_not_verify_submission_policy_gates",
                "pre_submission_compliance_check_not_recorded",
                "result_review_packet_not_recorded",
            ],
            "rank_or_kill_blockers": [
                "raw_auth_eval_not_rank_or_kill_authority",
                "requires_adjudicated_cuda_cpu_policy_review",
            ],
            "allowed_uses": [
                "cpu_axis_score_claim",
                "public_leaderboard_reproduction",
                "cpu_cuda_drift_diagnosis",
                "medal_band_context_with_matching_archive_runtime",
                "submission_packet_input_after_compliance_review",
            ],
        }
    if device == "cpu" and n_samples == 600:
        lane_tag = (
            "[macOS-CPU advisory]"
            if provenance.get("platform_system") == "Darwin"
            else "[CPU advisory]"
        )
        return {
            "evidence_grade": (
                "macOS-CPU advisory"
                if provenance.get("platform_system") == "Darwin"
                else "CPU advisory"
            ),
            "lane_tag": lane_tag,
            "score_axis": "cpu_advisory",
            "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
            "exact_cuda_eval_complete": False,
            "score_claim": False,
            "promotion_eligible": False,
            "score_claim_valid": False,
            "rank_or_kill_eligible": False,
            "cpu_leaderboard_reproduction_eligible": False,
            "hardware_compliance_blocker": "contest_cpu_requires_linux_x86_64",
            "allowed_uses": [
                "diagnostic_debugging",
                "cpu_cuda_drift_hypothesis_generation",
            ],
        }
    return {
        "evidence_grade": "B",
        "lane_tag": "[diagnostic-auth-eval]",
        "score_axis": f"diagnostic_{device}",
        "evidence_semantics": "diagnostic_auth_eval_non_promotable",
        "exact_cuda_eval_complete": False,
        "score_claim": False,
        "promotion_eligible": False,
        "score_claim_valid": False,
        "rank_or_kill_eligible": False,
        "cpu_leaderboard_reproduction_eligible": False,
        "allowed_uses": [
            "diagnostic_debugging",
            "smoke_or_infrastructure_triage",
        ],
    }


def _path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_durable_result_output(args: argparse.Namespace) -> None:
    """Require score-grade auth-eval JSON to survive outside temp cleanup."""

    temp_root = Path(tempfile.gettempdir())
    allow_temp = bool(getattr(args, "allow_temp_work_dir", False))
    work_dir = getattr(args, "work_dir", None)
    json_out = getattr(args, "json_out", None)
    if work_dir is None and json_out is None and not allow_temp:
        raise SystemExit(
            "contest_auth_eval score evidence requires --work-dir or --json-out. "
            "Use --allow-temp-work-dir only for diagnostic scratch runs that must "
            "not be treated as score custody."
        )
    for path in (work_dir, json_out):
        if path is None:
            continue
        path_obj = Path(path)
        if _path_is_under(path_obj, temp_root) and not allow_temp:
            raise SystemExit(
                f"contest_auth_eval evidence path is under temp storage: {path_obj}. "
                "Choose a durable repo/provider work dir or pass --allow-temp-work-dir "
                "for diagnostic scratch only."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--archive", type=Path, required=True,
                        help="Path to archive.zip — the submission to evaluate")
    parser.add_argument("--inflate-sh", type=Path,
                        default=Path("submissions/robust_current/inflate.sh"),
                        help="Submission's inflate.sh (default: robust_current)")
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"),
                        help="upstream/ root (has evaluate.py, modules.py, videos/)")
    parser.add_argument("--video-names-file", type=Path,
                        default=Path("upstream/public_test_video_names.txt"),
                        help="Test video names list (one per line)")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "mps", "cpu"],
                        help="Eval device. CUDA on T4/equivalent is the "
                             "promotion axis; CPU is contest-CPU only on "
                             "Linux x86_64, while macOS CPU is advisory. MPS "
                             "is diagnostic only.")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Working directory (default: tempfile)")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Optional durable copy of contest_auth_eval.json.")
    parser.add_argument("--allow-temp-work-dir", action="store_true",
                        help="Allow temp-dir evidence for diagnostic scratch only; "
                             "never use for score custody.")
    parser.add_argument("--inflate-timeout", type=int, default=1800,
                        help="Inflate.sh timeout in seconds. Contest budget "
                             "is 30 min (1800s) on T4. Default matches.")
    parser.add_argument("--evaluate-timeout", type=int, default=1800,
                        help="upstream/evaluate.py timeout in seconds.")
    parser.add_argument("--keep-work-dir", action="store_true",
                        help="Don't delete work dir on success (for debugging)")
    parser.add_argument("--expected-runtime-tree-sha256", default=None,
                        help="Fail if the inflate runtime dependency tree hash differs.")
    parser.add_argument(
        "--inflate-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Diagnostic-only environment override for inflate.sh. Overrides "
            "apply only to the inflate subprocess and demote the result to "
            "non-promotable diagnostic evidence. Allowed keys: PACT_*, "
            "INFLATE_*, CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument(
        "--inflate-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Diagnostic-only inflate device policy. 'auto' preserves the "
            "submission runtime default. 'cpu' sets PACT_INFLATE_DEVICE=cpu "
            "and hides CUDA from inflate; 'cuda' sets PACT_INFLATE_DEVICE=cuda. "
            "Non-auto values demote the result to diagnostic evidence."
        ),
    )
    parser.add_argument(
        "--scorer-input-cache-hashes-out",
        type=Path,
        default=None,
        help=(
            "Optional compact JSON artifact with streamed scorer-input tensor "
            "hashes for the inflated raw surface. This does not write tensor "
            "payloads and does not change score authority."
        ),
    )
    parser.add_argument(
        "--scorer-input-cache-hash-batch-pairs",
        type=int,
        default=8,
        help="Batch size for --scorer-input-cache-hashes-out streaming preprocessing.",
    )
    parser.add_argument(
        "--scorer-input-cache-tensors-out-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for full scorer-input NumPy tensor cache export "
            "from the inflated raw surface. Intended for local runs or explicitly "
            "mounted auth environments; tensor payloads are not returned through "
            "Modal result artifacts."
        ),
    )
    parser.add_argument(
        "--scorer-input-cache-tensor-batch-pairs",
        type=int,
        default=8,
        help="Batch size for --scorer-input-cache-tensors-out-dir preprocessing.",
    )
    parser.add_argument(
        "--scorer-input-cache-tensor-large-pair-threshold",
        type=int,
        default=64,
        help=(
            "Maximum pair count allowed for full tensor export without explicit "
            "--allow-large-scorer-input-cache-tensor-export acknowledgement."
        ),
    )
    parser.add_argument(
        "--allow-large-scorer-input-cache-tensor-export",
        action="store_true",
        help="Explicitly acknowledge full scorer-input tensor export for large surfaces.",
    )
    args = parser.parse_args()
    if args.scorer_input_cache_hash_batch_pairs < 1:
        raise SystemExit("--scorer-input-cache-hash-batch-pairs must be >= 1")
    if args.scorer_input_cache_tensor_batch_pairs < 1:
        raise SystemExit("--scorer-input-cache-tensor-batch-pairs must be >= 1")
    if args.scorer_input_cache_tensor_large_pair_threshold < 1:
        raise SystemExit("--scorer-input-cache-tensor-large-pair-threshold must be >= 1")

    # Resolve required paths
    archive = args.archive.resolve()
    if not archive.exists():
        raise SystemExit(f"--archive does not exist: {archive}")
    inflate_sh = args.inflate_sh.resolve()
    if not inflate_sh.exists():
        raise SystemExit(f"--inflate-sh does not exist: {inflate_sh}")
    upstream_dir = args.upstream_dir.resolve()
    if not (upstream_dir / "evaluate.py").exists():
        raise SystemExit(
            f"--upstream-dir missing evaluate.py: {upstream_dir}. "
            f"Did you forget to clone the pinned upstream snapshot?"
        )
    _validate_durable_result_output(args)

    # Codex F5 fix (2026-04-28, canonical guard for all lanes): the
    # submission's inflate.sh sources $SELF_DIR/config.env to read
    # PYTHON_INFLATE. If config.env is missing, inflate.sh falls into the
    # legacy ffmpeg path and tries to read extracted/0.mkv, which never
    # exists in a renderer-archive layout. Lane RM-d burned $1+ discovering
    # this; the launcher tarball used to silently exclude .env files
    # (fixed in scripts/launch_lane_on_vastai.py). Guard here so any future
    # lane reusing contest_auth_eval gets a clear error instead of an
    # opaque ffmpeg "No such file or directory" 200 lines downstream.
    # Placed AFTER the upstream check so existing tests that pass a fake
    # inflate.sh in tmp_path get the upstream-missing error first (the
    # config.env check fires only when the inflate.sh declares that contract.
    _validate_config_env_for_renderer_dispatch(inflate_sh)
    video_names_file = args.video_names_file.resolve()
    if not video_names_file.exists():
        # Common alt path
        alt = upstream_dir / "public_test_video_names.txt"
        if alt.exists():
            video_names_file = alt
        else:
            raise SystemExit(f"--video-names-file does not exist: {video_names_file}")

    _ensure_uv_available()
    try:
        raw_inflate_env_overrides = _parse_inflate_env_overrides(args.inflate_env)
        inflate_env_overrides, diagnostic_blockers = _inflate_env_for_device_policy(
            args.inflate_device,
            raw_inflate_env_overrides,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid inflate diagnostic override: {exc}") from exc

    # Set up working directory in canonical contest-shape:
    #   work/
    #     archive.zip       (the submission)
    #     extracted/        (archive contents)
    #     inflated/         (inflate.sh output)
    #     report.txt        (evaluate.py output)
    #     provenance.json   (env snapshot)
    #     contest_auth_eval.json  (final result)
    if args.work_dir:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    elif args.json_out is not None and not args.allow_temp_work_dir:
        json_out = args.json_out.resolve()
        work_dir = json_out.parent / f"{json_out.stem}_workdir"
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="contest_auth_"))
        cleanup = not args.keep_work_dir

    try:
        # Copy archive into work_dir so submission_dir layout matches what
        # upstream/evaluate.py expects: it reads (submission_dir / archive.zip).
        archive_in_work = work_dir / "archive.zip"
        shutil.copy2(archive, archive_in_work)

        # Provenance snapshot
        prov = _record_provenance(work_dir, archive, inflate_sh, upstream_dir, args)
        prov["inflate_device_policy"] = args.inflate_device
        if inflate_env_overrides:
            prov["inflate_env_overrides"] = inflate_env_overrides
            prov["inflate_env_override_mode"] = "diagnostic_non_promotable"
        _validate_expected_runtime_tree(prov, args.expected_runtime_tree_sha256)
        print(f"[contest_auth_eval] provenance saved: {work_dir / 'provenance.json'}")
        print(f"[contest_auth_eval] archive sha256: {prov['archive_sha256']}")

        # Stage 1: extract archive
        extracted = work_dir / "extracted"
        members = _extract_archive(archive_in_work, extracted)
        print(f"[contest_auth_eval] extracted {len(members)} member(s): {members}")

        # Stage 1b (deep hardening pass 3 dim 3): whitelist-based archive
        # validation. Catches stale debug artifacts, macOS resource forks,
        # and unknown file types BEFORE eval so wrong scores never escape.
        _validate_archive_members(members)
        print("[contest_auth_eval] archive members validated against whitelist")

        # Stage 2: run submission's inflate.sh on the extracted archive dir
        inflated = work_dir / "inflated"
        exact_eval_t0 = time.monotonic()
        inflate_elapsed_seconds = _run_inflate(
            inflate_sh, extracted, inflated, video_names_file,
            timeout=args.inflate_timeout,
            extra_env=inflate_env_overrides,
        )
        _record_inflate_runtime_artifacts(prov, work_dir, extracted)
        inflated_manifest = _record_inflated_output_artifacts(
            prov, work_dir, inflated, video_names_file
        )
        if args.scorer_input_cache_hashes_out is not None:
            _record_scorer_input_cache_hash_artifact(
                prov,
                work_dir,
                inflated,
                video_names_file,
                inflated_manifest,
                args.scorer_input_cache_hashes_out,
                batch_pairs=args.scorer_input_cache_hash_batch_pairs,
            )
        if args.scorer_input_cache_tensors_out_dir is not None:
            _record_scorer_input_cache_tensor_artifact(
                prov,
                work_dir,
                inflated,
                video_names_file,
                inflated_manifest,
                args.scorer_input_cache_tensors_out_dir,
                batch_pairs=args.scorer_input_cache_tensor_batch_pairs,
                allow_large_tensor_export=args.allow_large_scorer_input_cache_tensor_export,
                large_pair_threshold=args.scorer_input_cache_tensor_large_pair_threshold,
            )

        # Stage 3: run upstream/evaluate.py on submission_dir = work_dir
        # Note: evaluate.py needs (submission_dir / 'archive.zip') AND
        # (submission_dir / 'inflated/'). work_dir has both.
        result = _run_upstream_evaluate(
            upstream_dir, work_dir,
            uncompressed_dir=upstream_dir / "videos",
            video_names_file=video_names_file,
            device=args.device,
            timeout=args.evaluate_timeout,
        )
        result["inflate_elapsed_seconds"] = inflate_elapsed_seconds
        result["contest_auth_eval_elapsed_seconds"] = time.monotonic() - exact_eval_t0

        # Save final JSON next to the work dir
        result["provenance"] = prov
        result.update(
            _auth_eval_evidence_contract(
                args.device,
                int(result.get("n_samples") or 0),
                prov,
                diagnostic_blockers=diagnostic_blockers or None,
            )
        )
        result["work_dir"] = str(work_dir)
        out_json = work_dir / "contest_auth_eval.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        if args.json_out is not None:
            durable_json = args.json_out.resolve()
            durable_json.parent.mkdir(parents=True, exist_ok=True)
            with open(durable_json, "w") as f:
                json.dump(result, f, indent=2)
                f.write("\n")

        # Print sentinel line for downstream parsers (matches the format
        # auth_eval_renderer.py uses, so existing log scrapers keep working)
        print(f"\nRESULT_JSON: {json.dumps(result)}")
        print("\n=== CONTEST AUTH EVAL ===")
        print(f"  Canonical score: {result['canonical_score']:.12f}")
        print(f"  Reported final:  {result['final_score']:.4f}")
        print(f"  PoseNet dist:   {result['avg_posenet_dist']:.6f}")
        print(f"  SegNet dist:    {result['avg_segnet_dist']:.6f}")
        print(f"  Rate (unscaled): {result['rate_unscaled']:.6f}")
        print(f"  Archive bytes:  {result['archive_size_bytes']:,}")
        print(f"  Result JSON:    {out_json}")
        if args.json_out is not None:
            print(f"  Durable JSON:   {args.json_out.resolve()}")

        return 0
    finally:
        if cleanup:
            print(f"[contest_auth_eval] cleaning up {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
