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
import fcntl
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

from tac.contest_budget import budget_verdict_for_receipt
from tac.contest_compliance import compute_upstream_snapshot_sha256
from tac.device_axis_eval import is_contest_cuda_equivalent_gpu
from tac.gt_lineage import AUTHORITY_LINEAGE, GtLineageError, runtime_decode_lineage
from tac.process_group_kill import run_in_process_group

# Line-buffer stdout so progress flushes to log files immediately.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

# Schema version for the JSON we emit. Version 1 permits additive custody
# fields; bump for a breaking shape or semantic change so downstream tooling
# (BATTLE_PLAN parsers, leaderboard, etc.) can detect incompatibility.
SCHEMA_VERSION = 1
AUTH_EVAL_ENV_PACKAGES = ("torch", "torchvision", "timm", "numpy")
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
_FORBIDDEN_DECODE_RUNTIME_REFERENCE_PATTERNS = {
    "contest_source_video": re.compile(
        r"upstream.{0,160}videos.{0,160}0\.mkv",
        re.IGNORECASE | re.DOTALL,
    ),
    "upstream_scorer_weights": re.compile(
        r"models.{0,160}\.safetensors",
        re.IGNORECASE | re.DOTALL,
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_upstream_snapshot_sha256(upstream_dir: Path) -> str:
    """Hash the exact frozen evaluator tree or fail before score production."""

    try:
        digest = compute_upstream_snapshot_sha256(
            upstream_dir,
            upstream_subdir=".",
            reject_executable_artifacts=True,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "canonical upstream snapshot could not be hashed without omitted dependencies"
        ) from exc
    if digest is None:
        raise RuntimeError("canonical upstream snapshot is missing")
    return digest


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
        # AppleDouble sidecars (``._foo.py``) are ExFAT/HFS+ resource-fork
        # METADATA, not Python source: binary, and their header byte 37 is
        # 0xb0 so ``read_text()`` raises UnicodeDecodeError. Any archive staged
        # on an ExFAT volume (APDataStore/VertigoDataTier) carries one twin per
        # real file, so an unfiltered ``*.py`` glob crashes the whole manifest
        # — measured 2026-08-18 on a 29-twin staged runtime. They can never be
        # legitimate modules: ``._foo`` is not an importable identifier, so
        # skipping them cannot change which REAL sources are hashed.
        if path.name.startswith("._"):
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


def _path_exists_case_sensitive(path: Path) -> bool:
    """Return True iff ``path`` exists AND every component matches exact case.

    On macOS HFS+/APFS (case-insensitive by default) and on Windows NTFS,
    ``Path.exists()`` returns True even when the real on-disk basename uses
    a different case than ``path``. The contest runtime hash, however, is
    computed identically on Linux Modal workers where the filesystem IS
    case-sensitive. A LOCAL projector that succeeds on a case-fold match
    will add a phantom entry to ``repo_local_tac_import_manifest`` that the
    Linux worker cannot reproduce, producing a runtime_tree_sha256 mismatch
    that fails dispatch pre-validation (Catalog #229 / Catalog #146 contract).

    The canonical fix: walk parent ``iterdir()`` and require exact basename
    match. Symlinks and broken paths are handled conservatively (return
    False rather than raising).

    Anchor: PR111 paired-CUDA RATIFICATION 2026-05-28 failed 4× with
    ``inflate runtime tree hash mismatch`` because
    ``tac.dykstra_pareto_solver.Polytope`` (capital-P) resolved to LOCAL
    ``polytope.py`` via case-fold; Linux worker correctly returned False.
    """

    try:
        path = Path(path)
    except TypeError:
        return False
    # Walk from the filesystem root down: every component must match its
    # parent's iterdir() listing exactly. We use parent.iterdir() rather
    # than os.listdir() so test fixtures that monkeypatch Path behavior
    # still route through the canonical pathlib surface.
    components = path.parts
    if not components:
        return False
    current = Path(components[0])
    if not current.exists():
        return False
    for part in components[1:]:
        try:
            children = {child.name for child in current.iterdir()}
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return False
        if part not in children:
            return False
        current = current / part
    return current.exists()


def _module_exists(module_name: str, repo_root: Path) -> bool:
    if not _is_tac_module(module_name):
        return False
    rel_parts = module_name.split(".")[1:]
    tac_root = repo_root / "src" / "tac"
    if not rel_parts:
        return _path_exists_case_sensitive(tac_root / "__init__.py")
    return (
        _path_exists_case_sensitive(
            tac_root.joinpath(*rel_parts).with_suffix(".py")
        )
        or _path_exists_case_sensitive(
            tac_root.joinpath(*rel_parts) / "__init__.py"
        )
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
    except (SyntaxError, UnicodeDecodeError, OSError) as exc:
        # Defense-in-depth beside the AppleDouble filter in
        # ``_runtime_python_files``: a file that is unreadable or not valid
        # UTF-8 must become a RECORDED parse_error, never an uncaught crash
        # that takes down the whole auth-eval run. The filter removes the known
        # instance; this removes the class (the crash was UnicodeDecodeError,
        # which the SyntaxError-only catch let escape).
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
        # Case-sensitive check per _path_exists_case_sensitive contract above:
        # avoids macOS HFS+/APFS case-fold producing manifests that diverge
        # from Linux Modal workers (Catalog #229 / Catalog #146).
        if _path_exists_case_sensitive(init_path):
            paths.append(init_path)
    if rel_parts:
        module_path = tac_root.joinpath(*rel_parts).with_suffix(".py")
        if _path_exists_case_sensitive(module_path):
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


def _decode_runtime_forbidden_reference_guard(
    runtime_root: Path,
    external_dependency_roots: list[dict],
    repo_local_tac: dict,
    repo_root: Path,
) -> dict:
    """Refuse scorer/source dependencies reachable from ``inflate.sh``.

    The strict scorer rule is about the decode-time dependency closure, not a
    repository-wide token search.  Scan the selected ``--inflate-sh`` root,
    every explicitly declared external runtime root, and the recursively
    resolved repo-local ``tac`` import closure.  Encoder-side tools elsewhere
    in the repository are deliberately outside this surface.

    The patterns tolerate common source spellings such as ``Path(root) /
    "models" / "segnet.safetensors"`` and ``os.path.join(...)``.  A textual
    occurrence inside reachable runtime source is refused even when guarded by
    an environment flag: shipping that branch still makes the forbidden object
    part of the decoder program's dependency surface.
    """

    candidates: dict[Path, str] = {}
    direct_artifact_violations: list[dict[str, str]] = []

    def scan_forbidden_artifact_paths(root: Path, role: str) -> None:
        if not root.exists():
            return
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative_parts = tuple(part.lower() for part in path.relative_to(root).parts)
            if any(part in _RUNTIME_DEPENDENCY_SKIP_DIRS for part in relative_parts):
                continue
            full_parts = tuple(part.lower() for part in path.resolve().parts)
            reference_kind = None
            if path.suffix.lower() == ".safetensors" and "models" in full_parts:
                reference_kind = "upstream_scorer_weights"
            if len(full_parts) >= 3 and full_parts[-3:] == (
                "upstream",
                "videos",
                "0.mkv",
            ):
                reference_kind = "contest_source_video"
            if reference_kind is not None:
                direct_artifact_violations.append(
                    {
                        "path": _repo_rel(path, repo_root),
                        "role": role,
                        "reference_kind": reference_kind,
                    }
                )

    def add_manifest_files(root: Path, files: list[dict], role: str) -> None:
        for entry in files:
            relative = entry.get("relative_path")
            if not isinstance(relative, str) or not relative:
                continue
            candidates.setdefault((root / relative).resolve(), role)

    add_manifest_files(
        runtime_root,
        _runtime_root_file_manifest(runtime_root, repo_root),
        "inflate_runtime_root",
    )
    scan_forbidden_artifact_paths(runtime_root, "inflate_runtime_root")
    for external in external_dependency_roots:
        raw_root = external.get("root")
        if not isinstance(raw_root, str) or not raw_root:
            continue
        add_manifest_files(
            Path(raw_root).resolve(),
            list(external.get("files", [])),
            "declared_external_runtime_root",
        )
        scan_forbidden_artifact_paths(
            Path(raw_root).resolve(),
            "declared_external_runtime_root",
        )
    for entry in repo_local_tac.get("files", []):
        relative = entry.get("relative_path")
        if isinstance(relative, str) and relative:
            candidates.setdefault(
                (repo_root / relative).resolve(),
                "repo_local_tac_import_closure",
            )

    violations: list[dict[str, str]] = list(direct_artifact_violations)
    scanned: list[dict[str, str]] = []
    for path, role in sorted(candidates.items(), key=lambda item: str(item[0])):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RuntimeError(
                f"decode-time dependency could not be scanned: {path}: {exc}"
            ) from exc
        scanned.append({"path": _repo_rel(path, repo_root), "role": role})
        for reference_kind, pattern in (
            _FORBIDDEN_DECODE_RUNTIME_REFERENCE_PATTERNS.items()
        ):
            if pattern.search(source):
                violations.append(
                    {
                        "path": _repo_rel(path, repo_root),
                        "role": role,
                        "reference_kind": reference_kind,
                    }
                )

    receipt = {
        "schema": "contest_auth_eval_decode_runtime_forbidden_reference_guard_v1",
        "selection": "actual_inflate_root_plus_declared_roots_plus_recursive_tac_imports",
        "scanned_file_count": len(scanned),
        "scanned_files": scanned,
        "direct_forbidden_artifact_count": len(direct_artifact_violations),
        "forbidden_reference_kinds": sorted(
            _FORBIDDEN_DECODE_RUNTIME_REFERENCE_PATTERNS
        ),
        "violations": violations,
        "passed": not violations,
    }
    if violations:
        compact = ", ".join(
            f"{row['reference_kind']}@{row['path']}" for row in violations
        )
        raise RuntimeError(
            "decode-time scorer/source dependency forbidden by strict scorer rule: "
            + compact
        )
    return receipt


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
    forbidden_reference_guard = _decode_runtime_forbidden_reference_guard(
        root,
        external_dependency_roots,
        repo_local_tac,
        repo_root,
    )
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
    # Environment-free custody digest: ONLY the runtime files' (relative_path,
    # bytes, sha256) plus the upstream evaluate.py identity. Deliberately
    # excludes runtime_root_name, absolute paths, external dependency roots,
    # and the repo-local tac import scan — all of which legitimately differ
    # between the local packer and a provider host (Modal extracts under a
    # different root and resolves repo-local imports against a different
    # mount layout), so any digest that includes them cannot be validated
    # consistently across environments (the 2026-08-04 r9m deadlock).
    files_payload = {
        "files": sorted(
            (
                {
                    "relative_path": f["relative_path"],
                    "bytes": f["bytes"],
                    "sha256": f["sha256"],
                }
                for f in files
            ),
            key=lambda row: str(row["relative_path"]),
        ),
        "upstream_evaluate_py": upstream_eval,
    }
    runtime_files_sha = hashlib.sha256(
        json.dumps(files_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "contest_auth_eval_runtime_dependency_manifest_v1",
        "runtime_root": str(root),
        "runtime_file_count": len(files),
        "runtime_tree_sha256": tree_sha,
        "runtime_content_tree_sha256": content_tree_sha,
        "runtime_files_sha256": runtime_files_sha,
        "files": files,
        "external_dependency_roots": external_dependency_roots,
        "repo_local_tac_import_manifest": repo_local_tac,
        "decode_runtime_forbidden_reference_guard": forbidden_reference_guard,
        "upstream_evaluate_py": upstream_eval,
    }


def _current_python_environment_versions() -> dict:
    """Return exact interpreter/package versions for this process."""

    packages: dict[str, str | None] = {}
    for name in AUTH_EVAL_ENV_PACKAGES:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": packages,
    }


def _external_python_environment_versions(python_executable: Path) -> dict:
    """Query exact interpreter/package versions for another Python executable."""

    script = (
        "import importlib.metadata as m, json, platform, sys\n"
        f"packages = {list(AUTH_EVAL_ENV_PACKAGES)!r}\n"
        "out = {'python_executable': sys.executable, "
        "'python_version': platform.python_version(), 'packages': {}}\n"
        "for name in packages:\n"
        "    try:\n"
        "        out['packages'][name] = m.version(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        out['packages'][name] = None\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )
    try:
        raw = subprocess.check_output(
            [str(python_executable), "-c", script],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
        payload = json.loads(raw)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {
            "python_executable": str(python_executable),
            "query_error": repr(exc),
            "packages": dict.fromkeys(AUTH_EVAL_ENV_PACKAGES),
        }
    if not isinstance(payload, dict):
        return {
            "python_executable": str(python_executable),
            "query_error": "non_object_json",
            "packages": dict.fromkeys(AUTH_EVAL_ENV_PACKAGES),
        }
    return payload


def _resolve_exported_parity_requirements(
    python_executable: Path,
    requirements_text: str,
    *,
    requires_python: str | None = None,
    marker_environment_overrides: dict[str, str] | None = None,
) -> dict:
    """Resolve lock-export rows in the evaluation interpreter's marker env.

    ``uv export`` emits a universal requirements set. In particular, the CUDA
    groups currently emit two ``torchvision`` rows separated by environment
    markers. Marker evaluation therefore belongs to the interpreter that will
    actually run ``upstream/evaluate.py``; evaluating in this wrapper process
    would manufacture parity on a cross-platform dispatch.

    ``requires_python`` is the lock's own interpreter constraint. It is checked
    HERE, in the interpreter being judged, because the lock pins no exact
    Python version -- ``uv export`` emits none and ``uv.lock`` carries only a
    RANGE (currently ``>=3.11, <4``). An equality comparison against a
    lock-derived reference would therefore have to source both sides from this
    same interpreter and could never fail; range SATISFACTION is the strongest
    honest check the lock can license, and it does fail on 3.10 or 4.x.

    The subprocess returns a typed result rather than raising so the authority
    gate can preserve a precise fail-closed reason in provenance. The optional
    overrides exist only to execute both architecture branches in tests.
    """

    script = r"""
import json
import sys

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name

payload = json.load(sys.stdin)
targets = tuple(payload["targets"])
target_by_canonical = {canonicalize_name(name): name for name in targets}
environment = default_environment()
environment.update(payload.get("marker_environment_overrides") or {})
surviving = {name: [] for name in targets}

requires_python = payload.get("requires_python")
requires_python_satisfied = None
if requires_python:
    try:
        specifier_set = SpecifierSet(requires_python)
    except InvalidSpecifier as exc:
        print(json.dumps({
            "ok": False,
            "failure_reason": "requires_python_invalid",
            "requires_python": requires_python,
            "error": repr(exc),
            "marker_environment": environment,
        }, sort_keys=True))
        raise SystemExit(0)
    # prereleases=True so a release-candidate interpreter is judged on its
    # version, not silently excluded by the specifier's default filtering.
    requires_python_satisfied = specifier_set.contains(
        environment["python_full_version"], prereleases=True
    )
    if not requires_python_satisfied:
        print(json.dumps({
            "ok": False,
            "failure_reason": "evaluation_python_outside_lock_requires_python",
            "requires_python": requires_python,
            "python_full_version": environment["python_full_version"],
            "marker_environment": environment,
        }, sort_keys=True))
        raise SystemExit(0)

for line_number, raw_line in enumerate(payload["requirements_text"].splitlines(), 1):
    line = raw_line.strip()
    if not line or line.startswith("#") or line.startswith("--"):
        continue
    try:
        requirement = Requirement(line)
    except InvalidRequirement as exc:
        print(json.dumps({
            "ok": False,
            "failure_reason": "requirement_parse_failed",
            "line_number": line_number,
            "row": line,
            "error": repr(exc),
            "marker_environment": environment,
        }, sort_keys=True))
        raise SystemExit(0)
    target = target_by_canonical.get(canonicalize_name(requirement.name))
    if target is None:
        continue
    if requirement.marker is None or requirement.marker.evaluate(environment=environment):
        surviving[target].append({
            "line_number": line_number,
            "row": line,
            "specifier": str(requirement.specifier),
        })

resolved = {}
selected_rows = {}
for target in targets:
    rows = surviving[target]
    if not rows:
        print(json.dumps({
            "ok": False,
            "failure_reason": "parity_package_missing",
            "package": target,
            "surviving_rows": rows,
            "marker_environment": environment,
        }, sort_keys=True))
        raise SystemExit(0)
    if len(rows) != 1:
        print(json.dumps({
            "ok": False,
            "failure_reason": "marker_evaluation_ambiguous",
            "package": target,
            "surviving_rows": rows,
            "marker_environment": environment,
        }, sort_keys=True))
        raise SystemExit(0)
    requirement = Requirement(rows[0]["row"])
    specifiers = list(requirement.specifier)
    if len(specifiers) != 1 or specifiers[0].operator != "==":
        print(json.dumps({
            "ok": False,
            "failure_reason": "parity_requirement_not_exact",
            "package": target,
            "surviving_rows": rows,
            "marker_environment": environment,
        }, sort_keys=True))
        raise SystemExit(0)
    resolved[target] = specifiers[0].version
    selected_rows[target] = rows[0]

print(json.dumps({
    "ok": True,
    "packages": resolved,
    "selected_rows": selected_rows,
    "marker_environment": environment,
}, sort_keys=True))
"""
    payload = {
        "targets": list(AUTH_EVAL_ENV_PACKAGES),
        "requirements_text": requirements_text,
        "requires_python": requires_python,
        "marker_environment_overrides": marker_environment_overrides or {},
    }
    try:
        proc = subprocess.run(
            [str(python_executable), "-c", script],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "ok": False,
            "failure_reason": "marker_evaluation_exec_failed",
            "error": repr(exc),
        }
    if proc.returncode != 0:
        return {
            "ok": False,
            "failure_reason": "marker_evaluation_nonzero",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "failure_reason": "marker_evaluation_invalid_json",
            "error": repr(exc),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    if not isinstance(result, dict):
        return {
            "ok": False,
            "failure_reason": "marker_evaluation_non_object_json",
            "stdout_tail": proc.stdout[-2000:],
        }
    return result


def _derive_upstream_lock_environment_reference(
    upstream_dir: Path,
    evaluation_python: Path,
    uv_group: str | None,
) -> dict:
    """Derive parity versions from the immutable upstream lock, or refuse.

    This is used only when the historical ``upstream/.venv`` reference does
    not exist. The operator may declare an axis-specific dependency group;
    whether the selected evaluation interpreter matches that group is still
    measured package by package.
    """

    lock_path = upstream_dir / "uv.lock"
    group = str(uv_group or "").strip()
    reference: dict = {
        "schema": "contest_auth_eval_uv_lock_reference_v1",
        "reference_source": "uv_export_frozen_declared_group",
        # The lock pins package versions EXACTLY but the interpreter only by a
        # RANGE (``requires-python``), so the two channels carry different
        # strengths of guarantee and must be labelled as such. A reader who
        # mistakes the interpreter channel for an identity pin would over-trust
        # this reference.
        "python_version_reference_kind": "requires_python_range",
        "reference_identity": (
            f"{lock_path.resolve()}#group={group}" if group else str(lock_path.resolve())
        ),
        "python_executable": str(evaluation_python),
        "uv_group": group or None,
        "uv_lock_path": str(lock_path.resolve()),
        "packages": dict.fromkeys(AUTH_EVAL_ENV_PACKAGES),
    }

    def refuse(reason: str, **details: object) -> dict:
        reference["query_error"] = reason
        reference["failure_reason"] = reason
        reference.update(details)
        return reference

    if not group:
        return refuse("uv_group_not_declared")
    if not lock_path.is_file():
        return refuse("upstream_uv_lock_missing")
    try:
        lock_before = _sha256(lock_path, prefix=0)
    except OSError as exc:
        return refuse("upstream_uv_lock_unreadable", error=repr(exc))
    reference["uv_lock_sha256_before"] = lock_before

    # ``uv export`` emits no interpreter constraint at all, so the lock's own
    # top-level ``requires-python`` is the only interpreter statement the pinned
    # snapshot makes. Read it here rather than comparing versions later: a
    # lock-derived reference can only source a version from the evaluation
    # interpreter itself, so an equality check would compare that interpreter to
    # itself and could never fail. Satisfaction of this range CAN fail.
    try:
        lock_document = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        return refuse("upstream_uv_lock_unparseable", error=repr(exc))
    requires_python = lock_document.get("requires-python")
    if not isinstance(requires_python, str) or not requires_python.strip():
        return refuse("upstream_uv_lock_missing_requires_python")
    requires_python = requires_python.strip()
    reference["requires_python"] = requires_python

    uv_executable = shutil.which("uv")
    if uv_executable is None:
        return refuse("uv_not_found")
    reference["uv_executable"] = uv_executable
    export_argv = [
        uv_executable,
        "export",
        "--frozen",
        "--no-emit-project",
        "--no-hashes",
        "--format",
        "requirements-txt",
        "--directory",
        str(upstream_dir),
        "--group",
        group,
    ]
    reference["uv_export_argv"] = export_argv

    try:
        version_proc = subprocess.run(
            [uv_executable, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        reference["uv_version"] = (
            version_proc.stdout.strip() if version_proc.returncode == 0 else None
        )
        if version_proc.returncode != 0:
            reference["uv_version_query_error"] = {
                "returncode": version_proc.returncode,
                "stderr_tail": version_proc.stderr[-2000:],
            }
    except (OSError, subprocess.SubprocessError) as exc:
        reference["uv_version"] = None
        reference["uv_version_query_error"] = repr(exc)

    export_failure: tuple[str, dict[str, object]] | None = None
    export_text = ""
    try:
        export_proc = subprocess.run(
            export_argv,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        reference["uv_export_returncode"] = export_proc.returncode
        if export_proc.returncode != 0:
            export_failure = (
                "uv_export_nonzero",
                {
                    "uv_export_returncode": export_proc.returncode,
                    "uv_export_stdout_tail": export_proc.stdout[-2000:],
                    "uv_export_stderr_tail": export_proc.stderr[-2000:],
                },
            )
        else:
            export_text = export_proc.stdout
            reference["uv_export_sha256"] = hashlib.sha256(
                export_text.encode("utf-8")
            ).hexdigest()
    except subprocess.TimeoutExpired as exc:
        export_failure = (
            "uv_export_timeout",
            {"timeout_seconds": 60, "error": repr(exc)},
        )
    except OSError as exc:
        export_failure = ("uv_export_exec_failed", {"error": repr(exc)})

    marker_result: dict | None = None
    if export_failure is None:
        marker_result = _resolve_exported_parity_requirements(
            evaluation_python,
            export_text,
            requires_python=requires_python,
        )
        reference["marker_resolution"] = marker_result

    try:
        lock_after = _sha256(lock_path, prefix=0)
    except OSError as exc:
        return refuse("upstream_uv_lock_unreadable_after_export", error=repr(exc))
    reference["uv_lock_sha256_after"] = lock_after
    if lock_before != lock_after:
        return refuse(
            "upstream_uv_lock_mutated",
            uv_lock_sha256_before=lock_before,
            uv_lock_sha256_after=lock_after,
        )
    if export_failure is not None:
        reason, details = export_failure
        return refuse(reason, **details)
    assert marker_result is not None
    if not marker_result.get("ok"):
        return refuse(
            str(marker_result.get("failure_reason") or "marker_evaluation_failed"),
            marker_resolution=marker_result,
        )

    marker_environment = marker_result.get("marker_environment")
    if not isinstance(marker_environment, dict) or not marker_environment.get(
        "python_full_version"
    ):
        return refuse(
            "marker_environment_missing_python_full_version",
            marker_resolution=marker_result,
        )
    packages = marker_result.get("packages")
    if not isinstance(packages, dict):
        return refuse(
            "marker_resolution_packages_not_object",
            marker_resolution=marker_result,
        )

    reference["python_version"] = marker_environment["python_full_version"]
    reference["packages"] = packages
    reference["selected_requirement_rows"] = marker_result.get("selected_rows", {})
    reference["marker_environment"] = marker_environment
    return reference


def _resolve_evaluate_python(args: argparse.Namespace, upstream_dir: Path) -> tuple[Path, str]:
    """Select the interpreter that will run upstream/evaluate.py.

    Make the path absolute, but deliberately do NOT resolve symlinks. Python
    venv launchers are normally symlinks; invoking their base-interpreter
    target bypasses the adjacent ``pyvenv.cfg`` and therefore the locked
    site-packages the authority gate is meant to measure.
    """

    raw = getattr(args, "upstream_python", None)
    if raw is not None:
        path = Path(os.path.abspath(Path(raw)))
        if not path.exists():
            raise SystemExit(f"--upstream-python does not exist: {path}")
        return path, "cli_upstream_python"
    return Path(os.path.abspath(sys.executable)), "current_process_python"


def _build_auth_eval_environment_report(
    args: argparse.Namespace,
    upstream_dir: Path,
) -> dict:
    """Record eval-package versions and flag unproven root-venv parity."""

    eval_python, eval_python_source = _resolve_evaluate_python(args, upstream_dir)
    current = _current_python_environment_versions()
    evaluation = (
        dict(current)
        if eval_python == Path(os.path.abspath(sys.executable))
        else _external_python_environment_versions(eval_python)
    )
    evaluation["python_executable"] = str(eval_python)

    report: dict = {
        "schema": "contest_auth_eval_python_environment_v1",
        "packages_recorded": list(AUTH_EVAL_ENV_PACKAGES),
        "evaluation_python_source": eval_python_source,
        "evaluation": evaluation,
        "wrapper": current,
    }

    upstream_ref = Path(
        os.path.abspath(upstream_dir / ".venv" / "bin" / "python")
    )
    if upstream_ref.exists():
        if eval_python == upstream_ref:
            reference = dict(evaluation)
        else:
            reference = _external_python_environment_versions(upstream_ref)
            reference["python_executable"] = str(upstream_ref)
        report["upstream_reference"] = reference
    else:
        reference = _derive_upstream_lock_environment_reference(
            upstream_dir,
            eval_python,
            getattr(args, "upstream_uv_group", None),
        )
        report["upstream_reference"] = reference

    # PARITY IS ALWAYS EVALUATED (2026-08-10). This block previously ran only when
    # ``--upstream-python`` was ABSENT, so passing the flag disabled the check
    # entirely -- an operator DECLARATION of parity stood in for a MEASUREMENT of it,
    # which is precisely the assertion-instead-of-proof failure the gate exists to
    # prevent. Now it always runs. Nothing is made stricter for the historical honest
    # case: when the declared interpreter IS ``upstream/.venv/bin/python`` the identity
    # branch above sets ``reference = dict(evaluation)`` and the per-package comparison
    # is skipped, so parity passes BY IDENTITY -- proven, not asserted. When that path
    # is absent, the reference instead comes from a frozen uv export of the declared
    # lock group and is compared package by package. The interpreter declaration itself
    # survives for readers as ``evaluation_python_source``.
    mismatches: dict[str, dict[str, object]] = {}
    reference_identity = str(
        reference.get("reference_identity") or upstream_ref
    )
    if reference.get("query_error"):
        mismatches["upstream_reference_python"] = {
            "evaluation": str(eval_python),
            "reference": reference_identity,
            "reason": reference.get("query_error"),
        }
    elif not (upstream_ref.exists() and eval_python == upstream_ref):
        eval_packages = (
            evaluation.get("packages")
            if isinstance(evaluation.get("packages"), dict)
            else {}
        )
        ref_packages = (
            reference.get("packages")
            if isinstance(reference.get("packages"), dict)
            else {}
        )
        # The interpreter channel is checked DIFFERENTLY per reference kind, and
        # conflating them would fake a proof. Against a real ``upstream/.venv``
        # the reference is a DIFFERENT interpreter, so equality is a genuine
        # measurement. Against the lock it is not: the lock states only a RANGE,
        # so the reference's version can only come from the evaluation
        # interpreter itself and equality would be ``x != x`` -- a check that
        # cannot fire. Range satisfaction is enforced instead, inside the
        # derivation, where a failure refuses via ``query_error`` above.
        if (
            reference.get("python_version_reference_kind") != "requires_python_range"
            and evaluation.get("python_version") != reference.get("python_version")
        ):
            mismatches["python"] = {
                "evaluation": evaluation.get("python_version"),
                "reference": reference.get("python_version"),
            }
        for name in AUTH_EVAL_ENV_PACKAGES:
            if eval_packages.get(name) != ref_packages.get(name):
                mismatches[name] = {
                    "evaluation": eval_packages.get(name),
                    "reference": ref_packages.get(name),
                }
    if mismatches:
        mismatch_reason = str(
            reference.get("failure_reason")
            or "current_python_without_proven_upstream_lock_parity"
        )
        report["env_mismatch"] = {
            "schema": "contest_auth_eval_env_mismatch_v1",
            "reason": mismatch_reason,
            "evaluation_python": str(eval_python),
            "reference_python": reference_identity,
            "mismatches": mismatches,
            "advisory_only": True,
        }
    return report


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
                # GROUP_KILL_OK: every caller passes a LEAF binary — nvidia-smi
                # (:1269,:1270), ffmpeg -version/-encoders (:1288,:1290), git rev-parse
                # (:1297,:1300). None forks a worker tree, so a group kill has no
                # grandchild to reach. Catalog #408 flags this because `cmd` is a
                # PARAMETER the argv[0] resolver cannot bind to a literal; the shape is
                # a false positive, not an un-cured site. Also keeps stderr=STDOUT,
                # which the capture_output-based helper cannot express.
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
        "auth_eval_environment": _build_auth_eval_environment_report(args, upstream_dir),
        "upstream_dir": str(upstream_dir),
        "upstream_snapshot_sha256": _require_upstream_snapshot_sha256(upstream_dir),
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
    prov["auth_eval_python"] = prov["auth_eval_environment"]["evaluation"]["python_executable"]
    prov["package_versions"] = prov["auth_eval_environment"]["evaluation"].get("packages", {})
    if "env_mismatch" in prov["auth_eval_environment"]:
        prov["env_mismatch"] = prov["auth_eval_environment"]["env_mismatch"]
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


def _detect_decode_path(inflate_env_overrides: dict[str, str] | None) -> str | None:
    """Which rung of the decode dispatch ladder ran, when anything reports it.

    The fail-closed dispatch ladder (AVX-512 -> AVX2 -> scalar-C -> NEON -> Python) is SILENT
    by design, so the seconds alone cannot tell a native decode from a Python fallback. This
    reads the channels that actually exist today and returns ``None`` otherwise -- an honest
    "unreported" is worth more than a guessed label, and ``tac.contest_budget`` grades the
    unreported case explicitly rather than treating it as fine.

    No new CLI flag: a flag every caller must remember to pass is an orphan generator.
    """
    for key in ("F26_TOKEN_DECODER", "PACT_DECODE_PATH"):
        if inflate_env_overrides and inflate_env_overrides.get(key):
            return str(inflate_env_overrides[key])
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def _upstream_scorer_batch_shape(upstream_dir: Path) -> dict:
    """Read upstream's dataloader batch shape from source, not from memory.

    et4 law: a forward's VALUES move with (code, weights, threads, batch, device), so a receipt
    that omits the batch shape cannot be compared against another run. The harness deliberately
    does NOT pin these (see ``_run_upstream_evaluate``), so the effective values are upstream's
    argparse defaults -- which means they must be READ, and re-read on every upstream bump.
    """
    shape: dict = {
        "schema": "upstream_scorer_batch_shape_v1",
        "harness_pins_batch_shape": False,
        "harness_pin_note": (
            "contest_auth_eval passes no --batch-size/--num-threads/--prefetch-queue-depth/"
            "--seed, so upstream's own defaults are in force; these are read from source"
        ),
        "source": "upstream/evaluate.py argparse defaults + upstream/frame_utils.py seq_len",
    }
    wanted = {"batch_size", "num_threads", "prefetch_queue_depth", "seed"}
    try:
        tree = ast.parse((upstream_dir / "evaluate.py").read_text())
    except (OSError, SyntaxError) as exc:
        shape["parse_error"] = f"{type(exc).__name__}: {exc}"
        return shape
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        flag = node.args[0]
        if not (isinstance(flag, ast.Constant) and isinstance(flag.value, str)):
            continue
        dest = flag.value.lstrip("-").replace("-", "_")
        if dest not in wanted:
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                shape[dest] = kw.value.value
    try:
        fu = ast.parse((upstream_dir / "frame_utils.py").read_text())
    except (OSError, SyntaxError) as exc:
        shape["seq_len_parse_error"] = f"{type(exc).__name__}: {exc}"
        return shape
    for node in ast.walk(fu):
        if isinstance(node, ast.Assign):
            seq_targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            seq_targets = [node.target]
        else:
            continue
        if isinstance(node.value, ast.Constant):
            for target in seq_targets:
                if isinstance(target, ast.Name) and target.id == "seq_len":
                    shape["seq_len"] = node.value.value
    return shape


def _record_instrument_tuple(
    prov: dict,
    upstream_dir: Path,
    args: argparse.Namespace,
    *,
    decode_path: str | None,
) -> dict:
    """Record the full et4 instrument tuple: code, weights, threads, batch, device, decode path.

    ``ddm_et4`` MEASURED that a scorer forward's values move with the batch shape, so a receipt
    carrying only ``torch_version`` cannot be compared against another run -- and comparing two
    incomparable receipts is how a phantom delta gets published. Prior to this the harness
    recorded ONLY ``torch_version`` (one leg of five).
    """
    weights: list[dict] = []
    for name in ("posenet.safetensors", "segnet.safetensors"):
        path = upstream_dir / "models" / name
        weights.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256(path, prefix=0) if path.is_file() else None,
            }
        )

    threads: dict = {
        "os_cpu_count": os.cpu_count(),
        "env": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    try:
        import torch

        threads["torch_get_num_threads"] = torch.get_num_threads()
        threads["torch_get_num_interop_threads"] = torch.get_num_interop_threads()
    except (ImportError, RuntimeError) as exc:
        threads["torch_thread_probe_error"] = f"{type(exc).__name__}: {exc}"

    tup = {
        "schema": "contest_auth_eval_instrument_tuple_v1",
        "law": (
            "ddm_et4: forward VALUES move with (code, weights, threads, batch, device). Two "
            "receipts are comparable only when this whole tuple matches; a delta across a "
            "mismatched tuple is an instrument artifact, not a finding."
        ),
        "code": {
            "pact_commit": prov.get("pact_commit"),
            "pact_commit_source": prov.get("pact_commit_source"),
            "upstream_commit": prov.get("upstream_commit"),
            "upstream_snapshot_sha256": prov.get("upstream_snapshot_sha256"),
            "inflate_script_sha256": prov.get("inflate_script_sha256"),
            "runtime_files_sha256": (
                (prov.get("inflate_runtime_manifest") or {}).get("runtime_files_sha256")
            ),
            "upstream_evaluate_py": (
                (prov.get("inflate_runtime_manifest") or {}).get("upstream_evaluate_py")
            ),
        },
        "weights": weights,
        "threads": threads,
        "batch": _upstream_scorer_batch_shape(upstream_dir),
        "device": {
            "eval_device_requested": args.device,
            "inflate_device_policy": args.inflate_device,
            "gpu_model": prov.get("gpu_model"),
            "gpu_driver": prov.get("gpu_driver"),
            "gpu_t4_match": prov.get("gpu_t4_match"),
            "cuda_available": prov.get("cuda_available"),
            "cuda_version": prov.get("cuda_version"),
            "torch_version": prov.get("torch_version"),
            "platform_system": prov.get("platform_system"),
            "platform_machine": prov.get("platform_machine"),
        },
        "decode_path": decode_path,
        "decode_path_reported": decode_path is not None,
        "decode_path_source": (
            "F26_TOKEN_DECODER / PACT_DECODE_PATH env or --inflate-env override"
            if decode_path is not None
            else "not reported by this runtime generation"
        ),
    }
    prov["instrument_tuple"] = tup
    return tup


def _record_gt_lineage(prov: dict, upstream_dir: Path, video_names_file: Path,
                       args: argparse.Namespace) -> dict:
    """Record which GROUND-TRUTH decode lineage this run's score was measured against.

    ``ddm_pi2`` MEASURED that one instrument reading two ground truths costs 1.4425x on d_seg
    and +1.4061e-04 additive on d_pose -- it silently reproduces the contest-CPU axis while
    claiming otherwise -- and ``ddm_dg1``'s cure (``src/tac/gt_lineage``) never reached this
    emitter: ``gt_lineage`` had ZERO occurrences in contest_auth_eval.py.

    Here the lineage is DERIVABLE AT SOURCE rather than guessed: ``upstream/evaluate.py:31-42``
    forks ``DefaultDatasetClass`` on ``device.type == "cuda"`` -> ``DaliVideoDataset``, else
    ``AVVideoDataset``. The lineage NAMES come from ``tac.gt_lineage`` -- reused, never invented.
    """
    decoder = "DaliVideoDataset" if args.device == "cuda" else "AVVideoDataset"
    row: dict = {
        "schema": "contest_auth_eval_gt_lineage_v1",
        "runtime_decoder": decoder,
        "eval_device": args.device,
        "vocabulary_source": "tac.gt_lineage.RUNTIME_DECODE_LINEAGE (ddm_gl1/ddm_dg1)",
        "fork_source": "upstream/evaluate.py:31-42 (DefaultDatasetClass on device.type == 'cuda')",
        "authority_lineage": AUTHORITY_LINEAGE,
    }
    try:
        lineage = runtime_decode_lineage(decoder)
    except GtLineageError as exc:
        row["lineage"] = None
        row["lineage_error"] = f"{type(exc).__name__}: {exc}"
    else:
        row["lineage"] = lineage
        row["is_authority_lineage"] = lineage == AUTHORITY_LINEAGE
        row["evidence"] = "PRODUCER_DECLARED_BY_UPSTREAM_DEVICE_FORK"
        # This lineage is the one upstream ITSELF selects for this device, so it is always
        # correct FOR THIS AXIS. `is_authority_lineage: false` is therefore a comparability
        # flag, never a defect flag -- a contest-CPU row is supposed to be PyAV.
        row["lineage_is_axis_native"] = True
        row["lineage_is_axis_native_note"] = (
            "derived from upstream's own device fork, so it is by construction the lineage the "
            "contest runner for this device uses; is_authority_lineage compares it against the "
            "contest-CUDA row's lineage and is a COMPARABILITY flag, not a defect"
        )
        if lineage != AUTHORITY_LINEAGE:
            row["cross_lineage_note"] = (
                f"this run's GT is {lineage}, NOT the {AUTHORITY_LINEAGE} lineage the "
                "contest-CUDA authority row is measured against. Correct for this axis; NOT "
                "comparable across axes -- a delta taken against a DALI_NVDEC receipt is a "
                "CROSS-LINEAGE delta (ddm_pi2: 1.4425x d_seg, +1.4061e-04 additive d_pose)"
            )

    # The GT INPUTS themselves: lineage names the decoder, these name the bytes it decoded.
    uncompressed_dir = upstream_dir / "videos"
    inputs: list[dict] = []
    for name in [n.strip() for n in video_names_file.read_text().splitlines() if n.strip()]:
        src = uncompressed_dir / name
        inputs.append(
            {
                "video_name": name,
                "path": str(src),
                "exists": src.is_file(),
                "bytes": src.stat().st_size if src.is_file() else None,
                "sha256": _sha256(src, prefix=0) if src.is_file() else None,
            }
        )
    row["gt_video_inputs"] = inputs
    prov["gt_lineage"] = row
    return row


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
    allow_output_outside_work_dir: bool = False,
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
        allow_outside_work_dir=allow_output_outside_work_dir,
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
    allow_output_outside_work_dir: bool = False,
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
        allow_outside_work_dir=allow_output_outside_work_dir,
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
    allow_outside_work_dir: bool = False,
) -> Path:
    """Resolve auth-side MLX artifact paths without escaping eval custody."""

    work_root = work_dir.resolve()
    if allow_outside_work_dir:
        target = output_path if output_path.is_absolute() else Path.cwd() / output_path
    else:
        target = output_path if output_path.is_absolute() else work_root / output_path
    resolved = target.resolve(strict=False)
    if allow_outside_work_dir:
        return resolved
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


def _validate_expected_runtime_files(
    prov: dict, expected_runtime_files_sha256: str | None
) -> None:
    """Fail closed when the runtime FILES digest differs from the expectation.

    Unlike the tree hash, this digest is path- and environment-independent
    (relative paths + file sha256s + evaluate.py identity only), so a value
    computed by the local packer is valid on any host that extracted the same
    bytes. This is the custody channel remote dispatch wrappers should use.
    """

    if not expected_runtime_files_sha256:
        return
    manifest = prov.get("inflate_runtime_manifest")
    actual = manifest.get("runtime_files_sha256") if isinstance(manifest, dict) else None
    if actual != expected_runtime_files_sha256:
        raise RuntimeError(
            "inflate runtime files digest mismatch: "
            f"expected={expected_runtime_files_sha256} actual={actual}"
        )


def _existing_contest_auth_eval_reuse_blockers(
    *,
    result_path: Path,
    archive: Path,
    inflate_sh: Path,
    upstream_dir: Path,
    device: str,
    video_names_file: Path,
    expected_runtime_tree_sha256: str | None = None,
    scorer_input_cache_hashes_out: Path | None = None,
) -> tuple[dict | None, list[str]]:
    """Validate whether a durable auth-eval JSON can be reused.

    This is intentionally conservative. Reuse is a queue/DX optimization, not
    authority promotion: the cached payload must still carry its original axis
    and false-authority fields.
    """

    blockers: list[str] = []
    if not result_path.is_file():
        return None, ["contest_auth_eval_json_missing"]
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"contest_auth_eval_json_unreadable:{exc}"]
    if not isinstance(result, dict):
        return None, ["contest_auth_eval_json_not_object"]

    prov = result.get("provenance")
    if not isinstance(prov, dict):
        blockers.append("provenance_missing")
        prov = {}

    expected_archive_sha = _sha256(archive, prefix=0)
    if prov.get("archive_sha256") != expected_archive_sha:
        blockers.append("archive_sha256_mismatch")
    if result.get("archive_size_bytes") != archive.stat().st_size:
        blockers.append("archive_size_bytes_mismatch")

    expected_inflate_sha = _sha256(inflate_sh, prefix=0)
    if prov.get("inflate_script_sha256") != expected_inflate_sha:
        blockers.append("inflate_script_sha256_mismatch")
    if str(prov.get("device") or "") != str(device):
        blockers.append("device_mismatch")
    if Path(str(prov.get("video_names_file") or "")).resolve(strict=False) != video_names_file.resolve(strict=False):
        blockers.append("video_names_file_mismatch")
    manifest = prov.get("inflate_runtime_manifest")
    if not isinstance(manifest, dict):
        blockers.append("inflate_runtime_manifest_missing")
        manifest = {}
    actual_runtime_tree = manifest.get("runtime_tree_sha256")
    actual_runtime_content_tree = manifest.get("runtime_content_tree_sha256")
    live_runtime_manifest = _runtime_dependency_manifest(inflate_sh, upstream_dir)
    live_runtime_tree = live_runtime_manifest.get("runtime_tree_sha256")
    live_runtime_content_tree = live_runtime_manifest.get(
        "runtime_content_tree_sha256"
    )
    if actual_runtime_tree != live_runtime_tree:
        blockers.append("runtime_tree_sha256_mismatch")
    if actual_runtime_content_tree != live_runtime_content_tree:
        blockers.append("runtime_content_tree_sha256_mismatch")
    if (
        expected_runtime_tree_sha256
        and actual_runtime_tree != expected_runtime_tree_sha256
    ):
        blockers.append("expected_runtime_tree_sha256_mismatch")

    if result.get("canonical_score") is None:
        blockers.append("canonical_score_missing")
    if result.get("score_axis") is None:
        blockers.append("score_axis_missing")
    n_samples = result.get("n_samples")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        blockers.append("n_samples_missing_or_invalid")
        n_samples = 0
    expected_contract = _auth_eval_evidence_contract(
        device,
        n_samples,
        prov,
    )
    for key in (
        "score_axis",
        "evidence_semantics",
        "score_claim",
        "score_claim_valid",
        "exact_cuda_eval_complete",
        "cpu_leaderboard_reproduction_eligible",
    ):
        if result.get(key) != expected_contract.get(key):
            blockers.append(f"evidence_contract_{key}_mismatch")
    for key in (
        "promotion_eligible",
        "promotable",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
        "gpu_launched",
        "field_selection_ready_for_exact_eval_dispatch",
    ):
        if result.get(key) is True:
            blockers.append(f"truthy_authority_field:{key}")

    if scorer_input_cache_hashes_out is not None:
        if not scorer_input_cache_hashes_out.is_file():
            blockers.append("scorer_input_cache_hashes_missing")
        else:
            try:
                cache_manifest = json.loads(
                    scorer_input_cache_hashes_out.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                blockers.append(f"scorer_input_cache_hashes_unreadable:{exc}")
                cache_manifest = {}
            if isinstance(cache_manifest, dict):
                if (
                    cache_manifest.get("schema_version")
                    != "mlx_scorer_input_cache_hashes.v1"
                ):
                    blockers.append("scorer_input_cache_hashes_schema_mismatch")
                if cache_manifest.get("archive_sha256") != expected_archive_sha:
                    blockers.append("scorer_input_cache_hashes_archive_sha256_mismatch")
                if cache_manifest.get("score_claim") is True:
                    blockers.append("scorer_input_cache_hashes_truthy_score_claim")
            else:
                blockers.append("scorer_input_cache_hashes_not_object")

    return result, blockers


def _print_reused_contest_auth_eval_result(result: dict, *, result_path: Path) -> None:
    print(f"[contest_auth_eval] reusing valid durable JSON: {result_path}")
    print(f"\nRESULT_JSON: {json.dumps(result)}")
    print("\n=== CONTEST AUTH EVAL (REUSED) ===")
    print(f"  Canonical score: {float(result['canonical_score']):.12f}")
    print(f"  Reported final:  {float(result['final_score']):.4f}")
    print(f"  PoseNet dist:   {float(result['avg_posenet_dist']):.6f}")
    print(f"  SegNet dist:    {float(result['avg_segnet_dist']):.6f}")
    print(f"  Rate (unscaled): {float(result['rate_unscaled']):.6f}")
    print(f"  Archive bytes:  {int(result['archive_size_bytes']):,}")
    print(f"  Durable JSON:   {result_path}")


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
    ".ddj5",                     # DDM J5 grammar-state payload (WS1-line receiver states)
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
    "inflate.sh",               # charged portable receiver entrypoint
    "inflate.py",               # charged portable receiver entrypoint
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
                and not _is_known_archive_runtime_member(lower)
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


def _is_known_archive_runtime_member(lower_member: str) -> bool:
    """Return true for charged Python runtime closure members."""

    if not lower_member.endswith(".py"):
        return False
    parts = lower_member.split("/")
    if "" in parts or ".." in parts or "\\" in lower_member:
        return False
    if lower_member in {
        "src/tac/__init__.py",
        "src/tac/substrates/__init__.py",
    }:
        return True
    return (
        lower_member.startswith("src/tac/substrates/")
        or lower_member.startswith("src/tac/codec/")
        or lower_member.startswith("src/tac/runtime/")
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
                 extra_env: dict[str, str] | None = None,
                 expected_num_frames: int = 1200) -> float:
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
    # Public HNeRV/A1/FEC6-style runtimes commonly invoke `${PYTHON:-python3}`,
    # `${PYTHON_BIN:-python}`, or `${PACT_PYTHON_BIN:-...}`. Use the evaluator's
    # interpreter by default so exact-eval screens run in the repo venv that
    # loaded this tool; callers may still override these env vars for a contest
    # container or public replay environment.
    env.setdefault("PYTHON", sys.executable)
    env.setdefault("PYTHON_BIN", sys.executable)
    env.setdefault("PACT_PYTHON_BIN", sys.executable)
    env.setdefault("UV_PYTHON", sys.executable)
    if extra_env:
        env.update(extra_env)
        print(f"[inflate] diagnostic env override keys: {sorted(extra_env)}")
    try:
        result = run_in_process_group(cmd, timeout=timeout, check=False, env=env)
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
    # (1164w x 874h) x NUM_FRAMES (1200) x 3 channels = 3,662,409,600 B.
    # R4 #1 (CRITICAL): use Path.with_suffix('.raw') NOT .stem so subdir
    # paths like 'subdir/0.mkv' resolve to 'inflated_dir/subdir/0.raw'
    # (matching submissions/robust_current/inflate.sh layout). The .stem
    # version stripped the parent dir and missed nested .raw files.
    test_videos = [n.strip() for n in video_names_file.read_text().splitlines()
                   if n.strip()]
    OUT_W, OUT_H, NUM_FRAMES = 1164, 874, int(expected_num_frames)
    if NUM_FRAMES < 1:
        raise RuntimeError(f"[inflate] expected_num_frames must be >= 1, got {NUM_FRAMES}")
    EXPECTED_RAW_BYTES = OUT_W * OUT_H * NUM_FRAMES * 3  # 3,662,409,600
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
            f"{EXPECTED_RAW_BYTES:,} bytes (1164x874x{NUM_FRAMES}x3). Likely "
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
                           device: str, *, timeout: int = 1800,
                           python_executable: Path | None = None) -> dict:
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
    eval_python = Path(python_executable or sys.executable)
    cmd = [
        str(eval_python), str(upstream_dir / "evaluate.py"),
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
    report_component_rounding_abs_bound = 0.5e-8
    pose_lower = max(0.0, pose - report_component_rounding_abs_bound)
    pose_upper = pose + report_component_rounding_abs_bound
    pose_score_rounding_bound = max(
        abs((10.0 * pose_lower) ** 0.5 - score_pose),
        abs((10.0 * pose_upper) ** 0.5 - score_pose),
    )
    seg_score_rounding_bound = 100.0 * report_component_rounding_abs_bound
    rate_reported_score_rounding_bound = 25.0 * report_component_rounding_abs_bound
    report_8dp_score_bound = pose_score_rounding_bound + seg_score_rounding_bound

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
        "avg_posenet_dist_report_8dp_derived": pose,
        "avg_segnet_dist_report_8dp_derived": seg,
        "rate_unscaled": rate_unscaled,
        "rate_unscaled_reported_rounded": rate_unscaled_reported,
        "rate_unscaled_report_8dp_derived": rate_unscaled_reported,
        "original_uncompressed_size_bytes": original_size_bytes,
        "score_pose_contribution": score_pose,
        "score_seg_contribution": score_seg,
        "score_rate_contribution": score_rate,
        "score_recomputed_from_components": score_recomputed,
        "canonical_score": score_recomputed,
        "canonical_score_source": "report_8dp_components_plus_exact_archive_bytes",
        "legacy_canonical_score_source_alias": "score_recomputed_from_components",
        "reported_final_score_display_rounded": final,
        "reported_final_score_display_2dp": final,
        "report_component_decimal_places": 8,
        "report_component_rounding_abs_bound": report_component_rounding_abs_bound,
        "report_8dp_score_worst_case_abs_error_bound": report_8dp_score_bound,
        "report_8dp_pose_score_worst_case_abs_error_bound": pose_score_rounding_bound,
        "report_8dp_seg_score_worst_case_abs_error_bound": seg_score_rounding_bound,
        "report_8dp_printed_rate_score_worst_case_abs_error_bound": rate_reported_score_rounding_bound,
        "score_rounding_abs_delta": score_rounding_abs_delta,
        "score_reported_rounded_differs_from_canonical": score_rounding_abs_delta > 1e-12,
        "archive_size_bytes": archive_size,
        "n_samples": actual_n,
        "report_path": str(report_path),
    }


def _auth_eval_authority_fields(
    *,
    score_claim: bool,
    score_claim_valid: bool,
    exact_cuda_auth_eval: bool = False,
    contest_cuda_auth_eval: bool = False,
) -> dict:
    """Return explicit score/dispatch authority flags for auth-eval payloads."""

    return {
        "score_claim": score_claim,
        "promotion_eligible": False,
        "score_claim_valid": score_claim_valid,
        "score_claim_eligible": score_claim_valid,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "exact_cuda_auth_eval": exact_cuda_auth_eval,
        "contest_cuda_auth_eval": contest_cuda_auth_eval,
    }


def _auth_eval_evidence_contract(
    device: str,
    n_samples: int,
    provenance: dict,
    *,
    diagnostic_blockers: list[str] | None = None,
) -> dict:
    """Return explicit evidence semantics for the selected eval device."""

    if provenance.get("env_mismatch"):
        return {
            "evidence_grade": "auth-eval env mismatch advisory",
            "lane_tag": "[env-mismatch advisory]",
            "score_axis": f"{device}_env_mismatch_advisory",
            "evidence_semantics": "auth_eval_environment_mismatch_advisory",
            "exact_cuda_eval_complete": False,
            "score_claim": False,
            "promotion_eligible": False,
            "score_claim_valid": False,
            "rank_or_kill_eligible": False,
            **_auth_eval_authority_fields(score_claim=False, score_claim_valid=False),
            "cpu_leaderboard_reproduction_eligible": False,
            "env_mismatch": provenance["env_mismatch"],
            "diagnostic_blockers": ["auth_eval_environment_mismatch"],
            "allowed_uses": [
                "diagnostic_debugging",
                "environment_parity_triage",
            ],
        }

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
            **_auth_eval_authority_fields(score_claim=False, score_claim_valid=False),
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
            **_auth_eval_authority_fields(
                score_claim=True,
                score_claim_valid=True,
                exact_cuda_auth_eval=True,
                contest_cuda_auth_eval=True,
            ),
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
            **_auth_eval_authority_fields(score_claim=True, score_claim_valid=True),
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
            **_auth_eval_authority_fields(score_claim=False, score_claim_valid=False),
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
        **_auth_eval_authority_fields(score_claim=False, score_claim_valid=False),
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
    parser.add_argument(
        "--upstream-python",
        type=Path,
        default=None,
        help=(
            "Optional Python executable for authority replays, e.g. "
            "upstream/.venv/bin/python. Without this, runs under the current "
            "repo venv are labeled advisory if package parity with the "
            "upstream lock is not proven."
        ),
    )
    parser.add_argument(
        "--upstream-uv-group",
        default=None,
        help=(
            "Declared upstream dependency group (for example cpu or cu128). "
            "Required to derive the parity reference from upstream/uv.lock "
            "when upstream/.venv/bin/python is absent; the declaration selects "
            "an axis but never asserts parity."
        ),
    )
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
    parser.add_argument(
        "--reuse-valid-json-out",
        action="store_true",
        help=(
            "If --json-out already exists and matches the requested archive, "
            "inflate runtime, device, video list, and optional scorer hash "
            "artifact, reuse it instead of re-running inflate/evaluate. A "
            "per-output fcntl lock prevents duplicate local queue workers."
        ),
    )
    parser.add_argument("--allow-temp-work-dir", action="store_true",
                        help="Allow temp-dir evidence for diagnostic scratch only; "
                             "never use for score custody.")
    parser.add_argument("--inflate-timeout", type=int, default=1800,
                        help="Inflate.sh timeout in seconds. NOT the contest budget: "
                             "eval.yml:30's timeout-minutes: 30 bounds the WHOLE CI JOB "
                             "(checkout + lfs + uv sync + ffmpeg + unzip + inflate + "
                             "evaluate + upload), so the ceiling on inflate alone is the "
                             "RESIDUAL after CI setup and upstream evaluate -- CUDA "
                             "[822,1302]s / CPU [1044,1332]s (PROJECTION, tac.contest_budget). "
                             "This default deliberately does NOT enforce that; it is a "
                             "development stop-loss. The budget is graded, not enforced: see "
                             "contest_budget_verdict in the emitted receipt.")
    parser.add_argument("--evaluate-timeout", type=int, default=1800,
                        help="upstream/evaluate.py timeout in seconds. Note this default "
                             "plus --inflate-timeout permits 3600s -- TWICE the 30-minute "
                             "job wall. Both are development stop-losses, not budget gates; "
                             "the budget gate is contest_budget_verdict in the receipt.")
    parser.add_argument("--keep-work-dir", action="store_true",
                        help="Don't delete work dir on success (for debugging)")
    parser.add_argument("--expected-runtime-tree-sha256", default=None,
                        help="Fail if the inflate runtime dependency tree hash differs.")
    parser.add_argument("--expected-runtime-files-sha256", default=None,
                        help="Fail if the environment-free runtime FILES digest "
                             "(relative paths + file sha256s + evaluate.py) differs. "
                             "Use this channel for cross-host custody; the tree hash "
                             "is path-dependent and not portable across hosts.")
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
        "--retain-per-pair-distortion-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory in which to retain the PER-PAIR PoseNet/SegNet distortion "
            "vectors that upstream reduces to one scalar and discards. OFF by default because "
            "it is NOT free: upstream never exposes the per-pair array, so retention is a "
            "SECOND scorer pass (~40 s contest-CUDA T4, ~214 s contest-CPU). It runs strictly "
            "AFTER the scored result is in hand and cannot perturb any scored number; the "
            "retained vectors are verified to reduce back to the reported scalars."
        ),
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
    parser.add_argument(
        "--allow-scorer-input-cache-artifact-output-outside-work-dir",
        action="store_true",
        help=(
            "Allow scorer-input cache hash/tensor artifacts to be written outside "
            "work_dir, for example to an explicitly mounted Modal Volume. "
            "Default refuses path escapes."
        ),
    )
    args = parser.parse_args()
    retained_pair_env = os.environ.get("PACT_RETAIN_PER_PAIR_DISTORTION_DIR", "").strip()
    if retained_pair_env:
        retained_pair_path = Path(retained_pair_env).resolve()
        if (
            args.retain_per_pair_distortion_dir is not None
            and args.retain_per_pair_distortion_dir.resolve() != retained_pair_path
        ):
            raise SystemExit(
                "PACT_RETAIN_PER_PAIR_DISTORTION_DIR disagrees with "
                "--retain-per-pair-distortion-dir"
            )
        args.retain_per_pair_distortion_dir = retained_pair_path
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

    reuse_lock_fh = None
    if args.reuse_valid_json_out:
        if args.json_out is None:
            raise SystemExit("--reuse-valid-json-out requires --json-out")
        lock_path = args.json_out.resolve().with_name(args.json_out.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        reuse_lock_fh = lock_path.open("a+")
        fcntl.flock(reuse_lock_fh.fileno(), fcntl.LOCK_EX)
        existing_result, reuse_blockers = _existing_contest_auth_eval_reuse_blockers(
            result_path=args.json_out.resolve(),
            archive=archive,
            inflate_sh=inflate_sh,
            upstream_dir=args.upstream_dir.resolve(strict=False),
            device=args.device,
            video_names_file=video_names_file,
            expected_runtime_tree_sha256=args.expected_runtime_tree_sha256,
            scorer_input_cache_hashes_out=(
                args.scorer_input_cache_hashes_out.resolve()
                if args.scorer_input_cache_hashes_out is not None
                and args.allow_scorer_input_cache_artifact_output_outside_work_dir
                else args.scorer_input_cache_hashes_out
            ),
        )
        if existing_result is not None and not reuse_blockers:
            _print_reused_contest_auth_eval_result(
                existing_result,
                result_path=args.json_out.resolve(),
            )
            fcntl.flock(reuse_lock_fh.fileno(), fcntl.LOCK_UN)
            reuse_lock_fh.close()
            return 0
        print(
            "[contest_auth_eval] cached durable JSON not reusable; running eval. "
            f"blockers={reuse_blockers}"
        )

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
        _validate_expected_runtime_files(prov, args.expected_runtime_files_sha256)

        # Receipt hardening (ddm_wc2 Surface B). These ADD provenance; they never touch a
        # measured number, and every one of them writes a field a later reader would otherwise
        # have to guess at.
        decode_path = _detect_decode_path(inflate_env_overrides)
        _record_instrument_tuple(prov, upstream_dir, args, decode_path=decode_path)
        gt_lineage_row = _record_gt_lineage(prov, upstream_dir, video_names_file, args)
        with open(work_dir / "provenance.json", "w") as f:
            json.dump(prov, f, indent=2)
        print(f"[contest_auth_eval] provenance saved: {work_dir / 'provenance.json'}")
        print(f"[contest_auth_eval] archive sha256: {prov['archive_sha256']}")
        print(
            f"[contest_auth_eval] gt_lineage: {gt_lineage_row.get('lineage')} "
            f"via {gt_lineage_row['runtime_decoder']} "
            f"(authority={gt_lineage_row.get('is_authority_lineage')})"
        )

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
                allow_output_outside_work_dir=(
                    args.allow_scorer_input_cache_artifact_output_outside_work_dir
                ),
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
                allow_output_outside_work_dir=(
                    args.allow_scorer_input_cache_artifact_output_outside_work_dir
                ),
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
            python_executable=Path(prov["auth_eval_python"]),
        )
        result["inflate_elapsed_seconds"] = inflate_elapsed_seconds
        result["contest_auth_eval_elapsed_seconds"] = time.monotonic() - exact_eval_t0

        # ALWAYS KEEP THE PAYLOAD, opt-in half: upstream computes 600 per-pair distortion
        # values and keeps only their mean. Retention runs HERE -- after the scored result
        # exists -- so it cannot move a number, and it is a second pass, so it is off unless
        # asked for. A failure to retain is a diagnostic loss, never an eval failure.
        if args.retain_per_pair_distortion_dir is not None:
            try:
                from tac.pose_per_pair_retention import (
                    compute_per_pair_distortion,
                    retain_per_pair_distortion,
                )

                per_pair_pose, per_pair_seg, pose_vectors = compute_per_pair_distortion(
                    upstream_dir=upstream_dir,
                    submission_dir=work_dir,
                    uncompressed_dir=upstream_dir / "videos",
                    video_names_file=video_names_file,
                    device=args.device,
                )
                retention = retain_per_pair_distortion(
                    args.retain_per_pair_distortion_dir,
                    per_pair_pose=per_pair_pose,
                    per_pair_seg=per_pair_seg,
                    reported_pose=result.get("avg_posenet_dist"),
                    reported_seg=result.get("avg_segnet_dist"),
                    pose_vectors=pose_vectors,
                )
                result["per_pair_distortion_retention"] = retention.to_dict()
                print(
                    f"[per-pair] retained {retention.pairs} pairs -> "
                    f"{args.retain_per_pair_distortion_dir} "
                    f"(verified={retention.verified})"
                )
            except Exception as exc:
                result["per_pair_distortion_retention"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "note": "retention failed; the scored result above is unaffected",
                }
                print(f"[per-pair] retention FAILED (score unaffected): {type(exc).__name__}: {exc}")

        # Save final JSON next to the work dir
        result["provenance"] = prov
        result["auth_eval_environment"] = prov["auth_eval_environment"]
        result["package_versions"] = prov.get("package_versions", {})
        if "env_mismatch" in prov:
            result["env_mismatch"] = prov["env_mismatch"]
        result.update(
            _auth_eval_evidence_contract(
                args.device,
                int(result.get("n_samples") or 0),
                prov,
                diagnostic_blockers=diagnostic_blockers or None,
            )
        )
        # ddm_wc2 item 3: the wall-clock budget verdict. The harness allows inflate 1800 s AND
        # evaluate 1800 s -- 3,600 s, twice the 30-minute CI JOB wall (eval.yml:30) -- and until
        # now nothing summed setup + inflate + evaluate against that wall, so a candidate could
        # pass this gate and still time out in the real CI (measured precedent: lc2/PR130 at
        # 1,958 s -> rc=1 on a FASTER 8-core box than the contest's 4 vCPU). Runs AFTER the score
        # exists, adds a verdict and no measurement, and never raises: a guard on the gating
        # instrument must not be able to take down a valid score.
        budget = budget_verdict_for_receipt(result, decode_path=decode_path)
        result["contest_budget_verdict"] = budget
        result["gt_lineage"] = gt_lineage_row
        result["instrument_tuple"] = prov.get("instrument_tuple")

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
        print(f"  Axis:           {result.get('lane_tag')}")
        print(f"  GT lineage:     {gt_lineage_row.get('lineage')} "
              f"({gt_lineage_row['runtime_decoder']})")
        print(f"  Wall budget:    {budget['verdict']} "
              f"[{budget.get('axis') or 'non-contest axis'}] "
              f"inflate={result.get('inflate_elapsed_seconds')} "
              f"evaluate={result.get('evaluate_elapsed_seconds')}")
        print(f"                  {budget.get('rationale') or budget.get('reason')}")
        print(f"  Result JSON:    {out_json}")
        if args.json_out is not None:
            print(f"  Durable JSON:   {args.json_out.resolve()}")

        return 0
    finally:
        if cleanup:
            print(f"[contest_auth_eval] cleaning up {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)
        if reuse_lock_fh is not None and not reuse_lock_fh.closed:
            fcntl.flock(reuse_lock_fh.fileno(), fcntl.LOCK_UN)
            reuse_lock_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
