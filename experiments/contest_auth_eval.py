#!/usr/bin/env python3
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

# Line-buffer stdout so progress flushes to log files immediately.
import sys as _sys
try:
    _sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    _sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

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
    parts = module_name.split(".")
    if module_name.endswith(".__init__"):
        parts = parts[:-1]
    else:
        parts = parts[:-1]
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
    files: list[dict] = []
    if root.exists():
        for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(root).parts
            if any(part in _RUNTIME_DEPENDENCY_SKIP_DIRS for part in rel_parts):
                continue
            if path.name.startswith("._") or path.name in {".DS_Store", "Thumbs.db"}:
                continue
            if (
                path.resolve() != inflate_sh.resolve()
                and path.suffix.lower() not in _RUNTIME_DEPENDENCY_SUFFIXES
            ):
                continue
            files.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path, prefix=0),
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
        "repo_local_tac_import_manifest": repo_local_tac,
        "upstream_evaluate_py": upstream_eval,
    }
    tree_sha = hashlib.sha256(
        json.dumps(tree_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "contest_auth_eval_runtime_dependency_manifest_v1",
        "runtime_root": str(root),
        "runtime_file_count": len(files),
        "runtime_tree_sha256": tree_sha,
        "files": files,
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
        "inflate_timeout_seconds": int(args.inflate_timeout),
        "evaluate_timeout_seconds": int(args.evaluate_timeout),
        "video_names_file": str(args.video_names_file),
        "sys_argv": sys.argv,
        "env_vars": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
            "PYTHONHASHSEED", "PYTORCH_CUDA_ALLOC_CONF", "LD_LIBRARY_PATH",
            "CONFIG_ENV_PATH", "PYTHON_INFLATE", "LANE_MM_SIGMA",
            "INFLATE_BROTLI_SPEC", "INFLATE_AV_SPEC", "INFLATE_TORCH_SPEC",
            "INFLATE_NUMPY_SPEC", "UV_BIN", "UV_PROJECT_ENVIRONMENT",
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
    prov["pact_commit"] = _shell(["git", "rev-parse", "HEAD"])
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
    members: list[str] = []
    with zipfile.ZipFile(archive, "r") as z:
        _validate_zip_container_integrity(archive, z.infolist())
        for info in z.infolist():
            # zip-slip protection
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"Refusing zip-slip path: {info.filename}")
            z.extract(info, dest)
            members.append(info.filename)
    return members


def _decode_zip_name(raw: bytes, *, utf8: bool) -> str:
    encoding = "utf-8" if utf8 else "cp437"
    return raw.decode(encoding)


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


# 2026-04-28 deep hardening pass 3 dimension 3: Whitelist-based archive
# validation. Catches the bug class where unexpected files in the archive
# (stale TTO frames, debug snapshots, .DS_Store from macOS) silently inflate
# the rate term. Memory: feedback_catastrophic_failures_20260421 (Auto-bundle
# by file existence — compress.sh auto-included any .pt/.bin file sitting
# next to the submission).
_KNOWN_ARCHIVE_SUFFIXES = (
    ".bin", ".bin.br",          # renderer (raw or brotli'd)
    ".mkv", ".mp4",             # mask video (svtav1 / h264 / etc.)
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
            if (
                basename not in _KNOWN_ARCHIVE_BASENAMES
                and not any(lower.endswith(s) for s in _KNOWN_ARCHIVE_SUFFIXES)
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
            f"{_KNOWN_ARCHIVE_BASENAMES}. If a new artifact type was added "
            f"intentionally, append its suffix or exact basename to the "
            f"archive whitelist in experiments/contest_auth_eval.py."
        )


def _run_inflate(inflate_sh: Path, archive_dir: Path, inflated_dir: Path,
                 video_names_file: Path, *, timeout: int = 1800) -> float:
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
    try:
        result = subprocess.run(cmd, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"[inflate] TIMED OUT after {timeout}s. Contest budget is "
            f"30 min on T4. If this is a development run, pass "
            f"--inflate-timeout 7200 (or higher) to bypass."
        )
    elapsed = time.monotonic() - t0
    print(f"[inflate] returncode={result.returncode} elapsed={elapsed:.1f}s")
    if result.returncode != 0:
        raise RuntimeError(f"[inflate] FAILED with returncode={result.returncode}")

    # Council R3 #3 + R4 #1 fix: STRICT per-video byte-count validation.
    # Each .raw is uint8 RGB at upstream/frame_utils.py's camera_size
    # (1164w × 874h) × NUM_FRAMES (1200) × 3 channels = 3,663,237,120 B.
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
            f"{EXPECTED_RAW_BYTES:,} bytes (1164×874×1200×3). Likely "
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
        # Try to recover from stdout if report.txt is missing.
        if "Final score:" in result.stdout:
            print(f"[evaluate] report.txt missing — recovering from stdout")
            parsed = _parse_report(
                result.stdout,
                archive_size=archive_bytes_actual,
                source="stdout",
            )
            parsed["evaluate_elapsed_seconds"] = elapsed
            return parsed
        raise RuntimeError(
            f"[evaluate] no report.txt at {report_path} AND stdout has no "
            f"'Final score:' line. Run produced no usable measurement."
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
                if candidate.exists():
                    text = candidate.read_text()
                else:
                    text = report_path
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
    rate_unscaled = _grab(r"Compression Rate:\s*([0-9.eE+-]+)")
    final = _grab(r"Final score[^=]*=\s*([0-9.eE+-]+)")
    n_samples = re.search(r"results over (\d+) samples", text)

    if pose is None or seg is None or rate_unscaled is None or final is None:
        raise RuntimeError(
            f"[evaluate] could not parse report.txt:\n{text[:1024]}"
        )

    # Council R3 #5 (Medium): reject NaN/inf — float() parses both silently.
    # A divide-by-zero in upstream's distortion sum would slip through as
    # final_score=NaN that "looks like" a number. Refuse loud.
    import math as _math
    for label, val in (("posenet_dist", pose), ("segnet_dist", seg),
                       ("rate_unscaled", rate_unscaled), ("final_score", final)):
        if not _math.isfinite(val):
            raise RuntimeError(
                f"[evaluate] non-finite {label}={val} in report.txt — refuse "
                f"to ship a NaN/inf score. Investigate upstream evaluate run."
            )
    if pose < 0 or seg < 0 or rate_unscaled < 0 or final < 0:
        raise RuntimeError(
            f"[evaluate] negative metric in report (pose={pose}, seg={seg}, "
            f"rate={rate_unscaled}, final={final}) — distortions must be ≥0."
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
                        help="Eval device. WARNING: mps is NOISE on PoseNet "
                             "(23x drift vs CUDA per CLAUDE.md). Use cuda only "
                             "for trustworthy scores.")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Working directory (default: tempfile)")
    parser.add_argument("--inflate-timeout", type=int, default=1800,
                        help="Inflate.sh timeout in seconds. Contest budget "
                             "is 30 min (1800s) on T4. Default matches.")
    parser.add_argument("--evaluate-timeout", type=int, default=1800,
                        help="upstream/evaluate.py timeout in seconds.")
    parser.add_argument("--keep-work-dir", action="store_true",
                        help="Don't delete work dir on success (for debugging)")
    parser.add_argument("--expected-runtime-tree-sha256", default=None,
                        help="Fail if the inflate runtime dependency tree hash differs.")
    args = parser.parse_args()

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
        )
        _record_inflate_runtime_artifacts(prov, work_dir, extracted)

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
        result["work_dir"] = str(work_dir)
        out_json = work_dir / "contest_auth_eval.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

        # Print sentinel line for downstream parsers (matches the format
        # auth_eval_renderer.py uses, so existing log scrapers keep working)
        print(f"\nRESULT_JSON: {json.dumps(result)}")
        print(f"\n=== CONTEST AUTH EVAL ===")
        print(f"  Canonical score: {result['canonical_score']:.12f}")
        print(f"  Reported final:  {result['final_score']:.4f}")
        print(f"  PoseNet dist:   {result['avg_posenet_dist']:.6f}")
        print(f"  SegNet dist:    {result['avg_segnet_dist']:.6f}")
        print(f"  Rate (unscaled): {result['rate_unscaled']:.6f}")
        print(f"  Archive bytes:  {result['archive_size_bytes']:,}")
        print(f"  Result JSON:    {out_json}")

        return 0
    finally:
        if cleanup:
            print(f"[contest_auth_eval] cleaning up {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
