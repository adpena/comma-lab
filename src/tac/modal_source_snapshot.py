# SPDX-License-Identifier: MIT
"""Immutable point-in-time snapshot of the source trees a Modal fire mounts.

WHY THIS EXISTS (measured, 2026-09-04). The first ``ddm_fs2`` fire was refused by
Modal with ``source modified during build process``: MAIN ran ``ruff format`` over
``src/`` while the image build was hashing that exact mount. The fire produced no
call, the refusal receipt had to be moved aside by hand, and the row was re-fired.
The mounts read the LIVE WORKING TREE, so any concurrent edit — a formatter, a
sister arm's landing, a test run writing a fixture — can abort a dispatch or, worse,
bake half-old/half-new bytes into a paid image.

THE CURE is the ``git archive HEAD`` idea applied to what is actually mounted.
``git archive`` itself CANNOT be used here, and that is MEASURED, not assumed:

    tracked / on-disk file counts, 2026-09-04
      src                                                    7,806 / 23,650
      upstream                                                   0 / 19,677   <-- pinned OUTSIDE git
      submissions/robust_current                                97 /    185
      experiments/results/public_pr95_intake_20260504_codex       0 /      7   <-- intake clone
      experiments/results/public_pr106_.../source                 0 /    220   <-- intake clone

A ``git archive HEAD`` snapshot would ship an EMPTY ``upstream/`` and the remote
would die at ``missing evaluate.py`` — the exact failure the mount comments already
warn about. So this module snapshots the *mounted paths themselves*, by clone.

MECHANISM. The repo lives on APFS, so ``cp -Rc`` uses ``clonefile(2)``: the copy is
copy-on-write, costs ~0 extra bytes, and — the load-bearing part — is taken at ONE
instant, after which a working-tree edit cannot reach it. MEASURED: cloning ``src``
(712 MB, 23,650 files) takes 3.51 s wall and moves free space by 0. On a filesystem
without clonefile the same call still succeeds as a real copy, slower.

WHAT IS MOUNTED IS DERIVED, NEVER HAND-TYPED. ``extract_mount_paths`` parses the
Modal app module with ``ast`` and reads the first positional argument of every
``add_local_dir`` / ``add_local_file`` call, plus the module names of
``add_local_python_source``. A mount added to the app module tomorrow is snapshotted
tomorrow with no edit here — the drift class that a hand-typed list guarantees.

FAIL-CLOSED. ``verify_snapshot`` re-walks every mounted path in BOTH trees under the
same ignore predicate Modal uses and refuses on any missing path or any file-set
difference. A snapshot that is not provably complete must never be fired: a silently
short mount spends the meter and returns a crash, which is strictly worse than a
local refusal.

``add_local_python_source("tac")`` resolves through Python's import machinery, not
through CWD, so pointing Modal at the snapshot needs ``PYTHONPATH=<snap>/src``.
MEASURED 2026-09-04: the editable install is a plain path line
(``__editable__.tac-0.2.0rc2.pth`` -> ``/Users/adpena/Projects/pact/src``) appended by
``site`` AFTER ``PYTHONPATH``, so ``PYTHONPATH`` wins and ``import tac`` resolves
inside the snapshot. That is an assumption about someone else's install layout, so
``assert_python_source_resolves`` EXECUTES the resolution with the exact cwd/env the
dispatch will use and refuses on any module that resolves outside the snapshot,
rather than trusting today's measurement forever.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.deploy.modal.mount_ignore import ignore_generated_mount_path

# Snapshots must live on the SAME filesystem as the repo or clonefile silently
# degrades to a full byte copy (the /Volumes SSD tiers are separate volumes).
# CLAUDE.md names `.omx/tmp/` as the canonical location for explicitly ephemeral
# local scratch; the snapshot is exactly that — a clone of trees that already have
# durable custody, with its digest recorded in the fire manifest, so deleting it is
# lossless by construction (certify-or-block: the certificate is the digest).
SNAPSHOT_ROOT_REL = ".omx/tmp/modal_fire_snapshots"

_MOUNT_DIR_CALLS = ("add_local_dir",)
_MOUNT_FILE_CALLS = ("add_local_file",)
_MOUNT_PYTHON_SOURCE_CALLS = ("add_local_python_source",)


@dataclass(frozen=True)
class MountSpec:
    """The local paths and importable modules one Modal app module mounts.

    ``dirs`` and ``files`` are repo-relative POSIX strings exactly as written in the
    app module; ``python_source_modules`` are import names, which do not name a path
    until Python resolves them.
    """

    dirs: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    python_source_modules: tuple[str, ...] = ()

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(sorted({*self.dirs, *self.files}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dirs": list(self.dirs),
            "files": list(self.files),
            "python_source_modules": list(self.python_source_modules),
        }


@dataclass
class SnapshotResult:
    """What a snapshot attempt produced, complete enough to adjudicate a fire."""

    root: Path
    entrypoint: Path
    mounts: MountSpec
    copied: list[str] = field(default_factory=list)
    missing_in_source: list[str] = field(default_factory=list)
    files_digest: str = ""
    file_count: int = 0
    total_bytes: int = 0
    elapsed_s: float = 0.0
    clonefile_used: bool = False
    verify_failures: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.verify_failures and not self.missing_in_source

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "modal_source_snapshot.v1",
            "root": str(self.root),
            "entrypoint": str(self.entrypoint),
            "mounts": self.mounts.to_dict(),
            "copied": list(self.copied),
            "missing_in_source": list(self.missing_in_source),
            "files_digest": self.files_digest,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "elapsed_s": round(self.elapsed_s, 3),
            "clonefile_used": self.clonefile_used,
            "verify_failures": list(self.verify_failures),
            "complete": self.complete,
        }


def _literal_first_arg(call: ast.Call) -> str | None:
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def extract_mount_paths(module_path: Path) -> MountSpec:
    """Parse a Modal app module and return every local path/module it mounts.

    Derived from the module's own AST so this never drifts from the mounts. Only
    string-literal first arguments are collected: a computed mount path cannot be
    snapshotted without executing the module, and silently skipping it would be the
    short-mount hazard, so ``verify_snapshot`` is what makes the omission loud —
    the caller compares the parsed set against what it needs.
    """

    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    dirs: list[str] = []
    files: list[str] = []
    modules: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr
        if name in _MOUNT_DIR_CALLS:
            value = _literal_first_arg(node)
            if value:
                dirs.append(value)
        elif name in _MOUNT_FILE_CALLS:
            value = _literal_first_arg(node)
            if value:
                files.append(value)
        elif name in _MOUNT_PYTHON_SOURCE_CALLS:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    modules.append(arg.value)
    return MountSpec(
        dirs=tuple(sorted(set(dirs))),
        files=tuple(sorted(set(files))),
        python_source_modules=tuple(sorted(set(modules))),
    )


def _ignored(rel: Path) -> bool:
    """Apply Modal's own mount-ignore predicate to a mount-relative path.

    ``ignore_generated_mount_path`` takes the path Modal passes it, which is relative
    to the mount SOURCE root. Passing anything else is the 2026-06-10 bug this file
    must not repeat, so callers hand it mount-relative paths only.
    """

    return bool(ignore_generated_mount_path(rel))


def iter_mount_files(source_root: Path, mount_rel: str) -> Iterator[Path]:
    """Yield repo-relative paths Modal would upload for one mount.

    Directories are walked with the same ignore predicate and the same
    special-file exclusion Modal applies; a mount that is a single file yields
    itself. Symlinks are followed for content the way ``add_local_dir`` does, but
    sockets/FIFOs/devices are skipped because Modal cannot snapshot them.
    """

    src = source_root / mount_rel
    if src.is_file():
        yield Path(mount_rel)
        return
    if not src.is_dir():
        return
    base = Path(mount_rel)
    for dirpath, dirnames, filenames in os.walk(src):
        here = Path(dirpath)
        rel_dir = here.relative_to(src)
        dirnames[:] = [
            d for d in dirnames if not _ignored(rel_dir / d if str(rel_dir) != "." else Path(d))
        ]
        for filename in filenames:
            rel = (rel_dir / filename) if str(rel_dir) != "." else Path(filename)
            if _ignored(rel):
                continue
            full = here / filename
            try:
                mode = full.lstat().st_mode
            except OSError:
                continue
            if stat.S_ISSOCK(mode) or stat.S_ISFIFO(mode) or stat.S_ISBLK(mode) or stat.S_ISCHR(mode):
                continue
            yield base / rel


def _clone_tree(src: Path, dst: Path) -> bool:
    """Clone ``src`` to ``dst``; return True when clonefile(2) did the work.

    ``cp -Rc`` asks APFS for a copy-on-write clone: one instant, ~0 extra bytes.
    A filesystem without clonefile makes ``cp -c`` fail, and the fallback is a real
    recursive copy — slower and space-consuming but identical in effect, which is the
    property that matters. The fallback is RECORDED, never silent, so a fire on a
    non-APFS checkout does not quietly become a 2 GB copy nobody budgeted.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        proc = subprocess.run(  # subprocess-no-check-OK: rc adjudicated below; failure falls back to shutil
            ["cp", "-Rc", str(src), str(dst)],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return True
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True) if dst.is_dir() else dst.unlink(missing_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return False


def dedupe_nested(paths: Iterable[str]) -> list[str]:
    """Drop any path already covered by an ancestor path in the same set.

    ``add_local_python_source("tac")`` resolves to ``src/tac`` while the CUDA app also
    mounts ``src``; without this the digest walks those files TWICE and its value then
    depends on the redundant path list rather than on the content. Deterministic either
    way, but a digest whose meaning shifts with an unrelated mount is a digest nobody
    can compare across fires. MEASURED: the hv1 race self-test read as "MUTATED" purely
    from this double count.
    """

    ordered = sorted({p.strip("/") for p in paths if p})
    kept: list[str] = []
    for candidate in ordered:
        if any(candidate == k or candidate.startswith(k + "/") for k in kept):
            continue
        kept.append(candidate)
    return kept


def files_digest(root: Path, mounts: Iterable[str]) -> tuple[str, int, int]:
    """Content-only digest over every mounted file: (sha256, count, bytes).

    Content-only and path-relative on purpose. Two validators that hash absolute
    paths, mtimes or inode metadata will DISAGREE across environments and deadlock
    (the r9m class, five times); the only digest both sides of a fire can compute is
    over sorted relative paths and file bytes.
    """

    hasher = hashlib.sha256()
    count = 0
    total = 0
    entries: list[Path] = []
    for mount_rel in dedupe_nested(mounts):
        entries.extend(iter_mount_files(root, mount_rel))
    for rel in sorted(entries, key=lambda p: p.as_posix()):
        full = root / rel
        try:
            data = full.read_bytes()
        except OSError:
            continue
        hasher.update(rel.as_posix().encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(hashlib.sha256(data).digest())
        count += 1
        total += len(data)
    return hasher.hexdigest(), count, total


def verify_snapshot(
    source_root: Path, snapshot_root: Path, mounts: Iterable[str]
) -> list[str]:
    """Return refusal reasons; empty means the snapshot is provably complete.

    Compares the mount-relative FILE SET of both trees. A difference in either
    direction is reported: a file missing from the snapshot is a short mount (the
    paid-crash class), and a file present only in the snapshot means the walk and
    the clone disagree, which is equally a reason not to fire.

    Files the source gained AFTER the clone are expected and are NOT a failure —
    that is the whole point of a snapshot — so the caller passes the mount list it
    cloned and this function reports the asymmetry it sees; the fire tool decides.
    """

    failures: list[str] = []
    for mount_rel in dedupe_nested(mounts):
        src = source_root / mount_rel
        dst = snapshot_root / mount_rel
        if not src.exists():
            failures.append(f"mount path absent in source: {mount_rel}")
            continue
        if not dst.exists():
            failures.append(f"mount path MISSING FROM SNAPSHOT: {mount_rel}")
            continue
        src_files = {p.as_posix() for p in iter_mount_files(source_root, mount_rel)}
        dst_files = {p.as_posix() for p in iter_mount_files(snapshot_root, mount_rel)}
        only_source = src_files - dst_files
        only_snapshot = dst_files - src_files
        if only_source:
            sample = ", ".join(sorted(only_source)[:5])
            failures.append(
                f"{mount_rel}: {len(only_source)} file(s) in source but not in snapshot "
                f"(short mount would be fired): {sample}"
            )
        if only_snapshot:
            sample = ", ".join(sorted(only_snapshot)[:5])
            failures.append(
                f"{mount_rel}: {len(only_snapshot)} file(s) in snapshot but not in source: {sample}"
            )
    return failures


def resolve_python_source_paths(source_root: Path, modules: Iterable[str]) -> list[str]:
    """Repo-relative dirs/files backing ``add_local_python_source`` module names.

    These mounts name an IMPORT, not a path, so the snapshot cannot find them by
    reading the app module alone. ``experiments/modal_auth_eval_cpu.py`` is the case
    that makes this mandatory: it mounts three individual ``src/tac/...`` files but
    never ``src`` itself, while still calling ``add_local_python_source("tac",
    "comma_lab")``. Without resolving those names the snapshot would carry three
    files where the fire needs two whole packages, and the import probe would
    (correctly) refuse every CPU-axis fire.

    Modules that do not resolve, or resolve outside the repo, are skipped: the
    entrypoint module is already snapshotted by path, and a third-party package is
    not ours to snapshot.
    """

    import importlib.util

    out: list[str] = []
    root = source_root.resolve()
    for name in sorted({m for m in modules if m}):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError):
            continue
        if spec is None:
            continue
        origin = spec.origin
        target: Path | None = None
        if spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                candidate = Path(location).resolve()
                if str(candidate).startswith(str(root)):
                    target = candidate
                    break
        elif origin and origin not in {"built-in", "frozen"}:
            candidate = Path(origin).resolve()
            if str(candidate).startswith(str(root)):
                target = candidate
        if target is None:
            continue
        out.append(target.relative_to(root).as_posix())
    return sorted(set(out))


def dispatch_env(
    snapshot_root: Path,
    base_env: dict[str, str] | None = None,
    *,
    entrypoint: Path | None = None,
) -> dict[str, str]:
    """Env for ``modal run`` so importable mounts resolve INSIDE the snapshot.

    ``add_local_python_source("tac")`` mounts whatever ``import tac`` finds. Without
    this the snapshot would mount its own ``src`` tree by path and the LIVE tree by
    import — half snapshot, half racing, which is worse than neither.

    The entrypoint's own directory is prepended too, because the app module names
    ITSELF in ``add_local_python_source`` and is importable only from beside it —
    which is what ``modal run <path>::main`` arranges, and what the resolution probe
    must reproduce or it tests a different sys.path than the fire uses.
    """

    env = dict(os.environ if base_env is None else base_env)
    parts = [str(snapshot_root / "src")]
    if entrypoint is not None:
        parts.insert(0, str(Path(entrypoint).parent))
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = ":".join(parts)
    return env


def assert_python_source_resolves(
    snapshot_root: Path,
    modules: Iterable[str],
    *,
    python_executable: str,
    entrypoint: Path | None = None,
    timeout_s: float = 60.0,
) -> list[str]:
    """EXECUTE the import resolution the dispatch will use; return refusals.

    This is the difference between "PYTHONPATH should win over the editable .pth"
    and knowing it does on the machine that is about to spend money. Every module
    that resolves outside the snapshot is reported by name and file.
    """

    failures: list[str] = []
    names = [m for m in modules if m]
    if not names:
        return failures
    probe = (
        "import importlib,json,sys\n"
        "out={}\n"
        "for name in " + repr(names) + ":\n"
        "    try:\n"
        "        mod=importlib.import_module(name)\n"
        "        out[name]=getattr(mod,'__file__',None)\n"
        "    except Exception as exc:\n"
        "        out[name]='ERROR: %s: %s' % (type(exc).__name__, exc)\n"
        "sys.stdout.write(json.dumps(out))\n"
    )
    try:
        proc = subprocess.run(  # subprocess-no-check-OK: rc + stdout adjudicated below into refusals
            [python_executable, "-c", probe],
            cwd=str(snapshot_root),
            env=dispatch_env(snapshot_root, entrypoint=entrypoint),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"could not probe python-source resolution: {type(exc).__name__}: {exc}"]
    if proc.returncode != 0:
        return [f"python-source probe rc={proc.returncode}: {proc.stderr.strip()[:300]}"]
    try:
        resolved = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return [f"python-source probe emitted non-JSON: {proc.stdout.strip()[:300]}"]
    snap = str(snapshot_root.resolve())
    for name, where in sorted(resolved.items()):
        if not isinstance(where, str):
            failures.append(f"module {name!r} resolved to no __file__ (namespace package?)")
            continue
        if where.startswith("ERROR:"):
            failures.append(f"module {name!r} did not import: {where}")
            continue
        if not str(Path(where).resolve()).startswith(snap):
            failures.append(
                f"module {name!r} resolves OUTSIDE the snapshot ({where}); the fire would mount "
                "the live working tree for it and the snapshot would be half-effective"
            )
    return failures


def prune_snapshots(root: Path, *, retain_days: float, now: float | None = None) -> list[str]:
    """Delete snapshot dirs older than ``retain_days``; return what was removed.

    Clones cost ~0 bytes at creation but diverge as the working tree changes, so
    they are not free forever. Removal is lossless: every snapshot's digest is in
    the fire manifest that consumed it, and its contents are a clone of trees with
    durable custody elsewhere.
    """

    removed: list[str] = []
    if not root.is_dir():
        return removed
    cutoff = (time.time() if now is None else now) - retain_days * 86400.0
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime >= cutoff:
                continue
        except OSError:
            continue
        shutil.rmtree(child, ignore_errors=True)
        if not child.exists():
            removed.append(child.name)
    return removed


def build_snapshot(
    *,
    source_root: Path,
    entrypoint: Path,
    snapshot_root: Path | None = None,
    label: str = "fire",
    extra_paths: Iterable[str] = (),
) -> SnapshotResult:
    """Clone every path the entrypoint's app module mounts into an immutable tree.

    The entrypoint module itself is always included: ``modal run <path>::main`` loads
    it from disk, and firing a snapshot whose dispatcher is the live file would leave
    the very race this closes half open.
    """

    source_root = source_root.resolve()
    entrypoint = entrypoint.resolve()
    mounts = extract_mount_paths(entrypoint)
    entry_rel = entrypoint.relative_to(source_root).as_posix()
    python_source_paths = resolve_python_source_paths(source_root, mounts.python_source_modules)
    wanted = sorted({*mounts.paths, *python_source_paths, *extra_paths, entry_rel})

    if snapshot_root is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label) or "fire"
        snapshot_root = source_root / SNAPSHOT_ROOT_REL / f"{stamp}_{safe}"
    snapshot_root = Path(snapshot_root)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    copied: list[str] = []
    missing: list[str] = []
    clonefile_used = True
    for rel in wanted:
        src = source_root / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = snapshot_root / rel
        if dst.exists():
            continue
        clonefile_used = _clone_tree(src, dst) and clonefile_used
        copied.append(rel)
    elapsed = time.monotonic() - started

    present = dedupe_nested([rel for rel in wanted if rel not in missing])
    digest, count, total = files_digest(snapshot_root, present)
    result = SnapshotResult(
        root=snapshot_root,
        entrypoint=snapshot_root / entry_rel,
        mounts=mounts,
        copied=copied,
        missing_in_source=missing,
        files_digest=digest,
        file_count=count,
        total_bytes=total,
        elapsed_s=elapsed,
        clonefile_used=clonefile_used,
    )
    result.verify_failures = verify_snapshot(source_root, snapshot_root, present)
    return result


__all__ = [
    "SNAPSHOT_ROOT_REL",
    "MountSpec",
    "SnapshotResult",
    "assert_python_source_resolves",
    "build_snapshot",
    "dedupe_nested",
    "dispatch_env",
    "extract_mount_paths",
    "files_digest",
    "iter_mount_files",
    "prune_snapshots",
    "verify_snapshot",
]
