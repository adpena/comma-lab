"""Static worker-import closure against the interpreter that will run it.

Modal images can contain more than one Python environment.  A dependency in
the image's default interpreter does not imply that a worker launched through
an upstream locked venv can import it.  This module walks the worker's local
Python import graph, compares its third-party import roots with the target
lock plus explicitly provisioned target-venv dependencies, and fails before a
provider dispatch when the closure is incomplete.

The check is deliberately static and import-time exact. Module-scope imports
inside guards count, while imports inside deferred function bodies do not: a
worker can contain optional packaging paths that the sealed dispatch never
calls. Roots provided by a retained runtime payload (for example ``runtime``
or ``modules``) must be named explicitly; silently inheriting the launcher's
site-packages is never an acceptance path.
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

__all__ = [
    "WorkerDependencyClosureError",
    "require_worker_dependency_closure",
    "scan_worker_dependency_closure",
]


class WorkerDependencyClosureError(RuntimeError):
    """The worker can import a package that the target venv does not supply."""


_STDLIB_ROOTS: Final = frozenset(getattr(sys, "stdlib_module_names", ()))
_SPEC_NAME_RE: Final = re.compile(r"^[A-Za-z0-9_.-]+")
_DISTRIBUTION_IMPORT_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "brotli": ("brotli",),
    "nvidia-dali-cuda120": ("nvidia",),
    "opencv-python": ("cv2",),
    "pillow": ("PIL",),
    "pyyaml": ("yaml",),
    "scikit-learn": ("sklearn",),
    "segmentation-models-pytorch": ("segmentation_models_pytorch",),
}


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _distribution_import_roots(spec: str) -> set[str]:
    match = _SPEC_NAME_RE.match(spec.strip())
    if match is None:
        raise WorkerDependencyClosureError(f"invalid target dependency spec with no distribution name: {spec!r}")
    distribution = match.group(0).lower().replace("_", "-")
    aliases = _DISTRIBUTION_IMPORT_ALIASES.get(distribution)
    if aliases is not None:
        return set(aliases)
    return {distribution.replace("-", "_")}


def _lock_import_roots(lock_path: Path) -> tuple[set[str], list[str]]:
    try:
        payload = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise WorkerDependencyClosureError(f"cannot read target lock {lock_path}: {exc}") from exc
    packages = payload.get("package", [])
    if not isinstance(packages, list):
        raise WorkerDependencyClosureError(f"target lock {lock_path} has no [[package]] inventory")
    names: list[str] = []
    roots: set[str] = set()
    for row in packages:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            continue
        name = row["name"].strip()
        if not name:
            continue
        names.append(name)
        roots.update(_distribution_import_roots(name))
    if not names:
        raise WorkerDependencyClosureError(f"target lock {lock_path} contains zero named packages")
    return roots, sorted(set(names), key=str.lower)


def _module_name_for_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    for source_root in (repo_root / "src", repo_root):
        try:
            rel = resolved.relative_to(source_root.resolve())
        except ValueError:
            continue
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)
    raise WorkerDependencyClosureError(f"worker entrypoint is outside repo/source roots: {resolved}")


def _resolve_local_module(module: str, source_roots: tuple[Path, ...]) -> Path | None:
    if not module:
        return None
    parts = module.split(".")
    for root in source_roots:
        base = root.joinpath(*parts)
        candidates = (base.with_suffix(".py"), base / "__init__.py")
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None


def _absolute_from_module(*, current_module: str, current_path: Path, node: ast.ImportFrom) -> list[str]:
    if node.level == 0:
        return [node.module] if node.module else []
    package = current_module if current_path.name == "__init__.py" else current_module.rpartition(".")[0]
    parts = package.split(".") if package else []
    levels_up = node.level - 1
    if levels_up > len(parts):
        return []
    prefix = ".".join(parts[: len(parts) - levels_up])
    if node.module:
        return [".".join(value for value in (prefix, node.module) if value)]
    return [".".join(value for value in (prefix, alias.name) if value) for alias in node.names if alias.name != "*"]


def _module_scope_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Return imports executed while the module itself is imported."""

    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    imports: list[ast.Import | ast.ImportFrom] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        ancestor = parents.get(node)
        while ancestor is not None and not isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            ancestor = parents.get(ancestor)
        if ancestor is None:
            imports.append(node)
    return imports


def scan_worker_dependency_closure(
    *,
    repo_root: Path,
    worker_entrypoints: Iterable[Path],
    target_lock_path: Path,
    extra_target_dependencies: Iterable[str] = (),
    target_available_import_roots: Iterable[str] = (),
    payload_provided_import_roots: Iterable[str] = (),
) -> dict[str, Any]:
    """Return a deterministic worker/target dependency-closure receipt.

    ``extra_target_dependencies`` must be the same iterable consumed by the
    target-venv provisioning command.  ``payload_provided_import_roots`` is
    reserved for importable code shipped in the request/runtime payload, not
    packages incidentally available in the launcher's default interpreter.
    """

    root = repo_root.resolve()
    source_roots = (root, root / "src")
    lock_path = target_lock_path.resolve()
    lock_roots, lock_packages = _lock_import_roots(lock_path)
    extra_specs = tuple(sorted({str(value).strip() for value in extra_target_dependencies if str(value).strip()}))
    available_roots = set(lock_roots)
    for spec in extra_specs:
        available_roots.update(_distribution_import_roots(spec))
    available_roots.update(str(value).strip() for value in target_available_import_roots if str(value).strip())

    payload_roots = {str(value).strip() for value in payload_provided_import_roots if str(value).strip()}
    local_namespace_roots = {
        child.name
        for source_root in source_roots
        if source_root.is_dir()
        for child in source_root.iterdir()
        if child.is_dir() or child.suffix == ".py"
    }
    entry_rows: list[tuple[Path, str]] = []
    for raw_path in worker_entrypoints:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise WorkerDependencyClosureError(f"worker entrypoint is missing: {path}")
        entry_rows.append((path, _module_name_for_path(path, root)))
    if not entry_rows:
        raise WorkerDependencyClosureError("worker dependency closure has zero entrypoints")

    visited: dict[Path, str] = {}
    third_party_roots: set[str] = set()
    payload_imports: set[str] = set()
    unresolved_local_imports: set[str] = set()

    def visit(path: Path, module_name: str) -> None:
        if path in visited:
            return
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise WorkerDependencyClosureError(f"cannot parse worker dependency source {path}: {exc}") from exc
        visited[path] = module_name
        imports: list[str] = []
        for node in _module_scope_imports(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                bases = _absolute_from_module(
                    current_module=module_name,
                    current_path=path,
                    node=node,
                )
                imports.extend(bases)
                if node.module:
                    for base in bases:
                        for alias in node.names:
                            candidate = f"{base}.{alias.name}"
                            if _resolve_local_module(candidate, source_roots) is not None:
                                imports.append(candidate)
        for imported in imports:
            if not imported or imported == "__future__":
                continue
            imported_root = imported.split(".", 1)[0]
            local_path = _resolve_local_module(imported, source_roots)
            if local_path is not None:
                visit(local_path, imported)
                continue
            if imported_root in _STDLIB_ROOTS:
                continue
            if imported_root in payload_roots:
                payload_imports.add(imported)
                continue
            if any(source_root.joinpath(*imported.split(".")).is_dir() for source_root in source_roots):
                continue
            if imported_root in local_namespace_roots:
                unresolved_local_imports.add(imported)
                continue
            third_party_roots.add(imported_root)

    for entry_path, entry_module in entry_rows:
        visit(entry_path, entry_module)

    missing = sorted(third_party_roots - available_roots)
    passed = not missing and not unresolved_local_imports
    sources = [
        {
            "path": str(path.relative_to(root)),
            "module": visited[path],
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(visited, key=lambda value: str(value.relative_to(root)))
    ]
    return {
        "schema": "tac.worker_target_venv_dependency_closure.v1",
        "worker_entrypoints": [str(path.relative_to(root)) for path, _ in entry_rows],
        "source_file_count": len(sources),
        "sources": sources,
        "required_third_party_import_roots": sorted(third_party_roots),
        "payload_provided_import_roots": sorted(payload_roots),
        "payload_imports_observed": sorted(payload_imports),
        "unresolved_local_imports": sorted(unresolved_local_imports),
        "target_lock": {
            "path": str(lock_path.relative_to(root)) if lock_path.is_relative_to(root) else str(lock_path),
            "bytes": lock_path.stat().st_size,
            "sha256": _sha256_file(lock_path),
            "package_count": len(lock_packages),
            "packages": lock_packages,
        },
        "extra_target_dependencies": list(extra_specs),
        "available_import_roots": sorted(available_roots),
        "missing_import_roots": missing,
        "passed": passed,
        "selection_mode": (
            "recursive static AST import-time closure; module-scope guarded imports included; "
            "deferred function-body imports excluded"
        ),
    }


def require_worker_dependency_closure(**kwargs: Any) -> dict[str, Any]:
    """Return the receipt or refuse an incomplete target-venv closure."""

    receipt = scan_worker_dependency_closure(**kwargs)
    missing = receipt["missing_import_roots"]
    unresolved = receipt["unresolved_local_imports"]
    if missing or unresolved:
        raise WorkerDependencyClosureError(
            "worker dependency closure is incomplete for the target interpreter; "
            f"missing import roots={missing!r}; unresolved local imports={unresolved!r}. "
            "Add pinned dependencies to the "
            "same target-venv provisioning iterable consumed by this seal, or "
            "declare a request/runtime-payload root explicitly."
        )
    return receipt
