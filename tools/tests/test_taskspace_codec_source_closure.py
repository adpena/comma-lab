# SPDX-License-Identifier: MIT
"""Clean-checkout source-closure guard for the task-space codec entrypoints."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = (
    Path("src/tac/witness_dsl/taskspace_selected_preimage_operand_adapter_v1.py"),
    Path("src/tac/witness_control/taskspace_codec_adversarial_gate_v2.py"),
    Path("tools/run_taskspace_lossy_selected_plane_codec_n600.py"),
    Path("tools/run_taskspace_program_residual_n600.py"),
    Path("tools/build_taskspace_layered_public_closure.py"),
)


def _tracked_paths() -> frozenset[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return frozenset(Path(raw.decode()) for raw in result.stdout.split(b"\0") if raw)


def _tac_module_path(module: str) -> Path | None:
    if not module.startswith("tac."):
        return None
    stem = Path("src").joinpath(*module.split("."))
    module_path = stem.with_suffix(".py")
    if (REPO_ROOT / module_path).is_file():
        return module_path
    package_path = stem / "__init__.py"
    if (REPO_ROOT / package_path).is_file():
        return package_path
    return None


def _absolute_tac_imports(path: Path) -> tuple[Path, ...]:
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"), filename=str(path))
    imports: set[Path] = set()
    for node in ast.walk(tree):
        modules: tuple[str, ...] = ()
        if isinstance(node, ast.Import):
            modules = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = (node.module,)
        for module in modules:
            resolved = _tac_module_path(module)
            if resolved is not None:
                imports.add(resolved)
    return tuple(sorted(imports))


def test_taskspace_codec_entrypoints_have_tracked_local_import_closure() -> None:
    """A shared dirty checkout must not conceal a clean-checkout import failure."""

    tracked = _tracked_paths()
    pending = list(ENTRYPOINTS)
    visited: set[Path] = set()
    missing: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        if path not in tracked:
            missing.add(path)
        pending.extend(_absolute_tac_imports(path))

    assert not missing, (
        "task-space codec entrypoints import source files absent from git; "
        "focused tests in the dirty checkout are false source-closure authority: "
        + ", ".join(str(path) for path in sorted(missing))
    )
